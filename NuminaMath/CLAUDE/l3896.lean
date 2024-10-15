import Mathlib

namespace NUMINAMATH_CALUDE_complex_equation_solution_l3896_389620

theorem complex_equation_solution :
  ∀ z : ℂ, z = Complex.I * (2 - z) → z = 1 + Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3896_389620


namespace NUMINAMATH_CALUDE_bezdikovPopulationTheorem_l3896_389689

/-- Represents the population of Bezdíkov -/
structure BezdikovPopulation where
  women1966 : ℕ
  men1966 : ℕ
  womenNow : ℕ
  menNow : ℕ

/-- Conditions for the Bezdíkov population problem -/
def bezdikovConditions (p : BezdikovPopulation) : Prop :=
  p.women1966 = p.men1966 + 30 ∧
  p.womenNow = p.women1966 / 4 ∧
  p.menNow = p.men1966 - 196 ∧
  p.womenNow = p.menNow + 10

/-- The theorem stating that the current total population of Bezdíkov is 134 -/
theorem bezdikovPopulationTheorem (p : BezdikovPopulation) 
  (h : bezdikovConditions p) : p.womenNow + p.menNow = 134 :=
by
  sorry

#check bezdikovPopulationTheorem

end NUMINAMATH_CALUDE_bezdikovPopulationTheorem_l3896_389689


namespace NUMINAMATH_CALUDE_medicine_tablets_l3896_389666

theorem medicine_tablets (tablets_A tablets_B min_extraction : ℕ) : 
  tablets_A = 10 →
  min_extraction = 15 →
  min_extraction = tablets_A + 2 + (tablets_B - 2) →
  tablets_B = 5 := by
sorry

end NUMINAMATH_CALUDE_medicine_tablets_l3896_389666


namespace NUMINAMATH_CALUDE_prob_two_white_is_three_tenths_l3896_389663

-- Define the total number of balls
def total_balls : ℕ := 5

-- Define the number of white balls
def white_balls : ℕ := 3

-- Define the number of black balls
def black_balls : ℕ := 2

-- Define the number of ways to choose 2 balls from the total
def total_choices : ℕ := Nat.choose total_balls 2

-- Define the number of ways to choose 2 white balls
def white_choices : ℕ := Nat.choose white_balls 2

-- Define the probability of drawing two white balls given that they are the same color
def prob_two_white : ℚ := white_choices / total_choices

-- Theorem to prove
theorem prob_two_white_is_three_tenths : 
  prob_two_white = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_prob_two_white_is_three_tenths_l3896_389663


namespace NUMINAMATH_CALUDE_jeans_average_speed_l3896_389628

/-- Proves that Jean's average speed is 18/11 mph given the problem conditions --/
theorem jeans_average_speed :
  let trail_length : ℝ := 12
  let uphill_length : ℝ := 4
  let chantal_flat_speed : ℝ := 3
  let chantal_uphill_speed : ℝ := 1.5
  let chantal_downhill_speed : ℝ := 2.25
  let jean_delay : ℝ := 2

  let chantal_flat_time : ℝ := (trail_length - uphill_length) / chantal_flat_speed
  let chantal_uphill_time : ℝ := uphill_length / chantal_uphill_speed
  let chantal_downhill_time : ℝ := uphill_length / chantal_downhill_speed
  let chantal_total_time : ℝ := chantal_flat_time + chantal_uphill_time + chantal_downhill_time

  let jean_travel_time : ℝ := chantal_total_time - jean_delay
  let jean_travel_distance : ℝ := uphill_length

  jean_travel_distance / jean_travel_time = 18 / 11 := by sorry

end NUMINAMATH_CALUDE_jeans_average_speed_l3896_389628


namespace NUMINAMATH_CALUDE_power_difference_equals_seven_l3896_389600

theorem power_difference_equals_seven : 2^5 - 5^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equals_seven_l3896_389600


namespace NUMINAMATH_CALUDE_real_solutions_range_l3896_389655

theorem real_solutions_range (m : ℝ) : 
  (∃ x : ℝ, (m - 2) * x^2 - 2 * x + 1 = 0) → m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_real_solutions_range_l3896_389655


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3896_389696

theorem simplify_trig_expression (α β : ℝ) :
  1 - Real.sin α ^ 2 - Real.sin β ^ 2 + 2 * Real.sin α * Real.sin β * Real.cos (α - β) = Real.cos (α - β) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3896_389696


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l3896_389614

def A (m : ℝ) : Set ℝ := {1, 3, 2*m+3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_values (m : ℝ) : B m ⊆ A m → m = 1 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l3896_389614


namespace NUMINAMATH_CALUDE_fence_painting_fraction_l3896_389626

theorem fence_painting_fraction (total_time : ℝ) (part_time : ℝ) 
  (h1 : total_time = 60) 
  (h2 : part_time = 12) : 
  (part_time / total_time) = (1 : ℝ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_fence_painting_fraction_l3896_389626


namespace NUMINAMATH_CALUDE_successive_discounts_equivalent_to_single_discount_l3896_389638

-- Define the successive discounts
def discount1 : ℝ := 0.10
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.25

-- Define the equivalent single discount
def equivalent_discount : ℝ := 0.426

-- Theorem statement
theorem successive_discounts_equivalent_to_single_discount :
  (1 - discount1) * (1 - discount2) * (1 - discount3) = 1 - equivalent_discount :=
by sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalent_to_single_discount_l3896_389638


namespace NUMINAMATH_CALUDE_product_in_fourth_quadrant_l3896_389634

/-- Given two complex numbers Z₁ and Z₂, prove that their product Z is in the fourth quadrant -/
theorem product_in_fourth_quadrant (Z₁ Z₂ : ℂ) (h₁ : Z₁ = 3 + I) (h₂ : Z₂ = 1 - I) :
  let Z := Z₁ * Z₂
  (Z.re > 0 ∧ Z.im < 0) := by sorry

end NUMINAMATH_CALUDE_product_in_fourth_quadrant_l3896_389634


namespace NUMINAMATH_CALUDE_train_journey_encryption_train_journey_l3896_389621

/-- Represents a city name as a list of alphabet positions --/
def CityCode := List Nat

/-- Defines the alphabet positions for letters --/
def alphabetPosition (c : Char) : Nat :=
  match c with
  | 'A' => 1
  | 'B' => 2
  | 'U' => 21
  | 'K' => 11
  | _ => 0

/-- Encodes a city name to a list of alphabet positions --/
def encodeCity (name : String) : CityCode :=
  name.toList.map alphabetPosition

/-- Theorem: The encrypted city names represent Ufa and Baku --/
theorem train_journey_encryption (departure : CityCode) (arrival : CityCode) : 
  (departure = [21, 2, 1, 21] ∧ arrival = [2, 1, 11, 21]) →
  (encodeCity "UFA" = departure ∧ encodeCity "BAKU" = arrival) :=
by
  sorry

/-- Main theorem: The train traveled from Ufa to Baku --/
theorem train_journey : 
  ∃ (departure arrival : CityCode),
    departure = [21, 2, 1, 21] ∧
    arrival = [2, 1, 11, 21] ∧
    encodeCity "UFA" = departure ∧
    encodeCity "BAKU" = arrival :=
by
  sorry

end NUMINAMATH_CALUDE_train_journey_encryption_train_journey_l3896_389621


namespace NUMINAMATH_CALUDE_estimate_fish_population_l3896_389698

/-- Represents the fish pond scenario --/
structure FishPond where
  totalFish : ℕ  -- Total number of fish in the pond
  markedFish : ℕ  -- Number of fish initially marked
  secondSampleSize : ℕ  -- Size of the second sample
  markedInSecondSample : ℕ  -- Number of marked fish in the second sample

/-- Theorem stating the estimated number of fish in the pond --/
theorem estimate_fish_population (pond : FishPond) 
  (h1 : pond.markedFish = 100)
  (h2 : pond.secondSampleSize = 120)
  (h3 : pond.markedInSecondSample = 15) :
  pond.totalFish = 800 := by
  sorry

#check estimate_fish_population

end NUMINAMATH_CALUDE_estimate_fish_population_l3896_389698


namespace NUMINAMATH_CALUDE_trapezoid_construction_uniqueness_l3896_389664

-- Define the necessary types
def Line : Type := ℝ → ℝ → Prop
def Point : Type := ℝ × ℝ
def Direction : Type := ℝ × ℝ

-- Define the trapezoid structure
structure Trapezoid where
  side1 : Line
  side2 : Line
  diag1_start : Point
  diag1_end : Point
  diag2_direction : Direction

-- Define the theorem
theorem trapezoid_construction_uniqueness 
  (side1 side2 : Line)
  (E F : Point)
  (diag2_dir : Direction) :
  ∃! (trap : Trapezoid), 
    trap.side1 = side1 ∧ 
    trap.side2 = side2 ∧ 
    trap.diag1_start = E ∧ 
    trap.diag1_end = F ∧ 
    trap.diag2_direction = diag2_dir :=
sorry

end NUMINAMATH_CALUDE_trapezoid_construction_uniqueness_l3896_389664


namespace NUMINAMATH_CALUDE_intersection_k_value_l3896_389681

/-- Given two lines that intersect at a point, find the value of k -/
theorem intersection_k_value (k : ℝ) : 
  (∀ x y, y = 2 * x + 3 → (x = 1 ∧ y = 5)) →  -- Line m passes through (1, 5)
  (∀ x y, y = k * x + 2 → (x = 1 ∧ y = 5)) →  -- Line n passes through (1, 5)
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_k_value_l3896_389681


namespace NUMINAMATH_CALUDE_simplify_polynomial_no_x_squared_l3896_389653

-- Define the polynomial
def polynomial (x m : ℝ) : ℝ := 4*x^2 - 3*x + 5 - 2*m*x^2 - x + 1

-- Define the coefficient of x^2
def coeff_x_squared (m : ℝ) : ℝ := 4 - 2*m

-- Theorem statement
theorem simplify_polynomial_no_x_squared :
  ∃ (m : ℝ), coeff_x_squared m = 0 ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_simplify_polynomial_no_x_squared_l3896_389653


namespace NUMINAMATH_CALUDE_shortest_chord_length_for_given_circle_and_point_l3896_389687

/-- Circle represented by its equation -/
structure Circle where
  equation : ℝ → ℝ → ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the shortest chord length for a given circle and point -/
def shortestChordLength (c : Circle) (p : Point) : ℝ := sorry

/-- The main theorem -/
theorem shortest_chord_length_for_given_circle_and_point :
  let c : Circle := { equation := λ x y => x^2 + y^2 - 2*x - 3 }
  let p : Point := { x := 2, y := 1 }
  shortestChordLength c p = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_shortest_chord_length_for_given_circle_and_point_l3896_389687


namespace NUMINAMATH_CALUDE_darcys_shorts_l3896_389650

theorem darcys_shorts (total_shirts : ℕ) (folded_shirts : ℕ) (folded_shorts : ℕ) (remaining_to_fold : ℕ) : 
  total_shirts = 20 →
  folded_shirts = 12 →
  folded_shorts = 5 →
  remaining_to_fold = 11 →
  total_shirts + (folded_shorts + (remaining_to_fold - (total_shirts - folded_shirts))) = 28 :=
by sorry

end NUMINAMATH_CALUDE_darcys_shorts_l3896_389650


namespace NUMINAMATH_CALUDE_map_to_actual_distance_l3896_389659

/-- Given a map distance and scale, calculate the actual distance between two cities. -/
theorem map_to_actual_distance 
  (map_distance : ℝ) 
  (scale : ℝ) 
  (h1 : map_distance = 88) 
  (h2 : scale = 15) : 
  map_distance * scale = 1320 := by
  sorry

end NUMINAMATH_CALUDE_map_to_actual_distance_l3896_389659


namespace NUMINAMATH_CALUDE_sum_of_max_pairs_nonnegative_l3896_389693

theorem sum_of_max_pairs_nonnegative 
  (a b c d : ℝ) 
  (h : a + b + c + d = 0) : 
  max a b + max a c + max a d + max b c + max b d + max c d ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_pairs_nonnegative_l3896_389693


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l3896_389645

def second_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, α ∈ Set.Ioo (2 * k * Real.pi + Real.pi / 2) (2 * k * Real.pi + Real.pi)

theorem half_angle_quadrant (α : Real) (h : second_quadrant α) :
  ∃ k : ℤ, α / 2 ∈ Set.Ioo (k * Real.pi) (k * Real.pi + Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l3896_389645


namespace NUMINAMATH_CALUDE_max_volume_pyramid_l3896_389658

noncomputable def pyramid_volume (a b : ℝ) : ℝ :=
  (a * b * Real.sqrt (3 * a^2 - b^2)) / 6

theorem max_volume_pyramid (a : ℝ) (h : a > 0) :
  ∃ b : ℝ, b > 0 ∧ ∀ x : ℝ, x > 0 → pyramid_volume a b ≥ pyramid_volume a x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_max_volume_pyramid_l3896_389658


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3896_389625

theorem quadratic_roots_property (α β : ℝ) : 
  α ≠ β →
  α^2 + 3*α - 1 = 0 →
  β^2 + 3*β - 1 = 0 →
  α^2 + 4*α + β = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3896_389625


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3896_389639

theorem intersection_of_sets (a : ℝ) : 
  let A : Set ℝ := {-1, 0, 1}
  let B : Set ℝ := {a - 1, a + 1/a}
  (A ∩ B = {0}) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3896_389639


namespace NUMINAMATH_CALUDE_smallest_twin_egg_number_l3896_389607

def is_twin_egg_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 = n % 10) ∧
  ((n / 100) % 10 = (n / 10) % 10)

def exchange_digits (n : ℕ) : ℕ :=
  (n % 100) * 100 + (n / 100)

def F (m : ℕ) : ℚ :=
  (m - exchange_digits m : ℚ) / 11

theorem smallest_twin_egg_number :
  ∃ (m : ℕ),
    is_twin_egg_number m ∧
    (m / 1000 ≠ (m / 100) % 10) ∧
    ∃ (k : ℕ), F m / 54 = (k : ℚ) ^ 2 ∧
    ∀ (n : ℕ),
      is_twin_egg_number n ∧
      (n / 1000 ≠ (n / 100) % 10) ∧
      (∃ (j : ℕ), F n / 54 = (j : ℚ) ^ 2) →
      m ≤ n ∧
    m = 7117 :=
by sorry


end NUMINAMATH_CALUDE_smallest_twin_egg_number_l3896_389607


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3896_389672

theorem polynomial_factorization (x : ℝ) : 
  45 * x^6 - 270 * x^12 + 90 * x^7 = 45 * x^6 * (1 + 2*x - 6*x^6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3896_389672


namespace NUMINAMATH_CALUDE_port_vessels_l3896_389623

theorem port_vessels (cruise_ships cargo_ships sailboats fishing_boats : ℕ) :
  cruise_ships = 4 →
  cargo_ships = 2 * cruise_ships →
  ∃ (x : ℕ), sailboats = cargo_ships + x →
  sailboats = 7 * fishing_boats →
  cruise_ships + cargo_ships + sailboats + fishing_boats = 28 →
  sailboats - cargo_ships = 6 := by
  sorry

end NUMINAMATH_CALUDE_port_vessels_l3896_389623


namespace NUMINAMATH_CALUDE_vector_calculation_l3896_389649

def vector_subtraction (v w : Fin 2 → ℝ) : Fin 2 → ℝ := fun i => v i - w i

def scalar_mult (a : ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ := fun i => a * v i

theorem vector_calculation :
  let v : Fin 2 → ℝ := ![5, -3]
  let w : Fin 2 → ℝ := ![3, -4]
  vector_subtraction v (scalar_mult (-2) w) = ![11, -11] := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l3896_389649


namespace NUMINAMATH_CALUDE_expression_value_l3896_389633

theorem expression_value (p q r s : ℝ) 
  (hp : p ≠ 6) (hq : q ≠ 7) (hr : r ≠ 8) (hs : s ≠ 9) : 
  (p - 6) / (8 - r) * (q - 7) / (6 - p) * (r - 8) / (7 - q) * (s - 9) / (9 - s) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3896_389633


namespace NUMINAMATH_CALUDE_min_value_expression_l3896_389617

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  (8 / (x + 1)) + (1 / y) ≥ 25 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3896_389617


namespace NUMINAMATH_CALUDE_remaining_cooking_time_l3896_389685

def total_potatoes : ℕ := 15
def cooked_potatoes : ℕ := 6
def cooking_time_per_potato : ℕ := 8

theorem remaining_cooking_time : 
  (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 72 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cooking_time_l3896_389685


namespace NUMINAMATH_CALUDE_interest_rate_is_six_percent_l3896_389673

-- Define the loan parameters
def initial_loan : ℝ := 10000
def initial_period : ℝ := 2
def additional_loan : ℝ := 12000
def additional_period : ℝ := 3
def total_repayment : ℝ := 27160

-- Define the function to calculate the total amount to be repaid
def total_amount (rate : ℝ) : ℝ :=
  initial_loan * (1 + rate * (initial_period + additional_period)) +
  additional_loan * (1 + rate * additional_period)

-- Theorem statement
theorem interest_rate_is_six_percent :
  ∃ (rate : ℝ), rate > 0 ∧ rate < 1 ∧ total_amount rate = total_repayment ∧ rate = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_six_percent_l3896_389673


namespace NUMINAMATH_CALUDE_min_value_theorem_l3896_389678

theorem min_value_theorem (n : ℕ+) : 
  (n : ℝ) / 3 + 27 / (n : ℝ) ≥ 6 ∧ ∃ m : ℕ+, (m : ℝ) / 3 + 27 / (m : ℝ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3896_389678


namespace NUMINAMATH_CALUDE_cubic_roots_product_l3896_389667

theorem cubic_roots_product (a b c : ℂ) : 
  (3 * a^3 - 7 * a^2 + 4 * a - 9 = 0) ∧
  (3 * b^3 - 7 * b^2 + 4 * b - 9 = 0) ∧
  (3 * c^3 - 7 * c^2 + 4 * c - 9 = 0) →
  a * b * c = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_product_l3896_389667


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3896_389605

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + 16 = 12*x₁ ∧ x₂^2 + 16 = 12*x₂ ∧ 
   (∀ y : ℝ, y^2 + 16 = 12*y → y = x₁ ∨ y = x₂) ∧
   x₁ + x₂ = 12) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3896_389605


namespace NUMINAMATH_CALUDE_min_turns_to_win_l3896_389694

/-- Represents the state of the game -/
structure GameState :=
  (a₁ a₂ a₃ a₄ a₅ : ℕ)

/-- Defines a valid move in the game -/
def validMove (i : ℕ) : Prop :=
  2 ≤ i ∧ i ≤ 5

/-- Applies a move to the game state -/
def applyMove (state : GameState) (i : ℕ) : GameState :=
  match i with
  | 2 => GameState.mk state.a₁ (state.a₁ + state.a₂) state.a₃ state.a₄ state.a₅
  | 3 => GameState.mk state.a₁ state.a₂ (state.a₂ + state.a₃) state.a₄ state.a₅
  | 4 => GameState.mk state.a₁ state.a₂ state.a₃ (state.a₃ + state.a₄) state.a₅
  | 5 => GameState.mk state.a₁ state.a₂ state.a₃ state.a₄ (state.a₄ + state.a₅)
  | _ => state

/-- The initial state of the game -/
def initialState : GameState :=
  GameState.mk 1 0 0 0 0

/-- Predicate to check if the game is won -/
def isWinningState (state : GameState) : Prop :=
  state.a₅ > 1000000

/-- Theorem: The minimum number of turns to win the game is 127 -/
theorem min_turns_to_win :
  ∃ (moves : List ℕ),
    (∀ m ∈ moves, validMove m) ∧
    isWinningState (moves.foldl applyMove initialState) ∧
    moves.length = 127 ∧
    (∀ (other_moves : List ℕ),
      (∀ m ∈ other_moves, validMove m) →
      isWinningState (other_moves.foldl applyMove initialState) →
      other_moves.length ≥ 127) :=
by sorry


end NUMINAMATH_CALUDE_min_turns_to_win_l3896_389694


namespace NUMINAMATH_CALUDE_total_students_on_trip_l3896_389627

/-- The number of students who went on a trip to the zoo -/
def students_on_trip (num_buses : ℕ) (students_per_bus : ℕ) (students_in_cars : ℕ) : ℕ :=
  num_buses * students_per_bus + students_in_cars

/-- Theorem stating the total number of students on the trip -/
theorem total_students_on_trip :
  students_on_trip 7 56 4 = 396 := by
  sorry

end NUMINAMATH_CALUDE_total_students_on_trip_l3896_389627


namespace NUMINAMATH_CALUDE_son_work_time_l3896_389622

def work_problem (man_time son_time combined_time : ℝ) : Prop :=
  man_time > 0 ∧ son_time > 0 ∧ combined_time > 0 ∧
  1 / man_time + 1 / son_time = 1 / combined_time

theorem son_work_time (man_time combined_time : ℝ) 
  (h1 : man_time = 5)
  (h2 : combined_time = 4)
  : ∃ (son_time : ℝ), work_problem man_time son_time combined_time ∧ son_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_son_work_time_l3896_389622


namespace NUMINAMATH_CALUDE_money_division_l3896_389692

theorem money_division (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / q = 3 / 7 →
  q / r = 7 / 12 →
  q - p = 4400 →
  r - q = 5500 := by
sorry

end NUMINAMATH_CALUDE_money_division_l3896_389692


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3896_389609

theorem fraction_multiplication :
  (2 : ℚ) / 3 * (5 : ℚ) / 7 * (11 : ℚ) / 13 = (110 : ℚ) / 273 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3896_389609


namespace NUMINAMATH_CALUDE_tan_period_l3896_389695

/-- The smallest positive period of tan((a + b)x/2) given conditions -/
theorem tan_period (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : a + b = 1) : 
  let f := fun x => Real.tan ((a + b) * x / 2)
  ∃ p : ℝ, p > 0 ∧ (∀ x, f (x + p) = f x) ∧ 
    (∀ q, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q) ∧
  p = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_tan_period_l3896_389695


namespace NUMINAMATH_CALUDE_not_invertible_sum_of_squares_l3896_389674

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem not_invertible_sum_of_squares (M N : Matrix n n ℝ) 
  (h_neq : M ≠ N) 
  (h_cube : M ^ 3 = N ^ 3) 
  (h_comm : M ^ 2 * N = N ^ 2 * M) : 
  ¬(IsUnit (M ^ 2 + N ^ 2)) := by
sorry

end NUMINAMATH_CALUDE_not_invertible_sum_of_squares_l3896_389674


namespace NUMINAMATH_CALUDE_age_difference_l3896_389615

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 12) : A - C = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3896_389615


namespace NUMINAMATH_CALUDE_probability_same_color_plates_l3896_389686

def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def green_plates : ℕ := 3

def total_plates : ℕ := red_plates + blue_plates + green_plates

def same_color_combinations : ℕ := (
  Nat.choose red_plates 3 +
  Nat.choose blue_plates 3 +
  Nat.choose green_plates 3
)

def total_combinations : ℕ := Nat.choose total_plates 3

theorem probability_same_color_plates :
  (same_color_combinations : ℚ) / total_combinations = 31 / 364 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_plates_l3896_389686


namespace NUMINAMATH_CALUDE_paving_cost_theorem_l3896_389680

/-- Represents the dimensions and cost of a rectangular room -/
structure RectangularRoom where
  length : ℝ
  width : ℝ
  cost_per_sqm : ℝ

/-- Represents the dimensions and cost of a triangular room -/
structure TriangularRoom where
  base : ℝ
  height : ℝ
  cost_per_sqm : ℝ

/-- Represents the dimensions and cost of a trapezoidal room -/
structure TrapezoidalRoom where
  parallel_side1 : ℝ
  parallel_side2 : ℝ
  height : ℝ
  cost_per_sqm : ℝ

/-- Calculates the total cost of paving three rooms -/
def total_paving_cost (room1 : RectangularRoom) (room2 : TriangularRoom) (room3 : TrapezoidalRoom) : ℝ :=
  (room1.length * room1.width * room1.cost_per_sqm) +
  (0.5 * room2.base * room2.height * room2.cost_per_sqm) +
  (0.5 * (room3.parallel_side1 + room3.parallel_side2) * room3.height * room3.cost_per_sqm)

/-- Theorem stating the total cost of paving the three rooms -/
theorem paving_cost_theorem (room1 : RectangularRoom) (room2 : TriangularRoom) (room3 : TrapezoidalRoom)
  (h1 : room1 = { length := 5.5, width := 3.75, cost_per_sqm := 1400 })
  (h2 : room2 = { base := 4, height := 3, cost_per_sqm := 1500 })
  (h3 : room3 = { parallel_side1 := 6, parallel_side2 := 3.5, height := 2.5, cost_per_sqm := 1600 }) :
  total_paving_cost room1 room2 room3 = 56875 := by
  sorry

#eval total_paving_cost
  { length := 5.5, width := 3.75, cost_per_sqm := 1400 }
  { base := 4, height := 3, cost_per_sqm := 1500 }
  { parallel_side1 := 6, parallel_side2 := 3.5, height := 2.5, cost_per_sqm := 1600 }

end NUMINAMATH_CALUDE_paving_cost_theorem_l3896_389680


namespace NUMINAMATH_CALUDE_boss_salary_percentage_larger_l3896_389629

-- Define Werner's salary as a percentage of his boss's salary
def werner_salary_percentage : ℝ := 20

-- Theorem statement
theorem boss_salary_percentage_larger (werner_salary boss_salary : ℝ) 
  (h : werner_salary = (werner_salary_percentage / 100) * boss_salary) : 
  (boss_salary / werner_salary - 1) * 100 = 400 := by
  sorry

end NUMINAMATH_CALUDE_boss_salary_percentage_larger_l3896_389629


namespace NUMINAMATH_CALUDE_floor_length_approx_l3896_389657

/-- Represents a rectangular floor with length and breadth -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ

/-- The properties of our specific rectangular floor -/
def floor_properties (floor : RectangularFloor) : Prop :=
  floor.length = 3 * floor.breadth ∧
  floor.length * floor.breadth = 60

/-- The theorem stating the length of the floor -/
theorem floor_length_approx (floor : RectangularFloor) 
  (h : floor_properties floor) : 
  ∃ ε > 0, abs (floor.length - 13.416) < ε :=
sorry

end NUMINAMATH_CALUDE_floor_length_approx_l3896_389657


namespace NUMINAMATH_CALUDE_carnival_tickets_l3896_389610

def ticket_distribution (n : Nat) : Nat :=
  let ratio := [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
  let total_parts := ratio.sum
  let tickets_per_part := n / total_parts
  tickets_per_part * total_parts

theorem carnival_tickets :
  let friends : Nat := 17
  let initial_tickets : Nat := 865
  let ratio := [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
  let total_parts := ratio.sum
  let next_multiple := (initial_tickets / total_parts + 1) * total_parts
  next_multiple - initial_tickets = 26 := by
  sorry

end NUMINAMATH_CALUDE_carnival_tickets_l3896_389610


namespace NUMINAMATH_CALUDE_animal_sightings_proof_l3896_389646

/-- The number of times families see animals in January -/
def january_sightings : ℕ := 26

/-- The number of times families see animals in February -/
def february_sightings : ℕ := 3 * january_sightings

/-- The number of times families see animals in March -/
def march_sightings : ℕ := february_sightings / 2

/-- The total number of times families see animals in the first three months -/
def total_sightings : ℕ := january_sightings + february_sightings + march_sightings

theorem animal_sightings_proof : total_sightings = 143 := by
  sorry

end NUMINAMATH_CALUDE_animal_sightings_proof_l3896_389646


namespace NUMINAMATH_CALUDE_sand_art_proof_l3896_389682

/-- The amount of sand needed to fill a rectangular patch and a square patch -/
def total_sand_needed (rect_length rect_width square_side sand_per_inch : ℕ) : ℕ :=
  ((rect_length * rect_width) + (square_side * square_side)) * sand_per_inch

/-- Proof that the total amount of sand needed is 201 grams -/
theorem sand_art_proof :
  total_sand_needed 6 7 5 3 = 201 := by
  sorry

end NUMINAMATH_CALUDE_sand_art_proof_l3896_389682


namespace NUMINAMATH_CALUDE_money_loses_exchange_value_on_deserted_island_l3896_389611

-- Define the basic concepts
def Person : Type := String
def Money : Type := ℕ
def Item : Type := String

-- Define the properties of money
structure MoneyProperties :=
  (medium_of_exchange : Bool)
  (store_of_value : Bool)
  (unit_of_account : Bool)
  (standard_of_deferred_payment : Bool)

-- Define the island environment
structure Island :=
  (inhabitants : List Person)
  (items : List Item)
  (currency : Money)

-- Define the value of money in a given context
def money_value (island : Island) (props : MoneyProperties) : ℝ := 
  sorry

-- Theorem: Money loses its value as a medium of exchange on a deserted island
theorem money_loses_exchange_value_on_deserted_island 
  (island : Island) 
  (props : MoneyProperties) :
  island.inhabitants.length = 1 →
  money_value island props = 0 :=
sorry

end NUMINAMATH_CALUDE_money_loses_exchange_value_on_deserted_island_l3896_389611


namespace NUMINAMATH_CALUDE_probability_all_red_or_all_white_l3896_389699

/-- The probability of drawing either all red marbles or all white marbles when drawing 3 marbles
    without replacement from a bag containing 5 red, 4 white, and 6 blue marbles -/
theorem probability_all_red_or_all_white (total_marbles : ℕ) (red_marbles : ℕ) (white_marbles : ℕ) 
    (blue_marbles : ℕ) (drawn_marbles : ℕ) :
  total_marbles = red_marbles + white_marbles + blue_marbles →
  total_marbles = 15 →
  red_marbles = 5 →
  white_marbles = 4 →
  blue_marbles = 6 →
  drawn_marbles = 3 →
  (red_marbles.choose drawn_marbles * (total_marbles - drawn_marbles).factorial / total_marbles.factorial +
   white_marbles.choose drawn_marbles * (total_marbles - drawn_marbles).factorial / total_marbles.factorial : ℚ) = 14 / 455 := by
  sorry

#check probability_all_red_or_all_white

end NUMINAMATH_CALUDE_probability_all_red_or_all_white_l3896_389699


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l3896_389608

theorem real_part_of_complex_fraction :
  let i : ℂ := Complex.I
  (2 * i / (1 + i)).re = 1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l3896_389608


namespace NUMINAMATH_CALUDE_perpendicular_lines_theorem_l3896_389604

/-- Given two perpendicular lines and their point of intersection, prove that m + n - p = 0 --/
theorem perpendicular_lines_theorem (m n p : ℝ) : 
  (∀ x y, m * x + 4 * y - 2 = 0 ↔ 2 * x - 5 * y + n = 0) →  -- Lines are perpendicular
  (m * 1 + 4 * p - 2 = 0) →  -- (1, p) is on the first line
  (2 * 1 - 5 * p + n = 0) →  -- (1, p) is on the second line
  m + n - p = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_theorem_l3896_389604


namespace NUMINAMATH_CALUDE_bank_account_deposit_fraction_l3896_389662

theorem bank_account_deposit_fraction (B : ℝ) (f : ℝ) : 
  B > 0 →
  (3/5) * B = B - 400 →
  600 + f * 600 = 750 →
  f = 1/4 := by
sorry

end NUMINAMATH_CALUDE_bank_account_deposit_fraction_l3896_389662


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3896_389677

-- Define the universal set U
def U : Finset Nat := {1,2,3,4,5,6}

-- Define set P
def P : Finset Nat := {1,2,3,4}

-- Define set Q
def Q : Finset Nat := {3,4,5}

-- Theorem statement
theorem intersection_with_complement :
  P ∩ (U \ Q) = {1,2} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3896_389677


namespace NUMINAMATH_CALUDE_min_distance_exp_curve_to_line_l3896_389647

/-- The minimum distance from a point on the curve y = e^x to the line y = x is √2/2 -/
theorem min_distance_exp_curve_to_line : 
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧ 
  ∀ (x y : ℝ), y = Real.exp x → 
  d ≤ Real.sqrt ((x - y)^2 + (y - x)^2) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_exp_curve_to_line_l3896_389647


namespace NUMINAMATH_CALUDE_square_floor_tiles_l3896_389636

theorem square_floor_tiles (s : ℕ) (h1 : s > 0) : 
  (2 * s - 1 : ℝ) / (s^2 : ℝ) = 0.41 → s^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l3896_389636


namespace NUMINAMATH_CALUDE_not_all_problems_solvable_by_algorithm_l3896_389660

/-- Represents a problem that can be solved computationally -/
def Problem : Type := Unit

/-- Represents an algorithm -/
def Algorithm : Type := Unit

/-- Represents the characteristic that an algorithm is executed step by step -/
def stepwise (a : Algorithm) : Prop := sorry

/-- Represents the characteristic that each step of an algorithm yields a unique result -/
def uniqueStepResult (a : Algorithm) : Prop := sorry

/-- Represents the characteristic that algorithms are effective for a class of problems -/
def effectiveForClass (a : Algorithm) : Prop := sorry

/-- Represents the characteristic that algorithms are mechanical -/
def mechanical (a : Algorithm) : Prop := sorry

/-- Represents the characteristic that algorithms can require repetitive calculation -/
def repetitiveCalculation (a : Algorithm) : Prop := sorry

/-- Represents the characteristic that algorithms are a universal method -/
def universalMethod (a : Algorithm) : Prop := sorry

/-- Theorem stating that not all problems can be solved by algorithms -/
theorem not_all_problems_solvable_by_algorithm : 
  ¬ (∀ (p : Problem), ∃ (a : Algorithm), 
    stepwise a ∧ 
    uniqueStepResult a ∧ 
    effectiveForClass a ∧ 
    mechanical a ∧ 
    repetitiveCalculation a ∧ 
    universalMethod a) := by sorry


end NUMINAMATH_CALUDE_not_all_problems_solvable_by_algorithm_l3896_389660


namespace NUMINAMATH_CALUDE_cube_coloring_count_l3896_389668

/-- The number of distinct colorings of a cube's vertices -/
def distinctCubeColorings (m : ℕ) : ℚ :=
  (1 / 24) * m^2 * (m^6 + 17 * m^2 + 6)

/-- Theorem: The number of distinct ways to color the 8 vertices of a cube
    with m different colors, considering the symmetries of the cube,
    is equal to (1/24) * m^2 * (m^6 + 17m^2 + 6) -/
theorem cube_coloring_count (m : ℕ) :
  (distinctCubeColorings m) = (1 / 24) * m^2 * (m^6 + 17 * m^2 + 6) := by
  sorry

end NUMINAMATH_CALUDE_cube_coloring_count_l3896_389668


namespace NUMINAMATH_CALUDE_parabola_translation_l3896_389642

/-- Represents a parabola in 2D space -/
structure Parabola where
  f : ℝ → ℝ

/-- Applies a vertical translation to a parabola -/
def verticalTranslate (p : Parabola) (v : ℝ) : Parabola where
  f := fun x => p.f x + v

/-- Applies a horizontal translation to a parabola -/
def horizontalTranslate (p : Parabola) (h : ℝ) : Parabola where
  f := fun x => p.f (x + h)

/-- The original parabola y = -x^2 -/
def originalParabola : Parabola where
  f := fun x => -x^2

/-- Theorem stating that translating the parabola y = -x^2 upward by 2 units
    and to the left by 3 units results in the equation y = -(x + 3)^2 + 2 -/
theorem parabola_translation :
  (horizontalTranslate (verticalTranslate originalParabola 2) 3).f =
  fun x => -(x + 3)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l3896_389642


namespace NUMINAMATH_CALUDE_probability_defect_free_l3896_389697

/-- Represents the proportion of components from Company A in the warehouse -/
def proportion_A : ℝ := 0.60

/-- Represents the proportion of components from Company B in the warehouse -/
def proportion_B : ℝ := 0.40

/-- Represents the defect rate of components from Company A -/
def defect_rate_A : ℝ := 0.98

/-- Represents the defect rate of components from Company B -/
def defect_rate_B : ℝ := 0.95

/-- Theorem stating that the probability of a randomly selected component being defect-free is 0.032 -/
theorem probability_defect_free : 
  proportion_A * (1 - defect_rate_A) + proportion_B * (1 - defect_rate_B) = 0.032 := by
  sorry


end NUMINAMATH_CALUDE_probability_defect_free_l3896_389697


namespace NUMINAMATH_CALUDE_combined_original_price_l3896_389644

/-- Proves that the combined original price of a candy box, a can of soda, and a bag of chips
    was 34 pounds, given their new prices after specific percentage increases. -/
theorem combined_original_price (candy_new : ℝ) (soda_new : ℝ) (chips_new : ℝ)
    (h_candy : candy_new = 20)
    (h_soda : soda_new = 6)
    (h_chips : chips_new = 8)
    (h_candy_increase : candy_new = (5/4) * (candy_new - (1/4) * candy_new))
    (h_soda_increase : soda_new = (3/2) * (soda_new - (1/2) * soda_new))
    (h_chips_increase : chips_new = (11/10) * (chips_new - (1/10) * chips_new)) :
  (candy_new - (1/4) * candy_new) + (soda_new - (1/2) * soda_new) + (chips_new - (1/10) * chips_new) = 34 := by
  sorry

end NUMINAMATH_CALUDE_combined_original_price_l3896_389644


namespace NUMINAMATH_CALUDE_x_intercept_after_rotation_l3896_389669

/-- Given a line m with equation 2x - 3y + 30 = 0 in the coordinate plane,
    rotated 30° counterclockwise about the point (10, 10) to form line n,
    the x-coordinate of the x-intercept of line n is (20√3 + 20) / (2√3 + 3). -/
theorem x_intercept_after_rotation :
  let m : Set (ℝ × ℝ) := {(x, y) | 2 * x - 3 * y + 30 = 0}
  let center : ℝ × ℝ := (10, 10)
  let angle : ℝ := π / 6  -- 30° in radians
  let n : Set (ℝ × ℝ) := {(x, y) | ∃ (x₀ y₀ : ℝ), (x₀, y₀) ∈ m ∧
    x - 10 = (x₀ - 10) * Real.cos angle - (y₀ - 10) * Real.sin angle ∧
    y - 10 = (x₀ - 10) * Real.sin angle + (y₀ - 10) * Real.cos angle}
  let x_intercept : ℝ := (20 * Real.sqrt 3 + 20) / (2 * Real.sqrt 3 + 3)
  (0, x_intercept) ∈ n := by sorry

end NUMINAMATH_CALUDE_x_intercept_after_rotation_l3896_389669


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l3896_389635

-- Define the hyperbola equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  y^2 + x^2/m = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x

-- Theorem statement
theorem hyperbola_m_value :
  ∀ m : ℝ, (∀ x y : ℝ, hyperbola_equation x y m ↔ asymptote_equation x y) → m = -3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l3896_389635


namespace NUMINAMATH_CALUDE_common_point_theorem_l3896_389648

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Represents a geometric progression -/
def IsGeometricProgression (a c b : ℝ) : Prop :=
  ∃ r : ℝ, c = a * r ∧ b = a * r^2

theorem common_point_theorem :
  ∀ (l : Line), 
    IsGeometricProgression l.a l.c l.b →
    l.contains 0 0 :=
by sorry

end NUMINAMATH_CALUDE_common_point_theorem_l3896_389648


namespace NUMINAMATH_CALUDE_max_true_statements_l3896_389691

theorem max_true_statements (a b : ℝ) : ∃ a b : ℝ,
  (a^2 + b^2 < (a + b)^2) ∧
  (a * b > 0) ∧
  (a > b) ∧
  (a > 0) ∧
  (b > 0) := by
sorry

end NUMINAMATH_CALUDE_max_true_statements_l3896_389691


namespace NUMINAMATH_CALUDE_probability_at_least_two_same_8sided_dice_l3896_389675

theorem probability_at_least_two_same_8sided_dice (n : ℕ) (s : ℕ) (p : ℚ) :
  n = 5 →
  s = 8 →
  p = 1628 / 2048 →
  p = 1 - (s * (s - 1) * (s - 2) * (s - 3) * (s - 4) : ℚ) / s^n :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_two_same_8sided_dice_l3896_389675


namespace NUMINAMATH_CALUDE_quadratic_equations_root_range_l3896_389684

/-- The range of real numbers for a, such that at most two of the given three quadratic equations do not have real roots -/
theorem quadratic_equations_root_range : 
  {a : ℝ | (∃ x : ℝ, x^2 - a*x + 9 = 0) ∨ 
           (∃ x : ℝ, x^2 + a*x - 2*a = 0) ∨ 
           (∃ x : ℝ, x^2 + (a+1)*x + 9/4 = 0)} = 
  {a : ℝ | a ≤ -4 ∨ a ≥ 0} := by sorry

end NUMINAMATH_CALUDE_quadratic_equations_root_range_l3896_389684


namespace NUMINAMATH_CALUDE_route_comparison_l3896_389632

/-- Represents the time difference between two routes when all lights are red on the first route -/
def route_time_difference (first_route_base_time : ℕ) (red_light_delay : ℕ) (num_lights : ℕ) (second_route_time : ℕ) : ℕ :=
  (first_route_base_time + red_light_delay * num_lights) - second_route_time

theorem route_comparison :
  route_time_difference 10 3 3 14 = 5 := by
  sorry

end NUMINAMATH_CALUDE_route_comparison_l3896_389632


namespace NUMINAMATH_CALUDE_partition_6_4_l3896_389602

/-- The number of ways to partition n indistinguishable objects into at most k indistinguishable parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 9 ways to partition 6 indistinguishable objects into at most 4 indistinguishable parts -/
theorem partition_6_4 : partition_count 6 4 = 9 := by sorry

end NUMINAMATH_CALUDE_partition_6_4_l3896_389602


namespace NUMINAMATH_CALUDE_mark_kate_difference_l3896_389613

/-- The number of hours Kate charged to the project -/
def kate_hours : ℕ := 28

/-- The total number of hours charged to the project -/
def total_hours : ℕ := 180

/-- Pat's hours are twice Kate's -/
def pat_hours : ℕ := 2 * kate_hours

/-- Mark's hours are three times Kate's -/
def mark_hours : ℕ := 3 * kate_hours

/-- Linda's hours are half of Kate's -/
def linda_hours : ℕ := kate_hours / 2

theorem mark_kate_difference :
  mark_hours - kate_hours = 56 ∧
  pat_hours + kate_hours + mark_hours + linda_hours = total_hours :=
by sorry

end NUMINAMATH_CALUDE_mark_kate_difference_l3896_389613


namespace NUMINAMATH_CALUDE_log_27_3_l3896_389643

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_27_3_l3896_389643


namespace NUMINAMATH_CALUDE_function_equality_l3896_389679

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else a^x

theorem function_equality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (Real.exp 2) = f a (-2) → a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l3896_389679


namespace NUMINAMATH_CALUDE_bd_length_l3896_389676

-- Define the triangles and their properties
def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem bd_length (c : ℝ) :
  ∀ (AB BC AC AD BD : ℝ),
  right_triangle BC AC AB →
  right_triangle AD BD AB →
  BC = 3 →
  AC = c →
  AD = c - 1 →
  BD = Real.sqrt (2 * c + 8) := by
  sorry

end NUMINAMATH_CALUDE_bd_length_l3896_389676


namespace NUMINAMATH_CALUDE_equation_solution_l3896_389690

theorem equation_solution (x y : ℝ) : 
  y = 3 * x → 
  (4 * y^2 - 3 * y + 5 = 3 * (8 * x^2 - 3 * y + 1)) ↔ 
  (x = (Real.sqrt 19 - 3) / 4 ∨ x = (-Real.sqrt 19 - 3) / 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3896_389690


namespace NUMINAMATH_CALUDE_inequality_proof_l3896_389616

theorem inequality_proof (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a + b + c ≤ 2) : 
  Real.sqrt (b^2 + a*c) + Real.sqrt (a^2 + b*c) + Real.sqrt (c^2 + a*b) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3896_389616


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3896_389640

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {-2, -1, 0, 1}

theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3896_389640


namespace NUMINAMATH_CALUDE_prop_c_prop_d_l3896_389665

-- Proposition C
theorem prop_c (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = 1) :
  1/(2*a) + 1/b ≥ 4 := by sorry

-- Proposition D
theorem prop_d (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 4) :
  ∃ (m : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 4 →
    x^2/(x+1) + y^2/(y+1) ≥ m ∧ 
    (∃ (u v : ℝ), u > 0 ∧ v > 0 ∧ u + v = 4 ∧ u^2/(u+1) + v^2/(v+1) = m) ∧
    m = 8/3 := by sorry

end NUMINAMATH_CALUDE_prop_c_prop_d_l3896_389665


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3896_389688

theorem problem_1 : (1) - 2^2 + (-1/2)^4 + (3 - Real.pi)^0 = -47/16 := by sorry

theorem problem_2 : 5^2022 * (-1/5)^2023 = -1/5 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3896_389688


namespace NUMINAMATH_CALUDE_expression_evaluation_l3896_389630

theorem expression_evaluation : 
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3896_389630


namespace NUMINAMATH_CALUDE_telephone_fee_properties_l3896_389656

-- Define the telephone fee function
def telephone_fee (x : ℝ) : ℝ := 0.4 * x + 18

-- Theorem statement
theorem telephone_fee_properties :
  (∀ x : ℝ, telephone_fee x = 0.4 * x + 18) ∧
  (telephone_fee 10 = 22) ∧
  (telephone_fee 20 = 26) := by
  sorry


end NUMINAMATH_CALUDE_telephone_fee_properties_l3896_389656


namespace NUMINAMATH_CALUDE_solution_set_range_of_a_l3896_389683

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 4| + |x + 1|

-- Theorem for the solution of f(x) ≤ 9
theorem solution_set (x : ℝ) : f x ≤ 9 ↔ x ∈ Set.Icc (-2) 4 := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Icc 0 2, f x = -x^2 + a) ↔ a ∈ Set.Icc (19/4) 7 := by sorry

end NUMINAMATH_CALUDE_solution_set_range_of_a_l3896_389683


namespace NUMINAMATH_CALUDE_kevin_repaired_phones_l3896_389652

/-- The number of phones Kevin repaired by the afternoon -/
def phones_repaired : ℕ := 3

/-- The initial number of phones Kevin had to repair -/
def initial_phones : ℕ := 15

/-- The number of phones dropped off by a client -/
def new_phones : ℕ := 6

/-- The number of phones each person (Kevin and his coworker) needs to repair -/
def phones_per_person : ℕ := 9

theorem kevin_repaired_phones :
  phones_repaired = 3 ∧
  initial_phones - phones_repaired + new_phones = 2 * phones_per_person :=
sorry

end NUMINAMATH_CALUDE_kevin_repaired_phones_l3896_389652


namespace NUMINAMATH_CALUDE_cubic_tangent_line_problem_l3896_389641

/-- Given a cubic function f(x) = ax³ + x + 1, prove that if its tangent line
    at x = 1 passes through the point (2, 7), then a = 1. -/
theorem cubic_tangent_line_problem (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + x + 1
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 + 1
  let tangent_line : ℝ → ℝ := λ x ↦ f 1 + f' 1 * (x - 1)
  tangent_line 2 = 7 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_tangent_line_problem_l3896_389641


namespace NUMINAMATH_CALUDE_bus_riders_percentage_l3896_389671

/-- Represents the scenario of introducing a bus service in Johnstown --/
structure BusScenario where
  population : Nat
  car_pollution : Nat
  bus_pollution : Nat
  bus_capacity : Nat
  carbon_reduction : Nat

/-- Calculates the percentage of people who now take the bus --/
def percentage_bus_riders (scenario : BusScenario) : Rat :=
  let cars_removed := scenario.carbon_reduction / scenario.car_pollution
  (cars_removed : Rat) / scenario.population * 100

/-- Theorem stating that the percentage of people who now take the bus is 12.5% --/
theorem bus_riders_percentage (scenario : BusScenario) 
  (h1 : scenario.population = 80)
  (h2 : scenario.car_pollution = 10)
  (h3 : scenario.bus_pollution = 100)
  (h4 : scenario.bus_capacity = 40)
  (h5 : scenario.carbon_reduction = 100) :
  percentage_bus_riders scenario = 25/2 := by
  sorry

#eval percentage_bus_riders {
  population := 80,
  car_pollution := 10,
  bus_pollution := 100,
  bus_capacity := 40,
  carbon_reduction := 100
}

end NUMINAMATH_CALUDE_bus_riders_percentage_l3896_389671


namespace NUMINAMATH_CALUDE_tickets_left_kaleb_tickets_left_l3896_389661

theorem tickets_left (initial_tickets : ℕ) (ticket_cost : ℕ) (spent_on_ride : ℕ) : ℕ :=
  let tickets_used := spent_on_ride / ticket_cost
  initial_tickets - tickets_used

theorem kaleb_tickets_left :
  tickets_left 6 9 27 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tickets_left_kaleb_tickets_left_l3896_389661


namespace NUMINAMATH_CALUDE_point_line_distance_l3896_389618

/-- Given a point (4, 3) and a line 3x - 4y + a = 0, if the distance from the point to the line is 1, then a = ±5 -/
theorem point_line_distance (a : ℝ) : 
  let point : ℝ × ℝ := (4, 3)
  let line_equation (x y : ℝ) := 3 * x - 4 * y + a
  let distance := |line_equation point.1 point.2| / Real.sqrt (3^2 + (-4)^2)
  distance = 1 → a = 5 ∨ a = -5 := by
sorry

end NUMINAMATH_CALUDE_point_line_distance_l3896_389618


namespace NUMINAMATH_CALUDE_percentage_difference_l3896_389670

theorem percentage_difference (x y : ℝ) (h : x = y * (1 - 0.4)) :
  (y - x) / x = 0.4 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l3896_389670


namespace NUMINAMATH_CALUDE_tan_three_expression_value_l3896_389624

theorem tan_three_expression_value (θ : Real) (h : Real.tan θ = 3) :
  2 * (Real.sin θ)^2 - 3 * (Real.sin θ) * (Real.cos θ) - 4 * (Real.cos θ)^2 = -4/10 := by
  sorry

end NUMINAMATH_CALUDE_tan_three_expression_value_l3896_389624


namespace NUMINAMATH_CALUDE_alex_needs_three_packs_l3896_389654

/-- The number of burgers Alex plans to cook for each guest -/
def burgers_per_guest : ℕ := 3

/-- The number of friends Alex invited -/
def total_friends : ℕ := 10

/-- The number of friends who don't eat meat -/
def non_meat_eaters : ℕ := 1

/-- The number of friends who don't eat bread -/
def non_bread_eaters : ℕ := 1

/-- The number of buns in each pack -/
def buns_per_pack : ℕ := 8

/-- The function to calculate the number of packs of buns Alex needs to buy -/
def packs_of_buns_needed : ℕ :=
  let total_guests := total_friends - non_meat_eaters
  let total_burgers := burgers_per_guest * total_guests
  let burgers_needing_buns := total_burgers - (burgers_per_guest * non_bread_eaters)
  (burgers_needing_buns + buns_per_pack - 1) / buns_per_pack

/-- Theorem stating that Alex needs to buy 3 packs of buns -/
theorem alex_needs_three_packs : packs_of_buns_needed = 3 := by
  sorry

end NUMINAMATH_CALUDE_alex_needs_three_packs_l3896_389654


namespace NUMINAMATH_CALUDE_buddy_card_count_l3896_389603

def card_count (initial : ℕ) : ℕ := 
  let tuesday := initial - (initial * 30 / 100)
  let wednesday := tuesday + (tuesday * 20 / 100)
  let thursday := wednesday - (wednesday * 25 / 100)
  let friday := thursday + (thursday / 3)
  let saturday := friday + (friday * 2)
  let sunday := saturday + (saturday * 40 / 100) - 15
  let next_monday := sunday + ((saturday * 40 / 100) * 3)
  next_monday

theorem buddy_card_count : card_count 200 = 1297 := by
  sorry

end NUMINAMATH_CALUDE_buddy_card_count_l3896_389603


namespace NUMINAMATH_CALUDE_pedro_extra_squares_l3896_389612

theorem pedro_extra_squares (jesus_squares linden_squares pedro_squares : ℕ) 
  (h1 : jesus_squares = 60)
  (h2 : linden_squares = 75)
  (h3 : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 := by
  sorry

end NUMINAMATH_CALUDE_pedro_extra_squares_l3896_389612


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3896_389651

theorem two_numbers_difference (a b : ℕ) : 
  a + b = 20500 →
  b % 5 = 0 →
  b = 10 * a + 5 →
  b - a = 16777 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3896_389651


namespace NUMINAMATH_CALUDE_function_symmetry_and_translation_l3896_389601

-- Define a function that represents a horizontal translation
def translate (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := λ x ↦ f (x - h)

-- Define symmetry with respect to y-axis
def symmetric_to_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

-- State the theorem
theorem function_symmetry_and_translation :
  ∀ f : ℝ → ℝ,
  symmetric_to_y_axis (translate f 1) (λ x ↦ Real.exp x) →
  f = λ x ↦ Real.exp (-x - 1) :=
sorry

end NUMINAMATH_CALUDE_function_symmetry_and_translation_l3896_389601


namespace NUMINAMATH_CALUDE_cards_found_l3896_389637

theorem cards_found (initial_cards final_cards : ℕ) : 
  initial_cards = 7 → final_cards = 54 → final_cards - initial_cards = 47 := by
  sorry

end NUMINAMATH_CALUDE_cards_found_l3896_389637


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3896_389631

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 1 = 2) 
  (h3 : a 3 = 8) : 
  a 2 - a 1 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3896_389631


namespace NUMINAMATH_CALUDE_classroom_seating_l3896_389619

/-- Given a classroom with 53 students seated in rows of either 6 or 7 students,
    with all seats occupied, prove that the number of rows seating exactly 7 students is 5. -/
theorem classroom_seating (total_students : ℕ) (rows_with_seven : ℕ) : 
  total_students = 53 →
  (∃ (rows_with_six : ℕ), total_students = 7 * rows_with_seven + 6 * rows_with_six) →
  rows_with_seven = 5 := by
  sorry

end NUMINAMATH_CALUDE_classroom_seating_l3896_389619


namespace NUMINAMATH_CALUDE_sine_shift_left_specific_sine_shift_shift_result_l3896_389606

/-- Shifting a sine function to the left -/
theorem sine_shift_left (A : ℝ) (ω : ℝ) (φ : ℝ) (h : ℝ) :
  (fun x => A * Real.sin (ω * (x + h) + φ)) =
  (fun x => A * Real.sin (ω * x + (ω * h + φ))) :=
by sorry

/-- The specific case of shifting y = 3sin(2x + π/6) left by π/6 -/
theorem specific_sine_shift :
  (fun x => 3 * Real.sin (2 * x + π/6)) =
  (fun x => 3 * Real.sin (2 * (x - π/6) + π/6)) :=
by sorry

/-- The result of the shift is y = 3sin(2x - π/6) -/
theorem shift_result :
  (fun x => 3 * Real.sin (2 * (x - π/6) + π/6)) =
  (fun x => 3 * Real.sin (2 * x - π/6)) :=
by sorry

end NUMINAMATH_CALUDE_sine_shift_left_specific_sine_shift_shift_result_l3896_389606
