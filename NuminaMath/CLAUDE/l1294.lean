import Mathlib

namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_with_large_factors_l1294_129438

theorem smallest_non_prime_non_square_with_large_factors : ∃ n : ℕ+, 
  (n.val = 5183 ∧ 
   ¬ Nat.Prime n.val ∧ 
   ¬ ∃ m : ℕ, m ^ 2 = n.val ∧ 
   ∀ p : ℕ, Nat.Prime p → p < 70 → ¬ p ∣ n.val) ∧
  (∀ k : ℕ+, k.val < 5183 → 
    Nat.Prime k.val ∨ 
    (∃ m : ℕ, m ^ 2 = k.val) ∨ 
    (∃ p : ℕ, Nat.Prime p ∧ p < 70 ∧ p ∣ k.val)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_with_large_factors_l1294_129438


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1294_129435

theorem complex_equation_solution :
  ∃ (x : ℂ), 5 - 3 * Complex.I * x = -2 + 6 * Complex.I * x ∧ x = -7 * Complex.I / 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1294_129435


namespace NUMINAMATH_CALUDE_average_transformation_l1294_129413

theorem average_transformation (x₁ x₂ x₃ x₄ : ℝ) 
  (h : (x₁ + x₂ + x₃ + x₄) / 4 = 3) :
  ((2*x₁ - 3) + (2*x₂ - 3) + (2*x₃ - 3) + (2*x₄ - 3)) / 4 = 3 := by
sorry

end NUMINAMATH_CALUDE_average_transformation_l1294_129413


namespace NUMINAMATH_CALUDE_sunday_rounds_count_l1294_129478

/-- Proves the number of rounds completed on Sunday given the conditions of the problem -/
theorem sunday_rounds_count (round_time : ℕ) (saturday_rounds : ℕ) (total_time : ℕ) : 
  round_time = 30 →
  saturday_rounds = 11 →
  total_time = 780 →
  (total_time - saturday_rounds * round_time) / round_time = 15 := by
  sorry


end NUMINAMATH_CALUDE_sunday_rounds_count_l1294_129478


namespace NUMINAMATH_CALUDE_vasya_has_winning_strategy_l1294_129405

/-- Represents the state of the game board --/
structure GameState where
  piles : List Nat
  deriving Repr

/-- Represents a player in the game --/
inductive Player
  | Petya
  | Vasya
  deriving Repr

/-- Represents a move in the game --/
structure Move where
  pileIndices : List Nat
  stonesToRemove : Nat
  deriving Repr

/-- Checks if a move is valid for a given player and game state --/
def isValidMove (player : Player) (state : GameState) (move : Move) : Bool :=
  match player with
  | Player.Petya => move.pileIndices.length == 1 && move.stonesToRemove ≤ 3
  | Player.Vasya => move.pileIndices.length == move.stonesToRemove && move.stonesToRemove ≤ 3

/-- Applies a move to the game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is over --/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- The main theorem stating that Vasya has a winning strategy --/
theorem vasya_has_winning_strategy :
  ∃ (strategy : GameState → Move),
    ∀ (initialState : GameState),
      initialState.piles.length == 11 →
      (∀ pile ∈ initialState.piles, pile == 10) →
      ∀ (petyaStrategy : GameState → Move),
        isValidMove Player.Petya initialState (petyaStrategy initialState) →
        (∀ state : GameState, isValidMove Player.Petya state (petyaStrategy state)) →
        isValidMove Player.Vasya (applyMove initialState (petyaStrategy initialState)) (strategy (applyMove initialState (petyaStrategy initialState))) →
        (∀ state : GameState, isValidMove Player.Vasya state (strategy state)) →
        ∃ (finalState : GameState),
          isGameOver finalState ∧
          (finalState.piles.all (· == 0) ∨ ¬isValidMove Player.Petya finalState (petyaStrategy finalState)) :=
  sorry

end NUMINAMATH_CALUDE_vasya_has_winning_strategy_l1294_129405


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l1294_129470

/-- Given positive integers c and d where 58 in base c equals 85 in base d,
    the least possible sum of c and d is 15. -/
theorem least_sum_of_bases (c d : ℕ) (hc : c > 0) (hd : d > 0)
  (h_eq : 5 * c + 8 = 8 * d + 5) : 
  (∀ c' d' : ℕ, c' > 0 → d' > 0 → 5 * c' + 8 = 8 * d' + 5 → c' + d' ≥ c + d) ∧ c + d = 15 := by
  sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l1294_129470


namespace NUMINAMATH_CALUDE_circle_center_x_coordinate_l1294_129491

theorem circle_center_x_coordinate (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - a*x = 0 → (x - 1)^2 + y^2 = 1) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_x_coordinate_l1294_129491


namespace NUMINAMATH_CALUDE_composition_of_odd_is_odd_l1294_129407

-- Define an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem composition_of_odd_is_odd (f : ℝ → ℝ) (h : is_odd_function f) :
  is_odd_function (f ∘ f) :=
by
  sorry


end NUMINAMATH_CALUDE_composition_of_odd_is_odd_l1294_129407


namespace NUMINAMATH_CALUDE_average_minus_tenth_l1294_129466

theorem average_minus_tenth (x : ℚ) : x = (1/8 + 1/3) / 2 - 1/10 → x = 31/240 := by
  sorry

end NUMINAMATH_CALUDE_average_minus_tenth_l1294_129466


namespace NUMINAMATH_CALUDE_water_height_in_cylinder_l1294_129423

/-- The height of water in a cylinder when poured from an inverted cone -/
theorem water_height_in_cylinder (cone_radius cone_height cylinder_radius : ℝ) 
  (h_cone_radius : cone_radius = 10)
  (h_cone_height : cone_height = 15)
  (h_cylinder_radius : cylinder_radius = 20) :
  (1 / 3 * π * cone_radius^2 * cone_height) / (π * cylinder_radius^2) = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_water_height_in_cylinder_l1294_129423


namespace NUMINAMATH_CALUDE_negative_two_and_sqrt_four_are_opposite_l1294_129481

-- Define opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- State the theorem
theorem negative_two_and_sqrt_four_are_opposite : 
  are_opposite (-2 : ℝ) (Real.sqrt ((-2 : ℝ)^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_negative_two_and_sqrt_four_are_opposite_l1294_129481


namespace NUMINAMATH_CALUDE_problem_solution_l1294_129457

theorem problem_solution (a b : ℚ) 
  (h1 : 3 * a + 4 * b = 0) 
  (h2 : a = 2 * b - 3) : 
  9 * a - 6 * b = -81 / 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1294_129457


namespace NUMINAMATH_CALUDE_power_mod_500_l1294_129411

theorem power_mod_500 : 7^(7^(5^2)) ≡ 43 [ZMOD 500] := by sorry

end NUMINAMATH_CALUDE_power_mod_500_l1294_129411


namespace NUMINAMATH_CALUDE_product_plus_five_l1294_129402

theorem product_plus_five : (-11 * -8) + 5 = 93 := by
  sorry

end NUMINAMATH_CALUDE_product_plus_five_l1294_129402


namespace NUMINAMATH_CALUDE_probability_of_matching_pair_l1294_129437

def num_blue_socks : ℕ := 12
def num_green_socks : ℕ := 10

def total_socks : ℕ := num_blue_socks + num_green_socks

def ways_to_pick_two (n : ℕ) : ℕ := n * (n - 1) / 2

def matching_pairs : ℕ := ways_to_pick_two num_blue_socks + ways_to_pick_two num_green_socks

def total_ways : ℕ := ways_to_pick_two total_socks

theorem probability_of_matching_pair :
  (matching_pairs : ℚ) / total_ways = 111 / 231 := by sorry

end NUMINAMATH_CALUDE_probability_of_matching_pair_l1294_129437


namespace NUMINAMATH_CALUDE_g_of_quadratic_l1294_129486

/-- Given that g(x) = 3x + 1 for all real numbers x, 
    prove that g(x^2 + 2x + 2) = 3x^2 + 6x + 7 -/
theorem g_of_quadratic (x : ℝ) : 
  let g : ℝ → ℝ := fun x ↦ 3 * x + 1
  g (x^2 + 2*x + 2) = 3*x^2 + 6*x + 7 := by
  sorry

end NUMINAMATH_CALUDE_g_of_quadratic_l1294_129486


namespace NUMINAMATH_CALUDE_anya_hair_growth_l1294_129415

/-- Calculates the additional hairs Anya needs to grow in a week -/
def additional_hairs_needed (washes_per_week : ℕ) (hairs_lost_per_wash : ℕ) 
  (brushings_per_week : ℕ) (growth_rate : ℕ) (growth_period : ℕ) : ℕ :=
  let hairs_lost_washing := washes_per_week * hairs_lost_per_wash
  let hairs_lost_brushing := brushings_per_week * (hairs_lost_per_wash / 2)
  let total_hair_loss := hairs_lost_washing + hairs_lost_brushing
  let growth_periods := 7 / growth_period
  let total_hair_growth := growth_periods * growth_rate
  total_hair_loss - total_hair_growth + 1

/-- Theorem stating that Anya needs to grow 63 additional hairs -/
theorem anya_hair_growth : 
  additional_hairs_needed 5 32 7 70 2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_anya_hair_growth_l1294_129415


namespace NUMINAMATH_CALUDE_polynomial_roots_nature_l1294_129474

theorem polynomial_roots_nature :
  ∃ (r₁ r₂ r₃ : ℝ), 
    (r₁ > 0 ∧ r₂ < 0 ∧ r₃ < 0) ∧
    (∀ x : ℝ, x^3 - 7*x^2 + 14*x - 8 = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_nature_l1294_129474


namespace NUMINAMATH_CALUDE_yann_and_camille_combinations_l1294_129464

/-- The number of items on the menu -/
def menu_items : ℕ := 12

/-- The number of people ordering -/
def num_people : ℕ := 2

/-- Calculates the number of different meal combinations for two people
    when the first person's choice is not available for the second person -/
def meal_combinations (items : ℕ) : ℕ := items * (items - 1)

/-- Theorem stating that the number of different meal combinations
    for Yann and Camille is 132 -/
theorem yann_and_camille_combinations :
  meal_combinations menu_items = 132 := by sorry

end NUMINAMATH_CALUDE_yann_and_camille_combinations_l1294_129464


namespace NUMINAMATH_CALUDE_gamma_donuts_l1294_129408

/-- Given a total of 40 donuts, with Delta taking 8 and Beta taking three times as many as Gamma,
    prove that Gamma received 8 donuts. -/
theorem gamma_donuts (total : ℕ) (delta : ℕ) (beta : ℕ) (gamma : ℕ) : 
  total = 40 → 
  delta = 8 → 
  beta = 3 * gamma → 
  total = delta + beta + gamma → 
  gamma = 8 := by
sorry

end NUMINAMATH_CALUDE_gamma_donuts_l1294_129408


namespace NUMINAMATH_CALUDE_total_flowers_l1294_129480

/-- Represents the number of flowers of each color in the garden -/
structure FlowerCounts where
  orange : ℕ
  red : ℕ
  yellow : ℕ
  pink : ℕ
  purple : ℕ

/-- The conditions of the flower garden problem -/
def garden_conditions (f : FlowerCounts) : Prop :=
  f.red = 2 * f.orange ∧
  f.yellow = f.red - 5 ∧
  f.orange = 10 ∧
  f.pink = f.purple ∧
  f.pink + f.purple = 30

/-- The theorem stating the total number of flowers in the garden -/
theorem total_flowers (f : FlowerCounts) 
  (h : garden_conditions f) : 
  f.orange + f.red + f.yellow + f.pink + f.purple = 75 := by
  sorry

#check total_flowers

end NUMINAMATH_CALUDE_total_flowers_l1294_129480


namespace NUMINAMATH_CALUDE_main_diagonal_contains_all_numbers_l1294_129488

/-- A square matrix of odd size n, where each row and column contains all numbers from 1 to n,
    and the matrix is symmetric with respect to the main diagonal. -/
structure SymmetricLatinSquare (n : ℕ) :=
  (matrix : Fin n → Fin n → Fin n)
  (odd : Odd n)
  (latin_row : ∀ i j : Fin n, ∃ k : Fin n, matrix i k = j)
  (latin_col : ∀ i j : Fin n, ∃ k : Fin n, matrix k i = j)
  (symmetric : ∀ i j : Fin n, matrix i j = matrix j i)

/-- Theorem stating that all numbers from 1 to n appear on the main diagonal of a SymmetricLatinSquare. -/
theorem main_diagonal_contains_all_numbers (n : ℕ) (sls : SymmetricLatinSquare n) :
  ∀ k : Fin n, ∃ i : Fin n, sls.matrix i i = k := by
  sorry

end NUMINAMATH_CALUDE_main_diagonal_contains_all_numbers_l1294_129488


namespace NUMINAMATH_CALUDE_curve_is_circle_implies_a_eq_neg_one_l1294_129421

/-- A curve is a circle if and only if its equation can be written in the form
    (x - h)^2 + (y - k)^2 = r^2, where (h, k) is the center and r is the radius. -/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ h k r, r > 0 ∧ ∀ x y, f x y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The equation of the curve -/
def curve_equation (a : ℝ) (x y : ℝ) : ℝ :=
  a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a

/-- Theorem: If the curve represented by a^2x^2 + (a+2)y^2 + 2ax + a = 0 is a circle, then a = -1 -/
theorem curve_is_circle_implies_a_eq_neg_one (a : ℝ) :
  is_circle (curve_equation a) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_curve_is_circle_implies_a_eq_neg_one_l1294_129421


namespace NUMINAMATH_CALUDE_cosine_equation_solution_l1294_129476

theorem cosine_equation_solution :
  ∃! x : ℝ, 0 < x ∧ x < π ∧ 2 * Real.cos (x - π/4) = 1 :=
by
  use 7*π/12
  sorry

end NUMINAMATH_CALUDE_cosine_equation_solution_l1294_129476


namespace NUMINAMATH_CALUDE_function_with_two_zeros_properties_l1294_129403

/-- A function f(x) = ax² - e^x with two positive zeros -/
structure FunctionWithTwoZeros where
  a : ℝ
  x₁ : ℝ
  x₂ : ℝ
  h₁ : 0 < x₁
  h₂ : x₁ < x₂
  h₃ : a * x₁^2 = Real.exp x₁
  h₄ : a * x₂^2 = Real.exp x₂

/-- The range of a and the sum of zeros for a function with two positive zeros -/
theorem function_with_two_zeros_properties (f : FunctionWithTwoZeros) :
  f.a > Real.exp 2 / 4 ∧ f.x₁ + f.x₂ > 4 := by
  sorry

end NUMINAMATH_CALUDE_function_with_two_zeros_properties_l1294_129403


namespace NUMINAMATH_CALUDE_median_of_special_sequence_l1294_129447

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem median_of_special_sequence :
  let N : ℕ := sequence_sum 150
  let median_position : ℕ := (N + 1) / 2
  ∃ (k : ℕ), k = 106 ∧ 
    sequence_sum (k - 1) < median_position ∧ 
    median_position ≤ sequence_sum k :=
by sorry

end NUMINAMATH_CALUDE_median_of_special_sequence_l1294_129447


namespace NUMINAMATH_CALUDE_intersection_condition_union_condition_l1294_129484

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a-1)*x + (a^2-5) = 0}

-- Part 1: When A ∩ B = {2}, find the value of a
theorem intersection_condition (a : ℝ) : A ∩ B a = {2} → a = -5 ∨ a = 1 := by
  sorry

-- Part 2: When A ∪ B = A, find the range of a
theorem union_condition (a : ℝ) : A ∪ B a = A → a > 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_union_condition_l1294_129484


namespace NUMINAMATH_CALUDE_sandwich_count_l1294_129417

/-- Represents the number of items bought in a week -/
structure WeeklyPurchase where
  sandwiches : ℕ
  cookies : ℕ
  total_items : sandwiches + cookies = 7

/-- Represents the cost in cents -/
def cost (p : WeeklyPurchase) : ℕ := 60 * p.sandwiches + 90 * p.cookies

theorem sandwich_count : 
  ∃ (p : WeeklyPurchase), 
    500 ≤ cost p ∧ 
    cost p ≤ 700 ∧ 
    cost p % 100 = 0 ∧
    p.sandwiches = 11 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_count_l1294_129417


namespace NUMINAMATH_CALUDE_fraction_simplification_l1294_129465

theorem fraction_simplification :
  (3 : ℚ) / (2 - 4 / 5) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1294_129465


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_roots_min_value_achieved_l1294_129412

theorem min_value_of_sum_of_roots (x : ℝ) :
  Real.sqrt (x^2 + 2*x + 2) + Real.sqrt (x^2 - 2*x + 2) ≥ 2 * Real.sqrt 2 :=
by sorry

theorem min_value_achieved :
  ∃ x : ℝ, Real.sqrt (x^2 + 2*x + 2) + Real.sqrt (x^2 - 2*x + 2) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_roots_min_value_achieved_l1294_129412


namespace NUMINAMATH_CALUDE_max_moves_l1294_129468

def S : Set (ℕ × ℕ) := {p | p.1 ≥ 1 ∧ p.1 ≤ 2022 ∧ p.2 ≥ 1 ∧ p.2 ≤ 2022}

def isGoodRectangle (r : (ℕ × ℕ) × (ℕ × ℕ)) : Prop :=
  let ((x1, y1), (x2, y2)) := r
  x1 ≥ 1 ∧ x1 ≤ 2022 ∧ y1 ≥ 1 ∧ y1 ≤ 2022 ∧
  x2 ≥ 1 ∧ x2 ≤ 2022 ∧ y2 ≥ 1 ∧ y2 ≤ 2022 ∧
  x1 < x2 ∧ y1 < y2

def Move := (ℕ × ℕ) × (ℕ × ℕ)

def isValidMove (m : Move) : Prop := isGoodRectangle m

theorem max_moves : ∃ (moves : List Move), 
  (∀ m ∈ moves, isValidMove m) ∧ 
  moves.length = 1011^4 ∧ 
  (∀ (other_moves : List Move), (∀ m ∈ other_moves, isValidMove m) → other_moves.length ≤ 1011^4) := by
  sorry

end NUMINAMATH_CALUDE_max_moves_l1294_129468


namespace NUMINAMATH_CALUDE_homework_assignments_for_28_points_l1294_129431

/-- Calculates the number of assignments required for a given point in the homework system -/
def assignmentsForPoint (n : ℕ) : ℕ := (n - 1) / 7 + 1

/-- Calculates the total number of assignments required for a given number of points -/
def totalAssignments (points : ℕ) : ℕ := 
  Finset.sum (Finset.range points) (λ i => assignmentsForPoint (i + 1))

/-- Proves that 28 homework points require 70 assignments -/
theorem homework_assignments_for_28_points : totalAssignments 28 = 70 := by
  sorry

end NUMINAMATH_CALUDE_homework_assignments_for_28_points_l1294_129431


namespace NUMINAMATH_CALUDE_modular_sum_of_inverses_l1294_129425

theorem modular_sum_of_inverses (p : ℕ) (h_prime : Nat.Prime p) (h_p : p = 31) :
  ∃ (a b : ℕ), a < p ∧ b < p ∧
  (5 * a) % p = 1 ∧
  (25 * b) % p = 1 ∧
  (a + b) % p = 26 := by
sorry

end NUMINAMATH_CALUDE_modular_sum_of_inverses_l1294_129425


namespace NUMINAMATH_CALUDE_playground_snow_volume_l1294_129452

/-- Represents a rectangular playground covered in snow -/
structure SnowCoveredPlayground where
  length : ℝ
  width : ℝ
  snowDepth : ℝ

/-- Calculates the volume of snow on a rectangular playground -/
def snowVolume (p : SnowCoveredPlayground) : ℝ :=
  p.length * p.width * p.snowDepth

/-- Theorem stating that the volume of snow on the given playground is 50 cubic feet -/
theorem playground_snow_volume :
  let p : SnowCoveredPlayground := {
    length := 40,
    width := 5,
    snowDepth := 0.25
  }
  snowVolume p = 50 := by sorry

end NUMINAMATH_CALUDE_playground_snow_volume_l1294_129452


namespace NUMINAMATH_CALUDE_american_car_production_l1294_129472

/-- The number of cars American carmakers produce each year -/
def total_cars : ℕ := 5650000

/-- The number of car suppliers -/
def num_suppliers : ℕ := 5

/-- The number of cars the first supplier receives -/
def first_supplier : ℕ := 1000000

/-- The number of cars the second supplier receives -/
def second_supplier : ℕ := first_supplier + 500000

/-- The number of cars the third supplier receives -/
def third_supplier : ℕ := first_supplier + second_supplier

/-- The number of cars the fourth supplier receives -/
def fourth_supplier : ℕ := 325000

/-- The number of cars the fifth supplier receives -/
def fifth_supplier : ℕ := fourth_supplier

theorem american_car_production :
  total_cars = first_supplier + second_supplier + third_supplier + fourth_supplier + fifth_supplier :=
by sorry

end NUMINAMATH_CALUDE_american_car_production_l1294_129472


namespace NUMINAMATH_CALUDE_vending_machine_problem_l1294_129445

/-- The probability of the vending machine failing to drop a snack -/
def fail_prob : ℚ := 1 / 6

/-- The probability of the vending machine dropping two snacks -/
def double_prob : ℚ := 1 / 10

/-- The probability of the vending machine dropping one snack -/
def single_prob : ℚ := 1 - fail_prob - double_prob

/-- The expected number of snacks dropped per person -/
def expected_snacks : ℚ := fail_prob * 0 + double_prob * 2 + single_prob * 1

/-- The total number of snacks dropped -/
def total_snacks : ℕ := 28

/-- The number of people who have used the vending machine -/
def num_people : ℕ := 30

theorem vending_machine_problem :
  (↑num_people : ℚ) * expected_snacks = ↑total_snacks :=
sorry

end NUMINAMATH_CALUDE_vending_machine_problem_l1294_129445


namespace NUMINAMATH_CALUDE_trisection_intersection_l1294_129409

noncomputable section

def f (x : ℝ) : ℝ := Real.log (x^2)

theorem trisection_intersection (A B C D E F : ℝ × ℝ) :
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := E
  let (x₄, y₄) := F
  0 < x₁ → x₁ < x₂ →
  x₁ = 1 → x₂ = 8 →
  y₁ = f x₁ → y₂ = f x₂ →
  C.2 = (2/3 : ℝ) * y₁ + (1/3 : ℝ) * y₂ →
  D.2 = (1/3 : ℝ) * y₁ + (2/3 : ℝ) * y₂ →
  y₃ = f x₃ → y₃ = C.2 →
  y₄ = f x₄ → y₄ = D.2 →
  x₃ = 4 ∧ x₄ = 16 := by
  sorry

end

end NUMINAMATH_CALUDE_trisection_intersection_l1294_129409


namespace NUMINAMATH_CALUDE_rectangle_division_l1294_129424

/-- Given a rectangle with vertices (x, 0), (9, 0), (x, 2), and (9, 2),
    if a line passing through the origin with slope 0.2 divides the rectangle
    into two identical quadrilaterals, then x = 1. -/
theorem rectangle_division (x : ℝ) : 
  (∃ (l : Set (ℝ × ℝ)), 
    -- l is a line passing through the origin
    (0, 0) ∈ l ∧
    -- l has slope 0.2
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = 0.2) ∧
    -- l divides the rectangle into two identical quadrilaterals
    (∃ (m : ℝ × ℝ), m ∈ l ∧ m.1 = (x + 9) / 2 ∧ m.2 = 1)) →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_rectangle_division_l1294_129424


namespace NUMINAMATH_CALUDE_broken_sign_distance_l1294_129422

/-- Represents a town in the problem -/
inductive Town
| Atown
| Betown
| Cetown

/-- Represents a signpost along the path -/
structure Signpost where
  distanceToAtown : ℕ
  distanceToCetown : ℕ

/-- The problem setup -/
axiom signpostA : Signpost
axiom signpostB : Signpost
axiom path_through_Betown : True
axiom signpostA_distances : signpostA.distanceToAtown = 7 ∧ signpostA.distanceToCetown = 2
axiom signpostB_distances : signpostB.distanceToAtown = 9 ∧ signpostB.distanceToCetown = 4

/-- The theorem to be proved -/
theorem broken_sign_distance : ∃ (d : ℕ), d = 1 ∧ 
  (d = signpostA.distanceToAtown - signpostB.distanceToAtown + signpostB.distanceToCetown) :=
sorry

end NUMINAMATH_CALUDE_broken_sign_distance_l1294_129422


namespace NUMINAMATH_CALUDE_local_road_speed_l1294_129482

/-- Proves that the speed on local roads is 30 mph given the specified conditions --/
theorem local_road_speed (local_distance : ℝ) (highway_distance : ℝ) 
  (highway_speed : ℝ) (average_speed : ℝ) (v : ℝ) : 
  local_distance = 90 →
  highway_distance = 75 →
  highway_speed = 60 →
  average_speed = 38.82 →
  (local_distance + highway_distance) / average_speed = local_distance / v + highway_distance / highway_speed →
  v = 30 := by
  sorry

end NUMINAMATH_CALUDE_local_road_speed_l1294_129482


namespace NUMINAMATH_CALUDE_complex_modulus_l1294_129400

theorem complex_modulus (z : ℂ) : z = Complex.mk (Real.sin (π / 3)) (-Real.cos (π / 6)) → Complex.abs z = Real.sqrt (3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1294_129400


namespace NUMINAMATH_CALUDE_total_notes_count_l1294_129463

/-- Given a total amount of 400 rupees in equal numbers of one-rupee, five-rupee, and ten-rupee notes,
    prove that the total number of notes is 75. -/
theorem total_notes_count (total_amount : ℕ) (n : ℕ) 
  (h1 : total_amount = 400)
  (h2 : n * 1 + n * 5 + n * 10 = total_amount) : 
  3 * n = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_notes_count_l1294_129463


namespace NUMINAMATH_CALUDE_band_formation_max_members_l1294_129487

theorem band_formation_max_members :
  ∀ (r x : ℕ),
    r * x + 2 < 100 →
    (r - 2) * (x + 1) = r * x + 2 →
    ∀ (m : ℕ),
      m < 100 →
      ∃ (r' x' : ℕ),
        r' * x' + 2 = m →
        (r' - 2) * (x' + 1) = m →
        m ≤ 98 :=
by sorry

end NUMINAMATH_CALUDE_band_formation_max_members_l1294_129487


namespace NUMINAMATH_CALUDE_seed_mixture_problem_l1294_129441

/-- Proves that in a mixture of seed mixtures X and Y, where X is 40% ryegrass
    and Y is 25% ryegrass, if the final mixture contains 35% ryegrass,
    then the percentage of X in the final mixture is 200/3. -/
theorem seed_mixture_problem (x y : ℝ) :
  x + y = 100 →  -- x and y represent percentages of X and Y in the final mixture
  0.40 * x + 0.25 * y = 35 →  -- The final mixture contains 35% ryegrass
  x = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_seed_mixture_problem_l1294_129441


namespace NUMINAMATH_CALUDE_exists_negative_monomial_degree_5_l1294_129456

/-- A monomial in x and y -/
structure Monomial where
  coeff : ℤ
  x_power : ℕ
  y_power : ℕ

/-- The degree of a monomial -/
def Monomial.degree (m : Monomial) : ℕ := m.x_power + m.y_power

/-- A monomial is negative if its coefficient is negative -/
def Monomial.isNegative (m : Monomial) : Prop := m.coeff < 0

theorem exists_negative_monomial_degree_5 :
  ∃ m : Monomial, m.isNegative ∧ m.degree = 5 :=
sorry

end NUMINAMATH_CALUDE_exists_negative_monomial_degree_5_l1294_129456


namespace NUMINAMATH_CALUDE_delta_u_zero_iff_k_ge_five_l1294_129458

def u (n : ℕ) : ℕ := n^4 + n^2

def Δ : (ℕ → ℕ) → (ℕ → ℕ)
  | f => fun n => f (n + 1) - f n

def iterateΔ : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
  | 0 => id
  | k + 1 => Δ ∘ iterateΔ k

theorem delta_u_zero_iff_k_ge_five :
  ∀ k : ℕ, (∀ n : ℕ, iterateΔ k u n = 0) ↔ k ≥ 5 := by sorry

end NUMINAMATH_CALUDE_delta_u_zero_iff_k_ge_five_l1294_129458


namespace NUMINAMATH_CALUDE_missing_edge_length_l1294_129495

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.edge1 * d.edge2 * d.edge3

/-- Theorem: Given a cuboid with two known edges 5 cm and 8 cm, and a volume of 80 cm³,
    the length of the third edge is 2 cm -/
theorem missing_edge_length :
  ∀ (x : ℝ),
    let d := CuboidDimensions.mk x 5 8
    cuboidVolume d = 80 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_missing_edge_length_l1294_129495


namespace NUMINAMATH_CALUDE_age_ratio_problem_l1294_129444

theorem age_ratio_problem (a b : ℕ) : 
  a + b = 60 →                 -- Sum of present ages is 60
  ∃ k : ℕ, a = k * b →         -- A's age is some multiple of B's age
  a + b + 6 = 66 →             -- Sum of ages 3 years hence is 66
  a = 60 ∧ b = 5 :=            -- Implies A's age is 60 and B's age is 5
by sorry

-- The ratio can be derived from a = 60 and b = 5

end NUMINAMATH_CALUDE_age_ratio_problem_l1294_129444


namespace NUMINAMATH_CALUDE_camera_filter_savings_percentage_l1294_129440

theorem camera_filter_savings_percentage : 
  let kit_price : ℚ := 144.20
  let filter_prices : List ℚ := [21.75, 21.75, 18.60, 18.60, 23.80, 29.35, 29.35]
  let total_individual_price : ℚ := filter_prices.sum
  let savings : ℚ := total_individual_price - kit_price
  let savings_percentage : ℚ := (savings / total_individual_price) * 100
  savings_percentage = 11.64 := by sorry

end NUMINAMATH_CALUDE_camera_filter_savings_percentage_l1294_129440


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l1294_129436

theorem greatest_integer_inequality : ∃ (y : ℤ), (5 : ℚ) / 8 > (y : ℚ) / 17 ∧ 
  ∀ (z : ℤ), (5 : ℚ) / 8 > (z : ℚ) / 17 → z ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l1294_129436


namespace NUMINAMATH_CALUDE_parabola_vertex_l1294_129462

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 8*y + 3*x + 7 = 0

-- Define the vertex of a parabola
def is_vertex (x y : ℝ) : Prop :=
  ∀ t : ℝ, parabola_equation (x + t) y → t = 0

-- Theorem statement
theorem parabola_vertex :
  is_vertex 3 (-4) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1294_129462


namespace NUMINAMATH_CALUDE_sequence_properties_l1294_129489

/-- An arithmetic sequence a_n with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ a 4 = 14 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- A sequence b_n with given conditions -/
def b_sequence (b : ℕ → ℝ) : Prop :=
  b 1 = 1 ∧ b 4 = 6

/-- The difference sequence a_n - b_n is geometric -/
def difference_is_geometric (a b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, (a (n + 1) - b (n + 1)) / (a n - b n) = r

/-- The existence of a maximum term in b_n -/
def b_has_max (b : ℕ → ℝ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n > 0 → b n ≤ b k

/-- The main theorem -/
theorem sequence_properties
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : b_sequence b)
  (hd : difference_is_geometric a b)
  (hm : b_has_max b) :
  (∀ n : ℕ, n > 0 → a n = 4 * n - 2) ∧
  (∀ n : ℕ, n > 0 → b n = 4 * n - 2 - 2^(n-1)) ∧
  (∃ k : ℕ, (k = 3 ∨ k = 4) ∧ ∀ n : ℕ, n > 0 → b n ≤ b k) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1294_129489


namespace NUMINAMATH_CALUDE_f_three_minus_f_neg_three_l1294_129427

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 3*x^3 + 4*x

-- State the theorem
theorem f_three_minus_f_neg_three : f 3 - f (-3) = 672 := by
  sorry

end NUMINAMATH_CALUDE_f_three_minus_f_neg_three_l1294_129427


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l1294_129499

theorem product_of_sum_and_sum_of_cubes (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (sum_cubes_eq : x^3 + y^3 = 370) : 
  x * y = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l1294_129499


namespace NUMINAMATH_CALUDE_optimal_selling_price_l1294_129429

/-- Represents the store's pricing problem --/
structure StorePricing where
  purchasePrice : ℝ
  demandSlope : ℝ
  demandIntercept : ℝ
  maxPriceIncrease : ℝ
  desiredProfit : ℝ

/-- Calculates the profit for a given selling price --/
def profit (sp : StorePricing) (sellingPrice : ℝ) : ℝ :=
  (sellingPrice - sp.purchasePrice) * (sp.demandIntercept - sp.demandSlope * sellingPrice)

/-- Checks if the selling price satisfies the government restriction --/
def satisfiesRestriction (sp : StorePricing) (sellingPrice : ℝ) : Prop :=
  (sellingPrice - sp.purchasePrice) / sp.purchasePrice ≤ sp.maxPriceIncrease

/-- Theorem stating that 41 yuan is the optimal selling price --/
theorem optimal_selling_price (sp : StorePricing) 
  (h_purchase : sp.purchasePrice = 30)
  (h_demand : sp.demandSlope = 2 ∧ sp.demandIntercept = 112)
  (h_restriction : sp.maxPriceIncrease = 0.4)
  (h_profit : sp.desiredProfit = 330) :
  ∃ (optimalPrice : ℝ), 
    optimalPrice = 41 ∧ 
    satisfiesRestriction sp optimalPrice ∧ 
    profit sp optimalPrice = sp.desiredProfit ∧
    ∀ (price : ℝ), satisfiesRestriction sp price → profit sp price ≤ profit sp optimalPrice :=
sorry


end NUMINAMATH_CALUDE_optimal_selling_price_l1294_129429


namespace NUMINAMATH_CALUDE_expression_evaluation_l1294_129498

theorem expression_evaluation : 
  1 - (1 / (1 + Real.sqrt 5)) + (1 / (1 - Real.sqrt 5)) = 1 - (Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1294_129498


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l1294_129471

theorem unique_quadratic_solution (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + a * x + 1 = 0) ↔ a = 4 :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l1294_129471


namespace NUMINAMATH_CALUDE_dividend_calculation_l1294_129477

theorem dividend_calculation (remainder quotient divisor : ℕ) 
  (h1 : remainder = 6)
  (h2 : divisor = 5 * quotient)
  (h3 : divisor = 3 * remainder + 2) :
  divisor * quotient + remainder = 86 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1294_129477


namespace NUMINAMATH_CALUDE_audiobook_purchase_l1294_129483

theorem audiobook_purchase (audiobook_length : ℕ) (daily_listening : ℕ) (total_days : ℕ) : 
  audiobook_length = 30 → 
  daily_listening = 2 → 
  total_days = 90 → 
  (total_days * daily_listening) / audiobook_length = 6 := by
sorry

end NUMINAMATH_CALUDE_audiobook_purchase_l1294_129483


namespace NUMINAMATH_CALUDE_lemonade_revenue_is_110_l1294_129451

/-- Calculates the total revenue from selling lemonade over three weeks -/
def lemonade_revenue : ℝ :=
  let first_week_cups : ℕ := 20
  let first_week_price : ℝ := 1
  let second_week_increase : ℝ := 0.5
  let second_week_price : ℝ := 1.25
  let third_week_increase : ℝ := 0.75
  let third_week_price : ℝ := 1.5

  let first_week_revenue := first_week_cups * first_week_price
  let second_week_revenue := (first_week_cups * (1 + second_week_increase)) * second_week_price
  let third_week_revenue := (first_week_cups * (1 + third_week_increase)) * third_week_price

  first_week_revenue + second_week_revenue + third_week_revenue

theorem lemonade_revenue_is_110 : lemonade_revenue = 110 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_revenue_is_110_l1294_129451


namespace NUMINAMATH_CALUDE_benny_march_savings_l1294_129418

/-- The amount of money Benny added to his piggy bank in January -/
def january_amount : ℕ := 19

/-- The amount of money Benny added to his piggy bank in February -/
def february_amount : ℕ := 19

/-- The total amount of money in Benny's piggy bank by the end of March -/
def march_total : ℕ := 46

/-- The amount of money Benny added to his piggy bank in March -/
def march_amount : ℕ := march_total - (january_amount + february_amount)

/-- Proof that Benny added $8 to his piggy bank in March -/
theorem benny_march_savings : march_amount = 8 := by
  sorry

end NUMINAMATH_CALUDE_benny_march_savings_l1294_129418


namespace NUMINAMATH_CALUDE_punger_baseball_cards_l1294_129430

/-- The number of packs of baseball cards Punger bought -/
def number_of_packs : ℕ := sorry

/-- The number of cards in each pack -/
def cards_per_pack : ℕ := 7

/-- The number of cards each page can hold -/
def cards_per_page : ℕ := 10

/-- The number of pages Punger needs -/
def number_of_pages : ℕ := 42

theorem punger_baseball_cards : number_of_packs = 60 := by sorry

end NUMINAMATH_CALUDE_punger_baseball_cards_l1294_129430


namespace NUMINAMATH_CALUDE_walkway_area_is_116_l1294_129453

/-- Represents the garden layout --/
structure GardenLayout where
  bed_width : ℕ := 8
  bed_height : ℕ := 3
  beds_per_row : ℕ := 2
  num_rows : ℕ := 3
  walkway_width : ℕ := 1
  has_central_walkway : Bool := true

/-- Calculates the total area of walkways in the garden --/
def walkway_area (garden : GardenLayout) : ℕ :=
  let total_width := garden.bed_width * garden.beds_per_row + 
                     (garden.beds_per_row + 1) * garden.walkway_width + 
                     (if garden.has_central_walkway then garden.walkway_width else 0)
  let total_height := garden.bed_height * garden.num_rows + 
                      (garden.num_rows + 1) * garden.walkway_width
  let total_area := total_width * total_height
  let beds_area := garden.bed_width * garden.bed_height * garden.beds_per_row * garden.num_rows
  total_area - beds_area

/-- Theorem stating that the walkway area for the given garden layout is 116 square feet --/
theorem walkway_area_is_116 (garden : GardenLayout) : walkway_area garden = 116 := by
  sorry

end NUMINAMATH_CALUDE_walkway_area_is_116_l1294_129453


namespace NUMINAMATH_CALUDE_average_temperature_last_four_days_l1294_129454

/-- Given the temperatures for a week, prove the average temperature for the last four days. -/
theorem average_temperature_last_four_days 
  (temp_mon : ℝ)
  (temp_tue : ℝ)
  (temp_wed : ℝ)
  (temp_thu : ℝ)
  (temp_fri : ℝ)
  (h1 : (temp_mon + temp_tue + temp_wed + temp_thu) / 4 = 48)
  (h2 : temp_mon = 42)
  (h3 : temp_fri = 34) :
  (temp_tue + temp_wed + temp_thu + temp_fri) / 4 = 46 := by
sorry

end NUMINAMATH_CALUDE_average_temperature_last_four_days_l1294_129454


namespace NUMINAMATH_CALUDE_power_of_product_l1294_129433

theorem power_of_product (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1294_129433


namespace NUMINAMATH_CALUDE_triangle_acute_angles_l1294_129432

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : (angles 0) + (angles 1) + (angles 2) = 180
  positive : ∀ i, 0 < angles i

-- Define an acute angle
def is_acute (angle : ℝ) : Prop := 0 < angle ∧ angle < 90

-- Define an exterior angle
def exterior_angle (t : Triangle) (i : Fin 3) : ℝ := 180 - t.angles i

-- Theorem statement
theorem triangle_acute_angles (t : Triangle) : 
  (∃ i j : Fin 3, i ≠ j ∧ is_acute (t.angles i) ∧ is_acute (t.angles j)) ∧ 
  (∀ i j k : Fin 3, i ≠ j → j ≠ k → i ≠ k → 
    ¬(is_acute (exterior_angle t i) ∧ is_acute (exterior_angle t j))) :=
sorry

end NUMINAMATH_CALUDE_triangle_acute_angles_l1294_129432


namespace NUMINAMATH_CALUDE_fudge_ratio_is_one_to_three_l1294_129469

-- Define the amount of fudge eaten by each person in ounces
def tomas_fudge : ℚ := 1.5 * 16
def boris_fudge : ℚ := 2 * 16
def total_fudge : ℚ := 64

-- Define Katya's fudge as the remaining amount
def katya_fudge : ℚ := total_fudge - tomas_fudge - boris_fudge

-- Define the ratio of Katya's fudge to Tomas's fudge
def fudge_ratio : ℚ × ℚ := (katya_fudge, tomas_fudge)

-- Theorem to prove
theorem fudge_ratio_is_one_to_three : fudge_ratio = (1, 3) := by
  sorry

end NUMINAMATH_CALUDE_fudge_ratio_is_one_to_three_l1294_129469


namespace NUMINAMATH_CALUDE_game_eventually_ends_l1294_129455

/-- The game state after k rounds of questioning -/
structure GameState where
  a : ℕ+  -- Player A's number
  b : ℕ+  -- Player B's number
  x : ℕ+  -- Smaller number on the board
  y : ℕ+  -- Larger number on the board
  k : ℕ   -- Number of rounds of questioning

/-- The game's rules and conditions -/
def validGame (g : GameState) : Prop :=
  g.x < g.y ∧ (g.a + g.b = g.x ∨ g.a + g.b = g.y)

/-- The condition for the game to end (one player knows the other's number) -/
def gameEnds (g : GameState) : Prop :=
  (g.k % 2 = 0 → g.y - g.x * (g.k + 1) < g.b) ∧
  (g.k % 2 = 1 → g.y - g.x * (g.k + 1) < g.a)

/-- The main theorem: the game will eventually end -/
theorem game_eventually_ends :
  ∀ g : GameState, validGame g → ∃ n : ℕ, gameEnds {a := g.a, b := g.b, x := g.x, y := g.y, k := n} :=
sorry

end NUMINAMATH_CALUDE_game_eventually_ends_l1294_129455


namespace NUMINAMATH_CALUDE_firm_partners_count_l1294_129473

theorem firm_partners_count :
  ∀ (partners associates : ℕ),
  (partners : ℚ) / associates = 2 / 63 →
  partners / (associates + 50) = 1 / 34 →
  partners = 20 := by
sorry

end NUMINAMATH_CALUDE_firm_partners_count_l1294_129473


namespace NUMINAMATH_CALUDE_davids_english_marks_l1294_129401

theorem davids_english_marks :
  let math_marks : ℕ := 65
  let physics_marks : ℕ := 82
  let chemistry_marks : ℕ := 67
  let biology_marks : ℕ := 85
  let average_marks : ℕ := 78
  let num_subjects : ℕ := 5
  let total_marks : ℕ := average_marks * num_subjects
  let known_marks : ℕ := math_marks + physics_marks + chemistry_marks + biology_marks
  let english_marks : ℕ := total_marks - known_marks
  english_marks = 91 := by
sorry

end NUMINAMATH_CALUDE_davids_english_marks_l1294_129401


namespace NUMINAMATH_CALUDE_exam_failure_count_l1294_129439

/-- Given an examination where 740 students appeared and 35% passed,
    prove that 481 students failed the examination. -/
theorem exam_failure_count : ∀ (total_students : ℕ) (pass_percentage : ℚ),
  total_students = 740 →
  pass_percentage = 35 / 100 →
  (total_students : ℚ) * (1 - pass_percentage) = 481 := by
  sorry

end NUMINAMATH_CALUDE_exam_failure_count_l1294_129439


namespace NUMINAMATH_CALUDE_triangle_angle_c_l1294_129448

theorem triangle_angle_c (A B : ℝ) (hA : 3 * Real.sin A + 4 * Real.cos B = 6) 
  (hB : 4 * Real.sin B + 3 * Real.cos A = 1) : 
  ∃ C : ℝ, C = π / 6 ∧ A + B + C = π := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l1294_129448


namespace NUMINAMATH_CALUDE_problem_solution_l1294_129450

theorem problem_solution : 
  |(-5)| - 2 * 3^0 + Real.tan (π/4) + Real.sqrt 9 = 8 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1294_129450


namespace NUMINAMATH_CALUDE_michelle_taxi_cost_l1294_129485

/-- Calculates the total cost of a taxi ride given the initial fee, distance, and per-mile charge. -/
def taxi_cost (initial_fee : ℝ) (distance : ℝ) (per_mile_charge : ℝ) : ℝ :=
  initial_fee + distance * per_mile_charge

/-- Theorem stating that for the given conditions, the total cost of Michelle's taxi ride is $12. -/
theorem michelle_taxi_cost : taxi_cost 2 4 2.5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_michelle_taxi_cost_l1294_129485


namespace NUMINAMATH_CALUDE_students_with_A_grade_l1294_129467

theorem students_with_A_grade (total : ℕ) (h1 : total ≤ 50) 
  (h2 : total % 3 = 0) (h3 : total % 13 = 0) : ℕ := by
  have h4 : total = 39 := by sorry
  have h5 : (total / 3 : ℕ) + (5 * total / 13 : ℕ) + 1 + 10 = total := by sorry
  exact 10

#check students_with_A_grade

end NUMINAMATH_CALUDE_students_with_A_grade_l1294_129467


namespace NUMINAMATH_CALUDE_mixture_weight_l1294_129414

/-- Calculates the weight of a mixture of two brands of vegetable ghee -/
theorem mixture_weight (weight_a weight_b : ℝ) (ratio_a ratio_b total_volume : ℝ) : 
  weight_a = 900 →
  weight_b = 800 →
  ratio_a = 3 →
  ratio_b = 2 →
  total_volume = 4 →
  (((ratio_a / (ratio_a + ratio_b)) * total_volume * weight_a + 
   (ratio_b / (ratio_a + ratio_b)) * total_volume * weight_b) / 1000) = 3.44 := by
  sorry

#check mixture_weight

end NUMINAMATH_CALUDE_mixture_weight_l1294_129414


namespace NUMINAMATH_CALUDE_A_99_times_B_l1294_129426

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 1; 1, 0, 0; 0, 1, 0]
def B : Matrix (Fin 3) (Fin 3) ℝ := !![1, 1, 1; 0, 1, -1; 1, 0, 0]

theorem A_99_times_B : 
  A^99 * B = !![1, 0, 0; 1, 1, 1; 0, 1, -1] := by sorry

end NUMINAMATH_CALUDE_A_99_times_B_l1294_129426


namespace NUMINAMATH_CALUDE_daily_reading_goal_l1294_129442

def sunday_pages : ℕ := 43
def monday_pages : ℕ := 65
def tuesday_pages : ℕ := 28
def wednesday_pages : ℕ := 0
def thursday_pages : ℕ := 70
def friday_pages : ℕ := 56
def saturday_pages : ℕ := 88

def total_pages : ℕ := sunday_pages + monday_pages + tuesday_pages + wednesday_pages + thursday_pages + friday_pages + saturday_pages

def days_in_week : ℕ := 7

theorem daily_reading_goal :
  (total_pages : ℚ) / days_in_week = 50 := by sorry

end NUMINAMATH_CALUDE_daily_reading_goal_l1294_129442


namespace NUMINAMATH_CALUDE_pete_total_books_l1294_129492

theorem pete_total_books (matt_second_year : ℕ) 
  (h1 : matt_second_year = 75)
  (matt_first_year : ℕ) 
  (h2 : matt_second_year = (3/2 : ℚ) * matt_first_year)
  (pete_first_year : ℕ)
  (h3 : pete_first_year = 2 * matt_first_year)
  (pete_second_year : ℕ)
  (h4 : pete_second_year = 2 * pete_first_year) : 
  pete_first_year + pete_second_year = 300 := by
  sorry

#check pete_total_books

end NUMINAMATH_CALUDE_pete_total_books_l1294_129492


namespace NUMINAMATH_CALUDE_rachel_age_when_emily_half_is_eight_l1294_129428

/-- Rachel's age when Emily's age is half of Rachel's --/
def rachels_age_when_emily_half (emily_current_age rachel_current_age : ℕ) : ℕ :=
  let age_difference := rachel_current_age - emily_current_age
  let emily_half_age := age_difference
  emily_half_age + age_difference

theorem rachel_age_when_emily_half_is_eight :
  rachels_age_when_emily_half 20 24 = 8 := by sorry

end NUMINAMATH_CALUDE_rachel_age_when_emily_half_is_eight_l1294_129428


namespace NUMINAMATH_CALUDE_power_of_81_l1294_129461

theorem power_of_81 : (81 : ℝ) ^ (5/4) = 243 := by
  sorry

end NUMINAMATH_CALUDE_power_of_81_l1294_129461


namespace NUMINAMATH_CALUDE_derivative_of_product_l1294_129420

-- Define the function f(x) = (x+4)(x-7)
def f (x : ℝ) : ℝ := (x + 4) * (x - 7)

-- State the theorem
theorem derivative_of_product (x : ℝ) : 
  deriv f x = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_product_l1294_129420


namespace NUMINAMATH_CALUDE_stickers_per_page_l1294_129419

theorem stickers_per_page (total_pages : ℕ) (total_stickers : ℕ) (stickers_per_page : ℕ) : 
  total_pages = 22 → 
  total_stickers = 220 → 
  total_stickers = total_pages * stickers_per_page → 
  stickers_per_page = 10 := by
sorry

end NUMINAMATH_CALUDE_stickers_per_page_l1294_129419


namespace NUMINAMATH_CALUDE_candy_distribution_l1294_129493

theorem candy_distribution (k : ℕ) : 
  (∃ q : ℕ, k = 7 * q + 3) → 
  (∃ r : ℕ, 3 * k = 7 * r + 2) := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l1294_129493


namespace NUMINAMATH_CALUDE_circle_satisfies_conditions_l1294_129497

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line y = -4x --/
def line1 (x y : ℝ) : Prop := y = -4 * x

/-- The line x + y - 1 = 0 --/
def line2 (x y : ℝ) : Prop := x + y - 1 = 0

/-- Check if a point is on a circle --/
def isOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- Check if a circle is tangent to a line at a point --/
def isTangent (c : Circle) (p : ℝ × ℝ) : Prop :=
  isOnCircle c p ∧ line2 p.1 p.2

/-- The equation of the circle --/
def circleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y - 95 = 0

theorem circle_satisfies_conditions :
  ∃! c : Circle,
    (∃ x y : ℝ, line1 x y ∧ c.center = (x, y)) ∧
    isTangent c (3, -2) ∧
    isOnCircle c (1, 12) ∧
    isOnCircle c (7, 10) ∧
    isOnCircle c (-9, 2) ∧
    ∀ x y : ℝ, circleEquation x y ↔ isOnCircle c (x, y) :=
  sorry


end NUMINAMATH_CALUDE_circle_satisfies_conditions_l1294_129497


namespace NUMINAMATH_CALUDE_probability_is_three_sixty_fourth_l1294_129446

/-- Represents a person with their blocks -/
structure Person :=
  (blocks : Fin 4 → Color)

/-- Represents the possible colors of blocks -/
inductive Color
  | Red
  | Blue
  | Yellow
  | Green
  | White

/-- Represents a placement of blocks in boxes -/
def Placement := Fin 4 → Fin 3 → Color

/-- The set of all possible placements -/
def allPlacements : Set Placement := sorry

/-- Predicate for a placement having at least one box with 3 blocks of the same color -/
def hasThreeSameColor (p : Placement) : Prop := sorry

/-- The probability of a placement having at least one box with 3 blocks of the same color -/
def probability : ℚ := sorry

/-- The main theorem stating the probability -/
theorem probability_is_three_sixty_fourth : probability = 3 / 64 := sorry

end NUMINAMATH_CALUDE_probability_is_three_sixty_fourth_l1294_129446


namespace NUMINAMATH_CALUDE_compressed_music_space_l1294_129490

-- Define the parameters of the problem
def total_days : ℕ := 20
def total_space : ℕ := 25000
def compression_rate : ℚ := 1/10

-- Define the function to calculate the average space per hour
def avg_space_per_hour (days : ℕ) (space : ℕ) (rate : ℚ) : ℚ :=
  let total_hours : ℕ := days * 24
  let compressed_space : ℚ := space * (1 - rate)
  compressed_space / total_hours

-- Theorem statement
theorem compressed_music_space :
  round (avg_space_per_hour total_days total_space compression_rate) = 47 := by
  sorry

end NUMINAMATH_CALUDE_compressed_music_space_l1294_129490


namespace NUMINAMATH_CALUDE_max_choir_members_satisfies_conditions_max_choir_members_is_maximum_l1294_129443

/-- The maximum number of choir members satisfying the given conditions -/
def max_choir_members : ℕ := 54

/-- Predicate to check if a number satisfies the square formation condition -/
def satisfies_square_condition (n : ℕ) : Prop :=
  ∃ x : ℕ, n = x^2 + 11

/-- Predicate to check if a number satisfies the rectangle formation condition -/
def satisfies_rectangle_condition (n : ℕ) : Prop :=
  ∃ y : ℕ, n = y * (y + 3)

/-- Theorem stating that max_choir_members satisfies both conditions -/
theorem max_choir_members_satisfies_conditions :
  satisfies_square_condition max_choir_members ∧
  satisfies_rectangle_condition max_choir_members :=
by sorry

/-- Theorem stating that max_choir_members is the maximum number satisfying both conditions -/
theorem max_choir_members_is_maximum :
  ∀ n : ℕ, 
    satisfies_square_condition n ∧ 
    satisfies_rectangle_condition n → 
    n ≤ max_choir_members :=
by sorry

end NUMINAMATH_CALUDE_max_choir_members_satisfies_conditions_max_choir_members_is_maximum_l1294_129443


namespace NUMINAMATH_CALUDE_inequality_holds_iff_x_eq_pi_div_2_l1294_129434

theorem inequality_holds_iff_x_eq_pi_div_2 : 
  ∀ x : ℝ, 0 < x → x < π → 
  ((8 / (3 * Real.sin x - Real.sin (3 * x))) + 3 * (Real.sin x)^2 ≤ 5 ↔ x = π / 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_x_eq_pi_div_2_l1294_129434


namespace NUMINAMATH_CALUDE_triangle_prime_sides_area_not_integer_l1294_129406

theorem triangle_prime_sides_area_not_integer 
  (a b c : ℕ) 
  (ha : Prime a) 
  (hb : Prime b) 
  (hc : Prime c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  ¬(∃ (S : ℕ), S^2 * 16 = (a + b + c) * ((a + b + c) - 2*a) * ((a + b + c) - 2*b) * ((a + b + c) - 2*c)) :=
sorry

end NUMINAMATH_CALUDE_triangle_prime_sides_area_not_integer_l1294_129406


namespace NUMINAMATH_CALUDE_system_solution_l1294_129449

theorem system_solution : 
  ∀ x y : ℝ, 
    (x + y = (7 - x) + (7 - y) ∧ 
     x^2 - y = (x - 2) + (y - 2)) ↔ 
    ((x = -5 ∧ y = 12) ∨ (x = 2 ∧ y = 5)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1294_129449


namespace NUMINAMATH_CALUDE_symmetric_point_example_l1294_129479

/-- Given a point A and a line L, find the point B symmetric to A about L -/
def symmetric_point (A : ℝ × ℝ) (L : ℝ → ℝ → ℝ) : ℝ × ℝ :=
  sorry

/-- The line 2x - 4y + 9 = 0 -/
def line (x y : ℝ) : ℝ := 2 * x - 4 * y + 9

theorem symmetric_point_example :
  symmetric_point (2, 2) line = (1, 4) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_example_l1294_129479


namespace NUMINAMATH_CALUDE_simplify_sqrt_squared_l1294_129410

theorem simplify_sqrt_squared (a : ℝ) (h : a < 2) : Real.sqrt ((a - 2)^2) = 2 - a := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_squared_l1294_129410


namespace NUMINAMATH_CALUDE_even_function_symmetry_l1294_129460

/-- Definition of an even function -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

/-- Definition of an odd function -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

/-- The main theorem stating that only the third proposition is correct -/
theorem even_function_symmetry :
  (¬ ∀ f : ℝ → ℝ, EvenFunction f → ∃ y : ℝ, f 0 = y) ∧
  (¬ ∀ f : ℝ → ℝ, OddFunction f → f 0 = 0) ∧
  (∀ f : ℝ → ℝ, EvenFunction f → ∀ x : ℝ, f x = f (-x)) ∧
  (¬ ∀ f : ℝ → ℝ, (EvenFunction f ∧ OddFunction f) → ∀ x : ℝ, f x = 0) :=
sorry

end NUMINAMATH_CALUDE_even_function_symmetry_l1294_129460


namespace NUMINAMATH_CALUDE_pages_copied_l1294_129494

def cost_per_page : ℚ := 3/100
def total_money : ℚ := 15

theorem pages_copied (cost_per_page : ℚ) (total_money : ℚ) :
  total_money / cost_per_page = 500 := by sorry

end NUMINAMATH_CALUDE_pages_copied_l1294_129494


namespace NUMINAMATH_CALUDE_orange_cost_27_pounds_l1294_129459

/-- The cost of oranges in dollars per 3 pounds -/
def orange_rate : ℚ := 3

/-- The weight of oranges in pounds that we want to buy -/
def orange_weight : ℚ := 27

/-- The cost of oranges for a given weight -/
def orange_cost (weight : ℚ) : ℚ := (weight / 3) * orange_rate

theorem orange_cost_27_pounds :
  orange_cost orange_weight = 27 := by sorry

end NUMINAMATH_CALUDE_orange_cost_27_pounds_l1294_129459


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l1294_129416

/-- The area of a square with adjacent vertices at (1,5) and (4,-2) is 58 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, 5)
  let p2 : ℝ × ℝ := (4, -2)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 58 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l1294_129416


namespace NUMINAMATH_CALUDE_max_bus_stop_distance_l1294_129404

/-- Represents the problem of finding the maximum distance between bus stops -/
theorem max_bus_stop_distance (peter_speed : ℝ) (bus_speed : ℝ) (sight_distance : ℝ) :
  peter_speed > 0 →
  bus_speed = 3 * peter_speed →
  sight_distance = 0.8 →
  ∃ (max_distance : ℝ),
    max_distance = 0.6 ∧
    ∀ (d : ℝ), 0 < d ∧ d ≤ max_distance →
      (∀ (x : ℝ), 0 ≤ x ∧ x ≤ d →
        (x + sight_distance) / peter_speed ≤ (d - x) / bus_speed ∨
        (2 * x + sight_distance) / peter_speed ≤ d / bus_speed) :=
by sorry

end NUMINAMATH_CALUDE_max_bus_stop_distance_l1294_129404


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1294_129496

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x > y ∧ y > 0 → abs x > abs y) ∧
  ∃ a b : ℝ, abs a > abs b ∧ ¬(a > b ∧ b > 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1294_129496


namespace NUMINAMATH_CALUDE_ribbon_tape_remaining_l1294_129475

theorem ribbon_tape_remaining (total length_used_ribbon length_used_gift : ℝ) 
  (h1 : total = 1.6)
  (h2 : length_used_ribbon = 0.8)
  (h3 : length_used_gift = 0.3) :
  total - length_used_ribbon - length_used_gift = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_tape_remaining_l1294_129475
