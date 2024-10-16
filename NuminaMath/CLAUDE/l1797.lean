import Mathlib

namespace NUMINAMATH_CALUDE_complex_equation_solutions_l1797_179764

theorem complex_equation_solutions :
  ∃! (S : Finset ℂ), 
    (∀ z ∈ S, Complex.abs z < 15 ∧ Complex.exp (2 * z) = (z - 2) / (z + 2)) ∧
    Finset.card S = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l1797_179764


namespace NUMINAMATH_CALUDE_max_books_borrowed_l1797_179763

theorem max_books_borrowed (total_students : Nat) (zero_books : Nat) (one_book : Nat) (two_books : Nat) (avg_books : Nat) :
  total_students = 30 →
  zero_books = 5 →
  one_book = 12 →
  two_books = 8 →
  avg_books = 2 →
  ∃ (max_books : Nat), max_books = 20 ∧ 
    ∀ (student_books : Nat), student_books ≤ max_books ∧
    (total_students * avg_books = 
      zero_books * 0 + one_book * 1 + two_books * 2 + 
      (total_students - zero_books - one_book - two_books - 1) * 3 + max_books) :=
by sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l1797_179763


namespace NUMINAMATH_CALUDE_rock_paper_scissors_wins_l1797_179784

/-- Represents the outcome of a single round --/
inductive RoundResult
| Win
| Lose
| Tie

/-- Represents a player's position and game results --/
structure PlayerState :=
  (position : Int)
  (wins : Nat)
  (losses : Nat)
  (ties : Nat)

/-- Updates a player's state based on the round result --/
def updatePlayerState (state : PlayerState) (result : RoundResult) : PlayerState :=
  match result with
  | RoundResult.Win => { state with position := state.position + 3, wins := state.wins + 1 }
  | RoundResult.Lose => { state with position := state.position - 2, losses := state.losses + 1 }
  | RoundResult.Tie => { state with position := state.position + 1, ties := state.ties + 1 }

/-- Represents the state of the game --/
structure GameState :=
  (playerA : PlayerState)
  (playerB : PlayerState)
  (rounds : Nat)

/-- Updates the game state based on the round result for Player A --/
def updateGameState (state : GameState) (result : RoundResult) : GameState :=
  { state with
    playerA := updatePlayerState state.playerA result,
    playerB := updatePlayerState state.playerB (match result with
      | RoundResult.Win => RoundResult.Lose
      | RoundResult.Lose => RoundResult.Win
      | RoundResult.Tie => RoundResult.Tie),
    rounds := state.rounds + 1 }

/-- The main theorem to prove --/
theorem rock_paper_scissors_wins
  (initialDistance : Nat)
  (totalRounds : Nat)
  (finalPositionA : Int)
  (finalPositionB : Int)
  (h1 : initialDistance = 30)
  (h2 : totalRounds = 15)
  (h3 : finalPositionA = 17)
  (h4 : finalPositionB = 2) :
  ∃ (gameResults : List RoundResult),
    let finalState := gameResults.foldl updateGameState
      { playerA := ⟨0, 0, 0, 0⟩,
        playerB := ⟨initialDistance, 0, 0, 0⟩,
        rounds := 0 }
    finalState.rounds = totalRounds ∧
    finalState.playerA.position = finalPositionA ∧
    finalState.playerB.position = finalPositionB ∧
    finalState.playerA.wins = 7 :=
sorry

end NUMINAMATH_CALUDE_rock_paper_scissors_wins_l1797_179784


namespace NUMINAMATH_CALUDE_line_slope_equation_l1797_179779

/-- Given a line passing through points (-1, -4) and (3, k), where the slope
    of the line is equal to k, prove that k = 4/3 -/
theorem line_slope_equation (k : ℝ) : 
  (let x₁ : ℝ := -1
   let y₁ : ℝ := -4
   let x₂ : ℝ := 3
   let y₂ : ℝ := k
   let slope : ℝ := (y₂ - y₁) / (x₂ - x₁)
   slope = k) → k = 4/3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_equation_l1797_179779


namespace NUMINAMATH_CALUDE_interest_payment_time_l1797_179720

-- Define the principal amount
def principal : ℝ := 8000

-- Define the interest rates
def rate1 : ℝ := 0.08
def rate2 : ℝ := 0.10
def rate3 : ℝ := 0.12

-- Define the time periods
def time1 : ℝ := 4
def time2 : ℝ := 6

-- Define the total interest paid
def totalInterest : ℝ := 12160

-- Function to calculate interest
def calculateInterest (p : ℝ) (r : ℝ) (t : ℝ) : ℝ := p * r * t

-- Theorem statement
theorem interest_payment_time :
  ∃ t : ℝ, 
    calculateInterest principal rate1 time1 +
    calculateInterest principal rate2 time2 +
    calculateInterest principal rate3 (t - (time1 + time2)) = totalInterest ∧
    t = 15 := by sorry

end NUMINAMATH_CALUDE_interest_payment_time_l1797_179720


namespace NUMINAMATH_CALUDE_squirrel_acorns_l1797_179752

theorem squirrel_acorns (num_squirrels : ℕ) (acorns_collected : ℕ) (acorns_needed_per_squirrel : ℕ) :
  num_squirrels = 7 →
  acorns_collected = 875 →
  acorns_needed_per_squirrel = 170 →
  (acorns_needed_per_squirrel * num_squirrels - acorns_collected) / num_squirrels = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l1797_179752


namespace NUMINAMATH_CALUDE_range_of_m_l1797_179795

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 4*x + 3
def g (m x : ℝ) : ℝ := m*x + 3 - 2*m

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, 
  (∀ x₁ ∈ Set.Icc 0 4, ∃ x₂ ∈ Set.Icc 0 4, f x₁ = g m x₂) ↔ 
  m ∈ Set.Iic (-2) ∪ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1797_179795


namespace NUMINAMATH_CALUDE_perfect_square_equation_l1797_179776

theorem perfect_square_equation (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_equation_l1797_179776


namespace NUMINAMATH_CALUDE_remainder_r17_plus_1_div_r_plus_1_l1797_179767

theorem remainder_r17_plus_1_div_r_plus_1 (r : ℤ) : (r^17 + 1) % (r + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_r17_plus_1_div_r_plus_1_l1797_179767


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l1797_179765

theorem not_sufficient_not_necessary (a b : ℝ) :
  ¬(∀ a b : ℝ, 0 < a * b ∧ a * b < 1 → b < 1 / a) ∧
  ¬(∀ a b : ℝ, b < 1 / a → 0 < a * b ∧ a * b < 1) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l1797_179765


namespace NUMINAMATH_CALUDE_train_journey_equation_l1797_179792

/-- Represents the equation for a train journey where:
    - x is the distance in km
    - The speed increases from 160 km/h to 200 km/h
    - The travel time reduces by 2.5 hours
-/
theorem train_journey_equation (x : ℝ) : x / 160 - x / 200 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_equation_l1797_179792


namespace NUMINAMATH_CALUDE_smaller_number_in_sum_l1797_179712

theorem smaller_number_in_sum (x y : ℕ) : 
  x + y = 84 → y = 3 * x → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_sum_l1797_179712


namespace NUMINAMATH_CALUDE_a₃_value_l1797_179785

/-- The function f(x) = x^6 -/
def f (x : ℝ) : ℝ := x^6

/-- The expansion of f(x) in terms of (1+x) -/
def f_expansion (x a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : ℝ := 
  a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5 + a₆*(1+x)^6

/-- Theorem: If f(x) = x^6 can be expressed as the expansion, then a₃ = -20 -/
theorem a₃_value (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, f x = f_expansion x a₀ a₁ a₂ a₃ a₄ a₅ a₆) → a₃ = -20 := by
  sorry

end NUMINAMATH_CALUDE_a₃_value_l1797_179785


namespace NUMINAMATH_CALUDE_complex_number_location_l1797_179742

theorem complex_number_location (z : ℂ) (h : z * (1 + Complex.I) * (-2 * Complex.I) = z) :
  Complex.re z > 0 ∧ Complex.im z < 0 :=
sorry

end NUMINAMATH_CALUDE_complex_number_location_l1797_179742


namespace NUMINAMATH_CALUDE_exponent_calculation_l1797_179734

theorem exponent_calculation : (1 / ((-5^4)^2)) * (-5)^7 = -1/5 := by sorry

end NUMINAMATH_CALUDE_exponent_calculation_l1797_179734


namespace NUMINAMATH_CALUDE_square_pyramid_frustum_volume_fraction_l1797_179758

/-- The volume of a square pyramid frustum as a fraction of the original pyramid --/
theorem square_pyramid_frustum_volume_fraction 
  (base_edge : ℝ) 
  (altitude : ℝ) 
  (h_base : base_edge = 40) 
  (h_alt : altitude = 18) :
  let original_volume := (1/3) * base_edge^2 * altitude
  let small_base_edge := (1/5) * base_edge
  let small_altitude := (1/5) * altitude
  let small_volume := (1/3) * small_base_edge^2 * small_altitude
  let frustum_volume := original_volume - small_volume
  frustum_volume / original_volume = 2383 / 2400 := by
sorry

end NUMINAMATH_CALUDE_square_pyramid_frustum_volume_fraction_l1797_179758


namespace NUMINAMATH_CALUDE_largest_fraction_l1797_179761

theorem largest_fraction : 
  (151 : ℚ) / 301 > 3 / 7 ∧
  (151 : ℚ) / 301 > 4 / 9 ∧
  (151 : ℚ) / 301 > 17 / 35 ∧
  (151 : ℚ) / 301 > 100 / 201 := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l1797_179761


namespace NUMINAMATH_CALUDE_solve_seashells_problem_l1797_179721

def seashells_problem (initial_seashells current_seashells : ℕ) : Prop :=
  ∃ (given_seashells : ℕ), 
    initial_seashells = current_seashells + given_seashells

theorem solve_seashells_problem (initial_seashells current_seashells : ℕ) 
  (h : initial_seashells ≥ current_seashells) :
  seashells_problem initial_seashells current_seashells →
  ∃ (given_seashells : ℕ), given_seashells = initial_seashells - current_seashells :=
by
  sorry

end NUMINAMATH_CALUDE_solve_seashells_problem_l1797_179721


namespace NUMINAMATH_CALUDE_unique_quadratic_function_l1797_179730

/-- A quadratic function satisfying specific conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c) ∧
  f 0 = 1 ∧
  ∀ x, f (x + 1) - f x = 2 * x

/-- The unique quadratic function satisfying the given conditions -/
theorem unique_quadratic_function (f : ℝ → ℝ) (hf : QuadraticFunction f) :
  ∀ x, f x = x^2 - x + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_function_l1797_179730


namespace NUMINAMATH_CALUDE_reading_time_proof_l1797_179741

def total_chapters : Nat := 31
def reading_time_per_chapter : Nat := 20

def chapters_read (n : Nat) : Nat :=
  n - (n / 3)

def total_reading_time_minutes (n : Nat) (t : Nat) : Nat :=
  (chapters_read n) * t

def total_reading_time_hours (n : Nat) (t : Nat) : Nat :=
  (total_reading_time_minutes n t) / 60

theorem reading_time_proof :
  total_reading_time_hours total_chapters reading_time_per_chapter = 7 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_proof_l1797_179741


namespace NUMINAMATH_CALUDE_unique_divisible_number_l1797_179769

/-- A function that constructs a five-digit number of the form 6n272 -/
def construct_number (n : Nat) : Nat :=
  60000 + n * 1000 + 272

/-- Proposition: 63272 is the only number of the form 6n272 (where n is a single digit) 
    that is divisible by both 11 and 5 -/
theorem unique_divisible_number : 
  ∃! n : Nat, n < 10 ∧ 
  (construct_number n).mod 11 = 0 ∧ 
  (construct_number n).mod 5 = 0 ∧
  construct_number n = 63272 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_number_l1797_179769


namespace NUMINAMATH_CALUDE_line_intersection_l1797_179772

theorem line_intersection :
  ∀ (x y : ℚ),
  (12 * x - 3 * y = 33) →
  (8 * x + 2 * y = 18) →
  (x = 29/12 ∧ y = -2/3) :=
by
  sorry

end NUMINAMATH_CALUDE_line_intersection_l1797_179772


namespace NUMINAMATH_CALUDE_average_first_100_odd_numbers_l1797_179750

theorem average_first_100_odd_numbers : 
  let n := 100
  let nth_odd (k : ℕ) := 2 * k - 1
  let first_odd := nth_odd 1
  let last_odd := nth_odd n
  let sum := (n / 2) * (first_odd + last_odd)
  sum / n = 100 := by
sorry

end NUMINAMATH_CALUDE_average_first_100_odd_numbers_l1797_179750


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_l1797_179723

/-- If the terminal side of angle α passes through the point (-1, 2) in the Cartesian coordinate system, then sin α = (2√5) / 5 -/
theorem sin_alpha_for_point (α : Real) : 
  (∃ (x y : Real), x = -1 ∧ y = 2 ∧ x = Real.cos α ∧ y = Real.sin α) →
  Real.sin α = (2 * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_l1797_179723


namespace NUMINAMATH_CALUDE_kolya_purchase_l1797_179724

/-- Represents the cost of an item in kopecks -/
def item_cost (rubles : ℕ) : ℕ := 100 * rubles + 99

/-- Represents the total purchase cost in kopecks -/
def total_cost : ℕ := 200 * 100 + 83

/-- Predicate to check if a given number of items is a valid solution -/
def is_valid_solution (n : ℕ) : Prop :=
  ∃ (rubles : ℕ), n * (item_cost rubles) = total_cost

theorem kolya_purchase :
  ∀ n : ℕ, is_valid_solution n ↔ n = 17 ∨ n = 117 :=
sorry

end NUMINAMATH_CALUDE_kolya_purchase_l1797_179724


namespace NUMINAMATH_CALUDE_volleyball_lineup_combinations_l1797_179708

def team_size : ℕ := 12
def starting_lineup_size : ℕ := 6
def non_libero_positions : ℕ := 5

theorem volleyball_lineup_combinations :
  (team_size) * (Nat.choose (team_size - 1) non_libero_positions) = 5544 :=
sorry

end NUMINAMATH_CALUDE_volleyball_lineup_combinations_l1797_179708


namespace NUMINAMATH_CALUDE_chord_line_equation_l1797_179757

/-- The equation of a line containing a chord of a parabola -/
theorem chord_line_equation (x y : ℝ → ℝ) :
  (∀ t : ℝ, (y t)^2 = -8 * (x t)) →  -- parabola equation
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
    (x t₁ + x t₂) / 2 = -1 ∧ 
    (y t₁ + y t₂) / 2 = 1) →  -- midpoint condition
  ∃ a b c : ℝ, a ≠ 0 ∧ 
    (∀ t : ℝ, a * (x t) + b * (y t) + c = 0) ∧ 
    (4 * a = -b ∧ 3 * a = -c) :=  -- line equation
by sorry

end NUMINAMATH_CALUDE_chord_line_equation_l1797_179757


namespace NUMINAMATH_CALUDE_widget_selling_price_l1797_179798

-- Define the problem parameters
def widget_cost : ℝ := 3
def monthly_rent : ℝ := 10000
def tax_rate : ℝ := 0.20
def worker_salary : ℝ := 2500
def num_workers : ℕ := 4
def widgets_sold : ℕ := 5000
def total_profit : ℝ := 4000

-- Define the theorem
theorem widget_selling_price :
  let worker_expenses : ℝ := worker_salary * num_workers
  let total_expenses : ℝ := monthly_rent + worker_expenses
  let widget_expenses : ℝ := widget_cost * widgets_sold
  let taxes : ℝ := tax_rate * total_profit
  let total_expenses_with_taxes : ℝ := total_expenses + widget_expenses + taxes
  let total_revenue : ℝ := total_expenses_with_taxes + total_profit
  let selling_price : ℝ := total_revenue / widgets_sold
  selling_price = 7.96 := by
  sorry

end NUMINAMATH_CALUDE_widget_selling_price_l1797_179798


namespace NUMINAMATH_CALUDE_fish_cost_per_kg_proof_l1797_179740

-- Define the constants
def total_cost_case1 : ℕ := 530
def fish_kg_case1 : ℕ := 4
def pork_kg_case1 : ℕ := 2
def pork_kg_case2 : ℕ := 3
def total_cost_case2 : ℕ := 875
def fish_cost_per_kg : ℕ := 80

-- Define the theorem
theorem fish_cost_per_kg_proof :
  let pork_cost_case1 := total_cost_case1 - fish_cost_per_kg * fish_kg_case1
  let pork_cost_per_kg := pork_cost_case1 / pork_kg_case1
  let pork_cost_case2 := pork_cost_per_kg * pork_kg_case2
  let fish_cost_case2 := total_cost_case2 - pork_cost_case2
  fish_cost_case2 / (fish_cost_case2 / fish_cost_per_kg) = fish_cost_per_kg :=
by
  sorry

#check fish_cost_per_kg_proof

end NUMINAMATH_CALUDE_fish_cost_per_kg_proof_l1797_179740


namespace NUMINAMATH_CALUDE_problem_solution_l1797_179778

theorem problem_solution :
  ∀ (x y : ℕ), 
    y > 3 → 
    x^2 + y^4 = 2*((x-6)^2 + (y+1)^2) → 
    x^2 + y^4 = 1994 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1797_179778


namespace NUMINAMATH_CALUDE_will_chocolate_pieces_l1797_179786

theorem will_chocolate_pieces : 
  ∀ (total_boxes given_boxes pieces_per_box : ℕ),
  total_boxes = 7 →
  given_boxes = 3 →
  pieces_per_box = 4 →
  (total_boxes - given_boxes) * pieces_per_box = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_will_chocolate_pieces_l1797_179786


namespace NUMINAMATH_CALUDE_hansel_salary_l1797_179796

theorem hansel_salary (hansel_initial : ℝ) (gretel_initial : ℝ) :
  hansel_initial = gretel_initial →
  hansel_initial * 1.10 + 1500 = gretel_initial * 1.15 →
  hansel_initial = 30000 := by
  sorry

end NUMINAMATH_CALUDE_hansel_salary_l1797_179796


namespace NUMINAMATH_CALUDE_equation_solutions_l1797_179707

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(1, 2, 26), (1, 8, 8), (2, 2, 19), (2, 4, 12), (2, 5, 10), (4, 4, 8)}

def satisfies_equation (triple : ℕ × ℕ × ℕ) : Prop :=
  let (x, y, z) := triple
  x * y + y * z + z * x = 80 ∧ x ≤ y ∧ y ≤ z

theorem equation_solutions :
  ∀ (x y z : ℕ), satisfies_equation (x, y, z) ↔ (x, y, z) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1797_179707


namespace NUMINAMATH_CALUDE_range_of_f_leq_3_l1797_179799

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 8 then x^(1/3) else 2 * Real.exp (x - 8)

-- Theorem statement
theorem range_of_f_leq_3 :
  {x : ℝ | f x ≤ 3} = {x : ℝ | x ≤ 27} := by sorry

end NUMINAMATH_CALUDE_range_of_f_leq_3_l1797_179799


namespace NUMINAMATH_CALUDE_weather_probability_l1797_179747

theorem weather_probability (p_rain p_cloudy : ℝ) 
  (h_rain : p_rain = 0.45)
  (h_cloudy : p_cloudy = 0.20)
  (h_nonneg_rain : 0 ≤ p_rain)
  (h_nonneg_cloudy : 0 ≤ p_cloudy)
  (h_sum_le_one : p_rain + p_cloudy ≤ 1) :
  1 - p_rain - p_cloudy = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_weather_probability_l1797_179747


namespace NUMINAMATH_CALUDE_prime_factors_of_2008006_l1797_179787

theorem prime_factors_of_2008006 : 
  ∃ (p₁ p₂ p₃ p₄ p₅ p₆ : Nat), 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ 
    Nat.Prime p₄ ∧ Nat.Prime p₅ ∧ Nat.Prime p₆ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ p₁ ≠ p₆ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ p₂ ≠ p₆ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ p₃ ≠ p₆ ∧
    p₄ ≠ p₅ ∧ p₄ ≠ p₆ ∧
    p₅ ≠ p₆ ∧
    2008006 = p₁ * p₂ * p₃ * p₄ * p₅ * p₆ ∧
    (∀ q : Nat, Nat.Prime q → q ∣ 2008006 → (q = p₁ ∨ q = p₂ ∨ q = p₃ ∨ q = p₄ ∨ q = p₅ ∨ q = p₆)) :=
by sorry


end NUMINAMATH_CALUDE_prime_factors_of_2008006_l1797_179787


namespace NUMINAMATH_CALUDE_integer_root_of_polynomial_l1797_179782

-- Define the polynomial
def polynomial (d e f g x : ℚ) : ℚ := x^4 + d*x^3 + e*x^2 + f*x + g

-- State the theorem
theorem integer_root_of_polynomial (d e f g : ℚ) :
  (∃ (x : ℚ), x = 3 + Real.sqrt 5 ∧ polynomial d e f g x = 0) →
  (∃ (n : ℤ), polynomial d e f g (↑n) = 0 ∧ 
    (∀ (m : ℤ), m ≠ n → polynomial d e f g (↑m) ≠ 0)) →
  polynomial d e f g (-3) = 0 :=
sorry

end NUMINAMATH_CALUDE_integer_root_of_polynomial_l1797_179782


namespace NUMINAMATH_CALUDE_range_of_g_l1797_179753

open Set
open Function

def g (x : ℝ) : ℝ := 3 * (x - 4)

theorem range_of_g :
  range g = {y : ℝ | y < -27 ∨ y > -27} :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l1797_179753


namespace NUMINAMATH_CALUDE_random_selection_more_representative_l1797_179722

/-- Represents a student in the school -/
structure Student where
  grade : ℕ
  gender : Bool

/-- Represents the entire student population of the school -/
def StudentPopulation := List Student

/-- Represents a sample of students -/
def StudentSample := List Student

/-- Function to check if a sample is representative of the population -/
def isRepresentative (population : StudentPopulation) (sample : StudentSample) : Prop :=
  -- Definition of what makes a sample representative
  sorry

/-- Function to randomly select students from various grades -/
def randomSelectFromGrades (population : StudentPopulation) (sampleSize : ℕ) : StudentSample :=
  -- Implementation of random selection from various grades
  sorry

/-- Function to select students from a single class -/
def selectFromSingleClass (population : StudentPopulation) (sampleSize : ℕ) : StudentSample :=
  -- Implementation of selection from a single class
  sorry

/-- Function to select students of a single gender -/
def selectSingleGender (population : StudentPopulation) (sampleSize : ℕ) : StudentSample :=
  -- Implementation of selection of a single gender
  sorry

/-- Theorem stating that random selection from various grades is more representative -/
theorem random_selection_more_representative 
  (population : StudentPopulation) (sampleSize : ℕ) : 
  isRepresentative population (randomSelectFromGrades population sampleSize) ∧
  ¬isRepresentative population (selectFromSingleClass population sampleSize) ∧
  ¬isRepresentative population (selectSingleGender population sampleSize) :=
by
  sorry


end NUMINAMATH_CALUDE_random_selection_more_representative_l1797_179722


namespace NUMINAMATH_CALUDE_both_sports_solution_l1797_179709

/-- The number of students who like both basketball and cricket -/
def both_sports (basketball cricket total : ℕ) : ℕ :=
  basketball + cricket - total

theorem both_sports_solution : both_sports 7 8 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_both_sports_solution_l1797_179709


namespace NUMINAMATH_CALUDE_subtract_seven_percent_l1797_179766

theorem subtract_seven_percent (a : ℝ) : a - 0.07 * a = 0.93 * a := by
  sorry

end NUMINAMATH_CALUDE_subtract_seven_percent_l1797_179766


namespace NUMINAMATH_CALUDE_subtract_from_negative_problem_solution_l1797_179710

theorem subtract_from_negative (a b : ℝ) : -a - b = -(a + b) := by sorry

theorem problem_solution : -3.219 - 7.305 = -10.524 := by sorry

end NUMINAMATH_CALUDE_subtract_from_negative_problem_solution_l1797_179710


namespace NUMINAMATH_CALUDE_parallelogram_area_theorem_l1797_179773

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculates the area of a parallelogram given its vertices -/
def parallelogramArea (p : Parallelogram) : ℝ :=
  sorry

/-- Theorem stating that the area of a parallelogram with specific vertex coordinates is 4ap -/
theorem parallelogram_area_theorem (x a b p : ℝ) :
  let para := Parallelogram.mk
    (Point.mk x p)
    (Point.mk a b)
    (Point.mk x (-p))
    (Point.mk (-a) (-b))
  parallelogramArea para = 4 * a * p := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_theorem_l1797_179773


namespace NUMINAMATH_CALUDE_digit_mean_is_four_point_five_l1797_179749

/-- The period length of the repeating decimal expansion of 1/(98^2) -/
def period_length : ℕ := 9604

/-- The sum of all digits in one period of the repeating decimal expansion of 1/(98^2) -/
def digit_sum : ℕ := 432180

/-- The mean of the digits in one complete period of the repeating decimal expansion of 1/(98^2) -/
def digit_mean : ℚ := digit_sum / period_length

theorem digit_mean_is_four_point_five :
  digit_mean = 4.5 := by sorry

end NUMINAMATH_CALUDE_digit_mean_is_four_point_five_l1797_179749


namespace NUMINAMATH_CALUDE_function_bounds_l1797_179700

theorem function_bounds 
  (f : ℕ+ → ℕ+) 
  (h_increasing : ∀ n m : ℕ+, n < m → f n < f m) 
  (k : ℕ+) 
  (h_composition : ∀ n : ℕ+, f (f n) = k * n) :
  ∀ n : ℕ+, (2 * k * n : ℚ) / (k + 1) ≤ f n ∧ (f n : ℚ) ≤ ((k + 1) * n) / 2 :=
sorry

end NUMINAMATH_CALUDE_function_bounds_l1797_179700


namespace NUMINAMATH_CALUDE_audrey_twice_heracles_age_l1797_179788

def age_difference : ℕ := 7
def heracles_current_age : ℕ := 10

theorem audrey_twice_heracles_age (years : ℕ) : 
  (heracles_current_age + age_difference + years = 2 * heracles_current_age) → years = 3 := by
  sorry

end NUMINAMATH_CALUDE_audrey_twice_heracles_age_l1797_179788


namespace NUMINAMATH_CALUDE_two_solutions_sine_equation_l1797_179727

theorem two_solutions_sine_equation (x : ℝ) (a : ℝ) : 
  (x ∈ Set.Icc 0 Real.pi) →
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁ ∈ Set.Icc 0 Real.pi ∧ 
    x₂ ∈ Set.Icc 0 Real.pi ∧
    2 * Real.sin (x₁ + Real.pi / 3) = a ∧ 
    2 * Real.sin (x₂ + Real.pi / 3) = a) ↔
  (a > Real.sqrt 3 ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_two_solutions_sine_equation_l1797_179727


namespace NUMINAMATH_CALUDE_football_club_player_selling_price_l1797_179703

/-- Calculates the selling price of each player given the financial transactions of a football club. -/
theorem football_club_player_selling_price 
  (initial_balance : ℝ) 
  (players_sold : ℕ) 
  (players_bought : ℕ) 
  (buying_price : ℝ) 
  (final_balance : ℝ) : 
  initial_balance + players_sold * ((initial_balance - final_balance + players_bought * buying_price) / players_sold) - players_bought * buying_price = final_balance → 
  (initial_balance - final_balance + players_bought * buying_price) / players_sold = 10 :=
by sorry

end NUMINAMATH_CALUDE_football_club_player_selling_price_l1797_179703


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l1797_179706

-- Define the hyperbola equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (|k| - 2) + y^2 / (5 - k) = 1

-- Define the range of k
def k_range (k : ℝ) : Prop :=
  (-2 < k ∧ k < 2) ∨ k > 5

-- Theorem stating the relationship between the hyperbola equation and the range of k
theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k ↔ k_range k :=
sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l1797_179706


namespace NUMINAMATH_CALUDE_f_3_eq_2488_l1797_179735

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 7x^5 + 12x^4 - 5x^3 - 6x^2 + 3x - 5 -/
def f (x : ℝ) : ℝ := 
  horner_eval [7, 12, -5, -6, 3, -5] x

theorem f_3_eq_2488 : f 3 = 2488 := by
  sorry

end NUMINAMATH_CALUDE_f_3_eq_2488_l1797_179735


namespace NUMINAMATH_CALUDE_degree_of_composed_product_l1797_179732

-- Define polynomials f and g with their respective degrees
def f : Polynomial ℝ := sorry
def g : Polynomial ℝ := sorry

-- State the theorem
theorem degree_of_composed_product :
  (Polynomial.degree f = 4) →
  (Polynomial.degree g = 5) →
  Polynomial.degree (f.comp (Polynomial.X ^ 2) * g.comp (Polynomial.X ^ 4)) = 28 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_composed_product_l1797_179732


namespace NUMINAMATH_CALUDE_cos_210_degrees_l1797_179790

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l1797_179790


namespace NUMINAMATH_CALUDE_purple_balls_count_l1797_179719

theorem purple_balls_count (total : ℕ) (white green yellow red : ℕ) (prob : ℚ) :
  total = 100 ∧
  white = 50 ∧
  green = 30 ∧
  yellow = 8 ∧
  red = 9 ∧
  prob = 88/100 ∧
  prob = (white + green + yellow : ℚ) / total →
  total - (white + green + yellow + red) = 0 :=
by sorry

end NUMINAMATH_CALUDE_purple_balls_count_l1797_179719


namespace NUMINAMATH_CALUDE_min_x_value_l1797_179704

theorem min_x_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y^2 = x*y) :
  x ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_x_value_l1797_179704


namespace NUMINAMATH_CALUDE_shaded_cubes_count_l1797_179729

/-- Represents a cube made up of smaller cubes --/
structure Cube where
  size : Nat
  shaded_corners : Bool
  shaded_center : Bool

/-- Counts the number of smaller cubes with at least one face shaded --/
def count_shaded_cubes (c : Cube) : Nat :=
  sorry

/-- Theorem stating that a 4x4x4 cube with shaded corners and centers has 14 shaded cubes --/
theorem shaded_cubes_count (c : Cube) :
  c.size = 4 ∧ c.shaded_corners ∧ c.shaded_center →
  count_shaded_cubes c = 14 :=
by sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_l1797_179729


namespace NUMINAMATH_CALUDE_unique_solution_cubic_equation_l1797_179781

theorem unique_solution_cubic_equation :
  ∃! x : ℝ, x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_equation_l1797_179781


namespace NUMINAMATH_CALUDE_x_plus_reciprocal_squared_l1797_179733

theorem x_plus_reciprocal_squared (x : ℝ) (h : x^2 + 1/x^2 = 7) : (x + 1/x)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_reciprocal_squared_l1797_179733


namespace NUMINAMATH_CALUDE_book_distribution_theorem_l1797_179756

/-- The number of different books to be distributed -/
def num_books : ℕ := 6

/-- The number of students -/
def num_students : ℕ := 6

/-- The number of students who receive books -/
def num_receiving_students : ℕ := num_students - 1

/-- The number of ways to distribute the books -/
def distribution_ways : ℕ := num_students * (num_receiving_students ^ num_books)

theorem book_distribution_theorem : distribution_ways = 93750 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_theorem_l1797_179756


namespace NUMINAMATH_CALUDE_greatest_integer_radius_of_circle_l1797_179755

theorem greatest_integer_radius_of_circle (r : ℝ) : 
  (π * r^2 < 100 * π) → (∀ n : ℕ, n > 9 → π * (n : ℝ)^2 ≥ 100 * π) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_of_circle_l1797_179755


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1797_179718

theorem rationalize_denominator :
  5 / (2 + Real.sqrt 5) = -10 + 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1797_179718


namespace NUMINAMATH_CALUDE_ellipse_triangle_area_l1797_179738

/-- The ellipse with equation x²/49 + y²/24 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 49) + (p.2^2 / 24) = 1}

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

/-- A point on the ellipse -/
def P : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given three points -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

theorem ellipse_triangle_area :
  P ∈ Ellipse ∧ 
  (distance P F₁) / (distance P F₂) = 4 / 3 →
  triangleArea P F₁ F₂ = 24 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_triangle_area_l1797_179738


namespace NUMINAMATH_CALUDE_four_integers_with_average_five_l1797_179715

theorem four_integers_with_average_five (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a + b + c + d : ℚ) / 4 = 5 →
  (max a (max b (max c d)) - min a (min b (min c d)) : ℕ) = 
    max a (max b (max c d)) - min a (min b (min c d)) →
  ((a + b + c + d) - max a (max b (max c d)) - min a (min b (min c d)) : ℚ) / 2 = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_four_integers_with_average_five_l1797_179715


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1797_179743

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- An ellipse with axes parallel to coordinate axes -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- Check if a point lies on an ellipse -/
def onEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

theorem ellipse_major_axis_length :
  let p1 : Point := ⟨0, 0⟩
  let p2 : Point := ⟨2, 2⟩
  let p3 : Point := ⟨-2, 2⟩
  let p4 : Point := ⟨4, 0⟩
  let p5 : Point := ⟨4, 4⟩
  ∃ (e : Ellipse),
    (¬ collinear p1 p2 p3 ∧ ¬ collinear p1 p2 p4 ∧ ¬ collinear p1 p2 p5 ∧
     ¬ collinear p1 p3 p4 ∧ ¬ collinear p1 p3 p5 ∧ ¬ collinear p1 p4 p5 ∧
     ¬ collinear p2 p3 p4 ∧ ¬ collinear p2 p3 p5 ∧ ¬ collinear p2 p4 p5 ∧
     ¬ collinear p3 p4 p5) →
    (onEllipse p1 e ∧ onEllipse p2 e ∧ onEllipse p3 e ∧ onEllipse p4 e ∧ onEllipse p5 e) →
    2 * e.a = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1797_179743


namespace NUMINAMATH_CALUDE_cos_300_degrees_l1797_179705

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l1797_179705


namespace NUMINAMATH_CALUDE_max_projection_area_is_one_l1797_179714

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  /-- Two adjacent faces are isosceles right triangles -/
  adjacent_faces_isosceles_right : Bool
  /-- Hypotenuse of the isosceles right triangles is 2 -/
  hypotenuse : ℝ
  /-- Dihedral angle between the two adjacent faces is 60 degrees -/
  dihedral_angle : ℝ

/-- Calculates the maximum projection area of the rotating tetrahedron -/
def max_projection_area (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the maximum projection area is 1 -/
theorem max_projection_area_is_one (t : Tetrahedron) 
  (h1 : t.adjacent_faces_isosceles_right = true)
  (h2 : t.hypotenuse = 2)
  (h3 : t.dihedral_angle = π / 3) : 
  max_projection_area t = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_projection_area_is_one_l1797_179714


namespace NUMINAMATH_CALUDE_least_integer_with_divisibility_criteria_l1797_179793

def is_divisible (a b : ℕ) : Prop := ∃ k, a = b * k

def is_least_meeting_criteria (n : ℕ) : Prop :=
  (∀ i ∈ Finset.range 27, is_divisible n i) ∧
  (∀ i ∈ Finset.range 30 \ Finset.range 27, ¬ is_divisible n i) ∧
  is_divisible n 30 ∧
  (∀ m : ℕ, m < n → ¬(
    (∀ i ∈ Finset.range 27, is_divisible m i) ∧
    (∀ i ∈ Finset.range 30 \ Finset.range 27, ¬ is_divisible m i) ∧
    is_divisible m 30
  ))

theorem least_integer_with_divisibility_criteria :
  is_least_meeting_criteria 1225224000 :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_divisibility_criteria_l1797_179793


namespace NUMINAMATH_CALUDE_factor_expression_l1797_179711

theorem factor_expression (a m : ℝ) : a * m^2 - a = a * (m - 1) * (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1797_179711


namespace NUMINAMATH_CALUDE_ln_inequality_solution_set_l1797_179748

theorem ln_inequality_solution_set :
  {x : ℝ | Real.log (2 * x - 1) < 0} = Set.Ioo (1/2 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ln_inequality_solution_set_l1797_179748


namespace NUMINAMATH_CALUDE_unique_linear_function_l1797_179702

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem unique_linear_function :
  ∀ a b : ℝ,
  (∀ x y : ℝ, x ∈ [0, 1] → y ∈ [0, 1] → |f a b x + f a b y - x * y| ≤ 1/4) →
  f a b = f (1/2) (-1/8) := by
sorry

end NUMINAMATH_CALUDE_unique_linear_function_l1797_179702


namespace NUMINAMATH_CALUDE_complex_symmetric_division_l1797_179780

/-- Two complex numbers are symmetric about the origin if their sum is zero -/
def symmetric_about_origin (z₁ z₂ : ℂ) : Prop := z₁ + z₂ = 0

theorem complex_symmetric_division (z₁ z₂ : ℂ) 
  (h_sym : symmetric_about_origin z₁ z₂) (h_z₁ : z₁ = 2 - I) : 
  z₁ / z₂ = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_symmetric_division_l1797_179780


namespace NUMINAMATH_CALUDE_hotel_cost_proof_l1797_179774

theorem hotel_cost_proof (initial_share : ℝ) (final_share : ℝ) : 
  (∃ (total_cost : ℝ),
    (initial_share = total_cost / 4) ∧ 
    (final_share = total_cost / 7) ∧
    (initial_share - 15 = final_share)) →
  ∃ (total_cost : ℝ), total_cost = 140 := by
sorry

end NUMINAMATH_CALUDE_hotel_cost_proof_l1797_179774


namespace NUMINAMATH_CALUDE_circle_area_radius_increase_l1797_179713

theorem circle_area_radius_increase : 
  ∀ (r : ℝ) (r' : ℝ), r > 0 → r' > 0 →
  (π * r' ^ 2 = 4 * π * r ^ 2) → 
  (r' - r) / r * 100 = 100 := by
sorry

end NUMINAMATH_CALUDE_circle_area_radius_increase_l1797_179713


namespace NUMINAMATH_CALUDE_problem_solution_l1797_179768

theorem problem_solution (m n : ℝ) (h : |m - n - 5| + (2*m + n - 4)^2 = 0) : 
  3*m + n = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1797_179768


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1797_179736

theorem smallest_multiple_of_6_and_15 :
  ∃ b : ℕ+, (∀ n : ℕ+, 6 ∣ n ∧ 15 ∣ n → b ≤ n) ∧ 6 ∣ b ∧ 15 ∣ b ∧ b = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1797_179736


namespace NUMINAMATH_CALUDE_inequality_solution_l1797_179759

theorem inequality_solution (a : ℝ) (h : a ≠ 0) :
  let f := fun x => x^2 - 5*a*x + 6*a^2
  (∀ x, f x > 0 ↔ (a > 0 ∧ (x < 2*a ∨ x > 3*a)) ∨ (a < 0 ∧ (x < 3*a ∨ x > 2*a))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1797_179759


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_multiple_l1797_179797

theorem smallest_number_divisible_by_multiple (x : ℕ) : x = 34 ↔ 
  (∀ y : ℕ, y < x → ¬(∃ k : ℕ, y - 10 = 2 * k ∧ y - 10 = 6 * k ∧ y - 10 = 12 * k ∧ y - 10 = 24 * k)) ∧
  (∃ k : ℕ, x - 10 = 2 * k ∧ x - 10 = 6 * k ∧ x - 10 = 12 * k ∧ x - 10 = 24 * k) :=
by sorry

#check smallest_number_divisible_by_multiple

end NUMINAMATH_CALUDE_smallest_number_divisible_by_multiple_l1797_179797


namespace NUMINAMATH_CALUDE_min_correct_answers_for_space_talent_l1797_179744

/-- Represents the scoring system and conditions for the space knowledge competition --/
structure SpaceCompetition where
  total_questions : Nat
  correct_points : Int
  wrong_points : Int
  min_score_for_talent : Nat

/-- Calculates the score based on the number of correct answers --/
def calculate_score (comp : SpaceCompetition) (correct_answers : Nat) : Int :=
  (correct_answers : Int) * comp.correct_points + 
  (comp.total_questions - correct_answers : Int) * comp.wrong_points

/-- Theorem stating the minimum number of correct answers needed to be a "Space Talent" --/
theorem min_correct_answers_for_space_talent (comp : SpaceCompetition)
  (h1 : comp.total_questions = 25)
  (h2 : comp.correct_points = 4)
  (h3 : comp.wrong_points = -1)
  (h4 : comp.min_score_for_talent = 90) :
  ∃ n : Nat, n = 23 ∧ 
    (∀ m : Nat, calculate_score comp m ≥ comp.min_score_for_talent → m ≥ n) ∧
    calculate_score comp n ≥ comp.min_score_for_talent :=
  sorry


end NUMINAMATH_CALUDE_min_correct_answers_for_space_talent_l1797_179744


namespace NUMINAMATH_CALUDE_davids_biology_mark_l1797_179716

def marks_english : ℕ := 45
def marks_mathematics : ℕ := 35
def marks_physics : ℕ := 52
def marks_chemistry : ℕ := 47
def average_marks : ℚ := 46.8

theorem davids_biology_mark (marks_biology : ℕ) :
  (marks_english + marks_mathematics + marks_physics + marks_chemistry + marks_biology : ℚ) / 5 = average_marks →
  marks_biology = 55 := by
sorry

end NUMINAMATH_CALUDE_davids_biology_mark_l1797_179716


namespace NUMINAMATH_CALUDE_sams_remaining_money_l1797_179746

/-- Given an initial amount of money, the cost per book, and the number of books bought,
    calculate the remaining money after the purchase. -/
def remaining_money (initial_amount cost_per_book num_books : ℕ) : ℕ :=
  initial_amount - cost_per_book * num_books

/-- Theorem stating that given the specific conditions of Sam's book purchase,
    the remaining money is 16 dollars. -/
theorem sams_remaining_money :
  remaining_money 79 7 9 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sams_remaining_money_l1797_179746


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_zero_l1797_179771

theorem cos_sin_sum_equals_zero :
  Real.cos (5 * Real.pi / 8) * Real.cos (Real.pi / 8) + 
  Real.sin (5 * Real.pi / 8) * Real.sin (Real.pi / 8) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_zero_l1797_179771


namespace NUMINAMATH_CALUDE_line_mb_value_l1797_179737

/-- A line in the 2D plane -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Definition of a point on the line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y = l.m * x + l.b

theorem line_mb_value (l : Line) :
  l.contains 0 (-1) → l.contains 1 1 → l.m * l.b = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_mb_value_l1797_179737


namespace NUMINAMATH_CALUDE_prob_at_least_three_cured_value_l1797_179777

-- Define the probability of success for the drug
def drug_success_rate : ℝ := 0.9

-- Define the number of patients
def num_patients : ℕ := 4

-- Define the minimum number of successes we're interested in
def min_successes : ℕ := 3

-- Define the probability of at least 3 out of 4 patients being cured
def prob_at_least_three_cured : ℝ :=
  1 - (Nat.choose num_patients 0 * drug_success_rate^0 * (1 - drug_success_rate)^4 +
       Nat.choose num_patients 1 * drug_success_rate^1 * (1 - drug_success_rate)^3 +
       Nat.choose num_patients 2 * drug_success_rate^2 * (1 - drug_success_rate)^2)

-- Theorem statement
theorem prob_at_least_three_cured_value :
  prob_at_least_three_cured = 0.9477 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_three_cured_value_l1797_179777


namespace NUMINAMATH_CALUDE_zero_in_interval_l1797_179745

def f (x : ℝ) := -x^3 - 3*x + 5

theorem zero_in_interval :
  (∀ x y, x < y → f x > f y) →  -- f is monotonically decreasing
  Continuous f →               -- f is continuous
  f 1 > 0 →                    -- f(1) > 0
  f 2 < 0 →                    -- f(2) < 0
  ∃ z, z ∈ Set.Ioo 1 2 ∧ f z = 0 := by sorry

end NUMINAMATH_CALUDE_zero_in_interval_l1797_179745


namespace NUMINAMATH_CALUDE_total_ingredients_for_batches_l1797_179728

/-- The amount of flour needed for one batch of cookies, in cups. -/
def flour_per_batch : ℝ := 4

/-- The amount of sugar needed for one batch of cookies, in cups. -/
def sugar_per_batch : ℝ := 1.5

/-- The number of batches we want to make. -/
def num_batches : ℕ := 8

/-- Theorem: The total amount of flour and sugar needed for 8 batches of cookies is 44 cups. -/
theorem total_ingredients_for_batches : 
  (flour_per_batch + sugar_per_batch) * num_batches = 44 := by sorry

end NUMINAMATH_CALUDE_total_ingredients_for_batches_l1797_179728


namespace NUMINAMATH_CALUDE_circle_center_l1797_179754

/-- Given a circle with diameter endpoints (3, -3) and (13, 17), its center is (8, 7) -/
theorem circle_center (Q : Set (ℝ × ℝ)) (p₁ p₂ : ℝ × ℝ) (h₁ : p₁ = (3, -3)) (h₂ : p₂ = (13, 17)) 
    (h₃ : ∀ x ∈ Q, ∃ y ∈ Q, (x.1 - y.1)^2 + (x.2 - y.2)^2 = (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) :
  ∃ c : ℝ × ℝ, c = (8, 7) ∧ ∀ x ∈ Q, (x.1 - c.1)^2 + (x.2 - c.2)^2 = ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_center_l1797_179754


namespace NUMINAMATH_CALUDE_prime_absolute_value_quadratic_l1797_179775

theorem prime_absolute_value_quadratic (a : ℤ) : 
  Nat.Prime (Int.natAbs (a^2 - 3*a - 6)) ↔ a = -1 ∨ a = 4 := by
sorry

end NUMINAMATH_CALUDE_prime_absolute_value_quadratic_l1797_179775


namespace NUMINAMATH_CALUDE_intersection_A_B_l1797_179789

def A : Set ℝ := {x | x^2 - 3*x ≤ 0}
def B : Set ℝ := {1, 2}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1797_179789


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1797_179717

theorem polynomial_division_theorem (x : ℝ) :
  4 * x^4 - 3 * x^3 + 6 * x^2 - 9 * x + 3 = 
  (x + 2) * (4 * x^3 - 11 * x^2 + 28 * x - 65) + 133 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1797_179717


namespace NUMINAMATH_CALUDE_presidency_meeting_arrangements_l1797_179726

/-- The number of schools in the club -/
def num_schools : ℕ := 3

/-- The number of members from each school -/
def members_per_school : ℕ := 6

/-- The number of representatives sent by the host school -/
def host_representatives : ℕ := 3

/-- The number of representatives sent by each non-host school -/
def non_host_representatives : ℕ := 1

/-- The total number of possible ways to arrange the presidency meeting -/
def total_arrangements : ℕ := 2160

theorem presidency_meeting_arrangements :
  (num_schools * (members_per_school.choose host_representatives) *
   (members_per_school.choose non_host_representatives) *
   (members_per_school.choose non_host_representatives)) = total_arrangements :=
by sorry

end NUMINAMATH_CALUDE_presidency_meeting_arrangements_l1797_179726


namespace NUMINAMATH_CALUDE_power_product_cube_l1797_179770

theorem power_product_cube (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_product_cube_l1797_179770


namespace NUMINAMATH_CALUDE_gloria_money_calculation_l1797_179731

def combined_quarters_and_dimes (total_quarters : ℕ) (total_dimes : ℕ) : ℕ :=
  let quarters_put_aside := (2 * total_quarters) / 5
  let remaining_quarters := total_quarters - quarters_put_aside
  remaining_quarters + total_dimes

theorem gloria_money_calculation :
  ∀ (total_quarters : ℕ) (total_dimes : ℕ),
    total_dimes = 5 * total_quarters →
    total_dimes = 350 →
    combined_quarters_and_dimes total_quarters total_dimes = 392 :=
by
  sorry

end NUMINAMATH_CALUDE_gloria_money_calculation_l1797_179731


namespace NUMINAMATH_CALUDE_comic_book_stacks_theorem_l1797_179762

/-- The number of ways to stack comic books -/
def comic_book_stacks (spiderman : ℕ) (archie : ℕ) (garfield : ℕ) : ℕ :=
  (spiderman.factorial * archie.factorial * garfield.factorial * 2)

/-- Theorem: The number of ways to stack 7 Spiderman, 5 Archie, and 4 Garfield comic books,
    with Archie books on top and each series stacked together, is 29,030,400 -/
theorem comic_book_stacks_theorem :
  comic_book_stacks 7 5 4 = 29030400 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_stacks_theorem_l1797_179762


namespace NUMINAMATH_CALUDE_remainder_of_product_mod_17_l1797_179701

theorem remainder_of_product_mod_17 : (157^3 * 193^4) % 17 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_mod_17_l1797_179701


namespace NUMINAMATH_CALUDE_stock_face_value_l1797_179751

/-- Calculates the face value of a stock given the discount rate, brokerage rate, and final cost price. -/
def calculate_face_value (discount_rate : ℚ) (brokerage_rate : ℚ) (final_cost : ℚ) : ℚ :=
  final_cost / ((1 - discount_rate) * (1 + brokerage_rate))

/-- Theorem stating that for a stock with 2% discount, 1/5% brokerage, and Rs 98.2 final cost, the face value is Rs 100. -/
theorem stock_face_value : 
  let discount_rate : ℚ := 2 / 100
  let brokerage_rate : ℚ := 1 / 500
  let final_cost : ℚ := 982 / 10
  calculate_face_value discount_rate brokerage_rate final_cost = 100 := by
  sorry

end NUMINAMATH_CALUDE_stock_face_value_l1797_179751


namespace NUMINAMATH_CALUDE_friendly_match_schemes_l1797_179760

/-- The number of ways to form two teams from teachers and students -/
def formTeams (numTeachers numStudents : ℕ) : ℕ :=
  let teacherCombinations := 1 -- Always select both teachers
  let studentCombinations := numStudents.choose 3
  let studentDistributions := 3 -- Ways to distribute 3 students into 2 teams
  teacherCombinations * studentCombinations * studentDistributions

/-- Theorem stating the number of ways to form teams in the given scenario -/
theorem friendly_match_schemes :
  formTeams 2 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_friendly_match_schemes_l1797_179760


namespace NUMINAMATH_CALUDE_log_inequality_l1797_179739

theorem log_inequality (x : ℝ) : 
  (Real.log (1 + 8 * x^5) / Real.log (1 + x^2) + 
   Real.log (1 + x^2) / Real.log (1 - 3 * x^2 + 16 * x^4) ≤ 
   1 + Real.log (1 + 8 * x^5) / Real.log (1 - 3 * x^2 + 16 * x^4)) ↔ 
  (x ∈ Set.Ioc (-((1/8)^(1/5))) (-1/2) ∪ 
       Set.Ioo (-Real.sqrt 3 / 4) 0 ∪ 
       Set.Ioo 0 (Real.sqrt 3 / 4) ∪ 
       {1/2}) := by sorry

end NUMINAMATH_CALUDE_log_inequality_l1797_179739


namespace NUMINAMATH_CALUDE_smallest_initial_value_l1797_179725

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_initial_value : 
  ∀ n : ℕ, n ≥ 308 → 
  (is_perfect_square (n - 139) ∧ 
   ∀ m : ℕ, m < n → ¬ is_perfect_square (m - 139)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_initial_value_l1797_179725


namespace NUMINAMATH_CALUDE_intersection_M_N_l1797_179791

-- Define set M
def M : Set ℤ := {x | x^2 - x ≤ 0}

-- Define set N
def N : Set ℤ := {x | ∃ n : ℕ, x = 2 * n}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1797_179791


namespace NUMINAMATH_CALUDE_maximum_marks_l1797_179794

/-- 
Given:
1. The passing mark is 36% of the maximum marks.
2. A student gets 130 marks and fails by 14 marks.
Prove that the maximum number of marks is 400.
-/
theorem maximum_marks (passing_percentage : ℚ) (student_marks : ℕ) (failing_margin : ℕ) :
  passing_percentage = 36 / 100 →
  student_marks = 130 →
  failing_margin = 14 →
  ∃ (max_marks : ℕ), max_marks = 400 ∧ 
    (student_marks + failing_margin : ℚ) = passing_percentage * max_marks :=
by sorry

end NUMINAMATH_CALUDE_maximum_marks_l1797_179794


namespace NUMINAMATH_CALUDE_shorter_side_length_l1797_179783

-- Define the rectangle
def Rectangle (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ≥ b

-- Theorem statement
theorem shorter_side_length (a b : ℝ) 
  (h_rect : Rectangle a b) 
  (h_perim : 2 * a + 2 * b = 62) 
  (h_area : a * b = 240) : 
  b = 15 := by
  sorry

end NUMINAMATH_CALUDE_shorter_side_length_l1797_179783
