import Mathlib

namespace abs_of_complex_fraction_l217_21765

open Complex

theorem abs_of_complex_fraction : 
  let z : ℂ := (4 - 2*I) / (1 + I)
  abs z = Real.sqrt 10 := by
  sorry

end abs_of_complex_fraction_l217_21765


namespace traffic_light_probability_l217_21730

theorem traffic_light_probability (m : ℕ) : 
  (35 : ℝ) / (38 + m) > (m : ℝ) / (38 + m) → m = 30 :=
by sorry

end traffic_light_probability_l217_21730


namespace arithmetic_sequence_general_term_l217_21760

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 3 = 7 ∧ 
  a 5 = a 2 + 6

/-- The general term formula for the arithmetic sequence -/
def GeneralTerm (n : ℕ) : ℝ := 2 * n + 1

/-- Theorem stating that the general term formula is correct for the given arithmetic sequence -/
theorem arithmetic_sequence_general_term (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  ∀ n : ℕ, a n = GeneralTerm n :=
sorry

end arithmetic_sequence_general_term_l217_21760


namespace trig_identity_proof_l217_21784

theorem trig_identity_proof (θ : Real) (h : Real.tan θ = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4/5 := by
  sorry

end trig_identity_proof_l217_21784


namespace pure_imaginary_complex_number_l217_21789

/-- Given that z = m²(1+i) - m(3+6i) is a pure imaginary number, 
    prove that m = 3 is the only real solution. -/
theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := m^2 * (1 + Complex.I) - m * (3 + 6 * Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → m = 3 :=
by sorry

end pure_imaginary_complex_number_l217_21789


namespace average_of_a_and_b_l217_21796

theorem average_of_a_and_b (a b : ℝ) : 
  (4 + 5 + 9 + a + b) / 5 = 18 → (a + b) / 2 = 36 := by
  sorry

end average_of_a_and_b_l217_21796


namespace square_of_binomial_constant_l217_21731

theorem square_of_binomial_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 + 30*x + a = (3*x + b)^2) → a = 25 := by
  sorry

end square_of_binomial_constant_l217_21731


namespace set_relationship_l217_21772

theorem set_relationship (A B C : Set α) 
  (h1 : A ∩ B = C) 
  (h2 : B ∩ C = A) : 
  A = C ∧ A ⊆ B := by
sorry

end set_relationship_l217_21772


namespace thirty_divisor_numbers_l217_21716

def is_valid_number (n : ℕ) : Prop :=
  (n % 30 = 0) ∧ (Nat.divisors n).card = 30

def valid_numbers : Finset ℕ := {720, 1200, 1620, 4050, 7500, 11250}

theorem thirty_divisor_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n ∈ valid_numbers := by
  sorry

end thirty_divisor_numbers_l217_21716


namespace alice_has_winning_strategy_l217_21723

/-- Represents a position on the circular table -/
structure Position :=
  (x : ℝ)
  (y : ℝ)

/-- Represents the game state -/
structure GameState :=
  (placedCoins : Set Position)
  (currentPlayer : Bool)  -- true for Alice, false for Bob

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (pos : Position) : Prop :=
  pos ∉ state.placedCoins ∧ pos.x^2 + pos.y^2 ≤ 1

/-- Defines a winning strategy for a player -/
def hasWinningStrategy (player : Bool) : Prop :=
  ∀ (state : GameState), 
    state.currentPlayer = player → 
    ∃ (move : Position), isValidMove state move ∧
      ¬∃ (opponentMove : Position), 
        isValidMove (GameState.mk (state.placedCoins ∪ {move}) (¬player)) opponentMove

/-- The main theorem stating that Alice (the starting player) has a winning strategy -/
theorem alice_has_winning_strategy : 
  hasWinningStrategy true :=
sorry

end alice_has_winning_strategy_l217_21723


namespace cake_serving_solution_l217_21770

/-- Represents the number of cakes served for each type --/
structure CakeCount where
  chocolate : ℕ
  vanilla : ℕ
  strawberry : ℕ

/-- Represents the conditions of the cake serving problem --/
def cake_serving_conditions (c : CakeCount) : Prop :=
  c.chocolate = 2 * c.vanilla ∧
  c.strawberry = c.chocolate / 2 ∧
  c.vanilla = 12 + 18

/-- Theorem stating the correct number of cakes served for each type --/
theorem cake_serving_solution :
  ∃ c : CakeCount, cake_serving_conditions c ∧ 
    c.chocolate = 60 ∧ c.vanilla = 30 ∧ c.strawberry = 30 := by
  sorry

end cake_serving_solution_l217_21770


namespace inequality_solution_set_l217_21776

theorem inequality_solution_set (a : ℝ) (h : a > 0) :
  let S := {x : ℝ | x + 1/x > a}
  (a > 1 → S = {x | 0 < x ∧ x < 1/a} ∪ {x | x > a}) ∧
  (a = 1 → S = {x | x > 0 ∧ x ≠ 1}) ∧
  (0 < a ∧ a < 1 → S = {x | 0 < x ∧ x < a} ∪ {x | x > 1/a}) :=
by sorry

end inequality_solution_set_l217_21776


namespace cricket_average_l217_21702

theorem cricket_average (A : ℝ) : 
  (11 * (A + 4) = 10 * A + 86) → A = 42 := by
  sorry

end cricket_average_l217_21702


namespace pyramid_height_equals_cube_volume_l217_21724

theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) : 
  cube_edge = 5 →
  pyramid_base = 10 →
  cube_edge ^ 3 = (1 / 3) * pyramid_base ^ 2 * pyramid_height →
  pyramid_height = 3.75 := by
sorry

end pyramid_height_equals_cube_volume_l217_21724


namespace cost_of_second_set_l217_21783

/-- The cost of a set of pencils and pens -/
def cost_set (pencil_count : ℕ) (pen_count : ℕ) (pencil_cost : ℚ) (pen_cost : ℚ) : ℚ :=
  pencil_count * pencil_cost + pen_count * pen_cost

/-- The theorem stating that the cost of 4 pencils and 5 pens is 2.00 dollars -/
theorem cost_of_second_set :
  ∃ (pen_cost : ℚ),
    cost_set 4 5 0.1 pen_cost = 2 ∧
    cost_set 4 5 0.1 pen_cost = cost_set 4 5 0.1 ((2 - 4 * 0.1) / 5) :=
by
  sorry

end cost_of_second_set_l217_21783


namespace grid_rectangles_l217_21706

/-- The number of points in each row or column of the grid -/
def gridSize : ℕ := 4

/-- The number of ways to choose 2 items from n items -/
def choose2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of rectangles in a gridSize × gridSize grid -/
def numRectangles : ℕ := (choose2 gridSize) * (choose2 gridSize)

theorem grid_rectangles :
  numRectangles = 36 := by sorry

end grid_rectangles_l217_21706


namespace opposite_of_two_l217_21743

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- State the theorem
theorem opposite_of_two : opposite 2 = -2 := by
  -- The proof goes here
  sorry

end opposite_of_two_l217_21743


namespace monotonic_increase_interval_l217_21718

-- Define the function
def f (x : ℝ) := -x^2

-- State the theorem
theorem monotonic_increase_interval (a b : ℝ) :
  (∀ x y, x < y → x ∈ Set.Iio 0 → y ∈ Set.Iio 0 → f x < f y) ∧
  (∀ x, x ∈ Set.Iic 0 → f x ≤ f 0) ∧
  (∀ x, x > 0 → f x < f 0) :=
sorry

end monotonic_increase_interval_l217_21718


namespace smallest_a_value_l217_21721

theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) 
  (h3 : ∀ x : ℤ, Real.sin (a * (x : ℝ) + b) = Real.sin (17 * (x : ℝ))) : 
  a ≥ 17 ∧ ∃ (a₀ : ℝ), a₀ ≥ 17 ∧ (∀ a' ≥ 17, (∀ x : ℤ, Real.sin (a' * (x : ℝ) + b) = Real.sin (17 * (x : ℝ))) → a' ≥ a₀) :=
sorry

end smallest_a_value_l217_21721


namespace remainder_of_permutation_number_l217_21719

-- Define a type for permutations of numbers from 1 to 2018
def Permutation := Fin 2018 → Fin 2018

-- Define a function that creates a number from a permutation
def numberFromPermutation (p : Permutation) : ℕ := sorry

-- Theorem statement
theorem remainder_of_permutation_number (p : Permutation) :
  numberFromPermutation p % 3 = 0 := by sorry

end remainder_of_permutation_number_l217_21719


namespace ones_digit_of_6_to_34_l217_21715

theorem ones_digit_of_6_to_34 : ∃ k : ℕ, 6^34 = 10 * k + 6 := by
  sorry

end ones_digit_of_6_to_34_l217_21715


namespace exact_one_second_class_probability_l217_21745

/-- The probability of selecting exactly one second-class product when randomly
    selecting three products from a batch of 100 products containing 90 first-class
    and 10 second-class products. -/
theorem exact_one_second_class_probability
  (total : ℕ) (first_class : ℕ) (second_class : ℕ) (selected : ℕ)
  (h_total : total = 100)
  (h_first : first_class = 90)
  (h_second : second_class = 10)
  (h_selected : selected = 3)
  (h_sum : first_class + second_class = total) :
  (Nat.choose first_class 2 * Nat.choose second_class 1) / Nat.choose total selected = 267 / 1078 :=
sorry

end exact_one_second_class_probability_l217_21745


namespace arcade_spending_fraction_l217_21788

def allowance : ℚ := 4.5

theorem arcade_spending_fraction (x : ℚ) 
  (h1 : (2/3) * (1 - x) * allowance = 1.2) : x = 3/5 := by
  sorry

end arcade_spending_fraction_l217_21788


namespace f_monotone_range_l217_21734

/-- Definition of the function f --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

/-- The main theorem --/
theorem f_monotone_range (a : ℝ) :
  (a > 0 ∧ a ≠ 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) →
  a ∈ Set.Icc (3/2) 2 ∧ a < 2 :=
sorry

end f_monotone_range_l217_21734


namespace sin_plus_one_quasi_odd_quasi_odd_to_odd_cubic_quasi_odd_l217_21790

/-- Definition of a quasi-odd function -/
def QuasiOdd (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x, f (a + x) + f (a - x) = 2 * b

/-- Central point of a quasi-odd function -/
def CentralPoint (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x, f (a + x) + f (a - x) = 2 * b

/-- Theorem stating that sin(x) + 1 is a quasi-odd function with central point (0, 1) -/
theorem sin_plus_one_quasi_odd :
  QuasiOdd (fun x ↦ Real.sin x + 1) ∧ CentralPoint (fun x ↦ Real.sin x + 1) 0 1 := by
  sorry

/-- Theorem stating that if f is quasi-odd with central point (a, f(a)),
    then F(x) = f(x+a) - f(a) is odd -/
theorem quasi_odd_to_odd (f : ℝ → ℝ) (a : ℝ) :
  QuasiOdd f ∧ CentralPoint f a (f a) →
  ∀ x, f ((x + a) + a) - f a = -(f ((-x + a) + a) - f a) := by
  sorry

/-- Theorem stating that x^3 - 3x^2 + 6x - 2 is a quasi-odd function with central point (1, 2) -/
theorem cubic_quasi_odd :
  QuasiOdd (fun x ↦ x^3 - 3*x^2 + 6*x - 2) ∧
  CentralPoint (fun x ↦ x^3 - 3*x^2 + 6*x - 2) 1 2 := by
  sorry

end sin_plus_one_quasi_odd_quasi_odd_to_odd_cubic_quasi_odd_l217_21790


namespace balls_in_urns_l217_21711

/-- The number of ways to place k identical balls into n urns with at most one ball per urn -/
def place_balls_limited (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to place k identical balls into n urns with unlimited balls per urn -/
def place_balls_unlimited (n k : ℕ) : ℕ := Nat.choose (k+n-1) (n-1)

theorem balls_in_urns (n k : ℕ) :
  (place_balls_limited n k = Nat.choose n k) ∧
  (place_balls_unlimited n k = Nat.choose (k+n-1) (n-1)) := by
  sorry

end balls_in_urns_l217_21711


namespace unique_prime_product_sum_of_cubes_l217_21766

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- Product of the first n prime numbers -/
def primeProduct (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is expressible as the sum of two positive cubes -/
def isSumOfTwoCubes (n : ℕ) : Prop := ∃ a b : ℕ+, n = a^3 + b^3

theorem unique_prime_product_sum_of_cubes :
  ∀ k : ℕ+, (k = 1 ↔ isSumOfTwoCubes (primeProduct k)) := by sorry

end unique_prime_product_sum_of_cubes_l217_21766


namespace mans_age_fraction_l217_21709

theorem mans_age_fraction (mans_age father_age : ℕ) : 
  father_age = 25 →
  mans_age + 5 = (father_age + 5) / 2 →
  mans_age / father_age = 2 / 5 := by
sorry

end mans_age_fraction_l217_21709


namespace tailor_cut_difference_l217_21714

theorem tailor_cut_difference : 
  let skirt_cut : ℚ := 7/8
  let pants_cut : ℚ := 5/6
  skirt_cut - pants_cut = 1/24 := by sorry

end tailor_cut_difference_l217_21714


namespace grill_run_time_theorem_l217_21755

/-- Represents the time a charcoal grill runs given the rate of coal burning and the amount of coal available -/
def grillRunTime (burnRate : ℕ) (burnTime : ℕ) (bags : ℕ) (coalsPerBag : ℕ) : ℚ :=
  let totalCoals := bags * coalsPerBag
  let minutesPerCycle := burnTime * (totalCoals / burnRate)
  minutesPerCycle / 60

/-- Theorem stating that a grill burning 15 coals every 20 minutes, with 3 bags of 60 coals each, runs for 4 hours -/
theorem grill_run_time_theorem :
  grillRunTime 15 20 3 60 = 4 := by
  sorry

#eval grillRunTime 15 20 3 60

end grill_run_time_theorem_l217_21755


namespace find_m_l217_21761

/-- Given two functions f and g, prove that if 3f(5) = g(5), then m = 10 -/
theorem find_m (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x + m
  let g : ℝ → ℝ := λ x ↦ x^2 - 3*x + 5*m
  3 * f 5 = g 5 → m = 10 := by
sorry

end find_m_l217_21761


namespace compare_x_y_l217_21762

theorem compare_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx4 : x^4 = 2) (hy3 : y^3 = 3) : x < y := by
  sorry

end compare_x_y_l217_21762


namespace line_l_and_AB_are_skew_l217_21727

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relationships
variable (on_plane : Point → Plane → Prop)
variable (on_line : Point → Line → Prop)
variable (line_on_plane : Line → Plane → Prop)
variable (plane_intersection : Plane → Plane → Line)
variable (line_through_points : Point → Point → Line)
variable (skew_lines : Line → Line → Prop)

-- Define the given conditions
variable (α β : Plane)
variable (A B : Point)
variable (l : Line)

-- Theorem statement
theorem line_l_and_AB_are_skew :
  on_plane A α →
  on_plane B β →
  plane_intersection α β = l →
  ¬ on_line A l →
  ¬ on_line B l →
  skew_lines l (line_through_points A B) :=
by sorry

end line_l_and_AB_are_skew_l217_21727


namespace machine_count_l217_21748

theorem machine_count (hours_R hours_S total_hours : ℕ) (h1 : hours_R = 36) (h2 : hours_S = 9) (h3 : total_hours = 12) :
  ∃ (n : ℕ), n > 0 ∧ n * (1 / hours_R + 1 / hours_S) = 1 / total_hours ∧ n = 15 :=
by sorry

end machine_count_l217_21748


namespace total_people_on_large_seats_l217_21710

/-- The number of large seats on the Ferris wheel -/
def large_seats : ℕ := 7

/-- The number of people each large seat can hold -/
def people_per_large_seat : ℕ := 12

/-- Theorem: The total number of people who can ride on large seats is 84 -/
theorem total_people_on_large_seats : large_seats * people_per_large_seat = 84 := by
  sorry

end total_people_on_large_seats_l217_21710


namespace farm_animals_relation_l217_21704

/-- Given a farm with pigs, cows, and goats, prove the relationship between the number of goats and cows -/
theorem farm_animals_relation (pigs cows goats : ℕ) : 
  pigs = 10 →
  cows = 2 * pigs - 3 →
  pigs + cows + goats = 50 →
  goats = cows + 6 := by
sorry

end farm_animals_relation_l217_21704


namespace median_in_70_74_l217_21763

/-- Represents a score interval with its lower bound and student count -/
structure ScoreInterval :=
  (lower_bound : ℕ)
  (student_count : ℕ)

/-- The list of score intervals -/
def score_intervals : List ScoreInterval :=
  [⟨85, 10⟩, ⟨80, 15⟩, ⟨75, 20⟩, ⟨70, 25⟩, ⟨65, 15⟩, ⟨60, 10⟩, ⟨55, 5⟩]

/-- The total number of students -/
def total_students : ℕ := 100

/-- Find the interval containing the median score -/
def median_interval (intervals : List ScoreInterval) (total : ℕ) : Option ScoreInterval :=
  sorry

/-- Theorem: The interval containing the median score is 70-74 -/
theorem median_in_70_74 :
  median_interval score_intervals total_students = some ⟨70, 25⟩ :=
sorry

end median_in_70_74_l217_21763


namespace tangent_line_to_circle_l217_21778

theorem tangent_line_to_circle (m : ℝ) : 
  (∃ (x y : ℝ), x - Real.sqrt 3 * y + m = 0 ∧ x^2 + y^2 - 2*y - 2 = 0) →
  (m = -Real.sqrt 3 ∨ m = 3 * Real.sqrt 3) :=
sorry

end tangent_line_to_circle_l217_21778


namespace power_expression_simplification_l217_21795

theorem power_expression_simplification :
  (1 : ℚ) / ((-8^2)^4) * (-8)^9 = -8 := by sorry

end power_expression_simplification_l217_21795


namespace quadratic_rewrite_l217_21729

theorem quadratic_rewrite (a b c : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 48 * x + 56 = (a * x + b)^2 + c) →
  a * b = -24 := by
  sorry

end quadratic_rewrite_l217_21729


namespace gcd_of_polynomial_and_multiple_l217_21779

theorem gcd_of_polynomial_and_multiple (x : ℤ) (h : ∃ k : ℤ, x = 46200 * k) :
  let f := fun (x : ℤ) => (3*x + 5) * (5*x + 3) * (11*x + 6) * (x + 11)
  Int.gcd (f x) x = 990 := by
sorry

end gcd_of_polynomial_and_multiple_l217_21779


namespace unique_modular_congruence_l217_21735

theorem unique_modular_congruence :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 5 ∧ n ≡ -7458 [ZMOD 6] := by
  sorry

end unique_modular_congruence_l217_21735


namespace hexagonal_tiles_count_l217_21797

/-- Represents the number of edges for each type of tile -/
def edges_per_tile : Fin 3 → ℕ
| 0 => 3  -- triangular
| 1 => 4  -- square
| 2 => 6  -- hexagonal

/-- The total number of tiles in the box -/
def total_tiles : ℕ := 35

/-- The total number of edges from all tiles -/
def total_edges : ℕ := 128

theorem hexagonal_tiles_count :
  ∃ (a b c : ℕ),
    a + b + c = total_tiles ∧
    3 * a + 4 * b + 6 * c = total_edges ∧
    c = 6 :=
sorry

end hexagonal_tiles_count_l217_21797


namespace largest_digit_divisible_by_six_l217_21793

/-- The largest single-digit number N such that 5678N is divisible by 6 is 4 -/
theorem largest_digit_divisible_by_six : 
  (∀ N : ℕ, N ≤ 9 → 56780 + N ≤ 56789 → (56780 + N) % 6 = 0 → N ≤ 4) ∧ 
  (56784 % 6 = 0) := by
  sorry

end largest_digit_divisible_by_six_l217_21793


namespace quadratic_set_single_element_l217_21777

theorem quadratic_set_single_element (k : ℝ) :
  (∃! x : ℝ, k * x^2 + 4 * x + 4 = 0) → k = 0 ∨ k = 1 := by
  sorry

end quadratic_set_single_element_l217_21777


namespace x4_plus_81_factorization_l217_21712

theorem x4_plus_81_factorization (x : ℝ) :
  x^4 + 81 = (x^2 - 3*x + 4.5) * (x^2 + 3*x + 4.5) := by
  sorry

end x4_plus_81_factorization_l217_21712


namespace total_ways_eq_64_l217_21744

/-- The number of sports available to choose from -/
def num_sports : ℕ := 4

/-- The number of people choosing sports -/
def num_people : ℕ := 3

/-- The total number of different ways to choose sports -/
def total_ways : ℕ := num_sports ^ num_people

/-- Theorem stating that the total number of ways to choose sports is 64 -/
theorem total_ways_eq_64 : total_ways = 64 := by
  sorry

end total_ways_eq_64_l217_21744


namespace quadratic_inequality_solution_set_l217_21746

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 - a*x - b < 0}) : 
  {x : ℝ | b*x^2 - a*x - 1 > 0} = Set.Ioo (-1/2) (-1/3) := by
  sorry

end quadratic_inequality_solution_set_l217_21746


namespace parabola_y_axis_intersection_l217_21720

def parabola (x y : ℝ) : Prop := y = -(x + 2)^2 + 6

theorem parabola_y_axis_intersection :
  ∃ (y : ℝ), parabola 0 y ∧ y = 2 := by
  sorry

end parabola_y_axis_intersection_l217_21720


namespace unique_solution_l217_21768

/-- The solution set of the inequality |ax - 2| < 3 with respect to x -/
def SolutionSet (a : ℝ) : Set ℝ :=
  {x : ℝ | |a * x - 2| < 3}

/-- The given solution set -/
def GivenSet : Set ℝ :=
  {x : ℝ | -5/3 < x ∧ x < 1/3}

/-- The theorem stating that a = -3 is the unique value satisfying the conditions -/
theorem unique_solution : ∃! a : ℝ, SolutionSet a = GivenSet :=
sorry

end unique_solution_l217_21768


namespace quadratic_equation_conversion_l217_21759

theorem quadratic_equation_conversion :
  ∃ (a b c : ℝ), a = 1 ∧ b = -5 ∧ c = 3 ∧
  ∀ x, (x - 1)^2 = 3*x - 2 ↔ a*x^2 + b*x + c = 0 :=
by sorry

end quadratic_equation_conversion_l217_21759


namespace complex_expression_evaluation_l217_21739

theorem complex_expression_evaluation (z : ℂ) (h : z = 1 + I) :
  2 / z + z^2 = 1 + I := by sorry

end complex_expression_evaluation_l217_21739


namespace expression_evaluation_l217_21740

theorem expression_evaluation (a b : ℤ) (ha : a = -1) (hb : b = 4) :
  (a + b)^2 - 2*a*(a - b) + (a + 2*b)*(a - 2*b) = -64 := by
  sorry

end expression_evaluation_l217_21740


namespace smallest_perimeter_consecutive_odd_triangle_l217_21738

/-- Three consecutive odd integers -/
def ConsecutiveOddIntegers (a b c : ℕ) : Prop :=
  (∃ k : ℕ, a = 2 * k + 1 ∧ b = 2 * k + 3 ∧ c = 2 * k + 5)

/-- Triangle inequality -/
def IsValidTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

/-- Perimeter of a triangle -/
def Perimeter (a b c : ℕ) : ℕ := a + b + c

/-- The smallest possible perimeter of a triangle with consecutive odd integer side lengths is 15 -/
theorem smallest_perimeter_consecutive_odd_triangle :
  ∀ a b c : ℕ,
  ConsecutiveOddIntegers a b c →
  IsValidTriangle a b c →
  Perimeter a b c ≥ 15 :=
sorry

end smallest_perimeter_consecutive_odd_triangle_l217_21738


namespace distance_between_vertices_l217_21700

-- Define the two quadratic functions
def f (x : ℝ) : ℝ := x^2 - 4*x + 5
def g (x : ℝ) : ℝ := x^2 + 6*x + 20

-- Define the vertices of the two graphs
def C : ℝ × ℝ := (2, f 2)
def D : ℝ × ℝ := (-3, g (-3))

-- Theorem statement
theorem distance_between_vertices : 
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 5 * Real.sqrt 5 := by
  sorry

end distance_between_vertices_l217_21700


namespace product_mod_25_l217_21749

theorem product_mod_25 : 68 * 97 * 113 ≡ 23 [ZMOD 25] := by sorry

end product_mod_25_l217_21749


namespace perpendicular_line_plane_counterexample_l217_21794

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop := sorry

/-- A line is perpendicular to another line -/
def perp_line (l1 l2 : Line3D) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perp_plane (l : Line3D) (p : Plane) : Prop := sorry

/-- A line is within a plane -/
def line_in_plane (l : Line3D) (p : Plane) : Prop := sorry

theorem perpendicular_line_plane_counterexample :
  ∃ (l : Line3D) (p : Plane) (l1 l2 : Line3D),
    line_in_plane l1 p ∧
    line_in_plane l2 p ∧
    intersect l1 l2 ∧
    perp_line l l1 ∧
    perp_line l l2 ∧
    ¬(perp_plane l p) := by sorry

end perpendicular_line_plane_counterexample_l217_21794


namespace wendy_shoes_theorem_l217_21773

/-- The number of pairs of shoes Wendy gave away -/
def shoes_given_away : ℕ := 14

/-- The number of pairs of shoes Wendy kept -/
def shoes_kept : ℕ := 19

/-- The total number of pairs of shoes Wendy had -/
def total_shoes : ℕ := shoes_given_away + shoes_kept

theorem wendy_shoes_theorem : total_shoes = 33 := by
  sorry

end wendy_shoes_theorem_l217_21773


namespace exam_candidates_count_l217_21764

theorem exam_candidates_count :
  ∀ (T : ℕ),
    (T : ℚ) * (49 / 100) = T * (percent_failed_english : ℚ) →
    (T : ℚ) * (36 / 100) = T * (percent_failed_hindi : ℚ) →
    (T : ℚ) * (15 / 100) = T * (percent_failed_both : ℚ) →
    (T : ℚ) * ((51 / 100) - (15 / 100)) = 630 →
    T = 1750 :=
by
  sorry

end exam_candidates_count_l217_21764


namespace coffee_syrup_combinations_l217_21705

theorem coffee_syrup_combinations :
  let coffee_types : ℕ := 5
  let syrup_types : ℕ := 7
  let syrup_choices : ℕ := 3
  coffee_types * (syrup_types.choose syrup_choices) = 175 :=
by sorry

end coffee_syrup_combinations_l217_21705


namespace norris_september_savings_l217_21774

/-- The amount of money Norris saved in September -/
def september_savings : ℕ := sorry

/-- The amount of money Norris saved in October -/
def october_savings : ℕ := 25

/-- The amount of money Norris saved in November -/
def november_savings : ℕ := 31

/-- The amount of money Norris spent on an online game -/
def game_cost : ℕ := 75

/-- The amount of money Norris has left -/
def money_left : ℕ := 10

/-- Theorem stating that Norris saved $29 in September -/
theorem norris_september_savings :
  september_savings = 29 :=
by
  sorry

end norris_september_savings_l217_21774


namespace parabola_point_relationship_l217_21753

/-- Parabola function -/
def f (x m : ℝ) : ℝ := x^2 - 4*x - m

theorem parabola_point_relationship (m : ℝ) :
  let y₁ := f 2 m
  let y₂ := f (-3) m
  let y₃ := f (-1) m
  y₁ < y₃ ∧ y₃ < y₂ := by sorry

end parabola_point_relationship_l217_21753


namespace yw_approx_6_32_l217_21717

/-- Triangle XYZ with W on XY -/
structure TriangleXYZW where
  /-- Point X -/
  X : ℝ × ℝ
  /-- Point Y -/
  Y : ℝ × ℝ
  /-- Point Z -/
  Z : ℝ × ℝ
  /-- Point W on XY -/
  W : ℝ × ℝ
  /-- XZ = YZ = 10 -/
  xz_eq_yz : dist X Z = dist Y Z ∧ dist X Z = 10
  /-- W is on XY -/
  w_on_xy : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ W = (1 - t) • X + t • Y
  /-- XW = 5 -/
  xw_eq_5 : dist X W = 5
  /-- ZW = 6 -/
  zw_eq_6 : dist Z W = 6

/-- The length of YW is approximately 6.32 -/
theorem yw_approx_6_32 (t : TriangleXYZW) : 
  abs (dist t.Y t.W - 6.32) < 0.01 := by
  sorry

end yw_approx_6_32_l217_21717


namespace air_quality_consecutive_good_days_l217_21785

/-- Represents the air quality index for a given day -/
def AirQualityIndex := ℕ → ℝ

/-- Determines if the air quality is good for a given index -/
def is_good (index : ℝ) : Prop := index < 100

/-- Determines if two consecutive days have good air quality -/
def consecutive_good_days (aqi : AirQualityIndex) (day : ℕ) : Prop :=
  is_good (aqi day) ∧ is_good (aqi (day + 1))

/-- The air quality index for the 10 days -/
axiom aqi : AirQualityIndex

/-- The theorem to prove -/
theorem air_quality_consecutive_good_days :
  (consecutive_good_days aqi 1 ∧ consecutive_good_days aqi 5) ∧
  (∀ d : ℕ, d ≠ 1 ∧ d ≠ 5 → ¬consecutive_good_days aqi d) :=
sorry

end air_quality_consecutive_good_days_l217_21785


namespace derivative_ln_over_x_l217_21791

open Real

theorem derivative_ln_over_x (x : ℝ) (h : x > 0) :
  deriv (λ x => (log x) / x) x = (1 - log x) / x^2 := by
  sorry

end derivative_ln_over_x_l217_21791


namespace polynomial_value_at_two_l217_21786

/-- A polynomial with non-negative integer coefficients -/
def NonNegIntPoly (f : ℕ → ℕ) : Prop :=
  ∃ n : ℕ, ∀ k : ℕ, k > n → f k = 0

theorem polynomial_value_at_two
  (f : ℕ → ℕ)
  (h_poly : NonNegIntPoly f)
  (h_one : f 1 = 6)
  (h_seven : f 7 = 3438) :
  f 2 = 43 := by
sorry

end polynomial_value_at_two_l217_21786


namespace binary_11111_equals_31_l217_21751

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11111_equals_31 :
  binary_to_decimal [true, true, true, true, true] = 31 := by
  sorry

end binary_11111_equals_31_l217_21751


namespace two_pairs_probability_l217_21769

-- Define the total number of socks
def total_socks : ℕ := 10

-- Define the number of colors
def num_colors : ℕ := 5

-- Define the number of socks per color
def socks_per_color : ℕ := 2

-- Define the number of socks drawn
def socks_drawn : ℕ := 4

-- Define the probability of drawing two pairs of different colors
def prob_two_pairs : ℚ := 1 / 21

-- Theorem statement
theorem two_pairs_probability :
  (Nat.choose num_colors 2) / (Nat.choose total_socks socks_drawn) = prob_two_pairs :=
sorry

end two_pairs_probability_l217_21769


namespace expected_value_of_special_die_l217_21781

def die_faces : List ℕ := [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144]

theorem expected_value_of_special_die :
  (List.sum die_faces) / (List.length die_faces : ℚ) = 650 / 12 := by
  sorry

end expected_value_of_special_die_l217_21781


namespace middle_number_is_nine_l217_21775

theorem middle_number_is_nine (x y z : ℕ) (h1 : x < y) (h2 : y < z)
  (h3 : x + y = 15) (h4 : x + z = 23) (h5 : y + z = 26) : y = 9 := by
  sorry

end middle_number_is_nine_l217_21775


namespace gcd_360_128_l217_21742

theorem gcd_360_128 : Nat.gcd 360 128 = 8 := by sorry

end gcd_360_128_l217_21742


namespace company_workforce_l217_21754

theorem company_workforce (initial_employees : ℕ) 
  (h1 : (60 : ℚ) / 100 * initial_employees = (55 : ℚ) / 100 * (initial_employees + 30)) :
  initial_employees + 30 = 360 := by
sorry

end company_workforce_l217_21754


namespace max_value_of_f_l217_21767

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem max_value_of_f :
  ∃ (c : ℝ), c > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≤ 1 / Real.exp 1 := by
  sorry

end max_value_of_f_l217_21767


namespace pizza_pieces_l217_21798

theorem pizza_pieces (total_people : Nat) (half_eaters : Nat) (three_quarter_eaters : Nat) (pieces_left : Nat) :
  total_people = 4 →
  half_eaters = 2 →
  three_quarter_eaters = 2 →
  pieces_left = 6 →
  ∃ (pieces_per_pizza : Nat),
    pieces_per_pizza * (half_eaters * (1/2) + three_quarter_eaters * (1/4)) = pieces_left ∧
    pieces_per_pizza = 4 := by
  sorry

end pizza_pieces_l217_21798


namespace rectangle_area_l217_21787

/-- Represents a square in the rectangle --/
structure Square where
  sideLength : ℝ
  area : ℝ
  area_def : area = sideLength ^ 2

/-- The rectangle XYZW with its squares --/
structure Rectangle where
  smallSquares : Fin 3 → Square
  largeSquare : Square
  smallSquaresEqual : ∀ i j : Fin 3, (smallSquares i).sideLength = (smallSquares j).sideLength
  smallSquareArea : ∀ i : Fin 3, (smallSquares i).area = 4
  largeSquareSideLength : largeSquare.sideLength = 2 * (smallSquares 0).sideLength
  noOverlap : True  -- This condition is simplified as it's hard to represent geometrically

/-- The theorem to prove --/
theorem rectangle_area (rect : Rectangle) : 
  (3 * (rect.smallSquares 0).area + rect.largeSquare.area : ℝ) = 28 := by
  sorry

end rectangle_area_l217_21787


namespace train_speed_l217_21722

/-- The speed of a train given its length, time to cross a walking man, and the man's speed --/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) : 
  train_length = 700 →
  crossing_time = 41.9966402687785 →
  man_speed_kmh = 3 →
  ∃ (train_speed_kmh : ℝ), abs (train_speed_kmh - 63.0036) < 0.0001 := by
  sorry


end train_speed_l217_21722


namespace exists_sequence_satisfying_conditions_l217_21799

/-- The number of distinct prime factors shared by two positive integers -/
def d (m n : ℕ+) : ℕ := sorry

/-- The existence of a sequence satisfying the given conditions -/
theorem exists_sequence_satisfying_conditions :
  ∃ (a : ℕ+ → ℕ+),
    (a 1 ≥ 2018^2018) ∧
    (∀ m n, m ≤ n → a m ≤ a n) ∧
    (∀ m n, m ≠ n → d m n = d (a m) (a n)) :=
  sorry

end exists_sequence_satisfying_conditions_l217_21799


namespace intersection_point_m_value_l217_21756

theorem intersection_point_m_value (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ y = x + 1 ∧ y = -x) → m = 5 := by
  sorry

end intersection_point_m_value_l217_21756


namespace first_player_wins_6x8_l217_21728

/-- Represents a chocolate bar game --/
structure ChocolateGame where
  rows : Nat
  cols : Nat

/-- Calculates the number of moves required to completely break the chocolate bar --/
def totalMoves (game : ChocolateGame) : Nat :=
  game.rows * game.cols - 1

/-- Determines the winner of the game --/
def firstPlayerWins (game : ChocolateGame) : Prop :=
  Odd (totalMoves game)

/-- Theorem stating that the first player wins in a 6x8 chocolate bar game --/
theorem first_player_wins_6x8 :
  firstPlayerWins { rows := 6, cols := 8 } := by
  sorry

end first_player_wins_6x8_l217_21728


namespace race_probability_l217_21747

theorem race_probability (p_x p_y p_z : ℚ) : 
  p_x = 1/8 →
  p_y = 1/12 →
  p_x + p_y + p_z = 375/1000 →
  p_z = 1/6 := by
  sorry

end race_probability_l217_21747


namespace max_value_implies_a_equals_one_l217_21737

def f (a x : ℝ) : ℝ := -x^2 + 2*a*x + a - 1

theorem max_value_implies_a_equals_one :
  ∀ a : ℝ,
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≤ 1) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ f a x = 1) →
  a = 1 :=
by sorry

end max_value_implies_a_equals_one_l217_21737


namespace choose_3_from_12_l217_21758

theorem choose_3_from_12 : Nat.choose 12 3 = 220 := by
  sorry

end choose_3_from_12_l217_21758


namespace snails_removed_l217_21771

def original_snails : ℕ := 11760
def remaining_snails : ℕ := 8278

theorem snails_removed : original_snails - remaining_snails = 3482 := by
  sorry

end snails_removed_l217_21771


namespace evaluate_expression_l217_21726

theorem evaluate_expression (b y : ℤ) (h : y = b + 9) : y - b + 5 = 14 := by
  sorry

end evaluate_expression_l217_21726


namespace triangle_special_case_l217_21792

/-- Given a triangle with sides a, b, and c satisfying (a + b + c)(a + b - c) = 4ab,
    the angle opposite side c is 0 or 2π. -/
theorem triangle_special_case (a b c : ℝ) (h : (a + b + c) * (a + b - c) = 4 * a * b) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  C = 0 ∨ C = 2 * Real.pi := by
  sorry

end triangle_special_case_l217_21792


namespace jimmy_garden_servings_l217_21750

/-- Represents the number of servings produced by a single plant of each vegetable type -/
structure ServingsPerPlant where
  carrot : ℕ
  corn : ℕ
  greenBean : ℕ
  tomato : ℕ
  zucchini : ℕ
  bellPepper : ℕ

/-- Represents the number of plants for each vegetable type in Jimmy's garden -/
structure PlantsPerPlot where
  carrot : ℕ
  corn : ℕ
  greenBean : ℕ
  tomato : ℕ
  zucchini : ℕ
  bellPepper : ℕ

/-- Calculates the total number of servings in Jimmy's garden -/
def totalServings (s : ServingsPerPlant) (p : PlantsPerPlot) : ℕ :=
  s.carrot * p.carrot +
  s.corn * p.corn +
  s.greenBean * p.greenBean +
  s.tomato * p.tomato +
  s.zucchini * p.zucchini +
  s.bellPepper * p.bellPepper

/-- Theorem stating that Jimmy's garden produces 963 servings of vegetables -/
theorem jimmy_garden_servings 
  (s : ServingsPerPlant)
  (p : PlantsPerPlot)
  (h1 : s.carrot = 4)
  (h2 : s.corn = 5 * s.carrot)
  (h3 : s.greenBean = s.corn / 2)
  (h4 : s.tomato = s.carrot + 3)
  (h5 : s.zucchini = 4 * s.greenBean)
  (h6 : s.bellPepper = s.corn - 2)
  (h7 : p.greenBean = 10)
  (h8 : p.carrot = 8)
  (h9 : p.corn = 12)
  (h10 : p.tomato = 15)
  (h11 : p.zucchini = 9)
  (h12 : p.bellPepper = 7) :
  totalServings s p = 963 := by
  sorry


end jimmy_garden_servings_l217_21750


namespace bus_speed_on_national_road_l217_21725

/-- The speed of a bus on the original national road, given specific conditions about a new highway --/
theorem bus_speed_on_national_road :
  ∀ (x : ℝ),
    (200 : ℝ) / (x + 45) = (220 : ℝ) / x / 2 →
    x = 55 := by
  sorry

end bus_speed_on_national_road_l217_21725


namespace trapezoid_perimeter_l217_21707

/-- A trapezoid with sides A, B, C, and D -/
structure Trapezoid where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.A + t.B + t.C + t.D

/-- Theorem: The perimeter of the given trapezoid ABCD is 180 units -/
theorem trapezoid_perimeter : 
  ∀ (ABCD : Trapezoid), 
  ABCD.B = 50 → 
  ABCD.A = 30 → 
  ABCD.C = 25 → 
  ABCD.D = 75 → 
  perimeter ABCD = 180 := by
  sorry


end trapezoid_perimeter_l217_21707


namespace remainder_problem_l217_21703

theorem remainder_problem (N : ℕ) (h1 : N = 184) (h2 : N % 15 = 4) : N % 13 = 2 := by
  sorry

end remainder_problem_l217_21703


namespace lcm_from_product_and_hcf_l217_21752

theorem lcm_from_product_and_hcf (A B : ℕ+) :
  A * B = 62216 →
  Nat.gcd A B = 22 →
  Nat.lcm A B = 2828 := by
sorry

end lcm_from_product_and_hcf_l217_21752


namespace price_reduction_sales_increase_l217_21701

theorem price_reduction_sales_increase (price_reduction : Real) 
  (sales_increase : Real) (net_sale_increase : Real) :
  price_reduction = 20 → 
  net_sale_increase = 44 → 
  (1 - price_reduction / 100) * (1 + sales_increase / 100) = 1 + net_sale_increase / 100 →
  sales_increase = 80 := by
sorry

end price_reduction_sales_increase_l217_21701


namespace arithmetic_sequence_third_term_l217_21708

/-- Given an arithmetic sequence with first term 2 and sum of second and fourth terms 10,
    the third term is 5. -/
theorem arithmetic_sequence_third_term (a d : ℚ) : 
  a = 2 ∧ (a + d) + (a + 3*d) = 10 → a + 2*d = 5 := by
  sorry

end arithmetic_sequence_third_term_l217_21708


namespace roberto_outfits_l217_21741

/-- The number of different outfits Roberto can create -/
def number_of_outfits (trousers shirts jackets : ℕ) : ℕ :=
  trousers * shirts * jackets

/-- Theorem stating the number of outfits Roberto can create -/
theorem roberto_outfits :
  let trousers : ℕ := 5
  let shirts : ℕ := 8
  let jackets : ℕ := 4
  number_of_outfits trousers shirts jackets = 160 := by
  sorry

end roberto_outfits_l217_21741


namespace ratio_from_mean_ratio_l217_21780

theorem ratio_from_mean_ratio {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (a + b) / (2 * Real.sqrt (a * b)) = 25 / 24 →
  a / b = 16 / 9 ∨ a / b = 9 / 16 := by
sorry

end ratio_from_mean_ratio_l217_21780


namespace unique_modular_congruence_l217_21736

theorem unique_modular_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 13 ∧ n ≡ 1729 [ZMOD 13] := by
  sorry

end unique_modular_congruence_l217_21736


namespace weight_gain_proof_l217_21782

/-- Calculates the final weight after muscle and fat gain -/
def final_weight (initial_weight : ℝ) (muscle_gain_percent : ℝ) (fat_gain_ratio : ℝ) : ℝ :=
  let muscle_gain := initial_weight * muscle_gain_percent
  let fat_gain := muscle_gain * fat_gain_ratio
  initial_weight + muscle_gain + fat_gain

/-- Proves that given the specified conditions, the final weight is 150 kg -/
theorem weight_gain_proof :
  final_weight 120 0.2 0.25 = 150 := by
  sorry

end weight_gain_proof_l217_21782


namespace equation_solution_l217_21732

theorem equation_solution (x y : ℝ) : 
  x / (x - 2) = (y^3 + 3*y - 2) / (y^3 + 3*y - 5) → 
  x = (2*y^3 + 6*y - 4) / 3 := by
sorry

end equation_solution_l217_21732


namespace max_rope_piece_length_l217_21757

theorem max_rope_piece_length : Nat.gcd 60 (Nat.gcd 75 90) = 15 := by
  sorry

end max_rope_piece_length_l217_21757


namespace expression_evaluation_l217_21733

theorem expression_evaluation (x y z : ℝ) :
  let P := x + y + z
  let Q := x - y - z
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = (x^2 - y^2 - 2*y*z - z^2) / (x*(y + z)) :=
by sorry

end expression_evaluation_l217_21733


namespace quadratic_cubic_relation_l217_21713

theorem quadratic_cubic_relation (x₀ : ℝ) (h : x₀^2 + x₀ - 1 = 0) :
  x₀^3 + 2*x₀^2 + 2 = 3 := by
  sorry

end quadratic_cubic_relation_l217_21713
