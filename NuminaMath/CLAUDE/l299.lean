import Mathlib

namespace imaginary_part_of_complex_fraction_l299_29980

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (2 + Complex.I) / (1 + 3 * Complex.I)
  Complex.im z = -1/2 := by sorry

end imaginary_part_of_complex_fraction_l299_29980


namespace intersection_of_A_and_B_l299_29998

-- Define the sets A and B
def A : Set ℝ := { x | 2 < x ∧ x ≤ 4 }
def B : Set ℝ := { x | x^2 - 2*x < 3 }

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = { x : ℝ | 2 < x ∧ x < 3 } := by sorry

end intersection_of_A_and_B_l299_29998


namespace intersection_points_form_diameter_l299_29911

/-- Two circles in a plane -/
structure TwoCircles where
  S₁ : Set (ℝ × ℝ)
  S₂ : Set (ℝ × ℝ)

/-- Intersection points of the two circles -/
def intersection_points (tc : TwoCircles) : Set (ℝ × ℝ) :=
  tc.S₁ ∩ tc.S₂

/-- Tangent line to a circle at a point -/
def tangent_line (S : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Radius of a circle -/
def radius (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- Inner arc of a circle -/
def inner_arc (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- Line passing through two points -/
def line_through (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Diameter of a circle -/
def is_diameter (S : Set (ℝ × ℝ)) (p q : ℝ × ℝ) : Prop := sorry

/-- Main theorem -/
theorem intersection_points_form_diameter
  (tc : TwoCircles)
  (A B : ℝ × ℝ)
  (h_AB : A ∈ intersection_points tc ∧ B ∈ intersection_points tc)
  (h_tangent : tangent_line tc.S₁ A = radius tc.S₂ ∧ tangent_line tc.S₁ B = radius tc.S₂)
  (C : ℝ × ℝ)
  (h_C : C ∈ inner_arc tc.S₁)
  (K L : ℝ × ℝ)
  (h_K : K ∈ line_through A C ∩ tc.S₂)
  (h_L : L ∈ line_through B C ∩ tc.S₂) :
  is_diameter tc.S₂ K L := by sorry

end intersection_points_form_diameter_l299_29911


namespace smallest_c_for_inverse_l299_29995

/-- The function f(x) = (x+1)^2 - 3 -/
def f (x : ℝ) : ℝ := (x + 1)^2 - 3

/-- The theorem stating that -1 is the smallest value of c for which f has an inverse on [c,∞) -/
theorem smallest_c_for_inverse :
  ∀ c : ℝ, (∀ x y, x ∈ Set.Ici c → y ∈ Set.Ici c → f x = f y → x = y) ↔ c ≥ -1 :=
sorry

end smallest_c_for_inverse_l299_29995


namespace alice_winning_strategy_l299_29972

/-- Represents the game state with n objects and maximum removal of m objects per turn -/
structure GameState where
  n : ℕ  -- Number of objects in the pile
  m : ℕ  -- Maximum number of objects that can be removed per turn

/-- Predicate to check if a player has a winning strategy -/
def has_winning_strategy (state : GameState) : Prop :=
  ¬(state.n + 1 ∣ state.m)

/-- Theorem stating the condition for Alice to have a winning strategy -/
theorem alice_winning_strategy (state : GameState) :
  has_winning_strategy state ↔ ¬(state.n + 1 ∣ state.m) :=
sorry

end alice_winning_strategy_l299_29972


namespace x_plus_reciprocal_geq_two_l299_29970

theorem x_plus_reciprocal_geq_two (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 := by
  sorry

end x_plus_reciprocal_geq_two_l299_29970


namespace annual_interest_rate_is_eight_percent_l299_29932

def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

theorem annual_interest_rate_is_eight_percent 
  (principal : ℝ) 
  (interest : ℝ) 
  (total : ℝ) 
  (time : ℕ) 
  (h1 : principal + interest = total)
  (h2 : interest = 2828.80)
  (h3 : total = 19828.80)
  (h4 : time = 2) :
  compound_interest principal 0.08 time = interest := by
  sorry

#check annual_interest_rate_is_eight_percent

end annual_interest_rate_is_eight_percent_l299_29932


namespace solution_set_of_inequality_l299_29994

def f (x : ℝ) : ℝ := 3 - 2*x

theorem solution_set_of_inequality (x : ℝ) :
  (|f (x + 1) + 2| ≤ 3) ↔ (0 ≤ x ∧ x ≤ 3) :=
by sorry

end solution_set_of_inequality_l299_29994


namespace average_entry_exit_time_is_200_l299_29982

/-- Represents the position and movement of a car and storm -/
structure CarStormSystem where
  carSpeed : ℝ
  stormRadius : ℝ
  stormSpeedSouth : ℝ
  stormSpeedEast : ℝ
  initialNorthDistance : ℝ

/-- Calculates the average of the times when the car enters and exits the storm -/
def averageEntryExitTime (system : CarStormSystem) : ℝ :=
  200

/-- Theorem stating that the average entry/exit time is 200 minutes -/
theorem average_entry_exit_time_is_200 (system : CarStormSystem) 
  (h1 : system.carSpeed = 1)
  (h2 : system.stormRadius = 60)
  (h3 : system.stormSpeedSouth = 3/4)
  (h4 : system.stormSpeedEast = 1/4)
  (h5 : system.initialNorthDistance = 150) :
  averageEntryExitTime system = 200 := by
  sorry

end average_entry_exit_time_is_200_l299_29982


namespace four_integers_average_l299_29913

theorem four_integers_average (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a + b + c + d : ℚ) / 4 = 5 →
  ∀ w x y z : ℕ+, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
    (w + x + y + z : ℚ) / 4 = 5 →
    (d - a : ℤ) ≥ (z - w : ℤ) →
  ((b : ℚ) + c) / 2 = 5/2 := by
sorry

end four_integers_average_l299_29913


namespace completing_square_transformation_l299_29921

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) :=
by
  sorry

end completing_square_transformation_l299_29921


namespace optimal_removal_l299_29977

-- Define the grid
inductive Square
| a | b | c | d | e | f | g | j | k | l | m | n

-- Define the initial shape
def initial_shape : Set Square :=
  {Square.a, Square.b, Square.c, Square.d, Square.e, Square.f, Square.g, Square.j, Square.k, Square.l, Square.m, Square.n}

-- Define adjacency relation
def adjacent : Square → Square → Prop := sorry

-- Define connectivity
def is_connected (shape : Set Square) : Prop := sorry

-- Define perimeter calculation
def perimeter (shape : Set Square) : ℕ := sorry

-- Define the set of all possible pairs of squares to remove
def removable_pairs : Set (Square × Square) := sorry

theorem optimal_removal :
  ∀ (pair : Square × Square),
    pair ∈ removable_pairs →
    is_connected (initial_shape \ {pair.1, pair.2}) →
    perimeter (initial_shape \ {pair.1, pair.2}) ≤ 
    max (perimeter (initial_shape \ {Square.d, Square.k}))
        (perimeter (initial_shape \ {Square.e, Square.k})) :=
sorry

end optimal_removal_l299_29977


namespace sufficient_not_necessary_condition_l299_29968

theorem sufficient_not_necessary_condition : 
  (∀ x : ℝ, x > 5 → x^2 - 4*x - 5 > 0) ∧ 
  (∃ x : ℝ, x^2 - 4*x - 5 > 0 ∧ ¬(x > 5)) := by
  sorry

end sufficient_not_necessary_condition_l299_29968


namespace ellipse_intersection_theorem_l299_29936

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the lines
def line (k m x y : ℝ) : Prop := y = k * x + m

-- Define the chord length
def chord_length (k m : ℝ) : ℝ := 
  2 * Real.sqrt 2 * (Real.sqrt (1 + k^2) * Real.sqrt (2 * k^2 - m^2 + 1)) / (1 + 2 * k^2)

-- Define the area of the quadrilateral
def quad_area (k m₁ : ℝ) : ℝ := 
  4 * Real.sqrt 2 * Real.sqrt ((2 * k^2 - m₁^2 + 1) * m₁^2) / (1 + 2 * k^2)

-- State the theorem
theorem ellipse_intersection_theorem (k m₁ m₂ : ℝ) 
  (h₁ : m₁ ≠ m₂) 
  (h₂ : chord_length k m₁ = chord_length k m₂) : 
  m₁ + m₂ = 0 ∧ 
  ∀ m, quad_area k m ≤ 2 * Real.sqrt 2 := by
  sorry

end

end ellipse_intersection_theorem_l299_29936


namespace largest_odd_integer_with_coprime_primes_l299_29907

theorem largest_odd_integer_with_coprime_primes : ∃ (n : ℕ), 
  n = 105 ∧ 
  n % 2 = 1 ∧
  (∀ k : ℕ, 1 < k → k < n → k % 2 = 1 → Nat.gcd k n = 1 → Nat.Prime k) ∧
  (∀ m : ℕ, m > n → m % 2 = 1 → 
    ∃ k : ℕ, 1 < k ∧ k < m ∧ k % 2 = 1 ∧ Nat.gcd k m = 1 ∧ ¬Nat.Prime k) :=
by sorry

end largest_odd_integer_with_coprime_primes_l299_29907


namespace average_equation_solution_l299_29990

theorem average_equation_solution (x : ℝ) : 
  ((x + 8) + (5 * x + 4) + (2 * x + 7)) / 3 = 3 * x - 10 → x = 49 := by
  sorry

end average_equation_solution_l299_29990


namespace infinite_valid_moves_l299_29966

-- Define the grid
def InfiniteSquareGrid := ℤ × ℤ

-- Define the directions
inductive Direction
| North
| South
| East
| West

-- Define a car
structure Car where
  position : InfiniteSquareGrid
  direction : Direction

-- Define the state of the grid
structure GridState where
  cars : Finset Car

-- Define a valid move
def validMove (state : GridState) (car : Car) : Prop :=
  car ∈ state.cars ∧
  (∀ other : Car, other ∈ state.cars → other.position ≠ car.position) ∧
  (∀ other : Car, other ∈ state.cars → 
    match car.direction with
    | Direction.North => other.position ≠ (car.position.1, car.position.2 + 1)
    | Direction.South => other.position ≠ (car.position.1, car.position.2 - 1)
    | Direction.East => other.position ≠ (car.position.1 + 1, car.position.2)
    | Direction.West => other.position ≠ (car.position.1 - 1, car.position.2)
  ) ∧
  (∀ other : Car, other ∈ state.cars →
    (car.direction = Direction.East ∧ other.direction = Direction.West → car.position.1 < other.position.1) ∧
    (car.direction = Direction.West ∧ other.direction = Direction.East → car.position.1 > other.position.1) ∧
    (car.direction = Direction.North ∧ other.direction = Direction.South → car.position.2 < other.position.2) ∧
    (car.direction = Direction.South ∧ other.direction = Direction.North → car.position.2 > other.position.2))

-- Define the theorem
theorem infinite_valid_moves (initialState : GridState) : 
  ∃ (moveSequence : ℕ → Car), 
    (∀ n : ℕ, validMove initialState (moveSequence n)) ∧
    (∀ car : Car, car ∈ initialState.cars → ∀ k : ℕ, ∃ n > k, moveSequence n = car) :=
sorry

end infinite_valid_moves_l299_29966


namespace turns_for_both_buckets_l299_29908

/-- Represents the capacity of bucket Q -/
def capacity_Q : ℝ := 1

/-- Represents the capacity of bucket P -/
def capacity_P : ℝ := 3 * capacity_Q

/-- Represents the number of turns it takes bucket P to fill the drum -/
def turns_P : ℕ := 80

/-- Represents the capacity of the drum -/
def drum_capacity : ℝ := turns_P * capacity_P

/-- Represents the combined capacity of buckets P and Q -/
def combined_capacity : ℝ := capacity_P + capacity_Q

/-- 
Proves that the number of turns it takes for both buckets P and Q together 
to fill the drum is 60, given the conditions stated in the problem.
-/
theorem turns_for_both_buckets : 
  (drum_capacity / combined_capacity : ℝ) = 60 := by sorry

end turns_for_both_buckets_l299_29908


namespace polynomial_value_theorem_l299_29939

/-- A polynomial that takes integer values at integer points -/
def IntegerPolynomial := ℤ → ℤ

/-- Proposition: If a polynomial with integer coefficients takes the value 2 
    at three distinct integer points, it cannot take the value 3 at any integer point -/
theorem polynomial_value_theorem (P : IntegerPolynomial) 
  (h1 : ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ P a = 2 ∧ P b = 2 ∧ P c = 2) :
  ¬∃ x : ℤ, P x = 3 := by
  sorry

end polynomial_value_theorem_l299_29939


namespace greatest_x_value_l299_29958

theorem greatest_x_value (x : ℝ) :
  (x^2 - x - 90) / (x - 9) = 2 / (x + 6) →
  x ≤ -8 + Real.sqrt 6 :=
by sorry

end greatest_x_value_l299_29958


namespace max_f_and_min_sum_l299_29967

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| - |2*x + 4|

-- Theorem statement
theorem max_f_and_min_sum :
  (∃ m : ℝ, ∀ x : ℝ, f x ≤ m ∧ ∃ x₀ : ℝ, f x₀ = m) ∧
  (∃ m : ℝ, m = 4 ∧
   ∀ a b : ℝ, a > 0 → b > 0 → a + 2*b = m →
   2/a + 9/b ≥ 8 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = m ∧ 2/a₀ + 9/b₀ = 8) :=
by sorry

end max_f_and_min_sum_l299_29967


namespace base_conversion_sum_l299_29919

-- Define the base conversions
def base_8_to_10 (n : ℕ) : ℕ := 
  2 * (8^2) + 5 * (8^1) + 4 * (8^0)

def base_3_to_10 (n : ℕ) : ℕ := 
  1 * (3^1) + 3 * (3^0)

def base_7_to_10 (n : ℕ) : ℕ := 
  2 * (7^2) + 3 * (7^1) + 2 * (7^0)

def base_5_to_10 (n : ℕ) : ℕ := 
  3 * (5^1) + 2 * (5^0)

-- Theorem statement
theorem base_conversion_sum :
  (base_8_to_10 254 / base_3_to_10 13) + (base_7_to_10 232 / base_5_to_10 32) = 35 := by
  sorry

end base_conversion_sum_l299_29919


namespace four_numbers_theorem_l299_29918

def satisfies_condition (x y z t : ℝ) : Prop :=
  x + y * z * t = 2 ∧
  y + x * z * t = 2 ∧
  z + x * y * t = 2 ∧
  t + x * y * z = 2

theorem four_numbers_theorem :
  ∀ x y z t : ℝ,
    satisfies_condition x y z t ↔
      ((x = 1 ∧ y = 1 ∧ z = 1 ∧ t = 1) ∨
       (x = -1 ∧ y = -1 ∧ z = -1 ∧ t = 3) ∨
       (x = -1 ∧ y = -1 ∧ z = 3 ∧ t = -1) ∨
       (x = -1 ∧ y = 3 ∧ z = -1 ∧ t = -1) ∨
       (x = 3 ∧ y = -1 ∧ z = -1 ∧ t = -1)) :=
by sorry

end four_numbers_theorem_l299_29918


namespace base_10_to_6_conversion_l299_29991

/-- Converts a base-10 number to its base-6 representation --/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 6) ((m % 6) :: acc)
    go n []

/-- Converts a list of digits in base-6 to a natural number --/
def fromBase6 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 6 * acc + d) 0

theorem base_10_to_6_conversion :
  fromBase6 (toBase6 110) = 110 :=
sorry

end base_10_to_6_conversion_l299_29991


namespace not_necessarily_parallel_lines_l299_29915

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersect_planes : Plane → Plane → Line → Prop)

-- State the theorem
theorem not_necessarily_parallel_lines 
  (α β : Plane) (m n : Line) 
  (h1 : α ≠ β) 
  (h2 : m ≠ n) 
  (h3 : parallel_line_plane m α) 
  (h4 : intersect_planes α β n) : 
  ¬ (∀ m n, parallel_lines m n) :=
sorry

end not_necessarily_parallel_lines_l299_29915


namespace geometric_sequence_second_term_approximate_value_of_b_l299_29964

theorem geometric_sequence_second_term (b : ℝ) : 
  b > 0 ∧ 
  (∃ r : ℝ, 30 * r = b ∧ b * r = 9/4) → 
  b^2 = 270/4 :=
by sorry

theorem approximate_value_of_b : 
  ∃ b : ℝ, b > 0 ∧ 
  (∃ r : ℝ, 30 * r = b ∧ b * r = 9/4) ∧ 
  (b^2 = 270/4) ∧ 
  (abs (b - 8.215838362) < 0.000000001) :=
by sorry

end geometric_sequence_second_term_approximate_value_of_b_l299_29964


namespace basketball_game_result_l299_29944

/-- Represents a basketball team --/
structure Team where
  initial_score : ℕ
  baskets_scored : ℕ
  basket_value : ℕ

/-- Calculates the final score of a team --/
def final_score (team : Team) : ℕ := team.initial_score + team.baskets_scored * team.basket_value

/-- The basketball game scenario --/
def basketball_game_scenario : Prop :=
  let hornets : Team := { initial_score := 86, baskets_scored := 2, basket_value := 2 }
  let fireflies : Team := { initial_score := 74, baskets_scored := 7, basket_value := 3 }
  final_score fireflies - final_score hornets = 5

/-- Theorem stating the result of the basketball game --/
theorem basketball_game_result : basketball_game_scenario := by sorry

end basketball_game_result_l299_29944


namespace expression_evaluation_l299_29996

theorem expression_evaluation : 
  let f (x : ℝ) := (x - 1) / (x + 1)
  let expr (x : ℝ) := (f x + 1) / (f x - 1)
  expr 2 = -2 := by sorry

end expression_evaluation_l299_29996


namespace min_communication_size_l299_29920

/-- Represents a set of cards with positive numbers -/
def CardSet := Finset ℕ+

/-- The number of cards -/
def n : ℕ := 100

/-- A function that takes a set of cards and returns a set of communicated values -/
def communicate (cards : CardSet) : Finset ℕ := sorry

/-- Predicate to check if a set of communicated values uniquely determines the original card set -/
def uniquely_determines (comms : Finset ℕ) (cards : CardSet) : Prop := sorry

theorem min_communication_size :
  ∀ (cards : CardSet),
  (cards.card = n) →
  ∃ (comms : Finset ℕ),
    (communicate cards = comms) ∧
    (uniquely_determines comms cards) ∧
    (comms.card = n + 1) ∧
    (∀ (comms' : Finset ℕ),
      (communicate cards = comms') ∧
      (uniquely_determines comms' cards) →
      (comms'.card ≥ n + 1)) :=
by sorry

end min_communication_size_l299_29920


namespace vector_at_negative_one_l299_29957

/-- A line in 3D space parameterized by t -/
structure ParametricLine where
  -- The vector on the line at t = 0
  v0 : ℝ × ℝ × ℝ
  -- The vector on the line at t = 1
  v1 : ℝ × ℝ × ℝ

/-- The vector on the line at a given t -/
def vectorAtT (line : ParametricLine) (t : ℝ) : ℝ × ℝ × ℝ :=
  let (x0, y0, z0) := line.v0
  let (x1, y1, z1) := line.v1
  (x0 + t * (x1 - x0), y0 + t * (y1 - y0), z0 + t * (z1 - z0))

theorem vector_at_negative_one (line : ParametricLine) 
  (h1 : line.v0 = (2, 6, 16)) 
  (h2 : line.v1 = (1, 1, 4)) : 
  vectorAtT line (-1) = (3, 11, 28) := by
  sorry

end vector_at_negative_one_l299_29957


namespace root_equations_imply_m_n_values_l299_29971

theorem root_equations_imply_m_n_values (m n : ℝ) : 
  (∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧
    ((r1 + m) * (r1 + n) * (r1 + 8)) / ((r1 + 2)^2) = 0 ∧
    ((r2 + m) * (r2 + n) * (r2 + 8)) / ((r2 + 2)^2) = 0 ∧
    ((r3 + m) * (r3 + n) * (r3 + 8)) / ((r3 + 2)^2) = 0) →
  (∃! (r : ℝ), ((r + 2*m) * (r + 4) * (r + 10)) / ((r + n) * (r + 8)) = 0) →
  m = 1 ∧ n = 4 ∧ 50*m + n = 54 := by
sorry

end root_equations_imply_m_n_values_l299_29971


namespace quadratic_equation_root_and_q_l299_29974

theorem quadratic_equation_root_and_q (p q : ℝ) : 
  (∃ x : ℂ, 5 * x^2 + p * x + q = 0 ∧ x = 3 + 2*I) →
  q = 65 := by
  sorry

end quadratic_equation_root_and_q_l299_29974


namespace max_min_difference_l299_29973

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x - a

-- Define the interval
def interval : Set ℝ := Set.Icc 0 3

-- State the theorem
theorem max_min_difference (a : ℝ) :
  ∃ (M N : ℝ),
    (∀ x ∈ interval, f a x ≤ M) ∧
    (∃ x ∈ interval, f a x = M) ∧
    (∀ x ∈ interval, N ≤ f a x) ∧
    (∃ x ∈ interval, f a x = N) ∧
    M - N = 18 :=
sorry

end max_min_difference_l299_29973


namespace count_negative_rationals_l299_29941

/-- The number of negative rational numbers in the given set is 3 -/
theorem count_negative_rationals : 
  let S : Finset ℚ := {-|(-2:ℚ)|, -(2:ℚ)^2019, -(-1:ℚ), 0, -(-2:ℚ)^2}
  (S.filter (λ x => x < 0)).card = 3 := by sorry

end count_negative_rationals_l299_29941


namespace prob_heads_tails_tails_l299_29962

/-- The probability of getting heads on a fair coin flip -/
def prob_heads : ℚ := 1/2

/-- The probability of getting tails on a fair coin flip -/
def prob_tails : ℚ := 1/2

/-- The number of coin flips -/
def num_flips : ℕ := 3

/-- Theorem: The probability of getting heads on the first flip and tails on the last two flips
    when flipping a fair coin three times is 1/8 -/
theorem prob_heads_tails_tails : prob_heads * prob_tails * prob_tails = 1/8 := by
  sorry

end prob_heads_tails_tails_l299_29962


namespace hyperbola_eccentricity_l299_29951

/-- Given a hyperbola with equation x^2 - ty^2 = 3t and focal distance 6, its eccentricity is √6/2 -/
theorem hyperbola_eccentricity (t : ℝ) :
  (∃ (x y : ℝ), x^2 - t*y^2 = 3*t) →  -- Hyperbola equation
  (∃ (c : ℝ), c = 3) →  -- Focal distance is 6, so half of it (c) is 3
  (∃ (e : ℝ), e = (Real.sqrt 6) / 2 ∧ e = (Real.sqrt (t + 1))) -- Eccentricity
  :=
by sorry

end hyperbola_eccentricity_l299_29951


namespace no_prime_polynomial_l299_29986

-- Define a polynomial with integer coefficients
def IntPolynomial := ℕ → ℤ

-- Define what it means for a polynomial to be constant
def IsConstant (P : IntPolynomial) : Prop :=
  ∀ n m : ℕ, P n = P m

-- Define primality
def IsPrime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, 1 < m → m < n → ¬(n % m = 0)

-- The main theorem
theorem no_prime_polynomial :
  ¬∃ (P : IntPolynomial),
    (¬IsConstant P) ∧
    (∀ n : ℕ, n > 0 → IsPrime (P n)) :=
sorry

end no_prime_polynomial_l299_29986


namespace gabrielle_cardinals_count_l299_29992

/-- Represents the number of birds of each type seen by a person -/
structure BirdCount where
  robins : ℕ
  cardinals : ℕ
  bluejays : ℕ

/-- Calculates the total number of birds seen -/
def total_birds (count : BirdCount) : ℕ :=
  count.robins + count.cardinals + count.bluejays

/-- The bird counts for Chase and Gabrielle -/
def chase : BirdCount := { robins := 2, cardinals := 5, bluejays := 3 }
def gabrielle : BirdCount := { robins := 5, cardinals := 0, bluejays := 3 }

theorem gabrielle_cardinals_count : gabrielle.cardinals = 4 := by
  have h1 : total_birds chase = 10 := by sorry
  have h2 : total_birds gabrielle = (120 * total_birds chase) / 100 := by sorry
  have h3 : gabrielle.robins + gabrielle.bluejays = 8 := by sorry
  sorry

end gabrielle_cardinals_count_l299_29992


namespace gadget_production_75_workers_4_hours_l299_29903

/-- Represents the production rate of a worker per hour -/
structure ProductionRate :=
  (gadgets : ℝ)
  (gizmos : ℝ)

/-- Calculates the total production given workers, hours, and rate -/
def totalProduction (workers : ℕ) (hours : ℕ) (rate : ProductionRate) : ProductionRate :=
  { gadgets := workers * hours * rate.gadgets,
    gizmos := workers * hours * rate.gizmos }

theorem gadget_production_75_workers_4_hours 
  (rate1 : ProductionRate)
  (rate2 : ProductionRate)
  (h1 : totalProduction 150 1 rate1 = { gadgets := 450, gizmos := 300 })
  (h2 : totalProduction 100 2 rate2 = { gadgets := 400, gizmos := 500 }) :
  (totalProduction 75 4 rate2).gadgets = 600 := by
sorry

end gadget_production_75_workers_4_hours_l299_29903


namespace power_of_two_properties_l299_29925

theorem power_of_two_properties (n : ℕ) :
  (∃ k : ℕ, n = 3 * k ↔ 7 ∣ (2^n - 1)) ∧
  ¬(7 ∣ (2^n + 1)) := by
  sorry

end power_of_two_properties_l299_29925


namespace propositions_equivalent_l299_29909

-- Define P as a set
variable (P : Set α)

-- Define the original proposition
def original_prop (a b : α) : Prop :=
  a ∈ P → b ∉ P

-- Define the equivalent proposition (option D)
def equivalent_prop (a b : α) : Prop :=
  b ∈ P → a ∉ P

-- Theorem stating the equivalence of the two propositions
theorem propositions_equivalent (a b : α) :
  original_prop P a b ↔ equivalent_prop P a b :=
sorry

end propositions_equivalent_l299_29909


namespace distribute_five_balls_two_boxes_l299_29961

/-- The number of ways to distribute n indistinguishable balls into 2 indistinguishable boxes -/
def distribute_balls (n : ℕ) : ℕ :=
  (n + 2) / 2

/-- Theorem: There are 3 ways to distribute 5 indistinguishable balls into 2 indistinguishable boxes -/
theorem distribute_five_balls_two_boxes : distribute_balls 5 = 3 := by
  sorry

end distribute_five_balls_two_boxes_l299_29961


namespace total_money_from_stone_sale_l299_29902

def number_of_stones : ℕ := 8
def price_per_stone : ℕ := 1785

theorem total_money_from_stone_sale : number_of_stones * price_per_stone = 14280 := by
  sorry

end total_money_from_stone_sale_l299_29902


namespace systematic_sampling_interval_l299_29976

/-- Calculates the sampling interval for systematic sampling -/
def samplingInterval (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  let adjustedPopSize := (populationSize / sampleSize) * sampleSize
  adjustedPopSize / sampleSize

/-- Proves that the sampling interval is 10 for the given problem -/
theorem systematic_sampling_interval :
  samplingInterval 123 12 = 10 := by
  sorry

#eval samplingInterval 123 12

end systematic_sampling_interval_l299_29976


namespace dental_removal_fraction_l299_29975

theorem dental_removal_fraction :
  ∀ (x : ℚ),
  (∃ (t₁ t₂ t₃ t₄ : ℕ),
    t₁ + t₂ + t₃ + t₄ = 4 ∧  -- Four adults
    (∀ i, t₁ ≤ i ∧ i ≤ t₄ → 32 = 32) ∧  -- Each adult has 32 teeth
    x * 32 + 3/8 * 32 + 1/2 * 32 + 4 = 40)  -- Total teeth removed
  → x = 1/4 := by
sorry

end dental_removal_fraction_l299_29975


namespace lines_parallel_perpendicular_l299_29981

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := x + (1 + a) * y + a - 1 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0

-- Define parallel and perpendicular conditions
def parallel (a : ℝ) : Prop := a = 1
def perpendicular (a : ℝ) : Prop := a = -2/3

-- Theorem statement
theorem lines_parallel_perpendicular :
  (∀ a : ℝ, (∀ x y : ℝ, l₁ a x y ∧ l₂ a x y) → parallel a) ∧
  (∀ a : ℝ, (∀ x y : ℝ, l₁ a x y ∧ l₂ a x y) → perpendicular a) :=
sorry

end lines_parallel_perpendicular_l299_29981


namespace no_solution_equation_l299_29930

theorem no_solution_equation : ¬ ∃ (x y z : ℤ), x^3 + y^6 = 7*z + 3 := by
  sorry

end no_solution_equation_l299_29930


namespace evelyn_bottle_caps_l299_29948

def initial_caps : ℕ := 18
def found_caps : ℕ := 63

theorem evelyn_bottle_caps : initial_caps + found_caps = 81 := by
  sorry

end evelyn_bottle_caps_l299_29948


namespace election_winner_percentage_l299_29965

theorem election_winner_percentage (total_votes : ℕ) (majority : ℕ) (winner_percentage : ℚ) : 
  total_votes = 435 →
  majority = 174 →
  winner_percentage = 70 / 100 →
  (winner_percentage * total_votes : ℚ) - ((1 - winner_percentage) * total_votes : ℚ) = majority :=
by sorry

end election_winner_percentage_l299_29965


namespace trenton_earnings_goal_l299_29917

/-- Trenton's weekly earnings calculation --/
def weekly_earnings (base_pay : ℝ) (commission_rate : ℝ) (sales : ℝ) : ℝ :=
  base_pay + commission_rate * sales

theorem trenton_earnings_goal :
  let base_pay : ℝ := 190
  let commission_rate : ℝ := 0.04
  let min_sales : ℝ := 7750
  let goal : ℝ := 500
  weekly_earnings base_pay commission_rate min_sales = goal := by
sorry

end trenton_earnings_goal_l299_29917


namespace tristan_study_hours_l299_29946

/-- Represents the number of hours Tristan studies each day of the week -/
structure StudyHours where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ
  saturday : ℝ
  sunday : ℝ

/-- Theorem stating the number of hours Tristan studies from Wednesday to Friday -/
theorem tristan_study_hours (h : StudyHours) : 
  h.monday = 4 ∧ 
  h.tuesday = 2 * h.monday ∧ 
  h.wednesday = h.thursday ∧ 
  h.thursday = h.friday ∧ 
  h.monday + h.tuesday + h.wednesday + h.thursday + h.friday + h.saturday + h.sunday = 25 ∧ 
  h.saturday = h.sunday → 
  h.wednesday = 13/3 := by
sorry

#eval 13/3  -- To show the result is approximately 4.33

end tristan_study_hours_l299_29946


namespace root_equality_implies_b_equals_three_l299_29945

theorem root_equality_implies_b_equals_three
  (a b c N : ℝ)
  (ha : a > 1)
  (hb : b > 1)
  (hc : c > 1)
  (hN : N > 1)
  (h_int_a : ∃ k : ℤ, a = k)
  (h_int_b : ∃ k : ℤ, b = k)
  (h_int_c : ∃ k : ℤ, c = k)
  (h_eq : (N * (N^(1/b))^(1/c))^(1/a) = N^(25/36)) :
  b = 3 := by
sorry

end root_equality_implies_b_equals_three_l299_29945


namespace men_per_table_l299_29942

theorem men_per_table (num_tables : ℕ) (women_per_table : ℕ) (total_customers : ℕ) :
  num_tables = 5 →
  women_per_table = 5 →
  total_customers = 40 →
  (total_customers - num_tables * women_per_table) / num_tables = 3 :=
by sorry

end men_per_table_l299_29942


namespace infinite_pairs_and_odd_sum_l299_29910

theorem infinite_pairs_and_odd_sum :
  (∃ (S : Set (ℕ × ℕ)), Set.Infinite S ∧
    ∀ (p : ℕ × ℕ), p ∈ S →
      (⌊(4 + 2 * Real.sqrt 3) * p.1⌋ : ℤ) = ⌊(4 - 2 * Real.sqrt 3) * p.2⌋) ∧
  (∀ (m n : ℕ), 
    (⌊(4 + 2 * Real.sqrt 3) * m⌋ : ℤ) = ⌊(4 - 2 * Real.sqrt 3) * n⌋ →
    Odd (m + n)) :=
by sorry

end infinite_pairs_and_odd_sum_l299_29910


namespace trigonometric_equation_solution_l299_29959

theorem trigonometric_equation_solution (x : ℝ) :
  2 * Real.cos x - 5 * Real.sin x = 3 →
  (3 * Real.sin x + 2 * Real.cos x = (-21 + 13 * Real.sqrt 145) / 58) ∨
  (3 * Real.sin x + 2 * Real.cos x = (-21 - 13 * Real.sqrt 145) / 58) :=
by sorry

end trigonometric_equation_solution_l299_29959


namespace baduk_stone_difference_l299_29906

theorem baduk_stone_difference (total : ℕ) (white : ℕ) (h1 : total = 928) (h2 : white = 713) :
  white - (total - white) = 498 := by
  sorry

end baduk_stone_difference_l299_29906


namespace barbara_initial_candies_l299_29979

/-- The number of candies Barbara bought -/
def candies_bought : ℕ := 18

/-- The total number of candies Barbara has after buying more -/
def total_candies : ℕ := 27

/-- The number of candies Barbara had initially -/
def initial_candies : ℕ := total_candies - candies_bought

theorem barbara_initial_candies : initial_candies = 9 := by
  sorry

end barbara_initial_candies_l299_29979


namespace max_floor_length_l299_29987

/-- Represents a rectangular tile with length and width in centimeters. -/
structure Tile where
  length : ℕ
  width : ℕ

/-- Represents a rectangular floor with length and width in centimeters. -/
structure Floor where
  length : ℕ
  width : ℕ

/-- Checks if a given number of tiles can fit on the floor without overlap or overshooting. -/
def canFitTiles (t : Tile) (f : Floor) (n : ℕ) : Prop :=
  (f.length % t.length = 0 ∧ f.width ≥ t.width ∧ (f.length / t.length) * (f.width / t.width) ≥ n) ∨
  (f.length % t.width = 0 ∧ f.width ≥ t.length ∧ (f.length / t.width) * (f.width / t.length) ≥ n)

theorem max_floor_length (t : Tile) (maxTiles : ℕ) :
  t.length = 50 →
  t.width = 40 →
  maxTiles = 9 →
  ∃ (f : Floor), canFitTiles t f maxTiles ∧
    ∀ (f' : Floor), canFitTiles t f' maxTiles → f'.length ≤ f.length ∧ f.length = 450 := by
  sorry

end max_floor_length_l299_29987


namespace ratio_transformation_l299_29953

theorem ratio_transformation (x : ℚ) : 
  (2 + x) / (3 + x) = 4 / 5 → x = 2 := by
  sorry

end ratio_transformation_l299_29953


namespace perimeter_increase_theorem_l299_29938

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool

/-- Result of moving sides of a polygon outward -/
structure TransformedPolygon where
  original : ConvexPolygon
  distance : Real

/-- Perimeter of a polygon -/
def perimeter (p : ConvexPolygon) : Real := sorry

/-- Perimeter increase after transformation -/
def perimeter_increase (tp : TransformedPolygon) : Real :=
  perimeter (ConvexPolygon.mk tp.original.vertices true) - perimeter tp.original

/-- Theorem: Perimeter increase is greater than 30 cm when sides are moved by 5 cm -/
theorem perimeter_increase_theorem (p : ConvexPolygon) :
  perimeter_increase (TransformedPolygon.mk p 5) > 30 := by sorry

end perimeter_increase_theorem_l299_29938


namespace total_animals_count_l299_29984

def animal_count : ℕ → ℕ → ℕ → ℕ
| snakes, arctic_foxes, leopards =>
  let bee_eaters := 12 * leopards
  let cheetahs := snakes / 3
  let alligators := 2 * (arctic_foxes + leopards)
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators

theorem total_animals_count :
  animal_count 100 80 20 = 673 :=
by sorry

end total_animals_count_l299_29984


namespace binary_110011_equals_51_l299_29950

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end binary_110011_equals_51_l299_29950


namespace parabola_parameter_values_l299_29926

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- Define point M
def M : ℝ × ℝ := (1, 1)

-- Define the distance from M to the directrix
def distance_to_directrix (a : ℝ) : ℝ := 2

theorem parabola_parameter_values :
  ∃ (a : ℝ), (parabola a (M.1) = M.2) ∧ 
             (distance_to_directrix a = 2) ∧ 
             (a = 1/4 ∨ a = -1/12) :=
by sorry

end parabola_parameter_values_l299_29926


namespace root_product_theorem_l299_29963

theorem root_product_theorem (y₁ y₂ y₃ y₄ y₅ : ℂ) : 
  (y₁^5 - 3*y₁^3 + 2 = 0) →
  (y₂^5 - 3*y₂^3 + 2 = 0) →
  (y₃^5 - 3*y₃^3 + 2 = 0) →
  (y₄^5 - 3*y₄^3 + 2 = 0) →
  (y₅^5 - 3*y₅^3 + 2 = 0) →
  ((y₁^2 - 3) * (y₂^2 - 3) * (y₃^2 - 3) * (y₄^2 - 3) * (y₅^2 - 3) = -32) :=
by sorry

end root_product_theorem_l299_29963


namespace odd_digits_base4_233_l299_29904

/-- Counts the number of odd digits in the base-4 representation of a natural number. -/
def countOddDigitsBase4 (n : ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-4 representation of 233 is 2. -/
theorem odd_digits_base4_233 : countOddDigitsBase4 233 = 2 := by
  sorry

end odd_digits_base4_233_l299_29904


namespace value_added_to_fraction_l299_29999

theorem value_added_to_fraction (x y : ℝ) : 
  x = 8 → 0.75 * x + y = 8 → y = 2 := by sorry

end value_added_to_fraction_l299_29999


namespace problem_solution_l299_29923

theorem problem_solution (x y : ℝ) : (x + 3)^2 + Real.sqrt (2 - y) = 0 → (x + y)^2021 = -1 := by
  sorry

end problem_solution_l299_29923


namespace cases_in_1990_l299_29935

/-- Calculates the number of cases in a given year assuming linear decrease --/
def casesInYear (initialCases : ℕ) (finalCases : ℕ) (initialYear : ℕ) (finalYear : ℕ) (targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let yearlyDecrease := totalDecrease / totalYears
  let yearsFromInitial := targetYear - initialYear
  initialCases - (yearlyDecrease * yearsFromInitial)

/-- The number of cases in 1990 given linear decrease from 1970 to 2000 --/
theorem cases_in_1990 : 
  casesInYear 600000 200 1970 2000 1990 = 200133 := by
  sorry

end cases_in_1990_l299_29935


namespace quadratic_points_comparison_l299_29955

theorem quadratic_points_comparison (c : ℝ) (y₁ y₂ : ℝ) 
  (h1 : y₁ = (-1)^2 - 6*(-1) + c) 
  (h2 : y₂ = 2^2 - 6*2 + c) : 
  y₁ > y₂ := by sorry

end quadratic_points_comparison_l299_29955


namespace man_walking_speed_l299_29947

/-- The speed of a man walking alongside a train --/
theorem man_walking_speed (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) :
  train_length = 900 →
  crossing_time = 53.99568034557235 →
  train_speed_kmh = 63 →
  ∃ (man_speed : ℝ), abs (man_speed - 0.832) < 0.001 :=
by
  sorry

end man_walking_speed_l299_29947


namespace no_natural_solutions_for_cubic_equation_l299_29940

theorem no_natural_solutions_for_cubic_equation :
  ¬∃ (x y z : ℕ), x^3 + 2*y^3 = 4*z^3 := by
  sorry

end no_natural_solutions_for_cubic_equation_l299_29940


namespace special_integer_pairs_l299_29914

theorem special_integer_pairs (a b : ℕ+) :
  (∃ (p : ℕ) (k : ℕ), Prime p ∧ a^2 + b + 1 = p^k) →
  (a^2 + b + 1) ∣ (b^2 - a^3 - 1) →
  ¬((a^2 + b + 1) ∣ (a + b - 1)^2) →
  ∃ (s : ℕ), s ≥ 2 ∧ a = 2^s ∧ b = 2^(2*s) - 1 :=
by sorry

end special_integer_pairs_l299_29914


namespace cubic_meter_to_cubic_centimeters_total_volume_l299_29937

-- Define the conversion factor
def meters_to_centimeters : ℝ := 100

-- Theorem 1: One cubic meter is equal to 1,000,000 cubic centimeters
theorem cubic_meter_to_cubic_centimeters : 
  (meters_to_centimeters ^ 3 : ℝ) = 1000000 := by sorry

-- Theorem 2: The sum of one cubic meter and 500 cubic centimeters is equal to 1,000,500 cubic centimeters
theorem total_volume (cubic_cm_to_add : ℝ) : 
  cubic_cm_to_add = 500 → 
  (meters_to_centimeters ^ 3 + cubic_cm_to_add : ℝ) = 1000500 := by sorry

end cubic_meter_to_cubic_centimeters_total_volume_l299_29937


namespace unique_satisfying_function_l299_29988

/-- A function satisfying the given condition -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) * f (x - y) = (f x - f y)^2 - (4 * x * y) * f y

/-- Theorem stating that the only function satisfying the condition is the zero function -/
theorem unique_satisfying_function :
  ∃! f : ℝ → ℝ, SatisfyingFunction f ∧ ∀ x : ℝ, f x = 0 := by
  sorry

end unique_satisfying_function_l299_29988


namespace bridget_apples_l299_29960

theorem bridget_apples (x : ℕ) : 
  (2 * x) / 3 - 11 = 10 → x = 32 := by sorry

end bridget_apples_l299_29960


namespace polygon_division_theorem_l299_29985

/-- A polygon that can be divided into a specific number of rectangles -/
structure DivisiblePolygon where
  vertices : ℕ
  can_divide : ℕ → Prop
  h_100 : can_divide 100
  h_not_99 : ¬ can_divide 99

/-- The main theorem stating that a polygon divisible into 100 rectangles but not 99
    has more than 200 vertices and cannot be divided into 100 triangles -/
theorem polygon_division_theorem (P : DivisiblePolygon) :
  P.vertices > 200 ∧ ¬ ∃ (triangles : ℕ), triangles = 100 ∧ P.can_divide triangles := by
  sorry


end polygon_division_theorem_l299_29985


namespace multiplication_subtraction_difference_l299_29928

theorem multiplication_subtraction_difference : ∀ x : ℤ, 
  x = 11 → (3 * x) - (26 - x) = 18 := by
  sorry

end multiplication_subtraction_difference_l299_29928


namespace modular_inverse_37_mod_39_l299_29997

theorem modular_inverse_37_mod_39 : 
  ∃ x : ℕ, x < 39 ∧ (37 * x) % 39 = 1 ∧ x = 19 := by
  sorry

end modular_inverse_37_mod_39_l299_29997


namespace four_times_three_equals_thirtyone_l299_29901

-- Define the multiplication operation based on the given condition
def special_mult (a b : ℤ) : ℤ := a^2 + 2*a*b - b^2

-- State the theorem
theorem four_times_three_equals_thirtyone : special_mult 4 3 = 31 := by
  sorry

end four_times_three_equals_thirtyone_l299_29901


namespace song_book_cost_l299_29943

theorem song_book_cost (flute_cost music_stand_cost total_spent : ℚ)
  (h1 : flute_cost = 142.46)
  (h2 : music_stand_cost = 8.89)
  (h3 : total_spent = 158.35) :
  total_spent - (flute_cost + music_stand_cost) = 7.00 := by
  sorry

end song_book_cost_l299_29943


namespace smallest_cube_with_more_than_half_remaining_l299_29929

theorem smallest_cube_with_more_than_half_remaining : 
  ∀ n : ℕ, n > 0 → ((n : ℚ) - 4)^3 > (n : ℚ)^3 / 2 ↔ n ≥ 20 :=
sorry

end smallest_cube_with_more_than_half_remaining_l299_29929


namespace age_difference_and_future_relation_l299_29912

/-- A digit is a natural number from 0 to 9 -/
def Digit : Type := {n : ℕ // n ≤ 9}

/-- Jack's age given two digits -/
def jack_age (a b : Digit) : ℕ := 10 * a.val + b.val

/-- Bill's age given two digits -/
def bill_age (a b : Digit) : ℕ := b.val^2 + a.val

theorem age_difference_and_future_relation :
  ∃ (a b : Digit), 
    (jack_age a b - bill_age a b = 18) ∧ 
    (jack_age a b + 6 = 3 * (bill_age a b + 6)) := by
  sorry

end age_difference_and_future_relation_l299_29912


namespace last_term_before_one_l299_29978

def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1 : ℝ) * d

theorem last_term_before_one (a : ℝ) (d : ℝ) (n : ℕ) :
  a = 100 ∧ d = -4 →
  arithmetic_sequence a d 25 > 1 ∧
  arithmetic_sequence a d 26 ≤ 1 := by
sorry

end last_term_before_one_l299_29978


namespace subset_equivalence_l299_29927

theorem subset_equivalence (φ A : Set α) (p q : Prop) :
  (φ ⊆ A ↔ (φ = A ∨ φ ⊂ A)) →
  (φ ⊆ A ↔ p ∨ q) :=
by sorry

end subset_equivalence_l299_29927


namespace group_size_l299_29993

theorem group_size (total : ℕ) (over_30 : ℕ) (under_20 : ℕ) 
  (h1 : over_30 = 90)
  (h2 : total = over_30 + under_20)
  (h3 : (under_20 : ℚ) / total = 1 / 10) : 
  total = 100 := by
sorry

end group_size_l299_29993


namespace increase_in_circumference_l299_29969

/-- The increase in circumference when the diameter of a circle increases by 2π units -/
theorem increase_in_circumference (d : ℝ) : 
  let original_circumference := π * d
  let new_circumference := π * (d + 2 * π)
  let increase_in_circumference := new_circumference - original_circumference
  increase_in_circumference = 2 * π^2 := by
  sorry

end increase_in_circumference_l299_29969


namespace product_difference_squares_l299_29900

theorem product_difference_squares : (3 + Real.sqrt 7) * (3 - Real.sqrt 7) = 2 := by
  sorry

end product_difference_squares_l299_29900


namespace train_bridge_crossing_time_l299_29956

/-- Proves that a train with given length and speed takes the specified time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 120)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 255) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end train_bridge_crossing_time_l299_29956


namespace inequality_proof_l299_29931

theorem inequality_proof (a b : ℝ) (h : a ≠ b) : a^4 + 6*a^2*b^2 + b^4 > 4*a*b*(a^2 + b^2) := by
  sorry

end inequality_proof_l299_29931


namespace johns_candy_store_spending_l299_29952

def weekly_allowance : ℚ := 240 / 100

def arcade_spending : ℚ := (3 / 5) * weekly_allowance

def remaining_after_arcade : ℚ := weekly_allowance - arcade_spending

def toy_store_spending : ℚ := (1 / 3) * remaining_after_arcade

def candy_store_spending : ℚ := remaining_after_arcade - toy_store_spending

theorem johns_candy_store_spending :
  candy_store_spending = 64 / 100 := by sorry

end johns_candy_store_spending_l299_29952


namespace five_digit_multiple_of_nine_l299_29905

theorem five_digit_multiple_of_nine :
  ∃ (n : ℕ), n = 56781 ∧ n % 9 = 0 :=
by
  sorry

end five_digit_multiple_of_nine_l299_29905


namespace perpendicular_bisector_complex_l299_29989

/-- The set of points equidistant from two distinct complex numbers forms a perpendicular bisector -/
theorem perpendicular_bisector_complex (z₁ z₂ : ℂ) (hz : z₁ ≠ z₂) :
  {z : ℂ | Complex.abs (z - z₁) = Complex.abs (z - z₂)} =
  {z : ℂ | (z - (z₁ + z₂) / 2) • (z₁ - z₂) = 0} :=
by sorry

end perpendicular_bisector_complex_l299_29989


namespace x_values_l299_29949

def S : Set ℤ := {1, -1}

theorem x_values (a b c d e f : ℤ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S) (he : e ∈ S) (hf : f ∈ S) :
  {x | ∃ (a b c d e f : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ x = a - b + c - d + e - f} = {-6, -4, -2, 0, 2, 4, 6} := by
  sorry

end x_values_l299_29949


namespace stairs_climbing_time_l299_29983

def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem stairs_climbing_time : arithmeticSum 30 8 6 = 300 := by
  sorry

end stairs_climbing_time_l299_29983


namespace opposite_of_pi_l299_29933

theorem opposite_of_pi : -π = -π := by sorry

end opposite_of_pi_l299_29933


namespace tree_growth_rate_l299_29916

theorem tree_growth_rate (h : ℝ) (initial_height : ℝ) (growth_period : ℕ) :
  initial_height = 4 →
  growth_period = 6 →
  initial_height + 6 * h = (initial_height + 4 * h) * (1 + 1/7) →
  h = 2/5 := by
  sorry

end tree_growth_rate_l299_29916


namespace alice_spending_percentage_l299_29954

theorem alice_spending_percentage (alice_initial : ℝ) (bob_initial : ℝ) (alice_final : ℝ)
  (h1 : bob_initial = 0.9 * alice_initial)
  (h2 : alice_final = 0.9 * bob_initial) :
  (alice_initial - alice_final) / alice_initial = 0.19 :=
by sorry

end alice_spending_percentage_l299_29954


namespace sams_walking_speed_l299_29922

/-- Proves that Sam's walking speed is equal to Fred's given the problem conditions -/
theorem sams_walking_speed (total_distance : ℝ) (fred_speed : ℝ) (sam_distance : ℝ) :
  total_distance = 50 →
  fred_speed = 5 →
  sam_distance = 25 →
  let fred_distance := total_distance - sam_distance
  let time := fred_distance / fred_speed
  let sam_speed := sam_distance / time
  sam_speed = fred_speed :=
by sorry

end sams_walking_speed_l299_29922


namespace cos_pi_plus_alpha_implies_sin_5pi_over_2_minus_alpha_l299_29934

theorem cos_pi_plus_alpha_implies_sin_5pi_over_2_minus_alpha 
  (α : Real) 
  (h : Real.cos (Real.pi + α) = -1/3) : 
  Real.sin ((5/2) * Real.pi - α) = 1/3 := by
  sorry

end cos_pi_plus_alpha_implies_sin_5pi_over_2_minus_alpha_l299_29934


namespace greatest_common_divisor_540_126_under_60_l299_29924

theorem greatest_common_divisor_540_126_under_60 : 
  Nat.gcd 540 126 < 60 ∧ 
  ∀ d : Nat, d ∣ 540 ∧ d ∣ 126 ∧ d < 60 → d ≤ 18 := by
  sorry

end greatest_common_divisor_540_126_under_60_l299_29924
