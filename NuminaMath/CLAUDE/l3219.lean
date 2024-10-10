import Mathlib

namespace complement_union_theorem_l3219_321992

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_theorem : 
  (U \ (M ∪ N)) = {5} := by sorry

end complement_union_theorem_l3219_321992


namespace triangle_properties_l3219_321980

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin C = Real.sqrt 3 * c * Real.cos A →
  (A = π / 3) ∧
  (a = 2 → (1 / 2) * b * c * Real.sin A = Real.sqrt 3 → b = 2 ∧ c = 2) := by
  sorry

end triangle_properties_l3219_321980


namespace vending_machine_probability_l3219_321962

/-- Represents a vending machine with toys and their prices -/
structure VendingMachine :=
  (num_toys : ℕ)
  (min_price : ℚ)
  (price_increment : ℚ)

/-- Represents Peter's initial money -/
structure InitialMoney :=
  (quarters : ℕ)
  (bill : ℚ)

/-- The main theorem statement -/
theorem vending_machine_probability
  (vm : VendingMachine)
  (money : InitialMoney)
  (favorite_toy_price : ℚ) :
  vm.num_toys = 10 →
  vm.min_price = 25/100 →
  vm.price_increment = 25/100 →
  money.quarters = 10 →
  money.bill = 20 →
  favorite_toy_price = 2 →
  (probability_need_break_bill : ℚ) →
  probability_need_break_bill = 9/10 :=
by sorry

end vending_machine_probability_l3219_321962


namespace inequality_system_solution_set_l3219_321900

theorem inequality_system_solution_set :
  {x : ℝ | -2*x ≤ 6 ∧ x + 1 < 0} = {x : ℝ | -3 ≤ x ∧ x < -1} := by
  sorry

end inequality_system_solution_set_l3219_321900


namespace greatest_equal_distribution_l3219_321941

theorem greatest_equal_distribution (a b c : ℕ) (ha : a = 1050) (hb : b = 1260) (hc : c = 210) :
  Nat.gcd a (Nat.gcd b c) = 210 := by
  sorry

end greatest_equal_distribution_l3219_321941


namespace orange_bags_weight_l3219_321938

/-- If 12 bags of oranges weigh 24 pounds, then 8 bags of oranges weigh 16 pounds. -/
theorem orange_bags_weight (total_weight : ℝ) (total_bags : ℕ) (target_bags : ℕ) :
  total_weight = 24 ∧ total_bags = 12 ∧ target_bags = 8 →
  (target_bags : ℝ) * (total_weight / total_bags) = 16 :=
by sorry

end orange_bags_weight_l3219_321938


namespace intersection_of_M_and_N_l3219_321930

def M : Set ℝ := {x | (x + 1) * (x - 3) ≤ 0}
def N : Set ℝ := {x | 1 < x ∧ x < 4}

theorem intersection_of_M_and_N : M ∩ N = {x | 1 < x ∧ x ≤ 3} := by sorry

end intersection_of_M_and_N_l3219_321930


namespace parallelogram_diagonals_contain_conjugate_diameters_l3219_321979

-- Define an ellipse
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

-- Define a parallelogram
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ

-- Define conjugate diameters of an ellipse
def conjugate_diameters (e : Ellipse) : Set (ℝ × ℝ) := sorry

-- Define the diagonals of a parallelogram
def diagonals (p : Parallelogram) : Set (ℝ × ℝ) := sorry

-- Define what it means for a parallelogram to be inscribed around an ellipse
def is_inscribed (p : Parallelogram) (e : Ellipse) : Prop := sorry

-- Theorem statement
theorem parallelogram_diagonals_contain_conjugate_diameters 
  (e : Ellipse) (p : Parallelogram) (h : is_inscribed p e) :
  diagonals p ⊆ conjugate_diameters e := by sorry

end parallelogram_diagonals_contain_conjugate_diameters_l3219_321979


namespace chord_equation_l3219_321997

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

/-- The midpoint of the chord -/
def P : ℝ × ℝ := (8, 1)

/-- A point lies on the line containing the chord -/
def lies_on_chord_line (x y : ℝ) : Prop := 2*x - y - 15 = 0

theorem chord_equation :
  ∀ A B : ℝ × ℝ,
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  hyperbola x₁ y₁ →
  hyperbola x₂ y₂ →
  (x₁ + x₂) / 2 = P.1 →
  (y₁ + y₂) / 2 = P.2 →
  lies_on_chord_line x₁ y₁ ∧ lies_on_chord_line x₂ y₂ :=
by sorry

end chord_equation_l3219_321997


namespace largest_c_for_negative_four_in_range_l3219_321909

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + c

-- State the theorem
theorem largest_c_for_negative_four_in_range :
  (∃ (c : ℝ), ∀ (c' : ℝ), 
    (∃ (x : ℝ), f c' x = -4) → c' ≤ c) ∧
  (∃ (x : ℝ), f (9/4) x = -4) :=
sorry

end largest_c_for_negative_four_in_range_l3219_321909


namespace symmetric_lines_l3219_321963

/-- Given two lines L and K symmetric to each other with respect to y=x,
    where L has equation y = ax + b (a ≠ 0, b ≠ 0),
    prove that K has equation y = (1/a)x - (b/a) -/
theorem symmetric_lines (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let L : ℝ → ℝ := fun x => a * x + b
  let K : ℝ → ℝ := fun x => (1 / a) * x - (b / a)
  (∀ x y, y = L x ↔ x = L y) →
  (∀ x, K x = (1 / a) * x - (b / a)) := by
  sorry

end symmetric_lines_l3219_321963


namespace base_eight_solution_l3219_321969

theorem base_eight_solution : ∃! (b : ℕ), b > 1 ∧ (3 * b + 2)^2 = b^3 + b + 4 :=
by sorry

end base_eight_solution_l3219_321969


namespace sara_letters_count_l3219_321975

/-- The number of letters Sara sent in January -/
def january_letters : ℕ := 6

/-- The number of letters Sara sent in February -/
def february_letters : ℕ := 9

/-- The number of letters Sara sent in March -/
def march_letters : ℕ := 3 * january_letters

/-- The total number of letters Sara sent -/
def total_letters : ℕ := january_letters + february_letters + march_letters

theorem sara_letters_count :
  total_letters = 33 := by sorry

end sara_letters_count_l3219_321975


namespace no_linear_term_implies_a_value_l3219_321940

theorem no_linear_term_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, ∃ b c d : ℝ, (x^2 + a*x - 2)*(x - 1) = x^3 + b*x^2 + d) → a = -2 := by
  sorry

end no_linear_term_implies_a_value_l3219_321940


namespace matchbox_cars_percentage_l3219_321920

theorem matchbox_cars_percentage (total : ℕ) (truck_percent : ℚ) (convertibles : ℕ) : 
  total = 125 →
  truck_percent = 8 / 100 →
  convertibles = 35 →
  (((total : ℚ) - (truck_percent * total) - (convertibles : ℚ)) / total) * 100 = 64 := by
sorry

end matchbox_cars_percentage_l3219_321920


namespace empty_set_equality_l3219_321919

theorem empty_set_equality : 
  {x : ℝ | x^2 + 2 = 0} = {y : ℝ | y^2 + 1 < 0} := by sorry

end empty_set_equality_l3219_321919


namespace factorization_equality_l3219_321959

theorem factorization_equality (x y : ℝ) : x^2*y - 6*x*y + 9*y = y*(x-3)^2 := by
  sorry

end factorization_equality_l3219_321959


namespace dans_initial_amount_l3219_321993

/-- Dan's initial amount of money -/
def initial_amount : ℝ := 4

/-- The cost of the candy bar -/
def candy_cost : ℝ := 1

/-- The amount Dan had left after buying the candy bar -/
def remaining_amount : ℝ := 3

/-- Theorem stating that Dan's initial amount equals the sum of the remaining amount and the candy cost -/
theorem dans_initial_amount : initial_amount = remaining_amount + candy_cost := by
  sorry

end dans_initial_amount_l3219_321993


namespace sqrt_eight_minus_sqrt_two_l3219_321964

theorem sqrt_eight_minus_sqrt_two : Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end sqrt_eight_minus_sqrt_two_l3219_321964


namespace min_crossing_time_l3219_321954

/-- Represents a person with their crossing time -/
structure Person where
  crossingTime : ℕ

/-- Represents the state of the bridge crossing problem -/
structure BridgeState where
  peopleOnIsland : List Person
  peopleOnMainland : List Person
  lampOnIsland : Bool
  totalTime : ℕ

/-- Defines the initial state of the problem -/
def initialState : BridgeState where
  peopleOnIsland := [
    { crossingTime := 2 },
    { crossingTime := 4 },
    { crossingTime := 8 },
    { crossingTime := 16 }
  ]
  peopleOnMainland := []
  lampOnIsland := true
  totalTime := 0

/-- Represents a valid move across the bridge -/
inductive Move
  | cross (p1 : Person) (p2 : Option Person)
  | returnLamp (p : Person)

/-- Applies a move to the current state -/
def applyMove (state : BridgeState) (move : Move) : BridgeState :=
  sorry

/-- Checks if all people have crossed to the mainland -/
def isComplete (state : BridgeState) : Bool :=
  sorry

/-- Theorem: The minimum time required to cross the bridge is 30 minutes -/
theorem min_crossing_time (initialState : BridgeState) :
  ∃ (moves : List Move), 
    (moves.foldl applyMove initialState).totalTime = 30 ∧ 
    isComplete (moves.foldl applyMove initialState) ∧
    ∀ (otherMoves : List Move), 
      isComplete (otherMoves.foldl applyMove initialState) → 
      (otherMoves.foldl applyMove initialState).totalTime ≥ 30 :=
  sorry

end min_crossing_time_l3219_321954


namespace quadratic_equation_theorem_l3219_321911

/-- The quadratic equation x^2 - 2(m-1)x + m^2 = 0 has real roots -/
def has_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁^2 - 2*(m-1)*x₁ + m^2 = 0 ∧ x₂^2 - 2*(m-1)*x₂ + m^2 = 0

/-- The roots of the quadratic equation satisfy x₁^2 + x₂^2 = 8 - 3*x₁*x₂ -/
def roots_satisfy_condition (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁^2 - 2*(m-1)*x₁ + m^2 = 0 ∧ x₂^2 - 2*(m-1)*x₂ + m^2 = 0 ∧ x₁^2 + x₂^2 = 8 - 3*x₁*x₂

theorem quadratic_equation_theorem :
  (∀ m : ℝ, has_real_roots m → m ≤ 1/2) ∧
  (∀ m : ℝ, roots_satisfy_condition m → m = -2/5) :=
sorry

end quadratic_equation_theorem_l3219_321911


namespace x_squared_minus_four_y_squared_l3219_321939

theorem x_squared_minus_four_y_squared (x y : ℝ) 
  (eq1 : x + 2*y = 4) 
  (eq2 : x - 2*y = -1) : 
  x^2 - 4*y^2 = -4 := by
sorry

end x_squared_minus_four_y_squared_l3219_321939


namespace final_piggy_bank_amount_l3219_321927

def piggy_bank_savings (initial_amount : ℝ) (weekly_allowance : ℝ) (savings_fraction : ℝ) (weeks : ℕ) : ℝ :=
  initial_amount + (weekly_allowance * savings_fraction * weeks)

theorem final_piggy_bank_amount :
  piggy_bank_savings 43 10 0.5 8 = 83 := by
  sorry

end final_piggy_bank_amount_l3219_321927


namespace nephews_ages_sum_l3219_321952

theorem nephews_ages_sum :
  ∀ (a b c d : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →  -- single-digit
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →  -- distinct
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- positive
    ((a * b = 36 ∧ c * d = 40) ∨ (a * c = 36 ∧ b * d = 40) ∨ 
     (a * d = 36 ∧ b * c = 40) ∨ (b * c = 36 ∧ a * d = 40) ∨ 
     (b * d = 36 ∧ a * c = 40) ∨ (c * d = 36 ∧ a * b = 40)) →
    a + b + c + d = 26 :=
by
  sorry

end nephews_ages_sum_l3219_321952


namespace chess_tournament_games_l3219_321934

theorem chess_tournament_games (n : ℕ) (h : n = 18) : 
  (n * (n - 1)) / 2 = 153 := by
  sorry

end chess_tournament_games_l3219_321934


namespace floor_ceiling_sum_l3219_321916

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ - 4 = 27 := by
  sorry

end floor_ceiling_sum_l3219_321916


namespace m_minus_n_equals_l3219_321937

def M : Set Nat := {1, 3, 5, 7, 9}
def N : Set Nat := {2, 3, 5}

def setDifference (A B : Set Nat) : Set Nat :=
  {x | x ∈ A ∧ x ∉ B}

theorem m_minus_n_equals : setDifference M N = {1, 7, 9} := by
  sorry

end m_minus_n_equals_l3219_321937


namespace harry_change_problem_l3219_321960

theorem harry_change_problem (change : ℕ) : 
  change < 100 ∧ 
  change % 50 = 2 ∧ 
  change % 5 = 4 → 
  change = 52 := by sorry

end harry_change_problem_l3219_321960


namespace circle_line_distance_difference_l3219_321981

/-- Given a circle with equation x² + (y-1)² = 1 and a line x - y - 2 = 0,
    the difference between the maximum and minimum distances from points
    on the circle to the line is (√2)/2 + 1. -/
theorem circle_line_distance_difference :
  let circle := {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1}
  let line := {p : ℝ × ℝ | p.1 - p.2 - 2 = 0}
  let max_distance := Real.sqrt 8
  let min_distance := (3 * Real.sqrt 2) / 2 - 1
  max_distance - min_distance = Real.sqrt 2 / 2 + 1 := by
  sorry

end circle_line_distance_difference_l3219_321981


namespace sqrt_10_between_3_and_4_l3219_321903

theorem sqrt_10_between_3_and_4 : 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 := by
  sorry

end sqrt_10_between_3_and_4_l3219_321903


namespace problem_one_problem_two_l3219_321948

-- Problem 1
theorem problem_one : 2 * Real.cos (π / 4) + |1 - Real.sqrt 2| + (-2) ^ 0 = 2 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_two (a : ℝ) : 3 * a + 2 * a * (a - 1) = 2 * a ^ 2 + a := by
  sorry

end problem_one_problem_two_l3219_321948


namespace gcd_lcm_sum_60_45045_l3219_321967

theorem gcd_lcm_sum_60_45045 : Nat.gcd 60 45045 + Nat.lcm 60 45045 = 180195 := by
  sorry

end gcd_lcm_sum_60_45045_l3219_321967


namespace divisibility_of_repeating_digits_l3219_321914

theorem divisibility_of_repeating_digits : ∃ (k m : ℕ), k > 0 ∧ (1989 * (10^(4*k) - 1) / 9) * 10^m % 1988 = 0 ∧
                                          ∃ (n : ℕ), n > 0 ∧ (1988 * (10^(4*n) - 1) / 9) % 1989 = 0 := by
  sorry

end divisibility_of_repeating_digits_l3219_321914


namespace largest_of_three_consecutive_integers_sum_153_l3219_321918

theorem largest_of_three_consecutive_integers_sum_153 :
  ∀ (x y z : ℤ), 
    (y = x + 1) → 
    (z = y + 1) → 
    (x + y + z = 153) → 
    (max x (max y z) = 52) :=
by sorry

end largest_of_three_consecutive_integers_sum_153_l3219_321918


namespace factorization_theorem_l3219_321965

theorem factorization_theorem (a : ℝ) : 4 * a^2 - 4 = 4 * (a + 1) * (a - 1) := by
  sorry

end factorization_theorem_l3219_321965


namespace intersection_of_A_and_B_l3219_321974

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {1, 5} := by
  sorry

end intersection_of_A_and_B_l3219_321974


namespace inscribed_circle_max_radius_l3219_321970

/-- Given a triangle ABC with side lengths a, b, c, and area A,
    and an inscribed circle with radius r, 
    the radius r is at most (2 * A) / (a + b + c) --/
theorem inscribed_circle_max_radius 
  (a b c : ℝ) 
  (A : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hA : A > 0) 
  (h_triangle : A = a * b * c / (4 * (a * b + b * c + c * a - a * a - b * b - c * c).sqrt)) 
  (r : ℝ) 
  (hr : r > 0) 
  (h_inscribed : r * (a + b + c) ≤ 2 * A) :
  r ≤ 2 * A / (a + b + c) ∧ 
  (r = 2 * A / (a + b + c) ↔ r * (a + b + c) = 2 * A) :=
sorry

end inscribed_circle_max_radius_l3219_321970


namespace cubic_root_equation_solutions_l3219_321977

theorem cubic_root_equation_solutions :
  let f : ℝ → ℝ := λ x => (18 * x - 2)^(1/3) + (16 * x + 2)^(1/3) + (-72 * x)^(1/3) - 6 * x^(1/3)
  {x : ℝ | f x = 0} = {0, 1/9, -1/8} := by
  sorry

end cubic_root_equation_solutions_l3219_321977


namespace perfect_square_consecutive_integers_l3219_321947

theorem perfect_square_consecutive_integers (n : ℤ) : 
  (∃ k : ℤ, n * (n + 1) = k^2) ↔ (n = 0 ∨ n = -1) := by
sorry

end perfect_square_consecutive_integers_l3219_321947


namespace derivative_cos_2x_plus_1_l3219_321999

theorem derivative_cos_2x_plus_1 (x : ℝ) :
  deriv (fun x => Real.cos (2 * x + 1)) x = -2 * Real.sin (2 * x + 1) := by
  sorry

end derivative_cos_2x_plus_1_l3219_321999


namespace polynomial_factorization_l3219_321921

theorem polynomial_factorization (x y : ℝ) : 
  x^4 - 2*x^2*y - 3*y^2 + 8*y - 4 = (x^2 + y - 2)*(x^2 - 3*y + 2) := by
  sorry

end polynomial_factorization_l3219_321921


namespace expression_mod_18_l3219_321923

theorem expression_mod_18 : (234 * 18 - 23 * 9 + 5) % 18 = 14 := by
  sorry

end expression_mod_18_l3219_321923


namespace simplify_expression_l3219_321984

theorem simplify_expression (w : ℝ) : 2*w + 3 - 4*w - 5 + 6*w + 7 - 8*w - 9 = -4*w - 4 := by
  sorry

end simplify_expression_l3219_321984


namespace complex_equation_solution_l3219_321936

/-- Given that (1+i)z = |-4i|, prove that z = 2 - 2i --/
theorem complex_equation_solution :
  ∀ z : ℂ, (Complex.I + 1) * z = Complex.abs (-4 * Complex.I) → z = 2 - 2 * Complex.I := by
  sorry

end complex_equation_solution_l3219_321936


namespace closest_integer_to_cube_root_l3219_321910

theorem closest_integer_to_cube_root : ∃ n : ℤ, 
  n = 8 ∧ ∀ m : ℤ, |n - (5^3 + 7^3)^(1/3)| ≤ |m - (5^3 + 7^3)^(1/3)| := by
  sorry

end closest_integer_to_cube_root_l3219_321910


namespace divisors_of_30_l3219_321983

/-- The number of integer divisors (positive and negative) of 30 -/
def number_of_divisors_of_30 : ℕ :=
  (Finset.filter (· ∣ 30) (Finset.range 31)).card * 2

/-- Theorem stating that the number of integer divisors of 30 is 16 -/
theorem divisors_of_30 : number_of_divisors_of_30 = 16 := by
  sorry

end divisors_of_30_l3219_321983


namespace first_cyclist_overtakes_second_opposite_P_l3219_321978

/-- Represents the circular runway --/
structure CircularRunway where
  radius : ℝ

/-- Represents a moving entity on the circular runway --/
structure MovingEntity where
  velocity : ℝ

/-- Represents the scenario of cyclists and pedestrian on the circular runway --/
structure RunwayScenario where
  runway : CircularRunway
  cyclist1 : MovingEntity
  cyclist2 : MovingEntity
  pedestrian : MovingEntity

/-- The main theorem stating the point where the first cyclist overtakes the second --/
theorem first_cyclist_overtakes_second_opposite_P (scenario : RunwayScenario) 
  (h1 : scenario.cyclist1.velocity > scenario.cyclist2.velocity)
  (h2 : scenario.pedestrian.velocity = (scenario.cyclist1.velocity + scenario.cyclist2.velocity) / 12)
  (h3 : ∃ t1 t2, t2 - t1 = 91 ∧ 
        t1 = (2 * π * scenario.runway.radius) / (scenario.cyclist1.velocity + scenario.pedestrian.velocity) ∧
        t2 = (2 * π * scenario.runway.radius) / (scenario.cyclist2.velocity + scenario.pedestrian.velocity))
  (h4 : ∃ t3 t4, t4 - t3 = 187 ∧
        t3 = (2 * π * scenario.runway.radius) / (scenario.cyclist1.velocity - scenario.pedestrian.velocity) ∧
        t4 = (2 * π * scenario.runway.radius) / (scenario.cyclist2.velocity - scenario.pedestrian.velocity)) :
  ∃ t : ℝ, t * scenario.cyclist1.velocity = π * scenario.runway.radius ∧
          t * scenario.cyclist2.velocity = π * scenario.runway.radius :=
by sorry

end first_cyclist_overtakes_second_opposite_P_l3219_321978


namespace cube_of_thousands_l3219_321986

theorem cube_of_thousands (n : ℕ) : n = (n / 1000)^3 ↔ n = 32768 := by
  sorry

end cube_of_thousands_l3219_321986


namespace sqrt_x_plus_inverse_l3219_321949

theorem sqrt_x_plus_inverse (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 49) : 
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 51 := by
sorry

end sqrt_x_plus_inverse_l3219_321949


namespace boat_fee_ratio_l3219_321987

/-- Proves that the ratio of docking fees to license and registration fees is 3:1 given the conditions of Mitch's boat purchase. -/
theorem boat_fee_ratio :
  let total_savings : ℚ := 20000
  let boat_cost_per_foot : ℚ := 1500
  let license_fee : ℚ := 500
  let max_boat_length : ℚ := 12
  let available_for_boat : ℚ := total_savings - license_fee
  let boat_cost : ℚ := boat_cost_per_foot * max_boat_length
  let docking_fee : ℚ := available_for_boat - boat_cost
  docking_fee / license_fee = 3 := by
  sorry

end boat_fee_ratio_l3219_321987


namespace quadratic_inequality_solution_l3219_321953

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 - x + 2

-- Define the solution set
def solution_set : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x ≥ 0} = solution_set := by sorry

end quadratic_inequality_solution_l3219_321953


namespace casper_enter_exit_ways_l3219_321922

/-- The number of windows in the castle -/
def num_windows : ℕ := 8

/-- The number of ways Casper can enter and exit the castle -/
def num_ways : ℕ := num_windows * (num_windows - 1)

/-- Theorem stating that the number of ways Casper can enter and exit is 56 -/
theorem casper_enter_exit_ways : num_ways = 56 := by
  sorry

end casper_enter_exit_ways_l3219_321922


namespace horner_method_for_f_at_3_l3219_321968

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + x^3 + x^2 + x + 1

-- Theorem statement
theorem horner_method_for_f_at_3 : f 3 = 283 := by
  sorry

end horner_method_for_f_at_3_l3219_321968


namespace three_fourths_of_four_fifths_of_two_thirds_l3219_321935

theorem three_fourths_of_four_fifths_of_two_thirds : (3 : ℚ) / 4 * (4 : ℚ) / 5 * (2 : ℚ) / 3 = (2 : ℚ) / 5 := by
  sorry

end three_fourths_of_four_fifths_of_two_thirds_l3219_321935


namespace rod_pieces_count_l3219_321990

/-- The length of the rod in meters -/
def rod_length_m : ℝ := 34

/-- The length of each piece in centimeters -/
def piece_length_cm : ℝ := 85

/-- Conversion factor from meters to centimeters -/
def m_to_cm : ℝ := 100

theorem rod_pieces_count : 
  ⌊(rod_length_m * m_to_cm) / piece_length_cm⌋ = 40 := by sorry

end rod_pieces_count_l3219_321990


namespace problem_1_problem_2_l3219_321943

-- Problem 1
theorem problem_1 : (-2)^2 + (Real.sqrt 2 - 1)^0 - 1 = 4 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (A B : ℝ) (h1 : A = a - 1) (h2 : B = -a + 3) (h3 : A > B) :
  a > 2 := by sorry

end problem_1_problem_2_l3219_321943


namespace product_one_sum_square_and_products_geq_ten_l3219_321913

theorem product_one_sum_square_and_products_geq_ten 
  (a b c d : ℝ) (h : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end product_one_sum_square_and_products_geq_ten_l3219_321913


namespace only_B_and_C_have_inverses_l3219_321971

-- Define the set of functions
inductive Function : Type
| A | B | C | D | E

-- Define the property of having an inverse
def has_inverse (f : Function) : Prop :=
  match f with
  | Function.A => False
  | Function.B => True
  | Function.C => True
  | Function.D => False
  | Function.E => False

-- Theorem statement
theorem only_B_and_C_have_inverses :
  ∀ f : Function, has_inverse f ↔ (f = Function.B ∨ f = Function.C) :=
by sorry

end only_B_and_C_have_inverses_l3219_321971


namespace theater_ticket_sales_l3219_321925

theorem theater_ticket_sales (adult_price kid_price profit kid_tickets : ℕ) 
  (h1 : adult_price = 6)
  (h2 : kid_price = 2)
  (h3 : profit = 750)
  (h4 : kid_tickets = 75) :
  ∃ (adult_tickets : ℕ), adult_tickets * adult_price + kid_tickets * kid_price = profit ∧
                          adult_tickets + kid_tickets = 175 :=
by sorry

end theater_ticket_sales_l3219_321925


namespace zero_at_neg_one_one_zero_in_interval_l3219_321961

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 2 - a

-- Theorem 1: When a = -1, the function has a zero at x = 1
theorem zero_at_neg_one :
  f (-1) 1 = 0 := by sorry

-- Theorem 2: The function has exactly one zero in (0, 1] iff -1 ≤ a ≤ 0 or a ≤ -2
theorem one_zero_in_interval (a : ℝ) :
  (∃! x : ℝ, 0 < x ∧ x ≤ 1 ∧ f a x = 0) ↔ (-1 ≤ a ∧ a ≤ 0) ∨ a ≤ -2 := by sorry

end zero_at_neg_one_one_zero_in_interval_l3219_321961


namespace pool_filling_time_l3219_321929

/-- Represents the volume of the pool -/
def pool_volume : ℝ := 1

/-- Represents the rate at which pipe X fills the pool -/
def rate_X : ℝ := sorry

/-- Represents the rate at which pipe Y fills the pool -/
def rate_Y : ℝ := sorry

/-- Represents the rate at which pipe Z fills the pool -/
def rate_Z : ℝ := sorry

/-- Time taken by pipes X and Y together to fill the pool -/
def time_XY : ℝ := 3

/-- Time taken by pipes X and Z together to fill the pool -/
def time_XZ : ℝ := 6

/-- Time taken by pipes Y and Z together to fill the pool -/
def time_YZ : ℝ := 4.5

theorem pool_filling_time :
  let time_XYZ := pool_volume / (rate_X + rate_Y + rate_Z)
  pool_volume / (rate_X + rate_Y) = time_XY ∧
  pool_volume / (rate_X + rate_Z) = time_XZ ∧
  pool_volume / (rate_Y + rate_Z) = time_YZ →
  (time_XYZ ≥ 2.76 ∧ time_XYZ ≤ 2.78) := by sorry

end pool_filling_time_l3219_321929


namespace min_value_is_four_l3219_321976

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  d : ℚ
  hd : d ≠ 0
  ha1 : a 1 = 1
  hGeometric : (a 3) ^ 2 = (a 1) * (a 13)
  hArithmetic : ∀ n : ℕ+, a n = a 1 + (n - 1) * d

/-- Sum of the first n terms of the arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The expression to be minimized -/
def f (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (2 * S seq n + 16) / (seq.a n + 3)

/-- Theorem stating the minimum value of the expression -/
theorem min_value_is_four (seq : ArithmeticSequence) :
  ∃ n₀ : ℕ+, ∀ n : ℕ+, f seq n ≥ f seq n₀ ∧ f seq n₀ = 4 :=
sorry

end min_value_is_four_l3219_321976


namespace like_terms_exponent_sum_l3219_321945

theorem like_terms_exponent_sum (m n : ℤ) : 
  (∃ (k : ℚ), k * x * y^2 = x^(m-2) * y^(n+3)) → m + n = 2 :=
by sorry

end like_terms_exponent_sum_l3219_321945


namespace rectangular_box_dimensions_l3219_321917

theorem rectangular_box_dimensions (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  A * B = 50 →
  A * C = 90 →
  B * C = 100 →
  A + B + C = 24 := by
sorry

end rectangular_box_dimensions_l3219_321917


namespace product_of_sums_equals_difference_of_powers_l3219_321904

theorem product_of_sums_equals_difference_of_powers : 
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) * 
  (3^16 + 5^16) * (3^32 + 5^32) * (3^64 + 5^64) = 3^128 - 5^128 := by
  sorry

end product_of_sums_equals_difference_of_powers_l3219_321904


namespace negation_of_implication_l3219_321982

theorem negation_of_implication (a : ℝ) :
  ¬(a > -3 → a > -6) ↔ (a ≤ -3 → a ≤ -6) := by sorry

end negation_of_implication_l3219_321982


namespace uba_capital_suvs_l3219_321944

/-- Represents the number of SUVs purchased by UBA Capital --/
def num_suvs (total_vehicles : ℕ) : ℕ :=
  let toyota_count := (9 * total_vehicles) / 10
  let honda_count := total_vehicles - toyota_count
  let toyota_suvs := (90 * toyota_count) / 100
  let honda_suvs := (10 * honda_count) / 100
  toyota_suvs + honda_suvs

/-- Theorem stating that the number of SUVs purchased is 8 --/
theorem uba_capital_suvs :
  ∃ (total_vehicles : ℕ), num_suvs total_vehicles = 8 :=
sorry

end uba_capital_suvs_l3219_321944


namespace vector_perpendicular_value_l3219_321985

theorem vector_perpendicular_value (k : ℝ) : 
  let a : (ℝ × ℝ) := (3, 1)
  let b : (ℝ × ℝ) := (1, 3)
  let c : (ℝ × ℝ) := (k, -2)
  (((a.1 - c.1) * b.1 + (a.2 - c.2) * b.2) = 0) → k = 12 := by
  sorry

end vector_perpendicular_value_l3219_321985


namespace min_value_theorem_l3219_321989

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_line : m * 2 + n * 2 = 2) : 
  ∃ (min_val : ℝ), min_val = 3 + 2 * Real.sqrt 2 ∧ 
    ∀ (x y : ℝ), x > 0 → y > 0 → x * 2 + y * 2 = 2 → 1 / x + 2 / y ≥ min_val :=
sorry

end min_value_theorem_l3219_321989


namespace total_goals_after_five_matches_l3219_321946

/-- A football player's goal scoring record -/
structure FootballPlayer where
  goals_before_fifth : ℕ  -- Total goals before the fifth match
  matches_before_fifth : ℕ -- Number of matches before the fifth match (should be 4)

/-- The problem statement -/
theorem total_goals_after_five_matches (player : FootballPlayer) 
  (h1 : player.matches_before_fifth = 4)
  (h2 : (player.goals_before_fifth : ℚ) / 4 + 0.2 = 
        ((player.goals_before_fifth + 4) : ℚ) / 5) : 
  player.goals_before_fifth + 4 = 16 := by
  sorry

#check total_goals_after_five_matches

end total_goals_after_five_matches_l3219_321946


namespace triangle_area_l3219_321942

/-- Given a triangle with perimeter 20 and inradius 3, prove its area is 30 -/
theorem triangle_area (T : Set ℝ) (perimeter inradius : ℝ) : 
  perimeter = 20 →
  inradius = 3 →
  (∃ (area : ℝ), area = inradius * (perimeter / 2) ∧ area = 30) :=
by
  sorry

end triangle_area_l3219_321942


namespace period_of_inverse_a_l3219_321915

/-- Represents a 100-digit number with 1 at the start, 6 at the end, and 98 sevens in between -/
def a : ℕ := 1777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777776

/-- The period of the decimal representation of 1/n -/
def decimal_period (n : ℕ) : ℕ := sorry

theorem period_of_inverse_a : decimal_period a = 99 := by sorry

end period_of_inverse_a_l3219_321915


namespace joan_eggs_count_l3219_321931

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Joan bought -/
def dozens_bought : ℕ := 6

/-- Theorem: Joan bought 72 eggs -/
theorem joan_eggs_count : dozens_bought * eggs_per_dozen = 72 := by
  sorry

end joan_eggs_count_l3219_321931


namespace larger_number_of_two_l3219_321902

theorem larger_number_of_two (x y : ℝ) : 
  x - y = 7 → x + y = 41 → max x y = 24 := by
  sorry

end larger_number_of_two_l3219_321902


namespace tip_to_cost_ratio_l3219_321955

def pizza_order (boxes : ℕ) (cost_per_box : ℚ) (money_given : ℚ) (change_received : ℚ) : ℚ × ℚ :=
  let total_cost := boxes * cost_per_box
  let amount_paid := money_given - change_received
  let tip := amount_paid - total_cost
  (tip, total_cost)

theorem tip_to_cost_ratio : 
  let (tip, total_cost) := pizza_order 5 7 100 60
  (tip : ℚ) / total_cost = 1 / 7 := by sorry

end tip_to_cost_ratio_l3219_321955


namespace isosceles_triangle_rectangle_equal_area_l3219_321972

/-- 
Given an isosceles triangle with base l and height h, and a rectangle with length l and width w,
if their areas are equal, then the height of the triangle is twice the width of the rectangle.
-/
theorem isosceles_triangle_rectangle_equal_area 
  (l w h : ℝ) (l_pos : l > 0) (w_pos : w > 0) (h_pos : h > 0) : 
  (1 / 2 : ℝ) * l * h = l * w → h = 2 * w := by
  sorry

end isosceles_triangle_rectangle_equal_area_l3219_321972


namespace second_polygon_sides_l3219_321996

/-- Given two regular polygons with equal perimeters, where one polygon has 50 sides
    and each of its sides is three times as long as each side of the other polygon,
    the number of sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : s > 0 → n > 0 → 50 * (3 * s) = n * s → n = 150 := by
  sorry

end second_polygon_sides_l3219_321996


namespace painted_cubes_count_l3219_321951

/-- Represents a 3D shape composed of unit cubes -/
structure CubeShape where
  top_layer : Nat
  middle_layer : Nat
  bottom_layer : Nat
  unpainted_cubes : Nat

/-- Calculates the total number of cubes in the shape -/
def total_cubes (shape : CubeShape) : Nat :=
  shape.top_layer + shape.middle_layer + shape.bottom_layer

/-- Calculates the number of cubes with at least one face painted -/
def painted_cubes (shape : CubeShape) : Nat :=
  total_cubes shape - shape.unpainted_cubes

/-- Theorem stating the number of cubes with at least one face painted -/
theorem painted_cubes_count (shape : CubeShape) 
  (h1 : shape.top_layer = 9)
  (h2 : shape.middle_layer = 16)
  (h3 : shape.bottom_layer = 9)
  (h4 : shape.unpainted_cubes = 26) :
  painted_cubes shape = 8 := by
  sorry

end painted_cubes_count_l3219_321951


namespace cubic_function_properties_l3219_321994

-- Define the cubic function
def f (a b c x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

-- Define the first derivative of f
def f' (a b c x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- Define the second derivative of f
def f'' (a b : ℝ) (x : ℝ) : ℝ := 6 * a * x + 2 * b

-- State the theorem
theorem cubic_function_properties (a b c : ℝ) (h_a : a ≠ 0) :
  (∀ x : ℝ, x = 1 ∨ x = -1 → f' a b c x = 0) →
  f a b c 1 = -1 →
  a = -1/2 ∧ b = 0 ∧ c = 3/2 ∧
  f'' a b 1 < 0 ∧ f'' a b (-1) > 0 :=
by sorry

end cubic_function_properties_l3219_321994


namespace only_height_weight_correlated_l3219_321957

/-- Represents the relationship between two variables -/
inductive Relationship
  | Functional
  | Correlated
  | Unrelated

/-- Defines the relationship between a cube's volume and its edge length -/
def cube_volume_edge_relationship : Relationship := Relationship.Functional

/-- Defines the relationship between distance traveled and time for constant speed motion -/
def distance_time_relationship : Relationship := Relationship.Functional

/-- Defines the relationship between a person's height and eyesight -/
def height_eyesight_relationship : Relationship := Relationship.Unrelated

/-- Defines the relationship between a person's height and weight -/
def height_weight_relationship : Relationship := Relationship.Correlated

/-- Theorem stating that only height and weight have a correlation among the given pairs -/
theorem only_height_weight_correlated :
  (cube_volume_edge_relationship ≠ Relationship.Correlated) ∧
  (distance_time_relationship ≠ Relationship.Correlated) ∧
  (height_eyesight_relationship ≠ Relationship.Correlated) ∧
  (height_weight_relationship = Relationship.Correlated) :=
sorry

end only_height_weight_correlated_l3219_321957


namespace sum_of_squares_of_roots_l3219_321906

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) :
  (2 * x₁^2 + 5 * x₁ - 12 = 0) →
  (2 * x₂^2 + 5 * x₂ - 12 = 0) →
  x₁ ≠ x₂ →
  x₁^2 + x₂^2 = 73/4 := by
sorry

end sum_of_squares_of_roots_l3219_321906


namespace system_solution_l3219_321966

theorem system_solution : 
  let x : ℚ := 8 / 47
  let y : ℚ := 138 / 47
  (7 * x = 10 - 3 * y) ∧ (4 * x = 5 * y - 14) := by
  sorry

end system_solution_l3219_321966


namespace pen_count_is_39_l3219_321950

/-- Calculate the final number of pens after a series of operations -/
def final_pen_count (initial : ℕ) (mike_gives : ℕ) (sharon_takes : ℕ) : ℕ :=
  ((initial + mike_gives) * 2) - sharon_takes

/-- Theorem stating that given the initial conditions, the final number of pens is 39 -/
theorem pen_count_is_39 :
  final_pen_count 7 22 19 = 39 := by
  sorry

#eval final_pen_count 7 22 19

end pen_count_is_39_l3219_321950


namespace painted_cubes_count_l3219_321973

/-- Represents a cube with given dimensions -/
structure Cube where
  size : Nat

/-- Represents a painted cube -/
structure PaintedCube extends Cube where
  painted : Bool

/-- Calculates the number of 1-inch cubes with at least one painted face -/
def paintedCubes (c : PaintedCube) : Nat :=
  c.size ^ 3 - (c.size - 2) ^ 3

/-- Theorem: In a 10×10×10 painted cube, 488 small cubes have at least one painted face -/
theorem painted_cubes_count :
  let c : PaintedCube := { size := 10, painted := true }
  paintedCubes c = 488 := by sorry

end painted_cubes_count_l3219_321973


namespace divisible_by_2_3_5_less_than_300_l3219_321926

theorem divisible_by_2_3_5_less_than_300 : 
  (Finset.filter (fun n : ℕ => n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) (Finset.range 300)).card = 9 := by
  sorry

end divisible_by_2_3_5_less_than_300_l3219_321926


namespace sin_period_from_symmetric_center_l3219_321912

/-- Given a function f(x) = sin(ωx), if the minimum distance from a symmetric center
    to the axis of symmetry is π/4, then the minimum positive period of f(x) is π. -/
theorem sin_period_from_symmetric_center (ω : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (ω * x)
  let min_distance_to_axis : ℝ := π / 4
  let period : ℝ := 2 * π / ω
  min_distance_to_axis = period / 2 → period = π :=
by
  sorry


end sin_period_from_symmetric_center_l3219_321912


namespace fraction_invariance_l3219_321907

theorem fraction_invariance (x y : ℝ) (h : x ≠ y) : 
  (3 * x) / (3 * x - 3 * y) = x / (x - y) := by
  sorry

end fraction_invariance_l3219_321907


namespace factorial_products_squares_l3219_321928

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (fun i => i + 1)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem factorial_products_squares :
  (is_perfect_square (factorial 7 * factorial 8)) ∧
  (¬ is_perfect_square (factorial 5 * factorial 6)) ∧
  (¬ is_perfect_square (factorial 5 * factorial 7)) ∧
  (¬ is_perfect_square (factorial 6 * factorial 7)) ∧
  (¬ is_perfect_square (factorial 6 * factorial 8)) :=
by sorry

end factorial_products_squares_l3219_321928


namespace estimated_y_at_x_100_l3219_321998

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 1.43 * x + 257

-- Theorem statement
theorem estimated_y_at_x_100 :
  regression_equation 100 = 400 := by
  sorry

end estimated_y_at_x_100_l3219_321998


namespace probability_sum_eleven_l3219_321901

def seven_sided_die : Finset Nat := Finset.range 7
def five_sided_die : Finset Nat := Finset.range 5

def total_outcomes : Nat := seven_sided_die.card * five_sided_die.card

def successful_outcomes : Finset (Nat × Nat) :=
  {(4, 4), (5, 3), (6, 2)}

theorem probability_sum_eleven :
  (successful_outcomes.card : ℚ) / total_outcomes = 3 / 35 := by
sorry

end probability_sum_eleven_l3219_321901


namespace f_positive_iff_f_plus_3abs_min_f_plus_3abs_min_value_l3219_321908

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for part (1)
theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x > 1 ∨ x < -5 := by sorry

-- Theorem for part (2)
theorem f_plus_3abs_min (x : ℝ) : f x + 3 * |x - 4| ≥ 9 := by sorry

-- Theorem for the minimum value
theorem f_plus_3abs_min_value : ∃ x : ℝ, f x + 3 * |x - 4| = 9 := by sorry

end f_positive_iff_f_plus_3abs_min_f_plus_3abs_min_value_l3219_321908


namespace dara_waiting_time_l3219_321905

/-- Represents the company's employment requirements and employee information --/
structure CompanyData where
  initial_min_age : ℕ
  age_increase_rate : ℕ
  age_increase_period : ℕ
  jane_age : ℕ
  tom_age_diff : ℕ
  tom_join_min_age : ℕ
  dara_internship_age : ℕ
  dara_internship_duration : ℕ
  dara_training_age : ℕ
  dara_training_duration : ℕ

/-- Calculates the waiting time for Dara to be eligible for employment --/
def calculate_waiting_time (data : CompanyData) : ℕ :=
  sorry

/-- Theorem stating that Dara has to wait 19 years before she can be employed --/
theorem dara_waiting_time (data : CompanyData) :
  data.initial_min_age = 25 ∧
  data.age_increase_rate = 1 ∧
  data.age_increase_period = 5 ∧
  data.jane_age = 28 ∧
  data.tom_age_diff = 10 ∧
  data.tom_join_min_age = 24 ∧
  data.dara_internship_age = 22 ∧
  data.dara_internship_duration = 3 ∧
  data.dara_training_age = 24 ∧
  data.dara_training_duration = 2 →
  calculate_waiting_time data = 19 :=
by sorry

end dara_waiting_time_l3219_321905


namespace range_of_difference_l3219_321932

theorem range_of_difference (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x = x^2 - 2*x) →
  (∀ y ∈ Set.Icc (-1) 3, ∃ x ∈ Set.Icc a b, f x = y) →
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (-1) 3) →
  2 ≤ b - a ∧ b - a ≤ 4 :=
by sorry

end range_of_difference_l3219_321932


namespace cos_2theta_value_l3219_321988

theorem cos_2theta_value (θ : Real) 
  (h : Real.sin (2 * θ) - 4 * Real.sin (θ + Real.pi / 3) * Real.sin (θ - Real.pi / 6) = Real.sqrt 3 / 3) : 
  Real.cos (2 * θ) = 1 / 3 := by
  sorry

end cos_2theta_value_l3219_321988


namespace hexagon_painting_arrangements_l3219_321956

/-- The number of ways to paint a hexagonal arrangement of equilateral triangles -/
def paint_arrangements : ℕ := 3^6 * 2^6

/-- The hexagonal arrangement consists of 6 inner sticks -/
def inner_sticks : ℕ := 6

/-- The number of available colors -/
def colors : ℕ := 3

/-- The number of triangles in the hexagonal arrangement -/
def triangles : ℕ := 6

/-- The number of ways to paint the inner sticks -/
def inner_stick_arrangements : ℕ := colors^inner_sticks

/-- The number of ways to complete each triangle given the two-color constraint -/
def triangle_completions : ℕ := 2^triangles

theorem hexagon_painting_arrangements :
  paint_arrangements = inner_stick_arrangements * triangle_completions :=
by sorry

end hexagon_painting_arrangements_l3219_321956


namespace cafe_chairs_minimum_l3219_321995

theorem cafe_chairs_minimum (indoor_tables outdoor_tables : ℕ)
  (indoor_min indoor_max outdoor_min outdoor_max : ℕ)
  (total_customers indoor_customers : ℕ) :
  indoor_tables = 9 →
  outdoor_tables = 11 →
  indoor_min = 6 →
  indoor_max = 10 →
  outdoor_min = 3 →
  outdoor_max = 5 →
  total_customers = 35 →
  indoor_customers = 18 →
  indoor_min ≤ indoor_max →
  outdoor_min ≤ outdoor_max →
  indoor_customers ≤ total_customers →
  (∀ t, t ≤ indoor_tables → indoor_min ≤ t * indoor_min) →
  (∀ t, t ≤ outdoor_tables → outdoor_min ≤ t * outdoor_min) →
  87 ≤ indoor_tables * indoor_min + outdoor_tables * outdoor_min :=
by
  sorry

#check cafe_chairs_minimum

end cafe_chairs_minimum_l3219_321995


namespace parking_lot_wheels_l3219_321933

/-- Calculates the total number of wheels in a parking lot --/
def total_wheels (num_cars num_motorcycles num_trucks num_vans : ℕ) : ℕ :=
  let car_wheels := 4
  let motorcycle_wheels := 2
  let truck_wheels := 6
  let van_wheels := 4
  num_cars * car_wheels + 
  num_motorcycles * motorcycle_wheels + 
  num_trucks * truck_wheels + 
  num_vans * van_wheels

/-- The number of wheels in Dylan's parents' vehicles --/
def parents_wheels : ℕ := 8

theorem parking_lot_wheels : 
  total_wheels 7 4 3 2 + parents_wheels = 62 := by sorry

end parking_lot_wheels_l3219_321933


namespace average_equals_one_l3219_321958

theorem average_equals_one (x : ℝ) : 
  (5 + (-1) + (-2) + x) / 4 = 1 → x = 2 := by
sorry

end average_equals_one_l3219_321958


namespace M_equals_N_l3219_321924

/-- Definition of set M -/
def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}

/-- Definition of set N -/
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

/-- Theorem stating that M equals N -/
theorem M_equals_N : M = N := by sorry

end M_equals_N_l3219_321924


namespace greatest_multiple_of_four_cubed_less_than_800_l3219_321991

theorem greatest_multiple_of_four_cubed_less_than_800 :
  ∃ (x : ℕ), x = 8 ∧ 
  (∀ (y : ℕ), y > 0 ∧ 4 ∣ y ∧ y^3 < 800 → y ≤ x) :=
by sorry

end greatest_multiple_of_four_cubed_less_than_800_l3219_321991
