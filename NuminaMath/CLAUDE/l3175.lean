import Mathlib

namespace dealership_sales_expectation_l3175_317530

/-- The number of trucks expected to be sold -/
def expected_trucks : ℕ := 30

/-- The number of vans expected to be sold -/
def expected_vans : ℕ := 15

/-- The ratio of trucks to SUVs -/
def truck_suv_ratio : ℚ := 3 / 5

/-- The ratio of SUVs to vans -/
def suv_van_ratio : ℚ := 2 / 1

/-- The number of SUVs the dealership should expect to sell -/
def expected_suvs : ℕ := 30

theorem dealership_sales_expectation :
  (expected_trucks : ℚ) / truck_suv_ratio ≥ expected_suvs ∧
  suv_van_ratio * expected_vans = expected_suvs :=
sorry

end dealership_sales_expectation_l3175_317530


namespace equation_solution_l3175_317534

theorem equation_solution : ∃ x : ℝ, (2 / (x + 5) = 1 / (3 * x)) ∧ x = 1 := by
  sorry

end equation_solution_l3175_317534


namespace smallest_three_digit_multiple_of_17_l3175_317500

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by
  sorry

end smallest_three_digit_multiple_of_17_l3175_317500


namespace quadratic_factorization_l3175_317511

theorem quadratic_factorization (x : ℝ) : 9 * x^2 - 6 * x + 1 = (3 * x - 1)^2 := by
  sorry

end quadratic_factorization_l3175_317511


namespace complex_fraction_simplification_l3175_317508

theorem complex_fraction_simplification :
  let a : ℂ := 4 + 6*I
  let b : ℂ := 4 - 6*I
  (a/b) * (b/a) + (b/a) * (a/b) = 2 := by sorry

end complex_fraction_simplification_l3175_317508


namespace man_mass_on_boat_l3175_317522

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sink_height water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sink_height * water_density

/-- Theorem stating the mass of the man in the given problem. -/
theorem man_mass_on_boat : 
  let boat_length : ℝ := 8
  let boat_breadth : ℝ := 3
  let boat_sink_height : ℝ := 0.01  -- 1 cm in meters
  let water_density : ℝ := 1000     -- kg/m³
  mass_of_man boat_length boat_breadth boat_sink_height water_density = 240 := by
  sorry


end man_mass_on_boat_l3175_317522


namespace pencil_count_l3175_317578

/-- The total number of pencils in the drawer after Sarah's addition -/
def total_pencils (initial : ℕ) (mike_added : ℕ) (sarah_added : ℕ) : ℕ :=
  initial + mike_added + sarah_added

/-- Theorem stating the total number of pencils after all additions -/
theorem pencil_count (x : ℕ) :
  total_pencils 41 30 x = 71 + x := by
  sorry

end pencil_count_l3175_317578


namespace main_theorem_l3175_317509

-- Define the type for multiplicative functions
def MultFun := ℕ → Fin 2

-- Define the property of being multiplicative
def is_multiplicative (f : MultFun) : Prop :=
  ∀ a b : ℕ, f (a * b) = f a * f b

theorem main_theorem (a b c d : ℕ) (f g : MultFun)
  (h1 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h2 : a * b * c * d ≠ 1)
  (h3 : Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd a d = 1 ∧
        Nat.gcd b c = 1 ∧ Nat.gcd b d = 1 ∧ Nat.gcd c d = 1)
  (h4 : is_multiplicative f ∧ is_multiplicative g)
  (h5 : ∀ n : ℕ, f (a * n + b) = g (c * n + d)) :
  (∀ n : ℕ, f (a * n + b) = 0 ∧ g (c * n + d) = 0) ∨
  (∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, Nat.gcd n k = 1 → f n = 1 ∧ g n = 1) :=
by sorry

end main_theorem_l3175_317509


namespace prob_odd_then_even_eq_17_45_l3175_317505

/-- A box containing 6 cards numbered 1 to 6 -/
def Box : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The probability of drawing a specific card from the box -/
def prob_draw (n : ℕ) : ℚ := if n ∈ Box then 1 / 6 else 0

/-- The probability of drawing an even number from the remaining cards after drawing 'a' -/
def prob_even_after (a : ℕ) : ℚ :=
  let remaining := Box.filter (λ x => x > a)
  let even_remaining := remaining.filter (λ x => x % 2 = 0)
  (even_remaining.card : ℚ) / remaining.card

/-- The probability of the event: first draw is odd and second draw is even -/
def prob_odd_then_even : ℚ :=
  (prob_draw 1 * prob_even_after 1) +
  (prob_draw 3 * prob_even_after 3) +
  (prob_draw 5 * prob_even_after 5)

theorem prob_odd_then_even_eq_17_45 : prob_odd_then_even = 17 / 45 := by
  sorry

end prob_odd_then_even_eq_17_45_l3175_317505


namespace art_performance_probability_l3175_317513

def artDepartment : Finset Nat := {1, 2, 3, 4}
def firstGrade : Finset Nat := {1, 2}
def secondGrade : Finset Nat := {3, 4}

theorem art_performance_probability :
  let totalSelections := Finset.powerset artDepartment |>.filter (λ s => s.card = 2)
  let differentGradeSelections := totalSelections.filter (λ s => s ∩ firstGrade ≠ ∅ ∧ s ∩ secondGrade ≠ ∅)
  (differentGradeSelections.card : ℚ) / totalSelections.card = 2 / 3 := by
sorry

end art_performance_probability_l3175_317513


namespace value_of_S_l3175_317586

theorem value_of_S : ∀ S : ℕ, 
  S = 6 * 10000 + 5 * 1000 + 4 * 10 + 3 * 1 → S = 65043 := by
  sorry

end value_of_S_l3175_317586


namespace hyperbola_asymptote_tangent_to_circle_l3175_317558

/-- The value of m for which the asymptotes of the hyperbola y² - x²/m² = 1 
    are tangent to the circle x² + y² - 4y + 3 = 0, given m > 0 -/
theorem hyperbola_asymptote_tangent_to_circle (m : ℝ) 
  (hm : m > 0)
  (h_hyperbola : ∀ x y : ℝ, y^2 - x^2/m^2 = 1 → 
    (∃ k : ℝ, y = k*x/m ∨ y = -k*x/m))
  (h_circle : ∀ x y : ℝ, x^2 + y^2 - 4*y + 3 = 0 → 
    (x - 0)^2 + (y - 2)^2 = 1)
  (h_tangent : ∀ x y : ℝ, (y = x/m ∨ y = -x/m) → 
    ((0 - x)^2 + (2 - y)^2 = 1)) :
  m = Real.sqrt 3 / 3 :=
sorry

end hyperbola_asymptote_tangent_to_circle_l3175_317558


namespace total_carrots_l3175_317506

theorem total_carrots (sally fred mary : ℕ) 
  (h1 : sally = 6) 
  (h2 : fred = 4) 
  (h3 : mary = 10) : 
  sally + fred + mary = 20 := by
  sorry

end total_carrots_l3175_317506


namespace expression_equality_l3175_317517

theorem expression_equality : (1 + 0.25) / (2 * (3/4) - 0.75) + (3 * 0.5) / (1.5 + 3) = 2 := by
  sorry

end expression_equality_l3175_317517


namespace triangle_side_length_l3175_317573

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π/3 →  -- 60 degrees in radians
  B = π/4 →  -- 45 degrees in radians
  b = 2 → 
  (a / Real.sin A = b / Real.sin B) →  -- Law of sines
  a = Real.sqrt 6 := by
sorry

end triangle_side_length_l3175_317573


namespace optimal_strategy_l3175_317596

/-- Represents the expected score when answering question A first -/
def E_xi (P1 P2 a b : ℝ) : ℝ := a * P1 * (1 - P2) + (a + b) * P1 * P2

/-- Represents the expected score when answering question B first -/
def E_epsilon (P1 P2 a b : ℝ) : ℝ := b * P2 * (1 - P1) + (a + b) * P1 * P2

/-- The theorem states that given P1 = 2/5, a = 10, b = 20, 
    choosing to answer question A first is optimal when 0 ≤ P2 ≤ 1/4 -/
theorem optimal_strategy (P2 : ℝ) :
  0 ≤ P2 ∧ P2 ≤ 1/4 ↔ E_xi (2/5) P2 10 20 ≥ E_epsilon (2/5) P2 10 20 := by
  sorry

end optimal_strategy_l3175_317596


namespace smallest_M_for_Q_less_than_three_fourths_l3175_317594

def is_multiple_of_six (M : ℕ) : Prop := ∃ k : ℕ, M = 6 * k

def Q (M : ℕ) : ℚ := (⌈(2 / 3 : ℚ) * M + 1⌉ : ℚ) / (M + 1 : ℚ)

theorem smallest_M_for_Q_less_than_three_fourths :
  ∀ M : ℕ, is_multiple_of_six M → (Q M < 3 / 4 → M ≥ 6) ∧ (Q 6 < 3 / 4) := by sorry

end smallest_M_for_Q_less_than_three_fourths_l3175_317594


namespace fifteen_switch_network_connections_l3175_317510

/-- Represents a network of switches -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ

/-- Calculates the total number of connections in the network -/
def total_connections (network : SwitchNetwork) : ℕ :=
  (network.num_switches * network.connections_per_switch) / 2

/-- Theorem: In a network of 15 switches, where each switch is connected to 4 others,
    the total number of connections is 30 -/
theorem fifteen_switch_network_connections :
  let network : SwitchNetwork := ⟨15, 4⟩
  total_connections network = 30 := by
  sorry


end fifteen_switch_network_connections_l3175_317510


namespace max_k_value_l3175_317553

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∀ k : ℝ, (1/m + 2/(1-2*m) ≥ k) → k ≤ 8) ∧
  (∃ k : ℝ, k = 8 ∧ 1/m + 2/(1-2*m) ≥ k) :=
sorry

end max_k_value_l3175_317553


namespace distance_per_interval_l3175_317588

-- Define the total distance walked
def total_distance : ℝ := 3

-- Define the total time taken
def total_time : ℝ := 45

-- Define the interval time
def interval_time : ℝ := 15

-- Theorem to prove
theorem distance_per_interval : 
  (total_distance / (total_time / interval_time)) = 1 := by
  sorry

end distance_per_interval_l3175_317588


namespace expected_red_balls_l3175_317541

/-- Given a bag of balls with some red and some white, prove that the expected
    number of red balls is proportional to the number of red draws in a series
    of random draws with replacement. -/
theorem expected_red_balls
  (total_balls : ℕ)
  (total_draws : ℕ)
  (red_draws : ℕ)
  (h_total_balls : total_balls = 12)
  (h_total_draws : total_draws = 200)
  (h_red_draws : red_draws = 50) :
  (total_balls : ℚ) * (red_draws : ℚ) / (total_draws : ℚ) = 3 :=
sorry

end expected_red_balls_l3175_317541


namespace linear_equation_system_l3175_317533

theorem linear_equation_system (a b c : ℝ) 
  (eq1 : a + 2*b - 3*c = 4)
  (eq2 : 5*a - 6*b + 7*c = 8) :
  9*a + 2*b - 5*c = 24 := by
  sorry

end linear_equation_system_l3175_317533


namespace rowing_problem_l3175_317521

/-- A rowing problem in a river with current and headwind -/
theorem rowing_problem (downstream_speed current_speed headwind_reduction : ℝ) 
  (h1 : downstream_speed = 22)
  (h2 : current_speed = 4.5)
  (h3 : headwind_reduction = 1.5) :
  let still_water_speed := downstream_speed - current_speed
  still_water_speed - current_speed - headwind_reduction = 11.5 := by
  sorry

end rowing_problem_l3175_317521


namespace range_of_a_when_P_or_Q_false_l3175_317501

-- Define the function f
def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 2

-- Define proposition P
def P (a : ℝ) : Prop := ∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = 0

-- Define proposition Q
def Q (a : ℝ) : Prop := ∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0

-- Define the set of a values where either P or Q is false
def A : Set ℝ := {a | -1 < a ∧ a < 0 ∨ 0 < a ∧ a < 1}

-- Theorem statement
theorem range_of_a_when_P_or_Q_false :
  ∀ a : ℝ, (¬P a ∨ ¬Q a) ↔ a ∈ A :=
sorry

end range_of_a_when_P_or_Q_false_l3175_317501


namespace sector_angle_l3175_317587

theorem sector_angle (area : Real) (radius : Real) (h1 : area = 3 * Real.pi / 16) (h2 : radius = 1) :
  (2 * area) / (radius ^ 2) = 3 * Real.pi / 8 := by
  sorry

end sector_angle_l3175_317587


namespace average_study_time_difference_l3175_317520

/-- The differences in study times (Mia - Liam) for each day of the week --/
def study_time_differences : List Int := [15, -5, 25, 0, -15, 20, 10]

/-- The number of days in a week --/
def days_in_week : Nat := 7

/-- Theorem: The average difference in study time per day is 7 minutes --/
theorem average_study_time_difference :
  (study_time_differences.sum : ℚ) / days_in_week = 7 := by
  sorry

end average_study_time_difference_l3175_317520


namespace games_for_champion_l3175_317591

/-- Represents a single-elimination tournament -/
structure Tournament where
  num_players : ℕ
  single_elimination : Bool

/-- The number of games required to determine the champion in a single-elimination tournament -/
def games_required (t : Tournament) : ℕ :=
  t.num_players - 1

theorem games_for_champion (t : Tournament) (h1 : t.single_elimination = true) (h2 : t.num_players = 512) :
  games_required t = 511 := by
  sorry

end games_for_champion_l3175_317591


namespace unique_prime_pair_l3175_317561

theorem unique_prime_pair : ∃! (p q : ℕ), 
  Prime p ∧ Prime q ∧ 
  ∃ r : ℕ, Prime r ∧ 
  (1 : ℚ) + (p^q - q^p : ℚ) / (p + q : ℚ) = r := by
  sorry

end unique_prime_pair_l3175_317561


namespace scientific_notation_3900000000_l3175_317568

theorem scientific_notation_3900000000 :
  3900000000 = 3.9 * (10 ^ 9) := by sorry

end scientific_notation_3900000000_l3175_317568


namespace quadratic_solution_positive_l3175_317556

theorem quadratic_solution_positive (x : ℝ) : 
  x > 0 ∧ 4 * x^2 + 8 * x - 20 = 0 ↔ x = Real.sqrt 6 - 1 :=
by sorry

end quadratic_solution_positive_l3175_317556


namespace rate_squares_sum_l3175_317571

theorem rate_squares_sum : ∃ (b j s : ℕ),
  3 * b + 2 * j + 4 * s = 70 ∧
  4 * b + 3 * j + 2 * s = 88 ∧
  b^2 + j^2 + s^2 = 405 := by
sorry

end rate_squares_sum_l3175_317571


namespace blue_packs_bought_l3175_317529

/- Define the problem parameters -/
def white_pack_size : ℕ := 6
def blue_pack_size : ℕ := 9
def white_packs_bought : ℕ := 5
def total_tshirts : ℕ := 57

/- Define the theorem -/
theorem blue_packs_bought :
  ∃ (blue_packs : ℕ),
    blue_packs * blue_pack_size + white_packs_bought * white_pack_size = total_tshirts ∧
    blue_packs = 3 := by
  sorry

end blue_packs_bought_l3175_317529


namespace coincidence_time_l3175_317502

-- Define the movement pattern
def move_distance (n : ℕ) : ℤ := if n % 2 = 0 then -n else n

-- Define the position after n moves
def position (n : ℕ) : ℤ := (List.range n).map move_distance |>.sum

-- Define the total distance traveled after n moves
def total_distance (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the speed
def speed : ℕ := 4

-- Define the position of point A
def point_A : ℤ := -24

-- Theorem to prove
theorem coincidence_time :
  ∃ n : ℕ, position n = point_A ∧ (total_distance n / speed : ℚ) = 294 := by
  sorry


end coincidence_time_l3175_317502


namespace percentage_problem_l3175_317507

theorem percentage_problem (N : ℝ) (P : ℝ) 
  (h1 : 0.8 * N = 240) 
  (h2 : (P / 100) * N = 60) : 
  P = 20 := by
sorry

end percentage_problem_l3175_317507


namespace cost_calculation_theorem_l3175_317566

/-- Represents the cost calculation for purchasing table tennis equipment --/
def cost_calculation (x : ℕ) : Prop :=
  let racket_price : ℕ := 80
  let ball_price : ℕ := 20
  let racket_quantity : ℕ := 20
  let option1_cost : ℕ := racket_price * racket_quantity
  let option2_cost : ℕ := (racket_price * racket_quantity + ball_price * x) * 9 / 10
  x > 20 → option1_cost = 1600 ∧ option2_cost = 1440 + 18 * x

/-- Theorem stating the cost calculation for purchasing table tennis equipment --/
theorem cost_calculation_theorem (x : ℕ) : cost_calculation x := by
  sorry

#check cost_calculation_theorem

end cost_calculation_theorem_l3175_317566


namespace equation_value_l3175_317523

theorem equation_value (x y : ℝ) (h : x^2 - 3*y - 5 = 0) : 2*x^2 - 6*y - 6 = 4 := by
  sorry

end equation_value_l3175_317523


namespace geometric_sequence_operations_l3175_317572

-- Define a geometric sequence
def IsGeometric (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

-- Define the problem statement
theorem geometric_sequence_operations
  (a b : ℕ → ℝ)
  (ha : IsGeometric a)
  (hb : IsGeometric b)
  (hb_nonzero : ∀ n, b n ≠ 0) :
  IsGeometric (fun n ↦ a n * b n) ∧
  IsGeometric (fun n ↦ a n / b n) :=
by sorry

end geometric_sequence_operations_l3175_317572


namespace quadratic_inequality_range_l3175_317543

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → a ∈ Set.Icc (-1) 3 :=
by sorry

end quadratic_inequality_range_l3175_317543


namespace expression_equality_l3175_317597

theorem expression_equality : 
  Real.sqrt 12 + 2 * Real.tan (π / 4) - Real.sin (π / 3) - (1 / 2)⁻¹ = (3 * Real.sqrt 3) / 2 := by
  sorry

end expression_equality_l3175_317597


namespace ratio_problem_l3175_317563

theorem ratio_problem (first_part : ℝ) (percent : ℝ) (second_part : ℝ) : 
  first_part = 4 →
  percent = 20 →
  first_part / second_part = percent / 100 →
  second_part = 20 := by
sorry

end ratio_problem_l3175_317563


namespace complex_modulus_problem_l3175_317546

theorem complex_modulus_problem (z : ℂ) (h : (1 + Complex.I) * z = (1 - Complex.I)^2) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l3175_317546


namespace diamond_is_conditional_l3175_317595

/-- Represents shapes in a flowchart --/
inductive FlowchartShape
  | Diamond
  | Rectangle
  | Oval

/-- Represents logical structures in an algorithm --/
inductive LogicalStructure
  | Conditional
  | Loop
  | Sequential

/-- A function that maps flowchart shapes to logical structures --/
def shapeToStructure : FlowchartShape → LogicalStructure
  | FlowchartShape.Diamond => LogicalStructure.Conditional
  | FlowchartShape.Rectangle => LogicalStructure.Sequential
  | FlowchartShape.Oval => LogicalStructure.Sequential

/-- Theorem stating that a diamond shape in a flowchart represents a conditional structure --/
theorem diamond_is_conditional :
  shapeToStructure FlowchartShape.Diamond = LogicalStructure.Conditional :=
by
  sorry

end diamond_is_conditional_l3175_317595


namespace orange_harvest_existence_l3175_317570

theorem orange_harvest_existence :
  ∃ (A B C D : ℕ), A + B + C + D = 56 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 := by
  sorry

end orange_harvest_existence_l3175_317570


namespace bakery_inventory_theorem_l3175_317557

/-- Represents the inventory and sales of a bakery --/
structure BakeryInventory where
  cheesecakes_display : ℕ
  cheesecakes_fridge : ℕ
  cherry_pies_ready : ℕ
  cherry_pies_oven : ℕ
  chocolate_eclairs_counter : ℕ
  chocolate_eclairs_pantry : ℕ
  cheesecakes_sold : ℕ
  cherry_pies_sold : ℕ
  chocolate_eclairs_sold : ℕ

/-- Calculates the total number of desserts left to sell --/
def desserts_left_to_sell (inventory : BakeryInventory) : ℕ :=
  (inventory.cheesecakes_display + inventory.cheesecakes_fridge - inventory.cheesecakes_sold) +
  (inventory.cherry_pies_ready + inventory.cherry_pies_oven - inventory.cherry_pies_sold) +
  (inventory.chocolate_eclairs_counter + inventory.chocolate_eclairs_pantry - inventory.chocolate_eclairs_sold)

/-- Theorem stating that given the specific inventory and sales, there are 62 desserts left to sell --/
theorem bakery_inventory_theorem (inventory : BakeryInventory) 
  (h1 : inventory.cheesecakes_display = 10)
  (h2 : inventory.cheesecakes_fridge = 15)
  (h3 : inventory.cherry_pies_ready = 12)
  (h4 : inventory.cherry_pies_oven = 20)
  (h5 : inventory.chocolate_eclairs_counter = 20)
  (h6 : inventory.chocolate_eclairs_pantry = 10)
  (h7 : inventory.cheesecakes_sold = 7)
  (h8 : inventory.cherry_pies_sold = 8)
  (h9 : inventory.chocolate_eclairs_sold = 10) :
  desserts_left_to_sell inventory = 62 := by
  sorry

end bakery_inventory_theorem_l3175_317557


namespace intersection_of_A_and_B_l3175_317552

-- Define sets A and B
def A : Set ℝ := Set.univ
def B : Set ℝ := {x : ℝ | x ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = B := by sorry

end intersection_of_A_and_B_l3175_317552


namespace binomial_expansion_theorem_l3175_317528

theorem binomial_expansion_theorem (x a : ℝ) (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ 
    (n.choose 3) * x^(n - 3) * a^3 = 210 * k ∧
    (n.choose 4) * x^(n - 4) * a^4 = 420 * k ∧
    (n.choose 5) * x^(n - 5) * a^5 = 630 * k) →
  n = 19 := by
sorry

end binomial_expansion_theorem_l3175_317528


namespace exactly_three_valid_sequences_l3175_317527

/-- An arithmetic sequence with the given properties -/
structure ValidSequence where
  a₁ : ℕ
  d : ℕ
  h_a₁_single_digit : a₁ < 10
  h_100_in_seq : ∃ n : ℕ, a₁ + (n - 1) * d = 100
  h_3103_in_seq : ∃ m : ℕ, a₁ + (m - 1) * d = 3103
  h_max_terms : ∀ k : ℕ, a₁ + (k - 1) * d ≤ 3103 → k ≤ 240

/-- The set of all valid sequences -/
def validSequences : Set ValidSequence := {s | s.a₁ + 239 * s.d ≥ 3103}

theorem exactly_three_valid_sequences :
  ∃! (s₁ s₂ s₃ : ValidSequence),
    validSequences = {s₁, s₂, s₃} ∧
    s₁.a₁ = 9 ∧ s₁.d = 13 ∧
    s₂.a₁ = 1 ∧ s₂.d = 33 ∧
    s₃.a₁ = 9 ∧ s₃.d = 91 :=
  sorry

end exactly_three_valid_sequences_l3175_317527


namespace quadratic_sum_of_coefficients_l3175_317516

theorem quadratic_sum_of_coefficients (a b : ℝ) : 
  (∀ x, a * x^2 + b * x - 2 = 0 ↔ x = -2 ∨ x = 1/3) → 
  a + b = 8 := by
  sorry

end quadratic_sum_of_coefficients_l3175_317516


namespace basketball_score_proof_l3175_317567

theorem basketball_score_proof (total_points : ℕ) : 
  (∃ (linda_points maria_points other_points : ℕ),
    linda_points = total_points / 5 ∧ 
    maria_points = total_points * 3 / 8 ∧
    other_points ≤ 16 ∧
    linda_points + maria_points + 18 + other_points = total_points ∧
    other_points ≤ 8 * 2) →
  (∃ (other_points : ℕ), 
    other_points = 16 ∧
    other_points ≤ 8 * 2 ∧
    ∃ (linda_points maria_points : ℕ),
      linda_points = total_points / 5 ∧ 
      maria_points = total_points * 3 / 8 ∧
      linda_points + maria_points + 18 + other_points = total_points) :=
by sorry

end basketball_score_proof_l3175_317567


namespace twenty_loaves_slices_thirty_loaves_not_enough_l3175_317599

-- Define the number of slices per loaf
def slices_per_loaf : ℕ := 12

-- Define the function to calculate total slices
def total_slices (loaves : ℕ) : ℕ := slices_per_loaf * loaves

-- Theorem 1
theorem twenty_loaves_slices : total_slices 20 = 240 := by sorry

-- Theorem 2
theorem thirty_loaves_not_enough (children : ℕ) (h : children = 385) : 
  total_slices 30 < children := by sorry

end twenty_loaves_slices_thirty_loaves_not_enough_l3175_317599


namespace ways_to_buy_three_items_eq_646_l3175_317531

/-- Represents the inventory of a store --/
structure Inventory where
  headphones : Nat
  mice : Nat
  keyboards : Nat
  keyboard_mouse_sets : Nat
  headphone_mouse_sets : Nat

/-- Calculates the number of ways to buy three items (headphones, keyboard, mouse) --/
def ways_to_buy_three_items (inv : Inventory) : Nat :=
  inv.keyboard_mouse_sets * inv.headphones +
  inv.headphone_mouse_sets * inv.keyboards +
  inv.headphones * inv.mice * inv.keyboards

/-- The theorem stating the number of ways to buy three items --/
theorem ways_to_buy_three_items_eq_646 (inv : Inventory) 
  (h1 : inv.headphones = 9)
  (h2 : inv.mice = 13)
  (h3 : inv.keyboards = 5)
  (h4 : inv.keyboard_mouse_sets = 4)
  (h5 : inv.headphone_mouse_sets = 5) :
  ways_to_buy_three_items inv = 646 := by
  sorry


end ways_to_buy_three_items_eq_646_l3175_317531


namespace unique_solution_cube_equation_l3175_317564

theorem unique_solution_cube_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 ∧ x = 27 := by
  sorry

end unique_solution_cube_equation_l3175_317564


namespace diagonals_to_sides_ratio_for_pentagon_l3175_317555

-- Define the number of diagonals function
def num_diagonals (n : ℕ) : ℚ := n * (n - 3) / 2

-- Theorem statement
theorem diagonals_to_sides_ratio_for_pentagon :
  let n : ℕ := 5
  (num_diagonals n) / n = 1 := by sorry

end diagonals_to_sides_ratio_for_pentagon_l3175_317555


namespace matrix_power_2023_l3175_317504

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 0, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1, 2023; 0, 1] := by sorry

end matrix_power_2023_l3175_317504


namespace sum_product_inequality_l3175_317540

theorem sum_product_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hsum : a + b + c + d = 4) :
  a * b + b * c + c * d + d * a ≤ 4 := by
sorry

end sum_product_inequality_l3175_317540


namespace bounce_count_correct_l3175_317559

/-- The smallest positive integer k such that 800 * (0.4^k) < 5 -/
def bounce_count : ℕ := 6

/-- The initial height of the ball in feet -/
def initial_height : ℝ := 800

/-- The ratio of the height after each bounce to the previous height -/
def bounce_ratio : ℝ := 0.4

/-- The target height in feet -/
def target_height : ℝ := 5

theorem bounce_count_correct : 
  (∀ k : ℕ, k < bounce_count → initial_height * (bounce_ratio ^ k) ≥ target_height) ∧
  initial_height * (bounce_ratio ^ bounce_count) < target_height :=
sorry

end bounce_count_correct_l3175_317559


namespace eating_contest_l3175_317538

/-- Eating contest problem -/
theorem eating_contest (hotdog_weight burger_weight pie_weight : ℕ)
  (jacob_pies noah_burgers mason_hotdogs : ℕ)
  (h1 : hotdog_weight = 2)
  (h2 : burger_weight = 5)
  (h3 : pie_weight = 10)
  (h4 : jacob_pies + 3 = noah_burgers)
  (h5 : mason_hotdogs = 3 * jacob_pies)
  (h6 : mason_hotdogs * hotdog_weight = 30) :
  noah_burgers = 8 := by
  sorry

end eating_contest_l3175_317538


namespace rachel_winter_clothing_boxes_l3175_317547

theorem rachel_winter_clothing_boxes : 
  let scarves_per_box : ℕ := 3
  let mittens_per_box : ℕ := 4
  let total_pieces : ℕ := 49
  let pieces_per_box : ℕ := scarves_per_box + mittens_per_box
  let num_boxes : ℕ := total_pieces / pieces_per_box
  num_boxes = 7 := by
sorry

end rachel_winter_clothing_boxes_l3175_317547


namespace max_value_theorem_max_value_achieved_l3175_317545

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + 5 * b < 100) :
  ab * (100 - 4 * a - 5 * b) ≤ 50000 / 27 :=
by sorry

theorem max_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 4 * a + 5 * b < 100 ∧
  ab * (100 - 4 * a - 5 * b) > 50000 / 27 - ε :=
by sorry

end max_value_theorem_max_value_achieved_l3175_317545


namespace quadratic_equal_roots_l3175_317544

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + (2*m - 5) * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 + (2*m - 5) * y + 12 = 0 → y = x) ↔ 
  m = 8.5 ∨ m = -3.5 := by
sorry

end quadratic_equal_roots_l3175_317544


namespace quadratic_root_k_range_l3175_317574

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - k*x - 2

-- Theorem statement
theorem quadratic_root_k_range :
  ∀ k : ℝ, (∃ x : ℝ, 2 < x ∧ x < 5 ∧ f k x = 0) → (1 < k ∧ k < 23/5) :=
by sorry

end quadratic_root_k_range_l3175_317574


namespace min_sum_of_fractions_l3175_317577

def Digits : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem min_sum_of_fractions (A B C D E : Nat) 
  (h1 : A ∈ Digits) (h2 : B ∈ Digits) (h3 : C ∈ Digits) (h4 : D ∈ Digits) (h5 : E ∈ Digits)
  (h6 : A ≠ B) (h7 : A ≠ C) (h8 : A ≠ D) (h9 : A ≠ E)
  (h10 : B ≠ C) (h11 : B ≠ D) (h12 : B ≠ E)
  (h13 : C ≠ D) (h14 : C ≠ E)
  (h15 : D ≠ E) :
  (A : ℚ) / B + (C : ℚ) / D + (E : ℚ) / 9 ≥ 125 / 168 :=
sorry

end min_sum_of_fractions_l3175_317577


namespace carpet_shaded_area_l3175_317514

/-- The total shaded area on a square carpet -/
theorem carpet_shaded_area (S T : ℝ) : 
  S > 0 ∧ T > 0 ∧ (12 : ℝ) / S = 4 ∧ S / T = 2 →
  S^2 + 4 * T^2 = 18 :=
by sorry

end carpet_shaded_area_l3175_317514


namespace max_volume_l3175_317526

/-- Represents a tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  -- AB is perpendicular to BC and CD
  ab_perp_bc : True
  ab_perp_cd : True
  -- Length of BC is 2
  bc_length : ℝ
  bc_eq_two : bc_length = 2
  -- Dihedral angle between AB and CD is 60°
  dihedral_angle : ℝ
  dihedral_angle_eq_sixty : dihedral_angle = 60
  -- Circumradius is √5
  circumradius : ℝ
  circumradius_eq_sqrt_five : circumradius = Real.sqrt 5

/-- The volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- The maximum possible volume of the tetrahedron -/
theorem max_volume (t : Tetrahedron) : volume t ≤ 2 * Real.sqrt 3 := by
  sorry

end max_volume_l3175_317526


namespace unbroken_seashells_l3175_317584

theorem unbroken_seashells (total : ℕ) (broken : ℕ) (h1 : total = 23) (h2 : broken = 11) :
  total - broken = 12 := by
  sorry

end unbroken_seashells_l3175_317584


namespace projection_of_A_on_Oxz_l3175_317589

/-- The projection of a point (x, y, z) onto the Oxz plane is (x, 0, z) -/
def proj_oxz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p.1, 0, p.2.2)

/-- Point A in 3D space -/
def A : ℝ × ℝ × ℝ := (2, 3, 6)

/-- Point B is the projection of A onto the Oxz plane -/
def B : ℝ × ℝ × ℝ := proj_oxz A

theorem projection_of_A_on_Oxz :
  B = (2, 0, 6) := by sorry

end projection_of_A_on_Oxz_l3175_317589


namespace farmer_picked_thirty_today_l3175_317515

/-- Represents the number of tomatoes picked today by a farmer -/
def tomatoes_picked_today (initial : ℕ) (picked_yesterday : ℕ) (left_after_today : ℕ) : ℕ :=
  initial - picked_yesterday - left_after_today

/-- Theorem stating that the farmer picked 30 tomatoes today -/
theorem farmer_picked_thirty_today :
  tomatoes_picked_today 171 134 7 = 30 := by
  sorry

end farmer_picked_thirty_today_l3175_317515


namespace casper_candies_l3175_317580

/-- The number of candies Casper originally had -/
def original_candies : ℕ := 176

/-- The number of candies Casper gave to his brother on the first day -/
def candies_to_brother : ℕ := 3

/-- The number of candies Casper gave to his sister on the second day -/
def candies_to_sister : ℕ := 5

/-- The number of candies Casper ate on the third day -/
def final_candies : ℕ := 10

theorem casper_candies :
  let remaining_day1 := original_candies * 3 / 4 - candies_to_brother
  let remaining_day2 := remaining_day1 / 2 - candies_to_sister
  remaining_day2 = final_candies := by sorry

end casper_candies_l3175_317580


namespace people_eating_both_veg_and_nonveg_l3175_317593

theorem people_eating_both_veg_and_nonveg (veg_only : ℕ) (nonveg_only : ℕ) (total_veg : ℕ) 
  (h1 : veg_only = 15)
  (h2 : nonveg_only = 8)
  (h3 : total_veg = 26) :
  total_veg - veg_only = 11 := by
  sorry

#check people_eating_both_veg_and_nonveg

end people_eating_both_veg_and_nonveg_l3175_317593


namespace quadratic_integer_roots_l3175_317575

theorem quadratic_integer_roots (b : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ (x^2 - b*x + 3*b = 0) ∧ (y^2 - b*y + 3*b = 0)) → 
  (b = 9 ∨ b = -6) := by
sorry

end quadratic_integer_roots_l3175_317575


namespace invalid_prism_diagonals_l3175_317549

theorem invalid_prism_diagonals : ¬∃ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a^2 + b^2 = 5^2 ∨ a^2 + b^2 = 12^2 ∨ a^2 + b^2 = 13^2) ∧
  (b^2 + c^2 = 5^2 ∨ b^2 + c^2 = 12^2 ∨ b^2 + c^2 = 13^2) ∧
  (a^2 + c^2 = 5^2 ∨ a^2 + c^2 = 12^2 ∨ a^2 + c^2 = 13^2) ∧
  (a^2 + b^2 + c^2 = 14^2) :=
by sorry

end invalid_prism_diagonals_l3175_317549


namespace dice_probability_l3175_317548

def num_dice : ℕ := 15
def num_ones : ℕ := 3
def prob_one : ℚ := 1/6
def prob_not_one : ℚ := 5/6

theorem dice_probability : 
  (Nat.choose num_dice num_ones : ℚ) * prob_one ^ num_ones * prob_not_one ^ (num_dice - num_ones) = 
  455 * (1/6)^3 * (5/6)^12 := by sorry

end dice_probability_l3175_317548


namespace arithmetic_sequence_formula_l3175_317550

/-- An increasing arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) ∧  -- Arithmetic sequence
  (∀ n, a (n + 1) > a n) ∧  -- Increasing
  (a 1 = 1) ∧  -- a_1 = 1
  (a 3 = (a 2)^2 - 4)  -- a_3 = a_2^2 - 4

/-- The theorem stating the general formula for the sequence -/
theorem arithmetic_sequence_formula (a : ℕ → ℝ) 
    (h : ArithmeticSequence a) : 
    ∀ n : ℕ, a n = 3 * n - 2 := by
  sorry


end arithmetic_sequence_formula_l3175_317550


namespace all_equations_are_equalities_negative_two_solves_equation_one_and_negative_two_solve_equation_l3175_317542

-- Define what it means for a real number to be a solution to an equation
def IsSolution (x : ℝ) (f : ℝ → ℝ) : Prop := f x = 0

-- All equations are equalities
theorem all_equations_are_equalities : ∀ (f : ℝ → ℝ), ∃ (x : ℝ), f x = 0 → (∃ (y : ℝ), f x = f y) :=
sorry

-- -2 is a solution to 3 - 2x = 7
theorem negative_two_solves_equation : IsSolution (-2 : ℝ) (λ x => 3 - 2*x - 7) :=
sorry

-- 1 and -2 are solutions to (x - 1)(x + 2) = 0
theorem one_and_negative_two_solve_equation : 
  IsSolution (1 : ℝ) (λ x => (x - 1)*(x + 2)) ∧ 
  IsSolution (-2 : ℝ) (λ x => (x - 1)*(x + 2)) :=
sorry

end all_equations_are_equalities_negative_two_solves_equation_one_and_negative_two_solve_equation_l3175_317542


namespace aaron_age_proof_l3175_317539

def has_all_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → ∃ k : ℕ, n / 10^k % 10 = d

theorem aaron_age_proof :
  ∃! m : ℕ,
    1000 ≤ m^3 ∧ m^3 < 10000 ∧
    100000 ≤ m^4 ∧ m^4 < 1000000 ∧
    has_all_digits (m^3 + m^4) ∧
    m = 18 := by
  sorry

end aaron_age_proof_l3175_317539


namespace model_height_is_58_l3175_317524

/-- The scale ratio used for the model -/
def scale_ratio : ℚ := 1 / 25

/-- The actual height of the Empire State Building in feet -/
def actual_height : ℕ := 1454

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

/-- The height of the scale model rounded to the nearest foot -/
def model_height : ℕ := (round_to_nearest ((actual_height : ℚ) / scale_ratio)).natAbs

theorem model_height_is_58 : model_height = 58 := by sorry

end model_height_is_58_l3175_317524


namespace intersection_of_A_and_B_l3175_317569

-- Define set A
def A : Set ℝ := {x | x^2 ≤ 4*x}

-- Define set B
def B : Set ℝ := {x | |x| ≥ 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 4} := by sorry

end intersection_of_A_and_B_l3175_317569


namespace fixed_point_of_power_function_l3175_317582

/-- For any real α, the function f(x) = (x-1)^α passes through the point (2,1) -/
theorem fixed_point_of_power_function (α : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ (x - 1) ^ α
  f 2 = 1 := by sorry

end fixed_point_of_power_function_l3175_317582


namespace expedition_max_distance_l3175_317512

/-- Represents the state of the expedition --/
structure ExpeditionState where
  participants : Nat
  distance : Nat
  fuel_per_car : Nat

/-- Calculates the maximum distance the expedition can travel --/
def max_distance (initial_state : ExpeditionState) : Nat :=
  sorry

/-- Theorem stating the maximum distance the expedition can travel --/
theorem expedition_max_distance :
  let initial_state : ExpeditionState := {
    participants := 9,
    distance := 0,
    fuel_per_car := 10  -- 1 gallon in tank + 9 additional cans
  }
  max_distance initial_state = 360 := by
  sorry

end expedition_max_distance_l3175_317512


namespace min_value_theorem_l3175_317590

/-- Given positive real numbers m and n, vectors a and b, and a parallel to b,
    prove that the minimum value of 1/m + 2/n is 3 + 2√2 -/
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (a b : Fin 2 → ℝ)
  (ha : a = λ i => if i = 0 then m else 1)
  (hb : b = λ i => if i = 0 then 1 - n else 1)
  (parallel : ∃ (k : ℝ), a = λ i => k * (b i)) :
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 2/y ≥ 3 + 2 * Real.sqrt 2) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 2/y = 3 + 2 * Real.sqrt 2) := by
  sorry

end min_value_theorem_l3175_317590


namespace integer_roots_of_polynomial_l3175_317562

def polynomial (x : ℤ) : ℤ := x^3 + 2*x^2 - 3*x - 17

def is_root (x : ℤ) : Prop := polynomial x = 0

theorem integer_roots_of_polynomial :
  {x : ℤ | is_root x} = {-17, -1, 1, 17} := by sorry

end integer_roots_of_polynomial_l3175_317562


namespace train_crossing_time_l3175_317565

/-- Proves that a train 400 meters long, traveling at 144 km/hr, will take 10 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 400 ∧ train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 10 := by
  sorry

#check train_crossing_time

end train_crossing_time_l3175_317565


namespace binomial_variance_example_l3175_317598

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: The variance of X ~ B(10, 0.4) is 2.4 -/
theorem binomial_variance_example :
  let X : BinomialDistribution := ⟨10, 0.4, by norm_num⟩
  variance X = 2.4 := by sorry

end binomial_variance_example_l3175_317598


namespace sphere_area_ratio_l3175_317551

/-- The area of a region on a sphere is proportional to the square of its radius -/
axiom area_proportional_to_radius_squared {r₁ r₂ A₁ A₂ : ℝ} (h : r₁ > 0 ∧ r₂ > 0) :
  A₂ / A₁ = (r₂ / r₁) ^ 2

/-- Given two concentric spheres with radii 4 cm and 6 cm, if a region on the smaller sphere
    has an area of 37 square cm, then the corresponding region on the larger sphere
    has an area of 83.25 square cm -/
theorem sphere_area_ratio (r₁ r₂ A₁ : ℝ) (hr₁ : r₁ = 4) (hr₂ : r₂ = 6) (hA₁ : A₁ = 37) :
  ∃ A₂ : ℝ, A₂ = 83.25 ∧ A₂ / A₁ = (r₂ / r₁) ^ 2 := by
  sorry

end sphere_area_ratio_l3175_317551


namespace age_difference_l3175_317525

theorem age_difference (A B : ℕ) : B = 39 → A + 10 = 2 * (B - 10) → A - B = 9 := by
  sorry

end age_difference_l3175_317525


namespace distance_between_points_l3175_317537

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (10, 8)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 13 := by
  sorry

end distance_between_points_l3175_317537


namespace graph_not_simple_l3175_317554

-- Define the equation
def equation (x y : ℝ) : Prop := (x + y)^2 = x^2 + y^2 + 1

-- Define the set of points satisfying the equation
def graph : Set (ℝ × ℝ) := {p | equation p.1 p.2}

-- Theorem stating that the graph is not any of the given options
theorem graph_not_simple : 
  (graph ≠ ∅) ∧ 
  (∃ p q : ℝ × ℝ, p ∈ graph ∧ q ∈ graph ∧ p ≠ q) ∧ 
  (¬∃ a b : ℝ, graph = {p | p.2 = a * p.1 + b} ∪ {p | p.2 = a * p.1 + (b + 1)}) ∧
  (¬∃ c r : ℝ, graph = {p | (p.1 - c)^2 + (p.2 - c)^2 = r^2}) ∧
  (graph ≠ Set.univ) :=
sorry

end graph_not_simple_l3175_317554


namespace election_result_theorem_l3175_317519

/-- Represents the result of a mayoral election. -/
structure ElectionResult where
  total_votes : ℕ
  candidates : ℕ
  winner_votes : ℕ
  second_place_votes : ℕ
  third_place_votes : ℕ
  fourth_place_votes : ℕ
  winner_third_diff : ℕ
  winner_fourth_diff : ℕ

/-- Theorem stating the conditions and the result to be proved. -/
theorem election_result_theorem (e : ElectionResult) 
  (h1 : e.total_votes = 979)
  (h2 : e.candidates = 4)
  (h3 : e.winner_votes = e.fourth_place_votes + e.winner_fourth_diff)
  (h4 : e.winner_votes = e.third_place_votes + e.winner_third_diff)
  (h5 : e.fourth_place_votes = 199)
  (h6 : e.winner_fourth_diff = 105)
  (h7 : e.winner_third_diff = 79)
  (h8 : e.total_votes = e.winner_votes + e.second_place_votes + e.third_place_votes + e.fourth_place_votes) :
  e.winner_votes - e.second_place_votes = 53 := by
  sorry


end election_result_theorem_l3175_317519


namespace initial_bees_count_l3175_317535

/-- Given a hive where 8 bees fly in and the total becomes 24, prove that there were initially 16 bees. -/
theorem initial_bees_count (initial_bees : ℕ) : initial_bees + 8 = 24 → initial_bees = 16 := by
  sorry

end initial_bees_count_l3175_317535


namespace volume_ratio_is_one_over_three_root_three_l3175_317536

/-- A right circular cone -/
structure RightCircularCone where
  radius : ℝ
  height : ℝ

/-- A plane cutting the cone -/
structure CuttingPlane where
  tangent_to_base : Bool
  passes_through_midpoint : Bool

/-- The ratio of volumes -/
def volume_ratio (cone : RightCircularCone) (plane : CuttingPlane) : ℝ := 
  sorry

/-- Theorem statement -/
theorem volume_ratio_is_one_over_three_root_three 
  (cone : RightCircularCone) 
  (plane : CuttingPlane) 
  (h1 : plane.tangent_to_base = true) 
  (h2 : plane.passes_through_midpoint = true) : 
  volume_ratio cone plane = 1 / (3 * Real.sqrt 3) := by
  sorry

end volume_ratio_is_one_over_three_root_three_l3175_317536


namespace complex_equation_solution_l3175_317576

theorem complex_equation_solution (z : ℂ) : (z + Complex.I) * (2 + Complex.I) = 5 → z = 2 - 2*Complex.I := by
  sorry

end complex_equation_solution_l3175_317576


namespace complex_fraction_simplification_l3175_317518

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the problem
theorem complex_fraction_simplification :
  (2 + 4 * i) / (1 - 5 * i) = -9/13 + (7/13) * i :=
by
  -- Proof goes here
  sorry

end complex_fraction_simplification_l3175_317518


namespace integer_difference_l3175_317503

theorem integer_difference (x y : ℕ+) : 
  x > y → x + y = 5 → x^3 - y^3 = 63 → x - y = 3 := by
  sorry

end integer_difference_l3175_317503


namespace sum_lent_l3175_317583

/-- Given a sum of money divided into two parts where:
    1) The interest on the first part for 8 years at 3% per annum
       equals the interest on the second part for 3 years at 5% per annum
    2) The second part is Rs. 1680
    Prove that the total sum lent is Rs. 2730 -/
theorem sum_lent (first_part second_part : ℝ) : 
  second_part = 1680 →
  (first_part * 8 * 3) / 100 = (second_part * 3 * 5) / 100 →
  first_part + second_part = 2730 := by
  sorry

#check sum_lent

end sum_lent_l3175_317583


namespace tan_30_degrees_l3175_317532

theorem tan_30_degrees :
  let sin_30 := (1 : ℝ) / 2
  let cos_30 := Real.sqrt 3 / 2
  let tan_30 := sin_30 / cos_30
  tan_30 = Real.sqrt 3 / 3 := by sorry

end tan_30_degrees_l3175_317532


namespace complex_modulus_problem_l3175_317560

theorem complex_modulus_problem (z₁ z₂ : ℂ) : 
  (z₁ - 2) * Complex.I = 1 + Complex.I →
  z₂.im = 2 →
  ∃ (r : ℝ), z₁ * z₂ = r →
  Complex.abs z₂ = 2 * Real.sqrt 10 := by
sorry

end complex_modulus_problem_l3175_317560


namespace continuous_piecewise_function_sum_l3175_317592

-- Define the piecewise function g
noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then c * x + 4
  else if x ≥ -3 then x - 6
  else 3 * x - d

-- Theorem statement
theorem continuous_piecewise_function_sum (c d : ℝ) :
  (∀ x, ContinuousAt (g c d) x) → c + d = -7/3 := by sorry

end continuous_piecewise_function_sum_l3175_317592


namespace expression_value_l3175_317579

theorem expression_value (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  ∃ ε > 0, |x + (2 * x^3 / y^2) + (2 * y^3 / x^2) + y - 338| < ε :=
sorry

end expression_value_l3175_317579


namespace midpoint_coordinate_sum_l3175_317585

theorem midpoint_coordinate_sum (a b c d e f : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : d + e + f = 9) : 
  (a + b) / 2 + (b + c) / 2 + (c + a) / 2 = 15 ∧ 
  (d + e) / 2 + (e + f) / 2 + (f + d) / 2 = 9 := by
  sorry

end midpoint_coordinate_sum_l3175_317585


namespace problem_statement_l3175_317581

-- Define the basic geometric shapes
def Quadrilateral : Type := Unit
def Square : Type := Unit
def Trapezoid : Type := Unit
def Parallelogram : Type := Unit

-- Define the properties
def has_equal_sides (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry
def is_trapezoid (q : Quadrilateral) : Prop := sorry
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Define the propositions
def proposition_1 : Prop :=
  ∀ q : Quadrilateral, ¬(has_equal_sides q → is_square q)

def proposition_2 : Prop :=
  ∀ q : Quadrilateral, is_parallelogram q → ¬is_trapezoid q

def proposition_3 (a b c : ℝ) : Prop :=
  a > b → a * c^2 > b * c^2

theorem problem_statement :
  proposition_1 ∧
  proposition_2 ∧
  ¬(∀ a b c : ℝ, proposition_3 a b c) :=
sorry

end problem_statement_l3175_317581
