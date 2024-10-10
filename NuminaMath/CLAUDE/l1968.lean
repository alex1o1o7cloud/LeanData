import Mathlib

namespace bears_on_shelves_l1968_196888

/-- Given an initial stock of bears, a new shipment, and a number of bears per shelf,
    calculate the number of shelves required to store all bears. -/
def shelves_required (initial_stock : ℕ) (new_shipment : ℕ) (bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

/-- Theorem stating that with 17 initial bears, 10 new bears, and 9 bears per shelf,
    3 shelves are required. -/
theorem bears_on_shelves :
  shelves_required 17 10 9 = 3 := by
  sorry

end bears_on_shelves_l1968_196888


namespace sin_cos_pi_12_equals_neg_sqrt_2_l1968_196836

theorem sin_cos_pi_12_equals_neg_sqrt_2 :
  Real.sin (π / 12) - Real.sqrt 3 * Real.cos (π / 12) = -Real.sqrt 2 := by
  sorry

end sin_cos_pi_12_equals_neg_sqrt_2_l1968_196836


namespace cuboid_area_and_volume_l1968_196848

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℝ
  breadth : ℝ
  height : ℝ

/-- Calculate the surface area of a cuboid -/
def surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.breadth + c.length * c.height + c.breadth * c.height)

/-- Calculate the volume of a cuboid -/
def volume (c : Cuboid) : ℝ :=
  c.length * c.breadth * c.height

/-- Theorem stating the surface area and volume of a specific cuboid -/
theorem cuboid_area_and_volume :
  let c : Cuboid := ⟨10, 8, 6⟩
  surfaceArea c = 376 ∧ volume c = 480 := by
  sorry

#check cuboid_area_and_volume

end cuboid_area_and_volume_l1968_196848


namespace some_value_proof_l1968_196878

theorem some_value_proof (a : ℝ) : 
  (∀ x : ℝ, |x - a| = 100 → (a + 100) + (a - 100) = 24) → 
  a = 12 := by
  sorry

end some_value_proof_l1968_196878


namespace monkey_climb_time_l1968_196866

/-- A monkey climbing a tree with specific conditions -/
def monkey_climb (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) : ℕ :=
  let effective_climb := hop_distance - slip_distance
  let full_climbs := (tree_height - 1) / effective_climb
  let remaining_distance := (tree_height - 1) % effective_climb
  full_climbs + if remaining_distance > 0 then 1 else 0

/-- Theorem stating the time taken by the monkey to climb the tree -/
theorem monkey_climb_time :
  monkey_climb 17 3 2 = 17 := by sorry

end monkey_climb_time_l1968_196866


namespace lauras_blocks_l1968_196803

/-- Calculates the total number of blocks given the number of friends and blocks per friend -/
def total_blocks (num_friends : ℕ) (blocks_per_friend : ℕ) : ℕ :=
  num_friends * blocks_per_friend

/-- Proves that given 4 friends and 7 blocks per friend, the total number of blocks is 28 -/
theorem lauras_blocks : total_blocks 4 7 = 28 := by
  sorry

end lauras_blocks_l1968_196803


namespace scientific_notation_of_14900_l1968_196816

theorem scientific_notation_of_14900 : 
  14900 = 1.49 * (10 : ℝ)^4 := by sorry

end scientific_notation_of_14900_l1968_196816


namespace calculation_proof_l1968_196835

theorem calculation_proof : 
  let tan30 := Real.sqrt 3 / 3
  let π := 3.14
  (1/3)⁻¹ - Real.sqrt 27 + 3 * tan30 + (π - 3.14)^0 = 4 - 2 * Real.sqrt 3 := by
  sorry

end calculation_proof_l1968_196835


namespace catch_turtle_certain_l1968_196883

-- Define the type for idioms
inductive Idiom
| CatchTurtle
| CarveBoat
| WaitRabbit
| FishMoon

-- Define a function to determine if an idiom represents a certain event
def isCertainEvent (i : Idiom) : Prop :=
  match i with
  | Idiom.CatchTurtle => True
  | _ => False

-- Theorem statement
theorem catch_turtle_certain :
  ∀ i : Idiom, isCertainEvent i ↔ i = Idiom.CatchTurtle :=
by
  sorry


end catch_turtle_certain_l1968_196883


namespace roma_winning_strategy_l1968_196825

/-- The game state representing the positions of chips on a board -/
structure GameState where
  k : ℕ  -- number of cells
  n : ℕ  -- number of chips
  positions : List ℕ  -- positions of chips

/-- The rating of a chip at a given position -/
def chipRating (pos : ℕ) : ℕ := 2^pos

/-- The total rating of all chips in the game state -/
def totalRating (state : GameState) : ℕ :=
  state.positions.map chipRating |>.sum

/-- Roma's strategy to maintain or reduce the total rating -/
def romaStrategy (state : GameState) : GameState :=
  sorry

theorem roma_winning_strategy (k n : ℕ) (h : n < 2^(k-3)) :
  ∀ (state : GameState), state.k = k → state.n = n →
    ∀ (finalState : GameState), finalState = (romaStrategy state) →
      ∀ (pos : ℕ), pos ∈ finalState.positions → pos < k - 1 := by
  sorry

end roma_winning_strategy_l1968_196825


namespace max_segment_length_through_centroid_l1968_196864

/-- Given a triangle ABC with vertex A at (0,0), B at (b, 0), and C at (c_x, c_y),
    the maximum length of a line segment starting from A and passing through the centroid
    is equal to the distance between A and the centroid. -/
theorem max_segment_length_through_centroid (b c_x c_y : ℝ) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (b, 0)
  let C : ℝ × ℝ := (c_x, c_y)
  let centroid : ℝ × ℝ := ((b + c_x) / 3, c_y / 3)
  let max_length := Real.sqrt (((b + c_x) / 3)^2 + (c_y / 3)^2)
  ∃ (segment : ℝ × ℝ → ℝ × ℝ),
    (segment 0 = A) ∧
    (∃ t, segment t = centroid) ∧
    (∀ t, ‖segment t - A‖ ≤ max_length) ∧
    (∃ t, ‖segment t - A‖ = max_length) :=
by sorry


end max_segment_length_through_centroid_l1968_196864


namespace james_money_theorem_l1968_196855

/-- The amount of money James has now, given the conditions -/
def jamesTotal (billsFound : ℕ) (billValue : ℕ) (initialWallet : ℕ) : ℕ :=
  billsFound * billValue + initialWallet

/-- Theorem stating that James has $135 given the problem conditions -/
theorem james_money_theorem :
  jamesTotal 3 20 75 = 135 := by
  sorry

end james_money_theorem_l1968_196855


namespace polynomial_sum_of_coefficients_l1968_196880

theorem polynomial_sum_of_coefficients 
  (f : ℂ → ℂ) 
  (a b c d : ℝ) :
  (∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d) →
  f (2*I) = 0 →
  f (2 + I) = 0 →
  a + b + c + d = 9 := by
sorry

end polynomial_sum_of_coefficients_l1968_196880


namespace cos_450_degrees_eq_zero_l1968_196812

theorem cos_450_degrees_eq_zero : Real.cos (450 * π / 180) = 0 := by
  sorry

end cos_450_degrees_eq_zero_l1968_196812


namespace cube_vertex_distances_l1968_196869

/-- Given a cube with edge length a, after transformation by x₁₄ and x₄₅, 
    the sum of the squares of the distances between vertices 1 and 2, 1 and 4, and 1 and 5 
    is equal to 2a². -/
theorem cube_vertex_distances (a : ℝ) (x₁₄ x₄₅ : ℝ → ℝ → ℝ → ℝ × ℝ × ℝ) 
  (h : a > 0) : 
  ∃ (v₁ v₂ v₄ v₅ : ℝ × ℝ × ℝ), 
    let d₁₂ := ‖v₁ - v₂‖
    let d₁₄ := ‖v₁ - v₄‖
    let d₁₅ := ‖v₁ - v₅‖
    d₁₂^2 + d₁₄^2 + d₁₅^2 = 2 * a^2 :=
by
  sorry

end cube_vertex_distances_l1968_196869


namespace sasha_sticker_problem_l1968_196818

theorem sasha_sticker_problem (m n : ℕ) (t : ℝ) : 
  0 < m ∧ m < n ∧ 1 < t ∧ 
  m * t + n = 100 ∧ 
  m + n * t = 101 → 
  n = 34 ∨ n = 66 := by
sorry

end sasha_sticker_problem_l1968_196818


namespace power_of_power_l1968_196806

theorem power_of_power (a : ℝ) : (a^4)^4 = a^16 := by
  sorry

end power_of_power_l1968_196806


namespace consecutive_integers_problem_l1968_196896

theorem consecutive_integers_problem (n : ℕ) (x : ℤ) : 
  n > 0 → 
  x + n - 1 = 23 → 
  (n : ℝ) * 20 = (n / 2 : ℝ) * (2 * x + n - 1) → 
  n = 7 := by
  sorry

end consecutive_integers_problem_l1968_196896


namespace percentage_relation_l1968_196828

/-- Given that b as a percentage of x is equal to x as a percentage of (a + b),
    and this percentage is 61.80339887498949%, prove that a = b * (38.1966/61.8034) -/
theorem percentage_relation (a b x : ℝ) 
  (h1 : b / x = x / (a + b)) 
  (h2 : b / x = 61.80339887498949 / 100) : 
  a = b * (38.1966 / 61.8034) := by
sorry

end percentage_relation_l1968_196828


namespace quadratic_point_relation_l1968_196808

/-- The quadratic function f(x) = x^2 + x + 1 -/
def f (x : ℝ) : ℝ := x^2 + x + 1

theorem quadratic_point_relation :
  let y₁ := f (-3)
  let y₂ := f 2
  let y₃ := f (1/2)
  y₃ < y₁ ∧ y₁ = y₂ :=
by sorry

end quadratic_point_relation_l1968_196808


namespace buddy_system_l1968_196854

theorem buddy_system (s n : ℕ) (h1 : n ≠ 0) (h2 : s ≠ 0) : 
  (n / 4 : ℚ) = (s / 2 : ℚ) → 
  ((n / 4 + s / 2) / (n + s) : ℚ) = (1 / 3 : ℚ) := by
  sorry

end buddy_system_l1968_196854


namespace vampire_survival_l1968_196832

/-- The number of pints in a gallon -/
def pints_per_gallon : ℕ := 8

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The amount of blood (in gallons) a vampire needs per week -/
def blood_needed_per_week : ℕ := 7

/-- The amount of blood (in pints) a vampire sucks from each person -/
def blood_per_person : ℕ := 2

/-- The number of people a vampire needs to suck from each day to survive -/
def people_per_day : ℕ := 4

theorem vampire_survival :
  (blood_needed_per_week * pints_per_gallon) / days_per_week / blood_per_person = people_per_day :=
sorry

end vampire_survival_l1968_196832


namespace fraction_of_puppies_sold_l1968_196850

/-- Proves that the fraction of puppies sold is 3/8 given the problem conditions --/
theorem fraction_of_puppies_sold (total_puppies : ℕ) (price_per_puppy : ℕ) (total_received : ℕ) :
  total_puppies = 20 →
  price_per_puppy = 200 →
  total_received = 3000 →
  (total_received / price_per_puppy : ℚ) / total_puppies = 3 / 8 := by
  sorry

#check fraction_of_puppies_sold

end fraction_of_puppies_sold_l1968_196850


namespace circle_center_l1968_196823

/-- Given a circle with equation x^2 + y^2 - 2x + 4y - 4 = 0, 
    its center coordinates are (1, -2) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 - 2*x + 4*y - 4 = 0) → 
  ∃ (h : ℝ), (x - 1)^2 + (y + 2)^2 = h^2 :=
by sorry

end circle_center_l1968_196823


namespace inequality_solution_l1968_196877

theorem inequality_solution (x : ℝ) :
  x ≠ 1 ∧ x ≠ 2 →
  (x^3 - x^2 - 6*x) / (x^2 - 3*x + 2) > 0 ↔ (-2 < x ∧ x < 0) ∨ (1 < x ∧ x < 2) ∨ (3 < x) :=
by sorry

end inequality_solution_l1968_196877


namespace divide_seven_students_three_groups_l1968_196860

/-- The number of ways to divide students into groups and send them to different places -/
def divideAndSend (n : ℕ) (k : ℕ) (ratio : List ℕ) : ℕ :=
  sorry

/-- The theorem stating the correct number of ways for the given problem -/
theorem divide_seven_students_three_groups : divideAndSend 7 3 [3, 2, 2] = 630 := by
  sorry

end divide_seven_students_three_groups_l1968_196860


namespace at_least_one_equals_a_l1968_196833

theorem at_least_one_equals_a (x y z a : ℝ) 
  (sum_eq : x + y + z = a) 
  (inv_sum_eq : 1/x + 1/y + 1/z = 1/a) : 
  x = a ∨ y = a ∨ z = a := by
sorry

end at_least_one_equals_a_l1968_196833


namespace player_A_wins_l1968_196872

/-- Represents a player in the game -/
inductive Player : Type
| A : Player
| B : Player

/-- Represents a row of squares on the game board -/
structure Row :=
  (length : ℕ)

/-- Represents the state of the game -/
structure GameState :=
  (tokens : ℕ)
  (row_R : Row)
  (row_S : Row)

/-- Determines if a player has a winning strategy -/
def has_winning_strategy (player : Player) (state : GameState) : Prop :=
  match player with
  | Player.A => state.tokens > 10
  | Player.B => state.tokens ≤ 10

/-- The main theorem stating that Player A has a winning strategy when tokens > 10 -/
theorem player_A_wins (state : GameState) (h1 : state.row_R.length = 1492) (h2 : state.row_S.length = 1989) :
  has_winning_strategy Player.A state ↔ state.tokens > 10 :=
sorry

end player_A_wins_l1968_196872


namespace coin_count_l1968_196807

theorem coin_count (total_amount : ℕ) (five_dollar_count : ℕ) : 
  total_amount = 125 →
  five_dollar_count = 15 →
  ∃ (two_dollar_count : ℕ), 
    two_dollar_count * 2 + five_dollar_count * 5 = total_amount ∧
    two_dollar_count + five_dollar_count = 40 :=
by sorry

end coin_count_l1968_196807


namespace polynomial_coefficient_e_l1968_196834

/-- Polynomial Q(x) = 3x^3 + dx^2 + ex + f -/
def Q (d e f : ℝ) (x : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem polynomial_coefficient_e (d e f : ℝ) :
  (Q d e f 0 = 9) →
  (3 + d + e + f = -(f / 3)) →
  (e = -15 - 3 * Real.sqrt 3) := by
  sorry

end polynomial_coefficient_e_l1968_196834


namespace arc_length_sixty_degrees_l1968_196876

theorem arc_length_sixty_degrees (r : ℝ) (θ : ℝ) (l : ℝ) : 
  r = 1 → θ = 60 → l = (θ * π * r) / 180 → l = π / 3 := by
  sorry

end arc_length_sixty_degrees_l1968_196876


namespace line_not_in_first_quadrant_l1968_196821

/-- Given that mx + 3 = 4 has the solution x = 1, prove that y = (m-2)x - 3 does not pass through the first quadrant -/
theorem line_not_in_first_quadrant (m : ℝ) (h : m * 1 + 3 = 4) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y ≠ (m - 2) * x - 3 := by
  sorry

end line_not_in_first_quadrant_l1968_196821


namespace trenton_earning_goal_l1968_196809

/-- Calculates the earning goal for a salesperson given their fixed weekly earnings,
    commission rate, and sales amount. -/
def earning_goal (fixed_earnings : ℝ) (commission_rate : ℝ) (sales : ℝ) : ℝ :=
  fixed_earnings + commission_rate * sales

/-- Proves that Trenton's earning goal for the week is $500 given the specified conditions. -/
theorem trenton_earning_goal :
  let fixed_earnings : ℝ := 190
  let commission_rate : ℝ := 0.04
  let sales : ℝ := 7750
  earning_goal fixed_earnings commission_rate sales = 500 := by
  sorry


end trenton_earning_goal_l1968_196809


namespace volleyball_team_math_count_l1968_196899

theorem volleyball_team_math_count (total_players : ℕ) (physics_players : ℕ) (both_players : ℕ) 
  (h1 : total_players = 25)
  (h2 : physics_players = 10)
  (h3 : both_players = 6)
  (h4 : physics_players ≥ both_players)
  (h5 : ∀ player, player ∈ Set.range (Fin.val : Fin total_players → ℕ) → 
    (player ∈ Set.range (Fin.val : Fin physics_players → ℕ) ∨ 
     player ∈ Set.range (Fin.val : Fin (total_players - physics_players + both_players) → ℕ))) :
  total_players - physics_players + both_players = 21 := by
  sorry

end volleyball_team_math_count_l1968_196899


namespace paper_cutting_theorem_l1968_196867

/-- Represents the number of pieces after a series of cutting operations -/
def num_pieces (n : ℕ) : ℕ := 4 * n + 1

/-- The result we want to prove is valid -/
def target_result : ℕ := 1993

theorem paper_cutting_theorem :
  ∃ (n : ℕ), num_pieces n = target_result ∧
  ∀ (m : ℕ), ∃ (k : ℕ), num_pieces k = m → m = target_result ∨ m ≠ target_result :=
by sorry

end paper_cutting_theorem_l1968_196867


namespace quadratic_inequality_solution_set_l1968_196844

theorem quadratic_inequality_solution_set (a : ℝ) (h : a > 1) :
  let solution_set := {x : ℝ | (a - 1) * x^2 - a * x + 1 > 0}
  (a = 2 → solution_set = {x : ℝ | x ≠ 1}) ∧
  (1 < a ∧ a < 2 → solution_set = {x : ℝ | x < 1 ∨ x > 1 / (a - 1)}) ∧
  (a > 2 → solution_set = {x : ℝ | x < 1 / (a - 1) ∨ x > 1}) :=
by sorry

end quadratic_inequality_solution_set_l1968_196844


namespace whitney_total_spent_l1968_196839

def whale_books : ℕ := 9
def fish_books : ℕ := 7
def magazines : ℕ := 3
def book_cost : ℕ := 11
def magazine_cost : ℕ := 1

theorem whitney_total_spent : 
  (whale_books + fish_books) * book_cost + magazines * magazine_cost = 179 := by
  sorry

end whitney_total_spent_l1968_196839


namespace collinear_points_problem_l1968_196810

/-- Given three collinear points A, B, C in a plane with position vectors
    OA = (-2, m), OB = (n, 1), OC = (5, -1), and OA perpendicular to OB,
    prove that m = 6 and n = 3. -/
theorem collinear_points_problem (m n : ℝ) : 
  let OA : ℝ × ℝ := (-2, m)
  let OB : ℝ × ℝ := (n, 1)
  let OC : ℝ × ℝ := (5, -1)
  let AC : ℝ × ℝ := (OC.1 - OA.1, OC.2 - OA.2)
  let BC : ℝ × ℝ := (OC.1 - OB.1, OC.2 - OB.2)
  (∃ (k : ℝ), AC = k • BC) →  -- collinearity condition
  (OA.1 * OB.1 + OA.2 * OB.2 = 0) →  -- perpendicularity condition
  m = 6 ∧ n = 3 := by
  sorry

end collinear_points_problem_l1968_196810


namespace seats_per_row_l1968_196804

/-- Proves that given the specified conditions, the number of seats in each row is 8 -/
theorem seats_per_row (rows : ℕ) (base_cost : ℚ) (discount_rate : ℚ) (discount_group : ℕ) (total_cost : ℚ) :
  rows = 5 →
  base_cost = 30 →
  discount_rate = 1/10 →
  discount_group = 10 →
  total_cost = 1080 →
  ∃ (seats_per_row : ℕ),
    seats_per_row = 8 ∧
    total_cost = rows * (seats_per_row * base_cost - (seats_per_row / discount_group) * (discount_rate * base_cost * discount_group)) :=
by sorry

end seats_per_row_l1968_196804


namespace integral_reciprocal_plus_x_l1968_196852

theorem integral_reciprocal_plus_x : ∫ x in (1:ℝ)..(2:ℝ), (1/x + x) = Real.log 2 + 3/2 := by
  sorry

end integral_reciprocal_plus_x_l1968_196852


namespace composition_zero_iff_rank_sum_eq_dim_l1968_196894

variable {V : Type*} [AddCommGroup V] [Module ℝ V] [FiniteDimensional ℝ V]
variable (T U : V →ₗ[ℝ] V)

theorem composition_zero_iff_rank_sum_eq_dim (h : Function.Bijective (T + U)) :
  (T.comp U = 0 ∧ U.comp T = 0) ↔ LinearMap.rank T + LinearMap.rank U = FiniteDimensional.finrank ℝ V :=
sorry

end composition_zero_iff_rank_sum_eq_dim_l1968_196894


namespace expression_bounds_l1968_196822

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) : 
  4 + 2 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
                        Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ∧
  Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
  Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ≤ 8 := by
  sorry

end expression_bounds_l1968_196822


namespace equal_roots_quadratic_l1968_196885

/-- The quadratic equation (2kx^2 + 7kx + 2) = 0 has equal roots when k = 16/49 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + 7 * k * x + 2 = 0) → 
  (∃! r : ℝ, 2 * k * r^2 + 7 * k * r + 2 = 0) → 
  k = 16/49 := by
sorry

end equal_roots_quadratic_l1968_196885


namespace right_triangle_cosine_l1968_196851

theorem right_triangle_cosine (X Y Z : ℝ) (h1 : X = 90) (h2 : Real.sin Z = 4/5) : 
  Real.cos Z = 3/5 := by
  sorry

end right_triangle_cosine_l1968_196851


namespace supplementary_angle_of_10_degrees_l1968_196871

def is_supplementary (a b : ℝ) : Prop :=
  (a + b) % 360 = 180

theorem supplementary_angle_of_10_degrees (k : ℤ) :
  is_supplementary 10 (k * 360 + 250) :=
sorry

end supplementary_angle_of_10_degrees_l1968_196871


namespace money_distribution_l1968_196881

/-- The problem of distributing money among five people with specific conditions -/
theorem money_distribution (a b c d e : ℕ) : 
  a + b + c + d + e = 1010 ∧
  (a - 25) / 4 = (b - 10) / 3 ∧
  (a - 25) / 4 = (c - 15) / 6 ∧
  (a - 25) / 4 = (d - 20) / 2 ∧
  (a - 25) / 4 = (e - 30) / 5 →
  c = 288 := by
  sorry

end money_distribution_l1968_196881


namespace cube_inequality_l1968_196887

theorem cube_inequality (x y a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x < a^y) : x^3 > y^3 := by
  sorry

end cube_inequality_l1968_196887


namespace weight_of_b_l1968_196873

theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 43 →
  b = 31 := by
sorry

end weight_of_b_l1968_196873


namespace C₂_is_symmetric_to_C₁_l1968_196817

/-- Circle C₁ with equation (x+1)²+(y-1)²=1 -/
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

/-- The line of symmetry x-y-1=0 -/
def symmetry_line (x y : ℝ) : Prop := x - y - 1 = 0

/-- The symmetric point of (x, y) with respect to the line x-y-1=0 -/
def symmetric_point (x y : ℝ) : ℝ × ℝ := (y + 1, x - 1)

/-- Circle C₂, symmetric to C₁ with respect to the line x-y-1=0 -/
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

/-- Theorem stating that C₂ is indeed symmetric to C₁ with respect to the given line -/
theorem C₂_is_symmetric_to_C₁ :
  ∀ x y : ℝ, C₂ x y ↔ C₁ (symmetric_point x y).1 (symmetric_point x y).2 :=
sorry

end C₂_is_symmetric_to_C₁_l1968_196817


namespace smallest_two_digit_prime_with_composite_reversal_l1968_196831

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def reverse_digits (n : ℕ) : ℕ :=
  let units := n % 10
  let tens := n / 10
  units * 10 + tens

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def tens_digit_is_two (n : ℕ) : Prop := (n / 10) % 10 = 2

theorem smallest_two_digit_prime_with_composite_reversal :
  ∃ n : ℕ,
    is_two_digit n ∧
    tens_digit_is_two n ∧
    is_prime n ∧
    is_composite (reverse_digits n) ∧
    (∀ m : ℕ, is_two_digit m → tens_digit_is_two m → is_prime m → 
      is_composite (reverse_digits m) → n ≤ m) ∧
    n = 23 :=
sorry

end smallest_two_digit_prime_with_composite_reversal_l1968_196831


namespace surface_a_properties_surface_b_properties_surface_c_properties_l1968_196884

-- Part (a)
def surface_a1 (x y z : ℝ) : Prop := 2 * y = x^2 + z^2
def surface_a2 (x y z : ℝ) : Prop := x^2 + z^2 = 1

theorem surface_a_properties (x y z : ℝ) :
  surface_a1 x y z ∧ surface_a2 x y z → y ≥ 0 :=
sorry

-- Part (b)
def surface_b1 (x y z : ℝ) : Prop := z = 0
def surface_b2 (x y z : ℝ) : Prop := y + z = 2
def surface_b3 (x y z : ℝ) : Prop := y = x^2

theorem surface_b_properties (x y z : ℝ) :
  surface_b1 x y z ∧ surface_b2 x y z ∧ surface_b3 x y z → 
  y ≤ 2 ∧ y ≥ 0 ∧ z ≤ 2 ∧ z ≥ 0 :=
sorry

-- Part (c)
def surface_c1 (x y z : ℝ) : Prop := z = 6 - x^2 - y^2
def surface_c2 (x y z : ℝ) : Prop := x^2 + y^2 - z^2 = 0

theorem surface_c_properties (x y z : ℝ) :
  surface_c1 x y z ∧ surface_c2 x y z → 
  z ≤ 3 ∧ z ≥ 0 :=
sorry

end surface_a_properties_surface_b_properties_surface_c_properties_l1968_196884


namespace power_of_power_l1968_196820

theorem power_of_power (x : ℝ) : (x^3)^2 = x^6 := by
  sorry

end power_of_power_l1968_196820


namespace three_digit_number_times_seven_l1968_196847

theorem three_digit_number_times_seven (n : ℕ) : 
  (100 ≤ n ∧ n < 1000) ∧ (∃ k : ℕ, 7 * n = 1000 * k + 638) ↔ n = 234 :=
sorry

end three_digit_number_times_seven_l1968_196847


namespace eight_power_plus_six_divisible_by_seven_l1968_196846

theorem eight_power_plus_six_divisible_by_seven (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℤ, (8 : ℤ)^n + 6 = 7 * k :=
by sorry

end eight_power_plus_six_divisible_by_seven_l1968_196846


namespace second_smallest_packs_l1968_196865

/-- The number of hot dogs in each pack -/
def hot_dogs_per_pack : ℕ := 9

/-- The number of buns in each pack -/
def buns_per_pack : ℕ := 7

/-- The number of hot dogs left over after the barbecue -/
def leftover_hot_dogs : ℕ := 6

/-- 
Theorem: The second smallest number of packs of hot dogs that satisfies 
the conditions of the barbecue problem is 10.
-/
theorem second_smallest_packs : 
  (∃ m : ℕ, m < 10 ∧ hot_dogs_per_pack * m ≡ leftover_hot_dogs [MOD buns_per_pack]) ∧
  (∀ k : ℕ, k < 10 → hot_dogs_per_pack * k ≡ leftover_hot_dogs [MOD buns_per_pack] → 
    ∃ m : ℕ, m < k ∧ hot_dogs_per_pack * m ≡ leftover_hot_dogs [MOD buns_per_pack]) ∧
  hot_dogs_per_pack * 10 ≡ leftover_hot_dogs [MOD buns_per_pack] := by
  sorry


end second_smallest_packs_l1968_196865


namespace equation_solution_expression_simplification_l1968_196889

-- Part 1
theorem equation_solution :
  ∃ x : ℝ, (x / (2*x - 3) + 5 / (3 - 2*x) = 4) ∧ (x = 1) :=
sorry

-- Part 2
theorem expression_simplification (a : ℝ) (h : a ≠ 2 ∧ a ≠ -2) :
  (a - 2 - 4 / (a - 2)) / ((a - 4) / (a^2 - 4)) = a^2 + 2*a :=
sorry

end equation_solution_expression_simplification_l1968_196889


namespace shaded_area_l1968_196838

/-- The area of the shaded region in a square with two non-overlapping rectangles --/
theorem shaded_area (total_area : ℝ) (rect1_area rect2_area : ℝ) :
  total_area = 16 →
  rect1_area = 6 →
  rect2_area = 2 →
  total_area - (rect1_area + rect2_area) = 8 := by
sorry


end shaded_area_l1968_196838


namespace quadratic_equation_roots_l1968_196886

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - 3*a = 0 ∧ x = -2) → 
  (∃ y : ℝ, y^2 - a*y - 3*a = 0 ∧ y = 6) :=
by sorry

end quadratic_equation_roots_l1968_196886


namespace division_problem_l1968_196861

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 686)
  (h2 : divisor = 36)
  (h3 : remainder = 2)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 19 := by
sorry

end division_problem_l1968_196861


namespace change_per_bill_l1968_196819

/-- Proves that the value of each bill given as change is $5 -/
theorem change_per_bill (num_games : ℕ) (cost_per_game : ℕ) (payment : ℕ) (num_bills : ℕ) :
  num_games = 6 →
  cost_per_game = 15 →
  payment = 100 →
  num_bills = 2 →
  (payment - num_games * cost_per_game) / num_bills = 5 := by
  sorry

end change_per_bill_l1968_196819


namespace sum_of_powers_of_two_l1968_196829

theorem sum_of_powers_of_two : 2^4 + 2^4 + 2^4 = 2^5 := by
  sorry

end sum_of_powers_of_two_l1968_196829


namespace twelve_chairs_adjacent_subsets_l1968_196830

/-- The number of chairs arranged in a circle. -/
def n : ℕ := 12

/-- A function that calculates the number of subsets containing at least three adjacent chairs
    for a given number of chairs arranged in a circle. -/
def subsets_with_adjacent_chairs (num_chairs : ℕ) : ℕ := sorry

/-- Theorem stating that for 12 chairs arranged in a circle, 
    the number of subsets containing at least three adjacent chairs is 2066. -/
theorem twelve_chairs_adjacent_subsets : subsets_with_adjacent_chairs n = 2066 := by sorry

end twelve_chairs_adjacent_subsets_l1968_196830


namespace range_of_f_l1968_196824

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 2^x - 5 else 3 * Real.sin x

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Ioc (-5) 3 := by sorry

end range_of_f_l1968_196824


namespace frequency_of_six_is_nineteen_hundredths_l1968_196863

/-- Represents the outcome of rolling a fair six-sided die multiple times -/
structure DieRollOutcome where
  total_rolls : ℕ
  sixes_count : ℕ

/-- Calculates the frequency of rolling a 6 -/
def frequency_of_six (outcome : DieRollOutcome) : ℚ :=
  outcome.sixes_count / outcome.total_rolls

/-- Theorem stating that for the given die roll outcome, the frequency of rolling a 6 is 0.19 -/
theorem frequency_of_six_is_nineteen_hundredths 
  (outcome : DieRollOutcome) 
  (h1 : outcome.total_rolls = 100) 
  (h2 : outcome.sixes_count = 19) : 
  frequency_of_six outcome = 19 / 100 := by
  sorry

end frequency_of_six_is_nineteen_hundredths_l1968_196863


namespace matthew_rebecca_age_difference_l1968_196801

/-- Represents the ages of three children and their properties --/
structure ChildrenAges where
  freddy : ℕ
  matthew : ℕ
  rebecca : ℕ
  total_age : freddy + matthew + rebecca = 35
  freddy_age : freddy = 15
  matthew_younger : matthew = freddy - 4
  matthew_older : matthew > rebecca

/-- Theorem stating that Matthew is 2 years older than Rebecca --/
theorem matthew_rebecca_age_difference (ages : ChildrenAges) : ages.matthew = ages.rebecca + 2 := by
  sorry

end matthew_rebecca_age_difference_l1968_196801


namespace complex_number_location_l1968_196862

theorem complex_number_location : ∃ (z : ℂ), z = (5 - 6*I) + (-2 - I) - (3 + 4*I) ∧ z = -11*I := by
  sorry

end complex_number_location_l1968_196862


namespace quadratic_equation_coefficients_l1968_196868

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x, (2*x - 1)^2 = (x + 1)*(3*x + 4)) →
    (∀ x, a*x^2 + b*x + c = 0) ∧
    a = 1 ∧ b = -11 ∧ c = -3 :=
by sorry

end quadratic_equation_coefficients_l1968_196868


namespace wood_length_after_sawing_l1968_196859

theorem wood_length_after_sawing (original_length sawed_off_length : ℝ) 
  (h1 : original_length = 0.41)
  (h2 : sawed_off_length = 0.33) :
  original_length - sawed_off_length = 0.08 := by
  sorry

end wood_length_after_sawing_l1968_196859


namespace art_math_supplies_cost_l1968_196845

-- Define the prices of items
def folder_price : ℚ := 3.5
def notebook_price : ℚ := 3
def binder_price : ℚ := 5
def pencil_price : ℚ := 1
def eraser_price : ℚ := 0.75
def highlighter_price : ℚ := 3.25
def marker_price : ℚ := 3.5
def sticky_note_price : ℚ := 2.5
def calculator_price : ℚ := 10.5
def sketchbook_price : ℚ := 4.5
def paint_set_price : ℚ := 18
def color_pencil_price : ℚ := 7

-- Define the quantities
def num_classes : ℕ := 12
def folders_per_class : ℕ := 1
def notebooks_per_class : ℕ := 2
def binders_per_class : ℕ := 1
def pencils_per_class : ℕ := 3
def erasers_per_6_pencils : ℕ := 2

-- Define the total spent
def total_spent : ℚ := 210

-- Theorem statement
theorem art_math_supplies_cost : 
  paint_set_price + color_pencil_price + calculator_price + sketchbook_price = 40 := by
  sorry

end art_math_supplies_cost_l1968_196845


namespace trig_fraction_simplification_l1968_196826

theorem trig_fraction_simplification (α : ℝ) : 
  (Real.cos (π + α) * Real.sin (α + 2*π)) / (Real.sin (-α - π) * Real.cos (-π - α)) = 1 := by
  sorry

end trig_fraction_simplification_l1968_196826


namespace bananas_left_l1968_196870

/-- The number of bananas in a dozen -/
def dozen : ℕ := 12

/-- The number of bananas Elizabeth ate -/
def eaten : ℕ := 4

/-- Theorem: If Elizabeth bought a dozen bananas and ate 4 of them, then 8 bananas are left -/
theorem bananas_left (bought : ℕ) (ate : ℕ) (h1 : bought = dozen) (h2 : ate = eaten) :
  bought - ate = 8 := by
  sorry

end bananas_left_l1968_196870


namespace street_lights_on_triangular_playground_l1968_196856

theorem street_lights_on_triangular_playground (side_length : ℝ) (interval : ℝ) :
  side_length = 10 ∧ interval = 3 →
  (3 * side_length) / interval = 10 := by
sorry

end street_lights_on_triangular_playground_l1968_196856


namespace expression_value_l1968_196891

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x = 1 / y) (hzy : z = 1 / y) : (x + 1 / x) * (z - 1 / z) = 4 := by
  sorry

end expression_value_l1968_196891


namespace number_of_persons_l1968_196837

theorem number_of_persons (total_amount : ℕ) (amount_per_person : ℕ) 
  (h1 : total_amount = 42900) 
  (h2 : amount_per_person = 1950) : 
  total_amount / amount_per_person = 22 := by
  sorry

end number_of_persons_l1968_196837


namespace water_drip_relationship_faucet_left_on_time_l1968_196849

/-- Represents the water drip rate in mL per second -/
def drip_rate : ℝ := 2 * 0.05

/-- Represents the relationship between time (in hours) and water volume (in mL) -/
def water_volume (time : ℝ) : ℝ := (3600 * drip_rate) * time

theorem water_drip_relationship (time : ℝ) (volume : ℝ) (h : time ≥ 0) :
  water_volume time = 360 * time :=
sorry

theorem faucet_left_on_time (volume : ℝ) (h : volume = 1620) :
  ∃ (time : ℝ), water_volume time = volume ∧ time = 4.5 :=
sorry

end water_drip_relationship_faucet_left_on_time_l1968_196849


namespace survey_result_l1968_196802

theorem survey_result (total_surveyed : ℕ) 
  (believed_spread_diseases : ℕ) 
  (believed_flu : ℕ) : 
  (believed_spread_diseases : ℝ) / total_surveyed = 0.905 →
  (believed_flu : ℝ) / believed_spread_diseases = 0.503 →
  believed_flu = 26 →
  total_surveyed = 57 := by
sorry

end survey_result_l1968_196802


namespace intersection_k_range_l1968_196840

-- Define the line and hyperbola equations
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 2
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 6

-- Define the condition for intersection points
def intersects_right_branch (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧
  hyperbola x₁ (line k x₁) ∧ hyperbola x₂ (line k x₂)

-- Theorem statement
theorem intersection_k_range :
  ∀ k : ℝ, intersects_right_branch k ↔ -Real.sqrt 15 / 3 < k ∧ k < -1 :=
sorry

end intersection_k_range_l1968_196840


namespace sin_cos_power_relation_l1968_196811

theorem sin_cos_power_relation (x : ℝ) :
  (Real.sin x)^10 + (Real.cos x)^10 = 11/36 →
  (Real.sin x)^12 + (Real.cos x)^12 = 5/18 := by
  sorry

end sin_cos_power_relation_l1968_196811


namespace binary_1101101_equals_decimal_109_l1968_196827

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define the binary number 1101101
def binary_number : List Bool := [true, false, true, true, false, true, true]

-- Theorem statement
theorem binary_1101101_equals_decimal_109 :
  binary_to_decimal binary_number = 109 := by
  sorry

end binary_1101101_equals_decimal_109_l1968_196827


namespace middle_number_problem_l1968_196895

theorem middle_number_problem (x y z : ℕ) 
  (h1 : x < y) (h2 : y < z)
  (h3 : x + y = 20) (h4 : x + z = 25) (h5 : y + z = 29) (h6 : z - x = 11) :
  y = 13 := by
  sorry

end middle_number_problem_l1968_196895


namespace at_most_two_out_of_three_l1968_196892

-- Define the probability of a single event
def p : ℚ := 3 / 5

-- Define the number of events
def n : ℕ := 3

-- Define the maximum number of events we want to occur
def k : ℕ := 2

-- Theorem statement
theorem at_most_two_out_of_three (p : ℚ) (n : ℕ) (k : ℕ) 
  (h1 : p = 3 / 5) 
  (h2 : n = 3) 
  (h3 : k = 2) : 
  1 - p^n = 98 / 125 := by
  sorry

#check at_most_two_out_of_three p n k

end at_most_two_out_of_three_l1968_196892


namespace intersection_of_A_and_B_l1968_196815

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end intersection_of_A_and_B_l1968_196815


namespace stickers_remaining_proof_l1968_196842

/-- Calculates the number of stickers remaining after losing a page. -/
def stickers_remaining (stickers_per_page : ℕ) (initial_pages : ℕ) (pages_lost : ℕ) : ℕ :=
  (initial_pages - pages_lost) * stickers_per_page

/-- Proves that the number of stickers remaining is 220. -/
theorem stickers_remaining_proof :
  stickers_remaining 20 12 1 = 220 := by
  sorry

end stickers_remaining_proof_l1968_196842


namespace trapezoid_ratio_l1968_196875

/-- Represents a trapezoid ABCD with a point P inside -/
structure Trapezoid :=
  (AB CD : ℝ)
  (height : ℝ)
  (area_PCD area_PAD area_PBC area_PAB : ℝ)

/-- The theorem stating the ratio of parallel sides in the trapezoid -/
theorem trapezoid_ratio (T : Trapezoid) : 
  T.AB > T.CD →
  T.height = 8 →
  T.area_PCD = 4 →
  T.area_PAD = 6 →
  T.area_PBC = 5 →
  T.area_PAB = 7 →
  T.AB / T.CD = 4 := by
  sorry


end trapezoid_ratio_l1968_196875


namespace cyclic_sum_factorization_l1968_196890

theorem cyclic_sum_factorization (a b c : ℝ) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = (a - b) * (b - c) * (c - a) := by
  sorry

end cyclic_sum_factorization_l1968_196890


namespace train_length_l1968_196814

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 18 → ∃ (length_m : ℝ), 
    (length_m ≥ 300.05 ∧ length_m ≤ 300.07) ∧ 
    length_m = speed_kmh * (1000 / 3600) * time_s :=
by
  sorry

end train_length_l1968_196814


namespace fish_catch_total_l1968_196898

def fish_problem (bass : ℕ) (trout : ℕ) (blue_gill : ℕ) : Prop :=
  (bass = 32) ∧
  (trout = bass / 4) ∧
  (blue_gill = 2 * bass) ∧
  (bass + trout + blue_gill = 104)

theorem fish_catch_total :
  ∀ (bass trout blue_gill : ℕ), fish_problem bass trout blue_gill :=
by
  sorry

end fish_catch_total_l1968_196898


namespace max_area_triangle_OPQ_l1968_196858

/-- Parabola in Cartesian coordinates -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point on a parabola -/
structure PointOnParabola (c : Parabola) where
  x : ℝ
  y : ℝ
  h : x^2 = 2 * c.p * y

/-- Line intersecting a parabola -/
structure IntersectingLine (c : Parabola) where
  k : ℝ
  b : ℝ
  h : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 = 2 * c.p * (k * x₁ + b) ∧ x₂^2 = 2 * c.p * (k * x₂ + b)

/-- Theorem: Maximum area of triangle OPQ -/
theorem max_area_triangle_OPQ (c : Parabola) (a : PointOnParabola c) (l : IntersectingLine c) :
  a.x^2 + a.y^2 = (3/2)^2 →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧
    x₁^2 = 2 * c.p * y₁ ∧ x₂^2 = 2 * c.p * y₂ ∧
    y₁ = l.k * x₁ + l.b ∧ y₂ = l.k * x₂ + l.b ∧
    (y₁ + y₂) / 2 = 1) →
  (∃ (area : ℝ), area ≤ 2 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ →
      x₁^2 = 2 * c.p * y₁ → x₂^2 = 2 * c.p * y₂ →
      y₁ = l.k * x₁ + l.b → y₂ = l.k * x₂ + l.b →
      (y₁ + y₂) / 2 = 1 →
      area ≥ abs (x₁ * y₂ - x₂ * y₁) / 2) :=
sorry

end max_area_triangle_OPQ_l1968_196858


namespace turkey_cost_per_kg_turkey_cost_is_two_l1968_196874

/-- Given Dabbie's turkey purchase scenario, prove the cost per kilogram of turkey. -/
theorem turkey_cost_per_kg : ℝ → Prop :=
  fun cost_per_kg =>
    let first_turkey_weight := 6
    let second_turkey_weight := 9
    let third_turkey_weight := 2 * second_turkey_weight
    let total_weight := first_turkey_weight + second_turkey_weight + third_turkey_weight
    let total_cost := 66
    cost_per_kg = total_cost / total_weight

/-- The cost per kilogram of turkey is $2. -/
theorem turkey_cost_is_two : turkey_cost_per_kg 2 := by
  sorry

end turkey_cost_per_kg_turkey_cost_is_two_l1968_196874


namespace valid_seq_equals_fib_prob_no_consecutive_ones_l1968_196853

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Sequence of valid arrangements -/
def validSeq : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validSeq (n + 1) + validSeq n

theorem valid_seq_equals_fib (n : ℕ) : validSeq n = fib (n + 2) := by
  sorry

theorem prob_no_consecutive_ones : 
  (validSeq 12 : ℚ) / 2^12 = 377 / 4096 := by
  sorry

#eval fib 14 + 4096  -- Should output 4473


end valid_seq_equals_fib_prob_no_consecutive_ones_l1968_196853


namespace ab_value_l1968_196857

theorem ab_value (a b : ℝ) (h : 48 * (a * b) = (a * b) * 65) : a * b = 0 := by
  sorry

end ab_value_l1968_196857


namespace third_offense_percentage_increase_l1968_196882

/-- Calculates the percentage increase for a third offense in a burglary case -/
theorem third_offense_percentage_increase
  (base_rate : ℚ)  -- Base sentence rate in years per $5000
  (stolen_value : ℚ)  -- Total value of stolen goods in dollars
  (additional_penalty : ℚ)  -- Additional penalty in years
  (total_sentence : ℚ)  -- Total sentence in years
  (h1 : base_rate = 1)  -- Base rate is 1 year per $5000
  (h2 : stolen_value = 40000)  -- $40,000 worth of goods stolen
  (h3 : additional_penalty = 2)  -- 2 years additional penalty
  (h4 : total_sentence = 12)  -- Total sentence is 12 years
  : (total_sentence - additional_penalty - (stolen_value / 5000 * base_rate)) / (stolen_value / 5000 * base_rate) * 100 = 25 := by
  sorry

end third_offense_percentage_increase_l1968_196882


namespace greatest_integer_gcd_4_with_12_l1968_196843

theorem greatest_integer_gcd_4_with_12 : 
  ∃ n : ℕ, n < 100 ∧ Nat.gcd n 12 = 4 ∧ ∀ m : ℕ, m < 100 → Nat.gcd m 12 = 4 → m ≤ n := by
  sorry

end greatest_integer_gcd_4_with_12_l1968_196843


namespace history_class_grade_distribution_l1968_196800

theorem history_class_grade_distribution (total_students : ℕ) 
  (prob_A prob_B prob_C prob_D : ℝ) (B_count : ℕ) : 
  total_students = 52 →
  prob_A = 0.5 * prob_B →
  prob_C = 2 * prob_B →
  prob_D = 0.5 * prob_B →
  prob_A + prob_B + prob_C + prob_D = 1 →
  B_count = 13 →
  (0.5 * B_count : ℝ) + B_count + (2 * B_count) + (0.5 * B_count) = total_students := by
  sorry

end history_class_grade_distribution_l1968_196800


namespace fourth_column_unique_l1968_196813

/-- Represents a 9x9 Sudoku grid -/
def SudokuGrid := Fin 9 → Fin 9 → Fin 9

/-- Checks if a number is valid in a given position -/
def isValid (grid : SudokuGrid) (row col num : Fin 9) : Prop :=
  (∀ i : Fin 9, grid i col ≠ num) ∧
  (∀ j : Fin 9, grid row j ≠ num) ∧
  (∀ i j : Fin 3, grid (3 * (row / 3) + i) (3 * (col / 3) + j) ≠ num)

/-- Checks if the entire grid is valid -/
def isValidGrid (grid : SudokuGrid) : Prop :=
  ∀ row col : Fin 9, isValid grid row col (grid row col)

/-- Represents the pre-filled numbers in the 4th column -/
def fourthColumnPrefilled : Fin 9 → Option (Fin 9)
  | 0 => some 3
  | 1 => some 2
  | 3 => some 4
  | 7 => some 5
  | 8 => some 1
  | _ => none

/-- The theorem to be proved -/
theorem fourth_column_unique (grid : SudokuGrid) :
  isValidGrid grid →
  (∀ row : Fin 9, (fourthColumnPrefilled row).map (grid row 3) = fourthColumnPrefilled row) →
  (∀ row : Fin 9, grid row 3 = match row with
    | 0 => 3 | 1 => 2 | 2 => 7 | 3 => 4 | 4 => 6 | 5 => 8 | 6 => 9 | 7 => 5 | 8 => 1) :=
by sorry

end fourth_column_unique_l1968_196813


namespace right_triangles_on_circle_l1968_196897

theorem right_triangles_on_circle (n : ℕ) (h : n = 100) :
  ¬ (∃ (k : ℕ), k = 1000 ∧ k = (n / 2) * (n - 2)) :=
by
  sorry

end right_triangles_on_circle_l1968_196897


namespace line_through_ellipse_vertex_l1968_196879

/-- The value of 'a' when a line passes through the right vertex of an ellipse --/
theorem line_through_ellipse_vertex (t θ : ℝ) (a : ℝ) : 
  (∀ t, ∃ x y, x = t ∧ y = t - a) →  -- Line equation
  (∀ θ, ∃ x y, x = 3 * Real.cos θ ∧ y = 2 * Real.sin θ) →  -- Ellipse equation
  (∃ t, t = 3 ∧ t - a = 0) →  -- Line passes through right vertex (3, 0)
  a = 3 := by
sorry

end line_through_ellipse_vertex_l1968_196879


namespace sum_product_equals_negative_one_l1968_196805

theorem sum_product_equals_negative_one 
  (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_one : a^2 + b^2 + c^2 = 1) : 
  a*(b+c) + b*(a+c) + c*(a+b) = -1 := by
sorry

end sum_product_equals_negative_one_l1968_196805


namespace system_solution_l1968_196841

theorem system_solution : 
  ∃! (x y : ℝ), x + y = 5 ∧ 2 * x + 5 * y = 28 :=
by
  -- The proof goes here
  sorry

end system_solution_l1968_196841


namespace greatest_three_digit_multiple_of_17_l1968_196893

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by sorry

end greatest_three_digit_multiple_of_17_l1968_196893
