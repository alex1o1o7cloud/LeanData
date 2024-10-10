import Mathlib

namespace spurs_team_size_l3355_335560

/-- The number of basketballs each player has -/
def basketballs_per_player : ℕ := 11

/-- The total number of basketballs -/
def total_basketballs : ℕ := 242

/-- The number of players on the team -/
def number_of_players : ℕ := total_basketballs / basketballs_per_player

theorem spurs_team_size :
  number_of_players = 22 :=
by sorry

end spurs_team_size_l3355_335560


namespace product_five_fourth_sum_l3355_335564

theorem product_five_fourth_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → 
  a * b * c = 5^4 → 
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 131 := by sorry

end product_five_fourth_sum_l3355_335564


namespace complement_of_A_in_U_l3355_335592

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {3, 4, 5}

-- Theorem statement
theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {1, 2} := by
  sorry

end complement_of_A_in_U_l3355_335592


namespace unique_perpendicular_line_parallel_intersections_perpendicular_line_in_plane_l3355_335596

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships
variable (outside : Point → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Line)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (in_plane : Point → Plane → Prop)
variable (in_line : Point → Line → Prop)

-- Theorem 1
theorem unique_perpendicular_line 
  (p : Point) (π : Plane) (h : outside p π) :
  ∃! l : Line, perpendicular l π ∧ in_line p l :=
sorry

-- Theorem 2
theorem parallel_intersections 
  (π₁ π₂ π₃ : Plane) (h : parallel_planes π₁ π₂) :
  parallel (intersect π₁ π₃) (intersect π₂ π₃) :=
sorry

-- Theorem 3
theorem perpendicular_line_in_plane 
  (π₁ π₂ : Plane) (p : Point) (l : Line)
  (h₁ : perpendicular_planes π₁ π₂) (h₂ : in_plane p π₁)
  (h₃ : perpendicular l π₂) (h₄ : in_line p l) :
  ∀ q : Point, in_line q l → in_plane q π₁ :=
sorry

end unique_perpendicular_line_parallel_intersections_perpendicular_line_in_plane_l3355_335596


namespace brownie_division_l3355_335527

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the pan of brownies -/
def pan : Dimensions := ⟨24, 15⟩

/-- Represents a single piece of brownie -/
def piece : Dimensions := ⟨3, 2⟩

/-- Theorem stating that the pan can be divided into exactly 60 pieces -/
theorem brownie_division :
  (area pan) / (area piece) = 60 := by sorry

end brownie_division_l3355_335527


namespace condition_one_implies_right_triangle_condition_two_implies_right_triangle_condition_three_not_implies_right_triangle_condition_four_implies_right_triangle_l3355_335535

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_of_angles : A + B + C = 180

-- Define what it means for a triangle to be right-angled
def is_right_triangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

-- Condition 1
theorem condition_one_implies_right_triangle (t : Triangle) 
  (h : t.A + t.B = t.C) : is_right_triangle t :=
sorry

-- Condition 2
theorem condition_two_implies_right_triangle (t : Triangle) 
  (h : ∃ (k : Real), t.A = k ∧ t.B = 2*k ∧ t.C = 3*k) : is_right_triangle t :=
sorry

-- Condition 3
theorem condition_three_not_implies_right_triangle : ∃ (t : Triangle), 
  (t.A = t.B ∧ t.B = t.C) ∧ ¬(is_right_triangle t) :=
sorry

-- Condition 4
theorem condition_four_implies_right_triangle (t : Triangle) 
  (h : t.A = 90 - t.B) : is_right_triangle t :=
sorry

end condition_one_implies_right_triangle_condition_two_implies_right_triangle_condition_three_not_implies_right_triangle_condition_four_implies_right_triangle_l3355_335535


namespace vertex_coordinates_l3355_335566

/-- The quadratic function f(x) = x^2 - 2x -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := -1

/-- Theorem: The vertex of the quadratic function f(x) = x^2 - 2x has coordinates (1, -1) -/
theorem vertex_coordinates :
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ f vertex_x = vertex_y :=
sorry

end vertex_coordinates_l3355_335566


namespace isosceles_triangle_area_l3355_335526

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  -- The length of the altitude to the base
  altitude : ℝ
  -- The perimeter of the triangle
  perimeter : ℝ
  -- The triangle is isosceles
  isIsosceles : True

/-- The area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℝ :=
  sorry -- The actual calculation of the area would go here

/-- Theorem: The area of an isosceles triangle with altitude 10 and perimeter 40 is 75 -/
theorem isosceles_triangle_area :
  ∀ t : IsoscelesTriangle, t.altitude = 10 ∧ t.perimeter = 40 → area t = 75 :=
by sorry


end isosceles_triangle_area_l3355_335526


namespace arithmetic_geometric_sequence_l3355_335584

/-- Given an arithmetic sequence with common difference 2,
    if a_1, a_3, and a_4 form a geometric sequence, then a_1 = -8 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →         -- a_1, a_3, a_4 form a geometric sequence
  a 1 = -8 := by
sorry

end arithmetic_geometric_sequence_l3355_335584


namespace six_digit_number_theorem_l3355_335573

def is_valid_six_digit_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

def extract_digits (n : ℕ) : Fin 6 → ℕ
| 0 => n / 100000
| 1 => (n / 10000) % 10
| 2 => (n / 1000) % 10
| 3 => (n / 100) % 10
| 4 => (n / 10) % 10
| 5 => n % 10

theorem six_digit_number_theorem (n : ℕ) (hn : is_valid_six_digit_number n) :
  (extract_digits n 0 = 1) →
  (3 * n = (n % 100000) * 10 + 1) →
  (extract_digits n 1 + extract_digits n 2 + extract_digits n 3 + 
   extract_digits n 4 + extract_digits n 5 = 26) := by
  sorry

end six_digit_number_theorem_l3355_335573


namespace two_by_one_prism_net_squares_valid_nine_square_net_two_by_one_prism_net_property_l3355_335508

/-- Represents a rectangular prism --/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a net of a rectangular prism --/
structure PrismNet where
  squares : ℕ

/-- Function to calculate the number of squares in a prism net --/
def netSquares (prism : RectangularPrism) : ℕ :=
  2 * (prism.length * prism.width + prism.length * prism.height + prism.width * prism.height)

/-- Theorem stating that a 2x1x1 prism net has 10 squares --/
theorem two_by_one_prism_net_squares :
  let prism : RectangularPrism := ⟨2, 1, 1⟩
  netSquares prism = 10 := by sorry

/-- Theorem stating that removing one square from a 10-square net results in a 9-square net --/
theorem valid_nine_square_net (net : PrismNet) (h : net.squares = 10) :
  ∃ (reduced_net : PrismNet), reduced_net.squares = 9 := by sorry

/-- Main theorem combining the above results --/
theorem two_by_one_prism_net_property :
  let prism : RectangularPrism := ⟨2, 1, 1⟩
  let net : PrismNet := ⟨netSquares prism⟩
  ∃ (reduced_net : PrismNet), reduced_net.squares = 9 := by sorry

end two_by_one_prism_net_squares_valid_nine_square_net_two_by_one_prism_net_property_l3355_335508


namespace solve_system_1_l3355_335530

theorem solve_system_1 (x y : ℝ) : 
  2 * x + 3 * y = 16 ∧ x + 4 * y = 13 → x = 5 ∧ y = 2 := by
  sorry

#check solve_system_1

end solve_system_1_l3355_335530


namespace money_distribution_l3355_335539

theorem money_distribution (total : ℕ) (a b c d : ℕ) : 
  a + b + c + d = total →
  5 * b = 2 * a →
  4 * b = 2 * c →
  3 * b = 2 * d →
  c = d + 500 →
  d = 1500 :=
by sorry

end money_distribution_l3355_335539


namespace divisibility_product_l3355_335579

theorem divisibility_product (n a b c d : ℤ) 
  (ha : n ∣ a) (hb : n ∣ b) (hc : n ∣ c) (hd : n ∣ d) :
  n ∣ ((a - d) * (b - c)) := by
  sorry

end divisibility_product_l3355_335579


namespace line_circle_intersection_sufficient_not_necessary_condition_l3355_335593

theorem line_circle_intersection (k : ℝ) : 
  (∃ x y : ℝ, y = k * (x + 1) ∧ x^2 + y^2 - 2*x = 0) ↔ 
  (-Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3) :=
sorry

theorem sufficient_not_necessary_condition : 
  (∀ k : ℝ, -Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3 → 
    ∃ x y : ℝ, y = k * (x + 1) ∧ x^2 + y^2 - 2*x = 0) ∧
  (∃ k : ℝ, (k = -Real.sqrt 3 / 3 ∨ k = Real.sqrt 3 / 3) ∧
    ∃ x y : ℝ, y = k * (x + 1) ∧ x^2 + y^2 - 2*x = 0) :=
sorry

end line_circle_intersection_sufficient_not_necessary_condition_l3355_335593


namespace hcf_problem_l3355_335537

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 82500) (h2 : Nat.lcm a b = 1500) :
  Nat.gcd a b = 55 := by
  sorry

end hcf_problem_l3355_335537


namespace second_player_wins_l3355_335545

/-- A game where players take turns removing coins from a pile. -/
structure CoinGame where
  coins : ℕ              -- Number of coins in the pile
  max_take : ℕ           -- Maximum number of coins a player can take in one turn
  min_take : ℕ           -- Minimum number of coins a player can take in one turn

/-- Represents a player in the game. -/
inductive Player
| First
| Second

/-- Defines a winning strategy for a player. -/
def has_winning_strategy (game : CoinGame) (player : Player) : Prop :=
  ∃ (strategy : ℕ → ℕ), 
    (∀ n, game.min_take ≤ strategy n ∧ strategy n ≤ game.max_take) ∧
    (player = Player.First → strategy game.coins = game.coins) ∧
    (player = Player.Second → 
      ∀ first_move, game.min_take ≤ first_move ∧ first_move ≤ game.max_take →
        strategy (game.coins - first_move) = game.coins - first_move)

/-- The main theorem stating that the second player has a winning strategy in the specific game. -/
theorem second_player_wins :
  let game : CoinGame := { coins := 2016, max_take := 3, min_take := 1 }
  has_winning_strategy game Player.Second :=
sorry

end second_player_wins_l3355_335545


namespace quadratic_no_real_roots_l3355_335506

theorem quadratic_no_real_roots : ∀ x : ℝ, x^2 - 2*x + 2 ≠ 0 := by
  sorry

end quadratic_no_real_roots_l3355_335506


namespace lawnmower_blade_cost_l3355_335509

/-- The cost of a single lawnmower blade -/
def blade_cost : ℝ := sorry

/-- The number of lawnmower blades purchased -/
def num_blades : ℕ := 4

/-- The cost of the weed eater string -/
def string_cost : ℝ := 7

/-- The total cost of supplies -/
def total_cost : ℝ := 39

/-- Theorem stating that the cost of each lawnmower blade is $8 -/
theorem lawnmower_blade_cost : 
  blade_cost = 8 :=
by
  sorry

end lawnmower_blade_cost_l3355_335509


namespace airport_distance_l3355_335558

/-- The distance from David's home to the airport in miles. -/
def distance_to_airport : ℝ := 160

/-- David's initial speed in miles per hour. -/
def initial_speed : ℝ := 40

/-- The increase in David's speed in miles per hour. -/
def speed_increase : ℝ := 20

/-- The time in hours David would be late if he continued at the initial speed. -/
def time_late : ℝ := 0.75

/-- The time in hours David arrives early with increased speed. -/
def time_early : ℝ := 0.25

/-- Theorem stating that the distance to the airport is 160 miles. -/
theorem airport_distance : 
  ∃ (t : ℝ), 
    distance_to_airport = initial_speed * (t + time_late) ∧
    distance_to_airport - initial_speed = (initial_speed + speed_increase) * (t - 1 - time_early) :=
by
  sorry


end airport_distance_l3355_335558


namespace factorial_sum_equality_l3355_335538

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end factorial_sum_equality_l3355_335538


namespace find_x_l3355_335549

theorem find_x : ∃ x : ℝ, 
  (24 + 35 + 58) / 3 = ((19 + 51 + x) / 3) + 6 → x = 29 := by
  sorry

end find_x_l3355_335549


namespace standard_deviation_measures_stability_l3355_335507

-- Define a type for yield per acre
def YieldPerAcre := ℝ

-- Define a function to calculate the standard deviation
def standardDeviation (yields : List YieldPerAcre) : ℝ :=
  sorry  -- Implementation details omitted

-- Define a predicate for stability measure
def isStabilityMeasure (f : List YieldPerAcre → ℝ) : Prop :=
  sorry  -- Implementation details omitted

-- Theorem statement
theorem standard_deviation_measures_stability :
  ∀ (n : ℕ) (yields : List YieldPerAcre),
    n > 0 →
    yields.length = n →
    isStabilityMeasure standardDeviation :=
by sorry

end standard_deviation_measures_stability_l3355_335507


namespace trigonometric_simplification_l3355_335556

theorem trigonometric_simplification (x : ℝ) :
  (2 * Real.cos x ^ 4 - 2 * Real.cos x ^ 2 + 1/2) / 
  (2 * Real.tan (π/4 - x) * Real.sin (π/4 + x) ^ 2) = 
  (1/2) * Real.cos (2*x) := by
  sorry

end trigonometric_simplification_l3355_335556


namespace mean_home_runs_l3355_335520

theorem mean_home_runs : 
  let total_players : ℕ := 2 + 3 + 2 + 1 + 1
  let total_home_runs : ℕ := 2 * 5 + 3 * 6 + 2 * 8 + 1 * 9 + 1 * 11
  (total_home_runs : ℚ) / total_players = 64 / 9 := by sorry

end mean_home_runs_l3355_335520


namespace a_in_set_a_b_l3355_335503

universe u

variables {a b : Type u}

/-- Prove that a is an element of the set {a, b}. -/
theorem a_in_set_a_b : a ∈ ({a, b} : Set (Type u)) := by
  sorry

end a_in_set_a_b_l3355_335503


namespace parallel_resistors_existence_l3355_335571

theorem parallel_resistors_existence : ∃ (R R₁ R₂ : ℕ+), 
  R.val * (R₁.val + R₂.val) = R₁.val * R₂.val ∧ 
  R.val > 0 ∧ R₁.val > 0 ∧ R₂.val > 0 := by
  sorry

end parallel_resistors_existence_l3355_335571


namespace grid_paths_equals_choose_l3355_335585

/-- The number of paths from (0,0) to (m,n) in a grid, moving only right or up -/
def gridPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) n

/-- Theorem: The number of paths in an m × n grid is (m+n) choose n -/
theorem grid_paths_equals_choose (m n : ℕ) : 
  gridPaths m n = Nat.choose (m + n) n := by
  sorry

end grid_paths_equals_choose_l3355_335585


namespace cos120_plus_sin_neg45_l3355_335532

theorem cos120_plus_sin_neg45 : 
  Real.cos (120 * π / 180) + Real.sin (-45 * π / 180) = - (1 + Real.sqrt 2) / 2 := by
  sorry

end cos120_plus_sin_neg45_l3355_335532


namespace squareable_numbers_l3355_335540

/-- A natural number is squareable if the numbers from 1 to n can be arranged
    such that each number plus its index is a perfect square. -/
def Squareable (n : ℕ) : Prop :=
  ∃ (σ : Fin n → Fin n), Function.Bijective σ ∧
    ∀ (i : Fin n), ∃ (k : ℕ), (σ i).val + i.val + 1 = k^2

theorem squareable_numbers : 
  ¬ Squareable 7 ∧ Squareable 9 ∧ ¬ Squareable 11 ∧ Squareable 15 :=
sorry

end squareable_numbers_l3355_335540


namespace smallest_prime_factor_in_C_l3355_335554

def C : Set Nat := {65, 67, 68, 71, 73}

def hasSmallerPrimeFactor (a b : Nat) : Prop :=
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ p ∣ a ∧ q ∣ b ∧ ∀ r < q, Nat.Prime r → ¬(r ∣ b)

theorem smallest_prime_factor_in_C :
  ∀ n ∈ C, n ≠ 68 → hasSmallerPrimeFactor 68 n :=
by sorry

end smallest_prime_factor_in_C_l3355_335554


namespace fraction_equals_zero_l3355_335518

theorem fraction_equals_zero (x : ℝ) : (x - 2) / (x + 2) = 0 → x = 2 := by
  sorry

end fraction_equals_zero_l3355_335518


namespace no_constant_difference_integer_l3355_335590

theorem no_constant_difference_integer (x : ℤ) : 
  ¬∃ (k : ℤ), 
    (x^2 - 4*x + 5) - (2*x - 6) = k ∧ 
    (4*x - 8) - (x^2 - 4*x + 5) = k ∧ 
    (3*x^2 - 12*x + 11) - (4*x - 8) = k :=
by sorry

end no_constant_difference_integer_l3355_335590


namespace early_registration_percentage_l3355_335565

/-- The percentage of attendees who registered at least two weeks in advance and paid in full -/
def early_reg_and_paid : ℝ := 78

/-- The percentage of attendees who paid in full but did not register early -/
def paid_not_early : ℝ := 10

/-- Proves that the percentage of attendees who registered at least two weeks in advance is 78% -/
theorem early_registration_percentage : ℝ := by
  sorry

end early_registration_percentage_l3355_335565


namespace fixed_point_and_min_product_l3355_335543

/-- The line l passing through a fixed point P -/
def line_l (m x y : ℝ) : Prop := (3*m + 1)*x + (2 + 2*m)*y - 8 = 0

/-- The fixed point P -/
def point_P : ℝ × ℝ := (-4, 6)

/-- Line l₁ -/
def line_l1 (x : ℝ) : Prop := x = -1

/-- Line l₂ -/
def line_l2 (y : ℝ) : Prop := y = -1

/-- Theorem stating that P is the fixed point and the minimum value of |PM| · |PN| -/
theorem fixed_point_and_min_product :
  (∀ m : ℝ, line_l m (point_P.1) (point_P.2)) ∧
  (∃ min : ℝ, min = 42 ∧
    ∀ m : ℝ, ∀ M N : ℝ × ℝ,
      line_l m M.1 M.2 → line_l1 M.1 →
      line_l m N.1 N.2 → line_l2 N.2 →
      (M.1 - point_P.1)^2 + (M.2 - point_P.2)^2 *
      (N.1 - point_P.1)^2 + (N.2 - point_P.2)^2 ≥ min^2) :=
sorry

end fixed_point_and_min_product_l3355_335543


namespace partnership_profit_calculation_l3355_335542

/-- Calculates the total profit of a partnership given investments and one partner's profit share -/
def calculate_total_profit (a_investment b_investment c_investment : ℕ) (c_profit : ℕ) : ℕ :=
  let ratio_sum := (a_investment / 8000) + (b_investment / 8000) + (c_investment / 8000)
  let profit_per_part := c_profit / (c_investment / 8000)
  ratio_sum * profit_per_part

/-- Theorem stating that given the specific investments and C's profit, the total profit is 92000 -/
theorem partnership_profit_calculation :
  calculate_total_profit 24000 32000 36000 36000 = 92000 := by
  sorry

end partnership_profit_calculation_l3355_335542


namespace unique_line_exists_l3355_335524

def intersectionPoints (k m n c : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((k, k^2 + 8*k + c), (k, m*k + n))

def verticalDistance (p q : ℝ × ℝ) : ℝ :=
  |p.2 - q.2|

theorem unique_line_exists (c : ℝ) : 
  ∃! m n : ℝ, n ≠ 0 ∧ 
  (∃ k : ℝ, verticalDistance (intersectionPoints k m n c).1 (intersectionPoints k m n c).2 = 4) ∧
  (m * 2 + n = 7) :=
sorry

end unique_line_exists_l3355_335524


namespace different_color_probability_l3355_335522

/-- The probability of drawing two balls of different colors from a box with 2 red and 3 black balls -/
theorem different_color_probability : 
  let total_balls : ℕ := 2 + 3
  let red_balls : ℕ := 2
  let black_balls : ℕ := 3
  let different_color_draws : ℕ := red_balls * black_balls
  let total_draws : ℕ := (total_balls * (total_balls - 1)) / 2
  (different_color_draws : ℚ) / total_draws = 3 / 5 := by
  sorry

end different_color_probability_l3355_335522


namespace odd_digits_157_base5_l3355_335512

/-- Represents a number in base 5 as a list of digits (least significant digit first) -/
def Base5Rep := List Nat

/-- Converts a natural number to its base 5 representation -/
def toBase5 (n : Nat) : Base5Rep :=
  sorry

/-- Counts the number of odd digits in a base 5 representation -/
def countOddDigits (rep : Base5Rep) : Nat :=
  sorry

/-- The number of odd digits in the base-5 representation of 157₁₀ is 3 -/
theorem odd_digits_157_base5 : countOddDigits (toBase5 157) = 3 := by
  sorry

end odd_digits_157_base5_l3355_335512


namespace a_power_b_minus_a_power_neg_b_l3355_335521

theorem a_power_b_minus_a_power_neg_b (a b : ℝ) (ha : a > 1) (hb : b > 0) 
  (h : a^b + a^(-b) = 2 * Real.sqrt 2) : a^b - a^(-b) = 2 := by
  sorry

end a_power_b_minus_a_power_neg_b_l3355_335521


namespace keanu_destination_distance_l3355_335562

/-- Represents the distance to Keanu's destination -/
def destination_distance : ℝ := 280

/-- Represents the capacity of Keanu's motorcycle's gas tank in liters -/
def tank_capacity : ℝ := 8

/-- Represents the distance Keanu's motorcycle can travel with one full tank in miles -/
def miles_per_tank : ℝ := 40

/-- Represents the number of times Keanu refills his motorcycle for a round trip -/
def refills : ℕ := 14

/-- Theorem stating that the distance to Keanu's destination is 280 miles -/
theorem keanu_destination_distance :
  destination_distance = (refills : ℝ) * miles_per_tank / 2 :=
sorry

end keanu_destination_distance_l3355_335562


namespace rose_apples_l3355_335597

/-- The number of friends Rose shares her apples with -/
def num_friends : ℕ := 3

/-- The number of apples each friend would get if Rose shares her apples -/
def apples_per_friend : ℕ := 3

/-- The total number of apples Rose has -/
def total_apples : ℕ := num_friends * apples_per_friend

theorem rose_apples : total_apples = 9 := by sorry

end rose_apples_l3355_335597


namespace tetrahedron_pigeonhole_l3355_335534

/-- Represents the three possible states a point can be in -/
inductive PointState
  | Type1
  | Type2
  | Outside

/-- Represents a tetrahedron with vertices labeled by their state -/
structure Tetrahedron :=
  (vertices : Fin 4 → PointState)

/-- Theorem statement -/
theorem tetrahedron_pigeonhole (t : Tetrahedron) : 
  ∃ (i j : Fin 4), i ≠ j ∧ t.vertices i = t.vertices j :=
sorry

end tetrahedron_pigeonhole_l3355_335534


namespace point_C_y_coordinate_sum_of_digits_l3355_335557

/-- The function representing the graph y = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- Sum of digits of a real number -/
noncomputable def sumOfDigits (y : ℝ) : ℕ := sorry

theorem point_C_y_coordinate_sum_of_digits 
  (A B C : ℝ × ℝ) 
  (hA : A.2 = f A.1) 
  (hB : B.2 = f B.1) 
  (hC : C.2 = f C.1) 
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (hParallel : A.2 = B.2) 
  (hArea : abs ((B.1 - A.1) * (C.2 - A.2)) / 2 = 100) :
  sumOfDigits C.2 = 6 := by sorry

end point_C_y_coordinate_sum_of_digits_l3355_335557


namespace rectangle_area_ratio_l3355_335583

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if a / c = b / d = 2 / 3, then the ratio of the area of rectangle A
    to the area of rectangle B is 4:9. -/
theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 2 / 3) (h2 : b / d = 2 / 3) :
  (a * b) / (c * d) = 4 / 9 := by
  sorry

end rectangle_area_ratio_l3355_335583


namespace power_product_equals_reciprocal_l3355_335587

theorem power_product_equals_reciprocal (n : ℕ) :
  (125 : ℚ)^(2015 : ℕ) * (-0.008)^(2016 : ℕ) = 1 / 125 :=
by
  have h : (-0.008 : ℚ) = -1/125 := by sorry
  sorry

end power_product_equals_reciprocal_l3355_335587


namespace function_properties_l3355_335572

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| - 2 * |x + 1|

-- State the theorem
theorem function_properties :
  -- 1. The maximum value of f is 4
  (∃ (x : ℝ), f x = 4) ∧ (∀ (x : ℝ), f x ≤ 4) ∧
  -- 2. The solution set of f(x) < 1
  (∀ (x : ℝ), f x < 1 ↔ (x < -4 ∨ x > 0)) ∧
  -- 3. The maximum value of ab + bc given the constraints
  (∀ (a b c : ℝ), a > 0 → b > 0 → a^2 + 2*b^2 + c^2 = 4 → ab + bc ≤ 2) ∧
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ a^2 + 2*b^2 + c^2 = 4 ∧ ab + bc = 2) :=
by sorry

end function_properties_l3355_335572


namespace radish_patch_area_l3355_335546

theorem radish_patch_area (pea_patch : ℝ) (radish_patch : ℝ) : 
  pea_patch = 2 * radish_patch →
  pea_patch / 6 = 5 →
  radish_patch = 15 := by
  sorry

end radish_patch_area_l3355_335546


namespace mildreds_initial_blocks_l3355_335510

/-- Proves that Mildred's initial number of blocks was 2, given that she found 84 blocks and ended up with 86 blocks. -/
theorem mildreds_initial_blocks (found_blocks : ℕ) (final_blocks : ℕ) (h1 : found_blocks = 84) (h2 : final_blocks = 86) :
  final_blocks - found_blocks = 2 := by
  sorry

#check mildreds_initial_blocks

end mildreds_initial_blocks_l3355_335510


namespace monkey_giraffe_difference_l3355_335589

/-- The number of zebras Carla counted at the zoo -/
def num_zebras : ℕ := 12

/-- The number of camels at the zoo -/
def num_camels : ℕ := num_zebras / 2

/-- The number of monkeys at the zoo -/
def num_monkeys : ℕ := 4 * num_camels

/-- The number of giraffes at the zoo -/
def num_giraffes : ℕ := 2

/-- Theorem stating the difference between the number of monkeys and giraffes -/
theorem monkey_giraffe_difference : num_monkeys - num_giraffes = 22 := by
  sorry

end monkey_giraffe_difference_l3355_335589


namespace initial_number_of_persons_l3355_335541

theorem initial_number_of_persons 
  (average_weight_increase : ℝ) 
  (weight_of_leaving_person : ℝ) 
  (weight_of_new_person : ℝ) : 
  average_weight_increase = 4.5 ∧ 
  weight_of_leaving_person = 65 ∧ 
  weight_of_new_person = 74 → 
  (weight_of_new_person - weight_of_leaving_person) / average_weight_increase = 2 := by
  sorry

end initial_number_of_persons_l3355_335541


namespace dresser_shirts_count_l3355_335586

/-- Given a dresser with pants and shirts in the ratio of 7:10, 
    and 14 pants, prove that there are 20 shirts. -/
theorem dresser_shirts_count (pants_count : ℕ) (ratio_pants : ℕ) (ratio_shirts : ℕ) :
  pants_count = 14 →
  ratio_pants = 7 →
  ratio_shirts = 10 →
  (pants_count : ℚ) / ratio_pants * ratio_shirts = 20 := by
  sorry

end dresser_shirts_count_l3355_335586


namespace train_and_car_numbers_l3355_335514

/-- Represents a digit in the range 0 to 9 -/
def Digit := Fin 10

/-- Represents a mapping from characters to digits -/
def CodeMap := Char → Digit

/-- Checks if a CodeMap is valid (injective) -/
def isValidCodeMap (m : CodeMap) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → m c1 ≠ m c2

/-- Converts a string to a number using a CodeMap -/
def stringToNumber (s : String) (m : CodeMap) : ℕ :=
  s.foldl (fun acc c => acc * 10 + (m c).val) 0

/-- The main theorem -/
theorem train_and_car_numbers : ∃ (m : CodeMap),
  isValidCodeMap m ∧
  stringToNumber "SECRET" m - stringToNumber "OPEN" m = stringToNumber "ANSWER" m - stringToNumber "YOUR" m ∧
  stringToNumber "SECRET" m - stringToNumber "OPENED" m = 20010 ∧
  stringToNumber "TRAIN" m = 392 ∧
  stringToNumber "CAR" m = 2 := by
  sorry

end train_and_car_numbers_l3355_335514


namespace power_subtraction_equivalence_l3355_335511

theorem power_subtraction_equivalence :
  2^345 - 3^4 * 9^2 = 2^345 - 6561 :=
by sorry

end power_subtraction_equivalence_l3355_335511


namespace floor_length_percentage_more_than_breadth_l3355_335519

theorem floor_length_percentage_more_than_breadth 
  (length : Real) 
  (area : Real) 
  (h1 : length = 13.416407864998739)
  (h2 : area = 60) :
  let breadth := area / length
  (length - breadth) / breadth * 100 = 200 := by
  sorry

end floor_length_percentage_more_than_breadth_l3355_335519


namespace units_digit_of_41_cubed_plus_23_cubed_l3355_335588

theorem units_digit_of_41_cubed_plus_23_cubed : (41^3 + 23^3) % 10 = 8 := by
  sorry

end units_digit_of_41_cubed_plus_23_cubed_l3355_335588


namespace ryan_english_hours_l3355_335595

/-- The number of hours Ryan spends on learning Spanish -/
def spanish_hours : ℕ := 4

/-- The additional hours Ryan spends on learning English compared to Spanish -/
def additional_english_hours : ℕ := 3

/-- The number of hours Ryan spends on learning English -/
def english_hours : ℕ := spanish_hours + additional_english_hours

theorem ryan_english_hours : english_hours = 7 := by
  sorry

end ryan_english_hours_l3355_335595


namespace difference_value_l3355_335531

-- Define the variables x and y
variable (x y : ℝ)

-- Define the conditions
def sum_condition : Prop := x + y = 500
def ratio_condition : Prop := x / y = 0.8

-- Define the theorem
theorem difference_value (h1 : sum_condition x y) (h2 : ratio_condition x y) :
  ∃ ε > 0, |y - x - 55.56| < ε :=
sorry

end difference_value_l3355_335531


namespace derivative_at_one_l3355_335563

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f x = x^2 + 3*x*(f' 1)) →
  (∀ x, HasDerivAt f (f' x) x) →
  f' 1 = -1 := by
sorry

end derivative_at_one_l3355_335563


namespace complex_power_four_equals_negative_four_l3355_335591

theorem complex_power_four_equals_negative_four : 
  (1 + (1 / Complex.I)) ^ 4 = (-4 : ℂ) := by sorry

end complex_power_four_equals_negative_four_l3355_335591


namespace acute_triangle_angle_inequality_iff_sine_inequality_l3355_335568

theorem acute_triangle_angle_inequality_iff_sine_inequality 
  (A B C : Real) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  (A > B ∧ B > C) ↔ (Real.sin (2*A) < Real.sin (2*B) ∧ Real.sin (2*B) < Real.sin (2*C)) :=
sorry

end acute_triangle_angle_inequality_iff_sine_inequality_l3355_335568


namespace book_club_combinations_l3355_335567

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of people in the book club --/
def total_people : ℕ := 5

/-- The number of people who lead the discussion --/
def discussion_leaders : ℕ := 3

theorem book_club_combinations :
  choose total_people discussion_leaders = 10 := by
  sorry

end book_club_combinations_l3355_335567


namespace correct_operation_l3355_335500

theorem correct_operation (a b : ℝ) : 3 * a^2 * b - b * a^2 = 2 * a^2 * b := by
  sorry

end correct_operation_l3355_335500


namespace system_solution_l3355_335555

theorem system_solution (a b c d : ℚ) 
  (eq1 : 3 * a + 4 * b + 6 * c + 8 * d = 48)
  (eq2 : 4 * (d + c) = b)
  (eq3 : 4 * b + 2 * c = a)
  (eq4 : c + 1 = d) :
  a + b + c + d = 513 / 37 := by
sorry

end system_solution_l3355_335555


namespace complex_equation_solution_l3355_335533

theorem complex_equation_solution (b : ℝ) : (2 + b * Complex.I) * Complex.I = 2 + 2 * Complex.I → b = -2 := by
  sorry

end complex_equation_solution_l3355_335533


namespace complex_simplification_l3355_335501

theorem complex_simplification : (4 - 3*I)^2 + (1 + 2*I) = 8 - 22*I := by
  sorry

end complex_simplification_l3355_335501


namespace cookies_left_l3355_335536

theorem cookies_left (initial_cookies eaten_cookies : ℕ) 
  (h1 : initial_cookies = 93)
  (h2 : eaten_cookies = 15) :
  initial_cookies - eaten_cookies = 78 := by
  sorry

end cookies_left_l3355_335536


namespace problem_statement_l3355_335582

theorem problem_statement : (-4)^4 / 4^2 + 2^5 - 7^2 = -1 := by
  sorry

end problem_statement_l3355_335582


namespace pencil_sharpening_l3355_335505

/-- Given the initial and final lengths of a pencil, calculate the length sharpened off. -/
theorem pencil_sharpening (initial_length final_length : ℕ) : 
  initial_length ≥ final_length → 
  initial_length - final_length = initial_length - final_length :=
by sorry

end pencil_sharpening_l3355_335505


namespace function_forms_with_common_tangent_l3355_335529

/-- Given two functions f and g, prove that they have the specified forms
    when they pass through (2, 0) and have a common tangent at that point. -/
theorem function_forms_with_common_tangent 
  (f g : ℝ → ℝ) 
  (hf : ∃ a : ℝ, ∀ x, f x = 2 * x^3 + a * x)
  (hg : ∃ b c : ℝ, ∀ x, g x = b * x^2 + c)
  (pass_through : f 2 = 0 ∧ g 2 = 0)
  (common_tangent : (deriv f) 2 = (deriv g) 2) :
  (∀ x, f x = 2 * x^3 - 8 * x) ∧ 
  (∀ x, g x = 4 * x^2 - 16) := by
sorry

end function_forms_with_common_tangent_l3355_335529


namespace missing_digit_divisible_by_six_l3355_335525

theorem missing_digit_divisible_by_six : ∃ (n : ℕ), 
  n ≥ 31610 ∧ n ≤ 31619 ∧ n % 10 = 4 ∧ n % 100 = 14 ∧ n % 6 = 0 := by
  sorry

end missing_digit_divisible_by_six_l3355_335525


namespace conference_handshakes_l3355_335569

/-- Represents a conference with handshakes --/
structure Conference where
  total_people : Nat
  normal_handshakes : Nat
  restricted_people : Nat
  restricted_handshakes : Nat

/-- Calculates the maximum number of unique handshakes in a conference --/
def max_handshakes (c : Conference) : Nat :=
  let total_pairs := c.total_people.choose 2
  let reduced_handshakes := c.restricted_people * (c.normal_handshakes - c.restricted_handshakes)
  total_pairs - reduced_handshakes

/-- The theorem stating the maximum number of handshakes for the given conference --/
theorem conference_handshakes :
  let c : Conference := {
    total_people := 25,
    normal_handshakes := 20,
    restricted_people := 5,
    restricted_handshakes := 15
  }
  max_handshakes c = 250 := by
  sorry

end conference_handshakes_l3355_335569


namespace sqrt_39_equals_33_l3355_335598

theorem sqrt_39_equals_33 : Real.sqrt 39 = 33 := by
  sorry

end sqrt_39_equals_33_l3355_335598


namespace valuation_problems_l3355_335550

/-- The p-adic valuation of an integer n -/
noncomputable def padic_valuation (p : ℕ) (n : ℤ) : ℕ := sorry

theorem valuation_problems :
  (padic_valuation 3 (2^27 + 1) = 4) ∧
  (padic_valuation 7 (161^14 - 112^14) = 16) ∧
  (padic_valuation 2 (7^20 + 1) = 1) ∧
  (padic_valuation 2 (17^48 - 5^48) = 6) := by sorry

end valuation_problems_l3355_335550


namespace projection_problem_l3355_335516

/-- Given that the projection of (2, -3) onto some vector results in (1, -3/2),
    prove that the projection of (-3, 2) onto the same vector is (-24/13, 36/13) -/
theorem projection_problem (v : ℝ × ℝ) :
  let u₁ : ℝ × ℝ := (2, -3)
  let u₂ : ℝ × ℝ := (-3, 2)
  let proj₁ : ℝ × ℝ := (1, -3/2)
  (∃ (k : ℝ), v = k • proj₁) →
  (u₁ • v / (v • v)) • v = proj₁ →
  (u₂ • v / (v • v)) • v = (-24/13, 36/13) := by
  sorry


end projection_problem_l3355_335516


namespace ozone_effect_significant_l3355_335513

/-- Represents the data from the experiment -/
structure ExperimentData where
  control_group : List Float
  experimental_group : List Float

/-- Calculates the median of a sorted list -/
def median (sorted_list : List Float) : Float :=
  sorry

/-- Counts the number of elements less than a given value -/
def count_less_than (list : List Float) (value : Float) : Nat :=
  sorry

/-- Calculates K² statistic -/
def calculate_k_squared (a b c d : Nat) : Float :=
  sorry

/-- Main theorem: K² value is greater than the critical value for 95% confidence -/
theorem ozone_effect_significant (data : ExperimentData) :
  let all_data := data.control_group ++ data.experimental_group
  let m := median all_data
  let a := count_less_than data.control_group m
  let b := data.control_group.length - a
  let c := count_less_than data.experimental_group m
  let d := data.experimental_group.length - c
  let k_squared := calculate_k_squared a b c d
  k_squared > 3.841 := by
  sorry

#check ozone_effect_significant

end ozone_effect_significant_l3355_335513


namespace missing_sale_is_correct_l3355_335578

/-- Calculates the missing sale amount given the other 5 sales and the target average -/
def calculate_missing_sale (sale1 sale3 sale4 sale5 sale6 average : ℝ) : ℝ :=
  6 * average - (sale1 + sale3 + sale4 + sale5 + sale6)

/-- Proves that the calculated missing sale is correct given the problem conditions -/
theorem missing_sale_is_correct (sale1 sale3 sale4 sale5 sale6 average : ℝ) 
  (h1 : sale1 = 5420)
  (h3 : sale3 = 6200)
  (h4 : sale4 = 6350)
  (h5 : sale5 = 6500)
  (h6 : sale6 = 7070)
  (havg : average = 6200) :
  calculate_missing_sale sale1 sale3 sale4 sale5 sale6 average = 5660 := by
  sorry

#eval calculate_missing_sale 5420 6200 6350 6500 7070 6200

end missing_sale_is_correct_l3355_335578


namespace triangle_product_inequality_l3355_335594

/-- Triangle structure with sides a, b, c, perimeter P, and inscribed circle radius r -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  P : ℝ
  r : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_P : 0 < P
  pos_r : 0 < r
  perimeter_def : P = a + b + c
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- The product of any two sides of a triangle is not less than
    the product of its perimeter and the radius of its inscribed circle -/
theorem triangle_product_inequality (t : Triangle) : t.a * t.b ≥ t.P * t.r := by
  sorry

end triangle_product_inequality_l3355_335594


namespace intersection_A_B_l3355_335551

-- Define set A
def A : Set ℝ := {x | x - 1 ≤ 0}

-- Define set B
def B : Set ℝ := {0, 1, 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {0, 1} := by
  sorry

end intersection_A_B_l3355_335551


namespace quadratic_one_solution_l3355_335548

theorem quadratic_one_solution (q : ℝ) (h : q ≠ 0) : 
  (q = 64/9) ↔ (∃! x : ℝ, q * x^2 - 16 * x + 9 = 0) := by
  sorry

end quadratic_one_solution_l3355_335548


namespace solve_equation_1_solve_equation_2_solve_equation_3_l3355_335577

-- Equation 1: (x-2)^2 = 25
theorem solve_equation_1 : 
  ∃ x₁ x₂ : ℝ, (x₁ - 2)^2 = 25 ∧ (x₂ - 2)^2 = 25 ∧ x₁ = 7 ∧ x₂ = -3 :=
sorry

-- Equation 2: x^2 + 4x + 3 = 0
theorem solve_equation_2 : 
  ∃ x₁ x₂ : ℝ, x₁^2 + 4*x₁ + 3 = 0 ∧ x₂^2 + 4*x₂ + 3 = 0 ∧ x₁ = -3 ∧ x₂ = -1 :=
sorry

-- Equation 3: 2x^2 + 4x - 1 = 0
theorem solve_equation_3 : 
  ∃ x₁ x₂ : ℝ, 2*x₁^2 + 4*x₁ - 1 = 0 ∧ 2*x₂^2 + 4*x₂ - 1 = 0 ∧ 
  x₁ = (-2 + Real.sqrt 6) / 2 ∧ x₂ = (-2 - Real.sqrt 6) / 2 :=
sorry

end solve_equation_1_solve_equation_2_solve_equation_3_l3355_335577


namespace intersection_of_A_and_B_l3355_335502

def A : Set ℝ := {x | 2 * x^2 - 3 * x - 2 ≤ 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by sorry

end intersection_of_A_and_B_l3355_335502


namespace inradius_right_triangle_l3355_335599

/-- The inradius of a right triangle with side lengths 9, 40, and 41 is 4. -/
theorem inradius_right_triangle : ∀ (a b c r : ℝ),
  a = 9 ∧ b = 40 ∧ c = 41 →
  a^2 + b^2 = c^2 →
  r = (a * b) / (2 * (a + b + c)) →
  r = 4 := by sorry

end inradius_right_triangle_l3355_335599


namespace quadratic_rewrite_l3355_335570

theorem quadratic_rewrite (d e f : ℤ) : 
  (∀ x, 25 * x^2 - 40 * x - 75 = (d * x + e)^2 + f) → d * e = -20 := by
  sorry

end quadratic_rewrite_l3355_335570


namespace plan_b_more_cost_effective_l3355_335547

/-- Plan A's cost per megabyte in cents -/
def plan_a_cost_per_mb : ℚ := 12

/-- Plan B's setup fee in cents -/
def plan_b_setup_fee : ℚ := 3000

/-- Plan B's cost per megabyte in cents -/
def plan_b_cost_per_mb : ℚ := 8

/-- The minimum number of megabytes for Plan B to be more cost-effective -/
def min_mb_for_plan_b : ℕ := 751

theorem plan_b_more_cost_effective :
  (↑min_mb_for_plan_b * plan_b_cost_per_mb + plan_b_setup_fee < ↑min_mb_for_plan_b * plan_a_cost_per_mb) ∧
  ∀ m : ℕ, m < min_mb_for_plan_b →
    (↑m * plan_b_cost_per_mb + plan_b_setup_fee ≥ ↑m * plan_a_cost_per_mb) :=
by sorry

end plan_b_more_cost_effective_l3355_335547


namespace license_plate_count_l3355_335517

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of letter positions in the license plate -/
def letter_positions : ℕ := 5

/-- The number of digit positions in the license plate -/
def digit_positions : ℕ := 2

/-- The number of odd single-digit numbers -/
def odd_digits : ℕ := 5

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of license plate combinations -/
def license_plate_combinations : ℕ :=
  (choose alphabet_size 2) * (choose letter_positions 2) * (choose (letter_positions - 2) 2) * 
  (alphabet_size - 2) * odd_digits * (odd_digits - 1)

theorem license_plate_count : license_plate_combinations = 936000 := by
  sorry

end license_plate_count_l3355_335517


namespace arithmetic_geometric_progression_ratio_l3355_335559

theorem arithmetic_geometric_progression_ratio 
  (a₁ d : ℝ) (h : d ≠ 0) : 
  let a₂ := a₁ + d
  let a₃ := a₁ + 2*d
  let r := a₂ * a₃ / (a₁ * a₂)
  (r * r = 1 ∧ (a₂ * a₃) / (a₁ * a₂) = (a₃ * a₁) / (a₂ * a₃)) → r = -2 := by
  sorry

#check arithmetic_geometric_progression_ratio

end arithmetic_geometric_progression_ratio_l3355_335559


namespace largest_prime_factor_of_6370_l3355_335523

theorem largest_prime_factor_of_6370 : (Nat.factors 6370).maximum? = some 13 := by
  sorry

end largest_prime_factor_of_6370_l3355_335523


namespace odd_square_sum_parity_l3355_335515

theorem odd_square_sum_parity (n m : ℤ) (h : Odd (n^2 + m^2)) :
  ¬(Even n ∧ Even m) ∧ ¬(Odd n ∧ Odd m) := by
  sorry

end odd_square_sum_parity_l3355_335515


namespace monotonic_increasing_interval_f_l3355_335504

-- Define the function
def f (x : ℝ) : ℝ := |x - 2| - 1

-- State the theorem
theorem monotonic_increasing_interval_f :
  ∀ a b : ℝ, a ≥ 2 → b ≥ 2 → a ≤ b → f a ≤ f b :=
by sorry

end monotonic_increasing_interval_f_l3355_335504


namespace negation_of_implication_l3355_335544

theorem negation_of_implication :
  (¬(∀ x : ℝ, x > 1 → x^2 > 1)) ↔ (∀ x : ℝ, x ≤ 1 → x^2 ≤ 1) := by sorry

end negation_of_implication_l3355_335544


namespace larger_cube_volume_l3355_335580

/-- The volume of a cube composed of 125 smaller cubes is equal to 125 times the volume of one small cube. -/
theorem larger_cube_volume (small_cube_volume : ℝ) (larger_cube_volume : ℝ) 
  (h1 : small_cube_volume > 0)
  (h2 : larger_cube_volume > 0)
  (h3 : ∃ (n : ℕ), n ^ 3 = 125)
  (h4 : larger_cube_volume = (5 : ℝ) ^ 3 * small_cube_volume) :
  larger_cube_volume = 125 * small_cube_volume := by
  sorry

end larger_cube_volume_l3355_335580


namespace sum_of_four_consecutive_integers_divisible_by_two_l3355_335552

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  ∃ (k : ℤ), (n - 1) + n + (n + 1) + (n + 2) = 2 * k := by
  sorry

end sum_of_four_consecutive_integers_divisible_by_two_l3355_335552


namespace probability_sum_10_15_18_l3355_335561

def num_dice : ℕ := 3
def faces_per_die : ℕ := 6

def total_outcomes : ℕ := faces_per_die ^ num_dice

def sum_10_outcomes : ℕ := 27
def sum_15_outcomes : ℕ := 9
def sum_18_outcomes : ℕ := 1

def favorable_outcomes : ℕ := sum_10_outcomes + sum_15_outcomes + sum_18_outcomes

theorem probability_sum_10_15_18 : 
  (favorable_outcomes : ℚ) / total_outcomes = 37 / 216 := by sorry

end probability_sum_10_15_18_l3355_335561


namespace seventh_term_is_64_l3355_335528

/-- A geometric sequence with given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 2) * a n = (a (n + 1))^2
  sum_first_two : a 1 + a 2 = 3
  sum_second_third : a 2 + a 3 = 6

/-- The 7th term of the geometric sequence is 64 -/
theorem seventh_term_is_64 (seq : GeometricSequence) : seq.a 7 = 64 := by
  sorry

end seventh_term_is_64_l3355_335528


namespace aquarium_dolphins_l3355_335581

/-- The number of hours each dolphin requires for daily training -/
def training_hours_per_dolphin : ℕ := 3

/-- The number of trainers in the aquarium -/
def number_of_trainers : ℕ := 2

/-- The number of hours each trainer spends training dolphins -/
def hours_per_trainer : ℕ := 6

/-- The total number of training hours available -/
def total_training_hours : ℕ := number_of_trainers * hours_per_trainer

/-- The number of dolphins in the aquarium -/
def number_of_dolphins : ℕ := total_training_hours / training_hours_per_dolphin

theorem aquarium_dolphins :
  number_of_dolphins = 4 := by sorry

end aquarium_dolphins_l3355_335581


namespace inverse_proportion_k_value_l3355_335553

/-- Inverse proportion function passing through (2,1) has k = 2 -/
theorem inverse_proportion_k_value (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = k / x) ∧ f 2 = 1) → k = 2 := by
  sorry

end inverse_proportion_k_value_l3355_335553


namespace membership_change_l3355_335575

theorem membership_change (initial_members : ℝ) : 
  let fall_increase := 0.07
  let spring_decrease := 0.19
  let fall_members := initial_members * (1 + fall_increase)
  let spring_members := fall_members * (1 - spring_decrease)
  let total_change_percentage := (spring_members / initial_members - 1) * 100
  total_change_percentage = -13.33 := by
sorry

end membership_change_l3355_335575


namespace cube_volume_from_face_perimeter_l3355_335574

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 20) :
  let side_length := face_perimeter / 4
  (side_length ^ 3) = 125 := by sorry

end cube_volume_from_face_perimeter_l3355_335574


namespace exists_set_product_eq_sum_squares_l3355_335576

/-- For any finite set of positive integers, there exists a larger finite set
    where the product of its elements equals the sum of their squares. -/
theorem exists_set_product_eq_sum_squares (A : Finset ℕ) : ∃ B : Finset ℕ, 
  (∀ a ∈ A, a ∈ B) ∧ 
  (∀ b ∈ B, b > 0) ∧
  (B.prod id = B.sum (λ x => x^2)) := by
  sorry

end exists_set_product_eq_sum_squares_l3355_335576
