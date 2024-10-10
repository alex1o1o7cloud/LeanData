import Mathlib

namespace concave_function_triangle_inequality_l310_31095

def f (x : ℝ) := x^2 - 2*x + 2

theorem concave_function_triangle_inequality (m : ℝ) : 
  (∀ a b c : ℝ, 1/3 ≤ a ∧ a < b ∧ b < c ∧ c ≤ m^2 - m + 2 → 
    f a + f b > f c ∧ f b + f c > f a ∧ f c + f a > f b) ↔ 
  0 ≤ m ∧ m ≤ 1 :=
sorry

end concave_function_triangle_inequality_l310_31095


namespace no_universal_rational_compact_cover_l310_31033

theorem no_universal_rational_compact_cover :
  ¬ (∃ (A : ℕ → Set ℚ), 
    (∀ n, IsCompact (A n)) ∧ 
    (∀ K : Set ℚ, IsCompact K → ∃ n, K ⊆ A n)) := by
  sorry

end no_universal_rational_compact_cover_l310_31033


namespace f_min_value_l310_31091

/-- The function f(x) = 9x - 4x^2 -/
def f (x : ℝ) := 9 * x - 4 * x^2

/-- The minimum value of f(x) is -81/16 -/
theorem f_min_value : ∀ x : ℝ, f x ≥ -81/16 := by sorry

end f_min_value_l310_31091


namespace absolute_value_inequality_solution_set_l310_31041

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 1| ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
by sorry

end absolute_value_inequality_solution_set_l310_31041


namespace train_platform_time_l310_31079

/-- Given a train of length 1200 meters that takes 120 seconds to pass a tree,
    this theorem proves that the time required for the train to pass a platform
    of length 800 meters is 200 seconds. -/
theorem train_platform_time (train_length : ℝ) (tree_pass_time : ℝ) (platform_length : ℝ)
  (h1 : train_length = 1200)
  (h2 : tree_pass_time = 120)
  (h3 : platform_length = 800) :
  (train_length + platform_length) / (train_length / tree_pass_time) = 200 :=
by sorry

end train_platform_time_l310_31079


namespace hillary_friday_reading_time_l310_31084

/-- The total assigned reading time in minutes -/
def total_assigned_time : ℕ := 60

/-- The number of minutes Hillary read on Saturday -/
def saturday_reading_time : ℕ := 28

/-- The number of minutes Hillary needs to read on Sunday -/
def sunday_reading_time : ℕ := 16

/-- The number of minutes Hillary read on Friday night -/
def friday_reading_time : ℕ := total_assigned_time - (saturday_reading_time + sunday_reading_time)

theorem hillary_friday_reading_time :
  friday_reading_time = 16 := by sorry

end hillary_friday_reading_time_l310_31084


namespace sunzi_car_problem_l310_31048

theorem sunzi_car_problem (x : ℕ) : 
  (3 * (x - 2) = 2 * x + 9) → x = 15 := by
  sorry

end sunzi_car_problem_l310_31048


namespace gravitational_force_at_distance_l310_31098

/-- Represents the gravitational force at a given distance -/
structure GravitationalForce where
  distance : ℝ
  force : ℝ

/-- The gravitational constant k = f * d^2 -/
def gravitational_constant (gf : GravitationalForce) : ℝ :=
  gf.force * gf.distance^2

theorem gravitational_force_at_distance 
  (surface_force : GravitationalForce) 
  (space_force : GravitationalForce) :
  surface_force.distance = 5000 →
  surface_force.force = 800 →
  space_force.distance = 300000 →
  gravitational_constant surface_force = gravitational_constant space_force →
  space_force.force = 1/45 := by
sorry

end gravitational_force_at_distance_l310_31098


namespace meaningful_set_equiv_range_expression_meaningful_iff_in_set_l310_31046

-- Define the set of real numbers for which the expression is meaningful
def MeaningfulSet : Set ℝ :=
  {x : ℝ | x ≥ -2/3 ∧ x ≠ 0}

-- Theorem stating that the MeaningfulSet is equivalent to the given range
theorem meaningful_set_equiv_range :
  MeaningfulSet = Set.Icc (-2/3) 0 ∪ Set.Ioi 0 :=
sorry

-- Theorem proving that the expression is meaningful if and only if x is in MeaningfulSet
theorem expression_meaningful_iff_in_set (x : ℝ) :
  (3 * x + 2 ≥ 0 ∧ x ≠ 0) ↔ x ∈ MeaningfulSet :=
sorry

end meaningful_set_equiv_range_expression_meaningful_iff_in_set_l310_31046


namespace proportion_inequality_l310_31008

theorem proportion_inequality (a b c : ℝ) (h : a / b = b / c) : a^2 + c^2 ≥ 2 * b^2 := by
  sorry

end proportion_inequality_l310_31008


namespace probability_of_rerolling_two_is_one_over_144_l310_31047

/-- Represents the outcome of rolling a single die -/
inductive DieOutcome
| One | Two | Three | Four | Five | Six

/-- Represents the state of a die (original or rerolled) -/
inductive DieState
| Original (outcome : DieOutcome)
| Rerolled (original : DieOutcome) (new : DieOutcome)

/-- Represents the game state after Jason's decision -/
structure GameState :=
(dice : Fin 3 → DieState)
(rerolledCount : Nat)

/-- Determines if a game state is winning -/
def isWinningState (state : GameState) : Bool :=
  sorry

/-- Calculates the probability of a given game state -/
def probabilityOfState (state : GameState) : ℚ :=
  sorry

/-- Calculates the probability of Jason choosing to reroll exactly two dice -/
def probabilityOfRerollingTwo : ℚ :=
  sorry

theorem probability_of_rerolling_two_is_one_over_144 :
  probabilityOfRerollingTwo = 1 / 144 :=
sorry

end probability_of_rerolling_two_is_one_over_144_l310_31047


namespace purely_imaginary_complex_number_l310_31043

theorem purely_imaginary_complex_number (a : ℝ) : 
  (∀ z : ℂ, z = (a^2 - 3*a + 2 : ℝ) + (a - 1 : ℝ)*I → z.re = 0 ∧ z.im ≠ 0) → a = 2 := by
  sorry

end purely_imaginary_complex_number_l310_31043


namespace bags_found_next_day_l310_31083

theorem bags_found_next_day 
  (initial_bags : ℕ) 
  (total_bags : ℕ) 
  (h : initial_bags ≤ total_bags) :
  total_bags - initial_bags = total_bags - initial_bags :=
by sorry

end bags_found_next_day_l310_31083


namespace fraction_simplification_l310_31034

theorem fraction_simplification (a : ℝ) (h : a ≠ 1) :
  (a + 1) / (1 - a) * (a^2 + a) / (a^2 + 2*a + 1) - 1 / (1 - a) = -1 := by
  sorry

end fraction_simplification_l310_31034


namespace prime_square_minus_cube_eq_one_l310_31064

theorem prime_square_minus_cube_eq_one (p q : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ p > 0 ∧ q > 0 → (p^2 - q^3 = 1 ↔ p = 3 ∧ q = 2) := by
  sorry

end prime_square_minus_cube_eq_one_l310_31064


namespace min_value_theorem_l310_31085

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 27) :
  3 * x + 2 * y + z ≥ 18 * Real.rpow 2 (1/3) ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 27 ∧
    3 * x₀ + 2 * y₀ + z₀ = 18 * Real.rpow 2 (1/3) :=
by sorry

end min_value_theorem_l310_31085


namespace symphony_orchestra_members_l310_31094

theorem symphony_orchestra_members : ∃! n : ℕ,
  200 < n ∧ n < 300 ∧
  n % 6 = 2 ∧
  n % 8 = 3 ∧
  n % 9 = 4 ∧
  n = 260 := by
sorry

end symphony_orchestra_members_l310_31094


namespace unique_a_value_l310_31077

-- Define the set A
def A (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}

-- State the theorem
theorem unique_a_value : ∃! a : ℝ, 1 ∈ A a ∧ a = 0 := by
  sorry

end unique_a_value_l310_31077


namespace union_A_B_union_complement_A_B_l310_31069

-- Define the universe set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | 2 ≤ x ∧ x < 5}

-- Define set B
def B : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 7}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x : ℝ | 2 ≤ x ∧ x ≤ 7} := by sorry

-- Theorem for (∁ₓA) ∪ (∁ₓB)
theorem union_complement_A_B : (Set.compl A) ∪ (Set.compl B) = {x : ℝ | x < 3 ∨ x ≥ 5} := by sorry

end union_A_B_union_complement_A_B_l310_31069


namespace complex_arg_range_l310_31006

theorem complex_arg_range (z : ℂ) (h : Complex.abs (2 * z + 1 / z) = 1) :
  ∃ k : ℤ, k ∈ ({0, 1} : Set ℤ) ∧
    k * Real.pi + Real.pi / 2 - (1 / 2) * Real.arccos (3 / 4) ≤ Complex.arg z ∧
    Complex.arg z ≤ k * Real.pi + Real.pi / 2 + (1 / 2) * Real.arccos (3 / 4) :=
by sorry

end complex_arg_range_l310_31006


namespace parallel_lines_distance_l310_31070

theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  r > 0 ∧ d > 0 ∧ 
  40 * r^2 = 10 * d^2 + 16000 ∧ 
  36 * r^2 = 81 * d^2 + 11664 →
  d = 6 := by
sorry

end parallel_lines_distance_l310_31070


namespace car_speed_problem_l310_31009

/-- Proves that car R's speed is 50 mph given the problem conditions -/
theorem car_speed_problem (distance : ℝ) (time_diff : ℝ) (speed_diff : ℝ) :
  distance = 600 →
  time_diff = 2 →
  speed_diff = 10 →
  (distance / (distance / 50 - time_diff) = 50 + speed_diff) →
  50 = distance / (distance / 50) :=
by sorry

end car_speed_problem_l310_31009


namespace brendan_grass_cutting_l310_31016

/-- Proves that Brendan can cut 84 yards of grass in a week with his new lawnmower -/
theorem brendan_grass_cutting (initial_capacity : ℕ) (increase_percentage : ℚ) (days_in_week : ℕ) :
  initial_capacity = 8 →
  increase_percentage = 1/2 →
  days_in_week = 7 →
  (initial_capacity + initial_capacity * increase_percentage) * days_in_week = 84 :=
by sorry

end brendan_grass_cutting_l310_31016


namespace books_written_proof_l310_31056

def total_books (zig_books flo_books : ℕ) : ℕ :=
  zig_books + flo_books

theorem books_written_proof (zig_books flo_books : ℕ) 
  (h1 : zig_books = 60) 
  (h2 : zig_books = 4 * flo_books) : 
  total_books zig_books flo_books = 75 := by
  sorry

end books_written_proof_l310_31056


namespace moving_circle_properties_l310_31035

/-- The trajectory of the center of a moving circle M that is externally tangent to O₁ and internally tangent to O₂ -/
def trajectory_equation (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 27 = 1

/-- The product of slopes of lines connecting M(x,y) with fixed points -/
def slope_product (x y : ℝ) : Prop :=
  y ≠ 0 → (y / (x + 6)) * (y / (x - 6)) = -3/4

/-- Circle O₁ equation -/
def circle_O₁ (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x + 5 = 0

/-- Circle O₂ equation -/
def circle_O₂ (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 91 = 0

theorem moving_circle_properties
  (x y : ℝ)
  (h₁ : ∃ (r : ℝ), r > 0 ∧ ∀ (x' y' : ℝ), (x' - x)^2 + (y' - y)^2 = r^2 →
    (circle_O₁ x' y' ∨ circle_O₂ x' y') ∧ ¬(circle_O₁ x' y' ∧ circle_O₂ x' y')) :
  trajectory_equation x y ∧ slope_product x y :=
sorry

end moving_circle_properties_l310_31035


namespace cylinder_volume_relation_l310_31096

theorem cylinder_volume_relation (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let volume_C := π * h^2 * r
  let volume_D := π * r^2 * h
  (volume_D = 3 * volume_C) → (volume_D = 9 * π * h^3) := by
sorry

end cylinder_volume_relation_l310_31096


namespace equation_solutions_l310_31081

theorem equation_solutions : 
  let f (x : ℝ) := (x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 3) * (x - 5) * (x - 1)
  let g (x : ℝ) := (x - 3) * (x - 7) * (x - 3)
  let S := {x : ℝ | x ≠ 3 ∧ x ≠ 7 ∧ f x / g x = 1}
  S = {3 + Real.sqrt 3, 3 - Real.sqrt 3, 3 + Real.sqrt 5, 3 - Real.sqrt 5} := by
  sorry

end equation_solutions_l310_31081


namespace your_bill_before_tax_friend_order_equation_l310_31029

/-- The cost of a taco in dollars -/
def taco_cost : ℝ := sorry

/-- The cost of an enchilada in dollars -/
def enchilada_cost : ℝ := 2

/-- The cost of 3 tacos and 5 enchiladas in dollars -/
def friend_order_cost : ℝ := 12.70

theorem your_bill_before_tax :
  2 * taco_cost + 3 * enchilada_cost = 7.80 :=
by
  sorry

/-- The friend's order cost equation -/
theorem friend_order_equation :
  3 * taco_cost + 5 * enchilada_cost = friend_order_cost :=
by
  sorry

end your_bill_before_tax_friend_order_equation_l310_31029


namespace no_nonperiodic_function_satisfies_equation_l310_31080

theorem no_nonperiodic_function_satisfies_equation :
  ¬∃ f : ℝ → ℝ, (∀ x : ℝ, f (x + 1) = f x * (f x + 1)) ∧ (¬∃ p : ℝ, p ≠ 0 ∧ ∀ x : ℝ, f (x + p) = f x) :=
by sorry

end no_nonperiodic_function_satisfies_equation_l310_31080


namespace grasshopper_jumps_l310_31010

-- Define a circle
def Circle := ℝ × ℝ → Prop

-- Define a point
def Point := ℝ × ℝ

-- Define a segment
def Segment := Point × Point

-- Define the property of being inside a circle
def InsideCircle (c : Circle) (p : Point) : Prop := sorry

-- Define the property of being on the boundary of a circle
def OnCircleBoundary (c : Circle) (p : Point) : Prop := sorry

-- Define the property of two segments not intersecting
def DoNotIntersect (s1 s2 : Segment) : Prop := sorry

-- Define the property of a point being reachable from another point
def Reachable (c : Circle) (points_inside : List Point) (points_boundary : List Point) (p q : Point) : Prop := sorry

theorem grasshopper_jumps 
  (c : Circle) 
  (n : ℕ) 
  (points_inside : List Point) 
  (points_boundary : List Point) 
  (h1 : points_inside.length = n) 
  (h2 : points_boundary.length = n)
  (h3 : ∀ p ∈ points_inside, InsideCircle c p)
  (h4 : ∀ p ∈ points_boundary, OnCircleBoundary c p)
  (h5 : ∀ (i j : Fin n), i ≠ j → 
    DoNotIntersect (points_inside[i], points_boundary[i]) (points_inside[j], points_boundary[j]))
  : ∀ (p q : Point), p ∈ points_inside → q ∈ points_inside → Reachable c points_inside points_boundary p q :=
sorry

end grasshopper_jumps_l310_31010


namespace fair_special_savings_l310_31044

/-- Calculates the percentage saved when buying three pairs of sandals under the "fair special" --/
theorem fair_special_savings : 
  let regular_price : ℝ := 60
  let second_pair_discount : ℝ := 0.4
  let third_pair_discount : ℝ := 0.25
  let total_regular_price : ℝ := 3 * regular_price
  let discounted_price : ℝ := regular_price + 
                              (1 - second_pair_discount) * regular_price + 
                              (1 - third_pair_discount) * regular_price
  let savings : ℝ := total_regular_price - discounted_price
  let percentage_saved : ℝ := (savings / total_regular_price) * 100
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |percentage_saved - 22| < ε :=
by
  sorry

end fair_special_savings_l310_31044


namespace quadratic_solution_sum_l310_31088

theorem quadratic_solution_sum (c d : ℝ) : 
  (5 * (c + d * I) ^ 2 + 4 * (c + d * I) + 20 = 0) ∧ 
  (5 * (c - d * I) ^ 2 + 4 * (c - d * I) + 20 = 0) →
  c + d ^ 2 = 86 / 25 := by
sorry

end quadratic_solution_sum_l310_31088


namespace tickets_per_candy_l310_31058

def whack_a_mole_tickets : ℕ := 2
def skee_ball_tickets : ℕ := 13
def candies_bought : ℕ := 5

def total_tickets : ℕ := whack_a_mole_tickets + skee_ball_tickets

theorem tickets_per_candy : total_tickets / candies_bought = 3 := by
  sorry

end tickets_per_candy_l310_31058


namespace apple_juice_distribution_l310_31024

/-- Given a total amount of apple juice and the difference between two people's consumption,
    calculate the amount consumed by the person who drinks more. -/
theorem apple_juice_distribution (total : ℝ) (difference : ℝ) (kyu_yeon_amount : ℝ) : 
  total = 12.4 ∧ difference = 2.6 → kyu_yeon_amount = 7.5 := by
  sorry

#check apple_juice_distribution

end apple_juice_distribution_l310_31024


namespace binomial_coefficient_equality_l310_31093

theorem binomial_coefficient_equality (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
  sorry

end binomial_coefficient_equality_l310_31093


namespace rope_cutting_l310_31054

theorem rope_cutting (total_length : ℕ) (long_piece_length : ℕ) (num_short_pieces : ℕ) 
  (h1 : total_length = 27)
  (h2 : long_piece_length = 4)
  (h3 : num_short_pieces = 3) :
  ∃ (num_long_pieces : ℕ) (short_piece_length : ℕ),
    num_long_pieces * long_piece_length + num_short_pieces * short_piece_length = total_length ∧
    num_long_pieces = total_length / long_piece_length ∧
    short_piece_length = 1 :=
by
  sorry

end rope_cutting_l310_31054


namespace internal_borders_length_l310_31089

/-- Represents a square garden bed with integer side length -/
structure SquareBed where
  side : ℕ

/-- Represents a rectangular garden divided into square beds -/
structure Garden where
  width : ℕ
  height : ℕ
  beds : List SquareBed

/-- Calculates the total area of the garden -/
def Garden.area (g : Garden) : ℕ := g.width * g.height

/-- Calculates the total area covered by the beds -/
def Garden.bedArea (g : Garden) : ℕ := g.beds.map (fun b => b.side * b.side) |>.sum

/-- Calculates the perimeter of the garden -/
def Garden.perimeter (g : Garden) : ℕ := 2 * (g.width + g.height)

/-- Calculates the sum of perimeters of all beds -/
def Garden.bedPerimeters (g : Garden) : ℕ := g.beds.map (fun b => 4 * b.side) |>.sum

/-- Theorem stating the length of internal borders in a specific garden configuration -/
theorem internal_borders_length (g : Garden) : 
  g.width = 6 ∧ 
  g.height = 7 ∧ 
  g.beds.length = 5 ∧ 
  g.area = g.bedArea →
  (g.bedPerimeters - g.perimeter) / 2 = 15 := by
  sorry


end internal_borders_length_l310_31089


namespace range_of_f_l310_31004

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the domain
def domain : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | 0 ≤ y ∧ y ≤ 4} := by sorry

end range_of_f_l310_31004


namespace mickey_horses_per_week_l310_31068

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of horses Minnie mounts per day -/
def minnie_horses_per_day : ℕ := days_in_week + 3

/-- The number of horses Mickey mounts per day -/
def mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6

/-- Theorem: Mickey mounts 98 horses per week -/
theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end mickey_horses_per_week_l310_31068


namespace arithmetic_geometric_sequence_l310_31032

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- a_1, a_2, and a_4 form a geometric sequence -/
def geometric_subseq (a : ℕ → ℝ) : Prop :=
  a 2 ^ 2 = a 1 * a 4

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) 
  (h1 : arithmetic_seq a) (h2 : geometric_subseq a) : a 1 = 2 := by
  sorry

end arithmetic_geometric_sequence_l310_31032


namespace orange_count_theorem_l310_31030

/-- The number of oranges initially in the box -/
def initial_oranges : ℝ := 55.0

/-- The number of oranges Susan adds to the box -/
def added_oranges : ℝ := 35.0

/-- The total number of oranges in the box after Susan adds more -/
def total_oranges : ℝ := 90.0

/-- Theorem stating that the initial number of oranges plus the added oranges equals the total oranges -/
theorem orange_count_theorem : initial_oranges + added_oranges = total_oranges := by
  sorry

end orange_count_theorem_l310_31030


namespace max_value_x_plus_2y_l310_31042

theorem max_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2*x^2 + 8*y^2 + x*y = 2) : x + 2*y ≤ 4/3 := by
  sorry

end max_value_x_plus_2y_l310_31042


namespace sum_of_squares_of_reciprocals_l310_31018

theorem sum_of_squares_of_reciprocals (x y : ℝ) 
  (sum_eq : x + y = 12) 
  (product_eq : x * y = 32) : 
  (1 / x)^2 + (1 / y)^2 = 5 / 64 := by
  sorry

end sum_of_squares_of_reciprocals_l310_31018


namespace craft_store_optimal_solution_l310_31021

/-- Represents the craft store problem -/
structure CraftStore where
  profit_per_item : ℝ
  cost_50_items : ℝ
  revenue_40_items : ℝ
  initial_daily_sales : ℕ
  sales_increase_per_yuan : ℕ

/-- Theorem stating the optimal solution for the craft store problem -/
theorem craft_store_optimal_solution (store : CraftStore) 
  (h1 : store.profit_per_item = 45)
  (h2 : store.cost_50_items = store.revenue_40_items)
  (h3 : store.initial_daily_sales = 100)
  (h4 : store.sales_increase_per_yuan = 4) :
  ∃ (cost_price marked_price optimal_reduction max_profit : ℝ),
    cost_price = 180 ∧
    marked_price = 225 ∧
    optimal_reduction = 10 ∧
    max_profit = 4900 := by
  sorry

end craft_store_optimal_solution_l310_31021


namespace complex_fraction_equality_l310_31087

theorem complex_fraction_equality : (2 : ℂ) / (1 + Complex.I) = 1 - Complex.I := by
  sorry

end complex_fraction_equality_l310_31087


namespace parabola_point_coordinates_l310_31067

theorem parabola_point_coordinates :
  ∀ (x y : ℝ),
  y^2 = 4*x →                             -- Point P(x, y) lies on the parabola y^2 = 4x
  (x - 1)^2 + y^2 = 100 →                 -- Distance from P to focus F(1, 0) is 10
  x = 9 ∧ (y = 6 ∨ y = -6) :=             -- Conclusion: x = 9 and y = ±6
by
  sorry


end parabola_point_coordinates_l310_31067


namespace average_book_width_l310_31074

theorem average_book_width :
  let book_widths : List ℚ := [5, 3/4, 3/2, 3, 29/4, 12]
  (book_widths.sum / book_widths.length : ℚ) = 59/12 := by
sorry

end average_book_width_l310_31074


namespace interest_difference_l310_31076

/-- Calculate the difference between compound interest and simple interest -/
theorem interest_difference (principal : ℝ) (rate : ℝ) (time : ℝ) (compounding_frequency : ℕ) :
  principal = 1200 →
  rate = 0.1 →
  time = 1 →
  compounding_frequency = 2 →
  let simple_interest := principal * rate * time
  let compound_interest := principal * ((1 + rate / compounding_frequency) ^ (compounding_frequency * time) - 1)
  compound_interest - simple_interest = 3 := by
  sorry

end interest_difference_l310_31076


namespace function_properties_l310_31066

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x
noncomputable def g (x : ℝ) : ℝ := 2 / x

-- Define the combined function h
noncomputable def h (x : ℝ) : ℝ := f x + g x

-- Theorem statement
theorem function_properties :
  (∃ k : ℝ, ∀ x : ℝ, x > 0 → f x = k * x) ∧
  (∃ c : ℝ, ∀ x : ℝ, x > 0 → g x = c / x) ∧
  f 1 = 1 ∧
  g 1 = 2 →
  (∀ x : ℝ, x > 0 → f x = x ∧ g x = 2 / x) ∧
  (∀ x : ℝ, x ≠ 0 → h x = -h (-x)) ∧
  (∀ x : ℝ, 0 < x → x ≤ Real.sqrt 2 → h x ≥ 2 * Real.sqrt 2) ∧
  h (Real.sqrt 2) = 2 * Real.sqrt 2 :=
by sorry


end function_properties_l310_31066


namespace damages_cost_l310_31045

def tire_cost_1 : ℕ := 230
def tire_cost_2 : ℕ := 250
def tire_cost_3 : ℕ := 280
def window_cost_1 : ℕ := 700
def window_cost_2 : ℕ := 800
def window_cost_3 : ℕ := 900

def total_damages : ℕ := 
  2 * tire_cost_1 + 2 * tire_cost_2 + 2 * tire_cost_3 +
  window_cost_1 + window_cost_2 + window_cost_3

theorem damages_cost : total_damages = 3920 := by
  sorry

end damages_cost_l310_31045


namespace defective_percentage_is_3_6_percent_l310_31071

/-- Represents the percentage of products manufactured by each machine -/
structure MachineProduction where
  m1 : ℝ
  m2 : ℝ
  m3 : ℝ

/-- Represents the percentage of defective products for each machine -/
structure DefectivePercentage where
  m1 : ℝ
  m2 : ℝ
  m3 : ℝ

/-- Calculates the percentage of defective products in the stockpile -/
def calculateDefectivePercentage (prod : MachineProduction) (defect : DefectivePercentage) : ℝ :=
  prod.m1 * defect.m1 + prod.m2 * defect.m2 + prod.m3 * defect.m3

theorem defective_percentage_is_3_6_percent 
  (prod : MachineProduction)
  (defect : DefectivePercentage)
  (h1 : prod.m1 = 0.4)
  (h2 : prod.m2 = 0.3)
  (h3 : prod.m3 = 0.3)
  (h4 : defect.m1 = 0.03)
  (h5 : defect.m2 = 0.01)
  (h6 : defect.m3 = 0.07) :
  calculateDefectivePercentage prod defect = 0.036 := by
  sorry

#eval calculateDefectivePercentage 
  { m1 := 0.4, m2 := 0.3, m3 := 0.3 } 
  { m1 := 0.03, m2 := 0.01, m3 := 0.07 }

end defective_percentage_is_3_6_percent_l310_31071


namespace problem_solution_l310_31005

theorem problem_solution : ∃ x : ℝ, (0.2 * 30 = 0.25 * x + 2) ∧ x = 16 := by
  sorry

end problem_solution_l310_31005


namespace sum_of_squared_sums_l310_31023

theorem sum_of_squared_sums (a b c : ℝ) : 
  (a^3 - 15*a^2 + 17*a - 8 = 0) →
  (b^3 - 15*b^2 + 17*b - 8 = 0) →
  (c^3 - 15*c^2 + 17*c - 8 = 0) →
  (a+b)^2 + (b+c)^2 + (c+a)^2 = 416 := by
sorry

end sum_of_squared_sums_l310_31023


namespace quadratic_inequality_range_l310_31028

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 - 5*x + (5/4)*a > 0) ↔ a ≤ 5 :=
by sorry

end quadratic_inequality_range_l310_31028


namespace quadratic_function_bound_l310_31072

theorem quadratic_function_bound (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x^2 + b * x + c| ≤ 1) →
  (a + b) * c ≤ 1/4 := by
  sorry

end quadratic_function_bound_l310_31072


namespace sum_remainder_l310_31011

theorem sum_remainder (a b c d e : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9)
  (he : e % 13 = 11)
  (hsquare : ∃ k : ℕ, c = k * k) :
  (a + b + c + d + e) % 13 = 9 := by
  sorry

end sum_remainder_l310_31011


namespace smallest_interesting_number_l310_31062

def is_interesting (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 2 * n = a^2 ∧ 15 * n = b^3

theorem smallest_interesting_number : 
  (is_interesting 1800) ∧ (∀ m : ℕ, m < 1800 → ¬(is_interesting m)) :=
by sorry

end smallest_interesting_number_l310_31062


namespace sum_fourth_power_ge_two_min_sum_cube_and_reciprocal_cube_min_sum_cube_and_reciprocal_cube_equality_l310_31092

-- Part I
theorem sum_fourth_power_ge_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  a^4 + b^4 ≥ 2 := by sorry

-- Part II
theorem min_sum_cube_and_reciprocal_cube (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 + b^3 + c^3 + (1/a + 1/b + 1/c)^3 ≥ 18 := by sorry

theorem min_sum_cube_and_reciprocal_cube_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 + b^3 + c^3 + (1/a + 1/b + 1/c)^3 = 18 ↔ a = (3 : ℝ)^(1/3) ∧ b = (3 : ℝ)^(1/3) ∧ c = (3 : ℝ)^(1/3) := by sorry

end sum_fourth_power_ge_two_min_sum_cube_and_reciprocal_cube_min_sum_cube_and_reciprocal_cube_equality_l310_31092


namespace inequality_proof_l310_31061

theorem inequality_proof (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 0 < z ∧ z < 1) : 
  x / (1 - x) + y / (1 - y) + z / (1 - z) ≥ 3 * (x * y * z)^(1/3) / (1 - (x * y * z)^(1/3)) :=
by sorry

end inequality_proof_l310_31061


namespace quadratic_other_x_intercept_l310_31075

/-- Given a quadratic function f(x) = ax^2 + bx + c with vertex (5, 10) and
    one x-intercept at (1, 0), the x-coordinate of the other x-intercept is 9. -/
theorem quadratic_other_x_intercept
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h_quad : ∀ x, f x = a * x^2 + b * x + c)
  (h_vertex : f 5 = 10 ∧ ∀ x, f x ≤ f 5)
  (h_intercept : f 1 = 0) :
  ∃ x, x ≠ 1 ∧ f x = 0 ∧ x = 9 :=
sorry

end quadratic_other_x_intercept_l310_31075


namespace sum_inequality_l310_31051

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a * b) / (a + b) + (b * c) / (b + c) + (c * a) / (c + a) + 
    (1 / 2) * ((a * b) / c + (b * c) / a + (c * a) / b) := by
  sorry

end sum_inequality_l310_31051


namespace probability_is_one_twelfth_l310_31090

/-- The number of vertices in a regular decagon -/
def decagon_vertices : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The total number of possible triangles formed by choosing 3 vertices from a decagon -/
def total_triangles : ℕ := Nat.choose decagon_vertices triangle_vertices

/-- The number of triangles with at least two sides that are also sides of the decagon -/
def favorable_triangles : ℕ := decagon_vertices

/-- The probability of forming a triangle with at least two sides that are also sides of the decagon -/
def probability : ℚ := favorable_triangles / total_triangles

/-- Theorem stating the probability is 1/12 -/
theorem probability_is_one_twelfth : probability = 1 / 12 := by sorry

end probability_is_one_twelfth_l310_31090


namespace cutting_process_ends_at_1998_l310_31078

/-- Represents a shape with points on its boundary -/
structure Shape :=
  (points : ℕ)

/-- Represents the state of the cutting process -/
structure CuttingState :=
  (shape : Shape)
  (cuts : ℕ)

/-- Checks if a shape is a polygon -/
def is_polygon (s : Shape) : Prop :=
  s.points ≤ 3

/-- Checks if a shape can become a polygon with further cutting -/
def can_become_polygon (s : Shape) : Prop :=
  s.points > 3

/-- Performs a cut on the shape -/
def cut (state : CuttingState) : CuttingState :=
  { shape := { points := state.shape.points - 1 },
    cuts := state.cuts + 1 }

/-- The main theorem to be proved -/
theorem cutting_process_ends_at_1998 :
  ∀ (initial_state : CuttingState),
    initial_state.shape.points = 1001 →
    ∀ (n : ℕ),
      n ≤ 1998 →
      ¬(is_polygon (cut^[n] initial_state).shape) ∧
      can_become_polygon (cut^[n] initial_state).shape →
      ¬(∃ (m : ℕ),
        m > 1998 ∧
        ¬(is_polygon (cut^[m] initial_state).shape) ∧
        can_become_polygon (cut^[m] initial_state).shape) :=
sorry

end cutting_process_ends_at_1998_l310_31078


namespace hole_filling_proof_l310_31003

/-- The amount of water initially in the hole -/
def initial_water : ℕ := 676

/-- The additional amount of water needed to fill the hole -/
def additional_water : ℕ := 147

/-- The total amount of water needed to fill the hole -/
def total_water : ℕ := initial_water + additional_water

theorem hole_filling_proof : total_water = 823 := by
  sorry

end hole_filling_proof_l310_31003


namespace lines_perpendicular_to_plane_are_parallel_l310_31002

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end lines_perpendicular_to_plane_are_parallel_l310_31002


namespace problem_1_l310_31052

theorem problem_1 : Real.sqrt 12 + 3 - 2^2 + |1 - Real.sqrt 3| = 3 * Real.sqrt 3 - 2 := by
  sorry

end problem_1_l310_31052


namespace train_length_calculation_l310_31022

/-- Calculates the length of a train given the speeds of a jogger and the train,
    the initial distance between them, and the time it takes for the train to pass the jogger. -/
theorem train_length_calculation (jogger_speed train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (5 / 18) →
  train_speed = 45 * (5 / 18) →
  initial_distance = 190 →
  passing_time = 31 →
  (train_speed - jogger_speed) * passing_time - initial_distance = 120 :=
by sorry

end train_length_calculation_l310_31022


namespace profit_percentage_l310_31019

theorem profit_percentage (cost_price selling_price : ℚ) : 
  cost_price = 32 → 
  selling_price = 56 → 
  (selling_price - cost_price) / cost_price * 100 = 75 := by
  sorry

end profit_percentage_l310_31019


namespace cubic_equation_sum_l310_31012

theorem cubic_equation_sum (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 2020*a^2 + 1010 = 0 →
  b^3 - 2020*b^2 + 1010 = 0 →
  c^3 - 2020*c^2 + 1010 = 0 →
  1/(a*b) + 1/(b*c) + 1/(a*c) = -2 := by
sorry

end cubic_equation_sum_l310_31012


namespace prime_squared_plus_two_prime_l310_31017

theorem prime_squared_plus_two_prime (p : ℕ) : 
  Prime p → Prime (p^2 + 2) → p = 3 := by sorry

end prime_squared_plus_two_prime_l310_31017


namespace sample_size_calculation_l310_31027

/-- Given a factory producing three product models A, B, and C with quantities in the ratio 3:4:7,
    prove that a sample containing 15 units of product A has a total size of 70. -/
theorem sample_size_calculation (ratio_A ratio_B ratio_C : ℕ) (sample_A : ℕ) (n : ℕ) : 
  ratio_A = 3 → ratio_B = 4 → ratio_C = 7 → sample_A = 15 →
  n = (ratio_A + ratio_B + ratio_C) * sample_A / ratio_A → n = 70 := by
  sorry

#check sample_size_calculation

end sample_size_calculation_l310_31027


namespace inequality_solution_range_l310_31099

/-- Given that the inequality |x+a|+|x-1|+a>2009 (where a is a constant) has a non-empty set of solutions, 
    the range of values for a is (-∞, 1004) -/
theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + a| + |x - 1| + a > 2009) → 
  a ∈ Set.Iio 1004 := by
sorry

end inequality_solution_range_l310_31099


namespace value_difference_l310_31082

theorem value_difference (n : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) :
  n = 80 ∧ increase_percent = 0.125 ∧ decrease_percent = 0.25 →
  n * (1 + increase_percent) - n * (1 - decrease_percent) = 30 :=
by sorry

end value_difference_l310_31082


namespace fraction_calculation_l310_31001

theorem fraction_calculation : (1/2 - 1/3) / (3/4 + 1/8) = 4/21 := by
  sorry

end fraction_calculation_l310_31001


namespace light_flash_duration_l310_31038

/-- Proves that if a light flashes every 15 seconds and flashes 240 times, the time taken is exactly one hour. -/
theorem light_flash_duration (flash_interval : ℕ) (total_flashes : ℕ) (seconds_per_hour : ℕ) : 
  flash_interval = 15 → 
  total_flashes = 240 → 
  seconds_per_hour = 3600 → 
  flash_interval * total_flashes = seconds_per_hour :=
by
  sorry

#check light_flash_duration

end light_flash_duration_l310_31038


namespace bobs_raise_l310_31015

/-- Calculates the raise per hour given the following conditions:
  * Bob works 40 hours per week
  * His housing benefit is reduced by $60 per month
  * He earns $5 more per week after the changes
-/
theorem bobs_raise (hours_per_week : ℕ) (benefit_reduction_per_month : ℚ) (extra_earnings_per_week : ℚ) :
  hours_per_week = 40 →
  benefit_reduction_per_month = 60 →
  extra_earnings_per_week = 5 →
  ∃ (raise_per_hour : ℚ), 
    raise_per_hour * hours_per_week - (benefit_reduction_per_month / 4) + extra_earnings_per_week = 0 ∧
    raise_per_hour = 1/4 := by
  sorry

end bobs_raise_l310_31015


namespace circle_intersection_equation_l310_31014

noncomputable def circle_equation (t : ℝ) (x y : ℝ) : Prop :=
  (x - t)^2 + (y - 2/t)^2 = t^2 + (2/t)^2

theorem circle_intersection_equation :
  ∀ t : ℝ,
  t ≠ 0 →
  circle_equation t 0 0 →
  (∃ a : ℝ, a ≠ 0 ∧ circle_equation t a 0) →
  (∃ b : ℝ, b ≠ 0 ∧ circle_equation t 0 b) →
  (∀ x y : ℝ, 2*x + y = 4 → circle_equation t x y → 
    ∃ m n : ℝ, circle_equation t m n ∧ 2*m + n = 4 ∧ m^2 + n^2 = x^2 + y^2) →
  circle_equation 2 x y ∧ (x - 2)^2 + (y - 1)^2 = 5 :=
sorry

end circle_intersection_equation_l310_31014


namespace brother_difference_is_two_l310_31059

/-- The number of Aaron's brothers -/
def aaron_brothers : ℕ := 4

/-- The number of Bennett's brothers -/
def bennett_brothers : ℕ := 6

/-- The difference between twice the number of Aaron's brothers and the number of Bennett's brothers -/
def brother_difference : ℕ := 2 * aaron_brothers - bennett_brothers

/-- Theorem stating that the difference between twice the number of Aaron's brothers
    and the number of Bennett's brothers is 2 -/
theorem brother_difference_is_two : brother_difference = 2 := by
  sorry

end brother_difference_is_two_l310_31059


namespace sqrt_a_minus_4_real_l310_31057

theorem sqrt_a_minus_4_real (a : ℝ) : (∃ x : ℝ, x^2 = a - 4) ↔ a ≥ 4 := by sorry

end sqrt_a_minus_4_real_l310_31057


namespace five_dice_same_number_probability_l310_31060

theorem five_dice_same_number_probability : 
  let number_of_dice : ℕ := 5
  let faces_per_die : ℕ := 6
  let total_outcomes : ℕ := faces_per_die ^ number_of_dice
  let favorable_outcomes : ℕ := faces_per_die
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 1296 := by
sorry

end five_dice_same_number_probability_l310_31060


namespace range_of_m_l310_31065

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of f being increasing on [-2, 2]
def is_increasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y ∧ y ≤ 2 → f x < f y

-- Define the theorem
theorem range_of_m (h1 : is_increasing_on_interval f) (h2 : ∀ m, f (1 - m) < f m) :
  ∀ m, m ∈ Set.Ioo (1/2) 2 ↔ -2 ≤ 1 - m ∧ 1 - m < m ∧ m ≤ 2 :=
sorry

end range_of_m_l310_31065


namespace robotics_club_enrollment_l310_31013

theorem robotics_club_enrollment (total : ℕ) (cs : ℕ) (elec : ℕ) (both : ℕ)
  (h1 : total = 80)
  (h2 : cs = 50)
  (h3 : elec = 35)
  (h4 : both = 25) :
  total - (cs + elec - both) = 20 :=
by sorry

end robotics_club_enrollment_l310_31013


namespace solution_is_816_div_5_l310_31040

/-- The function g(y) = ∛(30y + ∛(30y + 17)) is increasing --/
axiom g_increasing (y : ℝ) : 
  Monotone (fun y => Real.rpow (30 * y + Real.rpow (30 * y + 17) (1/3 : ℝ)) (1/3 : ℝ))

/-- The equation ∛(30y + ∛(30y + 17)) = 17 has a unique solution --/
axiom unique_solution : ∃! y : ℝ, Real.rpow (30 * y + Real.rpow (30 * y + 17) (1/3 : ℝ)) (1/3 : ℝ) = 17

theorem solution_is_816_div_5 : 
  ∃! y : ℝ, Real.rpow (30 * y + Real.rpow (30 * y + 17) (1/3 : ℝ)) (1/3 : ℝ) = 17 ∧ y = 816 / 5 := by
  sorry

end solution_is_816_div_5_l310_31040


namespace square_diagonals_equal_l310_31031

-- Define the necessary structures
structure Rectangle where
  diagonals_equal : Bool

structure Square extends Rectangle

-- Define the theorem
theorem square_diagonals_equal (h1 : ∀ r : Rectangle, r.diagonals_equal) 
  (h2 : Square → Rectangle) : 
  ∀ s : Square, (h2 s).diagonals_equal :=
by
  sorry


end square_diagonals_equal_l310_31031


namespace unique_b_value_l310_31026

/-- The configuration of a circle and parabola with specific intersection properties -/
structure CircleParabolaConfig where
  b : ℝ
  circle_center : ℝ × ℝ
  parabola : ℝ → ℝ
  line : ℝ → ℝ
  intersect_origin : Bool
  intersect_line : Bool

/-- The theorem stating the unique value of b for the given configuration -/
theorem unique_b_value (config : CircleParabolaConfig) : 
  config.parabola = (λ x => (12/5) * x^2) →
  config.line = (λ x => (12/5) * x + config.b) →
  config.circle_center.2 = config.b →
  config.intersect_origin = true →
  config.intersect_line = true →
  config.b = 169/60 := by
  sorry

#check unique_b_value

end unique_b_value_l310_31026


namespace decreasing_quadratic_implies_a_geq_3_l310_31073

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 6

theorem decreasing_quadratic_implies_a_geq_3 (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 3 → f a x₁ > f a x₂) →
  a ≥ 3 :=
by sorry

end decreasing_quadratic_implies_a_geq_3_l310_31073


namespace refrigerator_savings_l310_31000

/-- Calculates the savings from switching to a more energy-efficient refrigerator -/
theorem refrigerator_savings 
  (old_cost : ℝ) 
  (new_cost : ℝ) 
  (days : ℕ) 
  (h1 : old_cost = 0.85) 
  (h2 : new_cost = 0.45) 
  (h3 : days = 30) : 
  (old_cost * days) - (new_cost * days) = 12 :=
by sorry

end refrigerator_savings_l310_31000


namespace smallest_prime_average_l310_31063

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a list contains five different prime numbers
def isFiveDifferentPrimes (list : List ℕ) : Prop :=
  list.length = 5 ∧ list.Nodup ∧ ∀ n ∈ list, isPrime n

-- Define a function to calculate the average of a list of numbers
def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem smallest_prime_average :
  ∀ list : List ℕ, isFiveDifferentPrimes list → (average list).isInt → average list ≥ 6 :=
sorry

end smallest_prime_average_l310_31063


namespace sum_of_roots_eq_eight_l310_31007

theorem sum_of_roots_eq_eight : 
  let f : ℝ → ℝ := λ x => (x - 4)^2 - 16
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 8 :=
by
  sorry

end sum_of_roots_eq_eight_l310_31007


namespace grid_paths_6_4_l310_31097

/-- The number of paths on a grid from (0,0) to (m,n) using exactly m+n steps -/
def grid_paths (m n : ℕ) : ℕ := Nat.choose (m + n) m

theorem grid_paths_6_4 : grid_paths 6 4 = 210 := by
  sorry

end grid_paths_6_4_l310_31097


namespace quadratic_has_two_real_roots_roots_difference_three_l310_31049

/-- The quadratic equation x^2 - (m-1)x + (m-2) = 0 -/
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - (m-1)*x + (m-2)

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := (m-1)^2 - 4*(m-2)

theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 :=
sorry

theorem roots_difference_three (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ |x₁ - x₂| = 3) →
  m = 0 ∨ m = 6 :=
sorry

end quadratic_has_two_real_roots_roots_difference_three_l310_31049


namespace inequality_proof_l310_31037

theorem inequality_proof (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 6)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 12) :
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧ 
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 :=
by sorry

end inequality_proof_l310_31037


namespace tracy_has_two_dogs_l310_31020

/-- The number of dogs Tracy has -/
def num_dogs : ℕ :=
  let cups_per_meal : ℚ := 3/2  -- 1.5 cups per meal
  let meals_per_day : ℕ := 3
  let total_pounds : ℕ := 4
  let cups_per_pound : ℚ := 9/4  -- 2.25 cups per pound

  let total_cups : ℚ := total_pounds * cups_per_pound
  let cups_per_dog_per_day : ℚ := cups_per_meal * meals_per_day

  (total_cups / cups_per_dog_per_day).num.toNat

/-- Theorem stating that Tracy has 2 dogs -/
theorem tracy_has_two_dogs : num_dogs = 2 := by
  sorry

end tracy_has_two_dogs_l310_31020


namespace anna_baking_trays_l310_31050

/-- The number of cupcakes per tray -/
def cupcakes_per_tray : ℕ := 20

/-- The price of each cupcake in dollars -/
def cupcake_price : ℚ := 2

/-- The fraction of cupcakes sold -/
def fraction_sold : ℚ := 3/5

/-- The total earnings in dollars -/
def total_earnings : ℚ := 96

/-- The number of baking trays Anna used -/
def num_trays : ℕ := 4

theorem anna_baking_trays :
  (cupcakes_per_tray : ℚ) * num_trays * fraction_sold * cupcake_price = total_earnings := by
  sorry

end anna_baking_trays_l310_31050


namespace final_student_count_l310_31036

/-- Represents the arrangement of students in the photo. -/
structure StudentArrangement where
  rows : ℕ
  columns : ℕ

/-- The initial arrangement of students before any changes. -/
def initial_arrangement : StudentArrangement := { rows := 0, columns := 0 }

/-- The arrangement after moving one student from each row. -/
def first_adjustment (a : StudentArrangement) : StudentArrangement :=
  { rows := a.rows + 1, columns := a.columns - 1 }

/-- The arrangement after moving a second student from each row. -/
def second_adjustment (a : StudentArrangement) : StudentArrangement :=
  { rows := a.rows + 1, columns := a.columns - 1 }

/-- Calculates the total number of students in the arrangement. -/
def total_students (a : StudentArrangement) : ℕ := a.rows * a.columns

/-- The theorem stating the final number of students in the photo. -/
theorem final_student_count :
  ∃ (a : StudentArrangement),
    (first_adjustment a).columns = (first_adjustment a).rows + 4 ∧
    (second_adjustment (first_adjustment a)).columns = (second_adjustment (first_adjustment a)).rows ∧
    total_students (second_adjustment (first_adjustment a)) = 24 :=
  sorry

end final_student_count_l310_31036


namespace shelves_needed_l310_31053

theorem shelves_needed (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 12) (h2 : books_per_shelf = 4) :
  total_books / books_per_shelf = 3 := by
  sorry

end shelves_needed_l310_31053


namespace base_conversion_2025_to_octal_l310_31025

theorem base_conversion_2025_to_octal :
  (2025 : ℕ) = (3 * 8^3 + 7 * 8^2 + 5 * 8^1 + 1 * 8^0 : ℕ) :=
by sorry

end base_conversion_2025_to_octal_l310_31025


namespace binomial_coefficient_n_minus_two_l310_31039

theorem binomial_coefficient_n_minus_two (n : ℕ) (h : n > 3) :
  Nat.choose n (n - 2) = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_n_minus_two_l310_31039


namespace new_average_after_multipliers_l310_31086

theorem new_average_after_multipliers (original_list : List ℝ) 
  (h1 : original_list.length = 7)
  (h2 : original_list.sum / original_list.length = 20)
  (multipliers : List ℝ := [2, 3, 4, 5, 6, 7, 8]) :
  (List.zipWith (· * ·) original_list multipliers).sum / original_list.length = 100 := by
  sorry

end new_average_after_multipliers_l310_31086


namespace factorial_sum_mod_20_l310_31055

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_sum_mod_20 : (factorial 1 + factorial 2 + factorial 3 + factorial 4 + factorial 5 + factorial 6) % 20 = 13 := by
  sorry

end factorial_sum_mod_20_l310_31055
