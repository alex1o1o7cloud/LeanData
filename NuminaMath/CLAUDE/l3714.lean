import Mathlib

namespace new_oranges_added_l3714_371468

theorem new_oranges_added (initial : ℕ) (thrown_away : ℕ) (final : ℕ) : 
  initial = 50 → thrown_away = 40 → final = 34 → 
  final - (initial - thrown_away) = 24 := by
  sorry

end new_oranges_added_l3714_371468


namespace not_sufficient_not_necessary_l3714_371458

/-- The statement "at least one of x and y is greater than 1" is neither a sufficient nor a necessary condition for x^2 + y^2 > 2 -/
theorem not_sufficient_not_necessary (x y : ℝ) : 
  ¬(((x > 1 ∨ y > 1) → x^2 + y^2 > 2) ∧ (x^2 + y^2 > 2 → (x > 1 ∨ y > 1))) := by
  sorry

end not_sufficient_not_necessary_l3714_371458


namespace candy_distribution_l3714_371481

theorem candy_distribution (total_candy : ℕ) (num_friends : ℕ) (candy_per_friend : ℕ) 
  (h1 : total_candy = 420)
  (h2 : num_friends = 35)
  (h3 : total_candy = num_friends * candy_per_friend) :
  candy_per_friend = 12 := by
  sorry

end candy_distribution_l3714_371481


namespace four_mutually_tangent_circles_exist_l3714_371453

-- Define a circle with a center point and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of two circles being externally tangent
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Theorem statement
theorem four_mutually_tangent_circles_exist : 
  ∃ (c1 c2 c3 c4 : Circle),
    are_externally_tangent c1 c2 ∧
    are_externally_tangent c1 c3 ∧
    are_externally_tangent c1 c4 ∧
    are_externally_tangent c2 c3 ∧
    are_externally_tangent c2 c4 ∧
    are_externally_tangent c3 c4 :=
sorry

end four_mutually_tangent_circles_exist_l3714_371453


namespace least_positive_solution_congruence_l3714_371439

theorem least_positive_solution_congruence :
  ∃! x : ℕ+, x.val + 7813 ≡ 2500 [ZMOD 15] ∧
  ∀ y : ℕ+, y.val + 7813 ≡ 2500 [ZMOD 15] → x ≤ y :=
by sorry

end least_positive_solution_congruence_l3714_371439


namespace max_value_ab_l3714_371412

theorem max_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^a * 2^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt 2 = Real.sqrt (2^x * 2^y) → a * b ≥ x * y) ∧ a * b = 1/4 :=
sorry

end max_value_ab_l3714_371412


namespace wire_cutting_problem_l3714_371404

theorem wire_cutting_problem (piece_length : ℕ) : 
  piece_length > 0 ∧
  9 * piece_length ≤ 1000 ∧
  9 * piece_length ≤ 1100 ∧
  10 * piece_length > 1100 →
  piece_length = 111 :=
by sorry

end wire_cutting_problem_l3714_371404


namespace eight_lines_form_784_parallelograms_intersecting_parallel_lines_theorem_l3714_371402

/-- The number of parallelograms formed by two sets of intersecting parallel lines -/
def parallelogramsCount (n m : ℕ) : ℕ := (n.choose 2) * (m.choose 2)

/-- Theorem stating that 8 lines in each set form 784 parallelograms -/
theorem eight_lines_form_784_parallelograms (n : ℕ) :
  parallelogramsCount n 8 = 784 → n = 8 := by
  sorry

/-- Main theorem proving that given 8 lines in one set and 784 parallelograms, 
    the other set must have 8 lines -/
theorem intersecting_parallel_lines_theorem :
  ∃ (n : ℕ), parallelogramsCount n 8 = 784 ∧ n = 8 := by
  sorry

end eight_lines_form_784_parallelograms_intersecting_parallel_lines_theorem_l3714_371402


namespace cab_delay_l3714_371495

theorem cab_delay (S : ℝ) (h : S > 0) : 
  let reduced_speed := (5 / 6) * S
  let usual_time := 30
  let new_time := usual_time * (S / reduced_speed)
  new_time - usual_time = 6 := by
sorry

end cab_delay_l3714_371495


namespace gold_bars_calculation_l3714_371489

theorem gold_bars_calculation (initial_bars : ℕ) (tax_rate : ℚ) (divorce_loss_fraction : ℚ) : 
  initial_bars = 60 →
  tax_rate = 1/10 →
  divorce_loss_fraction = 1/2 →
  initial_bars * (1 - tax_rate) * (1 - divorce_loss_fraction) = 27 := by
  sorry

end gold_bars_calculation_l3714_371489


namespace intersection_A_B_union_A_B_l3714_371420

open Set

-- Define sets A and B
def A : Set ℝ := {x | -2 < x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x < -1} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | x ≤ 3 ∨ x > 4} := by sorry

end intersection_A_B_union_A_B_l3714_371420


namespace article_price_reduction_l3714_371401

theorem article_price_reduction (reduced_price : ℝ) (reduction_percentage : ℝ) (original_price : ℝ) : 
  reduced_price = 608 ∧ 
  reduction_percentage = 24 ∧ 
  reduced_price = original_price * (1 - reduction_percentage / 100) → 
  original_price = 800 := by
  sorry

end article_price_reduction_l3714_371401


namespace jesse_remaining_money_l3714_371400

/-- Represents the currency exchange rates -/
structure ExchangeRates where
  usd_to_gbp : ℝ
  gbp_to_eur : ℝ

/-- Represents Jesse's shopping expenses -/
structure ShoppingExpenses where
  novel_price : ℝ
  novel_count : ℕ
  novel_discount : ℝ
  lunch_multiplier : ℕ
  lunch_tax : ℝ
  lunch_tip : ℝ
  jacket_price : ℝ
  jacket_discount : ℝ

/-- Calculates Jesse's remaining money after shopping -/
def remaining_money (initial_amount : ℝ) (rates : ExchangeRates) (expenses : ShoppingExpenses) : ℝ :=
  sorry

/-- Theorem stating that Jesse's remaining money is $174.66 -/
theorem jesse_remaining_money :
  let rates := ExchangeRates.mk (1/0.7) 1.15
  let expenses := ShoppingExpenses.mk 13 10 0.2 3 0.12 0.18 120 0.3
  remaining_money 500 rates expenses = 174.66 := by sorry

end jesse_remaining_money_l3714_371400


namespace new_person_age_l3714_371459

theorem new_person_age (T : ℕ) : 
  (T / 10 : ℚ) - 3 = ((T - 40 + 10) / 10 : ℚ) → 10 = 10 := by
sorry

end new_person_age_l3714_371459


namespace xy_positive_necessary_not_sufficient_l3714_371474

theorem xy_positive_necessary_not_sufficient (x y : ℝ) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 → x * y > 0) ∧
  (∃ x y : ℝ, x * y > 0 ∧ ¬(x > 0 ∧ y > 0)) :=
by sorry

end xy_positive_necessary_not_sufficient_l3714_371474


namespace modular_inverse_12_mod_997_l3714_371431

theorem modular_inverse_12_mod_997 : ∃ x : ℤ, 12 * x ≡ 1 [ZMOD 997] :=
by
  use 914
  sorry

end modular_inverse_12_mod_997_l3714_371431


namespace convex_polygon_diagonals_l3714_371417

theorem convex_polygon_diagonals (n : ℕ) (h : n = 49) : 
  (n * (n - 3)) / 2 = 23 * n := by
  sorry

end convex_polygon_diagonals_l3714_371417


namespace distribute_5_3_eq_31_l3714_371478

/-- The number of ways to distribute n different items into k identical bags -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 different items into 3 identical bags -/
def distribute_5_3 : ℕ := distribute 5 3

theorem distribute_5_3_eq_31 : distribute_5_3 = 31 := by sorry

end distribute_5_3_eq_31_l3714_371478


namespace triangular_array_coin_sum_l3714_371451

def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem triangular_array_coin_sum :
  ∃ N : ℕ, triangular_sum N = 3780 ∧ sum_of_digits N = 15 := by
  sorry

end triangular_array_coin_sum_l3714_371451


namespace sandwich_combinations_l3714_371414

theorem sandwich_combinations :
  let meat_types : ℕ := 12
  let cheese_types : ℕ := 12
  let spread_types : ℕ := 5
  let meat_selection : ℕ := meat_types
  let cheese_selection : ℕ := cheese_types.choose 2
  let spread_selection : ℕ := spread_types
  meat_selection * cheese_selection * spread_selection = 3960 :=
by sorry

end sandwich_combinations_l3714_371414


namespace prob_two_red_shoes_l3714_371472

/-- The probability of drawing two red shoes from a set of 4 red shoes and 4 green shoes -/
theorem prob_two_red_shoes : 
  let total_shoes : ℕ := 4 + 4
  let red_shoes : ℕ := 4
  let draw_count : ℕ := 2
  let total_ways := Nat.choose total_shoes draw_count
  let red_ways := Nat.choose red_shoes draw_count
  (red_ways : ℚ) / total_ways = 3 / 14 := by sorry

end prob_two_red_shoes_l3714_371472


namespace unique_solution_iff_prime_l3714_371450

theorem unique_solution_iff_prime (n : ℕ) : 
  (∃! (x y : ℕ), (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / n) ↔ Nat.Prime n :=
sorry

end unique_solution_iff_prime_l3714_371450


namespace completing_square_quadratic_l3714_371409

theorem completing_square_quadratic :
  ∀ x : ℝ, x^2 + 4*x - 1 = 0 ↔ (x + 2)^2 = 5 := by
  sorry

end completing_square_quadratic_l3714_371409


namespace f_properties_l3714_371430

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -a * x + x + a

-- Define the open interval (0,1]
def openUnitInterval : Set ℝ := { x | 0 < x ∧ x ≤ 1 }

-- Theorem statement
theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x ∈ openUnitInterval, ∀ y ∈ openUnitInterval, x < y → f a x < f a y) ↔ (0 < a ∧ a ≤ 1) ∧
  (∃ M : ℝ, M = 1 ∧ ∀ x ∈ openUnitInterval, f a x ≤ M) :=
by sorry

end f_properties_l3714_371430


namespace rectangular_box_surface_area_l3714_371407

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 200) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + a * c) = 1875 := by
sorry

end rectangular_box_surface_area_l3714_371407


namespace obtuse_triangle_properties_l3714_371454

/-- Properties of an obtuse triangle ABC -/
structure ObtuseTriangleABC where
  -- Side lengths
  a : ℝ
  b : ℝ
  -- Angle A in radians
  A : ℝ
  -- Triangle ABC is obtuse
  is_obtuse : Bool
  -- Given conditions
  ha : a = 7
  hb : b = 8
  hA : A = π / 3
  h_obtuse : is_obtuse = true

/-- Main theorem about the obtuse triangle ABC -/
theorem obtuse_triangle_properties (t : ObtuseTriangleABC) :
  -- 1. sin B = (4√3) / 7
  Real.sin (Real.arcsin ((t.b * Real.sin t.A) / t.a)) = (4 * Real.sqrt 3) / 7 ∧
  -- 2. Height on side BC = (12√3) / 7
  ∃ (h : ℝ), h = (12 * Real.sqrt 3) / 7 ∧ h = t.b * Real.sin (π - t.A - Real.arcsin ((t.b * Real.sin t.A) / t.a)) :=
by sorry

end obtuse_triangle_properties_l3714_371454


namespace quadratic_decreasing_interval_l3714_371426

-- Define the quadratic function
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_decreasing_interval (b c : ℝ) :
  (f b c 1 = 0) → (f b c 3 = 0) →
  ∃ (x : ℝ), ∀ (y : ℝ), y < x → (∀ (z : ℝ), y < z → f b c y > f b c z) ∧ x = 2 :=
by sorry

end quadratic_decreasing_interval_l3714_371426


namespace bike_ride_time_l3714_371466

/-- Represents the problem of calculating the time to cover a highway stretch on a bike --/
theorem bike_ride_time (highway_length : Real) (highway_width : Real) (bike_speed : Real) :
  highway_length = 2 → -- 2 miles
  highway_width = 60 / 5280 → -- 60 feet converted to miles
  bike_speed = 6 → -- 6 miles per hour
  (π * highway_length) / bike_speed = π / 6 := by
  sorry


end bike_ride_time_l3714_371466


namespace perpendicular_vectors_l3714_371494

/-- Given two vectors a and b in ℝ², where a is perpendicular to a + b, prove that the second component of b is -6. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a.1 = 4 ∧ a.2 = 2 ∧ b.1 = -2) 
  (perp : a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 0) : 
  b.2 = -6 := by
  sorry

end perpendicular_vectors_l3714_371494


namespace repeating_decimal_subtraction_l3714_371408

theorem repeating_decimal_subtraction (x : ℚ) : x = 1/3 → 5 - 7 * x = 8/3 := by
  sorry

end repeating_decimal_subtraction_l3714_371408


namespace inequality_equivalence_l3714_371488

theorem inequality_equivalence (x : ℝ) : 3 * x^2 - 2 * x - 1 > 4 * x + 5 ↔ x < 1 - Real.sqrt 3 ∨ x > 1 + Real.sqrt 3 := by
  sorry

end inequality_equivalence_l3714_371488


namespace max_cables_for_given_network_l3714_371462

/-- Represents a computer network with two brands of computers. -/
structure ComputerNetwork where
  total_employees : ℕ
  brand_x_count : ℕ
  brand_y_count : ℕ
  max_connections_per_computer : ℕ
  (total_is_sum : total_employees = brand_x_count + brand_y_count)
  (max_connections_positive : max_connections_per_computer > 0)

/-- The maximum number of cables that can be used in the network. -/
def max_cables (network : ComputerNetwork) : ℕ :=
  min (network.brand_x_count * network.max_connections_per_computer)
      (network.brand_y_count * network.max_connections_per_computer)

/-- The theorem stating the maximum number of cables for the given network configuration. -/
theorem max_cables_for_given_network :
  ∃ (network : ComputerNetwork),
    network.total_employees = 40 ∧
    network.brand_x_count = 25 ∧
    network.brand_y_count = 15 ∧
    network.max_connections_per_computer = 3 ∧
    max_cables network = 45 := by
  sorry

end max_cables_for_given_network_l3714_371462


namespace geometric_sequence_ratio_l3714_371490

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 3 = 2 →
  a 4 * a 6 = 64 →
  (a 5 + a 6) / (a 1 + a 2) = 16 := by
  sorry

end geometric_sequence_ratio_l3714_371490


namespace find_S_l3714_371418

def f : ℕ → ℕ
  | 0 => 0
  | n + 1 => f n + 3

theorem find_S (S : ℕ) (h : 2 * f S = 3996) : S = 666 := by
  sorry

end find_S_l3714_371418


namespace problem_statement_l3714_371457

theorem problem_statement (x y : ℝ) (h : x + 2*y - 1 = 0) : 3 + 2*x + 4*y = 5 := by
  sorry

end problem_statement_l3714_371457


namespace min_fence_length_is_28_l3714_371484

/-- Represents a rectangular flower bed -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Calculates the minimum fence length required for a flower bed with one side against a wall -/
def minFenceLength (fb : FlowerBed) : ℝ :=
  2 * fb.width + fb.length

/-- The specific flower bed in the problem -/
def problemFlowerBed : FlowerBed :=
  { length := 12, width := 8 }

theorem min_fence_length_is_28 :
  minFenceLength problemFlowerBed = 28 := by
  sorry

#eval minFenceLength problemFlowerBed

end min_fence_length_is_28_l3714_371484


namespace dice_sum_symmetry_l3714_371438

def num_dice : ℕ := 8
def min_face : ℕ := 1
def max_face : ℕ := 6

def sum_symmetric (s : ℕ) : ℕ :=
  2 * ((num_dice * min_face + num_dice * max_face) / 2) - s

theorem dice_sum_symmetry :
  sum_symmetric 12 = 44 :=
by sorry

end dice_sum_symmetry_l3714_371438


namespace diagonal_length_of_quadrilateral_l3714_371483

/-- The length of a diagonal in a quadrilateral with given offsets and area -/
theorem diagonal_length_of_quadrilateral (offset1 offset2 area : ℝ) 
  (h1 : offset1 = 9)
  (h2 : offset2 = 6)
  (h3 : area = 195) :
  ∃ d : ℝ, d = 26 ∧ (1/2 * d * offset1) + (1/2 * d * offset2) = area :=
by sorry

end diagonal_length_of_quadrilateral_l3714_371483


namespace baseball_game_earnings_l3714_371448

theorem baseball_game_earnings (total : ℝ) (difference : ℝ) (wednesday : ℝ) (sunday : ℝ)
  (h1 : total = 4994.50)
  (h2 : difference = 1330.50)
  (h3 : wednesday + sunday = total)
  (h4 : wednesday = sunday - difference) :
  wednesday = 1832 := by
sorry

end baseball_game_earnings_l3714_371448


namespace quadratic_equation_solution_l3714_371464

theorem quadratic_equation_solution :
  ∃ (x₁ x₂ : ℝ),
    (x₁ * (5 * x₁ - 11) = 2) ∧
    (x₂ * (5 * x₂ - 11) = 2) ∧
    (x₁ = (11 + Real.sqrt 161) / 10) ∧
    (x₂ = (11 - Real.sqrt 161) / 10) ∧
    (Nat.gcd 11 (Nat.gcd 161 10) = 1) ∧
    (11 + 161 + 10 = 182) :=
by
  sorry

end quadratic_equation_solution_l3714_371464


namespace triangle_perimeter_bound_l3714_371440

theorem triangle_perimeter_bound (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_angle_B : B = π / 3) (h_side_b : b = 2 * Real.sqrt 3) :
  a + b + c ≤ 6 * Real.sqrt 3 := by
sorry

end triangle_perimeter_bound_l3714_371440


namespace initial_leaves_count_l3714_371444

/-- The number of leaves that blew away -/
def leaves_blown_away : ℕ := 244

/-- The number of leaves left -/
def leaves_left : ℕ := 112

/-- The initial number of leaves -/
def initial_leaves : ℕ := leaves_blown_away + leaves_left

theorem initial_leaves_count : initial_leaves = 356 := by
  sorry

end initial_leaves_count_l3714_371444


namespace opposite_solutions_imply_a_l3714_371485

theorem opposite_solutions_imply_a (a : ℝ) : 
  (∃ x y : ℝ, 2 * (x - 1) - 6 = 0 ∧ 1 - (3 * a - x) / 3 = 0 ∧ x = -y) → 
  a = -1/3 := by
sorry

end opposite_solutions_imply_a_l3714_371485


namespace bug_probability_l3714_371425

/-- Probability of the bug being at vertex A after n steps -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - P n)

/-- The probability of the bug being at vertex A after 7 steps is 182/729 -/
theorem bug_probability : P 7 = 182 / 729 := by
  sorry

end bug_probability_l3714_371425


namespace train_speed_problem_l3714_371442

/-- Proves that Train B's speed is 80 mph given the problem conditions --/
theorem train_speed_problem (speed_a : ℝ) (time_difference : ℝ) (overtake_time : ℝ) :
  speed_a = 60 →
  time_difference = 40 / 60 →
  overtake_time = 120 / 60 →
  ∃ (speed_b : ℝ),
    speed_b * overtake_time = speed_a * (time_difference + overtake_time) ∧
    speed_b = 80 := by
  sorry


end train_speed_problem_l3714_371442


namespace intersection_determinant_l3714_371437

theorem intersection_determinant (a : ℝ) :
  (∃! p : ℝ × ℝ, a * p.1 + p.2 + 3 = 0 ∧ p.1 + p.2 + 2 = 0 ∧ 2 * p.1 - p.2 + 1 = 0) →
  Matrix.det !![a, 1, 3; 1, 1, 2; 2, -1, 1] = 0 := by
sorry

end intersection_determinant_l3714_371437


namespace unique_solution_linear_equation_l3714_371473

theorem unique_solution_linear_equation (a b : ℝ) :
  (a * 1 + b * 2 = 2) ∧ (a * 2 + b * 5 = 2) → a = 6 ∧ b = -2 := by
  sorry

end unique_solution_linear_equation_l3714_371473


namespace min_value_theorem_l3714_371499

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 5*x*y) :
  3*x + 4*y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3*y₀ = 5*x₀*y₀ ∧ 3*x₀ + 4*y₀ = 5 :=
by sorry


end min_value_theorem_l3714_371499


namespace second_sum_proof_l3714_371441

/-- Given a total sum and interest conditions, prove the second sum -/
theorem second_sum_proof (total : ℝ) (first : ℝ) (second : ℝ) : 
  total = 2743 →
  first + second = total →
  (first * 3 / 100 * 8) = (second * 5 / 100 * 3) →
  second = 1688 := by
  sorry

end second_sum_proof_l3714_371441


namespace even_Z_tetrominoes_l3714_371410

/-- Represents a lattice polygon -/
structure LatticePolygon where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents an S-tetromino -/
inductive STetromino

/-- Represents a Z-tetromino -/
inductive ZTetromino

/-- Represents either an S-tetromino or a Z-tetromino -/
inductive Tetromino
  | S : STetromino → Tetromino
  | Z : ZTetromino → Tetromino

/-- Predicate indicating if a lattice polygon can be tiled with S-tetrominoes -/
def canBeTiledWithS (P : LatticePolygon) : Prop := sorry

/-- Represents a tiling of a lattice polygon using S- and Z-tetrominoes -/
def Tiling (P : LatticePolygon) := List Tetromino

/-- Counts the number of Z-tetrominoes in a tiling -/
def countZTetrominoes (tiling : Tiling P) : Nat := sorry

/-- Main theorem: For any lattice polygon that can be tiled with S-tetrominoes,
    any tiling using S- and Z-tetrominoes will contain an even number of Z-tetrominoes -/
theorem even_Z_tetrominoes (P : LatticePolygon) (h : canBeTiledWithS P) :
  ∀ (tiling : Tiling P), Even (countZTetrominoes tiling) := by
  sorry

end even_Z_tetrominoes_l3714_371410


namespace tan_alpha_two_implies_fraction_l3714_371447

theorem tan_alpha_two_implies_fraction (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4 := by
  sorry

end tan_alpha_two_implies_fraction_l3714_371447


namespace triangle_angle_c_l3714_371467

theorem triangle_angle_c (A B C : ℝ) (m n : ℝ × ℝ) : 
  0 < C ∧ C < π →
  A + B + C = π →
  m = (Real.sqrt 3 * Real.sin A, Real.sin B) →
  n = (Real.cos B, Real.sqrt 3 * Real.cos A) →
  m.1 * n.1 + m.2 * n.2 = 1 + Real.cos (A + B) →
  C = 2 * π / 3 := by sorry

end triangle_angle_c_l3714_371467


namespace seven_equidistant_planes_l3714_371411

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A type representing a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Function to check if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Function to check if a plane is equidistant from four points -/
def isEquidistant (plane : Plane3D) (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Function to count the number of planes equidistant from four points -/
def countEquidistantPlanes (p1 p2 p3 p4 : Point3D) : ℕ := sorry

/-- Theorem stating that there are exactly 7 equidistant planes for four non-coplanar points -/
theorem seven_equidistant_planes
  (p1 p2 p3 p4 : Point3D)
  (h : ¬ areCoplanar p1 p2 p3 p4) :
  countEquidistantPlanes p1 p2 p3 p4 = 7 := by
  sorry

end seven_equidistant_planes_l3714_371411


namespace son_father_height_relationship_l3714_371406

/-- Represents the possible types of relationships between variables -/
inductive RelationshipType
  | Deterministic
  | Correlation
  | Functional
  | None

/-- Represents the relationship between a son's height and his father's height -/
structure HeightRelationship where
  type : RelationshipType
  isUncertain : Bool

/-- Theorem: The relationship between a son's height and his father's height is a correlation relationship -/
theorem son_father_height_relationship :
  ∀ (r : HeightRelationship), r.isUncertain → r.type = RelationshipType.Correlation :=
by sorry

end son_father_height_relationship_l3714_371406


namespace inequality_range_l3714_371460

theorem inequality_range (x y a : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  (∀ x y, x > 0 → y > 0 → x + y = 1 → (1/x) + (16/y) > a^2 + 24*a) ↔ -25 < a ∧ a < 1 :=
by sorry

end inequality_range_l3714_371460


namespace cos_angle_AMB_formula_l3714_371427

/-- Regular square pyramid with vertex A and square base BCDE -/
structure RegularSquarePyramid where
  s : ℝ  -- side length of the base
  h : ℝ  -- height of the pyramid
  l : ℝ  -- slant height of the pyramid

/-- Point M is the midpoint of diagonal BD -/
def midpoint_M (p : RegularSquarePyramid) : ℝ × ℝ × ℝ := sorry

/-- Angle AMB in the regular square pyramid -/
def angle_AMB (p : RegularSquarePyramid) : ℝ := sorry

theorem cos_angle_AMB_formula (p : RegularSquarePyramid) :
  Real.cos (angle_AMB p) = (p.l^2 + p.h^2) / (2 * p.l * Real.sqrt (p.h^2 + p.s^2 / 2)) :=
sorry

end cos_angle_AMB_formula_l3714_371427


namespace margie_change_theorem_l3714_371496

-- Define the problem parameters
def num_apples : ℕ := 5
def cost_per_apple : ℚ := 30 / 100  -- 30 cents in dollars
def discount_rate : ℚ := 10 / 100   -- 10% discount
def paid_amount : ℚ := 10           -- 10-dollar bill

-- Define the theorem
theorem margie_change_theorem :
  let total_cost := num_apples * cost_per_apple
  let discounted_cost := total_cost * (1 - discount_rate)
  let change := paid_amount - discounted_cost
  change = 865 / 100 := by
sorry


end margie_change_theorem_l3714_371496


namespace exactly_five_ladybugs_l3714_371428

/-- Represents a ladybug with a specific number of spots -/
inductive Ladybug
  | sixSpots
  | fourSpots

/-- Represents a statement made by a ladybug -/
inductive Statement
  | allSame
  | totalThirty
  | totalTwentySix

/-- The meadow containing ladybugs -/
structure Meadow where
  ladybugs : List Ladybug

/-- Evaluates whether a statement is true for a given meadow -/
def isStatementTrue (m : Meadow) (s : Statement) : Bool :=
  match s with
  | Statement.allSame => sorry
  | Statement.totalThirty => sorry
  | Statement.totalTwentySix => sorry

/-- Counts the number of true statements in a list of statements for a given meadow -/
def countTrueStatements (m : Meadow) (statements : List Statement) : Nat :=
  statements.filter (isStatementTrue m) |>.length

/-- Theorem stating that there are exactly 5 ladybugs in the meadow -/
theorem exactly_five_ladybugs :
  ∃ (m : Meadow),
    m.ladybugs.length = 5 ∧
    (∀ l : Ladybug, l ∈ m.ladybugs → (l = Ladybug.sixSpots ∨ l = Ladybug.fourSpots)) ∧
    countTrueStatements m [Statement.allSame, Statement.totalThirty, Statement.totalTwentySix] = 1 :=
  sorry

end exactly_five_ladybugs_l3714_371428


namespace complex_number_real_twice_imaginary_l3714_371423

theorem complex_number_real_twice_imaginary (m : ℝ) : 
  let z : ℂ := (1 + m * Complex.I) / (4 - 3 * Complex.I) + m / 25
  (z.re = 2 * z.im) → m = -1/5 := by
sorry

end complex_number_real_twice_imaginary_l3714_371423


namespace find_second_number_l3714_371463

theorem find_second_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((10 + 19 + x) / 3) + 7 →
  x = 70 := by
sorry

end find_second_number_l3714_371463


namespace correct_testing_schemes_l3714_371477

/-- The number of genuine products -/
def genuine_products : ℕ := 5

/-- The number of defective products -/
def defective_products : ℕ := 4

/-- The position at which the last defective product is detected -/
def last_defective_position : ℕ := 6

/-- The number of ways to arrange products such that the last defective product
    is at the specified position and all defective products are included -/
def testing_schemes : ℕ := defective_products * (genuine_products.choose 2) * (last_defective_position - 1).factorial

theorem correct_testing_schemes :
  testing_schemes = 4800 := by sorry

end correct_testing_schemes_l3714_371477


namespace apple_buying_problem_l3714_371461

/-- Proves that given the conditions of the apple-buying problem, each man bought 30 apples. -/
theorem apple_buying_problem (men women man_apples woman_apples total_apples : ℕ) 
  (h1 : men = 2)
  (h2 : women = 3)
  (h3 : man_apples + 20 = woman_apples)
  (h4 : men * man_apples + women * woman_apples = total_apples)
  (h5 : total_apples = 210) :
  man_apples = 30 := by
sorry

end apple_buying_problem_l3714_371461


namespace tan_product_pi_ninths_l3714_371432

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = Real.sqrt 3 := by
  sorry

end tan_product_pi_ninths_l3714_371432


namespace sally_has_six_cards_l3714_371422

/-- The number of baseball cards Sally has after selling some to Sara -/
def sallys_remaining_cards (initial_cards torn_cards cards_sold : ℕ) : ℕ :=
  initial_cards - torn_cards - cards_sold

/-- Theorem stating that Sally has 6 cards remaining -/
theorem sally_has_six_cards :
  sallys_remaining_cards 39 9 24 = 6 := by
  sorry

end sally_has_six_cards_l3714_371422


namespace unique_integer_solution_to_equation_l3714_371415

theorem unique_integer_solution_to_equation :
  ∀ x y : ℤ, x^4 + y^4 = 3*x^3*y → x = 0 ∧ y = 0 := by
sorry

end unique_integer_solution_to_equation_l3714_371415


namespace sum_of_x_and_y_l3714_371479

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 60) (h2 : x = 37) : x + y = 14 := by
  sorry

end sum_of_x_and_y_l3714_371479


namespace perpendicular_lines_from_parallel_planes_l3714_371456

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes
  (l m : Line) (α β : Plane)
  (h1 : perp_line_plane l α)
  (h2 : subset_line_plane m β)
  (h3 : parallel_planes α β) :
  perp_lines l m :=
sorry

end perpendicular_lines_from_parallel_planes_l3714_371456


namespace expansion_coefficient_l3714_371487

theorem expansion_coefficient (n : ℕ) : 
  (Nat.choose n 2) * 9 = 54 → n = 4 := by sorry

end expansion_coefficient_l3714_371487


namespace compute_fraction_power_l3714_371413

theorem compute_fraction_power : 8 * (1 / 3)^4 = 8 / 81 := by
  sorry

end compute_fraction_power_l3714_371413


namespace two_boys_three_girls_probability_l3714_371435

-- Define the number of children
def n : ℕ := 5

-- Define the number of boys
def k : ℕ := 2

-- Define the probability of having a boy
def p : ℚ := 1/2

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := sorry

-- Define the probability function
def probability (n k : ℕ) (p : ℚ) : ℚ := sorry

-- Theorem statement
theorem two_boys_three_girls_probability :
  probability n k p = 0.3125 := by sorry

end two_boys_three_girls_probability_l3714_371435


namespace late_passengers_l3714_371486

theorem late_passengers (total : ℕ) (on_time : ℕ) (h1 : total = 14720) (h2 : on_time = 14507) :
  total - on_time = 213 := by
  sorry

end late_passengers_l3714_371486


namespace lottery_jackpot_probability_l3714_371416

def megaBallCount : ℕ := 30
def winnerBallCount : ℕ := 50
def bonusBallCount : ℕ := 15
def winnerBallsDrawn : ℕ := 5

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def megaBallProb : ℚ := 1 / megaBallCount
def winnerBallsProb : ℚ := 1 / (binomial winnerBallCount winnerBallsDrawn)
def bonusBallProb : ℚ := 1 / bonusBallCount

theorem lottery_jackpot_probability : 
  megaBallProb * winnerBallsProb * bonusBallProb = 1 / 954594900 := by
  sorry

end lottery_jackpot_probability_l3714_371416


namespace sum_of_solution_l3714_371405

theorem sum_of_solution (a b : ℝ) : 
  3 * a + 7 * b = 1977 → 
  5 * a + b = 2007 → 
  a + b = 498 := by
  sorry

end sum_of_solution_l3714_371405


namespace product_of_three_numbers_l3714_371480

theorem product_of_three_numbers (x y z n : ℝ) : 
  x + y + z = 150 ∧ 
  x ≤ y ∧ x ≤ z ∧ 
  z ≤ y ∧
  7 * x = n ∧ 
  y - 10 = n ∧ 
  z + 10 = n → 
  x * y * z = 48000 := by
  sorry

end product_of_three_numbers_l3714_371480


namespace position_of_b_l3714_371455

theorem position_of_b (a b c : ℚ) (h : |a| + |b - c| = |a - c|) :
  (∃ a b c : ℚ, a < b ∧ c < b ∧ |a| + |b - c| = |a - c|) ∧
  (∃ a b c : ℚ, b < a ∧ b < c ∧ |a| + |b - c| = |a - c|) ∧
  (∃ a b c : ℚ, a < b ∧ b < c ∧ |a| + |b - c| = |a - c|) :=
sorry

end position_of_b_l3714_371455


namespace unique_reciprocal_function_l3714_371476

/-- Given a function f(x) = x / (ax + b) where a and b are constants, a ≠ 0,
    f(2) = 1, and f(x) = x has a unique solution, prove that f(x) = 2x / (x + 2) -/
theorem unique_reciprocal_function (a b : ℝ) (ha : a ≠ 0) :
  (∀ x, x ≠ -b/a → (x / (a * x + b) = x → ∀ y, y ≠ -b/a → y / (a * y + b) = y → x = y)) →
  (2 / (2 * a + b) = 1) →
  (∀ x, x ≠ -b/a → x / (a * x + b) = 2 * x / (x + 2)) :=
by sorry

end unique_reciprocal_function_l3714_371476


namespace parallel_lines_condition_l3714_371482

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def l₂ (a x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- State the theorem
theorem parallel_lines_condition (a : ℝ) :
  (a = 1 ↔ parallel (l₁ a) (l₂ a)) :=
sorry

end parallel_lines_condition_l3714_371482


namespace safari_count_l3714_371443

theorem safari_count (antelopes rabbits hyenas wild_dogs leopards : ℕ) : 
  antelopes = 80 →
  rabbits > antelopes →
  hyenas = antelopes + rabbits - 42 →
  wild_dogs = hyenas + 50 →
  leopards * 2 = rabbits →
  antelopes + rabbits + hyenas + wild_dogs + leopards = 605 →
  rabbits - antelopes = 70 := by
sorry

end safari_count_l3714_371443


namespace remaining_amount_after_purchase_l3714_371470

def lollipop_price : ℚ := 1.5
def gummy_pack_price : ℚ := 2
def chips_price : ℚ := 1.25
def chocolate_price : ℚ := 1.75
def discount_rate : ℚ := 0.1
def tax_rate : ℚ := 0.05
def initial_amount : ℚ := 25

def total_cost : ℚ := 4 * lollipop_price + 2 * gummy_pack_price + 3 * chips_price + chocolate_price

def discounted_cost : ℚ := total_cost * (1 - discount_rate)

def final_cost : ℚ := discounted_cost * (1 + tax_rate)

theorem remaining_amount_after_purchase : 
  initial_amount - final_cost = 10.35 := by sorry

end remaining_amount_after_purchase_l3714_371470


namespace cubic_roots_sum_l3714_371469

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 2*a - 2 = 0) → 
  (b^3 - 2*b - 2 = 0) → 
  (c^3 - 2*c - 2 = 0) → 
  a*(b - c)^2 + b*(c - a)^2 + c*(a - b)^2 = -18 := by
sorry

end cubic_roots_sum_l3714_371469


namespace locus_is_ellipse_l3714_371475

-- Define the circles O₁ and O₂
def O₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def O₂ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define the center of circle C
structure CircleCenter where
  x : ℝ
  y : ℝ

-- Define the property of being externally tangent to O₁ and internally tangent to O₂
def is_tangent_to_O₁_and_O₂ (c : CircleCenter) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    ((c.x - 1)^2 + c.y^2 = (r + 1)^2) ∧
    ((c.x + 1)^2 + c.y^2 = (4 - r)^2)

-- Define the locus of centers of circle C
def locus : Set CircleCenter :=
  {c : CircleCenter | is_tangent_to_O₁_and_O₂ c}

-- Theorem stating that the locus is an ellipse
theorem locus_is_ellipse :
  ∃ a b h k : ℝ, a > 0 ∧ b > 0 ∧
    ∀ c : CircleCenter, c ∈ locus ↔
      (c.x - h)^2 / a^2 + (c.y - k)^2 / b^2 = 1 :=
sorry

end locus_is_ellipse_l3714_371475


namespace min_max_x_sum_l3714_371433

theorem min_max_x_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 6) 
  (sum_sq_eq : x^2 + y^2 + z^2 = 10) : 
  ∃ (x_min x_max : ℝ), 
    (∀ x', ∃ y' z', x' + y' + z' = 6 ∧ x'^2 + y'^2 + z'^2 = 10 → x_min ≤ x') ∧
    (∀ x', ∃ y' z', x' + y' + z' = 6 ∧ x'^2 + y'^2 + z'^2 = 10 → x' ≤ x_max) ∧
    x_min = 8/3 ∧ 
    x_max = 2 ∧ 
    x_min + x_max = 14/3 := by
  sorry

end min_max_x_sum_l3714_371433


namespace total_crayons_l3714_371429

theorem total_crayons (orange_boxes : Nat) (orange_per_box : Nat)
                      (blue_boxes : Nat) (blue_per_box : Nat)
                      (red_boxes : Nat) (red_per_box : Nat) :
  orange_boxes = 6 →
  orange_per_box = 8 →
  blue_boxes = 7 →
  blue_per_box = 5 →
  red_boxes = 1 →
  red_per_box = 11 →
  orange_boxes * orange_per_box + blue_boxes * blue_per_box + red_boxes * red_per_box = 94 := by
  sorry

end total_crayons_l3714_371429


namespace increasing_f_implies_m_bound_l3714_371449

/-- A cubic function parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

/-- The derivative of f with respect to x -/
def f_deriv (m : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * m * x + 6

theorem increasing_f_implies_m_bound :
  (∀ x > 2, ∀ y > x, f m y > f m x) →
  m ≤ 5/2 := by
  sorry

end increasing_f_implies_m_bound_l3714_371449


namespace remainder_problem_l3714_371434

theorem remainder_problem : (5^7 + 9^6 + 3^5) % 7 = 5 := by
  sorry

end remainder_problem_l3714_371434


namespace expression_evaluation_l3714_371491

theorem expression_evaluation (a b c : ℝ) 
  (h1 : c = b - 11)
  (h2 : b = a + 3)
  (h3 : a = 5)
  (h4 : a + 1 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  ((a + 3) / (a + 1)) * ((b - 2) / (b - 3)) * ((c + 9) / (c + 7)) = 1 / 3 := by
  sorry

end expression_evaluation_l3714_371491


namespace min_dot_product_l3714_371419

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

-- Define the fixed point E
def E : ℝ × ℝ := (3, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Define perpendicularity condition
def perpendicular (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define dot product of vectors
def dot_product (A B C D : ℝ × ℝ) : ℝ :=
  (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2)

-- Theorem statement
theorem min_dot_product :
  ∀ P Q : ℝ × ℝ, 
  point_on_ellipse P → 
  point_on_ellipse Q → 
  perpendicular E P Q → 
  ∃ m : ℝ, 
  (∀ P' Q' : ℝ × ℝ, 
    point_on_ellipse P' → 
    point_on_ellipse Q' → 
    perpendicular E P' Q' → 
    m ≤ dot_product E P P Q) ∧ 
  m = 6 :=
sorry

end min_dot_product_l3714_371419


namespace ben_spending_l3714_371436

-- Define the prices and quantities
def apple_price : ℚ := 2
def apple_quantity : ℕ := 7
def milk_price : ℚ := 4
def milk_quantity : ℕ := 4
def bread_price : ℚ := 3
def bread_quantity : ℕ := 3
def sugar_price : ℚ := 6
def sugar_quantity : ℕ := 3

-- Define the discounts
def dairy_discount : ℚ := 0.25
def coupon_discount : ℚ := 10
def coupon_threshold : ℚ := 50

-- Define the total spending function
def total_spending : ℚ :=
  let apple_cost := apple_price * apple_quantity
  let milk_cost := milk_price * milk_quantity * (1 - dairy_discount)
  let bread_cost := bread_price * bread_quantity
  let sugar_cost := sugar_price * sugar_quantity
  let subtotal := apple_cost + milk_cost + bread_cost + sugar_cost
  if subtotal ≥ coupon_threshold then subtotal - coupon_discount else subtotal

-- Theorem to prove
theorem ben_spending :
  total_spending = 43 :=
sorry

end ben_spending_l3714_371436


namespace sandy_safe_moon_tokens_l3714_371493

theorem sandy_safe_moon_tokens :
  ∀ (T : ℕ),
    (T / 2 = T / 8 + 375000) →
    T = 1000000 := by
  sorry

end sandy_safe_moon_tokens_l3714_371493


namespace only_courses_form_set_l3714_371492

-- Define a type for the universe of discourse
def Universe : Type := Unit

-- Define predicates for each option
def likes_airplanes (x : Universe) : Prop := sorry
def is_sufficiently_small_negative (x : ℝ) : Prop := sorry
def has_poor_eyesight (x : Universe) : Prop := sorry
def is_course_of_class_on_day (x : Universe) : Prop := sorry

-- Define what it means for a predicate to form a well-defined set
def forms_well_defined_set {α : Type} (P : α → Prop) : Prop := sorry

-- State the theorem
theorem only_courses_form_set :
  ¬(forms_well_defined_set likes_airplanes) ∧
  ¬(forms_well_defined_set is_sufficiently_small_negative) ∧
  ¬(forms_well_defined_set has_poor_eyesight) ∧
  (forms_well_defined_set is_course_of_class_on_day) :=
sorry

end only_courses_form_set_l3714_371492


namespace parallel_lines_equation_l3714_371498

/-- A line in 2D space represented by its slope-intercept form -/
structure Line where
  slope : ℚ
  yIntercept : ℚ

/-- Distance between two parallel lines -/
def distanceBetweenParallelLines (l1 l2 : Line) : ℚ :=
  sorry

/-- Checks if two lines are parallel -/
def areParallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem parallel_lines_equation (l : Line) (P : ℚ × ℚ) (m : Line) :
  l.slope = -3/4 →
  P = (-2, 5) →
  areParallel l m →
  distanceBetweenParallelLines l m = 3 →
  (∃ (c : ℚ), (3 * P.1 + 4 * P.2 + c = 0 ∧ (c = 1 ∨ c = -29))) :=
sorry

end parallel_lines_equation_l3714_371498


namespace karen_pickup_cases_l3714_371403

/-- The number of boxes Karen sold -/
def boxes_sold : ℕ := 36

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 12

/-- The number of cases Karen needs to pick up -/
def cases_to_pickup : ℕ := boxes_sold / boxes_per_case

theorem karen_pickup_cases : cases_to_pickup = 3 := by
  sorry

end karen_pickup_cases_l3714_371403


namespace angle_A_is_60_degrees_max_area_is_5_exists_triangle_with_max_area_l3714_371471

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.a = t.b + t.c - 2

theorem angle_A_is_60_degrees (t : Triangle) 
  (h : triangle_condition t) : t.A = 60 := by sorry

theorem max_area_is_5 (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.a = 2) : 
  ∀ (s : ℝ), s = (1/2) * t.b * t.c * Real.sin t.A → s ≤ 5 := by sorry

theorem exists_triangle_with_max_area (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.a = 2) : 
  ∃ (s : ℝ), s = (1/2) * t.b * t.c * Real.sin t.A ∧ s = 5 := by sorry

end angle_A_is_60_degrees_max_area_is_5_exists_triangle_with_max_area_l3714_371471


namespace triangle_angle_proof_l3714_371445

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ A < π →
  B > 0 ∧ B < π →
  C > 0 ∧ C < π →
  A + B + C = π →
  a * Real.sin A = b * Real.sin B + (c - b) * Real.sin C →
  A = π / 3 := by
sorry

end triangle_angle_proof_l3714_371445


namespace square_fraction_is_perfect_square_l3714_371465

theorem square_fraction_is_perfect_square (a b : ℕ+) 
  (h : ∃ k : ℕ, (a + b)^2 = k * (4 * a * b + 1)) : 
  ∃ n : ℕ, (a + b)^2 / (4 * a * b + 1) = n^2 := by
  sorry

end square_fraction_is_perfect_square_l3714_371465


namespace difference_of_squares_special_case_l3714_371452

theorem difference_of_squares_special_case : (527 : ℕ) * 527 - 526 * 528 = 1 := by
  sorry

end difference_of_squares_special_case_l3714_371452


namespace number_ordering_l3714_371424

theorem number_ordering : 
  (3 : ℚ) / 8 < (3 : ℚ) / 4 ∧ 
  (3 : ℚ) / 4 < (7 : ℚ) / 5 ∧ 
  (7 : ℚ) / 5 < (143 : ℚ) / 100 ∧ 
  (143 : ℚ) / 100 < (13 : ℚ) / 8 := by
sorry

end number_ordering_l3714_371424


namespace scatter_plot_placement_l3714_371446

/-- Represents a variable in a scatter plot --/
inductive Variable
| Explanatory
| Forecast

/-- Represents an axis in a scatter plot --/
inductive Axis
| X
| Y

/-- Defines the relationship between variables and their roles in regression analysis --/
def is_independent (v : Variable) : Prop :=
  match v with
  | Variable.Explanatory => true
  | Variable.Forecast => false

/-- Defines the correct placement of variables on axes in a scatter plot --/
def correct_placement (v : Variable) (a : Axis) : Prop :=
  (v = Variable.Explanatory ∧ a = Axis.X) ∨ (v = Variable.Forecast ∧ a = Axis.Y)

/-- Theorem stating the correct placement of variables in a scatter plot for regression analysis --/
theorem scatter_plot_placement :
  ∀ (v : Variable) (a : Axis),
    is_independent v ↔ correct_placement v Axis.X :=
by sorry

end scatter_plot_placement_l3714_371446


namespace count_primes_with_squares_between_5000_and_8000_eq_5_l3714_371421

/-- The count of prime numbers whose squares are between 5000 and 8000 -/
def count_primes_with_squares_between_5000_and_8000 : Nat :=
  (Finset.filter (fun p => 5000 < p * p ∧ p * p < 8000) (Finset.filter Nat.Prime (Finset.range 90))).card

/-- Theorem stating that the count of prime numbers with squares between 5000 and 8000 is 5 -/
theorem count_primes_with_squares_between_5000_and_8000_eq_5 :
  count_primes_with_squares_between_5000_and_8000 = 5 := by
  sorry

end count_primes_with_squares_between_5000_and_8000_eq_5_l3714_371421


namespace lilly_fish_count_l3714_371497

/-- Given that Rosy has 12 fish and the total number of fish is 22,
    prove that Lilly has 10 fish. -/
theorem lilly_fish_count (rosy_fish : ℕ) (total_fish : ℕ) (h1 : rosy_fish = 12) (h2 : total_fish = 22) :
  total_fish - rosy_fish = 10 := by
  sorry

end lilly_fish_count_l3714_371497
