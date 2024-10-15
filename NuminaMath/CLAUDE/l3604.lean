import Mathlib

namespace NUMINAMATH_CALUDE_pencil_distribution_l3604_360477

/-- The number of ways to distribute n identical objects among k people,
    where each person gets at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- The number of friends -/
def num_friends : ℕ := 3

/-- The total number of pencils -/
def total_pencils : ℕ := 9

/-- Each friend must have at least one pencil -/
def min_pencils_per_friend : ℕ := 1

theorem pencil_distribution :
  distribute (total_pencils - num_friends * min_pencils_per_friend + num_friends) num_friends = 28 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l3604_360477


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3604_360413

theorem arithmetic_sequence_sum : 
  ∀ (a₁ aₙ d n : ℕ) (S : ℕ),
    a₁ = 1 →                   -- First term is 1
    aₙ = 25 →                  -- Last term is 25
    d = 2 →                    -- Common difference is 2
    aₙ = a₁ + (n - 1) * d →    -- Formula for the nth term of an arithmetic sequence
    S = n * (a₁ + aₙ) / 2 →    -- Formula for the sum of an arithmetic sequence
    S = 169 :=                 -- The sum is 169
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3604_360413


namespace NUMINAMATH_CALUDE_nabla_properties_and_equation_solution_l3604_360493

-- Define the ∇ operation
def nabla (X Y : ℤ) : ℤ := X + 2 * Y

-- State the theorem
theorem nabla_properties_and_equation_solution :
  (∀ X : ℤ, nabla X 0 = X) ∧
  (∀ X Y : ℤ, nabla X (Y - 1) = nabla X Y - 2) ∧
  (∀ X Y : ℤ, nabla X (Y + 1) = nabla X Y + 2) →
  (∀ X Y : ℤ, nabla X Y = X + 2 * Y) ∧
  (nabla (-673) (-673) = -2019 ∧ ∀ X : ℤ, nabla X X = -2019 → X = -673) :=
by sorry

end NUMINAMATH_CALUDE_nabla_properties_and_equation_solution_l3604_360493


namespace NUMINAMATH_CALUDE_skew_lines_angle_equals_dihedral_angle_l3604_360449

-- Define the dihedral angle
def dihedral_angle (α l β : Line3) : ℝ := sorry

-- Define perpendicularity between a line and a plane
def perpendicular (m : Line3) (α : Plane3) : Prop := sorry

-- Define the angle between two skew lines
def skew_line_angle (m n : Line3) : ℝ := sorry

-- Theorem statement
theorem skew_lines_angle_equals_dihedral_angle 
  (α l β : Line3) (m n : Line3) :
  dihedral_angle α l β = 60 →
  perpendicular m α →
  perpendicular n β →
  skew_line_angle m n = 60 := by sorry

end NUMINAMATH_CALUDE_skew_lines_angle_equals_dihedral_angle_l3604_360449


namespace NUMINAMATH_CALUDE_fruit_box_ratio_l3604_360446

/-- Proves that the ratio of peaches to oranges is 1:2 given the conditions of the fruit box problem -/
theorem fruit_box_ratio : 
  ∀ (total_fruits oranges apples peaches : ℕ),
  total_fruits = 56 →
  oranges = total_fruits / 4 →
  apples = 35 →
  apples = 5 * peaches →
  (peaches : ℚ) / oranges = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_fruit_box_ratio_l3604_360446


namespace NUMINAMATH_CALUDE_barrel_division_l3604_360425

theorem barrel_division (length width height : ℝ) (volume_small_barrel : ℝ) : 
  length = 6.4 ∧ width = 9 ∧ height = 5.2 ∧ volume_small_barrel = 1 →
  ⌈length * width * height / volume_small_barrel⌉ = 300 := by
  sorry

end NUMINAMATH_CALUDE_barrel_division_l3604_360425


namespace NUMINAMATH_CALUDE_points_divisible_by_ten_l3604_360471

/-- A configuration of points on a circle satisfying certain distance conditions -/
structure PointConfiguration where
  n : ℕ
  circle_length : ℕ
  distance_one : ∀ i : Fin n, ∃! j : Fin n, i ≠ j ∧ (i.val - j.val) % circle_length = 1
  distance_two : ∀ i : Fin n, ∃! j : Fin n, i ≠ j ∧ (i.val - j.val) % circle_length = 2

/-- Theorem stating that for a specific configuration, n is divisible by 10 -/
theorem points_divisible_by_ten (config : PointConfiguration) 
  (h_length : config.circle_length = 15) : 
  10 ∣ config.n :=
sorry

end NUMINAMATH_CALUDE_points_divisible_by_ten_l3604_360471


namespace NUMINAMATH_CALUDE_m_n_properties_l3604_360451

theorem m_n_properties (m n : ℤ) (hm : |m| = 1) (hn : |n| = 4) :
  (∃ k : ℤ, mn < 0 → m + n = k ∧ (k = 3 ∨ k = -3)) ∧
  (∀ x y : ℤ, |x| = 1 → |y| = 4 → x - y ≤ 5) ∧
  (∃ a b : ℤ, |a| = 1 ∧ |b| = 4 ∧ a - b = 5) :=
by sorry

end NUMINAMATH_CALUDE_m_n_properties_l3604_360451


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3604_360473

theorem arithmetic_sequence_common_difference 
  (a₁ : ℚ) 
  (aₙ : ℚ) 
  (sum : ℚ) 
  (h₁ : a₁ = 3) 
  (h₂ : aₙ = 50) 
  (h₃ : sum = 318) : 
  ∃ (n : ℕ) (d : ℚ), 
    n > 1 ∧ 
    aₙ = a₁ + (n - 1) * d ∧ 
    sum = (n / 2) * (a₁ + aₙ) ∧ 
    d = 47 / 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3604_360473


namespace NUMINAMATH_CALUDE_num_teams_is_nine_l3604_360418

/-- The number of games in a round-robin tournament with n teams -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of games played in the tournament -/
def total_games : ℕ := 36

/-- Theorem: The number of teams in the tournament is 9 -/
theorem num_teams_is_nine : ∃ (n : ℕ), n > 0 ∧ num_games n = total_games ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_num_teams_is_nine_l3604_360418


namespace NUMINAMATH_CALUDE_existence_of_sequence_l3604_360415

theorem existence_of_sequence (n : ℕ) (hn : n ≥ 2) (x : Fin n → ℝ) 
  (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  ∃ a : Fin (n + 1) → ℝ,
    (a 0 + a (Fin.last n) = 0) ∧
    (∀ i, |a i| ≤ 1) ∧
    (∀ i : Fin n, |a i.succ - a i| = x i) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_sequence_l3604_360415


namespace NUMINAMATH_CALUDE_expression_value_at_three_l3604_360490

theorem expression_value_at_three : 
  let x : ℝ := 3
  x + x * (x ^ (x - 1)) = 30 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l3604_360490


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3604_360419

theorem fraction_evaluation (x : ℝ) (h : x = 3) : (x^6 + 8*x^3 + 16) / (x^3 + 4) = 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3604_360419


namespace NUMINAMATH_CALUDE_dans_remaining_money_l3604_360455

theorem dans_remaining_money (initial_amount : ℚ) (candy_price : ℚ) (gum_price : ℚ) :
  initial_amount = 3.75 →
  candy_price = 1.25 →
  gum_price = 0.80 →
  initial_amount - (candy_price + gum_price) = 1.70 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_money_l3604_360455


namespace NUMINAMATH_CALUDE_four_thirteenths_cycle_sum_l3604_360433

/-- Represents a repeating decimal with a two-digit cycle -/
structure RepeatingDecimal where
  whole : ℕ
  cycle : ℕ × ℕ

/-- Converts a fraction to a repeating decimal -/
def fractionToRepeatingDecimal (n d : ℕ) : RepeatingDecimal :=
  sorry

/-- Extracts the cycle digits from a repeating decimal -/
def getCycleDigits (r : RepeatingDecimal) : ℕ × ℕ :=
  r.cycle

theorem four_thirteenths_cycle_sum :
  let r := fractionToRepeatingDecimal 4 13
  let (c, d) := getCycleDigits r
  c + d = 3 := by
    sorry

end NUMINAMATH_CALUDE_four_thirteenths_cycle_sum_l3604_360433


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l3604_360421

noncomputable def f (x : ℝ) := Real.exp (x - 1)

theorem intersection_point_x_coordinate 
  (A B C E : ℝ × ℝ) 
  (hA : A = (1, f 1)) 
  (hB : B = (Real.exp 3, f (Real.exp 3))) 
  (hC : C.2 = (2/3) * A.2 + (1/3) * B.2) 
  (hE : E.2 = f E.1 ∧ E.2 = C.2) :
  E.1 = Real.log ((2/3) + (1/3) * Real.exp (Real.exp 3 - 1)) + 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l3604_360421


namespace NUMINAMATH_CALUDE_solve_system_equations_solve_system_inequalities_l3604_360400

-- Part 1: System of Equations
theorem solve_system_equations :
  ∃! (x y : ℝ), x - 2*y = 1 ∧ 3*x + 4*y = 9 ∧ x = 2.2 ∧ y = 0.6 :=
by sorry

-- Part 2: System of Inequalities
theorem solve_system_inequalities :
  ∀ x : ℝ, ((x - 3) / 2 + 3 ≥ x + 1 ∧ 1 - 3*(x - 1) < 8 - x) ↔ (-2 < x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_solve_system_equations_solve_system_inequalities_l3604_360400


namespace NUMINAMATH_CALUDE_longest_tape_measure_l3604_360466

theorem longest_tape_measure (a b c : ℕ) 
  (ha : a = 600) (hb : b = 500) (hc : c = 1200) : 
  Nat.gcd a (Nat.gcd b c) = 100 := by
  sorry

end NUMINAMATH_CALUDE_longest_tape_measure_l3604_360466


namespace NUMINAMATH_CALUDE_second_coaster_speed_l3604_360460

/-- The speed of the second rollercoaster given the speeds of the other coasters and the average speed -/
theorem second_coaster_speed 
  (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h1 : x₁ = 50)
  (h3 : x₃ = 73)
  (h4 : x₄ = 70)
  (h5 : x₅ = 40)
  (avg : (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 59) :
  x₂ = 62 := by
  sorry

end NUMINAMATH_CALUDE_second_coaster_speed_l3604_360460


namespace NUMINAMATH_CALUDE_sequence_term_equality_l3604_360459

theorem sequence_term_equality (n : ℕ) : 
  2 * Real.log 5 + Real.log 3 = Real.log (4 * 19 - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_term_equality_l3604_360459


namespace NUMINAMATH_CALUDE_power_of_eight_mod_five_l3604_360464

theorem power_of_eight_mod_five : 8^2023 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_eight_mod_five_l3604_360464


namespace NUMINAMATH_CALUDE_greatest_power_of_three_l3604_360428

def p : ℕ := (List.range 35).foldl (· * ·) 1

theorem greatest_power_of_three (k : ℕ) : k ≤ 15 ∧ 3^k ∣ p ∧ ∀ m > k, ¬(3^m ∣ p) :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_l3604_360428


namespace NUMINAMATH_CALUDE_complex_magnitude_l3604_360472

theorem complex_magnitude (z : ℂ) (h : z = 4 + 3 * I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3604_360472


namespace NUMINAMATH_CALUDE_tan_22_5_identity_l3604_360453

theorem tan_22_5_identity : 
  (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_tan_22_5_identity_l3604_360453


namespace NUMINAMATH_CALUDE_unique_perfect_square_sum_diff_l3604_360469

theorem unique_perfect_square_sum_diff (a : ℕ) : 
  (∃ b : ℕ, a * a = (b + 1) * (b + 1) - b * b ∧ 
            a * a = b * b + (b + 1) * (b + 1)) ∧ 
  a * a < 20000 ↔ 
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_sum_diff_l3604_360469


namespace NUMINAMATH_CALUDE_largest_common_divisor_510_399_l3604_360437

theorem largest_common_divisor_510_399 : Nat.gcd 510 399 = 57 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_510_399_l3604_360437


namespace NUMINAMATH_CALUDE_quadratic_vertex_l3604_360412

/-- The quadratic function f(x) = 2(x-1)^2 + 5 -/
def f (x : ℝ) : ℝ := 2 * (x - 1)^2 + 5

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (1, 5)

/-- Theorem: The vertex of the quadratic function f(x) = 2(x-1)^2 + 5 is (1, 5) -/
theorem quadratic_vertex : 
  ∀ x : ℝ, f x ≥ f (vertex.1) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l3604_360412


namespace NUMINAMATH_CALUDE_prob_shortest_diagonal_15_sided_l3604_360495

/-- The number of sides in the regular polygon -/
def n : ℕ := 15

/-- The total number of diagonals in a regular n-sided polygon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in a regular n-sided polygon -/
def shortest_diagonals (n : ℕ) : ℕ := n

/-- The probability of selecting a shortest diagonal in a regular n-sided polygon -/
def prob_shortest_diagonal (n : ℕ) : ℚ :=
  shortest_diagonals n / total_diagonals n

theorem prob_shortest_diagonal_15_sided :
  prob_shortest_diagonal n = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_shortest_diagonal_15_sided_l3604_360495


namespace NUMINAMATH_CALUDE_coin_distribution_ways_l3604_360436

/-- The number of coin denominations available -/
def num_denominations : ℕ := 4

/-- The number of boys receiving coins -/
def num_boys : ℕ := 6

/-- Theorem stating the number of ways to distribute coins -/
theorem coin_distribution_ways : (num_denominations ^ num_boys : ℕ) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_ways_l3604_360436


namespace NUMINAMATH_CALUDE_exponential_continuous_l3604_360486

/-- The exponential function is continuous for any positive base -/
theorem exponential_continuous (a : ℝ) (h : a > 0) :
  Continuous (fun x => a^x) :=
by
  sorry

end NUMINAMATH_CALUDE_exponential_continuous_l3604_360486


namespace NUMINAMATH_CALUDE_cat_hunting_theorem_l3604_360439

/-- The number of birds caught during the day -/
def day_birds : ℕ := 8

/-- The number of birds caught at night -/
def night_birds : ℕ := 2 * day_birds

/-- The total number of birds caught -/
def total_birds : ℕ := 24

theorem cat_hunting_theorem : 
  day_birds + night_birds = total_birds ∧ night_birds = 2 * day_birds :=
by sorry

end NUMINAMATH_CALUDE_cat_hunting_theorem_l3604_360439


namespace NUMINAMATH_CALUDE_cube_of_product_l3604_360492

theorem cube_of_product (x y : ℝ) : (-2 * x^2 * y)^3 = -8 * x^6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_product_l3604_360492


namespace NUMINAMATH_CALUDE_decimal_73_is_four_digits_in_base_4_l3604_360474

/-- Converts a decimal number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The main theorem stating that 73 in decimal is a four-digit number in base 4 -/
theorem decimal_73_is_four_digits_in_base_4 :
  (toBase4 73).length = 4 :=
sorry

end NUMINAMATH_CALUDE_decimal_73_is_four_digits_in_base_4_l3604_360474


namespace NUMINAMATH_CALUDE_positive_expression_l3604_360440

theorem positive_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (b + c) + a * (b^2 + c^2 - b*c) > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l3604_360440


namespace NUMINAMATH_CALUDE_two_pencils_one_pen_cost_l3604_360458

-- Define the cost of a pencil and a pen
variable (pencil_cost pen_cost : ℚ)

-- Define the given conditions
axiom condition1 : 3 * pencil_cost + pen_cost = 3
axiom condition2 : 3 * pencil_cost + 4 * pen_cost = (15/2)

-- State the theorem to be proved
theorem two_pencils_one_pen_cost : 
  2 * pencil_cost + pen_cost = (5/2) := by
sorry

end NUMINAMATH_CALUDE_two_pencils_one_pen_cost_l3604_360458


namespace NUMINAMATH_CALUDE_combine_squares_simplify_expression_linear_combination_l3604_360402

-- Part 1
theorem combine_squares (a b : ℝ) :
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 := by sorry

-- Part 2
theorem simplify_expression (x y : ℝ) (h : x^2 - 2*y = 4) :
  3*x^2 - 6*y - 21 = -9 := by sorry

-- Part 3
theorem linear_combination (a b c d : ℝ) 
  (h1 : a - 5*b = 3) (h2 : 5*b - 3*c = -5) (h3 : 3*c - d = 10) :
  (a - 3*c) + (5*b - d) - (5*b - 3*c) = 8 := by sorry

end NUMINAMATH_CALUDE_combine_squares_simplify_expression_linear_combination_l3604_360402


namespace NUMINAMATH_CALUDE_smallest_n_for_quadruplets_l3604_360411

def count_quadruplets (n : ℕ) : ℕ :=
  sorry

theorem smallest_n_for_quadruplets : 
  (∃ (n : ℕ), 
    n > 0 ∧ 
    count_quadruplets n = 50000 ∧
    (∀ (a b c d : ℕ), 
      (Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 65 ∧ 
       Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = n) → 
      (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)) ∧
    (∀ (m : ℕ), m < n → 
      (count_quadruplets m ≠ 50000 ∨
       ∃ (a b c d : ℕ), 
         (Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 65 ∧ 
          Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = m) ∧
         (a = 0 ∨ b = 0 ∨ c = 0 ∨ d = 0)))) ∧
  (∀ (n : ℕ), 
    n > 0 ∧ 
    count_quadruplets n = 50000 ∧
    (∀ (a b c d : ℕ), 
      (Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 65 ∧ 
       Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = n) → 
      (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)) →
    n ≥ 4459000) ∧
  count_quadruplets 4459000 = 50000 ∧
  (∀ (a b c d : ℕ), 
    (Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 65 ∧ 
     Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 4459000) → 
    (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_quadruplets_l3604_360411


namespace NUMINAMATH_CALUDE_no_four_digit_number_divisible_by_94_sum_l3604_360447

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_two_digits (n : ℕ) : ℕ := n / 100

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem no_four_digit_number_divisible_by_94_sum :
  ¬ ∃ (n : ℕ), is_four_digit n ∧ 
    n % (first_two_digits n + last_two_digits n) = 0 ∧
    first_two_digits n + last_two_digits n = 94 := by
  sorry

end NUMINAMATH_CALUDE_no_four_digit_number_divisible_by_94_sum_l3604_360447


namespace NUMINAMATH_CALUDE_valid_pairs_l3604_360456

def is_valid_pair (a b : ℕ) : Prop :=
  (a^2 + b) % (b^2 - a) = 0 ∧ (b^2 + a) % (a^2 - b) = 0

theorem valid_pairs :
  ∀ a b : ℕ, is_valid_pair a b ↔ 
    ((a = 1 ∧ b = 2) ∨ 
     (a = 2 ∧ b = 1) ∨ 
     (a = 2 ∧ b = 2) ∨ 
     (a = 2 ∧ b = 3) ∨ 
     (a = 3 ∧ b = 2) ∨ 
     (a = 3 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l3604_360456


namespace NUMINAMATH_CALUDE_sin_80_sin_40_minus_cos_80_cos_40_l3604_360422

theorem sin_80_sin_40_minus_cos_80_cos_40 : 
  Real.sin (80 * π / 180) * Real.sin (40 * π / 180) - 
  Real.cos (80 * π / 180) * Real.cos (40 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_80_sin_40_minus_cos_80_cos_40_l3604_360422


namespace NUMINAMATH_CALUDE_existence_of_graph_with_chromatic_number_without_clique_l3604_360457

/-- A graph is a structure with vertices and an edge relation -/
structure Graph (V : Type) where
  edge : V → V → Prop

/-- The chromatic number of a graph is the minimum number of colors needed to color its vertices 
    such that no two adjacent vertices have the same color -/
def chromaticNumber (G : Graph V) : ℕ := sorry

/-- An n-clique in a graph is a complete subgraph with n vertices -/
def hasClique (G : Graph V) (n : ℕ) : Prop := sorry

theorem existence_of_graph_with_chromatic_number_without_clique :
  ∀ n : ℕ, n > 3 → ∃ (V : Type) (G : Graph V), chromaticNumber G = n ∧ ¬hasClique G n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_graph_with_chromatic_number_without_clique_l3604_360457


namespace NUMINAMATH_CALUDE_complex_subtraction_l3604_360468

theorem complex_subtraction (a b : ℂ) (h1 : a = 5 - 3*I) (h2 : b = 2 + 4*I) :
  a - 3*b = -1 - 15*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l3604_360468


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3604_360444

theorem complex_fraction_equality : (2 * Complex.I) / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3604_360444


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3s_l3604_360478

-- Define the displacement function
def s (t : ℝ) : ℝ := t^2 + 10

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_3s :
  v 3 = 6 :=
sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3s_l3604_360478


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l3604_360450

def vector_a : ℝ × ℝ := (2, 3)
def vector_b (y : ℝ) : ℝ × ℝ := (4, -1 + y)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_y_value :
  parallel vector_a (vector_b y) → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l3604_360450


namespace NUMINAMATH_CALUDE_parenthesizations_of_triple_exponent_l3604_360442

/-- Represents the number of distinct parenthesizations of 3^3^3^3 -/
def num_parenthesizations : ℕ := 5

/-- Represents the number of distinct values obtained from different parenthesizations of 3^3^3^3 -/
def num_distinct_values : ℕ := 5

/-- The expression 3^3^3^3 can be parenthesized in 5 different ways, resulting in 5 distinct values -/
theorem parenthesizations_of_triple_exponent :
  num_parenthesizations = num_distinct_values :=
by sorry

#check parenthesizations_of_triple_exponent

end NUMINAMATH_CALUDE_parenthesizations_of_triple_exponent_l3604_360442


namespace NUMINAMATH_CALUDE_shoes_price_calculation_shoes_price_proof_l3604_360403

theorem shoes_price_calculation (initial_price : ℝ) 
  (price_increase_percentage : ℝ) (discount_percentage : ℝ) : ℝ :=
  let thursday_price := initial_price * (1 + price_increase_percentage)
  let friday_price := thursday_price * (1 - discount_percentage)
  friday_price

theorem shoes_price_proof :
  shoes_price_calculation 50 0.15 0.2 = 46 := by
  sorry

end NUMINAMATH_CALUDE_shoes_price_calculation_shoes_price_proof_l3604_360403


namespace NUMINAMATH_CALUDE_chord_length_l3604_360480

-- Define the line l
def line_l (x y : ℝ) : Prop := y = -3/4 * x + 5/4

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Theorem statement
theorem chord_length :
  ∃ (chord_length : ℝ),
    (∀ x y : ℝ, line_l x y → circle_O x y → 
      chord_length = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l3604_360480


namespace NUMINAMATH_CALUDE_most_suitable_for_sample_survey_l3604_360408

/-- Represents a survey scenario -/
structure SurveyScenario where
  name : String
  quantity : Nat
  easySurvey : Bool

/-- Determines if a scenario is suitable for a sample survey -/
def suitableForSampleSurvey (scenario : SurveyScenario) : Prop :=
  scenario.quantity > 1000 ∧ ¬scenario.easySurvey

/-- The list of survey scenarios -/
def scenarios : List SurveyScenario := [
  { name := "Body temperature during H1N1", quantity := 100, easySurvey := false },
  { name := "Quality of Zongzi from Wufangzhai", quantity := 10000, easySurvey := false },
  { name := "Vision condition of classmates", quantity := 50, easySurvey := true },
  { name := "Mathematics learning in eighth grade", quantity := 200, easySurvey := true }
]

theorem most_suitable_for_sample_survey :
  ∃ (s : SurveyScenario), s ∈ scenarios ∧ 
  suitableForSampleSurvey s ∧ 
  (∀ (t : SurveyScenario), t ∈ scenarios → suitableForSampleSurvey t → s = t) :=
sorry

end NUMINAMATH_CALUDE_most_suitable_for_sample_survey_l3604_360408


namespace NUMINAMATH_CALUDE_mississippi_arrangements_l3604_360445

def word : String := "MISSISSIPPI"

def letter_counts : List (Char × Nat) := [('M', 1), ('I', 4), ('S', 4), ('P', 2)]

def total_letters : Nat := 11

def arrangements_starting_with_p : Nat := 6300

theorem mississippi_arrangements :
  (List.sum (letter_counts.map (fun p => p.2)) = total_letters) →
  (List.length letter_counts = 4) →
  (List.any letter_counts (fun p => p.1 = 'P' ∧ p.2 ≥ 1)) →
  (arrangements_starting_with_p = (Nat.factorial (total_letters - 1)) / 
    (List.prod (letter_counts.map (fun p => 
      if p.1 = 'P' then Nat.factorial (p.2 - 1) else Nat.factorial p.2)))) :=
by sorry

end NUMINAMATH_CALUDE_mississippi_arrangements_l3604_360445


namespace NUMINAMATH_CALUDE_product_sum_equation_l3604_360426

theorem product_sum_equation : (3 * 4 * 5) * (1/3 + 1/4 + 1/5 + 1) = 107 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_equation_l3604_360426


namespace NUMINAMATH_CALUDE_micahs_strawberries_l3604_360443

def strawberries_for_mom (picked : ℕ) (eaten : ℕ) : ℕ :=
  picked - eaten

theorem micahs_strawberries :
  strawberries_for_mom (2 * 12) 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_micahs_strawberries_l3604_360443


namespace NUMINAMATH_CALUDE_train_speed_problem_l3604_360410

theorem train_speed_problem (length1 length2 : Real) (crossing_time : Real) (speed1 : Real) (speed2 : Real) :
  length1 = 150 ∧ 
  length2 = 160 ∧ 
  crossing_time = 11.159107271418288 ∧
  speed1 = 60 ∧
  (length1 + length2) / crossing_time = (speed1 * 1000 / 3600) + (speed2 * 1000 / 3600) →
  speed2 = 40 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3604_360410


namespace NUMINAMATH_CALUDE_angle_measure_when_complement_is_half_supplement_l3604_360481

theorem angle_measure_when_complement_is_half_supplement :
  ∀ x : ℝ,
  (x > 0) →
  (x ≤ 180) →
  (90 - x = (180 - x) / 2) →
  x = 90 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_when_complement_is_half_supplement_l3604_360481


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3604_360465

theorem diophantine_equation_solution :
  ∀ x y z : ℕ+,
    (x + y = z ∧ x^2 * y = z^2 + 1) →
    ((x = 5 ∧ y = 2 ∧ z = 7) ∨ (x = 5 ∧ y = 13 ∧ z = 18)) :=
by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3604_360465


namespace NUMINAMATH_CALUDE_intersection_midpoint_sum_l3604_360476

/-- Given a line y = ax + b that intersects y = x^2 at two distinct points,
    if the midpoint of these points is (5, 101), then a + b = -41 -/
theorem intersection_midpoint_sum (a b : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    a * x₁ + b = x₁^2 ∧ 
    a * x₂ + b = x₂^2 ∧ 
    (x₁ + x₂) / 2 = 5 ∧ 
    (x₁^2 + x₂^2) / 2 = 101) →
  a + b = -41 := by
sorry

end NUMINAMATH_CALUDE_intersection_midpoint_sum_l3604_360476


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l3604_360414

theorem quadratic_roots_nature (a b c m n : ℝ) : 
  a ≠ 0 → c ≠ 0 →
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = m ∨ x = n) →
  m * n < 0 →
  m < abs m →
  ∀ x, c * x^2 + (m - n) * a * x - a = 0 → x < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l3604_360414


namespace NUMINAMATH_CALUDE_ice_cream_scoop_cost_l3604_360435

/-- The cost of a "Build Your Own Hot Brownie" dessert --/
structure BrownieDessert where
  brownieCost : ℝ
  syrupCost : ℝ
  nutsCost : ℝ
  iceCreamScoops : ℕ
  totalCost : ℝ

/-- The specific dessert order made by Juanita --/
def juanitaOrder : BrownieDessert where
  brownieCost := 2.50
  syrupCost := 0.50
  nutsCost := 1.50
  iceCreamScoops := 2
  totalCost := 7.00

/-- Theorem stating that each scoop of ice cream costs $1.00 --/
theorem ice_cream_scoop_cost (order : BrownieDessert) : 
  order.brownieCost = 2.50 →
  order.syrupCost = 0.50 →
  order.nutsCost = 1.50 →
  order.iceCreamScoops = 2 →
  order.totalCost = 7.00 →
  (order.totalCost - (order.brownieCost + 2 * order.syrupCost + order.nutsCost)) / order.iceCreamScoops = 1.00 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoop_cost_l3604_360435


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3604_360441

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (t m x y : ℝ) : Prop := x = t*y + m

-- Define point P
def point_P : ℝ × ℝ := (-2, 0)

-- Define the condition that l is not vertical to x-axis
def not_vertical (t : ℝ) : Prop := t ≠ 0

-- Define the bisection condition
def bisects_angle (A B : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  y₁ / (x₁ + 2) + y₂ / (x₂ + 2) = 0

-- Main theorem
theorem line_passes_through_point :
  ∀ (t m : ℝ) (A B : ℝ × ℝ),
  not_vertical t →
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  line_l t m A.1 A.2 →
  line_l t m B.1 B.2 →
  A ≠ B →
  bisects_angle A B →
  ∃ (x : ℝ), line_l t m x 0 ∧ x = 2 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3604_360441


namespace NUMINAMATH_CALUDE_largest_invertible_interval_containing_two_l3604_360448

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 8

-- State the theorem
theorem largest_invertible_interval_containing_two :
  ∃ (a : ℝ), a ≤ 2 ∧ 
  (∀ (x y : ℝ), a ≤ x ∧ x < y → g x < g y) ∧
  (∀ (b : ℝ), b < a → ¬(∀ (x y : ℝ), b ≤ x ∧ x < y → g x < g y)) :=
by sorry

end NUMINAMATH_CALUDE_largest_invertible_interval_containing_two_l3604_360448


namespace NUMINAMATH_CALUDE_f_properties_l3604_360489

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |2*x + 4| - 3

-- State the theorem
theorem f_properties (a : ℝ) (h : a ≠ -2) :
  (f a a > f a (-2)) ∧
  (∃ x y : ℝ, x < y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z ∈ Set.Ioo x y, f a z > 0) ↔
  a ∈ Set.Ioc (-5) (-7/2) ∪ Set.Ico (-1/2) 1 :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3604_360489


namespace NUMINAMATH_CALUDE_factors_of_48_l3604_360405

/-- The number of distinct positive factors of 48 is 10. -/
theorem factors_of_48 : Finset.card (Nat.divisors 48) = 10 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_48_l3604_360405


namespace NUMINAMATH_CALUDE_pencils_per_box_l3604_360485

theorem pencils_per_box (total_pencils : ℕ) (num_boxes : ℕ) 
  (h1 : total_pencils = 648) 
  (h2 : num_boxes = 162) 
  (h3 : total_pencils % num_boxes = 0) : 
  total_pencils / num_boxes = 4 := by
sorry

end NUMINAMATH_CALUDE_pencils_per_box_l3604_360485


namespace NUMINAMATH_CALUDE_max_ratio_on_unit_circle_l3604_360496

theorem max_ratio_on_unit_circle :
  let a : ℂ := Real.sqrt 17
  let b : ℂ := Complex.I * Real.sqrt 19
  (∃ (k : ℝ), k = 4/3 ∧
    ∀ (z : ℂ), Complex.abs z = 1 →
      Complex.abs (a - z) / Complex.abs (b - z) ≤ k) ∧
    ∀ (k' : ℝ), (∀ (z : ℂ), Complex.abs z = 1 →
      Complex.abs (a - z) / Complex.abs (b - z) ≤ k') →
      k' ≥ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_on_unit_circle_l3604_360496


namespace NUMINAMATH_CALUDE_number_fraction_theorem_l3604_360416

theorem number_fraction_theorem (number : ℚ) (fraction : ℚ) : 
  number = 64 →
  number = number * fraction + 40 →
  fraction = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_number_fraction_theorem_l3604_360416


namespace NUMINAMATH_CALUDE_tank_insulation_cost_l3604_360482

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Calculates the cost of insulating a rectangular tank -/
def insulationCost (l w h costPerSqFt : ℝ) : ℝ :=
  surfaceArea l w h * costPerSqFt

/-- Theorem: The cost of insulating a rectangular tank with given dimensions is $1240 -/
theorem tank_insulation_cost :
  insulationCost 5 3 2 20 = 1240 := by
  sorry

end NUMINAMATH_CALUDE_tank_insulation_cost_l3604_360482


namespace NUMINAMATH_CALUDE_bikers_meeting_time_l3604_360401

/-- The time (in minutes) it takes for two bikers to meet again at the starting point of a circular path -/
def meetingTime (t1 t2 : ℕ) : ℕ :=
  Nat.lcm t1 t2

/-- Theorem stating that two bikers with given round completion times will meet at the starting point after a specific time -/
theorem bikers_meeting_time :
  let t1 : ℕ := 12  -- Time for first biker to complete a round
  let t2 : ℕ := 18  -- Time for second biker to complete a round
  meetingTime t1 t2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_bikers_meeting_time_l3604_360401


namespace NUMINAMATH_CALUDE_quadratic_coefficients_identify_coefficients_l3604_360463

theorem quadratic_coefficients (x : ℝ) : 
  5 * x^2 + 1/2 = 6 * x ↔ 5 * x^2 + (-6) * x + 1/2 = 0 :=
by sorry

theorem identify_coefficients :
  ∃ (a b c : ℝ), (∀ x, a * x^2 + b * x + c = 0 ↔ 5 * x^2 + (-6) * x + 1/2 = 0) ∧
  a = 5 ∧ b = -6 ∧ c = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_identify_coefficients_l3604_360463


namespace NUMINAMATH_CALUDE_min_value_theorem_l3604_360470

/-- Represents a three-digit number with distinct digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  distinct : hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones
  valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The set of available digits -/
def availableDigits : Finset Nat := {5, 5, 6, 6, 6, 7, 8, 8, 9}

/-- Theorem stating the minimum value of A + B - C -/
theorem min_value_theorem (A B C : ThreeDigitNumber) 
  (h1 : A.hundreds ∈ availableDigits)
  (h2 : A.tens ∈ availableDigits)
  (h3 : A.ones ∈ availableDigits)
  (h4 : B.hundreds ∈ availableDigits)
  (h5 : B.tens ∈ availableDigits)
  (h6 : B.ones ∈ availableDigits)
  (h7 : C.hundreds ∈ availableDigits)
  (h8 : C.tens ∈ availableDigits)
  (h9 : C.ones ∈ availableDigits)
  (h10 : A.toNat + B.toNat - C.toNat ≥ 149) :
  ∃ (A' B' C' : ThreeDigitNumber),
    A'.hundreds ∈ availableDigits ∧
    A'.tens ∈ availableDigits ∧
    A'.ones ∈ availableDigits ∧
    B'.hundreds ∈ availableDigits ∧
    B'.tens ∈ availableDigits ∧
    B'.ones ∈ availableDigits ∧
    C'.hundreds ∈ availableDigits ∧
    C'.tens ∈ availableDigits ∧
    C'.ones ∈ availableDigits ∧
    A'.toNat + B'.toNat - C'.toNat = 149 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3604_360470


namespace NUMINAMATH_CALUDE_sum_of_values_l3604_360406

/-- A discrete random variable with two possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  p₁ : ℝ
  p₂ : ℝ
  h_prob_sum : p₁ + p₂ = 1
  h_prob_nonneg : 0 ≤ p₁ ∧ 0 ≤ p₂

/-- Expected value of the discrete random variable -/
def expectation (X : DiscreteRV) : ℝ := X.x₁ * X.p₁ + X.x₂ * X.p₂

/-- Variance of the discrete random variable -/
def variance (X : DiscreteRV) : ℝ :=
  X.p₁ * (X.x₁ - expectation X)^2 + X.p₂ * (X.x₂ - expectation X)^2

theorem sum_of_values (X : DiscreteRV)
  (h_p₁ : X.p₁ = 2/3)
  (h_p₂ : X.p₂ = 1/3)
  (h_order : X.x₁ < X.x₂)
  (h_expectation : expectation X = 4/9)
  (h_variance : variance X = 2) :
  X.x₁ + X.x₂ = 17/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_values_l3604_360406


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3604_360467

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (1 - 3*I) / (1 - I) → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3604_360467


namespace NUMINAMATH_CALUDE_negation_of_at_most_two_l3604_360479

theorem negation_of_at_most_two (P : ℕ → Prop) : 
  (¬ (∃ n : ℕ, P n ∧ (∀ m : ℕ, P m → m ≤ n) ∧ n ≤ 2)) ↔ 
  (∃ a b c : ℕ, P a ∧ P b ∧ P c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
sorry

end NUMINAMATH_CALUDE_negation_of_at_most_two_l3604_360479


namespace NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l3604_360423

/-- The polar coordinate equation ρ = 4sin(θ) + 2cos(θ) is equivalent to
    the rectangular coordinate equation (x-1)^2 + (y-2)^2 = 5 -/
theorem polar_to_rectangular_equivalence :
  ∀ (x y ρ θ : ℝ), 
  ρ = 4 * Real.sin θ + 2 * Real.cos θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  (x - 1)^2 + (y - 2)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l3604_360423


namespace NUMINAMATH_CALUDE_pillow_average_cost_l3604_360417

theorem pillow_average_cost (n : ℕ) (avg_cost : ℚ) (additional_cost : ℚ) :
  n = 4 →
  avg_cost = 5 →
  additional_cost = 10 →
  (n * avg_cost + additional_cost) / (n + 1) = 6 := by
sorry

end NUMINAMATH_CALUDE_pillow_average_cost_l3604_360417


namespace NUMINAMATH_CALUDE_systematic_sampling_l3604_360409

theorem systematic_sampling (n : Nat) (groups : Nat) (last_group_num : Nat) :
  n = 100 ∧ groups = 5 ∧ last_group_num = 94 →
  ∃ (interval : Nat) (first_group_num : Nat),
    interval * (groups - 1) + first_group_num = last_group_num ∧
    interval * 1 + first_group_num = 34 :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_l3604_360409


namespace NUMINAMATH_CALUDE_unused_signs_l3604_360491

theorem unused_signs (total_signs : Nat) (used_signs : Nat) (additional_codes : Nat) : 
  total_signs = 424 →
  used_signs = 422 →
  additional_codes = 1688 →
  total_signs ^ 2 - used_signs ^ 2 = additional_codes →
  total_signs - used_signs = 2 :=
by sorry

end NUMINAMATH_CALUDE_unused_signs_l3604_360491


namespace NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l3604_360461

def hyperbola_equation (x y : ℝ) : Prop := x^2 / 9 - y^2 / 7 = 1

def is_focus (F : ℝ × ℝ) (C : (ℝ × ℝ → Prop)) : Prop := sorry

def is_right_branch (P : ℝ × ℝ) (C : (ℝ × ℝ → Prop)) : Prop := sorry

def distance (P Q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_triangle_perimeter 
  (C : ℝ × ℝ → Prop)
  (F₁ F₂ P : ℝ × ℝ) :
  (∀ x y, C (x, y) ↔ hyperbola_equation x y) →
  is_focus F₁ C ∧ is_focus F₂ C →
  F₁.1 < F₂.1 →
  is_right_branch P C →
  distance P F₁ = 8 →
  distance P F₁ + distance P F₂ + distance F₁ F₂ = 18 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l3604_360461


namespace NUMINAMATH_CALUDE_center_sum_is_seven_l3604_360487

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x + 8*y - 15

/-- The center of a circle -/
def CircleCenter (h k : ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, circle x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - (6*h + 8*k - 15))

theorem center_sum_is_seven :
  ∃ h k, CircleCenter h k CircleEquation ∧ h + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_center_sum_is_seven_l3604_360487


namespace NUMINAMATH_CALUDE_expression_value_l3604_360475

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (sum_zero : x + y + z = 0) (sum_prod_nonzero : x*y + x*z + y*z ≠ 0) :
  (x^5 + y^5 + z^5) / (x*y*z * (x*y + x*z + y*z)) = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3604_360475


namespace NUMINAMATH_CALUDE_sum_of_three_squares_squared_l3604_360430

theorem sum_of_three_squares_squared (a b c : ℕ) :
  ∃ (x y z : ℕ), (a^2 + b^2 + c^2)^2 = x^2 + y^2 + z^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_squared_l3604_360430


namespace NUMINAMATH_CALUDE_lloyd_decks_required_l3604_360404

/-- Represents the number of cards in a standard deck --/
def cards_per_deck : ℕ := 52

/-- Represents the number of layers in Lloyd's house of cards --/
def num_layers : ℕ := 32

/-- Represents the number of cards per layer in Lloyd's house of cards --/
def cards_per_layer : ℕ := 26

/-- Calculates the total number of cards in the house of cards --/
def total_cards : ℕ := num_layers * cards_per_layer

/-- Theorem: The number of complete decks required for Lloyd's house of cards is 16 --/
theorem lloyd_decks_required : (total_cards / cards_per_deck) = 16 := by
  sorry

end NUMINAMATH_CALUDE_lloyd_decks_required_l3604_360404


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3604_360432

theorem negative_fraction_comparison : -5/4 < -4/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3604_360432


namespace NUMINAMATH_CALUDE_middle_school_running_average_middle_school_running_average_proof_l3604_360452

/-- The average number of minutes run per day by middle school students -/
theorem middle_school_running_average : ℝ :=
  let sixth_grade_minutes : ℝ := 14
  let seventh_grade_minutes : ℝ := 18
  let eighth_grade_minutes : ℝ := 12
  let sixth_to_seventh_ratio : ℝ := 3
  let seventh_to_eighth_ratio : ℝ := 4
  let sports_day_additional_minutes : ℝ := 4
  let days_per_week : ℝ := 7

  let sixth_grade_students : ℝ := seventh_to_eighth_ratio * sixth_to_seventh_ratio
  let seventh_grade_students : ℝ := seventh_to_eighth_ratio
  let eighth_grade_students : ℝ := 1

  let total_students : ℝ := sixth_grade_students + seventh_grade_students + eighth_grade_students

  let average_minutes_with_sports_day : ℝ :=
    (sixth_grade_students * (sixth_grade_minutes * days_per_week + sports_day_additional_minutes) +
     seventh_grade_students * (seventh_grade_minutes * days_per_week + sports_day_additional_minutes) +
     eighth_grade_students * (eighth_grade_minutes * days_per_week + sports_day_additional_minutes)) /
    (total_students * days_per_week)

  15.6

theorem middle_school_running_average_proof : 
  (middle_school_running_average : ℝ) = 15.6 := by sorry

end NUMINAMATH_CALUDE_middle_school_running_average_middle_school_running_average_proof_l3604_360452


namespace NUMINAMATH_CALUDE_special_collection_total_l3604_360438

/-- A collection of shapes consisting of circles, squares, and triangles. -/
structure ShapeCollection where
  circles : ℕ
  squares : ℕ
  triangles : ℕ

/-- The total number of shapes in the collection. -/
def ShapeCollection.total (sc : ShapeCollection) : ℕ :=
  sc.circles + sc.squares + sc.triangles

/-- A collection satisfying the given conditions. -/
def specialCollection : ShapeCollection :=
  { circles := 5, squares := 1, triangles := 9 }

theorem special_collection_total :
  (specialCollection.squares + specialCollection.triangles = 10) ∧
  (specialCollection.circles + specialCollection.triangles = 14) ∧
  (specialCollection.circles + specialCollection.squares = 6) ∧
  specialCollection.total = 15 := by
  sorry

#eval specialCollection.total

end NUMINAMATH_CALUDE_special_collection_total_l3604_360438


namespace NUMINAMATH_CALUDE_gas_purchase_l3604_360427

theorem gas_purchase (price_nc : ℝ) (amount : ℝ) : 
  price_nc > 0 →
  amount > 0 →
  price_nc * amount + (price_nc + 1) * amount = 50 →
  price_nc = 2 →
  amount = 10 := by
sorry

end NUMINAMATH_CALUDE_gas_purchase_l3604_360427


namespace NUMINAMATH_CALUDE_f_increasing_when_x_greater_than_one_l3604_360494

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 5

theorem f_increasing_when_x_greater_than_one :
  ∀ x : ℝ, x > 1 → (deriv f) x > 0 := by sorry

end NUMINAMATH_CALUDE_f_increasing_when_x_greater_than_one_l3604_360494


namespace NUMINAMATH_CALUDE_mutual_choice_exists_l3604_360434

/-- A monotonic increasing function from {1,...,n} to {1,...,n} -/
def MonotonicFunction (n : ℕ) := {f : Fin n → Fin n // ∀ i j, i ≤ j → f i ≤ f j}

/-- The theorem statement -/
theorem mutual_choice_exists (n : ℕ) (hn : n > 0) (f g : MonotonicFunction n) :
  ∃ k : Fin n, (f.val ∘ g.val) k = k :=
sorry

end NUMINAMATH_CALUDE_mutual_choice_exists_l3604_360434


namespace NUMINAMATH_CALUDE_range_of_f_l3604_360431

noncomputable def f (x : ℝ) : ℝ := 1 - x - 9 / x

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ 0 ∧ f x = y) ↔ y ≤ -5 ∨ y ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l3604_360431


namespace NUMINAMATH_CALUDE_function_equation_solution_l3604_360429

/-- Given a function f: ℝ → ℝ satisfying f(x-f(y)) = 1 - x - y for all x, y ∈ ℝ,
    prove that f(x) = 1/2 - x for all x ∈ ℝ. -/
theorem function_equation_solution (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x - f y) = 1 - x - y) : 
    ∀ x : ℝ, f x = 1/2 - x := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l3604_360429


namespace NUMINAMATH_CALUDE_jane_farm_eggs_l3604_360484

/-- Calculates the number of eggs laid per chicken per week -/
def eggs_per_chicken_per_week (num_chickens : ℕ) (price_per_dozen : ℚ) (total_revenue : ℚ) (num_weeks : ℕ) : ℚ :=
  (total_revenue / price_per_dozen * 12) / (num_chickens * num_weeks)

theorem jane_farm_eggs : 
  let num_chickens : ℕ := 10
  let price_per_dozen : ℚ := 2
  let total_revenue : ℚ := 20
  let num_weeks : ℕ := 2
  eggs_per_chicken_per_week num_chickens price_per_dozen total_revenue num_weeks = 6 := by
  sorry

#eval eggs_per_chicken_per_week 10 2 20 2

end NUMINAMATH_CALUDE_jane_farm_eggs_l3604_360484


namespace NUMINAMATH_CALUDE_cubic_function_unique_determination_l3604_360499

def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem cubic_function_unique_determination 
  (f : ℝ → ℝ) 
  (h_cubic : ∃ a b c d : ℝ, ∀ x, f x = cubic_function a b c d x) 
  (h_max : f 1 = 4 ∧ (deriv f) 1 = 0)
  (h_min : f 3 = 0 ∧ (deriv f) 3 = 0)
  (h_origin : f 0 = 0) :
  ∀ x, f x = x^3 - 6*x^2 + 9*x :=
sorry

end NUMINAMATH_CALUDE_cubic_function_unique_determination_l3604_360499


namespace NUMINAMATH_CALUDE_original_number_l3604_360462

theorem original_number (x : ℚ) (h : 1 - 1/x = 5/2) : x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l3604_360462


namespace NUMINAMATH_CALUDE_inequality_proof_l3604_360407

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a * (a + 1)) / (b + 1) + (b * (b + 1)) / (a + 1) ≥ a + b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3604_360407


namespace NUMINAMATH_CALUDE_distance_between_trees_l3604_360424

/-- Given a yard of length 150 meters with 11 trees planted at equal distances,
    including one tree at each end, the distance between consecutive trees is 15 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 150 ∧ num_trees = 11 →
  (yard_length / (num_trees - 1 : ℝ)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l3604_360424


namespace NUMINAMATH_CALUDE_base9_sequence_is_triangular_l3604_360497

/-- Definition of triangular numbers -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Definition of the sequence in base-9 -/
def base9_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => 9 * base9_sequence n + 1

/-- Theorem stating that each term in the base-9 sequence is a triangular number -/
theorem base9_sequence_is_triangular (n : ℕ) : 
  ∃ m : ℕ, base9_sequence n = triangular m := by sorry

end NUMINAMATH_CALUDE_base9_sequence_is_triangular_l3604_360497


namespace NUMINAMATH_CALUDE_alice_outfits_l3604_360488

/-- The number of different outfits Alice can create -/
def number_of_outfits (trousers shirts jackets shoes : ℕ) : ℕ :=
  trousers * shirts * jackets * shoes

/-- Theorem stating the number of outfits Alice can create with her wardrobe -/
theorem alice_outfits :
  number_of_outfits 5 8 4 2 = 320 := by
  sorry

end NUMINAMATH_CALUDE_alice_outfits_l3604_360488


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3604_360498

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3604_360498


namespace NUMINAMATH_CALUDE_company_works_four_weeks_per_month_l3604_360454

/-- Represents the company's employee and payroll information -/
structure Company where
  initial_employees : ℕ
  additional_employees : ℕ
  hourly_wage : ℚ
  hours_per_day : ℕ
  days_per_week : ℕ
  total_monthly_pay : ℚ

/-- Calculates the number of weeks worked per month -/
def weeks_per_month (c : Company) : ℚ :=
  let total_employees := c.initial_employees + c.additional_employees
  let daily_pay := c.hourly_wage * c.hours_per_day
  let weekly_pay := daily_pay * c.days_per_week
  let total_weekly_pay := weekly_pay * total_employees
  c.total_monthly_pay / total_weekly_pay

/-- Theorem stating that the company's employees work 4 weeks per month -/
theorem company_works_four_weeks_per_month :
  let c : Company := {
    initial_employees := 500,
    additional_employees := 200,
    hourly_wage := 12,
    hours_per_day := 10,
    days_per_week := 5,
    total_monthly_pay := 1680000
  }
  weeks_per_month c = 4 := by
  sorry


end NUMINAMATH_CALUDE_company_works_four_weeks_per_month_l3604_360454


namespace NUMINAMATH_CALUDE_secretary_work_time_l3604_360483

theorem secretary_work_time (x y z : ℕ) (h1 : x + y + z = 80) (h2 : 2 * x = 3 * y) (h3 : 2 * x = z) : z = 40 := by
  sorry

end NUMINAMATH_CALUDE_secretary_work_time_l3604_360483


namespace NUMINAMATH_CALUDE_alicia_tax_deduction_l3604_360420

/-- Represents Alicia's hourly wage in dollars -/
def hourly_wage : ℚ := 25

/-- Represents the local tax rate as a decimal -/
def tax_rate : ℚ := 18 / 1000

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℚ) : ℚ := dollars * 100

/-- Calculates the tax deduction in cents -/
def tax_deduction (wage : ℚ) (rate : ℚ) : ℚ :=
  dollars_to_cents (wage * rate)

/-- Theorem stating that Alicia's tax deduction is 45 cents per hour -/
theorem alicia_tax_deduction :
  tax_deduction hourly_wage tax_rate = 45 := by
  sorry

end NUMINAMATH_CALUDE_alicia_tax_deduction_l3604_360420
