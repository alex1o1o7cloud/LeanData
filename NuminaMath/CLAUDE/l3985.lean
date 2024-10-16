import Mathlib

namespace NUMINAMATH_CALUDE_x_squared_plus_inverse_squared_l3985_398529

theorem x_squared_plus_inverse_squared (x : ℝ) : x^2 - x - 1 = 0 → x^2 + 1/x^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_inverse_squared_l3985_398529


namespace NUMINAMATH_CALUDE_solution_value_l3985_398511

theorem solution_value (a : ℚ) : 
  (∃ x : ℚ, x = -2 ∧ 2 * x + 3 * a = 0) → a = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3985_398511


namespace NUMINAMATH_CALUDE_product_of_fractions_l3985_398501

theorem product_of_fractions : (2 : ℚ) / 3 * 5 / 7 * 11 / 13 = 110 / 273 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3985_398501


namespace NUMINAMATH_CALUDE_prob_different_cinemas_value_l3985_398560

/-- The number of cinemas in the city -/
def num_cinemas : ℕ := 10

/-- The number of boys going to the cinema -/
def num_boys : ℕ := 7

/-- The probability of 7 boys choosing different cinemas out of 10 cinemas -/
def prob_different_cinemas : ℚ :=
  (num_cinemas.factorial / (num_cinemas - num_boys).factorial) / num_cinemas ^ num_boys

theorem prob_different_cinemas_value : 
  prob_different_cinemas = 15120 / 250000 :=
sorry

end NUMINAMATH_CALUDE_prob_different_cinemas_value_l3985_398560


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3985_398561

theorem angle_measure_proof (x : ℝ) : 
  (180 - x = 3 * (90 - x)) → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3985_398561


namespace NUMINAMATH_CALUDE_chores_to_cartoons_ratio_l3985_398546

/-- Given that 2 hours (120 minutes) of cartoons requires 96 minutes of chores,
    prove that the ratio of chores to cartoons is 8 minutes of chores
    for every 10 minutes of cartoons. -/
theorem chores_to_cartoons_ratio :
  ∀ (cartoon_time chore_time : ℕ),
    cartoon_time = 120 →
    chore_time = 96 →
    (chore_time : ℚ) / (cartoon_time : ℚ) * 10 = 8 := by
  sorry

#check chores_to_cartoons_ratio

end NUMINAMATH_CALUDE_chores_to_cartoons_ratio_l3985_398546


namespace NUMINAMATH_CALUDE_fraction_problem_l3985_398597

theorem fraction_problem :
  ∃ (x : ℝ) (a b : ℕ),
    x > 0 ∧
    x^2 = 25 ∧
    2*x = (a / b : ℝ)*x + 9 →
    a = 1 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3985_398597


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3985_398547

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = d

theorem arithmetic_sequence_formula (a : ℕ → ℝ) (d t : ℝ) :
  arithmetic_sequence a d →
  d > 0 →
  a 1 = 1 →
  (∀ n, 2 * (a n * a (n + 1) + 1) = t * (1 + a n)) →
  ∀ n, a n = 2 * n - 1 + (-1)^n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3985_398547


namespace NUMINAMATH_CALUDE_clarinet_cost_calculation_l3985_398544

/-- The cost of items purchased at a music store -/
structure MusicStorePurchase where
  total_spent : ℝ
  songbook_cost : ℝ
  clarinet_cost : ℝ

/-- Theorem stating the cost of the clarinet given the total spent and songbook cost -/
theorem clarinet_cost_calculation (purchase : MusicStorePurchase) 
  (h1 : purchase.total_spent = 141.54)
  (h2 : purchase.songbook_cost = 11.24)
  : purchase.clarinet_cost = 130.30 := by
  sorry

end NUMINAMATH_CALUDE_clarinet_cost_calculation_l3985_398544


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3985_398537

def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {x | x < 3}

theorem intersection_of_A_and_B :
  A ∩ B = {x | -1 < x ∧ x < 3} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3985_398537


namespace NUMINAMATH_CALUDE_fraction_addition_l3985_398596

theorem fraction_addition : (7 : ℚ) / 8 + (9 : ℚ) / 12 = (13 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3985_398596


namespace NUMINAMATH_CALUDE_domino_path_count_l3985_398552

/-- The number of distinct paths from (0,0) to (m,n) on a grid -/
def grid_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- The grid dimensions -/
def grid_width : ℕ := 5
def grid_height : ℕ := 6

/-- The number of right and down steps required -/
def right_steps : ℕ := grid_width - 1
def down_steps : ℕ := grid_height - 1

theorem domino_path_count : grid_paths right_steps down_steps = 126 := by
  sorry

end NUMINAMATH_CALUDE_domino_path_count_l3985_398552


namespace NUMINAMATH_CALUDE_grid_domino_coverage_l3985_398569

/-- Represents a 5x5 grid with a square removed at (i, j) -/
structure Grid :=
  (i : Nat) (j : Nat)

/-- Predicate to check if a number is odd -/
def is_odd (n : Nat) : Prop := n % 2 = 1

/-- Predicate to check if the grid can be covered by dominoes -/
def can_cover_with_dominoes (g : Grid) : Prop :=
  is_odd g.i ∧ is_odd g.j

theorem grid_domino_coverage (g : Grid) :
  (g.i ≤ 5 ∧ g.j ≤ 5) →
  (can_cover_with_dominoes g ↔ (is_odd g.i ∧ is_odd g.j)) :=
sorry

end NUMINAMATH_CALUDE_grid_domino_coverage_l3985_398569


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l3985_398514

-- Define the displacement function
def s (t : ℝ) : ℝ := 2 * t^3

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 6 * t^2

-- Theorem statement
theorem instantaneous_velocity_at_2 :
  v 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l3985_398514


namespace NUMINAMATH_CALUDE_dart_probability_l3985_398584

/-- The probability of a dart landing in the center square of a regular hexagonal dartboard -/
theorem dart_probability (a : ℝ) (h : a > 0) : 
  let hexagon_side := a
  let square_side := a * Real.sqrt 3 / 2
  let hexagon_area := 3 * Real.sqrt 3 / 2 * a^2
  let square_area := (a * Real.sqrt 3 / 2)^2
  square_area / hexagon_area = 1 / (2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_dart_probability_l3985_398584


namespace NUMINAMATH_CALUDE_points_in_small_circle_l3985_398566

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A definition of a unit square -/
def UnitSquare : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

/-- A definition of a circle with center c and radius r -/
def Circle (c : Point) (r : ℝ) : Set Point :=
  {p : Point | (p.x - c.x)^2 + (p.y - c.y)^2 ≤ r^2}

theorem points_in_small_circle (points : Finset Point) 
  (h1 : points.card = 110) 
  (h2 : ∀ p ∈ points, p ∈ UnitSquare) :
  ∃ (c : Point) (S : Finset Point), 
    S ⊆ points ∧ 
    S.card = 4 ∧ 
    ∀ p ∈ S, p ∈ Circle c (1/8) := by
  sorry


end NUMINAMATH_CALUDE_points_in_small_circle_l3985_398566


namespace NUMINAMATH_CALUDE_profit_percentage_l3985_398506

theorem profit_percentage (P : ℝ) (P_positive : P > 0) : 
  (2 / 3 * P = 0.82 * (2 / 3 * P / 0.82)) → 
  ((P - (2 / 3 * P / 0.82)) / (2 / 3 * P / 0.82)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l3985_398506


namespace NUMINAMATH_CALUDE_canteen_seat_count_l3985_398509

/-- Represents the seating arrangements in the office canteen -/
structure CanteenSeating where
  round_tables : Nat
  rectangular_tables : Nat
  square_tables : Nat
  couches : Nat
  benches : Nat
  extra_chairs : Nat
  round_table_capacity : Nat
  rectangular_table_capacity : Nat
  square_table_capacity : Nat
  couch_capacity : Nat
  bench_capacity : Nat

/-- Calculates the total number of seats available in the canteen -/
def total_seats (s : CanteenSeating) : Nat :=
  s.round_tables * s.round_table_capacity +
  s.rectangular_tables * s.rectangular_table_capacity +
  s.square_tables * s.square_table_capacity +
  s.couches * s.couch_capacity +
  s.benches * s.bench_capacity +
  s.extra_chairs

/-- Theorem stating that the total number of seats in the given arrangement is 80 -/
theorem canteen_seat_count :
  let s : CanteenSeating := {
    round_tables := 3,
    rectangular_tables := 4,
    square_tables := 2,
    couches := 2,
    benches := 3,
    extra_chairs := 5,
    round_table_capacity := 6,
    rectangular_table_capacity := 7,
    square_table_capacity := 4,
    couch_capacity := 3,
    bench_capacity := 5
  }
  total_seats s = 80 := by
  sorry

end NUMINAMATH_CALUDE_canteen_seat_count_l3985_398509


namespace NUMINAMATH_CALUDE_m_range_when_exists_positive_root_l3985_398572

/-- The quadratic function f(x) = x^2 + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

/-- The proposition that there exists a positive x₀ such that f(x₀) < 0 -/
def exists_positive_root (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀ > 0 ∧ f m x₀ < 0

/-- Theorem stating that if there exists a positive x₀ such that f(x₀) < 0,
    then m is in the open interval (-∞, -2) -/
theorem m_range_when_exists_positive_root :
  ∀ m : ℝ, exists_positive_root m → m < -2 :=
by sorry

end NUMINAMATH_CALUDE_m_range_when_exists_positive_root_l3985_398572


namespace NUMINAMATH_CALUDE_kite_diagonal_length_l3985_398516

/-- A rectangle ABCD with a kite WXYZ inscribed -/
structure RectangleWithKite where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side BC -/
  bc : ℝ
  /-- Distance from A to W on AB -/
  aw : ℝ
  /-- Distance from C to Y on CD -/
  cy : ℝ
  /-- AB = CD = 5 -/
  h_ab : ab = 5
  /-- BC = AD = 10 -/
  h_bc : bc = 10
  /-- WX = WZ = √13 -/
  h_wx : aw ^ 2 + cy ^ 2 = 13
  /-- XY = ZY -/
  h_xy_zy : (bc - aw) ^ 2 + cy ^ 2 = (ab - cy) ^ 2 + aw ^ 2

/-- The length of XY in the kite WXYZ is √65 -/
theorem kite_diagonal_length (r : RectangleWithKite) : 
  (r.bc - r.aw) ^ 2 + r.cy ^ 2 = 65 := by
  sorry


end NUMINAMATH_CALUDE_kite_diagonal_length_l3985_398516


namespace NUMINAMATH_CALUDE_union_A_complement_B_I_l3985_398567

def I : Set ℤ := {x | x^2 < 9}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem union_A_complement_B_I : A ∪ (I \ B) = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_A_complement_B_I_l3985_398567


namespace NUMINAMATH_CALUDE_power_sum_equality_l3985_398592

theorem power_sum_equality (a b : ℕ+) (h1 : 2^(a:ℕ) = 8^(b:ℕ)) (h2 : a + 2*b = 5) :
  2^(a:ℕ) + 8^(b:ℕ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3985_398592


namespace NUMINAMATH_CALUDE_total_new_wattage_l3985_398526

def light_A_wattage : Real := 60
def light_B_wattage : Real := 40
def light_C_wattage : Real := 50
def light_D_wattage : Real := 80

def light_A_increase : Real := 0.12
def light_B_increase : Real := 0.20
def light_C_increase : Real := 0.15
def light_D_increase : Real := 0.10

def new_total_wattage : Real :=
  light_A_wattage * (1 + light_A_increase) +
  light_B_wattage * (1 + light_B_increase) +
  light_C_wattage * (1 + light_C_increase) +
  light_D_wattage * (1 + light_D_increase)

theorem total_new_wattage : new_total_wattage = 260.7 := by
  sorry

end NUMINAMATH_CALUDE_total_new_wattage_l3985_398526


namespace NUMINAMATH_CALUDE_choose_president_vice_president_l3985_398505

/-- The number of boys in the club -/
def num_boys : ℕ := 12

/-- The number of girls in the club -/
def num_girls : ℕ := 12

/-- The total number of members in the club -/
def total_members : ℕ := num_boys + num_girls

/-- The number of ways to choose a president and vice-president of opposite genders -/
def ways_to_choose : ℕ := num_boys * num_girls * 2

theorem choose_president_vice_president :
  ways_to_choose = 288 :=
by sorry

end NUMINAMATH_CALUDE_choose_president_vice_president_l3985_398505


namespace NUMINAMATH_CALUDE_intersection_point_y_coordinate_l3985_398522

theorem intersection_point_y_coordinate : ∃ (x : ℝ), 
  0 < x ∧ x < π / 2 ∧ 
  2 + 3 * Real.cos (2 * x) = 3 * Real.sqrt 3 * Real.sin x ∧
  2 + 3 * Real.cos (2 * x) = 3 := by sorry

end NUMINAMATH_CALUDE_intersection_point_y_coordinate_l3985_398522


namespace NUMINAMATH_CALUDE_expression_simplification_l3985_398508

theorem expression_simplification (x y : ℚ) 
  (hx : x = 1/2) (hy : y = 2/3) : 
  ((x - 2*y)^2 + (x - 2*y)*(x + 2*y) - 3*x*(2*x - y)) / (2*x) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3985_398508


namespace NUMINAMATH_CALUDE_magical_tree_properties_l3985_398532

structure FruitTree :=
  (bananas : Nat)
  (oranges : Nat)

inductive PickAction
  | PickOneBanana
  | PickOneOrange
  | PickTwoBananas
  | PickTwoOranges
  | PickBananaAndOrange

def applyAction (tree : FruitTree) (action : PickAction) : FruitTree :=
  match action with
  | PickAction.PickOneBanana => tree
  | PickAction.PickOneOrange => tree
  | PickAction.PickTwoBananas => 
      if tree.bananas ≥ 2 then { bananas := tree.bananas - 2, oranges := tree.oranges + 1 }
      else tree
  | PickAction.PickTwoOranges => 
      if tree.oranges ≥ 2 then { bananas := tree.bananas, oranges := tree.oranges - 1 }
      else tree
  | PickAction.PickBananaAndOrange =>
      if tree.bananas ≥ 1 && tree.oranges ≥ 1 then
        { bananas := tree.bananas, oranges := tree.oranges - 1 }
      else tree

def initialTree : FruitTree := { bananas := 15, oranges := 20 }

theorem magical_tree_properties :
  -- 1. It's possible to reach a state with exactly one fruit
  (∃ (actions : List PickAction), (actions.foldl applyAction initialTree).bananas + (actions.foldl applyAction initialTree).oranges = 1) ∧
  -- 2. If there's only one fruit left, it must be a banana
  (∀ (actions : List PickAction), 
    (actions.foldl applyAction initialTree).bananas + (actions.foldl applyAction initialTree).oranges = 1 →
    (actions.foldl applyAction initialTree).bananas = 1) ∧
  -- 3. It's impossible to reach a state with no fruits
  (∀ (actions : List PickAction), 
    (actions.foldl applyAction initialTree).bananas + (actions.foldl applyAction initialTree).oranges > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_magical_tree_properties_l3985_398532


namespace NUMINAMATH_CALUDE_propositions_truth_l3985_398520

-- Define the propositions
def proposition1 (a b : ℝ) : Prop := (a > b) → (a^2 > b^2)
def proposition2 (a b : ℝ) : Prop := (Real.log a = Real.log b) → (a = b)
def proposition3 (x y : ℝ) : Prop := (|x| = |y|) ↔ (x^2 = y^2)
def proposition4 (A B : ℝ) : Prop := (Real.sin A > Real.sin B) ↔ (A > B)

-- Theorem statement
theorem propositions_truth : 
  (∃ a b : ℝ, a > b ∧ a^2 ≤ b^2) ∧ 
  (∃ a b : ℝ, Real.log a = Real.log b ∧ a ≠ b) ∧
  (∀ x y : ℝ, (|x| = |y|) ↔ (x^2 = y^2)) ∧
  (∀ A B : ℝ, 0 < A ∧ A < π ∧ 0 < B ∧ B < π → ((Real.sin A > Real.sin B) ↔ (A > B))) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_l3985_398520


namespace NUMINAMATH_CALUDE_alices_favorite_number_l3985_398519

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem alices_favorite_number :
  ∃! n : ℕ,
    90 < n ∧ n < 150 ∧
    n % 13 = 0 ∧
    n % 4 ≠ 0 ∧
    digit_sum n % 4 = 0 ∧
    n = 143 := by sorry

end NUMINAMATH_CALUDE_alices_favorite_number_l3985_398519


namespace NUMINAMATH_CALUDE_floor_abs_negative_56_3_l3985_398503

theorem floor_abs_negative_56_3 : ⌊|(-56.3 : ℝ)|⌋ = 56 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_56_3_l3985_398503


namespace NUMINAMATH_CALUDE_particle_probability_l3985_398540

/-- Represents the probability of a particle hitting (0,0) starting from (x,y) -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (P (x-1) y + P x (y-1) + P (x-1) (y-1) + P (x-2) (y-2)) / 4

/-- The probability of hitting (0,0) starting from (5,5) is 3805/16384 -/
theorem particle_probability : P 5 5 = 3805 / 16384 := by
  sorry

end NUMINAMATH_CALUDE_particle_probability_l3985_398540


namespace NUMINAMATH_CALUDE_fraction_equality_l3985_398525

theorem fraction_equality (x : ℚ) (f : ℚ) (h1 : x = 2/3) (h2 : f * x = (64/216) * (1/x)) : f = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3985_398525


namespace NUMINAMATH_CALUDE_fixed_distance_point_l3985_398558

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given vectors a and b, if p satisfies ‖p - b‖ = 3 ‖p - a‖, 
    then p is at a fixed distance from (9/8)a - (1/8)b -/
theorem fixed_distance_point (a b p : V) 
  (h : ‖p - b‖ = 3 * ‖p - a‖) :
  ∃ (c : ℝ), ∀ (q : V), 
    (‖q - b‖ = 3 * ‖q - a‖) → 
    ‖q - ((9/8 : ℝ) • a - (1/8 : ℝ) • b)‖ = c :=
sorry

end NUMINAMATH_CALUDE_fixed_distance_point_l3985_398558


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3985_398521

theorem fraction_equivalence : (3 : ℚ) / 7 = 27 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3985_398521


namespace NUMINAMATH_CALUDE_train_length_l3985_398500

theorem train_length (platform1_length platform2_length : ℝ)
                     (platform1_time platform2_time : ℝ)
                     (h1 : platform1_length = 150)
                     (h2 : platform2_length = 250)
                     (h3 : platform1_time = 15)
                     (h4 : platform2_time = 20) :
  ∃ train_length : ℝ,
    train_length = 150 ∧
    (train_length + platform1_length) / platform1_time =
    (train_length + platform2_length) / platform2_time :=
by
  sorry


end NUMINAMATH_CALUDE_train_length_l3985_398500


namespace NUMINAMATH_CALUDE_iron_conducts_electricity_l3985_398545

-- Define the universe of discourse
variable (Object : Type)

-- Define predicates
variable (is_metal : Object → Prop)
variable (conducts_electricity : Object → Prop)

-- Define iron as a constant
variable (iron : Object)

-- Theorem statement
theorem iron_conducts_electricity 
  (all_metals_conduct : ∀ x, is_metal x → conducts_electricity x) 
  (iron_is_metal : is_metal iron) : 
  conducts_electricity iron := by
  sorry

end NUMINAMATH_CALUDE_iron_conducts_electricity_l3985_398545


namespace NUMINAMATH_CALUDE_solve_for_m_l3985_398510

theorem solve_for_m (n : ℝ) : 
  ∃ m : ℝ, 21 * (m + n) + 21 = 21 * (-m + n) + 21 ∧ m = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l3985_398510


namespace NUMINAMATH_CALUDE_customers_who_left_l3985_398598

theorem customers_who_left (initial : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 13 → new = 4 → final = 9 → initial - (initial - final + new) = 8 := by
  sorry

end NUMINAMATH_CALUDE_customers_who_left_l3985_398598


namespace NUMINAMATH_CALUDE_marble_selection_theorem_l3985_398581

theorem marble_selection_theorem (total_marbles special_marbles marbles_to_choose : ℕ) 
  (h1 : total_marbles = 18)
  (h2 : special_marbles = 6)
  (h3 : marbles_to_choose = 4) :
  (Nat.choose special_marbles 2) * (Nat.choose (total_marbles - special_marbles) 2) = 990 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_theorem_l3985_398581


namespace NUMINAMATH_CALUDE_det_scalar_multiple_l3985_398579

theorem det_scalar_multiple {a b c d : ℝ} (h : Matrix.det !![a, b; c, d] = 5) :
  Matrix.det !![3*a, 3*b; 3*c, 3*d] = 45 := by
  sorry

end NUMINAMATH_CALUDE_det_scalar_multiple_l3985_398579


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_factors_l3985_398573

theorem unique_x_with_three_prime_factors (x n : ℕ) : 
  x = 6^n + 1 →
  Odd n →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 11 * p * q) →
  (∀ r : ℕ, Prime r ∧ r ∣ x → r = 11 ∨ r = p ∨ r = q) →
  x = 7777 := by sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_factors_l3985_398573


namespace NUMINAMATH_CALUDE_not_parabola_l3985_398564

theorem not_parabola (k : ℝ) : 
  ¬ (∃ (a b c d e : ℝ), ∀ (x y : ℝ), x^2 + k*y^2 = 1 ↔ a*x^2 + b*x*y + c*y^2 + d*x + e*y = 0 ∧ b^2 = 4*a*c) :=
by sorry

end NUMINAMATH_CALUDE_not_parabola_l3985_398564


namespace NUMINAMATH_CALUDE_equation_equality_l3985_398583

theorem equation_equality (x y z : ℝ) (h1 : x ≠ y) 
  (h2 : (x^2 - y*z) / (x*(1 - y*z)) = (y^2 - x*z) / (y*(1 - x*z))) : 
  x + y + z = 1/x + 1/y + 1/z := by
sorry

end NUMINAMATH_CALUDE_equation_equality_l3985_398583


namespace NUMINAMATH_CALUDE_sum_consecutive_odd_integers_mod_18_l3985_398594

def consecutive_odd_integers (start : ℕ) (count : ℕ) : List ℕ :=
  List.range count |>.map (λ i => start + 2 * i)

theorem sum_consecutive_odd_integers_mod_18 (start : ℕ) (h : start = 11065) :
  (consecutive_odd_integers start 9).sum % 18 =
  ([1, 3, 5, 7, 9, 11, 13, 15, 17].map (λ x => x % 18)).sum % 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_consecutive_odd_integers_mod_18_l3985_398594


namespace NUMINAMATH_CALUDE_distinct_pairs_from_twelve_l3985_398562

theorem distinct_pairs_from_twelve (n : ℕ) : n = 12 → (n.choose 2 = 66) := by
  sorry

end NUMINAMATH_CALUDE_distinct_pairs_from_twelve_l3985_398562


namespace NUMINAMATH_CALUDE_probability_three_white_two_black_l3985_398590

/-- The probability of drawing exactly 3 white and 2 black balls from a box
    containing 8 white and 7 black balls, when 5 balls are drawn at random. -/
theorem probability_three_white_two_black : 
  let total_balls : ℕ := 8 + 7
  let white_balls : ℕ := 8
  let black_balls : ℕ := 7
  let drawn_balls : ℕ := 5
  let white_drawn : ℕ := 3
  let black_drawn : ℕ := 2
  let favorable_outcomes : ℕ := (Nat.choose white_balls white_drawn) * (Nat.choose black_balls black_drawn)
  let total_outcomes : ℕ := Nat.choose total_balls drawn_balls
  (favorable_outcomes : ℚ) / total_outcomes = 8 / 17 := by
sorry

end NUMINAMATH_CALUDE_probability_three_white_two_black_l3985_398590


namespace NUMINAMATH_CALUDE_f_monotonicity_and_range_l3985_398557

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 5

theorem f_monotonicity_and_range :
  (∀ x y, -1 < x ∧ x < y → f x < f y) ∧
  (∀ x y, x < y ∧ y < -1 → f y < f x) ∧
  (∀ z ∈ Set.Icc 0 1, 5 ≤ f z ∧ f z ≤ Real.exp 1 + 5) ∧
  (f 0 = 5) ∧
  (f 1 = Real.exp 1 + 5) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_range_l3985_398557


namespace NUMINAMATH_CALUDE_smallest_positive_solution_tan_cos_l3985_398587

theorem smallest_positive_solution_tan_cos (x : ℝ) : 
  (x > 0 ∧ x = Real.pi / 8 ∧ Real.tan (2 * x) + Real.tan (4 * x) = Real.cos (2 * x)) ∧
  (∀ y : ℝ, y > 0 ∧ y < x → Real.tan (2 * y) + Real.tan (4 * y) ≠ Real.cos (2 * y)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_tan_cos_l3985_398587


namespace NUMINAMATH_CALUDE_right_triangle_area_divisibility_l3985_398539

theorem right_triangle_area_divisibility (a b c : ℕ) : 
  a^2 + b^2 = c^2 → -- Pythagorean theorem
  c % 5 ≠ 0 → -- hypotenuse not divisible by 5
  ∃ k : ℕ, a * b = 20 * k -- area is divisible by 10
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_divisibility_l3985_398539


namespace NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l3985_398582

/-- A point (x, y) in the coordinate plane that is equally distant from the x-axis, y-axis, 
    line x + 2y = 4, and line y = 2x has x-coordinate equal to -4 / (√5 - 7) -/
theorem equidistant_point_x_coordinate (x y : ℝ) : 
  (abs x = abs y) ∧ 
  (abs x = abs (x + 2*y - 4) / Real.sqrt 5) ∧
  (abs x = abs (y - 2*x) / Real.sqrt 5) →
  x = -4 / (Real.sqrt 5 - 7) := by
sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l3985_398582


namespace NUMINAMATH_CALUDE_expression_simplification_l3985_398536

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 2) : 
  ((m^2 - 9) / (m^2 - 6*m + 9) - 3 / (m - 3)) / (m^2 / (m - 3)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3985_398536


namespace NUMINAMATH_CALUDE_stone_volume_l3985_398517

/-- The volume of a stone submerged in a cuboid-shaped container -/
theorem stone_volume (width length initial_height final_height : ℝ) 
  (hw : width = 15) 
  (hl : length = 20) 
  (hi : initial_height = 10) 
  (hf : final_height = 15) : 
  (final_height - initial_height) * width * length = 1500 := by
  sorry

end NUMINAMATH_CALUDE_stone_volume_l3985_398517


namespace NUMINAMATH_CALUDE_hummus_servings_thomas_hummus_servings_l3985_398556

/-- Calculates the number of servings of hummus Thomas is making -/
theorem hummus_servings (recipe_cup : ℕ) (can_ounces : ℕ) (cup_ounces : ℕ) (cans_bought : ℕ) : ℕ :=
  let total_ounces := can_ounces * cans_bought
  let servings := total_ounces / cup_ounces
  servings

/-- Proves that Thomas is making 21 servings of hummus -/
theorem thomas_hummus_servings :
  hummus_servings 1 16 6 8 = 21 := by
  sorry

end NUMINAMATH_CALUDE_hummus_servings_thomas_hummus_servings_l3985_398556


namespace NUMINAMATH_CALUDE_problem_solution_l3985_398563

theorem problem_solution :
  ∀ a b : ℕ,
  a + b = 54 →
  Nat.lcm a b - Nat.gcd a b = 114 →
  ((a = 24 ∧ b = 30) ∨ (a = 30 ∧ b = 24)) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3985_398563


namespace NUMINAMATH_CALUDE_root_implies_m_value_l3985_398591

theorem root_implies_m_value (m : ℝ) : 
  (2^2 - m*2 + 2 = 0) → m = 3 := by
sorry

end NUMINAMATH_CALUDE_root_implies_m_value_l3985_398591


namespace NUMINAMATH_CALUDE_callum_points_l3985_398568

theorem callum_points (total_matches : ℕ) (krishna_win_ratio : ℚ) (points_per_win : ℕ) : 
  total_matches = 8 →
  krishna_win_ratio = 3/4 →
  points_per_win = 10 →
  (total_matches - (krishna_win_ratio * total_matches).num) * points_per_win = 20 := by
  sorry

end NUMINAMATH_CALUDE_callum_points_l3985_398568


namespace NUMINAMATH_CALUDE_b_fraction_of_a_and_c_l3985_398504

def total_amount : ℕ := 1800

def a_share : ℕ := 600

theorem b_fraction_of_a_and_c (b_share c_share : ℕ) 
  (h1 : a_share = (2 : ℕ) * (b_share + c_share) / 5)
  (h2 : total_amount = a_share + b_share + c_share) :
  b_share * 6 = a_share + c_share :=
by sorry

end NUMINAMATH_CALUDE_b_fraction_of_a_and_c_l3985_398504


namespace NUMINAMATH_CALUDE_negation_equivalence_l3985_398553

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 1 ∧ x₀^2 - x₀ + 2016 > 0) ↔
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 - x + 2016 ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3985_398553


namespace NUMINAMATH_CALUDE_johns_piggy_bank_l3985_398577

theorem johns_piggy_bank (total_coins quarters dimes nickels : ℕ) : 
  total_coins = 63 →
  quarters = 22 →
  dimes = quarters + 3 →
  total_coins = quarters + dimes + nickels →
  quarters - nickels = 6 :=
by sorry

end NUMINAMATH_CALUDE_johns_piggy_bank_l3985_398577


namespace NUMINAMATH_CALUDE_product_of_constrained_values_l3985_398555

theorem product_of_constrained_values (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) : 
  a * b = 10 := by
sorry

end NUMINAMATH_CALUDE_product_of_constrained_values_l3985_398555


namespace NUMINAMATH_CALUDE_unique_three_digit_sum_27_l3985_398550

/-- A three-digit number is a natural number between 100 and 999 inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

/-- The main theorem: there is exactly one three-digit number whose digits sum to 27 -/
theorem unique_three_digit_sum_27 : ∃! n : ℕ, ThreeDigitNumber n ∧ sumOfDigits n = 27 := by
  sorry


end NUMINAMATH_CALUDE_unique_three_digit_sum_27_l3985_398550


namespace NUMINAMATH_CALUDE_ball_in_hole_within_six_bounces_l3985_398574

/-- Represents a point on the table -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hole on the table -/
structure Hole where
  location : Point

/-- Represents a rectangular table with holes -/
structure Table where
  length : ℝ
  width : ℝ
  holes : List Hole

/-- Represents a ball's trajectory -/
structure Trajectory where
  start : Point
  bounces : List Point

/-- Function to check if a trajectory ends in a hole within n bounces -/
def endsInHole (traj : Trajectory) (table : Table) (n : ℕ) : Prop :=
  ∃ (h : Hole), h ∈ table.holes ∧ traj.bounces.length ≤ n ∧ traj.bounces.getLast? = some h.location

/-- The main theorem -/
theorem ball_in_hole_within_six_bounces 
  (table : Table) 
  (a b c : Point) : 
  table.length = 8 ∧ 
  table.width = 5 ∧ 
  table.holes.length = 4 →
  ∃ (start : Point) (traj : Trajectory), 
    (start = a ∨ start = b ∨ start = c) ∧
    traj.start = start ∧
    endsInHole traj table 6 :=
sorry

end NUMINAMATH_CALUDE_ball_in_hole_within_six_bounces_l3985_398574


namespace NUMINAMATH_CALUDE_sum_70_terms_is_negative_350_l3985_398513

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference

/-- Sum of first n terms of an arithmetic progression -/
def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a + (n - 1 : ℚ) * ap.d)

/-- Theorem: For an arithmetic progression with specific properties, 
    the sum of its first 70 terms is -350 -/
theorem sum_70_terms_is_negative_350 
  (ap : ArithmeticProgression)
  (h1 : sum_n_terms ap 20 = 200)
  (h2 : sum_n_terms ap 50 = 50) :
  sum_n_terms ap 70 = -350 := by
  sorry

end NUMINAMATH_CALUDE_sum_70_terms_is_negative_350_l3985_398513


namespace NUMINAMATH_CALUDE_cistern_emptied_in_8_minutes_l3985_398507

/-- Given a pipe that can empty 2/3 of a cistern in 10 minutes,
    this function calculates the part of the cistern that will be empty in t minutes. -/
def cisternEmptied (t : ℚ) : ℚ :=
  (2/3) * (t / 10)

/-- Theorem stating that the part of the cistern emptied in 8 minutes is 8/15. -/
theorem cistern_emptied_in_8_minutes :
  cisternEmptied 8 = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_cistern_emptied_in_8_minutes_l3985_398507


namespace NUMINAMATH_CALUDE_cyclic_wins_count_l3985_398548

/-- Represents a round-robin tournament. -/
structure Tournament where
  /-- The number of teams in the tournament. -/
  num_teams : ℕ
  /-- The number of wins for each team. -/
  wins_per_team : ℕ
  /-- The number of losses for each team. -/
  losses_per_team : ℕ
  /-- No ties in the tournament. -/
  no_ties : wins_per_team + losses_per_team = num_teams - 1

/-- The number of sets of three teams {A, B, C} where A beat B, B beat C, and C beat A. -/
def cyclic_wins (t : Tournament) : ℕ := sorry

/-- The main theorem stating the number of cyclic win sets in the given tournament. -/
theorem cyclic_wins_count (t : Tournament) 
  (h1 : t.num_teams = 21)
  (h2 : t.wins_per_team = 10)
  (h3 : t.losses_per_team = 10) :
  cyclic_wins t = 385 := by sorry

end NUMINAMATH_CALUDE_cyclic_wins_count_l3985_398548


namespace NUMINAMATH_CALUDE_min_value_of_f_l3985_398593

open Real

noncomputable def f (x : ℝ) := 2 * x - log x

theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (exp 1) ∧
  (∀ (y : ℝ), y ∈ Set.Ioo 0 (exp 1) → f y ≥ f x) ∧
  f x = 1 + log 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3985_398593


namespace NUMINAMATH_CALUDE_max_sum_with_reciprocal_constraint_l3985_398586

theorem max_sum_with_reciprocal_constraint (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_eq : 1/a + 9/b = 1) : 
  a + b ≤ 16 := by
sorry

end NUMINAMATH_CALUDE_max_sum_with_reciprocal_constraint_l3985_398586


namespace NUMINAMATH_CALUDE_circle_area_tripled_radius_l3985_398571

theorem circle_area_tripled_radius (r : ℝ) (hr : r > 0) :
  let A := π * r^2
  let A' := π * (3*r)^2
  A' = 9 * A ∧ A' ≠ 3 * A :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_tripled_radius_l3985_398571


namespace NUMINAMATH_CALUDE_probability_smaller_triangle_l3985_398551

/-- The probability that a randomly chosen point in a right triangle
    forms a smaller triangle with area less than one-third of the original -/
theorem probability_smaller_triangle (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let triangle_area := a * b / 2
  let probability := (a * (b / 3)) / (2 * triangle_area)
  probability = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_smaller_triangle_l3985_398551


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l3985_398502

/-- Given a rectangle PQRS with width w and height h, and three congruent triangles
    STU, UVW, and WXR inscribed in the rectangle such that SU = UW = WR = w/3,
    prove that the total area of the three triangles is 1/2 of the rectangle's area. -/
theorem shaded_area_fraction (w h : ℝ) (hw : w > 0) (hh : h > 0) :
  let rectangle_area := w * h
  let triangle_base := w / 3
  let triangle_height := h
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let total_shaded_area := 3 * triangle_area
  total_shaded_area = (1 / 2) * rectangle_area := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l3985_398502


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3985_398580

/-- Given that the solution set of ax^2 + bx + c < 0 is (-∞, -1) ∪ (1/2, +∞),
    prove that the solution set of cx^2 - bx + a < 0 is (-2, 1) -/
theorem solution_set_inequality (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + b*x + c < 0 ↔ x < -1 ∨ x > 1/2) →
  (∀ x : ℝ, c*x^2 - b*x + a < 0 ↔ -2 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3985_398580


namespace NUMINAMATH_CALUDE_three_digit_number_divisibility_l3985_398512

theorem three_digit_number_divisibility (a b c : Nat) : 
  a ≠ 0 ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ a + b + c = 7 →
  (100 * a + 10 * b + c) % 7 = 0 ↔ b = c :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_divisibility_l3985_398512


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3985_398559

theorem complex_equation_solution (z : ℂ) :
  (Complex.I * 3 + Real.sqrt 3) * z = Complex.I * 3 →
  z = Complex.mk (3 / 4) (Real.sqrt 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3985_398559


namespace NUMINAMATH_CALUDE_perpendicular_construction_l3985_398534

-- Define the plane
structure Plane :=
  (Point : Type)
  (Line : Type)
  (on_line : Point → Line → Prop)
  (not_on_line : Point → Line → Prop)
  (draw_line : Point → Point → Line)
  (draw_perpendicular : Point → Line → Line)

-- Define the theorem
theorem perpendicular_construction 
  (P : Plane) (A : P.Point) (l : P.Line) (h : P.not_on_line A l) :
  ∃ (m : P.Line), P.on_line A m ∧ ∀ (X : P.Point), P.on_line X l → P.on_line X m → 
    ∃ (n : P.Line), P.on_line X n ∧ (∀ (Y : P.Point), P.on_line Y n → P.on_line Y m → Y = X) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_construction_l3985_398534


namespace NUMINAMATH_CALUDE_estate_division_valid_l3985_398535

/-- Represents the estate division problem in Ancient Rome --/
structure EstateDivision where
  total_estate : ℕ
  son_share : ℕ
  daughter_share : ℕ
  wife_share : ℕ

/-- Checks if the given division is valid according to the problem constraints --/
def is_valid_division (d : EstateDivision) : Prop :=
  d.total_estate = 210 ∧
  d.son_share + d.daughter_share + d.wife_share = d.total_estate ∧
  d.son_share > d.daughter_share ∧
  d.son_share > d.wife_share ∧
  7 * d.son_share = 4 * d.total_estate ∧
  7 * d.daughter_share = d.total_estate ∧
  7 * d.wife_share = 2 * d.total_estate

/-- The proposed solution satisfies the constraints of the problem --/
theorem estate_division_valid : 
  is_valid_division ⟨210, 120, 30, 60⟩ := by
  sorry

#check estate_division_valid

end NUMINAMATH_CALUDE_estate_division_valid_l3985_398535


namespace NUMINAMATH_CALUDE_curve_transformation_l3985_398531

theorem curve_transformation (x : ℝ) : 
  Real.sin (2 * x) = Real.sin (2 * (x + π / 8) + π / 4) := by sorry

end NUMINAMATH_CALUDE_curve_transformation_l3985_398531


namespace NUMINAMATH_CALUDE_floor_abs_sum_equals_eleven_l3985_398543

theorem floor_abs_sum_equals_eleven :
  ⌊|(-5.7 : ℝ)|⌋ + |⌊(-5.7 : ℝ)⌋| = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_equals_eleven_l3985_398543


namespace NUMINAMATH_CALUDE_unique_solution_l3985_398523

-- Define the equation
def equation (p x : ℝ) : Prop := x + 1 = Real.sqrt (p * x)

-- Define the conditions
def conditions (p x : ℝ) : Prop := p * x ≥ 0 ∧ x + 1 ≥ 0

-- Theorem statement
theorem unique_solution (p : ℝ) :
  (∃! x, equation p x ∧ conditions p x) ↔ (p = 4 ∨ p ≤ 0) := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3985_398523


namespace NUMINAMATH_CALUDE_line_increase_l3985_398595

/-- Given a line in the Cartesian plane where an increase of 2 units in x
    corresponds to an increase of 5 units in y, prove that an increase of 8 units
    in x will result in an increase of 20 units in y. -/
theorem line_increase (f : ℝ → ℝ) (h : ∀ x, f (x + 2) - f x = 5) :
  ∀ x, f (x + 8) - f x = 20 := by
  sorry

end NUMINAMATH_CALUDE_line_increase_l3985_398595


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l3985_398541

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 1) * x + 1

/-- Theorem corresponding to part (1) of the problem -/
theorem part_one (a : ℝ) : 
  (∀ x, f a x ≥ 0) ↔ -3 ≤ a ∧ a ≤ 1 := by sorry

/-- Theorem corresponding to part (2) of the problem -/
theorem part_two (a b : ℝ) :
  (∃ b, ∀ x, f a x < 0 ↔ b < x ∧ x < 2) ↔ a = 3/2 ∧ b = 1/2 := by sorry

/-- Theorem corresponding to part (3) of the problem -/
theorem part_three (a : ℝ) :
  ((∀ x, f a x ≤ 0) ∧ (∀ x, 0 ≤ x ∧ x ≤ 1 → f a x > 0)) ↔ a < 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l3985_398541


namespace NUMINAMATH_CALUDE_angle_not_necessarily_alpha_minus_beta_l3985_398565

noncomputable section

-- Define the vectors a and b
def a (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
def b (β : ℝ) : ℝ × ℝ := (Real.cos β, Real.sin β)

-- Define the angle between two vectors
def angle_between (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

-- Theorem statement
theorem angle_not_necessarily_alpha_minus_beta (α β : ℝ) :
  ¬ ∀ α β, angle_between (a α) (b β) = α - β :=
sorry

end

end NUMINAMATH_CALUDE_angle_not_necessarily_alpha_minus_beta_l3985_398565


namespace NUMINAMATH_CALUDE_room_length_proof_l3985_398554

theorem room_length_proof (width : ℝ) (area_covered : ℝ) (area_needed : ℝ) :
  width = 15 →
  area_covered = 16 →
  area_needed = 149 →
  (area_covered + area_needed) / width = 11 := by
  sorry

end NUMINAMATH_CALUDE_room_length_proof_l3985_398554


namespace NUMINAMATH_CALUDE_people_studying_cooking_and_weaving_l3985_398530

/-- Represents the number of people in various curriculum combinations -/
structure CurriculumParticipation where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  cookingAndYoga : ℕ
  allCurriculums : ℕ

/-- Theorem stating the number of people studying both cooking and weaving -/
theorem people_studying_cooking_and_weaving 
  (cp : CurriculumParticipation)
  (h1 : cp.yoga = 35)
  (h2 : cp.cooking = 20)
  (h3 : cp.weaving = 15)
  (h4 : cp.cookingOnly = 7)
  (h5 : cp.cookingAndYoga = 5)
  (h6 : cp.allCurriculums = 3) :
  ∃ n : ℕ, n = cp.cooking - cp.cookingOnly - (cp.cookingAndYoga - cp.allCurriculums) - cp.allCurriculums ∧ n = 8 := by
  sorry

#check people_studying_cooking_and_weaving

end NUMINAMATH_CALUDE_people_studying_cooking_and_weaving_l3985_398530


namespace NUMINAMATH_CALUDE_expression_simplification_expression_evaluation_l3985_398578

theorem expression_simplification (x y : ℝ) (h : y ≠ 0) :
  ((2*x + y) * (2*x - y) - (2*x - 3*y)^2) / (-2*y) = -6*x + 5*y := by
  sorry

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := -2
  ((2*x + y) * (2*x - y) - (2*x - 3*y)^2) / (-2*y) = -16 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_expression_evaluation_l3985_398578


namespace NUMINAMATH_CALUDE_f_properties_l3985_398599

open Real

noncomputable def f (x : ℝ) : ℝ := log x - x^2 + x

theorem f_properties :
  (∃ (x_max : ℝ), x_max > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≤ f x_max ∧ f x_max = 0) ∧
  (∀ (a : ℝ), a ≥ 2 → ∀ (x : ℝ), x > 0 → f x < (a/2 - 1) * x^2 + a * x - 1) ∧
  (∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 → 
    f x₁ + f x₂ + 2 * (x₁^2 + x₂^2) + x₁ * x₂ = 0 → 
    x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3985_398599


namespace NUMINAMATH_CALUDE_vector_magnitude_l3985_398533

theorem vector_magnitude (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  (a.1 * b.1 + a.2 * b.2 = 5) →
  ((a.1 - b.1)^2 + (a.2 - b.2)^2 = 20) →
  (b.1^2 + b.2^2 = 25) := by
    sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3985_398533


namespace NUMINAMATH_CALUDE_cellCount_after_8_days_l3985_398576

/-- The number of cells in a colony after a given number of days, 
    with specific growth and toxin conditions. -/
def cellCount (initialCells : ℕ) (days : ℕ) : ℕ :=
  let growthPeriods := days / 2
  let afterGrowth := initialCells * 3^growthPeriods
  if days ≥ 6 then
    (afterGrowth / 2 + if afterGrowth % 2 = 0 then 0 else 1) * 3^((days - 6) / 2)
  else
    afterGrowth

theorem cellCount_after_8_days : 
  cellCount 5 8 = 201 := by sorry

end NUMINAMATH_CALUDE_cellCount_after_8_days_l3985_398576


namespace NUMINAMATH_CALUDE_perfect_squares_divisibility_l3985_398538

theorem perfect_squares_divisibility (a b : ℕ+) :
  (∃ (S : Set (ℕ+ × ℕ+)), Set.Infinite S ∧
    ∀ (p : ℕ+ × ℕ+), p ∈ S →
      ∃ (k l : ℕ+), (p.1.val ^ 2 + a.val * p.2.val + b.val = k.val ^ 2) ∧
                    (p.2.val ^ 2 + a.val * p.1.val + b.val = l.val ^ 2)) →
  a.val ∣ (2 * b.val) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_divisibility_l3985_398538


namespace NUMINAMATH_CALUDE_problem_statement_l3985_398585

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = a * b) :
  ((a - 1) * (b - 1) = 1) ∧
  (∀ a b, a > 0 → b > 0 → a + b = a * b → a + 4 * b ≥ 9) ∧
  (∀ a b, a > 0 → b > 0 → a + b = a * b → 1 / a^2 + 2 / b^2 ≥ 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3985_398585


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l3985_398527

theorem cube_surface_area_increase :
  ∀ s : ℝ, s > 0 →
  let original_surface_area := 6 * s^2
  let new_edge_length := 1.4 * s
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area = 0.96 :=
by
  sorry

#check cube_surface_area_increase

end NUMINAMATH_CALUDE_cube_surface_area_increase_l3985_398527


namespace NUMINAMATH_CALUDE_fraction_sum_zero_l3985_398542

theorem fraction_sum_zero (a b : ℝ) (h : a ≠ b) : 
  1 / (a - b) + 1 / (b - a) = 0 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_zero_l3985_398542


namespace NUMINAMATH_CALUDE_total_leaves_count_l3985_398528

/-- The number of pots of basil -/
def basil_pots : ℕ := 3

/-- The number of pots of rosemary -/
def rosemary_pots : ℕ := 9

/-- The number of pots of thyme -/
def thyme_pots : ℕ := 6

/-- The number of leaves per basil plant -/
def basil_leaves : ℕ := 4

/-- The number of leaves per rosemary plant -/
def rosemary_leaves : ℕ := 18

/-- The number of leaves per thyme plant -/
def thyme_leaves : ℕ := 30

/-- The total number of leaves from all plants -/
def total_leaves : ℕ := basil_pots * basil_leaves + rosemary_pots * rosemary_leaves + thyme_pots * thyme_leaves

theorem total_leaves_count : total_leaves = 354 := by
  sorry

end NUMINAMATH_CALUDE_total_leaves_count_l3985_398528


namespace NUMINAMATH_CALUDE_mike_initial_cards_l3985_398515

theorem mike_initial_cards (sold : ℕ) (current : ℕ) (h1 : sold = 13) (h2 : current = 74) :
  current + sold = 87 := by
  sorry

end NUMINAMATH_CALUDE_mike_initial_cards_l3985_398515


namespace NUMINAMATH_CALUDE_power_equality_l3985_398524

theorem power_equality (p : ℕ) (h : (81 : ℕ)^10 = 3^p) : p = 40 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3985_398524


namespace NUMINAMATH_CALUDE_product_inequality_l3985_398518

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3985_398518


namespace NUMINAMATH_CALUDE_fourth_number_proof_l3985_398588

theorem fourth_number_proof (numbers : Fin 6 → ℝ) 
  (avg_all : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5) / 6 = 30)
  (avg_first_four : (numbers 0 + numbers 1 + numbers 2 + numbers 3) / 4 = 25)
  (avg_last_three : (numbers 3 + numbers 4 + numbers 5) / 3 = 35) :
  numbers 3 = 25 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_proof_l3985_398588


namespace NUMINAMATH_CALUDE_l_shaped_room_flooring_cost_l3985_398575

/-- Represents the dimensions of a rectangular room section -/
structure RoomSection where
  length : ℝ
  width : ℝ

/-- Calculates the total cost of replacing flooring in an L-shaped room -/
def total_flooring_cost (section1 section2 : RoomSection) (removal_cost per_sqft_cost : ℝ) : ℝ :=
  let total_area := section1.length * section1.width + section2.length * section2.width
  removal_cost + total_area * per_sqft_cost

/-- Theorem: The total cost to replace the floor in the given L-shaped room is $150 -/
theorem l_shaped_room_flooring_cost :
  let section1 : RoomSection := ⟨8, 7⟩
  let section2 : RoomSection := ⟨6, 4⟩
  let removal_cost : ℝ := 50
  let per_sqft_cost : ℝ := 1.25
  total_flooring_cost section1 section2 removal_cost per_sqft_cost = 150 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_room_flooring_cost_l3985_398575


namespace NUMINAMATH_CALUDE_perpendicular_line_through_circle_l3985_398549

/-- Given a circle C and a line l in polar coordinates, 
    this theorem proves the equation of a line passing through C 
    and perpendicular to l. -/
theorem perpendicular_line_through_circle 
  (C : ℝ → ℝ) 
  (l : ℝ → ℝ → ℝ) 
  (h_C : ∀ θ, C θ = 2 * Real.cos θ) 
  (h_l : ∀ ρ θ, l ρ θ = ρ * Real.cos θ - ρ * Real.sin θ - 4) :
  ∃ f : ℝ → ℝ → ℝ, 
    (∀ ρ θ, f ρ θ = ρ * (Real.cos θ + Real.sin θ) - 1) ∧
    (∃ θ₀, C θ₀ = f (C θ₀) θ₀) ∧
    (∀ ρ₁ θ₁ ρ₂ θ₂, 
      l ρ₁ θ₁ = 0 → l ρ₂ θ₂ = 0 → f ρ₁ θ₁ = 0 → f ρ₂ θ₂ = 0 →
      (ρ₁ * Real.cos θ₁ - ρ₂ * Real.cos θ₂) * (ρ₁ * Real.sin θ₁ - ρ₂ * Real.sin θ₂) = 
      -(ρ₁ * Real.cos θ₁ - ρ₂ * Real.cos θ₂) * (ρ₁ * Real.sin θ₁ - ρ₂ * Real.sin θ₂)) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_circle_l3985_398549


namespace NUMINAMATH_CALUDE_solution_system1_solution_system2_l3985_398570

-- System 1
def system1 (x y : ℝ) : Prop :=
  3 * (x - 1) = y + 5 ∧ 5 * (y - 1) = 3 * (x + 5)

-- System 2
def system2 (x y a : ℝ) : Prop :=
  2 * x + 4 * y = a ∧ 7 * x - 2 * y = 3 * a

theorem solution_system1 :
  ∃ x y : ℝ, system1 x y ∧ x = 5 ∧ y = 7 := by sorry

theorem solution_system2 :
  ∀ a : ℝ, ∃ x y : ℝ, system2 x y a ∧ x = 7 / 16 * a ∧ y = 1 / 32 * a := by sorry

end NUMINAMATH_CALUDE_solution_system1_solution_system2_l3985_398570


namespace NUMINAMATH_CALUDE_city_population_ratio_l3985_398589

theorem city_population_ratio :
  ∀ (pop_X pop_Y pop_Z : ℕ),
    pop_X = 3 * pop_Y →
    pop_Y = 2 * pop_Z →
    pop_X / pop_Z = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_city_population_ratio_l3985_398589
