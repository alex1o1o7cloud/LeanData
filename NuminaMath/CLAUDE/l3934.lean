import Mathlib

namespace NUMINAMATH_CALUDE_max_point_difference_is_n_l3934_393496

/-- Represents a hockey tournament with n teams -/
structure HockeyTournament where
  n : ℕ  -- number of teams
  n_pos : 0 < n  -- n is positive

/-- The maximum point difference between consecutively ranked teams in a hockey tournament -/
def maxPointDifference (tournament : HockeyTournament) : ℕ :=
  tournament.n

/-- Theorem: The maximum point difference between consecutively ranked teams is n -/
theorem max_point_difference_is_n (tournament : HockeyTournament) :
  maxPointDifference tournament = tournament.n := by
  sorry

end NUMINAMATH_CALUDE_max_point_difference_is_n_l3934_393496


namespace NUMINAMATH_CALUDE_regular_hexagon_diagonals_l3934_393450

/-- Regular hexagon with side length, shortest diagonal, and longest diagonal -/
structure RegularHexagon where
  a : ℝ  -- side length
  b : ℝ  -- shortest diagonal
  d : ℝ  -- longest diagonal

/-- Theorem: In a regular hexagon, the shortest diagonal is √3 times the side length,
    and the longest diagonal is 4/√3 times the side length -/
theorem regular_hexagon_diagonals (h : RegularHexagon) :
  h.b = Real.sqrt 3 * h.a ∧ h.d = (4 * h.a) / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_regular_hexagon_diagonals_l3934_393450


namespace NUMINAMATH_CALUDE_line_intersects_unit_circle_l3934_393462

theorem line_intersects_unit_circle 
  (a b : ℝ) (θ : ℝ) (h_neq : a ≠ b) 
  (h_a : a^2 * Real.sin θ + a * Real.cos θ - Real.pi/4 = 0)
  (h_b : b^2 * Real.sin θ + b * Real.cos θ - Real.pi/4 = 0) : 
  ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ (b + a) * x - y - a * b = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_unit_circle_l3934_393462


namespace NUMINAMATH_CALUDE_good_carrots_count_l3934_393485

/-- The number of good carrots given the number of carrots picked by Faye and her mom, and the number of bad carrots. -/
def goodCarrots (fayeCarrots momCarrots badCarrots : ℕ) : ℕ :=
  fayeCarrots + momCarrots - badCarrots

/-- Theorem stating that the number of good carrots is 12 given the problem conditions. -/
theorem good_carrots_count : goodCarrots 23 5 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_good_carrots_count_l3934_393485


namespace NUMINAMATH_CALUDE_speed_ratio_l3934_393445

/-- The speed of runner A in meters per hour -/
def speed_A : ℝ := sorry

/-- The speed of runner B in meters per hour -/
def speed_B : ℝ := sorry

/-- The length of the track in meters -/
def track_length : ℝ := sorry

/-- When running in the same direction, A catches up with B after 3 hours -/
axiom same_direction : 3 * (speed_A - speed_B) = track_length

/-- When running in opposite directions, A and B meet after 2 hours -/
axiom opposite_direction : 2 * (speed_A + speed_B) = track_length

/-- The ratio of A's speed to B's speed is 5:1 -/
theorem speed_ratio : speed_A / speed_B = 5 := by sorry

end NUMINAMATH_CALUDE_speed_ratio_l3934_393445


namespace NUMINAMATH_CALUDE_parabola_directrix_l3934_393406

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop :=
  y = (2 * x^2 - 8 * x + 6) / 16

/-- The directrix equation -/
def directrix (y : ℝ) : Prop :=
  y = -3/2

/-- Theorem stating that the given directrix is correct for the parabola -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → ∃ y_d : ℝ, directrix y_d ∧ 
  (∀ p q : ℝ × ℝ, parabola p.1 p.2 → 
    (p.1 - x)^2 + (p.2 - y)^2 = (p.2 - y_d)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3934_393406


namespace NUMINAMATH_CALUDE_first_group_size_l3934_393471

/-- Represents the work rate of a single beaver -/
def BeaverWorkRate : ℝ := 1

/-- Represents the total amount of work required to build the dam -/
def DamWork : ℝ := 1

theorem first_group_size (time1 : ℝ) (time2 : ℝ) (num_beavers2 : ℕ) :
  time1 > 0 → time2 > 0 → num_beavers2 > 0 →
  time1 = 3 → time2 = 5 → num_beavers2 = 12 →
  ∃ (num_beavers1 : ℕ), 
    num_beavers1 > 0 ∧ 
    (num_beavers1 : ℝ) * BeaverWorkRate * time1 = DamWork ∧
    (num_beavers2 : ℝ) * BeaverWorkRate * time2 = DamWork ∧
    num_beavers1 = 20 :=
by sorry

#check first_group_size

end NUMINAMATH_CALUDE_first_group_size_l3934_393471


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l3934_393461

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x > 1, p x) ↔ ∃ x > 1, ¬ p x :=
by sorry

theorem negation_of_specific_proposition :
  (¬ ∀ x > 1, x^3 + 16 > 8*x) ↔ ∃ x > 1, x^3 + 16 ≤ 8*x :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l3934_393461


namespace NUMINAMATH_CALUDE_intersection_is_empty_l3934_393475

def A : Set ℝ := {α | ∃ k : ℤ, α = (5 * k * Real.pi) / 3}
def B : Set ℝ := {β | ∃ k : ℤ, β = (3 * k * Real.pi) / 2}

theorem intersection_is_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_intersection_is_empty_l3934_393475


namespace NUMINAMATH_CALUDE_a_investment_value_l3934_393418

/-- Represents the investment and profit distribution in a partnership business. -/
structure Partnership where
  a_investment : ℝ
  b_investment : ℝ
  c_investment : ℝ
  total_profit : ℝ
  c_profit_share : ℝ

/-- Theorem stating that given the conditions of the problem, a's investment is 30000. -/
theorem a_investment_value (p : Partnership)
  (hb : p.b_investment = 45000)
  (hc : p.c_investment = 50000)
  (htotal : p.total_profit = 90000)
  (hc_share : p.c_profit_share = 36000) :
  p.a_investment = 30000 := by
  sorry

#check a_investment_value

end NUMINAMATH_CALUDE_a_investment_value_l3934_393418


namespace NUMINAMATH_CALUDE_probability_not_math_and_physics_is_four_fifths_l3934_393426

def subjects := 6
def selected := 3

def probability_not_math_and_physics : ℚ :=
  1 - (Nat.choose 4 1 : ℚ) / (Nat.choose subjects selected : ℚ)

theorem probability_not_math_and_physics_is_four_fifths :
  probability_not_math_and_physics = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_math_and_physics_is_four_fifths_l3934_393426


namespace NUMINAMATH_CALUDE_valid_lineups_count_l3934_393428

def team_size : ℕ := 15
def lineup_size : ℕ := 5
def excluded_players : ℕ := 3

/-- The number of valid lineups for a team with the given constraints -/
def valid_lineups : ℕ :=
  3 * (Nat.choose (team_size - excluded_players) (lineup_size - 1)) + 
  (Nat.choose (team_size - excluded_players) lineup_size)

/-- Theorem stating that the number of valid lineups is 2277 -/
theorem valid_lineups_count : valid_lineups = 2277 := by
  sorry

end NUMINAMATH_CALUDE_valid_lineups_count_l3934_393428


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_powers_l3934_393419

theorem min_value_of_sum_of_powers (a b : ℝ) (h : a + b = 2) :
  ∃ (m : ℝ), m = 6 ∧ ∀ (x y : ℝ), x + y = 2 → 3^x + 3^y ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_powers_l3934_393419


namespace NUMINAMATH_CALUDE_complement_of_supplement_30_l3934_393469

/-- The supplement of an angle in degrees -/
def supplement (angle : ℝ) : ℝ := 180 - angle

/-- The complement of an angle in degrees -/
def complement (angle : ℝ) : ℝ := 90 - angle

/-- The degree measure of the complement of the supplement of a 30-degree angle is 60° -/
theorem complement_of_supplement_30 : complement (supplement 30) = 60 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_supplement_30_l3934_393469


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt3_l3934_393423

theorem consecutive_integers_around_sqrt3 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt3_l3934_393423


namespace NUMINAMATH_CALUDE_quadratic_point_relation_l3934_393446

/-- A quadratic function y = x^2 - 4x + n, where n is a constant -/
def quadratic (n : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + n

theorem quadratic_point_relation (n : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  quadratic n x₁ = y₁ →
  quadratic n x₂ = y₂ →
  y₁ > y₂ →
  |x₁ - 2| > |x₂ - 2| :=
sorry

end NUMINAMATH_CALUDE_quadratic_point_relation_l3934_393446


namespace NUMINAMATH_CALUDE_pool_wall_area_ratio_l3934_393425

theorem pool_wall_area_ratio : 
  let pool_radius : ℝ := 20
  let wall_width : ℝ := 4
  let pool_area := π * pool_radius^2
  let total_area := π * (pool_radius + wall_width)^2
  let wall_area := total_area - pool_area
  wall_area / pool_area = 11 / 25 := by
sorry

end NUMINAMATH_CALUDE_pool_wall_area_ratio_l3934_393425


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l3934_393477

/-- Given a circle C: x^2 + y^2 + mx - 4 = 0 and two points on C symmetric 
    with respect to the line x - y + 3 = 0, prove that m = 6 -/
theorem circle_symmetry_line (m : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    (A.1^2 + A.2^2 + m*A.1 - 4 = 0) ∧ 
    (B.1^2 + B.2^2 + m*B.1 - 4 = 0) ∧ 
    (A.1 - A.2 + 3 = B.1 - B.2 + 3)) → 
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l3934_393477


namespace NUMINAMATH_CALUDE_reflection_line_sum_l3934_393433

/-- Given a line y = mx + b, if the reflection of point (2,3) across this line is (10,7), then m + b = 15 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), (x, y) = (10, 7) ∧ 
   (x - 2)^2 + (y - 3)^2 = (10 - 2)^2 + (7 - 3)^2 ∧
   (y - 3) = m * (x - 2) ∧
   y = m * x + b) →
  m + b = 15 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l3934_393433


namespace NUMINAMATH_CALUDE_min_socks_for_15_pairs_l3934_393416

/-- Represents the number of socks of each color in the drawer -/
def Drawer := List Nat

/-- The total number of socks in the drawer -/
def total_socks (d : Drawer) : Nat :=
  d.sum

/-- The number of different colors of socks in the drawer -/
def num_colors (d : Drawer) : Nat :=
  d.length

/-- The minimum number of socks needed to guarantee a certain number of pairs -/
def min_socks_for_pairs (num_pairs : Nat) (num_colors : Nat) : Nat :=
  num_colors + 2 * (num_pairs - 1)

theorem min_socks_for_15_pairs (d : Drawer) :
  num_colors d = 5 →
  total_socks d ≥ 400 →
  min_socks_for_pairs 15 (num_colors d) = 33 :=
by sorry

end NUMINAMATH_CALUDE_min_socks_for_15_pairs_l3934_393416


namespace NUMINAMATH_CALUDE_triangle_side_length_l3934_393488

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 2 →
  b = Real.sqrt 6 →
  A = 30 * π / 180 →
  (c = Real.sqrt 2 ∨ c = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3934_393488


namespace NUMINAMATH_CALUDE_reaction_properties_l3934_393407

-- Define the reaction components
structure Reaction where
  k2cr2o7 : ℕ
  hcl : ℕ
  kcl : ℕ
  crcl3 : ℕ
  cl2 : ℕ
  h2o : ℕ

-- Define oxidation states
def oxidation_state_cr_initial : Int := 6
def oxidation_state_cr_final : Int := 3
def oxidation_state_cl_initial : Int := -1
def oxidation_state_cl_final : Int := 0

-- Define the balanced equation
def balanced_reaction : Reaction := {
  k2cr2o7 := 2,
  hcl := 14,
  kcl := 2,
  crcl3 := 2,
  cl2 := 3,
  h2o := 7
}

-- Define the number of electrons transferred
def electrons_transferred : ℕ := 6

-- Define the oxidizing agent
def oxidizing_agent : String := "K2Cr2O7"

-- Define the element being oxidized
def element_oxidized : String := "Cl in HCl"

-- Define the oxidation product
def oxidation_product : String := "Cl2"

-- Define the mass ratio of oxidized to unoxidized HCl
def mass_ratio_oxidized_unoxidized : Rat := 3 / 4

-- Define the number of electrons transferred for 0.1 mol of Cl2
def electrons_transferred_for_0_1_mol_cl2 : ℕ := 120400000000000000000000

theorem reaction_properties :
  -- (1) Verify the oxidizing agent, element oxidized, and oxidation product
  (oxidizing_agent = "K2Cr2O7") ∧
  (element_oxidized = "Cl in HCl") ∧
  (oxidation_product = "Cl2") ∧
  -- (2) Verify the mass ratio of oxidized to unoxidized HCl
  (mass_ratio_oxidized_unoxidized = 3 / 4) ∧
  -- (3) Verify the number of electrons transferred for 0.1 mol of Cl2
  (electrons_transferred_for_0_1_mol_cl2 = 120400000000000000000000) := by
  sorry

#check reaction_properties

end NUMINAMATH_CALUDE_reaction_properties_l3934_393407


namespace NUMINAMATH_CALUDE_f_min_and_g_zeros_l3934_393448

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3*x + 2

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f x + (3 - m) * x

theorem f_min_and_g_zeros (h : ∀ x, x > 0 → f x ≥ 0) :
  (∃ x > 0, f x = 0) ∧
  (∀ m < 3, ¬∃ x₁ x₂, x₁ ≠ x₂ ∧ g m x₁ = 0 ∧ g m x₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_min_and_g_zeros_l3934_393448


namespace NUMINAMATH_CALUDE_sqrt_D_irrational_l3934_393421

theorem sqrt_D_irrational (k : ℤ) : 
  let a : ℤ := 3 * k
  let b : ℤ := 3 * k + 3
  let c : ℤ := a + b
  let D : ℤ := a^2 + b^2 + c^2
  Irrational (Real.sqrt D) := by sorry

end NUMINAMATH_CALUDE_sqrt_D_irrational_l3934_393421


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a12_l3934_393417

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 (a : ℕ → ℚ) :
  ArithmeticSequence a →
  a 7 + a 9 = 16 →
  a 4 = 1 →
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a12_l3934_393417


namespace NUMINAMATH_CALUDE_tournament_theorem_l3934_393451

/-- Represents a team in the tournament -/
structure Team :=
  (city : Fin 16)
  (is_team_a : Bool)

/-- The number of matches played by a team -/
def matches_played (t : Team) : Fin 32 := sorry

/-- The statement that all teams except one have unique match counts -/
def all_but_one_unique (exception : Team) : Prop :=
  ∀ t1 t2 : Team, t1 ≠ exception → t2 ≠ exception → t1 ≠ t2 → matches_played t1 ≠ matches_played t2

theorem tournament_theorem :
  ∃ (exception : Team),
    (all_but_one_unique exception) →
    (matches_played exception = 15) :=
  sorry

end NUMINAMATH_CALUDE_tournament_theorem_l3934_393451


namespace NUMINAMATH_CALUDE_FGH_supermarket_count_l3934_393424

def FGH_supermarkets : Type := Unit

def location : FGH_supermarkets → Bool
  | _ => sorry

def in_US (s : FGH_supermarkets) : Prop := location s = true
def in_Canada (s : FGH_supermarkets) : Prop := location s = false

axiom all_in_US_or_Canada : ∀ s : FGH_supermarkets, in_US s ∨ in_Canada s

def count_US : Nat := 42
def count_Canada : Nat := count_US - 14

def total_count : Nat := count_US + count_Canada

theorem FGH_supermarket_count : total_count = 70 := by sorry

end NUMINAMATH_CALUDE_FGH_supermarket_count_l3934_393424


namespace NUMINAMATH_CALUDE_expression_equality_l3934_393409

theorem expression_equality : 
  2013 * (2015/2014) + 2014 * (2016/2015) + 4029/(2014 * 2015) = 4029 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3934_393409


namespace NUMINAMATH_CALUDE_cookie_ratio_l3934_393482

/-- Represents the cookie distribution problem --/
def cookie_problem (initial_cookies : ℕ) (given_to_brother : ℕ) (left_at_end : ℕ) : Prop :=
  let mother_gift := given_to_brother / 2
  let total_after_mother := initial_cookies - given_to_brother + mother_gift
  let given_to_sister := total_after_mother - left_at_end
  (given_to_sister : ℚ) / total_after_mother = 2 / 3

/-- The main theorem stating the cookie distribution ratio --/
theorem cookie_ratio : 
  cookie_problem 20 10 5 := by sorry

end NUMINAMATH_CALUDE_cookie_ratio_l3934_393482


namespace NUMINAMATH_CALUDE_exists_non_intersecting_line_l3934_393455

/-- Represents a 6x6 chessboard -/
def Chessboard := Fin 6 × Fin 6

/-- Represents a domino placement on the chessboard -/
def Domino := (Chessboard × Chessboard)

/-- Predicate to check if a domino placement is valid -/
def valid_domino (d : Domino) : Prop :=
  let (a, b) := d
  ((a.1 = b.1 ∧ a.2.val + 1 = b.2.val) ∨ (a.2 = b.2 ∧ a.1.val + 1 = b.1.val))

/-- Represents a valid arrangement of 18 dominoes on the chessboard -/
def DominoArrangement := { arr : Fin 18 → Domino // ∀ i, valid_domino (arr i) }

/-- Represents a horizontal or vertical line on the chessboard -/
inductive CutLine
| horizontal (row : Fin 5) : CutLine
| vertical (col : Fin 5) : CutLine

/-- Predicate to check if a domino intersects a cut line -/
def intersects (d : Domino) (l : CutLine) : Prop :=
  match l with
  | CutLine.horizontal r => 
      (d.1.1.val ≤ r.val ∧ r.val < d.2.1.val) ∨ (d.2.1.val ≤ r.val ∧ r.val < d.1.1.val)
  | CutLine.vertical c => 
      (d.1.2.val ≤ c.val ∧ c.val < d.2.2.val) ∨ (d.2.2.val ≤ c.val ∧ c.val < d.1.2.val)

/-- The main theorem: there always exists a cut line that doesn't intersect any domino -/
theorem exists_non_intersecting_line (arr : DominoArrangement) : 
  ∃ l : CutLine, ∀ i : Fin 18, ¬ intersects (arr.val i) l := by
  sorry

end NUMINAMATH_CALUDE_exists_non_intersecting_line_l3934_393455


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3934_393476

/-- The line y = k(x-2) + 4 intersects the curve y = √(4-x²) if and only if k ∈ [3/4, +∞) -/
theorem line_intersects_circle (k : ℝ) : 
  (∃ x y : ℝ, y = k * (x - 2) + 4 ∧ y = Real.sqrt (4 - x^2)) ↔ k ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l3934_393476


namespace NUMINAMATH_CALUDE_negation_equivalence_l3934_393499

theorem negation_equivalence :
  (¬ (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → x ≠ 0 ∨ y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3934_393499


namespace NUMINAMATH_CALUDE_greatest_gcd_6Tn_n_minus_1_l3934_393486

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem greatest_gcd_6Tn_n_minus_1 :
  (∀ n : ℕ+, Nat.gcd (6 * triangular_number n) (n - 1) ≤ 3) ∧
  (∃ n : ℕ+, Nat.gcd (6 * triangular_number n) (n - 1) = 3) :=
by sorry

end NUMINAMATH_CALUDE_greatest_gcd_6Tn_n_minus_1_l3934_393486


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l3934_393465

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem
theorem derivative_f_at_1 : 
  deriv f 1 = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l3934_393465


namespace NUMINAMATH_CALUDE_jump_difference_l3934_393497

def monday_jumps : ℕ := 88
def tuesday_jumps : ℕ := 75
def wednesday_jumps : ℕ := 62
def thursday_jumps : ℕ := 91
def friday_jumps : ℕ := 80

def jump_counts : List ℕ := [monday_jumps, tuesday_jumps, wednesday_jumps, thursday_jumps, friday_jumps]

theorem jump_difference :
  (List.maximum jump_counts).get! - (List.minimum jump_counts).get! = 29 := by
  sorry

end NUMINAMATH_CALUDE_jump_difference_l3934_393497


namespace NUMINAMATH_CALUDE_total_path_is_2125_feet_l3934_393474

/-- Represents the scale of the plan in feet per inch -/
def scale : ℝ := 500

/-- Represents the initial path length on the plan in inches -/
def initial_path : ℝ := 3

/-- Represents the path extension on the plan in inches -/
def path_extension : ℝ := 1.25

/-- Calculates the total path length in feet -/
def total_path_length : ℝ := (initial_path + path_extension) * scale

/-- Theorem stating that the total path length is 2125 feet -/
theorem total_path_is_2125_feet : total_path_length = 2125 := by sorry

end NUMINAMATH_CALUDE_total_path_is_2125_feet_l3934_393474


namespace NUMINAMATH_CALUDE_range_of_sum_l3934_393487

def f (x : ℝ) : ℝ := |2 - x^2|

theorem range_of_sum (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  2 < a + b ∧ a + b < 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l3934_393487


namespace NUMINAMATH_CALUDE_shape_ratios_l3934_393491

/-- Given three shapes (cube A, cube B, and cylinder C) with specific volume ratios
    and height relationships, this theorem proves the ratios of their dimensions. -/
theorem shape_ratios (a b r : ℝ) (h : ℝ) :
  a > 0 ∧ b > 0 ∧ r > 0 ∧ h > 0 →
  h = a →
  a^3 / b^3 = 81 / 25 →
  a^3 / (π * r^2 * h) = 81 / 40 →
  (a / b = 3 / 5) ∧ (a / r = 9 * Real.sqrt π / Real.sqrt 40) := by
  sorry

#check shape_ratios

end NUMINAMATH_CALUDE_shape_ratios_l3934_393491


namespace NUMINAMATH_CALUDE_consecutive_squares_equality_l3934_393456

theorem consecutive_squares_equality : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (x^2 + (x+1)^2 + (x+2)^2 + (x+3)^2 = (x+4)^2 + (x+5)^2 + (x+6)^2) ∧ 
  (x^2 = 441) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_equality_l3934_393456


namespace NUMINAMATH_CALUDE_storks_joining_fence_l3934_393412

theorem storks_joining_fence (initial_birds initial_storks : ℕ) 
  (h1 : initial_birds = 6)
  (h2 : initial_storks = 3)
  (joined_storks : ℕ)
  (h3 : initial_birds = initial_storks + joined_storks + 1) :
  joined_storks = 2 := by
sorry

end NUMINAMATH_CALUDE_storks_joining_fence_l3934_393412


namespace NUMINAMATH_CALUDE_pascal_triangle_value_l3934_393402

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of elements in the row of Pascal's triangle we're considering -/
def row_length : ℕ := 51

/-- The position of the number we're looking for in the row (1-indexed) -/
def position : ℕ := 43

/-- The value we want to prove is correct -/
def target_value : ℕ := 10272278170

theorem pascal_triangle_value :
  binomial (row_length - 1) (position - 1) = target_value := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_value_l3934_393402


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3934_393401

def complex_number (a b : ℝ) : ℂ := a + b * Complex.I

theorem z_in_fourth_quadrant :
  let z : ℂ := 3 / (1 + 2 * Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3934_393401


namespace NUMINAMATH_CALUDE_initial_distance_is_point_eight_l3934_393441

/-- Two boats moving towards each other with given speeds and a known distance before collision -/
structure BoatProblem where
  speed1 : ℝ  -- Speed of boat 1 in miles/hr
  speed2 : ℝ  -- Speed of boat 2 in miles/hr
  distance_before_collision : ℝ  -- Distance between boats 1 minute before collision in miles
  time_before_collision : ℝ  -- Time before collision in hours

/-- The initial distance between the boats given the problem parameters -/
def initial_distance (p : BoatProblem) : ℝ :=
  p.distance_before_collision + (p.speed1 + p.speed2) * p.time_before_collision

/-- Theorem stating that the initial distance is 0.8 miles given the specific problem conditions -/
theorem initial_distance_is_point_eight :
  let p : BoatProblem := {
    speed1 := 4,
    speed2 := 20,
    distance_before_collision := 0.4,
    time_before_collision := 1 / 60
  }
  initial_distance p = 0.8 := by sorry

end NUMINAMATH_CALUDE_initial_distance_is_point_eight_l3934_393441


namespace NUMINAMATH_CALUDE_line_through_ellipse_focus_l3934_393490

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := 10 * x^2 + y^2 = 10

/-- The line equation -/
def line (x y b : ℝ) : Prop := 2 * x + b * y + 3 = 0

/-- Theorem: The value of b for a line passing through a focus of the given ellipse is either -1 or 1 -/
theorem line_through_ellipse_focus (b : ℝ) : 
  (∃ x y : ℝ, ellipse x y ∧ line x y b) → b = -1 ∨ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_ellipse_focus_l3934_393490


namespace NUMINAMATH_CALUDE_intersection_product_constant_l3934_393458

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  (a_pos : 0 < a)
  (b_pos : 0 < b)

/-- A point (x₀, y₀) on the hyperbola -/
structure PointOnHyperbola (H : Hyperbola a b) where
  x₀ : ℝ
  y₀ : ℝ
  on_hyperbola : x₀^2 / a^2 - y₀^2 / b^2 = 1

/-- The theorem stating that the product of x-coordinates of intersections is constant -/
theorem intersection_product_constant
  (H : Hyperbola a b) (P : PointOnHyperbola H) :
  ∃ (x₁ x₂ : ℝ),
    (x₁ * (b / a) = (P.x₀ * x₁) / a^2 - (P.y₀ * (b / a) * x₁) / b^2) ∧
    (x₂ * (-b / a) = (P.x₀ * x₂) / a^2 - (P.y₀ * (-b / a) * x₂) / b^2) ∧
    x₁ * x₂ = a^4 :=
sorry

end NUMINAMATH_CALUDE_intersection_product_constant_l3934_393458


namespace NUMINAMATH_CALUDE_probability_of_winning_pair_l3934_393400

/-- A card in the deck -/
structure Card where
  color : Bool  -- True for red, False for green
  label : Fin 5 -- Labels A, B, C, D, E represented as 0, 1, 2, 3, 4

/-- The deck of cards -/
def deck : Finset Card := sorry

/-- A pair of cards is winning if they have the same color or the same label -/
def is_winning_pair (c1 c2 : Card) : Bool :=
  c1.color = c2.color ∨ c1.label = c2.label

/-- The set of all possible pairs of cards -/
def all_pairs : Finset (Card × Card) := sorry

/-- The set of winning pairs -/
def winning_pairs : Finset (Card × Card) := sorry

/-- The probability of drawing a winning pair -/
theorem probability_of_winning_pair :
  (winning_pairs.card : ℚ) / all_pairs.card = 35 / 66 := by sorry

end NUMINAMATH_CALUDE_probability_of_winning_pair_l3934_393400


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3934_393432

theorem fraction_meaningful (x : ℝ) : 
  IsRegular (4 / (x + 2)) ↔ x ≠ -2 :=
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3934_393432


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3934_393408

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → 1 / a < 1 / b) ∧
  (∃ a b : ℝ, 1 / a < 1 / b ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3934_393408


namespace NUMINAMATH_CALUDE_hoodies_solution_l3934_393489

/-- Represents the number of hoodies owned by each person -/
structure HoodieOwnership where
  fiona : ℕ
  casey : ℕ
  alex : ℕ

/-- The conditions of the hoodies problem -/
def hoodies_problem (h : HoodieOwnership) : Prop :=
  h.fiona + h.casey + h.alex = 15 ∧
  h.casey = h.fiona + 2 ∧
  h.alex = 3

/-- The solution to the hoodies problem -/
theorem hoodies_solution :
  ∃ h : HoodieOwnership, hoodies_problem h ∧ h.fiona = 5 ∧ h.casey = 7 ∧ h.alex = 3 := by
  sorry

end NUMINAMATH_CALUDE_hoodies_solution_l3934_393489


namespace NUMINAMATH_CALUDE_inequality_solution_l3934_393410

theorem inequality_solution :
  ∃! x : ℝ, (Real.sqrt (x^3 + 2*x - 58) + 5) * |x^3 - 7*x^2 + 13*x - 3| ≤ 0 ∧ x = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3934_393410


namespace NUMINAMATH_CALUDE_pharmacy_masks_problem_l3934_393454

theorem pharmacy_masks_problem (first_batch_cost second_batch_cost : ℕ)
  (h1 : first_batch_cost = 1600)
  (h2 : second_batch_cost = 6000)
  (h3 : ∃ (x : ℕ), x > 0 ∧ 
    (second_batch_cost : ℚ) / (3 * x) - (first_batch_cost : ℚ) / x = 2) :
  ∃ (x : ℕ), x = 200 ∧ 
    (second_batch_cost : ℚ) / (3 * x) - (first_batch_cost : ℚ) / x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pharmacy_masks_problem_l3934_393454


namespace NUMINAMATH_CALUDE_count_numbers_with_seven_equals_133_l3934_393492

/-- Returns true if the given natural number contains the digit 7 at least once -/
def contains_seven (n : ℕ) : Bool := sorry

/-- Counts the number of natural numbers from 1 to 700 (inclusive) that contain the digit 7 at least once -/
def count_numbers_with_seven : ℕ := sorry

theorem count_numbers_with_seven_equals_133 : count_numbers_with_seven = 133 := by sorry

end NUMINAMATH_CALUDE_count_numbers_with_seven_equals_133_l3934_393492


namespace NUMINAMATH_CALUDE_salt_from_two_liters_seawater_l3934_393459

/-- The amount of salt in milliliters obtained from seawater when it fully evaporates -/
def salt_from_seawater (volume_liters : ℝ) (salt_percentage : ℝ) : ℝ :=
  volume_liters * salt_percentage * 1000

/-- Theorem: 2 liters of seawater with 20% salt concentration yields 400 ml of salt when evaporated -/
theorem salt_from_two_liters_seawater :
  salt_from_seawater 2 0.2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_salt_from_two_liters_seawater_l3934_393459


namespace NUMINAMATH_CALUDE_calculation_proof_l3934_393415

theorem calculation_proof : 
  Real.sqrt 3 * (Real.sqrt 3 + 2) - 2 * Real.tan (60 * π / 180) + (-1) ^ 2023 = 2 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l3934_393415


namespace NUMINAMATH_CALUDE_chocolate_cost_l3934_393404

theorem chocolate_cost (box_size : ℕ) (box_cost : ℚ) (total_candies : ℕ) : 
  box_size = 30 → 
  box_cost = 9 → 
  total_candies = 450 → 
  (total_candies / box_size : ℚ) * box_cost = 135 := by
sorry

end NUMINAMATH_CALUDE_chocolate_cost_l3934_393404


namespace NUMINAMATH_CALUDE_find_a_l3934_393411

theorem find_a (x y a : ℤ) 
  (eq1 : 3 * x + y = 40)
  (eq2 : a * x - y = 20)
  (eq3 : 3 * y^2 = 48) :
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l3934_393411


namespace NUMINAMATH_CALUDE_number_of_students_l3934_393460

/-- Represents the duration of a period in minutes. -/
def period_duration : ℕ := 40

/-- Represents the number of periods required for all students to present. -/
def total_periods : ℕ := 4

/-- Represents the duration of each student's presentation in minutes. -/
def presentation_duration : ℕ := 5

/-- Calculates the number of students in the class based on the given conditions. -/
def calculate_students : ℕ :=
  (period_duration / presentation_duration) * total_periods

/-- Theorem stating that the number of students in the class is 32. -/
theorem number_of_students : calculate_students = 32 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l3934_393460


namespace NUMINAMATH_CALUDE_zed_wye_value_l3934_393452

-- Define the types of coins
structure Coin where
  value : ℚ

-- Define the coins
def Ex : Coin := ⟨1⟩
def Wye : Coin := ⟨1⟩
def Zed : Coin := ⟨1⟩

-- Define the given conditions
axiom ex_wye_relation : 2 * Ex.value = 29 * Wye.value
axiom zed_ex_relation : Zed.value = 16 * Ex.value

theorem zed_wye_value : Zed.value = 232 * Wye.value :=
by sorry

end NUMINAMATH_CALUDE_zed_wye_value_l3934_393452


namespace NUMINAMATH_CALUDE_grasshopper_theorem_l3934_393447

/-- Represents the order of grasshoppers -/
inductive GrasshopperOrder
  | Even
  | Odd

/-- Represents a single jump of a grasshopper -/
def jump (order : GrasshopperOrder) : GrasshopperOrder :=
  match order with
  | GrasshopperOrder.Even => GrasshopperOrder.Odd
  | GrasshopperOrder.Odd => GrasshopperOrder.Even

/-- Represents multiple jumps of grasshoppers -/
def multipleJumps (initialOrder : GrasshopperOrder) (n : Nat) : GrasshopperOrder :=
  match n with
  | 0 => initialOrder
  | Nat.succ m => jump (multipleJumps initialOrder m)

theorem grasshopper_theorem :
  multipleJumps GrasshopperOrder.Even 1999 = GrasshopperOrder.Odd :=
by sorry

end NUMINAMATH_CALUDE_grasshopper_theorem_l3934_393447


namespace NUMINAMATH_CALUDE_sock_selection_theorem_l3934_393472

/-- The number of ways to select two socks of different colors -/
def differentColorPairs (white brown blue : ℕ) : ℕ :=
  white * brown + brown * blue + white * blue

/-- Theorem: The number of ways to select two socks of different colors
    from a drawer containing 5 white socks, 3 brown socks, and 4 blue socks
    is equal to 47. -/
theorem sock_selection_theorem :
  differentColorPairs 5 3 4 = 47 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_theorem_l3934_393472


namespace NUMINAMATH_CALUDE_square_side_length_l3934_393468

theorem square_side_length (d : ℝ) (h : d = 24) :
  ∃ s : ℝ, s > 0 ∧ s * s + s * s = d * d ∧ s = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3934_393468


namespace NUMINAMATH_CALUDE_michaels_birds_l3934_393438

/-- Given Michael's pets distribution, prove he has 12 birds -/
theorem michaels_birds (total_pets : ℕ) (dog_percent cat_percent bunny_percent : ℚ) : 
  total_pets = 120 →
  dog_percent = 30 / 100 →
  cat_percent = 40 / 100 →
  bunny_percent = 20 / 100 →
  (↑total_pets * (1 - dog_percent - cat_percent - bunny_percent) : ℚ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_michaels_birds_l3934_393438


namespace NUMINAMATH_CALUDE_tom_roses_count_l3934_393470

/-- The number of roses in a dozen -/
def dozen : ℕ := 12

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of dozens of roses Tom sends per day -/
def dozens_per_day : ℕ := 2

/-- The total number of roses Tom sent in a week -/
def total_roses : ℕ := dozens_per_day * dozen * days_in_week

theorem tom_roses_count : total_roses = 168 := by
  sorry

end NUMINAMATH_CALUDE_tom_roses_count_l3934_393470


namespace NUMINAMATH_CALUDE_interior_edges_sum_is_seven_l3934_393479

/-- A rectangular picture frame with specific properties -/
structure PictureFrame where
  /-- Width of the wood pieces used in the frame -/
  woodWidth : ℝ
  /-- Length of one outer edge of the frame -/
  outerEdgeLength : ℝ
  /-- Exposed area of the frame (excluding the picture) -/
  exposedArea : ℝ

/-- Calculates the sum of the lengths of the four interior edges of the frame -/
def interiorEdgesSum (frame : PictureFrame) : ℝ :=
  sorry

/-- Theorem stating that for a frame with given properties, the sum of interior edges is 7 inches -/
theorem interior_edges_sum_is_seven 
  (frame : PictureFrame)
  (h1 : frame.woodWidth = 2)
  (h2 : frame.outerEdgeLength = 6)
  (h3 : frame.exposedArea = 30) :
  interiorEdgesSum frame = 7 :=
sorry

end NUMINAMATH_CALUDE_interior_edges_sum_is_seven_l3934_393479


namespace NUMINAMATH_CALUDE_cube_root_of_eight_l3934_393484

theorem cube_root_of_eight (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_eight_l3934_393484


namespace NUMINAMATH_CALUDE_least_product_of_two_primes_above_50_l3934_393467

theorem least_product_of_two_primes_above_50 (p q : ℕ) : 
  p.Prime → q.Prime → p > 50 → q > 50 → p ≠ q → 
  ∃ (min_product : ℕ), min_product = 3127 ∧ 
    ∀ (r s : ℕ), r.Prime → s.Prime → r > 50 → s > 50 → r ≠ s → 
      p * q ≤ r * s := by
  sorry

end NUMINAMATH_CALUDE_least_product_of_two_primes_above_50_l3934_393467


namespace NUMINAMATH_CALUDE_sprinkler_system_water_usage_l3934_393480

theorem sprinkler_system_water_usage 
  (morning_usage : ℝ) 
  (evening_usage : ℝ) 
  (total_water : ℝ) 
  (h1 : morning_usage = 4)
  (h2 : evening_usage = 6)
  (h3 : total_water = 50) :
  (total_water / (morning_usage + evening_usage) = 5) :=
by sorry

end NUMINAMATH_CALUDE_sprinkler_system_water_usage_l3934_393480


namespace NUMINAMATH_CALUDE_subset_coloring_existence_l3934_393495

/-- The coloring function type -/
def ColoringFunction (α : Type*) := Set α → Bool

/-- Theorem statement -/
theorem subset_coloring_existence
  (S : Type*)
  [Fintype S]
  (h_card : Fintype.card S = 2002)
  (N : ℕ)
  (h_N : N ≤ 2^2002) :
  ∃ (f : ColoringFunction S),
    (∀ A B : Set S, f A ∧ f B → f (A ∪ B)) ∧
    (∀ A B : Set S, ¬f A ∧ ¬f B → ¬f (A ∪ B)) ∧
    (Fintype.card {A : Set S | f A} = N) :=
by sorry

end NUMINAMATH_CALUDE_subset_coloring_existence_l3934_393495


namespace NUMINAMATH_CALUDE_fishing_problem_l3934_393434

/-- Represents the number of fish caught by each person --/
structure FishCaught where
  jason : ℕ
  ryan : ℕ
  jeffery : ℕ

/-- The fishing problem statement --/
theorem fishing_problem (f : FishCaught) 
  (h1 : f.jason + f.ryan + f.jeffery = 100)
  (h2 : f.jeffery = 2 * f.ryan)
  (h3 : f.jeffery = 60) : 
  f.ryan = 30 := by
sorry


end NUMINAMATH_CALUDE_fishing_problem_l3934_393434


namespace NUMINAMATH_CALUDE_rocks_difference_l3934_393464

/-- Given the number of rocks collected by Joshua, Jose, and Albert, prove that Albert collected 20 more rocks than Jose. -/
theorem rocks_difference (joshua_rocks : ℕ) (jose_rocks : ℕ) (albert_rocks : ℕ)
  (h1 : joshua_rocks = 80)
  (h2 : jose_rocks = joshua_rocks - 14)
  (h3 : albert_rocks = joshua_rocks + 6) :
  albert_rocks - jose_rocks = 20 := by
sorry

end NUMINAMATH_CALUDE_rocks_difference_l3934_393464


namespace NUMINAMATH_CALUDE_rational_not_equiv_pos_neg_union_l3934_393427

-- Define the set of rational numbers
def RationalNumbers : Set ℝ := {x : ℝ | ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q}

-- Define the set of integers
def Integers : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = n}

-- Define the set of positive numbers
def PositiveNumbers : Set ℝ := {x : ℝ | x > 0}

-- Define the set of negative numbers
def NegativeNumbers : Set ℝ := {x : ℝ | x < 0}

-- Theorem statement
theorem rational_not_equiv_pos_neg_union :
  RationalNumbers ≠ PositiveNumbers ∪ NegativeNumbers :=
sorry

end NUMINAMATH_CALUDE_rational_not_equiv_pos_neg_union_l3934_393427


namespace NUMINAMATH_CALUDE_art_arrangement_probability_l3934_393436

/-- The probability of arranging n items with k specific items consecutive -/
def consecutive_probability (n : ℕ) (k : ℕ) : ℚ :=
  if k ≤ n ∧ k > 0 then
    (Nat.factorial (n - k + 1) * Nat.factorial k) / Nat.factorial n
  else
    0

/-- Theorem: The probability of arranging 12 pieces of art with 4 specific pieces consecutive is 1/55 -/
theorem art_arrangement_probability :
  consecutive_probability 12 4 = 1 / 55 := by
  sorry

#eval consecutive_probability 12 4

end NUMINAMATH_CALUDE_art_arrangement_probability_l3934_393436


namespace NUMINAMATH_CALUDE_x_equals_y_cubed_plus_2y_squared_minus_1_l3934_393430

theorem x_equals_y_cubed_plus_2y_squared_minus_1 (x y : ℝ) :
  x / (x - 1) = (y^3 + 2*y^2 - 1) / (y^3 + 2*y^2 - 2) → x = y^3 + 2*y^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_y_cubed_plus_2y_squared_minus_1_l3934_393430


namespace NUMINAMATH_CALUDE_probability_of_sum_seven_l3934_393444

def standard_die := Finset.range 6
def special_die := Finset.range 7

def sum_of_dice (a : ℕ) (b : ℕ) : ℕ :=
  a + if b = 6 then 0 else b + 1

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (standard_die.product special_die).filter (λ p => sum_of_dice p.1 p.2 = 7)

theorem probability_of_sum_seven :
  (favorable_outcomes.card : ℚ) / ((standard_die.card * special_die.card) : ℚ) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_sum_seven_l3934_393444


namespace NUMINAMATH_CALUDE_square_side_lengths_average_l3934_393463

theorem square_side_lengths_average (a b c : ℝ) (ha : a = 25) (hb : b = 64) (hc : c = 144) :
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_lengths_average_l3934_393463


namespace NUMINAMATH_CALUDE_perimeter_of_modified_square_l3934_393431

/-- The perimeter of figure ABFCDE formed by cutting a right isosceles triangle from a square and translating it -/
theorem perimeter_of_modified_square (side_length : ℝ) : 
  side_length > 0 →
  4 * side_length = 64 →
  let perimeter_ABFCDE := 4 * side_length + 2 * side_length * Real.sqrt 2
  perimeter_ABFCDE = 64 + 32 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_modified_square_l3934_393431


namespace NUMINAMATH_CALUDE_flooring_rate_calculation_l3934_393405

/-- Given a rectangular room with length 5.5 meters and width 3.75 meters,
    and a total flooring cost of 20625 rupees, the rate per square meter is 1000 rupees. -/
theorem flooring_rate_calculation (length : ℝ) (width : ℝ) (total_cost : ℝ) :
  length = 5.5 →
  width = 3.75 →
  total_cost = 20625 →
  total_cost / (length * width) = 1000 := by
  sorry

#check flooring_rate_calculation

end NUMINAMATH_CALUDE_flooring_rate_calculation_l3934_393405


namespace NUMINAMATH_CALUDE_line_through_point_and_intersection_l3934_393440

/-- The line passing through P(2, -3) and the intersection of two given lines -/
theorem line_through_point_and_intersection :
  let P : ℝ × ℝ := (2, -3)
  let line1 : ℝ → ℝ → ℝ := λ x y => 3 * x + 2 * y - 4
  let line2 : ℝ → ℝ → ℝ := λ x y => x - y + 5
  let result_line : ℝ → ℝ → ℝ := λ x y => 3.4 * x + 1.6 * y - 2
  -- The result line passes through P
  (result_line P.1 P.2 = 0) ∧
  -- The result line passes through the intersection point of line1 and line2
  (∃ x y : ℝ, line1 x y = 0 ∧ line2 x y = 0 ∧ result_line x y = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_line_through_point_and_intersection_l3934_393440


namespace NUMINAMATH_CALUDE_g_9_l3934_393473

/-- A function g satisfying g(x + y) = g(x) * g(y) for all real x and y, and g(3) = 4 -/
def g : ℝ → ℝ :=
  fun x => sorry

/-- The functional equation for g -/
axiom g_mul (x y : ℝ) : g (x + y) = g x * g y

/-- The initial condition for g -/
axiom g_3 : g 3 = 4

/-- Theorem stating that g(9) = 64 -/
theorem g_9 : g 9 = 64 := by
  sorry

end NUMINAMATH_CALUDE_g_9_l3934_393473


namespace NUMINAMATH_CALUDE_circle_translation_sum_l3934_393481

/-- The equation of circle D before translation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 10*y = -14

/-- The center of the circle after translation -/
def new_center : ℝ × ℝ := (1, -2)

/-- Theorem stating the sum of new center coordinates and radius after translation -/
theorem circle_translation_sum :
  ∃ (r : ℝ), 
    (∀ x y : ℝ, circle_equation x y → 
      ∃ a b : ℝ, new_center = (a, b) ∧ 
        a + b + r = -1 + Real.sqrt 27) :=
sorry

end NUMINAMATH_CALUDE_circle_translation_sum_l3934_393481


namespace NUMINAMATH_CALUDE_savings_calculation_l3934_393478

/-- Represents the financial situation of a person in a particular month --/
structure FinancialSituation where
  k : ℝ  -- Constant factor
  x : ℝ  -- Variable for income
  y : ℝ  -- Variable for expenditure
  I : ℝ  -- Total income
  E : ℝ  -- Regular expenditure
  U : ℝ  -- Unplanned expense
  S : ℝ  -- Savings

/-- The conditions of the financial situation --/
def financial_conditions (fs : FinancialSituation) : Prop :=
  fs.I = fs.k * fs.x ∧
  fs.E = fs.k * fs.y ∧
  fs.x / fs.y = 5 / 4 ∧
  fs.U = 0.2 * fs.E ∧
  fs.I = 16000 ∧
  fs.S = fs.I - (fs.E + fs.U)

/-- The theorem stating that under the given conditions, the savings is 640 --/
theorem savings_calculation (fs : FinancialSituation) :
  financial_conditions fs → fs.S = 640 := by
  sorry


end NUMINAMATH_CALUDE_savings_calculation_l3934_393478


namespace NUMINAMATH_CALUDE_crosswalk_stripe_distance_l3934_393420

/-- Given a street with parallel curbs and a crosswalk, calculate the distance between the stripes -/
theorem crosswalk_stripe_distance
  (curb_distance : ℝ)
  (curb_length : ℝ)
  (stripe_length : ℝ)
  (h_curb_distance : curb_distance = 50)
  (h_curb_length : curb_length = 20)
  (h_stripe_length : stripe_length = 65) :
  (curb_distance * curb_length) / stripe_length = 200 / 13 := by
sorry

end NUMINAMATH_CALUDE_crosswalk_stripe_distance_l3934_393420


namespace NUMINAMATH_CALUDE_sin_6theta_l3934_393449

theorem sin_6theta (θ : ℝ) : 
  Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 5 →
  Real.sin (6 * θ) = -630 * Real.sqrt 8 / 15625 := by
sorry

end NUMINAMATH_CALUDE_sin_6theta_l3934_393449


namespace NUMINAMATH_CALUDE_journey_equation_correct_l3934_393403

/-- Represents a car journey with a stop -/
structure Journey where
  initial_speed : ℝ
  final_speed : ℝ
  total_distance : ℝ
  total_time : ℝ
  stop_duration : ℝ

/-- Theorem stating that the given equation correctly represents the total distance traveled -/
theorem journey_equation_correct (j : Journey) 
  (h1 : j.initial_speed = 90)
  (h2 : j.final_speed = 110)
  (h3 : j.total_distance = 300)
  (h4 : j.total_time = 3.5)
  (h5 : j.stop_duration = 0.5) :
  ∃ t : ℝ, j.initial_speed * t + j.final_speed * (j.total_time - j.stop_duration - t) = j.total_distance :=
sorry

end NUMINAMATH_CALUDE_journey_equation_correct_l3934_393403


namespace NUMINAMATH_CALUDE_factor_divisor_proof_l3934_393429

theorem factor_divisor_proof :
  (∃ n : ℕ, 18 = 3 * n) ∧ 
  (∃ m : ℕ, 187 = 17 * m) ∧ 
  (¬ ∃ k : ℕ, 52 = 17 * k) ∧
  (∃ p : ℕ, 160 = 8 * p) := by
  sorry

end NUMINAMATH_CALUDE_factor_divisor_proof_l3934_393429


namespace NUMINAMATH_CALUDE_min_digits_to_remove_l3934_393498

def original_number : ℕ := 123454321

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

def sum_of_digits (n : ℕ) : ℕ :=
  (digits n).sum

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

def remove_digits (n : ℕ) (indices : List ℕ) : ℕ :=
  let digits := digits n
  let new_digits := (List.enum digits).filter (λ (i, _) => ¬ indices.contains i)
  new_digits.foldl (λ acc (_, d) => acc * 10 + d) 0

theorem min_digits_to_remove :
  ∃ (indices : List ℕ),
    indices.length = 2 ∧
    is_divisible_by_9 (remove_digits original_number indices) ∧
    ∀ (other_indices : List ℕ),
      other_indices.length < 2 →
      ¬ is_divisible_by_9 (remove_digits original_number other_indices) :=
by sorry

end NUMINAMATH_CALUDE_min_digits_to_remove_l3934_393498


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l3934_393414

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l3934_393414


namespace NUMINAMATH_CALUDE_faye_age_l3934_393422

/-- Represents the ages of Diana, Eduardo, Chad, and Faye -/
structure Ages where
  diana : ℕ
  eduardo : ℕ
  chad : ℕ
  faye : ℕ

/-- Defines the age relationships between Diana, Eduardo, Chad, and Faye -/
def valid_ages (ages : Ages) : Prop :=
  ages.diana + 3 = ages.eduardo ∧
  ages.eduardo = ages.chad + 4 ∧
  ages.faye = ages.chad + 3 ∧
  ages.diana = 14

/-- Theorem stating that given the age relationships and Diana's age, Faye's age is 18 -/
theorem faye_age (ages : Ages) (h : valid_ages ages) : ages.faye = 18 := by
  sorry

end NUMINAMATH_CALUDE_faye_age_l3934_393422


namespace NUMINAMATH_CALUDE_distance_to_point_l3934_393437

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 2*y + 4

-- Define the center of the circle
def circle_center : ℝ × ℝ := sorry

-- Define the distance function between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem distance_to_point : distance circle_center (10, 5) = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_l3934_393437


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3934_393494

def imaginary_part (z : ℂ) : ℝ := z.im

theorem imaginary_part_of_complex_number :
  imaginary_part (1/5 - 2/5 * I) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3934_393494


namespace NUMINAMATH_CALUDE_largest_root_bound_l3934_393413

/-- A polynomial of degree 4 with constrained coefficients -/
def ConstrainedPoly (b a₂ a₁ a₀ : ℝ) : ℝ → ℝ :=
  fun x ↦ x^4 + b*x^3 + a₂*x^2 + a₁*x + a₀

/-- The set of all constrained polynomials -/
def ConstrainedPolySet : Set (ℝ → ℝ) :=
  {p | ∃ b a₂ a₁ a₀, |b| < 3 ∧ |a₂| < 2 ∧ |a₁| < 2 ∧ |a₀| < 2 ∧ p = ConstrainedPoly b a₂ a₁ a₀}

theorem largest_root_bound :
  (∃ p ∈ ConstrainedPolySet, ∃ r, 3 < r ∧ r < 4 ∧ p r = 0) ∧
  (∀ p ∈ ConstrainedPolySet, ∀ r ≥ 4, p r ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_largest_root_bound_l3934_393413


namespace NUMINAMATH_CALUDE_sum_mod_eight_l3934_393466

theorem sum_mod_eight :
  (7145 + 7146 + 7147 + 7148 + 7149) % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_eight_l3934_393466


namespace NUMINAMATH_CALUDE_prime_composite_inequality_l3934_393493

theorem prime_composite_inequality (n : ℕ) : 
  (Prime (2 * n - 1) → 
    ∀ (a : Fin n → ℕ+), (∀ i j, i ≠ j → a i ≠ a j) → 
      ∃ i j, (a i + a j : ℝ) / (Nat.gcd (a i) (a j)) ≥ 2 * n - 1) ∧
  (¬Prime (2 * n - 1) → 
    ∃ (a : Fin n → ℕ+), (∀ i j, i ≠ j → a i ≠ a j) ∧ 
      ∀ i j, (a i + a j : ℝ) / (Nat.gcd (a i) (a j)) < 2 * n - 1) :=
by sorry

end NUMINAMATH_CALUDE_prime_composite_inequality_l3934_393493


namespace NUMINAMATH_CALUDE_earth_orbit_radius_scientific_notation_l3934_393457

theorem earth_orbit_radius_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 149000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.49 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_earth_orbit_radius_scientific_notation_l3934_393457


namespace NUMINAMATH_CALUDE_geometric_sequence_205th_term_l3934_393439

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n - 1)

theorem geometric_sequence_205th_term :
  let a₁ : ℝ := 6
  let r : ℝ := -1
  geometric_sequence a₁ r 205 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_205th_term_l3934_393439


namespace NUMINAMATH_CALUDE_friend_age_order_l3934_393442

-- Define the set of friends
inductive Friend : Type
  | David : Friend
  | Emma : Friend
  | Fiona : Friend

-- Define the age ordering relation
def AgeOrder : Friend → Friend → Prop := sorry

-- Define the property of being the oldest
def IsOldest (f : Friend) : Prop := ∀ g : Friend, g ≠ f → AgeOrder f g

-- Define the property of being the youngest
def IsYoungest (f : Friend) : Prop := ∀ g : Friend, g ≠ f → AgeOrder g f

-- State the theorem
theorem friend_age_order :
  -- Exactly one of the following statements is true
  (IsOldest Friend.Emma ∧ ¬IsYoungest Friend.Fiona ∧ IsOldest Friend.David) ∨
  (¬IsOldest Friend.Emma ∧ IsYoungest Friend.Fiona ∧ IsOldest Friend.David) ∨
  (¬IsOldest Friend.Emma ∧ ¬IsYoungest Friend.Fiona ∧ ¬IsOldest Friend.David) →
  -- The age order is David (oldest), Emma (middle), Fiona (youngest)
  AgeOrder Friend.David Friend.Emma ∧ AgeOrder Friend.Emma Friend.Fiona :=
by sorry

end NUMINAMATH_CALUDE_friend_age_order_l3934_393442


namespace NUMINAMATH_CALUDE_unique_solution_equation_l3934_393443

theorem unique_solution_equation (x : ℝ) (h : x ≥ 0) :
  2021 * (x^2020)^(1/202) - 1 = 2020 * x ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l3934_393443


namespace NUMINAMATH_CALUDE_function_identity_l3934_393483

theorem function_identity (f : ℕ → ℕ) 
  (h1 : f 1 > 0)
  (h2 : ∀ m n : ℕ, f (m^2 + n^2) = (f m)^2 + (f n)^2) :
  ∀ n : ℕ, f n = n :=
sorry

end NUMINAMATH_CALUDE_function_identity_l3934_393483


namespace NUMINAMATH_CALUDE_david_chemistry_marks_l3934_393435

/-- Represents the marks obtained in each subject --/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average of a list of natural numbers --/
def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

/-- Theorem: Given David's marks and average, his Chemistry mark must be 67 --/
theorem david_chemistry_marks (m : Marks) (h1 : m.english = 51) (h2 : m.mathematics = 65)
    (h3 : m.physics = 82) (h4 : m.biology = 85)
    (h5 : average [m.english, m.mathematics, m.physics, m.chemistry, m.biology] = 70) :
    m.chemistry = 67 := by
  sorry

#check david_chemistry_marks

end NUMINAMATH_CALUDE_david_chemistry_marks_l3934_393435


namespace NUMINAMATH_CALUDE_min_sum_reciprocal_constraint_l3934_393453

theorem min_sum_reciprocal_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 / x) + (2 / y) = 1) : 
  x + y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ (2 / x₀) + (2 / y₀) = 1 ∧ x₀ + y₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_min_sum_reciprocal_constraint_l3934_393453
