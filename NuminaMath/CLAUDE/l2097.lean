import Mathlib

namespace NUMINAMATH_CALUDE_exists_winning_strategy_2019_not_exists_winning_strategy_2020_l2097_209755

/-- Represents the state of the game at any point -/
structure GameState where
  piles : List Nat
  bag : Nat

/-- Represents a valid move in the game -/
inductive Move
  | Split (pile : Nat) (split1 : Nat) (split2 : Nat)

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  match move with
  | Move.Split pile split1 split2 =>
      pile ∈ state.piles ∧
      pile ≥ 4 ∧
      split1 > 0 ∧
      split2 > 0 ∧
      split1 + split2 = pile - 1

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.Split pile split1 split2 =>
      { piles := state.piles.filter (· ≠ pile) ++ [split1, split2],
        bag := state.bag + 1 }

/-- Checks if the game is in a winning state -/
def isWinningState (state : GameState) : Prop :=
  state.piles.all (· = 3)

/-- Defines a game strategy as a function that selects a move given a game state -/
def Strategy := GameState → Option Move

/-- Theorem: When N = 2019, there exists a winning strategy -/
theorem exists_winning_strategy_2019 :
  ∃ (strategy : Strategy),
    let initialState : GameState := { piles := [2019], bag := 0 }
    ∃ (finalState : GameState),
      (∀ (state : GameState),
        state.piles.sum + state.bag = 2019 →
        match strategy state with
        | some move => isValidMove state move
        | none => isWinningState state) ∧
      isWinningState finalState :=
sorry

/-- Theorem: When N = 2020, there does not exist a winning strategy -/
theorem not_exists_winning_strategy_2020 :
  ¬∃ (strategy : Strategy),
    let initialState : GameState := { piles := [2020], bag := 0 }
    ∃ (finalState : GameState),
      (∀ (state : GameState),
        state.piles.sum + state.bag = 2020 →
        match strategy state with
        | some move => isValidMove state move
        | none => isWinningState state) ∧
      isWinningState finalState :=
sorry

end NUMINAMATH_CALUDE_exists_winning_strategy_2019_not_exists_winning_strategy_2020_l2097_209755


namespace NUMINAMATH_CALUDE_problem_solution_l2097_209782

-- Define the function f(x) = ax^3 + bx^2
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

-- Define the derivative of f
def f_deriv (a b x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem problem_solution :
  ∀ (a b : ℝ),
    (f a b 1 = 4 ∧ f_deriv a b 1 = 9) →
    (a = 1 ∧ b = 3) ∧
    ∀ (m : ℝ),
      (∀ x ∈ Set.Icc m (m + 1), f_deriv 1 3 x ≥ 0) →
      (m ≥ 0 ∨ m ≤ -3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2097_209782


namespace NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l2097_209727

/-- Given vectors a and b in ℝ³, prove that a + 2b equals the expected result. -/
theorem vector_addition_and_scalar_multiplication (a b : ℝ × ℝ × ℝ) :
  a = (3, -2, 1) →
  b = (-2, 4, 0) →
  a + 2 • b = (-1, 6, 1) := by
sorry

end NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l2097_209727


namespace NUMINAMATH_CALUDE_problem_solution_l2097_209714

theorem problem_solution (x y : ℝ) 
  (h : |9*y + 1 - x| = Real.sqrt (x - 4) * Real.sqrt (4 - x)) : 
  2*x*Real.sqrt (1/x) + Real.sqrt (9*y) - Real.sqrt x / 2 + y*Real.sqrt (1/y) = 3 + 4*Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2097_209714


namespace NUMINAMATH_CALUDE_girls_on_playground_l2097_209700

theorem girls_on_playground (total_children boys : ℕ) 
  (h1 : total_children = 62) 
  (h2 : boys = 27) : 
  total_children - boys = 35 := by
sorry

end NUMINAMATH_CALUDE_girls_on_playground_l2097_209700


namespace NUMINAMATH_CALUDE_no_m_for_all_x_range_for_m_in_interval_l2097_209762

-- Part 1
theorem no_m_for_all_x : ∀ m : ℝ, ∃ x : ℝ, 2 * x - 1 ≤ m * (x^2 - 1) := by sorry

-- Part 2
def inequality_set (m : ℝ) : Set ℝ := {x | 2 * x - 1 > m * (x^2 - 1)}

theorem range_for_m_in_interval :
  ∀ m ∈ Set.Icc (-2 : ℝ) 2,
  inequality_set m = Set.Ioo (((-1 : ℝ) + Real.sqrt 7) / 2) ((1 + Real.sqrt 3) / 2) := by sorry

end NUMINAMATH_CALUDE_no_m_for_all_x_range_for_m_in_interval_l2097_209762


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l2097_209707

theorem scientific_notation_equivalence : 
  ∃ (a : ℝ) (n : ℤ), 0.0000002 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.0 ∧ n = -7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l2097_209707


namespace NUMINAMATH_CALUDE_tank_capacity_tank_capacity_1440_l2097_209722

/-- Given a tank with a leak and an inlet pipe, prove its capacity. -/
theorem tank_capacity (leak_time : ℝ) (inlet_rate : ℝ) (combined_time : ℝ) : ℝ :=
  let leak_rate := 1 / leak_time
  let inlet_rate_hourly := inlet_rate * 60
  let combined_rate := 1 / combined_time
  let capacity := (inlet_rate_hourly - combined_rate) / (leak_rate - combined_rate)
  by
    -- Assumptions
    have h1 : leak_time = 6 := by sorry
    have h2 : inlet_rate = 6 := by sorry
    have h3 : combined_time = 12 := by sorry
    
    -- Proof
    sorry

/-- The main theorem stating the tank's capacity. -/
theorem tank_capacity_1440 : tank_capacity 6 6 12 = 1440 := by sorry

end NUMINAMATH_CALUDE_tank_capacity_tank_capacity_1440_l2097_209722


namespace NUMINAMATH_CALUDE_tiles_per_row_l2097_209757

-- Define the area of the room
def room_area : ℝ := 144

-- Define the side length of a tile in meters
def tile_side : ℝ := 0.3

-- Theorem statement
theorem tiles_per_row (room_area : ℝ) (tile_side : ℝ) :
  room_area = 144 ∧ tile_side = 0.3 →
  (Real.sqrt room_area / tile_side : ℝ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_tiles_per_row_l2097_209757


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l2097_209731

/-- Given vectors a and b, if they are perpendicular, then k = 3 -/
theorem perpendicular_vectors_k_value (k : ℝ) :
  let a : ℝ × ℝ := (2*k - 3, -6)
  let b : ℝ × ℝ := (2, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l2097_209731


namespace NUMINAMATH_CALUDE_line_through_points_l2097_209766

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given distinct vectors a and b, and a scalar k, 
    prove that k*a + (1/2)*b lies on the line through a and b 
    if and only if k = 1/2 -/
theorem line_through_points (a b : V) (k : ℝ) 
    (h_distinct : a ≠ b) : 
    (∃ t : ℝ, k • a + (1/2) • b = a + t • (b - a)) ↔ k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2097_209766


namespace NUMINAMATH_CALUDE_exists_injection_with_property_l2097_209781

-- Define the set A as a finite type
variable {A : Type} [Finite A]

-- Define the set S as a predicate on triples of elements from A
variable (S : A → A → A → Prop)

-- State the conditions on S
variable (h1 : ∀ a b c : A, S a b c ↔ S b c a)
variable (h2 : ∀ a b c : A, S a b c ↔ ¬S c b a)
variable (h3 : ∀ a b c d : A, (S a b c ∧ S c d a) ↔ (S b c d ∧ S d a b))

-- State the theorem
theorem exists_injection_with_property :
  ∃ g : A → ℝ, Function.Injective g ∧
    ∀ a b c : A, g a < g b ∧ g b < g c → S a b c :=
sorry

end NUMINAMATH_CALUDE_exists_injection_with_property_l2097_209781


namespace NUMINAMATH_CALUDE_negation_equivalence_l2097_209796

theorem negation_equivalence :
  (¬ ∀ x : ℝ, ∃ n : ℕ+, (n : ℝ) > x^2) ↔ 
  (∃ x : ℝ, ∀ n : ℕ+, (n : ℝ) < x^2) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2097_209796


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l2097_209747

/-- Given a point M with polar coordinates (2, 2π/3), its Cartesian coordinates are (-1, √3) -/
theorem polar_to_cartesian :
  let ρ : ℝ := 2
  let θ : ℝ := 2 * π / 3
  let x : ℝ := ρ * Real.cos θ
  let y : ℝ := ρ * Real.sin θ
  (x = -1) ∧ (y = Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l2097_209747


namespace NUMINAMATH_CALUDE_problem_solution_l2097_209718

theorem problem_solution (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 2)  -- The absolute value of m is 2
  : (a + b) / (4 * m) + 2 * m^2 - 3 * c * d = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2097_209718


namespace NUMINAMATH_CALUDE_different_gender_selection_l2097_209738

theorem different_gender_selection (total_members : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total_members = 24)
  (h2 : boys = 12)
  (h3 : girls = 12)
  (h4 : total_members = boys + girls) :
  (boys * girls) + (girls * boys) = 288 := by
sorry

end NUMINAMATH_CALUDE_different_gender_selection_l2097_209738


namespace NUMINAMATH_CALUDE_each_person_receives_eight_doughnuts_l2097_209788

/-- The number of doughnuts each person receives when Samuel and Cathy share their doughnuts. -/
def doughnuts_per_person : ℕ :=
  let samuel_doughnuts : ℕ := 2 * 12
  let cathy_doughnuts : ℕ := 4 * 12
  let total_doughnuts : ℕ := samuel_doughnuts + cathy_doughnuts
  let total_people : ℕ := 10
  let dieting_friends : ℕ := 1
  let sharing_people : ℕ := total_people - dieting_friends
  total_doughnuts / sharing_people

/-- Theorem stating that each person receives 8 doughnuts. -/
theorem each_person_receives_eight_doughnuts : doughnuts_per_person = 8 := by
  sorry

end NUMINAMATH_CALUDE_each_person_receives_eight_doughnuts_l2097_209788


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l2097_209723

theorem inequality_not_always_true (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hz : z > 0) :
  ¬ (∀ x y z : ℝ, x > 0 → y > 0 → x > y → z > 0 → |x/z - y/z| = (x-y)/z) :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l2097_209723


namespace NUMINAMATH_CALUDE_magnitude_a_minus_2b_l2097_209775

def a : ℝ × ℝ × ℝ := (3, 5, -4)
def b : ℝ × ℝ × ℝ := (2, -1, -2)

theorem magnitude_a_minus_2b : 
  ‖(a.1 - 2 * b.1, a.2 - 2 * b.2, a.2.2 - 2 * b.2.2)‖ = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_a_minus_2b_l2097_209775


namespace NUMINAMATH_CALUDE_problem_solution_l2097_209702

/-- An increasing linear function on ℝ -/
def IncreasingLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ ∀ x, f x = a * x + b

theorem problem_solution (f g : ℝ → ℝ) (m : ℝ) :
  IncreasingLinearFunction f →
  (∀ x, g x = f x * (x + m)) →
  (∀ x, f (f x) = 16 * x + 5) →
  (∃ M, M = 13 ∧ ∀ x ∈ Set.Icc 1 3, g x ≤ M) →
  (∀ x, f x = 4 * x + 1) ∧ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2097_209702


namespace NUMINAMATH_CALUDE_students_going_to_zoo_l2097_209713

theorem students_going_to_zoo (teachers : ℕ) (students_per_group : ℕ) 
  (h1 : teachers = 8) 
  (h2 : students_per_group = 32) : 
  teachers * students_per_group = 256 := by
  sorry

end NUMINAMATH_CALUDE_students_going_to_zoo_l2097_209713


namespace NUMINAMATH_CALUDE_min_value_of_f_l2097_209728

noncomputable def f (x : ℝ) : ℝ := (1 / Real.sqrt (x^2 + 2)) + Real.sqrt (x^2 + 2)

theorem min_value_of_f :
  ∃ (min_val : ℝ), (∀ x, f x ≥ min_val) ∧ (min_val = (3 * Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2097_209728


namespace NUMINAMATH_CALUDE_book_length_l2097_209795

theorem book_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 3 → area = 6 → area = length * width → length = 2 := by
sorry

end NUMINAMATH_CALUDE_book_length_l2097_209795


namespace NUMINAMATH_CALUDE_mass_of_impurities_l2097_209786

/-- Given a sample of natural sulfur, prove that the mass of impurities
    is equal to the difference between the total mass and the mass of pure sulfur. -/
theorem mass_of_impurities (total_mass pure_sulfur_mass : ℝ) :
  total_mass ≥ pure_sulfur_mass →
  total_mass - pure_sulfur_mass = total_mass - pure_sulfur_mass :=
by sorry

end NUMINAMATH_CALUDE_mass_of_impurities_l2097_209786


namespace NUMINAMATH_CALUDE_convergence_condition_l2097_209705

/-- The iteration function for calculating 1/a -/
def f (a : ℝ) (x : ℝ) : ℝ := x * (2 - a * x)

/-- The sequence generated by the iteration -/
def iterSeq (a : ℝ) (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => f a (iterSeq a x₀ n)

theorem convergence_condition (a : ℝ) (x₀ : ℝ) (h : a > 0) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |iterSeq a x₀ n - 1/a| < ε) ↔ (0 < x₀ ∧ x₀ < 2/a) :=
sorry

end NUMINAMATH_CALUDE_convergence_condition_l2097_209705


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l2097_209719

theorem twenty_five_percent_less_than_80 (x : ℝ) : 
  (60 : ℝ) = 80 * (3/4) → x + x/4 = 60 → x = 48 := by sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l2097_209719


namespace NUMINAMATH_CALUDE_sum_in_range_l2097_209752

theorem sum_in_range : 
  let sum := 3 + 3/8 + 4 + 2/5 + 6 + 1/11
  13 < sum ∧ sum < 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_range_l2097_209752


namespace NUMINAMATH_CALUDE_max_abs_z_quadratic_equation_l2097_209742

theorem max_abs_z_quadratic_equation (a b c z : ℂ) 
  (h1 : Complex.abs a = 1)
  (h2 : Complex.abs b = 1)
  (h3 : Complex.abs c = 1)
  (h4 : a * z^2 + 2 * b * z + c = 0) :
  Complex.abs z ≤ 1 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_quadratic_equation_l2097_209742


namespace NUMINAMATH_CALUDE_opposite_sign_fractions_l2097_209759

theorem opposite_sign_fractions (x : ℚ) : 
  x = 7/5 → ((x - 1) / 2) * ((x - 2) / 3) < 0 := by sorry

end NUMINAMATH_CALUDE_opposite_sign_fractions_l2097_209759


namespace NUMINAMATH_CALUDE_expand_product_l2097_209767

theorem expand_product (x : ℝ) : (x + 4) * (x - 5) = x^2 - x - 20 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2097_209767


namespace NUMINAMATH_CALUDE_equation_solution_l2097_209783

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    x₁ > 0 ∧ x₂ > 0 ∧
    (1/3 * (4 * x₁^2 - 2) = (x₁^2 - 60*x₁ - 15) * (x₁^2 + 30*x₁ + 3)) ∧
    (1/3 * (4 * x₂^2 - 2) = (x₂^2 - 60*x₂ - 15) * (x₂^2 + 30*x₂ + 3)) ∧
    x₁ = 30 + Real.sqrt 917 ∧
    x₂ = -15 + Real.sqrt 8016 / 6 ∧
    ∀ (y : ℝ), 
      y > 0 ∧ (1/3 * (4 * y^2 - 2) = (y^2 - 60*y - 15) * (y^2 + 30*y + 3)) →
      (y = x₁ ∨ y = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2097_209783


namespace NUMINAMATH_CALUDE_weight_of_a_l2097_209763

theorem weight_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 80 →
  (a + b + c + d) / 4 = 82 →
  e = d + 3 →
  (b + c + d + e) / 4 = 81 →
  a = 95 := by
sorry

end NUMINAMATH_CALUDE_weight_of_a_l2097_209763


namespace NUMINAMATH_CALUDE_prob_queens_or_jacks_l2097_209709

/-- The probability of drawing either all three queens or at least 2 jacks from 3 cards in a standard deck -/
theorem prob_queens_or_jacks (total_cards : Nat) (num_queens : Nat) (num_jacks : Nat) 
  (h1 : total_cards = 52)
  (h2 : num_queens = 4)
  (h3 : num_jacks = 4) : 
  (Nat.choose num_queens 3) / (Nat.choose total_cards 3) + 
  (Nat.choose num_jacks 2 * (total_cards - num_jacks) + Nat.choose num_jacks 3) / (Nat.choose total_cards 3) = 290 / 5525 := by
  sorry

end NUMINAMATH_CALUDE_prob_queens_or_jacks_l2097_209709


namespace NUMINAMATH_CALUDE_roots_and_d_values_l2097_209774

-- Define the polynomial p(x)
def p (c d x : ℝ) : ℝ := x^3 + c*x + d

-- Define the polynomial q(x)
def q (c d x : ℝ) : ℝ := x^3 + c*x + d - 270

-- Theorem statement
theorem roots_and_d_values (u v c d : ℝ) : 
  (p c d u = 0 ∧ p c d v = 0) ∧ 
  (q c d (u+3) = 0 ∧ q c d (v-2) = 0) →
  d = -6 ∨ d = -120 := by
sorry

end NUMINAMATH_CALUDE_roots_and_d_values_l2097_209774


namespace NUMINAMATH_CALUDE_summer_sun_salutations_l2097_209725

/-- The number of sun salutations Summer performs each weekday -/
def sun_salutations_per_weekday : ℕ :=
  1300 / (365 / 7 * 5)

/-- Theorem stating that Summer performs 5 sun salutations each weekday -/
theorem summer_sun_salutations :
  sun_salutations_per_weekday = 5 := by
  sorry

end NUMINAMATH_CALUDE_summer_sun_salutations_l2097_209725


namespace NUMINAMATH_CALUDE_translation_iff_equal_movements_l2097_209706

/-- Represents the movement of a table's legs -/
structure TableMovement where
  leg1 : ℝ
  leg2 : ℝ
  leg3 : ℝ
  leg4 : ℝ

/-- Determines if a table movement represents a translation -/
def isTranslation (m : TableMovement) : Prop :=
  m.leg1 = m.leg2 ∧ m.leg2 = m.leg3 ∧ m.leg3 = m.leg4

/-- Theorem: A table movement is a translation if and only if all leg movements are equal -/
theorem translation_iff_equal_movements (m : TableMovement) :
  isTranslation m ↔ m.leg1 = m.leg2 ∧ m.leg1 = m.leg3 ∧ m.leg1 = m.leg4 := by sorry

end NUMINAMATH_CALUDE_translation_iff_equal_movements_l2097_209706


namespace NUMINAMATH_CALUDE_weight_change_l2097_209798

theorem weight_change (w : ℝ) (hw : w > 0) : w * 0.8 * 1.3 * 0.8 * 1.1 < w := by
  sorry

#check weight_change

end NUMINAMATH_CALUDE_weight_change_l2097_209798


namespace NUMINAMATH_CALUDE_real_part_of_one_plus_i_over_i_l2097_209744

/-- The real part of (1+i)/i is 1 -/
theorem real_part_of_one_plus_i_over_i : 
  Complex.re ((1 + Complex.I) / Complex.I) = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_one_plus_i_over_i_l2097_209744


namespace NUMINAMATH_CALUDE_product_inequality_l2097_209753

theorem product_inequality (a b c d : ℝ) 
  (sum_eq : a + d = b + c) 
  (abs_ineq : |a - d| < |b - c|) : 
  a * d > b * c := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l2097_209753


namespace NUMINAMATH_CALUDE_indefinite_game_l2097_209764

/-- Represents the web structure --/
structure Web where
  rings : ℕ
  radii : ℕ
  rings_ge_two : rings ≥ 2
  radii_ge_three : radii ≥ 3

/-- Represents the game state --/
inductive GameState
  | Ongoing
  | ButterflyWins
  | SpiderWins

/-- Defines the game outcome --/
def gameOutcome (web : Web) : GameState :=
  if web.radii % 2 = 0 ∧ web.rings ≥ web.radii / 2 then
    GameState.Ongoing
  else if web.radii % 2 = 1 ∧ web.rings ≥ (web.radii - 1) / 2 then
    GameState.Ongoing
  else
    GameState.Ongoing -- We use Ongoing as a placeholder, as the actual outcome might depend on the players' strategies

/-- Theorem stating that under certain conditions, the game continues indefinitely --/
theorem indefinite_game (web : Web) :
  (web.radii % 2 = 0 → web.rings ≥ web.radii / 2) ∧
  (web.radii % 2 = 1 → web.rings ≥ (web.radii - 1) / 2) →
  gameOutcome web = GameState.Ongoing :=
by
  sorry

#check indefinite_game

end NUMINAMATH_CALUDE_indefinite_game_l2097_209764


namespace NUMINAMATH_CALUDE_m_range_l2097_209737

def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem m_range (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ Set.Ioo 1 2 ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_m_range_l2097_209737


namespace NUMINAMATH_CALUDE_dan_bought_five_notebooks_l2097_209745

/-- Represents the purchase of school supplies -/
structure SchoolSupplies where
  totalSpent : ℕ
  backpackCost : ℕ
  penCost : ℕ
  pencilCost : ℕ
  notebookCost : ℕ

/-- Calculates the number of notebooks bought -/
def notebooksBought (supplies : SchoolSupplies) : ℕ :=
  (supplies.totalSpent - (supplies.backpackCost + supplies.penCost + supplies.pencilCost)) / supplies.notebookCost

/-- Theorem stating that Dan bought 5 notebooks -/
theorem dan_bought_five_notebooks (supplies : SchoolSupplies)
  (h1 : supplies.totalSpent = 32)
  (h2 : supplies.backpackCost = 15)
  (h3 : supplies.penCost = 1)
  (h4 : supplies.pencilCost = 1)
  (h5 : supplies.notebookCost = 3) :
  notebooksBought supplies = 5 := by
  sorry

end NUMINAMATH_CALUDE_dan_bought_five_notebooks_l2097_209745


namespace NUMINAMATH_CALUDE_solve_for_y_l2097_209760

theorem solve_for_y (x y : ℝ) (h1 : x^2 = 2*y - 6) (h2 : x = 7) : y = 55/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2097_209760


namespace NUMINAMATH_CALUDE_original_number_exists_and_unique_l2097_209711

theorem original_number_exists_and_unique : ∃! x : ℝ, 3 * (2 * x + 9) = 51 := by
  sorry

end NUMINAMATH_CALUDE_original_number_exists_and_unique_l2097_209711


namespace NUMINAMATH_CALUDE_equation_simplification_l2097_209703

theorem equation_simplification (x : ℝ) :
  x / 0.3 = 1 + (1.2 - 0.3 * x) / 0.2 ↔ 10 * x / 3 = 1 + (12 - 3 * x) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_simplification_l2097_209703


namespace NUMINAMATH_CALUDE_prob_red_white_blue_eq_two_fifty_five_l2097_209772

/-- The number of red marbles initially in the bag -/
def red_marbles : ℕ := 4

/-- The number of white marbles initially in the bag -/
def white_marbles : ℕ := 6

/-- The number of blue marbles initially in the bag -/
def blue_marbles : ℕ := 2

/-- The total number of marbles initially in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

/-- The probability of drawing a red marble first, then a white marble, then a blue marble -/
def prob_red_white_blue : ℚ :=
  (red_marbles : ℚ) / total_marbles *
  (white_marbles : ℚ) / (total_marbles - 1) *
  (blue_marbles : ℚ) / (total_marbles - 2)

theorem prob_red_white_blue_eq_two_fifty_five :
  prob_red_white_blue = 2 / 55 := by sorry

end NUMINAMATH_CALUDE_prob_red_white_blue_eq_two_fifty_five_l2097_209772


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2097_209769

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

/-- The property that three terms form a geometric sequence -/
def GeometricSequence (x y z : ℝ) : Prop :=
  y * y = x * z

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a)
    (h_sum : a 2 + a 3 + a 4 = 15)
    (h_geom : GeometricSequence (a 1 + 2) (a 3 + 4) (a 6 + 16)) :
    a 10 = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2097_209769


namespace NUMINAMATH_CALUDE_photography_preference_l2097_209729

-- Define the number of students who dislike photography
variable (x : ℕ)

-- Define the total number of students in the class
def total : ℕ := 9 * x

-- Define the number of students who like photography
def like : ℕ := 5 * x

-- Define the number of students who are neutral towards photography
def neutral : ℕ := x + 12

-- Theorem statement
theorem photography_preference (x : ℕ) :
  like x = (total x / 2) + 3 := by
  sorry

end NUMINAMATH_CALUDE_photography_preference_l2097_209729


namespace NUMINAMATH_CALUDE_margo_irma_pairing_probability_l2097_209797

/-- Represents the number of students in the class -/
def class_size : ℕ := 40

/-- Represents the probability of Margo being paired with Irma -/
def probability_paired_with_irma : ℚ := 1 / 39

/-- Theorem stating that the probability of Margo being paired with Irma is 1/39 -/
theorem margo_irma_pairing_probability :
  probability_paired_with_irma = 1 / (class_size - 1) :=
by sorry

end NUMINAMATH_CALUDE_margo_irma_pairing_probability_l2097_209797


namespace NUMINAMATH_CALUDE_acoustic_guitar_price_l2097_209768

theorem acoustic_guitar_price (total_guitars : ℕ) (total_revenue : ℕ) 
  (electric_price : ℕ) (electric_count : ℕ) :
  total_guitars = 9 →
  total_revenue = 3611 →
  electric_price = 479 →
  electric_count = 4 →
  (total_revenue - electric_price * electric_count) / (total_guitars - electric_count) = 339 := by
sorry

end NUMINAMATH_CALUDE_acoustic_guitar_price_l2097_209768


namespace NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l2097_209701

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_nonprimes (start : ℕ) (count : ℕ) : Prop :=
  ∀ i : ℕ, i < count → ¬(is_prime (start + i))

theorem smallest_prime_after_seven_nonprimes :
  (is_prime 97) ∧
  (consecutive_nonprimes 90 7) ∧
  (∀ p : ℕ, p < 97 → is_prime p → ¬(∃ start : ℕ, start < p ∧ consecutive_nonprimes start 7)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l2097_209701


namespace NUMINAMATH_CALUDE_isosceles_triangle_angles_l2097_209761

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the property of being isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- Define the exterior angle of A
def exteriorAngleA (t : Triangle) : ℝ := 180 - t.A

-- Theorem statement
theorem isosceles_triangle_angles (t : Triangle) :
  exteriorAngleA t = 110 →
  isIsosceles t →
  t.B = 70 ∨ t.B = 55 ∨ t.B = 40 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_angles_l2097_209761


namespace NUMINAMATH_CALUDE_parabola_opens_left_is_ellipse_parabola_ellipse_system_correct_l2097_209730

/-- Represents a parabola and an ellipse in a 2D coordinate system -/
structure ParabolaEllipseSystem where
  m : ℝ
  n : ℝ
  hm : m > 0
  hn : n > 0

/-- The parabola equation: mx + ny² = 0 -/
def parabola_equation (sys : ParabolaEllipseSystem) (x y : ℝ) : Prop :=
  sys.m * x + sys.n * y^2 = 0

/-- The ellipse equation: mx² + ny² = 1 -/
def ellipse_equation (sys : ParabolaEllipseSystem) (x y : ℝ) : Prop :=
  sys.m * x^2 + sys.n * y^2 = 1

/-- Theorem stating that the parabola opens to the left -/
theorem parabola_opens_left (sys : ParabolaEllipseSystem) :
  ∀ x y, parabola_equation sys x y → x ≤ 0 :=
sorry

/-- Theorem stating that the equation represents an ellipse -/
theorem is_ellipse (sys : ParabolaEllipseSystem) :
  ∃ a b, a > 0 ∧ b > 0 ∧ ∀ x y, ellipse_equation sys x y ↔ (x/a)^2 + (y/b)^2 = 1 :=
sorry

/-- Main theorem: The system represents a left-opening parabola and an ellipse -/
theorem parabola_ellipse_system_correct (sys : ParabolaEllipseSystem) :
  (∀ x y, parabola_equation sys x y → x ≤ 0) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ ∀ x y, ellipse_equation sys x y ↔ (x/a)^2 + (y/b)^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_parabola_opens_left_is_ellipse_parabola_ellipse_system_correct_l2097_209730


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2097_209777

theorem sqrt_equation_solution (z : ℚ) : 
  Real.sqrt (5 - 4 * z + 1) = 7 → z = -43 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2097_209777


namespace NUMINAMATH_CALUDE_A_expression_l2097_209734

theorem A_expression (a : ℝ) (A : ℝ) 
  (h : 2.353 * A = (3 * a + Real.sqrt (6 * a - 1))^(1/2) + (3 * a - Real.sqrt (6 * a - 1))^(1/2)) :
  ((1/6 ≤ a ∧ a < 1/3) → A = Real.sqrt 2 / (1 - 3 * a)) ∧
  (a > 1/3 → A = Real.sqrt (12 * a - 2) / (3 * a - 1)) := by
  sorry

end NUMINAMATH_CALUDE_A_expression_l2097_209734


namespace NUMINAMATH_CALUDE_parabola_equation_points_with_y_neg_three_l2097_209751

/-- A parabola passing through (1,0) and (0,-3) with axis of symmetry x=2 -/
structure Parabola where
  -- Define the parabola using a function
  f : ℝ → ℝ
  -- The parabola passes through (1,0)
  passes_through_A : f 1 = 0
  -- The parabola passes through (0,-3)
  passes_through_B : f 0 = -3
  -- The axis of symmetry is x=2
  symmetry_axis : ∀ x, f (2 + x) = f (2 - x)

/-- The equation of the parabola is y = -(x-2)^2 + 1 -/
theorem parabola_equation (p : Parabola) : 
  ∀ x, p.f x = -(x - 2)^2 + 1 := by sorry

/-- The points (0,-3) and (4,-3) are the only points on the parabola with y-coordinate -3 -/
theorem points_with_y_neg_three (p : Parabola) :
  ∀ x, p.f x = -3 ↔ x = 0 ∨ x = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_points_with_y_neg_three_l2097_209751


namespace NUMINAMATH_CALUDE_dr_strange_food_choices_l2097_209721

/-- Represents the number of food items and days --/
def n : ℕ := 12

/-- Represents the ways to choose food items each day --/
def choices : ℕ → ℕ
  | 0 => 2  -- First day has 2 choices
  | i => 2  -- Each subsequent day has 2 choices

/-- The total number of ways to choose food items over n days --/
def totalWays : ℕ := 2^n

theorem dr_strange_food_choices :
  totalWays = 2048 := by sorry

end NUMINAMATH_CALUDE_dr_strange_food_choices_l2097_209721


namespace NUMINAMATH_CALUDE_trig_identity_l2097_209790

theorem trig_identity (α β : ℝ) : 
  Real.sin α ^ 2 + Real.sin β ^ 2 - Real.sin α ^ 2 * Real.sin β ^ 2 + Real.cos α ^ 2 * Real.cos β ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2097_209790


namespace NUMINAMATH_CALUDE_subtracted_number_l2097_209780

theorem subtracted_number (x : ℕ) (some_number : ℕ) 
  (h1 : x = 88320) 
  (h2 : x + 1315 + 9211 - some_number = 11901) : 
  some_number = 86945 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l2097_209780


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l2097_209785

/-- Represents the number of painted faces a small cube can have -/
inductive PaintedFaces
  | one
  | two
  | three

/-- Represents a large cube that is painted on the outside and cut into smaller cubes -/
structure PaintedCube where
  edge_length : ℕ
  small_cube_length : ℕ

/-- Counts the number of small cubes with a specific number of painted faces -/
def count_painted_faces (cube : PaintedCube) (faces : PaintedFaces) : ℕ :=
  match faces with
  | PaintedFaces.one => 0   -- Placeholder, actual calculation needed
  | PaintedFaces.two => 0   -- Placeholder, actual calculation needed
  | PaintedFaces.three => 0 -- Placeholder, actual calculation needed

/-- Theorem stating the correct count of small cubes with different numbers of painted faces -/
theorem painted_cube_theorem (cube : PaintedCube) 
    (h1 : cube.edge_length = 10)
    (h2 : cube.small_cube_length = 1) :
    count_painted_faces cube PaintedFaces.three = 8 ∧
    count_painted_faces cube PaintedFaces.two = 96 ∧
    count_painted_faces cube PaintedFaces.one = 384 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l2097_209785


namespace NUMINAMATH_CALUDE_prime_square_remainders_mod_180_l2097_209771

theorem prime_square_remainders_mod_180 :
  ∃ (S : Finset Nat), 
    (∀ p : Nat, Prime p → p > 5 → ∃ r ∈ S, p^2 % 180 = r) ∧ 
    S.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_remainders_mod_180_l2097_209771


namespace NUMINAMATH_CALUDE_turning_to_similar_section_is_random_event_l2097_209749

/-- Represents the event of turning to a similar section in a textbook -/
def turning_to_similar_section : Type := Unit

/-- Defines the properties of the event -/
class EventProperties (α : Type) where
  not_guaranteed : ∀ (x y : α), x ≠ y → True
  possible : ∃ (x : α), True
  not_certain : ¬ (∀ (x : α), True)
  not_impossible : ∃ (x : α), True
  not_predictable : ∀ (x : α), ¬ (∀ (y : α), x = y)

/-- Defines a random event -/
class RandomEvent (α : Type) extends EventProperties α

/-- Theorem stating that turning to a similar section is a random event -/
theorem turning_to_similar_section_is_random_event :
  RandomEvent turning_to_similar_section :=
sorry

end NUMINAMATH_CALUDE_turning_to_similar_section_is_random_event_l2097_209749


namespace NUMINAMATH_CALUDE_spiral_grid_second_row_sum_l2097_209793

/-- Represents a position in the grid -/
structure Position :=
  (x : Fin 15)
  (y : Fin 15)

/-- Represents the spiral grid -/
def SpiralGrid := Fin 15 → Fin 15 → Nat

/-- Creates a spiral grid according to the problem description -/
def createSpiralGrid : SpiralGrid :=
  sorry

/-- Returns the center position of the grid -/
def centerPosition : Position :=
  ⟨7, 7⟩

/-- Checks if a given position is in the second row from the top -/
def isSecondRow (pos : Position) : Prop :=
  pos.y = 1

/-- Returns the maximum value in the second row -/
def maxSecondRow (grid : SpiralGrid) : Nat :=
  sorry

/-- Returns the minimum value in the second row -/
def minSecondRow (grid : SpiralGrid) : Nat :=
  sorry

theorem spiral_grid_second_row_sum :
  let grid := createSpiralGrid
  maxSecondRow grid + minSecondRow grid = 367 :=
sorry

end NUMINAMATH_CALUDE_spiral_grid_second_row_sum_l2097_209793


namespace NUMINAMATH_CALUDE_paint_calculation_l2097_209776

theorem paint_calculation (total_paint : ℚ) : 
  (1/4 : ℚ) * total_paint + (1/2 : ℚ) * ((3/4 : ℚ) * total_paint) = 225 → 
  total_paint = 360 := by
sorry

end NUMINAMATH_CALUDE_paint_calculation_l2097_209776


namespace NUMINAMATH_CALUDE_family_ages_solution_l2097_209765

/-- Represents the ages of a father and his two children -/
structure FamilyAges where
  father : ℕ
  son : ℕ
  daughter : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.father + ages.son + ages.daughter = 110 ∧
  ages.son = ages.daughter ∧
  3 * ages.father = 186

/-- The theorem to be proved -/
theorem family_ages_solution :
  ∃ (ages : FamilyAges), satisfiesConditions ages ∧ 
    ages.father = 62 ∧ ages.son = 24 ∧ ages.daughter = 24 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_solution_l2097_209765


namespace NUMINAMATH_CALUDE_opposite_to_silver_is_pink_l2097_209778

-- Define the colors
inductive Color
  | Pink
  | Teal
  | Maroon
  | Lilac
  | Silver
  | Crimson

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  faces : List Face
  hinged : List (Face × Face)

-- Define the property of opposite faces
def areOpposite (c : Cube) (f1 f2 : Face) : Prop :=
  f1 ∈ c.faces ∧ f2 ∈ c.faces ∧ f1 ≠ f2

-- State the theorem
theorem opposite_to_silver_is_pink (c : Cube) :
  (∃ f1 f2 : Face, f1.color = Color.Silver ∧ f2.color = Color.Pink ∧ areOpposite c f1 f2) :=
by sorry

end NUMINAMATH_CALUDE_opposite_to_silver_is_pink_l2097_209778


namespace NUMINAMATH_CALUDE_f_neg_one_eq_one_fifteenth_l2097_209792

/-- The function f satisfying the given equation for all x -/
noncomputable def f : ℝ → ℝ := 
  fun x => ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) - 1) / (x^(2^5 - 1) - 1)

/-- Theorem stating that f(-1) = 1/15 -/
theorem f_neg_one_eq_one_fifteenth : f (-1) = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_one_eq_one_fifteenth_l2097_209792


namespace NUMINAMATH_CALUDE_magic_box_pennies_l2097_209779

def double_daily (initial : ℕ) (days : ℕ) : ℕ :=
  initial * (2 ^ days)

theorem magic_box_pennies :
  ∃ (initial : ℕ), double_daily initial 4 = 48 ∧ initial = 3 :=
sorry

end NUMINAMATH_CALUDE_magic_box_pennies_l2097_209779


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2097_209717

theorem polynomial_expansion (z : ℝ) : 
  (3*z^2 + 4*z - 5) * (4*z^3 - 3*z + 2) = 
  12*z^5 + 16*z^4 - 29*z^3 - 6*z^2 + 23*z - 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2097_209717


namespace NUMINAMATH_CALUDE_quadratic_vertex_in_first_quadrant_l2097_209743

/-- Given a quadratic function y = ax² + bx + c where a, b, and c satisfy certain conditions,
    prove that its vertex lies in the first quadrant. -/
theorem quadratic_vertex_in_first_quadrant
  (a b c : ℝ)
  (eq1 : a - b + c = 0)
  (eq2 : 9*a + 3*b + c = 0)
  (b_pos : b > 0) :
  let x := -b / (2*a)
  let y := a * x^2 + b * x + c
  x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_in_first_quadrant_l2097_209743


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l2097_209746

theorem triangle_ratio_theorem (a b c : ℝ) (A B C : ℝ) :
  C = π / 3 →
  c = Real.sqrt 3 →
  (3 * a + b) / (3 * Real.sin A + Real.sin B) = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l2097_209746


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l2097_209735

/-- The number of ways to arrange 15 letters (4 D's, 6 E's, and 5 F's) with specific constraints -/
def letterArrangements : ℕ :=
  Finset.sum (Finset.range 5) (fun j =>
    Nat.choose 4 j * Nat.choose 6 (4 - j) * Nat.choose 5 j)

/-- Theorem stating that the number of valid arrangements is equal to the sum formula -/
theorem valid_arrangements_count :
  letterArrangements =
    Finset.sum (Finset.range 5) (fun j =>
      Nat.choose 4 j * Nat.choose 6 (4 - j) * Nat.choose 5 j) :=
by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l2097_209735


namespace NUMINAMATH_CALUDE_sum_always_positive_l2097_209784

theorem sum_always_positive (b : ℝ) (h : b = 2) : 
  (∀ x : ℝ, (3*x^2 - 2*x + b) + (x^2 + b*x - 1) = 4*x^2 + 1) ∧
  (∀ x : ℝ, 4*x^2 + 1 > 0) := by
sorry

end NUMINAMATH_CALUDE_sum_always_positive_l2097_209784


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l2097_209726

theorem sqrt_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l2097_209726


namespace NUMINAMATH_CALUDE_min_omega_value_l2097_209740

theorem min_omega_value (ω : ℝ) (h1 : ω > 0) : 
  (∀ x, 2 * Real.cos (ω * (x - π/5) + π/5) = 2 * Real.sin (ω * x + π/5)) →
  ω ≥ 5/2 ∧ (∀ ω' > 0, (∀ x, 2 * Real.cos (ω' * (x - π/5) + π/5) = 2 * Real.sin (ω' * x + π/5)) → ω' ≥ ω) :=
by sorry

end NUMINAMATH_CALUDE_min_omega_value_l2097_209740


namespace NUMINAMATH_CALUDE_stewart_farm_horse_food_l2097_209716

/-- Represents a farm with sheep and horses -/
structure Farm where
  sheep : ℕ
  horses : ℕ
  total_horse_food : ℕ

/-- Calculates the amount of food each horse needs per day -/
def horse_food_per_day (f : Farm) : ℚ :=
  f.total_horse_food / f.horses

/-- The Stewart farm satisfies the given conditions -/
def stewart_farm : Farm :=
  { sheep := 24,
    horses := 56,
    total_horse_food := 12880 }

theorem stewart_farm_horse_food :
  horse_food_per_day stewart_farm = 230 :=
sorry

end NUMINAMATH_CALUDE_stewart_farm_horse_food_l2097_209716


namespace NUMINAMATH_CALUDE_inverse_function_theorem_l2097_209756

noncomputable def f (x : ℝ) : ℝ := Real.log (3 * x + 1)

def f_domain (x : ℝ) : Prop := x > -1

noncomputable def g (x : ℝ) : ℝ := (Real.exp x - 1) ^ 3

theorem inverse_function_theorem (x : ℝ) (hx : f_domain x) :
  g (f x) = x ∧ f (g x) = x :=
sorry

end NUMINAMATH_CALUDE_inverse_function_theorem_l2097_209756


namespace NUMINAMATH_CALUDE_division_result_l2097_209789

theorem division_result : (4.036 : ℝ) / 0.02 = 201.8 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l2097_209789


namespace NUMINAMATH_CALUDE_cube_and_sphere_volume_l2097_209741

theorem cube_and_sphere_volume (cube_volume : Real) (sphere_volume : Real) : 
  cube_volume = 8 → sphere_volume = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cube_and_sphere_volume_l2097_209741


namespace NUMINAMATH_CALUDE_tea_store_profit_l2097_209773

theorem tea_store_profit (m n : ℝ) (h : m > n) : 
  let cost := 40 * m + 60 * n
  let revenue := 50 * (m + n)
  revenue - cost > 0 := by
sorry

end NUMINAMATH_CALUDE_tea_store_profit_l2097_209773


namespace NUMINAMATH_CALUDE_circle_tangent_and_chord_l2097_209758

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define point P
def P : ℝ × ℝ := (3, 4)

-- Define the tangent line l
def l (x y : ℝ) : Prop := 3*x + 4*y - 25 = 0

-- Define line m
def m (x y : ℝ) : Prop := x = 3 ∨ 7*x - 24*y + 75 = 0

-- Theorem statement
theorem circle_tangent_and_chord :
  (∀ x y, C x y → l x y → (x, y) = P) ∧
  (∀ x y, m x y → 
    (∃ x1 y1 x2 y2, C x1 y1 ∧ C x2 y2 ∧ m x1 y1 ∧ m x2 y2 ∧ 
     (x1 - x2)^2 + (y1 - y2)^2 = 64) ∧
    (x, y) = P) := by sorry

end NUMINAMATH_CALUDE_circle_tangent_and_chord_l2097_209758


namespace NUMINAMATH_CALUDE_cos_alpha_value_l2097_209715

-- Define the angle α
variable (α : Real)

-- Define the point P
def P : ℝ × ℝ := (4, 3)

-- Define the condition that the terminal side of α passes through P
def terminal_side_passes_through (α : Real) (p : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t > 0 ∧ p.1 = t * Real.cos α ∧ p.2 = t * Real.sin α

-- State the theorem
theorem cos_alpha_value (h : terminal_side_passes_through α P) : 
  Real.cos α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l2097_209715


namespace NUMINAMATH_CALUDE_line_equations_l2097_209787

/-- Given point M -/
def M : ℝ × ℝ := (-1, 2)

/-- Given line equation -/
def L : ℝ → ℝ → ℝ := λ x y ↦ 2*x + y + 5

/-- Parallel line -/
def L_parallel : ℝ → ℝ → ℝ := λ x y ↦ 2*x + y

/-- Perpendicular line -/
def L_perpendicular : ℝ → ℝ → ℝ := λ x y ↦ x - 2*y + 5

theorem line_equations :
  (L_parallel M.1 M.2 = 0 ∧ 
   ∀ (x y : ℝ), L_parallel x y = 0 → L x y = L_parallel x y + 5) ∧
  (L_perpendicular M.1 M.2 = 0 ∧ 
   ∀ (x y : ℝ), L x y = 0 → L_perpendicular x y = 0 → x = y) := by
  sorry

end NUMINAMATH_CALUDE_line_equations_l2097_209787


namespace NUMINAMATH_CALUDE_stationery_solution_l2097_209799

/-- Represents a pack of stationery -/
structure StationeryPack where
  sheets : ℕ
  envelopes : ℕ

/-- The problem setup -/
def stationeryProblem (pack : StationeryPack) : Prop :=
  ∃ (jack_leftover_sheets tom_leftover_envelopes : ℕ),
    -- Jack uses all envelopes and has 90 sheets left
    pack.sheets - 2 * pack.envelopes = jack_leftover_sheets ∧
    jack_leftover_sheets = 90 ∧
    -- Tom uses all sheets and has 30 envelopes left
    pack.sheets = 4 * (pack.envelopes - tom_leftover_envelopes) ∧
    tom_leftover_envelopes = 30

/-- The theorem to prove -/
theorem stationery_solution :
  ∃ (pack : StationeryPack),
    stationeryProblem pack ∧
    pack.sheets = 120 ∧
    pack.envelopes = 30 := by
  sorry

end NUMINAMATH_CALUDE_stationery_solution_l2097_209799


namespace NUMINAMATH_CALUDE_alex_bike_trip_l2097_209739

/-- Alex's cross-country bike trip problem -/
theorem alex_bike_trip (total_distance : ℝ) (flat_speed : ℝ) (uphill_speed : ℝ) (uphill_time : ℝ)
                       (downhill_speed : ℝ) (downhill_time : ℝ) (walking_distance : ℝ) :
  total_distance = 164 →
  flat_speed = 20 →
  uphill_speed = 12 →
  uphill_time = 2.5 →
  downhill_speed = 24 →
  downhill_time = 1.5 →
  walking_distance = 8 →
  ∃ (flat_time : ℝ), 
    flat_time = 4.5 ∧ 
    total_distance = flat_speed * flat_time + uphill_speed * uphill_time + 
                     downhill_speed * downhill_time + walking_distance :=
by sorry


end NUMINAMATH_CALUDE_alex_bike_trip_l2097_209739


namespace NUMINAMATH_CALUDE_bakery_storage_ratio_l2097_209704

/-- Given the conditions of a bakery's storage room, prove that the ratio of sugar to flour is 1 to 1. -/
theorem bakery_storage_ratio : ∀ (sugar flour baking_soda : ℝ),
  sugar = 2400 →
  flour = 10 * baking_soda →
  flour = 8 * (baking_soda + 60) →
  sugar / flour = 1 := by
  sorry

end NUMINAMATH_CALUDE_bakery_storage_ratio_l2097_209704


namespace NUMINAMATH_CALUDE_total_bones_in_pile_l2097_209712

def number_of_dogs : ℕ := 5

def bones_first_dog : ℕ := 3

def bones_second_dog (first : ℕ) : ℕ := first - 1

def bones_third_dog (second : ℕ) : ℕ := 2 * second

def bones_fourth_dog : ℕ := 1

def bones_fifth_dog (fourth : ℕ) : ℕ := 2 * fourth

theorem total_bones_in_pile :
  bones_first_dog +
  bones_second_dog bones_first_dog +
  bones_third_dog (bones_second_dog bones_first_dog) +
  bones_fourth_dog +
  bones_fifth_dog bones_fourth_dog = 12 :=
by sorry

end NUMINAMATH_CALUDE_total_bones_in_pile_l2097_209712


namespace NUMINAMATH_CALUDE_range_of_a_l2097_209748

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2097_209748


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l2097_209733

/-- Proves that Bobby ate 6 pieces of candy initially -/
theorem bobby_candy_problem :
  ∀ (initial_candy : ℕ) (eaten_initially : ℕ) (eaten_later : ℕ) (remaining_candy : ℕ),
    initial_candy = 22 →
    eaten_later = 5 →
    remaining_candy = 8 →
    initial_candy - (eaten_initially + eaten_initially / 2 + eaten_later) = remaining_candy →
    eaten_initially = 6 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l2097_209733


namespace NUMINAMATH_CALUDE_new_person_weight_is_81_l2097_209724

/-- The weight of a new person replacing one in a group, given the average weight increase --/
def new_person_weight (n : ℕ) (avg_increase : ℚ) (replaced_weight : ℚ) : ℚ :=
  replaced_weight + n * avg_increase

/-- Theorem: The weight of the new person is 81 kg --/
theorem new_person_weight_is_81 :
  new_person_weight 8 2 65 = 81 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_is_81_l2097_209724


namespace NUMINAMATH_CALUDE_interval_relationship_l2097_209791

theorem interval_relationship : 
  (∀ x, 2 < x ∧ x < 3 → 1 < x ∧ x < 5) ∧ 
  ¬(∀ x, 1 < x ∧ x < 5 → 2 < x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_interval_relationship_l2097_209791


namespace NUMINAMATH_CALUDE_intersection_singleton_implies_k_negative_one_intersection_and_union_when_k_is_two_l2097_209736

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}
def N (k : ℝ) : Set ℝ := {x | x - k ≤ 0}

theorem intersection_singleton_implies_k_negative_one :
  (∃! x, x ∈ M ∩ N k) → k = -1 :=
by sorry

theorem intersection_and_union_when_k_is_two :
  M ∩ N 2 = {x | -1 ≤ x ∧ x ≤ 2} ∧ M ∪ N 2 = {x | x ≤ 5} :=
by sorry

end NUMINAMATH_CALUDE_intersection_singleton_implies_k_negative_one_intersection_and_union_when_k_is_two_l2097_209736


namespace NUMINAMATH_CALUDE_graph_is_pair_of_straight_lines_l2097_209754

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := x^2 - 9*y^2 = 0

/-- Definition of a straight line -/
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ x, f x = m * x + b

/-- The graph consists of two straight lines -/
theorem graph_is_pair_of_straight_lines :
  ∃ (f g : ℝ → ℝ), 
    (is_straight_line f ∧ is_straight_line g) ∧
    (∀ x y, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end NUMINAMATH_CALUDE_graph_is_pair_of_straight_lines_l2097_209754


namespace NUMINAMATH_CALUDE_negative_reciprocal_inequality_l2097_209720

theorem negative_reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  -1/a < -1/b := by
  sorry

end NUMINAMATH_CALUDE_negative_reciprocal_inequality_l2097_209720


namespace NUMINAMATH_CALUDE_point_relationship_l2097_209708

theorem point_relationship (b y₁ y₂ : ℝ) 
  (h1 : y₁ = -(-2) + b) 
  (h2 : y₂ = -(3) + b) : 
  y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_point_relationship_l2097_209708


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l2097_209732

theorem binomial_expansion_sum (a b c d e f : ℤ) : 
  (∀ x : ℤ, (x - 2)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  16*(a + b) + 4*(c + d) + (e + f) = -256 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l2097_209732


namespace NUMINAMATH_CALUDE_train_length_l2097_209770

/-- Given a train that crosses a platform in 39 seconds and a signal pole in 18 seconds,
    where the platform is 350 meters long, prove that the length of the train is 300 meters. -/
theorem train_length (platform_crossing_time : ℕ) (pole_crossing_time : ℕ) (platform_length : ℕ)
    (h1 : platform_crossing_time = 39)
    (h2 : pole_crossing_time = 18)
    (h3 : platform_length = 350) :
    let train_length := (platform_crossing_time * platform_length) / (platform_crossing_time - pole_crossing_time)
    train_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2097_209770


namespace NUMINAMATH_CALUDE_yoongi_has_fewer_apples_l2097_209710

def jungkook_initial_apples : ℕ := 6
def jungkook_received_apples : ℕ := 3
def yoongi_apples : ℕ := 4

theorem yoongi_has_fewer_apples :
  yoongi_apples < jungkook_initial_apples + jungkook_received_apples :=
by
  sorry

end NUMINAMATH_CALUDE_yoongi_has_fewer_apples_l2097_209710


namespace NUMINAMATH_CALUDE_quarters_in_jar_l2097_209750

def pennies : ℕ := 123
def nickels : ℕ := 85
def dimes : ℕ := 35
def half_dollars : ℕ := 15
def dollar_coins : ℕ := 5
def family_members : ℕ := 8
def ice_cream_cost : ℚ := 4.5
def leftover : ℚ := 0.97

def total_other_coins : ℚ := 
  pennies * 0.01 + nickels * 0.05 + dimes * 0.1 + half_dollars * 0.5 + dollar_coins * 1.0

theorem quarters_in_jar : 
  ∃ (quarters : ℕ), 
    (quarters : ℚ) * 0.25 + total_other_coins = 
      family_members * ice_cream_cost + leftover ∧ 
    quarters = 140 := by sorry

end NUMINAMATH_CALUDE_quarters_in_jar_l2097_209750


namespace NUMINAMATH_CALUDE_lot_worth_l2097_209794

/-- Given a lot where a man owns half and sells a tenth of his share for $460, 
    prove that the worth of the entire lot is $9200. -/
theorem lot_worth (man_share : ℚ) (sold_fraction : ℚ) (sold_amount : ℕ) :
  man_share = 1/2 →
  sold_fraction = 1/10 →
  sold_amount = 460 →
  (sold_amount / sold_fraction) / man_share = 9200 := by
  sorry

end NUMINAMATH_CALUDE_lot_worth_l2097_209794
