import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_l856_85633

def equation1 (x y z : ℝ) : Prop := x^2 - 22*y - 69*z + 703 = 0
def equation2 (x y z : ℝ) : Prop := y^2 + 23*x + 23*z - 1473 = 0
def equation3 (x y z : ℝ) : Prop := z^2 - 63*x + 66*y + 2183 = 0

theorem unique_solution :
  ∃! (x y z : ℝ), equation1 x y z ∧ equation2 x y z ∧ equation3 x y z ∧ x = 20 ∧ y = -22 ∧ z = 23 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l856_85633


namespace NUMINAMATH_CALUDE_reemas_correct_marks_l856_85667

/-- Proves that given a class of 35 students with an initial average of 72,
    if one student's marks are changed from 46 to x, resulting in a new average of 71.71,
    then x = 36.85 -/
theorem reemas_correct_marks 
  (num_students : Nat)
  (initial_average : ℚ)
  (incorrect_marks : ℚ)
  (new_average : ℚ)
  (h1 : num_students = 35)
  (h2 : initial_average = 72)
  (h3 : incorrect_marks = 46)
  (h4 : new_average = 71.71)
  : ∃ x : ℚ, x = 36.85 ∧ 
    (num_students : ℚ) * initial_average - incorrect_marks + x = 
    (num_students : ℚ) * new_average :=
by
  sorry


end NUMINAMATH_CALUDE_reemas_correct_marks_l856_85667


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_l856_85625

-- Arithmetic Progression
def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (a + (n - 1) * d / 2)

-- Geometric Progression
def geometric_product (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a^n * r^(n * (n - 1) / 2)

theorem arithmetic_geometric_progression :
  (arithmetic_sum 0 (1/3) 15 = 35) ∧
  (geometric_product 1 (10^(1/3)) 15 = 10^35) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_l856_85625


namespace NUMINAMATH_CALUDE_urn_probability_l856_85698

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Blue

/-- Represents the state of the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents a single operation -/
def perform_operation (state : UrnState) (drawn : BallColor) : UrnState :=
  match drawn with
  | BallColor.Red => UrnState.mk (state.red + 3) state.blue
  | BallColor.Blue => UrnState.mk state.red (state.blue + 3)

/-- Represents the sequence of operations -/
def operation_sequence := List BallColor

/-- Calculates the probability of a specific operation sequence -/
def sequence_probability (seq : operation_sequence) : ℚ :=
  sorry

/-- Counts the number of valid operation sequences -/
def count_valid_sequences : ℕ :=
  sorry

theorem urn_probability :
  let initial_state : UrnState := UrnState.mk 2 1
  let final_state : UrnState := UrnState.mk 10 6
  let num_operations : ℕ := 5
  (count_valid_sequences * sequence_probability (List.replicate num_operations BallColor.Red)) = 16/115 :=
sorry

end NUMINAMATH_CALUDE_urn_probability_l856_85698


namespace NUMINAMATH_CALUDE_horner_rule_v3_l856_85675

def f (x : ℝ) : ℝ := 2*x^5 - 3*x^3 + 2*x^2 + x - 3

def horner_v3 (x : ℝ) : ℝ := 
  let v0 := 2*x
  let v1 := v0*x - 3
  let v2 := v1*x + 2
  v2*x + 1

theorem horner_rule_v3 : horner_v3 2 = 12 := by sorry

end NUMINAMATH_CALUDE_horner_rule_v3_l856_85675


namespace NUMINAMATH_CALUDE_coinciding_rest_days_count_l856_85659

/-- Al's schedule cycle length -/
def al_cycle : ℕ := 6

/-- Number of rest days in Al's cycle -/
def al_rest_days : ℕ := 2

/-- Barb's schedule cycle length -/
def barb_cycle : ℕ := 6

/-- Number of rest days in Barb's cycle -/
def barb_rest_days : ℕ := 1

/-- Total number of days -/
def total_days : ℕ := 1000

/-- The number of days both Al and Barb have rest-days on the same day -/
def coinciding_rest_days : ℕ := total_days / (al_cycle * barb_cycle / Nat.gcd al_cycle barb_cycle)

theorem coinciding_rest_days_count :
  coinciding_rest_days = 166 :=
by sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_count_l856_85659


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l856_85660

theorem completing_square_equivalence :
  ∀ x : ℝ, 2 * x^2 + 4 * x - 3 = 0 ↔ (x + 1)^2 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l856_85660


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l856_85613

/-- The coefficient of x^3 in the expansion of (1-1/x)(1+x)^5 is 5 -/
theorem coefficient_x_cubed_in_expansion : ∃ (f : ℝ → ℝ),
  (∀ x ≠ 0, f x = (1 - 1/x) * (1 + x)^5) ∧
  (∃ a b c d e g : ℝ, ∀ x ≠ 0, f x = a + b*x + c*x^2 + 5*x^3 + d*x^4 + e*x^5 + g/x) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l856_85613


namespace NUMINAMATH_CALUDE_isabellas_houses_l856_85657

theorem isabellas_houses (green yellow red : ℕ) : 
  green = 3 * yellow →
  yellow = red - 40 →
  green + red = 160 →
  green + red = 160 := by
sorry

end NUMINAMATH_CALUDE_isabellas_houses_l856_85657


namespace NUMINAMATH_CALUDE_trigonometric_identity_l856_85673

theorem trigonometric_identity (x : ℝ) :
  Real.sin (x + π / 3) + 2 * Real.sin (x - π / 3) - Real.sqrt 3 * Real.cos ((2 * π) / 3 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l856_85673


namespace NUMINAMATH_CALUDE_paper_towel_case_rolls_l856_85690

theorem paper_towel_case_rolls : ∀ (case_price individual_price : ℚ) (savings_percentage : ℚ),
  case_price = 9 →
  individual_price = 1 →
  savings_percentage = 25 →
  ∃ (n : ℕ), n = 12 ∧ case_price = (1 - savings_percentage / 100) * (n * individual_price) :=
by
  sorry

end NUMINAMATH_CALUDE_paper_towel_case_rolls_l856_85690


namespace NUMINAMATH_CALUDE_histogram_group_width_l856_85630

/-- Represents a group in a frequency histogram -/
structure HistogramGroup where
  a : ℝ
  b : ℝ
  m : ℝ  -- frequency
  h : ℝ  -- height
  h_pos : h > 0
  m_pos : m > 0
  a_lt_b : a < b

/-- 
The absolute value of the group width |a-b| in a frequency histogram 
is equal to the frequency m divided by the height h.
-/
theorem histogram_group_width (g : HistogramGroup) : 
  |g.b - g.a| = g.m / g.h := by
  sorry

end NUMINAMATH_CALUDE_histogram_group_width_l856_85630


namespace NUMINAMATH_CALUDE_sum_of_ace_l856_85620

/-- Given 5 children with player numbers, prove that the sum of numbers for A, C, and E is 24 -/
theorem sum_of_ace (a b c d e : ℕ) : 
  a + b + c + d + e = 35 →
  b + c = 13 →
  a + b + c + e = 31 →
  b + c + e = 21 →
  b = 7 →
  a + c + e = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ace_l856_85620


namespace NUMINAMATH_CALUDE_square_perimeter_l856_85676

theorem square_perimeter (s : ℝ) (h : s > 0) : 
  (2 * s = 32) → (4 * s = 64) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l856_85676


namespace NUMINAMATH_CALUDE_root_sum_cubes_l856_85621

theorem root_sum_cubes (r s t : ℝ) : 
  (6 * r^3 + 1506 * r + 3009 = 0) →
  (6 * s^3 + 1506 * s + 3009 = 0) →
  (6 * t^3 + 1506 * t + 3009 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 1504.5 := by
sorry

end NUMINAMATH_CALUDE_root_sum_cubes_l856_85621


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l856_85664

theorem diophantine_equation_solution (x y : ℤ) :
  5 * x - 7 * y = 3 →
  ∃ t : ℤ, x = 7 * t - 12 ∧ y = 5 * t - 9 :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l856_85664


namespace NUMINAMATH_CALUDE_triangular_front_view_solids_l856_85654

/-- Enumeration of possible solids --/
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

/-- Definition of a solid with a triangular front view --/
def has_triangular_front_view (s : Solid) : Prop :=
  match s with
  | Solid.TriangularPyramid => True
  | Solid.SquarePyramid => True
  | Solid.TriangularPrism => True
  | Solid.Cone => True
  | _ => False

/-- Theorem stating that a solid with a triangular front view must be one of the specified solids --/
theorem triangular_front_view_solids (s : Solid) :
  has_triangular_front_view s →
  (s = Solid.TriangularPyramid ∨ s = Solid.SquarePyramid ∨ s = Solid.TriangularPrism ∨ s = Solid.Cone) :=
by
  sorry

end NUMINAMATH_CALUDE_triangular_front_view_solids_l856_85654


namespace NUMINAMATH_CALUDE_ellipse_properties_l856_85647

/-- Ellipse C in the Cartesian coordinate system -/
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Line intersecting ellipse C -/
def L (k x : ℝ) : ℝ := k * (x - 4)

/-- Point A: left vertex of ellipse C -/
def A : ℝ × ℝ := (-2, 0)

/-- Point M: first intersection of line L and ellipse C -/
noncomputable def M (k : ℝ) : ℝ × ℝ := 
  let x₁ := (16 * k^2 + 4 * k * Real.sqrt (1 - 12 * k^2)) / (1 + 4 * k^2)
  (x₁, L k x₁)

/-- Point N: second intersection of line L and ellipse C -/
noncomputable def N (k : ℝ) : ℝ × ℝ := 
  let x₂ := (16 * k^2 - 4 * k * Real.sqrt (1 - 12 * k^2)) / (1 + 4 * k^2)
  (x₂, L k x₂)

/-- Point P: intersection of x = 1 and line BM -/
noncomputable def P (k : ℝ) : ℝ × ℝ := 
  let x₁ := (M k).1
  (1, k * (x₁ - 4) / (x₁ - 2))

/-- Area of triangle OMN -/
noncomputable def area_OMN (k : ℝ) : ℝ := 
  8 * Real.sqrt (1/k^2 - 12) / (1/k^2 + 4)

theorem ellipse_properties (k : ℝ) (hk : k ≠ 0) :
  (∃ (t : ℝ), t • (A.1 - (P k).1, A.2 - (P k).2) = ((N k).1 - A.1, (N k).2 - A.2)) ∧
  (∀ (k : ℝ), k ≠ 0 → area_OMN k ≤ 1) ∧
  (∃ (k : ℝ), k ≠ 0 ∧ area_OMN k = 1) := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l856_85647


namespace NUMINAMATH_CALUDE_exists_all_accessible_l856_85661

-- Define the type for cities
variable {City : Type}

-- Define the accessibility relation
variable (accessible : City → City → Prop)

-- Define the property that a city can access itself
variable (self_accessible : ∀ c : City, accessible c c)

-- Define the property that for any two cities, there's a third city that can access both
variable (exists_common_accessible : ∀ p q : City, ∃ r : City, accessible p r ∧ accessible q r)

-- The theorem to prove
theorem exists_all_accessible :
  ∃ c : City, ∀ other : City, accessible other c :=
sorry

end NUMINAMATH_CALUDE_exists_all_accessible_l856_85661


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l856_85636

theorem complex_fraction_equality (a b : ℝ) :
  (a / (1 - Complex.I)) + (b / (2 - Complex.I)) = 1 / (3 - Complex.I) →
  a = -1/5 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l856_85636


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l856_85606

theorem quadratic_complete_square (x : ℝ) : 
  (∃ r s : ℝ, (6 * x^2 - 24 * x - 54 = 0) ↔ ((x + r)^2 = s)) → 
  (∃ r s : ℝ, (6 * x^2 - 24 * x - 54 = 0) ↔ ((x + r)^2 = s) ∧ r + s = 11) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l856_85606


namespace NUMINAMATH_CALUDE_arbitrary_triangle_angle_ratio_not_arbitrary_quadrilateral_angle_ratio_not_arbitrary_pentagon_angle_ratio_l856_85658

-- Triangle
theorem arbitrary_triangle_angle_ratio (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧ A / a = B / b ∧ B / b = C / c :=
sorry

-- Convex Quadrilateral
theorem not_arbitrary_quadrilateral_angle_ratio :
  ∃ (p q r s : ℝ), p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧
    ¬∃ (A B C D : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧
      A + B + C + D = 360 ∧
      A < B + C + D ∧ B < A + C + D ∧ C < A + B + D ∧ D < A + B + C ∧
      A / p = B / q ∧ B / q = C / r ∧ C / r = D / s :=
sorry

-- Convex Pentagon
theorem not_arbitrary_pentagon_angle_ratio :
  ∃ (u v w x y : ℝ), u > 0 ∧ v > 0 ∧ w > 0 ∧ x > 0 ∧ y > 0 ∧
    ¬∃ (A B C D E : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧ E > 0 ∧
      A + B + C + D + E = 540 ∧
      2 * A < B + C + D + E ∧ 2 * B < A + C + D + E ∧ 2 * C < A + B + D + E ∧
      2 * D < A + B + C + E ∧ 2 * E < A + B + C + D ∧
      A / u = B / v ∧ B / v = C / w ∧ C / w = D / x ∧ D / x = E / y :=
sorry

end NUMINAMATH_CALUDE_arbitrary_triangle_angle_ratio_not_arbitrary_quadrilateral_angle_ratio_not_arbitrary_pentagon_angle_ratio_l856_85658


namespace NUMINAMATH_CALUDE_negation_equivalence_l856_85632

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, x^2 + 3*x + 2 < 0) ↔ (∀ x : ℝ, x^2 + 3*x + 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l856_85632


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l856_85684

theorem arithmetic_calculation : 90 + 5 * 12 / (180 / 3) = 91 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l856_85684


namespace NUMINAMATH_CALUDE_increase_when_multiplied_l856_85693

theorem increase_when_multiplied (n : ℕ) (m : ℕ) (increase : ℕ) : n = 14 → m = 15 → increase = m * n - n → increase = 196 := by
  sorry

end NUMINAMATH_CALUDE_increase_when_multiplied_l856_85693


namespace NUMINAMATH_CALUDE_solution_set_theorem_inequality_theorem_l856_85608

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part I
theorem solution_set_theorem :
  {x : ℝ | f (x - 1) + f (x + 3) ≥ 6} = {x : ℝ | x ≤ -3 ∨ x ≥ 3} := by sorry

-- Theorem for part II
theorem inequality_theorem (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) (ha_neq_zero : a ≠ 0) :
  f (a * b) > |a| * f (b / a) := by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_inequality_theorem_l856_85608


namespace NUMINAMATH_CALUDE_correct_rounding_l856_85671

def round_to_nearest_hundred (x : ℤ) : ℤ :=
  (x + 50) / 100 * 100

theorem correct_rounding : round_to_nearest_hundred ((58 + 44) * 3) = 300 := by
  sorry

end NUMINAMATH_CALUDE_correct_rounding_l856_85671


namespace NUMINAMATH_CALUDE_token_game_ends_in_37_rounds_l856_85600

/-- Represents a player in the token game -/
inductive Player : Type
  | A
  | B
  | C

/-- Represents the state of the game at any point -/
structure GameState :=
  (tokens : Player → Nat)
  (round : Nat)

/-- Determines if a player's tokens are divisible by 5 -/
def isDivisibleByFive (n : Nat) : Bool :=
  n % 5 = 0

/-- Determines the player with the most tokens -/
def playerWithMostTokens (state : GameState) : Player :=
  sorry

/-- Applies the rules of a single round to the game state -/
def applyRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (i.e., a player has run out of tokens) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- The initial state of the game -/
def initialState : GameState :=
  { tokens := λ p => match p with
    | Player.A => 17
    | Player.B => 15
    | Player.C => 14,
    round := 0 }

/-- The final state of the game -/
def finalState : GameState :=
  sorry

theorem token_game_ends_in_37_rounds :
  finalState.round = 37 ∧ isGameOver finalState :=
  sorry

end NUMINAMATH_CALUDE_token_game_ends_in_37_rounds_l856_85600


namespace NUMINAMATH_CALUDE_rectangle_area_integer_l856_85616

theorem rectangle_area_integer (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (n : ℕ), (a + b) * Real.sqrt (a * b) = n) ↔ (a = 9 ∧ b = 4) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_integer_l856_85616


namespace NUMINAMATH_CALUDE_sandras_mother_contribution_sandras_mother_contribution_proof_l856_85648

theorem sandras_mother_contribution : ℝ → Prop :=
  fun m =>
    let savings : ℝ := 10
    let father_contribution : ℝ := 2 * m
    let total_money : ℝ := savings + m + father_contribution
    let candy_cost : ℝ := 0.5
    let jelly_bean_cost : ℝ := 0.2
    let candy_quantity : ℕ := 14
    let jelly_bean_quantity : ℕ := 20
    let total_cost : ℝ := candy_cost * candy_quantity + jelly_bean_cost * jelly_bean_quantity
    let money_left : ℝ := 11
    total_money = total_cost + money_left → m = 4

theorem sandras_mother_contribution_proof : ∃ m, sandras_mother_contribution m :=
  sorry

end NUMINAMATH_CALUDE_sandras_mother_contribution_sandras_mother_contribution_proof_l856_85648


namespace NUMINAMATH_CALUDE_estimate_greater_than_exact_l856_85610

theorem estimate_greater_than_exact (a b c d : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (a' b' c' d' : ℕ) 
  (ha' : a' ≥ a) (hb' : b' ≤ b) (hc' : c' ≥ c) (hd' : d' ≥ d) : 
  (a' : ℚ) / b' + c' - d' > (a : ℚ) / b + c - d :=
by sorry

end NUMINAMATH_CALUDE_estimate_greater_than_exact_l856_85610


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l856_85617

def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {x | x^2 = x}

theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l856_85617


namespace NUMINAMATH_CALUDE_max_distance_complex_circle_l856_85686

theorem max_distance_complex_circle : 
  ∃ (M : ℝ), M = 7 ∧ 
  ∀ (z : ℂ), Complex.abs (z - (4 - 4*I)) ≤ 2 → Complex.abs (z - 1) ≤ M ∧ 
  ∃ (w : ℂ), Complex.abs (w - (4 - 4*I)) ≤ 2 ∧ Complex.abs (w - 1) = M :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_circle_l856_85686


namespace NUMINAMATH_CALUDE_four_propositions_l856_85644

-- Define a square
def Square (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ), S = {(x, y) | 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ a}

-- Define a rectangle
def Rectangle (R : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ), R = {(x, y) | 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ b}

theorem four_propositions :
  (∀ x : ℝ, x^2 - x + 1/4 ≥ 0) ∧
  (∀ S : Set (ℝ × ℝ), Square S → Rectangle S) ∧
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ∧
  (∃ x : ℝ, x^3 + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_four_propositions_l856_85644


namespace NUMINAMATH_CALUDE_club_choices_l856_85692

/-- Represents a club with boys and girls -/
structure Club where
  boys : ℕ
  girls : ℕ

/-- The number of ways to choose a president and vice-president of the same gender -/
def sameGenderChoices (c : Club) : ℕ :=
  c.boys * (c.boys - 1) + c.girls * (c.girls - 1)

/-- Theorem stating that for a club with 10 boys and 10 girls, 
    there are 180 ways to choose a president and vice-president of the same gender -/
theorem club_choices (c : Club) (h1 : c.boys = 10) (h2 : c.girls = 10) :
  sameGenderChoices c = 180 := by
  sorry

#check club_choices

end NUMINAMATH_CALUDE_club_choices_l856_85692


namespace NUMINAMATH_CALUDE_parabola_with_directrix_neg_one_l856_85666

/-- A parabola is defined by its directrix and focus. This structure represents a parabola with a vertical directrix. -/
structure Parabola where
  /-- The x-coordinate of the directrix -/
  directrix : ℝ

/-- The standard equation of a parabola with a vertical directrix -/
def standardEquation (p : Parabola) : Prop :=
  ∀ x y : ℝ, (y^2 = 4*(x - p.directrix/2))

/-- Theorem: For a parabola with directrix x = -1, its standard equation is y^2 = 4x -/
theorem parabola_with_directrix_neg_one (p : Parabola) (h : p.directrix = -1) :
  standardEquation p ↔ ∀ x y : ℝ, (y^2 = 4*x) :=
sorry

end NUMINAMATH_CALUDE_parabola_with_directrix_neg_one_l856_85666


namespace NUMINAMATH_CALUDE_sequence_can_be_arithmetic_and_geometric_l856_85602

theorem sequence_can_be_arithmetic_and_geometric :
  ∃ (a d : ℝ) (n : ℕ), a + d = 9 ∧ a + n * d = 729 ∧ a = 3 ∧
  ∃ (b r : ℝ) (m : ℕ), b * r = 9 ∧ b * r^m = 729 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_can_be_arithmetic_and_geometric_l856_85602


namespace NUMINAMATH_CALUDE_equal_intercepts_implies_a_value_l856_85670

/-- Given two points A(0, 1) and B(4, a) on a line, if the x-intercept and y-intercept of the line are equal, then a = -3. -/
theorem equal_intercepts_implies_a_value (a : ℝ) : 
  let A : ℝ × ℝ := (0, 1)
  let B : ℝ × ℝ := (4, a)
  let m : ℝ := (a - 1) / 4  -- Slope of the line AB
  let x_intercept : ℝ := 4 / (1 - m)  -- x-intercept formula
  let y_intercept : ℝ := a - m * 4  -- y-intercept formula
  x_intercept = y_intercept → a = -3 :=
by sorry

end NUMINAMATH_CALUDE_equal_intercepts_implies_a_value_l856_85670


namespace NUMINAMATH_CALUDE_twelve_chairs_adjacent_subsets_l856_85634

/-- The number of subsets containing at least three adjacent chairs 
    when n chairs are arranged in a circle. -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ := sorry

/-- Theorem stating that for 12 chairs arranged in a circle, 
    the number of subsets containing at least three adjacent chairs is 2040. -/
theorem twelve_chairs_adjacent_subsets : 
  subsets_with_adjacent_chairs 12 = 2040 := by sorry

end NUMINAMATH_CALUDE_twelve_chairs_adjacent_subsets_l856_85634


namespace NUMINAMATH_CALUDE_smallest_n_for_red_vertices_symmetry_l856_85609

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry

/-- A set of 5 red vertices in a regular polygon -/
def RedVertices (n : ℕ) := Fin 5 → Fin n

/-- An axis of symmetry of a regular polygon -/
def AxisOfSymmetry (n : ℕ) := ℕ

/-- Checks if a vertex is reflected onto another vertex across an axis -/
def isReflectedOnto (n : ℕ) (p : RegularPolygon n) (v1 v2 : Fin n) (axis : AxisOfSymmetry n) : Prop :=
  sorry

/-- The main theorem -/
theorem smallest_n_for_red_vertices_symmetry :
  (∀ n : ℕ, n ≥ 14 →
    ∀ p : RegularPolygon n,
    ∀ red : RedVertices n,
    ∃ axis : AxisOfSymmetry n,
    ∀ v1 v2 : Fin 5, v1 ≠ v2 → ¬isReflectedOnto n p (red v1) (red v2) axis) ∧
  (∀ n : ℕ, n < 14 →
    ∃ p : RegularPolygon n,
    ∃ red : RedVertices n,
    ∀ axis : AxisOfSymmetry n,
    ∃ v1 v2 : Fin 5, v1 ≠ v2 ∧ isReflectedOnto n p (red v1) (red v2) axis) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_red_vertices_symmetry_l856_85609


namespace NUMINAMATH_CALUDE_hyperbola_focus_k_value_l856_85642

/-- Theorem: For a hyperbola with equation 8kx^2 - ky^2 = 8 and one focus at (0, -3), the value of k is -1. -/
theorem hyperbola_focus_k_value (k : ℝ) : 
  (∀ x y : ℝ, 8 * k * x^2 - k * y^2 = 8) → -- hyperbola equation
  (∃ x : ℝ, (x, -3) ∈ {(x, y) | x^2 / (8 / k) + y^2 / (8 / k + 1) = 1}) → -- focus at (0, -3)
  k = -1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_k_value_l856_85642


namespace NUMINAMATH_CALUDE_quadratic_solution_range_l856_85640

theorem quadratic_solution_range (m : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^2 + (m-1)*x + 1 = 0) → m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_range_l856_85640


namespace NUMINAMATH_CALUDE_sector_area_90_degrees_l856_85607

/-- The area of a sector with radius 2 and central angle 90° is π. -/
theorem sector_area_90_degrees : 
  let r : ℝ := 2
  let angle_degrees : ℝ := 90
  let angle_radians : ℝ := angle_degrees * (π / 180)
  let sector_area : ℝ := (1/2) * r^2 * angle_radians
  sector_area = π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_90_degrees_l856_85607


namespace NUMINAMATH_CALUDE_complete_graph_edges_six_vertices_l856_85603

theorem complete_graph_edges_six_vertices :
  let n : ℕ := 6
  let E : ℕ := n * (n - 1) / 2
  E = 15 := by sorry

end NUMINAMATH_CALUDE_complete_graph_edges_six_vertices_l856_85603


namespace NUMINAMATH_CALUDE_inequality_proof_l856_85672

theorem inequality_proof (a b : ℝ) (h : a * b > 0) : b / a + a / b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l856_85672


namespace NUMINAMATH_CALUDE_range_of_p_l856_85677

def h (x : ℝ) : ℝ := 4 * x + 3

def p (x : ℝ) : ℝ := h (h (h (h x)))

theorem range_of_p :
  ∀ y ∈ Set.range p, -1 ≤ y ∧ y ≤ 1023 ∧
  ∀ z, -1 ≤ z ∧ z ≤ 1023 → ∃ x, -1 ≤ x ∧ x ≤ 3 ∧ p x = z :=
sorry

end NUMINAMATH_CALUDE_range_of_p_l856_85677


namespace NUMINAMATH_CALUDE_circumcircle_area_l856_85626

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right-angled at A
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- AB = 6
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 36 ∧
  -- AC = 8
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 64

-- Define the circumcircle of the triangle
def Circumcircle (A B C : ℝ × ℝ) : ℝ → Prop :=
  λ r => ∃ (center : ℝ × ℝ),
    (center.1 - A.1)^2 + (center.2 - A.2)^2 = r^2 ∧
    (center.1 - B.1)^2 + (center.2 - B.2)^2 = r^2 ∧
    (center.1 - C.1)^2 + (center.2 - C.2)^2 = r^2

-- Theorem statement
theorem circumcircle_area (A B C : ℝ × ℝ) :
  Triangle A B C →
  ∃ r, Circumcircle A B C r ∧ π * r^2 = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_area_l856_85626


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l856_85645

/-- Given a complex number z satisfying (3+4i)z=25, prove that z is in the fourth quadrant -/
theorem z_in_fourth_quadrant (z : ℂ) (h : (3 + 4*I) * z = 25) : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l856_85645


namespace NUMINAMATH_CALUDE_smallest_divisible_number_l856_85646

theorem smallest_divisible_number : ∃ N : ℕ,
  (∀ k : ℕ, 2 ≤ k → k ≤ 10 → (N + k) % k = 0) ∧
  (∀ M : ℕ, M < N → ∃ j : ℕ, 2 ≤ j ∧ j ≤ 10 ∧ (M + j) % j ≠ 0) ∧
  N = 2520 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_number_l856_85646


namespace NUMINAMATH_CALUDE_price_changes_l856_85601

theorem price_changes (jacket_price : ℝ) : 
  let shoes_price := 1.1 * jacket_price
  let jacket_price_after_initial_reduction := 0.85 * jacket_price
  let jacket_price_after_sale := 0.7 * jacket_price_after_initial_reduction
  let shoes_price_after_sale := 0.7 * shoes_price
  let jacket_price_after_tax := 1.1 * jacket_price_after_sale
  let shoes_price_after_tax := 1.1 * shoes_price_after_sale
  let combined_price_after_tax := jacket_price_after_tax + shoes_price_after_tax
  let price_increase := (jacket_price + shoes_price) - combined_price_after_tax
  let final_combined_price := combined_price_after_tax + price_increase
  final_combined_price = 2.1 * jacket_price :=
by sorry

end NUMINAMATH_CALUDE_price_changes_l856_85601


namespace NUMINAMATH_CALUDE_total_distance_is_66_l856_85638

def first_museum_distance : ℕ := 5
def second_museum_distance : ℕ := 15
def cultural_center_distance : ℕ := 10
def detour_distance : ℕ := 3

def total_distance : ℕ :=
  2 * (first_museum_distance + detour_distance) +
  2 * second_museum_distance +
  2 * cultural_center_distance

theorem total_distance_is_66 : total_distance = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_66_l856_85638


namespace NUMINAMATH_CALUDE_smallest_base_for_inequality_l856_85694

theorem smallest_base_for_inequality (k : ℕ) (h : k = 7) : 
  (∃ (base : ℕ), base^k > 4^20 ∧ ∀ (b : ℕ), b < base → b^k ≤ 4^20) ↔ 64^k > 4^20 ∧ ∀ (b : ℕ), b < 64 → b^k ≤ 4^20 :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_inequality_l856_85694


namespace NUMINAMATH_CALUDE_sqrt_6_simplest_l856_85627

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → x = Real.sqrt y → ¬∃ a b : ℝ, a > 0 ∧ b > 1 ∧ y = a * b^2

theorem sqrt_6_simplest :
  is_simplest_sqrt (Real.sqrt 6) ∧
  ¬is_simplest_sqrt (Real.sqrt 8) ∧
  ¬is_simplest_sqrt (Real.sqrt (1/3)) ∧
  ¬is_simplest_sqrt (Real.sqrt 4) :=
sorry

end NUMINAMATH_CALUDE_sqrt_6_simplest_l856_85627


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l856_85663

theorem absolute_value_inequality (x : ℝ) :
  abs (x - 2) + abs (x - 1) ≥ 5 ↔ x ≤ -1 ∨ x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l856_85663


namespace NUMINAMATH_CALUDE_park_area_is_1500000_l856_85643

/-- Represents the scale of the map in miles per inch -/
def scale : ℝ := 250

/-- Represents the length of the park on the map in inches -/
def map_length : ℝ := 6

/-- Represents the width of the park on the map in inches -/
def map_width : ℝ := 4

/-- Calculates the actual area of the park in square miles -/
def park_area : ℝ := (map_length * scale) * (map_width * scale)

/-- Theorem stating that the actual area of the park is 1500000 square miles -/
theorem park_area_is_1500000 : park_area = 1500000 := by
  sorry

end NUMINAMATH_CALUDE_park_area_is_1500000_l856_85643


namespace NUMINAMATH_CALUDE_cookie_making_time_l856_85635

/-- Proves that the time to make dough and cool cookies is equal to the total time minus the sum of baking time and icing hardening times. -/
theorem cookie_making_time (total_time baking_time white_icing_time chocolate_icing_time : ℕ)
  (h1 : total_time = 120)
  (h2 : baking_time = 15)
  (h3 : white_icing_time = 30)
  (h4 : chocolate_icing_time = 30) :
  total_time - (baking_time + white_icing_time + chocolate_icing_time) = 45 :=
by sorry

end NUMINAMATH_CALUDE_cookie_making_time_l856_85635


namespace NUMINAMATH_CALUDE_potato_price_proof_l856_85624

/-- The original price of a bag of potatoes in rubles -/
def original_price : ℝ := 250

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase factor -/
def andrey_increase : ℝ := 2

/-- Boris's first price increase factor -/
def boris_first_increase : ℝ := 1.6

/-- Boris's second price increase factor -/
def boris_second_increase : ℝ := 1.4

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The difference in earnings between Boris and Andrey -/
def earnings_difference : ℝ := 1200

theorem potato_price_proof :
  let andrey_earning := bags_bought * original_price * andrey_increase
  let boris_first_earning := boris_first_sale * original_price * boris_first_increase
  let boris_second_earning := boris_second_sale * original_price * boris_first_increase * boris_second_increase
  boris_first_earning + boris_second_earning - andrey_earning = earnings_difference :=
by sorry

end NUMINAMATH_CALUDE_potato_price_proof_l856_85624


namespace NUMINAMATH_CALUDE_quadratic_function_unique_solution_l856_85623

/-- Given a quadratic function f(x) = ax² + bx + c, prove that if f(-1) = 3, f(0) = 1, and f(1) = 1, then a = 1, b = -1, and c = 1. -/
theorem quadratic_function_unique_solution (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = (fun x => if x = -1 then 3 else if x = 0 then 1 else if x = 1 then 1 else 0) x) →
  a = 1 ∧ b = -1 ∧ c = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_solution_l856_85623


namespace NUMINAMATH_CALUDE_new_ratio_after_addition_l856_85631

theorem new_ratio_after_addition (a b : ℤ) : 
  (a : ℚ) / b = 1 / 4 →
  b = 72 →
  (a + 6 : ℚ) / b = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_new_ratio_after_addition_l856_85631


namespace NUMINAMATH_CALUDE_initial_ratio_of_partners_to_associates_l856_85685

theorem initial_ratio_of_partners_to_associates 
  (partners : ℕ) 
  (associates : ℕ) 
  (h1 : partners = 18) 
  (h2 : associates + 45 = 34 * partners) : 
  (2 : ℕ) / (63 : ℕ) = partners / associates :=
sorry

end NUMINAMATH_CALUDE_initial_ratio_of_partners_to_associates_l856_85685


namespace NUMINAMATH_CALUDE_portion_filled_in_twenty_minutes_l856_85651

/-- Represents the portion of a cistern filled by a pipe in a given time. -/
def portion_filled (time : ℝ) : ℝ := sorry

/-- The time it takes to fill a certain portion of the cistern. -/
def fill_time : ℝ := 20

/-- Theorem stating that the portion filled in 20 minutes is 1. -/
theorem portion_filled_in_twenty_minutes :
  portion_filled fill_time = 1 := by sorry

end NUMINAMATH_CALUDE_portion_filled_in_twenty_minutes_l856_85651


namespace NUMINAMATH_CALUDE_max_value_of_f_l856_85665

-- Define the function
def f (x : ℝ) : ℝ := -4 * x^2 + 10

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M ∧ M = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l856_85665


namespace NUMINAMATH_CALUDE_fred_cards_after_purchase_l856_85680

/-- The number of baseball cards Fred has after Melanie's purchase -/
def fred_remaining_cards (initial : ℕ) (bought : ℕ) : ℕ :=
  initial - bought

/-- Theorem: Fred has 2 baseball cards left after Melanie's purchase -/
theorem fred_cards_after_purchase :
  fred_remaining_cards 5 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fred_cards_after_purchase_l856_85680


namespace NUMINAMATH_CALUDE_volleyball_match_probability_l856_85688

-- Define the probability of team A winning a set in the first four sets
def p_win_first_four : ℚ := 2 / 3

-- Define the probability of team A winning the fifth set
def p_win_fifth : ℚ := 1 / 2

-- Define the number of ways to choose 2 wins out of 4 sets
def ways_to_win_two_of_four : ℕ := 6

-- State the theorem
theorem volleyball_match_probability :
  let p_three_two := ways_to_win_two_of_four * p_win_first_four^2 * (1 - p_win_first_four)^2 * p_win_fifth
  p_three_two = 4 / 27 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_match_probability_l856_85688


namespace NUMINAMATH_CALUDE_rhombus_area_l856_85695

/-- The area of a rhombus with side length 3 cm and an acute angle of 45 degrees is 9√2/2 square centimeters. -/
theorem rhombus_area (side_length : ℝ) (acute_angle : ℝ) :
  side_length = 3 →
  acute_angle = 45 * π / 180 →
  let area := side_length * side_length * Real.sin acute_angle
  area = 9 * Real.sqrt 2 / 2 := by
  sorry

#check rhombus_area

end NUMINAMATH_CALUDE_rhombus_area_l856_85695


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l856_85649

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem twentieth_term_of_sequence :
  arithmetic_sequence 2 3 20 = 59 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l856_85649


namespace NUMINAMATH_CALUDE_train_length_calculation_l856_85612

/-- Given a train crossing a bridge, calculate its length. -/
theorem train_length_calculation (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 255 →
  ∃ (train_length : ℝ), train_length = train_speed * crossing_time - bridge_length ∧ train_length = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l856_85612


namespace NUMINAMATH_CALUDE_roots_of_equation_l856_85629

theorem roots_of_equation (x : ℝ) : 
  (x - 3)^2 = 4 ↔ x = 5 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l856_85629


namespace NUMINAMATH_CALUDE_uno_card_price_l856_85678

/-- The original price of an Uno Giant Family Card -/
def original_price : ℝ := 12

/-- The number of cards purchased -/
def num_cards : ℕ := 10

/-- The discount applied to each card -/
def discount : ℝ := 2

/-- The total amount paid -/
def total_paid : ℝ := 100

/-- Theorem stating that the original price satisfies the given conditions -/
theorem uno_card_price : 
  num_cards * (original_price - discount) = total_paid := by
  sorry


end NUMINAMATH_CALUDE_uno_card_price_l856_85678


namespace NUMINAMATH_CALUDE_max_ratio_is_half_l856_85611

/-- A hyperbola with equation x^2 - y^2 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p | p.1^2 - p.2^2 = 1}

/-- The right focus of the hyperbola -/
def RightFocus : ℝ × ℝ := sorry

/-- The right directrix of the hyperbola -/
def RightDirectrix : Set (ℝ × ℝ) := sorry

/-- The right branch of the hyperbola -/
def RightBranch : Set (ℝ × ℝ) := sorry

/-- The projection of a point onto the right directrix -/
def ProjectOntoDirectrix (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The distance between two points -/
def Distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The midpoint of two points -/
def Midpoint (p q : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Statement: The maximum value of |MN|/|AB| is 1/2 -/
theorem max_ratio_is_half :
  ∀ A B : ℝ × ℝ,
  A ∈ RightBranch →
  B ∈ RightBranch →
  Distance A RightFocus * Distance B RightFocus = 0 →  -- This represents AF ⟂ BF
  let M := Midpoint A B
  let N := ProjectOntoDirectrix M
  Distance M N / Distance A B ≤ 1/2 := by
sorry

end NUMINAMATH_CALUDE_max_ratio_is_half_l856_85611


namespace NUMINAMATH_CALUDE_percentage_runs_by_running_l856_85614

def total_runs : ℕ := 150
def boundaries : ℕ := 6
def sixes : ℕ := 4
def no_balls : ℕ := 8
def wide_balls : ℕ := 5
def leg_byes : ℕ := 2

def runs_from_boundaries : ℕ := boundaries * 4
def runs_from_sixes : ℕ := sixes * 6
def runs_not_from_bat : ℕ := no_balls + wide_balls + leg_byes

def runs_by_running : ℕ := total_runs - runs_not_from_bat - (runs_from_boundaries + runs_from_sixes)

theorem percentage_runs_by_running :
  (runs_by_running : ℚ) / total_runs * 100 = 58 := by
  sorry

end NUMINAMATH_CALUDE_percentage_runs_by_running_l856_85614


namespace NUMINAMATH_CALUDE_hyperbola_focus_to_asymptote_distance_l856_85699

-- Define the hyperbola and its properties
theorem hyperbola_focus_to_asymptote_distance :
  ∀ (M : ℝ × ℝ),
  let F₁ : ℝ × ℝ := (-Real.sqrt 10, 0)
  let F₂ : ℝ × ℝ := (Real.sqrt 10, 0)
  let MF₁ : ℝ × ℝ := (M.1 - F₁.1, M.2 - F₁.2)
  let MF₂ : ℝ × ℝ := (M.2 - F₂.1, M.2 - F₂.2)
  -- M is on the hyperbola
  -- MF₁ · MF₂ = 0
  (MF₁.1 * MF₂.1 + MF₁.2 * MF₂.2 = 0) →
  -- |MF₁| · |MF₂| = 2
  (Real.sqrt (MF₁.1^2 + MF₁.2^2) * Real.sqrt (MF₂.1^2 + MF₂.2^2) = 2) →
  -- The distance from a focus to one of its asymptotes is 1
  (1 : ℝ) = 
    (Real.sqrt 10) / Real.sqrt (1 + (1/3)^2) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_focus_to_asymptote_distance_l856_85699


namespace NUMINAMATH_CALUDE_vasyas_numbers_l856_85696

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x + y = x / y) (h3 : x * y = x / y) :
  x = 1/2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l856_85696


namespace NUMINAMATH_CALUDE_product_equals_two_l856_85689

theorem product_equals_two : 
  (∀ (a b c : ℝ), a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) →
  6 * 15 * 5 = 2 :=
by sorry

end NUMINAMATH_CALUDE_product_equals_two_l856_85689


namespace NUMINAMATH_CALUDE_relay_for_life_total_miles_l856_85687

/-- Calculates the total miles walked in a relay event -/
def total_miles_walked (john_speed bob_speed alice_speed : ℝ) 
                       (john_time bob_time alice_time : ℝ) : ℝ :=
  john_speed * john_time + alice_speed * alice_time + bob_speed * bob_time

/-- The combined total miles walked by John, Alice, and Bob during the Relay for Life event -/
theorem relay_for_life_total_miles : 
  total_miles_walked 3.5 4 2.8 4 6 8 = 62.8 := by
  sorry

end NUMINAMATH_CALUDE_relay_for_life_total_miles_l856_85687


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l856_85622

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 20 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 5 + a 7 + a 9 + a 11 = 20

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) : 
  a 1 + a 13 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l856_85622


namespace NUMINAMATH_CALUDE_sine_cosine_difference_l856_85605

theorem sine_cosine_difference (θ₁ θ₂ : Real) :
  Real.sin (37.5 * π / 180) * Real.cos (7.5 * π / 180) -
  Real.cos (37.5 * π / 180) * Real.sin (7.5 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_difference_l856_85605


namespace NUMINAMATH_CALUDE_f_increasing_decreasing_l856_85604

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

theorem f_increasing_decreasing :
  let f : ℝ → ℝ := λ x => x^2 - Real.log x
  (∀ x, x > 0 → f x ∈ Set.univ) ∧
  (∀ x y, x > Real.sqrt 2 / 2 → y > Real.sqrt 2 / 2 → x < y → f x < f y) ∧
  (∀ x y, x > 0 → y > 0 → x < Real.sqrt 2 / 2 → y < Real.sqrt 2 / 2 → x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_decreasing_l856_85604


namespace NUMINAMATH_CALUDE_patty_weeks_without_chores_l856_85637

/-- Calculates the number of weeks Patty can go without doing chores --/
def weeks_without_chores (
  cookies_per_chore : ℕ) 
  (chores_per_kid_per_week : ℕ) 
  (money_available : ℕ) 
  (cookies_per_pack : ℕ) 
  (cost_per_pack : ℕ) 
  (num_siblings : ℕ) : ℕ :=
  let packs_bought := money_available / cost_per_pack
  let total_cookies := packs_bought * cookies_per_pack
  let cookies_per_sibling_per_week := chores_per_kid_per_week * cookies_per_chore
  let cookies_needed_per_week := cookies_per_sibling_per_week * num_siblings
  total_cookies / cookies_needed_per_week

theorem patty_weeks_without_chores :
  weeks_without_chores 3 4 15 24 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_patty_weeks_without_chores_l856_85637


namespace NUMINAMATH_CALUDE_permutations_of_eight_distinct_objects_l856_85683

theorem permutations_of_eight_distinct_objects : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_eight_distinct_objects_l856_85683


namespace NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l856_85697

theorem quadratic_inequality_no_solution (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - m * x + (m - 1) ≤ 0) ↔ m ≤ -2 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l856_85697


namespace NUMINAMATH_CALUDE_obstacle_course_time_l856_85619

/-- Represents the times for each segment of the obstacle course -/
structure ObstacleCourse :=
  (first_run : List Int)
  (door_opening : Int)
  (second_run : List Int)

/-- Calculates the total time to complete the obstacle course -/
def total_time (course : ObstacleCourse) : Int :=
  (course.first_run.sum + course.door_opening + course.second_run.sum)

/-- The theorem to prove -/
theorem obstacle_course_time :
  let course := ObstacleCourse.mk [225, 130, 88, 45, 120] 73 [175, 108, 75, 138]
  total_time course = 1177 := by
  sorry

end NUMINAMATH_CALUDE_obstacle_course_time_l856_85619


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l856_85655

theorem decimal_sum_to_fraction : 
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 = 733 / 12500 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l856_85655


namespace NUMINAMATH_CALUDE_geometric_sequence_10th_term_l856_85653

theorem geometric_sequence_10th_term :
  let a₁ : ℚ := 5
  let r : ℚ := 5 / 3
  let n : ℕ := 10
  let aₙ := a₁ * r^(n - 1)
  aₙ = 9765625 / 19683 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_10th_term_l856_85653


namespace NUMINAMATH_CALUDE_k_range_for_inequality_l856_85641

theorem k_range_for_inequality (k : ℝ) : 
  k ≠ 0 → 
  (k^2 * 1^2 - 6*k*1 + 8 ≥ 0) → 
  (k ≥ 4 ∨ k ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_k_range_for_inequality_l856_85641


namespace NUMINAMATH_CALUDE_discount_calculation_l856_85656

/-- Calculates the percentage discount given the original price and sale price -/
def percentage_discount (original_price sale_price : ℚ) : ℚ :=
  (original_price - sale_price) / original_price * 100

/-- Proves that the percentage discount for an item with original price $25 and sale price $18.75 is 25% -/
theorem discount_calculation :
  let original_price : ℚ := 25
  let sale_price : ℚ := 37/2  -- Representing 18.75 as a rational number
  percentage_discount original_price sale_price = 25 := by
  sorry


end NUMINAMATH_CALUDE_discount_calculation_l856_85656


namespace NUMINAMATH_CALUDE_squaredigital_numbers_l856_85650

/-- Sum of digits of a non-negative integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A number is squaredigital if it equals the square of the sum of its digits -/
def is_squaredigital (n : ℕ) : Prop := n = (sum_of_digits n)^2

/-- The only squaredigital numbers are 0, 1, and 81 -/
theorem squaredigital_numbers : 
  ∀ n : ℕ, is_squaredigital n ↔ n = 0 ∨ n = 1 ∨ n = 81 := by sorry

end NUMINAMATH_CALUDE_squaredigital_numbers_l856_85650


namespace NUMINAMATH_CALUDE_candy_problem_l856_85674

theorem candy_problem (x : ℚ) : 
  (2/9 * x - 2/3 - 4 = 8) → x = 57 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l856_85674


namespace NUMINAMATH_CALUDE_penny_revenue_l856_85662

/-- Calculates the total money earned from selling cheesecake pies -/
def cheesecake_revenue (price_per_slice : ℕ) (slices_per_pie : ℕ) (pies_sold : ℕ) : ℕ :=
  price_per_slice * slices_per_pie * pies_sold

/-- Proves that Penny makes $294 from selling 7 cheesecake pies -/
theorem penny_revenue : cheesecake_revenue 7 6 7 = 294 := by
  sorry

end NUMINAMATH_CALUDE_penny_revenue_l856_85662


namespace NUMINAMATH_CALUDE_minimum_guests_with_both_l856_85681

theorem minimum_guests_with_both (total : ℕ) 
  (sunglasses : ℕ) (wristbands : ℕ) (both : ℕ) : 
  (3 : ℚ) / 7 * total = sunglasses →
  (4 : ℚ) / 9 * total = wristbands →
  total = sunglasses + wristbands - both →
  total ≥ 63 →
  both ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_minimum_guests_with_both_l856_85681


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l856_85691

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 3}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l856_85691


namespace NUMINAMATH_CALUDE_max_cables_cut_theorem_l856_85615

/-- Represents a computer network -/
structure ComputerNetwork where
  num_computers : ℕ
  num_cables : ℕ
  num_clusters : ℕ

/-- Calculates the maximum number of cables that can be cut -/
def max_cables_cut (network : ComputerNetwork) : ℕ :=
  network.num_cables - (network.num_computers - network.num_clusters)

/-- Theorem stating the maximum number of cables that can be cut -/
theorem max_cables_cut_theorem (network : ComputerNetwork) 
  (h1 : network.num_computers = 200)
  (h2 : network.num_cables = 345)
  (h3 : network.num_clusters = 8) :
  max_cables_cut network = 153 := by
  sorry

#eval max_cables_cut ⟨200, 345, 8⟩

end NUMINAMATH_CALUDE_max_cables_cut_theorem_l856_85615


namespace NUMINAMATH_CALUDE_cut_cube_theorem_l856_85679

/-- Represents a cube that has been cut into smaller cubes -/
structure CutCube where
  size : ℕ  -- The number of smaller cubes along each edge
  painted_surfaces : ℕ  -- The number of painted surfaces on the original cube

/-- Counts the number of smaller cubes with no painted surfaces -/
def count_unpainted (c : CutCube) : ℕ :=
  (c.size - 2)^3

/-- The total number of smaller cubes -/
def total_cubes (c : CutCube) : ℕ :=
  c.size^3

/-- Theorem: If a cube is cut so that there's exactly one unpainted cube, 
    then the total number of smaller cubes is 27 -/
theorem cut_cube_theorem (c : CutCube) 
    (h1 : c.painted_surfaces = 6)  -- All surfaces of the original cube are painted
    (h2 : count_unpainted c = 1)   -- There is exactly one unpainted smaller cube
    : total_cubes c = 27 := by
  sorry


end NUMINAMATH_CALUDE_cut_cube_theorem_l856_85679


namespace NUMINAMATH_CALUDE_gravel_density_l856_85669

/-- Proves that the density of gravel is approximately 267 kg/m³ given the conditions of the bucket problem. -/
theorem gravel_density (bucket_volume : ℝ) (additional_water : ℝ) (full_bucket_weight : ℝ) (empty_bucket_weight : ℝ) 
  (h1 : bucket_volume = 12)
  (h2 : additional_water = 3)
  (h3 : full_bucket_weight = 28)
  (h4 : empty_bucket_weight = 1)
  (h5 : ∀ x, x > 0 → x * 1 = x) -- 1 liter of water weighs 1 kg
  : ∃ (density : ℝ), abs (density - 267) < 1 ∧ 
    density = (full_bucket_weight - empty_bucket_weight - additional_water) / 
              (bucket_volume - additional_water) * 1000 := by
  sorry

end NUMINAMATH_CALUDE_gravel_density_l856_85669


namespace NUMINAMATH_CALUDE_train_length_l856_85652

/-- Calculates the length of a train given its speed, time to pass a bridge, and the bridge length -/
theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) : 
  speed = 45 → time = 36.8 → bridge_length = 140 → 
  (speed * 1000 / 3600) * time - bridge_length = 320 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l856_85652


namespace NUMINAMATH_CALUDE_hillshire_population_l856_85628

theorem hillshire_population (num_cities : ℕ) (avg_lower : ℕ) (avg_upper : ℕ) :
  num_cities = 25 →
  avg_lower = 5000 →
  avg_upper = 5500 →
  (num_cities : ℝ) * ((avg_lower : ℝ) + (avg_upper : ℝ)) / 2 = 131250 :=
by sorry

end NUMINAMATH_CALUDE_hillshire_population_l856_85628


namespace NUMINAMATH_CALUDE_problem_solution_l856_85639

theorem problem_solution : 
  (Real.sqrt 75 + Real.sqrt 27 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 8 * Real.sqrt 3 + Real.sqrt 6) ∧
  ((Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) - (Real.sqrt 5 - 1)^2 = 2 * Real.sqrt 5 - 5) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l856_85639


namespace NUMINAMATH_CALUDE_prob_at_least_two_women_l856_85668

/-- The probability of selecting at least 2 women from a group of 8 men and 4 women when choosing 4 people at random -/
theorem prob_at_least_two_women (total : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) : 
  total = men + women →
  men = 8 →
  women = 4 →
  selected = 4 →
  (1 : ℚ) - (Nat.choose men selected / Nat.choose total selected + 
    (Nat.choose women 1 * Nat.choose men (selected - 1)) / Nat.choose total selected) = 67 / 165 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_women_l856_85668


namespace NUMINAMATH_CALUDE_distance_on_quadratic_curve_l856_85618

/-- The distance between two points on a quadratic curve -/
theorem distance_on_quadratic_curve (m k a c : ℝ) :
  let y (x : ℝ) := m * x^2 + k
  let point1 := (a, y a)
  let point2 := (c, y c)
  let distance := Real.sqrt ((c - a)^2 + (y c - y a)^2)
  distance = |a - c| * Real.sqrt (1 + m^2 * (c + a)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_on_quadratic_curve_l856_85618


namespace NUMINAMATH_CALUDE_prob_same_color_proof_l856_85682

def total_balls : ℕ := 4
def red_balls : ℕ := 2
def white_balls : ℕ := 2

def prob_same_color : ℚ := 1 / 3

theorem prob_same_color_proof :
  let prob_red := red_balls / total_balls * (red_balls - 1) / (total_balls - 1)
  let prob_white := white_balls / total_balls * (white_balls - 1) / (total_balls - 1)
  prob_red + prob_white = prob_same_color :=
sorry

end NUMINAMATH_CALUDE_prob_same_color_proof_l856_85682
