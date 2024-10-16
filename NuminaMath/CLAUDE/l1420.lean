import Mathlib

namespace NUMINAMATH_CALUDE_a_51_value_l1420_142028

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) - a n = 2

theorem a_51_value (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 51 = 101 := by
  sorry

end NUMINAMATH_CALUDE_a_51_value_l1420_142028


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l1420_142063

theorem average_of_a_and_b (a b c : ℝ) 
  (h1 : (a + b) / 2 = 50)
  (h2 : (b + c) / 2 = 70)
  (h3 : c - a = 40) :
  (a + b) / 2 = 50 := by
sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l1420_142063


namespace NUMINAMATH_CALUDE_line_equation_l1420_142019

-- Define the circle C
def Circle (x y : ℝ) : Prop := x^2 + (y-1)^2 = 5

-- Define the line l
def Line (m x y : ℝ) : Prop := m*x - y + 1 - m = 0

-- Define the condition that P(1,1) satisfies 2⃗AP = ⃗PB
def PointCondition (xa ya xb yb : ℝ) : Prop :=
  2*(1 - xa, 1 - ya) = (xb - 1, yb - 1)

theorem line_equation :
  ∀ (m : ℝ) (xa ya xb yb : ℝ),
    Circle xa ya → Circle xb yb →  -- A and B are on the circle
    Line m xa ya → Line m xb yb →  -- A and B are on the line
    PointCondition xa ya xb yb →   -- P(1,1) satisfies 2⃗AP = ⃗PB
    (m = 1 ∨ m = -1) :=             -- The slope of the line is either 1 or -1
by sorry

end NUMINAMATH_CALUDE_line_equation_l1420_142019


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1420_142081

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 40)
  (triple_minus_four : 3 * y - 4 * x = 14) :
  |y - x| = 9.714 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1420_142081


namespace NUMINAMATH_CALUDE_sum_of_squared_sums_l1420_142048

theorem sum_of_squared_sums (a b c : ℝ) : 
  (a^3 - 15*a^2 + 17*a - 8 = 0) →
  (b^3 - 15*b^2 + 17*b - 8 = 0) →
  (c^3 - 15*c^2 + 17*c - 8 = 0) →
  (a+b)^2 + (b+c)^2 + (c+a)^2 = 416 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squared_sums_l1420_142048


namespace NUMINAMATH_CALUDE_faye_bought_30_songs_l1420_142009

/-- The number of songs Faye bought -/
def total_songs (country_albums pop_albums songs_per_album : ℕ) : ℕ :=
  (country_albums + pop_albums) * songs_per_album

/-- Proof that Faye bought 30 songs -/
theorem faye_bought_30_songs :
  total_songs 2 3 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_faye_bought_30_songs_l1420_142009


namespace NUMINAMATH_CALUDE_sand_in_last_bag_l1420_142069

theorem sand_in_last_bag (total_sand : Nat) (bag_capacity : Nat) (h1 : total_sand = 757) (h2 : bag_capacity = 65) :
  total_sand % bag_capacity = 42 := by
sorry

end NUMINAMATH_CALUDE_sand_in_last_bag_l1420_142069


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1420_142017

theorem complex_equation_solution (z : ℂ) : (3 - 4*I)*z = 5*I → z = 4/5 + 3/5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1420_142017


namespace NUMINAMATH_CALUDE_orthogonality_condition_l1420_142039

/-- Two circles are orthogonal if their tangents at intersection points are perpendicular -/
def orthogonal (R₁ R₂ d : ℝ) : Prop :=
  d^2 = R₁^2 + R₂^2

theorem orthogonality_condition (R₁ R₂ d : ℝ) (h₁ : R₁ > 0) (h₂ : R₂ > 0) (h₃ : d > 0) :
  orthogonal R₁ R₂ d ↔ d^2 = R₁^2 + R₂^2 :=
sorry

end NUMINAMATH_CALUDE_orthogonality_condition_l1420_142039


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_S_l1420_142055

/-- The set of numbers from 9 to 999999999, where each number consists of all 9s -/
def S : Finset ℕ := Finset.image (λ i => (10^i - 1) / 9) (Finset.range 9)

/-- The arithmetic mean of the set S -/
def M : ℕ := (Finset.sum S id) / Finset.card S

theorem arithmetic_mean_of_S : M = 123456789 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_S_l1420_142055


namespace NUMINAMATH_CALUDE_min_swaps_upper_bound_min_swaps_lower_bound_min_swaps_exact_l1420_142033

/-- A swap operation on a matrix -/
def swap (M : Matrix (Fin n) (Fin n) ℕ) (i j k l : Fin n) : Matrix (Fin n) (Fin n) ℕ :=
  sorry

/-- Predicate to check if a matrix contains all numbers from 1 to n² -/
def valid_matrix (M : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  sorry

/-- The number of swaps needed to transform one matrix into another -/
def swaps_needed (A B : Matrix (Fin n) (Fin n) ℕ) : ℕ :=
  sorry

theorem min_swaps_upper_bound (n : ℕ) (h : n ≥ 2) :
  ∃ m : ℕ, ∀ (A B : Matrix (Fin n) (Fin n) ℕ),
    valid_matrix A → valid_matrix B →
    swaps_needed A B ≤ m ∧
    m = 2 * n * (n - 1) :=
  sorry

theorem min_swaps_lower_bound (n : ℕ) (h : n ≥ 2) :
  ∃ (A B : Matrix (Fin n) (Fin n) ℕ),
    valid_matrix A ∧ valid_matrix B ∧
    swaps_needed A B = 2 * n * (n - 1) :=
  sorry

theorem min_swaps_exact (n : ℕ) (h : n ≥ 2) :
  ∃! m : ℕ, (∀ (A B : Matrix (Fin n) (Fin n) ℕ),
    valid_matrix A → valid_matrix B →
    swaps_needed A B ≤ m) ∧
    (∃ (A B : Matrix (Fin n) (Fin n) ℕ),
      valid_matrix A ∧ valid_matrix B ∧
      swaps_needed A B = m) ∧
    m = 2 * n * (n - 1) :=
  sorry

end NUMINAMATH_CALUDE_min_swaps_upper_bound_min_swaps_lower_bound_min_swaps_exact_l1420_142033


namespace NUMINAMATH_CALUDE_largest_number_less_than_two_l1420_142037

theorem largest_number_less_than_two : 
  let numbers : Finset ℝ := {0.8, 1/2, 0.5}
  ∀ x ∈ numbers, x < 2 → 
  ∃ max ∈ numbers, ∀ y ∈ numbers, y ≤ max ∧ max = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_less_than_two_l1420_142037


namespace NUMINAMATH_CALUDE_circle_condition_l1420_142015

theorem circle_condition (x y m : ℝ) : 
  (∃ (a b r : ℝ), r > 0 ∧ (x - a)^2 + (y - b)^2 = r^2 ↔ x^2 + y^2 - x + y + m = 0) → 
  m < (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_circle_condition_l1420_142015


namespace NUMINAMATH_CALUDE_min_value_of_ab_l1420_142012

theorem min_value_of_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 8) :
  a * b ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_ab_l1420_142012


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1420_142099

-- Problem 1
theorem problem_1 : 2 * Real.sqrt 3 * 315 * 612 = 36600 := by sorry

-- Problem 2
theorem problem_2 : 2 * (Real.log 10 / Real.log 5) + (Real.log 0.25 / Real.log 5) = 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1420_142099


namespace NUMINAMATH_CALUDE_triangle_side_length_l1420_142035

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 5 →
  c = 2 →
  Real.cos A = 2 / 3 →
  b^2 + c^2 - a^2 = 2 * b * c * Real.cos A →
  b = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1420_142035


namespace NUMINAMATH_CALUDE_probability_yellow_ball_l1420_142092

theorem probability_yellow_ball (total_balls : ℕ) (yellow_balls : ℕ) 
  (h1 : total_balls = 8) (h2 : yellow_balls = 5) :
  (yellow_balls : ℚ) / total_balls = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_yellow_ball_l1420_142092


namespace NUMINAMATH_CALUDE_function_properties_l1420_142042

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / 4 + a / 4 - Real.log x - 3 / 2

theorem function_properties (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ (deriv (f a)) x = -2) →
  (a = 5 / 4 ∧
   ∀ x : ℝ, x > 0 → x < 5 → (deriv (f a)) x < 0) ∧
  (∀ x : ℝ, x > 5 → (deriv (f a)) x > 0) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1420_142042


namespace NUMINAMATH_CALUDE_max_dot_product_CA_CB_l1420_142024

/-- Given planar vectors OA, OB, and OC satisfying certain conditions,
    the maximum value of CA · CB is 3. -/
theorem max_dot_product_CA_CB (OA OB OC : ℝ × ℝ) : 
  (OA.1 * OB.1 + OA.2 * OB.2 = 0) →  -- OA · OB = 0
  (OA.1^2 + OA.2^2 = 1) →            -- |OA| = 1
  (OC.1^2 + OC.2^2 = 1) →            -- |OC| = 1
  (OB.1^2 + OB.2^2 = 3) →            -- |OB| = √3
  (∃ (CA CB : ℝ × ℝ), 
    CA = (OA.1 - OC.1, OA.2 - OC.2) ∧ 
    CB = (OB.1 - OC.1, OB.2 - OC.2) ∧
    ∀ (CA' CB' : ℝ × ℝ), 
      CA' = (OA.1 - OC.1, OA.2 - OC.2) → 
      CB' = (OB.1 - OC.1, OB.2 - OC.2) →
      CA'.1 * CB'.1 + CA'.2 * CB'.2 ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_max_dot_product_CA_CB_l1420_142024


namespace NUMINAMATH_CALUDE_floor_expression_equals_eight_l1420_142000

theorem floor_expression_equals_eight :
  ⌊(3005^3 : ℝ) / (3003 * 3004) - (3003^3 : ℝ) / (3004 * 3005)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_equals_eight_l1420_142000


namespace NUMINAMATH_CALUDE_added_amount_l1420_142016

theorem added_amount (x : ℝ) (y : ℝ) : 
  x = 15 → 3 * (2 * x + y) = 105 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_added_amount_l1420_142016


namespace NUMINAMATH_CALUDE_sin_330_degrees_l1420_142001

-- Define the angle in degrees
def angle : ℝ := 330

-- State the theorem
theorem sin_330_degrees : Real.sin (angle * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l1420_142001


namespace NUMINAMATH_CALUDE_identity_polynomial_form_l1420_142067

/-- A polynomial that satisfies the given identity. -/
def IdentityPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x * P (x - 1) = (x - 2) * P x

/-- The theorem stating the form of polynomials satisfying the identity. -/
theorem identity_polynomial_form (P : ℝ → ℝ) (h : IdentityPolynomial P) :
  ∃ a : ℝ, ∀ x : ℝ, P x = a * (x^2 - x) :=
by
  sorry

end NUMINAMATH_CALUDE_identity_polynomial_form_l1420_142067


namespace NUMINAMATH_CALUDE_hundred_chickens_problem_l1420_142020

theorem hundred_chickens_problem :
  ∀ x y z : ℕ,
  x + y + z = 100 →
  5 * x + 3 * y + (z / 3 : ℚ) = 100 →
  z = 81 →
  x = 8 ∧ y = 11 := by
sorry

end NUMINAMATH_CALUDE_hundred_chickens_problem_l1420_142020


namespace NUMINAMATH_CALUDE_binary_digit_difference_l1420_142010

theorem binary_digit_difference (n m : ℕ) (hn : n = 1280) (hm : m = 320) :
  (Nat.log 2 n + 1) - (Nat.log 2 m + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_digit_difference_l1420_142010


namespace NUMINAMATH_CALUDE_x_range_theorem_l1420_142078

-- Define the points and line
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 1)
def l (a : ℝ) : ℝ → ℝ := fun x => a - x

-- Define the condition for points relative to the line
def point_line_condition (a : ℝ) : Prop :=
  (l a O.1 = O.2 ∨ l a A.1 = A.2) ∨ (l a O.1 - O.2) * (l a A.1 - A.2) < 0

-- Define the function h
def h (a : ℝ) : ℝ := a^2 + 2*a + 3

-- State the theorem
theorem x_range_theorem :
  ∀ x : ℝ, (∀ a : ℝ, point_line_condition a → x^2 + 4*x - 2 ≤ h a) ↔ -5 ≤ x ∧ x ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_x_range_theorem_l1420_142078


namespace NUMINAMATH_CALUDE_new_students_count_l1420_142023

/-- Calculates the number of new students who came to school during the year. -/
def new_students (initial : ℕ) (left : ℕ) (final : ℕ) : ℕ :=
  final - (initial - left)

/-- Proves that the number of new students who came to school during the year is 42. -/
theorem new_students_count :
  new_students 4 3 43 = 42 := by
  sorry

end NUMINAMATH_CALUDE_new_students_count_l1420_142023


namespace NUMINAMATH_CALUDE_angle_d_measure_l1420_142074

/-- Given a triangle ABC with angles A = 85°, B = 34°, and C = 21°,
    if a smaller triangle is formed within ABC with one of its angles being D,
    then the measure of angle D is 140°. -/
theorem angle_d_measure (A B C D : Real) : 
  A = 85 → B = 34 → C = 21 → 
  A + B + C = 180 →
  ∃ (E F : Real), E ≥ 0 ∧ F ≥ 0 ∧ D + E + F = 180 ∧ A + B + C + E + F = 180 →
  D = 140 := by sorry

end NUMINAMATH_CALUDE_angle_d_measure_l1420_142074


namespace NUMINAMATH_CALUDE_quadratic_equation_root_values_l1420_142056

theorem quadratic_equation_root_values (a : ℝ) : 
  (∃ x : ℂ, x^2 - 2*a*x + a^2 - 4*a = 0 ∧ Complex.abs x = 3) →
  (a = 1 ∨ a = 9 ∨ a = 2 - Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_values_l1420_142056


namespace NUMINAMATH_CALUDE_baseball_card_count_l1420_142088

def final_card_count (initial_count : ℕ) : ℕ :=
  let after_maria := initial_count - (initial_count + 1) / 2
  let after_peter := after_maria - 1
  let final_count := after_peter * 3
  final_count

theorem baseball_card_count : final_card_count 15 = 18 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_count_l1420_142088


namespace NUMINAMATH_CALUDE_lizzie_wins_iff_composite_l1420_142082

/-- The game state represents the numbers on the blackboard -/
def GameState := List ℚ

/-- A move in the game is selecting a subset of numbers and replacing them with their average -/
def Move := List ℕ

/-- Represents whether a number is composite -/
def IsComposite (n : ℕ) : Prop := ∃ k, 1 < k ∧ k < n ∧ k ∣ n

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if all numbers in the game state are equal -/
def allEqual (state : GameState) : Prop :=
  sorry

/-- Represents a winning strategy for Lizzie -/
def WinningStrategy (n : ℕ) : Prop :=
  ∀ initialState : GameState, 
    initialState.length = n →
    ∃ moveSequence : List Move, 
      allEqual (moveSequence.foldl applyMove initialState)

theorem lizzie_wins_iff_composite (n : ℕ) (h : n ≥ 3) :
  WinningStrategy n ↔ IsComposite n :=
sorry

end NUMINAMATH_CALUDE_lizzie_wins_iff_composite_l1420_142082


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1420_142045

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![1, 0]
def b : Fin 2 → ℝ := ![2, 1]

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : Fin 2 → ℝ) : Prop :=
  ∃ t : ℝ, v = t • w ∨ w = t • v

theorem problem_1 (k : ℝ) :
  collinear (k • a - b) (a + 2 • b) ↔ k = -1/2 := by sorry

theorem problem_2 (m : ℝ) (A B C : Fin 2 → ℝ) :
  (B - A = 2 • a + 3 • b) →
  (C - B = a + m • b) →
  collinear (B - A) (C - B) →
  m = 3/2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1420_142045


namespace NUMINAMATH_CALUDE_compound_has_one_Al_l1420_142062

/-- The atomic weight of Aluminium in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- A compound with Aluminium and Iodine -/
structure Compound where
  Al_count : ℕ
  I_count : ℕ
  molecular_weight : ℝ

/-- The compound in question -/
def our_compound : Compound where
  Al_count := 1
  I_count := 3
  molecular_weight := 408

/-- Theorem stating that our compound has exactly 1 Aluminium atom -/
theorem compound_has_one_Al : 
  our_compound.Al_count = 1 ∧
  our_compound.I_count = 3 ∧
  our_compound.molecular_weight = 408 ∧
  (our_compound.Al_count : ℝ) * atomic_weight_Al + (our_compound.I_count : ℝ) * atomic_weight_I = our_compound.molecular_weight :=
by sorry

end NUMINAMATH_CALUDE_compound_has_one_Al_l1420_142062


namespace NUMINAMATH_CALUDE_coefficient_x_squared_zero_l1420_142085

theorem coefficient_x_squared_zero (x : ℝ) (a : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x ≠ 0, f x = (a + 1/x) * (1 + x)^4 ∧ 
   (∃ c₀ c₁ c₃ c₄ : ℝ, ∀ x ≠ 0, f x = c₀ + c₁*x + 0*x^2 + c₃*x^3 + c₄*x^4)) ↔ 
  a = -2/3 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_zero_l1420_142085


namespace NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_l1420_142051

-- Define the conditions
def p (x : ℝ) : Prop := x ≤ 1
def q (x : ℝ) : Prop := 1 / x < 1

-- Statement to prove
theorem neg_p_sufficient_not_necessary :
  (∀ x : ℝ, ¬(p x) → q x) ∧ ¬(∀ x : ℝ, q x → ¬(p x)) :=
sorry

end NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_l1420_142051


namespace NUMINAMATH_CALUDE_total_students_in_clubs_l1420_142091

def math_club_size : ℕ := 15
def science_club_size : ℕ := 10
def art_club_size : ℕ := 12
def math_science_overlap : ℕ := 5

theorem total_students_in_clubs : 
  math_club_size + science_club_size + art_club_size - math_science_overlap = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_students_in_clubs_l1420_142091


namespace NUMINAMATH_CALUDE_sum_4_equivalence_l1420_142058

-- Define the type for dice outcomes
def DiceOutcome := Fin 6

-- Define the type for a pair of dice outcomes
def DicePair := DiceOutcome × DiceOutcome

-- Define the sum of a pair of dice
def diceSum (pair : DicePair) : Nat :=
  pair.1.val + pair.2.val + 2

-- Define the event ξ = 4
def sumIs4 (pair : DicePair) : Prop :=
  diceSum pair = 4

-- Define the event where one die shows 3 and the other shows 1
def oneThreeOneOne (pair : DicePair) : Prop :=
  (pair.1.val = 2 ∧ pair.2.val = 0) ∨ (pair.1.val = 0 ∧ pair.2.val = 2)

-- Define the event where both dice show 2
def bothTwo (pair : DicePair) : Prop :=
  pair.1.val = 1 ∧ pair.2.val = 1

-- Theorem: ξ = 4 is equivalent to (one die shows 3 and the other shows 1) or (both dice show 2)
theorem sum_4_equivalence (pair : DicePair) :
  sumIs4 pair ↔ oneThreeOneOne pair ∨ bothTwo pair :=
by sorry

end NUMINAMATH_CALUDE_sum_4_equivalence_l1420_142058


namespace NUMINAMATH_CALUDE_exists_number_satisfying_equation_l1420_142031

theorem exists_number_satisfying_equation : ∃ N : ℝ, (0.47 * N - 0.36 * 1412) + 66 = 6 := by
  sorry

end NUMINAMATH_CALUDE_exists_number_satisfying_equation_l1420_142031


namespace NUMINAMATH_CALUDE_tangent_line_slope_l1420_142046

/-- Given that the line y = kx is tangent to the curve y = x + exp(-x), prove that k = 1 - exp(1) -/
theorem tangent_line_slope (k : ℝ) : 
  (∃ x₀ : ℝ, k * x₀ = x₀ + Real.exp (-x₀) ∧ 
              k = 1 - Real.exp (-x₀)) → 
  k = 1 - Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l1420_142046


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l1420_142086

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (d : Dimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Represents a cube -/
structure Cube where
  side : ℝ

/-- Theorem: The surface area of a rectangular solid remains unchanged
    when two unit cubes are removed from opposite corners -/
theorem surface_area_unchanged
  (solid : Dimensions)
  (cube : Cube)
  (h1 : solid.length = 2)
  (h2 : solid.width = 3)
  (h3 : solid.height = 4)
  (h4 : cube.side = 1) :
  surfaceArea solid = surfaceArea solid - 2 * (3 * cube.side^2) + 2 * (3 * cube.side^2) :=
by sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l1420_142086


namespace NUMINAMATH_CALUDE_lcm_180_560_l1420_142068

theorem lcm_180_560 : Nat.lcm 180 560 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_lcm_180_560_l1420_142068


namespace NUMINAMATH_CALUDE_min_value_sum_l1420_142079

/-- Given two circles C₁ and C₂, where C₁ always bisects the circumference of C₂,
    prove that the minimum value of 1/m + 2/n is 3 -/
theorem min_value_sum (m n : ℝ) : m > 0 → n > 0 → 
  (∀ x y : ℝ, (x - m)^2 + (y - 2*n)^2 = m^2 + 4*n^2 + 10 → 
              (x + 1)^2 + (y + 1)^2 = 2 → 
              ∃ k : ℝ, (m + 1)*x + (2*n + 1)*y + 5 = k * ((x + 1)^2 + (y + 1)^2 - 2)) →
  (1 / m + 2 / n) ≥ 3 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 1 / m₀ + 2 / n₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l1420_142079


namespace NUMINAMATH_CALUDE_base9_521_equals_base10_424_l1420_142002

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 9^2 + tens * 9^1 + ones * 9^0

/-- The theorem stating that 521 in base 9 is equal to 424 in base 10 -/
theorem base9_521_equals_base10_424 :
  base9ToBase10 5 2 1 = 424 := by
  sorry

end NUMINAMATH_CALUDE_base9_521_equals_base10_424_l1420_142002


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1420_142021

/-- The system of equations has exactly one solution if and only if 
    (3 - √5)/2 < t < (3 + √5)/2 -/
theorem unique_solution_condition (t : ℝ) : 
  (∃! x y z v : ℝ, x + y + z + v = 0 ∧ 
    (x*y + y*z + z*v) + t*(x*z + x*v + y*v) = 0) ↔ 
  ((3 - Real.sqrt 5) / 2 < t ∧ t < (3 + Real.sqrt 5) / 2) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1420_142021


namespace NUMINAMATH_CALUDE_myPolygonArea_l1420_142008

/-- A point in a 2D plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- A polygon defined by a list of points -/
def Polygon := List Point

/-- The polygon in question -/
def myPolygon : Polygon := [
  {x := 0, y := 0},
  {x := 0, y := 30},
  {x := 30, y := 30},
  {x := 30, y := 0}
]

/-- Calculate the area of a polygon -/
def calculateArea (p : Polygon) : ℤ :=
  sorry -- Implementation details omitted

/-- Theorem stating that the area of myPolygon is 15 square units -/
theorem myPolygonArea : calculateArea myPolygon = 15 := by
  sorry

end NUMINAMATH_CALUDE_myPolygonArea_l1420_142008


namespace NUMINAMATH_CALUDE_vessel_base_length_l1420_142057

/-- Given a cube immersed in a rectangular vessel, this theorem proves the length of the vessel's base. -/
theorem vessel_base_length 
  (cube_edge : ℝ) 
  (vessel_width : ℝ) 
  (water_rise : ℝ) 
  (h1 : cube_edge = 12)
  (h2 : vessel_width = 15)
  (h3 : water_rise = 5.76)
  : ∃ (vessel_length : ℝ), vessel_length = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_vessel_base_length_l1420_142057


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1420_142073

def U : Set ℕ := {x | x ≤ 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {4, 5, 6}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {4, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1420_142073


namespace NUMINAMATH_CALUDE_three_distinct_values_l1420_142053

-- Define a type for the expression
inductive Expr
  | const : ℕ → Expr
  | power : Expr → Expr → Expr

-- Define a function to evaluate the expression
def eval : Expr → ℕ
  | Expr.const n => n
  | Expr.power a b => (eval a) ^ (eval b)

-- Define the base expression
def baseExpr : Expr := Expr.const 3

-- Define a function to generate all possible parenthesizations
def allParenthesizations : Expr → List Expr
  | e => sorry  -- Implementation omitted

-- Theorem statement
theorem three_distinct_values :
  let allExpr := allParenthesizations (Expr.power (Expr.power (Expr.power baseExpr baseExpr) baseExpr) baseExpr)
  (allExpr.map eval).toFinset.card = 3 := by sorry


end NUMINAMATH_CALUDE_three_distinct_values_l1420_142053


namespace NUMINAMATH_CALUDE_product_of_numbers_l1420_142025

theorem product_of_numbers (x y : ℝ) 
  (h1 : x - y = 11) 
  (h2 : x^2 + y^2 = 185) : 
  x * y = 26 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1420_142025


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1420_142061

/-- A hyperbola sharing a focus with a parabola and having a specific eccentricity -/
def HyperbolaWithSharedFocus (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  ∃ (x₀ y₀ : ℝ), (x₀ = 2 ∧ y₀ = 0) ∧  -- Focus of parabola y² = 8x
  ∃ (c : ℝ), c = 2 ∧  -- Distance from center to focus
  ∃ (e : ℝ), e = 2 ∧ e = c / a  -- Eccentricity

/-- Theorem stating the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) 
  (h : HyperbolaWithSharedFocus a b) : 
  a = 1 ∧ b^2 = 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1420_142061


namespace NUMINAMATH_CALUDE_shirley_trefoil_boxes_l1420_142041

/-- The number of cases of boxes Shirley needs to deliver -/
def num_cases : ℕ := 5

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 2

/-- The total number of boxes Shirley sold -/
def total_boxes : ℕ := num_cases * boxes_per_case

theorem shirley_trefoil_boxes : total_boxes = 10 := by
  sorry

end NUMINAMATH_CALUDE_shirley_trefoil_boxes_l1420_142041


namespace NUMINAMATH_CALUDE_remaining_balls_for_given_n_l1420_142084

/-- Represents the remaining balls after the removal process -/
def RemainingBalls (n : ℕ) : Finset ℕ := sorry

/-- The rule for removing balls -/
def removalRule (n : ℕ) (i : ℕ) : Bool := sorry

/-- The recurrence relation for the remaining balls -/
def F (n : ℕ) (i : ℕ) : ℕ := sorry

theorem remaining_balls_for_given_n (n : ℕ) (h : n ≥ 56) :
  (RemainingBalls 56 = {10, 20, 29, 37, 56}) →
  (n = 57 → RemainingBalls n = {5, 16, 26, 35, 43}) ∧
  (n = 58 → RemainingBalls n = {11, 22, 32, 41, 49}) ∧
  (n = 59 → RemainingBalls n = {17, 28, 38, 47, 55}) ∧
  (n = 60 → RemainingBalls n = {11, 23, 34, 44, 53}) := by
  sorry

end NUMINAMATH_CALUDE_remaining_balls_for_given_n_l1420_142084


namespace NUMINAMATH_CALUDE_percentage_of_a_l1420_142080

theorem percentage_of_a (a b c : ℝ) (P : ℝ) : 
  (P / 100) * a = 8 →
  (8 / 100) * b = 4 →
  c = b / a →
  P = 16 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_a_l1420_142080


namespace NUMINAMATH_CALUDE_max_value_and_right_triangle_l1420_142087

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -x^3 + x^2 else a * Real.log x

theorem max_value_and_right_triangle (a : ℝ) :
  (∃ (m : ℝ), ∀ x ∈ Set.Icc (-1 : ℝ) (Real.exp 1), f a x ≤ m ∧
    (m = max 2 a ∨ (a < 2 ∧ m = 2))) ∧
  (a > 0 → ∃ (P Q : ℝ × ℝ),
    (P.1 > 0 ∧ P.2 = f a P.1) ∧
    (Q.1 < 0 ∧ Q.2 = f a Q.1) ∧
    (P.1 * Q.1 + P.2 * Q.2 = 0) ∧
    ((P.1 + Q.1) / 2 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_and_right_triangle_l1420_142087


namespace NUMINAMATH_CALUDE_football_tournament_max_points_l1420_142052

theorem football_tournament_max_points (n : ℕ) : 
  (∃ (scores : Fin 15 → ℕ), 
    (∀ i j : Fin 15, i ≠ j → scores i + scores j ≤ 3) ∧ 
    (∃ (successful : Finset (Fin 15)), 
      successful.card = 6 ∧ 
      ∀ i ∈ successful, n ≤ scores i)) →
  n ≤ 34 :=
sorry

end NUMINAMATH_CALUDE_football_tournament_max_points_l1420_142052


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1420_142060

theorem sum_of_roots_quadratic (a b : ℝ) : 
  (∀ x : ℝ, x^2 - (a+b)*x + a*b + 1 = 0 ↔ x = a ∨ x = b) → 
  a + b = a + b :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1420_142060


namespace NUMINAMATH_CALUDE_age_difference_theorem_l1420_142054

theorem age_difference_theorem (n : ℕ) 
  (ages : Fin n → ℕ) 
  (h1 : n = 5) 
  (h2 : ∃ (i j : Fin n), ages i = ages j + 1)
  (h3 : ∃ (i j : Fin n), ages i = ages j + 2)
  (h4 : ∃ (i j : Fin n), ages i = ages j + 3)
  (h5 : ∃ (i j : Fin n), ages i = ages j + 4) :
  ∃ (i j : Fin n), ages i = ages j + 10 :=
sorry

end NUMINAMATH_CALUDE_age_difference_theorem_l1420_142054


namespace NUMINAMATH_CALUDE_willey_farm_capital_l1420_142040

def total_land : ℕ := 4500
def corn_cost : ℕ := 42
def wheat_cost : ℕ := 35
def wheat_acres : ℕ := 3400

theorem willey_farm_capital :
  let corn_acres := total_land - wheat_acres
  let wheat_total_cost := wheat_cost * wheat_acres
  let corn_total_cost := corn_cost * corn_acres
  wheat_total_cost + corn_total_cost = 165200 := by sorry

end NUMINAMATH_CALUDE_willey_farm_capital_l1420_142040


namespace NUMINAMATH_CALUDE_additional_toothpicks_for_8_steps_l1420_142072

/-- The number of toothpicks needed for a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 4
  else toothpicks (n - 1) + 2 + 4 * (n - 1)

theorem additional_toothpicks_for_8_steps :
  toothpicks 4 = 30 →
  toothpicks 8 - toothpicks 4 = 88 :=
by sorry

end NUMINAMATH_CALUDE_additional_toothpicks_for_8_steps_l1420_142072


namespace NUMINAMATH_CALUDE_train_length_calculation_l1420_142047

/-- Calculates the length of a train given the speeds of a jogger and the train,
    the initial distance between them, and the time it takes for the train to pass the jogger. -/
theorem train_length_calculation (jogger_speed train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (5 / 18) →
  train_speed = 45 * (5 / 18) →
  initial_distance = 190 →
  passing_time = 31 →
  (train_speed - jogger_speed) * passing_time - initial_distance = 120 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1420_142047


namespace NUMINAMATH_CALUDE_intersection_distance_sum_l1420_142004

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (t, -1 + Real.sqrt 3 * t)

-- Define the curve C in polar coordinates
def curve_C (θ : ℝ) : ℝ := 2 * Real.sin θ + 2 * Real.cos θ

-- Define point P
def point_P : ℝ × ℝ := (0, -1)

-- Define the intersection points A and B
variable (A B : ℝ × ℝ)

-- Assume A and B are on both the line and the curve
axiom A_on_line : ∃ t : ℝ, line_l t = A
axiom B_on_line : ∃ t : ℝ, line_l t = B
axiom A_on_curve : ∃ θ : ℝ, (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ) = A
axiom B_on_curve : ∃ θ : ℝ, (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ) = B

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem intersection_distance_sum :
  1 / distance point_P A + 1 / distance point_P B = (2 * Real.sqrt 3 + 1) / 3 := by sorry

end NUMINAMATH_CALUDE_intersection_distance_sum_l1420_142004


namespace NUMINAMATH_CALUDE_comparison_theorems_l1420_142005

theorem comparison_theorems :
  (∀ a : ℝ, a < 0 → a / (a - 1) > 0) ∧
  (∀ x : ℝ, x < -1 → 2 / (x^2 - 1) > (x - 1) / (x^2 - 2*x + 1)) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x ≠ y → 2*x*y / (x + y) < (x + y) / 2) :=
by sorry

end NUMINAMATH_CALUDE_comparison_theorems_l1420_142005


namespace NUMINAMATH_CALUDE_prime_product_l1420_142018

theorem prime_product (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → 3 * p + 7 * q = 41 → (p + 1) * (q - 1) = 12 := by
  sorry

end NUMINAMATH_CALUDE_prime_product_l1420_142018


namespace NUMINAMATH_CALUDE_johns_running_speed_l1420_142027

/-- John's running problem -/
theorem johns_running_speed :
  ∀ (x : ℝ), -- x represents John's speed when running alone
  x > 0 → -- John's speed is positive
  6 * (1/2) + x * (1/2) = 5 → -- Total distance equation
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_johns_running_speed_l1420_142027


namespace NUMINAMATH_CALUDE_cube_side_ratio_l1420_142029

/-- The ratio of side lengths of two cubes with given weights -/
theorem cube_side_ratio (w₁ w₂ : ℝ) (h₁ : w₁ > 0) (h₂ : w₂ > 0) :
  w₁ = 7 → w₂ = 56 → (w₂ / w₁)^(1/3 : ℝ) = 2 := by
  sorry

#check cube_side_ratio

end NUMINAMATH_CALUDE_cube_side_ratio_l1420_142029


namespace NUMINAMATH_CALUDE_original_true_implies_contrapositive_true_l1420_142007

-- Define a proposition type
variable (P Q : Prop)

-- Define the contrapositive of an implication
def contrapositive (P Q : Prop) : Prop := ¬Q → ¬P

-- Theorem: If the original proposition is true, then its contrapositive is also true
theorem original_true_implies_contrapositive_true (h : P → Q) : contrapositive P Q :=
  sorry

end NUMINAMATH_CALUDE_original_true_implies_contrapositive_true_l1420_142007


namespace NUMINAMATH_CALUDE_read_distance_guangzhou_shenyang_l1420_142097

/-- Represents a number in words -/
inductive NumberWord
  | million : ℕ → NumberWord
  | thousand : ℕ → NumberWord
  | hundred : ℕ → NumberWord
  | ten : ℕ → NumberWord
  | one : ℕ → NumberWord

/-- Represents the distance from Guangzhou to Shenyang in meters -/
def distance_guangzhou_shenyang : ℕ := 3036000

/-- Converts a natural number to its word representation -/
def number_to_words (n : ℕ) : List NumberWord :=
  sorry

/-- Theorem stating that the correct way to read 3,036,000 is "three million thirty-six thousand" -/
theorem read_distance_guangzhou_shenyang :
  number_to_words distance_guangzhou_shenyang = 
    [NumberWord.million 3, NumberWord.thousand 36] :=
  sorry

end NUMINAMATH_CALUDE_read_distance_guangzhou_shenyang_l1420_142097


namespace NUMINAMATH_CALUDE_perpendicular_conditions_l1420_142032

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (line_perp_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (plane_perp_plane : Plane → Plane → Prop)
variable (plane_parallel_plane : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_conditions 
  (a b : Line) (α β : Plane) :
  (line_perp_plane a α ∧ line_perp_plane b β ∧ plane_perp_plane α β → perpendicular a b) ∧
  (line_in_plane a α ∧ line_perp_plane b β ∧ plane_parallel_plane α β → perpendicular a b) ∧
  (line_perp_plane a α ∧ line_parallel_plane b β ∧ plane_parallel_plane α β → perpendicular a b) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_conditions_l1420_142032


namespace NUMINAMATH_CALUDE_range_of_a_l1420_142043

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2*x + 3*a > 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1420_142043


namespace NUMINAMATH_CALUDE_day_of_week_theorem_l1420_142083

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a year -/
structure Year where
  number : ℕ

/-- Returns the day of the week for a given day number in a year -/
def dayOfWeek (y : Year) (day : ℕ) : DayOfWeek := sorry

/-- Checks if a year is a leap year -/
def isLeapYear (y : Year) : Bool := sorry

/-- The main theorem to prove -/
theorem day_of_week_theorem (N : Year) :
  dayOfWeek N 250 = DayOfWeek.Wednesday →
  dayOfWeek (Year.mk (N.number + 1)) 150 = DayOfWeek.Wednesday →
  dayOfWeek (Year.mk (N.number - 1)) 50 = DayOfWeek.Monday := by
  sorry

end NUMINAMATH_CALUDE_day_of_week_theorem_l1420_142083


namespace NUMINAMATH_CALUDE_circle_bisection_l1420_142098

/-- Given two circles in the plane:
    Circle 1: (x-a)^2 + (y-b)^2 = b^2 + 1
    Circle 2: (x+1)^2 + (y+1)^2 = 4
    If Circle 1 always bisects the circumference of Circle 2,
    then the relationship between a and b satisfies: a^2 + 2a + 2b + 5 = 0 -/
theorem circle_bisection (a b : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = b^2 + 1 → (x + 1)^2 + (y + 1)^2 = 4 → 
    ∃ t : ℝ, x = -1 + t * (2 + 2*a) ∧ y = -1 + t * (2 + 2*b)) → 
  a^2 + 2*a + 2*b + 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_bisection_l1420_142098


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l1420_142044

theorem square_plus_inverse_square (x : ℝ) (h : x + (1 / x) = 1.5) :
  x^2 + (1 / x^2) = 0.25 := by
sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l1420_142044


namespace NUMINAMATH_CALUDE_children_ages_exist_l1420_142059

theorem children_ages_exist :
  ∃ (a b c d : ℕ), 
    a + b + c + d = 33 ∧
    (a - 3) + (b - 3) + (c - 3) + (d - 3) = 22 ∧
    (a - 7) + (b - 7) + (c - 7) + (d - 7) = 11 ∧
    (a - 13) + (b - 13) + (c - 13) + (d - 13) = 1 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 :=
by sorry

end NUMINAMATH_CALUDE_children_ages_exist_l1420_142059


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_implies_m_range_l1420_142030

/-- Given a hyperbola with equation x² + y²/m = 1, if the asymptote's inclination angle α 
    is in the interval (0, π/3), then m is in the interval (-3, 0). -/
theorem hyperbola_asymptote_angle_implies_m_range (m : ℝ) (α : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2/m = 1 → ∃ k : ℝ, y = k*x ∧ Real.arctan k = α) →
  0 < α ∧ α < π/3 →
  -3 < m ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_implies_m_range_l1420_142030


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l1420_142089

theorem gain_percent_calculation (cost_price selling_price : ℝ) :
  50 * cost_price = 45 * selling_price →
  (selling_price - cost_price) / cost_price * 100 = 11.11 :=
by sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l1420_142089


namespace NUMINAMATH_CALUDE_nancy_marks_l1420_142095

theorem nancy_marks (history : ℕ) (home_economics : ℕ) (physical_education : ℕ) (art : ℕ) (average : ℕ) 
  (h1 : history = 75)
  (h2 : home_economics = 52)
  (h3 : physical_education = 68)
  (h4 : art = 89)
  (h5 : average = 70) :
  ∃ (american_literature : ℕ), 
    (history + home_economics + physical_education + art + american_literature) / 5 = average ∧ 
    american_literature = 66 := by
  sorry

end NUMINAMATH_CALUDE_nancy_marks_l1420_142095


namespace NUMINAMATH_CALUDE_solve_refrigerator_problem_l1420_142011

def refrigerator_problem (purchase_price installation_cost transport_cost selling_price : ℚ) : Prop :=
  let discount_rate : ℚ := 20 / 100
  let profit_rate : ℚ := 10 / 100
  let labelled_price : ℚ := purchase_price / (1 - discount_rate)
  let total_cost : ℚ := labelled_price + installation_cost + transport_cost
  (1 + profit_rate) * total_cost = selling_price

theorem solve_refrigerator_problem :
  refrigerator_problem 17500 250 125 24475 := by sorry

end NUMINAMATH_CALUDE_solve_refrigerator_problem_l1420_142011


namespace NUMINAMATH_CALUDE_caiden_roofing_cost_l1420_142036

/-- Calculates the cost of remaining metal roofing needed -/
def roofing_cost (total_required : ℕ) (free_provided : ℕ) (cost_per_foot : ℕ) : ℕ :=
  (total_required - free_provided) * cost_per_foot

/-- Theorem stating the cost calculation for Mr. Caiden's roofing -/
theorem caiden_roofing_cost :
  roofing_cost 300 250 8 = 400 := by
  sorry

end NUMINAMATH_CALUDE_caiden_roofing_cost_l1420_142036


namespace NUMINAMATH_CALUDE_fraction_subtraction_theorem_l1420_142049

theorem fraction_subtraction_theorem : 
  (3 + 6 + 9) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 6 + 9) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_theorem_l1420_142049


namespace NUMINAMATH_CALUDE_min_a_is_neg_two_l1420_142006

/-- The function f(x) as defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

/-- The theorem stating that -2 is the minimum value of a for which f(x) is always non-negative -/
theorem min_a_is_neg_two :
  (∀ a : ℝ, (∀ x : ℝ, f a x ≥ 0) → a ≥ -2) ∧
  (∀ x : ℝ, f (-2) x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_min_a_is_neg_two_l1420_142006


namespace NUMINAMATH_CALUDE_perpendicular_slope_is_five_thirds_l1420_142022

/-- The slope of a line perpendicular to the line containing points (3, 5) and (-2, 8) is 5/3 -/
theorem perpendicular_slope_is_five_thirds :
  let point1 : ℝ × ℝ := (3, 5)
  let point2 : ℝ × ℝ := (-2, 8)
  let slope_original := (point2.2 - point1.2) / (point2.1 - point1.1)
  let slope_perpendicular := -1 / slope_original
  slope_perpendicular = 5/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_slope_is_five_thirds_l1420_142022


namespace NUMINAMATH_CALUDE_managers_salary_correct_managers_salary_l1420_142064

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (avg_increase : ℝ) : ℝ :=
  let total_salary := num_employees * avg_salary
  let new_avg_salary := avg_salary + avg_increase
  let new_total_salary := (num_employees + 1) * new_avg_salary
  new_total_salary - total_salary

theorem correct_managers_salary :
  managers_salary 50 2000 250 = 14750 := by sorry

end NUMINAMATH_CALUDE_managers_salary_correct_managers_salary_l1420_142064


namespace NUMINAMATH_CALUDE_middle_part_of_proportional_distribution_l1420_142013

theorem middle_part_of_proportional_distribution (total : ℚ) (r1 r2 r3 : ℚ) :
  total = 120 →
  r1 = 1 →
  r2 = 1/4 →
  r3 = 1/8 →
  (r2 * total) / (r1 + r2 + r3) = 240/11 := by
  sorry

end NUMINAMATH_CALUDE_middle_part_of_proportional_distribution_l1420_142013


namespace NUMINAMATH_CALUDE_find_number_l1420_142093

theorem find_number : ∃ x : ℝ, 4.75 + 0.432 + x = 5.485 ∧ x = 0.303 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1420_142093


namespace NUMINAMATH_CALUDE_car_profit_percentage_l1420_142094

theorem car_profit_percentage (P : ℝ) (h : P > 0) : 
  let buying_price := P * (1 - 0.2)
  let selling_price := buying_price * (1 + 0.45)
  let profit := selling_price - P
  profit / P * 100 = 16 := by
sorry

end NUMINAMATH_CALUDE_car_profit_percentage_l1420_142094


namespace NUMINAMATH_CALUDE_expression_evaluation_l1420_142050

theorem expression_evaluation :
  let x : ℚ := -3
  let numerator := 5 + x * (5 + x) - 5^2
  let denominator := x - 5 + x^2
  numerator / denominator = -26 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1420_142050


namespace NUMINAMATH_CALUDE_range_of_a_for_line_separating_points_l1420_142014

/-- Given points A and B on opposite sides of the line 3x + 2y + a = 0, 
    prove that the range of a is (-19, -9) -/
theorem range_of_a_for_line_separating_points 
  (A B : ℝ × ℝ) 
  (h_A : A = (1, 3)) 
  (h_B : B = (5, 2)) 
  (h_opposite : (3 * A.1 + 2 * A.2 + a) * (3 * B.1 + 2 * B.2 + a) < 0) :
  ∀ a : ℝ, (a > -19 ∧ a < -9) ↔ 
    ∃ (x y : ℝ), (3 * x + 2 * y + a = 0 ∧ 
      (3 * A.1 + 2 * A.2 + a) * (3 * B.1 + 2 * B.2 + a) < 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_line_separating_points_l1420_142014


namespace NUMINAMATH_CALUDE_ted_work_time_l1420_142066

theorem ted_work_time (julie_rate ted_rate : ℚ) (julie_finish_time : ℚ) : 
  julie_rate = 1/10 →
  ted_rate = 1/8 →
  julie_finish_time = 999999999999999799 / 1000000000000000000 →
  ∃ t : ℚ, t = 4 ∧ (julie_rate + ted_rate) * t + julie_rate * julie_finish_time = 1 :=
by sorry

end NUMINAMATH_CALUDE_ted_work_time_l1420_142066


namespace NUMINAMATH_CALUDE_circle_circumference_l1420_142075

theorem circle_circumference (r : ℝ) (h : r = 4) : 
  2 * Real.pi * r = 8 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_l1420_142075


namespace NUMINAMATH_CALUDE_reyansh_farm_water_ratio_l1420_142071

/-- Represents the farm owned by Mr. Reyansh -/
structure Farm where
  num_cows : ℕ
  cow_water_daily : ℕ
  sheep_cow_ratio : ℕ
  total_water_weekly : ℕ

/-- Calculates the ratio of daily water consumption of a sheep to a cow -/
def water_consumption_ratio (f : Farm) : Rat :=
  let cow_water_weekly := f.num_cows * f.cow_water_daily * 7
  let sheep_water_weekly := f.total_water_weekly - cow_water_weekly
  let num_sheep := f.num_cows * f.sheep_cow_ratio
  let sheep_water_daily := sheep_water_weekly / (7 * num_sheep)
  sheep_water_daily / f.cow_water_daily

/-- Theorem stating that the water consumption ratio for Mr. Reyansh's farm is 1:4 -/
theorem reyansh_farm_water_ratio :
  let f : Farm := {
    num_cows := 40,
    cow_water_daily := 80,
    sheep_cow_ratio := 10,
    total_water_weekly := 78400
  }
  water_consumption_ratio f = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_reyansh_farm_water_ratio_l1420_142071


namespace NUMINAMATH_CALUDE_smallest_positive_period_of_even_odd_functions_l1420_142077

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function g is odd if g(x) = -g(-x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

/-- The period of a function f is p if f(x + p) = f(x) for all x -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

/-- The smallest positive period of a function -/
def SmallestPositivePeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ IsPeriod f p ∧ ∀ q, 0 < q ∧ q < p → ¬IsPeriod f q

theorem smallest_positive_period_of_even_odd_functions
  (f g : ℝ → ℝ) (c : ℝ)
  (hf : IsEven f)
  (hg : IsOdd g)
  (h : ∀ x, f x = -g (x + c))
  (hc : c > 0) :
  SmallestPositivePeriod f (4 * c) := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_period_of_even_odd_functions_l1420_142077


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1420_142034

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1420_142034


namespace NUMINAMATH_CALUDE_orange_count_theorem_l1420_142076

/-- The number of oranges initially in the box -/
def initial_oranges : ℝ := 55.0

/-- The number of oranges Susan adds to the box -/
def added_oranges : ℝ := 35.0

/-- The total number of oranges in the box after Susan adds more -/
def total_oranges : ℝ := 90.0

/-- Theorem stating that the initial number of oranges plus the added oranges equals the total oranges -/
theorem orange_count_theorem : initial_oranges + added_oranges = total_oranges := by
  sorry

end NUMINAMATH_CALUDE_orange_count_theorem_l1420_142076


namespace NUMINAMATH_CALUDE_quadratic_equation_complete_square_l1420_142038

theorem quadratic_equation_complete_square :
  ∃ (r s : ℝ), 
    (∀ x, 15 * x^2 - 60 * x - 135 = 0 ↔ (x + r)^2 = s) ∧
    r + s = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_complete_square_l1420_142038


namespace NUMINAMATH_CALUDE_kaleb_toy_purchase_l1420_142096

def number_of_toys (initial_money game_cost saving_amount toy_cost : ℕ) : ℕ :=
  ((initial_money - game_cost - saving_amount) / toy_cost)

theorem kaleb_toy_purchase :
  number_of_toys 12 8 2 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_toy_purchase_l1420_142096


namespace NUMINAMATH_CALUDE_curve_is_line_l1420_142090

/-- The equation (x^2 + y^2 - 2)√(x-3) = 0 represents a line -/
theorem curve_is_line : 
  ∃ (a b : ℝ), ∀ (x y : ℝ), (x^2 + y^2 - 2) * Real.sqrt (x - 3) = 0 → y = a * x + b :=
sorry

end NUMINAMATH_CALUDE_curve_is_line_l1420_142090


namespace NUMINAMATH_CALUDE_distance_between_points_l1420_142070

/-- The distance between two points given specific travel conditions -/
theorem distance_between_points (speed_A speed_B : ℝ) (stop_time : ℝ) : 
  speed_A = 80 →
  speed_B = 70 →
  stop_time = 1/4 →
  ∃ (distance : ℝ), 
    distance / speed_A = distance / speed_B - stop_time ∧
    distance = 2240 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1420_142070


namespace NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l1420_142026

/-- The function f(x) = x^2 + bx + 3 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 3

/-- Theorem stating that -3 is not in the range of f(x) if and only if b is in the open interval (-2√6, 2√6) -/
theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x, f b x ≠ -3) ↔ b ∈ Set.Ioo (-2 * Real.sqrt 6) (2 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l1420_142026


namespace NUMINAMATH_CALUDE_probability_three_tails_one_head_probability_three_tails_one_head_proof_l1420_142003

/-- The probability of getting exactly three tails and one head when tossing four coins simultaneously -/
theorem probability_three_tails_one_head : ℚ :=
  1 / 4

/-- Proof that the probability of getting exactly three tails and one head when tossing four coins simultaneously is 1/4 -/
theorem probability_three_tails_one_head_proof :
  probability_three_tails_one_head = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_tails_one_head_probability_three_tails_one_head_proof_l1420_142003


namespace NUMINAMATH_CALUDE_boat_travel_distance_l1420_142065

/-- Prove that the distance between two destinations is 40 km given the specified conditions. -/
theorem boat_travel_distance 
  (boatsman_speed : ℝ) 
  (river_speed : ℝ) 
  (time_difference : ℝ) 
  (h1 : boatsman_speed = 7)
  (h2 : river_speed = 3)
  (h3 : time_difference = 6)
  (h4 : (boatsman_speed + river_speed) * (boatsman_speed - river_speed) * time_difference = 
        2 * river_speed * boatsman_speed * (boatsman_speed - river_speed)) :
  (boatsman_speed + river_speed) * (boatsman_speed - river_speed) * time_difference / 
  (2 * river_speed) = 40 := by
  sorry

end NUMINAMATH_CALUDE_boat_travel_distance_l1420_142065
