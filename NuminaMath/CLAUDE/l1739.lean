import Mathlib

namespace NUMINAMATH_CALUDE_hilt_friends_cant_go_l1739_173945

/-- Given a total number of friends and the number of friends that can go to the movies,
    calculate the number of friends who can't go to the movies. -/
def friends_cant_go (total_friends : ℕ) (friends_going : ℕ) : ℕ :=
  total_friends - friends_going

/-- Theorem stating that with 25 total friends and 6 friends going to the movies,
    19 friends can't go to the movies. -/
theorem hilt_friends_cant_go :
  friends_cant_go 25 6 = 19 := by
  sorry

end NUMINAMATH_CALUDE_hilt_friends_cant_go_l1739_173945


namespace NUMINAMATH_CALUDE_max_integer_a_for_real_roots_l1739_173957

theorem max_integer_a_for_real_roots : 
  ∀ a : ℤ, (∃ x : ℝ, (a + 1 : ℝ) * x^2 - 2*x + 3 = 0) → a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_a_for_real_roots_l1739_173957


namespace NUMINAMATH_CALUDE_first_term_of_constant_ratio_l1739_173923

def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a + (n - 1 : ℕ) * d) / 2

theorem first_term_of_constant_ratio (a : ℚ) :
  (∃ c : ℚ, ∀ n : ℕ, n > 0 → arithmetic_sum a 5 (3 * n) / arithmetic_sum a 5 n = c) →
  a = 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_first_term_of_constant_ratio_l1739_173923


namespace NUMINAMATH_CALUDE_data_set_properties_l1739_173987

def data_set : List ℕ := [67, 57, 37, 40, 46, 62, 31, 47, 31, 30]

def mode (l : List ℕ) : ℕ := sorry

def range (l : List ℕ) : ℕ := sorry

def quantile (l : List ℕ) (p : ℚ) : ℚ := sorry

theorem data_set_properties :
  (mode data_set = 31) ∧
  (range data_set = 37) ∧
  (quantile data_set (1/10) = 30.5) := by
  sorry

end NUMINAMATH_CALUDE_data_set_properties_l1739_173987


namespace NUMINAMATH_CALUDE_estate_value_l1739_173960

/-- Represents the estate distribution problem --/
structure EstateDistribution where
  total : ℝ
  daughter_share : ℝ
  son_share : ℝ
  wife_share : ℝ
  brother_share : ℝ
  nanny_share : ℝ

/-- Theorem stating the conditions and the result to be proved --/
theorem estate_value (e : EstateDistribution) : 
  e.daughter_share + e.son_share = (3/5) * e.total ∧ 
  e.daughter_share = (5/7) * (e.daughter_share + e.son_share) ∧
  e.son_share = (2/7) * (e.daughter_share + e.son_share) ∧
  e.wife_share = 3 * e.son_share ∧
  e.brother_share = e.daughter_share ∧
  e.nanny_share = 400 ∧
  e.total = e.daughter_share + e.son_share + e.wife_share + e.brother_share + e.nanny_share
  →
  e.total = 825 := by
  sorry

#eval 825 -- To display the result

end NUMINAMATH_CALUDE_estate_value_l1739_173960


namespace NUMINAMATH_CALUDE_tax_savings_proof_l1739_173954

/-- Represents the tax brackets and rates -/
structure TaxBracket :=
  (lower : ℕ) (upper : ℕ) (rate : ℚ)

/-- Calculates the tax for a given income and tax brackets -/
def calculateTax (income : ℕ) (brackets : List TaxBracket) : ℚ :=
  sorry

/-- Represents the tax system -/
structure TaxSystem :=
  (brackets : List TaxBracket)
  (standardDeduction : ℕ)
  (childCredit : ℕ)

/-- Calculates the total tax liability for a given income and tax system -/
def calculateTaxLiability (income : ℕ) (children : ℕ) (system : TaxSystem) : ℚ :=
  sorry

/-- The current tax system -/
def currentSystem : TaxSystem :=
  { brackets := [
      ⟨0, 15000, 15/100⟩,
      ⟨15001, 45000, 42/100⟩,
      ⟨45001, 1000000, 50/100⟩  -- Using a large number for the upper bound of the highest bracket
    ],
    standardDeduction := 3000,
    childCredit := 1000
  }

/-- The proposed tax system -/
def proposedSystem : TaxSystem :=
  { brackets := [
      ⟨0, 15000, 12/100⟩,
      ⟨15001, 45000, 28/100⟩,
      ⟨45001, 1000000, 50/100⟩  -- Using a large number for the upper bound of the highest bracket
    ],
    standardDeduction := 3000,
    childCredit := 1000
  }

theorem tax_savings_proof (income : ℕ) (h : income = 34500) :
  calculateTaxLiability income 2 currentSystem - calculateTaxLiability income 2 proposedSystem = 2760 := by
  sorry

end NUMINAMATH_CALUDE_tax_savings_proof_l1739_173954


namespace NUMINAMATH_CALUDE_macaroon_weight_l1739_173986

theorem macaroon_weight
  (total_macaroons : ℕ)
  (num_bags : ℕ)
  (remaining_weight : ℚ)
  (h1 : total_macaroons = 12)
  (h2 : num_bags = 4)
  (h3 : remaining_weight = 45)
  (h4 : total_macaroons % num_bags = 0)  -- Ensures equal distribution
  : ∃ (weight_per_macaroon : ℚ),
    weight_per_macaroon * (total_macaroons - total_macaroons / num_bags) = remaining_weight ∧
    weight_per_macaroon = 5 := by
  sorry

end NUMINAMATH_CALUDE_macaroon_weight_l1739_173986


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_cube_l1739_173998

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_factor_for_perfect_cube (x : ℕ) (hx : x = 3 * 40 * 75) :
  ∃ y : ℕ, y > 0 ∧ is_perfect_cube (x * y) ∧ ∀ z : ℕ, z > 0 → is_perfect_cube (x * z) → y ≤ z :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_cube_l1739_173998


namespace NUMINAMATH_CALUDE_game_result_l1739_173995

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 ∧ n % 2 ≠ 0 then 8
  else if n % 2 = 0 ∧ n % 3 ≠ 0 then 3
  else 0

def chris_rolls : List ℕ := [5, 2, 1, 6]
def dana_rolls : List ℕ := [6, 2, 3, 3]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem game_result :
  let chris_points := total_points chris_rolls
  let dana_points := total_points dana_rolls
  dana_points = 27 ∧ chris_points * dana_points = 297 := by sorry

end NUMINAMATH_CALUDE_game_result_l1739_173995


namespace NUMINAMATH_CALUDE_second_player_wins_l1739_173947

/-- A game where two players alternate removing divisors from a pile of stones. -/
def StoneGame :=
  {initial : ℕ // initial = 2003}

/-- The set of valid moves for a given number of stones. -/
def validMoves (n : ℕ) : Set ℕ :=
  {m : ℕ | m ∣ n ∧ m ≠ 0}

/-- A strategy for playing the game. -/
def Strategy := ℕ → ℕ

/-- A strategy is valid if it always chooses a valid move. -/
def validStrategy (s : Strategy) : Prop :=
  ∀ n, s n ∈ validMoves n

/-- A strategy is winning if it always leads to a win for the player using it. -/
def winningStrategy (s : Strategy) : Prop :=
  validStrategy s ∧ 
  ∀ n, n > 1 → s n < n ∧ ¬∃ (m : ℕ), m ∈ validMoves (n - s n) ∧ m = n - s n

/-- The theorem stating that the second player has a winning strategy. -/
theorem second_player_wins (game : StoneGame) : 
  ∃ (s : Strategy), winningStrategy s :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l1739_173947


namespace NUMINAMATH_CALUDE_consecutive_sum_problem_l1739_173932

theorem consecutive_sum_problem 
  (p q r s t u v : ℕ+) 
  (h1 : p + q + r = 35)
  (h2 : q + r + s = 35)
  (h3 : r + s + t = 35)
  (h4 : s + t + u = 35)
  (h5 : t + u + v = 35)
  (h6 : q + u = 15) :
  p + q + r + s + t + u + v = 90 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_problem_l1739_173932


namespace NUMINAMATH_CALUDE_fencing_required_l1739_173962

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : area = 680 ∧ uncovered_side = 10 → 
  ∃ (width : ℝ), area = uncovered_side * width ∧ 2 * width + uncovered_side = 146 := by
  sorry

end NUMINAMATH_CALUDE_fencing_required_l1739_173962


namespace NUMINAMATH_CALUDE_lock_problem_l1739_173912

def num_buttons : ℕ := 10
def buttons_to_press : ℕ := 3
def time_per_attempt : ℕ := 2

def total_combinations : ℕ := (num_buttons.choose buttons_to_press)

theorem lock_problem :
  let total_time : ℕ := total_combinations * time_per_attempt
  let avg_attempts : ℚ := (1 + total_combinations : ℚ) / 2
  let avg_time : ℚ := avg_attempts * time_per_attempt
  let max_attempts_in_minute : ℕ := 60 / time_per_attempt
  (total_time = 240) ∧
  (avg_time = 121) ∧
  (max_attempts_in_minute : ℚ) / total_combinations = 29 / 120 := by
  sorry

end NUMINAMATH_CALUDE_lock_problem_l1739_173912


namespace NUMINAMATH_CALUDE_first_day_over_200_l1739_173978

def paperclips (n : ℕ) : ℕ := 5 * 3^(n - 1)

theorem first_day_over_200 :
  ∀ k : ℕ, k < 5 → paperclips k ≤ 200 ∧ paperclips 5 > 200 :=
by sorry

end NUMINAMATH_CALUDE_first_day_over_200_l1739_173978


namespace NUMINAMATH_CALUDE_unique_assignment_l1739_173930

/-- Represents the digits assigned to letters --/
structure Assignment where
  A : Fin 5
  M : Fin 5
  E : Fin 5
  H : Fin 5
  Z : Fin 5
  N : Fin 5

/-- Checks if all assigned digits are different --/
def Assignment.allDifferent (a : Assignment) : Prop :=
  a.A ≠ a.M ∧ a.A ≠ a.E ∧ a.A ≠ a.H ∧ a.A ≠ a.Z ∧ a.A ≠ a.N ∧
  a.M ≠ a.E ∧ a.M ≠ a.H ∧ a.M ≠ a.Z ∧ a.M ≠ a.N ∧
  a.E ≠ a.H ∧ a.E ≠ a.Z ∧ a.E ≠ a.N ∧
  a.H ≠ a.Z ∧ a.H ≠ a.N ∧
  a.Z ≠ a.N

/-- Checks if the assignment satisfies the given inequalities --/
def Assignment.satisfiesInequalities (a : Assignment) : Prop :=
  3 > a.A.val + 1 ∧ 
  a.A.val + 1 > a.M.val + 1 ∧ 
  a.M.val + 1 < a.E.val + 1 ∧ 
  a.E.val + 1 < a.H.val + 1 ∧ 
  a.H.val + 1 < a.A.val + 1

/-- Checks if the assignment results in the correct ZAMENA number --/
def Assignment.correctZAMENA (a : Assignment) : Prop :=
  a.Z.val + 1 = 5 ∧
  a.A.val + 1 = 4 ∧
  a.M.val + 1 = 1 ∧
  a.E.val + 1 = 2 ∧
  a.N.val + 1 = 4 ∧
  a.H.val + 1 = 3

theorem unique_assignment :
  ∀ a : Assignment,
    a.allDifferent ∧ a.satisfiesInequalities → a.correctZAMENA :=
by sorry

end NUMINAMATH_CALUDE_unique_assignment_l1739_173930


namespace NUMINAMATH_CALUDE_base_prime_rep_132_l1739_173906

def base_prime_representation (n : ℕ) : List ℕ :=
  sorry

theorem base_prime_rep_132 :
  base_prime_representation 132 = [2, 1, 0, 1] :=
by
  sorry

end NUMINAMATH_CALUDE_base_prime_rep_132_l1739_173906


namespace NUMINAMATH_CALUDE_quadratic_intersection_and_equivalence_l1739_173997

-- Define the quadratic function p
def p (x : ℝ) : Prop := x^2 - 7*x + 10 < 0

-- Define the function q
def q (x m : ℝ) : Prop := (x - m) * (x - 3*m) < 0

theorem quadratic_intersection_and_equivalence :
  (∃ (a b : ℝ), a = 4 ∧ b = 5 ∧ 
    (∀ x : ℝ, (p x ∧ q x 4) ↔ (a < x ∧ x < b))) ∧
  (∃ (c d : ℝ), c = 5/3 ∧ d = 2 ∧ 
    (∀ m : ℝ, m > 0 → 
      ((∀ x : ℝ, ¬(q x m) ↔ ¬(p x)) ↔ (c ≤ m ∧ m ≤ d)))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_and_equivalence_l1739_173997


namespace NUMINAMATH_CALUDE_tomato_shipment_ratio_l1739_173914

/-- Calculates the ratio of the second shipment to the first shipment of tomatoes -/
def shipment_ratio (initial_shipment : ℕ) (sold : ℕ) (rotted : ℕ) (final_amount : ℕ) : ℚ :=
  let remaining := initial_shipment - sold - rotted
  let second_shipment := final_amount - remaining
  (second_shipment : ℚ) / initial_shipment

/-- Proves that the ratio of the second shipment to the first shipment is 2:1 -/
theorem tomato_shipment_ratio :
  shipment_ratio 1000 300 200 2500 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tomato_shipment_ratio_l1739_173914


namespace NUMINAMATH_CALUDE_circle_trajectory_and_line_l1739_173979

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 2
def C₂ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 2

-- Define the trajectory of M
def M_trajectory (x y : ℝ) : Prop := x^2 / 2 - y^2 / 14 = 1 ∧ x ≥ Real.sqrt 2

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 14 * x - 27

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem circle_trajectory_and_line :
  ∃ (M : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ M → ∃ (r : ℝ), r > 0 ∧
      (∀ (x₁ y₁ : ℝ), C₁ x₁ y₁ → ((x - x₁)^2 + (y - y₁)^2 = (r + Real.sqrt 2)^2)) ∧
      (∀ (x₂ y₂ : ℝ), C₂ x₂ y₂ → ((x - x₂)^2 + (y - y₂)^2 = (r - Real.sqrt 2)^2))) ∧
    (∀ (x y : ℝ), (x, y) ∈ M ↔ M_trajectory x y) ∧
    (∃ (A B : ℝ × ℝ), A ∈ M ∧ B ∈ M ∧ A ≠ B ∧
      ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = P ∧
      (∀ (x y : ℝ), line_l x y ↔ (y - A.2) / (x - A.1) = (B.2 - A.2) / (B.1 - A.1) ∧ x ≠ A.1)) :=
by sorry

end NUMINAMATH_CALUDE_circle_trajectory_and_line_l1739_173979


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1739_173938

theorem absolute_value_inequality (a : ℝ) :
  (∃ x : ℝ, |x + 1| + |x - 3| ≤ a) → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1739_173938


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1739_173926

theorem inequality_solution_set : 
  {x : ℝ | |x|^3 - 2*x^2 - 4*|x| + 3 < 0} = 
  {x : ℝ | -3 < x ∧ x < -1} ∪ {x : ℝ | 1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1739_173926


namespace NUMINAMATH_CALUDE_quarters_put_aside_l1739_173902

theorem quarters_put_aside (original_quarters : ℕ) (remaining_quarters : ℕ) : 
  (5 * original_quarters = 350) →
  (remaining_quarters + 350 = 392) →
  (original_quarters - remaining_quarters : ℚ) / original_quarters = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_quarters_put_aside_l1739_173902


namespace NUMINAMATH_CALUDE_rectangles_in_4x5_grid_l1739_173970

/-- The number of rectangles in a grid with sides along the grid lines -/
def count_rectangles (m n : ℕ) : ℕ :=
  (m * (m + 1) * n * (n + 1)) / 4

/-- Theorem: In a 4 × 5 grid, the total number of rectangles with sides along the grid lines is 24 -/
theorem rectangles_in_4x5_grid :
  count_rectangles 4 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_4x5_grid_l1739_173970


namespace NUMINAMATH_CALUDE_parabola_equation_l1739_173955

/-- A parabola with axis of symmetry parallel to the y-axis -/
structure Parabola where
  a : ℝ
  eq : ℝ → ℝ
  eq_def : ∀ x, eq x = a * (x - 1) * (x - 4)

/-- The line y = 2x -/
def line (x : ℝ) : ℝ := 2 * x

theorem parabola_equation (p : Parabola) (h1 : p.eq 1 = 0) (h2 : p.eq 4 = 0)
  (h_tangent : ∃ x, p.eq x = line x ∧ ∀ y ≠ x, p.eq y ≠ line y) :
  p.a = -2/9 ∨ p.a = -2 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1739_173955


namespace NUMINAMATH_CALUDE_snow_probability_l1739_173929

theorem snow_probability (p : ℝ) (h : p = 3 / 4) :
  1 - (1 - p)^4 = 255 / 256 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l1739_173929


namespace NUMINAMATH_CALUDE_trail_mix_weight_l1739_173904

/-- The weight of peanuts in pounds -/
def weight_peanuts : ℝ := 0.17

/-- The weight of chocolate chips in pounds -/
def weight_chocolate_chips : ℝ := 0.17

/-- The weight of raisins in pounds -/
def weight_raisins : ℝ := 0.08

/-- The total weight of trail mix in pounds -/
def total_weight : ℝ := weight_peanuts + weight_chocolate_chips + weight_raisins

theorem trail_mix_weight : total_weight = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_weight_l1739_173904


namespace NUMINAMATH_CALUDE_min_toothpicks_for_five_squares_l1739_173974

/-- A square formed by toothpicks -/
structure ToothpickSquare where
  side_length : ℝ
  toothpicks_per_square : ℕ

/-- The arrangement of multiple toothpick squares -/
structure SquareArrangement where
  square : ToothpickSquare
  num_squares : ℕ

/-- The number of toothpicks needed for an arrangement of squares -/
def toothpicks_needed (arrangement : SquareArrangement) : ℕ :=
  sorry

/-- The theorem stating the minimum number of toothpicks needed -/
theorem min_toothpicks_for_five_squares
  (square : ToothpickSquare)
  (arrangement : SquareArrangement)
  (h1 : square.side_length = 6)
  (h2 : square.toothpicks_per_square = 4)
  (h3 : arrangement.square = square)
  (h4 : arrangement.num_squares = 5) :
  toothpicks_needed arrangement = 15 :=
sorry

end NUMINAMATH_CALUDE_min_toothpicks_for_five_squares_l1739_173974


namespace NUMINAMATH_CALUDE_abs_neg_eight_eq_eight_l1739_173961

theorem abs_neg_eight_eq_eight :
  abs (-8 : ℤ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_eight_eq_eight_l1739_173961


namespace NUMINAMATH_CALUDE_mountain_elevation_l1739_173911

/-- The relative elevation of a mountain given temperature information -/
theorem mountain_elevation (temp_decrease_rate : ℝ) (temp_summit temp_foot : ℝ) 
  (h1 : temp_decrease_rate = 0.7)
  (h2 : temp_summit = 14.1)
  (h3 : temp_foot = 26) :
  (temp_foot - temp_summit) / temp_decrease_rate * 100 = 1700 := by
  sorry

end NUMINAMATH_CALUDE_mountain_elevation_l1739_173911


namespace NUMINAMATH_CALUDE_apple_count_l1739_173934

theorem apple_count (blue_apples : ℕ) (yellow_apples : ℕ) : 
  blue_apples = 5 →
  yellow_apples = 2 * blue_apples →
  (blue_apples + yellow_apples) - ((blue_apples + yellow_apples) / 5) = 12 := by
sorry

end NUMINAMATH_CALUDE_apple_count_l1739_173934


namespace NUMINAMATH_CALUDE_range_of_expression_l1739_173981

open Real

theorem range_of_expression (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 ≤ β ∧ β ≤ π/2) : 
  -π/6 < 2*α - β/3 ∧ 2*α - β/3 < π := by
sorry

end NUMINAMATH_CALUDE_range_of_expression_l1739_173981


namespace NUMINAMATH_CALUDE_trillion_scientific_notation_l1739_173921

theorem trillion_scientific_notation :
  (10000 : ℝ) * 10000 * 10000 = 1 * (10 : ℝ)^12 := by sorry

end NUMINAMATH_CALUDE_trillion_scientific_notation_l1739_173921


namespace NUMINAMATH_CALUDE_train_passing_platform_l1739_173967

/-- Calculates the time for a train to pass a platform -/
theorem train_passing_platform (train_length : ℝ) (tree_passing_time : ℝ) (platform_length : ℝ) :
  train_length = 1200 →
  tree_passing_time = 120 →
  platform_length = 1100 →
  (train_length + platform_length) / (train_length / tree_passing_time) = 230 := by
sorry

end NUMINAMATH_CALUDE_train_passing_platform_l1739_173967


namespace NUMINAMATH_CALUDE_min_value_expression_l1739_173953

theorem min_value_expression (a b : ℝ) (ha : a > 1) (hb : b > 2) :
  (a + b)^2 / (Real.sqrt (a^2 - 1) + Real.sqrt (b^2 - 4)) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1739_173953


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l1739_173936

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryNumber := List Bool

/-- Converts a binary number to its decimal representation -/
def binaryToDecimal (b : BinaryNumber) : ℕ :=
  b.foldl (fun acc bit => 2 * acc + if bit then 1 else 0) 0

/-- The binary number 10110₂ -/
def b1 : BinaryNumber := [true, false, true, true, false]

/-- The binary number 1101₂ -/
def b2 : BinaryNumber := [true, true, false, true]

/-- The binary number 1010₂ -/
def b3 : BinaryNumber := [true, false, true, false]

/-- The binary number 1110₂ -/
def b4 : BinaryNumber := [true, true, true, false]

/-- The binary number 11111₂ (the expected result) -/
def result : BinaryNumber := [true, true, true, true, true]

theorem binary_addition_subtraction :
  binaryToDecimal b1 + binaryToDecimal b2 - binaryToDecimal b3 + binaryToDecimal b4 =
  binaryToDecimal result := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l1739_173936


namespace NUMINAMATH_CALUDE_five_houses_built_l1739_173915

/-- The number of houses that can be built given the specified conditions -/
def number_of_houses (builders_per_floor : ℕ) (days_per_floor : ℕ) (daily_wage : ℕ) 
  (total_cost : ℕ) (builders : ℕ) (floors_per_house : ℕ) : ℕ :=
  let daily_cost := builders * daily_wage
  let total_days := total_cost / daily_cost
  let days_per_floor_with_builders := days_per_floor * builders_per_floor / builders
  let total_floors := total_days / days_per_floor_with_builders
  total_floors / floors_per_house

/-- Theorem stating that 5 houses can be built under the given conditions -/
theorem five_houses_built :
  number_of_houses 3 30 100 270000 6 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_houses_built_l1739_173915


namespace NUMINAMATH_CALUDE_functional_equation_2013_l1739_173919

/-- Given a function f: ℝ → ℝ satisfying f(x-y) = f(x) + f(y) - 2xy for all real x and y,
    prove that f(2013) = 4052169 -/
theorem functional_equation_2013 (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x - y) = f x + f y - 2 * x * y) : 
    f 2013 = 4052169 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_2013_l1739_173919


namespace NUMINAMATH_CALUDE_derivative_at_negative_third_l1739_173968

theorem derivative_at_negative_third (f : ℝ → ℝ) 
  (h : ∀ x, f x = x^2 + 2 * (deriv f (-1/3)) * x) : 
  deriv f (-1/3) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_negative_third_l1739_173968


namespace NUMINAMATH_CALUDE_legs_in_room_is_40_l1739_173920

/-- Calculates the total number of legs in a room with various furniture items. -/
def total_legs_in_room : ℕ :=
  let four_legged_items := 4 + 1 + 2  -- 4 tables, 1 sofa, 2 chairs
  let three_legged_tables := 3
  let one_legged_table := 1
  let two_legged_rocking_chair := 1
  
  4 * four_legged_items + 
  3 * three_legged_tables + 
  1 * one_legged_table + 
  2 * two_legged_rocking_chair

/-- Theorem stating that the total number of legs in the room is 40. -/
theorem legs_in_room_is_40 : total_legs_in_room = 40 := by
  sorry

end NUMINAMATH_CALUDE_legs_in_room_is_40_l1739_173920


namespace NUMINAMATH_CALUDE_parallelepiped_volume_solution_l1739_173966

/-- The volume of a parallelepiped defined by vectors (3,4,5), (2,k,3), and (2,3,k) -/
def parallelepipedVolume (k : ℝ) : ℝ := |3 * k^2 - 15 * k + 27|

/-- Theorem stating that k = 5 is the positive solution for the parallelepiped volume equation -/
theorem parallelepiped_volume_solution :
  ∃! k : ℝ, k > 0 ∧ parallelepipedVolume k = 27 ∧ k = 5 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_solution_l1739_173966


namespace NUMINAMATH_CALUDE_totalDays_is_25_l1739_173910

/-- Calculates the total number of days in a work period given the following conditions:
  * A woman is paid $20 for each day she works
  * She forfeits $5 for each day she is idle
  * She nets $450
  * She worked for 23 days
-/
def totalDaysInPeriod (dailyPay : ℕ) (dailyForfeit : ℕ) (netEarnings : ℕ) (daysWorked : ℕ) : ℕ :=
  sorry

/-- Proves that the total number of days in the period is 25 -/
theorem totalDays_is_25 :
  totalDaysInPeriod 20 5 450 23 = 25 := by
  sorry

end NUMINAMATH_CALUDE_totalDays_is_25_l1739_173910


namespace NUMINAMATH_CALUDE_romance_movie_tickets_l1739_173948

theorem romance_movie_tickets (horror_tickets : ℕ) (romance_tickets : ℕ) : 
  horror_tickets = 3 * romance_tickets + 18 →
  horror_tickets = 93 →
  romance_tickets = 25 := by
sorry

end NUMINAMATH_CALUDE_romance_movie_tickets_l1739_173948


namespace NUMINAMATH_CALUDE_parallel_segments_y_coordinate_l1739_173918

/-- Given four points A, B, X, and Y in a 2D plane, where segment AB is parallel to segment XY,
    prove that the y-coordinate of Y is -1. -/
theorem parallel_segments_y_coordinate
  (A B X Y : ℝ × ℝ)
  (hA : A = (-2, -2))
  (hB : B = (2, -6))
  (hX : X = (1, 5))
  (hY : Y = (7, Y.2))
  (h_parallel : (B.1 - A.1) * (Y.2 - X.2) = (Y.1 - X.1) * (B.2 - A.2)) :
  Y.2 = -1 :=
sorry

end NUMINAMATH_CALUDE_parallel_segments_y_coordinate_l1739_173918


namespace NUMINAMATH_CALUDE_electric_guitars_sold_l1739_173994

theorem electric_guitars_sold (total_guitars : ℕ) (total_revenue : ℕ) 
  (electric_price : ℕ) (acoustic_price : ℕ) 
  (h1 : total_guitars = 9)
  (h2 : total_revenue = 3611)
  (h3 : electric_price = 479)
  (h4 : acoustic_price = 339) :
  ∃ (x : ℕ), x = 4 ∧ 
    ∃ (y : ℕ), x + y = total_guitars ∧ 
    electric_price * x + acoustic_price * y = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_electric_guitars_sold_l1739_173994


namespace NUMINAMATH_CALUDE_apple_price_per_kg_final_apple_price_is_correct_l1739_173928

/-- Calculates the final price per kilogram of apples after discounts -/
theorem apple_price_per_kg (weight : ℝ) (original_price : ℝ) 
  (discount_percent : ℝ) (volume_discount_percent : ℝ) 
  (volume_discount_threshold : ℝ) : ℝ :=
  let price_after_discount := original_price * (1 - discount_percent)
  let final_price := 
    if weight > volume_discount_threshold
    then price_after_discount * (1 - volume_discount_percent)
    else price_after_discount
  final_price / weight

/-- Proves that the final price per kilogram is $1.44 given the specific conditions -/
theorem final_apple_price_is_correct : 
  apple_price_per_kg 5 10 0.2 0.1 3 = 1.44 := by
  sorry

end NUMINAMATH_CALUDE_apple_price_per_kg_final_apple_price_is_correct_l1739_173928


namespace NUMINAMATH_CALUDE_first_divisor_problem_l1739_173933

theorem first_divisor_problem (x : ℕ) : x = 7 ↔ 
  x > 0 ∧ 
  x ≠ 15 ∧ 
  184 % x = 2 ∧ 
  184 % 15 = 4 ∧ 
  ∀ y : ℕ, y > 0 ∧ y < x ∧ y ≠ 15 → 184 % y ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_first_divisor_problem_l1739_173933


namespace NUMINAMATH_CALUDE_range_of_m_for_inverse_proposition_l1739_173976

theorem range_of_m_for_inverse_proposition : 
  ∀ m : ℝ, 
  (∀ x : ℝ, (1 < x ∧ x < 3) → (m < x ∧ x < m + 3)) → 
  (0 ≤ m ∧ m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_inverse_proposition_l1739_173976


namespace NUMINAMATH_CALUDE_christmas_tree_decoration_l1739_173991

theorem christmas_tree_decoration (b t : ℕ) : 
  (t = b + 1) →  -- Chuck's condition
  (2 * b = t - 1) →  -- Huck's condition
  (b = 3 ∧ t = 4) :=  -- Conclusion
by sorry

end NUMINAMATH_CALUDE_christmas_tree_decoration_l1739_173991


namespace NUMINAMATH_CALUDE_select_five_from_eight_l1739_173982

theorem select_five_from_eight : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_eight_l1739_173982


namespace NUMINAMATH_CALUDE_cards_per_deck_l1739_173922

theorem cards_per_deck 
  (num_decks : ℕ) 
  (num_layers : ℕ) 
  (cards_per_layer : ℕ) 
  (h1 : num_decks = 16) 
  (h2 : num_layers = 32) 
  (h3 : cards_per_layer = 26) : 
  (num_layers * cards_per_layer) / num_decks = 52 := by
  sorry

end NUMINAMATH_CALUDE_cards_per_deck_l1739_173922


namespace NUMINAMATH_CALUDE_condition_relationship_l1739_173977

theorem condition_relationship (a b : ℝ) : 
  (((a > 2 ∧ b > 2) → (a + b > 4)) ∧ 
   (∃ (x y : ℝ), x + y > 4 ∧ (x ≤ 2 ∨ y ≤ 2))) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l1739_173977


namespace NUMINAMATH_CALUDE_min_a_squared_plus_b_squared_l1739_173989

theorem min_a_squared_plus_b_squared : ∀ a b : ℝ,
  (∀ x : ℝ, x^2 + a*x + b - 3 = 0 → x = 2) →
  a^2 + b^2 ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_a_squared_plus_b_squared_l1739_173989


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1739_173940

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  -- The sum function for the first n terms
  S : ℕ → ℝ
  -- Property: The sequence of differences forms an arithmetic sequence
  difference_is_arithmetic : ∀ (k : ℕ), S (k + 1) - S k = S (k + 2) - S (k + 1)

/-- Theorem: For an arithmetic sequence with S_n = 30 and S_{2n} = 100, S_{3n} = 170 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) (n : ℕ) 
    (h1 : a.S n = 30) (h2 : a.S (2 * n) = 100) : 
    a.S (3 * n) = 170 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1739_173940


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l1739_173963

theorem polynomial_division_degree (f d q r : Polynomial ℝ) : 
  Polynomial.degree f = 15 →
  Polynomial.degree q = 9 →
  r = 5 * X^4 + 6 * X^3 - 2 * X + 7 →
  f = d * q + r →
  Polynomial.degree d = 6 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l1739_173963


namespace NUMINAMATH_CALUDE_exists_monomial_neg5_deg2_l1739_173952

/-- A monomial is a product of a coefficient and variables raised to non-negative integer powers. -/
structure Monomial where
  coefficient : ℤ
  degree : ℕ

/-- Checks if a given monomial has the specified coefficient and degree. -/
def has_coeff_and_degree (m : Monomial) (c : ℤ) (d : ℕ) : Prop :=
  m.coefficient = c ∧ m.degree = d

/-- There exists a monomial with coefficient -5 and degree 2. -/
theorem exists_monomial_neg5_deg2 : ∃ m : Monomial, has_coeff_and_degree m (-5) 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_monomial_neg5_deg2_l1739_173952


namespace NUMINAMATH_CALUDE_eggs_cooked_per_year_l1739_173941

/-- The number of eggs Lisa cooks for her family for breakfast in a year -/
def eggs_per_year : ℕ :=
  let days_per_week : ℕ := 5
  let weeks_per_year : ℕ := 52
  let num_children : ℕ := 4
  let eggs_per_child : ℕ := 2
  let eggs_for_husband : ℕ := 3
  let eggs_for_self : ℕ := 2
  let eggs_per_day : ℕ := num_children * eggs_per_child + eggs_for_husband + eggs_for_self
  eggs_per_day * days_per_week * weeks_per_year

theorem eggs_cooked_per_year :
  eggs_per_year = 3380 := by
  sorry

end NUMINAMATH_CALUDE_eggs_cooked_per_year_l1739_173941


namespace NUMINAMATH_CALUDE_problem_solution_l1739_173971

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) : (x^4 * y^2) / 8 = 162 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1739_173971


namespace NUMINAMATH_CALUDE_max_delta_success_ratio_l1739_173900

/-- Represents a participant's score in a competition day -/
structure DayScore where
  scored : ℕ
  attempted : ℕ

/-- Represents a participant's scores over three days -/
structure ThreeDayScore where
  day1 : DayScore
  day2 : DayScore
  day3 : DayScore

def success_ratio (score : DayScore) : ℚ :=
  score.scored / score.attempted

def three_day_success_ratio (score : ThreeDayScore) : ℚ :=
  (score.day1.scored + score.day2.scored + score.day3.scored) /
  (score.day1.attempted + score.day2.attempted + score.day3.attempted)

theorem max_delta_success_ratio
  (gamma : ThreeDayScore)
  (delta : ThreeDayScore)
  (h1 : gamma.day1 = ⟨180, 300⟩)
  (h2 : gamma.day2 = ⟨120, 200⟩)
  (h3 : gamma.day3 = ⟨0, 0⟩)
  (h4 : delta.day1.scored > 0 ∧ delta.day2.scored > 0 ∧ delta.day3.scored > 0)
  (h5 : success_ratio delta.day1 < success_ratio gamma.day1)
  (h6 : success_ratio delta.day2 < success_ratio gamma.day2)
  (h7 : success_ratio delta.day3 < 1)
  (h8 : delta.day1.attempted + delta.day2.attempted + delta.day3.attempted = 600)
  (h9 : three_day_success_ratio gamma = 3/5) :
  three_day_success_ratio delta ≤ 539/600 :=
sorry

end NUMINAMATH_CALUDE_max_delta_success_ratio_l1739_173900


namespace NUMINAMATH_CALUDE_find_x_l1739_173992

theorem find_x : ∃ x : ℕ, (2^x : ℝ) - (2^(x-2) : ℝ) = 3 * (2^12 : ℝ) ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1739_173992


namespace NUMINAMATH_CALUDE_chord_equation_of_ellipse_l1739_173988

/-- The equation of a line that forms a chord of the ellipse x^2/2 + y^2 = 1,
    bisected by the point (1/2, 1/2) -/
theorem chord_equation_of_ellipse (x y : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 / 2 + y₁^2 = 1) ∧
    (x₂^2 / 2 + y₂^2 = 1) ∧
    ((x₁ + x₂) / 2 = 1/2) ∧
    ((y₁ + y₂) / 2 = 1/2) ∧
    (y - y₁) = ((y₂ - y₁) / (x₂ - x₁)) * (x - x₁)) →
  2*x + 4*y - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_of_ellipse_l1739_173988


namespace NUMINAMATH_CALUDE_divisibility_rule_37_l1739_173951

/-- Represents a natural number as a list of its digits -/
def toDigits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

/-- Calculates the sum for the divisibility rule of 37 -/
def divSum (digits : List ℕ) : ℤ :=
  let rec aux (l : List ℕ) (k : ℕ) : ℤ :=
    match l with
    | [] => 0
    | [a] => a
    | [a, b] => a + 10 * b
    | a :: b :: c :: rest => a + 10 * b - 11 * c + aux rest (k + 1)
  aux digits 0

/-- The divisibility rule for 37 -/
theorem divisibility_rule_37 (n : ℕ) :
  n % 37 = 0 ↔ (divSum (toDigits n)) % 37 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_rule_37_l1739_173951


namespace NUMINAMATH_CALUDE_equation_solution_l1739_173901

/-- Given the equation and values for a, b, c, and d, prove that x equals 26544.74 -/
theorem equation_solution :
  let a : ℝ := 3
  let b : ℝ := 5
  let c : ℝ := 2
  let d : ℝ := 4
  let x : ℝ := ((a^2 * b * (47 / 100 * 1442)) - (c * d * (36 / 100 * 1412))) + 63
  x = 26544.74 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1739_173901


namespace NUMINAMATH_CALUDE_total_capacity_is_57600_l1739_173939

/-- The total capacity of James' fleet of vans -/
def total_capacity : ℕ := by
  -- Define the number of vans
  let total_vans : ℕ := 6
  let large_vans : ℕ := 2
  let medium_van : ℕ := 1
  let extra_large_vans : ℕ := 3

  -- Define the capacities
  let base_capacity : ℕ := 8000
  let medium_capacity : ℕ := base_capacity - (base_capacity * 30 / 100)
  let extra_large_capacity : ℕ := base_capacity + (base_capacity * 50 / 100)

  -- Calculate total capacity
  exact large_vans * base_capacity + 
        medium_van * medium_capacity + 
        extra_large_vans * extra_large_capacity

/-- Theorem stating that the total capacity is 57600 gallons -/
theorem total_capacity_is_57600 : total_capacity = 57600 := by
  sorry

end NUMINAMATH_CALUDE_total_capacity_is_57600_l1739_173939


namespace NUMINAMATH_CALUDE_second_train_length_l1739_173937

/-- The length of a train given crossing time and speeds -/
def train_length (l1 : ℝ) (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v1 + v2) * t - l1

/-- Theorem: Given the conditions, the length of the second train is 210 meters -/
theorem second_train_length :
  let l1 : ℝ := 290  -- Length of first train in meters
  let v1 : ℝ := 120 * 1000 / 3600  -- Speed of first train in m/s
  let v2 : ℝ := 80 * 1000 / 3600   -- Speed of second train in m/s
  let t : ℝ := 9    -- Crossing time in seconds
  train_length l1 v1 v2 t = 210 := by
sorry


end NUMINAMATH_CALUDE_second_train_length_l1739_173937


namespace NUMINAMATH_CALUDE_line_not_parallel_intersects_plane_l1739_173927

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Definition: A line is parallel to a plane -/
def is_parallel (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Definition: A line shares common points with a plane -/
def shares_common_points (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line is not parallel to a plane, then it shares common points with the plane -/
theorem line_not_parallel_intersects_plane (l : Line3D) (α : Plane3D) :
  ¬(is_parallel l α) → shares_common_points l α :=
by
  sorry

end NUMINAMATH_CALUDE_line_not_parallel_intersects_plane_l1739_173927


namespace NUMINAMATH_CALUDE_question_mark_solution_l1739_173999

theorem question_mark_solution : ∃! x : ℤ, x + 3699 + 1985 - 2047 = 31111 :=
  sorry

end NUMINAMATH_CALUDE_question_mark_solution_l1739_173999


namespace NUMINAMATH_CALUDE_martian_calendar_reform_l1739_173965

theorem martian_calendar_reform (old_short old_long new_short new_long : ℕ) 
  (h1 : old_short = 26)
  (h2 : old_long = 29)
  (h3 : new_short = 27)
  (h4 : new_long = 31)
  (h5 : ∃ (x y : ℕ), old_short * x + old_long * y = 687)
  (h6 : ∃ (p q : ℕ), new_short * p + new_long * q = 687) :
  ∃ (x y p q : ℕ), 
    old_short * x + old_long * y = 687 ∧ 
    new_short * p + new_long * q = 687 ∧ 
    p + q ≥ x + y :=
by sorry

end NUMINAMATH_CALUDE_martian_calendar_reform_l1739_173965


namespace NUMINAMATH_CALUDE_greatest_power_of_two_l1739_173905

theorem greatest_power_of_two (n : ℕ) : 
  (∃ k : ℕ, (10^1004 - 4^502) = k * 2^1007) ∧ 
  (∀ m : ℕ, m > 1007 → ¬(∃ k : ℕ, (10^1004 - 4^502) = k * 2^m)) :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_l1739_173905


namespace NUMINAMATH_CALUDE_segment_length_bound_polygon_perimeter_bound_l1739_173949

-- Define a segment in 2D plane
structure Segment where
  length : ℝ
  projection1 : ℝ
  projection2 : ℝ

-- Define a polygon in 2D plane
structure Polygon where
  perimeter : ℝ
  totalProjection1 : ℝ
  totalProjection2 : ℝ

-- Theorem for segment
theorem segment_length_bound (s : Segment) : 
  s.length ≥ (s.projection1 + s.projection2) / Real.sqrt 2 := by sorry

-- Theorem for polygon
theorem polygon_perimeter_bound (p : Polygon) :
  p.perimeter ≥ Real.sqrt 2 * (p.totalProjection1 + p.totalProjection2) := by sorry

end NUMINAMATH_CALUDE_segment_length_bound_polygon_perimeter_bound_l1739_173949


namespace NUMINAMATH_CALUDE_mod_19_equivalence_l1739_173917

theorem mod_19_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 19 ∧ 42568 % 19 = n % 19 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_mod_19_equivalence_l1739_173917


namespace NUMINAMATH_CALUDE_unique_number_divisible_by_792_l1739_173984

theorem unique_number_divisible_by_792 :
  ∀ (x y z : ℕ), x < 10 → y < 10 → z < 10 →
  (13 * 100000 + x * 10000 + y * 1000 + 45 * 10 + z) % 792 = 0 →
  (13 * 100000 + x * 10000 + y * 1000 + 45 * 10 + z) = 1380456 := by
sorry

end NUMINAMATH_CALUDE_unique_number_divisible_by_792_l1739_173984


namespace NUMINAMATH_CALUDE_min_square_value_l1739_173980

theorem min_square_value (a b : ℕ+) 
  (h1 : ∃ m : ℕ+, (15 * a + 16 * b : ℕ) = m ^ 2)
  (h2 : ∃ n : ℕ+, (16 * a - 15 * b : ℕ) = n ^ 2) :
  min (15 * a + 16 * b) (16 * a - 15 * b) ≥ 481 ^ 2 := by
sorry

end NUMINAMATH_CALUDE_min_square_value_l1739_173980


namespace NUMINAMATH_CALUDE_parabola_properties_l1739_173985

-- Define the parabola
def parabola (m n x : ℝ) : ℝ := m * x^2 - 2 * m^2 * x + n

-- Define the conditions and theorem
theorem parabola_properties
  (m n x₁ x₂ y₁ y₂ : ℝ)
  (h_m : m ≠ 0)
  (h_parabola₁ : parabola m n x₁ = y₁)
  (h_parabola₂ : parabola m n x₂ = y₂) :
  (x₁ = 1 ∧ x₂ = 3 ∧ y₁ = y₂ → 2 = (x₁ + x₂) / 2) ∧
  (x₁ + x₂ > 4 ∧ x₁ < x₂ ∧ y₁ < y₂ → 0 < m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1739_173985


namespace NUMINAMATH_CALUDE_arrangement_sum_l1739_173950

theorem arrangement_sum (n : ℕ+) 
  (h1 : n + 3 ≤ 2 * n) 
  (h2 : n + 1 ≤ 4) : 
  Nat.descFactorial (2 * n) (n + 3) + Nat.descFactorial 4 (n + 1) = 744 :=
sorry

end NUMINAMATH_CALUDE_arrangement_sum_l1739_173950


namespace NUMINAMATH_CALUDE_product_of_four_consecutive_integers_divisible_by_ten_l1739_173993

theorem product_of_four_consecutive_integers_divisible_by_ten (n : ℕ) (h : n % 2 = 1) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_consecutive_integers_divisible_by_ten_l1739_173993


namespace NUMINAMATH_CALUDE_determinant_of_special_matrix_l1739_173924

open Matrix Real

theorem determinant_of_special_matrix (α β : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![0, cos α, sin α;
                                       sin α, 0, cos β;
                                       -cos α, -sin β, 0]
  det M = cos (β - 2*α) := by
sorry

end NUMINAMATH_CALUDE_determinant_of_special_matrix_l1739_173924


namespace NUMINAMATH_CALUDE_kara_water_consumption_l1739_173973

/-- Amount of water Kara drinks with each medication dose -/
def water_per_dose (total_water : ℕ) (doses_per_day : ℕ) (total_days : ℕ) (missed_doses : ℕ) : ℚ :=
  total_water / (doses_per_day * total_days - missed_doses)

/-- Theorem stating that Kara drinks 4 ounces of water per medication dose -/
theorem kara_water_consumption :
  water_per_dose 160 3 14 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_kara_water_consumption_l1739_173973


namespace NUMINAMATH_CALUDE_nine_in_M_ten_not_in_M_l1739_173909

/-- The set M of integers that can be expressed as the difference of two squares of integers -/
def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

/-- 9 belongs to the set M -/
theorem nine_in_M : (9 : ℤ) ∈ M := by sorry

/-- 10 does not belong to the set M -/
theorem ten_not_in_M : (10 : ℤ) ∉ M := by sorry

end NUMINAMATH_CALUDE_nine_in_M_ten_not_in_M_l1739_173909


namespace NUMINAMATH_CALUDE_simplify_sqrt_one_minus_sin_twenty_deg_l1739_173943

theorem simplify_sqrt_one_minus_sin_twenty_deg :
  Real.sqrt (1 - Real.sin (20 * π / 180)) = Real.cos (10 * π / 180) - Real.sin (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_one_minus_sin_twenty_deg_l1739_173943


namespace NUMINAMATH_CALUDE_line_passes_through_point_two_two_l1739_173903

/-- The line equation is of the form (1+4k)x-(2-3k)y+2-14k=0 where k is a real parameter -/
def line_equation (k x y : ℝ) : Prop :=
  (1 + 4*k)*x - (2 - 3*k)*y + 2 - 14*k = 0

/-- Theorem: The line passes through the point (2, 2) for all values of k -/
theorem line_passes_through_point_two_two :
  ∀ k : ℝ, line_equation k 2 2 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_two_two_l1739_173903


namespace NUMINAMATH_CALUDE_pond_problem_l1739_173975

theorem pond_problem (initial_fish : ℕ) (fish_caught : ℕ) : 
  initial_fish = 50 →
  fish_caught = 7 →
  (initial_fish * 3 / 2) - (initial_fish - fish_caught) = 32 := by
  sorry

end NUMINAMATH_CALUDE_pond_problem_l1739_173975


namespace NUMINAMATH_CALUDE_triangle_perimeter_not_48_l1739_173972

theorem triangle_perimeter_not_48 (a b c : ℝ) : 
  a = 25 → b = 12 → a + b + c > a + b → a + c > b → b + c > a → a + b + c ≠ 48 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_not_48_l1739_173972


namespace NUMINAMATH_CALUDE_correct_operation_l1739_173990

theorem correct_operation (a : ℝ) : 4 * a - a = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1739_173990


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l1739_173996

/-- Given a triangle with side lengths 8, 10, and 12, the surface area of the circumscribed sphere
    of the triangular prism formed by connecting the midpoints of the triangle's sides
    is equal to 77π/2. -/
theorem circumscribed_sphere_surface_area (A₁ A₂ A₃ B C D : ℝ × ℝ × ℝ) : 
  let side_lengths := [8, 10, 12]
  ∀ (a b c : ℝ), 
    a ∈ side_lengths ∧ 
    b ∈ side_lengths ∧ 
    c ∈ side_lengths ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    (‖A₁ - A₂‖ = a ∧ ‖A₂ - A₃‖ = b ∧ ‖A₃ - A₁‖ = c) →
    (B = (A₁ + A₂) / 2 ∧ C = (A₂ + A₃) / 2 ∧ D = (A₃ + A₁) / 2) →
    let R := Real.sqrt (77 / 8)
    4 * π * R^2 = 77 * π / 2 :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l1739_173996


namespace NUMINAMATH_CALUDE_gcd_lcm_product_90_150_l1739_173935

theorem gcd_lcm_product_90_150 : Nat.gcd 90 150 * Nat.lcm 90 150 = 13500 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_90_150_l1739_173935


namespace NUMINAMATH_CALUDE_equation_solution_l1739_173956

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- Checks if the equation 3△_4 = △2_11 is satisfied -/
def equation_satisfied (triangle : Nat) : Prop :=
  to_base_10 [3, triangle] 4 = to_base_10 [triangle, 2] 11

/-- Theorem stating that the equation is satisfied when triangle is 1 -/
theorem equation_solution :
  equation_satisfied 1 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1739_173956


namespace NUMINAMATH_CALUDE_work_completion_proof_l1739_173969

/-- The number of days it takes x to complete the work -/
def x_total_days : ℝ := 40

/-- The number of days it takes y to complete the work -/
def y_total_days : ℝ := 35

/-- The number of days y worked to finish the work after x stopped -/
def y_actual_days : ℝ := 28

/-- The number of days x worked before y took over -/
def x_worked_days : ℝ := 8

theorem work_completion_proof :
  x_worked_days / x_total_days + y_actual_days / y_total_days = 1 := by
  sorry


end NUMINAMATH_CALUDE_work_completion_proof_l1739_173969


namespace NUMINAMATH_CALUDE_alloy_cut_theorem_l1739_173946

/-- Represents an alloy piece with its mass and copper concentration -/
structure AlloyPiece where
  mass : ℝ
  copper_concentration : ℝ

/-- Represents the result of cutting and swapping parts of two alloy pieces -/
def cut_and_swap (piece1 piece2 : AlloyPiece) (cut_mass : ℝ) : Prop :=
  let new_piece1 := AlloyPiece.mk piece1.mass 
    ((cut_mass * piece2.copper_concentration + (piece1.mass - cut_mass) * piece1.copper_concentration) / piece1.mass)
  let new_piece2 := AlloyPiece.mk piece2.mass 
    ((cut_mass * piece1.copper_concentration + (piece2.mass - cut_mass) * piece2.copper_concentration) / piece2.mass)
  new_piece1.copper_concentration = new_piece2.copper_concentration

theorem alloy_cut_theorem (piece1 piece2 : AlloyPiece) (cut_mass : ℝ) :
  piece1.mass = piece2.mass →
  piece1.copper_concentration ≠ piece2.copper_concentration →
  cut_and_swap piece1 piece2 cut_mass →
  cut_mass = piece1.mass / 2 :=
sorry

end NUMINAMATH_CALUDE_alloy_cut_theorem_l1739_173946


namespace NUMINAMATH_CALUDE_prob_three_odd_in_eight_rolls_l1739_173959

/-- The probability of getting an odd number on a single roll of a fair six-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 8

/-- The number of odd results we're interested in -/
def target_odd : ℕ := 3

/-- Binomial coefficient -/
def binom (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of exactly k successes in n trials with probability p of success on each trial -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binom n k : ℚ) * p^k * (1 - p)^(n - k)

theorem prob_three_odd_in_eight_rolls :
  binomial_probability num_rolls target_odd prob_odd = 7/32 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_odd_in_eight_rolls_l1739_173959


namespace NUMINAMATH_CALUDE_monomial_exponents_l1739_173931

/-- Two monomials are of the same type if they have the same exponents for each variable -/
def SameType (m1 m2 : ℕ → ℕ) : Prop :=
  ∀ i, m1 i = m2 i

theorem monomial_exponents (a b : ℕ) :
  SameType (fun i => if i = 0 then a + 1 else if i = 1 then 3 else 0)
           (fun i => if i = 0 then 3 else if i = 1 then b else 0) →
  a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_monomial_exponents_l1739_173931


namespace NUMINAMATH_CALUDE_stratified_sampling_equality_l1739_173907

/-- Represents the number of people in each age group -/
structure Population where
  elderly : ℕ
  middleAged : ℕ

/-- Represents the number of people selected from each age group -/
structure Selected where
  elderly : ℕ
  middleAged : ℕ

/-- Checks if the selection maintains equal probability across strata -/
def isEqualProbability (pop : Population) (sel : Selected) : Prop :=
  (sel.elderly : ℚ) / pop.elderly = (sel.middleAged : ℚ) / pop.middleAged

theorem stratified_sampling_equality 
  (pop : Population) (sel : Selected) 
  (h1 : pop.elderly = 140) 
  (h2 : pop.middleAged = 210) 
  (h3 : sel.elderly = 4) 
  (h4 : isEqualProbability pop sel) : 
  sel.middleAged = 6 := by
  sorry

#check stratified_sampling_equality

end NUMINAMATH_CALUDE_stratified_sampling_equality_l1739_173907


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1739_173942

theorem necessary_not_sufficient_condition :
  (∀ x : ℝ, |x - 2| < 1 → 1 < x ∧ x < 4) ∧
  ¬(∀ x : ℝ, 1 < x ∧ x < 4 → |x - 2| < 1) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1739_173942


namespace NUMINAMATH_CALUDE_proposition_A_sufficient_not_necessary_for_B_l1739_173958

theorem proposition_A_sufficient_not_necessary_for_B :
  (∀ x : ℝ, 0 < x ∧ x < 5 → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_A_sufficient_not_necessary_for_B_l1739_173958


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1739_173944

theorem diophantine_equation_solution (x y z : ℕ+) :
  1 + 4^x.val + 4^y.val = z.val^2 ↔ 
  (∃ n : ℕ+, (x = n ∧ y = 2*n - 1 ∧ z = 1 + 2^(2*n.val - 1)) ∨ 
             (x = 2*n - 1 ∧ y = n ∧ z = 1 + 2^(2*n.val - 1))) :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1739_173944


namespace NUMINAMATH_CALUDE_derivative_ln_at_e_l1739_173916

open Real

theorem derivative_ln_at_e (f : ℝ → ℝ) (h : ∀ x, f x = log x) : 
  deriv f e = 1 / e := by
  sorry

end NUMINAMATH_CALUDE_derivative_ln_at_e_l1739_173916


namespace NUMINAMATH_CALUDE_selection_methods_eq_twelve_l1739_173964

/-- Represents the number of teachers available for selection -/
def total_teachers : ℕ := 4

/-- Represents the number of teachers to be selected -/
def selected_teachers : ℕ := 3

/-- Represents the number of phases in the training -/
def training_phases : ℕ := 3

/-- Represents the number of teachers who cannot participate in the first phase -/
def restricted_teachers : ℕ := 2

/-- Calculates the number of different selection methods -/
def selection_methods : ℕ := sorry

/-- Theorem stating that the number of selection methods is 12 -/
theorem selection_methods_eq_twelve : selection_methods = 12 := by sorry

end NUMINAMATH_CALUDE_selection_methods_eq_twelve_l1739_173964


namespace NUMINAMATH_CALUDE_johnson_family_seating_l1739_173913

def num_boys : ℕ := 5
def num_girls : ℕ := 4
def total_children : ℕ := num_boys + num_girls

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

def permutations (n k : ℕ) : ℕ := 
  if k ≤ n then
    (Finset.range k).prod (λ i => n - i)
  else 
    0

theorem johnson_family_seating :
  factorial total_children - 
  (7 * permutations num_boys 3 * factorial (total_children - 3)) = 60480 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l1739_173913


namespace NUMINAMATH_CALUDE_continuous_thin_stripe_probability_l1739_173925

-- Define the cube and its properties
def Cube := Fin 6

-- Define stripe properties
inductive StripeThickness
| thin
| thick

def StripeOrientation := Fin 4

structure Stripe :=
  (thickness : StripeThickness)
  (orientation : StripeOrientation)

def CubeConfiguration := Cube → Stripe

-- Define a function to check if a configuration has a continuous thin stripe
def hasContinuousThinStripe (config : CubeConfiguration) : Prop :=
  sorry -- Implementation details omitted

-- Define the probability space
def totalConfigurations : ℕ := 8^6

-- Define the number of favorable configurations
def favorableConfigurations : ℕ := 6144

-- Theorem statement
theorem continuous_thin_stripe_probability :
  (favorableConfigurations : ℚ) / totalConfigurations = 3 / 128 :=
sorry

end NUMINAMATH_CALUDE_continuous_thin_stripe_probability_l1739_173925


namespace NUMINAMATH_CALUDE_normal_dist_probability_l1739_173908

variable (ξ : Real)
variable (μ δ : Real)

-- ξ follows a normal distribution with mean μ and variance δ²
def normal_dist (ξ μ δ : Real) : Prop := sorry

-- Probability function
noncomputable def P (event : Real → Prop) : Real := sorry

theorem normal_dist_probability 
  (h1 : normal_dist ξ μ δ)
  (h2 : P (λ x => x > 4) = P (λ x => x < 2))
  (h3 : P (λ x => x ≤ 0) = 0.2) :
  P (λ x => 0 < x ∧ x < 6) = 0.6 := by sorry

end NUMINAMATH_CALUDE_normal_dist_probability_l1739_173908


namespace NUMINAMATH_CALUDE_root_condition_implies_m_range_l1739_173983

theorem root_condition_implies_m_range :
  ∀ (m : ℝ) (x₁ x₂ : ℝ),
    (m + 3) * x₁^2 - 4 * m * x₁ + 2 * m - 1 = 0 →
    (m + 3) * x₂^2 - 4 * m * x₂ + 2 * m - 1 = 0 →
    x₁ * x₂ < 0 →
    (x₁ < 0 ∧ x₂ > 0 → |x₁| > x₂) →
    (x₂ < 0 ∧ x₁ > 0 → |x₂| > x₁) →
    -3 < m ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_root_condition_implies_m_range_l1739_173983
