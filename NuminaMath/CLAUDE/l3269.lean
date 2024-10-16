import Mathlib

namespace NUMINAMATH_CALUDE_least_n_factorial_divisible_by_32_l3269_326920

theorem least_n_factorial_divisible_by_32 :
  ∀ n : ℕ, n > 0 → (n.factorial % 32 = 0) → n ≥ 8 :=
by
  sorry

end NUMINAMATH_CALUDE_least_n_factorial_divisible_by_32_l3269_326920


namespace NUMINAMATH_CALUDE_mike_bought_two_for_friend_l3269_326986

/-- Represents the problem of calculating the number of rose bushes Mike bought for his friend. -/
def mike_rose_bushes_for_friend 
  (total_rose_bushes : ℕ)
  (rose_bush_price : ℕ)
  (total_aloes : ℕ)
  (aloe_price : ℕ)
  (spent_on_self : ℕ) : ℕ :=
  total_rose_bushes - (spent_on_self - total_aloes * aloe_price) / rose_bush_price

/-- Theorem stating that Mike bought 2 rose bushes for his friend. -/
theorem mike_bought_two_for_friend :
  mike_rose_bushes_for_friend 6 75 2 100 500 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mike_bought_two_for_friend_l3269_326986


namespace NUMINAMATH_CALUDE_student_arrangements_l3269_326936

/-- Represents a student with a unique height -/
structure Student :=
  (height : ℕ)

/-- The set of 7 students with different heights -/
def Students : Finset Student :=
  sorry

/-- Predicate for a valid arrangement in a row -/
def ValidRowArrangement (arrangement : List Student) : Prop :=
  sorry

/-- Predicate for a valid arrangement in two rows and three columns -/
def Valid2x3Arrangement (arrangement : List (List Student)) : Prop :=
  sorry

theorem student_arrangements :
  (∃ (arrangements : Finset (List Student)),
    (∀ arr ∈ arrangements, ValidRowArrangement arr) ∧
    Finset.card arrangements = 20) ∧
  (∃ (arrangements : Finset (List (List Student))),
    (∀ arr ∈ arrangements, Valid2x3Arrangement arr) ∧
    Finset.card arrangements = 630) :=
  sorry

end NUMINAMATH_CALUDE_student_arrangements_l3269_326936


namespace NUMINAMATH_CALUDE_problem_solution_l3269_326924

theorem problem_solution (a b c d : ℝ) : 
  a^2 + b^2 + c^2 + 4 = d + Real.sqrt (2*a + 2*b + 2*c - d) → d = 23/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3269_326924


namespace NUMINAMATH_CALUDE_shaded_probability_three_fourths_l3269_326930

-- Define the right-angled triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Define the game board
structure GameBoard where
  triangle : RightTriangle
  total_regions : ℕ
  shaded_regions : ℕ
  regions_by_altitudes : total_regions = 4
  shaded_count : shaded_regions = 3

-- Define the probability function
def probability_shaded (board : GameBoard) : ℚ :=
  board.shaded_regions / board.total_regions

-- Theorem statement
theorem shaded_probability_three_fourths 
  (board : GameBoard) 
  (h1 : board.triangle.leg1 = 6) 
  (h2 : board.triangle.leg2 = 8) : 
  probability_shaded board = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_shaded_probability_three_fourths_l3269_326930


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l3269_326928

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, x > 0 ∧ n = 4*x + 2}

theorem gcd_of_B_is_two : ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l3269_326928


namespace NUMINAMATH_CALUDE_multiples_5_or_7_not_both_main_theorem_l3269_326918

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  (n - 1) / m

theorem multiples_5_or_7_not_both (upper_bound : ℕ) 
  (h_upper_bound : upper_bound = 101) : ℕ := by
  let multiples_5 := count_multiples upper_bound 5
  let multiples_7 := count_multiples upper_bound 7
  let multiples_35 := count_multiples upper_bound 35
  exact (multiples_5 + multiples_7 - 2 * multiples_35)

theorem main_theorem : multiples_5_or_7_not_both 101 rfl = 30 := by
  sorry

end NUMINAMATH_CALUDE_multiples_5_or_7_not_both_main_theorem_l3269_326918


namespace NUMINAMATH_CALUDE_jackets_sold_after_noon_l3269_326917

theorem jackets_sold_after_noon :
  let total_jackets : ℕ := 214
  let price_before_noon : ℚ := 31.95
  let price_after_noon : ℚ := 18.95
  let total_receipts : ℚ := 5108.30
  let jackets_after_noon : ℕ := 133
  let jackets_before_noon : ℕ := total_jackets - jackets_after_noon
  (jackets_before_noon : ℚ) * price_before_noon + (jackets_after_noon : ℚ) * price_after_noon = total_receipts →
  jackets_after_noon = 133 :=
by
  sorry

end NUMINAMATH_CALUDE_jackets_sold_after_noon_l3269_326917


namespace NUMINAMATH_CALUDE_angle_sum_equals_arctangent_of_ratio_l3269_326981

theorem angle_sum_equals_arctangent_of_ratio
  (θ φ : ℝ)
  (θ_acute : 0 < θ ∧ θ < π / 2)
  (φ_acute : 0 < φ ∧ φ < π / 2)
  (tan_θ : Real.tan θ = 2 / 9)
  (sin_φ : Real.sin φ = 3 / 5) :
  θ + 2 * φ = Real.arctan (230 / 15) :=
sorry

end NUMINAMATH_CALUDE_angle_sum_equals_arctangent_of_ratio_l3269_326981


namespace NUMINAMATH_CALUDE_us_apples_sold_fresh_l3269_326989

/-- Calculates the amount of apples sold fresh given total production and mixing percentage -/
def apples_sold_fresh (total_production : ℝ) (mixing_percentage : ℝ) : ℝ :=
  let remaining := total_production * (1 - mixing_percentage)
  remaining * 0.4

/-- Theorem stating that given the U.S. apple production conditions, 
    the amount of apples sold fresh is 2.24 million tons -/
theorem us_apples_sold_fresh :
  apples_sold_fresh 8 0.3 = 2.24 := by
  sorry

#eval apples_sold_fresh 8 0.3

end NUMINAMATH_CALUDE_us_apples_sold_fresh_l3269_326989


namespace NUMINAMATH_CALUDE_composite_function_difference_l3269_326935

theorem composite_function_difference (A B : ℝ) (h : A ≠ B) :
  let f := λ x : ℝ => A * x + B
  let g := λ x : ℝ => B * x + A
  (∀ x, f (g x) - g (f x) = 2 * (B - A)) →
  A + B = -2 := by
sorry

end NUMINAMATH_CALUDE_composite_function_difference_l3269_326935


namespace NUMINAMATH_CALUDE_abs_a_plus_b_equals_three_minus_sqrt_two_l3269_326978

theorem abs_a_plus_b_equals_three_minus_sqrt_two 
  (a b : ℝ) (h : Real.sqrt (2*a + 6) + |b - Real.sqrt 2| = 0) : 
  |a + b| = 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_a_plus_b_equals_three_minus_sqrt_two_l3269_326978


namespace NUMINAMATH_CALUDE_red_face_probability_l3269_326922

/-- A cube with colored faces -/
structure ColoredCube where
  redFaces : Nat
  blueFaces : Nat
  is_cube : redFaces + blueFaces = 6

/-- The probability of rolling a specific color on a colored cube -/
def rollProbability (cube : ColoredCube) (color : Nat) : Rat :=
  color / 6

/-- Theorem: The probability of rolling a red face on a cube with 5 red faces and 1 blue face is 5/6 -/
theorem red_face_probability :
  ∀ (cube : ColoredCube), cube.redFaces = 5 → cube.blueFaces = 1 →
  rollProbability cube cube.redFaces = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_red_face_probability_l3269_326922


namespace NUMINAMATH_CALUDE_unique_outstanding_wins_all_l3269_326926

variable {α : Type*} [Fintype α] [DecidableEq α]

-- Define the winning relation
variable (wins : α → α → Prop)

-- Assumption: Every pair of contestants has a clear winner
axiom clear_winner (a b : α) : a ≠ b → (wins a b ∨ wins b a) ∧ ¬(wins a b ∧ wins b a)

-- Define what it means to be an outstanding contestant
def is_outstanding (a : α) : Prop :=
  ∀ b : α, b ≠ a → wins a b ∨ (∃ c : α, wins c b ∧ wins a c)

-- Theorem: If there is a unique outstanding contestant, they win against all others
theorem unique_outstanding_wins_all (a : α) :
  (is_outstanding wins a ∧ ∀ b : α, is_outstanding wins b → b = a) →
  ∀ b : α, b ≠ a → wins a b :=
by sorry

end NUMINAMATH_CALUDE_unique_outstanding_wins_all_l3269_326926


namespace NUMINAMATH_CALUDE_M_divisible_by_52_l3269_326944

/-- The number formed by concatenating integers from 1 to 51 -/
def M : ℕ :=
  -- We don't actually compute M, just define it conceptually
  sorry

/-- M is divisible by 52 -/
theorem M_divisible_by_52 : 52 ∣ M := by
  sorry

end NUMINAMATH_CALUDE_M_divisible_by_52_l3269_326944


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3269_326973

theorem complex_equation_solution :
  ∀ z : ℂ, -Complex.I * z = (3 + 2 * Complex.I) * (1 - Complex.I) → z = 1 + 5 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3269_326973


namespace NUMINAMATH_CALUDE_forecast_variation_determinants_l3269_326931

/-- Represents a variable in regression analysis -/
inductive RegressionVariable
  | Forecast
  | Explanatory
  | Residual

/-- Represents the components that determine the variation of a variable -/
structure VariationDeterminants where
  components : List RegressionVariable

/-- Axiom: In regression analysis, the variation of the forecast variable
    is determined by both explanatory and residual variables -/
axiom regression_variation_determinants :
  VariationDeterminants.components (VariationDeterminants.mk [RegressionVariable.Explanatory, RegressionVariable.Residual]) =
  (VariationDeterminants.mk [RegressionVariable.Explanatory, RegressionVariable.Residual]).components

/-- Theorem: The variation of the forecast variable in regression analysis
    is determined by both explanatory and residual variables -/
theorem forecast_variation_determinants :
  VariationDeterminants.components (VariationDeterminants.mk [RegressionVariable.Explanatory, RegressionVariable.Residual]) =
  (VariationDeterminants.mk [RegressionVariable.Explanatory, RegressionVariable.Residual]).components :=
by sorry

end NUMINAMATH_CALUDE_forecast_variation_determinants_l3269_326931


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3269_326940

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 3 + 127 / 999) ∧ (x = 3124 / 999) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3269_326940


namespace NUMINAMATH_CALUDE_number_of_students_l3269_326921

def candy_bar_cost : ℚ := 2
def chips_cost : ℚ := 1/2

def student_purchase_cost : ℚ := candy_bar_cost + 2 * chips_cost

def total_cost : ℚ := 15

theorem number_of_students : 
  (total_cost / student_purchase_cost : ℚ) = 5 := by sorry

end NUMINAMATH_CALUDE_number_of_students_l3269_326921


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_l3269_326929

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem derivative_f_at_pi : 
  deriv f π = -π := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_l3269_326929


namespace NUMINAMATH_CALUDE_smallest_divisible_by_four_and_five_l3269_326911

/-- A function that checks if a number contains the digits 1, 2, 3, 4, and 5 exactly once -/
def containsDigitsOnce (n : ℕ) : Prop := sorry

/-- A function that returns the set of all five-digit numbers containing 1, 2, 3, 4, and 5 exactly once -/
def fiveDigitSet : Set ℕ := {n : ℕ | 10000 ≤ n ∧ n < 100000 ∧ containsDigitsOnce n}

theorem smallest_divisible_by_four_and_five :
  ∃ (n : ℕ), n ∈ fiveDigitSet ∧ n % 4 = 0 ∧ n % 5 = 0 ∧
  ∀ (m : ℕ), m ∈ fiveDigitSet → m % 4 = 0 → m % 5 = 0 → n ≤ m ∧
  n = 14532 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_four_and_five_l3269_326911


namespace NUMINAMATH_CALUDE_number_problem_l3269_326958

theorem number_problem : ∃ x : ℚ, x / 3 = x - 30 ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3269_326958


namespace NUMINAMATH_CALUDE_triangle_problem_l3269_326998

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) 
  (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π) 
  (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) 
  (h5 : a * Real.sin B = Real.sqrt 3 * b * Real.cos A) 
  (h6 : b = 3) 
  (h7 : c = 2) : 
  A = π / 3 ∧ a = Real.sqrt 7 := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l3269_326998


namespace NUMINAMATH_CALUDE_infinite_indices_inequality_l3269_326908

def FastGrowingSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (∀ C : ℝ, ∃ N : ℕ, ∀ k > N, (a k : ℝ) > C * k)

theorem infinite_indices_inequality
  (a : ℕ → ℕ)
  (h : FastGrowingSequence a) :
  ∀ M : ℕ, ∃ k > M, 2 * (a k) < (a (k - 1)) + (a (k + 1)) :=
sorry

end NUMINAMATH_CALUDE_infinite_indices_inequality_l3269_326908


namespace NUMINAMATH_CALUDE_min_a_for_quadratic_inequality_l3269_326963

theorem min_a_for_quadratic_inequality :
  (∀ x : ℝ, 0 < x → x ≤ 1/2 → x^2 + a*x + 1 ≥ 0) ↔ a ≥ -5/2 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_quadratic_inequality_l3269_326963


namespace NUMINAMATH_CALUDE_heather_starting_blocks_l3269_326992

/-- The number of blocks Heather shared with Jose -/
def shared_blocks : ℕ := 41

/-- The number of blocks Heather ended up with -/
def remaining_blocks : ℕ := 45

/-- The total number of blocks Heather started with -/
def starting_blocks : ℕ := shared_blocks + remaining_blocks

theorem heather_starting_blocks : starting_blocks = 86 := by
  sorry

end NUMINAMATH_CALUDE_heather_starting_blocks_l3269_326992


namespace NUMINAMATH_CALUDE_circle_radius_is_six_l3269_326946

theorem circle_radius_is_six (r : ℝ) (h : r > 0) :
  2 * 3 * (2 * Real.pi * r) = 2 * (Real.pi * r^2) → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_six_l3269_326946


namespace NUMINAMATH_CALUDE_max_min_difference_z_l3269_326961

theorem max_min_difference_z (x y z : ℝ) 
  (sum_condition : x + y + z = 3) 
  (sum_squares_condition : x^2 + y^2 + z^2 = 18) : 
  ∃ (z_max z_min : ℝ), 
    (∀ w, w = x ∨ w = y ∨ w = z → z_min ≤ w ∧ w ≤ z_max) ∧ 
    z_max - z_min = 6 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_z_l3269_326961


namespace NUMINAMATH_CALUDE_equation_solution_l3269_326905

theorem equation_solution : 
  ∃ (x₁ x₂ : ℚ), x₁ = 7/15 ∧ x₂ = 4/5 ∧ 
  (∀ x : ℚ, ⌊(5 + 6*x)/8⌋ = (15*x - 7)/5 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3269_326905


namespace NUMINAMATH_CALUDE_long_tennis_players_l3269_326938

theorem long_tennis_players (total : ℕ) (football : ℕ) (both : ℕ) (neither : ℕ) :
  total = 38 →
  football = 26 →
  both = 17 →
  neither = 9 →
  ∃ long_tennis : ℕ, long_tennis = 20 ∧ total = football + long_tennis - both + neither :=
by sorry

end NUMINAMATH_CALUDE_long_tennis_players_l3269_326938


namespace NUMINAMATH_CALUDE_count_four_digit_integers_l3269_326942

theorem count_four_digit_integers (y : ℕ) : 
  (∃ (n : ℕ), 1000 ≤ y ∧ y < 10000 ∧ (5678 * y + 123) % 29 = 890 % 29) →
  (Finset.filter (λ y => 1000 ≤ y ∧ y < 10000 ∧ (5678 * y + 123) % 29 = 890 % 29) (Finset.range 10000)).card = 310 :=
by sorry

end NUMINAMATH_CALUDE_count_four_digit_integers_l3269_326942


namespace NUMINAMATH_CALUDE_family_ages_exist_and_unique_l3269_326923

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem family_ages_exist_and_unique :
  ∃! (father mother daughter son : ℕ),
    is_perfect_square father ∧
    digit_product father = mother ∧
    digit_sum father = daughter ∧
    digit_sum mother = son ∧
    father ≤ 121 ∧
    mother > 0 ∧
    daughter > 0 ∧
    son > 0 :=
by sorry

end NUMINAMATH_CALUDE_family_ages_exist_and_unique_l3269_326923


namespace NUMINAMATH_CALUDE_exists_special_function_l3269_326914

theorem exists_special_function : 
  ∃ f : ℕ+ → ℕ+, f 1 = 2 ∧ ∀ n : ℕ+, f (f n) = f n + n :=
by sorry

end NUMINAMATH_CALUDE_exists_special_function_l3269_326914


namespace NUMINAMATH_CALUDE_van_speed_problem_l3269_326951

theorem van_speed_problem (distance : ℝ) (original_time : ℝ) (new_time_factor : ℝ) :
  distance = 288 →
  original_time = 6 →
  new_time_factor = 3 / 2 →
  let new_time := original_time * new_time_factor
  let new_speed := distance / new_time
  new_speed = 32 := by
sorry

end NUMINAMATH_CALUDE_van_speed_problem_l3269_326951


namespace NUMINAMATH_CALUDE_gcd_of_B_is_five_l3269_326968

def B : Set ℕ := {n | ∃ x : ℕ, n = (x - 2) + (x - 1) + x + (x + 1) + (x + 2)}

theorem gcd_of_B_is_five : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ∣ d) ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_five_l3269_326968


namespace NUMINAMATH_CALUDE_y2_less_than_y1_l3269_326972

def f (x : ℝ) := -4 * x - 3

theorem y2_less_than_y1 (y₁ y₂ : ℝ) 
  (h1 : f (-2) = y₁) 
  (h2 : f 5 = y₂) : 
  y₂ < y₁ := by
sorry

end NUMINAMATH_CALUDE_y2_less_than_y1_l3269_326972


namespace NUMINAMATH_CALUDE_policeman_catches_thief_l3269_326937

/-- Represents a point on the square --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a pathway on the square --/
inductive Pathway
  | Edge : Pathway
  | Diagonal : Pathway

/-- Represents the square with its pathways --/
structure Square :=
  (side_length : ℝ)
  (pathways : List Pathway)

/-- Represents the positions and speeds of the policeman and thief --/
structure ChaseState :=
  (policeman_pos : Point)
  (thief_pos : Point)
  (policeman_speed : ℝ)
  (thief_speed : ℝ)

/-- Defines the chase dynamics --/
def chase (square : Square) (initial_state : ChaseState) : Prop :=
  sorry

theorem policeman_catches_thief 
  (square : Square) 
  (initial_state : ChaseState) 
  (h1 : square.side_length > 0)
  (h2 : square.pathways.length = 6)
  (h3 : initial_state.policeman_speed > 2.1 * initial_state.thief_speed)
  (h4 : initial_state.thief_speed > 0) :
  chase square initial_state :=
by sorry

end NUMINAMATH_CALUDE_policeman_catches_thief_l3269_326937


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3269_326943

theorem expression_simplification_and_evaluation :
  let x : ℝ := 2 + Real.sqrt 2
  (1 - 3 / (x + 1)) / ((x^2 - 4*x + 4) / (x + 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3269_326943


namespace NUMINAMATH_CALUDE_product_of_integers_l3269_326927

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 20)
  (diff_squares_eq : x^2 - y^2 = 40) :
  x * y = 99 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l3269_326927


namespace NUMINAMATH_CALUDE_broomsticks_count_l3269_326913

/-- Represents the Halloween decorations problem --/
def halloween_decorations (skulls spiderwebs pumpkins cauldrons budget_left to_put_up total broomsticks : ℕ) : Prop :=
  skulls = 12 ∧
  spiderwebs = 12 ∧
  pumpkins = 2 * spiderwebs ∧
  cauldrons = 1 ∧
  budget_left = 20 ∧
  to_put_up = 10 ∧
  total = 83 ∧
  total = skulls + spiderwebs + pumpkins + cauldrons + budget_left + to_put_up + broomsticks

/-- Theorem stating that the number of broomsticks is 4 --/
theorem broomsticks_count :
  ∀ skulls spiderwebs pumpkins cauldrons budget_left to_put_up total broomsticks,
  halloween_decorations skulls spiderwebs pumpkins cauldrons budget_left to_put_up total broomsticks →
  broomsticks = 4 := by
  sorry

end NUMINAMATH_CALUDE_broomsticks_count_l3269_326913


namespace NUMINAMATH_CALUDE_given_program_has_syntax_error_l3269_326975

/-- Represents the structure of a DO-UNTIL loop -/
inductive DOUntilLoop
| correct : (body : String) → (condition : String) → DOUntilLoop
| incorrect : (body : String) → (untilKeyword : String) → (condition : String) → DOUntilLoop

/-- The given program structure -/
def givenProgram : DOUntilLoop :=
  DOUntilLoop.incorrect "x=x*x" "UNTIL" "x>10"

/-- Checks if a DO-UNTIL loop has correct syntax -/
def hasCorrectSyntax (loop : DOUntilLoop) : Prop :=
  match loop with
  | DOUntilLoop.correct _ _ => True
  | DOUntilLoop.incorrect _ _ _ => False

/-- Theorem stating that the given program has a syntax error -/
theorem given_program_has_syntax_error :
  ¬(hasCorrectSyntax givenProgram) := by
  sorry


end NUMINAMATH_CALUDE_given_program_has_syntax_error_l3269_326975


namespace NUMINAMATH_CALUDE_monotonicity_condition_l3269_326957

/-- A function f is monotonically increasing on ℝ -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The cubic function f(x) = x³ - 2x² - mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 - m*x + 1

theorem monotonicity_condition (m : ℝ) :
  (m > 4/3 → MonotonicallyIncreasing (f m)) ∧
  (∃ m : ℝ, m ≤ 4/3 ∧ MonotonicallyIncreasing (f m)) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_condition_l3269_326957


namespace NUMINAMATH_CALUDE_power_of_negative_cube_l3269_326982

theorem power_of_negative_cube (a : ℝ) : (-(a^3))^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_cube_l3269_326982


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_chord_l3269_326919

-- Define the given line
def givenLine (x y : ℝ) : Prop := 2 * x + 3 * y + 1 = 0

-- Define the given circle
def givenCircle (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 4 * y = 0

-- Define the perpendicular bisector
def perpendicularBisector (x y : ℝ) : Prop := 3 * x - 2 * y - 7 = 0

-- Theorem statement
theorem perpendicular_bisector_of_chord (A B : ℝ × ℝ) :
  givenLine A.1 A.2 ∧ givenLine B.1 B.2 ∧
  givenCircle A.1 A.2 ∧ givenCircle B.1 B.2 →
  ∃ (M : ℝ × ℝ), perpendicularBisector M.1 M.2 ∧
    (M.1 - A.1)^2 + (M.2 - A.2)^2 = (M.1 - B.1)^2 + (M.2 - B.2)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_chord_l3269_326919


namespace NUMINAMATH_CALUDE_inequality_system_solution_condition_l3269_326945

theorem inequality_system_solution_condition (m : ℝ) :
  (∃ x : ℝ, x - 2 * m < 0 ∧ x + m > 2) → m > 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_condition_l3269_326945


namespace NUMINAMATH_CALUDE_expansion_has_four_nonzero_terms_count_equals_nonzero_terms_count_l3269_326933

/-- The number of nonzero terms in the expansion of (x-2)(3x^2-2x+5)+4(x^3+x^2-3x) -/
def nonzero_terms_count : ℕ := 4

/-- The expansion of (x-2)(3x^2-2x+5)+4(x^3+x^2-3x) -/
def expanded_polynomial (x : ℝ) : ℝ := 7*x^3 - 4*x^2 - 3*x - 10

theorem expansion_has_four_nonzero_terms :
  (∃ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    ∀ x, expanded_polynomial x = a*x^3 + b*x^2 + c*x + d) ∧
  (∀ (a b c d e : ℝ), (∀ x, expanded_polynomial x = a*x^4 + b*x^3 + c*x^2 + d*x + e) →
    (a = 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) ∨
    (a = 0 ∧ b = 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) ∨
    (a = 0 ∧ b ≠ 0 ∧ c = 0 ∧ d ≠ 0 ∧ e ≠ 0) ∨
    (a = 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d = 0 ∧ e ≠ 0) ∨
    (a = 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e = 0)) :=
by sorry

theorem count_equals_nonzero_terms_count :
  nonzero_terms_count = 4 :=
by sorry

end NUMINAMATH_CALUDE_expansion_has_four_nonzero_terms_count_equals_nonzero_terms_count_l3269_326933


namespace NUMINAMATH_CALUDE_original_price_calculation_l3269_326970

/-- 
Given an article sold at a 40% profit, where the profit amount is 700 (in some currency unit),
prove that the original price of the article is 1750 (in the same currency unit).
-/
theorem original_price_calculation (profit_percentage : ℝ) (profit_amount : ℝ) 
  (h1 : profit_percentage = 40) 
  (h2 : profit_amount = 700) : 
  ∃ (original_price : ℝ), 
    original_price * (1 + profit_percentage / 100) - original_price = profit_amount ∧ 
    original_price = 1750 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l3269_326970


namespace NUMINAMATH_CALUDE_unique_digit_satisfying_conditions_l3269_326997

/-- A digit is a natural number from 0 to 9 -/
def Digit : Type := { n : ℕ // n ≤ 9 }

/-- Checks if a number is divisible by another number -/
def isDivisibleBy (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

/-- Constructs a number in the form 282,1A4 given a digit A -/
def constructNumber (A : Digit) : ℕ := 282100 + 10 * A.val + 4

/-- The main theorem: there exists exactly one digit A satisfying both conditions -/
theorem unique_digit_satisfying_conditions : 
  ∃! (A : Digit), isDivisibleBy 75 A.val ∧ isDivisibleBy (constructNumber A) 4 :=
sorry

end NUMINAMATH_CALUDE_unique_digit_satisfying_conditions_l3269_326997


namespace NUMINAMATH_CALUDE_dinner_slices_count_l3269_326932

/-- Represents the number of slices of pie served at different times -/
structure PieSlices where
  lunch_today : ℕ
  total_today : ℕ
  dinner_today : ℕ

/-- Theorem stating that given 7 slices served at lunch and 12 slices served in total today,
    the number of slices served at dinner is 5 -/
theorem dinner_slices_count (ps : PieSlices) 
  (h1 : ps.lunch_today = 7)
  (h2 : ps.total_today = 12)
  : ps.dinner_today = 5 := by
  sorry

end NUMINAMATH_CALUDE_dinner_slices_count_l3269_326932


namespace NUMINAMATH_CALUDE_jack_sugar_final_amount_l3269_326910

/-- Calculates the final amount of sugar Jack has after a series of transactions -/
def final_sugar_amount (initial : ℤ) (use_day2 borrow_day2 buy_day3 buy_day4 use_day5 return_day5 : ℤ) : ℤ :=
  initial - use_day2 - borrow_day2 + buy_day3 + buy_day4 - use_day5 + return_day5

/-- Theorem stating that Jack's final sugar amount is 85 pounds -/
theorem jack_sugar_final_amount :
  final_sugar_amount 65 18 5 30 20 10 3 = 85 := by
  sorry

end NUMINAMATH_CALUDE_jack_sugar_final_amount_l3269_326910


namespace NUMINAMATH_CALUDE_parabola_max_value_l3269_326902

theorem parabola_max_value :
  let f : ℝ → ℝ := fun x ↦ -2 * x^2 + 4 * x + 3
  ∃ (max : ℝ), ∀ (x : ℝ), f x ≤ max ∧ ∃ (x_max : ℝ), f x_max = max ∧ max = 5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_max_value_l3269_326902


namespace NUMINAMATH_CALUDE_remaining_income_percentage_l3269_326969

theorem remaining_income_percentage (total_income : ℝ) (food_percentage : ℝ) (education_percentage : ℝ) (rent_percentage : ℝ) :
  food_percentage = 35 →
  education_percentage = 25 →
  rent_percentage = 80 →
  total_income > 0 →
  let remaining_after_food_education := total_income * (1 - (food_percentage + education_percentage) / 100)
  let remaining_after_rent := remaining_after_food_education * (1 - rent_percentage / 100)
  remaining_after_rent / total_income = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_remaining_income_percentage_l3269_326969


namespace NUMINAMATH_CALUDE_crayon_selection_theorem_l3269_326906

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of crayons in the box -/
def total_crayons : ℕ := 15

/-- The number of metallic crayons in the box -/
def metallic_crayons : ℕ := 2

/-- The number of crayons to be selected -/
def selection_size : ℕ := 5

/-- The number of ways to select crayons with the given conditions -/
def selection_ways : ℕ := metallic_crayons * choose (total_crayons - metallic_crayons) (selection_size - 1)

theorem crayon_selection_theorem : selection_ways = 1430 := by sorry

end NUMINAMATH_CALUDE_crayon_selection_theorem_l3269_326906


namespace NUMINAMATH_CALUDE_power_evaluation_l3269_326979

theorem power_evaluation (a b : ℕ) (h : 360 = 2^a * 3^2 * 5^b) 
  (h2 : ∀ k > a, ¬ 2^k ∣ 360) (h5 : ∀ k > b, ¬ 5^k ∣ 360) : 
  (2/3 : ℚ)^(b-a) = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_power_evaluation_l3269_326979


namespace NUMINAMATH_CALUDE_midpoint_of_translated_segment_l3269_326960

/-- Given a segment s₁ with endpoints (2, -3) and (10, 7), and segment s₂ obtained by
    translating s₁ by 3 units to the left and 2 units down, prove that the midpoint
    of s₂ is (3, 0). -/
theorem midpoint_of_translated_segment :
  let s₁_start : ℝ × ℝ := (2, -3)
  let s₁_end : ℝ × ℝ := (10, 7)
  let translation : ℝ × ℝ := (-3, -2)
  let s₂_start : ℝ × ℝ := (s₁_start.1 + translation.1, s₁_start.2 + translation.2)
  let s₂_end : ℝ × ℝ := (s₁_end.1 + translation.1, s₁_end.2 + translation.2)
  let s₂_midpoint : ℝ × ℝ := ((s₂_start.1 + s₂_end.1) / 2, (s₂_start.2 + s₂_end.2) / 2)
  s₂_midpoint = (3, 0) := by
  sorry

end NUMINAMATH_CALUDE_midpoint_of_translated_segment_l3269_326960


namespace NUMINAMATH_CALUDE_burger_meal_cost_l3269_326984

theorem burger_meal_cost (burger_cost soda_cost : ℝ) : 
  soda_cost = (1/3) * burger_cost →
  burger_cost + soda_cost + 2 * (burger_cost + soda_cost) = 24 →
  burger_cost = 6 := by
sorry

end NUMINAMATH_CALUDE_burger_meal_cost_l3269_326984


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l3269_326962

theorem quadratic_root_existence (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  ∃ x : ℝ, (x^2 + a*x + b = 0) ∨ (x^2 + c*x + d = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l3269_326962


namespace NUMINAMATH_CALUDE_expression_evaluation_l3269_326995

theorem expression_evaluation (a b c d : ℝ) :
  (a - (b - (c + d))) - ((a + b) - (c - d)) = -2 * b + 2 * c := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3269_326995


namespace NUMINAMATH_CALUDE_least_m_for_x_sequence_l3269_326954

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 7 * x n + 12) / (x n + 8)

theorem least_m_for_x_sequence :
  ∃ m : ℕ, (∀ k < m, x k > 6 + 1 / 2^22) ∧ x m ≤ 6 + 1 / 2^22 ∧ m = 204 :=
by sorry

end NUMINAMATH_CALUDE_least_m_for_x_sequence_l3269_326954


namespace NUMINAMATH_CALUDE_remaining_box_mass_l3269_326965

/-- Given a list of box masses, prove that the 20 kg box remains in the store when two companies buy five boxes, with one company taking twice the mass of the other. -/
theorem remaining_box_mass (boxes : List ℕ) : boxes = [15, 16, 18, 19, 20, 31] →
  ∃ (company1 company2 : List ℕ),
    (company1.sum + company2.sum = boxes.sum - 20) ∧
    (company2.sum = 2 * company1.sum) ∧
    (company1.length + company2.length = 5) ∧
    (∀ x ∈ company1, x ∈ boxes) ∧
    (∀ x ∈ company2, x ∈ boxes) :=
by sorry

end NUMINAMATH_CALUDE_remaining_box_mass_l3269_326965


namespace NUMINAMATH_CALUDE_continued_fraction_equality_l3269_326964

theorem continued_fraction_equality : 
  1 + 1 / (2 + 1 / (2 + 1 / 3)) = 24 / 17 := by
sorry

end NUMINAMATH_CALUDE_continued_fraction_equality_l3269_326964


namespace NUMINAMATH_CALUDE_chess_game_theorem_l3269_326948

/-- Represents a three-player turn-based game system -/
structure GameSystem where
  total_games : ℕ
  player1_games : ℕ
  player2_games : ℕ
  player3_games : ℕ

/-- The game system satisfies the conditions of the problem -/
def valid_game_system (g : GameSystem) : Prop :=
  g.total_games = 27 ∧
  g.player1_games = 27 ∧
  g.player2_games = 13 ∧
  g.player3_games = g.total_games - g.player2_games

theorem chess_game_theorem (g : GameSystem) (h : valid_game_system g) :
  g.player3_games = 14 := by
  sorry


end NUMINAMATH_CALUDE_chess_game_theorem_l3269_326948


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l3269_326990

theorem binary_to_quaternary_conversion : 
  (1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 0 * 2^0) = 
  (1 * 4^2 + 3 * 4^1 + 0 * 4^0) := by
  sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l3269_326990


namespace NUMINAMATH_CALUDE_class_average_mark_l3269_326994

theorem class_average_mark (total_students : Nat) (excluded_students : Nat) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) : 
  total_students = 25 → 
  excluded_students = 5 → 
  excluded_avg = 40 → 
  remaining_avg = 90 → 
  (total_students * (total_students * remaining_avg - excluded_students * excluded_avg)) / 
    (total_students * (total_students - excluded_students)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l3269_326994


namespace NUMINAMATH_CALUDE_sally_has_88_cards_l3269_326956

/-- The number of Pokemon cards Sally has after receiving a gift and making a purchase -/
def sallys_cards (initial : ℕ) (gift : ℕ) (purchase : ℕ) : ℕ :=
  initial + gift + purchase

/-- Theorem: Sally has 88 Pokemon cards after starting with 27, receiving 41 as a gift, and buying 20 -/
theorem sally_has_88_cards : sallys_cards 27 41 20 = 88 := by
  sorry

end NUMINAMATH_CALUDE_sally_has_88_cards_l3269_326956


namespace NUMINAMATH_CALUDE_participants_with_three_points_l3269_326991

/-- Represents the number of participants in a tennis tournament with a specific score -/
def participantsWithScore (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Represents the total number of participants in the tournament -/
def totalParticipants (n : ℕ) : ℕ := 2^n + 4

/-- Theorem stating the number of participants with exactly 3 points at the end of the tournament -/
theorem participants_with_three_points (n : ℕ) (h : n > 4) :
  ∃ (winner : ℕ), winner = participantsWithScore n 3 + 1 ∧
  winner ≤ totalParticipants n :=
by sorry

end NUMINAMATH_CALUDE_participants_with_three_points_l3269_326991


namespace NUMINAMATH_CALUDE_chessboard_tiling_l3269_326976

/-- Represents a chessboard configuration -/
inductive ChessboardConfig
  | OneCornerRemoved
  | TwoOppositeCorners
  | TwoNonOppositeCorners

/-- Represents whether a configuration is tileable or not -/
inductive Tileable
  | Yes
  | No

/-- Function to determine if a chessboard configuration is tileable with 2x1 dominoes -/
def isTileable (config : ChessboardConfig) : Tileable :=
  match config with
  | ChessboardConfig.OneCornerRemoved => Tileable.No
  | ChessboardConfig.TwoOppositeCorners => Tileable.No
  | ChessboardConfig.TwoNonOppositeCorners => Tileable.Yes

theorem chessboard_tiling :
  (isTileable ChessboardConfig.OneCornerRemoved = Tileable.No) ∧
  (isTileable ChessboardConfig.TwoOppositeCorners = Tileable.No) ∧
  (isTileable ChessboardConfig.TwoNonOppositeCorners = Tileable.Yes) := by
  sorry

end NUMINAMATH_CALUDE_chessboard_tiling_l3269_326976


namespace NUMINAMATH_CALUDE_complex_number_properties_l3269_326950

theorem complex_number_properties (w : ℂ) (h : w^2 = 16 - 48*I) : 
  Complex.abs w = 4 * (10 : ℝ)^(1/4) ∧ 
  Complex.arg w = (Real.arctan (-3) / 2 + Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3269_326950


namespace NUMINAMATH_CALUDE_original_number_l3269_326952

theorem original_number (x : ℝ) : 
  (x * 10) * 0.001 = 0.375 → x = 37.5 := by
sorry

end NUMINAMATH_CALUDE_original_number_l3269_326952


namespace NUMINAMATH_CALUDE_u_converges_to_L_least_k_is_zero_l3269_326904

def u : ℕ → ℚ
  | 0 => 1/3
  | n+1 => 3 * u n - 3 * (u n)^2

def L : ℚ := 1/3

theorem u_converges_to_L (n : ℕ) : |u n - L| ≤ 1 / 2^100 := by
  sorry

theorem least_k_is_zero : ∀ k : ℕ, (∀ n : ℕ, n < k → |u n - L| > 1 / 2^100) → k = 0 := by
  sorry

end NUMINAMATH_CALUDE_u_converges_to_L_least_k_is_zero_l3269_326904


namespace NUMINAMATH_CALUDE_natural_growth_determined_by_birth_and_death_rates_l3269_326966

/-- Represents the rate of change in a population -/
structure PopulationRate :=
  (value : ℝ)

/-- The natural growth rate of a population -/
def naturalGrowthRate (birthRate deathRate : PopulationRate) : PopulationRate :=
  ⟨birthRate.value - deathRate.value⟩

/-- Theorem stating that the natural growth rate is determined by both birth and death rates -/
theorem natural_growth_determined_by_birth_and_death_rates 
  (birthRate deathRate : PopulationRate) :
  ∃ (f : PopulationRate → PopulationRate → PopulationRate), 
    naturalGrowthRate birthRate deathRate = f birthRate deathRate :=
by
  sorry


end NUMINAMATH_CALUDE_natural_growth_determined_by_birth_and_death_rates_l3269_326966


namespace NUMINAMATH_CALUDE_plum_difference_l3269_326977

def sharon_plums : ℕ := 7
def allan_plums : ℕ := 10

theorem plum_difference : allan_plums - sharon_plums = 3 := by
  sorry

end NUMINAMATH_CALUDE_plum_difference_l3269_326977


namespace NUMINAMATH_CALUDE_complex_multiplication_l3269_326996

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3269_326996


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_linear_l3269_326903

/-- The system of equations y = x^2 and y = 2x + k has exactly one solution if and only if k = -1 -/
theorem unique_solution_quadratic_linear (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = p.1^2 ∧ p.2 = 2 * p.1 + k) ↔ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_linear_l3269_326903


namespace NUMINAMATH_CALUDE_eighth_root_of_549755289601_l3269_326993

theorem eighth_root_of_549755289601 :
  let n : ℕ := 549755289601
  (n = 1 * 100^8 + 8 * 100^7 + 28 * 100^6 + 56 * 100^5 + 70 * 100^4 + 
       56 * 100^3 + 28 * 100^2 + 8 * 100 + 1) →
  (n : ℝ)^(1/8 : ℝ) = 101 := by
sorry

end NUMINAMATH_CALUDE_eighth_root_of_549755289601_l3269_326993


namespace NUMINAMATH_CALUDE_range_of_a_l3269_326912

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 4^x - (a+3)*2^x + 1 = 0) → a ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3269_326912


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3269_326939

def U : Finset Int := {-3, -2, -1, 0, 1}
def A : Finset Int := {-2, -1}
def B : Finset Int := {-3, -1, 0}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {-3, 0} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3269_326939


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l3269_326999

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 1) :
  (3 / a + 2 / b) ≥ 25 := by sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l3269_326999


namespace NUMINAMATH_CALUDE_bailey_rawhide_bones_l3269_326959

theorem bailey_rawhide_bones (dog_treats chew_toys credit_cards items_per_charge : ℕ) 
  (h1 : dog_treats = 8)
  (h2 : chew_toys = 2)
  (h3 : credit_cards = 4)
  (h4 : items_per_charge = 5) :
  credit_cards * items_per_charge - (dog_treats + chew_toys) = 10 := by
  sorry

end NUMINAMATH_CALUDE_bailey_rawhide_bones_l3269_326959


namespace NUMINAMATH_CALUDE_bus_stop_time_l3269_326915

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops speed_with_stops : ℝ) : 
  speed_without_stops = 64 → speed_with_stops = 48 → 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 15 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_time_l3269_326915


namespace NUMINAMATH_CALUDE_family_age_relations_l3269_326983

/-- Given family ages and relationships, prove age difference and Teresa's age at Michiko's birth -/
theorem family_age_relations (teresa_age morio_age : ℕ) 
  (h1 : teresa_age = 59)
  (h2 : morio_age = 71)
  (h3 : morio_age - 38 = michiko_age)
  (h4 : michiko_age - 4 = kenji_age)
  (h5 : teresa_age - 10 = emiko_age)
  (h6 : kenji_age = hideki_age)
  (h7 : morio_age = ryuji_age) :
  michiko_age - hideki_age = 4 ∧ teresa_age - michiko_age = 26 :=
by sorry


end NUMINAMATH_CALUDE_family_age_relations_l3269_326983


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l3269_326901

-- Define the solution set of the first inequality
def solution_set_1 : Set ℝ := {x : ℝ | 2 < x ∧ x < 3}

-- Define the coefficients a and b
def a : ℝ := 5
def b : ℝ := -6

-- Define the solution set of the second inequality
def solution_set_2 : Set ℝ := {x : ℝ | -1/2 < x ∧ x < -1/3}

theorem inequality_solution_sets :
  (∀ x : ℝ, x ∈ solution_set_1 ↔ x^2 - a*x - b < 0) →
  (∀ x : ℝ, x ∈ solution_set_2 ↔ b*x^2 - a*x - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l3269_326901


namespace NUMINAMATH_CALUDE_remainder_proof_l3269_326909

theorem remainder_proof (a b : ℕ) (h : a > b) : 
  220070 % (a + b) = 220070 - (a + b) * (2 * (a - b)) :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l3269_326909


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3269_326949

/-- The equation 7x^2 + 13x + d = 0 has rational solutions for d -/
def has_rational_solution (d : ℕ+) : Prop :=
  ∃ x : ℚ, 7 * x^2 + 13 * x + d.val = 0

/-- The set of positive integers d for which the equation has rational solutions -/
def solution_set : Set ℕ+ :=
  {d | has_rational_solution d}

theorem quadratic_equation_solution :
  ∃ (d₁ d₂ : ℕ+), d₁ ≠ d₂ ∧ 
    solution_set = {d₁, d₂} ∧
    d₁.val * d₂.val = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3269_326949


namespace NUMINAMATH_CALUDE_quiz_score_difference_l3269_326980

def quiz_scores : List (Float × Float) := [
  (0.05, 65),
  (0.25, 75),
  (0.40, 85),
  (0.20, 95),
  (0.10, 105)
]

def mean (scores : List (Float × Float)) : Float :=
  (scores.map (λ (p, s) => p * s)).sum

def median (scores : List (Float × Float)) : Float :=
  if (scores.map (λ (p, _) => p)).sum ≥ 0.5 then
    scores.filter (λ (_, s) => s ≥ 85)
      |> List.head!
      |> (λ (_, s) => s)
  else 85

theorem quiz_score_difference :
  median quiz_scores - mean quiz_scores = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_quiz_score_difference_l3269_326980


namespace NUMINAMATH_CALUDE_polynomial_factor_sum_l3269_326971

theorem polynomial_factor_sum (d M N K : ℝ) :
  (∃ a b : ℝ, (X^2 + 3*X + 1) * (X^2 + a*X + b) = X^4 - d*X^3 + M*X^2 + N*X + K) →
  M + N + K = 5*K - 4*d - 11 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factor_sum_l3269_326971


namespace NUMINAMATH_CALUDE_inequality_proof_l3269_326974

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b + b * c + a * c)^2 ≥ 3 * a * b * c * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3269_326974


namespace NUMINAMATH_CALUDE_divisible_by_four_or_seven_l3269_326953

theorem divisible_by_four_or_seven : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 60 ∧ (4 ∣ n ∨ 7 ∣ n)) ∧ 
  (∀ n, 1 ≤ n ∧ n ≤ 60 ∧ (4 ∣ n ∨ 7 ∣ n) → n ∈ S) ∧
  Finset.card S = 21 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_four_or_seven_l3269_326953


namespace NUMINAMATH_CALUDE_train_journey_time_l3269_326985

theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0)
  (h3 : (6 / 7 * usual_speed) * (usual_time + 20) = usual_speed * usual_time) :
  usual_time = 140 := by
sorry

end NUMINAMATH_CALUDE_train_journey_time_l3269_326985


namespace NUMINAMATH_CALUDE_jimmy_cards_l3269_326900

/-- 
Given:
- Jimmy gives 3 cards to Bob
- Jimmy gives twice as many cards to Mary as he gave to Bob
- Jimmy has 9 cards left after giving away cards

Prove that Jimmy initially had 18 cards.
-/
theorem jimmy_cards : 
  ∀ (cards_to_bob cards_to_mary cards_left initial_cards : ℕ),
  cards_to_bob = 3 →
  cards_to_mary = 2 * cards_to_bob →
  cards_left = 9 →
  initial_cards = cards_to_bob + cards_to_mary + cards_left →
  initial_cards = 18 := by
sorry


end NUMINAMATH_CALUDE_jimmy_cards_l3269_326900


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3269_326987

/-- The constant term in the expansion of x(1 - 1/√x)^5 is 10 -/
theorem constant_term_expansion (x : ℝ) (x_pos : x > 0) :
  ∃ (f : ℝ → ℝ), (∀ y, y ≠ 0 → f y = y * (1 - 1 / Real.sqrt y)^5) ∧
  (∃ c, ∀ ε > 0, ∃ δ > 0, ∀ y, 0 < |y - x| → |y - x| < δ → |f y - (10 + c * (y - x))| < ε * |y - x|) :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3269_326987


namespace NUMINAMATH_CALUDE_rich_walk_distance_l3269_326934

-- Define the walking pattern
def house_to_sidewalk : ℕ := 20
def sidewalk_to_road_end : ℕ := 200
def left_turn_multiplier : ℕ := 2
def final_stretch_divisor : ℕ := 2

-- Define the total distance walked
def total_distance : ℕ :=
  let initial_distance := house_to_sidewalk + sidewalk_to_road_end
  let after_left_turn := initial_distance + left_turn_multiplier * initial_distance
  let to_end_point := after_left_turn + after_left_turn / final_stretch_divisor
  2 * to_end_point

-- Theorem statement
theorem rich_walk_distance : total_distance = 1980 := by sorry

end NUMINAMATH_CALUDE_rich_walk_distance_l3269_326934


namespace NUMINAMATH_CALUDE_cory_fruit_order_l3269_326988

def fruit_arrangement (a o b g : ℕ) : ℕ :=
  Nat.factorial 9 / (Nat.factorial a * Nat.factorial o * Nat.factorial b * Nat.factorial g)

theorem cory_fruit_order : fruit_arrangement 3 3 2 1 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_cory_fruit_order_l3269_326988


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3269_326907

/-- Given an arithmetic sequence -1, a, b, m, 7, prove the eccentricity of x²/a² - y²/b² = 1 is √10 -/
theorem hyperbola_eccentricity (a b m : ℝ) : 
  (∃ d : ℝ, a = -1 + d ∧ b = a + d ∧ m = b + d ∧ 7 = m + d) →
  Real.sqrt ((b / a)^2 + 1) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3269_326907


namespace NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_abs_a_positive_l3269_326916

theorem a_positive_sufficient_not_necessary_for_abs_a_positive :
  (∃ a : ℝ, a > 0 → |a| > 0) ∧
  (∃ a : ℝ, |a| > 0 ∧ ¬(a > 0)) := by
  sorry

end NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_abs_a_positive_l3269_326916


namespace NUMINAMATH_CALUDE_regression_equation_properties_l3269_326925

-- Define the concept of a regression equation
structure RegressionEquation where
  -- Add necessary fields here
  mk :: -- Constructor

-- Define the property of temporality for regression equations
def has_temporality (eq : RegressionEquation) : Prop := sorry

-- Define the concept of sample values affecting applicability
def sample_values_affect_applicability (eq : RegressionEquation) : Prop := sorry

-- Theorem stating the correct properties of regression equations
theorem regression_equation_properties :
  ∀ (eq : RegressionEquation),
    (has_temporality eq) ∧
    (sample_values_affect_applicability eq) := by
  sorry

end NUMINAMATH_CALUDE_regression_equation_properties_l3269_326925


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3269_326967

theorem quadratic_root_problem (p : ℝ) : 
  (∃ x : ℂ, 3 * x^2 + p * x - 8 = 0 ∧ x = 2 + Complex.I) → p = -12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3269_326967


namespace NUMINAMATH_CALUDE_bad_shape_cards_l3269_326941

/-- Calculates the number of baseball cards in bad shape given the initial conditions and distributions --/
theorem bad_shape_cards (initial : ℕ) (from_father : ℕ) (from_ebay : ℕ) (to_dexter : ℕ) (kept : ℕ) : 
  initial + from_father + from_ebay - (to_dexter + kept) = 4 :=
by
  sorry

#check bad_shape_cards 4 13 36 29 20

end NUMINAMATH_CALUDE_bad_shape_cards_l3269_326941


namespace NUMINAMATH_CALUDE_units_digit_G_100_l3269_326947

-- Define the sequence G_n
def G (n : ℕ) : ℕ := 3 * 2^(2^n) + 2

-- Define a function to get the units digit
def units_digit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_G_100 : units_digit (G 100) = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_100_l3269_326947


namespace NUMINAMATH_CALUDE_triangle_frame_angles_l3269_326955

/-- A frame consisting of congruent triangles surrounding a square --/
structure TriangleFrame where
  /-- The number of triangles in the frame --/
  num_triangles : ℕ
  /-- The angles of each triangle in the frame --/
  triangle_angles : Fin 3 → ℝ
  /-- The sum of angles in each triangle is 180° --/
  angle_sum : (triangle_angles 0) + (triangle_angles 1) + (triangle_angles 2) = 180
  /-- The triangles form a complete circle at each corner of the square --/
  corner_sum : 4 * (triangle_angles 0) + 90 = 360
  /-- The triangles along each side of the square form a straight line --/
  side_sum : (triangle_angles 1) + (triangle_angles 2) + 90 = 180

/-- The theorem stating the angles of the triangles in the frame --/
theorem triangle_frame_angles (frame : TriangleFrame) 
  (h : frame.num_triangles = 21) : 
  frame.triangle_angles 0 = 67.5 ∧ 
  frame.triangle_angles 1 = 22.5 ∧ 
  frame.triangle_angles 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_frame_angles_l3269_326955
