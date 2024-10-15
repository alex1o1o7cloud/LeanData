import Mathlib

namespace NUMINAMATH_CALUDE_bart_earnings_l1619_161936

/-- The amount Bart earns per question in dollars -/
def earnings_per_question : ℚ := 0.2

/-- The number of questions in each survey -/
def questions_per_survey : ℕ := 10

/-- The number of surveys Bart completed on Monday -/
def monday_surveys : ℕ := 3

/-- The number of surveys Bart completed on Tuesday -/
def tuesday_surveys : ℕ := 4

/-- Theorem stating Bart's total earnings over two days -/
theorem bart_earnings : 
  (earnings_per_question * questions_per_survey * (monday_surveys + tuesday_surveys) : ℚ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_bart_earnings_l1619_161936


namespace NUMINAMATH_CALUDE_translation_theorem_l1619_161955

def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

def g (x : ℝ) : ℝ := -2 * (x + 2)^2 + 4 * (x + 2) + 4

theorem translation_theorem :
  ∀ x : ℝ, g x = f (x + 2) + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l1619_161955


namespace NUMINAMATH_CALUDE_scientific_notation_378300_l1619_161973

/-- Proves that 378300 is equal to 3.783 × 10^5 in scientific notation -/
theorem scientific_notation_378300 :
  ∃ (a : ℝ) (n : ℤ), 378300 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.783 ∧ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_378300_l1619_161973


namespace NUMINAMATH_CALUDE_min_framing_for_specific_picture_l1619_161961

/-- Calculates the minimum number of linear feet of framing required for a picture with given dimensions and border width. -/
def min_framing_feet (original_width original_height border_width : ℕ) : ℕ :=
  let enlarged_width := 2 * original_width
  let enlarged_height := 2 * original_height
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter := 2 * (total_width + total_height)
  (perimeter + 11) / 12

/-- The minimum number of linear feet of framing required for a 5-inch by 8-inch picture,
    doubled in size and surrounded by a 4-inch border, is 7 feet. -/
theorem min_framing_for_specific_picture :
  min_framing_feet 5 8 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_framing_for_specific_picture_l1619_161961


namespace NUMINAMATH_CALUDE_complete_square_factorization_l1619_161991

theorem complete_square_factorization :
  ∀ x : ℝ, x^2 + 4 + 4*x = (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_factorization_l1619_161991


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1619_161968

/-- An isosceles triangle with two sides of length 12 and one side of length 17 has a perimeter of 41. -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (side1 side2 base : ℝ),
      side1 = 12 ∧
      side2 = 12 ∧
      base = 17 ∧
      perimeter = side1 + side2 + base ∧
      perimeter = 41

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 41 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1619_161968


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_fifteen_l1619_161949

theorem last_digit_of_one_over_two_to_fifteen (n : ℕ) :
  n = 15 →
  (1 : ℚ) / (2^n : ℚ) * 10^n % 10 = 5 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_fifteen_l1619_161949


namespace NUMINAMATH_CALUDE_student_competition_assignments_l1619_161928

/-- The number of ways to assign students to competitions -/
def num_assignments (num_students : ℕ) (num_competitions : ℕ) : ℕ :=
  num_competitions ^ num_students

/-- Theorem: For 4 students and 3 competitions, there are 3^4 different assignment outcomes -/
theorem student_competition_assignments :
  num_assignments 4 3 = 3^4 := by
  sorry

end NUMINAMATH_CALUDE_student_competition_assignments_l1619_161928


namespace NUMINAMATH_CALUDE_block3_can_reach_target_l1619_161975

-- Define the board
def Board := Fin 3 × Fin 7

-- Define a block
structure Block where
  label : Nat
  position : Board

-- Define the game state
structure GameState where
  blocks : List Block

-- Define a valid move
inductive Move
| Up : Block → Move
| Down : Block → Move
| Left : Block → Move
| Right : Block → Move

-- Define the initial game state
def initialState : GameState := {
  blocks := [
    { label := 1, position := ⟨2, 2⟩ },
    { label := 2, position := ⟨3, 5⟩ },
    { label := 3, position := ⟨1, 4⟩ }
  ]
}

-- Define the target position
def targetPosition : Board := ⟨2, 4⟩

-- Function to check if a move is valid
def isValidMove (state : GameState) (move : Move) : Bool := sorry

-- Function to apply a move to the game state
def applyMove (state : GameState) (move : Move) : GameState := sorry

-- Theorem: There exists a sequence of valid moves to bring Block 3 to the target position
theorem block3_can_reach_target :
  ∃ (moves : List Move), 
    let finalState := moves.foldl (λ s m => applyMove s m) initialState
    (finalState.blocks.find? (λ b => b.label = 3)).map (λ b => b.position) = some targetPosition :=
sorry

end NUMINAMATH_CALUDE_block3_can_reach_target_l1619_161975


namespace NUMINAMATH_CALUDE_stratified_sampling_pine_saplings_l1619_161941

/-- Calculates the expected number of pine saplings in a stratified sample -/
def expected_pine_saplings (total_saplings : ℕ) (pine_saplings : ℕ) (sample_size : ℕ) : ℕ :=
  (pine_saplings * sample_size) / total_saplings

theorem stratified_sampling_pine_saplings :
  expected_pine_saplings 30000 4000 150 = 20 := by
  sorry

#eval expected_pine_saplings 30000 4000 150

end NUMINAMATH_CALUDE_stratified_sampling_pine_saplings_l1619_161941


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1619_161916

theorem tangent_line_to_circle (r : ℝ) : 
  r > 0 → 
  (∀ x y : ℝ, x + 2*y = r → (x^2 + y^2 = 2*r → (∀ ε > 0, ∃ x' y', x' + 2*y' = r ∧ (x'-x)^2 + (y'-y)^2 < ε^2 ∧ x'^2 + y'^2 ≠ 2*r))) → 
  r = 10 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1619_161916


namespace NUMINAMATH_CALUDE_quadratic_equations_common_root_condition_l1619_161918

/-- Given three quadratic equations, this theorem states the necessary and sufficient condition
for each equation to have a common root with one another but not all share a single common root. -/
theorem quadratic_equations_common_root_condition 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) : 
  let A := (a₁ + a₂ + a₃) / 2
  ∀ x₁ x₂ x₃ : ℝ,
  (x₁^2 - a₁*x₁ + b₁ = 0 ∧ 
   x₂^2 - a₂*x₂ + b₂ = 0 ∧ 
   x₃^2 - a₃*x₃ + b₃ = 0) →
  ((x₁ = x₂ ∨ x₂ = x₃ ∨ x₃ = x₁) ∧ 
   ¬(x₁ = x₂ ∧ x₂ = x₃)) ↔
  (b₁ = (A - a₂)*(A - a₃) ∧
   b₂ = (A - a₃)*(A - a₁) ∧
   b₃ = (A - a₁)*(A - a₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_common_root_condition_l1619_161918


namespace NUMINAMATH_CALUDE_parabola_c_value_l1619_161976

/-- A parabola with equation y = 2x^2 + bx + c passes through the points (1,5) and (5,5). 
    The value of c is 15. -/
theorem parabola_c_value : ∀ b c : ℝ, 
  (5 = 2 * (1 : ℝ)^2 + b * 1 + c) → 
  (5 = 2 * (5 : ℝ)^2 + b * 5 + c) → 
  c = 15 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1619_161976


namespace NUMINAMATH_CALUDE_binomial_distribution_properties_l1619_161900

/-- Represents the probability of success in a single trial -/
def p : ℝ := 0.6

/-- Represents the number of trials -/
def n : ℕ := 5

/-- Expected value of a binomial distribution -/
def expected_value : ℝ := n * p

/-- Variance of a binomial distribution -/
def variance : ℝ := n * p * (1 - p)

theorem binomial_distribution_properties :
  expected_value = 3 ∧ variance = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_properties_l1619_161900


namespace NUMINAMATH_CALUDE_dennis_loose_coins_l1619_161945

def loose_coins_problem (initial_amount : ℕ) (shirt_cost : ℕ) (bill_value : ℕ) (num_bills : ℕ) : Prop :=
  let total_change := initial_amount - shirt_cost
  let bills_amount := bill_value * num_bills
  let loose_coins := total_change - bills_amount
  loose_coins = 3

theorem dennis_loose_coins : 
  loose_coins_problem 50 27 10 2 := by
  sorry

end NUMINAMATH_CALUDE_dennis_loose_coins_l1619_161945


namespace NUMINAMATH_CALUDE_sequence_inequality_l1619_161935

theorem sequence_inequality (k : ℝ) : 
  (∀ n : ℕ+, (n + 1)^2 + k*(n + 1) + 2 > n^2 + k*n + 2) → k > -3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1619_161935


namespace NUMINAMATH_CALUDE_complex_number_proof_l1619_161914

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_number_proof (Z : ℂ) 
  (h1 : Complex.abs Z = 3)
  (h2 : is_pure_imaginary (Z + 3*I)) : 
  Z = 3*I := by sorry

end NUMINAMATH_CALUDE_complex_number_proof_l1619_161914


namespace NUMINAMATH_CALUDE_vertical_strips_count_l1619_161977

/-- Represents a rectangular grid with a hole -/
structure GridWithHole where
  outer_perimeter : ℕ
  hole_perimeter : ℕ
  horizontal_strips : ℕ

/-- The number of vertical strips in a GridWithHole -/
def vertical_strips (g : GridWithHole) : ℕ :=
  g.outer_perimeter / 2 + g.hole_perimeter / 2 - g.horizontal_strips

theorem vertical_strips_count (g : GridWithHole) 
  (h1 : g.outer_perimeter = 50)
  (h2 : g.hole_perimeter = 32)
  (h3 : g.horizontal_strips = 20) :
  vertical_strips g = 21 := by
  sorry

#eval vertical_strips { outer_perimeter := 50, hole_perimeter := 32, horizontal_strips := 20 }

end NUMINAMATH_CALUDE_vertical_strips_count_l1619_161977


namespace NUMINAMATH_CALUDE_reciprocal_ratio_sum_inequality_l1619_161902

theorem reciprocal_ratio_sum_inequality (a b : ℝ) (h : a * b < 0) :
  b / a + a / b ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_ratio_sum_inequality_l1619_161902


namespace NUMINAMATH_CALUDE_prime_1993_equations_l1619_161910

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem prime_1993_equations (h : isPrime 1993) :
  (∃ x y : ℕ, x^2 - y^2 = 1993) ∧
  (¬∃ x y : ℕ, x^3 - y^3 = 1993) ∧
  (¬∃ x y : ℕ, x^4 - y^4 = 1993) :=
by sorry

end NUMINAMATH_CALUDE_prime_1993_equations_l1619_161910


namespace NUMINAMATH_CALUDE_unique_solution_for_k_l1619_161985

/-- The equation (2x + 3)/(kx - 2) = x has exactly one solution when k = -4/3 -/
theorem unique_solution_for_k (k : ℚ) : 
  (∃! x, (2 * x + 3) / (k * x - 2) = x) ↔ k = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_k_l1619_161985


namespace NUMINAMATH_CALUDE_leaf_movement_l1619_161912

theorem leaf_movement (forward_distance : ℕ) (backward_distance : ℕ) (total_distance : ℕ) : 
  forward_distance = 5 → 
  backward_distance = 2 → 
  total_distance = 33 → 
  (total_distance / (forward_distance - backward_distance) : ℕ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_leaf_movement_l1619_161912


namespace NUMINAMATH_CALUDE_product_equals_533_l1619_161906

/-- Converts a list of digits in a given base to its decimal representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

/-- The binary representation of the first number -/
def binary_num : List Nat := [1, 0, 1, 1]

/-- The ternary representation of the second number -/
def ternary_num : List Nat := [2, 1, 1, 1]

theorem product_equals_533 :
  (to_decimal binary_num 2) * (to_decimal ternary_num 3) = 533 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_533_l1619_161906


namespace NUMINAMATH_CALUDE_winter_sales_l1619_161903

/-- The number of pizzas sold in millions for each season -/
structure PizzaSales where
  spring : ℝ
  summer : ℝ
  fall : ℝ
  winter : ℝ

/-- The total number of pizzas sold in millions -/
def total_sales (sales : PizzaSales) : ℝ :=
  sales.spring + sales.summer + sales.fall + sales.winter

/-- The given conditions of the problem -/
def pizza_problem (sales : PizzaSales) : Prop :=
  sales.summer = 6 ∧
  sales.spring = 2.5 ∧
  sales.fall = 3.5 ∧
  sales.summer = 0.4 * (total_sales sales)

/-- The theorem to be proved -/
theorem winter_sales (sales : PizzaSales) :
  pizza_problem sales → sales.winter = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_winter_sales_l1619_161903


namespace NUMINAMATH_CALUDE_barrel_capacity_l1619_161930

/-- Represents a barrel with two taps -/
structure Barrel :=
  (capacity : ℝ)
  (midwayTapRate : ℝ) -- Liters per minute
  (bottomTapRate : ℝ) -- Liters per minute

/-- Represents the scenario of drawing beer from the barrel -/
def drawBeer (barrel : Barrel) (earlyUseTime : ℝ) (assistantUseTime : ℝ) : Prop :=
  -- The capacity is twice the amount drawn early plus the amount drawn by the assistant
  barrel.capacity = 2 * (earlyUseTime * barrel.midwayTapRate + assistantUseTime * barrel.bottomTapRate)

/-- The main theorem stating the capacity of the barrel -/
theorem barrel_capacity : ∃ (b : Barrel), 
  b.midwayTapRate = 1 / 6 ∧ 
  b.bottomTapRate = 1 / 4 ∧ 
  drawBeer b 24 16 ∧ 
  b.capacity = 16 := by
  sorry

end NUMINAMATH_CALUDE_barrel_capacity_l1619_161930


namespace NUMINAMATH_CALUDE_smallest_fourth_lucky_number_l1619_161905

theorem smallest_fourth_lucky_number : 
  ∃ (n : ℕ), 
    n ≥ 10 ∧ n < 100 ∧
    (∀ m : ℕ, m ≥ 10 ∧ m < 100 ∧ m < n →
      ¬((57 + 13 + 72 + m) * 5 = 
        (5 + 7 + 1 + 3 + 7 + 2 + (m / 10) + (m % 10)) * 25)) ∧
    (57 + 13 + 72 + n) * 5 = 
      (5 + 7 + 1 + 3 + 7 + 2 + (n / 10) + (n % 10)) * 25 ∧
    n = 38 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fourth_lucky_number_l1619_161905


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l1619_161978

theorem gcd_lcm_sum : Nat.gcd 44 64 + Nat.lcm 48 18 = 148 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l1619_161978


namespace NUMINAMATH_CALUDE_calculate_income_before_tax_l1619_161934

/-- Given tax rates and differential savings, calculate the annual income before tax -/
theorem calculate_income_before_tax 
  (original_rate : ℝ) 
  (new_rate : ℝ) 
  (differential_savings : ℝ) 
  (h1 : original_rate = 0.42)
  (h2 : new_rate = 0.32)
  (h3 : differential_savings = 4240) :
  ∃ (income : ℝ), income * (original_rate - new_rate) = differential_savings ∧ income = 42400 := by
  sorry

end NUMINAMATH_CALUDE_calculate_income_before_tax_l1619_161934


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_0000000033_l1619_161940

/-- Expresses a given number in scientific notation -/
def scientific_notation (n : ℝ) : ℝ × ℤ :=
  sorry

theorem scientific_notation_of_0_0000000033 :
  scientific_notation 0.0000000033 = (3.3, -9) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_0000000033_l1619_161940


namespace NUMINAMATH_CALUDE_special_number_unique_l1619_161993

/-- The unique three-digit positive integer that is one more than a multiple of 3, 4, 5, 6, and 7 -/
def special_number : ℕ := 421

/-- Predicate to check if a number is a three-digit positive integer -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Predicate to check if a number is one more than a multiple of 3, 4, 5, 6, and 7 -/
def is_special (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 3 * k + 1 ∧ n = 4 * k + 1 ∧ n = 5 * k + 1 ∧ n = 6 * k + 1 ∧ n = 7 * k + 1

theorem special_number_unique :
  is_three_digit special_number ∧
  is_special special_number ∧
  ∀ (n : ℕ), is_three_digit n → is_special n → n = special_number :=
sorry

end NUMINAMATH_CALUDE_special_number_unique_l1619_161993


namespace NUMINAMATH_CALUDE_election_winner_percentage_l1619_161938

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  winner_votes = 3744 →
  margin = 288 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 52 / 100 := by
sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l1619_161938


namespace NUMINAMATH_CALUDE_minimize_distance_l1619_161988

/-- Given points P(-2,-3) and Q(5,3) in the xy-plane, and R(2,m) chosen such that PR+RQ is minimized, prove that m = 3/7 -/
theorem minimize_distance (P Q R : ℝ × ℝ) (m : ℝ) :
  P = (-2, -3) →
  Q = (5, 3) →
  R = (2, m) →
  (∀ m' : ℝ, dist P R + dist R Q ≤ dist P (2, m') + dist (2, m') Q) →
  m = 3/7 := by
  sorry


end NUMINAMATH_CALUDE_minimize_distance_l1619_161988


namespace NUMINAMATH_CALUDE_adjacent_complex_numbers_max_sum_squares_l1619_161925

theorem adjacent_complex_numbers_max_sum_squares :
  ∀ (a b : ℝ),
  let z1 : ℂ := a + Complex.I * Real.sqrt 3
  let z2 : ℂ := 1 + Complex.I * b
  Complex.abs (z1 - z2) = 1 →
  ∃ (max : ℝ), max = 9 ∧ a^2 + b^2 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_adjacent_complex_numbers_max_sum_squares_l1619_161925


namespace NUMINAMATH_CALUDE_average_of_three_l1619_161996

theorem average_of_three (y : ℝ) : (15 + 24 + y) / 3 = 20 → y = 21 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_l1619_161996


namespace NUMINAMATH_CALUDE_lcm_problem_l1619_161990

theorem lcm_problem (a b : ℕ+) : 
  a = 1491 → Nat.lcm a b = 5964 → b = 4 := by sorry

end NUMINAMATH_CALUDE_lcm_problem_l1619_161990


namespace NUMINAMATH_CALUDE_bread_roll_combinations_l1619_161922

theorem bread_roll_combinations :
  let total_rolls : ℕ := 10
  let num_types : ℕ := 4
  let min_rolls_type1 : ℕ := 2
  let min_rolls_type2 : ℕ := 2
  let min_rolls_type3 : ℕ := 1
  let min_rolls_type4 : ℕ := 1
  let remaining_rolls : ℕ := total_rolls - (min_rolls_type1 + min_rolls_type2 + min_rolls_type3 + min_rolls_type4)
  (Nat.choose (remaining_rolls + num_types - 1) (num_types - 1)) = 35 :=
by sorry

end NUMINAMATH_CALUDE_bread_roll_combinations_l1619_161922


namespace NUMINAMATH_CALUDE_star_arrangements_l1619_161995

/-- The number of points on a regular ten-pointed star -/
def num_points : ℕ := 20

/-- The number of rotational symmetries of a regular ten-pointed star -/
def num_rotations : ℕ := 10

/-- The number of reflectional symmetries of a regular ten-pointed star -/
def num_reflections : ℕ := 2

/-- The total number of symmetries of a regular ten-pointed star -/
def total_symmetries : ℕ := num_rotations * num_reflections

/-- The number of distinct arrangements of objects on a regular ten-pointed star -/
def distinct_arrangements : ℕ := Nat.factorial num_points / total_symmetries

theorem star_arrangements :
  distinct_arrangements = Nat.factorial (num_points - 1) := by
  sorry

end NUMINAMATH_CALUDE_star_arrangements_l1619_161995


namespace NUMINAMATH_CALUDE_train_speed_crossing_bridge_l1619_161929

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge 
  (train_length : Real) 
  (bridge_length : Real) 
  (crossing_time : Real) 
  (h1 : train_length = 150) 
  (h2 : bridge_length = 320) 
  (h3 : crossing_time = 40) : 
  (train_length + bridge_length) / crossing_time = 11.75 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_crossing_bridge_l1619_161929


namespace NUMINAMATH_CALUDE_sum_of_roots_is_negative_4015_l1619_161913

/-- Represents the polynomial (x-1)^2009 + 3(x-2)^2008 + 5(x-3)^2007 + ⋯ + 4017(x-2009)^2 + 4019(x-4018) -/
def specialPolynomial : Polynomial ℝ := sorry

/-- The sum of the roots of the specialPolynomial -/
def sumOfRoots : ℝ := sorry

/-- Theorem stating that the sum of the roots of the specialPolynomial is -4015 -/
theorem sum_of_roots_is_negative_4015 : sumOfRoots = -4015 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_negative_4015_l1619_161913


namespace NUMINAMATH_CALUDE_abs_neg_three_l1619_161927

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_l1619_161927


namespace NUMINAMATH_CALUDE_fraction_equality_l1619_161951

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (1/a + 1/b) / (1/a - 1/b) = 1001) : 
  (a + b) / (a - b) = 1001 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1619_161951


namespace NUMINAMATH_CALUDE_prize_cost_l1619_161987

theorem prize_cost (total_cost : ℕ) (num_prizes : ℕ) (cost_per_prize : ℕ) 
  (h1 : total_cost = 120)
  (h2 : num_prizes = 6)
  (h3 : total_cost = num_prizes * cost_per_prize) :
  cost_per_prize = 20 := by
  sorry

end NUMINAMATH_CALUDE_prize_cost_l1619_161987


namespace NUMINAMATH_CALUDE_principal_is_15000_l1619_161909

/-- Represents the loan details and calculations -/
structure Loan where
  principal : ℝ
  interestRates : Fin 3 → ℝ
  totalInterest : ℝ

/-- Calculates the total interest paid over 3 years -/
def totalInterestPaid (loan : Loan) : ℝ :=
  (loan.interestRates 0 + loan.interestRates 1 + loan.interestRates 2) * loan.principal

/-- Theorem stating that given the conditions, the principal amount is 15000 -/
theorem principal_is_15000 (loan : Loan)
  (h1 : loan.interestRates 0 = 0.10)
  (h2 : loan.interestRates 1 = 0.12)
  (h3 : loan.interestRates 2 = 0.14)
  (h4 : loan.totalInterest = 5400)
  (h5 : totalInterestPaid loan = loan.totalInterest) :
  loan.principal = 15000 := by
  sorry

#check principal_is_15000

end NUMINAMATH_CALUDE_principal_is_15000_l1619_161909


namespace NUMINAMATH_CALUDE_juan_has_498_marbles_l1619_161981

/-- The number of marbles Connie has -/
def connies_marbles : ℕ := 323

/-- The number of additional marbles Juan has compared to Connie -/
def juans_additional_marbles : ℕ := 175

/-- The total number of marbles Juan has -/
def juans_marbles : ℕ := connies_marbles + juans_additional_marbles

/-- Theorem stating that Juan has 498 marbles -/
theorem juan_has_498_marbles : juans_marbles = 498 := by
  sorry

end NUMINAMATH_CALUDE_juan_has_498_marbles_l1619_161981


namespace NUMINAMATH_CALUDE_band_members_max_l1619_161937

theorem band_members_max (m r x : ℕ) : 
  m < 100 →
  r * x + 3 = m →
  (r - 1) * (x + 2) = m →
  ∀ n : ℕ, n < 100 ∧ (∃ r' x' : ℕ, r' * x' + 3 = n ∧ (r' - 1) * (x' + 2) = n) → n ≤ 91 :=
by sorry

end NUMINAMATH_CALUDE_band_members_max_l1619_161937


namespace NUMINAMATH_CALUDE_complex_square_sum_l1619_161942

theorem complex_square_sum (a b : ℝ) : 
  (Complex.mk a b = (1 + Complex.I)^2) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_sum_l1619_161942


namespace NUMINAMATH_CALUDE_parabola_focus_theorem_l1619_161946

/-- Represents a parabola y² = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of the parabola lies on the line x + y - 1 = 0 -/
def focus_on_line (para : Parabola) (F : Point) : Prop :=
  F.x + F.y = 1

/-- The equation of the parabola is y² = 4x -/
def parabola_equation (para : Parabola) : Prop :=
  para.p = 2

/-- A line through the focus at 45° angle -/
def line_through_focus (F : Point) (A B : Point) : Prop :=
  (A.y - F.y) = (A.x - F.x) ∧ (B.y - F.y) = (B.x - F.x)

/-- A and B are on the parabola -/
def points_on_parabola (para : Parabola) (A B : Point) : Prop :=
  A.y^2 = 2 * para.p * A.x ∧ B.y^2 = 2 * para.p * B.x

/-- The length of AB is 8 -/
def length_AB (A B : Point) : Prop :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 8

theorem parabola_focus_theorem (para : Parabola) (F A B : Point) :
  focus_on_line para F →
  line_through_focus F A B →
  points_on_parabola para A B →
  parabola_equation para ∧ length_AB A B :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_theorem_l1619_161946


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l1619_161953

def initial_amount : ℕ := 85

def book_prices : List ℕ := [4, 6, 3, 7, 5, 8, 2, 6, 3, 5, 7, 4, 5, 6, 3]

theorem money_left_after_purchase : 
  initial_amount - (book_prices.sum) = 11 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l1619_161953


namespace NUMINAMATH_CALUDE_theater_seats_count_l1619_161919

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increase : ℕ
  last_row_seats : ℕ

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := (t.last_row_seats - t.first_row_seats) / t.seat_increase + 1
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- Theorem stating that a theater with given conditions has 770 seats -/
theorem theater_seats_count :
  ∀ t : Theater,
    t.first_row_seats = 14 →
    t.seat_increase = 2 →
    t.last_row_seats = 56 →
    total_seats t = 770 :=
by
  sorry


end NUMINAMATH_CALUDE_theater_seats_count_l1619_161919


namespace NUMINAMATH_CALUDE_number_problem_l1619_161947

theorem number_problem (x : ℝ) : 0.3 * x = 0.6 * 50 + 30 → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1619_161947


namespace NUMINAMATH_CALUDE_chess_players_count_l1619_161932

theorem chess_players_count : ℕ :=
  let total_players : ℕ := 40
  let never_lost_fraction : ℚ := 1/4
  let lost_at_least_once : ℕ := 30
  have h1 : (1 - never_lost_fraction) * total_players = lost_at_least_once := by sorry
  have h2 : never_lost_fraction * total_players + lost_at_least_once = total_players := by sorry
  total_players

end NUMINAMATH_CALUDE_chess_players_count_l1619_161932


namespace NUMINAMATH_CALUDE_comprehensive_score_calculation_l1619_161992

theorem comprehensive_score_calculation (initial_score retest_score : ℝ) 
  (initial_weight retest_weight : ℝ) (h1 : initial_score = 400) 
  (h2 : retest_score = 85) (h3 : initial_weight = 0.4) (h4 : retest_weight = 0.6) :
  initial_score * initial_weight + retest_score * retest_weight = 211 := by
  sorry

end NUMINAMATH_CALUDE_comprehensive_score_calculation_l1619_161992


namespace NUMINAMATH_CALUDE_ac_in_open_interval_sum_of_endpoints_l1619_161948

/-- Represents a triangle ABC with an angle bisector from A to D on BC -/
structure AngleBisectorTriangle where
  -- The length of side AB
  ab : ℝ
  -- The length of CD (part of BC)
  cd : ℝ
  -- The length of AC
  ac : ℝ
  -- Assumption that AB = 15
  ab_eq : ab = 15
  -- Assumption that CD = 5
  cd_eq : cd = 5
  -- Assumption that AC is positive
  ac_pos : ac > 0
  -- Assumption that ABC forms a valid triangle
  triangle_inequality : ac + cd + (75 / ac) > ab ∧ ab + cd + (75 / ac) > ac ∧ ab + ac > cd + (75 / ac)
  -- Assumption that AD is the angle bisector
  angle_bisector : ab / ac = (75 / ac) / cd

/-- The main theorem stating that AC must be in the open interval (5, 25) -/
theorem ac_in_open_interval (t : AngleBisectorTriangle) : 5 < t.ac ∧ t.ac < 25 := by
  sorry

/-- The sum of the endpoints of the interval is 30 -/
theorem sum_of_endpoints : 5 + 25 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ac_in_open_interval_sum_of_endpoints_l1619_161948


namespace NUMINAMATH_CALUDE_fraction_equality_l1619_161971

theorem fraction_equality (x : ℝ) : 
  (4 + 2*x) / (7 + 3*x) = (2 + 3*x) / (4 + 5*x) ↔ x = -1 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1619_161971


namespace NUMINAMATH_CALUDE_cookies_percentage_increase_l1619_161931

def cookies_problem (monday tuesday wednesday : ℕ) : Prop :=
  monday = 5 ∧
  tuesday = 2 * monday ∧
  wednesday > tuesday ∧
  monday + tuesday + wednesday = 29

theorem cookies_percentage_increase :
  ∀ monday tuesday wednesday : ℕ,
  cookies_problem monday tuesday wednesday →
  (wednesday - tuesday : ℚ) / tuesday * 100 = 40 :=
by sorry

end NUMINAMATH_CALUDE_cookies_percentage_increase_l1619_161931


namespace NUMINAMATH_CALUDE_right_angled_triangles_with_special_property_l1619_161989

theorem right_angled_triangles_with_special_property :
  {(a, b, c) : ℕ × ℕ × ℕ | 
    0 < a ∧ a < b ∧ b < c ∧
    a * b = 4 * (a + b + c) ∧
    a * a + b * b = c * c} =
  {(10, 24, 26), (12, 16, 20), (9, 40, 41)} :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangles_with_special_property_l1619_161989


namespace NUMINAMATH_CALUDE_company_workers_l1619_161950

theorem company_workers (total : ℕ) (men : ℕ) : 
  (total / 3 : ℚ) = total / 3 →  -- One-third of workers don't have a retirement plan
  (1 / 5 : ℚ) * (total / 3 : ℚ) = total / 15 →  -- 20% of workers without a retirement plan are women
  (2 / 5 : ℚ) * ((2 * total) / 3 : ℚ) = (4 * total) / 15 →  -- 40% of workers with a retirement plan are men
  men = 144 →  -- There are 144 men
  total - men = 126  -- The number of women workers is 126
  := by sorry

end NUMINAMATH_CALUDE_company_workers_l1619_161950


namespace NUMINAMATH_CALUDE_subtraction_and_simplification_l1619_161954

theorem subtraction_and_simplification :
  (9 : ℚ) / 23 - 5 / 69 = 22 / 69 ∧ 
  ∀ (a b : ℤ), (a : ℚ) / b = 22 / 69 → (a.gcd b = 1 → a = 22 ∧ b = 69) :=
by sorry

end NUMINAMATH_CALUDE_subtraction_and_simplification_l1619_161954


namespace NUMINAMATH_CALUDE_one_of_each_color_probability_l1619_161980

def total_marbles : ℕ := 9
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 3
def green_marbles : ℕ := 3
def selected_marbles : ℕ := 3

def probability_one_of_each_color : ℚ := 9 / 28

theorem one_of_each_color_probability :
  probability_one_of_each_color = 
    (red_marbles * blue_marbles * green_marbles : ℚ) / 
    (Nat.choose total_marbles selected_marbles) :=
by sorry

end NUMINAMATH_CALUDE_one_of_each_color_probability_l1619_161980


namespace NUMINAMATH_CALUDE_f_of_3_equals_29_l1619_161944

/-- Given f(x) = x^2 + 4x + 8, prove that f(3) = 29 -/
theorem f_of_3_equals_29 :
  let f : ℝ → ℝ := λ x ↦ x^2 + 4*x + 8
  f 3 = 29 := by sorry

end NUMINAMATH_CALUDE_f_of_3_equals_29_l1619_161944


namespace NUMINAMATH_CALUDE_sugar_amount_in_recipe_l1619_161994

/-- Given a recipe that requires a total of 10 cups of flour, 
    with 2 cups already added, and the remaining flour needed 
    being 5 cups more than the amount of sugar, 
    prove that the recipe calls for 3 cups of sugar. -/
theorem sugar_amount_in_recipe 
  (total_flour : ℕ) 
  (added_flour : ℕ) 
  (sugar : ℕ) : 
  total_flour = 10 → 
  added_flour = 2 → 
  total_flour = added_flour + (sugar + 5) → 
  sugar = 3 := by
  sorry

end NUMINAMATH_CALUDE_sugar_amount_in_recipe_l1619_161994


namespace NUMINAMATH_CALUDE_adam_laundry_l1619_161998

/-- Given a total number of loads and a number of washed loads, calculate the remaining loads to wash. -/
def remaining_loads (total : ℕ) (washed : ℕ) : ℕ :=
  total - washed

/-- Theorem stating that given 14 total loads and 8 washed loads, the remaining loads is 6. -/
theorem adam_laundry : remaining_loads 14 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_adam_laundry_l1619_161998


namespace NUMINAMATH_CALUDE_no_integer_solution_l1619_161904

theorem no_integer_solution : ¬∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1619_161904


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l1619_161999

theorem smallest_number_with_given_remainders :
  ∃ n : ℕ, (n % 2 = 1) ∧ (n % 3 = 2) ∧ (n % 4 = 3) ∧
  (∀ m : ℕ, m < n → ¬((m % 2 = 1) ∧ (m % 3 = 2) ∧ (m % 4 = 3))) ∧
  n = 11 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l1619_161999


namespace NUMINAMATH_CALUDE_final_S_value_l1619_161962

/-- Calculates the final value of S after executing the loop three times -/
def final_S : ℕ → ℕ → ℕ → ℕ
| 0, s, i => s
| (n + 1), s, i => final_S n (s + i) (i + 2)

theorem final_S_value :
  final_S 3 0 1 = 9 := by
sorry

end NUMINAMATH_CALUDE_final_S_value_l1619_161962


namespace NUMINAMATH_CALUDE_abc_inequality_l1619_161908

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_abc : a + b + c = 1) :
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 ∧ 1/a + 1/b + 1/c ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l1619_161908


namespace NUMINAMATH_CALUDE_polynomial_expansion_sum_l1619_161965

theorem polynomial_expansion_sum (m : ℝ) (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (1 + m * x)^6 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6) →
  (a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 64) →
  (m = 1 ∨ m = -3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_sum_l1619_161965


namespace NUMINAMATH_CALUDE_reading_time_difference_l1619_161986

/-- Proves the difference in reading time between two people for a given book -/
theorem reading_time_difference
  (xanthia_speed : ℕ)  -- Xanthia's reading speed in pages per hour
  (molly_speed : ℕ)    -- Molly's reading speed in pages per hour
  (book_pages : ℕ)     -- Number of pages in the book
  (h1 : xanthia_speed = 120)
  (h2 : molly_speed = 60)
  (h3 : book_pages = 360) :
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 180 :=
by sorry

end NUMINAMATH_CALUDE_reading_time_difference_l1619_161986


namespace NUMINAMATH_CALUDE_children_savings_l1619_161979

/-- The total savings of Josiah, Leah, and Megan -/
def total_savings (josiah_daily : ℚ) (josiah_days : ℕ) 
                  (leah_daily : ℚ) (leah_days : ℕ)
                  (megan_daily : ℚ) (megan_days : ℕ) : ℚ :=
  josiah_daily * josiah_days + leah_daily * leah_days + megan_daily * megan_days

/-- Theorem stating that the total savings of the three children is $28 -/
theorem children_savings : 
  total_savings 0.25 24 0.50 20 1.00 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_children_savings_l1619_161979


namespace NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l1619_161984

def a (x : ℝ) : Fin 2 → ℝ := ![1, 2 - x]
def b (x : ℝ) : Fin 2 → ℝ := ![2 + x, 3]

def vectors_collinear (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ (i : Fin 2), u i = k * v i

def norm_squared (v : Fin 2 → ℝ) : ℝ :=
  (v 0) ^ 2 + (v 1) ^ 2

theorem condition_neither_sufficient_nor_necessary :
  ¬(∀ x : ℝ, norm_squared (a x) = 2 → vectors_collinear (a x) (b x)) ∧
  ¬(∀ x : ℝ, vectors_collinear (a x) (b x) → norm_squared (a x) = 2) := by
  sorry

end NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l1619_161984


namespace NUMINAMATH_CALUDE_sum_of_squares_l1619_161957

theorem sum_of_squares : (10 + 3)^2 + (7 - 5)^2 = 173 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1619_161957


namespace NUMINAMATH_CALUDE_sum_inequality_l1619_161911

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l1619_161911


namespace NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l1619_161952

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

/-- Theorem: The 50th term of the arithmetic sequence starting with 2 and incrementing by 5 is 247 -/
theorem fiftieth_term_of_sequence : arithmetic_sequence 2 5 50 = 247 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l1619_161952


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_l1619_161997

theorem simplify_trigonometric_expression (α : ℝ) :
  (1 - Real.cos (2 * α)) * Real.cos (π / 4 + 2 * α) / (2 * Real.sin (2 * α) ^ 2 - Real.sin (4 * α)) =
  -Real.sqrt 2 / 4 * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_l1619_161997


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1619_161939

theorem nested_fraction_evaluation :
  1 + (1 / (1 + (1 / (1 + (1 / 2))))) = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1619_161939


namespace NUMINAMATH_CALUDE_polynomial_equality_l1619_161924

theorem polynomial_equality (a b c : ℝ) : 
  (∀ x : ℝ, x * (x + 1) = a + b * x + c * x^2) → a + b + c = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1619_161924


namespace NUMINAMATH_CALUDE_m_range_l1619_161982

def A : Set ℝ := {x | (x + 1) / (x - 3) < 0}
def B (m : ℝ) : Set ℝ := {x | -1 < x ∧ x < m + 1}

theorem m_range (m : ℝ) : 
  (∀ x, x ∈ B m → x ∈ A) ∧ 
  (∃ x, x ∈ A ∧ x ∉ B m) → 
  m > 2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1619_161982


namespace NUMINAMATH_CALUDE_nancy_picked_three_apples_l1619_161983

/-- The number of apples Mike picked -/
def mike_apples : ℝ := 7.0

/-- The number of apples Keith ate -/
def keith_ate : ℝ := 6.0

/-- The number of apples left -/
def apples_left : ℝ := 4.0

/-- The number of apples Nancy picked -/
def nancy_apples : ℝ := 3.0

/-- Theorem stating that Nancy picked 3.0 apples -/
theorem nancy_picked_three_apples : 
  mike_apples + nancy_apples - keith_ate = apples_left :=
by sorry

end NUMINAMATH_CALUDE_nancy_picked_three_apples_l1619_161983


namespace NUMINAMATH_CALUDE_f_value_at_8pi_over_3_l1619_161901

def f (x : ℝ) : ℝ := sorry

theorem f_value_at_8pi_over_3 
  (h_even : ∀ x, f (-x) = f x)
  (h_periodic : ∀ x, f (x + π) = f x)
  (h_def : ∀ x, 0 ≤ x → x < π/2 → f x = Real.sqrt 3 * Real.tan x - 1) :
  f (8*π/3) = 2 := by sorry

end NUMINAMATH_CALUDE_f_value_at_8pi_over_3_l1619_161901


namespace NUMINAMATH_CALUDE_largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l1619_161917

theorem largest_integer_negative_quadratic : 
  ∀ n : ℤ, n^2 - 11*n + 24 < 0 → n ≤ 7 :=
by sorry

theorem seven_satisfies_inequality : 
  (7 : ℤ)^2 - 11*7 + 24 < 0 :=
by sorry

theorem eight_does_not_satisfy_inequality : 
  (8 : ℤ)^2 - 11*8 + 24 ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l1619_161917


namespace NUMINAMATH_CALUDE_marble_distribution_l1619_161972

def valid_divisors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (fun m => n % m = 0 ∧ m > 1 ∧ m < n ∧ n / m > 1)

theorem marble_distribution :
  (valid_divisors 420).card = 22 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l1619_161972


namespace NUMINAMATH_CALUDE_min_red_chips_l1619_161960

theorem min_red_chips (w b r : ℕ) : 
  b ≥ w / 3 →
  b ≤ r / 4 →
  w + b ≥ 75 →
  ∀ r' : ℕ, (∃ w' b' : ℕ, b' ≥ w' / 3 ∧ b' ≤ r' / 4 ∧ w' + b' ≥ 75) → r' ≥ 76 :=
by sorry

end NUMINAMATH_CALUDE_min_red_chips_l1619_161960


namespace NUMINAMATH_CALUDE_original_profit_percentage_l1619_161967

theorem original_profit_percentage 
  (cost_price : ℝ) 
  (original_selling_price : ℝ) 
  (h1 : original_selling_price > 0) 
  (h2 : cost_price > 0) 
  (h3 : (2 * original_selling_price - cost_price) / cost_price = 2.6) : 
  (original_selling_price - cost_price) / cost_price = 0.8 := by
sorry

end NUMINAMATH_CALUDE_original_profit_percentage_l1619_161967


namespace NUMINAMATH_CALUDE_rhombus_longest_diagonal_l1619_161921

/-- Given a rhombus with area 200 square units and diagonal ratio 4:3, 
    prove that the length of the longest diagonal is 40√3/3 -/
theorem rhombus_longest_diagonal (area : ℝ) (ratio : ℚ) (d1 d2 : ℝ) :
  area = 200 →
  ratio = 4 / 3 →
  d1 / d2 = ratio →
  area = (d1 * d2) / 2 →
  d1 > d2 →
  d1 = 40 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longest_diagonal_l1619_161921


namespace NUMINAMATH_CALUDE_triangle_properties_l1619_161956

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Given conditions
  (b + b * Real.cos A = Real.sqrt 3 * Real.sin B) →
  (a = Real.sqrt 21) →
  (b = 4) →
  -- Conclusions to prove
  (A = π / 3) ∧
  (1/2 * b * c * Real.sin A = 5 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1619_161956


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1619_161966

theorem sin_alpha_value (α β : Real) 
  (eq1 : 1 - Real.cos α - Real.cos β + Real.sin α * Real.cos β = 0)
  (eq2 : 1 + Real.cos α - Real.sin β + Real.sin α * Real.cos β = 0) :
  Real.sin α = (1 - Real.sqrt 10) / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1619_161966


namespace NUMINAMATH_CALUDE_odd_functions_sum_sufficient_not_necessary_l1619_161958

-- Define the concept of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_functions_sum_sufficient_not_necessary :
  (∀ f g : ℝ → ℝ, IsOdd f ∧ IsOdd g → IsOdd (f + g)) ∧
  (∃ f g : ℝ → ℝ, IsOdd (f + g) ∧ ¬(IsOdd f ∧ IsOdd g)) :=
sorry

end NUMINAMATH_CALUDE_odd_functions_sum_sufficient_not_necessary_l1619_161958


namespace NUMINAMATH_CALUDE_parabola_focal_chord_angle_l1619_161920

/-- Given a parabola y^2 = 2px and a focal chord AB of length 8p, 
    the angle of inclination θ of AB satisfies sin θ = ±1/2 -/
theorem parabola_focal_chord_angle (p : ℝ) (θ : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- parabola equation
  (8*p = 2*p / (Real.sin θ)^2) →  -- focal chord length formula
  (Real.sin θ = 1/2 ∨ Real.sin θ = -1/2) :=
sorry

end NUMINAMATH_CALUDE_parabola_focal_chord_angle_l1619_161920


namespace NUMINAMATH_CALUDE_running_days_calculation_l1619_161963

/-- 
Given:
- Peter runs 3 miles more than Andrew per day
- Andrew runs 2 miles per day
- Their total combined distance is 35 miles

Prove that they have been running for 5 days.
-/
theorem running_days_calculation (andrew_miles : ℕ) (peter_miles : ℕ) (total_miles : ℕ) (days : ℕ) :
  andrew_miles = 2 →
  peter_miles = andrew_miles + 3 →
  total_miles = 35 →
  days * (andrew_miles + peter_miles) = total_miles →
  days = 5 := by
  sorry

#check running_days_calculation

end NUMINAMATH_CALUDE_running_days_calculation_l1619_161963


namespace NUMINAMATH_CALUDE_second_attempt_score_l1619_161970

/-- Represents the score of a dart throw attempt -/
structure DartScore where
  score : ℕ
  darts : ℕ
  min_per_dart : ℕ
  max_per_dart : ℕ

/-- The relationship between three dart throw attempts -/
structure ThreeAttempts where
  first : DartScore
  second : DartScore
  third : DartScore
  second_twice_first : first.score * 2 = second.score
  third_1_5_second : second.score * 3 = third.score * 2

/-- The theorem stating the score of the second attempt -/
theorem second_attempt_score (attempts : ThreeAttempts) 
  (h1 : attempts.first.darts = 8)
  (h2 : attempts.second.darts = 8)
  (h3 : attempts.third.darts = 8)
  (h4 : attempts.first.min_per_dart = 3)
  (h5 : attempts.first.max_per_dart = 9)
  (h6 : attempts.second.min_per_dart = 3)
  (h7 : attempts.second.max_per_dart = 9)
  (h8 : attempts.third.min_per_dart = 3)
  (h9 : attempts.third.max_per_dart = 9)
  : attempts.second.score = 48 := by
  sorry

end NUMINAMATH_CALUDE_second_attempt_score_l1619_161970


namespace NUMINAMATH_CALUDE_yellow_candy_bounds_l1619_161907

/-- Represents the state of the candy game -/
structure CandyGame where
  total : ℕ
  yellow : ℕ
  colors : ℕ
  yi_turn : Bool

/-- Defines the rules of the candy game -/
def valid_game (game : CandyGame) : Prop :=
  game.total = 22 ∧
  game.colors = 4 ∧
  game.yellow ≤ game.total ∧
  ∀ other_color, other_color ≠ game.yellow → other_color < game.yellow

/-- Defines a valid move in the game -/
def valid_move (before after : CandyGame) : Prop :=
  (before.yi_turn ∧ 
    ((before.total ≥ 2 ∧ after.total = before.total - 2) ∨ 
     (before.total = 1 ∧ after.total = 0))) ∨
  (¬before.yi_turn ∧ 
    (after.total = before.total - before.colors + 1 ∨ after.total = 0))

/-- Defines the end state of the game -/
def game_end (game : CandyGame) : Prop :=
  game.total = 0

/-- Theorem stating the bounds on the number of yellow candies -/
theorem yellow_candy_bounds (initial : CandyGame) :
  valid_game initial →
  (∃ final : CandyGame, 
    game_end final ∧ 
    (∀ intermediate : CandyGame, valid_move initial intermediate → valid_move intermediate final)) →
  8 ≤ initial.yellow ∧ initial.yellow ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_yellow_candy_bounds_l1619_161907


namespace NUMINAMATH_CALUDE_quadratic_equality_l1619_161964

/-- A quadratic function f(x) = ax^2 + bx + 6 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 6

/-- Theorem: If f(-1) = f(3) for a quadratic function f(x) = ax^2 + bx + 6, then f(2) = 6 -/
theorem quadratic_equality (a b : ℝ) : f a b (-1) = f a b 3 → f a b 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equality_l1619_161964


namespace NUMINAMATH_CALUDE_discount_percentage_l1619_161943

theorem discount_percentage (num_tickets : ℕ) (price_per_ticket : ℚ) (total_spent : ℚ) : 
  num_tickets = 24 →
  price_per_ticket = 7 →
  total_spent = 84 →
  (1 - total_spent / (num_tickets * price_per_ticket)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l1619_161943


namespace NUMINAMATH_CALUDE_distinct_arrangements_eq_twelve_l1619_161933

/-- The number of distinct arrangements of a 4-letter word with one letter repeated twice -/
def distinct_arrangements : ℕ :=
  Nat.factorial 4 / Nat.factorial 2

/-- Theorem stating that the number of distinct arrangements is 12 -/
theorem distinct_arrangements_eq_twelve : distinct_arrangements = 12 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_eq_twelve_l1619_161933


namespace NUMINAMATH_CALUDE_work_completion_theorem_l1619_161959

theorem work_completion_theorem (days_group1 days_group2 : ℕ) 
  (men_group2 : ℕ) (total_work : ℕ) :
  days_group1 = 18 →
  days_group2 = 8 →
  men_group2 = 81 →
  total_work = men_group2 * days_group2 →
  ∃ men_group1 : ℕ, men_group1 * days_group1 = total_work ∧ men_group1 = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l1619_161959


namespace NUMINAMATH_CALUDE_number_division_problem_l1619_161969

theorem number_division_problem (x y : ℝ) 
  (h1 : (x - 5) / 7 = 7)
  (h2 : (x - 2) / y = 4) : 
  y = 13 := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l1619_161969


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l1619_161974

theorem arithmetic_evaluation : 68 + (108 * 3) + (29^2) - 310 - (6 * 9) = 869 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l1619_161974


namespace NUMINAMATH_CALUDE_parabola_equation_l1619_161915

/-- A parabola with vertex at the origin, focus on the y-axis, and directrix y = 3 has the equation x² = 12y -/
theorem parabola_equation (p : ℝ) (h1 : p > 0) (h2 : p / 2 = 3) :
  ∀ x y : ℝ, (x, y) ∈ {(x, y) | x^2 = 2 * p * y} ↔ x^2 = 12 * y := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1619_161915


namespace NUMINAMATH_CALUDE_area_eq_xy_l1619_161923

/-- A right-angled triangle with an inscribed circle. -/
structure RightTriangleWithIncircle where
  /-- The length of one segment of the hypotenuse. -/
  x : ℝ
  /-- The length of the other segment of the hypotenuse. -/
  y : ℝ
  /-- The radius of the inscribed circle. -/
  r : ℝ
  /-- x and y are positive -/
  x_pos : 0 < x
  y_pos : 0 < y
  /-- r is positive -/
  r_pos : 0 < r

/-- The area of a right-angled triangle with an inscribed circle. -/
def area (t : RightTriangleWithIncircle) : ℝ :=
  t.x * t.y

/-- Theorem: The area of a right-angled triangle with an inscribed circle
    touching the hypotenuse at a point dividing it into segments of lengths x and y
    is equal to x * y. -/
theorem area_eq_xy (t : RightTriangleWithIncircle) : area t = t.x * t.y := by
  sorry

end NUMINAMATH_CALUDE_area_eq_xy_l1619_161923


namespace NUMINAMATH_CALUDE_power_mod_five_l1619_161926

theorem power_mod_five : 3^19 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_five_l1619_161926
