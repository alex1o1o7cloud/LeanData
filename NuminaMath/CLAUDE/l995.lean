import Mathlib

namespace iphone_sales_l995_99553

theorem iphone_sales (iphone_price : ℝ) (ipad_count : ℕ) (ipad_price : ℝ)
                     (appletv_count : ℕ) (appletv_price : ℝ) (average_price : ℝ) :
  iphone_price = 1000 →
  ipad_count = 20 →
  ipad_price = 900 →
  appletv_count = 80 →
  appletv_price = 200 →
  average_price = 670 →
  ∃ (iphone_count : ℕ),
    (iphone_count : ℝ) * iphone_price + (ipad_count : ℝ) * ipad_price + (appletv_count : ℝ) * appletv_price =
    average_price * ((iphone_count : ℝ) + (ipad_count : ℝ) + (appletv_count : ℝ)) ∧
    iphone_count = 100 :=
by sorry

end iphone_sales_l995_99553


namespace genuine_coin_remains_l995_99550

/-- Represents the type of a coin -/
inductive CoinType
| Genuine
| Fake

/-- Represents the state of the coin selection process -/
structure CoinState where
  total : Nat
  genuine : Nat
  fake : Nat
  moves : Nat

/-- The initial state of coins -/
def initialState : CoinState :=
  { total := 2022
  , genuine := 1012  -- More than half of 2022
  , fake := 1010     -- Less than half of 2022
  , moves := 0 }

/-- Simulates a single move in the coin selection process -/
def move (state : CoinState) : CoinState :=
  { state with
    total := state.total - 1
    moves := state.moves + 1
    genuine := state.genuine - 1  -- Worst case: remove a genuine coin
  }

/-- Applies the move function n times -/
def applyMoves (n : Nat) (state : CoinState) : CoinState :=
  match n with
  | 0 => state
  | n + 1 => move (applyMoves n state)

theorem genuine_coin_remains : 
  (applyMoves 2021 initialState).genuine > 0 := by
  sorry

#check genuine_coin_remains

end genuine_coin_remains_l995_99550


namespace scientific_notation_of_chip_size_l995_99580

theorem scientific_notation_of_chip_size :
  ∃ (a : ℝ) (n : ℤ), 0.000000014 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4 ∧ n = -8 := by
  sorry

end scientific_notation_of_chip_size_l995_99580


namespace valid_pairs_characterization_l995_99570

/-- A function that checks if a given pair (m, n) of natural numbers
    satisfies the condition that 2^m * 3^n + 1 is a perfect square. -/
def is_valid_pair (m n : ℕ) : Prop :=
  ∃ x : ℕ, 2^m * 3^n + 1 = x^2

/-- The set of all valid pairs (m, n) that satisfy the condition. -/
def valid_pairs : Set (ℕ × ℕ) :=
  {p | is_valid_pair p.1 p.2}

/-- The theorem stating that the only valid pairs are (3, 1), (4, 1), and (5, 2). -/
theorem valid_pairs_characterization :
  valid_pairs = {(3, 1), (4, 1), (5, 2)} := by
  sorry


end valid_pairs_characterization_l995_99570


namespace range_of_m_for_exponential_equation_l995_99591

/-- The range of m for which the equation 9^(-x^x) = 4 * 3^(-x^x) + m has a real solution for x -/
theorem range_of_m_for_exponential_equation :
  ∀ m : ℝ, (∃ x : ℝ, (9 : ℝ)^(-x^x) = 4 * (3 : ℝ)^(-x^x) + m) ↔ -3 ≤ m ∧ m < 0 := by
  sorry

end range_of_m_for_exponential_equation_l995_99591


namespace knight_position_proof_l995_99556

/-- The total number of people in the line -/
def total_people : ℕ := 2022

/-- The position of the knight from the left -/
def knight_position : ℕ := 48

/-- The ratio of liars to the right compared to the left for each person (except the ends) -/
def liar_ratio : ℕ := 42

theorem knight_position_proof :
  ∀ k : ℕ, 
  1 < k ∧ k < total_people →
  (total_people - k = liar_ratio * (k - 1)) ↔ 
  k = knight_position :=
sorry

end knight_position_proof_l995_99556


namespace approximate_root_exists_l995_99588

def f (x : ℝ) := x^3 + x^2 - 2*x - 2

theorem approximate_root_exists :
  ∃ (r : ℝ), r ∈ Set.Icc 1.375 1.4375 ∧ f r = 0 ∧ 
  ∀ (x : ℝ), x ∈ Set.Icc 1.375 1.4375 → |x - 1.42| ≤ 0.05 := by
  sorry

#check approximate_root_exists

end approximate_root_exists_l995_99588


namespace divisibility_property_l995_99507

theorem divisibility_property (a b c d : ℤ) (h1 : a ≠ b) (h2 : (a - b) ∣ (a * c + b * d)) :
  (a - b) ∣ (a * d + b * c) := by sorry

end divisibility_property_l995_99507


namespace difference_of_squares_330_270_l995_99583

theorem difference_of_squares_330_270 : 330^2 - 270^2 = 36000 := by
  sorry

end difference_of_squares_330_270_l995_99583


namespace expected_original_positions_value_l995_99525

/-- Represents the number of balls in the circle -/
def num_balls : ℕ := 7

/-- Represents the probability of a ball being in its original position after two transpositions -/
def prob_original_position : ℚ := 9 / 14

/-- The expected number of balls in their original positions after two transpositions -/
def expected_original_positions : ℚ := num_balls * prob_original_position

theorem expected_original_positions_value :
  expected_original_positions = 4.5 := by sorry

end expected_original_positions_value_l995_99525


namespace geometric_progression_terms_l995_99520

/-- A finite geometric progression with first term 3, second term 12, and last term 3072 has 6 terms -/
theorem geometric_progression_terms : 
  ∀ (b : ℕ → ℝ), 
    b 1 = 3 → 
    b 2 = 12 → 
    (∃ n : ℕ, n > 2 ∧ b n = 3072 ∧ ∀ k : ℕ, 1 < k → k < n → b k / b (k-1) = b 2 / b 1) →
    ∃ n : ℕ, n = 6 ∧ b n = 3072 ∧ ∀ k : ℕ, 1 < k → k < n → b k / b (k-1) = b 2 / b 1 :=
by sorry


end geometric_progression_terms_l995_99520


namespace area_ratio_of_squares_l995_99549

/-- Given three squares A, B, and C with specific perimeters, 
    this theorem proves the ratio of areas of A to C. -/
theorem area_ratio_of_squares (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (pa : 4 * a = 16) -- perimeter of A is 16
  (pb : 4 * b = 40) -- perimeter of B is 40
  (pc : 4 * c = 120) -- perimeter of C is 120 (3 times B's perimeter)
  : (a * a) / (c * c) = 4 / 225 := by
  sorry

#check area_ratio_of_squares

end area_ratio_of_squares_l995_99549


namespace first_blend_cost_is_correct_l995_99567

/-- The cost of the first blend of coffee in dollars per pound -/
def first_blend_cost : ℝ := 9

/-- The cost of the second blend of coffee in dollars per pound -/
def second_blend_cost : ℝ := 8

/-- The total weight of the mixed blend in pounds -/
def total_blend_weight : ℝ := 20

/-- The selling price of the mixed blend in dollars per pound -/
def mixed_blend_price : ℝ := 8.4

/-- The weight of the first blend used in the mixture in pounds -/
def first_blend_weight : ℝ := 8

/-- Theorem stating that the cost of the first blend is correct given the conditions -/
theorem first_blend_cost_is_correct :
  first_blend_cost * first_blend_weight + 
  second_blend_cost * (total_blend_weight - first_blend_weight) = 
  mixed_blend_price * total_blend_weight := by
  sorry

end first_blend_cost_is_correct_l995_99567


namespace sum_expression_l995_99527

theorem sum_expression (x y z k : ℝ) (h1 : y = 3 * x) (h2 : z = k * y) :
  x + y + z = (4 + 3 * k) * x := by
  sorry

end sum_expression_l995_99527


namespace complex_vector_properties_l995_99593

/-- Represents a complex number in the Cartesian plane -/
structure ComplexVector where
  x : ℝ
  y : ℝ

/-- Given an imaginary number z, returns its corresponding vector representation -/
noncomputable def z_to_vector (z : ℂ) : ComplexVector :=
  { x := z.re, y := z.im }

/-- Given an imaginary number z, returns the vector representation of its conjugate -/
noncomputable def conj_to_vector (z : ℂ) : ComplexVector :=
  { x := z.re, y := -z.im }

/-- Given an imaginary number z, returns the vector representation of its reciprocal -/
noncomputable def recip_to_vector (z : ℂ) : ComplexVector :=
  { x := z.re / (z.re^2 + z.im^2), y := -z.im / (z.re^2 + z.im^2) }

/-- Checks if three points are collinear given two vectors -/
def are_collinear (v1 v2 : ComplexVector) : Prop :=
  v1.x * v2.y = v1.y * v2.x

/-- Adds two ComplexVectors -/
def add_vectors (v1 v2 : ComplexVector) : ComplexVector :=
  { x := v1.x + v2.x, y := v1.y + v2.y }

theorem complex_vector_properties (z : ℂ) (h : z^3 = 1) :
  let OA := z_to_vector z
  let OB := conj_to_vector z
  let OC := recip_to_vector z
  (are_collinear OB OC) ∧ (add_vectors OA OC = ComplexVector.mk (-1) 0) := by
  sorry

end complex_vector_properties_l995_99593


namespace midpoint_coordinate_sum_l995_99576

/-- Given that N(4,10) is the midpoint of CD and C(14,6), prove that the sum of D's coordinates is 8 -/
theorem midpoint_coordinate_sum (N C D : ℝ × ℝ) : 
  N = (4, 10) → 
  C = (14, 6) → 
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 8 := by
  sorry

end midpoint_coordinate_sum_l995_99576


namespace second_square_area_equal_l995_99523

/-- An isosceles right triangle with an inscribed square -/
structure IsoscelesRightTriangleWithSquare where
  /-- The length of a leg of the isosceles right triangle -/
  leg : ℝ
  /-- The side length of the inscribed square -/
  square_side : ℝ
  /-- The square is inscribed with two vertices on one leg, one on the hypotenuse, and one on the other leg -/
  inscribed : square_side > 0 ∧ square_side < leg
  /-- The area of the inscribed square is 625 cm² -/
  area_condition : square_side ^ 2 = 625

/-- The area of another inscribed square in the same triangle -/
def second_square_area (triangle : IsoscelesRightTriangleWithSquare) : ℝ :=
  triangle.square_side ^ 2

theorem second_square_area_equal (triangle : IsoscelesRightTriangleWithSquare) :
  second_square_area triangle = 625 := by
  sorry

end second_square_area_equal_l995_99523


namespace symmetric_point_in_first_quadrant_l995_99562

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def is_in_first_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Symmetry about the x-axis -/
def symmetric_about_x_axis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- The original point P -/
def P : Point :=
  ⟨2, -3⟩

theorem symmetric_point_in_first_quadrant :
  is_in_first_quadrant (symmetric_about_x_axis P) :=
by sorry

end symmetric_point_in_first_quadrant_l995_99562


namespace digit_one_more_frequent_than_zero_l995_99555

def concatenated_sequence (n : ℕ) : String :=
  String.join (List.map toString (List.range n))

def count_digit (s : String) (d : Char) : ℕ :=
  s.toList.filter (· = d) |>.length

theorem digit_one_more_frequent_than_zero (n : ℕ) :
  count_digit (concatenated_sequence n) '1' > count_digit (concatenated_sequence n) '0' :=
sorry

end digit_one_more_frequent_than_zero_l995_99555


namespace alternating_color_probability_l995_99545

/-- The probability of drawing 10 balls from a box containing 5 white and 5 black balls
    such that the colors alternate is equal to 1/126. -/
theorem alternating_color_probability (n : ℕ) (white_balls black_balls : ℕ) : 
  n = 10 → white_balls = 5 → black_balls = 5 →
  (Nat.choose n white_balls : ℚ)⁻¹ * 2 = 1 / 126 := by
  sorry

end alternating_color_probability_l995_99545


namespace calculation_proof_l995_99596

theorem calculation_proof : (0.0077 * 3.6) / (0.04 * 0.1 * 0.007) = 990 := by
  sorry

end calculation_proof_l995_99596


namespace angle_D_measure_l995_99579

/-- 
Given a geometric figure with angles A, B, C, D, and E, where:
1. The sum of angles A and B is 140 degrees
2. Angle C is equal to angle D
3. The sum of angles C, D, and E is 180 degrees

This theorem proves that the measure of angle D is 20 degrees.
-/
theorem angle_D_measure (A B C D E : ℝ) 
  (sum_AB : A + B = 140)
  (C_eq_D : C = D)
  (sum_CDE : C + D + E = 180) :
  D = 20 := by sorry

end angle_D_measure_l995_99579


namespace star_equality_implies_x_eight_l995_99566

/-- Binary operation ★ on ordered pairs of integers -/
def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b + d)

/-- Theorem stating that if (5,7) ★ (1,3) = (x,y) ★ (4,5), then x = 8 -/
theorem star_equality_implies_x_eight (x y : ℤ) :
  star 5 7 1 3 = star x y 4 5 → x = 8 := by
  sorry

end star_equality_implies_x_eight_l995_99566


namespace train_passing_time_l995_99544

theorem train_passing_time (slower_speed faster_speed : ℝ) (train_length : ℝ) : 
  slower_speed = 36 →
  faster_speed = 45 →
  train_length = 90.0072 →
  (train_length / ((slower_speed + faster_speed) * (1000 / 3600))) = 4 := by
  sorry

end train_passing_time_l995_99544


namespace craft_store_sales_l995_99541

theorem craft_store_sales (total_sales : ℕ) : 
  (total_sales / 3 : ℕ) + (total_sales / 4 : ℕ) + 15 = total_sales → 
  total_sales = 36 := by
  sorry

end craft_store_sales_l995_99541


namespace inverse_proportion_y_relationship_l995_99530

theorem inverse_proportion_y_relationship (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ > x₂ → x₂ > 0 → y₁ = -3 / x₁ → y₂ = -3 / x₂ → y₁ > y₂ := by
  sorry

end inverse_proportion_y_relationship_l995_99530


namespace sixty_degree_iff_arithmetic_progression_l995_99501

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = 180

/-- The property that the angles of a triangle are in arithmetic progression -/
def angles_in_arithmetic_progression (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B

/-- Theorem stating that B = 60° is necessary and sufficient for the angles to be in arithmetic progression -/
theorem sixty_degree_iff_arithmetic_progression (t : Triangle) :
  t.B = 60 ↔ angles_in_arithmetic_progression t := by
  sorry

end sixty_degree_iff_arithmetic_progression_l995_99501


namespace sam_seashells_l995_99577

/-- Given that Mary found 47 seashells and the total number of seashells
    found by Sam and Mary is 65, prove that Sam found 18 seashells. -/
theorem sam_seashells (mary_seashells : ℕ) (total_seashells : ℕ)
    (h1 : mary_seashells = 47)
    (h2 : total_seashells = 65) :
    total_seashells - mary_seashells = 18 := by
  sorry

end sam_seashells_l995_99577


namespace final_sum_after_operations_l995_99537

theorem final_sum_after_operations (x y D : ℝ) (h : x - y = D) :
  4 * ((x - 5) + (y - 5)) = 4 * (x + y) - 40 := by
  sorry

end final_sum_after_operations_l995_99537


namespace find_regular_working_hours_l995_99574

/-- Represents the problem of finding regular working hours per day --/
theorem find_regular_working_hours
  (working_days_per_week : ℕ)
  (regular_pay_rate : ℚ)
  (overtime_pay_rate : ℚ)
  (total_earnings : ℚ)
  (total_hours : ℕ)
  (h1 : working_days_per_week = 6)
  (h2 : regular_pay_rate = 21/10)
  (h3 : overtime_pay_rate = 42/10)
  (h4 : total_earnings = 525)
  (h5 : total_hours = 245) :
  ∃ (regular_hours_per_day : ℕ),
    regular_hours_per_day = 10 ∧
    regular_hours_per_day * working_days_per_week * 4 ≤ total_hours ∧
    regular_pay_rate * (regular_hours_per_day * working_days_per_week * 4) +
    overtime_pay_rate * (total_hours - regular_hours_per_day * working_days_per_week * 4) =
    total_earnings :=
by sorry

end find_regular_working_hours_l995_99574


namespace chromium_percentage_in_new_alloy_l995_99572

/-- The percentage of chromium in the new alloy formed by mixing two alloys -/
theorem chromium_percentage_in_new_alloy 
  (chromium_percentage1 : Real) 
  (chromium_percentage2 : Real)
  (weight1 : Real) 
  (weight2 : Real) 
  (h1 : chromium_percentage1 = 12 / 100)
  (h2 : chromium_percentage2 = 8 / 100)
  (h3 : weight1 = 15)
  (h4 : weight2 = 40) : 
  (chromium_percentage1 * weight1 + chromium_percentage2 * weight2) / (weight1 + weight2) = 1 / 11 := by
sorry

#eval (1 / 11 : Float) * 100 -- To show the approximate percentage

end chromium_percentage_in_new_alloy_l995_99572


namespace gcd_of_squares_gcd_130_215_310_131_216_309_l995_99571

theorem gcd_of_squares (a b c d e f : ℤ) : 
  Int.gcd (a^2 + b^2 + c^2) (d^2 + e^2 + f^2) = 
  Int.gcd ((d^2 + e^2 + f^2) : ℤ) (|((a - d) * (a + d) + (b - e) * (b + e) + (c - f) * (c + f))|) :=
by sorry

theorem gcd_130_215_310_131_216_309 : 
  Int.gcd (130^2 + 215^2 + 310^2) (131^2 + 216^2 + 309^2) = 1 :=
by sorry

end gcd_of_squares_gcd_130_215_310_131_216_309_l995_99571


namespace solution_set_and_inequality_l995_99585

-- Define the set T
def T : Set ℝ := {t | t > 1}

-- Define the function f(x) = |x-2| + |x-3|
def f (x : ℝ) : ℝ := |x - 2| + |x - 3|

theorem solution_set_and_inequality :
  -- Part 1: T is the set of all t for which |x-2|+|x-3| < t has a non-empty solution set
  (∀ t : ℝ, (∃ x : ℝ, f x < t) ↔ t ∈ T) ∧
  -- Part 2: For all a, b ∈ T, ab + 1 > a + b
  (∀ a b : ℝ, a ∈ T → b ∈ T → a * b + 1 > a + b) :=
by sorry

end solution_set_and_inequality_l995_99585


namespace tan_A_value_l995_99526

theorem tan_A_value (A : Real) (h1 : 0 < A ∧ A < π / 2) 
  (h2 : 4 * Real.sin A ^ 2 - 4 * Real.sin A * Real.cos A + Real.cos A ^ 2 = 0) : 
  Real.tan A = 1 / 2 := by
sorry

end tan_A_value_l995_99526


namespace fraction_simplification_l995_99531

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = (5 * Real.sqrt 2) / 28 := by
  sorry

end fraction_simplification_l995_99531


namespace zoo_cost_l995_99568

def goat_cost : ℕ := 400
def goat_count : ℕ := 3

def llama_count : ℕ := 2 * goat_count
def llama_cost : ℕ := goat_cost + (goat_cost / 2)

def kangaroo_count : ℕ := 3 * goat_count
def kangaroo_cost : ℕ := llama_cost - (llama_cost / 4)

def total_cost : ℕ := goat_cost * goat_count + llama_cost * llama_count + kangaroo_cost * kangaroo_count

theorem zoo_cost : total_cost = 8850 := by
  sorry

end zoo_cost_l995_99568


namespace division_chain_l995_99581

theorem division_chain : (132 / 6) / 2 = 11 := by sorry

end division_chain_l995_99581


namespace linda_coin_count_l995_99517

/-- Represents the number of coins Linda has initially and receives from her mother -/
structure CoinCounts where
  initial_dimes : Nat
  initial_quarters : Nat
  initial_nickels : Nat
  additional_dimes : Nat
  additional_quarters : Nat

/-- Calculates the total number of coins Linda has -/
def totalCoins (counts : CoinCounts) : Nat :=
  counts.initial_dimes + counts.initial_quarters + counts.initial_nickels +
  counts.additional_dimes + counts.additional_quarters +
  2 * counts.initial_nickels

theorem linda_coin_count :
  let counts : CoinCounts := {
    initial_dimes := 2,
    initial_quarters := 6,
    initial_nickels := 5,
    additional_dimes := 2,
    additional_quarters := 10
  }
  totalCoins counts = 35 := by
  sorry

end linda_coin_count_l995_99517


namespace exponent_simplification_l995_99547

theorem exponent_simplification :
  ((-5^2)^4 * (-5)^11) / ((-5)^3) = 5^16 := by sorry

end exponent_simplification_l995_99547


namespace star_sum_five_l995_99564

def star (a b : ℕ) : ℕ := a^b + a*b

theorem star_sum_five (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) (h_star : star a b = 15) : 
  a + b = 5 := by sorry

end star_sum_five_l995_99564


namespace check_error_l995_99532

theorem check_error (x y : ℕ) : 
  x ≥ 10 ∧ x ≤ 99 ∧ y ≥ 10 ∧ y ≤ 99 →
  100 * y + x - (100 * x + y) = 2376 →
  y = 2 * x + 12 →
  x = 12 ∧ y = 36 := by
sorry

end check_error_l995_99532


namespace range_of_P_l995_99552

theorem range_of_P (x y : ℝ) (h : x^2/3 + y^2 = 1) : 
  2 ≤ |2*x + y - 4| + |4 - x - 2*y| ∧ |2*x + y - 4| + |4 - x - 2*y| ≤ 14 := by
  sorry

end range_of_P_l995_99552


namespace iron_conducts_electricity_l995_99573

-- Define the universe of discourse
variable (Object : Type)

-- Define the predicates
variable (is_metal : Object → Prop)
variable (can_conduct_electricity : Object → Prop)

-- Define iron as a constant
variable (iron : Object)

-- Theorem statement
theorem iron_conducts_electricity 
  (all_metals_conduct : ∀ x, is_metal x → can_conduct_electricity x) 
  (iron_is_metal : is_metal iron) : 
  can_conduct_electricity iron := by
  sorry

end iron_conducts_electricity_l995_99573


namespace determinant_trig_matrix_equals_one_l995_99515

theorem determinant_trig_matrix_equals_one (α β γ : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := λ i j ↦ 
    match i, j with
    | 0, 0 => Real.cos (α + γ) * Real.cos β
    | 0, 1 => Real.cos (α + γ) * Real.sin β
    | 0, 2 => -Real.sin (α + γ)
    | 1, 0 => -Real.sin β
    | 1, 1 => Real.cos β
    | 1, 2 => 0
    | 2, 0 => Real.sin (α + γ) * Real.cos β
    | 2, 1 => Real.sin (α + γ) * Real.sin β
    | 2, 2 => Real.cos (α + γ)
  Matrix.det M = 1 := by sorry

end determinant_trig_matrix_equals_one_l995_99515


namespace jump_height_to_touch_hoop_l995_99524

/-- Calculates the jump height needed to touch a basketball hoop -/
theorem jump_height_to_touch_hoop 
  (yao_height_ft : ℕ) 
  (yao_height_in : ℕ) 
  (hoop_height_ft : ℕ) 
  (inches_per_foot : ℕ) : 
  hoop_height_ft * inches_per_foot - (yao_height_ft * inches_per_foot + yao_height_in) = 31 :=
by
  sorry

#check jump_height_to_touch_hoop 7 5 10 12

end jump_height_to_touch_hoop_l995_99524


namespace digit_product_sum_28_l995_99538

/-- Represents a base-10 digit (0-9) -/
def Digit := Fin 10

/-- Represents a two-digit number -/
def TwoDigitNumber := Fin 100

/-- Converts two digits to a two-digit number -/
def toTwoDigitNumber (a b : Digit) : TwoDigitNumber :=
  ⟨a.val * 10 + b.val, by sorry⟩

/-- Converts a digit to a three-digit number where all digits are the same -/
def toThreeDigitSameNumber (e : Digit) : Nat :=
  e.val * 100 + e.val * 10 + e.val

theorem digit_product_sum_28 
  (A B C D E : Digit) 
  (h_unique : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E)
  (h_product : (toTwoDigitNumber A B).val * (toTwoDigitNumber C D).val = toThreeDigitSameNumber E) :
  A.val + B.val + C.val + D.val + E.val = 28 := by
  sorry

end digit_product_sum_28_l995_99538


namespace triangle_ratio_theorem_l995_99569

/-- Given a triangle ABC with side lengths a, b, c, altitudes ha, hb, hc, and circumradius R,
    the ratio of the sum of pairwise products of side lengths to the sum of altitudes
    is equal to the diameter of the circumscribed circle. -/
theorem triangle_ratio_theorem (a b c ha hb hc R : ℝ) :
  a > 0 → b > 0 → c > 0 → ha > 0 → hb > 0 → hc > 0 → R > 0 →
  (a * b + b * c + a * c) / (ha + hb + hc) = 2 * R := by
  sorry

end triangle_ratio_theorem_l995_99569


namespace square_root_condition_l995_99519

theorem square_root_condition (x : ℝ) : 
  Real.sqrt ((x - 1)^2) = x - 1 → x ≥ 1 := by
  sorry

end square_root_condition_l995_99519


namespace isosceles_right_triangle_hypotenuse_l995_99518

/-- An isosceles right triangle with perimeter 4 + 4√2 has a hypotenuse of length 4. -/
theorem isosceles_right_triangle_hypotenuse (a c : ℝ) : 
  a > 0 → -- Side length is positive
  c > 0 → -- Hypotenuse length is positive
  2 * a + c = 4 + 4 * Real.sqrt 2 → -- Perimeter condition
  c = a * Real.sqrt 2 → -- Isosceles right triangle condition
  c = 4 := by
sorry


end isosceles_right_triangle_hypotenuse_l995_99518


namespace solution_satisfies_system_l995_99560

theorem solution_satisfies_system :
  let x : ℝ := 0
  let y : ℝ := 6
  let z : ℝ := 7
  let u : ℝ := 3
  let v : ℝ := -1
  (x - y + z = 1) ∧
  (y - z + u = 2) ∧
  (z - u + v = 3) ∧
  (u - v + x = 4) ∧
  (v - x + y = 5) := by
  sorry

end solution_satisfies_system_l995_99560


namespace convergence_iff_cauchy_l995_99506

/-- A sequence of real numbers -/
def RealSequence := ℕ → ℝ

/-- Convergence of a sequence -/
def converges (x : RealSequence) : Prop :=
  ∃ (l : ℝ), ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - l| < ε

/-- Cauchy criterion for a sequence -/
def is_cauchy (x : RealSequence) : Prop :=
  ∀ ε > 0, ∃ N, ∀ m n, m ≥ N → n ≥ N → |x m - x n| < ε

/-- Theorem: A sequence of real numbers converges if and only if it satisfies the Cauchy criterion -/
theorem convergence_iff_cauchy (x : RealSequence) :
  converges x ↔ is_cauchy x :=
sorry

end convergence_iff_cauchy_l995_99506


namespace two_numbers_product_sum_l995_99510

theorem two_numbers_product_sum (x y : ℕ) : 
  x ∈ Finset.range 38 ∧ 
  y ∈ Finset.range 38 ∧ 
  x ≠ y ∧
  (Finset.sum (Finset.range 38) id) - x - y = x * y ∧
  y - x = 10 := by
  sorry

end two_numbers_product_sum_l995_99510


namespace distinct_positions_selection_l995_99543

theorem distinct_positions_selection (n : ℕ) (k : ℕ) (ways : ℕ) : 
  n = 12 → k = 2 → ways = 132 → ways = n * (n - 1) :=
by sorry

end distinct_positions_selection_l995_99543


namespace second_boy_marbles_l995_99516

-- Define the number of marbles for each boy as functions of x
def boy1_marbles (x : ℚ) : ℚ := 4 * x + 2
def boy2_marbles (x : ℚ) : ℚ := 3 * x - 1
def boy3_marbles (x : ℚ) : ℚ := 5 * x + 3

-- Define the total number of marbles
def total_marbles : ℚ := 128

-- Theorem statement
theorem second_boy_marbles :
  ∃ x : ℚ, 
    boy1_marbles x + boy2_marbles x + boy3_marbles x = total_marbles ∧
    boy2_marbles x = 30 := by
  sorry

end second_boy_marbles_l995_99516


namespace opposite_of_negative_two_l995_99586

theorem opposite_of_negative_two :
  ∃ x : ℝ, (x + (-2) = 0 ∧ x = 2) :=
by
  sorry

end opposite_of_negative_two_l995_99586


namespace train_length_l995_99542

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (h1 : speed_kmh = 144) (h2 : time_sec = 20) :
  speed_kmh * (1000 / 3600) * time_sec = 800 := by
  sorry

end train_length_l995_99542


namespace no_real_roots_l995_99597

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 5) - Real.sqrt (x - 2) + 2 = 0 := by
  sorry

end no_real_roots_l995_99597


namespace largest_integer_with_remainder_l995_99584

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 85 ∧ n % 9 = 3 ∧ ∀ m : ℕ, m < 85 ∧ m % 9 = 3 → m ≤ n :=
by sorry

end largest_integer_with_remainder_l995_99584


namespace f_even_iff_a_zero_l995_99534

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The quadratic function f(x) = x^2 + ax + 1 -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + 1

/-- Theorem: f is an even function if and only if a = 0 -/
theorem f_even_iff_a_zero (a : ℝ) :
  IsEven (f a) ↔ a = 0 := by
  sorry

end f_even_iff_a_zero_l995_99534


namespace symmetry_coordinates_l995_99582

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetric_x (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

theorem symmetry_coordinates :
  let A : Point := ⟨1, 2⟩
  let A' : Point := ⟨1, -2⟩
  symmetric_x A A' :=
by sorry

end symmetry_coordinates_l995_99582


namespace hua_method_uses_golden_ratio_l995_99589

/-- The optimal selection method popularized by Hua Luogeng -/
structure OptimalSelectionMethod where
  author : String
  concept : String

/-- Definition of Hua Luogeng's optimal selection method -/
def huaMethod : OptimalSelectionMethod :=
  { author := "Hua Luogeng"
  , concept := "golden ratio" }

/-- Theorem stating that Hua Luogeng's optimal selection method uses the golden ratio -/
theorem hua_method_uses_golden_ratio :
  huaMethod.concept = "golden ratio" := by
  sorry

end hua_method_uses_golden_ratio_l995_99589


namespace park_to_grocery_distance_l995_99599

/-- The distance from Talia's house to the park, in miles -/
def distance_house_to_park : ℝ := 5

/-- The distance from Talia's house to the grocery store, in miles -/
def distance_house_to_grocery : ℝ := 8

/-- The total distance Talia drives, in miles -/
def total_distance : ℝ := 16

/-- The distance from the park to the grocery store, in miles -/
def distance_park_to_grocery : ℝ := total_distance - distance_house_to_park - distance_house_to_grocery

theorem park_to_grocery_distance :
  distance_park_to_grocery = 3 := by sorry

end park_to_grocery_distance_l995_99599


namespace distance_from_origin_l995_99536

theorem distance_from_origin (x y : ℝ) (h1 : y = 15) (h2 : x > 3)
  (h3 : Real.sqrt ((x - 3)^2 + (y - 8)^2) = 11) :
  Real.sqrt (x^2 + y^2) = Real.sqrt 306 := by sorry

end distance_from_origin_l995_99536


namespace train_length_l995_99558

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 8 → ∃ length : ℝ, 
  (length ≥ 133.36 ∧ length ≤ 133.37) ∧ length = speed * time * (1000 / 3600) := by
  sorry

end train_length_l995_99558


namespace total_earnings_is_4350_l995_99551

/-- Given investment ratios and return ratios for three investors a, b, and c,
    calculates their total earnings. -/
def total_earnings (invest_a invest_b invest_c : ℚ)
                   (return_a return_b return_c : ℚ)
                   (diff_b_a : ℚ) : ℚ :=
  let earnings_a := invest_a * return_a
  let earnings_b := invest_b * return_b
  let earnings_c := invest_c * return_c
  earnings_a + earnings_b + earnings_c

/-- Theorem stating that under given conditions, the total earnings are 4350. -/
theorem total_earnings_is_4350 :
  ∃ (x y : ℚ),
    let invest_a := 3 * x
    let invest_b := 4 * x
    let invest_c := 5 * x
    let return_a := 6 * y
    let return_b := 5 * y
    let return_c := 4 * y
    invest_b * return_b - invest_a * return_a = 150 ∧
    total_earnings invest_a invest_b invest_c return_a return_b return_c 150 = 4350 :=
by
  sorry

#check total_earnings_is_4350

end total_earnings_is_4350_l995_99551


namespace james_game_preparation_time_l995_99565

def time_before_main_game (download_time install_time update_time account_time 
  internet_issues_time discussion_time tutorial_video_time in_game_tutorial_time : ℕ) : ℕ :=
  download_time + install_time + update_time + account_time + internet_issues_time + 
  discussion_time + tutorial_video_time + in_game_tutorial_time

theorem james_game_preparation_time :
  let download_time := 10
  let install_time := download_time / 2
  let update_time := download_time * 2
  let account_time := 5
  let internet_issues_time := 15
  let discussion_time := 20
  let tutorial_video_time := 8
  let preparation_time := download_time + install_time + update_time + account_time + 
    internet_issues_time + discussion_time + tutorial_video_time
  let in_game_tutorial_time := preparation_time * 3
  time_before_main_game download_time install_time update_time account_time 
    internet_issues_time discussion_time tutorial_video_time in_game_tutorial_time = 332 := by
  sorry

end james_game_preparation_time_l995_99565


namespace box_2_neg2_3_l995_99529

def box (a b c : ℤ) : ℚ := (a ^ (2 * b) : ℚ) - (b ^ (2 * c) : ℚ) + (c ^ (2 * a) : ℚ)

theorem box_2_neg2_3 : box 2 (-2) 3 = 273 / 16 := by sorry

end box_2_neg2_3_l995_99529


namespace cube_root_unity_sum_l995_99557

theorem cube_root_unity_sum (ω : ℂ) : 
  ω^3 = 1 → ω ≠ 1 → (2 - ω + ω^2)^3 + (2 + ω - ω^2)^3 = -2 := by
sorry

end cube_root_unity_sum_l995_99557


namespace specific_field_perimeter_l995_99503

/-- A rectangular field with specific properties -/
structure RectangularField where
  breadth : ℝ
  length : ℝ
  area : ℝ
  length_eq : length = breadth + 30
  area_eq : area = length * breadth

/-- The perimeter of a rectangular field -/
def perimeter (field : RectangularField) : ℝ :=
  2 * (field.length + field.breadth)

/-- Theorem stating the perimeter of the specific field is 540 meters -/
theorem specific_field_perimeter :
  ∃ (field : RectangularField), field.area = 18000 ∧ perimeter field = 540 := by
  sorry

end specific_field_perimeter_l995_99503


namespace billy_horses_count_l995_99505

/-- The number of horses Billy has -/
def num_horses : ℕ := 4

/-- The amount of oats (in pounds) each horse eats per feeding -/
def oats_per_feeding : ℕ := 4

/-- The number of feedings per day -/
def feedings_per_day : ℕ := 2

/-- The number of days Billy needs to feed his horses -/
def days_to_feed : ℕ := 3

/-- The total amount of oats (in pounds) Billy needs for all his horses for the given days -/
def total_oats_needed : ℕ := 96

theorem billy_horses_count : 
  num_horses * oats_per_feeding * feedings_per_day * days_to_feed = total_oats_needed :=
sorry

end billy_horses_count_l995_99505


namespace rice_mixture_cost_l995_99561

/-- The cost of the cheaper rice variety in Rs per kg -/
def cost_cheap : ℚ := 9/2

/-- The cost of the more expensive rice variety in Rs per kg -/
def cost_expensive : ℚ := 35/4

/-- The ratio of cheaper rice to more expensive rice in the mixture -/
def mixture_ratio : ℚ := 5/12

/-- The cost of the mixture per kg -/
def mixture_cost : ℚ := 23/4

theorem rice_mixture_cost :
  (cost_cheap * mixture_ratio + cost_expensive * 1) / (mixture_ratio + 1) = mixture_cost := by
  sorry

end rice_mixture_cost_l995_99561


namespace jorge_total_goals_l995_99587

theorem jorge_total_goals (last_season_goals this_season_goals : ℕ) 
  (h1 : last_season_goals = 156)
  (h2 : this_season_goals = 187) :
  last_season_goals + this_season_goals = 343 := by
  sorry

end jorge_total_goals_l995_99587


namespace range_of_a_l995_99592

/-- Given sets A and B, and the condition that A is not a subset of B, 
    prove that the range of values for a is (1, 5) -/
theorem range_of_a (a : ℝ) : 
  let A : Set ℝ := {x | 2*a < x ∧ x < a + 5}
  let B : Set ℝ := {x | x < 6}
  ¬(A ⊆ B) → a ∈ Set.Ioo 1 5 := by
  sorry


end range_of_a_l995_99592


namespace gift_wrapping_theorem_l995_99559

/-- Cagney's gift wrapping rate in gifts per second -/
def cagney_rate : ℚ := 1 / 45

/-- Lacey's gift wrapping rate in gifts per second -/
def lacey_rate : ℚ := 1 / 60

/-- Total time available in seconds -/
def total_time : ℚ := 15 * 60

/-- The number of gifts that can be wrapped collectively -/
def total_gifts : ℕ := 35

theorem gift_wrapping_theorem :
  (cagney_rate + lacey_rate) * total_time = total_gifts := by
  sorry

end gift_wrapping_theorem_l995_99559


namespace quadratic_two_distinct_roots_l995_99578

theorem quadratic_two_distinct_roots (a b c : ℝ) (h : 2016 + a^2 + a*c < a*b) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + b*x₁ + c = 0 ∧ a*x₂^2 + b*x₂ + c = 0 :=
sorry

end quadratic_two_distinct_roots_l995_99578


namespace absolute_value_equation_solution_l995_99504

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 5| = 3 * x + 1 :=
by
  -- Proof goes here
  sorry

end absolute_value_equation_solution_l995_99504


namespace ticket_price_difference_l995_99546

def total_cost : ℝ := 77
def adult_ticket_cost : ℝ := 19
def num_adults : ℕ := 2
def num_children : ℕ := 3

theorem ticket_price_difference : 
  ∃ (child_ticket_cost : ℝ),
    total_cost = num_adults * adult_ticket_cost + num_children * child_ticket_cost ∧
    adult_ticket_cost - child_ticket_cost = 6 :=
by sorry

end ticket_price_difference_l995_99546


namespace quadratic_roots_sum_l995_99535

theorem quadratic_roots_sum (k₁ k₂ : ℝ) : 
  36 * k₁^2 - 200 * k₁ + 49 = 0 →
  36 * k₂^2 - 200 * k₂ + 49 = 0 →
  k₁ / k₂ + k₂ / k₁ = 6.25 := by
  sorry

end quadratic_roots_sum_l995_99535


namespace population_growth_proof_l995_99511

/-- The annual growth rate of the population -/
def annual_growth_rate : ℝ := 0.1

/-- The population after 2 years -/
def population_after_2_years : ℕ := 15730

/-- The initial population of the town -/
def initial_population : ℕ := 13000

theorem population_growth_proof :
  (1 + annual_growth_rate) * (1 + annual_growth_rate) * initial_population = population_after_2_years := by
  sorry

end population_growth_proof_l995_99511


namespace complex_fraction_simplification_l995_99508

theorem complex_fraction_simplification :
  ∀ (i : ℂ), i^2 = -1 →
  (5 + 7*i) / (3 - 4*i) = (43 : ℚ)/25 + (41 : ℚ)/25 * i := by
  sorry

end complex_fraction_simplification_l995_99508


namespace fifth_power_sum_l995_99514

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 4)
  (h2 : a * x^2 + b * y^2 = 9)
  (h3 : a * x^3 + b * y^3 = 23)
  (h4 : a * x^4 + b * y^4 = 54) :
  a * x^5 + b * y^5 = 470 := by
  sorry


end fifth_power_sum_l995_99514


namespace determinant_equals_x_squared_plus_y_squared_l995_99595

theorem determinant_equals_x_squared_plus_y_squared (x y : ℝ) : 
  Matrix.det !![1, x, y; 1, x - y, y; 1, x, y - x] = x^2 + y^2 := by
  sorry

end determinant_equals_x_squared_plus_y_squared_l995_99595


namespace sams_sitting_fee_is_correct_l995_99509

/-- The one-time sitting fee for Sam's Picture Emporium -/
def sams_sitting_fee : ℝ := 140

/-- The price per sheet for John's Photo World -/
def johns_price_per_sheet : ℝ := 2.75

/-- The one-time sitting fee for John's Photo World -/
def johns_sitting_fee : ℝ := 125

/-- The price per sheet for Sam's Picture Emporium -/
def sams_price_per_sheet : ℝ := 1.50

/-- The number of sheets for which the total price is the same -/
def num_sheets : ℕ := 12

theorem sams_sitting_fee_is_correct :
  johns_price_per_sheet * num_sheets + johns_sitting_fee =
  sams_price_per_sheet * num_sheets + sams_sitting_fee :=
by
  sorry

#check sams_sitting_fee_is_correct

end sams_sitting_fee_is_correct_l995_99509


namespace tobys_change_is_seven_l995_99500

/-- Represents the dining scenario and calculates Toby's change --/
def tobys_change (cheeseburger_price : ℚ) (milkshake_price : ℚ) (coke_price : ℚ) 
                 (fries_price : ℚ) (cookie_price : ℚ) (tax : ℚ) 
                 (toby_initial_money : ℚ) : ℚ :=
  let total_cost := 2 * cheeseburger_price + milkshake_price + coke_price + 
                    fries_price + 3 * cookie_price + tax
  let toby_share := total_cost / 2
  toby_initial_money - toby_share

/-- Theorem stating that Toby's change is $7.00 --/
theorem tobys_change_is_seven : 
  tobys_change 3.65 2 1 4 0.5 0.2 15 = 7 := by
  sorry

end tobys_change_is_seven_l995_99500


namespace polynomial_divisibility_l995_99502

/-- For any natural number n, the polynomial 
    x^(2n) - n^2 * x^(n+1) + 2(n^2 - 1) * x^n + 1 - n^2 * x^(n-1) 
    is divisible by (x-1)^3. -/
theorem polynomial_divisibility (n : ℕ) : 
  ∃ q : Polynomial ℚ, (X : Polynomial ℚ)^(2*n) - n^2 * X^(n+1) + 2*(n^2 - 1) * X^n + 1 - n^2 * X^(n-1) = 
    (X - 1)^3 * q := by
  sorry

end polynomial_divisibility_l995_99502


namespace preimage_of_4_3_l995_99548

/-- The mapping f from R² to R² -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2*p.2, 2*p.1 - p.2)

/-- Theorem: The pre-image of (4,3) under the mapping f is (2,1) -/
theorem preimage_of_4_3 :
  f (2, 1) = (4, 3) := by
  sorry

end preimage_of_4_3_l995_99548


namespace increase_by_percentage_l995_99513

/-- Prove that increasing 500 by 30% results in 650. -/
theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 500 → percentage = 30 → result = initial * (1 + percentage / 100) → result = 650 := by
  sorry

end increase_by_percentage_l995_99513


namespace jenga_initial_blocks_jenga_game_proof_l995_99598

theorem jenga_initial_blocks (players : ℕ) (complete_rounds : ℕ) (blocks_removed_last_round : ℕ) (blocks_remaining : ℕ) : ℕ :=
  let blocks_removed_complete_rounds := players * complete_rounds
  let total_blocks_removed := blocks_removed_complete_rounds + blocks_removed_last_round
  let initial_blocks := total_blocks_removed + blocks_remaining
  initial_blocks

theorem jenga_game_proof :
  jenga_initial_blocks 5 5 1 28 = 54 := by
  sorry

end jenga_initial_blocks_jenga_game_proof_l995_99598


namespace triangle_relationships_l995_99522

/-- Given a triangle with sides a, b, c, circumradius R, inradius r, and semiperimeter p,
    prove the following relationships. -/
theorem triangle_relationships 
  (a b c R r p : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ r > 0 ∧ p > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_circumradius : R = (a * b * c) / (4 * (p * (p - a) * (p - b) * (p - c))^(1/2)))
  (h_inradius : r = (p * (p - a) * (p - b) * (p - c))^(1/2) / p) :
  (a * b * c = 4 * p * r * R) ∧ 
  (a * b + b * c + c * a = r^2 + p^2 + 4 * r * R) := by
  sorry


end triangle_relationships_l995_99522


namespace triangle_area_l995_99521

/-- Given a triangle with perimeter 32 cm and inradius 3.5 cm, its area is 56 cm² -/
theorem triangle_area (p : ℝ) (r : ℝ) (h1 : p = 32) (h2 : r = 3.5) :
  r * p / 2 = 56 := by
  sorry

end triangle_area_l995_99521


namespace quadratic_transform_coefficient_l995_99554

/-- Given a quadratic equation 7x - 3 = 2x², prove that when transformed
    to general form ax² + bx + c = 0 with c = 3, the coefficient of x (b) is -7 -/
theorem quadratic_transform_coefficient (x : ℝ) : 
  (7 * x - 3 = 2 * x^2) → 
  ∃ (a b : ℝ), (a * x^2 + b * x + 3 = 0) ∧ (b = -7) := by
sorry

end quadratic_transform_coefficient_l995_99554


namespace max_inscribed_circle_area_l995_99512

/-- Definition of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Left focus of the ellipse -/
def F1 : ℝ × ℝ := (-1, 0)

/-- Right focus of the ellipse -/
def F2 : ℝ × ℝ := (1, 0)

/-- A line passing through the right focus -/
def line_through_F2 (m : ℝ) (y : ℝ) : ℝ := m * y + 1

/-- Points of intersection between the line and the ellipse -/
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ y, ellipse (line_through_F2 m y) y ∧ p = (line_through_F2 m y, y)}

/-- Triangle formed by F1 and two intersection points -/
def triangle_F1PQ (m : ℝ) : Set (ℝ × ℝ) :=
  {F1} ∪ intersection_points m

/-- The inscribed circle of a triangle -/
def inscribed_circle (t : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry  -- Definition of inscribed circle

/-- The area of a circle -/
def circle_area (c : Set (ℝ × ℝ)) : ℝ :=
  sorry  -- Definition of circle area

/-- The theorem to be proved -/
theorem max_inscribed_circle_area :
  ∃ (m : ℝ), ∀ (n : ℝ),
    circle_area (inscribed_circle (triangle_F1PQ m)) ≥
    circle_area (inscribed_circle (triangle_F1PQ n)) ∧
    circle_area (inscribed_circle (triangle_F1PQ m)) = 9 * Real.pi / 16 :=
sorry

end max_inscribed_circle_area_l995_99512


namespace odd_prime_factor_form_l995_99575

theorem odd_prime_factor_form (p q : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) (hq : Nat.Prime q) (h_div : q ∣ 2^p - 1) :
  ∃ k : ℕ, q = 2*k*p + 1 := by
sorry

end odd_prime_factor_form_l995_99575


namespace balloon_sum_l995_99528

theorem balloon_sum (x y : ℝ) (hx : x = 7.5) (hy : y = 5.2) : x + y = 12.7 := by
  sorry

end balloon_sum_l995_99528


namespace train_length_calculation_l995_99563

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length_calculation (train_speed_kmh : ℝ) (time_to_cross : ℝ) (bridge_length : ℝ) :
  train_speed_kmh = 90 →
  time_to_cross = 9.679225661947045 →
  bridge_length = 132 →
  ∃ train_length : ℝ, abs (train_length - 109.98) < 0.01 ∧
    train_length = train_speed_kmh * (1000 / 3600) * time_to_cross - bridge_length :=
by sorry

end train_length_calculation_l995_99563


namespace intersection_of_P_and_M_l995_99533

-- Define the sets P and M
def P : Set ℝ := {y | ∃ x, y = x^2 - 6*x + 10}
def M : Set ℝ := {y | ∃ x, y = -x^2 + 2*x + 8}

-- State the theorem
theorem intersection_of_P_and_M : P ∩ M = {y | 1 ≤ y ∧ y ≤ 9} := by sorry

end intersection_of_P_and_M_l995_99533


namespace solution_set_f_geq_x_min_value_a_min_a_is_three_l995_99539

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| - |2*x + 3|

-- Theorem for part (I)
theorem solution_set_f_geq_x : 
  {x : ℝ | f x ≥ x} = {x : ℝ | x ≥ 4/5} := by sorry

-- Theorem for part (II)
theorem min_value_a (m : ℝ) (h : m > 0) :
  (∀ (x y : ℝ), f x ≤ m^y + a/m^y) → a ≥ 3 := by sorry

-- Theorem for the minimum value of a
theorem min_a_is_three :
  ∃ (a : ℝ), a = 3 ∧ 
  (∀ (m : ℝ), m > 0 → ∀ (x y : ℝ), f x ≤ m^y + a/m^y) ∧
  (∀ (a' : ℝ), (∀ (m : ℝ), m > 0 → ∀ (x y : ℝ), f x ≤ m^y + a'/m^y) → a' ≥ a) := by sorry

end solution_set_f_geq_x_min_value_a_min_a_is_three_l995_99539


namespace parallelogram_sides_l995_99590

-- Define the triangle ABC
structure Triangle :=
  (AB : ℝ)
  (BC : ℝ)

-- Define the parallelogram BKLM
structure Parallelogram :=
  (BM : ℝ)
  (BK : ℝ)

-- Define the problem conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.AB = 18 ∧ t.BC = 12

def parallelogram_conditions (t : Triangle) (p : Parallelogram) : Prop :=
  -- Area of BKLM is 4/9 of the area of ABC
  p.BM * p.BK = (4/9) * (1/2) * t.AB * t.BC

-- Theorem statement
theorem parallelogram_sides (t : Triangle) (p : Parallelogram) :
  triangle_conditions t →
  parallelogram_conditions t p →
  ((p.BM = 8 ∧ p.BK = 6) ∨ (p.BM = 4 ∧ p.BK = 12)) :=
by sorry

end parallelogram_sides_l995_99590


namespace base_7_to_10_23456_l995_99594

def base_7_to_10 (d₁ d₂ d₃ d₄ d₅ : ℕ) : ℕ :=
  d₁ * 7^4 + d₂ * 7^3 + d₃ * 7^2 + d₄ * 7^1 + d₅ * 7^0

theorem base_7_to_10_23456 :
  base_7_to_10 2 3 4 5 6 = 6068 := by sorry

end base_7_to_10_23456_l995_99594


namespace book_pages_total_l995_99540

/-- A book with 5 chapters, each containing 111 pages, has a total of 555 pages. -/
theorem book_pages_total (num_chapters : ℕ) (pages_per_chapter : ℕ) :
  num_chapters = 5 → pages_per_chapter = 111 → num_chapters * pages_per_chapter = 555 := by
  sorry

end book_pages_total_l995_99540
