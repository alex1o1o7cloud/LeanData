import Mathlib

namespace shaded_area_is_700_l1214_121492

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The vertices of the large square -/
def squareVertices : List Point := [
  ⟨0, 0⟩, ⟨40, 0⟩, ⟨40, 40⟩, ⟨0, 40⟩
]

/-- The vertices of the shaded polygon -/
def shadedVertices : List Point := [
  ⟨0, 0⟩, ⟨10, 0⟩, ⟨40, 30⟩, ⟨40, 40⟩, ⟨30, 40⟩, ⟨0, 10⟩
]

/-- The side length of the large square -/
def squareSideLength : ℝ := 40

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)|

/-- Calculate the area of the shaded region -/
def shadedArea : ℝ :=
  squareSideLength ^ 2 -
  (triangleArea ⟨10, 0⟩ ⟨40, 0⟩ ⟨40, 30⟩ +
   triangleArea ⟨0, 10⟩ ⟨30, 40⟩ ⟨0, 40⟩)

/-- Theorem: The area of the shaded region is 700 square units -/
theorem shaded_area_is_700 : shadedArea = 700 := by
  sorry

end shaded_area_is_700_l1214_121492


namespace E_opposite_Z_l1214_121486

/-- Represents a face of the cube -/
inductive Face : Type
| A : Face
| B : Face
| C : Face
| D : Face
| E : Face
| Z : Face

/-- Represents the net of the cube before folding -/
structure CubeNet :=
(faces : List Face)
(can_fold_to_cube : Bool)

/-- Represents the folded cube -/
structure Cube :=
(net : CubeNet)
(opposite_faces : Face → Face)

/-- The theorem stating that E is opposite to Z in the folded cube -/
theorem E_opposite_Z (net : CubeNet) (cube : Cube) :
  net.can_fold_to_cube = true →
  cube.net = net →
  cube.opposite_faces Face.Z = Face.E :=
sorry

end E_opposite_Z_l1214_121486


namespace right_triangle_inequality_l1214_121419

theorem right_triangle_inequality (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) 
  (h_a_nonneg : a ≥ 0) (h_b_nonneg : b ≥ 0) (h_c_pos : c > 0) : 
  c ≥ (Real.sqrt 2 / 2) * (a + b) := by
  sorry

end right_triangle_inequality_l1214_121419


namespace balki_cereal_boxes_l1214_121487

theorem balki_cereal_boxes : 
  let total_raisins : ℕ := 437
  let box1_raisins : ℕ := 72
  let box2_raisins : ℕ := 74
  let other_boxes_raisins : ℕ := 97
  let num_other_boxes : ℕ := (total_raisins - box1_raisins - box2_raisins) / other_boxes_raisins
  let total_boxes : ℕ := 2 + num_other_boxes
  total_boxes = 5 ∧ 
  box1_raisins + box2_raisins + num_other_boxes * other_boxes_raisins = total_raisins :=
by sorry

end balki_cereal_boxes_l1214_121487


namespace mowing_time_ab_l1214_121437

/-- The time (in days) taken by a and b together to mow the field -/
def time_ab : ℝ := 28

/-- The time (in days) taken by a, b, and c together to mow the field -/
def time_abc : ℝ := 21

/-- The time (in days) taken by c alone to mow the field -/
def time_c : ℝ := 84

/-- Theorem stating that the time taken by a and b to mow the field together is 28 days -/
theorem mowing_time_ab :
  time_ab = 28 ∧
  (1 / time_ab + 1 / time_c = 1 / time_abc) :=
sorry

end mowing_time_ab_l1214_121437


namespace division_sum_dividend_l1214_121484

theorem division_sum_dividend (quotient divisor remainder : ℕ) : 
  quotient = 40 → divisor = 72 → remainder = 64 → 
  (divisor * quotient) + remainder = 2944 := by
sorry

end division_sum_dividend_l1214_121484


namespace penny_drawing_probability_l1214_121496

/-- The number of shiny pennies in the box -/
def shiny_pennies : ℕ := 3

/-- The number of dull pennies in the box -/
def dull_pennies : ℕ := 4

/-- The total number of pennies in the box -/
def total_pennies : ℕ := shiny_pennies + dull_pennies

/-- The probability of drawing more than four pennies until the third shiny penny appears -/
def prob_more_than_four_draws : ℚ := 31 / 35

theorem penny_drawing_probability :
  prob_more_than_four_draws = 31 / 35 :=
sorry

end penny_drawing_probability_l1214_121496


namespace lower_bound_of_set_A_l1214_121480

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def set_A : Set ℕ := {n : ℕ | is_prime n ∧ n ≥ 17 ∧ n ≤ 36}

theorem lower_bound_of_set_A :
  (∃ (max_A : ℕ), max_A ∈ set_A ∧ ∀ n ∈ set_A, n ≤ max_A) ∧
  (∃ (min_A : ℕ), min_A ∈ set_A ∧ ∀ n ∈ set_A, n ≥ min_A) ∧
  (∃ (max_A min_A : ℕ), max_A - min_A = 14) →
  (∃ (min_A : ℕ), min_A = 17 ∧ min_A ∈ set_A ∧ ∀ n ∈ set_A, n ≥ min_A) :=
by sorry

end lower_bound_of_set_A_l1214_121480


namespace midpoint_theorem_l1214_121488

/-- Given a line segment with midpoint (3, 1) and one endpoint (7, -4), 
    prove that the other endpoint is (-1, 6) -/
theorem midpoint_theorem :
  let midpoint : ℝ × ℝ := (3, 1)
  let endpoint1 : ℝ × ℝ := (7, -4)
  let endpoint2 : ℝ × ℝ := (-1, 6)
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) :=
by sorry

end midpoint_theorem_l1214_121488


namespace dog_yelling_problem_l1214_121438

theorem dog_yelling_problem (obedient_yells stubborn_ratio : ℕ) : 
  obedient_yells = 12 →
  stubborn_ratio = 4 →
  obedient_yells + stubborn_ratio * obedient_yells = 60 := by
  sorry

end dog_yelling_problem_l1214_121438


namespace evaluate_expression_l1214_121433

theorem evaluate_expression : 12.543 - 3.219 + 1.002 = 10.326 := by
  sorry

end evaluate_expression_l1214_121433


namespace smallest_b_value_l1214_121431

theorem smallest_b_value (a b : ℤ) (h1 : 9 < a ∧ a < 21) (h2 : b < 31) (h3 : (a : ℚ) / b = 2/3) :
  ∃ (n : ℤ), n = 14 ∧ n < b ∧ ∀ m, m < b → m ≤ n :=
sorry

end smallest_b_value_l1214_121431


namespace juvy_garden_chives_l1214_121453

theorem juvy_garden_chives (total_rows : ℕ) (plants_per_row : ℕ) 
  (parsley_rows : ℕ) (rosemary_rows : ℕ) (mint_rows : ℕ) (thyme_rows : ℕ) :
  total_rows = 50 →
  plants_per_row = 15 →
  parsley_rows = 5 →
  rosemary_rows = 7 →
  mint_rows = 10 →
  thyme_rows = 12 →
  (total_rows - (parsley_rows + rosemary_rows + mint_rows + thyme_rows)) * plants_per_row = 240 :=
by
  sorry

end juvy_garden_chives_l1214_121453


namespace quadratic_root_product_l1214_121409

theorem quadratic_root_product (x₁ x₂ : ℝ) : 
  (x₁^2 - 4*x₁ - 2 = 0) → 
  (x₂^2 - 4*x₂ - 2 = 0) → 
  x₁ * x₂ = -2 := by
sorry

end quadratic_root_product_l1214_121409


namespace parking_lot_car_decrease_l1214_121414

theorem parking_lot_car_decrease (initial_cars : ℕ) (cars_out : ℕ) (cars_in : ℕ) : 
  initial_cars = 25 → cars_out = 18 → cars_in = 12 → 
  initial_cars - ((initial_cars - cars_out) + cars_in) = 6 :=
by
  sorry

end parking_lot_car_decrease_l1214_121414


namespace sqrt_two_irrational_and_between_one_and_three_l1214_121494

theorem sqrt_two_irrational_and_between_one_and_three :
  Irrational (Real.sqrt 2) ∧ 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 3 := by
  sorry

end sqrt_two_irrational_and_between_one_and_three_l1214_121494


namespace nine_power_equation_solution_l1214_121439

theorem nine_power_equation_solution :
  ∃! n : ℝ, (9 : ℝ)^n * (9 : ℝ)^n * (9 : ℝ)^n * (9 : ℝ)^n = (81 : ℝ)^4 :=
by
  sorry

end nine_power_equation_solution_l1214_121439


namespace imaginary_part_of_z_l1214_121455

theorem imaginary_part_of_z (z : ℂ) (h : (z * (1 + Complex.I) * Complex.I^3) / (1 - Complex.I) = 1 - Complex.I) : 
  z.im = -1 := by
  sorry

end imaginary_part_of_z_l1214_121455


namespace fixed_point_power_function_l1214_121403

theorem fixed_point_power_function (a : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (x - 2)^a + 1
  f 3 = 2 := by
sorry

end fixed_point_power_function_l1214_121403


namespace bijection_existence_l1214_121448

theorem bijection_existence :
  (∃ f : ℕ+ × ℕ+ → ℕ+, Function.Bijective f ∧
    (f (1, 1) = 1) ∧
    (∀ i > 1, ∃ d > 1, ∀ j, d ∣ f (i, j)) ∧
    (∀ j > 1, ∃ d > 1, ∀ i, d ∣ f (i, j))) ∧
  (∃ g : ℕ+ × ℕ+ → {n : ℕ+ // n ≠ 1}, Function.Bijective g ∧
    (∀ i, ∃ d > 1, ∀ j, d ∣ (g (i, j)).val) ∧
    (∀ j, ∃ d > 1, ∀ i, d ∣ (g (i, j)).val)) :=
by sorry

end bijection_existence_l1214_121448


namespace no_two_digit_factors_of_2_pow_18_minus_1_l1214_121446

theorem no_two_digit_factors_of_2_pow_18_minus_1 :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → ¬(2^18 - 1) % n = 0 := by sorry

end no_two_digit_factors_of_2_pow_18_minus_1_l1214_121446


namespace value_of_d_l1214_121410

theorem value_of_d (c d : ℚ) (h1 : c / d = 4) (h2 : c = 15 - 4 * d) : d = 15 / 8 := by
  sorry

end value_of_d_l1214_121410


namespace smallest_lcm_with_gcd_5_l1214_121485

theorem smallest_lcm_with_gcd_5 :
  ∃ (a b : ℕ), 
    1000 ≤ a ∧ a < 10000 ∧
    1000 ≤ b ∧ b < 10000 ∧
    Nat.gcd a b = 5 ∧
    Nat.lcm a b = 203010 ∧
    (∀ (c d : ℕ), 1000 ≤ c ∧ c < 10000 ∧ 1000 ≤ d ∧ d < 10000 ∧ Nat.gcd c d = 5 → 
      Nat.lcm c d ≥ 203010) :=
by sorry

end smallest_lcm_with_gcd_5_l1214_121485


namespace power_multiplication_l1214_121461

theorem power_multiplication (a : ℝ) : a^5 * a^2 = a^7 := by
  sorry

end power_multiplication_l1214_121461


namespace acute_triangle_angle_sum_ratio_range_l1214_121447

theorem acute_triangle_angle_sum_ratio_range (A B C : Real) 
  (h_acute : 0 < A ∧ A ≤ B ∧ B ≤ C ∧ C < π/2) 
  (h_triangle : A + B + C = π) : 
  let F := (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C)
  1 + Real.sqrt 2 / 2 < F ∧ F < 2 := by
sorry

end acute_triangle_angle_sum_ratio_range_l1214_121447


namespace inequality_proof_l1214_121479

theorem inequality_proof (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 5/9 := by
  sorry

end inequality_proof_l1214_121479


namespace chocolate_bar_cost_l1214_121407

theorem chocolate_bar_cost (total_bars : ℕ) (bars_sold : ℕ) (revenue : ℝ) : 
  total_bars = 7 → 
  bars_sold = total_bars - 4 → 
  revenue = 9 → 
  revenue / bars_sold = 3 := by
sorry

end chocolate_bar_cost_l1214_121407


namespace difference_of_squares_l1214_121450

theorem difference_of_squares (m : ℝ) : m^2 - 144 = (m - 12) * (m + 12) := by
  sorry

end difference_of_squares_l1214_121450


namespace arithmetic_evaluation_l1214_121404

theorem arithmetic_evaluation : 1537 + 180 / 60 * 15 - 237 = 1345 := by
  sorry

end arithmetic_evaluation_l1214_121404


namespace ratio_average_problem_l1214_121405

theorem ratio_average_problem (a b c : ℕ+) (h_ratio : a.val / 2 = b.val / 3 ∧ b.val / 3 = c.val / 4) (h_a : a = 28) :
  (a.val + b.val + c.val) / 3 = 42 := by
sorry

end ratio_average_problem_l1214_121405


namespace cube_sum_equality_l1214_121422

theorem cube_sum_equality (x y z a b c : ℝ) 
  (hx : x^2 = a) (hy : y^2 = b) (hz : z^2 = c) :
  ∃ (s : ℝ), s = 1 ∨ s = -1 ∧ x^3 + y^3 + z^3 = s * (a^(3/2) + b^(3/2) + c^(3/2)) :=
sorry

end cube_sum_equality_l1214_121422


namespace classroom_tables_count_l1214_121411

/-- The number of tables in Miss Smith's classroom --/
def number_of_tables : ℕ :=
  let total_students : ℕ := 47
  let students_per_table : ℕ := 3
  let bathroom_students : ℕ := 3
  let canteen_students : ℕ := 3 * bathroom_students
  let new_group_students : ℕ := 2 * 4
  let exchange_students : ℕ := 3 * 3
  let missing_students : ℕ := bathroom_students + canteen_students + new_group_students + exchange_students
  let present_students : ℕ := total_students - missing_students
  present_students / students_per_table

theorem classroom_tables_count : number_of_tables = 6 := by
  sorry

end classroom_tables_count_l1214_121411


namespace exponential_inequality_l1214_121459

theorem exponential_inequality (a x : Real) (h1 : 0 < a) (h2 : a < 1) (h3 : x > 0) :
  a^(-x) > a^x := by
  sorry

end exponential_inequality_l1214_121459


namespace power_sum_theorem_l1214_121473

theorem power_sum_theorem (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 4) : a^(m+n) = 8 := by
  sorry

end power_sum_theorem_l1214_121473


namespace quadratic_function_properties_l1214_121436

-- Define the quadratic function
def f (a c x : ℝ) : ℝ := a * x^2 - 4*x + c

-- State the theorem
theorem quadratic_function_properties :
  ∀ a c : ℝ,
  (∀ x : ℝ, f a c x < 0 ↔ -1 < x ∧ x < 5) →
  (a = 1 ∧ c = -5) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f a c x ∈ Set.Icc (-9) (-5)) ∧
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 3 ∧ x₂ ∈ Set.Icc 0 3 ∧ f a c x₁ = -9 ∧ f a c x₂ = -5) :=
by sorry


end quadratic_function_properties_l1214_121436


namespace min_moves_to_win_l1214_121495

/-- Represents a box containing balls -/
structure Box where
  white : Nat
  black : Nat

/-- Represents the state of the game -/
structure GameState where
  round : Box
  square : Box

/-- Checks if two boxes have identical contents -/
def boxesIdentical (b1 b2 : Box) : Bool :=
  b1.white = b2.white ∧ b1.black = b2.black

/-- Defines a single move in the game -/
inductive Move
  | discard : Box → Move
  | transfer : Box → Box → Move

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is in a winning state -/
def isWinningState (state : GameState) : Bool :=
  boxesIdentical state.round state.square

/-- The initial state of the game -/
def initialState : GameState :=
  { round := { white := 3, black := 10 }
  , square := { white := 0, black := 8 } }

/-- Theorem: The minimum number of moves to reach a winning state is 17 -/
theorem min_moves_to_win :
  ∃ (moves : List Move),
    moves.length = 17 ∧
    isWinningState (moves.foldl applyMove initialState) ∧
    ∀ (shorter_moves : List Move),
      shorter_moves.length < 17 →
      ¬isWinningState (shorter_moves.foldl applyMove initialState) :=
  sorry

end min_moves_to_win_l1214_121495


namespace characterize_S_l1214_121430

open Set
open Real

-- Define the set of points satisfying the condition
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = sin p.1 / |sin p.1|}

-- Define the set of x-values where sin(x) = 0
def ZeroSin : Set ℝ :=
  {x : ℝ | ∃ n : ℤ, x = n * π}

-- Define the set of x-values where y should be 1
def X1 : Set ℝ :=
  {x : ℝ | ∃ n : ℤ, x ∈ Ioo (π * (2 * n - 1)) (2 * n * π) \ ZeroSin}

-- Define the set of x-values where y should be -1
def X2 : Set ℝ :=
  {x : ℝ | ∃ n : ℤ, x ∈ Ioo (2 * n * π) (π * (2 * n + 1)) \ ZeroSin}

-- The main theorem
theorem characterize_S : ∀ p ∈ S, 
  (p.1 ∈ X1 ∧ p.2 = 1) ∨ (p.1 ∈ X2 ∧ p.2 = -1) :=
by sorry

end characterize_S_l1214_121430


namespace fifteenth_term_is_198_l1214_121457

/-- A second-order arithmetic sequence is a sequence where the differences between consecutive terms form an arithmetic sequence. -/
def SecondOrderArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d₁ d₂ : ℕ, ∀ n : ℕ, a (n + 2) - a (n + 1) = (a (n + 1) - a n) + d₂

/-- The specific second-order arithmetic sequence from the problem. -/
def SpecificSequence (a : ℕ → ℕ) : Prop :=
  SecondOrderArithmeticSequence a ∧ a 1 = 2 ∧ a 2 = 3 ∧ a 3 = 6 ∧ a 4 = 11

theorem fifteenth_term_is_198 (a : ℕ → ℕ) (h : SpecificSequence a) : a 15 = 198 := by
  sorry

#check fifteenth_term_is_198

end fifteenth_term_is_198_l1214_121457


namespace unique_solution_exists_l1214_121444

theorem unique_solution_exists (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃! x : ℝ, a^x = Real.log x / Real.log (1/4) := by sorry

end unique_solution_exists_l1214_121444


namespace incorrect_number_calculation_l1214_121475

theorem incorrect_number_calculation (n : ℕ) (initial_avg correct_avg correct_num incorrect_num : ℚ) : 
  n = 10 ∧ 
  initial_avg = 16 ∧ 
  correct_avg = 18 ∧ 
  correct_num = 46 ∧ 
  n * initial_avg + (correct_num - incorrect_num) = n * correct_avg → 
  incorrect_num = 26 := by
sorry

end incorrect_number_calculation_l1214_121475


namespace special_function_form_l1214_121402

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  differentiable : Differentiable ℝ f
  f_zero_eq_one : f 0 = 1
  f_inequality : ∀ x₁ x₂, f (x₁ + x₂) ≥ f x₁ * f x₂

/-- The main theorem: any function satisfying the given conditions is of the form e^(kx) -/
theorem special_function_form (φ : SpecialFunction) :
  ∃ k : ℝ, ∀ x, φ.f x = Real.exp (k * x) := by
  sorry

end special_function_form_l1214_121402


namespace fraction_simplification_l1214_121468

theorem fraction_simplification :
  201920192019 / 191719171917 = 673 / 639 := by sorry

end fraction_simplification_l1214_121468


namespace sqrt_3x_minus_6_meaningful_l1214_121445

theorem sqrt_3x_minus_6_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 3 * x - 6) ↔ x ≥ 2 := by
  sorry

end sqrt_3x_minus_6_meaningful_l1214_121445


namespace eight_people_circular_arrangements_l1214_121452

/-- The number of distinct circular arrangements of n people around a round table,
    where rotations are considered identical. -/
def circularArrangements (n : ℕ) : ℕ :=
  Nat.factorial (n - 1)

/-- Theorem stating that the number of distinct circular arrangements
    of 8 people around a round table is 5040. -/
theorem eight_people_circular_arrangements :
  circularArrangements 8 = 5040 := by
  sorry

end eight_people_circular_arrangements_l1214_121452


namespace minimize_y_l1214_121443

variable (a b c : ℝ)

def y (x : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + (x - c)^2

theorem minimize_y :
  ∃ (x_min : ℝ), ∀ (x : ℝ), y a b c x_min ≤ y a b c x ∧ x_min = (a + b + c) / 3 :=
sorry

end minimize_y_l1214_121443


namespace scheduling_methods_count_l1214_121491

/-- The number of days for scheduling --/
def num_days : ℕ := 7

/-- The number of volunteers --/
def num_volunteers : ℕ := 4

/-- Function to calculate the number of scheduling methods --/
def scheduling_methods : ℕ := 
  (num_days.choose num_volunteers) * (num_volunteers.factorial / 2)

/-- Theorem stating that the number of scheduling methods is 420 --/
theorem scheduling_methods_count : scheduling_methods = 420 := by
  sorry

end scheduling_methods_count_l1214_121491


namespace gcd_2720_1530_l1214_121467

theorem gcd_2720_1530 : Nat.gcd 2720 1530 = 170 := by
  sorry

end gcd_2720_1530_l1214_121467


namespace charity_raffle_winnings_l1214_121476

theorem charity_raffle_winnings (W : ℝ) : 
  W / 2 - 2 = 55 → W = 114 := by
  sorry

end charity_raffle_winnings_l1214_121476


namespace sum_of_odds_15_to_45_l1214_121464

theorem sum_of_odds_15_to_45 :
  let a₁ : ℕ := 15  -- first term
  let aₙ : ℕ := 45  -- last term
  let d : ℕ := 2    -- common difference
  let n : ℕ := (aₙ - a₁) / d + 1  -- number of terms
  (n : ℚ) * (a₁ + aₙ) / 2 = 480 := by sorry

end sum_of_odds_15_to_45_l1214_121464


namespace available_storage_space_l1214_121449

/-- Represents a two-story warehouse with boxes stored on the second floor -/
structure Warehouse :=
  (second_floor_space : ℝ)
  (first_floor_space : ℝ)
  (box_space : ℝ)

/-- The conditions of the warehouse problem -/
def warehouse_conditions (w : Warehouse) : Prop :=
  w.first_floor_space = 2 * w.second_floor_space ∧
  w.box_space = w.second_floor_space / 4 ∧
  w.box_space = 5000

/-- The theorem stating the available storage space in the warehouse -/
theorem available_storage_space (w : Warehouse) 
  (h : warehouse_conditions w) : 
  w.first_floor_space + w.second_floor_space - w.box_space = 55000 := by
  sorry

end available_storage_space_l1214_121449


namespace probability_same_color_eq_l1214_121406

def total_marbles : ℕ := 5 + 4 + 6 + 3 + 2

def black_marbles : ℕ := 5
def red_marbles : ℕ := 4
def green_marbles : ℕ := 6
def blue_marbles : ℕ := 3
def yellow_marbles : ℕ := 2

def probability_same_color : ℚ :=
  (black_marbles * (black_marbles - 1) * (black_marbles - 2) * (black_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3)) +
  (green_marbles * (green_marbles - 1) * (green_marbles - 2) * (green_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))

theorem probability_same_color_eq : probability_same_color = 129 / 31250 := by
  sorry

end probability_same_color_eq_l1214_121406


namespace dwarfs_stabilize_l1214_121456

/-- Represents a dwarf in the forest -/
structure Dwarf :=
  (id : Fin 12)
  (hat_color : Bool)

/-- Represents the state of the dwarf system at a given time -/
structure DwarfSystem :=
  (dwarfs : Fin 12 → Dwarf)
  (friends : Fin 12 → Fin 12 → Bool)

/-- Counts the number of friend pairs wearing different colored hats -/
def different_hat_pairs (sys : DwarfSystem) : Nat :=
  sorry

/-- Updates the system for a single day -/
def update_system (sys : DwarfSystem) (day : Nat) : DwarfSystem :=
  sorry

/-- Theorem: The number of different hat pairs eventually reaches zero -/
theorem dwarfs_stabilize (initial_sys : DwarfSystem) :
  ∃ n : Nat, ∀ m : Nat, m ≥ n → different_hat_pairs (update_system initial_sys m) = 0 :=
sorry

end dwarfs_stabilize_l1214_121456


namespace initial_men_count_l1214_121463

/-- Represents the number of days the initial food supply lasts -/
def initial_days : ℕ := 22

/-- Represents the number of days that pass before additional men join -/
def days_before_addition : ℕ := 2

/-- Represents the number of additional men that join -/
def additional_men : ℕ := 3040

/-- Represents the number of days the food lasts after additional men join -/
def remaining_days : ℕ := 4

/-- Theorem stating that the initial number of men is 760 -/
theorem initial_men_count : ℕ := by
  sorry

#check initial_men_count

end initial_men_count_l1214_121463


namespace function_characterization_l1214_121465

-- Define the property for the function
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y = f (x - y)

-- State the theorem
theorem function_characterization (f : ℝ → ℝ) (h : SatisfiesProperty f) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) := by
  sorry

end function_characterization_l1214_121465


namespace tangent_line_min_sum_l1214_121478

noncomputable def f (x : ℝ) := x - Real.exp (-x)

theorem tangent_line_min_sum (m n : ℝ) :
  (∃ t : ℝ, (f t = m * t + n) ∧ 
    (∀ x : ℝ, f x ≤ m * x + n)) →
  m + n ≥ 1 - 1 / Real.exp 1 :=
sorry

end tangent_line_min_sum_l1214_121478


namespace third_chapter_pages_l1214_121435

/-- A book with three chapters -/
structure Book where
  chapter1 : ℕ
  chapter2 : ℕ
  chapter3 : ℕ

/-- The theorem stating the number of pages in the third chapter -/
theorem third_chapter_pages (b : Book) 
  (h1 : b.chapter1 = 35)
  (h2 : b.chapter2 = 18)
  (h3 : b.chapter2 = b.chapter3 + 15) :
  b.chapter3 = 3 := by
  sorry

end third_chapter_pages_l1214_121435


namespace car_speed_time_relation_l1214_121460

/-- Proves that reducing speed to 60 km/h increases travel time by a factor of 1.5 --/
theorem car_speed_time_relation (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) :
  distance = 540 ∧ original_time = 6 ∧ new_speed = 60 →
  (distance / new_speed) / original_time = 1.5 := by
  sorry

end car_speed_time_relation_l1214_121460


namespace shopping_time_calculation_l1214_121425

def total_trip_time : ℕ := 90
def wait_for_cart : ℕ := 3
def wait_for_employee : ℕ := 13
def wait_for_stocker : ℕ := 14
def wait_in_line : ℕ := 18

theorem shopping_time_calculation :
  total_trip_time - (wait_for_cart + wait_for_employee + wait_for_stocker + wait_in_line) = 42 := by
  sorry

end shopping_time_calculation_l1214_121425


namespace warehouse_paintable_area_l1214_121493

/-- Represents the dimensions of a rectangular warehouse -/
structure Warehouse where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents the dimensions of a window -/
structure Window where
  width : ℝ
  height : ℝ

/-- Calculates the total paintable area of a warehouse -/
def totalPaintableArea (w : Warehouse) (windowCount : ℕ) (windowDim : Window) : ℝ :=
  let wallArea1 := 2 * (w.width * w.height) * 2  -- Both sides of width walls
  let wallArea2 := 2 * (w.length * w.height - windowCount * windowDim.width * windowDim.height) * 2  -- Both sides of length walls with windows
  let ceilingArea := w.width * w.length
  let floorArea := w.width * w.length
  wallArea1 + wallArea2 + ceilingArea + floorArea

/-- Theorem stating the total paintable area of the warehouse -/
theorem warehouse_paintable_area :
  let w : Warehouse := { width := 12, length := 15, height := 7 }
  let windowDim : Window := { width := 2, height := 3 }
  totalPaintableArea w 3 windowDim = 876 := by sorry

end warehouse_paintable_area_l1214_121493


namespace evaluate_expression_l1214_121481

theorem evaluate_expression : (-3)^7 / 3^5 + 2^5 - 7^2 = -26 := by
  sorry

end evaluate_expression_l1214_121481


namespace largest_five_digit_base5_l1214_121417

theorem largest_five_digit_base5 : 
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 := by
  sorry

end largest_five_digit_base5_l1214_121417


namespace min_phi_for_even_shifted_sine_l1214_121418

/-- Given a function f and its left-shifted version g, proves that the minimum φ for g to be even is π/10 -/
theorem min_phi_for_even_shifted_sine (φ : ℝ) (f g : ℝ → ℝ) : 
  (φ > 0) →
  (∀ x, f x = 2 * Real.sin (2 * x + φ)) →
  (∀ x, g x = f (x + π/5)) →
  (∀ x, g x = g (-x)) →
  (∃ k : ℤ, φ = k * π + π/10) →
  φ ≥ π/10 := by
sorry

end min_phi_for_even_shifted_sine_l1214_121418


namespace depth_of_iron_cone_in_mercury_l1214_121408

/-- The depth of submersion of an iron cone in mercury -/
noncomputable def depth_of_submersion (cone_volume : ℝ) (iron_density : ℝ) (mercury_density : ℝ) : ℝ :=
  let submerged_volume := (iron_density * cone_volume) / mercury_density
  (3 * submerged_volume / Real.pi) ^ (1/3)

/-- The theorem stating the depth of submersion for the given problem -/
theorem depth_of_iron_cone_in_mercury :
  let cone_volume : ℝ := 350
  let iron_density : ℝ := 7.2
  let mercury_density : ℝ := 13.6
  abs (depth_of_submersion cone_volume iron_density mercury_density - 5.6141) < 0.0001 := by
  sorry


end depth_of_iron_cone_in_mercury_l1214_121408


namespace sum_of_reciprocal_G_powers_of_three_l1214_121497

def G : ℕ → ℚ
  | 0 => 1
  | 1 => 4/3
  | (n + 2) => 3 * G (n + 1) - 2 * G n

theorem sum_of_reciprocal_G_powers_of_three : ∑' n, 1 / G (3^n) = 1 := by sorry

end sum_of_reciprocal_G_powers_of_three_l1214_121497


namespace peanuts_per_visit_l1214_121471

def store_visits : ℕ := 3
def total_peanuts : ℕ := 21

theorem peanuts_per_visit : total_peanuts / store_visits = 7 := by
  sorry

end peanuts_per_visit_l1214_121471


namespace prob_sum_five_l1214_121462

/-- The number of sides on each die -/
def dice_sides : ℕ := 6

/-- The set of possible outcomes when rolling two dice -/
def roll_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range dice_sides) (Finset.range dice_sides)

/-- The set of outcomes that sum to 5 -/
def sum_five : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 + p.2 = 5) roll_outcomes

/-- The probability of rolling a sum of 5 with two fair dice -/
theorem prob_sum_five :
  (Finset.card sum_five : ℚ) / (Finset.card roll_outcomes : ℚ) = 1 / 9 := by
  sorry

end prob_sum_five_l1214_121462


namespace rectangle_relationships_l1214_121469

-- Define the rectangle
def rectangle (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ 2*x + 2*y = 10

-- Define the area function
def area (x y : ℝ) : ℝ := x * y

-- Theorem statement
theorem rectangle_relationships (x y : ℝ) (h : rectangle x y) :
  ∃ (a b : ℝ), y = a*x + b ∧    -- Linear relationship between y and x
  ∃ (p q r : ℝ), area x y = p*x^2 + q*x + r :=  -- Quadratic relationship between S and x
by sorry

end rectangle_relationships_l1214_121469


namespace complex_exponentiation_205_deg_72_l1214_121426

theorem complex_exponentiation_205_deg_72 :
  (Complex.exp (205 * π / 180 * Complex.I)) ^ 72 = -1/2 - Complex.I * Real.sqrt 3 / 2 := by
  sorry

end complex_exponentiation_205_deg_72_l1214_121426


namespace f_composition_equals_251_l1214_121451

def f (x : ℝ) : ℝ := 5 * x - 4

theorem f_composition_equals_251 : f (f (f 3)) = 251 := by
  sorry

end f_composition_equals_251_l1214_121451


namespace cow_count_l1214_121442

theorem cow_count (ducks cows : ℕ) : 
  (2 * ducks + 4 * cows = 2 * (ducks + cows) + 32) → 
  cows = 16 := by
  sorry

end cow_count_l1214_121442


namespace sin_2alpha_plus_sin_2beta_zero_l1214_121427

theorem sin_2alpha_plus_sin_2beta_zero (α β : ℝ) 
  (h : Real.sin α * Real.sin β + Real.cos α * Real.cos β = 0) : 
  Real.sin (2 * α) + Real.sin (2 * β) = 0 := by
  sorry

end sin_2alpha_plus_sin_2beta_zero_l1214_121427


namespace relationship_abcd_l1214_121441

theorem relationship_abcd (a b c d : ℝ) 
  (h1 : a < b) 
  (h2 : d < c) 
  (h3 : (c - a) * (c - b) < 0) 
  (h4 : (d - a) * (d - b) > 0) : 
  d < a ∧ a < c ∧ c < b := by
  sorry

end relationship_abcd_l1214_121441


namespace polynomial_identity_sum_l1214_121440

theorem polynomial_identity_sum (d₁ d₂ d₃ e₁ e₂ e₃ : ℝ) 
  (h : ∀ x : ℝ, x^8 - x^6 + x^4 - x^2 + 1 = 
    (x^2 + d₁*x + e₁) * (x^2 + d₂*x + e₂) * (x^2 + d₃*x + e₃) * (x^2 + 1)) :
  d₁*e₁ + d₂*e₂ + d₃*e₃ = -1 := by
  sorry

end polynomial_identity_sum_l1214_121440


namespace zmod_is_field_l1214_121499

/-- Given a prime number p, (ℤ/pℤ, +, ×, 0, 1) is a commutative field -/
theorem zmod_is_field (p : ℕ) (hp : Prime p) : Field (ZMod p) := by sorry

end zmod_is_field_l1214_121499


namespace november_rainfall_is_180_inches_l1214_121498

/-- Calculates the total rainfall in November given the conditions -/
def november_rainfall (days_in_november : ℕ) (first_half_days : ℕ) (first_half_daily_rainfall : ℝ) : ℝ :=
  let second_half_days := days_in_november - first_half_days
  let second_half_daily_rainfall := 2 * first_half_daily_rainfall
  let first_half_total := first_half_daily_rainfall * first_half_days
  let second_half_total := second_half_daily_rainfall * second_half_days
  first_half_total + second_half_total

/-- Theorem stating that the total rainfall in November is 180 inches -/
theorem november_rainfall_is_180_inches :
  november_rainfall 30 15 4 = 180 := by
  sorry

end november_rainfall_is_180_inches_l1214_121498


namespace working_light_bulbs_l1214_121477

theorem working_light_bulbs (total_lamps : ℕ) (bulbs_per_lamp : ℕ) 
  (quarter_lamps : ℕ) (half_lamps : ℕ) (remaining_lamps : ℕ) :
  total_lamps = 20 →
  bulbs_per_lamp = 7 →
  quarter_lamps = total_lamps / 4 →
  half_lamps = total_lamps / 2 →
  remaining_lamps = total_lamps - quarter_lamps - half_lamps →
  (quarter_lamps * (bulbs_per_lamp - 2) + 
   half_lamps * (bulbs_per_lamp - 1) + 
   remaining_lamps * (bulbs_per_lamp - 3)) = 105 := by
  sorry

end working_light_bulbs_l1214_121477


namespace min_value_theorem_l1214_121474

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + a*b + a*c + b*c = 4) :
  ∀ x y z, x > 0 → y > 0 → z > 0 → x^2 + x*y + x*z + y*z = 4 →
  2*a + b + c ≤ 2*x + y + z :=
sorry

end min_value_theorem_l1214_121474


namespace bag_original_price_l1214_121483

theorem bag_original_price (sale_price : ℝ) (discount_percentage : ℝ) : 
  sale_price = 135 → discount_percentage = 10 → 
  sale_price = (1 - discount_percentage / 100) * 150 := by
  sorry

end bag_original_price_l1214_121483


namespace dave_trips_l1214_121466

/-- The number of trays Dave can carry at a time -/
def trays_per_trip : ℕ := 12

/-- The number of trays on the first table -/
def trays_table1 : ℕ := 26

/-- The number of trays on the second table -/
def trays_table2 : ℕ := 49

/-- The number of trays on the third table -/
def trays_table3 : ℕ := 65

/-- The number of trays on the fourth table -/
def trays_table4 : ℕ := 38

/-- The total number of trays Dave needs to pick up -/
def total_trays : ℕ := trays_table1 + trays_table2 + trays_table3 + trays_table4

/-- The minimum number of trips Dave needs to make -/
def min_trips : ℕ := (total_trays + trays_per_trip - 1) / trays_per_trip

theorem dave_trips : min_trips = 15 := by
  sorry

end dave_trips_l1214_121466


namespace equation_solution_l1214_121421

theorem equation_solution : 
  ∀ x : ℝ, 4 * x^2 - (x^2 - 2*x + 1) = 0 ↔ x = 1/3 ∨ x = -1 := by
  sorry

end equation_solution_l1214_121421


namespace odd_function_value_l1214_121413

def f (x : ℝ) : ℝ := sorry

theorem odd_function_value (a : ℝ) : 
  (∀ x : ℝ, f x = -f (-x)) → 
  (∀ x : ℝ, x ≥ 0 → f x = 3^x - 2*x + a) → 
  f (-2) = -4 := by sorry

end odd_function_value_l1214_121413


namespace arithmetic_calculations_l1214_121429

theorem arithmetic_calculations :
  (- 4 - (- 2) + (- 5) + 8 = 1) ∧
  (- 1^2023 + 16 / (-2)^2 * |-(1/4)| = 0) := by
sorry

end arithmetic_calculations_l1214_121429


namespace triangle_ratio_l1214_121490

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 2b * sin(2A) = 3a * sin(B) and c = 2b, then a/b = √2 -/
theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  2 * b * Real.sin (2 * A) = 3 * a * Real.sin B →
  c = 2 * b →
  a / b = Real.sqrt 2 := by
sorry

end triangle_ratio_l1214_121490


namespace prism_volume_l1214_121458

/-- Given a right rectangular prism with face areas 24 cm², 32 cm², and 48 cm², 
    its volume is 192 cm³. -/
theorem prism_volume (x y z : ℝ) 
  (h1 : x * y = 24) 
  (h2 : x * z = 32) 
  (h3 : y * z = 48) : 
  x * y * z = 192 := by
  sorry

end prism_volume_l1214_121458


namespace quadratic_inequality_properties_l1214_121401

/-- Given that the solution set of ax^2 + bx + c > 0 is {x | -3 < x < 2}, prove the following statements -/
theorem quadratic_inequality_properties (a b c : ℝ) 
  (h : ∀ x, ax^2 + b*x + c > 0 ↔ -3 < x ∧ x < 2) :
  (a < 0) ∧ 
  (a + b + c > 0) ∧
  (∀ x, c*x^2 + b*x + a < 0 ↔ -1/3 < x ∧ x < 1/2) := by
  sorry

end quadratic_inequality_properties_l1214_121401


namespace largest_m_for_negative_integral_solutions_l1214_121434

def is_negative_integer (x : ℝ) : Prop := x < 0 ∧ ∃ n : ℤ, x = n

theorem largest_m_for_negative_integral_solutions :
  ∃ (m : ℝ),
    m = 570 ∧
    (∀ m' : ℝ, m' > m →
      ¬∃ (x y : ℝ),
        10 * x^2 - m' * x + 560 = 0 ∧
        10 * y^2 - m' * y + 560 = 0 ∧
        x ≠ y ∧
        is_negative_integer x ∧
        is_negative_integer y) ∧
    (∃ (x y : ℝ),
      10 * x^2 - m * x + 560 = 0 ∧
      10 * y^2 - m * y + 560 = 0 ∧
      x ≠ y ∧
      is_negative_integer x ∧
      is_negative_integer y) :=
by sorry

end largest_m_for_negative_integral_solutions_l1214_121434


namespace non_yellow_houses_count_l1214_121428

-- Define the number of houses of each color
def yellow_houses : ℕ := 30
def green_houses : ℕ := 90
def red_houses : ℕ := 70
def blue_houses : ℕ := 60
def pink_houses : ℕ := 50

-- State the theorem
theorem non_yellow_houses_count :
  -- Conditions
  (green_houses = 3 * yellow_houses) →
  (red_houses = yellow_houses + 40) →
  (green_houses = 90) →
  (blue_houses = (green_houses + yellow_houses) / 2) →
  (pink_houses = red_houses / 2 + 15) →
  -- Conclusion
  (green_houses + red_houses + blue_houses + pink_houses = 270) :=
by
  sorry

end non_yellow_houses_count_l1214_121428


namespace constant_remainder_implies_b_25_l1214_121482

def dividend (b x : ℝ) : ℝ := 8 * x^3 - b * x^2 + 2 * x + 5
def divisor (x : ℝ) : ℝ := x^2 - 2 * x + 2

theorem constant_remainder_implies_b_25 :
  (∃ (q r : ℝ → ℝ) (c : ℝ), ∀ x, dividend b x = divisor x * q x + c) → b = 25 := by
  sorry

end constant_remainder_implies_b_25_l1214_121482


namespace ceiling_floor_difference_l1214_121412

theorem ceiling_floor_difference : 
  ⌈(14 : ℚ) / 5 * (-31 : ℚ) / 3⌉ - ⌊(14 : ℚ) / 5 * ⌊(-31 : ℚ) / 3⌋⌋ = 3 := by
  sorry

end ceiling_floor_difference_l1214_121412


namespace sin_n_equals_cos_390_l1214_121454

theorem sin_n_equals_cos_390 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (390 * π / 180) →
  n = 60 ∨ n = 120 := by
sorry

end sin_n_equals_cos_390_l1214_121454


namespace combined_mpg_l1214_121432

/-- Combined miles per gallon calculation -/
theorem combined_mpg (alice_mpg bob_mpg alice_miles bob_miles : ℚ) 
  (h1 : alice_mpg = 30)
  (h2 : bob_mpg = 20)
  (h3 : alice_miles = 120)
  (h4 : bob_miles = 180) :
  (alice_miles + bob_miles) / (alice_miles / alice_mpg + bob_miles / bob_mpg) = 300 / 13 := by
  sorry

#eval (120 + 180) / (120 / 30 + 180 / 20) -- For verification

end combined_mpg_l1214_121432


namespace adult_tickets_sold_l1214_121415

theorem adult_tickets_sold (adult_price child_price total_tickets total_amount : ℕ) 
  (h1 : adult_price = 5)
  (h2 : child_price = 2)
  (h3 : total_tickets = 85)
  (h4 : total_amount = 275) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_amount ∧ 
    adult_tickets = 35 := by
  sorry

end adult_tickets_sold_l1214_121415


namespace quadratic_minimum_l1214_121472

theorem quadratic_minimum (x : ℝ) : 
  ∃ (min_x : ℝ), ∀ (y : ℝ), x^2 + 8*x + 7 ≥ min_x^2 + 8*min_x + 7 ∧ min_x = -4 := by
  sorry

end quadratic_minimum_l1214_121472


namespace two_digit_product_8640_l1214_121423

theorem two_digit_product_8640 (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8640 → 
  min a b = 60 := by
sorry

end two_digit_product_8640_l1214_121423


namespace expected_sixes_is_one_third_l1214_121400

/-- The probability of rolling a 6 on a standard die -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a standard die -/
def prob_not_six : ℚ := 1 - prob_six

/-- The expected number of 6's when rolling two standard dice -/
def expected_sixes : ℚ := 
  0 * (prob_not_six ^ 2) + 
  1 * (2 * prob_six * prob_not_six) + 
  2 * (prob_six ^ 2)

theorem expected_sixes_is_one_third : expected_sixes = 1 / 3 := by
  sorry

end expected_sixes_is_one_third_l1214_121400


namespace simplify_expression_l1214_121416

theorem simplify_expression : (7^4 + 4^5) * (2^3 - (-2)^2)^2 = 54800 := by
  sorry

end simplify_expression_l1214_121416


namespace number_division_problem_l1214_121420

theorem number_division_problem : ∃ N : ℕ,
  (N / (555 + 445) = 2 * (555 - 445)) ∧
  (N % (555 + 445) = 25) ∧
  (N = 220025) := by
sorry

end number_division_problem_l1214_121420


namespace steak_eaten_l1214_121424

theorem steak_eaten (original_weight : ℝ) (burned_fraction : ℝ) (eaten_fraction : ℝ) : 
  original_weight = 30 ∧ 
  burned_fraction = 0.5 ∧ 
  eaten_fraction = 0.8 → 
  original_weight * (1 - burned_fraction) * eaten_fraction = 12 := by
  sorry

end steak_eaten_l1214_121424


namespace sum_reciprocals_l1214_121470

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 4 / ω^2) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 := by
  sorry

end sum_reciprocals_l1214_121470


namespace max_cos_sum_in_triangle_l1214_121489

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_angles : 0 < angleA ∧ 0 < angleB ∧ 0 < angleC
  h_sum_angles : angleA + angleB + angleC = π
  h_cosine_law : b^2 + c^2 - a^2 = b * c

-- Theorem statement
theorem max_cos_sum_in_triangle (t : Triangle) : 
  ∃ (max : ℝ), max = 1 ∧ ∀ (x : ℝ), x = Real.cos t.angleB + Real.cos t.angleC → x ≤ max :=
sorry

end max_cos_sum_in_triangle_l1214_121489
