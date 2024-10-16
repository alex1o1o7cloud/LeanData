import Mathlib

namespace NUMINAMATH_CALUDE_triangle_angle_c_is_right_angle_l1096_109603

/-- Given a triangle ABC, if |sin A - 1/2| and (tan B - √3)² are opposite in sign, 
    then angle C is 90°. -/
theorem triangle_angle_c_is_right_angle 
  (A B C : ℝ) -- Angles of the triangle
  (h_triangle : A + B + C = PI) -- Sum of angles in a triangle is π radians (180°)
  (h_opposite_sign : (|Real.sin A - 1/2| * (Real.tan B - Real.sqrt 3)^2 < 0)) -- Opposite sign condition
  : C = PI / 2 := by -- C is π/2 radians (90°)
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_is_right_angle_l1096_109603


namespace NUMINAMATH_CALUDE_not_octal_7857_l1096_109632

def is_octal_digit (d : Nat) : Prop := d ≤ 7

def is_octal_number (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 8 → is_octal_digit d

theorem not_octal_7857 : ¬ is_octal_number 7857 := by
  sorry

end NUMINAMATH_CALUDE_not_octal_7857_l1096_109632


namespace NUMINAMATH_CALUDE_equation_solutions_l1096_109668

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ = 5/2 ∧ x₂ = 4 ∧ 
  (∀ x : ℝ, 3*(2*x - 5) = (2*x - 5)^2 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1096_109668


namespace NUMINAMATH_CALUDE_quadratic_completion_square_l1096_109611

theorem quadratic_completion_square (x : ℝ) : 
  (∃ (d e : ℤ), (x + d : ℝ)^2 = e ∧ x^2 - 6*x - 15 = 0) → 
  (∃ (d e : ℤ), (x + d : ℝ)^2 = e ∧ x^2 - 6*x - 15 = 0 ∧ d + e = 21) := by
sorry

end NUMINAMATH_CALUDE_quadratic_completion_square_l1096_109611


namespace NUMINAMATH_CALUDE_consortium_psychology_majors_l1096_109626

theorem consortium_psychology_majors 
  (total : ℝ) 
  (college_A_percent : ℝ) 
  (college_B_percent : ℝ) 
  (college_C_percent : ℝ) 
  (college_A_freshmen : ℝ) 
  (college_B_freshmen : ℝ) 
  (college_C_freshmen : ℝ) 
  (college_A_liberal_arts : ℝ) 
  (college_B_liberal_arts : ℝ) 
  (college_C_liberal_arts : ℝ) 
  (college_A_psychology : ℝ) 
  (college_B_psychology : ℝ) 
  (college_C_psychology : ℝ) 
  (h1 : college_A_percent = 0.40) 
  (h2 : college_B_percent = 0.35) 
  (h3 : college_C_percent = 0.25) 
  (h4 : college_A_freshmen = 0.80) 
  (h5 : college_B_freshmen = 0.70) 
  (h6 : college_C_freshmen = 0.60) 
  (h7 : college_A_liberal_arts = 0.60) 
  (h8 : college_B_liberal_arts = 0.50) 
  (h9 : college_C_liberal_arts = 0.40) 
  (h10 : college_A_psychology = 0.50) 
  (h11 : college_B_psychology = 0.40) 
  (h12 : college_C_psychology = 0.30) : 
  (college_A_percent * college_A_freshmen * college_A_liberal_arts * college_A_psychology + 
   college_B_percent * college_B_freshmen * college_B_liberal_arts * college_B_psychology + 
   college_C_percent * college_C_freshmen * college_C_liberal_arts * college_C_psychology) * 100 = 16.3 := by
sorry

end NUMINAMATH_CALUDE_consortium_psychology_majors_l1096_109626


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1096_109685

-- Problem 1
theorem problem_1 (x y : ℝ) : (x + y) * (x - y) + y * (y - 2) = x^2 - 2*y := by
  sorry

-- Problem 2
theorem problem_2 (m : ℝ) (hm1 : m ≠ 2) (hm2 : m ≠ -2) :
  (1 - m / (m + 2)) / ((m^2 - 4*m + 4) / (m^2 - 4)) = 2 / (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1096_109685


namespace NUMINAMATH_CALUDE_matrix_row_replacement_determinant_l1096_109691

theorem matrix_row_replacement_determinant :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![2, 5; 3, 7]
  let B : Matrix (Fin 2) (Fin 2) ℤ := 
    Matrix.updateRow A 1 (fun j => 2 * A 0 j + A 1 j)
  Matrix.det B = -1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_row_replacement_determinant_l1096_109691


namespace NUMINAMATH_CALUDE_equation_solutions_l1096_109682

theorem equation_solutions :
  (∃ x : ℚ, 2 * x - 3 = 3 * (x + 1) ∧ x = -6) ∧
  (∃ x : ℚ, (1/2) * x - (9 * x - 2) / 6 - 2 = 0 ∧ x = -5/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1096_109682


namespace NUMINAMATH_CALUDE_equation_solution_l1096_109699

theorem equation_solution : 
  ∃ y : ℚ, (1 : ℚ) / 3 + 1 / y = (4 : ℚ) / 5 ↔ y = 15 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1096_109699


namespace NUMINAMATH_CALUDE_binomial_150_150_equals_1_l1096_109670

theorem binomial_150_150_equals_1 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_equals_1_l1096_109670


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_closed_form_l1096_109640

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a b : ℝ) (u₀ : ℝ) : ℕ → ℝ
  | 0 => u₀
  | n + 1 => a * ArithmeticGeometricSequence a b u₀ n + b

/-- Theorem for the closed form of an arithmetic-geometric sequence -/
theorem arithmetic_geometric_sequence_closed_form (a b u₀ : ℝ) (ha : a ≠ 1) :
  ∀ n : ℕ, ArithmeticGeometricSequence a b u₀ n = a^n * u₀ + b * (a^n - 1) / (a - 1) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_closed_form_l1096_109640


namespace NUMINAMATH_CALUDE_vector_magnitude_l1096_109601

/-- The magnitude of the vector (-3, 4) is 5. -/
theorem vector_magnitude : Real.sqrt ((-3)^2 + 4^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1096_109601


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1096_109646

theorem partial_fraction_decomposition :
  ∃ (a b c : ℤ), 
    (1 : ℚ) / 2015 = a / 5 + b / 13 + c / 31 ∧
    0 ≤ a ∧ a < 5 ∧
    0 ≤ b ∧ b < 13 ∧
    a + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1096_109646


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1096_109655

/-- 
Given a principal amount P and an interest rate R, 
if increasing the rate by 3% for 4 years results in Rs. 120 more interest, 
then P = 1000.
-/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 3) * 4) / 100 - (P * R * 4) / 100 = 120 → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1096_109655


namespace NUMINAMATH_CALUDE_consecutive_natural_vs_even_product_l1096_109644

theorem consecutive_natural_vs_even_product (n : ℕ) (m : ℕ) (h : m > 0) (h_even : Even m) :
  n * (n + 1) ≠ m * (m + 2) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_natural_vs_even_product_l1096_109644


namespace NUMINAMATH_CALUDE_equation_solution_l1096_109647

theorem equation_solution : 
  ∃ x : ℝ, (5 + 3.2 * x = 2.4 * x - 15) ∧ (x = -25) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1096_109647


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1096_109642

theorem sqrt_equation_solution (a b c : ℝ) 
  (h1 : Real.sqrt a = Real.sqrt b + Real.sqrt c)
  (h2 : b = 52 - 30 * Real.sqrt 3)
  (h3 : c = a - 2) : 
  a = 27 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1096_109642


namespace NUMINAMATH_CALUDE_park_area_l1096_109643

/-- Proves that a rectangular park with a 4:1 length-to-breadth ratio,
    where a cyclist completes the boundary in 8 minutes at 12 km/hr,
    has an area of 102400 square meters. -/
theorem park_area (length breadth : ℝ) (h1 : length = 4 * breadth)
                  (h2 : (length + breadth) * 2 = 12000 / 60 * 8) : 
  length * breadth = 102400 := by
  sorry

end NUMINAMATH_CALUDE_park_area_l1096_109643


namespace NUMINAMATH_CALUDE_functional_polynomial_is_constant_l1096_109667

/-- A polynomial satisfying the given functional equation -/
def FunctionalPolynomial (p : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 1) ∧ p (-1) = 2

theorem functional_polynomial_is_constant
    (p : ℝ → ℝ) (hp : FunctionalPolynomial p) :
    ∀ x : ℝ, p x = 2 := by
  sorry

end NUMINAMATH_CALUDE_functional_polynomial_is_constant_l1096_109667


namespace NUMINAMATH_CALUDE_jasons_house_paintable_area_l1096_109631

/-- The total area to be painted in multiple identical bedrooms -/
def total_paintable_area (num_rooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - unpaintable_area
  num_rooms * paintable_area

/-- Theorem stating the total area to be painted in Jason's house -/
theorem jasons_house_paintable_area :
  total_paintable_area 4 14 11 9 80 = 1480 := by
  sorry

end NUMINAMATH_CALUDE_jasons_house_paintable_area_l1096_109631


namespace NUMINAMATH_CALUDE_player_B_wins_l1096_109688

/-- Represents the state of the pizza game -/
structure GameState :=
  (pizzeria1 : Nat)
  (pizzeria2 : Nat)

/-- Represents a player's move -/
inductive Move
  | EatFromOne (pizzeria : Nat) (amount : Nat)
  | EatFromBoth

/-- Defines the rules of the game -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  match move with
  | Move.EatFromOne 1 amount => amount > 0 ∧ amount ≤ state.pizzeria1
  | Move.EatFromOne 2 amount => amount > 0 ∧ amount ≤ state.pizzeria2
  | Move.EatFromBoth => state.pizzeria1 > 0 ∧ state.pizzeria2 > 0
  | _ => False

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.EatFromOne 1 amount => ⟨state.pizzeria1 - amount, state.pizzeria2⟩
  | Move.EatFromOne 2 amount => ⟨state.pizzeria1, state.pizzeria2 - amount⟩
  | Move.EatFromBoth => ⟨state.pizzeria1 - 1, state.pizzeria2 - 1⟩
  | _ => state

/-- Defines a winning strategy for player B -/
def hasWinningStrategy (player : Nat) : Prop :=
  ∀ (state : GameState),
    (state.pizzeria1 = 2010 ∧ state.pizzeria2 = 2010) →
    ∃ (strategy : GameState → Move),
      (∀ (s : GameState), isValidMove s (strategy s)) ∧
      (player = 2 → ∃ (n : Nat), state.pizzeria1 + state.pizzeria2 = n ∧ n % 2 = 1)

/-- The main theorem: Player B (second player) has a winning strategy -/
theorem player_B_wins : hasWinningStrategy 2 := by
  sorry

end NUMINAMATH_CALUDE_player_B_wins_l1096_109688


namespace NUMINAMATH_CALUDE_max_consecutive_sum_less_than_500_l1096_109684

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- 31 is the largest positive integer n such that the sum of the first n positive integers is less than 500 -/
theorem max_consecutive_sum_less_than_500 :
  ∀ n : ℕ, sum_first_n n < 500 ↔ n ≤ 31 :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_less_than_500_l1096_109684


namespace NUMINAMATH_CALUDE_negation_of_all_dogs_are_playful_l1096_109612

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for being a dog and being playful
variable (dog : U → Prop)
variable (playful : U → Prop)

-- State the theorem
theorem negation_of_all_dogs_are_playful :
  (¬∀ x, dog x → playful x) ↔ (∃ x, dog x ∧ ¬playful x) :=
sorry

end NUMINAMATH_CALUDE_negation_of_all_dogs_are_playful_l1096_109612


namespace NUMINAMATH_CALUDE_playground_children_l1096_109605

/-- The number of children on a playground given the number of boys and girls -/
theorem playground_children (boys girls : ℕ) (h1 : boys = 40) (h2 : girls = 77) :
  boys + girls = 117 := by
  sorry

end NUMINAMATH_CALUDE_playground_children_l1096_109605


namespace NUMINAMATH_CALUDE_largest_element_l1096_109604

def S (a : ℝ) : Set ℝ := {-3*a, 2*a, 18/a, a^2, 1}

theorem largest_element (a : ℝ) (h : a = 3) : ∀ x ∈ S a, x ≤ a^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_element_l1096_109604


namespace NUMINAMATH_CALUDE_square_sum_equals_fifteen_l1096_109649

theorem square_sum_equals_fifteen (x y : ℝ) (h1 : x * y = 3) (h2 : (x - y)^2 = 9) :
  x^2 + y^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_fifteen_l1096_109649


namespace NUMINAMATH_CALUDE_range_of_t_l1096_109681

/-- Given a set A containing 1 and a real number t, prove that the range of t is all real numbers except 1. -/
theorem range_of_t (t : ℝ) (A : Set ℝ) (h : A = {1, t}) : 
  {x : ℝ | x ≠ 1} = {x : ℝ | ∃ (s : Set ℝ), s = {1, x} ∧ s = A} :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_l1096_109681


namespace NUMINAMATH_CALUDE_pen_probabilities_l1096_109620

/-- Represents the total number of pens in the box -/
def total_pens : ℕ := 6

/-- Represents the number of first-class quality pens -/
def first_class_pens : ℕ := 4

/-- Represents the number of second-class quality pens -/
def second_class_pens : ℕ := 2

/-- Represents the number of pens drawn -/
def pens_drawn : ℕ := 2

/-- Calculates the probability of drawing exactly one first-class quality pen -/
def prob_one_first_class : ℚ :=
  (Nat.choose first_class_pens 1 * Nat.choose second_class_pens 1) / Nat.choose total_pens pens_drawn

/-- Calculates the probability of drawing at least one second-class quality pen -/
def prob_at_least_one_second_class : ℚ :=
  1 - (Nat.choose first_class_pens pens_drawn) / Nat.choose total_pens pens_drawn

theorem pen_probabilities :
  prob_one_first_class = 8/15 ∧
  prob_at_least_one_second_class = 3/5 :=
sorry

end NUMINAMATH_CALUDE_pen_probabilities_l1096_109620


namespace NUMINAMATH_CALUDE_monotonicity_and_minimum_l1096_109610

/-- The function f(x) = kx^3 - 3x^2 + 1 -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x^3 - 3 * x^2 + 1

/-- The derivative of f(x) -/
noncomputable def f_deriv (k : ℝ) (x : ℝ) : ℝ := 3 * k * x^2 - 6 * x

theorem monotonicity_and_minimum (k : ℝ) (h : k ≥ 0) :
  (∀ x y, x ≤ 0 → y ∈ Set.Ioo 0 (2/k) → f k x ≤ f k y) ∧ 
  (∀ x y, x ∈ Set.Icc 0 (2/k) → y ≥ 2/k → f k x ≤ f k y) ∧
  (k > 2 ↔ f k (2/k) > 0) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_and_minimum_l1096_109610


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_attained_l1096_109696

theorem max_value_expression (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h_sum : a^2 + b^2 + c^2 = 1) : 
  3 * a * b * Real.sqrt 3 + 9 * b * c ≤ 3 :=
by sorry

theorem max_value_attained (ε : ℝ) (hε : ε > 0) : 
  ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a^2 + b^2 + c^2 = 1 ∧ 
  3 * a * b * Real.sqrt 3 + 9 * b * c > 3 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_attained_l1096_109696


namespace NUMINAMATH_CALUDE_profit_percentage_l1096_109624

theorem profit_percentage (C S : ℝ) (h : 315 * C = 250 * S) : 
  (S - C) / C * 100 = 26 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_l1096_109624


namespace NUMINAMATH_CALUDE_probability_three_common_books_is_32_495_l1096_109664

def total_books : ℕ := 12
def books_selected : ℕ := 4

def probability_three_common_books : ℚ :=
  (Nat.choose total_books 3 * Nat.choose (total_books - 3) 1 * Nat.choose (total_books - 4) 1) /
  (Nat.choose total_books books_selected * Nat.choose total_books books_selected)

theorem probability_three_common_books_is_32_495 :
  probability_three_common_books = 32 / 495 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_common_books_is_32_495_l1096_109664


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1096_109606

/-- The complex number z = sin 2 + i cos 2 is located in the fourth quadrant of the complex plane -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := Complex.mk (Real.sin 2) (Real.cos 2)
  z.re > 0 ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1096_109606


namespace NUMINAMATH_CALUDE_maximize_product_l1096_109637

theorem maximize_product (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 50) :
  x^4 * y^3 ≤ (200/7)^4 * (150/7)^3 ∧
  x^4 * y^3 = (200/7)^4 * (150/7)^3 ↔ x = 200/7 ∧ y = 150/7 := by
  sorry

end NUMINAMATH_CALUDE_maximize_product_l1096_109637


namespace NUMINAMATH_CALUDE_ellipse_equation_l1096_109697

/-- An ellipse with one focus at (1,0) and eccentricity 1/2 has the standard equation x²/4 + y²/3 = 1 -/
theorem ellipse_equation (F : ℝ × ℝ) (e : ℝ) (h1 : F = (1, 0)) (h2 : e = 1/2) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 / 3 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1096_109697


namespace NUMINAMATH_CALUDE_maintenance_building_length_l1096_109690

/-- The length of the maintenance building on a square playground -/
theorem maintenance_building_length : 
  ∀ (playground_side : ℝ) (building_width : ℝ) (uncovered_area : ℝ),
  playground_side = 12 →
  building_width = 5 →
  uncovered_area = 104 →
  ∃ (building_length : ℝ),
    building_length = 8 ∧
    building_length * building_width = playground_side^2 - uncovered_area :=
by sorry

end NUMINAMATH_CALUDE_maintenance_building_length_l1096_109690


namespace NUMINAMATH_CALUDE_complex_magnitude_l1096_109683

theorem complex_magnitude (z : ℂ) : z = (2 - I) / (1 + I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1096_109683


namespace NUMINAMATH_CALUDE_min_coins_for_dollar_l1096_109675

/-- Represents the different types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- Returns the value of a coin in cents --/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- Calculates the total value of a list of coins in cents --/
def totalValue (coins : List Coin) : ℕ :=
  coins.foldl (fun acc c => acc + coinValue c) 0

/-- Theorem: The minimum number of coins to make one dollar is 3 --/
theorem min_coins_for_dollar :
  ∃ (coins : List Coin), totalValue coins = 100 ∧
    (∀ (other_coins : List Coin), totalValue other_coins = 100 →
      coins.length ≤ other_coins.length) ∧
    coins.length = 3 :=
  sorry

#check min_coins_for_dollar

end NUMINAMATH_CALUDE_min_coins_for_dollar_l1096_109675


namespace NUMINAMATH_CALUDE_large_block_length_multiple_l1096_109674

/-- Represents the dimensions of a block of cheese -/
structure CheeseDimensions where
  width : ℝ
  depth : ℝ
  length : ℝ

/-- Calculates the volume of a block of cheese given its dimensions -/
def volume (d : CheeseDimensions) : ℝ :=
  d.width * d.depth * d.length

theorem large_block_length_multiple (normal : CheeseDimensions) (large : CheeseDimensions) :
  volume normal = 3 →
  large.width = 2 * normal.width →
  large.depth = 2 * normal.depth →
  volume large = 36 →
  large.length = 3 * normal.length := by
  sorry

#check large_block_length_multiple

end NUMINAMATH_CALUDE_large_block_length_multiple_l1096_109674


namespace NUMINAMATH_CALUDE_bisecting_line_projection_ratio_l1096_109625

/-- A convex polygon type -/
structure ConvexPolygon where
  -- Add necessary fields

/-- A line type -/
structure Line where
  -- Add necessary fields

/-- Represents the projection of a polygon onto a line -/
structure Projection where
  -- Add necessary fields

/-- Checks if a line bisects the area of a polygon -/
def bisects_area (l : Line) (p : ConvexPolygon) : Prop :=
  sorry

/-- Gets the projection of a polygon onto a line perpendicular to the given line -/
def get_perpendicular_projection (p : ConvexPolygon) (l : Line) : Projection :=
  sorry

/-- Gets the ratio of the segments created by a line on a projection -/
def projection_ratio (proj : Projection) (l : Line) : ℝ :=
  sorry

/-- The main theorem -/
theorem bisecting_line_projection_ratio 
  (p : ConvexPolygon) (l : Line) :
  bisects_area l p →
  projection_ratio (get_perpendicular_projection p l) l ≤ 1 + Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_bisecting_line_projection_ratio_l1096_109625


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_one_l1096_109623

theorem sqrt_expression_equals_one :
  (Real.sqrt 6 - Real.sqrt 2) / Real.sqrt 2 + |Real.sqrt 3 - 2| = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_one_l1096_109623


namespace NUMINAMATH_CALUDE_absolute_value_and_square_root_l1096_109621

theorem absolute_value_and_square_root (x : ℝ) (h : 1 < x ∧ x ≤ 2) :
  |x - 3| + Real.sqrt ((x - 2)^2) = 5 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_square_root_l1096_109621


namespace NUMINAMATH_CALUDE_circle_area_increase_l1096_109609

theorem circle_area_increase (c : ℝ) (r_increase : ℝ) (h1 : c = 16 * Real.pi) (h2 : r_increase = 2) :
  let r := c / (2 * Real.pi)
  let new_r := r + r_increase
  let area_increase := Real.pi * new_r^2 - Real.pi * r^2
  area_increase = 36 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_area_increase_l1096_109609


namespace NUMINAMATH_CALUDE_problem1_l1096_109634

theorem problem1 (m n : ℝ) : (m + n) * (2 * m + n) + n * (m - n) = 2 * m^2 + 4 * m * n := by
  sorry

end NUMINAMATH_CALUDE_problem1_l1096_109634


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l1096_109636

-- Define the solution sets M and N
def M (p : ℝ) : Set ℝ := {x | x^2 - p*x + 8 = 0}
def N (p q : ℝ) : Set ℝ := {x | x^2 - q*x + p = 0}

-- State the theorem
theorem intersection_implies_sum (p q : ℝ) :
  M p ∩ N p q = {1} → p + q = 19 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l1096_109636


namespace NUMINAMATH_CALUDE_problem_solution_g_minimum_l1096_109615

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x - a| + a

-- State the theorem
theorem problem_solution :
  (∀ x, f 1 x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧
  (∀ m : ℝ, (∃ t : ℝ, f 1 (t/2) ≤ m - f 1 (-t)) ↔ 3.5 ≤ m) := by
  sorry

-- Define the function for the second part
def g (t : ℝ) : ℝ := |t - 1| + |2 * t + 1| + 2

-- State the minimum value theorem
theorem g_minimum : ∀ t : ℝ, g t ≥ 3.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_g_minimum_l1096_109615


namespace NUMINAMATH_CALUDE_jiyoung_pocket_money_l1096_109679

theorem jiyoung_pocket_money (total : ℕ) (difference : ℕ) (jiyoung : ℕ) :
  total = 12000 →
  difference = 1000 →
  total = jiyoung + (jiyoung - difference) →
  jiyoung = 6500 := by
  sorry

end NUMINAMATH_CALUDE_jiyoung_pocket_money_l1096_109679


namespace NUMINAMATH_CALUDE_different_suit_card_selection_l1096_109613

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def num_suits : ℕ := 4

/-- The number of cards per suit in a standard deck -/
def cards_per_suit : ℕ := 13

/-- The number of cards to be chosen -/
def cards_to_choose : ℕ := 4

/-- Theorem: The number of ways to choose 4 cards from a standard deck of 52 cards,
    where all four cards must be of different suits, is equal to 28561. -/
theorem different_suit_card_selection :
  (cards_per_suit ^ cards_to_choose : ℕ) = 28561 := by
  sorry

end NUMINAMATH_CALUDE_different_suit_card_selection_l1096_109613


namespace NUMINAMATH_CALUDE_money_lending_problem_l1096_109651

/-- Given a sum of money divided into two parts where:
    1. The interest on the first part for 8 years at 3% per annum is equal to
       the interest on the second part for 3 years at 5% per annum.
    2. The second part is Rs. 1656.
    Prove that the total sum lent is Rs. 2691. -/
theorem money_lending_problem (first_part second_part total_sum : ℚ) : 
  second_part = 1656 →
  (first_part * 3 / 100 * 8 = second_part * 5 / 100 * 3) →
  total_sum = first_part + second_part →
  total_sum = 2691 := by
  sorry

#check money_lending_problem

end NUMINAMATH_CALUDE_money_lending_problem_l1096_109651


namespace NUMINAMATH_CALUDE_subset_implies_m_value_l1096_109600

theorem subset_implies_m_value (A B : Set ℝ) (m : ℝ) : 
  A = {-1} →
  B = {x : ℝ | x^2 + m*x - 3 = 1} →
  A ⊆ B →
  m = -3 := by sorry

end NUMINAMATH_CALUDE_subset_implies_m_value_l1096_109600


namespace NUMINAMATH_CALUDE_cos_two_alpha_equals_zero_l1096_109695

theorem cos_two_alpha_equals_zero (α : ℝ) 
  (h : Real.sin (π / 6 - α) = Real.cos (π / 6 + α)) : 
  Real.cos (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_equals_zero_l1096_109695


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l1096_109692

/-- The range of 'a' for which the ellipse x^2 + 4(y-a)^2 = 4 intersects with the parabola x^2 = 2y -/
theorem ellipse_parabola_intersection_range :
  ∀ (a : ℝ),
  (∃ (x y : ℝ), x^2 + 4*(y-a)^2 = 4 ∧ x^2 = 2*y) →
  -1 ≤ a ∧ a ≤ 17/8 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l1096_109692


namespace NUMINAMATH_CALUDE_smallest_shift_l1096_109602

open Real

theorem smallest_shift (n : ℝ) : n > 0 ∧ 
  (∀ x, cos (2 * π * x - π / 3) = sin (2 * π * (x - n) + π / 3)) → 
  n ≥ 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_shift_l1096_109602


namespace NUMINAMATH_CALUDE_katie_math_problems_l1096_109618

/-- Given that Katie had 9 math problems for homework and 4 problems left to do after the bus ride,
    prove that she finished 5 problems on the bus ride home. -/
theorem katie_math_problems (total : ℕ) (remaining : ℕ) (h1 : total = 9) (h2 : remaining = 4) :
  total - remaining = 5 := by
  sorry

end NUMINAMATH_CALUDE_katie_math_problems_l1096_109618


namespace NUMINAMATH_CALUDE_circle_op_properties_l1096_109638

-- Define the set A as ordered pairs of real numbers
def A : Type := ℝ × ℝ

-- Define the operation ⊙
def circle_op (α β : A) : A :=
  let (a, b) := α
  let (c, d) := β
  (a * d + b * c, b * d - a * c)

-- Theorem statement
theorem circle_op_properties :
  -- Part 1: Specific calculation
  circle_op (2, 3) (-1, 4) = (5, 14) ∧
  -- Part 2: Commutativity
  (∀ α β : A, circle_op α β = circle_op β α) ∧
  -- Part 3: Identity element
  (∃ I : A, ∀ α : A, circle_op I α = α ∧ circle_op α I = α) ∧
  (∀ I : A, (∀ α : A, circle_op I α = α ∧ circle_op α I = α) → I = (0, 1)) :=
by sorry

end NUMINAMATH_CALUDE_circle_op_properties_l1096_109638


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_ratio_l1096_109654

theorem isosceles_triangle_side_ratio (a b : ℝ) (h_isosceles : b > 0) (h_vertex_angle : Real.cos (20 * π / 180) = a / (2 * b)) : 2 < b / a ∧ b / a < 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_ratio_l1096_109654


namespace NUMINAMATH_CALUDE_tv_price_change_l1096_109617

theorem tv_price_change (P : ℝ) : P > 0 →
  let price_after_decrease := P * (1 - 0.20)
  let price_after_increase := price_after_decrease * (1 + 0.30)
  price_after_increase = P * 1.04 :=
by
  sorry

end NUMINAMATH_CALUDE_tv_price_change_l1096_109617


namespace NUMINAMATH_CALUDE_sum_is_non_horizontal_line_l1096_109629

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure Quadratic where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola -/
def original_parabola : Quadratic → ℝ → ℝ := λ q x => q.a * x^2 + q.b * x + q.c

/-- Reflection of the parabola about the x-axis -/
def reflected_parabola : Quadratic → ℝ → ℝ := λ q x => -q.a * x^2 - q.b * x - q.c

/-- Horizontal translation of a function -/
def translate (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := λ x => f (x - h)

/-- The sum of the translated original and reflected parabolas -/
def sum_of_translated_parabolas (q : Quadratic) : ℝ → ℝ :=
  λ x => translate (original_parabola q) 3 x + translate (reflected_parabola q) (-3) x

/-- Theorem stating that the sum of translated parabolas is a non-horizontal line -/
theorem sum_is_non_horizontal_line (q : Quadratic) (h : q.a ≠ 0) :
  ∃ m k : ℝ, m ≠ 0 ∧ ∀ x, sum_of_translated_parabolas q x = m * x + k :=
sorry

end NUMINAMATH_CALUDE_sum_is_non_horizontal_line_l1096_109629


namespace NUMINAMATH_CALUDE_concentric_circles_equal_areas_l1096_109633

theorem concentric_circles_equal_areas (R : ℝ) (R₁ R₂ : ℝ) 
  (h₁ : R > 0) 
  (h₂ : R₁ > 0) 
  (h₃ : R₂ > 0) 
  (h₄ : R₁ < R₂) 
  (h₅ : R₂ < R) 
  (h₆ : π * R₁^2 = (π * R^2) / 3) 
  (h₇ : π * R₂^2 - π * R₁^2 = (π * R^2) / 3) 
  (h₈ : π * R^2 - π * R₂^2 = (π * R^2) / 3) : 
  R₁ = (R * Real.sqrt 3) / 3 ∧ R₂ = (R * Real.sqrt 6) / 3 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_equal_areas_l1096_109633


namespace NUMINAMATH_CALUDE_y_change_when_x_increases_y_decreases_by_1_5_l1096_109656

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 2 - 1.5 * x

-- Theorem stating the change in y when x increases by one unit
theorem y_change_when_x_increases (x : ℝ) :
  regression_equation (x + 1) = regression_equation x - 1.5 := by
  sorry

-- Theorem stating that y decreases by 1.5 units when x increases by one unit
theorem y_decreases_by_1_5 (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -1.5 := by
  sorry

end NUMINAMATH_CALUDE_y_change_when_x_increases_y_decreases_by_1_5_l1096_109656


namespace NUMINAMATH_CALUDE_equation_solution_l1096_109665

theorem equation_solution : ∃ y : ℝ, (125 : ℝ)^(3*y) = 25^(4*y - 5) ∧ y = -10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1096_109665


namespace NUMINAMATH_CALUDE_orthogonal_vectors_m_l1096_109693

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, -1)

theorem orthogonal_vectors_m (m : ℝ) : 
  (a.1 + m * b.1, a.2 + m * b.2) • (a.1 - b.1, a.2 - b.2) = 0 → m = 23 / 3 :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_m_l1096_109693


namespace NUMINAMATH_CALUDE_special_numbers_count_l1096_109658

def count_multiples (n : ℕ) (max : ℕ) : ℕ :=
  (max / n : ℕ)

def count_special_numbers (max : ℕ) : ℕ :=
  count_multiples 4 max + count_multiples 5 max - count_multiples 20 max - count_multiples 25 max

theorem special_numbers_count :
  count_special_numbers 3000 = 1080 := by sorry

end NUMINAMATH_CALUDE_special_numbers_count_l1096_109658


namespace NUMINAMATH_CALUDE_inequality_proof_l1096_109641

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 1) / (y + 1) + (y + 1) / (z + 1) + (z + 1) / (x + 1) ≤ x / y + y / z + z / x :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1096_109641


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1096_109660

def f (x : ℝ) : ℝ := 6*x^3 - 15*x^2 + 21*x - 23

theorem polynomial_remainder_theorem :
  ∃ (q : ℝ → ℝ), f = λ x => (3*x - 6) * q x + 7 :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1096_109660


namespace NUMINAMATH_CALUDE_dehydrated_men_fraction_l1096_109661

theorem dehydrated_men_fraction (total_men : ℕ) (finished_men : ℕ) 
  (h1 : total_men = 80)
  (h2 : finished_men = 52)
  (h3 : (1 : ℚ) / 4 * total_men = total_men - (total_men - (1 : ℚ) / 4 * total_men))
  (h4 : ∃ x : ℚ, x * (total_men - (1 : ℚ) / 4 * total_men) * (1 : ℚ) / 5 = 
    total_men - finished_men - (1 : ℚ) / 4 * total_men) :
  ∃ x : ℚ, x = 2 / 3 ∧ 
    x * (total_men - (1 : ℚ) / 4 * total_men) * (1 : ℚ) / 5 = 
    total_men - finished_men - (1 : ℚ) / 4 * total_men :=
by sorry


end NUMINAMATH_CALUDE_dehydrated_men_fraction_l1096_109661


namespace NUMINAMATH_CALUDE_smallest_sum_of_primes_with_all_digits_l1096_109678

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the digits of a number -/
def digits (n : ℕ) : List ℕ := sorry

/-- A function that checks if a list contains all digits from 0 to 9 exactly once -/
def hasAllDigitsOnce (l : List ℕ) : Prop := sorry

/-- The theorem stating the smallest sum of primes using all digits once -/
theorem smallest_sum_of_primes_with_all_digits : 
  ∃ (s : List ℕ), 
    (∀ n ∈ s, isPrime n) ∧ 
    hasAllDigitsOnce (s.bind digits) ∧
    (s.sum = 208) ∧
    (∀ (t : List ℕ), 
      (∀ m ∈ t, isPrime m) → 
      hasAllDigitsOnce (t.bind digits) → 
      t.sum ≥ 208) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_primes_with_all_digits_l1096_109678


namespace NUMINAMATH_CALUDE_elizabeth_pencil_purchase_l1096_109627

def pencil_cost : ℕ := 600
def elizabeth_money : ℕ := 500
def borrowed_money : ℕ := 53

theorem elizabeth_pencil_purchase : 
  pencil_cost - (elizabeth_money + borrowed_money) = 47 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_pencil_purchase_l1096_109627


namespace NUMINAMATH_CALUDE_alyssa_fruit_expenditure_l1096_109672

/-- The amount Alyssa paid for grapes in dollars -/
def grapes_cost : ℚ := 12.08

/-- The amount Alyssa paid for cherries in dollars -/
def cherries_cost : ℚ := 9.85

/-- The total amount Alyssa spent on fruits -/
def total_cost : ℚ := grapes_cost + cherries_cost

theorem alyssa_fruit_expenditure : total_cost = 21.93 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_fruit_expenditure_l1096_109672


namespace NUMINAMATH_CALUDE_laundry_loads_required_l1096_109657

def num_families : ℕ := 3
def people_per_family : ℕ := 4
def vacation_days : ℕ := 7
def towels_per_person_per_day : ℕ := 1
def washing_machine_capacity : ℕ := 14

def total_people : ℕ := num_families * people_per_family
def total_towels : ℕ := total_people * vacation_days * towels_per_person_per_day

theorem laundry_loads_required :
  (total_towels + washing_machine_capacity - 1) / washing_machine_capacity = 6 := by
  sorry

end NUMINAMATH_CALUDE_laundry_loads_required_l1096_109657


namespace NUMINAMATH_CALUDE_troll_count_l1096_109677

/-- The number of creatures at the table -/
def total_creatures : ℕ := 60

/-- The number of trolls at the table -/
def num_trolls : ℕ := 20

/-- The number of elves who made a mistake -/
def mistake_elves : ℕ := 2

theorem troll_count :
  ∀ t : ℕ,
  t = num_trolls →
  (∃ x : ℕ,
    x ∈ ({2, 4, 6} : Set ℕ) ∧
    3 * t + x = total_creatures + 4 ∧
    t + (total_creatures - t) = total_creatures ∧
    t - mistake_elves = (total_creatures - t) - x / 2) :=
by sorry

end NUMINAMATH_CALUDE_troll_count_l1096_109677


namespace NUMINAMATH_CALUDE_rectangle_circle_propositions_l1096_109698

theorem rectangle_circle_propositions (p q : Prop) 
  (hp : p) 
  (hq : ¬q) : 
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_propositions_l1096_109698


namespace NUMINAMATH_CALUDE_no_integer_solution_l1096_109608

theorem no_integer_solution :
  ¬ ∃ (x y : ℤ), x^3 + 3 = 4*y*(y + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1096_109608


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_l1096_109694

theorem mod_equivalence_unique : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 5 ∧ n ≡ -1723 [ZMOD 6] := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_l1096_109694


namespace NUMINAMATH_CALUDE_decagon_perimeter_l1096_109689

/-- The number of sides in a decagon -/
def decagon_sides : ℕ := 10

/-- The length of each side of the regular decagon -/
def side_length : ℝ := 3

/-- The perimeter of a regular polygon -/
def perimeter (n : ℕ) (s : ℝ) : ℝ := n * s

/-- Theorem: The perimeter of a regular decagon with side length 3 units is 30 units -/
theorem decagon_perimeter : 
  perimeter decagon_sides side_length = 30 := by sorry

end NUMINAMATH_CALUDE_decagon_perimeter_l1096_109689


namespace NUMINAMATH_CALUDE_units_digit_problem_l1096_109676

theorem units_digit_problem : ∃ n : ℕ, (8 * 14 * 1986 + 8^2) % 10 = 6 ∧ n * 10 ≤ (8 * 14 * 1986 + 8^2) ∧ (8 * 14 * 1986 + 8^2) < (n + 1) * 10 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l1096_109676


namespace NUMINAMATH_CALUDE_real_number_line_bijection_sqrt_six_representation_l1096_109628

-- Define the number line as a type isomorphic to ℝ
def NumberLine : Type := ℝ

-- Statement 1: There exists a bijective function between real numbers and points on the number line
theorem real_number_line_bijection : ∃ f : ℝ → NumberLine, Function.Bijective f :=
sorry

-- Statement 2: The arithmetic square root of 6 is represented by √6
theorem sqrt_six_representation : Real.sqrt 6 = (6 : ℝ).sqrt :=
sorry

end NUMINAMATH_CALUDE_real_number_line_bijection_sqrt_six_representation_l1096_109628


namespace NUMINAMATH_CALUDE_nancy_shoe_multiple_l1096_109645

/-- Given Nancy's shoe collection, prove the multiple relating heels to boots and slippers --/
theorem nancy_shoe_multiple :
  ∀ (boots slippers heels : ℕ),
  boots = 6 →
  slippers = boots + 9 →
  2 * (boots + slippers + heels) = 168 →
  ∃ (m : ℕ), heels = m * (boots + slippers) ∧ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_nancy_shoe_multiple_l1096_109645


namespace NUMINAMATH_CALUDE_A_greater_than_B_l1096_109663

theorem A_greater_than_B (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  a^(2*a) * b^(2*b) * c^(2*c) > a^(b+c) * b^(c+a) * c^(a+b) := by
  sorry

end NUMINAMATH_CALUDE_A_greater_than_B_l1096_109663


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1096_109666

theorem right_rectangular_prism_volume
  (x y z : ℝ)
  (h_side : x * y = 15)
  (h_front : y * z = 10)
  (h_bottom : x * z = 6) :
  x * y * z = 30 := by
sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1096_109666


namespace NUMINAMATH_CALUDE_inequality_implies_upper_bound_l1096_109648

theorem inequality_implies_upper_bound (m : ℝ) : 
  (∀ x : ℝ, |x + 4| + |x + 8| ≥ m) → m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_upper_bound_l1096_109648


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1096_109671

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set B
def B : Set ℕ := {1, 2}

-- Theorem statement
theorem intersection_A_complement_B (A : Set ℕ) 
  (h1 : A ⊆ U) 
  (h2 : B ⊆ U) 
  (h3 : (U \ (A ∪ B)) = {4}) : 
  A ∩ (U \ B) = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1096_109671


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l1096_109662

theorem quadratic_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (x₁^2 - 5*x₁ + 11 = x₁ + 53) ∧ 
  (x₂^2 - 5*x₂ + 11 = x₂ + 53) ∧ 
  (x₁ ≠ x₂) ∧
  (|x₁ - x₂| = 2 * Real.sqrt 51) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l1096_109662


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1096_109680

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - Real.cos α) / (3 * Real.sin α + Real.cos α) = 1/7) : 
  Real.tan α = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1096_109680


namespace NUMINAMATH_CALUDE_bridgette_has_three_cats_l1096_109687

/-- Represents the number of baths given to pets in a year -/
def total_baths : ℕ := 96

/-- Represents the number of dogs Bridgette has -/
def num_dogs : ℕ := 2

/-- Represents the number of birds Bridgette has -/
def num_birds : ℕ := 4

/-- Represents the number of baths given to a dog in a year -/
def dog_baths_per_year : ℕ := 24

/-- Represents the number of baths given to a bird in a year -/
def bird_baths_per_year : ℕ := 3

/-- Represents the number of baths given to a cat in a year -/
def cat_baths_per_year : ℕ := 12

/-- Theorem stating that Bridgette has 3 cats -/
theorem bridgette_has_three_cats :
  ∃ (num_cats : ℕ),
    num_cats * cat_baths_per_year = 
      total_baths - (num_dogs * dog_baths_per_year + num_birds * bird_baths_per_year) ∧
    num_cats = 3 :=
by sorry

end NUMINAMATH_CALUDE_bridgette_has_three_cats_l1096_109687


namespace NUMINAMATH_CALUDE_nested_sqrt_power_l1096_109616

theorem nested_sqrt_power (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (15/16) := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_power_l1096_109616


namespace NUMINAMATH_CALUDE_distance_is_six_miles_l1096_109607

/-- The distance between Amanda's house and Kimberly's house -/
def distance_between_houses (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating that given Amanda's speed and travel time, the distance between the houses is 6 miles -/
theorem distance_is_six_miles (amanda_speed : ℝ) (travel_time : ℝ) 
  (h1 : amanda_speed = 2)
  (h2 : travel_time = 3) :
  distance_between_houses amanda_speed travel_time = 6 := by
  sorry

#check distance_is_six_miles

end NUMINAMATH_CALUDE_distance_is_six_miles_l1096_109607


namespace NUMINAMATH_CALUDE_smallest_value_u3_plus_v3_l1096_109635

theorem smallest_value_u3_plus_v3 (u v : ℂ) 
  (h1 : Complex.abs (u + v) = 2) 
  (h2 : Complex.abs (u^2 + v^2) = 11) : 
  Complex.abs (u^3 + v^3) ≥ 14.5 := by
sorry

end NUMINAMATH_CALUDE_smallest_value_u3_plus_v3_l1096_109635


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l1096_109653

/-- The probability of drawing a white ball from a bag with red and white balls -/
theorem probability_of_white_ball (red_balls white_balls : ℕ) :
  red_balls = 3 → white_balls = 5 →
  (white_balls : ℚ) / ((red_balls : ℚ) + (white_balls : ℚ)) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l1096_109653


namespace NUMINAMATH_CALUDE_sqrt_1_minus_x_real_l1096_109614

theorem sqrt_1_minus_x_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = 1 - x) ↔ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_1_minus_x_real_l1096_109614


namespace NUMINAMATH_CALUDE_largest_s_value_l1096_109673

theorem largest_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3) : 
  (r - 2) * s * 61 = (s - 2) * r * 60 → s ≤ 121 ∧ ∃ (r' : ℕ), r' ≥ 121 ∧ (r' - 2) * 121 * 61 = 119 * r' * 60 :=
sorry

end NUMINAMATH_CALUDE_largest_s_value_l1096_109673


namespace NUMINAMATH_CALUDE_stock_value_change_l1096_109652

theorem stock_value_change (initial_value : ℝ) (h : initial_value > 0) : 
  let day1_value := initial_value * (1 - 0.25)
  let day2_value := day1_value * (1 + 0.35)
  (day2_value - initial_value) / initial_value * 100 = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_stock_value_change_l1096_109652


namespace NUMINAMATH_CALUDE_painting_wall_coverage_percentage_l1096_109622

/-- Represents the dimensions of a rectangular painting -/
structure PaintingDimensions where
  length : ℚ
  width : ℚ

/-- Represents the dimensions of an irregular pentagonal wall -/
structure WallDimensions where
  side1 : ℚ
  side2 : ℚ
  side3 : ℚ
  side4 : ℚ
  side5 : ℚ
  height : ℚ

/-- Calculates the area of a rectangular painting -/
def paintingArea (p : PaintingDimensions) : ℚ :=
  p.length * p.width

/-- Calculates the approximate area of the irregular pentagonal wall -/
def wallArea (w : WallDimensions) : ℚ :=
  (w.side3 * w.height) / 2

/-- Calculates the percentage of the wall covered by the painting -/
def coveragePercentage (p : PaintingDimensions) (w : WallDimensions) : ℚ :=
  (paintingArea p / wallArea w) * 100

/-- Theorem stating that the painting covers approximately 39.21% of the wall -/
theorem painting_wall_coverage_percentage :
  let painting := PaintingDimensions.mk (13/4) (38/5)
  let wall := WallDimensions.mk 4 12 14 10 8 9
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ 
    abs (coveragePercentage painting wall - 39.21) < ε :=
sorry

end NUMINAMATH_CALUDE_painting_wall_coverage_percentage_l1096_109622


namespace NUMINAMATH_CALUDE_f_form_m_range_l1096_109686

/-- A quadratic function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The properties of the quadratic function f -/
axiom f_diff (x : ℝ) : f (x + 1) - f x = 2 * x
axiom f_zero : f 0 = 1

/-- Theorem: The quadratic function f has the form x^2 - x + 1 -/
theorem f_form (x : ℝ) : f x = x^2 - x + 1 := sorry

/-- Theorem: If f(x) > 2x + m holds for all x in [-1, 1], then m < -1 -/
theorem m_range (m : ℝ) (h : ∀ x ∈ Set.Icc (-1) 1, f x > 2 * x + m) : m < -1 := sorry

end NUMINAMATH_CALUDE_f_form_m_range_l1096_109686


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_specific_hyperbola_eccentricity_l1096_109639

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²) / a -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (a^2 + b^2) / a
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  ∀ x y, hyperbola x y → e = Real.sqrt 5 / 2 :=
by
  sorry

/-- The eccentricity of the hyperbola x²/4 - y² = 1 is √5/2 -/
theorem specific_hyperbola_eccentricity :
  let e := Real.sqrt 5 / 2
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 4 - y^2 = 1
  ∀ x y, hyperbola x y → e = Real.sqrt 5 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_specific_hyperbola_eccentricity_l1096_109639


namespace NUMINAMATH_CALUDE_geometric_area_ratios_l1096_109630

theorem geometric_area_ratios (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let triangle_area := s^2 / 2
  let circle_area := π * (s/2)^2
  let small_square_area := (s/2)^2
  (triangle_area / square_area = 1/2) ∧
  (circle_area / square_area = π/4) ∧
  (small_square_area / square_area = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_geometric_area_ratios_l1096_109630


namespace NUMINAMATH_CALUDE_tangent_line_to_ellipse_problem_g1_1_l1096_109619

/-- The curve and line intersect at only one point if and only if
    the discriminant of the resulting quadratic equation is zero. -/
theorem tangent_line_to_ellipse (m : ℝ) :
  (∃! p : ℝ × ℝ, p.1^2 + 3*p.2^2 = 12 ∧ m*p.1 + p.2 = 16) ↔
  (9216*m^2 - 3072*(1 + 3*m^2) = 0) :=
sorry

/-- If the curve x^2 + 3y^2 = 12 and the line mx + y = 16 intersect at only one point,
    and a = m^2, then a = 21. -/
theorem problem_g1_1 (m : ℝ) (a : ℝ) 
  (h1 : ∃! p : ℝ × ℝ, p.1^2 + 3*p.2^2 = 12 ∧ m*p.1 + p.2 = 16)
  (h2 : a = m^2) :
  a = 21 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_ellipse_problem_g1_1_l1096_109619


namespace NUMINAMATH_CALUDE_car_travel_time_l1096_109669

/-- Proves that a car with given specifications traveling for a certain time uses the specified amount of fuel -/
theorem car_travel_time (speed : ℝ) (fuel_efficiency : ℝ) (tank_capacity : ℝ) (fuel_used_ratio : ℝ) : 
  speed = 40 →
  fuel_efficiency = 40 →
  tank_capacity = 12 →
  fuel_used_ratio = 0.4166666666666667 →
  (fuel_used_ratio * tank_capacity * fuel_efficiency) / speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_time_l1096_109669


namespace NUMINAMATH_CALUDE_parenthesization_pigeonhole_l1096_109650

theorem parenthesization_pigeonhole : ∃ (n : ℕ) (k : ℕ), 
  n > 0 ∧ 
  k > 0 ∧ 
  (2 ^ n > (k * (k + 1))) ∧ 
  (∀ (f : Fin (2^n) → ℤ), ∃ (i j : Fin (2^n)), i ≠ j ∧ f i = f j) := by
  sorry

end NUMINAMATH_CALUDE_parenthesization_pigeonhole_l1096_109650


namespace NUMINAMATH_CALUDE_nebula_boys_count_total_students_correct_total_students_by_gender_correct_l1096_109659

/-- Represents a school in the science camp. -/
inductive School
| Orion
| Nebula
| Galaxy

/-- Represents the gender of a student. -/
inductive Gender
| Boy
| Girl

/-- Represents the distribution of students in the science camp. -/
structure CampDistribution where
  total_students : ℕ
  total_boys : ℕ
  total_girls : ℕ
  students_by_school : School → ℕ
  boys_by_school : School → ℕ

/-- The actual distribution of students in the science camp. -/
def camp_data : CampDistribution :=
  { total_students := 150,
    total_boys := 84,
    total_girls := 66,
    students_by_school := fun s => match s with
      | School.Orion => 70
      | School.Nebula => 50
      | School.Galaxy => 30,
    boys_by_school := fun s => match s with
      | School.Orion => 30
      | _ => 0  -- We don't know these values yet
  }

/-- Theorem stating that the number of boys from Nebula Middle School is 32. -/
theorem nebula_boys_count (d : CampDistribution) (h : d = camp_data) :
  d.boys_by_school School.Nebula = 32 := by
  sorry

/-- Verify that the total number of students is correct. -/
theorem total_students_correct (d : CampDistribution) (h : d = camp_data) :
  d.total_students = d.students_by_school School.Orion +
                     d.students_by_school School.Nebula +
                     d.students_by_school School.Galaxy := by
  sorry

/-- Verify that the total number of students by gender is correct. -/
theorem total_students_by_gender_correct (d : CampDistribution) (h : d = camp_data) :
  d.total_students = d.total_boys + d.total_girls := by
  sorry

end NUMINAMATH_CALUDE_nebula_boys_count_total_students_correct_total_students_by_gender_correct_l1096_109659
