import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_exit_path_exists_l995_99519

/-- A forest in the shape of a long strip -/
structure StripForest where
  width : ℝ
  width_pos : width > 0

/-- A path in the forest -/
structure ForestPath where
  length : ℝ
  exits_forest : Bool

/-- Circular path with radius half the forest width -/
noncomputable def circular_path (f : StripForest) : ForestPath :=
  { length := Real.pi * f.width,
    exits_forest := true }

/-- Theorem stating the existence of a shorter exit path -/
theorem shorter_exit_path_exists (f : StripForest) : 
  ∃ (p : ForestPath), p.exits_forest ∧ p.length < 2.5 * f.width := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_exit_path_exists_l995_99519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_satisfies_equation_l995_99524

/-- Parametric curve in R^2 -/
noncomputable def curve (t : ℝ) : ℝ × ℝ := (3 * Real.cos t + 2 * Real.sin t, 3 * Real.sin t)

/-- The equation of the curve in Cartesian coordinates -/
def curve_equation (x y : ℝ) : Prop :=
  ∃ t : ℝ, curve t = (x, y)

/-- The values of a, b, and c in the equation ax^2 + bxy + cy^2 = 9 -/
noncomputable def a : ℝ := 1/9
noncomputable def b : ℝ := -4/27
noncomputable def c : ℝ := 23/243

theorem curve_satisfies_equation :
  ∀ x y : ℝ, curve_equation x y →
    a * x^2 + b * x * y + c * y^2 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_satisfies_equation_l995_99524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excluded_students_average_mark_l995_99586

theorem excluded_students_average_mark
  (total_students : ℕ)
  (total_average : ℝ)
  (excluded_students : ℕ)
  (remaining_average : ℝ)
  (h1 : total_students = 15)
  (h2 : total_average = 80)
  (h3 : excluded_students = 5)
  (h4 : remaining_average = 90) :
  let remaining_students := total_students - excluded_students
  let total_marks := total_students * total_average
  let remaining_marks := remaining_students * remaining_average
  let excluded_marks := total_marks - remaining_marks
  excluded_marks / excluded_students = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excluded_students_average_mark_l995_99586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_area_l995_99589

theorem cone_lateral_area (r l : ℝ) (h_r : r = 3) (h_l : l = 5) :
  π * r * l = 15 * π := by
  rw [h_r, h_l]
  ring

#check cone_lateral_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_area_l995_99589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l995_99540

-- Define the line
def line (x : ℝ) : ℝ := 1 - x

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Theorem statement
theorem intersection_distance :
  ∃ (B C : ℝ × ℝ),
    circle_eq B.1 B.2 ∧
    circle_eq C.1 C.2 ∧
    B.2 = line B.1 ∧
    C.2 = line C.1 ∧
    B ≠ C ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = 30 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l995_99540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_three_correct_probability_l995_99544

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of derangements of n items -/
def derangement (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | 1 => 0
  | n + 2 => (n + 1) * (derangement (n + 1) + derangement n)

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := n.factorial

/-- The probability of exactly 3 people getting the right letter when 6 letters are randomly distributed to 6 people -/
theorem exact_three_correct_probability :
  (choose 6 3 * derangement 3 : ℚ) / factorial 6 = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_three_correct_probability_l995_99544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_radius_formula_l995_99506

/-- The radius of a circle tangent to two given circles and their line of centers -/
noncomputable def tangent_circle_radius (R r : ℝ) (internal : Bool) : ℝ :=
  if internal then
    (4 * R * r * (R - r)) / ((R + r)^2)
  else
    (4 * R * r * (R + r)) / ((R - r)^2)

/-- Theorem stating the formula for the radius of a circle tangent to two given circles and their line of centers -/
theorem tangent_circle_radius_formula (R r : ℝ) (h : R > r) (internal : Bool) :
  ∃ (r₀ : ℝ), r₀ = tangent_circle_radius R r internal ∧ r₀ > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_radius_formula_l995_99506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l995_99564

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_points : distance (1, 5) (4, 1) = 5 := by
  -- Unfold the definition of distance
  unfold distance
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l995_99564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_condition_l995_99529

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*(a-1)*x + 2

-- Define the interval (-∞, 5]
def interval : Set ℝ := Set.Iic 5

-- State the theorem
theorem decreasing_function_condition (a : ℝ) : 
  (∀ x ∈ interval, ∀ y ∈ interval, x < y → f a x > f a y) ↔ a ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_condition_l995_99529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_restoration_theorem_l995_99516

/-- Calculates the price after multiple reductions --/
noncomputable def price_after_reductions (initial_price : ℝ) (reductions : List ℝ) : ℝ :=
  reductions.foldl (fun price reduction => price * (1 - reduction / 100)) initial_price

/-- Calculates the percentage increase needed to restore the original price --/
noncomputable def percentage_increase_needed (final_price : ℝ) (original_price : ℝ) : ℝ :=
  (original_price / final_price - 1) * 100

theorem price_restoration_theorem (initial_price : ℝ) (h_positive : initial_price > 0) :
  let final_price := price_after_reductions initial_price [10, 15, 25]
  let increase_needed := percentage_increase_needed final_price initial_price
  abs (increase_needed - 74.3) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_restoration_theorem_l995_99516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l995_99553

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

def arithmetic_sequence (b : ℕ → ℝ) := ∀ n, b (n + 1) - b n = b 2 - b 1

def S (n : ℕ) : ℝ := n^2 + 2*n

theorem sequence_properties (a b : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 = 1) 
    (h3 : a 5 = 27) (h4 : arithmetic_sequence b) (h5 : b 1 = a 3) (h6 : b 4 = a 4) :
  (∀ n, a n = 3^(n-2)) ∧ 
  (∀ n, b n = 2*n + 1) ∧ 
  (∀ n, (Finset.range n).sum (λ i ↦ 2 / S (i+1)) = 3/2 - 1/(n+1) - 1/(n+2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l995_99553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2NQO_value_l995_99575

-- Define the points
variable (M N O P Q : ℝ × ℝ)

-- Define the angles
noncomputable def angle_MQO : ℝ := Real.arccos ((3 : ℝ) / 5)
noncomputable def angle_NQP : ℝ := Real.arccos ((1 : ℝ) / 2)
noncomputable def angle_NQO (N O Q : ℝ × ℝ) : ℝ := 
  Real.arccos ((Q.1 - N.1) * (Q.1 - O.1) + (Q.2 - N.2) * (Q.2 - O.2)) / 
    (Real.sqrt ((Q.1 - N.1)^2 + (Q.2 - N.2)^2) * Real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2))

-- State the theorem
theorem sin_2NQO_value (h1 : N.1 - M.1 = O.1 - N.1) (h2 : O.1 - N.1 = P.1 - O.1)
                       (h3 : M.2 = N.2) (h4 : N.2 = O.2) (h5 : O.2 = P.2)
                       (h6 : Real.cos angle_MQO = 3/5) (h7 : Real.cos angle_NQP = 1/2) :
  Real.sin (2 * angle_NQO N O Q) = 6 * Real.sqrt 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2NQO_value_l995_99575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_square_area_ratio_l995_99582

noncomputable section

open Real

-- Define the rectangle dimensions
def rectangle_width : ℝ := 8
def rectangle_length : ℝ := 12

-- Define the square side length
def square_side : ℝ := rectangle_width

-- Define the areas of semicircles
def large_semicircle_area : ℝ := π * (rectangle_length / 2)^2
def small_semicircle_area : ℝ := π * (rectangle_width / 2)^2

-- Define the total area of semicircles
def total_semicircle_area : ℝ := 2 * large_semicircle_area + 2 * small_semicircle_area

-- Define the area of the square
def square_area : ℝ := square_side^2

-- Theorem statement
theorem semicircle_square_area_ratio :
  total_semicircle_area / square_area = 13 * π / 16 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_square_area_ratio_l995_99582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalence_point_exists_sum_product_roots_sum_squares_roots_equivalence_points_W_l995_99510

-- Definition of an equivalence point
noncomputable def is_equivalence_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Part (1)
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)

theorem equivalence_point_exists :
  ∃ x > 1, is_equivalence_point f x :=
sorry

-- Part (2)
noncomputable def g (x : ℝ) : ℝ := x^2 - 2

-- Part (2) ①
theorem sum_product_roots (m n : ℝ) :
  m^2 - m - 2 = 0 → n^2 - n - 2 = 0 → m ≠ n → m^2 * n + m * n^2 = -2 :=
sorry

-- Part (2) ②
theorem sum_squares_roots (p q : ℝ) :
  p^2 = p + 2 → 2 * q^2 = q + 1 → p ≠ 2 * q → p^2 + 4 * q^2 = 5 :=
sorry

-- Part (2) ③
def W₁ (x : ℝ) : Prop := x ≥ 1 ∧ g x = x
def W₂ (x : ℝ) : Prop := x ≤ 1 ∧ g (2 - x) = x

theorem equivalence_points_W :
  (∃ x, (W₁ x ∨ W₂ x) ∧ is_equivalence_point g x) ↔
  (∃ x, x = 2 ∨ x = (5 - Real.sqrt 17) / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalence_point_exists_sum_product_roots_sum_squares_roots_equivalence_points_W_l995_99510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_infinite_integer_solutions_l995_99592

/-- A function that generates integer solutions to x^2 + y^2 + z^2 = x^3 + y^3 + z^3 -/
def solution_generator (k : ℤ) : ℤ × ℤ × ℤ :=
  (k * (2 * k^2 + 1), 2 * k^2 + 1, -k * (2 * k^2 + 1))

/-- Theorem stating that the equation x^2 + y^2 + z^2 = x^3 + y^3 + z^3 has infinitely many integer solutions -/
theorem infinite_solutions :
  ∀ k : ℤ, let (x, y, z) := solution_generator k
    x^2 + y^2 + z^2 = x^3 + y^3 + z^3 := by
  sorry

/-- Corollary: There are infinitely many integer solutions to x^2 + y^2 + z^2 = x^3 + y^3 + z^3 -/
theorem infinite_integer_solutions :
  ∃ f : ℤ → ℤ × ℤ × ℤ, ∀ k : ℤ,
    let (x, y, z) := f k
    x^2 + y^2 + z^2 = x^3 + y^3 + z^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_infinite_integer_solutions_l995_99592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baoh2_formation_l995_99571

/-- Represents the chemical reaction between Barium oxide and Water to form Barium hydroxide -/
structure ChemicalReaction where
  bao_mass : ℚ  -- Mass of Barium oxide in grams
  h2o_moles : ℚ  -- Moles of Water
  molar_ratio : ℚ  -- Molar ratio of reactants to products
  bao_molar_mass : ℚ  -- Molar mass of Barium oxide in g/mol

/-- Calculates the moles of Barium hydroxide formed in the reaction -/
def baoh2_moles_formed (reaction : ChemicalReaction) : ℚ :=
  (reaction.bao_mass / reaction.bao_molar_mass) * reaction.molar_ratio

/-- Theorem stating that 2 moles of Barium hydroxide are formed in the given reaction -/
theorem baoh2_formation (reaction : ChemicalReaction) 
  (h1 : reaction.bao_mass = 306)
  (h2 : reaction.h2o_moles = 2)
  (h3 : reaction.molar_ratio = 1)
  (h4 : reaction.bao_molar_mass = 153333 / 1000) :
  baoh2_moles_formed reaction = 2 := by
  sorry

#eval baoh2_moles_formed { bao_mass := 306, h2o_moles := 2, molar_ratio := 1, bao_molar_mass := 153333 / 1000 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baoh2_formation_l995_99571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l995_99520

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties 
  (a b c : ℝ) 
  (h1 : f a b c 0 = 2) 
  (h2 : ∀ x, f a b c (x + 1) - f a b c x = 2 * x - 1) :
  (∃ a' b' c', ∀ x, f a' b' c' x = x^2 - 2*x + 2) ∧ 
  (∀ t, (∃ x ∈ Set.Icc (-1) 2, f a b c x - t > 0) ↔ t < 5) ∧
  (∀ m, (∃ x₁ ∈ Set.Ioo (-1) 2, f a b c x₁ - m * x₁ = 0) ∧ 
        (∃ x₂ ∈ Set.Ioo 2 4, f a b c x₂ - m * x₂ = 0) ↔ 
        1 < m ∧ m < 5/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l995_99520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l995_99545

/-- Given points P and Q, and a line l that intersects the line segment PQ,
    prove that the range of real number a in the line equation is a ≤ 1 or a ≥ 3 -/
theorem intersection_range (P Q : ℝ × ℝ) (a : ℝ) :
  P = (3, -1) →
  Q = (-1, 2) →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    let (x, y) := (1 - t) • P + t • Q
    a * x + 2 * y - 1 = 0) →
  a ≤ 1 ∨ a ≥ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l995_99545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_tan_a_l995_99598

theorem right_triangle_tan_a (A B C : Real) (h1 : 0 < A ∧ A < Real.pi/2) 
  (h2 : 0 < B ∧ B < Real.pi/2) (h3 : A + B = Real.pi/2) 
  (h4 : Real.sin B = 3/5) : Real.tan A = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_tan_a_l995_99598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_m_range_perpendicular_intersection_m_value_l995_99518

-- Define the circle equation
def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 + x - 6*y + m = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Part I
theorem no_intersection_m_range (m : ℝ) :
  (∀ x y : ℝ, ¬(circle_eq x y m ∧ line_eq x y)) → m > 8 ∧ m < 37/4 :=
sorry

-- Part II
theorem perpendicular_intersection_m_value :
  (∃ p q : ℝ × ℝ, 
    circle_eq p.1 p.2 3 ∧ line_eq p.1 p.2 ∧
    circle_eq q.1 q.2 3 ∧ line_eq q.1 q.2 ∧
    p ≠ q ∧
    p.1 * q.1 + p.2 * q.2 = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_m_range_perpendicular_intersection_m_value_l995_99518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l995_99502

/-- Given vectors a, b, and c in ℝ³, prove that lambda = 4 when c = 2a + b -/
theorem vector_equation_solution (a b c : ℝ × ℝ × ℝ) (lambda : ℝ) : 
  a = (2, -1, 3) →
  b = (-1, 4, -2) →
  c = (3, 2, lambda) →
  c = (2 • a) + b →
  lambda = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l995_99502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_max_l995_99574

theorem triangle_cosine_sum_max (A B C : ℝ) (h : A + B + C = Real.pi) : 
  Real.cos A + Real.cos B * Real.cos C ≤ 5/2 ∧ 
  ∃ A B C, A + B + C = Real.pi ∧ Real.cos A + Real.cos B * Real.cos C = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_max_l995_99574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_exponents_l995_99500

theorem order_of_exponents (a b c : ℝ) : 
  a = (2 : ℝ)^(1/5) → b = (2/5 : ℝ)^(1/5) → c = (2/5 : ℝ)^(3/5) → a > b ∧ b > c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_exponents_l995_99500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l995_99525

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

/-- The slope of a line in the form ax + by + c = 0 is -a/b -/
noncomputable def line_slope (a b c : ℝ) : ℝ := -a / b

theorem parallel_lines_a_value (a : ℝ) :
  are_parallel (line_slope 1 a (-2*a - 2)) (line_slope a 1 (-a - 1)) → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l995_99525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_selling_price_l995_99567

/-- Calculates the selling price of an article given the cost price and profit percentage -/
noncomputable def selling_price (cost_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  cost_price * (1 + profit_percentage / 100)

/-- Proves that the selling price of an article is 250 given the specified conditions -/
theorem article_selling_price :
  let cost_price : ℝ := 200
  let profit_percentage : ℝ := 25
  selling_price cost_price profit_percentage = 250 := by
  -- Unfold the definition of selling_price
  unfold selling_price
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_selling_price_l995_99567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ed_rates_sum_of_squares_l995_99539

/-- The sum of the squares of Ed's traipsing, cycling, and skating rates is 995 -/
theorem ed_rates_sum_of_squares (t c k : ℕ) : 
  (t + 4 * c + 3 * k = 60) →
  (4 * t + 3 * c + 2 * k = 86) →
  (t^2 + c^2 + k^2 = 995) :=
by
  sorry

#check ed_rates_sum_of_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ed_rates_sum_of_squares_l995_99539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kain_can_force_win_l995_99528

/-- Represents the state of the game board -/
def GameBoard := List Int

/-- Represents a move in the game -/
structure Move where
  start : Nat
  length : Nat
  add : Bool

/-- The game state -/
structure GameState where
  board : GameBoard
  moves : List Move

/-- Check if a number is divisible by 4 -/
def isDivisibleByFour (n : Int) : Bool :=
  n % 4 = 0

/-- Count how many numbers on the board are divisible by 4 -/
def countDivisibleByFour (board : GameBoard) : Nat :=
  board.filter isDivisibleByFour |>.length

/-- Apply a move to the game board -/
def applyMove (board : GameBoard) (move : Move) : GameBoard :=
  sorry

/-- Apply a list of moves to the game board -/
def applyMoves (board : GameBoard) (moves : List Move) : GameBoard :=
  moves.foldl applyMove board

/-- Check if the game is won (98 or more numbers divisible by 4) -/
def isGameWon (state : GameState) : Bool :=
  countDivisibleByFour state.board ≥ 98

/-- The main theorem: Kain can force a win in a finite number of moves -/
theorem kain_can_force_win (initialBoard : GameBoard) 
    (h : initialBoard.length = 100) : 
    ∃ (n : Nat) (moves : List Move), 
      moves.length ≤ n ∧ 
      isGameWon ⟨applyMoves initialBoard moves, moves⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kain_can_force_win_l995_99528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_coefficients_theorem_l995_99580

open Real

-- Define a continuous, 2π-periodic function on the real line
def ContinuousPeriodicFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ ∀ x, f (x + 2 * Real.pi) = f x

-- Define Fourier coefficients
def FourierCoefficients (f : ℝ → ℝ) (α₀ : ℝ) (α β : ℕ → ℝ) : Prop :=
  ContinuousPeriodicFunction f ∧
  α₀ = (1 / Real.pi) * ∫ x in -Real.pi..Real.pi, f x ∧
  ∀ n : ℕ, n ≥ 1 →
    α n = (1 / Real.pi) * ∫ x in -Real.pi..Real.pi, f x * cos (n * x) ∧
    β n = (1 / Real.pi) * ∫ x in -Real.pi..Real.pi, f x * sin (n * x)

-- Theorem statement
theorem fourier_coefficients_theorem (f : ℝ → ℝ) :
  ContinuousPeriodicFunction f →
  ∃ α₀ : ℝ, ∃ α β : ℕ → ℝ, FourierCoefficients f α₀ α β :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_coefficients_theorem_l995_99580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_when_line_passes_focus_midpoint_coordinates_l995_99573

-- Define the line l
noncomputable def line_l (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the parabola C
noncomputable def parabola_C (x y p : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus of a parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define symmetry about a line
noncomputable def symmetric_about_line (P Q : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (M : ℝ × ℝ), l M.1 M.2 ∧ M = ((P.1 + Q.1)/2, (P.2 + Q.2)/2)

-- Theorem 1
theorem parabola_equation_when_line_passes_focus :
  ∀ (p : ℝ), (line_l (focus p).1 (focus p).2) → 
  (∀ (x y : ℝ), parabola_C x y p ↔ y^2 = 8*x) :=
by
  sorry

-- Theorem 2
theorem midpoint_coordinates :
  ∀ (P Q : ℝ × ℝ),
    P ≠ Q →
    parabola_C P.1 P.2 1 →
    parabola_C Q.1 Q.2 1 →
    symmetric_about_line P Q line_l →
    ((P.1 + Q.1)/2, (P.2 + Q.2)/2) = (1, -1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_when_line_passes_focus_midpoint_coordinates_l995_99573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_error_is_ten_percent_l995_99541

/-- Represents a rectangle with a measured length and exact width -/
structure Rectangle where
  measured_length : ℝ
  width : ℝ
  length_error_rate : ℝ

/-- Calculates the maximum percent error in the area of a rectangle -/
noncomputable def max_area_percent_error (rect : Rectangle) : ℝ :=
  let true_min_length := rect.measured_length * (1 - rect.length_error_rate)
  let true_max_length := rect.measured_length * (1 + rect.length_error_rate)
  let measured_area := rect.measured_length * rect.width
  let min_area := true_min_length * rect.width
  let max_area := true_max_length * rect.width
  max ((measured_area - min_area) / measured_area) ((max_area - measured_area) / measured_area) * 100

/-- Theorem stating that the maximum percent error in area is 10% for the given rectangle -/
theorem max_error_is_ten_percent :
  let rect : Rectangle := { measured_length := 30, width := 15, length_error_rate := 0.1 }
  max_area_percent_error rect = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_error_is_ten_percent_l995_99541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_coverage_for_given_cube_l995_99513

/-- The coverage of paint in square feet per kilogram, given the cost to paint a cube and the cost of paint per kilogram. -/
noncomputable def paint_coverage (cube_side_length : ℝ) (cube_paint_cost : ℝ) (paint_cost_per_kg : ℝ) : ℝ :=
  (6 * cube_side_length^2) / (cube_paint_cost / paint_cost_per_kg)

/-- Theorem stating that for a 10-foot cube costing $1800 to paint and paint costing $60 per kg, the coverage is 20 sq ft per kg. -/
theorem paint_coverage_for_given_cube :
  paint_coverage 10 1800 60 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_coverage_for_given_cube_l995_99513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l995_99568

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the vectors
noncomputable def m (t : Triangle) : ℝ × ℝ := (t.b, Real.sqrt 3 * t.a)
noncomputable def n (t : Triangle) : ℝ × ℝ := (Real.cos t.B, Real.sin t.A)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h_parallel : m t = n t) 
  (h_b : t.b = 2)
  (h_area : 1/2 * t.a * t.c * Real.sin t.B = Real.sqrt 3) : 
  t.B = π/3 ∧ t.a + t.c = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l995_99568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mall_audit_is_systematic_l995_99596

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Other

/-- Represents the characteristics of the sampling method used --/
structure SamplingCharacteristics where
  sample_percentage : ℚ
  interval : ℕ
  random_start : Bool
  fixed_interval : Bool

/-- Defines systematic sampling --/
def is_systematic_sampling (sc : SamplingCharacteristics) : Prop :=
  sc.sample_percentage > 0 ∧ 
  sc.interval > 1 ∧
  sc.random_start ∧
  sc.fixed_interval

/-- The sampling method used in the mall audit --/
def mall_audit_sampling : SamplingCharacteristics :=
  { sample_percentage := 1/50,  -- 2% = 1/50
    interval := 25,
    random_start := true,
    fixed_interval := true }

/-- Theorem stating that the mall audit sampling method is systematic sampling --/
theorem mall_audit_is_systematic : 
  is_systematic_sampling mall_audit_sampling ∧ 
  SamplingMethod.Systematic = 
    (if mall_audit_sampling.random_start ∧ mall_audit_sampling.fixed_interval
     then SamplingMethod.Systematic
     else SamplingMethod.Other) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mall_audit_is_systematic_l995_99596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_calculation_l995_99581

/-- The weight of an equilateral triangle with side length s and weight w -/
structure TriangleWeight where
  side : ℝ
  weight : ℝ

/-- The weight of a circular piece with diameter d and weight w -/
structure CircleWeight where
  diameter : ℝ
  weight : ℝ

/-- Calculates the total weight of a larger triangle and a circular piece given the weight of a smaller triangle -/
noncomputable def totalWeight (t1 : TriangleWeight) (t2 : TriangleWeight) (c : CircleWeight) : ℝ :=
  (t1.weight * (t2.side / t1.side)^2) + c.weight

theorem weight_calculation (t1 : TriangleWeight) (t2 : TriangleWeight) (c : CircleWeight)
  (h1 : t1.side = 4)
  (h2 : t1.weight = 18)
  (h3 : t2.side = 6)
  (h4 : c.diameter = 4)
  (h5 : c.weight = 15) :
  totalWeight t1 t2 c = 55.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_calculation_l995_99581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABC_is_30_degrees_l995_99549

-- Define the vectors
noncomputable def vector_BA : ℝ × ℝ := (1/2, Real.sqrt 3/2)
noncomputable def vector_BC : ℝ × ℝ := (Real.sqrt 3/2, 1/2)

-- Define the angle between the vectors
noncomputable def angle_ABC : ℝ := Real.arccos ((vector_BA.1 * vector_BC.1 + vector_BA.2 * vector_BC.2) / 
  (Real.sqrt (vector_BA.1^2 + vector_BA.2^2) * Real.sqrt (vector_BC.1^2 + vector_BC.2^2)))

-- Theorem statement
theorem angle_ABC_is_30_degrees : 
  angle_ABC = π / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABC_is_30_degrees_l995_99549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barrels_in_ton_is_100_l995_99548

/-- The weight of a barrel of oil in kilograms -/
noncomputable def barrel_weight : ℚ := 10

/-- The weight of a ton in kilograms -/
noncomputable def ton_weight : ℚ := 1000

/-- The number of barrels of oil that weigh one ton -/
noncomputable def barrels_in_ton : ℚ := ton_weight / barrel_weight

theorem barrels_in_ton_is_100 : barrels_in_ton = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barrels_in_ton_is_100_l995_99548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_distances_l995_99532

theorem sum_of_possible_distances (p q r s t : ℝ) 
  (h1 : |p - q| = 1)
  (h2 : |q - r| = 2)
  (h3 : |r - s| = 3)
  (h4 : |s - t| = 5) :
  ∃ S : Finset ℝ, (∀ x ∈ S, ∃ p' t' : ℝ, 
    (|p' - q| = 1 ∧ |q - r| = 2 ∧ |r - s| = 3 ∧ |s - t'| = 5 ∧ |p' - t'| = x)) ∧ 
    (S.sum id = 32) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_distances_l995_99532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_discount_l995_99521

theorem equivalent_discount (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) 
  (equivalent_discount : ℝ) : 
  original_price = 50 →
  discount1 = 0.25 →
  discount2 = 0.10 →
  equivalent_discount = 0.325 →
  original_price * (1 - equivalent_discount) = 
    original_price * (1 - discount1) * (1 - discount2) := by
  intros h1 h2 h3 h4
  -- The proof steps would go here
  sorry

#check equivalent_discount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_discount_l995_99521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_pourings_is_six_glass_pouring_theorem_l995_99570

/-- Represents the state of the glasses --/
inductive GlassState
  | Full
  | Empty

/-- Represents the system of 4 glasses --/
structure GlassSystem :=
  (glass1 : GlassState)
  (glass2 : GlassState)
  (glass3 : GlassState)
  (glass4 : GlassState)

/-- Initial state of the system --/
def initialState : GlassSystem :=
  { glass1 := GlassState.Full,
    glass2 := GlassState.Empty,
    glass3 := GlassState.Full,
    glass4 := GlassState.Empty }

/-- Final state of the system --/
def finalState : GlassSystem :=
  { glass1 := GlassState.Empty,
    glass2 := GlassState.Full,
    glass3 := GlassState.Empty,
    glass4 := GlassState.Full }

/-- Represents a pouring action --/
def pour (fromGlass toGlass : Fin 4) (s : GlassSystem) : GlassSystem :=
  sorry  -- Implementation details omitted

/-- Probability of a specific pouring action --/
noncomputable def pourProbability : ℝ := 1 / 4

/-- Expected number of pourings to reach the final state --/
noncomputable def expectedPourings : ℝ := 6

/-- Theorem stating the expected number of pourings --/
theorem expected_pourings_is_six :
  expectedPourings = 6 :=
by sorry

/-- Main theorem to prove --/
theorem glass_pouring_theorem :
  ∃ (n : ℕ), (initialState = finalState) ∨ 
  (∃ (sequence : Fin n → Fin 4 × Fin 4), 
    (∀ i : Fin n, pourProbability = 1 / 4) ∧
    (List.foldl (λ s (p : Fin 4 × Fin 4) => pour p.1 p.2 s) initialState (List.ofFn sequence) = finalState)) ∧
  expectedPourings = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_pourings_is_six_glass_pouring_theorem_l995_99570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_problem_l995_99591

open Set

theorem complement_union_problem (U A B : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 2, 3})
  (hB : B = {2, 3, 4}) :
  (U \ A) ∪ (U \ B) = {1, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_problem_l995_99591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_club_members_count_l995_99590

/-- Represents a club with committees and members -/
structure Club where
  committees : Finset (Finset Nat)
  members : Finset Nat
  member_committees : Nat → Finset (Finset Nat)

/-- The conditions for the club structure -/
def ClubConditions (c : Club) : Prop :=
  (c.committees.card = 5) ∧ 
  (∀ m, m ∈ c.members → (c.member_committees m).card = 2) ∧
  (∀ comm1 comm2, comm1 ∈ c.committees → comm2 ∈ c.committees → comm1 ≠ comm2 → 
    (∃! m, m ∈ c.members ∧ m ∈ comm1 ∧ m ∈ comm2))

/-- The theorem stating that a club satisfying the conditions has 10 members -/
theorem club_members_count (c : Club) (h : ClubConditions c) : c.members.card = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_club_members_count_l995_99590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_perpendicular_distance_l995_99507

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ

/-- Point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Definition of the specific parabola y^2 = 4x -/
def standard_parabola : Parabola :=
  { focus := (1, 0),
    directrix := -1 }

/-- Theorem: For the parabola y^2 = 4x, if P is a point on the parabola
    such that PA ⊥ PF, then |PF| = √5 - 1 -/
theorem parabola_perpendicular_distance 
  (P : ParabolaPoint) 
  (h_on_parabola : P.y^2 = 4 * P.x) 
  (h_perp : (P.x + 1) * (P.x - 1) + P.y * P.y = 0) : 
  Real.sqrt ((P.x - 1)^2 + P.y^2) = Real.sqrt 5 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_perpendicular_distance_l995_99507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relation_between_x_and_y_l995_99542

theorem relation_between_x_and_y (p : ℝ) :
  let x := 1 + (3 : ℝ)^p
  let y := 1 + (3 : ℝ)^(-p)
  y = x / (x - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relation_between_x_and_y_l995_99542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_red_cells_correct_l995_99566

/-- A configuration of red cells on an infinite grid -/
structure RedCellConfiguration where
  n : ℕ
  n_ge_2 : n ≥ 2

/-- A special grid array -/
structure SpecialGridArray where
  red_cells : ℕ
  is_special : Bool

/-- The maximum number of red cells in a special grid array for a given configuration -/
def max_red_cells_in_special_array (config : RedCellConfiguration) : ℕ := sorry

/-- The minimum value of the maximum number of red cells in a special grid array over all configurations -/
def min_max_red_cells (n : ℕ) : ℕ := 1 + Int.toNat (Int.ceil ((n + 1) / 5 : ℚ))

theorem min_max_red_cells_correct (config : RedCellConfiguration) :
  max_red_cells_in_special_array config ≥ min_max_red_cells config.n ∧
  ∃ (optimal_config : RedCellConfiguration),
    optimal_config.n = config.n ∧
    max_red_cells_in_special_array optimal_config = min_max_red_cells config.n := by
  sorry

#check min_max_red_cells_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_red_cells_correct_l995_99566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l995_99531

/-- Given a hyperbola G with center at the origin, passing through (√5, 4),
    and having its right vertex at (1, 0) (which is the focus of y² = 4x),
    prove that the equation of G is x² - y²/4 = 1 -/
theorem hyperbola_equation :
  ∀ G : Set (ℝ × ℝ),
  (∀ p : ℝ × ℝ, p ∈ G ↔ (p.1^2 - p.2^2/4 = 1)) ↔
  (-- G has center at origin
   (0, 0) ∈ G ∧
   -- G passes through (√5, 4)
   (Real.sqrt 5, 4) ∈ G ∧
   -- Right vertex of G is at (1, 0), which is focus of y² = 4x
   (1, 0) ∈ G ∧
   (∀ p : ℝ × ℝ, p.2^2 = 4*p.1 → p ∈ G → p.1 = 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l995_99531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_α_f_minimum_on_interval_l995_99561

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * (Real.sqrt 3 * Real.cos x + Real.sin x) - 2

-- Define the angle α based on the given point P(√3, -1)
noncomputable def α : ℝ := Real.arctan (-(1 / Real.sqrt 3))

-- Theorem 1
theorem f_at_α : f α = -3 := by sorry

-- Theorem 2
theorem f_minimum_on_interval :
  ∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x₀ ≤ f x ∧ f x₀ = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_α_f_minimum_on_interval_l995_99561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_inequality_l995_99550

def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

def geometric_sequence (b₁ r : ℝ) (n : ℕ) : ℝ := b₁ * r^(n - 1)

theorem smallest_n_for_inequality (a₁ a₇ b₁ : ℝ) (h₁ : a₁ = 1) (h₂ : a₇ = 4) (h₃ : b₁ = 6) :
  let d := (a₇ - a₁) / 6
  let a := arithmetic_sequence a₁ d
  let r := a 3 / b₁
  let b := geometric_sequence b₁ r
  ∃ n, n = 7 ∧ ∀ m < n, b m * a 26 ≥ 1 ∧ b n * a 26 < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_inequality_l995_99550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l995_99597

theorem order_of_abc (a b c : ℝ) 
  (ha : a = Real.log 0.9 / Real.log 2)
  (hb : b = 3 ^ (-1/3 : ℝ))
  (hc : c = (1/3 : ℝ) ^ (1/2 : ℝ)) :
  a < c ∧ c < b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l995_99597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_king_placement_bounds_l995_99559

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)

/-- Represents a placement of kings on a chessboard --/
structure KingPlacement :=
  (board : Chessboard)
  (num_kings : ℕ)

/-- Predicate to check if a king is at a specific position --/
def king_at (p : KingPlacement) (k : ℕ) (pos : ℕ × ℕ) : Prop :=
  sorry

/-- Predicate to check if a king attacks a specific position --/
def king_attacks (p : KingPlacement) (k : ℕ) (pos : ℕ × ℕ) : Prop :=
  sorry

/-- Checks if a king placement is correct --/
def is_correct_placement (p : KingPlacement) : Prop :=
  ∀ (i j : ℕ), i < p.board.size ∧ j < p.board.size →
    (∃ (k : ℕ), k < p.num_kings ∧ (king_at p k (i, j) ∨ king_attacks p k (i, j))) ∧
    (∀ (k1 k2 : ℕ), k1 < p.num_kings ∧ k2 < p.num_kings ∧ k1 ≠ k2 →
      ¬(∃ (i j : ℕ), king_at p k1 (i, j) ∧ king_attacks p k2 (i, j)))

/-- Theorem stating the minimum and maximum number of correctly placed kings --/
theorem correct_king_placement_bounds :
  ∃ (min max : ℕ),
    (∀ (p : KingPlacement), p.board.size = 8 ∧ is_correct_placement p →
      min ≤ p.num_kings ∧ p.num_kings ≤ max) ∧
    (∃ (p_min p_max : KingPlacement),
      p_min.board.size = 8 ∧ p_max.board.size = 8 ∧
      is_correct_placement p_min ∧ is_correct_placement p_max ∧
      p_min.num_kings = min ∧ p_max.num_kings = max) ∧
    min = 9 ∧ max = 16 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_king_placement_bounds_l995_99559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_protein_percentage_in_other_meal_l995_99555

/-- Proves that given a 280 lb mixture with 13% protein, composed of 240 lb of soybean meal (14% protein) and another meal, the protein percentage of the other meal is 7%. -/
theorem protein_percentage_in_other_meal
  (total_weight : ℝ)
  (total_protein_percentage : ℝ)
  (soybean_weight : ℝ)
  (soybean_protein_percentage : ℝ)
  (h1 : total_weight = 280)
  (h2 : total_protein_percentage = 13)
  (h3 : soybean_weight = 240)
  (h4 : soybean_protein_percentage = 14)
  : (total_weight * total_protein_percentage / 100 - soybean_weight * soybean_protein_percentage / 100) / (total_weight - soybean_weight) * 100 = 7 := by
  sorry

-- Remove the #eval line as it's not necessary for building and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_protein_percentage_in_other_meal_l995_99555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l995_99583

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse centered at the origin -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculate the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- Calculate the dot product of two vectors -/
def dotProduct (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

/-- Theorem about the ellipse and its intersecting line -/
theorem ellipse_and_line_theorem (M P : Point) (e : Ellipse) :
  M.x = 1 ∧ M.y = 3/2 ∧ P.x = 2 ∧ P.y = 1 ∧
  pointOnEllipse M e ∧
  eccentricity e = 1/2 ∧
  e.a > e.b ∧ e.b > 0 →
  e.a^2 = 4 ∧ e.b^2 = 3 ∧
  ∃ (k : ℝ), k = 1/2 ∧
    ∃ (A B : Point),
      pointOnEllipse A e ∧
      pointOnEllipse B e ∧
      A.y = k * A.x ∧
      B.y = k * B.x ∧
      A ≠ B ∧
      dotProduct (Point.mk (A.x - P.x) (A.y - P.y)) (Point.mk (B.x - P.x) (B.y - P.y)) =
        dotProduct (Point.mk (M.x - P.x) (M.y - P.y)) (Point.mk (M.x - P.x) (M.y - P.y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l995_99583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_converges_to_zero_l995_99537

-- Define the sequence
noncomputable def x : ℕ → ℝ
  | 0 => 25
  | n + 1 => Real.arctan (x n)

-- State the theorem
theorem x_converges_to_zero :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |x n - 0| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_converges_to_zero_l995_99537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alien_sock_shoe_combinations_l995_99515

/-- The number of valid arrangements for n sock-shoe pairs. -/
def number_of_valid_arrangements (n : ℕ) : ℕ := Nat.choose (2*n) n

theorem alien_sock_shoe_combinations : ∀ n : ℕ, 
  number_of_valid_arrangements n = Nat.choose (2*n) n :=
by
  -- We define n as the number of feet (which is also the number of sock-shoe pairs)
  intro n
  -- The theorem holds by definition of number_of_valid_arrangements
  rfl

-- The specific case for 4 feet
example : number_of_valid_arrangements 4 = 70 :=
by
  -- Evaluate the function for n = 4
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alien_sock_shoe_combinations_l995_99515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_half_mileage_is_twenty_l995_99551

/-- Represents the weekly mileage for the first half of the year -/
def first_half_weekly_miles : ℝ → ℝ := fun x => x

/-- Represents the weekly mileage for the second half of the year -/
def second_half_weekly_miles : ℝ := 30

/-- Represents the total number of weeks in a year -/
def weeks_in_year : ℝ := 52

/-- Represents the total mileage for the year -/
def total_yearly_miles : ℝ := 1300

/-- Theorem stating that the weekly mileage for the first half of the year is 20 -/
theorem first_half_mileage_is_twenty :
  first_half_weekly_miles 20 = 20 ∧
  (first_half_weekly_miles 20 * (weeks_in_year / 2) + 
   second_half_weekly_miles * (weeks_in_year / 2) = total_yearly_miles) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_half_mileage_is_twenty_l995_99551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l995_99578

/-- The equation of the tangent line to y = sin(x + π/3) at (0, √3/2) -/
theorem tangent_line_equation (f : ℝ → ℝ) (x₀ y₀ : ℝ) :
  f = (λ x ↦ Real.sin (x + π/3)) →
  x₀ = 0 →
  y₀ = Real.sqrt 3 / 2 →
  f x₀ = y₀ →
  (deriv f) x₀ = 1/2 →
  ∃ (a b c : ℝ), a*x₀ + b*y₀ + c = 0 ∧
                 ∀ x y, a*x + b*y + c = 0 ↔ y - y₀ = (deriv f x₀) * (x - x₀) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l995_99578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_and_monotonicity_l995_99535

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x^2 + 3*b*x

-- State the theorem
theorem tangent_point_and_monotonicity (a b : ℝ) :
  (f a b 1 = -11) ∧ 
  ((12 : ℝ) * 1 + f a b 1 - 1 = 0) ∧
  (∀ x : ℝ, (12 : ℝ) * x + f a b x - 1 ≥ 0) →
  (a = 1 ∧ b = -3) ∧
  (∀ x : ℝ, x < -1 → (deriv (f 1 (-3))) x > 0) ∧
  (∀ x : ℝ, x > 3 → (deriv (f 1 (-3))) x > 0) ∧
  (∀ x : ℝ, -1 < x ∧ x < 3 → (deriv (f 1 (-3))) x < 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_and_monotonicity_l995_99535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cube_ending_368_l995_99505

theorem smallest_cube_ending_368 : 
  ∀ n : ℕ, n^3 % 1000 = 368 → n ≥ 34 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cube_ending_368_l995_99505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_only_coprime_to_all_terms_l995_99584

/-- Sequence a_n defined as 2^n + 3^n + 6^n - 1 -/
def a (n : ℕ) : ℕ := 2^n + 3^n + 6^n - 1

/-- The property of being coprime to all terms of the sequence -/
def coprimeToAllTerms (k : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → Nat.Coprime k (a n)

/-- Theorem: 1 is the only positive integer coprime to all terms of the sequence -/
theorem one_only_coprime_to_all_terms :
  ∀ k : ℕ, k ≥ 1 → (coprimeToAllTerms k ↔ k = 1) := by
  sorry

#check one_only_coprime_to_all_terms

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_only_coprime_to_all_terms_l995_99584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_box_twice_volume_of_second_l995_99560

/-- The volume of a cylinder with radius r and height h is π * r^2 * h -/
noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- Given two cylindrical boxes where the first box has height h₁ and diameter d₁,
    and the second box has height 2h₁ and diameter d₁/2,
    prove that the volume of the first box is twice the volume of the second box -/
theorem first_box_twice_volume_of_second (h₁ d₁ : ℝ) (h₁_pos : h₁ > 0) (d₁_pos : d₁ > 0) :
  cylinder_volume (d₁/2) h₁ = 2 * cylinder_volume (d₁/4) (2*h₁) := by
  -- Unfold the definition of cylinder_volume
  unfold cylinder_volume
  -- Simplify the expressions
  simp
  -- The rest of the proof
  sorry

#check first_box_twice_volume_of_second

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_box_twice_volume_of_second_l995_99560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zigzag_angle_l995_99546

/-- A circle with a zig-zag line from P to Q on its diameter -/
structure ZigZagCircle where
  /-- The circle -/
  circle : Set (Real × Real)
  /-- The diameter endpoints P and Q -/
  p : Real × Real
  q : Real × Real
  /-- The zig-zag line -/
  zigzag : Set (Real × Real)
  /-- The number of peaks in the zig-zag line -/
  num_peaks : Nat
  /-- The angle between each zig-zag segment and the diameter -/
  α : Real

/-- Properties of the ZigZagCircle -/
axiom zigzag_properties (z : ZigZagCircle) :
  z.p ∈ z.circle ∧ z.q ∈ z.circle ∧
  z.p ∈ z.zigzag ∧ z.q ∈ z.zigzag ∧
  z.num_peaks = 4 ∧
  ∀ peak ∈ z.zigzag, ∃ θ : Real, θ = z.α ∨ θ = -z.α

/-- Theorem: The angle α in a ZigZagCircle with 4 peaks is 72° -/
theorem zigzag_angle (z : ZigZagCircle) : z.α = 72 * Real.pi / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zigzag_angle_l995_99546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_g_property_l995_99576

theorem unique_g_property : ∃! (g : ℕ), g > 0 ∧ 
  ∀ (p : ℕ), Nat.Prime p → Odd p → 
    ∃ (n : ℕ), n > 0 ∧ 
      (p ∣ g^n - n) ∧ 
      (p ∣ g^(n+1) - (n + 1)) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_g_property_l995_99576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l995_99579

-- Define the function f on the closed interval [0, 1]
variable (f : ℝ → ℝ)

-- State the theorem
theorem integral_inequality
  (hf_domain : ∀ x, x ∈ Set.Icc 0 1 → f x ∈ Set.Icc 0 1)
  (hf_zero : f 0 = 0)
  (hf_diff : ContDiff ℝ 1 f)
  (hf_deriv : ∀ x, x ∈ Set.Ioo 0 1 → 0 < deriv f x ∧ deriv f x ≤ 1) :
  (∫ x in Set.Icc 0 1, f x)^2 ≥ ∫ x in Set.Icc 0 1, (f x)^3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l995_99579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kellan_car_wax_l995_99585

/-- The amount of wax needed to detail Kellan's car -/
def car_wax : ℝ := 3

/-- The amount of wax needed to detail Kellan's SUV -/
def suv_wax : ℝ := 4

/-- The initial amount of wax in the bottle -/
def initial_wax : ℝ := 11

/-- The amount of wax spilled -/
def spilled_wax : ℝ := 2

/-- The amount of wax left after detailing both vehicles -/
def remaining_wax : ℝ := 2

theorem kellan_car_wax : car_wax = 3 := by
  -- Proof goes here
  sorry

#check kellan_car_wax

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kellan_car_wax_l995_99585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_l995_99527

theorem sin_theta_value (a : ℝ) (θ : ℝ) (h1 : a ≠ 0) (h2 : Real.tan θ = -a) :
  Real.sin θ = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_l995_99527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marzuq_finished_eighth_l995_99523

/-- Represents a racer in the race --/
inductive Racer
| Lian
| Marzuq
| Rafsan
| Arabi
| Nabeel
| Rahul

/-- Represents the position of a racer in the race --/
def Position := Fin 12

/-- Represents the finishing order of the race --/
def RaceResult := Racer → Position

/-- Given race results satisfy the conditions of the problem --/
def ValidRaceResult (result : RaceResult) : Prop :=
  (result Racer.Nabeel).val + 6 = (result Racer.Marzuq).val ∧
  (result Racer.Arabi).val = (result Racer.Rafsan).val + 1 ∧
  (result Racer.Lian).val = (result Racer.Marzuq).val + 2 ∧
  (result Racer.Rafsan).val = (result Racer.Rahul).val + 2 ∧
  (result Racer.Rahul).val = (result Racer.Nabeel).val + 1 ∧
  (result Racer.Arabi).val = 6

theorem marzuq_finished_eighth (result : RaceResult) (h : ValidRaceResult result) :
  (result Racer.Marzuq).val = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marzuq_finished_eighth_l995_99523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_price_calculation_l995_99588

theorem cycle_price_calculation (selling_price : ℝ) (gain_percentage : ℝ) (cost_price : ℝ) : 
  selling_price = 1080 ∧ 
  gain_percentage = 27.058823529411764 ∧
  gain_percentage = (selling_price - cost_price) / cost_price * 100 →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |cost_price - 850| < ε :=
by
  intro h
  sorry

#check cycle_price_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_price_calculation_l995_99588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l995_99554

theorem min_value_of_a : 
  ∃ (min_a : ℝ), min_a = 2 * Real.sqrt 2 + 2 ∧ 
    (∀ b : ℝ, b > 1 → 
      let a := (b^2 - 1) / (b - 1) + 2 / (b - 1) + 2
      let perpendicular := (b^2 + 1) * 1 + a * (-(b - 1)) = 0
      perpendicular → a ≥ min_a) ∧
    (∃ b : ℝ, b > 1 ∧ 
      let a := (b^2 - 1) / (b - 1) + 2 / (b - 1) + 2
      let perpendicular := (b^2 + 1) * 1 + a * (-(b - 1)) = 0
      perpendicular ∧ a = min_a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l995_99554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l995_99569

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if a point (x, y) lies on the circle -/
def Circle.contains (c : Circle) (x y : ℝ) : Prop :=
  (x - c.h)^2 + (y - c.k)^2 = c.r^2

/-- Check if a circle is tangent to a line y = m -/
def Circle.tangentTo (c : Circle) (m : ℝ) : Prop :=
  |c.k - m| = c.r

/-- The standard equation of a circle -/
def Circle.equation (c : Circle) : ℝ → ℝ → Prop :=
  λ x y ↦ (x - c.h)^2 + (y - c.k)^2 = c.r^2

theorem circle_equation (c : Circle) :
  (c.h = c.k ∧ c.r = 2 ∧ c.tangentTo 6) ∨
  (c.contains 4 3 ∧ c.contains 5 2 ∧ c.contains 1 0) →
  (c.equation = λ x y ↦ (x - 4)^2 + (y - 4)^2 = 4) ∨
  (c.equation = λ x y ↦ (x - 8)^2 + (y - 8)^2 = 4) ∨
  (c.equation = λ x y ↦ x^2 + y^2 - 6*x - 2*y + 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l995_99569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_max_at_65_l995_99587

def A (k : ℕ) : ℚ := (19^k + 66^k) / (Nat.factorial k)

theorem A_max_at_65 : ∀ k : ℕ, A k ≤ A 65 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_max_at_65_l995_99587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neq_implies_angle_neq_sufficient_not_necessary_l995_99595

theorem cos_neq_implies_angle_neq_sufficient_not_necessary 
  (α β : ℝ) (h : Real.cos α ≠ Real.cos β) :
  (∃ γ δ : ℝ, γ ≠ δ ∧ Real.cos γ = Real.cos δ) ∧ (α ≠ β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neq_implies_angle_neq_sufficient_not_necessary_l995_99595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_angle_relation_l995_99534

theorem sine_angle_relation (α : ℝ) : 
  Real.sin (α - 2*π/3) = 1/4 → Real.sin (α + π/3) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_angle_relation_l995_99534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_second_quadrant_l995_99543

theorem tan_value_second_quadrant (α : ℝ) : 
  (π / 2 < α ∧ α < π) →  -- α is in the second quadrant
  Real.cos α = -12/13 → 
  Real.tan α = -5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_second_quadrant_l995_99543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_2_range_when_a_is_negative_l995_99594

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4*x + 3 + a) / (x - 1)

-- Theorem for the first part of the problem
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x ≥ 1 ∧ x ≠ 1} = Set.Ioo 1 2 ∪ Set.Ici 3 :=
by sorry

-- Theorem for the second part of the problem
theorem range_when_a_is_negative (a : ℝ) (h : a < 0) :
  {y : ℝ | ∃ x ∈ Set.Ioc 1 3, f a x = y} = Set.Iic (a / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_2_range_when_a_is_negative_l995_99594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_k_range_of_t_range_of_a_l995_99526

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x + 2 - k * x^2) / x^2

-- Theorem 1: Value of k
theorem value_of_k :
  ∃ k : ℝ, (∀ x : ℝ, x ∈ (Set.Ioo (-1) 0 ∪ Set.Ioo 0 2) → f k x > 0) ∧ k = 1 := by
  sorry

-- Theorem 2: Range of t
theorem range_of_t :
  ∃ t : ℝ, (∀ x : ℝ, x ∈ Set.Ioo (1/2) 1 → t - 1 < f 1 x) ∧
           (∃ x₀ : ℝ, x₀ ∈ Set.Ioo (-5) 0 ∧ t - 1 < f 1 x₀) ∧
           t ∈ Set.Icc (-3/25) 3 := by
  sorry

-- Theorem 3: Range of a
theorem range_of_a :
  ∃ a : ℝ, (∀ x : ℝ, x > 0 → x < 2 → Real.log (f 1 x) + 2 * Real.log x = Real.log (3 - a * x)) ∧
           (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ > 0 → x₁ < 2 → x₂ > 0 → x₂ < 2 →
              Real.log (f 1 x₁) + 2 * Real.log x₁ ≠ Real.log (3 - a * x₁) ∨
              Real.log (f 1 x₂) + 2 * Real.log x₂ ≠ Real.log (3 - a * x₂)) ∧
           (a = 1 ∨ a ≥ 3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_k_range_of_t_range_of_a_l995_99526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABCDEF_is_70_l995_99501

/-- Represents a polygon with vertices A, B, C, D, E, F, and G -/
structure Polygon where
  AB : ℝ
  BC : ℝ
  DC : ℝ
  FA : ℝ
  GF : ℝ
  ED : ℝ
  trapezoid_height : ℝ

/-- Calculate the area of polygon ABCDEF -/
noncomputable def area_ABCDEF (p : Polygon) : ℝ :=
  p.AB * p.BC - (p.GF + p.ED) / 2 * p.trapezoid_height

/-- Theorem stating that the area of polygon ABCDEF is 70 square units -/
theorem area_ABCDEF_is_70 (p : Polygon)
  (h1 : p.AB = 8)
  (h2 : p.BC = 10)
  (h3 : p.DC = 5)
  (h4 : p.FA = 7)
  (h5 : p.GF = 3)
  (h6 : p.ED = 7)
  (h7 : p.trapezoid_height = 2) :
  area_ABCDEF p = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABCDEF_is_70_l995_99501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_schedule_theorem_l995_99509

-- Define the number of periods and courses
def num_periods : ℕ := 7
def num_courses : ℕ := 3

-- Define a function to calculate the number of valid schedules
def valid_schedules (periods : ℕ) (courses : ℕ) : ℕ :=
  Nat.choose periods courses * Nat.factorial courses - 
  (periods - courses + 1) * Nat.factorial courses

-- State the theorem
theorem schedule_theorem : 
  valid_schedules num_periods num_courses = 180 := by
  -- Unfold the definition of valid_schedules
  unfold valid_schedules
  -- Unfold the definition of num_periods and num_courses
  unfold num_periods num_courses
  -- Evaluate the expression
  norm_num
  -- QED
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_schedule_theorem_l995_99509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l995_99511

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + x + 1)

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ 1/3 ≤ y ∧ y ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l995_99511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l995_99565

/-- A complex number z is in the second quadrant if its real part is negative and its imaginary part is positive -/
def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

/-- The complex number 5i/(2-i) -/
noncomputable def z : ℂ := (5 * Complex.I) / (2 - Complex.I)

/-- Theorem: The complex number 5i/(2-i) is in the second quadrant -/
theorem z_in_second_quadrant : is_in_second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l995_99565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_perimeter_l995_99538

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

/-- Checks if three positive integers form a valid triangle -/
def is_valid_triangle (l m n : ℕ) : Prop :=
  l > m ∧ m > n ∧ l < m + n

/-- Checks if three positive integers satisfy the given modular condition -/
def satisfies_modular_condition (l m n : ℕ) : Prop :=
  frac (3^l / 10000 : ℝ) = frac (3^m / 10000 : ℝ) ∧
  frac (3^m / 10000 : ℝ) = frac (3^n / 10000 : ℝ)

/-- The main theorem stating the smallest perimeter of the triangle -/
theorem smallest_triangle_perimeter :
  ∀ l m n : ℕ,
  is_valid_triangle l m n →
  satisfies_modular_condition l m n →
  l + m + n ≥ 3003 :=
by
  sorry

#check smallest_triangle_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_perimeter_l995_99538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l995_99562

noncomputable def f (x : ℝ) : ℝ := (x - 3) * (x - 5) * (x - 7) / ((x - 2) * (x - 6) * (x - 8))

def solution_set : Set ℝ := Set.Iio 2 ∪ Set.Ioo 3 5 ∪ Set.Ioo 6 7 ∪ Set.Ioi 8

theorem inequality_solution :
  ∀ x : ℝ, f x > 0 ↔ x ∈ solution_set := by
  sorry

#check inequality_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l995_99562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_equals_two_l995_99577

-- Define the curve
noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x)

-- Define the tangent line
def tangent_line (a : ℝ) (x : ℝ) : ℝ := a * x + 1

-- Define the given line
def given_line (x y : ℝ) : Prop := x + 2 * y + 1 = 0

-- Theorem statement
theorem tangent_perpendicular_implies_a_equals_two (a : ℝ) :
  (∀ x, curve a x = tangent_line a x) →
  (given_line 0 1) →
  (∀ x y, given_line x y → tangent_line a x = y → x = 0 ∧ y = 1) →
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_equals_two_l995_99577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_addition_l995_99547

theorem perfect_square_addition (rectangle_area rectangle_side : ℕ) 
  (h1 : rectangle_area = 13600) (h2 : rectangle_side = 136) :
  ∃ (n : ℕ), (rectangle_area + n = 13924 ∧ 
              IsSquare (rectangle_area + n) ∧ 
              (Nat.sqrt (rectangle_area + n)) = 118) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_addition_l995_99547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_points_l995_99552

-- Define the set of possible 'a' values
def A : Finset ℤ := {-3, -2, -1, 0, 1, 2, 3}

-- Define the set of possible 'b' values
def B : Finset ℤ := {-4, -3, -2, -1, 0, 1, 2, 3, 4}

-- Define a parabola type
structure Parabola where
  a : ℤ
  b : ℤ
deriving DecidableEq

-- Define the set of all parabolas
def allParabolas : Finset Parabola :=
  Finset.product A B |>.image (fun (a, b) => ⟨a, b⟩)

-- Function to check if two parabolas intersect
def intersect (p1 p2 : Parabola) : Bool :=
  if p1.a = p2.a then
    (p1.b > 0 ∧ p2.b < 0) ∨ (p1.b < 0 ∧ p2.b > 0)
  else true

-- Theorem statement
theorem parabola_intersection_points :
  (Finset.filter (fun p => 
    (Finset.filter (fun q => intersect p q ∧ p ≠ q) allParabolas).card = 2
  ) allParabolas).card = 2170 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_points_l995_99552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_roots_equation_l995_99517

variable (a b p q : ℝ)
variable (α : ℝ)

-- Define the two quadratic equations
def eq1 (a b : ℝ) (x : ℝ) : Prop := x^2 + a*x + b = 0
def eq2 (p q : ℝ) (x : ℝ) : Prop := x^2 + p*x + q = 0

-- Define the condition that α is a non-zero common root
def is_common_root (a b p q α : ℝ) : Prop := α ≠ 0 ∧ eq1 a b α ∧ eq2 p q α

-- Define the resulting quadratic equation
def result_eq (a b p q : ℝ) (x : ℝ) : Prop := 
  x^2 - ((b+q)*(a-p)/(q-b))*x + (b*q*(a-p)^2/(q-b)^2) = 0

-- State the theorem
theorem distinct_roots_equation 
  (h : is_common_root a b p q α) :
  ∃ (x y : ℝ), x ≠ y ∧ eq1 a b x ∧ eq2 p q y ∧ 
  (∀ (z : ℝ), result_eq a b p q z ↔ (z = x ∨ z = y)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_roots_equation_l995_99517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_formula_l995_99512

-- Define the angle α and the parameter t
variable (α : Real) (t : Real)

-- Define the point P as a function of t
def P (t : Real) : ℝ × ℝ := (-4 * t, 3 * t)

-- State the theorem
theorem angle_terminal_side_formula :
  t > 0 →  -- t is positive
  (∃ (r : Real), r > 0 ∧ P t = (r * Real.cos α, r * Real.sin α)) →  -- terminal side passes through P
  2 * Real.sin α + Real.cos α = 2 / 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_formula_l995_99512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_is_correct_l995_99514

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by parametric equations -/
structure Line3D where
  x : ℝ → ℝ
  y : ℝ → ℝ
  z : ℝ → ℝ

/-- A plane in 3D space defined by the equation Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- The given point that the plane passes through -/
def givenPoint : Point3D :=
  { x := 1, y := 6, z := -8 }

/-- The given line that the plane contains -/
def givenLine : Line3D :=
  { x := λ t => 4 * t + 2,
    y := λ t => -t + 4,
    z := λ t => 5 * t - 3 }

/-- The plane we want to prove is correct -/
def resultPlane : Plane :=
  { A := 5, B := 15, C := -7, D := -151 }

theorem plane_equation_is_correct :
  (resultPlane.A > 0) ∧
  (Int.gcd (Int.gcd (Int.natAbs resultPlane.A) (Int.natAbs resultPlane.B))
           (Int.gcd (Int.natAbs resultPlane.C) (Int.natAbs resultPlane.D)) = 1) ∧
  (resultPlane.A * givenPoint.x + resultPlane.B * givenPoint.y +
   resultPlane.C * givenPoint.z + resultPlane.D = 0) ∧
  (∀ t : ℝ, resultPlane.A * (givenLine.x t) + resultPlane.B * (givenLine.y t) +
             resultPlane.C * (givenLine.z t) + resultPlane.D = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_is_correct_l995_99514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ducks_in_marsh_correct_l995_99508

def ducks_in_marsh : ℕ :=
  let geese : ℕ := 58
  let more_geese : ℕ := 21
  geese - more_geese

#eval ducks_in_marsh

theorem ducks_in_marsh_correct : ducks_in_marsh = 37 := by
  unfold ducks_in_marsh
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ducks_in_marsh_correct_l995_99508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_deducible_l995_99599

-- Define the universe
variable (U : Type)

-- Define the sets as subsets of the universe
variable (Mem En Vee : Set U)

-- Hypothesis I: Some Mems are not Ens
axiom some_mems_not_ens : ∃ m ∈ Mem, m ∉ En

-- Hypothesis II: No Ens are Veens
axiom no_ens_are_veens : En ∩ Vee = ∅

-- The theorem to prove
theorem not_deducible :
  ¬(
    -- (A) Some Mems are not Veens
    (∃ m ∈ Mem, m ∉ Vee) ∨
    -- (B) Some Veens are not Mems
    (∃ v ∈ Vee, v ∉ Mem) ∨
    -- (C) No Mem is a Vee
    (Mem ∩ Vee = ∅) ∨
    -- (D) Some Mems are Vees
    (∃ m ∈ Mem, m ∈ Vee)
  ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_deducible_l995_99599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_from_tangents_l995_99504

theorem triangle_angle_from_tangents (A B C : ℝ) : 
  A + B + C = π → 
  Real.tan A = 1/3 → 
  Real.tan B = -2 → 
  C = π/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_from_tangents_l995_99504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terry_driving_time_l995_99536

/-- Calculates the time taken for a round trip given speed and one-way distance -/
noncomputable def round_trip_time (speed : ℝ) (one_way_distance : ℝ) : ℝ :=
  (2 * one_way_distance) / speed

/-- Terry's driving scenario -/
theorem terry_driving_time :
  let speed : ℝ := 40 -- miles per hour
  let one_way_distance : ℝ := 60 -- miles
  round_trip_time speed one_way_distance = 3 := by
  -- Unfold the definitions
  unfold round_trip_time
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_terry_driving_time_l995_99536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_20gon_power_1995_distinct_l995_99556

/-- A regular 20-gon inscribed in the unit circle -/
def regular_20gon : Finset ℂ :=
  sorry

/-- The property that the set is a regular 20-gon inscribed in the unit circle -/
def is_regular_20gon (S : Finset ℂ) : Prop :=
  S.card = 20 ∧
  ∀ z ∈ S, Complex.abs z = 1 ∧
  ∃ θ : ℝ, ∀ k : ℕ, k < 20 → Complex.exp (2 * Real.pi * Complex.I * (k : ℝ) / 20) ∈ S

theorem regular_20gon_power_1995_distinct :
  let S := regular_20gon
  (is_regular_20gon S) →
  (Finset.image (fun z => z^1995) S).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_20gon_power_1995_distinct_l995_99556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_ball_max_height_l995_99533

/-- The height function of a soccer ball's trajectory --/
def soccer_ball_height (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 5

/-- The maximum height achieved by the soccer ball --/
theorem soccer_ball_max_height :
  ∃ (max : ℝ), max = 85 ∧ ∀ t, soccer_ball_height t ≤ max := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_ball_max_height_l995_99533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_logarithm_sum_constant_l995_99530

theorem unique_logarithm_sum_constant (a : ℝ) (h_a : a > 1) :
  (∃! c : ℝ, ∀ x ∈ Set.Icc a (2 * a), ∃ y ∈ Set.Icc a (a^2),
    Real.log x / Real.log a + Real.log y / Real.log a = c) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_logarithm_sum_constant_l995_99530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_proportionality_l995_99572

/-- Two vectors are proportional if their corresponding components are in the same ratio. -/
def proportional (v w : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v.1 = k * w.1 ∧ v.2.1 = k * w.2.1 ∧ v.2.2 = k * w.2.2

/-- The problem statement -/
theorem vector_proportionality :
  ∀ c d : ℝ, proportional (3, c, -8) (6, 5, d) → c = 5/2 ∧ d = -16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_proportionality_l995_99572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_seven_failures_three_successes_l995_99558

def probability_exact_sequence (n k : ℕ) (p : ℝ) : ℝ :=
  (1 - p)^(n - k) * p^k

theorem probability_seven_failures_three_successes (p : ℝ) (hp : 0 < p ∧ p < 1) :
  let n : ℕ := 10
  let k : ℕ := 3
  probability_exact_sequence n k p = (1 - p)^(n - k) * p^k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_seven_failures_three_successes_l995_99558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l995_99557

theorem tan_alpha_value (α : ℝ)
  (h1 : Real.sin (α + π / 2) = 1 / 3)
  (h2 : α ∈ Set.Ioo (-π / 2) 0) :
  Real.tan α = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l995_99557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l995_99563

/-- The function f(x) = 2^|x - a| -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(|x - a|)

/-- Theorem stating the minimum value of m -/
theorem min_m_value (a : ℝ) (m : ℝ) :
  (∀ x, f a (1 + x) = f a (1 - x)) →
  (∀ x y, m ≤ x → x < y → f a x ≤ f a y) →
  m = 1 :=
by
  sorry

#check min_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l995_99563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_union_theorem_l995_99593

open Set

-- Define an open interval
def OpenInterval (a b : ℝ) := {x : ℝ | a < x ∧ x < b}

theorem interval_union_theorem (I₁ I₂ I₃ : Set ℝ) 
  (h_open₁ : ∃ a b, I₁ = OpenInterval a b)
  (h_open₂ : ∃ a b, I₂ = OpenInterval a b)
  (h_open₃ : ∃ a b, I₃ = OpenInterval a b)
  (h_not_subset : ¬(I₁ ⊆ I₂ ∨ I₁ ⊆ I₃ ∨ I₂ ⊆ I₁ ∨ I₂ ⊆ I₃ ∨ I₃ ⊆ I₁ ∨ I₃ ⊆ I₂))
  (h_nonempty_inter : (I₁ ∩ I₂ ∩ I₃).Nonempty) :
  ∃ i ∈ ({1, 2, 3} : Set Nat), 
    (i = 1 ∧ I₁ ⊆ I₂ ∪ I₃) ∨
    (i = 2 ∧ I₂ ⊆ I₁ ∪ I₃) ∨
    (i = 3 ∧ I₃ ⊆ I₁ ∪ I₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_union_theorem_l995_99593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juan_tshirt_cost_l995_99503

/-- Calculates the total cost of t-shirts for Juan's employees --/
def total_tshirt_cost (men_white_price men_black_price : ℕ)
  (women_price_diff total_employees : ℕ) : ℕ :=
  let women_white_price := men_white_price - women_price_diff
  let women_black_price := men_black_price - women_price_diff
  let employees_per_gender := total_employees / 2
  let total_cost := 
    employees_per_gender * men_white_price +
    employees_per_gender * men_black_price +
    employees_per_gender * women_white_price +
    employees_per_gender * women_black_price
  total_cost

/-- The total cost of t-shirts for Juan's employees is $1320 --/
theorem juan_tshirt_cost : total_tshirt_cost 20 18 5 40 = 1320 := by
  rfl

#eval total_tshirt_cost 20 18 5 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_juan_tshirt_cost_l995_99503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l995_99522

noncomputable def g (x : ℝ) : ℝ := (4 * (Real.cos x)^4 + 5 * (Real.sin x)^2) / (4 * (Real.sin x)^4 + 3 * (Real.cos x)^2)

theorem g_properties :
  (∀ k : ℤ, g (k * Real.pi / 3) = 4/3) ∧
  (∀ x : ℝ, g x ≥ 5/4) ∧
  (∀ x : ℝ, g x ≤ 55/39) ∧
  (∃ x : ℝ, g x = 5/4) ∧
  (∃ x : ℝ, g x = 55/39) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l995_99522
