import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_system_l1913_191302

/-- The system of equations has a unique solution when a = 1, 
    and the solution is x = -3/2, y = -1/2, z = 0 -/
theorem unique_solution_system (a x y z : ℝ) : 
  z = a * (x + 2 * y + 5/2) ∧ 
  x^2 + y^2 + 2*x - y + z = 0 ∧
  ((x + (a + 2)/2)^2 + (y + (2*a - 1)/2)^2 = ((a + 2)^2)/4 + ((2*a - 1)^2)/4 - 5*a/2) →
  (a = 1 ∧ x = -3/2 ∧ y = -1/2 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1913_191302


namespace NUMINAMATH_CALUDE_function_g_theorem_l1913_191379

theorem function_g_theorem (g : ℝ → ℝ) 
  (h1 : g 0 = 2)
  (h2 : ∀ x y : ℝ, g (x * y) = g (x^2 + y^2) + 2 * (x - y)^2) :
  ∀ x : ℝ, g x = 2 - 2 * x :=
by sorry

end NUMINAMATH_CALUDE_function_g_theorem_l1913_191379


namespace NUMINAMATH_CALUDE_profit_A_range_max_a_value_l1913_191391

-- Define the profit functions
def profit_A_before (x : ℝ) : ℝ := 120000 * 500

def profit_A_after (x : ℝ) : ℝ := 120000 * (500 - x) * (1 + 0.005 * x)

def profit_B (x a : ℝ) : ℝ := 120000 * x * (a - 0.013 * x)

-- Theorem for part (I)
theorem profit_A_range (x : ℝ) :
  (0 < x ∧ x ≤ 300) ↔ profit_A_after x ≥ profit_A_before x :=
sorry

-- Theorem for part (II)
theorem max_a_value :
  ∃ (a : ℝ), a = 5.5 ∧
  ∀ (x : ℝ), 0 < x → x ≤ 300 →
  (∀ (a' : ℝ), a' > 0 → profit_B x a' ≤ profit_A_after x → a' ≤ a) :=
sorry

end NUMINAMATH_CALUDE_profit_A_range_max_a_value_l1913_191391


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l1913_191351

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l1913_191351


namespace NUMINAMATH_CALUDE_problem_statement_l1913_191326

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) :
  (x - 1)^2 + 9/(x - 1)^2 = 3 + 8/x :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1913_191326


namespace NUMINAMATH_CALUDE_divisibility_of_all_ones_number_l1913_191322

/-- A positive integer whose decimal representation contains only ones -/
def all_ones_number (n : ℕ) : ℕ :=
  (10^n - 1) / 9

theorem divisibility_of_all_ones_number (n : ℕ) (h : n > 0) :
  7 ∣ all_ones_number n → 13 ∣ all_ones_number n :=
by
  sorry

#check divisibility_of_all_ones_number

end NUMINAMATH_CALUDE_divisibility_of_all_ones_number_l1913_191322


namespace NUMINAMATH_CALUDE_max_fleas_on_chessboard_l1913_191330

/-- Represents a 10x10 chessboard -/
def Chessboard := Fin 10 × Fin 10

/-- Represents the four possible directions a flea can move -/
inductive Direction
| Up
| Down
| Left
| Right

/-- Represents a flea's position and direction -/
structure Flea where
  position : Chessboard
  direction : Direction

/-- Represents the state of the board at a given time -/
def BoardState := List Flea

/-- Simulates the movement of fleas for one hour (60 minutes) -/
def simulateMovement (initial : BoardState) : List BoardState := sorry

/-- Checks if two fleas occupy the same square -/
def noCollision (state : BoardState) : Prop := sorry

/-- Checks if the simulation is valid (no collisions for 60 minutes) -/
def validSimulation (states : List BoardState) : Prop := sorry

/-- The main theorem: The maximum number of fleas on a 10x10 chessboard is 40 -/
theorem max_fleas_on_chessboard :
  ∀ (initial : BoardState),
    validSimulation (simulateMovement initial) →
    initial.length ≤ 40 := by
  sorry

end NUMINAMATH_CALUDE_max_fleas_on_chessboard_l1913_191330


namespace NUMINAMATH_CALUDE_circle_on_line_tangent_to_axes_l1913_191307

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x - y = 3

-- Define the tangency condition to both axes
def tangent_to_axes (center_x center_y radius : ℝ) : Prop :=
  (abs center_x = radius ∧ abs center_y = radius)

-- Define the circle equation
def circle_equation (center_x center_y radius x y : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = radius^2

-- The main theorem
theorem circle_on_line_tangent_to_axes :
  ∀ (center_x center_y radius : ℝ),
    line_equation center_x center_y →
    tangent_to_axes center_x center_y radius →
    (∀ (x y : ℝ), circle_equation center_x center_y radius x y ↔
      ((x - 3)^2 + (y - 3)^2 = 9 ∨ (x - 1)^2 + (y + 1)^2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_circle_on_line_tangent_to_axes_l1913_191307


namespace NUMINAMATH_CALUDE_bobbys_shoes_cost_l1913_191365

/-- The total cost for Bobby's handmade shoes -/
def total_cost (mold_cost labor_rate hours discount : ℝ) : ℝ :=
  mold_cost + discount * labor_rate * hours

/-- Theorem stating the total cost for Bobby's handmade shoes is $730 -/
theorem bobbys_shoes_cost :
  total_cost 250 75 8 0.8 = 730 := by
  sorry

end NUMINAMATH_CALUDE_bobbys_shoes_cost_l1913_191365


namespace NUMINAMATH_CALUDE_sum_of_even_numbers_1_to_200_l1913_191396

theorem sum_of_even_numbers_1_to_200 : 
  (Finset.filter (fun n => Even n) (Finset.range 201)).sum id = 10100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_numbers_1_to_200_l1913_191396


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l1913_191364

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_cubes : Nat
  painted_squares_per_face : Nat
  num_faces : Nat

/-- The number of unpainted cubes in a painted cube -/
def num_unpainted_cubes (c : PaintedCube) : Nat :=
  c.total_cubes - (c.painted_squares_per_face * c.num_faces / 2)

/-- Theorem: In a 6x6x6 cube with 216 unit cubes and 4 painted squares on each of 6 faces,
    the number of unpainted cubes is 208 -/
theorem unpainted_cubes_in_6x6x6 :
  let c : PaintedCube := {
    size := 6,
    total_cubes := 216,
    painted_squares_per_face := 4,
    num_faces := 6
  }
  num_unpainted_cubes c = 208 := by sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l1913_191364


namespace NUMINAMATH_CALUDE_complement_of_union_l1913_191374

-- Define the universal set U
def U : Set Int := {-2, -1, 0, 1, 2, 3}

-- Define set A
def A : Set Int := {-1, 2}

-- Define set B
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

-- State the theorem
theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l1913_191374


namespace NUMINAMATH_CALUDE_greatest_common_multiple_10_15_under_100_l1913_191324

def is_common_multiple (n m k : ℕ) : Prop := k % n = 0 ∧ k % m = 0

theorem greatest_common_multiple_10_15_under_100 :
  ∃ (k : ℕ), k < 100 ∧ is_common_multiple 10 15 k ∧
  ∀ (j : ℕ), j < 100 → is_common_multiple 10 15 j → j ≤ k :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_10_15_under_100_l1913_191324


namespace NUMINAMATH_CALUDE_stock_price_change_l1913_191373

theorem stock_price_change (initial_price : ℝ) (initial_price_pos : initial_price > 0) : 
  let week1 := initial_price * 1.3
  let week2 := week1 * 0.75
  let week3 := week2 * 1.2
  let week4 := week3 * 0.85
  week4 = initial_price := by sorry

end NUMINAMATH_CALUDE_stock_price_change_l1913_191373


namespace NUMINAMATH_CALUDE_fudge_piece_size_l1913_191321

/-- Given a rectangular pan of fudge with dimensions 18 inches by 29 inches,
    containing 522 square pieces, prove that each piece has a side length of 1 inch. -/
theorem fudge_piece_size (pan_length : ℝ) (pan_width : ℝ) (num_pieces : ℕ) 
    (h1 : pan_length = 18) 
    (h2 : pan_width = 29) 
    (h3 : num_pieces = 522) : 
  (pan_length * pan_width) / num_pieces = 1 := by
  sorry

#check fudge_piece_size

end NUMINAMATH_CALUDE_fudge_piece_size_l1913_191321


namespace NUMINAMATH_CALUDE_robin_gum_pieces_l1913_191380

/-- Calculates the total number of gum pieces Robin has. -/
def total_gum_pieces (packages : ℕ) (pieces_per_package : ℕ) (extra_pieces : ℕ) : ℕ :=
  packages * pieces_per_package + extra_pieces

/-- Proves that Robin has 997 pieces of gum in total. -/
theorem robin_gum_pieces :
  total_gum_pieces 43 23 8 = 997 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_pieces_l1913_191380


namespace NUMINAMATH_CALUDE_angle_sum_equal_pi_over_two_l1913_191320

theorem angle_sum_equal_pi_over_two (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β) →
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_equal_pi_over_two_l1913_191320


namespace NUMINAMATH_CALUDE_remainder_2022_power_2023_power_2024_mod_19_l1913_191383

theorem remainder_2022_power_2023_power_2024_mod_19 :
  (2022 ^ (2023 ^ 2024)) % 19 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2022_power_2023_power_2024_mod_19_l1913_191383


namespace NUMINAMATH_CALUDE_mitzi_remaining_money_l1913_191382

/-- Proves that Mitzi has $9 left after her amusement park expenses -/
theorem mitzi_remaining_money (initial_amount ticket_cost food_cost tshirt_cost : ℕ) 
  (h1 : initial_amount = 75)
  (h2 : ticket_cost = 30)
  (h3 : food_cost = 13)
  (h4 : tshirt_cost = 23) :
  initial_amount - (ticket_cost + food_cost + tshirt_cost) = 9 := by
  sorry

end NUMINAMATH_CALUDE_mitzi_remaining_money_l1913_191382


namespace NUMINAMATH_CALUDE_k_range_l1913_191332

/-- The logarithm function to base 10 -/
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- The equation lg kx = 2 lg (x+1) has only one real root -/
def has_unique_root (k : ℝ) : Prop :=
  ∃! x : ℝ, log10 (k * x) = 2 * log10 (x + 1)

/-- The range of k values for which the equation has only one real root -/
theorem k_range : ∀ k : ℝ, has_unique_root k ↔ k = 4 ∨ k < 0 := by sorry

end NUMINAMATH_CALUDE_k_range_l1913_191332


namespace NUMINAMATH_CALUDE_hunters_playing_time_l1913_191314

/-- Given Hunter's playing times for football and basketball, prove the total time played in hours. -/
theorem hunters_playing_time (football_minutes basketball_minutes : ℕ) 
  (h1 : football_minutes = 60) 
  (h2 : basketball_minutes = 30) : 
  (football_minutes + basketball_minutes : ℚ) / 60 = 1.5 := by
  sorry

#check hunters_playing_time

end NUMINAMATH_CALUDE_hunters_playing_time_l1913_191314


namespace NUMINAMATH_CALUDE_two_arrows_balance_l1913_191384

/-- A polygon with arrows on its sides -/
structure ArrowPolygon where
  n : ℕ  -- number of sides/vertices
  incoming : Fin n → Fin 2  -- number of incoming arrows for each vertex (0, 1, or 2)
  outgoing : Fin n → Fin 2  -- number of outgoing arrows for each vertex (0, 1, or 2)

/-- The sum of incoming arrows equals the number of sides -/
axiom total_arrows_incoming (p : ArrowPolygon) : 
  (Finset.univ.sum p.incoming) = p.n

/-- The sum of outgoing arrows equals the number of sides -/
axiom total_arrows_outgoing (p : ArrowPolygon) : 
  (Finset.univ.sum p.outgoing) = p.n

/-- Theorem: The number of vertices with two incoming arrows equals the number of vertices with two outgoing arrows -/
theorem two_arrows_balance (p : ArrowPolygon) :
  (Finset.univ.filter (fun i => p.incoming i = 2)).card = 
  (Finset.univ.filter (fun i => p.outgoing i = 2)).card := by
  sorry

end NUMINAMATH_CALUDE_two_arrows_balance_l1913_191384


namespace NUMINAMATH_CALUDE_abs_diff_eq_sum_abs_iff_product_nonpositive_l1913_191370

theorem abs_diff_eq_sum_abs_iff_product_nonpositive (a b : ℝ) :
  |a - b| = |a| + |b| ↔ a * b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_abs_diff_eq_sum_abs_iff_product_nonpositive_l1913_191370


namespace NUMINAMATH_CALUDE_joan_balloons_l1913_191352

/-- The number of blue balloons Joan has after gaining more -/
def total_balloons (initial : ℕ) (gained : ℕ) : ℕ :=
  initial + gained

/-- Theorem stating that Joan has 95 blue balloons after gaining more -/
theorem joan_balloons : total_balloons 72 23 = 95 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l1913_191352


namespace NUMINAMATH_CALUDE_complex_expression_value_l1913_191399

theorem complex_expression_value : 
  2.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_value_l1913_191399


namespace NUMINAMATH_CALUDE_coffee_container_weight_l1913_191377

def suki_bags : ℝ := 6.5
def suki_weight_per_bag : ℝ := 22
def jimmy_bags : ℝ := 4.5
def jimmy_weight_per_bag : ℝ := 18
def num_containers : ℕ := 28

theorem coffee_container_weight :
  (suki_bags * suki_weight_per_bag + jimmy_bags * jimmy_weight_per_bag) / num_containers = 8 := by
  sorry

end NUMINAMATH_CALUDE_coffee_container_weight_l1913_191377


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1913_191328

theorem polynomial_simplification (x y : ℝ) :
  (4 * x^9 + 3 * y^8 + 5 * x^7) + (2 * x^10 + 6 * x^9 + y^8 + 4 * x^7 + 2 * y^4 + 7 * x + 9) =
  2 * x^10 + 10 * x^9 + 4 * y^8 + 9 * x^7 + 2 * y^4 + 7 * x + 9 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1913_191328


namespace NUMINAMATH_CALUDE_exp_log_properties_l1913_191335

-- Define the exponential and logarithmic functions
noncomputable def exp (a : ℝ) (x : ℝ) : ℝ := a^x
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem for the properties of exponential and logarithmic functions
theorem exp_log_properties :
  -- Domain and range of exponential function
  (∀ x : ℝ, ∃ y : ℝ, exp 2 x = y) ∧
  (∀ y : ℝ, y > 0 → ∃ x : ℝ, exp 2 x = y) ∧
  (exp 2 0 = 1) ∧
  -- Domain and range of logarithmic function
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, log 2 x = y) ∧
  (∀ y : ℝ, ∃ x : ℝ, x > 0 ∧ log 2 x = y) ∧
  (log 2 1 = 0) ∧
  -- Logarithm properties
  (∀ a M N : ℝ, a > 0 ∧ a ≠ 1 ∧ M > 0 ∧ N > 0 →
    log a (M * N) = log a M + log a N) ∧
  (∀ a N : ℝ, a > 0 ∧ a ≠ 1 ∧ N > 0 →
    exp a (log a N) = N) ∧
  (∀ a b m n : ℝ, a > 0 ∧ a ≠ 1 ∧ b > 0 ∧ m ≠ 0 →
    log (exp a m) (exp b n) = (n / m) * log a b) :=
by sorry

#check exp_log_properties

end NUMINAMATH_CALUDE_exp_log_properties_l1913_191335


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l1913_191303

theorem point_in_first_quadrant (x y : ℝ) (h1 : x + y = 2) (h2 : x - y = 1) : x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l1913_191303


namespace NUMINAMATH_CALUDE_perpendicular_lines_l1913_191318

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, ax + 2*y + 6 = 0 ∧ x + a*(a+1)*y + (a^2-1) = 0 → 
    (a = -3/2 ∨ a = 0)) ∧
  (a = -3/2 ∨ a = 0 → 
    ∀ x y : ℝ, ax + 2*y + 6 = 0 ∧ x + a*(a+1)*y + (a^2-1) = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l1913_191318


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1913_191360

/-- The sum of three specific repeating decimals is 2 -/
theorem sum_of_repeating_decimals : ∃ (x y z : ℚ),
  (∀ n : ℕ, (10 * x - x) * 10^n = 3 * 10^n) ∧
  (∀ n : ℕ, (10 * y - y) * 10^n = 6 * 10^n) ∧
  (∀ n : ℕ, (10 * z - z) * 10^n = 9 * 10^n) ∧
  x + y + z = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1913_191360


namespace NUMINAMATH_CALUDE_x_equals_one_necessary_and_sufficient_l1913_191347

theorem x_equals_one_necessary_and_sufficient :
  ∀ x : ℝ, (x^2 - 2*x + 1 = 0) ↔ (x = 1) := by
  sorry

end NUMINAMATH_CALUDE_x_equals_one_necessary_and_sufficient_l1913_191347


namespace NUMINAMATH_CALUDE_fold_sequence_counts_l1913_191356

/-- Represents the possible shapes after folding -/
inductive Shape
  | Square
  | IsoscelesTriangle
  | Rectangle (k : ℕ)

/-- Represents a sequence of folds -/
def FoldSequence := List Shape

/-- Counts the number of possible folding sequences -/
def countFoldSequences (n : ℕ) : ℕ :=
  sorry

theorem fold_sequence_counts :
  (countFoldSequences 3 = 5) ∧
  (countFoldSequences 6 = 24) ∧
  (countFoldSequences 9 = 149) := by
  sorry

end NUMINAMATH_CALUDE_fold_sequence_counts_l1913_191356


namespace NUMINAMATH_CALUDE_find_number_l1913_191369

theorem find_number : ∃! x : ℝ, (x + 82 + 90 + 88 + 84) / 5 = 88 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1913_191369


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1913_191355

theorem exponent_multiplication (x : ℝ) : x^5 * x^3 = x^8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1913_191355


namespace NUMINAMATH_CALUDE_geometric_sequence_min_S3_l1913_191350

/-- Given a geometric sequence with positive terms, prove that the minimum value of S_3 is 6 -/
theorem geometric_sequence_min_S3 (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- positive terms
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence
  a 4 * a 8 = 2 * a 10 →  -- given condition
  (∃ S : ℕ → ℝ, ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- sum of first n terms
  (∃ min_S3 : ℝ, ∀ S3, S3 = a 1 + a 2 + a 3 → S3 ≥ min_S3 ∧ min_S3 = 6) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_S3_l1913_191350


namespace NUMINAMATH_CALUDE_sum_squares_distances_to_chord_ends_l1913_191367

/-- Given a circle with radius R and a point M on its diameter at distance a from the center,
    the sum of squares of distances from M to the ends of any chord parallel to the diameter
    is equal to 2(a² + R²). -/
theorem sum_squares_distances_to_chord_ends
  (R a : ℝ) -- R is the radius, a is the distance from M to the center
  (h₁ : 0 < R) -- R is positive (circle has positive radius)
  (h₂ : 0 ≤ a ∧ a ≤ 2*R) -- M is on the diameter, so 0 ≤ a ≤ 2R
  : ∀ A B : ℝ × ℝ, -- For any points A and B
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 * R^2 → -- If AB is a chord (distance AB = diameter)
    (∃ k : ℝ, A.2 = k ∧ B.2 = k) → -- If AB is parallel to x-axis (assuming diameter along x-axis)
    (A.1 - a)^2 + A.2^2 + (B.1 - a)^2 + B.2^2 = 2 * (a^2 + R^2) :=
by sorry

end NUMINAMATH_CALUDE_sum_squares_distances_to_chord_ends_l1913_191367


namespace NUMINAMATH_CALUDE_subtraction_problem_l1913_191387

theorem subtraction_problem (minuend : ℝ) (difference : ℝ) (subtrahend : ℝ)
  (h1 : minuend = 98.2)
  (h2 : difference = 17.03)
  (h3 : subtrahend = minuend - difference) :
  subtrahend = 81.17 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l1913_191387


namespace NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l1913_191317

theorem max_y_coordinate_sin_3theta (θ : Real) :
  let r := λ θ : Real => Real.sin (3 * θ)
  let y := λ θ : Real => r θ * Real.sin θ
  ∃ (max_y : Real), max_y = 9/64 ∧ ∀ θ', y θ' ≤ max_y := by sorry

end NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l1913_191317


namespace NUMINAMATH_CALUDE_derivative_value_at_two_l1913_191388

theorem derivative_value_at_two (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, HasDerivAt f (f' x) x) →
  (∀ x, f x = x^2 + 3*x*(f' 2)) →
  f' 2 = -2 := by
sorry

end NUMINAMATH_CALUDE_derivative_value_at_two_l1913_191388


namespace NUMINAMATH_CALUDE_sandy_change_theorem_l1913_191359

def cappuccino_price : ℝ := 2
def iced_tea_price : ℝ := 3
def cafe_latte_price : ℝ := 1.5
def espresso_price : ℝ := 1

def sandy_order_cappuccinos : ℕ := 3
def sandy_order_iced_teas : ℕ := 2
def sandy_order_cafe_lattes : ℕ := 2
def sandy_order_espressos : ℕ := 2

def paid_amount : ℝ := 20

theorem sandy_change_theorem :
  paid_amount - (cappuccino_price * sandy_order_cappuccinos +
                 iced_tea_price * sandy_order_iced_teas +
                 cafe_latte_price * sandy_order_cafe_lattes +
                 espresso_price * sandy_order_espressos) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandy_change_theorem_l1913_191359


namespace NUMINAMATH_CALUDE_triangle_equality_l1913_191393

-- Define the triangle ADC
structure TriangleADC where
  AD : ℝ
  DC : ℝ
  D : ℝ
  h1 : AD = DC
  h2 : D = 100

-- Define the triangle CAB
structure TriangleCAB where
  CA : ℝ
  AB : ℝ
  A : ℝ
  h3 : CA = AB
  h4 : A = 20

-- Define the theorem
theorem triangle_equality (ADC : TriangleADC) (CAB : TriangleCAB) :
  CAB.AB = ADC.DC + CAB.AB - CAB.CA :=
sorry

end NUMINAMATH_CALUDE_triangle_equality_l1913_191393


namespace NUMINAMATH_CALUDE_line_equation_sum_l1913_191362

/-- Given a line passing through points (1, -2) and (4, 7), prove that m + b = -2 where y = mx + b is the equation of the line. -/
theorem line_equation_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b → 
    ((x = 1 ∧ y = -2) ∨ (x = 4 ∧ y = 7))) → 
  m + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_sum_l1913_191362


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1913_191338

-- Define the vectors
def OA : ℝ × ℝ := (-2, 4)
def OB (a : ℝ) : ℝ × ℝ := (-a, 2)
def OC (b : ℝ) : ℝ × ℝ := (b, 0)

-- Define the collinearity condition
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), B.1 - A.1 = t * (C.1 - A.1) ∧ B.2 - A.2 = t * (C.2 - A.2)

-- State the theorem
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_collinear : collinear OA (OB a) (OC b)) :
  (1 / a + 1 / b) ≥ (3 + 2 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1913_191338


namespace NUMINAMATH_CALUDE_min_value_x2_plus_2y2_l1913_191334

theorem min_value_x2_plus_2y2 (x y : ℝ) (h : x^2 - 2*x*y + 2*y^2 = 2) :
  ∃ (m : ℝ), (∀ (a b : ℝ), a^2 - 2*a*b + 2*b^2 = 2 → a^2 + 2*b^2 ≥ m) ∧
             (∃ (c d : ℝ), c^2 - 2*c*d + 2*d^2 = 2 ∧ c^2 + 2*d^2 = m) ∧
             (m = 4 - 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_x2_plus_2y2_l1913_191334


namespace NUMINAMATH_CALUDE_seven_books_arrangement_l1913_191341

/-- The number of distinct arrangements of books on a shelf -/
def book_arrangements (total : ℕ) (group1 : ℕ) (group2 : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial group1 * Nat.factorial group2)

/-- Theorem stating the number of distinct arrangements for the given book configuration -/
theorem seven_books_arrangement :
  book_arrangements 7 3 2 = 420 := by
  sorry

end NUMINAMATH_CALUDE_seven_books_arrangement_l1913_191341


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l1913_191313

def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

theorem product_trailing_zeros :
  trailingZeros 2014 = 501 := by
  sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l1913_191313


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1913_191308

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 2}

-- Define set B
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1913_191308


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_for_minimum_2_l1913_191389

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem for the first part of the problem
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≥ 4} = {x : ℝ | x ≤ 0 ∨ x ≥ 4} := by sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_minimum_2 :
  {a : ℝ | ∀ x₁ : ℝ, f a x₁ ≥ 2} = {a : ℝ | a ≥ 3 ∨ a ≤ -1} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_for_minimum_2_l1913_191389


namespace NUMINAMATH_CALUDE_point_labeling_theorem_l1913_191311

/-- A point in the space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- The set of n points in the space -/
def PointSet (n : ℕ) := Fin n → Point

theorem point_labeling_theorem (n : ℕ) (points : PointSet n) 
  (h : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → 
    ∃ (p : Fin 3 → Fin n), angle (points (p 0)) (points (p 1)) (points (p 2)) > 120) :
  ∃ (σ : Equiv (Fin n) (Fin n)), 
    ∀ (i j k : Fin n), i < j → j < k → 
      angle (points (σ i)) (points (σ j)) (points (σ k)) > 120 :=
sorry

end NUMINAMATH_CALUDE_point_labeling_theorem_l1913_191311


namespace NUMINAMATH_CALUDE_h_at_two_equals_negative_three_l1913_191333

/-- The function h(x) = -5x + 7 -/
def h (x : ℝ) : ℝ := -5 * x + 7

/-- Theorem stating that h(2) = -3 -/
theorem h_at_two_equals_negative_three : h 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_h_at_two_equals_negative_three_l1913_191333


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l1913_191325

def is_multiple_of_75 (n : ℕ) : Prop := ∃ k : ℕ, n = 75 * k

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def satisfies_conditions (n : ℕ) : Prop :=
  is_multiple_of_75 n ∧ count_divisors n = 75

theorem smallest_n_satisfying_conditions :
  ∃! n : ℕ, satisfies_conditions n ∧ ∀ m : ℕ, satisfies_conditions m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l1913_191325


namespace NUMINAMATH_CALUDE_division_simplification_l1913_191345

theorem division_simplification (a : ℝ) (h : a ≠ 0) : 6 * a^3 / (2 * a^2) = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l1913_191345


namespace NUMINAMATH_CALUDE_different_orders_count_l1913_191349

def memo_count : ℕ := 11
def processed_memos : Finset ℕ := {9, 10}

def possible_remaining_memos : Finset ℕ := Finset.range 9 ∪ {11}

def insert_positions (n : ℕ) : ℕ := n + 2

/-- The number of different orders for processing the remaining memos -/
def different_orders : ℕ :=
  (Finset.range 9).sum fun j =>
    (Nat.choose 8 j) * (insert_positions j)

theorem different_orders_count :
  different_orders = 1536 := by
  sorry

end NUMINAMATH_CALUDE_different_orders_count_l1913_191349


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1913_191327

theorem sin_2alpha_value (f : ℝ → ℝ) (a α : ℝ) :
  (∀ x, f x = 2 * Real.sin (2 * x + π / 6) + a * Real.cos (2 * x)) →
  (∀ x, f x = f (2 * π / 3 - x)) →
  0 < α →
  α < π / 3 →
  f α = 6 / 5 →
  Real.sin (2 * α) = (4 + 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1913_191327


namespace NUMINAMATH_CALUDE_stream_speed_l1913_191376

/-- Given a boat traveling a round trip with known parameters, prove the speed of the stream -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) : 
  boat_speed = 16 → 
  distance = 7560 → 
  total_time = 960 → 
  ∃ (stream_speed : ℝ), 
    stream_speed = 2 ∧ 
    distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l1913_191376


namespace NUMINAMATH_CALUDE_garden_length_l1913_191310

/-- The length of a rectangular garden with perimeter 1800 m and breadth 400 m is 500 m. -/
theorem garden_length (perimeter breadth : ℝ) (h1 : perimeter = 1800) (h2 : breadth = 400) :
  (perimeter / 2 - breadth) = 500 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_l1913_191310


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1913_191342

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {3, 4, 5}
def N : Set Nat := {2, 3}

theorem complement_intersection_theorem :
  (U \ N) ∩ M = {4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1913_191342


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l1913_191368

theorem absolute_value_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + y^2 = 5*x*y) :
  |((x + y) / (x - y))| = Real.sqrt (7/3) := by
sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l1913_191368


namespace NUMINAMATH_CALUDE_product_of_exponents_l1913_191305

theorem product_of_exponents (p r s : ℕ) : 
  (2^p + 2^3 = 18) → 
  (3^r + 3 = 30) → 
  (4^s + 4^2 = 276) → 
  p * r * s = 48 := by
sorry

end NUMINAMATH_CALUDE_product_of_exponents_l1913_191305


namespace NUMINAMATH_CALUDE_division_problem_l1913_191386

theorem division_problem (dividend : ℕ) (divisor : ℝ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 17698 →
  divisor = 198.69662921348313 →
  remainder = 14 →
  quotient = 89 →
  (dividend : ℝ) = divisor * (quotient : ℝ) + (remainder : ℝ) :=
by
  sorry

#eval (17698 : ℝ) - 198.69662921348313 * 89 - 14

end NUMINAMATH_CALUDE_division_problem_l1913_191386


namespace NUMINAMATH_CALUDE_parabola_equidistant_point_l1913_191337

/-- 
For a parabola y^2 = 2px where p > 0, with point P(2, 2p) on the parabola, 
origin O(0, 0), and focus F, the point M satisfying |MP| = |MO| = |MF| 
has coordinates (1/4, 7/4).
-/
theorem parabola_equidistant_point (p : ℝ) (h : p > 0) : 
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  let P := (2, 2*p)
  let O := (0, 0)
  let F := (p/2, 0)
  ∃ M : ℝ × ℝ, M ∈ parabola ∧ 
    dist M P = dist M O ∧ 
    dist M O = dist M F ∧ 
    M = (1/4, 7/4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equidistant_point_l1913_191337


namespace NUMINAMATH_CALUDE_range_of_a_l1913_191366

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem statement
theorem range_of_a (a : ℝ) :
  p a ∧ q a → a = 1 ∨ a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1913_191366


namespace NUMINAMATH_CALUDE_officer_average_salary_l1913_191309

/-- Proves that the average salary of officers is 450 Rs/month --/
theorem officer_average_salary
  (total_avg : ℝ)
  (non_officer_avg : ℝ)
  (officer_count : ℕ)
  (non_officer_count : ℕ)
  (h_total_avg : total_avg = 120)
  (h_non_officer_avg : non_officer_avg = 110)
  (h_officer_count : officer_count = 15)
  (h_non_officer_count : non_officer_count = 495) :
  (total_avg * (officer_count + non_officer_count) - non_officer_avg * non_officer_count) / officer_count = 450 :=
by sorry

end NUMINAMATH_CALUDE_officer_average_salary_l1913_191309


namespace NUMINAMATH_CALUDE_parallel_lines_b_value_l1913_191392

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ : ℝ} : 
  (∃ (b₁ b₂ : ℝ), ∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) → m₁ = m₂

/-- Given two lines 3y - 3b = 9x and y - 2 = (b + 9)x that are parallel, prove b = -6 -/
theorem parallel_lines_b_value (b : ℝ) :
  (∃ (y₁ y₂ : ℝ → ℝ), (∀ x, 3 * y₁ x - 3 * b = 9 * x) ∧ 
                       (∀ x, y₂ x - 2 = (b + 9) * x) ∧
                       (∀ x y, y = y₁ x ↔ y = y₂ x)) →
  b = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_b_value_l1913_191392


namespace NUMINAMATH_CALUDE_abc_inequalities_l1913_191353

theorem abc_inequalities (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a > b) (h5 : b > c) (h6 : a + b + c = 0) :
  (c / a + a / c ≤ -2) ∧ (-2 < c / a ∧ c / a < -1/2) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequalities_l1913_191353


namespace NUMINAMATH_CALUDE_cheat_sheet_distribution_l1913_191304

/-- Represents the number of pockets --/
def num_pockets : ℕ := 4

/-- Represents the number of cheat sheets --/
def num_cheat_sheets : ℕ := 6

/-- Represents the number of ways to place cheat sheets 1 and 2 --/
def ways_to_place_1_and_2 : ℕ := num_pockets

/-- Represents the number of ways to place cheat sheets 4 and 5 --/
def ways_to_place_4_and_5 : ℕ := num_pockets - 1

/-- Represents the number of ways to distribute the remaining cheat sheets --/
def ways_to_distribute_remaining : ℕ := 5

/-- Theorem stating the total number of ways to distribute the cheat sheets --/
theorem cheat_sheet_distribution :
  ways_to_place_1_and_2 * ways_to_place_4_and_5 * ways_to_distribute_remaining = 60 := by
  sorry

end NUMINAMATH_CALUDE_cheat_sheet_distribution_l1913_191304


namespace NUMINAMATH_CALUDE_minimum_concerts_required_l1913_191378

/-- Represents a concert configuration --/
structure Concert where
  performers : Finset Nat
  listeners : Finset Nat

/-- Represents the festival configuration --/
structure Festival where
  musicians : Finset Nat
  concerts : List Concert

/-- Checks if a festival configuration is valid --/
def isValidFestival (f : Festival) : Prop :=
  f.musicians.card = 6 ∧
  ∀ c ∈ f.concerts, c.performers ⊆ f.musicians ∧
                    c.listeners ⊆ f.musicians ∧
                    c.performers ∩ c.listeners = ∅ ∧
                    c.performers ∪ c.listeners = f.musicians

/-- Checks if each musician has listened to all others --/
def allMusiciansListened (f : Festival) : Prop :=
  ∀ m ∈ f.musicians, ∀ n ∈ f.musicians, m ≠ n →
    ∃ c ∈ f.concerts, m ∈ c.listeners ∧ n ∈ c.performers

/-- The main theorem --/
theorem minimum_concerts_required :
  ∀ f : Festival,
    isValidFestival f →
    allMusiciansListened f →
    f.concerts.length ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_minimum_concerts_required_l1913_191378


namespace NUMINAMATH_CALUDE_danielle_travel_time_l1913_191348

-- Define the speeds and times
def chase_speed : ℝ := 1 -- Normalized speed
def chase_time : ℝ := 180 -- Minutes
def cameron_speed : ℝ := 2 * chase_speed
def danielle_speed : ℝ := 3 * cameron_speed

-- Define the distance (constant for all travelers)
def distance : ℝ := chase_speed * chase_time

-- Theorem to prove
theorem danielle_travel_time : 
  (distance / danielle_speed) = 30 := by
sorry

end NUMINAMATH_CALUDE_danielle_travel_time_l1913_191348


namespace NUMINAMATH_CALUDE_volume_ratio_l1913_191385

variable (V_A V_B V_C : ℝ)

theorem volume_ratio 
  (h1 : V_A = (V_B + V_C) / 2)
  (h2 : V_B = (V_A + V_C) / 5)
  (h3 : V_C ≠ 0) :
  V_C / (V_A + V_B) = 1 := by
sorry

end NUMINAMATH_CALUDE_volume_ratio_l1913_191385


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1913_191363

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x ↦ 2*x^2 - 3*x - 1
  ∃ x₁ x₂ : ℝ, x₁ = (3 + Real.sqrt 17) / 4 ∧ 
             x₂ = (3 - Real.sqrt 17) / 4 ∧ 
             f x₁ = 0 ∧ f x₂ = 0 ∧
             ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1913_191363


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l1913_191301

theorem absolute_value_equation_product (y₁ y₂ : ℝ) : 
  (|3 * y₁| + 7 = 40) ∧ (|3 * y₂| + 7 = 40) ∧ (y₁ ≠ y₂) → y₁ * y₂ = -121 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l1913_191301


namespace NUMINAMATH_CALUDE_center_transformation_l1913_191346

/-- Reflects a point across the y-axis -/
def reflectY (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Rotates a point 90 degrees clockwise around the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

/-- Translates a point up by a given amount -/
def translateUp (p : ℝ × ℝ) (amount : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + amount)

/-- Applies all transformations to a point -/
def applyTransformations (p : ℝ × ℝ) : ℝ × ℝ :=
  translateUp (rotate90Clockwise (reflectY p)) 4

theorem center_transformation :
  applyTransformations (3, -5) = (-5, 7) := by
  sorry

end NUMINAMATH_CALUDE_center_transformation_l1913_191346


namespace NUMINAMATH_CALUDE_cyclist_distance_difference_l1913_191312

/-- The difference in distance traveled by two cyclists after five hours -/
theorem cyclist_distance_difference
  (daniel_distance : ℝ)
  (evan_initial_distance : ℝ)
  (evan_initial_time : ℝ)
  (evan_break_time : ℝ)
  (total_time : ℝ)
  (h1 : daniel_distance = 65)
  (h2 : evan_initial_distance = 40)
  (h3 : evan_initial_time = 3)
  (h4 : evan_break_time = 0.5)
  (h5 : total_time = 5) :
  daniel_distance - (evan_initial_distance + (evan_initial_distance / evan_initial_time) * (total_time - evan_initial_time - evan_break_time)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_distance_difference_l1913_191312


namespace NUMINAMATH_CALUDE_odd_function_inequality_l1913_191397

-- Define f as a function from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the property of f being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the condition for x > 0
def positive_condition (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, x * (deriv f x) + 2 * f x > 0

-- State the theorem
theorem odd_function_inequality (h1 : is_odd f) (h2 : positive_condition f) :
  4 * f 2 < 9 * f 3 :=
sorry

end NUMINAMATH_CALUDE_odd_function_inequality_l1913_191397


namespace NUMINAMATH_CALUDE_max_xy_on_line_segment_l1913_191375

/-- Given points A(2,0) and B(0,1), prove that the maximum value of xy for any point P(x,y) on the line segment AB is 1/2 -/
theorem max_xy_on_line_segment : 
  ∀ x y : ℝ, 
  0 ≤ x ∧ x ≤ 2 → -- Condition for x being on the line segment
  x / 2 + y = 1 → -- Equation of the line AB
  x * y ≤ (1 : ℝ) / 2 ∧ 
  ∃ x₀ y₀ : ℝ, 0 ≤ x₀ ∧ x₀ ≤ 2 ∧ x₀ / 2 + y₀ = 1 ∧ x₀ * y₀ = (1 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_on_line_segment_l1913_191375


namespace NUMINAMATH_CALUDE_units_digit_of_33_power_l1913_191394

theorem units_digit_of_33_power : ∃ n : ℕ, 33^(33*(7^7)) ≡ 7 [ZMOD 10] :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_33_power_l1913_191394


namespace NUMINAMATH_CALUDE_equation_and_inequality_solution_l1913_191358

theorem equation_and_inequality_solution :
  (∃ x : ℝ, (x - 3) * (x - 2) + 18 = (x + 9) * (x + 1) ∧ x = 1) ∧
  (∀ x : ℝ, (3 * x + 4) * (3 * x - 4) < 9 * (x - 2) * (x + 3) ↔ x > 38 / 9) :=
by sorry

end NUMINAMATH_CALUDE_equation_and_inequality_solution_l1913_191358


namespace NUMINAMATH_CALUDE_unknown_number_proof_l1913_191339

theorem unknown_number_proof (x : ℝ) : x^2 + 94^2 = 19872 → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l1913_191339


namespace NUMINAMATH_CALUDE_tan_675_degrees_l1913_191390

theorem tan_675_degrees (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (675 * π / 180) →
  n = 135 ∨ n = -45 := by
sorry

end NUMINAMATH_CALUDE_tan_675_degrees_l1913_191390


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1913_191336

theorem tan_alpha_plus_pi_fourth (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  Real.tan (α + π / 4) = 3 / 22 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1913_191336


namespace NUMINAMATH_CALUDE_sara_hotdog_cost_l1913_191371

/-- The cost of Sara's lunch items -/
structure LunchCost where
  total : ℝ
  salad : ℝ
  hotdog : ℝ

/-- Sara's lunch satisfies the given conditions -/
def sara_lunch : LunchCost where
  total := 10.46
  salad := 5.1
  hotdog := 5.36

/-- Theorem: Sara's hotdog cost $5.36 -/
theorem sara_hotdog_cost : sara_lunch.hotdog = 5.36 := by
  sorry

#check sara_hotdog_cost

end NUMINAMATH_CALUDE_sara_hotdog_cost_l1913_191371


namespace NUMINAMATH_CALUDE_tree_growth_rate_l1913_191340

/-- Proves that a tree growing from 52 feet to 92 feet in 8 years has an annual growth rate of 5 feet --/
theorem tree_growth_rate (initial_height : ℝ) (final_height : ℝ) (years : ℕ) 
  (h1 : initial_height = 52)
  (h2 : final_height = 92)
  (h3 : years = 8) :
  (final_height - initial_height) / years = 5 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_rate_l1913_191340


namespace NUMINAMATH_CALUDE_chessboard_pawn_placement_l1913_191300

/-- Represents a chess board configuration -/
structure ChessBoard :=
  (size : Nat)
  (pawns : Nat)

/-- Calculates the number of ways to place distinct pawns on a chess board -/
def placementWays (board : ChessBoard) : Nat :=
  (Nat.factorial board.size) * (Nat.factorial board.size)

/-- Theorem: The number of ways to place 5 distinct pawns on a 5x5 chess board,
    such that no row and no column contains more than one pawn, is 14400 -/
theorem chessboard_pawn_placement :
  let board : ChessBoard := ⟨5, 5⟩
  placementWays board = 14400 := by
  sorry

#eval placementWays ⟨5, 5⟩

end NUMINAMATH_CALUDE_chessboard_pawn_placement_l1913_191300


namespace NUMINAMATH_CALUDE_cubic_sum_inequality_l1913_191316

theorem cubic_sum_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  2 * (a^3 + b^3 + c^3) ≥ a^2*b + a*b^2 + a^2*c + a*c^2 + b^2*c + b*c^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_inequality_l1913_191316


namespace NUMINAMATH_CALUDE_initial_candies_count_l1913_191357

-- Define the given conditions
def candies_given_to_chloe : ℝ := 28.0
def candies_left : ℕ := 6

-- Define the theorem to prove
theorem initial_candies_count : 
  candies_given_to_chloe + candies_left = 34.0 := by
  sorry

end NUMINAMATH_CALUDE_initial_candies_count_l1913_191357


namespace NUMINAMATH_CALUDE_complex_number_condition_l1913_191343

/-- A complex number z satisfying the given conditions -/
def Z : ℂ := sorry

/-- The real part of Z -/
def m : ℝ := Z.re

/-- The imaginary part of Z -/
def n : ℝ := Z.im

/-- The condition that z+2i is a real number -/
axiom h1 : (Z + 2*Complex.I).im = 0

/-- The condition that z/(2-i) is a real number -/
axiom h2 : ((Z / (2 - Complex.I))).im = 0

/-- Definition of the function representing (z+ai)^2 -/
def f (a : ℝ) : ℂ := (Z + a*Complex.I)^2

/-- The theorem to be proved -/
theorem complex_number_condition (a : ℝ) :
  (f a).re < 0 ∧ (f a).im > 0 → a > 6 := by sorry

end NUMINAMATH_CALUDE_complex_number_condition_l1913_191343


namespace NUMINAMATH_CALUDE_range_of_p_l1913_191331

def h (x : ℝ) : ℝ := 4 * x - 3

def p (x : ℝ) : ℝ := h (h (h x))

theorem range_of_p :
  ∀ y ∈ Set.range (fun x => p x),
  (1 ≤ x ∧ x ≤ 3) → (1 ≤ y ∧ y ≤ 129) :=
by sorry

end NUMINAMATH_CALUDE_range_of_p_l1913_191331


namespace NUMINAMATH_CALUDE_least_integer_with_ten_factors_l1913_191395

-- Define a function to count the number of distinct positive factors
def countFactors (n : ℕ) : ℕ := sorry

-- Define a function to check if a number has exactly ten distinct positive factors
def hasTenFactors (n : ℕ) : Prop :=
  countFactors n = 10

-- Theorem statement
theorem least_integer_with_ten_factors :
  ∀ n : ℕ, n > 0 → hasTenFactors n → n ≥ 48 :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_ten_factors_l1913_191395


namespace NUMINAMATH_CALUDE_expression_simplification_l1913_191319

theorem expression_simplification :
  ((1 + 2 + 3 + 4 + 5 + 6) / 3) + ((3 * 5 + 12) / 4) = 13.75 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l1913_191319


namespace NUMINAMATH_CALUDE_sports_equipment_purchase_l1913_191354

/-- Represents the cost function for Scheme A -/
def cost_scheme_a (x : ℕ) : ℝ := 25 * x + 550

/-- Represents the cost function for Scheme B -/
def cost_scheme_b (x : ℕ) : ℝ := 22.5 * x + 720

theorem sports_equipment_purchase :
  /- Cost functions are correct -/
  (∀ x : ℕ, x ≥ 10 → cost_scheme_a x = 25 * x + 550 ∧ cost_scheme_b x = 22.5 * x + 720) ∧
  /- Scheme A is more cost-effective for 15 boxes -/
  cost_scheme_a 15 < cost_scheme_b 15 ∧
  /- Scheme A allows purchasing more balls with 1800 yuan budget -/
  (∃ x_a x_b : ℕ, cost_scheme_a x_a ≤ 1800 ∧ cost_scheme_b x_b ≤ 1800 ∧ x_a > x_b) :=
by sorry

end NUMINAMATH_CALUDE_sports_equipment_purchase_l1913_191354


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l1913_191381

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| < 2}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_intersection_A_B :
  (Set.univ \ (A ∩ B)) = {x : ℝ | x < 1 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l1913_191381


namespace NUMINAMATH_CALUDE_promotion_savings_difference_l1913_191361

/-- Calculates the total cost of two pairs of shoes under a given promotion --/
def promotionCost (regularPrice : ℝ) (discountPercent : ℝ) : ℝ :=
  regularPrice + (regularPrice * (1 - discountPercent))

/-- Represents the difference in savings between two promotions --/
def savingsDifference (regularPrice : ℝ) (discountA : ℝ) (discountB : ℝ) : ℝ :=
  promotionCost regularPrice discountB - promotionCost regularPrice discountA

theorem promotion_savings_difference :
  savingsDifference 50 0.3 0.2 = 5 := by sorry

end NUMINAMATH_CALUDE_promotion_savings_difference_l1913_191361


namespace NUMINAMATH_CALUDE_factory_production_l1913_191344

/-- Represents the number of toys produced in a week at a factory -/
def toysPerWeek (daysWorked : ℕ) (toysPerDay : ℕ) : ℕ :=
  daysWorked * toysPerDay

/-- Theorem stating that the factory produces 4560 toys per week -/
theorem factory_production :
  toysPerWeek 4 1140 = 4560 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_l1913_191344


namespace NUMINAMATH_CALUDE_part_one_part_two_l1913_191315

-- Define propositions p and q
def p (x a : ℝ) : Prop := (x + a) / (x - 3 * a) < 0 ∧ a > 0

def q (x : ℝ) : Prop := x^2 - 5*x + 6 ≤ 0

-- Part I
theorem part_one (x : ℝ) (h : p x 1 ∧ q x) : 2 ≤ x ∧ x < 3 := by sorry

-- Part II
theorem part_two (a : ℝ) (h : ∀ x, q x → p x a) (h' : ∃ x, p x a ∧ ¬q x) : a > 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1913_191315


namespace NUMINAMATH_CALUDE_ana_overall_percentage_l1913_191323

-- Define the number of problems and percentage correct for each test
def test1_problems : ℕ := 20
def test1_percent : ℚ := 75 / 100

def test2_problems : ℕ := 50
def test2_percent : ℚ := 85 / 100

def test3_problems : ℕ := 30
def test3_percent : ℚ := 80 / 100

-- Define the total number of problems
def total_problems : ℕ := test1_problems + test2_problems + test3_problems

-- Define the total number of correct answers
def total_correct : ℚ := test1_problems * test1_percent + test2_problems * test2_percent + test3_problems * test3_percent

-- Theorem statement
theorem ana_overall_percentage :
  (total_correct / total_problems : ℚ) = 815 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_ana_overall_percentage_l1913_191323


namespace NUMINAMATH_CALUDE_sum_of_three_roots_is_zero_l1913_191398

/-- Given two quadratic polynomials with coefficients a and b, 
    where each has two distinct roots and their product has exactly three distinct roots,
    prove that the sum of these three roots is 0. -/
theorem sum_of_three_roots_is_zero (a b : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₃ ≠ x₄ ∧ 
    (∀ x : ℝ, x^2 + a*x + b = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
    (∀ x : ℝ, x^2 + b*x + a = 0 ↔ (x = x₃ ∨ x = x₄))) →
  (∃! y₁ y₂ y₃ : ℝ, y₁ ≠ y₂ ∧ y₁ ≠ y₃ ∧ y₂ ≠ y₃ ∧
    (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + b*x + a) = 0 ↔ (x = y₁ ∨ x = y₂ ∨ x = y₃))) →
  ∃ y₁ y₂ y₃ : ℝ, y₁ + y₂ + y₃ = 0 :=
by sorry


end NUMINAMATH_CALUDE_sum_of_three_roots_is_zero_l1913_191398


namespace NUMINAMATH_CALUDE_seashells_to_find_l1913_191372

def current_seashells : ℕ := 307
def target_seashells : ℕ := 500

theorem seashells_to_find : target_seashells - current_seashells = 193 := by
  sorry

end NUMINAMATH_CALUDE_seashells_to_find_l1913_191372


namespace NUMINAMATH_CALUDE_vasyas_numbers_l1913_191306

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l1913_191306


namespace NUMINAMATH_CALUDE_largest_sum_is_five_sixths_l1913_191329

theorem largest_sum_is_five_sixths : 
  let sums := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/2, 1/3 + 1/9, 1/3 + 1/6]
  ∀ x ∈ sums, x ≤ 5/6 ∧ (5/6 : ℚ) ∈ sums :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_is_five_sixths_l1913_191329
