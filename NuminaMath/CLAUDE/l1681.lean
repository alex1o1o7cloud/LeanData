import Mathlib

namespace NUMINAMATH_CALUDE_coefficient_a3b2_in_expansion_l1681_168117

theorem coefficient_a3b2_in_expansion : ∃ (coeff : ℕ),
  coeff = (Nat.choose 5 3) * (Nat.choose 8 4) ∧
  coeff = 700 := by sorry

end NUMINAMATH_CALUDE_coefficient_a3b2_in_expansion_l1681_168117


namespace NUMINAMATH_CALUDE_set_equality_implies_x_values_l1681_168199

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

-- State the theorem
theorem set_equality_implies_x_values (x : ℝ) :
  A x ∪ B x = A x → x = 2 ∨ x = -2 ∨ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_x_values_l1681_168199


namespace NUMINAMATH_CALUDE_second_tap_empty_time_l1681_168197

-- Define the filling time of the first tap
def fill_time : ℝ := 3

-- Define the simultaneous filling time when both taps are open
def simultaneous_fill_time : ℝ := 4.2857142857142865

-- Define the emptying time of the second tap
def empty_time : ℝ := 10

-- Theorem statement
theorem second_tap_empty_time :
  (1 / fill_time - 1 / empty_time = 1 / simultaneous_fill_time) ∧
  empty_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_tap_empty_time_l1681_168197


namespace NUMINAMATH_CALUDE_board_9x16_fills_12x12_square_l1681_168194

/-- Represents a rectangular board with integer dimensions -/
structure Board where
  width : ℕ
  length : ℕ

/-- Represents a square hole with integer side length -/
structure Square where
  side : ℕ

/-- Checks if a board can be cut to fill a square hole using the staircase method -/
def canFillSquare (b : Board) (s : Square) : Prop :=
  ∃ (steps : ℕ), 
    steps > 0 ∧
    b.width * (steps + 1) = s.side ∧
    b.length * steps = s.side

theorem board_9x16_fills_12x12_square :
  canFillSquare (Board.mk 9 16) (Square.mk 12) :=
sorry

end NUMINAMATH_CALUDE_board_9x16_fills_12x12_square_l1681_168194


namespace NUMINAMATH_CALUDE_restaurant_bill_theorem_l1681_168175

theorem restaurant_bill_theorem (num_people : ℕ) (total_bill : ℚ) (gratuity_rate : ℚ) :
  num_people = 7 →
  total_bill = 840 →
  gratuity_rate = 1/5 →
  (total_bill / (1 + gratuity_rate)) / num_people = 100 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_theorem_l1681_168175


namespace NUMINAMATH_CALUDE_twelfth_term_of_arithmetic_progression_l1681_168136

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

/-- Theorem: The 12th term of an arithmetic progression with first term 2 and common difference 8 is 90 -/
theorem twelfth_term_of_arithmetic_progression :
  arithmeticProgressionTerm 2 8 12 = 90 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_arithmetic_progression_l1681_168136


namespace NUMINAMATH_CALUDE_smallest_n_exceeding_1500_l1681_168187

def exterior_sum (n : ℕ) : ℕ := 8 + 24 * (n - 2) + 12 * (n - 2)^2

theorem smallest_n_exceeding_1500 :
  ∀ n : ℕ, n ≥ 13 ↔ exterior_sum n > 1500 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_exceeding_1500_l1681_168187


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1681_168149

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ, 
  let a : ℝ × ℝ := (x, -1)
  let b : ℝ × ℝ := (4, 2)
  parallel a b → x = -2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1681_168149


namespace NUMINAMATH_CALUDE_lowest_common_denominator_l1681_168141

theorem lowest_common_denominator (a b c : Nat) : a = 9 → b = 4 → c = 18 → Nat.lcm a (Nat.lcm b c) = 36 := by
  sorry

end NUMINAMATH_CALUDE_lowest_common_denominator_l1681_168141


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1681_168170

theorem arithmetic_sequence_sum (a₁ d n : ℝ) (S_n : ℕ → ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    y₁ = a₁ * x₁ ∧ 
    y₂ = a₁ * x₂ ∧ 
    (x₁ - 2)^2 + y₁^2 = 1 ∧ 
    (x₂ - 2)^2 + y₂^2 = 1 ∧
    (x₁ + y₁ + d) = -(x₂ + y₂ + d)) →
  (∀ k : ℕ, S_n k = k * a₁ + k * (k - 1) / 2 * d) →
  ∀ k : ℕ, S_n k = 2*k - k^2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1681_168170


namespace NUMINAMATH_CALUDE_parabola_equation_l1681_168124

/-- Given a parabola C: y^2 = 2px and a circle x^2 + y^2 - 2x - 15 = 0,
    if the focus of the parabola coincides with the center of the circle,
    then the equation of the parabola C is y^2 = 4x. -/
theorem parabola_equation (p : ℝ) :
  (∃ (x y : ℝ), y^2 = 2*p*x ∧ x^2 + y^2 - 2*x - 15 = 0 ∧
   (1, 0) = (x + p/2, 0)) →
  (∀ (x y : ℝ), y^2 = 2*p*x ↔ y^2 = 4*x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1681_168124


namespace NUMINAMATH_CALUDE_chosen_number_proof_l1681_168183

theorem chosen_number_proof : ∃ x : ℚ, (x / 8 : ℚ) - 100 = 6 ∧ x = 848 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l1681_168183


namespace NUMINAMATH_CALUDE_equal_means_sum_l1681_168111

theorem equal_means_sum (group1 group2 : Finset ℕ) : 
  (Finset.card group1 = 10) →
  (Finset.card group2 = 207) →
  (group1 ∪ group2 = Finset.range 217) →
  (group1 ∩ group2 = ∅) →
  (Finset.sum group1 id / Finset.card group1 = Finset.sum group2 id / Finset.card group2) →
  Finset.sum group1 id = 1090 := by
sorry

end NUMINAMATH_CALUDE_equal_means_sum_l1681_168111


namespace NUMINAMATH_CALUDE_sum_equals_eight_l1681_168167

theorem sum_equals_eight (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_ineq1 : b * (a + b + c) + a * c ≥ 16)
  (h_ineq2 : a + 2 * b + c ≤ 8) : 
  a + 2 * b + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_eight_l1681_168167


namespace NUMINAMATH_CALUDE_investment_change_l1681_168193

theorem investment_change (initial_investment : ℝ) 
  (loss_rate1 loss_rate3 gain_rate2 : ℝ) : 
  initial_investment = 200 →
  loss_rate1 = 0.1 →
  gain_rate2 = 0.15 →
  loss_rate3 = 0.05 →
  let year1 := initial_investment * (1 - loss_rate1)
  let year2 := year1 * (1 + gain_rate2)
  let year3 := year2 * (1 - loss_rate3)
  let percent_change := (year3 - initial_investment) / initial_investment * 100
  ∃ ε > 0, |percent_change + 1.68| < ε :=
by sorry

end NUMINAMATH_CALUDE_investment_change_l1681_168193


namespace NUMINAMATH_CALUDE_solution_set_inequality_holds_max_m_value_l1681_168174

-- Define the function f
def f (x : ℝ) := |2*x + 1| + |3*x - 2|

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -4/5 ≤ x ∧ x ≤ 6/5} :=
sorry

-- Theorem for the inequality |x-1| + |x+2| ≥ 3
theorem inequality_holds (x : ℝ) :
  |x - 1| + |x + 2| ≥ 3 :=
sorry

-- Theorem for the maximum value of m
theorem max_m_value :
  ∃ (m : ℝ), m = 2 ∧ 
  (∀ (x : ℝ), m^2 - 3*m + 5 ≤ |x - 1| + |x + 2|) ∧
  (∀ (m' : ℝ), (∀ (x : ℝ), m'^2 - 3*m' + 5 ≤ |x - 1| + |x + 2|) → m' ≤ m) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_holds_max_m_value_l1681_168174


namespace NUMINAMATH_CALUDE_problem_solution_l1681_168122

theorem problem_solution (p q : Prop) 
  (h1 : p ∨ q) 
  (h2 : ¬p) : 
  ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1681_168122


namespace NUMINAMATH_CALUDE_lisa_marble_distribution_l1681_168132

/-- The minimum number of additional marbles needed -/
def additional_marbles_needed (friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  let required_marbles := friends * (friends + 1) / 2
  max (required_marbles - initial_marbles) 0

/-- Theorem stating the solution to Lisa's marble distribution problem -/
theorem lisa_marble_distribution (friends : ℕ) (initial_marbles : ℕ)
    (h1 : friends = 12)
    (h2 : initial_marbles = 50) :
    additional_marbles_needed friends initial_marbles = 28 := by
  sorry

end NUMINAMATH_CALUDE_lisa_marble_distribution_l1681_168132


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1681_168110

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 6) = 7 → x = 43 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1681_168110


namespace NUMINAMATH_CALUDE_books_on_shelf_l1681_168163

/-- The number of books remaining on a shelf after some are removed. -/
def booksRemaining (initial : ℝ) (removed : ℝ) : ℝ :=
  initial - removed

theorem books_on_shelf (initial : ℝ) (removed : ℝ) :
  initial ≥ removed →
  booksRemaining initial removed = initial - removed :=
by
  sorry

end NUMINAMATH_CALUDE_books_on_shelf_l1681_168163


namespace NUMINAMATH_CALUDE_root_sum_product_reciprocal_sum_l1681_168156

theorem root_sum_product_reciprocal_sum (α β : ℝ) (hα : α ≠ 0) (hβ : β ≠ 0) 
  (x₁ x₂ x₃ : ℝ) (hroots : ∀ x, α * x^3 - α * x^2 + β * x + β = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) :
  (x₁ + x₂ + x₃) * (1/x₁ + 1/x₂ + 1/x₃) = -1 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_product_reciprocal_sum_l1681_168156


namespace NUMINAMATH_CALUDE_triangle_side_length_l1681_168177

theorem triangle_side_length (a b : ℝ) (C : ℝ) (h1 : a = 3) (h2 : b = 5) (h3 : C = 2 * π / 3) :
  let c := Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos C)
  c = 7 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1681_168177


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1681_168139

/-- The eccentricity of a hyperbola with equation x^2 - y^2 = k is sqrt(2) -/
theorem hyperbola_eccentricity (k : ℝ) (h : k > 0) :
  let e := Real.sqrt (1 + (Real.sqrt k / Real.sqrt k)^2)
  e = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1681_168139


namespace NUMINAMATH_CALUDE_mike_limes_l1681_168152

-- Define the given conditions
def total_limes : ℕ := 57
def alyssa_limes : ℕ := 25

-- State the theorem
theorem mike_limes : total_limes - alyssa_limes = 32 := by
  sorry

end NUMINAMATH_CALUDE_mike_limes_l1681_168152


namespace NUMINAMATH_CALUDE_a_value_theorem_l1681_168153

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x + 1

theorem a_value_theorem (a : ℝ) :
  (∀ x > 0, HasDerivAt (f a) ((a * Real.log x) + a) x) →
  HasDerivAt (f a) 2 1 →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_value_theorem_l1681_168153


namespace NUMINAMATH_CALUDE_point_on_graph_l1681_168137

def f (x : ℝ) : ℝ := |x^3 + 1| + |x^3 - 1|

theorem point_on_graph (a : ℝ) : (a, f (-a)) ∈ {p : ℝ × ℝ | p.2 = f p.1} := by
  sorry

end NUMINAMATH_CALUDE_point_on_graph_l1681_168137


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l1681_168113

-- Problem 1
theorem problem_one : 
  (2 / 3 : ℝ) * Real.sqrt 24 / (-Real.sqrt 3) * (1 / 3 : ℝ) * Real.sqrt 27 = -(4 / 3 : ℝ) * Real.sqrt 6 := by
  sorry

-- Problem 2
theorem problem_two : 
  Real.sqrt 3 * Real.sqrt 12 + (Real.sqrt 3 + 1)^2 = 10 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l1681_168113


namespace NUMINAMATH_CALUDE_intersection_A_B_l1681_168196

def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}

def B : Set ℝ := {-2, -1, 1, 2, 4}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1681_168196


namespace NUMINAMATH_CALUDE_correct_operations_l1681_168155

theorem correct_operations (a b : ℝ) : 
  (2 * a * (3 * b) = 6 * a * b) ∧ ((-a^3)^2 = a^6) := by
  sorry

end NUMINAMATH_CALUDE_correct_operations_l1681_168155


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l1681_168188

/-- A line with equation y = kx + 1 -/
structure Line (k : ℝ) where
  eq : ℝ → ℝ
  h : ∀ x, eq x = k * x + 1

/-- A parabola with equation y² = 4x -/
structure Parabola where
  eq : ℝ → ℝ → Prop
  h : ∀ x y, eq x y ↔ y^2 = 4*x

/-- The number of intersection points between a line and a parabola -/
def intersectionCount (l : Line k) (p : Parabola) : ℕ :=
  sorry

theorem line_parabola_intersection (k : ℝ) (l : Line k) (p : Parabola) :
  intersectionCount l p = 1 → k = 0 ∨ k = 1 :=
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l1681_168188


namespace NUMINAMATH_CALUDE_square_root_b_minus_a_l1681_168138

/-- Given that the square roots of a positive number are 2-3a and a+2,
    and the cube root of 5a+3b-1 is 3, prove that the square root of b-a is 2. -/
theorem square_root_b_minus_a (a b : ℝ) : 
  (∃ (k : ℝ), k > 0 ∧ (2 - 3*a)^2 = k ∧ (a + 2)^2 = k) →  -- square roots condition
  (5*a + 3*b - 1)^(1/3) = 3 →                             -- cube root condition
  Real.sqrt (b - a) = 2 := by
sorry

end NUMINAMATH_CALUDE_square_root_b_minus_a_l1681_168138


namespace NUMINAMATH_CALUDE_opposite_sides_range_l1681_168179

def line_equation (x y a : ℝ) : ℝ := x - 2*y + a

theorem opposite_sides_range (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ = 2 ∧ y₁ = -1 ∧ x₂ = -3 ∧ y₂ = 2 ∧ 
    (line_equation x₁ y₁ a) * (line_equation x₂ y₂ a) < 0) ↔ 
  -4 < a ∧ a < 7 :=
sorry

end NUMINAMATH_CALUDE_opposite_sides_range_l1681_168179


namespace NUMINAMATH_CALUDE_gross_profit_calculation_l1681_168162

theorem gross_profit_calculation (sales_price : ℝ) (gross_profit_percentage : ℝ) :
  sales_price = 91 →
  gross_profit_percentage = 1.6 →
  ∃ (cost : ℝ) (gross_profit : ℝ),
    cost > 0 ∧
    gross_profit = gross_profit_percentage * cost ∧
    sales_price = cost + gross_profit ∧
    gross_profit = 56 :=
by sorry

end NUMINAMATH_CALUDE_gross_profit_calculation_l1681_168162


namespace NUMINAMATH_CALUDE_min_value_theorem_l1681_168145

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  4 * x + 1 / x^6 ≥ 5 ∧ ∃ y : ℝ, y > 0 ∧ 4 * y + 1 / y^6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1681_168145


namespace NUMINAMATH_CALUDE_distance_circle_center_to_point_l1681_168107

-- Define the circle's equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 6*x - 2*y + 9

-- Define the center of the circle
def circle_center : ℝ × ℝ := (3, -1)

-- Define the given point
def given_point : ℝ × ℝ := (-3, 4)

-- Theorem statement
theorem distance_circle_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := given_point
  Real.sqrt ((cx - px)^2 + (cy - py)^2) = Real.sqrt 61 :=
by sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_point_l1681_168107


namespace NUMINAMATH_CALUDE_circle_equation_proof_line_equation_proof_l1681_168166

-- Define the points
def A : ℝ × ℝ := (-4, -3)
def B : ℝ × ℝ := (2, 9)
def P : ℝ × ℝ := (0, 2)

-- Define the circle C with AB as its diameter
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 3)^2 = 45}

-- Define the line l₀
def l₀ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 + 2 = 0}

theorem circle_equation_proof :
  C = {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 3)^2 = 45} :=
sorry

theorem line_equation_proof :
  l₀ = {p : ℝ × ℝ | p.1 - p.2 + 2 = 0} :=
sorry

end NUMINAMATH_CALUDE_circle_equation_proof_line_equation_proof_l1681_168166


namespace NUMINAMATH_CALUDE_car_repair_cost_l1681_168135

/-- Proves that the repair cost is approximately 13000, given the initial cost,
    selling price, and profit percentage of a car sale. -/
theorem car_repair_cost (initial_cost selling_price : ℕ) (profit_percentage : ℚ) :
  initial_cost = 42000 →
  selling_price = 66900 →
  profit_percentage = 21636363636363637 / 100000000000000 →
  ∃ (repair_cost : ℕ), 
    (repair_cost ≥ 12999 ∧ repair_cost ≤ 13001) ∧
    profit_percentage = (selling_price - (initial_cost + repair_cost)) / (initial_cost + repair_cost) :=
by sorry

end NUMINAMATH_CALUDE_car_repair_cost_l1681_168135


namespace NUMINAMATH_CALUDE_magnitude_of_sum_l1681_168151

/-- Given two vectors a and b in ℝ², prove that under certain conditions, 
    the magnitude of a + 2b is √29. -/
theorem magnitude_of_sum (a b : ℝ × ℝ) : 
  (a.1 = 4 ∧ a.2 = 3) → -- a = (4, 3)
  (a.1 * b.1 + a.2 * b.2 = 0) → -- a ⟂ b (dot product is 0)
  (b.1^2 + b.2^2 = 1) → -- |b| = 1
  ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2 = 29) := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_sum_l1681_168151


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l1681_168180

/-- Proves that a cube with edge length 7 cm, when cut into 1 cm cubes, results in a 600% increase in surface area. -/
theorem cube_surface_area_increase : 
  let original_edge_length : ℝ := 7
  let original_surface_area : ℝ := 6 * original_edge_length^2
  let new_surface_area : ℝ := 6 * original_edge_length^3
  new_surface_area = 7 * original_surface_area := by
  sorry

#check cube_surface_area_increase

end NUMINAMATH_CALUDE_cube_surface_area_increase_l1681_168180


namespace NUMINAMATH_CALUDE_A_is_half_of_B_l1681_168140

def A : ℕ → ℕ
| 0 => 0
| (n + 1) => A n + (n + 1) * (2023 - n)

def B : ℕ → ℕ
| 0 => 0
| (n + 1) => B n + (n + 1) * (2024 - n)

theorem A_is_half_of_B : A 2022 = (B 2022) / 2 := by
  sorry

end NUMINAMATH_CALUDE_A_is_half_of_B_l1681_168140


namespace NUMINAMATH_CALUDE_table_formula_proof_l1681_168105

def f (x : ℕ) : ℕ := x^2 + 3*x + 1

theorem table_formula_proof :
  (f 1 = 5) ∧ (f 2 = 11) ∧ (f 3 = 19) ∧ (f 4 = 29) ∧ (f 5 = 41) :=
by sorry

end NUMINAMATH_CALUDE_table_formula_proof_l1681_168105


namespace NUMINAMATH_CALUDE_added_number_after_doubling_l1681_168126

theorem added_number_after_doubling (x : ℕ) (y : ℕ) (h : x = 19) :
  3 * (2 * x + y) = 129 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_added_number_after_doubling_l1681_168126


namespace NUMINAMATH_CALUDE_prime_condition_characterization_l1681_168185

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- The main theorem statement -/
theorem prime_condition_characterization (n : ℕ) :
  (n > 0 ∧ ∀ k : ℕ, k < n → is_prime (4 * k^2 + n)) ↔ (n = 3 ∨ n = 7) :=
sorry

end NUMINAMATH_CALUDE_prime_condition_characterization_l1681_168185


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l1681_168178

theorem lcm_from_product_and_hcf (a b : ℕ+) 
  (h1 : a * b = 62216)
  (h2 : Nat.gcd a b = 22) : 
  Nat.lcm a b = 2828 := by
sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l1681_168178


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1681_168198

/-- Given a line L1 with equation 3x - 4y + 6 = 0 and a point P(4, -1),
    the line L2 passing through P and perpendicular to L1 has equation 4x + 3y - 13 = 0 -/
theorem perpendicular_line_equation :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 4 * y + 6 = 0
  let P : ℝ × ℝ := (4, -1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 4 * x + 3 * y - 13 = 0
  (∀ x y, L2 x y ↔ (y - P.2 = -(4/3) * (x - P.1))) ∧
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L2 x₂ y₂ → 
    ((x₂ - x₁) * (3/4) + (y₂ - y₁) * (-1) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1681_168198


namespace NUMINAMATH_CALUDE_translated_line_equation_translation_result_l1681_168123

/-- A line in the xy-plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically by a given amount -/
def translate_line_vertical (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + amount }

theorem translated_line_equation (original : Line) (translation : ℝ) :
  let translated := translate_line_vertical original (-translation)
  translated.slope = original.slope ∧
  translated.intercept = original.intercept - translation :=
by sorry

/-- The original line y = -2x + 1 -/
def original_line : Line :=
  { slope := -2, intercept := 1 }

/-- The amount of downward translation -/
def translation_amount : ℝ := 4

theorem translation_result :
  let translated := translate_line_vertical original_line (-translation_amount)
  translated.slope = -2 ∧ translated.intercept = -3 :=
by sorry

end NUMINAMATH_CALUDE_translated_line_equation_translation_result_l1681_168123


namespace NUMINAMATH_CALUDE_unique_point_on_curve_l1681_168127

/-- A point (x, y) is in the second quadrant if x < 0 and y > 0 -/
def second_quadrant (x y : ℤ) : Prop := x < 0 ∧ y > 0

/-- A point (x, y) is on the curve if x^2 + 5x + 1 = 3y -/
def on_curve (x y : ℤ) : Prop := x^2 + 5*x + 1 = 3*y

theorem unique_point_on_curve : 
  ∀ x y : ℤ, second_quadrant x y → on_curve x y → (x = -7 ∧ y = 5) :=
by sorry

end NUMINAMATH_CALUDE_unique_point_on_curve_l1681_168127


namespace NUMINAMATH_CALUDE_square_side_length_from_rectangle_l1681_168154

/-- The side length of a square with an area 7 times larger than a rectangle with length 400 feet and width 300 feet is approximately 916.515 feet. -/
theorem square_side_length_from_rectangle (ε : ℝ) (h : ε > 0) : ∃ (s : ℝ), 
  abs (s - Real.sqrt (7 * 400 * 300)) < ε ∧ 
  s^2 = 7 * 400 * 300 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_from_rectangle_l1681_168154


namespace NUMINAMATH_CALUDE_sets_properties_l1681_168129

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem statement
theorem sets_properties :
  (A ∩ B = {x | 3 ≤ x ∧ x < 6}) ∧
  (A ∪ B = {x | 2 < x ∧ x < 9}) ∧
  (∀ a : ℝ, C a ⊆ B → (2 < a ∧ a < 8)) :=
by sorry

end NUMINAMATH_CALUDE_sets_properties_l1681_168129


namespace NUMINAMATH_CALUDE_valid_speaking_orders_eq_1080_l1681_168158

-- Define the number of candidates
def num_candidates : ℕ := 8

-- Define the number of speakers to be selected
def num_speakers : ℕ := 4

-- Define a function to calculate the number of valid speaking orders
def valid_speaking_orders : ℕ :=
  -- Number of orders where only one of A or B participates
  (Nat.choose 2 1) * (Nat.choose 6 3) * (Nat.factorial 4) +
  -- Number of orders where both A and B participate with one person between them
  (Nat.choose 2 2) * (Nat.choose 6 2) * (Nat.choose 2 1) * (Nat.factorial 2) * (Nat.factorial 2)

-- Theorem stating that the number of valid speaking orders is 1080
theorem valid_speaking_orders_eq_1080 : valid_speaking_orders = 1080 := by
  sorry

end NUMINAMATH_CALUDE_valid_speaking_orders_eq_1080_l1681_168158


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_greater_than_sin_l1681_168184

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_greater_than_sin :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_greater_than_sin_l1681_168184


namespace NUMINAMATH_CALUDE_impossible_all_multiples_of_10_l1681_168112

/-- Represents a grid operation (adding 1 to each cell in a subgrid) -/
structure GridOperation where
  startRow : Fin 8
  startCol : Fin 8
  size : Fin 2  -- 0 for 3x3, 1 for 4x4

/-- Represents the 8x8 grid of non-negative integers -/
def Grid := Fin 8 → Fin 8 → ℕ

/-- Applies a single grid operation to the given grid -/
def applyOperation (grid : Grid) (op : GridOperation) : Grid :=
  sorry

/-- Checks if all numbers in the grid are multiples of 10 -/
def allMultiplesOf10 (grid : Grid) : Prop :=
  ∀ i j, (grid i j) % 10 = 0

/-- Main theorem: It's impossible to make all numbers multiples of 10 -/
theorem impossible_all_multiples_of_10 (initialGrid : Grid) :
  ¬∃ (ops : List GridOperation), allMultiplesOf10 (ops.foldl applyOperation initialGrid) :=
sorry

end NUMINAMATH_CALUDE_impossible_all_multiples_of_10_l1681_168112


namespace NUMINAMATH_CALUDE_lucas_L10_units_digit_l1681_168102

/-- Lucas numbers sequence -/
def lucas : ℕ → ℕ
  | 0 => 3
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- Units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

theorem lucas_L10_units_digit :
  unitsDigit (lucas (lucas 10)) = 4 := by sorry

end NUMINAMATH_CALUDE_lucas_L10_units_digit_l1681_168102


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_PAQR_l1681_168181

-- Define the points
variable (P A Q R B : ℝ × ℝ)

-- Define the distances
def AP : ℝ := 10
def PB : ℝ := 20
def PR : ℝ := 25

-- Define the right triangles
def is_right_triangle (X Y Z : ℝ × ℝ) : Prop :=
  (X.1 - Y.1) * (X.1 - Z.1) + (X.2 - Y.2) * (X.2 - Z.2) = 0

-- State the theorem
theorem area_of_quadrilateral_PAQR :
  is_right_triangle P A Q →
  is_right_triangle P B R →
  (let area_PAQ := (1/2) * ‖A - P‖ * ‖Q - P‖;
   let area_PBR := (1/2) * ‖B - P‖ * ‖R - B‖;
   area_PAQ + area_PBR = 174) :=
by sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_PAQR_l1681_168181


namespace NUMINAMATH_CALUDE_food_duration_l1681_168165

/-- The number of days the food was initially meant to last -/
def initial_days : ℝ := 22

/-- The initial number of men -/
def initial_men : ℝ := 760

/-- The number of men that join after two days -/
def additional_men : ℝ := 134.11764705882354

/-- The number of days the food lasts after the additional men join -/
def remaining_days : ℝ := 17

/-- The total number of men after the additional men join -/
def total_men : ℝ := initial_men + additional_men

theorem food_duration :
  initial_men * (initial_days - 2) = total_men * remaining_days :=
sorry

end NUMINAMATH_CALUDE_food_duration_l1681_168165


namespace NUMINAMATH_CALUDE_stair_climbing_time_l1681_168134

theorem stair_climbing_time (a₁ : ℕ) (d : ℕ) (n : ℕ) (h1 : a₁ = 30) (h2 : d = 7) (h3 : n = 8) :
  n * (2 * a₁ + (n - 1) * d) / 2 = 436 := by
  sorry

end NUMINAMATH_CALUDE_stair_climbing_time_l1681_168134


namespace NUMINAMATH_CALUDE_joker_spade_probability_l1681_168144

/-- Custom deck of cards -/
structure CustomDeck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (jokers : ℕ)
  (cards_per_suit : ℕ)

/-- Properties of the custom deck -/
def custom_deck_properties (d : CustomDeck) : Prop :=
  d.total_cards = 60 ∧
  d.ranks = 15 ∧
  d.suits = 4 ∧
  d.jokers = 4 ∧
  d.cards_per_suit = 15

/-- Probability of drawing a Joker first and any spade second -/
def joker_spade_prob (d : CustomDeck) : ℚ :=
  224 / 885

/-- Theorem stating the probability of drawing a Joker first and any spade second -/
theorem joker_spade_probability (d : CustomDeck) 
  (h : custom_deck_properties d) : 
  joker_spade_prob d = 224 / 885 := by
  sorry

end NUMINAMATH_CALUDE_joker_spade_probability_l1681_168144


namespace NUMINAMATH_CALUDE_fourth_root_over_sixth_root_of_seven_l1681_168173

theorem fourth_root_over_sixth_root_of_seven (x : ℝ) :
  (7 : ℝ) ^ (1 / 4) / (7 : ℝ) ^ (1 / 6) = (7 : ℝ) ^ (1 / 12) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_over_sixth_root_of_seven_l1681_168173


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1681_168125

theorem complex_fraction_simplification :
  let numerator := (10^4 + 500) * (25^4 + 500) * (40^4 + 500) * (55^4 + 500) * (70^4 + 500)
  let denominator := (5^4 + 500) * (20^4 + 500) * (35^4 + 500) * (50^4 + 500) * (65^4 + 500)
  ∀ x : ℕ, x^4 + 500 = (x^2 - 10*x + 50) * (x^2 + 10*x + 50) →
  (numerator / denominator : ℚ) = 240 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1681_168125


namespace NUMINAMATH_CALUDE_johns_umbrella_cost_l1681_168114

/-- The total cost of John's umbrellas -/
def total_cost (house_umbrellas car_umbrellas cost_per_umbrella : ℕ) : ℕ :=
  (house_umbrellas + car_umbrellas) * cost_per_umbrella

/-- Proof that John's total cost for umbrellas is $24 -/
theorem johns_umbrella_cost :
  total_cost 2 1 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_johns_umbrella_cost_l1681_168114


namespace NUMINAMATH_CALUDE_time_to_reach_room_l1681_168176

theorem time_to_reach_room (total_time gate_time building_time : ℕ) 
  (h1 : total_time = 30)
  (h2 : gate_time = 15)
  (h3 : building_time = 6) :
  total_time - (gate_time + building_time) = 9 := by
  sorry

end NUMINAMATH_CALUDE_time_to_reach_room_l1681_168176


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1681_168118

theorem condition_necessary_not_sufficient : 
  (∃ x : ℝ, x^2 - 2*x = 0 ∧ x ≠ 0) ∧ 
  (∀ x : ℝ, x = 0 → x^2 - 2*x = 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1681_168118


namespace NUMINAMATH_CALUDE_pencils_bought_is_three_l1681_168191

/-- Calculates the number of pencils bought given the total paid, cost per pencil, cost of glue, and change received. -/
def number_of_pencils (total_paid change cost_per_pencil cost_of_glue : ℕ) : ℕ :=
  ((total_paid - change - cost_of_glue) / cost_per_pencil)

/-- Proves that the number of pencils bought is 3 under the given conditions. -/
theorem pencils_bought_is_three :
  number_of_pencils 1000 100 210 270 = 3 := by
  sorry

#eval number_of_pencils 1000 100 210 270

end NUMINAMATH_CALUDE_pencils_bought_is_three_l1681_168191


namespace NUMINAMATH_CALUDE_problem_statement_l1681_168130

theorem problem_statement (x y : ℝ) 
  (hx : x * (Real.exp x + Real.log x + x) = 1)
  (hy : y * (2 * Real.log y + Real.log (Real.log y)) = 1) :
  0 < x ∧ x < 1 ∧ y - x > 1 ∧ y - x < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1681_168130


namespace NUMINAMATH_CALUDE_equiangular_iff_rectangle_l1681_168142

-- Define a quadrilateral
class Quadrilateral :=
(angles : Fin 4 → ℝ)

-- Define an equiangular quadrilateral
def is_equiangular (q : Quadrilateral) : Prop :=
∀ i j : Fin 4, q.angles i = q.angles j

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
∀ i : Fin 4, q.angles i = 90

-- Theorem statement
theorem equiangular_iff_rectangle (q : Quadrilateral) : 
  is_equiangular q ↔ is_rectangle q :=
sorry

end NUMINAMATH_CALUDE_equiangular_iff_rectangle_l1681_168142


namespace NUMINAMATH_CALUDE_min_value_condition_l1681_168100

def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |a * x + 1|

theorem min_value_condition (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ 3/2) ∧ (∃ x : ℝ, f a x = 3/2) ↔ a = -1/2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_condition_l1681_168100


namespace NUMINAMATH_CALUDE_towel_shrinkage_l1681_168172

theorem towel_shrinkage (original_length original_breadth : ℝ) 
  (original_length_pos : 0 < original_length) 
  (original_breadth_pos : 0 < original_breadth) : 
  let new_length := 0.7 * original_length
  let new_area := 0.595 * (original_length * original_breadth)
  ∃ new_breadth : ℝ, 
    new_breadth = 0.85 * original_breadth ∧ 
    new_area = new_length * new_breadth :=
by sorry

end NUMINAMATH_CALUDE_towel_shrinkage_l1681_168172


namespace NUMINAMATH_CALUDE_sum_reciprocal_product_l1681_168121

open BigOperators

theorem sum_reciprocal_product : ∑ n in Finset.range 6, 1 / ((n + 3) * (n + 4)) = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_product_l1681_168121


namespace NUMINAMATH_CALUDE_square_equation_solution_l1681_168115

theorem square_equation_solution (x y : ℝ) 
  (h1 : x^2 = y + 3) 
  (h2 : x = 6) : 
  y = 33 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l1681_168115


namespace NUMINAMATH_CALUDE_stratified_sampling_major_c_l1681_168192

theorem stratified_sampling_major_c (total_students : ℕ) (sample_size : ℕ) 
  (major_a_students : ℕ) (major_b_students : ℕ) : 
  total_students = 1200 →
  sample_size = 120 →
  major_a_students = 380 →
  major_b_students = 420 →
  (total_students - major_a_students - major_b_students) * sample_size / total_students = 40 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_major_c_l1681_168192


namespace NUMINAMATH_CALUDE_placement_count_l1681_168143

/-- Represents a painting with width and height -/
structure Painting :=
  (width : Nat)
  (height : Nat)

/-- Represents a wall with width and height -/
structure Wall :=
  (width : Nat)
  (height : Nat)

/-- Represents the collection of paintings -/
def paintings : List Painting := [
  ⟨2, 1⟩,
  ⟨1, 1⟩, ⟨1, 1⟩,
  ⟨1, 2⟩, ⟨1, 2⟩,
  ⟨2, 2⟩, ⟨2, 2⟩,
  ⟨4, 3⟩, ⟨4, 3⟩,
  ⟨4, 4⟩, ⟨4, 4⟩
]

/-- The wall on which paintings are to be placed -/
def wall : Wall := ⟨12, 6⟩

/-- Function to calculate the number of ways to place paintings on the wall -/
def numberOfPlacements (w : Wall) (p : List Painting) : Nat :=
  sorry

/-- Theorem stating that the number of placements is 16896 -/
theorem placement_count : numberOfPlacements wall paintings = 16896 := by
  sorry

end NUMINAMATH_CALUDE_placement_count_l1681_168143


namespace NUMINAMATH_CALUDE_r_amount_l1681_168148

def total_amount : ℕ := 1210
def num_persons : ℕ := 3

def ratio_p_q : Rat := 5 / 4
def ratio_q_r : Rat := 9 / 10

theorem r_amount (p q r : ℕ) (h1 : p + q + r = total_amount) 
  (h2 : (p : ℚ) / q = ratio_p_q) (h3 : (q : ℚ) / r = ratio_q_r) : r = 400 := by
  sorry

end NUMINAMATH_CALUDE_r_amount_l1681_168148


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l1681_168131

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l1681_168131


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_11_l1681_168171

theorem least_three_digit_multiple_of_11 : ∃ n : ℕ, 
  n = 110 ∧ 
  n % 11 = 0 ∧ 
  100 ≤ n ∧ n ≤ 999 ∧ 
  ∀ m : ℕ, (m % 11 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_11_l1681_168171


namespace NUMINAMATH_CALUDE_star_star_equation_l1681_168119

theorem star_star_equation : 
  ∀ (a b : ℕ), a * b = 34 → (a = 2 ∧ b = 17) ∨ (a = 1 ∧ b = 34) ∨ (a = 17 ∧ b = 2) ∨ (a = 34 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_star_star_equation_l1681_168119


namespace NUMINAMATH_CALUDE_set_condition_implies_range_l1681_168147

theorem set_condition_implies_range (a : ℝ) : 
  let A := {x : ℝ | x > 5}
  let B := {x : ℝ | x > a}
  (∀ x, x ∈ A → x ∈ B) ∧ (∃ x, x ∈ B ∧ x ∉ A) → a < 5 := by
  sorry

end NUMINAMATH_CALUDE_set_condition_implies_range_l1681_168147


namespace NUMINAMATH_CALUDE_decimal_calculation_l1681_168106

theorem decimal_calculation : (0.25 * 0.8) - 0.12 = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_decimal_calculation_l1681_168106


namespace NUMINAMATH_CALUDE_circumcircle_of_triangle_ABP_l1681_168169

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the external point P
def point_P : ℝ × ℝ := (4, 2)

-- Define the property of A and B being points of tangency
def tangent_points (A B : ℝ × ℝ) : Prop :=
  given_circle A.1 A.2 ∧ given_circle B.1 B.2 ∧
  (∀ t : ℝ, t ≠ 0 → ¬(given_circle (A.1 + t * (point_P.1 - A.1)) (A.2 + t * (point_P.2 - A.2)))) ∧
  (∀ t : ℝ, t ≠ 0 → ¬(given_circle (B.1 + t * (point_P.1 - B.1)) (B.2 + t * (point_P.2 - B.2))))

-- Define the equation of the circumcircle
def circumcircle_equation (x y : ℝ) : Prop := (x - 4)^2 + (y - 2)^2 = 16

-- Theorem statement
theorem circumcircle_of_triangle_ABP :
  ∀ A B : ℝ × ℝ, tangent_points A B →
  ∀ x y : ℝ, (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 →
  (x - point_P.1)^2 + (y - point_P.2)^2 = (x - A.1)^2 + (y - A.2)^2 →
  circumcircle_equation x y :=
sorry

end NUMINAMATH_CALUDE_circumcircle_of_triangle_ABP_l1681_168169


namespace NUMINAMATH_CALUDE_max_value_theorem_l1681_168150

theorem max_value_theorem (a b c : ℝ) (h : a^2 + b^2 + c^2 ≤ 8) :
  4 * (a^3 + b^3 + c^3) - (a^4 + b^4 + c^4) ≤ 48 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1681_168150


namespace NUMINAMATH_CALUDE_sculpture_surface_area_l1681_168108

/-- Represents a layer in the sculpture -/
structure Layer where
  cubes : Nat
  exposedTopFaces : Nat
  exposedSideFaces : Nat

/-- Represents the sculpture -/
def Sculpture : List Layer := [
  { cubes := 1, exposedTopFaces := 1, exposedSideFaces := 4 },
  { cubes := 4, exposedTopFaces := 4, exposedSideFaces := 12 },
  { cubes := 9, exposedTopFaces := 9, exposedSideFaces := 6 },
  { cubes := 6, exposedTopFaces := 6, exposedSideFaces := 0 }
]

/-- Calculates the exposed surface area of a layer -/
def layerSurfaceArea (layer : Layer) : Nat :=
  layer.exposedTopFaces + layer.exposedSideFaces

/-- Calculates the total exposed surface area of the sculpture -/
def totalSurfaceArea (sculpture : List Layer) : Nat :=
  List.foldl (λ acc layer => acc + layerSurfaceArea layer) 0 sculpture

/-- Theorem: The total exposed surface area of the sculpture is 42 square meters -/
theorem sculpture_surface_area : totalSurfaceArea Sculpture = 42 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_surface_area_l1681_168108


namespace NUMINAMATH_CALUDE_freds_remaining_balloons_l1681_168133

/-- The number of green balloons Fred has after giving some away -/
def remaining_balloons (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Fred's remaining balloons equals the difference between initial and given away -/
theorem freds_remaining_balloons :
  remaining_balloons 709 221 = 488 := by
  sorry

end NUMINAMATH_CALUDE_freds_remaining_balloons_l1681_168133


namespace NUMINAMATH_CALUDE_largest_valid_p_l1681_168160

def is_valid_p (p : ℝ) : Prop :=
  p > 1 ∧ ∀ a b c : ℝ, 
    1/p ≤ a ∧ a ≤ p ∧
    1/p ≤ b ∧ b ≤ p ∧
    1/p ≤ c ∧ c ≤ p →
    9 * (a*b + b*c + c*a) * (a^2 + b^2 + c^2) ≥ (a + b + c)^4

theorem largest_valid_p :
  ∃ p : ℝ, p = Real.sqrt (4 + 3 * Real.sqrt 2) ∧
    is_valid_p p ∧
    ∀ q : ℝ, q > p → ¬is_valid_p q :=
sorry

end NUMINAMATH_CALUDE_largest_valid_p_l1681_168160


namespace NUMINAMATH_CALUDE_perpendicular_vectors_sum_magnitude_l1681_168103

theorem perpendicular_vectors_sum_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![x - 1, -x]
  (a 0 * b 0 + a 1 * b 1 = 0) → 
  Real.sqrt ((a 0 + b 0)^2 + (a 1 + b 1)^2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_sum_magnitude_l1681_168103


namespace NUMINAMATH_CALUDE_solution_set_a_eq_1_range_of_a_l1681_168101

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| - |x + a|

-- Theorem 1: Solution set when a = 1
theorem solution_set_a_eq_1 :
  {x : ℝ | f 1 x < -2} = {x : ℝ | x > 3/2} := by sorry

-- Theorem 2: Range of 'a'
theorem range_of_a :
  {a : ℝ | ∀ x y : ℝ, -2 + f a y ≤ f a x ∧ f a x ≤ 2 + f a y} =
  {a : ℝ | -3 ≤ a ∧ a ≤ -1} := by sorry

end NUMINAMATH_CALUDE_solution_set_a_eq_1_range_of_a_l1681_168101


namespace NUMINAMATH_CALUDE_income_calculation_l1681_168120

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 4 = expenditure * 10 →  -- ratio of income to expenditure is 10:4
  savings = income - expenditure →  -- savings definition
  savings = 11400 →  -- given savings amount
  income = 19000 := by  -- prove that income is 19000
sorry

end NUMINAMATH_CALUDE_income_calculation_l1681_168120


namespace NUMINAMATH_CALUDE_line_parameterization_l1681_168190

/-- Given a line y = 2x - 40 parameterized by (x,y) = (g(t), 10t - 12), 
    prove that g(t) = 5t + 14 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ x y t : ℝ, y = 2*x - 40 ∧ x = g t ∧ y = 10*t - 12) → 
  (∀ t : ℝ, g t = 5*t + 14) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l1681_168190


namespace NUMINAMATH_CALUDE_max_profit_at_price_l1681_168161

/-- Represents the daily sales and profit model of a store --/
structure StoreModel where
  cost_price : ℝ
  max_price_factor : ℝ
  sales_function : ℝ → ℝ
  profit_function : ℝ → ℝ

/-- The store model satisfies the given conditions --/
def satisfies_conditions (model : StoreModel) : Prop :=
  model.cost_price = 100 ∧
  model.max_price_factor = 1.4 ∧
  model.sales_function 130 = 140 ∧
  model.sales_function 140 = 120 ∧
  (∀ x, model.sales_function x = -2 * x + 400) ∧
  (∀ x, model.profit_function x = (x - model.cost_price) * model.sales_function x)

/-- The maximum profit occurs at the given price and value --/
theorem max_profit_at_price (model : StoreModel) 
    (h : satisfies_conditions model) :
    (∀ x, x ≤ model.max_price_factor * model.cost_price → 
      model.profit_function x ≤ model.profit_function 140) ∧
    model.profit_function 140 = 4800 := by
  sorry

#check max_profit_at_price

end NUMINAMATH_CALUDE_max_profit_at_price_l1681_168161


namespace NUMINAMATH_CALUDE_perfect_square_sum_l1681_168146

theorem perfect_square_sum (a b c d : ℤ) (h : a + b + c + d = 0) :
  2 * (a^4 + b^4 + c^4 + d^4) + 8 * a * b * c * d = (a^2 + b^2 + c^2 + d^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l1681_168146


namespace NUMINAMATH_CALUDE_union_covers_reals_l1681_168104

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- State the theorem
theorem union_covers_reals (a : ℝ) : 
  (A a ∪ B a = Set.univ) ↔ a ∈ Set.Iic 2 :=
sorry

end NUMINAMATH_CALUDE_union_covers_reals_l1681_168104


namespace NUMINAMATH_CALUDE_value_of_y_l1681_168186

theorem value_of_y : ∃ y : ℚ, (3 * y) / 7 = 14 ∧ y = 98 / 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l1681_168186


namespace NUMINAMATH_CALUDE_min_value_problem_l1681_168164

theorem min_value_problem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 1) (hab : a + b = 1) :
  (((a^2 + 1) / (a * b) - 2) * c + Real.sqrt 2 / (c - 1)) ≥ 4 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l1681_168164


namespace NUMINAMATH_CALUDE_units_digit_of_3542_to_876_l1681_168109

theorem units_digit_of_3542_to_876 : ∃ n : ℕ, 3542^876 ≡ 6 [ZMOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_3542_to_876_l1681_168109


namespace NUMINAMATH_CALUDE_fermat_prime_sum_l1681_168157

theorem fermat_prime_sum (n : ℕ) (p : ℕ) (hn : Odd n) (hn1 : n > 1) (hp : Prime p) :
  ¬ ∃ (x y z : ℤ), x^n + y^n = z^n ∧ x + y = p := by
  sorry

end NUMINAMATH_CALUDE_fermat_prime_sum_l1681_168157


namespace NUMINAMATH_CALUDE_smallest_nonneg_minus_opposite_largest_neg_l1681_168189

theorem smallest_nonneg_minus_opposite_largest_neg : ∃ a b : ℤ,
  (∀ x : ℤ, x ≥ 0 → a ≤ x) ∧
  (∀ y : ℤ, y < 0 → y ≤ -b) ∧
  (a - b = 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_nonneg_minus_opposite_largest_neg_l1681_168189


namespace NUMINAMATH_CALUDE_midpoint_theorem_l1681_168128

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

-- Define the potential midpoints
def midpoint1 : ℝ × ℝ := (1, 1)
def midpoint2 : ℝ × ℝ := (-1, 2)
def midpoint3 : ℝ × ℝ := (1, 3)
def midpoint4 : ℝ × ℝ := (-1, -4)

-- Define a function to check if a point is a valid midpoint
def is_valid_midpoint (m : ℝ × ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    hyperbola x1 y1 ∧ hyperbola x2 y2 ∧
    m.1 = (x1 + x2) / 2 ∧ m.2 = (y1 + y2) / 2

-- Theorem statement
theorem midpoint_theorem :
  ¬(is_valid_midpoint midpoint1) ∧
  ¬(is_valid_midpoint midpoint2) ∧
  ¬(is_valid_midpoint midpoint3) ∧
  is_valid_midpoint midpoint4 := by sorry

end NUMINAMATH_CALUDE_midpoint_theorem_l1681_168128


namespace NUMINAMATH_CALUDE_inequality_proof_l1681_168159

theorem inequality_proof (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  (x^4 / (y-1)^2) + (y^4 / (z-1)^2) + (z^4 / (x-1)^2) ≥ 48 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1681_168159


namespace NUMINAMATH_CALUDE_intersection_in_first_quadrant_implies_a_greater_than_two_l1681_168168

/-- Two lines intersect in the first quadrant implies a > 2 -/
theorem intersection_in_first_quadrant_implies_a_greater_than_two 
  (a : ℝ) 
  (l₁ : ℝ → ℝ → Prop) 
  (l₂ : ℝ → ℝ → Prop) 
  (h₁ : ∀ x y, l₁ x y ↔ a * x - y + 2 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ x + y - a = 0)
  (h_intersection : ∃ x y, l₁ x y ∧ l₂ x y ∧ x > 0 ∧ y > 0) : 
  a > 2 := by
sorry


end NUMINAMATH_CALUDE_intersection_in_first_quadrant_implies_a_greater_than_two_l1681_168168


namespace NUMINAMATH_CALUDE_strategy_exists_l1681_168116

/-- Represents a question of the form "Is n smaller than a?" --/
structure Question where
  a : ℕ
  deriving Repr

/-- Represents an answer to a question --/
inductive Answer
  | Yes
  | No
  deriving Repr

/-- Represents a strategy for determining n --/
structure Strategy where
  questions : List Question
  decisionFunction : List Answer → ℕ

/-- Theorem stating that a strategy exists to determine n within the given constraints --/
theorem strategy_exists :
  ∃ (s : Strategy),
    (s.questions.length ≤ 10) ∧
    (∀ n : ℕ,
      n > 0 ∧ n ≤ 144 →
      ∃ (answers : List Answer),
        answers.length = s.questions.length ∧
        s.decisionFunction answers = n) :=
  sorry


end NUMINAMATH_CALUDE_strategy_exists_l1681_168116


namespace NUMINAMATH_CALUDE_license_plate_count_l1681_168182

def alphabet : ℕ := 26
def vowels : ℕ := 7  -- A, E, I, O, U, W, Y
def consonants : ℕ := alphabet - vowels
def even_digits : ℕ := 5  -- 0, 2, 4, 6, 8

def license_plate_combinations : ℕ := consonants * vowels * consonants * even_digits

theorem license_plate_count : license_plate_combinations = 12565 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1681_168182


namespace NUMINAMATH_CALUDE_cross_section_perimeter_bounds_l1681_168195

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) where
  edge_length : a > 0

/-- A triangular cross-section through a vertex of a regular tetrahedron -/
structure TriangularCrossSection (a : ℝ) (t : RegularTetrahedron a) where
  perimeter : ℝ

/-- The perimeter of any triangular cross-section through a vertex of a regular tetrahedron
    with edge length a satisfies 2a < P ≤ 3a -/
theorem cross_section_perimeter_bounds (a : ℝ) (t : RegularTetrahedron a) 
  (s : TriangularCrossSection a t) : 2 * a < s.perimeter ∧ s.perimeter ≤ 3 * a := by
  sorry


end NUMINAMATH_CALUDE_cross_section_perimeter_bounds_l1681_168195
