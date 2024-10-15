import Mathlib

namespace NUMINAMATH_CALUDE_obtuse_triangle_side_range_l1401_140198

-- Define the triangle
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define an obtuse triangle
def ObtuseTriangle (a b c : ℝ) : Prop :=
  Triangle a b c ∧ (a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2)

-- Theorem statement
theorem obtuse_triangle_side_range :
  ∀ c : ℝ, ObtuseTriangle 4 3 c → c ∈ Set.Ioo 1 (Real.sqrt 7) ∪ Set.Ioo 5 7 :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_side_range_l1401_140198


namespace NUMINAMATH_CALUDE_time_difference_is_six_minutes_l1401_140152

/-- The time difference between walking and biking to work -/
def time_difference (blocks : ℕ) (walk_time_per_block : ℚ) (bike_time_per_block : ℚ) : ℚ :=
  blocks * (walk_time_per_block - bike_time_per_block)

/-- Proof that the time difference is 6 minutes -/
theorem time_difference_is_six_minutes :
  time_difference 9 1 (20 / 60) = 6 := by
  sorry

#eval time_difference 9 1 (20 / 60)

end NUMINAMATH_CALUDE_time_difference_is_six_minutes_l1401_140152


namespace NUMINAMATH_CALUDE_no_acute_triangle_2016gon_l1401_140191

/-- A regular polygon with n sides --/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A function that determines if three points form an acute triangle --/
def isAcuteTriangle (a b c : ℝ × ℝ) : Prop :=
  sorry

/-- The minimum number of vertices to paint black in a 2016-gon to avoid acute triangles --/
def minBlackVertices : ℕ := 1008

theorem no_acute_triangle_2016gon (p : RegularPolygon 2016) :
  ∃ (blackVertices : Finset (Fin 2016)),
    blackVertices.card = minBlackVertices ∧
    ∀ (a b c : Fin 2016),
      a ∉ blackVertices → b ∉ blackVertices → c ∉ blackVertices →
      ¬isAcuteTriangle (p.vertices a) (p.vertices b) (p.vertices c) :=
sorry

end NUMINAMATH_CALUDE_no_acute_triangle_2016gon_l1401_140191


namespace NUMINAMATH_CALUDE_translate_right_2_units_l1401_140177

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translation of a point to the right -/
def translateRight (p : Point2D) (units : ℝ) : Point2D :=
  { x := p.x + units, y := p.y }

theorem translate_right_2_units (A : Point2D) (h : A = ⟨-2, 3⟩) :
  translateRight A 2 = ⟨0, 3⟩ := by
  sorry

end NUMINAMATH_CALUDE_translate_right_2_units_l1401_140177


namespace NUMINAMATH_CALUDE_toms_friend_decks_l1401_140149

/-- The problem of calculating how many decks Tom's friend bought -/
theorem toms_friend_decks :
  ∀ (cost_per_deck : ℕ) (toms_decks : ℕ) (total_spent : ℕ),
    cost_per_deck = 8 →
    toms_decks = 3 →
    total_spent = 64 →
    ∃ (friends_decks : ℕ),
      friends_decks * cost_per_deck + toms_decks * cost_per_deck = total_spent ∧
      friends_decks = 5 :=
by sorry

end NUMINAMATH_CALUDE_toms_friend_decks_l1401_140149


namespace NUMINAMATH_CALUDE_triangle_side_inequality_l1401_140159

theorem triangle_side_inequality (a b c : ℝ) (h_area : 1 = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)))) (h_order : a ≤ b ∧ b ≤ c) : b ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_inequality_l1401_140159


namespace NUMINAMATH_CALUDE_solve_turtle_problem_l1401_140179

def turtle_problem (owen_initial : ℕ) (johanna_difference : ℕ) : Prop :=
  let johanna_initial : ℕ := owen_initial - johanna_difference
  let owen_after_month : ℕ := owen_initial * 2
  let johanna_after_loss : ℕ := johanna_initial / 2
  let owen_final : ℕ := owen_after_month + johanna_after_loss
  owen_final = 50

theorem solve_turtle_problem :
  turtle_problem 21 5 := by sorry

end NUMINAMATH_CALUDE_solve_turtle_problem_l1401_140179


namespace NUMINAMATH_CALUDE_triangle_area_problem_l1401_140148

/-- Line with slope m passing through point (x0, y0) -/
def Line (m : ℚ) (x0 y0 : ℚ) : ℚ → ℚ → Prop :=
  fun x y => y - y0 = m * (x - x0)

/-- Area of a triangle given coordinates of its vertices -/
def TriangleArea (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem triangle_area_problem :
  let line1 := Line (3/4) 1 3
  let line2 := Line (-1/3) 1 3
  let line3 := fun x y => x + y = 8
  let x1 := 1
  let y1 := 3
  let x2 := 21/2
  let y2 := 11/2
  let x3 := 23/7
  let y3 := 32/7
  (∀ x y, line1 x y ↔ y = (3/4) * x + 9/4) ∧
  (∀ x y, line2 x y ↔ y = (-1/3) * x + 10/3) ∧
  line1 x1 y1 ∧
  line2 x1 y1 ∧
  line1 x3 y3 ∧
  line3 x3 y3 ∧
  line2 x2 y2 ∧
  line3 x2 y2 →
  TriangleArea x1 y1 x2 y2 x3 y3 = 475/28 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l1401_140148


namespace NUMINAMATH_CALUDE_down_payment_amount_l1401_140107

/-- Given a purchase with a payment plan, prove the down payment amount. -/
theorem down_payment_amount
  (purchase_price : ℝ)
  (monthly_payment : ℝ)
  (num_payments : ℕ)
  (interest_rate : ℝ)
  (h1 : purchase_price = 110)
  (h2 : monthly_payment = 10)
  (h3 : num_payments = 12)
  (h4 : interest_rate = 9.090909090909092 / 100) :
  ∃ (down_payment : ℝ),
    down_payment + num_payments * monthly_payment =
      purchase_price + interest_rate * purchase_price ∧
    down_payment = 0 := by
  sorry

end NUMINAMATH_CALUDE_down_payment_amount_l1401_140107


namespace NUMINAMATH_CALUDE_cannot_make_55_cents_l1401_140134

def coin_values : List Nat := [5, 10, 25, 50]

theorem cannot_make_55_cents (coins : List Nat) : 
  (coins.length = 6 ∧ 
   ∀ c ∈ coins, c ∈ coin_values) → 
  coins.sum ≠ 55 := by
  sorry

end NUMINAMATH_CALUDE_cannot_make_55_cents_l1401_140134


namespace NUMINAMATH_CALUDE_g_2009_divisors_l1401_140172

/-- g(n) returns the smallest positive integer k such that 1/k has exactly n+1 digits after the decimal point -/
def g (n : ℕ+) : ℕ+ := sorry

/-- The number of positive integer divisors of g(2009) -/
def num_divisors_g_2009 : ℕ := sorry

theorem g_2009_divisors : num_divisors_g_2009 = 2011 := by sorry

end NUMINAMATH_CALUDE_g_2009_divisors_l1401_140172


namespace NUMINAMATH_CALUDE_function_properties_l1401_140136

-- Define the function f(x)
noncomputable def f (x a : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + 7 * Real.pi / 6) + a

-- State the theorem
theorem function_properties :
  ∃ (a : ℝ),
    (∀ x, f x a ≤ 2) ∧  -- Maximum value is 2
    (∃ x, f x a = 2) ∧  -- Maximum value is attained
    (a = 1) ∧  -- Value of a
    (∀ x, f x a = f (x + Real.pi) a) ∧  -- Smallest positive period is π
    (∀ k : ℤ, ∀ x ∈ Set.Icc (Real.pi / 6 + k * Real.pi) (5 * Real.pi / 12 + k * Real.pi),
      ∀ y ∈ Set.Icc (Real.pi / 6 + k * Real.pi) (5 * Real.pi / 12 + k * Real.pi),
      x ≤ y → f y a ≤ f x a)  -- Monotonically decreasing intervals
    :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1401_140136


namespace NUMINAMATH_CALUDE_savings_proof_l1401_140127

/-- Calculates a person's savings given their income and income-to-expenditure ratio --/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Proves that given the specified conditions, the savings are 4000 --/
theorem savings_proof (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) 
    (h1 : income = 20000)
    (h2 : income_ratio = 5)
    (h3 : expenditure_ratio = 4) :
  calculate_savings income income_ratio expenditure_ratio = 4000 := by
  sorry

#eval calculate_savings 20000 5 4

end NUMINAMATH_CALUDE_savings_proof_l1401_140127


namespace NUMINAMATH_CALUDE_inequality_system_unique_solution_l1401_140188

/-- A system of inequalities with parameter a and variable x -/
structure InequalitySystem (a : ℝ) :=
  (x : ℤ)
  (ineq1 : x^3 + 3*x^2 - x - 3 > 0)
  (ineq2 : x^2 - 2*a*x - 1 ≤ 0)
  (a_pos : a > 0)

/-- The theorem stating the range of a for which the system has exactly one integer solution -/
theorem inequality_system_unique_solution :
  ∀ a : ℝ, (∃! s : InequalitySystem a, True) ↔ 3/4 ≤ a ∧ a < 4/3 :=
sorry

end NUMINAMATH_CALUDE_inequality_system_unique_solution_l1401_140188


namespace NUMINAMATH_CALUDE_unique_c_for_quadratic_equation_l1401_140110

theorem unique_c_for_quadratic_equation :
  ∃! (c : ℝ), c ≠ 0 ∧
    (∃! (b : ℝ), b > 0 ∧
      (∃! (x : ℝ), x^2 + (2*b + 2/b)*x + c = 0)) ∧
    c = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_c_for_quadratic_equation_l1401_140110


namespace NUMINAMATH_CALUDE_regular_hexagon_cosine_product_l1401_140182

/-- A regular hexagon ABCDEF inscribed in a circle -/
structure RegularHexagon where
  /-- Side length of the hexagon -/
  side_length : ℝ
  /-- Length of diagonal AC -/
  diagonal_length : ℝ
  /-- Side length is positive -/
  side_pos : side_length > 0
  /-- Diagonal length is positive -/
  diagonal_pos : diagonal_length > 0
  /-- Relationship between side length and diagonal length in a regular hexagon -/
  hexagon_property : diagonal_length^2 = side_length^2 + side_length^2 - 2 * side_length * side_length * (-1/2)

/-- Theorem about the product of cosines in a regular hexagon -/
theorem regular_hexagon_cosine_product (h : RegularHexagon) (h_side : h.side_length = 5) (h_diag : h.diagonal_length = 2) :
  (1 - Real.cos (2 * Real.pi / 3)) * (1 - Real.cos (2 * Real.pi / 3)) = 2.25 := by
  sorry


end NUMINAMATH_CALUDE_regular_hexagon_cosine_product_l1401_140182


namespace NUMINAMATH_CALUDE_point_on_parallel_segment_l1401_140142

/-- Given a point M and a line segment MN parallel to the x-axis, 
    prove that N has specific coordinates -/
theorem point_on_parallel_segment 
  (M : ℝ × ℝ) 
  (length_MN : ℝ) 
  (h_M : M = (2, -4)) 
  (h_length : length_MN = 5) : 
  ∃ (N : ℝ × ℝ), (N = (-3, -4) ∨ N = (7, -4)) ∧ 
                 (N.2 = M.2) ∧ 
                 ((N.1 - M.1)^2 + (N.2 - M.2)^2 = length_MN^2) := by
  sorry

end NUMINAMATH_CALUDE_point_on_parallel_segment_l1401_140142


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l1401_140161

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set N
def N : Set ℝ := {x | 2*x < 2}

-- Theorem statement
theorem intersection_M_complement_N :
  ∀ x : ℝ, x ∈ (M ∩ (Set.univ \ N)) ↔ 1 ≤ x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l1401_140161


namespace NUMINAMATH_CALUDE_square_rectangle_contradiction_l1401_140141

-- Define the square and rectangle
structure Square where
  side : ℝ
  area : ℝ := side ^ 2

structure Rectangle where
  length : ℝ
  width : ℝ
  area : ℝ := length * width

-- Define the theorem
theorem square_rectangle_contradiction 
  (s : Square) 
  (r : Rectangle) 
  (h1 : r.area = 0.25 * s.area) 
  (h2 : s.area = 0.5 * r.area) : 
  False := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_contradiction_l1401_140141


namespace NUMINAMATH_CALUDE_original_calculation_result_l1401_140102

theorem original_calculation_result (x : ℤ) (h : x - 63 = 24) : x - 36 = 51 := by
  sorry

end NUMINAMATH_CALUDE_original_calculation_result_l1401_140102


namespace NUMINAMATH_CALUDE_distinct_colorings_l1401_140118

-- Define the symmetry group of the circle
inductive CircleSymmetry
| id : CircleSymmetry
| rot120 : CircleSymmetry
| rot240 : CircleSymmetry
| refl1 : CircleSymmetry
| refl2 : CircleSymmetry
| refl3 : CircleSymmetry

-- Define the coloring function
def Coloring := Fin 3 → Fin 3

-- Define the action of symmetries on colorings
def act (g : CircleSymmetry) (c : Coloring) : Coloring :=
  sorry

-- Define the fixed points under a symmetry
def fixedPoints (g : CircleSymmetry) : Nat :=
  sorry

-- The main theorem
theorem distinct_colorings : 
  (List.sum (List.map fixedPoints [CircleSymmetry.id, CircleSymmetry.rot120, 
    CircleSymmetry.rot240, CircleSymmetry.refl1, CircleSymmetry.refl2, 
    CircleSymmetry.refl3])) / 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_distinct_colorings_l1401_140118


namespace NUMINAMATH_CALUDE_order_of_abc_l1401_140144

theorem order_of_abc (a b c : ℝ) : 
  a = 2/21 → b = Real.log 1.1 → c = 21/220 → a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l1401_140144


namespace NUMINAMATH_CALUDE_complex_quadrant_l1401_140195

theorem complex_quadrant (z : ℂ) (h : (1 + Complex.I * Real.sqrt 3) * z = 2 - Complex.I * Real.sqrt 3) : 
  z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l1401_140195


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l1401_140187

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The nth term of an arithmetic sequence. -/
def nthTerm (a : ℕ → ℤ) (n : ℕ) : ℤ := a n

theorem fifth_term_of_arithmetic_sequence
  (a : ℕ → ℤ) (h : ArithmeticSequence a)
  (h10 : nthTerm a 10 = 15)
  (h12 : nthTerm a 12 = 21) :
  nthTerm a 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l1401_140187


namespace NUMINAMATH_CALUDE_sine_graph_shift_l1401_140169

theorem sine_graph_shift (x : ℝ) :
  2 * Real.sin (2 * (x + π/8) - π/4) = 2 * Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sine_graph_shift_l1401_140169


namespace NUMINAMATH_CALUDE_car_speed_equality_l1401_140122

/-- Prove that given the conditions of the car problem, the average speed of Car Y is equal to the average speed of Car X. -/
theorem car_speed_equality (speed_x : ℝ) (start_delay : ℝ) (distance_after_y_starts : ℝ)
  (h1 : speed_x = 35)
  (h2 : start_delay = 72 / 60)
  (h3 : distance_after_y_starts = 105) :
  ∃ (speed_y : ℝ), speed_y = speed_x := by
  sorry

end NUMINAMATH_CALUDE_car_speed_equality_l1401_140122


namespace NUMINAMATH_CALUDE_cosine_function_properties_l1401_140174

/-- Given a cosine function f(x) = a * cos(b * x + c) with positive constants a, b, and c,
    if f(x) reaches its first maximum at x = -π/4 and has a maximum value of 3,
    then a = 3, b = 1, and c = π/4. -/
theorem cosine_function_properties (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.cos (b * x + c)
  (∀ x, f x ≤ 3) ∧ (f (-π/4) = 3) ∧ (∀ x < -π/4, f x < 3) →
  a = 3 ∧ b = 1 ∧ c = π/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_function_properties_l1401_140174


namespace NUMINAMATH_CALUDE_product_evaluation_l1401_140112

theorem product_evaluation (n : ℕ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l1401_140112


namespace NUMINAMATH_CALUDE_machine_present_value_l1401_140113

/-- The present value of a machine given its future value and depreciation rate -/
theorem machine_present_value (future_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) 
  (h1 : future_value = 810)
  (h2 : depreciation_rate = 0.1)
  (h3 : years = 2) :
  future_value = 1000 * (1 - depreciation_rate) ^ years := by
  sorry

end NUMINAMATH_CALUDE_machine_present_value_l1401_140113


namespace NUMINAMATH_CALUDE_uncle_bob_parking_probability_l1401_140153

def parking_spaces : ℕ := 18
def parked_cars : ℕ := 15
def rv_spaces : ℕ := 3

theorem uncle_bob_parking_probability :
  let total_arrangements := Nat.choose parking_spaces parked_cars
  let blocked_arrangements := Nat.choose (parking_spaces - rv_spaces + 1) (parked_cars - rv_spaces + 1)
  (total_arrangements - blocked_arrangements : ℚ) / total_arrangements = 16 / 51 := by
  sorry

end NUMINAMATH_CALUDE_uncle_bob_parking_probability_l1401_140153


namespace NUMINAMATH_CALUDE_sequence_difference_sum_l1401_140128

theorem sequence_difference_sum : 
  (Finset.sum (Finset.range 100) (fun i => 3001 + i)) - 
  (Finset.sum (Finset.range 100) (fun i => 201 + i)) = 280000 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_sum_l1401_140128


namespace NUMINAMATH_CALUDE_power_sum_2001_l1401_140193

theorem power_sum_2001 (x y : ℝ) (h1 : x - y = 2) (h2 : x^2 + y^2 = 4) :
  x^2001 + y^2001 = 2^2001 ∨ x^2001 + y^2001 = -(2^2001) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_2001_l1401_140193


namespace NUMINAMATH_CALUDE_diamond_symmetry_lines_l1401_140117

-- Define the binary operation
def diamond (a b : ℝ) : ℝ := a^2 + a*b - b^2

-- Theorem statement
theorem diamond_symmetry_lines :
  ∀ x y : ℝ, diamond x y = diamond y x ↔ y = x ∨ y = -x :=
sorry

end NUMINAMATH_CALUDE_diamond_symmetry_lines_l1401_140117


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l1401_140135

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def N : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem
theorem intersection_complement_theorem :
  M ∩ (Set.univ \ N) = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l1401_140135


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l1401_140168

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- The theorem to be proved -/
theorem parallel_lines_a_value :
  ∀ a : ℝ,
  let l1 : Line := ⟨a - 1, 2, 3⟩
  let l2 : Line := ⟨1, a, 3⟩
  parallel l1 l2 → a = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l1401_140168


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l1401_140146

/-- The circle equation: x^2 + y^2 - 2x - 2y = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y = 0

/-- The line equation: x + y + 2 = 0 -/
def line_equation (x y : ℝ) : Prop :=
  x + y + 2 = 0

/-- The maximum distance from a point on the circle to the line is 3√2 -/
theorem max_distance_circle_to_line :
  ∃ (x y : ℝ), circle_equation x y ∧
  (∀ (a b : ℝ), circle_equation a b →
    Real.sqrt ((x - a)^2 + (y - b)^2) ≤ 3 * Real.sqrt 2) ∧
  (∃ (p q : ℝ), circle_equation p q ∧
    Real.sqrt ((x - p)^2 + (y - q)^2) = 3 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_l1401_140146


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1401_140185

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | x * a - 1 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (A ∩ B a = B a) → (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1401_140185


namespace NUMINAMATH_CALUDE_disjunction_true_when_second_true_l1401_140116

theorem disjunction_true_when_second_true (p q : Prop) (hp : ¬p) (hq : q) : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_disjunction_true_when_second_true_l1401_140116


namespace NUMINAMATH_CALUDE_cost_per_minute_advertising_l1401_140163

/-- The cost of one minute of advertising during a race, given the number of advertisements,
    duration of each advertisement, and total cost of transmission. -/
theorem cost_per_minute_advertising (num_ads : ℕ) (duration_per_ad : ℕ) (total_cost : ℕ) :
  num_ads = 5 →
  duration_per_ad = 3 →
  total_cost = 60000 →
  total_cost / (num_ads * duration_per_ad) = 4000 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_minute_advertising_l1401_140163


namespace NUMINAMATH_CALUDE_power_division_calculation_l1401_140156

theorem power_division_calculation : ((6^6 / 6^5)^3 * 8^3) / 4^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_power_division_calculation_l1401_140156


namespace NUMINAMATH_CALUDE_equation_solution_l1401_140143

theorem equation_solution (x : ℝ) (h : x ≠ -2) :
  (x^2 + 2*x + 3) / (x + 2) = x + 3 ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1401_140143


namespace NUMINAMATH_CALUDE_function_properties_l1401_140124

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -a^2 * x - 2 * a * x + 1

-- State the theorem
theorem function_properties (a : ℝ) (h_a : a > 1) :
  -- Part 1: Range of f(x)
  (∀ y : ℝ, (∃ x : ℝ, f a x = y) ↔ y < 1) ∧
  -- Part 2: Value of a when minimum on [-2, 1] is -7
  (∃ x : ℝ, x ∈ Set.Icc (-2) 1 ∧ f a x = -7 ∧ ∀ y ∈ Set.Icc (-2) 1, f a y ≥ f a x) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1401_140124


namespace NUMINAMATH_CALUDE_total_rehabilitation_centers_l1401_140160

/-- The number of rehabilitation centers visited by Lisa, Jude, Han, and Jane -/
def total_centers (lisa jude han jane : ℕ) : ℕ := lisa + jude + han + jane

/-- Theorem stating the total number of rehabilitation centers visited -/
theorem total_rehabilitation_centers :
  ∃ (lisa jude han jane : ℕ),
    lisa = 6 ∧
    jude = lisa / 2 ∧
    han = 2 * jude - 2 ∧
    jane = 2 * han + 6 ∧
    total_centers lisa jude han jane = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_rehabilitation_centers_l1401_140160


namespace NUMINAMATH_CALUDE_three_sequence_comparison_l1401_140181

theorem three_sequence_comparison 
  (a b c : ℕ → ℕ) : 
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q := by
sorry

end NUMINAMATH_CALUDE_three_sequence_comparison_l1401_140181


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1401_140137

/-- Given a parabola x^2 = 2py (p > 0), if a point on the parabola with ordinate 1 
    is at distance 3 from the focus, then the distance from the focus to the directrix is 4. -/
theorem parabola_focus_directrix_distance (p : ℝ) (h1 : p > 0) : 
  (∃ x : ℝ, x^2 = 2*p*1 ∧ 
   ((x - 0)^2 + (1 - p/2)^2)^(1/2) = 3) → 
  (0 - (-p/2)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1401_140137


namespace NUMINAMATH_CALUDE_post_circumference_l1401_140133

/-- Given a cylindrical post and a squirrel's spiral path, calculate the post's circumference -/
theorem post_circumference (post_height : ℝ) (spiral_rise_per_circuit : ℝ) (squirrel_travel : ℝ) 
  (h1 : post_height = 25)
  (h2 : spiral_rise_per_circuit = 5)
  (h3 : squirrel_travel = 15) :
  squirrel_travel / (post_height / spiral_rise_per_circuit) = 5 := by
  sorry

end NUMINAMATH_CALUDE_post_circumference_l1401_140133


namespace NUMINAMATH_CALUDE_orange_juice_glasses_l1401_140178

theorem orange_juice_glasses (total_juice : ℕ) (juice_per_glass : ℕ) (h1 : total_juice = 153) (h2 : juice_per_glass = 30) :
  ∃ (num_glasses : ℕ), num_glasses * juice_per_glass ≥ total_juice ∧
  ∀ (m : ℕ), m * juice_per_glass ≥ total_juice → m ≥ num_glasses :=
by sorry

end NUMINAMATH_CALUDE_orange_juice_glasses_l1401_140178


namespace NUMINAMATH_CALUDE_negation_of_existence_square_plus_one_positive_negation_l1401_140194

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem square_plus_one_positive_negation :
  (¬∃ x : ℝ, x^2 + 1 > 0) ↔ (∀ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_square_plus_one_positive_negation_l1401_140194


namespace NUMINAMATH_CALUDE_division_in_third_quadrant_l1401_140130

/-- Given two complex numbers z₁ and z₂, prove that z₁/z₂ is in the third quadrant -/
theorem division_in_third_quadrant (z₁ z₂ : ℂ) 
  (h₁ : z₁ = 1 - 2 * Complex.I) 
  (h₂ : z₂ = 2 + 3 * Complex.I) : 
  (z₁ / z₂).re < 0 ∧ (z₁ / z₂).im < 0 := by
  sorry

end NUMINAMATH_CALUDE_division_in_third_quadrant_l1401_140130


namespace NUMINAMATH_CALUDE_k_travel_time_l1401_140123

theorem k_travel_time (x : ℝ) 
  (h1 : x > 0) -- K's speed is positive
  (h2 : x - 0.5 > 0) -- M's speed is positive
  (h3 : 45 / (x - 0.5) - 45 / x = 3/4) -- K takes 45 minutes (3/4 hour) less than M
  : 45 / x = 9 := by
  sorry

end NUMINAMATH_CALUDE_k_travel_time_l1401_140123


namespace NUMINAMATH_CALUDE_lecture_scheduling_l1401_140192

theorem lecture_scheduling (n : ℕ) (h : n = 7) :
  let total_permutations := n.factorial
  let valid_orderings := total_permutations / 4
  valid_orderings = 1260 :=
by sorry

end NUMINAMATH_CALUDE_lecture_scheduling_l1401_140192


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l1401_140106

def a (n : ℕ) : ℕ := n.factorial + 2 * n

theorem max_gcd_consecutive_terms :
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (a n) (a (n + 1)) ≤ k ∧
  ∃ (m : ℕ), Nat.gcd (a m) (a (m + 1)) = k ∧
  k = 3 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l1401_140106


namespace NUMINAMATH_CALUDE_total_amc8_students_l1401_140180

/-- Represents a math class at Euclid Middle School -/
structure MathClass where
  teacher : String
  totalStudents : Nat
  olympiadStudents : Nat

/-- Calculates the number of students in a class taking only AMC 8 -/
def studentsOnlyAMC8 (c : MathClass) : Nat :=
  c.totalStudents - c.olympiadStudents

/-- Theorem: The total number of students only taking AMC 8 is 26 -/
theorem total_amc8_students (germain newton young : MathClass)
  (h_germain : germain = { teacher := "Mrs. Germain", totalStudents := 13, olympiadStudents := 3 })
  (h_newton : newton = { teacher := "Mr. Newton", totalStudents := 10, olympiadStudents := 2 })
  (h_young : young = { teacher := "Mrs. Young", totalStudents := 12, olympiadStudents := 4 }) :
  studentsOnlyAMC8 germain + studentsOnlyAMC8 newton + studentsOnlyAMC8 young = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_amc8_students_l1401_140180


namespace NUMINAMATH_CALUDE_theresa_chocolate_bars_double_kayla_l1401_140120

/-- Represents the number of items Kayla bought -/
structure KaylasItems where
  chocolateBars : ℕ
  sodaCans : ℕ
  total : ℕ
  total_eq : chocolateBars + sodaCans = total

/-- Represents the number of items Theresa bought -/
structure TheresasItems where
  chocolateBars : ℕ
  sodaCans : ℕ

/-- The given conditions of the problem -/
class ProblemConditions where
  kayla : KaylasItems
  theresa : TheresasItems
  kayla_total_15 : kayla.total = 15
  theresa_double_kayla : theresa.chocolateBars = 2 * kayla.chocolateBars ∧
                         theresa.sodaCans = 2 * kayla.sodaCans

theorem theresa_chocolate_bars_double_kayla
  [conditions : ProblemConditions] :
  conditions.theresa.chocolateBars = 2 * conditions.kayla.chocolateBars :=
by sorry

end NUMINAMATH_CALUDE_theresa_chocolate_bars_double_kayla_l1401_140120


namespace NUMINAMATH_CALUDE_tangent_alpha_equals_four_l1401_140184

theorem tangent_alpha_equals_four (α : Real) 
  (h : 3 * Real.tan α - Real.sin α + 4 * Real.cos α = 12) : 
  Real.tan α = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_alpha_equals_four_l1401_140184


namespace NUMINAMATH_CALUDE_particular_number_plus_eight_l1401_140129

theorem particular_number_plus_eight (n : ℝ) : n * 6 = 72 → n + 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_plus_eight_l1401_140129


namespace NUMINAMATH_CALUDE_bollards_contract_l1401_140157

theorem bollards_contract (total : ℕ) (installed : ℕ) (remaining : ℕ) : 
  installed = (3 * total) / 4 →
  remaining = 2000 →
  remaining = total / 4 →
  total = 8000 := by
  sorry

end NUMINAMATH_CALUDE_bollards_contract_l1401_140157


namespace NUMINAMATH_CALUDE_f_max_value_l1401_140103

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * (x - 2)^2 - 3

-- Theorem stating that the maximum value of f is -3
theorem f_max_value : ∃ (M : ℝ), M = -3 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l1401_140103


namespace NUMINAMATH_CALUDE_stating_sidorov_cash_sum_l1401_140115

/-- The disposable cash of the Sidorov family as of June 1, 2018 -/
def sidorov_cash : ℝ := 724506.3

/-- The first part of the Sidorov family's cash -/
def first_part : ℝ := 496941.3

/-- The second part of the Sidorov family's cash -/
def second_part : ℝ := 227565

/-- 
Theorem stating that the disposable cash of the Sidorov family 
as of June 1, 2018, is the sum of two given parts
-/
theorem sidorov_cash_sum : 
  sidorov_cash = first_part + second_part := by
  sorry

end NUMINAMATH_CALUDE_stating_sidorov_cash_sum_l1401_140115


namespace NUMINAMATH_CALUDE_specific_boy_girl_not_adjacent_girls_not_adjacent_l1401_140111

-- Define the number of boys and girls
def num_boys : ℕ := 5
def num_girls : ℕ := 3

-- Define the total number of people
def total_people : ℕ := num_boys + num_girls

-- Define the total number of arrangements without restrictions
def total_arrangements : ℕ := (total_people - 1).factorial

-- Theorem for the first part of the problem
theorem specific_boy_girl_not_adjacent :
  (total_arrangements - 2 * (total_people - 2).factorial) = 3600 := by sorry

-- Theorem for the second part of the problem
theorem girls_not_adjacent :
  (num_boys - 1).factorial * (num_boys.choose num_girls) * num_girls.factorial = 1440 := by sorry

end NUMINAMATH_CALUDE_specific_boy_girl_not_adjacent_girls_not_adjacent_l1401_140111


namespace NUMINAMATH_CALUDE_digit_sum_problem_l1401_140196

theorem digit_sum_problem (J K L : ℕ) : 
  J ≠ K ∧ J ≠ L ∧ K ≠ L →
  J < 10 ∧ K < 10 ∧ L < 10 →
  100 * J + 10 * K + L + 100 * J + 10 * L + L + 100 * J + 10 * K + L = 479 →
  J + K + L = 11 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l1401_140196


namespace NUMINAMATH_CALUDE_correct_observation_value_l1401_140100

theorem correct_observation_value 
  (n : ℕ) 
  (initial_mean : ℝ) 
  (wrong_value : ℝ) 
  (corrected_mean : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 36)
  (h3 : wrong_value = 23)
  (h4 : corrected_mean = 36.5) : 
  ∃ (correct_value : ℝ), correct_value = 48 ∧ 
    n * corrected_mean = n * initial_mean - wrong_value + correct_value :=
by sorry

end NUMINAMATH_CALUDE_correct_observation_value_l1401_140100


namespace NUMINAMATH_CALUDE_all_configurations_exist_l1401_140108

-- Define the geometric shapes
structure Rectangle where
  angles : Fin 4 → ℝ
  sides : Fin 4 → ℝ
  all_right_angles : ∀ i, angles i = 90
  opposite_sides_equal : sides 0 = sides 2 ∧ sides 1 = sides 3

structure Rhombus where
  angles : Fin 4 → ℝ
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ i j, sides i = sides j
  opposite_angles_equal : angles 0 = angles 2 ∧ angles 1 = angles 3

structure Parallelogram where
  angles : Fin 4 → ℝ
  sides : Fin 4 → ℝ
  opposite_sides_equal : sides 0 = sides 2 ∧ sides 1 = sides 3
  adjacent_angles_supplementary : ∀ i, angles i + angles ((i + 1) % 4) = 180

structure Quadrilateral where
  angles : Fin 4 → ℝ
  sides : Fin 4 → ℝ

structure Triangle where
  angles : Fin 3 → ℝ
  sum_of_angles : angles 0 + angles 1 + angles 2 = 180

-- Theorem stating that all configurations can exist
theorem all_configurations_exist :
  (∃ r : Rectangle, r.sides 0 ≠ r.sides 1) ∧
  (∃ rh : Rhombus, ∀ i, rh.angles i = 90) ∧
  (∃ p : Parallelogram, True) ∧
  (∃ q : Quadrilateral, (∀ i, q.angles i = 90) ∧ q.sides 0 ≠ q.sides 1) ∧
  (∃ t : Triangle, t.angles 0 = 100 ∧ t.angles 1 = 40 ∧ t.angles 2 = 40) :=
by sorry

end NUMINAMATH_CALUDE_all_configurations_exist_l1401_140108


namespace NUMINAMATH_CALUDE_quadratic_minimum_min_value_is_zero_l1401_140101

theorem quadratic_minimum (x : ℝ) : 
  (∀ y : ℝ, x^2 - 12*x + 36 ≤ y^2 - 12*y + 36) ↔ x = 6 :=
by sorry

theorem min_value_is_zero : 
  (6:ℝ)^2 - 12*(6:ℝ) + 36 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_min_value_is_zero_l1401_140101


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l1401_140104

/-- Acme T-Shirt Company's pricing structure -/
def acme_cost (x : ℕ) : ℕ := 50 + 9 * x

/-- Beta T-shirt Company's pricing structure -/
def beta_cost (x : ℕ) : ℕ := 14 * x

/-- The minimum number of shirts for which Acme is cheaper than Beta -/
def min_shirts_for_acme_cheaper : ℕ := 11

theorem acme_cheaper_at_min_shirts :
  acme_cost min_shirts_for_acme_cheaper < beta_cost min_shirts_for_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_for_acme_cheaper →
    acme_cost n ≥ beta_cost n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l1401_140104


namespace NUMINAMATH_CALUDE_unit_digit_of_15_100_pow_20_l1401_140139

-- Define a function to get the unit digit of a natural number
def unitDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem unit_digit_of_15_100_pow_20 :
  unitDigit ((15^100)^20) = 5 := by sorry

end NUMINAMATH_CALUDE_unit_digit_of_15_100_pow_20_l1401_140139


namespace NUMINAMATH_CALUDE_function_analysis_l1401_140166

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*a*x^2 - 5

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 6*a*x

-- Theorem statement
theorem function_analysis (a : ℝ) :
  (f' a 2 = 0) →  -- x = 2 is a critical point
  (a = 1) ∧       -- The value of a is 1
  (∀ x ∈ Set.Icc (-2 : ℝ) (4 : ℝ), f 1 x ≤ 15) ∧  -- Maximum value on [-2, 4] is 15
  (∀ x ∈ Set.Icc (-2 : ℝ) (4 : ℝ), f 1 x ≥ -21)   -- Minimum value on [-2, 4] is -21
:= by sorry

end NUMINAMATH_CALUDE_function_analysis_l1401_140166


namespace NUMINAMATH_CALUDE_original_light_wattage_l1401_140132

theorem original_light_wattage (new_wattage : ℝ) (increase_percentage : ℝ) :
  new_wattage = 67.2 ∧ 
  increase_percentage = 0.12 →
  ∃ original_wattage : ℝ,
    new_wattage = original_wattage * (1 + increase_percentage) ∧
    original_wattage = 60 := by
  sorry

end NUMINAMATH_CALUDE_original_light_wattage_l1401_140132


namespace NUMINAMATH_CALUDE_population_doubling_time_l1401_140190

/-- The annual birth rate per 1000 people -/
def birth_rate : ℝ := 39.4

/-- The annual death rate per 1000 people -/
def death_rate : ℝ := 19.4

/-- The number of years for the population to double -/
def doubling_time : ℝ := 35

/-- Theorem stating that given the birth and death rates, the population will double in 35 years -/
theorem population_doubling_time :
  let net_growth_rate := birth_rate - death_rate
  let percentage_growth_rate := net_growth_rate / 10  -- Converted to percentage
  70 / percentage_growth_rate = doubling_time := by sorry

end NUMINAMATH_CALUDE_population_doubling_time_l1401_140190


namespace NUMINAMATH_CALUDE_quadratic_roots_and_sum_l1401_140199

theorem quadratic_roots_and_sum : ∃ (m n p : ℕ), 
  (∀ x : ℝ, 2 * x * (5 * x - 11) = -5 ↔ x = (m + Real.sqrt n : ℝ) / p ∨ x = (m - Real.sqrt n : ℝ) / p) ∧ 
  Nat.gcd m (Nat.gcd n p) = 1 ∧
  m + n + p = 92 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_sum_l1401_140199


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l1401_140138

theorem sum_of_x_and_y_equals_two (x y : ℝ) 
  (h : 2 * x^2 + 2 * y^2 = 20 * x - 12 * y + 68) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l1401_140138


namespace NUMINAMATH_CALUDE_no_right_obtuse_triangle_l1401_140109

-- Define a triangle
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real

-- Define properties of triangles
def Triangle.isValid (t : Triangle) : Prop :=
  t.angle1 + t.angle2 + t.angle3 = 180

def Triangle.hasRightAngle (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

def Triangle.hasObtuseAngle (t : Triangle) : Prop :=
  t.angle1 > 90 ∨ t.angle2 > 90 ∨ t.angle3 > 90

-- Theorem: A right obtuse triangle cannot exist
theorem no_right_obtuse_triangle (t : Triangle) :
  t.isValid → ¬(t.hasRightAngle ∧ t.hasObtuseAngle) :=
by
  sorry


end NUMINAMATH_CALUDE_no_right_obtuse_triangle_l1401_140109


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l1401_140164

theorem gcd_lcm_sum : Nat.gcd 42 70 + Nat.lcm 20 15 = 74 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l1401_140164


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l1401_140154

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 2
  f 2 = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l1401_140154


namespace NUMINAMATH_CALUDE_impossible_all_coeffs_roots_l1401_140145

/-- Given n > 1 monic quadratic polynomials and 2n distinct coefficients,
    prove that not all coefficients can be roots of the polynomials. -/
theorem impossible_all_coeffs_roots (n : ℕ) (a b : Fin n → ℝ) 
    (h_n : n > 1)
    (h_distinct : ∀ (i j : Fin n), i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ b j)
    (h_poly : ∀ (i : Fin n), ∃ (x : ℝ), x^2 - a i * x + b i = 0) :
    ¬(∀ (i : Fin n), (∃ (j : Fin n), a i^2 - a j * a i + b j = 0) ∧
                     (∃ (k : Fin n), b i^2 - a k * b i + b k = 0)) :=
by sorry

end NUMINAMATH_CALUDE_impossible_all_coeffs_roots_l1401_140145


namespace NUMINAMATH_CALUDE_plane_perpendicular_through_perpendicular_line_line_not_perpendicular_in_perpendicular_planes_l1401_140105

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the necessary relations
variable (perpendicular : Plane → Plane → Prop)
variable (passes_through : Plane → Line → Prop)
variable (perpendicular_line : Line → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (line_of_intersection : Plane → Plane → Line)
variable (perpendicular_to_line : Line → Line → Prop)

-- Proposition 2
theorem plane_perpendicular_through_perpendicular_line 
  (p1 p2 : Plane) (l : Line) :
  perpendicular_line l p2 → passes_through p1 l → perpendicular p1 p2 :=
sorry

-- Proposition 4
theorem line_not_perpendicular_in_perpendicular_planes 
  (p1 p2 : Plane) (l : Line) :
  perpendicular p1 p2 →
  in_plane l p1 →
  ¬ perpendicular_to_line l (line_of_intersection p1 p2) →
  ¬ perpendicular_line l p2 :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicular_through_perpendicular_line_line_not_perpendicular_in_perpendicular_planes_l1401_140105


namespace NUMINAMATH_CALUDE_perpendicular_lines_solution_l1401_140175

theorem perpendicular_lines_solution (a : ℝ) : 
  (∀ x y : ℝ, ax + 2*y + 6 = 0 ∧ x + a*(a+1)*y + a^2 - 1 = 0 → 
   (a * 1 + 2 * (a*(a+1)) = 0)) → 
  (a = 0 ∨ a = -3/2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_solution_l1401_140175


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l1401_140151

theorem smallest_positive_solution :
  let f : ℝ → ℝ := λ x => Real.sqrt (3 * x) - 5 * x
  ∃ x : ℝ, x > 0 ∧ f x = 0 ∧ ∀ y : ℝ, y > 0 ∧ f y = 0 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l1401_140151


namespace NUMINAMATH_CALUDE_fraction_simplification_l1401_140189

theorem fraction_simplification : (1 : ℚ) / 462 + 23 / 42 = 127 / 231 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1401_140189


namespace NUMINAMATH_CALUDE_remainder_of_b_mod_13_l1401_140170

/-- Given that b ≡ (2^(-1) + 3^(-1) + 5^(-1))^(-1) (mod 13), prove that b ≡ 6 (mod 13) -/
theorem remainder_of_b_mod_13 :
  (((2 : ZMod 13)⁻¹ + (3 : ZMod 13)⁻¹ + (5 : ZMod 13)⁻¹)⁻¹ : ZMod 13) = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_b_mod_13_l1401_140170


namespace NUMINAMATH_CALUDE_triangle_inequality_l1401_140165

theorem triangle_inequality (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  2 * Real.sin A * Real.sin B < -Real.cos (2 * B + C) →
  a^2 + b^2 < c^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1401_140165


namespace NUMINAMATH_CALUDE_inequality_range_l1401_140186

theorem inequality_range (x y : ℝ) :
  y - x^2 < Real.sqrt (x^2) →
  ((x ≥ 0 → y < x + x^2) ∧ (x < 0 → y < -x + x^2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l1401_140186


namespace NUMINAMATH_CALUDE_expression_evaluation_l1401_140131

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 15) - 2 = -x^4 + 3*x^3 - 5*x^2 + 15*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1401_140131


namespace NUMINAMATH_CALUDE_problem_solution_l1401_140162

theorem problem_solution (x y : ℝ) 
  (h1 : 5^2 = x - 5)
  (h2 : (x + y)^(1/3) = 3) :
  x = 30 ∧ y = -3 ∧ Real.sqrt (x + 2*y) = 2 * Real.sqrt 6 ∨ Real.sqrt (x + 2*y) = -2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1401_140162


namespace NUMINAMATH_CALUDE_f_x₁_gt_f_x₂_l1401_140147

noncomputable section

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- f(x+1) is an even function
axiom f_even : ∀ x, f (x + 1) = f (-x + 1)

-- (x-1)f'(x) < 0
axiom f_decreasing : ∀ x, (x - 1) * f' x < 0

-- x₁ < x₂
variable (x₁ x₂ : ℝ)
axiom x₁_lt_x₂ : x₁ < x₂

-- x₁ + x₂ > 2
axiom sum_gt_two : x₁ + x₂ > 2

-- The theorem to prove
theorem f_x₁_gt_f_x₂ : f x₁ > f x₂ := by sorry

end NUMINAMATH_CALUDE_f_x₁_gt_f_x₂_l1401_140147


namespace NUMINAMATH_CALUDE_quadratic_root_range_l1401_140126

theorem quadratic_root_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 1 ∧ x₂ < 1 ∧ 
   7 * x₁^2 - (m + 13) * x₁ + m^2 - m - 2 = 0 ∧
   7 * x₂^2 - (m + 13) * x₂ + m^2 - m - 2 = 0) →
  -2 < m ∧ m < 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l1401_140126


namespace NUMINAMATH_CALUDE_twenty_paise_coins_l1401_140140

theorem twenty_paise_coins (total_coins : ℕ) (total_value : ℚ) : 
  total_coins = 324 →
  total_value = 71 →
  ∃ (coins_20 : ℕ) (coins_25 : ℕ),
    coins_20 + coins_25 = total_coins ∧
    (20 * coins_20 + 25 * coins_25 : ℚ) / 100 = total_value ∧
    coins_20 = 200 := by
  sorry

end NUMINAMATH_CALUDE_twenty_paise_coins_l1401_140140


namespace NUMINAMATH_CALUDE_prob_red_after_transfer_l1401_140125

/-- Represents the contents of a bag as a pair of natural numbers (white balls, red balls) -/
def BagContents := ℕ × ℕ

/-- The initial contents of bag A -/
def bagA : BagContents := (2, 1)

/-- The initial contents of bag B -/
def bagB : BagContents := (1, 2)

/-- Calculates the probability of drawing a red ball from a bag -/
def probRedBall (bag : BagContents) : ℚ :=
  (bag.2 : ℚ) / ((bag.1 + bag.2) : ℚ)

/-- Calculates the probability of transferring a red ball from bag A to bag B -/
def probTransferRed (bagA : BagContents) : ℚ :=
  (bagA.2 : ℚ) / ((bagA.1 + bagA.2) : ℚ)

/-- Theorem: The probability of drawing a red ball from bag B after transferring a random ball from bag A is 7/12 -/
theorem prob_red_after_transfer (bagA bagB : BagContents) :
  let probWhiteTransfer := 1 - probTransferRed bagA
  let probRedAfterWhite := probRedBall (bagB.1 + 1, bagB.2)
  let probRedAfterRed := probRedBall (bagB.1, bagB.2 + 1)
  probWhiteTransfer * probRedAfterWhite + probTransferRed bagA * probRedAfterRed = 7 / 12 :=
sorry

end NUMINAMATH_CALUDE_prob_red_after_transfer_l1401_140125


namespace NUMINAMATH_CALUDE_sin_120_cos_1290_l1401_140171

theorem sin_120_cos_1290 : Real.sin (-120 * π / 180) * Real.cos (1290 * π / 180) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_cos_1290_l1401_140171


namespace NUMINAMATH_CALUDE_candy_count_l1401_140150

theorem candy_count (initial_bags : ℕ) (initial_cookies : ℕ) (remaining_bags : ℕ) 
  (h1 : initial_bags = 14)
  (h2 : initial_cookies = 28)
  (h3 : remaining_bags = 2)
  (h4 : initial_cookies % initial_bags = 0) :
  initial_cookies - (remaining_bags * (initial_cookies / initial_bags)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_l1401_140150


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1401_140158

/-- Given two lines in the form of linear equations,
    returns true if they are perpendicular. -/
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

/-- The slope of the first line 3y + 2x - 6 = 0 -/
def m1 : ℚ := -2/3

/-- The slope of the second line 4y + ax - 5 = 0 in terms of a -/
def m2 (a : ℚ) : ℚ := -a/4

/-- Theorem stating that if the two given lines are perpendicular, then a = -6 -/
theorem perpendicular_lines_a_value :
  are_perpendicular m1 (m2 a) → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1401_140158


namespace NUMINAMATH_CALUDE_no_real_roots_condition_l1401_140119

theorem no_real_roots_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_condition_l1401_140119


namespace NUMINAMATH_CALUDE_inner_automorphism_is_automorphism_l1401_140155

variable {G : Type*} [Group G]

def inner_automorphism (x : G) (y : G) : G := x⁻¹ * y * x

theorem inner_automorphism_is_automorphism (x : G) :
  Function.Bijective (inner_automorphism x) ∧
  ∀ y z : G, inner_automorphism x (y * z) = inner_automorphism x y * inner_automorphism x z :=
sorry

end NUMINAMATH_CALUDE_inner_automorphism_is_automorphism_l1401_140155


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l1401_140167

/-- The volume of a rectangular prism -/
def volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a rectangular prism with dimensions 0.6m, 0.3m, and 0.2m is 0.036 m³ -/
theorem rectangular_prism_volume : volume 0.6 0.3 0.2 = 0.036 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l1401_140167


namespace NUMINAMATH_CALUDE_edith_book_ratio_l1401_140121

/-- Given that Edith has 80 novels on her schoolbook shelf and a total of 240 books (novels and writing books combined), 
    prove that the ratio of novels on the shelf to writing books in the suitcase is 1:2. -/
theorem edith_book_ratio :
  let novels_on_shelf : ℕ := 80
  let total_books : ℕ := 240
  let writing_books : ℕ := total_books - novels_on_shelf
  novels_on_shelf * 2 = writing_books := by
  sorry

end NUMINAMATH_CALUDE_edith_book_ratio_l1401_140121


namespace NUMINAMATH_CALUDE_yq_length_l1401_140173

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  side_pq : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 21
  side_qr : Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) = 29
  side_pr : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = 28

-- Define the inscribed triangle XYZ
structure InscribedTriangle (X Y Z : ℝ × ℝ) (P Q R : ℝ × ℝ) : Prop where
  x_on_qr : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (t * Q.1 + (1 - t) * R.1, t * Q.2 + (1 - t) * R.2)
  y_on_rp : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Y = (t * R.1 + (1 - t) * P.1, t * R.2 + (1 - t) * P.2)
  z_on_pq : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Z = (t * P.1 + (1 - t) * Q.1, t * P.2 + (1 - t) * Q.2)

-- Define the arc equality conditions
def ArcEquality (P Q R X Y Z : ℝ × ℝ) : Prop :=
  ∃ (O₄ O₅ O₆ : ℝ × ℝ),
    (Real.sqrt ((P.1 - Y.1)^2 + (P.2 - Y.2)^2) = Real.sqrt ((X.1 - Q.1)^2 + (X.2 - Q.2)^2)) ∧
    (Real.sqrt ((Q.1 - Z.1)^2 + (Q.2 - Z.2)^2) = Real.sqrt ((Y.1 - R.1)^2 + (Y.2 - R.2)^2)) ∧
    (Real.sqrt ((P.1 - Z.1)^2 + (P.2 - Z.2)^2) = Real.sqrt ((Y.1 - Q.1)^2 + (Y.2 - Q.2)^2))

theorem yq_length 
  (P Q R X Y Z : ℝ × ℝ)
  (h₁ : Triangle P Q R)
  (h₂ : InscribedTriangle X Y Z P Q R)
  (h₃ : ArcEquality P Q R X Y Z) :
  Real.sqrt ((Y.1 - Q.1)^2 + (Y.2 - Q.2)^2) = 15 := by sorry

end NUMINAMATH_CALUDE_yq_length_l1401_140173


namespace NUMINAMATH_CALUDE_exponent_increase_l1401_140114

theorem exponent_increase (x : ℝ) (y : ℝ) (h : 3^x = y) : 3^(x+1) = 3*y := by
  sorry

end NUMINAMATH_CALUDE_exponent_increase_l1401_140114


namespace NUMINAMATH_CALUDE_sum_equals_fraction_l1401_140176

def sumFunction (n : ℕ) : ℚ :=
  (n^4 - 1) / (n^4 + 1)

def sumRange : List ℕ := [2, 3, 4, 5]

theorem sum_equals_fraction :
  (sumRange.map sumFunction).sum = 21182880 / 349744361 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_fraction_l1401_140176


namespace NUMINAMATH_CALUDE_family_probability_theorem_l1401_140197

-- Define the family structure
structure Family :=
  (boys : ℕ)
  (girls : ℕ)

-- Define the list of families
def families : List Family := [
  ⟨0, 0⟩,  -- A
  ⟨1, 0⟩,  -- B
  ⟨0, 1⟩,  -- C
  ⟨1, 1⟩,  -- D
  ⟨1, 2⟩   -- E
]

-- Define the probability of selecting a girl from family E
def prob_girl_from_E : ℚ := 1/2

-- Define the probability distribution of X
def prob_dist_X : ℕ → ℚ
  | 0 => 1/10
  | 1 => 3/5
  | 2 => 3/10
  | _ => 0

-- Define the expected value of X
def expected_X : ℚ := 6/5

-- Theorem statement
theorem family_probability_theorem :
  (prob_girl_from_E = 1/2) ∧
  (prob_dist_X 0 = 1/10) ∧
  (prob_dist_X 1 = 3/5) ∧
  (prob_dist_X 2 = 3/10) ∧
  (expected_X = 6/5) := by
  sorry

end NUMINAMATH_CALUDE_family_probability_theorem_l1401_140197


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1401_140183

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a₂ + a₈ = 15 - a₅, then a₅ = 5 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_eq : a 2 + a 8 = 15 - a 5) : 
  a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1401_140183
