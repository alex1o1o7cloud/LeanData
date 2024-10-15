import Mathlib

namespace NUMINAMATH_CALUDE_car_sale_price_l3135_313527

/-- The final sale price of a car after multiple discounts and tax --/
theorem car_sale_price (original_price : ℝ) (discount1 discount2 discount3 tax_rate : ℝ) :
  original_price = 20000 ∧
  discount1 = 0.12 ∧
  discount2 = 0.10 ∧
  discount3 = 0.05 ∧
  tax_rate = 0.08 →
  (original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) * (1 + tax_rate)) = 16251.84 := by
  sorry

#eval (20000 : ℝ) * (1 - 0.12) * (1 - 0.10) * (1 - 0.05) * (1 + 0.08)

end NUMINAMATH_CALUDE_car_sale_price_l3135_313527


namespace NUMINAMATH_CALUDE_min_value_of_a2_plus_b2_l3135_313588

/-- Given a quadratic function f(x) = x^2 + ax + b - 3 that passes through (2, 0),
    the minimum value of a^2 + b^2 is 1/5 -/
theorem min_value_of_a2_plus_b2 (a b : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + b - 3 = 0) → x = 2) → 
  (∃ m : ℝ, m = (1 : ℝ) / 5 ∧ ∀ a' b' : ℝ, (∀ x : ℝ, (x^2 + a'*x + b' - 3 = 0) → x = 2) → a'^2 + b'^2 ≥ m) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_a2_plus_b2_l3135_313588


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3135_313594

theorem complex_magnitude_problem (z : ℂ) (h : 3 * z * Complex.I = -6 + 2 * Complex.I) :
  Complex.abs z = 2 * Real.sqrt 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3135_313594


namespace NUMINAMATH_CALUDE_percentage_apartments_with_two_residents_l3135_313574

theorem percentage_apartments_with_two_residents
  (total_apartments : ℕ)
  (percentage_with_at_least_one : ℚ)
  (apartments_with_one : ℕ)
  (h1 : total_apartments = 120)
  (h2 : percentage_with_at_least_one = 85 / 100)
  (h3 : apartments_with_one = 30) :
  (((percentage_with_at_least_one * total_apartments) - apartments_with_one) / total_apartments) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_percentage_apartments_with_two_residents_l3135_313574


namespace NUMINAMATH_CALUDE_table_length_proof_l3135_313511

theorem table_length_proof (table_width : ℝ) (sheet_width sheet_height : ℝ) 
  (h1 : table_width = 80)
  (h2 : sheet_width = 8)
  (h3 : sheet_height = 5)
  (h4 : ∃ n : ℕ, n * 1 = table_width - sheet_width ∧ n * 1 = table_width - sheet_height) :
  ∃ x : ℝ, x = 77 ∧ x = table_width - (sheet_width - sheet_height) := by
sorry

end NUMINAMATH_CALUDE_table_length_proof_l3135_313511


namespace NUMINAMATH_CALUDE_rectangle_division_l3135_313599

/-- Given a rectangle with length 3y and width y, divided into a smaller rectangle
    of length x and width y-x surrounded by four congruent right-angled triangles,
    this theorem proves the perimeter of one triangle and the area of the smaller rectangle. -/
theorem rectangle_division (x y : ℝ) : 
  let triangle_perimeter := 3 * y + Real.sqrt (2 * x^2 - 6 * y * x + 9 * y^2)
  let smaller_rectangle_area := x * y - x^2
  ∀ (triangle_side_a triangle_side_b : ℝ),
    triangle_side_a = x ∧ 
    triangle_side_b = 3 * y - x →
    triangle_perimeter = triangle_side_a + triangle_side_b + 
      Real.sqrt (triangle_side_a^2 + triangle_side_b^2) ∧
    smaller_rectangle_area = x * (y - x) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_l3135_313599


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l3135_313500

/-- Given a nonreal cube root of unity ω, prove that (1 - ω + ω^2)^6 + (1 + ω - ω^2)^6 = 128 -/
theorem cube_root_unity_sum (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω ≠ 1) :
  (1 - ω + ω^2)^6 + (1 + ω - ω^2)^6 = 128 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l3135_313500


namespace NUMINAMATH_CALUDE_gcf_of_75_and_105_l3135_313505

theorem gcf_of_75_and_105 : Nat.gcd 75 105 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_105_l3135_313505


namespace NUMINAMATH_CALUDE_vector_perpendicularity_l3135_313534

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : ℝ :=
  v.x * w.x + v.y * w.y

/-- Two vectors are perpendicular if their dot product is zero -/
def is_perpendicular (v w : Vector2D) : Prop :=
  dot_product v w = 0

/-- Unit vector in positive x direction -/
def i : Vector2D :=
  ⟨1, 0⟩

/-- Unit vector in positive y direction -/
def j : Vector2D :=
  ⟨0, 1⟩

/-- Vector addition -/
def add_vectors (v w : Vector2D) : Vector2D :=
  ⟨v.x + w.x, v.y + w.y⟩

/-- Vector subtraction -/
def subtract_vectors (v w : Vector2D) : Vector2D :=
  ⟨v.x - w.x, v.y - w.y⟩

/-- Scalar multiplication of a vector -/
def scalar_mult (k : ℝ) (v : Vector2D) : Vector2D :=
  ⟨k * v.x, k * v.y⟩

theorem vector_perpendicularity :
  let a := scalar_mult 2 i
  let b := add_vectors i j
  is_perpendicular (subtract_vectors a b) b := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicularity_l3135_313534


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3135_313501

def M : Set Int := {-1, 0, 1}
def N : Set Int := {0, 1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3135_313501


namespace NUMINAMATH_CALUDE_min_max_values_of_f_l3135_313582

def f (x : ℝ) : ℝ := 4 * x^2 + 8 * x + 16

theorem min_max_values_of_f :
  let a := -3
  let b := 2
  ∃ (x_min x_max : ℝ), a ≤ x_min ∧ x_min ≤ b ∧ a ≤ x_max ∧ x_max ≤ b ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x_min ≤ f x) ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f x_max) ∧
    f x_min = 12 ∧ f x_max = 48 :=
by sorry

end NUMINAMATH_CALUDE_min_max_values_of_f_l3135_313582


namespace NUMINAMATH_CALUDE_point_movement_l3135_313577

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Calculates the new position of a point after moving with a given velocity for a certain time -/
def move (p : Point2D) (v : Vector2D) (t : ℝ) : Point2D :=
  { x := p.x + v.x * t,
    y := p.y + v.y * t }

theorem point_movement :
  let initialPoint : Point2D := { x := -10, y := 10 }
  let velocity : Vector2D := { x := 4, y := -3 }
  let time : ℝ := 5
  let finalPoint : Point2D := move initialPoint velocity time
  finalPoint = { x := 10, y := -5 } := by sorry

end NUMINAMATH_CALUDE_point_movement_l3135_313577


namespace NUMINAMATH_CALUDE_unique_perpendicular_tangent_perpendicular_tangent_equation_slope_angle_range_l3135_313571

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

-- Statement for the unique perpendicular tangent line
theorem unique_perpendicular_tangent :
  ∃! a : ℝ, ∃! x : ℝ, f' a x = -1 ∧ a = 3 := by sorry

-- Statement for the equation of the perpendicular tangent line
theorem perpendicular_tangent_equation (a : ℝ) (h : a = 3) :
  ∃ x y : ℝ, 3*x + 3*y - 8 = 0 ∧ y = f a x ∧ f' a x = -1 := by sorry

-- Statement for the range of the slope angle
theorem slope_angle_range (a : ℝ) (h : a = 3) :
  ∀ x : ℝ, -π/4 ≤ Real.arctan (f' a x) ∧ Real.arctan (f' a x) < π/2 := by sorry

end

end NUMINAMATH_CALUDE_unique_perpendicular_tangent_perpendicular_tangent_equation_slope_angle_range_l3135_313571


namespace NUMINAMATH_CALUDE_fraction_equality_l3135_313519

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (2 * x - 3 * y) / (x + 2 * y) = 3) : 
  (x - 2 * y) / (2 * x + 3 * y) = 11 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3135_313519


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3135_313570

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the properties of the sequence
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 3 * a 7 = 2 * (a 4)^2 →
  a 3 = 1 →
  a 2 = Real.sqrt 2 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3135_313570


namespace NUMINAMATH_CALUDE_bridge_length_l3135_313523

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 225 :=
by
  sorry


end NUMINAMATH_CALUDE_bridge_length_l3135_313523


namespace NUMINAMATH_CALUDE_second_account_interest_rate_l3135_313584

/-- Proves that the interest rate of the second account is 5% given the problem conditions -/
theorem second_account_interest_rate 
  (total_investment : ℝ) 
  (first_account_investment : ℝ) 
  (first_account_rate : ℝ) 
  (total_interest : ℝ) 
  (h1 : total_investment = 8000)
  (h2 : first_account_investment = 3000)
  (h3 : first_account_rate = 0.08)
  (h4 : total_interest = 490) :
  let second_account_investment := total_investment - first_account_investment
  let first_account_interest := first_account_investment * first_account_rate
  let second_account_interest := total_interest - first_account_interest
  let second_account_rate := second_account_interest / second_account_investment
  second_account_rate = 0.05 := by
sorry


end NUMINAMATH_CALUDE_second_account_interest_rate_l3135_313584


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l3135_313598

theorem max_value_of_fraction (x : ℝ) (h : x ≠ 0) :
  x^2 / (x^6 - 2*x^5 - 2*x^4 + 4*x^3 + 4*x^2 + 16) ≤ 1/8 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l3135_313598


namespace NUMINAMATH_CALUDE_symmetric_point_and_line_in_quadrant_l3135_313596

-- Define the symmetric point function
def symmetric_point (x y : ℝ) (a b c : ℝ) : ℝ × ℝ := sorry

-- Define the line equation
def line_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x + y + m - 1 = 0

theorem symmetric_point_and_line_in_quadrant :
  -- Statement C
  symmetric_point 1 0 1 (-1) 1 = (-1, 2) ∧
  -- Statement D
  ∀ m : ℝ, line_equation m (-1) 1 := by sorry

end NUMINAMATH_CALUDE_symmetric_point_and_line_in_quadrant_l3135_313596


namespace NUMINAMATH_CALUDE_solution_set_l3135_313563

theorem solution_set (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let S := {(x, y, z) : ℝ × ℝ × ℝ | 
    a * x + b * y = (x - y)^2 ∧
    b * y + c * z = (y - z)^2 ∧
    c * z + a * x = (z - x)^2}
  S = {(0, 0, 0), (a, 0, 0), (0, b, 0), (0, 0, c)} := by
sorry

end NUMINAMATH_CALUDE_solution_set_l3135_313563


namespace NUMINAMATH_CALUDE_sum_of_cubes_zero_l3135_313549

theorem sum_of_cubes_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum : a / (2 * (b - c)) + b / (2 * (c - a)) + c / (2 * (a - b)) = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_zero_l3135_313549


namespace NUMINAMATH_CALUDE_f_increasing_range_l3135_313503

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * (x - 1)^2 + 1 else (a + 3) * x + 4 * a

theorem f_increasing_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔
  a ∈ Set.Icc (-2/5 : ℝ) 0 ∧ a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_range_l3135_313503


namespace NUMINAMATH_CALUDE_age_difference_l3135_313539

theorem age_difference (alice_age carol_age betty_age : ℕ) : 
  carol_age = 5 * alice_age →
  carol_age = 2 * betty_age →
  betty_age = 6 →
  carol_age - alice_age = 10 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l3135_313539


namespace NUMINAMATH_CALUDE_divide_fractions_l3135_313529

theorem divide_fractions : (3 : ℚ) / 4 / ((7 : ℚ) / 8) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_divide_fractions_l3135_313529


namespace NUMINAMATH_CALUDE_complex_computations_l3135_313561

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_computations :
  (Complex.abs (3 - i) = Real.sqrt 10) ∧
  ((10 * i) / (3 - i) = -1 + 3 * i) :=
by sorry

end NUMINAMATH_CALUDE_complex_computations_l3135_313561


namespace NUMINAMATH_CALUDE_equation_solutions_l3135_313595

-- Define the equation
def equation (x : ℝ) : Prop :=
  (x - 2)^6 + (x - 6)^6 = 432

-- Define the approximate solutions
def solution1 : ℝ := 4.795
def solution2 : ℝ := 3.205

-- State the theorem
theorem equation_solutions :
  ∃ (ε : ℝ), ε > 0 ∧ 
  (∀ (x : ℝ), equation x → (|x - solution1| < ε ∨ |x - solution2| < ε)) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3135_313595


namespace NUMINAMATH_CALUDE_average_weight_problem_l3135_313528

theorem average_weight_problem (a b c : ℝ) : 
  (a + b) / 2 = 40 →
  (b + c) / 2 = 41 →
  b = 27 →
  (a + b + c) / 3 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l3135_313528


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3135_313564

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ = 0 ∧ x₂^2 + 4*x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3135_313564


namespace NUMINAMATH_CALUDE_square_side_length_l3135_313524

-- Define the perimeter of the square
def perimeter : ℝ := 34.8

-- Theorem: The length of one side of a square with perimeter 34.8 cm is 8.7 cm
theorem square_side_length : 
  perimeter / 4 = 8.7 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3135_313524


namespace NUMINAMATH_CALUDE_restaurant_bill_entree_cost_l3135_313587

/-- Given the conditions of a restaurant bill, prove the cost of each entree -/
theorem restaurant_bill_entree_cost 
  (appetizer_cost : ℝ)
  (tip_percentage : ℝ)
  (total_spent : ℝ)
  (num_entrees : ℕ)
  (h_appetizer : appetizer_cost = 10)
  (h_tip : tip_percentage = 0.2)
  (h_total : total_spent = 108)
  (h_num_entrees : num_entrees = 4) :
  ∃ (entree_cost : ℝ), 
    entree_cost * num_entrees + appetizer_cost + 
    (entree_cost * num_entrees + appetizer_cost) * tip_percentage = total_spent ∧
    entree_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_entree_cost_l3135_313587


namespace NUMINAMATH_CALUDE_min_value_of_w_l3135_313585

def w (x y : ℝ) : ℝ := 2 * x^2 + 3 * y^2 + 8 * x - 6 * y + 30

theorem min_value_of_w :
  ∀ x y : ℝ, w x y ≥ 19 ∧ ∃ x₀ y₀ : ℝ, w x₀ y₀ = 19 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_w_l3135_313585


namespace NUMINAMATH_CALUDE_intersecting_lines_coefficient_sum_l3135_313554

/-- Two lines intersecting at a point implies a specific sum of their coefficients -/
theorem intersecting_lines_coefficient_sum 
  (m b : ℝ) 
  (h1 : 8 = m * 5 + 3) 
  (h2 : 8 = 4 * 5 + b) : 
  b + m = -11 := by sorry

end NUMINAMATH_CALUDE_intersecting_lines_coefficient_sum_l3135_313554


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l3135_313572

def geometric_sequence (a : ℕ → ℤ) (q : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem seventh_term_of_geometric_sequence 
  (a : ℕ → ℤ) (q : ℤ) 
  (h_seq : geometric_sequence a q)
  (h_a4 : a 4 = 27)
  (h_q : q = -3) :
  a 7 = -729 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l3135_313572


namespace NUMINAMATH_CALUDE_neznaika_claims_l3135_313508

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def divisibility_claims (n : ℕ) : List Bool :=
  [n % 3 = 0, n % 4 = 0, n % 5 = 0, n % 9 = 0, n % 10 = 0, n % 15 = 0, n % 18 = 0, n % 30 = 0]

theorem neznaika_claims (n : ℕ) : 
  is_two_digit n → (divisibility_claims n).count false = 4 → n = 36 ∨ n = 45 ∨ n = 72 := by
  sorry

end NUMINAMATH_CALUDE_neznaika_claims_l3135_313508


namespace NUMINAMATH_CALUDE_chimney_bricks_l3135_313569

/-- The number of bricks in the chimney -/
def h : ℕ := 360

/-- Brenda's time to build the chimney alone (in hours) -/
def brenda_time : ℕ := 8

/-- Brandon's time to build the chimney alone (in hours) -/
def brandon_time : ℕ := 12

/-- Efficiency decrease when working together (in bricks per hour) -/
def efficiency_decrease : ℕ := 15

/-- Time taken to build the chimney together (in hours) -/
def time_together : ℕ := 6

theorem chimney_bricks : 
  time_together * ((h / brenda_time + h / brandon_time) - efficiency_decrease) = h := by
  sorry

#check chimney_bricks

end NUMINAMATH_CALUDE_chimney_bricks_l3135_313569


namespace NUMINAMATH_CALUDE_insufficient_apples_l3135_313548

def apples_picked : ℕ := 150
def num_children : ℕ := 4
def apples_per_child_per_day : ℕ := 12
def days_in_week : ℕ := 7
def apples_per_pie : ℕ := 12
def num_pies : ℕ := 2
def apples_per_salad : ℕ := 15
def salads_per_week : ℕ := 2
def apples_taken_by_sister : ℕ := 5

theorem insufficient_apples :
  apples_picked < 
    (num_children * apples_per_child_per_day * days_in_week) +
    (num_pies * apples_per_pie) +
    (apples_per_salad * salads_per_week) +
    apples_taken_by_sister := by
  sorry

end NUMINAMATH_CALUDE_insufficient_apples_l3135_313548


namespace NUMINAMATH_CALUDE_jumping_contest_l3135_313525

theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) : 
  grasshopper_jump = 14 →
  frog_jump = grasshopper_jump + 37 →
  mouse_jump = frog_jump - 16 →
  mouse_jump - grasshopper_jump = 21 :=
by sorry

end NUMINAMATH_CALUDE_jumping_contest_l3135_313525


namespace NUMINAMATH_CALUDE_large_box_height_is_four_l3135_313502

-- Define the dimensions of the larger box
def large_box_length : ℝ := 6
def large_box_width : ℝ := 5

-- Define the dimensions of the smaller box in meters
def small_box_length : ℝ := 0.6
def small_box_width : ℝ := 0.5
def small_box_height : ℝ := 0.4

-- Define the maximum number of small boxes
def max_small_boxes : ℕ := 1000

-- Theorem statement
theorem large_box_height_is_four :
  ∃ (h : ℝ), 
    h = 4 ∧ 
    large_box_length * large_box_width * h = 
      (max_small_boxes : ℝ) * small_box_length * small_box_width * small_box_height :=
by sorry

end NUMINAMATH_CALUDE_large_box_height_is_four_l3135_313502


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3135_313518

/-- A hyperbola with equation mx^2 - y^2 = 1 and asymptotes y = ±3x has m = 9 -/
theorem hyperbola_asymptotes (m : ℝ) : 
  (∀ x y : ℝ, m * x^2 - y^2 = 1) → 
  (∀ x : ℝ, (∃ y : ℝ, y = 3 * x ∨ y = -3 * x) → m * x^2 - y^2 = 0) → 
  m = 9 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3135_313518


namespace NUMINAMATH_CALUDE_basketball_probability_l3135_313536

theorem basketball_probability (adam beth jack jill sandy : ℚ)
  (h_adam : adam = 1/5)
  (h_beth : beth = 2/9)
  (h_jack : jack = 1/6)
  (h_jill : jill = 1/7)
  (h_sandy : sandy = 1/8) :
  (1 - adam) * beth * (1 - jack) * jill * sandy = 1/378 :=
by sorry

end NUMINAMATH_CALUDE_basketball_probability_l3135_313536


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3135_313560

/-- The eccentricity of a hyperbola with equation y² - x²/4 = 1 is √5 -/
theorem hyperbola_eccentricity : 
  let hyperbola := fun (x y : ℝ) => y^2 - x^2/4 = 1
  ∃ e : ℝ, e = Real.sqrt 5 ∧ 
    ∀ x y : ℝ, hyperbola x y → 
      e = Real.sqrt ((1 + 4) / 1) := by
        sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3135_313560


namespace NUMINAMATH_CALUDE_crayons_left_correct_l3135_313578

/-- Given an initial number of crayons and a number of crayons lost or given away,
    calculate the number of crayons left. -/
def crayons_left (initial : ℕ) (lost_or_given : ℕ) : ℕ :=
  initial - lost_or_given

/-- Theorem: The number of crayons left is equal to the initial number minus
    the number lost or given away. -/
theorem crayons_left_correct (initial : ℕ) (lost_or_given : ℕ) 
  (h : lost_or_given ≤ initial) : 
  crayons_left initial lost_or_given = initial - lost_or_given :=
by sorry

end NUMINAMATH_CALUDE_crayons_left_correct_l3135_313578


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3135_313565

theorem simplify_trig_expression :
  Real.sqrt (1 - 2 * Real.sin 4 * Real.cos 4) = Real.cos 4 - Real.sin 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3135_313565


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3135_313538

theorem logarithm_expression_equality : 
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 - 5 ^ (Real.log 3 / Real.log 5) = -1 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3135_313538


namespace NUMINAMATH_CALUDE_largest_multiple_of_5_and_6_under_1000_l3135_313526

theorem largest_multiple_of_5_and_6_under_1000 :
  ∃ n : ℕ, n = 990 ∧
  (∀ m : ℕ, m < 1000 ∧ 5 ∣ m ∧ 6 ∣ m → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_5_and_6_under_1000_l3135_313526


namespace NUMINAMATH_CALUDE_marble_exchange_ratio_l3135_313540

theorem marble_exchange_ratio : 
  ∀ (ben_initial john_initial ben_final john_final marbles_given : ℕ),
    ben_initial = 18 →
    john_initial = 17 →
    ben_final = ben_initial - marbles_given →
    john_final = john_initial + marbles_given →
    john_final = ben_final + 17 →
    marbles_given * 2 = ben_initial :=
by
  sorry

end NUMINAMATH_CALUDE_marble_exchange_ratio_l3135_313540


namespace NUMINAMATH_CALUDE_painting_price_change_l3135_313556

/-- Calculates the final price of a painting after a series of value changes and currency depreciation. -/
def final_price_percentage (initial_increase : ℝ) (first_decrease : ℝ) (second_decrease : ℝ) 
  (discount : ℝ) (currency_depreciation : ℝ) : ℝ :=
  let year1 := 1 + initial_increase
  let year2 := year1 * (1 - first_decrease)
  let year3 := year2 * (1 - second_decrease)
  let discounted := year3 * (1 - discount)
  discounted * (1 + currency_depreciation)

/-- Theorem stating that the final price of the painting is 113.373% of the original price -/
theorem painting_price_change : 
  ∀ (ε : ℝ), ε > 0 → 
  |final_price_percentage 0.30 0.15 0.10 0.05 0.20 - 1.13373| < ε :=
sorry

end NUMINAMATH_CALUDE_painting_price_change_l3135_313556


namespace NUMINAMATH_CALUDE_infinitely_many_y_greater_than_sqrt_n_l3135_313541

theorem infinitely_many_y_greater_than_sqrt_n
  (x y : ℕ → ℕ+)
  (h : ∀ n : ℕ, n ≥ 1 → (y (n + 1) : ℚ) / (x (n + 1) : ℚ) > (y n : ℚ) / (x n : ℚ)) :
  Set.Infinite {n : ℕ | (y n : ℝ) > Real.sqrt n} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_y_greater_than_sqrt_n_l3135_313541


namespace NUMINAMATH_CALUDE_triangle_value_l3135_313580

theorem triangle_value (triangle q r : ℚ) 
  (eq1 : triangle + q = 75)
  (eq2 : (triangle + q) + r = 138)
  (eq3 : r = q / 3) :
  triangle = -114 := by
sorry

end NUMINAMATH_CALUDE_triangle_value_l3135_313580


namespace NUMINAMATH_CALUDE_satellite_height_scientific_notation_l3135_313550

/-- The height of a medium-high orbit satellite in China's Beidou navigation system. -/
def satellite_height : ℝ := 21500000

/-- Scientific notation representation of the satellite height. -/
def satellite_height_scientific : ℝ := 2.15 * (10 ^ 7)

/-- Theorem stating that the satellite height is equal to its scientific notation representation. -/
theorem satellite_height_scientific_notation :
  satellite_height = satellite_height_scientific := by sorry

end NUMINAMATH_CALUDE_satellite_height_scientific_notation_l3135_313550


namespace NUMINAMATH_CALUDE_windows_preference_count_survey_results_l3135_313520

/-- Represents the survey results of college students' computer brand preferences --/
structure SurveyResults where
  total : ℕ
  mac_preference : ℕ
  no_preference : ℕ
  both_preference : ℕ
  windows_preference : ℕ

/-- Theorem stating the number of students preferring Windows to Mac --/
theorem windows_preference_count (survey : SurveyResults) : 
  survey.total = 210 →
  survey.mac_preference = 60 →
  survey.no_preference = 90 →
  survey.both_preference = survey.mac_preference / 3 →
  survey.windows_preference = 40 := by
  sorry

/-- Main theorem proving the survey results --/
theorem survey_results : ∃ (survey : SurveyResults), 
  survey.total = 210 ∧
  survey.mac_preference = 60 ∧
  survey.no_preference = 90 ∧
  survey.both_preference = survey.mac_preference / 3 ∧
  survey.windows_preference = 40 := by
  sorry

end NUMINAMATH_CALUDE_windows_preference_count_survey_results_l3135_313520


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3135_313506

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 7
  let b := (6 : ℚ) / 11
  (a + b) / 2 = 75 / 154 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3135_313506


namespace NUMINAMATH_CALUDE_minimum_employees_needed_l3135_313583

theorem minimum_employees_needed 
  (water_pollution : ℕ) 
  (air_pollution : ℕ) 
  (both : ℕ) 
  (h1 : water_pollution = 85) 
  (h2 : air_pollution = 73) 
  (h3 : both = 27) 
  (h4 : both ≤ water_pollution ∧ both ≤ air_pollution) : 
  water_pollution + air_pollution - both = 131 :=
sorry

end NUMINAMATH_CALUDE_minimum_employees_needed_l3135_313583


namespace NUMINAMATH_CALUDE_tangent_line_equality_l3135_313517

noncomputable def f (x : ℝ) : ℝ := Real.log x

def g (a x : ℝ) : ℝ := a * x^2 - a

theorem tangent_line_equality (a : ℝ) : 
  (∀ x, deriv f x = deriv (g a) x) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equality_l3135_313517


namespace NUMINAMATH_CALUDE_average_equals_5x_minus_9_l3135_313509

theorem average_equals_5x_minus_9 (x : ℚ) : 
  (1/3 : ℚ) * ((x + 8) + (8*x + 3) + (3*x + 9)) = 5*x - 9 → x = 47/3 := by
sorry

end NUMINAMATH_CALUDE_average_equals_5x_minus_9_l3135_313509


namespace NUMINAMATH_CALUDE_customers_in_other_countries_l3135_313537

/-- Represents the number of customers in different regions --/
structure CustomerDistribution where
  total : Nat
  usa : Nat
  canada : Nat

/-- Calculates the number of customers in other countries --/
def customersInOtherCountries (d : CustomerDistribution) : Nat :=
  d.total - (d.usa + d.canada)

/-- Theorem stating the number of customers in other countries --/
theorem customers_in_other_countries :
  let d : CustomerDistribution := {
    total := 7422,
    usa := 723,
    canada := 1297
  }
  customersInOtherCountries d = 5402 := by
  sorry

#eval customersInOtherCountries {total := 7422, usa := 723, canada := 1297}

end NUMINAMATH_CALUDE_customers_in_other_countries_l3135_313537


namespace NUMINAMATH_CALUDE_height_classification_groups_l3135_313586

/-- Given the heights of students in a class, calculate the number of groups needed for classification --/
theorem height_classification_groups 
  (tallest_height : ℕ) 
  (shortest_height : ℕ) 
  (class_width : ℕ) 
  (h1 : tallest_height = 175) 
  (h2 : shortest_height = 150) 
  (h3 : class_width = 3) : 
  ℕ := by
  sorry

#check height_classification_groups

end NUMINAMATH_CALUDE_height_classification_groups_l3135_313586


namespace NUMINAMATH_CALUDE_quadratic_roots_distance_bounds_l3135_313559

theorem quadratic_roots_distance_bounds (z₁ z₂ m : ℂ) (α β : ℂ) :
  (∀ x : ℂ, x^2 + z₁*x + z₂ + m = 0 ↔ x = α ∨ x = β) →
  z₁^2 - 4*z₂ = 16 + 20*I →
  Complex.abs (α - β) = 2 * Real.sqrt 7 →
  (Complex.abs m ≤ 7 + Real.sqrt 41 ∧ Complex.abs m ≥ 7 - Real.sqrt 41) ∧
  (∃ m₁ m₂ : ℂ, Complex.abs m₁ = 7 + Real.sqrt 41 ∧ Complex.abs m₂ = 7 - Real.sqrt 41) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_distance_bounds_l3135_313559


namespace NUMINAMATH_CALUDE_temperature_difference_product_of_N_values_l3135_313535

theorem temperature_difference (N : ℤ) : 
  (∃ D M : ℤ, 
    M = D + N ∧ 
    abs ((M - 8) - (D + 5)) = 3) → 
  (N = 10 ∨ N = 16) :=
by sorry

theorem product_of_N_values : 
  (∀ N : ℤ, (∃ D M : ℤ, 
    M = D + N ∧ 
    abs ((M - 8) - (D + 5)) = 3) → 
  (N = 10 ∨ N = 16)) → 
  (10 * 16 = 160) :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_product_of_N_values_l3135_313535


namespace NUMINAMATH_CALUDE_octal_subtraction_l3135_313504

/-- Converts a base 8 number represented as a list of digits to a natural number -/
def octalToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a natural number to its base 8 representation as a list of digits -/
def natToOctal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else go (m / 8) ((m % 8) :: acc)
    go n []

theorem octal_subtraction :
  let a := [1, 3, 5, 2]
  let b := [0, 6, 7, 4]
  let result := [1, 4, 5, 6]
  octalToNat a - octalToNat b = octalToNat result := by
  sorry

end NUMINAMATH_CALUDE_octal_subtraction_l3135_313504


namespace NUMINAMATH_CALUDE_thirty_is_seventy_five_percent_of_forty_l3135_313562

theorem thirty_is_seventy_five_percent_of_forty :
  ∀ x : ℝ, (75 / 100) * x = 30 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_thirty_is_seventy_five_percent_of_forty_l3135_313562


namespace NUMINAMATH_CALUDE_sticker_theorem_l3135_313547

def sticker_problem (initial_stickers : ℕ) (stickers_per_friend : ℕ) (num_friends : ℕ) 
  (remaining_stickers : ℕ) (justin_diff : ℕ) : Prop :=
  let total_to_friends := stickers_per_friend * num_friends
  let total_given_away := initial_stickers - remaining_stickers
  let mandy_and_justin := total_given_away - total_to_friends
  let mandy_stickers := (mandy_and_justin + justin_diff) / 2
  mandy_stickers - total_to_friends = 2

theorem sticker_theorem : 
  sticker_problem 72 4 3 42 10 := by sorry

end NUMINAMATH_CALUDE_sticker_theorem_l3135_313547


namespace NUMINAMATH_CALUDE_solve_watermelon_problem_l3135_313532

def watermelon_problem (michael_weight : ℝ) (clay_multiplier : ℝ) (john_fraction : ℝ) : Prop :=
  let clay_weight := michael_weight * clay_multiplier
  let john_weight := clay_weight * john_fraction
  john_weight = 12

theorem solve_watermelon_problem :
  watermelon_problem 8 3 (1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_watermelon_problem_l3135_313532


namespace NUMINAMATH_CALUDE_permutation_formula_l3135_313590

def A (n k : ℕ) : ℕ :=
  (List.range k).foldl (fun acc i => acc * (n - i)) n

theorem permutation_formula (n k : ℕ) (h : k ≤ n) :
  A n k = (List.range k).foldl (fun acc i => acc * (n - i)) n :=
by sorry

end NUMINAMATH_CALUDE_permutation_formula_l3135_313590


namespace NUMINAMATH_CALUDE_union_implies_a_zero_l3135_313521

theorem union_implies_a_zero (a : ℝ) : 
  let A : Set ℝ := {1, a^2}
  let B : Set ℝ := {a, -1}
  A ∪ B = {-1, a, 1} → a = 0 := by
sorry

end NUMINAMATH_CALUDE_union_implies_a_zero_l3135_313521


namespace NUMINAMATH_CALUDE_max_value_expression_l3135_313592

theorem max_value_expression (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 3) (h5 : x = y) : 
  (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 9/4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3135_313592


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l3135_313530

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- The main theorem stating that if the given points are collinear, then k = 24. -/
theorem collinear_points_k_value (k : ℝ) :
  collinear 1 (-2) 3 2 6 (k/3) → k = 24 := by
  sorry

#check collinear_points_k_value

end NUMINAMATH_CALUDE_collinear_points_k_value_l3135_313530


namespace NUMINAMATH_CALUDE_inequality_proof_l3135_313597

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (b + c - a) + b^2 * (a + c - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3135_313597


namespace NUMINAMATH_CALUDE_smallest_prime_digit_sum_28_l3135_313567

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a natural number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem: 1999 is the smallest prime number whose digits sum to 28 -/
theorem smallest_prime_digit_sum_28 :
  (∀ p : ℕ, p < 1999 → (is_prime p ∧ digit_sum p = 28) → False) ∧
  is_prime 1999 ∧ digit_sum 1999 = 28 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_digit_sum_28_l3135_313567


namespace NUMINAMATH_CALUDE_class_b_wins_l3135_313579

/-- Represents the grades in a class --/
structure ClassGrades where
  excellent : ℕ
  good : ℕ
  average : ℕ
  satisfactory : ℕ

/-- Calculates the average grade for a class --/
def averageGrade (cg : ClassGrades) (totalStudents : ℕ) : ℚ :=
  (5 * cg.excellent + 4 * cg.good + 3 * cg.average + 2 * cg.satisfactory) / totalStudents

theorem class_b_wins (classA classB : ClassGrades) : 
  classA.excellent = 6 ∧
  classA.good = 16 ∧
  classA.average = 10 ∧
  classA.satisfactory = 8 ∧
  classB.excellent = 5 ∧
  classB.good = 15 ∧
  classB.average = 15 ∧
  classB.satisfactory = 3 →
  averageGrade classB 38 > averageGrade classA 40 := by
  sorry

#eval averageGrade ⟨6, 16, 10, 8⟩ 40
#eval averageGrade ⟨5, 15, 15, 3⟩ 38

end NUMINAMATH_CALUDE_class_b_wins_l3135_313579


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3135_313512

theorem complex_fraction_simplification :
  let z : ℂ := (5 + 7*I) / (2 + 3*I)
  z = 31/13 - (1/13)*I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3135_313512


namespace NUMINAMATH_CALUDE_complex_product_QED_l3135_313589

theorem complex_product_QED : 
  let Q : ℂ := 5 + 3 * Complex.I
  let E : ℂ := 2 * Complex.I
  let D : ℂ := 5 - 3 * Complex.I
  Q * E * D = 68 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_product_QED_l3135_313589


namespace NUMINAMATH_CALUDE_square_mirror_side_length_l3135_313552

theorem square_mirror_side_length 
  (wall_width : ℝ) 
  (wall_length : ℝ) 
  (mirror_area_ratio : ℝ) :
  wall_width = 42 →
  wall_length = 27.428571428571427 →
  mirror_area_ratio = 1 / 2 →
  ∃ (mirror_side : ℝ), 
    mirror_side = 24 ∧ 
    mirror_side^2 = mirror_area_ratio * wall_width * wall_length :=
by sorry

end NUMINAMATH_CALUDE_square_mirror_side_length_l3135_313552


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l3135_313555

/-- 
Given a quadratic polynomial of the form 6x^2 + nx + 144, where n is an integer,
this theorem states that the largest value of n for which the polynomial 
can be factored as the product of two linear factors with integer coefficients is 865.
-/
theorem largest_n_for_factorization : 
  (∃ (n : ℤ), ∀ (A B : ℤ), 
    (6 * A = 6 ∧ A + 6 * B = n ∧ A * B = 144) → 
    (∀ (m : ℤ), (∃ (C D : ℤ), 6 * C = 6 ∧ C + 6 * D = m ∧ C * D = 144) → m ≤ n)) ∧
  (∃ (A B : ℤ), 6 * A = 6 ∧ A + 6 * B = 865 ∧ A * B = 144) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l3135_313555


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3135_313533

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality : B ∩ (U \ A) = {3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3135_313533


namespace NUMINAMATH_CALUDE_unique_positive_number_l3135_313515

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 8 = 128 / x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l3135_313515


namespace NUMINAMATH_CALUDE_binomial_30_3_minus_10_l3135_313575

theorem binomial_30_3_minus_10 : Nat.choose 30 3 - 10 = 4050 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_minus_10_l3135_313575


namespace NUMINAMATH_CALUDE_prob_different_suits_l3135_313568

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suits in a standard deck -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- Function to get the suit of a card -/
def getSuit (card : Fin 52) : Suit := sorry

/-- Probability of drawing three cards of different suits -/
def probDifferentSuits (d : Deck) : ℚ :=
  (39 : ℚ) / 51 * (26 : ℚ) / 50

/-- Theorem stating the probability of drawing three cards of different suits -/
theorem prob_different_suits (d : Deck) :
  probDifferentSuits d = 169 / 425 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_suits_l3135_313568


namespace NUMINAMATH_CALUDE_discount_ratio_proof_l3135_313516

/-- Proves that given a 15% discount on an item, if a person with $500 still needs $95 more to purchase it, the ratio of the additional money needed to the initial amount is 19:100. -/
theorem discount_ratio_proof (initial_amount : ℝ) (additional_needed : ℝ) (discount_rate : ℝ) :
  initial_amount = 500 →
  additional_needed = 95 →
  discount_rate = 0.15 →
  (additional_needed / initial_amount) = (19 / 100) :=
by sorry

end NUMINAMATH_CALUDE_discount_ratio_proof_l3135_313516


namespace NUMINAMATH_CALUDE_quadratic_intersection_and_vertex_l3135_313546

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*(m+1)*x - m + 1

-- Define the discriminant of the quadratic function
def discriminant (m : ℝ) : ℝ := 4*(m^2 + 3*m)

-- Define the x-coordinate of the vertex
def vertex_x (m : ℝ) : ℝ := -(m + 1)

-- Define the y-coordinate of the vertex
def vertex_y (m : ℝ) : ℝ := -(m^2 + 3*m)

theorem quadratic_intersection_and_vertex (m : ℝ) :
  -- Part 1: The number of intersection points with the x-axis is 0, 1, or 2
  (∃ x : ℝ, f m x = 0 ∧ 
    (∀ y : ℝ, f m y = 0 → y = x ∨ 
    (∃ z : ℝ, z ≠ x ∧ z ≠ y ∧ f m z = 0))) ∨
  (∀ x : ℝ, f m x ≠ 0) ∧
  -- Part 2: If the line y = x + 1 passes through the vertex, then m = -2 or m = 0
  (vertex_y m = vertex_x m + 1 → m = -2 ∨ m = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersection_and_vertex_l3135_313546


namespace NUMINAMATH_CALUDE_square_value_l3135_313522

theorem square_value (square : ℚ) (h : (1:ℚ)/9 + (1:ℚ)/18 = (1:ℚ)/square) : square = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l3135_313522


namespace NUMINAMATH_CALUDE_shekars_average_marks_l3135_313557

def mathematics_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 47
def biology_score : ℕ := 85

def total_subjects : ℕ := 5

theorem shekars_average_marks :
  (mathematics_score + science_score + social_studies_score + english_score + biology_score) / total_subjects = 71 := by
  sorry

end NUMINAMATH_CALUDE_shekars_average_marks_l3135_313557


namespace NUMINAMATH_CALUDE_sum_of_binary_numbers_l3135_313542

/-- Convert a binary string to a natural number -/
def binary_to_nat (s : String) : ℕ :=
  s.foldl (fun acc c => 2 * acc + (if c = '1' then 1 else 0)) 0

/-- Convert a natural number to a binary string -/
def nat_to_binary (n : ℕ) : String :=
  if n = 0 then "0" else
    let rec aux (m : ℕ) : String :=
      if m = 0 then "" else aux (m / 2) ++ (if m % 2 = 1 then "1" else "0")
    aux n

theorem sum_of_binary_numbers :
  let a := binary_to_nat "1100"
  let b := binary_to_nat "101"
  let c := binary_to_nat "11"
  let d := binary_to_nat "11011"
  let e := binary_to_nat "100"
  nat_to_binary (a + b + c + d + e) = "1000101" := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_numbers_l3135_313542


namespace NUMINAMATH_CALUDE_banking_problem_l3135_313507

/-- Calculates the final amount after deposit growth and withdrawal fee --/
def finalAmount (initialDeposit : ℝ) (growthRate : ℝ) (feeRate : ℝ) : ℝ :=
  initialDeposit * (1 + growthRate) * (1 - feeRate)

/-- Represents the banking problem with Vlad and Dima's deposits --/
theorem banking_problem (initialDeposit : ℝ) 
  (h_initial : initialDeposit = 3000) 
  (vladGrowthRate dimaGrowthRate vladFeeRate dimaFeeRate : ℝ)
  (h_vlad_growth : vladGrowthRate = 0.2)
  (h_vlad_fee : vladFeeRate = 0.1)
  (h_dima_growth : dimaGrowthRate = 0.4)
  (h_dima_fee : dimaFeeRate = 0.2) :
  finalAmount initialDeposit dimaGrowthRate dimaFeeRate - 
  finalAmount initialDeposit vladGrowthRate vladFeeRate = 120 := by
  sorry


end NUMINAMATH_CALUDE_banking_problem_l3135_313507


namespace NUMINAMATH_CALUDE_river_distance_l3135_313544

theorem river_distance (d : ℝ) : 
  (¬ (d ≥ 8)) → (¬ (d ≤ 7)) → (¬ (d ≤ 6)) → (7 < d ∧ d < 8) := by
  sorry

end NUMINAMATH_CALUDE_river_distance_l3135_313544


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l3135_313573

theorem consecutive_integers_sum_of_squares : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a * (a + 1) * (a + 2) = 12 * (3 * a + 3)) → 
  (a^2 + (a + 1)^2 + (a + 2)^2 = 149) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l3135_313573


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l3135_313514

theorem abs_sum_inequality (k : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 3| > k) ↔ k < 4 :=
sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l3135_313514


namespace NUMINAMATH_CALUDE_total_hamburgers_bought_l3135_313591

/-- Calculates the total number of hamburgers bought given the conditions --/
theorem total_hamburgers_bought
  (total_spent : ℚ)
  (single_burger_cost : ℚ)
  (double_burger_cost : ℚ)
  (double_burgers_count : ℕ)
  (h1 : total_spent = 68.5)
  (h2 : single_burger_cost = 1)
  (h3 : double_burger_cost = 1.5)
  (h4 : double_burgers_count = 37) :
  ∃ (single_burgers_count : ℕ),
    single_burgers_count + double_burgers_count = 50 ∧
    total_spent = single_burger_cost * single_burgers_count + double_burger_cost * double_burgers_count :=
by
  sorry


end NUMINAMATH_CALUDE_total_hamburgers_bought_l3135_313591


namespace NUMINAMATH_CALUDE_job_completion_time_l3135_313543

theorem job_completion_time (p_rate q_rate : ℚ) (t : ℚ) : 
  p_rate = 1/4 →
  q_rate = 1/15 →
  t * (p_rate + q_rate) + 1/5 * p_rate = 1 →
  t = 3 :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_l3135_313543


namespace NUMINAMATH_CALUDE_complement_of_P_l3135_313553

def U : Set ℝ := Set.univ

def P : Set ℝ := {x : ℝ | x^2 - 5*x - 6 ≥ 0}

theorem complement_of_P (x : ℝ) : x ∈ Set.compl P ↔ x ∈ Set.Ioo (-1) 6 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_P_l3135_313553


namespace NUMINAMATH_CALUDE_edward_tickets_l3135_313513

/-- The number of tickets Edward spent at the 'dunk a clown' booth -/
def spent_tickets : ℕ := 23

/-- The cost of each ride in tickets -/
def ride_cost : ℕ := 7

/-- The number of rides Edward could have gone on with the remaining tickets -/
def possible_rides : ℕ := 8

/-- The total number of tickets Edward bought at the state fair -/
def total_tickets : ℕ := spent_tickets + ride_cost * possible_rides

theorem edward_tickets : total_tickets = 79 := by sorry

end NUMINAMATH_CALUDE_edward_tickets_l3135_313513


namespace NUMINAMATH_CALUDE_least_common_addition_of_primes_l3135_313566

theorem least_common_addition_of_primes (x y : ℕ) : 
  Nat.Prime x → Nat.Prime y → x < y → 4 * x + y = 87 → x + y = 81 := by
  sorry

end NUMINAMATH_CALUDE_least_common_addition_of_primes_l3135_313566


namespace NUMINAMATH_CALUDE_class_average_problem_l3135_313545

theorem class_average_problem (avg_class1 avg_combined : ℝ) (n1 n2 : ℕ) 
  (h1 : avg_class1 = 40)
  (h2 : n1 = 24)
  (h3 : n2 = 50)
  (h4 : avg_combined = 53.513513513513516)
  (h5 : (n1 : ℝ) * avg_class1 + (n2 : ℝ) * (((n1 + n2 : ℕ) : ℝ) * avg_combined - (n1 : ℝ) * avg_class1) / (n2 : ℝ) = 
        (n1 + n2 : ℕ) * avg_combined) :
  (((n1 + n2 : ℕ) : ℝ) * avg_combined - (n1 : ℝ) * avg_class1) / (n2 : ℝ) = 60 := by
  sorry

#check class_average_problem

end NUMINAMATH_CALUDE_class_average_problem_l3135_313545


namespace NUMINAMATH_CALUDE_range_of_a_l3135_313551

def p (x : ℝ) : Prop := 0 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 1

def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

theorem range_of_a :
  (∀ x a : ℝ, q x a → p x) ∧
  (∃ x : ℝ, p x ∧ ∀ a : ℝ, ¬(q x a)) →
  ∀ a : ℝ, (0 ≤ a ∧ a ≤ 1/2) ↔ (∃ x : ℝ, q x a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3135_313551


namespace NUMINAMATH_CALUDE_square_area_error_l3135_313558

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * 1.1
  let actual_area := s ^ 2
  let calculated_area := measured_side ^ 2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.21 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l3135_313558


namespace NUMINAMATH_CALUDE_alien_martian_limb_difference_l3135_313593

/-- The number of arms an Alien has -/
def alien_arms : ℕ := 3

/-- The number of legs an Alien has -/
def alien_legs : ℕ := 8

/-- The number of arms a Martian has -/
def martian_arms : ℕ := 2 * alien_arms

/-- The number of legs a Martian has -/
def martian_legs : ℕ := alien_legs / 2

/-- The total number of limbs for one Alien -/
def alien_limbs : ℕ := alien_arms + alien_legs

/-- The total number of limbs for one Martian -/
def martian_limbs : ℕ := martian_arms + martian_legs

/-- The number of Aliens and Martians we're comparing -/
def number_of_creatures : ℕ := 5

theorem alien_martian_limb_difference :
  number_of_creatures * alien_limbs - number_of_creatures * martian_limbs = 5 := by
  sorry

end NUMINAMATH_CALUDE_alien_martian_limb_difference_l3135_313593


namespace NUMINAMATH_CALUDE_line_circle_intersection_k_range_l3135_313581

/-- Given a line y = kx + 3 intersecting a circle (x-4)^2 + (y-3)^2 = 4 at two points M and N,
    where |MN| ≥ 2√3, prove that -√15/15 ≤ k ≤ √15/15 -/
theorem line_circle_intersection_k_range (k : ℝ) :
  (∃ M N : ℝ × ℝ,
    (M.1 - 4)^2 + (M.2 - 3)^2 = 4 ∧
    (N.1 - 4)^2 + (N.2 - 3)^2 = 4 ∧
    M.2 = k * M.1 + 3 ∧
    N.2 = k * N.1 + 3 ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12) →
  -Real.sqrt 15 / 15 ≤ k ∧ k ≤ Real.sqrt 15 / 15 := by
  sorry


end NUMINAMATH_CALUDE_line_circle_intersection_k_range_l3135_313581


namespace NUMINAMATH_CALUDE_problem_solid_surface_area_l3135_313531

/-- Represents a solid formed by unit cubes --/
structure CubeSolid where
  base_layer : Nat
  second_layer : Nat
  third_layer : Nat
  top_layer : Nat

/-- Calculates the surface area of a CubeSolid --/
def surface_area (solid : CubeSolid) : Nat :=
  sorry

/-- The specific solid described in the problem --/
def problem_solid : CubeSolid :=
  { base_layer := 4
  , second_layer := 4
  , third_layer := 3
  , top_layer := 1 }

/-- Theorem stating that the surface area of the problem_solid is 28 --/
theorem problem_solid_surface_area :
  surface_area problem_solid = 28 :=
sorry

end NUMINAMATH_CALUDE_problem_solid_surface_area_l3135_313531


namespace NUMINAMATH_CALUDE_interesting_2018_gon_after_marked_removal_l3135_313576

/-- A convex polygon with n vertices --/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_convex : sorry

/-- A coloring of vertices in two colors --/
def Coloring (n : ℕ) := Fin n → Bool

/-- The sum of angles at vertices of a given color in a polygon --/
def sumAngles (p : ConvexPolygon n) (c : Coloring n) (color : Bool) : ℝ := sorry

/-- A polygon is interesting if the sum of angles of one color equals the sum of angles of the other color --/
def isInteresting (p : ConvexPolygon n) (c : Coloring n) : Prop :=
  sumAngles p c true = sumAngles p c false

/-- Remove a vertex from a polygon --/
def removeVertex (p : ConvexPolygon (n + 1)) (i : Fin (n + 1)) : ConvexPolygon n := sorry

/-- The theorem to be proved --/
theorem interesting_2018_gon_after_marked_removal
  (p : ConvexPolygon 2019)
  (marked : Fin 2019)
  (h : ∀ (i : Fin 2019), i ≠ marked → ∃ (c : Coloring 2018), isInteresting (removeVertex p i) c) :
  ∃ (c : Coloring 2018), isInteresting (removeVertex p marked) c :=
sorry

end NUMINAMATH_CALUDE_interesting_2018_gon_after_marked_removal_l3135_313576


namespace NUMINAMATH_CALUDE_horner_operations_for_f_l3135_313510

def f (x : ℝ) := 6 * x^6 + 5

def horner_operations (p : ℝ → ℝ) (degree : ℕ) : ℕ × ℕ :=
  (degree, degree)

theorem horner_operations_for_f :
  horner_operations f 6 = (6, 6) := by sorry

end NUMINAMATH_CALUDE_horner_operations_for_f_l3135_313510
