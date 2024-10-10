import Mathlib

namespace acute_triangle_exists_l1121_112114

/-- Given 5 real numbers representing lengths of line segments,
    if any three of these numbers can form a triangle,
    then there exists a combination of three numbers that forms a triangle with all acute angles. -/
theorem acute_triangle_exists (a b c d e : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) 
  (h_triangle : ∀ (x y z : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) →
                               (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) →
                               (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) →
                               x ≠ y ∧ y ≠ z ∧ x ≠ z →
                               x + y > z ∧ y + z > x ∧ x + z > y) :
  ∃ (x y z : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
                 (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
                 (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧
                 x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
                 x^2 + y^2 > z^2 ∧ y^2 + z^2 > x^2 ∧ x^2 + z^2 > y^2 :=
by sorry


end acute_triangle_exists_l1121_112114


namespace train_passing_jogger_time_l1121_112177

/-- The time taken for a train to pass a jogger under specific conditions -/
theorem train_passing_jogger_time (jogger_speed train_speed : ℝ) 
  (initial_distance train_length : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 240 →
  train_length = 120 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 36 := by
  sorry

end train_passing_jogger_time_l1121_112177


namespace min_value_expression_l1121_112181

theorem min_value_expression (a : ℝ) (h : a > 1) :
  (4 / (a - 1)) + a ≥ 5 ∧ ((4 / (a - 1)) + a = 5 ↔ a = 3) :=
sorry

end min_value_expression_l1121_112181


namespace cube_sum_equals_negative_27_l1121_112157

theorem cube_sum_equals_negative_27 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) : 
  a^3 + b^3 + c^3 = -27 := by
  sorry

end cube_sum_equals_negative_27_l1121_112157


namespace pool_capacity_l1121_112178

theorem pool_capacity (C : ℝ) 
  (h1 : C / 4 - C / 6 = C / 12)  -- Net rate of water level change
  (h2 : C - 3 * (C / 12) = 90)   -- Remaining water after 3 hours
  : C = 120 := by
  sorry

end pool_capacity_l1121_112178


namespace sufficient_not_necessary_condition_l1121_112131

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ a, a > 1 → a^2 > 1) ∧ (∃ a, a^2 > 1 ∧ a ≤ 1) :=
by sorry

end sufficient_not_necessary_condition_l1121_112131


namespace regular_polygon_side_length_l1121_112137

theorem regular_polygon_side_length 
  (n : ℕ) 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 48) 
  (h₂ : a₂ = 55) 
  (h₃ : n > 2) 
  (h₄ : (n * a₃^2) / (4 * Real.tan (π / n)) = 
        (n * a₁^2) / (4 * Real.tan (π / n)) + 
        (n * a₂^2) / (4 * Real.tan (π / n))) : 
  a₃ = 73 := by
sorry

end regular_polygon_side_length_l1121_112137


namespace intersection_point_is_unique_l1121_112141

/-- The line in 3D space --/
def line (t : ℝ) : ℝ × ℝ × ℝ :=
  (1 + 2*t, -2 - 5*t, 3 - 2*t)

/-- The plane in 3D space --/
def plane (x y z : ℝ) : Prop :=
  x + 2*y - 5*z + 16 = 0

/-- The intersection point --/
def intersection_point : ℝ × ℝ × ℝ :=
  (3, -7, 1)

theorem intersection_point_is_unique :
  (∃! p : ℝ × ℝ × ℝ, ∃ t : ℝ, line t = p ∧ plane p.1 p.2.1 p.2.2) ∧
  (∃ t : ℝ, line t = intersection_point ∧ plane intersection_point.1 intersection_point.2.1 intersection_point.2.2) :=
sorry

end intersection_point_is_unique_l1121_112141


namespace not_sufficient_not_necessary_l1121_112148

theorem not_sufficient_not_necessary (a b : ℝ) : 
  (∃ x y : ℝ, x + y > 1 ∧ x^2 + y^2 ≤ 1) ∧ 
  (∃ u v : ℝ, u^2 + v^2 > 1 ∧ u + v ≤ 1) := by
  sorry

end not_sufficient_not_necessary_l1121_112148


namespace number_operations_equivalence_l1121_112193

theorem number_operations_equivalence (x : ℝ) : ((x * (5/6)) / (2/3)) - 2 = (x * (5/4)) - 2 := by
  sorry

end number_operations_equivalence_l1121_112193


namespace conjunction_is_false_l1121_112127

theorem conjunction_is_false :
  let p := ∀ x : ℝ, x < 1 → x < 2
  let q := ∃ x : ℝ, x^2 + 1 = 0
  ¬(p ∧ q) := by sorry

end conjunction_is_false_l1121_112127


namespace parabola_max_area_l1121_112171

/-- A parabola with y-axis symmetry -/
structure Parabola where
  a : ℝ
  c : ℝ

/-- The equation of a parabola -/
def Parabola.equation (p : Parabola) (x : ℝ) : ℝ := p.a * x^2 + p.c

/-- The condition that the parabola is concave up -/
def Parabola.concaveUp (p : Parabola) : Prop := p.a > 0

/-- The condition that the parabola touches the graph y = 1 - |x| -/
def Parabola.touchesGraph (p : Parabola) : Prop :=
  ∃ x₀ : ℝ, p.equation x₀ = 1 - |x₀| ∧ 
    (deriv p.equation) x₀ = if x₀ ≥ 0 then -1 else 1

/-- The area between the parabola and the x-axis -/
noncomputable def Parabola.area (p : Parabola) : ℝ :=
  ∫ x in (-Real.sqrt (1/p.a))..(Real.sqrt (1/p.a)), p.equation x

/-- The theorem statement -/
theorem parabola_max_area :
  ∀ p : Parabola, 
    p.concaveUp → 
    p.touchesGraph → 
    p.area ≤ Parabola.area ⟨1, 3/4⟩ :=
sorry

end parabola_max_area_l1121_112171


namespace opposite_of_negative_three_l1121_112159

theorem opposite_of_negative_three : -((-3) : ℝ) = 3 := by sorry

end opposite_of_negative_three_l1121_112159


namespace initial_plant_ratio_l1121_112164

/-- Represents the number and types of plants in Roxy's garden -/
structure Garden where
  flowering : ℕ
  fruiting : ℕ

/-- Represents the transactions of buying and giving away plants -/
structure Transactions where
  bought_flowering : ℕ
  bought_fruiting : ℕ
  given_flowering : ℕ
  given_fruiting : ℕ

/-- Calculates the final number of plants after transactions -/
def final_plants (initial : Garden) (trans : Transactions) : ℕ :=
  initial.flowering + initial.fruiting + trans.bought_flowering + trans.bought_fruiting - 
  trans.given_flowering - trans.given_fruiting

/-- Theorem stating the initial ratio of fruiting to flowering plants -/
theorem initial_plant_ratio (initial : Garden) (trans : Transactions) :
  initial.flowering = 7 ∧ 
  trans.bought_flowering = 3 ∧ 
  trans.bought_fruiting = 2 ∧
  trans.given_flowering = 1 ∧
  trans.given_fruiting = 4 ∧
  final_plants initial trans = 21 →
  initial.fruiting = 2 * initial.flowering :=
by
  sorry


end initial_plant_ratio_l1121_112164


namespace max_basketballs_l1121_112136

/-- Represents the prices and quantities of soccer balls and basketballs --/
structure BallPurchase where
  soccer_price : ℝ
  basketball_price : ℝ
  soccer_quantity : ℕ
  basketball_quantity : ℕ

/-- The conditions of the ball purchase problem --/
def valid_purchase (p : BallPurchase) : Prop :=
  p.basketball_price = 2 * p.soccer_price - 30 ∧
  3 * p.soccer_price * p.soccer_quantity = 2 * p.basketball_price * p.basketball_quantity ∧
  p.soccer_quantity + p.basketball_quantity = 200 ∧
  p.soccer_price * p.soccer_quantity + p.basketball_price * p.basketball_quantity ≤ 15500

/-- The theorem stating the maximum number of basketballs that can be purchased --/
theorem max_basketballs (p : BallPurchase) :
  valid_purchase p → p.basketball_quantity ≤ 116 := by
  sorry

end max_basketballs_l1121_112136


namespace arithmetic_expression_equality_l1121_112188

theorem arithmetic_expression_equality : 2 + 3 * 4 - 5 + 6 * (2 - 1) = 15 := by
  sorry

end arithmetic_expression_equality_l1121_112188


namespace total_books_l1121_112135

-- Define the number of books for each person
def sam_books : ℕ := 110

-- Joan has twice as many books as Sam
def joan_books : ℕ := 2 * sam_books

-- Tom has half the number of books as Joan
def tom_books : ℕ := joan_books / 2

-- Alice has 3 times the number of books Tom has
def alice_books : ℕ := 3 * tom_books

-- Theorem statement
theorem total_books : sam_books + joan_books + tom_books + alice_books = 770 := by
  sorry

end total_books_l1121_112135


namespace complex_equation_solution_product_l1121_112156

theorem complex_equation_solution_product (x : ℂ) :
  x^3 + x^2 + 3*x = 2 + 2*Complex.I →
  ∃ (x₁ x₂ : ℂ), x₁ ≠ x₂ ∧ 
    x₁^3 + x₁^2 + 3*x₁ = 2 + 2*Complex.I ∧
    x₂^3 + x₂^2 + 3*x₂ = 2 + 2*Complex.I ∧
    (x₁.re * x₂.re = 1 - Real.sqrt 2) :=
by sorry

end complex_equation_solution_product_l1121_112156


namespace sqrt_product_simplification_l1121_112158

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (12 * q) * Real.sqrt (8 * q^2) * Real.sqrt (9 * q^5) = 12 * q^4 * Real.sqrt 6 := by
  sorry

end sqrt_product_simplification_l1121_112158


namespace range_of_m_l1121_112113

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 2 / x + 1 / y = 1 / 3) (h_ineq : ∀ m : ℝ, x + 2 * y > m^2 - 2 * m) :
  -4 < m ∧ m < 6 := by
sorry

end range_of_m_l1121_112113


namespace angle_sum_from_tangent_roots_l1121_112122

theorem angle_sum_from_tangent_roots (α β : Real) :
  (∃ x y : Real, x^2 + 3 * Real.sqrt 3 * x + 4 = 0 ∧ 
                 y^2 + 3 * Real.sqrt 3 * y + 4 = 0 ∧ 
                 x = Real.tan α ∧ 
                 y = Real.tan β) →
  -π/2 < α ∧ α < π/2 →
  -π/2 < β ∧ β < π/2 →
  α + β = -2*π/3 := by
sorry

end angle_sum_from_tangent_roots_l1121_112122


namespace reciprocal_sum_of_quadratic_roots_l1121_112168

theorem reciprocal_sum_of_quadratic_roots :
  ∀ (α β : ℝ),
  (∃ (a b : ℝ), 7 * a^2 + 2 * a + 6 = 0 ∧ 
                 7 * b^2 + 2 * b + 6 = 0 ∧ 
                 α = 1 / a ∧ 
                 β = 1 / b) →
  α + β = -1/3 := by
sorry

end reciprocal_sum_of_quadratic_roots_l1121_112168


namespace expected_points_is_seventeen_thirds_l1121_112133

/-- Represents the outcomes of the biased die -/
inductive Outcome
| Odd
| EvenNotSix
| Six

/-- The probability of each outcome -/
def probability (o : Outcome) : ℚ :=
  match o with
  | Outcome.Odd => 1/2
  | Outcome.EvenNotSix => 1/3
  | Outcome.Six => 1/6

/-- The points gained for each outcome -/
def points (o : Outcome) : ℚ :=
  match o with
  | Outcome.Odd => 9/2  -- Average of 1, 3, and 5
  | Outcome.EvenNotSix => 3  -- Average of 2 and 4
  | Outcome.Six => -5

/-- The expected value of points gained -/
def expected_value : ℚ :=
  (probability Outcome.Odd * points Outcome.Odd) +
  (probability Outcome.EvenNotSix * points Outcome.EvenNotSix) +
  (probability Outcome.Six * points Outcome.Six)

theorem expected_points_is_seventeen_thirds :
  expected_value = 17/3 := by
  sorry

end expected_points_is_seventeen_thirds_l1121_112133


namespace cos_equality_angle_l1121_112117

theorem cos_equality_angle (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (370 * π / 180) → n = 10 := by
  sorry

end cos_equality_angle_l1121_112117


namespace a_greater_than_b_l1121_112163

theorem a_greater_than_b (n : ℕ) (a b : ℝ) 
  (h_n : n ≥ 2) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0) 
  (h_a : a^n = a + 1) 
  (h_b : b^(2*n) = b + 3*a) : 
  a > b := by sorry

end a_greater_than_b_l1121_112163


namespace arithmetic_sequence_sum_36_l1121_112166

/-- An arithmetic sequence with sum Sₙ of the first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_36 (seq : ArithmeticSequence) 
  (h1 : 2 * seq.S 3 = 3 * seq.S 2 + 3)
  (h2 : seq.S 4 = seq.a 10) : 
  seq.S 36 = 666 := by
  sorry

end arithmetic_sequence_sum_36_l1121_112166


namespace selection_ways_l1121_112116

def total_students : ℕ := 10
def selected_students : ℕ := 4
def specific_students : ℕ := 2

theorem selection_ways : 
  (Nat.choose total_students selected_students - 
   Nat.choose (total_students - specific_students) selected_students) = 140 :=
by sorry

end selection_ways_l1121_112116


namespace point_transformation_l1121_112182

/-- Rotate a point (x, y) counterclockwise by 90° around (h, k) -/
def rotate90 (x y h k : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

/-- Reflect a point (x, y) about the line y = x -/
def reflectAboutYEqX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (c d : ℝ) :
  let (x₁, y₁) := rotate90 c d 2 3
  let (x₂, y₂) := reflectAboutYEqX x₁ y₁
  (x₂ = -3 ∧ y₂ = 8) → d - c = 1 := by
  sorry

end point_transformation_l1121_112182


namespace common_root_condition_rational_roots_if_common_root_l1121_112140

structure QuadraticEquation (α : Type) [Field α] where
  p : α
  q : α

def hasCommonRoot {α : Type} [Field α] (eq1 eq2 : QuadraticEquation α) : Prop :=
  ∃ x : α, x^2 + eq1.p * x + eq1.q = 0 ∧ x^2 + eq2.p * x + eq2.q = 0

theorem common_root_condition {α : Type} [Field α] (eq1 eq2 : QuadraticEquation α) :
  hasCommonRoot eq1 eq2 ↔ (eq1.p - eq2.p) * (eq1.p * eq2.q - eq2.p * eq1.q) + (eq1.q - eq2.q)^2 = 0 :=
sorry

theorem rational_roots_if_common_root (eq1 eq2 : QuadraticEquation ℚ) 
  (h1 : hasCommonRoot eq1 eq2) (h2 : eq1 ≠ eq2) :
  ∃ (x y : ℚ), (x^2 + eq1.p * x + eq1.q = 0 ∧ y^2 + eq1.p * y + eq1.q = 0) ∧
                (x^2 + eq2.p * x + eq2.q = 0 ∧ y^2 + eq2.p * y + eq2.q = 0) :=
sorry

end common_root_condition_rational_roots_if_common_root_l1121_112140


namespace point_wrt_y_axis_point_4_neg8_wrt_y_axis_l1121_112172

/-- Given a point A with coordinates (x, y) in a 2D plane,
    this theorem states that the coordinates of A with respect to the y-axis are (-x, y). -/
theorem point_wrt_y_axis (x y : ℝ) : 
  let A : ℝ × ℝ := (x, y)
  let A_wrt_y_axis : ℝ × ℝ := (-x, y)
  A_wrt_y_axis = (- (A.1), A.2) := by
sorry

/-- The coordinates of the point A(4, -8) with respect to the y-axis are (-4, -8). -/
theorem point_4_neg8_wrt_y_axis : 
  let A : ℝ × ℝ := (4, -8)
  let A_wrt_y_axis : ℝ × ℝ := (-4, -8)
  A_wrt_y_axis = (- (A.1), A.2) := by
sorry

end point_wrt_y_axis_point_4_neg8_wrt_y_axis_l1121_112172


namespace function_properties_l1121_112106

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 - 5 * x^2 - b * x

-- Define the derivative of f
def f_derivative (a b x : ℝ) : ℝ := 3 * a * x^2 - 10 * x - b

theorem function_properties (a b : ℝ) :
  (f_derivative a b 3 = 0) →  -- x = 3 is an extreme point
  (f a b 1 = -1) →           -- f(1) = -1
  (a = 1 ∧ b = -3) ∧         -- Part 1: a = 1 and b = -3
  (∀ x ∈ Set.Icc 2 4, f 1 (-3) x ≥ -9) ∧  -- Part 2: Minimum value on [2, 4] is -9
  (∀ x ∈ Set.Icc 2 4, f 1 (-3) x ≤ 0) ∧   -- Part 3: Maximum value on [2, 4] is 0
  (f 1 (-3) 3 = -9) ∧        -- Minimum occurs at x = 3
  (f 1 (-3) 4 = 0)           -- Maximum occurs at x = 4
  := by sorry

end function_properties_l1121_112106


namespace cubic_equation_solution_l1121_112100

theorem cubic_equation_solution (y : ℝ) :
  (((30 * y + (30 * y + 27) ^ (1/3 : ℝ)) ^ (1/3 : ℝ)) = 15) → y = 1674/15 := by
  sorry

end cubic_equation_solution_l1121_112100


namespace perpendicular_iff_x_eq_neg_one_third_l1121_112109

/-- Two vectors in R² -/
def a (x : ℝ) : Fin 2 → ℝ := ![x, 1]
def b : Fin 2 → ℝ := ![3, 1]

/-- Dot product of two vectors in R² -/
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

/-- Theorem: Vectors a and b are perpendicular if and only if x = -1/3 -/
theorem perpendicular_iff_x_eq_neg_one_third (x : ℝ) : 
  dot_product (a x) b = 0 ↔ x = -1/3 := by
  sorry

end perpendicular_iff_x_eq_neg_one_third_l1121_112109


namespace sachin_age_l1121_112142

theorem sachin_age (sachin rahul : ℝ) 
  (h1 : sachin = rahul - 9)
  (h2 : sachin / rahul = 7 / 9) : 
  sachin = 31.5 := by
sorry

end sachin_age_l1121_112142


namespace canada_population_1998_l1121_112103

/-- Proves that 30.3 million is equal to 30,300,000 --/
theorem canada_population_1998 : (30.3 : ℝ) * 1000000 = 30300000 := by
  sorry

end canada_population_1998_l1121_112103


namespace third_number_value_l1121_112120

-- Define the proportion
def proportion (a b c d : ℚ) : Prop := a * d = b * c

-- State the theorem
theorem third_number_value : 
  ∃ (third_number : ℚ), 
    proportion (75/100) (6/5) third_number 8 ∧ third_number = 5 := by
  sorry

end third_number_value_l1121_112120


namespace set_operation_result_l1121_112184

def A : Set Int := {-1, 0}
def B : Set Int := {0, 1}
def C : Set Int := {1, 2}

theorem set_operation_result : (A ∩ B) ∪ C = {0, 1, 2} := by sorry

end set_operation_result_l1121_112184


namespace mitzi_remaining_money_l1121_112196

def amusement_park_spending (initial_amount ticket_cost food_cost tshirt_cost : ℕ) : ℕ :=
  initial_amount - (ticket_cost + food_cost + tshirt_cost)

theorem mitzi_remaining_money :
  amusement_park_spending 75 30 13 23 = 9 := by
  sorry

end mitzi_remaining_money_l1121_112196


namespace smallest_n_with_divisibility_l1121_112155

def is_divisible (a b : ℕ) : Prop := ∃ k, a = b * k

theorem smallest_n_with_divisibility : ∃! N : ℕ, 
  N > 0 ∧ 
  (is_divisible N (2^2) ∨ is_divisible (N+1) (2^2) ∨ is_divisible (N+2) (2^2) ∨ is_divisible (N+3) (2^2)) ∧
  (is_divisible N (3^2) ∨ is_divisible (N+1) (3^2) ∨ is_divisible (N+2) (3^2) ∨ is_divisible (N+3) (3^2)) ∧
  (is_divisible N (5^2) ∨ is_divisible (N+1) (5^2) ∨ is_divisible (N+2) (5^2) ∨ is_divisible (N+3) (5^2)) ∧
  (is_divisible N (11^2) ∨ is_divisible (N+1) (11^2) ∨ is_divisible (N+2) (11^2) ∨ is_divisible (N+3) (11^2)) ∧
  (∀ M : ℕ, M < N → 
    ¬((is_divisible M (2^2) ∨ is_divisible (M+1) (2^2) ∨ is_divisible (M+2) (2^2) ∨ is_divisible (M+3) (2^2)) ∧
      (is_divisible M (3^2) ∨ is_divisible (M+1) (3^2) ∨ is_divisible (M+2) (3^2) ∨ is_divisible (M+3) (3^2)) ∧
      (is_divisible M (5^2) ∨ is_divisible (M+1) (5^2) ∨ is_divisible (M+2) (5^2) ∨ is_divisible (M+3) (5^2)) ∧
      (is_divisible M (11^2) ∨ is_divisible (M+1) (11^2) ∨ is_divisible (M+2) (11^2) ∨ is_divisible (M+3) (11^2)))) ∧
  N = 484 :=
by sorry

end smallest_n_with_divisibility_l1121_112155


namespace equation_solution_l1121_112185

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (4 * (x₁ - 1)^2 = 36 ∧ 4 * (x₂ - 1)^2 = 36) ∧ 
  x₁ = 4 ∧ x₂ = -2 :=
by
  sorry

end equation_solution_l1121_112185


namespace cannot_achieve_multiple_100s_l1121_112187

/-- Represents the scores for Russian, Physics, and Mathematics exams -/
structure Scores where
  russian : ℕ
  physics : ℕ
  math : ℕ

/-- Defines the initial relationship between scores -/
def initial_score_relationship (s : Scores) : Prop :=
  s.russian = s.physics - 5 ∧ s.physics = s.math - 9

/-- Represents the two types of operations allowed -/
inductive Operation
  | add_one_to_all
  | decrease_one_increase_two

/-- Applies an operation to the scores -/
def apply_operation (s : Scores) (op : Operation) : Scores :=
  match op with
  | Operation.add_one_to_all => 
      { russian := s.russian + 1, physics := s.physics + 1, math := s.math + 1 }
  | Operation.decrease_one_increase_two => 
      { russian := s.russian - 3, physics := s.physics + 1, math := s.math + 1 }
      -- Note: This is just one possible application of the second operation

/-- Checks if any score exceeds 100 -/
def exceeds_100 (s : Scores) : Prop :=
  s.russian > 100 ∨ s.physics > 100 ∨ s.math > 100

/-- Checks if more than one score is equal to 100 -/
def more_than_one_100 (s : Scores) : Prop :=
  (s.russian = 100 ∧ s.physics = 100) ∨
  (s.russian = 100 ∧ s.math = 100) ∨
  (s.physics = 100 ∧ s.math = 100)

/-- The main theorem to be proved -/
theorem cannot_achieve_multiple_100s (s : Scores) 
  (h : initial_score_relationship s) : 
  ¬ ∃ (ops : List Operation), 
    let final_scores := ops.foldl apply_operation s
    ¬ exceeds_100 final_scores ∧ more_than_one_100 final_scores :=
sorry


end cannot_achieve_multiple_100s_l1121_112187


namespace smallest_proportional_part_l1121_112150

theorem smallest_proportional_part (total : ℕ) (parts : List ℕ) : 
  total = 360 → 
  parts = [5, 7, 4, 8] → 
  (parts.sum : ℚ) > 0 → 
  let proportional_parts := parts.map (λ p => (p : ℚ) * total / parts.sum)
  List.minimum proportional_parts = some 60 := by
sorry

end smallest_proportional_part_l1121_112150


namespace lcm_problem_l1121_112115

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 60 := by sorry

end lcm_problem_l1121_112115


namespace line_intersects_circle_l1121_112154

/-- Given a point M(x₀, y₀) outside the circle x² + y² = 2,
    prove that the line x₀x + y₀y = 2 intersects the circle. -/
theorem line_intersects_circle (x₀ y₀ : ℝ) (h : x₀^2 + y₀^2 > 2) :
  ∃ (x y : ℝ), x^2 + y^2 = 2 ∧ x₀*x + y₀*y = 2 := by
  sorry

end line_intersects_circle_l1121_112154


namespace stating_inspection_probability_theorem_l1121_112125

/-- Represents the total number of items -/
def total_items : ℕ := 5

/-- Represents the number of defective items -/
def defective_items : ℕ := 2

/-- Represents the number of good items -/
def good_items : ℕ := 3

/-- Represents the number of inspections after which we want to calculate the probability -/
def target_inspections : ℕ := 4

/-- Represents the probability of the inspection stopping after exactly the target number of inspections -/
noncomputable def inspection_probability : ℚ := 3/5

/-- 
Theorem stating that the probability of the inspection stopping after exactly 
the target number of inspections is equal to the calculated probability
-/
theorem inspection_probability_theorem : 
  let p := inspection_probability
  p = (1 : ℚ) - (defective_items.choose 2 / total_items.choose 2) - 
      ((good_items.choose 3 + defective_items.choose 1 * good_items.choose 1 * (total_items - 3).choose 1) / total_items.choose 3) :=
by sorry

end stating_inspection_probability_theorem_l1121_112125


namespace circle_center_and_sum_l1121_112186

/-- Given a circle described by the equation x^2 + y^2 = 4x - 2y + 10,
    prove that its center is at (2, -1) and the sum of the center's coordinates is 1. -/
theorem circle_center_and_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 2*y + 10) → 
  (∃ (center_x center_y : ℝ), 
    center_x = 2 ∧ 
    center_y = -1 ∧ 
    (x - center_x)^2 + (y - center_y)^2 = 15 ∧
    center_x + center_y = 1) := by
  sorry


end circle_center_and_sum_l1121_112186


namespace correct_equation_by_moving_digit_l1121_112175

theorem correct_equation_by_moving_digit : ∃ (a b c : ℕ), 
  (101 = 10^2 - 1 → False) ∧ 
  (101 = a * 10^2 + b * 10 + c - 1) ∧
  (a = 1 ∧ b = 0 ∧ c = 2) :=
by sorry

end correct_equation_by_moving_digit_l1121_112175


namespace ticket_revenue_calculation_l1121_112161

/-- Calculates the total revenue from ticket sales given the specified conditions -/
theorem ticket_revenue_calculation (total_tickets : ℕ) (student_price nonstudent_price : ℚ)
  (student_tickets : ℕ) (h1 : total_tickets = 821) (h2 : student_price = 2)
  (h3 : nonstudent_price = 3) (h4 : student_tickets = 530) :
  (student_tickets : ℚ) * student_price +
  ((total_tickets - student_tickets) : ℚ) * nonstudent_price = 1933 := by
  sorry

end ticket_revenue_calculation_l1121_112161


namespace sams_balloons_l1121_112174

theorem sams_balloons (fred_balloons : ℝ) (dan_destroyed : ℝ) (total_after : ℝ) 
  (h1 : fred_balloons = 10.0)
  (h2 : dan_destroyed = 16.0)
  (h3 : total_after = 40.0) :
  fred_balloons + (fred_balloons + dan_destroyed + total_after - fred_balloons) - dan_destroyed = total_after ∧
  fred_balloons + dan_destroyed + total_after - fred_balloons = 46.0 := by
  sorry

end sams_balloons_l1121_112174


namespace james_weekday_coffees_l1121_112190

/-- Represents the number of weekdays in a week -/
def weekdays : Nat := 5

/-- Represents the number of weekend days in a week -/
def weekend_days : Nat := 2

/-- Cost of a donut in cents -/
def donut_cost : Nat := 60

/-- Cost of a coffee in cents -/
def coffee_cost : Nat := 90

/-- Calculates the total cost for the week in cents -/
def total_cost (weekday_coffees : Nat) : Nat :=
  let weekday_donuts := weekdays - weekday_coffees
  let weekday_cost := weekday_coffees * coffee_cost + weekday_donuts * donut_cost
  let weekend_cost := weekend_days * (coffee_cost + donut_cost)
  weekday_cost + weekend_cost

theorem james_weekday_coffees :
  ∃ (weekday_coffees : Nat),
    weekday_coffees ≤ weekdays ∧
    (∃ (k : Nat), total_cost weekday_coffees = k * 100) ∧
    weekday_coffees = 2 := by
  sorry

end james_weekday_coffees_l1121_112190


namespace b_completion_time_l1121_112195

/-- Represents the time (in days) it takes for a worker to complete a job alone -/
structure WorkerTime where
  days : ℝ
  days_pos : days > 0

/-- Represents the share of earnings for a worker -/
structure EarningShare where
  amount : ℝ
  amount_pos : amount > 0

/-- Represents a job with multiple workers -/
structure Job where
  a : WorkerTime
  c : WorkerTime
  total_earnings : ℝ
  b_share : EarningShare
  total_earnings_pos : total_earnings > 0

theorem b_completion_time (job : Job) 
  (ha : job.a.days = 6)
  (hc : job.c.days = 12)
  (htotal : job.total_earnings = 1170)
  (hb_share : job.b_share.amount = 390) :
  ∃ (b : WorkerTime), b.days = 8 := by
  sorry

end b_completion_time_l1121_112195


namespace range_of_a_l1121_112192

def p (a : ℝ) : Prop := ∀ x, a^x > 1 ↔ x < 0

def q (a : ℝ) : Prop := ∀ x, ∃ y, y = Real.sqrt (a*x^2 - x + a)

theorem range_of_a :
  (∃ a, p a ∨ q a) ∧ (∀ a, ¬(p a ∧ q a)) →
  ∃ S : Set ℝ, S = {a | a ∈ (Set.Ioo 0 (1/2)) ∪ (Set.Ici 1)} :=
sorry

end range_of_a_l1121_112192


namespace brendan_rounds_won_all_l1121_112121

/-- The number of rounds where Brendan won all matches in a kickboxing competition -/
def rounds_won_all (total_matches_won : ℕ) (matches_per_full_round : ℕ) (last_round_matches : ℕ) : ℕ :=
  ((total_matches_won - (last_round_matches / 2)) / matches_per_full_round)

/-- Theorem stating that Brendan won all matches in 2 rounds -/
theorem brendan_rounds_won_all :
  rounds_won_all 14 6 4 = 2 := by
  sorry

end brendan_rounds_won_all_l1121_112121


namespace f_2017_value_l1121_112153

theorem f_2017_value (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - x * (deriv f 0) - 1) :
  f 2017 = 2016 * 2018 := by
  sorry

end f_2017_value_l1121_112153


namespace purchase_cost_l1121_112101

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 4

/-- The cost of a soda in dollars -/
def soda_cost : ℚ := 3/2

/-- The discount applied when purchasing at least 10 sandwiches -/
def bulk_discount : ℚ := 5

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 10

/-- The number of sodas purchased -/
def num_sodas : ℕ := 6

/-- The total cost of the purchase -/
def total_cost : ℚ := 
  (num_sandwiches * sandwich_cost - bulk_discount) + (num_sodas * soda_cost)

theorem purchase_cost : total_cost = 44 := by
  sorry

end purchase_cost_l1121_112101


namespace bouquet_cost_is_45_l1121_112167

/-- The cost of a bouquet consisting of two dozens of red roses and 3 sunflowers -/
def bouquet_cost (rose_price sunflower_price : ℚ) : ℚ :=
  (24 * rose_price) + (3 * sunflower_price)

/-- Theorem stating that the cost of the bouquet with given prices is $45 -/
theorem bouquet_cost_is_45 :
  bouquet_cost (3/2) 3 = 45 := by
  sorry

end bouquet_cost_is_45_l1121_112167


namespace max_cab_value_l1121_112144

/-- Represents a two-digit number AB --/
def TwoDigitNumber (a b : Nat) : Prop :=
  10 ≤ 10 * a + b ∧ 10 * a + b < 100

/-- Represents a three-digit number CAB --/
def ThreeDigitNumber (c a b : Nat) : Prop :=
  100 ≤ 100 * c + 10 * a + b ∧ 100 * c + 10 * a + b < 1000

/-- The main theorem statement --/
theorem max_cab_value :
  ∀ a b c : Nat,
  a < 10 → b < 10 → c < 10 →
  TwoDigitNumber a b →
  ThreeDigitNumber c a b →
  (10 * a + b) * a = 100 * c + 10 * a + b →
  100 * c + 10 * a + b ≤ 895 :=
by sorry

end max_cab_value_l1121_112144


namespace john_stereo_trade_in_l1121_112179

/-- The cost of John's old stereo system -/
def old_system_cost : ℝ := 250

/-- The trade-in value as a percentage of the old system's cost -/
def trade_in_percentage : ℝ := 0.80

/-- The cost of the new stereo system before discount -/
def new_system_cost : ℝ := 600

/-- The discount percentage on the new system -/
def discount_percentage : ℝ := 0.25

/-- The amount John spent out of pocket -/
def out_of_pocket : ℝ := 250

theorem john_stereo_trade_in :
  old_system_cost * trade_in_percentage + out_of_pocket =
  new_system_cost * (1 - discount_percentage) :=
by sorry

end john_stereo_trade_in_l1121_112179


namespace f_monotone_implies_a_range_l1121_112128

/-- A piecewise function f depending on parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 2*(a-1)*x else (8-a)*x + 4

/-- f is monotonically increasing on ℝ -/
def monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Theorem stating that if f is monotonically increasing, then 2 ≤ a ≤ 5 -/
theorem f_monotone_implies_a_range (a : ℝ) :
  monotone_increasing (f a) → 2 ≤ a ∧ a ≤ 5 :=
by
  sorry

#check f_monotone_implies_a_range

end f_monotone_implies_a_range_l1121_112128


namespace completing_square_sum_l1121_112119

theorem completing_square_sum (d e f : ℤ) : 
  d > 0 ∧ 
  (∀ x : ℝ, 25 * x^2 + 30 * x - 24 = 0 ↔ (d * x + e)^2 = f) → 
  d + e + f = 41 :=
by sorry

end completing_square_sum_l1121_112119


namespace sugar_salt_diff_is_one_l1121_112151

/-- A baking recipe with specified ingredient amounts -/
structure Recipe where
  flour : ℕ
  sugar : ℕ
  salt : ℕ

/-- The difference in cups between sugar and salt in a recipe -/
def sugar_salt_difference (r : Recipe) : ℤ :=
  r.sugar - r.salt

/-- Theorem: The difference between sugar and salt in the given recipe is 1 cup -/
theorem sugar_salt_diff_is_one (r : Recipe) (h : r.flour = 6 ∧ r.sugar = 8 ∧ r.salt = 7) : 
  sugar_salt_difference r = 1 := by
  sorry

#eval sugar_salt_difference {flour := 6, sugar := 8, salt := 7}

end sugar_salt_diff_is_one_l1121_112151


namespace set_equality_solution_l1121_112111

theorem set_equality_solution (x y : ℝ) : 
  ({x, y, x + y} : Set ℝ) = ({0, x^2, x*y} : Set ℝ) →
  ((x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1)) :=
by sorry

end set_equality_solution_l1121_112111


namespace third_difference_zero_implies_quadratic_l1121_112138

/-- A function from integers to real numbers -/
def IntFunction := ℤ → ℝ

/-- The third difference of a function -/
def thirdDifference (f : IntFunction) : IntFunction :=
  fun n => f (n + 3) - 3 * f (n + 2) + 3 * f (n + 1) - f n

/-- A function is quadratic if it can be expressed as a*n^2 + b*n + c for some real a, b, c -/
def isQuadratic (f : IntFunction) : Prop :=
  ∃ a b c : ℝ, ∀ n : ℤ, f n = a * n^2 + b * n + c

theorem third_difference_zero_implies_quadratic (f : IntFunction) 
  (h : ∀ n : ℤ, thirdDifference f n = 0) : 
  isQuadratic f := by
  sorry

end third_difference_zero_implies_quadratic_l1121_112138


namespace race_speeds_l1121_112169

theorem race_speeds (x : ℝ) (h : x > 0) : 
  ∃ (a b : ℝ),
    (1000 = a * x) ∧ 
    (1000 - 167 = b * x) ∧
    (a = 1000 / x) ∧ 
    (b = 833 / x) := by
  sorry

end race_speeds_l1121_112169


namespace geometric_mean_of_4_and_9_l1121_112170

theorem geometric_mean_of_4_and_9 :
  ∃ x : ℝ, x^2 = 4 * 9 ∧ (x = 6 ∨ x = -6) := by
  sorry

end geometric_mean_of_4_and_9_l1121_112170


namespace irrational_equality_l1121_112123

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem irrational_equality (α β : ℝ) (h_irrational_α : Irrational α) (h_irrational_β : Irrational β) 
  (h_equality : ∀ x : ℝ, x > 0 → floor (α * floor (β * x)) = floor (β * floor (α * x))) :
  α = β :=
sorry

end irrational_equality_l1121_112123


namespace cathys_money_ratio_is_two_to_one_l1121_112191

/-- The ratio of the amount Cathy's mom sent her to the amount her dad sent her -/
def cathys_money_ratio (initial_amount dad_amount mom_amount final_amount : ℚ) : ℚ :=
  mom_amount / dad_amount

/-- Proves that the ratio of the amount Cathy's mom sent her to the amount her dad sent her is 2:1 -/
theorem cathys_money_ratio_is_two_to_one 
  (initial_amount : ℚ) 
  (dad_amount : ℚ) 
  (mom_amount : ℚ) 
  (final_amount : ℚ) 
  (h1 : initial_amount = 12)
  (h2 : dad_amount = 25)
  (h3 : final_amount = 87)
  (h4 : initial_amount + dad_amount + mom_amount = final_amount) :
  cathys_money_ratio initial_amount dad_amount mom_amount final_amount = 2 := by
sorry

#eval cathys_money_ratio 12 25 50 87

end cathys_money_ratio_is_two_to_one_l1121_112191


namespace salt_fraction_in_solution_l1121_112162

theorem salt_fraction_in_solution (salt_weight water_weight : ℚ) :
  salt_weight = 6 → water_weight = 30 →
  salt_weight / (salt_weight + water_weight) = 1 / 6 := by
  sorry

end salt_fraction_in_solution_l1121_112162


namespace expression_evaluation_l1121_112147

theorem expression_evaluation : 
  4 * Real.sin (60 * π / 180) - abs (-2) - Real.sqrt 12 + (-1) ^ 2016 = -1 := by
  sorry

end expression_evaluation_l1121_112147


namespace polygon_area_is_12_l1121_112173

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The polygon defined by the given points -/
def polygon : List Point := [
  ⟨0, 0⟩, ⟨4, 0⟩, ⟨4, 4⟩, ⟨2, 4⟩, ⟨2, 2⟩, ⟨0, 2⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vertices : List Point) : ℝ := sorry

/-- Theorem: The area of the given polygon is 12 square units -/
theorem polygon_area_is_12 : polygonArea polygon = 12 := by sorry

end polygon_area_is_12_l1121_112173


namespace polynomial_sum_at_zero_and_four_l1121_112145

/-- Given a polynomial f(x) = x⁴ + ax³ + bx² + cx + d with zeros 1, 2, and 3,
    prove that f(0) + f(4) = 24 -/
theorem polynomial_sum_at_zero_and_four 
  (f : ℝ → ℝ) 
  (a b c d : ℝ) 
  (h1 : ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d) 
  (h2 : f 1 = 0) 
  (h3 : f 2 = 0) 
  (h4 : f 3 = 0) : 
  f 0 + f 4 = 24 := by
  sorry

end polynomial_sum_at_zero_and_four_l1121_112145


namespace supervisors_per_bus_l1121_112132

theorem supervisors_per_bus (total_buses : ℕ) (total_supervisors : ℕ) 
  (h1 : total_buses = 7) 
  (h2 : total_supervisors = 21) : 
  total_supervisors / total_buses = 3 := by
  sorry

end supervisors_per_bus_l1121_112132


namespace speed_ratio_of_perpendicular_paths_l1121_112118

/-- The ratio of speeds of two objects moving along perpendicular paths -/
theorem speed_ratio_of_perpendicular_paths
  (vA vB : ℝ) -- Speeds of objects A and B
  (h1 : vA > 0 ∧ vB > 0) -- Both speeds are positive
  (h2 : ∃ t1 : ℝ, t1 > 0 ∧ t1 * vA = |700 - t1 * vB|) -- Equidistant at time t1
  (h3 : ∃ t2 : ℝ, t2 > t1 ∧ t2 * vA = |700 - t2 * vB|) -- Equidistant at time t2 > t1
  : vA / vB = 6 / 7 :=
sorry

end speed_ratio_of_perpendicular_paths_l1121_112118


namespace final_coin_count_l1121_112180

def coin_collection (initial : ℕ) (years : ℕ) : ℕ :=
  let year1 := initial * 2
  let year2 := year1 + 12 * 3
  let year3 := year2 + 12 / 3
  let year4 := year3 - year3 / 4
  year4

theorem final_coin_count : coin_collection 50 4 = 105 := by
  sorry

end final_coin_count_l1121_112180


namespace restaurant_group_kids_l1121_112134

/-- Proves that in a group of 12 people, where adult meals cost $3 each and kids eat free,
    if the total cost is $15, then the number of kids in the group is 7. -/
theorem restaurant_group_kids (total_people : ℕ) (adult_meal_cost : ℕ) (total_cost : ℕ) 
  (h1 : total_people = 12)
  (h2 : adult_meal_cost = 3)
  (h3 : total_cost = 15) :
  total_people - (total_cost / adult_meal_cost) = 7 :=
by sorry

end restaurant_group_kids_l1121_112134


namespace union_of_M_and_N_l1121_112104

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | -3 < x ∧ x < 2}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | x > -3} := by
  sorry

end union_of_M_and_N_l1121_112104


namespace alice_unanswered_questions_l1121_112176

/-- Represents a scoring system for a test --/
structure ScoringSystem where
  startPoints : ℤ
  correctPoints : ℤ
  wrongPoints : ℤ
  unansweredPoints : ℤ

/-- Calculates the score based on a scoring system and the number of correct, wrong, and unanswered questions --/
def calculateScore (system : ScoringSystem) (correct wrong unanswered : ℤ) : ℤ :=
  system.startPoints + system.correctPoints * correct + system.wrongPoints * wrong + system.unansweredPoints * unanswered

theorem alice_unanswered_questions : ∃ (correct wrong unanswered : ℤ),
  let newSystem : ScoringSystem := ⟨0, 6, 0, 3⟩
  let oldSystem : ScoringSystem := ⟨50, 5, -2, 0⟩
  let hypotheticalSystem : ScoringSystem := ⟨40, 7, -1, -1⟩
  correct + wrong + unanswered = 25 ∧
  calculateScore newSystem correct wrong unanswered = 130 ∧
  calculateScore oldSystem correct wrong unanswered = 100 ∧
  calculateScore hypotheticalSystem correct wrong unanswered = 120 ∧
  unanswered = 20 := by
  sorry

#check alice_unanswered_questions

end alice_unanswered_questions_l1121_112176


namespace three_digit_prime_with_special_property_l1121_112108

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def all_digits_different (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

def last_digit_is_sum_of_first_two (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones = hundreds + tens

theorem three_digit_prime_with_special_property (p : ℕ) 
  (h_prime : Nat.Prime p)
  (h_three_digit : is_three_digit p)
  (h_different : all_digits_different p)
  (h_sum : last_digit_is_sum_of_first_two p) :
  p % 10 = 7 := by
  sorry

end three_digit_prime_with_special_property_l1121_112108


namespace min_value_expression_l1121_112124

theorem min_value_expression (x : ℝ) : 
  (∀ y : ℝ, (15 - y) * (8 - y) * (15 + y) * (8 + y) - 200 ≥ (15 - x) * (8 - x) * (15 + x) * (8 + x) - 200) → 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) - 200 = -6680.25 :=
by
  sorry

end min_value_expression_l1121_112124


namespace area_of_extended_quadrilateral_l1121_112160

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (EF FG GH HE : ℝ)
  (area : ℝ)

-- Define the extended quadrilateral E'F'G'H'
structure ExtendedQuadrilateral :=
  (base : Quadrilateral)
  (EE' FF' GG' HH' : ℝ)

-- Define our specific quadrilateral
def EFGH : Quadrilateral :=
  { EF := 5
  , FG := 10
  , GH := 9
  , HE := 7
  , area := 12 }

-- Define our specific extended quadrilateral
def EFGH_extended : ExtendedQuadrilateral :=
  { base := EFGH
  , EE' := 7
  , FF' := 5
  , GG' := 10
  , HH' := 9 }

-- State the theorem
theorem area_of_extended_quadrilateral :
  (EFGH_extended.base.area + 
   EFGH_extended.base.EF * EFGH_extended.FF' +
   EFGH_extended.base.FG * EFGH_extended.GG' +
   EFGH_extended.base.GH * EFGH_extended.HH' +
   EFGH_extended.base.HE * EFGH_extended.EE') = 36 := by
  sorry

end area_of_extended_quadrilateral_l1121_112160


namespace smallest_divisible_page_number_l1121_112152

theorem smallest_divisible_page_number : 
  (∀ n : ℕ, n > 0 ∧ 4 ∣ n ∧ 13 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n ∧ 17 ∣ n → n ≥ 68068) ∧ 
  (4 ∣ 68068) ∧ (13 ∣ 68068) ∧ (7 ∣ 68068) ∧ (11 ∣ 68068) ∧ (17 ∣ 68068) := by
  sorry

end smallest_divisible_page_number_l1121_112152


namespace geometric_sequence_property_l1121_112130

/-- A geometric sequence with the given properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a)
  (h_a5 : a 5 = -9)
  (h_a8 : a 8 = 6) :
  a 11 = -4 := by
sorry

end geometric_sequence_property_l1121_112130


namespace equal_intercept_line_equation_l1121_112165

/-- A line passing through (2, 3) with equal x and y intercepts -/
def EqualInterceptLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (a : ℝ), (p.1 / a + p.2 / a = 1 ∧ a ≠ 0) ∨ (p.1 = 2 ∧ p.2 = 3) ∨ (p.1 = 0 ∧ p.2 = 0)}

theorem equal_intercept_line_equation :
  EqualInterceptLine = {p : ℝ × ℝ | p.1 + p.2 - 5 = 0} ∪ {p : ℝ × ℝ | 3 * p.1 - 2 * p.2 = 0} :=
sorry

end equal_intercept_line_equation_l1121_112165


namespace inconsistent_school_population_l1121_112198

theorem inconsistent_school_population (total_students : Real) 
  (boy_percentage : Real) (representative_students : Nat) : 
  total_students = 113.38934190276818 → 
  boy_percentage = 0.70 → 
  representative_students = 90 → 
  (representative_students : Real) / (total_students * boy_percentage) > 1 := by
  sorry

end inconsistent_school_population_l1121_112198


namespace two_a_minus_three_b_value_l1121_112102

theorem two_a_minus_three_b_value (a b : ℝ) (h1 : |a| = 2) (h2 : |b| = 3) (h3 : b < a) :
  2 * a - 3 * b = 13 ∨ 2 * a - 3 * b = 5 :=
by sorry

end two_a_minus_three_b_value_l1121_112102


namespace distribute_balls_result_l1121_112143

/-- The number of ways to distribute balls to students -/
def distribute_balls (red black white : ℕ) : ℕ :=
  let min_boys := 2
  let min_girl := 3
  let remaining_red := red - (2 * min_boys + min_girl)
  let remaining_black := black - (2 * min_boys + min_girl)
  let remaining_white := white - (2 * min_boys + min_girl)
  (Nat.choose (remaining_red + 2) 2) *
  (Nat.choose (remaining_black + 2) 2) *
  (Nat.choose (remaining_white + 2) 2)

/-- Theorem stating the number of ways to distribute the balls -/
theorem distribute_balls_result : distribute_balls 10 15 20 = 47250 := by
  sorry

end distribute_balls_result_l1121_112143


namespace unique_quadratic_solution_l1121_112183

/-- 
Given a quadratic equation bx^2 + 6x + d = 0 with exactly one solution,
where b + d = 7 and b < d, prove that b = (7 - √13) / 2 and d = (7 + √13) / 2
-/
theorem unique_quadratic_solution (b d : ℝ) : 
  (∃! x, b * x^2 + 6 * x + d = 0) →
  b + d = 7 →
  b < d →
  b = (7 - Real.sqrt 13) / 2 ∧ d = (7 + Real.sqrt 13) / 2 := by
  sorry

end unique_quadratic_solution_l1121_112183


namespace sanda_minutes_per_day_l1121_112189

/-- The number of minutes Javier exercised per day -/
def javier_minutes_per_day : ℕ := 50

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of minutes Javier and Sanda exercised -/
def total_minutes : ℕ := 620

/-- The number of days Sanda exercised -/
def sanda_exercise_days : ℕ := 3

/-- Theorem stating that Sanda exercised 90 minutes each day -/
theorem sanda_minutes_per_day :
  (total_minutes - javier_minutes_per_day * days_in_week) / sanda_exercise_days = 90 := by
  sorry

end sanda_minutes_per_day_l1121_112189


namespace inequality_solution_l1121_112194

theorem inequality_solution (x : ℝ) :
  (x - 1) * (x - 4) * (x - 5) * (x - 7) / ((x - 3) * (x - 6) * (x - 8) * (x - 9)) > 0 →
  |x - 2| ≥ 1 →
  x ∈ Set.Ioo 3 4 ∪ Set.Ioo 6 7 ∪ Set.Ioo 8 9 :=
by sorry

end inequality_solution_l1121_112194


namespace unique_quadratic_function_l1121_112112

/-- A quadratic function is a function of the form f(x) = ax^2 + bx + c where a ≠ 0 -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The property that f(x^2) = f(f(x)) = (f(x))^2 for all x -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x^2) = f (f x) ∧ f (x^2) = (f x)^2

theorem unique_quadratic_function :
  ∃! f : ℝ → ℝ, QuadraticFunction f ∧ SatisfiesCondition f ∧ ∀ x, f x = x^2 :=
by
  sorry

#check unique_quadratic_function

end unique_quadratic_function_l1121_112112


namespace product_xyz_equals_one_l1121_112110

theorem product_xyz_equals_one 
  (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 2) 
  (h3 : z + 1/x = 2) : 
  x * y * z = 1 := by
sorry

end product_xyz_equals_one_l1121_112110


namespace triangle_centroid_distances_l1121_112149

/-- Given a triangle DEF with centroid G, prove that if the sum of squared distances
    from G to the vertices is 72, then the sum of squared side lengths is 216. -/
theorem triangle_centroid_distances (D E F G : ℝ × ℝ) : 
  G = ((D.1 + E.1 + F.1) / 3, (D.2 + E.2 + F.2) / 3) →  -- G is the centroid
  (G.1 - D.1)^2 + (G.2 - D.2)^2 +    -- GD^2
  (G.1 - E.1)^2 + (G.2 - E.2)^2 +    -- GE^2
  (G.1 - F.1)^2 + (G.2 - F.2)^2 = 72 →  -- GF^2
  (D.1 - E.1)^2 + (D.2 - E.2)^2 +    -- DE^2
  (D.1 - F.1)^2 + (D.2 - F.2)^2 +    -- DF^2
  (E.1 - F.1)^2 + (E.2 - F.2)^2 = 216  -- EF^2
:= by sorry

end triangle_centroid_distances_l1121_112149


namespace new_person_weight_bus_weight_problem_l1121_112146

theorem new_person_weight (initial_count : ℕ) (initial_average : ℝ) (weight_decrease : ℝ) : ℝ :=
  let total_weight := initial_count * initial_average
  let new_count := initial_count + 1
  let new_average := initial_average - weight_decrease
  let new_total_weight := new_count * new_average
  new_total_weight - total_weight

theorem bus_weight_problem :
  new_person_weight 30 102 2 = 40 := by
  sorry

end new_person_weight_bus_weight_problem_l1121_112146


namespace ratio_x_to_y_l1121_112129

theorem ratio_x_to_y (x y : ℝ) (h : y = x * (1 - 0.6666666666666666)) :
  x / y = 3 := by
  sorry

end ratio_x_to_y_l1121_112129


namespace cylinder_cut_surface_area_l1121_112126

/-- Represents a right circular cylinder -/
structure RightCircularCylinder where
  radius : ℝ
  height : ℝ

/-- Represents the area of a flat surface created by cutting the cylinder -/
def cutSurfaceArea (c : RightCircularCylinder) (arcAngle : ℝ) : ℝ :=
  sorry

theorem cylinder_cut_surface_area :
  let c : RightCircularCylinder := { radius := 8, height := 10 }
  let arcAngle : ℝ := π / 2  -- 90 degrees in radians
  cutSurfaceArea c arcAngle = 40 * π - 40 * Real.sqrt 2 := by
  sorry

end cylinder_cut_surface_area_l1121_112126


namespace intersection_of_A_and_B_l1121_112199

def A : Set ℕ := {0,1,2,3,4,6,7}
def B : Set ℕ := {1,2,4,8,0}

theorem intersection_of_A_and_B : A ∩ B = {1,2,4,0} := by
  sorry

end intersection_of_A_and_B_l1121_112199


namespace percentage_of_B_grades_l1121_112105

def scores : List ℕ := [86, 73, 55, 98, 76, 93, 88, 72, 77, 62, 81, 79, 68, 82, 91]

def is_grade_B (score : ℕ) : Bool :=
  87 ≤ score ∧ score ≤ 93

def count_grade_B (scores : List ℕ) : ℕ :=
  (scores.filter is_grade_B).length

theorem percentage_of_B_grades :
  (count_grade_B scores : ℚ) / (scores.length : ℚ) * 100 = 20 := by
  sorry

end percentage_of_B_grades_l1121_112105


namespace apollo_pays_168_apples_l1121_112197

/-- Represents the number of months in a year --/
def months_in_year : ℕ := 12

/-- Represents Hephaestus's charging rate for the first half of the year --/
def hephaestus_rate_first_half : ℕ := 3

/-- Represents Hephaestus's charging rate for the second half of the year --/
def hephaestus_rate_second_half : ℕ := 2 * hephaestus_rate_first_half

/-- Represents Athena's charging rate for the entire year --/
def athena_rate : ℕ := 5

/-- Represents Ares's charging rate for the first 9 months --/
def ares_rate_first_nine : ℕ := 4

/-- Represents Ares's charging rate for the last 3 months --/
def ares_rate_last_three : ℕ := 6

/-- Calculates the total number of golden apples Apollo pays for a year --/
def total_golden_apples : ℕ :=
  (hephaestus_rate_first_half * 6 + hephaestus_rate_second_half * 6) +
  (athena_rate * months_in_year) +
  (ares_rate_first_nine * 9 + ares_rate_last_three * 3)

/-- Theorem stating that the total number of golden apples Apollo pays is 168 --/
theorem apollo_pays_168_apples : total_golden_apples = 168 := by
  sorry

end apollo_pays_168_apples_l1121_112197


namespace cosine_derivative_at_pi_over_two_l1121_112139

theorem cosine_derivative_at_pi_over_two :
  deriv (fun x => Real.cos x) (π / 2) = -1 := by
  sorry

end cosine_derivative_at_pi_over_two_l1121_112139


namespace water_price_l1121_112107

/-- Given that six bottles of 2 liters of water cost $12, prove that the price of 1 liter of water is $1. -/
theorem water_price (bottles : ℕ) (liters_per_bottle : ℝ) (total_cost : ℝ) :
  bottles = 6 →
  liters_per_bottle = 2 →
  total_cost = 12 →
  total_cost / (bottles * liters_per_bottle) = 1 := by
  sorry

end water_price_l1121_112107
