import Mathlib

namespace f_2014_equals_zero_l1189_118928

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of f being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the periodicity property of f
def HasPeriodicity (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f x + f 2

-- Theorem statement
theorem f_2014_equals_zero 
  (h_even : IsEven f) 
  (h_periodicity : HasPeriodicity f) : 
  f 2014 = 0 := by
  sorry

end f_2014_equals_zero_l1189_118928


namespace range_of_m_l1189_118961

/-- The line x - 2y + 3 = 0 and the parabola y² = mx (m ≠ 0) have no points of intersection -/
def p (m : ℝ) : Prop :=
  ∀ x y : ℝ, x - 2*y + 3 = 0 → y^2 = m*x → m ≠ 0 → False

/-- The equation x²/(5-2m) + y²/m = 1 represents a hyperbola -/
def q (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2/(5-2*m) + y^2/m = 1 ∧ m*(5-2*m) < 0

theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) →
    m ≥ 3 ∨ m < 0 ∨ (0 < m ∧ m ≤ 5/2) :=
by sorry

end range_of_m_l1189_118961


namespace slope_relationship_l1189_118948

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Definition of point A₁ -/
def A₁ : ℝ × ℝ := (-2, 0)

/-- Definition of point A₂ -/
def A₂ : ℝ × ℝ := (2, 0)

/-- Definition of the line PQ -/
def line_PQ (x y : ℝ) : Prop := ∃ m : ℝ, x = m * y + 1/2

/-- Theorem stating the relationship between slopes -/
theorem slope_relationship (P Q : ℝ × ℝ) :
  ellipse_C P.1 P.2 →
  ellipse_C Q.1 Q.2 →
  line_PQ P.1 P.2 →
  line_PQ Q.1 Q.2 →
  P ≠ A₁ →
  P ≠ A₂ →
  Q ≠ A₁ →
  Q ≠ A₂ →
  (P.2 - A₁.2) / (P.1 - A₁.1) = 3/5 * (Q.2 - A₂.2) / (Q.1 - A₂.1) :=
sorry

end slope_relationship_l1189_118948


namespace gcd_factorial_eight_ten_l1189_118905

theorem gcd_factorial_eight_ten : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end gcd_factorial_eight_ten_l1189_118905


namespace geometric_sequence_theorem_l1189_118929

/-- A geometric sequence with given second and fifth terms -/
structure GeometricSequence where
  b₂ : ℝ
  b₅ : ℝ
  h₁ : b₂ = 24.5
  h₂ : b₅ = 196

/-- Properties of the geometric sequence -/
def GeometricSequence.properties (g : GeometricSequence) : Prop :=
  ∃ (b₁ r : ℝ),
    r > 0 ∧
    g.b₂ = b₁ * r ∧
    g.b₅ = b₁ * r^4 ∧
    let b₃ := b₁ * r^2
    let S₄ := b₁ * (r^4 - 1) / (r - 1)
    b₃ = 49 ∧ S₄ = 183.75

/-- Main theorem: The third term is 49 and the sum of first four terms is 183.75 -/
theorem geometric_sequence_theorem (g : GeometricSequence) :
  g.properties := by sorry

end geometric_sequence_theorem_l1189_118929


namespace inequality_condition_sum_l1189_118972

theorem inequality_condition_sum (a₁ a₂ : ℝ) : 
  (∀ x : ℝ, (x^2 - a₁*x + 2) / (x^2 - x + 1) < 3) ∧
  (∀ x : ℝ, (x^2 - a₂*x + 2) / (x^2 - x + 1) < 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, (x^2 - a*x + 2) / (x^2 - x + 1) < 3) → a > a₁ ∧ a < a₂) →
  a₁ = 3 - 2*Real.sqrt 2 ∧ a₂ = 3 + 2*Real.sqrt 2 ∧ a₁ + a₂ = 6 :=
by sorry

end inequality_condition_sum_l1189_118972


namespace quadratic_root_sum_squares_l1189_118930

theorem quadratic_root_sum_squares (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  (∃! x : ℝ, x^2 + a*x + b = 0 ∧ x^2 + b*x + c = 0) ∧
  (∃! y : ℝ, y^2 + b*y + c = 0 ∧ y^2 + c*y + a = 0) ∧
  (∃! z : ℝ, z^2 + c*z + a = 0 ∧ z^2 + a*z + b = 0) →
  a^2 + b^2 + c^2 = 6 := by
sorry

end quadratic_root_sum_squares_l1189_118930


namespace quadratic_expression_value_l1189_118974

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 2*x + y = 4) 
  (eq2 : x + 2*y = 5) : 
  5*x^2 + 8*x*y + 5*y^2 = 41 := by
sorry

end quadratic_expression_value_l1189_118974


namespace range_of_a_l1189_118998

-- Define set A
def A : Set ℝ := {x | x^2 - x < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := Set.Ioo 0 a

-- Theorem statement
theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : A ⊆ B a) : a ≥ 1 := by
  sorry

end range_of_a_l1189_118998


namespace gcd_1978_2017_l1189_118992

theorem gcd_1978_2017 : Nat.gcd 1978 2017 = 1 := by sorry

end gcd_1978_2017_l1189_118992


namespace initial_stops_l1189_118946

/-- Represents the number of stops made by a delivery driver -/
structure DeliveryStops where
  total : Nat
  after_initial : Nat
  initial : Nat

/-- Theorem stating the number of initial stops given the total and after-initial stops -/
theorem initial_stops (d : DeliveryStops) 
  (h1 : d.total = 7) 
  (h2 : d.after_initial = 4) 
  (h3 : d.total = d.initial + d.after_initial) : 
  d.initial = 3 := by
  sorry

end initial_stops_l1189_118946


namespace snail_final_position_l1189_118993

/-- Represents the direction the snail is facing -/
inductive Direction
  | Up
  | Right
  | Down
  | Left

/-- Represents a position on the grid -/
structure Position where
  row : Nat
  col : Nat

/-- Represents the state of the snail -/
structure SnailState where
  pos : Position
  dir : Direction
  visited : Set Position

/-- The grid dimensions -/
def gridWidth : Nat := 300
def gridHeight : Nat := 50

/-- Check if a position is within the grid -/
def isValidPosition (p : Position) : Bool :=
  p.row >= 1 && p.row <= gridHeight && p.col >= 1 && p.col <= gridWidth

/-- Move the snail according to the rules -/
def moveSnail (state : SnailState) : SnailState :=
  sorry -- Implementation of snail movement logic

/-- The main theorem stating the final position of the snail -/
theorem snail_final_position :
  ∃ (finalState : SnailState),
    (∀ (p : Position), isValidPosition p → p ∈ finalState.visited) ∧
    finalState.pos = Position.mk 25 26 := by
  sorry


end snail_final_position_l1189_118993


namespace no_natural_square_diff_2018_l1189_118950

theorem no_natural_square_diff_2018 : ¬∃ (a b : ℕ), a^2 - b^2 = 2018 := by
  sorry

end no_natural_square_diff_2018_l1189_118950


namespace max_smoothie_servings_l1189_118991

/-- Represents the recipe for 8 servings -/
structure Recipe where
  bananas : ℕ
  strawberries : ℕ
  yogurt : ℕ
  milk : ℕ

/-- Represents Sarah's available ingredients -/
structure Ingredients where
  bananas : ℕ
  strawberries : ℕ
  yogurt : ℕ
  milk : ℕ

/-- Calculates the maximum number of servings possible for a given ingredient -/
def max_servings (recipe_amount : ℕ) (available_amount : ℕ) : ℚ :=
  (available_amount : ℚ) / (recipe_amount : ℚ) * 8

/-- Theorem stating the maximum number of servings Sarah can make -/
theorem max_smoothie_servings (recipe : Recipe) (sarah_ingredients : Ingredients) :
  recipe.bananas = 3 ∧ 
  recipe.strawberries = 2 ∧ 
  recipe.yogurt = 1 ∧ 
  recipe.milk = 4 ∧
  sarah_ingredients.bananas = 10 ∧
  sarah_ingredients.strawberries = 5 ∧
  sarah_ingredients.yogurt = 3 ∧
  sarah_ingredients.milk = 10 →
  ⌊min 
    (min (max_servings recipe.bananas sarah_ingredients.bananas) (max_servings recipe.strawberries sarah_ingredients.strawberries))
    (min (max_servings recipe.yogurt sarah_ingredients.yogurt) (max_servings recipe.milk sarah_ingredients.milk))
  ⌋ = 20 := by
  sorry

end max_smoothie_servings_l1189_118991


namespace semicircle_perimeter_l1189_118941

/-- The perimeter of a semi-circle with radius 21.005164601010506 cm is 108.01915941002101 cm -/
theorem semicircle_perimeter : 
  let r : ℝ := 21.005164601010506
  (π * r + 2 * r) = 108.01915941002101 := by
sorry

end semicircle_perimeter_l1189_118941


namespace pencil_order_cost_l1189_118903

/-- Calculates the cost of pencils with a potential discount -/
def pencilCost (boxSize : ℕ) (boxPrice : ℚ) (discountThreshold : ℕ) (discountRate : ℚ) (quantity : ℕ) : ℚ :=
  let basePrice := (quantity : ℚ) * boxPrice / (boxSize : ℚ)
  if quantity > discountThreshold then
    basePrice * (1 - discountRate)
  else
    basePrice

theorem pencil_order_cost :
  pencilCost 200 40 1000 (1/10) 2400 = 432 :=
by sorry

end pencil_order_cost_l1189_118903


namespace fifth_term_is_fifteen_l1189_118940

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : a 2 + a 4 = 16
  first_term : a 1 = 1

/-- The fifth term of the arithmetic sequence is 15 -/
theorem fifth_term_is_fifteen (seq : ArithmeticSequence) : seq.a 5 = 15 := by
  sorry

end fifth_term_is_fifteen_l1189_118940


namespace gcd_8251_6105_l1189_118934

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  have h1 : 8251 = 6105 * 1 + 2146 := by sorry
  have h2 : 6105 = 2146 * 2 + 1813 := by sorry
  have h3 : 2146 = 1813 * 1 + 333 := by sorry
  have h4 : 333 = 148 * 2 + 37 := by sorry
  have h5 : 148 = 37 * 4 := by sorry
  sorry

end gcd_8251_6105_l1189_118934


namespace complex_number_sum_l1189_118906

theorem complex_number_sum (z : ℂ) 
  (h : 16 * Complex.abs z ^ 2 = 3 * Complex.abs (z + 1) ^ 2 + Complex.abs (z ^ 2 - 2) ^ 2 + 43) : 
  z + 8 / z = -2 := by
  sorry

end complex_number_sum_l1189_118906


namespace sqrt_product_equality_l1189_118915

theorem sqrt_product_equality (a : ℝ) (h : a ≥ 0) : Real.sqrt (2 * a) * Real.sqrt (3 * a) = a * Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l1189_118915


namespace sum_of_distinct_prime_factors_l1189_118987

theorem sum_of_distinct_prime_factors : 
  (let n := 7^6 - 2 * 7^4
   Finset.sum (Finset.filter (fun p => Nat.Prime p ∧ n % p = 0) (Finset.range (n + 1))) id) = 54 := by
  sorry

end sum_of_distinct_prime_factors_l1189_118987


namespace apples_left_theorem_l1189_118960

def num_baskets : ℕ := 11
def num_children : ℕ := 10
def initial_apples : ℕ := 1000

def apples_picked (basket : ℕ) : ℕ := basket * num_children

def total_apples_picked : ℕ := (List.range num_baskets).map (λ i => apples_picked (i + 1)) |>.sum

theorem apples_left_theorem :
  initial_apples - total_apples_picked = 340 := by sorry

end apples_left_theorem_l1189_118960


namespace unique_positive_solution_l1189_118966

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 5) / 10 = 5 / (x - 10) := by sorry

end unique_positive_solution_l1189_118966


namespace divisibility_by_37_l1189_118917

theorem divisibility_by_37 (a b c : ℕ) :
  let p := 100 * a + 10 * b + c
  let q := 100 * b + 10 * c + a
  let r := 100 * c + 10 * a + b
  37 ∣ p → (37 ∣ q ∧ 37 ∣ r) := by
sorry

end divisibility_by_37_l1189_118917


namespace z_in_second_quadrant_l1189_118921

-- Define the complex number z
def z : ℂ := sorry

-- State the given condition
axiom z_condition : (1 - Complex.I) * z = 2 * Complex.I

-- Define the second quadrant
def second_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im > 0

-- Theorem statement
theorem z_in_second_quadrant : second_quadrant z := by sorry

end z_in_second_quadrant_l1189_118921


namespace math_score_calculation_math_score_is_75_l1189_118911

theorem math_score_calculation (avg_four : ℝ) (drop : ℝ) : ℝ :=
  let total_four := 4 * avg_four
  let avg_five := avg_four - drop
  let total_five := 5 * avg_five
  total_five - total_four

theorem math_score_is_75 :
  math_score_calculation 90 3 = 75 := by
  sorry

end math_score_calculation_math_score_is_75_l1189_118911


namespace essay_introduction_length_l1189_118953

theorem essay_introduction_length 
  (total_words : ℕ) 
  (body_section_words : ℕ) 
  (body_section_count : ℕ) 
  (h1 : total_words = 5000)
  (h2 : body_section_words = 800)
  (h3 : body_section_count = 4) :
  ∃ (intro_words : ℕ),
    intro_words = 450 ∧ 
    total_words = intro_words + (body_section_count * body_section_words) + (3 * intro_words) :=
by sorry

end essay_introduction_length_l1189_118953


namespace system_solution_l1189_118978

theorem system_solution (a b c : ℝ) 
  (eq1 : b + c = 15 - 2*a)
  (eq2 : a + c = -10 - 4*b)
  (eq3 : a + b = 8 - 2*c) :
  3*a + 3*b + 3*c = 39/4 := by
  sorry

end system_solution_l1189_118978


namespace constant_value_c_l1189_118927

theorem constant_value_c (b c : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) → c = 7 := by
  sorry

end constant_value_c_l1189_118927


namespace eight_divided_by_repeating_third_l1189_118976

-- Define the repeating decimal 0.333...
def repeating_third : ℚ := 1/3

-- State the theorem
theorem eight_divided_by_repeating_third (h : repeating_third = 1/3) : 
  8 / repeating_third = 24 := by
  sorry

end eight_divided_by_repeating_third_l1189_118976


namespace order_combinations_l1189_118995

theorem order_combinations (num_drinks num_salads num_pizzas : ℕ) 
  (h1 : num_drinks = 3)
  (h2 : num_salads = 2)
  (h3 : num_pizzas = 5) :
  num_drinks * num_salads * num_pizzas = 30 := by
  sorry

end order_combinations_l1189_118995


namespace rectangle_side_length_l1189_118902

/-- Given two rectangles A and B, where A has sides a and b, and B has sides c and d,
    with the ratio of corresponding sides being 3/4, prove that when a = 3 and b = 6,
    the length of side d in Rectangle B is 8. -/
theorem rectangle_side_length (a b c d : ℝ) : 
  a = 3 → b = 6 → a / c = 3 / 4 → b / d = 3 / 4 → d = 8 := by
  sorry

end rectangle_side_length_l1189_118902


namespace max_intersections_circle_two_lines_triangle_l1189_118975

/-- Represents a circle in a plane -/
structure Circle where
  -- Definition of a circle (not needed for this proof)

/-- Represents a line in a plane -/
structure Line where
  -- Definition of a line (not needed for this proof)

/-- Represents a triangle in a plane -/
structure Triangle where
  -- Definition of a triangle (not needed for this proof)

/-- The maximum number of intersection points between a circle and a line -/
def maxCircleLineIntersections : ℕ := 2

/-- The maximum number of intersection points between two distinct lines -/
def maxTwoLinesIntersections : ℕ := 1

/-- The maximum number of intersection points between a circle and a triangle -/
def maxCircleTriangleIntersections : ℕ := 6

/-- The maximum number of intersection points between two lines and a triangle -/
def maxTwoLinesTriangleIntersections : ℕ := 6

/-- Theorem: The maximum number of intersection points between a circle, two distinct lines, and a triangle is 17 -/
theorem max_intersections_circle_two_lines_triangle :
  ∀ (c : Circle) (l1 l2 : Line) (t : Triangle),
    l1 ≠ l2 →
    (maxCircleLineIntersections * 2 + maxTwoLinesIntersections +
     maxCircleTriangleIntersections + maxTwoLinesTriangleIntersections) = 17 :=
by
  sorry


end max_intersections_circle_two_lines_triangle_l1189_118975


namespace square_number_plus_minus_five_is_square_l1189_118900

theorem square_number_plus_minus_five_is_square : ∃ (n : ℕ), 
  (∃ (a : ℕ), n = a^2) ∧ 
  (∃ (b : ℕ), n + 5 = b^2) ∧ 
  (∃ (c : ℕ), n - 5 = c^2) :=
by
  -- The proof goes here
  sorry

end square_number_plus_minus_five_is_square_l1189_118900


namespace twenty_five_percent_less_than_80_l1189_118988

theorem twenty_five_percent_less_than_80 : ∃ x : ℝ, x + (1/4 * x) = 0.75 * 80 ∧ x = 48 := by
  sorry

end twenty_five_percent_less_than_80_l1189_118988


namespace certain_number_existence_and_value_l1189_118983

theorem certain_number_existence_and_value :
  ∃ n : ℝ, 8 * n - (0.6 * 10) / 1.2 = 31.000000000000004 ∧ 
  ∃ ε > 0, |n - 4.5| < ε := by
  sorry

end certain_number_existence_and_value_l1189_118983


namespace polynomial_with_arithmetic_progression_roots_l1189_118955

theorem polynomial_with_arithmetic_progression_roots (j : ℝ) : 
  (∃ a b c d : ℝ, a < b ∧ b < c ∧ c < d ∧ 
    (∀ x : ℝ, x^4 + j*x^2 + 16*x + 64 = (x - a) * (x - b) * (x - c) * (x - d)) ∧
    b - a = c - b ∧ d - c = c - b) →
  j = -160/9 := by
sorry

end polynomial_with_arithmetic_progression_roots_l1189_118955


namespace rectangular_equation_chord_length_l1189_118904

-- Define the polar equation of curve C
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ * (Real.sin θ)^2 = 8 * Real.cos θ

-- Define the parametric equations of line l
def line_equation (t x y : ℝ) : Prop :=
  x = 2 + (1/2) * t ∧ y = (Real.sqrt 3 / 2) * t

-- Theorem for the rectangular equation of curve C
theorem rectangular_equation (x y : ℝ) :
  (∃ ρ θ, polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔
  y^2 = 8 * x :=
sorry

-- Theorem for the length of chord AB
theorem chord_length :
  ∃ t₁ t₂ x₁ y₁ x₂ y₂,
    line_equation t₁ x₁ y₁ ∧ line_equation t₂ x₂ y₂ ∧
    y₁^2 = 8 * x₁ ∧ y₂^2 = 8 * x₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 32/3 :=
sorry

end rectangular_equation_chord_length_l1189_118904


namespace elberta_amount_l1189_118967

-- Define the amounts for each person
def granny_smith : ℕ := 75
def anjou : ℕ := granny_smith / 4
def elberta : ℕ := anjou + 3

-- Theorem statement
theorem elberta_amount : elberta = 22 := by
  sorry

end elberta_amount_l1189_118967


namespace perimeter_is_20_l1189_118937

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define a line passing through the right focus
def line_through_right_focus (x y : ℝ) : Prop := sorry

-- Define points A and B on the ellipse and the line
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry

-- Assumption that A and B are on the ellipse and the line
axiom A_on_ellipse : ellipse point_A.1 point_A.2
axiom B_on_ellipse : ellipse point_B.1 point_B.2
axiom A_on_line : line_through_right_focus point_A.1 point_A.2
axiom B_on_line : line_through_right_focus point_B.1 point_B.2

-- Define the perimeter of triangle F₁AB
def perimeter_F1AB : ℝ := sorry

-- Theorem statement
theorem perimeter_is_20 : perimeter_F1AB = 20 := by sorry

end perimeter_is_20_l1189_118937


namespace quadratic_non_real_roots_l1189_118912

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, 2 * x^2 + b * x + 16 = 0 → x.im ≠ 0) ↔ b ∈ Set.Ioo (-8 * Real.sqrt 2) (8 * Real.sqrt 2) := by
  sorry

end quadratic_non_real_roots_l1189_118912


namespace coprime_20172019_l1189_118933

theorem coprime_20172019 :
  (Nat.gcd 20172019 20172017 = 1) ∧
  (Nat.gcd 20172019 20172018 = 1) ∧
  (Nat.gcd 20172019 20172020 = 1) ∧
  (Nat.gcd 20172019 20172021 = 1) :=
by sorry

end coprime_20172019_l1189_118933


namespace negation_of_every_constant_is_geometric_l1189_118923

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- A sequence is constant if all its terms are equal. -/
def IsConstant (s : Sequence) : Prop := ∀ n m : ℕ, s n = s m

/-- A sequence is geometric if the ratio between any two consecutive terms is constant and nonzero. -/
def IsGeometric (s : Sequence) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, s (n + 1) = r * s n

/-- The statement "Every constant sequence is a geometric sequence" -/
def EveryConstantIsGeometric : Prop :=
  ∀ s : Sequence, IsConstant s → IsGeometric s

/-- The negation of "Every constant sequence is a geometric sequence" -/
theorem negation_of_every_constant_is_geometric :
  ¬EveryConstantIsGeometric ↔ ∃ s : Sequence, IsConstant s ∧ ¬IsGeometric s :=
by
  sorry


end negation_of_every_constant_is_geometric_l1189_118923


namespace yellow_marbles_count_l1189_118982

theorem yellow_marbles_count (total : ℕ) (yellow green red blue : ℕ) : 
  total = 60 →
  green = yellow / 2 →
  red = blue →
  blue = total / 4 →
  total = yellow + green + red + blue →
  yellow = 20 := by sorry

end yellow_marbles_count_l1189_118982


namespace cube_vertices_l1189_118985

/-- A cube is a polyhedron with 6 faces and 12 edges -/
structure Cube where
  faces : ℕ
  edges : ℕ
  faces_eq : faces = 6
  edges_eq : edges = 12

/-- The number of vertices in a cube -/
def num_vertices (c : Cube) : ℕ := sorry

theorem cube_vertices (c : Cube) : num_vertices c = 8 := by sorry

end cube_vertices_l1189_118985


namespace hadley_books_added_l1189_118962

theorem hadley_books_added (initial_books : ℕ) (borrowed_by_lunch : ℕ) (borrowed_by_evening : ℕ) (remaining_books : ℕ) : 
  initial_books = 100 →
  borrowed_by_lunch = 50 →
  borrowed_by_evening = 30 →
  remaining_books = 60 →
  initial_books - borrowed_by_lunch + (remaining_books + borrowed_by_evening - (initial_books - borrowed_by_lunch)) = 100 :=
by
  sorry

#check hadley_books_added

end hadley_books_added_l1189_118962


namespace ram_price_increase_l1189_118954

theorem ram_price_increase (original_price current_price : ℝ) 
  (h1 : original_price = 50)
  (h2 : current_price = 52)
  (h3 : current_price = 0.8 * (original_price * (1 + increase_percentage / 100))) :
  increase_percentage = 30 := by
  sorry

end ram_price_increase_l1189_118954


namespace parking_lot_rows_parking_lot_rows_example_l1189_118926

/-- Given a parking lot with the following properties:
    - A car is 5th from the right and 4th from the left in a row
    - The parking lot has 10 floors
    - There are 1600 cars in total
    The number of rows on each floor is 20. -/
theorem parking_lot_rows (car_position_right : Nat) (car_position_left : Nat)
                         (num_floors : Nat) (total_cars : Nat) : Nat :=
  let cars_in_row := car_position_right + car_position_left - 1
  let cars_per_floor := total_cars / num_floors
  cars_per_floor / cars_in_row

#check parking_lot_rows 5 4 10 1600 = 20

/-- The parking_lot_rows theorem holds for the given values. -/
theorem parking_lot_rows_example : parking_lot_rows 5 4 10 1600 = 20 := by
  sorry

end parking_lot_rows_parking_lot_rows_example_l1189_118926


namespace candy_bar_difference_l1189_118963

theorem candy_bar_difference (lena nicole kevin : ℕ) : 
  lena = 23 →
  lena + 7 = 4 * kevin →
  nicole = kevin + 6 →
  lena - nicole = 10 :=
by
  sorry

end candy_bar_difference_l1189_118963


namespace mango_ratio_l1189_118949

def alexis_mangoes : ℕ := 60
def total_mangoes : ℕ := 75

def others_mangoes : ℕ := total_mangoes - alexis_mangoes

theorem mango_ratio : 
  (alexis_mangoes : ℚ) / (others_mangoes : ℚ) = 4 / 1 := by
  sorry

end mango_ratio_l1189_118949


namespace two_tails_probability_l1189_118986

theorem two_tails_probability (n : ℕ) (h : n = 5) : 
  (Nat.choose n 2 : ℚ) / (2^n : ℚ) = 10/32 := by
  sorry

end two_tails_probability_l1189_118986


namespace roots_of_unity_sum_one_l1189_118959

theorem roots_of_unity_sum_one (n : ℕ) (h : Even n) (h_pos : n > 0) :
  ∃ (z₁ z₂ z₃ : ℂ), (z₁^n = 1) ∧ (z₂^n = 1) ∧ (z₃^n = 1) ∧ (z₁ + z₂ + z₃ = 1) :=
sorry

end roots_of_unity_sum_one_l1189_118959


namespace max_handshakes_60_men_l1189_118901

/-- The maximum number of handshakes for n people without cyclic handshakes -/
def max_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 60 men, the maximum number of handshakes without cyclic handshakes is 1770 -/
theorem max_handshakes_60_men :
  max_handshakes 60 = 1770 := by
  sorry

end max_handshakes_60_men_l1189_118901


namespace greatest_three_digit_multiple_of_23_l1189_118924

theorem greatest_three_digit_multiple_of_23 : 
  (∀ n : ℕ, n ≤ 999 ∧ n ≥ 100 ∧ 23 ∣ n → n ≤ 989) ∧ 
  989 ≤ 999 ∧ 989 ≥ 100 ∧ 23 ∣ 989 := by
  sorry

end greatest_three_digit_multiple_of_23_l1189_118924


namespace angle_B_value_max_sum_sides_l1189_118932

-- Define the triangle
variable (A B C a b c : ℝ)

-- Define the conditions
variable (triangle_abc : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
variable (side_angle_relation : b * Real.cos C = (2 * a - c) * Real.cos B)

-- First theorem: B = π/3
theorem angle_B_value : B = π / 3 := by sorry

-- Second theorem: Maximum value of a + c when b = √3
theorem max_sum_sides (h_b : b = Real.sqrt 3) :
  ∃ (max : ℝ), max = 2 * Real.sqrt 3 ∧ 
  ∀ (a c : ℝ), a + c ≤ max := by sorry

end angle_B_value_max_sum_sides_l1189_118932


namespace robins_hair_length_l1189_118994

theorem robins_hair_length :
  ∀ (initial_length : ℝ),
    initial_length + 8 - 20 = 2 →
    initial_length = 14 :=
by sorry

end robins_hair_length_l1189_118994


namespace task_assignment_count_l1189_118939

def select_and_assign (n m : ℕ) : ℕ :=
  Nat.choose n m * Nat.choose m 2 * 2

theorem task_assignment_count : select_and_assign 10 4 = 2520 := by
  sorry

end task_assignment_count_l1189_118939


namespace ryan_reads_more_pages_l1189_118909

/-- Given that Ryan read 2100 pages in 7 days and his brother read 200 pages per day for 7 days,
    prove that Ryan read 100 more pages per day on average compared to his brother. -/
theorem ryan_reads_more_pages (ryan_total_pages : ℕ) (days : ℕ) (brother_daily_pages : ℕ)
    (h1 : ryan_total_pages = 2100)
    (h2 : days = 7)
    (h3 : brother_daily_pages = 200) :
    ryan_total_pages / days - brother_daily_pages = 100 := by
  sorry

end ryan_reads_more_pages_l1189_118909


namespace min_tangent_length_l1189_118956

/-- The minimum length of a tangent from a point on y = x + 1 to (x-3)^2 + y^2 = 1 is √7 -/
theorem min_tangent_length :
  let line := {p : ℝ × ℝ | p.2 = p.1 + 1}
  let circle := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 1}
  ∃ (min_length : ℝ),
    min_length = Real.sqrt 7 ∧
    ∀ (p : ℝ × ℝ) (t : ℝ × ℝ),
      p ∈ line → t ∈ circle →
      dist p t ≥ min_length :=
by sorry


end min_tangent_length_l1189_118956


namespace watermelon_duration_example_l1189_118914

/-- The number of weeks watermelons will last -/
def watermelon_duration (total : ℕ) (weekly_usage : ℕ) : ℕ :=
  total / weekly_usage

/-- Theorem: Given 30 watermelons and using 5 per week, they will last 6 weeks -/
theorem watermelon_duration_example : watermelon_duration 30 5 = 6 := by
  sorry

end watermelon_duration_example_l1189_118914


namespace friday_13th_more_frequent_l1189_118968

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a year in the Gregorian calendar -/
structure GregorianYear where
  year : Nat

/-- Determines if a given year is a leap year -/
def isLeapYear (y : GregorianYear) : Bool :=
  (y.year % 4 == 0 && y.year % 100 != 0) || y.year % 400 == 0

/-- Calculates the day of the week for the 13th of a given month in a given year -/
def dayOf13th (y : GregorianYear) (month : Nat) : DayOfWeek :=
  sorry

/-- Counts the frequency of each day of the week being the 13th over a 400-year cycle -/
def countDayOf13thIn400Years : DayOfWeek → Nat :=
  sorry

/-- Theorem: The 13th is more likely to be a Friday than any other day of the week -/
theorem friday_13th_more_frequent :
  ∀ d : DayOfWeek, d ≠ DayOfWeek.Friday →
    countDayOf13thIn400Years DayOfWeek.Friday > countDayOf13thIn400Years d :=
  sorry

end friday_13th_more_frequent_l1189_118968


namespace problem_one_problem_two_l1189_118969

-- Problem 1
theorem problem_one : 4 * Real.sqrt 5 + Real.sqrt 45 - Real.sqrt 8 + 4 * Real.sqrt 2 = 7 * Real.sqrt 5 + 2 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_two : (4 * Real.sqrt 3 - 6 * Real.sqrt (1/3) + 3 * Real.sqrt 12) / (2 * Real.sqrt 3) = 4 := by
  sorry

end problem_one_problem_two_l1189_118969


namespace school_teacher_student_ratio_l1189_118979

theorem school_teacher_student_ratio 
  (b c k h : ℕ) 
  (h_positive : h > 0) 
  (k_ge_two : k ≥ 2) 
  (c_ge_two : c ≥ 2) :
  (b : ℚ) / h = (c * (c - 1)) / (k * (k - 1)) := by
sorry

end school_teacher_student_ratio_l1189_118979


namespace volume_of_circumscribed_polyhedron_l1189_118958

-- Define a polyhedron circumscribed around a sphere
structure CircumscribedPolyhedron where
  -- R is the radius of the inscribed sphere
  R : ℝ
  -- S is the surface area of the polyhedron
  S : ℝ
  -- V is the volume of the polyhedron
  V : ℝ
  -- Ensure R and S are positive
  R_pos : 0 < R
  S_pos : 0 < S

-- Theorem statement
theorem volume_of_circumscribed_polyhedron (p : CircumscribedPolyhedron) :
  p.V = (p.S * p.R) / 3 := by
  sorry

end volume_of_circumscribed_polyhedron_l1189_118958


namespace falcons_minimum_wins_l1189_118980

/-- The minimum number of additional games the Falcons need to win -/
def min_additional_games : ℕ := 29

/-- The total number of initial games played -/
def initial_games : ℕ := 5

/-- The number of games won by the Falcons initially -/
def initial_falcons_wins : ℕ := 2

/-- The minimum winning percentage required for the Falcons -/
def min_winning_percentage : ℚ := 91 / 100

theorem falcons_minimum_wins (N : ℕ) :
  (N ≥ min_additional_games) →
  ((initial_falcons_wins + N : ℚ) / (initial_games + N)) ≥ min_winning_percentage ∧
  ∀ M : ℕ, M < min_additional_games →
    ((initial_falcons_wins + M : ℚ) / (initial_games + M)) < min_winning_percentage :=
by sorry

end falcons_minimum_wins_l1189_118980


namespace cube_roots_of_unity_l1189_118945

theorem cube_roots_of_unity :
  let z₁ : ℂ := 1
  let z₂ : ℂ := -1/2 + Complex.I * Real.sqrt 3 / 2
  let z₃ : ℂ := -1/2 - Complex.I * Real.sqrt 3 / 2
  ∀ z : ℂ, z^3 = 1 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃ := by
sorry

end cube_roots_of_unity_l1189_118945


namespace point_P_satisfies_conditions_l1189_118952

def P₁ : ℝ × ℝ := (1, 3)
def P₂ : ℝ × ℝ := (4, -6)
def P : ℝ × ℝ := (3, -3)

def vector (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, vector A C = (t • (vector A B).1, t • (vector A B).2)

theorem point_P_satisfies_conditions :
  collinear P₁ P₂ P ∧ vector P₁ P = (2 • (vector P P₂).1, 2 • (vector P P₂).2) := by
  sorry

end point_P_satisfies_conditions_l1189_118952


namespace lucas_50th_term_mod_5_l1189_118938

def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => lucas n + lucas (n + 1)

theorem lucas_50th_term_mod_5 : lucas 49 % 5 = 3 := by
  sorry

end lucas_50th_term_mod_5_l1189_118938


namespace square_and_ln_exp_are_geometric_l1189_118943

/-- A function is geometric if it preserves geometric sequences -/
def IsGeometricFunction (f : ℝ → ℝ) : Prop :=
  ∀ (a : ℕ → ℝ), (∀ n : ℕ, a n ≠ 0) →
    (∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) →
    (∀ n : ℕ, f (a (n + 1)) / f (a n) = f (a (n + 2)) / f (a (n + 1)))

theorem square_and_ln_exp_are_geometric :
  IsGeometricFunction (fun x ↦ x^2) ∧
  IsGeometricFunction (fun x ↦ Real.log (2^x)) :=
sorry

end square_and_ln_exp_are_geometric_l1189_118943


namespace r_equals_four_l1189_118931

/-- Given pr = 360 and 6cr = 15, prove that r = 4 is a valid solution. -/
theorem r_equals_four (p c : ℚ) (h1 : p * 4 = 360) (h2 : 6 * c * 4 = 15) : 4 = 4 := by
  sorry

end r_equals_four_l1189_118931


namespace cyclic_quadrilateral_theorem_l1189_118908

/-- A cyclic quadrilateral with side lengths a, b, c, d, area Q, and circumradius R -/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  Q : ℝ
  R : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ Q > 0 ∧ R > 0

/-- The main theorem about cyclic quadrilaterals -/
theorem cyclic_quadrilateral_theorem (ABCD : CyclicQuadrilateral) :
  ABCD.R^2 = ((ABCD.a * ABCD.b + ABCD.c * ABCD.d) * (ABCD.a * ABCD.c + ABCD.b * ABCD.d) * (ABCD.a * ABCD.d + ABCD.b * ABCD.c)) / (16 * ABCD.Q^2) ∧
  ABCD.R ≥ ((ABCD.a * ABCD.b * ABCD.c * ABCD.d)^(3/4)) / (ABCD.Q * Real.sqrt 2) ∧
  (ABCD.R = ((ABCD.a * ABCD.b * ABCD.c * ABCD.d)^(3/4)) / (ABCD.Q * Real.sqrt 2) ↔ ABCD.a = ABCD.b ∧ ABCD.b = ABCD.c ∧ ABCD.c = ABCD.d) :=
by sorry


end cyclic_quadrilateral_theorem_l1189_118908


namespace perpendicular_vectors_x_component_l1189_118957

/-- Given two 2D vectors m and n, if they are perpendicular and have specific components,
    then the x-component of m must be 2. -/
theorem perpendicular_vectors_x_component
  (m n : ℝ × ℝ)  -- m and n are 2D real vectors
  (h1 : m.1 = x ∧ m.2 = 2)  -- m = (x, 2)
  (h2 : n = (1, -1))  -- n = (1, -1)
  (h3 : m • n = 0)  -- m is perpendicular to n (dot product is zero)
  : x = 2 :=
by sorry

end perpendicular_vectors_x_component_l1189_118957


namespace composite_cube_three_diff_squares_l1189_118916

/-- A number is composite if it has a non-trivial factorization -/
def IsComposite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- The proposition that the cube of a composite number can be represented as the difference of two squares in at least three ways -/
theorem composite_cube_three_diff_squares (n : ℕ) (h : IsComposite n) : 
  ∃ (A₁ B₁ A₂ B₂ A₃ B₃ : ℕ), 
    (n^3 = A₁^2 - B₁^2) ∧ 
    (n^3 = A₂^2 - B₂^2) ∧ 
    (n^3 = A₃^2 - B₃^2) ∧ 
    (A₁, B₁) ≠ (A₂, B₂) ∧ 
    (A₁, B₁) ≠ (A₃, B₃) ∧ 
    (A₂, B₂) ≠ (A₃, B₃) :=
sorry

end composite_cube_three_diff_squares_l1189_118916


namespace largest_four_digit_divisible_by_five_l1189_118942

theorem largest_four_digit_divisible_by_five : ∃ n : ℕ, 
  (n ≤ 9999 ∧ n ≥ 1000) ∧ 
  n % 5 = 0 ∧
  ∀ m : ℕ, (m ≤ 9999 ∧ m ≥ 1000) → m % 5 = 0 → m ≤ n :=
by
  -- The proof goes here
  sorry

end largest_four_digit_divisible_by_five_l1189_118942


namespace cycling_jogging_swimming_rates_l1189_118947

theorem cycling_jogging_swimming_rates : ∃ (b j s : ℕ), 
  (3 * b + 2 * j + 4 * s = 66) ∧ 
  (3 * j + 2 * s + 4 * b = 96) ∧ 
  (b^2 + j^2 + s^2 = 612) := by
  sorry

end cycling_jogging_swimming_rates_l1189_118947


namespace triangle_existence_l1189_118918

-- Define the basic types and structures
structure Point := (x y : ℝ)

def Angle := ℝ

-- Define the given elements
variable (F T : Point) -- F is midpoint of AB, T is foot of altitude
variable (α : Angle) -- angle at vertex A

-- Define the properties of the triangle
def is_midpoint (F A B : Point) : Prop := F = Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2)

def is_altitude_foot (T A C : Point) : Prop := 
  (T.x - A.x) * (C.x - A.x) + (T.y - A.y) * (C.y - A.y) = 0

def angle_at_vertex (A B C : Point) (α : Angle) : Prop :=
  let v1 := Point.mk (B.x - A.x) (B.y - A.y)
  let v2 := Point.mk (C.x - A.x) (C.y - A.y)
  Real.cos α = (v1.x * v2.x + v1.y * v2.y) / 
    (Real.sqrt (v1.x^2 + v1.y^2) * Real.sqrt (v2.x^2 + v2.y^2))

-- The main theorem
theorem triangle_existence (F T : Point) (α : Angle) :
  ∃ (A B C : Point),
    is_midpoint F A B ∧
    is_altitude_foot T A C ∧
    angle_at_vertex A B C α ∧
    ¬(∀ (C' : Point), is_altitude_foot T A C' → C' = C) :=
by sorry

end triangle_existence_l1189_118918


namespace quadratic_sum_zero_discriminants_l1189_118964

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The discriminant of a quadratic polynomial -/
def discriminant (p : QuadraticPolynomial) : ℝ :=
  p.b ^ 2 - 4 * p.a * p.c

/-- A quadratic polynomial has zero discriminant -/
def hasZeroDiscriminant (p : QuadraticPolynomial) : Prop :=
  discriminant p = 0

/-- The sum of two quadratic polynomials -/
def sumQuadratic (p q : QuadraticPolynomial) : QuadraticPolynomial where
  a := p.a + q.a
  b := p.b + q.b
  c := p.c + q.c

theorem quadratic_sum_zero_discriminants :
  ∀ (p : QuadraticPolynomial),
  ∃ (q r : QuadraticPolynomial),
  hasZeroDiscriminant q ∧
  hasZeroDiscriminant r ∧
  p = sumQuadratic q r :=
sorry

end quadratic_sum_zero_discriminants_l1189_118964


namespace negative_sixty_four_to_seven_thirds_l1189_118981

theorem negative_sixty_four_to_seven_thirds :
  (-64 : ℝ) ^ (7/3) = -1024 := by sorry

end negative_sixty_four_to_seven_thirds_l1189_118981


namespace fraction_power_four_l1189_118936

theorem fraction_power_four : (5 / 3 : ℚ) ^ 4 = 625 / 81 := by
  sorry

end fraction_power_four_l1189_118936


namespace fifteen_percent_problem_l1189_118920

theorem fifteen_percent_problem : ∃ x : ℝ, (15 / 100) * x = 90 ∧ x = 600 := by
  sorry

end fifteen_percent_problem_l1189_118920


namespace lucky_5n_is_52000_l1189_118935

/-- A natural number is lucky if the sum of its digits is 7 -/
def isLucky (n : ℕ) : Prop :=
  (n.digits 10).sum = 7

/-- The sequence of lucky numbers in increasing order -/
def luckySeq : ℕ → ℕ := sorry

/-- The nth element of the lucky number sequence is 2005 -/
axiom nth_lucky_is_2005 (n : ℕ) : luckySeq n = 2005

theorem lucky_5n_is_52000 (n : ℕ) : luckySeq (5 * n) = 52000 :=
  sorry

end lucky_5n_is_52000_l1189_118935


namespace original_proposition_is_true_negation_is_false_l1189_118944

theorem original_proposition_is_true : ∀ (a b : ℝ), a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1) := by
  sorry

theorem negation_is_false : ¬(∀ (a b : ℝ), a + b ≥ 2 → (a < 1 ∧ b < 1)) := by
  sorry

end original_proposition_is_true_negation_is_false_l1189_118944


namespace watermelon_sales_l1189_118925

/-- Proves that the number of watermelons sold is 18, given the weight, price per pound, and total revenue -/
theorem watermelon_sales
  (weight : ℕ)
  (price_per_pound : ℕ)
  (total_revenue : ℕ)
  (h1 : weight = 23)
  (h2 : price_per_pound = 2)
  (h3 : total_revenue = 828) :
  total_revenue / (weight * price_per_pound) = 18 :=
by sorry

end watermelon_sales_l1189_118925


namespace tyrones_money_value_l1189_118913

/-- Represents the total value of Tyrone's money in US dollars -/
def tyrones_money : ℚ :=
  let us_currency : ℚ :=
    4 * 1 +  -- $1 bills
    1 * 10 +  -- $10 bill
    2 * 5 +  -- $5 bills
    30 * (1/4) +  -- quarters
    5 * (1/2) +  -- half-dollar coins
    48 * (1/10) +  -- dimes
    12 * (1/20) +  -- nickels
    4 * 1 +  -- one-dollar coins
    64 * (1/100) +  -- pennies
    3 * 2 +  -- two-dollar bills
    5 * (1/2)  -- 50-cent coins

  let foreign_currency : ℚ :=
    20 * (11/10) +  -- Euro coins
    15 * (132/100) +  -- British Pound coins
    6 * (76/100)  -- Canadian Dollar coins

  us_currency + foreign_currency

/-- The theorem stating that Tyrone's money equals $98.90 -/
theorem tyrones_money_value : tyrones_money = 989/10 := by
  sorry

end tyrones_money_value_l1189_118913


namespace parts_count_l1189_118919

/-- Represents the number of parts in pile A -/
def pile_a : ℕ := sorry

/-- Represents the number of parts in pile B -/
def pile_b : ℕ := sorry

/-- The condition that transferring 15 parts from A to B makes them equal -/
axiom equal_after_a_to_b : pile_a - 15 = pile_b + 15

/-- The condition that transferring 15 parts from B to A makes A three times B -/
axiom triple_after_b_to_a : pile_a + 15 = 3 * (pile_b - 15)

/-- The theorem stating the original number in pile A and the total number of parts -/
theorem parts_count : pile_a = 75 ∧ pile_a + pile_b = 120 := by sorry

end parts_count_l1189_118919


namespace section_b_average_weight_l1189_118922

/-- Given a class with two sections, prove the average weight of section B --/
theorem section_b_average_weight
  (total_students : ℕ)
  (section_a_students : ℕ)
  (section_b_students : ℕ)
  (section_a_avg_weight : ℝ)
  (total_avg_weight : ℝ)
  (h1 : total_students = section_a_students + section_b_students)
  (h2 : section_a_students = 50)
  (h3 : section_b_students = 50)
  (h4 : section_a_avg_weight = 60)
  (h5 : total_avg_weight = 70)
  : (total_students * total_avg_weight - section_a_students * section_a_avg_weight) / section_b_students = 80 := by
  sorry

end section_b_average_weight_l1189_118922


namespace sum_base5_equals_1333_l1189_118965

/-- Converts a number from base 5 to decimal --/
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from decimal to base 5 --/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- The sum of 213₅, 324₅, and 141₅ is equal to 1333₅ in base 5 --/
theorem sum_base5_equals_1333 :
  decimalToBase5 (base5ToDecimal [3, 1, 2] + base5ToDecimal [4, 2, 3] + base5ToDecimal [1, 4, 1]) = [3, 3, 3, 1] :=
sorry

end sum_base5_equals_1333_l1189_118965


namespace systematic_sampling_result_l1189_118996

/-- Represents the systematic sampling problem -/
def SystematicSampling (total_population : ℕ) (sample_size : ℕ) (first_number : ℕ) (threshold : ℕ) : Prop :=
  let interval := total_population / sample_size
  let selected_numbers := List.range sample_size |>.map (fun i => first_number + i * interval)
  (selected_numbers.filter (fun n => n > threshold)).length = 8

/-- Theorem stating the result of the systematic sampling problem -/
theorem systematic_sampling_result : 
  SystematicSampling 960 32 9 750 := by
  sorry

end systematic_sampling_result_l1189_118996


namespace book_club_member_ratio_l1189_118977

theorem book_club_member_ratio :
  ∀ (r p : ℕ), 
    r > 0 → p > 0 →
    (5 * r + 12 * p : ℚ) / (r + p) = 8 →
    (r : ℚ) / p = 4 / 3 := by
  sorry

end book_club_member_ratio_l1189_118977


namespace min_value_xy_l1189_118971

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y + 12 = x * y) :
  x * y ≥ 36 :=
sorry

end min_value_xy_l1189_118971


namespace garden_perimeter_l1189_118984

/-- A rectangular garden with a diagonal of 20 meters and an area of 96 square meters has a perimeter of 49 meters. -/
theorem garden_perimeter : ∀ a b : ℝ,
  a > 0 → b > 0 →
  a^2 + b^2 = 20^2 →
  a * b = 96 →
  2 * (a + b) = 49 := by
sorry

end garden_perimeter_l1189_118984


namespace ball_painting_probability_l1189_118989

def num_balls : ℕ := 8
def num_red : ℕ := 4
def num_blue : ℕ := 4

def prob_red : ℚ := 1 / 2
def prob_blue : ℚ := 1 / 2

theorem ball_painting_probability :
  (prob_red ^ num_red) * (prob_blue ^ num_blue) = 1 / 256 := by
  sorry

end ball_painting_probability_l1189_118989


namespace sum_ratio_equals_3_l1189_118973

def sum_multiples_of_3 (n : ℕ) : ℕ := 
  3 * (n * (n + 1) / 2)

def sum_integers (m : ℕ) : ℕ := 
  m * (m + 1) / 2

theorem sum_ratio_equals_3 : 
  (sum_multiples_of_3 200) / (sum_integers 200) = 3 := by
  sorry

end sum_ratio_equals_3_l1189_118973


namespace unique_solution_modular_equation_l1189_118999

theorem unique_solution_modular_equation :
  ∃! n : ℤ, 0 ≤ n ∧ n < 102 ∧ 99 * n % 102 = 73 % 102 ∧ n = 97 := by
  sorry

end unique_solution_modular_equation_l1189_118999


namespace acute_triangle_properties_l1189_118910

theorem acute_triangle_properties (A B C : Real) (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π) 
  (h_equation : Real.sqrt 3 * Real.sin ((B + C) / 2) - Real.cos A = 1) : 
  A = π / 3 ∧ ∀ x, x = Real.cos B + Real.cos C → Real.sqrt 3 / 2 < x ∧ x ≤ 1 := by
  sorry

end acute_triangle_properties_l1189_118910


namespace percentage_of_women_parents_l1189_118970

theorem percentage_of_women_parents (P : ℝ) (M : ℝ) (F : ℝ) : 
  P > 0 →
  M + F = P →
  (1 / 8) * M + (1 / 4) * F = (17.5 / 100) * P →
  M / P = 0.6 :=
by sorry

end percentage_of_women_parents_l1189_118970


namespace unique_equidistant_cell_l1189_118990

-- Define the distance function for cells on an infinite chessboard
def distance (a b : ℤ × ℤ) : ℕ :=
  max (Int.natAbs (a.1 - b.1)) (Int.natAbs (a.2 - b.2))

-- Define the theorem
theorem unique_equidistant_cell
  (A B C : ℤ × ℤ)
  (hab : distance A B = 100)
  (hac : distance A C = 100)
  (hbc : distance B C = 100) :
  ∃! X : ℤ × ℤ, distance X A = 50 ∧ distance X B = 50 ∧ distance X C = 50 :=
sorry

end unique_equidistant_cell_l1189_118990


namespace malcolm_green_lights_l1189_118951

/-- The number of green lights Malcolm bought -/
def green_lights (red blue green total_needed : ℕ) : Prop :=
  green = total_needed - (red + blue)

/-- Theorem stating the number of green lights Malcolm bought -/
theorem malcolm_green_lights :
  ∃ (green : ℕ), 
    let red := 12
    let blue := 3 * red
    let total_needed := 59 - 5
    green_lights red blue green total_needed ∧ green = 6 := by
  sorry

end malcolm_green_lights_l1189_118951


namespace equation_holds_iff_l1189_118907

theorem equation_holds_iff (a b c : ℝ) (ha : a ≠ 0) (hab : a + b ≠ 0) :
  (a + b + c) / a = (b + c) / (a + b) ↔ a = -(b + c) := by
  sorry

end equation_holds_iff_l1189_118907


namespace room_occupancy_l1189_118997

theorem room_occupancy (total_chairs : ℕ) (seated_people : ℕ) (total_people : ℕ) :
  total_chairs = 25 →
  seated_people = (4 : ℕ) * total_chairs / 5 →
  seated_people = (3 : ℕ) * total_people / 5 →
  total_people = 33 :=
by
  sorry

#check room_occupancy

end room_occupancy_l1189_118997
