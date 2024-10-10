import Mathlib

namespace dot_product_special_vectors_l3499_349959

/-- The dot product of vectors a = (sin 55°, sin 35°) and b = (sin 25°, sin 65°) is equal to √3/2 -/
theorem dot_product_special_vectors :
  let a : ℝ × ℝ := (Real.sin (55 * π / 180), Real.sin (35 * π / 180))
  let b : ℝ × ℝ := (Real.sin (25 * π / 180), Real.sin (65 * π / 180))
  (a.1 * b.1 + a.2 * b.2) = Real.sqrt 3 / 2 := by
  sorry

end dot_product_special_vectors_l3499_349959


namespace first_pipe_fill_time_l3499_349916

def cistern_problem (x : ℝ) : Prop :=
  let second_pipe_time : ℝ := 15
  let both_pipes_time : ℝ := 6
  let remaining_time : ℝ := 1.5
  (both_pipes_time / x + both_pipes_time / second_pipe_time + remaining_time / second_pipe_time) = 1

theorem first_pipe_fill_time :
  ∃ x : ℝ, cistern_problem x ∧ x = 12 := by
  sorry

end first_pipe_fill_time_l3499_349916


namespace circus_tent_seating_l3499_349907

theorem circus_tent_seating (total_capacity : ℕ) (num_sections : ℕ) : 
  total_capacity = 984 → num_sections = 4 → 
  (total_capacity / num_sections : ℕ) = 246 := by
  sorry

end circus_tent_seating_l3499_349907


namespace stratified_sample_teachers_under_40_l3499_349908

/-- Calculates the number of teachers under 40 in a stratified sample -/
def stratified_sample_size (total_population : ℕ) (under_40_population : ℕ) (sample_size : ℕ) : ℕ :=
  (under_40_population * sample_size) / total_population

/-- Theorem: The stratified sample size for teachers under 40 is 50 -/
theorem stratified_sample_teachers_under_40 :
  stratified_sample_size 490 350 70 = 50 := by
  sorry

end stratified_sample_teachers_under_40_l3499_349908


namespace opposite_faces_sum_seven_l3499_349937

-- Define a type for the faces of a die
inductive DieFace : Type
  | one : DieFace
  | two : DieFace
  | three : DieFace
  | four : DieFace
  | five : DieFace
  | six : DieFace

-- Define a function to get the numeric value of a face
def faceValue : DieFace → Nat
  | DieFace.one => 1
  | DieFace.two => 2
  | DieFace.three => 3
  | DieFace.four => 4
  | DieFace.five => 5
  | DieFace.six => 6

-- Define a function to get the opposite face
def oppositeFace : DieFace → DieFace
  | DieFace.one => DieFace.six
  | DieFace.two => DieFace.five
  | DieFace.three => DieFace.four
  | DieFace.four => DieFace.three
  | DieFace.five => DieFace.two
  | DieFace.six => DieFace.one

-- Theorem: The sum of values on opposite faces is always 7
theorem opposite_faces_sum_seven (face : DieFace) :
  faceValue face + faceValue (oppositeFace face) = 7 := by
  sorry


end opposite_faces_sum_seven_l3499_349937


namespace five_Y_three_equals_four_l3499_349950

-- Define the Y operation
def Y (a b : ℤ) : ℤ := a^2 - 2*a*b + b^2

-- Theorem statement
theorem five_Y_three_equals_four : Y 5 3 = 4 := by
  sorry

end five_Y_three_equals_four_l3499_349950


namespace function_is_exponential_base_3_l3499_349941

-- Define the properties of the function f
def satisfies_functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x * f y

def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem function_is_exponential_base_3 (f : ℝ → ℝ) 
  (h1 : satisfies_functional_equation f)
  (h2 : monotonically_increasing f) :
  ∀ x, f x = 3^x :=
sorry

end function_is_exponential_base_3_l3499_349941


namespace quadratic_equation_solution_l3499_349977

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 10 * x + 6
  ∃ x₁ x₂ : ℝ, x₁ = 5/3 + Real.sqrt 7/3 ∧ 
             x₂ = 5/3 - Real.sqrt 7/3 ∧
             f x₁ = 0 ∧ f x₂ = 0 ∧
             ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end quadratic_equation_solution_l3499_349977


namespace alice_bob_meet_time_l3499_349936

def circlePoints : ℕ := 15
def aliceMove : ℕ := 4
def bobMove : ℕ := 8  -- Equivalent clockwise movement

theorem alice_bob_meet_time :
  let relativeMove := (bobMove - aliceMove) % circlePoints
  ∃ n : ℕ, n > 0 ∧ (n * relativeMove) % circlePoints = 0 ∧
  ∀ m : ℕ, 0 < m ∧ m < n → (m * relativeMove) % circlePoints ≠ 0 ∧
  n = 15 := by
sorry

end alice_bob_meet_time_l3499_349936


namespace sine_function_properties_l3499_349906

/-- Given a function y = a * sin(x) + 2b where a > 0, with maximum value 4 and minimum value 0,
    prove that a + b = 3 and the minimum positive period of y = b * sin(ax) is π -/
theorem sine_function_properties (a b : ℝ) (h_a_pos : a > 0)
  (h_max : ∀ x, a * Real.sin x + 2 * b ≤ 4)
  (h_min : ∀ x, a * Real.sin x + 2 * b ≥ 0)
  (h_max_achievable : ∃ x, a * Real.sin x + 2 * b = 4)
  (h_min_achievable : ∃ x, a * Real.sin x + 2 * b = 0) :
  (a + b = 3) ∧
  (∀ T > 0, (∀ x, b * Real.sin (a * x) = b * Real.sin (a * (x + T))) → T ≥ π) ∧
  (∀ x, b * Real.sin (a * x) = b * Real.sin (a * (x + π))) := by
  sorry


end sine_function_properties_l3499_349906


namespace p_sufficient_not_necessary_for_q_l3499_349934

-- Define the conditions
def condition_p (m : ℝ) : Prop := ∀ x, x^2 + m*x + 1 > 0

def condition_q (m : ℝ) : Prop := ∀ x y, x < y → (m+3)^x < (m+3)^y

-- State the theorem
theorem p_sufficient_not_necessary_for_q :
  (∃ m : ℝ, condition_p m ∧ condition_q m) ∧
  (∃ m : ℝ, ¬condition_p m ∧ condition_q m) :=
sorry

end p_sufficient_not_necessary_for_q_l3499_349934


namespace gcd_fraction_equality_l3499_349982

theorem gcd_fraction_equality (a b c d : ℕ) (h : a * b = c * d) :
  (Nat.gcd a c * Nat.gcd a d) / Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = a := by
  sorry

end gcd_fraction_equality_l3499_349982


namespace no_xyz_single_double_triple_digits_l3499_349983

theorem no_xyz_single_double_triple_digits :
  ¬∃ (x y z : ℕ+),
    (1 : ℚ) / x = 1 / y + 1 / z ∧
    ((1 ≤ x ∧ x < 10) ∨ (1 ≤ y ∧ y < 10) ∨ (1 ≤ z ∧ z < 10)) ∧
    ((10 ≤ x ∧ x < 100) ∨ (10 ≤ y ∧ y < 100) ∨ (10 ≤ z ∧ z < 100)) ∧
    ((100 ≤ x ∧ x < 1000) ∨ (100 ≤ y ∧ y < 1000) ∨ (100 ≤ z ∧ z < 1000)) :=
by sorry

end no_xyz_single_double_triple_digits_l3499_349983


namespace seven_by_seven_grid_shaded_percentage_l3499_349985

/-- Represents a square grid -/
structure SquareGrid :=
  (size : ℕ)
  (shaded : ℕ)

/-- Calculates the percentage of shaded area in a square grid -/
def shadedPercentage (grid : SquareGrid) : ℚ :=
  (grid.shaded : ℚ) / (grid.size * grid.size : ℚ) * 100

/-- Theorem: The percentage of shaded area in a 7x7 grid with 7 shaded squares is (1/7) * 100% -/
theorem seven_by_seven_grid_shaded_percentage :
  let grid : SquareGrid := ⟨7, 7⟩
  shadedPercentage grid = 100 / 7 := by sorry

end seven_by_seven_grid_shaded_percentage_l3499_349985


namespace range_symmetric_range_b_decreasing_l3499_349954

-- Define the function f
def f (a b x : ℝ) : ℝ := -2 * x^2 + a * x + b

-- Theorem for part (1)
theorem range_symmetric (a b : ℝ) :
  f a b 2 = -3 →
  (∀ x : ℝ, f a b (1 + x) = f a b (1 - x)) →
  (∀ x : ℝ, x ∈ Set.Icc (-2) 3 → f a b x ∈ Set.Icc (-19) (-1)) :=
sorry

-- Theorem for part (2)
theorem range_b_decreasing (a b : ℝ) :
  f a b 2 = -3 →
  (∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x ≤ y → f a b x ≥ f a b y) →
  b ≥ -3 :=
sorry

end range_symmetric_range_b_decreasing_l3499_349954


namespace apples_on_ground_l3499_349992

/-- The number of apples that have fallen to the ground -/
def fallen_apples : ℕ := sorry

/-- The number of apples hanging on the tree -/
def hanging_apples : ℕ := 5

/-- The number of apples eaten by the dog -/
def eaten_apples : ℕ := 3

/-- The number of apples left after the dog eats -/
def remaining_apples : ℕ := 10

theorem apples_on_ground :
  fallen_apples = 13 :=
by sorry

end apples_on_ground_l3499_349992


namespace inequality_proof_l3499_349970

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (sum_eq_two : a + b + c + d = 2) :
  (a^2 / (a^2 + 1)^2) + (b^2 / (b^2 + 1)^2) + 
  (c^2 / (c^2 + 1)^2) + (d^2 / (d^2 + 1)^2) ≤ 16/25 := by
  sorry

end inequality_proof_l3499_349970


namespace inverse_product_positive_implies_one_greater_than_one_l3499_349955

theorem inverse_product_positive_implies_one_greater_than_one 
  (a b c : ℝ) (h : (a⁻¹) * (b⁻¹) * (c⁻¹) > 0) : 
  (a > 1) ∨ (b > 1) ∨ (c > 1) := by
  sorry

end inverse_product_positive_implies_one_greater_than_one_l3499_349955


namespace cylinder_diameter_from_sphere_surface_area_l3499_349911

theorem cylinder_diameter_from_sphere_surface_area (r_sphere : ℝ) (h_cylinder : ℝ) :
  r_sphere = 3 →
  h_cylinder = 6 →
  4 * Real.pi * r_sphere^2 = 2 * Real.pi * (6 / 2) * h_cylinder →
  6 = 2 * (6 / 2) :=
by sorry

end cylinder_diameter_from_sphere_surface_area_l3499_349911


namespace largest_divisor_of_expression_l3499_349981

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (12*x + 3) * (12*x + 9) * (6*x + 6) = 324 * k) ∧
  (∀ (m : ℤ), m > 324 → ¬(∀ (x : ℤ), Odd x → ∃ (k : ℤ), (12*x + 3) * (12*x + 9) * (6*x + 6) = m * k)) :=
by sorry

end largest_divisor_of_expression_l3499_349981


namespace angle_sum_quarter_range_l3499_349901

-- Define acute and obtuse angles
def is_acute (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2
def is_obtuse (β : Real) : Prop := Real.pi / 2 < β ∧ β < Real.pi

-- Main theorem
theorem angle_sum_quarter_range (α β : Real) 
  (h_acute : is_acute α) (h_obtuse : is_obtuse β) :
  Real.pi / 8 < 0.25 * (α + β) ∧ 0.25 * (α + β) < 3 * Real.pi / 8 := by
  sorry

#check angle_sum_quarter_range

end angle_sum_quarter_range_l3499_349901


namespace arithmetic_sequence_general_term_l3499_349996

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 1 = 3 ∧
  a 2 + a 5 = 36

/-- The general term formula for the arithmetic sequence -/
def GeneralTermFormula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = 6 * n - 3

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  GeneralTermFormula a :=
sorry

end arithmetic_sequence_general_term_l3499_349996


namespace expression_equivalence_l3499_349927

theorem expression_equivalence (x y z : ℝ) :
  let P := x + y
  let Q := x - y
  ((P + Q + z) / (P - Q - z) - (P - Q - z) / (P + Q + z)) = 
    (4 * (x^2 + y^2 + x*z)) / ((2*y - z) * (2*x + z)) := by
  sorry

end expression_equivalence_l3499_349927


namespace books_sold_l3499_349947

theorem books_sold (initial_books : ℕ) (new_books : ℕ) (final_books : ℕ) 
  (h1 : initial_books = 34)
  (h2 : new_books = 7)
  (h3 : final_books = 24) :
  initial_books - (initial_books - new_books - final_books) = 17 := by
  sorry

end books_sold_l3499_349947


namespace point_on_transformed_graph_l3499_349986

/-- Given a function g: ℝ → ℝ, prove that if (2,5) lies on the graph of y = g(x),
    then (1,8) lies on the graph of 4y = 5g(3x-1) + 7, and the sum of the coordinates of (1,8) is 9. -/
theorem point_on_transformed_graph (g : ℝ → ℝ) (h : g 2 = 5) :
  4 * 8 = 5 * g (3 * 1 - 1) + 7 ∧ 1 + 8 = 9 := by
  sorry

end point_on_transformed_graph_l3499_349986


namespace sum_reciprocal_squares_geq_sum_squares_l3499_349922

theorem sum_reciprocal_squares_geq_sum_squares 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (sum_eq_3 : a + b + c = 3) :
  1/a^2 + 1/b^2 + 1/c^2 ≥ a^2 + b^2 + c^2 := by
  sorry

end sum_reciprocal_squares_geq_sum_squares_l3499_349922


namespace main_theorem_l3499_349929

/-- A function with the property that (f x + y) * (f y + x) > 0 implies f x + y = f y + x -/
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x + y) * (f y + x) > 0 → f x + y = f y + x

/-- The main theorem: if f has the property, then f x + y ≤ f y + x whenever x > y -/
theorem main_theorem (f : ℝ → ℝ) (hf : has_property f) :
  ∀ x y : ℝ, x > y → f x + y ≤ f y + x :=
sorry

end main_theorem_l3499_349929


namespace system_solution_l3499_349972

theorem system_solution (x y z u : ℚ) (a b c d : ℚ) :
  (x * y) / (x + y) = 1 / a ∧
  (y * z) / (y + z) = 1 / b ∧
  (z * u) / (z + u) = 1 / c ∧
  (x * y * z * u) / (x + y + z + u) = 1 / d →
  ((a = 1 ∧ b = 2 ∧ c = -1 ∧ d = 1 →
    x = -4/3 ∧ y = 4/7 ∧ z = 4 ∧ u = -4/5) ∧
   (a = 1 ∧ b = 3 ∧ c = -2 ∧ d = 1 →
    (x = -1 ∧ y = 1/2 ∧ z = 1 ∧ u = -1/3) ∨
    (x = 1/9 ∧ y = -1/8 ∧ z = 1/11 ∧ u = -1/13))) :=
by sorry

end system_solution_l3499_349972


namespace train_length_problem_l3499_349973

theorem train_length_problem (speed1 speed2 time length2 : Real) 
  (h1 : speed1 = 120)
  (h2 : speed2 = 80)
  (h3 : time = 9)
  (h4 : length2 = 210.04)
  : ∃ length1 : Real, length1 = 290 := by
  sorry

end train_length_problem_l3499_349973


namespace painting_price_change_l3499_349943

/-- The percentage increase in the first year given the conditions of the problem -/
def first_year_increase : ℝ := 30

/-- The percentage decrease in the second year -/
def second_year_decrease : ℝ := 15

/-- The final price as a percentage of the original price -/
def final_price_percentage : ℝ := 110.5

theorem painting_price_change : 
  (100 + first_year_increase) * (100 - second_year_decrease) / 100 = final_price_percentage := by
  sorry

#check painting_price_change

end painting_price_change_l3499_349943


namespace expand_and_simplify_l3499_349976

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end expand_and_simplify_l3499_349976


namespace vector_equation_solution_l3499_349903

variable (V : Type) [AddCommGroup V] [Module ℝ V]

variable (cross : V → V → V)

variable (crossProperties : 
  (∀ a b c : V, cross (a + b) c = cross a c + cross b c) ∧ 
  (∀ a b : V, cross a b = -cross b a) ∧
  (∀ a : V, cross a a = 0))

theorem vector_equation_solution :
  ∃! (k m : ℝ), ∀ (a b c d : V), 
    a + b + c + d = 0 → 
    k • (cross b a) + m • (cross d c) + cross b c + cross c a + cross d b = 0 :=
by sorry

end vector_equation_solution_l3499_349903


namespace tiled_square_theorem_l3499_349913

/-- A square area tiled with identical square tiles -/
structure TiledSquare where
  /-- The number of tiles adjoining the four sides -/
  perimeter_tiles : ℕ
  /-- The total number of tiles in the square -/
  total_tiles : ℕ

/-- Theorem stating that a square area with 20 tiles adjoining its sides contains 36 tiles in total -/
theorem tiled_square_theorem (ts : TiledSquare) (h : ts.perimeter_tiles = 20) : ts.total_tiles = 36 := by
  sorry

end tiled_square_theorem_l3499_349913


namespace tenth_term_of_geometric_sequence_l3499_349931

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_term (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

/-- The 10th term of a geometric sequence with first term 2 and second term 5/2 -/
theorem tenth_term_of_geometric_sequence :
  let a : ℚ := 2
  let second_term : ℚ := 5/2
  let r : ℚ := second_term / a
  geometric_term a r 10 = 3906250/262144 := by sorry

end tenth_term_of_geometric_sequence_l3499_349931


namespace rectangle_area_increase_l3499_349928

theorem rectangle_area_increase (L W P : ℝ) (h : L > 0) (h' : W > 0) (h'' : P > 0) :
  (L * (1 + P)) * (W * (1 + P)) = 4 * (L * W) → P = 1 := by
  sorry

end rectangle_area_increase_l3499_349928


namespace complement_of_union_l3499_349938

universe u

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def A : Set ℕ := {1,3,5,7}
def B : Set ℕ := {2,4,5}

theorem complement_of_union : 
  (U \ (A ∪ B)) = {6,8} := by sorry

end complement_of_union_l3499_349938


namespace quadratic_root_in_unit_interval_l3499_349971

/-- Given a quadratic function f(x) = ax² + bx + c where 2a + 3b + 6c = 0,
    there exists an x in the interval (0,1) such that f(x) = 0. -/
theorem quadratic_root_in_unit_interval (a b c : ℝ) 
  (h : 2*a + 3*b + 6*c = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end quadratic_root_in_unit_interval_l3499_349971


namespace vertex_of_quadratic_l3499_349915

/-- The quadratic function f(x) = -(x+1)^2 - 8 -/
def f (x : ℝ) : ℝ := -(x + 1)^2 - 8

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (-1, -8)

/-- Theorem: The vertex of the quadratic function f(x) = -(x+1)^2 - 8 is at the point (-1, -8) -/
theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≤ f (vertex.1)) ∧ f (vertex.1) = vertex.2 := by
  sorry

end vertex_of_quadratic_l3499_349915


namespace total_cost_theorem_l3499_349975

/-- The cost of cherries per pound in yuan -/
def cherry_cost : ℝ := sorry

/-- The cost of apples per pound in yuan -/
def apple_cost : ℝ := sorry

/-- The total cost of 2 pounds of cherries and 3 pounds of apples is 58 yuan -/
axiom condition1 : 2 * cherry_cost + 3 * apple_cost = 58

/-- The total cost of 3 pounds of cherries and 2 pounds of apples is 72 yuan -/
axiom condition2 : 3 * cherry_cost + 2 * apple_cost = 72

/-- The theorem states that the total cost of 3 pounds of cherries and 3 pounds of apples is 78 yuan -/
theorem total_cost_theorem : 3 * cherry_cost + 3 * apple_cost = 78 := by
  sorry

end total_cost_theorem_l3499_349975


namespace multiply_and_add_equality_l3499_349991

theorem multiply_and_add_equality : 45 * 52 + 48 * 45 = 4500 := by
  sorry

end multiply_and_add_equality_l3499_349991


namespace gcd_3869_6497_l3499_349974

theorem gcd_3869_6497 : Nat.gcd 3869 6497 = 73 := by
  sorry

end gcd_3869_6497_l3499_349974


namespace build_time_relation_l3499_349924

/-- Represents the time taken to build a cottage given the number of builders and their rate -/
def build_time (builders : ℕ) (rate : ℚ) : ℚ :=
  1 / (builders.cast * rate)

/-- Theorem stating the relationship between build times for different numbers of builders -/
theorem build_time_relation (n : ℕ) (rate : ℚ) :
  n > 0 → 6 > 0 → build_time n rate = 8 → 
  build_time 6 rate = (n.cast / 6 : ℚ) * 8 := by
  sorry

#check build_time_relation

end build_time_relation_l3499_349924


namespace restaurant_group_children_correct_number_of_children_l3499_349902

theorem restaurant_group_children (adults : ℕ) (meal_cost : ℕ) (total_bill : ℕ) : ℕ :=
  let children := (total_bill - adults * meal_cost) / meal_cost
  children

theorem correct_number_of_children :
  restaurant_group_children 2 3 21 = 5 := by
  sorry

end restaurant_group_children_correct_number_of_children_l3499_349902


namespace bisection_method_root_interval_l3499_349904

-- Define the function f(x) = x^3 - 2x - 1
def f (x : ℝ) : ℝ := x^3 - 2*x - 1

-- State the theorem
theorem bisection_method_root_interval :
  f 1 < 0 → f 2 > 0 → f 1.5 < 0 →
  ∃ r ∈ Set.Ioo 1.5 2, f r = 0 :=
by sorry

end bisection_method_root_interval_l3499_349904


namespace number_difference_and_division_l3499_349968

theorem number_difference_and_division (S L : ℕ) : 
  L - S = 8327 → L = 21 * S + 125 → S = 410 ∧ L = 8735 := by
  sorry

end number_difference_and_division_l3499_349968


namespace distinct_cube_constructions_l3499_349930

/-- The group of rotational symmetries of a cube -/
def CubeRotationGroup : Type := Unit

/-- The number of elements in the cube rotation group -/
def CubeRotationGroup.order : ℕ := 24

/-- The number of ways to place 5 white cubes in a 2x2x2 cube -/
def WhiteCubePlacements : ℕ := Nat.choose 8 5

/-- The number of fixed points under the identity rotation -/
def FixedPointsUnderIdentity : ℕ := WhiteCubePlacements

/-- The number of fixed points under all non-identity rotations -/
def FixedPointsUnderNonIdentity : ℕ := 0

/-- The total number of fixed points under all rotations -/
def TotalFixedPoints : ℕ := FixedPointsUnderIdentity + 23 * FixedPointsUnderNonIdentity

theorem distinct_cube_constructions :
  (TotalFixedPoints : ℚ) / CubeRotationGroup.order = 7 / 3 := by sorry

end distinct_cube_constructions_l3499_349930


namespace vector_b_components_l3499_349957

def vector_a : ℝ × ℝ := (2, -1)

theorem vector_b_components :
  ∀ (b : ℝ × ℝ),
  (∃ (k : ℝ), k < 0 ∧ b = (k * vector_a.1, k * vector_a.2)) →
  (b.1 * b.1 + b.2 * b.2 = 20) →
  b = (-4, 2) := by sorry

end vector_b_components_l3499_349957


namespace inequality_solution_range_l3499_349964

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 7| ≥ a^2 - 3*a) → 
  a ∈ Set.Icc (-2 : ℝ) 5 := by
sorry

end inequality_solution_range_l3499_349964


namespace square_inscribed_in_circle_l3499_349956

theorem square_inscribed_in_circle (r : ℝ) (S : ℝ) :
  r > 0 →
  r^2 * π = 16 * π →
  S = (2 * r)^2 / 2 →
  S = 32 :=
by sorry

end square_inscribed_in_circle_l3499_349956


namespace triangle_circumcircle_identity_l3499_349912

/-- Given a triangle inscribed in a circle, this theorem states the relationship
    between the sides, angles, and the radius of the circumscribed circle. -/
theorem triangle_circumcircle_identity 
  (R : ℝ) (A B C : ℝ) (a b c : ℝ)
  (h_triangle : A + B + C = π)
  (h_a : a = 2 * R * Real.sin A)
  (h_b : b = 2 * R * Real.sin B)
  (h_c : c = 2 * R * Real.sin C) :
  a * Real.cos A + b * Real.cos B + c * Real.cos C = 4 * R * Real.sin A * Real.sin B * Real.sin C :=
by sorry

end triangle_circumcircle_identity_l3499_349912


namespace conditional_probability_wind_rain_l3499_349919

/-- Given probabilities of events A and B, and their intersection,
    prove that the conditional probability P(B|A) is 3/4 -/
theorem conditional_probability_wind_rain 
  (P_A P_B P_AB : ℝ) 
  (h_A : P_A = 0.4)
  (h_B : P_B = 0.5)
  (h_AB : P_AB = 0.3) :
  P_AB / P_A = 3/4 := by
  sorry

end conditional_probability_wind_rain_l3499_349919


namespace profit_ratio_theorem_l3499_349961

/-- Represents a partner's investment details -/
structure Partner where
  investment : ℚ
  time : ℕ

/-- Calculates the profit factor for a partner -/
def profitFactor (p : Partner) : ℚ :=
  p.investment * p.time

/-- Theorem: Given the investment ratio and time periods, prove the profit ratio -/
theorem profit_ratio_theorem (p q : Partner) 
  (h1 : p.investment / q.investment = 7 / 5)
  (h2 : p.time = 20)
  (h3 : q.time = 40) :
  profitFactor p / profitFactor q = 7 / 10 := by
  sorry

#check profit_ratio_theorem

end profit_ratio_theorem_l3499_349961


namespace incorrect_number_value_l3499_349939

theorem incorrect_number_value (n : ℕ) (initial_avg correct_avg incorrect_value : ℚ) 
  (h1 : n = 10)
  (h2 : initial_avg = 46)
  (h3 : correct_avg = 51)
  (h4 : incorrect_value = 25) :
  let correct_value := n * correct_avg - (n * initial_avg - incorrect_value)
  correct_value = 75 := by sorry

end incorrect_number_value_l3499_349939


namespace base10_to_base5_453_l3499_349952

-- Define a function to convert from base 10 to base 5
def toBase5 (n : ℕ) : List ℕ :=
  sorry

-- Theorem stating that 453 in base 10 is equal to 3303 in base 5
theorem base10_to_base5_453 : toBase5 453 = [3, 3, 0, 3] :=
  sorry

end base10_to_base5_453_l3499_349952


namespace rational_inequality_l3499_349905

theorem rational_inequality (x : ℝ) : (x^2 - 9) / (x + 3) < 0 ↔ x ∈ Set.Ioo (-3 : ℝ) 3 := by
  sorry

end rational_inequality_l3499_349905


namespace linear_expr_pythagorean_relation_l3499_349951

-- Define linear expressions
def LinearExpr (α : Type*) [Ring α] := α → α

-- Theorem statement
theorem linear_expr_pythagorean_relation
  {α : Type*} [Field α]
  (A B C : LinearExpr α)
  (h : ∀ x, (A x)^2 + (B x)^2 = (C x)^2) :
  ∃ (k₁ k₂ : α), ∀ x, A x = k₁ * (C x) ∧ B x = k₂ * (C x) := by
  sorry

end linear_expr_pythagorean_relation_l3499_349951


namespace quadratic_has_two_distinct_roots_find_k_value_l3499_349953

/-- A quadratic equation with parameter k -/
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 + (2*k - 1)*x - k - 2

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ := (2*k - 1)^2 - 4*1*(-k - 2)

theorem quadratic_has_two_distinct_roots (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 :=
sorry

theorem find_k_value (k : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : quadratic k x₁ = 0)
  (h₂ : quadratic k x₂ = 0)
  (h₃ : x₁ + x₂ - 4*x₁*x₂ = 1) :
  k = -4 :=
sorry

end quadratic_has_two_distinct_roots_find_k_value_l3499_349953


namespace rationalize_denominator_l3499_349990

theorem rationalize_denominator : 15 / Real.sqrt 45 = Real.sqrt 5 := by
  sorry

end rationalize_denominator_l3499_349990


namespace cord_cutting_problem_l3499_349914

theorem cord_cutting_problem (cord1 : ℕ) (cord2 : ℕ) 
  (h1 : cord1 = 15) (h2 : cord2 = 12) : 
  Nat.gcd cord1 cord2 = 3 := by
sorry

end cord_cutting_problem_l3499_349914


namespace unique_prime_in_range_l3499_349925

theorem unique_prime_in_range : ∃! (n : ℕ), 
  50 < n ∧ n < 60 ∧ 
  Nat.Prime n ∧ 
  n % 7 = 3 := by
sorry

end unique_prime_in_range_l3499_349925


namespace cubic_difference_l3499_349923

theorem cubic_difference (x : ℝ) (h : x - 1/x = 3) : x^3 - 1/x^3 = 36 := by
  sorry

end cubic_difference_l3499_349923


namespace same_type_as_reference_l3499_349917

-- Define the type of polynomial expressions
def PolynomialExpr (α : Type) := List (α × ℕ)

-- Function to get the type of a polynomial expression
def exprType (expr : PolynomialExpr ℚ) : PolynomialExpr ℚ :=
  expr.map (λ (c, e) ↦ (1, e))

-- Define the reference expression 3a²b
def reference : PolynomialExpr ℚ := [(3, 2), (1, 1)]

-- Define the given expressions
def expr1 : PolynomialExpr ℚ := [(-2, 2), (1, 1)]  -- -2a²b
def expr2 : PolynomialExpr ℚ := [(-2, 1), (1, 1)]  -- -2ab
def expr3 : PolynomialExpr ℚ := [(2, 1), (2, 1)]   -- 2ab²
def expr4 : PolynomialExpr ℚ := [(2, 2)]           -- 2a²

theorem same_type_as_reference :
  (exprType expr1 = exprType reference) ∧
  (exprType expr2 ≠ exprType reference) ∧
  (exprType expr3 ≠ exprType reference) ∧
  (exprType expr4 ≠ exprType reference) :=
by sorry

end same_type_as_reference_l3499_349917


namespace at_least_one_chooses_probability_l3499_349910

-- Define the probabilities for Students A and B
def prob_A : ℚ := 1/3
def prob_B : ℚ := 1/4

-- Define the event that at least one student chooses the "Inequality Lecture"
def at_least_one_chooses : ℚ := 1 - (1 - prob_A) * (1 - prob_B)

-- Theorem statement
theorem at_least_one_chooses_probability :
  at_least_one_chooses = 1/2 :=
sorry

end at_least_one_chooses_probability_l3499_349910


namespace prob_a_prob_b_prob_c_prob_d_prob_e_chess_probabilities_l3499_349948

/-- The total number of chess pieces -/
def total_pieces : ℕ := 32

/-- The number of pieces of each color -/
def pieces_per_color : ℕ := total_pieces / 2

/-- The number of pawns of each color -/
def pawns_per_color : ℕ := 8

/-- The number of bishops of each color -/
def bishops_per_color : ℕ := 2

/-- The number of rooks of each color -/
def rooks_per_color : ℕ := 2

/-- The number of knights of each color -/
def knights_per_color : ℕ := 2

/-- The number of kings of each color -/
def kings_per_color : ℕ := 1

/-- The number of queens of each color -/
def queens_per_color : ℕ := 1

/-- The probability of drawing 2 dark pieces or 2 pieces of different colors -/
theorem prob_a : ℚ :=
  47 / 62

/-- The probability of drawing 1 bishop and 1 pawn or 2 pieces of different colors -/
theorem prob_b : ℚ :=
  18 / 31

/-- The probability of drawing 2 different-colored rooks or 2 pieces of the same color but different sizes -/
theorem prob_c : ℚ :=
  91 / 248

/-- The probability of drawing 1 king and one knight of the same color, or two pieces of the same color -/
theorem prob_d : ℚ :=
  15 / 31

/-- The probability of drawing 2 pieces of the same size or 2 pieces of the same color -/
theorem prob_e : ℚ :=
  159 / 248

/-- The main theorem combining all probabilities -/
theorem chess_probabilities :
  (prob_a = 47 / 62) ∧
  (prob_b = 18 / 31) ∧
  (prob_c = 91 / 248) ∧
  (prob_d = 15 / 31) ∧
  (prob_e = 159 / 248) :=
by sorry

end prob_a_prob_b_prob_c_prob_d_prob_e_chess_probabilities_l3499_349948


namespace exists_multiple_with_digit_sum_l3499_349980

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number that is a multiple of 2015 and whose sum of digits equals 2015 -/
theorem exists_multiple_with_digit_sum :
  ∃ (n : ℕ), (n % 2015 = 0) ∧ (sum_of_digits n = 2015) := by sorry

end exists_multiple_with_digit_sum_l3499_349980


namespace problem_solution_l3499_349969

theorem problem_solution (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : 
  x = 50 ∨ x = 16 := by
sorry

end problem_solution_l3499_349969


namespace smallest_k_with_remainder_one_l3499_349987

theorem smallest_k_with_remainder_one : ∃! k : ℕ,
  k > 1 ∧
  k % 19 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 19 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  -- Proof goes here
  sorry

end smallest_k_with_remainder_one_l3499_349987


namespace correct_sum_after_reversing_tens_digit_l3499_349989

/-- Represents a three-digit number with digits a, b, and c --/
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- Represents the same number with tens digit reversed --/
def reversed_tens_digit (a b c : ℕ) : ℕ := 100 * a + 10 * c + b

theorem correct_sum_after_reversing_tens_digit 
  (m n : ℕ) 
  (a b c : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : m = three_digit_number a b c) 
  (h4 : reversed_tens_digit a b c + n = 128) :
  m + n = 128 := by
sorry

end correct_sum_after_reversing_tens_digit_l3499_349989


namespace inequality_proof_l3499_349979

theorem inequality_proof (x y : ℝ) : 
  |((x + y) * (1 - x * y)) / ((1 + x^2) * (1 + y^2))| ≤ (1 / 2) := by
  sorry

end inequality_proof_l3499_349979


namespace right_triangle_legs_l3499_349966

theorem right_triangle_legs (c n : ℝ) (h1 : c > 0) (h2 : n > 0) : 
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧ 
    a^2 + b^2 = c^2 ∧ 
    (n / Real.sqrt 3)^2 = a * b * (1 - ((a + b) / c)^2) ∧
    a = n / 2 ∧ b = c * Real.sqrt 3 / 2 := by
  sorry

end right_triangle_legs_l3499_349966


namespace min_values_xy_and_x_plus_2y_l3499_349988

theorem min_values_xy_and_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1/x + 9/y = 1) :
  xy ≥ 36 ∧ x + 2*y ≥ 19 + 6*Real.sqrt 2 := by
sorry

end min_values_xy_and_x_plus_2y_l3499_349988


namespace sunzi_wood_measurement_l3499_349995

/-- Given a piece of wood of length x and a rope of length y, 
    if there are 4.5 feet of rope left when measuring the wood 
    and 1 foot left when measuring with half the rope, 
    then the system of equations y - x = 4.5 and x - y/2 = 1 holds. -/
theorem sunzi_wood_measurement (x y : ℝ) 
  (h1 : y - x = 4.5) 
  (h2 : x - y / 2 = 1) : 
  y - x = 4.5 ∧ x - y / 2 = 1 := by
  sorry

end sunzi_wood_measurement_l3499_349995


namespace sin_cos_pi_12_l3499_349932

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end sin_cos_pi_12_l3499_349932


namespace fiona_reaches_food_l3499_349946

/-- Represents a lily pad --/
structure LilyPad :=
  (number : ℕ)

/-- Represents Fiona the frog --/
structure Frog :=
  (position : LilyPad)

/-- Represents the probability of a jump --/
def JumpProbability : ℚ := 1/3

/-- The total number of lily pads --/
def TotalPads : ℕ := 16

/-- The position of the first predator --/
def Predator1 : LilyPad := ⟨4⟩

/-- The position of the second predator --/
def Predator2 : LilyPad := ⟨9⟩

/-- The position of the food --/
def FoodPosition : LilyPad := ⟨14⟩

/-- Fiona's starting position --/
def StartPosition : LilyPad := ⟨0⟩

/-- Function to calculate the probability of Fiona reaching the food --/
noncomputable def probabilityToReachFood (f : Frog) : ℚ :=
  sorry

theorem fiona_reaches_food :
  probabilityToReachFood ⟨StartPosition⟩ = 52/59049 :=
sorry

end fiona_reaches_food_l3499_349946


namespace two_digit_number_difference_l3499_349997

theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 72 → x - y = 8 := by
  sorry

end two_digit_number_difference_l3499_349997


namespace boat_distance_proof_l3499_349909

-- Define the given constants
def boat_speed : ℝ := 10
def stream_speed : ℝ := 2
def time_difference : ℝ := 1.5  -- 90 minutes in hours

-- Define the theorem
theorem boat_distance_proof :
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let upstream_time := (downstream_speed * time_difference) / (downstream_speed - upstream_speed)
  let distance := upstream_speed * upstream_time
  distance = 36 := by sorry

end boat_distance_proof_l3499_349909


namespace axis_of_symmetry_can_be_left_of_y_axis_l3499_349935

theorem axis_of_symmetry_can_be_left_of_y_axis :
  ∃ (a : ℝ), a > 0 ∧ ∃ (x : ℝ), x < 0 ∧
    x = -(1 - 2*a) / (2*a) ∧
    ∀ (y : ℝ), y = a*x^2 + (1 - 2*a)*x :=
by sorry

end axis_of_symmetry_can_be_left_of_y_axis_l3499_349935


namespace quadratic_and_inequality_system_solution_l3499_349945

theorem quadratic_and_inequality_system_solution :
  (∃ x : ℝ, x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) ∧
  (∀ x : ℝ, 3*x + 5 ≥ 2 ∧ (x - 1) / 2 < (x + 1) / 4 ↔ -1 ≤ x ∧ x < 3) :=
by sorry

end quadratic_and_inequality_system_solution_l3499_349945


namespace max_xy_constraint_min_x_plus_y_constraint_l3499_349999

-- Part 1
theorem max_xy_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y + x*y = 12) :
  x*y ≤ 4 ∧ (x*y = 4 → x = 4 ∧ y = 1) :=
sorry

-- Part 2
theorem min_x_plus_y_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y = x*y) :
  x + y ≥ 9 ∧ (x + y = 9 → x = 6 ∧ y = 3) :=
sorry

end max_xy_constraint_min_x_plus_y_constraint_l3499_349999


namespace twelfth_day_is_monday_l3499_349962

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with specific properties -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  fridayCount : Nat
  dayCount : Nat
  firstNotFriday : firstDay ≠ DayOfWeek.Friday
  lastNotFriday : lastDay ≠ DayOfWeek.Friday
  exactlyFiveFridays : fridayCount = 5
  validDayCount : dayCount = 30 ∨ dayCount = 31

/-- Function to get the day of the week for a given day number -/
def getDayOfWeek (m : Month) (day : Nat) : DayOfWeek :=
  sorry

/-- Theorem stating that the 12th day is a Monday -/
theorem twelfth_day_is_monday (m : Month) : 
  getDayOfWeek m 12 = DayOfWeek.Monday :=
sorry

end twelfth_day_is_monday_l3499_349962


namespace f_even_not_odd_implies_a_gt_one_l3499_349944

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 - 1) + Real.sqrt (a - x^2)

theorem f_even_not_odd_implies_a_gt_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) ∧ 
  (∃ x, f a x ≠ -(f a (-x))) →
  a > 1 := by sorry

end f_even_not_odd_implies_a_gt_one_l3499_349944


namespace households_with_appliances_l3499_349940

theorem households_with_appliances 
  (total : ℕ) 
  (tv : ℕ) 
  (fridge : ℕ) 
  (both : ℕ) 
  (h1 : total = 100) 
  (h2 : tv = 65) 
  (h3 : fridge = 84) 
  (h4 : both = 53) : 
  tv + fridge - both = 96 := by
  sorry

end households_with_appliances_l3499_349940


namespace trapezoid_mn_length_l3499_349942

/-- Represents a trapezoid ABCD with points M and N on its sides -/
structure Trapezoid (α : Type*) [LinearOrderedField α] :=
  (a b : α)  -- lengths of BC and AD respectively
  (area_ratio : α)  -- ratio of areas of MBCN to MADN

/-- 
  Given a trapezoid ABCD with BC = a and AD = b, 
  if MN is parallel to AD and the areas of trapezoids MBCN and MADN are in the ratio 1:5, 
  then MN = sqrt((5a^2 + b^2) / 6)
-/
theorem trapezoid_mn_length 
  {α : Type*} [LinearOrderedField α] (t : Trapezoid α) 
  (h_ratio : t.area_ratio = 1/5) :
  ∃ mn : α, mn^2 = (5*t.a^2 + t.b^2) / 6 :=
sorry

end trapezoid_mn_length_l3499_349942


namespace pizza_piece_volume_l3499_349960

/-- The volume of a piece of pizza -/
theorem pizza_piece_volume (thickness : ℝ) (diameter : ℝ) (num_pieces : ℕ) : 
  thickness = 1/3 →
  diameter = 12 →
  num_pieces = 12 →
  (π * (diameter/2)^2 * thickness) / num_pieces = π := by
  sorry

#check pizza_piece_volume

end pizza_piece_volume_l3499_349960


namespace min_value_f1_div_f2prime0_l3499_349958

/-- A quadratic function f(x) = ax² + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  f_prime_0_pos : 2 * a * 0 + b > 0
  range_nonneg : ∀ x, a * x^2 + b * x + c ≥ 0

/-- The theorem stating the minimum value of f(1) / f''(0) for quadratic functions with specific properties -/
theorem min_value_f1_div_f2prime0 (f : QuadraticFunction) :
  (∀ g : QuadraticFunction, (g.a + g.b + g.c) / (2 * g.a) ≥ (f.a + f.b + f.c) / (2 * f.a)) →
  (f.a + f.b + f.c) / (2 * f.a) = 2 :=
sorry

end min_value_f1_div_f2prime0_l3499_349958


namespace circular_pool_area_l3499_349998

theorem circular_pool_area (AB DC : ℝ) (h1 : AB = 20) (h2 : DC = 12) : ∃ (r : ℝ), r^2 = 244 ∧ π * r^2 = 244 * π := by
  sorry

end circular_pool_area_l3499_349998


namespace domain_of_f_l3499_349918

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define our function f(x) = lg(x+1)
noncomputable def f (x : ℝ) := lg (x + 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | f x = f x} = {x : ℝ | x > -1} :=
sorry

end domain_of_f_l3499_349918


namespace gas_pressure_calculation_l3499_349967

/-- Represents the pressure-volume relationship for a gas at constant temperature -/
structure GasState where
  volume : ℝ
  pressure : ℝ
  inv_prop : volume * pressure = volume * pressure

/-- The initial state of the gas -/
def initial_state : GasState where
  volume := 3
  pressure := 8
  inv_prop := by sorry

/-- The final state of the gas -/
def final_state : GasState where
  volume := 7.5
  pressure := 3.2
  inv_prop := by sorry

/-- Theorem stating that the final pressure is correct given the initial conditions -/
theorem gas_pressure_calculation (initial : GasState) (final : GasState)
    (h_initial : initial = initial_state)
    (h_final_volume : final.volume = 7.5)
    (h_const : initial.volume * initial.pressure = final.volume * final.pressure) :
    final.pressure = 3.2 := by
  sorry

end gas_pressure_calculation_l3499_349967


namespace special_1992_gon_exists_l3499_349965

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : Fin n → ℝ
  convex : sorry -- Condition for convexity

/-- An inscribed circle in a polygon -/
structure InscribedCircle {n : ℕ} (p : ConvexPolygon n) where
  center : ℝ × ℝ
  radius : ℝ
  touches_all_sides : sorry -- Condition that the circle touches all sides

/-- The theorem stating the existence of the special 1992-gon -/
theorem special_1992_gon_exists : ∃ (p : ConvexPolygon 1992),
  (∃ (σ : Equiv (Fin 1992) (Fin 1992)), ∀ i, p.sides i = σ i + 1) ∧
  ∃ (c : InscribedCircle p), True :=
sorry

end special_1992_gon_exists_l3499_349965


namespace sequence_property_l3499_349933

def arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b ∧ a > 0 ∧ b > 0 ∧ c > 0

def geometric_sequence (a b c : ℝ) : Prop :=
  b / a = c / b ∧ a ≠ 0 ∧ b ≠ 0

def general_term (n : ℕ) : ℝ := 2^(n - 1)

theorem sequence_property :
  ∀ a b c : ℝ,
    arithmetic_sequence a b c →
    a + b + c = 6 →
    geometric_sequence (a + 3) (b + 6) (c + 13) →
    (∀ n : ℕ, n ≥ 3 → general_term n = (a + 3) * 2^(n - 3)) :=
by sorry

end sequence_property_l3499_349933


namespace no_primes_in_list_l3499_349978

/-- Represents a number formed by repeating 57 a certain number of times -/
def repeatedNumber (repetitions : ℕ) : ℕ :=
  57 * ((10^(2*repetitions) - 1) / 99)

/-- The list of numbers formed by repeating 57 from 1 to n times -/
def numberList (n : ℕ) : List ℕ :=
  List.map repeatedNumber (List.range n)

/-- Counts the number of prime numbers in the list -/
def countPrimes (list : List ℕ) : ℕ :=
  (list.filter Nat.Prime).length

theorem no_primes_in_list (n : ℕ) : countPrimes (numberList n) = 0 := by
  sorry

end no_primes_in_list_l3499_349978


namespace sum_of_angles_l3499_349926

theorem sum_of_angles (angle1 angle2 angle3 angle4 angle5 angle6 angleA angleB angleC : ℝ) :
  angle1 + angle3 + angle5 = 180 →
  angle2 + angle4 + angle6 = 180 →
  angleA + angleB + angleC = 180 →
  angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angleA + angleB + angleC = 540 := by
sorry

end sum_of_angles_l3499_349926


namespace inequality_on_unit_circle_l3499_349994

/-- The complex unit circle -/
def unit_circle : Set ℂ := {z : ℂ | Complex.abs z = 1}

/-- The inequality holds for all points on the unit circle -/
theorem inequality_on_unit_circle :
  ∀ z ∈ unit_circle, (Complex.abs (z + 1) - Real.sqrt 2) * (Complex.abs (z - 1) - Real.sqrt 2) ≤ 0 := by
  sorry

end inequality_on_unit_circle_l3499_349994


namespace arithmetic_mean_problem_l3499_349963

theorem arithmetic_mean_problem (a b c d : ℝ) :
  (a + b + c + d + 120) / 5 = 100 →
  (a + b + c + d) / 4 = 95 := by
  sorry

end arithmetic_mean_problem_l3499_349963


namespace problem_solution_l3499_349900

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 3 - f a 2 = 1) :
  (∃ m_lower m_upper : ℝ, m_lower = 2/3 ∧ m_upper = 7 ∧
    ∀ m : ℝ, m_lower < m ∧ m < m_upper ↔ f a (3*m - 2) < f a (2*m + 5)) ∧
  (∃ x : ℝ, x = 4 ∧ f a (x - 2/x) = Real.log (7/2) / Real.log (3/2)) :=
by sorry

end problem_solution_l3499_349900


namespace ellipse_parabola_line_equations_l3499_349949

/-- Given an ellipse and a parabola with specific properties, prove the equations of both curves and a line. -/
theorem ellipse_parabola_line_equations :
  ∀ (a b c p : ℝ) (F A B P Q D : ℝ × ℝ),
  a > 0 → b > 0 → a > b →
  c / a = 1 / 2 →
  A.1 - F.1 = a →
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 →
  ∀ (x y : ℝ), y^2 = 2 * p * x →
  A.1 - F.1 = 1 / 2 →
  P.1 = Q.1 ∧ P.2 = -Q.2 →
  B ≠ A →
  D.2 = 0 →
  abs ((A.1 - P.1) * (D.2 - P.2) - (A.2 - P.2) * (D.1 - P.1)) / 2 = Real.sqrt 6 / 2 →
  ((∀ (x y : ℝ), x^2 + 4 * y^2 / 3 = 1) ∧
   (∀ (x y : ℝ), y^2 = 4 * x) ∧
   ((3 * P.1 + Real.sqrt 6 * P.2 - 3 = 0) ∨ (3 * P.1 - Real.sqrt 6 * P.2 - 3 = 0))) := by
  sorry

end ellipse_parabola_line_equations_l3499_349949


namespace one_hundred_twentieth_letter_l3499_349984

def letter_pattern (n : ℕ) : Char :=
  match n % 4 with
  | 0 => 'D'
  | 1 => 'A'
  | 2 => 'B'
  | 3 => 'C'
  | _ => 'D'  -- This case is unreachable, but Lean requires it for exhaustiveness

theorem one_hundred_twentieth_letter :
  letter_pattern 120 = 'D' := by
  sorry

end one_hundred_twentieth_letter_l3499_349984


namespace currency_denomination_problem_l3499_349993

theorem currency_denomination_problem (total_notes : ℕ) (total_amount : ℕ) (amount_50 : ℕ) (d : ℕ) :
  total_notes = 85 →
  total_amount = 5000 →
  amount_50 = 3500 →
  (amount_50 / 50 + (total_notes - amount_50 / 50)) = total_notes →
  50 * (amount_50 / 50) + d * (total_notes - amount_50 / 50) = total_amount →
  d = 100 := by
sorry

end currency_denomination_problem_l3499_349993


namespace half_AB_equals_2_1_l3499_349920

def MA : ℝ × ℝ := (-2, 4)
def MB : ℝ × ℝ := (2, 6)

theorem half_AB_equals_2_1 : (1 / 2 : ℝ) • (MB - MA) = (2, 1) := by sorry

end half_AB_equals_2_1_l3499_349920


namespace sqrt_six_div_sqrt_two_eq_sqrt_three_l3499_349921

theorem sqrt_six_div_sqrt_two_eq_sqrt_three : 
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end sqrt_six_div_sqrt_two_eq_sqrt_three_l3499_349921
