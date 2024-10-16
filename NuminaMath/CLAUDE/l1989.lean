import Mathlib

namespace NUMINAMATH_CALUDE_cube_sum_theorem_l1989_198996

theorem cube_sum_theorem (a b c : ℝ) 
  (sum_eq : a + b + c = 4)
  (sum_prod_eq : a * b + a * c + b * c = 6)
  (prod_eq : a * b * c = -8) :
  a^3 + b^3 + c^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l1989_198996


namespace NUMINAMATH_CALUDE_insecticide_potency_range_specific_insecticide_potency_range_l1989_198968

/-- Given two insecticide powders, find the range of potency for the second powder
    to achieve a specific mixture potency. -/
theorem insecticide_potency_range 
  (weight1 : ℝ) (potency1 : ℝ) (weight2 : ℝ) 
  (lower_bound : ℝ) (upper_bound : ℝ) :
  weight1 > 0 ∧ weight2 > 0 ∧
  0 < potency1 ∧ potency1 < 1 ∧
  0 < lower_bound ∧ lower_bound < upper_bound ∧ upper_bound < 1 →
  ∃ (lower_x upper_x : ℝ),
    lower_x > potency1 ∧
    ∀ x, lower_x < x ∧ x < upper_x →
      lower_bound < (weight1 * potency1 + weight2 * x) / (weight1 + weight2) ∧
      (weight1 * potency1 + weight2 * x) / (weight1 + weight2) < upper_bound :=
by sorry

/-- The specific insecticide potency range problem. -/
theorem specific_insecticide_potency_range :
  ∃ (lower_x upper_x : ℝ),
    lower_x = 0.33 ∧ upper_x = 0.42 ∧
    ∀ x, 0.33 < x ∧ x < 0.42 →
      0.25 < (40 * 0.15 + 50 * x) / (40 + 50) ∧
      (40 * 0.15 + 50 * x) / (40 + 50) < 0.30 :=
by sorry

end NUMINAMATH_CALUDE_insecticide_potency_range_specific_insecticide_potency_range_l1989_198968


namespace NUMINAMATH_CALUDE_equation_solution_l1989_198912

theorem equation_solution : 
  ∃! x : ℚ, (x - 20) / 3 = (4 - 3 * x) / 4 ∧ x = 92 / 13 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1989_198912


namespace NUMINAMATH_CALUDE_rectangle_length_from_square_wire_l1989_198957

/-- Given a square with side length 12 and a rectangle with the same perimeter and width 6,
    prove that the length of the rectangle is 18. -/
theorem rectangle_length_from_square_wire (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) :
  square_side = 12 →
  rect_width = 6 →
  4 * square_side = 2 * (rect_width + rect_length) →
  rect_length = 18 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_from_square_wire_l1989_198957


namespace NUMINAMATH_CALUDE_greatest_b_not_in_range_l1989_198975

/-- The quadratic function f(x) = x^2 + bx + 20 -/
def f (b : ℤ) (x : ℝ) : ℝ := x^2 + b*x + 20

/-- Predicate that checks if -9 is not in the range of f for a given b -/
def not_in_range (b : ℤ) : Prop := ∀ x : ℝ, f b x ≠ -9

/-- The theorem stating that 10 is the greatest integer b such that -9 is not in the range of f -/
theorem greatest_b_not_in_range : 
  (not_in_range 10 ∧ ∀ b : ℤ, b > 10 → ¬(not_in_range b)) := by sorry

end NUMINAMATH_CALUDE_greatest_b_not_in_range_l1989_198975


namespace NUMINAMATH_CALUDE_length_PQ_is_4_l1989_198910

-- Define the semicircle (C)
def semicircle (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1 ∧ 0 ≤ y ∧ y ≤ 1

-- Define the polar equation of line (l)
def line_l (ρ θ : ℝ) : Prop :=
  ρ * (Real.sin θ + Real.sqrt 3 * Real.cos θ) = 5 * Real.sqrt 3

-- Define the ray OM
def ray_OM (θ : ℝ) : Prop :=
  θ = Real.pi / 3

-- Define the point P as the intersection of semicircle (C) and ray OM
def point_P (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos θ ∧ ray_OM θ

-- Define the point Q as the intersection of line (l) and ray OM
def point_Q (ρ θ : ℝ) : Prop :=
  line_l ρ θ ∧ ray_OM θ

-- Theorem statement
theorem length_PQ_is_4 :
  ∀ (ρ_P θ_P ρ_Q θ_Q : ℝ),
    point_P ρ_P θ_P →
    point_Q ρ_Q θ_Q →
    |ρ_P - ρ_Q| = 4 :=
sorry

end NUMINAMATH_CALUDE_length_PQ_is_4_l1989_198910


namespace NUMINAMATH_CALUDE_three_isosceles_triangles_l1989_198920

-- Define a point in 2D space
structure Point :=
  (x : Int) (y : Int)

-- Define a triangle by its three vertices
structure Triangle :=
  (v1 : Point) (v2 : Point) (v3 : Point)

-- Function to calculate the squared distance between two points
def squaredDistance (p1 p2 : Point) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d1 := squaredDistance t.v1 t.v2
  let d2 := squaredDistance t.v2 t.v3
  let d3 := squaredDistance t.v3 t.v1
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

-- Define the five triangles
def triangle1 : Triangle := ⟨⟨1, 5⟩, ⟨3, 5⟩, ⟨2, 2⟩⟩
def triangle2 : Triangle := ⟨⟨4, 3⟩, ⟨4, 6⟩, ⟨7, 3⟩⟩
def triangle3 : Triangle := ⟨⟨1, 1⟩, ⟨5, 2⟩, ⟨9, 1⟩⟩
def triangle4 : Triangle := ⟨⟨7, 5⟩, ⟨6, 6⟩, ⟨9, 3⟩⟩
def triangle5 : Triangle := ⟨⟨8, 2⟩, ⟨10, 5⟩, ⟨10, 0⟩⟩

-- Theorem: Exactly 3 out of the 5 given triangles are isosceles
theorem three_isosceles_triangles :
  (isIsosceles triangle1 ∧ isIsosceles triangle2 ∧ isIsosceles triangle3 ∧
   ¬isIsosceles triangle4 ∧ ¬isIsosceles triangle5) :=
by sorry

end NUMINAMATH_CALUDE_three_isosceles_triangles_l1989_198920


namespace NUMINAMATH_CALUDE_range_of_a_l1989_198974

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 4*x + 3 ≤ 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x < x - a}

-- Theorem statement
theorem range_of_a :
  (∀ x : ℝ, x ∈ B a → x ∈ A) ∧ 
  (∃ x : ℝ, x ∈ A ∧ x ∉ B a) →
  a ∈ Set.Icc 1 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1989_198974


namespace NUMINAMATH_CALUDE_broccoli_carrot_calorie_ratio_l1989_198949

/-- The number of calories in a pound of carrots -/
def carrot_calories : ℕ := 51

/-- The number of pounds of carrots Tom eats -/
def carrot_pounds : ℕ := 1

/-- The number of pounds of broccoli Tom eats -/
def broccoli_pounds : ℕ := 2

/-- The total number of calories Tom ate -/
def total_calories : ℕ := 85

/-- The number of calories in a pound of broccoli -/
def broccoli_calories : ℚ := (total_calories - carrot_calories * carrot_pounds) / broccoli_pounds

theorem broccoli_carrot_calorie_ratio :
  broccoli_calories / carrot_calories = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_broccoli_carrot_calorie_ratio_l1989_198949


namespace NUMINAMATH_CALUDE_statement_I_statement_II_statement_III_statement_IV_l1989_198983

-- Define the complex square root function
noncomputable def complexSqrt : ℂ → ℂ := sorry

-- Statement (I)
theorem statement_I (a b : ℂ) : complexSqrt (a^2 + b^2) = 0 ↔ a = 0 ∧ b = 0 := by sorry

-- Statement (II)
theorem statement_II : ¬∃ (a b : ℂ), (a ≠ 0 ∨ b ≠ 0) ∧ complexSqrt (a^2 + b^2) = a * b := by sorry

-- Statement (III)
theorem statement_III : ¬∃ (a b : ℂ), (a ≠ 0 ∨ b ≠ 0) ∧ complexSqrt (a^2 + b^2) = a + b := by sorry

-- Statement (IV)
theorem statement_IV : ¬∃ (a b : ℂ), (a ≠ 0 ∨ b ≠ 0) ∧ complexSqrt (a^2 + b^2) = a * b := by sorry

end NUMINAMATH_CALUDE_statement_I_statement_II_statement_III_statement_IV_l1989_198983


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l1989_198905

theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) :
  initial_stock = 120 →
  books_sold = 39 →
  books_per_shelf = 9 →
  (initial_stock - books_sold) / books_per_shelf = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l1989_198905


namespace NUMINAMATH_CALUDE_smallest_group_size_exists_group_size_l1989_198994

theorem smallest_group_size (n : ℕ) : 
  (n % 6 = 1) ∧ (n % 8 = 3) ∧ (n % 9 = 5) → n ≥ 187 :=
by sorry

theorem exists_group_size : 
  ∃ n : ℕ, (n % 6 = 1) ∧ (n % 8 = 3) ∧ (n % 9 = 5) ∧ n = 187 :=
by sorry

end NUMINAMATH_CALUDE_smallest_group_size_exists_group_size_l1989_198994


namespace NUMINAMATH_CALUDE_x_value_l1989_198980

theorem x_value (x y : ℝ) : 
  (x = y * 0.9) → (y = 125 * 1.1) → x = 123.75 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1989_198980


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_l1989_198948

-- Define the sample space
def Ω : Type := Fin 3 → Bool

-- Define the events
def A (ω : Ω) : Prop := ∀ i, ω i = true
def B (ω : Ω) : Prop := ∀ i, ω i = false
def C (ω : Ω) : Prop := ∃ i j, ω i ≠ ω j

-- Theorem statement
theorem events_mutually_exclusive :
  (∀ ω, ¬(A ω ∧ B ω)) ∧
  (∀ ω, ¬(A ω ∧ C ω)) ∧
  (∀ ω, ¬(B ω ∧ C ω)) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_l1989_198948


namespace NUMINAMATH_CALUDE_polygon_sides_count_polygon_has_2023_sides_l1989_198971

/-- A polygon with the property that at most 2021 triangles can be formed
    when a diagonal is drawn from a vertex has 2023 sides. -/
theorem polygon_sides_count : ℕ :=
  2023

/-- The maximum number of triangles formed when drawing a diagonal from a vertex
    of a polygon with n sides is n - 2. -/
def max_triangles (n : ℕ) : ℕ := n - 2

/-- The condition that at most 2021 triangles can be formed. -/
axiom triangle_condition : max_triangles polygon_sides_count ≤ 2021

/-- Theorem stating that the polygon has 2023 sides. -/
theorem polygon_has_2023_sides : polygon_sides_count = 2023 := by
  sorry

#check polygon_has_2023_sides

end NUMINAMATH_CALUDE_polygon_sides_count_polygon_has_2023_sides_l1989_198971


namespace NUMINAMATH_CALUDE_quarter_percentage_approx_l1989_198984

def dimes : ℕ := 60
def quarters : ℕ := 30
def nickels : ℕ := 40

def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5

def total_value : ℕ := dimes * dime_value + quarters * quarter_value + nickels * nickel_value
def quarter_value_total : ℕ := quarters * quarter_value

theorem quarter_percentage_approx (ε : ℝ) (h : ε > 0) :
  ∃ (p : ℝ), abs (p - 48.4) < ε ∧ p = (quarter_value_total : ℝ) / total_value * 100 :=
sorry

end NUMINAMATH_CALUDE_quarter_percentage_approx_l1989_198984


namespace NUMINAMATH_CALUDE_trailing_zeros_of_power_sum_l1989_198903

theorem trailing_zeros_of_power_sum : ∃ n : ℕ, n > 0 ∧ 
  (4^(5^6) + 6^(5^4) : ℕ) % (10^n) = 0 ∧ 
  (4^(5^6) + 6^(5^4) : ℕ) % (10^(n+1)) ≠ 0 ∧ 
  n = 5 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_power_sum_l1989_198903


namespace NUMINAMATH_CALUDE_solve_for_a_l1989_198932

theorem solve_for_a : ∃ a : ℝ, 
  (2 : ℝ) - a * (1 : ℝ) = -1 ∧ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1989_198932


namespace NUMINAMATH_CALUDE_digit_206788_is_7_l1989_198929

/-- The sequence of digits formed by concatenating all natural numbers from 1 onwards -/
def digit_sequence : ℕ → ℕ :=
  sorry

/-- The number of digits used to represent all natural numbers up to n -/
def digits_used_up_to (n : ℕ) : ℕ :=
  sorry

/-- The function that returns the digit at a given position in the sequence -/
def digit_at_position (pos : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the 206788th digit in the sequence is 7 -/
theorem digit_206788_is_7 : digit_at_position 206788 = 7 :=
  sorry

end NUMINAMATH_CALUDE_digit_206788_is_7_l1989_198929


namespace NUMINAMATH_CALUDE_cleaning_time_ratio_with_help_cleaning_time_ratio_l1989_198977

/-- Represents the grove of trees -/
structure Grove where
  rows : Nat
  columns : Nat

/-- Represents the time spent cleaning trees -/
structure CleaningTime where
  minutes : Nat

theorem cleaning_time_ratio_with_help (g : Grove) 
  (time_per_tree_without_help : Nat) 
  (total_time_with_help : CleaningTime) : 
  2 * (total_time_with_help.minutes / (g.rows * g.columns)) = time_per_tree_without_help :=
by
  sorry

#check cleaning_time_ratio_with_help

/-- Main theorem that proves the ratio of cleaning time with help to without help is 1:2 -/
theorem cleaning_time_ratio (g : Grove) 
  (time_per_tree_without_help : Nat) 
  (total_time_with_help : CleaningTime) : 
  (total_time_with_help.minutes / (g.rows * g.columns)) / time_per_tree_without_help = 1 / 2 :=
by
  sorry

#check cleaning_time_ratio

end NUMINAMATH_CALUDE_cleaning_time_ratio_with_help_cleaning_time_ratio_l1989_198977


namespace NUMINAMATH_CALUDE_sqrt_meaningfulness_l1989_198947

theorem sqrt_meaningfulness (x : ℝ) : x = 5 → (2*x - 4 ≥ 0) ∧ 
  (x = -1 → ¬(2*x - 4 ≥ 0)) ∧ 
  (x = 0 → ¬(2*x - 4 ≥ 0)) ∧ 
  (x = 1 → ¬(2*x - 4 ≥ 0)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningfulness_l1989_198947


namespace NUMINAMATH_CALUDE_sasha_took_right_triangle_l1989_198901

-- Define the triangle types
inductive TriangleType
  | Acute
  | Right
  | Obtuse

-- Define a function to check if two triangles can form the third
def canFormThird (t1 t2 t3 : TriangleType) : Prop :=
  (t1 ≠ t2) ∧ (t2 ≠ t3) ∧ (t1 ≠ t3) ∧
  ((t1 = TriangleType.Acute ∧ t2 = TriangleType.Obtuse) ∨
   (t1 = TriangleType.Obtuse ∧ t2 = TriangleType.Acute)) ∧
  t3 = TriangleType.Right

-- Theorem statement
theorem sasha_took_right_triangle (t1 t2 t3 : TriangleType) :
  (t1 ≠ t2) ∧ (t2 ≠ t3) ∧ (t1 ≠ t3) →
  canFormThird t1 t2 t3 →
  t3 = TriangleType.Right :=
by sorry

end NUMINAMATH_CALUDE_sasha_took_right_triangle_l1989_198901


namespace NUMINAMATH_CALUDE_outer_circle_radius_l1989_198993

/-- Given a circular race track with an inner circumference of 440 meters and a width of 14 meters,
    the radius of the outer circle is equal to (440 / (2 * π)) + 14. -/
theorem outer_circle_radius (inner_circumference : ℝ) (track_width : ℝ)
    (h1 : inner_circumference = 440)
    (h2 : track_width = 14) :
    (inner_circumference / (2 * Real.pi) + track_width) = (440 / (2 * Real.pi) + 14) := by
  sorry

end NUMINAMATH_CALUDE_outer_circle_radius_l1989_198993


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l1989_198995

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 4 then a * x - 8 else x^2 - 2 * a * x

-- Define what it means for a function to be increasing
def IsIncreasing (g : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → g x < g y

-- State the theorem
theorem f_increasing_iff_a_in_range (a : ℝ) :
  IsIncreasing (f a) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l1989_198995


namespace NUMINAMATH_CALUDE_tsunami_area_theorem_l1989_198966

/-- Regular tetrahedron with edge length 900 km -/
structure Tetrahedron where
  edge_length : ℝ
  regular : edge_length = 900

/-- Tsunami propagation properties -/
structure Tsunami where
  speed : ℝ
  time : ℝ
  speed_is_300 : speed = 300
  time_is_2 : time = 2

/-- Epicenter location -/
inductive EpicenterLocation
  | FaceCenter
  | EdgeMidpoint

/-- Area covered by tsunami -/
noncomputable def tsunami_area (t : Tetrahedron) (w : Tsunami) (loc : EpicenterLocation) : ℝ :=
  match loc with
  | EpicenterLocation.FaceCenter => 180000 * Real.pi + 270000 * Real.sqrt 3
  | EpicenterLocation.EdgeMidpoint => 720000 * Real.arccos (3/4) + 135000 * Real.sqrt 7

/-- Main theorem -/
theorem tsunami_area_theorem (t : Tetrahedron) (w : Tsunami) :
  (tsunami_area t w EpicenterLocation.FaceCenter = 180000 * Real.pi + 270000 * Real.sqrt 3) ∧
  (tsunami_area t w EpicenterLocation.EdgeMidpoint = 720000 * Real.arccos (3/4) + 135000 * Real.sqrt 7) := by
  sorry


end NUMINAMATH_CALUDE_tsunami_area_theorem_l1989_198966


namespace NUMINAMATH_CALUDE_queen_high_school_teachers_l1989_198946

/-- The number of teachers at Queen High School -/
def num_teachers (total_students : ℕ) (classes_per_student : ℕ) (students_per_class : ℕ) (classes_per_teacher : ℕ) : ℕ :=
  (total_students * classes_per_student) / (students_per_class * classes_per_teacher)

/-- Theorem: There are 72 teachers at Queen High School -/
theorem queen_high_school_teachers :
  num_teachers 1500 6 25 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_queen_high_school_teachers_l1989_198946


namespace NUMINAMATH_CALUDE_pencil_profit_proof_l1989_198917

/-- Proves that selling 1600 pencils results in a profit of $180.00 --/
theorem pencil_profit_proof :
  let total_pencils : ℕ := 2000
  let purchase_price : ℚ := 15/100
  let selling_price : ℚ := 30/100
  let pencils_sold : ℕ := 1600
  let profit : ℚ := pencils_sold * selling_price - total_pencils * purchase_price
  profit = 180 := by
  sorry


end NUMINAMATH_CALUDE_pencil_profit_proof_l1989_198917


namespace NUMINAMATH_CALUDE_greta_letter_difference_greta_letter_difference_proof_l1989_198923

theorem greta_letter_difference : ℕ → ℕ → ℕ → Prop :=
fun greta_letters brother_letters mother_letters =>
  greta_letters > brother_letters ∧
  mother_letters = 2 * (greta_letters + brother_letters) ∧
  greta_letters + brother_letters + mother_letters = 270 ∧
  brother_letters = 40 →
  greta_letters - brother_letters = 10

-- Proof
theorem greta_letter_difference_proof :
  ∃ (greta_letters brother_letters mother_letters : ℕ),
    greta_letter_difference greta_letters brother_letters mother_letters :=
by
  sorry

end NUMINAMATH_CALUDE_greta_letter_difference_greta_letter_difference_proof_l1989_198923


namespace NUMINAMATH_CALUDE_function_composition_equality_l1989_198904

/-- Given two functions f and g, where f is quadratic and g is linear,
    if f(g(x)) = g(f(x)) for all x, then certain conditions on their coefficients must hold. -/
theorem function_composition_equality
  (a b c d e : ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∀ x, g x = d * x + e)
  (h_eq : ∀ x, f (g x) = g (f x)) :
  a * (d - 1) = 0 ∧ a * e = 0 ∧ c - e = a * e^2 :=
by sorry

end NUMINAMATH_CALUDE_function_composition_equality_l1989_198904


namespace NUMINAMATH_CALUDE_increasing_function_equivalence_l1989_198985

/-- A function f is increasing on ℝ -/
def IncreasingOnReals (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem increasing_function_equivalence (f : ℝ → ℝ) (h : IncreasingOnReals f) :
  ∀ a b : ℝ, (a + b ≥ 0 ↔ f a + f b ≥ f (-a) + f (-b)) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_equivalence_l1989_198985


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1989_198926

-- Define the function f
def f (x : ℝ) : ℝ := x^(1/4)

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x > f (8*x - 16)} = {x : ℝ | 2 ≤ x ∧ x < 16/7} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1989_198926


namespace NUMINAMATH_CALUDE_valid_seating_count_l1989_198952

-- Define the set of people
inductive Person : Type
| Alice : Person
| Bob : Person
| Carla : Person
| Derek : Person
| Eric : Person

-- Define a seating arrangement as a function from position to person
def SeatingArrangement := Fin 5 → Person

-- Define the condition that two people cannot sit next to each other
def NotAdjacent (arr : SeatingArrangement) (p1 p2 : Person) : Prop :=
  ∀ i : Fin 4, arr i ≠ p1 ∨ arr (i + 1) ≠ p2

-- Define a valid seating arrangement
def ValidSeating (arr : SeatingArrangement) : Prop :=
  NotAdjacent arr Person.Alice Person.Bob ∧
  NotAdjacent arr Person.Alice Person.Carla ∧
  NotAdjacent arr Person.Derek Person.Eric ∧
  NotAdjacent arr Person.Derek Person.Carla

-- The theorem to prove
theorem valid_seating_count :
  ∃ (arrangements : Finset SeatingArrangement),
    (∀ arr ∈ arrangements, ValidSeating arr) ∧
    (∀ arr, ValidSeating arr → arr ∈ arrangements) ∧
    arrangements.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_valid_seating_count_l1989_198952


namespace NUMINAMATH_CALUDE_females_with_advanced_degrees_l1989_198906

theorem females_with_advanced_degrees 
  (total_employees : ℕ)
  (total_females : ℕ)
  (employees_with_advanced_degrees : ℕ)
  (males_with_college_only : ℕ)
  (h1 : total_employees = 148)
  (h2 : total_females = 92)
  (h3 : employees_with_advanced_degrees = 78)
  (h4 : males_with_college_only = 31)
  : total_females - (total_employees - employees_with_advanced_degrees - males_with_college_only) = 53 := by
  sorry

end NUMINAMATH_CALUDE_females_with_advanced_degrees_l1989_198906


namespace NUMINAMATH_CALUDE_dress_price_difference_l1989_198934

theorem dress_price_difference (original_price : ℝ) : 
  (0.85 * original_price = 85) →
  (original_price - (85 + 0.25 * 85) = -6.25) := by
sorry

end NUMINAMATH_CALUDE_dress_price_difference_l1989_198934


namespace NUMINAMATH_CALUDE_parabola_decreases_left_of_vertex_given_parabola_decreases_left_of_vertex_l1989_198914

/-- Represents a parabola of the form y = (x - h)^2 + k -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  (x - p.h)^2 + p.k

theorem parabola_decreases_left_of_vertex (p : Parabola) :
  ∀ x₁ x₂, x₁ < x₂ → x₂ < p.h → p.y_coord x₁ > p.y_coord x₂ := by
  sorry

/-- The specific parabola y = (x - 2)^2 + 1 -/
def given_parabola : Parabola :=
  { h := 2, k := 1 }

theorem given_parabola_decreases_left_of_vertex :
  ∀ x₁ x₂, x₁ < x₂ → x₂ < 2 → given_parabola.y_coord x₁ > given_parabola.y_coord x₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_decreases_left_of_vertex_given_parabola_decreases_left_of_vertex_l1989_198914


namespace NUMINAMATH_CALUDE_largest_square_tile_l1989_198979

theorem largest_square_tile (board_width board_height : ℕ) 
  (hw : board_width = 17) (hh : board_height = 23) :
  Nat.gcd board_width board_height = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_tile_l1989_198979


namespace NUMINAMATH_CALUDE_star_calculation_l1989_198943

def star (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem star_calculation : star 2 (star 3 (star 1 2)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l1989_198943


namespace NUMINAMATH_CALUDE_cube_opposite_face_l1989_198935

-- Define the faces of the cube
inductive Face : Type
| X | Y | Z | U | V | W

-- Define the adjacency relation
def adjacent : Face → Face → Prop := sorry

-- Define the opposite relation
def opposite : Face → Face → Prop := sorry

-- State the theorem
theorem cube_opposite_face :
  (∀ f : Face, f ≠ Face.X ∧ f ≠ Face.Y → adjacent Face.X f) →
  opposite Face.X Face.Y := by sorry

end NUMINAMATH_CALUDE_cube_opposite_face_l1989_198935


namespace NUMINAMATH_CALUDE_min_additional_coins_l1989_198963

def friends : ℕ := 15
def initial_coins : ℕ := 100

theorem min_additional_coins :
  let required_coins := (friends * (friends + 1)) / 2
  required_coins - initial_coins = 20 := by
sorry

end NUMINAMATH_CALUDE_min_additional_coins_l1989_198963


namespace NUMINAMATH_CALUDE_max_perimeter_rectangle_from_triangles_l1989_198925

theorem max_perimeter_rectangle_from_triangles :
  let num_triangles : ℕ := 60
  let leg1 : ℝ := 2
  let leg2 : ℝ := 3
  let triangle_area : ℝ := (1 / 2) * leg1 * leg2
  let total_area : ℝ := num_triangles * triangle_area
  ∀ a b : ℝ,
    a > 0 → b > 0 →
    a * b = total_area →
    2 * (a + b) ≤ 184 :=
by sorry

end NUMINAMATH_CALUDE_max_perimeter_rectangle_from_triangles_l1989_198925


namespace NUMINAMATH_CALUDE_debate_team_boys_l1989_198902

theorem debate_team_boys (girls : ℕ) (groups : ℕ) (group_size : ℕ) (total : ℕ) (boys : ℕ) : 
  girls = 4 → 
  groups = 8 → 
  group_size = 4 → 
  total = groups * group_size → 
  boys = total - girls → 
  boys = 28 := by
sorry

end NUMINAMATH_CALUDE_debate_team_boys_l1989_198902


namespace NUMINAMATH_CALUDE_right_triangles_qr_length_l1989_198940

theorem right_triangles_qr_length 
  (PQ PR PS : ℝ) 
  (h_PQ : PQ = 15) 
  (h_PR : PR = 21) 
  (h_PS : PS = 33) 
  (h_PQR_right : PQ ^ 2 + PR ^ 2 = (PQ + PR) ^ 2) 
  (h_PRS_right : PR ^ 2 + PS ^ 2 = (PR + PS) ^ 2) : 
  ∃ QR : ℝ, QR ^ 2 = PQ ^ 2 + PR ^ 2 ∧ QR = 3 * Real.sqrt 74 := by
sorry

end NUMINAMATH_CALUDE_right_triangles_qr_length_l1989_198940


namespace NUMINAMATH_CALUDE_value_of_a_l1989_198919

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {a, b, 2}
def B (a b : ℝ) : Set ℝ := {2, b^2, 2*a}

-- State the theorem
theorem value_of_a (a b : ℝ) :
  A a b = B a b → a = 0 ∨ a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1989_198919


namespace NUMINAMATH_CALUDE_division_remainder_l1989_198981

theorem division_remainder : ∃ (q : ℕ), 37 = 8 * q + 5 ∧ 5 < 8 := by sorry

end NUMINAMATH_CALUDE_division_remainder_l1989_198981


namespace NUMINAMATH_CALUDE_divided_square_area_is_eight_l1989_198909

/-- A square with a diagonal divided into three segments -/
structure DividedSquare where
  side : ℝ
  diagonal_length : ℝ
  de : ℝ
  ef : ℝ
  fb : ℝ
  diagonal_sum : de + ef + fb = diagonal_length
  diagonal_pythagoras : 2 * side * side = diagonal_length * diagonal_length

/-- The area of a square with a divided diagonal is 8 -/
theorem divided_square_area_is_eight (s : DividedSquare) 
  (h1 : s.de = 1) (h2 : s.ef = 2) (h3 : s.fb = 1) : s.side * s.side = 8 := by
  sorry

#check divided_square_area_is_eight

end NUMINAMATH_CALUDE_divided_square_area_is_eight_l1989_198909


namespace NUMINAMATH_CALUDE_total_cost_theorem_l1989_198986

/-- Calculates the total cost of items with tax --/
def total_cost_with_tax (prices : List ℝ) (tax_rate : ℝ) : ℝ :=
  let subtotal := prices.sum
  let tax_amount := subtotal * tax_rate
  subtotal + tax_amount

/-- Theorem: The total cost of three items with given prices and 5% tax is $15.75 --/
theorem total_cost_theorem :
  let prices := [4.20, 7.60, 3.20]
  let tax_rate := 0.05
  total_cost_with_tax prices tax_rate = 15.75 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l1989_198986


namespace NUMINAMATH_CALUDE_cable_lengths_theorem_l1989_198954

/-- Given two pieces of cable with specific mass and length relationships,
    prove that their lengths are either (5, 8) meters or (19.5, 22.5) meters. -/
theorem cable_lengths_theorem (mass1 mass2 : ℝ) (length_diff mass_per_meter_diff : ℝ) :
  mass1 = 65 →
  mass2 = 120 →
  length_diff = 3 →
  mass_per_meter_diff = 2 →
  ∃ (l1 l2 : ℝ),
    ((l1 = 5 ∧ l2 = 8) ∨ (l1 = 19.5 ∧ l2 = 22.5)) ∧
    (mass1 / l1 + mass_per_meter_diff) * (l1 + length_diff) = mass2 :=
by sorry

end NUMINAMATH_CALUDE_cable_lengths_theorem_l1989_198954


namespace NUMINAMATH_CALUDE_correct_equation_transformation_l1989_198959

theorem correct_equation_transformation (x : ℝ) : 
  3 * x - (2 - 4 * x) = 5 ↔ 3 * x + 4 * x - 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_transformation_l1989_198959


namespace NUMINAMATH_CALUDE_nina_taller_than_lena_probability_zero_l1989_198988

-- Define the set of friends
inductive Friend : Type
| Masha : Friend
| Nina : Friend
| Lena : Friend
| Olya : Friend

-- Define a height ordering relation
def TallerThan : Friend → Friend → Prop :=
  sorry

-- Define the conditions
axiom all_different_heights :
  ∀ (a b : Friend), a ≠ b → (TallerThan a b ∨ TallerThan b a)

axiom nina_shorter_than_masha :
  TallerThan Friend.Masha Friend.Nina

axiom lena_taller_than_olya :
  TallerThan Friend.Lena Friend.Olya

-- Define the probability function
def Probability (event : Prop) : ℚ :=
  sorry

-- The theorem to prove
theorem nina_taller_than_lena_probability_zero :
  Probability (TallerThan Friend.Nina Friend.Lena) = 0 :=
sorry

end NUMINAMATH_CALUDE_nina_taller_than_lena_probability_zero_l1989_198988


namespace NUMINAMATH_CALUDE_f_x_plus_3_l1989_198982

/-- Given a function f: ℝ → ℝ defined as f(x) = x^2 for all real numbers x,
    prove that f(x + 3) = x^2 + 6x + 9 for all real numbers x. -/
theorem f_x_plus_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x^2) :
  ∀ x : ℝ, f (x + 3) = x^2 + 6*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_f_x_plus_3_l1989_198982


namespace NUMINAMATH_CALUDE_diamond_calculation_l1989_198961

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := Int.natAbs (a + b - 10)

-- Theorem statement
theorem diamond_calculation : diamond 5 (diamond 3 8) = 4 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l1989_198961


namespace NUMINAMATH_CALUDE_sams_age_five_years_ago_l1989_198900

/-- Proves Sam's age 5 years ago given the conditions about John, Sam, and Ted's ages --/
theorem sams_age_five_years_ago (sam_current_age : ℕ) : 
  -- John is 3 times as old as Sam
  (3 * sam_current_age = 3 * sam_current_age) →
  -- In 15 years, John will be twice as old as Sam
  (3 * sam_current_age + 15 = 2 * (sam_current_age + 15)) →
  -- Ted is 5 years younger than Sam
  (sam_current_age - 5 = sam_current_age - 5) →
  -- In 15 years, Ted will be three-fourths the age of Sam
  ((sam_current_age - 5 + 15) * 4 = (sam_current_age + 15) * 3) →
  -- Sam's age 5 years ago was 10
  sam_current_age - 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sams_age_five_years_ago_l1989_198900


namespace NUMINAMATH_CALUDE_alice_twice_bob_age_l1989_198955

theorem alice_twice_bob_age (alice_age bob_age : ℕ) : 
  alice_age = bob_age + 10 →
  alice_age + 5 = 19 →
  ∃ (years : ℕ), (alice_age + years = 2 * (bob_age + years)) ∧ years = 6 :=
by sorry

end NUMINAMATH_CALUDE_alice_twice_bob_age_l1989_198955


namespace NUMINAMATH_CALUDE_complement_intersection_empty_and_range_l1989_198951

-- Define the sets A and B as functions of a
def A (a : ℝ) : Set ℝ :=
  if 3 * a + 1 > 2 then {x : ℝ | 2 < x ∧ x < 3 * a + 1}
  else {x : ℝ | 3 * a + 1 < x ∧ x < 2}

def B (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < a^2 + 2}

-- Define propositions p and q
def p (x : ℝ) (a : ℝ) : Prop := x ∈ A a
def q (x : ℝ) (a : ℝ) : Prop := x ∈ B a

theorem complement_intersection_empty_and_range (a : ℝ) :
  (A a ≠ ∅ ∧ B a ≠ ∅) →
  ((a = 1/3 → (Set.univ \ B a) ∩ A a = ∅) ∧
   (∀ x, p x a → q x a) ↔ (1/3 ≤ a ∧ a ≤ (Real.sqrt 5 - 1) / 2)) :=
sorry

end NUMINAMATH_CALUDE_complement_intersection_empty_and_range_l1989_198951


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l1989_198927

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → A * B * C = 3003 → 
  A + B + C ≤ 45 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l1989_198927


namespace NUMINAMATH_CALUDE_sheila_fewer_acorns_l1989_198913

/-- The number of acorns Shawna, Sheila, and Danny have altogether -/
def total_acorns : ℕ := 80

/-- The number of acorns Shawna has -/
def shawna_acorns : ℕ := 7

/-- The ratio of Sheila's acorns to Shawna's acorns -/
def sheila_ratio : ℕ := 5

/-- The number of acorns Sheila has -/
def sheila_acorns : ℕ := sheila_ratio * shawna_acorns

/-- The number of acorns Danny has -/
def danny_acorns : ℕ := total_acorns - sheila_acorns - shawna_acorns

/-- The difference in acorns between Danny and Sheila -/
def acorn_difference : ℕ := danny_acorns - sheila_acorns

theorem sheila_fewer_acorns : acorn_difference = 3 := by
  sorry

end NUMINAMATH_CALUDE_sheila_fewer_acorns_l1989_198913


namespace NUMINAMATH_CALUDE_arccos_sqrt3_div2_l1989_198930

theorem arccos_sqrt3_div2 : Real.arccos (Real.sqrt 3 / 2) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sqrt3_div2_l1989_198930


namespace NUMINAMATH_CALUDE_binary_to_base5_equivalence_l1989_198960

-- Define the binary number
def binary_num : ℕ := 168  -- 10101000 in binary is 168 in decimal

-- Define the base-5 number
def base5_num : List ℕ := [1, 1, 3, 3]  -- 1133 in base-5

-- Theorem to prove the equivalence
theorem binary_to_base5_equivalence :
  (binary_num : ℕ) = (List.foldl (λ acc d => acc * 5 + d) 0 base5_num) :=
sorry

end NUMINAMATH_CALUDE_binary_to_base5_equivalence_l1989_198960


namespace NUMINAMATH_CALUDE_both_make_basket_l1989_198922

-- Define the probabilities
def prob_A : ℚ := 2/5
def prob_B : ℚ := 1/2

-- Define the theorem
theorem both_make_basket : 
  prob_A * prob_B = 1/5 := by sorry

end NUMINAMATH_CALUDE_both_make_basket_l1989_198922


namespace NUMINAMATH_CALUDE_equation_solutions_l1989_198928

theorem equation_solutions :
  (∃ x : ℚ, 5 * x - 2 * (x - 1) = 3 ∧ x = 1/3) ∧
  (∃ x : ℚ, (3 * x - 2) / 6 = 1 + (x - 1) / 3 ∧ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1989_198928


namespace NUMINAMATH_CALUDE_fraction_calculation_l1989_198970

theorem fraction_calculation (w x y : ℝ) 
  (h1 : w / y = 1 / 5)
  (h2 : (x + y) / y = 2.2) :
  w / x = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1989_198970


namespace NUMINAMATH_CALUDE_distance_AB_is_600_l1989_198998

-- Define the cities
structure City where
  name : String

-- Define the travelers
structure Traveler where
  name : String
  start : City
  destination : City
  travelTime : ℝ
  averageSpeed : ℝ

-- Define the problem setup
def cityA : City := ⟨"A"⟩
def cityB : City := ⟨"B"⟩
def cityC : City := ⟨"C"⟩

def eddy : Traveler := ⟨"Eddy", cityA, cityB, 3, 2⟩
def freddy : Traveler := ⟨"Freddy", cityA, cityC, 3, 1⟩

-- Define the distances
def distanceAC : ℝ := 300

-- Theorem statement
theorem distance_AB_is_600 :
  let distanceAB := eddy.averageSpeed * eddy.travelTime
  distanceAC = freddy.averageSpeed * freddy.travelTime →
  eddy.averageSpeed = 2 * freddy.averageSpeed →
  distanceAB = 600 := by
  sorry

end NUMINAMATH_CALUDE_distance_AB_is_600_l1989_198998


namespace NUMINAMATH_CALUDE_nail_pierces_one_shape_l1989_198942

/-- Represents a shape that can be placed on a rectangular surface --/
structure Shape where
  area : ℝ
  -- Other properties of the shape could be added here

/-- Represents a rectangular box --/
structure Box where
  length : ℝ
  width : ℝ
  center : ℝ × ℝ

/-- Represents the placement of a shape on the box's bottom --/
structure Placement where
  shape : Shape
  position : ℝ × ℝ

/-- Checks if two placements completely cover the box's bottom --/
def covers (b : Box) (p1 p2 : Placement) : Prop := sorry

/-- Checks if a point is inside a placed shape --/
def pointInPlacement (point : ℝ × ℝ) (p : Placement) : Prop := sorry

/-- Main theorem: It's possible to arrange two identical shapes to cover a box's bottom
    such that the center point is in only one of the shapes --/
theorem nail_pierces_one_shape (b : Box) (s : Shape) :
  ∃ (p1 p2 : Placement),
    p1.shape = s ∧ p2.shape = s ∧
    covers b p1 p2 ∧
    (pointInPlacement b.center p1 ↔ ¬pointInPlacement b.center p2) := sorry

end NUMINAMATH_CALUDE_nail_pierces_one_shape_l1989_198942


namespace NUMINAMATH_CALUDE_lines_parallel_lines_perpendicular_l1989_198941

/-- Two lines in the plane --/
structure Lines where
  a : ℝ
  l1 : ℝ → ℝ → ℝ := λ x y => a * x + 2 * y + 6
  l2 : ℝ → ℝ → ℝ := λ x y => x + (a - 1) * y + a^2 - 1

/-- The lines are parallel iff a = -1 --/
theorem lines_parallel (lines : Lines) : 
  (∃ k : ℝ, ∀ x y : ℝ, lines.l1 x y = k * lines.l2 x y) ↔ lines.a = -1 :=
sorry

/-- The lines are perpendicular iff a = 2/3 --/
theorem lines_perpendicular (lines : Lines) :
  (∀ x1 y1 x2 y2 : ℝ, 
    (lines.l1 x1 y1 = 0 ∧ lines.l1 x2 y2 = 0) → 
    (lines.l2 x1 y1 = 0 ∧ lines.l2 x2 y2 = 0) → 
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) * 
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) = 
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))^2) 
  ↔ lines.a = 2/3 :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_lines_perpendicular_l1989_198941


namespace NUMINAMATH_CALUDE_calculate_expression_l1989_198962

theorem calculate_expression : ((28 / (5 + 3 - 6)) * 7) = 98 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1989_198962


namespace NUMINAMATH_CALUDE_randy_tower_blocks_l1989_198933

/-- Given information about Randy's blocks and constructions -/
structure RandysBlocks where
  total : ℕ
  house : ℕ
  tower_and_house : ℕ

/-- The number of blocks Randy used for the tower -/
def blocks_for_tower (r : RandysBlocks) : ℕ :=
  r.tower_and_house - r.house

/-- Theorem stating that Randy used 27 blocks for the tower -/
theorem randy_tower_blocks (r : RandysBlocks)
  (h1 : r.total = 58)
  (h2 : r.house = 53)
  (h3 : r.tower_and_house = 80) :
  blocks_for_tower r = 27 := by
  sorry

end NUMINAMATH_CALUDE_randy_tower_blocks_l1989_198933


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l1989_198990

theorem largest_prime_factor_of_expression : 
  let expression := 18^4 + 3 * 18^2 + 1 - 17^4
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ expression ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ expression → q ≤ p ∧ p = 307 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l1989_198990


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l1989_198921

/-- Scientific notation representation of 185000 -/
def scientific_notation : ℝ := 1.85 * (10 : ℝ) ^ 5

/-- The original number -/
def original_number : ℕ := 185000

theorem scientific_notation_proof : 
  (original_number : ℝ) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l1989_198921


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1989_198907

theorem solution_set_of_inequality (x : ℝ) :
  (1/2 - x) * (x - 1/3) > 0 ↔ 1/3 < x ∧ x < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1989_198907


namespace NUMINAMATH_CALUDE_range_of_f_l1989_198918

def f (x : ℝ) : ℝ := x^2 - 4*x

theorem range_of_f :
  {y | ∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = y} = Set.Icc (-4 : ℝ) 21 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l1989_198918


namespace NUMINAMATH_CALUDE_ice_cream_choices_l1989_198908

/-- The number of ways to choose n items from k types with repetition -/
def choose_with_repetition (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to choose 5 scoops from 14 flavors with repetition -/
theorem ice_cream_choices : choose_with_repetition 5 14 = 3060 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_choices_l1989_198908


namespace NUMINAMATH_CALUDE_card_distribution_theorem_l1989_198976

/-- Represents the number of cards -/
def num_cards : ℕ := 6

/-- Represents the number of envelopes -/
def num_envelopes : ℕ := 3

/-- Represents the number of cards per envelope -/
def cards_per_envelope : ℕ := 2

/-- Calculates the number of ways to distribute cards into envelopes -/
def distribute_cards : ℕ := sorry

theorem card_distribution_theorem : 
  distribute_cards = 18 := by sorry

end NUMINAMATH_CALUDE_card_distribution_theorem_l1989_198976


namespace NUMINAMATH_CALUDE_routes_from_bristol_to_carlisle_l1989_198999

/-- The number of routes from Bristol to Birmingham -/
def bristol_to_birmingham : ℕ := 6

/-- The number of routes from Birmingham to Sheffield -/
def birmingham_to_sheffield : ℕ := 3

/-- The number of routes from Sheffield to Carlisle -/
def sheffield_to_carlisle : ℕ := 2

/-- The total number of routes from Bristol to Carlisle -/
def total_routes : ℕ := bristol_to_birmingham * birmingham_to_sheffield * sheffield_to_carlisle

theorem routes_from_bristol_to_carlisle : total_routes = 36 := by
  sorry

end NUMINAMATH_CALUDE_routes_from_bristol_to_carlisle_l1989_198999


namespace NUMINAMATH_CALUDE_original_raspberry_count_l1989_198936

/-- The number of lemon candies Liam originally had -/
def original_lemon : ℕ := sorry

/-- The number of raspberry candies Liam originally had -/
def original_raspberry : ℕ := sorry

/-- The condition that Liam originally had three times as many raspberry candies as lemon candies -/
axiom original_ratio : original_raspberry = 3 * original_lemon

/-- The condition that after giving away 15 raspberry candies and 5 lemon candies, 
    he has five times as many raspberry candies as lemon candies -/
axiom new_ratio : original_raspberry - 15 = 5 * (original_lemon - 5)

/-- The theorem stating that the original number of raspberry candies is 15 -/
theorem original_raspberry_count : original_raspberry = 15 := by sorry

end NUMINAMATH_CALUDE_original_raspberry_count_l1989_198936


namespace NUMINAMATH_CALUDE_amy_music_files_l1989_198911

/-- Represents the number of music files Amy initially had -/
def initial_music_files : ℕ := sorry

/-- Represents the initial total number of files -/
def initial_total_files : ℕ := initial_music_files + 36

/-- Represents the number of deleted files -/
def deleted_files : ℕ := 48

/-- Represents the number of remaining files after deletion -/
def remaining_files : ℕ := 14

theorem amy_music_files :
  initial_total_files - deleted_files = remaining_files ∧
  initial_music_files = 26 := by
  sorry

end NUMINAMATH_CALUDE_amy_music_files_l1989_198911


namespace NUMINAMATH_CALUDE_division_problem_l1989_198937

theorem division_problem (divisor quotient remainder : ℕ) : 
  divisor = 10 * quotient →
  divisor = 5 * remainder →
  remainder = 46 →
  divisor * quotient + remainder = 5336 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l1989_198937


namespace NUMINAMATH_CALUDE_number_of_students_l1989_198992

-- Define the number of 8th-grade students
variable (x : ℕ)

-- Define the conditions
axiom retail_threshold : x < 250
axiom wholesale_threshold : x + 60 ≥ 250
axiom retail_cost : 240 / x * 240 = 240
axiom wholesale_cost : 260 / (x + 60) * (x + 60) = 260
axiom equal_cost : 260 / (x + 60) * 288 = 240 / x * 240

-- Theorem to prove
theorem number_of_students : x = 200 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l1989_198992


namespace NUMINAMATH_CALUDE_largest_temperature_time_l1989_198989

theorem largest_temperature_time (t : ℝ) : 
  (-t^2 + 10*t + 40 = 60) → t ≤ 5 + Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_temperature_time_l1989_198989


namespace NUMINAMATH_CALUDE_cubic_sum_problem_l1989_198987

theorem cubic_sum_problem (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = -14) :
  a^3 + a^2*b + a*b^2 + b^3 = 265 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_problem_l1989_198987


namespace NUMINAMATH_CALUDE_cost_of_thousand_gum_in_dollars_l1989_198958

/-- The cost of a single piece of gum in cents -/
def cost_of_one_gum : ℕ := 1

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of gum pieces we're calculating the cost for -/
def num_gum_pieces : ℕ := 1000

/-- Theorem: The cost of 1000 pieces of gum in dollars is 10.00 -/
theorem cost_of_thousand_gum_in_dollars : 
  (num_gum_pieces * cost_of_one_gum : ℚ) / cents_per_dollar = 10 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_thousand_gum_in_dollars_l1989_198958


namespace NUMINAMATH_CALUDE_exists_equidistant_point_l1989_198945

/-- A line in a plane --/
structure Line where
  -- Add necessary fields for a line

/-- A point in a plane --/
structure Point where
  -- Add necessary fields for a point

/-- Three lines in a plane --/
def three_lines : Fin 3 → Line := sorry

/-- Condition that at most two lines are parallel --/
def at_most_two_parallel (lines : Fin 3 → Line) : Prop := sorry

/-- A point is equidistant from three lines --/
def equidistant_from_lines (p : Point) (lines : Fin 3 → Line) : Prop := sorry

/-- Main theorem: There always exists a point equidistant from three lines
    given that at most two of them are parallel --/
theorem exists_equidistant_point (lines : Fin 3 → Line) 
  (h : at_most_two_parallel lines) : 
  ∃ (p : Point), equidistant_from_lines p lines := by
  sorry

end NUMINAMATH_CALUDE_exists_equidistant_point_l1989_198945


namespace NUMINAMATH_CALUDE_h_at_two_l1989_198978

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the tangent line function g
def g (x : ℝ) : ℝ := (3*x^2 - 3)*x - 2*x^3

-- Define the function h
def h (x : ℝ) : ℝ := f x - g x

-- Theorem statement
theorem h_at_two : h 2 = 2^3 - 12*2 + 16 := by sorry

end NUMINAMATH_CALUDE_h_at_two_l1989_198978


namespace NUMINAMATH_CALUDE_line_intersection_regions_l1989_198991

theorem line_intersection_regions (h s : ℕ+) : 
  (s + 1) * (s + 2 * h) = 3984 ↔ (h = 176 ∧ s = 10) ∨ (h = 80 ∧ s = 21) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_regions_l1989_198991


namespace NUMINAMATH_CALUDE_motel_pricing_solution_l1989_198964

/-- A motel pricing structure with a flat fee for the first night and a consistent nightly fee thereafter. -/
structure MotelPricing where
  flat_fee : ℝ
  nightly_fee : ℝ

/-- The total cost for a stay at the motel given the number of nights. -/
def total_cost (p : MotelPricing) (nights : ℕ) : ℝ :=
  p.flat_fee + p.nightly_fee * (nights - 1)

theorem motel_pricing_solution :
  ∃ (p : MotelPricing),
    total_cost p 4 = 215 ∧
    total_cost p 3 = 155 ∧
    p.flat_fee = 35 ∧
    p.nightly_fee = 60 := by
  sorry

end NUMINAMATH_CALUDE_motel_pricing_solution_l1989_198964


namespace NUMINAMATH_CALUDE_first_question_percentage_l1989_198997

theorem first_question_percentage (second : ℝ) (neither : ℝ) (both : ℝ)
  (h1 : second = 50)
  (h2 : neither = 20)
  (h3 : both = 33)
  : ∃ first : ℝ, first = 63 ∧ first + second - both + neither = 100 :=
by sorry

end NUMINAMATH_CALUDE_first_question_percentage_l1989_198997


namespace NUMINAMATH_CALUDE_hyperbola_equilateral_triangle_l1989_198956

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define an equilateral triangle
def is_equilateral_triangle (P Q R : ℝ × ℝ) : Prop :=
  let d₁ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
  let d₂ := (Q.1 - R.1)^2 + (Q.2 - R.2)^2
  let d₃ := (R.1 - P.1)^2 + (R.2 - P.2)^2
  d₁ = d₂ ∧ d₂ = d₃

-- Define being on the same branch of the hyperbola
def same_branch (P Q R : ℝ × ℝ) : Prop :=
  (P.1 > 0 ∧ Q.1 > 0 ∧ R.1 > 0) ∨ (P.1 < 0 ∧ Q.1 < 0 ∧ R.1 < 0)

-- Main theorem
theorem hyperbola_equilateral_triangle :
  ∀ P Q R : ℝ × ℝ,
  hyperbola P.1 P.2 →
  hyperbola Q.1 Q.2 →
  hyperbola R.1 R.2 →
  is_equilateral_triangle P Q R →
  (¬ same_branch P Q R) ∧
  (P = (-1, -1) →
   Q.1 > 0 →
   R.1 > 0 →
   Q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧
   R = (2 + Real.sqrt 3, 2 - Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equilateral_triangle_l1989_198956


namespace NUMINAMATH_CALUDE_car_hire_total_amount_l1989_198965

/-- Represents the hire charges for a car -/
structure CarHire where
  hourly_rate : ℕ
  hours_a : ℕ
  hours_b : ℕ
  hours_c : ℕ
  amount_b : ℕ

/-- Calculates the total amount paid for hiring the car -/
def total_amount (hire : CarHire) : ℕ :=
  hire.hourly_rate * (hire.hours_a + hire.hours_b + hire.hours_c)

/-- Theorem stating the total amount paid for the car hire -/
theorem car_hire_total_amount (hire : CarHire)
  (h1 : hire.hours_a = 7)
  (h2 : hire.hours_b = 8)
  (h3 : hire.hours_c = 11)
  (h4 : hire.amount_b = 160)
  (h5 : hire.hourly_rate = hire.amount_b / hire.hours_b) :
  total_amount hire = 520 := by
  sorry

#check car_hire_total_amount

end NUMINAMATH_CALUDE_car_hire_total_amount_l1989_198965


namespace NUMINAMATH_CALUDE_angle_sum_in_triangle_l1989_198931

theorem angle_sum_in_triangle (A B C : ℝ) : 
  A + B + C = 180 →
  A + B = 150 →
  C = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_triangle_l1989_198931


namespace NUMINAMATH_CALUDE_equation_solution_l1989_198972

theorem equation_solution : 
  {x : ℝ | (5 + x) / (7 + x) = (2 + x^2) / (4 + x)} = {1, -2, -3} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1989_198972


namespace NUMINAMATH_CALUDE_quadratic_function_unique_form_l1989_198915

/-- A quadratic function f(x) = x^2 + ax + b that intersects the x-axis at (1,0) 
    and has an axis of symmetry at x = 2 is equal to x^2 - 4x + 3. -/
theorem quadratic_function_unique_form 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = x^2 + a*x + b) 
  (h2 : f 1 = 0) 
  (h3 : ∀ x, f (2 + x) = f (2 - x)) : 
  ∀ x, f x = x^2 - 4*x + 3 := by 
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_form_l1989_198915


namespace NUMINAMATH_CALUDE_coupon1_best_in_range_best_price_is_209_95_l1989_198950

def coupon1_discount (x : ℝ) : ℝ := 0.12 * x

def coupon2_discount : ℝ := 25

def coupon3_discount (x : ℝ) : ℝ := 0.15 * (x - 120)

theorem coupon1_best_in_range (x : ℝ) 
  (h1 : 208.33 < x) (h2 : x < 600) : 
  coupon1_discount x > coupon2_discount ∧ 
  coupon1_discount x > coupon3_discount x := by
  sorry

def listed_prices : List ℝ := [189.95, 209.95, 229.95, 249.95, 269.95]

theorem best_price_is_209_95 : 
  ∃ p ∈ listed_prices, p > 208.33 ∧ p < 600 ∧ 
  ∀ q ∈ listed_prices, q > 208.33 ∧ q < 600 → p ≤ q := by
  sorry

end NUMINAMATH_CALUDE_coupon1_best_in_range_best_price_is_209_95_l1989_198950


namespace NUMINAMATH_CALUDE_expression_simplification_l1989_198938

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  ((a + b)^2 + 2*b^2) / (a^3 - b^3) - 1 / (a - b) + (a + b) / (a^2 + a*b + b^2) *
  (1 / b - 1 / a) = 1 / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1989_198938


namespace NUMINAMATH_CALUDE_ellipse_tangent_line_l1989_198916

def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 2 = 1

def tangent_line (x y : ℝ) : Prop := x / 4 + y / 2 = 1

theorem ellipse_tangent_line : 
  ellipse 2 1 → 
  (∀ x y : ℝ, ellipse x y → (x - 2) * 2 / 4 + (y - 1) * 1 / 2 ≤ 0) →
  (∀ x y : ℝ, tangent_line x y → (x - 2) * 2 / 4 + (y - 1) * 1 / 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_tangent_line_l1989_198916


namespace NUMINAMATH_CALUDE_no_complete_turn_l1989_198924

/-- Represents the position of a bead on a ring as an angle in radians -/
def BeadPosition := ℝ

/-- Represents the state of all beads on the ring -/
def RingState := List BeadPosition

/-- A move that places a bead between its two neighbors -/
def move (state : RingState) (index : Nat) : RingState :=
  sorry

/-- Predicate to check if a bead has made a complete turn -/
def hasMadeCompleteTurn (initialState finalState : RingState) (beadIndex : Nat) : Prop :=
  sorry

/-- The main theorem stating that no bead can make a complete turn -/
theorem no_complete_turn (initialState : RingState) :
    initialState.length = 2009 →
    ∀ (moves : List Nat) (beadIndex : Nat),
      let finalState := moves.foldl move initialState
      ¬ hasMadeCompleteTurn initialState finalState beadIndex :=
  sorry

end NUMINAMATH_CALUDE_no_complete_turn_l1989_198924


namespace NUMINAMATH_CALUDE_joans_kittens_l1989_198973

theorem joans_kittens (initial : ℕ) (additional : ℕ) (total : ℕ) 
  (h1 : initial = 15)
  (h2 : additional = 5)
  (h3 : total = initial + additional) : 
  total = 20 := by
  sorry

end NUMINAMATH_CALUDE_joans_kittens_l1989_198973


namespace NUMINAMATH_CALUDE_inequality_proof_l1989_198969

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ 0) (h4 : a + b + c = 3) : 
  a * b^2 + b * c^2 + c * a^2 ≤ 27/8 ∧ 
  (a * b^2 + b * c^2 + c * a^2 = 27/8 ↔ a = 3/2 ∧ b = 3/2 ∧ c = 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1989_198969


namespace NUMINAMATH_CALUDE_johns_annual_profit_l1989_198944

/-- Calculates the annual profit for John's subletting arrangement -/
def annual_profit (rent_a rent_b rent_c apartment_rent utilities maintenance : ℕ) : ℕ := 
  let total_income := rent_a + rent_b + rent_c
  let total_expenses := apartment_rent + utilities + maintenance
  let monthly_profit := total_income - total_expenses
  12 * monthly_profit

/-- Theorem stating John's annual profit given the specified conditions -/
theorem johns_annual_profit : 
  annual_profit 350 400 450 900 100 50 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_johns_annual_profit_l1989_198944


namespace NUMINAMATH_CALUDE_radical_simplification_l1989_198953

theorem radical_simplification (x : ℝ) (hx : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (98 * x) = 210 * x * Real.sqrt (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l1989_198953


namespace NUMINAMATH_CALUDE_circle_satisfies_conditions_l1989_198967

-- Define the given circle
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 5 = 0

-- Define the sought circle
def sought_circle (x y : ℝ) : Prop :=
  (x - 5)^2 + (y + 1)^2 = 5

-- Define the tangent condition
def is_tangent (c1 c2 : (ℝ → ℝ → Prop)) (x y : ℝ) : Prop :=
  c1 x y ∧ c2 x y ∧ ∃ (m : ℝ), ∀ (dx dy : ℝ),
    (c1 (x + dx) (y + dy) → m * dx = dy) ∧
    (c2 (x + dx) (y + dy) → m * dx = dy)

theorem circle_satisfies_conditions :
  sought_circle 3 (-2) ∧
  is_tangent given_circle sought_circle 0 1 :=
sorry

end NUMINAMATH_CALUDE_circle_satisfies_conditions_l1989_198967


namespace NUMINAMATH_CALUDE_slope_of_line_l1989_198939

theorem slope_of_line (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ ≠ x₂) 
  (h₂ : 1 / x₁ + 2 / y₁ = 0) (h₃ : 1 / x₂ + 2 / y₂ = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -2 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l1989_198939
