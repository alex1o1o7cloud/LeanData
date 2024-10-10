import Mathlib

namespace shifted_function_eq_l553_55382

def original_function (x : ℝ) : ℝ := 2 * x

def vertical_shift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  fun x => f x - shift

def shifted_function : ℝ → ℝ := vertical_shift original_function 2

theorem shifted_function_eq : shifted_function = fun x => 2 * x - 2 := by sorry

end shifted_function_eq_l553_55382


namespace amys_birthday_money_l553_55327

theorem amys_birthday_money (initial : ℕ) (chore_money : ℕ) (final_total : ℕ) : 
  initial = 2 → chore_money = 13 → final_total = 18 → 
  final_total - (initial + chore_money) = 3 := by
  sorry

end amys_birthday_money_l553_55327


namespace smallest_T_for_162_l553_55310

/-- Represents the removal process of tokens in a circle -/
def removeTokens (T : ℕ) : ℕ → ℕ
| 0 => T
| n + 1 => removeTokens (T / 2) n

/-- Checks if a given T results in 162 as the last token -/
def lastTokenIs162 (T : ℕ) : Prop :=
  removeTokens T (Nat.log2 T) = 162

/-- Theorem stating that 209 is the smallest T where the last token is 162 -/
theorem smallest_T_for_162 :
  lastTokenIs162 209 ∧ ∀ k < 209, ¬lastTokenIs162 k :=
sorry

end smallest_T_for_162_l553_55310


namespace quadruple_solution_l553_55388

theorem quadruple_solution :
  ∀ (a b c d : ℝ), 
    a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
    a^2 * b = c →
    b * c^2 = a →
    c * a^2 = b →
    a + b + c = d →
    a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 3 :=
by sorry

end quadruple_solution_l553_55388


namespace complex_multiplication_result_l553_55301

theorem complex_multiplication_result : ∃ (a b : ℝ), (Complex.I + 1) * (2 - Complex.I) = Complex.mk a b ∧ a = 3 ∧ b = 1 := by
  sorry

end complex_multiplication_result_l553_55301


namespace inequality_always_true_l553_55366

theorem inequality_always_true : ∀ x : ℝ, 3 * x - 5 ≤ 12 - 2 * x + x^2 := by sorry

end inequality_always_true_l553_55366


namespace stratified_sample_sophomores_l553_55386

/-- Represents the number of sophomores in a stratified sample -/
def sophomores_in_sample (total_students : ℕ) (total_sophomores : ℕ) (sample_size : ℕ) : ℕ :=
  (sample_size * total_sophomores) / total_students

/-- Theorem: In a school with 1500 students, of which 600 are sophomores,
    a stratified sample of 100 students should include 40 sophomores -/
theorem stratified_sample_sophomores :
  sophomores_in_sample 1500 600 100 = 40 := by
  sorry

end stratified_sample_sophomores_l553_55386


namespace water_remaining_calculation_l553_55319

/-- Calculates the remaining water in a bucket after some has leaked out. -/
def remaining_water (initial : ℚ) (leaked : ℚ) : ℚ :=
  initial - leaked

/-- Theorem stating that given the initial amount and leaked amount, 
    the remaining water is 0.50 gallon. -/
theorem water_remaining_calculation (initial leaked : ℚ) 
  (h1 : initial = 3/4) 
  (h2 : leaked = 1/4) : 
  remaining_water initial leaked = 1/2 := by
  sorry

end water_remaining_calculation_l553_55319


namespace parallel_line_to_hyperbola_asymptote_l553_55326

/-- Given a hyperbola x²/16 - y²/9 = 1 and a line y = kx - 1 parallel to one of its asymptotes, 
    prove that k = 3/4 -/
theorem parallel_line_to_hyperbola_asymptote (k : ℝ) 
  (h1 : k > 0)
  (h2 : ∃ (x y : ℝ), y = k * x - 1 ∧ (x^2 / 16 - y^2 / 9 = 1) ∧ 
        (∀ (x' y' : ℝ), x'^2 / 16 - y'^2 / 9 = 1 → 
          (y - y') / (x - x') = k ∨ (y - y') / (x - x') = -k)) : 
  k = 3/4 := by sorry

end parallel_line_to_hyperbola_asymptote_l553_55326


namespace correct_average_l553_55365

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ 
  incorrect_avg = 46 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 65 →
  (n : ℚ) * incorrect_avg - incorrect_num + correct_num = n * 50 :=
by sorry

end correct_average_l553_55365


namespace solution_set_reciprocal_gt_one_l553_55394

theorem solution_set_reciprocal_gt_one (x : ℝ) : 1 / x > 1 ↔ 0 < x ∧ x < 1 := by
  sorry

end solution_set_reciprocal_gt_one_l553_55394


namespace infinitely_many_non_square_plus_prime_numbers_l553_55342

theorem infinitely_many_non_square_plus_prime_numbers :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ n ∈ S, ¬∃ (m : ℤ) (p : ℕ), Nat.Prime p ∧ n = m^2 + p := by
  sorry

end infinitely_many_non_square_plus_prime_numbers_l553_55342


namespace fraction_equality_l553_55309

theorem fraction_equality (x y b : ℝ) (hb : b ≠ 0) :
  x / b = y / b → x = y := by sorry

end fraction_equality_l553_55309


namespace couponA_provides_greatest_discount_l553_55339

-- Define the coupon discount functions
def couponA (price : Real) : Real := 0.12 * price

def couponB (price : Real) : Real := 25

def couponC (price : Real) : Real := 0.15 * (price - 150)

def couponD (price : Real) : Real := 0.1 * price + 13.5

-- Define the listed price
def listedPrice : Real := 229.95

-- Theorem statement
theorem couponA_provides_greatest_discount :
  couponA listedPrice > couponB listedPrice ∧
  couponA listedPrice > couponC listedPrice ∧
  couponA listedPrice > couponD listedPrice := by
  sorry

end couponA_provides_greatest_discount_l553_55339


namespace factor_and_divisor_properties_l553_55385

theorem factor_and_divisor_properties :
  (∃ n : ℕ, 25 = 5 * n) ∧
  (∃ m : ℕ, 171 = 9 * m) ∧
  ¬(209 % 19 = 0 ∧ 57 % 19 ≠ 0) ∧
  (90 % 30 = 0 ∨ 75 % 30 = 0) ∧
  ¬(51 % 17 = 0 ∧ 68 % 17 ≠ 0) :=
by
  sorry

end factor_and_divisor_properties_l553_55385


namespace parabola_equation_l553_55315

/-- A parabola with vertex at the origin, focus on the y-axis, and a point P(m, 1) on the parabola that is 5 units away from the focus has the standard equation x^2 = 16y. -/
theorem parabola_equation (m : ℝ) : 
  let p : ℝ → ℝ → Prop := λ x y => x^2 = 16*y  -- Standard equation of the parabola
  let focus : ℝ × ℝ := (0, 4)  -- Focus on y-axis, 4 units above origin
  let vertex : ℝ × ℝ := (0, 0)  -- Vertex at origin
  let point_on_parabola : ℝ × ℝ := (m, 1)  -- Given point on parabola
  (vertex = (0, 0)) →  -- Vertex condition
  (focus.1 = 0) →  -- Focus on y-axis condition
  ((point_on_parabola.1 - focus.1)^2 + (point_on_parabola.2 - focus.2)^2 = 5^2) →  -- Distance condition
  p point_on_parabola.1 point_on_parabola.2  -- Conclusion: point satisfies parabola equation
  := by sorry

end parabola_equation_l553_55315


namespace least_number_with_remainder_l553_55314

theorem least_number_with_remainder (n : ℕ) : n = 125 →
  (∃ k : ℕ, n = 20 * k + 5) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m = 20 * k + 5)) :=
by sorry

end least_number_with_remainder_l553_55314


namespace largest_divisible_by_all_less_than_cube_root_l553_55304

theorem largest_divisible_by_all_less_than_cube_root : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 ∧ k < n^(1/3) → n % k = 0) ∧
  (∀ (m : ℕ), m > n → ∃ (j : ℕ), j > 0 ∧ j < m^(1/3) ∧ m % j ≠ 0) ∧
  n = 420 := by
sorry

end largest_divisible_by_all_less_than_cube_root_l553_55304


namespace largest_number_l553_55384

theorem largest_number : 
  let numbers : List ℝ := [0.9791, 0.97019, 0.97909, 0.971, 0.97109]
  ∀ x ∈ numbers, x ≤ 0.9791 := by
  sorry

end largest_number_l553_55384


namespace sector_central_angle_l553_55336

theorem sector_central_angle (r : ℝ) (p : ℝ) (h1 : r = 10) (h2 : p = 45) :
  let θ := (p - 2 * r) / r
  θ = 2.5 := by sorry

end sector_central_angle_l553_55336


namespace derivative_f_at_1_l553_55370

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x^2 - 2*x - 2

-- State the theorem
theorem derivative_f_at_1 :
  HasDerivAt f 3 1 := by sorry

end derivative_f_at_1_l553_55370


namespace lucas_running_speed_l553_55376

theorem lucas_running_speed :
  let eugene_speed : ℚ := 5
  let brianna_speed : ℚ := (3 / 4) * eugene_speed
  let katie_speed : ℚ := (4 / 3) * brianna_speed
  let lucas_speed : ℚ := (5 / 6) * katie_speed
  lucas_speed = 25 / 6 := by sorry

end lucas_running_speed_l553_55376


namespace derivative_of_y_l553_55316

-- Define the function y
def y (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

-- State the theorem
theorem derivative_of_y (x : ℝ) : 
  deriv y x = 4 * x - 2 := by sorry

end derivative_of_y_l553_55316


namespace cube_difference_equals_product_plus_constant_l553_55312

theorem cube_difference_equals_product_plus_constant
  (x y : ℤ)
  (h1 : x ≥ 1)
  (h2 : y ≥ 1)
  (h3 : x^3 - y^3 = x*y + 61) :
  x = 6 ∧ y = 5 := by
sorry

end cube_difference_equals_product_plus_constant_l553_55312


namespace cecilia_B_count_l553_55360

/-- The number of students who received a 'B' in Mrs. Cecilia's class -/
def students_with_B_cecilia (jacob_total : ℕ) (jacob_B : ℕ) (cecilia_total : ℕ) (cecilia_absent : ℕ) : ℕ :=
  let jacob_proportion : ℚ := jacob_B / jacob_total
  let cecilia_present : ℕ := cecilia_total - cecilia_absent
  ⌊(jacob_proportion * cecilia_present : ℚ)⌋₊

theorem cecilia_B_count :
  students_with_B_cecilia 20 12 30 6 = 14 :=
by sorry

end cecilia_B_count_l553_55360


namespace x_not_negative_one_l553_55305

theorem x_not_negative_one (x : ℝ) (h : (x + 1)^0 = 1) : x ≠ -1 := by
  sorry

end x_not_negative_one_l553_55305


namespace count_distinct_values_l553_55399

def is_pythagorean_triple (a b c : ℕ) : Prop := a^2 + b^2 = c^2

def satisfies_conditions (f : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, f n ∣ n^2016) ∧
  (∀ a b c : ℕ, is_pythagorean_triple a b c → f a * f b = f c)

theorem count_distinct_values :
  ∃ (S : Finset ℕ),
    (∀ f : ℕ → ℕ, satisfies_conditions f →
      (f 2014 + f 2 - f 2016) ∈ S) ∧
    S.card = 2^2017 - 1 :=
sorry

end count_distinct_values_l553_55399


namespace hyperbola_m_value_l553_55306

-- Define the hyperbola equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / m - y^2 / (3 + m) = 1

-- Define the focus point
def focus : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem hyperbola_m_value :
  ∀ m : ℝ, (∀ x y : ℝ, hyperbola_equation x y m) → focus.1 = 2 → focus.2 = 0 → m = 1/2 :=
by sorry

end hyperbola_m_value_l553_55306


namespace complex_coordinates_l553_55303

theorem complex_coordinates (z : ℂ) : z = Complex.I * (2 - Complex.I) → (z.re = 1 ∧ z.im = 2) := by
  sorry

end complex_coordinates_l553_55303


namespace second_number_problem_l553_55331

theorem second_number_problem (x y : ℤ) : 
  y = 2 * x - 3 → 
  x + y = 57 → 
  y = 37 := by
sorry

end second_number_problem_l553_55331


namespace parabola_vertex_l553_55390

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 - 4*y + 3*x + 7 = 0

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-1, 2)

/-- Theorem stating that the vertex of the parabola is (-1, 2) -/
theorem parabola_vertex :
  ∀ (x y : ℝ), parabola_equation x y → 
  ∃! (vx vy : ℝ), vx = vertex.1 ∧ vy = vertex.2 ∧
  (∀ (x' y' : ℝ), parabola_equation x' y' → (x' - vx)^2 + (y' - vy)^2 ≤ (x - vx)^2 + (y - vy)^2) :=
by sorry

end parabola_vertex_l553_55390


namespace apple_pyramid_theorem_l553_55391

/-- Calculates the number of apples in a layer of the pyramid -/
def apples_in_layer (base_width : ℕ) (base_length : ℕ) (layer : ℕ) : ℕ :=
  (base_width - layer + 1) * (base_length - layer + 1)

/-- Calculates the total number of apples in the pyramid stack -/
def total_apples (base_width : ℕ) (base_length : ℕ) : ℕ :=
  let num_layers := min base_width base_length
  (List.range num_layers).foldl (fun acc i => acc + apples_in_layer base_width base_length i) 0 + 1

theorem apple_pyramid_theorem :
  total_apples 6 9 = 155 := by
  sorry

#eval total_apples 6 9

end apple_pyramid_theorem_l553_55391


namespace simplify_and_evaluate_l553_55328

theorem simplify_and_evaluate (x y : ℚ) (hx : x = 1/2) (hy : y = -3) :
  (x - 2*y)^2 - (x + y)*(x - y) - 5*y^2 = 6 := by
  sorry

end simplify_and_evaluate_l553_55328


namespace three_lines_intersection_l553_55341

/-- Three distinct lines in 2D space -/
structure ThreeLines where
  a : ℝ
  b : ℝ
  l₁ : ℝ → ℝ → ℝ := λ x y => a * x + 2 * b * y + 3 * (a + b + 1)
  l₂ : ℝ → ℝ → ℝ := λ x y => b * x + 2 * (a + b + 1) * y + 3 * a
  l₃ : ℝ → ℝ → ℝ := λ x y => (a + b + 1) * x + 2 * a * y + 3 * b
  distinct : l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₃ ≠ l₁

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point lying on a line -/
def PointOnLine (p : Point) (l : ℝ → ℝ → ℝ) : Prop :=
  l p.x p.y = 0

/-- Definition of three lines intersecting at a single point -/
def IntersectAtSinglePoint (lines : ThreeLines) : Prop :=
  ∃! p : Point, PointOnLine p lines.l₁ ∧ PointOnLine p lines.l₂ ∧ PointOnLine p lines.l₃

/-- Theorem statement -/
theorem three_lines_intersection (lines : ThreeLines) :
  IntersectAtSinglePoint lines ↔ lines.a + lines.b = -1/2 := by sorry

end three_lines_intersection_l553_55341


namespace rectangular_to_polar_conversion_l553_55397

theorem rectangular_to_polar_conversion :
  let x : ℝ := 8
  let y : ℝ := 2 * Real.sqrt 2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  (r = 6 * Real.sqrt 2) ∧ 
  (θ = Real.arctan (Real.sqrt 2 / 4)) ∧
  (r > 0) ∧ 
  (0 ≤ θ) ∧ 
  (θ < 2 * Real.pi) := by
sorry

end rectangular_to_polar_conversion_l553_55397


namespace victory_circle_count_l553_55347

/-- Represents the different types of medals -/
inductive Medal
  | Gold
  | Silver
  | Bronze
  | Titanium
  | Copper

/-- Represents a runner in the race -/
structure Runner :=
  (position : Nat)
  (medal : Option Medal)

/-- Represents a victory circle configuration -/
def VictoryCircle := List Runner

/-- The number of runners in the race -/
def num_runners : Nat := 8

/-- The maximum number of medals that can be awarded -/
def max_medals : Nat := 5

/-- The minimum number of medals that can be awarded -/
def min_medals : Nat := 3

/-- Generates all possible victory circles for the given scenarios -/
def generate_victory_circles : List VictoryCircle := sorry

/-- Counts the number of unique victory circles -/
def count_victory_circles (circles : List VictoryCircle) : Nat := sorry

/-- Main theorem: The number of different victory circles is 28 -/
theorem victory_circle_count :
  count_victory_circles generate_victory_circles = 28 := by sorry

end victory_circle_count_l553_55347


namespace baker_revenue_l553_55318

/-- The intended revenue for a baker selling birthday cakes -/
theorem baker_revenue (n : ℝ) : 
  (∀ (reduced_price : ℝ), reduced_price = 0.8 * n → 10 * reduced_price = 8 * n) →
  8 * n = 8 * n := by sorry

end baker_revenue_l553_55318


namespace parallelogram_side_length_l553_55335

/-- Represents a parallelogram with side lengths -/
structure Parallelogram where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ

/-- The property that opposite sides of a parallelogram are equal -/
def Parallelogram.oppositeSidesEqual (p : Parallelogram) : Prop :=
  p.ab = p.cd ∧ p.bc = p.da

/-- The theorem to be proved -/
theorem parallelogram_side_length 
  (p : Parallelogram) 
  (h1 : p.oppositeSidesEqual) 
  (h2 : p.ab + p.bc + p.cd + p.da = 14) 
  (h3 : p.da = 5) : 
  p.ab = 2 := by
  sorry

end parallelogram_side_length_l553_55335


namespace larger_number_problem_l553_55307

theorem larger_number_problem (L S : ℕ) (hL : L > S) : 
  L - S = 1365 → L = 6 * S + 15 → L = 1635 := by
  sorry

end larger_number_problem_l553_55307


namespace student_count_difference_l553_55379

/-- Represents the number of students in each grade level -/
structure StudentCounts where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- The problem statement -/
theorem student_count_difference (counts : StudentCounts) : 
  counts.freshmen + counts.sophomores + counts.juniors + counts.seniors = 800 →
  counts.juniors = 208 →
  counts.sophomores = 200 →
  counts.seniors = 160 →
  counts.freshmen - counts.sophomores = 32 := by
  sorry

end student_count_difference_l553_55379


namespace root_sum_squares_reciprocal_l553_55356

theorem root_sum_squares_reciprocal (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 49/36 := by
sorry

end root_sum_squares_reciprocal_l553_55356


namespace extremum_condition_l553_55329

open Real

-- Define a differentiable function
variable (f : ℝ → ℝ) (hf : Differentiable ℝ f)

-- Define the concept of an extremum
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ ∀ x, f x ≥ f x₀

-- Theorem statement
theorem extremum_condition (x₀ : ℝ) :
  (HasExtremumAt f x₀ → deriv f x₀ = 0) ∧
  ∃ g : ℝ → ℝ, Differentiable ℝ g ∧ deriv g 0 = 0 ∧ ¬HasExtremumAt g 0 := by
  sorry

end extremum_condition_l553_55329


namespace divisor_is_one_l553_55344

theorem divisor_is_one (x d : ℕ) (k n : ℤ) : 
  x % d = 5 →
  (x + 17) % 41 = 22 →
  x = k * d + 5 →
  x = 41 * n + 5 →
  d = 1 := by
sorry

end divisor_is_one_l553_55344


namespace arithmetic_sequence_sum_property_not_necessary_condition_l553_55332

def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

theorem arithmetic_sequence_sum_property :
  ∀ (a b c d : ℝ), is_arithmetic_sequence a b c d → a + d = b + c :=
sorry

theorem not_necessary_condition :
  ∃ (a b c d : ℝ), a + d = b + c ∧ ¬(is_arithmetic_sequence a b c d) :=
sorry

end arithmetic_sequence_sum_property_not_necessary_condition_l553_55332


namespace gray_opposite_black_l553_55337

-- Define the colors
inductive Color
| A -- Aqua
| B -- Black
| C -- Crimson
| D -- Dark Blue
| E -- Emerald
| F -- Fuchsia
| G -- Gray
| H -- Hazel

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  faces : List Face
  adjacent : Color → Color → Prop
  opposite : Color → Color → Prop

-- Define the problem conditions
axiom cube_has_eight_faces : ∀ (c : Cube), c.faces.length = 8

axiom aqua_adjacent_to_dark_blue_and_emerald : 
  ∀ (c : Cube), c.adjacent Color.A Color.D ∧ c.adjacent Color.A Color.E

-- The theorem to prove
theorem gray_opposite_black (c : Cube) : c.opposite Color.G Color.B := by
  sorry


end gray_opposite_black_l553_55337


namespace rectangle_problem_l553_55364

theorem rectangle_problem (x : ℝ) :
  (∃ a b : ℝ, 
    a > 0 ∧ b > 0 ∧
    a = 2 * b ∧
    2 * (a + b) = x ∧
    a * b = x) →
  x = 18 := by
sorry

end rectangle_problem_l553_55364


namespace board_number_after_60_minutes_l553_55308

/-- Calculates the product of digits of a natural number -/
def productOfDigits (n : ℕ) : ℕ := sorry

/-- Applies the transformation rule to a number -/
def transform (n : ℕ) : ℕ := productOfDigits n + 12

/-- Applies the transformation n times to the initial number -/
def applyNTimes (initial : ℕ) (n : ℕ) : ℕ := sorry

theorem board_number_after_60_minutes :
  applyNTimes 27 60 = 14 := by sorry

end board_number_after_60_minutes_l553_55308


namespace arithmetic_mean_of_special_set_l553_55372

theorem arithmetic_mean_of_special_set : 
  let S : Finset ℕ := Finset.range 9
  let special_number (n : ℕ) : ℕ := n * ((10^n - 1) / 9)
  let sum_of_set : ℕ := S.sum special_number
  sum_of_set / 9 = 123456790 := by sorry

end arithmetic_mean_of_special_set_l553_55372


namespace functional_equation_solution_l553_55358

/-- A function f: ℝ⁺* → ℝ⁺* satisfying the functional equation f(x) f(y f(x)) = f(x + y) for all x, y > 0 -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f x > 0 → f (y * f x) > 0 → f x * f (y * f x) = f (x + y)

/-- The theorem stating that functions satisfying the given functional equation
    are either of the form f(x) = 1/(1 + ax) for some a > 0, or f(x) = 1 -/
theorem functional_equation_solution (f : ℝ → ℝ) :
  FunctionalEquation f →
  (∃ a : ℝ, a > 0 ∧ ∀ x, x > 0 → f x = 1 / (1 + a * x)) ∨
  (∀ x, x > 0 → f x = 1) :=
sorry

end functional_equation_solution_l553_55358


namespace newton_method_convergence_l553_55322

noncomputable def newtonSequence : ℕ → ℝ
  | 0 => 2
  | n + 1 => (newtonSequence n ^ 2 + 2) / (2 * newtonSequence n)

theorem newton_method_convergence :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |newtonSequence n - Real.sqrt 2| < ε :=
by sorry

end newton_method_convergence_l553_55322


namespace bicycle_sampling_is_systematic_l553_55355

-- Define the sampling method
structure SamplingMethod where
  location : String
  selectionCriteria : String

-- Define systematic sampling
def isSystematicSampling (method : SamplingMethod) : Prop :=
  method.location = "main road" ∧ 
  method.selectionCriteria = "6-digit license plate numbers"

-- Define the specific sampling method used in the problem
def bicycleSamplingMethod : SamplingMethod :=
  { location := "main road"
  , selectionCriteria := "6-digit license plate numbers" }

-- Theorem statement
theorem bicycle_sampling_is_systematic :
  isSystematicSampling bicycleSamplingMethod :=
by sorry


end bicycle_sampling_is_systematic_l553_55355


namespace ten_factorial_divided_by_four_factorial_l553_55348

theorem ten_factorial_divided_by_four_factorial :
  (10 : ℕ).factorial / (4 : ℕ).factorial = 151200 := by
  have h1 : (10 : ℕ).factorial = 3628800 := by sorry
  sorry

end ten_factorial_divided_by_four_factorial_l553_55348


namespace students_present_l553_55392

theorem students_present (total : ℕ) (absent_percent : ℚ) (present : ℕ) : 
  total = 50 → 
  absent_percent = 12 / 100 → 
  present = total - (total * (absent_percent : ℚ)).floor → 
  present = 44 := by
sorry

end students_present_l553_55392


namespace initial_investment_rate_is_five_percent_l553_55374

/-- Proves that given specific investment conditions, the initial investment rate is 5% --/
theorem initial_investment_rate_is_five_percent
  (initial_investment : ℝ)
  (additional_investment : ℝ)
  (additional_rate : ℝ)
  (total_income_rate : ℝ)
  (h1 : initial_investment = 2800)
  (h2 : additional_investment = 1400)
  (h3 : additional_rate = 0.08)
  (h4 : total_income_rate = 0.06)
  (h5 : initial_investment * x + additional_investment * additional_rate = 
        (initial_investment + additional_investment) * total_income_rate) :
  x = 0.05 := by
  sorry

end initial_investment_rate_is_five_percent_l553_55374


namespace composite_sum_of_fourth_power_and_64_power_l553_55354

theorem composite_sum_of_fourth_power_and_64_power (n : ℕ) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 64^n = a * b :=
sorry

end composite_sum_of_fourth_power_and_64_power_l553_55354


namespace part_I_part_II_l553_55378

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

-- Define the complement of A relative to ℝ
def C_R_A : Set ℝ := {x | x ≤ 3 ∨ x ≥ 7}

-- Theorem for part (I)
theorem part_I : (C_R_A ∩ B) = {x | (2 < x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

-- Theorem for part (II)
theorem part_II (a : ℝ) : C a ⊆ (A ∪ B) → a ≤ 3 := by sorry

end part_I_part_II_l553_55378


namespace sqrt_equation_l553_55333

theorem sqrt_equation (n : ℝ) : Real.sqrt (8 + n) = 9 → n = 73 := by
  sorry

end sqrt_equation_l553_55333


namespace pyramid_volume_in_cylinder_l553_55302

/-- Given a cylinder of volume V and a pyramid inscribed in it such that:
    - The base of the pyramid is an isosceles triangle with angle α between equal sides
    - The pyramid's base is inscribed in the base of the cylinder
    - The pyramid's apex coincides with the midpoint of one of the cylinder's generatrices
    Then the volume of the pyramid is (V / (6π)) * sin(α) * cos²(α/2) -/
theorem pyramid_volume_in_cylinder (V : ℝ) (α : ℝ) : ℝ := by
  sorry

end pyramid_volume_in_cylinder_l553_55302


namespace base_6_divisibility_l553_55387

def base_6_to_decimal (y : ℕ) : ℕ := 2 * 6^3 + 4 * 6^2 + y * 6 + 2

def is_valid_base_6_digit (y : ℕ) : Prop := y ≤ 5

theorem base_6_divisibility (y : ℕ) : 
  is_valid_base_6_digit y → (base_6_to_decimal y % 13 = 0 ↔ y = 3) := by
  sorry

end base_6_divisibility_l553_55387


namespace arithmetic_sequence_problem_l553_55346

/-- An arithmetic sequence {a_n} with the given properties -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 1/3)
  (h3 : a 2 + a 5 = 4)
  (h4 : ∃ n : ℕ, a n = 27) :
  ∃ n : ℕ, n = 9 ∧ a n = 27 :=
sorry

end arithmetic_sequence_problem_l553_55346


namespace probability_of_defective_product_l553_55352

/-- Given a product with three grades (first-grade, second-grade, and defective),
    prove that the probability of selecting a defective product is 0.05,
    given the probabilities of selecting first-grade and second-grade products. -/
theorem probability_of_defective_product
  (p_first : ℝ)
  (p_second : ℝ)
  (h_first : p_first = 0.65)
  (h_second : p_second = 0.3)
  (h_nonneg_first : 0 ≤ p_first)
  (h_nonneg_second : 0 ≤ p_second)
  (h_sum_le_one : p_first + p_second ≤ 1) :
  1 - (p_first + p_second) = 0.05 := by
sorry

end probability_of_defective_product_l553_55352


namespace sum_in_base8_l553_55375

def base8_to_decimal (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.foldr (fun d acc => acc * 8 + d) 0

theorem sum_in_base8 :
  let a := base8_to_decimal 245
  let b := base8_to_decimal 174
  let c := base8_to_decimal 354
  let sum := a + b + c
  base8_to_decimal 1015 = sum := by
sorry

end sum_in_base8_l553_55375


namespace no_maximum_value_l553_55350

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def symmetric_about_point (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (2*a - x) = 2*b - f x

theorem no_maximum_value (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_sym : symmetric_about_point f 1 1) : 
  ¬ ∃ M, ∀ x, f x ≤ M :=
sorry

end no_maximum_value_l553_55350


namespace joan_cake_flour_l553_55330

theorem joan_cake_flour (recipe_total : ℕ) (already_added : ℕ) (h1 : recipe_total = 7) (h2 : already_added = 3) :
  recipe_total - already_added = 4 := by
  sorry

end joan_cake_flour_l553_55330


namespace min_value_expression_l553_55369

theorem min_value_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + y = 2) :
  ((x + 1)^2 + 3) / (x + 2) + y^2 / (y + 1) ≥ 14/5 := by
  sorry

end min_value_expression_l553_55369


namespace correspondence_C_is_mapping_l553_55377

def is_mapping (A B : Type) (f : A → B) : Prop :=
  ∀ x : A, ∃! y : B, f x = y

theorem correspondence_C_is_mapping :
  let A := Nat
  let B := { x : Int // x = -1 ∨ x = 0 ∨ x = 1 }
  let f : A → B := λ x => ⟨(-1)^x, by sorry⟩
  is_mapping A B f := by sorry

end correspondence_C_is_mapping_l553_55377


namespace travis_payment_l553_55371

/-- Calculates the payment for Travis given the specified conditions --/
def calculate_payment (total_bowls : ℕ) (fixed_fee : ℕ) (safe_delivery_fee : ℕ) (penalty : ℕ) (lost_bowls : ℕ) (broken_bowls : ℕ) : ℕ :=
  let damaged_bowls := lost_bowls + broken_bowls
  let safe_bowls := total_bowls - damaged_bowls
  let safe_delivery_payment := safe_bowls * safe_delivery_fee
  let total_payment := safe_delivery_payment + fixed_fee
  let penalty_amount := damaged_bowls * penalty
  total_payment - penalty_amount

/-- Theorem stating that Travis should be paid $1825 given the specified conditions --/
theorem travis_payment :
  calculate_payment 638 100 3 4 12 15 = 1825 := by
  sorry

end travis_payment_l553_55371


namespace ken_steak_purchase_l553_55313

/-- The cost of one pound of steak, given the conditions of Ken's purchase -/
def steak_cost (total_pounds : ℕ) (paid : ℕ) (change : ℕ) : ℕ :=
  (paid - change) / total_pounds

theorem ken_steak_purchase :
  steak_cost 2 20 6 = 7 := by
  sorry

end ken_steak_purchase_l553_55313


namespace round_trip_speed_calculation_l553_55383

/-- Proves that given a round trip with total distance 72 miles, total time 7 hours,
    and return speed 18 miles per hour, the outbound speed is 7.2 miles per hour. -/
theorem round_trip_speed_calculation (total_distance : ℝ) (total_time : ℝ) (return_speed : ℝ) :
  total_distance = 72 ∧ total_time = 7 ∧ return_speed = 18 →
  ∃ outbound_speed : ℝ,
    outbound_speed = 7.2 ∧
    total_distance / 2 / outbound_speed + total_distance / 2 / return_speed = total_time :=
by sorry

end round_trip_speed_calculation_l553_55383


namespace geometric_sequence_common_ratio_sum_l553_55349

theorem geometric_sequence_common_ratio_sum 
  (k : ℝ) (p r : ℝ) (h_distinct : p ≠ r) (h_nonzero : k ≠ 0) 
  (h_equation : k * p^2 - k * r^2 = 3 * (k * p - k * r)) : 
  p + r = 3 := by sorry

end geometric_sequence_common_ratio_sum_l553_55349


namespace robotics_club_theorem_l553_55321

theorem robotics_club_theorem (total : ℕ) (cs : ℕ) (eng : ℕ) (both : ℕ) 
  (h1 : total = 120)
  (h2 : cs = 75)
  (h3 : eng = 50)
  (h4 : both = 10) :
  total - (cs + eng - both) = 5 := by
  sorry

end robotics_club_theorem_l553_55321


namespace certain_number_problem_l553_55380

theorem certain_number_problem (x : ℝ) : 
  (0.20 * x) - (1/3) * (0.20 * x) = 24 → x = 180 := by
  sorry

end certain_number_problem_l553_55380


namespace triple_solution_l553_55368

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def is_solution (a b c : ℕ+) : Prop :=
  (a.val > 0 ∧ b.val > 0 ∧ c.val > 0) ∧
  is_integer ((a + b : ℚ)^4 / c + (b + c : ℚ)^4 / a + (c + a : ℚ)^4 / b) ∧
  Nat.Prime (a + b + c)

theorem triple_solution :
  ∀ a b c : ℕ+, is_solution a b c ↔
    ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (2, 2, 1) ∨ (a, b, c) = (6, 3, 2)) ∨
    ((a, b, c) = (1, 2, 2) ∨ (a, b, c) = (2, 1, 2) ∨ (a, b, c) = (2, 2, 1)) ∨
    ((a, b, c) = (6, 3, 2) ∨ (a, b, c) = (6, 2, 3) ∨ (a, b, c) = (3, 6, 2) ∨
     (a, b, c) = (3, 2, 6) ∨ (a, b, c) = (2, 6, 3) ∨ (a, b, c) = (2, 3, 6)) :=
by sorry

end triple_solution_l553_55368


namespace max_pieces_of_pie_l553_55381

def is_valid_assignment (p k u s o i r g : ℕ) : Prop :=
  p ≠ k ∧ p ≠ u ∧ p ≠ s ∧ p ≠ o ∧ p ≠ i ∧ p ≠ r ∧ p ≠ g ∧
  k ≠ u ∧ k ≠ s ∧ k ≠ o ∧ k ≠ i ∧ k ≠ r ∧ k ≠ g ∧
  u ≠ s ∧ u ≠ o ∧ u ≠ i ∧ u ≠ r ∧ u ≠ g ∧
  s ≠ o ∧ s ≠ i ∧ s ≠ r ∧ s ≠ g ∧
  o ≠ i ∧ o ≠ r ∧ o ≠ g ∧
  i ≠ r ∧ i ≠ g ∧
  r ≠ g ∧
  p ≠ 0 ∧ k ≠ 0

def pirog (p i r o g : ℕ) : ℕ := p * 10000 + i * 1000 + r * 100 + o * 10 + g

def kusok (k u s o k : ℕ) : ℕ := k * 10000 + u * 1000 + s * 100 + o * 10 + k

theorem max_pieces_of_pie :
  ∀ p i r o g k u s n,
    is_valid_assignment p k u s o i r g →
    pirog p i r o g = n * kusok k u s o k →
    n ≤ 7 :=
sorry

end max_pieces_of_pie_l553_55381


namespace relationship_x_y_l553_55362

theorem relationship_x_y (x y : ℝ) 
  (h1 : 3 * x - 2 * y > 4 * x + 1) 
  (h2 : 2 * x + 3 * y < 5 * y - 2) : 
  x < 1 - y := by
sorry

end relationship_x_y_l553_55362


namespace at_least_one_red_certain_l553_55325

/-- Represents the number of red balls in the pocket -/
def num_red_balls : ℕ := 2

/-- Represents the number of white balls in the pocket -/
def num_white_balls : ℕ := 1

/-- Represents the total number of balls in the pocket -/
def total_balls : ℕ := num_red_balls + num_white_balls

/-- Represents the number of balls drawn from the pocket -/
def num_drawn : ℕ := 2

/-- Theorem stating that drawing at least one red ball when drawing 2 balls
    from a pocket containing 2 red balls and 1 white ball is a certain event -/
theorem at_least_one_red_certain :
  (num_red_balls.choose num_drawn + num_red_balls.choose (num_drawn - 1) * num_white_balls.choose 1) / total_balls.choose num_drawn = 1 :=
sorry

end at_least_one_red_certain_l553_55325


namespace square_area_from_diagonal_l553_55334

/-- Given a square with diagonal length 10√2 cm, its area is 100 cm². -/
theorem square_area_from_diagonal (diagonal : ℝ) (area : ℝ) : 
  diagonal = 10 * Real.sqrt 2 → area = diagonal ^ 2 / 2 → area = 100 := by sorry

end square_area_from_diagonal_l553_55334


namespace apples_sale_theorem_l553_55345

/-- Calculate the total money made from selling boxes of apples -/
def total_money_from_apples (total_apples : ℕ) (apples_per_box : ℕ) (price_per_box : ℕ) : ℕ :=
  ((total_apples / apples_per_box) * price_per_box)

/-- Theorem: Given 275 apples, with 20 apples per box sold at 8,000 won each,
    the total money made from selling all full boxes is 104,000 won -/
theorem apples_sale_theorem :
  total_money_from_apples 275 20 8000 = 104000 := by
  sorry

end apples_sale_theorem_l553_55345


namespace inequality_equivalence_l553_55373

theorem inequality_equivalence (x : ℝ) : 
  (2*x + 3)/(3*x + 5) > (4*x + 1)/(x + 4) ↔ -4 < x ∧ x < -5/3 := by
  sorry

end inequality_equivalence_l553_55373


namespace path_order_paths_through_A_paths_through_B_total_paths_correct_l553_55351

-- Define the grid points
inductive GridPoint
| X | Y | A | B | C | D | E | F | G

-- Define a function to count paths through a point
def pathsThrough (p : GridPoint) : ℕ := sorry

-- Total number of paths from X to Y
def totalPaths : ℕ := 924

-- Theorem stating the order of points based on number of paths
theorem path_order :
  pathsThrough GridPoint.A > pathsThrough GridPoint.F ∧
  pathsThrough GridPoint.F > pathsThrough GridPoint.C ∧
  pathsThrough GridPoint.C > pathsThrough GridPoint.G ∧
  pathsThrough GridPoint.G > pathsThrough GridPoint.E ∧
  pathsThrough GridPoint.E > pathsThrough GridPoint.D ∧
  pathsThrough GridPoint.D > pathsThrough GridPoint.B :=
by sorry

-- Theorem stating that the sum of paths through A and the point below X equals totalPaths
theorem paths_through_A :
  pathsThrough GridPoint.A = totalPaths / 2 :=
by sorry

-- Theorem stating that there's only one path through B
theorem paths_through_B :
  pathsThrough GridPoint.B = 1 :=
by sorry

-- Theorem stating that the total number of paths is correct
theorem total_paths_correct :
  (pathsThrough GridPoint.A) * 2 = totalPaths :=
by sorry

end path_order_paths_through_A_paths_through_B_total_paths_correct_l553_55351


namespace algebraic_expression_simplification_l553_55338

theorem algebraic_expression_simplification (a : ℝ) :
  a = 2 * Real.sin (60 * π / 180) + 3 →
  (a + 1) / (a - 3) - (a - 3) / (a + 2) / ((a^2 - 6*a + 9) / (a^2 - 4)) = Real.sqrt 3 := by
  sorry

end algebraic_expression_simplification_l553_55338


namespace emmett_jumping_jacks_l553_55311

/-- The number of jumping jacks Emmett did -/
def jumping_jacks : ℕ := sorry

/-- The number of pushups Emmett did -/
def pushups : ℕ := 8

/-- The number of situps Emmett did -/
def situps : ℕ := 20

/-- The total number of exercises Emmett did -/
def total_exercises : ℕ := jumping_jacks + pushups + situps

/-- The percentage of exercises that were pushups -/
def pushup_percentage : ℚ := 1/5

theorem emmett_jumping_jacks : 
  jumping_jacks = 12 :=
by
  sorry

end emmett_jumping_jacks_l553_55311


namespace smallest_number_in_specific_set_l553_55324

theorem smallest_number_in_specific_set (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Three positive integers
  (a + b + c) / 3 = 30 →   -- Arithmetic mean is 30
  (min (max a b) (max b c)) = 31 →  -- Median is 31
  max a (max b c) = 31 + 8 →  -- Largest number is 8 more than median
  min a (min b c) = 20 := by  -- Smallest number is 20
sorry

end smallest_number_in_specific_set_l553_55324


namespace janes_shopping_theorem_l553_55393

theorem janes_shopping_theorem :
  ∀ (s f : ℕ),
  s + f = 7 →
  (90 * s + 60 * f) % 100 = 0 →
  s = 4 :=
by sorry

end janes_shopping_theorem_l553_55393


namespace negative_ten_meters_westward_l553_55363

-- Define the direction as an enumeration
inductive Direction
  | East
  | West

-- Define a function to convert a signed distance to a direction and magnitude
def interpretDistance (d : ℤ) : Direction × ℕ :=
  if d ≥ 0 then (Direction.East, d.natAbs) else (Direction.West, d.natAbs)

-- State the theorem
theorem negative_ten_meters_westward :
  interpretDistance (-10) = (Direction.West, 10) := by
  sorry

end negative_ten_meters_westward_l553_55363


namespace polynomial_expansion_l553_55320

theorem polynomial_expansion :
  ∀ x : ℝ, (3 * x - 2) * (2 * x^2 + 4 * x - 6) = 6 * x^3 + 8 * x^2 - 26 * x + 12 := by
  sorry

end polynomial_expansion_l553_55320


namespace percentage_problem_l553_55396

theorem percentage_problem (a b c : ℝ) : 
  a = 0.8 * b → 
  c = 1.4 * b → 
  c - a = 72 → 
  a = 96 ∧ b = 120 ∧ c = 168 := by
  sorry

end percentage_problem_l553_55396


namespace strawberry_distribution_l553_55389

theorem strawberry_distribution (initial : ℕ) (additional : ℕ) (boxes : ℕ) 
  (h1 : initial = 42)
  (h2 : additional = 78)
  (h3 : boxes = 6)
  (h4 : boxes ≠ 0) :
  (initial + additional) / boxes = 20 := by
  sorry

end strawberry_distribution_l553_55389


namespace circle_area_ratio_l553_55323

/-- Given two circles X and Y, where an arc of 60° on X has the same length as an arc of 20° on Y,
    the ratio of the area of X to the area of Y is 9. -/
theorem circle_area_ratio (X Y : Real) (hX : X > 0) (hY : Y > 0) 
  (h : X * (60 / 360) = Y * (20 / 360)) : 
  (X^2 * Real.pi) / (Y^2 * Real.pi) = 9 := by
  sorry

end circle_area_ratio_l553_55323


namespace expression_evaluation_l553_55317

theorem expression_evaluation :
  let x : ℚ := -3
  let numerator := 7 + x * (4 + x) - 4^2
  let denominator := x - 4 + x^2
  numerator / denominator = -6 :=
by sorry

end expression_evaluation_l553_55317


namespace binomial_20_4_l553_55395

theorem binomial_20_4 : Nat.choose 20 4 = 4845 := by sorry

end binomial_20_4_l553_55395


namespace max_books_borrowed_l553_55398

theorem max_books_borrowed (total_students : Nat) (zero_books : Nat) (one_book : Nat) (two_books : Nat) 
  (avg_books : Nat) (h1 : total_students = 20) (h2 : zero_books = 3) (h3 : one_book = 9) 
  (h4 : two_books = 4) (h5 : avg_books = 2) : ∃ (max_books : Nat), max_books = 14 ∧ 
  max_books = total_students * avg_books - (zero_books * 0 + one_book * 1 + two_books * 2 + 
  (total_students - zero_books - one_book - two_books - 1) * 3) := by
  sorry

end max_books_borrowed_l553_55398


namespace player5_score_breakdown_l553_55357

/-- Represents the scoring breakdown for a basketball player -/
structure PlayerScore where
  threes : Nat
  twos : Nat
  frees : Nat

/-- Calculates the total points scored by a player -/
def totalPoints (score : PlayerScore) : Nat :=
  3 * score.threes + 2 * score.twos + score.frees

theorem player5_score_breakdown :
  ∀ (team_total : Nat) (other_players_total : Nat),
    team_total = 75 →
    other_players_total = 61 →
    ∃ (score : PlayerScore),
      totalPoints score = team_total - other_players_total ∧
      score.threes ≥ 2 ∧
      score.twos ≥ 1 ∧
      score.frees ≤ 4 ∧
      score.threes = 2 ∧
      score.twos = 2 ∧
      score.frees = 4 :=
by sorry

end player5_score_breakdown_l553_55357


namespace statements_evaluation_l553_55353

theorem statements_evaluation :
  (∃ a b : ℝ, a > b ∧ ¬(a^2 > b^2)) ∧
  (∀ a b : ℝ, |a| > |b| → a^2 > b^2) ∧
  (∃ a b c : ℝ, a > b ∧ ¬(a*c^2 > b*c^2)) ∧
  (∀ a b : ℝ, a > b ∧ b > 0 → 1/a < 1/b) := by
  sorry


end statements_evaluation_l553_55353


namespace sqrt_x_plus_inverse_sqrt_x_l553_55361

theorem sqrt_x_plus_inverse_sqrt_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end sqrt_x_plus_inverse_sqrt_x_l553_55361


namespace existence_of_multiple_factorizations_l553_55343

def V_n (n : ℕ) := {m : ℕ | ∃ k : ℕ, k ≥ 1 ∧ m = 1 + k * n}

def irreducible_in_V_n (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ p q : ℕ, p ∈ V_n n → q ∈ V_n n → p * q ≠ m

theorem existence_of_multiple_factorizations (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ V_n n ∧
    ∃ (factors1 factors2 : List ℕ),
      factors1 ≠ factors2 ∧
      (∀ f ∈ factors1, irreducible_in_V_n n f) ∧
      (∀ f ∈ factors2, irreducible_in_V_n n f) ∧
      r = factors1.prod ∧
      r = factors2.prod :=
  sorry

end existence_of_multiple_factorizations_l553_55343


namespace f_max_min_values_f_max_min_m_neg_f_max_min_m_0_to_4_f_max_min_m_gt_4_l553_55359

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + m - 1

-- Define the domain
def domain : Set ℝ := Set.Icc 0 4

-- Theorem for the maximum and minimum values
theorem f_max_min_values (m : ℝ) :
  (∀ x ∈ domain, f m x ≥ (m - 1) ∧ f m x ≤ (15 - 7*m)) ∨
  ((∀ x ∈ domain, f m x ≥ (-m^2 + m - 1)) ∧
   ((0 ≤ m ∧ m ≤ 2 → ∀ x ∈ domain, f m x ≤ (15 - 7*m)) ∧
    (2 ≤ m ∧ m ≤ 4 → ∀ x ∈ domain, f m x ≤ (m - 1)))) ∨
  (∀ x ∈ domain, f m x ≥ (15 - 7*m) ∧ f m x ≤ (m - 1)) :=
by sorry

-- Helper theorems for each case
theorem f_max_min_m_neg (m : ℝ) (hm : m < 0) :
  ∀ x ∈ domain, f m x ≥ (m - 1) ∧ f m x ≤ (15 - 7*m) :=
by sorry

theorem f_max_min_m_0_to_4 (m : ℝ) (hm : 0 ≤ m ∧ m ≤ 4) :
  (∀ x ∈ domain, f m x ≥ (-m^2 + m - 1)) ∧
  ((0 ≤ m ∧ m ≤ 2 → ∀ x ∈ domain, f m x ≤ (15 - 7*m)) ∧
   (2 ≤ m ∧ m ≤ 4 → ∀ x ∈ domain, f m x ≤ (m - 1))) :=
by sorry

theorem f_max_min_m_gt_4 (m : ℝ) (hm : m > 4) :
  ∀ x ∈ domain, f m x ≥ (15 - 7*m) ∧ f m x ≤ (m - 1) :=
by sorry

end f_max_min_values_f_max_min_m_neg_f_max_min_m_0_to_4_f_max_min_m_gt_4_l553_55359


namespace function_inequality_l553_55367

-- Define a differentiable function f
variable (f : ℝ → ℝ)

-- Assume f is differentiable
variable (hf : Differentiable ℝ f)

-- Assume f'(x) < f(x) for all x in ℝ
variable (h : ∀ x : ℝ, deriv f x < f x)

-- Theorem statement
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x : ℝ, deriv f x < f x) : 
  f 1 < Real.exp 1 * f 0 ∧ f 2014 < Real.exp 2014 * f 0 := by
  sorry

end function_inequality_l553_55367


namespace albert_oranges_l553_55340

/-- The number of boxes Albert has -/
def num_boxes : ℕ := 7

/-- The number of oranges in each box -/
def oranges_per_box : ℕ := 5

/-- The total number of oranges Albert has -/
def total_oranges : ℕ := num_boxes * oranges_per_box

theorem albert_oranges : total_oranges = 35 := by
  sorry

end albert_oranges_l553_55340


namespace smallest_non_odd_units_digit_l553_55300

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def units_digit (n : ℕ) : ℕ := n % 10

def is_digit (d : ℕ) : Prop := d < 10

theorem smallest_non_odd_units_digit :
  ∀ d : ℕ, is_digit d →
    (∀ n : ℕ, is_odd n → units_digit n ≠ d) →
    (∀ e : ℕ, is_digit e → (∀ m : ℕ, is_odd m → units_digit m ≠ e) → d ≤ e) →
    d = 0 :=
sorry

end smallest_non_odd_units_digit_l553_55300
