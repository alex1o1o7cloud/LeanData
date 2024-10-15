import Mathlib

namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l1372_137246

/-- Given two digits X and Y in base d > 8, prove that X - Y = -1 in base d
    when XY + XX = 234 in base d. -/
theorem digit_difference_in_base_d (d : ℕ) (X Y : ℕ) : d > 8 →
  X < d → Y < d →
  (X * d + Y) + (X * d + X) = 2 * d * d + 3 * d + 4 →
  X - Y = d - 1 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l1372_137246


namespace NUMINAMATH_CALUDE_room_occupancy_correct_answer_l1372_137242

theorem room_occupancy (num_empty_chairs : ℕ) : ℕ :=
  let total_chairs := 3 * num_empty_chairs
  let seated_people := (2 * total_chairs) / 3
  let total_people := 2 * seated_people
  total_people

theorem correct_answer : room_occupancy 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_room_occupancy_correct_answer_l1372_137242


namespace NUMINAMATH_CALUDE_three_digit_square_mod_1000_l1372_137208

theorem three_digit_square_mod_1000 (n : ℕ) : 
  (100 ≤ n ∧ n ≤ 999) → (n^2 ≡ n [ZMOD 1000] ↔ n = 376 ∨ n = 625) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_square_mod_1000_l1372_137208


namespace NUMINAMATH_CALUDE_eight_liter_solution_exists_l1372_137212

/-- Represents the state of the buckets -/
structure BucketState :=
  (bucket10 : ℕ)
  (bucket6 : ℕ)

/-- Represents a valid operation on the buckets -/
inductive BucketOperation
  | FillFrom10To6
  | FillFrom6To10
  | Empty10
  | Empty6
  | Fill10
  | Fill6

/-- Applies a bucket operation to a given state -/
def applyOperation (state : BucketState) (op : BucketOperation) : BucketState :=
  match op with
  | BucketOperation.FillFrom10To6 => 
      let transfer := min state.bucket10 (6 - state.bucket6)
      ⟨state.bucket10 - transfer, state.bucket6 + transfer⟩
  | BucketOperation.FillFrom6To10 => 
      let transfer := min state.bucket6 (10 - state.bucket10)
      ⟨state.bucket10 + transfer, state.bucket6 - transfer⟩
  | BucketOperation.Empty10 => ⟨0, state.bucket6⟩
  | BucketOperation.Empty6 => ⟨state.bucket10, 0⟩
  | BucketOperation.Fill10 => ⟨10, state.bucket6⟩
  | BucketOperation.Fill6 => ⟨state.bucket10, 6⟩

/-- Checks if the given sequence of operations results in 8 liters in one bucket -/
def checkSolution (ops : List BucketOperation) : Bool :=
  let finalState := ops.foldl applyOperation ⟨0, 0⟩
  finalState.bucket10 = 8 ∨ finalState.bucket6 = 8

/-- Theorem: There exists a sequence of operations that results in 8 liters in one bucket -/
theorem eight_liter_solution_exists : ∃ (ops : List BucketOperation), checkSolution ops := by
  sorry


end NUMINAMATH_CALUDE_eight_liter_solution_exists_l1372_137212


namespace NUMINAMATH_CALUDE_theodore_sturgeon_books_l1372_137233

theorem theodore_sturgeon_books (h p : ℕ) : 
  h + p = 10 →
  30 * h + 20 * p = 250 →
  h = 5 :=
by sorry

end NUMINAMATH_CALUDE_theodore_sturgeon_books_l1372_137233


namespace NUMINAMATH_CALUDE_hyperbola_canonical_equation_l1372_137240

/-- Represents a hyperbola with foci on the x-axis -/
structure Hyperbola where
  ε : ℝ  -- eccentricity
  f : ℝ  -- focal distance

/-- The canonical equation of a hyperbola -/
def canonical_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

/-- Theorem: For a hyperbola with ε = 1.5 and focal distance 6, the canonical equation is x²/4 - y²/5 = 1 -/
theorem hyperbola_canonical_equation (h : Hyperbola) 
    (h_ε : h.ε = 1.5) 
    (h_f : h.f = 6) :
    ∀ x y : ℝ, canonical_equation h x y :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_canonical_equation_l1372_137240


namespace NUMINAMATH_CALUDE_statement_II_must_be_true_l1372_137231

-- Define the possible digits
inductive Digit
| two
| three
| five
| six
| other

-- Define the statements
def statement_I (d : Digit) : Prop := d = Digit.two
def statement_II (d : Digit) : Prop := d ≠ Digit.three
def statement_III (d : Digit) : Prop := d = Digit.five
def statement_IV (d : Digit) : Prop := d ≠ Digit.six

-- Define the problem conditions
def conditions (d : Digit) : Prop :=
  ∃ (s1 s2 s3 : Prop),
    (s1 ∧ s2 ∧ s3) ∧
    (s1 = statement_I d ∨ s1 = statement_II d ∨ s1 = statement_III d ∨ s1 = statement_IV d) ∧
    (s2 = statement_I d ∨ s2 = statement_II d ∨ s2 = statement_III d ∨ s2 = statement_IV d) ∧
    (s3 = statement_I d ∨ s3 = statement_II d ∨ s3 = statement_III d ∨ s3 = statement_IV d) ∧
    (s1 ≠ s2 ∧ s1 ≠ s3 ∧ s2 ≠ s3)

-- Theorem: Given the conditions, Statement II must be true
theorem statement_II_must_be_true :
  ∀ d : Digit, conditions d → statement_II d :=
by
  sorry

end NUMINAMATH_CALUDE_statement_II_must_be_true_l1372_137231


namespace NUMINAMATH_CALUDE_largest_harmonious_n_is_correct_l1372_137289

/-- A coloring of a regular polygon's sides and diagonals. -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 2018

/-- A harmonious coloring has no two-colored triangles. -/
def Harmonious (n : ℕ) (c : Coloring n) : Prop :=
  ∀ i j k : Fin n, (c i j = c i k ∧ c i j ≠ c j k) → c i k = c j k

/-- The largest N for which a harmonious coloring of a regular N-gon exists. -/
def LargestHarmoniousN : ℕ := 2017^2

theorem largest_harmonious_n_is_correct :
  (∃ (c : Coloring LargestHarmoniousN), Harmonious LargestHarmoniousN c) ∧
  (∀ n > LargestHarmoniousN, ¬∃ (c : Coloring n), Harmonious n c) :=
sorry

end NUMINAMATH_CALUDE_largest_harmonious_n_is_correct_l1372_137289


namespace NUMINAMATH_CALUDE_max_a_for_decreasing_cos_minus_sin_l1372_137260

/-- The maximum value of a for which f(x) = cos x - sin x is decreasing on [-a, a] --/
theorem max_a_for_decreasing_cos_minus_sin (a : ℝ) : 
  (∀ x ∈ Set.Icc (-a) a, 
    ∀ y ∈ Set.Icc (-a) a, 
    x < y → (Real.cos x - Real.sin x) > (Real.cos y - Real.sin y)) → 
  a ≤ π/4 :=
sorry

end NUMINAMATH_CALUDE_max_a_for_decreasing_cos_minus_sin_l1372_137260


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l1372_137262

/-- The number of small seats on the Ferris wheel -/
def num_small_seats : ℕ := 2

/-- The total number of people that can ride on small seats -/
def total_people_small_seats : ℕ := 28

/-- The number of people each small seat can hold -/
def people_per_small_seat : ℕ := total_people_small_seats / num_small_seats

theorem ferris_wheel_capacity : people_per_small_seat = 14 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l1372_137262


namespace NUMINAMATH_CALUDE_quadratic_function_range_l1372_137222

/-- Given a quadratic function f(x) = ax^2 - 2ax + c where f(2017) < f(-2016),
    prove that the set of real numbers m satisfying f(m) ≤ f(0) is [0, 2] -/
theorem quadratic_function_range (a c : ℝ) : 
  let f := λ x : ℝ => a * x^2 - 2 * a * x + c
  (f 2017 < f (-2016)) → 
  {m : ℝ | f m ≤ f 0} = Set.Icc 0 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l1372_137222


namespace NUMINAMATH_CALUDE_quadratic_function_m_values_l1372_137225

theorem quadratic_function_m_values (m : ℝ) :
  (∃ a b c : ℝ, ∀ x, (m^2 - m) * x^(m^2 - 2*m - 1) + (m - 3) * x + m^2 = a * x^2 + b * x + c) →
  (m = 3 ∨ m = -1) ∧
  ((m = 3 → ∀ x, (m^2 - m) * x^(m^2 - 2*m - 1) + (m - 3) * x + m^2 = 6 * x^2 + 9) ∧
   (m = -1 → ∀ x, (m^2 - m) * x^(m^2 - 2*m - 1) + (m - 3) * x + m^2 = 2 * x^2 - 4 * x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_m_values_l1372_137225


namespace NUMINAMATH_CALUDE_equality_from_inequalities_l1372_137264

theorem equality_from_inequalities (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h1 : (x₁^2 - x₃*x₅)*(x₂^2 - x₃*x₅) ≤ 0)
  (h2 : (x₂^2 - x₄*x₁)*(x₃^2 - x₄*x₁) ≤ 0)
  (h3 : (x₃^2 - x₅*x₂)*(x₄^2 - x₅*x₂) ≤ 0)
  (h4 : (x₄^2 - x₁*x₃)*(x₅^2 - x₁*x₃) ≤ 0)
  (h5 : (x₅^2 - x₂*x₄)*(x₁^2 - x₂*x₄) ≤ 0) :
  x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅ :=
by sorry

end NUMINAMATH_CALUDE_equality_from_inequalities_l1372_137264


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1372_137234

theorem complex_equation_solution (a b : ℝ) : 
  (a : ℂ) + (b : ℂ) * Complex.I = (1 - Complex.I) / (2 * Complex.I) → 
  a = -1/2 ∧ b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1372_137234


namespace NUMINAMATH_CALUDE_max_largest_integer_l1372_137254

theorem max_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℚ) / 5 = 50 →
  max a (max b (max c (max d e))) - min a (min b (min c (min d e))) = 10 →
  max a (max b (max c (max d e))) ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_max_largest_integer_l1372_137254


namespace NUMINAMATH_CALUDE_list_number_fraction_l1372_137292

theorem list_number_fraction (list : List ℝ) (n : ℝ) : 
  list.length = 31 ∧ 
  n ∉ list ∧
  n = 5 * ((list.sum) / 30) →
  n / (list.sum + n) = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_list_number_fraction_l1372_137292


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l1372_137278

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, |3*y - 4| ≤ 18 → y ≥ x) → |3*x - 4| ≤ 18 → x = -4 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l1372_137278


namespace NUMINAMATH_CALUDE_points_on_line_equidistant_from_axes_l1372_137249

theorem points_on_line_equidistant_from_axes :
  ∃ (x y : ℝ), 4 * x - 3 * y = 24 ∧ |x| = |y| ∧
  ((x > 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by sorry

end NUMINAMATH_CALUDE_points_on_line_equidistant_from_axes_l1372_137249


namespace NUMINAMATH_CALUDE_max_k_value_l1372_137290

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∀ k : ℝ, 1/m + 2/(1-2*m) ≥ k) → (∃ k : ℝ, k = 8 ∧ ∀ k' : ℝ, (∀ m' : ℝ, 0 < m' ∧ m' < 1/2 → 1/m' + 2/(1-2*m') ≥ k') → k' ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l1372_137290


namespace NUMINAMATH_CALUDE_quadratic_root_sqrt5_minus3_l1372_137281

theorem quadratic_root_sqrt5_minus3 :
  ∃ (a b c : ℚ), a ≠ 0 ∧ 
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 + 1) ∧
  a = 1 ∧ b = 2 ∧ c = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_sqrt5_minus3_l1372_137281


namespace NUMINAMATH_CALUDE_equation_represents_empty_set_l1372_137202

theorem equation_represents_empty_set : 
  ∀ (x y : ℝ), 3 * x^2 + 5 * y^2 - 9 * x - 20 * y + 30 + 4 * x * y ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_represents_empty_set_l1372_137202


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1372_137293

/-- A line with slope m passing through point (x₀, y₀) -/
def Line (m x₀ y₀ : ℝ) : ℝ → ℝ := λ x ↦ m * (x - x₀) + y₀

theorem quadrilateral_area : 
  let line1 := Line (-3) 5 5
  let line2 := Line (-1) 10 0
  let B := (0, line1 0)
  let E := (5, 5)
  let C := (10, 0)
  (B.2 * C.1 - (B.2 * E.1 + C.1 * E.2)) / 2 = 125 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1372_137293


namespace NUMINAMATH_CALUDE_solve_animal_videos_problem_l1372_137257

def animal_videos_problem (cat_video_length : ℕ) : Prop :=
  let dog_video_length := 2 * cat_video_length
  let gorilla_video_length := 2 * (cat_video_length + dog_video_length)
  let elephant_video_length := cat_video_length + dog_video_length + gorilla_video_length
  let dolphin_video_length := cat_video_length + dog_video_length + gorilla_video_length + elephant_video_length
  let total_time := cat_video_length + dog_video_length + gorilla_video_length + elephant_video_length + dolphin_video_length
  (cat_video_length = 4) → (total_time = 144)

theorem solve_animal_videos_problem :
  animal_videos_problem 4 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_animal_videos_problem_l1372_137257


namespace NUMINAMATH_CALUDE_zeros_in_Q_l1372_137223

/-- R_k is an integer whose base-ten representation is a sequence of k ones -/
def R (k : ℕ+) : ℕ := (10^k.val - 1) / 9

/-- Q is the quotient of R_28 divided by R_8 -/
def Q : ℕ := R 28 / R 8

/-- count_zeros counts the number of zeros in the base-ten representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 21 := by sorry

end NUMINAMATH_CALUDE_zeros_in_Q_l1372_137223


namespace NUMINAMATH_CALUDE_ellipse_interfocal_distance_l1372_137261

/-- An ellipse with given latus rectum and focus-to-vertex distance has a specific interfocal distance -/
theorem ellipse_interfocal_distance 
  (latus_rectum : ℝ) 
  (focus_to_vertex : ℝ) 
  (h1 : latus_rectum = 5.4)
  (h2 : focus_to_vertex = 1.5) :
  ∃ (a b c : ℝ),
    a^2 = b^2 + c^2 ∧
    a - c = focus_to_vertex ∧
    b = latus_rectum / 2 ∧
    2 * c = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_interfocal_distance_l1372_137261


namespace NUMINAMATH_CALUDE_operation_equivalence_l1372_137204

-- Define the set of operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply the operation
def apply_op (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_equivalence 
  (star mul : Operation) 
  (h_unique : star ≠ mul) 
  (h_eq : (apply_op star 16 4) / (apply_op mul 10 2) = 4) :
  (apply_op star 5 15) / (apply_op mul 8 12) = 30 := by
  sorry


end NUMINAMATH_CALUDE_operation_equivalence_l1372_137204


namespace NUMINAMATH_CALUDE_no_prime_satisfies_condition_l1372_137284

theorem no_prime_satisfies_condition : ¬ ∃ p : ℕ, Nat.Prime p ∧ (10 : ℝ) * p = p + 5.4 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_satisfies_condition_l1372_137284


namespace NUMINAMATH_CALUDE_cookies_per_bag_example_l1372_137276

/-- Given the number of chocolate chip cookies, oatmeal cookies, and baggies,
    calculate the number of cookies in each bag. -/
def cookies_per_bag (chocolate_chip : ℕ) (oatmeal : ℕ) (baggies : ℕ) : ℕ :=
  (chocolate_chip + oatmeal) / baggies

/-- Theorem stating that with 5 chocolate chip cookies, 19 oatmeal cookies,
    and 3 baggies, there are 8 cookies in each bag. -/
theorem cookies_per_bag_example : cookies_per_bag 5 19 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_example_l1372_137276


namespace NUMINAMATH_CALUDE_smallest_divisible_by_4_13_7_l1372_137224

theorem smallest_divisible_by_4_13_7 : ∀ n : ℕ, n > 0 ∧ 4 ∣ n ∧ 13 ∣ n ∧ 7 ∣ n → n ≥ 364 := by
  sorry

#check smallest_divisible_by_4_13_7

end NUMINAMATH_CALUDE_smallest_divisible_by_4_13_7_l1372_137224


namespace NUMINAMATH_CALUDE_origami_distribution_l1372_137210

theorem origami_distribution (total_papers : ℕ) (num_cousins : ℕ) (papers_per_cousin : ℕ) : 
  total_papers = 48 → 
  num_cousins = 6 → 
  total_papers = num_cousins * papers_per_cousin →
  papers_per_cousin = 8 := by
  sorry

end NUMINAMATH_CALUDE_origami_distribution_l1372_137210


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_l1372_137250

theorem simplify_trigonometric_expression (α : Real) 
  (h : π < α ∧ α < 3*π/2) : 
  Real.sqrt ((1 + Real.cos (9*π/2 - α)) / (1 + Real.sin (α - 5*π))) - 
  Real.sqrt ((1 - Real.cos (-3*π/2 - α)) / (1 - Real.sin (α - 9*π))) = 
  -2 * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_l1372_137250


namespace NUMINAMATH_CALUDE_tank_filled_at_10pm_l1372_137280

/-- Represents the rainfall rate at a given hour after 1 pm -/
def rainfall_rate (hour : ℕ) : ℝ :=
  if hour = 0 then 2
  else if hour ≤ 4 then 1
  else 3

/-- Calculates the total rainfall up to a given hour after 1 pm -/
def total_rainfall (hour : ℕ) : ℝ :=
  (Finset.range (hour + 1)).sum rainfall_rate

/-- The height of the fish tank in inches -/
def tank_height : ℝ := 18

/-- Theorem stating that the fish tank will be filled at 10 pm -/
theorem tank_filled_at_10pm :
  ∃ (h : ℕ), h = 9 ∧ total_rainfall h ≥ tank_height ∧ total_rainfall (h - 1) < tank_height :=
sorry

end NUMINAMATH_CALUDE_tank_filled_at_10pm_l1372_137280


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1372_137207

theorem arithmetic_calculation : (30 / (10 + 2 - 5) + 4) * 7 = 58 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1372_137207


namespace NUMINAMATH_CALUDE_daisy_milk_leftover_l1372_137272

/-- Calculates the amount of milk left over given the total production, percentage consumed by kids, and percentage of remainder used for cooking. -/
def milk_left_over (total_milk : ℝ) (kids_consumption_percent : ℝ) (cooking_percent : ℝ) : ℝ :=
  let remaining_after_kids := total_milk * (1 - kids_consumption_percent)
  remaining_after_kids * (1 - cooking_percent)

/-- Theorem stating that given 16 cups of milk per day, with 75% consumed by kids and 50% of the remainder used for cooking, 2 cups of milk are left over. -/
theorem daisy_milk_leftover :
  milk_left_over 16 0.75 0.5 = 2 := by
  sorry

#eval milk_left_over 16 0.75 0.5

end NUMINAMATH_CALUDE_daisy_milk_leftover_l1372_137272


namespace NUMINAMATH_CALUDE_tan_theta_negative_three_l1372_137282

theorem tan_theta_negative_three (θ : Real) (h : Real.tan θ = -3) :
  (Real.sin θ - 2 * Real.cos θ) / (Real.cos θ + Real.sin θ) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_negative_three_l1372_137282


namespace NUMINAMATH_CALUDE_shortest_tree_height_l1372_137288

/-- Given three trees with specified height relationships, prove the height of the shortest tree -/
theorem shortest_tree_height (tallest middle shortest : ℝ) : 
  tallest = 150 →
  middle = 2/3 * tallest →
  shortest = 1/2 * middle →
  shortest = 50 := by
sorry

end NUMINAMATH_CALUDE_shortest_tree_height_l1372_137288


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1372_137255

theorem min_value_expression (y : ℝ) (h : y > 2) :
  (y^2 + y + 1) / Real.sqrt (y - 2) ≥ 3 * Real.sqrt 35 :=
by sorry

theorem min_value_achievable :
  ∃ y : ℝ, y > 2 ∧ (y^2 + y + 1) / Real.sqrt (y - 2) = 3 * Real.sqrt 35 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1372_137255


namespace NUMINAMATH_CALUDE_smallest_base_for_fraction_l1372_137253

theorem smallest_base_for_fraction (k : ℕ) : k = 14 ↔ 
  (k > 0 ∧ 
   ∀ m : ℕ, m > 0 ∧ m < k → (5 : ℚ) / 27 ≠ (m + 4 : ℚ) / (m^2 - 1) ∧
   (5 : ℚ) / 27 = (k + 4 : ℚ) / (k^2 - 1)) := by sorry

end NUMINAMATH_CALUDE_smallest_base_for_fraction_l1372_137253


namespace NUMINAMATH_CALUDE_jana_kelly_height_difference_l1372_137200

/-- Proves that Jana is 5 inches taller than Kelly given the heights of Jess and Jana, and the height difference between Jess and Kelly. -/
theorem jana_kelly_height_difference :
  ∀ (jess_height jana_height kelly_height : ℕ),
    jess_height = 72 →
    jana_height = 74 →
    kelly_height = jess_height - 3 →
    jana_height - kelly_height = 5 := by
  sorry

end NUMINAMATH_CALUDE_jana_kelly_height_difference_l1372_137200


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1372_137235

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := fun x ↦ 3 - a^(x + 1)
  f (-1) = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1372_137235


namespace NUMINAMATH_CALUDE_tan_beta_value_l1372_137237

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = -2/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_value_l1372_137237


namespace NUMINAMATH_CALUDE_max_b_value_l1372_137214

theorem max_b_value (a b c : ℕ) : 
  a * b * c = 360 →
  1 < c →
  c ≤ b →
  b < a →
  b ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_l1372_137214


namespace NUMINAMATH_CALUDE_booklet_pages_theorem_l1372_137229

theorem booklet_pages_theorem (n : ℕ) (r : ℕ) : 
  (∃ (n : ℕ) (r : ℕ), 2 * n * (2 * n + 1) / 2 - (4 * r - 1) = 963 ∧ 
   1 ≤ r ∧ r ≤ n) → 
  (n = 22 ∧ r = 7) := by
  sorry

end NUMINAMATH_CALUDE_booklet_pages_theorem_l1372_137229


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1372_137213

theorem quadratic_equation_solution : 
  ∃! x : ℝ, x^2 + 6*x + 8 = -(x + 4)*(x + 6) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1372_137213


namespace NUMINAMATH_CALUDE_moe_has_least_money_l1372_137244

-- Define the set of people
inductive Person : Type
  | Bo | Coe | Flo | Jo | Moe | Zoe

-- Define a function to represent the amount of money each person has
variable (money : Person → ℕ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q

axiom flo_more_than_jo_bo : money Person.Flo > money Person.Jo ∧ money Person.Flo > money Person.Bo

axiom bo_coe_more_than_moe_less_than_zoe : 
  money Person.Bo > money Person.Moe ∧ 
  money Person.Coe > money Person.Moe ∧
  money Person.Zoe > money Person.Bo ∧
  money Person.Zoe > money Person.Coe

axiom jo_more_than_moe_zoe_less_than_bo : 
  money Person.Jo > money Person.Moe ∧
  money Person.Jo > money Person.Zoe ∧
  money Person.Bo > money Person.Jo

-- Theorem to prove
theorem moe_has_least_money : 
  ∀ (p : Person), p ≠ Person.Moe → money Person.Moe < money p :=
sorry

end NUMINAMATH_CALUDE_moe_has_least_money_l1372_137244


namespace NUMINAMATH_CALUDE_count_four_digit_with_five_thousands_l1372_137219

/-- A four-digit positive integer with thousands digit 5 -/
def FourDigitWithFiveThousands : Type := { n : ℕ // 5000 ≤ n ∧ n ≤ 5999 }

/-- The count of four-digit positive integers with thousands digit 5 -/
def CountFourDigitWithFiveThousands : ℕ := Finset.card (Finset.filter (λ n => 5000 ≤ n ∧ n ≤ 5999) (Finset.range 10000))

theorem count_four_digit_with_five_thousands :
  CountFourDigitWithFiveThousands = 1000 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_with_five_thousands_l1372_137219


namespace NUMINAMATH_CALUDE_island_not_named_Maya_l1372_137279

-- Define the inhabitants
inductive Inhabitant : Type
| A : Inhabitant
| B : Inhabitant

-- Define the possible states of an inhabitant
inductive State : Type
| TruthTeller : State
| Liar : State

-- Define the name of the island
def IslandName : Type := Bool

-- Define the statements made by A and B
def statement_A (state_A state_B : State) (island_name : IslandName) : Prop :=
  (state_A = State.Liar ∧ state_B = State.Liar) ∧ island_name = true

def statement_B (state_A state_B : State) (island_name : IslandName) : Prop :=
  (state_A = State.Liar ∨ state_B = State.Liar) ∧ island_name = false

-- The main theorem
theorem island_not_named_Maya :
  ∀ (state_A state_B : State) (island_name : IslandName),
    (state_A = State.Liar → ¬statement_A state_A state_B island_name) ∧
    (state_A = State.TruthTeller → statement_A state_A state_B island_name) ∧
    (state_B = State.Liar → ¬statement_B state_A state_B island_name) ∧
    (state_B = State.TruthTeller → statement_B state_A state_B island_name) →
    island_name = false :=
by
  sorry


end NUMINAMATH_CALUDE_island_not_named_Maya_l1372_137279


namespace NUMINAMATH_CALUDE_successive_numbers_product_l1372_137267

theorem successive_numbers_product (n : ℤ) : 
  n * (n + 1) = 2652 → n = 51 := by
  sorry

end NUMINAMATH_CALUDE_successive_numbers_product_l1372_137267


namespace NUMINAMATH_CALUDE_arithmetic_problem_l1372_137251

theorem arithmetic_problem : 40 + 5 * 12 / (180 / 3) = 41 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l1372_137251


namespace NUMINAMATH_CALUDE_sin_300_degrees_l1372_137268

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  -- Define the properties of sine function
  have sin_periodic : ∀ θ k, Real.sin θ = Real.sin (θ + k * 2 * π) := by sorry
  have sin_symmetry : ∀ θ, Real.sin θ = Real.sin (-θ) := by sorry
  have sin_odd : ∀ θ, Real.sin (-θ) = -Real.sin θ := by sorry
  have sin_60_degrees : Real.sin (60 * π / 180) = Real.sqrt 3 / 2 := by sorry

  -- Proof steps would go here, but we're skipping them as per instructions
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l1372_137268


namespace NUMINAMATH_CALUDE_total_tickets_after_sharing_l1372_137286

def tate_initial_tickets : ℕ := 32
def tate_bought_tickets : ℕ := 2

def tate_final_tickets : ℕ := tate_initial_tickets + tate_bought_tickets

def peyton_initial_tickets : ℕ := tate_final_tickets / 2

def peyton_given_away_tickets : ℕ := peyton_initial_tickets / 3

def peyton_final_tickets : ℕ := peyton_initial_tickets - peyton_given_away_tickets

theorem total_tickets_after_sharing :
  tate_final_tickets + peyton_final_tickets = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_after_sharing_l1372_137286


namespace NUMINAMATH_CALUDE_paths_in_7x8_grid_l1372_137218

/-- The number of paths in a grid with only upward and rightward movements -/
def gridPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem: The number of paths in a 7x8 grid is 6435 -/
theorem paths_in_7x8_grid :
  gridPaths 7 8 = 6435 := by
  sorry

end NUMINAMATH_CALUDE_paths_in_7x8_grid_l1372_137218


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1372_137215

theorem stratified_sampling_theorem :
  let total_employees : ℕ := 150
  let senior_titles : ℕ := 15
  let intermediate_titles : ℕ := 45
  let junior_titles : ℕ := 90
  let sample_size : ℕ := 30
  
  senior_titles + intermediate_titles + junior_titles = total_employees →
  
  let senior_sample := sample_size * senior_titles / total_employees
  let intermediate_sample := sample_size * intermediate_titles / total_employees
  let junior_sample := sample_size * junior_titles / total_employees
  
  (senior_sample = 3 ∧ intermediate_sample = 9 ∧ junior_sample = 18) :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1372_137215


namespace NUMINAMATH_CALUDE_weight_of_b_l1372_137256

/-- Given three weights a, b, and c, prove that b equals 31 when:
    1. The average of a, b, and c is 45.
    2. The average of a and b is 40.
    3. The average of b and c is 43. -/
theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) : 
  b = 31 := by
  sorry


end NUMINAMATH_CALUDE_weight_of_b_l1372_137256


namespace NUMINAMATH_CALUDE_monochromatic_triangle_probability_l1372_137227

/-- The number of vertices in the complete graph -/
def n : ℕ := 6

/-- The number of colors used for coloring the edges -/
def num_colors : ℕ := 3

/-- The probability of a specific triangle being non-monochromatic -/
def p_non_monochromatic : ℚ := 24 / 27

/-- The number of triangles in a complete graph with n vertices -/
def num_triangles : ℕ := n.choose 3

/-- The probability of having at least one monochromatic triangle -/
noncomputable def p_at_least_one_monochromatic : ℚ :=
  1 - p_non_monochromatic ^ num_triangles

theorem monochromatic_triangle_probability :
  p_at_least_one_monochromatic = 872 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_probability_l1372_137227


namespace NUMINAMATH_CALUDE_rectangle_max_m_l1372_137285

/-- Given a rectangle with area S and perimeter p, 
    M = (16 - p) / (p^2 + 2p) is maximized when the rectangle is a square -/
theorem rectangle_max_m (S : ℝ) (p : ℝ) (h_S : S > 0) (h_p : p > 0) :
  let M := (16 - p) / (p^2 + 2*p)
  M ≤ (4 - Real.sqrt S) / (4*S + 2*Real.sqrt S) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_m_l1372_137285


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l1372_137295

theorem rectangle_area_increase (p a : ℝ) (h_a : a > 0) :
  let perimeter := 2 * p
  let increase := a
  let area_increase := 
    fun (x y : ℝ) => 
      ((x + increase) * (y + increase)) - (x * y)
  ∀ x y, x + y = p → area_increase x y = a * (p + a) := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l1372_137295


namespace NUMINAMATH_CALUDE_larger_integer_value_l1372_137273

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * (b : ℕ) = 189) :
  max a b = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l1372_137273


namespace NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l1372_137211

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_fourth_and_fifth_terms 
  (a : ℕ → ℕ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 11)
  (h_sixth : a 6 = 43) :
  a 4 + a 5 = 62 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l1372_137211


namespace NUMINAMATH_CALUDE_probability_king_then_heart_l1372_137275

/- Define a standard deck of cards -/
def StandardDeck : ℕ := 52

/- Define the number of Kings in a standard deck -/
def NumberOfKings : ℕ := 4

/- Define the number of hearts in a standard deck -/
def NumberOfHearts : ℕ := 13

/- Theorem statement -/
theorem probability_king_then_heart (deck : ℕ) (kings : ℕ) (hearts : ℕ) 
  (h1 : deck = StandardDeck) 
  (h2 : kings = NumberOfKings) 
  (h3 : hearts = NumberOfHearts) : 
  (kings : ℚ) / deck * hearts / (deck - 1) = 1 / 52 := by
  sorry


end NUMINAMATH_CALUDE_probability_king_then_heart_l1372_137275


namespace NUMINAMATH_CALUDE_sampling_survey_appropriate_l1372_137299

/-- Represents a survey method -/
inductive SurveyMethod
| Census
| SamplingSurvey

/-- Represents the characteristics of a population survey -/
structure PopulationSurvey where
  populationSize : Nat
  needsEfficiency : Bool

/-- Determines the appropriate survey method based on population characteristics -/
def appropriateSurveyMethod (survey : PopulationSurvey) : SurveyMethod :=
  if survey.populationSize > 1000000 && survey.needsEfficiency
  then SurveyMethod.SamplingSurvey
  else SurveyMethod.Census

/-- Theorem: For a large population requiring efficient data collection,
    sampling survey is the appropriate method -/
theorem sampling_survey_appropriate
  (survey : PopulationSurvey)
  (h1 : survey.populationSize > 1000000)
  (h2 : survey.needsEfficiency) :
  appropriateSurveyMethod survey = SurveyMethod.SamplingSurvey :=
by
  sorry


end NUMINAMATH_CALUDE_sampling_survey_appropriate_l1372_137299


namespace NUMINAMATH_CALUDE_place_value_ratio_l1372_137226

/-- Represents a decimal number with its integer and fractional parts -/
structure DecimalNumber where
  integerPart : ℕ
  fractionalPart : ℕ
  fractionalDigits : ℕ

/-- Returns the place value of a digit at a given position in a decimal number -/
def placeValue (n : DecimalNumber) (position : ℤ) : ℚ :=
  10 ^ position

/-- The decimal number 50467.8912 -/
def number : DecimalNumber :=
  { integerPart := 50467
  , fractionalPart := 8912
  , fractionalDigits := 4 }

/-- The position of digit 8 in the number (counting from right, negative for fractional part) -/
def pos8 : ℤ := -1

/-- The position of digit 7 in the number (counting from right) -/
def pos7 : ℤ := 1

theorem place_value_ratio :
  (placeValue number pos8) / (placeValue number pos7) = 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratio_l1372_137226


namespace NUMINAMATH_CALUDE_range_of_x_range_of_a_l1372_137247

-- Define propositions p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

-- Part 1
theorem range_of_x (x : ℝ) (h1 : p x 1) (h2 : q x) : x ∈ Set.Ioo 2 3 := by
  sorry

-- Part 2
theorem range_of_a (a : ℝ) (h : ∀ x, ¬(p x a) → ¬(q x)) : a ∈ Set.Ioc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_a_l1372_137247


namespace NUMINAMATH_CALUDE_sum_of_series_l1372_137232

/-- The sum of the infinite series ∑(n=1 to ∞) (4n+1)/3^n is equal to 7/2 -/
theorem sum_of_series : ∑' n : ℕ, (4 * n + 1 : ℝ) / 3^n = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_series_l1372_137232


namespace NUMINAMATH_CALUDE_sp_length_l1372_137265

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8)
  (bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 9)
  (ca_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 10)

-- Define the point T
def T (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the point S
def S (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the point P
def P (triangle : Triangle) : ℝ × ℝ := sorry

-- Theorem statement
theorem sp_length (triangle : Triangle) : 
  Real.sqrt ((S triangle).1 - (P triangle).1)^2 + ((S triangle).2 - (P triangle).2)^2 = 225/13 := by
  sorry

end NUMINAMATH_CALUDE_sp_length_l1372_137265


namespace NUMINAMATH_CALUDE_runners_in_picture_probability_l1372_137291

/-- Represents a runner on a circular track -/
structure Runner where
  name : String
  lap_time : ℕ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the photographer's setup -/
structure Photographer where
  start_time : ℕ  -- in seconds
  end_time : ℕ    -- in seconds
  track_coverage : ℚ  -- fraction of track covered in picture

/-- Calculates the probability of both runners being in the picture -/
def probability_both_in_picture (linda : Runner) (luis : Runner) (photographer : Photographer) : ℚ :=
  sorry  -- Proof goes here

/-- The main theorem statement -/
theorem runners_in_picture_probability :
  let linda : Runner := { name := "Linda", lap_time := 120, direction := true }
  let luis : Runner := { name := "Luis", lap_time := 75, direction := false }
  let photographer : Photographer := { start_time := 900, end_time := 960, track_coverage := 1/3 }
  probability_both_in_picture linda luis photographer = 5/6 := by
  sorry  -- Proof goes here

end NUMINAMATH_CALUDE_runners_in_picture_probability_l1372_137291


namespace NUMINAMATH_CALUDE_seating_theorem_l1372_137230

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def seating_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  factorial n - factorial (n - k + 1) * factorial k

theorem seating_theorem :
  seating_arrangements 8 3 = 36000 :=
sorry

end NUMINAMATH_CALUDE_seating_theorem_l1372_137230


namespace NUMINAMATH_CALUDE_symmetric_line_l1372_137209

/-- Given a line L1 with equation y = 2x + 1 and a line of symmetry L2 with equation y + 2 = 0,
    the symmetric line L3 has the equation 2x + y + 5 = 0 -/
theorem symmetric_line (x y : ℝ) : 
  (y = 2*x + 1) →  -- Original line L1
  (y = -2)      →  -- Line of symmetry L2 (y + 2 = 0 rearranged)
  (2*x + y + 5 = 0) -- Symmetric line L3
  := by sorry

end NUMINAMATH_CALUDE_symmetric_line_l1372_137209


namespace NUMINAMATH_CALUDE_tan_half_angle_formula_22_5_degrees_l1372_137205

theorem tan_half_angle_formula_22_5_degrees : 
  (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_angle_formula_22_5_degrees_l1372_137205


namespace NUMINAMATH_CALUDE_isosceles_triangle_most_stable_isosceles_triangle_stable_other_shapes_not_stable_l1372_137238

/-- Represents a geometric shape -/
inductive Shape
  | IsoscelesTriangle
  | Rectangle
  | Square
  | Parallelogram

/-- Stability measure of a shape -/
def stability (s : Shape) : ℕ :=
  match s with
  | Shape.IsoscelesTriangle => 3
  | Shape.Rectangle => 2
  | Shape.Square => 2
  | Shape.Parallelogram => 1

/-- A shape is considered stable if its stability measure is greater than 2 -/
def is_stable (s : Shape) : Prop := stability s > 2

theorem isosceles_triangle_most_stable :
  ∀ s : Shape, s ≠ Shape.IsoscelesTriangle → stability Shape.IsoscelesTriangle > stability s :=
by sorry

theorem isosceles_triangle_stable :
  is_stable Shape.IsoscelesTriangle :=
by sorry

theorem other_shapes_not_stable :
  ¬ is_stable Shape.Rectangle ∧
  ¬ is_stable Shape.Square ∧
  ¬ is_stable Shape.Parallelogram :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_most_stable_isosceles_triangle_stable_other_shapes_not_stable_l1372_137238


namespace NUMINAMATH_CALUDE_sanctuary_animal_pairs_l1372_137259

theorem sanctuary_animal_pairs : 
  let bird_species : ℕ := 29
  let bird_pairs_per_species : ℕ := 7
  let marine_species : ℕ := 15
  let marine_pairs_per_species : ℕ := 9
  let mammal_species : ℕ := 22
  let mammal_pairs_per_species : ℕ := 6
  
  bird_species * bird_pairs_per_species + 
  marine_species * marine_pairs_per_species + 
  mammal_species * mammal_pairs_per_species = 470 :=
by
  sorry

end NUMINAMATH_CALUDE_sanctuary_animal_pairs_l1372_137259


namespace NUMINAMATH_CALUDE_polynomial_transformation_l1372_137270

theorem polynomial_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 5*x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 7) = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_transformation_l1372_137270


namespace NUMINAMATH_CALUDE_sin_special_angle_l1372_137221

/-- Given a function f(x) = sin(x/2 + π/4), prove that f(π/2) = 1 -/
theorem sin_special_angle (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin (x / 2 + π / 4)) :
  f (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_special_angle_l1372_137221


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1372_137243

theorem inequality_system_solution (x : ℤ) :
  (-3/2 : ℚ) < x ∧ (x : ℚ) ≤ 2 →
  -2*x + 7 < 10 ∧ (7*x + 1)/5 - 1 ≤ x := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1372_137243


namespace NUMINAMATH_CALUDE_marly_soup_bags_l1372_137294

/-- The number of bags needed for Marly's soup -/
def bags_needed (milk_quarts chicken_stock_multiplier vegetable_quarts bag_capacity : ℚ) : ℚ :=
  (milk_quarts + chicken_stock_multiplier * milk_quarts + vegetable_quarts) / bag_capacity

/-- Theorem: Marly needs 3 bags for his soup -/
theorem marly_soup_bags :
  bags_needed 2 3 1 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_marly_soup_bags_l1372_137294


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l1372_137274

theorem cube_sum_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  2 * (x^3 + y^3 + z^3) ≥ x^2 * y + x^2 * z + y^2 * z + y^2 * x + z^2 * x + z^2 * y :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l1372_137274


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l1372_137277

theorem smaller_number_in_ratio (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ 
  b / a = 11 / 7 ∧
  b - a = 16 →
  a = 28 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l1372_137277


namespace NUMINAMATH_CALUDE_cement_mixture_water_fraction_l1372_137269

/-- The fraction of water in a cement mixture -/
def water_fraction (total_weight sand_fraction gravel_weight : ℚ) : ℚ :=
  1 - sand_fraction - (gravel_weight / total_weight)

/-- Proof that the fraction of water in the cement mixture is 2/5 -/
theorem cement_mixture_water_fraction :
  let total_weight : ℚ := 40
  let sand_fraction : ℚ := 1/4
  let gravel_weight : ℚ := 14
  water_fraction total_weight sand_fraction gravel_weight = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_water_fraction_l1372_137269


namespace NUMINAMATH_CALUDE_multiplication_problem_solution_l1372_137236

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the multiplication problem -/
structure MultiplicationProblem where
  A : Digit
  B : Digit
  C : Digit
  D : Digit
  different : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D
  equation : (100 * A.val + 10 * B.val + A.val) * (10 * B.val + C.val) = 
              1000 * B.val + 100 * C.val + 10 * B.val + C.val

theorem multiplication_problem_solution (p : MultiplicationProblem) : 
  p.A.val + p.C.val = 5 := by sorry

end NUMINAMATH_CALUDE_multiplication_problem_solution_l1372_137236


namespace NUMINAMATH_CALUDE_room_occupancy_l1372_137201

theorem room_occupancy (x y : ℕ) : 
  x + y = 76 → 
  x - 30 = y - 40 → 
  (x = 33 ∧ y = 43) ∨ (x = 43 ∧ y = 33) :=
by sorry

end NUMINAMATH_CALUDE_room_occupancy_l1372_137201


namespace NUMINAMATH_CALUDE_cubic_foot_to_cubic_inches_l1372_137297

theorem cubic_foot_to_cubic_inches :
  ∀ (foot inch : ℝ), 
    foot > 0 →
    inch > 0 →
    foot = 12 * inch →
    foot^3 = 1728 * inch^3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_foot_to_cubic_inches_l1372_137297


namespace NUMINAMATH_CALUDE_bound_difference_for_elements_in_A_l1372_137206

/-- The function f(x) = |x+2| + |x-2| -/
def f (x : ℝ) : ℝ := |x + 2| + |x - 2|

/-- The set A of all x such that f(x) ≤ 6 -/
def A : Set ℝ := {x | f x ≤ 6}

/-- Theorem stating that if m and n are in A, then |1/3 * m - 1/2 * n| ≤ 5/2 -/
theorem bound_difference_for_elements_in_A (m n : ℝ) (hm : m ∈ A) (hn : n ∈ A) :
  |1/3 * m - 1/2 * n| ≤ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_bound_difference_for_elements_in_A_l1372_137206


namespace NUMINAMATH_CALUDE_no_solution_exists_l1372_137203

theorem no_solution_exists : ¬ ∃ (a b c t x₁ x₂ x₃ : ℝ),
  (a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁) ∧
  (a * x₁^2 + b * t * x₁ + c = 0) ∧
  (a * x₂^2 + b * t * x₂ + c = 0) ∧
  (b * x₂^2 + c * x₂ + a = 0) ∧
  (b * x₃^2 + c * x₃ + a = 0) ∧
  (c * x₃^2 + a * t * x₃ + b = 0) ∧
  (c * x₁^2 + a * t * x₁ + b = 0) :=
sorry

#check no_solution_exists

end NUMINAMATH_CALUDE_no_solution_exists_l1372_137203


namespace NUMINAMATH_CALUDE_decimal_place_150_l1372_137266

/-- The decimal representation of 5/6 -/
def decimal_rep_5_6 : ℚ := 5/6

/-- The length of the repeating cycle in the decimal representation of 5/6 -/
def cycle_length : ℕ := 6

/-- The nth digit in the decimal representation of 5/6 -/
def nth_digit (n : ℕ) : ℕ := sorry

theorem decimal_place_150 :
  nth_digit 150 = 3 :=
sorry

end NUMINAMATH_CALUDE_decimal_place_150_l1372_137266


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l1372_137248

/-- The value of j for which the line 4x - 7y + j = 0 is tangent to the ellipse x^2 + 4y^2 = 16 -/
theorem line_tangent_to_ellipse (x y j : ℝ) : 
  (∀ x y, 4*x - 7*y + j = 0 → x^2 + 4*y^2 = 16) ↔ j^2 = 450.5 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l1372_137248


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l1372_137216

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with given dimensions and rate -/
theorem paving_cost_calculation (length width rate : ℝ) 
  (h1 : length = 5)
  (h2 : width = 4.75)
  (h3 : rate = 900) :
  paving_cost length width rate = 21375 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l1372_137216


namespace NUMINAMATH_CALUDE_floor_sum_example_l1372_137239

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l1372_137239


namespace NUMINAMATH_CALUDE_parabola_slope_relation_l1372_137287

/-- Parabola struct representing y^2 = 2px --/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point on a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Line on a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Theorem statement --/
theorem parabola_slope_relation 
  (C : Parabola) 
  (A : Point)
  (B : Point)
  (P M N : Point)
  (h_A : A.y^2 = 2 * C.p * A.x)
  (h_A_x : A.x = 1)
  (h_B : B.x = -C.p/2 ∧ B.y = 0)
  (h_AB : (A.x - B.x)^2 + (A.y - B.y)^2 = 8)
  (h_P : P.y^2 = 2 * C.p * P.x ∧ P.y = 2)
  (h_M : M.y^2 = 2 * C.p * M.x)
  (h_N : N.y^2 = 2 * C.p * N.x)
  (k₁ k₂ k₃ : ℝ)
  (h_k₁ : k₁ ≠ 0)
  (h_k₂ : k₂ ≠ 0)
  (h_k₃ : k₃ ≠ 0)
  (h_PM : (M.y - P.y) = k₁ * (M.x - P.x))
  (h_PN : (N.y - P.y) = k₂ * (N.x - P.x))
  (h_MN : (N.y - M.y) = k₃ * (N.x - M.x)) :
  1/k₁ + 1/k₂ - 1/k₃ = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_slope_relation_l1372_137287


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1372_137220

theorem sqrt_product_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (10 * p) * Real.sqrt (5 * p^2) * Real.sqrt (6 * p^4) = 10 * p^3 * Real.sqrt (3 * p) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1372_137220


namespace NUMINAMATH_CALUDE_initial_cells_theorem_l1372_137241

/-- Calculates the number of cells after one hour given the initial number of cells -/
def cellsAfterOneHour (initialCells : ℕ) : ℕ :=
  2 * (initialCells - 2)

/-- Calculates the number of cells after n hours given the initial number of cells -/
def cellsAfterNHours (initialCells : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initialCells
  | n + 1 => cellsAfterOneHour (cellsAfterNHours initialCells n)

/-- Theorem stating that 9 initial cells result in 164 cells after 5 hours -/
theorem initial_cells_theorem :
  cellsAfterNHours 9 5 = 164 :=
by sorry

end NUMINAMATH_CALUDE_initial_cells_theorem_l1372_137241


namespace NUMINAMATH_CALUDE_sequence_sum_property_l1372_137217

theorem sequence_sum_property (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n : ℕ+, S n = n^2 - a n) →
  (∃ k : ℕ+, 1 < S k ∧ S k < 9) →
  (∃ k : ℕ+, k = 2 ∧ 1 < S k ∧ S k < 9) :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l1372_137217


namespace NUMINAMATH_CALUDE_iesha_school_books_l1372_137263

/-- The number of books Iesha has about school -/
def books_about_school (total_books sports_books : ℕ) : ℕ :=
  total_books - sports_books

/-- Theorem stating that Iesha has 19 books about school -/
theorem iesha_school_books : 
  books_about_school 58 39 = 19 := by
  sorry

end NUMINAMATH_CALUDE_iesha_school_books_l1372_137263


namespace NUMINAMATH_CALUDE_final_number_lower_bound_l1372_137298

/-- Represents a sequence of operations on the blackboard -/
def BlackboardOperation := List (Nat × Nat)

/-- The result of applying a sequence of operations to the initial numbers -/
def applyOperations (n : Nat) (ops : BlackboardOperation) : Nat :=
  sorry

/-- Theorem: The final number after any sequence of operations is at least 4/9 * n^3 -/
theorem final_number_lower_bound (n : Nat) (ops : BlackboardOperation) :
  applyOperations n ops ≥ (4 * n^3) / 9 := by
  sorry

end NUMINAMATH_CALUDE_final_number_lower_bound_l1372_137298


namespace NUMINAMATH_CALUDE_contest_prize_distribution_l1372_137245

theorem contest_prize_distribution (total_prize : ℕ) (total_winners : ℕ) 
  (first_prize : ℕ) (second_prize : ℕ) (third_prize : ℕ) 
  (h1 : total_prize = 800) (h2 : total_winners = 18) 
  (h3 : first_prize = 200) (h4 : second_prize = 150) (h5 : third_prize = 120) :
  let remaining_prize := total_prize - (first_prize + second_prize + third_prize)
  let remaining_winners := total_winners - 3
  remaining_prize / remaining_winners = 22 := by
sorry

end NUMINAMATH_CALUDE_contest_prize_distribution_l1372_137245


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1372_137258

theorem complex_modulus_problem (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (1 + 2*a*i)*i = 1 - b*i) : 
  Complex.abs (a + b*i) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1372_137258


namespace NUMINAMATH_CALUDE_books_count_l1372_137296

/-- The number of books Jason has -/
def jason_books : ℕ := 18

/-- The number of books Mary has -/
def mary_books : ℕ := 42

/-- The total number of books Jason and Mary have together -/
def total_books : ℕ := jason_books + mary_books

theorem books_count : total_books = 60 := by
  sorry

end NUMINAMATH_CALUDE_books_count_l1372_137296


namespace NUMINAMATH_CALUDE_board_theorem_l1372_137228

/-- Represents a board with gold and silver cells. -/
structure Board :=
  (size : ℕ)
  (is_gold : ℕ → ℕ → Bool)

/-- Counts the number of gold cells in a given rectangle of the board. -/
def count_gold (b : Board) (x y w h : ℕ) : ℕ :=
  (Finset.range w).sum (λ i =>
    (Finset.range h).sum (λ j =>
      if b.is_gold (x + i) (y + j) then 1 else 0))

/-- Checks if the board satisfies the conditions for all 3x3 squares and 2x4/4x2 rectangles. -/
def valid_board (b : Board) (A Z : ℕ) : Prop :=
  (∀ x y, x + 3 ≤ b.size → y + 3 ≤ b.size →
    count_gold b x y 3 3 = A) ∧
  (∀ x y, x + 2 ≤ b.size → y + 4 ≤ b.size →
    count_gold b x y 2 4 = Z) ∧
  (∀ x y, x + 4 ≤ b.size → y + 2 ≤ b.size →
    count_gold b x y 4 2 = Z)

theorem board_theorem :
  ∀ b : Board, b.size = 2016 →
    (∃ A Z, valid_board b A Z) →
    (∃ A Z, valid_board b A Z ∧ ((A = 0 ∧ Z = 0) ∨ (A = 9 ∧ Z = 8))) :=
sorry

end NUMINAMATH_CALUDE_board_theorem_l1372_137228


namespace NUMINAMATH_CALUDE_min_value_sum_of_fractions_l1372_137252

theorem min_value_sum_of_fractions (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 3) : 
  (4 / x) + (9 / y) + (16 / z) ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_fractions_l1372_137252


namespace NUMINAMATH_CALUDE_cos_pi_4_minus_alpha_l1372_137283

theorem cos_pi_4_minus_alpha (α : ℝ) (h : Real.sin (π / 4 + α) = 2 / 3) :
  Real.cos (π / 4 - α) = -Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_4_minus_alpha_l1372_137283


namespace NUMINAMATH_CALUDE_k_gt_one_sufficient_k_gt_one_not_necessary_k_gt_one_sufficient_not_necessary_l1372_137271

/-- The equation of a possible hyperbola -/
def hyperbola_equation (k x y : ℝ) : Prop :=
  x^2 / (k - 1) - y^2 / (k + 1) = 1

/-- Condition for the equation to represent a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  (k - 1) * (k + 1) > 0

/-- k > 1 is sufficient for the equation to represent a hyperbola -/
theorem k_gt_one_sufficient (k : ℝ) (h : k > 1) : is_hyperbola k := by sorry

/-- k > 1 is not necessary for the equation to represent a hyperbola -/
theorem k_gt_one_not_necessary : ∃ k : ℝ, is_hyperbola k ∧ ¬(k > 1) := by sorry

/-- k > 1 is a sufficient but not necessary condition for the equation to represent a hyperbola -/
theorem k_gt_one_sufficient_not_necessary :
  (∀ k : ℝ, k > 1 → is_hyperbola k) ∧ (∃ k : ℝ, is_hyperbola k ∧ ¬(k > 1)) := by sorry

end NUMINAMATH_CALUDE_k_gt_one_sufficient_k_gt_one_not_necessary_k_gt_one_sufficient_not_necessary_l1372_137271
