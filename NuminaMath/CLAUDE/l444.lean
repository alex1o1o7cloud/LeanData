import Mathlib

namespace NUMINAMATH_CALUDE_containers_per_truck_is_160_l444_44473

/-- The number of trucks with 20 boxes each -/
def trucks_with_20_boxes : ℕ := 7

/-- The number of trucks with 12 boxes each -/
def trucks_with_12_boxes : ℕ := 5

/-- The number of boxes on trucks with 20 boxes -/
def boxes_on_20_box_trucks : ℕ := 20

/-- The number of boxes on trucks with 12 boxes -/
def boxes_on_12_box_trucks : ℕ := 12

/-- The number of containers of oil in each box -/
def containers_per_box : ℕ := 8

/-- The number of trucks for redistribution -/
def redistribution_trucks : ℕ := 10

/-- The total number of containers of oil -/
def total_containers : ℕ := 
  (trucks_with_20_boxes * boxes_on_20_box_trucks + 
   trucks_with_12_boxes * boxes_on_12_box_trucks) * containers_per_box

/-- The number of containers per truck after redistribution -/
def containers_per_truck : ℕ := total_containers / redistribution_trucks

theorem containers_per_truck_is_160 : containers_per_truck = 160 := by
  sorry

end NUMINAMATH_CALUDE_containers_per_truck_is_160_l444_44473


namespace NUMINAMATH_CALUDE_prob_not_blue_marble_l444_44459

/-- Given a bag of marbles where the odds of drawing a blue marble are 5:9,
    the probability of not drawing a blue marble is 9/14. -/
theorem prob_not_blue_marble (odds_blue : ℚ) (h : odds_blue = 5 / 9) :
  1 - odds_blue / (1 + odds_blue) = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_blue_marble_l444_44459


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l444_44483

/-- A line with equation x = my + 2 is tangent to the circle x^2 + 2x + y^2 + 2y = 0 
    if and only if m = 1 or m = -7 -/
theorem line_tangent_to_circle (m : ℝ) : 
  (∀ x y : ℝ, x = m * y + 2 → x^2 + 2*x + y^2 + 2*y ≠ 0) ∧ 
  (∃ x y : ℝ, x = m * y + 2 ∧ x^2 + 2*x + y^2 + 2*y = 0) ↔ 
  m = 1 ∨ m = -7 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l444_44483


namespace NUMINAMATH_CALUDE_temperature_decrease_l444_44463

theorem temperature_decrease (initial_temp final_temp decrease : ℤ) :
  initial_temp = -3 →
  decrease = 6 →
  final_temp = initial_temp - decrease →
  final_temp = -9 :=
by sorry

end NUMINAMATH_CALUDE_temperature_decrease_l444_44463


namespace NUMINAMATH_CALUDE_total_glasses_displayed_l444_44477

/-- Represents the number of cupboards of each type -/
def num_tall_cupboards : ℕ := 2
def num_wide_cupboards : ℕ := 2
def num_narrow_cupboards : ℕ := 2

/-- Represents the capacity of each type of cupboard -/
def tall_cupboard_capacity : ℕ := 30
def wide_cupboard_capacity : ℕ := 2 * tall_cupboard_capacity
def narrow_cupboard_capacity : ℕ := 45

/-- Represents the number of shelves in a narrow cupboard -/
def shelves_per_narrow_cupboard : ℕ := 3

/-- Represents the number of broken shelves -/
def broken_shelves : ℕ := 1

/-- Theorem stating the total number of glasses displayed -/
theorem total_glasses_displayed : 
  num_tall_cupboards * tall_cupboard_capacity +
  num_wide_cupboards * wide_cupboard_capacity +
  (num_narrow_cupboards * narrow_cupboard_capacity - 
   broken_shelves * (narrow_cupboard_capacity / shelves_per_narrow_cupboard)) = 255 := by
  sorry

end NUMINAMATH_CALUDE_total_glasses_displayed_l444_44477


namespace NUMINAMATH_CALUDE_magnitude_of_vector_combination_l444_44493

/-- Given two vectors a and b in R^2, prove that the magnitude of 2a - b is √17 -/
theorem magnitude_of_vector_combination (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (3, -2) → ‖2 • a - b‖ = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_combination_l444_44493


namespace NUMINAMATH_CALUDE_percentage_of_C_grades_l444_44496

def gradeC (score : ℕ) : Bool :=
  76 ≤ score ∧ score ≤ 85

def scores : List ℕ := [93, 71, 55, 98, 81, 89, 77, 72, 78, 62, 87, 80, 68, 82, 91, 67, 76, 84, 70, 95]

theorem percentage_of_C_grades (scores : List ℕ) : 
  (100 * (scores.filter gradeC).length) / scores.length = 35 :=
sorry

end NUMINAMATH_CALUDE_percentage_of_C_grades_l444_44496


namespace NUMINAMATH_CALUDE_sum_of_ages_l444_44453

/-- Given the ages of Eunji, Yuna, and Eunji's uncle, prove that the sum of Eunji's and Yuna's ages is 35 years. -/
theorem sum_of_ages (uncle_age : ℕ) (eunji_age : ℕ) (yuna_age : ℕ)
  (h1 : uncle_age = 41)
  (h2 : uncle_age = eunji_age + 25)
  (h3 : yuna_age = eunji_age + 3) :
  eunji_age + yuna_age = 35 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_ages_l444_44453


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_powers_l444_44438

theorem infinitely_many_divisible_powers (a b c : ℤ) 
  (h : (a + b + c) ∣ (a^2 + b^2 + c^2)) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, (a + b + c) ∣ (a^n + b^n + c^n) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_powers_l444_44438


namespace NUMINAMATH_CALUDE_range_of_a_l444_44448

-- Define the propositions p and q
def p (x : ℝ) : Prop := abs (4 * x - 3) ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, 
  (∀ x : ℝ, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x : ℝ, ¬(p x) ∧ q x a) → 
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l444_44448


namespace NUMINAMATH_CALUDE_book_sale_loss_l444_44451

/-- Represents the sale of two books with given conditions -/
def book_sale (total_cost cost_book1 loss_percent1 gain_percent2 : ℚ) : ℚ :=
  let cost_book2 := total_cost - cost_book1
  let selling_price1 := cost_book1 * (1 - loss_percent1 / 100)
  let selling_price2 := cost_book2 * (1 + gain_percent2 / 100)
  let total_selling_price := selling_price1 + selling_price2
  total_cost - total_selling_price

/-- Theorem stating the overall loss from the book sale -/
theorem book_sale_loss :
  book_sale 460 268.33 15 19 = 3.8322 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_loss_l444_44451


namespace NUMINAMATH_CALUDE_product_of_solutions_l444_44401

theorem product_of_solutions (x₁ x₂ : ℚ) : 
  (|6 * x₁ + 2| + 5 = 47) → 
  (|6 * x₂ + 2| + 5 = 47) → 
  x₁ ≠ x₂ → 
  x₁ * x₂ = -440 / 9 := by
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l444_44401


namespace NUMINAMATH_CALUDE_folded_rectangle_area_l444_44461

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  topLeft : Point
  bottomRight : Point

/-- Represents the folding scenario -/
structure FoldedRectangle where
  original : Rectangle
  t : Point
  u : Point
  qPrime : Point
  pPrime : Point

theorem folded_rectangle_area (fold : FoldedRectangle) 
  (h1 : fold.t.x - fold.original.topLeft.x < fold.original.bottomRight.x - fold.u.x)
  (h2 : (fold.pPrime.x - fold.qPrime.x)^2 + (fold.pPrime.y - fold.qPrime.y)^2 = 
        (fold.original.bottomRight.y - fold.original.topLeft.y)^2)
  (h3 : fold.qPrime.x - fold.original.topLeft.x = 8)
  (h4 : fold.t.x - fold.original.topLeft.x = 36) :
  (fold.original.bottomRight.x - fold.original.topLeft.x) * 
  (fold.original.bottomRight.y - fold.original.topLeft.y) = 288 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_folded_rectangle_area_l444_44461


namespace NUMINAMATH_CALUDE_sine_cosine_equation_solution_l444_44471

theorem sine_cosine_equation_solution (x : ℝ) :
  12 * Real.sin x - 5 * Real.cos x = 13 →
  ∃ k : ℤ, x = π / 2 + Real.arctan (5 / 12) + 2 * π * ↑k :=
by sorry

end NUMINAMATH_CALUDE_sine_cosine_equation_solution_l444_44471


namespace NUMINAMATH_CALUDE_average_page_count_l444_44466

theorem average_page_count (n : ℕ) (g1 g2 g3 : ℕ) (p1 p2 p3 : ℕ) :
  n = g1 + g2 + g3 →
  g1 = g2 →
  g2 = g3 →
  g1 = 5 →
  p1 = 2 →
  p2 = 3 →
  p3 = 1 →
  (g1 * p1 + g2 * p2 + g3 * p3) / n = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_page_count_l444_44466


namespace NUMINAMATH_CALUDE_triangulation_count_l444_44494

/-- A triangulation of a square with marked interior points. -/
structure SquareTriangulation where
  /-- The number of marked points inside the square. -/
  num_points : ℕ
  /-- The number of triangles in the triangulation. -/
  num_triangles : ℕ

/-- Theorem stating the number of triangles in a specific triangulation. -/
theorem triangulation_count (t : SquareTriangulation) 
  (h_points : t.num_points = 100) : 
  t.num_triangles = 202 := by sorry

end NUMINAMATH_CALUDE_triangulation_count_l444_44494


namespace NUMINAMATH_CALUDE_odd_digits_base4_523_l444_44442

/-- Represents a digit in base 4 --/
def Base4Digit := Fin 4

/-- Converts a natural number to its base 4 representation --/
def toBase4 (n : ℕ) : List Base4Digit := sorry

/-- Checks if a Base4Digit is odd --/
def isOddBase4Digit (d : Base4Digit) : Bool := sorry

/-- Counts the number of odd digits in a list of Base4Digits --/
def countOddDigits (digits : List Base4Digit) : ℕ := sorry

theorem odd_digits_base4_523 :
  countOddDigits (toBase4 523) = 2 := by sorry

end NUMINAMATH_CALUDE_odd_digits_base4_523_l444_44442


namespace NUMINAMATH_CALUDE_angle_from_point_l444_44434

/-- Given a point A with coordinates (sin 23°, -cos 23°) on the terminal side of angle α,
    where 0° < α < 360°, prove that α = 293°. -/
theorem angle_from_point (α : Real) : 
  0 < α ∧ α < 360 ∧ 
  (∃ (A : ℝ × ℝ), A.1 = Real.sin (23 * π / 180) ∧ A.2 = -Real.cos (23 * π / 180) ∧ 
    A.1 = Real.cos α ∧ A.2 = Real.sin α) →
  α = 293 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_from_point_l444_44434


namespace NUMINAMATH_CALUDE_initial_money_calculation_l444_44416

/-- Proves that the initial amount of money is $160 given the conditions of the problem -/
theorem initial_money_calculation (your_weekly_savings : ℕ) (friend_initial_money : ℕ) 
  (friend_weekly_savings : ℕ) (weeks : ℕ) (h1 : your_weekly_savings = 7) 
  (h2 : friend_initial_money = 210) (h3 : friend_weekly_savings = 5) (h4 : weeks = 25) :
  ∃ (your_initial_money : ℕ), 
    your_initial_money + your_weekly_savings * weeks = 
    friend_initial_money + friend_weekly_savings * weeks ∧ 
    your_initial_money = 160 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l444_44416


namespace NUMINAMATH_CALUDE_park_breadth_l444_44439

/-- The breadth of a rectangular park given its perimeter and length -/
theorem park_breadth (perimeter length breadth : ℝ) : 
  perimeter = 1000 →
  length = 300 →
  perimeter = 2 * (length + breadth) →
  breadth = 200 := by
sorry

end NUMINAMATH_CALUDE_park_breadth_l444_44439


namespace NUMINAMATH_CALUDE_base_eight_47_to_base_ten_l444_44456

/-- Converts a two-digit base-eight number to base-ten -/
def base_eight_to_ten (d1 d2 : Nat) : Nat :=
  d1 * 8 + d2

/-- The base-eight number 47 -/
def base_eight_47 : Nat × Nat := (4, 7)

theorem base_eight_47_to_base_ten :
  base_eight_to_ten base_eight_47.1 base_eight_47.2 = 39 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_47_to_base_ten_l444_44456


namespace NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l444_44417

theorem no_real_solution_for_log_equation :
  ¬ ∃ (x : ℝ), (Real.log (x + 4) + Real.log (x - 2) = Real.log (x^2 - 6*x - 5)) ∧
               (x + 4 > 0) ∧ (x - 2 > 0) ∧ (x^2 - 6*x - 5 > 0) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l444_44417


namespace NUMINAMATH_CALUDE_spinner_probability_l444_44475

/-- Given a spinner with three regions A, B, and C, where the probability of
    stopping on A is 1/2 and on B is 1/5, prove that the probability of
    stopping on C is 3/10. -/
theorem spinner_probability (p_A p_B p_C : ℚ) : 
  p_A = 1/2 → p_B = 1/5 → p_A + p_B + p_C = 1 → p_C = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l444_44475


namespace NUMINAMATH_CALUDE_pyramid_a_value_l444_44441

/-- Represents a pyramid of numbers where each number is the product of the two numbers above it. -/
structure Pyramid where
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ
  e : ℚ
  f : ℚ
  second_row_prod : b * c = a
  third_row_left_prod : d = 20
  third_row_middle_prod : e = 30
  third_row_right_prod : f = 72

/-- Theorem stating that given b = 6 and d = 20 in the pyramid, a must equal 54. -/
theorem pyramid_a_value (p : Pyramid) (h_b : p.b = 6) : p.a = 54 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_a_value_l444_44441


namespace NUMINAMATH_CALUDE_min_distance_squared_l444_44427

/-- Given real numbers a, b, c, d satisfying the condition,
    the minimum value of (a - c)^2 + (b - d)^2 is 25/2 -/
theorem min_distance_squared (a b c d : ℝ) 
  (h : (a - 2 * Real.exp a) / b = (2 - c) / (d - 1) ∧ (a - 2 * Real.exp a) / b = 1) :
  ∃ (min : ℝ), min = 25 / 2 ∧ ∀ (x y : ℝ), 
    (x - 2 * Real.exp x) / y = (2 - c) / (d - 1) ∧ (x - 2 * Real.exp x) / y = 1 →
    (x - c)^2 + (y - d)^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_squared_l444_44427


namespace NUMINAMATH_CALUDE_least_possible_lcm_a_c_l444_44435

theorem least_possible_lcm_a_c (a b c : ℕ) 
  (h1 : Nat.lcm a b = 18) 
  (h2 : Nat.lcm b c = 20) : 
  ∃ (a' c' : ℕ), Nat.lcm a' c' = 90 ∧ 
    (∀ (x y : ℕ), Nat.lcm x b = 18 → Nat.lcm b y = 20 → Nat.lcm a' c' ≤ Nat.lcm x y) := by
  sorry

end NUMINAMATH_CALUDE_least_possible_lcm_a_c_l444_44435


namespace NUMINAMATH_CALUDE_apples_left_over_l444_44436

theorem apples_left_over (liam mia noah : ℕ) (h1 : liam = 53) (h2 : mia = 68) (h3 : noah = 22) : 
  (liam + mia + noah) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_over_l444_44436


namespace NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l444_44404

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ+, n ∣ (2^n.val - 1) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l444_44404


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_power_imaginary_part_of_specific_complex_l444_44491

theorem imaginary_part_of_complex_power (r θ : ℝ) (n : ℕ) :
  let z := (r * (Complex.cos θ + Complex.I * Complex.sin θ)) ^ n
  Complex.im z = r^n * Real.sin (n * θ) := by sorry

theorem imaginary_part_of_specific_complex (π : ℝ) :
  let z := (2 * (Complex.cos (π/4) + Complex.I * Complex.sin (π/4))) ^ 5
  Complex.im z = -16 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_power_imaginary_part_of_specific_complex_l444_44491


namespace NUMINAMATH_CALUDE_right_triangle_median_on_hypotenuse_l444_44418

/-- Given a right triangle with legs of lengths 6 and 8, 
    the length of the median on the hypotenuse is 5. -/
theorem right_triangle_median_on_hypotenuse : 
  ∀ (a b c m : ℝ), 
    a = 6 → 
    b = 8 → 
    c^2 = a^2 + b^2 → 
    m = c / 2 → 
    m = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_median_on_hypotenuse_l444_44418


namespace NUMINAMATH_CALUDE_negation_of_universal_inequality_l444_44470

theorem negation_of_universal_inequality : 
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 + 1 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_inequality_l444_44470


namespace NUMINAMATH_CALUDE_eagle_types_total_l444_44409

theorem eagle_types_total (types_per_section : ℕ) (num_sections : ℕ) (h1 : types_per_section = 6) (h2 : num_sections = 3) :
  types_per_section * num_sections = 18 := by
  sorry

end NUMINAMATH_CALUDE_eagle_types_total_l444_44409


namespace NUMINAMATH_CALUDE_ellipse_m_value_l444_44432

/-- The value of m for an ellipse with given properties -/
theorem ellipse_m_value (m : ℝ) (h1 : m > 0) : 
  (∀ x y : ℝ, x^2 / 25 + y^2 / m^2 = 1) →
  (∃ c : ℝ, c = 4 ∧ c^2 = 25 - m^2) →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l444_44432


namespace NUMINAMATH_CALUDE_g_2010_equals_one_l444_44407

/-- A function satisfying the given properties -/
def g_function (g : ℝ → ℝ) : Prop :=
  (∀ x > 0, g x > 0) ∧ 
  (∀ x y, x > y ∧ y > 0 → g (x - y) = (g (x * y) + 1) ^ (1/3)) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x - y = x * y ∧ x * y = 2010)

/-- The main theorem stating that g(2010) = 1 -/
theorem g_2010_equals_one (g : ℝ → ℝ) (h : g_function g) : g 2010 = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_2010_equals_one_l444_44407


namespace NUMINAMATH_CALUDE_horner_rule_v₃_l444_44479

/-- Horner's Rule for a polynomial of degree 6 -/
def horner_rule (a₀ a₁ a₂ a₃ a₄ a₅ a₆ x : ℝ) : ℝ :=
  ((((((a₆ * x + a₅) * x + a₄) * x + a₃) * x + a₂) * x + a₁) * x + a₀)

/-- The third intermediate value in Horner's Rule calculation -/
def v₃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ x : ℝ) : ℝ :=
  (((a₆ * x + a₅) * x + a₄) * x + a₃)

theorem horner_rule_v₃ :
  v₃ 64 (-192) 240 (-160) 60 (-12) 1 2 = -80 :=
sorry

end NUMINAMATH_CALUDE_horner_rule_v₃_l444_44479


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l444_44400

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

-- Theorem statement
theorem subset_implies_a_equals_one :
  ∀ a : ℝ, A a ⊆ B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l444_44400


namespace NUMINAMATH_CALUDE_triangle_side_length_l444_44411

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a = 3 ∧ c = 5 ∧ B = 2 * A ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  b = 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l444_44411


namespace NUMINAMATH_CALUDE_inequality_solution_set_l444_44481

theorem inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, (x - 1) * (x + a) > 0 ↔
    (a < -1 ∧ (x < -a ∨ x > 1)) ∨
    (a = -1 ∧ x ≠ 1) ∨
    (a > -1 ∧ (x < -a ∨ x > 1))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l444_44481


namespace NUMINAMATH_CALUDE_circle_tangency_line_intersection_l444_44482

-- Define the circle C
def circle_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 8*y + m = 0

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop :=
  x + y - 3 = 0

-- Part I
theorem circle_tangency (m : ℝ) :
  (∃ x y : ℝ, circle_C m x y ∧ unit_circle x y) →
  (∀ x y : ℝ, circle_C m x y → ¬(unit_circle x y)) →
  m = 9 :=
sorry

-- Part II
theorem line_intersection (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_C m x₁ y₁ ∧ circle_C m x₂ y₂ ∧
    line x₁ y₁ ∧ line x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 14) →
  m = 10 :=
sorry

end NUMINAMATH_CALUDE_circle_tangency_line_intersection_l444_44482


namespace NUMINAMATH_CALUDE_ace_distribution_probability_l444_44484

def num_players : ℕ := 4
def num_cards : ℕ := 32
def num_aces : ℕ := 4
def cards_per_player : ℕ := num_cards / num_players

theorem ace_distribution_probability :
  let remaining_players := num_players - 1
  let remaining_cards := num_cards - cards_per_player
  let p_no_ace_for_one := 1 / num_players
  let p_two_aces_for_others := 
    (Nat.choose remaining_players 1 * Nat.choose num_aces 2 * Nat.choose (remaining_cards - num_aces) (cards_per_player - 2)) /
    (Nat.choose remaining_cards cards_per_player)
  p_two_aces_for_others = 8 / 11 :=
sorry

end NUMINAMATH_CALUDE_ace_distribution_probability_l444_44484


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l444_44455

/-- Given that 3/4 of 12 bananas are worth 9 oranges, 
    prove that 1/3 of 6 bananas are worth 2 oranges -/
theorem banana_orange_equivalence (banana orange : ℚ) 
  (h : (3/4 : ℚ) * 12 * banana = 9 * orange) : 
  (1/3 : ℚ) * 6 * banana = 2 * orange := by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l444_44455


namespace NUMINAMATH_CALUDE_sock_distribution_l444_44428

-- Define the total number of socks
def total_socks : ℕ := 9

-- Define the property that among any 4 socks, at least 2 belong to the same child
def at_least_two_same (socks : Finset ℕ) : Prop :=
  ∀ (s : Finset ℕ), s ⊆ socks → s.card = 4 → ∃ (child : ℕ), (s.filter (λ x => x = child)).card ≥ 2

-- Define the property that among any 5 socks, no more than 3 belong to the same child
def no_more_than_three (socks : Finset ℕ) : Prop :=
  ∀ (s : Finset ℕ), s ⊆ socks → s.card = 5 → ∀ (child : ℕ), (s.filter (λ x => x = child)).card ≤ 3

-- Theorem statement
theorem sock_distribution (socks : Finset ℕ) 
  (h_total : socks.card = total_socks)
  (h_at_least_two : at_least_two_same socks)
  (h_no_more_than_three : no_more_than_three socks) :
  ∃ (children : Finset ℕ), 
    children.card = 3 ∧ 
    (∀ child ∈ children, (socks.filter (λ x => x = child)).card = 3) :=
sorry

end NUMINAMATH_CALUDE_sock_distribution_l444_44428


namespace NUMINAMATH_CALUDE_cube_order_equivalence_l444_44464

theorem cube_order_equivalence (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end NUMINAMATH_CALUDE_cube_order_equivalence_l444_44464


namespace NUMINAMATH_CALUDE_complete_square_k_value_l444_44403

theorem complete_square_k_value (x : ℝ) : 
  ∃ (p k : ℝ), (x^2 - 6*x + 5 = 0) ↔ ((x - p)^2 = k) ∧ k = 4 := by
sorry

end NUMINAMATH_CALUDE_complete_square_k_value_l444_44403


namespace NUMINAMATH_CALUDE_average_marks_l444_44402

structure Marks where
  physics : ℕ
  chemistry : ℕ
  mathematics : ℕ
  biology : ℕ
  english : ℕ
  history : ℕ
  geography : ℕ

def valid_marks (m : Marks) : Prop :=
  m.chemistry = m.physics + 75 ∧
  m.mathematics = m.chemistry + 30 ∧
  m.biology = m.physics - 15 ∧
  m.english = m.biology - 10 ∧
  m.history = m.biology - 10 ∧
  m.geography = m.biology - 10 ∧
  m.physics + m.chemistry + m.mathematics + m.biology + m.english + m.history + m.geography = m.physics + 520 ∧
  m.physics ≥ 40 ∧ m.chemistry ≥ 40 ∧ m.mathematics ≥ 40 ∧ m.biology ≥ 40 ∧
  m.english ≥ 40 ∧ m.history ≥ 40 ∧ m.geography ≥ 40

theorem average_marks (m : Marks) (h : valid_marks m) :
  (m.mathematics + m.biology + m.history + m.geography) / 4 = 82 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_l444_44402


namespace NUMINAMATH_CALUDE_largest_integer_quadratic_inequality_six_satisfies_inequality_seven_does_not_satisfy_inequality_l444_44419

theorem largest_integer_quadratic_inequality :
  ∀ n : ℤ, n^2 - 9*n + 14 < 0 → n ≤ 6 :=
by
  sorry

theorem six_satisfies_inequality :
  (6 : ℤ)^2 - 9*6 + 14 < 0 :=
by
  sorry

theorem seven_does_not_satisfy_inequality :
  (7 : ℤ)^2 - 9*7 + 14 ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_quadratic_inequality_six_satisfies_inequality_seven_does_not_satisfy_inequality_l444_44419


namespace NUMINAMATH_CALUDE_tomato_field_area_l444_44444

/-- Given a rectangular field with length 3.6 meters and width 2.5 times the length,
    the area of half of this field is 16.2 square meters. -/
theorem tomato_field_area :
  let length : ℝ := 3.6
  let width : ℝ := 2.5 * length
  let total_area : ℝ := length * width
  let tomato_area : ℝ := total_area / 2
  tomato_area = 16.2 := by
sorry

end NUMINAMATH_CALUDE_tomato_field_area_l444_44444


namespace NUMINAMATH_CALUDE_distance_A_to_C_l444_44492

/-- Proves that the distance between city A and C is 300 km given the provided conditions -/
theorem distance_A_to_C (
  eddy_time : ℝ)
  (freddy_time : ℝ)
  (distance_A_to_B : ℝ)
  (speed_ratio : ℝ)
  (h1 : eddy_time = 3)
  (h2 : freddy_time = 4)
  (h3 : distance_A_to_B = 510)
  (h4 : speed_ratio = 2.2666666666666666)
  : ℝ := by
  sorry

#check distance_A_to_C

end NUMINAMATH_CALUDE_distance_A_to_C_l444_44492


namespace NUMINAMATH_CALUDE_box_balls_count_l444_44443

theorem box_balls_count : ∃ x : ℕ, (x > 20 ∧ x < 30 ∧ x - 20 = 30 - x) ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_box_balls_count_l444_44443


namespace NUMINAMATH_CALUDE_nested_expression_evaluation_l444_44424

theorem nested_expression_evaluation : (2*(2*(2*(2*(2*(2*(3+2)+2)+2)+2)+2)+2)+2) = 446 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_evaluation_l444_44424


namespace NUMINAMATH_CALUDE_binomial_divisibility_l444_44469

theorem binomial_divisibility (n k : ℕ) (h : k ≤ n - 1) :
  (∀ k ≤ n - 1, n ∣ Nat.choose n k) ↔ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l444_44469


namespace NUMINAMATH_CALUDE_knives_percentage_after_trade_l444_44426

/-- Represents Carolyn's silverware set --/
structure SilverwareSet where
  knives : ℕ
  forks : ℕ
  spoons : ℕ

/-- Calculates the total number of pieces in a silverware set --/
def SilverwareSet.total (s : SilverwareSet) : ℕ :=
  s.knives + s.forks + s.spoons

/-- Represents the initial state of Carolyn's silverware set --/
def initial_set : SilverwareSet :=
  { knives := 6
  , forks := 12
  , spoons := 6 * 3 }

/-- Represents the trade operation --/
def trade (s : SilverwareSet) : SilverwareSet :=
  { knives := s.knives - 6
  , forks := s.forks
  , spoons := s.spoons + 6 }

/-- Theorem stating that after the trade, 0% of Carolyn's silverware is knives --/
theorem knives_percentage_after_trade :
  let final_set := trade initial_set
  (final_set.knives : ℚ) / (final_set.total : ℚ) * 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_knives_percentage_after_trade_l444_44426


namespace NUMINAMATH_CALUDE_books_about_science_l444_44458

theorem books_about_science 
  (total_books : ℕ) 
  (school_books : ℕ) 
  (sports_books : ℕ) 
  (h1 : total_books = 85) 
  (h2 : school_books = 19) 
  (h3 : sports_books = 35) :
  total_books - (school_books + sports_books) = 31 :=
by sorry

end NUMINAMATH_CALUDE_books_about_science_l444_44458


namespace NUMINAMATH_CALUDE_research_budget_allocation_l444_44454

theorem research_budget_allocation (microphotonics : ℝ) (home_electronics : ℝ) 
  (food_additives : ℝ) (industrial_lubricants : ℝ) (basic_astrophysics_degrees : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 20 →
  industrial_lubricants = 8 →
  basic_astrophysics_degrees = 18 →
  ∃ (genetically_modified_microorganisms : ℝ),
    genetically_modified_microorganisms = 29 ∧
    microphotonics + home_electronics + food_additives + industrial_lubricants + 
    (basic_astrophysics_degrees / 360 * 100) + genetically_modified_microorganisms = 100 :=
by sorry

end NUMINAMATH_CALUDE_research_budget_allocation_l444_44454


namespace NUMINAMATH_CALUDE_parabola_translation_l444_44413

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola -/
def original_parabola : Parabola := { a := 1, b := -2, c := 4 }

/-- Translates a parabola vertically -/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + d }

/-- Translates a parabola horizontally -/
def translate_horizontal (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b - 2 * p.a * d, c := p.c + p.a * d^2 - p.b * d }

/-- The resulting parabola after translation -/
def translated_parabola : Parabola :=
  translate_horizontal (translate_vertical original_parabola 3) 1

theorem parabola_translation :
  translated_parabola = { a := 1, b := -4, c := 10 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l444_44413


namespace NUMINAMATH_CALUDE_x_range_proof_l444_44415

theorem x_range_proof (x : ℝ) : 
  (∀ θ : ℝ, 0 < θ ∧ θ < π/2 → 1/(Real.sin θ)^2 + 4/(Real.cos θ)^2 ≥ |2*x - 1|) 
  ↔ -4 ≤ x ∧ x ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_x_range_proof_l444_44415


namespace NUMINAMATH_CALUDE_sandwich_shop_jalapeno_requirement_l444_44457

/-- Represents the number of jalapeno peppers required for a day's operation --/
def jalapeno_peppers_required (strips_per_sandwich : ℕ) (slices_per_pepper : ℕ) 
  (minutes_per_sandwich : ℕ) (hours_of_operation : ℕ) : ℕ :=
  let peppers_per_sandwich := strips_per_sandwich / slices_per_pepper
  let sandwiches_per_hour := 60 / minutes_per_sandwich
  let peppers_per_hour := peppers_per_sandwich * sandwiches_per_hour
  peppers_per_hour * hours_of_operation

/-- Theorem stating the number of jalapeno peppers required for the Sandwich Shop's 8-hour day --/
theorem sandwich_shop_jalapeno_requirement :
  jalapeno_peppers_required 4 8 5 8 = 48 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_shop_jalapeno_requirement_l444_44457


namespace NUMINAMATH_CALUDE_decreasing_linear_function_not_in_third_quadrant_l444_44450

/-- A linear function y = kx + 1 where k ≠ 0 and y decreases as x increases -/
structure DecreasingLinearFunction where
  k : ℝ
  hk_nonzero : k ≠ 0
  hk_negative : k < 0

/-- The third quadrant -/
def ThirdQuadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 < 0}

/-- The graph of a linear function y = kx + 1 -/
def LinearFunctionGraph (f : DecreasingLinearFunction) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = f.k * p.1 + 1}

/-- The theorem stating that the graph of a decreasing linear function
    does not pass through the third quadrant -/
theorem decreasing_linear_function_not_in_third_quadrant
  (f : DecreasingLinearFunction) :
  LinearFunctionGraph f ∩ ThirdQuadrant = ∅ := by
  sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_not_in_third_quadrant_l444_44450


namespace NUMINAMATH_CALUDE_hour_hand_path_l444_44405

/-- The number of times the hour hand covers its path in a day -/
def coverages_per_day : ℕ := 2

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of hours for one full rotation of the hour hand -/
def hours_per_rotation : ℕ := 12

/-- The path covered by the hour hand in one rotation, in degrees -/
def path_per_rotation : ℝ := 360

theorem hour_hand_path :
  path_per_rotation = 360 :=
sorry

end NUMINAMATH_CALUDE_hour_hand_path_l444_44405


namespace NUMINAMATH_CALUDE_exterior_angle_not_sum_of_adjacent_angles_l444_44485

-- Define a triangle with interior angles A, B, C and exterior angle A_ext at vertex A
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  A_ext : ℝ

-- State the theorem
theorem exterior_angle_not_sum_of_adjacent_angles (t : Triangle) : 
  t.A_ext ≠ t.B + t.C :=
sorry

end NUMINAMATH_CALUDE_exterior_angle_not_sum_of_adjacent_angles_l444_44485


namespace NUMINAMATH_CALUDE_product_of_roots_quadratic_equation_l444_44410

theorem product_of_roots_quadratic_equation :
  ∀ x₁ x₂ : ℝ, (x₁^2 + x₁ - 2 = 0) → (x₂^2 + x₂ - 2 = 0) → x₁ * x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_quadratic_equation_l444_44410


namespace NUMINAMATH_CALUDE_coefficient_value_l444_44465

-- Define the polynomial Q(x)
def Q (d : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + d*x - 8

-- State the theorem
theorem coefficient_value (d : ℝ) :
  (∀ x, (x + 2 : ℝ) ∣ Q d x) → d = -2 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_value_l444_44465


namespace NUMINAMATH_CALUDE_speaking_orders_count_l444_44408

def total_people : Nat := 7
def speakers : Nat := 4
def special_people : Nat := 2  -- A and B

theorem speaking_orders_count : 
  (total_people.choose speakers * speakers.factorial - 
   (total_people - special_people).choose speakers * speakers.factorial) = 720 := by
  sorry

end NUMINAMATH_CALUDE_speaking_orders_count_l444_44408


namespace NUMINAMATH_CALUDE_cars_given_to_vinnie_l444_44449

def initial_cars : ℕ := 14
def bought_cars : ℕ := 28
def birthday_cars : ℕ := 12
def cars_to_sister : ℕ := 8
def cars_left : ℕ := 43

theorem cars_given_to_vinnie :
  initial_cars + bought_cars + birthday_cars - cars_to_sister - cars_left = 3 :=
by sorry

end NUMINAMATH_CALUDE_cars_given_to_vinnie_l444_44449


namespace NUMINAMATH_CALUDE_min_exposed_surface_area_l444_44462

/- Define a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ
  volume_eq : length * width * height = 128
  positive : length > 0 ∧ width > 0 ∧ height > 0

/- Define the three solids -/
def solid1 : RectangularSolid := {
  length := 4,
  width := 1,
  height := 32,
  volume_eq := by norm_num,
  positive := by simp
}

def solid2 : RectangularSolid := {
  length := 8,
  width := 8,
  height := 2,
  volume_eq := by norm_num,
  positive := by simp
}

def solid3 : RectangularSolid := {
  length := 4,
  width := 2,
  height := 16,
  volume_eq := by norm_num,
  positive := by simp
}

/- Calculate the exposed surface area of the tower -/
def exposedSurfaceArea (s1 s2 s3 : RectangularSolid) : ℝ :=
  2 * (s1.length * s1.width + s2.length * s2.width + s3.length * s3.width) +
  2 * (s1.length * s1.height + s2.length * s2.height + s3.length * s3.height) +
  2 * (s1.width * s1.height + s2.width * s2.height + s3.width * s3.height) -
  2 * (s1.length * s1.width + s2.length * s2.width)

/- Theorem statement -/
theorem min_exposed_surface_area :
  exposedSurfaceArea solid1 solid2 solid3 = 832 := by sorry

end NUMINAMATH_CALUDE_min_exposed_surface_area_l444_44462


namespace NUMINAMATH_CALUDE_negative_abs_negative_five_l444_44474

theorem negative_abs_negative_five : -|-5| = -5 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_negative_five_l444_44474


namespace NUMINAMATH_CALUDE_total_tabs_is_300_l444_44412

/-- Calculates the total number of tabs opened across all browsers --/
def totalTabs (numBrowsers : ℕ) (windowsPerBrowser : ℕ) (initialTabsPerWindow : ℕ) (additionalTabsPerTwelve : ℕ) : ℕ :=
  let tabsPerWindow := initialTabsPerWindow + additionalTabsPerTwelve
  let tabsPerBrowser := tabsPerWindow * windowsPerBrowser
  tabsPerBrowser * numBrowsers

/-- Proves that the total number of tabs is 300 given the specified conditions --/
theorem total_tabs_is_300 :
  totalTabs 4 5 12 3 = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_tabs_is_300_l444_44412


namespace NUMINAMATH_CALUDE_triangle_area_bounds_l444_44478

/-- The parabola function y = x^2 - 1 -/
def parabola (x : ℝ) : ℝ := x^2 - 1

/-- The line function y = r -/
def line (r : ℝ) (x : ℝ) : ℝ := r

/-- The area of the triangle formed by the vertex of the parabola and its intersections with the line y = r -/
def triangleArea (r : ℝ) : ℝ := (r + 1)^(3/2)

theorem triangle_area_bounds (r : ℝ) :
  (8 ≤ triangleArea r ∧ triangleArea r ≤ 64) → (3 ≤ r ∧ r ≤ 15) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_bounds_l444_44478


namespace NUMINAMATH_CALUDE_fraction_power_seven_l444_44440

theorem fraction_power_seven : (5 / 7 : ℚ) ^ 7 = 78125 / 823543 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_seven_l444_44440


namespace NUMINAMATH_CALUDE_simple_interest_double_rate_l444_44490

/-- The rate of interest for simple interest when a sum doubles in 10 years -/
theorem simple_interest_double_rate : 
  ∀ (principal : ℝ) (rate : ℝ),
  principal > 0 →
  principal * (1 + rate * 10) = 2 * principal →
  rate = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_double_rate_l444_44490


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l444_44429

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 4*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 4

-- Define the point of interest
def point : ℝ × ℝ := (1, -3)

-- Theorem statement
theorem tangent_slope_angle :
  let slope := f' point.1
  let angle := Real.arctan slope
  angle = 3 * π / 4 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l444_44429


namespace NUMINAMATH_CALUDE_employment_agency_payroll_l444_44499

/-- Calculates the total payroll for an employment agency given the number of employees,
    number of laborers, and pay rates for heavy operators and laborers. -/
theorem employment_agency_payroll
  (total_employees : ℕ)
  (num_laborers : ℕ)
  (heavy_operator_pay : ℕ)
  (laborer_pay : ℕ)
  (h1 : total_employees = 31)
  (h2 : num_laborers = 1)
  (h3 : heavy_operator_pay = 129)
  (h4 : laborer_pay = 82) :
  (total_employees - num_laborers) * heavy_operator_pay + num_laborers * laborer_pay = 3952 :=
by
  sorry


end NUMINAMATH_CALUDE_employment_agency_payroll_l444_44499


namespace NUMINAMATH_CALUDE_flour_sugar_difference_l444_44422

/-- Given a recipe and current ingredients, calculate the difference between additional flour needed and total sugar needed. -/
theorem flour_sugar_difference 
  (total_flour : ℕ) 
  (total_sugar : ℕ) 
  (added_flour : ℕ) 
  (h1 : total_flour = 10) 
  (h2 : total_sugar = 2) 
  (h3 : added_flour = 7) : 
  (total_flour - added_flour) - total_sugar = 1 := by
  sorry

#check flour_sugar_difference

end NUMINAMATH_CALUDE_flour_sugar_difference_l444_44422


namespace NUMINAMATH_CALUDE_paul_reading_time_l444_44468

/-- The number of hours Paul spent reading after nine weeks -/
def reading_hours (books_per_week : ℕ) (pages_per_book : ℕ) (pages_per_hour : ℕ) (weeks : ℕ) : ℕ :=
  books_per_week * pages_per_book * weeks / pages_per_hour

/-- Theorem stating that Paul spent 540 hours reading after nine weeks -/
theorem paul_reading_time : reading_hours 10 300 50 9 = 540 := by
  sorry

end NUMINAMATH_CALUDE_paul_reading_time_l444_44468


namespace NUMINAMATH_CALUDE_train_speed_problem_l444_44472

/-- Proves that the initial speed of a train is 110 km/h given specific journey conditions -/
theorem train_speed_problem (T : ℝ) : ∃ v : ℝ,
  v > 0 ∧
  v - 50 > 0 ∧
  T > 0 ∧
  T + 2/3 = 212/v + 88/(v - 50) ∧
  v = 110 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l444_44472


namespace NUMINAMATH_CALUDE_vacation_cost_share_l444_44495

/-- Calculates each person's share of vacation costs -/
theorem vacation_cost_share
  (num_people : ℕ)
  (airbnb_cost : ℕ)
  (car_cost : ℕ)
  (h1 : num_people = 8)
  (h2 : airbnb_cost = 3200)
  (h3 : car_cost = 800) :
  (airbnb_cost + car_cost) / num_people = 500 := by
  sorry

#check vacation_cost_share

end NUMINAMATH_CALUDE_vacation_cost_share_l444_44495


namespace NUMINAMATH_CALUDE_cube_surface_area_l444_44480

/-- The surface area of a cube with side length 8 centimeters is 384 square centimeters. -/
theorem cube_surface_area : 
  let side_length : ℝ := 8
  let surface_area : ℝ := 6 * side_length * side_length
  surface_area = 384 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l444_44480


namespace NUMINAMATH_CALUDE_square_difference_ends_with_two_l444_44406

theorem square_difference_ends_with_two (a b : ℕ) (h1 : a^2 > b^2) 
  (h2 : ∃ (m n : ℕ), a^2 = m^2 ∧ b^2 = n^2) 
  (h3 : (a^2 - b^2) % 10 = 2) :
  a^2 % 10 = 6 ∧ b^2 % 10 = 4 := by
sorry

end NUMINAMATH_CALUDE_square_difference_ends_with_two_l444_44406


namespace NUMINAMATH_CALUDE_paint_calculation_l444_44488

theorem paint_calculation (P : ℚ) : 
  (1/6 : ℚ) * P + (1/5 : ℚ) * (P - (1/6 : ℚ) * P) = 120 → P = 360 := by
sorry

end NUMINAMATH_CALUDE_paint_calculation_l444_44488


namespace NUMINAMATH_CALUDE_lawrence_county_kids_at_home_l444_44498

/-- The number of kids who stay home during the break in Lawrence county -/
def kids_staying_home (total_kids : ℕ) (kids_at_camp : ℕ) : ℕ :=
  total_kids - kids_at_camp

/-- Theorem stating the number of kids staying home during the break in Lawrence county -/
theorem lawrence_county_kids_at_home :
  kids_staying_home 313473 38608 = 274865 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_at_home_l444_44498


namespace NUMINAMATH_CALUDE_min_xy_value_l444_44446

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  x * y ≥ 64 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 8*y₀ - x₀*y₀ = 0 ∧ x₀ * y₀ = 64 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l444_44446


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_neg_one_subset_iff_m_range_l444_44489

-- Define sets A and B
def A : Set ℝ := {x | x > 1}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 3}

-- Theorem for part (1)
theorem intersection_and_union_when_m_neg_one :
  (A ∩ B (-1) = {x | 1 < x ∧ x ≤ 2}) ∧
  (A ∪ B (-1) = {x | x ≥ -1}) := by sorry

-- Theorem for part (2)
theorem subset_iff_m_range :
  ∀ m : ℝ, B m ⊆ A ↔ m > 1 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_neg_one_subset_iff_m_range_l444_44489


namespace NUMINAMATH_CALUDE_parallel_line_segment_length_l444_44452

/-- Given a triangle with sides a, b, c, and lines parallel to the sides drawn through an interior point,
    if the segments of these lines within the triangle are equal in length x, then
    x = 2 / (1/a + 1/b + 1/c) -/
theorem parallel_line_segment_length (a b c x : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  x > 0 → x = 2 / (1/a + 1/b + 1/c) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_segment_length_l444_44452


namespace NUMINAMATH_CALUDE_solve_for_x_l444_44460

theorem solve_for_x (x y : ℚ) (h1 : x / y = 7 / 3) (h2 : y = 21) : x = 49 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l444_44460


namespace NUMINAMATH_CALUDE_max_value_of_a_l444_44433

def S (n : ℕ) : ℤ := -n^2 + 6*n + 7

def a (n : ℕ) : ℤ := S n - S (n-1)

theorem max_value_of_a : ∃ (M : ℤ), M = 12 ∧ ∀ (n : ℕ), n > 0 → a n ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l444_44433


namespace NUMINAMATH_CALUDE_range_of_2x_minus_y_l444_44467

theorem range_of_2x_minus_y (x y : ℝ) (hx : 2 < x ∧ x < 4) (hy : -1 < y ∧ y < 3) :
  1 < 2 * x - y ∧ 2 * x - y < 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2x_minus_y_l444_44467


namespace NUMINAMATH_CALUDE_sons_age_l444_44486

theorem sons_age (father_age : ℕ) (h1 : father_age = 38) : ℕ :=
  let son_age := 14
  let years_ago := 10
  have h2 : father_age - years_ago = 7 * (son_age - years_ago) := by sorry
  son_age

#check sons_age

end NUMINAMATH_CALUDE_sons_age_l444_44486


namespace NUMINAMATH_CALUDE_hippo_ratio_l444_44445

/-- Represents the number of female hippos -/
def F : ℕ := sorry

/-- The initial number of elephants -/
def initial_elephants : ℕ := 20

/-- The initial number of hippos -/
def initial_hippos : ℕ := 35

/-- The number of baby hippos born per female hippo -/
def babies_per_hippo : ℕ := 5

/-- The total number of animals after births -/
def total_animals : ℕ := 315

theorem hippo_ratio :
  let newborn_hippos := F * babies_per_hippo
  let newborn_elephants := newborn_hippos + 10
  let total_hippos := initial_hippos + newborn_hippos
  (F : ℚ) / total_hippos = 5 / 32 :=
by sorry

end NUMINAMATH_CALUDE_hippo_ratio_l444_44445


namespace NUMINAMATH_CALUDE_power_of_fraction_l444_44414

theorem power_of_fraction :
  (5 : ℚ) / 6 ^ 4 = 625 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_power_of_fraction_l444_44414


namespace NUMINAMATH_CALUDE_f_shifted_is_even_f_monotonicity_f_satisfies_properties_l444_44437

-- Define the function f(x) = (x-2)^2
def f (x : ℝ) : ℝ := (x - 2)^2

-- Property 1: f(x+2) is an even function
theorem f_shifted_is_even : ∀ x : ℝ, f (x + 2) = f (-x + 2) := by sorry

-- Property 2: f(x) is decreasing on (-∞, 2) and increasing on (2, +∞)
theorem f_monotonicity :
  (∀ x y : ℝ, x < y → y < 2 → f y < f x) ∧
  (∀ x y : ℝ, 2 < x → x < y → f x < f y) := by sorry

-- Theorem combining both properties
theorem f_satisfies_properties : 
  (∀ x : ℝ, f (x + 2) = f (-x + 2)) ∧
  (∀ x y : ℝ, x < y → y < 2 → f y < f x) ∧
  (∀ x y : ℝ, 2 < x → x < y → f x < f y) := by sorry

end NUMINAMATH_CALUDE_f_shifted_is_even_f_monotonicity_f_satisfies_properties_l444_44437


namespace NUMINAMATH_CALUDE_point_on_x_axis_l444_44487

theorem point_on_x_axis (m : ℝ) :
  (m + 5, 2 * m + 8) = (1, 0) ↔ (m + 5, 2 * m + 8).2 = 0 := by
sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l444_44487


namespace NUMINAMATH_CALUDE_first_class_students_l444_44420

theorem first_class_students (x : ℕ) : 
  (∃ (total_students : ℕ),
    total_students = x + 50 ∧
    (50 * x + 60 * 50 : ℚ) / total_students = 56.25) →
  x = 30 := by
sorry

end NUMINAMATH_CALUDE_first_class_students_l444_44420


namespace NUMINAMATH_CALUDE_geese_left_is_10_l444_44447

/-- The number of geese that left the duck park -/
def geese_left : ℕ := by sorry

theorem geese_left_is_10 :
  let initial_ducks : ℕ := 25
  let initial_geese : ℕ := 2 * initial_ducks - 10
  let final_ducks : ℕ := initial_ducks + 4
  let final_geese : ℕ := initial_geese - geese_left
  (final_geese = final_ducks + 1) →
  geese_left = 10 := by sorry

end NUMINAMATH_CALUDE_geese_left_is_10_l444_44447


namespace NUMINAMATH_CALUDE_yearly_savings_multiple_l444_44423

theorem yearly_savings_multiple (monthly_salary : ℝ) (h : monthly_salary > 0) :
  let monthly_spending := 0.75 * monthly_salary
  let monthly_savings := monthly_salary - monthly_spending
  let yearly_savings := 12 * monthly_savings
  yearly_savings = 4 * monthly_spending :=
by sorry

end NUMINAMATH_CALUDE_yearly_savings_multiple_l444_44423


namespace NUMINAMATH_CALUDE_stevens_height_l444_44430

-- Define the building's height and shadow length
def building_height : ℝ := 50
def building_shadow : ℝ := 25

-- Define Steven's shadow length
def steven_shadow : ℝ := 20

-- Define the theorem
theorem stevens_height :
  ∃ (h : ℝ), h = (building_height / building_shadow) * steven_shadow ∧ h = 40 :=
by sorry

end NUMINAMATH_CALUDE_stevens_height_l444_44430


namespace NUMINAMATH_CALUDE_negation_of_p_l444_44431

-- Define the proposition p
def p (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0

-- State the theorem
theorem negation_of_p (f : ℝ → ℝ) :
  ¬(p f) ↔ ∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_p_l444_44431


namespace NUMINAMATH_CALUDE_no_integer_solutions_l444_44425

theorem no_integer_solutions :
  ¬∃ (x y : ℤ), x^3 + 4*x^2 - 11*x + 30 = 8*y^3 + 24*y^2 + 18*y + 7 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l444_44425


namespace NUMINAMATH_CALUDE_parabola_focus_l444_44497

/-- A parabola is defined by the equation y = 4x^2 -/
def parabola (x y : ℝ) : Prop := y = 4 * x^2

/-- The focus of a parabola y = ax^2 is at (0, 1/(4a)) -/
def is_focus (a : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 = 0 ∧ p.2 = 1 / (4 * a)

/-- Theorem: The focus of the parabola y = 4x^2 is at (0, 1/16) -/
theorem parabola_focus :
  ∃ (f : ℝ × ℝ), (∀ x y, parabola x y → is_focus 4 f) ∧ f = (0, 1/16) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l444_44497


namespace NUMINAMATH_CALUDE_chord_equation_of_ellipse_l444_44421

/-- The equation of a line passing through a chord of an ellipse -/
theorem chord_equation_of_ellipse (x₁ y₁ x₂ y₂ : ℝ) :
  (x₁^2 / 36 + y₁^2 / 9 = 1) →  -- Point (x₁, y₁) is on the ellipse
  (x₂^2 / 36 + y₂^2 / 9 = 1) →  -- Point (x₂, y₂) is on the ellipse
  ((x₁ + x₂) / 2 = 2) →         -- Midpoint x-coordinate is 2
  ((y₁ + y₂) / 2 = 2) →         -- Midpoint y-coordinate is 2
  ∃ (x y : ℝ), x + 4*y - 10 = 0  -- Equation of the line
  := by sorry

end NUMINAMATH_CALUDE_chord_equation_of_ellipse_l444_44421


namespace NUMINAMATH_CALUDE_video_game_expenditure_l444_44476

/-- The cost of the basketball game -/
def basketball_cost : ℚ := 5.20

/-- The cost of the racing game -/
def racing_cost : ℚ := 4.23

/-- The total cost of video games -/
def total_cost : ℚ := basketball_cost + racing_cost

theorem video_game_expenditure : total_cost = 9.43 := by
  sorry

end NUMINAMATH_CALUDE_video_game_expenditure_l444_44476
