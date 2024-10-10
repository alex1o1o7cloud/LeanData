import Mathlib

namespace cube_opposite_face_l2467_246700

-- Define a cube type
structure Cube :=
  (faces : Fin 6 → Char)

-- Define adjacency relation
def adjacent (c : Cube) (x y : Char) : Prop :=
  ∃ (i j : Fin 6), i ≠ j ∧ c.faces i = x ∧ c.faces j = y

-- Define opposite relation
def opposite (c : Cube) (x y : Char) : Prop :=
  ∃ (i j : Fin 6), i ≠ j ∧ c.faces i = x ∧ c.faces j = y ∧
  ∀ (k : Fin 6), k ≠ i → k ≠ j → ¬(adjacent c (c.faces i) (c.faces k) ∧ adjacent c (c.faces j) (c.faces k))

theorem cube_opposite_face (c : Cube) :
  (c.faces = λ i => ['А', 'Б', 'В', 'Г', 'Д', 'Е'][i]) →
  (adjacent c 'А' 'Б') →
  (adjacent c 'А' 'Г') →
  (adjacent c 'Г' 'Д') →
  (adjacent c 'Г' 'Е') →
  (adjacent c 'В' 'Д') →
  (adjacent c 'В' 'Б') →
  opposite c 'Д' 'Б' :=
by sorry

end cube_opposite_face_l2467_246700


namespace injective_function_characterization_l2467_246770

theorem injective_function_characterization (f : ℤ → ℤ) :
  Function.Injective f ∧ (∀ x y : ℤ, |f x - f y| ≤ |x - y|) →
  ∃ a : ℤ, (∀ x : ℤ, f x = a + x) ∨ (∀ x : ℤ, f x = a - x) :=
by sorry

end injective_function_characterization_l2467_246770


namespace log_sqrt7_343sqrt7_equals_7_l2467_246769

theorem log_sqrt7_343sqrt7_equals_7 :
  Real.log (343 * Real.sqrt 7) / Real.log (Real.sqrt 7) = 7 := by
  sorry

end log_sqrt7_343sqrt7_equals_7_l2467_246769


namespace arithmetic_sequence_ratio_l2467_246737

/-- Two arithmetic sequences a and b with their respective sums S and T -/
structure ArithmeticSequences where
  a : ℕ → ℚ
  b : ℕ → ℚ
  S : ℕ → ℚ
  T : ℕ → ℚ

/-- The ratio of sums S_n and T_n for any n -/
def sum_ratio (seq : ArithmeticSequences) : ℕ → ℚ :=
  fun n => seq.S n / seq.T n

/-- The given condition that S_n / T_n = (2n + 1) / (3n + 2) -/
def sum_ratio_condition (seq : ArithmeticSequences) : Prop :=
  ∀ n : ℕ, sum_ratio seq n = (2 * n + 1) / (3 * n + 2)

/-- The theorem to be proved -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequences) 
  (h : sum_ratio_condition seq) : 
  (seq.a 3 + seq.a 11 + seq.a 19) / (seq.b 7 + seq.b 15) = 129 / 130 := by
  sorry

end arithmetic_sequence_ratio_l2467_246737


namespace point_coordinates_l2467_246720

/-- A point in the second quadrant with specific x and y values -/
structure SecondQuadrantPoint where
  x : ℝ
  y : ℝ
  second_quadrant : x < 0 ∧ y > 0
  x_abs : |x| = 2
  y_squared : y^2 = 1

/-- The coordinates of the point P are (-2, 1) -/
theorem point_coordinates (P : SecondQuadrantPoint) : P.x = -2 ∧ P.y = 1 := by
  sorry

end point_coordinates_l2467_246720


namespace equilateral_triangle_area_perimeter_ratio_l2467_246752

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 10
  let perimeter : ℝ := 3 * side_length
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  area / perimeter = (5 * Real.sqrt 3) / 6 := by sorry

end equilateral_triangle_area_perimeter_ratio_l2467_246752


namespace number_sequence_count_l2467_246722

theorem number_sequence_count : ∀ (N : ℕ) (S : ℝ),
  S / N = 44 →
  (11 * 48 + 11 * 41 - 55) / N = 44 →
  N = 21 := by
sorry

end number_sequence_count_l2467_246722


namespace liz_total_spent_l2467_246758

/-- The total amount spent by Liz on her baking purchases -/
def total_spent (recipe_book_cost : ℕ) (ingredient_cost : ℕ) (num_ingredients : ℕ) : ℕ :=
  let baking_dish_cost := 2 * recipe_book_cost
  let ingredients_total_cost := ingredient_cost * num_ingredients
  let apron_cost := recipe_book_cost + 1
  recipe_book_cost + baking_dish_cost + ingredients_total_cost + apron_cost

/-- Theorem stating that Liz spent $40 in total -/
theorem liz_total_spent : total_spent 6 3 5 = 40 := by
  sorry

end liz_total_spent_l2467_246758


namespace hyperbola_equation_l2467_246755

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * Real.sqrt 5 = 2 * Real.sqrt (a^2 + b^2)) →
  (b / a = 1 / 2) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 4 - y^2 = 1) :=
by sorry

end hyperbola_equation_l2467_246755


namespace intersection_count_theorem_m_value_theorem_l2467_246710

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 2

-- Define the line l
def l (x y : ℝ) : Prop := y = x ∧ x ≥ 0

-- Define the number of intersection points
def intersection_count : ℕ := 1

-- Define the equation for C₂ when θ = π/4
def C₂_equation (ρ m : ℝ) : Prop := ρ^2 - 3 * Real.sqrt 2 * ρ + 2 * m = 0

-- Theorem for the number of intersection points
theorem intersection_count_theorem :
  ∃! (x y : ℝ), C₁ x y ∧ l x y :=
sorry

-- Theorem for the value of m
theorem m_value_theorem (ρ₁ ρ₂ m : ℝ) :
  C₂_equation ρ₁ m ∧ C₂_equation ρ₂ m ∧ ρ₂ = 2 * ρ₁ → m = 2 :=
sorry

end intersection_count_theorem_m_value_theorem_l2467_246710


namespace finitely_many_odd_divisors_l2467_246727

theorem finitely_many_odd_divisors (k : ℕ+) :
  (∃ c : ℕ, k + 1 = 2^c) ↔
  (∃ S : Finset ℕ, ∀ n : ℕ, n % 2 = 1 → (n ∣ k^n + 1) → n ∈ S) :=
by sorry

end finitely_many_odd_divisors_l2467_246727


namespace liquor_and_beer_cost_l2467_246708

/-- The price of one bottle of beer in yuan -/
def beer_price : ℚ := 2

/-- The price of one bottle of liquor in yuan -/
def liquor_price : ℚ := 16

/-- The total cost of 2 bottles of liquor and 12 bottles of beer in yuan -/
def total_cost : ℚ := 56

/-- The number of bottles of beer equivalent in price to one bottle of liquor -/
def liquor_to_beer_ratio : ℕ := 8

theorem liquor_and_beer_cost :
  (2 * liquor_price + 12 * beer_price = total_cost) →
  (liquor_price = liquor_to_beer_ratio * beer_price) →
  (liquor_price + beer_price = 18) := by
    sorry

end liquor_and_beer_cost_l2467_246708


namespace equidistant_point_existence_l2467_246706

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane represented by y = mx + b -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Distance between a point and a circle -/
def distanceToCircle (p : Point) (c : Circle) : ℝ :=
  sorry

/-- Distance between a point and a line -/
def distanceToLine (p : Point) (l : Line) : ℝ :=
  sorry

/-- The main theorem -/
theorem equidistant_point_existence (c : Circle) (upper_tangent lower_tangent : Line) :
  c.radius = 5 →
  distanceToLine (0, c.center.2 + c.radius) upper_tangent = 3 →
  distanceToLine (0, c.center.2 - c.radius) lower_tangent = 7 →
  ∃! p : Point, 
    distanceToCircle p c = distanceToLine p upper_tangent ∧ 
    distanceToCircle p c = distanceToLine p lower_tangent :=
  sorry

end equidistant_point_existence_l2467_246706


namespace james_bed_purchase_l2467_246730

theorem james_bed_purchase (bed_frame_price : ℝ) (discount_rate : ℝ) : 
  bed_frame_price = 75 →
  discount_rate = 0.2 →
  let bed_price := 10 * bed_frame_price
  let total_before_discount := bed_frame_price + bed_price
  let discount_amount := discount_rate * total_before_discount
  let final_price := total_before_discount - discount_amount
  final_price = 660 := by sorry

end james_bed_purchase_l2467_246730


namespace divisible_by_45_digits_l2467_246725

theorem divisible_by_45_digits (a b : Nat) : 
  a < 10 → b < 10 → (72000 + 100 * a + 30 + b) % 45 = 0 → 
  ((a = 6 ∧ b = 0) ∨ (a = 1 ∧ b = 5)) := by
sorry

end divisible_by_45_digits_l2467_246725


namespace intersection_unique_l2467_246788

/-- Two lines in 2D space -/
def line1 (t : ℝ) : ℝ × ℝ := (4 + 3 * t, 1 - 2 * t)
def line2 (u : ℝ) : ℝ × ℝ := (-2 + 4 * u, 5 - u)

/-- The point of intersection -/
def intersection_point : ℝ × ℝ := (-2, 5)

/-- Theorem stating that the intersection_point is the unique point of intersection for the two lines -/
theorem intersection_unique :
  (∃ (t : ℝ), line1 t = intersection_point) ∧
  (∃ (u : ℝ), line2 u = intersection_point) ∧
  (∀ (p : ℝ × ℝ), (∃ (t : ℝ), line1 t = p) ∧ (∃ (u : ℝ), line2 u = p) → p = intersection_point) :=
by sorry

end intersection_unique_l2467_246788


namespace min_sum_box_dimensions_l2467_246702

theorem min_sum_box_dimensions (a b c : ℕ+) : 
  a * b * c = 3003 → 
  ∀ x y z : ℕ+, x * y * z = 3003 → a + b + c ≤ x + y + z → 
  a + b + c = 45 :=
sorry

end min_sum_box_dimensions_l2467_246702


namespace certain_number_problem_l2467_246765

theorem certain_number_problem : ∃ x : ℝ, 45 * x = 0.45 * 900 ∧ x = 9 := by sorry

end certain_number_problem_l2467_246765


namespace reflection_of_line_over_x_axis_l2467_246714

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Reflects a line over the x-axis --/
def reflect_over_x_axis (l : Line) : Line :=
  { slope := -l.slope, intercept := -l.intercept }

theorem reflection_of_line_over_x_axis :
  let original_line : Line := { slope := 2, intercept := 3 }
  let reflected_line : Line := reflect_over_x_axis original_line
  reflected_line = { slope := -2, intercept := -3 } := by
  sorry

end reflection_of_line_over_x_axis_l2467_246714


namespace damaged_polynomial_satisfies_equation_damaged_polynomial_value_l2467_246762

-- Define the damaged polynomial
def damaged_polynomial (x y : ℚ) : ℚ := -3 * x + y^2

-- Define the given equation
def equation_holds (x y : ℚ) : Prop :=
  damaged_polynomial x y + 2 * (x - 1/3 * y^2) = -x + 1/3 * y^2

-- Theorem 1: The damaged polynomial satisfies the equation
theorem damaged_polynomial_satisfies_equation :
  ∀ x y : ℚ, equation_holds x y :=
sorry

-- Theorem 2: The value of the damaged polynomial for given x and y
theorem damaged_polynomial_value :
  damaged_polynomial (-3) (3/2) = 45/4 :=
sorry

end damaged_polynomial_satisfies_equation_damaged_polynomial_value_l2467_246762


namespace square_area_increase_l2467_246749

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  let new_side := 1.25 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.5625 := by
sorry

end square_area_increase_l2467_246749


namespace factorial_500_trailing_zeroes_l2467_246719

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 500! has 124 trailing zeroes -/
theorem factorial_500_trailing_zeroes :
  trailingZeroes 500 = 124 := by
  sorry

end factorial_500_trailing_zeroes_l2467_246719


namespace ship_speed_upstream_l2467_246715

/-- Given a ship traveling downstream at 26 km/h and a water flow speed of v km/h,
    the speed of the ship traveling upstream is 26 - 2v km/h. -/
theorem ship_speed_upstream 
  (v : ℝ) -- Water flow speed in km/h
  (h1 : v > 0) -- Assumption that water flow speed is positive
  (h2 : v < 26) -- Assumption that water flow speed is less than downstream speed
  : ℝ :=
  26 - 2 * v

#check ship_speed_upstream

end ship_speed_upstream_l2467_246715


namespace min_sum_of_product_36_l2467_246732

theorem min_sum_of_product_36 (a b : ℤ) (h : a * b = 36) : 
  ∀ (x y : ℤ), x * y = 36 → a + b ≤ x + y ∧ ∃ (a₀ b₀ : ℤ), a₀ * b₀ = 36 ∧ a₀ + b₀ = -37 :=
by sorry

end min_sum_of_product_36_l2467_246732


namespace expression_value_l2467_246772

theorem expression_value (x y : ℝ) (h1 : x + y = 17) (h2 : x * y = 17) :
  (x^2 - 17*x) * (y + 17/y) = -289 := by
  sorry

end expression_value_l2467_246772


namespace product_in_fourth_quadrant_l2467_246766

def z₁ : ℂ := 3 + Complex.I
def z₂ : ℂ := 1 - Complex.I

theorem product_in_fourth_quadrant :
  let z := z₁ * z₂
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end product_in_fourth_quadrant_l2467_246766


namespace detached_calculations_l2467_246712

theorem detached_calculations : 
  (78 * 12 - 531 = 405) ∧ 
  (32 * (69 - 54) = 480) ∧ 
  (58 / 2 * 16 = 464) ∧ 
  (352 / 8 / 4 = 11) := by
  sorry

end detached_calculations_l2467_246712


namespace parabola_distance_theorem_l2467_246711

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem parabola_distance_theorem (A : ℝ × ℝ) :
  parabola A.1 A.2 →  -- A lies on the parabola
  (A.1 - focus.1)^2 + (A.2 - focus.2)^2 = (B.1 - focus.1)^2 + (B.2 - focus.2)^2 →  -- |AF| = |BF|
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8 :=  -- |AB| = 2√2
by sorry

end parabola_distance_theorem_l2467_246711


namespace prime_power_equation_solutions_l2467_246739

theorem prime_power_equation_solutions :
  ∀ p n : ℕ,
    Nat.Prime p →
    n > 0 →
    p^3 - 2*p^2 + p + 1 = 3^n →
    ((p = 2 ∧ n = 1) ∨ (p = 5 ∧ n = 4)) :=
by sorry

end prime_power_equation_solutions_l2467_246739


namespace cube_sum_not_2016_l2467_246748

theorem cube_sum_not_2016 (a b : ℤ) : a^3 + 5*b^3 ≠ 2016 := by
  sorry

end cube_sum_not_2016_l2467_246748


namespace sock_count_proof_l2467_246736

def total_socks (john_initial mary_initial kate_initial : ℕ)
                (john_thrown john_bought : ℕ)
                (mary_thrown mary_bought : ℕ)
                (kate_thrown kate_bought : ℕ) : ℕ :=
  (john_initial - john_thrown + john_bought) +
  (mary_initial - mary_thrown + mary_bought) +
  (kate_initial - kate_thrown + kate_bought)

theorem sock_count_proof :
  total_socks 33 20 15 19 13 6 10 5 8 = 69 := by
  sorry

end sock_count_proof_l2467_246736


namespace parallel_vectors_x_value_l2467_246763

-- Define the vectors
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

-- Theorem statement
theorem parallel_vectors_x_value :
  parallel a (b x) → x = 6 := by
  sorry

end parallel_vectors_x_value_l2467_246763


namespace fixed_circle_theorem_l2467_246746

noncomputable section

-- Define the hyperbola C
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / (3 * a^2) = 1

-- Define the foci F₁ and F₂
def F₁ (a : ℝ) : ℝ × ℝ := (-2, 0)
def F₂ (a : ℝ) : ℝ × ℝ := (2, 0)

-- Define the distance from F₂ to the asymptote
def distance_to_asymptote (a : ℝ) : ℝ := Real.sqrt 3

-- Define a line passing through the left vertex and not coinciding with x-axis
def line_through_left_vertex (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

-- Define the intersection point B
def point_B (a k : ℝ) : ℝ × ℝ := ((3 + k^2) / (3 - k^2), 6 * k / (3 - k^2))

-- Define the intersection point P
def point_P (k : ℝ) : ℝ × ℝ := (1/2, k * 3/2)

-- Define the line parallel to PF₂ passing through F₁
def parallel_line (k : ℝ) (x : ℝ) : ℝ := -k * (x + 2)

-- Define the theorem
theorem fixed_circle_theorem (a : ℝ) (k : ℝ) :
  a > 0 →
  ∀ Q : ℝ × ℝ,
  (∃ x, Q.1 = x ∧ Q.2 = line_through_left_vertex k x) →
  (∃ x, Q.1 = x ∧ Q.2 = parallel_line k x) →
  (Q.1 - (F₂ a).1)^2 + (Q.2 - (F₂ a).2)^2 = 16 :=
by sorry

end

end fixed_circle_theorem_l2467_246746


namespace smallest_fruit_distribution_l2467_246789

theorem smallest_fruit_distribution (N : ℕ) : N = 79 ↔ 
  N > 0 ∧
  (N - 1) % 3 = 0 ∧
  (2 * (N - 1) / 3 - 1) % 3 = 0 ∧
  ((2 * N - 5) / 3 - 1) % 3 = 0 ∧
  ((4 * N - 28) / 9 - 1) % 3 = 0 ∧
  ((8 * N - 56) / 27 - 1) % 3 = 0 ∧
  ∀ (M : ℕ), M < N → 
    (M > 0 ∧
    (M - 1) % 3 = 0 ∧
    (2 * (M - 1) / 3 - 1) % 3 = 0 ∧
    ((2 * M - 5) / 3 - 1) % 3 = 0 ∧
    ((4 * M - 28) / 9 - 1) % 3 = 0 ∧
    ((8 * M - 56) / 27 - 1) % 3 = 0) → False :=
by sorry

end smallest_fruit_distribution_l2467_246789


namespace upper_limit_correct_l2467_246798

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def upper_limit : ℕ := 7533

theorem upper_limit_correct :
  ∀ h : ℕ, h > 0 ∧ digit_product h = 210 → h < upper_limit :=
by sorry

end upper_limit_correct_l2467_246798


namespace min_sum_squared_distances_l2467_246741

open Real
open InnerProductSpace

theorem min_sum_squared_distances (a b c : EuclideanSpace ℝ (Fin 2)) 
  (ha : ‖a‖^2 = 4)
  (hb : ‖b‖^2 = 1)
  (hc : ‖c‖^2 = 9) :
  ∃ (min : ℝ), min = 2 ∧ 
    ∀ (x y z : EuclideanSpace ℝ (Fin 2)), 
      ‖x‖^2 = 4 → ‖y‖^2 = 1 → ‖z‖^2 = 9 →
      ‖x - y‖^2 + ‖x - z‖^2 + ‖y - z‖^2 ≥ min :=
by sorry

end min_sum_squared_distances_l2467_246741


namespace titu_andreescu_inequality_l2467_246750

theorem titu_andreescu_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 := by
  sorry

end titu_andreescu_inequality_l2467_246750


namespace triangle_ratio_l2467_246790

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  A = Real.pi / 3 →
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
sorry


end triangle_ratio_l2467_246790


namespace max_trig_fraction_l2467_246705

theorem max_trig_fraction (x : ℝ) : 
  (Real.sin x)^4 + (Real.cos x)^4 ≤ (Real.sin x)^2 + (Real.cos x)^2 + 2*(Real.sin x)^2*(Real.cos x)^2 := by
  sorry

#check max_trig_fraction

end max_trig_fraction_l2467_246705


namespace unique_sequence_existence_l2467_246717

theorem unique_sequence_existence :
  ∃! a : ℕ → ℝ,
    (∀ n, a n > 0) ∧
    a 0 = 1 ∧
    (∀ n : ℕ, a (n + 1) = a (n - 1) - a n) :=
by sorry

end unique_sequence_existence_l2467_246717


namespace lexie_picked_12_apples_l2467_246713

/-- The number of apples Lexie and Tom picked together -/
def total_apples : ℕ := 36

/-- Lexie's apples -/
def lexie_apples : ℕ := 12

/-- Tom's apples -/
def tom_apples : ℕ := 2 * lexie_apples

/-- Theorem stating that Lexie picked 12 apples given the conditions -/
theorem lexie_picked_12_apples : 
  (tom_apples = 2 * lexie_apples) ∧ (lexie_apples + tom_apples = total_apples) → 
  lexie_apples = 12 := by
  sorry

#check lexie_picked_12_apples

end lexie_picked_12_apples_l2467_246713


namespace f_less_than_three_zeros_l2467_246735

/-- The cubic function f(x) = x³ - ax² + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

/-- Theorem: f(x) has less than 3 zeros if and only if a ≤ 3 -/
theorem f_less_than_three_zeros (a : ℝ) :
  (∀ x : ℝ, f a x = 0 → (∃ y z : ℝ, y ≠ z ∧ f a y = 0 ∧ f a z = 0 → False)) ↔ a ≤ 3 :=
sorry

end f_less_than_three_zeros_l2467_246735


namespace green_ball_probability_l2467_246731

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a container -/
def containerProb : ℚ := 1 / 3

/-- Calculate the probability of drawing a green ball from a container -/
def greenProb (c : Container) : ℚ := c.green / (c.red + c.green)

/-- The three containers A, B, and C -/
def containerA : Container := ⟨5, 5⟩
def containerB : Container := ⟨8, 2⟩
def containerC : Container := ⟨3, 7⟩

/-- The probability of selecting a green ball -/
def probGreenBall : ℚ :=
  containerProb * greenProb containerA +
  containerProb * greenProb containerB +
  containerProb * greenProb containerC

theorem green_ball_probability :
  probGreenBall = 7 / 15 := by
  sorry

end green_ball_probability_l2467_246731


namespace solution_fraction_proof_l2467_246723

def initial_amount : ℚ := 2

def first_day_usage (amount : ℚ) : ℚ := (1 / 4) * amount

def second_day_usage (amount : ℚ) : ℚ := (1 / 2) * amount

def remaining_after_two_days (initial : ℚ) : ℚ :=
  initial - first_day_usage initial - second_day_usage (initial - first_day_usage initial)

theorem solution_fraction_proof :
  remaining_after_two_days initial_amount / initial_amount = 3 / 8 := by
  sorry

end solution_fraction_proof_l2467_246723


namespace quadrilateral_sum_l2467_246740

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a quadrilateral PQRS -/
structure Quadrilateral where
  P : Point
  Q : Point
  R : Point
  S : Point

def area (q : Quadrilateral) : ℚ :=
  sorry  -- Area calculation implementation

theorem quadrilateral_sum (a b : ℤ) :
  a > b ∧ b > 0 →
  let q := Quadrilateral.mk
    (Point.mk (2*a) (2*b))
    (Point.mk (2*b) (2*a))
    (Point.mk (-2*a) (-2*b))
    (Point.mk (-2*b) (-2*a))
  area q = 32 →
  a + b = 4 := by
    sorry

end quadrilateral_sum_l2467_246740


namespace valid_numbers_l2467_246743

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  (n / 10000 % 10) * 5 = (n / 1000 % 10) ∧
  (n / 10000 % 10) * (n / 1000 % 10) * (n / 100 % 10) * (n / 10 % 10) * (n % 10) = 1000

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n = 15558 ∨ n = 15585 ∨ n = 15855 :=
by sorry

end valid_numbers_l2467_246743


namespace square_sum_given_difference_and_product_l2467_246751

theorem square_sum_given_difference_and_product (a b : ℝ) 
  (h1 : a - b = 2) 
  (h2 : a * b = 10.5) : 
  a^2 + b^2 = 25 := by
sorry

end square_sum_given_difference_and_product_l2467_246751


namespace quadratic_roots_coefficients_l2467_246759

theorem quadratic_roots_coefficients (p q : ℝ) :
  (∀ x, x^2 + p*x + q = 0 ↔ x = p ∨ x = q) →
  ((p = 0 ∧ q = 0) ∨ (p = 1 ∧ q = -2)) := by
sorry

end quadratic_roots_coefficients_l2467_246759


namespace sum_of_self_opposite_and_self_reciprocal_l2467_246787

theorem sum_of_self_opposite_and_self_reciprocal (a b : ℝ) : 
  ((-a) = a) → ((1 / b) = b) → (a + b = 1 ∨ a + b = -1) := by
  sorry

end sum_of_self_opposite_and_self_reciprocal_l2467_246787


namespace square_even_implies_even_sqrt_2_irrational_l2467_246744

-- Part 1: If p² is even, then p is even
theorem square_even_implies_even (p : ℤ) : Even (p^2) → Even p := by sorry

-- Part 2: √2 is irrational
theorem sqrt_2_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a : ℚ) / b = Real.sqrt 2 := by sorry

end square_even_implies_even_sqrt_2_irrational_l2467_246744


namespace contrapositive_x_squared_greater_than_one_l2467_246771

theorem contrapositive_x_squared_greater_than_one (x : ℝ) : 
  x ≤ 1 → x^2 ≤ 1 := by sorry

end contrapositive_x_squared_greater_than_one_l2467_246771


namespace correct_total_paths_l2467_246795

/-- The number of paths from Wolfburg to the Green Meadows -/
def paths_wolfburg_to_meadows : ℕ := 6

/-- The number of paths from the Green Meadows to Sheep Village -/
def paths_meadows_to_village : ℕ := 20

/-- Wolfburg and Sheep Village are separated by the Green Meadows -/
axiom separated_by_meadows : True

/-- The number of different ways to travel from Wolfburg to Sheep Village -/
def total_paths : ℕ := paths_wolfburg_to_meadows * paths_meadows_to_village

theorem correct_total_paths : total_paths = 120 := by sorry

end correct_total_paths_l2467_246795


namespace perpendicular_vectors_l2467_246773

/-- Given vectors a and b, if a is perpendicular to (t*a + b), then t = -1 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (t : ℝ) :
  a = (1, -1) →
  b = (6, -4) →
  (a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) →
  t = -1 := by
  sorry

end perpendicular_vectors_l2467_246773


namespace sum_of_squares_l2467_246793

theorem sum_of_squares (a b c : ℝ) : 
  a + b + c = 23 → 
  a * b + b * c + a * c = 131 → 
  a^2 + b^2 + c^2 = 267 := by
sorry

end sum_of_squares_l2467_246793


namespace hot_dogs_dinner_l2467_246760

def hot_dogs_today : ℕ := 11
def hot_dogs_lunch : ℕ := 9

theorem hot_dogs_dinner : hot_dogs_today - hot_dogs_lunch = 2 := by
  sorry

end hot_dogs_dinner_l2467_246760


namespace geometric_subsequence_exists_l2467_246721

/-- An arithmetic progression with first term 1 -/
def ArithmeticProgression (d : ℕ) : ℕ → ℕ :=
  fun n => 1 + (n - 1) * d

/-- A geometric progression -/
def GeometricProgression (a : ℕ) : ℕ → ℕ :=
  fun k => a^k

theorem geometric_subsequence_exists :
  ∃ (d a : ℕ), ∃ (start : ℕ),
    (∀ k, k ∈ Finset.range 2015 →
      ArithmeticProgression d (start + k) = GeometricProgression a (k + 1)) :=
sorry

end geometric_subsequence_exists_l2467_246721


namespace average_matches_is_four_l2467_246796

/-- Represents the distribution of matches played in a badminton club --/
structure MatchDistribution :=
  (one_match : Nat)
  (two_matches : Nat)
  (four_matches : Nat)
  (six_matches : Nat)
  (eight_matches : Nat)

/-- Calculates the average number of matches played, rounded to the nearest whole number --/
def averageMatchesPlayed (d : MatchDistribution) : Nat :=
  let totalMatches := d.one_match * 1 + d.two_matches * 2 + d.four_matches * 4 + d.six_matches * 6 + d.eight_matches * 8
  let totalPlayers := d.one_match + d.two_matches + d.four_matches + d.six_matches + d.eight_matches
  let average := totalMatches / totalPlayers
  if totalMatches % totalPlayers >= totalPlayers / 2 then average + 1 else average

/-- The specific distribution of matches in the badminton club --/
def clubDistribution : MatchDistribution :=
  { one_match := 4
  , two_matches := 3
  , four_matches := 2
  , six_matches := 2
  , eight_matches := 8 }

theorem average_matches_is_four :
  averageMatchesPlayed clubDistribution = 4 := by sorry

end average_matches_is_four_l2467_246796


namespace product_equality_l2467_246799

theorem product_equality : 500 * 2019 * 0.02019 * 5 = 0.25 * 2019^2 := by sorry

end product_equality_l2467_246799


namespace f_five_eq_zero_l2467_246753

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_periodic : ∀ x, f (x + 2) = f x

-- State the theorem
theorem f_five_eq_zero : f 5 = 0 := by sorry

end f_five_eq_zero_l2467_246753


namespace one_third_to_fifth_power_l2467_246734

theorem one_third_to_fifth_power :
  (1 / 3 : ℚ) ^ 5 = 1 / 243 := by sorry

end one_third_to_fifth_power_l2467_246734


namespace product_trailing_zeros_l2467_246780

def trailing_zeros (n : ℕ) : ℕ := sorry

theorem product_trailing_zeros : 
  let a : ℕ := 35
  let b : ℕ := 4900
  let a_factorization := 5 * 7
  let b_factorization := 2^2 * 5^2 * 7^2
  trailing_zeros (a * b) = 2 := by sorry

end product_trailing_zeros_l2467_246780


namespace spatial_relationships_l2467_246767

-- Define the basic concepts
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships
def intersect (l1 l2 : Line) : Prop := sorry
def skew (l1 l2 : Line) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def line_of_intersection (p1 p2 : Plane) : Line := sorry

-- State the propositions
def proposition_1 (l1 l2 l3 l4 : Line) : Prop :=
  intersect l1 l3 → intersect l2 l4 → skew l3 l4 → skew l1 l2

def proposition_2 (l1 l2 : Line) (p1 p2 : Plane) : Prop :=
  parallel_planes p1 p2 → parallel_lines l1 p1 → parallel_lines l2 p2 → parallel_lines l1 l2

def proposition_3 (l1 l2 : Line) (p : Plane) : Prop :=
  perpendicular_to_plane l1 p → perpendicular_to_plane l2 p → parallel_lines l1 l2

def proposition_4 (p1 p2 : Plane) (l : Line) : Prop :=
  perpendicular_planes p1 p2 →
  line_in_plane l p1 →
  ¬perpendicular_to_plane l (line_of_intersection p1 p2) →
  ¬perpendicular_to_plane l p2

theorem spatial_relationships :
  (∀ l1 l2 l3 l4 : Line, ¬proposition_1 l1 l2 l3 l4) ∧
  (∀ l1 l2 : Line, ∀ p1 p2 : Plane, ¬proposition_2 l1 l2 p1 p2) ∧
  (∀ l1 l2 : Line, ∀ p : Plane, proposition_3 l1 l2 p) ∧
  (∀ p1 p2 : Plane, ∀ l : Line, proposition_4 p1 p2 l) :=
sorry

end spatial_relationships_l2467_246767


namespace decreasing_f_implies_a_range_l2467_246701

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - a*x + 5 else a/x

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

theorem decreasing_f_implies_a_range (a : ℝ) :
  is_decreasing (f a) → 2 ≤ a ∧ a ≤ 3 :=
by sorry

end decreasing_f_implies_a_range_l2467_246701


namespace ceiling_floor_sum_l2467_246779

theorem ceiling_floor_sum : ⌈(7 : ℚ) / 3⌉ + ⌊-(7 : ℚ) / 3⌋ = 0 := by
  sorry

end ceiling_floor_sum_l2467_246779


namespace greatest_valid_number_l2467_246757

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ 
  ∃ k : ℕ, n = 9 * k + 2 ∧
  ∃ m : ℕ, n = 5 * m + 3

theorem greatest_valid_number : 
  is_valid_number 9962 ∧ ∀ n : ℕ, is_valid_number n → n ≤ 9962 :=
sorry

end greatest_valid_number_l2467_246757


namespace lower_bound_second_inequality_l2467_246764

theorem lower_bound_second_inequality (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : ∃ n, n < x ∧ x < 18)
  (h3 : -2 < x ∧ x < 9)
  (h4 : 0 < x ∧ x < 8)
  (h5 : x + 1 < 9) :
  ∀ n, n < x → n ≤ 3 :=
by sorry

end lower_bound_second_inequality_l2467_246764


namespace pricing_scenario_l2467_246776

/-- The number of articles in a pricing scenario -/
def num_articles : ℕ := 50

/-- The number of articles used for selling price comparison -/
def comparison_articles : ℕ := 45

/-- The gain percentage as a rational number -/
def gain_percentage : ℚ := 1 / 9

theorem pricing_scenario :
  (∀ (cost_price selling_price : ℚ),
    cost_price * num_articles = selling_price * comparison_articles →
    selling_price = cost_price * (1 + gain_percentage)) →
  num_articles = 50 :=
sorry

end pricing_scenario_l2467_246776


namespace mowgli_nuts_theorem_l2467_246783

/-- The number of monkeys --/
def num_monkeys : ℕ := 5

/-- The number of nuts each monkey gathered initially --/
def nuts_per_monkey : ℕ := 8

/-- The number of nuts thrown by each monkey during the quarrel --/
def nuts_thrown_per_monkey : ℕ := num_monkeys - 1

/-- The total number of nuts thrown during the quarrel --/
def total_nuts_thrown : ℕ := num_monkeys * nuts_thrown_per_monkey

/-- The number of nuts Mowgli received --/
def nuts_received : ℕ := (num_monkeys * nuts_per_monkey) / 2

theorem mowgli_nuts_theorem :
  nuts_received = total_nuts_thrown :=
by sorry

end mowgli_nuts_theorem_l2467_246783


namespace solution_system_l2467_246738

theorem solution_system (x y : ℝ) 
  (eq1 : ⌊x⌋ + (y - ⌊y⌋) = 7.2)
  (eq2 : (x - ⌊x⌋) + ⌊y⌋ = 10.3) : 
  |x - y| = 2.9 := by
  sorry

end solution_system_l2467_246738


namespace triangle_solution_l2467_246778

/-- Given a triangle ABC with side lengths a, b, c and angles α, β, γ,
    if a : b = 1 : 2, α : β = 1 : 3, and c = 5 cm,
    then a = 5√3/3 cm, b = 10√3/3 cm, α = 30°, β = 90°, and γ = 60°. -/
theorem triangle_solution (a b c : ℝ) (α β γ : ℝ) : 
  a / b = 1 / 2 →
  α / β = 1 / 3 →
  c = 5 →
  a = 5 * Real.sqrt 3 / 3 ∧
  b = 10 * Real.sqrt 3 / 3 ∧
  α = Real.pi / 6 ∧
  β = Real.pi / 2 ∧
  γ = Real.pi / 3 := by
sorry

end triangle_solution_l2467_246778


namespace min_k_inequality_k_lower_bound_l2467_246784

theorem min_k_inequality (x y z : ℝ) :
  (16/9 : ℝ) * (x^2 - x + 1) * (y^2 - y + 1) * (z^2 - z + 1) ≥ (x*y*z)^2 - x*y*z + 1 :=
by sorry

theorem k_lower_bound (k : ℝ) 
  (h : ∀ x y z : ℝ, k * (x^2 - x + 1) * (y^2 - y + 1) * (z^2 - z + 1) ≥ (x*y*z)^2 - x*y*z + 1) :
  k ≥ 16/9 :=
by sorry

end min_k_inequality_k_lower_bound_l2467_246784


namespace combined_weight_equals_3655_574_l2467_246791

-- Define molar masses of elements
def mass_C : ℝ := 12.01
def mass_H : ℝ := 1.008
def mass_O : ℝ := 16.00
def mass_Na : ℝ := 22.99

-- Define molar masses of compounds
def mass_citric_acid : ℝ := 6 * mass_C + 8 * mass_H + 7 * mass_O
def mass_sodium_carbonate : ℝ := 2 * mass_Na + mass_C + 3 * mass_O
def mass_sodium_citrate : ℝ := 3 * mass_Na + 6 * mass_C + 5 * mass_H + 7 * mass_O
def mass_carbon_dioxide : ℝ := mass_C + 2 * mass_O
def mass_water : ℝ := 2 * mass_H + mass_O

-- Define number of moles for each substance
def moles_citric_acid : ℝ := 3
def moles_sodium_carbonate : ℝ := 4.5
def moles_sodium_citrate : ℝ := 9
def moles_carbon_dioxide : ℝ := 4.5
def moles_water : ℝ := 4.5

-- Theorem statement
theorem combined_weight_equals_3655_574 :
  moles_citric_acid * mass_citric_acid +
  moles_sodium_carbonate * mass_sodium_carbonate +
  moles_sodium_citrate * mass_sodium_citrate +
  moles_carbon_dioxide * mass_carbon_dioxide +
  moles_water * mass_water = 3655.574 := by
  sorry

end combined_weight_equals_3655_574_l2467_246791


namespace f_sin_75_eq_zero_l2467_246777

-- Define the function f
def f (a₄ a₃ a₂ a₁ a₀ : ℤ) (x : ℝ) : ℝ :=
  a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

-- State the theorem
theorem f_sin_75_eq_zero 
  (a₄ a₃ a₂ a₁ a₀ : ℤ) 
  (h₁ : f a₄ a₃ a₂ a₁ a₀ (Real.cos (75 * π / 180)) = 0) 
  (h₂ : a₄ ≠ 0) : 
  f a₄ a₃ a₂ a₁ a₀ (Real.sin (75 * π / 180)) = 0 := by
  sorry

end f_sin_75_eq_zero_l2467_246777


namespace tangent_line_equation_l2467_246742

/-- The curve S defined by y = x³ + 4 -/
def S : ℝ → ℝ := fun x ↦ x^3 + 4

/-- The point A -/
def A : ℝ × ℝ := (1, 5)

/-- The first possible tangent line equation: 3x - y - 2 = 0 -/
def tangent1 (x y : ℝ) : Prop := 3 * x - y - 2 = 0

/-- The second possible tangent line equation: 3x - 4y + 17 = 0 -/
def tangent2 (x y : ℝ) : Prop := 3 * x - 4 * y + 17 = 0

/-- Theorem: The tangent line to curve S passing through point A
    is either tangent1 or tangent2 -/
theorem tangent_line_equation :
  ∃ (x y : ℝ), (y = S x ∧ (x, y) ≠ A) →
  (∀ (h k : ℝ), tangent1 h k ∨ tangent2 h k ↔ 
    (k - A.2) / (h - A.1) = 3 * x^2 ∧ k = S h) :=
sorry

end tangent_line_equation_l2467_246742


namespace max_value_of_z_l2467_246745

-- Define the system of inequalities and z
def system (x y : ℝ) : Prop :=
  x + y - Real.sqrt 2 ≤ 0 ∧
  x - y + Real.sqrt 2 ≥ 0 ∧
  y ≥ 0

def z (x y : ℝ) : ℝ := 2 * x - y

-- State the theorem
theorem max_value_of_z :
  ∃ (max_z : ℝ) (x_max y_max : ℝ),
    system x_max y_max ∧
    z x_max y_max = max_z ∧
    max_z = 2 * Real.sqrt 2 ∧
    x_max = Real.sqrt 2 ∧
    y_max = 0 ∧
    ∀ (x y : ℝ), system x y → z x y ≤ max_z :=
by sorry

end max_value_of_z_l2467_246745


namespace four_color_arrangement_l2467_246729

theorem four_color_arrangement : ∀ n : ℕ, n = 4 → (Nat.factorial n) = 24 := by
  sorry

end four_color_arrangement_l2467_246729


namespace bag_weight_l2467_246774

theorem bag_weight (w : ℝ) (h : w = 16 / (w / 4)) : w = 16 := by
  sorry

end bag_weight_l2467_246774


namespace complex_square_root_l2467_246786

theorem complex_square_root (a b : ℕ+) (h : (a - b * Complex.I) ^ 2 = 8 - 6 * Complex.I) :
  a - b * Complex.I = 3 - Complex.I := by
  sorry

end complex_square_root_l2467_246786


namespace decreasing_interval_of_f_shifted_l2467_246782

def f (x : ℝ) : ℝ := x^2 + 2*x - 5

theorem decreasing_interval_of_f_shifted :
  let g := fun (x : ℝ) => f (x - 1)
  ∀ x y : ℝ, x < y ∧ y ≤ 0 → g x > g y :=
by sorry

end decreasing_interval_of_f_shifted_l2467_246782


namespace negation_of_existence_negation_of_proposition_l2467_246703

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ n, p n) ↔ ∀ n, ¬ p n := by sorry

theorem negation_of_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end negation_of_existence_negation_of_proposition_l2467_246703


namespace zoo_animal_count_l2467_246707

/-- Represents the zoo layout and animal counts -/
structure Zoo where
  tigerEnclosures : Nat
  zebraEnclosuresPerTiger : Nat
  giraffeEnclosureMultiplier : Nat
  tigersPerEnclosure : Nat
  zebrasPerEnclosure : Nat
  giraffesPerEnclosure : Nat

/-- Calculates the total number of animals in the zoo -/
def totalAnimals (zoo : Zoo) : Nat :=
  let zebraEnclosures := zoo.tigerEnclosures * zoo.zebraEnclosuresPerTiger
  let giraffeEnclosures := zebraEnclosures * zoo.giraffeEnclosureMultiplier
  let tigers := zoo.tigerEnclosures * zoo.tigersPerEnclosure
  let zebras := zebraEnclosures * zoo.zebrasPerEnclosure
  let giraffes := giraffeEnclosures * zoo.giraffesPerEnclosure
  tigers + zebras + giraffes

/-- Theorem stating that the total number of animals in the zoo is 144 -/
theorem zoo_animal_count :
  ∀ (zoo : Zoo),
    zoo.tigerEnclosures = 4 →
    zoo.zebraEnclosuresPerTiger = 2 →
    zoo.giraffeEnclosureMultiplier = 3 →
    zoo.tigersPerEnclosure = 4 →
    zoo.zebrasPerEnclosure = 10 →
    zoo.giraffesPerEnclosure = 2 →
    totalAnimals zoo = 144 := by
  sorry

end zoo_animal_count_l2467_246707


namespace figure_reassemble_to_square_l2467_246797

/-- Represents a figure on a graph paper --/
structure GraphFigure where
  area : ℝ
  triangles : ℕ

/-- Represents a square --/
structure Square where
  side : ℝ

/-- Function to check if a figure can be reassembled into a square --/
def can_reassemble_to_square (figure : GraphFigure) (square : Square) : Prop :=
  figure.area = square.side ^ 2 ∧ figure.triangles = 5

/-- Theorem stating that the given figure can be reassembled into a square --/
theorem figure_reassemble_to_square :
  ∃ (figure : GraphFigure) (square : Square),
    figure.area = 20 ∧ can_reassemble_to_square figure square :=
by sorry

end figure_reassemble_to_square_l2467_246797


namespace tetrahedron_medians_intersect_l2467_246785

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A₁ : Point3D
  A₂ : Point3D
  A₃ : Point3D
  A₄ : Point3D

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Median of a tetrahedron -/
def median (t : Tetrahedron) (v : Fin 4) : Line3D :=
  sorry  -- Definition of median based on tetrahedron and vertex index

/-- Intersection point of two lines -/
def intersectionPoint (l1 l2 : Line3D) : Option Point3D :=
  sorry  -- Definition of intersection point of two lines

/-- Theorem: All medians of a tetrahedron intersect at a single point -/
theorem tetrahedron_medians_intersect (t : Tetrahedron) :
  ∃ (c : Point3D), ∀ (i j : Fin 4),
    intersectionPoint (median t i) (median t j) = some c :=
  sorry

end tetrahedron_medians_intersect_l2467_246785


namespace f_properties_l2467_246704

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 4
  else if x ≤ 4 then x^2 - 2*x
  else -x + 2

theorem f_properties :
  (f 0 = 4) ∧
  (f 5 = -3) ∧
  (f (f (f 5)) = -1) ∧
  (∃! a, f a = 8 ∧ a = 4) :=
sorry

end f_properties_l2467_246704


namespace sum_of_absolute_values_zero_l2467_246733

theorem sum_of_absolute_values_zero (a b : ℝ) :
  |a - 5| + |b + 8| = 0 → a + b = -3 := by
  sorry

end sum_of_absolute_values_zero_l2467_246733


namespace hyperbola_equation_l2467_246775

/-- Given an ellipse and a hyperbola with shared foci, prove the equation of the hyperbola -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    -- Ellipse equation
    (x^2 + 4*y^2 = 64) ∧
    -- Hyperbola shares foci with ellipse
    (a^2 + b^2 = 48) ∧
    -- Asymptote equation
    (x - Real.sqrt 3 * y = 0)) →
  -- Hyperbola equation
  x^2/36 - y^2/12 = 1 := by
sorry

end hyperbola_equation_l2467_246775


namespace rectangle_existence_l2467_246724

theorem rectangle_existence : ∃ (x y : ℝ), 
  (2 * (x + y) = 2 * (2 + 1) * 2) ∧ 
  (x * y = 2 * 1 * 2) ∧ 
  (x > 0) ∧ (y > 0) ∧
  (x = 3 + Real.sqrt 5) ∧ (y = 3 - Real.sqrt 5) := by
  sorry

end rectangle_existence_l2467_246724


namespace missy_dog_yells_l2467_246756

/-- Represents the number of times Missy yells at her dogs -/
structure DogYells where
  obedient : ℕ
  stubborn : ℕ
  total : ℕ

/-- Theorem: If Missy yells at the obedient dog 12 times and yells at both dogs combined 60 times,
    then she yells at the stubborn dog 4 times for every one time she yells at the obedient dog -/
theorem missy_dog_yells (d : DogYells) 
    (h1 : d.obedient = 12)
    (h2 : d.total = 60)
    (h3 : d.total = d.obedient + d.stubborn) :
    d.stubborn = 4 * d.obedient := by
  sorry

#check missy_dog_yells

end missy_dog_yells_l2467_246756


namespace bounded_sequence_characterization_l2467_246754

def sequence_rule (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 2) = (a (n + 1) + a n) / (Nat.gcd (a (n + 1)) (a n))

def is_bounded (a : ℕ → ℕ) : Prop :=
  ∃ M, ∀ n, a n ≤ M

theorem bounded_sequence_characterization (a : ℕ → ℕ) :
  (∀ n, a n > 0) →
  sequence_rule a →
  is_bounded a ↔ a 1 = 2 ∧ a 2 = 2 :=
by sorry

end bounded_sequence_characterization_l2467_246754


namespace insertion_sort_comparison_bounds_l2467_246747

/-- Insertion sort comparison count bounds -/
theorem insertion_sort_comparison_bounds (n : ℕ) :
  ∀ (list : List ℕ), list.length = n →
  ∃ (comparisons : ℕ),
    (n - 1 : ℝ) ≤ comparisons ∧ comparisons ≤ (n * (n - 1) : ℝ) / 2 :=
by sorry

end insertion_sort_comparison_bounds_l2467_246747


namespace min_value_of_function_l2467_246781

theorem min_value_of_function (a : ℝ) (h : a > 1) :
  ∃ (min_val : ℝ), min_val = 3 + 2 * Real.sqrt 2 ∧
  ∀ x > 1, x + x^2 / (x - 1) ≥ min_val ∧
  ∃ y > 1, y + y^2 / (y - 1) = min_val :=
by sorry

end min_value_of_function_l2467_246781


namespace count_multiples_of_three_is_12960_l2467_246794

/-- A function that returns the count of six-digit multiples of 3 where each digit is not greater than 5 -/
def count_multiples_of_three : ℕ :=
  let first_digit_options := 5  -- digits 1 to 5
  let other_digit_options := 6  -- digits 0 to 5
  let last_digit_options := 2   -- two options to make the sum divisible by 3
  first_digit_options * (other_digit_options ^ 4) * last_digit_options

/-- Theorem stating that the count of six-digit multiples of 3 where each digit is not greater than 5 is 12960 -/
theorem count_multiples_of_three_is_12960 : count_multiples_of_three = 12960 := by
  sorry

#eval count_multiples_of_three

end count_multiples_of_three_is_12960_l2467_246794


namespace exists_valid_distribution_with_plate_B_size_l2467_246792

/-- Represents a distribution of balls across three plates -/
structure BallDistribution where
  plateA : List Nat
  plateB : List Nat
  plateC : List Nat

/-- Checks if a given distribution satisfies the problem conditions -/
def isValidDistribution (d : BallDistribution) : Prop :=
  let allBalls := d.plateA ++ d.plateB ++ d.plateC
  (∀ n ∈ allBalls, 1 ≤ n ∧ n ≤ 15) ∧ 
  (allBalls.length = 15) ∧
  (d.plateA.length ≥ 4 ∧ d.plateB.length ≥ 4 ∧ d.plateC.length ≥ 4) ∧
  ((d.plateA.sum : Rat) / d.plateA.length = 3) ∧
  ((d.plateB.sum : Rat) / d.plateB.length = 8) ∧
  ((d.plateC.sum : Rat) / d.plateC.length = 13)

/-- The main theorem to be proved -/
theorem exists_valid_distribution_with_plate_B_size :
  ∃ d : BallDistribution, isValidDistribution d ∧ (d.plateB.length = 7 ∨ d.plateB.length = 5) := by
  sorry


end exists_valid_distribution_with_plate_B_size_l2467_246792


namespace inequalities_given_ordered_reals_l2467_246761

theorem inequalities_given_ordered_reals (a b c : ℝ) 
  (h1 : a > 0) (h2 : 0 > b) (h3 : b > c) : 
  (a / c > a / b) ∧ 
  ((a - b) / (a - c) > b / c) ∧ 
  (a - c ≥ 2 * Real.sqrt ((a - b) * (b - c))) := by
  sorry

end inequalities_given_ordered_reals_l2467_246761


namespace field_trip_total_l2467_246768

theorem field_trip_total (num_vans num_buses students_per_van students_per_bus teachers_per_van teachers_per_bus : ℕ) 
  (h1 : num_vans = 6)
  (h2 : num_buses = 8)
  (h3 : students_per_van = 6)
  (h4 : students_per_bus = 18)
  (h5 : teachers_per_van = 1)
  (h6 : teachers_per_bus = 2) :
  num_vans * students_per_van + num_buses * students_per_bus + 
  num_vans * teachers_per_van + num_buses * teachers_per_bus = 202 :=
by sorry

end field_trip_total_l2467_246768


namespace max_remainder_of_division_by_11_l2467_246709

theorem max_remainder_of_division_by_11 (A B C : ℕ) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A = 11 * B + C →
  C ≤ 10 :=
by sorry

end max_remainder_of_division_by_11_l2467_246709


namespace line_circle_intersection_range_l2467_246716

/-- The line equation y = kx + 1 -/
def line_equation (k x : ℝ) : ℝ := k * x + 1

/-- The circle equation x^2 + y^2 - 2ax + a^2 - 2a - 4 = 0 -/
def circle_equation (x y a : ℝ) : Prop := x^2 + y^2 - 2*a*x + a^2 - 2*a - 4 = 0

/-- The condition that the line always intersects with the circle -/
def always_intersects (k a : ℝ) : Prop :=
  ∀ x y : ℝ, line_equation k x = y → circle_equation x y a

theorem line_circle_intersection_range :
  ∀ k : ℝ, (∀ a : ℝ, always_intersects k a) ↔ -1 ≤ a ∧ a ≤ 3 :=
by sorry

end line_circle_intersection_range_l2467_246716


namespace blocks_in_specific_box_l2467_246728

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks that can fit in the box -/
def blocksInBox (box : BoxDimensions) (block : BoxDimensions) : ℕ :=
  (box.length / block.length) * (box.width / block.width) * (box.height / block.height)

theorem blocks_in_specific_box :
  let box := BoxDimensions.mk 4 3 2
  let block := BoxDimensions.mk 3 1 1
  blocksInBox box block = 6 := by sorry

end blocks_in_specific_box_l2467_246728


namespace max_wins_l2467_246726

/-- Given the ratio of Chloe's wins to Max's wins and Chloe's total wins,
    calculate Max's wins. -/
theorem max_wins (chloe_wins : ℕ) (chloe_ratio : ℕ) (max_ratio : ℕ) 
    (h1 : chloe_wins = 24)
    (h2 : chloe_ratio = 8)
    (h3 : max_ratio = 3) :
  chloe_wins * max_ratio / chloe_ratio = 9 := by
  sorry

#check max_wins

end max_wins_l2467_246726


namespace deal_or_no_deal_probability_l2467_246718

/-- The total number of boxes in the game -/
def total_boxes : ℕ := 30

/-- The number of boxes containing at least $50,000 -/
def high_value_boxes : ℕ := 9

/-- The target probability (50%) expressed as a fraction -/
def target_probability : ℚ := 1/2

/-- The minimum number of boxes that need to be eliminated -/
def boxes_to_eliminate : ℕ := 12

theorem deal_or_no_deal_probability :
  boxes_to_eliminate = total_boxes - 2 * high_value_boxes :=
by sorry

end deal_or_no_deal_probability_l2467_246718
