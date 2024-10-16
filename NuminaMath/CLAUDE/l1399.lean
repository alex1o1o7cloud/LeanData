import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1399_139995

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 6 * a 4 + 2 * a 8 * a 5 + a 9 * a 7 = 36 →
  a 5 + a 8 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1399_139995


namespace NUMINAMATH_CALUDE_markup_calculation_l1399_139934

-- Define the parameters for the two articles
def purchase_price1 : ℝ := 48
def purchase_price2 : ℝ := 60
def overhead_rate1 : ℝ := 0.35
def overhead_rate2 : ℝ := 0.40
def net_profit1 : ℝ := 18
def net_profit2 : ℝ := 22

-- Define the function to calculate markup
def calculate_markup (purchase_price overhead_rate net_profit : ℝ) : ℝ :=
  let overhead_cost := purchase_price * overhead_rate
  let total_cost := purchase_price + overhead_cost
  let selling_price := total_cost + net_profit
  selling_price - purchase_price

-- State the theorem
theorem markup_calculation :
  calculate_markup purchase_price1 overhead_rate1 net_profit1 = 34.80 ∧
  calculate_markup purchase_price2 overhead_rate2 net_profit2 = 46 := by
  sorry


end NUMINAMATH_CALUDE_markup_calculation_l1399_139934


namespace NUMINAMATH_CALUDE_eighth_grade_students_l1399_139901

/-- The number of students in eighth grade -/
def total_students (num_girls : ℕ) (num_boys : ℕ) : ℕ :=
  num_girls + num_boys

/-- The relationship between the number of boys and girls -/
def boys_girls_relation (num_girls : ℕ) (num_boys : ℕ) : Prop :=
  num_boys = 2 * num_girls - 16

theorem eighth_grade_students :
  ∃ (num_boys : ℕ),
    boys_girls_relation 28 num_boys ∧
    total_students 28 num_boys = 68 :=
by sorry

end NUMINAMATH_CALUDE_eighth_grade_students_l1399_139901


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1399_139903

theorem complex_number_in_third_quadrant : 
  let z : ℂ := (2 - I) / I
  (z.re < 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1399_139903


namespace NUMINAMATH_CALUDE_milk_pour_problem_l1399_139904

theorem milk_pour_problem (initial_milk : ℚ) (pour_fraction : ℚ) :
  initial_milk = 3/8 →
  pour_fraction = 5/6 →
  pour_fraction * initial_milk = 5/16 := by
sorry

end NUMINAMATH_CALUDE_milk_pour_problem_l1399_139904


namespace NUMINAMATH_CALUDE_chord_count_for_concentric_circles_l1399_139938

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle formed by two adjacent chords is 60°, then exactly 3 such chords are needed to complete a full circle. -/
theorem chord_count_for_concentric_circles (angle : ℝ) (n : ℕ) : 
  angle = 60 → n * angle = 360 → n = 3 := by sorry

end NUMINAMATH_CALUDE_chord_count_for_concentric_circles_l1399_139938


namespace NUMINAMATH_CALUDE_no_triangle_solution_l1399_139950

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  A : ℝ

-- Theorem stating that no triangle exists with the given conditions
theorem no_triangle_solution :
  ¬ ∃ (t : Triangle), t.a = 181 ∧ t.b = 209 ∧ t.A = 121 := by
  sorry

end NUMINAMATH_CALUDE_no_triangle_solution_l1399_139950


namespace NUMINAMATH_CALUDE_derivative_at_pi_third_l1399_139959

theorem derivative_at_pi_third (f : ℝ → ℝ) (h : ∀ x, f x = x + Real.sin x) :
  deriv f (π / 3) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_pi_third_l1399_139959


namespace NUMINAMATH_CALUDE_hexagram_ratio_is_three_l1399_139968

/-- A hexagram formed by overlapping two equilateral triangles -/
structure Hexagram where
  /-- The hexagram's vertices coincide with those of a regular hexagon -/
  vertices_coincide : Bool
  /-- The number of smaller triangles in the shaded region -/
  shaded_triangles : Nat
  /-- The number of smaller triangles in the unshaded region -/
  unshaded_triangles : Nat

/-- The ratio of shaded to unshaded area in a hexagram -/
def shaded_unshaded_ratio (h : Hexagram) : ℚ :=
  h.shaded_triangles / h.unshaded_triangles

/-- Theorem: The ratio of shaded to unshaded area in the specified hexagram is 3 -/
theorem hexagram_ratio_is_three (h : Hexagram) 
  (h_vertices : h.vertices_coincide = true)
  (h_shaded : h.shaded_triangles = 18)
  (h_unshaded : h.unshaded_triangles = 6) : 
  shaded_unshaded_ratio h = 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagram_ratio_is_three_l1399_139968


namespace NUMINAMATH_CALUDE_sqrt_190_44_sqrt_176_9_and_18769_integer_between_sqrt_l1399_139956

-- Define the square root function
noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Theorem 1
theorem sqrt_190_44 : sqrt 190.44 = 13.8 ∨ sqrt 190.44 = -13.8 := by sorry

-- Theorem 2
theorem sqrt_176_9_and_18769 :
  (13.3 < sqrt 176.9 ∧ sqrt 176.9 < 13.4) ∧ sqrt 18769 = 137 := by sorry

-- Theorem 3
theorem integer_between_sqrt :
  ∀ n : ℤ, (13.5 < sqrt (n : ℝ) ∧ sqrt (n : ℝ) < 13.6) → (n = 183 ∨ n = 184) := by sorry

end NUMINAMATH_CALUDE_sqrt_190_44_sqrt_176_9_and_18769_integer_between_sqrt_l1399_139956


namespace NUMINAMATH_CALUDE_change_after_purchase_l1399_139905

/-- Calculates the change after a purchase given initial amount, number of items, and cost per item. -/
def calculate_change (initial_amount : ℕ) (num_items : ℕ) (cost_per_item : ℕ) : ℕ :=
  initial_amount - (num_items * cost_per_item)

/-- Theorem stating that given $20 initially, buying 3 items at $2 each results in $14 change. -/
theorem change_after_purchase :
  calculate_change 20 3 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_change_after_purchase_l1399_139905


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1399_139975

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 3 ∧ b = 4 ∧ c^2 = a^2 + b^2 ∨ (a = 3 ∧ b = 4 ∧ c = b) → c = 5 ∨ c = 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1399_139975


namespace NUMINAMATH_CALUDE_two_roses_more_expensive_than_three_carnations_l1399_139913

/-- Price of a single rose in yuan -/
def rose_price : ℝ := sorry

/-- Price of a single carnation in yuan -/
def carnation_price : ℝ := sorry

/-- The total price of 6 roses and 3 carnations is greater than 24 yuan -/
axiom condition1 : 6 * rose_price + 3 * carnation_price > 24

/-- The total price of 4 roses and 5 carnations is less than 22 yuan -/
axiom condition2 : 4 * rose_price + 5 * carnation_price < 22

/-- Theorem: The price of 2 roses is higher than the price of 3 carnations -/
theorem two_roses_more_expensive_than_three_carnations :
  2 * rose_price > 3 * carnation_price := by sorry

end NUMINAMATH_CALUDE_two_roses_more_expensive_than_three_carnations_l1399_139913


namespace NUMINAMATH_CALUDE_problem_solution_l1399_139953

def y (m x : ℝ) : ℝ := (m + 1) * x^2 - m * x + m - 1

theorem problem_solution :
  (∀ m : ℝ, (∀ x : ℝ, y m x ≥ 0) ↔ m ≥ 2 * Real.sqrt 3 / 3) ∧
  (∀ m : ℝ, m > -2 →
    (∀ x : ℝ, y m x ≥ m) ↔
      (m = -1 ∧ x ≥ 1) ∨
      (m > -1 ∧ (x ≤ -1 / (m + 1) ∨ x ≥ 1)) ∨
      (-2 < m ∧ m < -1 ∧ 1 ≤ x ∧ x ≤ -1 / (m + 1))) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1399_139953


namespace NUMINAMATH_CALUDE_area_triangle_pqr_l1399_139920

/-- Given two lines intersecting at point P(2,8), with slopes 1 and 3 respectively,
    and Q and R being the intersections of these lines with the x-axis,
    the area of triangle PQR is 64/3. -/
theorem area_triangle_pqr :
  let P : ℝ × ℝ := (2, 8)
  let slope1 : ℝ := 1
  let slope2 : ℝ := 3
  let Q : ℝ × ℝ := (P.1 - P.2 / slope1, 0)
  let R : ℝ × ℝ := (P.1 - P.2 / slope2, 0)
  let area : ℝ := (1 / 2) * |Q.1 - R.1| * P.2
  area = 64 / 3 := by
sorry

end NUMINAMATH_CALUDE_area_triangle_pqr_l1399_139920


namespace NUMINAMATH_CALUDE_picture_area_l1399_139943

/-- The area of a picture on a sheet of paper with given dimensions and margins. -/
theorem picture_area (paper_width paper_length margin : ℝ) 
  (hw : paper_width = 8.5)
  (hl : paper_length = 10)
  (hm : margin = 1.5) : 
  (paper_width - 2 * margin) * (paper_length - 2 * margin) = 38.5 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l1399_139943


namespace NUMINAMATH_CALUDE_pet_store_cages_used_l1399_139939

def pet_store_problem (initial_puppies sold_puppies puppies_per_cage : ℕ) : ℕ :=
  (initial_puppies - sold_puppies) / puppies_per_cage

theorem pet_store_cages_used :
  pet_store_problem 78 30 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_used_l1399_139939


namespace NUMINAMATH_CALUDE_solution_of_linear_system_l1399_139902

theorem solution_of_linear_system :
  ∃ (x y : ℝ), x + 3 * y = 7 ∧ y = 2 * x ∧ x = 1 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_linear_system_l1399_139902


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1399_139946

/-- Given a square with perimeter 160 units divided into 4 congruent rectangles,
    prove that the perimeter of one rectangle is 120 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 160) :
  let square_side : ℝ := square_perimeter / 4
  let rect_width : ℝ := square_side / 2
  let rect_height : ℝ := square_side
  let rect_perimeter : ℝ := 2 * (rect_width + rect_height)
  rect_perimeter = 120 := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1399_139946


namespace NUMINAMATH_CALUDE_age_of_30th_student_l1399_139970

theorem age_of_30th_student 
  (total_students : Nat)
  (avg_age_all : ℝ)
  (num_group1 : Nat) (avg_age_group1 : ℝ)
  (num_group2 : Nat) (avg_age_group2 : ℝ)
  (num_group3 : Nat) (avg_age_group3 : ℝ)
  (age_single_student : ℝ)
  (h1 : total_students = 30)
  (h2 : avg_age_all = 23.5)
  (h3 : num_group1 = 9)
  (h4 : avg_age_group1 = 21.3)
  (h5 : num_group2 = 12)
  (h6 : avg_age_group2 = 19.7)
  (h7 : num_group3 = 7)
  (h8 : avg_age_group3 = 24.2)
  (h9 : age_single_student = 35)
  (h10 : num_group1 + num_group2 + num_group3 + 1 + 1 = total_students) :
  total_students * avg_age_all - 
  (num_group1 * avg_age_group1 + num_group2 * avg_age_group2 + 
   num_group3 * avg_age_group3 + age_single_student) = 72.5 := by
  sorry

end NUMINAMATH_CALUDE_age_of_30th_student_l1399_139970


namespace NUMINAMATH_CALUDE_total_percentage_increase_approx_l1399_139979

/-- Calculates the total percentage increase in USD for a purchase of three items with given initial and final prices in different currencies. -/
theorem total_percentage_increase_approx (book_initial book_final : ℝ)
                                         (album_initial album_final : ℝ)
                                         (poster_initial poster_final : ℝ)
                                         (usd_to_eur usd_to_gbp : ℝ)
                                         (h1 : book_initial = 300)
                                         (h2 : book_final = 480)
                                         (h3 : album_initial = 15)
                                         (h4 : album_final = 20)
                                         (h5 : poster_initial = 5)
                                         (h6 : poster_final = 10)
                                         (h7 : usd_to_eur = 0.85)
                                         (h8 : usd_to_gbp = 0.75) :
  ∃ ε > 0, abs (((book_final - book_initial + 
                 (album_final - album_initial) / usd_to_eur + 
                 (poster_final - poster_initial) / usd_to_gbp) / 
                (book_initial + album_initial / usd_to_eur + 
                 poster_initial / usd_to_gbp)) - 0.5937) < ε :=
by sorry


end NUMINAMATH_CALUDE_total_percentage_increase_approx_l1399_139979


namespace NUMINAMATH_CALUDE_derivative_of_x_plus_exp_l1399_139910

/-- The derivative of f(x) = x + e^x is f'(x) = 1 + e^x -/
theorem derivative_of_x_plus_exp (x : ℝ) :
  deriv (fun x => x + Real.exp x) x = 1 + Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_x_plus_exp_l1399_139910


namespace NUMINAMATH_CALUDE_negative_two_a_cubed_l1399_139960

theorem negative_two_a_cubed (a : ℝ) : (-2 * a)^3 = -8 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_a_cubed_l1399_139960


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1399_139999

/-- Simple interest calculation problem -/
theorem simple_interest_problem (rate : ℚ) (principal : ℚ) (interest_diff : ℚ) (years : ℚ) :
  rate = 4 / 100 →
  principal = 2400 →
  principal * rate * years = principal - interest_diff →
  interest_diff = 1920 →
  years = 5 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1399_139999


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l1399_139952

theorem triangle_angle_problem (α : Real) 
  (h1 : 0 < α ∧ α < π) -- α is an internal angle of a triangle
  (h2 : Real.sin α + Real.cos α = 1/5) :
  (Real.tan α = -4/3) ∧ 
  ((Real.sin (3*π/2 + α) * Real.sin (π/2 - α) * Real.tan (π - α)^3) / 
   (Real.cos (π/2 + α) * Real.cos (3*π/2 - α)) = -4/3) := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l1399_139952


namespace NUMINAMATH_CALUDE_volume_of_specific_box_l1399_139932

/-- The volume of a box formed by cutting squares from corners of a rectangle --/
def box_volume (length width y : ℝ) : ℝ :=
  (length - 2*y) * (width - 2*y) * y

/-- Theorem: The volume of the box formed from a 12 by 15 inch sheet --/
theorem volume_of_specific_box (y : ℝ) :
  box_volume 15 12 y = 180*y - 54*y^2 + 4*y^3 :=
by sorry

end NUMINAMATH_CALUDE_volume_of_specific_box_l1399_139932


namespace NUMINAMATH_CALUDE_max_value_of_a_l1399_139912

theorem max_value_of_a : 
  (∃ (a : ℝ), ∀ (x : ℝ), x < a → x^2 - 2*x - 3 > 0) ∧ 
  (∃ (x : ℝ), x^2 - 2*x - 3 > 0 ∧ ¬(∀ (a : ℝ), x < a)) →
  (∀ (b : ℝ), (∀ (x : ℝ), x < b → x^2 - 2*x - 3 > 0) → b ≤ -1) ∧
  (∀ (x : ℝ), x < -1 → x^2 - 2*x - 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1399_139912


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_l1399_139916

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.width)

/-- Calculates the perimeter of a square -/
def Square.perimeter (s : Square) : ℝ :=
  4 * s.side

/-- Theorem stating the perimeter of the large rectangle -/
theorem large_rectangle_perimeter 
  (square : Square)
  (small_rect : Rectangle)
  (h1 : square.perimeter = 24)
  (h2 : small_rect.perimeter = 16)
  (h3 : small_rect.length = square.side)
  (h4 : small_rect.width + square.side = small_rect.length) :
  let large_rect := Rectangle.mk (square.side + 2 * small_rect.length) (small_rect.width + square.side)
  large_rect.perimeter = 52 := by
  sorry


end NUMINAMATH_CALUDE_large_rectangle_perimeter_l1399_139916


namespace NUMINAMATH_CALUDE_square_difference_l1399_139957

theorem square_difference (x y : ℝ) 
  (h1 : (x + y) / 2 = 5)
  (h2 : (x - y) / 2 = 2) : 
  x^2 - y^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l1399_139957


namespace NUMINAMATH_CALUDE_prime_sequence_ones_digit_l1399_139948

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def ones_digit (n : ℕ) : ℕ := n % 10

theorem prime_sequence_ones_digit (p q r s : ℕ) :
  is_prime p → is_prime q → is_prime r → is_prime s →
  p > 5 →
  q = p + 8 →
  r = q + 8 →
  s = r + 8 →
  ones_digit p = 3 := by
sorry

end NUMINAMATH_CALUDE_prime_sequence_ones_digit_l1399_139948


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1399_139918

-- Define the given line
def given_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

-- Define the point that the desired line passes through
def point : ℝ × ℝ := (-1, 3)

-- Define the desired line
def desired_line (x y : ℝ) : Prop := y + 2*x - 1 = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  (∀ x y, given_line x y → desired_line x y → (x - point.1) * (x - point.1) + (y - point.2) * (y - point.2) = 0) ∧
  (∀ x₁ y₁ x₂ y₂, given_line x₁ y₁ → given_line x₂ y₂ → desired_line x₁ y₁ → desired_line x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (y₂ - y₁)) / ((x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁)) = -1/2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1399_139918


namespace NUMINAMATH_CALUDE_concession_stand_sales_l1399_139935

/-- Concession stand sales problem -/
theorem concession_stand_sales
  (hot_dog_cost : ℚ)
  (soda_cost : ℚ)
  (total_revenue : ℚ)
  (hot_dogs_sold : ℕ)
  (h1 : hot_dog_cost = 3/2)
  (h2 : soda_cost = 1/2)
  (h3 : total_revenue = 157/2)
  (h4 : hot_dogs_sold = 35) :
  ∃ (sodas_sold : ℕ), hot_dogs_sold + sodas_sold = 87 :=
by sorry

end NUMINAMATH_CALUDE_concession_stand_sales_l1399_139935


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1399_139982

theorem quadratic_inequality_solution_set (x : ℝ) : 
  x^2 - 2*x ≤ 0 ↔ 0 ≤ x ∧ x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1399_139982


namespace NUMINAMATH_CALUDE_factorial_6_eq_720_l1399_139931

def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_6_eq_720 : factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_factorial_6_eq_720_l1399_139931


namespace NUMINAMATH_CALUDE_f_of_two_eq_two_l1399_139969

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x + 3 * f (8 - x) = x

/-- Theorem stating that for any function satisfying the functional equation, f(2) = 2 -/
theorem f_of_two_eq_two (f : ℝ → ℝ) (h : FunctionalEquation f) : f 2 = 2 := by
  sorry

#check f_of_two_eq_two

end NUMINAMATH_CALUDE_f_of_two_eq_two_l1399_139969


namespace NUMINAMATH_CALUDE_dimitri_calories_l1399_139926

/-- Calculates the total calories consumed by Dimitri over two days -/
def calories_two_days (burgers_per_day : ℕ) (calories_per_burger : ℕ) : ℕ :=
  2 * burgers_per_day * calories_per_burger

/-- Proves that Dimitri consumes 120 calories over two days -/
theorem dimitri_calories : calories_two_days 3 20 = 120 := by
  sorry

end NUMINAMATH_CALUDE_dimitri_calories_l1399_139926


namespace NUMINAMATH_CALUDE_max_value_problem_l1399_139961

theorem max_value_problem (m n k : ℕ) (a b c : ℕ → ℕ) :
  (∀ i ∈ Finset.range m, a i % 3 = 1) →
  (∀ i ∈ Finset.range n, b i % 3 = 2) →
  (∀ i ∈ Finset.range k, c i % 3 = 0) →
  (∀ i j, i ≠ j → (a i ≠ a j ∧ b i ≠ b j ∧ c i ≠ c j ∧ 
                   a i ≠ b j ∧ a i ≠ c j ∧ b i ≠ c j)) →
  (Finset.sum (Finset.range m) a + Finset.sum (Finset.range n) b + 
   Finset.sum (Finset.range k) c = 2007) →
  4 * m + 3 * n + 5 * k ≤ 256 := by
sorry

end NUMINAMATH_CALUDE_max_value_problem_l1399_139961


namespace NUMINAMATH_CALUDE_disjoint_subsets_remainder_l1399_139930

def T : Finset Nat := Finset.range 15

def m : Nat :=
  (3^15 - 2 * 2^15 + 1) / 2

theorem disjoint_subsets_remainder : m % 1000 = 686 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subsets_remainder_l1399_139930


namespace NUMINAMATH_CALUDE_fraction_irreducible_fraction_simplification_l1399_139907

-- Part (a)
theorem fraction_irreducible (a : ℤ) : 
  Int.gcd (a^3 + 2*a) (a^4 + 3*a^2 + 1) = 1 := by sorry

-- Part (b)
theorem fraction_simplification (n : ℤ) : 
  Int.gcd (5*n + 6) (8*n + 7) = 1 ∨ Int.gcd (5*n + 6) (8*n + 7) = 13 := by sorry

end NUMINAMATH_CALUDE_fraction_irreducible_fraction_simplification_l1399_139907


namespace NUMINAMATH_CALUDE_age_pencil_ratio_l1399_139929

/-- Given the ages and pencil counts of Asaf and Alexander, prove the ratio of their age difference to Asaf's pencil count -/
theorem age_pencil_ratio (asaf_age alexander_age asaf_pencils alexander_pencils : ℕ) : 
  asaf_age = 50 →
  asaf_age + alexander_age = 140 →
  alexander_pencils = asaf_pencils + 60 →
  asaf_pencils + alexander_pencils = 220 →
  (alexander_age - asaf_age : ℚ) / asaf_pencils = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_age_pencil_ratio_l1399_139929


namespace NUMINAMATH_CALUDE_x_minus_y_value_l1399_139942

theorem x_minus_y_value (x y : ℝ) 
  (h1 : |x| = 3) 
  (h2 : y^2 = 1/4) 
  (h3 : x + y < 0) : 
  x - y = -7/2 ∨ x - y = -5/2 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l1399_139942


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1399_139915

theorem quadratic_equation_solution : 
  ∃ x : ℝ, 3 * x^2 - 6 * x + 3 = 0 ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1399_139915


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l1399_139984

-- Define the triangles
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangles
def is_30_60_90 (t : Triangle) : Prop := sorry

def is_right_angled_isosceles (t : Triangle) : Prop := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define the theorem
theorem area_ratio_theorem (FGH IFG EGH IEH : Triangle) :
  is_30_60_90 FGH →
  is_right_angled_isosceles EGH →
  (area IFG) / (area IEH) = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l1399_139984


namespace NUMINAMATH_CALUDE_building_height_calculation_l1399_139986

-- Define the given constants
def box_height : ℝ := 3
def box_shadow : ℝ := 12
def building_shadow : ℝ := 36

-- Define the theorem
theorem building_height_calculation :
  ∃ (building_height : ℝ),
    (box_height / box_shadow = building_height / building_shadow) ∧
    building_height = 9 := by
  sorry

end NUMINAMATH_CALUDE_building_height_calculation_l1399_139986


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1399_139900

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 15 + 20 + 5 + y) / 5 = 12 → y = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1399_139900


namespace NUMINAMATH_CALUDE_driveway_snow_volume_l1399_139974

/-- The volume of snow on a driveway -/
def snow_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- Theorem: The volume of snow on a driveway with length 30 feet, width 3 feet, 
    and snow depth 0.75 feet is equal to 67.5 cubic feet -/
theorem driveway_snow_volume :
  snow_volume 30 3 0.75 = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_driveway_snow_volume_l1399_139974


namespace NUMINAMATH_CALUDE_hyperbola_condition_l1399_139998

/-- A curve of the form ax^2 + by^2 = 1 is a hyperbola if ab < 0 -/
def is_hyperbola (a b : ℝ) : Prop := a * b < 0

/-- The curve mx^2 - (m-2)y^2 = 1 -/
def curve (m : ℝ) : (ℝ → ℝ → Prop) := λ x y => m * x^2 - (m - 2) * y^2 = 1

theorem hyperbola_condition (m : ℝ) :
  (∀ m > 3, is_hyperbola m (2 - m)) ∧
  (∃ m ≤ 3, is_hyperbola m (2 - m)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l1399_139998


namespace NUMINAMATH_CALUDE_pie_distribution_problem_l1399_139925

theorem pie_distribution_problem :
  ∃! (p b a h : ℕ),
    p + b + a + h = 30 ∧
    b + p = a + h ∧
    p + a = 6 * (b + h) ∧
    h < p ∧ h < b ∧ h < a ∧
    h ≥ 1 ∧ p ≥ 1 ∧ b ≥ 1 ∧ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_pie_distribution_problem_l1399_139925


namespace NUMINAMATH_CALUDE_problem_1999_squared_minus_1998_times_2002_l1399_139964

theorem problem_1999_squared_minus_1998_times_2002 : 1999^2 - 1998 * 2002 = -3991 := by
  sorry

end NUMINAMATH_CALUDE_problem_1999_squared_minus_1998_times_2002_l1399_139964


namespace NUMINAMATH_CALUDE_average_income_P_and_R_l1399_139987

/-- Given the average monthly incomes of different pairs of people and the income of one person,
    prove that the average monthly income of P and R is 5200. -/
theorem average_income_P_and_R (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 →
  (Q + R) / 2 = 6250 →
  P = 4000 →
  (P + R) / 2 = 5200 := by
  sorry

end NUMINAMATH_CALUDE_average_income_P_and_R_l1399_139987


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1399_139980

/-- Given a > 0 and the coefficient of the 1/x term in the expansion of (a√x - 1/√x)^6 is 135, prove that a = 3 -/
theorem binomial_expansion_coefficient (a : ℝ) (h1 : a > 0) 
  (h2 : (Nat.choose 6 4) * a^2 = 135) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1399_139980


namespace NUMINAMATH_CALUDE_contradiction_proof_l1399_139996

theorem contradiction_proof (x : ℝ) : (x^2 - 1 = 0) → (x = -1 ∨ x = 1) := by
  contrapose
  intro h
  have h1 : x ≠ -1 ∧ x ≠ 1 := by
    push_neg at h
    exact h
  sorry

end NUMINAMATH_CALUDE_contradiction_proof_l1399_139996


namespace NUMINAMATH_CALUDE_paula_bumper_car_rides_l1399_139971

/-- Calculates the number of bumper car rides Paula can take given the total tickets,
    go-kart ticket cost, and bumper car ticket cost. -/
def bumper_car_rides (total_tickets go_kart_cost bumper_car_cost : ℕ) : ℕ :=
  (total_tickets - go_kart_cost) / bumper_car_cost

/-- Proves that Paula can ride the bumper cars 4 times given the conditions. -/
theorem paula_bumper_car_rides :
  let total_tickets : ℕ := 24
  let go_kart_cost : ℕ := 4
  let bumper_car_cost : ℕ := 5
  bumper_car_rides total_tickets go_kart_cost bumper_car_cost = 4 := by
  sorry


end NUMINAMATH_CALUDE_paula_bumper_car_rides_l1399_139971


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l1399_139908

theorem six_digit_numbers_with_zero (total_six_digit : Nat) (six_digit_no_zero : Nat) :
  total_six_digit = 900000 →
  six_digit_no_zero = 531441 →
  total_six_digit - six_digit_no_zero = 368559 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l1399_139908


namespace NUMINAMATH_CALUDE_same_side_probability_is_seven_twentyfourths_l1399_139958

/-- Represents a 12-sided die with specific colored sides. -/
structure TwelveSidedDie :=
  (maroon : Nat)
  (teal : Nat)
  (cyan : Nat)
  (sparkly : Nat)
  (total_sides : Nat)
  (side_sum : maroon + teal + cyan + sparkly = total_sides)

/-- The probability of two dice showing the same side when rolled. -/
def same_side_probability (d : TwelveSidedDie) : Rat :=
  (d.maroon^2 + d.teal^2 + d.cyan^2 + d.sparkly^2) / d.total_sides^2

/-- The specific die used in the problem. -/
def problem_die : TwelveSidedDie :=
  { maroon := 3
    teal := 4
    cyan := 4
    sparkly := 1
    total_sides := 12
    side_sum := by decide }

/-- Theorem stating that the probability of two problem dice showing the same side is 7/24. -/
theorem same_side_probability_is_seven_twentyfourths :
  same_side_probability problem_die = 7 / 24 := by
  sorry

end NUMINAMATH_CALUDE_same_side_probability_is_seven_twentyfourths_l1399_139958


namespace NUMINAMATH_CALUDE_angle_DEB_value_l1399_139945

-- Define the geometric configuration
structure GeometricConfig where
  -- Triangle ABC
  angleABC : ℝ
  angleACB : ℝ
  -- Other angles
  angleCDE : ℝ
  -- Straight line and angle conditions
  angleADC_straight : angleADC = 180
  angleAEB_straight : angleAEB = 180
  -- Given conditions
  h1 : angleABC = 72
  h2 : angleACB = 90
  h3 : angleCDE = 36

-- Theorem statement
theorem angle_DEB_value (config : GeometricConfig) : ∃ (angleDEB : ℝ), angleDEB = 162 := by
  sorry


end NUMINAMATH_CALUDE_angle_DEB_value_l1399_139945


namespace NUMINAMATH_CALUDE_oblique_drawing_parallelogram_oblique_drawing_other_shapes_l1399_139944

/-- Represents a shape in 2D space -/
inductive Shape
  | Triangle
  | Parallelogram
  | Square
  | Rhombus

/-- Represents the result of applying the oblique drawing method to a shape -/
def obliqueDrawing (s : Shape) : Shape :=
  match s with
  | Shape.Parallelogram => Shape.Parallelogram
  | _ => Shape.Parallelogram  -- Simplified for this problem

/-- Theorem stating that the oblique drawing of a parallelogram is always a parallelogram -/
theorem oblique_drawing_parallelogram :
  ∀ s : Shape, s = Shape.Parallelogram → obliqueDrawing s = Shape.Parallelogram :=
by sorry

/-- Theorem stating that the oblique drawing of non-parallelogram shapes may not preserve the original shape -/
theorem oblique_drawing_other_shapes :
  ∃ s : Shape, s ≠ Shape.Parallelogram ∧ obliqueDrawing s ≠ s :=
by sorry

end NUMINAMATH_CALUDE_oblique_drawing_parallelogram_oblique_drawing_other_shapes_l1399_139944


namespace NUMINAMATH_CALUDE_unique_linear_function_l1399_139906

/-- A linear function passing through two given points -/
def linear_function_through_points (k b : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  k ≠ 0 ∧ y₁ = k * x₁ + b ∧ y₂ = k * x₂ + b

theorem unique_linear_function :
  ∃! k b : ℝ, linear_function_through_points k b 1 3 0 (-2) ∧ 
  ∀ x : ℝ, k * x + b = 5 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_linear_function_l1399_139906


namespace NUMINAMATH_CALUDE_total_cost_for_group_stay_l1399_139923

-- Define the rates and conditions
def weekdayRateFirstWeek : ℚ := 18
def weekendRateFirstWeek : ℚ := 20
def weekdayRateAdditionalWeeks : ℚ := 11
def weekendRateAdditionalWeeks : ℚ := 13
def securityDeposit : ℚ := 50
def groupDiscountRate : ℚ := 0.1
def groupSize : ℕ := 5
def stayDuration : ℕ := 23

-- Define the function to calculate the total cost
def calculateTotalCost : ℚ := sorry

-- Theorem statement
theorem total_cost_for_group_stay :
  calculateTotalCost = 327.6 := by sorry

end NUMINAMATH_CALUDE_total_cost_for_group_stay_l1399_139923


namespace NUMINAMATH_CALUDE_fruit_basket_total_l1399_139928

/-- Represents the number of fruit pieces in a basket -/
structure FruitBasket where
  redApples : Nat
  greenApples : Nat
  purpleGrapes : Nat
  yellowBananas : Nat
  orangeOranges : Nat

/-- Calculates the total number of fruit pieces in the basket -/
def totalFruits (basket : FruitBasket) : Nat :=
  basket.redApples + basket.greenApples + basket.purpleGrapes + basket.yellowBananas + basket.orangeOranges

/-- Theorem stating that the total number of fruit pieces in the given basket is 24 -/
theorem fruit_basket_total :
  let basket : FruitBasket := {
    redApples := 9,
    greenApples := 4,
    purpleGrapes := 3,
    yellowBananas := 6,
    orangeOranges := 2
  }
  totalFruits basket = 24 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_total_l1399_139928


namespace NUMINAMATH_CALUDE_rockham_soccer_league_members_l1399_139940

theorem rockham_soccer_league_members :
  let sock_cost : ℕ := 6
  let tshirt_cost : ℕ := sock_cost + 10
  let cap_cost : ℕ := 3
  let items_per_member : ℕ := 2  -- for both home and away games
  let cost_per_member : ℕ := items_per_member * (sock_cost + tshirt_cost + cap_cost)
  let total_expenditure : ℕ := 4620
  total_expenditure / cost_per_member = 92 :=
by sorry

end NUMINAMATH_CALUDE_rockham_soccer_league_members_l1399_139940


namespace NUMINAMATH_CALUDE_x_greater_than_y_l1399_139951

theorem x_greater_than_y (x y z : ℝ) 
  (eq1 : x + y + z = 28)
  (eq2 : 2 * x - y = 32)
  (pos_x : x > 0)
  (pos_y : y > 0)
  (pos_z : z > 0) :
  x > y := by
  sorry

end NUMINAMATH_CALUDE_x_greater_than_y_l1399_139951


namespace NUMINAMATH_CALUDE_lucilles_earnings_l1399_139972

/-- Represents the earnings in cents for each type of weed -/
structure WeedEarnings where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Represents the number of weeds in a garden area -/
structure WeedCount where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total earnings from weeding a garden area -/
def calculateEarnings (earnings : WeedEarnings) (count : WeedCount) : ℕ :=
  earnings.small * count.small + earnings.medium * count.medium + earnings.large * count.large

/-- Represents Lucille's weeding earnings problem -/
structure LucillesProblem where
  earnings : WeedEarnings
  flowerBed : WeedCount
  vegetablePatch : WeedCount
  grass : WeedCount
  sodaCost : ℕ
  salesTaxPercent : ℕ

/-- Theorem stating that Lucille has 130 cents left after buying the soda -/
theorem lucilles_earnings (problem : LucillesProblem)
  (h1 : problem.earnings = ⟨4, 8, 12⟩)
  (h2 : problem.flowerBed = ⟨6, 3, 2⟩)
  (h3 : problem.vegetablePatch = ⟨10, 2, 2⟩)
  (h4 : problem.grass = ⟨20, 10, 2⟩)
  (h5 : problem.sodaCost = 99)
  (h6 : problem.salesTaxPercent = 15) :
  let totalEarnings := calculateEarnings problem.earnings problem.flowerBed +
                       calculateEarnings problem.earnings problem.vegetablePatch +
                       calculateEarnings problem.earnings ⟨problem.grass.small / 2, problem.grass.medium / 2, problem.grass.large / 2⟩
  let sodaTotalCost := problem.sodaCost + (problem.sodaCost * problem.salesTaxPercent / 100 + 1)
  totalEarnings - sodaTotalCost = 130 := by sorry


end NUMINAMATH_CALUDE_lucilles_earnings_l1399_139972


namespace NUMINAMATH_CALUDE_right_triangle_area_l1399_139917

theorem right_triangle_area (a b : ℝ) (h1 : a^2 - 7*a + 12 = 0) (h2 : b^2 - 7*b + 12 = 0) (h3 : a ≠ b) :
  ∃ (area : ℝ), (area = 6 ∨ area = (3 * Real.sqrt 7) / 2) ∧
  ((area = a * b / 2) ∨ (area = a * Real.sqrt (b^2 - a^2) / 2) ∨ (area = b * Real.sqrt (a^2 - b^2) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1399_139917


namespace NUMINAMATH_CALUDE_perimeter_ratio_from_area_ratio_l1399_139966

theorem perimeter_ratio_from_area_ratio (s1 s2 : ℝ) (h : s1 > 0 ∧ s2 > 0) 
  (h_area_ratio : s1^2 / s2^2 = 49 / 64) : 
  (4 * s1) / (4 * s2) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_from_area_ratio_l1399_139966


namespace NUMINAMATH_CALUDE_weight_problem_l1399_139949

/-- Given three weights a, b, and c, prove that their average weights satisfy the given conditions -/
theorem weight_problem (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 43)
  (h2 : (b + c) / 2 = 43)
  (h3 : b = 37) :
  (a + b) / 2 = 40 := by
sorry

end NUMINAMATH_CALUDE_weight_problem_l1399_139949


namespace NUMINAMATH_CALUDE_ellipse_focus_l1399_139985

/-- An ellipse with semi-major axis 5 and semi-minor axis m has its left focus at (-3,0) -/
theorem ellipse_focus (m : ℝ) (h1 : m > 0) : 
  (∀ x y : ℝ, x^2 / 25 + y^2 / m^2 = 1) → (-3 : ℝ)^2 = 25 - m^2 → m = 4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_l1399_139985


namespace NUMINAMATH_CALUDE_sector_area_l1399_139988

/-- Given a circular sector with central angle 1 radian and circumference 6,
    prove that its area is 2. -/
theorem sector_area (θ : Real) (c : Real) (h1 : θ = 1) (h2 : c = 6) :
  let r := c / 3
  (1/2) * r^2 * θ = 2 := by sorry

end NUMINAMATH_CALUDE_sector_area_l1399_139988


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l1399_139997

theorem necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, (a > 2 ∧ b > 2) → a + b > 4) ∧
  (∃ a b : ℝ, a + b > 4 ∧ ¬(a > 2 ∧ b > 2)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l1399_139997


namespace NUMINAMATH_CALUDE_mean_equality_implies_x_equals_8_l1399_139962

theorem mean_equality_implies_x_equals_8 :
  let mean1 := (8 + 10 + 24) / 3
  let mean2 := (16 + x + 18) / 3
  mean1 = mean2 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_x_equals_8_l1399_139962


namespace NUMINAMATH_CALUDE_octahedron_side_length_l1399_139936

/-- A unit cube in 3D space -/
structure UnitCube where
  A : ℝ × ℝ × ℝ := (0, 0, 0)
  A' : ℝ × ℝ × ℝ := (1, 1, 1)

/-- A regular octahedron inscribed in a unit cube -/
structure InscribedOctahedron (cube : UnitCube) where
  vertices : List (ℝ × ℝ × ℝ)

/-- The side length of an inscribed octahedron -/
def sideLength (octahedron : InscribedOctahedron cube) : ℝ :=
  sorry

/-- Theorem: The side length of the inscribed octahedron is √2/3 -/
theorem octahedron_side_length (cube : UnitCube) 
  (octahedron : InscribedOctahedron cube) : 
  sideLength octahedron = Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_side_length_l1399_139936


namespace NUMINAMATH_CALUDE_jimin_calculation_l1399_139965

theorem jimin_calculation (x : ℤ) : x + 20 = 60 → 34 - x = -6 := by
  sorry

end NUMINAMATH_CALUDE_jimin_calculation_l1399_139965


namespace NUMINAMATH_CALUDE_money_sharing_problem_l1399_139978

theorem money_sharing_problem (total : ℕ) (amanda ben carlos : ℕ) : 
  amanda + ben + carlos = total →
  amanda = 2 * (total / 13) →
  ben = 3 * (total / 13) →
  carlos = 8 * (total / 13) →
  ben = 30 →
  total = 130 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_problem_l1399_139978


namespace NUMINAMATH_CALUDE_smallest_with_eight_odd_ten_even_divisors_l1399_139911

/-- A function that returns the number of positive odd integer divisors of a given natural number. -/
def num_odd_divisors (n : ℕ) : ℕ := sorry

/-- A function that returns the number of positive even integer divisors of a given natural number. -/
def num_even_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating that 53760 is the smallest positive integer with 8 odd divisors and 10 even divisors. -/
theorem smallest_with_eight_odd_ten_even_divisors :
  (∀ m : ℕ, m > 0 ∧ m < 53760 → num_odd_divisors m ≠ 8 ∨ num_even_divisors m ≠ 10) ∧
  num_odd_divisors 53760 = 8 ∧
  num_even_divisors 53760 = 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_with_eight_odd_ten_even_divisors_l1399_139911


namespace NUMINAMATH_CALUDE_min_real_roots_l1399_139919

/-- A polynomial of degree 12 with real coefficients -/
def RealPolynomial12 : Type := { p : Polynomial ℝ // p.degree = 12 }

/-- The roots of a polynomial -/
def roots (p : RealPolynomial12) : Finset ℂ := sorry

/-- The number of distinct absolute values among the roots -/
def distinctAbsValues (p : RealPolynomial12) : ℕ := sorry

/-- The number of real roots of a polynomial -/
def realRootCount (p : RealPolynomial12) : ℕ := sorry

/-- The theorem stating the minimum number of real roots -/
theorem min_real_roots (p : RealPolynomial12) 
  (h : distinctAbsValues p = 6) : 
  ∃ q : RealPolynomial12, realRootCount q = 1 ∧ 
    ∀ r : RealPolynomial12, realRootCount r ≥ 1 := by sorry

end NUMINAMATH_CALUDE_min_real_roots_l1399_139919


namespace NUMINAMATH_CALUDE_total_students_correct_l1399_139933

/-- The total number of students in Misha's grade -/
def total_students : ℕ := 69

/-- Misha's position from the top -/
def position_from_top : ℕ := 30

/-- Misha's position from the bottom -/
def position_from_bottom : ℕ := 40

/-- Theorem stating that the total number of students is correct given Misha's positions -/
theorem total_students_correct :
  total_students = position_from_top + position_from_bottom - 1 :=
by sorry

end NUMINAMATH_CALUDE_total_students_correct_l1399_139933


namespace NUMINAMATH_CALUDE_walkers_on_same_side_l1399_139963

/-- Represents a person walking around a regular pentagon -/
structure Walker where
  speed : ℝ
  startPosition : ℕ

/-- The time when two walkers start walking on the same side of a regular pentagon -/
def timeOnSameSide (perimeterLength : ℝ) (walker1 walker2 : Walker) : ℝ :=
  sorry

/-- Theorem stating the time when two specific walkers start on the same side of a regular pentagon -/
theorem walkers_on_same_side :
  let perimeterLength : ℝ := 2000
  let walker1 : Walker := { speed := 50, startPosition := 0 }
  let walker2 : Walker := { speed := 46, startPosition := 2 }
  timeOnSameSide perimeterLength walker1 walker2 = 104 := by
  sorry

end NUMINAMATH_CALUDE_walkers_on_same_side_l1399_139963


namespace NUMINAMATH_CALUDE_polynomial_sum_l1399_139990

/-- Given polynomial f -/
def f (a b c : ℤ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Given polynomial g -/
def g (a b c : ℤ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + a

/-- The main theorem -/
theorem polynomial_sum (a b c : ℤ) : c ≠ 0 →
  f a b c 1 = 0 →
  (∀ x : ℝ, g a b c x = 0 ↔ ∃ y : ℝ, f a b c y = 0 ∧ x = y^2) →
  a^2013 + b^2013 + c^2013 = -1 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_sum_l1399_139990


namespace NUMINAMATH_CALUDE_min_value_z_l1399_139967

theorem min_value_z (x y : ℝ) :
  let z := 3 * x^2 + 4 * y^2 + 8 * x - 6 * y + 30
  ∀ a b : ℝ, z ≥ 3 * a^2 + 4 * b^2 + 8 * a - 6 * b + 30 → z ≥ 24.1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_z_l1399_139967


namespace NUMINAMATH_CALUDE_digit_721_of_3_over_11_l1399_139973

theorem digit_721_of_3_over_11 : ∃ (d : ℕ), d = 2 ∧ 
  (∃ (seq : ℕ → ℕ), 
    (∀ n, seq n < 10) ∧ 
    (∀ n, (3 * 10^(n+1)) % 11 = seq n) ∧
    seq 720 = d) := by
  sorry

end NUMINAMATH_CALUDE_digit_721_of_3_over_11_l1399_139973


namespace NUMINAMATH_CALUDE_overlapping_sectors_area_l1399_139955

/-- The area of the overlapping region of two 45° sectors in a circle with radius 15 -/
theorem overlapping_sectors_area (r : ℝ) (angle : ℝ) : 
  r = 15 → angle = 45 → 
  2 * (angle / 360 * π * r^2 - 1/2 * r^2 * Real.sin (angle * π / 180)) = 225/4 * (π - 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_overlapping_sectors_area_l1399_139955


namespace NUMINAMATH_CALUDE_ninth_day_practice_correct_l1399_139947

/-- The number of minutes Jenna practices piano on the 9th day to achieve
    an average of 100 minutes per day over a 9-day period, given her
    practice times for the first 8 days. -/
def ninth_day_practice (days_type1 days_type2 : ℕ) 
                       (minutes_type1 minutes_type2 : ℕ) : ℕ :=
  let total_days := days_type1 + days_type2 + 1
  let target_total := total_days * 100
  let current_total := days_type1 * minutes_type1 + days_type2 * minutes_type2
  target_total - current_total

theorem ninth_day_practice_correct :
  ninth_day_practice 6 2 80 105 = 210 :=
by sorry

end NUMINAMATH_CALUDE_ninth_day_practice_correct_l1399_139947


namespace NUMINAMATH_CALUDE_angle_bisector_intersection_ratio_l1399_139991

/-- Given a triangle PQR with points M on PQ and N on PR such that
    PM:MQ = 2:6 and PN:NR = 3:9, if PS is the angle bisector of angle P
    intersecting MN at L, then PL:PS = 1:4 -/
theorem angle_bisector_intersection_ratio (P Q R M N S L : EuclideanSpace ℝ (Fin 2)) :
  (∃ t : ℝ, M = (1 - t) • P + t • Q ∧ 2 * t = 6 * (1 - t)) →
  (∃ u : ℝ, N = (1 - u) • P + u • R ∧ 3 * u = 9 * (1 - u)) →
  (∃ v : ℝ, S = (1 - v) • P + v • Q ∧ 
            ∃ w : ℝ, S = (1 - w) • P + w • R ∧
            v / (1 - v) = w / (1 - w)) →
  (∃ k : ℝ, L = (1 - k) • M + k • N) →
  (∃ r : ℝ, L = (1 - r) • P + r • S ∧ r = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_angle_bisector_intersection_ratio_l1399_139991


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l1399_139927

theorem more_girls_than_boys (total_students : ℕ) 
  (h_total : total_students = 42)
  (h_ratio : ∃ (x : ℕ), 3 * x + 4 * x = total_students) : 
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    4 * boys = 3 * girls ∧ 
    girls - boys = 6 := by
sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l1399_139927


namespace NUMINAMATH_CALUDE_restaurant_tax_calculation_l1399_139976

/-- Proves that the tax amount is $3 given the initial money, order costs, and change received --/
theorem restaurant_tax_calculation (lee_money : ℕ) (friend_money : ℕ) 
  (wings_cost : ℕ) (salad_cost : ℕ) (soda_cost : ℕ) (change : ℕ) : ℕ :=
by
  -- Define the given conditions
  have h1 : lee_money = 10 := by sorry
  have h2 : friend_money = 8 := by sorry
  have h3 : wings_cost = 6 := by sorry
  have h4 : salad_cost = 4 := by sorry
  have h5 : soda_cost = 1 := by sorry
  have h6 : change = 3 := by sorry

  -- Calculate the total initial money
  let total_money := lee_money + friend_money

  -- Calculate the cost before tax
  let cost_before_tax := wings_cost + salad_cost + 2 * soda_cost

  -- Calculate the total spent including tax
  let total_spent := total_money - change

  -- Calculate the tax
  let tax := total_spent - cost_before_tax

  -- Prove that the tax is 3
  exact 3

end NUMINAMATH_CALUDE_restaurant_tax_calculation_l1399_139976


namespace NUMINAMATH_CALUDE_orange_picking_fraction_l1399_139993

/-- Proves that the fraction of oranges picked from each tree is 2/5 --/
theorem orange_picking_fraction
  (num_trees : ℕ)
  (fruits_per_tree : ℕ)
  (remaining_fruits : ℕ)
  (h1 : num_trees = 8)
  (h2 : fruits_per_tree = 200)
  (h3 : remaining_fruits = 960)
  (h4 : remaining_fruits < num_trees * fruits_per_tree) :
  (num_trees * fruits_per_tree - remaining_fruits) / (num_trees * fruits_per_tree) = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_orange_picking_fraction_l1399_139993


namespace NUMINAMATH_CALUDE_symmetry_of_point_l1399_139914

/-- Given a point P in 3D space, this function returns the point symmetric to P with respect to the y-axis --/
def symmetry_y_axis (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := P
  (-x, y, -z)

/-- Theorem stating that the point symmetric to (2, -3, -5) with respect to the y-axis is (-2, -3, 5) --/
theorem symmetry_of_point :
  symmetry_y_axis (2, -3, -5) = (-2, -3, 5) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l1399_139914


namespace NUMINAMATH_CALUDE_circle_area_from_polar_equation_l1399_139981

theorem circle_area_from_polar_equation :
  ∀ (r : ℝ → ℝ) (θ : ℝ),
    (r θ = 3 * Real.cos θ - 4 * Real.sin θ) →
    (∃ (c : ℝ × ℝ) (R : ℝ), ∀ (x y : ℝ),
      (x - c.1)^2 + (y - c.2)^2 = R^2 ↔ 
      ∃ θ, x = r θ * Real.cos θ ∧ y = r θ * Real.sin θ) →
    (π * (5/2)^2 = 25*π/4) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_from_polar_equation_l1399_139981


namespace NUMINAMATH_CALUDE_equation_solution_polynomial_expansion_l1399_139909

-- Part 1: Equation solution
theorem equation_solution :
  {x : ℝ | 9 * (x - 3)^2 - 121 = 0} = {20/3, -2/3} := by sorry

-- Part 2: Polynomial expansion
theorem polynomial_expansion (x y : ℝ) :
  (x - 2*y) * (x^2 + 2*x*y + 4*y^2) = x^3 - 8*y^3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_polynomial_expansion_l1399_139909


namespace NUMINAMATH_CALUDE_complex_equation_proof_l1399_139921

theorem complex_equation_proof (a b : ℝ) : (-2 * I + 1 : ℂ) = a + b * I → a - b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l1399_139921


namespace NUMINAMATH_CALUDE_farm_chicken_count_l1399_139977

theorem farm_chicken_count :
  ∀ (num_hens num_roosters : ℕ),
    num_hens = 52 →
    num_roosters = num_hens + 16 →
    num_hens + num_roosters = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_chicken_count_l1399_139977


namespace NUMINAMATH_CALUDE_f_properties_l1399_139983

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + Real.log (x + 1) - a * x - Real.cos x

theorem f_properties (a : ℝ) :
  (∀ x > -1, a ≤ 1 → Monotone (f a)) ∧
  (∃ a, deriv (f a) 0 = 0) := by sorry

end NUMINAMATH_CALUDE_f_properties_l1399_139983


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l1399_139937

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop := x^2 + 25*y + 19*z = -471
def equation2 (x y z : ℝ) : Prop := y^2 + 23*x + 21*z = -397
def equation3 (x y z : ℝ) : Prop := z^2 + 21*x + 21*y = -545

-- Theorem statement
theorem solution_satisfies_system :
  equation1 (-22) (-23) (-20) ∧
  equation2 (-22) (-23) (-20) ∧
  equation3 (-22) (-23) (-20) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l1399_139937


namespace NUMINAMATH_CALUDE_shoulder_width_conversion_l1399_139954

/-- Converts centimeters to millimeters -/
def cm_to_mm (cm : ℝ) : ℝ := cm * 10

theorem shoulder_width_conversion :
  let cm_per_m : ℝ := 100
  let mm_per_m : ℝ := 1000
  let shoulder_width_cm : ℝ := 45
  cm_to_mm shoulder_width_cm = 450 := by
  sorry

end NUMINAMATH_CALUDE_shoulder_width_conversion_l1399_139954


namespace NUMINAMATH_CALUDE_ice_cream_earnings_theorem_l1399_139989

def ice_cream_earnings (daily_increase : ℕ) : List ℕ :=
  [10, 10 + daily_increase, 10 + 2 * daily_increase, 10 + 3 * daily_increase, 10 + 4 * daily_increase]

theorem ice_cream_earnings_theorem (daily_increase : ℕ) :
  (List.sum (ice_cream_earnings daily_increase) = 90) → daily_increase = 4 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_earnings_theorem_l1399_139989


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1399_139994

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + y ≥ 16 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 1 ∧ x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1399_139994


namespace NUMINAMATH_CALUDE_plant_supplier_pots_cost_l1399_139992

/-- The cost of new pots for a plant supplier --/
theorem plant_supplier_pots_cost :
  let orchid_count : ℕ := 20
  let orchid_price : ℕ := 50
  let money_plant_count : ℕ := 15
  let money_plant_price : ℕ := 25
  let worker_count : ℕ := 2
  let worker_pay : ℕ := 40
  let remaining_money : ℕ := 1145
  let total_earnings := orchid_count * orchid_price + money_plant_count * money_plant_price
  let total_expenses := worker_count * worker_pay + remaining_money
  total_earnings - total_expenses = 150 :=
by sorry

end NUMINAMATH_CALUDE_plant_supplier_pots_cost_l1399_139992


namespace NUMINAMATH_CALUDE_power_product_eq_four_l1399_139924

theorem power_product_eq_four (a b : ℕ+) (h : (3 ^ a.val) ^ b.val = 3 ^ 3) :
  3 ^ a.val * 3 ^ b.val = 3 ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_eq_four_l1399_139924


namespace NUMINAMATH_CALUDE_multiply_to_325027405_l1399_139922

theorem multiply_to_325027405 (m : ℕ) : m * 32519 = 325027405 → m = 9995 := by
  sorry

end NUMINAMATH_CALUDE_multiply_to_325027405_l1399_139922


namespace NUMINAMATH_CALUDE_math_club_composition_l1399_139941

theorem math_club_composition :
  ∀ (initial_males initial_females : ℕ),
    initial_males = initial_females →
    (3 * (initial_males + initial_females - 1) = 4 * (initial_females - 1)) →
    initial_males = 2 ∧ initial_females = 3 := by
  sorry

end NUMINAMATH_CALUDE_math_club_composition_l1399_139941
