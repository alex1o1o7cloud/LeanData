import Mathlib

namespace NUMINAMATH_CALUDE_num_perfect_square_factors_is_440_l2948_294822

/-- The number of positive perfect square factors of (2^14)(3^9)(5^20) -/
def num_perfect_square_factors : ℕ :=
  (Finset.range 8).card * (Finset.range 5).card * (Finset.range 11).card

/-- Theorem stating that the number of positive perfect square factors of (2^14)(3^9)(5^20) is 440 -/
theorem num_perfect_square_factors_is_440 : num_perfect_square_factors = 440 := by
  sorry

end NUMINAMATH_CALUDE_num_perfect_square_factors_is_440_l2948_294822


namespace NUMINAMATH_CALUDE_ellipse_intersection_dot_product_l2948_294873

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the foci of the ellipse
def focus_1 : ℝ × ℝ := (1, 0)
def focus_2 : ℝ × ℝ := (-1, 0)

-- Define a line passing through a focus at 45°
def line_through_focus (f : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - f.2 = (x - f.1)

-- Define the intersection points
def intersection_points (f : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | is_on_ellipse p.1 p.2 ∧ line_through_focus f p.1 p.2}

-- Theorem statement
theorem ellipse_intersection_dot_product :
  ∀ (f : ℝ × ℝ) (A B : ℝ × ℝ),
    (f = focus_1 ∨ f = focus_2) →
    A ∈ intersection_points f →
    B ∈ intersection_points f →
    A ≠ B →
    A.1 * B.1 + A.2 * B.2 = -1/3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_dot_product_l2948_294873


namespace NUMINAMATH_CALUDE_max_perimeter_right_triangle_l2948_294805

theorem max_perimeter_right_triangle (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^2 + b^2 = 36) : 
  a + b + 6 ≤ 6 * Real.sqrt 2 + 6 :=
sorry

end NUMINAMATH_CALUDE_max_perimeter_right_triangle_l2948_294805


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l2948_294851

theorem tangent_line_intersection (f : ℝ → ℝ) (x₀ y₀ : ℝ) :
  f x₀ = y₀ →
  (∀ x, f x = x^3 + 11) →
  x₀ = 1 →
  y₀ = 12 →
  ∃ m : ℝ, ∀ x y, y - y₀ = m * (x - x₀) →
    y = 0 →
    x = -3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l2948_294851


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2948_294861

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 1 + a 5 + a 9 = 6) : 
  a 5 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2948_294861


namespace NUMINAMATH_CALUDE_andrew_eggs_l2948_294847

/-- The number of eggs Andrew ends up with after buying more -/
def total_eggs (initial : ℕ) (bought : ℕ) : ℕ := initial + bought

/-- Theorem: Andrew ends up with 70 eggs when starting with 8 and buying 62 more -/
theorem andrew_eggs : total_eggs 8 62 = 70 := by
  sorry

end NUMINAMATH_CALUDE_andrew_eggs_l2948_294847


namespace NUMINAMATH_CALUDE_ball_probability_l2948_294800

theorem ball_probability (total : ℕ) (red : ℕ) (purple : ℕ) 
  (h1 : total = 100) 
  (h2 : red = 9) 
  (h3 : purple = 3) : 
  (total - (red + purple)) / total = 88 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l2948_294800


namespace NUMINAMATH_CALUDE_remainder_2022_power_mod_11_l2948_294817

theorem remainder_2022_power_mod_11 : 2022^(2022^2022) ≡ 5 [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_remainder_2022_power_mod_11_l2948_294817


namespace NUMINAMATH_CALUDE_cory_fruit_orders_l2948_294880

def number_of_orders (apples oranges lemons : ℕ) : ℕ :=
  Nat.factorial (apples + oranges + lemons) / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial lemons)

theorem cory_fruit_orders :
  number_of_orders 4 2 1 = 105 := by
  sorry

end NUMINAMATH_CALUDE_cory_fruit_orders_l2948_294880


namespace NUMINAMATH_CALUDE_palic_function_is_quadratic_l2948_294808

-- Define the Palić function
def PalicFunction (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  Continuous f ∧
  ∀ x y z : ℝ, f x + f y + f z = f (a*x + b*y + c*z) + f (b*x + c*y + a*z) + f (c*x + a*y + b*z)

-- Define the theorem
theorem palic_function_is_quadratic 
  (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 1) 
  (h3 : a^3 + b^3 + c^3 ≠ 1) 
  (f : ℝ → ℝ) 
  (hf : PalicFunction f a b c) : 
  ∃ A B C : ℝ, ∀ x : ℝ, f x = A * x^2 + B * x + C := by
sorry

end NUMINAMATH_CALUDE_palic_function_is_quadratic_l2948_294808


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2948_294845

theorem shaded_area_calculation (carpet_side : ℝ) (large_square_side : ℝ) (small_square_side : ℝ) :
  carpet_side = 12 →
  carpet_side / large_square_side = 2 →
  large_square_side / small_square_side = 2 →
  12 * (small_square_side ^ 2) + large_square_side ^ 2 = 144 := by
  sorry

#check shaded_area_calculation

end NUMINAMATH_CALUDE_shaded_area_calculation_l2948_294845


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l2948_294881

theorem quadratic_solution_property : ∀ d e : ℝ,
  (4 * d^2 + 8 * d - 48 = 0) →
  (4 * e^2 + 8 * e - 48 = 0) →
  d ≠ e →
  (d - e)^2 + 4 = 68 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l2948_294881


namespace NUMINAMATH_CALUDE_paper_I_passing_percentage_l2948_294818

/-- Calculates the passing percentage for an exam given the maximum marks,
    the marks secured by a candidate, and the marks by which they failed. -/
def calculate_passing_percentage (max_marks : ℕ) (secured_marks : ℕ) (failed_by : ℕ) : ℚ :=
  let passing_marks : ℕ := secured_marks + failed_by
  (passing_marks : ℚ) / max_marks * 100

/-- Theorem stating that the passing percentage for Paper I is 40% -/
theorem paper_I_passing_percentage :
  calculate_passing_percentage 150 40 20 = 40 := by
sorry

end NUMINAMATH_CALUDE_paper_I_passing_percentage_l2948_294818


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2948_294865

theorem constant_term_expansion : 
  let p₁ : Polynomial ℤ := X^4 + 2*X^2 + 7
  let p₂ : Polynomial ℤ := 2*X^5 + 3*X^3 + 25
  (p₁ * p₂).coeff 0 = 175 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2948_294865


namespace NUMINAMATH_CALUDE_triangle_area_l2948_294863

/-- A triangle with integral sides and perimeter 12 has area 6 -/
theorem triangle_area (a b c : ℕ) : 
  a + b + c = 12 → 
  a + b > c → b + c > a → a + c > b → 
  (a : ℝ) * (b : ℝ) / 2 = 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_l2948_294863


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2948_294823

theorem inequality_solution_set (x : ℝ) :
  (x + 3) * (x - 2) < 0 ↔ -3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2948_294823


namespace NUMINAMATH_CALUDE_inequality_proof_l2948_294895

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h_sum : x + y + z = 3 * Real.sqrt 3) :
  (x^2 / (x + 2*y + 3*z)) + (y^2 / (y + 2*z + 3*x)) + (z^2 / (z + 2*x + 3*y)) ≥ Real.sqrt 3 / 2 := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l2948_294895


namespace NUMINAMATH_CALUDE_complex_reciprocal_sum_magnitude_l2948_294843

theorem complex_reciprocal_sum_magnitude (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 :=
by sorry

end NUMINAMATH_CALUDE_complex_reciprocal_sum_magnitude_l2948_294843


namespace NUMINAMATH_CALUDE_star_value_proof_l2948_294885

theorem star_value_proof (star : ℝ) : 
  45 - (28 - (37 - (15 - star^2))) = 59 → star = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_star_value_proof_l2948_294885


namespace NUMINAMATH_CALUDE_ant_movement_theorem_l2948_294886

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the movement of an ant --/
structure AntMovement where
  seconds : ℕ
  unitPerSecond : ℝ

/-- Calculates the expected area of the convex quadrilateral formed by ants --/
def expectedArea (rect : Rectangle) (movement : AntMovement) : ℝ :=
  (rect.length - 2 * movement.seconds * movement.unitPerSecond) *
  (rect.width - 2 * movement.seconds * movement.unitPerSecond)

/-- Theorem statement for the ant movement problem --/
theorem ant_movement_theorem (rect : Rectangle) (movement : AntMovement) :
  rect.length = 20 ∧ rect.width = 23 ∧ movement.seconds = 10 ∧ movement.unitPerSecond = 0.5 →
  expectedArea rect movement = 130 := by
  sorry


end NUMINAMATH_CALUDE_ant_movement_theorem_l2948_294886


namespace NUMINAMATH_CALUDE_ellipse_condition_l2948_294804

/-- The equation of the curve -/
def curve_equation (x y c : ℝ) : Prop :=
  9 * x^2 + y^2 + 54 * x - 8 * y = c

/-- Definition of a non-degenerate ellipse -/
def is_non_degenerate_ellipse (c : ℝ) : Prop :=
  ∃ a b h k : ℝ, a > 0 ∧ b > 0 ∧
  ∀ x y : ℝ, curve_equation x y c ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- Theorem: The curve is a non-degenerate ellipse if and only if c > -97 -/
theorem ellipse_condition (c : ℝ) :
  is_non_degenerate_ellipse c ↔ c > -97 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2948_294804


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l2948_294844

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (10000 ≤ n ∧ n ≤ 99999) ∧ 
  (∃ a : ℕ, n = a^2) ∧
  (∃ b : ℕ, n = b^3) ∧
  (∀ m : ℕ, (10000 ≤ m ∧ m < n) → ¬(∃ x : ℕ, m = x^2) ∨ ¬(∃ y : ℕ, m = y^3)) ∧
  n = 15625 := by
sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l2948_294844


namespace NUMINAMATH_CALUDE_paul_shopping_money_left_l2948_294858

theorem paul_shopping_money_left 
  (initial_money : ℝ)
  (bread_price : ℝ)
  (butter_original_price : ℝ)
  (butter_discount : ℝ)
  (juice_price_multiplier : ℝ)
  (sales_tax_rate : ℝ)
  (h1 : initial_money = 15)
  (h2 : bread_price = 2)
  (h3 : butter_original_price = 3)
  (h4 : butter_discount = 0.1)
  (h5 : juice_price_multiplier = 2)
  (h6 : sales_tax_rate = 0.05) :
  initial_money - 
  ((bread_price + 
    (butter_original_price * (1 - butter_discount)) + 
    (bread_price * juice_price_multiplier)) * 
   (1 + sales_tax_rate)) = 5.86 := by
sorry

end NUMINAMATH_CALUDE_paul_shopping_money_left_l2948_294858


namespace NUMINAMATH_CALUDE_surveyed_not_population_l2948_294897

/-- Represents the total number of students in the seventh grade. -/
def total_students : ℕ := 800

/-- Represents the number of students surveyed. -/
def surveyed_students : ℕ := 200

/-- Represents whether a given number of students constitutes the entire population. -/
def is_population (n : ℕ) : Prop := n = total_students

/-- Theorem stating that the surveyed students do not constitute the entire population. -/
theorem surveyed_not_population : ¬(is_population surveyed_students) := by
  sorry

end NUMINAMATH_CALUDE_surveyed_not_population_l2948_294897


namespace NUMINAMATH_CALUDE_square_difference_plus_six_b_l2948_294884

theorem square_difference_plus_six_b (a b : ℝ) (h : a + b = 3) : 
  a^2 - b^2 + 6*b = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_plus_six_b_l2948_294884


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2948_294835

/-- Theorem: Train Speed Calculation
Given a train of length 120 meters crossing a bridge of length 240 meters in 3 minutes,
prove that the speed of the train is 2 m/s. -/
theorem train_speed_calculation (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  bridge_length = 240 →
  crossing_time = 3 * 60 →
  (train_length + bridge_length) / crossing_time = 2 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l2948_294835


namespace NUMINAMATH_CALUDE_function_inequality_l2948_294864

/-- Given f(x) = e^(2x) - ax, for all x > 0, if f(x) > ax^2 + 1, then a ≤ 2 -/
theorem function_inequality (a : ℝ) : 
  (∀ x > 0, Real.exp (2 * x) - a * x > a * x^2 + 1) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2948_294864


namespace NUMINAMATH_CALUDE_school_teachers_count_l2948_294819

theorem school_teachers_count (total : ℕ) (sample_size : ℕ) (sample_students : ℕ) : 
  total = 2400 →
  sample_size = 320 →
  sample_students = 280 →
  ∃ (teachers students : ℕ),
    teachers + students = total ∧
    teachers * sample_students = students * (sample_size - sample_students) ∧
    teachers = 300 := by
  sorry

end NUMINAMATH_CALUDE_school_teachers_count_l2948_294819


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2948_294806

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h_geometric : is_geometric_sequence a) 
  (h_sum : a 2013 + a 2015 = ∫ x in (0:ℝ)..2, Real.sqrt (4 - x^2)) :
  a 2014 * (a 2012 + 2 * a 2014 + a 2016) = π^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2948_294806


namespace NUMINAMATH_CALUDE_earnings_ratio_l2948_294821

theorem earnings_ratio (total_earnings lottie_earnings jerusha_earnings : ℕ)
  (h1 : total_earnings = 85)
  (h2 : jerusha_earnings = 68)
  (h3 : total_earnings = lottie_earnings + jerusha_earnings)
  (h4 : ∃ k : ℕ, jerusha_earnings = k * lottie_earnings) :
  jerusha_earnings = 4 * lottie_earnings :=
by sorry

end NUMINAMATH_CALUDE_earnings_ratio_l2948_294821


namespace NUMINAMATH_CALUDE_two_zeros_implies_a_is_inverse_e_l2948_294867

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x + x else a*x - Real.log x

theorem two_zeros_implies_a_is_inverse_e (a : ℝ) (h_a_pos : a > 0) :
  (∃! x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  a = Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_two_zeros_implies_a_is_inverse_e_l2948_294867


namespace NUMINAMATH_CALUDE_parallelogram_area_l2948_294853

/-- A parallelogram with base 10 and altitude twice the base has area 200. -/
theorem parallelogram_area (base : ℝ) (altitude : ℝ) : 
  base = 10 → altitude = 2 * base → base * altitude = 200 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2948_294853


namespace NUMINAMATH_CALUDE_triangle_parallelogram_altitude_relation_l2948_294879

theorem triangle_parallelogram_altitude_relation 
  (base : ℝ) 
  (triangle_area parallelogram_area : ℝ) 
  (triangle_altitude parallelogram_altitude : ℝ) 
  (h1 : triangle_area = parallelogram_area) 
  (h2 : parallelogram_altitude = 100) 
  (h3 : triangle_area = 1/2 * base * triangle_altitude) 
  (h4 : parallelogram_area = base * parallelogram_altitude) : 
  triangle_altitude = 200 := by
sorry

end NUMINAMATH_CALUDE_triangle_parallelogram_altitude_relation_l2948_294879


namespace NUMINAMATH_CALUDE_color_fractions_l2948_294828

-- Define the color type
inductive Color
  | Red
  | Blue

-- Define the coloring function
def color : ℚ → Color := sorry

-- Define the coloring rules
axiom color_one : color 1 = Color.Red
axiom color_diff_one (x : ℚ) : color (x + 1) ≠ color x
axiom color_reciprocal (x : ℚ) (h : x ≠ 1) : color (1 / x) ≠ color x

-- State the theorem
theorem color_fractions :
  color (2013 / 2014) = Color.Red ∧ color (2 / 7) = Color.Blue :=
sorry

end NUMINAMATH_CALUDE_color_fractions_l2948_294828


namespace NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l2948_294834

/-- The area of a right triangle with sides of length 8 and 3 is 12 -/
theorem right_triangle_area : ℝ → ℝ → ℝ → Prop :=
  fun side1 side2 area =>
    side1 = 8 ∧ side2 = 3 ∧ area = (1 / 2) * side1 * side2 → area = 12

/-- Proof of the theorem -/
theorem right_triangle_area_proof : right_triangle_area 8 3 12 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l2948_294834


namespace NUMINAMATH_CALUDE_lunch_with_tip_l2948_294878

/-- Calculate the total amount spent on lunch including tip -/
theorem lunch_with_tip (lunch_cost : ℝ) (tip_percentage : ℝ) :
  lunch_cost = 50.20 →
  tip_percentage = 20 →
  lunch_cost * (1 + tip_percentage / 100) = 60.24 := by
  sorry

end NUMINAMATH_CALUDE_lunch_with_tip_l2948_294878


namespace NUMINAMATH_CALUDE_side_length_equation_l2948_294888

/-- Rectangle ABCD with equilateral triangles AEF and XYZ -/
structure SpecialRectangle where
  /-- Length of rectangle ABCD -/
  length : ℝ
  /-- Width of rectangle ABCD -/
  width : ℝ
  /-- Point E on BC such that BE = EC -/
  E : ℝ × ℝ
  /-- Point F on CD -/
  F : ℝ × ℝ
  /-- Side length of equilateral triangle XYZ -/
  s : ℝ
  /-- Rectangle ABCD has length 2 and width 1 -/
  length_eq : length = 2
  /-- Rectangle ABCD has length 2 and width 1 -/
  width_eq : width = 1
  /-- BE = EC = 1 -/
  BE_eq_EC : E.1 = 1
  /-- Angle AEF is 60 degrees -/
  angle_AEF : Real.cos (60 * π / 180) = 1 / 2
  /-- Triangle AEF is equilateral -/
  AEF_equilateral : (E.1 - 0)^2 + (E.2 - 0)^2 = (F.1 - E.1)^2 + (F.2 - E.2)^2
  /-- XY is parallel to AB -/
  XY_parallel_AB : s ≤ width

theorem side_length_equation (r : SpecialRectangle) :
  r.s^2 + 4 * r.s - 8 / Real.sqrt 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_side_length_equation_l2948_294888


namespace NUMINAMATH_CALUDE_no_three_digit_perfect_square_difference_l2948_294807

theorem no_three_digit_perfect_square_difference :
  ¬ ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
    ∃ (k : ℕ), (100 * a + 10 * b + c) - (100 * c + 10 * b + a) = k^2 :=
by sorry

end NUMINAMATH_CALUDE_no_three_digit_perfect_square_difference_l2948_294807


namespace NUMINAMATH_CALUDE_cream_cheese_amount_l2948_294831

/-- Calculates the amount of cream cheese used in a spinach quiche recipe. -/
theorem cream_cheese_amount
  (raw_spinach : ℝ)
  (cooked_spinach_percentage : ℝ)
  (eggs : ℝ)
  (total_volume : ℝ)
  (h1 : raw_spinach = 40)
  (h2 : cooked_spinach_percentage = 0.20)
  (h3 : eggs = 4)
  (h4 : total_volume = 18) :
  total_volume - (raw_spinach * cooked_spinach_percentage) - eggs = 6 := by
  sorry

end NUMINAMATH_CALUDE_cream_cheese_amount_l2948_294831


namespace NUMINAMATH_CALUDE_solution_in_interval_l2948_294868

open Real

/-- A monotonically increasing function on (0, +∞) satisfying f[f(x) - ln x] = 1 -/
def MonotonicFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 < x ∧ x < y → f x < f y) ∧
  (∀ x, 0 < x → f (f x - log x) = 1)

/-- The solution to f(x) - f'(x) = 1 lies in (1, 2) -/
theorem solution_in_interval (f : ℝ → ℝ) (hf : MonotonicFunction f) :
  ∃ x, 1 < x ∧ x < 2 ∧ f x - (deriv f) x = 1 :=
sorry

end NUMINAMATH_CALUDE_solution_in_interval_l2948_294868


namespace NUMINAMATH_CALUDE_log_properties_l2948_294820

-- Define the logarithm function for base b
noncomputable def log_b (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_properties (b : ℝ) (h : 0 < b ∧ b < 1) :
  (log_b b 1 = 0) ∧ 
  (log_b b b = 1) ∧ 
  (∀ x : ℝ, 1 < x → x < b → log_b b x > 0) ∧
  (∀ x y : ℝ, 1 < x → x < y → y < b → log_b b x > log_b b y) :=
by sorry

end NUMINAMATH_CALUDE_log_properties_l2948_294820


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2948_294840

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {1, 3, 5, 7}

theorem complement_of_A_in_U :
  (U \ A) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2948_294840


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_subset_l2948_294848

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b ^ 2 = a * c

theorem arithmetic_sequence_with_geometric_subset (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 = 1 →
  is_geometric_sequence (a 1) (a 3) (a 9) →
  (∀ n : ℕ, a n = n) ∨ (∀ n : ℕ, a n = 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_subset_l2948_294848


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l2948_294889

-- Define the points in the plane
variable (A B C D : ℝ × ℝ)

-- Define the distances
def AC : ℝ := 15
def AB : ℝ := 17
def DC : ℝ := 9

-- Define the angle D as a right angle
def angle_D_is_right : Prop := sorry

-- Define that the points are coplanar
def points_are_coplanar : Prop := sorry

-- Define the area of triangle ABC
def area_ABC : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_ABC :
  points_are_coplanar →
  angle_D_is_right →
  area_ABC = 54 + 6 * Real.sqrt 145 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l2948_294889


namespace NUMINAMATH_CALUDE_marble_bag_problem_l2948_294890

theorem marble_bag_problem (red blue : ℕ) (p : ℚ) (total : ℕ) : 
  red = 12 →
  blue = 8 →
  p = 81 / 256 →
  (((total - red : ℚ) / total) ^ 4 = p) →
  total = 48 :=
by sorry

end NUMINAMATH_CALUDE_marble_bag_problem_l2948_294890


namespace NUMINAMATH_CALUDE_expression_equality_l2948_294850

theorem expression_equality : 200 * (200 - 5) - (200 * 200 - 5) = -995 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2948_294850


namespace NUMINAMATH_CALUDE_y_relationship_l2948_294891

/-- The function f(x) = -x² + 5 -/
def f (x : ℝ) : ℝ := -x^2 + 5

/-- y₁ is the y-coordinate of the point (-4, y₁) on the graph of f -/
def y₁ : ℝ := f (-4)

/-- y₂ is the y-coordinate of the point (-1, y₂) on the graph of f -/
def y₂ : ℝ := f (-1)

/-- y₃ is the y-coordinate of the point (2, y₃) on the graph of f -/
def y₃ : ℝ := f 2

theorem y_relationship : y₂ > y₃ ∧ y₃ > y₁ := by sorry

end NUMINAMATH_CALUDE_y_relationship_l2948_294891


namespace NUMINAMATH_CALUDE_opposite_reciprocal_expression_zero_l2948_294871

theorem opposite_reciprocal_expression_zero
  (a b c d : ℝ)
  (h1 : a = -b)
  (h2 : c = 1 / d)
  : 2 * c - a - 2 / d - b = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_expression_zero_l2948_294871


namespace NUMINAMATH_CALUDE_pictures_per_album_l2948_294836

/-- Given pictures from a phone and camera, prove the number of pictures in each album when equally distributed. -/
theorem pictures_per_album 
  (phone_pics : ℕ) 
  (camera_pics : ℕ) 
  (num_albums : ℕ) 
  (h1 : phone_pics = 2) 
  (h2 : camera_pics = 4) 
  (h3 : num_albums = 3) 
  (h4 : num_albums > 0) : 
  (phone_pics + camera_pics) / num_albums = 2 := by
sorry

end NUMINAMATH_CALUDE_pictures_per_album_l2948_294836


namespace NUMINAMATH_CALUDE_second_tree_groups_count_l2948_294857

/-- Represents the number of rings in a group -/
def rings_per_group : ℕ := 6

/-- Represents the number of ring groups in the first tree -/
def first_tree_groups : ℕ := 70

/-- Represents the age difference between the first and second tree in years -/
def age_difference : ℕ := 180

/-- Calculates the number of ring groups in the second tree -/
def second_tree_groups : ℕ := 
  (first_tree_groups * rings_per_group - age_difference) / rings_per_group

theorem second_tree_groups_count : second_tree_groups = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_tree_groups_count_l2948_294857


namespace NUMINAMATH_CALUDE_cube_sum_problem_l2948_294839

theorem cube_sum_problem (x y : ℝ) 
  (h1 : 1/x + 1/y = 4)
  (h2 : x*y + x^2 + y^2 = 17) :
  x^3 + y^3 = 52 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l2948_294839


namespace NUMINAMATH_CALUDE_mayoral_election_votes_l2948_294866

theorem mayoral_election_votes (Z Y X : ℕ) : 
  Z = 25000 → 
  Y = Z - (2/5 : ℚ) * Z →
  X = Y + (1/2 : ℚ) * Y →
  X = 22500 := by
  sorry

end NUMINAMATH_CALUDE_mayoral_election_votes_l2948_294866


namespace NUMINAMATH_CALUDE_nancy_tortilla_chips_l2948_294870

/-- Nancy's tortilla chip distribution problem -/
theorem nancy_tortilla_chips : ∀ (initial brother sister : ℕ),
  initial = 22 →
  brother = 7 →
  sister = 5 →
  initial - (brother + sister) = 10 := by
  sorry

end NUMINAMATH_CALUDE_nancy_tortilla_chips_l2948_294870


namespace NUMINAMATH_CALUDE_binomial_coefficient_26_6_l2948_294854

theorem binomial_coefficient_26_6 (h1 : Nat.choose 24 5 = 42504) (h2 : Nat.choose 24 6 = 134596) :
  Nat.choose 26 6 = 230230 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_26_6_l2948_294854


namespace NUMINAMATH_CALUDE_melanie_dimes_given_l2948_294852

/-- The number of dimes Melanie gave to her dad -/
def dimes_given_to_dad : ℕ := 7

/-- The initial number of dimes Melanie had -/
def initial_dimes : ℕ := 8

/-- The number of dimes Melanie received from her mother -/
def dimes_from_mother : ℕ := 4

/-- The number of dimes Melanie has now -/
def current_dimes : ℕ := 5

theorem melanie_dimes_given :
  initial_dimes - dimes_given_to_dad + dimes_from_mother = current_dimes :=
by sorry

end NUMINAMATH_CALUDE_melanie_dimes_given_l2948_294852


namespace NUMINAMATH_CALUDE_symmetry_implies_phi_value_l2948_294825

/-- Given a function f and its translation g, proves that if g is symmetric about π/2, then φ = π/2 -/
theorem symmetry_implies_phi_value 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (φ : ℝ) 
  (h1 : 0 < φ ∧ φ < π)
  (h2 : ∀ x, f x = Real.cos (2 * x + φ))
  (h3 : ∀ x, g x = f (x - π/4))
  (h4 : ∀ x, g x = g (π - x)) : 
  φ = π/2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_phi_value_l2948_294825


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2948_294874

def solution_set : Set ℝ := {x | x ≤ -5/2}

def inequality (x : ℝ) : Prop := |x - 2| + |x + 3| ≥ 4

theorem solution_set_equivalence :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2948_294874


namespace NUMINAMATH_CALUDE_parlor_game_solution_l2948_294849

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h1 : a < 10
  h2 : b < 10
  h3 : c < 10
  h4 : a > 0

/-- Calculates the sum of permutations for a three-digit number -/
def sumOfPermutations (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.c + n.b +
  100 * n.a + 10 * n.b + n.c +
  100 * n.b + 10 * n.c + n.a +
  100 * n.b + 10 * n.a + n.c +
  100 * n.c + 10 * n.a + n.b +
  100 * n.c + 10 * n.b + n.a

/-- The main theorem -/
theorem parlor_game_solution :
  ∃ (n : ThreeDigitNumber), sumOfPermutations n = 4326 ∧ n.a = 3 ∧ n.b = 9 ∧ n.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_parlor_game_solution_l2948_294849


namespace NUMINAMATH_CALUDE_total_shells_count_l2948_294892

def morning_shells : ℕ := 292
def afternoon_shells : ℕ := 324

theorem total_shells_count : morning_shells + afternoon_shells = 616 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_count_l2948_294892


namespace NUMINAMATH_CALUDE_sum_of_squares_l2948_294869

theorem sum_of_squares (x y z : ℤ) 
  (sum_eq : x + y + z = 3) 
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2948_294869


namespace NUMINAMATH_CALUDE_least_subtrahend_proof_l2948_294841

/-- The product of the first four prime numbers -/
def product_of_first_four_primes : ℕ := 2 * 3 * 5 * 7

/-- The original number from which we subtract -/
def original_number : ℕ := 427751

/-- The least number to be subtracted -/
def least_subtrahend : ℕ := 91

theorem least_subtrahend_proof :
  (∀ k : ℕ, k < least_subtrahend → ¬((original_number - k) % product_of_first_four_primes = 0)) ∧
  ((original_number - least_subtrahend) % product_of_first_four_primes = 0) :=
sorry

end NUMINAMATH_CALUDE_least_subtrahend_proof_l2948_294841


namespace NUMINAMATH_CALUDE_green_ball_probability_l2948_294855

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The containers in the problem -/
def containerX : Container := ⟨5, 5⟩
def containerY : Container := ⟨7, 3⟩
def containerZ : Container := ⟨7, 3⟩

/-- The list of all containers -/
def containers : List Container := [containerX, containerY, containerZ]

/-- The probability of selecting each container -/
def containerProbability : ℚ := 1 / containers.length

/-- The theorem stating the probability of selecting a green ball -/
theorem green_ball_probability :
  (containers.map (fun c => containerProbability * greenProbability c)).sum = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l2948_294855


namespace NUMINAMATH_CALUDE_group_formation_count_l2948_294829

def total_people : ℕ := 7
def group_size_1 : ℕ := 3
def group_size_2 : ℕ := 4

theorem group_formation_count :
  Nat.choose total_people group_size_1 = 35 :=
by sorry

end NUMINAMATH_CALUDE_group_formation_count_l2948_294829


namespace NUMINAMATH_CALUDE_octal_sum_theorem_l2948_294802

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- Converts a decimal number to its octal representation as a list of digits -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- The main theorem stating that the sum of 642₈ and 157₈ in base 8 is 1021₈ -/
theorem octal_sum_theorem :
  decimal_to_octal (octal_to_decimal [6, 4, 2] + octal_to_decimal [1, 5, 7]) = [1, 0, 2, 1] :=
sorry

end NUMINAMATH_CALUDE_octal_sum_theorem_l2948_294802


namespace NUMINAMATH_CALUDE_different_color_probability_l2948_294893

/-- Given 6 cards with 3 red and 3 yellow, the probability of drawing 2 cards of different colors is 3/5 -/
theorem different_color_probability (total_cards : Nat) (red_cards : Nat) (yellow_cards : Nat) :
  total_cards = 6 →
  red_cards = 3 →
  yellow_cards = 3 →
  (Nat.choose red_cards 1 * Nat.choose yellow_cards 1 : Rat) / Nat.choose total_cards 2 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l2948_294893


namespace NUMINAMATH_CALUDE_art_students_count_l2948_294877

theorem art_students_count (total : ℕ) (music : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : music = 20)
  (h3 : both = 10)
  (h4 : neither = 470) :
  ∃ art : ℕ, art = 20 ∧ 
    total = (music - both) + (art - both) + both + neither :=
by sorry

end NUMINAMATH_CALUDE_art_students_count_l2948_294877


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2948_294856

theorem quadratic_one_solution (k : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + k = 0) ↔ k = 49/12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2948_294856


namespace NUMINAMATH_CALUDE_power_of_two_inequality_l2948_294860

theorem power_of_two_inequality (k l m : ℕ) :
  2^(k+1) + 2^(k+m) + 2^(l+m) ≤ 2^(k+l+m+1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_inequality_l2948_294860


namespace NUMINAMATH_CALUDE_election_result_l2948_294846

/-- Represents an election with three candidates -/
structure Election :=
  (total_votes : ℕ)
  (votes_A : ℕ)
  (votes_B : ℕ)
  (votes_C : ℕ)

/-- The election satisfies the given conditions -/
def valid_election (e : Election) : Prop :=
  e.votes_A = (35 * e.total_votes) / 100 ∧
  e.votes_C = (25 * e.total_votes) / 100 ∧
  e.votes_B = e.votes_A + 2460 ∧
  e.total_votes = e.votes_A + e.votes_B + e.votes_C

theorem election_result (e : Election) (h : valid_election e) :
  e.votes_B = (40 * e.total_votes) / 100 ∧ e.total_votes = 49200 := by
  sorry


end NUMINAMATH_CALUDE_election_result_l2948_294846


namespace NUMINAMATH_CALUDE_negation_of_all_not_divisible_by_two_are_odd_l2948_294898

theorem negation_of_all_not_divisible_by_two_are_odd :
  (¬ ∀ n : ℤ, ¬(2 ∣ n) → Odd n) ↔ (∃ n : ℤ, ¬(2 ∣ n) ∧ ¬(Odd n)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_not_divisible_by_two_are_odd_l2948_294898


namespace NUMINAMATH_CALUDE_range_of_m_l2948_294810

theorem range_of_m (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∨ 
  (∀ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 ≠ 0) →
  ¬(∃ x y : ℝ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  (∃ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 = 0) →
  (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2948_294810


namespace NUMINAMATH_CALUDE_factorial_34_representation_l2948_294815

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def decimal_rep (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
    aux n []

theorem factorial_34_representation (a b : ℕ) :
  decimal_rep (factorial 34) = [2, 9, 5, 2, 3, 2, 7, 9, 9, 0, 3, 9, a, 0, 4, 1, 4, 0, 8, 4, 7, 6, 1, 8, 6, 0, 9, 6, 4, 3, 5, b, 0, 0, 0, 0, 0, 0, 0] →
  a = 6 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_34_representation_l2948_294815


namespace NUMINAMATH_CALUDE_probability_independent_of_radius_constant_probability_l2948_294832

-- Define a circular dartboard
structure Dartboard where
  radius : ℝ
  radius_pos : radius > 0

-- Define the probability function
def probability_closer_to_center (d : Dartboard) : ℝ := 0.25

-- Theorem statement
theorem probability_independent_of_radius (d : Dartboard) :
  probability_closer_to_center d = 0.25 := by
  sorry

-- The distance from the thrower is not relevant to the probability,
-- but we include it to match the original problem description
def distance_from_thrower : ℝ := 20

-- Theorem stating that the probability is constant regardless of radius
theorem constant_probability (d1 d2 : Dartboard) :
  probability_closer_to_center d1 = probability_closer_to_center d2 := by
  sorry

end NUMINAMATH_CALUDE_probability_independent_of_radius_constant_probability_l2948_294832


namespace NUMINAMATH_CALUDE_solve_equation_l2948_294816

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 10
def g (x : ℝ) : ℝ := x^2 - 5

-- State the theorem
theorem solve_equation (a : ℝ) (ha : a > 0) (h : f (g a) = 18) :
  a = Real.sqrt (5 + 2 * Real.sqrt 2) ∨ a = Real.sqrt (5 - 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_solve_equation_l2948_294816


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_ratio_l2948_294887

theorem quadratic_roots_imply_ratio (a b : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x = -1/2 ∧ y = 1/3 ∧ a * x^2 + b * x + 2 = 0 ∧ a * y^2 + b * y + 2 = 0) →
  (a - b) / a = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_ratio_l2948_294887


namespace NUMINAMATH_CALUDE_a_investment_is_6300_l2948_294811

/-- Represents the investment and profit scenario of a partnership business -/
structure BusinessPartnership where
  /-- A's investment amount -/
  a_investment : ℝ
  /-- B's investment amount -/
  b_investment : ℝ
  /-- C's investment amount -/
  c_investment : ℝ
  /-- Total profit -/
  total_profit : ℝ
  /-- A's share of the profit -/
  a_profit : ℝ

/-- Theorem stating that given the conditions, A's investment is 6300 -/
theorem a_investment_is_6300 (bp : BusinessPartnership)
  (h1 : bp.b_investment = 4200)
  (h2 : bp.c_investment = 10500)
  (h3 : bp.total_profit = 12700)
  (h4 : bp.a_profit = 3810)
  (h5 : bp.a_profit / bp.total_profit = bp.a_investment / (bp.a_investment + bp.b_investment + bp.c_investment)) :
  bp.a_investment = 6300 := by
  sorry

end NUMINAMATH_CALUDE_a_investment_is_6300_l2948_294811


namespace NUMINAMATH_CALUDE_nested_expression_equals_4094_l2948_294827

def nested_expression : ℕ := 2*(1+2*(1+2*(1+2*(1+2*(1+2*(1+2*(1+2*(1+2*(1+2*(1+2))))))))))

theorem nested_expression_equals_4094 : nested_expression = 4094 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_equals_4094_l2948_294827


namespace NUMINAMATH_CALUDE_unread_pages_after_two_weeks_l2948_294838

theorem unread_pages_after_two_weeks (total_pages : ℕ) (pages_per_day : ℕ) (days : ℕ) (unread_pages : ℕ) : 
  total_pages = 200 →
  pages_per_day = 12 →
  days = 14 →
  unread_pages = total_pages - (pages_per_day * days) →
  unread_pages = 32 := by
sorry

end NUMINAMATH_CALUDE_unread_pages_after_two_weeks_l2948_294838


namespace NUMINAMATH_CALUDE_problem_solution_l2948_294814

open Real

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := b * log x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := log x + a * x^2 - x

theorem problem_solution :
  (∀ x > 0, Monotone (F (1/8))) ∧
  (∀ a ≥ 1/8, ∀ x > 0, Monotone (F a)) ∧
  (∃ b : ℝ, (b < -2 ∨ b > (ℯ^2 + 2)/(ℯ - 1)) ↔
    ∃ x₀ ∈ Set.Icc 1 ℯ, x₀ - f b x₀ < -(1 + b)/x₀) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2948_294814


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l2948_294833

theorem unique_modular_congruence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ -250 ≡ n [ZMOD 17] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l2948_294833


namespace NUMINAMATH_CALUDE_fraction_used_is_47_48_l2948_294813

/-- Represents the car's journey with given parameters -/
structure CarJourney where
  tankCapacity : ℚ
  firstLegDuration : ℚ
  firstLegSpeed : ℚ
  firstLegConsumptionRate : ℚ
  refillAmount : ℚ
  secondLegDuration : ℚ
  secondLegSpeed : ℚ
  secondLegConsumptionRate : ℚ

/-- Calculates the fraction of a full tank used after the entire journey -/
def fractionUsed (journey : CarJourney) : ℚ :=
  let firstLegDistance := journey.firstLegDuration * journey.firstLegSpeed
  let firstLegUsed := firstLegDistance / journey.firstLegConsumptionRate
  let secondLegDistance := journey.secondLegDuration * journey.secondLegSpeed
  let secondLegUsed := secondLegDistance / journey.secondLegConsumptionRate
  (firstLegUsed + secondLegUsed) / journey.tankCapacity

/-- The specific journey described in the problem -/
def specificJourney : CarJourney :=
  { tankCapacity := 12
  , firstLegDuration := 3
  , firstLegSpeed := 50
  , firstLegConsumptionRate := 40
  , refillAmount := 5
  , secondLegDuration := 4
  , secondLegSpeed := 60
  , secondLegConsumptionRate := 30
  }

/-- Theorem stating that the fraction of tank used in the specific journey is 47/48 -/
theorem fraction_used_is_47_48 : fractionUsed specificJourney = 47 / 48 := by
  sorry


end NUMINAMATH_CALUDE_fraction_used_is_47_48_l2948_294813


namespace NUMINAMATH_CALUDE_problem_solution_l2948_294837

theorem problem_solution : (0.15 : ℝ)^3 - (0.06 : ℝ)^3 / (0.15 : ℝ)^2 + 0.009 + (0.06 : ℝ)^2 = 0.006375 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2948_294837


namespace NUMINAMATH_CALUDE_find_M_l2948_294876

theorem find_M : ∃ M : ℕ+, (15^2 * 25^2 : ℕ) = 5^2 * M^2 ∧ M = 375 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l2948_294876


namespace NUMINAMATH_CALUDE_starting_number_proof_l2948_294830

theorem starting_number_proof : ∃ (n : ℕ), 
  n = 220 ∧ 
  n < 580 ∧ 
  (∃ (m : ℕ), m = 6 ∧ 
    (∀ k : ℕ, n ≤ k ∧ k ≤ 580 → (k % 4 = 0 ∧ k % 5 = 0 ∧ k % 6 = 0) ↔ k ∈ Finset.range (m + 1) ∧ k ≠ n)) ∧
  (∀ n' : ℕ, n < n' → n' < 580 → 
    ¬(∃ (m : ℕ), m = 6 ∧ 
      (∀ k : ℕ, n' ≤ k ∧ k ≤ 580 → (k % 4 = 0 ∧ k % 5 = 0 ∧ k % 6 = 0) ↔ k ∈ Finset.range (m + 1) ∧ k ≠ n'))) :=
by sorry

end NUMINAMATH_CALUDE_starting_number_proof_l2948_294830


namespace NUMINAMATH_CALUDE_snake_paint_theorem_l2948_294872

/-- The amount of paint needed for a single cube -/
def paint_per_cube : ℕ := 60

/-- The number of cubes in the snake -/
def total_cubes : ℕ := 2016

/-- The number of cubes in one segment of the snake -/
def cubes_per_segment : ℕ := 6

/-- The amount of paint needed for one segment -/
def paint_per_segment : ℕ := 240

/-- The amount of extra paint needed for the ends of the snake -/
def extra_paint_for_ends : ℕ := 20

/-- Theorem stating the total amount of paint needed for the snake -/
theorem snake_paint_theorem :
  let segments := total_cubes / cubes_per_segment
  let paint_for_segments := segments * paint_per_segment
  paint_for_segments + extra_paint_for_ends = 80660 := by
  sorry

end NUMINAMATH_CALUDE_snake_paint_theorem_l2948_294872


namespace NUMINAMATH_CALUDE_expression_simplification_l2948_294859

theorem expression_simplification (m : ℝ) (h : m ≠ 2) :
  (m + 2 - 5 / (m - 2)) / ((m - 3) / (2 * m - 4)) = 2 * m + 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2948_294859


namespace NUMINAMATH_CALUDE_sugar_recipe_reduction_l2948_294883

theorem sugar_recipe_reduction : 
  (3 + 3 / 4 : ℚ) / 3 = 1 + 1 / 4 := by sorry

end NUMINAMATH_CALUDE_sugar_recipe_reduction_l2948_294883


namespace NUMINAMATH_CALUDE_line_equation_through_points_l2948_294809

/-- Given two points (m, n) and (m + 3, n + 9) in the coordinate plane,
    prove that the equation y = 3x + (n - 3m) represents the line passing through these points. -/
theorem line_equation_through_points (m n : ℝ) :
  ∀ x y : ℝ, y = 3 * x + (n - 3 * m) ↔ (∃ t : ℝ, x = m + 3 * t ∧ y = n + 9 * t) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l2948_294809


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2948_294812

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 8192; 0, -8192] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2948_294812


namespace NUMINAMATH_CALUDE_octahedron_containment_l2948_294896

-- Define the plane equation
def plane_equation (x y z : ℚ) (n : ℤ) : Prop :=
  (x + y + z = n) ∨ (x + y - z = n) ∨ (x - y + z = n) ∨ (x - y - z = n)

-- Define a point not on any plane
def not_on_planes (x y z : ℚ) : Prop :=
  ∀ n : ℤ, ¬ plane_equation x y z n

-- Define a point inside an octahedron
def inside_octahedron (x y z : ℚ) : Prop :=
  ∃ n : ℤ, 
    n < x + y + z ∧ x + y + z < n + 1 ∧
    n < x + y - z ∧ x + y - z < n + 1 ∧
    n < x - y + z ∧ x - y + z < n + 1 ∧
    n < -x + y + z ∧ -x + y + z < n + 1

-- The main theorem
theorem octahedron_containment (x₀ y₀ z₀ : ℚ) 
  (h : not_on_planes x₀ y₀ z₀) :
  ∃ k : ℕ, inside_octahedron (k * x₀) (k * y₀) (k * z₀) := by
  sorry

end NUMINAMATH_CALUDE_octahedron_containment_l2948_294896


namespace NUMINAMATH_CALUDE_arithmetic_sequence_log_property_l2948_294824

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- Define the arithmetic sequence property
def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

-- Define the theorem
theorem arithmetic_sequence_log_property
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : is_arithmetic_sequence (log (a^2 * b^6)) (log (a^4 * b^11)) (log (a^7 * b^14)))
  (h4 : ∃ m : ℕ, (log (b^m)) = (log (a^2 * b^6)) + 7 * ((log (a^4 * b^11)) - (log (a^2 * b^6))))
  : ∃ m : ℕ, m = 73 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_log_property_l2948_294824


namespace NUMINAMATH_CALUDE_total_time_is_186_l2948_294803

def total_time (mac_download : ℕ) (windows_multiplier : ℕ)
  (ny_audio_glitch_count ny_audio_glitch_duration : ℕ)
  (ny_video_glitch_count ny_video_glitch_duration : ℕ)
  (ny_unglitched_multiplier : ℕ)
  (berlin_audio_glitch_count berlin_audio_glitch_duration : ℕ)
  (berlin_video_glitch_count berlin_video_glitch_duration : ℕ)
  (berlin_unglitched_multiplier : ℕ) : ℕ :=
  let windows_download := mac_download * windows_multiplier
  let total_download := mac_download + windows_download

  let ny_audio_glitch := ny_audio_glitch_count * ny_audio_glitch_duration
  let ny_video_glitch := ny_video_glitch_count * ny_video_glitch_duration
  let ny_total_glitch := ny_audio_glitch + ny_video_glitch
  let ny_unglitched := ny_total_glitch * ny_unglitched_multiplier
  let ny_total := ny_total_glitch + ny_unglitched

  let berlin_audio_glitch := berlin_audio_glitch_count * berlin_audio_glitch_duration
  let berlin_video_glitch := berlin_video_glitch_count * berlin_video_glitch_duration
  let berlin_total_glitch := berlin_audio_glitch + berlin_video_glitch
  let berlin_unglitched := berlin_total_glitch * berlin_unglitched_multiplier
  let berlin_total := berlin_total_glitch + berlin_unglitched

  total_download + ny_total + berlin_total

theorem total_time_is_186 :
  total_time 10 3 2 6 1 8 3 3 4 2 5 2 = 186 := by sorry

end NUMINAMATH_CALUDE_total_time_is_186_l2948_294803


namespace NUMINAMATH_CALUDE_zara_goats_l2948_294882

/-- The number of cows Zara bought -/
def num_cows : ℕ := 24

/-- The number of sheep Zara bought -/
def num_sheep : ℕ := 7

/-- The number of groups for transportation -/
def num_groups : ℕ := 3

/-- The number of animals per group -/
def animals_per_group : ℕ := 48

/-- The total number of animals -/
def total_animals : ℕ := num_groups * animals_per_group

/-- The number of goats Zara owns -/
def num_goats : ℕ := total_animals - (num_cows + num_sheep)

theorem zara_goats : num_goats = 113 := by
  sorry

end NUMINAMATH_CALUDE_zara_goats_l2948_294882


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_is_3148_l2948_294842

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ := 5 * (x^3 - 3*x^2 + 3) - 9 * (x^4 - 4*x^2 + 4)

/-- The sum of squares of coefficients of the fully simplified expression -/
def sum_of_squared_coefficients : ℝ := 3148

/-- Theorem stating that the sum of squares of coefficients of the fully simplified expression is 3148 -/
theorem sum_of_squared_coefficients_is_3148 :
  ∃ (a b c d : ℝ), ∀ (x : ℝ), 
    expression x = a*x^4 + b*x^3 + c*x^2 + d ∧
    a^2 + b^2 + c^2 + d^2 = sum_of_squared_coefficients :=
sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_is_3148_l2948_294842


namespace NUMINAMATH_CALUDE_square_sum_equals_two_l2948_294862

theorem square_sum_equals_two (x y : ℝ) 
  (h1 : x - y = -1) 
  (h2 : x * y = 1/2) : 
  x^2 + y^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_two_l2948_294862


namespace NUMINAMATH_CALUDE_contiguous_substring_divisible_by_2011_l2948_294899

def isContiguousSubstring (s t : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℕ), t = (s / 10^k) % 10^m

theorem contiguous_substring_divisible_by_2011 :
  ∃ (N : ℕ), ∀ (a : ℕ), a > N →
    ∃ (s : ℕ), isContiguousSubstring a s ∧ s % 2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_contiguous_substring_divisible_by_2011_l2948_294899


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2948_294826

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x > 1 → 1 / x < 1) ∧
  (∃ x, 1 / x < 1 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2948_294826


namespace NUMINAMATH_CALUDE_sqrt_product_l2948_294801

theorem sqrt_product (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) : Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_l2948_294801


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l2948_294894

theorem fraction_inequality_solution_set :
  {x : ℝ | (2*x + 1) / (x - 3) ≤ 0} = {x : ℝ | -1/2 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l2948_294894


namespace NUMINAMATH_CALUDE_parabola_point_value_l2948_294875

/-- Given points A(a,m), B(b,m), P(a+b,n) on the parabola y=x^2-2x-2, prove that n = -2 -/
theorem parabola_point_value (a b m n : ℝ) : 
  (m = a^2 - 2*a - 2) →  -- A is on the parabola
  (m = b^2 - 2*b - 2) →  -- B is on the parabola
  (n = (a+b)^2 - 2*(a+b) - 2) →  -- P is on the parabola
  (n = -2) := by
sorry

end NUMINAMATH_CALUDE_parabola_point_value_l2948_294875
