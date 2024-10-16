import Mathlib

namespace NUMINAMATH_CALUDE_intersection_S_T_l2207_220771

def S : Set ℝ := {x | x + 1 ≥ 2}
def T : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_S_T : S ∩ T = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_S_T_l2207_220771


namespace NUMINAMATH_CALUDE_puppy_cost_proof_l2207_220797

/-- Given a purchase of puppies with specific conditions, prove the cost of non-sale puppies. -/
theorem puppy_cost_proof (total_cost : ℕ) (sale_price : ℕ) (num_puppies : ℕ) (num_sale_puppies : ℕ) :
  total_cost = 800 →
  sale_price = 150 →
  num_puppies = 5 →
  num_sale_puppies = 3 →
  ∃ (non_sale_price : ℕ), 
    non_sale_price * (num_puppies - num_sale_puppies) + sale_price * num_sale_puppies = total_cost ∧
    non_sale_price = 175 := by
  sorry

end NUMINAMATH_CALUDE_puppy_cost_proof_l2207_220797


namespace NUMINAMATH_CALUDE_courtyard_length_l2207_220723

/-- Proves that the length of a rectangular courtyard is 18 meters -/
theorem courtyard_length (width : ℝ) (brick_length : ℝ) (brick_width : ℝ) (total_bricks : ℕ) :
  width = 12 →
  brick_length = 0.12 →
  brick_width = 0.06 →
  total_bricks = 30000 →
  (width * (width * total_bricks * brick_length * brick_width)⁻¹) = 18 :=
by sorry

end NUMINAMATH_CALUDE_courtyard_length_l2207_220723


namespace NUMINAMATH_CALUDE_sqrt_14_bounds_l2207_220745

theorem sqrt_14_bounds : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_14_bounds_l2207_220745


namespace NUMINAMATH_CALUDE_white_tiles_count_l2207_220795

theorem white_tiles_count (total : Nat) (yellow : Nat) (purple : Nat) : 
  total = 20 → yellow = 3 → purple = 6 → 
  ∃ (blue white : Nat), blue = yellow + 1 ∧ white = total - (yellow + blue + purple) ∧ white = 7 := by
  sorry

end NUMINAMATH_CALUDE_white_tiles_count_l2207_220795


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_min_value_achievable_l2207_220786

theorem min_value_quadratic_form (x y z : ℝ) :
  3 * x^2 + 2*x*y + 3 * y^2 + 2*y*z + 3 * z^2 - 3*x + 3*y - 3*z + 9 ≥ (3/2 : ℝ) :=
by sorry

theorem min_value_achievable :
  ∃ (x y z : ℝ), 3 * x^2 + 2*x*y + 3 * y^2 + 2*y*z + 3 * z^2 - 3*x + 3*y - 3*z + 9 = (3/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_min_value_achievable_l2207_220786


namespace NUMINAMATH_CALUDE_internet_discount_percentage_l2207_220765

theorem internet_discount_percentage
  (monthly_rate : ℝ)
  (total_payment : ℝ)
  (num_months : ℕ)
  (h1 : monthly_rate = 50)
  (h2 : total_payment = 190)
  (h3 : num_months = 4) :
  (monthly_rate - total_payment / num_months) / monthly_rate * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_internet_discount_percentage_l2207_220765


namespace NUMINAMATH_CALUDE_interest_rate_is_30_percent_l2207_220764

-- Define the compound interest function
def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * (1 + r) ^ t

-- State the theorem
theorem interest_rate_is_30_percent 
  (P : ℝ) 
  (h1 : compound_interest P r 2 = 17640) 
  (h2 : compound_interest P r 3 = 22932) : 
  r = 0.3 := by
  sorry


end NUMINAMATH_CALUDE_interest_rate_is_30_percent_l2207_220764


namespace NUMINAMATH_CALUDE_p_plus_q_value_l2207_220738

theorem p_plus_q_value (p q : ℝ) 
  (hp : p^3 - 18*p^2 + 25*p - 75 = 0) 
  (hq : 10*q^3 - 75*q^2 - 365*q + 3375 = 0) : 
  p + q = 39/4 := by
sorry

end NUMINAMATH_CALUDE_p_plus_q_value_l2207_220738


namespace NUMINAMATH_CALUDE_divisibility_condition_l2207_220728

theorem divisibility_condition (n : ℕ+) :
  (∃ m : ℤ, (2^n.val - 1) ∣ (m^2 + 9)) ↔ ∃ x : ℕ, n = 2^x :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2207_220728


namespace NUMINAMATH_CALUDE_triangle_properties_l2207_220760

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  2 * t.c = t.a + Real.cos t.A * t.b / Real.cos t.B ∧
  t.b = 4 ∧
  t.a + t.c = 3 * Real.sqrt 2

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.B = π / 3 ∧ 
  (1 / 2 * t.a * t.c * Real.sin t.B : ℝ) = Real.sqrt 3 / 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2207_220760


namespace NUMINAMATH_CALUDE_small_cube_side_length_l2207_220758

theorem small_cube_side_length (large_cube_side : ℝ) (num_small_cubes : ℕ) (small_cube_side : ℝ) :
  large_cube_side = 1 →
  num_small_cubes = 1000 →
  large_cube_side ^ 3 = num_small_cubes * small_cube_side ^ 3 →
  small_cube_side = 0.1 := by
sorry

end NUMINAMATH_CALUDE_small_cube_side_length_l2207_220758


namespace NUMINAMATH_CALUDE_equilateral_triangle_figure_divisible_l2207_220719

/-- A figure composed of equilateral triangles -/
structure EquilateralTriangleFigure where
  /-- The set of points in the figure -/
  points : Set ℝ × ℝ
  /-- Predicate asserting that the figure is composed of equal equilateral triangles -/
  is_composed_of_equilateral_triangles : Prop

/-- A straight line in 2D space -/
structure Line where
  /-- Slope of the line -/
  slope : ℝ
  /-- Y-intercept of the line -/
  intercept : ℝ

/-- Predicate asserting that a line divides a figure into two congruent parts -/
def divides_into_congruent_parts (f : EquilateralTriangleFigure) (l : Line) : Prop :=
  sorry

/-- Theorem stating that any figure composed of equal equilateral triangles
    can be divided into two congruent parts by a straight line -/
theorem equilateral_triangle_figure_divisible (f : EquilateralTriangleFigure) :
  ∃ l : Line, divides_into_congruent_parts f l :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_figure_divisible_l2207_220719


namespace NUMINAMATH_CALUDE_largest_class_size_l2207_220711

/-- Proves that in a school with 5 classes, where each class has 2 students less than the previous class,
    and the total number of students is 115, the largest class has 27 students. -/
theorem largest_class_size (total_students : ℕ) (num_classes : ℕ) (diff : ℕ) :
  total_students = 115 →
  num_classes = 5 →
  diff = 2 →
  ∃ (x : ℕ), x = 27 ∧ 
    (x + (x - diff) + (x - 2*diff) + (x - 3*diff) + (x - 4*diff) = total_students) :=
by sorry

end NUMINAMATH_CALUDE_largest_class_size_l2207_220711


namespace NUMINAMATH_CALUDE_point_translation_rotation_l2207_220702

/-- Represents a point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally -/
def translate (p : Point) (dx : ℝ) : Point :=
  ⟨p.x + dx, p.y⟩

/-- Rotates a point 90 degrees clockwise around the origin -/
def rotate90Clockwise (p : Point) : Point :=
  ⟨p.y, -p.x⟩

theorem point_translation_rotation (p : Point) :
  p = ⟨-5, 4⟩ →
  (rotate90Clockwise (translate p 8)) = ⟨4, -3⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_translation_rotation_l2207_220702


namespace NUMINAMATH_CALUDE_jogger_train_distance_l2207_220712

/-- Calculates the distance a jogger is ahead of a train given their speeds and the time it takes for the train to pass the jogger. -/
theorem jogger_train_distance (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) : 
  jogger_speed = 10 * (5/18) → 
  train_speed = 46 * (5/18) → 
  train_length = 120 → 
  passing_time = 46 → 
  (train_speed - jogger_speed) * passing_time - train_length = 340 := by
  sorry

#check jogger_train_distance

end NUMINAMATH_CALUDE_jogger_train_distance_l2207_220712


namespace NUMINAMATH_CALUDE_integral_x_plus_sin_x_l2207_220726

theorem integral_x_plus_sin_x (x : ℝ) : 
  ∫ x in (0)..(π/2), (x + Real.sin x) = π^2/8 + 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_plus_sin_x_l2207_220726


namespace NUMINAMATH_CALUDE_ratio_xyz_l2207_220752

theorem ratio_xyz (x y z : ℝ) (h1 : 0.1 * x = 0.2 * y) (h2 : 0.3 * y = 0.4 * z) :
  ∃ (k : ℝ), k > 0 ∧ x = 8 * k ∧ y = 4 * k ∧ z = 3 * k :=
sorry

end NUMINAMATH_CALUDE_ratio_xyz_l2207_220752


namespace NUMINAMATH_CALUDE_average_weight_decrease_l2207_220715

theorem average_weight_decrease (initial_average : ℝ) : 
  let initial_total : ℝ := 8 * initial_average
  let new_total : ℝ := initial_total - 86 + 46
  let new_average : ℝ := new_total / 8
  initial_average - new_average = 5 := by sorry

end NUMINAMATH_CALUDE_average_weight_decrease_l2207_220715


namespace NUMINAMATH_CALUDE_product_of_sums_inequality_l2207_220732

theorem product_of_sums_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_inequality_l2207_220732


namespace NUMINAMATH_CALUDE_water_in_bucket_A_l2207_220772

theorem water_in_bucket_A : ∃ (A B : ℝ),
  A > 0 ∧ B > 0 ∧
  (A - 6 = (1/3) * (B + 6)) ∧
  (B - 6 = (1/2) * (A + 6)) ∧
  A = 13.2 := by
sorry

end NUMINAMATH_CALUDE_water_in_bucket_A_l2207_220772


namespace NUMINAMATH_CALUDE_pta_spending_ratio_l2207_220701

theorem pta_spending_ratio (initial_amount : ℚ) (spent_on_supplies : ℚ) (amount_left : ℚ) 
  (h1 : initial_amount = 400)
  (h2 : amount_left = 150)
  (h3 : amount_left = initial_amount - spent_on_supplies - (initial_amount - spent_on_supplies) / 2) :
  spent_on_supplies = 100 ∧ spent_on_supplies / initial_amount = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_pta_spending_ratio_l2207_220701


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l2207_220725

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define proposition Q
def prop_Q (f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, |f' x| < 2017

-- Define proposition P
def prop_P (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → |(f x₁ - f x₂) / (x₁ - x₂)| < 2017

-- State the theorem
theorem necessary_not_sufficient
  (hf : Differentiable ℝ f)
  (hf' : ∀ x, HasDerivAt f (f' x) x) :
  (prop_Q f' → prop_P f) ∧ ¬(prop_P f → prop_Q f') :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l2207_220725


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2207_220720

def a : ℂ := 3 + 2 * Complex.I
def b : ℂ := 2 - 3 * Complex.I

theorem complex_expression_equality : 3 * a + 4 * b + a^2 + b^2 = 35 - 6 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2207_220720


namespace NUMINAMATH_CALUDE_inscribed_parallelepiped_volume_l2207_220793

/-- The volume of a rectangular parallelepiped inscribed in a pyramid -/
theorem inscribed_parallelepiped_volume
  (a : ℝ) -- Side length of the square base of the pyramid
  (α β : ℝ) -- Angles α and β as described in the problem
  (h1 : 0 < a)
  (h2 : 0 < α ∧ α < π / 2)
  (h3 : 0 < β ∧ β < π / 2)
  (h4 : α + β < π / 2) :
  ∃ V : ℝ, -- Volume of the parallelepiped
    V = (a^3 * Real.sqrt 2 * Real.sin α * Real.cos α^2 * Real.sin β^3) /
        Real.sin (α + β)^3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_parallelepiped_volume_l2207_220793


namespace NUMINAMATH_CALUDE_third_column_sum_l2207_220780

/-- Represents a 3x3 grid of numbers -/
def Grid := Matrix (Fin 3) (Fin 3) ℤ

/-- The sum of a row in the grid -/
def row_sum (g : Grid) (i : Fin 3) : ℤ :=
  (g i 0) + (g i 1) + (g i 2)

/-- The sum of a column in the grid -/
def col_sum (g : Grid) (j : Fin 3) : ℤ :=
  (g 0 j) + (g 1 j) + (g 2 j)

/-- The theorem statement -/
theorem third_column_sum (g : Grid) 
  (h1 : row_sum g 0 = 24)
  (h2 : row_sum g 1 = 26)
  (h3 : row_sum g 2 = 40)
  (h4 : col_sum g 0 = 27)
  (h5 : col_sum g 1 = 20) :
  col_sum g 2 = 43 := by
  sorry


end NUMINAMATH_CALUDE_third_column_sum_l2207_220780


namespace NUMINAMATH_CALUDE_chord_bisected_by_point_4_2_l2207_220707

/-- The equation of an ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

/-- A point is the midpoint of two other points -/
def is_midpoint (x y x1 y1 x2 y2 : ℝ) : Prop :=
  x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2

/-- A point lies on a line -/
def point_on_line (x y : ℝ) : Prop := x + 2*y - 8 = 0

theorem chord_bisected_by_point_4_2 (x1 y1 x2 y2 : ℝ) :
  is_on_ellipse x1 y1 →
  is_on_ellipse x2 y2 →
  is_midpoint 4 2 x1 y1 x2 y2 →
  point_on_line x1 y1 ∧ point_on_line x2 y2 :=
sorry

end NUMINAMATH_CALUDE_chord_bisected_by_point_4_2_l2207_220707


namespace NUMINAMATH_CALUDE_average_pages_is_23_l2207_220766

/-- The number of pages in the storybook Taesoo read -/
def total_pages : ℕ := 161

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The average number of pages read per day -/
def average_pages : ℚ := total_pages / days_in_week

/-- Theorem stating that the average number of pages read per day is 23 -/
theorem average_pages_is_23 : average_pages = 23 := by
  sorry

end NUMINAMATH_CALUDE_average_pages_is_23_l2207_220766


namespace NUMINAMATH_CALUDE_car_rental_rates_equal_l2207_220716

/-- The daily rate of Safety Rent-a-Car in dollars -/
def safety_daily_rate : ℝ := 21.95

/-- The per-mile rate of Safety Rent-a-Car in dollars -/
def safety_mile_rate : ℝ := 0.19

/-- The per-mile rate of the second company in dollars -/
def second_mile_rate : ℝ := 0.21

/-- The number of miles driven -/
def miles_driven : ℝ := 150

/-- The daily rate of the second company in dollars -/
def second_daily_rate : ℝ := 18.95

theorem car_rental_rates_equal :
  safety_daily_rate + safety_mile_rate * miles_driven =
  second_daily_rate + second_mile_rate * miles_driven :=
by sorry

end NUMINAMATH_CALUDE_car_rental_rates_equal_l2207_220716


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l2207_220788

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (candidate_a_percentage : ℚ)
  (candidate_a_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 80 / 100)
  (h3 : candidate_a_votes = 380800) :
  (total_votes - (candidate_a_votes / candidate_a_percentage)) / total_votes = 15 / 100 := by
sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l2207_220788


namespace NUMINAMATH_CALUDE_problem_statement_l2207_220708

def p : Prop := ∀ a : ℝ, a^2 ≥ 0

def f (x : ℝ) : ℝ := x^2 - x

def q : Prop := ∀ x y : ℝ, 0 < x ∧ x < y → f x < f y

theorem problem_statement : p ∨ q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2207_220708


namespace NUMINAMATH_CALUDE_decreasing_function_k_bound_l2207_220757

/-- The function f(x) = kx³ + 3(k-1)x² - k² + 1 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^3 + 3 * (k - 1) * x^2 - k^2 + 1

/-- The derivative of f(x) with respect to x -/
def f_deriv (k : ℝ) (x : ℝ) : ℝ := 3 * k * x^2 + 6 * (k - 1) * x

theorem decreasing_function_k_bound :
  ∀ k : ℝ, (∀ x ∈ Set.Ioo 0 4, f_deriv k x ≤ 0) → k ≤ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_k_bound_l2207_220757


namespace NUMINAMATH_CALUDE_quadratic_properties_l2207_220777

-- Define a quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties :
  ∀ (a b c : ℝ),
  (∃ (x_min : ℝ), ∀ (x : ℝ), quadratic a b c x ≥ quadratic a b c x_min ∧ quadratic a b c x_min = 1) →
  quadratic a b c 0 = 3 →
  quadratic a b c 2 = 3 →
  (a = 2 ∧ b = -4 ∧ c = 3) ∧
  (∀ (a_range : ℝ), (∃ (x y : ℝ), 2 * a_range ≤ x ∧ x < y ∧ y ≤ a_range + 1 ∧
    (quadratic 2 (-4) 3 x < quadratic 2 (-4) 3 y ∧ quadratic 2 (-4) 3 y > quadratic 2 (-4) 3 (a_range + 1))) ↔
    (0 < a_range ∧ a_range < 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2207_220777


namespace NUMINAMATH_CALUDE_total_peaches_l2207_220790

theorem total_peaches (initial_baskets : Nat) (initial_peaches_per_basket : Nat)
                      (additional_baskets : Nat) (additional_peaches_per_basket : Nat) :
  initial_baskets = 5 →
  initial_peaches_per_basket = 20 →
  additional_baskets = 4 →
  additional_peaches_per_basket = 25 →
  initial_baskets * initial_peaches_per_basket +
  additional_baskets * additional_peaches_per_basket = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_l2207_220790


namespace NUMINAMATH_CALUDE_sqrt_product_equals_three_halves_l2207_220759

theorem sqrt_product_equals_three_halves : 
  Real.sqrt 5 * Real.sqrt (9 / 20) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_three_halves_l2207_220759


namespace NUMINAMATH_CALUDE_no_linear_term_condition_l2207_220763

theorem no_linear_term_condition (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (x - 4) = a * x^2 + b) ↔ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_condition_l2207_220763


namespace NUMINAMATH_CALUDE_min_value_f_l2207_220754

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + (2 - a)*Real.log x

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2*x - 4 - (2 - a)/x

-- Theorem statement
theorem min_value_f (a : ℝ) :
  ∃ (min_val : ℝ), ∀ x ∈ Set.Icc (Real.exp 1) (Real.exp 2), f a x ≥ min_val ∧
  (min_val = f a (Real.exp 1) ∨
   min_val = f a (Real.exp 2) ∨
   (∃ y ∈ Set.Ioo (Real.exp 1) (Real.exp 2), min_val = f a y ∧ f_deriv a y = 0)) :=
sorry

end

end NUMINAMATH_CALUDE_min_value_f_l2207_220754


namespace NUMINAMATH_CALUDE_unique_g_function_l2207_220756

-- Define the properties of function g
def is_valid_g (g : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂ : ℝ, g (x₁ + x₂) = g x₁ * g x₂) ∧
  (g 1 = 3) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂)

-- Theorem statement
theorem unique_g_function :
  ∃! g : ℝ → ℝ, is_valid_g g ∧ (∀ x : ℝ, g x = 3^x) :=
by sorry

end NUMINAMATH_CALUDE_unique_g_function_l2207_220756


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l2207_220750

theorem largest_solution_of_equation (x : ℝ) :
  (3 * (9 * x^2 + 10 * x + 11) = x * (9 * x - 45)) →
  x ≤ (-1 / 2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l2207_220750


namespace NUMINAMATH_CALUDE_martin_financial_calculation_l2207_220734

theorem martin_financial_calculation (g u q : ℂ) (h1 : g * q - u = 15000) (h2 : g = 10) (h3 : u = 10 + 200 * Complex.I) : q = 1501 + 20 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_martin_financial_calculation_l2207_220734


namespace NUMINAMATH_CALUDE_cosine_pi_third_derivative_l2207_220749

theorem cosine_pi_third_derivative :
  let y : ℝ → ℝ := λ _ => Real.cos (π / 3)
  ∀ x : ℝ, deriv y x = 0 := by
sorry

end NUMINAMATH_CALUDE_cosine_pi_third_derivative_l2207_220749


namespace NUMINAMATH_CALUDE_fraction_problem_l2207_220742

theorem fraction_problem (x : ℚ) : (3/4 : ℚ) * x * (2/3 : ℚ) = (2/5 : ℚ) → x = (4/5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2207_220742


namespace NUMINAMATH_CALUDE_sum_min_max_value_l2207_220768

theorem sum_min_max_value (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 10)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 + e^2 = 30) : 
  let f := fun (x y z w v : ℝ) => 5 * (x^3 + y^3 + z^3 + w^3 + v^3) - (x^4 + y^4 + z^4 + w^4 + v^4)
  ∃ (m M : ℝ), 
    (∀ x y z w v, f x y z w v ≥ m) ∧ 
    (∃ x y z w v, f x y z w v = m) ∧
    (∀ x y z w v, f x y z w v ≤ M) ∧ 
    (∃ x y z w v, f x y z w v = M) ∧
    m + M = 94 :=
by sorry

end NUMINAMATH_CALUDE_sum_min_max_value_l2207_220768


namespace NUMINAMATH_CALUDE_new_number_correct_l2207_220735

/-- Given a two-digit number with tens' digit t and units' digit u,
    the function calculates the new three-digit number formed by
    reversing the digits and placing 2 after the reversed number. -/
def new_number (t u : ℕ) : ℕ :=
  100 * u + 10 * t + 2

/-- Theorem stating that the new_number function correctly calculates
    the desired three-digit number for any two-digit number. -/
theorem new_number_correct (t u : ℕ) (h1 : t ≥ 1) (h2 : t ≤ 9) (h3 : u ≤ 9) :
  new_number t u = 100 * u + 10 * t + 2 :=
by sorry

end NUMINAMATH_CALUDE_new_number_correct_l2207_220735


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2207_220714

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a + a^2 > b + b^2) ∧
  (∃ a b, a + a^2 > b + b^2 ∧ ¬(a > b ∧ b > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2207_220714


namespace NUMINAMATH_CALUDE_inventory_net_change_l2207_220743

/-- Represents the quantity of an ingredient on a given day -/
structure IngredientQuantity where
  day1 : Float
  day7 : Float

/-- Calculates the change in quantity for an ingredient -/
def calculateChange (q : IngredientQuantity) : Float :=
  q.day1 - q.day7

/-- Represents the inventory of all ingredients -/
structure Inventory where
  bakingPowder : IngredientQuantity
  flour : IngredientQuantity
  sugar : IngredientQuantity
  chocolateChips : IngredientQuantity

/-- Calculates the net change for all ingredients -/
def calculateNetChange (inv : Inventory) : Float :=
  calculateChange inv.bakingPowder +
  calculateChange inv.flour +
  calculateChange inv.sugar +
  calculateChange inv.chocolateChips

theorem inventory_net_change (inv : Inventory) 
  (h1 : inv.bakingPowder = { day1 := 4, day7 := 2.5 })
  (h2 : inv.flour = { day1 := 12, day7 := 7 })
  (h3 : inv.sugar = { day1 := 10, day7 := 6.5 })
  (h4 : inv.chocolateChips = { day1 := 6, day7 := 3.7 }) :
  calculateNetChange inv = 12.3 := by
  sorry

end NUMINAMATH_CALUDE_inventory_net_change_l2207_220743


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l2207_220706

theorem lcm_factor_problem (A B : ℕ+) (hcf other_factor : ℕ+) :
  hcf = 23 →
  A = 345 →
  Nat.lcm A B = hcf * other_factor * 15 →
  other_factor = 23 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l2207_220706


namespace NUMINAMATH_CALUDE_trombone_players_count_l2207_220787

/-- Represents the Oprah Winfrey High School marching band -/
structure MarchingBand where
  trumpet_weight : ℕ := 5
  clarinet_weight : ℕ := 5
  trombone_weight : ℕ := 10
  tuba_weight : ℕ := 20
  drum_weight : ℕ := 15
  trumpet_count : ℕ := 6
  clarinet_count : ℕ := 9
  tuba_count : ℕ := 3
  drum_count : ℕ := 2
  total_weight : ℕ := 245

/-- Calculates the number of trombone players in the marching band -/
def trombone_players (band : MarchingBand) : ℕ :=
  let other_weight := band.trumpet_weight * band.trumpet_count +
                      band.clarinet_weight * band.clarinet_count +
                      band.tuba_weight * band.tuba_count +
                      band.drum_weight * band.drum_count
  let trombone_total_weight := band.total_weight - other_weight
  trombone_total_weight / band.trombone_weight

/-- Theorem stating that the number of trombone players is 8 -/
theorem trombone_players_count (band : MarchingBand) : trombone_players band = 8 := by
  sorry

end NUMINAMATH_CALUDE_trombone_players_count_l2207_220787


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2207_220775

theorem polynomial_factorization (x : ℝ) : 
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 19) * (x^2 + 6*x - 2) := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2207_220775


namespace NUMINAMATH_CALUDE_joan_payment_l2207_220740

/-- Represents the purchase amounts for Joan, Karl, and Lea --/
structure Purchases where
  joan : ℝ
  karl : ℝ
  lea : ℝ

/-- Defines the conditions of the telescope purchase problem --/
def validPurchases (p : Purchases) : Prop :=
  p.joan + p.karl + p.lea = 600 ∧
  2 * p.joan = p.karl + 74 ∧
  p.lea - p.karl = 52

/-- Theorem stating that if the purchases satisfy the given conditions, 
    then Joan's payment is $139.20 --/
theorem joan_payment (p : Purchases) (h : validPurchases p) : 
  p.joan = 139.20 := by
  sorry

end NUMINAMATH_CALUDE_joan_payment_l2207_220740


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l2207_220721

theorem quadratic_rewrite_sum (k : ℝ) : 
  ∃ (d r s : ℝ), 8 * k^2 + 12 * k + 18 = d * (k + r)^2 + s ∧ r + s = 57/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l2207_220721


namespace NUMINAMATH_CALUDE_isabella_hair_growth_l2207_220703

/-- Calculates hair growth given initial and final hair lengths -/
def hair_growth (initial_length final_length : ℝ) : ℝ :=
  final_length - initial_length

theorem isabella_hair_growth :
  let initial_length : ℝ := 18
  let final_length : ℝ := 24
  hair_growth initial_length final_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_isabella_hair_growth_l2207_220703


namespace NUMINAMATH_CALUDE_quadratic_function_values_l2207_220746

theorem quadratic_function_values (p q : ℝ) : ¬ (∀ x ∈ ({1, 2, 3} : Set ℝ), |x^2 + p*x + q| < (1/2 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_values_l2207_220746


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l2207_220724

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical_conversion 
  (x y z : ℝ) 
  (h_x : x = -2) 
  (h_y : y = -2 * Real.sqrt 3) 
  (h_z : z = -1) :
  ∃ (r θ : ℝ),
    r > 0 ∧
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    r = 4 ∧
    θ = 4 * Real.pi / 3 ∧
    x = r * Real.cos θ ∧
    y = r * Real.sin θ ∧
    z = -1 :=
by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l2207_220724


namespace NUMINAMATH_CALUDE_prob_white_second_is_half_l2207_220789

/-- Represents the number of black balls initially in the bag -/
def initial_black_balls : ℕ := 4

/-- Represents the number of white balls initially in the bag -/
def initial_white_balls : ℕ := 3

/-- Represents the total number of balls initially in the bag -/
def total_balls : ℕ := initial_black_balls + initial_white_balls

/-- Represents the probability of drawing a white ball on the second draw,
    given that a black ball was drawn on the first draw -/
def prob_white_second_given_black_first : ℚ :=
  initial_white_balls / (total_balls - 1)

theorem prob_white_second_is_half :
  prob_white_second_given_black_first = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_second_is_half_l2207_220789


namespace NUMINAMATH_CALUDE_triangle_side_equation_l2207_220779

theorem triangle_side_equation (a b x : ℝ) (θ : ℝ) : 
  a = 6 → b = 2 * Real.sqrt 7 → θ = π / 3 → 
  x ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos θ → 
  x ^ 2 - 6 * x + 8 = 0 := by sorry

end NUMINAMATH_CALUDE_triangle_side_equation_l2207_220779


namespace NUMINAMATH_CALUDE_vector_addition_l2207_220784

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![1, 5]

-- Define the operation 2a + b
def result : Fin 2 → ℝ := fun i => 2 * a i + b i

-- Theorem statement
theorem vector_addition : result = ![5, 7] := by sorry

end NUMINAMATH_CALUDE_vector_addition_l2207_220784


namespace NUMINAMATH_CALUDE_no_real_roots_condition_l2207_220709

theorem no_real_roots_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_condition_l2207_220709


namespace NUMINAMATH_CALUDE_meeting_probability_for_seven_steps_l2207_220762

/-- Represents a position on the coordinate plane -/
structure Position where
  x : ℕ
  y : ℕ

/-- Represents the possible movements for an object -/
inductive Movement
  | Right
  | Up
  | Left
  | Down

/-- Represents an object on the coordinate plane -/
structure Object where
  position : Position
  allowedMovements : List Movement

/-- Calculates the number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Calculates the number of intersection paths for given number of steps -/
def intersectionPaths (steps : ℕ) : ℕ := sorry

/-- The probability of two objects meeting given their initial positions and movement constraints -/
def meetingProbability (obj1 obj2 : Object) (steps : ℕ) : ℚ := sorry

theorem meeting_probability_for_seven_steps :
  let c : Object := ⟨⟨1, 1⟩, [Movement.Right, Movement.Up]⟩
  let d : Object := ⟨⟨6, 7⟩, [Movement.Left, Movement.Down]⟩
  meetingProbability c d 7 = 1715 / 16384 := by sorry

end NUMINAMATH_CALUDE_meeting_probability_for_seven_steps_l2207_220762


namespace NUMINAMATH_CALUDE_javiers_dogs_l2207_220792

theorem javiers_dogs (total_legs : ℕ) (human_count : ℕ) (human_legs : ℕ) (dog_legs : ℕ) :
  total_legs = 22 →
  human_count = 5 →
  human_legs = 2 →
  dog_legs = 4 →
  (human_count * human_legs + (total_legs - human_count * human_legs) / dog_legs : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_javiers_dogs_l2207_220792


namespace NUMINAMATH_CALUDE_triangle_inequality_l2207_220769

theorem triangle_inequality (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^3 / c^3 + b^3 / c^3 + 3 * a * b / c^2 > 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2207_220769


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2207_220767

theorem determinant_of_specific_matrix : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, -4, 4; 0, 6, -2; 5, -3, 2]
  Matrix.det A = -68 := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2207_220767


namespace NUMINAMATH_CALUDE_distance_proof_l2207_220791

/-- Proves that the distance between two points is 2 km given specific travel conditions -/
theorem distance_proof (T : ℝ) : 
  (4 * (T + 7/60) = 8 * (T - 8/60)) → 
  (4 * (T + 7/60) = 2) := by
  sorry

end NUMINAMATH_CALUDE_distance_proof_l2207_220791


namespace NUMINAMATH_CALUDE_limit_one_minus_cos_x_over_x_squared_l2207_220744

theorem limit_one_minus_cos_x_over_x_squared :
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |((1 - Real.cos x) / x^2) - (1/2)| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_one_minus_cos_x_over_x_squared_l2207_220744


namespace NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l2207_220729

theorem coefficient_x4_in_expansion (x : ℝ) : 
  ∃ (a b c d e f : ℝ), (2*x + 1) * (x - 1)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f ∧ b = 15 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l2207_220729


namespace NUMINAMATH_CALUDE_vector_parallel_proof_l2207_220794

def vector_a (m : ℚ) : Fin 2 → ℚ := ![1, m]
def vector_b : Fin 2 → ℚ := ![3, -2]

def parallel (u v : Fin 2 → ℚ) : Prop :=
  ∃ (k : ℚ), ∀ (i : Fin 2), u i = k * v i

theorem vector_parallel_proof (m : ℚ) :
  parallel (vector_a m + vector_b) vector_b → m = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_proof_l2207_220794


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_squares_l2207_220781

theorem product_of_sum_and_sum_of_squares (a b : ℝ) 
  (sum_of_squares : a^2 + b^2 = 26) 
  (sum : a + b = 7) : 
  a * b = 23 / 2 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_squares_l2207_220781


namespace NUMINAMATH_CALUDE_inequality_always_true_l2207_220747

theorem inequality_always_true : ∀ x : ℝ, (x + 1) * (2 - x) < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l2207_220747


namespace NUMINAMATH_CALUDE_number_of_girls_in_school_l2207_220753

/-- Represents the number of students in a section -/
def SectionSize : ℕ := 24

/-- Represents the total number of boys in the school -/
def TotalBoys : ℕ := 408

/-- Represents the total number of sections -/
def TotalSections : ℕ := 26

/-- Represents the number of sections for boys -/
def BoySections : ℕ := 17

/-- Represents the number of sections for girls -/
def GirlSections : ℕ := 9

/-- Theorem stating the number of girls in the school -/
theorem number_of_girls_in_school : 
  TotalBoys / BoySections = SectionSize ∧ 
  BoySections + GirlSections = TotalSections → 
  GirlSections * SectionSize = 216 :=
by sorry

end NUMINAMATH_CALUDE_number_of_girls_in_school_l2207_220753


namespace NUMINAMATH_CALUDE_range_of_m_l2207_220770

-- Define the propositions p and q
def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the condition that ¬p is necessary but not sufficient for ¬q
def not_p_necessary_not_sufficient_for_not_q (m : ℝ) : Prop :=
  (∀ x, ¬(q x m) → ¬(p x)) ∧ (∃ x, ¬(p x) ∧ q x m)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (m > 0 ∧ not_p_necessary_not_sufficient_for_not_q m) ↔ m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2207_220770


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2207_220710

/-- Represents a repeating decimal of the form 0.abcabc... where abc is a finite sequence of digits -/
def RepeatingDecimal (numerator denominator : ℕ) : ℚ := numerator / denominator

theorem repeating_decimal_sum : 
  RepeatingDecimal 4 33 + RepeatingDecimal 2 999 + RepeatingDecimal 2 99999 = 12140120 / 99999 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2207_220710


namespace NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l2207_220761

theorem arithmetic_progression_common_difference 
  (n : ℕ) 
  (d : ℚ) 
  (sum_original : ℚ) 
  (sum_decrease_min : ℚ) 
  (sum_decrease_max : ℚ) :
  (n > 0) →
  (sum_original = 63) →
  (sum_original = (n / 2) * (3 * d + (n - 1) * d)) →
  (sum_decrease_min = 7) →
  (sum_decrease_max = 8) →
  (sum_original - (n / 2) * (2 * d + (n - 1) * d) ≥ sum_decrease_min) →
  (sum_original - (n / 2) * (2 * d + (n - 1) * d) ≤ sum_decrease_max) →
  (d = 21/8 ∨ d = 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l2207_220761


namespace NUMINAMATH_CALUDE_no_solution_implies_a_equals_one_l2207_220718

theorem no_solution_implies_a_equals_one (a : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → (a * x) / (x - 2) ≠ 4 / (x - 2) + 1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_equals_one_l2207_220718


namespace NUMINAMATH_CALUDE_certain_number_problem_l2207_220739

theorem certain_number_problem (x : ℝ) : 
  (0.3 * x) - (1/3) * (0.3 * x) = 36 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2207_220739


namespace NUMINAMATH_CALUDE_characterization_of_k_l2207_220730

/-- The greatest odd divisor of a natural number -/
def greatestOddDivisor (m : ℕ) : ℕ := sorry

/-- The property that n does not divide the greatest odd divisor of k^n + 1 -/
def noDivide (k n : ℕ) : Prop :=
  ¬(n ∣ greatestOddDivisor ((k^n + 1) : ℕ))

/-- The main theorem -/
theorem characterization_of_k (k : ℕ) (h : k ≥ 2) :
  (∃ l : ℕ, l ≥ 2 ∧ k = 2^l - 1) ↔ (∀ n : ℕ, n ≥ 2 → noDivide k n) := by
  sorry

end NUMINAMATH_CALUDE_characterization_of_k_l2207_220730


namespace NUMINAMATH_CALUDE_magician_earnings_l2207_220799

theorem magician_earnings 
  (price_per_deck : ℕ) 
  (initial_decks : ℕ) 
  (final_decks : ℕ) :
  price_per_deck = 2 →
  initial_decks = 5 →
  final_decks = 3 →
  (initial_decks - final_decks) * price_per_deck = 4 :=
by sorry

end NUMINAMATH_CALUDE_magician_earnings_l2207_220799


namespace NUMINAMATH_CALUDE_count_true_propositions_l2207_220748

/-- Represents a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The proposition p --/
def p (l1 l2 : Line) : Prop :=
  (∃ (k : ℝ), k ≠ 0 ∧ l1.a = k * l2.a ∧ l1.b = k * l2.b) → l1.a * l2.b - l2.a * l1.b = 0

/-- The converse of p --/
def p_converse (l1 l2 : Line) : Prop :=
  l1.a * l2.b - l2.a * l1.b = 0 → (∃ (k : ℝ), k ≠ 0 ∧ l1.a = k * l2.a ∧ l1.b = k * l2.b)

/-- Count of true propositions among p, its converse, negation, and contrapositive --/
def f_p : ℕ := 2

/-- The main theorem --/
theorem count_true_propositions :
  (∀ l1 l2 : Line, p l1 l2) ∧
  (∃ l1 l2 : Line, ¬(p_converse l1 l2)) ∧
  f_p = 2 := by sorry

end NUMINAMATH_CALUDE_count_true_propositions_l2207_220748


namespace NUMINAMATH_CALUDE_multiple_of_six_square_greater_144_less_30_l2207_220755

theorem multiple_of_six_square_greater_144_less_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 6 * k)
  (h2 : x^2 > 144)
  (h3 : x < 30) :
  x = 18 ∨ x = 24 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_six_square_greater_144_less_30_l2207_220755


namespace NUMINAMATH_CALUDE_michelle_crayon_count_l2207_220727

/-- The number of crayons in a box of the first type -/
def crayons_in_first_type : ℕ := 5

/-- The number of crayons in a box of the second type -/
def crayons_in_second_type : ℕ := 12

/-- The number of boxes of the first type -/
def boxes_of_first_type : ℕ := 4

/-- The number of boxes of the second type -/
def boxes_of_second_type : ℕ := 3

/-- The number of crayons missing from one box of the first type -/
def missing_crayons : ℕ := 2

/-- The total number of boxes -/
def total_boxes : ℕ := boxes_of_first_type + boxes_of_second_type

theorem michelle_crayon_count : 
  (boxes_of_first_type * crayons_in_first_type - missing_crayons) + 
  (boxes_of_second_type * crayons_in_second_type) = 54 := by
  sorry

#check michelle_crayon_count

end NUMINAMATH_CALUDE_michelle_crayon_count_l2207_220727


namespace NUMINAMATH_CALUDE_total_cost_is_112_l2207_220776

/-- The cost of a spiral notebook before discount -/
def spiral_notebook_cost : ℚ := 15

/-- The cost of a personal planner before discount -/
def personal_planner_cost : ℚ := 10

/-- The discount percentage -/
def discount_percentage : ℚ := 20

/-- The number of spiral notebooks to buy -/
def num_notebooks : ℕ := 4

/-- The number of personal planners to buy -/
def num_planners : ℕ := 8

/-- Calculate the discounted price -/
def apply_discount (price : ℚ) : ℚ :=
  price * (1 - discount_percentage / 100)

/-- Calculate the total cost after discount -/
def total_cost : ℚ :=
  (num_notebooks : ℚ) * apply_discount spiral_notebook_cost +
  (num_planners : ℚ) * apply_discount personal_planner_cost

theorem total_cost_is_112 : total_cost = 112 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_112_l2207_220776


namespace NUMINAMATH_CALUDE_square_area_in_circle_l2207_220736

theorem square_area_in_circle (r : ℝ) (h : r = 10) : 
  let s := r * Real.sqrt 2
  let small_square_side := r / Real.sqrt 2
  let center_distance := s / 2
  2 * center_distance^2 = 100 := by sorry

end NUMINAMATH_CALUDE_square_area_in_circle_l2207_220736


namespace NUMINAMATH_CALUDE_pattern_proof_l2207_220717

theorem pattern_proof (n : ℕ) (h : n > 0) : 
  Real.sqrt (n - n / (n^2 + 1)) = n * Real.sqrt (n / (n^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_pattern_proof_l2207_220717


namespace NUMINAMATH_CALUDE_chosen_number_calculation_l2207_220778

theorem chosen_number_calculation : 
  let chosen_number : ℕ := 208
  let divided_result : ℚ := chosen_number / 2
  let final_result : ℚ := divided_result - 100
  final_result = 4 := by
sorry

end NUMINAMATH_CALUDE_chosen_number_calculation_l2207_220778


namespace NUMINAMATH_CALUDE_initial_customers_count_l2207_220737

/-- The number of customers who left -/
def customers_left : ℕ := 5

/-- The number of customers remaining -/
def customers_remaining : ℕ := 9

/-- The initial number of customers -/
def initial_customers : ℕ := customers_left + customers_remaining

theorem initial_customers_count : initial_customers = 14 := by
  sorry

end NUMINAMATH_CALUDE_initial_customers_count_l2207_220737


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_equals_product_sqrt_main_theorem_l2207_220705

theorem sqrt_product_sqrt_equals_product_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt (a * Real.sqrt b) = Real.sqrt a * Real.sqrt (Real.sqrt b) :=
by sorry

theorem main_theorem : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_equals_product_sqrt_main_theorem_l2207_220705


namespace NUMINAMATH_CALUDE_tangent_point_on_curve_l2207_220783

theorem tangent_point_on_curve (x y : ℝ) : 
  y = x^4 ∧ (4 : ℝ) * x^3 = 4 → x = 1 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_on_curve_l2207_220783


namespace NUMINAMATH_CALUDE_existence_of_100_pairs_l2207_220785

def has_all_digits_at_least_6 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ (n.digits 10) → d ≥ 6

theorem existence_of_100_pairs :
  ∃ S : Finset (ℕ × ℕ),
    S.card = 100 ∧
    (∀ (a b : ℕ), (a, b) ∈ S →
      has_all_digits_at_least_6 a ∧
      has_all_digits_at_least_6 b ∧
      has_all_digits_at_least_6 (a * b)) :=
sorry

end NUMINAMATH_CALUDE_existence_of_100_pairs_l2207_220785


namespace NUMINAMATH_CALUDE_product_not_fifty_l2207_220704

theorem product_not_fifty : ∃! (a b : ℚ), (a = 5 ∧ b = 11) ∧ a * b ≠ 50 ∧
  ((a = 1/2 ∧ b = 100) ∨ (a = -5 ∧ b = -10) ∨ (a = 2 ∧ b = 25) ∨ (a = 5/2 ∧ b = 20)) → a * b = 50 :=
by sorry

end NUMINAMATH_CALUDE_product_not_fifty_l2207_220704


namespace NUMINAMATH_CALUDE_ways_to_choose_all_suits_formula_l2207_220713

/-- The number of ways to choose 13 cards from a 52-card deck such that all four suits are represented -/
def waysToChooseAllSuits : ℕ :=
  Nat.choose 52 13 - 4 * Nat.choose 39 13 + 6 * Nat.choose 26 13 - 4 * Nat.choose 13 13

/-- Theorem stating that the number of ways to choose 13 cards from a 52-card deck
    such that all four suits are represented is equal to the given formula -/
theorem ways_to_choose_all_suits_formula :
  waysToChooseAllSuits =
    Nat.choose 52 13 - 4 * Nat.choose 39 13 + 6 * Nat.choose 26 13 - 4 * Nat.choose 13 13 := by
  sorry

#eval waysToChooseAllSuits

end NUMINAMATH_CALUDE_ways_to_choose_all_suits_formula_l2207_220713


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2207_220798

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2207_220798


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2207_220722

theorem arithmetic_calculations :
  (15 + (-23) - (-10) = 2) ∧
  (-1^2 - (-2)^3 / 4 * (1/4) = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2207_220722


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l2207_220774

-- Define the vectors m and n
def m : Fin 2 → ℝ := ![1, 3]
def n (t : ℝ) : Fin 2 → ℝ := ![2, t]

-- Define the condition for perpendicularity
def perpendicular (t : ℝ) : Prop :=
  (m 0 + n t 0) * (m 0 - n t 0) + (m 1 + n t 1) * (m 1 - n t 1) = 0

-- State the theorem
theorem vector_perpendicular_condition (t : ℝ) :
  perpendicular t → t = Real.sqrt 6 ∨ t = -Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l2207_220774


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_defined_l2207_220751

theorem sqrt_x_minus_one_defined (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_defined_l2207_220751


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l2207_220741

def is_valid_ratio (a r : ℕ+) : Prop :=
  a * r^2 + a * r^4 + a * r^6 = 819 * 6^2016

theorem geometric_progression_ratio :
  ∃ (a : ℕ+), is_valid_ratio a 1 ∧ is_valid_ratio a 2 ∧ is_valid_ratio a 3 ∧ is_valid_ratio a 4 ∧
  ∀ (r : ℕ+), r ≠ 1 ∧ r ≠ 2 ∧ r ≠ 3 ∧ r ≠ 4 → ¬(∃ (b : ℕ+), is_valid_ratio b r) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l2207_220741


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2207_220733

-- Define the sets P and Q
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_P_Q : P ∩ Q = Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2207_220733


namespace NUMINAMATH_CALUDE_inequality_solutions_l2207_220773

theorem inequality_solutions :
  -- Part 1
  (∀ x : ℝ, (3*x - 2)/(x - 1) > 1 ↔ (x > 1 ∨ x < 1/2)) ∧
  -- Part 2
  (∀ a x : ℝ, 
    (a = 0 → x^2 - a*x - 2*a^2 < 0 ↔ False) ∧
    (a > 0 → (x^2 - a*x - 2*a^2 < 0 ↔ -a < x ∧ x < 2*a)) ∧
    (a < 0 → (x^2 - a*x - 2*a^2 < 0 ↔ 2*a < x ∧ x < -a))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l2207_220773


namespace NUMINAMATH_CALUDE_player_A_best_performance_l2207_220782

structure Player where
  name : String
  average_score : Float
  variance : Float

def players : List Player := [
  ⟨"A", 9.9, 4.2⟩,
  ⟨"B", 9.8, 5.2⟩,
  ⟨"C", 9.9, 5.2⟩,
  ⟨"D", 9.0, 4.2⟩
]

def has_best_performance (p : Player) (ps : List Player) : Prop :=
  ∀ q ∈ ps, p.average_score ≥ q.average_score ∧ 
    (p.average_score > q.average_score ∨ p.variance ≤ q.variance)

theorem player_A_best_performance :
  ∃ p ∈ players, p.name = "A" ∧ has_best_performance p players := by
  sorry

end NUMINAMATH_CALUDE_player_A_best_performance_l2207_220782


namespace NUMINAMATH_CALUDE_max_projection_area_is_one_l2207_220731

/-- A tetrahedron with two adjacent isosceles right triangle faces -/
structure Tetrahedron where
  /-- The length of the hypotenuse of the isosceles right triangle faces -/
  hypotenuse : ℝ
  /-- The dihedral angle between the two adjacent isosceles right triangle faces -/
  dihedral_angle : ℝ

/-- The maximum area of the projection of a rotating tetrahedron -/
def max_projection_area (t : Tetrahedron) : ℝ := 1

/-- Theorem stating that the maximum area of the projection is 1 -/
theorem max_projection_area_is_one (t : Tetrahedron) 
  (h1 : t.hypotenuse = 2)
  (h2 : t.dihedral_angle = π / 3) : 
  max_projection_area t = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_projection_area_is_one_l2207_220731


namespace NUMINAMATH_CALUDE_ace_of_hearts_probability_l2207_220796

def standard_deck := 52
def ace_of_hearts_per_deck := 1

theorem ace_of_hearts_probability (combined_deck : ℕ) (ace_of_hearts : ℕ) :
  combined_deck = 2 * standard_deck →
  ace_of_hearts = 2 * ace_of_hearts_per_deck →
  (ace_of_hearts : ℚ) / combined_deck = 1 / 52 :=
by sorry

end NUMINAMATH_CALUDE_ace_of_hearts_probability_l2207_220796


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l2207_220700

theorem quadratic_root_relation : ∀ x₁ x₂ : ℝ, 
  x₁^2 - 12*x₁ + 5 = 0 → 
  x₂^2 - 12*x₂ + 5 = 0 → 
  x₁ + x₂ - x₁*x₂ = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l2207_220700
