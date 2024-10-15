import Mathlib

namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l909_90939

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the condition for the tangent line to be parallel to y = 4x - 1
def tangent_parallel (x : ℝ) : Prop := f' x = 4

-- Define the point P₀
structure Point_P₀ where
  x : ℝ
  y : ℝ
  on_curve : f x = y
  tangent_parallel : tangent_parallel x

-- State the theorem
theorem tangent_point_coordinates :
  ∀ p : Point_P₀, (p.x = 1 ∧ p.y = 0) ∨ (p.x = -1 ∧ p.y = -4) :=
sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l909_90939


namespace NUMINAMATH_CALUDE_cube_root_three_equation_l909_90917

theorem cube_root_three_equation (s : ℝ) : 
  s = 1 / (2 - Real.rpow 3 (1/3)) → 
  s = ((2 + Real.rpow 3 (1/3)) * (4 + Real.sqrt 3)) / 13 := by
sorry

end NUMINAMATH_CALUDE_cube_root_three_equation_l909_90917


namespace NUMINAMATH_CALUDE_profit_difference_maddox_profit_exceeds_theo_by_15_l909_90972

/-- Calculates the profit difference between two sellers of Polaroid cameras. -/
theorem profit_difference (num_cameras : ℕ) (cost_per_camera : ℕ) 
  (maddox_selling_price : ℕ) (theo_selling_price : ℕ) : ℕ :=
  let maddox_profit := num_cameras * maddox_selling_price - num_cameras * cost_per_camera
  let theo_profit := num_cameras * theo_selling_price - num_cameras * cost_per_camera
  maddox_profit - theo_profit

/-- Proves that Maddox made $15 more profit than Theo. -/
theorem maddox_profit_exceeds_theo_by_15 : 
  profit_difference 3 20 28 23 = 15 := by
  sorry

end NUMINAMATH_CALUDE_profit_difference_maddox_profit_exceeds_theo_by_15_l909_90972


namespace NUMINAMATH_CALUDE_prism_sides_plus_two_l909_90952

/-- A prism is a polyhedron with two congruent parallel faces (bases) and rectangular sides. -/
structure Prism where
  sides : ℕ

/-- The number of edges in a prism. -/
def Prism.edges (p : Prism) : ℕ := 3 * p.sides

/-- The number of vertices in a prism. -/
def Prism.vertices (p : Prism) : ℕ := 2 * p.sides

/-- Theorem: For a prism where the sum of its edges and vertices is 30,
    the number of sides plus 2 equals 8. -/
theorem prism_sides_plus_two (p : Prism) 
    (h : p.edges + p.vertices = 30) : p.sides + 2 = 8 := by
  sorry


end NUMINAMATH_CALUDE_prism_sides_plus_two_l909_90952


namespace NUMINAMATH_CALUDE_sum_of_function_values_positive_l909_90984

def is_monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem sum_of_function_values_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h_monotone : is_monotone_increasing f)
  (h_odd : is_odd_function f)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_a3_positive : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_function_values_positive_l909_90984


namespace NUMINAMATH_CALUDE_increasing_function_property_l909_90958

theorem increasing_function_property (f : ℝ → ℝ) (a b : ℝ)
  (h_increasing : ∀ x y : ℝ, x ≤ y → f x ≤ f y) :
  (a + b ≥ 0 ↔ f a + f b ≥ f (-a) + f (-b)) := by
sorry

end NUMINAMATH_CALUDE_increasing_function_property_l909_90958


namespace NUMINAMATH_CALUDE_ideal_function_iff_l909_90983

def IdealFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = 0) ∧
  (∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂)

theorem ideal_function_iff (f : ℝ → ℝ) :
  IdealFunction f ↔
  ((∀ x, f x + f (-x) = 0) ∧
   (∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0)) :=
by sorry

end NUMINAMATH_CALUDE_ideal_function_iff_l909_90983


namespace NUMINAMATH_CALUDE_both_vegan_and_kosher_l909_90948

/-- Represents the meal delivery scenario -/
structure MealDelivery where
  total : ℕ
  vegan : ℕ
  kosher : ℕ
  neither : ℕ

/-- Theorem stating the number of clients needing both vegan and kosher meals -/
theorem both_vegan_and_kosher (m : MealDelivery) 
  (h_total : m.total = 30)
  (h_vegan : m.vegan = 7)
  (h_kosher : m.kosher = 8)
  (h_neither : m.neither = 18) :
  m.total - m.neither - (m.vegan + m.kosher - (m.total - m.neither)) = 3 := by
  sorry

#check both_vegan_and_kosher

end NUMINAMATH_CALUDE_both_vegan_and_kosher_l909_90948


namespace NUMINAMATH_CALUDE_routes_on_3x2_grid_l909_90945

/-- The number of routes on a grid from (0, m) to (n, 0) moving only right or down -/
def num_routes (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- The dimensions of the grid -/
def grid_width : ℕ := 3
def grid_height : ℕ := 2

theorem routes_on_3x2_grid :
  num_routes grid_height grid_width = Nat.choose (grid_height + grid_width) grid_height :=
by sorry

end NUMINAMATH_CALUDE_routes_on_3x2_grid_l909_90945


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l909_90999

theorem negative_fractions_comparison : -1/3 < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l909_90999


namespace NUMINAMATH_CALUDE_bridge_length_problem_l909_90982

/-- The length of a bridge crossed by a man walking at a given speed in a given time -/
def bridge_length (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A man walking at 10 km/hr crosses a bridge in 18 minutes. The bridge length is 3 km. -/
theorem bridge_length_problem :
  let walking_speed : ℝ := 10  -- km/hr
  let crossing_time : ℝ := 18 / 60  -- 18 minutes converted to hours
  bridge_length walking_speed crossing_time = 3 := by
sorry


end NUMINAMATH_CALUDE_bridge_length_problem_l909_90982


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l909_90928

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 2 * Nat.factorial 5 = 36120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l909_90928


namespace NUMINAMATH_CALUDE_gcd_of_779_209_589_l909_90953

theorem gcd_of_779_209_589 : Nat.gcd 779 (Nat.gcd 209 589) = 19 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_779_209_589_l909_90953


namespace NUMINAMATH_CALUDE_school_weeks_l909_90929

/-- Proves that the number of school weeks is 36 given the conditions --/
theorem school_weeks (sandwiches_per_week : ℕ) (missed_days : ℕ) (total_sandwiches : ℕ) 
  (h1 : sandwiches_per_week = 2)
  (h2 : missed_days = 3)
  (h3 : total_sandwiches = 69) : 
  (total_sandwiches + missed_days) / sandwiches_per_week = 36 := by
  sorry

#check school_weeks

end NUMINAMATH_CALUDE_school_weeks_l909_90929


namespace NUMINAMATH_CALUDE_largest_divisor_of_m_squared_minus_4n_squared_l909_90987

theorem largest_divisor_of_m_squared_minus_4n_squared (m n : ℤ) 
  (h_m_odd : Odd m) (h_n_odd : Odd n) (h_m_gt_n : m > n) : 
  (∀ k : ℤ, k ∣ (m^2 - 4*n^2) → k = 1 ∨ k = -1) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_m_squared_minus_4n_squared_l909_90987


namespace NUMINAMATH_CALUDE_first_podium_height_calculation_l909_90903

/-- The height of the second prize podium in centimeters -/
def second_podium_height : ℚ := 53 + 7 / 10

/-- Hyeonjoo's measured height on the second prize podium in centimeters -/
def height_on_second_podium : ℚ := 190

/-- Hyeonjoo's measured height on the first prize podium in centimeters -/
def height_on_first_podium : ℚ := 232 + 5 / 10

/-- The height of the first prize podium in centimeters -/
def first_podium_height : ℚ := height_on_first_podium - (height_on_second_podium - second_podium_height)

theorem first_podium_height_calculation :
  first_podium_height = 96.2 := by sorry

end NUMINAMATH_CALUDE_first_podium_height_calculation_l909_90903


namespace NUMINAMATH_CALUDE_counterexample_exists_l909_90938

theorem counterexample_exists : ∃ n : ℕ, 15 ≤ n ∧ n ≤ 30 ∧ ¬(Nat.Prime n) ∧ Nat.Prime (n - 5) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l909_90938


namespace NUMINAMATH_CALUDE_power_product_inequality_l909_90931

/-- Given positive real numbers a, b, and c, 
    a^a * b^b * c^c ≥ (a * b * c)^((a+b+c)/3) -/
theorem power_product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c) / 3) := by
  sorry

end NUMINAMATH_CALUDE_power_product_inequality_l909_90931


namespace NUMINAMATH_CALUDE_sum_properties_l909_90981

theorem sum_properties (a b : ℤ) (ha : 6 ∣ a) (hb : 9 ∣ b) : 
  (∃ x y : ℤ, a + b = 2*x + 1 ∧ a + b = 2*y) ∧ 
  (∃ z : ℤ, a + b = 6*z + 3) ∧ 
  (∃ w : ℤ, a + b = 9*w + 3) ∧ 
  (∃ v : ℤ, a + b = 9*v) :=
by sorry

end NUMINAMATH_CALUDE_sum_properties_l909_90981


namespace NUMINAMATH_CALUDE_triangle_side_length_l909_90976

noncomputable section

/-- Given a triangle ABC with angles A, B, C and opposite sides a, b, c respectively,
    if A = 30°, B = 45°, and a = √2, then b = 2 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π/6 → B = π/4 → a = Real.sqrt 2 → 
  (a / Real.sin A = b / Real.sin B) → 
  b = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l909_90976


namespace NUMINAMATH_CALUDE_discounted_good_price_l909_90992

/-- The price of a good after applying successive discounts -/
def discounted_price (initial_price : ℝ) : ℝ :=
  initial_price * 0.75 * 0.85 * 0.90 * 0.93

theorem discounted_good_price (P : ℝ) :
  discounted_price P = 6600 → P = 11118.75 := by
  sorry

end NUMINAMATH_CALUDE_discounted_good_price_l909_90992


namespace NUMINAMATH_CALUDE_matrix_power_50_l909_90912

theorem matrix_power_50 (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A = ![![1, 1], ![0, 1]] →
  A ^ 50 = ![![1, 50], ![0, 1]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_50_l909_90912


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l909_90968

/-- Given points A, B, C, and D (midpoint of AB), prove that the sum of 
    the slope and y-intercept of the line passing through C and D is 27/5 -/
theorem slope_intercept_sum (A B C D : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 2) →
  C = (10, 0) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := D.2
  m + b = 27 / 5 := by
  sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_l909_90968


namespace NUMINAMATH_CALUDE_exists_solution_l909_90921

theorem exists_solution : ∃ (a b c d : ℕ+), 2014 = (a^2 + b^2) * (c^3 - d^3) := by
  sorry

end NUMINAMATH_CALUDE_exists_solution_l909_90921


namespace NUMINAMATH_CALUDE_rectangle_max_area_l909_90998

/-- Given a rectangle with sides a and b and perimeter p, 
    the area is maximized when the rectangle is a square. -/
theorem rectangle_max_area (a b p : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_perimeter : p = 2 * (a + b)) :
  ∃ (max_area : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → 2 * (x + y) = p → x * y ≤ max_area ∧ 
  (x * y = max_area ↔ x = y) :=
sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l909_90998


namespace NUMINAMATH_CALUDE_system_solution_unique_l909_90988

theorem system_solution_unique (x y : ℚ) : 
  (2 * x - 3 * y = 1) ∧ ((2 + x) / 3 = (y + 1) / 4) ↔ (x = -3 ∧ y = -7/3) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_unique_l909_90988


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_l909_90961

theorem floor_plus_self_eq (r : ℝ) : ⌊r⌋ + r = 12.4 ↔ r = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_l909_90961


namespace NUMINAMATH_CALUDE_imaginary_part_of_fraction_l909_90980

open Complex

theorem imaginary_part_of_fraction (i : ℂ) (h : i * i = -1) :
  (((1 : ℂ) + i) / ((1 : ℂ) - i)).im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_fraction_l909_90980


namespace NUMINAMATH_CALUDE_simple_interest_principal_l909_90946

/-- Simple interest calculation -/
theorem simple_interest_principal
  (rate : ℚ)
  (time : ℚ)
  (interest : ℚ)
  (h_rate : rate = 4 / 100)
  (h_time : time = 1)
  (h_interest : interest = 400) :
  interest = (10000 : ℚ) * rate * time :=
sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l909_90946


namespace NUMINAMATH_CALUDE_min_value_of_f_fourth_composition_l909_90997

/-- The function f(x) = x^2 + 6x + 7 -/
def f (x : ℝ) : ℝ := x^2 + 6*x + 7

/-- The statement that the minimum value of f(f(f(f(x)))) over all real x is 23 -/
theorem min_value_of_f_fourth_composition :
  ∀ x : ℝ, f (f (f (f x))) ≥ 23 ∧ ∃ y : ℝ, f (f (f (f y))) = 23 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_fourth_composition_l909_90997


namespace NUMINAMATH_CALUDE_fourth_grade_students_l909_90979

theorem fourth_grade_students (initial_students leaving_students new_students : ℕ) :
  initial_students = 35 →
  leaving_students = 10 →
  new_students = 10 →
  initial_students - leaving_students + new_students = 35 := by
sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l909_90979


namespace NUMINAMATH_CALUDE_factorization_problems_l909_90914

theorem factorization_problems (x y : ℝ) : 
  (x^2 - 6*x + 9 = (x - 3)^2) ∧ 
  (x^2*(y - 2) - 4*(y - 2) = (y - 2)*(x + 2)*(x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l909_90914


namespace NUMINAMATH_CALUDE_largest_multiple_proof_l909_90919

/-- The largest three-digit number that is divisible by 6, 5, 8, and 9 -/
def largest_multiple : ℕ := 720

theorem largest_multiple_proof :
  (∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 6 ∣ n ∧ 5 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n → n ≤ largest_multiple) ∧
  100 ≤ largest_multiple ∧
  largest_multiple < 1000 ∧
  6 ∣ largest_multiple ∧
  5 ∣ largest_multiple ∧
  8 ∣ largest_multiple ∧
  9 ∣ largest_multiple :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_proof_l909_90919


namespace NUMINAMATH_CALUDE_samuel_spent_one_fifth_l909_90930

theorem samuel_spent_one_fifth (total : ℕ) (samuel_initial : ℚ) (samuel_left : ℕ) : 
  total = 240 →
  samuel_initial = 3/4 * total →
  samuel_left = 132 →
  (samuel_initial - samuel_left : ℚ) / total = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_samuel_spent_one_fifth_l909_90930


namespace NUMINAMATH_CALUDE_fred_gave_233_marbles_l909_90947

/-- The number of black marbles Fred gave to Sara -/
def marbles_from_fred (initial_marbles final_marbles : ℕ) : ℕ :=
  final_marbles - initial_marbles

/-- Theorem stating that Fred gave Sara 233 black marbles -/
theorem fred_gave_233_marbles :
  let initial_marbles : ℕ := 792
  let final_marbles : ℕ := 1025
  marbles_from_fred initial_marbles final_marbles = 233 := by
  sorry

end NUMINAMATH_CALUDE_fred_gave_233_marbles_l909_90947


namespace NUMINAMATH_CALUDE_line_L_equation_l909_90994

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := 3 * x + 2 * y - 1 = 0
def l₂ (x y : ℝ) : Prop := 5 * x + 2 * y + 1 = 0
def l₃ (x y : ℝ) : Prop := 3 * x - 5 * y + 6 = 0

-- Define the intersection point of l₁ and l₂
def intersection_point : ℝ × ℝ := (-1, 2)

-- Define line L
def L (x y : ℝ) : Prop := 5 * x + 3 * y - 1 = 0

-- Theorem statement
theorem line_L_equation :
  (∀ x y : ℝ, L x y ↔ 
    (x = intersection_point.1 ∧ y = intersection_point.2 ∨
    ∃ t : ℝ, x = intersection_point.1 + t ∧ y = intersection_point.2 - (5/3) * t)) ∧
  (∀ x y : ℝ, L x y → l₃ x y → 
    (x - intersection_point.1) * 3 + (y - intersection_point.2) * (-5) = 0) :=
by sorry


end NUMINAMATH_CALUDE_line_L_equation_l909_90994


namespace NUMINAMATH_CALUDE_width_of_sum_l909_90993

/-- A convex curve in a 2D plane -/
structure ConvexCurve where
  -- Add necessary fields here
  
/-- The width of a convex curve in a given direction -/
def width (K : ConvexCurve) (direction : ℝ × ℝ) : ℝ :=
  sorry

/-- The sum of two convex curves -/
def curve_sum (K₁ K₂ : ConvexCurve) : ConvexCurve :=
  sorry

/-- Theorem: The width of the sum of two convex curves is the sum of their individual widths -/
theorem width_of_sum (K₁ K₂ : ConvexCurve) (direction : ℝ × ℝ) :
  width (curve_sum K₁ K₂) direction = width K₁ direction + width K₂ direction :=
sorry

end NUMINAMATH_CALUDE_width_of_sum_l909_90993


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l909_90996

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l909_90996


namespace NUMINAMATH_CALUDE_dot_only_count_l909_90909

/-- Represents an alphabet with dots and straight lines -/
structure Alphabet where
  total : ℕ
  both : ℕ
  line_only : ℕ
  has_dot_or_line : total = both + line_only + (total - (both + line_only))

/-- The number of letters with a dot but no straight line in the given alphabet -/
def letters_with_dot_only (α : Alphabet) : ℕ :=
  α.total - (α.both + α.line_only)

/-- Theorem stating the number of letters with a dot but no straight line -/
theorem dot_only_count (α : Alphabet) 
  (h1 : α.total = 80)
  (h2 : α.both = 28)
  (h3 : α.line_only = 47) :
  letters_with_dot_only α = 5 := by
  sorry

#check dot_only_count

end NUMINAMATH_CALUDE_dot_only_count_l909_90909


namespace NUMINAMATH_CALUDE_investment_interest_rate_proof_l909_90966

theorem investment_interest_rate_proof 
  (total_investment : ℝ)
  (first_part : ℝ)
  (first_rate : ℝ)
  (total_interest : ℝ)
  (h1 : total_investment = 3500)
  (h2 : first_part = 1549.9999999999998)
  (h3 : first_rate = 3)
  (h4 : total_interest = 144)
  (h5 : first_part * (first_rate / 100) + (total_investment - first_part) * (second_rate / 100) = total_interest) :
  second_rate = 5 := by
  sorry


end NUMINAMATH_CALUDE_investment_interest_rate_proof_l909_90966


namespace NUMINAMATH_CALUDE_pen_cost_is_30_l909_90923

-- Define the daily expenditures
def daily_expenditures : List ℝ := [450, 600, 400, 500, 550, 300]

-- Define the mean expenditure
def mean_expenditure : ℝ := 500

-- Define the number of days
def num_days : ℕ := 7

-- Define the cost of the notebook
def notebook_cost : ℝ := 50

-- Define the cost of the earphone
def earphone_cost : ℝ := 620

-- Theorem to prove
theorem pen_cost_is_30 :
  let total_week_expenditure := mean_expenditure * num_days
  let total_other_days := daily_expenditures.sum
  let friday_expenditure := total_week_expenditure - total_other_days
  friday_expenditure - (notebook_cost + earphone_cost) = 30 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_is_30_l909_90923


namespace NUMINAMATH_CALUDE_sum_of_first_50_digits_l909_90954

/-- The decimal expansion of 1/10101 -/
def decimal_expansion : ℕ → ℕ
| n => match n % 5 with
  | 0 => 0
  | 1 => 0
  | 2 => 0
  | 3 => 9
  | 4 => 9
  | _ => 0  -- This case is technically unreachable

/-- Sum of the first n digits in the decimal expansion -/
def sum_of_digits (n : ℕ) : ℕ :=
  (List.range n).map decimal_expansion |>.sum

/-- The theorem stating the sum of the first 50 digits after the decimal point in 1/10101 -/
theorem sum_of_first_50_digits :
  sum_of_digits 50 = 180 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_50_digits_l909_90954


namespace NUMINAMATH_CALUDE_mike_total_games_l909_90916

/-- The total number of basketball games Mike attended over two years -/
def total_games (games_this_year games_last_year : ℕ) : ℕ :=
  games_this_year + games_last_year

/-- Theorem stating that Mike attended 54 games in total -/
theorem mike_total_games : 
  total_games 15 39 = 54 := by
  sorry

end NUMINAMATH_CALUDE_mike_total_games_l909_90916


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_8_mod_17_l909_90942

theorem least_five_digit_congruent_to_8_mod_17 : 
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n ≡ 8 [ZMOD 17] → n ≥ 10009 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_8_mod_17_l909_90942


namespace NUMINAMATH_CALUDE_prime_solution_equation_l909_90967

theorem prime_solution_equation : ∃! (p q : ℕ), 
  Prime p ∧ Prime q ∧ p^2 - 6*p*q + q^2 + 3*q - 1 = 0 ∧ p = 17 ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_solution_equation_l909_90967


namespace NUMINAMATH_CALUDE_complex_equation_solution_l909_90907

theorem complex_equation_solution (b : ℝ) : (1 + b * Complex.I) * Complex.I = 1 + Complex.I → b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l909_90907


namespace NUMINAMATH_CALUDE_sum_reciprocals_zero_implies_sum_diff_zero_l909_90908

theorem sum_reciprocals_zero_implies_sum_diff_zero 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (h : 1 / (a + 1) + 1 / (a - 1) + 1 / (b + 1) + 1 / (b - 1) = 0) : 
  a - 1 / a + b - 1 / b = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_zero_implies_sum_diff_zero_l909_90908


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_example_l909_90918

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : 0 < std_dev

/-- The value that is exactly 2 standard deviations less than the mean -/
def two_std_dev_below_mean (d : NormalDistribution) : ℝ :=
  d.mean - 2 * d.std_dev

/-- Theorem: For a normal distribution with mean 15.5 and standard deviation 1.5,
    the value that is exactly 2 standard deviations less than the mean is 12.5 -/
theorem two_std_dev_below_mean_example :
  let d : NormalDistribution := ⟨15.5, 1.5, by norm_num⟩
  two_std_dev_below_mean d = 12.5 := by sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_example_l909_90918


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l909_90936

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 10 = 9 ∧
  (∀ m : ℕ, m > 0 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 10 = 9 → n ≤ m) ∧
  n = 59 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l909_90936


namespace NUMINAMATH_CALUDE_number_of_divisors_sum_of_divisors_l909_90937

def n : ℕ := 120

-- Number of positive divisors
theorem number_of_divisors : (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 16 := by sorry

-- Sum of positive divisors
theorem sum_of_divisors : (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id = 3240 := by sorry

end NUMINAMATH_CALUDE_number_of_divisors_sum_of_divisors_l909_90937


namespace NUMINAMATH_CALUDE_money_distribution_l909_90957

/-- Given three people A, B, and C with some money, prove that A and C together have 300 Rs. -/
theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 400)
  (bc_sum : B + C = 150)
  (c_amount : C = 50) : 
  A + C = 300 := by sorry

end NUMINAMATH_CALUDE_money_distribution_l909_90957


namespace NUMINAMATH_CALUDE_solve_equation_l909_90900

theorem solve_equation (y : ℝ) : 3/4 + 1/y = 7/8 ↔ y = 8 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l909_90900


namespace NUMINAMATH_CALUDE_equation_solution_l909_90926

theorem equation_solution : ∃ x : ℝ, x = 25 ∧ Real.sqrt (1 + Real.sqrt (2 + x^2)) = (3 + Real.sqrt x) ^ (1/3 : ℝ) :=
  sorry

end NUMINAMATH_CALUDE_equation_solution_l909_90926


namespace NUMINAMATH_CALUDE_additional_workers_needed_l909_90941

/-- Represents the problem of calculating additional workers needed to complete a construction project on time -/
theorem additional_workers_needed
  (total_days : ℕ) 
  (initial_workers : ℕ) 
  (days_passed : ℕ) 
  (work_completed : ℚ) 
  (h1 : total_days = 50)
  (h2 : initial_workers = 20)
  (h3 : days_passed = 25)
  (h4 : work_completed = 2/5)
  : ℕ := by
  sorry

#check additional_workers_needed

end NUMINAMATH_CALUDE_additional_workers_needed_l909_90941


namespace NUMINAMATH_CALUDE_mike_total_cards_l909_90995

def initial_cards : ℕ := 87
def received_cards : ℕ := 13

theorem mike_total_cards : initial_cards + received_cards = 100 := by
  sorry

end NUMINAMATH_CALUDE_mike_total_cards_l909_90995


namespace NUMINAMATH_CALUDE_broomstick_race_orderings_l909_90933

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of competitors in the race -/
def num_competitors : ℕ := 4

theorem broomstick_race_orderings : 
  permutations num_competitors = 24 := by
  sorry

end NUMINAMATH_CALUDE_broomstick_race_orderings_l909_90933


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l909_90971

theorem price_reduction_percentage (initial_price final_price : ℝ) 
  (h1 : initial_price = 25)
  (h2 : final_price = 16)
  (h3 : ∃ x : ℝ, initial_price * (1 - x)^2 = final_price ∧ 0 < x ∧ x < 1) :
  ∃ x : ℝ, initial_price * (1 - x)^2 = final_price ∧ x = 0.2 :=
sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l909_90971


namespace NUMINAMATH_CALUDE_triangle_side_length_l909_90949

theorem triangle_side_length (a b c : ℝ) (R : ℝ) :
  a > 0 → b > 0 → c > 0 → R > 0 →
  (a^2 / (b * c)) - (c / b) - (b / c) = Real.sqrt 3 →
  R = 3 →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l909_90949


namespace NUMINAMATH_CALUDE_women_in_salem_l909_90977

def leesburg_population : ℕ := 58940
def salem_population_multiplier : ℕ := 15
def people_moved_out : ℕ := 130000

def salem_original_population : ℕ := leesburg_population * salem_population_multiplier
def salem_current_population : ℕ := salem_original_population - people_moved_out

theorem women_in_salem : 
  (salem_current_population / 2 : ℕ) = 377050 := by sorry

end NUMINAMATH_CALUDE_women_in_salem_l909_90977


namespace NUMINAMATH_CALUDE_candy_distribution_l909_90969

theorem candy_distribution (hugh tommy melany lily : ℝ) 
  (h_hugh : hugh = 8.5)
  (h_tommy : tommy = 6.75)
  (h_melany : melany = 7.25)
  (h_lily : lily = 5.5) :
  let total := hugh + tommy + melany + lily
  let num_people := 4
  (total / num_people) = 7 := by sorry

end NUMINAMATH_CALUDE_candy_distribution_l909_90969


namespace NUMINAMATH_CALUDE_tan_alpha_eq_neg_five_l909_90944

theorem tan_alpha_eq_neg_five (α : ℝ) :
  (Real.cos (π / 2 - α) - 3 * Real.cos α) / (Real.sin α - Real.cos (π + α)) = 2 →
  Real.tan α = -5 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_eq_neg_five_l909_90944


namespace NUMINAMATH_CALUDE_rohans_salary_l909_90964

/-- Rohan's monthly salary in rupees -/
def monthly_salary : ℝ := 7500

/-- Percentage of salary spent on food -/
def food_percentage : ℝ := 40

/-- Percentage of salary spent on house rent -/
def rent_percentage : ℝ := 20

/-- Percentage of salary spent on entertainment -/
def entertainment_percentage : ℝ := 10

/-- Percentage of salary spent on conveyance -/
def conveyance_percentage : ℝ := 10

/-- Rohan's savings at the end of the month in rupees -/
def savings : ℝ := 1500

theorem rohans_salary :
  monthly_salary = 7500 ∧
  food_percentage + rent_percentage + entertainment_percentage + conveyance_percentage + (savings / monthly_salary * 100) = 100 :=
by sorry

end NUMINAMATH_CALUDE_rohans_salary_l909_90964


namespace NUMINAMATH_CALUDE_bake_sale_total_l909_90910

theorem bake_sale_total (cookies : ℕ) (brownies : ℕ) : 
  cookies = 48 → 
  brownies * 6 = cookies * 7 →
  cookies + brownies = 104 := by
sorry

end NUMINAMATH_CALUDE_bake_sale_total_l909_90910


namespace NUMINAMATH_CALUDE_A_intersection_B_eq_A_l909_90986

def A (k : ℝ) : Set ℝ := {x | k + 1 ≤ x ∧ x ≤ 2 * k}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem A_intersection_B_eq_A (k : ℝ) : A k ∩ B = A k ↔ k ∈ Set.Iic (3/2) :=
sorry

end NUMINAMATH_CALUDE_A_intersection_B_eq_A_l909_90986


namespace NUMINAMATH_CALUDE_proper_subset_of_singleton_l909_90925

-- Define the set P
def P : Set ℕ := {0}

-- State the theorem
theorem proper_subset_of_singleton :
  ∀ (S : Set ℕ), S ⊂ P → S = ∅ :=
sorry

end NUMINAMATH_CALUDE_proper_subset_of_singleton_l909_90925


namespace NUMINAMATH_CALUDE_second_point_x_coordinate_l909_90989

/-- Given two points (m, n) and (m + 2, n + 1) on the line x = 2y + 3,
    the x-coordinate of the second point is m + 2. -/
theorem second_point_x_coordinate (m n : ℝ) : 
  (m = 2 * n + 3) → -- First point (m, n) lies on the line
  (m + 2 = 2 * (n + 1) + 3) → -- Second point (m + 2, n + 1) lies on the line
  (m + 2 = m + 2) -- The x-coordinate of the second point is m + 2
:= by sorry

end NUMINAMATH_CALUDE_second_point_x_coordinate_l909_90989


namespace NUMINAMATH_CALUDE_subset_condition_l909_90940

def A : Set ℝ := {x | x < -1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | a * x + 1 ≤ 0}

theorem subset_condition (a : ℝ) : B a ⊆ A ↔ -1/3 ≤ a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l909_90940


namespace NUMINAMATH_CALUDE_tablet_screen_area_difference_l909_90960

theorem tablet_screen_area_difference : 
  let diagonal_8 : ℝ := 8
  let diagonal_7 : ℝ := 7
  let area_8 : ℝ := (diagonal_8^2) / 2
  let area_7 : ℝ := (diagonal_7^2) / 2
  area_8 - area_7 = 7.5 := by sorry

end NUMINAMATH_CALUDE_tablet_screen_area_difference_l909_90960


namespace NUMINAMATH_CALUDE_line_plane_parallelism_condition_l909_90991

-- Define the concepts of line and plane
variable (m : Line) (α : Plane)

-- Define what it means for a line to be parallel to a plane
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Define what it means for a line to be parallel to countless lines in a plane
def line_parallel_to_countless_lines_in_plane (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem line_plane_parallelism_condition :
  (line_parallel_to_countless_lines_in_plane m α → line_parallel_to_plane m α) ∧
  ¬(line_parallel_to_plane m α → line_parallel_to_countless_lines_in_plane m α) := by sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_condition_l909_90991


namespace NUMINAMATH_CALUDE_equation_solution_l909_90913

theorem equation_solution :
  ∃ k : ℚ, (3 * k - 4) / (k + 7) = 2 / 5 ↔ k = 34 / 13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l909_90913


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solutions_equation_four_solution_l909_90935

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  (x + 1)^2 - 144 = 0 ↔ x = -13 ∨ x = 11 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  x^2 - 4*x - 32 = 0 ↔ x = 8 ∨ x = -4 := by sorry

-- Equation 3
theorem equation_three_solutions (x : ℝ) :
  3*(x - 2)^2 = x*(x - 2) ↔ x = 2 ∨ x = 3 := by sorry

-- Equation 4
theorem equation_four_solution (x : ℝ) :
  (x + 3)^2 = 2*x + 5 ↔ x = -2 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solutions_equation_four_solution_l909_90935


namespace NUMINAMATH_CALUDE_investment_amount_l909_90978

/-- Calculates the investment amount given the dividend received and share details --/
def calculate_investment (share_value : ℕ) (premium_percentage : ℕ) (dividend_percentage : ℕ) (dividend_received : ℕ) : ℕ :=
  let premium_factor := 1 + premium_percentage / 100
  let share_price := share_value * premium_factor
  let dividend_per_share := share_value * dividend_percentage / 100
  let num_shares := dividend_received / dividend_per_share
  num_shares * share_price

/-- Proves that the investment amount is 14375 given the problem conditions --/
theorem investment_amount : calculate_investment 100 25 5 576 = 14375 := by
  sorry

#eval calculate_investment 100 25 5 576

end NUMINAMATH_CALUDE_investment_amount_l909_90978


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l909_90965

theorem simplify_trig_expression (h : π / 2 < 2 ∧ 2 < π) :
  2 * Real.sqrt (1 + Real.sin 4) + Real.sqrt (2 + 2 * Real.cos 4) = 2 * Real.sin 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l909_90965


namespace NUMINAMATH_CALUDE_min_120_degree_turns_l909_90975

/-- A triangular graph representing a city --/
structure TriangularCity where
  /-- The number of triangular blocks in the city --/
  blocks : Nat
  /-- The number of intersections (squares) in the city --/
  intersections : Nat
  /-- The path taken by the tourist --/
  tourist_path : List Nat
  /-- Ensures the number of blocks is 16 --/
  blocks_count : blocks = 16
  /-- Ensures the number of intersections is 15 --/
  intersections_count : intersections = 15
  /-- Ensures the tourist visits each intersection exactly once --/
  path_visits_all_once : tourist_path.length = intersections ∧ tourist_path.Nodup

/-- The number of 120° turns in a given path --/
def count_120_degree_turns (path : List Nat) : Nat :=
  sorry

/-- Theorem stating that a tourist in a triangular city must make at least 4 turns of 120° --/
theorem min_120_degree_turns (city : TriangularCity) :
  count_120_degree_turns city.tourist_path ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_120_degree_turns_l909_90975


namespace NUMINAMATH_CALUDE_perpendicular_line_slope_OA_longer_than_OB_l909_90934

/-- The ellipse C with equation x² + y²/4 = 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2/4 = 1}

/-- The line y = kx + 1 for a given k -/
def Line (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + 1}

/-- A and B are the intersection points of C and the line -/
def A (k : ℝ) : ℝ × ℝ := sorry
def B (k : ℝ) : ℝ × ℝ := sorry

/-- Condition that A is in the first quadrant -/
def A_in_first_quadrant (k : ℝ) : Prop := (A k).1 > 0 ∧ (A k).2 > 0

theorem perpendicular_line_slope (k : ℝ) :
  (A k).1 * (B k).1 + (A k).2 * (B k).2 = 0 → k = 1/2 ∨ k = -1/2 := sorry

theorem OA_longer_than_OB (k : ℝ) :
  k > 0 → A_in_first_quadrant k →
  (A k).1^2 + (A k).2^2 > (B k).1^2 + (B k).2^2 := sorry

end NUMINAMATH_CALUDE_perpendicular_line_slope_OA_longer_than_OB_l909_90934


namespace NUMINAMATH_CALUDE_gcf_of_lcms_l909_90905

theorem gcf_of_lcms : Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_l909_90905


namespace NUMINAMATH_CALUDE_decreasing_power_function_l909_90911

/-- A power function y = ax^b is decreasing on (0, +∞) if and only if b = -3 -/
theorem decreasing_power_function (a : ℝ) (b : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → a * x₁^b > a * x₂^b) ↔ b = -3 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_power_function_l909_90911


namespace NUMINAMATH_CALUDE_four_students_in_all_activities_l909_90956

/-- The number of students participating in all three activities in a summer camp. -/
def students_in_all_activities (total_students : ℕ) 
  (swimming_students : ℕ) (archery_students : ℕ) (chess_students : ℕ) 
  (at_least_two_activities : ℕ) : ℕ :=
  let a := swimming_students + archery_students + chess_students - at_least_two_activities - total_students
  a

/-- Theorem stating that 4 students participate in all three activities. -/
theorem four_students_in_all_activities : 
  students_in_all_activities 25 15 17 10 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_students_in_all_activities_l909_90956


namespace NUMINAMATH_CALUDE_expected_intersection_value_l909_90924

/-- A subset of consecutive integers from {1,2,3,4,5,6,7,8} -/
def ConsecutiveSubset := List ℕ

/-- The set of all possible consecutive subsets -/
def allSubsets : Finset ConsecutiveSubset :=
  sorry

/-- The probability of an element x being in a randomly chosen subset -/
def P (x : ℕ) : ℚ :=
  sorry

/-- The expected number of elements in the intersection of three independently chosen subsets -/
def expectedIntersection : ℚ :=
  sorry

theorem expected_intersection_value :
  expectedIntersection = 178 / 243 := by
  sorry

end NUMINAMATH_CALUDE_expected_intersection_value_l909_90924


namespace NUMINAMATH_CALUDE_area_of_inscribing_square_l909_90902

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 2*x + y^2 - 4*y = 12

/-- The circle is inscribed in a square with sides parallel to y-axis -/
axiom inscribed_in_square : ∃ (side : ℝ), ∀ (x y : ℝ), 
  circle_equation x y → (0 ≤ x ∧ x ≤ side) ∧ (0 ≤ y ∧ y ≤ side)

/-- The area of the square inscribing the circle -/
def square_area : ℝ := 68

/-- Theorem: The area of the square inscribing the circle is 68 square units -/
theorem area_of_inscribing_square : 
  ∃ (side : ℝ), (∀ (x y : ℝ), circle_equation x y → 
    (0 ≤ x ∧ x ≤ side) ∧ (0 ≤ y ∧ y ≤ side)) ∧ side^2 = square_area := by
  sorry

end NUMINAMATH_CALUDE_area_of_inscribing_square_l909_90902


namespace NUMINAMATH_CALUDE_card_statements_l909_90974

/-- Represents the number of true statements on the card -/
def TrueStatements : Nat → Prop
  | 0 => True
  | 1 => False
  | 2 => False
  | 3 => False
  | 4 => False
  | 5 => False
  | _ => False

/-- The five statements on the card -/
def Statement : Nat → Prop
  | 1 => TrueStatements 1
  | 2 => TrueStatements 2
  | 3 => TrueStatements 3
  | 4 => TrueStatements 4
  | 5 => TrueStatements 5
  | _ => False

/-- Theorem stating that the number of true statements is 0 -/
theorem card_statements :
  (∀ n : Nat, Statement n ↔ TrueStatements n) →
  TrueStatements 0 := by
  sorry

end NUMINAMATH_CALUDE_card_statements_l909_90974


namespace NUMINAMATH_CALUDE_negative_three_star_five_l909_90990

-- Define the operation *
def star (a b : ℚ) : ℚ := (a - 2*b) / (2*a - b)

-- Theorem statement
theorem negative_three_star_five :
  star (-3) 5 = 13/11 := by sorry

end NUMINAMATH_CALUDE_negative_three_star_five_l909_90990


namespace NUMINAMATH_CALUDE_prob_zero_or_one_white_is_four_fifths_l909_90943

def total_balls : ℕ := 6
def red_balls : ℕ := 4
def white_balls : ℕ := 2
def selected_balls : ℕ := 3

def prob_zero_or_one_white (total : ℕ) (red : ℕ) (white : ℕ) (selected : ℕ) : ℚ :=
  (Nat.choose red selected + Nat.choose white 1 * Nat.choose red (selected - 1)) /
  Nat.choose total selected

theorem prob_zero_or_one_white_is_four_fifths :
  prob_zero_or_one_white total_balls red_balls white_balls selected_balls = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_zero_or_one_white_is_four_fifths_l909_90943


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l909_90951

/-- 
Given two cylinders with the same height and radii in the ratio 1:3,
if the volume of the larger cylinder is 360 cc, then the volume of the smaller cylinder is 40 cc.
-/
theorem cylinder_volume_ratio (h : ℝ) (r : ℝ) : 
  h > 0 → r > 0 → π * (3 * r)^2 * h = 360 → π * r^2 * h = 40 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l909_90951


namespace NUMINAMATH_CALUDE_increasing_function_property_l909_90922

-- Define an increasing function on the real line
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- State the theorem
theorem increasing_function_property (f : ℝ → ℝ) (h : IncreasingFunction f) :
  (∀ a b : ℝ, a + b ≥ 0 → f a + f b ≥ f (-a) + f (-b)) ∧
  (∀ a b : ℝ, f a + f b ≥ f (-a) + f (-b) → a + b ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_property_l909_90922


namespace NUMINAMATH_CALUDE_largest_circle_area_l909_90963

theorem largest_circle_area (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) -- Right triangle condition
  (h5 : π * (a/2)^2 + π * (b/2)^2 + π * (c/2)^2 = 338 * π) : -- Sum of circle areas
  π * (c/2)^2 = 169 * π := by
sorry

end NUMINAMATH_CALUDE_largest_circle_area_l909_90963


namespace NUMINAMATH_CALUDE_weight_difference_l909_90950

theorem weight_difference (rachel jimmy adam : ℝ) 
  (h1 : rachel = 75)
  (h2 : rachel < jimmy)
  (h3 : rachel = adam + 15)
  (h4 : (rachel + jimmy + adam) / 3 = 72) :
  jimmy - rachel = 6 := by
sorry

end NUMINAMATH_CALUDE_weight_difference_l909_90950


namespace NUMINAMATH_CALUDE_no_positive_integer_triples_l909_90901

theorem no_positive_integer_triples : 
  ¬∃ (a b c : ℕ+), (Nat.factorial a.val + b.val^3 = 18 + c.val^3) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_triples_l909_90901


namespace NUMINAMATH_CALUDE_arithmetic_computation_l909_90955

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 6 * 5 / 3 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l909_90955


namespace NUMINAMATH_CALUDE_prime_cube_difference_l909_90973

theorem prime_cube_difference (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 11 * p = q^3 - r^3 → 
  p = 199 ∧ q = 13 ∧ r = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_cube_difference_l909_90973


namespace NUMINAMATH_CALUDE_problem_statement_l909_90970

theorem problem_statement (a b c d : ℝ) : 
  (a * b > 0 ∧ b * c - a * d > 0 → c / a - d / b > 0) ∧
  (a * b > 0 ∧ c / a - d / b > 0 → b * c - a * d > 0) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l909_90970


namespace NUMINAMATH_CALUDE_license_plate_ratio_l909_90985

/-- The number of possible letters in a license plate -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate -/
def num_digits : ℕ := 10

/-- The number of letters in an old license plate -/
def old_letters : ℕ := 2

/-- The number of digits in an old license plate -/
def old_digits : ℕ := 3

/-- The number of letters in a new license plate -/
def new_letters : ℕ := 3

/-- The number of digits in a new license plate -/
def new_digits : ℕ := 4

/-- The ratio of new license plates to old license plates -/
theorem license_plate_ratio :
  (num_letters ^ new_letters * num_digits ^ new_digits) /
  (num_letters ^ old_letters * num_digits ^ old_digits) = 260 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_ratio_l909_90985


namespace NUMINAMATH_CALUDE_M_minus_N_equals_closed_open_l909_90920

-- Definition of set difference
def set_difference (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- Definition of set M
def M : Set ℝ := {x | |x + 1| ≤ 2}

-- Definition of set N
def N : Set ℝ := {x | ∃ α : ℝ, x = |Real.sin α|}

-- Theorem statement
theorem M_minus_N_equals_closed_open :
  set_difference M N = Set.Ico (-3) 0 := by sorry

end NUMINAMATH_CALUDE_M_minus_N_equals_closed_open_l909_90920


namespace NUMINAMATH_CALUDE_subtracted_number_l909_90915

theorem subtracted_number (a b : ℕ) (x : ℚ) 
  (h1 : a / b = 6 / 5)
  (h2 : (a - x) / (b - x) = 5 / 4)
  (h3 : a - b = 5) :
  x = 5 := by sorry

end NUMINAMATH_CALUDE_subtracted_number_l909_90915


namespace NUMINAMATH_CALUDE_base_approximation_l909_90932

/-- The base value we're looking for -/
def base : ℝ := 21.5

/-- The function representing the left side of the inequality -/
def f (b : ℝ) : ℝ := 2.134 * b^3

theorem base_approximation :
  ∀ b : ℝ, f b < 21000 → b ≤ base :=
sorry

end NUMINAMATH_CALUDE_base_approximation_l909_90932


namespace NUMINAMATH_CALUDE_rex_driving_lessons_l909_90906

theorem rex_driving_lessons 
  (total_hours : ℕ) 
  (hours_per_week : ℕ) 
  (remaining_weeks : ℕ) 
  (h1 : total_hours = 40)
  (h2 : hours_per_week = 4)
  (h3 : remaining_weeks = 4) :
  total_hours - (remaining_weeks * hours_per_week) = 6 * hours_per_week :=
by sorry

end NUMINAMATH_CALUDE_rex_driving_lessons_l909_90906


namespace NUMINAMATH_CALUDE_upper_limit_of_set_A_l909_90904

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def SetA : Set ℕ := {n : ℕ | isPrime n ∧ n > 15}

theorem upper_limit_of_set_A (lower_bound : ℕ) (h1 : lower_bound ∈ SetA) 
  (h2 : ∀ x ∈ SetA, x ≥ lower_bound) 
  (h3 : ∃ upper_bound : ℕ, upper_bound ∈ SetA ∧ upper_bound - lower_bound = 14) :
  ∃ max_element : ℕ, max_element ∈ SetA ∧ max_element = 31 :=
sorry

end NUMINAMATH_CALUDE_upper_limit_of_set_A_l909_90904


namespace NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_l909_90962

/-- Two lines are parallel -/
def parallel_lines (a b : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two planes are parallel -/
def parallel_planes (p q : Plane) : Prop := sorry

/-- A line is distinct from another line -/
def distinct_lines (a b : Line) : Prop := sorry

/-- A plane is distinct from another plane -/
def distinct_planes (p q : Plane) : Prop := sorry

theorem parallel_planes_from_perpendicular_lines 
  (a b : Line) (α β : Plane) 
  (h1 : distinct_lines a b) 
  (h2 : distinct_planes α β) 
  (h3 : perpendicular_line_plane a α) 
  (h4 : perpendicular_line_plane b β) 
  (h5 : parallel_lines a b) : 
  parallel_planes α β := 
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_l909_90962


namespace NUMINAMATH_CALUDE_emilys_initial_lives_l909_90959

theorem emilys_initial_lives :
  ∀ (initial_lives : ℕ),
  initial_lives - 25 + 24 = 41 →
  initial_lives = 42 :=
by sorry

end NUMINAMATH_CALUDE_emilys_initial_lives_l909_90959


namespace NUMINAMATH_CALUDE_coin_toss_sequences_count_l909_90927

/-- The number of ways to distribute n indistinguishable balls into k distinguishable urns -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of different sequences of 18 coin tosses with specific subsequence counts -/
def coinTossSequences : ℕ :=
  let numTosses := 18
  let numHH := 3
  let numHT := 4
  let numTH := 5
  let numTT := 6
  let numTGaps := numTH + 1
  let numHGaps := numHT + 1
  let tDistributions := starsAndBars numTT numTGaps
  let hDistributions := starsAndBars numHH numHGaps
  tDistributions * hDistributions

theorem coin_toss_sequences_count : coinTossSequences = 4200 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_sequences_count_l909_90927
