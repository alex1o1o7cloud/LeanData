import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_min_value_l3638_363856

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 4 = 1

/-- The distance from a point to the left focus -/
def dist_to_left_focus (x y : ℝ) : ℝ :=
  sorry

/-- The distance from a point to the right focus -/
def dist_to_right_focus (x y : ℝ) : ℝ :=
  sorry

theorem ellipse_min_value (x y : ℝ) :
  is_on_ellipse x y →
  let m := dist_to_left_focus x y
  let n := dist_to_right_focus x y
  1 / m + 4 / n ≥ 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_min_value_l3638_363856


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l3638_363802

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 4) (hb : b = 5) :
  ∃ (c : ℝ), c > 0 ∧ a^2 + c^2 = b^2 ∧
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
  ((x = a ∧ y = b) ∨ (x = b ∧ y = a) ∨ (y = a ∧ z = b) ∨ (y = b ∧ z = a)) →
  x^2 + y^2 = z^2 →
  (1/2) * x * y ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l3638_363802


namespace NUMINAMATH_CALUDE_johns_father_age_l3638_363896

theorem johns_father_age (john_age father_age : ℕ) : 
  john_age + father_age = 77 →
  father_age = 2 * john_age + 32 →
  john_age = 15 →
  father_age = 62 := by
sorry

end NUMINAMATH_CALUDE_johns_father_age_l3638_363896


namespace NUMINAMATH_CALUDE_tangent_line_property_l3638_363839

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_property (h : (fun x => x + f x - 5) = fun x => 0) :
  f 5 + (deriv f) 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_property_l3638_363839


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3638_363874

-- Define the start point of the line segment
def start_point : ℝ × ℝ := (1, 3)

-- Define the end point of the line segment
def end_point (x : ℝ) : ℝ × ℝ := (x, 7)

-- Define the length of the line segment
def segment_length : ℝ := 15

-- Theorem statement
theorem line_segment_endpoint (x : ℝ) : 
  x < 0 → 
  Real.sqrt ((x - 1)^2 + (7 - 3)^2) = segment_length → 
  x = 1 - Real.sqrt 209 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3638_363874


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l3638_363829

theorem reciprocal_of_negative_fraction :
  ((-5 : ℚ) / 3)⁻¹ = -3 / 5 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l3638_363829


namespace NUMINAMATH_CALUDE_sin_translation_l3638_363822

/-- Given a function f(x) = sin(2x), when translated π/3 units to the right,
    the resulting function g(x) is equal to sin(2x - 2π/3). -/
theorem sin_translation (x : ℝ) :
  let f : ℝ → ℝ := fun x => Real.sin (2 * x)
  let g : ℝ → ℝ := fun x => f (x - π / 3)
  g x = Real.sin (2 * x - 2 * π / 3) :=
by sorry

end NUMINAMATH_CALUDE_sin_translation_l3638_363822


namespace NUMINAMATH_CALUDE_transformed_square_properties_l3638_363830

/-- A point in the xy-plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The transformation from xy-plane to uv-plane -/
def transform (p : Point) : Point :=
  { x := p.x^2 + p.y^2,
    y := p.x^2 * p.y^2 }

/-- The unit square PQRST in the xy-plane -/
def unitSquare : Set Point :=
  {p | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

/-- The image of the unit square under the transformation -/
def transformedSquare : Set Point :=
  {q | ∃ p ∈ unitSquare, q = transform p}

/-- Definition of vertical symmetry -/
def verticallySymmetric (s : Set Point) : Prop :=
  ∀ p ∈ s, ∃ q ∈ s, q.x = p.x ∧ q.y = -p.y

/-- Definition of curved upper boundary -/
def hasCurvedUpperBoundary (s : Set Point) : Prop :=
  ∃ f : ℝ → ℝ, (∀ x, f x ≥ 0) ∧ 
    (∀ p ∈ s, p.y ≤ f p.x) ∧
    (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ ≠ f x₂)

theorem transformed_square_properties :
  verticallySymmetric transformedSquare ∧ 
  hasCurvedUpperBoundary transformedSquare :=
sorry

end NUMINAMATH_CALUDE_transformed_square_properties_l3638_363830


namespace NUMINAMATH_CALUDE_sum_reciprocals_perpendicular_lines_equilateral_triangle_perimeter_polynomial_divisibility_l3638_363862

-- Problem 1
theorem sum_reciprocals (x y : ℝ) (h1 : x + y = 50) (h2 : x * y = 25) :
  1 / x + 1 / y = 2 := by sorry

-- Problem 2
theorem perpendicular_lines (a b : ℝ) 
  (h : ∀ x y, (a * x + 2 * y + 1 = 0 ∧ 3 * x + b * y + 5 = 0) → 
    ((-a / 2) * (-3 / b) = -1)) :
  b = -3 := by sorry

-- Problem 3
theorem equilateral_triangle_perimeter (A : ℝ) (h : A = 100 * Real.sqrt 3) :
  let s := Real.sqrt (4 * A / Real.sqrt 3);
  3 * s = 60 := by sorry

-- Problem 4
theorem polynomial_divisibility (p q : ℝ) 
  (h : ∀ x, (x + 2) ∣ (x^3 - 2*x^2 + p*x + q)) 
  (h_p : p = 60) :
  q = 136 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_perpendicular_lines_equilateral_triangle_perimeter_polynomial_divisibility_l3638_363862


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3638_363812

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem geometric_sequence_property
  (b : ℕ → ℝ)
  (h_geometric : geometric_sequence b)
  (h_b1 : b 1 = 1)
  (s t : ℕ)
  (h_distinct : s ≠ t)
  (h_positive : s > 0 ∧ t > 0) :
  (b t) ^ (s - 1) / (b s) ^ (t - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3638_363812


namespace NUMINAMATH_CALUDE_sale_ratio_l3638_363848

def floral_shop_sales (monday_sales : ℕ) (total_sales : ℕ) : Prop :=
  let tuesday_sales := 3 * monday_sales
  let wednesday_sales := total_sales - (monday_sales + tuesday_sales)
  (wednesday_sales : ℚ) / tuesday_sales = 1 / 3

theorem sale_ratio : floral_shop_sales 12 60 := by
  sorry

end NUMINAMATH_CALUDE_sale_ratio_l3638_363848


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l3638_363809

theorem x_eq_one_sufficient_not_necessary_for_x_squared_eq_one :
  (∃ x : ℝ, x ^ 2 = 1 ∧ x ≠ 1) ∧
  (∀ x : ℝ, x = 1 → x ^ 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l3638_363809


namespace NUMINAMATH_CALUDE_dog_ate_cost_l3638_363890

-- Define the given conditions
def total_slices : ℕ := 6
def total_cost : ℚ := 9
def mother_slices : ℕ := 2

-- Define the theorem
theorem dog_ate_cost : 
  (total_cost / total_slices) * (total_slices - mother_slices) = 6 := by
  sorry

end NUMINAMATH_CALUDE_dog_ate_cost_l3638_363890


namespace NUMINAMATH_CALUDE_no_perfect_square_sum_l3638_363837

theorem no_perfect_square_sum (x y z : ℤ) (h : x^2 + y^2 + z^2 = 1993) :
  ¬ ∃ (a : ℤ), x + y + z = a^2 := by
sorry

end NUMINAMATH_CALUDE_no_perfect_square_sum_l3638_363837


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l3638_363869

def B : Set ℕ := {n | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

theorem gcd_of_B_is_two :
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l3638_363869


namespace NUMINAMATH_CALUDE_equation1_unique_solution_equation2_no_solution_l3638_363882

-- Define the first equation
def equation1 (x : ℝ) : Prop :=
  5 / (2 * x) - 1 / (x - 3) = 0

-- Define the second equation
def equation2 (x : ℝ) : Prop :=
  1 / (x - 2) = 4 / (x^2 - 4)

-- Theorem for the first equation
theorem equation1_unique_solution :
  ∃! x : ℝ, equation1 x ∧ x = 5 :=
sorry

-- Theorem for the second equation
theorem equation2_no_solution :
  ∀ x : ℝ, ¬ equation2 x :=
sorry

end NUMINAMATH_CALUDE_equation1_unique_solution_equation2_no_solution_l3638_363882


namespace NUMINAMATH_CALUDE_parabola_properties_l3638_363885

/-- A function representing a parabola -/
def f (x : ℝ) : ℝ := -x^2 + 1

/-- The parabola opens downwards -/
def opens_downwards (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a < 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The parabola intersects the y-axis at (0,1) -/
def intersects_y_axis_at_0_1 (f : ℝ → ℝ) : Prop :=
  f 0 = 1

theorem parabola_properties :
  opens_downwards f ∧ intersects_y_axis_at_0_1 f :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l3638_363885


namespace NUMINAMATH_CALUDE_ratio_problem_l3638_363844

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 5) :
  x / y = 13 / 9 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l3638_363844


namespace NUMINAMATH_CALUDE_sum_1_to_50_base6_l3638_363818

/-- Converts a base 10 number to base 6 --/
def toBase6 (n : ℕ) : ℕ := sorry

/-- Converts a base 6 number to base 10 --/
def fromBase6 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of integers from 1 to n in base 6 --/
def sumInBase6 (n : ℕ) : ℕ := sorry

theorem sum_1_to_50_base6 :
  sumInBase6 (fromBase6 50) = toBase6 55260 := by sorry

end NUMINAMATH_CALUDE_sum_1_to_50_base6_l3638_363818


namespace NUMINAMATH_CALUDE_three_thousandths_decimal_l3638_363872

theorem three_thousandths_decimal : (3 : ℚ) / 1000 = (3 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_three_thousandths_decimal_l3638_363872


namespace NUMINAMATH_CALUDE_hybrid_cars_with_full_headlights_l3638_363851

/-- Proves that in a car dealership with 600 cars, where 60% are hybrids and 40% of hybrids have only one headlight, the number of hybrids with full headlights is 216. -/
theorem hybrid_cars_with_full_headlights (total_cars : ℕ) (hybrid_percentage : ℚ) (one_headlight_percentage : ℚ) :
  total_cars = 600 →
  hybrid_percentage = 60 / 100 →
  one_headlight_percentage = 40 / 100 →
  (total_cars : ℚ) * hybrid_percentage * (1 - one_headlight_percentage) = 216 := by
  sorry

end NUMINAMATH_CALUDE_hybrid_cars_with_full_headlights_l3638_363851


namespace NUMINAMATH_CALUDE_hotel_room_charges_l3638_363823

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R - 0.5 * R) 
  (h2 : P = G - 0.1 * G) : 
  R = 1.8 * G := by
sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l3638_363823


namespace NUMINAMATH_CALUDE_inequality_proof_l3638_363899

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c = d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3638_363899


namespace NUMINAMATH_CALUDE_no_cube_in_range_l3638_363815

theorem no_cube_in_range : ¬ ∃ n : ℕ, 4 ≤ n ∧ n ≤ 12 ∧ ∃ k : ℕ, n^2 + 3*n + 1 = k^3 := by
  sorry

end NUMINAMATH_CALUDE_no_cube_in_range_l3638_363815


namespace NUMINAMATH_CALUDE_adams_lawn_mowing_l3638_363850

/-- Given that Adam earns 9 dollars per lawn, forgot to mow 8 lawns, and actually earned 36 dollars,
    prove that the total number of lawns he had to mow is 12. -/
theorem adams_lawn_mowing (dollars_per_lawn : ℕ) (forgotten_lawns : ℕ) (actual_earnings : ℕ) :
  dollars_per_lawn = 9 →
  forgotten_lawns = 8 →
  actual_earnings = 36 →
  (actual_earnings / dollars_per_lawn) + forgotten_lawns = 12 :=
by sorry

end NUMINAMATH_CALUDE_adams_lawn_mowing_l3638_363850


namespace NUMINAMATH_CALUDE_min_width_is_correct_l3638_363854

/-- The minimum width of a rectangular area satisfying the given conditions -/
def min_width : ℝ := 4

/-- The length of the rectangular area -/
def length (w : ℝ) : ℝ := w + 20

/-- The area of the rectangular area -/
def area (w : ℝ) : ℝ := w * length w

theorem min_width_is_correct :
  (∀ w : ℝ, w > 0 → area w ≥ 120 → w ≥ min_width) ∧
  (area min_width ≥ 120) ∧
  (min_width > 0) := by
  sorry

end NUMINAMATH_CALUDE_min_width_is_correct_l3638_363854


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3638_363861

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 < 2*x} = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3638_363861


namespace NUMINAMATH_CALUDE_vacation_probability_l3638_363871

theorem vacation_probability (prob_A prob_B : ℝ) 
  (h1 : prob_A = 1/4)
  (h2 : prob_B = 1/5)
  (h3 : 0 ≤ prob_A ∧ prob_A ≤ 1)
  (h4 : 0 ≤ prob_B ∧ prob_B ≤ 1) :
  1 - (1 - prob_A) * (1 - prob_B) = 2/5 := by
sorry

end NUMINAMATH_CALUDE_vacation_probability_l3638_363871


namespace NUMINAMATH_CALUDE_roots_condition_inequality_condition_max_value_condition_l3638_363853

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a + 2

-- Part 1
theorem roots_condition (a : ℝ) :
  (∃ x y, x ≠ y ∧ x < 2 ∧ y < 2 ∧ f a x = 0 ∧ f a y = 0) → a < -1 := by sorry

-- Part 2
theorem inequality_condition (a : ℝ) :
  (∀ x, f a x ≥ -1 - a*x) → -2 ≤ a ∧ a ≤ 6 := by sorry

-- Part 3
theorem max_value_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 0 2, f a x ≤ 4) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 4) → a = 2/3 ∨ a = 2 := by sorry

end NUMINAMATH_CALUDE_roots_condition_inequality_condition_max_value_condition_l3638_363853


namespace NUMINAMATH_CALUDE_investment_gain_percentage_l3638_363834

/-- Calculate the overall gain percentage for an investment portfolio --/
theorem investment_gain_percentage
  (stock_initial : ℝ)
  (artwork_initial : ℝ)
  (crypto_initial : ℝ)
  (stock_return : ℝ)
  (artwork_return : ℝ)
  (crypto_return_rub : ℝ)
  (rub_to_rs_rate : ℝ)
  (artwork_tax_rate : ℝ)
  (crypto_fee_rate : ℝ)
  (h1 : stock_initial = 5000)
  (h2 : artwork_initial = 10000)
  (h3 : crypto_initial = 15000)
  (h4 : stock_return = 6000)
  (h5 : artwork_return = 12000)
  (h6 : crypto_return_rub = 17000)
  (h7 : rub_to_rs_rate = 1.03)
  (h8 : artwork_tax_rate = 0.05)
  (h9 : crypto_fee_rate = 0.02) :
  let total_initial := stock_initial + artwork_initial + crypto_initial
  let artwork_net_return := artwork_return * (1 - artwork_tax_rate)
  let crypto_return_rs := crypto_return_rub * rub_to_rs_rate
  let crypto_net_return := crypto_return_rs * (1 - crypto_fee_rate)
  let total_return := stock_return + artwork_net_return + crypto_net_return
  let gain_percentage := (total_return - total_initial) / total_initial * 100
  ∃ ε > 0, |gain_percentage - 15.20| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_investment_gain_percentage_l3638_363834


namespace NUMINAMATH_CALUDE_tangent_line_at_P_l3638_363867

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 + 1

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the two possible tangent line equations
def tangent1 (x y : ℝ) : Prop := 3*x - y - 1 = 0
def tangent2 (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

-- Theorem statement
theorem tangent_line_at_P :
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m*x + b ↔ (tangent1 x y ∨ tangent2 x y)) ∧
  (curve P.1 = P.2) ∧
  (∃ ε > 0, ∀ h ∈ Set.Ioo (-ε) ε, 
    (curve (P.1 + h) - curve P.1) / h - m < ε) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_P_l3638_363867


namespace NUMINAMATH_CALUDE_typing_speed_ratio_l3638_363894

-- Define Tim's typing speed
def tim_speed : ℝ := 2

-- Define Tom's normal typing speed
def tom_speed : ℝ := 10

-- Define Tom's increased typing speed (30% increase)
def tom_increased_speed : ℝ := tom_speed * 1.3

-- Theorem to prove
theorem typing_speed_ratio :
  -- Condition 1: Tim and Tom can type 12 pages in one hour together
  tim_speed + tom_speed = 12 →
  -- Condition 2: With Tom's increased speed, they can type 15 pages in one hour
  tim_speed + tom_increased_speed = 15 →
  -- Conclusion: The ratio of Tom's normal speed to Tim's is 5:1
  tom_speed / tim_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_typing_speed_ratio_l3638_363894


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3638_363842

def f (x : ℝ) : ℝ := x * abs (x - 2)

theorem inequality_solution_set (x : ℝ) :
  f (Real.sqrt 2 - x) ≤ f 1 ↔ x ≥ -1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3638_363842


namespace NUMINAMATH_CALUDE_geometry_problem_l3638_363801

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 1

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the point P
def point_P : ℝ × ℝ := (2, -1)

-- Define the line 2x + y = 0
def line_center (x y : ℝ) : Prop := 2*x + y = 0

theorem geometry_problem :
  -- Conditions
  (∀ x y, line_l x y → (x = 2 ∧ y = -1) → True) ∧  -- l passes through P(2,-1)
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ ∀ x y, line_l x y ↔ x/a + y/b = 1) ∧  -- Sum of intercepts is 2
  (∃ m, line_center m (-2*m) ∧ ∀ x y, circle_M x y → line_center x y) ∧  -- M's center on 2x+y=0
  (∀ x y, circle_M x y → line_l x y → (x = 2 ∧ y = -1)) →  -- M tangent to l at P
  -- Conclusions
  (∀ x y, line_l x y ↔ x + y = 1) ∧  -- Equation of line l
  (∀ x y, circle_M x y ↔ (x - 1)^2 + (y + 2)^2 = 2) ∧  -- Equation of circle M
  (∃ y₁ y₂, y₁ < y₂ ∧ circle_M 0 y₁ ∧ circle_M 0 y₂ ∧ y₂ - y₁ = 2)  -- Length of chord on y-axis
  := by sorry

end NUMINAMATH_CALUDE_geometry_problem_l3638_363801


namespace NUMINAMATH_CALUDE_pam_has_ten_bags_l3638_363805

/-- The number of apples in one of Gerald's bags -/
def geralds_bag_count : ℕ := 40

/-- The number of Gerald's bags equivalent to one of Pam's bags -/
def bags_ratio : ℕ := 3

/-- The total number of apples Pam has -/
def pams_total_apples : ℕ := 1200

/-- The number of bags Pam has -/
def pams_bag_count : ℕ := pams_total_apples / (bags_ratio * geralds_bag_count)

theorem pam_has_ten_bags : pams_bag_count = 10 := by
  sorry

end NUMINAMATH_CALUDE_pam_has_ten_bags_l3638_363805


namespace NUMINAMATH_CALUDE_exists_x_tan_eq_two_l3638_363860

theorem exists_x_tan_eq_two : ∃ x : ℝ, Real.tan x = 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_tan_eq_two_l3638_363860


namespace NUMINAMATH_CALUDE_arithmetic_progression_product_not_square_l3638_363828

/-- Four distinct positive integers in arithmetic progression -/
structure ArithmeticProgression :=
  (a : ℕ+) -- First term
  (r : ℕ+) -- Common difference
  (distinct : a < a + r ∧ a + r < a + 2*r ∧ a + 2*r < a + 3*r)

/-- The product of four terms in arithmetic progression is not a perfect square -/
theorem arithmetic_progression_product_not_square (ap : ArithmeticProgression) :
  ¬ ∃ (m : ℕ), (ap.a * (ap.a + ap.r) * (ap.a + 2*ap.r) * (ap.a + 3*ap.r) : ℕ) = m^2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_progression_product_not_square_l3638_363828


namespace NUMINAMATH_CALUDE_square_division_problem_l3638_363825

theorem square_division_problem :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 + y^2 = 100 ∧ x/y = 4/3 ∧ x = 8 ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_division_problem_l3638_363825


namespace NUMINAMATH_CALUDE_helmet_discount_percentage_l3638_363832

def original_price : ℝ := 40
def amount_saved : ℝ := 8
def amount_spent : ℝ := 32

theorem helmet_discount_percentage :
  (amount_saved / original_price) * 100 = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_helmet_discount_percentage_l3638_363832


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l3638_363824

theorem rectangle_area_with_inscribed_circle (r : ℝ) (ratio : ℝ) : 
  r = 8 → ratio = 3 → 
  let width := 2 * r
  let length := ratio * width
  width * length = 768 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l3638_363824


namespace NUMINAMATH_CALUDE_appended_ages_digits_l3638_363820

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def append_numbers (a b : ℕ) : ℕ := a * 100 + b

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem appended_ages_digits (j a : ℕ) :
  is_two_digit j →
  is_two_digit a →
  is_perfect_square (append_numbers j a) →
  digit_sum (append_numbers j a) = 7 →
  ∃ n : ℕ, append_numbers j a = n ∧ 1000 ≤ n ∧ n ≤ 9999 :=
sorry

end NUMINAMATH_CALUDE_appended_ages_digits_l3638_363820


namespace NUMINAMATH_CALUDE_max_x_squared_minus_y_squared_l3638_363833

theorem max_x_squared_minus_y_squared (x y : ℝ) 
  (h : 2 * (x^3 + y^3) = x^2 + y^2) : 
  ∀ a b : ℝ, 2 * (a^3 + b^3) = a^2 + b^2 → x^2 - y^2 ≤ a^2 - b^2 := by
sorry

end NUMINAMATH_CALUDE_max_x_squared_minus_y_squared_l3638_363833


namespace NUMINAMATH_CALUDE_coin_difference_l3638_363881

def coin_values : List Nat := [1, 5, 10, 25, 50]
def target_amount : Nat := 65

def min_coins (amount : Nat) (coins : List Nat) : Nat :=
  sorry

def max_coins (amount : Nat) (coins : List Nat) : Nat :=
  sorry

theorem coin_difference :
  max_coins target_amount coin_values - min_coins target_amount coin_values = 62 :=
by sorry

end NUMINAMATH_CALUDE_coin_difference_l3638_363881


namespace NUMINAMATH_CALUDE_devin_basketball_chance_l3638_363866

/-- Represents the chance of making the basketball team based on height -/
def basketballChance (initialHeight : ℕ) (growth : ℕ) : ℝ :=
  let baseHeight : ℕ := 66
  let baseChance : ℝ := 0.1
  let chanceIncreasePerInch : ℝ := 0.1
  let finalHeight : ℕ := initialHeight + growth
  let additionalInches : ℕ := max (finalHeight - baseHeight) 0
  baseChance + (additionalInches : ℝ) * chanceIncreasePerInch

/-- Theorem stating Devin's chance of making the team after growing -/
theorem devin_basketball_chance :
  basketballChance 65 3 = 0.3 := by
  sorry

#eval basketballChance 65 3

end NUMINAMATH_CALUDE_devin_basketball_chance_l3638_363866


namespace NUMINAMATH_CALUDE_range_of_m_l3638_363873

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + m * x + 1 > 0) → 0 ≤ m ∧ m < 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3638_363873


namespace NUMINAMATH_CALUDE_pentagonal_dodecahedron_properties_l3638_363889

/-- A polyhedron with pentagonal faces -/
structure PentagonalPolyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- Euler's formula for polyhedra -/
axiom eulers_formula {p : PentagonalPolyhedron} : p.vertices - p.edges + p.faces = 2

/-- Each face is a pentagon -/
axiom pentagonal_faces {p : PentagonalPolyhedron} : p.edges * 2 = p.faces * 5

/-- Theorem: A polyhedron with 12 pentagonal faces has 30 edges and 20 vertices -/
theorem pentagonal_dodecahedron_properties :
  ∃ (p : PentagonalPolyhedron), p.faces = 12 ∧ p.edges = 30 ∧ p.vertices = 20 :=
sorry

end NUMINAMATH_CALUDE_pentagonal_dodecahedron_properties_l3638_363889


namespace NUMINAMATH_CALUDE_second_vessel_capacity_l3638_363884

/-- Proves that the capacity of the second vessel is 3.625 liters given the conditions of the problem -/
theorem second_vessel_capacity :
  let vessel1_capacity : ℝ := 3
  let vessel1_alcohol_percentage : ℝ := 0.25
  let vessel2_alcohol_percentage : ℝ := 0.40
  let total_liquid : ℝ := 8
  let new_concentration : ℝ := 0.275
  ∃ vessel2_capacity : ℝ,
    vessel2_capacity > 0 ∧
    vessel1_capacity * vessel1_alcohol_percentage + 
    vessel2_capacity * vessel2_alcohol_percentage = 
    total_liquid * new_concentration ∧
    vessel2_capacity = 3.625 := by
  sorry


end NUMINAMATH_CALUDE_second_vessel_capacity_l3638_363884


namespace NUMINAMATH_CALUDE_no_divisibility_by_4_or_8_l3638_363810

/-- The set of all numbers which are the sum of the squares of three consecutive odd integers -/
def T : Set ℤ :=
  {x | ∃ n : ℤ, Odd n ∧ x = 3 * n^2 + 8}

/-- Theorem stating that no member of T is divisible by 4 or 8 -/
theorem no_divisibility_by_4_or_8 (x : ℤ) (hx : x ∈ T) :
  ¬(4 ∣ x) ∧ ¬(8 ∣ x) := by
  sorry

#check no_divisibility_by_4_or_8

end NUMINAMATH_CALUDE_no_divisibility_by_4_or_8_l3638_363810


namespace NUMINAMATH_CALUDE_james_sales_theorem_l3638_363803

theorem james_sales_theorem (houses_day1 : ℕ) (houses_day2 : ℕ) (sale_rate_day2 : ℚ) (items_per_house : ℕ) :
  houses_day1 = 20 →
  houses_day2 = 2 * houses_day1 →
  sale_rate_day2 = 4/5 →
  items_per_house = 2 →
  houses_day1 * items_per_house + (houses_day2 : ℚ) * sale_rate_day2 * items_per_house = 104 :=
by sorry

end NUMINAMATH_CALUDE_james_sales_theorem_l3638_363803


namespace NUMINAMATH_CALUDE_cauchy_schwarz_and_inequality_sum_of_roots_inequality_l3638_363863

theorem cauchy_schwarz_and_inequality (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) ≥ (a*x + b*y + c*z)^2 :=
sorry

theorem sum_of_roots_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  Real.sqrt a + Real.sqrt (2 * b) + Real.sqrt (3 * c) ≤ Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_and_inequality_sum_of_roots_inequality_l3638_363863


namespace NUMINAMATH_CALUDE_thief_speed_calculation_l3638_363835

/-- The speed of the thief's car in km/h -/
def thief_speed : ℝ := 43.75

/-- The head start time of the thief in hours -/
def head_start : ℝ := 0.5

/-- The speed of the owner's bike in km/h -/
def owner_speed : ℝ := 50

/-- The total time until the owner overtakes the thief in hours -/
def total_time : ℝ := 4

theorem thief_speed_calculation :
  thief_speed * total_time = owner_speed * (total_time - head_start) := by sorry

#check thief_speed_calculation

end NUMINAMATH_CALUDE_thief_speed_calculation_l3638_363835


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_3_l3638_363864

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

theorem parallel_lines_imply_a_equals_3 :
  ∀ a : ℝ,
  let l1 : Line := ⟨a, 2, 3*a⟩
  let l2 : Line := ⟨3, a-1, a-7⟩
  parallel l1 l2 → a = 3 := by
  sorry

#check parallel_lines_imply_a_equals_3

end NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_3_l3638_363864


namespace NUMINAMATH_CALUDE_kopeck_ruble_exchange_l3638_363895

/-- Represents the denominations of coins available in kopecks -/
def Denominations : List ℕ := [1, 2, 5, 10, 20, 50, 100]

/-- Represents a valid coin exchange -/
def IsValidExchange (amount : ℕ) (coinCount : ℕ) : Prop :=
  ∃ (coins : List ℕ), 
    (coins.length = coinCount) ∧ 
    (coins.sum = amount) ∧
    (∀ c ∈ coins, c ∈ Denominations)

/-- The main theorem: if A kopecks can be exchanged with B coins,
    then B rubles can be exchanged with A coins -/
theorem kopeck_ruble_exchange 
  (A B : ℕ) 
  (h : IsValidExchange A B) : 
  IsValidExchange (100 * B) A := by
  sorry

#check kopeck_ruble_exchange

end NUMINAMATH_CALUDE_kopeck_ruble_exchange_l3638_363895


namespace NUMINAMATH_CALUDE_x_squared_plus_four_y_squared_lt_one_l3638_363808

theorem x_squared_plus_four_y_squared_lt_one
  (x y : ℝ)
  (x_pos : x > 0)
  (y_pos : y > 0)
  (h : x^3 + y^3 = x - y) :
  x^2 + 4*y^2 < 1 :=
by sorry

end NUMINAMATH_CALUDE_x_squared_plus_four_y_squared_lt_one_l3638_363808


namespace NUMINAMATH_CALUDE_rectangle_side_equality_l3638_363883

theorem rectangle_side_equality (X : ℝ) : 
  (∀ (top bottom : ℝ), top = 5 + X ∧ bottom = 10 ∧ top = bottom) → X = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_equality_l3638_363883


namespace NUMINAMATH_CALUDE_smallest_angle_is_three_l3638_363817

/-- Represents a polygon divided into sectors with central angles forming an arithmetic sequence -/
structure PolygonSectors where
  num_sectors : ℕ
  angle_sum : ℕ
  is_arithmetic_sequence : Bool
  all_angles_integer : Bool

/-- The smallest possible sector angle for a polygon with given properties -/
def smallest_sector_angle (p : PolygonSectors) : ℕ :=
  sorry

/-- Theorem stating the smallest possible sector angle for a specific polygon configuration -/
theorem smallest_angle_is_three :
  ∀ (p : PolygonSectors),
    p.num_sectors = 16 ∧
    p.angle_sum = 360 ∧
    p.is_arithmetic_sequence = true ∧
    p.all_angles_integer = true →
    smallest_sector_angle p = 3 :=
  sorry

end NUMINAMATH_CALUDE_smallest_angle_is_three_l3638_363817


namespace NUMINAMATH_CALUDE_sin_arccos_tan_arcsin_product_one_l3638_363878

theorem sin_arccos_tan_arcsin_product_one :
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧
  x₁ ≠ x₂ ∧
  (∀ (x : ℝ), x > 0 → Real.sin (Real.arccos (Real.tan (Real.arcsin x))) = x → (x = x₁ ∨ x = x₂)) ∧
  x₁ * x₂ = 1 :=
sorry

end NUMINAMATH_CALUDE_sin_arccos_tan_arcsin_product_one_l3638_363878


namespace NUMINAMATH_CALUDE_fruit_bowl_total_l3638_363879

/-- Represents the number of pieces of each type of fruit in the bowl -/
structure FruitBowl where
  apples : ℕ
  pears : ℕ
  bananas : ℕ

/-- Theorem stating the total number of fruits in the bowl under given conditions -/
theorem fruit_bowl_total (bowl : FruitBowl) 
  (h1 : bowl.pears = bowl.apples + 2)
  (h2 : bowl.bananas = bowl.pears + 3)
  (h3 : bowl.bananas = 9) : 
  bowl.apples + bowl.pears + bowl.bananas = 19 := by
  sorry

end NUMINAMATH_CALUDE_fruit_bowl_total_l3638_363879


namespace NUMINAMATH_CALUDE_angle_bisector_ratio_l3638_363840

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the angle bisectors
def angleBisector (T : Triangle) (vertex : ℝ × ℝ) (side1 side2 : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the intersection point Q
def intersectionPoint (T : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_bisector_ratio (T : Triangle) :
  let X := T.X
  let Y := T.Y
  let Z := T.Z
  let U := angleBisector T X Y Z
  let V := angleBisector T Y X Z
  let Q := intersectionPoint T
  distance X Y = 8 ∧ distance X Z = 6 ∧ distance Y Z = 4 →
  distance Y Q / distance Q V = 2 := by sorry

end NUMINAMATH_CALUDE_angle_bisector_ratio_l3638_363840


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3638_363898

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Moves the last digit to the front -/
def ThreeDigitNumber.rotateDigits (n : ThreeDigitNumber) : ThreeDigitNumber :=
  ⟨n.ones, n.hundreds, n.tens, by sorry⟩

theorem unique_three_digit_number :
  ∃! (n : ThreeDigitNumber),
    n.ones = 1 ∧
    (n.toNat - n.rotateDigits.toNat : Int) = (10 * (3 ^ 2) : Int) ∧
    n.toNat = 211 := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l3638_363898


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l3638_363858

/-- The number of ways to arrange a group for a photo --/
def photo_arrangements (n_teacher : ℕ) (n_female : ℕ) (n_male : ℕ) : ℕ :=
  2 * (n_teacher + n_female + n_male).factorial

/-- Theorem: There are 12 ways to arrange 1 teacher, 2 female students, and 2 male students
    in a row for a photo, where the two female students are separated only by the teacher. --/
theorem photo_arrangement_count :
  photo_arrangements 1 2 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l3638_363858


namespace NUMINAMATH_CALUDE_reynalds_volleyballs_l3638_363865

/-- The number of volleyballs in Reynald's purchase --/
def num_volleyballs (total : ℕ) (soccer : ℕ) : ℕ :=
  total - (soccer + (soccer + 5) + (2 * soccer) + (soccer + 10))

/-- Theorem stating the number of volleyballs Reynald bought --/
theorem reynalds_volleyballs : num_volleyballs 145 20 = 30 := by
  sorry

#eval num_volleyballs 145 20

end NUMINAMATH_CALUDE_reynalds_volleyballs_l3638_363865


namespace NUMINAMATH_CALUDE_parallelogram_area_smallest_real_part_l3638_363875

theorem parallelogram_area_smallest_real_part (z : ℂ) :
  (z.im > 0) →
  (abs ((z - z⁻¹).re) ≥ 0) →
  (abs (z.im * z⁻¹.re - z.re * z⁻¹.im) = 1) →
  ∃ (w : ℂ), (w.im > 0) ∧ 
             (abs ((w - w⁻¹).re) ≥ 0) ∧ 
             (abs (w.im * w⁻¹.re - w.re * w⁻¹.im) = 1) ∧
             (abs ((w - w⁻¹).re) = 0) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_smallest_real_part_l3638_363875


namespace NUMINAMATH_CALUDE_fraction_addition_l3638_363811

theorem fraction_addition : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3638_363811


namespace NUMINAMATH_CALUDE_apple_count_is_36_l3638_363855

/-- Given a ratio of mangoes to oranges to apples and the number of mangoes,
    calculate the number of apples -/
def calculate_apples (mango_ratio : ℕ) (orange_ratio : ℕ) (apple_ratio : ℕ) (mango_count : ℕ) : ℕ :=
  (mango_count / mango_ratio) * apple_ratio

/-- Theorem stating that given the specific ratio and mango count, 
    the number of apples is 36 -/
theorem apple_count_is_36 :
  calculate_apples 10 2 3 120 = 36 := by
  sorry

end NUMINAMATH_CALUDE_apple_count_is_36_l3638_363855


namespace NUMINAMATH_CALUDE_G_simplification_l3638_363852

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := F ((2 * x - x^2) / (1 + 2 * x + x^2))

theorem G_simplification (x : ℝ) (h : x ≠ -1/2) : G x = Real.log (1 + 4 * x) - Real.log (1 + 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_G_simplification_l3638_363852


namespace NUMINAMATH_CALUDE_same_function_constant_one_and_x_power_zero_l3638_363859

theorem same_function_constant_one_and_x_power_zero :
  ∀ x : ℝ, (1 : ℝ) = x^(0 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_same_function_constant_one_and_x_power_zero_l3638_363859


namespace NUMINAMATH_CALUDE_value_of_expression_l3638_363814

theorem value_of_expression (x : ℝ) (h : x = -3) : 3 * x^2 + 2 * x = 21 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3638_363814


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3638_363816

-- Define the universal set I
def I : Set ℕ := {0, 1, 2, 3, 4}

-- Define set M
def M : Set ℕ := {1, 2, 3}

-- Define set N
def N : Set ℕ := {0, 3, 4}

-- Theorem statement
theorem complement_intersection_theorem :
  (I \ M) ∩ N = {0, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3638_363816


namespace NUMINAMATH_CALUDE_outfits_count_l3638_363826

/-- The number of shirts available. -/
def num_shirts : ℕ := 8

/-- The number of pants available. -/
def num_pants : ℕ := 5

/-- The number of ties available. -/
def num_ties : ℕ := 4

/-- The number of belts available. -/
def num_belts : ℕ := 2

/-- The total number of outfit combinations. -/
def total_outfits : ℕ := num_shirts * num_pants * (num_ties + 1) * (num_belts + 1)

/-- Theorem stating that the total number of outfits is 600. -/
theorem outfits_count : total_outfits = 600 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l3638_363826


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l3638_363827

theorem max_value_of_sum_products (x y z : ℝ) (h : x + 2 * y + z = 6) :
  ∃ (max : ℝ), max = 6 ∧ ∀ (a b c : ℝ), a + 2 * b + c = 6 → a * b + a * c + b * c ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l3638_363827


namespace NUMINAMATH_CALUDE_longs_interest_l3638_363876

/-- Calculates the total interest earned after a given number of years with compound interest and an additional deposit -/
def totalInterest (initialInvestment : ℝ) (interestRate : ℝ) (additionalDeposit : ℝ) (depositYear : ℕ) (totalYears : ℕ) : ℝ :=
  let finalAmount := 
    (initialInvestment * (1 + interestRate) ^ depositYear + additionalDeposit) * (1 + interestRate) ^ (totalYears - depositYear)
  finalAmount - initialInvestment - additionalDeposit

/-- The total interest earned by Long after 4 years -/
theorem longs_interest : 
  totalInterest 1200 0.08 500 2 4 = 515.26 := by sorry

end NUMINAMATH_CALUDE_longs_interest_l3638_363876


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3638_363886

theorem quadratic_equation_solution (y : ℝ) : 
  (((8 * y^2 + 50 * y + 5) / (3 * y + 21)) = 4 * y + 3) ↔ 
  (y = (-43 + Real.sqrt 921) / 8 ∨ y = (-43 - Real.sqrt 921) / 8) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3638_363886


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3638_363888

/-- Proves that if an article is sold for $1200 with a 20% profit, then the cost price is $1000. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 1200 ∧ profit_percentage = 20 →
  (selling_price = (100 + profit_percentage) / 100 * 1000) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3638_363888


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3638_363806

theorem complex_fraction_simplification :
  (5 + 12 * Complex.I) / (2 - 3 * Complex.I) = -2 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3638_363806


namespace NUMINAMATH_CALUDE_inequality_solution_l3638_363868

theorem inequality_solution (x : ℝ) : 
  1 / (x + 2) + 7 / (x + 6) ≥ 1 ↔ 
  x ≤ -6 ∨ (-2 < x ∧ x ≤ -Real.sqrt 15) ∨ x ≥ Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3638_363868


namespace NUMINAMATH_CALUDE_exam_failure_percentage_l3638_363813

theorem exam_failure_percentage 
  (total_candidates : ℕ) 
  (hindi_failure_rate : ℚ)
  (both_failure_rate : ℚ)
  (english_only_pass : ℕ) :
  total_candidates = 3000 →
  hindi_failure_rate = 36/100 →
  both_failure_rate = 15/100 →
  english_only_pass = 630 →
  ∃ (english_failure_rate : ℚ),
    english_failure_rate = 85/100 ∧
    english_only_pass = total_candidates * ((1 - english_failure_rate) + (hindi_failure_rate - both_failure_rate)) :=
by sorry

end NUMINAMATH_CALUDE_exam_failure_percentage_l3638_363813


namespace NUMINAMATH_CALUDE_A_share_is_one_third_l3638_363836

structure Partnership where
  initial_investment : ℝ
  total_gain : ℝ

def investment_share (p : Partnership) (months : ℝ) (multiplier : ℝ) : ℝ :=
  p.initial_investment * multiplier * months

theorem A_share_is_one_third (p : Partnership) :
  p.total_gain = 12000 →
  investment_share p 12 1 = investment_share p 6 2 →
  investment_share p 12 1 = investment_share p 4 3 →
  investment_share p 12 1 = p.total_gain / 3 := by
sorry

end NUMINAMATH_CALUDE_A_share_is_one_third_l3638_363836


namespace NUMINAMATH_CALUDE_subtract_problem_l3638_363887

theorem subtract_problem (x : ℕ) (h : 913 - x = 514) : 514 - x = 115 := by
  sorry

end NUMINAMATH_CALUDE_subtract_problem_l3638_363887


namespace NUMINAMATH_CALUDE_bus_driver_hours_l3638_363891

theorem bus_driver_hours (regular_rate overtime_rate_factor total_compensation : ℚ) : 
  regular_rate = 14 →
  overtime_rate_factor = 1.75 →
  total_compensation = 982 →
  ∃ (regular_hours overtime_hours : ℕ),
    regular_hours = 40 ∧
    overtime_hours = 17 ∧
    regular_hours + overtime_hours = 57 ∧
    regular_rate * regular_hours + (regular_rate * overtime_rate_factor) * overtime_hours = total_compensation :=
by sorry

end NUMINAMATH_CALUDE_bus_driver_hours_l3638_363891


namespace NUMINAMATH_CALUDE_min_type_b_workers_l3638_363892

/-- The number of workers in the workshop -/
def total_workers : ℕ := 20

/-- The number of Type A parts a worker can produce per day -/
def type_a_production : ℕ := 6

/-- The number of Type B parts a worker can produce per day -/
def type_b_production : ℕ := 5

/-- The profit (in yuan) from producing one Type A part -/
def type_a_profit : ℕ := 150

/-- The profit (in yuan) from producing one Type B part -/
def type_b_profit : ℕ := 260

/-- The daily profit function (in yuan) based on the number of workers producing Type A parts -/
def daily_profit (x : ℝ) : ℝ :=
  type_a_profit * type_a_production * x + type_b_profit * type_b_production * (total_workers - x)

/-- The minimum required daily profit (in yuan) -/
def min_profit : ℝ := 24000

theorem min_type_b_workers :
  ∀ x : ℝ, 0 ≤ x → x ≤ total_workers →
  (∀ y : ℝ, y ≥ min_profit → daily_profit x ≥ y) →
  total_workers - x ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_min_type_b_workers_l3638_363892


namespace NUMINAMATH_CALUDE_rick_ironed_45_pieces_l3638_363847

/-- Represents Rick's ironing rates and time spent ironing --/
structure IroningData where
  weekday_shirt_rate : ℕ
  weekday_pants_rate : ℕ
  weekday_jacket_rate : ℕ
  weekend_shirt_rate : ℕ
  weekend_pants_rate : ℕ
  weekend_jacket_rate : ℕ
  weekday_shirt_time : ℕ
  weekday_pants_time : ℕ
  weekday_jacket_time : ℕ
  weekend_shirt_time : ℕ
  weekend_pants_time : ℕ
  weekend_jacket_time : ℕ

/-- Calculates the total number of pieces of clothing ironed --/
def total_ironed (data : IroningData) : ℕ :=
  (data.weekday_shirt_rate * data.weekday_shirt_time + data.weekend_shirt_rate * data.weekend_shirt_time) +
  (data.weekday_pants_rate * data.weekday_pants_time + data.weekend_pants_rate * data.weekend_pants_time) +
  (data.weekday_jacket_rate * data.weekday_jacket_time + data.weekend_jacket_rate * data.weekend_jacket_time)

/-- Theorem stating that Rick irons 45 pieces of clothing given the specified rates and times --/
theorem rick_ironed_45_pieces : 
  ∀ (data : IroningData), 
    data.weekday_shirt_rate = 4 ∧ 
    data.weekday_pants_rate = 3 ∧ 
    data.weekday_jacket_rate = 2 ∧
    data.weekend_shirt_rate = 5 ∧ 
    data.weekend_pants_rate = 4 ∧ 
    data.weekend_jacket_rate = 3 ∧
    data.weekday_shirt_time = 2 ∧ 
    data.weekday_pants_time = 3 ∧ 
    data.weekday_jacket_time = 1 ∧
    data.weekend_shirt_time = 3 ∧ 
    data.weekend_pants_time = 2 ∧ 
    data.weekend_jacket_time = 1 
    → total_ironed data = 45 := by
  sorry

end NUMINAMATH_CALUDE_rick_ironed_45_pieces_l3638_363847


namespace NUMINAMATH_CALUDE_xy_equals_one_l3638_363843

theorem xy_equals_one (x y : ℝ) (h : x + y = 1/x + 1/y) (h_neq : x + y ≠ 0) : x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_one_l3638_363843


namespace NUMINAMATH_CALUDE_division_remainder_l3638_363849

theorem division_remainder : ∃ (A : ℕ), 17 = 6 * 2 + A ∧ A < 6 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l3638_363849


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l3638_363897

theorem opposite_of_negative_2023 : 
  ∃ x : ℤ, x + (-2023) = 0 ∧ x = 2023 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l3638_363897


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l3638_363807

def z : ℂ := (3 - Complex.I) * (1 + Complex.I)

theorem z_in_first_quadrant : z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l3638_363807


namespace NUMINAMATH_CALUDE_seventh_term_is_twenty_l3638_363893

/-- An arithmetic sequence with first term 2 and common difference 3 -/
def arithmeticSequence (n : ℕ) : ℕ := 2 + 3 * (n - 1)

/-- Theorem stating that the 7th term of the arithmetic sequence is 20 -/
theorem seventh_term_is_twenty : arithmeticSequence 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_twenty_l3638_363893


namespace NUMINAMATH_CALUDE_solve_system_l3638_363838

theorem solve_system (x y : ℚ) 
  (h1 : 1 / x + 3 / y = 1 / 2) 
  (h2 : 1 / y - 3 / x = 1 / 3) : 
  x = -20 ∧ y = 60 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3638_363838


namespace NUMINAMATH_CALUDE_smallest_integer_square_equation_l3638_363831

theorem smallest_integer_square_equation : ∃ x : ℤ, 
  (∀ y : ℤ, y^2 = 3*y + 72 → x ≤ y) ∧ x^2 = 3*x + 72 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_square_equation_l3638_363831


namespace NUMINAMATH_CALUDE_balls_in_original_positions_l3638_363800

/-- The number of balls arranged in a circle -/
def num_balls : ℕ := 7

/-- The number of transpositions performed -/
def num_transpositions : ℕ := 3

/-- The probability of a ball being in its original position after the transpositions -/
def prob_original_position : ℚ := 127 / 343

/-- The expected number of balls in their original positions after the transpositions -/
def expected_original_positions : ℚ := 889 / 343

theorem balls_in_original_positions :
  num_balls * prob_original_position = expected_original_positions := by sorry

end NUMINAMATH_CALUDE_balls_in_original_positions_l3638_363800


namespace NUMINAMATH_CALUDE_crow_worm_consumption_l3638_363821

theorem crow_worm_consumption 
  (crows_per_hour : ℕ) 
  (worms_per_hour : ℕ) 
  (new_crows : ℕ) 
  (new_hours : ℕ) 
  (h1 : crows_per_hour = 3) 
  (h2 : worms_per_hour = 30) 
  (h3 : new_crows = 5) 
  (h4 : new_hours = 2) : 
  (worms_per_hour / crows_per_hour) * new_crows * new_hours = 100 := by
  sorry

end NUMINAMATH_CALUDE_crow_worm_consumption_l3638_363821


namespace NUMINAMATH_CALUDE_car_dealership_problem_l3638_363880

/-- Represents the initial number of cars on the lot -/
def initial_cars : ℕ := 280

/-- Represents the number of cars in the new shipment -/
def new_shipment : ℕ := 80

/-- Represents the initial percentage of silver cars -/
def initial_silver_percent : ℚ := 20 / 100

/-- Represents the percentage of non-silver cars in the new shipment -/
def new_non_silver_percent : ℚ := 35 / 100

/-- Represents the final percentage of silver cars after the new shipment -/
def final_silver_percent : ℚ := 30 / 100

theorem car_dealership_problem :
  let initial_silver := initial_silver_percent * initial_cars
  let new_silver := (1 - new_non_silver_percent) * new_shipment
  let total_cars := initial_cars + new_shipment
  let total_silver := initial_silver + new_silver
  (total_silver : ℚ) / total_cars = final_silver_percent := by
  sorry

end NUMINAMATH_CALUDE_car_dealership_problem_l3638_363880


namespace NUMINAMATH_CALUDE_brianna_book_savings_l3638_363804

theorem brianna_book_savings (m : ℚ) (p : ℚ) : 
  (1/4 : ℚ) * m = (1/2 : ℚ) * p → m - p = (1/2 : ℚ) * m :=
by
  sorry

end NUMINAMATH_CALUDE_brianna_book_savings_l3638_363804


namespace NUMINAMATH_CALUDE_odd_function_solution_set_l3638_363857

/-- An odd function f: ℝ → ℝ satisfying certain conditions -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (f 2 = 0) ∧ 
  (∀ x > 0, x * (deriv f x) - f x < 0)

/-- The solution set for f(x)/x > 0 given the conditions on f -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | x ∈ Set.Ioo (-2) 0 ∪ Set.Ioo 0 2}

theorem odd_function_solution_set (f : ℝ → ℝ) (hf : OddFunction f) :
  {x : ℝ | f x / x > 0} = SolutionSet f :=
sorry

end NUMINAMATH_CALUDE_odd_function_solution_set_l3638_363857


namespace NUMINAMATH_CALUDE_exchange_calculation_l3638_363877

/-- Exchange rate from USD to JPY -/
def exchange_rate : ℚ := 5000 / 45

/-- Amount in USD to be exchanged -/
def usd_amount : ℚ := 15

/-- Theorem stating the correct exchange amount -/
theorem exchange_calculation :
  usd_amount * exchange_rate = 5000 / 3 := by
  sorry

end NUMINAMATH_CALUDE_exchange_calculation_l3638_363877


namespace NUMINAMATH_CALUDE_cubic_sum_inequality_quadratic_roots_bound_l3638_363841

theorem cubic_sum_inequality (p q : ℝ) (h : p^3 + q^3 = 2) : p + q ≤ 2 := by
  sorry

theorem quadratic_roots_bound (a b : ℝ) (h : |a| + |b| < 1) :
  ∀ x, x^2 + a*x + b = 0 → |x| < 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_inequality_quadratic_roots_bound_l3638_363841


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l3638_363870

/-- Given a circle with circumference 24 cm, its area is 144/π square centimeters. -/
theorem circle_area_from_circumference :
  ∀ (r : ℝ), 2 * π * r = 24 → π * r^2 = 144 / π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l3638_363870


namespace NUMINAMATH_CALUDE_casey_owns_five_hoodies_l3638_363845

/-- The number of hoodies Fiona and Casey own together -/
def total_hoodies : ℕ := 8

/-- The number of hoodies Fiona owns -/
def fiona_hoodies : ℕ := 3

/-- The number of hoodies Casey owns -/
def casey_hoodies : ℕ := total_hoodies - fiona_hoodies

theorem casey_owns_five_hoodies : casey_hoodies = 5 := by
  sorry

end NUMINAMATH_CALUDE_casey_owns_five_hoodies_l3638_363845


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l3638_363819

-- Define the fixed point M
def M : ℝ × ℝ := (-4, 0)

-- Define the equation of the known circle N
def circle_N (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 16

-- Define the trajectory equation
def trajectory (x y : ℝ) : Prop := x^2/4 - y^2/12 = 1

-- Theorem statement
theorem moving_circle_trajectory :
  ∀ (x y : ℝ),
    (∃ (r : ℝ), 
      -- The moving circle passes through M
      (x + 4)^2 + y^2 = r^2 ∧
      -- The moving circle is tangent to N
      ∃ (x_n y_n : ℝ), circle_N x_n y_n ∧ (x - x_n)^2 + (y - y_n)^2 = r^2) →
    trajectory x y :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l3638_363819


namespace NUMINAMATH_CALUDE_females_with_advanced_degrees_l3638_363846

theorem females_with_advanced_degrees 
  (total_employees : ℕ) 
  (total_females : ℕ) 
  (employees_with_advanced_degrees : ℕ) 
  (males_with_college_degree_only : ℕ) 
  (h1 : total_employees = 180) 
  (h2 : total_females = 110) 
  (h3 : employees_with_advanced_degrees = 90) 
  (h4 : males_with_college_degree_only = 35) :
  total_females - (total_employees - employees_with_advanced_degrees) + 
  (employees_with_advanced_degrees - (total_employees - total_females - males_with_college_degree_only)) = 55 :=
by sorry

end NUMINAMATH_CALUDE_females_with_advanced_degrees_l3638_363846
