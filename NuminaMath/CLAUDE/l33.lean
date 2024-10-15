import Mathlib

namespace NUMINAMATH_CALUDE_intersection_range_l33_3322

-- Define the line equation
def line_equation (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := x^2 - y^2 = 6

-- Define the condition for intersection
def intersects_hyperbola (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  hyperbola_equation x₁ (line_equation k x₁) ∧
  hyperbola_equation x₂ (line_equation k x₂) ∧
  x₁ * x₂ < 0  -- Ensures points are on different branches

-- Theorem statement
theorem intersection_range :
  ∀ k : ℝ, intersects_hyperbola k ↔ -1 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l33_3322


namespace NUMINAMATH_CALUDE_triangle_area_is_seven_l33_3382

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line by two points
structure Line2D where
  p1 : Point2D
  p2 : Point2D

-- Define the three lines
def line1 : Line2D := { p1 := { x := 0, y := 5 }, p2 := { x := 10, y := 2 } }
def line2 : Line2D := { p1 := { x := 2, y := 6 }, p2 := { x := 8, y := 1 } }
def line3 : Line2D := { p1 := { x := 0, y := 3 }, p2 := { x := 5, y := 0 } }

-- Function to calculate the area of a triangle formed by three lines
def triangleArea (l1 l2 l3 : Line2D) : ℝ :=
  sorry

-- Theorem stating that the area of the triangle is 7
theorem triangle_area_is_seven :
  triangleArea line1 line2 line3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_seven_l33_3382


namespace NUMINAMATH_CALUDE_union_condition_intersection_condition_l33_3398

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | m < x ∧ x < m + 3}

-- Theorem for part (I)
theorem union_condition (m : ℝ) :
  A ∪ B m = A ↔ m ∈ Set.Icc (-2) 2 := by sorry

-- Theorem for part (II)
theorem intersection_condition (m : ℝ) :
  (A ∩ B m).Nonempty ↔ m ∈ Set.Ioo (-5) 2 := by sorry

end NUMINAMATH_CALUDE_union_condition_intersection_condition_l33_3398


namespace NUMINAMATH_CALUDE_modulus_z_l33_3318

/-- Given complex numbers w and z such that wz = 20 - 15i and |w| = √34, prove that |z| = (25√34) / 34 -/
theorem modulus_z (w z : ℂ) (h1 : w * z = 20 - 15 * I) (h2 : Complex.abs w = Real.sqrt 34) :
  Complex.abs z = (25 * Real.sqrt 34) / 34 := by
  sorry

end NUMINAMATH_CALUDE_modulus_z_l33_3318


namespace NUMINAMATH_CALUDE_theo_cookie_consumption_l33_3379

def cookies_per_sitting : ℕ := 25
def sittings_per_day : ℕ := 5
def days_per_month : ℕ := 27
def months : ℕ := 9

theorem theo_cookie_consumption :
  cookies_per_sitting * sittings_per_day * days_per_month * months = 30375 :=
by
  sorry

end NUMINAMATH_CALUDE_theo_cookie_consumption_l33_3379


namespace NUMINAMATH_CALUDE_rational_trig_sums_l33_3392

theorem rational_trig_sums (x : ℝ) 
  (s_rational : ∃ q : ℚ, (Real.sin (64 * x) + Real.sin (65 * x)) = ↑q)
  (t_rational : ∃ q : ℚ, (Real.cos (64 * x) + Real.cos (65 * x)) = ↑q) :
  (∃ q1 q2 : ℚ, Real.cos (64 * x) = ↑q1 ∧ Real.cos (65 * x) = ↑q2) ∨
  (∃ q1 q2 : ℚ, Real.sin (64 * x) = ↑q1 ∧ Real.sin (65 * x) = ↑q2) :=
by sorry

end NUMINAMATH_CALUDE_rational_trig_sums_l33_3392


namespace NUMINAMATH_CALUDE_twenty_paise_coins_count_l33_3380

/-- Given a total of 324 coins consisting of 20 paise and 25 paise denominations,
    and a total sum of Rs. 70, prove that the number of 20 paise coins is 220. -/
theorem twenty_paise_coins_count (x y : ℕ) : 
  x + y = 324 → 
  20 * x + 25 * y = 7000 → 
  x = 220 :=
by sorry

end NUMINAMATH_CALUDE_twenty_paise_coins_count_l33_3380


namespace NUMINAMATH_CALUDE_closest_fraction_l33_3365

def medals_won : ℚ := 20 / 120

def options : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (x : ℚ), x ∈ options ∧ 
  ∀ (y : ℚ), y ∈ options → |x - medals_won| ≤ |y - medals_won| :=
by sorry

end NUMINAMATH_CALUDE_closest_fraction_l33_3365


namespace NUMINAMATH_CALUDE_unique_natural_pair_l33_3394

theorem unique_natural_pair : 
  ∃! (k n : ℕ), 
    120 < k * n ∧ k * n < 130 ∧ 
    2 < (k : ℚ) / n ∧ (k : ℚ) / n < 3 ∧
    k = 18 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_natural_pair_l33_3394


namespace NUMINAMATH_CALUDE_perimeter_of_right_triangle_with_circles_l33_3369

/-- A right triangle with inscribed circles -/
structure RightTriangleWithCircles where
  -- The side lengths of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The radius of the inscribed circles
  r : ℝ
  -- Conditions
  right_triangle : a^2 + b^2 = c^2
  isosceles : a = b
  circle_radius : r = 2
  -- Relationship between side lengths and circle radius
  side_circle_relation : a = 4 * r

/-- The perimeter of a right triangle with inscribed circles -/
def perimeter (t : RightTriangleWithCircles) : ℝ :=
  t.a + t.b + t.c

/-- Theorem: The perimeter of the specified right triangle with inscribed circles is 16 + 8√2 -/
theorem perimeter_of_right_triangle_with_circles (t : RightTriangleWithCircles) :
  perimeter t = 16 + 8 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_of_right_triangle_with_circles_l33_3369


namespace NUMINAMATH_CALUDE_slab_rate_calculation_l33_3358

/-- Given a room with specified dimensions and total flooring cost, 
    calculate the rate per square meter for the slabs. -/
theorem slab_rate_calculation (length width total_cost : ℝ) 
    (h_length : length = 5.5)
    (h_width : width = 3.75)
    (h_total_cost : total_cost = 24750) : 
  total_cost / (length * width) = 1200 := by
  sorry


end NUMINAMATH_CALUDE_slab_rate_calculation_l33_3358


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l33_3329

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  (b^2 - 4*a*c > 0 → ∃ x : ℝ, a*x^2 + b*x + c = 0) ∧
  (∃ b c : ℝ, (∃ x : ℝ, a*x^2 + b*x + c = 0) ∧ ¬(b^2 - 4*a*c > 0)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l33_3329


namespace NUMINAMATH_CALUDE_correct_calculation_l33_3339

-- Define the variables
variable (AB : ℝ) (C : ℝ) (D : ℝ) (E : ℝ)

-- Define the conditions
def xiao_hu_error := AB * C + D * E * 10 = 39.6
def da_hu_error := AB * C * D * E = 36.9

-- State the theorem
theorem correct_calculation (h1 : xiao_hu_error AB C D E) (h2 : da_hu_error AB C D E) :
  AB * C + D * E = 26.1 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l33_3339


namespace NUMINAMATH_CALUDE_total_spending_l33_3388

/-- The amount Ben spends -/
def ben_spent : ℝ := 50

/-- The amount David spends -/
def david_spent : ℝ := 37.5

/-- The difference in spending between Ben and David -/
def spending_difference : ℝ := 12.5

/-- The difference in cost per item between Ben and David -/
def cost_difference_per_item : ℝ := 0.25

theorem total_spending :
  ben_spent + david_spent = 87.5 ∧
  ben_spent - david_spent = spending_difference ∧
  ben_spent / david_spent = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_total_spending_l33_3388


namespace NUMINAMATH_CALUDE_student_bicycle_speed_l33_3325

theorem student_bicycle_speed
  (distance : ℝ)
  (speed_ratio : ℝ)
  (time_difference : ℝ)
  (h_distance : distance = 12)
  (h_speed_ratio : speed_ratio = 1.2)
  (h_time_difference : time_difference = 1/6) :
  ∃ (speed_B : ℝ), speed_B = 12 ∧
    distance / speed_B - distance / (speed_ratio * speed_B) = time_difference :=
by sorry

end NUMINAMATH_CALUDE_student_bicycle_speed_l33_3325


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l33_3303

theorem quadratic_equation_solution (x m k : ℝ) :
  (x + m) * (x - 5) = x^2 - 3*x + k →
  k = -10 ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l33_3303


namespace NUMINAMATH_CALUDE_area_swept_is_14_l33_3327

/-- The area swept by a line segment during a transformation -/
def area_swept (length1 width1 length2 width2 : ℝ) : ℝ :=
  length1 * width1 + length2 * width2

/-- Theorem: The area swept by the line segment is 14 -/
theorem area_swept_is_14 :
  area_swept 4 2 3 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_area_swept_is_14_l33_3327


namespace NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l33_3337

theorem cos_sum_of_complex_exponentials (α β : ℝ) 
  (h1 : Complex.exp (α * Complex.I) = (4 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I)
  (h2 : Complex.exp (β * Complex.I) = -(5 / 13 : ℂ) + (12 / 13 : ℂ) * Complex.I) :
  Real.cos (α + β) = -(7 / 13) := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l33_3337


namespace NUMINAMATH_CALUDE_clothing_discount_l33_3310

theorem clothing_discount (original_price : ℝ) (first_sale_price second_sale_price : ℝ) :
  first_sale_price = (4 / 5) * original_price →
  second_sale_price = (1 - 0.4) * first_sale_price →
  second_sale_price = (12 / 25) * original_price :=
by sorry

end NUMINAMATH_CALUDE_clothing_discount_l33_3310


namespace NUMINAMATH_CALUDE_inequality_proof_l33_3396

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  (a / Real.sqrt (a^2 + 1)) + (b / Real.sqrt (b^2 + 1)) + (c / Real.sqrt (c^2 + 1)) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l33_3396


namespace NUMINAMATH_CALUDE_collinear_points_a_equals_9_l33_3305

/-- Three points (x₁, y₁), (x₂, y₂), and (x₃, y₃) are collinear if and only if
    (y₂ - y₁)*(x₃ - x₂) = (y₃ - y₂)*(x₂ - x₁) -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁)*(x₃ - x₂) = (y₃ - y₂)*(x₂ - x₁)

/-- If the points (1, 3), (2, 5), and (4, a) are collinear, then a = 9 -/
theorem collinear_points_a_equals_9 :
  collinear 1 3 2 5 4 a → a = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_collinear_points_a_equals_9_l33_3305


namespace NUMINAMATH_CALUDE_train_passing_tree_l33_3307

/-- Proves that a train 280 meters long, traveling at 72 km/hr, will take 14 seconds to pass a tree. -/
theorem train_passing_tree (train_length : ℝ) (train_speed_kmh : ℝ) (time : ℝ) :
  train_length = 280 ∧ 
  train_speed_kmh = 72 →
  time = train_length / (train_speed_kmh * (5/18)) ∧ 
  time = 14 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_tree_l33_3307


namespace NUMINAMATH_CALUDE_parabola_intersection_ratio_l33_3330

/-- Given a parabola y = 2p(x - a) where a > 0, and a line y = kx passing through the origin
    (k ≠ 0) intersecting the parabola at two points, the ratio of the sum of x-coordinates
    to the product of x-coordinates of these intersection points is equal to 1/a. -/
theorem parabola_intersection_ratio (p a k : ℝ) (ha : a > 0) (hk : k ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ 2 * p * (x - a)
  let g : ℝ → ℝ := λ x ↦ k * x
  let roots := {x : ℝ | f x = g x}
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ roots ∧ x₂ ∈ roots ∧ (x₁ + x₂) / (x₁ * x₂) = 1 / a :=
by
  sorry


end NUMINAMATH_CALUDE_parabola_intersection_ratio_l33_3330


namespace NUMINAMATH_CALUDE_rectangular_field_area_l33_3363

/-- The area of a rectangular field with one side of 15 m and a diagonal of 18 m -/
theorem rectangular_field_area : 
  ∀ (a b : ℝ), 
  a = 15 → 
  a^2 + b^2 = 18^2 → 
  a * b = 45 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l33_3363


namespace NUMINAMATH_CALUDE_race_length_proof_l33_3399

/-- The length of a race where runner A beats runner B by 35 meters in 7 seconds,
    and A's time over the course is 33 seconds. -/
def race_length : ℝ := 910

theorem race_length_proof :
  let time_A : ℝ := 33
  let lead_distance : ℝ := 35
  let lead_time : ℝ := 7
  race_length = (lead_distance * time_A) / (lead_time / time_A) := by
  sorry

#check race_length_proof

end NUMINAMATH_CALUDE_race_length_proof_l33_3399


namespace NUMINAMATH_CALUDE_cosine_of_geometric_triangle_l33_3371

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a, b, c form a geometric sequence and c = 2a, then cos B = 3/4 -/
theorem cosine_of_geometric_triangle (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_geometric : b^2 = a * c)
  (h_relation : c = 2 * a) :
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  cos_B = 3/4 := by sorry

end NUMINAMATH_CALUDE_cosine_of_geometric_triangle_l33_3371


namespace NUMINAMATH_CALUDE_max_intersections_cubic_curve_l33_3357

/-- Given a cubic curve y = x^3 - x, the maximum number of intersections
    with any tangent line passing through a point (t, 0) on the x-axis is 3 -/
theorem max_intersections_cubic_curve (t : ℝ) :
  let f (x : ℝ) := x^3 - x
  let tangent_line (x₀ : ℝ) (x : ℝ) := (3 * x₀^2 - 1) * (x - x₀) + f x₀
  ∃ (n : ℕ), n ≤ 3 ∧
    ∀ (m : ℕ), (∃ (S : Finset ℝ), S.card = m ∧
      (∀ x ∈ S, f x = tangent_line x x ∧ tangent_line x t = 0)) →
    m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_max_intersections_cubic_curve_l33_3357


namespace NUMINAMATH_CALUDE_equation_solution_l33_3367

theorem equation_solution : 
  ∀ x : ℝ, (x - 5)^2 = (1/16)⁻¹ ↔ x = 1 ∨ x = 9 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l33_3367


namespace NUMINAMATH_CALUDE_sams_morning_run_l33_3335

theorem sams_morning_run (morning_run : ℝ) 
  (store_walk : ℝ)
  (bike_ride : ℝ)
  (total_distance : ℝ)
  (h1 : store_walk = 2 * morning_run)
  (h2 : bike_ride = 12)
  (h3 : total_distance = 18)
  (h4 : morning_run + store_walk + bike_ride = total_distance) :
  morning_run = 2 := by
sorry

end NUMINAMATH_CALUDE_sams_morning_run_l33_3335


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l33_3351

theorem max_value_theorem (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 29 :=
by sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 1) = Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l33_3351


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l33_3366

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a function type for edge coloring
def EdgeColoring := Fin 6 → Fin 6 → Color

-- Main theorem
theorem monochromatic_triangle_exists (coloring : EdgeColoring) : 
  ∃ (a b c : Fin 6), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  ((coloring a b = coloring b c ∧ coloring b c = coloring a c) ∨
   (coloring a b = Color.Red ∧ coloring b c = Color.Red ∧ coloring a c = Color.Red) ∨
   (coloring a b = Color.Blue ∧ coloring b c = Color.Blue ∧ coloring a c = Color.Blue)) :=
sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l33_3366


namespace NUMINAMATH_CALUDE_polynomial_simplification_l33_3360

theorem polynomial_simplification (r : ℝ) :
  (2 * r^3 + 5 * r^2 - 4 * r + 8) - (r^3 + 9 * r^2 - 2 * r - 3) =
  r^3 - 4 * r^2 - 2 * r + 11 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l33_3360


namespace NUMINAMATH_CALUDE_lollipop_reimbursement_l33_3320

/-- Given that Sarah bought 12 lollipops for 3 dollars and shared one-quarter with Julie,
    prove that Julie reimbursed Sarah 75 cents. -/
theorem lollipop_reimbursement (total_lollipops : ℕ) (total_cost : ℚ) (share_fraction : ℚ) :
  total_lollipops = 12 →
  total_cost = 3 →
  share_fraction = 1/4 →
  (share_fraction * total_lollipops : ℚ) * (total_cost / total_lollipops) * 100 = 75 := by
  sorry

#check lollipop_reimbursement

end NUMINAMATH_CALUDE_lollipop_reimbursement_l33_3320


namespace NUMINAMATH_CALUDE_soccer_field_kids_l33_3324

/-- The number of kids initially on the soccer field -/
def initial_kids : ℕ := 14

/-- The number of kids who decided to join -/
def joining_kids : ℕ := 22

/-- The total number of kids on the soccer field after new kids join -/
def total_kids : ℕ := initial_kids + joining_kids

theorem soccer_field_kids : total_kids = 36 := by
  sorry

end NUMINAMATH_CALUDE_soccer_field_kids_l33_3324


namespace NUMINAMATH_CALUDE_bisecting_line_theorem_l33_3364

/-- The pentagon vertices -/
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (11, 0)
def C : ℝ × ℝ := (11, 2)
def D : ℝ × ℝ := (6, 2)
def E : ℝ × ℝ := (0, 8)

/-- The area of the pentagon -/
noncomputable def pentagonArea : ℝ := sorry

/-- The x-coordinate of the bisecting line -/
noncomputable def bisectingLineX : ℝ := 8 - 2 * Real.sqrt 6

/-- The area of the left part of the pentagon when divided by the line x = bisectingLineX -/
noncomputable def leftArea : ℝ := sorry

/-- The area of the right part of the pentagon when divided by the line x = bisectingLineX -/
noncomputable def rightArea : ℝ := sorry

/-- Theorem stating that the line x = 8 - 2√6 bisects the area of the pentagon -/
theorem bisecting_line_theorem : leftArea = rightArea ∧ leftArea + rightArea = pentagonArea := by sorry

end NUMINAMATH_CALUDE_bisecting_line_theorem_l33_3364


namespace NUMINAMATH_CALUDE_f_10_sqrt_3_l33_3390

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_10_sqrt_3 (f : ℝ → ℝ) 
    (hodd : OddFunction f)
    (hperiod : ∀ x, f (x + 2) = -f x)
    (hunit : ∀ x ∈ Set.Icc 0 1, f x = 2 * x) :
    f (10 * Real.sqrt 3) = -1.36 := by
  sorry

end NUMINAMATH_CALUDE_f_10_sqrt_3_l33_3390


namespace NUMINAMATH_CALUDE_integer_sum_problem_l33_3312

theorem integer_sum_problem (x y : ℤ) : 
  x > 0 → y > 0 → x - y = 12 → x * y = 45 → x + y = 18 := by sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l33_3312


namespace NUMINAMATH_CALUDE_c_investment_is_81000_l33_3397

/-- Calculates the investment of partner C in a partnership business -/
def calculate_c_investment (a_investment b_investment : ℕ) (total_profit c_profit : ℕ) : ℕ :=
  let total_investment_ab := a_investment + b_investment
  let c_investment := (c_profit * (total_investment_ab + c_profit * total_investment_ab / (total_profit - c_profit))) / total_profit
  c_investment

/-- Theorem: Given the specific investments and profits, C's investment is 81000 -/
theorem c_investment_is_81000 :
  calculate_c_investment 27000 72000 80000 36000 = 81000 := by
  sorry

end NUMINAMATH_CALUDE_c_investment_is_81000_l33_3397


namespace NUMINAMATH_CALUDE_negative_m_squared_n_identity_l33_3341

theorem negative_m_squared_n_identity (m n : ℝ) : -m^2*n - 2*m^2*n = -3*m^2*n := by
  sorry

end NUMINAMATH_CALUDE_negative_m_squared_n_identity_l33_3341


namespace NUMINAMATH_CALUDE_max_quadratic_equations_l33_3361

def is_valid_number (n : ℕ) : Prop :=
  300 ≤ n ∧ n ≤ 999 ∧ (n / 100) % 2 = 1 ∧ (n / 100) > 1

def has_real_roots (a b c : ℕ) : Prop :=
  b * b ≥ 4 * a * c

def valid_equation (a b c : ℕ) : Prop :=
  is_valid_number a ∧ is_valid_number b ∧ is_valid_number c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  has_real_roots a b c

theorem max_quadratic_equations :
  ∃ (equations : Finset (ℕ × ℕ × ℕ)),
    (∀ (e : ℕ × ℕ × ℕ), e ∈ equations → valid_equation e.1 e.2.1 e.2.2) ∧
    equations.card = 100 ∧
    (∀ (equations' : Finset (ℕ × ℕ × ℕ)),
      (∀ (e : ℕ × ℕ × ℕ), e ∈ equations' → valid_equation e.1 e.2.1 e.2.2) →
      equations'.card ≤ 100) :=
sorry

end NUMINAMATH_CALUDE_max_quadratic_equations_l33_3361


namespace NUMINAMATH_CALUDE_craig_apples_l33_3302

/-- Theorem: If Craig shares 7 apples and has 13 apples left after sharing,
    then Craig initially had 20 apples. -/
theorem craig_apples (initial : ℕ) (shared : ℕ) (remaining : ℕ)
    (h1 : shared = 7)
    (h2 : remaining = 13)
    (h3 : initial = shared + remaining) :
  initial = 20 := by
  sorry

end NUMINAMATH_CALUDE_craig_apples_l33_3302


namespace NUMINAMATH_CALUDE_oliver_workout_ratio_l33_3376

/-- Oliver's workout schedule problem -/
theorem oliver_workout_ratio :
  let monday : ℕ := 4  -- Monday's workout hours
  let tuesday : ℕ := monday - 2  -- Tuesday's workout hours
  let thursday : ℕ := 2 * tuesday  -- Thursday's workout hours
  let total : ℕ := 18  -- Total workout hours over four days
  let wednesday : ℕ := total - (monday + tuesday + thursday)  -- Wednesday's workout hours
  (wednesday : ℚ) / monday = 2 := by
  sorry

end NUMINAMATH_CALUDE_oliver_workout_ratio_l33_3376


namespace NUMINAMATH_CALUDE_root_property_l33_3344

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem root_property (f : ℝ → ℝ) (x₀ : ℝ) 
  (h_odd : is_odd f) 
  (h_root : f x₀ = Real.exp x₀) :
  f (-x₀) * Real.exp (-x₀) + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l33_3344


namespace NUMINAMATH_CALUDE_age_ratio_l33_3301

/-- Given that Billy is 4 years old and you were 12 years older than Billy when he was born,
    prove that the ratio of your current age to Billy's current age is 4:1. -/
theorem age_ratio (billy_age : ℕ) (age_difference : ℕ) : 
  billy_age = 4 → age_difference = 12 → (age_difference + billy_age) / billy_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_l33_3301


namespace NUMINAMATH_CALUDE_find_other_number_l33_3377

theorem find_other_number (A B : ℕ+) 
  (h_lcm : Nat.lcm A B = 2310)
  (h_gcd : Nat.gcd A B = 30)
  (h_A : A = 770) : 
  B = 90 := by
sorry

end NUMINAMATH_CALUDE_find_other_number_l33_3377


namespace NUMINAMATH_CALUDE_distinct_power_differences_exist_l33_3315

theorem distinct_power_differences_exist : ∃ (N : ℕ) (a₁ a₂ b₁ b₂ c₁ c₂ d₁ d₂ : ℕ),
  (∃ (x₁ x₂ : ℕ), a₁ = x₁^2 ∧ a₂ = x₂^2) ∧
  (∃ (y₁ y₂ : ℕ), b₁ = y₁^3 ∧ b₂ = y₂^3) ∧
  (∃ (z₁ z₂ : ℕ), c₁ = z₁^5 ∧ c₂ = z₂^5) ∧
  (∃ (w₁ w₂ : ℕ), d₁ = w₁^7 ∧ d₂ = w₂^7) ∧
  N = a₁ - a₂ ∧
  N = b₁ - b₂ ∧
  N = c₁ - c₂ ∧
  N = d₁ - d₂ ∧
  a₁ ≠ b₁ ∧ a₁ ≠ c₁ ∧ a₁ ≠ d₁ ∧ b₁ ≠ c₁ ∧ b₁ ≠ d₁ ∧ c₁ ≠ d₁ :=
by sorry

end NUMINAMATH_CALUDE_distinct_power_differences_exist_l33_3315


namespace NUMINAMATH_CALUDE_new_supervisor_salary_l33_3342

-- Define the number of workers
def num_workers : ℕ := 8

-- Define the total number of people (workers + supervisor)
def total_people : ℕ := num_workers + 1

-- Define the initial average salary
def initial_average : ℚ := 430

-- Define the old supervisor's salary
def old_supervisor_salary : ℚ := 870

-- Define the new average salary
def new_average : ℚ := 390

-- Theorem to prove
theorem new_supervisor_salary :
  ∃ (workers_total_salary new_supervisor_salary : ℚ),
    (workers_total_salary + old_supervisor_salary) / total_people = initial_average ∧
    workers_total_salary / num_workers ≤ old_supervisor_salary ∧
    (workers_total_salary + new_supervisor_salary) / total_people = new_average ∧
    new_supervisor_salary = 510 :=
sorry

end NUMINAMATH_CALUDE_new_supervisor_salary_l33_3342


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l33_3326

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a ≤ -2 → ∀ x y, -1 ≤ x ∧ x ≤ y → f a x ≤ f a y) ∧
  (∃ a', a' > -2 ∧ ∀ x y, -1 ≤ x ∧ x ≤ y → f a' x ≤ f a' y) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l33_3326


namespace NUMINAMATH_CALUDE_spherical_coordinate_conversion_l33_3386

/-- Proves that the given spherical coordinates are equivalent to the standard representation -/
theorem spherical_coordinate_conversion (ρ θ φ : Real) :
  ρ > 0 →
  0 ≤ θ ∧ θ < 2 * π →
  0 ≤ φ ∧ φ ≤ π →
  (ρ, θ, φ) = (4, 4 * π / 3, π / 5) ↔ (ρ, θ, φ) = (4, π / 3, 9 * π / 5) :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_conversion_l33_3386


namespace NUMINAMATH_CALUDE_selling_price_loss_percentage_l33_3384

theorem selling_price_loss_percentage (cost_price : ℝ) 
  (h : cost_price > 0) : 
  let selling_price_100 := 40 * cost_price
  let cost_price_100 := 100 * cost_price
  (selling_price_100 / cost_price_100) * 100 = 40 → 
  ((cost_price_100 - selling_price_100) / cost_price_100) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_selling_price_loss_percentage_l33_3384


namespace NUMINAMATH_CALUDE_locus_characterization_l33_3331

def has_solution (u v : ℝ) (n : ℕ) : Prop :=
  ∃ x y : ℝ, (Real.sin x)^(2*n) + (Real.cos y)^(2*n) = u ∧ (Real.sin x)^n + (Real.cos y)^n = v

theorem locus_characterization (u v : ℝ) (n : ℕ) :
  has_solution u v n ↔ 
    (v^2 ≤ 2*u ∧ (v - 1)^2 ≥ (u - 1)) ∧
    ((n % 2 = 0 → (0 ≤ v ∧ v ≤ 2 ∧ v^2 ≥ u)) ∧
     (n % 2 = 1 → (-2 ≤ v ∧ v ≤ 2 ∧ (v + 1)^2 ≥ (u - 1)))) :=
by sorry

end NUMINAMATH_CALUDE_locus_characterization_l33_3331


namespace NUMINAMATH_CALUDE_water_consumption_l33_3353

theorem water_consumption (yesterday_amount : ℝ) (percentage_decrease : ℝ) 
  (h1 : yesterday_amount = 48)
  (h2 : percentage_decrease = 4)
  (h3 : yesterday_amount = (100 - percentage_decrease) / 100 * two_days_ago_amount) :
  two_days_ago_amount = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_water_consumption_l33_3353


namespace NUMINAMATH_CALUDE_abc_inequality_l33_3308

theorem abc_inequality (a b c : ℝ) (ha : -1 < a ∧ a < 1) (hb : -1 < b ∧ b < 1) (hc : -1 < c ∧ c < 1) :
  a * b * c + 2 > a + b + c := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l33_3308


namespace NUMINAMATH_CALUDE_proportional_function_quadrants_l33_3350

/-- A proportional function passing through the second and fourth quadrants has a negative coefficient. -/
theorem proportional_function_quadrants (k : ℝ) :
  (∀ x y : ℝ, y = k * x →
    ((x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))) →
  k < 0 := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_quadrants_l33_3350


namespace NUMINAMATH_CALUDE_sum_remainders_divisible_by_500_l33_3378

/-- The set of all possible remainders when 3^n (n is a nonnegative integer) is divided by 500 -/
def R : Finset ℕ :=
  sorry

/-- The sum of all elements in R -/
def S : ℕ := sorry

/-- Theorem: The sum of all distinct remainders when 3^n (n is a nonnegative integer) 
    is divided by 500 is divisible by 500 -/
theorem sum_remainders_divisible_by_500 : 500 ∣ S := by
  sorry

end NUMINAMATH_CALUDE_sum_remainders_divisible_by_500_l33_3378


namespace NUMINAMATH_CALUDE_sum_reciprocal_F_powers_of_two_converges_to_one_l33_3374

/-- Definition of the sequence F -/
def F : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | (n+2) => (3/2) * F (n+1) - (1/2) * F n

/-- The sum of the reciprocals of F(2^n) converges to 1 -/
theorem sum_reciprocal_F_powers_of_two_converges_to_one :
  ∑' n, (1 : ℝ) / F (2^n) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_F_powers_of_two_converges_to_one_l33_3374


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_complementary_l33_3373

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Green : Card
| Blue : Card

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "A receives the red card"
def A_gets_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "B receives the red card"
def B_gets_red (d : Distribution) : Prop := d Person.B = Card.Red

-- State the theorem
theorem events_mutually_exclusive_but_not_complementary :
  (∀ d : Distribution, ¬(A_gets_red d ∧ B_gets_red d)) ∧
  (∃ d : Distribution, ¬(A_gets_red d ∨ B_gets_red d)) :=
by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_complementary_l33_3373


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l33_3311

theorem triangle_angle_sum (A B C : ℝ) (h : A + B = 80) : C = 100 :=
  by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l33_3311


namespace NUMINAMATH_CALUDE_min_abc_value_l33_3355

-- Define the set M
def M : Set ℝ := {x | 2/3 < x ∧ x < 2}

-- Define t as the largest positive integer in M
def t : ℕ := 1

-- Theorem statement
theorem min_abc_value (a b c : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (h_abc : (a - 1) * (b - 1) * (c - 1) = t) :
  a * b * c ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_abc_value_l33_3355


namespace NUMINAMATH_CALUDE_function_periodicity_l33_3359

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_periodicity (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 2) = f (2 - x))
  (h2 : ∀ x, f (x + 7) = f (7 - x)) :
  is_periodic f 10 := by
sorry

end NUMINAMATH_CALUDE_function_periodicity_l33_3359


namespace NUMINAMATH_CALUDE_green_balls_count_l33_3370

theorem green_balls_count (total : ℕ) (white yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 ∧
  white = 50 ∧
  yellow = 8 ∧
  red = 9 ∧
  purple = 3 ∧
  prob_not_red_purple = 88/100 →
  ∃ green : ℕ, green = 30 ∧ white + yellow + green + red + purple = total ∧
  (white + yellow + green : ℚ) / total = prob_not_red_purple :=
by sorry

end NUMINAMATH_CALUDE_green_balls_count_l33_3370


namespace NUMINAMATH_CALUDE_max_sin_C_in_triangle_l33_3346

theorem max_sin_C_in_triangle (A B C : Real) (h : ∀ A B C, (1 / Real.tan A) + (1 / Real.tan B) = 6 / Real.tan C) :
  ∃ (max_sin_C : Real), max_sin_C = Real.sqrt 15 / 4 ∧ ∀ (sin_C : Real), sin_C ≤ max_sin_C := by
  sorry

end NUMINAMATH_CALUDE_max_sin_C_in_triangle_l33_3346


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l33_3345

/-- A line is tangent to a parabola if and only if the discriminant of the resulting quadratic equation is zero -/
theorem line_tangent_to_parabola (k : ℝ) :
  (∃ x y : ℝ, 3 * x + 5 * y + k = 0 ∧ y^2 = 24 * x ∧
    ∀ x' y' : ℝ, 3 * x' + 5 * y' + k = 0 ∧ y'^2 = 24 * x' → (x', y') = (x, y))
  ↔ k = 50 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l33_3345


namespace NUMINAMATH_CALUDE_multiply_three_point_six_by_zero_point_twenty_five_l33_3333

theorem multiply_three_point_six_by_zero_point_twenty_five :
  3.6 * 0.25 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_three_point_six_by_zero_point_twenty_five_l33_3333


namespace NUMINAMATH_CALUDE_empty_pencil_cases_l33_3313

theorem empty_pencil_cases (total : ℕ) (pencils : ℕ) (pens : ℕ) (both : ℕ) (empty : ℕ) : 
  total = 10 ∧ pencils = 5 ∧ pens = 4 ∧ both = 2 → empty = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_empty_pencil_cases_l33_3313


namespace NUMINAMATH_CALUDE_pastries_sold_l33_3336

theorem pastries_sold (cupcakes cookies left : ℕ) 
  (h1 : cupcakes = 7) 
  (h2 : cookies = 5) 
  (h3 : left = 8) : 
  cupcakes + cookies - left = 4 := by
  sorry

end NUMINAMATH_CALUDE_pastries_sold_l33_3336


namespace NUMINAMATH_CALUDE_garden_breadth_l33_3381

/-- The breadth of a rectangular garden with given perimeter and length -/
theorem garden_breadth (perimeter length : ℝ) (h₁ : perimeter = 950) (h₂ : length = 375) :
  perimeter = 2 * (length + 100) := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_l33_3381


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l33_3362

theorem marble_fraction_after_tripling (total : ℚ) (h_pos : total > 0) : 
  let green := (4/7) * total
  let blue := (1/7) * total
  let initial_white := total - green - blue
  let new_white := 3 * initial_white
  let new_total := green + blue + new_white
  new_white / new_total = 6/11 := by sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l33_3362


namespace NUMINAMATH_CALUDE_additional_employees_hired_l33_3323

/-- Calculates the number of additional employees hired by a company --/
theorem additional_employees_hired (
  initial_employees : ℕ)
  (hourly_wage : ℚ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (new_total_wages : ℚ)
  (h1 : initial_employees = 500)
  (h2 : hourly_wage = 12)
  (h3 : hours_per_day = 10)
  (h4 : days_per_week = 5)
  (h5 : weeks_per_month = 4)
  (h6 : new_total_wages = 1680000) :
  (new_total_wages - (initial_employees * hourly_wage * hours_per_day * days_per_week * weeks_per_month)) / 
  (hourly_wage * hours_per_day * days_per_week * weeks_per_month) = 200 := by
  sorry

#check additional_employees_hired

end NUMINAMATH_CALUDE_additional_employees_hired_l33_3323


namespace NUMINAMATH_CALUDE_infinite_power_tower_equals_four_l33_3309

-- Define the infinite power tower function
noncomputable def powerTower (x : ℝ) : ℝ := Real.sqrt (4 : ℝ)

-- State the theorem
theorem infinite_power_tower_equals_four (x : ℝ) (h₁ : x > 0) :
  powerTower x = 4 → x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_power_tower_equals_four_l33_3309


namespace NUMINAMATH_CALUDE_time_for_type_A_is_60_l33_3356

/-- Represents the time allocation for an examination with different problem types. -/
structure ExamTime where
  totalQuestions : ℕ
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ
  totalTime : ℕ
  lastHour : ℕ

/-- Calculates the time spent on Type A problems in the examination. -/
def timeForTypeA (e : ExamTime) : ℕ :=
  let typeB_time := (e.lastHour * 2) / e.typeC
  (e.typeA * typeB_time * 2)

/-- Theorem stating that the time spent on Type A problems is 60 minutes. -/
theorem time_for_type_A_is_60 (e : ExamTime) 
  (h1 : e.totalQuestions = 200)
  (h2 : e.typeA = 20)
  (h3 : e.typeB = 100)
  (h4 : e.typeC = 80)
  (h5 : e.totalTime = 180)
  (h6 : e.lastHour = 60) :
  timeForTypeA e = 60 := by
  sorry

end NUMINAMATH_CALUDE_time_for_type_A_is_60_l33_3356


namespace NUMINAMATH_CALUDE_improper_fraction_subtraction_l33_3349

theorem improper_fraction_subtraction (a b n : ℕ) 
  (h1 : a > b) 
  (h2 : n < b) : 
  (a - n : ℚ) / (b - n) > (a : ℚ) / b := by
sorry

end NUMINAMATH_CALUDE_improper_fraction_subtraction_l33_3349


namespace NUMINAMATH_CALUDE_race_speed_ratio_l33_3300

/-- 
Given two runners a and b, where:
- a's speed is some multiple of b's speed
- a gives b a head start of 1/16 of the race length
- They finish at the same time (dead heat)
Then the ratio of a's speed to b's speed is 15/16
-/
theorem race_speed_ratio (v_a v_b : ℝ) (h : v_a > 0 ∧ v_b > 0) :
  (∃ k : ℝ, v_a = k * v_b) →
  (v_a * 1 = v_b * (15/16)) →
  v_a / v_b = 15/16 := by
sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l33_3300


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l33_3368

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l33_3368


namespace NUMINAMATH_CALUDE_orange_calories_l33_3317

/-- Proves that the number of calories per orange is 80 given the problem conditions -/
theorem orange_calories (orange_cost : ℚ) (initial_amount : ℚ) (required_calories : ℕ) (remaining_amount : ℚ) :
  orange_cost = 6/5 ∧ 
  initial_amount = 10 ∧ 
  required_calories = 400 ∧ 
  remaining_amount = 4 →
  (initial_amount - remaining_amount) / orange_cost * required_calories / ((initial_amount - remaining_amount) / orange_cost) = 80 := by
sorry

end NUMINAMATH_CALUDE_orange_calories_l33_3317


namespace NUMINAMATH_CALUDE_average_speed_calculation_l33_3343

theorem average_speed_calculation (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  let first_half_distance := total_distance / 2
  let first_half_time := first_half_distance / first_half_speed
  let second_half_time := first_half_time * second_half_time_factor
  let total_time := first_half_time + second_half_time
  (total_distance / total_time) = 40 := by
sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l33_3343


namespace NUMINAMATH_CALUDE_no_solution_to_equation_l33_3352

theorem no_solution_to_equation :
  ¬∃ s : ℝ, (s^2 - 6*s + 8) / (s^2 - 9*s + 20) = (s^2 - 3*s - 18) / (s^2 - 2*s - 15) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_to_equation_l33_3352


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l33_3347

theorem triangle_angle_calculation (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 2 →
  b = Real.sqrt 3 →
  B = π / 3 →  -- 60° in radians
  (a / Real.sin A = b / Real.sin B) →  -- Law of Sines
  A = π / 4  -- 45° in radians
:= by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l33_3347


namespace NUMINAMATH_CALUDE_average_trees_planted_l33_3306

theorem average_trees_planted (trees_A trees_B trees_C : ℕ) : 
  trees_A = 225 →
  trees_B = trees_A + 48 →
  trees_C = trees_A - 24 →
  (trees_A + trees_B + trees_C) / 3 = 233 := by
  sorry

end NUMINAMATH_CALUDE_average_trees_planted_l33_3306


namespace NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l33_3321

theorem consecutive_integers_product_plus_one_is_square (n : ℤ) :
  ∃ m : ℤ, n * (n + 1) * (n + 2) * (n + 3) + 1 = m^2 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l33_3321


namespace NUMINAMATH_CALUDE_hexagon_minus_sectors_area_l33_3385

/-- The area of the region inside a regular hexagon but outside circular sectors --/
theorem hexagon_minus_sectors_area (s : ℝ) (r : ℝ) (θ : ℝ) : 
  s = 10 → r = 5 → θ = 120 → 
  (6 * (s^2 * Real.sqrt 3 / 4)) - (6 * (θ / 360) * Real.pi * r^2) = 150 * Real.sqrt 3 - 50 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_hexagon_minus_sectors_area_l33_3385


namespace NUMINAMATH_CALUDE_local_road_speed_l33_3340

theorem local_road_speed (local_distance : ℝ) (highway_distance : ℝ) 
  (highway_speed : ℝ) (average_speed : ℝ) (local_speed : ℝ) : 
  local_distance = 60 ∧ 
  highway_distance = 65 ∧ 
  highway_speed = 65 ∧ 
  average_speed = 41.67 ∧
  (local_distance + highway_distance) / ((local_distance / local_speed) + (highway_distance / highway_speed)) = average_speed →
  local_speed = 30 := by
sorry

end NUMINAMATH_CALUDE_local_road_speed_l33_3340


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_reciprocals_l33_3354

theorem quadratic_roots_sum_reciprocals (a b : ℝ) 
  (ha : a^2 + a - 1 = 0) (hb : b^2 + b - 1 = 0) : 
  a/b + b/a = 2 ∨ a/b + b/a = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_reciprocals_l33_3354


namespace NUMINAMATH_CALUDE_solution_sets_equality_l33_3393

theorem solution_sets_equality (a b : ℝ) : 
  (∀ x, |x - 2| > 1 ↔ x^2 + a*x + b > 0) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_equality_l33_3393


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l33_3372

/-- An isosceles triangle with sides of 4cm and 8cm has a perimeter of 20cm -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive sides
  (a = 4 ∧ b = 8) ∨ (a = 8 ∧ b = 4) →  -- given side lengths
  (a = b ∨ b = c ∨ a = c) →  -- isosceles condition
  a + b + c = 20 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l33_3372


namespace NUMINAMATH_CALUDE_third_term_of_geometric_series_l33_3338

/-- Given an infinite geometric series with common ratio 1/4 and sum 16,
    the third term of the series is 3/4. -/
theorem third_term_of_geometric_series :
  ∀ (a : ℝ),
  (a / (1 - (1/4 : ℝ)) = 16) →  -- Sum of infinite geometric series
  (a * (1/4 : ℝ)^2 = 3/4) :=    -- Third term of the series
by sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_series_l33_3338


namespace NUMINAMATH_CALUDE_corrected_mean_l33_3375

theorem corrected_mean (n : ℕ) (incorrect_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) :
  n = 50 ∧ incorrect_mean = 30 ∧ incorrect_value = 23 ∧ correct_value = 48 →
  (n : ℚ) * incorrect_mean - incorrect_value + correct_value = n * (30.5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l33_3375


namespace NUMINAMATH_CALUDE_complex_equation_solution_l33_3391

theorem complex_equation_solution (z : ℂ) : (2 - Complex.I) * z = 4 + 3 * Complex.I → z = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l33_3391


namespace NUMINAMATH_CALUDE_right_triangle_altitude_segment_ratio_l33_3389

theorem right_triangle_altitude_segment_ratio :
  ∀ (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0),
  a^2 + b^2 = c^2 →  -- right triangle condition
  a = 3 * b →        -- leg ratio condition
  ∃ (d e : ℝ), d > 0 ∧ e > 0 ∧ d + e = c ∧ d / e = 9 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_segment_ratio_l33_3389


namespace NUMINAMATH_CALUDE_sin4_tan2_product_positive_l33_3332

theorem sin4_tan2_product_positive :
  ∀ (sin4 tan2 : ℝ), sin4 < 0 → tan2 < 0 → sin4 * tan2 > 0 := by sorry

end NUMINAMATH_CALUDE_sin4_tan2_product_positive_l33_3332


namespace NUMINAMATH_CALUDE_speed_conversion_l33_3334

-- Define the conversion factor from m/s to km/h
def meters_per_second_to_kmph : ℝ := 3.6

-- Define the given speed in meters per second
def speed_ms : ℝ := 200.016

-- Define the speed in km/h that we want to prove
def speed_kmph : ℝ := 720.0576

-- Theorem statement
theorem speed_conversion :
  speed_ms * meters_per_second_to_kmph = speed_kmph :=
by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l33_3334


namespace NUMINAMATH_CALUDE_equation_solution_l33_3395

theorem equation_solution : ∃! x : ℝ, (x^2 + 2*x + 3) / (x + 2) = x + 3 := by
  use -1
  constructor
  · -- Prove that x = -1 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l33_3395


namespace NUMINAMATH_CALUDE_min_white_pairs_8x8_20black_l33_3328

/-- Represents a grid with black and white cells -/
structure Grid :=
  (size : ℕ)
  (black_cells : ℕ)

/-- Calculates the total number of adjacent cell pairs in a square grid -/
def total_pairs (g : Grid) : ℕ :=
  2 * (g.size - 1) * g.size

/-- Calculates the maximum number of central black cells that can be placed without adjacency -/
def max_central_black (g : Grid) : ℕ :=
  (g.size - 2)^2 / 2

/-- Calculates the minimum number of adjacent white cell pairs -/
def min_white_pairs (g : Grid) : ℕ :=
  total_pairs g - (60 + min g.black_cells (max_central_black g))

/-- Theorem stating the minimum number of adjacent white cell pairs for an 8x8 grid with 20 black cells -/
theorem min_white_pairs_8x8_20black :
  let g : Grid := { size := 8, black_cells := 20 }
  min_white_pairs g = 34 := by
  sorry

end NUMINAMATH_CALUDE_min_white_pairs_8x8_20black_l33_3328


namespace NUMINAMATH_CALUDE_parabola_points_order_l33_3387

theorem parabola_points_order : 
  let y₁ : ℝ := -1/2 * (-2)^2 + 2 * (-2)
  let y₂ : ℝ := -1/2 * (-1)^2 + 2 * (-1)
  let y₃ : ℝ := -1/2 * 8^2 + 2 * 8
  y₃ < y₁ ∧ y₁ < y₂ := by sorry

end NUMINAMATH_CALUDE_parabola_points_order_l33_3387


namespace NUMINAMATH_CALUDE_product_543_7_base9_l33_3304

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (n : ℕ) : ℕ :=
  sorry

/-- Converts a base-10 number to base-9 --/
def base10ToBase9 (n : ℕ) : ℕ :=
  sorry

/-- Multiplies two base-9 numbers and returns the result in base-9 --/
def multiplyBase9 (a b : ℕ) : ℕ :=
  base10ToBase9 (base9ToBase10 a * base9ToBase10 b)

theorem product_543_7_base9 :
  multiplyBase9 543 7 = 42333 :=
sorry

end NUMINAMATH_CALUDE_product_543_7_base9_l33_3304


namespace NUMINAMATH_CALUDE_ohara_triple_49_16_l33_3316

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x

/-- Theorem: The value of x in the O'Hara triple (49, 16, x) is 11 -/
theorem ohara_triple_49_16 :
  ∃ x : ℕ, is_ohara_triple 49 16 x ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_49_16_l33_3316


namespace NUMINAMATH_CALUDE_correct_calculation_l33_3383

theorem correct_calculation (x : ℤ) : 
  x + 238 = 637 → x - 382 = 17 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l33_3383


namespace NUMINAMATH_CALUDE_cubic_roots_determinant_l33_3348

theorem cubic_roots_determinant (r s t : ℝ) (a b c : ℝ) : 
  a^3 - r*a^2 + s*a + t = 0 →
  b^3 - r*b^2 + s*b + t = 0 →
  c^3 - r*c^2 + s*c + t = 0 →
  Matrix.det !![1 + a^2, 1, 1; 1, 1 + b^2, 1; 1, 1, 1 + c^2] = r^2 + s^2 - 2*t :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_determinant_l33_3348


namespace NUMINAMATH_CALUDE_lcm_of_5_8_12_20_l33_3314

theorem lcm_of_5_8_12_20 : Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 12 20)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_5_8_12_20_l33_3314


namespace NUMINAMATH_CALUDE_fib_inequality_l33_3319

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Proof of the inequality for Fibonacci numbers -/
theorem fib_inequality (n : ℕ) (hn : n > 0) :
  (fib (n + 2) : ℝ) ^ (1 / n : ℝ) ≥ 1 + 1 / ((fib (n + 1) : ℝ) ^ (1 / n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_fib_inequality_l33_3319
