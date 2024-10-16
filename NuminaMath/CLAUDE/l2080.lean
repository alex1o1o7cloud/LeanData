import Mathlib

namespace NUMINAMATH_CALUDE_find_z_when_y_is_6_l2080_208060

-- Define the direct variation relationship
def varies_directly (y z : ℝ) : Prop := ∃ k : ℝ, y^3 = k * z^(1/3)

-- State the theorem
theorem find_z_when_y_is_6 (y z : ℝ) (h1 : varies_directly y z) (h2 : y = 3 ∧ z = 8) :
  y = 6 → z = 4096 := by
  sorry

end NUMINAMATH_CALUDE_find_z_when_y_is_6_l2080_208060


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l2080_208079

/-- The equation of a conic section -/
def conic_equation (x y : ℝ) : Prop :=
  (3*x - 2)^2 - 2*(5*y + 1)^2 = 288

/-- Definition of a hyperbola -/
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
    ∀ x y, f x y ↔ a * x^2 + b * y^2 + c * x + d * y + e = 0

/-- Theorem: The given conic equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l2080_208079


namespace NUMINAMATH_CALUDE_rice_mixture_price_l2080_208001

/-- Proves that the price of the second type of rice is 9.60 Rs/kg -/
theorem rice_mixture_price (price1 : ℝ) (weight1 : ℝ) (weight2 : ℝ) (mixture_price : ℝ) 
  (h1 : price1 = 6.60)
  (h2 : weight1 = 49)
  (h3 : weight2 = 56)
  (h4 : mixture_price = 8.20)
  (h5 : weight1 + weight2 = 105) :
  ∃ (price2 : ℝ), price2 = 9.60 ∧ 
  (price1 * weight1 + price2 * weight2) / (weight1 + weight2) = mixture_price :=
by sorry

end NUMINAMATH_CALUDE_rice_mixture_price_l2080_208001


namespace NUMINAMATH_CALUDE_circle_line_symmetry_l2080_208096

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane of the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two points are symmetric with respect to a line -/
def symmetric_points (p q : ℝ × ℝ) (l : Line) : Prop :=
  sorry

theorem circle_line_symmetry (c : Circle) (l : Line) :
  c.center.1 = -1 ∧ c.center.2 = 3 ∧ c.radius = 3 ∧
  l.a = 1 ∧ l.c = 4 ∧
  ∃ (p q : ℝ × ℝ), (p.1 + 1)^2 + (p.2 - 3)^2 = 9 ∧
                   (q.1 + 1)^2 + (q.2 - 3)^2 = 9 ∧
                   symmetric_points p q l →
  l.b = -1 :=
sorry

end NUMINAMATH_CALUDE_circle_line_symmetry_l2080_208096


namespace NUMINAMATH_CALUDE_stating_regular_duck_price_is_correct_l2080_208094

/-- The price of a regular size rubber duck in the city's charity race. -/
def regular_duck_price : ℚ :=
  3

/-- The price of a large size rubber duck in the city's charity race. -/
def large_duck_price : ℚ :=
  5

/-- The number of regular size ducks sold in the charity race. -/
def regular_ducks_sold : ℕ :=
  221

/-- The number of large size ducks sold in the charity race. -/
def large_ducks_sold : ℕ :=
  185

/-- The total amount raised in the charity race. -/
def total_raised : ℚ :=
  1588

/-- 
Theorem stating that the regular duck price is correct given the conditions of the charity race.
-/
theorem regular_duck_price_is_correct :
  regular_duck_price * regular_ducks_sold + large_duck_price * large_ducks_sold = total_raised :=
by sorry

end NUMINAMATH_CALUDE_stating_regular_duck_price_is_correct_l2080_208094


namespace NUMINAMATH_CALUDE_unit_square_quadrilateral_inequalities_l2080_208064

/-- A quadrilateral formed by selecting one point on each side of a unit square -/
structure UnitSquareQuadrilateral where
  a : Real
  b : Real
  c : Real
  d : Real
  a_nonneg : 0 ≤ a
  b_nonneg : 0 ≤ b
  c_nonneg : 0 ≤ c
  d_nonneg : 0 ≤ d
  a_le_one : a ≤ 1
  b_le_one : b ≤ 1
  c_le_one : c ≤ 1
  d_le_one : d ≤ 1

theorem unit_square_quadrilateral_inequalities (q : UnitSquareQuadrilateral) :
  2 ≤ q.a ^ 2 + q.b ^ 2 + q.c ^ 2 + q.d ^ 2 ∧
  q.a ^ 2 + q.b ^ 2 + q.c ^ 2 + q.d ^ 2 ≤ 4 ∧
  2 * Real.sqrt 2 ≤ q.a + q.b + q.c + q.d ∧
  q.a + q.b + q.c + q.d ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_unit_square_quadrilateral_inequalities_l2080_208064


namespace NUMINAMATH_CALUDE_campaign_donation_proof_l2080_208084

theorem campaign_donation_proof (max_donors : ℕ) (half_donors : ℕ) (total_raised : ℚ) 
  (h1 : max_donors = 500)
  (h2 : half_donors = 3 * max_donors)
  (h3 : total_raised = 3750000)
  (h4 : (max_donors * x + half_donors * (x / 2)) / total_raised = 2 / 5) :
  x = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_campaign_donation_proof_l2080_208084


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2080_208042

/-- Given a hyperbola with standard equation x²/36 - y²/64 = 1, 
    prove its asymptote equations and eccentricity. -/
theorem hyperbola_properties :
  let a : ℝ := 6
  let b : ℝ := 8
  let c : ℝ := (a^2 + b^2).sqrt
  let asymptote (x : ℝ) : ℝ := (b / a) * x
  let eccentricity : ℝ := c / a
  (∀ x y : ℝ, x^2 / 36 - y^2 / 64 = 1 → 
    (y = asymptote x ∨ y = -asymptote x) ∧ eccentricity = 5/3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2080_208042


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2080_208089

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), ((-2 : ℝ) * m + 2 * m + 1 = 1) := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2080_208089


namespace NUMINAMATH_CALUDE_boys_total_toys_l2080_208054

/-- The number of toys Bill has -/
def bill_toys : ℕ := 60

/-- The number of toys Hash has -/
def hash_toys : ℕ := bill_toys / 2 + 9

/-- The total number of toys both boys have -/
def total_toys : ℕ := bill_toys + hash_toys

theorem boys_total_toys : total_toys = 99 := by
  sorry

end NUMINAMATH_CALUDE_boys_total_toys_l2080_208054


namespace NUMINAMATH_CALUDE_trajectory_equation_l2080_208051

/-- The trajectory of a point whose sum of distances to the coordinate axes is 6 -/
theorem trajectory_equation (x y : ℝ) : 
  (dist x 0 + dist y 0 = 6) → (|x| + |y| = 6) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2080_208051


namespace NUMINAMATH_CALUDE_special_gp_ratio_is_one_l2080_208028

/-- A geometric progression with positive terms where any term is the product of the next two -/
structure SpecialGP where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : a > 0
  r_pos : r > 0
  term_product : ∀ n : ℕ, a * r^n = (a * r^(n+1)) * (a * r^(n+2))

/-- The common ratio of a SpecialGP is 1 -/
theorem special_gp_ratio_is_one (gp : SpecialGP) : gp.r = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_gp_ratio_is_one_l2080_208028


namespace NUMINAMATH_CALUDE_equal_adjacent_sides_not_imply_square_l2080_208026

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define properties of quadrilaterals
def has_equal_adjacent_sides (q : Quadrilateral) : Prop :=
  sorry

def is_square (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statement
theorem equal_adjacent_sides_not_imply_square :
  ∃ q : Quadrilateral, has_equal_adjacent_sides q ∧ ¬ is_square q :=
sorry

end NUMINAMATH_CALUDE_equal_adjacent_sides_not_imply_square_l2080_208026


namespace NUMINAMATH_CALUDE_diophantine_equation_unique_solution_l2080_208013

theorem diophantine_equation_unique_solution :
  ∀ a b c : ℤ, a^2 = 2*b^2 + 3*c^2 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_unique_solution_l2080_208013


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2080_208015

/-- The repeating decimal 0.363636... expressed as a real number -/
def repeating_decimal : ℚ := 0.363636

/-- Theorem stating that the repeating decimal 0.363636... is equal to 4/11 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2080_208015


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l2080_208032

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(2*x - 1) - 2
  f (1/2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l2080_208032


namespace NUMINAMATH_CALUDE_xyz_inequality_l2080_208010

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z ≥ x * y + y * z + z * x) : 
  Real.sqrt (x * y * z) ≥ Real.sqrt x + Real.sqrt y + Real.sqrt z := by
sorry

end NUMINAMATH_CALUDE_xyz_inequality_l2080_208010


namespace NUMINAMATH_CALUDE_number_problem_l2080_208018

theorem number_problem (x : ℝ) : 0.35 * x = 0.50 * x - 24 → x = 160 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2080_208018


namespace NUMINAMATH_CALUDE_fraction_product_l2080_208044

theorem fraction_product : (3 : ℚ) / 4 * 4 / 5 * 5 / 6 * 6 / 7 = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l2080_208044


namespace NUMINAMATH_CALUDE_chennys_friends_l2080_208088

theorem chennys_friends (initial_candies : ℕ) (additional_candies : ℕ) (candies_per_friend : ℕ) : 
  initial_candies = 10 →
  additional_candies = 4 →
  candies_per_friend = 2 →
  (initial_candies + additional_candies) / candies_per_friend = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_chennys_friends_l2080_208088


namespace NUMINAMATH_CALUDE_cubic_discriminant_example_l2080_208086

/-- The discriminant of a cubic equation ax^3 + bx^2 + cx + d -/
def cubic_discriminant (a b c d : ℝ) : ℝ :=
  -27 * a^2 * d^2 + 18 * a * b * c * d - 4 * b^3 * d + b^2 * c^2 - 4 * a * c^3

/-- The coefficients of the cubic equation x^3 - 2x^2 + 5x + 2 -/
def a : ℝ := 1
def b : ℝ := -2
def c : ℝ := 5
def d : ℝ := 2

theorem cubic_discriminant_example : cubic_discriminant a b c d = -640 := by
  sorry

end NUMINAMATH_CALUDE_cubic_discriminant_example_l2080_208086


namespace NUMINAMATH_CALUDE_vector_problem_l2080_208073

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, -3)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem vector_problem (c : ℝ × ℝ) :
  perpendicular c (a.1 + b.1, a.2 + b.2) ∧
  parallel b (a.1 - c.1, a.2 - c.2) →
  c = (7/9, 7/3) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l2080_208073


namespace NUMINAMATH_CALUDE_special_triangle_properties_l2080_208025

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Vector m
  m : ℝ × ℝ
  -- Vector n
  n : ℝ × ℝ
  -- Conditions
  m_def : m = (Real.cos B, Real.cos C)
  n_def : n = (2 * a + c, b)
  perpendicular : m.1 * n.1 + m.2 * n.2 = 0
  b_value : b = Real.sqrt 13
  a_c_sum : a + c = 4

/-- The main theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  -- 1. The area of the triangle
  (1 / 2 : ℝ) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 4 ∧
  -- 2. The range of sin²A + sin²C
  (∀ x, ((Real.sin t.A)^2 + (Real.sin t.C)^2 = x) → (1 / 2 ≤ x ∧ x < 3 / 4)) :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l2080_208025


namespace NUMINAMATH_CALUDE_final_balance_l2080_208046

def account_balance (initial : ℕ) (coffee_beans : ℕ) (tumbler : ℕ) (coffee_filter : ℕ) (refund : ℕ) : ℕ :=
  initial - (coffee_beans + tumbler + coffee_filter) + refund

theorem final_balance :
  account_balance 50 10 30 5 20 = 25 := by
  sorry

end NUMINAMATH_CALUDE_final_balance_l2080_208046


namespace NUMINAMATH_CALUDE_car_down_payment_l2080_208012

/-- Given a total down payment to be split equally among a number of people,
    rounding up to the nearest dollar, calculate the amount each person must pay. -/
def splitPayment (total : ℕ) (people : ℕ) : ℕ :=
  (total + people - 1) / people

theorem car_down_payment :
  splitPayment 3500 3 = 1167 := by
  sorry

end NUMINAMATH_CALUDE_car_down_payment_l2080_208012


namespace NUMINAMATH_CALUDE_blue_tickets_per_red_l2080_208034

/-- The number of yellow tickets needed to win a Bible -/
def yellow_tickets_needed : ℕ := 10

/-- The number of red tickets needed for one yellow ticket -/
def red_per_yellow : ℕ := 10

/-- The number of yellow tickets Tom has -/
def tom_yellow : ℕ := 8

/-- The number of red tickets Tom has -/
def tom_red : ℕ := 3

/-- The number of blue tickets Tom has -/
def tom_blue : ℕ := 7

/-- The number of additional blue tickets Tom needs to win a Bible -/
def additional_blue_needed : ℕ := 163

/-- The number of blue tickets required to obtain one red ticket -/
def blue_per_red : ℕ := 10

theorem blue_tickets_per_red : 
  yellow_tickets_needed = 10 ∧ 
  red_per_yellow = 10 ∧ 
  tom_yellow = 8 ∧ 
  tom_red = 3 ∧ 
  tom_blue = 7 ∧ 
  additional_blue_needed = 163 → 
  blue_per_red = 10 := by sorry

end NUMINAMATH_CALUDE_blue_tickets_per_red_l2080_208034


namespace NUMINAMATH_CALUDE_parabola_symmetric_intersection_l2080_208008

/-- Represents a parabola of the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: For a parabola with symmetric axis x=1 and one x-axis intersection at (3,0),
    the other x-axis intersection is at (-1,0) --/
theorem parabola_symmetric_intersection
  (p : Parabola)
  (symmetric_axis : ℝ)
  (intersection : Point)
  (h1 : symmetric_axis = 1)
  (h2 : intersection = Point.mk 3 0)
  (h3 : p.a * intersection.x^2 + p.b * intersection.x + p.c = 0)
  : ∃ (other : Point), 
    other = Point.mk (-1) 0 ∧ 
    p.a * other.x^2 + p.b * other.x + p.c = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetric_intersection_l2080_208008


namespace NUMINAMATH_CALUDE_parabola_point_value_l2080_208014

/-- Given a parabola y = x^2 + (a+1)x + a that passes through the point (-1, m),
    prove that m = 0 -/
theorem parabola_point_value (a m : ℝ) : 
  ((-1)^2 + (a+1)*(-1) + a = m) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_value_l2080_208014


namespace NUMINAMATH_CALUDE_joey_studies_five_nights_per_week_l2080_208005

/-- Represents Joey's study schedule and calculates the number of weekday study nights per week -/
def joeys_study_schedule (weekday_hours_per_night : ℕ) (weekend_hours_per_day : ℕ) 
  (total_weeks : ℕ) (total_study_hours : ℕ) : ℕ :=
  let weekend_days := 2 * total_weeks
  let weekend_hours := weekend_hours_per_day * weekend_days
  let weekday_hours := total_study_hours - weekend_hours
  let weekday_nights := weekday_hours / weekday_hours_per_night
  weekday_nights / total_weeks

/-- Theorem stating that Joey studies 5 nights per week on weekdays -/
theorem joey_studies_five_nights_per_week :
  joeys_study_schedule 2 3 6 96 = 5 := by
  sorry

end NUMINAMATH_CALUDE_joey_studies_five_nights_per_week_l2080_208005


namespace NUMINAMATH_CALUDE_barefoot_kids_count_l2080_208036

theorem barefoot_kids_count (total kids_with_socks kids_with_shoes kids_with_both : ℕ) :
  total = 35 →
  kids_with_socks = 18 →
  kids_with_shoes = 15 →
  kids_with_both = 8 →
  total - ((kids_with_socks - kids_with_both) + (kids_with_shoes - kids_with_both) + kids_with_both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_barefoot_kids_count_l2080_208036


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l2080_208057

/-- Represents the composition of teachers in a school -/
structure TeacherComposition where
  total : ℕ
  senior : ℕ
  intermediate : ℕ
  junior : ℕ
  total_sum : total = senior + intermediate + junior

/-- Represents the sample of teachers -/
structure TeacherSample where
  size : ℕ
  senior : ℕ
  intermediate : ℕ
  junior : ℕ
  size_sum : size = senior + intermediate + junior

/-- Theorem stating the correct stratified sampling for the given teacher composition -/
theorem stratified_sampling_theorem 
  (school : TeacherComposition) 
  (sample : TeacherSample) 
  (h1 : school.total = 300) 
  (h2 : school.senior = 90) 
  (h3 : school.intermediate = 150) 
  (h4 : school.junior = 60) 
  (h5 : sample.size = 40) : 
  sample.senior = 12 ∧ sample.intermediate = 20 ∧ sample.junior = 8 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l2080_208057


namespace NUMINAMATH_CALUDE_line_circle_intersection_slope_range_l2080_208030

/-- Given a line intersecting a circle, prove the range of its slope. -/
theorem line_circle_intersection_slope_range :
  ∀ (a : ℝ),
  (∃ (x y : ℝ), x + a * y + 2 = 0 ∧ x^2 + y^2 + 2*x - 2*y + 1 = 0) →
  a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_slope_range_l2080_208030


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2080_208087

/-- The slope of the line perpendicular to 2x - 6y + 1 = 0 -/
def perpendicular_slope : ℝ := -3

/-- The equation of the curve -/
def curve (x : ℝ) : ℝ := x^3 + 3*x^2 - 1

/-- The derivative of the curve -/
def curve_derivative (x : ℝ) : ℝ := 3*x^2 + 6*x

theorem tangent_line_equation :
  ∃ (x₀ y₀ : ℝ),
    curve x₀ = y₀ ∧
    curve_derivative x₀ = perpendicular_slope ∧
    ∀ (x y : ℝ), y - y₀ = perpendicular_slope * (x - x₀) ↔ 3*x + y + 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2080_208087


namespace NUMINAMATH_CALUDE_march_2020_production_theorem_l2080_208027

/-- Calculates the total toilet paper production for March 2020 after a production increase -/
def march_2020_toilet_paper_production (initial_production : ℕ) (increase_factor : ℕ) (days : ℕ) : ℕ :=
  (initial_production + initial_production * increase_factor) * days

/-- Theorem stating the total toilet paper production for March 2020 -/
theorem march_2020_production_theorem :
  march_2020_toilet_paper_production 7000 3 31 = 868000 := by
  sorry

#eval march_2020_toilet_paper_production 7000 3 31

end NUMINAMATH_CALUDE_march_2020_production_theorem_l2080_208027


namespace NUMINAMATH_CALUDE_unique_eventually_one_l2080_208066

def f (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^2 + 1 else 3 * n

def eventually_one (n : ℕ) : Prop :=
  ∃ k : ℕ, (f^[k] n) = 1

theorem unique_eventually_one :
  ∃! n : ℕ, n ∈ Finset.range 200 ∧ eventually_one n :=
sorry

end NUMINAMATH_CALUDE_unique_eventually_one_l2080_208066


namespace NUMINAMATH_CALUDE_night_day_worker_loading_ratio_l2080_208063

theorem night_day_worker_loading_ratio
  (day_workers : ℚ)
  (night_workers : ℚ)
  (total_boxes : ℚ)
  (h1 : night_workers = (4/5) * day_workers)
  (h2 : (5/6) * total_boxes = day_workers * (boxes_per_day_worker : ℚ))
  (h3 : (1/6) * total_boxes = night_workers * (boxes_per_night_worker : ℚ)) :
  boxes_per_night_worker / boxes_per_day_worker = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_night_day_worker_loading_ratio_l2080_208063


namespace NUMINAMATH_CALUDE_root_in_interval_l2080_208065

-- Define the function f(x) = x^3 - x - 1
def f (x : ℝ) : ℝ := x^3 - x - 1

-- Theorem statement
theorem root_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2080_208065


namespace NUMINAMATH_CALUDE_room_width_is_correct_l2080_208050

/-- The width of a room satisfying given conditions -/
def room_width : ℝ :=
  let length : ℝ := 25
  let height : ℝ := 12
  let door_area : ℝ := 6 * 3
  let window_area : ℝ := 4 * 3
  let num_windows : ℕ := 3
  let whitewash_cost_per_sqft : ℝ := 3
  let total_whitewash_cost : ℝ := 2718
  15

/-- Theorem stating that the room width is correct given the conditions -/
theorem room_width_is_correct :
  let length : ℝ := 25
  let height : ℝ := 12
  let door_area : ℝ := 6 * 3
  let window_area : ℝ := 4 * 3
  let num_windows : ℕ := 3
  let whitewash_cost_per_sqft : ℝ := 3
  let total_whitewash_cost : ℝ := 2718
  let width := room_width
  whitewash_cost_per_sqft * (2 * (length * height) + 2 * (width * height) - door_area - num_windows * window_area) = total_whitewash_cost :=
by
  sorry


end NUMINAMATH_CALUDE_room_width_is_correct_l2080_208050


namespace NUMINAMATH_CALUDE_total_students_l2080_208040

theorem total_students (general : ℕ) (biology : ℕ) (chemistry : ℕ) (math : ℕ) (arts : ℕ) 
  (physics : ℕ) (history : ℕ) (literature : ℕ) : 
  general = 30 →
  biology = 2 * general →
  chemistry = general + 10 →
  math = (3 * (general + biology + chemistry)) / 5 →
  arts * 20 / 100 = general →
  physics = general + chemistry - 5 →
  history = (3 * general) / 4 →
  literature = history + 15 →
  general + biology + chemistry + math + arts + physics + history + literature = 484 :=
by sorry

end NUMINAMATH_CALUDE_total_students_l2080_208040


namespace NUMINAMATH_CALUDE_circle_symmetry_l2080_208062

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  (x - 1/2)^2 + (y + 1)^2 = 5/4

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x - y + 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 3/2)^2 = 5/4

-- Theorem statement
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    original_circle x₁ y₁ →
    symmetric_circle x₂ y₂ →
    ∃ (x y : ℝ),
      symmetry_line x y ∧
      (x₁ + x₂ = 2*x) ∧
      (y₁ + y₂ = 2*y) :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2080_208062


namespace NUMINAMATH_CALUDE_hexagon_triangle_area_l2080_208007

/-- The area of a regular hexagon with side length 2, topped by an equilateral triangle with side length 2, is 7√3 square units. -/
theorem hexagon_triangle_area : 
  let hexagon_side : ℝ := 2
  let triangle_side : ℝ := 2
  let hexagon_area : ℝ := (3 * Real.sqrt 3 / 2) * hexagon_side^2
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side^2
  hexagon_area + triangle_area = 7 * Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_hexagon_triangle_area_l2080_208007


namespace NUMINAMATH_CALUDE_y_relationship_l2080_208081

/-- A quadratic function of the form y = -x² + 2x + c -/
def quadratic (c : ℝ) (x : ℝ) : ℝ := -x^2 + 2*x + c

/-- Theorem stating the relationship between y-coordinates of three points on the quadratic function -/
theorem y_relationship (c : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : y₁ = quadratic c (-1))
  (h₂ : y₂ = quadratic c 2)
  (h₃ : y₃ = quadratic c 5) :
  y₂ > y₁ ∧ y₁ > y₃ := by
  sorry


end NUMINAMATH_CALUDE_y_relationship_l2080_208081


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_15_20_30_l2080_208091

/-- The sum of the greatest common factor and the least common multiple of 15, 20, and 30 is 65 -/
theorem gcf_lcm_sum_15_20_30 : 
  (Nat.gcd 15 (Nat.gcd 20 30) + Nat.lcm 15 (Nat.lcm 20 30)) = 65 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_15_20_30_l2080_208091


namespace NUMINAMATH_CALUDE_cycle_sale_theorem_l2080_208048

/-- Calculates the net total amount received after selling three cycles and paying tax -/
def net_total_amount (price1 price2 price3 : ℚ) (profit1 loss2 profit3 tax_rate : ℚ) : ℚ :=
  let sell1 := price1 * (1 + profit1)
  let sell2 := price2 * (1 - loss2)
  let sell3 := price3 * (1 + profit3)
  let total_sell := sell1 + sell2 + sell3
  let tax := total_sell * tax_rate
  total_sell - tax

theorem cycle_sale_theorem :
  net_total_amount 3600 4800 6000 (20/100) (15/100) (10/100) (5/100) = 14250 := by
  sorry

#eval net_total_amount 3600 4800 6000 (20/100) (15/100) (10/100) (5/100)

end NUMINAMATH_CALUDE_cycle_sale_theorem_l2080_208048


namespace NUMINAMATH_CALUDE_trapezoid_to_square_l2080_208006

/-- Represents a trapezoid with bases a and b, and height h -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  area_eq : (a + b) * h / 2 = 5

/-- Represents a square with side length s -/
structure Square where
  s : ℝ
  area_eq : s^2 = 5

/-- Theorem stating that a trapezoid with area 5 can be cut into three parts to form a square -/
theorem trapezoid_to_square (t : Trapezoid) : ∃ (sq : Square), True := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_to_square_l2080_208006


namespace NUMINAMATH_CALUDE_product_remainder_l2080_208071

theorem product_remainder (a b c : ℕ) (ha : a = 2457) (hb : b = 6273) (hc : c = 91409) :
  (a * b * c) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l2080_208071


namespace NUMINAMATH_CALUDE_hcf_problem_l2080_208039

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 2560) (h2 : Nat.lcm a b = 160) :
  Nat.gcd a b = 16 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l2080_208039


namespace NUMINAMATH_CALUDE_correct_equation_l2080_208068

theorem correct_equation (a b : ℝ) : 3 * a^2 * b - 4 * a^2 * b = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l2080_208068


namespace NUMINAMATH_CALUDE_f_greater_than_g_l2080_208074

/-- Given two quadratic functions f and g, prove that f(x) > g(x) for all real x. -/
theorem f_greater_than_g : ∀ x : ℝ, (3 * x^2 - x + 1) > (2 * x^2 + x - 1) := by
  sorry

end NUMINAMATH_CALUDE_f_greater_than_g_l2080_208074


namespace NUMINAMATH_CALUDE_sin_increases_with_angle_sum_of_cosines_positive_l2080_208080

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  -- Ensure the angles form a triangle
  angle_sum : A + B + C = π
  -- Ensure all sides and angles are positive
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0
  positive_angles : A > 0 ∧ B > 0 ∧ C > 0

-- Theorem 1: If angle A is greater than angle B, then sin A is greater than sin B
theorem sin_increases_with_angle (t : Triangle) :
  t.A > t.B → Real.sin t.A > Real.sin t.B := by
  sorry

-- Theorem 2: The sum of cosines of all three angles is always positive
theorem sum_of_cosines_positive (t : Triangle) :
  Real.cos t.A + Real.cos t.B + Real.cos t.C > 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_increases_with_angle_sum_of_cosines_positive_l2080_208080


namespace NUMINAMATH_CALUDE_rectangular_hall_area_l2080_208072

theorem rectangular_hall_area (length width : ℝ) : 
  width = (1 / 2) * length →
  length - width = 17 →
  length * width = 578 := by
sorry

end NUMINAMATH_CALUDE_rectangular_hall_area_l2080_208072


namespace NUMINAMATH_CALUDE_arithmetic_mean_greater_than_harmonic_mean_l2080_208011

theorem arithmetic_mean_greater_than_harmonic_mean 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  (a + b) / 2 > 2 * a * b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_greater_than_harmonic_mean_l2080_208011


namespace NUMINAMATH_CALUDE_sequence_properties_l2080_208061

def S (n : ℕ) : ℝ := -n^2 + 7*n + 1

def a (n : ℕ) : ℝ :=
  if n = 1 then 7
  else -2*n + 8

theorem sequence_properties :
  (∀ n > 4, a n < 0) ∧
  (∀ n : ℕ, n ≠ 0 → S n ≤ S 3 ∧ S n ≤ S 4) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2080_208061


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l2080_208020

-- Define the functions f and g
def f (x : ℝ) := |x + 3| + |x - 1|
def g (m : ℝ) (x : ℝ) := -x^2 + 2*m*x

-- Statement for the solution set of f(x) > 4
theorem solution_set_f (x : ℝ) : f x > 4 ↔ x < -3 ∨ x > 1 := by sorry

-- Statement for the range of m
theorem range_of_m (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, f x₁ ≥ g m x₂) → -2 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l2080_208020


namespace NUMINAMATH_CALUDE_ratio_equality_l2080_208055

theorem ratio_equality (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : a * b ≠ 0) :
  (a / 5) / (b / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2080_208055


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2080_208029

theorem sqrt_equation_solution :
  ∃ (x : ℝ), x = (1225 : ℝ) / 36 ∧ Real.sqrt x + Real.sqrt (x + 4) = 12 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2080_208029


namespace NUMINAMATH_CALUDE_number_of_factors_7200_l2080_208037

theorem number_of_factors_7200 : Nat.card (Nat.divisors 7200) = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_7200_l2080_208037


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2080_208095

theorem exponent_multiplication (x : ℝ) : x^2 * x^4 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2080_208095


namespace NUMINAMATH_CALUDE_gcd_of_rope_lengths_l2080_208024

theorem gcd_of_rope_lengths : Nat.gcd 825 1275 = 75 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_rope_lengths_l2080_208024


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2080_208049

theorem max_value_of_expression (a b : ℝ) (h1 : 300 ≤ a ∧ a ≤ 500) 
  (h2 : 500 ≤ b ∧ b ≤ 1500) : 
  let c : ℝ := 100
  ∀ x ∈ Set.Icc 300 500, ∀ y ∈ Set.Icc 500 1500, 
    (b + c) / (a - c) ≤ 8 ∧ (y + c) / (x - c) ≤ 8 :=
by
  sorry

#check max_value_of_expression

end NUMINAMATH_CALUDE_max_value_of_expression_l2080_208049


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l2080_208090

/-- The slope of the original line -/
def original_slope : ℚ := 3 / 2

/-- The slope of the perpendicular line -/
def perpendicular_slope : ℚ := -2 / 3

/-- The y-intercept of the perpendicular line -/
def y_intercept : ℚ := 5

/-- The x-intercept of the perpendicular line -/
def x_intercept : ℚ := 15 / 2

theorem perpendicular_line_x_intercept :
  let line := fun (x : ℚ) => perpendicular_slope * x + y_intercept
  (∀ x, line x = 0 ↔ x = x_intercept) ∧
  perpendicular_slope * original_slope = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l2080_208090


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2080_208047

-- Define the arithmetic sequence and its properties
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℚ) * d)

-- State the theorem
theorem arithmetic_sequence_properties
  (a₁ d : ℚ) (h_d : d ≠ 0)
  (h_sum : sum_arithmetic_sequence a₁ d 6 = sum_arithmetic_sequence a₁ d 12) :
  (2 * a₁ + 17 * d = 0) ∧
  (sum_arithmetic_sequence a₁ d 18 = 0) ∧
  (d > 0 → arithmetic_sequence a₁ d 6 + arithmetic_sequence a₁ d 14 > 0) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2080_208047


namespace NUMINAMATH_CALUDE_fractional_equation_simplification_l2080_208021

theorem fractional_equation_simplification (x : ℝ) : 
  (x / (3 - x) - 4 = 6 / (x - 3)) ↔ (x - 4 * (3 - x) = -6) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_simplification_l2080_208021


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l2080_208023

/-- A quadratic function satisfying certain conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) ∧
  f (-1) = 0 ∧
  ∀ x, x ≤ f x ∧ f x ≤ (x^2 + 1) / 2

/-- The theorem stating that the quadratic function satisfying the given conditions
    must be f(x) = 1/4(x+1)^2 -/
theorem quadratic_function_unique :
  ∀ f : ℝ → ℝ, QuadraticFunction f → ∀ x, f x = (1/4) * (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l2080_208023


namespace NUMINAMATH_CALUDE_correct_calculation_l2080_208022

theorem correct_calculation : 
  (67 * 17 ≠ 1649) ∧ 
  (150 * 60 ≠ 900) ∧ 
  (250 * 70 = 17500) ∧ 
  (98 * 36 ≠ 3822) :=
by sorry

end NUMINAMATH_CALUDE_correct_calculation_l2080_208022


namespace NUMINAMATH_CALUDE_quadratic_monotone_increasing_iff_l2080_208033

/-- A quadratic function f(x) = x^2 + bx + c is monotonically increasing 
    on the interval [0, +∞) if and only if b ≥ 0 -/
theorem quadratic_monotone_increasing_iff (b c : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ → x₁^2 + b*x₁ + c < x₂^2 + b*x₂ + c) ↔ b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotone_increasing_iff_l2080_208033


namespace NUMINAMATH_CALUDE_train_distance_l2080_208078

theorem train_distance (v1 v2 t : ℝ) (h1 : v1 = 11) (h2 : v2 = 31) (h3 : t = 8) :
  (v2 * t) - (v1 * t) = 160 :=
by sorry

end NUMINAMATH_CALUDE_train_distance_l2080_208078


namespace NUMINAMATH_CALUDE_equation_solution_l2080_208098

theorem equation_solution : ∃! x : ℚ, 2 * (x - 1) = 2 - (5 * x - 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2080_208098


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2080_208056

theorem expand_and_simplify (y : ℝ) : 5 * (6 * y^2 - 3 * y + 2) = 30 * y^2 - 15 * y + 10 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2080_208056


namespace NUMINAMATH_CALUDE_product_scaled_down_l2080_208069

theorem product_scaled_down (h : 213 * 16 = 3408) : 0.016 * 2.13 = 0.03408 := by
  sorry

end NUMINAMATH_CALUDE_product_scaled_down_l2080_208069


namespace NUMINAMATH_CALUDE_min_value_expression_l2080_208083

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2080_208083


namespace NUMINAMATH_CALUDE_perp_foot_curve_equation_l2080_208019

/-- The curve traced by the feet of perpendiculars from the origin to a moving unit segment -/
def PerpFootCurve (x y : ℝ) : Prop :=
  (x^2 + y^2)^3 = x^2 * y^2

/-- A point on the x-axis -/
def PointOnXAxis (p : ℝ × ℝ) : Prop :=
  p.2 = 0

/-- A point on the y-axis -/
def PointOnYAxis (p : ℝ × ℝ) : Prop :=
  p.1 = 0

/-- The distance between two points is 1 -/
def UnitDistance (p q : ℝ × ℝ) : Prop :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = 1

/-- The perpendicular foot from the origin to a line segment -/
def PerpFoot (p : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  (p.1 * (b.1 - a.1) + p.2 * (b.2 - a.2) = 0) ∧
  (∃ t : ℝ, p = (a.1 + t * (b.1 - a.1), a.2 + t * (b.2 - a.2)) ∧ 0 ≤ t ∧ t ≤ 1)

theorem perp_foot_curve_equation (x y : ℝ) :
  (∃ a b : ℝ × ℝ, PointOnXAxis a ∧ PointOnYAxis b ∧ UnitDistance a b ∧
    PerpFoot (x, y) a b) →
  PerpFootCurve x y :=
sorry

end NUMINAMATH_CALUDE_perp_foot_curve_equation_l2080_208019


namespace NUMINAMATH_CALUDE_sin_ten_pi_thirds_l2080_208082

theorem sin_ten_pi_thirds : Real.sin (10 * π / 3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_ten_pi_thirds_l2080_208082


namespace NUMINAMATH_CALUDE_quadrilateral_angle_measure_l2080_208009

theorem quadrilateral_angle_measure (E F G H : ℝ) : 
  E = 3 * F ∧ E = 4 * G ∧ E = 6 * H ∧ E + F + G + H = 360 → E = 540 / 7 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_measure_l2080_208009


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l2080_208058

theorem scientific_notation_equality : 3790000 = 3.79 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l2080_208058


namespace NUMINAMATH_CALUDE_factor_expression_l2080_208031

theorem factor_expression (x : ℝ) : 4 * x^2 - 36 = 4 * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2080_208031


namespace NUMINAMATH_CALUDE_mushroom_collection_problem_l2080_208093

/-- Represents the mushroom distribution pattern for a given girl --/
def mushroom_distribution (total : ℕ) (girl_number : ℕ) : ℚ :=
  (girl_number + 19) + 0.04 * (total - (girl_number + 19))

/-- Theorem stating the solution to the mushroom collection problem --/
theorem mushroom_collection_problem :
  ∃ (n : ℕ) (total : ℕ), 
    (∀ i j, i ≤ n → j ≤ n → mushroom_distribution total i = mushroom_distribution total j) ∧
    n = 5 ∧
    total = 120 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_collection_problem_l2080_208093


namespace NUMINAMATH_CALUDE_budget_circle_graph_l2080_208097

theorem budget_circle_graph (transportation research_development utilities equipment supplies : ℝ)
  (h1 : transportation = 15)
  (h2 : research_development = 9)
  (h3 : utilities = 5)
  (h4 : equipment = 4)
  (h5 : supplies = 2)
  (h6 : transportation + research_development + utilities + equipment + supplies < 100) :
  let salaries := 100 - (transportation + research_development + utilities + equipment + supplies)
  (salaries / 100) * 360 = 234 := by
sorry

end NUMINAMATH_CALUDE_budget_circle_graph_l2080_208097


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l2080_208099

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_inequality :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 1 < 2*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l2080_208099


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2080_208092

theorem polynomial_simplification (x : ℝ) :
  (12 * x^10 + 5 * x^9 + 3 * x^7) + (2 * x^12 - x^10 + 2 * x^9 + 5 * x^6 + 7 * x^4 + 9 * x^2 + 4) =
  2 * x^12 + 11 * x^10 + 7 * x^9 + 3 * x^7 + 5 * x^6 + 7 * x^4 + 9 * x^2 + 4 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2080_208092


namespace NUMINAMATH_CALUDE_frame_diameter_l2080_208017

/-- Given two circular frames X and Y, where X has a diameter of 16 cm and Y covers 0.5625 of X's area, prove that Y's diameter is 12 cm. -/
theorem frame_diameter (dX : ℝ) (coverage : ℝ) (dY : ℝ) : 
  dX = 16 → coverage = 0.5625 → dY = 12 → 
  (π * (dY / 2)^2) = coverage * (π * (dX / 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_frame_diameter_l2080_208017


namespace NUMINAMATH_CALUDE_expression_equals_one_l2080_208045

theorem expression_equals_one (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h_sum : p + q + r = 0) :
  (p^2 * q^2 / ((p^2 - q*r) * (q^2 - p*r))) +
  (p^2 * r^2 / ((p^2 - q*r) * (r^2 - p*q))) +
  (q^2 * r^2 / ((q^2 - p*r) * (r^2 - p*q))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2080_208045


namespace NUMINAMATH_CALUDE_difference_of_squares_l2080_208004

theorem difference_of_squares (m n : ℝ) : m^2 - 4*n^2 = (m + 2*n) * (m - 2*n) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2080_208004


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2080_208043

theorem geometric_sequence_problem :
  ∀ (a b c d : ℝ),
    (∃ (r : ℝ), b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric sequence condition
    (a + d = 20) →                                    -- sum of extreme terms
    (b + c = 34) →                                    -- sum of middle terms
    (a^2 + b^2 + c^2 + d^2 = 1300) →                  -- sum of squares
    ((a = 16 ∧ b = 4 ∧ c = 32 ∧ d = 2) ∨ 
     (a = 4 ∧ b = 16 ∧ c = 2 ∧ d = 32)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2080_208043


namespace NUMINAMATH_CALUDE_gcd_2146_1813_l2080_208067

theorem gcd_2146_1813 : Nat.gcd 2146 1813 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2146_1813_l2080_208067


namespace NUMINAMATH_CALUDE_x_value_l2080_208000

theorem x_value : ∃ x : ℝ, x = 88 * (1 + 0.40) ∧ x = 123.2 :=
by sorry

end NUMINAMATH_CALUDE_x_value_l2080_208000


namespace NUMINAMATH_CALUDE_H_perimeter_is_36_l2080_208070

/-- Calculates the perimeter of a rectangle given its length and width -/
def rectanglePerimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Represents the H-shaped figure -/
structure HShape where
  largeRectLength : ℝ
  largeRectWidth : ℝ
  smallRectLength : ℝ
  smallRectWidth : ℝ

/-- Calculates the perimeter of the H-shaped figure -/
def HPerimeter (h : HShape) : ℝ :=
  2 * rectanglePerimeter h.largeRectLength h.largeRectWidth +
  rectanglePerimeter h.smallRectLength h.smallRectWidth -
  2 * 2 * h.smallRectLength

theorem H_perimeter_is_36 :
  let h : HShape := {
    largeRectLength := 3,
    largeRectWidth := 5,
    smallRectLength := 1,
    smallRectWidth := 3
  }
  HPerimeter h = 36 := by
  sorry

end NUMINAMATH_CALUDE_H_perimeter_is_36_l2080_208070


namespace NUMINAMATH_CALUDE_smallest_integer_fraction_sum_l2080_208076

theorem smallest_integer_fraction_sum (A B C D : ℕ) : 
  A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  (A + B) % (C + D) = 0 →
  (∀ E F G H : ℕ, E ≤ 9 ∧ F ≤ 9 ∧ G ≤ 9 ∧ H ≤ 9 →
    E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H →
    (E + F) % (G + H) = 0 →
    (A + B) / (C + D) ≤ (E + F) / (G + H)) →
  C + D = 17 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_fraction_sum_l2080_208076


namespace NUMINAMATH_CALUDE_triangle_theorem_l2080_208035

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition sin C * sin(A - B) = sin B * sin(C - A) -/
def condition (t : Triangle) : Prop :=
  Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) (h : condition t) :
  2 * t.a^2 = t.b^2 + t.c^2 ∧
  (t.a = 5 ∧ Real.cos t.A = 25/31 → t.a + t.b + t.c = 14) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2080_208035


namespace NUMINAMATH_CALUDE_storage_to_total_ratio_l2080_208041

def total_planks : ℕ := 200
def friends_planks : ℕ := 20
def store_planks : ℕ := 30

def parents_planks : ℕ := total_planks / 2

def storage_planks : ℕ := total_planks - parents_planks - friends_planks - store_planks

theorem storage_to_total_ratio :
  (storage_planks : ℚ) / total_planks = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_storage_to_total_ratio_l2080_208041


namespace NUMINAMATH_CALUDE_f_composition_negative_two_l2080_208052

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -1 / x else 2 * Real.sqrt x

theorem f_composition_negative_two : f (f (-2)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_two_l2080_208052


namespace NUMINAMATH_CALUDE_reconstruct_triangle_from_altitude_feet_l2080_208075

-- Define the basic types
def Point : Type := ℝ × ℝ

-- Define the triangle
structure Triangle :=
  (A B C : Point)

-- Define the orthocenter
def orthocenter (t : Triangle) : Point := sorry

-- Define the feet of altitudes
def altitude_foot (t : Triangle) (v : Point) : Point := sorry

-- Define an acute-angled triangle
def is_acute_angled (t : Triangle) : Prop := sorry

-- Define compass and straightedge constructibility
def constructible (p : Point) : Prop := sorry

-- Main theorem
theorem reconstruct_triangle_from_altitude_feet 
  (t : Triangle) 
  (h_acute : is_acute_angled t) 
  (A1 : Point) 
  (B1 : Point) 
  (C1 : Point) 
  (h_A1 : A1 = altitude_foot t t.A) 
  (h_B1 : B1 = altitude_foot t t.B) 
  (h_C1 : C1 = altitude_foot t t.C) :
  constructible t.A ∧ constructible t.B ∧ constructible t.C :=
sorry

end NUMINAMATH_CALUDE_reconstruct_triangle_from_altitude_feet_l2080_208075


namespace NUMINAMATH_CALUDE_school_population_l2080_208077

/-- The total number of students in a school -/
def total_students : ℕ := sorry

/-- The number of boys in the school -/
def boys : ℕ := 75

/-- The number of girls in the school -/
def girls : ℕ := sorry

/-- Theorem stating the total number of students in the school -/
theorem school_population :
  (total_students = boys + girls) ∧ 
  (girls = (75 : ℚ) / 100 * total_students) →
  total_students = 300 := by sorry

end NUMINAMATH_CALUDE_school_population_l2080_208077


namespace NUMINAMATH_CALUDE_time_spent_playing_games_l2080_208053

/-- Calculates the time spent playing games during a flight -/
theorem time_spent_playing_games 
  (total_flight_time : ℕ) 
  (reading_time : ℕ) 
  (movie_time : ℕ) 
  (dinner_time : ℕ) 
  (radio_time : ℕ) 
  (nap_time : ℕ) : 
  total_flight_time = 11 * 60 + 20 → 
  reading_time = 2 * 60 → 
  movie_time = 4 * 60 → 
  dinner_time = 30 → 
  radio_time = 40 → 
  nap_time = 3 * 60 → 
  total_flight_time - (reading_time + movie_time + dinner_time + radio_time + nap_time) = 70 := by
sorry

end NUMINAMATH_CALUDE_time_spent_playing_games_l2080_208053


namespace NUMINAMATH_CALUDE_clown_balloon_count_l2080_208038

/-- The number of balloons a clown has after a series of events -/
def final_balloon_count (initial : ℕ) (additional : ℕ) (given_away : ℕ) (popped : ℕ) : ℕ :=
  initial + additional - given_away - popped

/-- Theorem stating the final number of balloons the clown has -/
theorem clown_balloon_count :
  final_balloon_count 47 13 20 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloon_count_l2080_208038


namespace NUMINAMATH_CALUDE_sum_remainder_mod_nine_l2080_208016

theorem sum_remainder_mod_nine : 
  (34125 + 34126 + 34127 + 34128 + 34129 + 34130 + 34131) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_nine_l2080_208016


namespace NUMINAMATH_CALUDE_petya_larger_than_vasya_l2080_208059

theorem petya_larger_than_vasya : 2^25 > 4^12 := by
  sorry

end NUMINAMATH_CALUDE_petya_larger_than_vasya_l2080_208059


namespace NUMINAMATH_CALUDE_find_n_l2080_208002

theorem find_n : ∃ n : ℤ, n + (n + 1) + (n + 2) + 3 = 15 ∧ n = 3 := by sorry

end NUMINAMATH_CALUDE_find_n_l2080_208002


namespace NUMINAMATH_CALUDE_west_movement_negative_l2080_208003

-- Define a type for direction
inductive Direction
| East
| West

-- Define a function to represent movement
def movement (d : Direction) (distance : ℝ) : ℝ :=
  match d with
  | Direction.East => distance
  | Direction.West => -distance

-- Theorem statement
theorem west_movement_negative (distance : ℝ) :
  movement Direction.East distance = distance →
  movement Direction.West distance = -distance :=
by
  sorry

end NUMINAMATH_CALUDE_west_movement_negative_l2080_208003


namespace NUMINAMATH_CALUDE_min_bound_of_f_min_value_of_expression_l2080_208085

/-- The function f(x) = |x+1| + |x-1| -/
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

/-- The minimum value M such that f(x) ≤ M for all x ∈ ℝ is 2 -/
theorem min_bound_of_f : ∃ M : ℝ, (∀ x, f x ≤ M) ∧ (∀ N, (∀ x, f x ≤ N) → M ≤ N) ∧ M = 2 := by sorry

/-- Given positive a, b satisfying 3a + b = 2, the minimum value of 1/(2a) + 1/(a+b) is 2 -/
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3 * a + b = 2) :
  ∃ m : ℝ, (1 / (2 * a) + 1 / (a + b) ≥ m) ∧
           (∀ x y, x > 0 → y > 0 → 3 * x + y = 2 → 1 / (2 * x) + 1 / (x + y) ≥ m) ∧
           m = 2 := by sorry

end NUMINAMATH_CALUDE_min_bound_of_f_min_value_of_expression_l2080_208085
