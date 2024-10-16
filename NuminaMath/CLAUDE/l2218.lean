import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2218_221833

/-- Given a hyperbola with equation x²/a² - y²/16 = 1 where a > 0,
    if one of its asymptotes has equation 2x - y = 0, then a = 2 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) : 
  (∃ x y : ℝ, x^2 / a^2 - y^2 / 16 = 1 ∧ 2*x - y = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2218_221833


namespace NUMINAMATH_CALUDE_custom_mult_factorial_difference_l2218_221823

-- Define the custom multiplication operation
def custom_mult (a b : ℕ) : ℕ := a * b + a + b

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to calculate the chained custom multiplication
def chained_custom_mult (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => custom_mult (chained_custom_mult n) (n + 1)

theorem custom_mult_factorial_difference :
  factorial 10 - chained_custom_mult 9 = 1 := by
  sorry


end NUMINAMATH_CALUDE_custom_mult_factorial_difference_l2218_221823


namespace NUMINAMATH_CALUDE_fraction_zero_implies_a_equals_one_l2218_221830

theorem fraction_zero_implies_a_equals_one (a : ℝ) : 
  (|a| - 1) / (a + 1) = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_a_equals_one_l2218_221830


namespace NUMINAMATH_CALUDE_complex_imaginary_x_value_l2218_221863

/-- A complex number z is imaginary if its real part is zero -/
def IsImaginary (z : ℂ) : Prop := z.re = 0

theorem complex_imaginary_x_value (x : ℝ) :
  let z : ℂ := Complex.mk (x^2 - 1) (x + 1)
  IsImaginary z → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_imaginary_x_value_l2218_221863


namespace NUMINAMATH_CALUDE_third_visit_next_month_l2218_221825

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a schedule of pool visits -/
structure PoolSchedule :=
  (visit_days : List DayOfWeek)

/-- Represents a month's pool visits -/
structure MonthVisits :=
  (count : Nat)

/-- Function to calculate the date of the nth visit in the next month -/
def nextMonthVisitDate (schedule : PoolSchedule) (current_month : MonthVisits) (n : Nat) : Nat :=
  sorry

/-- Theorem statement -/
theorem third_visit_next_month 
  (schedule : PoolSchedule)
  (current_month : MonthVisits)
  (h1 : schedule.visit_days = [DayOfWeek.Wednesday, DayOfWeek.Friday])
  (h2 : current_month.count = 10) :
  nextMonthVisitDate schedule current_month 3 = 12 :=
sorry

end NUMINAMATH_CALUDE_third_visit_next_month_l2218_221825


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l2218_221871

theorem quadratic_solution_difference (x : ℝ) : 
  x^2 - 5*x + 15 = x + 35 → 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1^2 - 5*x1 + 15 = x1 + 35) ∧ (x2^2 - 5*x2 + 15 = x2 + 35) ∧ 
  (max x1 x2 - min x1 x2 = 2 * Real.sqrt 29) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l2218_221871


namespace NUMINAMATH_CALUDE_fish_thrown_back_l2218_221870

theorem fish_thrown_back (morning_catch : ℕ) (afternoon_catch : ℕ) (dad_catch : ℕ) (total_catch : ℕ) 
  (h1 : morning_catch = 8)
  (h2 : afternoon_catch = 5)
  (h3 : dad_catch = 13)
  (h4 : total_catch = 23)
  (h5 : total_catch = morning_catch - thrown_back + afternoon_catch + dad_catch) :
  thrown_back = 3 := by
  sorry

end NUMINAMATH_CALUDE_fish_thrown_back_l2218_221870


namespace NUMINAMATH_CALUDE_tangent_line_at_point_l2218_221845

/-- The equation of a curve -/
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

/-- The point on the curve -/
def point : ℝ × ℝ := (1, 2)

/-- The equation of the tangent line -/
def tangent_line (x : ℝ) : ℝ := 3*x - 1

theorem tangent_line_at_point :
  let (x₀, y₀) := point
  (f x₀ = y₀) ∧ 
  (∀ x : ℝ, tangent_line x = f x₀ + (tangent_line x₀ - f x₀) * (x - x₀)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_l2218_221845


namespace NUMINAMATH_CALUDE_sara_payment_l2218_221821

/-- The amount Sara gave to the cashier --/
def amount_given (balloon_cost tablecloth_cost streamer_cost banner_cost confetti_cost change : ℚ) : ℚ :=
  balloon_cost + tablecloth_cost + streamer_cost + banner_cost + confetti_cost + change

/-- Theorem stating the amount Sara gave to the cashier --/
theorem sara_payment :
  amount_given 3.5 18.25 9.1 14.65 7.4 6.38 = 59.28 := by
  sorry

end NUMINAMATH_CALUDE_sara_payment_l2218_221821


namespace NUMINAMATH_CALUDE_largest_four_digit_square_base_7_l2218_221888

/-- The largest integer whose square has exactly 4 digits in base 7 -/
def N : ℕ := 48

/-- Conversion of a natural number to its base 7 representation -/
def to_base_7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- Number of digits in the base 7 representation of a natural number -/
def num_digits_base_7 (n : ℕ) : ℕ :=
  (to_base_7 n).length

/-- Theorem stating that N is the largest integer whose square has exactly 4 digits in base 7 -/
theorem largest_four_digit_square_base_7 :
  (∀ m : ℕ, m > N → num_digits_base_7 (m^2) > 4) ∧
  num_digits_base_7 (N^2) = 4 ∧
  to_base_7 N = [6, 6] :=
sorry

#eval N
#eval to_base_7 N
#eval num_digits_base_7 (N^2)

end NUMINAMATH_CALUDE_largest_four_digit_square_base_7_l2218_221888


namespace NUMINAMATH_CALUDE_expression_simplification_l2218_221801

theorem expression_simplification (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let num := a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)
  let den := a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)
  num / den = a + b + c := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2218_221801


namespace NUMINAMATH_CALUDE_line_passes_through_point_l2218_221890

/-- The line equation passes through the point (3, 1) for all values of m -/
theorem line_passes_through_point :
  ∀ (m : ℝ), (2 * m + 1) * 3 + (m + 1) * 1 - 7 * m - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l2218_221890


namespace NUMINAMATH_CALUDE_parking_lot_spaces_l2218_221854

theorem parking_lot_spaces (total_spaces : ℕ) (full_ratio compact_ratio : ℕ) 
  (h1 : total_spaces = 450)
  (h2 : full_ratio = 11)
  (h3 : compact_ratio = 4) :
  (total_spaces * full_ratio) / (full_ratio + compact_ratio) = 330 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_spaces_l2218_221854


namespace NUMINAMATH_CALUDE_set_operations_l2218_221816

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 2}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 4}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x | 2 ≤ x ∧ x ≤ 4}) ∧
  (A ∪ B = {x | x > 1}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2218_221816


namespace NUMINAMATH_CALUDE_quadratic_equation_root_zero_l2218_221815

theorem quadratic_equation_root_zero (a : ℝ) :
  (∀ x, x^2 + x + a^2 - 1 = 0 → x = 0 ∨ x ≠ 0) →
  (∃ x, x^2 + x + a^2 - 1 = 0 ∧ x = 0) →
  a = 1 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_zero_l2218_221815


namespace NUMINAMATH_CALUDE_sum_value_theorem_l2218_221851

theorem sum_value_theorem (a b c : ℚ) (h1 : |a + 1| + (b - 2)^2 = 0) (h2 : |c| = 3) :
  a + b + 2*c = 7 ∨ a + b + 2*c = -5 := by
  sorry

end NUMINAMATH_CALUDE_sum_value_theorem_l2218_221851


namespace NUMINAMATH_CALUDE_frames_cost_l2218_221877

theorem frames_cost (lens_cost insurance_coverage coupon total_cost : ℚ) : 
  lens_cost = 500 →
  insurance_coverage = 0.8 →
  coupon = 50 →
  total_cost = 250 →
  ∃ frame_cost : ℚ, 
    frame_cost - coupon + lens_cost * (1 - insurance_coverage) = total_cost ∧ 
    frame_cost = 200 :=
by sorry

end NUMINAMATH_CALUDE_frames_cost_l2218_221877


namespace NUMINAMATH_CALUDE_dots_not_visible_l2218_221806

/-- The number of dice -/
def num_dice : ℕ := 5

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The list of visible numbers on the dice -/
def visible_numbers : List ℕ := [1, 2, 2, 3, 3, 3, 4, 4, 5, 6]

/-- The theorem stating the number of dots not visible -/
theorem dots_not_visible :
  num_dice * die_sum - visible_numbers.sum = 72 := by sorry

end NUMINAMATH_CALUDE_dots_not_visible_l2218_221806


namespace NUMINAMATH_CALUDE_irrational_sqrt_sin_cos_l2218_221883

theorem irrational_sqrt_sin_cos (θ : Real) (h : 0 < θ ∧ θ < π / 2) :
  ¬(∃ (a b c d : ℤ), b ≠ 0 ∧ d ≠ 0 ∧ 
    Real.sqrt (Real.sin θ) = a / b ∧ 
    Real.sqrt (Real.cos θ) = c / d) :=
by sorry

end NUMINAMATH_CALUDE_irrational_sqrt_sin_cos_l2218_221883


namespace NUMINAMATH_CALUDE_four_numbers_property_l2218_221827

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem four_numbers_property (a b c d : ℕ) : 
  a = 1 → b = 2 → c = 3 → d = 5 →
  is_prime (a * b + c * d) ∧ 
  is_prime (a * c + b * d) ∧ 
  is_prime (a * d + b * c) := by
sorry

end NUMINAMATH_CALUDE_four_numbers_property_l2218_221827


namespace NUMINAMATH_CALUDE_triangle_with_smallest_semi_prime_angle_l2218_221876

def is_semi_prime (n : ℕ) : Prop := ∃ a b : ℕ, a.Prime ∧ b.Prime ∧ n = a * b

theorem triangle_with_smallest_semi_prime_angle :
  ∀ p q : ℕ,
  is_semi_prime p →
  is_semi_prime q →
  p = 2 * q →
  q = 4 →
  ∃ x : ℕ,
  p + q + x = 180 ∧
  x = 168 := by
sorry

end NUMINAMATH_CALUDE_triangle_with_smallest_semi_prime_angle_l2218_221876


namespace NUMINAMATH_CALUDE_tax_revenue_change_l2218_221892

theorem tax_revenue_change 
  (original_tax : ℝ) 
  (original_consumption : ℝ) 
  (tax_reduction_rate : ℝ) 
  (consumption_increase_rate : ℝ) 
  (h1 : tax_reduction_rate = 0.16) 
  (h2 : consumption_increase_rate = 0.15) : 
  let new_tax := original_tax * (1 - tax_reduction_rate)
  let new_consumption := original_consumption * (1 + consumption_increase_rate)
  let original_revenue := original_tax * original_consumption
  let new_revenue := new_tax * new_consumption
  (new_revenue - original_revenue) / original_revenue = -0.034 :=
by sorry

end NUMINAMATH_CALUDE_tax_revenue_change_l2218_221892


namespace NUMINAMATH_CALUDE_original_sweets_per_child_l2218_221814

/-- Proves that the original number of sweets per child is 15 --/
theorem original_sweets_per_child (total_children : ℕ) (absent_children : ℕ) (extra_sweets : ℕ) : 
  total_children = 112 → 
  absent_children = 32 → 
  extra_sweets = 6 → 
  ∃ (total_sweets : ℕ), 
    total_sweets = total_children * 15 ∧ 
    total_sweets = (total_children - absent_children) * (15 + extra_sweets) := by
  sorry


end NUMINAMATH_CALUDE_original_sweets_per_child_l2218_221814


namespace NUMINAMATH_CALUDE_parabola_focus_vertex_ratio_l2218_221808

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- The locus of midpoints of line segments AB on a parabola P where ∠AV₁B = 90° -/
def midpoint_locus (p : Parabola) : Parabola := sorry

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_focus_vertex_ratio :
  let p := Parabola.mk 4 0 0
  let q := midpoint_locus p
  let v1 := vertex p
  let v2 := vertex q
  let f1 := focus p
  let f2 := focus q
  distance f1 f2 / distance v1 v2 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_vertex_ratio_l2218_221808


namespace NUMINAMATH_CALUDE_iceland_visitors_l2218_221891

theorem iceland_visitors (total : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) :
  total = 100 →
  norway = 43 →
  both = 61 →
  neither = 63 →
  total - neither = norway + (total - neither - norway + both) - both :=
by
  sorry

#eval 100 - 63 - 43 + 61  -- Should evaluate to 55

end NUMINAMATH_CALUDE_iceland_visitors_l2218_221891


namespace NUMINAMATH_CALUDE_simplify_cube_root_l2218_221853

theorem simplify_cube_root : 
  (20^3 + 30^3 + 40^3 + 60^3 : ℝ)^(1/3) = 10 * 315^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_root_l2218_221853


namespace NUMINAMATH_CALUDE_logarithm_sum_equality_l2218_221842

theorem logarithm_sum_equality : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_equality_l2218_221842


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2218_221831

/-- Given a rectangular field with perimeter 120 meters and length three times the width,
    prove that its area is 675 square meters. -/
theorem rectangular_field_area (l w : ℝ) : 
  (2 * l + 2 * w = 120) → 
  (l = 3 * w) → 
  (l * w = 675) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2218_221831


namespace NUMINAMATH_CALUDE_sheila_mwf_hours_l2218_221800

/-- Represents Sheila's work schedule and earnings -/
structure SheilaWork where
  mwf_hours : ℝ  -- Hours worked on Monday, Wednesday, and Friday
  tt_hours : ℝ   -- Hours worked on Tuesday and Thursday
  hourly_rate : ℝ -- Hourly rate in dollars
  weekly_earnings : ℝ -- Total weekly earnings in dollars

/-- Theorem stating Sheila's work hours on Monday, Wednesday, and Friday -/
theorem sheila_mwf_hours (s : SheilaWork) 
  (h1 : s.tt_hours = 6)
  (h2 : s.hourly_rate = 11)
  (h3 : s.weekly_earnings = 396)
  (h4 : s.weekly_earnings = s.hourly_rate * (3 * s.mwf_hours + 2 * s.tt_hours)) :
  s.mwf_hours = 8 := by
  sorry

#check sheila_mwf_hours

end NUMINAMATH_CALUDE_sheila_mwf_hours_l2218_221800


namespace NUMINAMATH_CALUDE_f_2x_eq_3_l2218_221897

/-- A function that is constant 3 for all real inputs -/
def f : ℝ → ℝ := fun x ↦ 3

/-- Theorem: f(2x) = 3 given that f(x) = 3 for all real x -/
theorem f_2x_eq_3 : ∀ x : ℝ, f (2 * x) = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_2x_eq_3_l2218_221897


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l2218_221896

theorem square_perimeter_ratio (d1 d11 s1 s11 P1 P11 : ℝ) : 
  d1 > 0 → 
  d11 = 11 * d1 → 
  d1 = s1 * Real.sqrt 2 → 
  d11 = s11 * Real.sqrt 2 → 
  P1 = 4 * s1 → 
  P11 = 4 * s11 → 
  P11 / P1 = 11 := by
sorry


end NUMINAMATH_CALUDE_square_perimeter_ratio_l2218_221896


namespace NUMINAMATH_CALUDE_max_crayfish_revenue_l2218_221829

/-- The revenue function for selling crayfish -/
def revenue (x : ℝ) : ℝ := (32 - x) * (x - 4.5)

/-- The theorem stating the maximum revenue and number of crayfish sold -/
theorem max_crayfish_revenue :
  ∃ (x : ℕ), x ≤ 32 ∧ 
  revenue (32 - x : ℝ) = 189 ∧
  ∀ (y : ℕ), y ≤ 32 → revenue (32 - y : ℝ) ≤ 189 ∧
  x = 14 :=
sorry

end NUMINAMATH_CALUDE_max_crayfish_revenue_l2218_221829


namespace NUMINAMATH_CALUDE_elizabeth_pencil_cost_l2218_221835

/-- The cost of a pencil given Elizabeth's shopping constraints -/
def pencil_cost (total_money : ℚ) (pen_cost : ℚ) (num_pens : ℕ) (num_pencils : ℕ) : ℚ :=
  (total_money - pen_cost * num_pens) / num_pencils

theorem elizabeth_pencil_cost :
  pencil_cost 20 2 6 5 = 1.60 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_pencil_cost_l2218_221835


namespace NUMINAMATH_CALUDE_arccos_cos_eleven_l2218_221838

theorem arccos_cos_eleven : 
  Real.arccos (Real.cos 11) = 11 - 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_arccos_cos_eleven_l2218_221838


namespace NUMINAMATH_CALUDE_coffee_package_size_l2218_221805

/-- Proves that the size of the larger coffee package is 10 ounces given the conditions -/
theorem coffee_package_size (total_coffee : ℕ) (larger_package_count : ℕ) 
  (small_package_size : ℕ) (small_package_count : ℕ) (larger_package_size : ℕ) :
  total_coffee = 115 ∧ 
  larger_package_count = 7 ∧
  small_package_size = 5 ∧
  small_package_count = larger_package_count + 2 ∧
  total_coffee = larger_package_count * larger_package_size + small_package_count * small_package_size →
  larger_package_size = 10 := by
  sorry

#check coffee_package_size

end NUMINAMATH_CALUDE_coffee_package_size_l2218_221805


namespace NUMINAMATH_CALUDE_sin_2theta_value_l2218_221881

theorem sin_2theta_value (θ : ℝ) (h : ∑' n, (Real.sin θ ^ 2) ^ n = 3) : 
  Real.sin (2 * θ) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l2218_221881


namespace NUMINAMATH_CALUDE_range_of_m_l2218_221810

-- Define the equations
def P (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- State the theorem
theorem range_of_m :
  (∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m)) →
  (∀ m : ℝ, m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2218_221810


namespace NUMINAMATH_CALUDE_expression_evaluation_l2218_221839

theorem expression_evaluation : 2^3 + 4 * 5 - Real.sqrt 9 + (3^2 * 2) / 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2218_221839


namespace NUMINAMATH_CALUDE_ellipse_properties_l2218_221826

-- Define the ellipse E
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection of a line with the ellipse
def line_ellipse_intersection (k : ℝ) (a b : ℝ) (x : ℝ) : Prop :=
  (3 + 4*k^2) * x^2 - 8*k^2 * x + (4*k^2 - 12) = 0

-- Main theorem
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ (x y : ℝ), ellipse a b x y ∧ parabola x y) →
  (∃ (x1 y1 x2 y2 : ℝ), ellipse a b x1 y1 ∧ ellipse a b x2 y2 ∧ 
    parabola x1 y1 ∧ parabola x2 y2 ∧ 
    ((x2 - x1)^2 + (y2 - y1)^2)^(1/2 : ℝ) = 3) →
  (a = 2 ∧ b = (3 : ℝ)^(1/2 : ℝ)) ∧
  (∀ (k : ℝ), k ≠ 0 →
    (∃ (x1 x2 x3 x4 : ℝ), 
      line_ellipse_intersection k a b x1 ∧
      line_ellipse_intersection k a b x2 ∧
      line_ellipse_intersection (-1/k) a b x3 ∧
      line_ellipse_intersection (-1/k) a b x4 ∧
      (∃ (r : ℝ), 
        (x1 - 1)^2 + (k*(x1 - 1))^2 = r^2 ∧
        (x2 - 1)^2 + (k*(x2 - 1))^2 = r^2 ∧
        (x3 - 1)^2 + (-1/k*(x3 - 1))^2 = r^2 ∧
        (x4 - 1)^2 + (-1/k*(x4 - 1))^2 = r^2)) ↔
    (k = 1 ∨ k = -1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2218_221826


namespace NUMINAMATH_CALUDE_remainder_divisibility_l2218_221811

theorem remainder_divisibility (x : ℤ) : x % 72 = 19 → x % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l2218_221811


namespace NUMINAMATH_CALUDE_homothetic_cubes_sum_l2218_221818

-- Define a cube in ℝ³
def Cube : Type := ℝ × ℝ × ℝ → Prop

-- Define a homothetic cube
def HomotheticCube (Q : Cube) (a : ℝ) : Cube := sorry

-- Define a sequence of homothetic cubes
def HomotheticCubeSequence (Q : Cube) : Type := ℕ → Cube

-- Define the property of completely filling a cube
def CompletelyFills (Q : Cube) (seq : HomotheticCubeSequence Q) : Prop := sorry

-- Define the coefficients of homothety for a sequence
def CoefficientsOfHomothety (Q : Cube) (seq : HomotheticCubeSequence Q) : ℕ → ℝ := sorry

-- The main theorem
theorem homothetic_cubes_sum (Q : Cube) (seq : HomotheticCubeSequence Q) :
  (∀ n, CoefficientsOfHomothety Q seq n < 1) →
  CompletelyFills Q seq →
  ∑' n, CoefficientsOfHomothety Q seq n ≥ 4 := by sorry

end NUMINAMATH_CALUDE_homothetic_cubes_sum_l2218_221818


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2218_221861

theorem abs_neg_three_eq_three : |(-3 : ℝ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2218_221861


namespace NUMINAMATH_CALUDE_max_profit_at_70_l2218_221860

-- Define the linear function for weekly sales quantity
def sales_quantity (x : ℝ) : ℝ := -2 * x + 200

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - 40) * (sales_quantity x)

-- Theorem stating the maximum profit and the price at which it occurs
theorem max_profit_at_70 :
  ∃ (max_profit : ℝ), max_profit = 1800 ∧
  ∀ (x : ℝ), profit x ≤ max_profit ∧
  profit 70 = max_profit :=
sorry

#check max_profit_at_70

end NUMINAMATH_CALUDE_max_profit_at_70_l2218_221860


namespace NUMINAMATH_CALUDE_fraction_power_zero_l2218_221868

theorem fraction_power_zero :
  let a : ℤ := 756321948
  let b : ℤ := -3958672103
  (a / b : ℚ) ^ (0 : ℤ) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_power_zero_l2218_221868


namespace NUMINAMATH_CALUDE_problem_statement_l2218_221875

theorem problem_statement : 
  (¬∃ x₀ : ℝ, x₀^2 - x₀ + 1 < 0) ∧ 
  (∃ a b : ℝ, a^2 < b^2 ∧ a ≥ b) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2218_221875


namespace NUMINAMATH_CALUDE_quadrilateral_front_view_solids_l2218_221848

-- Define the possible solids
inductive Solid
| Cone
| Cylinder
| TriangularPyramid
| RectangularPrism

-- Define a property for having a quadrilateral front view
def has_quadrilateral_front_view (s : Solid) : Prop :=
  match s with
  | Solid.Cylinder => True
  | Solid.RectangularPrism => True
  | _ => False

-- Theorem statement
theorem quadrilateral_front_view_solids :
  ∀ s : Solid, has_quadrilateral_front_view s ↔ (s = Solid.Cylinder ∨ s = Solid.RectangularPrism) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_front_view_solids_l2218_221848


namespace NUMINAMATH_CALUDE_line_through_origin_and_intersection_l2218_221850

-- Define the two lines
def line1 (x y : ℝ) : Prop := 2*x + 3*y + 8 = 0
def line2 (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ := (x, y) where
  x := -1
  y := -2

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y = 0

-- Theorem statement
theorem line_through_origin_and_intersection :
  ∃ (x y : ℝ),
    line1 x y ∧ 
    line2 x y ∧ 
    line_l 0 0 ∧ 
    line_l (intersection_point.1) (intersection_point.2) ∧
    ∀ (a b : ℝ), line_l a b ↔ 2*a - b = 0 :=
sorry

end NUMINAMATH_CALUDE_line_through_origin_and_intersection_l2218_221850


namespace NUMINAMATH_CALUDE_partnership_profit_l2218_221882

/-- The total profit of a business partnership --/
def total_profit (a_investment b_investment : ℤ) (management_fee_percent : ℚ) (a_total_received : ℤ) : ℚ :=
  let total_investment := a_investment + b_investment
  let a_share_percent := a_investment / total_investment
  let remaining_profit_percent := 1 - management_fee_percent
  let a_total_percent := management_fee_percent + (a_share_percent * remaining_profit_percent)
  (a_total_received : ℚ) / a_total_percent

/-- The proposition that the total profit is 9600 given the specified conditions --/
theorem partnership_profit : 
  total_profit 15000 25000 (1/10) 4200 = 9600 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_l2218_221882


namespace NUMINAMATH_CALUDE_max_sum_of_digits_24hour_l2218_221887

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def sumOfDigitsTime24 (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits in a 24-hour format digital watch display is 24 -/
theorem max_sum_of_digits_24hour : (⨆ t : Time24, sumOfDigitsTime24 t) = 24 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_24hour_l2218_221887


namespace NUMINAMATH_CALUDE_not_right_angled_triangle_l2218_221858

theorem not_right_angled_triangle : ∃ (a b c : ℝ),
  ((a = 30 ∧ b = 60 ∧ c = 90) → a^2 + b^2 ≠ c^2) ∧
  ((a = 3*Real.sqrt 2 ∧ b = 4*Real.sqrt 2 ∧ c = 5*Real.sqrt 2) → a^2 + b^2 = c^2) ∧
  ((a = 1 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 3) → a^2 + b^2 = c^2) ∧
  ((a = 5 ∧ b = 12 ∧ c = 13) → a^2 + b^2 = c^2) :=
by sorry

end NUMINAMATH_CALUDE_not_right_angled_triangle_l2218_221858


namespace NUMINAMATH_CALUDE_fraction_not_zero_l2218_221879

theorem fraction_not_zero (x : ℝ) (h : x ≠ 1) : 1 / (x - 1) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_not_zero_l2218_221879


namespace NUMINAMATH_CALUDE_median_is_90_l2218_221884

/-- Represents the score distribution of students -/
structure ScoreDistribution where
  score_70 : Nat
  score_80 : Nat
  score_90 : Nat
  score_100 : Nat

/-- Calculates the total number of students -/
def total_students (sd : ScoreDistribution) : Nat :=
  sd.score_70 + sd.score_80 + sd.score_90 + sd.score_100

/-- Defines the median score for a given score distribution -/
def median_score (sd : ScoreDistribution) : Nat :=
  if sd.score_70 + sd.score_80 ≥ (total_students sd + 1) / 2 then 80
  else if sd.score_70 + sd.score_80 + sd.score_90 ≥ (total_students sd + 1) / 2 then 90
  else 100

/-- Theorem stating that the median score for the given distribution is 90 -/
theorem median_is_90 (sd : ScoreDistribution) 
  (h1 : sd.score_70 = 1)
  (h2 : sd.score_80 = 6)
  (h3 : sd.score_90 = 5)
  (h4 : sd.score_100 = 3) :
  median_score sd = 90 := by
  sorry

end NUMINAMATH_CALUDE_median_is_90_l2218_221884


namespace NUMINAMATH_CALUDE_cost_of_seven_sandwiches_five_sodas_l2218_221874

def sandwich_cost : ℝ := 4
def soda_cost : ℝ := 3
def discount_threshold : ℕ := 10
def discount_rate : ℝ := 0.1

def total_cost (num_sandwiches num_sodas : ℕ) : ℝ :=
  let total_items := num_sandwiches + num_sodas
  let subtotal := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  if total_items > discount_threshold then
    subtotal * (1 - discount_rate)
  else
    subtotal

theorem cost_of_seven_sandwiches_five_sodas :
  total_cost 7 5 = 38.7 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_seven_sandwiches_five_sodas_l2218_221874


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l2218_221899

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def PurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- For a real number a, (a+i)(1+2i) is purely imaginary if and only if a = 2. -/
theorem purely_imaginary_condition (a : ℝ) : 
  PurelyImaginary ((a : ℂ) + I * (1 + 2*I)) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l2218_221899


namespace NUMINAMATH_CALUDE_sum_of_complex_magnitudes_l2218_221828

theorem sum_of_complex_magnitudes : 
  Complex.abs (3 - 5*I) + Complex.abs (3 + 5*I) + Complex.abs (6 - 8*I) = 2 * Real.sqrt 34 + 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_complex_magnitudes_l2218_221828


namespace NUMINAMATH_CALUDE_state_fair_revenue_l2218_221803

/-- Represents the revenue calculation for a state fair -/
theorem state_fair_revenue
  (ticket_price : ℝ)
  (total_ticket_revenue : ℝ)
  (food_price : ℝ)
  (ride_price : ℝ)
  (souvenir_price : ℝ)
  (game_price : ℝ)
  (h1 : ticket_price = 8)
  (h2 : total_ticket_revenue = 8000)
  (h3 : food_price = 10)
  (h4 : ride_price = 6)
  (h5 : souvenir_price = 18)
  (h6 : game_price = 5) :
  ∃ (total_revenue : ℝ),
    total_revenue = total_ticket_revenue +
      (3/5 * (total_ticket_revenue / ticket_price) * food_price) +
      (1/3 * (total_ticket_revenue / ticket_price) * ride_price) +
      (1/6 * (total_ticket_revenue / ticket_price) * souvenir_price) +
      (1/10 * (total_ticket_revenue / ticket_price) * game_price) ∧
    total_revenue = 19486 := by
  sorry


end NUMINAMATH_CALUDE_state_fair_revenue_l2218_221803


namespace NUMINAMATH_CALUDE_initial_girls_count_l2218_221862

theorem initial_girls_count (total : ℕ) (initial_girls : ℕ) : 
  (initial_girls : ℚ) / total = 3 / 10 →
  ((initial_girls : ℚ) - 3) / total = 1 / 5 →
  initial_girls = 9 :=
sorry

end NUMINAMATH_CALUDE_initial_girls_count_l2218_221862


namespace NUMINAMATH_CALUDE_min_value_expression_l2218_221855

theorem min_value_expression (a b : ℝ) (ha : 0 < a ∧ a < 2) (hb : 0 < b ∧ b < 2) (hab : a * b = 1) :
  (1 / (2 - a)) + (2 / (2 - b)) ≥ 2 + (2 * Real.sqrt 2) / 3 ∧
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ a₀ < 2 ∧ 0 < b₀ ∧ b₀ < 2 ∧ a₀ * b₀ = 1 ∧
    (1 / (2 - a₀)) + (2 / (2 - b₀)) = 2 + (2 * Real.sqrt 2) / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2218_221855


namespace NUMINAMATH_CALUDE_multiple_of_all_positive_integers_l2218_221822

theorem multiple_of_all_positive_integers (n : ℤ) : 
  (∀ m : ℕ+, ∃ k : ℤ, n = k * m) ↔ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_all_positive_integers_l2218_221822


namespace NUMINAMATH_CALUDE_find_coefficient_l2218_221895

/-- Given a polynomial equation and a sum condition, prove the value of a specific coefficient. -/
theorem find_coefficient (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (x - a) * (x + 2)^5 = a₀ + a₁*(x + 1) + a₂*(x + 1)^2 + a₃*(x + 1)^3 + a₄*(x + 1)^4 + a₅*(x + 1)^5 + a₆*(x + 1)^6) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = -96) →
  a₄ = -10 :=
by sorry

end NUMINAMATH_CALUDE_find_coefficient_l2218_221895


namespace NUMINAMATH_CALUDE_mess_expenditure_theorem_l2218_221820

/-- Calculates the original expenditure of a mess given the initial and new conditions --/
def original_expenditure (initial_students : ℕ) (new_students : ℕ) (expense_increase : ℕ) (avg_decrease : ℕ) : ℕ :=
  let total_students : ℕ := initial_students + new_students
  let x : ℕ := (expense_increase + total_students * avg_decrease) / (total_students - initial_students)
  initial_students * x

/-- Theorem stating the original expenditure of the mess --/
theorem mess_expenditure_theorem :
  original_expenditure 35 7 84 1 = 630 := by
  sorry

end NUMINAMATH_CALUDE_mess_expenditure_theorem_l2218_221820


namespace NUMINAMATH_CALUDE_money_distribution_l2218_221844

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (ac_sum : A + C = 200)
  (c_amount : C = 30) :
  B + C = 330 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2218_221844


namespace NUMINAMATH_CALUDE_elsa_remaining_data_l2218_221898

/-- Calculates the remaining data after Elsa's usage -/
def remaining_data (total : ℚ) (youtube : ℚ) (facebook_fraction : ℚ) : ℚ :=
  let after_youtube := total - youtube
  let facebook_usage := facebook_fraction * after_youtube
  after_youtube - facebook_usage

/-- Theorem stating that Elsa's remaining data is 120 MB -/
theorem elsa_remaining_data :
  remaining_data 500 300 (2/5) = 120 := by
  sorry

#eval remaining_data 500 300 (2/5)

end NUMINAMATH_CALUDE_elsa_remaining_data_l2218_221898


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l2218_221819

/-- Two vectors in R² are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (6, y)
  are_parallel a b → y = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l2218_221819


namespace NUMINAMATH_CALUDE_team_selection_count_l2218_221893

def num_boys : ℕ := 7
def num_girls : ℕ := 10
def team_size : ℕ := 5
def min_girls : ℕ := 2

theorem team_selection_count :
  (Finset.sum (Finset.range (team_size - min_girls + 1))
    (λ k => Nat.choose num_girls (min_girls + k) * Nat.choose num_boys (team_size - (min_girls + k)))) = 5817 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_count_l2218_221893


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt343_l2218_221802

theorem rationalize_denominator_sqrt343 : 
  7 / Real.sqrt 343 = Real.sqrt 7 / 7 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt343_l2218_221802


namespace NUMINAMATH_CALUDE_percentage_problem_l2218_221834

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 400) : 1.2 * x = 2400 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2218_221834


namespace NUMINAMATH_CALUDE_cone_volume_l2218_221864

/-- A cone with surface area π and lateral surface that unfolds into a semicircle has volume π/9 -/
theorem cone_volume (r l h : ℝ) : 
  r > 0 → l > 0 → h > 0 →
  l = 2 * r →  -- lateral surface unfolds into a semicircle
  π * r^2 + π * r * l = π →  -- surface area is π
  h^2 + r^2 = l^2 →  -- Pythagorean theorem for cone
  (1/3) * π * r^2 * h = π/9 := by
sorry


end NUMINAMATH_CALUDE_cone_volume_l2218_221864


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_problem_l2218_221843

-- Define the cyclic quadrilateral ABCD
def CyclicQuadrilateral (A B C D : Point) : Prop := sorry

-- Define the angle measure
def AngleMeasure (P Q R : Point) : ℝ := sorry

-- Define a point inside a triangle
def PointInsideTriangle (X A B C : Point) : Prop := sorry

-- Define angle bisector
def AngleBisector (A X B C : Point) : Prop := sorry

theorem cyclic_quadrilateral_angle_problem 
  (A B C D X : Point) 
  (h1 : CyclicQuadrilateral A B C D) 
  (h2 : AngleMeasure A D B = 48)
  (h3 : AngleMeasure B D C = 56)
  (h4 : PointInsideTriangle X A B C)
  (h5 : AngleMeasure B C X = 24)
  (h6 : AngleBisector A X B C) :
  AngleMeasure C B X = 38 := by
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_problem_l2218_221843


namespace NUMINAMATH_CALUDE_final_pet_count_l2218_221885

/-- Represents the number of pets in the pet center -/
structure PetCount where
  dogs : ℕ
  cats : ℕ
  rabbits : ℕ
  birds : ℕ

/-- Calculates the total number of pets -/
def totalPets (pets : PetCount) : ℕ :=
  pets.dogs + pets.cats + pets.rabbits + pets.birds

/-- Initial pet count -/
def initialPets : PetCount :=
  { dogs := 36, cats := 29, rabbits := 15, birds := 10 }

/-- First adoption -/
def firstAdoption (pets : PetCount) : PetCount :=
  { dogs := pets.dogs - 20, cats := pets.cats, rabbits := pets.rabbits - 5, birds := pets.birds }

/-- New pets added -/
def newPetsAdded (pets : PetCount) : PetCount :=
  { dogs := pets.dogs, cats := pets.cats + 12, rabbits := pets.rabbits + 8, birds := pets.birds + 5 }

/-- Second adoption -/
def secondAdoption (pets : PetCount) : PetCount :=
  { dogs := pets.dogs, cats := pets.cats - 10, rabbits := pets.rabbits, birds := pets.birds - 4 }

/-- The main theorem stating the final number of pets -/
theorem final_pet_count :
  totalPets (secondAdoption (newPetsAdded (firstAdoption initialPets))) = 76 := by
  sorry

end NUMINAMATH_CALUDE_final_pet_count_l2218_221885


namespace NUMINAMATH_CALUDE_class_size_l2218_221817

/-- Represents the number of students who borrowed at least 3 books -/
def R : ℕ := sorry

/-- Represents the total number of students in the class -/
def S : ℕ := sorry

/-- The average number of books per student -/
def average_books : ℕ := 2

theorem class_size :
  (0 * 2 + 1 * 12 + 2 * 4 + 3 * R = average_books * S) ∧
  (S = 2 + 12 + 4 + R) →
  S = 34 := by sorry

end NUMINAMATH_CALUDE_class_size_l2218_221817


namespace NUMINAMATH_CALUDE_max_multiplication_table_sum_l2218_221824

theorem max_multiplication_table_sum : 
  ∀ (a b c d e f : ℕ), 
    a ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    b ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    c ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    d ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    e ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    f ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f → 
    b ≠ c → b ≠ d → b ≠ e → b ≠ f → 
    c ≠ d → c ≠ e → c ≠ f → 
    d ≠ e → d ≠ f → 
    e ≠ f → 
    (a * d + a * e + a * f + b * d + b * e + b * f + c * d + c * e + c * f) ≤ 961 :=
by sorry

end NUMINAMATH_CALUDE_max_multiplication_table_sum_l2218_221824


namespace NUMINAMATH_CALUDE_car_speed_when_serviced_l2218_221878

/-- Proves that the speed of a car when serviced is 110 km/h, given the conditions of the problem -/
theorem car_speed_when_serviced 
  (speed_not_serviced : ℝ) 
  (time_serviced : ℝ) 
  (time_not_serviced : ℝ) 
  (h1 : speed_not_serviced = 55)
  (h2 : time_serviced = 3)
  (h3 : time_not_serviced = 6)
  (h4 : speed_not_serviced * time_not_serviced = speed_when_serviced * time_serviced) :
  speed_when_serviced = 110 := by
  sorry

#check car_speed_when_serviced

end NUMINAMATH_CALUDE_car_speed_when_serviced_l2218_221878


namespace NUMINAMATH_CALUDE_arithmetic_mean_square_inequality_and_minimum_t_l2218_221865

theorem arithmetic_mean_square_inequality_and_minimum_t :
  (∀ a b c : ℝ, (((a + b + c) / 3) ^ 2 ≤ (a ^ 2 + b ^ 2 + c ^ 2) / 3) ∧
    (((a + b + c) / 3) ^ 2 = (a ^ 2 + b ^ 2 + c ^ 2) / 3 ↔ a = b ∧ b = c)) ∧
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    Real.sqrt x + Real.sqrt y + Real.sqrt z ≤ Real.sqrt 3 * Real.sqrt (x + y + z)) ∧
  (∀ t : ℝ, (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    Real.sqrt x + Real.sqrt y + Real.sqrt z ≤ t * Real.sqrt (x + y + z)) →
    t ≥ Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_square_inequality_and_minimum_t_l2218_221865


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l2218_221873

theorem unique_solution_exponential_equation (p q : ℝ) :
  (∀ x : ℝ, 2^(p*x + q) = p * 2^x + q) → p = 1 ∧ q = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l2218_221873


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l2218_221849

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo 0 2, ∀ y ∈ Set.Ioo 0 2, x < y → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l2218_221849


namespace NUMINAMATH_CALUDE_homework_time_decrease_l2218_221894

theorem homework_time_decrease (initial_time final_time : ℝ) (x : ℝ) 
  (h_initial : initial_time = 100)
  (h_final : final_time = 70)
  (h_positive : 0 < x ∧ x < 1) :
  initial_time * (1 - x)^2 = final_time := by
  sorry

end NUMINAMATH_CALUDE_homework_time_decrease_l2218_221894


namespace NUMINAMATH_CALUDE_direction_vector_y_component_l2218_221859

/-- Given a line passing through two points, prove that if its direction vector
    has a specific form, then the y-component of the direction vector is 4.5. -/
theorem direction_vector_y_component
  (p1 : ℝ × ℝ)
  (p2 : ℝ × ℝ)
  (h1 : p1 = (1, -1))
  (h2 : p2 = (5, 5))
  (direction_vector : ℝ × ℝ)
  (h3 : direction_vector.1 = 3)
  (h4 : ∃ (t : ℝ), t • (p2 - p1) = direction_vector) :
  direction_vector.2 = 4.5 := by
sorry

end NUMINAMATH_CALUDE_direction_vector_y_component_l2218_221859


namespace NUMINAMATH_CALUDE_min_value_M_l2218_221866

theorem min_value_M : ∃ (M : ℝ), (∀ (x : ℝ), -x^2 + 2*x ≤ M) ∧ (∀ (N : ℝ), (∀ (x : ℝ), -x^2 + 2*x ≤ N) → M ≤ N) := by
  sorry

end NUMINAMATH_CALUDE_min_value_M_l2218_221866


namespace NUMINAMATH_CALUDE_coin_denomination_problem_l2218_221804

theorem coin_denomination_problem (total_coins : ℕ) (twenty_paise_coins : ℕ) (total_value : ℕ) :
  total_coins = 324 →
  twenty_paise_coins = 220 →
  total_value = 7000 →
  (twenty_paise_coins * 20 + (total_coins - twenty_paise_coins) * 25 = total_value) :=
by sorry

end NUMINAMATH_CALUDE_coin_denomination_problem_l2218_221804


namespace NUMINAMATH_CALUDE_fraction_of_fraction_one_sixth_of_three_fourths_l2218_221880

theorem fraction_of_fraction (a b c d : ℚ) (h1 : b ≠ 0) (h2 : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem one_sixth_of_three_fourths :
  (1 / 6) / (3 / 4) = 2 / 9 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_one_sixth_of_three_fourths_l2218_221880


namespace NUMINAMATH_CALUDE_no_prime_between_30_40_congruent_7_mod_9_l2218_221856

theorem no_prime_between_30_40_congruent_7_mod_9 : ¬ ∃ (n : ℕ), Nat.Prime n ∧ 30 < n ∧ n < 40 ∧ n % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_between_30_40_congruent_7_mod_9_l2218_221856


namespace NUMINAMATH_CALUDE_ten_two_zero_one_composite_l2218_221847

theorem ten_two_zero_one_composite (n : ℕ) (h : n > 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 2*n^2 + 1 = a * b :=
sorry

end NUMINAMATH_CALUDE_ten_two_zero_one_composite_l2218_221847


namespace NUMINAMATH_CALUDE_square_diagonal_l2218_221813

theorem square_diagonal (area : ℝ) (h : area = 800) :
  ∃ (diagonal : ℝ), diagonal = 40 ∧ diagonal^2 = 2 * area := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_l2218_221813


namespace NUMINAMATH_CALUDE_walnut_trees_before_planting_l2218_221857

theorem walnut_trees_before_planting 
  (initial : ℕ) 
  (planted : ℕ) 
  (final : ℕ) 
  (h1 : planted = 6) 
  (h2 : final = 10) 
  (h3 : final = initial + planted) : 
  initial = 4 :=
by sorry

end NUMINAMATH_CALUDE_walnut_trees_before_planting_l2218_221857


namespace NUMINAMATH_CALUDE_smallest_integer_quadratic_inequality_l2218_221852

theorem smallest_integer_quadratic_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 13*m + 36 ≤ 0 → n ≤ m) ∧ n^2 - 13*n + 36 ≤ 0 ∧ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_quadratic_inequality_l2218_221852


namespace NUMINAMATH_CALUDE_inequality_proof_l2218_221841

theorem inequality_proof (a b c d : ℝ) (h : a + b + c + d = 8) :
  a / (8 + b - d)^(1/3) + b / (8 + c - a)^(1/3) + c / (8 + d - b)^(1/3) + d / (8 + a - c)^(1/3) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2218_221841


namespace NUMINAMATH_CALUDE_clothing_prices_l2218_221886

-- Define the original prices
def original_sweater_price : ℝ := 43.11
def original_shirt_price : ℝ := original_sweater_price - 7.43
def original_pants_price : ℝ := 2 * original_shirt_price

-- Define the discount rate
def discount_rate : ℝ := 0.15

-- Define the total cost after discount
def total_cost : ℝ := 143.67

-- Theorem statement
theorem clothing_prices :
  (original_shirt_price = 35.68) ∧
  (original_sweater_price = 43.11) ∧
  (original_pants_price = 71.36) ∧
  (original_shirt_price + (1 - discount_rate) * original_sweater_price + original_pants_price = total_cost) := by
  sorry

end NUMINAMATH_CALUDE_clothing_prices_l2218_221886


namespace NUMINAMATH_CALUDE_boat_width_proof_l2218_221869

theorem boat_width_proof (river_width : ℝ) (num_boats : ℕ) (min_space : ℝ) 
  (h1 : river_width = 42)
  (h2 : num_boats = 8)
  (h3 : min_space = 2)
  (h4 : ∃ boat_width : ℝ, river_width = num_boats * boat_width + (num_boats + 1) * min_space) :
  ∃ boat_width : ℝ, boat_width = 3 := by
sorry

end NUMINAMATH_CALUDE_boat_width_proof_l2218_221869


namespace NUMINAMATH_CALUDE_unpartnered_students_count_l2218_221809

/-- Represents the number of students in a class -/
structure ClassCount where
  males : ℕ
  females : ℕ

/-- The number of students unable to partner with the opposite gender -/
def unpartnered_students (classes : List ClassCount) : ℕ :=
  let total_males := classes.map (·.males) |>.sum
  let total_females := classes.map (·.females) |>.sum
  Int.natAbs (total_males - total_females)

/-- The main theorem stating the number of unpartnered students -/
theorem unpartnered_students_count : 
  let classes : List ClassCount := [
    ⟨18, 12⟩,  -- First 6th grade class
    ⟨16, 20⟩,  -- Second 6th grade class
    ⟨13, 19⟩,  -- Third 6th grade class
    ⟨23, 21⟩   -- 7th grade class
  ]
  unpartnered_students classes = 2 := by
  sorry

end NUMINAMATH_CALUDE_unpartnered_students_count_l2218_221809


namespace NUMINAMATH_CALUDE_ranges_of_a_and_b_l2218_221889

theorem ranges_of_a_and_b (a b : ℝ) (h : Real.sqrt (a^2 * b) = -a * Real.sqrt b) :
  b ≥ 0 ∧ 
  (b > 0 → a ≤ 0) ∧
  (b = 0 → ∀ x : ℝ, ∃ a : ℝ, Real.sqrt ((a : ℝ)^2 * 0) = -(a : ℝ) * Real.sqrt 0) :=
by sorry

end NUMINAMATH_CALUDE_ranges_of_a_and_b_l2218_221889


namespace NUMINAMATH_CALUDE_cone_ratio_after_ten_rotations_l2218_221867

/-- Represents a right circular cone -/
structure RightCircularCone where
  r : ℝ  -- base radius
  h : ℝ  -- height

/-- Predicate for a cone that makes 10 complete rotations when rolling on its side -/
def makesTenRotations (cone : RightCircularCone) : Prop :=
  2 * Real.pi * Real.sqrt (cone.r^2 + cone.h^2) = 20 * Real.pi * cone.r

theorem cone_ratio_after_ten_rotations (cone : RightCircularCone) :
  makesTenRotations cone → cone.h / cone.r = 3 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_cone_ratio_after_ten_rotations_l2218_221867


namespace NUMINAMATH_CALUDE_bottle_caps_difference_l2218_221812

/-- Represents the number of bottle caps in various states --/
structure BottleCaps where
  thrown_away : ℕ
  found : ℕ
  final_collection : ℕ

/-- Theorem stating the difference between found and thrown away bottle caps --/
theorem bottle_caps_difference (caps : BottleCaps)
  (h1 : caps.thrown_away = 6)
  (h2 : caps.found = 50)
  (h3 : caps.final_collection = 60) :
  caps.found - caps.thrown_away = 44 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_difference_l2218_221812


namespace NUMINAMATH_CALUDE_payment_difference_l2218_221836

def original_price : ℝ := 40.00
def discount_rate : ℝ := 0.10
def tip_rate : ℝ := 0.15

def discounted_price : ℝ := original_price * (1 - discount_rate)

def john_payment : ℝ := discounted_price + (original_price * tip_rate)
def jane_payment : ℝ := discounted_price + (discounted_price * tip_rate)

theorem payment_difference : john_payment - jane_payment = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_payment_difference_l2218_221836


namespace NUMINAMATH_CALUDE_sequence_characterization_l2218_221846

def isValidSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, 0 ≤ a n) ∧
  (∀ n, a n ≤ a (n + 1)) ∧
  (∀ m n, a (m^2 + n^2) = (a m)^2 + (a n)^2)

theorem sequence_characterization (a : ℕ → ℝ) :
  isValidSequence a →
  ((∀ n, a n = 1/2) ∨ (∀ n, a n = 0) ∨ (∀ n, a n = n)) :=
sorry

end NUMINAMATH_CALUDE_sequence_characterization_l2218_221846


namespace NUMINAMATH_CALUDE_xy_value_l2218_221840

theorem xy_value (x y : ℝ) (h_distinct : x ≠ y) 
  (h_eq : x^2 + 2/x^2 = y^2 + 2/y^2) : 
  x * y = Real.sqrt 2 ∨ x * y = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2218_221840


namespace NUMINAMATH_CALUDE_dakota_bill_is_12190_l2218_221872

/-- Calculates Dakota's total medical bill based on given conditions -/
def dakota_medical_bill (
  days : ℕ)
  (bed_charge : ℝ)
  (specialist_rate : ℝ)
  (specialist_time : ℝ)
  (num_specialists : ℕ)
  (ambulance_charge : ℝ)
  (surgery_duration : ℝ)
  (surgeon_rate : ℝ)
  (assistant_rate : ℝ)
  (therapy_rate : ℝ)
  (therapy_duration : ℝ)
  (med_a_cost : ℝ)
  (med_b_cost : ℝ)
  (med_c_rate : ℝ)
  (med_c_duration : ℝ)
  (pills_per_day : ℕ) : ℝ :=
  let bed_total := days * bed_charge
  let specialist_total := days * specialist_rate * specialist_time * num_specialists
  let surgery_total := surgery_duration * (surgeon_rate + assistant_rate)
  let therapy_total := days * therapy_rate * therapy_duration
  let med_a_total := days * med_a_cost * pills_per_day
  let med_b_total := days * med_b_cost * pills_per_day
  let med_c_total := days * med_c_rate * med_c_duration
  bed_total + specialist_total + ambulance_charge + surgery_total + therapy_total + med_a_total + med_b_total + med_c_total

/-- Theorem stating that Dakota's medical bill is $12,190 -/
theorem dakota_bill_is_12190 :
  dakota_medical_bill 3 900 250 0.25 2 1800 2 1500 800 300 1 20 45 80 2 3 = 12190 := by
  sorry

end NUMINAMATH_CALUDE_dakota_bill_is_12190_l2218_221872


namespace NUMINAMATH_CALUDE_intersection_count_theorem_l2218_221832

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  mk :: -- Add constructor parameters

/-- Represents the number of intersection points between two lines -/
def intersectionCount (l1 l2 : Line3D) : ℕ := sorry

/-- Represents if two lines are skew -/
def areSkew (l1 l2 : Line3D) : Prop := sorry

/-- Represents if two lines are parallel -/
def areParallel (l1 l2 : Line3D) : Prop := sorry

/-- Represents if a line is perpendicular to two other lines -/
def isCommonPerpendicular (l l1 l2 : Line3D) : Prop := sorry

theorem intersection_count_theorem 
  (a b EF l : Line3D) 
  (h1 : isCommonPerpendicular EF a b) 
  (h2 : areSkew a b) 
  (h3 : areParallel l EF) : 
  (intersectionCount l a + intersectionCount l b = 0) ∨ 
  (intersectionCount l a + intersectionCount l b = 1) := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_theorem_l2218_221832


namespace NUMINAMATH_CALUDE_negation_of_universal_quadratic_inequality_l2218_221807

theorem negation_of_universal_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quadratic_inequality_l2218_221807


namespace NUMINAMATH_CALUDE_race_time_calculation_l2218_221837

/-- The time it takes for John to catch up and overtake Steve in a race --/
theorem race_time_calculation (john_speed steve_speed initial_distance final_distance : ℝ) 
  (h1 : john_speed = 4.2)
  (h2 : steve_speed = 3.7)
  (h3 : initial_distance = 14)
  (h4 : final_distance = 2) :
  (initial_distance + final_distance) / (john_speed - steve_speed) = 32 := by
  sorry

end NUMINAMATH_CALUDE_race_time_calculation_l2218_221837
