import Mathlib

namespace NUMINAMATH_CALUDE_range_of_expressions_l3810_381030

theorem range_of_expressions (a b : ℝ) 
  (ha : -6 < a ∧ a < 8) 
  (hb : 2 < b ∧ b < 3) : 
  (-10 < 2*a + b ∧ 2*a + b < 19) ∧ 
  (-9 < a - b ∧ a - b < 6) ∧ 
  (-2 < a / b ∧ a / b < 4) := by
sorry

end NUMINAMATH_CALUDE_range_of_expressions_l3810_381030


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_l3810_381019

theorem parabola_x_intercepts :
  ∃! x : ℝ, ∃ y : ℝ, x = -3 * y^2 + 2 * y + 3 ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_l3810_381019


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l3810_381075

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis (p : Point) :
  p.x = -3 ∧ p.y = 2 → (reflect_x p).x = -3 ∧ (reflect_x p).y = -2 := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l3810_381075


namespace NUMINAMATH_CALUDE_time_difference_steve_jennifer_l3810_381064

/-- Represents the time in minutes for various running distances --/
structure RunningTimes where
  danny_to_steve : ℝ
  jennifer_to_danny : ℝ

/-- Theorem stating the difference in time between Steve and Jennifer reaching their respective halfway points --/
theorem time_difference_steve_jennifer (times : RunningTimes) 
  (h1 : times.danny_to_steve = 35)
  (h2 : times.jennifer_to_danny = 10)
  (h3 : times.jennifer_to_danny * 2 = times.danny_to_steve) : 
  (2 * times.danny_to_steve) / 2 - times.jennifer_to_danny / 2 = 30 := by
  sorry


end NUMINAMATH_CALUDE_time_difference_steve_jennifer_l3810_381064


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l3810_381011

theorem unique_four_digit_number : ∃! n : ℕ, 
  (1000 ≤ n ∧ n < 10000) ∧ 
  (∃ d₁ d₂ : ℕ, d₁ ≠ d₂ ∧ d₁ < 10 ∧ d₂ < 10 ∧
    (∃ x y : ℕ, x < 10 ∧ y < 10 ∧ 10 + (n / 100 % 10) = x * y) ∧
    (10 + (n / 100 % 10) - (n / d₂ % 10) = 1)) ∧
  n = 1014 :=
sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l3810_381011


namespace NUMINAMATH_CALUDE_total_geckos_sold_l3810_381031

def geckos_sold_last_year : ℕ := 86

theorem total_geckos_sold (geckos_sold_before : ℕ) 
  (h : geckos_sold_before = 2 * geckos_sold_last_year) : 
  geckos_sold_last_year + geckos_sold_before = 258 := by
  sorry

end NUMINAMATH_CALUDE_total_geckos_sold_l3810_381031


namespace NUMINAMATH_CALUDE_rental_cost_calculation_l3810_381093

def base_daily_rate : ℚ := 30
def per_mile_rate : ℚ := 0.25
def discount_rate : ℚ := 0.1
def discount_threshold : ℕ := 5
def rental_days : ℕ := 6
def miles_driven : ℕ := 500

def calculate_total_cost : ℚ :=
  let daily_cost := if rental_days > discount_threshold
                    then base_daily_rate * (1 - discount_rate) * rental_days
                    else base_daily_rate * rental_days
  let mileage_cost := per_mile_rate * miles_driven
  daily_cost + mileage_cost

theorem rental_cost_calculation :
  calculate_total_cost = 287 :=
by sorry

end NUMINAMATH_CALUDE_rental_cost_calculation_l3810_381093


namespace NUMINAMATH_CALUDE_two_pizzas_not_enough_l3810_381046

/-- Represents a pizza with its toppings -/
structure Pizza where
  hasTomatoes : Bool
  hasMushrooms : Bool
  hasSausage : Bool

/-- Represents a child's pizza preference -/
structure Preference where
  wantsTomatoes : Option Bool
  wantsMushrooms : Option Bool
  wantsSausage : Option Bool

/-- Checks if a pizza satisfies a child's preference -/
def satisfiesPreference (pizza : Pizza) (pref : Preference) : Bool :=
  (pref.wantsTomatoes.isNone || pref.wantsTomatoes == some pizza.hasTomatoes) &&
  (pref.wantsMushrooms.isNone || pref.wantsMushrooms == some pizza.hasMushrooms) &&
  (pref.wantsSausage.isNone || pref.wantsSausage == some pizza.hasSausage)

def masha : Preference := { wantsTomatoes := some true, wantsMushrooms := none, wantsSausage := some false }
def vanya : Preference := { wantsTomatoes := none, wantsMushrooms := some true, wantsSausage := none }
def dasha : Preference := { wantsTomatoes := some false, wantsMushrooms := none, wantsSausage := none }
def nikita : Preference := { wantsTomatoes := some true, wantsMushrooms := some false, wantsSausage := none }
def igor : Preference := { wantsTomatoes := none, wantsMushrooms := some false, wantsSausage := some true }

theorem two_pizzas_not_enough : 
  ∀ (pizza1 pizza2 : Pizza), 
  ¬(satisfiesPreference pizza1 masha ∨ satisfiesPreference pizza2 masha) ∨
  ¬(satisfiesPreference pizza1 vanya ∨ satisfiesPreference pizza2 vanya) ∨
  ¬(satisfiesPreference pizza1 dasha ∨ satisfiesPreference pizza2 dasha) ∨
  ¬(satisfiesPreference pizza1 nikita ∨ satisfiesPreference pizza2 nikita) ∨
  ¬(satisfiesPreference pizza1 igor ∨ satisfiesPreference pizza2 igor) :=
sorry

end NUMINAMATH_CALUDE_two_pizzas_not_enough_l3810_381046


namespace NUMINAMATH_CALUDE_product_of_ab_is_one_l3810_381088

theorem product_of_ab_is_one (a b : ℝ) 
  (h1 : a + 1/b = 4) 
  (h2 : 1/a + b = 16/15) : 
  (a * b) * (a * b) - 34/15 * (a * b) + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_ab_is_one_l3810_381088


namespace NUMINAMATH_CALUDE_cos_12_cos_18_minus_sin_12_sin_18_l3810_381012

theorem cos_12_cos_18_minus_sin_12_sin_18 :
  Real.cos (12 * π / 180) * Real.cos (18 * π / 180) - 
  Real.sin (12 * π / 180) * Real.sin (18 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_12_cos_18_minus_sin_12_sin_18_l3810_381012


namespace NUMINAMATH_CALUDE_parabola_vertex_below_x_axis_l3810_381055

/-- A parabola with equation y = x^2 + 2x + a has its vertex below the x-axis -/
def vertex_below_x_axis (a : ℝ) : Prop :=
  ∃ (x y : ℝ), y = x^2 + 2*x + a ∧ y < 0 ∧ ∀ (x' : ℝ), x'^2 + 2*x' + a ≥ y

theorem parabola_vertex_below_x_axis (a : ℝ) :
  vertex_below_x_axis a → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_below_x_axis_l3810_381055


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l3810_381043

theorem quadratic_inequality_empty_solution (b : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + 1 > 0) ↔ -2 < b ∧ b < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l3810_381043


namespace NUMINAMATH_CALUDE_division_simplification_l3810_381022

theorem division_simplification (a b : ℝ) (h : a ≠ 0) :
  (-4 * a^2 + 12 * a^3 * b) / (-4 * a^2) = 1 - 3 * a * b :=
by sorry

end NUMINAMATH_CALUDE_division_simplification_l3810_381022


namespace NUMINAMATH_CALUDE_power_equality_l3810_381097

theorem power_equality (q : ℕ) : 81^10 = 3^q → q = 40 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3810_381097


namespace NUMINAMATH_CALUDE_initial_bags_calculation_l3810_381036

/-- Given the total number of cookies, total number of candies, and current number of bags,
    calculate the initial number of bags. -/
def initialBags (totalCookies : ℕ) (totalCandies : ℕ) (currentBags : ℕ) : ℕ :=
  sorry

theorem initial_bags_calculation (totalCookies totalCandies currentBags : ℕ) 
    (h1 : totalCookies = 28)
    (h2 : totalCandies = 86)
    (h3 : currentBags = 2)
    (h4 : totalCookies % currentBags = 0)  -- Ensures equal distribution of cookies
    (h5 : totalCandies % (initialBags totalCookies totalCandies currentBags) = 0)  -- Ensures equal distribution of candies
    (h6 : totalCookies / currentBags = totalCandies / (initialBags totalCookies totalCandies currentBags))  -- Cookies per bag equals candies per bag
    : initialBags totalCookies totalCandies currentBags = 6 :=
  sorry

end NUMINAMATH_CALUDE_initial_bags_calculation_l3810_381036


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3810_381044

/-- The eccentricity of a hyperbola with the given properties is 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let C : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  let asymptote₁ : ℝ → ℝ := λ x => (b / a) * x
  let asymptote₂ : ℝ → ℝ := λ x => -(b / a) * x
  ∃ (G H : ℝ × ℝ) (c : ℝ),
    (∃ x, G.1 = x ∧ G.2 = asymptote₁ x) ∧ 
    (∃ x, H.1 = x ∧ H.2 = asymptote₂ x) ∧
    (G.2 - F₁.2) * (G.1 - F₂.1) = -(G.1 - F₁.1) * (G.2 - F₂.2) ∧
    H = ((G.1 + F₁.1) / 2, (G.2 + F₁.2) / 2) →
    c = 2 * a :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3810_381044


namespace NUMINAMATH_CALUDE_square_area_problem_l3810_381006

theorem square_area_problem (s : ℝ) : 
  (2 / 5 : ℝ) * s * 10 = 140 → s^2 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_square_area_problem_l3810_381006


namespace NUMINAMATH_CALUDE_total_candies_l3810_381065

/-- The number of candies each person has -/
structure Candies where
  adam : ℕ
  james : ℕ
  rubert : ℕ
  lisa : ℕ
  chris : ℕ
  max : ℕ
  emily : ℕ

/-- The conditions of the candy distribution -/
def candy_conditions (c : Candies) : Prop :=
  c.adam = 6 ∧
  c.james = 3 * c.adam ∧
  c.rubert = 4 * c.james ∧
  c.lisa = 2 * c.rubert - 5 ∧
  c.chris = c.lisa / 2 + 7 ∧
  c.max = c.rubert + c.chris + 2 ∧
  c.emily = 3 * c.chris - (c.max - c.lisa)

/-- The theorem stating the total number of candies -/
theorem total_candies (c : Candies) (h : candy_conditions c) : 
  c.adam + c.james + c.rubert + c.lisa + c.chris + c.max + c.emily = 678 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l3810_381065


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_range_l3810_381087

-- Define the ellipse and line
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def line (x y : ℝ) : Prop := x - y = 1

-- Define the theorem
theorem ellipse_line_intersection_range (a b : ℝ) (P Q : ℝ × ℝ) :
  a > 0 → b > 0 → a > b →
  ellipse a b P.1 P.2 →
  ellipse a b Q.1 Q.2 →
  line P.1 P.2 →
  line Q.1 Q.2 →
  (P.1 * Q.1 + P.2 * Q.2 = 0) →
  Real.sqrt 2 / 2 * a ≤ b →
  b ≤ Real.sqrt 6 / 3 * a →
  Real.sqrt 5 / 2 ≤ a ∧ a ≤ Real.sqrt 6 / 2 :=
by sorry

#check ellipse_line_intersection_range

end NUMINAMATH_CALUDE_ellipse_line_intersection_range_l3810_381087


namespace NUMINAMATH_CALUDE_point_in_unit_circle_l3810_381060

theorem point_in_unit_circle (z : ℂ) (h : Complex.abs z ≤ 1) :
  (z.re)^2 + (z.im)^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_point_in_unit_circle_l3810_381060


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_l3810_381056

/-- Given an ellipse with equation 4x^2 + 9y^2 = 144 containing a point P(3, 2),
    the slope of the line containing the chord with P as its midpoint is -2/3. -/
theorem ellipse_chord_slope :
  let ellipse := {(x, y) : ℝ × ℝ | 4 * x^2 + 9 * y^2 = 144}
  let P : ℝ × ℝ := (3, 2)
  P ∈ ellipse →
  ∃ (A B : ℝ × ℝ),
    A ∈ ellipse ∧ B ∈ ellipse ∧
    P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
    (B.2 - A.2) / (B.1 - A.1) = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_l3810_381056


namespace NUMINAMATH_CALUDE_total_slices_today_l3810_381000

def lunch_slices : ℕ := 7
def dinner_slices : ℕ := 5

theorem total_slices_today : lunch_slices + dinner_slices = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_slices_today_l3810_381000


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l3810_381003

theorem smallest_angle_solution (θ : Real) : 
  (θ > 0) → 
  (θ < 360) → 
  (Real.cos (θ * π / 180) = Real.sin (70 * π / 180) + Real.cos (50 * π / 180) - Real.sin (20 * π / 180) - Real.cos (10 * π / 180)) → 
  (∀ φ, 0 < φ ∧ φ < θ → Real.cos (φ * π / 180) ≠ Real.sin (70 * π / 180) + Real.cos (50 * π / 180) - Real.sin (20 * π / 180) - Real.cos (10 * π / 180)) → 
  θ = 50 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l3810_381003


namespace NUMINAMATH_CALUDE_problem_solution_l3810_381039

def p (a : ℝ) : Prop := (1 + a)^2 + (1 - a)^2 < 4

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 1 ≥ 0

theorem problem_solution (a : ℝ) :
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) ↔ a ∈ Set.Icc (-2) (-1) ∪ Set.Icc 1 2 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3810_381039


namespace NUMINAMATH_CALUDE_jared_earnings_proof_l3810_381054

/-- The monthly salary of a diploma holder in dollars -/
def diploma_salary : ℕ := 4000

/-- The ratio of a degree holder's salary to a diploma holder's salary -/
def degree_to_diploma_ratio : ℕ := 3

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- Jared's annual earnings after graduating with a degree -/
def jared_annual_earnings : ℕ := degree_to_diploma_ratio * diploma_salary * months_in_year

theorem jared_earnings_proof :
  jared_annual_earnings = 144000 :=
sorry

end NUMINAMATH_CALUDE_jared_earnings_proof_l3810_381054


namespace NUMINAMATH_CALUDE_second_class_size_l3810_381049

theorem second_class_size (students1 : ℕ) (avg1 : ℕ) (avg2 : ℕ) (avg_total : ℕ) :
  students1 = 12 →
  avg1 = 40 →
  avg2 = 60 →
  avg_total = 54 →
  ∃ students2 : ℕ, 
    students2 = 28 ∧
    (students1 * avg1 + students2 * avg2) = (students1 + students2) * avg_total :=
by sorry


end NUMINAMATH_CALUDE_second_class_size_l3810_381049


namespace NUMINAMATH_CALUDE_actual_journey_equation_hypothetical_journey_equation_distance_AB_l3810_381037

/-- The distance between dock A and dock B in kilometers -/
def distance : ℝ := 270

/-- The initial speed of the steamboat in km/hr -/
noncomputable def initial_speed : ℝ := distance / 22.5

/-- Time equation for the actual journey -/
theorem actual_journey_equation :
  distance / initial_speed + 3.5 = 3 + (distance - 2 * initial_speed) / (0.8 * initial_speed) :=
sorry

/-- Time equation for the hypothetical journey with later stop -/
theorem hypothetical_journey_equation :
  distance / initial_speed + 1.5 = 3 + 180 / initial_speed + (distance - 2 * initial_speed - 180) / (0.8 * initial_speed) :=
sorry

/-- The distance AB is 270 km -/
theorem distance_AB : distance = 270 :=
sorry

end NUMINAMATH_CALUDE_actual_journey_equation_hypothetical_journey_equation_distance_AB_l3810_381037


namespace NUMINAMATH_CALUDE_obtuse_triangle_equilateral_triangle_l3810_381059

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (angle_sum : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Theorem 1: If cos A * cos B * cos C < 0, then the triangle is obtuse
theorem obtuse_triangle (t : Triangle) :
  Real.cos t.A * Real.cos t.B * Real.cos t.C < 0 →
  (t.A > π/2 ∨ t.B > π/2 ∨ t.C > π/2) :=
sorry

-- Theorem 2: If cos(A-C) * cos(B-C) * cos(C-A) = 1, then the triangle is equilateral
theorem equilateral_triangle (t : Triangle) :
  Real.cos (t.A - t.C) * Real.cos (t.B - t.C) * Real.cos (t.C - t.A) = 1 →
  t.A = π/3 ∧ t.B = π/3 ∧ t.C = π/3 :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_equilateral_triangle_l3810_381059


namespace NUMINAMATH_CALUDE_square_root_sum_l3810_381004

theorem square_root_sum (x y : ℝ) : (x + 2)^2 + Real.sqrt (y - 18) = 0 → Real.sqrt (x + y) = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_l3810_381004


namespace NUMINAMATH_CALUDE_tangent_intersection_x_coordinate_l3810_381042

/-- Given two circles with radii 3 and 5, centered at (0, 0) and (12, 0) respectively,
    the x-coordinate of the point where a common tangent line intersects the x-axis is 9/2. -/
theorem tangent_intersection_x_coordinate :
  let circle1_radius : ℝ := 3
  let circle1_center : ℝ × ℝ := (0, 0)
  let circle2_radius : ℝ := 5
  let circle2_center : ℝ × ℝ := (12, 0)
  ∃ x : ℝ, x > 0 ∧ 
    (x / (12 - x) = circle1_radius / circle2_radius) ∧
    x = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_intersection_x_coordinate_l3810_381042


namespace NUMINAMATH_CALUDE_complex_number_problem_l3810_381015

theorem complex_number_problem (z : ℂ) :
  (∃ (a : ℝ), z = Complex.I * a) →
  (∃ (b : ℝ), (z + 2)^2 - Complex.I * 8 = Complex.I * b) →
  z = -2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3810_381015


namespace NUMINAMATH_CALUDE_jace_travel_distance_l3810_381025

/-- Calculates the total distance traveled given a constant speed and two driving periods -/
def total_distance (speed : ℝ) (time1 : ℝ) (time2 : ℝ) : ℝ :=
  speed * (time1 + time2)

/-- Theorem stating that given the specified conditions, the total distance traveled is 780 miles -/
theorem jace_travel_distance :
  let speed : ℝ := 60
  let time1 : ℝ := 4
  let time2 : ℝ := 9
  total_distance speed time1 time2 = 780 := by
  sorry

end NUMINAMATH_CALUDE_jace_travel_distance_l3810_381025


namespace NUMINAMATH_CALUDE_b_plus_3b_squared_positive_l3810_381008

theorem b_plus_3b_squared_positive (b : ℝ) (h1 : -0.5 < b) (h2 : b < 0) : 
  b + 3 * b^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_b_plus_3b_squared_positive_l3810_381008


namespace NUMINAMATH_CALUDE_payment_problem_l3810_381057

/-- The payment problem -/
theorem payment_problem (a_days b_days total_days : ℕ) (total_payment : ℚ) : 
  a_days = 6 →
  b_days = 8 →
  total_days = 3 →
  total_payment = 3680 →
  let a_work_per_day : ℚ := 1 / a_days
  let b_work_per_day : ℚ := 1 / b_days
  let ab_work_in_total_days : ℚ := (a_work_per_day + b_work_per_day) * total_days
  let c_work : ℚ := 1 - ab_work_in_total_days
  let c_payment : ℚ := c_work * total_payment
  c_payment = 460 :=
sorry

end NUMINAMATH_CALUDE_payment_problem_l3810_381057


namespace NUMINAMATH_CALUDE_haley_tree_count_l3810_381070

/-- The number of trees Haley has after growing some, losing some to a typhoon, and growing more. -/
def final_tree_count (initial : ℕ) (lost : ℕ) (new : ℕ) : ℕ :=
  initial - lost + new

/-- Theorem stating that with 9 initial trees, 4 lost, and 5 new, the final count is 10. -/
theorem haley_tree_count : final_tree_count 9 4 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_haley_tree_count_l3810_381070


namespace NUMINAMATH_CALUDE_solution_value_l3810_381052

theorem solution_value (m : ℝ) : 
  (∃ x : ℝ, x = 1 ∧ 2 * x - m = -3) → m = 5 :=
by sorry

end NUMINAMATH_CALUDE_solution_value_l3810_381052


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l3810_381061

/-- A coloring function that satisfies the given conditions -/
def valid_coloring (n : ℕ) (S : Finset ℕ) (f : Finset ℕ → Fin 8) : Prop :=
  (S.card = 3 * n) ∧
  ∀ A B C : Finset ℕ,
    A ⊆ S ∧ B ⊆ S ∧ C ⊆ S →
    A.card = n ∧ B.card = n ∧ C.card = n →
    A ≠ B ∧ A ≠ C ∧ B ≠ C →
    (A ∩ B).card ≤ 1 ∧ (A ∩ C).card ≤ 1 ∧ (B ∩ C).card ≤ 1 →
    f A ≠ f B ∨ f A ≠ f C ∨ f B ≠ f C

/-- There exists a valid coloring for any set S with 3n elements -/
theorem exists_valid_coloring (n : ℕ) :
  ∀ S : Finset ℕ, S.card = 3 * n → ∃ f : Finset ℕ → Fin 8, valid_coloring n S f := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l3810_381061


namespace NUMINAMATH_CALUDE_smallest_d_for_injective_g_l3810_381007

def g (x : ℝ) : ℝ := (x - 3)^2 - 7

theorem smallest_d_for_injective_g :
  ∀ d : ℝ, (∀ x y, x ≥ d → y ≥ d → g x = g y → x = y) ↔ d ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_for_injective_g_l3810_381007


namespace NUMINAMATH_CALUDE_grid_filling_ways_l3810_381040

/-- Represents a 6x6 grid with special cells -/
structure Grid :=
  (size : Nat)
  (specialCells : Nat)
  (valuesPerSpecialCell : Nat)

/-- Calculates the number of ways to fill the grid -/
def numberOfWays (g : Grid) : Nat :=
  (g.valuesPerSpecialCell ^ g.specialCells) ^ 4

/-- Theorem: The number of ways to fill the grid is 16 -/
theorem grid_filling_ways (g : Grid) 
  (h1 : g.size = 6)
  (h2 : g.specialCells = 4)
  (h3 : g.valuesPerSpecialCell = 2) :
  numberOfWays g = 16 := by
  sorry

#eval numberOfWays { size := 6, specialCells := 4, valuesPerSpecialCell := 2 }

end NUMINAMATH_CALUDE_grid_filling_ways_l3810_381040


namespace NUMINAMATH_CALUDE_dividend_divisor_quotient_remainder_l3810_381071

theorem dividend_divisor_quotient_remainder (n : ℕ) : 
  n / 9 = 6 ∧ n % 9 = 4 → n = 58 := by
  sorry

end NUMINAMATH_CALUDE_dividend_divisor_quotient_remainder_l3810_381071


namespace NUMINAMATH_CALUDE_ice_cream_cost_calculation_l3810_381084

/-- Calculates the cost of each ice-cream cup given the order details and total amount paid --/
theorem ice_cream_cost_calculation
  (chapati_count : ℕ)
  (rice_count : ℕ)
  (vegetable_count : ℕ)
  (ice_cream_count : ℕ)
  (chapati_cost : ℕ)
  (rice_cost : ℕ)
  (vegetable_cost : ℕ)
  (total_paid : ℕ)
  (h1 : chapati_count = 16)
  (h2 : rice_count = 5)
  (h3 : vegetable_count = 7)
  (h4 : ice_cream_count = 6)
  (h5 : chapati_cost = 6)
  (h6 : rice_cost = 45)
  (h7 : vegetable_cost = 70)
  (h8 : total_paid = 961) :
  (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) / ice_cream_count = 25 := by
  sorry


end NUMINAMATH_CALUDE_ice_cream_cost_calculation_l3810_381084


namespace NUMINAMATH_CALUDE_school_sample_theorem_l3810_381091

theorem school_sample_theorem (total_students sample_size : ℕ) 
  (h_total : total_students = 1200)
  (h_sample : sample_size = 200)
  (h_stratified : ∃ (boys girls : ℕ), boys + girls = sample_size ∧ boys = girls + 10) :
  ∃ (school_boys : ℕ), 
    school_boys * sample_size = 105 * total_students ∧
    school_boys = 630 := by
sorry

end NUMINAMATH_CALUDE_school_sample_theorem_l3810_381091


namespace NUMINAMATH_CALUDE_problem_solution_l3810_381066

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi/6) + Real.cos (x - Real.pi/3)

noncomputable def g (x : ℝ) : ℝ := 2 * (Real.sin (x/2))^2

theorem problem_solution (θ : ℝ) (k : ℤ) :
  (0 < θ ∧ θ < Real.pi/2) →  -- θ is in the first quadrant
  f θ = 3 * Real.sqrt 3 / 5 →
  g θ = 1/5 ∧
  (∀ x, f x ≥ g x ↔ ∃ k, 2 * k * Real.pi ≤ x ∧ x ≤ 2 * k * Real.pi + 2 * Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3810_381066


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3810_381045

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define what it means for a quadratic radical to be in its simplest form
def isSimplestQuadraticRadical (x : ℝ) : Prop :=
  x > 0 ∧ ¬∃ (a b : ℕ), (b > 1) ∧ (¬isPerfectSquare b) ∧ (x = (a : ℝ) * Real.sqrt b)

-- Theorem statement
theorem simplest_quadratic_radical :
  isSimplestQuadraticRadical (Real.sqrt 7) ∧
  ¬isSimplestQuadraticRadical (Real.sqrt 12) ∧
  ¬isSimplestQuadraticRadical (Real.sqrt (2/3)) ∧
  ¬isSimplestQuadraticRadical (Real.sqrt 0.3) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3810_381045


namespace NUMINAMATH_CALUDE_fish_count_l3810_381021

/-- The number of fish per white duck -/
def fish_per_white_duck : ℕ := 5

/-- The number of fish per black duck -/
def fish_per_black_duck : ℕ := 10

/-- The number of fish per multicolor duck -/
def fish_per_multicolor_duck : ℕ := 12

/-- The number of white ducks -/
def white_ducks : ℕ := 3

/-- The number of black ducks -/
def black_ducks : ℕ := 7

/-- The number of multicolor ducks -/
def multicolor_ducks : ℕ := 6

/-- The total number of fish in the lake -/
def total_fish : ℕ := fish_per_white_duck * white_ducks + 
                      fish_per_black_duck * black_ducks + 
                      fish_per_multicolor_duck * multicolor_ducks

theorem fish_count : total_fish = 157 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l3810_381021


namespace NUMINAMATH_CALUDE_percentage_reduction_l3810_381063

theorem percentage_reduction (initial : ℝ) (increase_percent : ℝ) (final : ℝ) : 
  initial = 1500 →
  increase_percent = 20 →
  final = 1080 →
  let increased := initial * (1 + increase_percent / 100)
  (increased - final) / increased * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_reduction_l3810_381063


namespace NUMINAMATH_CALUDE_lance_reading_plan_l3810_381085

/-- Given a book with a certain number of pages, calculate the number of pages 
    to read on the third day to finish the book, given the pages read on the 
    first two days and constraints for the third day. -/
def pagesOnThirdDay (totalPages : ℕ) (day1Pages : ℕ) (day2Reduction : ℕ) : ℕ :=
  let day2Pages := day1Pages - day2Reduction
  let remainingPages := totalPages - (day1Pages + day2Pages)
  ((remainingPages + 9) / 10) * 10

theorem lance_reading_plan :
  pagesOnThirdDay 100 35 5 = 40 := by sorry

end NUMINAMATH_CALUDE_lance_reading_plan_l3810_381085


namespace NUMINAMATH_CALUDE_evans_class_enrollment_l3810_381074

theorem evans_class_enrollment (q1 q2 both not_taken : ℕ) 
  (h1 : q1 = 19)
  (h2 : q2 = 24)
  (h3 : both = 19)
  (h4 : not_taken = 5) :
  q1 + q2 - both + not_taken = 29 := by
sorry

end NUMINAMATH_CALUDE_evans_class_enrollment_l3810_381074


namespace NUMINAMATH_CALUDE_power_seven_137_mod_nine_l3810_381017

theorem power_seven_137_mod_nine : 7^137 % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_137_mod_nine_l3810_381017


namespace NUMINAMATH_CALUDE_total_people_in_tribes_l3810_381041

theorem total_people_in_tribes (cannoneers : ℕ) (women : ℕ) (men : ℕ) : 
  cannoneers = 63 → 
  women = 2 * cannoneers → 
  men = 2 * women → 
  cannoneers + women + men = 378 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_tribes_l3810_381041


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3810_381018

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

end NUMINAMATH_CALUDE_hyperbola_properties_l3810_381018


namespace NUMINAMATH_CALUDE_parabola_line_tangency_l3810_381069

theorem parabola_line_tangency (m : ℝ) : 
  (∃ x y : ℝ, y = x^2 ∧ x + y = Real.sqrt m ∧ 
   (∀ x' y' : ℝ, y' = x'^2 → x' + y' = Real.sqrt m → (x', y') = (x, y))) → 
  m = 1/16 := by
sorry

end NUMINAMATH_CALUDE_parabola_line_tangency_l3810_381069


namespace NUMINAMATH_CALUDE_red_black_red_probability_l3810_381053

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of red cards in a standard deck -/
def RedCards : ℕ := 26

/-- Number of black cards in a standard deck -/
def BlackCards : ℕ := 26

/-- Probability of drawing a red card, then a black card, then a red card from a standard deck -/
theorem red_black_red_probability :
  (RedCards : ℚ) * BlackCards * (RedCards - 1) / (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2)) = 13 / 102 := by
  sorry

end NUMINAMATH_CALUDE_red_black_red_probability_l3810_381053


namespace NUMINAMATH_CALUDE_protein_percentage_in_mixture_l3810_381089

/-- Calculates the protein percentage in a mixture of soybean meal and cornmeal. -/
theorem protein_percentage_in_mixture 
  (soybean_protein_percent : ℝ)
  (cornmeal_protein_percent : ℝ)
  (total_mixture_weight : ℝ)
  (soybean_weight : ℝ)
  (cornmeal_weight : ℝ)
  (h1 : soybean_protein_percent = 0.14)
  (h2 : cornmeal_protein_percent = 0.07)
  (h3 : total_mixture_weight = 280)
  (h4 : soybean_weight = 240)
  (h5 : cornmeal_weight = 40)
  (h6 : total_mixture_weight = soybean_weight + cornmeal_weight) :
  (soybean_weight * soybean_protein_percent + cornmeal_weight * cornmeal_protein_percent) / total_mixture_weight = 0.13 := by
  sorry


end NUMINAMATH_CALUDE_protein_percentage_in_mixture_l3810_381089


namespace NUMINAMATH_CALUDE_shaniqua_earnings_l3810_381076

/-- Calculates the total earnings for Shaniqua's hair services -/
def total_earnings (haircut_price : ℕ) (style_price : ℕ) (num_haircuts : ℕ) (num_styles : ℕ) : ℕ :=
  haircut_price * num_haircuts + style_price * num_styles

/-- Proves that Shaniqua's total earnings for 8 haircuts and 5 styles are $221 -/
theorem shaniqua_earnings : total_earnings 12 25 8 5 = 221 := by
  sorry

end NUMINAMATH_CALUDE_shaniqua_earnings_l3810_381076


namespace NUMINAMATH_CALUDE_max_volume_right_prism_l3810_381067

theorem max_volume_right_prism (b c : ℝ) (h1 : b + c = 8) (h2 : b > 0) (h3 : c > 0) :
  let volume := fun x => (1/2) * b * x^2
  let x := Real.sqrt (64 - 16*b)
  (∀ y, volume y ≤ volume x) ∧ volume x = 32 := by
  sorry

end NUMINAMATH_CALUDE_max_volume_right_prism_l3810_381067


namespace NUMINAMATH_CALUDE_intersection_empty_iff_k_greater_than_six_l3810_381035

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 7}
def N (k : ℝ) : Set ℝ := {x | k + 1 ≤ x ∧ x ≤ 2*k - 1}

theorem intersection_empty_iff_k_greater_than_six (k : ℝ) : 
  M ∩ N k = ∅ ↔ k > 6 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_k_greater_than_six_l3810_381035


namespace NUMINAMATH_CALUDE_det_eq_ten_l3810_381082

/-- The matrix for which we need to calculate the determinant -/
def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![3*x, 2; 3, 2*x]

/-- The theorem stating the condition for the determinant to be 10 -/
theorem det_eq_ten (x : ℝ) : 
  Matrix.det (A x) = 10 ↔ x = Real.sqrt (8/3) ∨ x = -Real.sqrt (8/3) := by
  sorry

end NUMINAMATH_CALUDE_det_eq_ten_l3810_381082


namespace NUMINAMATH_CALUDE_mean_score_remaining_students_l3810_381098

theorem mean_score_remaining_students 
  (n : ℕ) 
  (h1 : n > 20) 
  (h2 : (15 : ℝ) * 10 = (15 : ℝ) * mean_first_15)
  (h3 : (5 : ℝ) * 16 = (5 : ℝ) * mean_next_5)
  (h4 : ((15 : ℝ) * mean_first_15 + (5 : ℝ) * mean_next_5 + (n - 20 : ℝ) * mean_remaining) / n = 11) :
  mean_remaining = (11 * n - 230) / (n - 20) := by
  sorry

end NUMINAMATH_CALUDE_mean_score_remaining_students_l3810_381098


namespace NUMINAMATH_CALUDE_range_of_m_l3810_381028

theorem range_of_m (p q : Prop) (h1 : p ↔ ∀ x : ℝ, x^2 - 2*x + 1 - m ≥ 0)
  (h2 : q ↔ ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ a^2 = 1 ∧ b^2 = 1 / (m + 2))
  (h3 : (p ∨ q) ∧ ¬(p ∧ q)) :
  m ≤ -2 ∨ m > 0 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l3810_381028


namespace NUMINAMATH_CALUDE_equation_solution_l3810_381072

theorem equation_solution : ∃ f : ℝ, 
  ((10 * 0.3 + 2) / 4 - (3 * 0.3 - 6) / f = (2 * 0.3 + 4) / 3) ∧ 
  (abs (f - 18) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3810_381072


namespace NUMINAMATH_CALUDE_contour_area_ratio_l3810_381090

theorem contour_area_ratio (r₁ r₂ : ℝ) (A₁ A₂ : ℝ) (h₁ : 0 < r₁) (h₂ : r₁ < r₂) (h₃ : 0 < A₁) :
  A₂ / A₁ = (r₂ / r₁)^2 :=
sorry

end NUMINAMATH_CALUDE_contour_area_ratio_l3810_381090


namespace NUMINAMATH_CALUDE_cos_identity_l3810_381016

theorem cos_identity : Real.cos (70 * π / 180) + 8 * Real.cos (20 * π / 180) * Real.cos (40 * π / 180) * Real.cos (80 * π / 180) = 2 * (Real.cos (35 * π / 180))^2 := by
  sorry

end NUMINAMATH_CALUDE_cos_identity_l3810_381016


namespace NUMINAMATH_CALUDE_driver_net_hourly_rate_l3810_381099

/-- Calculates the driver's net hourly rate after deducting gas expenses -/
theorem driver_net_hourly_rate
  (travel_time : ℝ)
  (speed : ℝ)
  (gasoline_efficiency : ℝ)
  (gasoline_cost : ℝ)
  (driver_compensation : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : gasoline_efficiency = 25)
  (h4 : gasoline_cost = 2.5)
  (h5 : driver_compensation = 0.6)
  : (driver_compensation * speed * travel_time - 
     (speed * travel_time / gasoline_efficiency) * gasoline_cost) / travel_time = 25 :=
by sorry

end NUMINAMATH_CALUDE_driver_net_hourly_rate_l3810_381099


namespace NUMINAMATH_CALUDE_sugar_amount_proof_l3810_381077

/-- Recipe proportions and conversion factors -/
def butter_to_flour : ℚ := 5 / 7
def salt_to_flour : ℚ := 3 / 1.5
def sugar_to_flour : ℚ := 2 / 2.5
def butter_multiplier : ℚ := 4
def salt_multiplier : ℚ := 3.5
def sugar_multiplier : ℚ := 3
def butter_used : ℚ := 12
def ounce_to_gram : ℚ := 28.35
def cup_flour_to_gram : ℚ := 125
def tsp_salt_to_gram : ℚ := 5
def tbsp_sugar_to_gram : ℚ := 15

/-- Theorem stating that the amount of sugar needed is 604.8 grams -/
theorem sugar_amount_proof :
  let flour_cups := butter_used / butter_to_flour
  let flour_grams := flour_cups * cup_flour_to_gram
  let sugar_tbsp := (sugar_to_flour * flour_cups * sugar_multiplier)
  sugar_tbsp * tbsp_sugar_to_gram = 604.8 := by
  sorry

end NUMINAMATH_CALUDE_sugar_amount_proof_l3810_381077


namespace NUMINAMATH_CALUDE_amount_left_after_pool_l3810_381009

-- Define the given conditions
def total_earned : ℝ := 30
def cost_per_person : ℝ := 2.5
def number_of_people : ℕ := 10

-- Define the theorem
theorem amount_left_after_pool : 
  total_earned - (cost_per_person * number_of_people) = 5 := by
  sorry

end NUMINAMATH_CALUDE_amount_left_after_pool_l3810_381009


namespace NUMINAMATH_CALUDE_prob_A_and_B_selected_is_three_tenths_l3810_381068

def total_students : ℕ := 5
def students_to_select : ℕ := 3

def probability_A_and_B_selected : ℚ :=
  (total_students - students_to_select + 1 : ℚ) / (total_students.choose students_to_select)

theorem prob_A_and_B_selected_is_three_tenths :
  probability_A_and_B_selected = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_and_B_selected_is_three_tenths_l3810_381068


namespace NUMINAMATH_CALUDE_exactly_one_common_course_l3810_381079

/-- The number of ways two people can choose 2 courses each from 4 courses with exactly one course in common -/
def common_course_choices (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k * Nat.choose n k - Nat.choose n k - Nat.choose n k

theorem exactly_one_common_course :
  common_course_choices 4 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_common_course_l3810_381079


namespace NUMINAMATH_CALUDE_usb_drive_available_space_l3810_381038

theorem usb_drive_available_space (total_capacity : ℝ) (used_percentage : ℝ) 
  (h1 : total_capacity = 16)
  (h2 : used_percentage = 50)
  : total_capacity * (1 - used_percentage / 100) = 8 := by
  sorry

end NUMINAMATH_CALUDE_usb_drive_available_space_l3810_381038


namespace NUMINAMATH_CALUDE_existence_of_polynomial_l3810_381027

/-- The polynomial a(x, y) -/
def a (x y : ℝ) : ℝ := x^2 * y + x * y^2

/-- The polynomial b(x, y) -/
def b (x y : ℝ) : ℝ := x^2 + x * y + y^2

/-- The statement to be proved -/
theorem existence_of_polynomial (n : ℕ) : 
  ∃ (p : ℝ → ℝ → ℝ), ∀ (x y : ℝ), 
    p (a x y) (b x y) = (x + y)^n + (-1)^n * (x^n + y^n) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_polynomial_l3810_381027


namespace NUMINAMATH_CALUDE_quadratic_monotone_increasing_iff_l3810_381014

/-- A quadratic function f(x) = x^2 + bx + c is monotonically increasing 
    on the interval [0, +∞) if and only if b ≥ 0 -/
theorem quadratic_monotone_increasing_iff (b c : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ → x₁^2 + b*x₁ + c < x₂^2 + b*x₂ + c) ↔ b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotone_increasing_iff_l3810_381014


namespace NUMINAMATH_CALUDE_probability_of_one_each_item_l3810_381080

def drawer_items : ℕ := 8

def total_items : ℕ := 4 * drawer_items

def items_removed : ℕ := 4

def total_combinations : ℕ := Nat.choose total_items items_removed

def favorable_outcomes : ℕ := drawer_items^items_removed

theorem probability_of_one_each_item : 
  (favorable_outcomes : ℚ) / total_combinations = 128 / 1125 := by sorry

end NUMINAMATH_CALUDE_probability_of_one_each_item_l3810_381080


namespace NUMINAMATH_CALUDE_congruence_solutions_count_l3810_381078

theorem congruence_solutions_count : 
  (Finset.filter (fun x : ℕ => x < 150 ∧ (x + 17) % 46 = 75 % 46) (Finset.range 150)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solutions_count_l3810_381078


namespace NUMINAMATH_CALUDE_linear_dependence_iff_k_eq_8_l3810_381010

def vector1 : ℝ × ℝ × ℝ := (1, 4, -1)
def vector2 (k : ℝ) : ℝ × ℝ × ℝ := (2, k, 3)

def is_linearly_dependent (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧ 
    c1 • v1 + c2 • v2 = (0, 0, 0)

theorem linear_dependence_iff_k_eq_8 :
  ∀ k : ℝ, is_linearly_dependent vector1 (vector2 k) ↔ k = 8 := by
  sorry

end NUMINAMATH_CALUDE_linear_dependence_iff_k_eq_8_l3810_381010


namespace NUMINAMATH_CALUDE_decimal_8543_to_base7_l3810_381005

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: decimal_to_base7 (n / 7)

def base7_to_decimal (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 7 * acc) 0

theorem decimal_8543_to_base7 :
  decimal_to_base7 8543 = [3, 2, 6, 3, 3] ∧
  base7_to_decimal [3, 2, 6, 3, 3] = 8543 := by
  sorry

end NUMINAMATH_CALUDE_decimal_8543_to_base7_l3810_381005


namespace NUMINAMATH_CALUDE_intersection_range_l3810_381083

open Set Real

theorem intersection_range (m : ℝ) : 
  let A : Set ℝ := {x | |x - 1| + |x + 1| ≤ 3}
  let B : Set ℝ := {x | x^2 - (2*m + 1)*x + m^2 + m < 0}
  (A ∩ B).Nonempty → m > -3/2 ∧ m < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_range_l3810_381083


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3810_381092

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x = 3 * Real.sqrt 2 ∧
  (∀ y : ℝ, y > 0 → ⌊y^2⌋ - ⌊y⌋^2 = 12 → y ≥ x) ∧
  ⌊x^2⌋ - ⌊x⌋^2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3810_381092


namespace NUMINAMATH_CALUDE_real_roots_imply_b_equals_one_l3810_381058

theorem real_roots_imply_b_equals_one (b : ℝ) : 
  (∃ x : ℝ, x^2 - 2*I*x + b = 1) → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_imply_b_equals_one_l3810_381058


namespace NUMINAMATH_CALUDE_rectangular_fence_length_l3810_381024

/-- A rectangular fence with a perimeter of 30 meters and a length that is twice its width has a length of 10 meters. -/
theorem rectangular_fence_length (width : ℝ) (length : ℝ) : 
  width > 0 → 
  length > 0 → 
  length = 2 * width → 
  2 * length + 2 * width = 30 → 
  length = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangular_fence_length_l3810_381024


namespace NUMINAMATH_CALUDE_parabola_chord_constant_sum_l3810_381096

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y = x^2 -/
def parabola (p : Point) : Prop :=
  p.y = p.x^2

/-- Point C on the y-axis -/
def C : Point :=
  ⟨0, 2⟩

/-- Distance squared between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- The theorem to be proved -/
theorem parabola_chord_constant_sum :
  ∀ A B : Point,
  parabola A → parabola B →
  (C.y - A.y) / (C.x - A.x) = (B.y - A.y) / (B.x - A.x) →
  (1 / distanceSquared A C + 1 / distanceSquared B C) = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_chord_constant_sum_l3810_381096


namespace NUMINAMATH_CALUDE_bottles_left_l3810_381023

theorem bottles_left (initial_bottles drunk_bottles : ℕ) :
  initial_bottles = 17 →
  drunk_bottles = 3 →
  initial_bottles - drunk_bottles = 14 :=
by sorry

end NUMINAMATH_CALUDE_bottles_left_l3810_381023


namespace NUMINAMATH_CALUDE_horse_speed_around_square_field_l3810_381095

theorem horse_speed_around_square_field 
  (field_area : ℝ) 
  (time_to_run_around : ℝ) 
  (horse_speed : ℝ) : 
  field_area = 400 ∧ 
  time_to_run_around = 4 → 
  horse_speed = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_horse_speed_around_square_field_l3810_381095


namespace NUMINAMATH_CALUDE_smallest_marble_count_l3810_381081

def is_valid_marble_count (n : ℕ) : Prop :=
  n > 2 ∧ n % 6 = 2 ∧ n % 7 = 2 ∧ n % 8 = 2 ∧ n % 11 = 2

theorem smallest_marble_count :
  ∃ (n : ℕ), is_valid_marble_count n ∧ ∀ (m : ℕ), is_valid_marble_count m → n ≤ m :=
by
  use 3698
  sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l3810_381081


namespace NUMINAMATH_CALUDE_min_value_of_f_l3810_381034

open Real

noncomputable def f (x : ℝ) := 2 * x - log x

theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (exp 1) ∧
  (∀ (y : ℝ), y ∈ Set.Ioo 0 (exp 1) → f y ≥ f x) ∧
  f x = 1 + log 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3810_381034


namespace NUMINAMATH_CALUDE_power_sum_equality_l3810_381033

theorem power_sum_equality (a b : ℕ+) (h1 : 2^(a:ℕ) = 8^(b:ℕ)) (h2 : a + 2*b = 5) :
  2^(a:ℕ) + 8^(b:ℕ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3810_381033


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3810_381086

theorem fraction_equivalence : 
  ∃ (n : ℤ), (4 + n) / (7 + n) = 6 / 7 :=
by
  use 14
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3810_381086


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3810_381073

theorem quadratic_inequality_solution_sets 
  (a b c : ℝ) 
  (h1 : ∀ x, ax^2 + b*x + c ≥ 0 ↔ -1/3 ≤ x ∧ x ≤ 2) :
  ∀ x, c*x^2 + b*x + a < 0 ↔ -3 < x ∧ x < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3810_381073


namespace NUMINAMATH_CALUDE_road_repair_hours_l3810_381047

/-- Given that 39 persons can repair a road in 12 days working h hours a day,
    and 30 persons working 6 hours a day can complete the same work in 26 days,
    prove that h = 10. -/
theorem road_repair_hours (h : ℝ) : 
  39 * h * 12 = 30 * 6 * 26 → h = 10 := by
sorry

end NUMINAMATH_CALUDE_road_repair_hours_l3810_381047


namespace NUMINAMATH_CALUDE_birds_and_storks_l3810_381002

/-- Given a fence with birds and storks, prove that the initial number of birds
    is equal to the initial number of storks plus 3. -/
theorem birds_and_storks (initial_birds initial_storks : ℕ) : 
  initial_storks = 3 → 
  (initial_birds + initial_storks + 2 = initial_birds + 1) → 
  initial_birds = initial_storks + 3 := by
sorry

end NUMINAMATH_CALUDE_birds_and_storks_l3810_381002


namespace NUMINAMATH_CALUDE_unitsDigitOfSumOfSquares2023_l3810_381050

/-- The units digit of the sum of the squares of the first n odd, positive integers -/
def unitsDigitOfSumOfSquares (n : ℕ) : ℕ :=
  (n * 1 + n * 9 + (n / 2 + n % 2) * 5) % 10

/-- The theorem stating that the units digit of the sum of the squares 
    of the first 2023 odd, positive integers is 5 -/
theorem unitsDigitOfSumOfSquares2023 : 
  unitsDigitOfSumOfSquares 2023 = 5 := by
  sorry

#eval unitsDigitOfSumOfSquares 2023

end NUMINAMATH_CALUDE_unitsDigitOfSumOfSquares2023_l3810_381050


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_l3810_381048

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: A rectangle with length 5 and width 3 has area 15 and perimeter 16 -/
theorem rectangle_area_perimeter :
  let r : Rectangle := ⟨5, 3⟩
  area r = 15 ∧ perimeter r = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_l3810_381048


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l3810_381029

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 400)
  (h2 : rate = 15)
  (h3 : time = 2) :
  (principal * rate * time) / 100 = 60 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l3810_381029


namespace NUMINAMATH_CALUDE_B_is_largest_l3810_381032

def A : ℚ := 2010 / 2009 + 2010 / 2011
def B : ℚ := 2010 / 2011 + 2012 / 2011
def C : ℚ := 2011 / 2010 + 2011 / 2012

theorem B_is_largest : B > A ∧ B > C := by
  sorry

end NUMINAMATH_CALUDE_B_is_largest_l3810_381032


namespace NUMINAMATH_CALUDE_equal_edge_length_relation_l3810_381001

/-- Represents a hexagonal prism -/
structure HexagonalPrism :=
  (edge_length : ℝ)
  (total_edge_length : ℝ)
  (h_total : total_edge_length = 18 * edge_length)

/-- Represents a quadrangular pyramid -/
structure QuadrangularPyramid :=
  (edge_length : ℝ)
  (total_edge_length : ℝ)
  (h_total : total_edge_length = 8 * edge_length)

/-- 
Given a hexagonal prism and a quadrangular pyramid with equal edge lengths,
if the total edge length of the hexagonal prism is 81 cm,
then the total edge length of the quadrangular pyramid is 36 cm.
-/
theorem equal_edge_length_relation 
  (prism : HexagonalPrism) 
  (pyramid : QuadrangularPyramid) 
  (h_equal_edges : prism.edge_length = pyramid.edge_length) 
  (h_prism_total : prism.total_edge_length = 81) : 
  pyramid.total_edge_length = 36 := by
  sorry

end NUMINAMATH_CALUDE_equal_edge_length_relation_l3810_381001


namespace NUMINAMATH_CALUDE_problem_solution_l3810_381062

theorem problem_solution (x y : ℝ) 
  (h1 : x * y + x + y = 17) 
  (h2 : x^2 * y + x * y^2 = 66) : 
  x^4 + x^3 * y + x^2 * y^2 + x * y^3 + y^4 = 12499 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3810_381062


namespace NUMINAMATH_CALUDE_inequality_solution_one_inequality_solution_two_l3810_381094

-- Part 1
theorem inequality_solution_one : 
  {x : ℝ | 1 < x^2 - 3*x + 1 ∧ x^2 - 3*x + 1 < 9 - x} = 
  {x : ℝ | (-2 < x ∧ x < 0) ∨ (3 < x ∧ x < 4)} := by sorry

-- Part 2
def solution_set (a : ℝ) : Set ℝ :=
  {x : ℝ | (x - a) / (x - a^2) < 0}

theorem inequality_solution_two (a : ℝ) : 
  (a = 0 ∨ a = 1 → solution_set a = ∅) ∧
  (0 < a ∧ a < 1 → solution_set a = {x : ℝ | a^2 < x ∧ x < a}) ∧
  ((a < 0 ∨ a > 1) → solution_set a = {x : ℝ | a < x ∧ x < a^2}) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_one_inequality_solution_two_l3810_381094


namespace NUMINAMATH_CALUDE_unique_intersection_main_theorem_l3810_381020

/-- The curve C generated by rotating P(t, √(2)t^2 - 2t) by 45° anticlockwise -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + 2*p.1*p.2 + p.2^2 - p.1 - 3*p.2 = 0}

/-- The line y = -1/8 -/
def L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -1/8}

/-- The intersection of C and L is a singleton -/
theorem unique_intersection : (C ∩ L).Finite ∧ (C ∩ L).Nonempty := by
  sorry

/-- The main theorem stating that y = -1/8 intersects C at exactly one point -/
theorem main_theorem : ∃! p : ℝ × ℝ, p ∈ C ∧ p ∈ L := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_main_theorem_l3810_381020


namespace NUMINAMATH_CALUDE_ababab_divisible_by_101_l3810_381051

/-- Represents a 6-digit number of the form ababab -/
def ababab_number (a b : Nat) : Nat :=
  100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b

/-- Theorem stating that 101 is a factor of any ababab number -/
theorem ababab_divisible_by_101 (a b : Nat) (h : 0 < a ∧ a ≤ 9 ∧ b ≤ 9) :
  101 ∣ ababab_number a b :=
sorry

end NUMINAMATH_CALUDE_ababab_divisible_by_101_l3810_381051


namespace NUMINAMATH_CALUDE_policeman_catches_gangster_l3810_381026

-- Define the square
def Square := {p : ℝ × ℝ | -3 ≤ p.1 ∧ p.1 ≤ 3 ∧ -3 ≤ p.2 ∧ p.2 ≤ 3}

-- Define the perimeter of the square
def Perimeter := {p : ℝ × ℝ | (p.1 = -3 ∨ p.1 = 3) ∧ -3 ≤ p.2 ∧ p.2 ≤ 3} ∪
                 {p : ℝ × ℝ | (p.2 = -3 ∨ p.2 = 3) ∧ -3 ≤ p.1 ∧ p.1 ≤ 3}

-- Define the center of the square
def Center : ℝ × ℝ := (0, 0)

-- Define a vertex of the square
def Vertex : ℝ × ℝ := (3, 3)

-- Define the theorem
theorem policeman_catches_gangster (u v : ℝ) (hu : u > 0) (hv : v > 0) :
  ∃ (t : ℝ) (p g : ℝ × ℝ), t ≥ 0 ∧ p ∈ Square ∧ g ∈ Perimeter ∧ p = g ↔ u/v > 1/3 :=
sorry

end NUMINAMATH_CALUDE_policeman_catches_gangster_l3810_381026


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l3810_381013

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(2*x - 1) - 2
  f (1/2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l3810_381013
