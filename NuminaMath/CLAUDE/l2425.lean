import Mathlib

namespace NUMINAMATH_CALUDE_four_digit_numbers_two_repeated_l2425_242535

/-- The number of ways to choose 3 different digits from 0 to 9 -/
def three_digit_choices : ℕ := 10 * 9 * 8

/-- The number of ways to arrange 3 different digits with one repeated (forming a 4-digit number) -/
def repeated_digit_arrangements : ℕ := 6

/-- The number of four-digit numbers with exactly two repeated digits, including those starting with 0 -/
def total_with_leading_zero : ℕ := three_digit_choices * repeated_digit_arrangements

/-- The number of three-digit numbers with exactly two repeated digits (those starting with 0) -/
def starting_with_zero : ℕ := 9 * 8 * repeated_digit_arrangements

/-- The number of four-digit numbers with exactly two repeated digits -/
def four_digit_repeated : ℕ := total_with_leading_zero - starting_with_zero

theorem four_digit_numbers_two_repeated : four_digit_repeated = 3888 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_two_repeated_l2425_242535


namespace NUMINAMATH_CALUDE_first_tribe_term_longer_l2425_242532

/-- Represents the calendar system of the first tribe -/
structure Tribe1Calendar where
  months_per_year : Nat := 12
  days_per_month : Nat := 30

/-- Represents the calendar system of the second tribe -/
structure Tribe2Calendar where
  moons_per_year : Nat := 13
  weeks_per_moon : Nat := 4
  days_per_week : Nat := 7

/-- Calculates the number of days for the first tribe's term -/
def tribe1_term_days (cal : Tribe1Calendar) : Nat :=
  7 * cal.months_per_year * cal.days_per_month +
  1 * cal.days_per_month +
  18

/-- Calculates the number of days for the second tribe's term -/
def tribe2_term_days (cal : Tribe2Calendar) : Nat :=
  6 * cal.moons_per_year * cal.weeks_per_moon * cal.days_per_week +
  12 * cal.weeks_per_moon * cal.days_per_week +
  1 * cal.days_per_week +
  3

/-- Theorem stating that the first tribe's term is longer -/
theorem first_tribe_term_longer (cal1 : Tribe1Calendar) (cal2 : Tribe2Calendar) :
  tribe1_term_days cal1 > tribe2_term_days cal2 := by
  sorry

end NUMINAMATH_CALUDE_first_tribe_term_longer_l2425_242532


namespace NUMINAMATH_CALUDE_find_number_l2425_242544

theorem find_number : ∃! x : ℤ, (x + 12) / 4 = 12 ∧ (x + 12) % 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2425_242544


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_count_l2425_242520

theorem quadratic_integer_roots_count :
  let f (m : ℤ) := (∃ x₁ x₂ : ℤ, x₁ * x₂ = 30 ∧ x₁ + x₂ = m)
  (∃! s : Finset ℤ, (∀ m : ℤ, m ∈ s ↔ f m) ∧ s.card = 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_count_l2425_242520


namespace NUMINAMATH_CALUDE_freshman_class_size_l2425_242596

theorem freshman_class_size : ∃! n : ℕ, n < 500 ∧ n % 23 = 22 ∧ n % 21 = 14 ∧ n = 413 := by
  sorry

end NUMINAMATH_CALUDE_freshman_class_size_l2425_242596


namespace NUMINAMATH_CALUDE_subtracted_value_l2425_242590

theorem subtracted_value (chosen_number : ℕ) (subtracted_value : ℕ) : 
  chosen_number = 990 →
  (chosen_number / 9 : ℚ) - subtracted_value = 10 →
  subtracted_value = 100 := by
sorry

end NUMINAMATH_CALUDE_subtracted_value_l2425_242590


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2425_242504

-- Define the equation
def is_ellipse (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (3 + k) + y^2 / (2 - k) = 1 ∧
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

-- Theorem statement
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ k ∈ (Set.Ioo (-3) (-1/2) ∪ Set.Ioo (-1/2) 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2425_242504


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l2425_242551

def p (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 - 3 * x
def q (x : ℝ) : ℝ := 3 * x^2 - 4 * x - 5

theorem coefficient_of_x_squared :
  ∃ (a b c d e : ℝ), p x * q x = a * x^5 + b * x^4 + c * x^3 - 37 * x^2 + d * x + e :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l2425_242551


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l2425_242515

theorem min_value_and_inequality (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (∃ (y : ℝ), y = (a + 1/a) * (b + 1/b) ∧ 
    (∀ (z : ℝ), z = (a + 1/a) * (b + 1/b) → y ≤ z) ∧ 
    y = 25/4) ∧ 
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l2425_242515


namespace NUMINAMATH_CALUDE_fish_difference_l2425_242565

/-- Proves that Matthias has 15 fewer fish than Kenneth given the conditions in the problem -/
theorem fish_difference (micah_fish : ℕ) (total_fish : ℕ) : 
  micah_fish = 7 →
  total_fish = 34 →
  let kenneth_fish := 3 * micah_fish
  let matthias_fish := total_fish - micah_fish - kenneth_fish
  kenneth_fish - matthias_fish = 15 := by
sorry


end NUMINAMATH_CALUDE_fish_difference_l2425_242565


namespace NUMINAMATH_CALUDE_triangle_problem_l2425_242572

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The conditions given in the problem -/
def satisfiesConditions (t : Triangle) : Prop :=
  8 * t.a * t.b * Real.sin t.C = 3 * (t.b^2 + t.c^2 - t.a^2) ∧
  t.a = Real.sqrt 10 ∧
  t.c = 5

/-- The theorem to be proved -/
theorem triangle_problem (t : Triangle) (h : satisfiesConditions t) :
  Real.cos t.A = 4/5 ∧
  (t.a * t.b * Real.sin t.C / 2 = 15/2 ∨ t.a * t.b * Real.sin t.C / 2 = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2425_242572


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2425_242586

theorem trigonometric_identities (θ : Real) 
  (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1) : 
  Real.tan (2 * θ) = 4/3 ∧ 
  (Real.sin θ + Real.cos θ) / (Real.cos θ - 3 * Real.sin θ) = -3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2425_242586


namespace NUMINAMATH_CALUDE_integral_solution_square_sum_eq_product_l2425_242548

theorem integral_solution_square_sum_eq_product (a b c : ℤ) :
  a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_integral_solution_square_sum_eq_product_l2425_242548


namespace NUMINAMATH_CALUDE_log_sum_equals_three_l2425_242571

theorem log_sum_equals_three : 
  Real.log 0.125 / Real.log 0.5 + Real.log (Real.log (Real.log 64 / Real.log 4) / Real.log 3) / Real.log 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_three_l2425_242571


namespace NUMINAMATH_CALUDE_exterior_angle_theorem_l2425_242521

/-- The measure of the exterior angle BAC in a coplanar arrangement 
    where a square and a regular nonagon share a common side AD -/
def exterior_angle_BAC : ℝ := 130

/-- The measure of the interior angle of a regular nonagon -/
def nonagon_interior_angle : ℝ := 140

/-- The measure of the interior angle of a square -/
def square_interior_angle : ℝ := 90

theorem exterior_angle_theorem :
  exterior_angle_BAC = 360 - nonagon_interior_angle - square_interior_angle :=
by sorry

end NUMINAMATH_CALUDE_exterior_angle_theorem_l2425_242521


namespace NUMINAMATH_CALUDE_ariel_age_quadruples_l2425_242562

/-- Proves that it takes 15 years for Ariel to be four times her current age -/
theorem ariel_age_quadruples (current_age : ℕ) (years_passed : ℕ) : current_age = 5 →
  current_age + years_passed = 4 * current_age →
  years_passed = 15 := by
  sorry

end NUMINAMATH_CALUDE_ariel_age_quadruples_l2425_242562


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2425_242502

theorem geometric_sequence_product (a : ℕ → ℝ) (h : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) :
  a 3 * a 7 = 6 → a 2 * a 4 * a 6 * a 8 = 36 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2425_242502


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l2425_242523

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 9 ∣ n ∧
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ 3 ∣ m ∧ 4 ∣ m ∧ 9 ∣ m → n ≤ m) ∧
  n = 108 :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l2425_242523


namespace NUMINAMATH_CALUDE_parabola_equation_l2425_242516

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form x^2 = 2py -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line passing through a point M and tangent to a parabola at two points A and B -/
structure TangentLine where
  M : Point
  A : Point
  B : Point
  parabola : Parabola
  h1 : M.x = 2
  h2 : M.y = -2 * parabola.p
  h3 : A.x^2 = 2 * parabola.p * A.y
  h4 : B.x^2 = 2 * parabola.p * B.y

/-- The main theorem to prove -/
theorem parabola_equation (t : TangentLine) 
  (h : (t.A.y + t.B.y) / 2 = 6) : 
  t.parabola.p = 1 ∨ t.parabola.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2425_242516


namespace NUMINAMATH_CALUDE_paper_strip_dimensions_l2425_242557

theorem paper_strip_dimensions 
  (a b c : ℕ) 
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43) :
  a = 1 ∧ b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_paper_strip_dimensions_l2425_242557


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2425_242505

theorem infinite_series_sum : 
  (∑' k : ℕ, (k : ℝ) / (3 : ℝ) ^ k) = (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2425_242505


namespace NUMINAMATH_CALUDE_tom_remaining_pieces_l2425_242568

/-- The number of boxes Tom initially bought -/
def initial_boxes : ℕ := 12

/-- The number of boxes Tom gave to his little brother -/
def boxes_given : ℕ := 7

/-- The number of pieces in each box -/
def pieces_per_box : ℕ := 6

/-- Theorem: Tom still has 30 pieces of candy -/
theorem tom_remaining_pieces : 
  (initial_boxes - boxes_given) * pieces_per_box = 30 := by
  sorry

end NUMINAMATH_CALUDE_tom_remaining_pieces_l2425_242568


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2425_242594

theorem quadratic_inequality (x : ℝ) : x^2 - 9*x + 18 ≤ 0 ↔ 3 ≤ x ∧ x ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2425_242594


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2425_242559

theorem inscribed_square_area (triangle_area : ℝ) (square1_area : ℝ) (square2_fraction : ℝ) :
  square1_area = 441 →
  square2_fraction = 4 / 9 →
  triangle_area = 2 * square1_area →
  square2_fraction * triangle_area = 392 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2425_242559


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l2425_242524

/-- Given a quadratic polynomial 6x^2 + 36x + 150, prove that when rewritten in the form a(x+b)^2+c, 
    where a, b, and c are constants, a + b + c = 105 -/
theorem quadratic_rewrite_sum (x : ℝ) : 
  ∃ (a b c : ℝ), (∀ x, 6*x^2 + 36*x + 150 = a*(x+b)^2 + c) ∧ (a + b + c = 105) := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l2425_242524


namespace NUMINAMATH_CALUDE_james_stickers_l2425_242517

theorem james_stickers (x : ℕ) : x + 22 = 61 → x = 39 := by
  sorry

end NUMINAMATH_CALUDE_james_stickers_l2425_242517


namespace NUMINAMATH_CALUDE_total_fish_is_36_l2425_242556

/-- The total number of fish caught by Carla, Kyle, and Tasha -/
def total_fish (carla_fish kyle_fish : ℕ) : ℕ :=
  carla_fish + kyle_fish + kyle_fish

/-- Theorem: Given the conditions, the total number of fish caught is 36 -/
theorem total_fish_is_36 (carla_fish kyle_fish : ℕ) 
  (h1 : carla_fish = 8)
  (h2 : kyle_fish = 14) :
  total_fish carla_fish kyle_fish = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_is_36_l2425_242556


namespace NUMINAMATH_CALUDE_discount_ratio_proof_l2425_242599

/-- Proves that given a 15% discount on an item, if a person with $500 still needs $95 more to purchase it, the ratio of the additional money needed to the initial amount is 19:100. -/
theorem discount_ratio_proof (initial_amount : ℝ) (additional_needed : ℝ) (discount_rate : ℝ) :
  initial_amount = 500 →
  additional_needed = 95 →
  discount_rate = 0.15 →
  (additional_needed / initial_amount) = (19 / 100) :=
by sorry

end NUMINAMATH_CALUDE_discount_ratio_proof_l2425_242599


namespace NUMINAMATH_CALUDE_percentage_calculation_l2425_242541

theorem percentage_calculation (x : ℝ) (h : 0.4 * x = 160) : 0.5 * x = 200 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2425_242541


namespace NUMINAMATH_CALUDE_greatest_integer_solution_greatest_integer_value_l2425_242581

theorem greatest_integer_solution (x : ℤ) : (5 - 4*x > 17) ↔ (x < -3) :=
  sorry

theorem greatest_integer_value : ∃ (x : ℤ), (∀ (y : ℤ), (5 - 4*y > 17) → y ≤ x) ∧ (5 - 4*x > 17) ∧ x = -4 :=
  sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_greatest_integer_value_l2425_242581


namespace NUMINAMATH_CALUDE_smallest_sum_squared_pythagorean_triple_l2425_242514

theorem smallest_sum_squared_pythagorean_triple (p q r : ℤ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h : p^2 + q^2 = r^2) : 
  ∃ (p' q' r' : ℤ), p'^2 + q'^2 = r'^2 ∧ (p' + q' + r')^2 = 4 ∧ 
  ∀ (a b c : ℤ), a^2 + b^2 = c^2 → (a + b + c)^2 ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_squared_pythagorean_triple_l2425_242514


namespace NUMINAMATH_CALUDE_sally_net_earnings_l2425_242569

def calculate_net_earnings (
  first_month_income : ℝ)
  (first_month_expenses : ℝ)
  (side_hustle : ℝ)
  (income_raise_percentage : ℝ)
  (expense_increase_percentage : ℝ) : ℝ :=
  let first_month := first_month_income + side_hustle - first_month_expenses
  let second_month_income := first_month_income * (1 + income_raise_percentage)
  let second_month_expenses := first_month_expenses * (1 + expense_increase_percentage)
  let second_month := second_month_income + side_hustle - second_month_expenses
  first_month + second_month

theorem sally_net_earnings :
  calculate_net_earnings 1000 200 150 0.1 0.15 = 1970 := by
  sorry

end NUMINAMATH_CALUDE_sally_net_earnings_l2425_242569


namespace NUMINAMATH_CALUDE_second_car_speed_l2425_242513

theorem second_car_speed 
  (highway_length : ℝ) 
  (first_car_speed : ℝ) 
  (meeting_time : ℝ) 
  (h1 : highway_length = 175) 
  (h2 : first_car_speed = 25) 
  (h3 : meeting_time = 2.5) : 
  ∃ second_car_speed : ℝ, 
    first_car_speed * meeting_time + second_car_speed * meeting_time = highway_length ∧ 
    second_car_speed = 45 := by
sorry

end NUMINAMATH_CALUDE_second_car_speed_l2425_242513


namespace NUMINAMATH_CALUDE_shaded_triangle_area_and_percentage_l2425_242578

/-- Given an equilateral triangle with side length 4 cm, prove the area and percentage of a shaded region -/
theorem shaded_triangle_area_and_percentage :
  let side_length : ℝ := 4
  let original_height : ℝ := side_length * (Real.sqrt 3) / 2
  let original_area : ℝ := side_length^2 * (Real.sqrt 3) / 4
  let shaded_base : ℝ := side_length * 3 / 4
  let shaded_height : ℝ := original_height / 2
  let shaded_area : ℝ := shaded_base * shaded_height / 2
  let percentage : ℝ := shaded_area / original_area * 100
  shaded_area = 3 * (Real.sqrt 3) / 2 ∧ percentage = 37.5 := by
  sorry


end NUMINAMATH_CALUDE_shaded_triangle_area_and_percentage_l2425_242578


namespace NUMINAMATH_CALUDE_vertex_on_x_axis_l2425_242595

/-- The parabola equation -/
def parabola (x d : ℝ) : ℝ := x^2 - 6*x + d

/-- The x-coordinate of the vertex -/
def vertex_x : ℝ := 3

/-- The y-coordinate of the vertex -/
def vertex_y (d : ℝ) : ℝ := parabola vertex_x d

theorem vertex_on_x_axis (d : ℝ) : vertex_y d = 0 ↔ d = 9 := by sorry

end NUMINAMATH_CALUDE_vertex_on_x_axis_l2425_242595


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2425_242583

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : 
  3 < (c^3 - a^3 - b^3) / (c * (c - a) * (c - b)) ∧ 
  (c^3 - a^3 - b^3) / (c * (c - a) * (c - b)) < Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l2425_242583


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2425_242546

/-- Given a square with side length x + 2 and an equilateral triangle with equal perimeter,
    where x = 4, prove that the side length of the equilateral triangle is 8. -/
theorem equilateral_triangle_side_length (x : ℝ) (square_side : ℝ) (triangle_side : ℝ) : 
  x = 4 →
  square_side = x + 2 →
  4 * square_side = 3 * triangle_side →
  triangle_side = 8 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2425_242546


namespace NUMINAMATH_CALUDE_distance_between_places_l2425_242527

/-- The distance between two places given speed changes and time differences --/
theorem distance_between_places (x : ℝ) (y : ℝ) : 
  ((x + 6) * (y - 5/60) = x * y) →
  ((x - 5) * (y + 6/60) = x * y) →
  x * y = 15 := by
sorry

end NUMINAMATH_CALUDE_distance_between_places_l2425_242527


namespace NUMINAMATH_CALUDE_supermarket_sales_results_l2425_242525

/-- Supermarket sales model -/
structure SupermarketSales where
  cost_price : ℝ
  min_selling_price : ℝ
  base_price : ℝ
  base_sales : ℝ
  price_increase : ℝ
  sales_decrease : ℝ

/-- Daily sales volume as a function of selling price -/
def sales_volume (model : SupermarketSales) (x : ℝ) : ℝ :=
  model.base_sales - model.sales_decrease * (x - model.base_price)

/-- Daily profit as a function of selling price -/
def daily_profit (model : SupermarketSales) (x : ℝ) : ℝ :=
  (x - model.cost_price) * (sales_volume model x)

/-- Theorem stating the main results of the supermarket sales problem -/
theorem supermarket_sales_results (model : SupermarketSales) 
  (h1 : model.cost_price = 40)
  (h2 : model.min_selling_price = 45)
  (h3 : model.base_price = 45)
  (h4 : model.base_sales = 700)
  (h5 : model.price_increase = 1)
  (h6 : model.sales_decrease = 20) :
  (∀ x, sales_volume model x = -20 * x + 1600) ∧
  (∃ x, x ≥ model.min_selling_price ∧ daily_profit model x = 6000 ∧ x = 50) ∧
  (∃ x, x ≥ model.min_selling_price ∧ 
    ∀ y, y ≥ model.min_selling_price → daily_profit model x ≥ daily_profit model y) ∧
  (daily_profit model 60 = 8000) := by
  sorry

#check supermarket_sales_results

end NUMINAMATH_CALUDE_supermarket_sales_results_l2425_242525


namespace NUMINAMATH_CALUDE_line_CR_tangent_to_circumcircle_l2425_242539

-- Define the square ABCD
structure Square (A B C D : ℝ × ℝ) : Prop where
  is_square : A = (0, 0) ∧ B = (0, 1) ∧ C = (1, 1) ∧ D = (1, 0)

-- Define point P on BC
def P (k : ℝ) : ℝ × ℝ := (k, 1)

-- Define square APRS
structure SquareAPRS (A P R S : ℝ × ℝ) : Prop where
  is_square : A = (0, 0) ∧ P.1 = k ∧ P.2 = 1 ∧
              S = (1, -k) ∧ R = (1+k, 1-k)

-- Define the circumcircle of triangle ABC
def CircumcircleABC (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 0.5)^2 + (p.2 - 0.5)^2 = 0.5^2}

-- Define the line CR
def LineCR (C R : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.2 - C.2) = -1 * (p.1 - C.1)}

-- Theorem statement
theorem line_CR_tangent_to_circumcircle 
  (A B C D : ℝ × ℝ) 
  (k : ℝ) 
  (P R S : ℝ × ℝ) 
  (h1 : Square A B C D) 
  (h2 : 0 ≤ k ∧ k ≤ 1) 
  (h3 : P = (k, 1)) 
  (h4 : SquareAPRS A P R S) :
  ∃ (x : ℝ × ℝ), x ∈ CircumcircleABC A B C ∧ x ∈ LineCR C R ∧
  ∀ (y : ℝ × ℝ), y ≠ x → y ∈ CircumcircleABC A B C → y ∉ LineCR C R :=
sorry


end NUMINAMATH_CALUDE_line_CR_tangent_to_circumcircle_l2425_242539


namespace NUMINAMATH_CALUDE_special_function_unique_l2425_242542

/-- A function satisfying the given properties -/
def special_function (g : ℝ → ℝ) : Prop :=
  g 2 = 2 ∧ ∀ x y : ℝ, g (x * y + g x) = x * g y + g x

theorem special_function_unique (g : ℝ → ℝ) (h : special_function g) :
  ∀ x : ℝ, g x = 2 * x :=
sorry

end NUMINAMATH_CALUDE_special_function_unique_l2425_242542


namespace NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l2425_242540

theorem min_value_of_expression (x y : ℝ) : (x * y + 1)^2 + (x - y)^2 ≥ 1 := by
  sorry

theorem lower_bound_achievable : ∃ x y : ℝ, (x * y + 1)^2 + (x - y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l2425_242540


namespace NUMINAMATH_CALUDE_identity_proof_l2425_242567

theorem identity_proof (a b c x y z : ℝ) : 
  (a*x + b*y + c*z)^2 + (b*x + c*y + a*z)^2 + (c*x + a*y + b*z)^2 = 
  (c*x + b*y + a*z)^2 + (b*x + a*y + c*z)^2 + (a*x + c*y + b*z)^2 := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l2425_242567


namespace NUMINAMATH_CALUDE_euston_carriages_l2425_242508

/-- The number of carriages in different towns --/
structure Carriages where
  euston : ℕ
  norfolk : ℕ
  norwich : ℕ
  flying_scotsman : ℕ

/-- The conditions of the carriage problem --/
def carriage_conditions (c : Carriages) : Prop :=
  c.euston = c.norfolk + 20 ∧
  c.norwich = 100 ∧
  c.flying_scotsman = c.norwich + 20 ∧
  c.euston + c.norfolk + c.norwich + c.flying_scotsman = 460

/-- Theorem stating that under the given conditions, Euston had 130 carriages --/
theorem euston_carriages (c : Carriages) (h : carriage_conditions c) : c.euston = 130 := by
  sorry

end NUMINAMATH_CALUDE_euston_carriages_l2425_242508


namespace NUMINAMATH_CALUDE_parabola_fv_unique_value_l2425_242511

/-- A parabola with vertex V and focus F -/
structure Parabola where
  V : ℝ × ℝ
  F : ℝ × ℝ

/-- A point on a parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_fv_unique_value (p : Parabola) 
  (A B : PointOnParabola p)
  (h1 : distance A.point p.F = 25)
  (h2 : distance A.point p.V = 24)
  (h3 : distance B.point p.F = 9) :
  distance p.F p.V = 9 := sorry

end NUMINAMATH_CALUDE_parabola_fv_unique_value_l2425_242511


namespace NUMINAMATH_CALUDE_henry_twice_jills_age_l2425_242537

/-- Given that Henry and Jill's present ages sum to 40, with Henry being 23 and Jill being 17,
    this theorem proves that 11 years ago, Henry was twice the age of Jill. -/
theorem henry_twice_jills_age (henry_age : ℕ) (jill_age : ℕ) :
  henry_age + jill_age = 40 →
  henry_age = 23 →
  jill_age = 17 →
  ∃ (years_ago : ℕ), henry_age - years_ago = 2 * (jill_age - years_ago) ∧ years_ago = 11 := by
  sorry

end NUMINAMATH_CALUDE_henry_twice_jills_age_l2425_242537


namespace NUMINAMATH_CALUDE_parabola_vertex_l2425_242593

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = (x - 6)^2 + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (6, 3)

/-- Theorem: The vertex of the parabola y = (x - 6)^2 + 3 is at (6, 3) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2425_242593


namespace NUMINAMATH_CALUDE_rationalization_sqrt_five_l2425_242549

/-- Rationalization of (2+√5)/(2-√5) -/
theorem rationalization_sqrt_five : ∃ (A B C : ℤ), 
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C ∧ 
  A = -9 ∧ B = -4 ∧ C = 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalization_sqrt_five_l2425_242549


namespace NUMINAMATH_CALUDE_number_division_problem_l2425_242529

theorem number_division_problem (x : ℝ) : x / 5 = 80 + x / 6 ↔ x = 2400 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2425_242529


namespace NUMINAMATH_CALUDE_jerry_speed_is_40_l2425_242533

-- Define the given conditions
def jerry_time : ℚ := 1/2  -- 30 minutes in hours
def beth_time : ℚ := 5/6   -- 50 minutes in hours
def beth_speed : ℚ := 30   -- miles per hour
def route_difference : ℚ := 5  -- miles

-- Theorem to prove
theorem jerry_speed_is_40 :
  let beth_distance : ℚ := beth_speed * beth_time
  let jerry_distance : ℚ := beth_distance - route_difference
  jerry_distance / jerry_time = 40 := by
  sorry


end NUMINAMATH_CALUDE_jerry_speed_is_40_l2425_242533


namespace NUMINAMATH_CALUDE_bus_children_difference_l2425_242500

theorem bus_children_difference (initial : ℕ) (got_off : ℕ) (final : ℕ) :
  initial = 5 → got_off = 63 → final = 14 →
  ∃ (got_on : ℕ), got_on - got_off = 9 ∧ initial - got_off + got_on = final :=
by sorry

end NUMINAMATH_CALUDE_bus_children_difference_l2425_242500


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l2425_242579

theorem square_difference_of_integers (a b : ℤ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 60) (h4 : a - b = 16) : 
  a^2 - b^2 = 960 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l2425_242579


namespace NUMINAMATH_CALUDE_triangular_number_gcd_bound_l2425_242587

/-- The nth triangular number -/
def T (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The statement to be proved -/
theorem triangular_number_gcd_bound :
  (∀ n : ℕ, n > 0 → Nat.gcd (8 * T n) (n + 1) ≤ 4) ∧
  (∃ n : ℕ, n > 0 ∧ Nat.gcd (8 * T n) (n + 1) = 4) := by
  sorry

end NUMINAMATH_CALUDE_triangular_number_gcd_bound_l2425_242587


namespace NUMINAMATH_CALUDE_least_integer_abs_value_l2425_242526

theorem least_integer_abs_value (y : ℤ) : 
  (∀ z : ℤ, 3 * |z| + 2 < 20 → y ≤ z) ↔ y = -5 := by sorry

end NUMINAMATH_CALUDE_least_integer_abs_value_l2425_242526


namespace NUMINAMATH_CALUDE_expected_socks_theorem_l2425_242576

/-- The expected number of socks picked to retrieve both favorite socks -/
def expected_socks_picked (n : ℕ) : ℚ :=
  2 * (n + 1) / 3

/-- Theorem: The expected number of socks picked to retrieve both favorite socks is 2(n+1)/3 -/
theorem expected_socks_theorem (n : ℕ) (h : n ≥ 2) :
  expected_socks_picked n = 2 * (n + 1) / 3 := by
  sorry

#check expected_socks_theorem

end NUMINAMATH_CALUDE_expected_socks_theorem_l2425_242576


namespace NUMINAMATH_CALUDE_brothers_age_relation_l2425_242553

theorem brothers_age_relation : 
  ∃ x : ℕ, (15 + x = 2 * (5 + x)) ∧ (x = 5) := by sorry

end NUMINAMATH_CALUDE_brothers_age_relation_l2425_242553


namespace NUMINAMATH_CALUDE_cube_painting_theorem_l2425_242507

/-- The number of rotational symmetries of a cube -/
def cube_symmetries : ℕ := 24

/-- The number of faces on a cube -/
def cube_faces : ℕ := 6

/-- The number of available colors -/
def available_colors : ℕ := 7

/-- The number of distinguishable ways to paint a cube -/
def distinguishable_cubes : ℕ := 210

theorem cube_painting_theorem :
  (Nat.choose available_colors cube_faces * Nat.factorial cube_faces) / cube_symmetries = distinguishable_cubes :=
sorry

end NUMINAMATH_CALUDE_cube_painting_theorem_l2425_242507


namespace NUMINAMATH_CALUDE_luca_drink_cost_l2425_242561

/-- The cost of Luca's lunch items and the total bill -/
structure LunchCost where
  sandwich : ℝ
  discount_rate : ℝ
  avocado : ℝ
  salad : ℝ
  total_bill : ℝ

/-- Calculate the cost of Luca's drink given his lunch costs -/
def drink_cost (lunch : LunchCost) : ℝ :=
  lunch.total_bill - (lunch.sandwich * (1 - lunch.discount_rate) + lunch.avocado + lunch.salad)

/-- Theorem: Given Luca's lunch costs, the cost of his drink is $2 -/
theorem luca_drink_cost :
  let lunch : LunchCost := {
    sandwich := 8,
    discount_rate := 0.25,
    avocado := 1,
    salad := 3,
    total_bill := 12
  }
  drink_cost lunch = 2 := by sorry

end NUMINAMATH_CALUDE_luca_drink_cost_l2425_242561


namespace NUMINAMATH_CALUDE_distance_to_directrix_l2425_242545

/-- A parabola C is defined by the equation y² = 2px. -/
structure Parabola where
  p : ℝ

/-- A point on a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The theorem states that for a parabola C where point A(1, √5) lies on it,
    the distance from A to the directrix of C is 9/4. -/
theorem distance_to_directrix (C : Parabola) (A : Point) :
  A.x = 1 →
  A.y = Real.sqrt 5 →
  A.y ^ 2 = 2 * C.p * A.x →
  (A.x + C.p / 2) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_directrix_l2425_242545


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2425_242563

theorem polynomial_remainder (x : ℂ) : 
  x^2 - x + 1 = 0 → (2*x^5 - x^4 + x^2 - 1)*(x^3 - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2425_242563


namespace NUMINAMATH_CALUDE_unique_positive_number_l2425_242598

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 8 = 128 / x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l2425_242598


namespace NUMINAMATH_CALUDE_A_intersect_B_l2425_242575

def A : Set ℕ := {0, 2, 4}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2^x}

theorem A_intersect_B : A ∩ B = {4} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2425_242575


namespace NUMINAMATH_CALUDE_lines_neither_perpendicular_nor_parallel_l2425_242555

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (contained : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- State the theorem
theorem lines_neither_perpendicular_nor_parallel
  (m n l : Line) (α β : Plane)
  (h1 : contained m α)
  (h2 : contained n β)
  (h3 : perpendicularPlanes α β)
  (h4 : intersect α β l)
  (h5 : ¬ perpendicular m l ∧ ¬ parallel m l)
  (h6 : ¬ perpendicular n l ∧ ¬ parallel n l) :
  ¬ perpendicular m n ∧ ¬ parallel m n :=
by
  sorry

end NUMINAMATH_CALUDE_lines_neither_perpendicular_nor_parallel_l2425_242555


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_and_even_l2425_242501

def f (x : ℝ) : ℝ := -2 * x^2

theorem f_monotone_decreasing_and_even :
  (∀ x y, x > 0 → y > 0 → x < y → f x > f y) ∧
  (∀ x, x > 0 → f x = f (-x)) :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_and_even_l2425_242501


namespace NUMINAMATH_CALUDE_expression_evaluation_l2425_242570

theorem expression_evaluation (x : ℤ) (h1 : -1 ≤ x) (h2 : x ≤ 2) 
  (h3 : x ≠ 1) (h4 : x ≠ 0) (h5 : x ≠ 2) : 
  (x^2 - 1) / (x^2 - 2*x + 1) + (x^2 - 2*x) / (x - 2) / x = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2425_242570


namespace NUMINAMATH_CALUDE_crease_line_equation_l2425_242538

/-- Given a circle with radius R and a point A inside the circle at distance a from the center,
    the set of all points (x, y) on the crease lines formed by folding the paper so that any point
    on the circumference coincides with A satisfies the equation:
    (2x - a)^2 / R^2 + 4y^2 / (R^2 - a^2) = 1 -/
theorem crease_line_equation (R a x y : ℝ) (h1 : R > 0) (h2 : 0 ≤ a) (h3 : a < R) :
  (∃ (A' : ℝ × ℝ), (A'.1^2 + A'.2^2 = R^2) ∧
   ((x - A'.1)^2 + (y - A'.2)^2 = (x - a)^2 + y^2)) ↔
  (2*x - a)^2 / R^2 + 4*y^2 / (R^2 - a^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_crease_line_equation_l2425_242538


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2425_242522

/-- Given a line with slope 4 passing through (5, -2), prove that m + b = -18 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = 4 ∧ 
  -2 = 4 * 5 + b → 
  m + b = -18 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2425_242522


namespace NUMINAMATH_CALUDE_equality_of_two_numbers_l2425_242584

theorem equality_of_two_numbers (x y z : ℝ) 
  (h : x * y + z = y * z + x ∧ y * z + x = z * x + y) : 
  x = y ∨ y = z ∨ z = x := by
  sorry

end NUMINAMATH_CALUDE_equality_of_two_numbers_l2425_242584


namespace NUMINAMATH_CALUDE_four_heads_in_five_tosses_l2425_242580

def n : ℕ := 5
def k : ℕ := 4
def p : ℚ := 1/2

def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1 - p)^(n - k)

theorem four_heads_in_five_tosses : 
  binomial_probability n k p = 5/32 := by sorry

end NUMINAMATH_CALUDE_four_heads_in_five_tosses_l2425_242580


namespace NUMINAMATH_CALUDE_complete_square_for_given_equation_l2425_242528

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the result of completing the square for a quadratic equation -/
structure CompletedSquareForm where
  r : ℝ
  s : ℝ

/-- Completes the square for a given quadratic equation -/
def completeSquare (eq : QuadraticEquation) : CompletedSquareForm :=
  sorry

theorem complete_square_for_given_equation :
  let eq := QuadraticEquation.mk 9 (-18) (-720)
  let result := completeSquare eq
  result.s = 81 := by sorry

end NUMINAMATH_CALUDE_complete_square_for_given_equation_l2425_242528


namespace NUMINAMATH_CALUDE_volume_cube_inscribed_sphere_l2425_242585

/-- The volume of a cube inscribed in a sphere -/
theorem volume_cube_inscribed_sphere (R : ℝ) (h : R > 0) :
  ∃ (V : ℝ), V = (8 / 9) * Real.sqrt 3 * R^3 ∧ V > 0 := by sorry

end NUMINAMATH_CALUDE_volume_cube_inscribed_sphere_l2425_242585


namespace NUMINAMATH_CALUDE_parabola_y1_gt_y2_l2425_242574

/-- A parabola with axis of symmetry at x = 1 -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a > 0

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_y1_gt_y2 (p : Parabola) :
  p.y_at (-1) > p.y_at 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_y1_gt_y2_l2425_242574


namespace NUMINAMATH_CALUDE_field_trip_participation_l2425_242531

/-- Given a class of students where:
    - 4/5 of students left on the first vehicle
    - Of those who stayed, 1/3 didn't want to go
    - When another vehicle was found, 1/2 of the remaining students who wanted to go were able to join
    Prove that the fraction of students who went on the field trip is 13/15 -/
theorem field_trip_participation (total_students : ℕ) (total_students_pos : total_students > 0) :
  let first_vehicle := (4 : ℚ) / 5 * total_students
  let stayed_behind := total_students - first_vehicle
  let not_wanting_to_go := (1 : ℚ) / 3 * stayed_behind
  let wanting_to_go := stayed_behind - not_wanting_to_go
  let additional_joiners := (1 : ℚ) / 2 * wanting_to_go
  first_vehicle + additional_joiners = (13 : ℚ) / 15 * total_students :=
by sorry

end NUMINAMATH_CALUDE_field_trip_participation_l2425_242531


namespace NUMINAMATH_CALUDE_pencil_count_multiple_of_ten_l2425_242536

/-- Given that 1230 pens and some pencils are distributed among students, 
    with each student receiving the same number of pens and pencils, 
    and the maximum number of students is 10, 
    prove that the total number of pencils is a multiple of 10. -/
theorem pencil_count_multiple_of_ten (total_pens : ℕ) (total_pencils : ℕ) (num_students : ℕ) :
  total_pens = 1230 →
  num_students ≤ 10 →
  num_students ∣ total_pens →
  num_students ∣ total_pencils →
  num_students = 10 →
  10 ∣ total_pencils :=
by sorry

end NUMINAMATH_CALUDE_pencil_count_multiple_of_ten_l2425_242536


namespace NUMINAMATH_CALUDE_deck_size_l2425_242564

theorem deck_size (r b : ℕ) : 
  r ≠ 0 → 
  b ≠ 0 → 
  r / (r + b) = 1 / 4 → 
  r / (r + b + 6) = 1 / 6 → 
  r + b = 12 := by
sorry

end NUMINAMATH_CALUDE_deck_size_l2425_242564


namespace NUMINAMATH_CALUDE_final_racers_count_l2425_242582

def race_elimination (initial_racers : ℕ) : ℕ :=
  let after_first := initial_racers - 10
  let after_second := after_first - (after_first / 3)
  let after_third := after_second - (after_second / 4)
  let after_fourth := after_third - (after_third / 3)
  let after_fifth := after_fourth - (after_fourth / 2)
  after_fifth - (after_fifth * 3 / 4)

theorem final_racers_count :
  race_elimination 200 = 8 := by
  sorry

end NUMINAMATH_CALUDE_final_racers_count_l2425_242582


namespace NUMINAMATH_CALUDE_zoo_field_trip_l2425_242577

theorem zoo_field_trip (students : ℕ) (adults : ℕ) (vans : ℕ) : 
  students = 12 → adults = 3 → vans = 3 → (students + adults) / vans = 5 := by
  sorry

end NUMINAMATH_CALUDE_zoo_field_trip_l2425_242577


namespace NUMINAMATH_CALUDE_reciprocal_of_complex_l2425_242558

theorem reciprocal_of_complex (z : ℂ) (h : z = 5 + I) : 
  z⁻¹ = 5 / 26 - (1 / 26) * I :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_complex_l2425_242558


namespace NUMINAMATH_CALUDE_decimal_2015_is_octal_3737_l2425_242554

/-- Converts a natural number from decimal to octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Checks if a list of digits represents a valid octal number -/
def is_valid_octal (l : List ℕ) : Prop :=
  l.all (· < 8) ∧ l ≠ []

theorem decimal_2015_is_octal_3737 :
  decimal_to_octal 2015 = [3, 7, 3, 7] ∧ is_valid_octal [3, 7, 3, 7] := by
  sorry

#eval decimal_to_octal 2015

end NUMINAMATH_CALUDE_decimal_2015_is_octal_3737_l2425_242554


namespace NUMINAMATH_CALUDE_solution_a_amount_l2425_242519

/-- Proves that the amount of Solution A used is 100 milliliters -/
theorem solution_a_amount (solution_a : ℝ) (solution_b : ℝ) : solution_a = 100 :=
  by
  -- Solution B is 500 milliliters more than Solution A
  have h1 : solution_b = solution_a + 500 := by sorry
  
  -- Solution A is 16% alcohol
  have h2 : solution_a * 0.16 = solution_a * (16 / 100) := by sorry
  
  -- Solution B is 10% alcohol
  have h3 : solution_b * 0.10 = solution_b * (10 / 100) := by sorry
  
  -- The resulting mixture has 76 milliliters of pure alcohol
  have h4 : solution_a * (16 / 100) + solution_b * (10 / 100) = 76 := by sorry
  
  sorry -- Skip the proof


end NUMINAMATH_CALUDE_solution_a_amount_l2425_242519


namespace NUMINAMATH_CALUDE_circle_equation_l2425_242552

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def is_in_first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def is_tangent_to_line (C : Circle) (a b c : ℝ) : Prop :=
  abs (a * C.center.1 + b * C.center.2 + c) = C.radius * Real.sqrt (a^2 + b^2)

def is_tangent_to_x_axis (C : Circle) : Prop :=
  C.center.2 = C.radius

-- State the theorem
theorem circle_equation (C : Circle) :
  C.radius = 1 →
  is_in_first_quadrant C.center →
  is_tangent_to_line C 4 (-3) 0 →
  is_tangent_to_x_axis C →
  ∀ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1 - C.center.1)^2 + (p.2 - C.center.2)^2 = C.radius^2} :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2425_242552


namespace NUMINAMATH_CALUDE_cost_of_pancakes_l2425_242591

/-- The cost of pancakes given initial order, tax, payment, and change --/
theorem cost_of_pancakes 
  (eggs_cost : ℕ)
  (cocoa_cost : ℕ)
  (cocoa_quantity : ℕ)
  (tax : ℕ)
  (payment : ℕ)
  (change : ℕ)
  (h1 : eggs_cost = 3)
  (h2 : cocoa_cost = 2)
  (h3 : cocoa_quantity = 2)
  (h4 : tax = 1)
  (h5 : payment = 15)
  (h6 : change = 1)
  : ℕ := by
  sorry

#check cost_of_pancakes

end NUMINAMATH_CALUDE_cost_of_pancakes_l2425_242591


namespace NUMINAMATH_CALUDE_pure_imaginary_m_equals_four_l2425_242509

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number formed by the given expression. -/
def ComplexExpression (m : ℝ) : ℂ :=
  ⟨m^2 - 3*m - 4, m^2 - 5*m - 6⟩

theorem pure_imaginary_m_equals_four :
  ∃ m : ℝ, IsPureImaginary (ComplexExpression m) → m = 4 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_m_equals_four_l2425_242509


namespace NUMINAMATH_CALUDE_product_n_n_plus_one_is_even_l2425_242530

theorem product_n_n_plus_one_is_even (n : ℕ) : Even (n * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_product_n_n_plus_one_is_even_l2425_242530


namespace NUMINAMATH_CALUDE_price_per_book_is_4_80_l2425_242589

/-- The price per book when saving for a clarinet --/
def price_per_book (clarinet_cost initial_savings total_books : ℚ) : ℚ :=
  let additional_savings := clarinet_cost - initial_savings
  let halfway_savings := additional_savings / 2
  let total_to_save := halfway_savings + additional_savings
  total_to_save / total_books

/-- Theorem: The price per book is $4.80 --/
theorem price_per_book_is_4_80 :
  price_per_book 90 10 25 = 4.80 := by
  sorry

end NUMINAMATH_CALUDE_price_per_book_is_4_80_l2425_242589


namespace NUMINAMATH_CALUDE_bolt_defect_probability_l2425_242566

theorem bolt_defect_probability :
  let machine1_production : ℝ := 0.30
  let machine2_production : ℝ := 0.25
  let machine3_production : ℝ := 0.45
  let machine1_defect_rate : ℝ := 0.02
  let machine2_defect_rate : ℝ := 0.01
  let machine3_defect_rate : ℝ := 0.03
  machine1_production + machine2_production + machine3_production = 1 →
  machine1_production * machine1_defect_rate +
  machine2_production * machine2_defect_rate +
  machine3_production * machine3_defect_rate = 0.022 := by
sorry

end NUMINAMATH_CALUDE_bolt_defect_probability_l2425_242566


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l2425_242573

/-- A quadratic polynomial that satisfies specific conditions -/
def q (x : ℚ) : ℚ := (4 * x^2 - 6 * x + 5) / 3

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-1) = 5 ∧ q 2 = 3 ∧ q 4 = 15 := by
  sorry


end NUMINAMATH_CALUDE_q_satisfies_conditions_l2425_242573


namespace NUMINAMATH_CALUDE_b_investment_is_7200_l2425_242588

/-- Represents the investment and profit distribution in a partnership business. -/
structure PartnershipBusiness where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  a_profit_share : ℕ

/-- The investment of partner B in the business. -/
def b_investment (pb : PartnershipBusiness) : ℕ :=
  7200

/-- Theorem stating that B's investment is 7200, given the conditions of the problem. -/
theorem b_investment_is_7200 (pb : PartnershipBusiness) 
    (h1 : pb.a_investment = 2400)
    (h2 : pb.c_investment = 9600)
    (h3 : pb.total_profit = 9000)
    (h4 : pb.a_profit_share = 1125) :
  b_investment pb = 7200 := by
  sorry

end NUMINAMATH_CALUDE_b_investment_is_7200_l2425_242588


namespace NUMINAMATH_CALUDE_expand_and_simplify_polynomial_l2425_242597

theorem expand_and_simplify_polynomial (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_polynomial_l2425_242597


namespace NUMINAMATH_CALUDE_bob_distance_when_met_l2425_242560

/-- The distance between X and Y in miles -/
def total_distance : ℝ := 60

/-- Yolanda's walking rate in miles per hour -/
def yolanda_rate : ℝ := 5

/-- Bob's walking rate in miles per hour -/
def bob_rate : ℝ := 6

/-- The time difference in hours between Yolanda's and Bob's start -/
def time_difference : ℝ := 1

/-- The theorem stating that Bob walked 30 miles when they met -/
theorem bob_distance_when_met : 
  ∃ (t : ℝ), 
    t > 0 ∧ 
    yolanda_rate * (t + time_difference) + bob_rate * t = total_distance ∧ 
    bob_rate * t = 30 := by
  sorry

end NUMINAMATH_CALUDE_bob_distance_when_met_l2425_242560


namespace NUMINAMATH_CALUDE_twentieth_meeting_at_D_l2425_242512

/-- Represents a meeting point in the pool lane -/
inductive MeetingPoint
| C
| D

/-- Represents an athlete swimming in the pool lane -/
structure Athlete where
  speed : ℝ
  speed_positive : speed > 0

/-- Represents a swimming scenario with two athletes -/
structure SwimmingScenario where
  athlete1 : Athlete
  athlete2 : Athlete
  different_speeds : athlete1.speed ≠ athlete2.speed
  first_meeting : MeetingPoint
  second_meeting : MeetingPoint
  first_meeting_is_C : first_meeting = MeetingPoint.C
  second_meeting_is_D : second_meeting = MeetingPoint.D

/-- The theorem stating that the 20th meeting occurs at point D -/
theorem twentieth_meeting_at_D (scenario : SwimmingScenario) :
  (fun n => if n % 2 = 0 then MeetingPoint.D else MeetingPoint.C) 20 = MeetingPoint.D :=
sorry

end NUMINAMATH_CALUDE_twentieth_meeting_at_D_l2425_242512


namespace NUMINAMATH_CALUDE_three_quantities_problem_l2425_242543

theorem three_quantities_problem (x y z : ℕ) : 
  y = x + 8 →
  z = 3 * x →
  x + y + z = 108 →
  (x = 20 ∧ y = 28 ∧ z = 60) := by
  sorry

end NUMINAMATH_CALUDE_three_quantities_problem_l2425_242543


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l2425_242510

theorem solve_cubic_equation :
  ∃ y : ℝ, (y - 3)^3 = (1/27)⁻¹ ∧ y = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l2425_242510


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2425_242503

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions, prove that 2a_10 - a_12 = 24 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2425_242503


namespace NUMINAMATH_CALUDE_field_trip_students_l2425_242518

/-- Given a field trip with buses, prove the number of students. -/
theorem field_trip_students (seats_per_bus : ℕ) (num_buses : ℕ) : 
  seats_per_bus = 3 → num_buses = 3 → seats_per_bus * num_buses = 9 := by
  sorry

#check field_trip_students

end NUMINAMATH_CALUDE_field_trip_students_l2425_242518


namespace NUMINAMATH_CALUDE_lara_has_largest_result_l2425_242592

def starting_number : ℕ := 12

def john_result : ℕ := ((starting_number + 3) * 2) - 4
def lara_result : ℕ := (starting_number * 3 + 5) - 6
def miguel_result : ℕ := (starting_number * 2 - 2) + 2

theorem lara_has_largest_result :
  lara_result > john_result ∧ lara_result > miguel_result := by
  sorry

end NUMINAMATH_CALUDE_lara_has_largest_result_l2425_242592


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2425_242534

theorem circle_area_ratio (R : ℝ) (h : R > 0) : 
  let total_area := π * R^2
  let part_area := total_area / 8
  let shaded_area := 2 * part_area
  let unshaded_area := total_area - shaded_area
  shaded_area / unshaded_area = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2425_242534


namespace NUMINAMATH_CALUDE_range_of_a_l2425_242506

theorem range_of_a (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 3) 
  (square_condition : a^2 + 2*b^2 + 3*c^2 + 6*d^2 = 5) : 
  1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2425_242506


namespace NUMINAMATH_CALUDE_expected_value_twelve_sided_die_l2425_242547

/-- A twelve-sided die with faces numbered from 1 to 12 -/
def TwelveSidedDie : Finset ℕ := Finset.range 12

/-- The expected value of rolling a twelve-sided die -/
def expectedValue : ℚ :=
  (Finset.sum TwelveSidedDie (fun i => i + 1)) / 12

/-- Theorem: The expected value of rolling a twelve-sided die is 6.5 -/
theorem expected_value_twelve_sided_die :
  expectedValue = 13/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_twelve_sided_die_l2425_242547


namespace NUMINAMATH_CALUDE_equality_and_inequality_of_expressions_l2425_242550

variable (a : ℝ)

def f (n : ℕ) (x : ℝ) : ℝ := x ^ n

theorem equality_and_inequality_of_expressions (h : a ≠ 1) :
  (∀ n : ℕ, f n a = a ^ n) →
  ((f 11 (f 13 a)) ^ 14 = f 2002 a) ∧
  (f 11 (f 13 (f 14 a)) = f 2002 a) ∧
  ((f 11 a * f 13 a) ^ 14 ≠ f 2002 a) ∧
  (f 11 a * f 13 a * f 14 a ≠ f 2002 a) := by
  sorry

end NUMINAMATH_CALUDE_equality_and_inequality_of_expressions_l2425_242550
