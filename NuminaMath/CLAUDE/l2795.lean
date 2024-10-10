import Mathlib

namespace initial_time_is_six_hours_l2795_279546

/-- Proves that the initial time to cover 288 km is 6 hours -/
theorem initial_time_is_six_hours (distance : ℝ) (speed_new : ℝ) (time_factor : ℝ) :
  distance = 288 →
  speed_new = 32 →
  time_factor = 3 / 2 →
  ∃ (time_initial : ℝ),
    distance = speed_new * (time_factor * time_initial) ∧
    time_initial = 6 := by
  sorry


end initial_time_is_six_hours_l2795_279546


namespace simple_interest_problem_l2795_279515

/-- Given a principal P put at simple interest for 3 years, if increasing the interest rate by 2% 
    results in Rs. 360 more interest, then P = 6000. -/
theorem simple_interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 2) * 3) / 100 = (P * R * 3) / 100 + 360 → P = 6000 := by
  sorry

end simple_interest_problem_l2795_279515


namespace sine_cosine_relation_l2795_279561

theorem sine_cosine_relation (α : Real) (h : Real.sin (α + π/6) = 1/3) : 
  Real.cos (α - π/3) = 1/3 := by
sorry

end sine_cosine_relation_l2795_279561


namespace geometric_progression_special_ratio_l2795_279599

/-- A geometric progression with positive terms where any term is equal to the sum of the next two following terms has a common ratio of (√5 - 1)/2. -/
theorem geometric_progression_special_ratio (a : ℝ) (r : ℝ) :
  a > 0 →  -- First term is positive
  r > 0 →  -- Common ratio is positive
  (∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2)) →  -- Any term is sum of next two
  r = (Real.sqrt 5 - 1) / 2 := by
  sorry

end geometric_progression_special_ratio_l2795_279599


namespace maria_budget_excess_l2795_279544

theorem maria_budget_excess : 
  let sweater_price : ℚ := 35
  let scarf_price : ℚ := 25
  let mittens_price : ℚ := 15
  let hat_price : ℚ := 12
  let family_members : ℕ := 15
  let discount_threshold : ℚ := 800
  let discount_rate : ℚ := 0.1
  let sales_tax_rate : ℚ := 0.05
  let spending_limit : ℚ := 1500

  let set_price := 2 * sweater_price + scarf_price + mittens_price + hat_price
  let total_price := family_members * set_price
  let discounted_price := if total_price > discount_threshold 
                          then total_price * (1 - discount_rate) 
                          else total_price
  let final_price := discounted_price * (1 + sales_tax_rate)

  final_price - spending_limit = 229.35 := by sorry

end maria_budget_excess_l2795_279544


namespace relationship_abc_l2795_279576

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 4 then 4 / x + 1 else Real.log x / Real.log 2

theorem relationship_abc (a b c : ℝ) :
  (0 < a ∧ a < 4) →
  (b ≥ 4) →
  (f a = c) →
  (f b = c) →
  (deriv f b < 0) →
  b > a ∧ a > c :=
sorry

end relationship_abc_l2795_279576


namespace prime_power_difference_l2795_279598

theorem prime_power_difference (n : ℕ) (p : ℕ) (k : ℕ) 
  (h1 : n > 0) 
  (h2 : Nat.Prime p) 
  (h3 : 3^n - 2^n = p^k) : 
  Nat.Prime n := by
sorry

end prime_power_difference_l2795_279598


namespace matrix_product_abc_l2795_279514

def A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 3, -1; 0, 5, -4; -2, 5, 2]
def B : Matrix (Fin 3) (Fin 3) ℝ := !![3, -3, 0; 2, 1, -4; 5, 0, 1]
def C : Matrix (Fin 3) (Fin 2) ℝ := !![1, -1; 0, 2; 1, 0]

theorem matrix_product_abc :
  A * B * C = !![(-6 : ℝ), -13; -34, 20; -4, 8] := by sorry

end matrix_product_abc_l2795_279514


namespace locus_of_centers_l2795_279568

/-- Circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

/-- A circle is externally tangent to C₁ if the distance between their centers is the sum of their radii -/
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C₂ if the distance between their centers is the difference of their radii -/
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 2)^2 + b^2 = (3 - r)^2

/-- The locus of centers (a, b) of circles externally tangent to C₁ and internally tangent to C₂ -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) ↔ 
  84 * a^2 + 100 * b^2 - 64 * a - 64 = 0 :=
sorry

end locus_of_centers_l2795_279568


namespace jimmy_action_figures_sale_earnings_l2795_279590

theorem jimmy_action_figures_sale_earnings :
  let regular_figure_count : ℕ := 4
  let special_figure_count : ℕ := 1
  let regular_figure_value : ℕ := 15
  let special_figure_value : ℕ := 20
  let discount : ℕ := 5

  let regular_sale_price : ℕ := regular_figure_value - discount
  let special_sale_price : ℕ := special_figure_value - discount

  let total_earnings : ℕ := regular_figure_count * regular_sale_price + special_figure_count * special_sale_price

  total_earnings = 55 := by sorry

end jimmy_action_figures_sale_earnings_l2795_279590


namespace investment_interest_rate_calculation_l2795_279527

theorem investment_interest_rate_calculation 
  (total_investment : ℝ) 
  (known_rate : ℝ) 
  (unknown_investment : ℝ) 
  (income_difference : ℝ) :
  let known_investment := total_investment - unknown_investment
  let unknown_rate := (known_investment * known_rate - income_difference) / unknown_investment
  total_investment = 2000 ∧ 
  known_rate = 0.10 ∧ 
  unknown_investment = 800 ∧ 
  income_difference = 56 → 
  unknown_rate = 0.08 := by
sorry

end investment_interest_rate_calculation_l2795_279527


namespace final_result_l2795_279583

/-- The number of different five-digit even numbers that can be formed using the digits 0, 1, 2, 3, and 4 -/
def even_numbers : ℕ := 60

/-- The number of different five-digit numbers that can be formed using the digits 1, 2, 3, 4, and 5 such that 2 and 3 are not adjacent -/
def non_adjacent_23 : ℕ := 72

/-- The number of different five-digit numbers that can be formed using the digits 1, 2, 3, 4, and 5 such that the digits 1, 2, and 3 must be arranged in descending order -/
def descending_123 : ℕ := 20

/-- The final result is the sum of the three subproblems -/
theorem final_result : even_numbers + non_adjacent_23 + descending_123 = 152 := by
  sorry

end final_result_l2795_279583


namespace binomial_max_probability_l2795_279552

/-- The number of trials in the binomial distribution -/
def n : ℕ := 10

/-- The probability of success in each trial -/
def p : ℝ := 0.8

/-- The probability mass function of the binomial distribution -/
def binomialPMF (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The value of k that maximizes the binomial PMF -/
def kMax : ℕ := 8

theorem binomial_max_probability :
  ∀ k : ℕ, k ≠ kMax → binomialPMF k ≤ binomialPMF kMax :=
sorry

end binomial_max_probability_l2795_279552


namespace data_set_mode_l2795_279538

def data_set : List ℕ := [9, 7, 10, 8, 10, 9, 10]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem data_set_mode :
  mode data_set = 10 := by sorry

end data_set_mode_l2795_279538


namespace jogger_train_distance_l2795_279581

/-- Calculates the distance a jogger is ahead of a train's engine given their speeds and the time it takes for the train to pass the jogger. -/
theorem jogger_train_distance
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (passing_time : ℝ)
  (h1 : jogger_speed = 9 / 3.6)  -- Convert 9 km/hr to m/s
  (h2 : train_speed = 45 / 3.6)  -- Convert 45 km/hr to m/s
  (h3 : train_length = 120)
  (h4 : passing_time = 32) :
  train_speed * passing_time - jogger_speed * passing_time - train_length = 200 :=
by sorry

end jogger_train_distance_l2795_279581


namespace reflection_maps_points_l2795_279547

/-- Reflects a point across the line y = x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem reflection_maps_points :
  let A : ℝ × ℝ := (-3, 2)
  let B : ℝ × ℝ := (-2, 5)
  let A' : ℝ × ℝ := (2, -3)
  let B' : ℝ × ℝ := (5, -2)
  reflect_y_eq_x A = A' ∧ reflect_y_eq_x B = B' := by
  sorry


end reflection_maps_points_l2795_279547


namespace diamond_example_l2795_279517

/-- Diamond operation for real numbers -/
def diamond (a b : ℝ) : ℝ := (a + b) * (a - b) + a

/-- Theorem stating that 2 ◊ (3 ◊ 4) = -10 -/
theorem diamond_example : diamond 2 (diamond 3 4) = -10 := by
  sorry

end diamond_example_l2795_279517


namespace unique_solution_is_four_l2795_279586

/-- Function that returns the product of digits of a positive integer -/
def digit_product (n : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that 4 is the only positive integer solution to n^2 - 17n + 56 = a(n) -/
theorem unique_solution_is_four :
  ∃! (n : ℕ+), n^2 - 17*n + 56 = digit_product n :=
by sorry

end unique_solution_is_four_l2795_279586


namespace min_value_sum_reciprocals_l2795_279508

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0) (pos_e : e > 0) (pos_f : f > 0)
  (sum_eq_8 : a + b + c + d + e + f = 8) :
  (1 / a + 9 / b + 4 / c + 25 / d + 16 / e + 49 / f) ≥ 1352 :=
by sorry

end min_value_sum_reciprocals_l2795_279508


namespace root_sum_theorem_l2795_279532

theorem root_sum_theorem (a b c : ℝ) : 
  a^3 - 24*a^2 + 50*a - 14 = 0 →
  b^3 - 24*b^2 + 50*b - 14 = 0 →
  c^3 - 24*c^2 + 50*c - 14 = 0 →
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b) = 476/15 := by
sorry

end root_sum_theorem_l2795_279532


namespace sum_abc_equals_33_l2795_279565

theorem sum_abc_equals_33 
  (a b c N : ℕ+) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_eq1 : N = 5 * a + 3 * b + 5 * c)
  (h_eq2 : N = 4 * a + 5 * b + 4 * c)
  (h_range : 131 < N ∧ N < 150) :
  a + b + c = 33 := by
sorry

end sum_abc_equals_33_l2795_279565


namespace arc_length_from_sector_area_l2795_279596

/-- Given a circle with radius 5 cm and a sector with area 10 cm², 
    prove that the length of the arc forming the sector is 4 cm. -/
theorem arc_length_from_sector_area (r : ℝ) (area : ℝ) (arc_length : ℝ) : 
  r = 5 → 
  area = 10 → 
  area = (arc_length / (2 * r)) * r^2 → 
  arc_length = 4 := by
  sorry

end arc_length_from_sector_area_l2795_279596


namespace initial_boarders_l2795_279595

theorem initial_boarders (initial_ratio_boarders initial_ratio_day_scholars : ℕ)
  (new_ratio_boarders new_ratio_day_scholars : ℕ)
  (new_boarders : ℕ) :
  initial_ratio_boarders = 7 →
  initial_ratio_day_scholars = 16 →
  new_ratio_boarders = 1 →
  new_ratio_day_scholars = 2 →
  new_boarders = 80 →
  ∃ (x : ℕ),
    x * initial_ratio_boarders + new_boarders = x * initial_ratio_day_scholars * new_ratio_boarders / new_ratio_day_scholars →
    x * initial_ratio_boarders = 560 :=
by sorry

end initial_boarders_l2795_279595


namespace root_of_equation_l2795_279507

theorem root_of_equation (x : ℝ) : 
  (18 / (x^2 - 9) - 3 / (x - 3) = 2) ↔ (x = -4.5) :=
by sorry

end root_of_equation_l2795_279507


namespace smallest_lcm_with_gcd_five_l2795_279541

theorem smallest_lcm_with_gcd_five (a b : ℕ) : 
  1000 ≤ a ∧ a < 10000 ∧ 
  1000 ≤ b ∧ b < 10000 ∧ 
  Nat.gcd a b = 5 →
  201000 ≤ Nat.lcm a b :=
by sorry

end smallest_lcm_with_gcd_five_l2795_279541


namespace jellybeans_theorem_l2795_279511

def jellybeans_problem (initial_jellybeans : ℕ) (normal_class_size : ℕ) (sick_children : ℕ) (jellybeans_per_child : ℕ) : Prop :=
  let attending_children := normal_class_size - sick_children
  let eaten_jellybeans := attending_children * jellybeans_per_child
  let remaining_jellybeans := initial_jellybeans - eaten_jellybeans
  remaining_jellybeans = 34

theorem jellybeans_theorem :
  jellybeans_problem 100 24 2 3 := by
  sorry

end jellybeans_theorem_l2795_279511


namespace root_power_equality_l2795_279537

theorem root_power_equality (x : ℝ) (h : x > 0) :
  (x^((1:ℝ)/5)) / (x^((1:ℝ)/2)) = x^(-(3:ℝ)/10) := by sorry

end root_power_equality_l2795_279537


namespace clara_cookie_sales_l2795_279518

/-- Proves the number of boxes of the third type of cookies Clara sells -/
theorem clara_cookie_sales (cookies_per_box1 cookies_per_box2 cookies_per_box3 : ℕ)
  (boxes_sold1 boxes_sold2 : ℕ) (total_cookies : ℕ)
  (h1 : cookies_per_box1 = 12)
  (h2 : cookies_per_box2 = 20)
  (h3 : cookies_per_box3 = 16)
  (h4 : boxes_sold1 = 50)
  (h5 : boxes_sold2 = 80)
  (h6 : total_cookies = 3320)
  (h7 : total_cookies = cookies_per_box1 * boxes_sold1 + cookies_per_box2 * boxes_sold2 + cookies_per_box3 * boxes_sold3) :
  boxes_sold3 = 70 := by
  sorry

end clara_cookie_sales_l2795_279518


namespace equation_implication_l2795_279539

theorem equation_implication (x : ℝ) : 3 * x + 2 = 11 → 6 * x + 4 = 22 := by
  sorry

end equation_implication_l2795_279539


namespace unique_intersection_l2795_279525

/-- A function f(x) that represents a quadratic or linear equation depending on the value of a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 3) * x + 1

/-- Theorem stating that f(x) intersects the x-axis at only one point iff a = 0, 1, or 9 -/
theorem unique_intersection (a : ℝ) :
  (∃! x, f a x = 0) ↔ (a = 0 ∨ a = 1 ∨ a = 9) := by
  sorry

end unique_intersection_l2795_279525


namespace cubic_root_equation_solution_l2795_279545

theorem cubic_root_equation_solution :
  ∃ x : ℝ, (30 * x + (30 * x + 15) ^ (1/3)) ^ (1/3) = 15 ∧ x = 112 :=
by
  sorry

end cubic_root_equation_solution_l2795_279545


namespace function_value_theorem_l2795_279554

theorem function_value_theorem (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = 2 * x + 3) :
  f 1 = 3 := by sorry

end function_value_theorem_l2795_279554


namespace divisibility_by_6p_l2795_279543

theorem divisibility_by_6p (p : ℕ) (hp : Prime p) (hp2 : p > 2) :
  ∃ k : ℤ, 7^p - 5^p - 2 = 6 * p * k := by
  sorry

end divisibility_by_6p_l2795_279543


namespace simplify_fraction_l2795_279562

theorem simplify_fraction (a b : ℝ) (h1 : b ≠ 1/2) (h2 : b ≠ 1) :
  (2*a + 1) / (1 - b / (2*b - 1)) = (2*a + 1) * (2*b - 1) / (b - 1) := by
  sorry

end simplify_fraction_l2795_279562


namespace proportion_condition_l2795_279504

theorem proportion_condition (a b c : ℝ) (h : b ≠ 0 ∧ c ≠ 0) : 
  (∃ x y : ℝ, x / y = a / b ∧ y / x = b / c ∧ x^2 ≠ y * x) ∧
  (a / b = b / c → b^2 = a * c) ∧
  ¬(b^2 = a * c → a / b = b / c) :=
sorry

end proportion_condition_l2795_279504


namespace rectangle_area_theorem_l2795_279530

/-- A rectangle divided into four identical squares with a given perimeter -/
structure RectangleWithSquares where
  perimeter : ℝ
  square_side : ℝ
  perimeter_eq : perimeter = 8 * square_side

/-- The area of a rectangle divided into four identical squares -/
def area (rect : RectangleWithSquares) : ℝ :=
  4 * rect.square_side^2

/-- Theorem: A rectangle with perimeter 160 cm divided into four identical squares has an area of 1600 cm² -/
theorem rectangle_area_theorem (rect : RectangleWithSquares) (h : rect.perimeter = 160) :
  area rect = 1600 := by
  sorry

#check rectangle_area_theorem

end rectangle_area_theorem_l2795_279530


namespace trigonometric_equation_solution_l2795_279569

theorem trigonometric_equation_solution (x : Real) : 
  (8.456 * (Real.tan x)^2 * (Real.tan (3*x))^2 * Real.tan (4*x) = 
   (Real.tan x)^2 - (Real.tan (3*x))^2 + Real.tan (4*x)) ↔ 
  (∃ k : Int, x = k * Real.pi ∨ x = (Real.pi / 4) * (2 * k + 1)) :=
by sorry

end trigonometric_equation_solution_l2795_279569


namespace correct_algebraic_equation_l2795_279578

theorem correct_algebraic_equation (x y : ℝ) : 3 * x^2 * y - 2 * y * x^2 = x^2 * y := by
  sorry

end correct_algebraic_equation_l2795_279578


namespace complex_vector_properties_l2795_279501

open Complex

theorem complex_vector_properties (x y : ℝ) : 
  let z₁ : ℂ := (1 + I) / I
  let z₂ : ℂ := x + y * I
  true → 
  (∃ (k : ℝ), z₁.re * k = z₂.re ∧ z₁.im * k = z₂.im → x + y = 0) ∧
  (z₁.re * z₂.re + z₁.im * z₂.im = 0 → abs (z₁ + z₂) = abs (z₁ - z₂)) := by
  sorry

end complex_vector_properties_l2795_279501


namespace sum_two_condition_l2795_279588

theorem sum_two_condition (a b : ℝ) :
  (a = 1 ∧ b = 1 → a + b = 2) ∧
  (∃ a b : ℝ, a + b = 2 ∧ ¬(a = 1 ∧ b = 1)) :=
by sorry

end sum_two_condition_l2795_279588


namespace f_extremum_and_monotonicity_l2795_279556

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.log x + Real.log x / x

theorem f_extremum_and_monotonicity :
  (∀ x > 0, f (-1/2) x ≤ f (-1/2) 1) ∧ f (-1/2) 1 = 0 ∧
  (∀ a : ℝ, (∀ x > 0, ∀ y > 0, x < y → f a x < f a y) ↔ a ≥ 1 / (2 * Real.exp 2)) :=
sorry

end f_extremum_and_monotonicity_l2795_279556


namespace tan_negative_405_degrees_l2795_279577

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by
  sorry

end tan_negative_405_degrees_l2795_279577


namespace crimson_valley_skirts_l2795_279526

/-- The number of skirts in each valley -/
structure ValleySkirts where
  ember : ℕ
  azure : ℕ
  seafoam : ℕ
  purple : ℕ
  crimson : ℕ

/-- The conditions for the valley skirts problem -/
def valley_conditions (v : ValleySkirts) : Prop :=
  v.crimson = v.purple / 3 ∧
  v.purple = v.seafoam / 4 ∧
  v.seafoam = v.azure * 3 / 5 ∧
  v.azure = v.ember * 2 ∧
  v.ember = 120

/-- Theorem stating that given the conditions, Crimson Valley has 12 skirts -/
theorem crimson_valley_skirts (v : ValleySkirts) 
  (h : valley_conditions v) : v.crimson = 12 := by
  sorry

end crimson_valley_skirts_l2795_279526


namespace circle_diameter_from_area_l2795_279557

theorem circle_diameter_from_area (A : Real) (r : Real) (d : Real) : 
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end circle_diameter_from_area_l2795_279557


namespace number_exceeding_twelve_percent_l2795_279587

theorem number_exceeding_twelve_percent : ∃ x : ℝ, x = 0.12 * x + 52.8 ∧ x = 60 := by
  sorry

end number_exceeding_twelve_percent_l2795_279587


namespace equilateral_triangle_perimeter_equilateral_triangle_perimeter_alt_l2795_279510

/-- Given an equilateral triangle and an isosceles triangle sharing a side,
    prove that the perimeter of the equilateral triangle is 60 -/
theorem equilateral_triangle_perimeter
  (s : ℝ)  -- side length of the equilateral triangle
  (h1 : s > 0)  -- side length is positive
  (h2 : 2 * s + 5 = 45)  -- condition from isosceles triangle
  : 3 * s = 60 := by
  sorry

/-- Alternative formulation using more basic definitions -/
theorem equilateral_triangle_perimeter_alt
  (s : ℝ)  -- side length of the equilateral triangle
  (P_isosceles : ℝ)  -- perimeter of the isosceles triangle
  (b : ℝ)  -- base of the isosceles triangle
  (h1 : s > 0)  -- side length is positive
  (h2 : P_isosceles = 45)  -- given perimeter of isosceles triangle
  (h3 : b = 5)  -- given base of isosceles triangle
  (h4 : P_isosceles = 2 * s + b)  -- definition of isosceles triangle perimeter
  : 3 * s = 60 := by
  sorry

end equilateral_triangle_perimeter_equilateral_triangle_perimeter_alt_l2795_279510


namespace age_sum_problem_l2795_279593

theorem age_sum_problem (a b c : ℕ+) : 
  b = c → b < a → a * b * c = 144 → a + b + c = 22 := by sorry

end age_sum_problem_l2795_279593


namespace days_from_friday_l2795_279594

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def addDays (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | Nat.succ m => nextDay (addDays start m)

theorem days_from_friday :
  addDays DayOfWeek.Friday 72 = DayOfWeek.Sunday :=
by
  sorry

end days_from_friday_l2795_279594


namespace point_on_line_l2795_279548

/-- Given a line passing through points (3, -5) and (5, 1), 
    prove that any point (7, y) on this line must have y = 7. -/
theorem point_on_line (y : ℝ) : 
  (∀ (x : ℝ), (x - 3) * (1 - (-5)) = (y - (-5)) * (5 - 3) → x = 7) → y = 7 := by
  sorry

end point_on_line_l2795_279548


namespace min_r_for_perfect_square_l2795_279572

theorem min_r_for_perfect_square : 
  ∃ (r : ℕ), r > 0 ∧ 
  (∃ (n : ℕ), 4^3 + 4^r + 4^4 = n^2) ∧
  (∀ (s : ℕ), s > 0 ∧ s < r → ¬∃ (m : ℕ), 4^3 + 4^s + 4^4 = m^2) ∧
  r = 4 := by
sorry

end min_r_for_perfect_square_l2795_279572


namespace algebraic_identities_l2795_279564

variable (a b : ℝ)

theorem algebraic_identities :
  ((a - 2*b)^2 - (b - a)*(a + b) = 2*a^2 - 4*a*b + 3*b^2) ∧
  ((2*a - b)^2 * (2*a + b)^2 = 16*a^4 - 8*a^2*b^2 + b^4) := by
  sorry

end algebraic_identities_l2795_279564


namespace average_price_of_books_l2795_279503

/-- The average price of books bought by Rahim -/
theorem average_price_of_books (books_shop1 : ℕ) (price_shop1 : ℕ) 
  (books_shop2 : ℕ) (price_shop2 : ℕ) :
  books_shop1 = 40 →
  price_shop1 = 600 →
  books_shop2 = 20 →
  price_shop2 = 240 →
  (price_shop1 + price_shop2) / (books_shop1 + books_shop2) = 14 := by
  sorry

#check average_price_of_books

end average_price_of_books_l2795_279503


namespace monthly_income_problem_l2795_279549

/-- Given the average monthly incomes of three people, prove the income of one person -/
theorem monthly_income_problem (A B C : ℝ) 
  (h1 : (A + B) / 2 = 4050)
  (h2 : (B + C) / 2 = 5250)
  (h3 : (A + C) / 2 = 4200) :
  A = 3000 := by
  sorry

end monthly_income_problem_l2795_279549


namespace johnnys_jogging_speed_l2795_279555

/-- Proves that given the specified conditions, Johnny's jogging speed to school is approximately 9.333333333333334 miles per hour -/
theorem johnnys_jogging_speed 
  (total_time : ℝ) 
  (distance : ℝ) 
  (bus_speed : ℝ) 
  (h1 : total_time = 1) 
  (h2 : distance = 6.461538461538462) 
  (h3 : bus_speed = 21) : 
  ∃ (jogging_speed : ℝ), 
    (distance / jogging_speed + distance / bus_speed = total_time) ∧ 
    (abs (jogging_speed - 9.333333333333334) < 0.000001) := by
  sorry

end johnnys_jogging_speed_l2795_279555


namespace inequality_system_solution_range_l2795_279524

theorem inequality_system_solution_range (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 3 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x > 2*a - 3 ∧ 2*x ≥ 3*(x-2) + 5))) →
  (1/2 : ℝ) ≤ a ∧ a < 1 :=
by sorry

end inequality_system_solution_range_l2795_279524


namespace arithmetic_geometric_mean_difference_l2795_279520

theorem arithmetic_geometric_mean_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) / 2 = 2 * Real.sqrt 3 ∧ Real.sqrt (x * y) = Real.sqrt 3 → |x - y| = 6 :=
by sorry

end arithmetic_geometric_mean_difference_l2795_279520


namespace sqrt_equation_equivalence_l2795_279513

theorem sqrt_equation_equivalence (x : ℝ) (h : x > 6) :
  Real.sqrt (x - 6 * Real.sqrt (x - 6)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 6)) - 3 ↔ x ≥ 18 := by
  sorry

end sqrt_equation_equivalence_l2795_279513


namespace cube_sum_divisibility_l2795_279531

theorem cube_sum_divisibility (a b c : ℤ) 
  (h1 : 6 ∣ (a^2 + b^2 + c^2))
  (h2 : 3 ∣ (a*b + b*c + c*a)) :
  6 ∣ (a^3 + b^3 + c^3) := by
sorry

end cube_sum_divisibility_l2795_279531


namespace distinct_results_count_l2795_279563

/-- Represents the possible operators that can replace * in the expression -/
inductive Operator
| Add
| Sub
| Mul
| Div

/-- Represents the expression as a list of operators -/
def Expression := List Operator

/-- Evaluates an expression according to the given rules -/
def evaluate (expr : Expression) : ℚ :=
  sorry

/-- Generates all possible expressions -/
def allExpressions : List Expression :=
  sorry

/-- Counts the number of distinct results -/
def countDistinctResults (exprs : List Expression) : ℕ :=
  sorry

/-- The main theorem stating that the number of distinct results is 15 -/
theorem distinct_results_count :
  countDistinctResults allExpressions = 15 := by
  sorry

end distinct_results_count_l2795_279563


namespace trig_expression_equals_one_l2795_279506

theorem trig_expression_equals_one : 
  Real.sqrt 3 * Real.tan (30 * π / 180) * Real.cos (60 * π / 180) + Real.sin (45 * π / 180) ^ 2 = 1 := by
  sorry

end trig_expression_equals_one_l2795_279506


namespace sum_of_coefficients_l2795_279512

theorem sum_of_coefficients (k : ℝ) (h : k ≠ 0) : ∃ (a b c d : ℤ),
  (8 * k + 9 + 10 * k^2 - 3 * k^3) + (4 * k + 6 + k^2 + k^3) = 
  (a : ℝ) * k^3 + (b : ℝ) * k^2 + (c : ℝ) * k + (d : ℝ) ∧ 
  a + b + c + d = 36 := by
  sorry

end sum_of_coefficients_l2795_279512


namespace sin_plus_cos_equals_sqrt_a_plus_one_l2795_279567

theorem sin_plus_cos_equals_sqrt_a_plus_one (θ : Real) (a : Real) 
  (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sin (2 * θ) = a) : 
  Real.sin θ + Real.cos θ = Real.sqrt (a + 1) := by
  sorry

end sin_plus_cos_equals_sqrt_a_plus_one_l2795_279567


namespace calculate_expression_l2795_279522

theorem calculate_expression : (-5) / ((1 / 4) - (1 / 3)) * 12 = 720 := by
  sorry

end calculate_expression_l2795_279522


namespace sequence_term_correct_l2795_279519

def sequence_sum (n : ℕ) : ℕ := 2^n + 3

def sequence_term (n : ℕ) : ℕ :=
  match n with
  | 1 => 5
  | _ => 2^(n-1)

theorem sequence_term_correct :
  ∀ n : ℕ, n ≥ 1 → sequence_term n = 
    if n = 1 
    then sequence_sum 1
    else sequence_sum n - sequence_sum (n-1) :=
by sorry

end sequence_term_correct_l2795_279519


namespace exam_question_count_exam_question_count_proof_l2795_279540

theorem exam_question_count (marks_per_correct : ℕ) (marks_per_incorrect : ℕ) 
  (total_marks : ℕ) (correct_answers : ℕ) (total_questions : ℕ) : Prop :=
  (marks_per_correct = 4) →
  (marks_per_incorrect = 1) →
  (total_marks = 120) →
  (correct_answers = 40) →
  (marks_per_correct * correct_answers - marks_per_incorrect * (total_questions - correct_answers) = total_marks) →
  total_questions = 80

-- Proof
theorem exam_question_count_proof : 
  exam_question_count 4 1 120 40 80 := by sorry

end exam_question_count_exam_question_count_proof_l2795_279540


namespace f_symmetry_l2795_279571

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_symmetry (a b : ℝ) : f a b (-2) = 10 → f a b 2 = -26 := by
  sorry

end f_symmetry_l2795_279571


namespace area_is_24_l2795_279573

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := |3 * x| + |4 * y| = 12

/-- The graph is symmetric with respect to both x-axis and y-axis -/
axiom symmetry : ∀ x y : ℝ, equation x y ↔ equation (-x) y ∧ equation x (-y)

/-- The area enclosed by the graph -/
noncomputable def enclosed_area : ℝ := sorry

/-- Theorem stating that the enclosed area is 24 square units -/
theorem area_is_24 : enclosed_area = 24 :=
sorry

end area_is_24_l2795_279573


namespace cylinder_lateral_surface_area_l2795_279535

/-- The lateral surface area of a cylinder, given the diagonal length and intersection angle of its rectangular lateral surface. -/
theorem cylinder_lateral_surface_area 
  (d : ℝ) 
  (α : ℝ) 
  (h_d_pos : d > 0) 
  (h_α_pos : α > 0) 
  (h_α_lt_pi : α < π) : 
  ∃ (S : ℝ), S = (1/2) * d^2 * Real.sin α := by
  sorry

end cylinder_lateral_surface_area_l2795_279535


namespace prob_four_green_out_of_seven_l2795_279589

/-- The probability of drawing exactly 4 green marbles out of 7 draws, with replacement,
    from a bag containing 10 green marbles and 5 purple marbles. -/
theorem prob_four_green_out_of_seven (total_marbles : ℕ) (green_marbles : ℕ) (purple_marbles : ℕ)
  (h1 : total_marbles = green_marbles + purple_marbles)
  (h2 : green_marbles = 10)
  (h3 : purple_marbles = 5)
  (h4 : total_marbles > 0) :
  (Nat.choose 7 4 : ℚ) * (green_marbles / total_marbles : ℚ)^4 * (purple_marbles / total_marbles : ℚ)^3 =
  35 * (2/3 : ℚ)^4 * (1/3 : ℚ)^3 :=
by sorry

end prob_four_green_out_of_seven_l2795_279589


namespace gcd_g_x_l2795_279529

def g (x : ℤ) : ℤ := (5*x+3)*(8*x+2)*(11*x+7)*(4*x+11)

theorem gcd_g_x (x : ℤ) (h : ∃ k : ℤ, x = 17248 * k) : 
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 14 := by
  sorry

end gcd_g_x_l2795_279529


namespace solution_to_system_of_equations_l2795_279559

theorem solution_to_system_of_equations :
  let solutions : List (ℝ × ℝ) := [
    (Real.sqrt 5, Real.sqrt 6), (Real.sqrt 5, -Real.sqrt 6),
    (-Real.sqrt 5, Real.sqrt 6), (-Real.sqrt 5, -Real.sqrt 6),
    (Real.sqrt 6, Real.sqrt 5), (Real.sqrt 6, -Real.sqrt 5),
    (-Real.sqrt 6, Real.sqrt 5), (-Real.sqrt 6, -Real.sqrt 5)
  ]
  ∀ (x y : ℝ),
    (3 * x^2 + 3 * y^2 - x^2 * y^2 = 3 ∧
     x^4 + y^4 - x^2 * y^2 = 31) ↔
    (x, y) ∈ solutions := by
  sorry

end solution_to_system_of_equations_l2795_279559


namespace perimeter_ratio_of_similar_squares_l2795_279585

theorem perimeter_ratio_of_similar_squares (s : ℝ) (h : s > 0) : 
  let s1 := s * ((Real.sqrt 5 + 1) / 2)
  let p1 := 4 * s1
  let p2 := 4 * s
  let diagonal_first := Real.sqrt (2 * s1 ^ 2)
  diagonal_first = s → p1 / p2 = (Real.sqrt 5 + 1) / 2 := by
  sorry

end perimeter_ratio_of_similar_squares_l2795_279585


namespace permutation_cover_iff_m_gt_half_n_l2795_279575

/-- A permutation of the set {1, ..., n} -/
def Permutation (n : ℕ) := { f : Fin n → Fin n // Function.Bijective f }

/-- Two permutations have common points if they agree on at least one element -/
def have_common_points {n : ℕ} (f g : Permutation n) : Prop :=
  ∃ k : Fin n, f.val k = g.val k

/-- The main theorem: m permutations cover all permutations iff m > n/2 -/
theorem permutation_cover_iff_m_gt_half_n (n m : ℕ) :
  (∃ (fs : Fin m → Permutation n), ∀ f : Permutation n, ∃ i : Fin m, have_common_points f (fs i)) ↔
  m > n / 2 := by sorry

end permutation_cover_iff_m_gt_half_n_l2795_279575


namespace employee_pay_percentage_l2795_279553

/-- Given two employees X and Y with a total pay of 330 and Y's pay of 150,
    prove that X's pay as a percentage of Y's pay is 120%. -/
theorem employee_pay_percentage (total_pay : ℝ) (y_pay : ℝ) :
  total_pay = 330 →
  y_pay = 150 →
  (total_pay - y_pay) / y_pay * 100 = 120 := by
  sorry

end employee_pay_percentage_l2795_279553


namespace marias_first_stop_distance_l2795_279597

def total_distance : ℝ := 560

def distance_before_first_stop : ℝ → Prop := λ x =>
  let remaining_after_first := total_distance - x
  let second_stop_distance := (1/4) * remaining_after_first
  let final_leg := 210
  second_stop_distance + final_leg = remaining_after_first

theorem marias_first_stop_distance :
  ∃ x, distance_before_first_stop x ∧ x = 280 :=
sorry

end marias_first_stop_distance_l2795_279597


namespace zachary_pushups_l2795_279580

theorem zachary_pushups (david_pushups : ℕ) (difference : ℕ) (zachary_pushups : ℕ) :
  david_pushups = 37 →
  david_pushups = zachary_pushups + difference →
  difference = 30 →
  zachary_pushups = 7 :=
by
  sorry

end zachary_pushups_l2795_279580


namespace biased_coin_expected_value_l2795_279551

/-- The expected value of winnings for a biased coin flip -/
theorem biased_coin_expected_value :
  let p_heads : ℚ := 1/4  -- Probability of heads
  let p_tails : ℚ := 3/4  -- Probability of tails
  let win_heads : ℚ := 4  -- Amount won for heads
  let lose_tails : ℚ := 3 -- Amount lost for tails
  p_heads * win_heads - p_tails * lose_tails = -5/4 := by
  sorry

end biased_coin_expected_value_l2795_279551


namespace part1_range_of_m_part2_range_of_m_l2795_279550

-- Define the function f
def f (a m x : ℝ) : ℝ := x^3 + a*x^2 - a^2*x + m

-- Part 1
theorem part1_range_of_m :
  ∀ m : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f 1 m x = 0 ∧ f 1 m y = 0 ∧ f 1 m z = 0) →
  -1 < m ∧ m < 5/27 :=
sorry

-- Part 2
theorem part2_range_of_m :
  ∀ m : ℝ, (∀ a : ℝ, 3 ≤ a ∧ a ≤ 6 →
    ∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f a m x ≤ 1) →
  m ≤ -87 :=
sorry

end part1_range_of_m_part2_range_of_m_l2795_279550


namespace fraction_cube_equality_l2795_279521

theorem fraction_cube_equality : (64000 ^ 3 : ℚ) / (16000 ^ 3) = 64 := by
  sorry

end fraction_cube_equality_l2795_279521


namespace repeating_decimal_to_fraction_l2795_279592

/-- Proves that the repeating decimal 0.53207207207... is equal to 5316750/999900 -/
theorem repeating_decimal_to_fraction : 
  ∃ (x : ℚ), x = 0.53207207207 ∧ x = 5316750 / 999900 := by
  sorry

end repeating_decimal_to_fraction_l2795_279592


namespace concert_group_discount_l2795_279584

theorem concert_group_discount (P : ℝ) (h : P > 0) :
  ∃ (x : ℕ), 3 * P = (3 + x) * (0.75 * P) ∧ 3 + x = 4 := by
  sorry

end concert_group_discount_l2795_279584


namespace circle_radius_is_zero_l2795_279502

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 10*y + 41 = 0

/-- The radius of the circle -/
def circle_radius : ℝ := 0

/-- Theorem: The radius of the circle described by the given equation is 0 -/
theorem circle_radius_is_zero :
  ∀ x y : ℝ, circle_equation x y → ∃ c : ℝ × ℝ, ∀ p : ℝ × ℝ, circle_equation p.1 p.2 ↔ (p.1 - c.1)^2 + (p.2 - c.2)^2 = circle_radius^2 :=
sorry

end circle_radius_is_zero_l2795_279502


namespace no_prime_solution_l2795_279570

def base_p_to_decimal (digits : List Nat) (p : Nat) : Nat :=
  digits.foldl (fun acc d => acc * p + d) 0

theorem no_prime_solution :
  ¬∃ p : Nat, Prime p ∧
    (base_p_to_decimal [1, 0, 3, 2] p + 
     base_p_to_decimal [5, 0, 7] p + 
     base_p_to_decimal [2, 1, 4] p + 
     base_p_to_decimal [2, 0, 5] p + 
     base_p_to_decimal [1, 0] p = 
     base_p_to_decimal [4, 2, 3] p + 
     base_p_to_decimal [5, 4, 1] p + 
     base_p_to_decimal [6, 6, 0] p) :=
by sorry

end no_prime_solution_l2795_279570


namespace unique_quadratic_solution_l2795_279591

theorem unique_quadratic_solution (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b = 2*x ↔ x = 2) → 
  (a = -2 ∧ b = 4) := by
sorry

end unique_quadratic_solution_l2795_279591


namespace marta_textbook_expenses_l2795_279558

/-- The total amount Marta spent on textbooks -/
def total_spent (sale_price : ℕ) (sale_quantity : ℕ) (online_total : ℕ) (bookstore_multiplier : ℕ) : ℕ :=
  sale_price * sale_quantity + online_total + bookstore_multiplier * online_total

/-- Theorem stating the total amount Marta spent on textbooks -/
theorem marta_textbook_expenses : total_spent 10 5 40 3 = 210 := by
  sorry

end marta_textbook_expenses_l2795_279558


namespace angle_D_measure_l2795_279500

-- Define the hexagon and its angles
def Hexagon (A B C D E F : ℝ) : Prop :=
  -- Convexity condition (sum of angles = 720°)
  A + B + C + D + E + F = 720 ∧
  -- Angle congruence conditions
  A = B ∧ B = C ∧
  D = E ∧
  F = 2 * D ∧
  -- Relationship between angles A and D
  A + 30 = D

-- Theorem statement
theorem angle_D_measure (A B C D E F : ℝ) :
  Hexagon A B C D E F → D = 120 := by
  sorry

end angle_D_measure_l2795_279500


namespace rectangles_in_5x4_grid_l2795_279528

/-- Calculates the number of rectangles in a grid with sides along the grid lines -/
def count_rectangles (m n : ℕ) : ℕ :=
  let horizontal := (m * (m + 1) * (n + 1)) / 2
  let vertical := (n * (n + 1) * (m + 1)) / 2
  horizontal + vertical - (m * n)

/-- The theorem stating that a 5x4 grid contains 24 rectangles -/
theorem rectangles_in_5x4_grid :
  count_rectangles 5 4 = 24 := by
  sorry

end rectangles_in_5x4_grid_l2795_279528


namespace angelina_speed_l2795_279566

/-- Angelina's journey from home to gym via grocery store -/
def angelina_journey (v : ℝ) : Prop :=
  let time_home_to_grocery := 200 / v
  let time_grocery_to_gym := 300 / (2 * v)
  time_home_to_grocery = time_grocery_to_gym + 50

theorem angelina_speed : ∃ v : ℝ, angelina_journey v ∧ v > 0 ∧ 2 * v = 2 := by
  sorry

end angelina_speed_l2795_279566


namespace blithe_toy_collection_l2795_279579

/-- Given Blithe's toy collection changes, prove the initial number of toys. -/
theorem blithe_toy_collection (X : ℕ) : 
  X - 6 + 9 + 5 - 3 = 43 → X = 38 := by
  sorry

end blithe_toy_collection_l2795_279579


namespace exists_unresolved_conjecture_l2795_279536

/-- A structure representing a mathematical conjecture -/
structure Conjecture where
  statement : Prop
  is_proven : Prop
  is_disproven : Prop

/-- A predicate that determines if a conjecture is unresolved -/
def is_unresolved (c : Conjecture) : Prop :=
  ¬c.is_proven ∧ ¬c.is_disproven

/-- There exists at least one unresolved conjecture in mathematics -/
theorem exists_unresolved_conjecture : ∃ c : Conjecture, is_unresolved c := by
  sorry

#check exists_unresolved_conjecture

end exists_unresolved_conjecture_l2795_279536


namespace min_value_sum_reciprocals_l2795_279505

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (sum_eq : p + q + r + s + t + u = 8) : 
  1/p + 9/q + 16/r + 25/s + 36/t + 49/u ≥ 84.5 ∧ 
  ∃ (p' q' r' s' t' u' : ℝ),
    p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧
    p' + q' + r' + s' + t' + u' = 8 ∧
    1/p' + 9/q' + 16/r' + 25/s' + 36/t' + 49/u' = 84.5 := by
  sorry

end min_value_sum_reciprocals_l2795_279505


namespace largest_in_systematic_sample_l2795_279560

/-- Represents a systematic sample --/
structure SystematicSample where
  total : Nat
  start : Nat
  interval : Nat

/-- Checks if a number is in the systematic sample --/
def inSample (s : SystematicSample) (n : Nat) : Prop :=
  ∃ k : Nat, n = s.start + k * s.interval ∧ n ≤ s.total

/-- The largest number in the sample --/
def largestInSample (s : SystematicSample) : Nat :=
  s.start + ((s.total - s.start) / s.interval) * s.interval

theorem largest_in_systematic_sample
  (employees : Nat)
  (first : Nat)
  (second : Nat)
  (h1 : employees = 500)
  (h2 : first = 6)
  (h3 : second = 31)
  (h4 : second - first = 31 - 6) :
  let s := SystematicSample.mk employees first (second - first)
  largestInSample s = 481 :=
by
  sorry

#check largest_in_systematic_sample

end largest_in_systematic_sample_l2795_279560


namespace inscribed_squares_ratio_l2795_279516

theorem inscribed_squares_ratio : ∀ x y : ℝ,
  (5 : ℝ) ^ 2 + 12 ^ 2 = 13 ^ 2 →
  (12 - x) / 12 = x / 5 →
  y + 2 * (5 * y / 13) = 13 →
  x / y = 1380 / 2873 := by
sorry

end inscribed_squares_ratio_l2795_279516


namespace projects_equal_volume_projects_equal_days_l2795_279574

/-- Represents the dimensions of an excavation project -/
structure ProjectDimensions where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of earth to be dug given project dimensions -/
def calculateVolume (dimensions : ProjectDimensions) : ℝ :=
  dimensions.depth * dimensions.length * dimensions.breadth

/-- The dimensions of Project 1 -/
def project1 : ProjectDimensions := {
  depth := 100,
  length := 25,
  breadth := 30
}

/-- The dimensions of Project 2 -/
def project2 : ProjectDimensions := {
  depth := 75,
  length := 20,
  breadth := 50
}

/-- Theorem stating that the volumes of both projects are equal -/
theorem projects_equal_volume : calculateVolume project1 = calculateVolume project2 := by
  sorry

/-- Corollary stating that the number of days required for both projects is the same -/
theorem projects_equal_days (days1 days2 : ℕ) 
    (h : calculateVolume project1 = calculateVolume project2) : days1 = days2 := by
  sorry

end projects_equal_volume_projects_equal_days_l2795_279574


namespace f_properties_l2795_279533

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x - 15

-- Define the theorem
theorem f_properties (a : ℝ) (x : ℝ) (h : |x - a| < 1) :
  (∃ (y : ℝ), |f y| > 5 ↔ (y < -4 ∨ y > 5 ∨ ((1 - Real.sqrt 41) / 2 < y ∧ y < (1 + Real.sqrt 41) / 2))) ∧
  |f x - f a| < 2 * (|a| + 1) :=
sorry

end f_properties_l2795_279533


namespace perpendicular_line_equation_line_through_point_with_segment_l2795_279542

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 = 0
def l₂ (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 3 = 0

-- Define the perpendicular line n
def n (x y : ℝ) : Prop := y = -(Real.sqrt 3 / 3) * x + 2 ∨ y = -(Real.sqrt 3 / 3) * x - 2

-- Define the line m
def m (x y : ℝ) : Prop := x = Real.sqrt 3 ∨ y = (Real.sqrt 3 / 3) * x + 3

-- Theorem for part (1)
theorem perpendicular_line_equation
  (h_parallel : ∀ x y, l₁ x y ↔ l₂ x y)
  (h_perp : ∀ x y, n x y → (∀ x' y', l₁ x' y' → (y - y') = (Real.sqrt 3 / 3) * (x - x')))
  (h_area : ∃ a b, n a 0 ∧ n 0 b ∧ a * b / 2 = 2 * Real.sqrt 3) :
  ∀ x y, n x y :=
sorry

-- Theorem for part (2)
theorem line_through_point_with_segment
  (h_parallel : ∀ x y, l₁ x y ↔ l₂ x y)
  (h_point : m (Real.sqrt 3) 4)
  (h_segment : ∃ x₁ y₁ x₂ y₂,
    m x₁ y₁ ∧ m x₂ y₂ ∧ l₁ x₁ y₁ ∧ l₂ x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 2) :
  ∀ x y, m x y :=
sorry

end perpendicular_line_equation_line_through_point_with_segment_l2795_279542


namespace circle_radius_l2795_279582

theorem circle_radius (A C : ℝ) (h : A / C = 15) : 
  ∃ r : ℝ, r > 0 ∧ A = π * r^2 ∧ C = 2 * π * r ∧ r = 30 := by
sorry

end circle_radius_l2795_279582


namespace athlete_heartbeats_l2795_279523

/-- The number of heartbeats during a race --/
def heartbeats_during_race (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Proof that the athlete's heart beats 19200 times during the race --/
theorem athlete_heartbeats :
  heartbeats_during_race 160 20 6 = 19200 := by
  sorry

#eval heartbeats_during_race 160 20 6

end athlete_heartbeats_l2795_279523


namespace derivative_reciprocal_l2795_279534

theorem derivative_reciprocal (x : ℝ) (hx : x ≠ 0) :
  deriv (fun x => 1 / x) x = -(1 / x^2) := by
  sorry

end derivative_reciprocal_l2795_279534


namespace pencil_box_problem_l2795_279509

structure BoxOfPencils where
  blue : ℕ
  green : ℕ

def Vasya (box : BoxOfPencils) : Prop := box.blue ≥ 4
def Kolya (box : BoxOfPencils) : Prop := box.green ≥ 5
def Petya (box : BoxOfPencils) : Prop := box.blue ≥ 3 ∧ box.green ≥ 4
def Misha (box : BoxOfPencils) : Prop := box.blue ≥ 4 ∧ box.green ≥ 4

theorem pencil_box_problem :
  ∃ (box : BoxOfPencils),
    (Vasya box ∧ ¬Kolya box ∧ Petya box ∧ Misha box) ∧
    ¬∃ (other_box : BoxOfPencils),
      ((¬Vasya other_box ∧ Kolya other_box ∧ Petya other_box ∧ Misha other_box) ∨
       (Vasya other_box ∧ Kolya other_box ∧ ¬Petya other_box ∧ Misha other_box) ∨
       (Vasya other_box ∧ Kolya other_box ∧ Petya other_box ∧ ¬Misha other_box)) :=
by
  sorry

#check pencil_box_problem

end pencil_box_problem_l2795_279509
