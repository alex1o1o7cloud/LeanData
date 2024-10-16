import Mathlib

namespace NUMINAMATH_CALUDE_lego_set_cost_l1978_197873

/-- Represents the sale of toys with given conditions and calculates the cost of a Lego set --/
def toy_sale (total_after_tax : ℚ) (car_price : ℚ) (car_discount : ℚ) (num_cars : ℕ) 
             (num_action_figures : ℕ) (tax_rate : ℚ) : ℚ :=
  let discounted_car_price := car_price * (1 - car_discount)
  let action_figure_price := 2 * discounted_car_price
  let board_game_price := action_figure_price + discounted_car_price
  let known_items_total := num_cars * discounted_car_price + 
                           num_action_figures * action_figure_price + 
                           board_game_price
  let total_before_tax := total_after_tax / (1 + tax_rate)
  total_before_tax - known_items_total

/-- Theorem stating that the Lego set costs $85 before tax --/
theorem lego_set_cost : 
  toy_sale 136.5 5 0.1 3 2 0.05 = 85 := by
  sorry

end NUMINAMATH_CALUDE_lego_set_cost_l1978_197873


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1978_197855

theorem complex_fraction_simplification :
  (2 / 5 + 3 / 4) / (4 / 9 + 1 / 6) = 207 / 110 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1978_197855


namespace NUMINAMATH_CALUDE_product_closest_to_63_l1978_197853

theorem product_closest_to_63 : 
  let product := 2.1 * (30.3 + 0.13)
  ∀ x ∈ ({55, 60, 63, 65, 70} : Set ℝ), 
    x ≠ 63 → |product - 63| < |product - x| := by
  sorry

end NUMINAMATH_CALUDE_product_closest_to_63_l1978_197853


namespace NUMINAMATH_CALUDE_S_bounds_l1978_197898

def S : Set ℝ := { y | ∃ x : ℝ, x ≥ 0 ∧ y = (2 * x + 3) / (x + 2) }

theorem S_bounds :
  ∃ (m M : ℝ),
    (∀ y ∈ S, m ≤ y) ∧
    (∀ y ∈ S, y ≤ M) ∧
    m ∈ S ∧
    M ∉ S ∧
    m = 3/2 ∧
    M = 2 := by
  sorry


end NUMINAMATH_CALUDE_S_bounds_l1978_197898


namespace NUMINAMATH_CALUDE_profit_share_difference_example_l1978_197872

/-- Given a total profit and a ratio of profit division between two parties,
    calculate the difference between their profit shares. -/
def profit_share_difference (total_profit : ℚ) (ratio_x : ℚ) (ratio_y : ℚ) : ℚ :=
  let total_ratio := ratio_x + ratio_y
  let share_x := (ratio_x / total_ratio) * total_profit
  let share_y := (ratio_y / total_ratio) * total_profit
  share_x - share_y

/-- Theorem stating that for a total profit of 800 and a profit division ratio of 1/2 : 1/3,
    the difference between the profit shares is 160. -/
theorem profit_share_difference_example :
  profit_share_difference 800 (1/2) (1/3) = 160 := by
  sorry


end NUMINAMATH_CALUDE_profit_share_difference_example_l1978_197872


namespace NUMINAMATH_CALUDE_parentheses_removal_correct_l1978_197849

theorem parentheses_removal_correct (a b : ℤ) : -2*a + 3*(b - 1) = -2*a + 3*b - 3 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_correct_l1978_197849


namespace NUMINAMATH_CALUDE_function_behavior_implies_a_range_l1978_197845

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + (a-1)*x + 1

/-- The derivative of f(x) with respect to x -/
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + (a-1)

theorem function_behavior_implies_a_range :
  ∀ a : ℝ,
  (∀ x ∈ Set.Ioo 1 4, (f_prime a x) < 0) →
  (∀ x ∈ Set.Ioi 6, (f_prime a x) > 0) →
  5 ≤ a ∧ a ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_function_behavior_implies_a_range_l1978_197845


namespace NUMINAMATH_CALUDE_line_point_k_value_l1978_197870

/-- A line contains the points (2, 4), (7, k), and (15, 8). The value of k is 72/13. -/
theorem line_point_k_value : ∀ (k : ℚ), 
  (∃ (m b : ℚ), 
    (4 : ℚ) = m * 2 + b ∧ 
    k = m * 7 + b ∧ 
    (8 : ℚ) = m * 15 + b) → 
  k = 72 / 13 := by
sorry

end NUMINAMATH_CALUDE_line_point_k_value_l1978_197870


namespace NUMINAMATH_CALUDE_isosceles_triangle_properties_l1978_197813

/-- An isosceles triangle with perimeter 6 cm -/
structure IsoscelesTriangle where
  /-- Length of each leg in cm -/
  x : ℝ
  /-- Length of the base in cm -/
  y : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : y = 6 - 2 * x
  /-- The perimeter is 6 cm -/
  perimeter : 2 * x + y = 6
  /-- All sides are positive -/
  positivity : 0 < x ∧ 0 < y

/-- Theorem about the base length and leg length of the isosceles triangle -/
theorem isosceles_triangle_properties (t : IsoscelesTriangle) :
  t.y = 6 - 2 * t.x ∧ 3/2 < t.x ∧ t.x < 3 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_properties_l1978_197813


namespace NUMINAMATH_CALUDE_intersection_points_correct_l1978_197851

/-- Parallelogram with given dimensions divided into three equal areas -/
structure EqualAreaParallelogram where
  AB : ℝ
  AD : ℝ
  BE : ℝ
  h_AB : AB = 153
  h_AD : AD = 180
  h_BE : BE = 135

/-- The points where perpendicular lines intersect AD -/
def intersection_points (p : EqualAreaParallelogram) : ℝ × ℝ :=
  (96, 156)

/-- Theorem stating that the intersection points are correct -/
theorem intersection_points_correct (p : EqualAreaParallelogram) :
  intersection_points p = (96, 156) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_correct_l1978_197851


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l1978_197830

/-- The point corresponding to (1+3i)(3-i) is located in the first quadrant of the complex plane. -/
theorem point_in_first_quadrant : 
  let z : ℂ := (1 + 3*I) * (3 - I)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l1978_197830


namespace NUMINAMATH_CALUDE_owl_cost_in_gold_harry_owl_cost_l1978_197893

/-- Calculates the cost of an owl given the total cost and the cost of other items. -/
theorem owl_cost_in_gold (spellbook_cost : ℕ) (spellbook_count : ℕ) 
  (potion_kit_cost : ℕ) (potion_kit_count : ℕ) (silver_per_gold : ℕ) (total_cost_silver : ℕ) : ℕ :=
  let spellbook_total_cost := spellbook_cost * spellbook_count * silver_per_gold
  let potion_kit_total_cost := potion_kit_cost * potion_kit_count
  let other_items_cost := spellbook_total_cost + potion_kit_total_cost
  let owl_cost_silver := total_cost_silver - other_items_cost
  owl_cost_silver / silver_per_gold

/-- Proves that the owl costs 28 gold given the specific conditions in Harry's purchase. -/
theorem harry_owl_cost : 
  owl_cost_in_gold 5 5 20 3 9 537 = 28 := by
  sorry

end NUMINAMATH_CALUDE_owl_cost_in_gold_harry_owl_cost_l1978_197893


namespace NUMINAMATH_CALUDE_paper_I_max_mark_l1978_197863

/-- Represents a test with a maximum mark and a passing percentage -/
structure Test where
  maxMark : ℕ
  passingPercentage : ℚ

/-- Calculates the passing mark for a given test -/
def passingMark (test : Test) : ℚ :=
  test.passingPercentage * test.maxMark

theorem paper_I_max_mark :
  ∃ (test : Test),
    test.passingPercentage = 42 / 100 ∧
    passingMark test = 42 + 22 ∧
    test.maxMark = 152 := by
  sorry

end NUMINAMATH_CALUDE_paper_I_max_mark_l1978_197863


namespace NUMINAMATH_CALUDE_exam_probability_l1978_197850

/-- The probability of passing the exam -/
def prob_pass : ℚ := 4/7

/-- The probability of not passing the exam -/
def prob_not_pass : ℚ := 1 - prob_pass

theorem exam_probability : prob_not_pass = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_exam_probability_l1978_197850


namespace NUMINAMATH_CALUDE_age_ratio_dan_james_l1978_197816

theorem age_ratio_dan_james : 
  ∀ (dan_future_age james_age : ℕ),
    dan_future_age = 28 →
    james_age = 20 →
    ∃ (dan_age : ℕ),
      dan_age + 4 = dan_future_age ∧
      dan_age * 5 = james_age * 6 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_dan_james_l1978_197816


namespace NUMINAMATH_CALUDE_sum_of_square_and_two_cubes_sum_of_square_and_three_cubes_l1978_197854

-- Part (a)
theorem sum_of_square_and_two_cubes (k : ℤ) :
  ∃ (a b c : ℤ), 3 * k - 2 = a^2 + b^3 + c^3 := by sorry

-- Part (b)
theorem sum_of_square_and_three_cubes (n : ℤ) :
  ∃ (w x y z : ℤ), n = w^2 + x^3 + y^3 + z^3 := by sorry

end NUMINAMATH_CALUDE_sum_of_square_and_two_cubes_sum_of_square_and_three_cubes_l1978_197854


namespace NUMINAMATH_CALUDE_stratified_sampling_total_size_l1978_197822

theorem stratified_sampling_total_size 
  (district1_ratio : ℚ) 
  (district2_ratio : ℚ) 
  (district3_ratio : ℚ) 
  (largest_district_sample : ℕ) : 
  district1_ratio + district2_ratio + district3_ratio = 1 →
  district3_ratio > district1_ratio →
  district3_ratio > district2_ratio →
  district3_ratio = 1/2 →
  largest_district_sample = 60 →
  2 * largest_district_sample = 120 :=
by
  sorry

#check stratified_sampling_total_size

end NUMINAMATH_CALUDE_stratified_sampling_total_size_l1978_197822


namespace NUMINAMATH_CALUDE_zeros_order_l1978_197852

noncomputable def f (x : ℝ) := Real.exp x + x
noncomputable def g (x : ℝ) := Real.log x + x
noncomputable def h (x : ℝ) := Real.log x - 1

theorem zeros_order (a b c : ℝ) 
  (ha : f a = 0) 
  (hb : g b = 0) 
  (hc : h c = 0) : 
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_zeros_order_l1978_197852


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1978_197877

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 3 - 4 * Complex.I → z = -4 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1978_197877


namespace NUMINAMATH_CALUDE_fifteenth_thirty_seventh_215th_digit_l1978_197858

def decimal_representation (n d : ℕ) : List ℕ := sorry

def nth_digit (n : ℕ) (l : List ℕ) : ℕ := sorry

theorem fifteenth_thirty_seventh_215th_digit :
  let rep := decimal_representation 15 37
  nth_digit 215 rep = 0 := by sorry

end NUMINAMATH_CALUDE_fifteenth_thirty_seventh_215th_digit_l1978_197858


namespace NUMINAMATH_CALUDE_state_tax_calculation_l1978_197895

/-- Calculate the state tax for a partial-year resident -/
theorem state_tax_calculation 
  (months_resident : ℕ) 
  (taxable_income : ℝ) 
  (tax_rate : ℝ) : 
  months_resident = 9 → 
  taxable_income = 42500 → 
  tax_rate = 0.04 → 
  (months_resident : ℝ) / 12 * taxable_income * tax_rate = 1275 := by
  sorry

end NUMINAMATH_CALUDE_state_tax_calculation_l1978_197895


namespace NUMINAMATH_CALUDE_triangle_side_length_theorem_l1978_197886

/-- Represents a triangle with two known side lengths and one unknown side length. -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Checks if the given lengths can form a valid triangle. -/
def is_valid_triangle (t : Triangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- The theorem stating that 12 is a possible length for the third side of the triangle,
    while 4, 5, and 13 are not. -/
theorem triangle_side_length_theorem :
  ∃ (t : Triangle), t.side1 = 4 ∧ t.side2 = 9 ∧ t.side3 = 12 ∧ is_valid_triangle t ∧
  (∀ (t' : Triangle), t'.side1 = 4 ∧ t'.side2 = 9 ∧ (t'.side3 = 4 ∨ t'.side3 = 5 ∨ t'.side3 = 13) → ¬is_valid_triangle t') :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_theorem_l1978_197886


namespace NUMINAMATH_CALUDE_simplify_expression_l1978_197814

theorem simplify_expression (x : ℝ) : 7*x + 9 - 2*x + 15 = 5*x + 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1978_197814


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1978_197812

theorem quadratic_rewrite (d e f : ℤ) : 
  (∀ x, 25 * x^2 - 40 * x - 75 = (d * x + e)^2 + f) → d * e = -20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1978_197812


namespace NUMINAMATH_CALUDE_absolute_value_inequality_implies_m_equals_negative_four_l1978_197843

theorem absolute_value_inequality_implies_m_equals_negative_four (m : ℝ) :
  (∀ x : ℝ, |2*x - m| ≤ |3*x + 6|) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_implies_m_equals_negative_four_l1978_197843


namespace NUMINAMATH_CALUDE_exponent_division_l1978_197860

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^6 / a^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1978_197860


namespace NUMINAMATH_CALUDE_isabella_currency_exchange_l1978_197890

theorem isabella_currency_exchange (d : ℕ) : 
  (11 * d / 8 : ℚ) - 80 = d →
  (d / 100 + (d / 10) % 10 + d % 10 : ℕ) = 6 :=
by sorry

end NUMINAMATH_CALUDE_isabella_currency_exchange_l1978_197890


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l1978_197882

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^3 + 1/a^3 = 2 * Real.sqrt 5 ∨ a^3 + 1/a^3 = -2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l1978_197882


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l1978_197832

theorem smallest_addition_for_divisibility (n : ℕ) (h : n = 8261955) :
  ∃ x : ℕ, x = 2 ∧ 
  (∀ y : ℕ, y < x → ¬(11 ∣ (n + y))) ∧
  (11 ∣ (n + x)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l1978_197832


namespace NUMINAMATH_CALUDE_distance_equals_abs_l1978_197817

theorem distance_equals_abs (x : ℝ) : |x - 0| = |x| := by
  sorry

end NUMINAMATH_CALUDE_distance_equals_abs_l1978_197817


namespace NUMINAMATH_CALUDE_rectangle_length_width_difference_l1978_197844

theorem rectangle_length_width_difference
  (perimeter : ℝ)
  (diagonal : ℝ)
  (h_perimeter : perimeter = 80)
  (h_diagonal : diagonal = 20 * Real.sqrt 2) :
  ∃ (length width : ℝ),
    length > 0 ∧ width > 0 ∧
    2 * (length + width) = perimeter ∧
    length^2 + width^2 = diagonal^2 ∧
    length - width = 0 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_width_difference_l1978_197844


namespace NUMINAMATH_CALUDE_twenty_nine_free_travelers_l1978_197880

/-- Represents the promotion scenario for a travel agency -/
structure TravelPromotion where
  /-- Number of tourists who came on their own -/
  self_arrivals : ℕ
  /-- Number of tourists who didn't bring anyone -/
  no_referrals : ℕ
  /-- Total number of tourists -/
  total_tourists : ℕ

/-- Calculates the number of tourists who traveled for free -/
def free_travelers (promo : TravelPromotion) : ℕ :=
  (promo.total_tourists - promo.self_arrivals - promo.no_referrals) / 4

/-- Theorem stating that 29 tourists traveled for free -/
theorem twenty_nine_free_travelers (promo : TravelPromotion)
  (h1 : promo.self_arrivals = 13)
  (h2 : promo.no_referrals = 100)
  (h3 : promo.total_tourists = promo.self_arrivals + promo.no_referrals + 4 * (free_travelers promo)) :
  free_travelers promo = 29 := by
  sorry

#eval free_travelers { self_arrivals := 13, no_referrals := 100, total_tourists := 229 }

end NUMINAMATH_CALUDE_twenty_nine_free_travelers_l1978_197880


namespace NUMINAMATH_CALUDE_cone_surface_area_l1978_197869

/-- A cone with slant height 2 and lateral surface unfolding into a semicircle has surface area 3π -/
theorem cone_surface_area (h : ℝ) (r : ℝ) : 
  h = 2 → -- slant height is 2
  2 * π * r = 2 * π → -- lateral surface unfolds into a semicircle (circumference of base equals arc length of semicircle)
  π * r * (r + h) = 3 * π := by
  sorry


end NUMINAMATH_CALUDE_cone_surface_area_l1978_197869


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1978_197897

theorem polynomial_divisibility (p q : ℚ) : 
  (∀ x : ℚ, (x + 3) * (x - 2) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x - 8)) →
  p = -67/3 ∧ q = -158/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1978_197897


namespace NUMINAMATH_CALUDE_product_of_roots_implies_k_l1978_197828

-- Define the polynomial P(X)
def P (k X : ℝ) : ℝ := X^4 - 18*X^3 + k*X^2 + 200*X - 1984

-- Define the theorem
theorem product_of_roots_implies_k (k : ℝ) :
  (∃ a b c d : ℝ, 
    P k a = 0 ∧ P k b = 0 ∧ P k c = 0 ∧ P k d = 0 ∧
    ((a * b = -32) ∨ (a * c = -32) ∨ (a * d = -32) ∨ 
     (b * c = -32) ∨ (b * d = -32) ∨ (c * d = -32))) →
  k = 86 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_implies_k_l1978_197828


namespace NUMINAMATH_CALUDE_violet_necklace_problem_l1978_197838

theorem violet_necklace_problem (x : ℝ) 
  (h1 : (1/2 : ℝ) * x + 30 = (3/4 : ℝ) * x) : 
  (1/4 : ℝ) * x = 30 := by
  sorry

end NUMINAMATH_CALUDE_violet_necklace_problem_l1978_197838


namespace NUMINAMATH_CALUDE_committee_selection_count_l1978_197892

/-- The number of ways to choose a committee from a club -/
def choose_committee (n : ℕ) (r : ℕ) : ℕ := Nat.choose n r

/-- The size of the club -/
def club_size : ℕ := 10

/-- The size of the committee -/
def committee_size : ℕ := 5

/-- Theorem: The number of ways to choose a 5-person committee from a club of 10 people is 252 -/
theorem committee_selection_count : 
  choose_committee club_size committee_size = 252 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_count_l1978_197892


namespace NUMINAMATH_CALUDE_root_difference_zero_l1978_197847

theorem root_difference_zero : ∃ (r : ℝ), 
  (∀ x : ℝ, x^2 + 20*x + 75 = -25 ↔ x = r) ∧ 
  (abs (r - r) = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_difference_zero_l1978_197847


namespace NUMINAMATH_CALUDE_path_count_theorem_l1978_197815

/-- Represents a grid with width and height -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Calculates the number of paths in a grid with the given constraints -/
def count_paths (g : Grid) : ℕ :=
  Nat.choose (g.width + g.height - 1) g.height -
  Nat.choose (g.width + g.height - 2) g.height +
  Nat.choose (g.width + g.height - 3) g.height

/-- The problem statement -/
theorem path_count_theorem (g : Grid) (h1 : g.width = 7) (h2 : g.height = 6) :
  count_paths g = 1254 := by
  sorry

end NUMINAMATH_CALUDE_path_count_theorem_l1978_197815


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_xy_8th_power_l1978_197881

theorem coefficient_x3y5_in_xy_8th_power :
  let n : ℕ := 8
  let k : ℕ := 5
  Nat.choose n k = 56 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_xy_8th_power_l1978_197881


namespace NUMINAMATH_CALUDE_f_properties_l1978_197807

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin (2*x) + Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3 / 2

theorem f_properties :
  (∃ (t : ℝ), t > 0 ∧ ∀ (x : ℝ), f (x + t) = f x ∧ ∀ (s : ℝ), s > 0 ∧ (∀ (x : ℝ), f (x + s) = f x) → t ≤ s) ∧
  (∀ (x : ℝ), x ≥ -π/12 ∧ x ≤ 5*π/12 → f x ≥ -1/2 ∧ f x ≤ 1) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ≥ -π/12 ∧ x₁ ≤ 5*π/12 ∧ x₂ ≥ -π/12 ∧ x₂ ≤ 5*π/12 ∧ f x₁ = -1/2 ∧ f x₂ = 1) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l1978_197807


namespace NUMINAMATH_CALUDE_pokemon_cards_remaining_l1978_197856

theorem pokemon_cards_remaining (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  initial = 13 → given_away = 9 → remaining = initial - given_away → remaining = 4 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_cards_remaining_l1978_197856


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l1978_197821

/-- An isosceles right triangle with perimeter 4 + 4√2 has a hypotenuse of length 4. -/
theorem isosceles_right_triangle_hypotenuse (a c : ℝ) : 
  a > 0 → -- Side length is positive
  c > 0 → -- Hypotenuse length is positive
  2 * a + c = 4 + 4 * Real.sqrt 2 → -- Perimeter condition
  c = a * Real.sqrt 2 → -- Isosceles right triangle condition
  c = 4 := by
sorry


end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l1978_197821


namespace NUMINAMATH_CALUDE_number_operation_result_l1978_197842

theorem number_operation_result : 
  let n : ℚ := 55
  (n / 5 + 10) = 21 := by sorry

end NUMINAMATH_CALUDE_number_operation_result_l1978_197842


namespace NUMINAMATH_CALUDE_range_of_a_l1978_197846

-- Define the inequality
def inequality (x a : ℝ) : Prop :=
  (x - a) / (x^2 - 3*x + 2) ≥ 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x | 1 < x ∧ x ≤ a} ∪ {x | x > 2}

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, inequality x a ↔ x ∈ solution_set a) →
  a ∈ Set.Ioo 1 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1978_197846


namespace NUMINAMATH_CALUDE_albert_earnings_increase_l1978_197894

theorem albert_earnings_increase (E : ℝ) (P : ℝ) : 
  E * (1 + P / 100) = 598 →
  E * 1.35 = 621 →
  P = 30 := by
sorry

end NUMINAMATH_CALUDE_albert_earnings_increase_l1978_197894


namespace NUMINAMATH_CALUDE_max_b_minus_a_l1978_197889

theorem max_b_minus_a (a b : ℝ) : 
  a < 0 → 
  (∀ x : ℝ, (x^2 + 2017*a)*(x + 2016*b) ≥ 0) → 
  b - a ≤ 2017 :=
by sorry

end NUMINAMATH_CALUDE_max_b_minus_a_l1978_197889


namespace NUMINAMATH_CALUDE_min_max_inequality_l1978_197866

theorem min_max_inequality (a b x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : 0 < a) (h₂ : a < b) 
  (h₃ : a ≤ x₁ ∧ x₁ ≤ b) (h₄ : a ≤ x₂ ∧ x₂ ≤ b) 
  (h₅ : a ≤ x₃ ∧ x₃ ≤ b) (h₆ : a ≤ x₄ ∧ x₄ ≤ b) :
  1 ≤ (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ∧ 
  (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ≤ a/b + b/a - 1 :=
by sorry

end NUMINAMATH_CALUDE_min_max_inequality_l1978_197866


namespace NUMINAMATH_CALUDE_modulus_of_z_l1978_197808

-- Define the complex number z
def z : ℂ := 3 + 4 * Complex.I

-- State the theorem
theorem modulus_of_z : Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1978_197808


namespace NUMINAMATH_CALUDE_solution1_solution2_a_solution2_b_l1978_197839

-- Part 1
def equation1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 6*y + 13 = 0

theorem solution1 : equation1 2 (-3) := by sorry

-- Part 2
def equation2 (x y : ℝ) : Prop :=
  x*y - 1 = x - y

theorem solution2_a (y : ℝ) : equation2 1 y := by sorry

theorem solution2_b (x : ℝ) (h : x ≠ 1) : equation2 x 1 := by sorry

end NUMINAMATH_CALUDE_solution1_solution2_a_solution2_b_l1978_197839


namespace NUMINAMATH_CALUDE_square_playground_area_l1978_197803

theorem square_playground_area (w : ℝ) (s : ℝ) : 
  s = 3 * w + 10 →
  4 * s = 480 →
  s * s = 14400 := by
sorry

end NUMINAMATH_CALUDE_square_playground_area_l1978_197803


namespace NUMINAMATH_CALUDE_angle_measure_l1978_197800

theorem angle_measure (PQR PQS : ℝ) (h1 : PQR = 40) (h2 : PQS = 15) : PQR - PQS = 25 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l1978_197800


namespace NUMINAMATH_CALUDE_factor_expression_l1978_197825

theorem factor_expression (x y : ℝ) : 60 * x^2 + 40 * y = 20 * (3 * x^2 + 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1978_197825


namespace NUMINAMATH_CALUDE_rachel_age_problem_l1978_197887

/-- Rachel's age problem -/
theorem rachel_age_problem (rachel_age : ℕ) (grandfather_age : ℕ) (mother_age : ℕ) (father_age : ℕ) : 
  rachel_age = 12 →
  grandfather_age = 7 * rachel_age →
  mother_age = grandfather_age / 2 →
  father_age = mother_age + 5 →
  father_age + (25 - rachel_age) = 60 :=
by sorry

end NUMINAMATH_CALUDE_rachel_age_problem_l1978_197887


namespace NUMINAMATH_CALUDE_fraction_power_product_equals_three_halves_l1978_197834

theorem fraction_power_product_equals_three_halves :
  (3 / 2 : ℝ) ^ 2023 * (2 / 3 : ℝ) ^ 2022 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_equals_three_halves_l1978_197834


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1978_197827

/-- The distance between the foci of an ellipse with equation x^2 + 9y^2 = 576 is 32√2 -/
theorem ellipse_foci_distance : 
  let a : ℝ := Real.sqrt (576 / 1)
  let b : ℝ := Real.sqrt (576 / 9)
  2 * Real.sqrt (a^2 - b^2) = 32 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1978_197827


namespace NUMINAMATH_CALUDE_min_max_of_expression_l1978_197806

open Real

theorem min_max_of_expression (x : ℝ) (h : x > 2) :
  let f := fun x => (x + 9) / sqrt (x - 2)
  (∃ (m : ℝ), m = 2 * sqrt 11 ∧ 
    (∀ y, y > 2 → f y ≥ m) ∧ 
    f 13 = m) ∧
  (∀ M : ℝ, ∃ y, y > 2 ∧ f y > M) :=
by sorry

end NUMINAMATH_CALUDE_min_max_of_expression_l1978_197806


namespace NUMINAMATH_CALUDE_dorokhov_vacation_cost_l1978_197802

/-- Represents a travel agency with its pricing structure -/
structure TravelAgency where
  name : String
  under_age_price : ℕ
  over_age_price : ℕ
  age_threshold : ℕ
  discount_or_commission : ℚ
  is_discount : Bool

/-- Calculates the total cost for a family's vacation package -/
def calculate_cost (agency : TravelAgency) (num_adults num_children : ℕ) (child_age : ℕ) : ℚ :=
  let base_cost := 
    if child_age < agency.age_threshold
    then agency.under_age_price * num_children + agency.over_age_price * num_adults
    else agency.over_age_price * (num_adults + num_children)
  let adjustment := base_cost * agency.discount_or_commission
  if agency.is_discount
  then base_cost - adjustment
  else base_cost + adjustment

/-- The Dorokhov family vacation problem -/
theorem dorokhov_vacation_cost : 
  let globus : TravelAgency := {
    name := "Globus",
    under_age_price := 11200,
    over_age_price := 25400,
    age_threshold := 5,
    discount_or_commission := 2 / 100,
    is_discount := true
  }
  let around_world : TravelAgency := {
    name := "Around the World",
    under_age_price := 11400,
    over_age_price := 23500,
    age_threshold := 6,
    discount_or_commission := 1 / 100,
    is_discount := false
  }
  let globus_cost := calculate_cost globus 2 1 5
  let around_world_cost := calculate_cost around_world 2 1 5
  min globus_cost around_world_cost = 58984 := by sorry

end NUMINAMATH_CALUDE_dorokhov_vacation_cost_l1978_197802


namespace NUMINAMATH_CALUDE_recreation_spending_comparison_l1978_197876

theorem recreation_spending_comparison (wages_last_week : ℝ) : 
  let recreation_last_week := 0.60 * wages_last_week
  let wages_this_week := 0.90 * wages_last_week
  let recreation_this_week := 0.70 * wages_this_week
  recreation_this_week / recreation_last_week = 1.05 := by
sorry

end NUMINAMATH_CALUDE_recreation_spending_comparison_l1978_197876


namespace NUMINAMATH_CALUDE_cone_slant_height_l1978_197837

theorem cone_slant_height (V : ℝ) (θ : ℝ) (l : ℝ) : 
  V = 9 * Real.pi → θ = Real.pi / 4 → l = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_slant_height_l1978_197837


namespace NUMINAMATH_CALUDE_linda_coin_count_l1978_197820

/-- Represents the number of coins Linda has initially and receives from her mother -/
structure CoinCounts where
  initial_dimes : Nat
  initial_quarters : Nat
  initial_nickels : Nat
  additional_dimes : Nat
  additional_quarters : Nat

/-- Calculates the total number of coins Linda has -/
def totalCoins (counts : CoinCounts) : Nat :=
  counts.initial_dimes + counts.initial_quarters + counts.initial_nickels +
  counts.additional_dimes + counts.additional_quarters +
  2 * counts.initial_nickels

theorem linda_coin_count :
  let counts : CoinCounts := {
    initial_dimes := 2,
    initial_quarters := 6,
    initial_nickels := 5,
    additional_dimes := 2,
    additional_quarters := 10
  }
  totalCoins counts = 35 := by
  sorry

end NUMINAMATH_CALUDE_linda_coin_count_l1978_197820


namespace NUMINAMATH_CALUDE_interest_calculation_l1978_197833

/-- Calculates simple interest given principal, rate, and time -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (principal * rate * time) / 100

theorem interest_calculation (principal : ℚ) (rate : ℚ) (time : ℚ) 
  (h1 : principal = 3000)
  (h2 : rate = 5)
  (h3 : time = 5)
  (h4 : simple_interest principal rate time = principal - 2250) :
  simple_interest principal rate time = 750 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l1978_197833


namespace NUMINAMATH_CALUDE_bird_cost_l1978_197829

/-- The cost of birds at a pet store -/
theorem bird_cost (small_bird_cost large_bird_cost : ℚ) : 
  large_bird_cost = 2 * small_bird_cost →
  5 * large_bird_cost + 3 * small_bird_cost = 
    5 * small_bird_cost + 3 * large_bird_cost + 20 →
  small_bird_cost = 10 ∧ large_bird_cost = 20 := by
sorry

end NUMINAMATH_CALUDE_bird_cost_l1978_197829


namespace NUMINAMATH_CALUDE_parabola_chord_midpoint_l1978_197824

/-- Given a parabola y^2 = 2px and a chord with midpoint (3, 1) and slope 2, prove that p = 2 -/
theorem parabola_chord_midpoint (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- Equation of the parabola
  (∃ x₁ y₁ x₂ y₂ : ℝ,        -- Existence of two points on the chord
    y₁^2 = 2*p*x₁ ∧          -- First point satisfies parabola equation
    y₂^2 = 2*p*x₂ ∧          -- Second point satisfies parabola equation
    (x₁ + x₂)/2 = 3 ∧        -- x-coordinate of midpoint is 3
    (y₁ + y₂)/2 = 1 ∧        -- y-coordinate of midpoint is 1
    (y₂ - y₁)/(x₂ - x₁) = 2  -- Slope of the chord is 2
  ) →
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_chord_midpoint_l1978_197824


namespace NUMINAMATH_CALUDE_not_always_determinable_l1978_197804

/-- Represents a weight with a mass -/
structure Weight where
  mass : ℝ

/-- Represents a question about the order of three weights -/
structure Question where
  a : Weight
  b : Weight
  c : Weight

/-- The set of all possible permutations of five weights -/
def AllPermutations : Finset (List Weight) :=
  sorry

/-- The number of questions we can ask -/
def NumQuestions : ℕ := 9

/-- A function that simulates asking a question -/
def askQuestion (q : Question) (perm : List Weight) : Bool :=
  sorry

/-- The main theorem stating that it's not always possible to determine the exact order -/
theorem not_always_determinable (weights : Finset Weight) 
  (h : weights.card = 5) :
  ∃ (perm₁ perm₂ : List Weight),
    perm₁ ∈ AllPermutations ∧ 
    perm₂ ∈ AllPermutations ∧ 
    perm₁ ≠ perm₂ ∧
    ∀ (questions : Finset Question),
      questions.card ≤ NumQuestions →
      ∀ (q : Question),
        q ∈ questions →
        askQuestion q perm₁ = askQuestion q perm₂ :=
  sorry

end NUMINAMATH_CALUDE_not_always_determinable_l1978_197804


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l1978_197875

theorem smallest_dual_base_representation : ∃ n : ℕ, ∃ a b : ℕ, 
  (a > 2 ∧ b > 2) ∧ 
  (1 * a + 3 = n) ∧ 
  (3 * b + 1 = n) ∧
  (∀ m : ℕ, ∀ c d : ℕ, 
    (c > 2 ∧ d > 2) → 
    (1 * c + 3 = m) → 
    (3 * d + 1 = m) → 
    m ≥ n) ∧
  n = 13 :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l1978_197875


namespace NUMINAMATH_CALUDE_soccer_ball_properties_l1978_197831

/-- A soccer-ball polyhedron has faces that are m-gons or n-gons (m ≠ n),
    and in every vertex, three faces meet: two m-gons and one n-gon. -/
structure SoccerBallPolyhedron where
  m : ℕ
  n : ℕ
  m_ne_n : m ≠ n
  vertex_config : 2 * ((m - 2) * π / m) + ((n - 2) * π / n) = 2 * π

theorem soccer_ball_properties (P : SoccerBallPolyhedron) :
  Even P.m ∧ P.m = 6 ∧ P.n = 5 := by
  sorry

#check soccer_ball_properties

end NUMINAMATH_CALUDE_soccer_ball_properties_l1978_197831


namespace NUMINAMATH_CALUDE_min_n_for_constant_term_l1978_197883

theorem min_n_for_constant_term (n : ℕ) : 
  (∃ k : ℕ, (2 * n = 3 * k) ∧ (k ≤ n)) ↔ n ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_n_for_constant_term_l1978_197883


namespace NUMINAMATH_CALUDE_condition_relationship_l1978_197868

theorem condition_relationship (p q r s : Prop) 
  (h1 : (r → q) ∧ ¬(q → r))  -- q is necessary but not sufficient for r
  (h2 : (s ↔ r))             -- s is sufficient and necessary for r
  : (s → q) ∧ ¬(q → s) :=    -- s is sufficient but not necessary for q
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l1978_197868


namespace NUMINAMATH_CALUDE_metallic_sheet_width_l1978_197874

/-- Represents the dimensions and properties of a metallic sheet and the box formed from it. -/
structure MetallicSheet where
  length : ℝ
  width : ℝ
  cutSize : ℝ
  boxVolume : ℝ

/-- Theorem stating the width of the metallic sheet given the conditions -/
theorem metallic_sheet_width (sheet : MetallicSheet)
  (h1 : sheet.length = 50)
  (h2 : sheet.cutSize = 8)
  (h3 : sheet.boxVolume = 5440)
  (h4 : sheet.boxVolume = (sheet.length - 2 * sheet.cutSize) * (sheet.width - 2 * sheet.cutSize) * sheet.cutSize) :
  sheet.width = 36 := by
  sorry


end NUMINAMATH_CALUDE_metallic_sheet_width_l1978_197874


namespace NUMINAMATH_CALUDE_range_of_m_l1978_197835

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x ≤ 5}

-- Define set B parameterized by m
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem statement
theorem range_of_m (m : ℝ) : B m ⊆ A → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1978_197835


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1978_197879

/-- A hyperbola in the Cartesian coordinate system -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form (x^2 / a^2) - (y^2 / b^2) = 1 -/
  equation : ℝ → ℝ → Prop

/-- A parabola in the Cartesian coordinate system -/
structure Parabola where
  /-- The equation of the parabola -/
  equation : ℝ → ℝ → Prop

/-- The focus of a parabola -/
def parabola_focus (p : Parabola) : ℝ × ℝ := sorry

/-- The right focus of a hyperbola -/
def hyperbola_right_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- Theorem: Given a hyperbola C with its center at the origin, passing through (1, 0),
    and its right focus coinciding with the focus of y^2 = 8x, 
    the standard equation of C is x^2 - y^2/3 = 1 -/
theorem hyperbola_equation 
  (C : Hyperbola)
  (center_origin : C.equation 0 0)
  (passes_through_1_0 : C.equation 1 0)
  (p : Parabola)
  (p_eq : p.equation = fun x y ↦ y^2 = 8*x)
  (focus_coincide : hyperbola_right_focus C = parabola_focus p) :
  C.equation = fun x y ↦ x^2 - y^2/3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1978_197879


namespace NUMINAMATH_CALUDE_additional_spend_for_free_shipping_l1978_197884

/-- The amount needed to qualify for free shipping -/
def free_shipping_threshold : ℝ := 50

/-- The cost of a bottle of shampoo or conditioner -/
def shampoo_conditioner_cost : ℝ := 10

/-- The cost of a bottle of lotion -/
def lotion_cost : ℝ := 6

/-- The number of bottles of lotion Jackie bought -/
def lotion_quantity : ℕ := 3

/-- Jackie's current spend -/
def current_spend : ℝ := 2 * shampoo_conditioner_cost + lotion_quantity * lotion_cost

/-- The additional amount Jackie needs to spend for free shipping -/
theorem additional_spend_for_free_shipping :
  free_shipping_threshold - current_spend = 12 := by sorry

end NUMINAMATH_CALUDE_additional_spend_for_free_shipping_l1978_197884


namespace NUMINAMATH_CALUDE_six_legs_is_insect_l1978_197859

/-- Represents an animal with a certain number of legs -/
structure Animal where
  legs : ℕ

/-- Definition of an insect based on number of legs -/
def is_insect (a : Animal) : Prop := a.legs = 6

/-- Theorem stating that an animal with 6 legs satisfies the definition of an insect -/
theorem six_legs_is_insect (a : Animal) (h : a.legs = 6) : is_insect a := by
  sorry

end NUMINAMATH_CALUDE_six_legs_is_insect_l1978_197859


namespace NUMINAMATH_CALUDE_ninas_money_l1978_197891

theorem ninas_money (x : ℚ) 
  (h1 : 10 * x = 14 * (x - 1)) : 10 * x = 35 :=
by sorry

end NUMINAMATH_CALUDE_ninas_money_l1978_197891


namespace NUMINAMATH_CALUDE_weight_range_proof_l1978_197810

/-- Given the weights of Tracy, John, and Jake, prove the range of their weights. -/
theorem weight_range_proof (tracy_weight john_weight jake_weight : ℕ) 
  (h1 : tracy_weight + john_weight + jake_weight = 158)
  (h2 : tracy_weight = 52)
  (h3 : jake_weight = tracy_weight + 8) : 
  (max tracy_weight (max john_weight jake_weight)) - 
  (min tracy_weight (min john_weight jake_weight)) = 14 := by
  sorry

#check weight_range_proof

end NUMINAMATH_CALUDE_weight_range_proof_l1978_197810


namespace NUMINAMATH_CALUDE_sweets_ratio_l1978_197871

/-- Proves that the ratio of sweets received by the youngest child to the eldest child is 1:2 --/
theorem sweets_ratio (total : ℕ) (eldest : ℕ) (second : ℕ) : 
  total = 27 →
  eldest = 8 →
  second = 6 →
  (total - (total / 3) - eldest - second) * 2 = eldest := by
  sorry

end NUMINAMATH_CALUDE_sweets_ratio_l1978_197871


namespace NUMINAMATH_CALUDE_pool_paint_area_calculation_l1978_197864

/-- Calculates the total area to be painted in a cuboid-shaped pool -/
def poolPaintArea (length width depth : ℝ) : ℝ :=
  2 * (length * depth + width * depth) + length * width

theorem pool_paint_area_calculation :
  let length : ℝ := 20
  let width : ℝ := 12
  let depth : ℝ := 2
  poolPaintArea length width depth = 368 := by
  sorry

end NUMINAMATH_CALUDE_pool_paint_area_calculation_l1978_197864


namespace NUMINAMATH_CALUDE_zahar_process_terminates_l1978_197818

/-- Represents the state of the notebooks -/
def NotebookState := List Nat

/-- Represents a single operation in Zahar's process -/
def ZaharOperation (state : NotebookState) : Option NotebookState := sorry

/-- Predicate to check if the notebooks are in ascending order -/
def IsAscendingOrder (state : NotebookState) : Prop := sorry

/-- Predicate to check if a state is valid (contains numbers 1 to n) -/
def IsValidState (state : NotebookState) : Prop := sorry

/-- The main theorem stating that Zahar's process will terminate -/
theorem zahar_process_terminates (n : Nat) (initial_state : NotebookState) :
  n ≥ 1 →
  IsValidState initial_state →
  ∃ (final_state : NotebookState) (steps : Nat),
    (∀ k : Nat, k < steps → ∃ intermediate_state, ZaharOperation (intermediate_state) ≠ none) ∧
    ZaharOperation final_state = none ∧
    IsAscendingOrder final_state :=
  sorry

end NUMINAMATH_CALUDE_zahar_process_terminates_l1978_197818


namespace NUMINAMATH_CALUDE_only_solution_is_two_l1978_197809

theorem only_solution_is_two : 
  ∃! (n : ℤ), n + 13 > 15 ∧ -6*n > -18 :=
by sorry

end NUMINAMATH_CALUDE_only_solution_is_two_l1978_197809


namespace NUMINAMATH_CALUDE_sixty_degree_iff_arithmetic_progression_l1978_197862

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = 180

/-- The property that the angles of a triangle are in arithmetic progression -/
def angles_in_arithmetic_progression (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B

/-- Theorem stating that B = 60° is necessary and sufficient for the angles to be in arithmetic progression -/
theorem sixty_degree_iff_arithmetic_progression (t : Triangle) :
  t.B = 60 ↔ angles_in_arithmetic_progression t := by
  sorry

end NUMINAMATH_CALUDE_sixty_degree_iff_arithmetic_progression_l1978_197862


namespace NUMINAMATH_CALUDE_company_employees_l1978_197805

theorem company_employees (wednesday_birthdays : ℕ) 
  (other_day_birthdays : ℕ) : 
  wednesday_birthdays = 13 →
  wednesday_birthdays > other_day_birthdays →
  (7 * other_day_birthdays + wednesday_birthdays - other_day_birthdays : ℕ) = 85 :=
by sorry

end NUMINAMATH_CALUDE_company_employees_l1978_197805


namespace NUMINAMATH_CALUDE_mateo_deducted_salary_l1978_197841

theorem mateo_deducted_salary (weekly_salary : ℝ) (work_days : ℕ) (absent_days : ℕ) : 
  weekly_salary = 791 ∧ work_days = 5 ∧ absent_days = 4 →
  weekly_salary - (weekly_salary / work_days * absent_days) = 158.20 := by
  sorry

end NUMINAMATH_CALUDE_mateo_deducted_salary_l1978_197841


namespace NUMINAMATH_CALUDE_fruits_eaten_over_two_meals_l1978_197896

/-- Calculates the total number of fruits eaten over two meals given specific conditions --/
theorem fruits_eaten_over_two_meals : 
  let apples_last_night : ℕ := 3
  let bananas_last_night : ℕ := 1
  let oranges_last_night : ℕ := 4
  let strawberries_last_night : ℕ := 2
  
  let apples_today : ℕ := apples_last_night + 4
  let bananas_today : ℕ := bananas_last_night * 10
  let oranges_today : ℕ := apples_today * 2
  let strawberries_today : ℕ := (oranges_last_night + apples_last_night) * 3
  
  let total_fruits : ℕ := 
    (apples_last_night + apples_today) +
    (bananas_last_night + bananas_today) +
    (oranges_last_night + oranges_today) +
    (strawberries_last_night + strawberries_today)
  
  total_fruits = 62 := by sorry

end NUMINAMATH_CALUDE_fruits_eaten_over_two_meals_l1978_197896


namespace NUMINAMATH_CALUDE_conference_handshakes_l1978_197811

/-- Represents a conference with handshakes --/
structure Conference where
  total_people : Nat
  normal_handshakes : Nat
  restricted_people : Nat
  restricted_handshakes : Nat

/-- Calculates the maximum number of unique handshakes in a conference --/
def max_handshakes (c : Conference) : Nat :=
  let total_pairs := c.total_people.choose 2
  let reduced_handshakes := c.restricted_people * (c.normal_handshakes - c.restricted_handshakes)
  total_pairs - reduced_handshakes

/-- The theorem stating the maximum number of handshakes for the given conference --/
theorem conference_handshakes :
  let c : Conference := {
    total_people := 25,
    normal_handshakes := 20,
    restricted_people := 5,
    restricted_handshakes := 15
  }
  max_handshakes c = 250 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1978_197811


namespace NUMINAMATH_CALUDE_quadratic_root_equivalence_l1978_197878

theorem quadratic_root_equivalence (a : ℝ) : 
  (∃ k : ℝ, k > 0 ∧ Real.sqrt 12 = k * Real.sqrt 3) ∧ 
  (∃ m : ℝ, m > 0 ∧ 5 * Real.sqrt (a + 1) = m * Real.sqrt 3) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_equivalence_l1978_197878


namespace NUMINAMATH_CALUDE_sin_cos_inequality_l1978_197823

theorem sin_cos_inequality (x : ℝ) : 
  0 ≤ x ∧ x ≤ 2 * π ∧ Real.sin (x - π / 6) > Real.cos x → 
  π / 3 < x ∧ x < 4 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_inequality_l1978_197823


namespace NUMINAMATH_CALUDE_black_area_from_white_area_l1978_197867

/-- Represents a square divided into 9 equal smaller squares -/
structure DividedSquare where
  total_area : ℝ
  white_squares : ℕ
  black_squares : ℕ
  white_area : ℝ

/-- Theorem stating the relation between white and black areas in the divided square -/
theorem black_area_from_white_area (s : DividedSquare) 
  (h1 : s.white_squares + s.black_squares = 9)
  (h2 : s.white_squares = 5)
  (h3 : s.black_squares = 4)
  (h4 : s.white_area = 180) :
  s.total_area * (s.black_squares / 9 : ℝ) = 144 := by
sorry

end NUMINAMATH_CALUDE_black_area_from_white_area_l1978_197867


namespace NUMINAMATH_CALUDE_tobys_change_is_seven_l1978_197861

/-- Represents the dining scenario and calculates Toby's change --/
def tobys_change (cheeseburger_price : ℚ) (milkshake_price : ℚ) (coke_price : ℚ) 
                 (fries_price : ℚ) (cookie_price : ℚ) (tax : ℚ) 
                 (toby_initial_money : ℚ) : ℚ :=
  let total_cost := 2 * cheeseburger_price + milkshake_price + coke_price + 
                    fries_price + 3 * cookie_price + tax
  let toby_share := total_cost / 2
  toby_initial_money - toby_share

/-- Theorem stating that Toby's change is $7.00 --/
theorem tobys_change_is_seven : 
  tobys_change 3.65 2 1 4 0.5 0.2 15 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tobys_change_is_seven_l1978_197861


namespace NUMINAMATH_CALUDE_shortest_side_length_l1978_197840

/-- Represents a triangle with angles in the ratio 1:2:3 and longest side of length 6 -/
structure SpecialTriangle where
  /-- The smallest angle of the triangle -/
  smallest_angle : ℝ
  /-- The ratio of angles is 1:2:3 -/
  angle_ratio : smallest_angle > 0 ∧ smallest_angle + 2 * smallest_angle + 3 * smallest_angle = 180
  /-- The length of the longest side is 6 -/
  longest_side : ℝ
  longest_side_eq : longest_side = 6

/-- The length of the shortest side in a SpecialTriangle is 3 -/
theorem shortest_side_length (t : SpecialTriangle) : ∃ (shortest_side : ℝ), shortest_side = 3 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_length_l1978_197840


namespace NUMINAMATH_CALUDE_fruit_mix_cherries_l1978_197826

/-- Proves that in a fruit mix with the given conditions, the number of cherries is 167 -/
theorem fruit_mix_cherries (b r c : ℕ) : 
  b + r + c = 300 → 
  r = 3 * b → 
  c = 5 * b → 
  c = 167 := by
  sorry

end NUMINAMATH_CALUDE_fruit_mix_cherries_l1978_197826


namespace NUMINAMATH_CALUDE_f_lower_bound_and_equality_l1978_197899

def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

theorem f_lower_bound_and_equality (a b : ℝ) 
  (h : (1 / (2 * a)) + (2 / b) = 1) :
  (∀ x, f x a b ≥ 9/2) ∧
  (∃ x, f x a b = 9/2 → a = 3/2 ∧ b = 3) := by
sorry

end NUMINAMATH_CALUDE_f_lower_bound_and_equality_l1978_197899


namespace NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l1978_197848

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (∀ x : ℂ, (3 : ℂ) * x^2 + (a : ℂ) * x + (b : ℂ) = 0 ↔ x = 5 + 2*I ∨ x = 5 - 2*I) ∧
    (3 : ℝ) * (5 + 2*I)^2 + a * (5 + 2*I) + b = 0 ∧
    a = -30 ∧ b = 87 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l1978_197848


namespace NUMINAMATH_CALUDE_disinfectant_transport_theorem_l1978_197865

/-- Represents the number of bottles a box can hold -/
structure BoxCapacity where
  large : Nat
  small : Nat

/-- Represents the cost of a box in yuan -/
structure BoxCost where
  large : Nat
  small : Nat

/-- Represents the carrying capacity of a vehicle -/
structure VehicleCapacity where
  large : Nat
  small : Nat

/-- Represents the number of boxes purchased -/
structure Boxes where
  large : Nat
  small : Nat

/-- Represents the number of vehicles of each type -/
structure Vehicles where
  typeA : Nat
  typeB : Nat

def total_bottles : Nat := 3250
def total_cost : Nat := 1700
def total_vehicles : Nat := 10

def box_capacity : BoxCapacity := { large := 10, small := 5 }
def box_cost : BoxCost := { large := 5, small := 3 }
def vehicle_capacity_A : VehicleCapacity := { large := 30, small := 10 }
def vehicle_capacity_B : VehicleCapacity := { large := 20, small := 40 }

def is_valid_box_purchase (boxes : Boxes) : Prop :=
  boxes.large * box_capacity.large + boxes.small * box_capacity.small = total_bottles ∧
  boxes.large * box_cost.large + boxes.small * box_cost.small = total_cost

def is_valid_vehicle_arrangement (vehicles : Vehicles) (boxes : Boxes) : Prop :=
  vehicles.typeA + vehicles.typeB = total_vehicles ∧
  vehicles.typeA * vehicle_capacity_A.large + vehicles.typeB * vehicle_capacity_B.large ≥ boxes.large ∧
  vehicles.typeA * vehicle_capacity_A.small + vehicles.typeB * vehicle_capacity_B.small ≥ boxes.small

def is_optimal_arrangement (vehicles : Vehicles) (boxes : Boxes) : Prop :=
  is_valid_vehicle_arrangement vehicles boxes ∧
  ∀ (other : Vehicles), is_valid_vehicle_arrangement other boxes → vehicles.typeA ≥ other.typeA

theorem disinfectant_transport_theorem : 
  ∃ (boxes : Boxes) (vehicles : Vehicles),
    is_valid_box_purchase boxes ∧
    boxes.large = 250 ∧
    boxes.small = 150 ∧
    is_optimal_arrangement vehicles boxes ∧
    vehicles.typeA = 8 ∧
    vehicles.typeB = 2 := by sorry

end NUMINAMATH_CALUDE_disinfectant_transport_theorem_l1978_197865


namespace NUMINAMATH_CALUDE_train_a_speed_l1978_197888

/-- The speed of Train A in miles per hour -/
def speed_train_a : ℝ := 30

/-- The speed of Train B in miles per hour -/
def speed_train_b : ℝ := 36

/-- The time difference in hours between Train A and Train B's departure -/
def time_difference : ℝ := 2

/-- The distance in miles at which Train B overtakes Train A -/
def overtake_distance : ℝ := 360

/-- Theorem stating that the speed of Train A is 30 mph given the conditions -/
theorem train_a_speed :
  ∃ (t : ℝ), 
    t > time_difference ∧
    speed_train_a * t = overtake_distance ∧
    speed_train_b * (t - time_difference) = overtake_distance ∧
    speed_train_a = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_a_speed_l1978_197888


namespace NUMINAMATH_CALUDE_prime_power_minus_cube_eq_one_l1978_197801

theorem prime_power_minus_cube_eq_one (p : ℕ) (hp : Prime p) :
  ∀ x y : ℕ, x > 0 → y > 0 → p^x - y^3 = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_prime_power_minus_cube_eq_one_l1978_197801


namespace NUMINAMATH_CALUDE_function_zeros_imply_k_nonnegative_l1978_197885

-- Define the piecewise function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x / (x - 2) + k * x^2 else Real.log x

-- State the theorem
theorem function_zeros_imply_k_nonnegative (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f k x = 0 ∧ f k y = 0) ∧
  (∀ z w v : ℝ, f k z = 0 ∧ f k w = 0 ∧ f k v = 0 → z = w ∨ w = v ∨ z = v) →
  k ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_function_zeros_imply_k_nonnegative_l1978_197885


namespace NUMINAMATH_CALUDE_range_of_m_l1978_197857

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * x + 1

-- Define the condition for two real roots
def has_two_real_roots (m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic_equation m x₁ = 0 ∧ quadratic_equation m x₂ = 0

-- Theorem statement
theorem range_of_m (m : ℝ) :
  has_two_real_roots m ↔ m ≤ 2 ∧ m ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1978_197857


namespace NUMINAMATH_CALUDE_second_boy_marbles_l1978_197819

-- Define the number of marbles for each boy as functions of x
def boy1_marbles (x : ℚ) : ℚ := 4 * x + 2
def boy2_marbles (x : ℚ) : ℚ := 3 * x - 1
def boy3_marbles (x : ℚ) : ℚ := 5 * x + 3

-- Define the total number of marbles
def total_marbles : ℚ := 128

-- Theorem statement
theorem second_boy_marbles :
  ∃ x : ℚ, 
    boy1_marbles x + boy2_marbles x + boy3_marbles x = total_marbles ∧
    boy2_marbles x = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_boy_marbles_l1978_197819


namespace NUMINAMATH_CALUDE_rational_equation_solution_no_solution_rational_equation_l1978_197836

-- Problem 1
theorem rational_equation_solution (x : ℝ) :
  x ≠ 2 →
  ((2*x - 5) / (x - 2) = 3 / (2 - x)) ↔ (x = 4) :=
sorry

-- Problem 2
theorem no_solution_rational_equation (x : ℝ) :
  x ≠ 3 →
  x ≠ -3 →
  ¬(12 / (x^2 - 9) - 2 / (x - 3) = 1 / (x + 3)) :=
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_no_solution_rational_equation_l1978_197836
