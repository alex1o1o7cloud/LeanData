import Mathlib

namespace exists_unformable_figure_l1267_126715

/-- Represents a geometric shape --/
inductive Shape
  | Square : Shape
  | Rectangle1x3 : Shape
  | Rectangle2x1 : Shape
  | LShape : Shape

/-- Represents a geometric figure --/
structure Figure where
  area : ℕ
  canBeFormed : Bool

/-- The set of available shapes --/
def availableShapes : List Shape :=
  [Shape.Square, Shape.Square, Shape.Rectangle1x3, Shape.Rectangle2x1, Shape.LShape]

/-- The total area of all available shapes --/
def totalArea : ℕ := 13

/-- There are eight different geometric figures --/
def figures : List Figure := sorry

/-- Theorem: There exists a figure that cannot be formed from the available shapes --/
theorem exists_unformable_figure :
  ∃ (f : Figure), f ∈ figures ∧ f.canBeFormed = false :=
sorry

end exists_unformable_figure_l1267_126715


namespace relationship_abc_l1267_126728

theorem relationship_abc : 
  let a : ℝ := Real.rpow 0.6 0.6
  let b : ℝ := Real.rpow 0.6 1.5
  let c : ℝ := Real.rpow 1.5 0.6
  b < a ∧ a < c := by sorry

end relationship_abc_l1267_126728


namespace smallest_divisible_by_1_to_10_is_correct_l1267_126787

/-- The smallest positive integer divisible by all integers from 1 to 10 -/
def smallest_divisible_by_1_to_10 : ℕ := 2520

/-- Predicate to check if a number is divisible by all integers from 1 to 10 -/
def divisible_by_1_to_10 (n : ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i → i ≤ 10 → n % i = 0

theorem smallest_divisible_by_1_to_10_is_correct :
  (divisible_by_1_to_10 smallest_divisible_by_1_to_10) ∧
  (∀ n : ℕ, n > 0 → divisible_by_1_to_10 n → n ≥ smallest_divisible_by_1_to_10) :=
by sorry

end smallest_divisible_by_1_to_10_is_correct_l1267_126787


namespace intersection_nonempty_intersection_equals_B_l1267_126717

def A : Set ℝ := {x | x + 1 ≤ 0 ∨ x - 4 ≥ 0}
def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 2}

theorem intersection_nonempty (a : ℝ) :
  (A ∩ B a).Nonempty ↔ a ≤ -1/2 ∨ a = 2 := by sorry

theorem intersection_equals_B (a : ℝ) :
  A ∩ B a = B a ↔ a ≤ -1/2 ∨ a ≥ 2 := by sorry

end intersection_nonempty_intersection_equals_B_l1267_126717


namespace valid_numbers_l1267_126723

def is_valid_number (n : ℕ) : Prop :=
  (1000 ≤ n) ∧ (n < 10000) ∧
  (∃ (a d : ℕ), 
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
    n = 120 * (10 * a + d))

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {1200, 2400, 3600, 4800} := by sorry

end valid_numbers_l1267_126723


namespace odd_coefficients_equals_two_pow_binary_ones_l1267_126774

/-- The number of 1s in the binary representation of a natural number -/
def binaryOnes (n : ℕ) : ℕ := sorry

/-- The number of odd coefficients in the polynomial expansion of (1+x)^n -/
def oddCoefficients (n : ℕ) : ℕ := sorry

/-- Theorem: The number of odd coefficients in (1+x)^n is 2^d, where d is the number of 1s in n's binary representation -/
theorem odd_coefficients_equals_two_pow_binary_ones (n : ℕ) :
  oddCoefficients n = 2^(binaryOnes n) := by sorry

end odd_coefficients_equals_two_pow_binary_ones_l1267_126774


namespace math_club_composition_l1267_126716

theorem math_club_composition (boys girls : ℕ) : 
  boys = girls →
  (girls : ℚ) = 3/4 * (boys + girls - 1 : ℚ) →
  boys = 2 ∧ girls = 3 := by
sorry

end math_club_composition_l1267_126716


namespace quadratic_form_sum_l1267_126791

/-- Given that 2x^2 - 8x + 1 can be expressed as a(x-h)^2 + k, prove that a + h + k = -3 -/
theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 2*x^2 - 8*x + 1 = a*(x-h)^2 + k) → a + h + k = -3 := by
  sorry

end quadratic_form_sum_l1267_126791


namespace roses_per_flat_l1267_126708

/-- Represents the number of flats of petunias -/
def petunia_flats : ℕ := 4

/-- Represents the number of petunias per flat -/
def petunias_per_flat : ℕ := 8

/-- Represents the number of flats of roses -/
def rose_flats : ℕ := 3

/-- Represents the number of Venus flytraps -/
def venus_flytraps : ℕ := 2

/-- Represents the amount of fertilizer needed for each petunia (in ounces) -/
def fertilizer_per_petunia : ℕ := 8

/-- Represents the amount of fertilizer needed for each rose (in ounces) -/
def fertilizer_per_rose : ℕ := 3

/-- Represents the amount of fertilizer needed for each Venus flytrap (in ounces) -/
def fertilizer_per_venus_flytrap : ℕ := 2

/-- Represents the total amount of fertilizer needed (in ounces) -/
def total_fertilizer : ℕ := 314

/-- Proves that the number of roses in each flat is 6 -/
theorem roses_per_flat : ℕ := by
  sorry

end roses_per_flat_l1267_126708


namespace complex_arithmetic_l1267_126795

theorem complex_arithmetic (A M S : ℂ) (P : ℝ) 
  (hA : A = 5 - 2*I) 
  (hM : M = -3 + 2*I) 
  (hS : S = 2*I) 
  (hP : P = 3) : 
  A - M + S - (P : ℂ) = 5 - 2*I := by
  sorry

end complex_arithmetic_l1267_126795


namespace cookie_sheet_perimeter_l1267_126786

/-- The perimeter of a rectangular cookie sheet -/
theorem cookie_sheet_perimeter (width : ℝ) (length : ℝ) (inch_to_cm : ℝ) : 
  width = 15.2 ∧ length = 3.7 ∧ inch_to_cm = 2.54 →
  2 * (width * inch_to_cm + length * inch_to_cm) = 96.012 := by
  sorry

end cookie_sheet_perimeter_l1267_126786


namespace line_parameterization_l1267_126713

/-- Given a line y = 5x - 7 parameterized as (x, y) = (s, -3) + t(3, m),
    prove that s = 4/5 and m = 8 -/
theorem line_parameterization (s m : ℝ) : 
  (∀ t x y : ℝ, x = s + 3*t ∧ y = -3 + m*t → y = 5*x - 7) →
  s = 4/5 ∧ m = 8 := by
sorry

end line_parameterization_l1267_126713


namespace sqrt_81_division_l1267_126729

theorem sqrt_81_division :
  ∃ x : ℝ, x > 0 ∧ (Real.sqrt 81) / x = 3 := by sorry

end sqrt_81_division_l1267_126729


namespace percentage_passed_both_l1267_126799

theorem percentage_passed_both (failed_hindi : ℝ) (failed_english : ℝ) (failed_both : ℝ)
  (h1 : failed_hindi = 20)
  (h2 : failed_english = 70)
  (h3 : failed_both = 10) :
  100 - (failed_hindi + failed_english - failed_both) = 20 := by
  sorry

end percentage_passed_both_l1267_126799


namespace average_lawn_cuts_l1267_126780

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of months Mr. Roper cuts his lawn 15 times -/
def high_frequency_months : ℕ := 6

/-- The number of months Mr. Roper cuts his lawn 3 times -/
def low_frequency_months : ℕ := 6

/-- The number of times Mr. Roper cuts his lawn in high frequency months -/
def high_frequency_cuts : ℕ := 15

/-- The number of times Mr. Roper cuts his lawn in low frequency months -/
def low_frequency_cuts : ℕ := 3

/-- Theorem stating the average number of times Mr. Roper cuts his yard per month -/
theorem average_lawn_cuts :
  (high_frequency_months * high_frequency_cuts + low_frequency_months * low_frequency_cuts) / months_in_year = 9 := by
  sorry

end average_lawn_cuts_l1267_126780


namespace children_on_tricycles_l1267_126700

/-- The number of wheels on a bicycle -/
def bicycle_wheels : ℕ := 2

/-- The number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- The number of adults riding bicycles -/
def adults_on_bicycles : ℕ := 6

/-- The total number of wheels observed -/
def total_wheels : ℕ := 57

/-- Theorem stating that the number of children riding tricycles is 15 -/
theorem children_on_tricycles : 
  ∃ (c : ℕ), c * tricycle_wheels + adults_on_bicycles * bicycle_wheels = total_wheels ∧ c = 15 := by
  sorry

end children_on_tricycles_l1267_126700


namespace day5_sale_correct_l1267_126771

/-- Represents the sales data for a grocer over 6 days -/
structure GrocerSales where
  average_target : ℕ  -- Target average sale for 5 consecutive days
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day5 : ℕ  -- The day we want to calculate
  day6 : ℕ
  total_days : ℕ  -- Number of days for average calculation

/-- Calculates the required sale on the fifth day to meet the average target -/
def calculate_day5_sale (sales : GrocerSales) : ℕ :=
  sales.average_target * sales.total_days - (sales.day1 + sales.day2 + sales.day3 + sales.day5 + sales.day6)

/-- Theorem stating that the calculated sale for day 5 is correct -/
theorem day5_sale_correct (sales : GrocerSales) 
  (h1 : sales.average_target = 625)
  (h2 : sales.day1 = 435)
  (h3 : sales.day2 = 927)
  (h4 : sales.day3 = 855)
  (h5 : sales.day5 = 562)
  (h6 : sales.day6 = 741)
  (h7 : sales.total_days = 5) :
  calculate_day5_sale sales = 167 := by
  sorry

#eval calculate_day5_sale { 
  average_target := 625, 
  day1 := 435, 
  day2 := 927, 
  day3 := 855, 
  day5 := 562, 
  day6 := 741, 
  total_days := 5 
}

end day5_sale_correct_l1267_126771


namespace sequence_a_2006_bounds_l1267_126793

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | n+1 => sequence_a n + (1/2006) * (sequence_a n)^2

theorem sequence_a_2006_bounds : 
  1 - 1/2008 < sequence_a 2006 ∧ sequence_a 2006 < 1 := by
  sorry

end sequence_a_2006_bounds_l1267_126793


namespace cube_surface_area_proof_l1267_126784

-- Define the edge length of the cube
def edge_length : ℝ → ℝ := λ a => 7 * a

-- Define the surface area of a cube given its edge length
def cube_surface_area (edge : ℝ) : ℝ := 6 * edge^2

-- Theorem statement
theorem cube_surface_area_proof (a : ℝ) :
  cube_surface_area (edge_length a) = 294 * a^2 := by
  sorry

end cube_surface_area_proof_l1267_126784


namespace woojoo_initial_score_l1267_126710

theorem woojoo_initial_score 
  (num_students : ℕ) 
  (initial_avg : ℚ) 
  (new_score : ℕ) 
  (new_avg : ℚ) 
  (h1 : num_students = 10)
  (h2 : initial_avg = 42)
  (h3 : new_score = 50)
  (h4 : new_avg = 44) :
  ∃ (initial_score : ℕ), 
    (initial_score : ℚ) + (num_students - 1 : ℚ) * initial_avg = num_students * initial_avg ∧
    (new_score : ℚ) + (num_students - 1 : ℚ) * initial_avg = num_students * new_avg ∧
    initial_score = 30 := by
  sorry

end woojoo_initial_score_l1267_126710


namespace highway_speed_is_30_l1267_126755

-- Define the problem parameters
def initial_reading : ℕ := 12321
def next_palindrome : ℕ := 12421
def total_time : ℕ := 4
def highway_time : ℕ := 2
def urban_time : ℕ := 2
def speed_difference : ℕ := 10
def total_distance : ℕ := 100

-- Define the theorem
theorem highway_speed_is_30 :
  let urban_speed := (total_distance - speed_difference * highway_time) / total_time
  urban_speed + speed_difference = 30 := by
  sorry


end highway_speed_is_30_l1267_126755


namespace kittens_left_tim_kittens_left_l1267_126739

/-- Given an initial number of kittens and the number of kittens given to two people,
    calculate the number of kittens left. -/
theorem kittens_left (initial : ℕ) (given_to_jessica : ℕ) (given_to_sara : ℕ) :
  initial - (given_to_jessica + given_to_sara) = initial - given_to_jessica - given_to_sara :=
by sorry

/-- Prove that Tim has 9 kittens left after giving away some kittens. -/
theorem tim_kittens_left :
  let initial := 18
  let given_to_jessica := 3
  let given_to_sara := 6
  initial - (given_to_jessica + given_to_sara) = 9 :=
by sorry

end kittens_left_tim_kittens_left_l1267_126739


namespace geometric_sum_not_always_geometric_arithmetic_and_geometric_is_constant_sum_power_not_always_arithmetic_or_geometric_arithmetic_sequence_no_equal_terms_l1267_126782

-- Definition of a geometric sequence
def is_geometric_sequence (a b c d : ℝ) : Prop :=
  (b / a = c / b) ∧ (c / b = d / c)

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Definition of a geometric sequence (for infinite sequences)
def is_geometric_sequence_inf (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Definition of a constant sequence
def is_constant_sequence (a : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, a n = c

theorem geometric_sum_not_always_geometric :
  ∃ a b c d : ℝ, is_geometric_sequence a b c d ∧
  ¬ is_geometric_sequence (a + b) (b + c) (c + d) (d + a) :=
sorry

theorem arithmetic_and_geometric_is_constant (a : ℕ → ℝ) :
  is_arithmetic_sequence a → is_geometric_sequence_inf a → is_constant_sequence a :=
sorry

theorem sum_power_not_always_arithmetic_or_geometric :
  ∃ (a : ℝ) (S : ℕ → ℝ), (∀ n : ℕ, S n = a^n - 1) ∧
  ¬ (is_arithmetic_sequence S ∨ is_geometric_sequence_inf S) :=
sorry

theorem arithmetic_sequence_no_equal_terms (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a → d ≠ 0 → ∀ m n : ℕ, m ≠ n → a m ≠ a n :=
sorry

end geometric_sum_not_always_geometric_arithmetic_and_geometric_is_constant_sum_power_not_always_arithmetic_or_geometric_arithmetic_sequence_no_equal_terms_l1267_126782


namespace correct_equation_transformation_l1267_126735

theorem correct_equation_transformation (x : ℝ) : 
  3 * x - (2 - 4 * x) = 5 ↔ 3 * x + 4 * x - 2 = 5 := by
  sorry

end correct_equation_transformation_l1267_126735


namespace train_passing_pole_time_l1267_126752

/-- The time it takes for a train to pass a pole given its speed and the time it takes to cross a stationary train of known length -/
theorem train_passing_pole_time (v : ℝ) (t_cross : ℝ) (l_stationary : ℝ) :
  v = 64.8 →
  t_cross = 25 →
  l_stationary = 360 →
  ∃ t_pole : ℝ, abs (t_pole - 19.44) < 0.01 ∧ 
  t_pole = (v * t_cross - l_stationary) / v :=
by sorry

end train_passing_pole_time_l1267_126752


namespace area_ratio_triangle_to_hexagon_l1267_126705

/-- A regular hexagon ABCDEF with vertices A, B, C, D, E, F -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- The area of a regular hexagon -/
def area_hexagon (h : RegularHexagon) : ℝ := sorry

/-- The area of triangle ACE in a regular hexagon -/
def area_triangle_ACE (h : RegularHexagon) : ℝ := sorry

/-- Theorem: The area of triangle ACE is 2/3 of the area of the regular hexagon -/
theorem area_ratio_triangle_to_hexagon (h : RegularHexagon) :
  area_triangle_ACE h / area_hexagon h = 2 / 3 := by sorry

end area_ratio_triangle_to_hexagon_l1267_126705


namespace reflection_of_P_across_y_axis_l1267_126751

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

/-- The original point P -/
def P : Point2D := { x := 4, y := 1 }

/-- Theorem: The reflection of P(4,1) across the y-axis is (-4,1) -/
theorem reflection_of_P_across_y_axis :
  reflectAcrossYAxis P = { x := -4, y := 1 } := by
  sorry


end reflection_of_P_across_y_axis_l1267_126751


namespace trigonometric_identity_l1267_126746

theorem trigonometric_identity : 
  Real.sin (37 * π / 180) * Real.cos (34 * π / 180)^2 + 
  2 * Real.sin (34 * π / 180) * Real.cos (37 * π / 180) * Real.cos (34 * π / 180) - 
  Real.sin (37 * π / 180) * Real.sin (34 * π / 180)^2 = 
  (Real.sqrt 6 + Real.sqrt 2) / 4 := by sorry

end trigonometric_identity_l1267_126746


namespace inequality_equivalence_l1267_126773

theorem inequality_equivalence (x : ℝ) : 
  |x - 2| + |x + 3| < 7 ↔ -4 < x ∧ x < 3 := by
sorry

end inequality_equivalence_l1267_126773


namespace reciprocal_sum_equals_one_l1267_126734

theorem reciprocal_sum_equals_one (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1 / x + 1 / y = 1 := by
sorry

end reciprocal_sum_equals_one_l1267_126734


namespace line_equation_from_triangle_l1267_126726

/-- Given a line passing through (-a, b) and intersecting the y-axis in the second quadrant,
    forming a triangle with area T and base ka along the x-axis, prove that
    the equation of the line is 2Tx - ka²y + ka²b + 2aT = 0 -/
theorem line_equation_from_triangle (a T k : ℝ) (b : ℝ) (hb : b ≠ 0) :
  ∃ (m c : ℝ), 
    (∀ x y, y = m * x + c ↔ 2 * T * x - k * a^2 * y + k * a^2 * b + 2 * a * T = 0) ∧
    m * (-a) + c = b ∧
    m > 0 ∧
    c > 0 ∧
    k > 0 ∧
    T = (1/2) * k * a * (c - b) :=
by sorry

end line_equation_from_triangle_l1267_126726


namespace root_in_interval_implies_m_range_l1267_126748

theorem root_in_interval_implies_m_range (m : ℝ) :
  (∃ x ∈ Set.Icc (-2) 1, 2 * m * x + 4 = 0) →
  m ∈ Set.Iic (-2) ∪ Set.Ici 1 := by
sorry

end root_in_interval_implies_m_range_l1267_126748


namespace green_peaches_per_basket_l1267_126772

/-- Given 7 baskets with a total of 14 green peaches evenly distributed,
    prove that each basket contains 2 green peaches. -/
theorem green_peaches_per_basket :
  ∀ (num_baskets : ℕ) (total_green : ℕ) (green_per_basket : ℕ),
    num_baskets = 7 →
    total_green = 14 →
    total_green = num_baskets * green_per_basket →
    green_per_basket = 2 := by
  sorry

end green_peaches_per_basket_l1267_126772


namespace inequality_equivalence_l1267_126709

theorem inequality_equivalence (x : ℝ) : (x - 5) / 2 + 1 > x - 3 ↔ x < 3 := by
  sorry

end inequality_equivalence_l1267_126709


namespace max_value_of_expression_l1267_126749

theorem max_value_of_expression (x y : Real) 
  (hx : 0 < x ∧ x < π/2) (hy : 0 < y ∧ y < π/2) : 
  (Real.sqrt (Real.sqrt (Real.sin x * Real.sin y))) / 
  (Real.sqrt (Real.sqrt (Real.tan x)) + Real.sqrt (Real.sqrt (Real.tan y))) 
  ≤ Real.sqrt (Real.sqrt 8) / 4 := by
sorry

end max_value_of_expression_l1267_126749


namespace cricket_team_right_handed_players_l1267_126783

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 67)
  (h2 : throwers = 37)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) :
  throwers + (total_players - throwers) * 2 / 3 = 57 := by
  sorry

end cricket_team_right_handed_players_l1267_126783


namespace store_profit_calculation_l1267_126796

/-- Represents the pricing strategy and profit calculation for a store selling turtleneck sweaters -/
theorem store_profit_calculation (C : ℝ) (h : C > 0) :
  let initial_markup := 1.20
  let new_year_markup := 1.25
  let february_discount := 0.80
  let final_price := C * initial_markup * new_year_markup * february_discount
  final_price = 1.20 * C ∧ (final_price - C) / C = 0.20 := by
  sorry

end store_profit_calculation_l1267_126796


namespace equal_payment_payment_difference_l1267_126779

/-- Represents the pizza scenario with given conditions -/
structure PizzaScenario where
  total_slices : ℕ
  meat_slices : ℕ
  plain_cost : ℚ
  meat_cost : ℚ
  joe_meat_slices : ℕ
  joe_veg_slices : ℕ

/-- Calculate the total cost of the pizza -/
def total_cost (p : PizzaScenario) : ℚ :=
  p.plain_cost + p.meat_cost

/-- Calculate the cost per slice -/
def cost_per_slice (p : PizzaScenario) : ℚ :=
  total_cost p / p.total_slices

/-- Calculate Joe's payment -/
def joe_payment (p : PizzaScenario) : ℚ :=
  cost_per_slice p * (p.joe_meat_slices + p.joe_veg_slices)

/-- Calculate Karen's payment -/
def karen_payment (p : PizzaScenario) : ℚ :=
  cost_per_slice p * (p.total_slices - p.joe_meat_slices - p.joe_veg_slices)

/-- The main theorem stating that Joe and Karen paid the same amount -/
theorem equal_payment (p : PizzaScenario) 
  (h1 : p.total_slices = 12)
  (h2 : p.meat_slices = 4)
  (h3 : p.plain_cost = 12)
  (h4 : p.meat_cost = 4)
  (h5 : p.joe_meat_slices = 4)
  (h6 : p.joe_veg_slices = 2) :
  joe_payment p = karen_payment p :=
by sorry

/-- The difference in payment is zero -/
theorem payment_difference (p : PizzaScenario) 
  (h1 : p.total_slices = 12)
  (h2 : p.meat_slices = 4)
  (h3 : p.plain_cost = 12)
  (h4 : p.meat_cost = 4)
  (h5 : p.joe_meat_slices = 4)
  (h6 : p.joe_veg_slices = 2) :
  joe_payment p - karen_payment p = 0 :=
by sorry

end equal_payment_payment_difference_l1267_126779


namespace three_digit_numbers_divisible_by_11_equal_sum_of_squares_of_digits_l1267_126794

/-- A 3-digit number -/
def ThreeDigitNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

/-- The sum of squares of digits of a natural number -/
def SumOfSquaresOfDigits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * hundreds + tens * tens + ones * ones

/-- The main theorem -/
theorem three_digit_numbers_divisible_by_11_equal_sum_of_squares_of_digits :
  ∃! (s : Finset ℕ), s.card = 2 ∧ 
    ∀ n ∈ s, ThreeDigitNumber n ∧ 
             n % 11 = 0 ∧
             n / 11 = SumOfSquaresOfDigits n :=
by sorry

end three_digit_numbers_divisible_by_11_equal_sum_of_squares_of_digits_l1267_126794


namespace abs_and_recip_of_neg_one_point_two_l1267_126768

theorem abs_and_recip_of_neg_one_point_two :
  let x : ℝ := -1.2
  abs x = 1.2 ∧ x⁻¹ = -5/6 := by
  sorry

end abs_and_recip_of_neg_one_point_two_l1267_126768


namespace cookie_count_bounds_l1267_126725

/-- Represents the number of cookies in a package -/
inductive PackageSize
| small : PackageSize  -- 6 cookies
| large : PackageSize  -- 12 cookies

/-- Profit from selling a package -/
def profit : PackageSize → ℕ
| PackageSize.small => 4
| PackageSize.large => 9

/-- Number of cookies in a package -/
def cookiesInPackage : PackageSize → ℕ
| PackageSize.small => 6
| PackageSize.large => 12

/-- Total profit from selling packages -/
def totalProfit : ℕ → ℕ → ℕ := λ x y => x * profit PackageSize.large + y * profit PackageSize.small

/-- Total number of cookies in packages -/
def totalCookies : ℕ → ℕ → ℕ := λ x y => x * cookiesInPackage PackageSize.large + y * cookiesInPackage PackageSize.small

theorem cookie_count_bounds :
  ∃ (x_min y_min x_max y_max : ℕ),
    totalProfit x_min y_min = 219 ∧
    totalProfit x_max y_max = 219 ∧
    totalCookies x_min y_min = 294 ∧
    totalCookies x_max y_max = 324 ∧
    (∀ x y, totalProfit x y = 219 → totalCookies x y ≥ 294 ∧ totalCookies x y ≤ 324) :=
by sorry

end cookie_count_bounds_l1267_126725


namespace liters_conversion_hours_conversion_cubic_meters_conversion_l1267_126757

-- Define conversion factors
def liters_to_milliliters : ℝ := 1000
def hours_per_day : ℝ := 24
def cubic_meters_to_cubic_centimeters : ℝ := 1000000

-- Theorem for 9.12 liters conversion
theorem liters_conversion (x : ℝ) (h : x = 9.12) :
  ∃ (l m : ℝ), x * liters_to_milliliters = l * liters_to_milliliters + m ∧ l = 9 ∧ m = 120 :=
sorry

-- Theorem for 4 hours conversion
theorem hours_conversion (x : ℝ) (h : x = 4) :
  x / hours_per_day = 1 / 6 :=
sorry

-- Theorem for 0.25 cubic meters conversion
theorem cubic_meters_conversion (x : ℝ) (h : x = 0.25) :
  x * cubic_meters_to_cubic_centimeters = 250000 :=
sorry

end liters_conversion_hours_conversion_cubic_meters_conversion_l1267_126757


namespace divisible_by_101_l1267_126711

def repeat_two_digit (n : ℕ) : ℕ :=
  100000 * n + 1000 * n + n

theorem divisible_by_101 (n : ℕ) (h : n < 100) :
  101 ∣ repeat_two_digit n :=
sorry

end divisible_by_101_l1267_126711


namespace percent_of_l_equal_to_75_percent_of_m_l1267_126712

-- Define variables
variable (j k l m : ℝ)

-- Define the conditions
def condition1 : Prop := 1.25 * j = 0.25 * k
def condition2 : Prop := 1.5 * k = 0.5 * l
def condition3 : Prop := 0.2 * m = 7 * j

-- Define the theorem
theorem percent_of_l_equal_to_75_percent_of_m 
  (h1 : condition1 j k)
  (h2 : condition2 k l)
  (h3 : condition3 j m) :
  ∃ x : ℝ, x / 100 * l = 0.75 * m ∧ x = 175 := by
  sorry

end percent_of_l_equal_to_75_percent_of_m_l1267_126712


namespace total_water_volume_is_10750_l1267_126701

def tank1_capacity : ℚ := 7000
def tank2_capacity : ℚ := 5000
def tank3_capacity : ℚ := 3000

def tank1_fill_ratio : ℚ := 3/4
def tank2_fill_ratio : ℚ := 4/5
def tank3_fill_ratio : ℚ := 1/2

def total_water_volume : ℚ := 
  tank1_capacity * tank1_fill_ratio + 
  tank2_capacity * tank2_fill_ratio + 
  tank3_capacity * tank3_fill_ratio

theorem total_water_volume_is_10750 : total_water_volume = 10750 := by
  sorry

end total_water_volume_is_10750_l1267_126701


namespace mary_money_left_l1267_126792

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def money_left (p : ℝ) : ℝ :=
  let initial_money := 50
  let drink_cost := p
  let medium_pizza_cost := 3 * p
  let large_pizza_cost := 5 * p
  let num_drinks := 4
  let num_medium_pizzas := 3
  let num_large_pizzas := 2
  initial_money - (num_drinks * drink_cost + num_medium_pizzas * medium_pizza_cost + num_large_pizzas * large_pizza_cost)

/-- Theorem stating that Mary has 50 - 23p dollars left after her purchases -/
theorem mary_money_left (p : ℝ) : money_left p = 50 - 23 * p := by
  sorry

end mary_money_left_l1267_126792


namespace no_rational_solutions_l1267_126714

theorem no_rational_solutions (n : ℕ) (x y : ℚ) : (x + Real.sqrt 3 * y) ^ n ≠ Real.sqrt (1 + Real.sqrt 3) := by
  sorry

end no_rational_solutions_l1267_126714


namespace set_A_enumeration_l1267_126775

def A : Set ℚ := {z | ∃ p q : ℕ+, z = p / q ∧ p + q = 5}

theorem set_A_enumeration : A = {1/4, 2/3, 3/2, 4} := by
  sorry

end set_A_enumeration_l1267_126775


namespace abs_a_minus_b_equals_eight_l1267_126732

theorem abs_a_minus_b_equals_eight (a b : ℝ) (h1 : a * b = 9) (h2 : a + b = 10) : 
  |a - b| = 8 := by
sorry

end abs_a_minus_b_equals_eight_l1267_126732


namespace tarun_worked_days_l1267_126788

/-- Represents the number of days it takes for Arun and Tarun to complete the work together -/
def combined_days : ℝ := 10

/-- Represents the number of days it takes for Arun to complete the work alone -/
def arun_alone_days : ℝ := 60

/-- Represents the number of days Arun worked alone after Tarun left -/
def arun_remaining_days : ℝ := 36

/-- Represents the total amount of work to be done -/
def total_work : ℝ := 1

/-- Theorem stating that Tarun worked for 4 days before leaving -/
theorem tarun_worked_days : 
  ∃ (t : ℝ), 
    t > 0 ∧ 
    t < combined_days ∧ 
    (t / combined_days + arun_remaining_days / arun_alone_days = total_work) ∧ 
    t = 4 := by
  sorry


end tarun_worked_days_l1267_126788


namespace sum_product_theorem_l1267_126758

def number_list : List ℕ := [2, 3, 4, 6]

theorem sum_product_theorem :
  ∃! (subset : Finset ℕ),
    subset.card = 3 ∧ 
    (∀ x ∈ subset, x ∈ number_list) ∧
    (subset.sum id = 11) ∧
    (subset.prod id = 36) :=
sorry

end sum_product_theorem_l1267_126758


namespace largest_whole_number_less_than_120_over_8_l1267_126727

theorem largest_whole_number_less_than_120_over_8 : 
  (∀ n : ℕ, n > 14 → 8 * n ≥ 120) ∧ (8 * 14 < 120) := by
  sorry

end largest_whole_number_less_than_120_over_8_l1267_126727


namespace spencer_walk_distance_l1267_126724

/-- The total distance Spencer walked on his errands -/
def total_distance (house_to_library : ℝ) (library_to_post : ℝ) (post_to_grocery : ℝ) (grocery_to_coffee : ℝ) (coffee_to_house : ℝ) : ℝ :=
  house_to_library + library_to_post + post_to_grocery + grocery_to_coffee + coffee_to_house

/-- Theorem stating that Spencer walked 6.1 miles in total -/
theorem spencer_walk_distance :
  total_distance 1.2 0.8 1.5 0.6 2 = 6.1 := by
  sorry


end spencer_walk_distance_l1267_126724


namespace john_earnings_proof_l1267_126762

def hours_per_workday : ℕ := 12
def days_in_month : ℕ := 30
def former_hourly_wage : ℚ := 20
def raise_percentage : ℚ := 30 / 100

def john_monthly_earnings : ℚ :=
  (days_in_month / 2) * hours_per_workday * (former_hourly_wage * (1 + raise_percentage))

theorem john_earnings_proof :
  john_monthly_earnings = 4680 := by
  sorry

end john_earnings_proof_l1267_126762


namespace opposite_of_negative_two_l1267_126702

theorem opposite_of_negative_two : 
  (∃ x : ℝ, -2 + x = 0) → (∃ x : ℝ, -2 + x = 0 ∧ x = 2) :=
by sorry

end opposite_of_negative_two_l1267_126702


namespace square_area_error_l1267_126731

theorem square_area_error (x : ℝ) (h : x > 0) :
  let actual_edge := x
  let calculated_edge := x * (1 + 0.02)
  let actual_area := x^2
  let calculated_area := calculated_edge^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.0404 := by sorry

end square_area_error_l1267_126731


namespace club_officer_selection_l1267_126754

/-- Represents the number of ways to choose officers in a club --/
def choose_officers (total_members : ℕ) (founding_members : ℕ) (positions : ℕ) : ℕ :=
  founding_members * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

/-- Theorem stating the number of ways to choose officers in the given scenario --/
theorem club_officer_selection :
  choose_officers 12 4 5 = 25920 := by
  sorry

end club_officer_selection_l1267_126754


namespace unique_number_triple_and_square_l1267_126741

theorem unique_number_triple_and_square (x : ℝ) : 
  (x > 0 ∧ 3 * x = (x / 2)^2 + 45) ↔ x = 18 := by
  sorry

end unique_number_triple_and_square_l1267_126741


namespace research_budget_allocation_l1267_126770

theorem research_budget_allocation (microphotonics : ℝ) (home_electronics : ℝ)
  (genetically_modified_microorganisms : ℝ) (industrial_lubricants : ℝ)
  (basic_astrophysics_degrees : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  genetically_modified_microorganisms = 29 →
  industrial_lubricants = 8 →
  basic_astrophysics_degrees = 18 →
  ∃ (food_additives : ℝ),
    food_additives = 20 ∧
    microphotonics + home_electronics + genetically_modified_microorganisms +
    industrial_lubricants + (basic_astrophysics_degrees / 360 * 100) + food_additives = 100 :=
by sorry

end research_budget_allocation_l1267_126770


namespace arithmetic_sequence_problem_l1267_126763

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℚ) := ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 7 + a 9 = 16) 
  (h_fourth : a 4 = 1) : 
  a 12 = 15 := by
sorry


end arithmetic_sequence_problem_l1267_126763


namespace sin_negative_225_degrees_l1267_126719

theorem sin_negative_225_degrees :
  Real.sin (-(225 * π / 180)) = Real.sqrt 2 / 2 := by
  sorry

end sin_negative_225_degrees_l1267_126719


namespace complement_union_theorem_l1267_126737

def U : Finset Nat := {0, 1, 2, 3, 4, 5}
def A : Finset Nat := {1, 2, 3, 5}
def B : Finset Nat := {2, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 4} := by sorry

end complement_union_theorem_l1267_126737


namespace simplify_and_evaluate_l1267_126766

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by sorry

end simplify_and_evaluate_l1267_126766


namespace circle_C_properties_l1267_126721

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 2}

-- Define points A and B
def point_A : ℝ × ℝ := (4, 1)
def point_B : ℝ × ℝ := (2, 1)

-- Define the line x - y - 1 = 0
def tangent_line (p : ℝ × ℝ) : Prop := p.1 - p.2 - 1 = 0

-- Theorem stating the properties of circle C
theorem circle_C_properties :
  point_A ∈ circle_C ∧
  point_B ∈ circle_C ∧
  tangent_line point_B ∧
  (∀ p ∈ circle_C, (p.1 - 3)^2 + p.2^2 = 2) ∧
  (3, 0) ∈ circle_C ∧
  (∀ p ∈ circle_C, (p.1 - 3)^2 + p.2^2 = 2) :=
by
  sorry

#check circle_C_properties

end circle_C_properties_l1267_126721


namespace abcd_product_magnitude_l1267_126778

theorem abcd_product_magnitude (a b c d : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  a^2 + 1/b = b^2 + 1/c → b^2 + 1/c = c^2 + 1/d → c^2 + 1/d = d^2 + 1/a →
  |a*b*c*d| = 1 := by
sorry

end abcd_product_magnitude_l1267_126778


namespace certain_number_proof_l1267_126704

theorem certain_number_proof : ∃ x : ℝ, (0.7 * x = 0.4 * 1050) ∧ (x = 600) := by
  sorry

end certain_number_proof_l1267_126704


namespace arithmetic_geometric_mean_inequality_l1267_126767

theorem arithmetic_geometric_mean_inequality (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x + y + z) / 3 ≥ (x * y * z) ^ (1/3) := by
  sorry

end arithmetic_geometric_mean_inequality_l1267_126767


namespace john_shoe_purchase_cost_l1267_126797

/-- Calculate the total cost including tax for two items -/
def total_cost (price1 : ℝ) (price2 : ℝ) (tax_rate : ℝ) : ℝ :=
  let total_before_tax := price1 + price2
  let tax_amount := total_before_tax * tax_rate
  total_before_tax + tax_amount

/-- Theorem stating the total cost for the given problem -/
theorem john_shoe_purchase_cost :
  total_cost 150 120 0.1 = 297 := by
  sorry

end john_shoe_purchase_cost_l1267_126797


namespace sum_of_fourth_powers_is_square_l1267_126776

theorem sum_of_fourth_powers_is_square (a b c : ℤ) (h : a + b + c = 0) :
  2 * (a^4 + b^4 + c^4) = (a^2 + b^2 + c^2)^2 := by
  sorry

end sum_of_fourth_powers_is_square_l1267_126776


namespace solution_set_quadratic_inequality_l1267_126777

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, x * (x - 2) ≤ 0 ↔ 0 ≤ x ∧ x ≤ 2 := by sorry

end solution_set_quadratic_inequality_l1267_126777


namespace point_on_linear_function_l1267_126740

/-- Given that point P(a, b) is on the graph of y = -2x + 3, prove that 2a + b - 2 = 1 -/
theorem point_on_linear_function (a b : ℝ) (h : b = -2 * a + 3) : 2 * a + b - 2 = 1 := by
  sorry

end point_on_linear_function_l1267_126740


namespace triangle_inequality_l1267_126744

/-- Given three line segments of lengths a, 2, and 6, they can form a triangle if and only if 4 < a < 8 -/
theorem triangle_inequality (a : ℝ) : 
  (∃ (x y z : ℝ), x = a ∧ y = 2 ∧ z = 6 ∧ x + y > z ∧ y + z > x ∧ z + x > y) ↔ 4 < a ∧ a < 8 :=
sorry

end triangle_inequality_l1267_126744


namespace minimum_score_raises_average_l1267_126769

def scores : List ℕ := [92, 88, 74, 65, 80]

def current_average : ℚ := (scores.sum : ℚ) / scores.length

def target_average : ℚ := current_average + 5

def minimum_score : ℕ := 110

theorem minimum_score_raises_average : 
  (((scores.sum + minimum_score) : ℚ) / (scores.length + 1)) = target_average := by sorry

end minimum_score_raises_average_l1267_126769


namespace solution_to_system_of_equations_l1267_126738

theorem solution_to_system_of_equations :
  ∃! (x y : ℝ), (2*x + 3*y = (6-x) + (6-3*y)) ∧ (x - 2*y = (x-2) - (y+2)) ∧ x = -4 ∧ y = 4 := by
  sorry

end solution_to_system_of_equations_l1267_126738


namespace g_of_2_equals_6_l1267_126733

/-- The function g defined as g(x) = x³ - 2 for all real x -/
def g (x : ℝ) : ℝ := x^3 - 2

/-- Theorem stating that g(2) = 6 -/
theorem g_of_2_equals_6 : g 2 = 6 := by
  sorry

end g_of_2_equals_6_l1267_126733


namespace faucet_filling_time_l1267_126703

/-- Given that four faucets can fill a 150-gallon tub in 8 minutes,
    prove that eight faucets will fill a 50-gallon tub in 4/3 minutes. -/
theorem faucet_filling_time 
  (volume_large : ℝ) 
  (volume_small : ℝ)
  (time_large : ℝ)
  (faucets_large : ℕ)
  (faucets_small : ℕ)
  (h1 : volume_large = 150)
  (h2 : volume_small = 50)
  (h3 : time_large = 8)
  (h4 : faucets_large = 4)
  (h5 : faucets_small = 8) :
  (volume_small * time_large * faucets_large) / (volume_large * faucets_small) = 4/3 := by
  sorry

end faucet_filling_time_l1267_126703


namespace fourth_root_power_eight_l1267_126722

theorem fourth_root_power_eight : (((5 ^ (1/2)) ^ 5) ^ (1/4)) ^ 8 = 3125 := by
  sorry

end fourth_root_power_eight_l1267_126722


namespace empty_solution_set_range_l1267_126718

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 4| + |3 - x| < a)) → a ∈ Set.Iic 1 := by
  sorry

end empty_solution_set_range_l1267_126718


namespace min_rubles_to_win_l1267_126720

/-- Represents the state of the game --/
structure GameState :=
  (points : ℕ)
  (rubles : ℕ)

/-- Applies a move to the game state --/
def applyMove (state : GameState) (move : Bool) : GameState :=
  if move
  then { points := state.points * 2, rubles := state.rubles + 2 }
  else { points := state.points + 1, rubles := state.rubles + 1 }

/-- Checks if the game state is valid (not exceeding 50 points) --/
def isValidState (state : GameState) : Bool :=
  state.points <= 50

/-- Checks if the game is won (exactly 50 points) --/
def isWinningState (state : GameState) : Bool :=
  state.points = 50

/-- Theorem: The minimum number of rubles to win the game is 11 --/
theorem min_rubles_to_win :
  ∃ (moves : List Bool),
    let finalState := moves.foldl applyMove { points := 0, rubles := 0 }
    isWinningState finalState ∧
    finalState.rubles = 11 ∧
    (∀ (otherMoves : List Bool),
      let otherFinalState := otherMoves.foldl applyMove { points := 0, rubles := 0 }
      isWinningState otherFinalState →
      otherFinalState.rubles ≥ 11) :=
by
  sorry

end min_rubles_to_win_l1267_126720


namespace least_months_to_triple_l1267_126756

/-- The initial borrowed amount in dollars -/
def initial_amount : ℝ := 2000

/-- The monthly interest rate as a decimal -/
def monthly_rate : ℝ := 0.04

/-- The function that calculates the owed amount after t months -/
def owed_amount (t : ℕ) : ℝ := initial_amount * (1 + monthly_rate) ^ t

/-- Theorem stating that 30 is the least integer number of months 
    after which the owed amount exceeds three times the initial amount -/
theorem least_months_to_triple : 
  (∀ k : ℕ, k < 30 → owed_amount k ≤ 3 * initial_amount) ∧ 
  (owed_amount 30 > 3 * initial_amount) := by
  sorry

#check least_months_to_triple

end least_months_to_triple_l1267_126756


namespace triangle_side_sum_l1267_126760

theorem triangle_side_sum (side_length : ℚ) (h : side_length = 14/8) : 
  3 * side_length = 21/4 := by sorry

end triangle_side_sum_l1267_126760


namespace ellipse_constraint_l1267_126747

/-- An ellipse passing through (2,1) with |y| > 1 -/
def EllipseWithConstraint (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (4 / a^2 + 1 / b^2 = 1)

theorem ellipse_constraint (a b : ℝ) (h : EllipseWithConstraint a b) :
  {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1 ∧ |p.2| > 1} =
  {p : ℝ × ℝ | p.1^2 + p.2^2 < 5 ∧ |p.2| > 1} := by
  sorry

end ellipse_constraint_l1267_126747


namespace number117_is_1983_l1267_126750

/-- The set of digits used to form the four-digit numbers -/
def digits : Finset Nat := {1, 3, 4, 5, 7, 8, 9}

/-- A four-digit number formed from the given digits without repetition -/
structure FourDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  h1 : d1 ∈ digits
  h2 : d2 ∈ digits
  h3 : d3 ∈ digits
  h4 : d4 ∈ digits
  h5 : d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4

/-- The value of a four-digit number -/
def FourDigitNumber.value (n : FourDigitNumber) : Nat :=
  1000 * n.d1 + 100 * n.d2 + 10 * n.d3 + n.d4

/-- The set of all valid four-digit numbers -/
def validNumbers : Finset FourDigitNumber := sorry

/-- The 117th number in the ascending sequence of valid four-digit numbers -/
def number117 : FourDigitNumber := sorry

theorem number117_is_1983 : number117.value = 1983 := by sorry

end number117_is_1983_l1267_126750


namespace largest_c_for_quadratic_range_l1267_126753

theorem largest_c_for_quadratic_range (c : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + c = 2) ↔ c ≤ 11 :=
sorry

end largest_c_for_quadratic_range_l1267_126753


namespace consecutive_squares_sum_l1267_126759

theorem consecutive_squares_sum (n : ℕ) (h : n = 26) :
  (n - 1)^2 + n^2 + (n + 1)^2 = 2030 := by
  sorry

end consecutive_squares_sum_l1267_126759


namespace hyperbola_eccentricity_theorem_l1267_126764

/-- Hyperbola eccentricity theorem -/
theorem hyperbola_eccentricity_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  (b / a ≥ Real.sqrt 3) → e ≥ 2 := by
  sorry

end hyperbola_eccentricity_theorem_l1267_126764


namespace quadratic_roots_condition_l1267_126745

theorem quadratic_roots_condition (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) ↔ c < (1/4 : ℝ) :=
by sorry

end quadratic_roots_condition_l1267_126745


namespace three_zeros_condition_l1267_126743

/-- The cubic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- Theorem stating the condition for f to have exactly 3 real zeros -/
theorem three_zeros_condition (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    ∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) ↔ 
  a < -3 :=
sorry

end three_zeros_condition_l1267_126743


namespace proportion_equality_l1267_126790

theorem proportion_equality (x : ℝ) : (x / 5 = 1.2 / 8) → x = 0.75 := by
  sorry

end proportion_equality_l1267_126790


namespace initial_flea_distance_l1267_126798

/-- Represents a flea's position on a 2D plane -/
structure FleaPosition where
  x : ℝ
  y : ℝ

/-- Represents the jump pattern of a flea -/
inductive JumpDirection
  | Right
  | Up
  | Left
  | Down

/-- Calculates the position of a flea after n jumps -/
def flea_position_after_jumps (initial_pos : FleaPosition) (direction : JumpDirection) (n : ℕ) : FleaPosition :=
  sorry

/-- Calculates the distance between two points on a 2D plane -/
def distance (p1 p2 : FleaPosition) : ℝ :=
  sorry

/-- Theorem stating the initial distance between the fleas -/
theorem initial_flea_distance (flea1_start flea2_start : FleaPosition)
  (h1 : flea_position_after_jumps flea1_start JumpDirection.Right 100 = 
        FleaPosition.mk (flea1_start.x - 50) (flea1_start.y - 50))
  (h2 : flea_position_after_jumps flea2_start JumpDirection.Left 100 = 
        FleaPosition.mk (flea2_start.x + 50) (flea2_start.y - 50))
  (h3 : distance (flea_position_after_jumps flea1_start JumpDirection.Right 100)
                 (flea_position_after_jumps flea2_start JumpDirection.Left 100) = 300) :
  distance flea1_start flea2_start = 2 :=
sorry

end initial_flea_distance_l1267_126798


namespace A_is_irrational_l1267_126785

/-- The sequence of consecutive prime numbers -/
def consecutive_primes : ℕ → ℕ := sorry

/-- The decimal representation of our number -/
def A : ℝ := sorry

/-- Dirichlet's theorem on arithmetic progressions -/
axiom dirichlet_theorem : ∃ (infinitely_many : Set ℕ), ∀ p ∈ infinitely_many, 
  ∃ (n x : ℕ), p = 10^(n+1) * x + 1 ∧ Prime p

/-- The main theorem: A is irrational -/
theorem A_is_irrational : Irrational A := sorry

end A_is_irrational_l1267_126785


namespace balls_picked_is_two_l1267_126781

/-- The number of red balls in the bag -/
def red_balls : ℕ := 3

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 2

/-- The number of green balls in the bag -/
def green_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := red_balls + blue_balls + green_balls

/-- The probability of picking two red balls -/
def prob_two_red : ℚ := 3 / 28

/-- The number of balls picked at random -/
def balls_picked : ℕ := 2

/-- Theorem stating that the number of balls picked is 2 given the conditions -/
theorem balls_picked_is_two :
  (red_balls = 3 ∧ blue_balls = 2 ∧ green_balls = 3) →
  (prob_two_red = 3 / 28) →
  (balls_picked = 2) := by sorry

end balls_picked_is_two_l1267_126781


namespace range_of_a_in_second_quadrant_l1267_126761

/-- A complex number z = x + yi is in the second quadrant if x < 0 and y > 0 -/
def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem range_of_a_in_second_quadrant (a : ℝ) :
  is_in_second_quadrant ((a - 2) + (a + 1) * I) ↔ -1 < a ∧ a < 2 := by
  sorry

end range_of_a_in_second_quadrant_l1267_126761


namespace theater_line_arrangements_l1267_126742

def number_of_people : ℕ := 8
def number_of_fixed_group : ℕ := 3

theorem theater_line_arrangements :
  (number_of_people - number_of_fixed_group + 1).factorial = 720 := by
  sorry

end theater_line_arrangements_l1267_126742


namespace division_multiplication_problem_l1267_126789

theorem division_multiplication_problem : (0.45 / 0.005) * 2 = 180 := by
  sorry

end division_multiplication_problem_l1267_126789


namespace m_range_l1267_126730

def f (x : ℝ) : ℝ := -x^3 - 2*x^2 + 4*x

theorem m_range (m : ℝ) : 
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ m^2 - 14*m) → 
  m ∈ Set.Icc 3 11 := by
  sorry

end m_range_l1267_126730


namespace line_l_equation_l1267_126736

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y - 1 = 0
def l₂ (x y : ℝ) : Prop := x + y + 2 = 0

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define the property of l passing through P
def passes_through_P (l : ℝ → ℝ → Prop) : Prop := l P.1 P.2

-- Define the intersection points A and B
def A (l : ℝ → ℝ → Prop) : ℝ × ℝ := sorry
def B (l : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

-- Define the property of P being the midpoint of AB
def P_is_midpoint (l : ℝ → ℝ → Prop) : Prop :=
  P.1 = (A l).1 / 2 + (B l).1 / 2 ∧ P.2 = (A l).2 / 2 + (B l).2 / 2

-- Define the property of A and B being on l₁ and l₂ respectively
def A_on_l₁ (l : ℝ → ℝ → Prop) : Prop := l₁ (A l).1 (A l).2
def B_on_l₂ (l : ℝ → ℝ → Prop) : Prop := l₂ (B l).1 (B l).2

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := 4 * x - y - 7 = 0

theorem line_l_equation : 
  ∀ l : ℝ → ℝ → Prop, 
    passes_through_P l → 
    P_is_midpoint l → 
    A_on_l₁ l → 
    B_on_l₂ l → 
    ∀ x y : ℝ, l x y ↔ line_l x y :=
sorry

end line_l_equation_l1267_126736


namespace rectangle_inscribed_area_bound_l1267_126706

/-- A triangle represented by three points in the plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A rectangle represented by four points in the plane -/
structure Rectangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ

/-- Function to calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Function to calculate the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := sorry

/-- Predicate to check if a rectangle is inscribed in a triangle -/
def isInscribed (r : Rectangle) (t : Triangle) : Prop := sorry

/-- Theorem: The area of a rectangle inscribed in a triangle does not exceed half of the area of the triangle -/
theorem rectangle_inscribed_area_bound (t : Triangle) (r : Rectangle) :
  isInscribed r t → rectangleArea r ≤ (1/2) * triangleArea t := by
  sorry

end rectangle_inscribed_area_bound_l1267_126706


namespace inscribed_circle_radius_bound_l1267_126707

/-- The radius of a circle inscribed in a quadrilateral with sides 3, 6, 5, and 8 is less than 3 -/
theorem inscribed_circle_radius_bound (r : ℝ) : 
  r > 0 → -- r is positive (radius)
  r * 11 = 12 * Real.sqrt 5 → -- area formula: S = r * s, where s = (3 + 6 + 5 + 8) / 2 = 11
  r < 3 := by
sorry

end inscribed_circle_radius_bound_l1267_126707


namespace common_root_of_quadratic_equations_l1267_126765

theorem common_root_of_quadratic_equations (a b x : ℝ) :
  (x^2 + 2019*a*x + b = 0) ∧
  (x^2 + 2019*b*x + a = 0) ∧
  (a ≠ b) →
  x = 1/2019 :=
by sorry

end common_root_of_quadratic_equations_l1267_126765
