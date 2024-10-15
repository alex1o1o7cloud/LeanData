import Mathlib

namespace NUMINAMATH_CALUDE_rectangular_plot_length_l1803_180325

/-- The length of a rectangular plot in meters -/
def length : ℝ := 55

/-- The breadth of a rectangular plot in meters -/
def breadth : ℝ := 45

/-- The cost of fencing per meter in rupees -/
def cost_per_meter : ℝ := 26.50

/-- The total cost of fencing in rupees -/
def total_cost : ℝ := 5300

/-- Theorem stating the length of the rectangular plot -/
theorem rectangular_plot_length :
  (length = breadth + 10) ∧
  (total_cost = cost_per_meter * (2 * (length + breadth))) →
  length = 55 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l1803_180325


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_retailer_profit_is_65_percent_l1803_180330

/-- Calculates the profit percentage for a retailer selling pens -/
theorem retailer_profit_percentage 
  (num_pens : ℕ) 
  (cost_price : ℝ) 
  (market_price : ℝ) 
  (discount_rate : ℝ) : ℝ :=
  let selling_price := num_pens * (market_price * (1 - discount_rate))
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage

/-- Proves that the retailer's profit percentage is 65% under given conditions -/
theorem retailer_profit_is_65_percent : 
  retailer_profit_percentage 60 36 1 0.01 = 65 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percentage_retailer_profit_is_65_percent_l1803_180330


namespace NUMINAMATH_CALUDE_kim_shirts_left_l1803_180329

def initial_shirts : ℚ := 4.5 * 12
def bought_shirts : ℕ := 7
def lost_shirts : ℕ := 2
def fraction_given : ℚ := 2 / 5

theorem kim_shirts_left : 
  let total_before_giving := initial_shirts + bought_shirts - lost_shirts
  let given_to_sister := ⌊fraction_given * total_before_giving⌋
  total_before_giving - given_to_sister = 36 := by
  sorry

end NUMINAMATH_CALUDE_kim_shirts_left_l1803_180329


namespace NUMINAMATH_CALUDE_quadratic_roots_bounds_l1803_180359

theorem quadratic_roots_bounds (m : ℝ) (x₁ x₂ : ℝ) 
  (hm : m < 0)
  (hroots : x₁^2 - x₁ - 6 = m ∧ x₂^2 - x₂ - 6 = m)
  (horder : x₁ < x₂) :
  -2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_bounds_l1803_180359


namespace NUMINAMATH_CALUDE_x_value_l1803_180335

theorem x_value : ∀ x : ℕ, x = 225 + 2 * 15 * 9 + 81 → x = 576 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1803_180335


namespace NUMINAMATH_CALUDE_leg_ratio_is_sqrt_seven_l1803_180334

/-- Configuration of squares and triangles -/
structure SquareTriangleConfig where
  /-- Side length of the inner square -/
  s : ℝ
  /-- Length of the shorter leg of each triangle -/
  a : ℝ
  /-- Length of the longer leg of each triangle -/
  b : ℝ
  /-- Side length of the outer square -/
  t : ℝ
  /-- The triangles are right triangles -/
  triangle_right : a^2 + b^2 = t^2
  /-- The area of the outer square is twice the area of the inner square -/
  area_relation : t^2 = 2 * s^2
  /-- The shorter legs of two triangles form one side of the inner square -/
  inner_side : 2 * a = s

/-- The ratio of the longer leg to the shorter leg is √7 -/
theorem leg_ratio_is_sqrt_seven (config : SquareTriangleConfig) :
  config.b / config.a = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_leg_ratio_is_sqrt_seven_l1803_180334


namespace NUMINAMATH_CALUDE_other_asymptote_equation_l1803_180339

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One of the asymptotes of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- x-coordinate of the foci -/
  foci_x : ℝ

/-- Theorem: Given a hyperbola with one asymptote y = 4x - 3 and foci with x-coordinate 3,
    the equation of the other asymptote is y = -4x + 21 -/
theorem other_asymptote_equation (h : Hyperbola) 
    (h1 : h.asymptote1 = fun x ↦ 4 * x - 3) 
    (h2 : h.foci_x = 3) : 
    ∃ asymptote2 : ℝ → ℝ, asymptote2 = fun x ↦ -4 * x + 21 := by
  sorry

end NUMINAMATH_CALUDE_other_asymptote_equation_l1803_180339


namespace NUMINAMATH_CALUDE_expression_factorization_l1803_180355

theorem expression_factorization (x : ℝ) :
  (20 * x^3 + 100 * x - 10) - (-5 * x^3 + 5 * x - 10) = 5 * x * (5 * x^2 + 19) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1803_180355


namespace NUMINAMATH_CALUDE_abhays_speed_l1803_180316

/-- Proves that Abhay's speed is 10.5 km/h given the problem conditions --/
theorem abhays_speed (distance : ℝ) (abhay_speed sameer_speed : ℝ) 
  (h1 : distance = 42)
  (h2 : distance / abhay_speed = distance / sameer_speed + 2)
  (h3 : distance / (2 * abhay_speed) = distance / sameer_speed - 1) :
  abhay_speed = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_abhays_speed_l1803_180316


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l1803_180363

theorem quadratic_is_square_of_binomial :
  ∃ (r s : ℝ), (225/16 : ℝ) * x^2 + 15 * x + 4 = (r * x + s)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l1803_180363


namespace NUMINAMATH_CALUDE_ramanujan_number_l1803_180306

/-- Given Hardy's complex number and the product of Hardy's and Ramanujan's numbers,
    prove that Ramanujan's number is 144/25 + 8/25i. -/
theorem ramanujan_number (h r : ℂ) : 
  h = 3 + 4*I ∧ r * h = 16 + 24*I → r = 144/25 + 8/25*I := by
  sorry

end NUMINAMATH_CALUDE_ramanujan_number_l1803_180306


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1803_180357

/-- Given a triangle in the Cartesian plane with vertices (a, d), (b, e), and (c, f),
    if the sum of x-coordinates (a + b + c) is 15 and the sum of y-coordinates (d + e + f) is 12,
    then the sum of x-coordinates of the midpoints of its sides is 15 and
    the sum of y-coordinates of the midpoints of its sides is 12. -/
theorem midpoint_coordinate_sum (a b c d e f : ℝ) 
  (h1 : a + b + c = 15) (h2 : d + e + f = 12) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 ∧ 
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 12 := by
  sorry


end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1803_180357


namespace NUMINAMATH_CALUDE_sin_960_degrees_l1803_180372

theorem sin_960_degrees : Real.sin (960 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_960_degrees_l1803_180372


namespace NUMINAMATH_CALUDE_rectangles_on_specific_grid_l1803_180309

/-- Represents a grid with specified dimensions and properties. -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (unit_distance : ℕ)
  (allow_diagonals : Bool)

/-- Counts the number of rectangles that can be formed on the grid. -/
def count_rectangles (g : Grid) : ℕ := sorry

/-- The specific 3x3 grid with 2-unit spacing and allowed diagonals. -/
def specific_grid : Grid :=
  { rows := 3
  , cols := 3
  , unit_distance := 2
  , allow_diagonals := true }

/-- Theorem stating that the number of rectangles on the specific grid is 60. -/
theorem rectangles_on_specific_grid :
  count_rectangles specific_grid = 60 := by sorry

end NUMINAMATH_CALUDE_rectangles_on_specific_grid_l1803_180309


namespace NUMINAMATH_CALUDE_final_balance_approx_l1803_180314

/-- Calculates the final amount in Steve's bank account after 5 years --/
def bank_account_balance : ℝ := 
  let initial_deposit : ℝ := 100
  let interest_rate_1 : ℝ := 0.1  -- 10% for first 3 years
  let interest_rate_2 : ℝ := 0.08 -- 8% for next 2 years
  let deposit_1 : ℝ := 10 -- annual deposit for first 2 years
  let deposit_2 : ℝ := 15 -- annual deposit for remaining 3 years
  let year_1 : ℝ := initial_deposit * (1 + interest_rate_1) + deposit_1
  let year_2 : ℝ := year_1 * (1 + interest_rate_1) + deposit_1
  let year_3 : ℝ := year_2 * (1 + interest_rate_1) + deposit_2
  let year_4 : ℝ := year_3 * (1 + interest_rate_2) + deposit_2
  let year_5 : ℝ := year_4 * (1 + interest_rate_2) + deposit_2
  year_5

/-- The final balance in Steve's bank account after 5 years is approximately $230.89 --/
theorem final_balance_approx : 
  ∃ ε > 0, |bank_account_balance - 230.89| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_final_balance_approx_l1803_180314


namespace NUMINAMATH_CALUDE_parabola_vertex_l1803_180308

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = (x - 2)^2 + 1

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 1)

/-- Theorem: The vertex of the parabola y = (x-2)^2 + 1 is (2, 1) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1803_180308


namespace NUMINAMATH_CALUDE_solid_identification_l1803_180310

-- Define the structure of a solid
structure Solid :=
  (faces : Nat)
  (hasParallelCongruentHexagons : Bool)
  (hasRectangularFaces : Bool)
  (hasSquareFace : Bool)
  (hasCongruentTriangles : Bool)
  (hasCommonVertex : Bool)

-- Define the types of solids
inductive SolidType
  | RegularHexagonalPrism
  | RegularSquarePyramid
  | Other

-- Function to determine the type of solid based on its structure
def identifySolid (s : Solid) : SolidType :=
  if s.faces == 8 && s.hasParallelCongruentHexagons && s.hasRectangularFaces then
    SolidType.RegularHexagonalPrism
  else if s.faces == 5 && s.hasSquareFace && s.hasCongruentTriangles && s.hasCommonVertex then
    SolidType.RegularSquarePyramid
  else
    SolidType.Other

-- Theorem stating that the given descriptions correspond to the correct solid types
theorem solid_identification :
  (∀ s : Solid, s.faces = 8 ∧ s.hasParallelCongruentHexagons ∧ s.hasRectangularFaces →
    identifySolid s = SolidType.RegularHexagonalPrism) ∧
  (∀ s : Solid, s.faces = 5 ∧ s.hasSquareFace ∧ s.hasCongruentTriangles ∧ s.hasCommonVertex →
    identifySolid s = SolidType.RegularSquarePyramid) :=
by sorry


end NUMINAMATH_CALUDE_solid_identification_l1803_180310


namespace NUMINAMATH_CALUDE_customer_departure_l1803_180397

theorem customer_departure (initial_customers : Real) 
  (second_departure : Real) (final_customers : Real) :
  initial_customers = 36.0 →
  second_departure = 14.0 →
  final_customers = 3 →
  ∃ (first_departure : Real),
    initial_customers - first_departure - second_departure = final_customers ∧
    first_departure = 19.0 := by
  sorry

end NUMINAMATH_CALUDE_customer_departure_l1803_180397


namespace NUMINAMATH_CALUDE_tims_children_treats_l1803_180371

/-- The total number of treats Tim's children get while trick or treating --/
def total_treats (num_children : ℕ) (hours : ℕ) (houses_per_hour : ℕ) (treats_per_kid : ℕ) : ℕ :=
  num_children * hours * houses_per_hour * treats_per_kid

/-- Theorem stating that Tim's children get 180 treats in total --/
theorem tims_children_treats :
  total_treats 3 4 5 3 = 180 := by
  sorry

#eval total_treats 3 4 5 3

end NUMINAMATH_CALUDE_tims_children_treats_l1803_180371


namespace NUMINAMATH_CALUDE_distance_travelled_l1803_180352

theorem distance_travelled (normal_speed : ℝ) (faster_speed : ℝ) (additional_distance : ℝ) :
  normal_speed = 10 →
  faster_speed = 14 →
  additional_distance = 20 →
  (∃ (actual_distance : ℝ), 
    actual_distance / normal_speed = (actual_distance + additional_distance) / faster_speed ∧
    actual_distance = 50) :=
by sorry

end NUMINAMATH_CALUDE_distance_travelled_l1803_180352


namespace NUMINAMATH_CALUDE_gas_price_difference_l1803_180377

/-- Represents the price difference per gallon between two states --/
def price_difference (nc_price va_price : ℚ) : ℚ := va_price - nc_price

/-- Proves the price difference per gallon between Virginia and North Carolina --/
theorem gas_price_difference 
  (nc_gallons va_gallons : ℚ) 
  (nc_price : ℚ) 
  (total_spent : ℚ) :
  nc_gallons = 10 →
  va_gallons = 10 →
  nc_price = 2 →
  total_spent = 50 →
  price_difference nc_price ((total_spent - nc_gallons * nc_price) / va_gallons) = 1 := by
  sorry


end NUMINAMATH_CALUDE_gas_price_difference_l1803_180377


namespace NUMINAMATH_CALUDE_find_T_l1803_180301

theorem find_T : ∃ T : ℚ, (1/3 : ℚ) * (1/5 : ℚ) * T = (1/4 : ℚ) * (1/6 : ℚ) * 120 ∧ T = 75 := by
  sorry

end NUMINAMATH_CALUDE_find_T_l1803_180301


namespace NUMINAMATH_CALUDE_additive_inverse_of_zero_l1803_180307

theorem additive_inverse_of_zero : 
  (∀ x : ℝ, x + 0 = x) → 
  (∀ x : ℝ, x + (-x) = 0) → 
  (0 : ℝ) + 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_additive_inverse_of_zero_l1803_180307


namespace NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l1803_180322

/-- The intersection point of two lines is in the fourth quadrant if and only if k is within a specific range -/
theorem intersection_in_fourth_quadrant (k : ℝ) :
  (∃ x y : ℝ, y = -2 * x + 3 * k + 14 ∧ x - 4 * y = -3 * k - 2 ∧ x > 0 ∧ y < 0) ↔ 
  -6 < k ∧ k < -2 := by
sorry

end NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l1803_180322


namespace NUMINAMATH_CALUDE_proposition_truth_l1803_180311

theorem proposition_truth : 
  (¬ (∀ x : ℝ, x + 1/x ≥ 2)) ∧ 
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi/2) ∧ Real.sin x + Real.cos x = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l1803_180311


namespace NUMINAMATH_CALUDE_millet_cost_is_60_cents_l1803_180362

/-- Represents the cost of millet seed per pound -/
def millet_cost : ℝ := sorry

/-- The total weight of millet seed in pounds -/
def millet_weight : ℝ := 100

/-- The cost of sunflower seeds per pound -/
def sunflower_cost : ℝ := 1.10

/-- The total weight of sunflower seeds in pounds -/
def sunflower_weight : ℝ := 25

/-- The desired cost per pound of the mixture -/
def mixture_cost_per_pound : ℝ := 0.70

/-- The total weight of the mixture -/
def total_weight : ℝ := millet_weight + sunflower_weight

/-- Theorem stating that the cost of millet seed per pound is $0.60 -/
theorem millet_cost_is_60_cents :
  millet_cost = 0.60 :=
by
  sorry

end NUMINAMATH_CALUDE_millet_cost_is_60_cents_l1803_180362


namespace NUMINAMATH_CALUDE_ashley_interest_earned_l1803_180336

/-- Calculates the total interest earned in one year given the investment conditions --/
def total_interest (contest_winnings : ℝ) (investment1 : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let investment2 := 2 * investment1 - 400
  let interest1 := investment1 * rate1
  let interest2 := investment2 * rate2
  interest1 + interest2

/-- Theorem stating that the total interest earned is $298 --/
theorem ashley_interest_earned :
  total_interest 5000 1800 0.05 0.065 = 298 := by
  sorry

end NUMINAMATH_CALUDE_ashley_interest_earned_l1803_180336


namespace NUMINAMATH_CALUDE_root_inequality_l1803_180382

noncomputable def f (x : ℝ) : ℝ := (1 + 2 * Real.log x) / x^2

theorem root_inequality (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : f x₁ = k) 
  (h2 : f x₂ = k) 
  (h3 : x₁ < x₂) :
  x₁ + x₂ > 2 ∧ 2 > 1/x₁ + 1/x₂ := by
  sorry

end NUMINAMATH_CALUDE_root_inequality_l1803_180382


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l1803_180389

theorem unique_four_digit_number :
  ∃! (a b c d : ℕ), 
    0 < a ∧ a < 10 ∧
    0 ≤ b ∧ b < 10 ∧
    0 ≤ c ∧ c < 10 ∧
    0 ≤ d ∧ d < 10 ∧
    a + b + c + d = 10 * a + b ∧
    a * b * c * d = 10 * c + d :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l1803_180389


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l1803_180395

theorem fraction_sum_theorem : (1/2 : ℚ) * (1/3 : ℚ) + (1/3 : ℚ) * (1/4 : ℚ) + (1/4 : ℚ) * (1/5 : ℚ) = (3/10 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l1803_180395


namespace NUMINAMATH_CALUDE_xy_value_l1803_180394

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 15) : x * y = 15 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1803_180394


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_surface_area_l1803_180338

/-- The total surface area of a right pyramid with a regular hexagonal base -/
theorem hexagonal_pyramid_surface_area 
  (base_edge : ℝ) 
  (slant_height : ℝ) 
  (h : base_edge = 8) 
  (k : slant_height = 10) : 
  ∃ (area : ℝ), area = 48 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_surface_area_l1803_180338


namespace NUMINAMATH_CALUDE_percentage_below_eight_l1803_180375

/-- Proves that the percentage of students below 8 years of age is 20% -/
theorem percentage_below_eight (total : ℕ) (eight_years : ℕ) (above_eight : ℕ) 
  (h1 : total = 25)
  (h2 : eight_years = 12)
  (h3 : above_eight = 2 * eight_years / 3)
  : (total - eight_years - above_eight) * 100 / total = 20 := by
  sorry

#check percentage_below_eight

end NUMINAMATH_CALUDE_percentage_below_eight_l1803_180375


namespace NUMINAMATH_CALUDE_value_not_unique_l1803_180384

/-- A quadratic function passing through (0, 1) and (1, 0), and concave down -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  pass_origin : c = 1
  pass_one : a + b + c = 0
  concave_down : a < 0

/-- The value of a - b + c cannot be uniquely determined for all quadratic functions satisfying the given conditions -/
theorem value_not_unique (f g : QuadraticFunction) : ∃ (f g : QuadraticFunction), f.a - f.b + f.c ≠ g.a - g.b + g.c := by
  sorry

end NUMINAMATH_CALUDE_value_not_unique_l1803_180384


namespace NUMINAMATH_CALUDE_least_number_of_grapes_l1803_180369

theorem least_number_of_grapes (n : ℕ) : 
  (n > 0) → 
  (n % 3 = 1) → 
  (n % 5 = 1) → 
  (n % 7 = 1) → 
  (∀ m : ℕ, m > 0 → m % 3 = 1 → m % 5 = 1 → m % 7 = 1 → m ≥ n) → 
  n = 106 := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_grapes_l1803_180369


namespace NUMINAMATH_CALUDE_divisibility_by_power_of_three_l1803_180399

def sequence_a (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 4 → a n = a (n - 1) - min (a (n - 2)) (a (n - 3))

theorem divisibility_by_power_of_three (a : ℕ → ℤ) (h : sequence_a a) :
  ∀ k : ℕ, k > 0 → ∃ n : ℕ, (3^k : ℤ) ∣ a n :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_power_of_three_l1803_180399


namespace NUMINAMATH_CALUDE_infinite_solutions_infinitely_many_solutions_l1803_180342

/-- The type of solutions to the equation x^3 + y^3 = z^4 - t^2 -/
def Solution := ℤ × ℤ × ℤ × ℤ

/-- Predicate to check if a tuple (x, y, z, t) is a solution to the equation -/
def is_solution (s : Solution) : Prop :=
  let (x, y, z, t) := s
  x^3 + y^3 = z^4 - t^2

/-- Function to transform a solution using an integer k -/
def transform (k : ℤ) (s : Solution) : Solution :=
  let (x, y, z, t) := s
  (k^4 * x, k^4 * y, k^3 * z, k^6 * t)

/-- Theorem stating that if (x, y, z, t) is a solution, then (k^4*x, k^4*y, k^3*z, k^6*t) is also a solution for any integer k -/
theorem infinite_solutions (s : Solution) (k : ℤ) :
  is_solution s → is_solution (transform k s) := by
  sorry

/-- Corollary: There are infinitely many solutions to the equation -/
theorem infinitely_many_solutions :
  ∃ f : ℕ → Solution, ∀ n : ℕ, is_solution (f n) ∧ ∀ m : ℕ, m ≠ n → f m ≠ f n := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_infinitely_many_solutions_l1803_180342


namespace NUMINAMATH_CALUDE_positive_integer_solutions_count_l1803_180398

theorem positive_integer_solutions_count :
  let n : ℕ := 30
  let k : ℕ := 3
  (Nat.choose (n - 1) (k - 1) : ℕ) = 406 := by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_count_l1803_180398


namespace NUMINAMATH_CALUDE_solve_for_a_l1803_180390

theorem solve_for_a : ∀ a : ℝ, (2 * 2 + a - 5 = 0) → a = 1 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l1803_180390


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l1803_180300

/-- The lateral surface area of a cylinder with given circumference and height -/
def lateral_surface_area (circumference : ℝ) (height : ℝ) : ℝ :=
  circumference * height

/-- Theorem: The lateral surface area of a cylinder with circumference 5cm and height 2cm is 10 cm² -/
theorem cylinder_lateral_area :
  lateral_surface_area 5 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l1803_180300


namespace NUMINAMATH_CALUDE_find_number_l1803_180344

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem find_number (xy : ℕ) (h1 : is_two_digit xy) 
  (h2 : (xy / 10) + (xy % 10) = 8)
  (h3 : reverse_digits xy - xy = 18) : xy = 35 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1803_180344


namespace NUMINAMATH_CALUDE_fourth_month_sale_l1803_180343

/-- Proves that the sale in the fourth month is 5399, given the sales for other months and the required average. -/
theorem fourth_month_sale
  (sale1 sale2 sale3 sale5 sale6 : ℕ)
  (average : ℕ)
  (h1 : sale1 = 5124)
  (h2 : sale2 = 5366)
  (h3 : sale3 = 5808)
  (h5 : sale5 = 6124)
  (h6 : sale6 = 4579)
  (h_avg : average = 5400)
  (h_total : sale1 + sale2 + sale3 + sale5 + sale6 + (6 * average - (sale1 + sale2 + sale3 + sale5 + sale6)) = 6 * average) :
  6 * average - (sale1 + sale2 + sale3 + sale5 + sale6) = 5399 :=
by sorry


end NUMINAMATH_CALUDE_fourth_month_sale_l1803_180343


namespace NUMINAMATH_CALUDE_garden_flower_ratio_l1803_180373

/-- Represents the number of flowers of each color in the garden -/
structure FlowerCounts where
  red : ℕ
  orange : ℕ
  yellow : ℕ
  pink : ℕ
  purple : ℕ

/-- The conditions of the garden problem -/
def gardenConditions (f : FlowerCounts) : Prop :=
  f.red + f.orange + f.yellow + f.pink + f.purple = 105 ∧
  f.orange = 10 ∧
  f.yellow = f.red - 5 ∧
  f.pink = f.purple ∧
  f.pink + f.purple = 30

theorem garden_flower_ratio : 
  ∀ f : FlowerCounts, gardenConditions f → (f.red : ℚ) / f.orange = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_garden_flower_ratio_l1803_180373


namespace NUMINAMATH_CALUDE_six_good_points_l1803_180396

/-- A lattice point on a 9x9 grid -/
structure LatticePoint where
  x : Fin 9
  y : Fin 9

/-- A triangle defined by three lattice points -/
structure Triangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Calculates the area of a triangle given three lattice points -/
def triangleArea (P Q R : LatticePoint) : ℚ :=
  sorry

/-- Checks if a point is a "good point" for a given triangle -/
def isGoodPoint (T : Triangle) (P : LatticePoint) : Prop :=
  triangleArea P T.A T.B = triangleArea P T.A T.C

/-- The main theorem stating that there are exactly 6 "good points" -/
theorem six_good_points (T : Triangle) : 
  ∃! (goodPoints : Finset LatticePoint), 
    (∀ P ∈ goodPoints, isGoodPoint T P) ∧ 
    goodPoints.card = 6 :=
  sorry

end NUMINAMATH_CALUDE_six_good_points_l1803_180396


namespace NUMINAMATH_CALUDE_snow_probability_first_week_february_l1803_180354

theorem snow_probability_first_week_february : 
  let prob_snow_first_three_days : ℚ := 1/4
  let prob_snow_next_four_days : ℚ := 1/3
  let days_in_week : ℕ := 7
  let first_period : ℕ := 3
  let second_period : ℕ := 4
  
  first_period + second_period = days_in_week →
  
  (1 - (1 - prob_snow_first_three_days)^first_period * 
       (1 - prob_snow_next_four_days)^second_period) = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_first_week_february_l1803_180354


namespace NUMINAMATH_CALUDE_candy_difference_l1803_180386

theorem candy_difference (sandra_bags : Nat) (sandra_pieces_per_bag : Nat)
  (roger_bag1 : Nat) (roger_bag2 : Nat) :
  sandra_bags = 2 →
  sandra_pieces_per_bag = 6 →
  roger_bag1 = 11 →
  roger_bag2 = 3 →
  (roger_bag1 + roger_bag2) - (sandra_bags * sandra_pieces_per_bag) = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_difference_l1803_180386


namespace NUMINAMATH_CALUDE_cosine_value_for_given_point_l1803_180376

theorem cosine_value_for_given_point (α : Real) :
  (∃ r : Real, r > 0 ∧ r^2 = 1 + 3) →
  (1, -Real.sqrt 3) ∈ {(x, y) | x = r * Real.cos α ∧ y = r * Real.sin α} →
  Real.cos α = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cosine_value_for_given_point_l1803_180376


namespace NUMINAMATH_CALUDE_min_guests_at_banquet_l1803_180317

/-- The minimum number of guests at a football banquet given the total food consumed and maximum consumption per guest -/
theorem min_guests_at_banquet (total_food : ℝ) (max_per_guest : ℝ) (h1 : total_food = 319) (h2 : max_per_guest = 2.0) : ℕ := by
  sorry

end NUMINAMATH_CALUDE_min_guests_at_banquet_l1803_180317


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l1803_180391

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- Statement: f is an even function and increasing on (0, +∞)
theorem f_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l1803_180391


namespace NUMINAMATH_CALUDE_investment_problem_l1803_180360

/-- Prove the existence and uniqueness of the investment amount and interest rate -/
theorem investment_problem :
  ∃! (P y : ℝ), P > 0 ∧ y > 0 ∧
    P * y * 2 / 100 = 800 ∧
    P * ((1 + y / 100)^2 - 1) = 820 ∧
    P = 8000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l1803_180360


namespace NUMINAMATH_CALUDE_wall_width_l1803_180328

theorem wall_width (w h l : ℝ) (volume : ℝ) : 
  h = 6 * w →
  l = 7 * h →
  volume = w * h * l →
  volume = 6804 →
  w = 3 := by
sorry

end NUMINAMATH_CALUDE_wall_width_l1803_180328


namespace NUMINAMATH_CALUDE_point_symmetry_l1803_180340

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = p.y

theorem point_symmetry (M N P : Point) (hM : M = Point.mk (-4) 3)
    (hN : symmetricOrigin M N) (hP : symmetricYAxis N P) :
    P = Point.mk 4 3 := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_l1803_180340


namespace NUMINAMATH_CALUDE_syllogism_arrangement_correct_l1803_180312

-- Define the statements
def statement1 : Prop := 2012 % 2 = 0
def statement2 : Prop := ∀ n : ℕ, Even n → n % 2 = 0
def statement3 : Prop := Even 2012

-- Define the syllogism structure
inductive SyllogismStep
| MajorPremise
| MinorPremise
| Conclusion

-- Define a function to represent the correct arrangement
def correctArrangement : List (SyllogismStep × Prop) :=
  [(SyllogismStep.MajorPremise, statement2),
   (SyllogismStep.MinorPremise, statement3),
   (SyllogismStep.Conclusion, statement1)]

-- Theorem to prove
theorem syllogism_arrangement_correct :
  correctArrangement = 
    [(SyllogismStep.MajorPremise, statement2),
     (SyllogismStep.MinorPremise, statement3),
     (SyllogismStep.Conclusion, statement1)] :=
by sorry

end NUMINAMATH_CALUDE_syllogism_arrangement_correct_l1803_180312


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1803_180353

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -4)
  are_parallel a b → x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1803_180353


namespace NUMINAMATH_CALUDE_characterization_theorem_l1803_180350

/-- A function that checks if a number satisfies the given conditions -/
def satisfies_condition (n : ℕ) : Prop :=
  ∃ a b : ℕ, 
    n ≥ 2 ∧
    n = a^2 + b^2 ∧
    a > 1 ∧
    a ∣ n ∧
    b ∣ n ∧
    ∀ d : ℕ, d > 1 → d ∣ n → d ≥ a

/-- The main theorem stating the characterization of numbers satisfying the condition -/
theorem characterization_theorem :
  ∀ n : ℕ, satisfies_condition n ↔ 
    (n = 4) ∨ 
    (∃ k j : ℕ, k ≥ 2 ∧ j ≥ 1 ∧ j ≤ k ∧ n = 2^k * (2^(k*(j-1)) + 1)) :=
by sorry

end NUMINAMATH_CALUDE_characterization_theorem_l1803_180350


namespace NUMINAMATH_CALUDE_circle_division_theorem_l1803_180305

/-- The number of regions a circle is divided into by radii and concentric circles -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

theorem circle_division_theorem :
  num_regions 16 10 = 176 := by
  sorry

end NUMINAMATH_CALUDE_circle_division_theorem_l1803_180305


namespace NUMINAMATH_CALUDE_walking_speed_problem_l1803_180302

/-- The walking speeds of two people meeting on a path --/
theorem walking_speed_problem (total_distance : ℝ) (time_diff : ℝ) (meeting_time : ℝ) (speed_diff : ℝ) :
  total_distance = 1200 →
  time_diff = 6 →
  meeting_time = 12 →
  speed_diff = 20 →
  ∃ (v : ℝ),
    v > 0 ∧
    (meeting_time + time_diff) * v + meeting_time * (v + speed_diff) = total_distance ∧
    v = 32 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l1803_180302


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l1803_180348

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles : 
  let circle1 := {(x, y) : ℝ × ℝ | x^2 - 6*x + y^2 + 2*y - 11 = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + 10*x + y^2 - 8*y + 25 = 0}
  (shortest_distance : ℝ) →
  shortest_distance = Real.sqrt 89 - Real.sqrt 21 - 4 ∧
  ∀ (p1 : ℝ × ℝ) (p2 : ℝ × ℝ), 
    p1 ∈ circle1 → p2 ∈ circle2 → 
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) ≥ shortest_distance :=
by
  sorry


end NUMINAMATH_CALUDE_shortest_distance_between_circles_l1803_180348


namespace NUMINAMATH_CALUDE_range_of_k_min_value_when_k_4_min_value_of_reciprocal_sum_l1803_180383

-- Define the function f
def f (x k : ℝ) : ℝ := |2*x - 1| + |2*x - k|

-- Part 1
theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, f x k ≥ 1) ↔ k ≤ 0 ∨ k ≥ 2 :=
sorry

-- Part 2
theorem min_value_when_k_4 :
  ∃ m : ℝ, m = 3 ∧ (∀ x : ℝ, f x 4 ≥ m) :=
sorry

-- Part 3
theorem min_value_of_reciprocal_sum (a b : ℝ) :
  a > 0 → b > 0 → a + 4*b = 3 →
  1/a + 1/b ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_min_value_when_k_4_min_value_of_reciprocal_sum_l1803_180383


namespace NUMINAMATH_CALUDE_max_value_when_h_3_h_values_when_max_negative_one_l1803_180368

-- Define the quadratic function
def f (h : ℝ) (x : ℝ) : ℝ := -(x - h)^2

-- Define the range of x
def x_range (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 5

-- Part 1: Maximum value when h = 3
theorem max_value_when_h_3 :
  ∀ x, x_range x → f 3 x ≤ 0 ∧ ∃ x₀, x_range x₀ ∧ f 3 x₀ = 0 :=
sorry

-- Part 2: Values of h when maximum is -1
theorem h_values_when_max_negative_one :
  (∀ x, x_range x → f h x ≤ -1 ∧ ∃ x₀, x_range x₀ ∧ f h x₀ = -1) →
  h = 6 ∨ h = 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_when_h_3_h_values_when_max_negative_one_l1803_180368


namespace NUMINAMATH_CALUDE_automotive_test_distance_l1803_180321

/-- Calculates the total distance driven in an automotive test -/
theorem automotive_test_distance (d : ℝ) (t : ℝ) : 
  t = d / 4 + d / 5 + d / 6 ∧ t = 37 → 3 * d = 180 := by
  sorry

#check automotive_test_distance

end NUMINAMATH_CALUDE_automotive_test_distance_l1803_180321


namespace NUMINAMATH_CALUDE_largest_of_four_consecutive_evens_l1803_180351

theorem largest_of_four_consecutive_evens (a b c d : ℤ) : 
  (∀ n : ℤ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4 ∧ d = 2*n + 6) →
  a + b + c + d = 140 →
  d = 38 := by
sorry

end NUMINAMATH_CALUDE_largest_of_four_consecutive_evens_l1803_180351


namespace NUMINAMATH_CALUDE_product_xyz_equals_one_l1803_180374

theorem product_xyz_equals_one 
  (x y z : ℝ) 
  (eq1 : x + 1/y = 2) 
  (eq2 : y + 1/z = 3) 
  (eq3 : z + 1/x = 4) : 
  x * y * z = 1 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_equals_one_l1803_180374


namespace NUMINAMATH_CALUDE_no_solution_for_diophantine_equation_l1803_180356

theorem no_solution_for_diophantine_equation :
  ¬ ∃ (m n : ℕ+), 5 * m.val^2 - 6 * m.val * n.val + 7 * n.val^2 = 2006 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_diophantine_equation_l1803_180356


namespace NUMINAMATH_CALUDE_city_distance_ratio_l1803_180320

/-- Prove that the ratio of distances between cities is 2:1 --/
theorem city_distance_ratio :
  ∀ (AB BC CD AD : ℝ),
  AB = 100 →
  BC = AB + 50 →
  AD = 550 →
  AD = AB + BC + CD →
  ∃ (k : ℝ), CD = k * BC →
  CD / BC = 2 := by
sorry

end NUMINAMATH_CALUDE_city_distance_ratio_l1803_180320


namespace NUMINAMATH_CALUDE_double_seven_eighths_of_48_l1803_180345

theorem double_seven_eighths_of_48 : 2 * (7 / 8 * 48) = 84 := by
  sorry

end NUMINAMATH_CALUDE_double_seven_eighths_of_48_l1803_180345


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_fourth_minus_n_l1803_180379

theorem largest_divisor_of_n_fourth_minus_n (n : ℤ) (h : 4 ∣ n) :
  (∃ k : ℤ, n^4 - n = 4 * k) ∧ 
  (∀ m : ℤ, m > 4 → ¬(∀ n : ℤ, 4 ∣ n → m ∣ (n^4 - n))) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_fourth_minus_n_l1803_180379


namespace NUMINAMATH_CALUDE_ratio_equality_l1803_180387

theorem ratio_equality (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4 ∧ a / 2 ≠ 0) :
  (a - 2 * c) / (a - 2 * b) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1803_180387


namespace NUMINAMATH_CALUDE_simplify_fraction_l1803_180361

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1803_180361


namespace NUMINAMATH_CALUDE_binary_digit_difference_l1803_180378

/-- Returns the number of digits in the base-2 representation of a natural number -/
def numDigitsBinary (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

/-- The difference in the number of binary digits between 800 and 250 is 2 -/
theorem binary_digit_difference : numDigitsBinary 800 - numDigitsBinary 250 = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_digit_difference_l1803_180378


namespace NUMINAMATH_CALUDE_related_transitive_l1803_180380

/-- A function is great if it satisfies the given condition for all nonnegative integers m and n. -/
def IsGreat (f : ℕ → ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, f (m + 1) (n + 1) * f m n - f (m + 1) n * f m (n + 1) = 1

/-- Two sequences are related (∼) if there exists a great function satisfying the given conditions. -/
def Related (A B : ℕ → ℤ) : Prop :=
  ∃ f : ℕ → ℕ → ℤ, IsGreat f ∧ (∀ n : ℕ, f n 0 = A n ∧ f 0 n = B n)

/-- The main theorem to be proved. -/
theorem related_transitive (A B C D : ℕ → ℤ) 
  (hAB : Related A B) (hBC : Related B C) (hCD : Related C D) : Related D A := by
  sorry

end NUMINAMATH_CALUDE_related_transitive_l1803_180380


namespace NUMINAMATH_CALUDE_geometric_seq_increasing_condition_l1803_180326

/-- A sequence is geometric if there exists a constant r such that aₙ₊₁ = r * aₙ for all n. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence is increasing if aₙ₊₁ > aₙ for all n. -/
def IsIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem geometric_seq_increasing_condition (a : ℕ → ℝ) (h : IsGeometric a) :
  (IsIncreasing a → a 2 > a 1) ∧ ¬(a 2 > a 1 → IsIncreasing a) :=
by sorry

end NUMINAMATH_CALUDE_geometric_seq_increasing_condition_l1803_180326


namespace NUMINAMATH_CALUDE_sin_4theta_from_exp_itheta_l1803_180333

theorem sin_4theta_from_exp_itheta (θ : ℝ) :
  Complex.exp (Complex.I * θ) = (4 + Complex.I * Real.sqrt 7) / 5 →
  Real.sin (4 * θ) = 144 * Real.sqrt 7 / 625 := by
  sorry

end NUMINAMATH_CALUDE_sin_4theta_from_exp_itheta_l1803_180333


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l1803_180349

def f (x : ℝ) := x^3 - x^2 - x

def f_derivative (x : ℝ) := 3*x^2 - 2*x - 1

theorem monotonic_decreasing_interval :
  {x : ℝ | f_derivative x < 0} = {x : ℝ | -1/3 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l1803_180349


namespace NUMINAMATH_CALUDE_original_equals_scientific_l1803_180319

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 75500000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation := {
  coefficient := 7.55
  exponent := 7
  property := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : 
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l1803_180319


namespace NUMINAMATH_CALUDE_traffic_light_probability_l1803_180331

theorem traffic_light_probability (p_A p_B p_C : ℚ) 
  (h_A : p_A = 1/3) (h_B : p_B = 1/2) (h_C : p_C = 2/3) : 
  (1 - p_A) * p_B * p_C + p_A * (1 - p_B) * p_C + p_A * p_B * (1 - p_C) = 7/18 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_probability_l1803_180331


namespace NUMINAMATH_CALUDE_water_displacement_theorem_l1803_180358

/-- Represents a cylindrical barrel --/
structure Barrel where
  radius : ℝ
  height : ℝ

/-- Represents a cube --/
structure Cube where
  side_length : ℝ

/-- Calculates the volume of water displaced by a cube in a barrel --/
def water_displaced (barrel : Barrel) (cube : Cube) : ℝ :=
  cube.side_length ^ 3

/-- The main theorem about water displacement and its square --/
theorem water_displacement_theorem (barrel : Barrel) (cube : Cube)
    (h1 : barrel.radius = 5)
    (h2 : barrel.height = 15)
    (h3 : cube.side_length = 10) :
    let v := water_displaced barrel cube
    v = 1000 ∧ v^2 = 1000000 := by
  sorry

#check water_displacement_theorem

end NUMINAMATH_CALUDE_water_displacement_theorem_l1803_180358


namespace NUMINAMATH_CALUDE_dress_difference_l1803_180346

theorem dress_difference (total_dresses : ℕ) (ana_dresses : ℕ) 
  (h1 : total_dresses = 48) 
  (h2 : ana_dresses = 15) : 
  total_dresses - ana_dresses - ana_dresses = 18 := by
  sorry

end NUMINAMATH_CALUDE_dress_difference_l1803_180346


namespace NUMINAMATH_CALUDE_expression_simplification_l1803_180318

theorem expression_simplification (x y : ℝ) (h : |x - 2| + (y + 1)^2 = 0) :
  3*x - 2*(x^2 - 1/2*y^2) + (x - 1/2*y^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1803_180318


namespace NUMINAMATH_CALUDE_locus_of_circle_centers_l1803_180393

/-- Given a point O and a radius R, the locus of centers C of circles with radius R
    passing through O is a circle with center O and radius R. -/
theorem locus_of_circle_centers (O : ℝ × ℝ) (R : ℝ) :
  {C : ℝ × ℝ | ∃ P, dist P C = R ∧ P = O} = {C : ℝ × ℝ | dist C O = R} := by sorry

end NUMINAMATH_CALUDE_locus_of_circle_centers_l1803_180393


namespace NUMINAMATH_CALUDE_unique_factorial_sum_power_l1803_180332

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def factorial_sum (m : ℕ) : ℕ := (List.range m).map factorial |>.sum

theorem unique_factorial_sum_power (m n k : ℕ) :
  m > 1 ∧ n^k > 1 ∧ factorial_sum m = n^k → m = 3 ∧ n = 3 ∧ k = 2 := by sorry

end NUMINAMATH_CALUDE_unique_factorial_sum_power_l1803_180332


namespace NUMINAMATH_CALUDE_boards_per_package_not_integer_l1803_180341

theorem boards_per_package_not_integer (total_boards : ℕ) (num_packages : ℕ) 
  (h1 : total_boards = 154) (h2 : num_packages = 52) : 
  ¬ ∃ (n : ℕ), (total_boards : ℚ) / (num_packages : ℚ) = n := by
  sorry

end NUMINAMATH_CALUDE_boards_per_package_not_integer_l1803_180341


namespace NUMINAMATH_CALUDE_unique_line_intersection_l1803_180315

theorem unique_line_intersection (m b : ℝ) : 
  (∃! k, ∃ y₁ y₂, y₁ = k^2 + 4*k + 4 ∧ y₂ = m*k + b ∧ |y₁ - y₂| = 4) ∧
  (m * 2 + b = 8) ∧
  (b ≠ 0) →
  m = 12 ∧ b = -16 := by sorry

end NUMINAMATH_CALUDE_unique_line_intersection_l1803_180315


namespace NUMINAMATH_CALUDE_three_x_squared_y_squared_l1803_180347

theorem three_x_squared_y_squared (x y : ℤ) 
  (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 
  3*x^2*y^2 = 588 := by
sorry

end NUMINAMATH_CALUDE_three_x_squared_y_squared_l1803_180347


namespace NUMINAMATH_CALUDE_expand_polynomial_l1803_180381

theorem expand_polynomial (x : ℝ) : (x + 3) * (x + 8) = x^2 + 11*x + 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l1803_180381


namespace NUMINAMATH_CALUDE_teds_fruit_purchase_cost_l1803_180313

/-- The total cost of purchasing fruits given their quantities and unit prices -/
def total_cost (banana_qty : ℕ) (orange_qty : ℕ) (apple_qty : ℕ) (grape_qty : ℕ)
                (banana_price : ℚ) (orange_price : ℚ) (apple_price : ℚ) (grape_price : ℚ) : ℚ :=
  banana_qty * banana_price + orange_qty * orange_price + 
  apple_qty * apple_price + grape_qty * grape_price

/-- Theorem stating that the total cost of Ted's fruit purchase is $47 -/
theorem teds_fruit_purchase_cost : 
  total_cost 7 15 6 4 2 1.5 1.25 0.75 = 47 := by
  sorry

end NUMINAMATH_CALUDE_teds_fruit_purchase_cost_l1803_180313


namespace NUMINAMATH_CALUDE_f_at_three_l1803_180392

/-- The polynomial function f(x) = 9x^4 + 7x^3 - 5x^2 + 3x - 6 -/
def f (x : ℝ) : ℝ := 9 * x^4 + 7 * x^3 - 5 * x^2 + 3 * x - 6

/-- Theorem stating that f(3) = 876 -/
theorem f_at_three : f 3 = 876 := by
  sorry

end NUMINAMATH_CALUDE_f_at_three_l1803_180392


namespace NUMINAMATH_CALUDE_stan_typing_speed_l1803_180385

theorem stan_typing_speed : 
  -- Define constants
  let pages : ℕ := 5
  let words_per_page : ℕ := 400
  let water_per_hour : ℚ := 15
  let water_needed : ℚ := 10
  -- Calculate total words and time
  let total_words : ℕ := pages * words_per_page
  let time_hours : ℚ := water_needed / water_per_hour
  -- Calculate words per minute
  let words_per_minute : ℚ := total_words / (time_hours * 60)
  -- Prove the result
  words_per_minute = 50 := by sorry

end NUMINAMATH_CALUDE_stan_typing_speed_l1803_180385


namespace NUMINAMATH_CALUDE_flower_shop_expenses_l1803_180304

/-- Calculates the weekly expenses for running a flower shop --/
theorem flower_shop_expenses 
  (rent : ℝ) 
  (utility_rate : ℝ) 
  (hours_per_day : ℕ) 
  (days_per_week : ℕ) 
  (employees_per_shift : ℕ) 
  (hourly_wage : ℝ) 
  (h_rent : rent = 1200) 
  (h_utility : utility_rate = 0.2) 
  (h_hours : hours_per_day = 16) 
  (h_days : days_per_week = 5) 
  (h_employees : employees_per_shift = 2) 
  (h_wage : hourly_wage = 12.5) : 
  rent + rent * utility_rate + 
  (↑hours_per_day * ↑days_per_week * ↑employees_per_shift * hourly_wage) = 3440 := by
  sorry

#check flower_shop_expenses

end NUMINAMATH_CALUDE_flower_shop_expenses_l1803_180304


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1803_180327

/-- A sequence a, b, c forms a geometric sequence if there exists a non-zero real number r such that b = ar and c = br -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r

theorem geometric_sequence_property (a b c : ℝ) :
  (IsGeometricSequence a b c → b^2 = a * c) ∧
  ¬(b^2 = a * c → IsGeometricSequence a b c) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1803_180327


namespace NUMINAMATH_CALUDE_distance_between_centers_l1803_180323

-- Define a circle in the first quadrant tangent to both axes
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_axes : center.1 = center.2
  in_first_quadrant : center.1 > 0 ∧ center.2 > 0

-- Define the property of passing through (4,1)
def passes_through_point (c : Circle) : Prop :=
  (c.center.1 - 4)^2 + (c.center.2 - 1)^2 = c.radius^2

-- Theorem statement
theorem distance_between_centers (c1 c2 : Circle)
  (h1 : passes_through_point c1)
  (h2 : passes_through_point c2)
  (h3 : c1 ≠ c2) :
  Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = 8 :=
sorry

end NUMINAMATH_CALUDE_distance_between_centers_l1803_180323


namespace NUMINAMATH_CALUDE_total_cost_theorem_l1803_180364

/-- The cost of a single shirt -/
def shirt_cost : ℝ := sorry

/-- The cost of a single trouser -/
def trouser_cost : ℝ := sorry

/-- The cost of a single tie -/
def tie_cost : ℝ := sorry

/-- The total cost of 7 shirts, 2 trousers, and 2 ties is $50 -/
axiom condition1 : 7 * shirt_cost + 2 * trouser_cost + 2 * tie_cost = 50

/-- The total cost of 3 trousers, 5 shirts, and 2 ties is $70 -/
axiom condition2 : 5 * shirt_cost + 3 * trouser_cost + 2 * tie_cost = 70

/-- The theorem to be proved -/
theorem total_cost_theorem : 
  3 * shirt_cost + 4 * trouser_cost + 2 * tie_cost = 90 := by sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l1803_180364


namespace NUMINAMATH_CALUDE_negation_of_existential_quantifier_l1803_180365

theorem negation_of_existential_quantifier :
  (¬ ∃ x : ℝ, x^2 ≤ |x|) ↔ (∀ x : ℝ, x^2 > |x|) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_quantifier_l1803_180365


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1803_180366

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 = 8)
  (h_sum2 : a 2 + a 4 = 12) :
  (∀ n : ℕ, a n = 2 * n) ∧
  (∃ n : ℕ, n * (n + 1) = 420) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1803_180366


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l1803_180324

theorem more_girls_than_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 35 → 
  3 * girls = 4 * boys → 
  total = boys + girls → 
  girls - boys = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l1803_180324


namespace NUMINAMATH_CALUDE_three_unique_circles_l1803_180370

/-- A square with vertices P, Q, R, and S -/
structure Square where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- A circle defined by two points as its diameter endpoints -/
structure Circle where
  endpoint1 : Point
  endpoint2 : Point

/-- Function to count unique circles defined by square vertices -/
def count_unique_circles (s : Square) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 3 unique circles -/
theorem three_unique_circles (s : Square) : 
  count_unique_circles s = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_unique_circles_l1803_180370


namespace NUMINAMATH_CALUDE_quadratic_function_determination_l1803_180303

theorem quadratic_function_determination (a b c : ℝ) (h_a : a > 0) : 
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∀ x : ℝ, a * x + b ≤ 2) →
  (∃ x : ℝ, a * x + b = 2) →
  (a * x^2 + b * x + c = 2 * x^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_determination_l1803_180303


namespace NUMINAMATH_CALUDE_waiter_remaining_customers_l1803_180367

/-- Calculates the number of remaining customers after some customers leave. -/
def remainingCustomers (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

theorem waiter_remaining_customers :
  remainingCustomers 21 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_waiter_remaining_customers_l1803_180367


namespace NUMINAMATH_CALUDE_enrique_shredder_y_feeds_l1803_180388

/-- Calculates the number of times a shredder needs to be fed to shred all pages of a contract type -/
def shredder_feeds (num_contracts : ℕ) (pages_per_contract : ℕ) (pages_per_shred : ℕ) : ℕ :=
  ((num_contracts * pages_per_contract + pages_per_shred - 1) / pages_per_shred : ℕ)

theorem enrique_shredder_y_feeds : 
  let type_b_contracts : ℕ := 350
  let pages_per_type_b : ℕ := 10
  let shredder_y_capacity : ℕ := 8
  shredder_feeds type_b_contracts pages_per_type_b shredder_y_capacity = 438 := by
sorry

end NUMINAMATH_CALUDE_enrique_shredder_y_feeds_l1803_180388


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1803_180337

theorem polynomial_division_theorem :
  let p (z : ℝ) := 4 * z^3 - 8 * z^2 + 9 * z - 7
  let d (z : ℝ) := 4 * z + 2
  let q (z : ℝ) := z^2 - 2.5 * z + 3.5
  let r : ℝ := -14
  ∀ z : ℝ, p z = d z * q z + r := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1803_180337
