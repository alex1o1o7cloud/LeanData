import Mathlib

namespace NUMINAMATH_CALUDE_z_polyomino_placement_count_l2584_258448

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (h_size : size = 8)

/-- Represents a Z-shaped polyomino -/
structure ZPolyomino :=
  (cells : Nat)
  (h_cells : cells = 4)

/-- Represents the number of ways to place a Z-shaped polyomino on a chessboard -/
def placement_count (board : Chessboard) (poly : ZPolyomino) : Nat :=
  168

/-- Theorem stating that the number of ways to place a Z-shaped polyomino on an 8x8 chessboard is 168 -/
theorem z_polyomino_placement_count :
  ∀ (board : Chessboard) (poly : ZPolyomino),
  placement_count board poly = 168 :=
by sorry

end NUMINAMATH_CALUDE_z_polyomino_placement_count_l2584_258448


namespace NUMINAMATH_CALUDE_even_function_decreasing_l2584_258452

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f is decreasing on an interval (a, b) if
    for all x, y in (a, b), x < y implies f(x) > f(y) -/
def IsDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

theorem even_function_decreasing :
  let f := fun x => (m - 1) * x^2 + 2 * m * x + 3
  IsEven f →
  IsDecreasing f 2 5 :=
by
  sorry

#check even_function_decreasing

end NUMINAMATH_CALUDE_even_function_decreasing_l2584_258452


namespace NUMINAMATH_CALUDE_c_alone_time_l2584_258434

-- Define the work rates for A, B, and C
variable (rA rB rC : ℝ)

-- Define the conditions
def condition1 : Prop := rA + rB = 1/3
def condition2 : Prop := rB + rC = 1/3
def condition3 : Prop := rA + rC = 2/3

-- Theorem to prove
theorem c_alone_time (h1 : condition1 rA rB) (h2 : condition2 rB rC) (h3 : condition3 rA rC) :
  1 / rC = 3 := by
  sorry


end NUMINAMATH_CALUDE_c_alone_time_l2584_258434


namespace NUMINAMATH_CALUDE_ladybugs_without_spots_l2584_258456

theorem ladybugs_without_spots (total : ℕ) (with_spots : ℕ) (without_spots : ℕ) : 
  total = 67082 → with_spots = 12170 → without_spots = total - with_spots → without_spots = 54912 := by
  sorry

end NUMINAMATH_CALUDE_ladybugs_without_spots_l2584_258456


namespace NUMINAMATH_CALUDE_smallest_undefined_value_l2584_258413

theorem smallest_undefined_value (x : ℝ) : 
  (∀ y < 1, ∃ z, (y + 2) / (10 * y^2 - 90 * y + 20) = z) ∧ 
  ¬∃ z, (1 + 2) / (10 * 1^2 - 90 * 1 + 20) = z := by
  sorry

end NUMINAMATH_CALUDE_smallest_undefined_value_l2584_258413


namespace NUMINAMATH_CALUDE_ten_percent_of_point_one_l2584_258412

theorem ten_percent_of_point_one (x : ℝ) : x = 0.1 * 0.10 → x = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ten_percent_of_point_one_l2584_258412


namespace NUMINAMATH_CALUDE_real_number_pure_imaginary_condition_l2584_258497

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for a pure imaginary number
def isPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem real_number_pure_imaginary_condition (m : ℝ) : 
  isPureImaginary (m^2 * (1 + i) + (m - i) - 2) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_real_number_pure_imaginary_condition_l2584_258497


namespace NUMINAMATH_CALUDE_root_difference_absolute_value_specific_root_difference_l2584_258478

theorem root_difference_absolute_value (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → |r₁ - r₂| = 1 :=
by
  sorry

-- Specific instance for the given problem
theorem specific_root_difference :
  let r₁ := (-(-7) + Real.sqrt ((-7)^2 - 4*1*12)) / (2*1)
  let r₂ := (-(-7) - Real.sqrt ((-7)^2 - 4*1*12)) / (2*1)
  x^2 - 7*x + 12 = 0 → |r₁ - r₂| = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_root_difference_absolute_value_specific_root_difference_l2584_258478


namespace NUMINAMATH_CALUDE_regression_line_not_fixed_point_l2584_258445

/-- A regression line in the form y = bx + a -/
structure RegressionLine where
  a : ℝ
  b : ℝ

/-- Predicate to check if a point lies on a regression line -/
def lies_on (line : RegressionLine) (x y : ℝ) : Prop :=
  y = line.b * x + line.a

theorem regression_line_not_fixed_point :
  ∃ (line : RegressionLine), 
    (¬ lies_on line 0 0) ∧ 
    (∀ x, ¬ lies_on line x 0) ∧ 
    (∀ x y, ¬ lies_on line x y) :=
sorry

end NUMINAMATH_CALUDE_regression_line_not_fixed_point_l2584_258445


namespace NUMINAMATH_CALUDE_only_2017_is_prime_l2584_258400

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem only_2017_is_prime :
  ¬(is_prime 2015) ∧
  ¬(is_prime 2016) ∧
  is_prime 2017 ∧
  ¬(is_prime 2018) ∧
  ¬(is_prime 2019) :=
sorry

end NUMINAMATH_CALUDE_only_2017_is_prime_l2584_258400


namespace NUMINAMATH_CALUDE_park_area_l2584_258426

/-- Represents a rectangular park with given properties -/
structure RectangularPark where
  length : ℝ
  breadth : ℝ
  ratio : length / breadth = 1 / 3
  perimeter : length * 2 + breadth * 2 = 1600

/-- The area of the rectangular park is 120000 square meters -/
theorem park_area (park : RectangularPark) : park.length * park.breadth = 120000 := by
  sorry

end NUMINAMATH_CALUDE_park_area_l2584_258426


namespace NUMINAMATH_CALUDE_two_problems_require_loop_l2584_258428

/-- Represents a problem that may or may not require a loop statement to solve. -/
inductive Problem
| SumGeometricSeries
| CompareNumbers
| PiecewiseFunction
| LargestSquareLessThan100

/-- Determines if a given problem requires a loop statement to solve. -/
def requiresLoop (p : Problem) : Bool :=
  match p with
  | Problem.SumGeometricSeries => true
  | Problem.CompareNumbers => false
  | Problem.PiecewiseFunction => false
  | Problem.LargestSquareLessThan100 => true

/-- The list of all problems given in the original question. -/
def allProblems : List Problem :=
  [Problem.SumGeometricSeries, Problem.CompareNumbers, 
   Problem.PiecewiseFunction, Problem.LargestSquareLessThan100]

theorem two_problems_require_loop : 
  (allProblems.filter requiresLoop).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_problems_require_loop_l2584_258428


namespace NUMINAMATH_CALUDE_sum_of_squares_l2584_258469

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_sum : a + b + c = 0) (h_power : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) : 
  a^2 + b^2 + c^2 = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2584_258469


namespace NUMINAMATH_CALUDE_min_visible_sum_l2584_258458

/-- Represents a die in the cube -/
structure Die where
  sides : Fin 6 → ℕ
  opposite_sum : ∀ i : Fin 3, sides i + sides (i + 3) = 7

/-- Represents the 4x4x4 cube made of dice -/
def Cube := Fin 4 → Fin 4 → Fin 4 → Die

/-- Calculates the sum of visible faces on the large cube -/
def visible_sum (c : Cube) : ℕ := sorry

/-- The theorem stating the minimum possible visible sum -/
theorem min_visible_sum (c : Cube) : visible_sum c ≥ 144 := by sorry

end NUMINAMATH_CALUDE_min_visible_sum_l2584_258458


namespace NUMINAMATH_CALUDE_triangle_determines_plane_l2584_258482

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point3D) : Prop := sorry

/-- Function to determine a plane from a triangle -/
def planeFromTriangle (t : Triangle3D) : Plane3D := sorry

theorem triangle_determines_plane (t : Triangle3D) : 
  ¬collinear t.a t.b t.c → ∃! p : Plane3D, p = planeFromTriangle t :=
sorry

end NUMINAMATH_CALUDE_triangle_determines_plane_l2584_258482


namespace NUMINAMATH_CALUDE_arithmetic_sequence_term_number_l2584_258485

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_term_number
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a5 : a 5 = 33)
  (h_a45 : a 45 = 153)
  : (∃ n : ℕ, a n = 201) ∧ (∀ n : ℕ, a n = 201 → n = 61) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_term_number_l2584_258485


namespace NUMINAMATH_CALUDE_prob_not_six_l2584_258401

/-- Given a die where the odds of rolling a six are 1:5, 
    the probability of not rolling a six is 5/6 -/
theorem prob_not_six (favorable : ℕ) (unfavorable : ℕ) :
  favorable = 1 →
  unfavorable = 5 →
  (unfavorable : ℚ) / (favorable + unfavorable : ℚ) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_six_l2584_258401


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2584_258476

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Theorem stating the properties of a specific quadratic function -/
theorem quadratic_function_properties (f : QuadraticFunction)
  (point1 : f.a * (-1)^2 + f.b * (-1) + f.c = -1)
  (point2 : f.c = 1)
  (condition : f.a * (-2)^2 + f.b * (-2) + f.c > 1) :
  (f.a * f.b * f.c > 0) ∧
  (∃ x y : ℝ, x ≠ y ∧ f.a * x^2 + f.b * x + f.c - 3 = 0 ∧ f.a * y^2 + f.b * y + f.c - 3 = 0) ∧
  (f.a + f.b + f.c > 7) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2584_258476


namespace NUMINAMATH_CALUDE_expression_simplification_l2584_258409

theorem expression_simplification (a b : ℤ) (ha : a = -3) (hb : b = -2) :
  (a + b) * (b - a) + (2 * a^2 * b - a^3) / (-a) = b^2 - 2 * a * b ∧
  b^2 - 2 * a * b = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2584_258409


namespace NUMINAMATH_CALUDE_product_of_solutions_l2584_258422

theorem product_of_solutions (x : ℝ) : 
  (12 = 3 * x^2 + 18 * x) → 
  (∃ x₁ x₂ : ℝ, (12 = 3 * x₁^2 + 18 * x₁) ∧ (12 = 3 * x₂^2 + 18 * x₂) ∧ (x₁ * x₂ = -4)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2584_258422


namespace NUMINAMATH_CALUDE_optimal_coin_strategy_l2584_258479

/-- Probability of winning with n turns left and difference k between heads and tails -/
noncomputable def q (n : ℕ) (k : ℕ) : ℝ :=
  sorry

/-- The optimal strategy theorem -/
theorem optimal_coin_strategy (n : ℕ) (k : ℕ) : q n k ≥ q n (k + 2) := by
  sorry

end NUMINAMATH_CALUDE_optimal_coin_strategy_l2584_258479


namespace NUMINAMATH_CALUDE_factorization_equality_l2584_258475

theorem factorization_equality (a b : ℝ) : 3 * a^2 + 6 * a * b = 3 * a * (a + 2 * b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2584_258475


namespace NUMINAMATH_CALUDE_remainder_negation_l2584_258403

theorem remainder_negation (a : ℤ) : 
  (a % 1999 = 1) → ((-a) % 1999 = 1998) := by
  sorry

end NUMINAMATH_CALUDE_remainder_negation_l2584_258403


namespace NUMINAMATH_CALUDE_no_suitable_dishes_l2584_258432

theorem no_suitable_dishes (total_dishes : ℕ) (vegetarian_dishes : ℕ) 
  (gluten_dishes : ℕ) (nut_dishes : ℕ) 
  (h1 : vegetarian_dishes = 6)
  (h2 : vegetarian_dishes = total_dishes / 4)
  (h3 : gluten_dishes = 4)
  (h4 : nut_dishes = 2)
  (h5 : gluten_dishes + nut_dishes = vegetarian_dishes) :
  (vegetarian_dishes - gluten_dishes - nut_dishes : ℚ) / total_dishes = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_suitable_dishes_l2584_258432


namespace NUMINAMATH_CALUDE_function_domain_real_iff_m_in_range_l2584_258495

/-- The function y = (mx - 1) / (mx² + 4mx + 3) has domain ℝ if and only if m ∈ [0, 3/4) -/
theorem function_domain_real_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, mx^2 + 4*m*x + 3 ≠ 0) ↔ m ∈ Set.Ici 0 ∩ Set.Iio (3/4) :=
by sorry

end NUMINAMATH_CALUDE_function_domain_real_iff_m_in_range_l2584_258495


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l2584_258421

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) :
  (1/2 : ℝ) * x * (3*x) = 72 → x = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l2584_258421


namespace NUMINAMATH_CALUDE_min_value_sum_and_reciprocal_l2584_258474

theorem min_value_sum_and_reciprocal (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (((a^2 + b^2 : ℚ) / (a * b)) + ((a * b : ℚ) / (a^2 + b^2))) ≥ 2 ∧
  ∃ (a' b' : ℤ), a' ≠ 0 ∧ b' ≠ 0 ∧ (((a'^2 + b'^2 : ℚ) / (a' * b')) + ((a' * b' : ℚ) / (a'^2 + b'^2))) = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_and_reciprocal_l2584_258474


namespace NUMINAMATH_CALUDE_tommy_saw_13_cars_l2584_258460

/-- The number of cars Tommy saw -/
def num_cars : ℕ := 13

/-- The number of wheels per vehicle -/
def wheels_per_vehicle : ℕ := 4

/-- The number of trucks Tommy saw -/
def num_trucks : ℕ := 12

/-- The total number of wheels Tommy saw -/
def total_wheels : ℕ := 100

theorem tommy_saw_13_cars :
  num_cars = (total_wheels - num_trucks * wheels_per_vehicle) / wheels_per_vehicle :=
by sorry

end NUMINAMATH_CALUDE_tommy_saw_13_cars_l2584_258460


namespace NUMINAMATH_CALUDE_fraction_equality_l2584_258473

theorem fraction_equality : (1 + 3 + 5) / (10 + 6 + 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2584_258473


namespace NUMINAMATH_CALUDE_root_of_fifth_unity_l2584_258437

theorem root_of_fifth_unity (p q r s t k : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (h1 : p * k^4 + q * k^3 + r * k^2 + s * k + t = 0)
  (h2 : q * k^4 + r * k^3 + s * k^2 + t * k + p = 0) :
  k^5 = 1 := by
sorry

end NUMINAMATH_CALUDE_root_of_fifth_unity_l2584_258437


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l2584_258416

theorem two_digit_number_problem : ∃ (x : ℕ), 
  (x ≥ 10 ∧ x ≤ 99) ∧ 
  (500 + x = 9 * x - 12) ∧ 
  x = 64 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l2584_258416


namespace NUMINAMATH_CALUDE_log_sum_sixteen_sixtyfour_l2584_258443

theorem log_sum_sixteen_sixtyfour : Real.log 64 / Real.log 16 + Real.log 16 / Real.log 64 = 13 / 6 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_sixteen_sixtyfour_l2584_258443


namespace NUMINAMATH_CALUDE_pair_2017_is_1_64_l2584_258453

/-- Represents an integer pair -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- Calculates the total number of pairs in the first n groups -/
def totalPairsInGroups (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Generates the first pair of the nth group -/
def firstPairOfGroup (n : ℕ) : IntPair :=
  ⟨1, n⟩

theorem pair_2017_is_1_64 :
  ∃ n : ℕ, totalPairsInGroups n = 2016 ∧ firstPairOfGroup (n + 1) = ⟨1, 64⟩ := by
  sorry

#check pair_2017_is_1_64

end NUMINAMATH_CALUDE_pair_2017_is_1_64_l2584_258453


namespace NUMINAMATH_CALUDE_correct_calculation_l2584_258449

theorem correct_calculation (a b : ℝ) : 2 * a^2 * b - 3 * a^2 * b = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2584_258449


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2584_258436

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x + y > 2 → (x > 1 ∨ y > 1)) ∧
  ¬((x > 1 ∨ y > 1) → (x + y > 2)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2584_258436


namespace NUMINAMATH_CALUDE_maria_trip_expenses_l2584_258481

def initial_amount : ℕ := 760
def ticket_cost : ℕ := 300
def hotel_cost : ℕ := ticket_cost / 2
def total_spent : ℕ := ticket_cost + hotel_cost
def remaining_amount : ℕ := initial_amount - total_spent

theorem maria_trip_expenses :
  remaining_amount = 310 :=
sorry

end NUMINAMATH_CALUDE_maria_trip_expenses_l2584_258481


namespace NUMINAMATH_CALUDE_christine_savings_510_l2584_258455

/-- Calculates Christine's savings based on her commission rates, sales, and allocation percentages. -/
def christine_savings (
  electronics_rate : ℚ)
  (clothing_rate : ℚ)
  (furniture_rate : ℚ)
  (electronics_sales : ℚ)
  (clothing_sales : ℚ)
  (furniture_sales : ℚ)
  (personal_needs_rate : ℚ)
  (investment_rate : ℚ) : ℚ :=
  let total_commission := 
    electronics_rate * electronics_sales + 
    clothing_rate * clothing_sales + 
    furniture_rate * furniture_sales
  let personal_needs := personal_needs_rate * total_commission
  let investments := investment_rate * total_commission
  total_commission - personal_needs - investments

/-- Theorem stating that Christine's savings for the month equal $510. -/
theorem christine_savings_510 : 
  christine_savings (15/100) (10/100) (20/100) 12000 8000 4000 (55/100) (30/100) = 510 := by
  sorry

end NUMINAMATH_CALUDE_christine_savings_510_l2584_258455


namespace NUMINAMATH_CALUDE_concert_ticket_price_l2584_258415

/-- Given concert ticket information, prove the cost of a section B ticket -/
theorem concert_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℚ) 
  (section_a_tickets : ℕ) 
  (section_b_tickets : ℕ) 
  (section_a_price : ℚ)
  (h1 : total_tickets = section_a_tickets + section_b_tickets)
  (h2 : total_tickets = 4500)
  (h3 : total_revenue = 30000)
  (h4 : section_a_tickets = 2900)
  (h5 : section_b_tickets = 1600)
  (h6 : section_a_price = 8) :
  ∃ (section_b_price : ℚ), 
    section_b_price = 4.25 ∧ 
    total_revenue = section_a_tickets * section_a_price + section_b_tickets * section_b_price :=
by sorry

end NUMINAMATH_CALUDE_concert_ticket_price_l2584_258415


namespace NUMINAMATH_CALUDE_number_of_cars_on_road_prove_number_of_cars_l2584_258442

theorem number_of_cars_on_road (total_distance : ℝ) (car_spacing : ℝ) : ℝ :=
  let number_of_cars := (total_distance / car_spacing) + 1
  number_of_cars

theorem prove_number_of_cars :
  number_of_cars_on_road 242 5.5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_of_cars_on_road_prove_number_of_cars_l2584_258442


namespace NUMINAMATH_CALUDE_beaker_capacity_ratio_l2584_258450

theorem beaker_capacity_ratio : 
  ∀ (S L : ℝ), 
  S > 0 → L > 0 →
  (∃ k : ℝ, L = k * S) →
  (1/2 * S + 1/5 * L = 3/10 * L) →
  L / S = 5 := by
sorry

end NUMINAMATH_CALUDE_beaker_capacity_ratio_l2584_258450


namespace NUMINAMATH_CALUDE_square_roots_product_l2584_258420

theorem square_roots_product (a b : ℝ) : 
  (a * a = 9) ∧ (b * b = 9) ∧ (a ≠ b) → a * b = -9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_product_l2584_258420


namespace NUMINAMATH_CALUDE_restaurant_production_in_june_l2584_258427

/-- Represents the daily production of a restaurant -/
structure DailyProduction where
  cheesePizzas : ℕ
  pepperoniPizzas : ℕ
  beefHotDogs : ℕ
  chickenHotDogs : ℕ

/-- Calculates the monthly production based on daily production and number of days -/
def monthlyProduction (daily : DailyProduction) (days : ℕ) : DailyProduction :=
  { cheesePizzas := daily.cheesePizzas * days
  , pepperoniPizzas := daily.pepperoniPizzas * days
  , beefHotDogs := daily.beefHotDogs * days
  , chickenHotDogs := daily.chickenHotDogs * days
  }

theorem restaurant_production_in_june 
  (daily : DailyProduction)
  (cheese_more_than_hotdogs : daily.cheesePizzas = daily.beefHotDogs + daily.chickenHotDogs + 40)
  (pepperoni_twice_cheese : daily.pepperoniPizzas = 2 * daily.cheesePizzas)
  (total_hotdogs : daily.beefHotDogs + daily.chickenHotDogs = 60)
  (beef_hotdogs : daily.beefHotDogs = 30)
  (chicken_hotdogs : daily.chickenHotDogs = 30)
  : monthlyProduction daily 30 = 
    { cheesePizzas := 3000
    , pepperoniPizzas := 6000
    , beefHotDogs := 900
    , chickenHotDogs := 900
    } := by
  sorry


end NUMINAMATH_CALUDE_restaurant_production_in_june_l2584_258427


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2584_258487

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^5 + 5 * X^4 - 13 * X^3 - 7 * X^2 + 52 * X - 34 = 
  (X^3 + 6 * X^2 + 5 * X - 7) * q + (50 * X^3 + 79 * X^2 - 39 * X - 34) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2584_258487


namespace NUMINAMATH_CALUDE_sequence_2002nd_term_l2584_258470

theorem sequence_2002nd_term : 
  let sequence : ℕ → ℕ := λ n => n^2 - 1
  sequence 2002 = 4008003 := by sorry

end NUMINAMATH_CALUDE_sequence_2002nd_term_l2584_258470


namespace NUMINAMATH_CALUDE_min_value_of_z_l2584_258486

theorem min_value_of_z (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x - 2*y + 3 = 0) :
  ∀ z : ℝ, z = y^2 / x → z ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l2584_258486


namespace NUMINAMATH_CALUDE_only_solutions_for_exponential_equation_l2584_258444

theorem only_solutions_for_exponential_equation :
  ∀ a b : ℕ+, a ^ b.val = b.val ^ (a.val ^ 2) →
    ((a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 16) ∨ (a = 3 ∧ b = 27)) := by
  sorry

end NUMINAMATH_CALUDE_only_solutions_for_exponential_equation_l2584_258444


namespace NUMINAMATH_CALUDE_root_property_l2584_258424

theorem root_property (x₀ : ℝ) (h : 2 * x₀^2 * Real.exp (2 * x₀) + Real.log x₀ = 0) :
  2 * x₀ + Real.log x₀ = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l2584_258424


namespace NUMINAMATH_CALUDE_emily_necklaces_l2584_258404

def necklaces (total_beads : ℕ) (beads_per_necklace : ℕ) : ℕ :=
  total_beads / beads_per_necklace

theorem emily_necklaces :
  necklaces 28 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_emily_necklaces_l2584_258404


namespace NUMINAMATH_CALUDE_quadratic_form_h_value_l2584_258472

theorem quadratic_form_h_value (a b c : ℝ) :
  (∃ (n k : ℝ), ∀ x, 5 * (a * x^2 + b * x + c) = n * (x - 5)^2 + k) →
  (∀ x, a * x^2 + b * x + c = 3 * (x - 5)^2 + 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_h_value_l2584_258472


namespace NUMINAMATH_CALUDE_max_three_digit_sum_l2584_258465

theorem max_three_digit_sum (A B C : Nat) : 
  A ≠ B ∧ B ≠ C ∧ C ≠ A → 
  A < 10 ∧ B < 10 ∧ C < 10 →
  110 * A + 10 * B + 3 * C ≤ 981 ∧ 
  (∃ A' B' C', A' ≠ B' ∧ B' ≠ C' ∧ C' ≠ A' ∧ 
               A' < 10 ∧ B' < 10 ∧ C' < 10 ∧ 
               110 * A' + 10 * B' + 3 * C' = 981) :=
by sorry

end NUMINAMATH_CALUDE_max_three_digit_sum_l2584_258465


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2584_258425

theorem sum_of_x_and_y (x y : ℝ) : 
  ((x + Real.sqrt y) + (x - Real.sqrt y) = 8) →
  ((x + Real.sqrt y) * (x - Real.sqrt y) = 15) →
  (x + y = 5) := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2584_258425


namespace NUMINAMATH_CALUDE_cannot_form_square_l2584_258492

/-- The number of sticks of length 1 cm -/
def sticks_1cm : ℕ := 6

/-- The number of sticks of length 2 cm -/
def sticks_2cm : ℕ := 3

/-- The number of sticks of length 3 cm -/
def sticks_3cm : ℕ := 6

/-- The number of sticks of length 4 cm -/
def sticks_4cm : ℕ := 5

/-- The total length of all sticks in cm -/
def total_length : ℕ := sticks_1cm * 1 + sticks_2cm * 2 + sticks_3cm * 3 + sticks_4cm * 4

/-- Theorem stating that it's impossible to form a square with the given sticks -/
theorem cannot_form_square : ¬(total_length % 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_cannot_form_square_l2584_258492


namespace NUMINAMATH_CALUDE_dad_steps_l2584_258461

/-- Represents the number of steps taken by each person -/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- Defines the relationship between steps taken by Dad, Masha, and Yasha -/
def step_relation (s : Steps) : Prop :=
  5 * s.dad = 3 * s.masha ∧ 5 * s.masha = 3 * s.yasha

/-- Theorem stating that given the conditions, Dad took 90 steps -/
theorem dad_steps (s : Steps) :
  step_relation s → s.masha + s.yasha = 400 → s.dad = 90 := by
  sorry


end NUMINAMATH_CALUDE_dad_steps_l2584_258461


namespace NUMINAMATH_CALUDE_hexagon_area_theorem_l2584_258410

/-- A regular hexagon inscribed in a circle of unit area -/
structure RegularHexagonInCircle where
  /-- The circle has unit area -/
  circle_area : ℝ := 1
  /-- The hexagon is inscribed in the circle -/
  hexagon_inscribed : Bool

/-- A point Q inside the circle -/
structure PointInCircle where
  /-- The point is inside the circle -/
  inside : Bool

/-- The area of a region bounded by two sides of the hexagon and a minor arc -/
def area_region (h : RegularHexagonInCircle) (q : PointInCircle) (i j : Fin 6) : ℝ := sorry

theorem hexagon_area_theorem (h : RegularHexagonInCircle) (q : PointInCircle) :
  area_region h q 0 1 = 1 / 12 →
  area_region h q 2 3 = 1 / 15 →
  ∃ m : ℕ+, area_region h q 4 5 = 1 / 18 - Real.sqrt 3 / m →
  m = 20 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_theorem_l2584_258410


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l2584_258454

/-- The orthocenter of a triangle in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC is at (-29/12, 77/12, 49/12) -/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 1)
  let B : ℝ × ℝ × ℝ := (4, -1, 2)
  let C : ℝ × ℝ × ℝ := (1, 1, 4)
  orthocenter A B C = (-29/12, 77/12, 49/12) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l2584_258454


namespace NUMINAMATH_CALUDE_triangle_not_tileable_with_sphinx_l2584_258459

/-- Represents a triangle that can be divided into smaller triangles -/
structure DivisibleTriangle where
  side_length : ℕ
  upward_triangles : ℕ
  downward_triangles : ℕ

/-- Represents a sphinx tile -/
structure SphinxTile where
  covers_even_orientations : Bool

/-- Checks if a triangle can be tiled with sphinx tiles -/
def can_be_tiled_with_sphinx (t : DivisibleTriangle) (s : SphinxTile) : Prop :=
  s.covers_even_orientations →
    (t.upward_triangles % 2 = 0 ∧ t.downward_triangles % 2 = 0)

/-- The main theorem stating that the specific triangle cannot be tiled with sphinx tiles -/
theorem triangle_not_tileable_with_sphinx :
  ∀ (t : DivisibleTriangle) (s : SphinxTile),
    t.side_length = 6 →
    t.upward_triangles = 21 →
    t.downward_triangles = 15 →
    s.covers_even_orientations →
    ¬(can_be_tiled_with_sphinx t s) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_not_tileable_with_sphinx_l2584_258459


namespace NUMINAMATH_CALUDE_tv_cost_l2584_258435

theorem tv_cost (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  savings = 600 →
  furniture_fraction = 3/4 →
  tv_cost = savings - (furniture_fraction * savings) →
  tv_cost = 150 := by
sorry

end NUMINAMATH_CALUDE_tv_cost_l2584_258435


namespace NUMINAMATH_CALUDE_expression_simplification_l2584_258466

theorem expression_simplification (x : ℝ) : 
  (x - 1)^5 + 5*(x - 1)^4 + 10*(x - 1)^3 + 10*(x - 1)^2 + 5*(x - 1) = x^5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2584_258466


namespace NUMINAMATH_CALUDE_ratio_p_to_r_l2584_258493

theorem ratio_p_to_r (p q r s : ℚ) 
  (h1 : p / q = 5 / 4)
  (h2 : r / s = 4 / 3)
  (h3 : s / q = 1 / 8) :
  p / r = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_p_to_r_l2584_258493


namespace NUMINAMATH_CALUDE_sum_squared_l2584_258480

theorem sum_squared (x y : ℝ) (h1 : x * (x + y) = 27) (h2 : y * (x + y) = 54) : 
  (x + y)^2 = 81 := by
sorry

end NUMINAMATH_CALUDE_sum_squared_l2584_258480


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2017_l2584_258430

/-- The last four digits of a natural number -/
def lastFourDigits (n : ℕ) : ℕ := n % 10000

/-- The function that raises 5 to a power and takes the last four digits -/
def f (n : ℕ) : ℕ := lastFourDigits (5^n)

theorem last_four_digits_of_5_pow_2017 :
  f 5 = 3125 → f 6 = 5625 → f 7 = 8125 → f 2017 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2017_l2584_258430


namespace NUMINAMATH_CALUDE_union_not_all_reals_l2584_258494

def M : Set ℝ := {x | 0 < x ∧ x < 1}
def N : Set ℝ := {y | 0 < y}

theorem union_not_all_reals : M ∪ N ≠ Set.univ := by sorry

end NUMINAMATH_CALUDE_union_not_all_reals_l2584_258494


namespace NUMINAMATH_CALUDE_triangle_median_angle_equivalence_l2584_258467

/-- In a triangle ABC, prove that (1/a + 1/b = 1/t_a) ⟺ (C = 2π/3) -/
theorem triangle_median_angle_equivalence 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (t_a : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : A > 0 ∧ B > 0 ∧ C > 0)
  (h_sum_angles : A + B + C = π)
  (h_side_a : a = 2 * (Real.sin A))
  (h_side_b : b = 2 * (Real.sin B))
  (h_side_c : c = 2 * (Real.sin C))
  (h_median : t_a = (2 * b * c) / (b + c) * Real.cos (A / 2)) :
  (1 / a + 1 / b = 1 / t_a) ↔ (C = 2 * π / 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_median_angle_equivalence_l2584_258467


namespace NUMINAMATH_CALUDE_order_of_fractions_l2584_258402

theorem order_of_fractions (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hab : a > b) : 
  b / a < (b + c) / (a + c) ∧ (b + c) / (a + c) < (a + d) / (b + d) ∧ (a + d) / (b + d) < a / b := by
  sorry

end NUMINAMATH_CALUDE_order_of_fractions_l2584_258402


namespace NUMINAMATH_CALUDE_rohan_earnings_l2584_258462

/-- Represents a coconut farm with given properties -/
structure CoconutFarm where
  size : Nat
  treesPerSquareMeter : Nat
  coconutsPerTree : Nat
  harvestFrequency : Nat
  pricePerCoconut : Rat
  timePeriod : Nat

/-- Calculates the earnings from a coconut farm over a given time period -/
def calculateEarnings (farm : CoconutFarm) : Rat :=
  let totalTrees := farm.size * farm.treesPerSquareMeter
  let totalCoconuts := totalTrees * farm.coconutsPerTree
  let harvests := farm.timePeriod / farm.harvestFrequency
  let totalCoconutsHarvested := totalCoconuts * harvests
  totalCoconutsHarvested * farm.pricePerCoconut

/-- Rohan's coconut farm -/
def rohansFarm : CoconutFarm :=
  { size := 20
  , treesPerSquareMeter := 2
  , coconutsPerTree := 6
  , harvestFrequency := 3
  , pricePerCoconut := 1/2
  , timePeriod := 6 }

theorem rohan_earnings : calculateEarnings rohansFarm = 240 := by
  sorry

end NUMINAMATH_CALUDE_rohan_earnings_l2584_258462


namespace NUMINAMATH_CALUDE_chess_club_games_l2584_258489

/-- 
Given a chess club with the following properties:
- There are 15 participants in total
- 5 of the participants are instructors
- Each member plays one game against each instructor

The total number of games played is 70.
-/
theorem chess_club_games (total_participants : ℕ) (instructors : ℕ) :
  total_participants = 15 →
  instructors = 5 →
  instructors * (total_participants - 1) = 70 := by
  sorry

end NUMINAMATH_CALUDE_chess_club_games_l2584_258489


namespace NUMINAMATH_CALUDE_even_function_property_l2584_258419

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

def has_min_value (f : ℝ → ℝ) (m : ℝ) : Prop := ∀ x, m ≤ f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

theorem even_function_property (f : ℝ → ℝ) :
  is_even f →
  is_increasing_on f 1 3 →
  has_min_value f 0 →
  is_decreasing_on f (-3) (-1) ∧ has_min_value f 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_property_l2584_258419


namespace NUMINAMATH_CALUDE_reciprocal_sum_fourths_sixths_l2584_258414

theorem reciprocal_sum_fourths_sixths : (1 / (1/4 + 1/6) : ℚ) = 12/5 := by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_fourths_sixths_l2584_258414


namespace NUMINAMATH_CALUDE_divisors_of_cube_l2584_258446

theorem divisors_of_cube (m : ℕ) : 
  (∃ p : ℕ, Prime p ∧ m = p^4) → (Finset.card (Nat.divisors (m^3)) = 13) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_cube_l2584_258446


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l2584_258464

theorem smallest_sum_of_squares : ∃ (x y : ℕ), 
  x^2 - y^2 = 175 ∧ 
  x^2 ≥ 36 ∧ 
  y^2 ≥ 36 ∧ 
  x^2 + y^2 = 625 ∧ 
  (∀ (a b : ℕ), a^2 - b^2 = 175 → a^2 ≥ 36 → b^2 ≥ 36 → a^2 + b^2 ≥ 625) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l2584_258464


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l2584_258488

/-- A regular decagon is a 10-sided polygon with all sides and angles equal -/
structure RegularDecagon where
  sides : Nat
  sides_eq : sides = 10

/-- The probability that two randomly chosen diagonals of a regular decagon intersect inside the decagon -/
def diagonal_intersection_probability (d : RegularDecagon) : ℚ :=
  42 / 119

/-- Theorem stating that the probability of two randomly chosen diagonals 
    of a regular decagon intersecting inside the decagon is 42/119 -/
theorem decagon_diagonal_intersection_probability (d : RegularDecagon) :
  diagonal_intersection_probability d = 42 / 119 := by
  sorry

#check decagon_diagonal_intersection_probability

end NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l2584_258488


namespace NUMINAMATH_CALUDE_horner_v2_equals_24_l2584_258447

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 + 5x^4 + 10x^3 + 10x^2 + 5x + 1 -/
def f : ℝ → ℝ := fun x => x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

/-- Coefficients of f(x) in descending order -/
def f_coeffs : List ℝ := [1, 5, 10, 10, 5, 1]

theorem horner_v2_equals_24 :
  let x := 2
  let v2 := (horner (f_coeffs.take 3) x) * x + f_coeffs[3]!
  v2 = 24 := by sorry

end NUMINAMATH_CALUDE_horner_v2_equals_24_l2584_258447


namespace NUMINAMATH_CALUDE_johnnys_laps_per_minute_l2584_258418

/-- 
Given that Johnny ran 10 laps in 3.333 minutes, 
prove that the number of laps he runs per minute is equal to 10 divided by 3.333.
-/
theorem johnnys_laps_per_minute (total_laps : ℝ) (total_minutes : ℝ) 
  (h1 : total_laps = 10) 
  (h2 : total_minutes = 3.333) : 
  total_laps / total_minutes = 10 / 3.333 := by
  sorry

end NUMINAMATH_CALUDE_johnnys_laps_per_minute_l2584_258418


namespace NUMINAMATH_CALUDE_fabric_area_calculation_l2584_258433

/-- The area of a rectangular piece of fabric -/
def fabric_area (width : ℝ) (length : ℝ) : ℝ := width * length

/-- Theorem: The area of a rectangular piece of fabric with width 3 cm and length 8 cm is 24 square cm -/
theorem fabric_area_calculation :
  fabric_area 3 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_fabric_area_calculation_l2584_258433


namespace NUMINAMATH_CALUDE_identity_function_only_l2584_258477

def P (f : ℕ → ℕ) : ℕ → ℕ 
  | 0 => 1
  | n + 1 => (P f n) * (f (n + 1))

theorem identity_function_only (f : ℕ → ℕ) :
  (∀ a b : ℕ, (P f a + P f b) ∣ (Nat.factorial a + Nat.factorial b)) →
  (∀ n : ℕ, f n = n) := by
  sorry

end NUMINAMATH_CALUDE_identity_function_only_l2584_258477


namespace NUMINAMATH_CALUDE_subtraction_equation_solution_l2584_258438

def is_valid_subtraction (a b c d : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
  1000 * a + 100 * b + 82 - (900 + 10 * c + 9) = 4000 + 90 * d + 3

theorem subtraction_equation_solution :
  ∀ a b c d : Nat, is_valid_subtraction a b c d → a = 5 :=
sorry

end NUMINAMATH_CALUDE_subtraction_equation_solution_l2584_258438


namespace NUMINAMATH_CALUDE_square_area_increase_l2584_258498

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.3 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.69 := by
sorry

end NUMINAMATH_CALUDE_square_area_increase_l2584_258498


namespace NUMINAMATH_CALUDE_actual_average_speed_l2584_258471

/-- Given that increasing the speed by 12 miles per hour reduces the time by 1/4,
    prove that the actual average speed is 36 miles per hour. -/
theorem actual_average_speed : 
  ∃ v : ℝ, v > 0 ∧ v / (v + 12) = 3/4 ∧ v = 36 := by sorry

end NUMINAMATH_CALUDE_actual_average_speed_l2584_258471


namespace NUMINAMATH_CALUDE_derek_dogs_at_six_l2584_258440

theorem derek_dogs_at_six (dogs_at_six cars_at_six : ℕ) 
  (h1 : dogs_at_six = 3 * cars_at_six)
  (h2 : cars_at_six + 210 = 2 * 120)
  : dogs_at_six = 90 := by
  sorry

end NUMINAMATH_CALUDE_derek_dogs_at_six_l2584_258440


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2584_258406

theorem quadratic_factorization (C D E F : ℤ) :
  (∀ y : ℝ, 10 * y^2 - 51 * y + 21 = (C * y - D) * (E * y - F)) →
  C * E + C = 15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2584_258406


namespace NUMINAMATH_CALUDE_round_trip_average_speed_average_speed_approx_31_5_l2584_258429

/-- Calculates the average speed for a round trip with given conditions -/
theorem round_trip_average_speed 
  (total_distance : ℝ) 
  (plain_speed : ℝ) 
  (uphill_increase : ℝ) 
  (uphill_decrease : ℝ) : ℝ :=
  let half_distance := total_distance / 2
  let uphill_speed := plain_speed * (1 + uphill_increase) * (1 - uphill_decrease)
  let plain_time := half_distance / plain_speed
  let uphill_time := half_distance / uphill_speed
  let total_time := plain_time + uphill_time
  total_distance / total_time

/-- Proves that the average speed for the given round trip is approximately 31.5 km/hr -/
theorem average_speed_approx_31_5 : 
  ∃ ε > 0, |round_trip_average_speed 240 30 0.3 0.15 - 31.5| < ε :=
sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_average_speed_approx_31_5_l2584_258429


namespace NUMINAMATH_CALUDE_parallel_from_equation_basis_transformation_l2584_258451

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Two non-zero vectors are parallel if one is a scalar multiple of the other -/
def Parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

theorem parallel_from_equation (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) (h : 2 • a = -3 • b) :
  Parallel a b := by sorry

theorem basis_transformation {e₁ e₂ : V} (h : LinearIndependent ℝ ![e₁, e₂]) :
  LinearIndependent ℝ ![e₁ + 2 • e₂, e₁ - 2 • e₂] := by sorry

end NUMINAMATH_CALUDE_parallel_from_equation_basis_transformation_l2584_258451


namespace NUMINAMATH_CALUDE_no_distinct_perfect_squares_sum_to_100_l2584_258405

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def distinct_perfect_squares_sum_to_100 : Prop :=
  ∃ a b c : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c ∧
    a + b + c = 100

theorem no_distinct_perfect_squares_sum_to_100 : ¬distinct_perfect_squares_sum_to_100 := by
  sorry

end NUMINAMATH_CALUDE_no_distinct_perfect_squares_sum_to_100_l2584_258405


namespace NUMINAMATH_CALUDE_sum_of_other_y_coordinates_l2584_258483

/-- A rectangle in a 2D plane --/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- 
Given a rectangle with two opposite vertices at (2, 10) and (8, -6),
the sum of the y-coordinates of the other two vertices is 4.
-/
theorem sum_of_other_y_coordinates (rect : Rectangle) 
  (h1 : rect.v1 = (2, 10))
  (h2 : rect.v3 = (8, -6))
  (h_opposite : (rect.v1 = (2, 10) ∧ rect.v3 = (8, -6)) ∨ 
                (rect.v2 = (2, 10) ∧ rect.v4 = (8, -6))) :
  (rect.v2).2 + (rect.v4).2 = 4 := by
  sorry

#check sum_of_other_y_coordinates

end NUMINAMATH_CALUDE_sum_of_other_y_coordinates_l2584_258483


namespace NUMINAMATH_CALUDE_f_properties_l2584_258423

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := |x| + m / x - 1

theorem f_properties (m : ℝ) :
  -- 1. Monotonicity when m = 2
  (m = 2 → ∀ x y, x < y ∧ y < 0 → f m x > f m y) ∧
  -- 2. Condition for f(2^x) > 0
  (∀ x, f m (2^x) > 0 ↔ m > 1/4) ∧
  -- 3. Number of zeros
  ((∃! x, f m x = 0) ↔ (m > 1/4 ∨ m < -1/4)) ∧
  ((∃ x y, x ≠ y ∧ f m x = 0 ∧ f m y = 0 ∧ ∀ z, f m z = 0 → z = x ∨ z = y) ↔ (m = 1/4 ∨ m = 0 ∨ m = -1/4)) ∧
  ((∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f m x = 0 ∧ f m y = 0 ∧ f m z = 0 ∧ ∀ w, f m w = 0 → w = x ∨ w = y ∨ w = z) ↔ (0 < m ∧ m < 1/4) ∨ (-1/4 < m ∧ m < 0)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2584_258423


namespace NUMINAMATH_CALUDE_tenth_pirate_share_l2584_258468

/-- Represents the number of coins each pirate takes -/
def pirate_share (i : Nat) (remaining : Nat) : Nat :=
  if i ≤ 5 then
    (i * remaining) / 10
  else if i < 10 then
    20
  else
    remaining

/-- Calculates the remaining coins after each pirate takes their share -/
def remaining_coins (i : Nat) (initial : Nat) : Nat :=
  if i = 0 then initial
  else
    remaining_coins (i - 1) initial - pirate_share (i - 1) (remaining_coins (i - 1) initial)

/-- The main theorem stating that the 10th pirate receives 376 coins -/
theorem tenth_pirate_share : pirate_share 10 (remaining_coins 10 3000) = 376 := by
  sorry

end NUMINAMATH_CALUDE_tenth_pirate_share_l2584_258468


namespace NUMINAMATH_CALUDE_positive_real_inequalities_l2584_258407

theorem positive_real_inequalities (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^3 - y^3 ≥ 4*x → x^2 > 2*y) ∧ (x^5 - y^3 ≥ 2*x → x^3 ≥ 2*y) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequalities_l2584_258407


namespace NUMINAMATH_CALUDE_abs_sum_inequality_range_l2584_258499

theorem abs_sum_inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3*a) ↔ a ∈ Set.Icc (-1) 4 :=
sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_range_l2584_258499


namespace NUMINAMATH_CALUDE_jimin_class_size_l2584_258490

/-- The number of students in Jimin's class -/
def total_students : ℕ := 45

/-- The number of students who like Korean -/
def korean_students : ℕ := 38

/-- The number of students who like math -/
def math_students : ℕ := 39

/-- The number of students who like both Korean and math -/
def both_subjects : ℕ := 32

/-- Theorem stating the total number of students in Jimin's class -/
theorem jimin_class_size :
  total_students = korean_students + math_students - both_subjects :=
by sorry

end NUMINAMATH_CALUDE_jimin_class_size_l2584_258490


namespace NUMINAMATH_CALUDE_chord_length_squared_in_sector_l2584_258411

/-- Given a circular sector with central angle 60° and radius 6 cm, 
    the square of the chord length subtending the central angle is 36 cm^2 -/
theorem chord_length_squared_in_sector (r : ℝ) (θ : ℝ) (h1 : r = 6) (h2 : θ = π/3) :
  (2 * r * Real.sin (θ/2))^2 = 36 :=
sorry

end NUMINAMATH_CALUDE_chord_length_squared_in_sector_l2584_258411


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l2584_258439

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 24 x = Nat.choose 24 (3*x - 8)) → (x = 4 ∨ x = 8) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l2584_258439


namespace NUMINAMATH_CALUDE_january_31_is_wednesday_l2584_258457

/-- Days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Function to advance a day by n days -/
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

/-- Theorem: If January 1 is a Monday, then January 31 is a Wednesday -/
theorem january_31_is_wednesday : 
  advanceDay DayOfWeek.Monday 30 = DayOfWeek.Wednesday := by
  sorry

end NUMINAMATH_CALUDE_january_31_is_wednesday_l2584_258457


namespace NUMINAMATH_CALUDE_fewer_girls_than_boys_l2584_258417

theorem fewer_girls_than_boys (total_students : ℕ) (girls : ℕ) (boys : ℕ) : 
  total_students = 27 →
  girls = 11 →
  total_students = girls + boys →
  boys - girls = 5 := by
sorry

end NUMINAMATH_CALUDE_fewer_girls_than_boys_l2584_258417


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2584_258408

/-- A rectangle with given diagonal and area -/
structure Rectangle where
  x : ℝ  -- length
  y : ℝ  -- width
  diagonal_sq : x^2 + y^2 = 17^2
  area : x * y = 120

theorem rectangle_perimeter (r : Rectangle) : 2 * (r.x + r.y) = 46 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2584_258408


namespace NUMINAMATH_CALUDE_angle_C_measure_l2584_258431

-- Define the angles
variable (A B C : ℝ)

-- Define the parallel lines property
variable (p_parallel_q : Bool)

-- State the theorem
theorem angle_C_measure :
  p_parallel_q ∧ A = (1/6) * B → C = 25.71 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l2584_258431


namespace NUMINAMATH_CALUDE_arctan_sum_eq_pi_over_four_l2584_258496

theorem arctan_sum_eq_pi_over_four :
  ∃! (n : ℕ), n > 0 ∧ Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/n) = π/4 :=
by sorry

end NUMINAMATH_CALUDE_arctan_sum_eq_pi_over_four_l2584_258496


namespace NUMINAMATH_CALUDE_locus_of_Q_is_ellipse_l2584_258484

/-- The locus of point Q is an ellipse centered at P with the same eccentricity as the original ellipse -/
theorem locus_of_Q_is_ellipse (m n x₀ y₀ : ℝ) (hm : m > 0) (hn : n > 0) 
  (hP : m * x₀^2 + n * y₀^2 < 1) : 
  ∃ (x y : ℝ), m * (x - x₀)^2 + n * (y - y₀)^2 = m * x₀^2 + n * y₀^2 := by
  sorry

#check locus_of_Q_is_ellipse

end NUMINAMATH_CALUDE_locus_of_Q_is_ellipse_l2584_258484


namespace NUMINAMATH_CALUDE_equal_area_line_equation_l2584_258463

/-- A circle in the coordinate plane -/
structure Circle where
  center : ℝ × ℝ
  diameter : ℝ

/-- The region S formed by the union of nine circular regions -/
def region_S : Set (ℝ × ℝ) :=
  sorry

/-- A line in the coordinate plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if a line divides a region into two equal areas -/
def divides_equally (l : Line) (r : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- The equation of a line in the form ax = by + c -/
structure LineEquation where
  a : ℕ
  b : ℕ
  c : ℕ
  gcd_one : Nat.gcd a (Nat.gcd b c) = 1

theorem equal_area_line_equation :
  ∃ (eq : LineEquation),
    let l : Line := { slope := 2, intercept := sorry }
    divides_equally l region_S ∧
    eq.a^2 + eq.b^2 + eq.c^2 = 69 :=
  sorry

end NUMINAMATH_CALUDE_equal_area_line_equation_l2584_258463


namespace NUMINAMATH_CALUDE_a_range_l2584_258491

theorem a_range (a : ℝ) (h1 : a > 0) :
  let p := ∀ x y : ℝ, x < y → a^x < a^y
  let q := ∀ x : ℝ, a*x^2 + a*x + 1 > 0
  (¬p ∧ ¬q) ∧ (p ∨ q) → a ∈ Set.Ioo 0 1 ∪ Set.Ici 4 :=
sorry

end NUMINAMATH_CALUDE_a_range_l2584_258491


namespace NUMINAMATH_CALUDE_equation_solution_expression_factorization_inequalities_solution_l2584_258441

-- Part 1: Equation solution
theorem equation_solution (x : ℝ) : 
  1 / (x - 2) - x / (x^2 - 4) = 2 / (x + 2) → x = 6 :=
by sorry

-- Part 2: Expression factorization
theorem expression_factorization (x : ℝ) :
  x^2 * (x - 2) - 9 * (x - 2) = (x - 2) * (x + 3) * (x - 3) :=
by sorry

-- Part 3: System of inequalities solution
theorem inequalities_solution (x : ℝ) :
  ((x - 3) / 3 < 1 ∧ 3 * x - 4 ≤ 2 * (3 * x - 2)) ↔ (0 ≤ x ∧ x < 6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_expression_factorization_inequalities_solution_l2584_258441
