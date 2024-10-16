import Mathlib

namespace NUMINAMATH_CALUDE_max_value_a_l1963_196300

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b) 
  (h2 : b < 4 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 80) : 
  a ≤ 4724 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 4724 ∧ 
    b' = 1575 ∧ 
    c' = 394 ∧ 
    d' = 79 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 80 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l1963_196300


namespace NUMINAMATH_CALUDE_oliver_age_l1963_196305

/-- Given the ages of Oliver, Mia, and Lucas, prove that Oliver is 18 years old. -/
theorem oliver_age :
  ∀ (oliver_age mia_age lucas_age : ℕ),
    oliver_age = mia_age - 2 →
    mia_age = lucas_age + 5 →
    lucas_age = 15 →
    oliver_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_oliver_age_l1963_196305


namespace NUMINAMATH_CALUDE_line_contains_point_l1963_196303

theorem line_contains_point (k : ℝ) : 
  (3 / 4 - 3 * k * (1 / 3) = 7 * (-4)) ↔ k = 28.75 := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l1963_196303


namespace NUMINAMATH_CALUDE_opposite_violet_is_blue_l1963_196360

-- Define the colors
inductive Color
  | Orange
  | Black
  | Yellow
  | Violet
  | Blue
  | Pink

-- Define a cube
structure Cube where
  faces : Fin 6 → Color

-- Define the three views of the cube
def view1 (c : Cube) : Prop :=
  c.faces 0 = Color.Blue ∧ c.faces 1 = Color.Yellow ∧ c.faces 2 = Color.Orange

def view2 (c : Cube) : Prop :=
  c.faces 0 = Color.Blue ∧ c.faces 1 = Color.Pink ∧ c.faces 2 = Color.Orange

def view3 (c : Cube) : Prop :=
  c.faces 0 = Color.Blue ∧ c.faces 1 = Color.Black ∧ c.faces 2 = Color.Orange

-- Define the opposite face relation
def oppositeFace (i j : Fin 6) : Prop :=
  (i = 0 ∧ j = 5) ∨ (i = 1 ∧ j = 3) ∨ (i = 2 ∧ j = 4) ∨
  (i = 3 ∧ j = 1) ∨ (i = 4 ∧ j = 2) ∨ (i = 5 ∧ j = 0)

-- Theorem statement
theorem opposite_violet_is_blue (c : Cube) :
  (∀ i j : Fin 6, i ≠ j → c.faces i ≠ c.faces j) →
  view1 c → view2 c → view3 c →
  ∃ i j : Fin 6, oppositeFace i j ∧ c.faces i = Color.Violet ∧ c.faces j = Color.Blue :=
sorry

end NUMINAMATH_CALUDE_opposite_violet_is_blue_l1963_196360


namespace NUMINAMATH_CALUDE_complement_of_union_equals_open_interval_l1963_196393

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_of_union_equals_open_interval :
  (A ∪ B)ᶜ = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_open_interval_l1963_196393


namespace NUMINAMATH_CALUDE_negation_of_all_squares_positive_l1963_196362

theorem negation_of_all_squares_positive :
  ¬(∀ n : ℕ, n^2 > 0) ↔ ∃ n : ℕ, n^2 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_positive_l1963_196362


namespace NUMINAMATH_CALUDE_intersection_complement_equals_interval_l1963_196331

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {y | y ≥ 0}
def B : Set ℝ := {y | ∃ x, y = Real.sqrt x + 1}

-- State the theorem
theorem intersection_complement_equals_interval :
  A ∩ (U \ B) = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_interval_l1963_196331


namespace NUMINAMATH_CALUDE_parabola_equation_correct_l1963_196396

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the left focus of the ellipse
def left_focus : ℝ × ℝ := (-1, 0)

-- Define the vertex of the parabola
def vertex : ℝ × ℝ := (0, 0)

-- Define the parabola equation
def parabola_eq (x y : ℝ) : Prop := y^2 = -4*x

-- Theorem statement
theorem parabola_equation_correct :
  ∀ x y : ℝ,
  ellipse x y →
  parabola_eq x y →
  (left_focus.1 < 0 ∧ left_focus.2 = 0) →
  (vertex.1 = 0 ∧ vertex.2 = 0) →
  ∃ p : ℝ, p = 2 ∧ y^2 = -2*p*x :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_correct_l1963_196396


namespace NUMINAMATH_CALUDE_sin_720_equals_0_l1963_196339

theorem sin_720_equals_0 (n : ℤ) (h1 : -90 ≤ n ∧ n ≤ 90) (h2 : Real.sin (n * π / 180) = Real.sin (720 * π / 180)) : n = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_720_equals_0_l1963_196339


namespace NUMINAMATH_CALUDE_rectangle_area_around_square_total_area_of_rectangles_l1963_196375

theorem rectangle_area_around_square (l₁ l₂ : ℝ) : 
  l₁ + l₂ = 11 → 
  2 * (6 * l₁ + 6 * l₂) = 132 := by
  sorry

theorem total_area_of_rectangles : 
  ∀ (l₁ l₂ : ℝ), 
  (4 * (12 + l₁ + l₂) = 92) → 
  (2 * (6 * l₁ + 6 * l₂) = 132) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_around_square_total_area_of_rectangles_l1963_196375


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1963_196398

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (4, x) (-4, 4) → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1963_196398


namespace NUMINAMATH_CALUDE_smallest_five_digit_in_pascal_l1963_196363

/-- Pascal's triangle function -/
def pascal (n k : ℕ) : ℕ := sorry

/-- A number is five-digit if it's between 10000 and 99999 inclusive -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem smallest_five_digit_in_pascal :
  ∃ (n k : ℕ), pascal n k = 10000 ∧
  (∀ (m l : ℕ), pascal m l < 10000 ∨ (pascal m l = 10000 ∧ m ≥ n)) :=
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_in_pascal_l1963_196363


namespace NUMINAMATH_CALUDE_birthday_candles_cost_l1963_196370

/-- Calculates the total cost of blue and green candles given the ratio and number of red candles -/
def total_cost_blue_green_candles (ratio_red blue green : ℕ) (num_red : ℕ) (cost_blue cost_green : ℕ) : ℕ :=
  let units_per_ratio := num_red / ratio_red
  let num_blue := units_per_ratio * blue
  let num_green := units_per_ratio * green
  num_blue * cost_blue + num_green * cost_green

/-- Theorem stating that the total cost of blue and green candles is $333 given the problem conditions -/
theorem birthday_candles_cost : 
  total_cost_blue_green_candles 5 3 7 45 3 4 = 333 := by
  sorry

end NUMINAMATH_CALUDE_birthday_candles_cost_l1963_196370


namespace NUMINAMATH_CALUDE_complex_number_theorem_l1963_196326

theorem complex_number_theorem (z : ℂ) : 
  (∃ (k : ℝ), z / (1 + Complex.I) = k * Complex.I) ∧ 
  Complex.abs (z / (1 + Complex.I)) = 1 → 
  z = -1 + Complex.I ∨ z = 1 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l1963_196326


namespace NUMINAMATH_CALUDE_import_tax_calculation_l1963_196384

theorem import_tax_calculation (tax_rate : ℝ) (tax_threshold : ℝ) (tax_paid : ℝ) (total_value : ℝ) : 
  tax_rate = 0.07 →
  tax_threshold = 1000 →
  tax_paid = 87.50 →
  tax_rate * (total_value - tax_threshold) = tax_paid →
  total_value = 2250 := by
sorry

end NUMINAMATH_CALUDE_import_tax_calculation_l1963_196384


namespace NUMINAMATH_CALUDE_division_of_decimals_l1963_196333

theorem division_of_decimals : (2.4 : ℝ) / 0.06 = 40 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l1963_196333


namespace NUMINAMATH_CALUDE_georgie_initial_avocados_l1963_196347

/-- The number of avocados needed per serving of guacamole -/
def avocados_per_serving : ℕ := 3

/-- The number of avocados Georgie's sister buys -/
def sister_bought : ℕ := 4

/-- The number of servings Georgie can make -/
def servings : ℕ := 3

/-- Georgie's initial number of avocados -/
def initial_avocados : ℕ := servings * avocados_per_serving - sister_bought

theorem georgie_initial_avocados : initial_avocados = 5 := by
  sorry

end NUMINAMATH_CALUDE_georgie_initial_avocados_l1963_196347


namespace NUMINAMATH_CALUDE_d_in_N_l1963_196348

def M : Set ℤ := {x | ∃ n : ℤ, x = 3 * n}
def N : Set ℤ := {x | ∃ n : ℤ, |x| = 3 * n + 1}
def P : Set ℤ := {x | ∃ n : ℤ, x = 3 * n - 1}

theorem d_in_N (a b c : ℤ) (ha : a ∈ M) (hb : b ∈ N) (hc : c ∈ P) :
  (a - b + c) ∈ N := by
  sorry

end NUMINAMATH_CALUDE_d_in_N_l1963_196348


namespace NUMINAMATH_CALUDE_bookstore_discount_proof_l1963_196345

/-- Calculates the final price as a percentage of the original price
    given an initial discount and an additional discount on the already discounted price. -/
def final_price_percentage (initial_discount : ℝ) (additional_discount : ℝ) : ℝ :=
  (1 - initial_discount) * (1 - additional_discount) * 100

/-- Proves that with a 40% initial discount and an additional 20% discount,
    the final price is 48% of the original price. -/
theorem bookstore_discount_proof :
  final_price_percentage 0.4 0.2 = 48 := by
sorry

end NUMINAMATH_CALUDE_bookstore_discount_proof_l1963_196345


namespace NUMINAMATH_CALUDE_hotel_payment_ratio_l1963_196337

/-- Given a hotel with operations expenses and a loss, compute the ratio of total payments to operations cost -/
theorem hotel_payment_ratio (operations_cost loss : ℚ) 
  (h1 : operations_cost = 100)
  (h2 : loss = 25) :
  (operations_cost - loss) / operations_cost = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_hotel_payment_ratio_l1963_196337


namespace NUMINAMATH_CALUDE_odd_function_a_range_l1963_196379

/-- An odd function f: ℝ → ℝ satisfying given conditions -/
def OddFunction (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x < 0, f x = 9*x + a^2/x + 7) ∧
  (∀ x ≥ 0, f x ≥ 0)

/-- Theorem stating the range of values for a -/
theorem odd_function_a_range (f : ℝ → ℝ) (a : ℝ) 
  (h : OddFunction f a) : 
  a ≥ 7/6 ∨ a ≤ -7/6 :=
sorry

end NUMINAMATH_CALUDE_odd_function_a_range_l1963_196379


namespace NUMINAMATH_CALUDE_solution_of_linear_equation_l1963_196385

theorem solution_of_linear_equation :
  ∃ x : ℝ, 2 * x + 6 = 0 ∧ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_solution_of_linear_equation_l1963_196385


namespace NUMINAMATH_CALUDE_cellphone_purchase_cost_l1963_196343

/-- The total cost for two cellphones with a discount -/
theorem cellphone_purchase_cost 
  (price_per_phone : ℝ) 
  (number_of_phones : ℕ) 
  (discount_rate : ℝ) 
  (h1 : price_per_phone = 800)
  (h2 : number_of_phones = 2)
  (h3 : discount_rate = 0.05) : 
  price_per_phone * number_of_phones * (1 - discount_rate) = 1520 := by
  sorry

#check cellphone_purchase_cost

end NUMINAMATH_CALUDE_cellphone_purchase_cost_l1963_196343


namespace NUMINAMATH_CALUDE_product_bcd_value_l1963_196320

theorem product_bcd_value
  (a b c d e f : ℝ)
  (h1 : a * b * c = 130)
  (h2 : c * d * e = 500)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 1) :
  b * c * d = 65 := by
  sorry

end NUMINAMATH_CALUDE_product_bcd_value_l1963_196320


namespace NUMINAMATH_CALUDE_public_library_book_count_l1963_196312

/-- The number of books in Oak Grove's public library -/
def public_library_books : ℕ := 7092 - 5106

/-- The total number of books in Oak Grove libraries -/
def total_books : ℕ := 7092

/-- The number of books in Oak Grove's school libraries -/
def school_library_books : ℕ := 5106

theorem public_library_book_count :
  public_library_books = 1986 ∧
  total_books = public_library_books + school_library_books :=
by sorry

end NUMINAMATH_CALUDE_public_library_book_count_l1963_196312


namespace NUMINAMATH_CALUDE_point_b_coordinates_l1963_196358

/-- A line segment parallel to the x-axis -/
structure ParallelSegment where
  A : ℝ × ℝ  -- Coordinates of point A
  B : ℝ × ℝ  -- Coordinates of point B
  length : ℝ  -- Length of the segment
  parallel_to_x : B.2 = A.2  -- y-coordinates are equal

/-- The theorem statement -/
theorem point_b_coordinates (seg : ParallelSegment) 
  (h1 : seg.A = (3, 2))  -- Point A is at (3, 2)
  (h2 : seg.length = 3)  -- Length of AB is 3
  : seg.B = (0, 2) ∨ seg.B = (6, 2) := by
  sorry

end NUMINAMATH_CALUDE_point_b_coordinates_l1963_196358


namespace NUMINAMATH_CALUDE_salary_comparison_l1963_196341

/-- Represents the salary distribution for graduates --/
structure SalaryDistribution where
  total_students : ℕ
  graduating_students : ℕ
  dropout_salary : ℝ
  high_salary : ℝ
  mid_salary : ℝ
  low_salary : ℝ
  default_salary : ℝ
  high_salary_ratio : ℝ
  mid_salary_ratio : ℝ
  low_salary_ratio : ℝ

/-- Represents Fyodor's salary growth --/
structure SalaryGrowth where
  initial_salary : ℝ
  yearly_increase : ℝ
  years : ℕ

/-- Calculates the expected salary based on the given distribution --/
def expected_salary (d : SalaryDistribution) : ℝ :=
  let graduate_prob := d.graduating_students / d.total_students
  let default_salary_ratio := 1 - d.high_salary_ratio - d.mid_salary_ratio - d.low_salary_ratio
  graduate_prob * (d.high_salary_ratio * d.high_salary + 
                   d.mid_salary_ratio * d.mid_salary + 
                   d.low_salary_ratio * d.low_salary + 
                   default_salary_ratio * d.default_salary) +
  (1 - graduate_prob) * d.dropout_salary

/-- Calculates Fyodor's salary after a given number of years --/
def fyodor_salary (g : SalaryGrowth) : ℝ :=
  g.initial_salary + g.yearly_increase * g.years

/-- The main theorem to prove --/
theorem salary_comparison 
  (d : SalaryDistribution)
  (g : SalaryGrowth)
  (h1 : d.total_students = 300)
  (h2 : d.graduating_students = 270)
  (h3 : d.dropout_salary = 25000)
  (h4 : d.high_salary = 60000)
  (h5 : d.mid_salary = 80000)
  (h6 : d.low_salary = 25000)
  (h7 : d.default_salary = 40000)
  (h8 : d.high_salary_ratio = 1/5)
  (h9 : d.mid_salary_ratio = 1/10)
  (h10 : d.low_salary_ratio = 1/20)
  (h11 : g.initial_salary = 25000)
  (h12 : g.yearly_increase = 3000)
  (h13 : g.years = 4)
  : expected_salary d = 39625 ∧ expected_salary d - fyodor_salary g = 2625 := by
  sorry


end NUMINAMATH_CALUDE_salary_comparison_l1963_196341


namespace NUMINAMATH_CALUDE_sams_age_l1963_196378

theorem sams_age (sam masc : ℕ) 
  (h1 : masc = sam + 7)
  (h2 : sam + masc = 27) : 
  sam = 10 := by
sorry

end NUMINAMATH_CALUDE_sams_age_l1963_196378


namespace NUMINAMATH_CALUDE_total_fruit_weight_l1963_196389

/-- The total weight of fruit sold by an orchard -/
theorem total_fruit_weight (frozen_fruit fresh_fruit : ℕ) 
  (h1 : frozen_fruit = 3513)
  (h2 : fresh_fruit = 6279) :
  frozen_fruit + fresh_fruit = 9792 := by
  sorry

end NUMINAMATH_CALUDE_total_fruit_weight_l1963_196389


namespace NUMINAMATH_CALUDE_root_value_theorem_l1963_196340

theorem root_value_theorem (a : ℝ) : 
  a^2 - 4*a + 3 = 0 → -2*a^2 + 8*a - 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l1963_196340


namespace NUMINAMATH_CALUDE_no_x_squared_term_l1963_196383

theorem no_x_squared_term (a : ℝ) : 
  (∀ x, (x + 1) * (x^2 - 5*a*x + a) = x^3 + (-4*a)*x + a) → a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l1963_196383


namespace NUMINAMATH_CALUDE_city_population_ratio_l1963_196369

/-- The population of Lake View -/
def lake_view_population : ℕ := 24000

/-- The difference between Lake View and Seattle populations -/
def population_difference : ℕ := 4000

/-- The total population of the three cities -/
def total_population : ℕ := 56000

/-- The ratio of Boise's population to Seattle's population -/
def population_ratio : ℚ := 3 / 5

theorem city_population_ratio :
  ∃ (boise seattle : ℕ),
    boise + seattle + lake_view_population = total_population ∧
    lake_view_population = seattle + population_difference ∧
    population_ratio = boise / seattle := by
  sorry

end NUMINAMATH_CALUDE_city_population_ratio_l1963_196369


namespace NUMINAMATH_CALUDE_replaced_person_weight_l1963_196388

/-- Proves that the weight of the replaced person is 35 kg given the conditions -/
theorem replaced_person_weight (initial_count : Nat) (weight_increase : Real) (new_person_weight : Real) :
  initial_count = 8 ∧ 
  weight_increase = 2.5 ∧ 
  new_person_weight = 55 →
  (initial_count * weight_increase = new_person_weight - (initial_count * weight_increase - new_person_weight)) :=
by
  sorry

#check replaced_person_weight

end NUMINAMATH_CALUDE_replaced_person_weight_l1963_196388


namespace NUMINAMATH_CALUDE_clothing_percentage_is_half_l1963_196304

/-- The percentage of total amount spent on clothing -/
def clothing_percentage : ℝ := sorry

/-- The percentage of total amount spent on food -/
def food_percentage : ℝ := 0.20

/-- The percentage of total amount spent on other items -/
def other_percentage : ℝ := 0.30

/-- The tax rate on clothing -/
def clothing_tax_rate : ℝ := 0.05

/-- The tax rate on food -/
def food_tax_rate : ℝ := 0

/-- The tax rate on other items -/
def other_tax_rate : ℝ := 0.10

/-- The total tax rate as a percentage of the total amount spent excluding taxes -/
def total_tax_rate : ℝ := 0.055

theorem clothing_percentage_is_half :
  clothing_percentage +
  food_percentage +
  other_percentage = 1 ∧
  clothing_percentage * clothing_tax_rate +
  food_percentage * food_tax_rate +
  other_percentage * other_tax_rate = total_tax_rate →
  clothing_percentage = 0.5 := by sorry

end NUMINAMATH_CALUDE_clothing_percentage_is_half_l1963_196304


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1963_196316

theorem tan_alpha_value (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 1/3) 
  (h2 : Real.sin (2 * α) > 0) : 
  Real.tan α = Real.sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1963_196316


namespace NUMINAMATH_CALUDE_fuel_cost_calculation_l1963_196301

/-- Calculates the total cost to fill up both a truck's diesel tank and a car's gasoline tank --/
def total_fuel_cost (truck_capacity : ℝ) (car_capacity : ℝ) (truck_fullness : ℝ) (car_fullness : ℝ) (diesel_price : ℝ) (gasoline_price : ℝ) : ℝ :=
  let truck_to_fill := truck_capacity * (1 - truck_fullness)
  let car_to_fill := car_capacity * (1 - car_fullness)
  truck_to_fill * diesel_price + car_to_fill * gasoline_price

/-- Theorem stating that the total cost to fill up both tanks is $75.75 --/
theorem fuel_cost_calculation :
  total_fuel_cost 25 15 0.5 (1/3) 3.5 3.2 = 75.75 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_calculation_l1963_196301


namespace NUMINAMATH_CALUDE_negative_one_exp_zero_l1963_196342

-- Define the exponentiation rule for any non-zero number raised to the power of 0
axiom exp_zero (x : ℝ) (h : x ≠ 0) : x^0 = 1

-- State the theorem to be proved
theorem negative_one_exp_zero : (-1 : ℝ)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_exp_zero_l1963_196342


namespace NUMINAMATH_CALUDE_park_area_l1963_196310

/-- A rectangular park with width one-third of length and perimeter 72 meters has area 243 square meters -/
theorem park_area (w : ℝ) (l : ℝ) : 
  w > 0 → l > 0 → w = l / 3 → 2 * (w + l) = 72 → w * l = 243 := by
  sorry

end NUMINAMATH_CALUDE_park_area_l1963_196310


namespace NUMINAMATH_CALUDE_exists_larger_area_same_perimeter_l1963_196313

/-- A convex figure in a 2D plane -/
structure ConvexFigure where
  -- We assume some representation of a convex figure
  -- This could be a set of points, a function, etc.
  -- The exact implementation is not crucial for this statement

/-- Perimeter of a convex figure -/
noncomputable def perimeter (f : ConvexFigure) : ℝ :=
  sorry

/-- Area of a convex figure -/
noncomputable def area (f : ConvexFigure) : ℝ :=
  sorry

/-- Predicate to check if a convex figure is a circle -/
def isCircle (f : ConvexFigure) : Prop :=
  sorry

/-- Theorem: For any non-circular convex figure, there exists another figure
    with the same perimeter but larger area -/
theorem exists_larger_area_same_perimeter (Φ : ConvexFigure) 
    (h : ¬isCircle Φ) : 
    ∃ (Φ' : ConvexFigure), 
      perimeter Φ' = perimeter Φ ∧ 
      area Φ' > area Φ :=
  sorry

end NUMINAMATH_CALUDE_exists_larger_area_same_perimeter_l1963_196313


namespace NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_five_l1963_196371

def numbers : List Nat := [3546, 3550, 3565, 3570, 3585]

def is_divisible_by_five (n : Nat) : Bool :=
  n % 5 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem product_of_digits_not_divisible_by_five :
  ∃ n ∈ numbers, ¬is_divisible_by_five n ∧
  units_digit n * tens_digit n = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_five_l1963_196371


namespace NUMINAMATH_CALUDE_total_devices_l1963_196311

theorem total_devices (computers televisions : ℕ) : 
  computers = 32 → televisions = 66 → computers + televisions = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_devices_l1963_196311


namespace NUMINAMATH_CALUDE_circle_center_condition_l1963_196392

-- Define the equation of the circle
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 2*y + 3 - a = 0

-- Define the condition for the center to be in the second quadrant
def center_in_second_quadrant (a : ℝ) : Prop :=
  a < 0 ∧ 1 > 0

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  a < -2

-- Theorem statement
theorem circle_center_condition (a : ℝ) :
  (∃ x y : ℝ, circle_equation x y a) ∧
  center_in_second_quadrant a
  ↔ a_range a :=
sorry

end NUMINAMATH_CALUDE_circle_center_condition_l1963_196392


namespace NUMINAMATH_CALUDE_dot_product_AB_AC_t_value_l1963_196319

-- Define the points
def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (2, 3)
def C : ℝ × ℝ := (-2, -1)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def OB : ℝ × ℝ := B
def OC : ℝ × ℝ := C

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem 1: Dot product of AB and AC
theorem dot_product_AB_AC : dot_product AB AC = 2 := by sorry

-- Theorem 2: Value of t
theorem t_value : ∃ t : ℝ, t = -3 ∧ dot_product (AB.1 - t * OC.1, AB.2 - t * OC.2) OB = 0 := by sorry

end NUMINAMATH_CALUDE_dot_product_AB_AC_t_value_l1963_196319


namespace NUMINAMATH_CALUDE_mall_walking_methods_l1963_196356

/-- The number of entrances in the mall -/
def num_entrances : ℕ := 4

/-- The number of different walking methods through the mall -/
def num_walking_methods : ℕ := num_entrances * (num_entrances - 1)

/-- Theorem stating the number of different walking methods -/
theorem mall_walking_methods :
  num_walking_methods = 12 :=
sorry

end NUMINAMATH_CALUDE_mall_walking_methods_l1963_196356


namespace NUMINAMATH_CALUDE_moon_temperature_difference_l1963_196361

/-- The temperature difference between day and night on the moon's surface. -/
def moonTemperatureDifference (dayTemp : ℝ) (nightTemp : ℝ) : ℝ :=
  dayTemp - nightTemp

/-- Theorem stating the temperature difference on the moon's surface. -/
theorem moon_temperature_difference :
  moonTemperatureDifference 127 (-183) = 310 := by
  sorry

end NUMINAMATH_CALUDE_moon_temperature_difference_l1963_196361


namespace NUMINAMATH_CALUDE_calculation_proofs_l1963_196365

theorem calculation_proofs :
  (1 - 2^3 / 8 - 1/4 * (-2)^2 = -2) ∧
  ((-1/12 - 1/16 + 3/4 - 1/6) * (-48) = -21) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l1963_196365


namespace NUMINAMATH_CALUDE_burglar_sentence_l1963_196372

def painting_values : List ℝ := [9385, 12470, 7655, 8120, 13880]
def base_sentence_rate : ℝ := 3000
def assault_sentence : ℝ := 1.5
def resisting_arrest_sentence : ℝ := 2
def prior_offense_penalty : ℝ := 0.25

def calculate_total_sentence (values : List ℝ) (rate : ℝ) (assault : ℝ) (resisting : ℝ) (penalty : ℝ) : ℕ :=
  sorry

theorem burglar_sentence :
  calculate_total_sentence painting_values base_sentence_rate assault_sentence resisting_arrest_sentence prior_offense_penalty = 26 :=
sorry

end NUMINAMATH_CALUDE_burglar_sentence_l1963_196372


namespace NUMINAMATH_CALUDE_a_upper_bound_l1963_196374

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 2 * a (n / 2) + 3 * a (n / 3) + 6 * a (n / 6)

theorem a_upper_bound : ∀ n : ℕ, a n ≤ 10 * n^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_a_upper_bound_l1963_196374


namespace NUMINAMATH_CALUDE_gasoline_price_increase_l1963_196391

theorem gasoline_price_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (price_increase : ℝ) 
  (budget_increase : ℝ) 
  (quantity_decrease : ℝ) 
  (h1 : budget_increase = 0.15)
  (h2 : quantity_decrease = 0.08000000000000007)
  (h3 : original_price > 0)
  (h4 : original_quantity > 0) :
  original_price * original_quantity * (1 + budget_increase) = 
  (original_price * (1 + price_increase)) * (original_quantity * (1 - quantity_decrease)) →
  price_increase = 0.25 := by
sorry

end NUMINAMATH_CALUDE_gasoline_price_increase_l1963_196391


namespace NUMINAMATH_CALUDE_total_money_value_l1963_196376

-- Define the number of nickels (and quarters)
def num_coins : ℕ := 40

-- Define the value of a nickel in cents
def nickel_value : ℕ := 5

-- Define the value of a quarter in cents
def quarter_value : ℕ := 25

-- Define the conversion rate from cents to dollars
def cents_per_dollar : ℕ := 100

-- Theorem statement
theorem total_money_value : 
  (num_coins * nickel_value + num_coins * quarter_value) / cents_per_dollar = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_money_value_l1963_196376


namespace NUMINAMATH_CALUDE_profit_increase_l1963_196336

theorem profit_increase (march_profit : ℝ) (h1 : march_profit > 0) : 
  let april_profit := 1.20 * march_profit
  let may_profit := 0.80 * april_profit
  let june_profit := 1.50 * may_profit
  (june_profit - march_profit) / march_profit * 100 = 44 := by
sorry

end NUMINAMATH_CALUDE_profit_increase_l1963_196336


namespace NUMINAMATH_CALUDE_corner_square_probability_l1963_196327

-- Define the grid size
def gridSize : Nat := 4

-- Define the number of squares to be selected
def squaresSelected : Nat := 3

-- Define the number of corner squares
def cornerSquares : Nat := 4

-- Define the total number of squares
def totalSquares : Nat := gridSize * gridSize

-- Define the probability of selecting at least one corner square
def probabilityAtLeastOneCorner : Rat := 17 / 28

theorem corner_square_probability :
  (1 : Rat) - (Nat.choose (totalSquares - cornerSquares) squaresSelected : Rat) / 
  (Nat.choose totalSquares squaresSelected) = probabilityAtLeastOneCorner := by
  sorry

end NUMINAMATH_CALUDE_corner_square_probability_l1963_196327


namespace NUMINAMATH_CALUDE_richards_day2_distance_l1963_196395

/-- Richard's journey from Cincinnati to New York City -/
def richards_journey (day2_distance : ℝ) : Prop :=
  let total_distance : ℝ := 70
  let day1_distance : ℝ := 20
  let day3_distance : ℝ := 10
  let remaining_distance : ℝ := 36
  (day2_distance < day1_distance / 2) ∧
  (day1_distance + day2_distance + day3_distance + remaining_distance = total_distance)

theorem richards_day2_distance :
  ∃ (day2_distance : ℝ), richards_journey day2_distance ∧ day2_distance = 4 := by
  sorry

end NUMINAMATH_CALUDE_richards_day2_distance_l1963_196395


namespace NUMINAMATH_CALUDE_ice_cream_scoops_l1963_196324

/-- The number of ice cream scoops served to a family at Ice Cream Palace -/
def total_scoops (single_cone waffle_bowl banana_split double_cone : ℕ) : ℕ :=
  single_cone + waffle_bowl + banana_split + double_cone

/-- Theorem: Given the conditions of the ice cream orders, the total number of scoops served is 10 -/
theorem ice_cream_scoops :
  ∀ (single_cone waffle_bowl banana_split double_cone : ℕ),
    single_cone = 1 →
    banana_split = 3 * single_cone →
    waffle_bowl = banana_split + 1 →
    double_cone = 2 →
    total_scoops single_cone waffle_bowl banana_split double_cone = 10 :=
by
  sorry

#check ice_cream_scoops

end NUMINAMATH_CALUDE_ice_cream_scoops_l1963_196324


namespace NUMINAMATH_CALUDE_coordinate_sum_of_point_B_l1963_196306

theorem coordinate_sum_of_point_B (A B : ℝ × ℝ) : 
  A = (0, 0) →
  B.2 = 5 →
  (B.2 - A.2) / (B.1 - A.1) = 3/4 →
  B.1 + B.2 = 35/3 := by
sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_point_B_l1963_196306


namespace NUMINAMATH_CALUDE_johns_sister_age_l1963_196390

/-- Given the ages of John, his dad, and his sister, prove that John's sister is 37.5 years old -/
theorem johns_sister_age :
  ∀ (john dad sister : ℝ),
  dad = john + 15 →
  john + dad = 100 →
  sister = john - 5 →
  sister = 37.5 := by
sorry

end NUMINAMATH_CALUDE_johns_sister_age_l1963_196390


namespace NUMINAMATH_CALUDE_exists_password_with_twenty_combinations_l1963_196381

/-- Represents a character in the password --/
structure PasswordChar :=
  (value : Char)

/-- Represents a 5-character password --/
structure Password :=
  (chars : Fin 5 → PasswordChar)

/-- Counts the number of unique permutations of a password --/
def countUniqueCombinations (password : Password) : ℕ :=
  sorry

/-- Theorem: There exists a 5-character password with exactly 20 different combinations --/
theorem exists_password_with_twenty_combinations : 
  ∃ (password : Password), countUniqueCombinations password = 20 := by
  sorry

end NUMINAMATH_CALUDE_exists_password_with_twenty_combinations_l1963_196381


namespace NUMINAMATH_CALUDE_compute_expression_l1963_196367

theorem compute_expression : 8 * (243 / 3 + 81 / 9 + 25 / 25 + 3) = 752 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1963_196367


namespace NUMINAMATH_CALUDE_initial_girls_count_l1963_196382

theorem initial_girls_count (n : ℕ) : 
  n > 0 →
  (n : ℚ) / 2 - 2 = (2 * n : ℚ) / 5 →
  (n : ℚ) / 2 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_girls_count_l1963_196382


namespace NUMINAMATH_CALUDE_faye_initial_giveaway_l1963_196349

/-- The number of coloring books Faye bought initially -/
def initial_books : ℝ := 48.0

/-- The number of coloring books Faye gave away after the initial giveaway -/
def additional_giveaway : ℝ := 3.0

/-- The number of coloring books Faye has left -/
def remaining_books : ℝ := 11.0

/-- The number of coloring books Faye gave away initially -/
def initial_giveaway : ℝ := initial_books - additional_giveaway - remaining_books

theorem faye_initial_giveaway : initial_giveaway = 34.0 := by
  sorry

end NUMINAMATH_CALUDE_faye_initial_giveaway_l1963_196349


namespace NUMINAMATH_CALUDE_magical_red_knights_fraction_l1963_196357

theorem magical_red_knights_fraction (total : ℕ) (red blue magical : ℕ) :
  total > 0 →
  red + blue = total →
  red = (3 * total) / 8 →
  magical = total / 4 →
  ∃ (p q : ℕ), q > 0 ∧ 
    red * p * 3 * blue = blue * q * 3 * red ∧
    red * p * q + blue * p * q = magical * q * 3 →
  (3 : ℚ) / 7 = p / q := by
  sorry

end NUMINAMATH_CALUDE_magical_red_knights_fraction_l1963_196357


namespace NUMINAMATH_CALUDE_grandfather_is_73_l1963_196373

/-- Xiaowen's age in years -/
def xiaowens_age : ℕ := 13

/-- Xiaowen's grandfather's age in years -/
def grandfathers_age : ℕ := 5 * xiaowens_age + 8

/-- Theorem stating that Xiaowen's grandfather is 73 years old -/
theorem grandfather_is_73 : grandfathers_age = 73 := by
  sorry

end NUMINAMATH_CALUDE_grandfather_is_73_l1963_196373


namespace NUMINAMATH_CALUDE_infinite_valid_points_l1963_196380

-- Define the circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 25}

-- Define the center of the circle
def Center : ℝ × ℝ := (0, 0)

-- Define the diameter endpoints
def DiameterEndpoints : (ℝ × ℝ) × (ℝ × ℝ) := ((-5, 0), (5, 0))

-- Define the condition for points P
def ValidPoint (p : ℝ × ℝ) : Prop :=
  let (a, b) := DiameterEndpoints
  ((p.1 - a.1)^2 + (p.2 - a.2)^2) + ((p.1 - b.1)^2 + (p.2 - b.2)^2) = 50 ∧
  (p.1^2 + p.2^2 < 25)

-- Theorem statement
theorem infinite_valid_points : ∃ (S : Set (ℝ × ℝ)), Set.Infinite S ∧ ∀ p ∈ S, p ∈ Circle ∧ ValidPoint p :=
sorry

end NUMINAMATH_CALUDE_infinite_valid_points_l1963_196380


namespace NUMINAMATH_CALUDE_calendar_reuse_2080_l1963_196330

/-- A year is reusable after a fixed number of years if both years have the same leap year status and start on the same day of the week. -/
def is_calendar_reusable (initial_year target_year : ℕ) : Prop :=
  (initial_year % 4 = 0 ↔ target_year % 4 = 0) ∧
  (initial_year + 1) % 7 = (target_year + 1) % 7

/-- The theorem states that the calendar of the year 2080 can be reused after 28 years. -/
theorem calendar_reuse_2080 :
  let initial_year := 2080
  let year_difference := 28
  is_calendar_reusable initial_year (initial_year + year_difference) :=
by sorry

end NUMINAMATH_CALUDE_calendar_reuse_2080_l1963_196330


namespace NUMINAMATH_CALUDE_sum_of_integers_l1963_196307

theorem sum_of_integers (x y : ℕ+) (h1 : x.val - y.val = 8) (h2 : x.val * y.val = 180) : 
  x.val + y.val = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1963_196307


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1963_196352

theorem unique_positive_solution (x y z : ℝ) :
  x > 0 ∧ y > 0 ∧ z > 0 →
  x^3 + 2*y^2 + 1/(4*z) = 1 →
  y^3 + 2*z^2 + 1/(4*x) = 1 →
  z^3 + 2*x^2 + 1/(4*y) = 1 →
  x = (-1 + Real.sqrt 3) / 2 ∧
  y = (-1 + Real.sqrt 3) / 2 ∧
  z = (-1 + Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1963_196352


namespace NUMINAMATH_CALUDE_find_p_l1963_196399

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (p w : ℂ) : Prop := 10 * p - w = 50000

-- State the theorem
theorem find_p :
  ∀ (p w : ℂ),
  equation p w →
  (10 : ℂ) = 2 →
  w = 10 + 250 * i →
  p = 5001 + 25 * i :=
by sorry

end NUMINAMATH_CALUDE_find_p_l1963_196399


namespace NUMINAMATH_CALUDE_count_triangles_five_divisions_l1963_196317

/-- Represents an equilateral triangle with sides divided into n equal parts -/
structure DividedEquilateralTriangle (n : ℕ) where
  -- Any additional properties can be added here if needed

/-- Counts the number of distinct equilateral triangles of different sizes in a divided equilateral triangle -/
def count_distinct_triangles (t : DividedEquilateralTriangle 5) : ℕ :=
  sorry

/-- Theorem stating that there are 48 distinct equilateral triangles in a triangle divided into 5 parts -/
theorem count_triangles_five_divisions :
  ∀ (t : DividedEquilateralTriangle 5), count_distinct_triangles t = 48 :=
by sorry

end NUMINAMATH_CALUDE_count_triangles_five_divisions_l1963_196317


namespace NUMINAMATH_CALUDE_min_students_required_l1963_196366

/-- Represents a set of days in which a student participates -/
def ParticipationSet := Finset (Fin 6)

/-- The property that for any 3 days, there's a student participating in all 3 -/
def CoversAllTriples (sets : Finset ParticipationSet) : Prop :=
  ∀ (days : Finset (Fin 6)), days.card = 3 → ∃ s ∈ sets, days ⊆ s

/-- The property that no student participates in all 4 days of any 4-day selection -/
def NoQuadruplesCovered (sets : Finset ParticipationSet) : Prop :=
  ∀ (days : Finset (Fin 6)), days.card = 4 → ∀ s ∈ sets, ¬(days ⊆ s)

/-- The main theorem stating the minimum number of students required -/
theorem min_students_required :
  ∃ (sets : Finset ParticipationSet),
    sets.card = 20 ∧
    (∀ s ∈ sets, s.card = 3) ∧
    CoversAllTriples sets ∧
    NoQuadruplesCovered sets ∧
    (∀ (sets' : Finset ParticipationSet),
      (∀ s' ∈ sets', s'.card = 3) →
      CoversAllTriples sets' →
      NoQuadruplesCovered sets' →
      sets'.card ≥ 20) :=
sorry

end NUMINAMATH_CALUDE_min_students_required_l1963_196366


namespace NUMINAMATH_CALUDE_circle_properties_l1963_196346

/-- The parabola to which the circle is tangent -/
def parabola (x y : ℝ) : Prop := y^2 = 5*x + 9

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := 2*x^2 - 10*x*y - 31*y^2 + 175*x - 6*y + 297 = 0

/-- Points through which the circle passes -/
def point_P : ℝ × ℝ := (0, 3)
def point_Q : ℝ × ℝ := (-1, -2)
def point_A : ℝ × ℝ := (-2, 1)

theorem circle_properties :
  (∀ x y : ℝ, circle_equation x y → ∃ r : ℝ, (x - 0)^2 + (y - 0)^2 = r^2) ∧
  (parabola (point_P.1) (point_P.2)) ∧
  (parabola (point_Q.1) (point_Q.2)) ∧
  (circle_equation (point_P.1) (point_P.2)) ∧
  (circle_equation (point_Q.1) (point_Q.2)) ∧
  (circle_equation (point_A.1) (point_A.2)) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l1963_196346


namespace NUMINAMATH_CALUDE_number_increased_by_twenty_percent_l1963_196368

theorem number_increased_by_twenty_percent (x : ℝ) : x * 1.2 = 1080 ↔ x = 900 := by sorry

end NUMINAMATH_CALUDE_number_increased_by_twenty_percent_l1963_196368


namespace NUMINAMATH_CALUDE_first_interest_rate_is_eight_percent_l1963_196350

/-- Proves that the first interest rate is 8% given the problem conditions -/
theorem first_interest_rate_is_eight_percent 
  (total_investment : ℝ) 
  (first_investment : ℝ) 
  (second_investment : ℝ) 
  (second_rate : ℝ) 
  (h1 : total_investment = 5400)
  (h2 : first_investment = 3000)
  (h3 : second_investment = total_investment - first_investment)
  (h4 : second_rate = 0.10)
  (h5 : first_investment * (first_rate : ℝ) = second_investment * second_rate) :
  first_rate = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_first_interest_rate_is_eight_percent_l1963_196350


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1963_196302

theorem consecutive_integers_average (x y : ℝ) : 
  (∃ (a b : ℝ), a = x + 2 ∧ b = x + 4 ∧ y = (x + a + b) / 3) →
  (x + 3 + (x + 4) + (x + 5) + (x + 6)) / 4 = x + 4.5 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1963_196302


namespace NUMINAMATH_CALUDE_picture_on_wall_l1963_196355

theorem picture_on_wall (wall_width picture_width : ℝ) 
  (hw : wall_width = 22) 
  (hp : picture_width = 4) : 
  (wall_width - picture_width) / 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_picture_on_wall_l1963_196355


namespace NUMINAMATH_CALUDE_gcf_of_75_and_100_l1963_196397

theorem gcf_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_100_l1963_196397


namespace NUMINAMATH_CALUDE_log_difference_equals_negative_two_l1963_196315

-- Define the common logarithm (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_difference_equals_negative_two :
  log10 (1/4) - log10 25 = -2 := by sorry

end NUMINAMATH_CALUDE_log_difference_equals_negative_two_l1963_196315


namespace NUMINAMATH_CALUDE_tangent_slope_of_circle_l1963_196314

/-- Given a circle with center (2,3) and a point (7,4) on the circle,
    the slope of the tangent line at (7,4) is -5. -/
theorem tangent_slope_of_circle (center : ℝ × ℝ) (point : ℝ × ℝ) : 
  center = (2, 3) →
  point = (7, 4) →
  (point.2 - center.2) / (point.1 - center.1) = 1/5 →
  -(point.1 - center.1) / (point.2 - center.2) = -5 :=
by sorry


end NUMINAMATH_CALUDE_tangent_slope_of_circle_l1963_196314


namespace NUMINAMATH_CALUDE_A_not_always_in_second_quadrant_l1963_196308

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of being on the negative x-axis -/
def isOnNegativeXAxis (p : Point) : Prop :=
  p.x < 0 ∧ p.y = 0

/-- The point A(-a^2-1,|b|) -/
def A (a b : ℝ) : Point :=
  { x := -a^2 - 1, y := |b| }

/-- Theorem stating that A(-a^2-1,|b|) is not always in the second quadrant -/
theorem A_not_always_in_second_quadrant :
  ∃ a b : ℝ, ¬(isInSecondQuadrant (A a b)) ∧ (isInSecondQuadrant (A a b) ∨ isOnNegativeXAxis (A a b)) :=
sorry

end NUMINAMATH_CALUDE_A_not_always_in_second_quadrant_l1963_196308


namespace NUMINAMATH_CALUDE_prob_two_primes_equals_216_625_l1963_196387

-- Define a 10-sided die
def tenSidedDie : Finset ℕ := Finset.range 10

-- Define the set of prime numbers on a 10-sided die
def primes : Finset ℕ := {2, 3, 5, 7}

-- Define the probability of rolling a prime number on one die
def probPrime : ℚ := (primes.card : ℚ) / (tenSidedDie.card : ℚ)

-- Define the probability of not rolling a prime number on one die
def probNotPrime : ℚ := 1 - probPrime

-- Define the number of ways to choose 2 dice out of 4
def waysToChoose : ℕ := Nat.choose 4 2

-- Define the probability of exactly two dice showing a prime number
def probTwoPrimes : ℚ := (waysToChoose : ℚ) * probPrime^2 * probNotPrime^2

-- Theorem statement
theorem prob_two_primes_equals_216_625 : probTwoPrimes = 216 / 625 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_primes_equals_216_625_l1963_196387


namespace NUMINAMATH_CALUDE_professor_seating_arrangements_l1963_196322

/-- Represents the number of chairs in a row -/
def total_chairs : ℕ := 9

/-- Represents the number of students -/
def num_students : ℕ := 6

/-- Represents the number of professors -/
def num_professors : ℕ := 3

/-- Represents the condition that professors cannot sit in the first or last chair -/
def available_chairs : ℕ := total_chairs - 2

/-- Represents the effective number of chair choices after accounting for spacing -/
def effective_choices : ℕ := available_chairs - (num_professors - 1)

/-- The number of ways to choose professor positions -/
def choose_positions : ℕ := Nat.choose effective_choices num_professors

/-- The number of ways to arrange professors in the chosen positions -/
def arrange_professors : ℕ := Nat.factorial num_professors

/-- Theorem stating the number of ways professors can choose their chairs -/
theorem professor_seating_arrangements :
  choose_positions * arrange_professors = 60 := by sorry

end NUMINAMATH_CALUDE_professor_seating_arrangements_l1963_196322


namespace NUMINAMATH_CALUDE_regular_ticket_price_l1963_196344

/-- Calculates the price of each regular ticket given the initial savings,
    VIP ticket information, number of regular tickets, and remaining money. -/
theorem regular_ticket_price
  (initial_savings : ℕ)
  (vip_ticket_count : ℕ)
  (vip_ticket_price : ℕ)
  (regular_ticket_count : ℕ)
  (remaining_money : ℕ)
  (h1 : initial_savings = 500)
  (h2 : vip_ticket_count = 2)
  (h3 : vip_ticket_price = 100)
  (h4 : regular_ticket_count = 3)
  (h5 : remaining_money = 150)
  (h6 : initial_savings ≥ vip_ticket_count * vip_ticket_price + remaining_money) :
  (initial_savings - (vip_ticket_count * vip_ticket_price + remaining_money)) / regular_ticket_count = 50 :=
by sorry

end NUMINAMATH_CALUDE_regular_ticket_price_l1963_196344


namespace NUMINAMATH_CALUDE_rectangle_containment_l1963_196328

/-- A rectangle defined by its width and height -/
structure Rectangle where
  width : ℕ+
  height : ℕ+

/-- The set of all rectangles -/
def RectangleSet : Set Rectangle := {r : Rectangle | True}

/-- One rectangle is contained within another -/
def contained (r1 r2 : Rectangle) : Prop :=
  r1.width ≤ r2.width ∧ r1.height ≤ r2.height

theorem rectangle_containment (h : Set.Infinite RectangleSet) :
  ∃ (r1 r2 : Rectangle), r1 ∈ RectangleSet ∧ r2 ∈ RectangleSet ∧ contained r1 r2 :=
sorry

end NUMINAMATH_CALUDE_rectangle_containment_l1963_196328


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l1963_196318

/-- Two functions representing the graphs -/
def f (x a b : ℝ) : ℝ := -|x - a| + b
def g (x c d : ℝ) : ℝ := |x - c| + d

/-- The theorem stating that a+c = 10 given the intersection points -/
theorem intersection_implies_sum (a b c d : ℝ) :
  (f 2 a b = 5 ∧ g 2 c d = 5) ∧
  (f 8 a b = 3 ∧ g 8 c d = 3) →
  a + c = 10 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l1963_196318


namespace NUMINAMATH_CALUDE_coach_b_baseballs_l1963_196335

/-- The number of baseballs Coach B bought -/
def num_baseballs : ℕ := 14

/-- The cost of each basketball -/
def basketball_cost : ℚ := 29

/-- The cost of each baseball -/
def baseball_cost : ℚ := 5/2

/-- The cost of the baseball bat -/
def bat_cost : ℚ := 18

/-- The number of basketballs Coach A bought -/
def num_basketballs : ℕ := 10

/-- The difference in spending between Coach A and Coach B -/
def spending_difference : ℚ := 237

theorem coach_b_baseballs :
  (num_basketballs * basketball_cost) = 
  spending_difference + (num_baseballs * baseball_cost + bat_cost) :=
by sorry

end NUMINAMATH_CALUDE_coach_b_baseballs_l1963_196335


namespace NUMINAMATH_CALUDE_tangent_line_and_inequality_l1963_196386

open Real

noncomputable def f (x : ℝ) : ℝ := exp x / x

theorem tangent_line_and_inequality :
  (∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ exp 2 * x - 4 * y = 0) ∧
  (∀ x, x > 0 → f x > 2 * (x - log x)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequality_l1963_196386


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l1963_196323

theorem rectangle_triangle_area_ratio : 
  ∀ (L W : ℝ), L > 0 → W > 0 →
  (L * W) / ((1/2) * L * W) = 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l1963_196323


namespace NUMINAMATH_CALUDE_log_inequality_l1963_196334

theorem log_inequality : ∃ (a b : ℝ), 
  (a = Real.log 0.8 / Real.log 0.7) ∧ 
  (b = Real.log 0.4 / Real.log 0.5) ∧ 
  (b > a) ∧ (a > 0) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1963_196334


namespace NUMINAMATH_CALUDE_nancy_alyssa_book_ratio_l1963_196359

def alyssa_books : ℕ := 36
def nancy_books : ℕ := 252

theorem nancy_alyssa_book_ratio :
  nancy_books / alyssa_books = 7 := by
  sorry

end NUMINAMATH_CALUDE_nancy_alyssa_book_ratio_l1963_196359


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1963_196338

/-- The quadratic function f(c) = 3/4*c^2 - 6c + 4 is minimized when c = 4 -/
theorem quadratic_minimum : ∃ (c : ℝ), ∀ (x : ℝ), (3/4 : ℝ) * c^2 - 6*c + 4 ≤ (3/4 : ℝ) * x^2 - 6*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1963_196338


namespace NUMINAMATH_CALUDE_triangle_properties_l1963_196377

/-- Triangle ABC with given properties -/
structure TriangleABC where
  /-- Point B has coordinates (4,4) -/
  B : ℝ × ℝ
  B_coord : B = (4, 4)
  
  /-- The angle bisector of angle A lies on the line y=0 -/
  angle_bisector_A : ℝ → ℝ
  angle_bisector_A_eq : ∀ x, angle_bisector_A x = 0
  
  /-- The altitude from B to side AC lies on the line x-2y+2=0 -/
  altitude_B : ℝ → ℝ
  altitude_B_eq : ∀ x, altitude_B x = (x + 2) / 2

/-- The main theorem stating the properties of the triangle -/
theorem triangle_properties (t : TriangleABC) :
  ∃ (C : ℝ × ℝ) (area : ℝ),
    C = (10, -8) ∧ 
    area = 48 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1963_196377


namespace NUMINAMATH_CALUDE_train_crossing_time_l1963_196394

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time (train_length platform_length : ℝ) (train_speed_kmph : ℝ) : 
  train_length = 250 →
  platform_length = 200 →
  train_speed_kmph = 90 →
  (train_length + platform_length) / (train_speed_kmph * 1000 / 3600) = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1963_196394


namespace NUMINAMATH_CALUDE_rectangle_circle_equality_l1963_196325

/-- The length of a rectangle with width 3 units whose perimeter equals the circumference of a circle with radius 5 units is 5π - 3. -/
theorem rectangle_circle_equality (l : ℝ) : 
  (2 * (l + 3) = 2 * π * 5) → l = 5 * π - 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_equality_l1963_196325


namespace NUMINAMATH_CALUDE_complex_multiplication_l1963_196354

theorem complex_multiplication (R S T : ℂ) : 
  R = 3 + 4*I ∧ S = 2*I ∧ T = 3 - 4*I → R * S * T = 50 * I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1963_196354


namespace NUMINAMATH_CALUDE_special_quadrilateral_not_necessarily_square_l1963_196364

/-- A quadrilateral with perpendicular and bisecting diagonals -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has perpendicular diagonals -/
  perpendicular_diagonals : Bool
  /-- The quadrilateral has bisecting diagonals -/
  bisecting_diagonals : Bool

/-- Definition of a square -/
structure Square where
  /-- All sides of the square are equal -/
  equal_sides : Bool
  /-- All angles of the square are right angles -/
  right_angles : Bool

/-- Theorem stating that a quadrilateral with perpendicular and bisecting diagonals is not necessarily a square -/
theorem special_quadrilateral_not_necessarily_square :
  ∃ (q : SpecialQuadrilateral), q.perpendicular_diagonals ∧ q.bisecting_diagonals ∧
  ∃ (s : Square), ¬(q.perpendicular_diagonals → s.equal_sides ∧ s.right_angles) :=
sorry

end NUMINAMATH_CALUDE_special_quadrilateral_not_necessarily_square_l1963_196364


namespace NUMINAMATH_CALUDE_inequalities_not_necessarily_true_l1963_196351

theorem inequalities_not_necessarily_true
  (x y z a b c : ℝ)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hxa : x < a) (hyb : y > b) (hzc : z < c) :
  ∃ (x' y' z' a' b' c' : ℝ),
    x' < a' ∧ y' > b' ∧ z' < c' ∧
    ¬(x'*z' + y' < a'*z' + b') ∧
    ¬(x'*y' < a'*b') ∧
    ¬((x' + y') / z' < (a' + b') / c') ∧
    ¬(x'*y'*z' < a'*b'*c') :=
by sorry

end NUMINAMATH_CALUDE_inequalities_not_necessarily_true_l1963_196351


namespace NUMINAMATH_CALUDE_book_arrangement_count_l1963_196332

/-- The number of ways to arrange books on a shelf -/
def arrange_books (arabic : ℕ) (german : ℕ) (spanish : ℕ) : ℕ :=
  Nat.factorial (arabic + german + spanish - 2) * Nat.factorial arabic * Nat.factorial spanish

/-- Theorem stating the number of arrangements for the given book configuration -/
theorem book_arrangement_count :
  arrange_books 2 3 4 = 5760 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l1963_196332


namespace NUMINAMATH_CALUDE_tan_alpha_minus_pi_eighth_l1963_196329

theorem tan_alpha_minus_pi_eighth (α : Real) 
  (h : 2 * Real.sin α = Real.sin (α - π/4)) : 
  Real.tan (α - π/8) = 3 - 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_pi_eighth_l1963_196329


namespace NUMINAMATH_CALUDE_smallest_digits_to_append_l1963_196309

def is_divisible_by_all_less_than_10 (n : ℕ) : Prop :=
  ∀ k : ℕ, k < 10 → k > 0 → n % k = 0

def append_digits (base n digits : ℕ) : ℕ :=
  base * (10 ^ digits) + n

theorem smallest_digits_to_append : 
  (∀ k < 4, ¬∃ n : ℕ, n < 10^k ∧ is_divisible_by_all_less_than_10 (append_digits 2014 n k)) ∧
  (∃ n : ℕ, n < 10^4 ∧ is_divisible_by_all_less_than_10 (append_digits 2014 n 4)) :=
sorry

end NUMINAMATH_CALUDE_smallest_digits_to_append_l1963_196309


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_negative_five_l1963_196321

-- Define the sets P and Q
def P : Set ℝ := {y | y^2 - y - 2 > 0}
def Q : Set ℝ := {x | ∃ (a b : ℝ), x^2 + a*x + b ≤ 0}

-- State the theorem
theorem sum_of_a_and_b_is_negative_five 
  (h1 : P ∪ Q = Set.univ)
  (h2 : P ∩ Q = Set.Ioc 2 3)
  : ∃ (a b : ℝ), Q = {x | x^2 + a*x + b ≤ 0} ∧ a + b = -5 :=
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_negative_five_l1963_196321


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l1963_196353

/-- A ball rolling on a half circular track and bouncing on the floor -/
theorem ball_bounce_distance 
  (R : ℝ) -- radius of the half circular track
  (v : ℝ) -- velocity of the ball when leaving the track
  (g : ℝ) -- acceleration due to gravity
  (h : R > 0) -- radius is positive
  (hv : v > 0) -- velocity is positive
  (hg : g > 0) -- gravity is positive
  : ∃ (d : ℝ), d = 2 * R - (2 * v / 3) * Real.sqrt (R / g) :=
by sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l1963_196353
