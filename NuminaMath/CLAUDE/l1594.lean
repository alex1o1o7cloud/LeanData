import Mathlib

namespace NUMINAMATH_CALUDE_optimal_price_and_range_l1594_159425

-- Define the linear relationship between quantity and price
def quantity (x : ℝ) : ℝ := -2 * x + 100

-- Define the cost per item
def cost : ℝ := 20

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost) * quantity x

-- Statement to prove
theorem optimal_price_and_range :
  -- The price that maximizes profit is 35
  (∃ (x_max : ℝ), x_max = 35 ∧ ∀ (x : ℝ), profit x ≤ profit x_max) ∧
  -- The range of prices that ensures at least 30 items sold and a profit of at least 400
  (∀ (x : ℝ), 30 ≤ x ∧ x ≤ 35 ↔ quantity x ≥ 30 ∧ profit x ≥ 400) :=
by sorry

end NUMINAMATH_CALUDE_optimal_price_and_range_l1594_159425


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1594_159409

theorem min_value_quadratic (x : ℝ) : x^2 + 4*x + 5 ≥ 1 ∧ (x^2 + 4*x + 5 = 1 ↔ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1594_159409


namespace NUMINAMATH_CALUDE_unique_solution_l1594_159417

/-- Sum of digits function for positive integers in base 10 -/
def S (n : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that 17 is the only positive integer solution to the equation -/
theorem unique_solution : ∀ n : ℕ+, (n : ℕ)^3 = 8 * (S n)^3 + 6 * (S n) * (n : ℕ) + 1 ↔ n = 17 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1594_159417


namespace NUMINAMATH_CALUDE_car_count_total_l1594_159400

/-- Given the car counting scenario, prove the total count of cars. -/
theorem car_count_total (jared_count : ℕ) (ann_count : ℕ) (alfred_count : ℕ) :
  jared_count = 300 →
  ann_count = (115 * jared_count) / 100 →
  alfred_count = ann_count - 7 →
  jared_count + ann_count + alfred_count = 983 :=
by sorry

end NUMINAMATH_CALUDE_car_count_total_l1594_159400


namespace NUMINAMATH_CALUDE_log_inequality_l1594_159439

/-- The number of distinct prime factors of a positive integer -/
def num_distinct_prime_factors (n : ℕ+) : ℕ :=
  sorry

theorem log_inequality (n : ℕ+) :
  Real.log n ≥ (num_distinct_prime_factors n : ℝ) * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_log_inequality_l1594_159439


namespace NUMINAMATH_CALUDE_one_adult_in_family_l1594_159492

/-- Represents the cost of tickets for a family visit to an aquarium -/
structure AquariumTickets where
  adultPrice : ℕ
  childPrice : ℕ
  numChildren : ℕ
  totalCost : ℕ

/-- Calculates the number of adults in the family based on ticket prices and total cost -/
def calculateAdults (tickets : AquariumTickets) : ℕ :=
  (tickets.totalCost - tickets.childPrice * tickets.numChildren) / tickets.adultPrice

/-- Theorem stating that for the given ticket prices and family composition, there is 1 adult -/
theorem one_adult_in_family (tickets : AquariumTickets) 
  (h1 : tickets.adultPrice = 35)
  (h2 : tickets.childPrice = 20)
  (h3 : tickets.numChildren = 6)
  (h4 : tickets.totalCost = 155) : 
  calculateAdults tickets = 1 := by
  sorry

#eval calculateAdults { adultPrice := 35, childPrice := 20, numChildren := 6, totalCost := 155 }

end NUMINAMATH_CALUDE_one_adult_in_family_l1594_159492


namespace NUMINAMATH_CALUDE_problem_statement_l1594_159459

theorem problem_statement (w x y : ℝ) 
  (h1 : 6/w + 6/x = 6/y) 
  (h2 : w*x = y) 
  (h3 : (w + x)/2 = 0.5) : 
  y = 0.25 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1594_159459


namespace NUMINAMATH_CALUDE_log_problem_l1594_159498

theorem log_problem (y : ℝ) (k : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → 
  (Real.log 125 / Real.log 2 = k * y) → 
  k = 9 := by
sorry

end NUMINAMATH_CALUDE_log_problem_l1594_159498


namespace NUMINAMATH_CALUDE_periodic_even_function_theorem_l1594_159496

def periodic_even_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 2) = f x) ∧ 
  (∀ x, f (-x) = f x)

theorem periodic_even_function_theorem (f : ℝ → ℝ) 
  (h_periodic_even : periodic_even_function f)
  (h_interval : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
  sorry

end NUMINAMATH_CALUDE_periodic_even_function_theorem_l1594_159496


namespace NUMINAMATH_CALUDE_abs_ac_plus_bd_le_one_l1594_159488

theorem abs_ac_plus_bd_le_one (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 1) : 
  |a*c + b*d| ≤ 1 := by sorry

end NUMINAMATH_CALUDE_abs_ac_plus_bd_le_one_l1594_159488


namespace NUMINAMATH_CALUDE_day_90_of_year_N_minus_1_is_thursday_l1594_159476

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a specific day in a year -/
structure DayInYear where
  year : ℤ
  dayNumber : ℕ

/-- Returns the day of the week for a given day in a year -/
def dayOfWeek (d : DayInYear) : DayOfWeek :=
  sorry

/-- Checks if a given year is a leap year -/
def isLeapYear (year : ℤ) : Bool :=
  sorry

theorem day_90_of_year_N_minus_1_is_thursday
  (N : ℤ)
  (h1 : dayOfWeek ⟨N, 150⟩ = DayOfWeek.Sunday)
  (h2 : dayOfWeek ⟨N + 2, 220⟩ = DayOfWeek.Sunday) :
  dayOfWeek ⟨N - 1, 90⟩ = DayOfWeek.Thursday :=
by sorry

end NUMINAMATH_CALUDE_day_90_of_year_N_minus_1_is_thursday_l1594_159476


namespace NUMINAMATH_CALUDE_triangle_exists_l1594_159418

/-- Represents a point in 2D space with integer coordinates -/
structure Point :=
  (x : Int) (y : Int)

/-- Calculates the square of the distance between two points -/
def distanceSquared (p q : Point) : Int :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (a b c : Point) : Rat :=
  let det := a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)
  Rat.ofInt (abs det) / 2

/-- Theorem stating the existence of a triangle with the specified properties -/
theorem triangle_exists : ∃ (a b c : Point),
  (triangleArea a b c < 1) ∧
  (distanceSquared a b > 4) ∧
  (distanceSquared b c > 4) ∧
  (distanceSquared c a > 4) :=
sorry

end NUMINAMATH_CALUDE_triangle_exists_l1594_159418


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l1594_159473

/-- Given a quadratic equation x^2 + 12x + k = 0 where k is a real number,
    if the nonzero roots are in the ratio 3:1, then k = 27. -/
theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 12*x + k = 0 ∧ y^2 + 12*y + k = 0) → k = 27 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l1594_159473


namespace NUMINAMATH_CALUDE_roots_sum_greater_than_a_l1594_159414

noncomputable section

-- Define the function f(x) = x ln x
def f (x : ℝ) : ℝ := x * Real.log x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := Real.log x + 1

-- Define the function F(x) = x^2 - a[x + f'(x)] + 2x
def F (a : ℝ) (x : ℝ) : ℝ := x^2 - a * (x + f' x) + 2 * x

-- Theorem statement
theorem roots_sum_greater_than_a (a m x₁ x₂ : ℝ) 
  (h₁ : x₁ ≠ x₂)
  (h₂ : F a x₁ = m)
  (h₃ : F a x₂ = m)
  : x₁ + x₂ > a :=
sorry

end

end NUMINAMATH_CALUDE_roots_sum_greater_than_a_l1594_159414


namespace NUMINAMATH_CALUDE_derivative_implies_power_l1594_159431

/-- Given a function f(x) = m * x^(m-n) where its derivative f'(x) = 8 * x^3,
    prove that m^n = 1/4 -/
theorem derivative_implies_power (m n : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = m * x^(m-n))
  (h2 : ∀ x, deriv f x = 8 * x^3) :
  m^n = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_derivative_implies_power_l1594_159431


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1594_159442

theorem sufficient_not_necessary (a : ℝ) : 
  (a = 3 → a^2 = 9) ∧ (∃ b : ℝ, b ≠ 3 ∧ b^2 = 9) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1594_159442


namespace NUMINAMATH_CALUDE_vowel_classification_l1594_159483

-- Define the set of all English letters
def EnglishLetters : Type := Fin 26

-- Define the categories
inductive Category
| one
| two
| three
| four
| five

-- Define the classification function
def classify : EnglishLetters → Category := sorry

-- Define the vowels
def vowels : Fin 5 → EnglishLetters := sorry

-- Theorem statement
theorem vowel_classification :
  (classify (vowels 0) = Category.four) ∧
  (classify (vowels 1) = Category.three) ∧
  (classify (vowels 2) = Category.one) ∧
  (classify (vowels 3) = Category.one) ∧
  (classify (vowels 4) = Category.four) := by
  sorry

end NUMINAMATH_CALUDE_vowel_classification_l1594_159483


namespace NUMINAMATH_CALUDE_tonyas_age_l1594_159485

/-- Proves Tonya's age given the conditions of the problem -/
theorem tonyas_age (john mary tonya : ℕ) 
  (h1 : john = 2 * mary)
  (h2 : john = tonya / 2)
  (h3 : (john + mary + tonya) / 3 = 35) :
  tonya = 60 := by sorry

end NUMINAMATH_CALUDE_tonyas_age_l1594_159485


namespace NUMINAMATH_CALUDE_chocolate_division_l1594_159456

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_given : ℕ) : 
  total_chocolate = 60 / 7 →
  num_piles = 5 →
  piles_given = 2 →
  (total_chocolate / num_piles) * piles_given = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l1594_159456


namespace NUMINAMATH_CALUDE_isosceles_triangles_same_perimeter_area_l1594_159440

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ  -- length of equal sides
  base : ℕ  -- length of the base
  isIsosceles : leg > 0 ∧ base > 0

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * (((t.leg : ℝ)^2 - ((t.base : ℝ)/2)^2).sqrt) / 2

/-- Theorem: There exist two non-congruent isosceles triangles with integer side lengths,
    having the same perimeter and area, where the ratio of their bases is 5:4,
    and we can determine the minimum possible value of their common perimeter. -/
theorem isosceles_triangles_same_perimeter_area :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t1.base = 4 * t2.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s1.base = 4 * s2.base →
      perimeter t1 ≤ perimeter s1 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangles_same_perimeter_area_l1594_159440


namespace NUMINAMATH_CALUDE_puzzle_pieces_problem_l1594_159402

theorem puzzle_pieces_problem (pieces_first : ℕ) (pieces_second : ℕ) (pieces_third : ℕ) :
  pieces_second = pieces_third ∧
  pieces_second = (3 : ℕ) / 2 * pieces_first ∧
  pieces_first + pieces_second + pieces_third = 4000 →
  pieces_first = 1000 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_pieces_problem_l1594_159402


namespace NUMINAMATH_CALUDE_simplify_fraction_l1594_159470

theorem simplify_fraction : 
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 2 + Real.sqrt 7) = 
  -3.6 * (1 + Real.sqrt 2 - 2 * Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1594_159470


namespace NUMINAMATH_CALUDE_solution_range_l1594_159445

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 4 < 0) ↔ (a < -4 ∨ a > 4) := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l1594_159445


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l1594_159437

/-- Circle equation: x^2 + y^2 - 4x - 6y - 3 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y - 3 = 0

/-- Point M(-2, 0) -/
def point_M : ℝ × ℝ := (-2, 0)

/-- First tangent line equation: x + 2 = 0 -/
def tangent_line1 (x y : ℝ) : Prop :=
  x + 2 = 0

/-- Second tangent line equation: 7x + 24y + 14 = 0 -/
def tangent_line2 (x y : ℝ) : Prop :=
  7*x + 24*y + 14 = 0

/-- Theorem stating that the given lines are tangent to the circle through point M -/
theorem tangent_lines_to_circle :
  (∀ x y, tangent_line1 x y → circle_equation x y → x = point_M.1 ∧ y = point_M.2) ∧
  (∀ x y, tangent_line2 x y → circle_equation x y → x = point_M.1 ∧ y = point_M.2) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l1594_159437


namespace NUMINAMATH_CALUDE_cos_thirty_degrees_l1594_159451

theorem cos_thirty_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_thirty_degrees_l1594_159451


namespace NUMINAMATH_CALUDE_arthurs_hamburgers_l1594_159444

/-- Given the prices and quantities of hamburgers and hot dogs purchased over two days,
    prove that Arthur bought 2 hamburgers on the second day. -/
theorem arthurs_hamburgers (H D x : ℚ) : 
  3 * H + 4 * D = 10 →  -- Day 1 purchase
  x * H + 3 * D = 7 →   -- Day 2 purchase
  D = 1 →               -- Price of a hot dog
  x = 2 := by            
  sorry

end NUMINAMATH_CALUDE_arthurs_hamburgers_l1594_159444


namespace NUMINAMATH_CALUDE_path_count_l1594_159412

/-- The number of paths from A to each blue arrow -/
def paths_to_blue : ℕ := 2

/-- The number of blue arrows -/
def num_blue_arrows : ℕ := 2

/-- The number of distinct ways from each blue arrow to each green arrow -/
def paths_blue_to_green : ℕ := 3

/-- The number of green arrows -/
def num_green_arrows : ℕ := 2

/-- The number of distinct final approaches from each green arrow to C -/
def paths_green_to_C : ℕ := 2

/-- The total number of paths from A to C -/
def total_paths : ℕ := 
  paths_to_blue * num_blue_arrows * 
  (paths_blue_to_green * num_blue_arrows) * num_green_arrows * 
  paths_green_to_C

theorem path_count : total_paths = 288 := by
  sorry

end NUMINAMATH_CALUDE_path_count_l1594_159412


namespace NUMINAMATH_CALUDE_salt_to_flour_ratio_l1594_159497

/-- Represents the ingredients for making pizza --/
structure PizzaIngredients where
  water : ℕ
  flour : ℕ
  salt : ℕ

/-- Theorem stating the ratio of salt to flour in the pizza recipe --/
theorem salt_to_flour_ratio (ingredients : PizzaIngredients) : 
  ingredients.water = 10 →
  ingredients.flour = 16 →
  ingredients.water + ingredients.flour + ingredients.salt = 34 →
  ingredients.salt * 2 = ingredients.flour := by
  sorry

end NUMINAMATH_CALUDE_salt_to_flour_ratio_l1594_159497


namespace NUMINAMATH_CALUDE_smartphone_transactions_l1594_159438

def initial_price : ℝ := 300
def selling_price : ℝ := 255
def repurchase_price : ℝ := 275

theorem smartphone_transactions :
  (((initial_price - selling_price) / initial_price) * 100 = 15) ∧
  (((initial_price - repurchase_price) / repurchase_price) * 100 = 9.09) := by
sorry

end NUMINAMATH_CALUDE_smartphone_transactions_l1594_159438


namespace NUMINAMATH_CALUDE_shaded_square_area_ratio_l1594_159477

/-- The ratio of the area of a square with vertices at (2,3), (4,3), (4,5), and (2,5) 
    to the area of a 6x6 square is 1/9. -/
theorem shaded_square_area_ratio : 
  let grid_size : ℕ := 6
  let vertex1 : (ℕ × ℕ) := (2, 3)
  let vertex2 : (ℕ × ℕ) := (4, 3)
  let vertex3 : (ℕ × ℕ) := (4, 5)
  let vertex4 : (ℕ × ℕ) := (2, 5)
  let shaded_square_side : ℕ := vertex2.1 - vertex1.1
  let shaded_square_area : ℕ := shaded_square_side * shaded_square_side
  let grid_area : ℕ := grid_size * grid_size
  (shaded_square_area : ℚ) / grid_area = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_shaded_square_area_ratio_l1594_159477


namespace NUMINAMATH_CALUDE_right_triangle_altitude_l1594_159461

theorem right_triangle_altitude (a b c m : ℝ) (h_positive : a > 0) : 
  b^2 + c^2 = a^2 →   -- Pythagorean theorem
  m^2 = (b - c)^2 →   -- Difference of legs equals altitude
  b * c = a * m →     -- Area relation
  m = (a * (Real.sqrt 5 - 1)) / 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_l1594_159461


namespace NUMINAMATH_CALUDE_school_event_ticket_revenue_l1594_159494

theorem school_event_ticket_revenue :
  ∀ (f h : ℕ) (p : ℚ),
    f + h = 160 →
    f * p + h * (p / 2) = 2400 →
    f * p = 800 :=
by sorry

end NUMINAMATH_CALUDE_school_event_ticket_revenue_l1594_159494


namespace NUMINAMATH_CALUDE_b_value_l1594_159426

theorem b_value (a b c m : ℝ) (h : m = (c * a * b) / (a + b)) : 
  b = (m * a) / (c * a - m) :=
sorry

end NUMINAMATH_CALUDE_b_value_l1594_159426


namespace NUMINAMATH_CALUDE_rectangle_equation_l1594_159478

/-- A rectangle centered at the origin with width 2a and height 2b can be described by the equation
    √x * √(a - x) * √y * √(b - y) = 0, where 0 ≤ x ≤ a and 0 ≤ y ≤ b. -/
theorem rectangle_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ 
    0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ b ∧
    Real.sqrt x * Real.sqrt (a - x) * Real.sqrt y * Real.sqrt (b - y) = 0} =
  {p : ℝ × ℝ | -a ≤ p.1 ∧ p.1 ≤ a ∧ -b ≤ p.2 ∧ p.2 ≤ b} :=
by sorry

end NUMINAMATH_CALUDE_rectangle_equation_l1594_159478


namespace NUMINAMATH_CALUDE_test_probabilities_l1594_159424

def prob_A : ℝ := 0.8
def prob_B : ℝ := 0.6
def prob_C : ℝ := 0.5

theorem test_probabilities :
  let prob_all := prob_A * prob_B * prob_C
  let prob_none := (1 - prob_A) * (1 - prob_B) * (1 - prob_C)
  let prob_at_least_one := 1 - prob_none
  prob_all = 0.24 ∧ prob_at_least_one = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_test_probabilities_l1594_159424


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l1594_159460

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^22 + i^222 = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l1594_159460


namespace NUMINAMATH_CALUDE_trapezoid_top_width_l1594_159447

/-- Proves that a trapezoid with given dimensions has a top width of 14 meters -/
theorem trapezoid_top_width :
  ∀ (area bottom_width height top_width : ℝ),
    area = 880 →
    bottom_width = 8 →
    height = 80 →
    area = (1 / 2) * (top_width + bottom_width) * height →
    top_width = 14 :=
by
  sorry

#check trapezoid_top_width

end NUMINAMATH_CALUDE_trapezoid_top_width_l1594_159447


namespace NUMINAMATH_CALUDE_average_student_headcount_l1594_159422

def student_headcount : List ℕ := [11700, 10900, 11500, 10500, 11600, 10700, 11300]

theorem average_student_headcount : 
  (student_headcount.sum / student_headcount.length : ℚ) = 11029 := by
  sorry

end NUMINAMATH_CALUDE_average_student_headcount_l1594_159422


namespace NUMINAMATH_CALUDE_parabola_with_focus_at_origin_five_l1594_159420

/-- A parabola is a set of points in a plane that are equidistant from a fixed point (focus) and a fixed line (directrix). -/
structure Parabola where
  /-- The focus of the parabola -/
  focus : ℝ × ℝ
  /-- The vertex of the parabola -/
  vertex : ℝ × ℝ

/-- The equation of a parabola given its focus and vertex -/
def parabola_equation (p : Parabola) : ℝ → ℝ → Prop :=
  sorry

theorem parabola_with_focus_at_origin_five : 
  let p : Parabola := { focus := (0, 5), vertex := (0, 0) }
  ∀ x y : ℝ, parabola_equation p x y ↔ x^2 = 20*y :=
sorry

end NUMINAMATH_CALUDE_parabola_with_focus_at_origin_five_l1594_159420


namespace NUMINAMATH_CALUDE_friends_meeting_problem_l1594_159482

/-- Two friends walk in opposite directions and then run towards each other -/
theorem friends_meeting_problem 
  (misha_initial_speed : ℝ) 
  (vasya_initial_speed : ℝ) 
  (initial_walk_time : ℝ) 
  (speed_increase_factor : ℝ) :
  misha_initial_speed = 8 →
  vasya_initial_speed = misha_initial_speed / 2 →
  initial_walk_time = 3/4 →
  speed_increase_factor = 3/2 →
  ∃ (meeting_time total_distance : ℝ),
    meeting_time = 1/2 ∧ 
    total_distance = 18 :=
by sorry

end NUMINAMATH_CALUDE_friends_meeting_problem_l1594_159482


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l1594_159495

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- The angles are complementary (sum to 90°)
  a = 4 * b →   -- The ratio of the angles is 4:1
  b = 18 :=     -- The smaller angle is 18°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l1594_159495


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1594_159443

theorem inequality_solution_set (a x : ℝ) :
  (a^2 - 4) * x^2 + 4 * x - 1 > 0 ↔
  (a = 2 ∨ a = -2 → x > 1/4) ∧
  (a > 2 → x > 1/(a + 2) ∨ x < 1/(2 - a)) ∧
  (a < -2 → x < 1/(a + 2) ∨ x > 1/(2 - a)) ∧
  (-2 < a ∧ a < 2 → 1/(a + 2) < x ∧ x < 1/(2 - a)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1594_159443


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1594_159452

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1594_159452


namespace NUMINAMATH_CALUDE_area_between_curves_l1594_159419

-- Define the curves
def curve1 (x y : ℝ) : Prop := y^2 = 4*x
def curve2 (x y : ℝ) : Prop := x^2 = 4*y

-- Define the bounded area
def bounded_area (A : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ A ↔ (curve1 x y ∧ x ≥ 0 ∧ y ≥ 0) ∨ (curve2 x y ∧ x ≥ 0 ∧ y ≥ 0)

-- State the theorem
theorem area_between_curves :
  ∃ (A : Set (ℝ × ℝ)), bounded_area A ∧ MeasureTheory.volume A = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_area_between_curves_l1594_159419


namespace NUMINAMATH_CALUDE_certain_number_proof_l1594_159404

theorem certain_number_proof (x : ℚ) : 
  (5 / 6 : ℚ) * x = (5 / 16 : ℚ) * x + 300 → x = 576 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1594_159404


namespace NUMINAMATH_CALUDE_barry_head_standing_duration_l1594_159430

def head_standing_duration (total_time minutes_between_turns num_turns : ℕ) : ℚ :=
  (total_time - minutes_between_turns * (num_turns - 1)) / num_turns

theorem barry_head_standing_duration :
  ∃ (x : ℕ), x ≥ 11 ∧ x < 12 ∧ head_standing_duration 120 5 8 < x :=
sorry

end NUMINAMATH_CALUDE_barry_head_standing_duration_l1594_159430


namespace NUMINAMATH_CALUDE_line_slope_condition_l1594_159421

/-- Given a line passing through points (5, m) and (m, 8), prove that its slope is greater than 1
    if and only if m is in the open interval (5, 13/2). -/
theorem line_slope_condition (m : ℝ) :
  (8 - m) / (m - 5) > 1 ↔ 5 < m ∧ m < 13 / 2 := by
sorry

end NUMINAMATH_CALUDE_line_slope_condition_l1594_159421


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1594_159464

/-- Proves that given a round trip of 240 miles total, where the return trip speed is 38.71 miles per hour, 
and the total travel time is 5.5 hours, the speed of the first leg of the trip is 50 miles per hour. -/
theorem train_speed_calculation (total_distance : ℝ) (return_speed : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 240)
  (h2 : return_speed = 38.71)
  (h3 : total_time = 5.5) :
  ∃ (outbound_speed : ℝ), outbound_speed = 50 ∧ 
  (total_distance / 2) / outbound_speed + (total_distance / 2) / return_speed = total_time :=
by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l1594_159464


namespace NUMINAMATH_CALUDE_alex_sandwich_production_l1594_159450

/-- Given that Alex can prepare 18 sandwiches using 3 loaves of bread,
    this theorem proves that he can make 60 sandwiches with 10 loaves of bread. -/
theorem alex_sandwich_production (sandwiches_per_three_loaves : ℕ) 
    (h1 : sandwiches_per_three_loaves = 18) : 
    (sandwiches_per_three_loaves / 3) * 10 = 60 := by
  sorry

#check alex_sandwich_production

end NUMINAMATH_CALUDE_alex_sandwich_production_l1594_159450


namespace NUMINAMATH_CALUDE_jason_gave_four_cards_l1594_159429

/-- The number of Pokemon cards Jason gave to his friends -/
def cards_given (initial_cards current_cards : ℕ) : ℕ :=
  initial_cards - current_cards

theorem jason_gave_four_cards :
  let initial_cards : ℕ := 9
  let current_cards : ℕ := 5
  cards_given initial_cards current_cards = 4 := by
  sorry

end NUMINAMATH_CALUDE_jason_gave_four_cards_l1594_159429


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1594_159487

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^3 - (x + 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₂ + a₄ = -54 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1594_159487


namespace NUMINAMATH_CALUDE_pencils_per_box_l1594_159499

/-- Given Arnel's pencil distribution scenario, prove that each box contains 5 pencils. -/
theorem pencils_per_box (num_boxes : ℕ) (num_friends : ℕ) (pencils_kept : ℕ) (pencils_per_friend : ℕ) :
  num_boxes = 10 →
  num_friends = 5 →
  pencils_kept = 10 →
  pencils_per_friend = 8 →
  (∃ (pencils_per_box : ℕ), 
    pencils_per_box * num_boxes = pencils_kept + num_friends * pencils_per_friend ∧
    pencils_per_box = 5) :=
by sorry

end NUMINAMATH_CALUDE_pencils_per_box_l1594_159499


namespace NUMINAMATH_CALUDE_raven_current_age_l1594_159411

/-- Represents a person's age -/
structure Person where
  age : ℕ

/-- The current ages of Raven and Phoebe -/
def raven_phoebe_ages : Person × Person → Prop
  | (raven, phoebe) => 
    -- In 5 years, Raven will be 4 times as old as Phoebe
    raven.age + 5 = 4 * (phoebe.age + 5) ∧
    -- Phoebe is currently 10 years old
    phoebe.age = 10

/-- Theorem stating Raven's current age -/
theorem raven_current_age : 
  ∀ (raven phoebe : Person), raven_phoebe_ages (raven, phoebe) → raven.age = 55 := by
  sorry

end NUMINAMATH_CALUDE_raven_current_age_l1594_159411


namespace NUMINAMATH_CALUDE_student_sister_weight_ratio_l1594_159455

/-- Proves that the ratio of a student's weight after losing 5 kg to his sister's weight is 2:1 -/
theorem student_sister_weight_ratio 
  (student_weight : ℕ) 
  (total_weight : ℕ) 
  (weight_loss : ℕ) :
  student_weight = 75 →
  total_weight = 110 →
  weight_loss = 5 →
  (student_weight - weight_loss) / (total_weight - student_weight) = 2 := by
  sorry

end NUMINAMATH_CALUDE_student_sister_weight_ratio_l1594_159455


namespace NUMINAMATH_CALUDE_least_coins_seventeen_coins_coins_in_jar_l1594_159480

theorem least_coins (n : ℕ) : 
  (n % 7 = 3) ∧ (n % 4 = 1) ∧ (n % 6 = 5) → n ≥ 17 :=
by
  sorry

theorem seventeen_coins : 
  (17 % 7 = 3) ∧ (17 % 4 = 1) ∧ (17 % 6 = 5) :=
by
  sorry

theorem coins_in_jar : 
  ∃ (n : ℕ), (n % 7 = 3) ∧ (n % 4 = 1) ∧ (n % 6 = 5) ∧ 
  (∀ (m : ℕ), (m % 7 = 3) ∧ (m % 4 = 1) ∧ (m % 6 = 5) → m ≥ n) :=
by
  sorry

end NUMINAMATH_CALUDE_least_coins_seventeen_coins_coins_in_jar_l1594_159480


namespace NUMINAMATH_CALUDE_correct_addition_result_l1594_159490

theorem correct_addition_result 
  (correct_addend : ℕ)
  (mistaken_addend : ℕ)
  (other_addend : ℕ)
  (mistaken_result : ℕ)
  (h1 : correct_addend = 420)
  (h2 : mistaken_addend = 240)
  (h3 : mistaken_result = mistaken_addend + other_addend)
  : correct_addend + other_addend = 570 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_addition_result_l1594_159490


namespace NUMINAMATH_CALUDE_water_heater_problem_l1594_159427

/-- Represents the capacity and current water level of a water heater -/
structure WaterHeater where
  capacity : ℚ
  fillRatio : ℚ

/-- Calculates the total amount of water in all water heaters -/
def totalWater (wallace catherine albert belinda : WaterHeater) : ℚ :=
  wallace.capacity * wallace.fillRatio +
  catherine.capacity * catherine.fillRatio +
  albert.capacity * albert.fillRatio - 5 +
  belinda.capacity * belinda.fillRatio

theorem water_heater_problem 
  (wallace catherine albert belinda : WaterHeater)
  (h1 : wallace.capacity = 2 * catherine.capacity)
  (h2 : albert.capacity = 3/2 * wallace.capacity)
  (h3 : wallace.capacity = 40)
  (h4 : wallace.fillRatio = 3/4)
  (h5 : albert.fillRatio = 2/3)
  (h6 : belinda.capacity = 1/2 * catherine.capacity)
  (h7 : belinda.fillRatio = 5/8)
  (h8 : catherine.fillRatio = 7/8) :
  totalWater wallace catherine albert belinda = 89 := by
  sorry


end NUMINAMATH_CALUDE_water_heater_problem_l1594_159427


namespace NUMINAMATH_CALUDE_a_values_l1594_159413

def A : Set ℝ := {x | 2 * x^2 - 7 * x - 4 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem a_values (h : B a ⊆ A) : a = 0 ∨ a = -2 ∨ a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_a_values_l1594_159413


namespace NUMINAMATH_CALUDE_sedan_acceleration_l1594_159479

def v (t : ℝ) : ℝ := t^2 + 3

theorem sedan_acceleration : 
  let a (t : ℝ) := (deriv v) t
  a 3 = 6 := by sorry

end NUMINAMATH_CALUDE_sedan_acceleration_l1594_159479


namespace NUMINAMATH_CALUDE_negative_143_coterminal_with_37_l1594_159441

/-- An angle is coterminal with 37° if it can be represented as 37° + 180°k, where k is an integer -/
def is_coterminal_with_37 (angle : ℝ) : Prop :=
  ∃ k : ℤ, angle = 37 + 180 * k

/-- Theorem: -143° is coterminal with 37° -/
theorem negative_143_coterminal_with_37 : is_coterminal_with_37 (-143) := by
  sorry

end NUMINAMATH_CALUDE_negative_143_coterminal_with_37_l1594_159441


namespace NUMINAMATH_CALUDE_log_157489_between_consecutive_integers_l1594_159465

theorem log_157489_between_consecutive_integers :
  ∃ c d : ℤ, c + 1 = d ∧ (c : ℝ) < Real.log 157489 / Real.log 10 ∧ Real.log 157489 / Real.log 10 < (d : ℝ) ∧ c + d = 11 := by
  sorry

end NUMINAMATH_CALUDE_log_157489_between_consecutive_integers_l1594_159465


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1594_159433

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∀ x : ℝ, x^2 + c*x + d = 0 ↔ x = c ∨ x = 2*d) →
  c = 1/2 ∧ d = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1594_159433


namespace NUMINAMATH_CALUDE_infinitely_many_L_for_fibonacci_ratio_l1594_159491

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Lucas sequence -/
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- The sequence defined in the problem -/
def a (L : ℕ) : ℕ → ℚ
  | 0 => 0
  | (n + 1) => 1 / (L - a L n)

theorem infinitely_many_L_for_fibonacci_ratio :
  ∃ f : ℕ → ℕ, Monotone f ∧ ∀ k, ∃ i j,
    ∀ n, a (lucas (f k)) n = (fib i) / (fib j) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_L_for_fibonacci_ratio_l1594_159491


namespace NUMINAMATH_CALUDE_function_graph_relationship_l1594_159416

theorem function_graph_relationship (a : ℝ) (h1 : a > 0) :
  (∀ x : ℝ, x > 0 → Real.log x < a * x^2 - 1/2) → a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_graph_relationship_l1594_159416


namespace NUMINAMATH_CALUDE_teacher_assignment_theorem_l1594_159458

/-- The number of ways to assign 4 teachers to 3 schools, with each school having at least 1 teacher -/
def teacher_assignment_count : ℕ := 36

/-- The number of teachers -/
def num_teachers : ℕ := 4

/-- The number of schools -/
def num_schools : ℕ := 3

theorem teacher_assignment_theorem :
  (∀ assignment : Fin num_teachers → Fin num_schools,
    (∀ s : Fin num_schools, ∃ t : Fin num_teachers, assignment t = s) →
    (∃ s : Fin num_schools, ∃ t₁ t₂ : Fin num_teachers, t₁ ≠ t₂ ∧ assignment t₁ = s ∧ assignment t₂ = s)) →
  (Fintype.card {assignment : Fin num_teachers → Fin num_schools |
    ∀ s : Fin num_schools, ∃ t : Fin num_teachers, assignment t = s}) = teacher_assignment_count :=
by sorry

#check teacher_assignment_theorem

end NUMINAMATH_CALUDE_teacher_assignment_theorem_l1594_159458


namespace NUMINAMATH_CALUDE_complement_A_in_U_l1594_159428

def U : Set ℕ := {x : ℕ | x ≥ 2}
def A : Set ℕ := {x : ℕ | x^2 ≥ 5}

theorem complement_A_in_U : (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l1594_159428


namespace NUMINAMATH_CALUDE_dwarf_heights_l1594_159467

/-- The heights of Mr. Ticháček's dwarfs -/
theorem dwarf_heights :
  ∀ (F J M : ℕ),
  (J + F = M) →
  (M + F = J + 34) →
  (M + J = F + 72) →
  (F = 17 ∧ J = 36 ∧ M = 53) :=
by
  sorry

end NUMINAMATH_CALUDE_dwarf_heights_l1594_159467


namespace NUMINAMATH_CALUDE_fraction_sum_integer_l1594_159457

theorem fraction_sum_integer (n : ℕ) (h1 : n > 0) 
  (h2 : ∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n = k) : 
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_l1594_159457


namespace NUMINAMATH_CALUDE_adults_on_bus_l1594_159481

/-- Given a bus with 60 passengers where children make up 25% of the riders,
    prove that there are 45 adults on the bus. -/
theorem adults_on_bus (total_passengers : ℕ) (children_percentage : ℚ) : 
  total_passengers = 60 →
  children_percentage = 25 / 100 →
  (total_passengers : ℚ) * (1 - children_percentage) = 45 := by
sorry

end NUMINAMATH_CALUDE_adults_on_bus_l1594_159481


namespace NUMINAMATH_CALUDE_proposition_p_false_and_q_true_l1594_159405

theorem proposition_p_false_and_q_true :
  (∃ x : ℝ, 2^x ≤ x^2) ∧
  ((∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) ∧
   (∃ a b : ℝ, a * b > 1 ∧ (a ≤ 1 ∨ b ≤ 1))) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_false_and_q_true_l1594_159405


namespace NUMINAMATH_CALUDE_league_games_l1594_159474

theorem league_games (num_teams : ℕ) (total_games : ℕ) (games_per_matchup : ℕ) : 
  num_teams = 20 →
  total_games = 760 →
  games_per_matchup = 4 →
  games_per_matchup * (num_teams - 1) * num_teams / 2 = total_games :=
by
  sorry

end NUMINAMATH_CALUDE_league_games_l1594_159474


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l1594_159475

theorem cube_edge_ratio (a b : ℝ) (h : a^3 / b^3 = 27 / 1) : a / b = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l1594_159475


namespace NUMINAMATH_CALUDE_largest_fraction_sum_l1594_159449

theorem largest_fraction_sum (a b c d : ℤ) (ha : a = 3) (hb : b = 4) (hc : c = 6) (hd : d = 7) :
  (max ((a : ℚ) / b) ((a : ℚ) / c) + max ((b : ℚ) / a) ((b : ℚ) / c) + 
   max ((c : ℚ) / a) ((c : ℚ) / b) + max ((d : ℚ) / a) ((d : ℚ) / b)) ≤ 23 / 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_sum_l1594_159449


namespace NUMINAMATH_CALUDE_triangle_inequality_with_sum_zero_l1594_159484

theorem triangle_inequality_with_sum_zero (a b c p q r : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) → 
  (a + b > c) → (b + c > a) → (c + a > b) → 
  (p + q + r = 0) → 
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_sum_zero_l1594_159484


namespace NUMINAMATH_CALUDE_race_probability_l1594_159471

theorem race_probability (total_cars : ℕ) (prob_Y prob_Z prob_XYZ : ℚ) : 
  total_cars = 8 →
  prob_Y = 1/4 →
  prob_Z = 1/3 →
  prob_XYZ = 13/12 →
  ∃ prob_X : ℚ, prob_X + prob_Y + prob_Z = prob_XYZ ∧ prob_X = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_race_probability_l1594_159471


namespace NUMINAMATH_CALUDE_infinite_solutions_equation_l1594_159466

theorem infinite_solutions_equation (A B C : ℚ) : 
  (∀ x : ℚ, (x + B) * (A * x + 40) = 3 * (x + C) * (x + 10)) →
  (A = 3 ∧ B = 10/9 ∧ C = 40/9 ∧ 
   (- 40/9) + (-10) = -130/9) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_equation_l1594_159466


namespace NUMINAMATH_CALUDE_triangle_inequality_l1594_159434

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_inequality (ABC : Triangle) (P : ℝ × ℝ) :
  let S := area ABC
  distance P ABC.A + distance P ABC.B + distance P ABC.C ≥ 2 * (3 ^ (1/4)) * Real.sqrt S := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1594_159434


namespace NUMINAMATH_CALUDE_min_value_implies_a_eq_one_l1594_159469

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 / x - 2 + 2 * a * Real.log x

theorem min_value_implies_a_eq_one (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, f a x ≥ 0) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f a x = 0) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_min_value_implies_a_eq_one_l1594_159469


namespace NUMINAMATH_CALUDE_tan_fifteen_identity_l1594_159408

theorem tan_fifteen_identity : (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_identity_l1594_159408


namespace NUMINAMATH_CALUDE_expression_simplification_l1594_159401

theorem expression_simplification (a : ℚ) (h : a = 3) :
  (((a + 3) / (a - 1) - 1 / (a - 1)) / ((a^2 + 4*a + 4) / (a^2 - a))) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1594_159401


namespace NUMINAMATH_CALUDE_expression_evaluation_l1594_159415

theorem expression_evaluation : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1594_159415


namespace NUMINAMATH_CALUDE_blue_balls_count_l1594_159486

theorem blue_balls_count (total : ℕ) (yellow : ℕ) (blue : ℕ) : 
  total = 35 →
  yellow + blue = total →
  4 * blue = 3 * yellow →
  blue = 15 := by
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l1594_159486


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1594_159407

theorem complex_number_in_second_quadrant (z : ℂ) (a : ℝ) :
  z = a + Complex.I * Real.sqrt 3 →
  (Complex.re z < 0 ∧ Complex.im z > 0) →
  Complex.abs z = 2 →
  z = -1 + Complex.I * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1594_159407


namespace NUMINAMATH_CALUDE_least_n_with_j_geq_10_remainder_M_mod_100_l1594_159448

/-- Sum of digits in base 6 representation -/
def h (n : ℕ) : ℕ := sorry

/-- Sum of digits in base 10 representation -/
def j (n : ℕ) : ℕ := sorry

/-- The least value of n such that j(n) ≥ 10 -/
def M : ℕ := sorry

theorem least_n_with_j_geq_10 : M = 14 := by sorry

theorem remainder_M_mod_100 : M % 100 = 14 := by sorry

end NUMINAMATH_CALUDE_least_n_with_j_geq_10_remainder_M_mod_100_l1594_159448


namespace NUMINAMATH_CALUDE_cos_sin_inequalities_l1594_159423

theorem cos_sin_inequalities (x : ℝ) (h : x > 0) : 
  (Real.cos x > 1 - x^2 / 2) ∧ (Real.sin x > x - x^3 / 6) := by sorry

end NUMINAMATH_CALUDE_cos_sin_inequalities_l1594_159423


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1594_159406

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (h_r : r = 1/4) (h_S : S = 24) :
  let a := S * (1 - r)
  a = 18 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1594_159406


namespace NUMINAMATH_CALUDE_circle_radius_five_l1594_159435

/-- The value of c for which the circle x^2 + 8x + y^2 - 2y + c = 0 has radius 5 -/
theorem circle_radius_five (x y : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ, x^2 + 8*x + y^2 - 2*y + c = 0 ↔ (x + 4)^2 + (y - 1)^2 = 5^2) →
  (∃ c : ℝ, c = -8) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_five_l1594_159435


namespace NUMINAMATH_CALUDE_original_average_age_of_class_l1594_159410

theorem original_average_age_of_class (A : ℝ) : 
  (12 : ℝ) * A + (12 : ℝ) * 32 = (24 : ℝ) * (A - 4) → A = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_average_age_of_class_l1594_159410


namespace NUMINAMATH_CALUDE_simplify_square_roots_l1594_159472

theorem simplify_square_roots : 
  Real.sqrt 24 - Real.sqrt 12 + 6 * Real.sqrt (2/3) = 4 * Real.sqrt 6 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l1594_159472


namespace NUMINAMATH_CALUDE_cheese_cost_for_order_l1594_159403

/-- Represents the cost of cheese for a Mexican restaurant order --/
def cheese_cost (burrito_count : ℕ) (taco_count : ℕ) (enchilada_count : ℕ) : ℚ :=
  let cheddar_per_burrito : ℚ := 4
  let cheddar_per_taco : ℚ := 9
  let mozzarella_per_enchilada : ℚ := 5
  let cheddar_cost_per_ounce : ℚ := 4/5
  let mozzarella_cost_per_ounce : ℚ := 1
  (burrito_count * cheddar_per_burrito + taco_count * cheddar_per_taco) * cheddar_cost_per_ounce +
  (enchilada_count * mozzarella_per_enchilada) * mozzarella_cost_per_ounce

/-- Theorem stating the total cost of cheese for a specific order --/
theorem cheese_cost_for_order :
  cheese_cost 7 1 3 = 446/10 :=
sorry

end NUMINAMATH_CALUDE_cheese_cost_for_order_l1594_159403


namespace NUMINAMATH_CALUDE_signup_ways_4_3_l1594_159446

/-- The number of ways 4 students can sign up for one of 3 interest groups -/
def signup_ways (num_students : ℕ) (num_groups : ℕ) : ℕ :=
  num_groups ^ num_students

/-- Theorem stating that the number of ways 4 students can sign up for one of 3 interest groups is 81 -/
theorem signup_ways_4_3 : signup_ways 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_signup_ways_4_3_l1594_159446


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1594_159462

/-- Given vectors OA, OB, OC, where O is the origin, prove that the minimum value of 1/a + 2/b is 8 -/
theorem min_value_of_expression (a b : ℝ) (OA OB OC : ℝ × ℝ) : 
  a > 0 → b > 0 → 
  OA = (1, -2) → OB = (a, -1) → OC = (-b, 0) →
  (∃ (t : ℝ), (OB.1 - OA.1, OB.2 - OA.2) = t • (OC.1 - OA.1, OC.2 - OA.2)) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 1 / a' + 2 / b' ≥ 8) ∧ 
  (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 1 / a' + 2 / b' = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1594_159462


namespace NUMINAMATH_CALUDE_additional_week_rate_is_12_l1594_159493

/-- The daily rate for additional weeks in a student youth hostel -/
def additional_week_rate (first_week_rate : ℚ) (total_days : ℕ) (total_cost : ℚ) : ℚ :=
  (total_cost - first_week_rate * 7) / (total_days - 7)

/-- Theorem stating that the additional week rate is $12.00 per day -/
theorem additional_week_rate_is_12 :
  additional_week_rate 18 23 318 = 12 := by
  sorry

end NUMINAMATH_CALUDE_additional_week_rate_is_12_l1594_159493


namespace NUMINAMATH_CALUDE_total_students_l1594_159468

theorem total_students (boys_ratio : ℕ) (girls_ratio : ℕ) (num_girls : ℕ) : 
  boys_ratio = 8 → girls_ratio = 5 → num_girls = 160 → 
  (boys_ratio + girls_ratio) * (num_girls / girls_ratio) = 416 := by
sorry

end NUMINAMATH_CALUDE_total_students_l1594_159468


namespace NUMINAMATH_CALUDE_curve_C_cartesian_to_polar_l1594_159489

/-- The curve C in the Cartesian plane -/
def C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

/-- The polar equation of curve C -/
def polar_C (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The relationship between Cartesian and polar coordinates -/
def polar_to_cartesian (ρ θ x y : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

theorem curve_C_cartesian_to_polar :
  ∀ x y ρ θ : ℝ, 
    polar_to_cartesian ρ θ x y →
    (C x y ↔ polar_C ρ θ) :=
by sorry

end NUMINAMATH_CALUDE_curve_C_cartesian_to_polar_l1594_159489


namespace NUMINAMATH_CALUDE_interval_intersection_l1594_159454

theorem interval_intersection (x : ℝ) : 
  (3/4 < x ∧ x < 5/4) ↔ (2 < 3*x ∧ 3*x < 4) ∧ (3 < 4*x ∧ 4*x < 5) := by
  sorry

end NUMINAMATH_CALUDE_interval_intersection_l1594_159454


namespace NUMINAMATH_CALUDE_tangent_line_sin_at_pi_l1594_159453

theorem tangent_line_sin_at_pi :
  let f (x : ℝ) := Real.sin x
  let x₀ : ℝ := Real.pi
  let y₀ : ℝ := f x₀
  let m : ℝ := Real.cos x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x + y - Real.pi = 0) := by sorry

end NUMINAMATH_CALUDE_tangent_line_sin_at_pi_l1594_159453


namespace NUMINAMATH_CALUDE_cube_volume_from_side_area_l1594_159436

theorem cube_volume_from_side_area (side_area : ℝ) (h : side_area = 64) :
  let side_length := Real.sqrt side_area
  side_length ^ 3 = 512 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_side_area_l1594_159436


namespace NUMINAMATH_CALUDE_smallest_integer_l1594_159463

theorem smallest_integer (a b : ℕ+) (h1 : a = 60) 
  (h2 : Nat.lcm a b / Nat.gcd a b = 75) : 
  ∀ c : ℕ+, (c < b → ¬(Nat.lcm a c / Nat.gcd a c = 75)) → b = 500 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_l1594_159463


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l1594_159432

theorem smallest_factor_for_perfect_square (n : ℕ) (h : n = 2 * 3^2 * 5^2 * 7) :
  ∃ (a : ℕ), a > 0 ∧ 
  (∀ (b : ℕ), b > 0 → ∃ (k : ℕ), n * b = k^2 → a ≤ b) ∧
  (∃ (k : ℕ), n * a = k^2) ∧
  a = 14 := by
sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l1594_159432
