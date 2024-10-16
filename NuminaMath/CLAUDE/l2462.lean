import Mathlib

namespace NUMINAMATH_CALUDE_boat_speed_correct_l2462_246263

/-- The speed of the boat in still water -/
def boat_speed : ℝ := 15

/-- The speed of the stream -/
def stream_speed : ℝ := 3

/-- The time taken to travel downstream -/
def downstream_time : ℝ := 1

/-- The time taken to travel upstream -/
def upstream_time : ℝ := 1.5

/-- Theorem stating that the boat speed is correct given the conditions -/
theorem boat_speed_correct :
  ∃ (distance : ℝ),
    distance = (boat_speed + stream_speed) * downstream_time ∧
    distance = (boat_speed - stream_speed) * upstream_time :=
by
  sorry

end NUMINAMATH_CALUDE_boat_speed_correct_l2462_246263


namespace NUMINAMATH_CALUDE_interest_discount_sum_l2462_246247

/-- Given a sum, rate, and time, if the simple interest is 85 and the true discount is 75, then the sum is 637.5 -/
theorem interest_discount_sum (P r t : ℝ) : 
  (P * r * t / 100 = 85) → 
  (P * r * t / (100 + r * t) = 75) → 
  P = 637.5 := by
sorry

end NUMINAMATH_CALUDE_interest_discount_sum_l2462_246247


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l2462_246223

theorem largest_multiple_of_8_under_100 : 
  ∃ n : ℕ, n * 8 = 96 ∧ 
    ∀ m : ℕ, m * 8 < 100 → m * 8 ≤ 96 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l2462_246223


namespace NUMINAMATH_CALUDE_radical_simplification_l2462_246291

theorem radical_simplification (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (15 * q) * Real.sqrt (10 * q^3) * Real.sqrt (14 * q^5) = 10 * q^4 * Real.sqrt (21 * q) :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l2462_246291


namespace NUMINAMATH_CALUDE_box_surface_area_l2462_246281

/-- Calculates the surface area of the interior of a box formed by cutting out square corners from a rectangular sheet and folding the sides. -/
def interior_surface_area (sheet_length sheet_width corner_size : ℕ) : ℕ :=
  let remaining_area := sheet_length * sheet_width - 4 * (corner_size * corner_size)
  remaining_area

/-- The surface area of the interior of a box formed by cutting out 8-unit squares from the corners of a 40x50 unit sheet and folding the sides is 1744 square units. -/
theorem box_surface_area : interior_surface_area 40 50 8 = 1744 := by
  sorry

end NUMINAMATH_CALUDE_box_surface_area_l2462_246281


namespace NUMINAMATH_CALUDE_percentage_less_than_twice_yesterday_l2462_246243

def students_yesterday : ℕ := 70
def students_absent_today : ℕ := 30
def students_registered : ℕ := 156

def students_today : ℕ := students_registered - students_absent_today
def twice_students_yesterday : ℕ := 2 * students_yesterday
def difference : ℕ := twice_students_yesterday - students_today

theorem percentage_less_than_twice_yesterday (h : difference = 14) :
  (difference : ℚ) / (twice_students_yesterday : ℚ) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_less_than_twice_yesterday_l2462_246243


namespace NUMINAMATH_CALUDE_equal_distribution_of_boxes_l2462_246283

theorem equal_distribution_of_boxes (total_boxes : ℕ) (num_stops : ℕ) 
  (h1 : total_boxes = 27) (h2 : num_stops = 3) :
  total_boxes / num_stops = 9 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_boxes_l2462_246283


namespace NUMINAMATH_CALUDE_plane_equation_l2462_246296

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if two planes are parallel -/
def parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ p1.a = k * p2.a ∧ p1.b = k * p2.b ∧ p1.c = k * p2.c

/-- Check if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Check if coefficients are integers and their GCD is 1 -/
def validCoefficients (plane : Plane) : Prop :=
  Int.gcd (Int.natAbs (Int.floor plane.a)) (Int.gcd (Int.natAbs (Int.floor plane.b)) (Int.gcd (Int.natAbs (Int.floor plane.c)) (Int.natAbs (Int.floor plane.d)))) = 1

theorem plane_equation : ∃ (result : Plane),
  result.a = 2 ∧ result.b = -1 ∧ result.c = 3 ∧ result.d = -14 ∧
  pointOnPlane ⟨2, -1, 3⟩ result ∧
  parallel result ⟨4, -2, 6, -5⟩ ∧
  result.a > 0 ∧
  validCoefficients result :=
sorry

end NUMINAMATH_CALUDE_plane_equation_l2462_246296


namespace NUMINAMATH_CALUDE_mapping_properties_l2462_246206

-- Define the sets A and B
variable {A B : Type}

-- Define the mapping f from A to B
variable (f : A → B)

-- Theorem stating the properties of the mapping
theorem mapping_properties :
  (∀ a : A, ∃! b : B, f a = b) ∧
  (∃ a₁ a₂ : A, a₁ ≠ a₂ ∧ f a₁ = f a₂) :=
by sorry

end NUMINAMATH_CALUDE_mapping_properties_l2462_246206


namespace NUMINAMATH_CALUDE_pizza_theorem_l2462_246287

/-- Represents a pizza with given topping distributions -/
structure Pizza where
  total_slices : ℕ
  pepperoni_slices : ℕ
  mushroom_slices : ℕ
  olive_slices : ℕ
  all_toppings_slices : ℕ

/-- Conditions for a valid pizza configuration -/
def is_valid_pizza (p : Pizza) : Prop :=
  p.total_slices = 20 ∧
  p.pepperoni_slices = 12 ∧
  p.mushroom_slices = 14 ∧
  p.olive_slices = 12 ∧
  p.all_toppings_slices ≤ p.total_slices ∧
  p.all_toppings_slices ≤ p.pepperoni_slices ∧
  p.all_toppings_slices ≤ p.mushroom_slices ∧
  p.all_toppings_slices ≤ p.olive_slices

theorem pizza_theorem (p : Pizza) (h : is_valid_pizza p) : p.all_toppings_slices = 6 := by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l2462_246287


namespace NUMINAMATH_CALUDE_probability_triangle_or_hexagon_l2462_246211

theorem probability_triangle_or_hexagon :
  let total_figures : ℕ := 12
  let triangles : ℕ := 3
  let squares : ℕ := 4
  let circles : ℕ := 3
  let hexagons : ℕ := 2
  let favorable_outcomes : ℕ := triangles + hexagons
  (favorable_outcomes : ℚ) / total_figures = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_triangle_or_hexagon_l2462_246211


namespace NUMINAMATH_CALUDE_second_divisor_existence_l2462_246277

theorem second_divisor_existence : ∃ (D : ℕ+), 
  (∃ (N : ℤ), N % 35 = 25 ∧ N % D.val = 4) ∧ D.val = 21 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_existence_l2462_246277


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2462_246250

def M : Set ℝ := {x | x^2 ≤ 4}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2462_246250


namespace NUMINAMATH_CALUDE_capri_sun_cost_per_pouch_l2462_246255

/-- Calculates the cost per pouch in cents -/
def cost_per_pouch (num_boxes : ℕ) (pouches_per_box : ℕ) (total_cost_dollars : ℕ) : ℕ :=
  (total_cost_dollars * 100) / (num_boxes * pouches_per_box)

/-- Theorem: The cost per pouch is 20 cents -/
theorem capri_sun_cost_per_pouch :
  cost_per_pouch 10 6 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_capri_sun_cost_per_pouch_l2462_246255


namespace NUMINAMATH_CALUDE_medicine_container_problem_l2462_246280

theorem medicine_container_problem (initial_volume : ℝ) (remaining_volume : ℝ) : 
  initial_volume = 63 ∧ remaining_volume = 28 →
  ∃ (x : ℝ), x = 18 ∧ 
    initial_volume * (1 - x / initial_volume) * (1 - x / initial_volume) = remaining_volume :=
by sorry

end NUMINAMATH_CALUDE_medicine_container_problem_l2462_246280


namespace NUMINAMATH_CALUDE_greg_savings_needed_l2462_246290

/-- The amount of additional money Greg needs to buy a scooter, helmet, and lock -/
def additional_money_needed (scooter_price helmet_price lock_price discount_rate tax_rate gift_card savings : ℚ) : ℚ :=
  let discounted_scooter := scooter_price * (1 - discount_rate)
  let subtotal := discounted_scooter + helmet_price + lock_price
  let total_with_tax := subtotal * (1 + tax_rate)
  let final_price := total_with_tax - gift_card
  final_price - savings

/-- Theorem stating the additional amount Greg needs to save -/
theorem greg_savings_needed :
  additional_money_needed 90 30 15 0.1 0.1 20 57 = 61.6 := by
  sorry

end NUMINAMATH_CALUDE_greg_savings_needed_l2462_246290


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l2462_246295

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The y-axis in 2D space -/
def yAxis : Set (ℝ × ℝ) := {p | p.1 = 0}

/-- Symmetry of a line with respect to the y-axis -/
def symmetricLine (l : Line) : Line :=
  { slope := -l.slope, intercept := l.intercept }

/-- The original line y = 2x + 1 -/
def originalLine : Line :=
  { slope := 2, intercept := 1 }

theorem symmetric_line_equation :
  symmetricLine originalLine = { slope := -2, intercept := 1 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l2462_246295


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2462_246294

theorem inequality_solution_set (m : ℝ) :
  {x : ℝ | x^2 - (2*m + 1)*x + m^2 + m < 0} = {x : ℝ | m < x ∧ x < m + 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2462_246294


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_l2462_246289

theorem sqrt_expression_simplification :
  (Real.sqrt 2 + Real.sqrt 3)^2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) = 6 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_l2462_246289


namespace NUMINAMATH_CALUDE_sin_70_degrees_l2462_246244

theorem sin_70_degrees (k : ℝ) (h : Real.sin (10 * π / 180) = k) :
  Real.sin (70 * π / 180) = 1 - 2 * k^2 := by sorry

end NUMINAMATH_CALUDE_sin_70_degrees_l2462_246244


namespace NUMINAMATH_CALUDE_prime_dates_count_l2462_246240

/-- A prime date is a date where both the month and day are prime numbers -/
def PrimeDate (month : ℕ) (day : ℕ) : Prop :=
  Nat.Prime month ∧ Nat.Prime day

/-- The list of prime months in our scenario -/
def PrimeMonths : List ℕ := [2, 3, 5, 7, 11, 13]

/-- The number of days in each prime month for a non-leap year -/
def DaysInPrimeMonth (month : ℕ) : ℕ :=
  if month = 2 then 28
  else if month = 11 then 30
  else 31

/-- The list of prime days -/
def PrimeDays : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

/-- The number of prime dates in a given month -/
def PrimeDatesInMonth (month : ℕ) : ℕ :=
  (PrimeDays.filter (· ≤ DaysInPrimeMonth month)).length

theorem prime_dates_count : 
  (PrimeMonths.map PrimeDatesInMonth).sum = 62 := by
  sorry

end NUMINAMATH_CALUDE_prime_dates_count_l2462_246240


namespace NUMINAMATH_CALUDE_x_plus_y_values_l2462_246248

theorem x_plus_y_values (x y : ℝ) (h1 : |x| = 3) (h2 : y^2 = 4) (h3 : x < y) :
  x + y = -5 ∨ x + y = -1 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l2462_246248


namespace NUMINAMATH_CALUDE_coordinates_of_D_l2462_246292

-- Define the points
def C : ℝ × ℝ := (5, -1)
def M : ℝ × ℝ := (3, 7)

-- Define D as a point that satisfies the midpoint condition
def D : ℝ × ℝ := (2 * M.1 - C.1, 2 * M.2 - C.2)

-- Theorem statement
theorem coordinates_of_D :
  D.1 * D.2 = 15 ∧ D.1 + D.2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_D_l2462_246292


namespace NUMINAMATH_CALUDE_expression_evaluation_l2462_246203

theorem expression_evaluation (m : ℝ) (h : m = -Real.sqrt 5) :
  (2 * m - 1)^2 - (m - 5) * (m + 1) = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2462_246203


namespace NUMINAMATH_CALUDE_problem_solution_l2462_246246

theorem problem_solution (x y : ℝ) (hx : x = 2 + Real.sqrt 3) (hy : y = 2 - Real.sqrt 3) :
  (x^2 + y^2 = 14) ∧ (x / y - y / x = 8 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2462_246246


namespace NUMINAMATH_CALUDE_lee_cookies_l2462_246258

/-- Given that Lee can make 24 cookies with 3 cups of flour,
    this function calculates how many cookies he can make with any amount of flour. -/
def cookies_from_flour (flour : ℚ) : ℚ :=
  (24 * flour) / 3

/-- Theorem stating that Lee can make 40 cookies with 5 cups of flour. -/
theorem lee_cookies : cookies_from_flour 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_lee_cookies_l2462_246258


namespace NUMINAMATH_CALUDE_monomial_count_l2462_246298

/-- An algebraic expression is a monomial if it is a single number, a single variable, or a product of numbers and variables without variables in the denominator. -/
def is_monomial (expr : String) : Bool :=
  sorry

/-- The set of given algebraic expressions -/
def expressions : List String := ["2x^2", "-3", "x-2y", "t", "6m^2/π", "1/a", "m^3+2m^2-m"]

/-- Count the number of monomials in a list of expressions -/
def count_monomials (exprs : List String) : Nat :=
  sorry

/-- The main theorem: The number of monomials in the given set of expressions is 4 -/
theorem monomial_count : count_monomials expressions = 4 := by
  sorry

end NUMINAMATH_CALUDE_monomial_count_l2462_246298


namespace NUMINAMATH_CALUDE_no_fixed_point_function_l2462_246282

-- Define the types for our polynomials
variable (p q h : ℝ → ℝ)

-- Define m and n as natural numbers
variable (m n : ℕ)

-- Define the descending property for p
def IsDescending (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem no_fixed_point_function
  (hp : IsDescending p)
  (hpqh : ∀ x, p (q (n * x + m) + h x) = n * (q (p x) + h x) + m) :
  ¬ ∃ f : ℝ → ℝ, ∀ x, f (q (p x) + h x) = f x ^ 2 + 1 :=
sorry

end NUMINAMATH_CALUDE_no_fixed_point_function_l2462_246282


namespace NUMINAMATH_CALUDE_haris_joining_time_l2462_246272

theorem haris_joining_time (praveen_investment hari_investment : ℝ) 
  (profit_ratio_praveen profit_ratio_hari : ℕ) (x : ℝ) :
  praveen_investment = 3780 →
  hari_investment = 9720 →
  profit_ratio_praveen = 2 →
  profit_ratio_hari = 3 →
  (praveen_investment * 12) / (hari_investment * (12 - x)) = 
    profit_ratio_praveen / profit_ratio_hari →
  x = 5 := by sorry

end NUMINAMATH_CALUDE_haris_joining_time_l2462_246272


namespace NUMINAMATH_CALUDE_initial_pizza_slices_l2462_246230

-- Define the number of slices eaten at each meal and the number of slices left
def breakfast_slices : ℕ := 4
def lunch_slices : ℕ := 2
def snack_slices : ℕ := 2
def dinner_slices : ℕ := 5
def slices_left : ℕ := 2

-- Define the total number of slices eaten
def total_eaten : ℕ := breakfast_slices + lunch_slices + snack_slices + dinner_slices

-- Theorem: The initial number of pizza slices is 15
theorem initial_pizza_slices : 
  total_eaten + slices_left = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_pizza_slices_l2462_246230


namespace NUMINAMATH_CALUDE_inequality_holds_iff_k_in_range_l2462_246217

theorem inequality_holds_iff_k_in_range (k : ℝ) :
  (∀ x : ℝ, |((x^2 - k*x + 1) / (x^2 + x + 1))| < 3) ↔ -5 ≤ k ∧ k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_k_in_range_l2462_246217


namespace NUMINAMATH_CALUDE_factorial_3_equals_6_l2462_246257

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_3_equals_6 : factorial 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_factorial_3_equals_6_l2462_246257


namespace NUMINAMATH_CALUDE_union_of_sets_l2462_246242

theorem union_of_sets : 
  let A : Set Nat := {1, 2, 4}
  let B : Set Nat := {2, 6}
  A ∪ B = {1, 2, 4, 6} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l2462_246242


namespace NUMINAMATH_CALUDE_triangle_area_after_median_division_l2462_246237

-- Define a triangle type
structure Triangle where
  area : ℝ

-- Define a function that represents dividing a triangle by a median
def divideByMedian (t : Triangle) : (Triangle × Triangle) :=
  sorry

-- Theorem statement
theorem triangle_area_after_median_division (t : Triangle) :
  let (t1, t2) := divideByMedian t
  t1.area = 7 → t.area = 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_after_median_division_l2462_246237


namespace NUMINAMATH_CALUDE_final_cow_count_l2462_246213

def cow_count (initial : ℕ) (died : ℕ) (sold : ℕ) (increase : ℕ) (bought : ℕ) (gift : ℕ) : ℕ :=
  initial - died - sold + increase + bought + gift

theorem final_cow_count :
  cow_count 39 25 6 24 43 8 = 83 := by
  sorry

end NUMINAMATH_CALUDE_final_cow_count_l2462_246213


namespace NUMINAMATH_CALUDE_problem_statement_l2462_246256

theorem problem_statement (p q r : ℝ) 
  (h1 : p < q)
  (h2 : ∀ x, ((x - p) * (x - q)) / (x - r) ≥ 0 ↔ (x > 5 ∨ (7 ≤ x ∧ x ≤ 15))) :
  p + 2*q + 3*r = 52 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2462_246256


namespace NUMINAMATH_CALUDE_alex_gumballs_problem_l2462_246225

theorem alex_gumballs_problem : ∃ n : ℕ, 
  n ≥ 50 ∧ 
  n % 7 = 5 ∧ 
  ∀ m : ℕ, (m ≥ 50 ∧ m % 7 = 5) → m ≥ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_alex_gumballs_problem_l2462_246225


namespace NUMINAMATH_CALUDE_missed_both_equiv_l2462_246293

-- Define propositions
variable (p q : Prop)

-- Define the meaning of "missed the target on both shots"
def missed_both (p q : Prop) : Prop := (¬p) ∧ (¬q)

-- Theorem: "missed the target on both shots" is equivalent to ¬(p ∨ q)
theorem missed_both_equiv (p q : Prop) : missed_both p q ↔ ¬(p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_missed_both_equiv_l2462_246293


namespace NUMINAMATH_CALUDE_system_solution_l2462_246205

theorem system_solution (x y k : ℝ) 
  (eq1 : 2 * x - y = 5 * k + 6)
  (eq2 : 4 * x + 7 * y = k)
  (eq3 : x + y = 2024) :
  k = 2023 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2462_246205


namespace NUMINAMATH_CALUDE_problem_statement_l2462_246233

theorem problem_statement (A B C D : ℤ) 
  (h1 : A - B = 30) 
  (h2 : C + D = 20) : 
  (B + C) - (A - D) = -10 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2462_246233


namespace NUMINAMATH_CALUDE_cd_cost_with_tax_l2462_246251

/-- The cost of a CD including sales tax -/
def total_cost (price : ℝ) (tax_rate : ℝ) : ℝ :=
  price * (1 + tax_rate)

/-- Theorem stating that the total cost of a CD priced at $14.99 with 15% sales tax is $17.24 -/
theorem cd_cost_with_tax : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ 
  |total_cost 14.99 0.15 - 17.24| < ε :=
sorry

end NUMINAMATH_CALUDE_cd_cost_with_tax_l2462_246251


namespace NUMINAMATH_CALUDE_beef_pounds_calculation_l2462_246269

theorem beef_pounds_calculation (total_cost : ℝ) (chicken_cost : ℝ) (oil_cost : ℝ) (beef_cost_per_pound : ℝ) :
  total_cost = 16 ∧ 
  chicken_cost = 3 ∧ 
  oil_cost = 1 ∧ 
  beef_cost_per_pound = 4 →
  (total_cost - chicken_cost - oil_cost) / beef_cost_per_pound = 3 := by
  sorry

end NUMINAMATH_CALUDE_beef_pounds_calculation_l2462_246269


namespace NUMINAMATH_CALUDE_function_properties_l2462_246274

noncomputable def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b*x + 1

theorem function_properties (a b : ℝ) :
  (f a b 3 = -26) ∧ 
  (3*(3^2) - 2*a*3 + b = 0) →
  (a = 3 ∧ b = -9) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 4, f 3 (-9) x ≤ 6) ∧
  (∃ x ∈ Set.Icc (-4 : ℝ) 4, f 3 (-9) x = 6) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2462_246274


namespace NUMINAMATH_CALUDE_range_of_a_l2462_246224

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2*a - 1)^x < (2*a - 1)^y

def q (a : ℝ) : Prop := ∀ x : ℝ, 2*a*x^2 - 2*a*x + 1 > 0

-- Define the range of a
def range_a (a : ℝ) : Prop := (0 ≤ a ∧ a ≤ 1) ∨ (a ≥ 2)

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_a a :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2462_246224


namespace NUMINAMATH_CALUDE_yoga_to_exercise_ratio_l2462_246222

/-- Proves that the ratio of yoga time to total exercise time is 1:1 -/
theorem yoga_to_exercise_ratio : 
  ∀ (gym_time bicycle_time yoga_time : ℝ),
  gym_time / bicycle_time = 2 / 3 →
  bicycle_time = 12 →
  yoga_time = 20 →
  yoga_time / (gym_time + bicycle_time) = 1 := by
  sorry

end NUMINAMATH_CALUDE_yoga_to_exercise_ratio_l2462_246222


namespace NUMINAMATH_CALUDE_test_passing_requirement_l2462_246286

def total_questions : ℕ := 80
def arithmetic_questions : ℕ := 15
def algebra_questions : ℕ := 25
def geometry_questions : ℕ := 40

def arithmetic_correct_rate : ℚ := 60 / 100
def algebra_correct_rate : ℚ := 50 / 100
def geometry_correct_rate : ℚ := 70 / 100

def passing_rate : ℚ := 65 / 100

def additional_correct_answers_needed : ℕ := 3

theorem test_passing_requirement : 
  let current_correct := 
    (arithmetic_questions * arithmetic_correct_rate).floor +
    (algebra_questions * algebra_correct_rate).floor +
    (geometry_questions * geometry_correct_rate).floor
  (current_correct + additional_correct_answers_needed : ℚ) / total_questions ≥ passing_rate :=
by sorry

end NUMINAMATH_CALUDE_test_passing_requirement_l2462_246286


namespace NUMINAMATH_CALUDE_parking_garage_open_spots_l2462_246249

/-- Represents the number of open parking spots on each level of a parking garage -/
structure ParkingGarage where
  first_level : ℕ
  second_level : ℕ
  third_level : ℕ
  fourth_level : ℕ

/-- Theorem stating the number of open parking spots on the first level of the parking garage -/
theorem parking_garage_open_spots (g : ParkingGarage) : g.first_level = 58 :=
  by
  have h1 : g.second_level = g.first_level + 2 := sorry
  have h2 : g.third_level = g.second_level + 5 := sorry
  have h3 : g.fourth_level = 31 := sorry
  have h4 : g.first_level + g.second_level + g.third_level + g.fourth_level = 400 - 186 := sorry
  sorry

#check parking_garage_open_spots

end NUMINAMATH_CALUDE_parking_garage_open_spots_l2462_246249


namespace NUMINAMATH_CALUDE_flensburgian_iff_even_l2462_246236

/-- A system of equations is Flensburgian if there exists a variable that is always greater than the others for all pairwise different solutions. -/
def isFlensburgian (f : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ i : Fin 3, ∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x → f x y z →
    match i with
    | 0 => x > y ∧ x > z
    | 1 => y > x ∧ y > z
    | 2 => z > x ∧ z > y

/-- The system of equations for the Flensburgian problem. -/
def flensburgSystem (n : ℕ) (a b c : ℝ) : Prop :=
  a^n + b = a ∧ c^(n+1) + b^2 = a*b

/-- The main theorem stating that the system is Flensburgian if and only if n is even. -/
theorem flensburgian_iff_even (n : ℕ) (h : n ≥ 2) :
  isFlensburgian (flensburgSystem n) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_flensburgian_iff_even_l2462_246236


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l2462_246201

theorem smallest_constant_inequality (C : ℝ) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ↔ C ≤ Real.sqrt (4/3) :=
sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l2462_246201


namespace NUMINAMATH_CALUDE_toy_purchase_cost_l2462_246207

theorem toy_purchase_cost (num_toys : ℕ) (cost_per_toy : ℝ) (discount_percent : ℝ) :
  num_toys = 5 →
  cost_per_toy = 3 →
  discount_percent = 20 →
  (num_toys : ℝ) * cost_per_toy * (1 - discount_percent / 100) = 12 := by
  sorry

end NUMINAMATH_CALUDE_toy_purchase_cost_l2462_246207


namespace NUMINAMATH_CALUDE_impossible_to_use_all_components_l2462_246227

theorem impossible_to_use_all_components (p q r : ℤ) : 
  ¬ ∃ (x y z : ℤ), 
    (2 * x + 2 * z = 2 * p + 2 * r + 2) ∧ 
    (2 * x + y = 2 * p + q + 1) ∧ 
    (y + z = q + r) :=
by sorry

end NUMINAMATH_CALUDE_impossible_to_use_all_components_l2462_246227


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2462_246271

theorem inequality_solution_set :
  ∀ x : ℝ, abs (2*x - 1) - abs (x - 2) < 0 ↔ -1 < x ∧ x < 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2462_246271


namespace NUMINAMATH_CALUDE_row_length_theorem_l2462_246209

/-- The length of a row of boys standing with 1 meter between adjacent boys -/
def row_length (n : ℕ) : ℕ := n - 1

/-- Theorem: For n boys standing in a row with 1 meter between adjacent boys,
    the length of the row in meters is equal to n - 1 -/
theorem row_length_theorem (n : ℕ) (h : n > 0) : row_length n = n - 1 := by
  sorry

end NUMINAMATH_CALUDE_row_length_theorem_l2462_246209


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l2462_246254

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, (m - 1) * x^2 + 5 * x + m^2 - 1 = 0) → 
  (m^2 - 1 = 0) → 
  m = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l2462_246254


namespace NUMINAMATH_CALUDE_bean_game_uniqueness_l2462_246278

/-- Represents the state of beans on an infinite row of squares -/
def BeanState := ℤ → ℕ

/-- Represents a single move in the bean game -/
def Move := ℤ

/-- Applies a move to a given state -/
def applyMove (state : BeanState) (move : Move) : BeanState :=
  sorry

/-- Checks if a state is terminal (no square has more than one bean) -/
def isTerminal (state : BeanState) : Prop :=
  ∀ i : ℤ, state i ≤ 1

/-- Represents a sequence of moves -/
def MoveSequence := List Move

/-- Applies a sequence of moves to an initial state -/
def applyMoveSequence (initial : BeanState) (moves : MoveSequence) : BeanState :=
  sorry

/-- The final state after applying a sequence of moves -/
def finalState (initial : BeanState) (moves : MoveSequence) : BeanState :=
  applyMoveSequence initial moves

/-- The number of steps (moves) in a sequence -/
def numSteps (moves : MoveSequence) : ℕ :=
  moves.length

/-- Theorem: All valid move sequences result in the same final state and number of steps -/
theorem bean_game_uniqueness (initial : BeanState) 
    (moves1 moves2 : MoveSequence) 
    (h1 : isTerminal (finalState initial moves1))
    (h2 : isTerminal (finalState initial moves2)) :
    finalState initial moves1 = finalState initial moves2 ∧ 
    numSteps moves1 = numSteps moves2 :=
  sorry

end NUMINAMATH_CALUDE_bean_game_uniqueness_l2462_246278


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2462_246267

/-- The remainder when x^4 - 8x^3 + 5x^2 + 22x - 7 is divided by x-4 is -95 -/
theorem polynomial_remainder : 
  (fun x : ℝ => x^4 - 8*x^3 + 5*x^2 + 22*x - 7) 4 = -95 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2462_246267


namespace NUMINAMATH_CALUDE_jack_additional_apples_l2462_246229

/-- Represents the capacity of apple baskets and current apple counts -/
structure AppleBaskets where
  jack_capacity : ℕ
  jill_capacity : ℕ
  jack_current : ℕ

/-- The conditions of the apple picking problem -/
def apple_picking_conditions (ab : AppleBaskets) : Prop :=
  ab.jill_capacity = 2 * ab.jack_capacity ∧
  ab.jack_capacity = 12 ∧
  3 * ab.jack_current = ab.jill_capacity

/-- The theorem stating how many more apples Jack's basket can hold -/
theorem jack_additional_apples (ab : AppleBaskets) 
  (h : apple_picking_conditions ab) : 
  ab.jack_capacity - ab.jack_current = 4 := by
  sorry


end NUMINAMATH_CALUDE_jack_additional_apples_l2462_246229


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l2462_246234

/-- Given two positive integers with specific LCM and HCF, prove that if one number is 385, the other is 180 -/
theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  A = 385 →
  B = 180 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l2462_246234


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2462_246276

theorem fraction_subtraction (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  1 / x - 1 / y = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2462_246276


namespace NUMINAMATH_CALUDE_inequality_proof_l2462_246275

theorem inequality_proof (a b c d : ℝ) (h1 : c < d) (h2 : a > b) (h3 : b > 0) :
  a - c > b - d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2462_246275


namespace NUMINAMATH_CALUDE_sequence_property_l2462_246216

/-- Given a sequence a and its partial sum S, prove that a_n = 2^n + n for all n ∈ ℕ⁺ -/
theorem sequence_property (a : ℕ+ → ℕ) (S : ℕ+ → ℕ) 
  (h : ∀ n : ℕ+, 2 * S n = 4 * a n + (n - 4) * (n + 1)) :
  ∀ n : ℕ+, a n = 2^(n : ℕ) + n := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l2462_246216


namespace NUMINAMATH_CALUDE_doughnuts_given_away_l2462_246288

theorem doughnuts_given_away (total_doughnuts : ℕ) (small_boxes_sold : ℕ) (large_boxes_sold : ℕ)
  (h1 : total_doughnuts = 300)
  (h2 : small_boxes_sold = 20)
  (h3 : large_boxes_sold = 10) :
  total_doughnuts - (small_boxes_sold * 6 + large_boxes_sold * 12) = 60 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_given_away_l2462_246288


namespace NUMINAMATH_CALUDE_complement_of_A_l2462_246219

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | |x - 1| > 2}

theorem complement_of_A : 
  Set.compl A = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2462_246219


namespace NUMINAMATH_CALUDE_thabos_book_collection_l2462_246218

/-- Theorem: Thabo's Book Collection
Given:
- Thabo owns exactly 250 books in total
- Books are of 5 types: paperback fiction, hardcover fiction, paperback nonfiction, hardcover nonfiction, and audiobooks
- Thabo owns 15 more paperback nonfiction books than hardcover nonfiction books
- Thabo owns 10 more hardcover fiction books than paperback fiction books
- Thabo owns 3 times as many paperback fiction books as audiobooks
- The combined total of audiobooks and hardcover fiction books equals 70

Prove: Thabo owns 30 hardcover nonfiction books
-/
theorem thabos_book_collection 
  (total_books : ℕ) 
  (paperback_fiction : ℕ) 
  (hardcover_fiction : ℕ) 
  (paperback_nonfiction : ℕ) 
  (hardcover_nonfiction : ℕ) 
  (audiobooks : ℕ) 
  (h1 : total_books = 250)
  (h2 : total_books = paperback_fiction + hardcover_fiction + paperback_nonfiction + hardcover_nonfiction + audiobooks)
  (h3 : paperback_nonfiction = hardcover_nonfiction + 15)
  (h4 : hardcover_fiction = paperback_fiction + 10)
  (h5 : paperback_fiction = 3 * audiobooks)
  (h6 : audiobooks + hardcover_fiction = 70) :
  hardcover_nonfiction = 30 := by
  sorry

end NUMINAMATH_CALUDE_thabos_book_collection_l2462_246218


namespace NUMINAMATH_CALUDE_game_probability_l2462_246228

/-- Represents the probability of winning for each player -/
structure PlayerProbabilities where
  alex : ℚ
  mel : ℚ
  chelsea : ℚ

/-- Calculates the probability of a specific outcome in the game -/
def outcome_probability (probs : PlayerProbabilities) (alex_wins mel_wins chelsea_wins : ℕ) : ℚ :=
  (probs.alex ^ alex_wins) * (probs.mel ^ mel_wins) * (probs.chelsea ^ chelsea_wins)

/-- Calculates the number of ways to arrange wins in a given number of rounds -/
def arrangements (total_rounds alex_wins mel_wins chelsea_wins : ℕ) : ℕ :=
  Nat.choose total_rounds alex_wins * Nat.choose (total_rounds - alex_wins) mel_wins

/-- The main theorem stating the probability of the specific outcome -/
theorem game_probability : ∃ (probs : PlayerProbabilities),
  probs.alex = 1/4 ∧
  probs.mel = 2 * probs.chelsea ∧
  probs.alex + probs.mel + probs.chelsea = 1 ∧
  (outcome_probability probs 2 3 3 * arrangements 8 2 3 3 : ℚ) = 35/512 := by
  sorry

end NUMINAMATH_CALUDE_game_probability_l2462_246228


namespace NUMINAMATH_CALUDE_busy_squirrel_nuts_calculation_l2462_246264

/-- The number of nuts stockpiled per day by each busy squirrel -/
def busy_squirrel_nuts_per_day : ℕ := 30

/-- The number of busy squirrels -/
def num_busy_squirrels : ℕ := 2

/-- The number of nuts stockpiled per day by the sleepy squirrel -/
def sleepy_squirrel_nuts_per_day : ℕ := 20

/-- The number of days the squirrels have been stockpiling -/
def num_days : ℕ := 40

/-- The total number of nuts in Mason's car -/
def total_nuts : ℕ := 3200

theorem busy_squirrel_nuts_calculation :
  num_busy_squirrels * busy_squirrel_nuts_per_day * num_days + 
  sleepy_squirrel_nuts_per_day * num_days = total_nuts :=
by sorry

end NUMINAMATH_CALUDE_busy_squirrel_nuts_calculation_l2462_246264


namespace NUMINAMATH_CALUDE_function_extrema_and_inequality_l2462_246252

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - 0.5 * x^2 + x

def g (x : ℝ) : ℝ := 0.5 * x^2 - 2 * x + 1

theorem function_extrema_and_inequality (e : ℝ) (h_e : e = Real.exp 1) :
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 1 (e^2), f 2 x ≤ max) ∧ 
    (∃ x₀ ∈ Set.Icc 1 (e^2), f 2 x₀ = max) ∧
    (∀ x ∈ Set.Icc 1 (e^2), min ≤ f 2 x) ∧ 
    (∃ x₁ ∈ Set.Icc 1 (e^2), f 2 x₁ = min) ∧
    max = 2 * Real.log 2 ∧
    min = 4 + e^2 - 0.5 * e^4) ∧
  (∀ a : ℝ, (∀ x > 0, f a x + g x ≤ 0) ↔ a = 1) :=
sorry

end NUMINAMATH_CALUDE_function_extrema_and_inequality_l2462_246252


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2462_246270

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2462_246270


namespace NUMINAMATH_CALUDE_sandy_walk_l2462_246210

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a direction -/
inductive Direction
  | North
  | South
  | East
  | West

/-- Moves a point in a given direction by a specified distance -/
def move (p : Point) (dir : Direction) (distance : ℝ) : Point :=
  match dir with
  | Direction.North => { x := p.x, y := p.y + distance }
  | Direction.South => { x := p.x, y := p.y - distance }
  | Direction.East => { x := p.x + distance, y := p.y }
  | Direction.West => { x := p.x - distance, y := p.y }

/-- Sandy's walk -/
theorem sandy_walk (start : Point) : 
  let p1 := move start Direction.South 20
  let p2 := move p1 Direction.East 20
  let p3 := move p2 Direction.North 20
  let final := move p3 Direction.East 20
  final.x = start.x + 40 ∧ final.y = start.y :=
by
  sorry

#check sandy_walk

end NUMINAMATH_CALUDE_sandy_walk_l2462_246210


namespace NUMINAMATH_CALUDE_bus_capacity_problem_l2462_246202

theorem bus_capacity_problem (capacity : ℕ) (first_trip_fraction : ℚ) (total_people : ℕ) 
  (h1 : capacity = 200)
  (h2 : first_trip_fraction = 3 / 4)
  (h3 : total_people = 310) :
  ∃ (return_trip_fraction : ℚ), 
    (first_trip_fraction * capacity + return_trip_fraction * capacity = total_people) ∧
    return_trip_fraction = 4 / 5 :=
by sorry

end NUMINAMATH_CALUDE_bus_capacity_problem_l2462_246202


namespace NUMINAMATH_CALUDE_inequality_comparison_l2462_246200

theorem inequality_comparison : 
  (¬ (0 < -1/2)) ∧ 
  (¬ (4/5 < -6/7)) ∧ 
  (9/8 > 8/9) ∧ 
  (¬ (-4 > -3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_comparison_l2462_246200


namespace NUMINAMATH_CALUDE_parallel_postulate_l2462_246273

/-- A line in a plane --/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Determines if a point is on a line --/
def Point.isOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Two lines are parallel if they have the same slope --/
def Line.isParallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- The statement of the parallel postulate --/
theorem parallel_postulate (L : Line) (P : Point) 
  (h : ¬ P.isOnLine L) : 
  ∃! (M : Line), M.isParallel L ∧ P.isOnLine M :=
sorry


end NUMINAMATH_CALUDE_parallel_postulate_l2462_246273


namespace NUMINAMATH_CALUDE_dune_buggy_speed_l2462_246204

theorem dune_buggy_speed (S : ℝ) : 
  (1/3 : ℝ) * S + (1/3 : ℝ) * (S + 12) + (1/3 : ℝ) * (S - 18) = 58 → S = 60 := by
  sorry

end NUMINAMATH_CALUDE_dune_buggy_speed_l2462_246204


namespace NUMINAMATH_CALUDE_sin_300_degrees_l2462_246226

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l2462_246226


namespace NUMINAMATH_CALUDE_identity_is_unique_solution_l2462_246238

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

/-- The main theorem stating that the identity function is the only solution -/
theorem identity_is_unique_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f → ∀ x : ℝ, f x = x :=
by sorry

end NUMINAMATH_CALUDE_identity_is_unique_solution_l2462_246238


namespace NUMINAMATH_CALUDE_profit_share_ratio_l2462_246261

/-- Given two investors P and Q with their respective investments, 
    calculate the ratio of their profit shares. -/
theorem profit_share_ratio 
  (p_investment q_investment : ℕ) 
  (h_p : p_investment = 40000) 
  (h_q : q_investment = 60000) : 
  (p_investment : ℚ) / q_investment = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_ratio_l2462_246261


namespace NUMINAMATH_CALUDE_poster_enlargement_l2462_246215

/-- Calculates the height of an enlarged poster while maintaining proportions -/
def enlargedPosterHeight (originalWidth originalHeight newWidth : ℚ) : ℚ :=
  (newWidth / originalWidth) * originalHeight

/-- Theorem: Given a poster with original dimensions of 3 inches wide and 2 inches tall,
    when enlarged proportionally to a width of 12 inches, the new height will be 8 inches -/
theorem poster_enlargement :
  let originalWidth : ℚ := 3
  let originalHeight : ℚ := 2
  let newWidth : ℚ := 12
  enlargedPosterHeight originalWidth originalHeight newWidth = 8 := by
  sorry

end NUMINAMATH_CALUDE_poster_enlargement_l2462_246215


namespace NUMINAMATH_CALUDE_five_integer_chords_l2462_246285

/-- Represents a circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distanceFromCenter : ℝ

/-- Counts the number of integer-length chords through P -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem five_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 10)
  (h2 : c.distanceFromCenter = 6) : 
  countIntegerChords c = 5 :=
sorry

end NUMINAMATH_CALUDE_five_integer_chords_l2462_246285


namespace NUMINAMATH_CALUDE_point_on_circle_l2462_246279

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculate the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Check if a point lies on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  squaredDistance p c.center = c.radius^2

/-- The origin point (0,0) -/
def origin : Point := ⟨0, 0⟩

/-- The given point P(-3,4) -/
def pointP : Point := ⟨-3, 4⟩

/-- The circle with center at origin and radius 5 -/
def circleO : Circle := ⟨origin, 5⟩

theorem point_on_circle : isOnCircle pointP circleO := by
  sorry

end NUMINAMATH_CALUDE_point_on_circle_l2462_246279


namespace NUMINAMATH_CALUDE_contribution_problem_l2462_246284

/-- The contribution problem -/
theorem contribution_problem (total_sum : ℕ) : 
  (10 : ℕ) * 300 = total_sum ∧ 
  (15 : ℕ) * (300 - 100) = total_sum := by
  sorry

#check contribution_problem

end NUMINAMATH_CALUDE_contribution_problem_l2462_246284


namespace NUMINAMATH_CALUDE_sin_difference_simplification_l2462_246220

theorem sin_difference_simplification (x y : ℝ) : 
  Real.sin (x + y) * Real.cos y - Real.cos (x + y) * Real.sin y = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_simplification_l2462_246220


namespace NUMINAMATH_CALUDE_line_parameterization_solution_l2462_246208

/-- The line equation y = 2x - 8 -/
def line_eq (x y : ℝ) : Prop := y = 2 * x - 8

/-- The parameterization of the line -/
def parameterization (s m t : ℝ) : ℝ × ℝ :=
  (s + 6 * t, 5 + m * t)

/-- The theorem stating that s = 13/2 and m = 11 satisfy the conditions -/
theorem line_parameterization_solution :
  let s : ℝ := 13/2
  let m : ℝ := 11
  ∃ t : ℝ, 
    let (x, y) := parameterization s m t
    x = 12 ∧ line_eq x y :=
  sorry

end NUMINAMATH_CALUDE_line_parameterization_solution_l2462_246208


namespace NUMINAMATH_CALUDE_revenue_increase_percentage_l2462_246235

/-- Calculates the percentage increase in revenue given initial and new package volumes and prices. -/
theorem revenue_increase_percentage
  (initial_volume : ℝ)
  (initial_price : ℝ)
  (new_volume : ℝ)
  (new_price : ℝ)
  (h1 : initial_volume = 1)
  (h2 : initial_price = 60)
  (h3 : new_volume = 0.9)
  (h4 : new_price = 81) :
  (new_price / new_volume - initial_price / initial_volume) / (initial_price / initial_volume) * 100 = 50 := by
sorry


end NUMINAMATH_CALUDE_revenue_increase_percentage_l2462_246235


namespace NUMINAMATH_CALUDE_computer_science_marks_l2462_246214

theorem computer_science_marks 
  (geography : ℕ) 
  (history_government : ℕ) 
  (art : ℕ) 
  (modern_literature : ℕ) 
  (average : ℚ) 
  (h1 : geography = 56)
  (h2 : history_government = 60)
  (h3 : art = 72)
  (h4 : modern_literature = 80)
  (h5 : average = 70.6)
  : ∃ (computer_science : ℕ),
    (geography + history_government + art + computer_science + modern_literature) / 5 = average ∧ 
    computer_science = 85 := by
sorry

end NUMINAMATH_CALUDE_computer_science_marks_l2462_246214


namespace NUMINAMATH_CALUDE_sqrt2_not_in_rational_intervals_l2462_246221

theorem sqrt2_not_in_rational_intervals (p q : ℕ) (h_coprime : Nat.Coprime p q) 
  (h_p_lt_q : p < q) (h_q_ne_0 : q ≠ 0) : 
  |Real.sqrt 2 / 2 - p / q| > 1 / (4 * q^2) :=
sorry

end NUMINAMATH_CALUDE_sqrt2_not_in_rational_intervals_l2462_246221


namespace NUMINAMATH_CALUDE_baseball_cards_distribution_l2462_246268

theorem baseball_cards_distribution (n : ℕ) (h : n > 0) :
  ∃ (cards_per_friend : ℕ), 
    cards_per_friend * n = 12 ∧ 
    cards_per_friend = 12 / n :=
sorry

end NUMINAMATH_CALUDE_baseball_cards_distribution_l2462_246268


namespace NUMINAMATH_CALUDE_modular_inverse_7_mod_120_l2462_246260

theorem modular_inverse_7_mod_120 :
  ∃ (x : ℕ), x < 120 ∧ (7 * x) % 120 = 1 ∧ x = 103 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_7_mod_120_l2462_246260


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_l2462_246266

theorem sum_of_digits_of_power : ∃ (tens ones : ℕ),
  (tens * 10 + ones = (3 + 4)^15 % 100) ∧ 
  (tens + ones = 7) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_l2462_246266


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2462_246297

theorem negation_of_proposition (p : ℝ → Prop) :
  (∀ n ∈ Set.Icc 1 2, n^2 < 3*n + 4) ↔ ¬(∃ n ∈ Set.Icc 1 2, n^2 ≥ 3*n + 4) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2462_246297


namespace NUMINAMATH_CALUDE_oscillating_cosine_shift_l2462_246232

theorem oscillating_cosine_shift (a b c d : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d →
  (∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) →
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_oscillating_cosine_shift_l2462_246232


namespace NUMINAMATH_CALUDE_cosine_equality_l2462_246241

theorem cosine_equality (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (745 * π / 180) →
  n = 25 ∨ n = -25 := by
sorry

end NUMINAMATH_CALUDE_cosine_equality_l2462_246241


namespace NUMINAMATH_CALUDE_fuel_consumption_result_l2462_246239

/-- Represents the fuel consumption problem --/
structure FuelConsumption where
  initial_capacity : ℝ
  january_level : ℝ
  may_level : ℝ

/-- Calculates the total fuel consumption given the problem parameters --/
def total_consumption (fc : FuelConsumption) : ℝ :=
  (fc.initial_capacity - fc.january_level) + (fc.initial_capacity - fc.may_level)

/-- Theorem stating that the total fuel consumption is 4582 L --/
theorem fuel_consumption_result (fc : FuelConsumption) 
  (h1 : fc.initial_capacity = 3000)
  (h2 : fc.january_level = 180)
  (h3 : fc.may_level = 1238) :
  total_consumption fc = 4582 := by
  sorry

#eval total_consumption ⟨3000, 180, 1238⟩

end NUMINAMATH_CALUDE_fuel_consumption_result_l2462_246239


namespace NUMINAMATH_CALUDE_greatest_consecutive_integers_sum_120_l2462_246231

theorem greatest_consecutive_integers_sum_120 :
  (∀ n : ℕ, n > 15 → ¬∃ a : ℕ, (Finset.range n).sum (λ i => a + i) = 120) ∧
  ∃ a : ℕ, (Finset.range 15).sum (λ i => a + i) = 120 :=
by sorry

end NUMINAMATH_CALUDE_greatest_consecutive_integers_sum_120_l2462_246231


namespace NUMINAMATH_CALUDE_at_least_one_third_l2462_246253

theorem at_least_one_third (a b c : ℝ) (h : a + b + c = 1) :
  (a ≥ 1/3) ∨ (b ≥ 1/3) ∨ (c ≥ 1/3) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_third_l2462_246253


namespace NUMINAMATH_CALUDE_cricket_overs_l2462_246299

theorem cricket_overs (initial_rate : ℝ) (remaining_rate : ℝ) (remaining_overs : ℝ) (target : ℝ) :
  initial_rate = 4.2 →
  remaining_rate = 8 →
  remaining_overs = 30 →
  target = 324 →
  ∃ (initial_overs : ℝ), 
    initial_overs * initial_rate + remaining_overs * remaining_rate = target ∧ 
    initial_overs = 20 := by
  sorry

end NUMINAMATH_CALUDE_cricket_overs_l2462_246299


namespace NUMINAMATH_CALUDE_exponent_rule_product_power_l2462_246265

theorem exponent_rule_product_power (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_rule_product_power_l2462_246265


namespace NUMINAMATH_CALUDE_complex_number_solutions_l2462_246259

theorem complex_number_solutions : 
  ∀ z : ℂ, z^2 = -45 - 28*I ∧ z^3 = 8 + 26*I →
  z = Complex.mk (Real.sqrt 10) (-Real.sqrt 140) ∨
  z = Complex.mk (-Real.sqrt 10) (Real.sqrt 140) := by
sorry

end NUMINAMATH_CALUDE_complex_number_solutions_l2462_246259


namespace NUMINAMATH_CALUDE_min_value_theorem_l2462_246212

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (a + b) * (a + c) = 4) : 
  2 * a + b + c ≥ 4 ∧ (2 * a + b + c = 4 ↔ b = c) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2462_246212


namespace NUMINAMATH_CALUDE_essay_competition_probability_l2462_246262

theorem essay_competition_probability (n : ℕ) (h : n = 6) :
  let p := (n - 1) / n
  p = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_essay_competition_probability_l2462_246262


namespace NUMINAMATH_CALUDE_mean_height_is_60_l2462_246245

/-- Represents the stem and leaf plot of player heights --/
def stemAndLeaf : List (Nat × List Nat) := [
  (4, [9]),
  (5, [2, 3, 5, 8, 8, 9]),
  (6, [0, 1, 1, 2, 6, 8, 9, 9])
]

/-- Calculates the total sum of heights from the stem and leaf plot --/
def sumHeights (plot : List (Nat × List Nat)) : Nat :=
  plot.foldl (fun acc (stem, leaves) => 
    acc + stem * 10 * leaves.length + leaves.sum
  ) 0

/-- Calculates the number of players from the stem and leaf plot --/
def countPlayers (plot : List (Nat × List Nat)) : Nat :=
  plot.foldl (fun acc (_, leaves) => acc + leaves.length) 0

/-- The mean height of the players --/
def meanHeight : ℚ := (sumHeights stemAndLeaf : ℚ) / (countPlayers stemAndLeaf : ℚ)

theorem mean_height_is_60 : meanHeight = 60 := by
  sorry

end NUMINAMATH_CALUDE_mean_height_is_60_l2462_246245
