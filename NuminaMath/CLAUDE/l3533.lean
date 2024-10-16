import Mathlib

namespace NUMINAMATH_CALUDE_obese_employee_is_male_prob_l3533_353332

-- Define the company's employee structure
structure Company where
  male_ratio : ℚ
  female_ratio : ℚ
  male_obese_ratio : ℚ
  female_obese_ratio : ℚ

-- Define the probability function
def prob_obese_is_male (c : Company) : ℚ :=
  (c.male_ratio * c.male_obese_ratio) / 
  (c.male_ratio * c.male_obese_ratio + c.female_ratio * c.female_obese_ratio)

-- Theorem statement
theorem obese_employee_is_male_prob 
  (c : Company) 
  (h1 : c.male_ratio = 3/5) 
  (h2 : c.female_ratio = 2/5)
  (h3 : c.male_obese_ratio = 1/5)
  (h4 : c.female_obese_ratio = 1/10) :
  prob_obese_is_male c = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_obese_employee_is_male_prob_l3533_353332


namespace NUMINAMATH_CALUDE_solution_l3533_353325

def problem (a b c d : ℚ) : Prop :=
  a + b + c + d = 406 ∧
  a = (1/2) * b ∧
  b = (1/2) * c ∧
  d = (1/3) * c

theorem solution :
  ∃ a b c d : ℚ,
    problem a b c d ∧
    a = 48.72 ∧
    b = 97.44 ∧
    c = 194.88 ∧
    d = 64.96 :=
by sorry

end NUMINAMATH_CALUDE_solution_l3533_353325


namespace NUMINAMATH_CALUDE_construction_workers_l3533_353323

theorem construction_workers (initial_workers : ℕ) (initial_days : ℕ) (remaining_days : ℕ)
  (initial_work_fraction : ℚ) (h1 : initial_workers = 60)
  (h2 : initial_days = 18) (h3 : remaining_days = 12)
  (h4 : initial_work_fraction = 1/3) :
  ∃ (additional_workers : ℕ),
    additional_workers = 60 ∧
    (additional_workers + initial_workers : ℚ) * remaining_days * initial_work_fraction =
    (1 - initial_work_fraction) * initial_workers * initial_days :=
by sorry

end NUMINAMATH_CALUDE_construction_workers_l3533_353323


namespace NUMINAMATH_CALUDE_franks_decks_l3533_353397

theorem franks_decks (deck_cost : ℕ) (friend_decks : ℕ) (total_spent : ℕ) :
  deck_cost = 7 →
  friend_decks = 2 →
  total_spent = 35 →
  ∃ (frank_decks : ℕ), frank_decks * deck_cost + friend_decks * deck_cost = total_spent ∧ frank_decks = 3 :=
by sorry

end NUMINAMATH_CALUDE_franks_decks_l3533_353397


namespace NUMINAMATH_CALUDE_min_value_of_product_l3533_353353

theorem min_value_of_product (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 1) :
  (a - b) * (b - c) * (c - d) * (d - a) ≥ -1/8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_product_l3533_353353


namespace NUMINAMATH_CALUDE_consecutive_integers_square_difference_l3533_353369

theorem consecutive_integers_square_difference (n : ℕ) : 
  n > 0 ∧ n + (n + 1) < 100 → (n + 1)^2 - n^2 = 79 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_difference_l3533_353369


namespace NUMINAMATH_CALUDE_election_win_percentage_l3533_353399

theorem election_win_percentage 
  (total_voters : ℕ) 
  (republican_ratio : ℚ) 
  (democrat_ratio : ℚ) 
  (republican_for_x : ℚ) 
  (democrat_for_x : ℚ) :
  republican_ratio + democrat_ratio = 1 →
  republican_ratio / democrat_ratio = 3 / 2 →
  republican_for_x = 3 / 4 →
  democrat_for_x = 3 / 20 →
  let total_for_x := republican_ratio * republican_for_x + democrat_ratio * democrat_for_x
  let total_for_y := 1 - total_for_x
  (total_for_x - total_for_y) / (total_for_x + total_for_y) = 1 / 50 :=
by sorry

end NUMINAMATH_CALUDE_election_win_percentage_l3533_353399


namespace NUMINAMATH_CALUDE_lilith_cap_collection_l3533_353364

/-- Calculates the number of caps Lilith has collected after a given number of years -/
def caps_collected (years : ℕ) : ℕ :=
  let first_year_caps := 3 * 12
  let subsequent_years_caps := 5 * 12 * (years - 1)
  let christmas_caps := 40 * years
  let total_caps := first_year_caps + subsequent_years_caps + christmas_caps
  let lost_caps := 15 * years
  total_caps - lost_caps

/-- Theorem stating that Lilith has collected 401 caps after 5 years -/
theorem lilith_cap_collection : caps_collected 5 = 401 := by
  sorry

end NUMINAMATH_CALUDE_lilith_cap_collection_l3533_353364


namespace NUMINAMATH_CALUDE_tg_arccos_leq_sin_arctg_l3533_353366

theorem tg_arccos_leq_sin_arctg (x : ℝ) : 
  x ∈ Set.Icc (-1 : ℝ) 1 →
  (Real.tan (Real.arccos x) ≤ Real.sin (Real.arctan x) ↔ 
   x ∈ Set.Icc (-(Real.sqrt (Real.sqrt (1/2)))) 0 ∪ Set.Icc (Real.sqrt (Real.sqrt (1/2))) 1) :=
by sorry

end NUMINAMATH_CALUDE_tg_arccos_leq_sin_arctg_l3533_353366


namespace NUMINAMATH_CALUDE_cake_sale_theorem_l3533_353309

/-- Represents the pricing and sales model for small cakes in a charity sale event -/
structure CakeSaleModel where
  initial_price : ℝ
  initial_sales : ℕ
  price_increase : ℝ
  sales_decrease : ℕ
  max_price : ℝ

/-- Calculates the new price after two equal percentage increases -/
def price_after_two_increases (model : CakeSaleModel) (percent : ℝ) : ℝ :=
  model.initial_price * (1 + percent) ^ 2

/-- Calculates the total sales per hour given a price increase -/
def total_sales (model : CakeSaleModel) (price_increase : ℝ) : ℝ :=
  (model.initial_price + price_increase) * 
  (model.initial_sales - model.sales_decrease * price_increase)

/-- The main theorem stating the correct percentage increase and optimal selling price -/
theorem cake_sale_theorem (model : CakeSaleModel) 
  (h1 : model.initial_price = 6)
  (h2 : model.initial_sales = 30)
  (h3 : model.price_increase = 1)
  (h4 : model.sales_decrease = 2)
  (h5 : model.max_price = 10) :
  ∃ (percent : ℝ) (optimal_price : ℝ),
    price_after_two_increases model percent = 8.64 ∧
    percent = 0.2 ∧
    total_sales model (optimal_price - model.initial_price) = 216 ∧
    optimal_price = 9 ∧
    optimal_price ≤ model.max_price :=
by sorry

end NUMINAMATH_CALUDE_cake_sale_theorem_l3533_353309


namespace NUMINAMATH_CALUDE_adams_earnings_l3533_353306

/-- Adam's lawn mowing earnings problem -/
theorem adams_earnings (rate : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ) 
  (h1 : rate = 9)
  (h2 : total_lawns = 12)
  (h3 : forgotten_lawns = 8) :
  (total_lawns - forgotten_lawns) * rate = 36 := by
  sorry

#check adams_earnings

end NUMINAMATH_CALUDE_adams_earnings_l3533_353306


namespace NUMINAMATH_CALUDE_investment_earnings_l3533_353329

/-- Calculates the earnings from a stock investment --/
def calculate_earnings (investment : ℕ) (dividend_rate : ℕ) (market_price : ℕ) (face_value : ℕ) : ℕ :=
  let shares := investment / market_price
  let total_face_value := shares * face_value
  (dividend_rate * total_face_value) / 100

/-- Theorem stating that the given investment yields the expected earnings --/
theorem investment_earnings : 
  calculate_earnings 5760 1623 64 100 = 146070 := by
  sorry

end NUMINAMATH_CALUDE_investment_earnings_l3533_353329


namespace NUMINAMATH_CALUDE_percentage_without_fulltime_jobs_is_19_l3533_353301

/-- The percentage of parents who do not hold full-time jobs -/
def percentage_without_fulltime_jobs (total_parents : ℕ) 
  (mother_ratio : ℚ) (father_ratio : ℚ) (women_ratio : ℚ) : ℚ :=
  let mothers := (women_ratio * total_parents).floor
  let fathers := total_parents - mothers
  let mothers_with_jobs := (mother_ratio * mothers).floor
  let fathers_with_jobs := (father_ratio * fathers).floor
  let parents_without_jobs := total_parents - mothers_with_jobs - fathers_with_jobs
  (parents_without_jobs : ℚ) / total_parents * 100

/-- Theorem stating that given the conditions in the problem, 
    the percentage of parents without full-time jobs is 19% -/
theorem percentage_without_fulltime_jobs_is_19 :
  ∀ n : ℕ, n > 0 → 
  percentage_without_fulltime_jobs n (9/10) (3/4) (2/5) = 19 := by
  sorry

end NUMINAMATH_CALUDE_percentage_without_fulltime_jobs_is_19_l3533_353301


namespace NUMINAMATH_CALUDE_cone_syrup_amount_l3533_353376

/-- The amount of chocolate syrup used on each shake in ounces -/
def syrup_per_shake : ℕ := 4

/-- The number of shakes sold -/
def num_shakes : ℕ := 2

/-- The number of cones sold -/
def num_cones : ℕ := 1

/-- The total amount of chocolate syrup used in ounces -/
def total_syrup : ℕ := 14

/-- The amount of chocolate syrup used on each cone in ounces -/
def syrup_per_cone : ℕ := total_syrup - (syrup_per_shake * num_shakes)

theorem cone_syrup_amount : syrup_per_cone = 6 := by
  sorry

end NUMINAMATH_CALUDE_cone_syrup_amount_l3533_353376


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3533_353319

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (1/2 * (4*x^2 - 1) = (x^2 - 50*x - 20) * (x^2 + 25*x + 10)) ∧ x = 26 + Real.sqrt 677 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3533_353319


namespace NUMINAMATH_CALUDE_total_legs_in_farm_l3533_353333

theorem total_legs_in_farm (total_animals : Nat) (num_ducks : Nat) (duck_legs : Nat) (dog_legs : Nat) :
  total_animals = 8 →
  num_ducks = 4 →
  duck_legs = 2 →
  dog_legs = 4 →
  (num_ducks * duck_legs + (total_animals - num_ducks) * dog_legs) = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_in_farm_l3533_353333


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3533_353351

/-- Proves that given a sum P at simple interest for 10 years, 
    if increasing the interest rate by 3% results in $300 more interest, 
    then P = $1000. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 3) * 10 / 100 - P * R * 10 / 100 = 300) → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3533_353351


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3533_353395

theorem quadratic_factorization (x : ℝ) : x^2 + 2*x - 3 = (x + 3) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3533_353395


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_l3533_353389

-- Define the quadratic equations
def quadratic1 (a x : ℝ) : ℝ := a * x^2 + (a + 1) * x - 2
def quadratic2 (a x : ℝ) : ℝ := (1 - a) * x^2 + (a + 1) * x - 2

-- Define the conditions for real solutions
def realSolutions1 (a : ℝ) : Prop :=
  a < -5 - 2 * Real.sqrt 6 ∨ (2 * Real.sqrt 6 - 5 < a ∧ a < 0) ∨ a > 0

def realSolutions2 (a : ℝ) : Prop :=
  a < 1 ∨ (1 < a ∧ a < 3) ∨ a > 3

-- Theorem statement
theorem quadratic_real_solutions :
  ∀ a : ℝ,
    (∃ x : ℝ, quadratic1 a x = 0) ↔ realSolutions1 a ∧
    (∃ x : ℝ, quadratic2 a x = 0) ↔ realSolutions2 a :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_l3533_353389


namespace NUMINAMATH_CALUDE_books_read_proof_l3533_353341

def total_books (megan_books kelcie_books greg_books : ℕ) : ℕ :=
  megan_books + kelcie_books + greg_books

theorem books_read_proof (megan_books : ℕ) 
  (h1 : megan_books = 32)
  (h2 : ∃ kelcie_books : ℕ, kelcie_books = megan_books / 4)
  (h3 : ∃ greg_books : ℕ, greg_books = 2 * (megan_books / 4) + 9) :
  ∃ total : ℕ, total_books megan_books (megan_books / 4) (2 * (megan_books / 4) + 9) = 65 := by
  sorry

end NUMINAMATH_CALUDE_books_read_proof_l3533_353341


namespace NUMINAMATH_CALUDE_valid_grid_exists_l3533_353382

/-- A type representing a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if two numbers are adjacent in the grid -/
def adjacent (i j i' j' : Fin 3) : Prop :=
  (i = i' ∧ j.val + 1 = j'.val) ∨
  (i = i' ∧ j'.val + 1 = j.val) ∨
  (j = j' ∧ i.val + 1 = i'.val) ∨
  (j = j' ∧ i'.val + 1 = i.val)

/-- The main theorem stating the existence of a valid grid -/
theorem valid_grid_exists : ∃ (g : Grid),
  (∀ i j i' j', adjacent i j i' j' → (g i j ∣ g i' j' ∨ g i' j' ∣ g i j)) ∧
  (∀ i j, g i j ≤ 25) ∧
  (∀ i j i' j', (i, j) ≠ (i', j') → g i j ≠ g i' j') :=
sorry

end NUMINAMATH_CALUDE_valid_grid_exists_l3533_353382


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3533_353327

theorem unique_solution_condition (a b c : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = c * x + b) ↔ c ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3533_353327


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3533_353302

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 16) (h2 : x - y = 2) : x^2 - y^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3533_353302


namespace NUMINAMATH_CALUDE_right_triangle_area_l3533_353315

/-- The area of a right triangle with one leg of length 3 and hypotenuse of length 5 is 6. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 3) (h2 : c = 5) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3533_353315


namespace NUMINAMATH_CALUDE_total_fleas_is_40_l3533_353354

/-- The number of fleas on Gertrude the chicken -/
def gertrudeFleas : ℕ := 10

/-- The number of fleas on Olive the chicken -/
def oliveFleas : ℕ := gertrudeFleas / 2

/-- The number of fleas on Maud the chicken -/
def maudFleas : ℕ := 5 * oliveFleas

/-- The total number of fleas on all three chickens -/
def totalFleas : ℕ := gertrudeFleas + oliveFleas + maudFleas

theorem total_fleas_is_40 : totalFleas = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_fleas_is_40_l3533_353354


namespace NUMINAMATH_CALUDE_barbed_wire_cost_l3533_353330

theorem barbed_wire_cost (field_area : ℝ) (wire_cost_per_meter : ℝ) (gate_width : ℝ) (num_gates : ℕ) : 
  field_area = 3136 ∧ 
  wire_cost_per_meter = 1.4 ∧ 
  gate_width = 1 ∧ 
  num_gates = 2 → 
  (Real.sqrt field_area * 4 - (gate_width * num_gates)) * wire_cost_per_meter = 310.8 := by
  sorry

end NUMINAMATH_CALUDE_barbed_wire_cost_l3533_353330


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l3533_353361

/-- A line passing through a point with equal intercepts on both axes -/
structure EqualInterceptLine where
  -- The point the line passes through
  point : ℝ × ℝ
  -- The equation of the line in the form ax + by = c
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through the given point
  point_on_line : a * point.1 + b * point.2 = c
  -- The line has equal intercepts on both axes
  equal_intercepts : c / a = c / b

/-- The theorem stating the equation of the line -/
theorem equal_intercept_line_equation :
  ∀ (l : EqualInterceptLine),
  l.point = (3, 2) →
  (l.a = 2 ∧ l.b = -3 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = 5) :=
by sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l3533_353361


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2017_l3533_353350

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f is increasing if f(x) < f(y) whenever x < y -/
def IsIncreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

/-- An arithmetic sequence with common difference d -/
def ArithmeticSequence (x : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, x (n + 1) = x n + d

theorem arithmetic_sequence_2017 (f : ℝ → ℝ) (x : ℕ → ℝ) :
  IsOdd f →
  IsIncreasing f →
  ArithmeticSequence x 2 →
  f (x 7) + f (x 8) = 0 →
  x 2017 = 4019 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2017_l3533_353350


namespace NUMINAMATH_CALUDE_worker_completion_time_l3533_353367

/-- Given two workers who can complete a job together in a certain time,
    and one worker's individual completion time, find the other worker's time. -/
theorem worker_completion_time
  (total_time : ℝ)
  (together_time : ℝ)
  (b_time : ℝ)
  (h1 : together_time > 0)
  (h2 : b_time > 0)
  (h3 : total_time > 0)
  (h4 : 1 / together_time = 1 / total_time + 1 / b_time) :
  total_time = 15 :=
sorry

end NUMINAMATH_CALUDE_worker_completion_time_l3533_353367


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3533_353348

theorem smallest_number_satisfying_conditions : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬((y + 3).ModEq 0 7 ∧ (y - 5).ModEq 0 8)) ∧
  (x + 3).ModEq 0 7 ∧ 
  (x - 5).ModEq 0 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3533_353348


namespace NUMINAMATH_CALUDE_sphere_volume_from_cube_surface_l3533_353371

theorem sphere_volume_from_cube_surface (L : ℝ) : 
  let cube_side : ℝ := 3
  let cube_surface_area : ℝ := 6 * cube_side^2
  let sphere_radius : ℝ := (cube_surface_area / (4 * Real.pi))^(1/2)
  let sphere_volume : ℝ := (4/3) * Real.pi * sphere_radius^3
  sphere_volume = L * (15^(1/2)) / (Real.pi^(1/2)) →
  L = 108 * (5^(1/2)) / 5 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_from_cube_surface_l3533_353371


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l3533_353349

theorem simplify_nested_roots (a : ℝ) (ha : a > 0) :
  (((a^16)^(1/5))^(1/4))^6 * (((a^16)^(1/4))^(1/5))^6 = a^(48/5) :=
sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l3533_353349


namespace NUMINAMATH_CALUDE_ferry_tourist_count_l3533_353379

/-- Represents the ferry schedule and passenger count --/
structure FerrySchedule where
  start_time : Nat -- 10 AM represented as 0
  end_time : Nat   -- 3 PM represented as 10
  initial_passengers : Nat
  passenger_decrease : Nat

/-- Calculates the total number of tourists transported by the ferry --/
def total_tourists (schedule : FerrySchedule) : Nat :=
  let num_trips := schedule.end_time - schedule.start_time + 1
  let arithmetic_sum := num_trips * (2 * schedule.initial_passengers - (num_trips - 1) * schedule.passenger_decrease)
  arithmetic_sum / 2

/-- Theorem stating that the total number of tourists is 990 --/
theorem ferry_tourist_count :
  ∀ (schedule : FerrySchedule),
    schedule.start_time = 0 ∧
    schedule.end_time = 10 ∧
    schedule.initial_passengers = 100 ∧
    schedule.passenger_decrease = 2 →
    total_tourists schedule = 990 := by
  sorry

end NUMINAMATH_CALUDE_ferry_tourist_count_l3533_353379


namespace NUMINAMATH_CALUDE_circle_equation_radius_6_l3533_353377

theorem circle_equation_radius_6 (x y k : ℝ) : 
  (∃ h i : ℝ, ∀ x y : ℝ, (x - h)^2 + (y - i)^2 = 6^2 ↔ x^2 + 10*x + y^2 + 6*y - k = 0) ↔ 
  k = 2 := by sorry

end NUMINAMATH_CALUDE_circle_equation_radius_6_l3533_353377


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3533_353312

theorem polynomial_division_remainder :
  let f (x : ℝ) := x^4 - 7*x^3 + 18*x^2 - 28*x + 15
  let g (x : ℝ) := x^2 - 3*x + 16/3
  let q (x : ℝ) := x^2 - 4*x + 10/3
  let r (x : ℝ) := 2*x + 103/9
  ∀ x, f x = g x * q x + r x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3533_353312


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_cyclic_polygon_l3533_353388

/-- A cyclic polygon is a polygon whose vertices all lie on a single circle. -/
structure CyclicPolygon where
  n : ℕ
  sides_ge_4 : n ≥ 4

/-- The sum of interior angles of a cyclic polygon. -/
def sum_of_interior_angles (p : CyclicPolygon) : ℝ :=
  (p.n - 2) * 180

/-- Theorem: The sum of interior angles of a cyclic polygon with n sides is (n-2) * 180°. -/
theorem sum_of_interior_angles_cyclic_polygon (p : CyclicPolygon) :
  sum_of_interior_angles p = (p.n - 2) * 180 := by
  sorry

#check sum_of_interior_angles_cyclic_polygon

end NUMINAMATH_CALUDE_sum_of_interior_angles_cyclic_polygon_l3533_353388


namespace NUMINAMATH_CALUDE_probability_of_square_l3533_353345

/-- The probability of selecting a square from a set of figures -/
theorem probability_of_square (total_figures : ℕ) (square_count : ℕ) 
  (h1 : total_figures = 10) (h2 : square_count = 3) : 
  (square_count : ℚ) / total_figures = 3 / 10 := by
  sorry

#check probability_of_square

end NUMINAMATH_CALUDE_probability_of_square_l3533_353345


namespace NUMINAMATH_CALUDE_root_intersection_l3533_353398

-- Define the original equation
def original_equation (x : ℝ) : Prop := x^2 - 2*x = 0

-- Define the roots of the original equation
def is_root (x : ℝ) : Prop := original_equation x

-- Define the pairs of equations
def pair_A (x y : ℝ) : Prop := (y = x^2 ∧ y = 2*x)
def pair_B (x y : ℝ) : Prop := (y = x^2 - 2*x ∧ y = 0)
def pair_C (x y : ℝ) : Prop := (y = x ∧ y = x - 2)
def pair_D (x y : ℝ) : Prop := (y = x^2 - 2*x + 1 ∧ y = 1)
def pair_E (x y : ℝ) : Prop := (y = x^2 - 1 ∧ y = 2*x - 1)

-- Theorem stating that pair C does not yield the roots while others do
theorem root_intersection :
  (∃ x y : ℝ, pair_C x y ∧ is_root x) = false ∧
  (∃ x y : ℝ, pair_A x y ∧ is_root x) = true ∧
  (∃ x y : ℝ, pair_B x y ∧ is_root x) = true ∧
  (∃ x y : ℝ, pair_D x y ∧ is_root x) = true ∧
  (∃ x y : ℝ, pair_E x y ∧ is_root x) = true :=
by sorry

end NUMINAMATH_CALUDE_root_intersection_l3533_353398


namespace NUMINAMATH_CALUDE_overlap_angle_is_90_degrees_l3533_353375

/-- A regular octagon -/
structure RegularOctagon where
  sides : Fin 8 → ℝ × ℝ
  is_regular : ∀ (i j : Fin 8), dist (sides i) (sides ((i + 1) % 8)) = dist (sides j) (sides ((j + 1) % 8))

/-- The angle at the intersection point when two non-adjacent sides of a regular octagon overlap -/
def overlap_angle (octagon : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The angle at the intersection point when two non-adjacent sides of a regular octagon overlap is 90° -/
theorem overlap_angle_is_90_degrees (octagon : RegularOctagon) : 
  overlap_angle octagon = 90 :=
sorry

end NUMINAMATH_CALUDE_overlap_angle_is_90_degrees_l3533_353375


namespace NUMINAMATH_CALUDE_log_six_eighteen_l3533_353334

theorem log_six_eighteen (a b : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 3 / Real.log 10 = b) :
  Real.log 18 / Real.log 6 = (a + 2*b) / (a + b) := by
sorry

end NUMINAMATH_CALUDE_log_six_eighteen_l3533_353334


namespace NUMINAMATH_CALUDE_transport_cost_calculation_l3533_353393

/-- The transport cost for Ramesh's refrigerator purchase --/
def transport_cost : ℕ := by sorry

/-- The labelled price of the refrigerator before discount --/
def labelled_price : ℕ := by sorry

/-- The discounted price Ramesh paid for the refrigerator --/
def discounted_price : ℕ := 17500

/-- The installation cost --/
def installation_cost : ℕ := 250

/-- The selling price to earn 10% profit without discount --/
def selling_price : ℕ := 24475

/-- The discount rate applied to the labelled price --/
def discount_rate : ℚ := 1/5

/-- The profit rate desired if no discount was offered --/
def profit_rate : ℚ := 1/10

theorem transport_cost_calculation :
  discounted_price = labelled_price * (1 - discount_rate) ∧
  selling_price = labelled_price * (1 + profit_rate) ∧
  transport_cost + discounted_price + installation_cost = selling_price ∧
  transport_cost = 6725 := by sorry

end NUMINAMATH_CALUDE_transport_cost_calculation_l3533_353393


namespace NUMINAMATH_CALUDE_unique_line_existence_l3533_353320

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def line_passes_through (a b : ℚ) (x y : ℚ) : Prop :=
  x / a + y / b = 1

theorem unique_line_existence :
  ∃! (a b : ℚ), 
    (∃ n : ℕ, a = n ∧ is_prime n ∧ n < 10) ∧ 
    (∃ m : ℕ, b = m ∧ is_even m) ∧ 
    line_passes_through a b 5 4 :=
sorry

end NUMINAMATH_CALUDE_unique_line_existence_l3533_353320


namespace NUMINAMATH_CALUDE_quarter_to_fourth_power_decimal_l3533_353385

theorem quarter_to_fourth_power_decimal : (1 / 4 : ℚ) ^ 4 = 0.00390625 := by
  sorry

end NUMINAMATH_CALUDE_quarter_to_fourth_power_decimal_l3533_353385


namespace NUMINAMATH_CALUDE_ten_thousandths_place_of_5_32_l3533_353308

theorem ten_thousandths_place_of_5_32 : 
  (5 : ℚ) / 32 * 10000 - ((5 : ℚ) / 32 * 10000).floor = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousandths_place_of_5_32_l3533_353308


namespace NUMINAMATH_CALUDE_unique_complex_solution_l3533_353336

theorem unique_complex_solution :
  ∃! (z : ℂ), Complex.abs z < 20 ∧ Complex.cos z = (z - 2) / (z + 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_complex_solution_l3533_353336


namespace NUMINAMATH_CALUDE_woo_jun_age_l3533_353311

theorem woo_jun_age :
  ∀ (w m : ℕ),
  w = m / 4 - 1 →
  m = 5 * w - 5 →
  w = 9 := by
sorry

end NUMINAMATH_CALUDE_woo_jun_age_l3533_353311


namespace NUMINAMATH_CALUDE_smallest_base_for_62_l3533_353391

theorem smallest_base_for_62 : 
  ∃ (b : ℕ), b = 4 ∧ 
  (∀ (x : ℕ), x < b → ¬(b^2 ≤ 62 ∧ 62 < b^3)) ∧
  (b^2 ≤ 62 ∧ 62 < b^3) := by
sorry

end NUMINAMATH_CALUDE_smallest_base_for_62_l3533_353391


namespace NUMINAMATH_CALUDE_combined_salaries_l3533_353396

/-- The combined salaries of B, C, D, and E given A's salary and the average salary of all five -/
theorem combined_salaries 
  (salary_A : ℕ) 
  (average_salary : ℕ) 
  (h1 : salary_A = 8000)
  (h2 : average_salary = 8600) :
  (5 * average_salary) - salary_A = 35000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_l3533_353396


namespace NUMINAMATH_CALUDE_cone_volume_l3533_353360

/-- The volume of a cone with given slant height and lateral surface angle -/
theorem cone_volume (s : ℝ) (θ : ℝ) (h : s = 6) (h' : θ = 2 * π / 3) :
  ∃ (v : ℝ), v = (16 * Real.sqrt 2 / 3) * π ∧ v = (1/3) * π * (s * θ / (2 * π))^2 * Real.sqrt (s^2 - (s * θ / (2 * π))^2) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l3533_353360


namespace NUMINAMATH_CALUDE_congruence_problem_l3533_353347

theorem congruence_problem (x : ℤ) : 
  (5 * x + 9) % 20 = 3 → (3 * x + 14) % 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3533_353347


namespace NUMINAMATH_CALUDE_marias_painting_price_l3533_353394

/-- The selling price of Maria's painting --/
def selling_price (brush_cost canvas_cost paint_cost_per_liter paint_liters earnings : ℕ) : ℕ :=
  brush_cost + canvas_cost + paint_cost_per_liter * paint_liters + earnings

/-- Theorem stating the selling price of Maria's painting --/
theorem marias_painting_price :
  selling_price 20 (3 * 20) 8 5 80 = 200 := by
  sorry

end NUMINAMATH_CALUDE_marias_painting_price_l3533_353394


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l3533_353359

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 17 % 31 ∧ ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 17 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l3533_353359


namespace NUMINAMATH_CALUDE_marble_average_l3533_353321

/-- Given the conditions about the average numbers of marbles of different colors,
    prove that the average number of all three colors is 30. -/
theorem marble_average (R Y B : ℕ) : 
  (R + Y : ℚ) / 2 = 26.5 →
  (B + Y : ℚ) / 2 = 34.5 →
  (R + B : ℚ) / 2 = 29 →
  (R + Y + B : ℚ) / 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_marble_average_l3533_353321


namespace NUMINAMATH_CALUDE_interest_rate_proof_l3533_353339

/-- Represents the rate of interest per annum as a percentage -/
def rate : ℝ := 9

/-- The amount lent to B -/
def principal_B : ℝ := 5000

/-- The amount lent to C -/
def principal_C : ℝ := 3000

/-- The time period for B's loan in years -/
def time_B : ℝ := 2

/-- The time period for C's loan in years -/
def time_C : ℝ := 4

/-- The total interest received from both B and C -/
def total_interest : ℝ := 1980

/-- Theorem stating that the given rate satisfies the problem conditions -/
theorem interest_rate_proof :
  (principal_B * rate * time_B / 100 + principal_C * rate * time_C / 100) = total_interest :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l3533_353339


namespace NUMINAMATH_CALUDE_rabbit_turning_point_theorem_l3533_353303

/-- The point where the rabbit starts moving away from the fox -/
def rabbit_turning_point : ℝ × ℝ := (2.8, 5.6)

/-- The location of the fox -/
def fox_location : ℝ × ℝ := (10, 8)

/-- The slope of the rabbit's path -/
def rabbit_path_slope : ℝ := -3

/-- The y-intercept of the rabbit's path -/
def rabbit_path_intercept : ℝ := 14

/-- The equation of the rabbit's path: y = mx + b -/
def rabbit_path (x : ℝ) : ℝ := rabbit_path_slope * x + rabbit_path_intercept

theorem rabbit_turning_point_theorem :
  let (c, d) := rabbit_turning_point
  let (fx, fy) := fox_location
  -- The turning point lies on the rabbit's path
  d = rabbit_path c ∧
  -- The line from the fox to the turning point is perpendicular to the rabbit's path
  (d - fy) / (c - fx) = -1 / rabbit_path_slope := by
  sorry

end NUMINAMATH_CALUDE_rabbit_turning_point_theorem_l3533_353303


namespace NUMINAMATH_CALUDE_robin_cupcakes_proof_l3533_353305

/-- Represents the number of cupcakes Robin initially made -/
def initial_cupcakes : ℕ := 42

/-- Represents the number of cupcakes Robin sold -/
def sold_cupcakes : ℕ := 22

/-- Represents the number of cupcakes Robin made later -/
def new_cupcakes : ℕ := 39

/-- Represents the final number of cupcakes Robin had -/
def final_cupcakes : ℕ := 59

/-- Proves that the initial number of cupcakes is correct given the conditions -/
theorem robin_cupcakes_proof : 
  initial_cupcakes - sold_cupcakes + new_cupcakes = final_cupcakes :=
by sorry

end NUMINAMATH_CALUDE_robin_cupcakes_proof_l3533_353305


namespace NUMINAMATH_CALUDE_rsa_factorization_l3533_353365

theorem rsa_factorization :
  ∃ (p q : ℕ), 
    p.Prime ∧ 
    q.Prime ∧ 
    p * q = 400000001 ∧ 
    p = 19801 ∧ 
    q = 20201 := by
  sorry

end NUMINAMATH_CALUDE_rsa_factorization_l3533_353365


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3533_353378

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 6 + a 10 = 16)
  (h_a4 : a 4 = 1) :
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3533_353378


namespace NUMINAMATH_CALUDE_tennis_tournament_theorem_l3533_353324

-- Define the number of women players
def n : ℕ := sorry

-- Define the total number of players
def total_players : ℕ := n + 3 * n

-- Define the total number of matches
def total_matches : ℕ := (total_players * (total_players - 1)) / 2

-- Define the number of matches won by women
def women_wins : ℕ := sorry

-- Define the number of matches won by men
def men_wins : ℕ := sorry

-- Theorem stating the conditions and the result to be proved
theorem tennis_tournament_theorem :
  -- Each player plays with every other player
  (∀ p : ℕ, p < total_players → (total_players - 1) * p = total_matches * 2) →
  -- No ties
  (women_wins + men_wins = total_matches) →
  -- Ratio of women's wins to men's wins is 3/2
  (3 * men_wins = 2 * women_wins) →
  -- n equals 4
  n = 4 := by sorry

end NUMINAMATH_CALUDE_tennis_tournament_theorem_l3533_353324


namespace NUMINAMATH_CALUDE_store_sales_growth_rate_l3533_353344

theorem store_sales_growth_rate 
  (initial_sales : ℝ) 
  (final_sales : ℝ) 
  (months : ℕ) 
  (h1 : initial_sales = 20000)
  (h2 : final_sales = 45000)
  (h3 : months = 2) :
  ∃ (growth_rate : ℝ), 
    growth_rate = 0.5 ∧ 
    final_sales = initial_sales * (1 + growth_rate) ^ months :=
sorry

end NUMINAMATH_CALUDE_store_sales_growth_rate_l3533_353344


namespace NUMINAMATH_CALUDE_chess_players_per_game_l3533_353338

/-- The number of combinations of n items taken k at a time -/
def combinations (n k : ℕ) : ℕ := n.choose k

/-- The total number of games played when each player plays each other once -/
def totalGames (n : ℕ) (k : ℕ) : ℕ := combinations n k

theorem chess_players_per_game (n k : ℕ) (h1 : n = 30) (h2 : totalGames n k = 435) : k = 2 := by
  sorry

end NUMINAMATH_CALUDE_chess_players_per_game_l3533_353338


namespace NUMINAMATH_CALUDE_bonus_threshold_correct_l3533_353368

/-- The sales amount that triggers the bonus commission --/
def bonus_threshold : ℝ := 10000

/-- The total sales amount --/
def total_sales : ℝ := 14000

/-- The commission rate on total sales --/
def commission_rate : ℝ := 0.09

/-- The bonus commission rate on excess sales --/
def bonus_rate : ℝ := 0.03

/-- The total commission received --/
def total_commission : ℝ := 1380

/-- The bonus commission received --/
def bonus_commission : ℝ := 120

theorem bonus_threshold_correct :
  commission_rate * total_sales = total_commission - bonus_commission ∧
  bonus_rate * (total_sales - bonus_threshold) = bonus_commission :=
by sorry

end NUMINAMATH_CALUDE_bonus_threshold_correct_l3533_353368


namespace NUMINAMATH_CALUDE_babysitting_earnings_l3533_353386

def payment_cycle : List Nat := [1, 2, 3, 4, 5, 6, 7]

def total_earnings (hours : Nat) : Nat :=
  let full_cycles := hours / payment_cycle.length
  let remaining_hours := hours % payment_cycle.length
  full_cycles * payment_cycle.sum + (payment_cycle.take remaining_hours).sum

theorem babysitting_earnings :
  total_earnings 25 = 94 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_earnings_l3533_353386


namespace NUMINAMATH_CALUDE_range_of_a_l3533_353392

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x - a * x^2 else a^x

theorem range_of_a (a : ℝ) :
  (∀ x y, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/2 : ℝ) 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3533_353392


namespace NUMINAMATH_CALUDE_toby_journey_distance_l3533_353384

def unloaded_speed : ℝ := 20
def loaded_speed : ℝ := 10
def first_loaded_distance : ℝ := 180
def third_loaded_distance : ℝ := 80
def fourth_unloaded_distance : ℝ := 140
def total_time : ℝ := 39

def second_unloaded_distance : ℝ := 120

theorem toby_journey_distance :
  let first_time := first_loaded_distance / loaded_speed
  let third_time := third_loaded_distance / loaded_speed
  let fourth_time := fourth_unloaded_distance / unloaded_speed
  let second_time := second_unloaded_distance / unloaded_speed
  first_time + second_time + third_time + fourth_time = total_time :=
by sorry

end NUMINAMATH_CALUDE_toby_journey_distance_l3533_353384


namespace NUMINAMATH_CALUDE_residue_mod_16_l3533_353335

theorem residue_mod_16 : (198 * 6 - 16 * 8^2 + 5) % 16 = 9 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_16_l3533_353335


namespace NUMINAMATH_CALUDE_work_solution_l3533_353314

def work_problem (a b : ℝ) : Prop :=
  b = 15 ∧
  (3 / a + 5 * (1 / a + 1 / b) = 1) →
  a = 12

theorem work_solution : ∃ a b : ℝ, work_problem a b := by
  sorry

end NUMINAMATH_CALUDE_work_solution_l3533_353314


namespace NUMINAMATH_CALUDE_alex_fourth_test_score_l3533_353307

theorem alex_fourth_test_score :
  ∀ (s1 s2 s3 s4 s5 : ℕ),
  (85 ≤ s1 ∧ s1 ≤ 95) ∧
  (85 ≤ s2 ∧ s2 ≤ 95) ∧
  (85 ≤ s3 ∧ s3 ≤ 95) ∧
  (85 ≤ s4 ∧ s4 ≤ 95) ∧
  (s5 = 90) ∧
  (s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s1 ≠ s5 ∧
   s2 ≠ s3 ∧ s2 ≠ s4 ∧ s2 ≠ s5 ∧
   s3 ≠ s4 ∧ s3 ≠ s5 ∧
   s4 ≠ s5) ∧
  (∃ (k : ℕ), (s1 + s2) = 2 * k) ∧
  (∃ (k : ℕ), (s1 + s2 + s3) = 3 * k) ∧
  (∃ (k : ℕ), (s1 + s2 + s3 + s4) = 4 * k) ∧
  (∃ (k : ℕ), (s1 + s2 + s3 + s4 + s5) = 5 * k) →
  s4 = 95 :=
by sorry

end NUMINAMATH_CALUDE_alex_fourth_test_score_l3533_353307


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l3533_353357

theorem circle_area_from_circumference (c : ℝ) (h : c = 24) :
  let r := c / (2 * Real.pi)
  (Real.pi * r ^ 2) = 144 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l3533_353357


namespace NUMINAMATH_CALUDE_container_capacity_solution_l3533_353362

def container_capacity (replace_volume : ℝ) (num_replacements : ℕ) 
  (final_ratio_original : ℝ) (final_ratio_new : ℝ) : ℝ → Prop :=
  λ C => (C - replace_volume)^num_replacements / C^(num_replacements - 1) = 
    (final_ratio_original / (final_ratio_original + final_ratio_new)) * C

theorem container_capacity_solution :
  ∃ C : ℝ, C > 0 ∧ container_capacity 15 4 81 256 C ∧ 
    C = 15 / (1 - 3 / (337 : ℝ)^(1/4)) :=
by sorry

end NUMINAMATH_CALUDE_container_capacity_solution_l3533_353362


namespace NUMINAMATH_CALUDE_spelling_bee_points_ratio_l3533_353331

/-- Represents the spelling bee problem and proves the ratio of Val's points to Max and Dulce's combined points --/
theorem spelling_bee_points_ratio :
  -- Define the given points
  let max_points : ℕ := 5
  let dulce_points : ℕ := 3
  let opponents_points : ℕ := 40
  let points_behind : ℕ := 16

  -- Define Val's points as a multiple of Max and Dulce's combined points
  let val_points : ℕ → ℕ := λ k ↦ k * (max_points + dulce_points)

  -- Define the total points of Max, Dulce, and Val's team
  let team_total_points : ℕ → ℕ := λ k ↦ max_points + dulce_points + val_points k

  -- State the condition that their team's total points plus the points they're behind equals the opponents' points
  ∀ k : ℕ, team_total_points k + points_behind = opponents_points →

  -- Prove that the ratio of Val's points to Max and Dulce's combined points is 2:1
  ∃ k : ℕ, val_points k = 2 * (max_points + dulce_points) := by
  sorry


end NUMINAMATH_CALUDE_spelling_bee_points_ratio_l3533_353331


namespace NUMINAMATH_CALUDE_krista_drives_six_hours_l3533_353380

/-- Represents a road trip with two drivers -/
structure RoadTrip where
  days : ℕ
  jade_hours_per_day : ℕ
  total_hours : ℕ

/-- Calculates the number of hours Krista drives each day -/
def krista_hours_per_day (trip : RoadTrip) : ℕ :=
  (trip.total_hours - trip.days * trip.jade_hours_per_day) / trip.days

/-- Theorem stating that in the given road trip scenario, Krista drives 6 hours per day -/
theorem krista_drives_six_hours (trip : RoadTrip) 
  (h1 : trip.days = 3)
  (h2 : trip.jade_hours_per_day = 8)
  (h3 : trip.total_hours = 42) :
  krista_hours_per_day trip = 6 := by
  sorry

end NUMINAMATH_CALUDE_krista_drives_six_hours_l3533_353380


namespace NUMINAMATH_CALUDE_max_first_term_l3533_353300

/-- A sequence of positive real numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, 0 < a n) ∧ 
  (∀ n, n > 0 → (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0) ∧
  (a 1 = a 10)

/-- The theorem stating the maximum possible value of the first term -/
theorem max_first_term (a : ℕ → ℝ) (h : SpecialSequence a) : 
  a 1 ≤ 16 := by
  sorry

end NUMINAMATH_CALUDE_max_first_term_l3533_353300


namespace NUMINAMATH_CALUDE_congruence_problem_l3533_353342

theorem congruence_problem (n : ℤ) : 
  0 ≤ n ∧ n < 127 ∧ (126 * n) % 127 = 103 % 127 → n % 127 = 24 % 127 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3533_353342


namespace NUMINAMATH_CALUDE_independence_test_incorrect_judgment_l3533_353316

/-- The chi-squared test statistic -/
def K_squared : ℝ := 4.05

/-- The significance level (α) for the test -/
def significance_level : ℝ := 0.05

/-- The critical value for the chi-squared distribution with 1 degree of freedom at 0.05 significance level -/
def critical_value : ℝ := 3.841

/-- The probability of incorrect judgment in an independence test -/
def probability_incorrect_judgment : ℝ := significance_level

theorem independence_test_incorrect_judgment :
  K_squared > critical_value →
  probability_incorrect_judgment = significance_level :=
sorry

end NUMINAMATH_CALUDE_independence_test_incorrect_judgment_l3533_353316


namespace NUMINAMATH_CALUDE_combined_cost_price_theorem_l3533_353322

/-- Calculates the cost price of a stock given its face value, discount/premium rate, and brokerage rate -/
def stockCostPrice (faceValue : ℝ) (discountRate : ℝ) (brokerageRate : ℝ) : ℝ :=
  let purchasePrice := faceValue * (1 + discountRate)
  let brokerageFee := purchasePrice * brokerageRate
  purchasePrice + brokerageFee

/-- Theorem stating the combined cost price of two stocks -/
theorem combined_cost_price_theorem :
  let stockA := stockCostPrice 100 (-0.02) 0.002
  let stockB := stockCostPrice 100 0.015 0.002
  stockA + stockB = 199.899 := by
  sorry


end NUMINAMATH_CALUDE_combined_cost_price_theorem_l3533_353322


namespace NUMINAMATH_CALUDE_cube_root_sum_theorem_l3533_353318

theorem cube_root_sum_theorem :
  ∃ (x : ℝ), (x^(1/3) + (27 - x)^(1/3) = 3) ∧
  (∀ (y : ℝ), (y^(1/3) + (27 - y)^(1/3) = 3) → x ≤ y) →
  ∃ (r s : ℤ), (x = r - Real.sqrt s) ∧ (r + s = 0) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_sum_theorem_l3533_353318


namespace NUMINAMATH_CALUDE_all_non_negative_l3533_353355

theorem all_non_negative (a b c d : ℤ) (h : (2 : ℝ)^a + (2 : ℝ)^b = (3 : ℝ)^c + (3 : ℝ)^d) :
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_all_non_negative_l3533_353355


namespace NUMINAMATH_CALUDE_no_real_solutions_l3533_353328

theorem no_real_solutions (k : ℝ) : 
  (∀ x : ℝ, x^2 ≠ 5*x + k) ↔ k < -25/4 := by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3533_353328


namespace NUMINAMATH_CALUDE_initial_choir_size_l3533_353313

/-- The number of girls initially in the choir is equal to the sum of blonde-haired and black-haired girls. -/
theorem initial_choir_size (blonde_girls black_girls : ℕ) 
  (h1 : blonde_girls = 30) 
  (h2 : black_girls = 50) : 
  blonde_girls + black_girls = 80 := by
  sorry

end NUMINAMATH_CALUDE_initial_choir_size_l3533_353313


namespace NUMINAMATH_CALUDE_cloth_sold_meters_l3533_353317

/-- Proves that the number of meters of cloth sold is 85 given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sold_meters (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ)
    (h1 : total_selling_price = 8925)
    (h2 : profit_per_meter = 25)
    (h3 : cost_price_per_meter = 80) :
    (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 85 := by
  sorry

#eval (8925 / (80 + 25) : ℕ)  -- Should output 85

end NUMINAMATH_CALUDE_cloth_sold_meters_l3533_353317


namespace NUMINAMATH_CALUDE_largest_sphere_on_torus_l3533_353381

/-- Represents a torus generated by revolving a circle about the z-axis -/
structure Torus where
  inner_radius : ℝ
  outer_radius : ℝ
  circle_center : ℝ × ℝ × ℝ
  circle_radius : ℝ

/-- Represents a sphere centered on the z-axis -/
structure Sphere where
  radius : ℝ
  center : ℝ × ℝ × ℝ

/-- Checks if a sphere touches the horizontal plane -/
def touches_plane (s : Sphere) : Prop :=
  s.center.2.2 = s.radius

/-- Checks if a sphere touches the top of a torus -/
def touches_torus (t : Torus) (s : Sphere) : Prop :=
  (t.circle_center.1 - s.center.1) ^ 2 + (t.circle_center.2.2 - s.center.2.2) ^ 2 = (s.radius + t.circle_radius) ^ 2

theorem largest_sphere_on_torus (t : Torus) (s : Sphere) :
  t.inner_radius = 3 ∧
  t.outer_radius = 5 ∧
  t.circle_center = (4, 0, 1) ∧
  t.circle_radius = 1 ∧
  s.center.1 = 0 ∧
  s.center.2.1 = 0 ∧
  touches_plane s ∧
  touches_torus t s →
  s.radius = 4 :=
sorry

end NUMINAMATH_CALUDE_largest_sphere_on_torus_l3533_353381


namespace NUMINAMATH_CALUDE_complementary_events_l3533_353370

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents the outcome of drawing two balls -/
structure Draw where
  first : Color
  second : Color

/-- The set of all possible draws -/
def all_draws : Finset Draw :=
  sorry

/-- The event "Exactly no red ball" -/
def exactly_no_red (draw : Draw) : Prop :=
  draw.first = Color.White ∧ draw.second = Color.White

/-- The event "At most 1 white ball" -/
def at_most_one_white (draw : Draw) : Prop :=
  draw.first = Color.Red ∨ draw.second = Color.Red

/-- Theorem stating that "Exactly no red ball" and "At most 1 white ball" are complementary events -/
theorem complementary_events :
  ∀ (draw : Draw), draw ∈ all_draws → (exactly_no_red draw ↔ ¬at_most_one_white draw) :=
sorry

end NUMINAMATH_CALUDE_complementary_events_l3533_353370


namespace NUMINAMATH_CALUDE_bamboo_pole_sections_l3533_353387

theorem bamboo_pole_sections (n : ℕ) (a : ℕ → ℝ) : 
  (∀ i j, i < j → j ≤ n → a j - a i = (j - i) * (a 2 - a 1)) →  -- arithmetic sequence
  (a 1 = 10) →  -- top section length
  (a n + a (n-1) + a (n-2) = 114) →  -- last three sections total
  (a 6 ^ 2 = a 1 * a n) →  -- 6th section is geometric mean of first and last
  (n > 6) →
  n = 16 :=
by sorry

end NUMINAMATH_CALUDE_bamboo_pole_sections_l3533_353387


namespace NUMINAMATH_CALUDE_gcd_2197_2209_l3533_353358

theorem gcd_2197_2209 : Nat.gcd 2197 2209 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2197_2209_l3533_353358


namespace NUMINAMATH_CALUDE_odd_terms_sum_l3533_353356

def sequence_sum (n : ℕ) : ℕ := n^2 + 2*n - 1

def arithmetic_sum (first last : ℕ) (step : ℕ) : ℕ :=
  ((last - first) / step + 1) * (first + last) / 2

theorem odd_terms_sum :
  (arithmetic_sum 1 25 2) = 350 :=
by sorry

end NUMINAMATH_CALUDE_odd_terms_sum_l3533_353356


namespace NUMINAMATH_CALUDE_pens_kept_each_l3533_353363

/-- Calculates the number of pens Kendra and Tony keep each after giving some away to friends. -/
theorem pens_kept_each (kendra_packs tony_packs pens_per_pack friends_given : ℕ) :
  kendra_packs = 4 →
  tony_packs = 2 →
  pens_per_pack = 3 →
  friends_given = 14 →
  let total_pens := (kendra_packs + tony_packs) * pens_per_pack
  let remaining_pens := total_pens - friends_given
  remaining_pens / 2 = 2 := by
sorry

end NUMINAMATH_CALUDE_pens_kept_each_l3533_353363


namespace NUMINAMATH_CALUDE_stamp_solution_l3533_353390

/-- Represents the stamp collection and sale problem --/
def stamp_problem (red_count blue_count : ℕ) (red_price blue_price yellow_price : ℚ) (total_goal : ℚ) : Prop :=
  let red_earnings := red_count * red_price
  let blue_earnings := blue_count * blue_price
  let remaining_earnings := total_goal - (red_earnings + blue_earnings)
  ∃ yellow_count : ℕ, yellow_count * yellow_price = remaining_earnings

/-- Theorem stating the solution to the stamp problem --/
theorem stamp_solution :
  stamp_problem 20 80 1.1 0.8 2 100 → ∃ yellow_count : ℕ, yellow_count = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_stamp_solution_l3533_353390


namespace NUMINAMATH_CALUDE_triangle_area_l3533_353310

theorem triangle_area (h : Real) (base : Real) (hypotenuse : Real) :
  h = 10 →
  base = 10 * Real.sqrt 3 →
  hypotenuse = 20 →
  (1 / 2) * base * h = 50 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3533_353310


namespace NUMINAMATH_CALUDE_fraction_increase_l3533_353346

theorem fraction_increase (x y : ℝ) (h : 2*x ≠ 3*y) : 
  (5*x * 5*y) / (2*(5*x) - 3*(5*y)) = 5 * (x*y / (2*x - 3*y)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_increase_l3533_353346


namespace NUMINAMATH_CALUDE_profit_at_45_price_for_1200_profit_l3533_353372

-- Define the parameters of the problem
def cost_price : ℝ := 30
def initial_price : ℝ := 40
def initial_sales : ℝ := 80
def price_change : ℝ := 1
def sales_change : ℝ := 2
def max_price : ℝ := 55

-- Define the sales volume as a function of price
def sales_volume (price : ℝ) : ℝ :=
  initial_sales - sales_change * (price - initial_price)

-- Define the daily profit as a function of price
def daily_profit (price : ℝ) : ℝ :=
  (price - cost_price) * sales_volume price

-- Theorem for part 1
theorem profit_at_45 :
  daily_profit 45 = 1050 := by sorry

-- Theorem for part 2
theorem price_for_1200_profit :
  ∃ (price : ℝ), price ≤ max_price ∧ daily_profit price = 1200 ∧ price = 50 := by sorry

end NUMINAMATH_CALUDE_profit_at_45_price_for_1200_profit_l3533_353372


namespace NUMINAMATH_CALUDE_range_of_a_l3533_353352

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 5 * x + 6 = 0}

-- Theorem statement
theorem range_of_a (a : ℝ) : Set.Nonempty (A a) → a ∈ Set.Iic (25/24) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3533_353352


namespace NUMINAMATH_CALUDE_derivative_x_squared_cos_x_l3533_353373

theorem derivative_x_squared_cos_x :
  let y : ℝ → ℝ := λ x ↦ x^2 * Real.cos x
  deriv y = λ x ↦ 2 * x * Real.cos x - x^2 * Real.sin x := by
sorry

end NUMINAMATH_CALUDE_derivative_x_squared_cos_x_l3533_353373


namespace NUMINAMATH_CALUDE_eighth_term_of_specific_sequence_l3533_353326

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  firstTerm : ℚ
  lastTerm : ℚ
  numTerms : ℕ

/-- Calculates the nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  let commonDiff := (seq.lastTerm - seq.firstTerm) / (seq.numTerms - 1)
  seq.firstTerm + (n - 1) * commonDiff

/-- Theorem: The 8th term of the specified arithmetic sequence is 731/29 -/
theorem eighth_term_of_specific_sequence :
  let seq : ArithmeticSequence := ⟨3, 95, 30⟩
  nthTerm seq 8 = 731 / 29 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_specific_sequence_l3533_353326


namespace NUMINAMATH_CALUDE_truncated_tetrahedron_volume_squared_l3533_353337

/-- A truncated tetrahedron is a solid with 4 triangular faces and 4 hexagonal faces. --/
structure TruncatedTetrahedron where
  side_length : ℝ
  triangular_faces : Fin 4
  hexagonal_faces : Fin 4

/-- The volume of a truncated tetrahedron. --/
noncomputable def volume (t : TruncatedTetrahedron) : ℝ := sorry

/-- Theorem: The square of the volume of a truncated tetrahedron with side length 1 is 529/72. --/
theorem truncated_tetrahedron_volume_squared :
  ∀ (t : TruncatedTetrahedron), t.side_length = 1 → (volume t)^2 = 529/72 := by sorry

end NUMINAMATH_CALUDE_truncated_tetrahedron_volume_squared_l3533_353337


namespace NUMINAMATH_CALUDE_quadratic_passes_through_points_l3533_353340

/-- A quadratic function passing through the points (2,0), (0,4), and (-2,0) -/
def quadratic_function (x : ℝ) : ℝ := -x^2 + 4

/-- Theorem stating that the quadratic function passes through the given points -/
theorem quadratic_passes_through_points :
  (quadratic_function 2 = 0) ∧
  (quadratic_function 0 = 4) ∧
  (quadratic_function (-2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_passes_through_points_l3533_353340


namespace NUMINAMATH_CALUDE_decimalRep_periodic_first_seven_digits_digit_150_l3533_353343

/-- The decimal representation of 17/70 as a sequence of digits after the decimal point -/
def decimalRep : ℕ → ℕ := sorry

/-- The decimal representation of 17/70 is periodic with period 7 -/
theorem decimalRep_periodic : ∀ n : ℕ, decimalRep (n + 7) = decimalRep n := sorry

/-- The first 7 digits of the decimal representation of 17/70 -/
theorem first_seven_digits : 
  (decimalRep 0, decimalRep 1, decimalRep 2, decimalRep 3, decimalRep 4, decimalRep 5, decimalRep 6) 
  = (2, 4, 2, 8, 5, 7, 1) := sorry

/-- The 150th digit after the decimal point in the decimal representation of 17/70 is 2 -/
theorem digit_150 : decimalRep 149 = 2 := sorry

end NUMINAMATH_CALUDE_decimalRep_periodic_first_seven_digits_digit_150_l3533_353343


namespace NUMINAMATH_CALUDE_simplify_absolute_value_l3533_353374

theorem simplify_absolute_value : |-4^2 + (6 - 2)| = 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_absolute_value_l3533_353374


namespace NUMINAMATH_CALUDE_certain_number_proof_l3533_353383

theorem certain_number_proof (x : ℝ) : 
  (0.15 * x - (1/3) * (0.15 * x)) = 18 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3533_353383


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3533_353304

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
   (k - 1) * x^2 - 2 * x + 1 = 0 ∧ 
   (k - 1) * y^2 - 2 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3533_353304
