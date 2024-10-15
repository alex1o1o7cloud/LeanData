import Mathlib

namespace NUMINAMATH_CALUDE_sarah_ate_one_apple_l2094_209480

/-- The number of apples Sarah ate while walking home -/
def apples_eaten (total : ℕ) (to_teachers : ℕ) (to_friends : ℕ) (left : ℕ) : ℕ :=
  total - (to_teachers + to_friends) - left

/-- Theorem stating that Sarah ate 1 apple while walking home -/
theorem sarah_ate_one_apple :
  apples_eaten 25 16 5 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sarah_ate_one_apple_l2094_209480


namespace NUMINAMATH_CALUDE_first_group_count_l2094_209490

theorem first_group_count (avg_first : ℝ) (avg_second : ℝ) (count_second : ℕ) (avg_all : ℝ)
  (h1 : avg_first = 20)
  (h2 : avg_second = 30)
  (h3 : count_second = 20)
  (h4 : avg_all = 24) :
  ∃ (count_first : ℕ), 
    (count_first : ℝ) * avg_first + (count_second : ℝ) * avg_second = 
    (count_first + count_second : ℝ) * avg_all ∧ count_first = 30 := by
  sorry

end NUMINAMATH_CALUDE_first_group_count_l2094_209490


namespace NUMINAMATH_CALUDE_exists_x_sin_minus_x_negative_l2094_209433

open Real

theorem exists_x_sin_minus_x_negative :
  ∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ sin x - x < 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_sin_minus_x_negative_l2094_209433


namespace NUMINAMATH_CALUDE_function_properties_l2094_209402

noncomputable def f (a b x : ℝ) := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem function_properties (a b : ℝ) (h1 : a * b ≠ 0) 
  (h2 : ∀ x : ℝ, f a b x ≤ |f a b (π/6)|) : 
  (f a b (11*π/12) = 0) ∧ 
  (|f a b (7*π/12)| < |f a b (π/5)|) ∧ 
  (∀ x : ℝ, f a b (-x) ≠ f a b x ∧ f a b (-x) ≠ -f a b x) ∧
  (∀ k m : ℝ, ∃ x : ℝ, k * x + m = f a b x) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2094_209402


namespace NUMINAMATH_CALUDE_population_difference_after_two_years_l2094_209461

/-- The difference in population between city A and city C after 2 years -/
def population_difference (A B C : ℝ) : ℝ :=
  A * (1 + 0.03)^2 - C * (1 + 0.02)^2

/-- Theorem stating the difference in population after 2 years -/
theorem population_difference_after_two_years (A B C : ℝ) 
  (h : A + B = B + C + 5000) :
  population_difference A B C = 0.0205 * A + 5202 := by
  sorry

end NUMINAMATH_CALUDE_population_difference_after_two_years_l2094_209461


namespace NUMINAMATH_CALUDE_inclination_angle_range_l2094_209458

/-- The range of inclination angles for a line passing through (1, 1) and (2, m²) -/
theorem inclination_angle_range (m : ℝ) : 
  let α := Real.arctan (m^2 - 1)
  0 ≤ α ∧ α < π/2 ∨ 3*π/4 ≤ α ∧ α < π := by
  sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l2094_209458


namespace NUMINAMATH_CALUDE_gecko_sale_price_l2094_209430

/-- The amount Brandon sold the geckos for -/
def brandon_sale_price : ℝ := 100

/-- The pet store's selling price -/
def pet_store_price (x : ℝ) : ℝ := 3 * x + 5

/-- The pet store's profit -/
def pet_store_profit : ℝ := 205

theorem gecko_sale_price :
  pet_store_price brandon_sale_price - brandon_sale_price = pet_store_profit :=
by sorry

end NUMINAMATH_CALUDE_gecko_sale_price_l2094_209430


namespace NUMINAMATH_CALUDE_chocolate_candy_cost_difference_l2094_209456

/-- The difference in cost between chocolate and candy bar --/
def cost_difference (chocolate_cost candy_cost : ℕ) : ℕ :=
  chocolate_cost - candy_cost

/-- Theorem stating the difference in cost between chocolate and candy bar --/
theorem chocolate_candy_cost_difference :
  let chocolate_cost : ℕ := 7
  let candy_cost : ℕ := 2
  cost_difference chocolate_cost candy_cost = 5 := by
sorry

end NUMINAMATH_CALUDE_chocolate_candy_cost_difference_l2094_209456


namespace NUMINAMATH_CALUDE_min_value_trig_function_l2094_209483

theorem min_value_trig_function (x : ℝ) : 
  Real.sin x ^ 4 + Real.cos x ^ 4 + (1 / Real.cos x) ^ 4 + (1 / Real.sin x) ^ 4 ≥ 8.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_function_l2094_209483


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l2094_209422

theorem quadratic_equation_condition (a : ℝ) : 
  (∀ x, ∃ p q r : ℝ, (a + 4) * x^(a^2 - 14) - 3 * x + 8 = p * x^2 + q * x + r) ∧ 
  (a + 4 ≠ 0) → 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l2094_209422


namespace NUMINAMATH_CALUDE_max_full_books_read_l2094_209431

def pages_per_hour : ℕ := 120
def pages_per_book : ℕ := 360
def available_hours : ℕ := 8

def books_read : ℕ := available_hours * pages_per_hour / pages_per_book

theorem max_full_books_read :
  books_read = 2 :=
sorry

end NUMINAMATH_CALUDE_max_full_books_read_l2094_209431


namespace NUMINAMATH_CALUDE_factorization_of_cubic_l2094_209484

theorem factorization_of_cubic (b : ℝ) : 2*b^3 - 4*b^2 + 2*b = 2*b*(b-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_l2094_209484


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l2094_209432

/-- Given a pentagon STARS where four of its angles are congruent and two of these are equal, 
    prove that the measure of one of these angles is 108°. -/
theorem pentagon_angle_measure (S T A R : ℝ) : 
  (S + T + A + R + S = 540) → -- Sum of angles in a pentagon
  (S = T) → (T = A) → (A = R) → -- Four angles are congruent
  (A = S) → -- Two of these angles are equal
  R = 108 := by sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l2094_209432


namespace NUMINAMATH_CALUDE_parallelogram_area_60_16_l2094_209425

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 60 cm and height 16 cm is 960 square centimeters -/
theorem parallelogram_area_60_16 : 
  parallelogram_area 60 16 = 960 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_60_16_l2094_209425


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_with_geometric_sides_is_square_l2094_209473

/-- A quadrilateral inscribed around a circle with sides in geometric progression is a square -/
theorem inscribed_quadrilateral_with_geometric_sides_is_square
  (R : ℝ) -- radius of the inscribed circle
  (a : ℝ) -- first term of the geometric progression
  (r : ℝ) -- common ratio of the geometric progression
  (h1 : R > 0)
  (h2 : a > 0)
  (h3 : r > 0)
  (h4 : a + a * r^3 = a * r + a * r^2) -- Pitot's theorem
  : 
  r = 1 ∧ -- all sides are equal
  R = a / 2 ∧ -- radius is half the side length
  a^2 = 4 * R^2 -- area of the quadrilateral
  := by sorry

#check inscribed_quadrilateral_with_geometric_sides_is_square

end NUMINAMATH_CALUDE_inscribed_quadrilateral_with_geometric_sides_is_square_l2094_209473


namespace NUMINAMATH_CALUDE_inequality_proof_l2094_209478

-- Define the function f(x) = |x-1|
def f (x : ℝ) : ℝ := |x - 1|

-- State the theorem
theorem inequality_proof (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) (ha_neq_zero : a ≠ 0) :
  f (a * b) / |a| > f (b / a) := by
  sorry


end NUMINAMATH_CALUDE_inequality_proof_l2094_209478


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2094_209467

-- Define the arithmetic sequence
def a (n : ℕ) : ℝ := sorry

-- Define the sum of the first n terms
def S (n : ℕ) : ℝ := sorry

-- Theorem statement
theorem arithmetic_sequence_problem :
  (a 1 + a 2 = 10) ∧ (a 5 = a 3 + 4) →
  (∀ n : ℕ, a n = 2 * n + 2) ∧
  (∃! k : ℕ, k > 0 ∧ S (k + 1) < 2 * a k + a 2 ∧ k = 1) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2094_209467


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2094_209445

theorem sufficient_but_not_necessary (a b : ℝ) :
  (((2 : ℝ) ^ a > (2 : ℝ) ^ b ∧ (2 : ℝ) ^ b > 1) → (a ^ (1/3) > b ^ (1/3))) ∧
  ¬(∀ a b : ℝ, a ^ (1/3) > b ^ (1/3) → ((2 : ℝ) ^ a > (2 : ℝ) ^ b ∧ (2 : ℝ) ^ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2094_209445


namespace NUMINAMATH_CALUDE_triangle_ratio_sum_l2094_209499

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_ratio_sum (t : Triangle) (h : t.B = 60) :
  (t.c / (t.a + t.b)) + (t.a / (t.b + t.c)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_sum_l2094_209499


namespace NUMINAMATH_CALUDE_oil_leak_total_l2094_209460

/-- The total amount of oil leaked from four pipes -/
def total_oil_leaked (pipe_a_before pipe_a_during pipe_b_before pipe_b_during pipe_c_first pipe_c_second pipe_d_first pipe_d_second pipe_d_third : ℕ) : ℕ :=
  pipe_a_before + pipe_a_during + 
  pipe_b_before + pipe_b_during + 
  pipe_c_first + pipe_c_second + 
  pipe_d_first + pipe_d_second + pipe_d_third

/-- Theorem stating the total amount of oil leaked from the four pipes -/
theorem oil_leak_total : 
  total_oil_leaked 6522 5165 4378 3250 2897 7562 1789 3574 5110 = 40247 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_total_l2094_209460


namespace NUMINAMATH_CALUDE_min_tablets_for_both_types_l2094_209408

/-- Given a box with tablets of two types of medicine, this theorem proves
    the minimum number of tablets needed to ensure at least one of each type
    when extracting a specific total number. -/
theorem min_tablets_for_both_types 
  (total_A : ℕ) 
  (total_B : ℕ) 
  (extract_total : ℕ) 
  (h1 : total_A = 10) 
  (h2 : total_B = 16) 
  (h3 : extract_total = 18) :
  extract_total = min (total_A + total_B) extract_total := by
sorry

end NUMINAMATH_CALUDE_min_tablets_for_both_types_l2094_209408


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2094_209463

theorem quadratic_factorization (x : ℝ) : x^2 - 4*x + 4 = (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2094_209463


namespace NUMINAMATH_CALUDE_least_perimeter_of_triangle_l2094_209452

theorem least_perimeter_of_triangle (a b x : ℕ) : 
  a = 33 → b = 42 → x > 0 → 
  x + a > b → x + b > a → a + b > x →
  ∀ y : ℕ, y > 0 → y + a > b → y + b > a → a + b > y → x ≤ y →
  a + b + x = 85 := by
sorry

end NUMINAMATH_CALUDE_least_perimeter_of_triangle_l2094_209452


namespace NUMINAMATH_CALUDE_no_integer_function_satisfies_condition_l2094_209439

theorem no_integer_function_satisfies_condition :
  ¬ ∃ (f : ℤ → ℤ), ∀ (x y : ℤ), f (x + f y) = f x - y :=
by sorry

end NUMINAMATH_CALUDE_no_integer_function_satisfies_condition_l2094_209439


namespace NUMINAMATH_CALUDE_cycle_selling_price_l2094_209434

def cost_price : ℝ := 2800
def loss_percentage : ℝ := 25

theorem cycle_selling_price :
  let loss := (loss_percentage / 100) * cost_price
  let selling_price := cost_price - loss
  selling_price = 2100 := by sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l2094_209434


namespace NUMINAMATH_CALUDE_comparison_of_roots_l2094_209495

theorem comparison_of_roots : 
  let a := (16 : ℝ) ^ (1/4)
  let b := (27 : ℝ) ^ (1/3)
  let c := (25 : ℝ) ^ (1/2)
  let d := (32 : ℝ) ^ (1/5)
  (c > b ∧ b > a ∧ b > d) := by sorry

end NUMINAMATH_CALUDE_comparison_of_roots_l2094_209495


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2094_209449

/-- Given real numbers a, b, and c such that the infinite geometric series
    a/b + a/b^2 + a/b^3 + ... equals 3, prove that the sum of the series
    ca/(a+b) + ca/(a+b)^2 + ca/(a+b)^3 + ... equals 3c/4 -/
theorem geometric_series_sum (a b c : ℝ) 
  (h : ∑' n, a / b^n = 3) : 
  ∑' n, c * a / (a + b)^n = 3/4 * c := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2094_209449


namespace NUMINAMATH_CALUDE_car_cost_difference_l2094_209475

/-- Represents the cost and characteristics of a car --/
structure Car where
  initialCost : ℕ
  fuelConsumption : ℕ
  annualInsurance : ℕ
  annualMaintenance : ℕ
  resaleValue : ℕ

/-- Calculates the total cost of owning a car for 5 years --/
def totalCost (c : Car) (annualDistance : ℕ) (fuelCost : ℕ) (years : ℕ) : ℕ :=
  c.initialCost +
  (annualDistance * c.fuelConsumption * fuelCost * years) / 10000 +
  c.annualInsurance * years +
  c.annualMaintenance * years -
  c.resaleValue

/-- The statement to be proved --/
theorem car_cost_difference :
  let carA : Car := {
    initialCost := 900000,
    fuelConsumption := 9,
    annualInsurance := 35000,
    annualMaintenance := 25000,
    resaleValue := 500000
  }
  let carB : Car := {
    initialCost := 600000,
    fuelConsumption := 10,
    annualInsurance := 32000,
    annualMaintenance := 20000,
    resaleValue := 350000
  }
  let annualDistance := 15000
  let fuelCost := 40
  let years := 5
  totalCost carA annualDistance fuelCost years - totalCost carB annualDistance fuelCost years = 160000 := by
  sorry

end NUMINAMATH_CALUDE_car_cost_difference_l2094_209475


namespace NUMINAMATH_CALUDE_proportional_relationship_l2094_209466

/-- The constant of proportionality -/
def k : ℝ := 3

/-- The functional relationship between y and x -/
def f (x : ℝ) : ℝ := -k * x + 10

theorem proportional_relationship (x y : ℝ) :
  (y + 2 = k * (4 - x)) ∧ (f 3 = 1) →
  (∀ x, f x = -3 * x + 10) ∧
  (∀ y, -2 < y → y < 1 → ∃ x, 3 < x ∧ x < 4 ∧ f x = y) :=
by sorry

end NUMINAMATH_CALUDE_proportional_relationship_l2094_209466


namespace NUMINAMATH_CALUDE_factors_of_1320_l2094_209414

/-- The number of distinct, positive factors of 1320 -/
def num_factors_1320 : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem stating that the number of distinct, positive factors of 1320 is 32 -/
theorem factors_of_1320 : num_factors_1320 = 32 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_1320_l2094_209414


namespace NUMINAMATH_CALUDE_initial_amount_80_leads_to_128_each_l2094_209474

/-- Represents the amount of money each person has at each stage -/
structure Money where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Performs the first transaction where A gives to B and C -/
def transaction1 (m : Money) : Money :=
  { a := m.a - m.b - m.c,
    b := 2 * m.b,
    c := 2 * m.c }

/-- Performs the second transaction where B gives to A and C -/
def transaction2 (m : Money) : Money :=
  { a := 2 * m.a,
    b := m.b - m.a - m.c,
    c := 2 * m.c }

/-- Performs the third transaction where C gives to A and B -/
def transaction3 (m : Money) : Money :=
  { a := 2 * m.a,
    b := 2 * m.b,
    c := m.c - m.a - m.b }

/-- The main theorem stating that if the initial amount for A is 80,
    after all transactions, each person will have 128 cents -/
theorem initial_amount_80_leads_to_128_each (m : Money)
    (h_total : m.a + m.b + m.c = 128 + 128 + 128)
    (h_initial_a : m.a = 80) :
    let m1 := transaction1 m
    let m2 := transaction2 m1
    let m3 := transaction3 m2
    m3.a = 128 ∧ m3.b = 128 ∧ m3.c = 128 := by
  sorry


end NUMINAMATH_CALUDE_initial_amount_80_leads_to_128_each_l2094_209474


namespace NUMINAMATH_CALUDE_gcd_876543_765432_l2094_209448

theorem gcd_876543_765432 : Nat.gcd 876543 765432 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_876543_765432_l2094_209448


namespace NUMINAMATH_CALUDE_jolyn_older_than_clarisse_l2094_209423

/-- Represents an age difference in months and days -/
structure AgeDifference where
  months : ℕ
  days : ℕ

/-- Adds two age differences -/
def addAgeDifference (ad1 ad2 : AgeDifference) : AgeDifference :=
  { months := ad1.months + ad2.months + (ad1.days + ad2.days) / 30,
    days := (ad1.days + ad2.days) % 30 }

/-- Subtracts two age differences -/
def subtractAgeDifference (ad1 ad2 : AgeDifference) : AgeDifference :=
  { months := ad1.months - ad2.months - (if ad1.days < ad2.days then 1 else 0),
    days := if ad1.days < ad2.days then ad1.days + 30 - ad2.days else ad1.days - ad2.days }

theorem jolyn_older_than_clarisse
  (jolyn_therese : AgeDifference)
  (therese_aivo : AgeDifference)
  (leon_aivo : AgeDifference)
  (clarisse_leon : AgeDifference)
  (h1 : jolyn_therese = { months := 2, days := 10 })
  (h2 : therese_aivo = { months := 5, days := 15 })
  (h3 : leon_aivo = { months := 2, days := 25 })
  (h4 : clarisse_leon = { months := 3, days := 20 })
  : subtractAgeDifference (addAgeDifference jolyn_therese therese_aivo)
                          (addAgeDifference clarisse_leon leon_aivo)
    = { months := 1, days := 10 } := by
  sorry


end NUMINAMATH_CALUDE_jolyn_older_than_clarisse_l2094_209423


namespace NUMINAMATH_CALUDE_total_problems_l2094_209440

def daily_record : List Int := [-3, 5, -4, 2, -1, 1, 0, -3, 8, 7]

theorem total_problems (record : List Int) (h : record = daily_record) :
  (List.sum record + 60 : Int) = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_problems_l2094_209440


namespace NUMINAMATH_CALUDE_number_of_red_balls_l2094_209443

theorem number_of_red_balls (blue : ℕ) (green : ℕ) (red : ℕ) : 
  blue = 3 → green = 1 → (red : ℚ) / (blue + green + red : ℚ) = 1 / 2 → red = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_red_balls_l2094_209443


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2094_209464

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  sum_1_5 : a 1 + a 5 = -20
  sum_3_8 : a 3 + a 8 = -10

/-- The general term of the sequence -/
def general_term (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  2 * n - 16

/-- The sum of the first n terms of the sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = general_term seq n) ∧
  (∃ n : ℕ, sum_n_terms seq n = -56 ∧ (n = 7 ∨ n = 8)) ∧
  (∀ m : ℕ, sum_n_terms seq m ≥ -56) :=
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2094_209464


namespace NUMINAMATH_CALUDE_puppies_given_away_l2094_209441

theorem puppies_given_away (initial_puppies : ℝ) (current_puppies : ℕ) : 
  initial_puppies = 6.0 →
  current_puppies = 4 →
  initial_puppies - current_puppies = 2 := by
sorry

end NUMINAMATH_CALUDE_puppies_given_away_l2094_209441


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l2094_209492

theorem divisibility_of_expression (n : ℕ) (h : Odd n) (h' : n > 0) :
  ∃ k : ℤ, n^4 - n^2 - n = n * k :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l2094_209492


namespace NUMINAMATH_CALUDE_inequality_solution_l2094_209481

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 1) + 10 / (x + 4) ≥ 3 / (x + 2)) ↔ 
  (x ∈ Set.Ioc (-4) (-1) ∪ Set.Ioi (-4/3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2094_209481


namespace NUMINAMATH_CALUDE_company_size_proof_l2094_209442

/-- The total number of employees in the company -/
def total_employees : ℕ := 100

/-- The number of employees in group C -/
def group_C_employees : ℕ := 10

/-- The ratio of employees in levels A:B:C -/
def employee_ratio : Fin 3 → ℕ
| 0 => 5  -- Level A
| 1 => 4  -- Level B
| 2 => 1  -- Level C

/-- The size of the stratified sample -/
def sample_size : ℕ := 20

/-- The probability of selecting both people from group C in the sample -/
def prob_both_from_C : ℚ := 1 / 45

theorem company_size_proof :
  (total_employees = 100) ∧
  (group_C_employees = 10) ∧
  (∀ i : Fin 3, employee_ratio i = [5, 4, 1].get i) ∧
  (sample_size = 20) ∧
  (prob_both_from_C = 1 / 45) ∧
  (group_C_employees.choose 2 = prob_both_from_C * total_employees.choose 2) ∧
  (group_C_employees * (employee_ratio 0 + employee_ratio 1 + employee_ratio 2) = total_employees) :=
by sorry

#check company_size_proof

end NUMINAMATH_CALUDE_company_size_proof_l2094_209442


namespace NUMINAMATH_CALUDE_sin_sum_inverse_sin_tan_l2094_209437

theorem sin_sum_inverse_sin_tan : 
  Real.sin (Real.arcsin (4/5) + Real.arctan 3) = 13 * Real.sqrt 10 / 50 := by
sorry

end NUMINAMATH_CALUDE_sin_sum_inverse_sin_tan_l2094_209437


namespace NUMINAMATH_CALUDE_odd_divisibility_l2094_209407

def sum_of_powers (n : ℕ) : ℕ := (Finset.range (n - 1)).sum (λ k => k^n)

theorem odd_divisibility (n : ℕ) (h : n > 1) :
  n ∣ sum_of_powers n ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_odd_divisibility_l2094_209407


namespace NUMINAMATH_CALUDE_min_sum_power_mod_l2094_209472

theorem min_sum_power_mod (m n : ℕ) : 
  n > m → 
  m > 1 → 
  (1978^m) % 1000 = (1978^n) % 1000 → 
  ∃ (m₀ n₀ : ℕ), m₀ + n₀ = 106 ∧ 
    ∀ (m' n' : ℕ), n' > m' → m' > 1 → 
      (1978^m') % 1000 = (1978^n') % 1000 → 
      m' + n' ≥ m₀ + n₀ :=
by sorry

end NUMINAMATH_CALUDE_min_sum_power_mod_l2094_209472


namespace NUMINAMATH_CALUDE_min_value_expressions_l2094_209450

theorem min_value_expressions (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (max x (1/y) + max y (2/x) ≥ 2 * Real.sqrt 2) ∧
  (max x (1/y) + max y (2/z) + max z (3/x) ≥ 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expressions_l2094_209450


namespace NUMINAMATH_CALUDE_area_comparison_l2094_209405

-- Define a polygon as a list of points in 2D space
def Polygon := List (Real × Real)

-- Function to calculate the area of a polygon
noncomputable def area (p : Polygon) : Real := sorry

-- Function to check if a polygon is convex
def isConvex (p : Polygon) : Prop := sorry

-- Function to check if two polygons have equal corresponding sides
def equalSides (p1 p2 : Polygon) : Prop := sorry

-- Function to check if a polygon is inscribed in a circle
def isInscribed (p : Polygon) : Prop := sorry

-- Theorem statement
theorem area_comparison 
  (A B : Polygon) 
  (h1 : isConvex A) 
  (h2 : isConvex B) 
  (h3 : equalSides A B) 
  (h4 : isInscribed B) : 
  area B ≥ area A := by sorry

end NUMINAMATH_CALUDE_area_comparison_l2094_209405


namespace NUMINAMATH_CALUDE_remainder_2519_div_7_l2094_209436

theorem remainder_2519_div_7 : 2519 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2519_div_7_l2094_209436


namespace NUMINAMATH_CALUDE_manager_percentage_l2094_209470

theorem manager_percentage (total_employees : ℕ) (initial_percentage : ℚ) 
  (managers_leaving : ℚ) (final_percentage : ℚ) : 
  total_employees = 300 →
  initial_percentage = 99/100 →
  managers_leaving = 149.99999999999986 →
  final_percentage = 49/100 →
  (↑total_employees * initial_percentage - managers_leaving) / ↑total_employees = final_percentage :=
by sorry

end NUMINAMATH_CALUDE_manager_percentage_l2094_209470


namespace NUMINAMATH_CALUDE_total_travel_time_l2094_209482

def luke_bus_time : ℕ := 70
def paula_bus_time : ℕ := (3 * luke_bus_time) / 5
def jane_train_time : ℕ := 120
def michael_cycle_time : ℕ := jane_train_time / 4

def luke_total_time : ℕ := luke_bus_time + 5 * luke_bus_time
def paula_total_time : ℕ := 2 * paula_bus_time
def jane_total_time : ℕ := jane_train_time + 2 * jane_train_time
def michael_total_time : ℕ := 2 * michael_cycle_time

theorem total_travel_time :
  luke_total_time + paula_total_time + jane_total_time + michael_total_time = 924 :=
by sorry

end NUMINAMATH_CALUDE_total_travel_time_l2094_209482


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2094_209447

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_increasing : ∀ x y, x < y → f x < f y)
variable (h_f_0 : f 0 = -1)
variable (h_f_3 : f 3 = 1)

-- Define the solution set
def solution_set := {x : ℝ | |f (x + 1)| < 1}

-- State the theorem
theorem solution_set_equivalence : 
  solution_set f = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2094_209447


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l2094_209415

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 243 ways to put 5 distinguishable balls in 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l2094_209415


namespace NUMINAMATH_CALUDE_walnut_price_l2094_209494

/-- Represents the price of a nut in Forints -/
structure NutPrice where
  price : ℕ
  is_two_digit : price ≥ 10 ∧ price < 100

/-- Checks if two digits are consecutive -/
def consecutive_digits (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ (a = b + 1 ∨ b = a + 1) ∧ n = 10 * a + b

/-- Checks if two prices are digit swaps of each other -/
def is_digit_swap (p1 p2 : NutPrice) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ p1.price = 10 * a + b ∧ p2.price = 10 * b + a

theorem walnut_price (walnut hazelnut : NutPrice)
  (total_value : ℕ) (total_weight : ℕ)
  (h1 : total_value = 1978)
  (h2 : total_weight = 55)
  (h3 : walnut.price > hazelnut.price)
  (h4 : is_digit_swap walnut hazelnut)
  (h5 : consecutive_digits walnut.price)
  (h6 : ∃ (w h : ℕ), w * walnut.price + h * hazelnut.price = total_value ∧ w + h = total_weight) :
  walnut.price = 43 := by
  sorry

end NUMINAMATH_CALUDE_walnut_price_l2094_209494


namespace NUMINAMATH_CALUDE_expression_value_l2094_209468

theorem expression_value (x y : ℝ) (h1 : x + y = 4) (h2 : x * y = -2) :
  ∃ ε > 0, |x + x^3/y^2 + y^3/x^2 + y - 440| < ε :=
sorry

end NUMINAMATH_CALUDE_expression_value_l2094_209468


namespace NUMINAMATH_CALUDE_max_log_product_l2094_209413

theorem max_log_product (x y : ℝ) (hx : x > 1) (hy : y > 1) (h_sum : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) :
  (Real.log x / Real.log 10) * (Real.log y / Real.log 10) ≤ 4 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 1 ∧ y₀ > 1 ∧
    Real.log x₀ / Real.log 10 + Real.log y₀ / Real.log 10 = 4 ∧
    (Real.log x₀ / Real.log 10) * (Real.log y₀ / Real.log 10) = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_log_product_l2094_209413


namespace NUMINAMATH_CALUDE_soccer_team_enrollment_l2094_209488

theorem soccer_team_enrollment (total : ℕ) (physics : ℕ) (both : ℕ) (mathematics : ℕ)
  (h1 : total = 15)
  (h2 : physics = 9)
  (h3 : both = 3)
  (h4 : physics + mathematics - both = total) :
  mathematics = 9 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_enrollment_l2094_209488


namespace NUMINAMATH_CALUDE_prob_at_least_six_heads_in_eight_flips_prob_at_least_six_heads_in_eight_flips_proof_l2094_209403

/-- The probability of getting at least 6 heads in 8 fair coin flips -/
theorem prob_at_least_six_heads_in_eight_flips : ℚ :=
  37 / 256

/-- Proof that the probability of getting at least 6 heads in 8 fair coin flips is 37/256 -/
theorem prob_at_least_six_heads_in_eight_flips_proof :
  prob_at_least_six_heads_in_eight_flips = 37 / 256 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_six_heads_in_eight_flips_prob_at_least_six_heads_in_eight_flips_proof_l2094_209403


namespace NUMINAMATH_CALUDE_power_inequality_l2094_209491

theorem power_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^2 > x + y) (h2 : x^4 > x^3 + y) : x^3 > x^2 + y := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2094_209491


namespace NUMINAMATH_CALUDE_sticker_distribution_l2094_209457

theorem sticker_distribution (gold : ℕ) (students : ℕ) : 
  gold = 50 →
  students = 5 →
  (gold + 2 * gold + (2 * gold - 20)) / students = 46 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2094_209457


namespace NUMINAMATH_CALUDE_division_sum_theorem_l2094_209496

theorem division_sum_theorem (quotient divisor remainder : ℕ) 
  (h_quotient : quotient = 40)
  (h_divisor : divisor = 72)
  (h_remainder : remainder = 64) :
  divisor * quotient + remainder = 2944 :=
by sorry

end NUMINAMATH_CALUDE_division_sum_theorem_l2094_209496


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2094_209418

/-- An isosceles triangle with sides 4 and 6 has a perimeter of either 14 or 16. -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  (a = 4 ∧ b = 4 ∧ c = 6) ∨ (a = 4 ∧ b = 6 ∧ c = 6) →  -- possible configurations
  a + b > c ∧ b + c > a ∧ c + a > b →  -- triangle inequality
  a + b + c = 14 ∨ a + b + c = 16 :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2094_209418


namespace NUMINAMATH_CALUDE_quadratic_symmetry_solution_set_l2094_209479

theorem quadratic_symmetry_solution_set 
  (a b c m n p : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (hm : m ≠ 0) 
  (hn : n ≠ 0) 
  (hp : p ≠ 0) : 
  let f := fun (x : ℝ) ↦ a * x^2 + b * x + c
  let solution_set := {x : ℝ | m * (f x)^2 + n * (f x) + p = 0}
  solution_set ≠ {1, 4, 16, 64} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_solution_set_l2094_209479


namespace NUMINAMATH_CALUDE_equation_solution_l2094_209455

theorem equation_solution : ∃ x : ℚ, x ≠ 1 ∧ (x^2 - 2*x + 3) / (x - 1) = x + 4 ∧ x = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2094_209455


namespace NUMINAMATH_CALUDE_polyhedron_has_triangle_l2094_209427

/-- A polyhedron with edges of non-increasing lengths -/
structure Polyhedron where
  n : ℕ
  edges : Fin n → ℝ
  edges_decreasing : ∀ i j, i ≤ j → edges i ≥ edges j

/-- Three edges can form a triangle if the sum of any two is greater than the third -/
def CanFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- In any polyhedron, there exist three edges that can form a triangle -/
theorem polyhedron_has_triangle (P : Polyhedron) :
  ∃ i j k, i < j ∧ j < k ∧ CanFormTriangle (P.edges i) (P.edges j) (P.edges k) := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_has_triangle_l2094_209427


namespace NUMINAMATH_CALUDE_rosy_fish_count_l2094_209465

/-- Given that Lilly has 10 fish and the total number of fish is 22,
    prove that Rosy has 12 fish. -/
theorem rosy_fish_count (lilly_fish : ℕ) (total_fish : ℕ) (h1 : lilly_fish = 10) (h2 : total_fish = 22) :
  total_fish - lilly_fish = 12 := by
  sorry

end NUMINAMATH_CALUDE_rosy_fish_count_l2094_209465


namespace NUMINAMATH_CALUDE_x_factor_change_l2094_209476

/-- Given a function q defined in terms of e, x, and z, prove that when e is quadrupled,
    z is tripled, and q is multiplied by 0.2222222222222222, x is doubled. -/
theorem x_factor_change (e x z : ℝ) (h : x ≠ 0) (hz : z ≠ 0) :
  let q := 5 * e / (4 * x * z^2)
  let q' := 0.2222222222222222 * (5 * (4 * e) / (4 * x * (3 * z)^2))
  ∃ x' : ℝ, x' = 2 * x ∧ q' = 5 * (4 * e) / (4 * x' * (3 * z)^2) :=
by sorry

end NUMINAMATH_CALUDE_x_factor_change_l2094_209476


namespace NUMINAMATH_CALUDE_probability_of_prime_sum_two_dice_l2094_209410

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Bool := sorry

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The set of possible sums when rolling two dice -/
def possibleSums : Finset ℕ := sorry

/-- The set of prime sums when rolling two dice -/
def primeSums : Finset ℕ := sorry

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numSides * numSides

theorem probability_of_prime_sum_two_dice :
  (Finset.card primeSums : ℚ) / totalOutcomes = 23 / 64 := by sorry

end NUMINAMATH_CALUDE_probability_of_prime_sum_two_dice_l2094_209410


namespace NUMINAMATH_CALUDE_product_ab_l2094_209428

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_ab_l2094_209428


namespace NUMINAMATH_CALUDE_triangle_area_13_13_24_l2094_209453

/-- The area of a triangle with side lengths 13, 13, and 24 is 60 square units. -/
theorem triangle_area_13_13_24 : ∃ (A : ℝ), 
  A = (1/2) * 24 * Real.sqrt (13^2 - 12^2) ∧ A = 60 := by sorry

end NUMINAMATH_CALUDE_triangle_area_13_13_24_l2094_209453


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l2094_209438

theorem fraction_sum_simplification :
  8 / 19 - 5 / 57 + 1 / 3 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l2094_209438


namespace NUMINAMATH_CALUDE_keith_attended_games_l2094_209416

def total_games : ℕ := 8
def missed_games : ℕ := 4

theorem keith_attended_games :
  total_games - missed_games = 4 := by sorry

end NUMINAMATH_CALUDE_keith_attended_games_l2094_209416


namespace NUMINAMATH_CALUDE_min_colors_is_23_l2094_209444

/-- A coloring scheme for boxes of balls -/
structure ColoringScheme where
  n : ℕ  -- number of colors
  boxes : Fin 8 → Fin 6 → Fin n  -- coloring function

/-- Predicate to check if a coloring scheme is valid -/
def is_valid_coloring (c : ColoringScheme) : Prop :=
  -- No two balls in the same box have the same color
  (∀ i : Fin 8, ∀ j k : Fin 6, j ≠ k → c.boxes i j ≠ c.boxes i k) ∧
  -- No two colors occur together in more than one box
  (∀ i j : Fin 8, i ≠ j → ∀ c1 c2 : Fin c.n, c1 ≠ c2 →
    (∃ k : Fin 6, c.boxes i k = c1 ∧ ∃ l : Fin 6, c.boxes i l = c2) →
    ¬(∃ m : Fin 6, c.boxes j m = c1 ∧ ∃ n : Fin 6, c.boxes j n = c2))

/-- The main theorem: the minimum number of colors is 23 -/
theorem min_colors_is_23 :
  (∃ c : ColoringScheme, c.n = 23 ∧ is_valid_coloring c) ∧
  (∀ c : ColoringScheme, c.n < 23 → ¬is_valid_coloring c) := by
  sorry

end NUMINAMATH_CALUDE_min_colors_is_23_l2094_209444


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2094_209486

theorem complex_equation_sum (x y : ℝ) : (2*x - y : ℂ) + (x + 3)*I = 0 → x + y = -9 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2094_209486


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt2_over_2_l2094_209424

theorem cos_sin_sum_equals_sqrt2_over_2 : 
  Real.cos (80 * π / 180) * Real.cos (35 * π / 180) + 
  Real.sin (80 * π / 180) * Real.cos (55 * π / 180) = 
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt2_over_2_l2094_209424


namespace NUMINAMATH_CALUDE_quadratic_function_bounds_l2094_209469

/-- Given a quadratic function f(x) = ax² + bx with certain constraints on f(-1) and f(1),
    prove that f(-2) is bounded between 6 and 10. -/
theorem quadratic_function_bounds (a b : ℝ) :
  let f := fun (x : ℝ) => a * x^2 + b * x
  (1 ≤ f (-1) ∧ f (-1) ≤ 2) →
  (3 ≤ f 1 ∧ f 1 ≤ 4) →
  (6 ≤ f (-2) ∧ f (-2) ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_bounds_l2094_209469


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l2094_209459

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a < b →            -- a is the shorter leg
  a ≤ b →            -- Ensure a is not equal to b
  a = 16 :=          -- The shorter leg is 16 units
by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l2094_209459


namespace NUMINAMATH_CALUDE_line_inclination_trig_identity_l2094_209487

/-- Given a line with equation x - 2y + 1 = 0 and inclination angle α, 
    prove that cos²α + sin(2α) = 8/5 -/
theorem line_inclination_trig_identity (α : ℝ) : 
  (∃ x y : ℝ, x - 2*y + 1 = 0 ∧ Real.tan α = 1/2) → 
  Real.cos α ^ 2 + Real.sin (2 * α) = 8/5 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_trig_identity_l2094_209487


namespace NUMINAMATH_CALUDE_equation_has_solution_in_interval_l2094_209400

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 - 3*x + 5

-- State the theorem
theorem equation_has_solution_in_interval :
  (Continuous f) → ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_equation_has_solution_in_interval_l2094_209400


namespace NUMINAMATH_CALUDE_three_lines_intersection_l2094_209435

/-- The curve (x + 2y + a)(x^2 - y^2) = 0 represents three lines intersecting at a single point if and only if a = 0 -/
theorem three_lines_intersection (a : ℝ) : 
  (∃! p : ℝ × ℝ, ∀ x y : ℝ, (x + 2*y + a)*(x^2 - y^2) = 0 ↔ 
    (x = p.1 ∧ y = p.2) ∨ (x = -y ∧ x = p.1) ∨ (x = y ∧ x = p.1)) ↔ 
  a = 0 :=
sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l2094_209435


namespace NUMINAMATH_CALUDE_max_uncovered_corridor_length_l2094_209485

theorem max_uncovered_corridor_length 
  (corridor_length : ℝ) 
  (num_rugs : ℕ) 
  (total_rug_length : ℝ) 
  (h1 : corridor_length = 100)
  (h2 : num_rugs = 20)
  (h3 : total_rug_length = 1000) :
  (corridor_length - (total_rug_length - corridor_length)) ≤ 50 := by
sorry

end NUMINAMATH_CALUDE_max_uncovered_corridor_length_l2094_209485


namespace NUMINAMATH_CALUDE_intersection_point_l2094_209477

theorem intersection_point (x y : ℚ) : 
  (5 * x - 3 * y = 8) ∧ (4 * x + 2 * y = 20) ↔ x = 38/11 ∧ y = 34/11 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l2094_209477


namespace NUMINAMATH_CALUDE_candy_distribution_proof_l2094_209489

/-- Given a number of candy pieces and sisters, returns the minimum number of pieces to remove for equal distribution. -/
def minPiecesToRemove (totalPieces sisters : ℕ) : ℕ :=
  totalPieces % sisters

theorem candy_distribution_proof :
  minPiecesToRemove 20 3 = 2 := by
  sorry

#eval minPiecesToRemove 20 3

end NUMINAMATH_CALUDE_candy_distribution_proof_l2094_209489


namespace NUMINAMATH_CALUDE_custom_operation_theorem_l2094_209421

-- Define the custom operation *
def star (a b : ℝ) : ℝ := (a + b)^2

-- State the theorem
theorem custom_operation_theorem (x y : ℝ) : 
  star ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 := by sorry

end NUMINAMATH_CALUDE_custom_operation_theorem_l2094_209421


namespace NUMINAMATH_CALUDE_pencils_per_row_l2094_209493

theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (h1 : total_pencils = 30) (h2 : num_rows = 6) :
  total_pencils / num_rows = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l2094_209493


namespace NUMINAMATH_CALUDE_wire_parts_used_l2094_209401

theorem wire_parts_used (total_length : ℝ) (total_parts : ℕ) (unused_length : ℝ) : 
  total_length = 50 →
  total_parts = 5 →
  unused_length = 20 →
  (total_parts : ℝ) - (unused_length / (total_length / total_parts)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_wire_parts_used_l2094_209401


namespace NUMINAMATH_CALUDE_expression_simplification_l2094_209420

theorem expression_simplification (a₁ a₂ a₃ a₄ : ℝ) :
  1 + a₁ / (1 - a₁) + a₂ / ((1 - a₁) * (1 - a₂)) + 
  a₃ / ((1 - a₁) * (1 - a₂) * (1 - a₃)) + 
  (a₄ - a₁) / ((1 - a₁) * (1 - a₂) * (1 - a₃) * (1 - a₄)) = 
  1 / ((1 - a₂) * (1 - a₃) * (1 - a₄)) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2094_209420


namespace NUMINAMATH_CALUDE_jessica_letter_paper_weight_l2094_209497

/-- The weight of each piece of paper in Jessica's letter -/
def paper_weight (num_papers : ℕ) (envelope_weight total_weight : ℚ) : ℚ :=
  (total_weight - envelope_weight) / num_papers

/-- Theorem stating that each piece of paper in Jessica's letter weighs 1/5 of an ounce -/
theorem jessica_letter_paper_weight :
  let num_papers : ℕ := 8
  let envelope_weight : ℚ := 2/5
  let total_weight : ℚ := 2
  paper_weight num_papers envelope_weight total_weight = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_jessica_letter_paper_weight_l2094_209497


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2094_209412

theorem quadratic_inequality_solution (a : ℝ) (x : ℝ) :
  a * x^2 - 2 ≥ 2 * x - a * x ↔
    (a = 0 ∧ x ≤ -1) ∨
    (a > 0 ∧ (x ≥ 2 / a ∨ x ≤ -1)) ∨
    (-2 < a ∧ a < 0 ∧ 2 / a ≤ x ∧ x ≤ -1) ∨
    (a = -2 ∧ x = -1) ∨
    (a < -2 ∧ -1 ≤ x ∧ x ≤ 2 / a) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2094_209412


namespace NUMINAMATH_CALUDE_horner_method_v3_l2094_209454

def horner_polynomial (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v1 (x : ℝ) : ℝ := 3*x + 5

def horner_v2 (x : ℝ) : ℝ := horner_v1 x * x + 6

def horner_v3 (x : ℝ) : ℝ := horner_v2 x * x + 79

theorem horner_method_v3 :
  horner_v3 (-4) = -57 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l2094_209454


namespace NUMINAMATH_CALUDE_jerichos_remaining_money_l2094_209406

def jerichos_money_problem (jerichos_money : ℚ) (debt_to_annika : ℚ) : Prop :=
  2 * jerichos_money = 60 ∧
  debt_to_annika = 14 ∧
  let debt_to_manny := debt_to_annika / 2
  let remaining_money := jerichos_money - debt_to_annika - debt_to_manny
  remaining_money = 9

theorem jerichos_remaining_money :
  ∀ (jerichos_money : ℚ) (debt_to_annika : ℚ),
  jerichos_money_problem jerichos_money debt_to_annika :=
by
  sorry

end NUMINAMATH_CALUDE_jerichos_remaining_money_l2094_209406


namespace NUMINAMATH_CALUDE_a_greater_than_b_l2094_209462

theorem a_greater_than_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l2094_209462


namespace NUMINAMATH_CALUDE_line_satisfies_conditions_l2094_209409

theorem line_satisfies_conditions : ∃! k : ℝ,
  let f (x : ℝ) := x^2 + 8*x + 7
  let g (x : ℝ) := 19.5*x - 32
  let p1 := (k, f k)
  let p2 := (k, g k)
  (g 2 = 7) ∧
  (abs (f k - g k) = 6) ∧
  (-32 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_line_satisfies_conditions_l2094_209409


namespace NUMINAMATH_CALUDE_fourth_vertex_of_parallelogram_l2094_209417

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Addition of a point and a vector -/
def Point2D.add (p : Point2D) (v : Vector2D) : Point2D :=
  ⟨p.x + v.x, p.y + v.y⟩

/-- Subtraction of two points to get a vector -/
def Point2D.sub (p q : Point2D) : Vector2D :=
  ⟨p.x - q.x, p.y - q.y⟩

/-- The given points of the parallelogram -/
def Q : Point2D := ⟨1, -1⟩
def R : Point2D := ⟨-1, 0⟩
def S : Point2D := ⟨0, 1⟩

/-- The theorem stating that the fourth vertex of the parallelogram is (-2, 2) -/
theorem fourth_vertex_of_parallelogram :
  let V := S.add (R.sub Q)
  V = Point2D.mk (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_vertex_of_parallelogram_l2094_209417


namespace NUMINAMATH_CALUDE_graph_translation_l2094_209446

/-- Translating the graph of f(x) = cos(2x - π/3) to the left by π/6 units results in y = cos(2x) -/
theorem graph_translation (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.cos (2 * x - π / 3)
  let g : ℝ → ℝ := λ x => Real.cos (2 * x)
  let h : ℝ → ℝ := λ x => f (x + π / 6)
  h x = g x := by sorry

end NUMINAMATH_CALUDE_graph_translation_l2094_209446


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l2094_209451

/-- Calculate the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Theorem stating that the loss percentage for a radio with
    cost price 1500 and selling price 1290 is 14% -/
theorem radio_loss_percentage :
  loss_percentage 1500 1290 = 14 := by sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l2094_209451


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2094_209426

theorem quadratic_equation_properties (x y : ℝ) 
  (h : (x - y)^2 - 2*(x + y) + 1 = 0) : 
  (x ≥ 0 ∧ y ≥ 0) ∧ 
  (x > 1 ∧ y < x → Real.sqrt x - Real.sqrt y = 1) ∧
  (x < 1 ∧ y < 1 → Real.sqrt x + Real.sqrt y = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2094_209426


namespace NUMINAMATH_CALUDE_correct_average_weight_l2094_209411

/-- Proves that the correct average weight of a class is 61.2 kg given the initial miscalculation and corrections. -/
theorem correct_average_weight 
  (num_students : ℕ) 
  (initial_average : ℝ)
  (student_A_misread student_A_correct : ℝ)
  (student_B_misread student_B_correct : ℝ)
  (student_C_misread student_C_correct : ℝ)
  (h1 : num_students = 30)
  (h2 : initial_average = 60.2)
  (h3 : student_A_misread = 54)
  (h4 : student_A_correct = 64)
  (h5 : student_B_misread = 58)
  (h6 : student_B_correct = 68)
  (h7 : student_C_misread = 50)
  (h8 : student_C_correct = 60) :
  (num_students : ℝ) * initial_average + 
  (student_A_correct - student_A_misread) + 
  (student_B_correct - student_B_misread) + 
  (student_C_correct - student_C_misread) / num_students = 61.2 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_weight_l2094_209411


namespace NUMINAMATH_CALUDE_book_choice_theorem_l2094_209471

/-- The number of ways to choose 1 book from different sets of books -/
def choose_one_book (diff_lit : Nat) (diff_math : Nat) (id_lit : Nat) (id_math : Nat) : Nat :=
  if diff_lit > 0 ∧ diff_math > 0 then
    diff_lit + diff_math
  else if id_lit > 0 ∧ id_math > 0 then
    (if diff_math > 0 then diff_math + 1 else 2)
  else
    0

/-- Theorem stating the number of ways to choose 1 book in different scenarios -/
theorem book_choice_theorem :
  (choose_one_book 5 4 0 0 = 9) ∧
  (choose_one_book 0 0 5 4 = 5) ∧
  (choose_one_book 0 4 5 0 = 2) := by
  sorry

end NUMINAMATH_CALUDE_book_choice_theorem_l2094_209471


namespace NUMINAMATH_CALUDE_decimal_sum_and_subtraction_l2094_209419

theorem decimal_sum_and_subtraction :
  (0.5 + 0.003 + 0.070) - 0.008 = 0.565 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_and_subtraction_l2094_209419


namespace NUMINAMATH_CALUDE_concentrate_water_ratio_is_one_to_three_l2094_209498

/-- The ratio of concentrate to water for orange juice -/
def concentrate_to_water_ratio : ℚ := 1 / 3

/-- The number of cans of concentrate used -/
def concentrate_cans : ℕ := 40

/-- The number of cans of water per can of concentrate -/
def water_cans_per_concentrate : ℕ := 3

/-- Theorem: The ratio of cans of concentrate to cans of water is 1:3 -/
theorem concentrate_water_ratio_is_one_to_three :
  concentrate_to_water_ratio = 1 / (water_cans_per_concentrate : ℚ) ∧
  concentrate_to_water_ratio = (concentrate_cans : ℚ) / ((water_cans_per_concentrate * concentrate_cans) : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_concentrate_water_ratio_is_one_to_three_l2094_209498


namespace NUMINAMATH_CALUDE_line_AB_equation_l2094_209429

-- Define the ellipses
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def C₂ (x y : ℝ) : Prop := y^2 / 16 + x^2 / 4 = 1

-- Define points A and B
def A : ℝ × ℝ → Prop := λ p => C₁ p.1 p.2
def B : ℝ × ℝ → Prop := λ p => C₂ p.1 p.2

-- Define the relation between OA and OB
def OB_eq_2OA (a b : ℝ × ℝ) : Prop := b.1 = 2 * a.1 ∧ b.2 = 2 * a.2

-- Theorem statement
theorem line_AB_equation (a b : ℝ × ℝ) (ha : A a) (hb : B b) (hab : OB_eq_2OA a b) :
  (b.2 - a.2) / (b.1 - a.1) = 1 ∨ (b.2 - a.2) / (b.1 - a.1) = -1 :=
sorry

end NUMINAMATH_CALUDE_line_AB_equation_l2094_209429


namespace NUMINAMATH_CALUDE_students_suggesting_bacon_l2094_209404

theorem students_suggesting_bacon (total : ℕ) (mashed_potatoes : ℕ) (tomatoes : ℕ) 
  (h1 : total = 826)
  (h2 : mashed_potatoes = 324)
  (h3 : tomatoes = 128) :
  total - (mashed_potatoes + tomatoes) = 374 := by
  sorry

end NUMINAMATH_CALUDE_students_suggesting_bacon_l2094_209404
