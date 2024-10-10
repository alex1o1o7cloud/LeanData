import Mathlib

namespace even_count_pascal_15_rows_l4092_409225

/-- Counts the number of even entries in a single row of Pascal's Triangle -/
def countEvenInRow (n : ℕ) : ℕ := sorry

/-- Counts the total number of even entries in the first n rows of Pascal's Triangle -/
def countEvenInTriangle (n : ℕ) : ℕ := sorry

/-- The number of even integers in the first 15 rows of Pascal's Triangle is 97 -/
theorem even_count_pascal_15_rows : countEvenInTriangle 15 = 97 := by sorry

end even_count_pascal_15_rows_l4092_409225


namespace smallest_other_integer_l4092_409295

theorem smallest_other_integer (m n x : ℕ) : 
  m > 0 → n > 0 → x > 0 →
  Nat.gcd m n = x + 6 →
  Nat.lcm m n = x * (x + 6) →
  m = 60 →
  (∀ k : ℕ, k > 0 ∧ k < n → 
    (Nat.gcd 60 k ≠ x + 6 ∨ Nat.lcm 60 k ≠ x * (x + 6))) →
  n = 93 := by
sorry

end smallest_other_integer_l4092_409295


namespace misread_weight_l4092_409271

theorem misread_weight (class_size : ℕ) (incorrect_avg : ℚ) (correct_avg : ℚ) (correct_weight : ℚ) (x : ℚ) :
  class_size = 20 →
  incorrect_avg = 58.4 →
  correct_avg = 58.85 →
  correct_weight = 65 →
  class_size * correct_avg = class_size * incorrect_avg - x + correct_weight →
  x = 56 := by
sorry

end misread_weight_l4092_409271


namespace least_common_period_is_24_l4092_409264

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) + f (x - 4) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

/-- The least positive period of a function -/
def IsLeastPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  IsPeriod f p ∧ ∀ q, IsPeriod f q → p ≤ q

theorem least_common_period_is_24 :
  ∃ p : ℝ, p = 24 ∧
    (∀ f : ℝ → ℝ, FunctionalEquation f → IsLeastPeriod f p) ∧
    (∀ q : ℝ, (∀ f : ℝ → ℝ, FunctionalEquation f → IsLeastPeriod f q) → p ≤ q) :=
sorry

end least_common_period_is_24_l4092_409264


namespace drama_club_problem_l4092_409251

theorem drama_club_problem (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) (drama_only : ℕ)
  (h_total : total = 75)
  (h_math : math = 42)
  (h_physics : physics = 35)
  (h_both : both = 25)
  (h_drama_only : drama_only = 10) :
  total - ((math + physics - both) + drama_only) = 13 := by
  sorry

end drama_club_problem_l4092_409251


namespace positive_square_iff_greater_l4092_409262

theorem positive_square_iff_greater (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a > b ↔ a^2 > b^2 := by sorry

end positive_square_iff_greater_l4092_409262


namespace first_year_after_2020_with_digit_sum_7_l4092_409247

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_valid_year (year : ℕ) : Prop :=
  year ≥ 1000 ∧ year < 10000

theorem first_year_after_2020_with_digit_sum_7 :
  ∃ (year : ℕ), is_valid_year year ∧ 
    year > 2020 ∧ 
    sum_of_digits year = 7 ∧
    (∀ y, is_valid_year y → y > 2020 → y < year → sum_of_digits y ≠ 7) ∧
    year = 2021 := by
  sorry

end first_year_after_2020_with_digit_sum_7_l4092_409247


namespace perimeter_VWX_equals_5_plus_10_root_5_l4092_409202

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  baseSideLength : ℝ

/-- Midpoints of edges in the right prism -/
structure Midpoints where
  v : ℝ × ℝ × ℝ
  w : ℝ × ℝ × ℝ
  x : ℝ × ℝ × ℝ

/-- Calculate the perimeter of triangle VWX in the right prism -/
def perimeterVWX (prism : RightPrism) (midpoints : Midpoints) : ℝ :=
  sorry

/-- Theorem stating the perimeter of triangle VWX -/
theorem perimeter_VWX_equals_5_plus_10_root_5 (prism : RightPrism) (midpoints : Midpoints) 
  (h1 : prism.height = 20)
  (h2 : prism.baseSideLength = 10)
  (h3 : midpoints.v = (5, 0, 0))
  (h4 : midpoints.w = (10, 5, 0))
  (h5 : midpoints.x = (5, 5, 10)) :
  perimeterVWX prism midpoints = 5 + 10 * Real.sqrt 5 := by
  sorry

end perimeter_VWX_equals_5_plus_10_root_5_l4092_409202


namespace strawberry_jelly_sales_l4092_409263

/-- Represents the number of jars sold for each type of jelly -/
structure JellySales where
  grape : ℕ
  strawberry : ℕ
  raspberry : ℕ
  plum : ℕ

/-- Conditions for jelly sales -/
def valid_jelly_sales (s : JellySales) : Prop :=
  s.grape = 2 * s.strawberry ∧
  s.raspberry = 2 * s.plum ∧
  s.raspberry = s.grape / 3 ∧
  s.plum = 6

theorem strawberry_jelly_sales (s : JellySales) :
  valid_jelly_sales s → s.strawberry = 18 := by
  sorry

end strawberry_jelly_sales_l4092_409263


namespace trigonometric_identities_l4092_409268

theorem trigonometric_identities :
  (∃ (x y : ℝ), 
    x = Real.sin (-14 * Real.pi / 3) + Real.cos (20 * Real.pi / 3) + Real.tan (-53 * Real.pi / 6) ∧
    x = (-3 - Real.sqrt 3) / 6 ∧
    y = Real.tan (675 * Real.pi / 180) - Real.sin (-330 * Real.pi / 180) - Real.cos (960 * Real.pi / 180) ∧
    y = -2) := by sorry

end trigonometric_identities_l4092_409268


namespace landscape_ratio_l4092_409292

theorem landscape_ratio (length : ℝ) (playground_area : ℝ) (playground_ratio : ℝ) :
  length = 240 →
  playground_area = 1200 →
  playground_ratio = 1 / 6 →
  ∃ breadth : ℝ, breadth > 0 ∧ length / breadth = 8 := by
  sorry

end landscape_ratio_l4092_409292


namespace quadratic_solution_difference_squared_l4092_409290

theorem quadratic_solution_difference_squared :
  ∀ a b : ℝ,
  (2 * a^2 - 7 * a + 3 = 0) →
  (2 * b^2 - 7 * b + 3 = 0) →
  (a - b)^2 = 25 / 4 :=
by
  sorry

end quadratic_solution_difference_squared_l4092_409290


namespace probability_of_blue_is_four_thirteenths_l4092_409284

/-- Represents the number of jelly beans of each color in the bag -/
structure JellyBeanBag where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ

/-- Calculates the total number of jelly beans in the bag -/
def totalJellyBeans (bag : JellyBeanBag) : ℕ :=
  bag.red + bag.green + bag.yellow + bag.blue

/-- Calculates the probability of selecting a blue jelly bean -/
def probabilityOfBlue (bag : JellyBeanBag) : ℚ :=
  bag.blue / (totalJellyBeans bag)

/-- Theorem: The probability of selecting a blue jelly bean from the given bag is 4/13 -/
theorem probability_of_blue_is_four_thirteenths :
  let bag : JellyBeanBag := { red := 5, green := 6, yellow := 7, blue := 8 }
  probabilityOfBlue bag = 4 / 13 := by
  sorry

end probability_of_blue_is_four_thirteenths_l4092_409284


namespace asymptote_sum_l4092_409274

theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x ≠ -1 ∧ x ≠ 2 ∧ x ≠ 3 → 
    (x^3 + A*x^2 + B*x + C ≠ 0)) →
  ((x + 1) * (x - 2) * (x - 3) = x^3 + A*x^2 + B*x + C) →
  A + B + C = -5 := by
sorry

end asymptote_sum_l4092_409274


namespace min_sum_xy_l4092_409257

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y - x*y = 0) :
  x + y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 4*y₀ - x₀*y₀ = 0 ∧ x₀ + y₀ = 9 :=
by sorry

end min_sum_xy_l4092_409257


namespace total_cars_is_180_l4092_409201

/-- The total number of cars produced over two days, given the production on the first day and that the second day's production is twice the first day's. -/
def total_cars (day1_production : ℕ) : ℕ :=
  day1_production + 2 * day1_production

/-- Theorem stating that the total number of cars produced is 180 when 60 cars were produced on the first day. -/
theorem total_cars_is_180 : total_cars 60 = 180 := by
  sorry

end total_cars_is_180_l4092_409201


namespace stock_investment_l4092_409235

theorem stock_investment (dividend_rate : ℚ) (dividend_earned : ℚ) (stock_price : ℚ) :
  dividend_rate = 9 / 100 →
  dividend_earned = 120 →
  stock_price = 135 →
  ∃ (investment : ℚ), investment = 1800 ∧ 
    dividend_earned = dividend_rate * (investment * 100 / stock_price) :=
by sorry

end stock_investment_l4092_409235


namespace condition_relationship_l4092_409296

theorem condition_relationship (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 2 → a + b > 3 ∧ a * b > 2) ∧
  (∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 1 ∧ b > 2)) :=
sorry

end condition_relationship_l4092_409296


namespace min_abs_sum_l4092_409294

theorem min_abs_sum (x : ℝ) : 
  ∀ a : ℝ, (∃ x : ℝ, |x + 1| + |x - 3| ≤ a) → a ≥ 4 :=
sorry

end min_abs_sum_l4092_409294


namespace perpendicular_angle_values_l4092_409275

theorem perpendicular_angle_values (α : Real) : 
  (4 * Real.pi < α ∧ α < 6 * Real.pi) →
  (∃ k : ℤ, α = -Real.pi/6 + k * Real.pi) →
  (α = 29 * Real.pi / 6 ∨ α = 35 * Real.pi / 6) := by
  sorry

end perpendicular_angle_values_l4092_409275


namespace plant_arrangements_eq_1271040_l4092_409244

/-- The number of ways to arrange 5 basil plants and 5 tomato plants with given conditions -/
def plant_arrangements : ℕ :=
  let basil_count : ℕ := 5
  let tomato_count : ℕ := 5
  let tomato_group1_size : ℕ := 2
  let tomato_group2_size : ℕ := 3
  let total_groups : ℕ := basil_count + 2  -- 5 basil plants + 2 tomato groups

  Nat.factorial total_groups *
  (Nat.choose total_groups basil_count * Nat.choose 2 1) *
  Nat.factorial tomato_group1_size *
  Nat.factorial tomato_group2_size

theorem plant_arrangements_eq_1271040 : plant_arrangements = 1271040 := by
  sorry

end plant_arrangements_eq_1271040_l4092_409244


namespace discount_calculation_l4092_409280

theorem discount_calculation (price_per_person : ℕ) (num_people : ℕ) (total_cost_with_discount : ℕ) 
  (h1 : price_per_person = 147)
  (h2 : num_people = 2)
  (h3 : total_cost_with_discount = 266) :
  (price_per_person * num_people - total_cost_with_discount) / num_people = 14 := by
  sorry

end discount_calculation_l4092_409280


namespace triangle_BC_equation_l4092_409232

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line in general form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.medianAB (t : Triangle) : Line :=
  { a := 5, b := -3, c := -3 }

def Triangle.medianAC (t : Triangle) : Line :=
  { a := 7, b := -3, c := -5 }

def Triangle.sideBC (t : Triangle) : Line :=
  { a := 2, b := -1, c := -2 }

theorem triangle_BC_equation (t : Triangle) 
  (h1 : t.A = (1, 2))
  (h2 : t.medianAB = { a := 5, b := -3, c := -3 })
  (h3 : t.medianAC = { a := 7, b := -3, c := -5 }) :
  t.sideBC = { a := 2, b := -1, c := -2 } := by
  sorry


end triangle_BC_equation_l4092_409232


namespace equation_solutions_l4092_409273

theorem equation_solutions :
  (∀ x : ℝ, 5 * x^2 - 10 = 0 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2) ∧
  (∀ x : ℝ, 3 * (x - 4)^2 = 375 ↔ x = 4 + 5 * Real.sqrt 5 ∨ x = 4 - 5 * Real.sqrt 5) :=
by sorry

end equation_solutions_l4092_409273


namespace stratified_sampling_most_appropriate_l4092_409239

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | RandomNumber
  | Stratified
  | Systematic

/-- Represents a population with two equal-sized subgroups -/
structure Population :=
  (total_size : ℕ)
  (subgroup_size : ℕ)
  (h_equal_subgroups : subgroup_size * 2 = total_size)

/-- Represents a sampling scenario -/
structure SamplingScenario :=
  (population : Population)
  (sample_size : ℕ)
  (h_sample_size_valid : sample_size ≤ population.total_size)

/-- Determines if a sampling method is appropriate for investigating subgroup differences -/
def is_appropriate_for_subgroup_investigation (method : SamplingMethod) (scenario : SamplingScenario) : Prop :=
  method = SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is the most appropriate method
    for investigating differences between equal-sized subgroups -/
theorem stratified_sampling_most_appropriate
  (scenario : SamplingScenario)
  (h_equal_subgroups : scenario.population.subgroup_size * 2 = scenario.population.total_size) :
  is_appropriate_for_subgroup_investigation SamplingMethod.Stratified scenario :=
sorry

end stratified_sampling_most_appropriate_l4092_409239


namespace parabola_properties_l4092_409222

/-- Represents a parabola of the form y = ax^2 - 2ax + 3 -/
structure Parabola where
  a : ℝ
  h : a ≠ 0

/-- The axis of symmetry of the parabola -/
def axisOfSymmetry (p : Parabola) : ℝ := 1

/-- The shifted parabola's vertex is on the x-axis -/
def vertexOnXAxis (p : Parabola) : Prop :=
  p.a = 3/4 ∨ p.a = -3/2

theorem parabola_properties (p : Parabola) :
  (axisOfSymmetry p = 1) ∧
  (vertexOnXAxis p ↔ (p.a = 3/4 ∨ p.a = -3/2)) := by
  sorry

end parabola_properties_l4092_409222


namespace inverse_function_zero_solution_l4092_409205

/-- Given a function f(x) = 2 / (ax + b) where a and b are nonzero constants,
    prove that the solution to f⁻¹(x) = 0 is x = 2/b -/
theorem inverse_function_zero_solution
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f : ℝ → ℝ := λ x => 2 / (a * x + b)
  (f⁻¹) 0 = 2 / b :=
sorry

end inverse_function_zero_solution_l4092_409205


namespace parking_garage_floor_distance_l4092_409217

/-- Calculates the distance between floors in a parking garage --/
theorem parking_garage_floor_distance (
  total_floors : ℕ) 
  (gate_interval : ℕ) 
  (gate_time : ℝ) 
  (driving_speed : ℝ) 
  (total_time : ℝ)
  (h1 : total_floors = 12)
  (h2 : gate_interval = 3)
  (h3 : gate_time = 120) -- 2 minutes in seconds
  (h4 : driving_speed = 10)
  (h5 : total_time = 1440) :
  ∃ (distance : ℝ), abs (distance - 872.7) < 0.1 := by
  sorry

end parking_garage_floor_distance_l4092_409217


namespace ad_sequence_count_l4092_409281

/-- Represents the number of Olympic ads -/
def num_olympic_ads : ℕ := 3

/-- Represents the number of commercial ads -/
def num_commercial_ads : ℕ := 2

/-- Represents the total number of ads -/
def total_ads : ℕ := num_olympic_ads + num_commercial_ads

/-- Represents the constraint that the last ad must be an Olympic ad -/
def last_ad_is_olympic : Prop := true

/-- Represents the constraint that commercial ads cannot be played consecutively -/
def no_consecutive_commercial_ads : Prop := true

/-- The number of different playback sequences -/
def num_sequences : ℕ := 36

theorem ad_sequence_count :
  num_olympic_ads = 3 →
  num_commercial_ads = 2 →
  total_ads = 5 →
  last_ad_is_olympic →
  no_consecutive_commercial_ads →
  num_sequences = 36 :=
by sorry

end ad_sequence_count_l4092_409281


namespace mary_hourly_wage_l4092_409223

/-- Represents Mary's work schedule and earnings --/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ
  tue_thu_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week --/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.mon_wed_fri_hours + 2 * schedule.tue_thu_hours

/-- Calculates the hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Mary's work schedule --/
def mary_schedule : WorkSchedule :=
  { mon_wed_fri_hours := 9
  , tue_thu_hours := 5
  , weekly_earnings := 407 }

/-- Theorem stating Mary's hourly wage is $11 --/
theorem mary_hourly_wage :
  hourly_wage mary_schedule = 11 := by sorry

end mary_hourly_wage_l4092_409223


namespace right_triangle_segment_ratio_l4092_409204

theorem right_triangle_segment_ratio (a b c r s : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ s > 0 →  -- Positive lengths
  a^2 + b^2 = c^2 →                        -- Right triangle (Pythagorean theorem)
  a / b = 3 / 4 →                          -- Ratio of legs
  r * c = a^2 →                            -- Altitude theorem for r
  s * c = b^2 →                            -- Altitude theorem for s
  r + s = c →                              -- Segments sum to hypotenuse
  r / s = 9 / 16 :=                        -- Conclusion to prove
by sorry

end right_triangle_segment_ratio_l4092_409204


namespace interest_rate_calculation_l4092_409224

theorem interest_rate_calculation 
  (principal : ℝ) 
  (time : ℝ) 
  (interest_difference : ℝ) 
  (h1 : principal = 6100) 
  (h2 : time = 2) 
  (h3 : interest_difference = 61) : 
  ∃ (rate : ℝ), 
    rate = 1 ∧ 
    principal * ((1 + rate / 100) ^ time - 1) - principal * rate * time / 100 = interest_difference :=
sorry

end interest_rate_calculation_l4092_409224


namespace sum_of_ages_in_five_years_l4092_409248

/-- Represents the ages of Viggo, his younger brother, and his sister -/
structure FamilyAges where
  viggo : ℕ
  brother : ℕ
  sister : ℕ

/-- Calculate the ages of the family members after a given number of years -/
def ageAfterYears (ages : FamilyAges) (years : ℕ) : FamilyAges :=
  { viggo := ages.viggo + years
  , brother := ages.brother + years
  , sister := ages.sister + years }

/-- The sum of ages of Viggo, his brother, and his sister -/
def sumOfAges (ages : FamilyAges) : ℕ :=
  ages.viggo + ages.brother + ages.sister

/-- Theorem stating the sum of ages five years from now -/
theorem sum_of_ages_in_five_years :
  ∃ (initialAges : FamilyAges),
    (initialAges.viggo = initialAges.brother + 12) ∧
    (initialAges.sister = initialAges.viggo + 5) ∧
    (initialAges.brother = 10) ∧
    (sumOfAges (ageAfterYears initialAges 5) = 74) := by
  sorry

end sum_of_ages_in_five_years_l4092_409248


namespace miss_walter_stickers_l4092_409276

theorem miss_walter_stickers (gold : ℕ) (silver : ℕ) (bronze : ℕ) (students : ℕ) (stickers_per_student : ℕ)
  (h1 : gold = 50)
  (h2 : silver = 2 * gold)
  (h3 : students = 5)
  (h4 : stickers_per_student = 46)
  (h5 : gold + silver + bronze = students * stickers_per_student) :
  silver - bronze = 20 := by
  sorry

end miss_walter_stickers_l4092_409276


namespace binomial_coefficient_10_3_l4092_409213

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l4092_409213


namespace isabellas_hair_length_l4092_409266

/-- Given Isabella's initial hair length and growth, calculate her final hair length -/
theorem isabellas_hair_length 
  (initial_length : ℕ) 
  (growth : ℕ) 
  (h1 : initial_length = 18) 
  (h2 : growth = 6) : 
  initial_length + growth = 24 := by
  sorry

end isabellas_hair_length_l4092_409266


namespace solution_set_inequality_for_a_b_l4092_409238

-- Define the inequality
def satisfies_inequality (x : ℝ) : Prop := abs (x + 1) + abs (x + 3) < 4

-- Theorem for the solution set
theorem solution_set :
  ∀ x : ℝ, satisfies_inequality x ↔ -4 < x ∧ x < 0 := by sorry

-- Theorem for the inequality between a and b
theorem inequality_for_a_b (a b : ℝ) 
  (ha : satisfies_inequality a) (hb : satisfies_inequality b) :
  2 * abs (a - b) < abs (a * b + 2 * a + 2 * b) := by sorry

end solution_set_inequality_for_a_b_l4092_409238


namespace legoland_animals_l4092_409229

theorem legoland_animals (num_kangaroos : ℕ) (num_koalas : ℕ) : 
  num_kangaroos = 384 → 
  num_kangaroos = 8 * num_koalas → 
  num_kangaroos + num_koalas = 432 := by
sorry

end legoland_animals_l4092_409229


namespace part_one_part_two_l4092_409227

-- Define the inequalities p and q
def p (x a : ℝ) : Prop := x^2 - 6*a*x + 8*a^2 < 0
def q (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

-- Part (1)
theorem part_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x ≤ 3) :=
sorry

-- Part (2)
theorem part_two :
  ∀ a : ℝ, (∀ x : ℝ, p x a → q x) ∧ (∃ x : ℝ, q x ∧ ¬(p x a)) ↔ (1/2 ≤ a ∧ a ≤ 3/4) :=
sorry

end part_one_part_two_l4092_409227


namespace smallest_number_l4092_409211

theorem smallest_number : ∀ (a b c d : ℚ), a = -2 ∧ b = 2 ∧ c = -1/2 ∧ d = 1/2 → a < b ∧ a < c ∧ a < d := by
  sorry

end smallest_number_l4092_409211


namespace disjoint_subsets_count_l4092_409218

theorem disjoint_subsets_count (S : Finset ℕ) : 
  S = Finset.range 12 →
  (Finset.powerset S).card = 2^12 →
  let n := (3^12 - 2 * 2^12 + 1) / 2
  (n : ℕ) = 261625 ∧ n % 1000 = 625 := by
  sorry

end disjoint_subsets_count_l4092_409218


namespace francine_work_schedule_l4092_409208

/-- The number of days Francine does not go to work every week -/
def days_not_working : ℕ :=
  7 - (2240 / (4 * 140))

theorem francine_work_schedule :
  days_not_working = 3 :=
sorry

end francine_work_schedule_l4092_409208


namespace battery_problem_l4092_409249

theorem battery_problem :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (4 * x + 18 * y + 16 * z = 2 * x + 15 * y + 24 * z) ∧
  (4 * x + 18 * y + 16 * z = 6 * x + 12 * y + 20 * z) →
  (4 * x + 18 * y + 16 * z) / z = 48 := by
sorry

end battery_problem_l4092_409249


namespace three_digit_number_problem_l4092_409236

theorem three_digit_number_problem : ∃! n : ℕ, 
  100 ≤ n ∧ n ≤ 999 ∧ 6 * n = 41 * 18 := by
  sorry

end three_digit_number_problem_l4092_409236


namespace maryann_rescue_l4092_409200

/-- The number of friends Maryann needs to rescue -/
def rescue_problem (cheap_time expensive_time total_time : ℕ) : Prop :=
  let time_per_friend := cheap_time + expensive_time
  ∃ (num_friends : ℕ), num_friends * time_per_friend = total_time

theorem maryann_rescue :
  rescue_problem 6 8 42 → ∃ (num_friends : ℕ), num_friends = 3 :=
by
  sorry

end maryann_rescue_l4092_409200


namespace square_sum_reciprocal_l4092_409293

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end square_sum_reciprocal_l4092_409293


namespace purely_imaginary_condition_l4092_409286

/-- Proof that when m(m-1) + mi is purely imaginary, m = 1 -/
theorem purely_imaginary_condition (m : ℝ) : 
  (m * (m - 1) : ℂ) + m * Complex.I = Complex.I * (r : ℝ) → m = 1 :=
by sorry

end purely_imaginary_condition_l4092_409286


namespace cost_price_is_36_l4092_409258

/-- Given the total cloth length, total selling price, and loss per metre, 
    calculate the cost price for one metre of cloth. -/
def cost_price_per_metre (total_length : ℕ) (total_selling_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  (total_selling_price + total_length * loss_per_metre) / total_length

/-- Theorem: The cost price for one metre of cloth is Rs. 36 given the problem conditions. -/
theorem cost_price_is_36 :
  cost_price_per_metre 300 9000 6 = 36 := by
  sorry

#eval cost_price_per_metre 300 9000 6

end cost_price_is_36_l4092_409258


namespace smallest_integer_solution_l4092_409259

theorem smallest_integer_solution : 
  ∃ x : ℤ, (x ≥ 0) ∧ 
    (⌊x / 8⌋ - ⌊x / 40⌋ + ⌊x / 240⌋ = 210) ∧ 
    (∀ y : ℤ, y ≥ 0 → ⌊y / 8⌋ - ⌊y / 40⌋ + ⌊y / 240⌋ = 210 → y ≥ x) ∧
    x = 2016 := by
  sorry

end smallest_integer_solution_l4092_409259


namespace determine_x_value_l4092_409255

theorem determine_x_value (w y z x : ℕ) 
  (hw : w = 90)
  (hz : z = w + 25)
  (hy : y = z + 15)
  (hx : x = y + 8) : 
  x = 138 := by
  sorry

end determine_x_value_l4092_409255


namespace amount_spent_on_toys_l4092_409210

def initial_amount : ℕ := 16
def amount_left : ℕ := 8

theorem amount_spent_on_toys :
  initial_amount - amount_left = 8 :=
by sorry

end amount_spent_on_toys_l4092_409210


namespace spherical_sector_volume_equals_cone_volume_l4092_409241

/-- The volume of a spherical sector is equal to the volume of specific cones -/
theorem spherical_sector_volume_equals_cone_volume (R h : ℝ) (h_pos : 0 < h) (R_pos : 0 < R) :
  let V := (2 * Real.pi * R^2 * h) / 3
  (V = (1/3) * Real.pi * R^2 * (2*h)) ∧ 
  (V = (1/3) * Real.pi * (R*Real.sqrt 2)^2 * h) :=
by
  sorry


end spherical_sector_volume_equals_cone_volume_l4092_409241


namespace alex_born_in_1989_l4092_409207

/-- The year when the first Math Kangaroo test was held -/
def first_math_kangaroo_year : ℕ := 1991

/-- The number of the Math Kangaroo test Alex participated in -/
def alex_participation_number : ℕ := 9

/-- Alex's age when he participated in the Math Kangaroo test -/
def alex_age_at_participation : ℕ := 10

/-- Calculate the year of Alex's birth -/
def alex_birth_year : ℕ := first_math_kangaroo_year + alex_participation_number - 1 - alex_age_at_participation

theorem alex_born_in_1989 : alex_birth_year = 1989 := by
  sorry

end alex_born_in_1989_l4092_409207


namespace intersection_with_y_axis_l4092_409291

/-- The intersection point of the line y = 5x + 1 with the y-axis is (0, 1) -/
theorem intersection_with_y_axis :
  let f : ℝ → ℝ := λ x ↦ 5 * x + 1
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = f p.1 ∧ p = (0, 1) :=
by sorry

end intersection_with_y_axis_l4092_409291


namespace point_A_not_on_transformed_plane_l4092_409299

/-- The similarity transformation coefficient -/
def k : ℚ := 2/3

/-- The original plane equation -/
def plane_a (x y z : ℚ) : Prop := 5*x + y - z + 6 = 0

/-- The transformed plane equation -/
def plane_a' (x y z : ℚ) : Prop := 5*x + y - z + 4 = 0

/-- The point A -/
def point_A : ℚ × ℚ × ℚ := (1, -2, 1)

/-- Theorem stating that point A is not on the transformed plane -/
theorem point_A_not_on_transformed_plane : 
  ¬ plane_a' point_A.1 point_A.2.1 point_A.2.2 :=
sorry

end point_A_not_on_transformed_plane_l4092_409299


namespace shaded_area_is_ten_l4092_409243

/-- Represents a square with a given side length -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- Represents the configuration of two adjacent squares -/
structure TwoSquares where
  small : Square
  large : Square
  adjacent : True  -- This is a placeholder for the adjacency condition

/-- Calculates the area of the shaded region formed by the diagonal of the larger square
    overlapping with the smaller square in a TwoSquares configuration -/
def shaded_area (squares : TwoSquares) : ℝ :=
  sorry

/-- Theorem stating that for a TwoSquares configuration with sides 4 and 12,
    the shaded area is 10 square units -/
theorem shaded_area_is_ten (squares : TwoSquares)
  (h1 : squares.small.side = 4)
  (h2 : squares.large.side = 12) :
  shaded_area squares = 10 :=
sorry

end shaded_area_is_ten_l4092_409243


namespace chocolate_candies_cost_l4092_409260

/-- The cost of buying a specific number of chocolate candies -/
theorem chocolate_candies_cost
  (candies_per_box : ℕ)
  (cost_per_box : ℚ)
  (total_candies : ℕ)
  (h1 : candies_per_box = 30)
  (h2 : cost_per_box = 7.5)
  (h3 : total_candies = 450) :
  (total_candies / candies_per_box : ℚ) * cost_per_box = 112.5 :=
sorry

end chocolate_candies_cost_l4092_409260


namespace intersection_M_N_l4092_409267

def M : Set ℤ := {-2, -1, 0, 1, 2}

def N : Set ℤ := {x | x ≥ 3 ∨ x ≤ -2}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l4092_409267


namespace triangle_area_l4092_409261

theorem triangle_area (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : c = Real.sqrt 19) :
  let S := (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2)
  S = (3 * Real.sqrt 3) / 2 := by
sorry

end triangle_area_l4092_409261


namespace log_inequality_relationship_l4092_409277

theorem log_inequality_relationship (a b : ℝ) :
  (∀ a b, Real.log a > Real.log b → a > b) ∧
  (∃ a b, a > b ∧ ¬(Real.log a > Real.log b)) :=
by sorry

end log_inequality_relationship_l4092_409277


namespace power_function_properties_l4092_409283

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^m

theorem power_function_properties :
  ∃ (m : ℝ), ∀ (x : ℝ), f m x = x^2 ∧
  ∀ (k : ℝ),
    (∀ (x : ℝ), x < 2 ∨ x > k → f m x > (k + 2) * x - 2 * k) ∧
    (k = 2 → ∀ (x : ℝ), x ≠ 2 → f m x > (k + 2) * x - 2 * k) ∧
    (k < 2 → ∀ (x : ℝ), x < k ∨ x > 2 → f m x > (k + 2) * x - 2 * k) :=
by sorry

end power_function_properties_l4092_409283


namespace probability_three_by_three_square_l4092_409234

/-- A square with 16 equally spaced points around its perimeter -/
structure SquareWithPoints :=
  (side_length : ℕ)
  (num_points : ℕ)

/-- The probability of selecting two points that are one unit apart -/
def probability_one_unit_apart (s : SquareWithPoints) : ℚ :=
  sorry

/-- Theorem stating the probability for a 3x3 square with 16 points -/
theorem probability_three_by_three_square :
  ∃ s : SquareWithPoints, s.side_length = 3 ∧ s.num_points = 16 ∧ 
  probability_one_unit_apart s = 1 / 10 :=
sorry

end probability_three_by_three_square_l4092_409234


namespace travel_time_calculation_l4092_409214

/-- Given a constant rate of travel where 1 mile takes 4 minutes,
    prove that the time required to travel 5 miles is 20 minutes. -/
theorem travel_time_calculation (rate : ℝ) (distance : ℝ) :
  rate = 1 / 4 → distance = 5 → rate * distance = 20 := by
  sorry

end travel_time_calculation_l4092_409214


namespace intersection_N_complement_M_l4092_409230

open Set Real

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (1 - 2/x)}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

-- State the theorem
theorem intersection_N_complement_M :
  N ∩ (univ \ M) = Icc 1 2 := by sorry

end intersection_N_complement_M_l4092_409230


namespace game_cost_before_tax_l4092_409231

theorem game_cost_before_tax 
  (weekly_savings : ℝ) 
  (weeks : ℕ) 
  (tax_rate : ℝ) 
  (total_saved : ℝ) 
  (h1 : weekly_savings = 5)
  (h2 : weeks = 11)
  (h3 : tax_rate = 0.1)
  (h4 : total_saved = weekly_savings * weeks)
  : ∃ (pre_tax_cost : ℝ), pre_tax_cost = 50 ∧ total_saved = pre_tax_cost * (1 + tax_rate) :=
by
  sorry

end game_cost_before_tax_l4092_409231


namespace cookout_ratio_l4092_409265

def cookout_2004 : ℕ := 60
def cookout_2005 : ℕ := cookout_2004 / 2
def cookout_2006 : ℕ := 20

theorem cookout_ratio : 
  (cookout_2006 : ℚ) / cookout_2005 = 2 / 3 := by
  sorry

end cookout_ratio_l4092_409265


namespace circle_radius_problem_l4092_409285

theorem circle_radius_problem (r : ℝ) : 
  3 * (2 * Real.pi * r) + 6 = 2 * (Real.pi * r^2) → 
  r = (3 + Real.sqrt 21) / 2 := by
  sorry

end circle_radius_problem_l4092_409285


namespace cylinder_radius_comparison_l4092_409250

theorem cylinder_radius_comparison (h : ℝ) (r₁ : ℝ) (r₂ : ℝ) : 
  h > 0 → r₁ > 0 → r₂ > 0 → h = 4 → r₁ = 6 → 
  (π * r₂^2 * h = 3 * (π * r₁^2 * h)) → r₂ = 6 * Real.sqrt 3 := by
  sorry

end cylinder_radius_comparison_l4092_409250


namespace twenty_five_percent_less_twenty_five_percent_less_proof_l4092_409254

theorem twenty_five_percent_less : ℝ → Prop :=
  fun x => (x + x / 4 = 80 * 3 / 4) → x = 48

-- The proof goes here
theorem twenty_five_percent_less_proof : twenty_five_percent_less 48 := by
  sorry

end twenty_five_percent_less_twenty_five_percent_less_proof_l4092_409254


namespace x_plus_y_equals_32_l4092_409206

theorem x_plus_y_equals_32 (x y : ℝ) 
  (h1 : (4 : ℝ)^x = 16^(y+1)) 
  (h2 : (27 : ℝ)^y = 9^(x-6)) : 
  x + y = 32 := by
sorry

end x_plus_y_equals_32_l4092_409206


namespace parallel_line_through_point_l4092_409220

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point :
  ∀ (given_line : Line),
  given_line.a = 1 ∧ given_line.b = -2 ∧ given_line.c = -2 →
  ∃ (parallel_line : Line),
    parallel parallel_line given_line ∧
    point_on_line 1 0 parallel_line ∧
    parallel_line.a = 1 ∧ parallel_line.b = -2 ∧ parallel_line.c = -1 :=
by sorry

end parallel_line_through_point_l4092_409220


namespace logarithm_expression_equals_one_l4092_409216

-- Define the logarithm base 2
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem logarithm_expression_equals_one :
  2 * (lg (Real.sqrt 2))^2 + lg (Real.sqrt 2) * lg 5 + 
  Real.sqrt ((lg (Real.sqrt 2))^2 - lg 2 + 1) = 1 := by
sorry

end logarithm_expression_equals_one_l4092_409216


namespace watch_correction_theorem_l4092_409297

/-- Represents the time difference between two dates in hours -/
def timeDifference : ℚ := 189

/-- Represents the daily loss rate of the watch in minutes per day -/
def dailyLossRate : ℚ := 13 / 4

/-- Calculates the positive correction in minutes to be added to the watch -/
def watchCorrection (timeDiff : ℚ) (lossRate : ℚ) : ℚ :=
  timeDiff * (lossRate / 24)

/-- Theorem stating that the watch correction is 2457/96 minutes -/
theorem watch_correction_theorem :
  watchCorrection timeDifference dailyLossRate = 2457 / 96 := by
  sorry

#eval watchCorrection timeDifference dailyLossRate

end watch_correction_theorem_l4092_409297


namespace mutual_fund_yield_range_theorem_l4092_409221

/-- Represents the range of annual yields for mutual funds -/
structure YieldRange where
  last_year : ℝ
  improvement_rate : ℝ

/-- Calculates the new range of annual yields after improvement -/
def new_range (yr : YieldRange) : ℝ :=
  yr.last_year * (1 + yr.improvement_rate)

theorem mutual_fund_yield_range_theorem (yr : YieldRange) 
  (h1 : yr.last_year = 10000)
  (h2 : yr.improvement_rate = 0.15) : 
  new_range yr = 11500 := by
  sorry

#check mutual_fund_yield_range_theorem

end mutual_fund_yield_range_theorem_l4092_409221


namespace family_weights_calculation_l4092_409253

/-- Represents the weights of three generations in a family -/
structure FamilyWeights where
  grandmother : ℝ
  daughter : ℝ
  grandchild : ℝ

/-- Given the total weight of all three, the weight of daughter and grandchild, 
    and the relation between grandmother and grandchild weights, 
    prove the individual weights -/
theorem family_weights_calculation (w : FamilyWeights) : 
  w.grandmother + w.daughter + w.grandchild = 110 →
  w.daughter + w.grandchild = 60 →
  w.grandchild = w.grandmother / 5 →
  w.grandmother = 50 ∧ w.daughter = 50 ∧ w.grandchild = 10 := by
  sorry


end family_weights_calculation_l4092_409253


namespace stratified_sample_green_and_carp_l4092_409219

/-- Represents the total number of fish -/
def total_fish : ℕ := 200

/-- Represents the sample size -/
def sample_size : ℕ := 20

/-- Represents the number of green fish -/
def green_fish : ℕ := 20

/-- Represents the number of carp -/
def carp : ℕ := 40

/-- Represents the sum of green fish and carp -/
def green_and_carp : ℕ := green_fish + carp

/-- Theorem stating the number of green fish and carp in the stratified sample -/
theorem stratified_sample_green_and_carp :
  (green_and_carp : ℚ) * sample_size / total_fish = 6 := by sorry

end stratified_sample_green_and_carp_l4092_409219


namespace joan_remaining_books_l4092_409288

/-- Given an initial number of books and a number of books sold, 
    calculate the remaining number of books. -/
def remaining_books (initial : ℕ) (sold : ℕ) : ℕ :=
  initial - sold

/-- Theorem: Given 33 initial books and 26 books sold, 
    the remaining number of books is 7. -/
theorem joan_remaining_books :
  remaining_books 33 26 = 7 := by
  sorry

end joan_remaining_books_l4092_409288


namespace rectangular_plot_length_l4092_409228

theorem rectangular_plot_length 
  (width : ℝ) 
  (num_poles : ℕ) 
  (pole_distance : ℝ) 
  (h1 : width = 40) 
  (h2 : num_poles = 52) 
  (h3 : pole_distance = 5) : 
  let perimeter := (num_poles - 1 : ℝ) * pole_distance
  let length := perimeter / 2 - width
  length = 87.5 := by sorry

end rectangular_plot_length_l4092_409228


namespace f_properties_l4092_409278

/-- The function f(x) -/
def f (a b : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 1

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

/-- Theorem stating the properties of f(x) and its extreme values -/
theorem f_properties :
  ∀ a b : ℝ,
  (∀ x : ℝ, f' a b (x + 1/2) = f' a b (-x + 1/2)) →  -- f'(x) is symmetric about x = -1/2
  f' a b 1 = 0 →                                     -- f'(1) = 0
  a = 3 ∧ b = -12 ∧                                  -- Values of a and b
  f a b (-2) = 21 ∧                                  -- Local maximum
  f a b 1 = -6 ∧                                     -- Local minimum
  (∀ x : ℝ, x < -2 → f' a b x > 0) ∧                 -- f(x) increasing on (-∞, -2)
  (∀ x : ℝ, -2 < x ∧ x < 1 → f' a b x < 0) ∧         -- f(x) decreasing on (-2, 1)
  (∀ x : ℝ, x > 1 → f' a b x > 0)                    -- f(x) increasing on (1, ∞)
  := by sorry


end f_properties_l4092_409278


namespace power_twelve_minus_one_divisible_by_five_l4092_409203

theorem power_twelve_minus_one_divisible_by_five (a : ℤ) (h : ¬ 5 ∣ a) : 
  5 ∣ (a^12 - 1) := by
  sorry

end power_twelve_minus_one_divisible_by_five_l4092_409203


namespace bigger_part_of_54_l4092_409282

theorem bigger_part_of_54 (x y : ℝ) (h1 : x + y = 54) (h2 : 10 * x + 22 * y = 780) (h3 : x > 0) (h4 : y > 0) :
  max x y = 34 := by
sorry

end bigger_part_of_54_l4092_409282


namespace bridge_length_bridge_length_specific_l4092_409289

/-- The length of a bridge given specific train characteristics and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof that the bridge length is 205 meters given the specific conditions -/
theorem bridge_length_specific : bridge_length 170 45 30 = 205 := by
  sorry

end bridge_length_bridge_length_specific_l4092_409289


namespace equation_solution_l4092_409233

theorem equation_solution :
  ∃ x : ℚ, (4 * x^2 + 6 * x + 2) / (x + 2) = 4 * x + 7 ∧ x = -4/3 :=
by sorry

end equation_solution_l4092_409233


namespace pattern_proof_l4092_409237

theorem pattern_proof (a : ℕ) : 4 * a * (a + 1) + 1 = (2 * a + 1)^2 := by
  sorry

end pattern_proof_l4092_409237


namespace complement_M_intersect_N_l4092_409240

def M : Set ℤ := {m : ℤ | m ≤ -3 ∨ m ≥ 2}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem complement_M_intersect_N :
  (Mᶜ : Set ℤ) ∩ N = {-1, 0, 1} := by sorry

end complement_M_intersect_N_l4092_409240


namespace paper_area_difference_paper_area_difference_proof_l4092_409279

/-- The difference in combined area (front and back) between a square sheet of paper
    with side length 11 inches and a rectangular sheet of paper measuring 5.5 inches
    by 11 inches is 121 square inches. -/
theorem paper_area_difference : ℝ → Prop :=
  λ (inch : ℝ) =>
    let square_sheet_side := 11 * inch
    let rect_sheet_length := 5.5 * inch
    let rect_sheet_width := 11 * inch
    let square_sheet_area := 2 * (square_sheet_side * square_sheet_side)
    let rect_sheet_area := 2 * (rect_sheet_length * rect_sheet_width)
    square_sheet_area - rect_sheet_area = 121 * inch * inch

/-- Proof of the paper_area_difference theorem. -/
theorem paper_area_difference_proof : paper_area_difference 1 := by
  sorry

end paper_area_difference_paper_area_difference_proof_l4092_409279


namespace no_integer_root_for_any_a_l4092_409212

theorem no_integer_root_for_any_a : ∀ (a : ℤ), ¬∃ (x : ℤ), x^2 - 2023*x + 2022*a + 1 = 0 := by
  sorry

end no_integer_root_for_any_a_l4092_409212


namespace problem_solution_l4092_409245

theorem problem_solution :
  (∀ n : ℕ, 2 * 8^n * 32^n = 2^17 → n = 2) ∧
  (∀ n : ℕ, ∀ x : ℝ, n > 0 → x^(2*n) = 2 → (2*x^(3*n))^2 - 3*(x^2)^(2*n) = 20) := by
  sorry

end problem_solution_l4092_409245


namespace people_per_car_l4092_409226

/-- Given 63 people and 3 cars, prove that each car will contain 21 people when evenly distributed. -/
theorem people_per_car (total_people : Nat) (num_cars : Nat) (people_per_car : Nat) : 
  total_people = 63 → num_cars = 3 → people_per_car * num_cars = total_people → people_per_car = 21 := by
  sorry

end people_per_car_l4092_409226


namespace infinitely_many_palindromes_in_x_seq_l4092_409252

/-- A sequence is defined as x_n = 2013 + 317n, where n ≥ 0. -/
def x_seq (n : ℕ) : ℕ := 2013 + 317 * n

/-- A number is palindromic if its decimal representation reads the same forwards and backwards. -/
def is_palindrome (n : ℕ) : Prop := sorry

/-- There are infinitely many palindromic numbers in the sequence x_n. -/
theorem infinitely_many_palindromes_in_x_seq :
  ∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ is_palindrome (x_seq n) :=
sorry

end infinitely_many_palindromes_in_x_seq_l4092_409252


namespace quadratic_completion_of_square_l4092_409246

theorem quadratic_completion_of_square :
  ∀ x : ℝ, (x^2 - 8*x + 10 = 0) ↔ ((x - 4)^2 = 6) :=
by sorry

end quadratic_completion_of_square_l4092_409246


namespace john_steve_race_l4092_409270

theorem john_steve_race (john_speed steve_speed : ℝ) (final_push_time : ℝ) (finish_ahead : ℝ) :
  john_speed = 4.2 →
  steve_speed = 3.8 →
  final_push_time = 42.5 →
  finish_ahead = 2 →
  john_speed * final_push_time - steve_speed * final_push_time - finish_ahead = 15 := by
  sorry

end john_steve_race_l4092_409270


namespace xavier_probability_of_success_l4092_409287

theorem xavier_probability_of_success 
  (p_yvonne : ℝ) 
  (p_zelda : ℝ) 
  (p_xavier_and_yvonne_not_zelda : ℝ) 
  (h1 : p_yvonne = 2/3) 
  (h2 : p_zelda = 5/8) 
  (h3 : p_xavier_and_yvonne_not_zelda = 0.0625) :
  ∃ p_xavier : ℝ, 
    p_xavier * p_yvonne * (1 - p_zelda) = p_xavier_and_yvonne_not_zelda ∧ 
    p_xavier = 0.25 :=
by sorry

end xavier_probability_of_success_l4092_409287


namespace skiing_scavenger_ratio_is_two_to_one_l4092_409256

/-- Given a total number of students and the number of students for a scavenger hunting trip,
    calculates the ratio of skiing trip students to scavenger hunting trip students. -/
def skiing_to_scavenger_ratio (total : ℕ) (scavenger : ℕ) : ℚ :=
  let skiing := total - scavenger
  (skiing : ℚ) / (scavenger : ℚ)

/-- Theorem stating that given 12000 total students and 4000 for scavenger hunting,
    the ratio of skiing to scavenger hunting students is 2:1. -/
theorem skiing_scavenger_ratio_is_two_to_one :
  skiing_to_scavenger_ratio 12000 4000 = 2 := by
  sorry

end skiing_scavenger_ratio_is_two_to_one_l4092_409256


namespace nearest_integer_to_3_plus_sqrt5_pow6_l4092_409298

theorem nearest_integer_to_3_plus_sqrt5_pow6 :
  ∃ n : ℤ, n = 22608 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 5)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^6 - (m : ℝ)| :=
by sorry

end nearest_integer_to_3_plus_sqrt5_pow6_l4092_409298


namespace stratified_sampling_l4092_409272

theorem stratified_sampling (total_A total_B sample_A : ℕ) 
  (h1 : total_A = 800)
  (h2 : total_B = 500)
  (h3 : sample_A = 48) :
  (total_B : ℚ) / (total_A + total_B) * sample_A = 30 := by
  sorry

end stratified_sampling_l4092_409272


namespace marble_difference_l4092_409209

theorem marble_difference (drew_original : ℕ) (marcus_original : ℕ) : 
  (drew_original / 4 = 35) →  -- Drew gave 1/4 of his marbles, which is 35
  (drew_original * 3 / 4 = 35) →  -- Drew has 35 marbles after giving 1/4 away
  (marcus_original + 35 = 35) →  -- Marcus has 35 marbles after receiving Drew's 1/4
  (drew_original - marcus_original = 140) := by
  sorry

end marble_difference_l4092_409209


namespace AMC9_paths_l4092_409215

-- Define the grid structure
structure Grid :=
  (has_A : Bool)
  (has_M_left : Bool)
  (has_M_right : Bool)
  (C_count_left : Nat)
  (C_count_right : Nat)
  (nine_count_per_C : Nat)

-- Define the path counting function
def count_paths (g : Grid) : Nat :=
  let left_paths := if g.has_M_left then g.C_count_left * g.nine_count_per_C else 0
  let right_paths := if g.has_M_right then g.C_count_right * g.nine_count_per_C else 0
  left_paths + right_paths

-- Theorem statement
theorem AMC9_paths (g : Grid) 
  (h1 : g.has_A)
  (h2 : g.has_M_left)
  (h3 : g.has_M_right)
  (h4 : g.C_count_left = 4)
  (h5 : g.C_count_right = 2)
  (h6 : g.nine_count_per_C = 2) :
  count_paths g = 24 := by
  sorry


end AMC9_paths_l4092_409215


namespace interval_condition_l4092_409269

theorem interval_condition (x : ℝ) : 
  (2 < 4*x ∧ 4*x < 5 ∧ 2 < 5*x ∧ 5*x < 5) ↔ (1/2 < x ∧ x < 1) :=
sorry

end interval_condition_l4092_409269


namespace initial_coloring_books_l4092_409242

theorem initial_coloring_books (books_removed : ℝ) (coupons_per_book : ℝ) (total_coupons : ℕ) :
  books_removed = 20 →
  coupons_per_book = 4 →
  total_coupons = 80 →
  ∃ (initial_books : ℕ), initial_books = 40 ∧ 
    (initial_books : ℝ) - books_removed = (total_coupons : ℝ) / coupons_per_book :=
by
  sorry

end initial_coloring_books_l4092_409242
