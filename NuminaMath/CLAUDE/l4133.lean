import Mathlib

namespace NUMINAMATH_CALUDE_candy_mixture_cost_l4133_413364

/-- Proves that the desired cost per pound of a candy mixture is $6.00 -/
theorem candy_mixture_cost
  (weight_expensive : ℝ)
  (price_expensive : ℝ)
  (weight_cheap : ℝ)
  (price_cheap : ℝ)
  (h1 : weight_expensive = 25)
  (h2 : price_expensive = 8)
  (h3 : weight_cheap = 50)
  (h4 : price_cheap = 5) :
  (weight_expensive * price_expensive + weight_cheap * price_cheap) /
  (weight_expensive + weight_cheap) = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_l4133_413364


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l4133_413305

theorem fraction_sum_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  b / a + a / b > 2 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l4133_413305


namespace NUMINAMATH_CALUDE_number_of_pigs_l4133_413368

theorem number_of_pigs (total_cost : ℕ) (num_hens : ℕ) (avg_price_hen : ℕ) (avg_price_pig : ℕ) :
  total_cost = 1200 →
  num_hens = 10 →
  avg_price_hen = 30 →
  avg_price_pig = 300 →
  ∃ (num_pigs : ℕ), num_pigs = 3 ∧ total_cost = num_pigs * avg_price_pig + num_hens * avg_price_hen :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_pigs_l4133_413368


namespace NUMINAMATH_CALUDE_small_prism_surface_area_l4133_413374

/-- Represents the dimensions of a rectangular prism -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (d : Dimensions) : ℕ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Theorem: Surface area of small prism in arrangement of 9 identical prisms -/
theorem small_prism_surface_area 
  (small : Dimensions) 
  (large_surface_area : ℕ) 
  (h1 : large_surface_area = 360) 
  (h2 : 3 * small.width = 2 * small.length) 
  (h3 : small.length = 3 * small.height) 
  (h4 : surfaceArea { length := 3 * small.width, 
                      width := 3 * small.width, 
                      height := small.length + small.height } = large_surface_area) : 
  surfaceArea small = 88 := by
sorry

end NUMINAMATH_CALUDE_small_prism_surface_area_l4133_413374


namespace NUMINAMATH_CALUDE_cubic_function_extrema_condition_l4133_413336

/-- Given a cubic function f(x) = x³ - 3x² + ax - b that has both a maximum and a minimum value,
    prove that the parameter a must be less than 3. -/
theorem cubic_function_extrema_condition (a b : ℝ) : 
  (∃ (x_min x_max : ℝ), ∀ x : ℝ, 
    x^3 - 3*x^2 + a*x - b ≤ x_max^3 - 3*x_max^2 + a*x_max - b ∧ 
    x^3 - 3*x^2 + a*x - b ≥ x_min^3 - 3*x_min^2 + a*x_min - b) →
  a < 3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_condition_l4133_413336


namespace NUMINAMATH_CALUDE_yoga_studio_average_weight_l4133_413312

theorem yoga_studio_average_weight 
  (num_men : Nat) 
  (num_women : Nat) 
  (avg_weight_men : ℝ) 
  (avg_weight_women : ℝ) 
  (h1 : num_men = 8) 
  (h2 : num_women = 6) 
  (h3 : avg_weight_men = 190) 
  (h4 : avg_weight_women = 120) : 
  (num_men * avg_weight_men + num_women * avg_weight_women) / (num_men + num_women : ℝ) = 160 := by
  sorry

end NUMINAMATH_CALUDE_yoga_studio_average_weight_l4133_413312


namespace NUMINAMATH_CALUDE_pump_x_time_is_4_hours_l4133_413300

/-- Represents the rate of a pump in terms of fraction of total water pumped per hour -/
structure PumpRate where
  rate : ℝ
  rate_positive : rate > 0

/-- Represents the scenario of two pumps working on draining a flooded basement -/
structure BasementPumpScenario where
  pump_x : PumpRate
  pump_y : PumpRate
  total_water : ℝ
  total_water_positive : total_water > 0
  y_alone_time : ℝ
  y_alone_time_eq : pump_y.rate * y_alone_time = total_water
  combined_time : ℝ
  combined_time_eq : (pump_x.rate + pump_y.rate) * combined_time = total_water / 2

/-- The main theorem stating that pump X takes 4 hours to pump out half the water -/
theorem pump_x_time_is_4_hours (scenario : BasementPumpScenario) : 
  scenario.pump_x.rate * 4 = scenario.total_water / 2 ∧ 
  scenario.y_alone_time = 20 ∧ 
  scenario.combined_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_pump_x_time_is_4_hours_l4133_413300


namespace NUMINAMATH_CALUDE_last_number_is_30_l4133_413322

theorem last_number_is_30 (numbers : Fin 8 → ℝ) 
  (h1 : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5 + numbers 6 + numbers 7) / 8 = 25)
  (h2 : (numbers 0 + numbers 1) / 2 = 20)
  (h3 : (numbers 2 + numbers 3 + numbers 4) / 3 = 26)
  (h4 : numbers 5 = numbers 6 - 4)
  (h5 : numbers 5 = numbers 7 - 6) :
  numbers 7 = 30 := by
sorry

end NUMINAMATH_CALUDE_last_number_is_30_l4133_413322


namespace NUMINAMATH_CALUDE_stratified_sample_is_proportional_l4133_413376

/-- Represents the number of students in each grade and the sample size -/
structure School :=
  (total : ℕ)
  (freshmen : ℕ)
  (sophomores : ℕ)
  (seniors : ℕ)
  (sample_size : ℕ)

/-- Represents the number of students sampled from each grade -/
structure Sample :=
  (freshmen : ℕ)
  (sophomores : ℕ)
  (seniors : ℕ)

/-- Calculates the proportional sample size for a given grade -/
def proportional_sample (grade_size : ℕ) (school : School) : ℕ :=
  (grade_size * school.sample_size) / school.total

/-- Checks if a sample is proportionally correct -/
def is_proportional_sample (school : School) (sample : Sample) : Prop :=
  sample.freshmen = proportional_sample school.freshmen school ∧
  sample.sophomores = proportional_sample school.sophomores school ∧
  sample.seniors = proportional_sample school.seniors school

/-- Theorem: The stratified sample is proportional for the given school -/
theorem stratified_sample_is_proportional (school : School)
  (h1 : school.total = 900)
  (h2 : school.freshmen = 300)
  (h3 : school.sophomores = 200)
  (h4 : school.seniors = 400)
  (h5 : school.sample_size = 45)
  (h6 : school.total = school.freshmen + school.sophomores + school.seniors) :
  is_proportional_sample school ⟨15, 10, 20⟩ := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_is_proportional_l4133_413376


namespace NUMINAMATH_CALUDE_extremum_sum_l4133_413358

theorem extremum_sum (a b : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = x^3 + a*x^2 + b*x + a^2) ∧ 
   (f 1 = 10) ∧ 
   (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ 10)) →
  a + b = -7 := by
sorry

end NUMINAMATH_CALUDE_extremum_sum_l4133_413358


namespace NUMINAMATH_CALUDE_andrew_donation_start_age_l4133_413395

/-- Proves the age at which Andrew started donating given his current age, total donation, and yearly donation amount -/
theorem andrew_donation_start_age 
  (current_age : ℕ) 
  (total_donation : ℕ) 
  (yearly_donation : ℕ) 
  (h1 : current_age = 29) 
  (h2 : total_donation = 133) 
  (h3 : yearly_donation = 7) : 
  current_age - (total_donation / yearly_donation) = 10 := by
  sorry

end NUMINAMATH_CALUDE_andrew_donation_start_age_l4133_413395


namespace NUMINAMATH_CALUDE_cost_of_goods_l4133_413308

/-- The cost of goods problem -/
theorem cost_of_goods
  (mango_rice_ratio : ℝ)
  (flour_rice_ratio : ℝ)
  (flour_cost : ℝ)
  (h1 : 10 * mango_rice_ratio = 24)
  (h2 : 6 * flour_rice_ratio = 2)
  (h3 : flour_cost = 21) :
  4 * (24 / 10 * (2 / 6 * flour_cost)) + 3 * (2 / 6 * flour_cost) + 5 * flour_cost = 898.80 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_goods_l4133_413308


namespace NUMINAMATH_CALUDE_distance_A_B_min_value_expression_solutions_equation_max_product_mn_l4133_413326

-- Define the distance function on a number line
def distance (a b : ℝ) : ℝ := |a - b|

-- Statement 1
theorem distance_A_B : distance (-10) 8 = 18 := by sorry

-- Statement 2
theorem min_value_expression : 
  ∀ x : ℝ, |x - 3| + |x + 2| ≥ 5 := by sorry

-- Statement 3
theorem solutions_equation : 
  ∀ y : ℝ, |y - 3| + |y + 1| = 8 ↔ y = 5 ∨ y = -3 := by sorry

-- Statement 4
theorem max_product_mn : 
  ∀ m n : ℤ, (|m + 1| + |2 - m|) * (|n - 1| + |n + 3|) = 12 → 
  m * n ≤ 3 := by sorry

end NUMINAMATH_CALUDE_distance_A_B_min_value_expression_solutions_equation_max_product_mn_l4133_413326


namespace NUMINAMATH_CALUDE_equation_solution_l4133_413339

theorem equation_solution :
  ∃ (x : ℝ), x ≠ 0 ∧ x ≠ -3 ∧ (2 / x + x / (x + 3) = 1) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4133_413339


namespace NUMINAMATH_CALUDE_nidas_chocolates_l4133_413370

theorem nidas_chocolates (x : ℕ) 
  (h1 : 3 * x + 5 + 25 = 5 * x) : 3 * x + 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_nidas_chocolates_l4133_413370


namespace NUMINAMATH_CALUDE_bernoulli_expected_value_l4133_413330

/-- A random variable with a Bernoulli distribution -/
structure BernoulliRV (p : ℝ) :=
  (prob : 0 < p ∧ p < 1)

/-- The probability mass function for a Bernoulli random variable -/
def pmf (p : ℝ) (X : BernoulliRV p) (k : ℕ) : ℝ :=
  if k = 0 then (1 - p) else if k = 1 then p else 0

/-- The expected value of a Bernoulli random variable -/
def expectedValue (p : ℝ) (X : BernoulliRV p) : ℝ :=
  0 * pmf p X 0 + 1 * pmf p X 1

/-- Theorem: The expected value of a Bernoulli random variable is p -/
theorem bernoulli_expected_value (p : ℝ) (X : BernoulliRV p) :
  expectedValue p X = p := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_expected_value_l4133_413330


namespace NUMINAMATH_CALUDE_five_circles_arrangement_exists_four_circles_arrangement_not_exists_l4133_413321

-- Define a circle on a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a ray starting from a point
structure Ray where
  start : ℝ × ℝ
  direction : ℝ × ℝ
  direction_nonzero : direction ≠ (0, 0)

-- Function to check if a ray intersects a circle
def ray_intersects_circle (r : Ray) (c : Circle) : Prop :=
  sorry

-- Function to check if a ray intersects at least two circles from a list
def ray_intersects_at_least_two (r : Ray) (circles : List Circle) : Prop :=
  sorry

-- Function to check if a circle covers a point
def circle_covers_point (c : Circle) (p : ℝ × ℝ) : Prop :=
  sorry

-- Theorem for part (a)
theorem five_circles_arrangement_exists :
  ∃ (circles : List Circle), circles.length = 5 ∧
  ∀ (r : Ray), r.start = (0, 0) → ray_intersects_at_least_two r circles :=
sorry

-- Theorem for part (b)
theorem four_circles_arrangement_not_exists :
  ¬ ∃ (circles : List Circle), circles.length = 4 ∧
  (∀ c ∈ circles, ¬ circle_covers_point c (0, 0)) ∧
  (∀ (r : Ray), r.start = (0, 0) → ray_intersects_at_least_two r circles) :=
sorry

end NUMINAMATH_CALUDE_five_circles_arrangement_exists_four_circles_arrangement_not_exists_l4133_413321


namespace NUMINAMATH_CALUDE_sum_calculation_l4133_413328

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

def sum_squares_odd_integers (a b : ℕ) : ℕ :=
  List.sum (List.map (λ x => x * x) (List.filter (λ x => x % 2 = 1) (List.range (b - a + 1) |>.map (λ x => x + a))))

theorem sum_calculation :
  sum_integers 30 50 + count_even_integers 30 50 + sum_squares_odd_integers 30 50 = 17661 := by
  sorry

end NUMINAMATH_CALUDE_sum_calculation_l4133_413328


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_equals_one_l4133_413351

def a : ℝ × ℝ := (-2, 1)
def b (x : ℝ) : ℝ × ℝ := (x, x^2 + 1)

theorem perpendicular_vectors_x_equals_one :
  ∀ x : ℝ, (a.1 * (b x).1 + a.2 * (b x).2 = 0) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_equals_one_l4133_413351


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l4133_413369

theorem triangle_abc_properties (b c : ℝ) (A B : ℝ) :
  A = π / 3 →
  3 * b = 2 * c →
  (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 →
  b = 2 ∧ Real.sin B = Real.sqrt 21 / 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l4133_413369


namespace NUMINAMATH_CALUDE_function_decomposition_l4133_413356

-- Define the domain
def Domain : Set ℝ := {x : ℝ | x ≠ 1 ∧ x ≠ -1}

-- Define odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x ∈ Domain, f (-x) = -f x
def IsEven (g : ℝ → ℝ) : Prop := ∀ x ∈ Domain, g (-x) = g x

-- State the theorem
theorem function_decomposition
  (f g : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_even : IsEven g)
  (h_sum : ∀ x ∈ Domain, f x + g x = 1 / (x - 1)) :
  (∀ x ∈ Domain, f x = x / (x^2 - 1)) ∧
  (∀ x ∈ Domain, g x = 1 / (x^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_decomposition_l4133_413356


namespace NUMINAMATH_CALUDE_sum_of_digits_B_is_seven_l4133_413394

def sum_of_digits (n : ℕ) : ℕ := sorry

def A : ℕ := sum_of_digits (4444^4444)

def B : ℕ := sum_of_digits A

theorem sum_of_digits_B_is_seven : sum_of_digits B = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_B_is_seven_l4133_413394


namespace NUMINAMATH_CALUDE_cylinder_cone_volume_relation_l4133_413353

/-- The volume of a cylinder with the same base and height as a cone is 3 times the volume of the cone -/
theorem cylinder_cone_volume_relation (Vcone : ℝ) (Vcylinder : ℝ) :
  Vcone > 0 → Vcylinder = 3 * Vcone := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cone_volume_relation_l4133_413353


namespace NUMINAMATH_CALUDE_crazy_silly_school_movies_l4133_413357

/-- The 'crazy silly school' series problem -/
theorem crazy_silly_school_movies :
  ∀ (total_books watched_movies remaining_movies : ℕ),
    total_books = 21 →
    watched_movies = 4 →
    remaining_movies = 4 →
    watched_movies + remaining_movies = 8 :=
by sorry

end NUMINAMATH_CALUDE_crazy_silly_school_movies_l4133_413357


namespace NUMINAMATH_CALUDE_june_net_income_l4133_413381

def daily_milk_production : ℕ := 200
def milk_price : ℚ := 355/100
def monthly_expenses : ℕ := 3000
def days_in_june : ℕ := 30

def daily_income : ℚ := daily_milk_production * milk_price

def total_income : ℚ := daily_income * days_in_june

def net_income : ℚ := total_income - monthly_expenses

theorem june_net_income : net_income = 18300 := by
  sorry

end NUMINAMATH_CALUDE_june_net_income_l4133_413381


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l4133_413350

theorem point_in_fourth_quadrant (a b : ℝ) (z z₁ z₂ : ℂ) :
  z = a + b * Complex.I ∧
  z₁ = 1 + Complex.I ∧
  z₂ = 3 - Complex.I ∧
  z = z₁ * z₂ →
  a > 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l4133_413350


namespace NUMINAMATH_CALUDE_min_distance_to_line_l4133_413382

/-- The minimum distance from integral points to the line y = (5/3)x + 4/5 -/
theorem min_distance_to_line : 
  ∃ (d : ℝ), d = Real.sqrt 34 / 85 ∧ 
  ∀ (x y : ℤ), 
    d ≤ (|(5 : ℝ) / 3 * x + 4 / 5 - y| / Real.sqrt (1 + (5 / 3)^2)) ∧
    ∃ (x₀ y₀ : ℤ), (|(5 : ℝ) / 3 * x₀ + 4 / 5 - y₀| / Real.sqrt (1 + (5 / 3)^2)) = d := by
  sorry


end NUMINAMATH_CALUDE_min_distance_to_line_l4133_413382


namespace NUMINAMATH_CALUDE_adams_shopping_cost_l4133_413367

/-- Calculates the total cost of Adam's shopping, including discount and sales tax -/
def total_cost (sandwich_price : ℚ) (chips_price : ℚ) (water_price : ℚ) 
                (sandwich_count : ℕ) (chips_count : ℕ) (water_count : ℕ) 
                (tax_rate : ℚ) : ℚ :=
  let sandwich_cost := (sandwich_count - 1) * sandwich_price
  let chips_cost := chips_count * chips_price
  let water_cost := water_count * water_price
  let subtotal := sandwich_cost + chips_cost + water_cost
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Theorem stating that Adam's total shopping cost is $29.15 -/
theorem adams_shopping_cost : 
  total_cost 4 3.5 2 4 3 2 0.1 = 29.15 := by
  sorry

end NUMINAMATH_CALUDE_adams_shopping_cost_l4133_413367


namespace NUMINAMATH_CALUDE_no_three_intersections_l4133_413384

-- Define a circle in Euclidean space
structure EuclideanCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define an intersection point
def IntersectionPoint (c1 c2 : EuclideanCircle) := 
  {p : ℝ × ℝ | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∧
               (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2}

-- Theorem statement
theorem no_three_intersections 
  (c1 c2 : EuclideanCircle) 
  (h_distinct : c1 ≠ c2) : 
  ¬∃ (p1 p2 p3 : ℝ × ℝ), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    p1 ∈ IntersectionPoint c1 c2 ∧
    p2 ∈ IntersectionPoint c1 c2 ∧
    p3 ∈ IntersectionPoint c1 c2 :=
sorry

end NUMINAMATH_CALUDE_no_three_intersections_l4133_413384


namespace NUMINAMATH_CALUDE_power_equation_l4133_413341

theorem power_equation (a : ℝ) (m n k : ℤ) (h1 : a^m = 2) (h2 : a^n = 4) (h3 : a^k = 32) :
  a^(3*m + 2*n - k) = 4 := by
sorry

end NUMINAMATH_CALUDE_power_equation_l4133_413341


namespace NUMINAMATH_CALUDE_utilities_percentage_l4133_413373

/-- Represents the budget allocation of a company -/
structure BudgetAllocation where
  salaries : ℝ
  research_and_development : ℝ
  equipment : ℝ
  supplies : ℝ
  transportation_degrees : ℝ
  total_budget : ℝ

/-- The theorem stating that given the specific budget allocation, the percentage spent on utilities is 5% -/
theorem utilities_percentage (budget : BudgetAllocation) : 
  budget.salaries = 60 ∧ 
  budget.research_and_development = 9 ∧ 
  budget.equipment = 4 ∧ 
  budget.supplies = 2 ∧ 
  budget.transportation_degrees = 72 ∧ 
  budget.total_budget = 100 →
  100 - (budget.salaries + budget.research_and_development + budget.equipment + budget.supplies + (budget.transportation_degrees * 100 / 360)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_utilities_percentage_l4133_413373


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l4133_413331

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a - b) / c = (Real.sin B + Real.sin C) / (Real.sin B + Real.sin A) ∧
  a = Real.sqrt 7 ∧
  b = 2 * c

theorem triangle_ABC_properties (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C) : 
  A = 2 * Real.pi / 3 ∧ 
  (1/2 : ℝ) * b * c * Real.sin A = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l4133_413331


namespace NUMINAMATH_CALUDE_min_h25_for_tenuous_min_sum_l4133_413371

/-- A function h : ℕ → ℤ is tenuous if h(x) + h(y) > 2 * y^2 for all positive integers x and y. -/
def Tenuous (h : ℕ → ℤ) : Prop :=
  ∀ x y : ℕ, x > 0 → y > 0 → h x + h y > 2 * y^2

/-- The sum of h(1) to h(30) for a function h : ℕ → ℤ. -/
def SumH30 (h : ℕ → ℤ) : ℤ :=
  (Finset.range 30).sum (λ i => h (i + 1))

theorem min_h25_for_tenuous_min_sum (h : ℕ → ℤ) :
  Tenuous h → (∀ g : ℕ → ℤ, Tenuous g → SumH30 h ≤ SumH30 g) → h 25 ≥ 1189 := by
  sorry

end NUMINAMATH_CALUDE_min_h25_for_tenuous_min_sum_l4133_413371


namespace NUMINAMATH_CALUDE_normal_lemon_tree_production_l4133_413392

/-- The number of lemons produced by a normal lemon tree per year. -/
def normal_lemon_production : ℕ := 60

/-- The number of trees in Jim's grove. -/
def jims_trees : ℕ := 1500

/-- The number of lemons Jim's grove produces per year. -/
def jims_production : ℕ := 135000

/-- Jim's trees produce 50% more lemons than normal trees. -/
def jims_tree_efficiency : ℚ := 3/2

theorem normal_lemon_tree_production :
  normal_lemon_production * jims_trees * jims_tree_efficiency = jims_production :=
by sorry

end NUMINAMATH_CALUDE_normal_lemon_tree_production_l4133_413392


namespace NUMINAMATH_CALUDE_final_value_calculation_l4133_413344

theorem final_value_calculation : 
  let initial_value := 52
  let first_increase := initial_value * 1.20
  let second_decrease := first_increase * 0.90
  let final_increase := second_decrease * 1.15
  final_increase = 64.584 := by
sorry

end NUMINAMATH_CALUDE_final_value_calculation_l4133_413344


namespace NUMINAMATH_CALUDE_complex_cube_inequality_l4133_413337

theorem complex_cube_inequality (z : ℂ) (h : Complex.abs (z + 1) > 2) :
  Complex.abs (z^3 + 1) > 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_inequality_l4133_413337


namespace NUMINAMATH_CALUDE_permutation_count_mod_1000_l4133_413342

/-- The number of permutations of a 15-character string with 4 A's, 5 B's, and 6 C's -/
def N : ℕ := sorry

/-- Condition: None of the first four letters is an A -/
axiom cond1 : sorry

/-- Condition: None of the next five letters is a B -/
axiom cond2 : sorry

/-- Condition: None of the last six letters is a C -/
axiom cond3 : sorry

/-- Theorem: The number of permutations N satisfying the conditions is congruent to 320 modulo 1000 -/
theorem permutation_count_mod_1000 : N ≡ 320 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_permutation_count_mod_1000_l4133_413342


namespace NUMINAMATH_CALUDE_even_odd_handshakers_l4133_413318

theorem even_odd_handshakers (population : ℕ) : ∃ (even_shakers odd_shakers : ℕ),
  even_shakers + odd_shakers = population ∧ 
  Even odd_shakers := by
  sorry

end NUMINAMATH_CALUDE_even_odd_handshakers_l4133_413318


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4133_413316

/-- 
Given a geometric sequence with:
- First term a = 5
- Common ratio r = 2
- Number of terms n = 5

Prove that the sum of this sequence is 155.
-/
theorem geometric_sequence_sum : 
  let a : ℕ := 5  -- first term
  let r : ℕ := 2  -- common ratio
  let n : ℕ := 5  -- number of terms
  (a * (r^n - 1)) / (r - 1) = 155 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4133_413316


namespace NUMINAMATH_CALUDE_restaurant_bill_example_l4133_413343

/-- Calculates the total cost for a group at a restaurant with specific pricing and discount rules. -/
def restaurant_bill (total_people : ℕ) (num_kids : ℕ) (num_upgrades : ℕ) 
  (adult_meal_cost : ℚ) (upgrade_cost : ℚ) (adult_drink_cost : ℚ) (kid_drink_cost : ℚ) 
  (discount_rate : ℚ) : ℚ :=
  let num_adults := total_people - num_kids
  let meal_cost := num_adults * adult_meal_cost
  let upgrade_total := num_upgrades * upgrade_cost
  let drink_cost := num_adults * adult_drink_cost + num_kids * kid_drink_cost
  let subtotal := meal_cost + upgrade_total + drink_cost
  let discount := subtotal * discount_rate
  subtotal - discount

/-- Theorem stating that the total cost for the given group is $97.20 -/
theorem restaurant_bill_example : 
  restaurant_bill 11 2 4 8 4 2 1 (1/10) = 97.2 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_example_l4133_413343


namespace NUMINAMATH_CALUDE_incorrect_fraction_equality_l4133_413311

theorem incorrect_fraction_equality (a b : ℝ) (h : 0.7 * a ≠ b) :
  (0.2 * a + b) / (0.7 * a - b) ≠ (2 * a + b) / (7 * a - b) :=
sorry

end NUMINAMATH_CALUDE_incorrect_fraction_equality_l4133_413311


namespace NUMINAMATH_CALUDE_temperature_conversion_l4133_413334

theorem temperature_conversion (C F : ℝ) : 
  C = 4/7 * (F - 40) → C = 25 → F = 83.75 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l4133_413334


namespace NUMINAMATH_CALUDE_escalator_length_l4133_413355

/-- The length of an escalator given its speed, a person's walking speed on it, and the time taken to cover the entire length. -/
theorem escalator_length 
  (escalator_speed : ℝ) 
  (walking_speed : ℝ) 
  (time_taken : ℝ) 
  (h1 : escalator_speed = 15) 
  (h2 : walking_speed = 3) 
  (h3 : time_taken = 10) : 
  escalator_speed * time_taken + walking_speed * time_taken = 180 := by
  sorry

end NUMINAMATH_CALUDE_escalator_length_l4133_413355


namespace NUMINAMATH_CALUDE_age_puzzle_solution_l4133_413319

theorem age_puzzle_solution :
  ∃! x : ℕ, 6 * (x + 6) - 6 * (x - 6) = x ∧ x = 72 :=
by sorry

end NUMINAMATH_CALUDE_age_puzzle_solution_l4133_413319


namespace NUMINAMATH_CALUDE_new_people_in_country_l4133_413389

theorem new_people_in_country (born : ℕ) (immigrated : ℕ) : 
  born = 90171 → immigrated = 16320 → born + immigrated = 106491 :=
by
  sorry

end NUMINAMATH_CALUDE_new_people_in_country_l4133_413389


namespace NUMINAMATH_CALUDE_sqrt_fifth_power_of_sqrt5_to_4th_l4133_413379

theorem sqrt_fifth_power_of_sqrt5_to_4th : (((5 : ℝ) ^ (1/2)) ^ 5) ^ (1/2) ^ 4 = 9765625 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fifth_power_of_sqrt5_to_4th_l4133_413379


namespace NUMINAMATH_CALUDE_fence_cost_square_plot_l4133_413307

/-- The cost of building a fence around a square plot -/
theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 54) :
  let side_length : ℝ := Real.sqrt area
  let perimeter : ℝ := 4 * side_length
  let total_cost : ℝ := perimeter * price_per_foot
  total_cost = 3672 := by
sorry


end NUMINAMATH_CALUDE_fence_cost_square_plot_l4133_413307


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_value_l4133_413345

-- Define the curve
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

-- Define the derivative of the curve
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem tangent_line_implies_a_value (a : ℝ) :
  (∀ x, x ≠ 0 → (f a x - f a 0) / (x - 0) ≤ 2) ∧
  (∀ x, x ≠ 0 → (f a x - f a 0) / (x - 0) ≥ 2) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_value_l4133_413345


namespace NUMINAMATH_CALUDE_driver_speed_problem_l4133_413314

theorem driver_speed_problem (S : ℝ) : 
  (∀ (day : ℕ), day ≤ 6 → 
    (3 * S + 4 * 25) * day = 1140) →
  S = 30 := by
sorry

end NUMINAMATH_CALUDE_driver_speed_problem_l4133_413314


namespace NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l4133_413348

theorem parametric_to_ordinary_equation (α : ℝ) :
  let x := Real.sin (α / 2) + Real.cos (α / 2)
  let y := Real.sqrt (2 + Real.sin α)
  (y ^ 2 - x ^ 2 = 1) ∧
  (|x| ≤ Real.sqrt 2) ∧
  (1 ≤ y) ∧
  (y ≤ Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l4133_413348


namespace NUMINAMATH_CALUDE_contractor_absence_l4133_413332

/-- A contractor's work problem -/
theorem contractor_absence (total_days : ℕ) (daily_wage : ℚ) (daily_fine : ℚ) (total_amount : ℚ) :
  total_days = 30 ∧ 
  daily_wage = 25 ∧ 
  daily_fine = (15/2) ∧ 
  total_amount = 425 →
  ∃ (days_worked days_absent : ℕ),
    days_worked + days_absent = total_days ∧
    daily_wage * days_worked - daily_fine * days_absent = total_amount ∧
    days_absent = 10 := by
  sorry

end NUMINAMATH_CALUDE_contractor_absence_l4133_413332


namespace NUMINAMATH_CALUDE_parabola_properties_l4133_413315

/-- Represents a quadratic function of the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

def Parabola.opensDownward (p : Parabola) : Prop := p.a < 0

def Parabola.axisOfSymmetry (p : Parabola) : ℝ := p.h

def Parabola.vertex (p : Parabola) : ℝ × ℝ := (p.h, p.k)

def Parabola.increasingOnInterval (p : Parabola) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → p.a * (x - p.h)^2 + p.k < p.a * (y - p.h)^2 + p.k

theorem parabola_properties (p : Parabola) (h1 : p.a = -1) (h2 : p.h = -1) (h3 : p.k = 3) :
  (p.opensDownward ∧ 
   p.vertex = (-1, 3) ∧ 
   ¬(p.axisOfSymmetry = 1) ∧ 
   ¬(p.increasingOnInterval 0 (-p.h))) := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l4133_413315


namespace NUMINAMATH_CALUDE_sunway_taihulight_performance_l4133_413333

theorem sunway_taihulight_performance :
  (12.5 * (10^12 : ℝ)) = (1.25 * (10^13 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_sunway_taihulight_performance_l4133_413333


namespace NUMINAMATH_CALUDE_smallest_square_cover_l4133_413390

/-- The smallest square that can be covered by 3x4 rectangles -/
def smallest_square_side : ℕ := 12

/-- The area of a 3x4 rectangle -/
def rectangle_area : ℕ := 3 * 4

/-- The number of 3x4 rectangles needed to cover the smallest square -/
def num_rectangles : ℕ := smallest_square_side^2 / rectangle_area

theorem smallest_square_cover :
  ∀ (side : ℕ), 
  side % smallest_square_side = 0 →
  side^2 % rectangle_area = 0 →
  (side^2 / rectangle_area ≥ num_rectangles) ∧
  (num_rectangles * rectangle_area = smallest_square_side^2) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_cover_l4133_413390


namespace NUMINAMATH_CALUDE_quartic_root_product_l4133_413317

theorem quartic_root_product (a : ℝ) : 
  (∃ x y : ℝ, x * y = -32 ∧ 
   x^4 - 18*x^3 + a*x^2 + 200*x - 1984 = 0 ∧
   y^4 - 18*y^3 + a*y^2 + 200*y - 1984 = 0) →
  a = 86 := by
sorry

end NUMINAMATH_CALUDE_quartic_root_product_l4133_413317


namespace NUMINAMATH_CALUDE_largest_number_proof_l4133_413375

theorem largest_number_proof (x y z : ℝ) 
  (h_order : x < y ∧ y < z)
  (h_sum : x + y + z = 102)
  (h_diff1 : z - y = 10)
  (h_diff2 : y - x = 5) :
  z = 127 / 3 := by
sorry

end NUMINAMATH_CALUDE_largest_number_proof_l4133_413375


namespace NUMINAMATH_CALUDE_max_candy_leftover_l4133_413362

theorem max_candy_leftover (x : ℕ) : ∃ (q r : ℕ), x = 11 * q + r ∧ r ≤ 10 ∧ ∀ (r' : ℕ), x = 11 * q + r' → r' ≤ r := by
  sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l4133_413362


namespace NUMINAMATH_CALUDE_smaller_square_area_equals_larger_l4133_413396

/-- A circle with a square inscribed in it and a smaller square with one side coinciding with the larger square and two vertices on the circle. -/
structure SquaresInCircle where
  /-- Radius of the circle -/
  r : ℝ
  /-- Side length of the larger square -/
  s : ℝ
  /-- Side length of the smaller square -/
  x : ℝ
  /-- The larger square is inscribed in the circle -/
  h1 : s = 2 * r
  /-- The smaller square has two vertices on the circle -/
  h2 : x^2 + (r + x)^2 = r^2

/-- The area of the smaller square is equal to the area of the larger square -/
theorem smaller_square_area_equals_larger (sqc : SquaresInCircle) : 
  sqc.x^2 = sqc.s^2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_smaller_square_area_equals_larger_l4133_413396


namespace NUMINAMATH_CALUDE_triangle_ABC_is_obtuse_angled_l4133_413377

/-- Triangle ABC is obtuse-angled given the specified angle conditions -/
theorem triangle_ABC_is_obtuse_angled (A B C : ℝ) 
  (h1 : A + B = 141)
  (h2 : C + B = 165)
  (h3 : A + B + C = 180) : 
  ∃ (angle : ℝ), angle > 90 ∧ (angle = A ∨ angle = B ∨ angle = C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_is_obtuse_angled_l4133_413377


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l4133_413387

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ x : ℝ, x^3 - 18*x^2 + 77*x - 120 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 18*s^2 + 77*s - 120) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 196 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l4133_413387


namespace NUMINAMATH_CALUDE_solution_is_two_equation4_solution_l4133_413313

theorem solution_is_two : ∃ x : ℝ, x = 2 ∧ 7 * x - 14 = 0 := by
  sorry

-- Additional definitions to represent the given equations
def equation1 (x : ℝ) : Prop := 4 * x = 2
def equation2 (x : ℝ) : Prop := 3 * x + 6 = 0
def equation3 (x : ℝ) : Prop := (1 / 2) * x = 0
def equation4 (x : ℝ) : Prop := 7 * x - 14 = 0

-- Theorem stating that equation4 has a solution of x = 2
theorem equation4_solution : ∃ x : ℝ, x = 2 ∧ equation4 x := by
  sorry

end NUMINAMATH_CALUDE_solution_is_two_equation4_solution_l4133_413313


namespace NUMINAMATH_CALUDE_symmetry_point_xoz_l4133_413303

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoz plane in 3D space -/
def xoz_plane : Set Point3D := {p : Point3D | p.y = 0}

/-- Symmetry with respect to the xoz plane -/
def symmetry_xoz (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, p.z⟩

theorem symmetry_point_xoz :
  let p : Point3D := ⟨1, 2, 3⟩
  symmetry_xoz p = ⟨1, -2, 3⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetry_point_xoz_l4133_413303


namespace NUMINAMATH_CALUDE_prob_both_selected_l4133_413302

/-- The probability of both Ram and Ravi being selected in an exam -/
theorem prob_both_selected (prob_ram prob_ravi : ℚ) 
  (h_ram : prob_ram = 3/7)
  (h_ravi : prob_ravi = 1/5) :
  prob_ram * prob_ravi = 3/35 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_selected_l4133_413302


namespace NUMINAMATH_CALUDE_parabola_through_point_l4133_413359

theorem parabola_through_point (x y : ℝ) :
  (x = 1 ∧ y = 2) →
  (y^2 = 4*x ∨ x^2 = (1/2)*y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_through_point_l4133_413359


namespace NUMINAMATH_CALUDE_max_value_of_f_l4133_413327

noncomputable def f (a b x : ℝ) : ℝ := (4 - x^2) * (a * x^2 + b * x + 5)

theorem max_value_of_f (a b : ℝ) :
  (∀ x : ℝ, f a b x = f a b (-3 - x)) →
  (∃ x : ℝ, ∀ y : ℝ, f a b y ≤ f a b x) ∧
  (∃ x : ℝ, f a b x = 36) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l4133_413327


namespace NUMINAMATH_CALUDE_intersection_line_equation_l4133_413361

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of the line passing through the intersection points of two circles --/
def intersectionLine (c1 c2 : Circle) : ℝ → ℝ → Prop :=
  fun x y => x + y = -143/117

/-- The main theorem statement --/
theorem intersection_line_equation (c1 c2 : Circle) 
  (h1 : c1 = ⟨(-5, -6), 10⟩) 
  (h2 : c2 = ⟨(4, 7), Real.sqrt 85⟩) : 
  ∀ x y, (x - c1.center.1)^2 + (y - c1.center.2)^2 = c1.radius^2 ∧ 
         (x - c2.center.1)^2 + (y - c2.center.2)^2 = c2.radius^2 → 
  intersectionLine c1 c2 x y := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l4133_413361


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l4133_413378

theorem algebraic_expression_equality (x y : ℝ) : 
  x - 2*y + 8 = 18 → 3*x - 6*y + 4 = 34 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l4133_413378


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l4133_413393

def U : Finset Int := {-2, -1, 0, 1, 2}
def A : Finset Int := {-2, -1, 0}
def B : Finset Int := {0, 1, 2}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l4133_413393


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l4133_413399

theorem smallest_prime_divisor_of_sum : 
  ∃ (n : ℕ), n = 6^15 + 9^11 ∧ (∀ p : ℕ, Prime p → p ∣ n → p ≥ 3) ∧ 3 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l4133_413399


namespace NUMINAMATH_CALUDE_alternating_draw_probability_l4133_413386

def total_balls : ℕ := 9
def white_balls : ℕ := 5
def black_balls : ℕ := 4

def alternating_sequence_probability : ℚ :=
  1 / (total_balls.choose black_balls)

theorem alternating_draw_probability :
  alternating_sequence_probability = 1 / 126 :=
by sorry

end NUMINAMATH_CALUDE_alternating_draw_probability_l4133_413386


namespace NUMINAMATH_CALUDE_equation_solution_l4133_413310

theorem equation_solution : ∃! x : ℝ, 
  x ≠ 2 ∧ x ≠ 3 ∧ (x^3 - 4*x^2)/(x^2 - 5*x + 6) - x = 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4133_413310


namespace NUMINAMATH_CALUDE_servant_months_worked_l4133_413309

/-- Calculates the number of months served given the annual salary and the amount paid -/
def months_served (annual_salary : ℚ) (amount_paid : ℚ) : ℚ :=
  (amount_paid * 12) / annual_salary

theorem servant_months_worked (annual_salary : ℚ) (amount_paid : ℚ) 
  (h1 : annual_salary = 90)
  (h2 : amount_paid = 75) :
  months_served annual_salary amount_paid = 10 := by
  sorry

end NUMINAMATH_CALUDE_servant_months_worked_l4133_413309


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l4133_413391

theorem cubic_roots_sum (r s t : ℝ) : 
  r^3 - 15*r^2 + 25*r - 10 = 0 →
  s^3 - 15*s^2 + 25*s - 10 = 0 →
  t^3 - 15*t^2 + 25*t - 10 = 0 →
  (r / (1/r + s*t)) + (s / (1/s + t*r)) + (t / (1/t + r*s)) = 175/11 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l4133_413391


namespace NUMINAMATH_CALUDE_sarahs_flour_purchase_l4133_413398

/-- Sarah's flour purchase problem -/
theorem sarahs_flour_purchase
  (rye : ℝ)
  (chickpea : ℝ)
  (pastry : ℝ)
  (total : ℝ)
  (h_rye : rye = 5)
  (h_chickpea : chickpea = 3)
  (h_pastry : pastry = 2)
  (h_total : total = 20)
  : total - (rye + chickpea + pastry) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_flour_purchase_l4133_413398


namespace NUMINAMATH_CALUDE_new_cards_count_l4133_413363

def cards_per_page : ℕ := 3
def old_cards : ℕ := 10
def pages_used : ℕ := 6

theorem new_cards_count : 
  pages_used * cards_per_page - old_cards = 8 := by sorry

end NUMINAMATH_CALUDE_new_cards_count_l4133_413363


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l4133_413388

/-- p is a sufficient but not necessary condition for q -/
def is_sufficient_not_necessary (p q : Prop) : Prop :=
  (p → q) ∧ ¬(q → p)

theorem sufficient_condition_range (a : ℝ) :
  is_sufficient_not_necessary (∀ x : ℝ, 4 - x ≤ 6) (∀ x : ℝ, x > a - 1) →
  a < -1 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l4133_413388


namespace NUMINAMATH_CALUDE_special_function_properties_l4133_413354

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x - f y)

/-- The theorem stating the properties of the special function -/
theorem special_function_properties (f : ℝ → ℝ) (hf : special_function f) :
  f 2 = 0 ∧ ∃! v : ℝ, f 2 = v :=
sorry

end NUMINAMATH_CALUDE_special_function_properties_l4133_413354


namespace NUMINAMATH_CALUDE_factorial_expression_equals_2015_l4133_413320

-- Define factorial function
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_expression_equals_2015 : 
  (factorial (factorial 2014 - 1) * factorial 2015) / factorial (factorial 2014) = 2015 := by
  sorry

end NUMINAMATH_CALUDE_factorial_expression_equals_2015_l4133_413320


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l4133_413329

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 19) 
  (h2 : speed_against_stream = 11) : 
  (speed_with_stream + speed_against_stream) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l4133_413329


namespace NUMINAMATH_CALUDE_quadratic_roots_exist_l4133_413325

theorem quadratic_roots_exist (a b c : ℝ) (h : a * c < 0) : 
  ∃ x : ℝ, a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_exist_l4133_413325


namespace NUMINAMATH_CALUDE_max_horses_for_25_and_7_l4133_413347

/-- Given a total number of horses and a minimum number of races to determine the top 3 fastest,
    calculate the maximum number of horses that can race together at a time. -/
def max_horses_per_race (total_horses : ℕ) (min_races : ℕ) : ℕ :=
  sorry

/-- Theorem stating that for 25 horses and 7 minimum races, the maximum number of horses
    that can race together is 5. -/
theorem max_horses_for_25_and_7 :
  max_horses_per_race 25 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_horses_for_25_and_7_l4133_413347


namespace NUMINAMATH_CALUDE_town_population_male_count_l4133_413397

theorem town_population_male_count (total_population : ℕ) (num_groups : ℕ) (male_groups : ℕ) : 
  total_population = 480 →
  num_groups = 4 →
  male_groups = 2 →
  (total_population / num_groups) * male_groups = 240 := by
sorry

end NUMINAMATH_CALUDE_town_population_male_count_l4133_413397


namespace NUMINAMATH_CALUDE_pablo_candy_cost_l4133_413383

/-- The cost of candy given Pablo's reading and spending habits -/
def candy_cost (pages_per_book : ℕ) (books_read : ℕ) (earnings_per_page : ℚ) (money_left : ℚ) : ℚ :=
  (pages_per_book * books_read : ℕ) * earnings_per_page - money_left

/-- Theorem stating the cost of candy given Pablo's specific situation -/
theorem pablo_candy_cost :
  candy_cost 150 12 (1 / 100) 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_pablo_candy_cost_l4133_413383


namespace NUMINAMATH_CALUDE_masking_tape_length_l4133_413304

/-- The total length of masking tape needed for four walls -/
def total_tape_length (wall_width1 : ℝ) (wall_width2 : ℝ) : ℝ :=
  2 * wall_width1 + 2 * wall_width2

/-- Theorem: The total length of masking tape needed is 20 meters -/
theorem masking_tape_length :
  total_tape_length 4 6 = 20 :=
by sorry

end NUMINAMATH_CALUDE_masking_tape_length_l4133_413304


namespace NUMINAMATH_CALUDE_grid_separation_impossible_l4133_413360

/-- Represents a point on the grid -/
structure GridPoint where
  x : Fin 8
  y : Fin 8

/-- Represents a line on the grid -/
structure GridLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a line passes through a point -/
def line_passes_through (l : GridLine) (p : GridPoint) : Prop :=
  l.a * p.x.val + l.b * p.y.val + l.c = 0

/-- Checks if two points are separated by a line -/
def points_separated_by_line (l : GridLine) (p1 p2 : GridPoint) : Prop :=
  (l.a * p1.x.val + l.b * p1.y.val + l.c) * (l.a * p2.x.val + l.b * p2.y.val + l.c) < 0

/-- The main theorem stating the impossibility of the grid separation -/
theorem grid_separation_impossible :
  ¬ ∃ (lines : Fin 13 → GridLine),
    (∀ (l : Fin 13) (p : GridPoint), ¬ line_passes_through (lines l) p) ∧
    (∀ (p1 p2 : GridPoint), p1 ≠ p2 → ∃ (l : Fin 13), points_separated_by_line (lines l) p1 p2) :=
by sorry

end NUMINAMATH_CALUDE_grid_separation_impossible_l4133_413360


namespace NUMINAMATH_CALUDE_quadratic_min_bound_l4133_413346

theorem quadratic_min_bound (p q α β : ℝ) (n : ℤ) (h : ℝ → ℝ) :
  (∀ x, h x = x^2 + p*x + q) →
  h α = 0 →
  h β = 0 →
  α ≠ β →
  (n : ℝ) < α →
  α < β →
  β < (n + 1 : ℝ) →
  min (h n) (h (n + 1)) < (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_min_bound_l4133_413346


namespace NUMINAMATH_CALUDE_octagon_coloring_count_l4133_413372

/-- Represents a coloring of 8 disks arranged in an octagon. -/
structure OctagonColoring where
  blue : Finset (Fin 8)
  red : Finset (Fin 8)
  yellow : Finset (Fin 8)
  partition : Disjoint blue red ∧ Disjoint blue yellow ∧ Disjoint red yellow
  cover : blue ∪ red ∪ yellow = Finset.univ
  blue_count : blue.card = 4
  red_count : red.card = 3
  yellow_count : yellow.card = 1

/-- The group of symmetries of an octagon. -/
def OctagonSymmetry : Type := Unit -- Placeholder, actual implementation would be more complex

/-- Two colorings are equivalent if one can be obtained from the other by a symmetry. -/
def equivalent (c₁ c₂ : OctagonColoring) (sym : OctagonSymmetry) : Prop := sorry

/-- The number of distinct colorings under symmetry. -/
def distinctColorings : ℕ := sorry

/-- The main theorem: There are exactly 26 distinct colorings. -/
theorem octagon_coloring_count : distinctColorings = 26 := by sorry

end NUMINAMATH_CALUDE_octagon_coloring_count_l4133_413372


namespace NUMINAMATH_CALUDE_inequality_solution_l4133_413385

theorem inequality_solution (x : ℝ) : 
  (x + 3) / (x + 4) > (4 * x + 5) / (3 * x + 10) ↔ 
  (x > -10/3 ∧ x < -4) ∨ 
  (x > (-1 - Real.sqrt 41) / 4 ∧ x < (-1 + Real.sqrt 41) / 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4133_413385


namespace NUMINAMATH_CALUDE_diamond_calculation_l4133_413349

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_calculation :
  let x := diamond (diamond 1 3) 2
  let y := diamond 1 (diamond 3 2)
  x - y = -13/30 := by sorry

end NUMINAMATH_CALUDE_diamond_calculation_l4133_413349


namespace NUMINAMATH_CALUDE_solve_equation_l4133_413352

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 4
def g (x : ℝ) : ℝ := x^2 - 2

-- State the theorem
theorem solve_equation (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 18) : a = Real.sqrt (Real.sqrt 14 + 2) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4133_413352


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l4133_413338

/-- Proves that the weight of the replaced person is 65 kg given the conditions of the problem -/
theorem weight_of_replaced_person
  (n : ℕ)
  (original_average : ℝ)
  (new_average_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : n = 10)
  (h2 : new_average_increase = 7.2)
  (h3 : new_person_weight = 137)
  : ∃ (replaced_weight : ℝ),
    replaced_weight = new_person_weight - n * new_average_increase ∧
    replaced_weight = 65 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l4133_413338


namespace NUMINAMATH_CALUDE_contradiction_assumption_for_greater_than_l4133_413380

theorem contradiction_assumption_for_greater_than (a b : ℝ) : 
  (¬(a > b) ↔ (a ≤ b)) := by sorry

end NUMINAMATH_CALUDE_contradiction_assumption_for_greater_than_l4133_413380


namespace NUMINAMATH_CALUDE_pancake_stacks_sold_l4133_413365

/-- The number of stacks of pancakes sold at a fundraiser -/
def pancake_stacks : ℕ := sorry

/-- The cost of one stack of pancakes in dollars -/
def pancake_cost : ℚ := 4

/-- The cost of one slice of bacon in dollars -/
def bacon_cost : ℚ := 2

/-- The number of bacon slices sold -/
def bacon_slices : ℕ := 90

/-- The total revenue from the fundraiser in dollars -/
def total_revenue : ℚ := 420

/-- Theorem stating that the number of pancake stacks sold is 60 -/
theorem pancake_stacks_sold : pancake_stacks = 60 := by sorry

end NUMINAMATH_CALUDE_pancake_stacks_sold_l4133_413365


namespace NUMINAMATH_CALUDE_kevin_initial_phones_l4133_413306

/-- The number of phones Kevin had at the beginning of the day -/
def initial_phones : ℕ := 33

/-- The number of phones Kevin repaired by afternoon -/
def repaired_phones : ℕ := 3

/-- The number of phones dropped off by a client -/
def new_phones : ℕ := 6

/-- The number of phones each person (Kevin and coworker) will repair -/
def phones_per_person : ℕ := 9

theorem kevin_initial_phones :
  initial_phones = 33 ∧
  repaired_phones = 3 ∧
  new_phones = 6 ∧
  phones_per_person = 9 →
  initial_phones + new_phones - repaired_phones = 2 * phones_per_person :=
by sorry

end NUMINAMATH_CALUDE_kevin_initial_phones_l4133_413306


namespace NUMINAMATH_CALUDE_dolphins_score_l4133_413324

theorem dolphins_score (total_points sharks_points dolphins_points : ℕ) : 
  total_points = 36 →
  sharks_points = dolphins_points + 12 →
  sharks_points + dolphins_points = total_points →
  dolphins_points = 12 := by
sorry

end NUMINAMATH_CALUDE_dolphins_score_l4133_413324


namespace NUMINAMATH_CALUDE_principal_calculation_l4133_413301

def interest_rates : List ℚ := [6/100, 75/1000, 8/100, 85/1000, 9/100]

theorem principal_calculation (total_interest : ℚ) (rates : List ℚ) :
  total_interest = 6016.75 ∧ rates = interest_rates →
  (total_interest / rates.sum) = 15430 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l4133_413301


namespace NUMINAMATH_CALUDE_helen_cookies_proof_l4133_413323

/-- The number of cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 31

/-- The number of cookies Helen baked the day before yesterday -/
def cookies_day_before_yesterday : ℕ := 419

/-- The total number of cookies Helen baked until last night -/
def total_cookies : ℕ := cookies_yesterday + cookies_day_before_yesterday

theorem helen_cookies_proof : total_cookies = 450 := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_proof_l4133_413323


namespace NUMINAMATH_CALUDE_marcus_car_mpg_l4133_413335

/-- Represents a car with its mileage and fuel efficiency characteristics -/
structure Car where
  initial_mileage : ℕ
  final_mileage : ℕ
  tank_capacity : ℕ
  num_fills : ℕ

/-- Calculates the miles per gallon for a given car -/
def miles_per_gallon (c : Car) : ℚ :=
  (c.final_mileage - c.initial_mileage : ℚ) / (c.tank_capacity * c.num_fills : ℚ)

/-- Theorem stating that Marcus's car gets 30 miles per gallon -/
theorem marcus_car_mpg :
  let marcus_car : Car := {
    initial_mileage := 1728,
    final_mileage := 2928,
    tank_capacity := 20,
    num_fills := 2
  }
  miles_per_gallon marcus_car = 30 := by
  sorry

end NUMINAMATH_CALUDE_marcus_car_mpg_l4133_413335


namespace NUMINAMATH_CALUDE_box_weight_l4133_413340

/-- Given a pallet with boxes, calculate the weight of each box. -/
theorem box_weight (total_weight : ℝ) (num_boxes : ℕ) (h1 : total_weight = 267) (h2 : num_boxes = 3) :
  total_weight / num_boxes = 89 := by
  sorry

end NUMINAMATH_CALUDE_box_weight_l4133_413340


namespace NUMINAMATH_CALUDE_not_always_possible_within_30_moves_l4133_413366

/-- Represents a move on the board -/
inductive Move
  | add_two : Fin 3 → Fin 3 → Move
  | subtract_all : Move

/-- The state of the board -/
def Board := Fin 3 → ℕ

/-- Apply a move to the board -/
def apply_move (b : Board) (m : Move) : Board :=
  match m with
  | Move.add_two i j => fun k => if k = i ∨ k = j then b k + 1 else b k
  | Move.subtract_all => fun k => if b k > 0 then b k - 1 else 0

/-- Check if all numbers on the board are zero -/
def all_zero (b : Board) : Prop := ∀ i, b i = 0

/-- The main theorem -/
theorem not_always_possible_within_30_moves :
  ∃ (initial : Board),
    (∀ i, 1 ≤ initial i ∧ initial i ≤ 9) ∧
    (∀ i j, i ≠ j → initial i ≠ initial j) ∧
    ¬∃ (moves : List Move),
      moves.length ≤ 30 ∧
      all_zero (moves.foldl apply_move initial) :=
by sorry

end NUMINAMATH_CALUDE_not_always_possible_within_30_moves_l4133_413366
