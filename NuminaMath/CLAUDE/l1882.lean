import Mathlib

namespace odd_power_eight_minus_one_mod_nine_l1882_188240

theorem odd_power_eight_minus_one_mod_nine (n : ℕ) (h : Odd n) : (8^n - 1) % 9 = 7 := by
  sorry

end odd_power_eight_minus_one_mod_nine_l1882_188240


namespace amoebas_after_two_weeks_l1882_188205

/-- The number of amoebas in the tank on a given day -/
def amoebas (day : ℕ) : ℕ :=
  if day ≤ 7 then
    2^day
  else
    2^7 * 3^(day - 7)

/-- Theorem stating the number of amoebas after 14 days -/
theorem amoebas_after_two_weeks : amoebas 14 = 279936 := by
  sorry

end amoebas_after_two_weeks_l1882_188205


namespace alberts_current_funds_l1882_188279

/-- The problem of calculating Albert's current funds --/
theorem alberts_current_funds
  (total_cost : ℝ)
  (additional_needed : ℝ)
  (h1 : total_cost = 18.50)
  (h2 : additional_needed = 12) :
  total_cost - additional_needed = 6.50 := by
  sorry

end alberts_current_funds_l1882_188279


namespace largest_four_digit_divisible_by_four_l1882_188235

theorem largest_four_digit_divisible_by_four :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 4 = 0 → n ≤ 9996 :=
by sorry

end largest_four_digit_divisible_by_four_l1882_188235


namespace product_of_primes_summing_to_91_l1882_188277

theorem product_of_primes_summing_to_91 (p q : ℕ) : 
  Prime p → Prime q → p + q = 91 → p * q = 178 := by sorry

end product_of_primes_summing_to_91_l1882_188277


namespace distinct_book_selections_l1882_188292

theorem distinct_book_selections (n k : ℕ) (h1 : n = 15) (h2 : k = 3) :
  Nat.choose n k = 455 := by
  sorry

end distinct_book_selections_l1882_188292


namespace lawn_length_is_80_l1882_188202

/-- Represents a rectangular lawn with roads -/
structure LawnWithRoads where
  width : ℝ
  length : ℝ
  road_width : ℝ
  travel_cost_per_sqm : ℝ
  total_travel_cost : ℝ

/-- Calculates the area of the roads on the lawn -/
def road_area (l : LawnWithRoads) : ℝ :=
  l.road_width * l.length + l.road_width * (l.width - l.road_width)

/-- Theorem stating the length of the lawn given specific conditions -/
theorem lawn_length_is_80 (l : LawnWithRoads) 
    (h1 : l.width = 60)
    (h2 : l.road_width = 10)
    (h3 : l.travel_cost_per_sqm = 5)
    (h4 : l.total_travel_cost = 6500)
    (h5 : l.total_travel_cost = l.travel_cost_per_sqm * road_area l) :
  l.length = 80 := by
  sorry

end lawn_length_is_80_l1882_188202


namespace min_value_theorem_l1882_188230

theorem min_value_theorem (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  2/x + 9/(1-2*x) ≥ 25 ∧ ∃ y ∈ Set.Ioo 0 (1/2), 2/y + 9/(1-2*y) = 25 :=
sorry

end min_value_theorem_l1882_188230


namespace tangent_line_x_squared_at_one_l1882_188255

/-- The equation of the tangent line to y = x^2 at x = 1 is y = 2x - 1 -/
theorem tangent_line_x_squared_at_one :
  let f (x : ℝ) := x^2
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 2*x - 1 := by
sorry

end tangent_line_x_squared_at_one_l1882_188255


namespace wand_price_l1882_188293

theorem wand_price (price : ℝ) (original_price : ℝ) : 
  price = 12 → price = (1/8) * original_price → original_price = 96 := by
sorry

end wand_price_l1882_188293


namespace t_range_l1882_188232

/-- The quadratic function -/
def f (x : ℝ) : ℝ := -x^2 + 6*x - 7

/-- The maximum value function -/
def y_max (t : ℝ) : ℝ := -(t-3)^2 + 2

/-- Theorem stating the range of t -/
theorem t_range (t : ℝ) :
  (∀ x, t ≤ x ∧ x ≤ t+2 → f x ≤ y_max t) →
  (∃ x, t ≤ x ∧ x ≤ t+2 ∧ f x = y_max t) →
  t ≥ 3 :=
sorry

end t_range_l1882_188232


namespace constant_function_proof_l1882_188237

theorem constant_function_proof (f g h : ℕ → ℕ)
  (h_injective : Function.Injective h)
  (g_surjective : Function.Surjective g)
  (f_def : ∀ n, f n = g n - h n + 1) :
  ∀ n, f n = 1 := by
  sorry

end constant_function_proof_l1882_188237


namespace fixed_points_of_quadratic_l1882_188274

/-- The quadratic function f(x) always passes through two fixed points -/
theorem fixed_points_of_quadratic (a : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + (3*a - 1)*x - (10*a + 3)
  (f 2 = -5 ∧ f (-5) = 2) := by sorry

end fixed_points_of_quadratic_l1882_188274


namespace fermats_little_theorem_l1882_188250

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (hp : Prime p) : a^p ≡ a [MOD p] := by
  sorry

end fermats_little_theorem_l1882_188250


namespace problem_1_problem_2_l1882_188241

-- Problem 1
theorem problem_1 (a b : ℝ) : (a + 2*b)^2 - 4*b*(a + b) = a^2 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  ((x^2 - 2*x) / (x^2 - 4*x + 4) + 1 / (2 - x)) / ((x - 1) / (x^2 - 4)) = x + 2 := by sorry

end problem_1_problem_2_l1882_188241


namespace sum_denominator_divisible_by_prime_l1882_188259

theorem sum_denominator_divisible_by_prime (p : ℕ) (n : ℕ) (b : Fin n → ℕ) :
  Prime p →
  (∃! i : Fin n, p ∣ b i) →
  (∀ i : Fin n, 0 < b i) →
  ∃ (num den : ℕ), 
    (0 < den) ∧
    (Nat.gcd num den = 1) ∧
    (p ∣ den) ∧
    (Finset.sum Finset.univ (λ i => 1 / (b i : ℚ)) = num / den) :=
by sorry

end sum_denominator_divisible_by_prime_l1882_188259


namespace min_value_sum_reciprocals_l1882_188212

theorem min_value_sum_reciprocals (a b : ℝ) (h : Real.log a + Real.log b = 0) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ Real.log x + Real.log y = 0 ∧ 2/x + 1/y < 2/a + 1/b) ∨ 
  (2/a + 1/b = 2 * Real.sqrt 2) :=
sorry

end min_value_sum_reciprocals_l1882_188212


namespace exponential_equation_solution_l1882_188248

theorem exponential_equation_solution :
  ∃! x : ℝ, (32 : ℝ) ^ (x - 2) / (16 : ℝ) ^ (x - 1) = (512 : ℝ) ^ (x + 1) ∧ x = -15/8 := by
  sorry

end exponential_equation_solution_l1882_188248


namespace eighth_box_books_l1882_188220

theorem eighth_box_books (total_books : ℕ) (num_boxes : ℕ) (books_per_box : ℕ) 
  (h1 : total_books = 800)
  (h2 : num_boxes = 8)
  (h3 : books_per_box = 105) :
  total_books - (num_boxes - 1) * books_per_box = 65 := by
  sorry

end eighth_box_books_l1882_188220


namespace point_on_transformed_graph_l1882_188290

/-- Given a function g where g(3) = 8, there exists a point (x, y) on the graph of 
    y = 4g(3x-1) + 6 such that x + y = 40 -/
theorem point_on_transformed_graph (g : ℝ → ℝ) (h : g 3 = 8) :
  ∃ x y : ℝ, 4 * g (3 * x - 1) + 6 = y ∧ x + y = 40 := by
  sorry

end point_on_transformed_graph_l1882_188290


namespace alternating_color_probability_l1882_188266

/-- The probability of drawing 8 balls from a box containing 5 white and 3 black balls,
    such that the draws alternate in color starting with a white ball. -/
theorem alternating_color_probability :
  let total_balls : ℕ := 8
  let white_balls : ℕ := 5
  let black_balls : ℕ := 3
  let total_arrangements : ℕ := Nat.choose total_balls black_balls
  let favorable_arrangements : ℕ := 1
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 56 := by
  sorry

end alternating_color_probability_l1882_188266


namespace earth_fresh_water_coverage_l1882_188200

theorem earth_fresh_water_coverage : 
  ∀ (land_coverage : ℝ) (salt_water_percentage : ℝ),
  land_coverage = 3 / 10 →
  salt_water_percentage = 97 / 100 →
  (1 - land_coverage) * (1 - salt_water_percentage) = 21 / 1000 := by
sorry

end earth_fresh_water_coverage_l1882_188200


namespace parabola_intersection_length_l1882_188234

/-- Given a parabola y = x^2 - mx - 3 that intersects the x-axis at points A and B,
    where m is an integer, the length of AB is 4. -/
theorem parabola_intersection_length (m : ℤ) (A B : ℝ) : 
  (∃ x : ℝ, x^2 - m*x - 3 = 0) → 
  (A^2 - m*A - 3 = 0) → 
  (B^2 - m*B - 3 = 0) → 
  |A - B| = 4 := by
sorry

end parabola_intersection_length_l1882_188234


namespace tan_two_implies_specific_trig_ratio_l1882_188222

theorem tan_two_implies_specific_trig_ratio (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin θ * Real.sin (π/2 - θ)) / (Real.sin θ^2 + Real.cos (2*θ) + Real.cos θ^2) = 1/3 :=
by sorry

end tan_two_implies_specific_trig_ratio_l1882_188222


namespace cosine_identity_l1882_188270

theorem cosine_identity (a : Real) (h : 3 * Real.pi / 2 < a ∧ a < 2 * Real.pi) :
  Real.sqrt (1/2 + 1/2 * Real.sqrt (1/2 + 1/2 * Real.cos (2 * a))) = -Real.cos (a / 2) := by
  sorry

end cosine_identity_l1882_188270


namespace tan_value_from_trig_equation_l1882_188285

theorem tan_value_from_trig_equation (θ : Real) 
  (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 2 * Real.sin θ * Real.sin (θ + π / 4) = 5 * Real.cos (2 * θ)) : 
  Real.tan θ = 5 / 6 := by
sorry

end tan_value_from_trig_equation_l1882_188285


namespace purely_imaginary_complex_number_l1882_188275

theorem purely_imaginary_complex_number (a : ℝ) :
  let z : ℂ := a^2 - 4 + (a - 2) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → a = -2 := by
sorry

end purely_imaginary_complex_number_l1882_188275


namespace set_equality_implies_values_l1882_188265

theorem set_equality_implies_values (A B : Set ℝ) (x y : ℝ) :
  A = {3, 4, x} → B = {2, 3, y} → A = B → x = 2 ∧ y = 4 := by
  sorry

end set_equality_implies_values_l1882_188265


namespace sample_size_is_fifteen_l1882_188245

/-- Represents the stratified sampling scenario -/
structure StratifiedSampling where
  total_employees : ℕ
  young_employees : ℕ
  young_in_sample : ℕ

/-- Calculates the sample size for a given stratified sampling scenario -/
def sample_size (s : StratifiedSampling) : ℕ :=
  s.total_employees / (s.young_employees / s.young_in_sample)

/-- Theorem stating that the sample size is 15 for the given scenario -/
theorem sample_size_is_fifteen :
  let s : StratifiedSampling := {
    total_employees := 75,
    young_employees := 35,
    young_in_sample := 7
  }
  sample_size s = 15 := by
  sorry

end sample_size_is_fifteen_l1882_188245


namespace lagaan_collection_l1882_188269

/-- The total amount of lagaan collected from a village, given the payment of one farmer and their land proportion. -/
theorem lagaan_collection (farmer_payment : ℝ) (farmer_land_proportion : ℝ) 
  (h1 : farmer_payment = 480) 
  (h2 : farmer_land_proportion = 0.23255813953488372 / 100) : 
  (farmer_payment / farmer_land_proportion) = 206400000 := by
  sorry

end lagaan_collection_l1882_188269


namespace janous_inequality_l1882_188252

theorem janous_inequality (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + 2) * (b + 2) ≥ c * d ∧
  ∃ (a₀ b₀ c₀ d₀ : ℝ), a₀^2 + b₀^2 + c₀^2 + d₀^2 = 4 ∧ (a₀ + 2) * (b₀ + 2) = c₀ * d₀ := by
  sorry

#check janous_inequality

end janous_inequality_l1882_188252


namespace complex_number_location_l1882_188206

theorem complex_number_location (z : ℂ) : 
  z = Complex.mk (Real.sin (2019 * π / 180)) (Real.cos (2019 * π / 180)) →
  Real.sin (2019 * π / 180) < 0 ∧ Real.cos (2019 * π / 180) < 0 :=
by
  sorry

#check complex_number_location

end complex_number_location_l1882_188206


namespace average_of_first_50_even_numbers_l1882_188276

def first_even_number : ℕ := 2

def last_even_number (n : ℕ) : ℕ := first_even_number + 2 * (n - 1)

def average_of_arithmetic_sequence (a₁ a_n n : ℕ) : ℚ :=
  (a₁ + a_n : ℚ) / 2

theorem average_of_first_50_even_numbers :
  average_of_arithmetic_sequence first_even_number (last_even_number 50) 50 = 51 := by
  sorry

end average_of_first_50_even_numbers_l1882_188276


namespace fifth_sixth_sum_l1882_188236

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  sum_1_2 : a 1 + a 2 = 20
  sum_3_4 : a 3 + a 4 = 40

/-- The theorem stating that a₅ + a₆ = 80 for the given geometric sequence -/
theorem fifth_sixth_sum (seq : GeometricSequence) : seq.a 5 + seq.a 6 = 80 := by
  sorry

end fifth_sixth_sum_l1882_188236


namespace solution_to_linear_equation_l1882_188204

theorem solution_to_linear_equation :
  ∃ (x y : ℝ), 3 * x + 2 = 2 * y ∧ x = 2 ∧ y = 4 := by
  sorry

end solution_to_linear_equation_l1882_188204


namespace quadratic_inequality_solution_sets_l1882_188244

theorem quadratic_inequality_solution_sets (a : ℝ) :
  (∀ x, 6 * x^2 + a * x - a^2 < 0 ↔ 
    (a > 0 ∧ -a/2 < x ∧ x < a/3) ∨
    (a < 0 ∧ a/3 < x ∧ x < -a/2)) ∧
  (a = 0 → ∀ x, ¬(6 * x^2 + a * x - a^2 < 0)) :=
by sorry

end quadratic_inequality_solution_sets_l1882_188244


namespace volunteer_average_age_l1882_188262

theorem volunteer_average_age (total_members : ℕ) (teens : ℕ) (parents : ℕ) (volunteers : ℕ)
  (teen_avg_age : ℝ) (parent_avg_age : ℝ) (overall_avg_age : ℝ) :
  total_members = 50 →
  teens = 30 →
  parents = 15 →
  volunteers = 5 →
  teen_avg_age = 16 →
  parent_avg_age = 35 →
  overall_avg_age = 23 →
  (total_members : ℝ) * overall_avg_age = 
    (teens : ℝ) * teen_avg_age + (parents : ℝ) * parent_avg_age + (volunteers : ℝ) * ((total_members : ℝ) * overall_avg_age - (teens : ℝ) * teen_avg_age - (parents : ℝ) * parent_avg_age) / (volunteers : ℝ) →
  ((total_members : ℝ) * overall_avg_age - (teens : ℝ) * teen_avg_age - (parents : ℝ) * parent_avg_age) / (volunteers : ℝ) = 29 :=
by
  sorry

#check volunteer_average_age

end volunteer_average_age_l1882_188262


namespace average_weight_increase_l1882_188211

theorem average_weight_increase (original_group_size : ℕ) 
  (original_weight : ℝ) (new_weight : ℝ) : 
  original_group_size = 5 → 
  original_weight = 50 → 
  new_weight = 70 → 
  (new_weight - original_weight) / original_group_size = 4 := by
sorry

end average_weight_increase_l1882_188211


namespace negation_of_implication_l1882_188209

theorem negation_of_implication (a b c : ℝ) :
  ¬(a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) := by sorry

end negation_of_implication_l1882_188209


namespace bucket_capacity_l1882_188247

/-- Calculates the capacity of a bucket used to fill a pool -/
theorem bucket_capacity
  (fill_time : ℕ)           -- Time to fill and empty one bucket (in seconds)
  (pool_capacity : ℕ)       -- Capacity of the pool (in gallons)
  (total_time : ℕ)          -- Total time to fill the pool (in minutes)
  (h1 : fill_time = 20)     -- Given: Time to fill and empty one bucket is 20 seconds
  (h2 : pool_capacity = 84) -- Given: Pool capacity is 84 gallons
  (h3 : total_time = 14)    -- Given: Total time to fill the pool is 14 minutes
  : ℕ := by
  sorry

#check bucket_capacity

end bucket_capacity_l1882_188247


namespace extreme_point_implies_a_zero_l1882_188287

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2 - (a + 2) * x + 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * x - (a + 2)

theorem extreme_point_implies_a_zero :
  ∀ a : ℝ, (f_derivative a 1 = 0) → a = 0 :=
by sorry

#check extreme_point_implies_a_zero

end extreme_point_implies_a_zero_l1882_188287


namespace m_range_l1882_188294

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, |x - m| < 1 ↔ 1/3 < x ∧ x < 1/2) → 
  -1/2 ≤ m ∧ m ≤ 4/3 :=
by sorry

end m_range_l1882_188294


namespace polynomial_transformation_l1882_188253

/-- Given y = x + 1/x, prove that x^6 + x^5 - 5x^4 + x^3 + 3x^2 + x + 1 = 0 is equivalent to x^4*y^2 - 4*x^2*y^2 + 3*x^2 = 0 -/
theorem polynomial_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^6 + x^5 - 5*x^4 + x^3 + 3*x^2 + x + 1 = 0 ↔ x^4*y^2 - 4*x^2*y^2 + 3*x^2 = 0 :=
by sorry

end polynomial_transformation_l1882_188253


namespace equation_implies_equal_variables_l1882_188217

theorem equation_implies_equal_variables (a b : ℝ) 
  (h : (1 / (3 * a)) + (2 / (3 * b)) = 3 / (a + 2 * b)) : a = b :=
by sorry

end equation_implies_equal_variables_l1882_188217


namespace kevins_toads_l1882_188238

/-- The number of toads in Kevin's shoebox -/
def num_toads : ℕ := 8

/-- The number of worms each toad is fed daily -/
def worms_per_toad : ℕ := 3

/-- The time (in minutes) it takes Kevin to find each worm -/
def minutes_per_worm : ℕ := 15

/-- The time (in hours) it takes Kevin to find enough worms for all toads -/
def total_hours : ℕ := 6

/-- Theorem stating that the number of toads is 8 given the conditions -/
theorem kevins_toads : 
  num_toads = (total_hours * 60) / minutes_per_worm / worms_per_toad :=
by sorry

end kevins_toads_l1882_188238


namespace average_weight_problem_l1882_188284

/-- Given the average weights of pairs and the weight of one individual, 
    prove the average weight of all three. -/
theorem average_weight_problem (a b c : ℝ) 
    (h1 : (a + b) / 2 = 25) 
    (h2 : (b + c) / 2 = 28) 
    (h3 : b = 16) : 
    (a + b + c) / 3 = 30 := by
  sorry

end average_weight_problem_l1882_188284


namespace cricketer_average_score_l1882_188224

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (average_all : ℚ) 
  (average_last : ℚ) 
  (last_matches : ℕ) : 
  total_matches = 10 → 
  average_all = 389/10 → 
  average_last = 137/4 → 
  last_matches = 4 → 
  (total_matches * average_all - last_matches * average_last) / (total_matches - last_matches) = 42 := by
  sorry

end cricketer_average_score_l1882_188224


namespace sean_blocks_l1882_188233

theorem sean_blocks (initial_blocks : ℕ) (eaten_blocks : ℕ) (remaining_blocks : ℕ) : 
  initial_blocks = 55 → eaten_blocks = 29 → remaining_blocks = initial_blocks - eaten_blocks → 
  remaining_blocks = 26 := by
sorry

end sean_blocks_l1882_188233


namespace hyperbola_equation_l1882_188288

/-- Given a hyperbola with equation (x^2 / a^2) - (y^2 / b^2) = 1, where a > 0, b > 0,
    if one focus is at (2,0) and one asymptote has a slope of √3,
    then the equation of the hyperbola is x^2 - (y^2 / 3) = 1. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (2 : ℝ)^2 = a^2 + b^2 →
  b / a = Real.sqrt 3 →
  ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1 :=
by sorry

end hyperbola_equation_l1882_188288


namespace sequence_problem_l1882_188203

/-- Given a sequence of positive integers x₁, x₂, ..., x₇ satisfying
    x₆ = 144 and x_{n+3} = x_{n+2}(x_{n+1} + x_n) for n = 1, 2, 3, 4,
    prove that x₇ = 3456. -/
theorem sequence_problem (x : Fin 7 → ℕ+) 
    (h1 : x 6 = 144)
    (h2 : ∀ n : Fin 4, x (n + 3) = x (n + 2) * (x (n + 1) + x n)) :
  x 7 = 3456 := by
  sorry

end sequence_problem_l1882_188203


namespace six_by_six_grid_squares_l1882_188272

/-- The number of squares of size n×n in a grid of size m×m -/
def count_squares (n m : ℕ) : ℕ := (m - n + 1) * (m - n + 1)

/-- The total number of squares in a 6×6 grid -/
def total_squares : ℕ :=
  count_squares 1 6 + count_squares 2 6 + count_squares 3 6 + count_squares 4 6

theorem six_by_six_grid_squares :
  total_squares = 86 :=
sorry

end six_by_six_grid_squares_l1882_188272


namespace finite_common_terms_l1882_188201

/-- Two sequences of natural numbers with specific recurrence relations have only finitely many common terms -/
theorem finite_common_terms 
  (a b : ℕ → ℕ) 
  (ha : ∀ n : ℕ, n ≥ 1 → a (n + 1) = n * a n + 1)
  (hb : ∀ n : ℕ, n ≥ 1 → b (n + 1) = n * b n - 1) :
  Set.Finite {n : ℕ | ∃ m : ℕ, a n = b m} :=
sorry

end finite_common_terms_l1882_188201


namespace sara_apples_l1882_188228

theorem sara_apples (total : ℕ) (ali_ratio : ℕ) (sara_apples : ℕ) : 
  total = 80 →
  ali_ratio = 4 →
  total = sara_apples * (ali_ratio + 1) →
  sara_apples = 16 := by
sorry

end sara_apples_l1882_188228


namespace shoe_cost_comparison_l1882_188258

/-- Calculates the percentage increase in average cost per year of new shoes compared to repaired used shoes -/
theorem shoe_cost_comparison (used_repair_cost : ℝ) (used_lifespan : ℝ) (new_cost : ℝ) (new_lifespan : ℝ)
  (h1 : used_repair_cost = 11.50)
  (h2 : used_lifespan = 1)
  (h3 : new_cost = 28.00)
  (h4 : new_lifespan = 2)
  : (((new_cost / new_lifespan) - (used_repair_cost / used_lifespan)) / (used_repair_cost / used_lifespan)) * 100 = 21.74 := by
  sorry

end shoe_cost_comparison_l1882_188258


namespace prime_not_divides_difference_l1882_188239

theorem prime_not_divides_difference (a b c d p : ℕ) : 
  0 < a → 0 < b → 0 < c → 0 < d → 
  p = a + b + c + d → 
  Nat.Prime p → 
  ¬(p ∣ a * b - c * d) := by
sorry

end prime_not_divides_difference_l1882_188239


namespace new_revenue_is_354375_l1882_188207

/-- Calculates the total revenue at the new price given the conditions --/
def calculate_new_revenue (price_increase : ℕ) (sales_decrease : ℕ) (revenue_increase : ℕ) (new_sales : ℕ) : ℕ :=
  let original_sales := new_sales + sales_decrease
  let original_price := (revenue_increase + price_increase * new_sales) / sales_decrease
  let new_price := original_price + price_increase
  new_price * new_sales

/-- Theorem stating that the total revenue at the new price is $354,375 --/
theorem new_revenue_is_354375 :
  calculate_new_revenue 1000 8 26000 63 = 354375 := by
  sorry

end new_revenue_is_354375_l1882_188207


namespace a_8_value_l1882_188278

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem a_8_value (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 6 + a 10 = -6) →
  (a 6 * a 10 = 2) →
  (a 6 < 0) →
  (a 10 < 0) →
  a 8 = -Real.sqrt 2 := by
  sorry

end a_8_value_l1882_188278


namespace first_reduction_percentage_l1882_188254

theorem first_reduction_percentage (x : ℝ) : 
  (1 - x / 100) * (1 - 0.1) = 0.765 → x = 15 := by
  sorry

end first_reduction_percentage_l1882_188254


namespace initial_students_count_l1882_188219

/-- The number of students who got off the bus at the first stop -/
def students_off : ℕ := 3

/-- The number of students remaining on the bus after the first stop -/
def students_remaining : ℕ := 7

/-- The initial number of students on the bus -/
def initial_students : ℕ := students_remaining + students_off

theorem initial_students_count : initial_students = 10 := by sorry

end initial_students_count_l1882_188219


namespace min_value_reciprocal_sum_l1882_188246

theorem min_value_reciprocal_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  (1/a + 1/b + 1/c ≥ 9) ∧ 
  (1/a + 1/b + 1/c = 9 ↔ a = 1/3 ∧ b = 1/3 ∧ c = 1/3) :=
sorry

end min_value_reciprocal_sum_l1882_188246


namespace largest_common_value_of_aps_l1882_188296

/-- The largest common value less than 300 between two arithmetic progressions -/
theorem largest_common_value_of_aps : ∃ (n m : ℕ),
  7 * (n + 1) = 5 + 10 * m ∧
  7 * (n + 1) < 300 ∧
  ∀ (k l : ℕ), 7 * (k + 1) = 5 + 10 * l → 7 * (k + 1) < 300 → 7 * (k + 1) ≤ 7 * (n + 1) :=
by sorry

end largest_common_value_of_aps_l1882_188296


namespace lost_shoes_count_l1882_188264

/-- Given a number of initial shoe pairs and remaining matching pairs,
    calculates the number of individual shoes lost. -/
def shoes_lost (initial_pairs : ℕ) (remaining_pairs : ℕ) : ℕ :=
  2 * initial_pairs - 2 * remaining_pairs

/-- Theorem stating that with 24 initial pairs and 19 remaining pairs,
    10 individual shoes were lost. -/
theorem lost_shoes_count :
  shoes_lost 24 19 = 10 := by
  sorry


end lost_shoes_count_l1882_188264


namespace rectangular_prism_diagonal_l1882_188273

/-- The length of the diagonal of a rectangular prism with dimensions 12, 16, and 21 -/
def prism_diagonal : ℝ := 29

/-- Theorem: The diagonal of a rectangular prism with dimensions 12, 16, and 21 is 29 -/
theorem rectangular_prism_diagonal :
  let a : ℝ := 12
  let b : ℝ := 16
  let c : ℝ := 21
  Real.sqrt (a^2 + b^2 + c^2) = prism_diagonal :=
by sorry

end rectangular_prism_diagonal_l1882_188273


namespace marking_implies_prime_f_1997_l1882_188283

/-- Represents the marking procedure on a 2N-gon -/
def mark_procedure (N : ℕ) : Set ℕ := sorry

/-- The function f(N) that counts non-marked vertices -/
def f (N : ℕ) : ℕ := sorry

/-- Main theorem: If f(N) = 0, then 2N + 1 is prime -/
theorem marking_implies_prime (N : ℕ) (h1 : N > 2) (h2 : f N = 0) : Nat.Prime (2 * N + 1) := by
  sorry

/-- Computation of f(1997) -/
theorem f_1997 : f 1997 = 3810 := by
  sorry

end marking_implies_prime_f_1997_l1882_188283


namespace missing_number_odd_l1882_188260

def set_a : Finset Nat := {11, 44, 55}

def is_odd (n : Nat) : Prop := n % 2 = 1

def probability_even_sum (b : Nat) : Rat :=
  (set_a.filter (fun a => (a + b) % 2 = 0)).card / set_a.card

theorem missing_number_odd (b : Nat) :
  probability_even_sum b = 1/2 → is_odd b := by
  sorry

end missing_number_odd_l1882_188260


namespace largest_k_for_tree_graph_condition_l1882_188218

/-- A tree graph with k vertices -/
structure TreeGraph (k : ℕ) where
  (vertices : Finset (Fin k))
  (edges : Finset (Fin k × Fin k))
  -- Add properties to ensure it's a tree

/-- Path between two vertices in a graph -/
def path (G : TreeGraph k) (u v : Fin k) : Finset (Fin k) := sorry

/-- Length of a path -/
def pathLength (p : Finset (Fin k)) : ℕ := sorry

/-- The condition for the existence of vertices u and v -/
def satisfiesCondition (G : TreeGraph k) (m n : ℕ) : Prop :=
  ∃ u v : Fin k, ∀ w : Fin k, 
    (∃ p : Finset (Fin k), p = path G u w ∧ pathLength p ≤ m) ∨
    (∃ p : Finset (Fin k), p = path G v w ∧ pathLength p ≤ n)

theorem largest_k_for_tree_graph_condition (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (∀ k : ℕ, k ≤ min (2*n + 2*m + 2) (3*n + 2) → 
    ∀ G : TreeGraph k, satisfiesCondition G m n) ∧
  (∀ k : ℕ, k > min (2*n + 2*m + 2) (3*n + 2) → 
    ∃ G : TreeGraph k, ¬satisfiesCondition G m n) :=
sorry

end largest_k_for_tree_graph_condition_l1882_188218


namespace sqrt_product_simplification_l1882_188271

theorem sqrt_product_simplification (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) :=
by sorry

end sqrt_product_simplification_l1882_188271


namespace f_has_two_zeros_l1882_188215

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x - 3 else -2 + Real.log x

-- State the theorem
theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end f_has_two_zeros_l1882_188215


namespace price_and_distance_proportions_l1882_188299

-- Define the relationships
def inverse_proportion (x y : ℝ) (k : ℝ) : Prop := x * y = k
def direct_proportion (x y : ℝ) (k : ℝ) : Prop := x / y = k

-- State the theorem
theorem price_and_distance_proportions :
  -- For any positive real numbers representing unit price, quantity, and total price
  ∀ (unit_price quantity total_price : ℝ) (hp : unit_price > 0) (hq : quantity > 0) (ht : total_price > 0),
  -- When the total price is fixed
  (unit_price * quantity = total_price) →
  -- The unit price and quantity are in inverse proportion
  inverse_proportion unit_price quantity total_price ∧
  -- For any positive real numbers representing map distance, actual distance, and scale
  ∀ (map_distance actual_distance scale : ℝ) (hm : map_distance > 0) (ha : actual_distance > 0) (hs : scale > 0),
  -- When the scale is fixed
  (map_distance / actual_distance = scale) →
  -- The map distance and actual distance are in direct proportion
  direct_proportion map_distance actual_distance scale :=
by sorry

end price_and_distance_proportions_l1882_188299


namespace vector_magnitude_l1882_188295

def a : ℝ × ℝ := (2, 1)

theorem vector_magnitude (b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = 10)
  (h2 : Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5 * Real.sqrt 2) :
  Real.sqrt (b.1^2 + b.2^2) = 5 := by
  sorry

end vector_magnitude_l1882_188295


namespace ninth_term_of_geometric_sequence_l1882_188221

/-- A geometric sequence with a₃ = 16 and a₆ = 144 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ 
  (∀ n : ℕ, a (n + 1) = a n * r) ∧
  a 3 = 16 ∧ 
  a 6 = 144

theorem ninth_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h : geometric_sequence a) : 
  a 9 = 1296 := by
sorry

end ninth_term_of_geometric_sequence_l1882_188221


namespace sum_of_compositions_l1882_188286

def p (x : ℝ) : ℝ := x^2 - 3

def q (x : ℝ) : ℝ := x - 2

def x_values : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_compositions : 
  (x_values.map (λ x => q (p x))).sum = 15 := by sorry

end sum_of_compositions_l1882_188286


namespace flight_duration_is_two_hours_l1882_188213

/-- Calculates the flight duration in hours given the number of peanut bags, 
    peanuts per bag, and consumption rate. -/
def flight_duration (bags : ℕ) (peanuts_per_bag : ℕ) (minutes_per_peanut : ℕ) : ℚ :=
  (bags * peanuts_per_bag * minutes_per_peanut) / 60

/-- Proves that the flight duration is 2 hours given the specified conditions. -/
theorem flight_duration_is_two_hours : 
  flight_duration 4 30 1 = 2 := by
  sorry

#eval flight_duration 4 30 1

end flight_duration_is_two_hours_l1882_188213


namespace modified_prism_surface_area_difference_l1882_188282

/-- Calculates the surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Calculates the surface area added by removing a cube from the center of a face -/
def added_surface_area (cube_side : ℝ) : ℝ := 5 * cube_side^2

theorem modified_prism_surface_area_difference :
  let original_sa := surface_area 2 4 5
  let modified_sa := original_sa + added_surface_area 1
  modified_sa - original_sa = 5 := by sorry

end modified_prism_surface_area_difference_l1882_188282


namespace sin_cos_values_l1882_188267

theorem sin_cos_values (α : Real) (h : Real.sin α + 3 * Real.cos α = 0) :
  (Real.sin α = 3 * (Real.sqrt 10) / 10 ∧ Real.cos α = -(Real.sqrt 10) / 10) ∨
  (Real.sin α = -(3 * (Real.sqrt 10) / 10) ∧ Real.cos α = (Real.sqrt 10) / 10) := by
  sorry

end sin_cos_values_l1882_188267


namespace rectangle_to_triangle_altitude_l1882_188298

/-- A rectangle with width 7 and length 21 can be rearranged into a triangle with altitude 14 -/
theorem rectangle_to_triangle_altitude (w h b : ℝ) : 
  w = 7 → h = 21 → b = 21 → 
  ∃ (altitude : ℝ), 
    w * h = (1/2) * b * altitude ∧ 
    altitude = 14 := by
  sorry

end rectangle_to_triangle_altitude_l1882_188298


namespace tens_digit_of_power_five_l1882_188226

theorem tens_digit_of_power_five : ∃ (n : ℕ), 5^(5^5) ≡ 25 [MOD 100] ∧ n = 2 := by sorry

end tens_digit_of_power_five_l1882_188226


namespace mother_three_times_daughter_age_l1882_188268

/-- Proves that the number of years until the mother is three times as old as her daughter is 9,
    given that the mother is currently 42 years old and the daughter is currently 8 years old. -/
theorem mother_three_times_daughter_age (mother_age : ℕ) (daughter_age : ℕ) 
  (h1 : mother_age = 42) (h2 : daughter_age = 8) : 
  ∃ (years : ℕ), mother_age + years = 3 * (daughter_age + years) ∧ years = 9 := by
  sorry

end mother_three_times_daughter_age_l1882_188268


namespace power_division_equality_l1882_188251

theorem power_division_equality : 8^15 / 64^3 = 8^9 := by sorry

end power_division_equality_l1882_188251


namespace rectangle_area_l1882_188229

/-- The area of a rectangle with length thrice its breadth and perimeter 104 meters is 507 square meters. -/
theorem rectangle_area (breadth length perimeter area : ℝ) : 
  length = 3 * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 104 →
  area = length * breadth →
  area = 507 := by
sorry

end rectangle_area_l1882_188229


namespace max_B_bins_l1882_188291

/-- The cost of an A brand garbage bin in yuan -/
def cost_A : ℕ := 120

/-- The cost of a B brand garbage bin in yuan -/
def cost_B : ℕ := 150

/-- The total number of garbage bins to be purchased -/
def total_bins : ℕ := 30

/-- The maximum budget in yuan -/
def max_budget : ℕ := 4000

/-- Theorem stating the maximum number of B brand bins that can be purchased -/
theorem max_B_bins : 
  ∀ m : ℕ, 
  m ≤ total_bins ∧ 
  cost_B * m + cost_A * (total_bins - m) ≤ max_budget →
  m ≤ 13 :=
by sorry

end max_B_bins_l1882_188291


namespace eleventh_number_with_digit_sum_13_l1882_188280

/-- A function that returns the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nth_number_with_digit_sum_13 (n : ℕ+) : ℕ+ := sorry

/-- The theorem stating that the 11th number with digit sum 13 is 175 -/
theorem eleventh_number_with_digit_sum_13 : 
  nth_number_with_digit_sum_13 11 = 175 := by sorry

end eleventh_number_with_digit_sum_13_l1882_188280


namespace expression_value_l1882_188210

theorem expression_value : 
  Real.sqrt (2018 * 2021 * 2022 * 2023 + 2024^2) - 2024^2 = -12138 := by
  sorry

end expression_value_l1882_188210


namespace luke_good_games_l1882_188242

def budget : ℕ := 100
def price_a : ℕ := 15
def price_b : ℕ := 8
def price_c : ℕ := 6
def num_a : ℕ := 3
def num_b : ℕ := 5
def sold_games : ℕ := 2
def sold_price : ℕ := 12
def broken_a : ℕ := 3
def broken_b : ℕ := 2

def remaining_budget : ℕ := budget - (num_a * price_a + num_b * price_b) + (sold_games * sold_price)

def num_c : ℕ := remaining_budget / price_c

theorem luke_good_games : 
  (num_a - broken_a) + (num_b - broken_b) + num_c = 9 :=
sorry

end luke_good_games_l1882_188242


namespace investment_interest_calculation_l1882_188257

theorem investment_interest_calculation
  (total_investment : ℝ)
  (investment_at_6_percent : ℝ)
  (interest_rate_6_percent : ℝ)
  (interest_rate_9_percent : ℝ)
  (h1 : total_investment = 10000)
  (h2 : investment_at_6_percent = 7200)
  (h3 : interest_rate_6_percent = 0.06)
  (h4 : interest_rate_9_percent = 0.09) :
  let investment_at_9_percent := total_investment - investment_at_6_percent
  let interest_from_6_percent := investment_at_6_percent * interest_rate_6_percent
  let interest_from_9_percent := investment_at_9_percent * interest_rate_9_percent
  let total_interest := interest_from_6_percent + interest_from_9_percent
  total_interest = 684 := by sorry

end investment_interest_calculation_l1882_188257


namespace pages_read_second_day_l1882_188208

theorem pages_read_second_day 
  (total_pages : ℕ) 
  (pages_first_day : ℕ) 
  (pages_left : ℕ) 
  (h1 : total_pages = 95) 
  (h2 : pages_first_day = 18) 
  (h3 : pages_left = 19) : 
  total_pages - pages_left - pages_first_day = 58 := by
  sorry

end pages_read_second_day_l1882_188208


namespace divisibility_by_33_l1882_188256

def five_digit_number (n : ℕ) : ℕ := 70000 + 1000 * n + 933

theorem divisibility_by_33 (n : ℕ) : 
  n < 10 → (five_digit_number n % 33 = 0 ↔ n = 5) := by
  sorry

end divisibility_by_33_l1882_188256


namespace arithmetic_sequence_n_equals_15_l1882_188227

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  third_term : a 3 = 5
  sum_2_5 : a 2 + a 5 = 12
  nth_term : ∃ n, a n = 29

/-- The theorem stating that n = 15 for the given arithmetic sequence -/
theorem arithmetic_sequence_n_equals_15 (seq : ArithmeticSequence) : 
  ∃ n, seq.a n = 29 ∧ n = 15 := by
  sorry

end arithmetic_sequence_n_equals_15_l1882_188227


namespace number_of_brown_dogs_l1882_188263

/-- Given a group of dogs with white, black, and brown colors, 
    prove that the number of brown dogs is 20. -/
theorem number_of_brown_dogs 
  (total : ℕ) 
  (white : ℕ) 
  (black : ℕ) 
  (h1 : total = 45) 
  (h2 : white = 10) 
  (h3 : black = 15) : 
  total - (white + black) = 20 := by
  sorry

end number_of_brown_dogs_l1882_188263


namespace arithmetic_sequence_sin_value_l1882_188231

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sin_value (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 6 = 3 * Real.pi / 2 →
  Real.sin (2 * a 4 - Real.pi / 3) = -1 / 2 := by
  sorry

end arithmetic_sequence_sin_value_l1882_188231


namespace bus_ticket_impossibility_prove_bus_ticket_impossibility_l1882_188225

theorem bus_ticket_impossibility 
  (num_passengers : ℕ) 
  (ticket_price : ℕ) 
  (coin_denominations : List ℕ) 
  (total_coins : ℕ) : Prop :=
  num_passengers = 40 →
  ticket_price = 5 →
  coin_denominations = [10, 15, 20] →
  total_coins = 49 →
  ¬∃ (payment : List ℕ),
    payment.sum = num_passengers * ticket_price ∧
    payment.length ≤ total_coins - num_passengers ∧
    ∀ c ∈ payment, c ∈ coin_denominations

theorem prove_bus_ticket_impossibility : 
  bus_ticket_impossibility 40 5 [10, 15, 20] 49 := by
  sorry

end bus_ticket_impossibility_prove_bus_ticket_impossibility_l1882_188225


namespace fraction_sum_theorem_l1882_188261

theorem fraction_sum_theorem (a b c d x y z w : ℝ) 
  (h1 : x / a + y / b + z / c + w / d = 4)
  (h2 : a / x + b / y + c / z + d / w = 0) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 + w^2 / d^2 = 16 := by
  sorry

end fraction_sum_theorem_l1882_188261


namespace pear_sales_ratio_l1882_188216

/-- Given the total pears sold and the amount sold in the afternoon, 
    prove the ratio of afternoon sales to morning sales. -/
theorem pear_sales_ratio 
  (total_pears : ℕ) 
  (afternoon_pears : ℕ) 
  (h1 : total_pears = 480)
  (h2 : afternoon_pears = 320) :
  afternoon_pears / (total_pears - afternoon_pears) = 2 := by
  sorry

end pear_sales_ratio_l1882_188216


namespace instrument_players_fraction_l1882_188214

theorem instrument_players_fraction (total : ℕ) (two_or_more : ℕ) (prob_exactly_one : ℚ) :
  total = 800 →
  two_or_more = 64 →
  prob_exactly_one = 12 / 100 →
  (prob_exactly_one * total + two_or_more : ℚ) / total = 1 / 5 := by
  sorry

end instrument_players_fraction_l1882_188214


namespace jerrys_collection_cost_l1882_188243

/-- The amount of money Jerry needs to finish his action figure collection -/
def jerrysMoney (currentFigures : ℕ) (totalRequired : ℕ) (costPerFigure : ℕ) : ℕ :=
  (totalRequired - currentFigures) * costPerFigure

/-- Proof that Jerry needs $72 to finish his collection -/
theorem jerrys_collection_cost : jerrysMoney 7 16 8 = 72 := by
  sorry

end jerrys_collection_cost_l1882_188243


namespace recurrence_sequence_property_l1882_188281

/-- A sequence of integers satisfying the recurrence relation a_{n+2} = a_{n+1} - m * a_n -/
def RecurrenceSequence (m : ℤ) (a : ℕ → ℤ) : Prop :=
  (a 1 ≠ 0 ∨ a 2 ≠ 0) ∧ ∀ n : ℕ, a (n + 2) = a (n + 1) - m * a n

/-- The main theorem -/
theorem recurrence_sequence_property (m : ℤ) (a : ℕ → ℤ) 
    (hm : |m| ≥ 2) 
    (ha : RecurrenceSequence m a) 
    (r s : ℕ) 
    (hrs : r > s ∧ s ≥ 2) 
    (heq : a r = a s ∧ a s = a 1) : 
  r - s ≥ |m| := by
  sorry

end recurrence_sequence_property_l1882_188281


namespace sin_30_sin_75_minus_sin_60_cos_105_l1882_188223

theorem sin_30_sin_75_minus_sin_60_cos_105 :
  Real.sin (30 * π / 180) * Real.sin (75 * π / 180) -
  Real.sin (60 * π / 180) * Real.cos (105 * π / 180) =
  Real.sqrt 2 / 2 := by
  sorry

end sin_30_sin_75_minus_sin_60_cos_105_l1882_188223


namespace cookie_distribution_l1882_188297

theorem cookie_distribution (x y z : ℚ) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x * (y / z) = 35 →
  (35 : ℚ) / 5 = 7 := by
  sorry

end cookie_distribution_l1882_188297


namespace julie_weed_hours_l1882_188249

/-- Represents Julie's landscaping business earnings --/
def julie_earnings (weed_hours : ℕ) : ℕ :=
  let mowing_rate : ℕ := 4
  let weed_rate : ℕ := 8
  let mowing_hours : ℕ := 25
  2 * (mowing_rate * mowing_hours + weed_rate * weed_hours)

/-- Proves that Julie spent 3 hours pulling weeds in September --/
theorem julie_weed_hours : 
  ∃ (weed_hours : ℕ), julie_earnings weed_hours = 248 ∧ weed_hours = 3 :=
by
  sorry

end julie_weed_hours_l1882_188249


namespace cafe_tables_l1882_188289

theorem cafe_tables (outdoor_tables : ℕ) (indoor_chairs : ℕ) (outdoor_chairs : ℕ) (total_chairs : ℕ) :
  outdoor_tables = 11 →
  indoor_chairs = 10 →
  outdoor_chairs = 3 →
  total_chairs = 123 →
  ∃ indoor_tables : ℕ, indoor_tables * indoor_chairs + outdoor_tables * outdoor_chairs = total_chairs ∧ indoor_tables = 9 :=
by sorry

end cafe_tables_l1882_188289
