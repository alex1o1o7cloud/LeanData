import Mathlib

namespace NUMINAMATH_CALUDE_hardcover_books_purchased_l2274_227450

/-- The number of hardcover books purchased -/
def num_hardcover : ℕ := 8

/-- The number of paperback books purchased -/
def num_paperback : ℕ := 12 - num_hardcover

/-- The price of a paperback book -/
def price_paperback : ℕ := 18

/-- The price of a hardcover book -/
def price_hardcover : ℕ := 30

/-- The total amount spent -/
def total_spent : ℕ := 312

/-- Theorem stating that the number of hardcover books purchased is 8 -/
theorem hardcover_books_purchased :
  num_hardcover = 8 ∧
  num_hardcover + num_paperback = 12 ∧
  price_hardcover * num_hardcover + price_paperback * num_paperback = total_spent :=
by sorry

end NUMINAMATH_CALUDE_hardcover_books_purchased_l2274_227450


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2274_227408

/-- The coefficient of x^2 in the expansion of (2x+1)^5 is 40 -/
theorem coefficient_x_squared_in_expansion : 
  (Finset.range 6).sum (fun k => 
    Nat.choose 5 k * (2^(5-k)) * (1^k) * 
    if 5 - k = 2 then 1 else 0) = 40 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2274_227408


namespace NUMINAMATH_CALUDE_square_inequality_l2274_227421

theorem square_inequality (x : ℝ) : (x^2 + x + 1)^2 ≤ 3 * (x^4 + x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l2274_227421


namespace NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l2274_227473

theorem unique_solution_ceiling_equation :
  ∃! b : ℝ, b + ⌈b⌉ = 17.8 ∧ b = 8.8 := by sorry

end NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l2274_227473


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2274_227468

theorem smallest_positive_integer_with_remainders : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 4 = 1) ∧ 
  (n % 5 = 2) ∧ 
  (n % 6 = 3) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 2 ∧ m % 6 = 3 → m ≥ n) ∧
  n = 21 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2274_227468


namespace NUMINAMATH_CALUDE_complex_sum_to_polar_l2274_227461

theorem complex_sum_to_polar : ∃ (r θ : ℝ), 
  5 * Complex.exp (3 * Real.pi * Complex.I / 4) + 5 * Complex.exp (-3 * Real.pi * Complex.I / 4) = r * Complex.exp (θ * Complex.I) ∧ 
  r = -5 * Real.sqrt 2 ∧ 
  θ = Real.pi := by
sorry

end NUMINAMATH_CALUDE_complex_sum_to_polar_l2274_227461


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2274_227480

theorem quadratic_equation_solution (x : ℝ) :
  2 * x^2 + 2 * x - 1 = 0 ↔ x = (-1 + Real.sqrt 3) / 2 ∨ x = (-1 - Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2274_227480


namespace NUMINAMATH_CALUDE_melissa_driving_hours_l2274_227472

/-- Calculates the total driving hours in a year given the number of trips per month,
    hours per trip, and months in a year. -/
def annual_driving_hours (trips_per_month : ℕ) (hours_per_trip : ℕ) (months_in_year : ℕ) : ℕ :=
  trips_per_month * hours_per_trip * months_in_year

/-- Proves that Melissa spends 72 hours driving in a year given the specified conditions. -/
theorem melissa_driving_hours :
  let trips_per_month : ℕ := 2
  let hours_per_trip : ℕ := 3
  let months_in_year : ℕ := 12
  annual_driving_hours trips_per_month hours_per_trip months_in_year = 72 :=
by
  sorry


end NUMINAMATH_CALUDE_melissa_driving_hours_l2274_227472


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2274_227435

/-- Given two vectors a and b in ℝ², where a = (x, x+1) and b = (1, 2),
    if a is perpendicular to b, then x = -2/3 -/
theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![x, x + 1]
  let b : Fin 2 → ℝ := ![1, 2]
  (∀ i, i < 2 → a i * b i = 0) → x = -2/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2274_227435


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2274_227424

theorem complex_equation_sum (a b : ℝ) :
  (1 - Complex.I) * (a + Complex.I) = 3 - b * Complex.I →
  a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2274_227424


namespace NUMINAMATH_CALUDE_translation_problem_l2274_227479

/-- A translation in the complex plane. -/
def Translation (w : ℂ) : ℂ → ℂ := fun z ↦ z + w

/-- The theorem statement -/
theorem translation_problem (T : ℂ → ℂ) (h : T (1 + 3*I) = 4 + 6*I) :
  T (2 - I) = 5 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_translation_problem_l2274_227479


namespace NUMINAMATH_CALUDE_line_slope_angle_l2274_227490

theorem line_slope_angle : 
  let x : ℝ → ℝ := λ t => 3 + t * Real.sin (π / 6)
  let y : ℝ → ℝ := λ t => -t * Real.cos (π / 6)
  (∃ m : ℝ, ∀ t₁ t₂ : ℝ, t₁ ≠ t₂ → 
    (y t₂ - y t₁) / (x t₂ - x t₁) = m ∧ 
    Real.arctan m = 2 * π / 3) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_angle_l2274_227490


namespace NUMINAMATH_CALUDE_infinite_solutions_l2274_227463

/-- The equation that x, y, and z must satisfy -/
def satisfies_equation (x y z : ℕ+) : Prop :=
  (x + y + z)^2 + 2*(x + y + z) = 5*(x*y + y*z + z*x)

/-- The set of all positive integer solutions to the equation -/
def solution_set : Set (ℕ+ × ℕ+ × ℕ+) :=
  {xyz | satisfies_equation xyz.1 xyz.2.1 xyz.2.2}

/-- The main theorem stating that the solution set is infinite -/
theorem infinite_solutions : Set.Infinite solution_set := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_l2274_227463


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2274_227436

theorem arithmetic_calculations :
  ((-3) - (-5) - 6 + (-4) = -8) ∧
  ((1/9 + 1/6 - 1/2) / (-1/18) = 4) ∧
  (-1^4 + |3-6| - 2 * (-2)^2 = -6) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2274_227436


namespace NUMINAMATH_CALUDE_smaller_number_problem_l2274_227418

theorem smaller_number_problem (x y : ℝ) : 
  y = 3 * x + 11 → x + y = 55 → x = 11 := by sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l2274_227418


namespace NUMINAMATH_CALUDE_chord_bisected_by_point_one_one_l2274_227471

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define a chord of the ellipse
def is_chord (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  is_on_ellipse x₁ y₁ ∧ is_on_ellipse x₂ y₂

-- Define the midpoint of a chord
def is_midpoint (x y x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2

-- Define a line equation
def line_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Theorem statement
theorem chord_bisected_by_point_one_one :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  is_chord x₁ y₁ x₂ y₂ →
  is_midpoint 1 1 x₁ y₁ x₂ y₂ →
  line_equation 4 9 (-13) x₁ y₁ ∧ line_equation 4 9 (-13) x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_chord_bisected_by_point_one_one_l2274_227471


namespace NUMINAMATH_CALUDE_harry_green_weights_l2274_227438

/-- Represents the weight configuration of Harry's custom creation at the gym -/
structure WeightConfiguration where
  blue_weight : ℕ        -- Weight of each blue weight in pounds
  green_weight : ℕ       -- Weight of each green weight in pounds
  bar_weight : ℕ         -- Weight of the bar in pounds
  num_blue : ℕ           -- Number of blue weights used
  total_weight : ℕ       -- Total weight of the custom creation in pounds

/-- Calculates the number of green weights in Harry's custom creation -/
def num_green_weights (config : WeightConfiguration) : ℕ :=
  (config.total_weight - config.bar_weight - config.num_blue * config.blue_weight) / config.green_weight

/-- Theorem stating that Harry put 5 green weights on the bar -/
theorem harry_green_weights :
  let config : WeightConfiguration := {
    blue_weight := 2,
    green_weight := 3,
    bar_weight := 2,
    num_blue := 4,
    total_weight := 25
  }
  num_green_weights config = 5 := by
  sorry

end NUMINAMATH_CALUDE_harry_green_weights_l2274_227438


namespace NUMINAMATH_CALUDE_ratio_common_value_l2274_227464

theorem ratio_common_value (x y z : ℝ) (k : ℝ) 
  (h1 : (x + y) / z = k)
  (h2 : (x + z) / y = k)
  (h3 : (y + z) / x = k)
  (h4 : x ≠ 0)
  (h5 : y ≠ 0)
  (h6 : z ≠ 0) :
  k = -1 ∨ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_common_value_l2274_227464


namespace NUMINAMATH_CALUDE_slope_angle_of_negative_sqrt3_over_3_l2274_227409

/-- The slope angle of a line with slope -√3/3 is 5π/6 -/
theorem slope_angle_of_negative_sqrt3_over_3 :
  let slope : ℝ := -Real.sqrt 3 / 3
  let slope_angle : ℝ := Real.arctan slope
  slope_angle = 5 * Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_slope_angle_of_negative_sqrt3_over_3_l2274_227409


namespace NUMINAMATH_CALUDE_inequality_proof_l2274_227476

theorem inequality_proof (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2274_227476


namespace NUMINAMATH_CALUDE_gcd_1151_3079_l2274_227440

theorem gcd_1151_3079 : Nat.gcd 1151 3079 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1151_3079_l2274_227440


namespace NUMINAMATH_CALUDE_monica_savings_l2274_227430

theorem monica_savings (weekly_saving : ℕ) (weeks_per_cycle : ℕ) (num_cycles : ℕ) 
  (h1 : weekly_saving = 15)
  (h2 : weeks_per_cycle = 60)
  (h3 : num_cycles = 5) :
  weekly_saving * weeks_per_cycle * num_cycles = 4500 := by
  sorry

end NUMINAMATH_CALUDE_monica_savings_l2274_227430


namespace NUMINAMATH_CALUDE_calculate_expression_l2274_227459

theorem calculate_expression : (81 : ℝ) ^ (1/4) * (81 : ℝ) ^ (1/5) * 2 = 20.09 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2274_227459


namespace NUMINAMATH_CALUDE_installation_charge_company_x_l2274_227478

/-- Represents a company's pricing for an air conditioner --/
structure CompanyPricing where
  price : ℝ
  surcharge_rate : ℝ
  installation_charge : ℝ

/-- Calculates the total cost for a company --/
def total_cost (c : CompanyPricing) : ℝ :=
  c.price + c.price * c.surcharge_rate + c.installation_charge

theorem installation_charge_company_x (
  company_x : CompanyPricing)
  (company_y : CompanyPricing)
  (h1 : company_x.price = 575)
  (h2 : company_x.surcharge_rate = 0.04)
  (h3 : company_y.price = 530)
  (h4 : company_y.surcharge_rate = 0.03)
  (h5 : company_y.installation_charge = 93)
  (h6 : total_cost company_x - total_cost company_y = 41.60) :
  company_x.installation_charge = 82.50 := by
  sorry

end NUMINAMATH_CALUDE_installation_charge_company_x_l2274_227478


namespace NUMINAMATH_CALUDE_semicircle_perimeter_specific_semicircle_perimeter_l2274_227467

/-- The perimeter of a semi-circle with radius r is equal to π * r + 2 * r -/
theorem semicircle_perimeter (r : ℝ) (h : r > 0) :
  let perimeter := π * r + 2 * r
  perimeter = π * r + 2 * r :=
by sorry

/-- The perimeter of a semi-circle with radius 6.7 cm is approximately 34.45 cm -/
theorem specific_semicircle_perimeter :
  let r : ℝ := 6.7
  let perimeter := π * r + 2 * r
  ∃ ε > 0, |perimeter - 34.45| < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_specific_semicircle_perimeter_l2274_227467


namespace NUMINAMATH_CALUDE_parallel_line_distance_l2274_227457

/-- Represents a circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- Radius of the circle -/
  radius : ℝ
  /-- Distance between adjacent parallel lines -/
  line_distance : ℝ
  /-- Length of the first chord -/
  chord1 : ℝ
  /-- Length of the second chord -/
  chord2 : ℝ
  /-- Length of the third chord -/
  chord3 : ℝ
  /-- Assertion that the first and third chords are equal -/
  chord1_eq_chord3 : chord1 = chord3
  /-- Assertion that the first chord has length 42 -/
  chord1_length : chord1 = 42
  /-- Assertion that the second chord has length 40 -/
  chord2_length : chord2 = 40

/-- Theorem stating that the distance between adjacent parallel lines is √(92/11) -/
theorem parallel_line_distance (c : CircleWithParallelLines) : 
  c.line_distance = Real.sqrt (92 / 11) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_distance_l2274_227457


namespace NUMINAMATH_CALUDE_base_conversion_and_arithmetic_l2274_227499

-- Define the base conversion functions
def to_base_10 (digits : List Nat) (base : Nat) : Rat :=
  (digits.reverse.enum.map (λ (i, d) => d * base^i)).sum

-- Define the given numbers in their respective bases
def num1 : Rat := 2468
def num2 : Rat := to_base_10 [1, 2, 1] 3
def num3 : Rat := to_base_10 [6, 5, 4, 3] 7
def num4 : Rat := to_base_10 [6, 7, 8, 9] 9

-- State the theorem
theorem base_conversion_and_arithmetic :
  num1 / num2 + num3 - num4 = -5857.75 := by sorry

end NUMINAMATH_CALUDE_base_conversion_and_arithmetic_l2274_227499


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l2274_227486

/-- The number of dogwood trees currently in the park -/
def current_trees : ℕ := 39

/-- The number of trees to be planted today -/
def planted_today : ℕ := 41

/-- The number of trees to be planted tomorrow -/
def planted_tomorrow : ℕ := 20

/-- The total number of trees after planting -/
def total_trees : ℕ := 100

/-- Theorem stating that the current number of trees plus the trees to be planted
    equals the total number of trees after planting -/
theorem dogwood_tree_count : 
  current_trees + planted_today + planted_tomorrow = total_trees := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l2274_227486


namespace NUMINAMATH_CALUDE_margin_formula_in_terms_of_selling_price_l2274_227429

/-- Prove that the margin formula can be expressed in terms of selling price -/
theorem margin_formula_in_terms_of_selling_price 
  (n : ℝ) (C S M : ℝ) 
  (h1 : M = (C + S) / n) 
  (h2 : M = S - C) : 
  M = 2 * S / (n + 1) :=
sorry

end NUMINAMATH_CALUDE_margin_formula_in_terms_of_selling_price_l2274_227429


namespace NUMINAMATH_CALUDE_track_length_is_360_l2274_227489

/-- Represents a circular running track with two runners -/
structure RunningTrack where
  length : ℝ
  sally_first_meeting : ℝ
  john_second_meeting : ℝ

/-- Theorem stating that given the conditions, the track length is 360 meters -/
theorem track_length_is_360 (track : RunningTrack) 
  (h1 : track.sally_first_meeting = 90)
  (h2 : track.john_second_meeting = 200)
  (h3 : track.sally_first_meeting > 0)
  (h4 : track.john_second_meeting > 0)
  (h5 : track.length > 0) :
  track.length = 360 := by
  sorry

#check track_length_is_360

end NUMINAMATH_CALUDE_track_length_is_360_l2274_227489


namespace NUMINAMATH_CALUDE_hammond_discarded_marble_l2274_227426

/-- The weight of discarded marble after carving statues -/
def discarded_marble (initial_block : ℕ) (statue1 statue2 statue3 statue4 : ℕ) : ℕ :=
  initial_block - (statue1 + statue2 + statue3 + statue4)

/-- Theorem stating the amount of discarded marble for Hammond's statues -/
theorem hammond_discarded_marble :
  discarded_marble 80 10 18 15 15 = 22 := by
  sorry

end NUMINAMATH_CALUDE_hammond_discarded_marble_l2274_227426


namespace NUMINAMATH_CALUDE_cos_150_degrees_l2274_227443

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l2274_227443


namespace NUMINAMATH_CALUDE_sum_base3_equals_100212_l2274_227413

/-- Converts a base-3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 3 * acc + d) 0

/-- Converts a decimal number to its base-3 representation as a list of digits -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
    aux n []

theorem sum_base3_equals_100212 :
  let a := base3ToDecimal [1]
  let b := base3ToDecimal [1, 0, 2]
  let c := base3ToDecimal [2, 0, 2, 1]
  let d := base3ToDecimal [1, 1, 0, 1, 2]
  let e := base3ToDecimal [2, 2, 1, 1, 1]
  decimalToBase3 (a + b + c + d + e) = [1, 0, 0, 2, 1, 2] := by
  sorry

end NUMINAMATH_CALUDE_sum_base3_equals_100212_l2274_227413


namespace NUMINAMATH_CALUDE_equation_solution_l2274_227470

theorem equation_solution : ∃ x : ℝ, (24 - 5 = 3 + x) ∧ (x = 16) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2274_227470


namespace NUMINAMATH_CALUDE_prism_volume_is_six_times_pyramid_volume_l2274_227420

/-- A regular quadrilateral prism with an inscribed pyramid -/
structure PrismWithPyramid where
  /-- Side length of the prism's base -/
  a : ℝ
  /-- Height of the prism -/
  h : ℝ
  /-- Volume of the inscribed pyramid -/
  V : ℝ
  /-- The inscribed pyramid has vertices at the center of the upper base
      and the midpoints of the sides of the lower base -/
  pyramid_vertices : Unit

/-- The volume of the prism is 6 times the volume of the inscribed pyramid -/
theorem prism_volume_is_six_times_pyramid_volume (p : PrismWithPyramid) :
  p.a^2 * p.h = 6 * p.V := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_is_six_times_pyramid_volume_l2274_227420


namespace NUMINAMATH_CALUDE_line_inclination_l2274_227446

/-- Given a line with equation y = √3x + 2, its angle of inclination is π/3 -/
theorem line_inclination (x y : ℝ) :
  y = Real.sqrt 3 * x + 2 → 
  ∃ θ : ℝ, θ ∈ Set.Icc 0 π ∧ θ = π / 3 ∧ Real.tan θ = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_line_inclination_l2274_227446


namespace NUMINAMATH_CALUDE_wage_decrease_percentage_l2274_227404

theorem wage_decrease_percentage (wages_last_week : ℝ) (x : ℝ) : 
  wages_last_week > 0 →
  (0.2 * wages_last_week * (1 - x / 100) = 0.6999999999999999 * (0.2 * wages_last_week)) →
  x = 30 := by
  sorry

end NUMINAMATH_CALUDE_wage_decrease_percentage_l2274_227404


namespace NUMINAMATH_CALUDE_opposite_numbers_solution_same_type_radicals_solution_l2274_227483

-- Part 1
theorem opposite_numbers_solution (x : ℝ) :
  2 * x^2 + 3 * x - 5 = -(-2 * x + 2) →
  x = -3/2 ∨ x = 1 :=
by sorry

-- Part 2
theorem same_type_radicals_solution (m : ℝ) :
  m^2 - 6 = 6 * m + 1 →
  m^2 - 6 ≥ 0 →
  m = 7 :=
by sorry

end NUMINAMATH_CALUDE_opposite_numbers_solution_same_type_radicals_solution_l2274_227483


namespace NUMINAMATH_CALUDE_calculate_salary_e_l2274_227458

/-- Calculates the salary of person E given the salaries of A, B, C, D, and the average salary of all five people. -/
theorem calculate_salary_e (salary_a salary_b salary_c salary_d avg_salary : ℕ) :
  salary_a = 8000 →
  salary_b = 5000 →
  salary_c = 11000 →
  salary_d = 7000 →
  avg_salary = 8000 →
  (salary_a + salary_b + salary_c + salary_d + (avg_salary * 5 - (salary_a + salary_b + salary_c + salary_d))) / 5 = avg_salary →
  avg_salary * 5 - (salary_a + salary_b + salary_c + salary_d) = 9000 := by
sorry

end NUMINAMATH_CALUDE_calculate_salary_e_l2274_227458


namespace NUMINAMATH_CALUDE_division_remainder_l2274_227474

theorem division_remainder : ∃ q : ℕ, 1234567 = 256 * q + 229 ∧ 229 < 256 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l2274_227474


namespace NUMINAMATH_CALUDE_prime_square_plus_two_prime_l2274_227454

theorem prime_square_plus_two_prime (P : ℕ) (h1 : Nat.Prime P) (h2 : Nat.Prime (P^2 + 2)) :
  P^4 + 1921 = 2002 := by
sorry

end NUMINAMATH_CALUDE_prime_square_plus_two_prime_l2274_227454


namespace NUMINAMATH_CALUDE_intersection_line_canonical_form_l2274_227449

/-- Given two planes in 3D space, prove that their intersection forms a line with specific canonical equations. -/
theorem intersection_line_canonical_form (x y z : ℝ) :
  (2 * x - 3 * y - 2 * z + 6 = 0) →
  (x - 3 * y + z + 3 = 0) →
  ∃ (t : ℝ), x = 9 * t - 3 ∧ y = 4 * t ∧ z = 3 * t :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_canonical_form_l2274_227449


namespace NUMINAMATH_CALUDE_library_shelves_l2274_227447

theorem library_shelves (books_per_shelf : ℕ) (total_books : ℕ) (h1 : books_per_shelf = 8) (h2 : total_books = 113920) :
  total_books / books_per_shelf = 14240 := by
  sorry

end NUMINAMATH_CALUDE_library_shelves_l2274_227447


namespace NUMINAMATH_CALUDE_rational_equation_zero_solution_l2274_227492

theorem rational_equation_zero_solution (x y z : ℚ) :
  x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_zero_solution_l2274_227492


namespace NUMINAMATH_CALUDE_james_total_vegetables_l2274_227437

/-- The total number of vegetables James ate -/
def total_vegetables (before_carrot before_cucumber after_carrot after_cucumber after_celery : ℕ) : ℕ :=
  before_carrot + before_cucumber + after_carrot + after_cucumber + after_celery

/-- Theorem stating that James ate 77 vegetables in total -/
theorem james_total_vegetables :
  total_vegetables 22 18 15 10 12 = 77 := by
  sorry

end NUMINAMATH_CALUDE_james_total_vegetables_l2274_227437


namespace NUMINAMATH_CALUDE_shuttle_speed_kph_l2274_227406

/-- Conversion factor from seconds to hours -/
def seconds_per_hour : ℕ := 3600

/-- Speed of the space shuttle in kilometers per second -/
def shuttle_speed_kps : ℝ := 6

/-- Theorem stating that the space shuttle's speed in kilometers per hour
    is equal to 21600 -/
theorem shuttle_speed_kph :
  shuttle_speed_kps * seconds_per_hour = 21600 := by
  sorry

end NUMINAMATH_CALUDE_shuttle_speed_kph_l2274_227406


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2274_227431

def A : Set ℤ := {1, 2, 3, 5, 7}
def B : Set ℤ := {x : ℤ | 1 < x ∧ x ≤ 6}
def U : Set ℤ := A ∪ B

theorem intersection_A_complement_B : A ∩ (U \ B) = {1, 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2274_227431


namespace NUMINAMATH_CALUDE_partnership_profit_share_l2274_227411

/-- Given a partnership with three investors A, B, and C, where A invests 3 times as much as B
    and 2/3 of what C invests, prove that C's share of a total profit of 11000 is (9/17) * 11000. -/
theorem partnership_profit_share (a b c : ℝ) (profit : ℝ) : 
  a = 3 * b → 
  a = (2/3) * c → 
  profit = 11000 → 
  c * profit / (a + b + c) = (9/17) * 11000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_share_l2274_227411


namespace NUMINAMATH_CALUDE_k_value_l2274_227444

/-- Two circles centered at the origin with given points and distance --/
structure TwoCircles where
  P : ℝ × ℝ
  S : ℝ × ℝ
  QR : ℝ

/-- The value of k in the point S(0, k) --/
def k (c : TwoCircles) : ℝ := c.S.2

/-- Theorem stating the value of k --/
theorem k_value (c : TwoCircles) 
  (h1 : c.P = (12, 5)) 
  (h2 : c.S.1 = 0) 
  (h3 : c.QR = 4) : 
  k c = 9 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l2274_227444


namespace NUMINAMATH_CALUDE_replaced_man_age_l2274_227455

theorem replaced_man_age (A B C D : ℝ) (new_avg : ℝ) :
  A = 23 →
  (A + B + C + D) / 4 < (52 + C + D) / 4 →
  B < 29 := by
sorry

end NUMINAMATH_CALUDE_replaced_man_age_l2274_227455


namespace NUMINAMATH_CALUDE_distance_traveled_l2274_227498

/-- Given a speed of 40 km/hr and a time of 6 hr, prove that the distance traveled is 240 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 40) (h2 : time = 6) :
  speed * time = 240 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l2274_227498


namespace NUMINAMATH_CALUDE_flour_sugar_difference_l2274_227417

theorem flour_sugar_difference (recipe_sugar : ℕ) (recipe_flour : ℕ) (recipe_salt : ℕ) (flour_added : ℕ) :
  recipe_sugar = 9 →
  recipe_flour = 14 →
  recipe_salt = 40 →
  flour_added = 4 →
  recipe_flour - flour_added - recipe_sugar = 1 := by
sorry

end NUMINAMATH_CALUDE_flour_sugar_difference_l2274_227417


namespace NUMINAMATH_CALUDE_globe_surface_parts_l2274_227414

/-- Represents a globe with a given number of parallels and meridians. -/
structure Globe where
  parallels : ℕ
  meridians : ℕ

/-- Calculates the number of parts the surface of a globe is divided into. -/
def surfaceParts (g : Globe) : ℕ :=
  (g.parallels + 1) * g.meridians

/-- Theorem: A globe with 17 parallels and 24 meridians has its surface divided into 432 parts. -/
theorem globe_surface_parts :
  let g : Globe := { parallels := 17, meridians := 24 }
  surfaceParts g = 432 := by
  sorry

end NUMINAMATH_CALUDE_globe_surface_parts_l2274_227414


namespace NUMINAMATH_CALUDE_probability_opposite_rooms_is_one_fifth_l2274_227493

/-- Represents a hotel with 6 rooms -/
structure Hotel :=
  (rooms : Fin 6 → ℕ)
  (opposite : Fin 3 → Fin 2 → Fin 6)
  (opposite_bijective : ∀ i, Function.Bijective (opposite i))

/-- Represents the random selection of room keys by 6 people -/
def RoomSelection := Fin 6 → Fin 6

/-- The probability of two specific people selecting opposite rooms -/
def probability_opposite_rooms (h : Hotel) : ℚ :=
  1 / 5

/-- Theorem stating that the probability of two specific people
    selecting opposite rooms is 1/5 -/
theorem probability_opposite_rooms_is_one_fifth (h : Hotel) :
  probability_opposite_rooms h = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_opposite_rooms_is_one_fifth_l2274_227493


namespace NUMINAMATH_CALUDE_prob_HHT_fair_coin_l2274_227441

/-- A fair coin has equal probability of heads and tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of a sequence of independent events is the product of their individual probabilities -/
def prob_independent_events (p q r : ℝ) : ℝ := p * q * r

/-- The probability of getting heads on first two flips and tails on third flip for a fair coin -/
theorem prob_HHT_fair_coin (p : ℝ) (h : fair_coin p) : 
  prob_independent_events p p (1 - p) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_prob_HHT_fair_coin_l2274_227441


namespace NUMINAMATH_CALUDE_sum_inequality_l2274_227419

theorem sum_inequality (a b c : ℕ+) (h : (a * b * c : ℚ) = 1) :
  (1 / (b * (a + b)) + 1 / (c * (b + c)) + 1 / (a * (c + a)) : ℚ) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l2274_227419


namespace NUMINAMATH_CALUDE_system_solution_l2274_227401

theorem system_solution (x y k : ℝ) : 
  (x + 2*y = 7 + k) → 
  (5*x - y = k) → 
  (y = -x) → 
  (k = -6) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2274_227401


namespace NUMINAMATH_CALUDE_blueberry_bonnie_ratio_l2274_227466

/-- Represents the number of fruits eaten by each dog -/
structure DogFruits where
  apples : ℕ
  blueberries : ℕ
  bonnies : ℕ

/-- The problem setup -/
def fruitProblem (dogs : Vector DogFruits 3) : Prop :=
  let d1 := dogs.get 0
  let d2 := dogs.get 1
  let d3 := dogs.get 2
  d1.apples = 3 * d2.blueberries ∧
  d3.bonnies = 60 ∧
  d1.apples + d2.blueberries + d3.bonnies = 240

/-- The theorem to prove -/
theorem blueberry_bonnie_ratio (dogs : Vector DogFruits 3) 
  (h : fruitProblem dogs) : 
  (dogs.get 1).blueberries * 4 = (dogs.get 2).bonnies * 3 := by
  sorry


end NUMINAMATH_CALUDE_blueberry_bonnie_ratio_l2274_227466


namespace NUMINAMATH_CALUDE_value_of_n_l2274_227442

theorem value_of_n (x : ℝ) (n : ℝ) 
  (h1 : Real.log (Real.sin x) + Real.log (Real.cos x) = -1)
  (h2 : Real.tan x = Real.sqrt 3)
  (h3 : Real.log (Real.sin x + Real.cos x) = (1/3) * (Real.log n - 1)) :
  n = Real.exp (3 * (-1/2 + Real.log (1 + 1 / Real.sqrt (Real.sqrt 3))) + 1) := by
sorry

end NUMINAMATH_CALUDE_value_of_n_l2274_227442


namespace NUMINAMATH_CALUDE_minute_hand_rotation_1_to_3_20_l2274_227433

/-- The number of radians a clock's minute hand turns through in a given time interval -/
def minute_hand_rotation (start_hour start_minute end_hour end_minute : ℕ) : ℝ :=
  sorry

/-- The number of radians a clock's minute hand turns through from 1:00 to 3:20 -/
theorem minute_hand_rotation_1_to_3_20 :
  minute_hand_rotation 1 0 3 20 = -14/3 * π :=
sorry

end NUMINAMATH_CALUDE_minute_hand_rotation_1_to_3_20_l2274_227433


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l2274_227407

theorem smallest_number_satisfying_conditions : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 3 = 1 ∧ 
  n % 5 = 3 ∧ 
  n % 8 = 4 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 3 = 1 ∧ m % 5 = 3 ∧ m % 8 = 4 → m ≥ n) ∧
  n = 28 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l2274_227407


namespace NUMINAMATH_CALUDE_sin_continuous_l2274_227481

theorem sin_continuous : ContinuousOn Real.sin Set.univ := by
  sorry

end NUMINAMATH_CALUDE_sin_continuous_l2274_227481


namespace NUMINAMATH_CALUDE_nine_steps_climb_l2274_227462

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

def ways_to_climb (n : ℕ) : ℕ := fibonacci (n + 1)

theorem nine_steps_climb : ways_to_climb 9 = 55 := by
  sorry

end NUMINAMATH_CALUDE_nine_steps_climb_l2274_227462


namespace NUMINAMATH_CALUDE_speed_of_A_is_correct_l2274_227410

/-- Represents the speed of boy A in mph -/
def speed_A : ℝ := 7.5

/-- Represents the speed of boy B in mph -/
def speed_B : ℝ := speed_A + 5

/-- Represents the speed of boy C in mph -/
def speed_C : ℝ := speed_A + 3

/-- Represents the total distance between Port Jervis and Poughkeepsie in miles -/
def total_distance : ℝ := 80

/-- Represents the distance from Poughkeepsie where A and B meet in miles -/
def meeting_distance : ℝ := 20

theorem speed_of_A_is_correct :
  speed_A * (total_distance / speed_B + meeting_distance / speed_B) =
  total_distance - meeting_distance := by sorry

end NUMINAMATH_CALUDE_speed_of_A_is_correct_l2274_227410


namespace NUMINAMATH_CALUDE_yasmin_children_count_l2274_227422

def john_children (yasmin_children : ℕ) : ℕ := 2 * yasmin_children

theorem yasmin_children_count :
  ∃ (yasmin_children : ℕ),
    yasmin_children = 2 ∧
    john_children yasmin_children + yasmin_children = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_yasmin_children_count_l2274_227422


namespace NUMINAMATH_CALUDE_power_mod_1111_l2274_227402

theorem power_mod_1111 : ∃ n : ℕ, 0 ≤ n ∧ n < 1111 ∧ 2^1110 ≡ n [ZMOD 1111] := by
  use 1024
  sorry

end NUMINAMATH_CALUDE_power_mod_1111_l2274_227402


namespace NUMINAMATH_CALUDE_find_a_l2274_227477

theorem find_a : ∃ a : ℚ, 3 * a - 2 = 2 / 2 + 3 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l2274_227477


namespace NUMINAMATH_CALUDE_exponential_growth_dominates_power_growth_l2274_227405

theorem exponential_growth_dominates_power_growth 
  (a : ℝ) (α : ℝ) (ha : a > 1) (hα : α > 0) :
  ∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > x₀, 
    (deriv (fun x => a^x) x) / a^x > (deriv (fun x => x^α) x) / x^α :=
sorry

end NUMINAMATH_CALUDE_exponential_growth_dominates_power_growth_l2274_227405


namespace NUMINAMATH_CALUDE_stream_speed_l2274_227485

/-- The speed of a stream given downstream and upstream speeds -/
theorem stream_speed (downstream upstream : ℝ) (h1 : downstream = 13) (h2 : upstream = 8) :
  (downstream - upstream) / 2 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2274_227485


namespace NUMINAMATH_CALUDE_exponent_equation_solution_l2274_227475

theorem exponent_equation_solution : ∃ n : ℤ, 5^3 - 7 = 6^2 + n ∧ n = 82 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_solution_l2274_227475


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l2274_227453

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 20 → p.Prime → ¬(p ∣ n)

theorem smallest_composite_no_small_factors : 
  (is_composite 667 ∧ has_no_small_prime_factors 667) ∧ 
  (∀ m : ℕ, m < 667 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l2274_227453


namespace NUMINAMATH_CALUDE_unit_circle_point_x_coordinate_l2274_227445

theorem unit_circle_point_x_coordinate 
  (P : ℝ × ℝ) 
  (α : ℝ) 
  (h1 : P.1 ≥ 0 ∧ P.2 ≥ 0) -- P is in the first quadrant
  (h2 : P.1^2 + P.2^2 = 1) -- P is on the unit circle
  (h3 : P.1 = Real.cos α ∧ P.2 = Real.sin α) -- Definition of α
  (h4 : Real.cos (α + π/3) = -11/13) -- Given condition
  : P.1 = 1/26 := by
  sorry

end NUMINAMATH_CALUDE_unit_circle_point_x_coordinate_l2274_227445


namespace NUMINAMATH_CALUDE_part_one_part_two_l2274_227432

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Part 1 of the problem -/
theorem part_one (t : Triangle) (h1 : t.A = π / 3) (h2 : t.a = 4 * Real.sqrt 3) (h3 : t.b = 4 * Real.sqrt 2) :
  t.B = π / 4 := by
  sorry

/-- Part 2 of the problem -/
theorem part_two (t : Triangle) (h1 : t.a = 3 * Real.sqrt 3) (h2 : t.c = 2) (h3 : t.B = 5 * π / 6) :
  t.b = 7 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2274_227432


namespace NUMINAMATH_CALUDE_gini_coefficient_change_l2274_227448

/-- Represents a region in the country -/
structure Region where
  population : ℕ
  ppc : ℝ → ℝ
  maxKits : ℝ

/-- Calculates the Gini coefficient given two regions -/
def giniCoefficient (r1 r2 : Region) : ℝ :=
  sorry

/-- Calculates the new Gini coefficient after collaboration -/
def newGiniCoefficient (r1 r2 : Region) (compensation : ℝ) : ℝ :=
  sorry

theorem gini_coefficient_change
  (north : Region)
  (south : Region)
  (h1 : north.population = 24)
  (h2 : south.population = 6)
  (h3 : north.ppc = fun x => 13.5 - 9 * x)
  (h4 : south.ppc = fun x => 24 - 1.5 * x^2)
  (h5 : north.maxKits = 18)
  (h6 : south.maxKits = 12)
  (setPrice : ℝ)
  (h7 : setPrice = 6000)
  (compensation : ℝ)
  (h8 : compensation = 109983) :
  (giniCoefficient north south = 0.2) ∧
  (newGiniCoefficient north south compensation = 0.199) :=
sorry

end NUMINAMATH_CALUDE_gini_coefficient_change_l2274_227448


namespace NUMINAMATH_CALUDE_quadrupled_base_exponent_l2274_227427

theorem quadrupled_base_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (4 * a)^(4 * b) = a^b * x^2 → x = 16^b * a^(3/2 * b) := by
  sorry

end NUMINAMATH_CALUDE_quadrupled_base_exponent_l2274_227427


namespace NUMINAMATH_CALUDE_kite_to_square_area_ratio_l2274_227488

/-- The ratio of the area of a kite formed by the diagonals of four central
    smaller squares to the area of a large square --/
theorem kite_to_square_area_ratio :
  let large_side : ℝ := 60
  let small_side : ℝ := 10
  let large_area : ℝ := large_side ^ 2
  let kite_diagonal1 : ℝ := 2 * small_side
  let kite_diagonal2 : ℝ := 2 * small_side * Real.sqrt 2
  let kite_area : ℝ := (1 / 2) * kite_diagonal1 * kite_diagonal2
  kite_area / large_area = 100 * Real.sqrt 2 / 3600 := by
sorry

end NUMINAMATH_CALUDE_kite_to_square_area_ratio_l2274_227488


namespace NUMINAMATH_CALUDE_smallest_X_l2274_227416

/-- A function that checks if a natural number consists only of 0s and 1s --/
def onlyZerosAndOnes (n : ℕ) : Prop := sorry

/-- The smallest positive integer T consisting of only 0s and 1s that is divisible by 18 --/
def T : ℕ := 111111111000

/-- X is defined as T divided by 18 --/
def X : ℕ := T / 18

/-- Main theorem: X is the smallest positive integer satisfying the given conditions --/
theorem smallest_X : 
  (onlyZerosAndOnes T) ∧ 
  (X * 18 = T) ∧ 
  (∀ Y : ℕ, (∃ S : ℕ, onlyZerosAndOnes S ∧ Y * 18 = S) → X ≤ Y) ∧
  X = 6172839500 := by sorry

end NUMINAMATH_CALUDE_smallest_X_l2274_227416


namespace NUMINAMATH_CALUDE_S_31_primes_less_than_20000_l2274_227425

/-- Sum of digits in base k -/
def S (k : ℕ) (n : ℕ) : ℕ := sorry

/-- The theorem to be proved -/
theorem S_31_primes_less_than_20000 (p : ℕ) (h_prime : Nat.Prime p) (h_bound : p < 20000) :
  S 31 p = 49 ∨ S 31 p = 77 := by sorry

end NUMINAMATH_CALUDE_S_31_primes_less_than_20000_l2274_227425


namespace NUMINAMATH_CALUDE_choose_officers_count_l2274_227439

/-- Represents a club with boys and girls -/
structure Club where
  total_members : ℕ
  boys : ℕ
  girls : ℕ

/-- Calculates the number of ways to choose officers in the club -/
def choose_officers (club : Club) : ℕ :=
  club.boys * club.girls * (club.boys - 1) +
  club.girls * club.boys * (club.girls - 1)

/-- The main theorem stating the number of ways to choose officers -/
theorem choose_officers_count (club : Club)
  (h1 : club.total_members = 25)
  (h2 : club.boys = 12)
  (h3 : club.girls = 13)
  (h4 : club.total_members = club.boys + club.girls) :
  choose_officers club = 3588 := by
  sorry

#eval choose_officers ⟨25, 12, 13⟩

end NUMINAMATH_CALUDE_choose_officers_count_l2274_227439


namespace NUMINAMATH_CALUDE_x_plus_one_equals_four_l2274_227497

theorem x_plus_one_equals_four (x : ℝ) (h : x = 3) : x + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_one_equals_four_l2274_227497


namespace NUMINAMATH_CALUDE_reflection_line_equation_l2274_227496

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Triangle PQR -/
def triangle_PQR : (Point2D × Point2D × Point2D) :=
  (⟨1, 2⟩, ⟨6, 7⟩, ⟨-3, 5⟩)

/-- Reflected triangle P'Q'R' -/
def reflected_triangle : (Point2D × Point2D × Point2D) :=
  (⟨1, -4⟩, ⟨6, -9⟩, ⟨-3, -7⟩)

/-- Line of reflection -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: The line of reflection for the given triangle is y = -1 -/
theorem reflection_line_equation : 
  ∃ (M : Line), M.a = 0 ∧ M.b = 1 ∧ M.c = 1 ∧
  (∀ (P : Point2D) (P' : Point2D), 
    (P = triangle_PQR.1 ∧ P' = reflected_triangle.1) ∨
    (P = triangle_PQR.2.1 ∧ P' = reflected_triangle.2.1) ∨
    (P = triangle_PQR.2.2 ∧ P' = reflected_triangle.2.2) →
    M.a * P.x + M.b * P.y + M.c = M.a * P'.x + M.b * P'.y + M.c) :=
sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l2274_227496


namespace NUMINAMATH_CALUDE_january_rainfall_l2274_227415

theorem january_rainfall (first_week : ℝ) (second_week : ℝ) :
  second_week = 1.5 * first_week →
  second_week = 21 →
  first_week + second_week = 35 := by
sorry

end NUMINAMATH_CALUDE_january_rainfall_l2274_227415


namespace NUMINAMATH_CALUDE_team_probability_l2274_227456

/-- Given 27 players randomly split into 3 teams of 9 each, with Zack, Mihir, and Andrew on different teams,
    the probability that Zack and Andrew are on the same team is 8/17. -/
theorem team_probability (total_players : Nat) (teams : Nat) (players_per_team : Nat)
  (h1 : total_players = 27)
  (h2 : teams = 3)
  (h3 : players_per_team = 9)
  (h4 : total_players = teams * players_per_team)
  (zack mihir andrew : Nat)
  (h5 : zack ≠ mihir)
  (h6 : mihir ≠ andrew)
  (h7 : zack ≠ andrew) :
  (8 : ℚ) / 17 = (players_per_team - 1 : ℚ) / (total_players - 2 * players_per_team) :=
sorry

end NUMINAMATH_CALUDE_team_probability_l2274_227456


namespace NUMINAMATH_CALUDE_number_relation_l2274_227465

theorem number_relation (A B : ℝ) (h : A = B - (4/5) * B) : B = (A + B) / (4/5) := by
  sorry

end NUMINAMATH_CALUDE_number_relation_l2274_227465


namespace NUMINAMATH_CALUDE_direct_proportion_points_l2274_227487

/-- A direct proportion function passing through (-1, 2) also passes through (1, -2) -/
theorem direct_proportion_points : 
  ∀ (f : ℝ → ℝ), 
  (∃ k : ℝ, ∀ x, f x = k * x) → -- f is a direct proportion function
  f (-1) = 2 →                  -- f passes through (-1, 2)
  f 1 = -2                      -- f passes through (1, -2)
:= by sorry

end NUMINAMATH_CALUDE_direct_proportion_points_l2274_227487


namespace NUMINAMATH_CALUDE_a_percentage_less_than_b_l2274_227494

def full_marks : ℕ := 500
def d_marks : ℕ := (80 * full_marks) / 100
def c_marks : ℕ := (80 * d_marks) / 100
def b_marks : ℕ := (125 * c_marks) / 100
def a_marks : ℕ := 360

theorem a_percentage_less_than_b :
  (b_marks - a_marks) * 100 / b_marks = 10 := by sorry

end NUMINAMATH_CALUDE_a_percentage_less_than_b_l2274_227494


namespace NUMINAMATH_CALUDE_max_m_value_l2274_227460

def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ m + 1}

theorem max_m_value (m : ℝ) (h : B m ⊆ A) : m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_m_value_l2274_227460


namespace NUMINAMATH_CALUDE_wednesday_speed_l2274_227400

/-- Jonathan's exercise routine -/
structure ExerciseRoutine where
  monday_speed : ℝ
  wednesday_speed : ℝ
  friday_speed : ℝ
  daily_distance : ℝ
  weekly_time : ℝ

/-- Theorem: Jonathan's walking speed on Wednesdays is 3 miles per hour -/
theorem wednesday_speed (routine : ExerciseRoutine)
  (h1 : routine.monday_speed = 2)
  (h2 : routine.friday_speed = 6)
  (h3 : routine.daily_distance = 6)
  (h4 : routine.weekly_time = 6) :
  routine.wednesday_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_speed_l2274_227400


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2274_227403

theorem quadratic_equation_solution (p q : ℤ) (h1 : p + q = 2010) 
  (h2 : ∃ x1 x2 : ℤ, x1 > 0 ∧ x2 > 0 ∧ 67 * x1^2 + p * x1 + q = 0 ∧ 67 * x2^2 + p * x2 + q = 0) :
  p = -2278 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2274_227403


namespace NUMINAMATH_CALUDE_bus_journey_distance_l2274_227495

/-- Given a bus journey with the following parameters:
  * total_distance: The total distance covered by the bus
  * speed1: The first speed at which the bus travels for part of the journey
  * speed2: The second speed at which the bus travels for the remaining part of the journey
  * total_time: The total time taken for the entire journey

  This theorem proves that the distance covered at the first speed (speed1) is equal to
  the calculated value when the given conditions are met.
-/
theorem bus_journey_distance (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ)
  (h1 : total_distance = 250)
  (h2 : speed1 = 40)
  (h3 : speed2 = 60)
  (h4 : total_time = 5)
  (h5 : speed1 > 0)
  (h6 : speed2 > 0) :
  ∃ (distance1 : ℝ),
    distance1 / speed1 + (total_distance - distance1) / speed2 = total_time ∧
    distance1 = 100 := by
  sorry


end NUMINAMATH_CALUDE_bus_journey_distance_l2274_227495


namespace NUMINAMATH_CALUDE_fixed_point_sum_l2274_227434

/-- The function f(x) = a^(x-2) + 2 with a > 0 and a ≠ 1 has a fixed point (m, n) such that m + n = 5 -/
theorem fixed_point_sum (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  ∃ (m n : ℝ), (∀ x : ℝ, a^(x - 2) + 2 = a^(m - 2) + 2) ∧ m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_sum_l2274_227434


namespace NUMINAMATH_CALUDE_parakeet_cost_graph_is_finite_distinct_points_l2274_227412

def parakeet_cost (n : ℕ) : ℚ :=
  if n ≤ 10 then 20 * n
  else if n ≤ 20 then 18 * n
  else if n ≤ 25 then 18 * n
  else 0

def cost_graph : Set (ℕ × ℚ) :=
  {p | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 25 ∧ p = (n, parakeet_cost n)}

theorem parakeet_cost_graph_is_finite_distinct_points :
  Finite cost_graph ∧ ∀ p q : ℕ × ℚ, p ∈ cost_graph → q ∈ cost_graph → p ≠ q → p.1 ≠ q.1 :=
sorry

end NUMINAMATH_CALUDE_parakeet_cost_graph_is_finite_distinct_points_l2274_227412


namespace NUMINAMATH_CALUDE_division_result_l2274_227482

theorem division_result : (712.5 : ℝ) / 12.5 = 57 := by sorry

end NUMINAMATH_CALUDE_division_result_l2274_227482


namespace NUMINAMATH_CALUDE_smallest_three_digit_sum_product_l2274_227484

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digits_sum_product (a b c : ℕ) : ℕ := a + b + c + a*b + b*c + a*c + a*b*c

theorem smallest_three_digit_sum_product :
  ∃ (n : ℕ) (a b c : ℕ),
    is_three_digit n ∧
    n = 100*a + 10*b + c ∧
    n = digits_sum_product a b c ∧
    (∀ (m : ℕ) (x y z : ℕ),
      is_three_digit m ∧
      m = 100*x + 10*y + z ∧
      m = digits_sum_product x y z →
      n ≤ m) ∧
    n = 199 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_sum_product_l2274_227484


namespace NUMINAMATH_CALUDE_marta_number_proof_l2274_227491

/-- A function that checks if a number has three different non-zero digits -/
def has_three_different_nonzero_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100) ≠ ((n / 10) % 10) ∧
  (n / 100) ≠ (n % 10) ∧
  ((n / 10) % 10) ≠ (n % 10) ∧
  (n / 100) ≠ 0 ∧ ((n / 10) % 10) ≠ 0 ∧ (n % 10) ≠ 0

/-- A function that checks if a number has three identical digits -/
def has_three_identical_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100) = ((n / 10) % 10) ∧
  (n / 100) = (n % 10)

theorem marta_number_proof :
  ∀ n : ℕ,
  has_three_different_nonzero_digits n →
  has_three_identical_digits (3 * n) →
  ((n / 10) % 10) = (3 * n / 100) →
  n = 148 :=
by
  sorry

#check marta_number_proof

end NUMINAMATH_CALUDE_marta_number_proof_l2274_227491


namespace NUMINAMATH_CALUDE_inequality_solution_l2274_227423

theorem inequality_solution (x : ℝ) :
  (2 / (x^2 + 1) > 4 / x + 5 / 2) ↔ -2 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2274_227423


namespace NUMINAMATH_CALUDE_birds_joining_fence_l2274_227428

theorem birds_joining_fence (initial_birds : ℕ) (total_birds : ℕ) (joined_birds : ℕ) : 
  initial_birds = 1 → total_birds = 5 → joined_birds = total_birds - initial_birds → joined_birds = 4 := by
  sorry

end NUMINAMATH_CALUDE_birds_joining_fence_l2274_227428


namespace NUMINAMATH_CALUDE_graham_younger_than_mark_l2274_227469

/-- Represents a person with a birth year and month -/
structure Person where
  birthYear : ℕ
  birthMonth : ℕ
  deriving Repr

def currentYear : ℕ := 2021
def currentMonth : ℕ := 2

def Mark : Person := { birthYear := 1976, birthMonth := 1 }

def JaniceAge : ℕ := 21

/-- Calculates the age of a person in years -/
def age (p : Person) : ℕ :=
  if currentMonth >= p.birthMonth then
    currentYear - p.birthYear
  else
    currentYear - p.birthYear - 1

/-- Calculates Graham's age based on Janice's age -/
def GrahamAge : ℕ := 2 * JaniceAge

theorem graham_younger_than_mark :
  age Mark - GrahamAge = 3 := by
  sorry

end NUMINAMATH_CALUDE_graham_younger_than_mark_l2274_227469


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2274_227451

theorem gcd_of_three_numbers : Nat.gcd 12903 (Nat.gcd 18239 37422) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2274_227451


namespace NUMINAMATH_CALUDE_solution_correctness_l2274_227452

/-- The set of solutions for x^2 - y^2 = 105 where x and y are natural numbers -/
def SolutionsA : Set (ℕ × ℕ) :=
  {(53, 52), (19, 16), (13, 8), (11, 4)}

/-- The set of solutions for 2x^2 + 5xy - 12y^2 = 28 where x and y are natural numbers -/
def SolutionsB : Set (ℕ × ℕ) :=
  {(8, 5)}

theorem solution_correctness :
  (∀ (x y : ℕ), x^2 - y^2 = 105 ↔ (x, y) ∈ SolutionsA) ∧
  (∀ (x y : ℕ), 2*x^2 + 5*x*y - 12*y^2 = 28 ↔ (x, y) ∈ SolutionsB) := by
  sorry

end NUMINAMATH_CALUDE_solution_correctness_l2274_227452
