import Mathlib

namespace NUMINAMATH_CALUDE_pentagon_area_l3603_360354

/-- A point on a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- A pentagon on a grid -/
structure GridPentagon where
  v1 : GridPoint
  v2 : GridPoint
  v3 : GridPoint
  v4 : GridPoint
  v5 : GridPoint

/-- Count the number of integer points strictly inside a polygon -/
def countInteriorPoints (p : GridPentagon) : Int :=
  sorry

/-- Count the number of integer points on the boundary of a polygon -/
def countBoundaryPoints (p : GridPentagon) : Int :=
  sorry

/-- Calculate the area of a polygon using Pick's theorem -/
def polygonArea (p : GridPentagon) : Int :=
  countInteriorPoints p + (countBoundaryPoints p / 2) - 1

theorem pentagon_area :
  let p : GridPentagon := {
    v1 := { x := 0, y := 1 },
    v2 := { x := 2, y := 5 },
    v3 := { x := 6, y := 3 },
    v4 := { x := 5, y := 0 },
    v5 := { x := 1, y := 0 }
  }
  polygonArea p = 17 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_l3603_360354


namespace NUMINAMATH_CALUDE_exactly_two_successes_probability_l3603_360310

def probability_of_success : ℝ := 0.6

def number_of_trials : ℕ := 4

def number_of_successes : ℕ := 2

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exactly_two_successes_probability :
  binomial_probability number_of_trials number_of_successes probability_of_success = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_successes_probability_l3603_360310


namespace NUMINAMATH_CALUDE_selection_probabilities_l3603_360368

/-- Represents the probabilities of passing selections for a student -/
structure StudentProb where
  first : ℝ  -- Probability of passing first selection
  second : ℝ  -- Probability of passing second selection

/-- Given probabilities for students A, B, and C, prove the required probabilities -/
theorem selection_probabilities (a b c : StudentProb)
  (ha_first : a.first = 0.5) (ha_second : a.second = 0.6)
  (hb_first : b.first = 0.6) (hb_second : b.second = 0.5)
  (hc_first : c.first = 0.4) (hc_second : c.second = 0.5) :
  (a.first * (1 - b.first) = 0.2) ∧
  (a.first * a.second * (1 - b.first * b.second) * (1 - c.first * c.second) +
   (1 - a.first * a.second) * b.first * b.second * (1 - c.first * c.second) +
   (1 - a.first * a.second) * (1 - b.first * b.second) * c.first * c.second = 217 / 500) :=
by sorry


end NUMINAMATH_CALUDE_selection_probabilities_l3603_360368


namespace NUMINAMATH_CALUDE_sum_of_squared_differences_equals_three_l3603_360371

theorem sum_of_squared_differences_equals_three (a b c : ℝ) 
  (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) : 
  (b - c)^2 / ((a - b) * (a - c)) + 
  (c - a)^2 / ((b - c) * (b - a)) + 
  (a - b)^2 / ((c - a) * (c - b)) = 3 := by
  sorry

#check sum_of_squared_differences_equals_three

end NUMINAMATH_CALUDE_sum_of_squared_differences_equals_three_l3603_360371


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l3603_360356

/-- Given a geometric sequence {a_n} where a_5 = 7 and a_8 = 56, 
    prove that the general formula is a_n = (7/32) * 2^n -/
theorem geometric_sequence_formula (a : ℕ → ℝ) 
  (h1 : a 5 = 7) 
  (h2 : a 8 = 56) 
  (h_geom : ∀ n m : ℕ, a (n + m) = a n * (a (n + 1) / a n) ^ m) :
  ∃ q : ℝ, ∀ n : ℕ, a n = (7 / 32) * 2^n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l3603_360356


namespace NUMINAMATH_CALUDE_existence_of_non_square_product_l3603_360389

theorem existence_of_non_square_product (d : ℕ) 
  (h_d_pos : d > 0) 
  (h_d_neq_2 : d ≠ 2) 
  (h_d_neq_5 : d ≠ 5) 
  (h_d_neq_13 : d ≠ 13) : 
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ 
               b ∈ ({2, 5, 13, d} : Set ℕ) ∧ 
               a ≠ b ∧ 
               ¬∃ (k : ℕ), a * b - 1 = k * k :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_non_square_product_l3603_360389


namespace NUMINAMATH_CALUDE_diamond_calculation_l3603_360386

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement
theorem diamond_calculation :
  (diamond (diamond 3 4) 2) - (diamond 3 (diamond 4 2)) = -13/28 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l3603_360386


namespace NUMINAMATH_CALUDE_complex_subtraction_l3603_360314

theorem complex_subtraction (a b : ℂ) (h1 : a = 5 - 3*I) (h2 : b = 4 + 3*I) :
  a - 3*b = -7 - 12*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l3603_360314


namespace NUMINAMATH_CALUDE_wage_increase_with_productivity_l3603_360328

/-- Represents the linear regression equation for workers' wages as a function of labor productivity -/
def wage_equation (x : ℝ) : ℝ := 50 + 80 * x

/-- Theorem stating that an increase of 1 in labor productivity leads to an increase of 80 in wages -/
theorem wage_increase_with_productivity (x : ℝ) :
  wage_equation (x + 1) - wage_equation x = 80 := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_with_productivity_l3603_360328


namespace NUMINAMATH_CALUDE_michael_truck_rental_cost_l3603_360302

/-- Calculates the total cost of renting a truck given the rental fee, charge per mile, and miles driven. -/
def truckRentalCost (rentalFee : ℚ) (chargePerMile : ℚ) (milesDriven : ℕ) : ℚ :=
  rentalFee + chargePerMile * milesDriven

/-- Proves that the total cost for Michael's truck rental is $95.74 -/
theorem michael_truck_rental_cost :
  truckRentalCost 20.99 0.25 299 = 95.74 := by
  sorry

end NUMINAMATH_CALUDE_michael_truck_rental_cost_l3603_360302


namespace NUMINAMATH_CALUDE_evaluate_expression_l3603_360365

theorem evaluate_expression : (3200 - 3131)^2 / 121 = 36 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3603_360365


namespace NUMINAMATH_CALUDE_job_completion_time_l3603_360350

/-- The number of days it takes for B to do the job alone -/
def B_days : ℕ := 30

/-- The number of days it takes for A and B to do 4 times the job together -/
def AB_days : ℕ := 72

/-- The number of days it takes for A to do the job alone -/
def A_days : ℕ := 45

theorem job_completion_time :
  (1 : ℚ) / A_days + (1 : ℚ) / B_days = 4 / AB_days :=
sorry

end NUMINAMATH_CALUDE_job_completion_time_l3603_360350


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l3603_360391

/-- Given an inverse proportion function f(x) = k/x where k ≠ 0 and 1 ≤ x ≤ 3,
    if the difference between the maximum and minimum values of f(x) is 4,
    then k = ±6 -/
theorem inverse_proportion_k_value (k : ℝ) (h1 : k ≠ 0) :
  let f : ℝ → ℝ := fun x ↦ k / x
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≤ f 1 ∧ f x ≥ f 3) →
  f 1 - f 3 = 4 →
  k = 6 ∨ k = -6 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l3603_360391


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l3603_360338

-- Define the ⋈ operation
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ y : ℝ, bowtie 7 y = 14 ∧ y = 42 :=
by sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l3603_360338


namespace NUMINAMATH_CALUDE_geometric_sequence_m_range_l3603_360320

theorem geometric_sequence_m_range 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (m : ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = q * a n)
  (h_q_range : q > Real.rpow 5 (1/3) ∧ q < 2)
  (h_equation : m * a 6 * a 7 = a 8 ^ 2 - 2 * a 4 * a 9) :
  m > 3 ∧ m < 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_m_range_l3603_360320


namespace NUMINAMATH_CALUDE_distance_to_x_axis_reflection_triangle_DEF_reflection_distance_l3603_360335

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_x_axis_reflection (x y : ℝ) : 
  Real.sqrt ((x - x)^2 + ((-y) - y)^2) = 2 * |y| := by
  sorry

/-- The specific case for the triangle DEF --/
theorem triangle_DEF_reflection_distance : 
  Real.sqrt ((2 - 2)^2 + ((-1) - 1)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_reflection_triangle_DEF_reflection_distance_l3603_360335


namespace NUMINAMATH_CALUDE_min_value_ab_l3603_360318

theorem min_value_ab (a b : ℝ) (ha : a > 1) (hb : b > 1) (heq : a * b + 2 = 2 * (a + b)) :
  ∃ (min : ℝ), min = 6 + 4 * Real.sqrt 2 ∧ a * b ≥ min := by
sorry

end NUMINAMATH_CALUDE_min_value_ab_l3603_360318


namespace NUMINAMATH_CALUDE_money_distribution_exists_l3603_360355

/-- Represents the money distribution problem with Ram, Gopal, Krishan, and Shekhar -/
def MoneyDistribution (x : ℚ) : Prop :=
  let ram_share := 7
  let gopal_share := 17
  let krishan_share := 17
  let shekhar_share := x
  let ram_money := 490
  let unit_value := ram_money / ram_share
  let gopal_shekhar_ratio := 2 / 1
  (gopal_share / shekhar_share = gopal_shekhar_ratio) ∧
  (shekhar_share * unit_value = 595)

/-- Theorem stating that there exists a valid money distribution satisfying all conditions -/
theorem money_distribution_exists : ∃ x, MoneyDistribution x := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_exists_l3603_360355


namespace NUMINAMATH_CALUDE_distance_can_be_four_l3603_360362

/-- A circle with radius 3 -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (h_radius : radius = 3)

/-- A point outside the circle -/
structure OutsidePoint (c : Circle) :=
  (point : ℝ × ℝ)
  (h_outside : dist point c.center > c.radius)

/-- The theorem stating that the distance between the center and the outside point can be 4 -/
theorem distance_can_be_four (c : Circle) (p : OutsidePoint c) : 
  ∃ (q : OutsidePoint c), dist q.point c.center = 4 :=
sorry

end NUMINAMATH_CALUDE_distance_can_be_four_l3603_360362


namespace NUMINAMATH_CALUDE_sum_of_digits_up_to_5000_l3603_360347

def sumOfDigits (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def sumOfDigitsUpTo (n : ℕ) : ℕ :=
  (List.range n).map sumOfDigits |>.sum

theorem sum_of_digits_up_to_5000 : 
  sumOfDigitsUpTo 5000 = 167450 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_up_to_5000_l3603_360347


namespace NUMINAMATH_CALUDE_max_xy_value_l3603_360340

theorem max_xy_value (x y : ℝ) (hx : x < 0) (hy : y < 0) (h_eq : 3 * x + y = -2) :
  ∃ (max_xy : ℝ), max_xy = 1/3 ∧ ∀ z, z = x * y → z ≤ max_xy :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l3603_360340


namespace NUMINAMATH_CALUDE_shoe_size_mode_and_median_l3603_360363

def shoe_sizes : List ℝ := [25, 25, 25.5, 25.5, 25.5, 25.5, 26, 26, 26.5, 27]

def mode (list : List ℝ) : ℝ := sorry

def median (list : List ℝ) : ℝ := sorry

theorem shoe_size_mode_and_median :
  mode shoe_sizes = 25.5 ∧ median shoe_sizes = 25.5 := by sorry

end NUMINAMATH_CALUDE_shoe_size_mode_and_median_l3603_360363


namespace NUMINAMATH_CALUDE_jaeho_received_most_notebooks_l3603_360378

def notebooks_given : ℕ := 30
def jaehyuk_notebooks : ℕ := 12
def kyunghwan_notebooks : ℕ := 3
def jaeho_notebooks : ℕ := 15

theorem jaeho_received_most_notebooks :
  jaeho_notebooks > jaehyuk_notebooks ∧ jaeho_notebooks > kyunghwan_notebooks :=
by sorry

end NUMINAMATH_CALUDE_jaeho_received_most_notebooks_l3603_360378


namespace NUMINAMATH_CALUDE_factorization_proof_l3603_360358

theorem factorization_proof (x y : ℝ) : 
  (x^2 - 9*y^2 = (x+3*y)*(x-3*y)) ∧ 
  (x^2*y - 6*x*y + 9*y = y*(x-3)^2) ∧ 
  (9*(x+2*y)^2 - 4*(x-y)^2 = (5*x+4*y)*(x+8*y)) ∧ 
  ((x-1)*(x-3) + 1 = (x-2)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3603_360358


namespace NUMINAMATH_CALUDE_pyramid_volume_l3603_360309

/-- Given a pyramid with square base ABCD and vertex E, prove its volume is 1280 -/
theorem pyramid_volume (base_area height_ABE height_CDE distance_E_midpoint : ℝ) 
  (h_base : base_area = 256)
  (h_ABE : height_ABE * base_area.sqrt / 2 = 120)
  (h_CDE : height_CDE * base_area.sqrt / 2 = 136)
  (h_distance : distance_E_midpoint = 17)
  (h_height : (height_ABE ^ 2 + (base_area.sqrt / 2) ^ 2).sqrt = distance_E_midpoint) :
  (1 / 3) * base_area * height_ABE = 1280 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3603_360309


namespace NUMINAMATH_CALUDE_quadratic_function_bound_l3603_360396

/-- Quadratic function with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_bound
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b ≠ 0)
  (h0 : |QuadraticFunction a b c 0| ≤ 1)
  (h1 : |QuadraticFunction a b c 1| ≤ 1)
  (h_1 : |QuadraticFunction a b c (-1)| ≤ 1)
  (hba : |b| ≤ a) :
  ∀ x : ℝ, |x| ≤ 1 → |QuadraticFunction a b c x| ≤ 5/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_bound_l3603_360396


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_greatest_integer_with_gcd_six_exists_l3603_360326

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 150 ∧ Nat.gcd n 18 = 6 → n ≤ 132 :=
by sorry

theorem greatest_integer_with_gcd_six_exists : ∃ n : ℕ, n = 132 ∧ n < 150 ∧ Nat.gcd n 18 = 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_greatest_integer_with_gcd_six_exists_l3603_360326


namespace NUMINAMATH_CALUDE_helga_shoe_shopping_l3603_360380

theorem helga_shoe_shopping (first_store : ℕ) (second_store : ℕ) (third_store : ℕ) :
  first_store = 7 →
  second_store = first_store + 2 →
  third_store = 0 →
  let total_first_three := first_store + second_store + third_store
  let fourth_store := 2 * total_first_three
  first_store + second_store + third_store + fourth_store = 48 :=
by sorry

end NUMINAMATH_CALUDE_helga_shoe_shopping_l3603_360380


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3603_360367

theorem inequality_solution_range (m : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + m| > 3) ↔ (m > 2 ∨ m < -4) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3603_360367


namespace NUMINAMATH_CALUDE_jenny_weight_capacity_l3603_360322

/-- Represents the recycling problem Jenny faces --/
structure RecyclingProblem where
  bottle_weight : ℕ
  can_weight : ℕ
  num_cans : ℕ
  bottle_price : ℕ
  can_price : ℕ
  total_earnings : ℕ

/-- Calculates the total weight Jenny can carry --/
def total_weight (p : RecyclingProblem) : ℕ :=
  let num_bottles := (p.total_earnings - p.num_cans * p.can_price) / p.bottle_price
  num_bottles * p.bottle_weight + p.num_cans * p.can_weight

/-- Theorem stating that Jenny can carry 100 ounces --/
theorem jenny_weight_capacity :
  ∃ (p : RecyclingProblem),
    p.bottle_weight = 6 ∧
    p.can_weight = 2 ∧
    p.num_cans = 20 ∧
    p.bottle_price = 10 ∧
    p.can_price = 3 ∧
    p.total_earnings = 160 ∧
    total_weight p = 100 := by
  sorry


end NUMINAMATH_CALUDE_jenny_weight_capacity_l3603_360322


namespace NUMINAMATH_CALUDE_measure_all_masses_l3603_360301

-- Define the set of weights
def weights : List ℕ := [1, 3, 9, 27, 81]

-- Define a function to check if a mass can be measured
def can_measure (mass : ℕ) : Prop :=
  ∃ (a b c d e : ℤ), 
    a * 1 + b * 3 + c * 9 + d * 27 + e * 81 = mass ∧ 
    (a ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (b ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (c ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (d ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (e ∈ ({-1, 0, 1} : Set ℤ))

-- Theorem statement
theorem measure_all_masses : 
  ∀ m : ℕ, 1 ≤ m ∧ m ≤ 121 → can_measure m :=
by sorry

end NUMINAMATH_CALUDE_measure_all_masses_l3603_360301


namespace NUMINAMATH_CALUDE_ellipse_properties_l3603_360388

/-- Properties of an ellipse with equation (x^2 / 25) + (y^2 / 9) = 1 -/
theorem ellipse_properties :
  let a : ℝ := 5
  let b : ℝ := 3
  let c : ℝ := 4
  let ellipse := fun (x y : ℝ) ↦ x^2 / 25 + y^2 / 9 = 1
  -- Eccentricity
  (c / a = 0.8) ∧
  -- Foci
  (ellipse (-c) 0 ∧ ellipse c 0) ∧
  -- Vertices
  (ellipse (-a) 0 ∧ ellipse a 0) := by
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3603_360388


namespace NUMINAMATH_CALUDE_max_garden_area_l3603_360352

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ := d.length * d.width

/-- Calculates the perimeter of a rectangular garden with one side against a house -/
def gardenPerimeter (d : GardenDimensions) : ℝ := d.length + 2 * d.width

/-- The available fencing length -/
def availableFencing : ℝ := 400

theorem max_garden_area :
  ∃ (d : GardenDimensions),
    gardenPerimeter d = availableFencing ∧
    ∀ (d' : GardenDimensions),
      gardenPerimeter d' = availableFencing →
      gardenArea d' ≤ gardenArea d ∧
      gardenArea d = 20000 := by
  sorry

end NUMINAMATH_CALUDE_max_garden_area_l3603_360352


namespace NUMINAMATH_CALUDE_triangle_angle_sine_identity_l3603_360342

theorem triangle_angle_sine_identity 
  (A B C : ℝ) (n : ℤ) 
  (h_triangle : A + B + C = Real.pi) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0) :
  Real.sin (2 * n * A) + Real.sin (2 * n * B) + Real.sin (2 * n * C) = 
  (-1)^(n+1) * 4 * Real.sin (n * A) * Real.sin (n * B) * Real.sin (n * C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sine_identity_l3603_360342


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_constant_l3603_360323

/-- Given a geometric sequence {a_n} where the sum of the first n terms
    is S_n = 3^(n-2) + m, prove that m = -1/9 -/
theorem geometric_sequence_sum_constant (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℝ) :
  (∀ n, S n = 3^(n-2) + m) →
  (∀ n, a (n+1) / a n = a (n+2) / a (n+1)) →
  (a 1 = S 1) →
  (∀ n, a (n+1) = S (n+1) - S n) →
  m = -1/9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_constant_l3603_360323


namespace NUMINAMATH_CALUDE_multiples_of_four_l3603_360307

/-- Given a natural number n, if there are exactly 25 multiples of 4
    between n and 108 (inclusive), then n = 12. -/
theorem multiples_of_four (n : ℕ) : 
  (∃ (l : List ℕ), l.length = 25 ∧ 
    (∀ x ∈ l, x % 4 = 0 ∧ n ≤ x ∧ x ≤ 108) ∧
    (∀ y, n ≤ y ∧ y ≤ 108 ∧ y % 4 = 0 → y ∈ l)) →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_four_l3603_360307


namespace NUMINAMATH_CALUDE_youngsville_population_l3603_360382

theorem youngsville_population (P : ℝ) : 
  (P * 1.25 * 0.6 = 513) → P = 684 := by
  sorry

end NUMINAMATH_CALUDE_youngsville_population_l3603_360382


namespace NUMINAMATH_CALUDE_function_always_negative_l3603_360357

theorem function_always_negative (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ m ∈ Set.Ioc (-4) 0 := by
sorry

end NUMINAMATH_CALUDE_function_always_negative_l3603_360357


namespace NUMINAMATH_CALUDE_min_value_a_l3603_360327

theorem min_value_a (a : ℕ) (h : 17 ∣ (50^2023 + a)) : 
  ∀ b : ℕ, (17 ∣ (50^2023 + b)) → a ≤ b → 18 ≤ a := by
sorry

end NUMINAMATH_CALUDE_min_value_a_l3603_360327


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l3603_360312

theorem least_k_for_inequality (k : ℤ) : 
  (∀ m : ℤ, m < k → (0.0010101 * (10 : ℝ) ^ m ≤ 100)) ∧ 
  (0.0010101 * (10 : ℝ) ^ k > 100) → 
  k = 6 := by sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l3603_360312


namespace NUMINAMATH_CALUDE_lisa_spoons_count_l3603_360359

/-- The number of spoons Lisa has after combining all sets -/
def total_spoons (num_children : ℕ) (baby_spoons_per_child : ℕ) (decorative_spoons : ℕ)
  (large_spoons : ℕ) (dessert_spoons : ℕ) (soup_spoons : ℕ) (teaspoons : ℕ) : ℕ :=
  num_children * baby_spoons_per_child + decorative_spoons +
  large_spoons + dessert_spoons + soup_spoons + teaspoons

/-- Theorem stating that Lisa has 98 spoons in total -/
theorem lisa_spoons_count :
  total_spoons 6 4 4 20 10 15 25 = 98 := by
  sorry

end NUMINAMATH_CALUDE_lisa_spoons_count_l3603_360359


namespace NUMINAMATH_CALUDE_theater_ticket_cost_l3603_360385

/-- Proves that the cost of an orchestra seat is $12 given the conditions of the theater ticket sales --/
theorem theater_ticket_cost (balcony_cost : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (balcony_excess : ℕ) :
  balcony_cost = 8 →
  total_tickets = 350 →
  total_revenue = 3320 →
  balcony_excess = 90 →
  ∃ (orchestra_cost : ℕ), 
    orchestra_cost = 12 ∧
    (total_tickets - balcony_excess) / 2 * orchestra_cost + 
    (total_tickets + balcony_excess) / 2 * balcony_cost = total_revenue :=
by
  sorry

#check theater_ticket_cost

end NUMINAMATH_CALUDE_theater_ticket_cost_l3603_360385


namespace NUMINAMATH_CALUDE_concert_attendance_l3603_360373

theorem concert_attendance (num_buses : ℕ) (students_per_bus : ℕ) (students_in_minivan : ℕ) :
  num_buses = 12 →
  students_per_bus = 38 →
  students_in_minivan = 5 →
  num_buses * students_per_bus + students_in_minivan = 461 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l3603_360373


namespace NUMINAMATH_CALUDE_baking_powder_difference_l3603_360360

def baking_powder_yesterday : ℝ := 0.4
def baking_powder_today : ℝ := 0.3

theorem baking_powder_difference :
  baking_powder_yesterday - baking_powder_today = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_baking_powder_difference_l3603_360360


namespace NUMINAMATH_CALUDE_brick_length_is_correct_l3603_360344

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 25

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 11.25

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 6

/-- The length of the wall in centimeters -/
def wall_length : ℝ := 700

/-- The height of the wall in centimeters -/
def wall_height : ℝ := 600

/-- The width of the wall in centimeters -/
def wall_width : ℝ := 22.5

/-- The number of bricks needed to build the wall -/
def num_bricks : ℕ := 5600

/-- Theorem stating that the brick length is correct given the wall and brick dimensions -/
theorem brick_length_is_correct :
  brick_length * brick_width * brick_height * num_bricks = wall_length * wall_height * wall_width := by
  sorry

end NUMINAMATH_CALUDE_brick_length_is_correct_l3603_360344


namespace NUMINAMATH_CALUDE_sugar_calculation_l3603_360324

/-- The total amount of sugar given the number of packs, weight per pack, and leftover sugar -/
def total_sugar (num_packs : ℕ) (weight_per_pack : ℕ) (leftover : ℕ) : ℕ :=
  num_packs * weight_per_pack + leftover

/-- Theorem: Given 12 packs of sugar weighing 250 grams each and 20 grams of leftover sugar,
    the total amount of sugar is 3020 grams -/
theorem sugar_calculation :
  total_sugar 12 250 20 = 3020 := by
  sorry

end NUMINAMATH_CALUDE_sugar_calculation_l3603_360324


namespace NUMINAMATH_CALUDE_arithmetic_seq_sum_l3603_360370

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)

/-- Theorem stating that for an arithmetic sequence with S_17 = 170, a_7 + a_8 + a_12 = 30 -/
theorem arithmetic_seq_sum (seq : ArithmeticSequence) (h : seq.S 17 = 170) :
  seq.a 7 + seq.a 8 + seq.a 12 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_sum_l3603_360370


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l3603_360303

theorem ceiling_floor_difference : 
  ⌈(20 : ℝ) / 9 * ⌈(-53 : ℝ) / 4⌉⌉ - ⌊(20 : ℝ) / 9 * ⌊(-53 : ℝ) / 4⌋⌋ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l3603_360303


namespace NUMINAMATH_CALUDE_function_proof_l3603_360300

theorem function_proof (f : ℕ → ℕ) 
  (h1 : f 0 = 1)
  (h2 : f 2016 = 2017)
  (h3 : ∀ n, f (f n) + f n = 2 * n + 3) :
  ∀ n, f n = n + 1 := by
sorry

end NUMINAMATH_CALUDE_function_proof_l3603_360300


namespace NUMINAMATH_CALUDE_ratio_of_powers_l3603_360351

theorem ratio_of_powers (p q : ℝ) (n : ℕ) (h1 : n > 1) (h2 : p^n / q^n = 7) :
  (p^n + q^n) / (p^n - q^n) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_powers_l3603_360351


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l3603_360341

-- Define a function to represent the power tower of 2's
def powerTower (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2 ^ (powerTower n)

-- Define the right-hand side of the inequality
def rightHandSide : ℕ := 3^(3^(3^3))

-- Theorem statement
theorem smallest_n_for_inequality :
  (∀ k < 6, powerTower k ≤ rightHandSide) ∧
  (powerTower 6 > rightHandSide) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l3603_360341


namespace NUMINAMATH_CALUDE_trapezoid_area_is_72_l3603_360331

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- One of the base angles in radians -/
  baseAngle : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The trapezoid is circumscribed around a circle -/
  isCircumscribed : True

/-- Calculate the area of the isosceles trapezoid -/
def areaOfTrapezoid (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem trapezoid_area_is_72 (t : IsoscelesTrapezoid) 
  (h1 : t.longerBase = 20)
  (h2 : t.baseAngle = Real.arcsin 0.6) :
  areaOfTrapezoid t = 72 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_72_l3603_360331


namespace NUMINAMATH_CALUDE_existence_of_A_for_any_E_l3603_360308

/-- Property P: A sequence is a permutation of {1, 2, ..., n} -/
def has_property_P (A : List ℕ) : Prop :=
  A.length ≥ 2 ∧ A.Nodup ∧ ∀ i, i ∈ A → i ∈ Finset.range A.length

/-- T(A) sequence definition -/
def T (A : List ℕ) : List ℕ :=
  List.zipWith (fun a b => if a < b then 1 else 0) A A.tail

theorem existence_of_A_for_any_E (n : ℕ) (E : List ℕ) 
    (h_n : n ≥ 2) 
    (h_E_length : E.length = n - 1) 
    (h_E_elements : ∀ e ∈ E, e = 0 ∨ e = 1) :
    ∃ A : List ℕ, has_property_P A ∧ T A = E :=
  sorry

end NUMINAMATH_CALUDE_existence_of_A_for_any_E_l3603_360308


namespace NUMINAMATH_CALUDE_integer_sum_problem_l3603_360377

theorem integer_sum_problem : 
  ∃ (a b : ℕ+), 
    (a.val * b.val + a.val + b.val = 143) ∧ 
    (Nat.gcd a.val b.val = 1) ∧ 
    (a.val < 30 ∧ b.val < 30) ∧ 
    (a.val + b.val = 23 ∨ a.val + b.val = 24 ∨ a.val + b.val = 28) := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l3603_360377


namespace NUMINAMATH_CALUDE_P_in_xoz_plane_l3603_360311

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoz plane in 3D space -/
def xoz_plane : Set Point3D :=
  {p : Point3D | p.y = 0}

/-- Point P with coordinates (-2, 0, 3) -/
def P : Point3D :=
  ⟨-2, 0, 3⟩

theorem P_in_xoz_plane : P ∈ xoz_plane := by
  sorry


end NUMINAMATH_CALUDE_P_in_xoz_plane_l3603_360311


namespace NUMINAMATH_CALUDE_blueberry_zucchini_trade_l3603_360348

/-- The number of bushes needed to obtain a specific number of zucchinis -/
def bushes_needed (total_containers_per_bush : ℕ) (containers_for_jam : ℕ) 
                  (containers_per_trade : ℕ) (zucchinis_per_trade : ℕ) 
                  (target_zucchinis : ℕ) : ℕ :=
  let usable_containers := total_containers_per_bush - containers_for_jam
  let zucchinis_per_container := zucchinis_per_trade / containers_per_trade
  let zucchinis_per_bush := usable_containers * zucchinis_per_container
  target_zucchinis / zucchinis_per_bush

/-- Theorem stating that 18 bushes are needed to obtain 72 zucchinis under given conditions -/
theorem blueberry_zucchini_trade : bushes_needed 10 2 6 3 72 = 18 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_zucchini_trade_l3603_360348


namespace NUMINAMATH_CALUDE_perfect_square_sum_l3603_360364

theorem perfect_square_sum (n : ℕ) : 
  n > 0 ∧ n < 200 ∧ (∃ k : ℕ, n^2 + (n+1)^2 = k^2) ↔ n = 3 ∨ n = 20 ∨ n = 119 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l3603_360364


namespace NUMINAMATH_CALUDE_max_acute_triangles_formula_l3603_360369

/-- Represents a line with marked points -/
structure MarkedLine where
  points : Finset ℝ
  distinct : points.card = 50

/-- The maximum number of acute-angled triangles formed by points on two parallel lines -/
def max_acute_triangles (a b : MarkedLine) : ℕ :=
  (50^3 - 50) / 3

/-- Theorem stating the maximum number of acute-angled triangles -/
theorem max_acute_triangles_formula (a b : MarkedLine) (h : a.points ∩ b.points = ∅) :
  max_acute_triangles a b = 41650 :=
sorry

end NUMINAMATH_CALUDE_max_acute_triangles_formula_l3603_360369


namespace NUMINAMATH_CALUDE_num_ways_to_place_pawns_l3603_360325

/-- Represents a chess board configuration -/
def ChessBoard := Fin 5 → Fin 5

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- Checks if a chess board configuration is valid (no more than one pawn per row and column) -/
def is_valid_configuration (board : ChessBoard) : Prop :=
  (∀ i j : Fin 5, i ≠ j → board i ≠ board j) ∧
  (∀ i : Fin 5, ∃ j : Fin 5, board j = i)

/-- The number of valid chess board configurations -/
def num_valid_configurations : ℕ := factorial 5

/-- The main theorem stating the number of ways to place five distinct pawns -/
theorem num_ways_to_place_pawns :
  (num_valid_configurations * factorial 5 : ℕ) = 14400 :=
sorry

end NUMINAMATH_CALUDE_num_ways_to_place_pawns_l3603_360325


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3603_360329

/-- The eccentricity of a hyperbola with asymptotes tangent to a specific circle -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 ∧
   (b * x + a * y = 0 ∨ b * x - a * y = 0) ∧
   (x - Real.sqrt 2)^2 + y^2 = 1) →
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3603_360329


namespace NUMINAMATH_CALUDE_divisor_and_totient_properties_l3603_360319

/-- Sum of divisors function -/
def τ (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def φ (n : ℕ) : ℕ := sorry

theorem divisor_and_totient_properties (n : ℕ) :
  (n > 1 → φ n * τ n < n^2) ∧
  (φ n * τ n + 1 = n^2 ↔ Nat.Prime n) ∧
  ¬∃ (m : ℕ), φ m * τ m + 2023 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_divisor_and_totient_properties_l3603_360319


namespace NUMINAMATH_CALUDE_max_distance_OM_l3603_360339

/-- The ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The circle O -/
def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- The tangent line l -/
def tangent_line_l (x y m t : ℝ) : Prop :=
  x = m * y + t

theorem max_distance_OM (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : (a^2 - b^2).sqrt / a = Real.sqrt 3 / 2) (h4 : 2 * b = 2)
  (A B M : ℝ × ℝ) (m t : ℝ)
  (hA : ellipse_C A.1 A.2 a b)
  (hB : ellipse_C B.1 B.2 a b)
  (hM : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hl : tangent_line_l A.1 A.2 m t ∧ tangent_line_l B.1 B.2 m t)
  (htangent : t^2 = m^2 + 1) :
  (∀ P : ℝ × ℝ, ellipse_C P.1 P.2 a b → (P.1^2 + P.2^2).sqrt ≤ 5/4) ∧
  (∃ Q : ℝ × ℝ, ellipse_C Q.1 Q.2 a b ∧ (Q.1^2 + Q.2^2).sqrt = 5/4) :=
sorry

end NUMINAMATH_CALUDE_max_distance_OM_l3603_360339


namespace NUMINAMATH_CALUDE_marbles_fraction_l3603_360399

theorem marbles_fraction (initial_marbles : ℕ) (fraction_taken : ℚ) (cleo_final : ℕ) : 
  initial_marbles = 30 →
  fraction_taken = 3/5 →
  cleo_final = 15 →
  (cleo_final - (fraction_taken * initial_marbles / 2)) / (initial_marbles - fraction_taken * initial_marbles) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_marbles_fraction_l3603_360399


namespace NUMINAMATH_CALUDE_purely_imaginary_Z_implies_m_equals_two_l3603_360305

-- Define the complex number Z as a function of m
def Z (m : ℝ) : ℂ := Complex.mk (m^2 - m - 2) (m^2 - 2*m - 3)

-- State the theorem
theorem purely_imaginary_Z_implies_m_equals_two :
  ∀ m : ℝ, (Z m).re = 0 ∧ (Z m).im ≠ 0 → m = 2 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_Z_implies_m_equals_two_l3603_360305


namespace NUMINAMATH_CALUDE_min_value_expression_l3603_360395

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 3) :
  a^2 + 8*a*b + 32*b^2 + 24*b*c + 8*c^2 ≥ 72 ∧
  ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ * b₀ * c₀ = 3 ∧
    a₀^2 + 8*a₀*b₀ + 32*b₀^2 + 24*b₀*c₀ + 8*c₀^2 = 72 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3603_360395


namespace NUMINAMATH_CALUDE_battery_current_at_12_ohms_l3603_360306

/-- Given a battery with voltage 48V and a relationship between current I and resistance R,
    prove that when R = 12Ω, I = 4A. -/
theorem battery_current_at_12_ohms :
  let voltage : ℝ := 48
  let R : ℝ := 12
  let I : ℝ := voltage / R
  I = 4 := by sorry

end NUMINAMATH_CALUDE_battery_current_at_12_ohms_l3603_360306


namespace NUMINAMATH_CALUDE_greening_project_optimization_l3603_360372

/-- The optimization problem for greening project --/
theorem greening_project_optimization (total_area : ℝ) (team_a_rate : ℝ) (team_b_rate : ℝ)
  (team_a_wage : ℝ) (team_b_wage : ℝ) (h1 : total_area = 1200)
  (h2 : team_a_rate = 100) (h3 : team_b_rate = 50) (h4 : team_a_wage = 4000) (h5 : team_b_wage = 3000) :
  ∃ (days_a days_b : ℝ),
    days_a ≥ 3 ∧ days_b ≥ days_a ∧
    team_a_rate * days_a + team_b_rate * days_b = total_area ∧
    ∀ (x y : ℝ),
      x ≥ 3 → y ≥ x →
      team_a_rate * x + team_b_rate * y = total_area →
      team_a_wage * days_a + team_b_wage * days_b ≤ team_a_wage * x + team_b_wage * y ∧
      team_a_wage * days_a + team_b_wage * days_b = 56000 :=
by
  sorry

end NUMINAMATH_CALUDE_greening_project_optimization_l3603_360372


namespace NUMINAMATH_CALUDE_fruit_store_inventory_l3603_360384

/-- Represents the fruit store inventory and gift basket composition. -/
structure FruitStore where
  cantaloupes : ℕ
  dragonFruits : ℕ
  kiwis : ℕ
  basketCantaloupes : ℕ
  basketDragonFruits : ℕ
  basketKiwis : ℕ

/-- Theorem stating the original number of dragon fruits and remaining kiwis. -/
theorem fruit_store_inventory (store : FruitStore)
  (h1 : store.basketCantaloupes = 2)
  (h2 : store.basketDragonFruits = 4)
  (h3 : store.basketKiwis = 10)
  (h4 : store.dragonFruits = 3 * store.cantaloupes + 10)
  (h5 : store.kiwis = 2 * store.dragonFruits)
  (h6 : store.dragonFruits - store.basketDragonFruits * store.cantaloupes = 130) :
  store.dragonFruits = 370 ∧ 
  store.kiwis - store.basketKiwis * store.cantaloupes = 140 := by
  sorry


end NUMINAMATH_CALUDE_fruit_store_inventory_l3603_360384


namespace NUMINAMATH_CALUDE_law_of_sines_symmetry_l3603_360361

/-- The Law of Sines for a triangle ABC with sides a, b, c and angles A, B, C -/
def law_of_sines (a b c : ℝ) (A B C : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

/-- A property representing symmetry in mathematical expressions -/
def has_symmetry (P : Prop) : Prop :=
  -- This is a placeholder definition. In a real implementation, 
  -- this would need to be defined based on specific criteria for symmetry.
  true

/-- Theorem stating that the Law of Sines exhibits mathematical symmetry -/
theorem law_of_sines_symmetry (a b c : ℝ) (A B C : ℝ) :
  has_symmetry (law_of_sines a b c A B C) :=
sorry

end NUMINAMATH_CALUDE_law_of_sines_symmetry_l3603_360361


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l3603_360393

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem lines_perp_to_plane_are_parallel 
  (a b : Line) (α : Plane) 
  (h1 : perp a α) (h2 : perp b α) : 
  parallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l3603_360393


namespace NUMINAMATH_CALUDE_third_place_prize_l3603_360387

theorem third_place_prize (total_prize : ℕ) (num_novels : ℕ) (first_prize : ℕ) (second_prize : ℕ) (other_prize : ℕ) :
  total_prize = 800 →
  num_novels = 18 →
  first_prize = 200 →
  second_prize = 150 →
  other_prize = 22 →
  (num_novels - 3) * other_prize + first_prize + second_prize + 120 = total_prize :=
by sorry

end NUMINAMATH_CALUDE_third_place_prize_l3603_360387


namespace NUMINAMATH_CALUDE_area_between_circles_l3603_360390

theorem area_between_circles (r : ℝ) (R : ℝ) : 
  r = 3 →  -- radius of smaller circle
  R = 3 * r →  -- radius of larger circle is three times the smaller
  π * R^2 - π * r^2 = 72 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_l3603_360390


namespace NUMINAMATH_CALUDE_correct_probability_l3603_360345

/-- The number of options for the first three digits -/
def first_three_options : ℕ := 3

/-- The number of remaining digits to arrange -/
def remaining_digits : ℕ := 5

/-- The probability of correctly guessing the phone number -/
def probability_correct_guess : ℚ := 1 / (first_three_options * remaining_digits.factorial)

theorem correct_probability :
  probability_correct_guess = 1 / 360 := by
  sorry

end NUMINAMATH_CALUDE_correct_probability_l3603_360345


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3603_360343

theorem fraction_subtraction : 
  (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3603_360343


namespace NUMINAMATH_CALUDE_parallel_lines_length_l3603_360381

-- Define the parallel lines and their lengths
def AB : ℝ := 210
def CD : ℝ := 140
def EF : ℝ := 84

-- Define the parallel relation
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry

-- State the theorem
theorem parallel_lines_length :
  ∀ (ab gh cd ef : ℝ → ℝ → Prop),
    parallel ab gh → parallel gh cd → parallel cd ef →
    AB = 210 → CD = 140 →
    EF = 84 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_length_l3603_360381


namespace NUMINAMATH_CALUDE_simplify_expression_l3603_360353

theorem simplify_expression (a b : ℝ) : (22*a + 60*b) + (10*a + 29*b) - (9*a + 50*b) = 23*a + 39*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3603_360353


namespace NUMINAMATH_CALUDE_type_A_nutrition_l3603_360336

/-- Represents the nutritional content of food types A and B -/
structure FoodNutrition where
  protein_A : ℝ  -- Protein content per gram of type A food
  protein_B : ℝ  -- Protein content per gram of type B food
  iron_A : ℝ     -- Iron content per gram of type A food
  iron_B : ℝ     -- Iron content per gram of type B food

/-- Represents the meal composition and nutritional requirements -/
structure MealComposition where
  weight_A : ℝ    -- Weight of type A food in grams
  weight_B : ℝ    -- Weight of type B food in grams
  protein_req : ℝ -- Required protein in units
  iron_req : ℝ    -- Required iron in units

/-- Theorem stating the nutritional content of type A food -/
theorem type_A_nutrition 
  (food : FoodNutrition) 
  (meal : MealComposition) 
  (h1 : food.iron_A = 2 * food.protein_A)
  (h2 : food.iron_B = (4/7) * food.protein_B)
  (h3 : meal.weight_A * food.protein_A + meal.weight_B * food.protein_B = meal.protein_req)
  (h4 : meal.weight_A * food.iron_A + meal.weight_B * food.iron_B = meal.iron_req)
  (h5 : meal.weight_A = 28)
  (h6 : meal.weight_B = 30)
  (h7 : meal.protein_req = 35)
  (h8 : meal.iron_req = 40)
  : food.protein_A = 0.5 ∧ food.iron_A = 1 := by
  sorry

#check type_A_nutrition

end NUMINAMATH_CALUDE_type_A_nutrition_l3603_360336


namespace NUMINAMATH_CALUDE_expression_evaluation_l3603_360316

theorem expression_evaluation (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  (2*x + y)^2 + (x + y)*(x - y) - x^2 = -4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3603_360316


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3603_360376

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 0 → x * Real.exp x > 0)) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ * Real.exp x₀ ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3603_360376


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l3603_360394

theorem arccos_one_over_sqrt_two (π : ℝ) : Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l3603_360394


namespace NUMINAMATH_CALUDE_fraction_problem_l3603_360334

theorem fraction_problem (f : ℚ) : f * (-72 : ℚ) = -60 → f = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3603_360334


namespace NUMINAMATH_CALUDE_m_range_l3603_360321

theorem m_range (x : ℝ) :
  (∀ x, 1/3 < x ∧ x < 1/2 → m - 1 < x ∧ x < m + 1) ∧
  (∃ x, m - 1 < x ∧ x < m + 1 ∧ (x ≤ 1/3 ∨ 1/2 ≤ x)) →
  -1/2 ≤ m ∧ m ≤ 4/3 ∧ m ≠ -1/2 ∧ m ≠ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3603_360321


namespace NUMINAMATH_CALUDE_sphere_in_cube_untouchable_area_l3603_360317

/-- The area of a cube's inner surface that a sphere can't touch -/
def untouchableArea (cubeEdge : ℝ) (sphereRadius : ℝ) : ℝ :=
  12 * cubeEdge * sphereRadius - 24 * sphereRadius^2

theorem sphere_in_cube_untouchable_area :
  untouchableArea 5 1 = 96 := by
  sorry

end NUMINAMATH_CALUDE_sphere_in_cube_untouchable_area_l3603_360317


namespace NUMINAMATH_CALUDE_grandfather_age_proof_l3603_360375

/-- The age of Xiaoming's grandfather -/
def grandfather_age : ℕ := 79

/-- The result after processing the grandfather's age -/
def processed_age (age : ℕ) : ℕ :=
  ((age - 15) / 4 - 6) * 10

theorem grandfather_age_proof :
  processed_age grandfather_age = 100 :=
by sorry

end NUMINAMATH_CALUDE_grandfather_age_proof_l3603_360375


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3603_360392

theorem sqrt_product_equality : 
  2 * Real.sqrt 3 * (1.5 ^ (1/3)) * (12 ^ (1/6)) = 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3603_360392


namespace NUMINAMATH_CALUDE_largest_angle_in_3_4_5_ratio_triangle_l3603_360304

theorem largest_angle_in_3_4_5_ratio_triangle : 
  ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  (a + b + c = 180) →
  (b = (4/3) * a) →
  (c = (5/3) * a) →
  c = 75 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_3_4_5_ratio_triangle_l3603_360304


namespace NUMINAMATH_CALUDE_percentage_of_number_l3603_360379

theorem percentage_of_number (x : ℝ) (h : x = 16) : x * 0.0025 = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_number_l3603_360379


namespace NUMINAMATH_CALUDE_find_P_l3603_360337

theorem find_P : ∃ P : ℝ, (1/3 : ℝ) * (1/6 : ℝ) * P = (1/4 : ℝ) * (1/8 : ℝ) * 64 + (1/5 : ℝ) * (1/10 : ℝ) * 100 → P = 72 := by
  sorry

end NUMINAMATH_CALUDE_find_P_l3603_360337


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3603_360313

theorem complex_modulus_problem (z : ℂ) (h : (1 + 2*I)*z = 1 - I) : Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3603_360313


namespace NUMINAMATH_CALUDE_divisibility_by_20p_l3603_360397

theorem divisibility_by_20p (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ k : ℤ, (⌊(Real.sqrt 5 + 2)^p - 2^(p+1)⌋ : ℤ) = 20 * p * k :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_20p_l3603_360397


namespace NUMINAMATH_CALUDE_original_savings_calculation_l3603_360332

theorem original_savings_calculation (savings : ℝ) : 
  (5/6 : ℝ) * savings + 500 = savings → savings = 3000 := by
  sorry

end NUMINAMATH_CALUDE_original_savings_calculation_l3603_360332


namespace NUMINAMATH_CALUDE_base3_to_base10_equality_l3603_360366

/-- Converts a base 3 number represented as a list of digits to its base 10 equivalent -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number we want to convert -/
def base3Number : List Nat := [2, 1, 2, 0, 1]

/-- Theorem stating that the base 3 number 10212₃ is equal to 104 in base 10 -/
theorem base3_to_base10_equality : base3ToBase10 base3Number = 104 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_equality_l3603_360366


namespace NUMINAMATH_CALUDE_expression_equals_three_l3603_360333

theorem expression_equals_three (x : ℝ) (h : x^2 - 4*x = 5) : 
  ∃ (f : ℝ → ℝ), f x = 3 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_three_l3603_360333


namespace NUMINAMATH_CALUDE_min_value_x_plus_reciprocal_min_value_is_three_l3603_360374

theorem min_value_x_plus_reciprocal (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 := by
  sorry

theorem min_value_is_three : ∃ (x : ℝ), x > 1 ∧ x + 1 / (x - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_reciprocal_min_value_is_three_l3603_360374


namespace NUMINAMATH_CALUDE_junior_percentage_is_22_l3603_360383

/-- Represents the number of students in each grade --/
structure StudentCount where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- The total number of students in the sample --/
def totalStudents : ℕ := 800

/-- The conditions of the problem --/
def sampleConditions (s : StudentCount) : Prop :=
  s.freshmen + s.sophomores + s.juniors + s.seniors = totalStudents ∧
  s.sophomores = totalStudents / 4 ∧
  s.seniors = 160 ∧
  s.freshmen = s.sophomores + 64

/-- The percentage of juniors in the sample --/
def juniorPercentage (s : StudentCount) : ℚ :=
  s.juniors * 100 / totalStudents

/-- Theorem stating that the percentage of juniors is 22% --/
theorem junior_percentage_is_22 (s : StudentCount) 
  (h : sampleConditions s) : juniorPercentage s = 22 := by
  sorry

end NUMINAMATH_CALUDE_junior_percentage_is_22_l3603_360383


namespace NUMINAMATH_CALUDE_min_value_of_function_l3603_360330

open Real

theorem min_value_of_function (θ : ℝ) (h1 : sin θ ≠ 0) (h2 : cos θ ≠ 0) :
  ∃ (min_val : ℝ), min_val = 5 * sqrt 5 ∧ 
  ∀ θ', sin θ' ≠ 0 → cos θ' ≠ 0 → 1 / sin θ' + 8 / cos θ' ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3603_360330


namespace NUMINAMATH_CALUDE_garden_tomato_percentage_l3603_360315

theorem garden_tomato_percentage :
  let total_plants : ℕ := 20 + 15
  let second_garden_tomatoes : ℕ := 15 / 3
  let total_tomatoes : ℕ := (total_plants * 20) / 100
  let first_garden_tomatoes : ℕ := total_tomatoes - second_garden_tomatoes
  (first_garden_tomatoes : ℚ) / 20 * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_garden_tomato_percentage_l3603_360315


namespace NUMINAMATH_CALUDE_trailing_zeros_of_main_expression_l3603_360346

/-- The number of trailing zeros in n -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- Prime factorization of 15 -/
def fifteen : ℕ := 3 * 5

/-- Prime factorization of 28 -/
def twentyEight : ℕ := 2^2 * 7

/-- Prime factorization of 55 -/
def fiftyFive : ℕ := 5 * 11

/-- The main expression -/
def mainExpression : ℕ := fifteen^6 * twentyEight^5 * fiftyFive^7

theorem trailing_zeros_of_main_expression :
  trailingZeros mainExpression = 10 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_main_expression_l3603_360346


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l3603_360398

theorem lcm_gcd_product (a b : ℕ) (ha : a = 36) (hb : b = 48) :
  Nat.lcm a b * Nat.gcd a b = 1728 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l3603_360398


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l3603_360349

/-- Given a cube with face perimeter of 40 cm, its volume is 1000 cubic centimeters. -/
theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (volume : ℝ) :
  face_perimeter = 40 → volume = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l3603_360349
