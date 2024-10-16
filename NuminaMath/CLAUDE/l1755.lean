import Mathlib

namespace NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l1755_175576

theorem gcd_polynomial_and_multiple (a : ℤ) (h : ∃ k : ℤ, a = 532 * k) :
  Int.gcd (5 * a^3 + 2 * a^2 + 6 * a + 76) a = 76 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l1755_175576


namespace NUMINAMATH_CALUDE_coin_array_problem_l1755_175531

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The problem statement -/
theorem coin_array_problem :
  ∃ (N : ℕ), triangular_sum N = 3003 ∧ sum_of_digits N = 14 :=
sorry

end NUMINAMATH_CALUDE_coin_array_problem_l1755_175531


namespace NUMINAMATH_CALUDE_inequality_solution_l1755_175554

/-- Given an inequality ax^2 - 3x + 2 > 0 with solution set {x | x < 1 or x > b},
    where b > 1 and a > 0, prove that a = 1, b = 2, and the solution set for
    x^2 - 3x + 2 > 0 is {x | 1 < x < 2}. -/
theorem inequality_solution (a b : ℝ) 
    (h1 : ∀ x, a * x^2 - 3*x + 2 > 0 ↔ x < 1 ∨ x > b)
    (h2 : b > 1) 
    (h3 : a > 0) : 
    a = 1 ∧ b = 2 ∧ (∀ x, x^2 - 3*x + 2 > 0 ↔ 1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1755_175554


namespace NUMINAMATH_CALUDE_profit_maximization_profit_function_correct_sales_at_price_l1755_175532

/-- Represents the daily profit function for a product -/
def profit_function (x : ℝ) : ℝ := (200 - x) * (x - 120)

/-- The cost price of the product -/
def cost_price : ℝ := 120

/-- The reference price point -/
def reference_price : ℝ := 130

/-- The daily sales at the reference price -/
def reference_sales : ℝ := 70

/-- The rate of change in sales with respect to price -/
def sales_price_ratio : ℝ := -1

theorem profit_maximization :
  ∃ (max_price max_profit : ℝ),
    (∀ x, profit_function x ≤ max_profit) ∧
    profit_function max_price = max_profit ∧
    max_price = 160 ∧
    max_profit = 1600 := by sorry

theorem profit_function_correct :
  ∀ x, profit_function x = (200 - x) * (x - cost_price) := by sorry

theorem sales_at_price (x : ℝ) :
  x ≥ reference_price →
  profit_function x = (reference_sales + sales_price_ratio * (x - reference_price)) * (x - cost_price) := by sorry

end NUMINAMATH_CALUDE_profit_maximization_profit_function_correct_sales_at_price_l1755_175532


namespace NUMINAMATH_CALUDE_pythagorean_linear_function_l1755_175591

theorem pythagorean_linear_function (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (a^2 + b^2 = c^2) →  -- Pythagorean theorem
  ((-a/c + b/c)^2 = 1/3) →  -- Point (-1, √3/3) lies on y = (a/c)x + (b/c)
  (a * b / 2 = 4) →  -- Area of triangle is 4
  c = 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_linear_function_l1755_175591


namespace NUMINAMATH_CALUDE_power_multiplication_l1755_175583

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1755_175583


namespace NUMINAMATH_CALUDE_average_of_three_quantities_l1755_175500

theorem average_of_three_quantities 
  (total_count : Nat) 
  (total_average : ℚ) 
  (subset_count : Nat) 
  (subset_average : ℚ) 
  (h1 : total_count = 5) 
  (h2 : total_average = 11) 
  (h3 : subset_count = 2) 
  (h4 : subset_average = 21.5) : 
  (total_count * total_average - subset_count * subset_average) / (total_count - subset_count) = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_quantities_l1755_175500


namespace NUMINAMATH_CALUDE_coffee_stock_problem_l1755_175568

/-- Proves that the weight of the second batch of coffee is 100 pounds given the initial conditions --/
theorem coffee_stock_problem (initial_stock : ℝ) (initial_decaf_percent : ℝ) 
  (second_batch_decaf_percent : ℝ) (total_decaf_percent : ℝ) 
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 0.20)
  (h3 : second_batch_decaf_percent = 0.70)
  (h4 : total_decaf_percent = 0.30) : 
  ∃ (second_batch : ℝ), 
    initial_decaf_percent * initial_stock + second_batch_decaf_percent * second_batch = 
    total_decaf_percent * (initial_stock + second_batch) ∧ 
    second_batch = 100 := by
  sorry

end NUMINAMATH_CALUDE_coffee_stock_problem_l1755_175568


namespace NUMINAMATH_CALUDE_ellipse_y_axis_intersection_l1755_175580

/-- Definition of the ellipse with given foci and one intersection point -/
def ellipse (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (-1, 3)
  let F₂ : ℝ × ℝ := (4, 1)
  let P₁ : ℝ × ℝ := (0, 1)
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) + 
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 
  Real.sqrt ((P₁.1 - F₁.1)^2 + (P₁.2 - F₁.2)^2) + 
  Real.sqrt ((P₁.1 - F₂.1)^2 + (P₁.2 - F₂.2)^2)

/-- The theorem stating that (0, -2) is the other intersection point -/
theorem ellipse_y_axis_intersection :
  ∃ (y : ℝ), y ≠ 1 ∧ ellipse (0, y) ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_y_axis_intersection_l1755_175580


namespace NUMINAMATH_CALUDE_binomial_12_choose_6_l1755_175599

theorem binomial_12_choose_6 : Nat.choose 12 6 = 924 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_6_l1755_175599


namespace NUMINAMATH_CALUDE_horse_cattle_price_problem_l1755_175556

theorem horse_cattle_price_problem (x y : ℚ) :
  (4 * x + 6 * y = 48) ∧ (3 * x + 5 * y = 38) →
  x = 6 ∧ y = 4 := by
sorry

end NUMINAMATH_CALUDE_horse_cattle_price_problem_l1755_175556


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1755_175565

theorem matrix_equation_solution (N : Matrix (Fin 2) (Fin 2) ℝ) :
  N * !![2, -3; 4, -1] = !![-8, 7; 20, -11] →
  N = !![-20, -10; 24, 38] := by
sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1755_175565


namespace NUMINAMATH_CALUDE_direction_vector_b_l1755_175540

/-- A line passing through two points with a specific direction vector form -/
def Line (p1 p2 : ℝ × ℝ) (dir : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p2 = (p1.1 + t * dir.1, p1.2 + t * dir.2)

theorem direction_vector_b (b : ℝ) :
  Line (-3, 0) (0, 3) (3, b) → b = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_direction_vector_b_l1755_175540


namespace NUMINAMATH_CALUDE_travel_time_calculation_l1755_175571

/-- Represents the travel times of a motorcyclist and cyclist meeting on a road --/
def TravelTimes (t m c : ℝ) : Prop :=
  t > 0 ∧ 
  m > t ∧ 
  c > t ∧
  m - t = 2 ∧ 
  c - t = 4.5

theorem travel_time_calculation (t m c : ℝ) (h : TravelTimes t m c) : 
  m = 5 ∧ c = 7.5 := by
  sorry

#check travel_time_calculation

end NUMINAMATH_CALUDE_travel_time_calculation_l1755_175571


namespace NUMINAMATH_CALUDE_snake_toy_cost_l1755_175561

theorem snake_toy_cost (cage_cost total_cost : ℚ) (found_money : ℚ) : 
  cage_cost = 14.54 → 
  found_money = 1 → 
  total_cost = 26.3 → 
  total_cost = cage_cost + (12.76 : ℚ) - found_money := by sorry

end NUMINAMATH_CALUDE_snake_toy_cost_l1755_175561


namespace NUMINAMATH_CALUDE_power_product_squared_l1755_175588

theorem power_product_squared (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_squared_l1755_175588


namespace NUMINAMATH_CALUDE_min_triangles_to_cover_l1755_175513

theorem min_triangles_to_cover (side_large : ℝ) (side_small : ℝ) : 
  side_large = 12 → side_small = 1 → 
  (side_large / side_small) ^ 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_min_triangles_to_cover_l1755_175513


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l1755_175502

/-- Represents a sampling method -/
inductive SamplingMethod
  | Simple
  | Stratified
  | Systematic

/-- Represents a population with subgroups -/
structure Population where
  subgroups : List (Set α)
  significant_differences : Bool

/-- Determines the most appropriate sampling method for a given population -/
def most_appropriate_sampling_method (pop : Population) : SamplingMethod :=
  if pop.significant_differences then
    SamplingMethod.Stratified
  else
    SamplingMethod.Simple

/-- Theorem stating that stratified sampling is most appropriate for populations with significant differences between subgroups -/
theorem stratified_sampling_most_appropriate
  (pop : Population)
  (h : pop.significant_differences = true) :
  most_appropriate_sampling_method pop = SamplingMethod.Stratified :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l1755_175502


namespace NUMINAMATH_CALUDE_hostel_provisions_l1755_175584

/-- The number of days the provisions would last for the initial number of men -/
def initial_days : ℕ := 48

/-- The number of days the provisions would last after some men left -/
def final_days : ℕ := 60

/-- The number of men who left the hostel -/
def men_left : ℕ := 50

/-- The initial number of men in the hostel -/
def initial_men : ℕ := 250

theorem hostel_provisions :
  initial_men * initial_days = (initial_men - men_left) * final_days :=
by sorry

end NUMINAMATH_CALUDE_hostel_provisions_l1755_175584


namespace NUMINAMATH_CALUDE_sum_x_y_equals_three_halves_l1755_175569

theorem sum_x_y_equals_three_halves (x y : ℝ) : 
  y = Real.sqrt (3 - 2*x) + Real.sqrt (2*x - 3) → x + y = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_three_halves_l1755_175569


namespace NUMINAMATH_CALUDE_house_problem_theorem_l1755_175542

def house_problem (original_price selling_price years_owned broker_commission_rate
                   property_tax_rate yearly_maintenance mortgage_interest_rate : ℝ) : Prop :=
  let profit_rate := (selling_price - original_price) / original_price
  let broker_commission := broker_commission_rate * original_price
  let total_property_tax := property_tax_rate * original_price * years_owned
  let total_maintenance := yearly_maintenance * years_owned
  let total_mortgage_interest := mortgage_interest_rate * original_price * years_owned
  let total_costs := broker_commission + total_property_tax + total_maintenance + total_mortgage_interest
  let net_profit := selling_price - original_price - total_costs
  (original_price = 80000) ∧
  (years_owned = 5) ∧
  (profit_rate = 0.2) ∧
  (broker_commission_rate = 0.05) ∧
  (property_tax_rate = 0.012) ∧
  (yearly_maintenance = 1500) ∧
  (mortgage_interest_rate = 0.04) →
  net_profit = -16300

theorem house_problem_theorem :
  ∀ (original_price selling_price years_owned broker_commission_rate
     property_tax_rate yearly_maintenance mortgage_interest_rate : ℝ),
  house_problem original_price selling_price years_owned broker_commission_rate
                 property_tax_rate yearly_maintenance mortgage_interest_rate :=
by
  sorry

end NUMINAMATH_CALUDE_house_problem_theorem_l1755_175542


namespace NUMINAMATH_CALUDE_inequality_proofs_l1755_175578

theorem inequality_proofs :
  (∀ (a b : ℝ), a > 0 → b > 0 → (b/a) + (a/b) ≥ 2) ∧
  (∀ (x y : ℝ), x*y < 0 → (x/y) + (y/x) ≤ -2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proofs_l1755_175578


namespace NUMINAMATH_CALUDE_integral_minus_x_squared_plus_one_l1755_175596

theorem integral_minus_x_squared_plus_one : ∫ (x : ℝ) in (0)..(1), -x^2 + 1 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_minus_x_squared_plus_one_l1755_175596


namespace NUMINAMATH_CALUDE_triangle_area_l1755_175530

/-- A triangle with sides of length 9, 40, and 41 has an area of 180. -/
theorem triangle_area (a b c : ℝ) (ha : a = 9) (hb : b = 40) (hc : c = 41) : 
  (1/2) * a * b = 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1755_175530


namespace NUMINAMATH_CALUDE_find_k_l1755_175520

theorem find_k (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
    (h1 : a^2 + b^2 + c^2 = 49)
    (h2 : x^2 + y^2 + z^2 = 64)
    (h3 : a*x + b*y + c*z = 56)
    (h4 : ∃ k, a = k*x ∧ b = k*y ∧ c = k*z) :
  ∃ k, a = k*x ∧ b = k*y ∧ c = k*z ∧ k = 7/8 := by
sorry

end NUMINAMATH_CALUDE_find_k_l1755_175520


namespace NUMINAMATH_CALUDE_triangle_parallelogram_properties_l1755_175577

/-- A triangle with a parallelogram inscribed in it. -/
structure TriangleWithParallelogram where
  /-- The length of the first side of the triangle -/
  side1 : ℝ
  /-- The length of the second side of the triangle -/
  side2 : ℝ
  /-- The length of the first side of the parallelogram -/
  para_side1 : ℝ
  /-- Assumption that the first side of the triangle is 9 -/
  h1 : side1 = 9
  /-- Assumption that the second side of the triangle is 15 -/
  h2 : side2 = 15
  /-- Assumption that the first side of the parallelogram is 6 -/
  h3 : para_side1 = 6
  /-- Assumption that the parallelogram is inscribed in the triangle -/
  h4 : para_side1 ≤ side1 ∧ para_side1 ≤ side2
  /-- Assumption that the diagonals of the parallelogram are parallel to the sides of the triangle -/
  h5 : True  -- This is a placeholder as we can't directly represent this geometrical property

/-- The theorem stating the properties of the triangle and parallelogram -/
theorem triangle_parallelogram_properties (tp : TriangleWithParallelogram) :
  ∃ (para_side2 triangle_side3 : ℝ),
    para_side2 = 4 * Real.sqrt 2 ∧
    triangle_side3 = 18 :=
  sorry

end NUMINAMATH_CALUDE_triangle_parallelogram_properties_l1755_175577


namespace NUMINAMATH_CALUDE_shooting_probabilities_l1755_175522

-- Define the probabilities
def prob_A : ℚ := 1/2
def prob_B : ℚ := 1/3

-- Define the event of hitting the target exactly twice
def hit_twice : ℚ := prob_A * prob_B

-- Define the event of hitting the target at least once
def hit_at_least_once : ℚ := 1 - (1 - prob_A) * (1 - prob_B)

-- Theorem to prove
theorem shooting_probabilities :
  (hit_twice = 1/6) ∧ (hit_at_least_once = 1 - 1/2 * 2/3) :=
sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l1755_175522


namespace NUMINAMATH_CALUDE_pencil_sharpening_l1755_175597

theorem pencil_sharpening (original_length sharpened_length : ℕ) 
  (h1 : original_length = 31)
  (h2 : sharpened_length = 14) :
  original_length - sharpened_length = 17 := by
  sorry

end NUMINAMATH_CALUDE_pencil_sharpening_l1755_175597


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1755_175592

theorem quadratic_always_positive (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * x + 2 > 0) → a > 9/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1755_175592


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1755_175555

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 + 8*x - 12 = 0 → ∃ r₁ r₂ : ℝ, r₁ + r₂ = -8 ∧ r₁ * r₂ = -12 ∧ r₁^2 + r₂^2 = 88 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1755_175555


namespace NUMINAMATH_CALUDE_sum_of_root_products_l1755_175521

theorem sum_of_root_products (p q r s : ℂ) : 
  (4 * p^4 - 8 * p^3 + 12 * p^2 - 16 * p + 9 = 0) →
  (4 * q^4 - 8 * q^3 + 12 * q^2 - 16 * q + 9 = 0) →
  (4 * r^4 - 8 * r^3 + 12 * r^2 - 16 * r + 9 = 0) →
  (4 * s^4 - 8 * s^3 + 12 * s^2 - 16 * s + 9 = 0) →
  p * q + p * r + p * s + q * r + q * s + r * s = -3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_root_products_l1755_175521


namespace NUMINAMATH_CALUDE_cyclist_club_members_count_l1755_175512

/-- The set of digits that can be used in the identification numbers. -/
def ValidDigits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 9}

/-- The number of digits in each identification number. -/
def IdentificationNumberLength : Nat := 3

/-- The total number of possible identification numbers. -/
def TotalIdentificationNumbers : Nat := ValidDigits.card ^ IdentificationNumberLength

/-- Theorem stating that the total number of possible identification numbers is 512. -/
theorem cyclist_club_members_count :
  TotalIdentificationNumbers = 512 := by sorry

end NUMINAMATH_CALUDE_cyclist_club_members_count_l1755_175512


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l1755_175563

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 6

-- Define the solution set condition
def solution_set (a b : ℝ) : Set ℝ := {x | x < 1 ∨ x > b}

-- Define the theorem
theorem quadratic_inequality_theorem (a b : ℝ) :
  (∀ x, f a x > 4 ↔ x ∈ solution_set a b) →
  (a = 1 ∧ b = 2) ∧
  (∀ c, 
    let g (x : ℝ) := a * x^2 - (a * c + b) * x + b * c
    if c > 2 then
      {x | g x < 0} = {x | 2 < x ∧ x < c}
    else if c < 2 then
      {x | g x < 0} = {x | c < x ∧ x < 2}
    else
      {x | g x < 0} = ∅) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l1755_175563


namespace NUMINAMATH_CALUDE_students_drawn_from_C_l1755_175544

/-- Represents the total number of students in the college -/
def total_students : ℕ := 1500

/-- Represents the planned sample size -/
def sample_size : ℕ := 150

/-- Represents the number of students in major A -/
def students_major_A : ℕ := 420

/-- Represents the number of students in major B -/
def students_major_B : ℕ := 580

/-- Theorem stating the number of students to be drawn from major C -/
theorem students_drawn_from_C : 
  (sample_size : ℚ) / total_students * (total_students - students_major_A - students_major_B) = 50 :=
by sorry

end NUMINAMATH_CALUDE_students_drawn_from_C_l1755_175544


namespace NUMINAMATH_CALUDE_product_remainder_l1755_175541

theorem product_remainder (a b c : ℕ) (h : a * b * c = 1225 * 1227 * 1229) : 
  (a * b * c) % 12 = 7 := by
sorry

end NUMINAMATH_CALUDE_product_remainder_l1755_175541


namespace NUMINAMATH_CALUDE_infinite_non_triangular_arithmetic_sequence_l1755_175536

-- Define triangular numbers
def isTriangular (k : ℕ) : Prop :=
  ∃ n : ℕ, k = n * (n - 1) / 2

-- Define an arithmetic sequence
def isArithmeticSequence (s : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, s (n + 1) = s n + d

-- Theorem statement
theorem infinite_non_triangular_arithmetic_sequence :
  ∃ s : ℕ → ℕ, isArithmeticSequence s ∧ (∀ n : ℕ, ¬ isTriangular (s n)) :=
sorry

end NUMINAMATH_CALUDE_infinite_non_triangular_arithmetic_sequence_l1755_175536


namespace NUMINAMATH_CALUDE_value_of_expression_l1755_175535

theorem value_of_expression (a b : ℝ) 
  (h1 : |a| = 2) 
  (h2 : |-b| = 5) 
  (h3 : a < b) : 
  2*a - 3*b = -11 ∨ 2*a - 3*b = -19 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1755_175535


namespace NUMINAMATH_CALUDE_vince_ride_length_l1755_175510

-- Define the length of Zachary's bus ride
def zachary_ride : ℝ := 0.5

-- Define how much longer Vince's ride is compared to Zachary's
def difference : ℝ := 0.13

-- Define Vince's bus ride length
def vince_ride : ℝ := zachary_ride + difference

-- Theorem statement
theorem vince_ride_length : vince_ride = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_vince_ride_length_l1755_175510


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_seven_l1755_175506

theorem gcd_of_powers_of_seven : Nat.gcd (7^11 + 1) (7^11 + 7^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_seven_l1755_175506


namespace NUMINAMATH_CALUDE_tennis_balls_per_pack_l1755_175562

theorem tennis_balls_per_pack (num_packs : ℕ) (total_cost : ℕ) (cost_per_ball : ℕ) : 
  num_packs = 4 → total_cost = 24 → cost_per_ball = 2 → 
  (total_cost / cost_per_ball) / num_packs = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_tennis_balls_per_pack_l1755_175562


namespace NUMINAMATH_CALUDE_tan_315_degrees_l1755_175587

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l1755_175587


namespace NUMINAMATH_CALUDE_first_complete_column_coverage_l1755_175579

theorem first_complete_column_coverage : ∃ n : ℕ, 
  n = 32 ∧ 
  (∀ k ≤ n, ∃ m ≤ n, m * (m + 1) / 2 % 12 = k % 12) ∧
  (∀ j < n, ¬(∀ k ≤ 11, ∃ m ≤ j, m * (m + 1) / 2 % 12 = k % 12)) := by
  sorry

end NUMINAMATH_CALUDE_first_complete_column_coverage_l1755_175579


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1755_175557

theorem smallest_n_congruence (n : ℕ) : n = 11 ↔ 
  (n > 0 ∧ 19 * n ≡ 546 [ZMOD 13] ∧ 
   ∀ m : ℕ, m > 0 ∧ m < n → ¬(19 * m ≡ 546 [ZMOD 13])) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1755_175557


namespace NUMINAMATH_CALUDE_fraction_addition_l1755_175594

theorem fraction_addition (d : ℝ) : (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1755_175594


namespace NUMINAMATH_CALUDE_book_cost_l1755_175546

theorem book_cost (total_paid : ℕ) (change : ℕ) (pen_cost : ℕ) (ruler_cost : ℕ) 
  (h1 : total_paid = 50)
  (h2 : change = 20)
  (h3 : pen_cost = 4)
  (h4 : ruler_cost = 1) :
  total_paid - change - (pen_cost + ruler_cost) = 25 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_l1755_175546


namespace NUMINAMATH_CALUDE_special_sum_value_l1755_175519

theorem special_sum_value (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ + 13*x₇ = 0)
  (eq2 : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ + 15*x₇ = 10)
  (eq3 : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ + 17*x₇ = 104) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ + 19*x₇ = 282 :=
by
  sorry

end NUMINAMATH_CALUDE_special_sum_value_l1755_175519


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1755_175528

theorem smallest_multiple_of_6_and_15 (b : ℕ) : 
  (∃ k : ℕ, b = 6 * k) ∧ 
  (∃ m : ℕ, b = 15 * m) ∧ 
  (∀ c : ℕ, c > 0 ∧ (∃ p : ℕ, c = 6 * p) ∧ (∃ q : ℕ, c = 15 * q) → b ≤ c) →
  b = 30 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1755_175528


namespace NUMINAMATH_CALUDE_complex_quadrant_l1755_175538

theorem complex_quadrant (z : ℂ) (h : z * (1 - Complex.I) = 2 * Complex.I) :
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l1755_175538


namespace NUMINAMATH_CALUDE_min_value_of_trig_function_l1755_175560

theorem min_value_of_trig_function :
  let y (x : ℝ) := Real.tan (x + π/3) - Real.tan (x + π/4) + Real.sin (x + π/4)
  ∀ x ∈ Set.Icc (-π/2) (-π/4), y x ≥ 1 ∧ ∃ x₀ ∈ Set.Icc (-π/2) (-π/4), y x₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_trig_function_l1755_175560


namespace NUMINAMATH_CALUDE_sum_15_27_in_base4_l1755_175545

/-- Converts a natural number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 4 to a natural number -/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_15_27_in_base4 :
  toBase4 (15 + 27) = [2, 2, 2] :=
sorry

end NUMINAMATH_CALUDE_sum_15_27_in_base4_l1755_175545


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l1755_175504

/-- The nth term of a geometric sequence with first term a and common ratio r -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r ^ (n - 1)

/-- The 8th term of a geometric sequence with first term 27 and common ratio 2/3 -/
theorem eighth_term_of_sequence :
  geometric_sequence 27 (2/3) 8 = 128/81 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l1755_175504


namespace NUMINAMATH_CALUDE_job_completion_time_l1755_175559

theorem job_completion_time (a d : ℝ) : 
  (a > 0) → 
  (d > 0) → 
  (1 / a + 1 / d = 1 / 5) → 
  (d = 10) → 
  (a = 10) := by sorry

end NUMINAMATH_CALUDE_job_completion_time_l1755_175559


namespace NUMINAMATH_CALUDE_percentage_relationship_l1755_175574

theorem percentage_relationship (p j t : ℝ) (r : ℝ) 
  (h1 : j = p * (1 - 0.25))
  (h2 : j = t * (1 - 0.20))
  (h3 : t = p * (1 - r / 100)) :
  r = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relationship_l1755_175574


namespace NUMINAMATH_CALUDE_unknown_score_value_l1755_175589

/-- Given 5 scores where 4 are known and the average of all 5 scores is 9.3, 
    prove that the unknown score x must be 9.5 -/
theorem unknown_score_value (s1 s2 s3 s4 : ℝ) (x : ℝ) (h_s1 : s1 = 9.1) 
    (h_s2 : s2 = 9.3) (h_s3 : s3 = 9.2) (h_s4 : s4 = 9.4) 
    (h_avg : (s1 + s2 + x + s3 + s4) / 5 = 9.3) : x = 9.5 := by
  sorry

#check unknown_score_value

end NUMINAMATH_CALUDE_unknown_score_value_l1755_175589


namespace NUMINAMATH_CALUDE_cylinder_volume_scaling_l1755_175593

/-- Proves that doubling both the radius and height of a cylindrical container increases its volume by a factor of 8 -/
theorem cylinder_volume_scaling (r h V : ℝ) (h1 : V = Real.pi * r^2 * h) :
  Real.pi * (2*r)^2 * (2*h) = 8 * V := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_scaling_l1755_175593


namespace NUMINAMATH_CALUDE_circle_symmetry_l1755_175518

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 2016

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 2016

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ),
  original_circle x y ∧ symmetry_line x y →
  ∃ (x' y' : ℝ), symmetric_circle x' y' ∧ symmetry_line ((x + x') / 2) ((y + y') / 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1755_175518


namespace NUMINAMATH_CALUDE_divisibility_problem_l1755_175566

theorem divisibility_problem (n : ℤ) (h : n % 30 = 16) : (2 * n) % 30 = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1755_175566


namespace NUMINAMATH_CALUDE_quarters_needed_l1755_175553

/-- Represents the cost of items in cents -/
def CandyBarCost : ℕ := 25
def ChocolatePieceCost : ℕ := 75
def JuicePackCost : ℕ := 50

/-- Represents the number of each item to be purchased -/
def CandyBarCount : ℕ := 3
def ChocolatePieceCount : ℕ := 2
def JuicePackCount : ℕ := 1

/-- Represents the value of a quarter in cents -/
def QuarterValue : ℕ := 25

/-- Calculates the total cost in cents -/
def TotalCost : ℕ := 
  CandyBarCost * CandyBarCount + 
  ChocolatePieceCost * ChocolatePieceCount + 
  JuicePackCost * JuicePackCount

/-- Theorem: The number of quarters needed is 11 -/
theorem quarters_needed : TotalCost / QuarterValue = 11 := by
  sorry

end NUMINAMATH_CALUDE_quarters_needed_l1755_175553


namespace NUMINAMATH_CALUDE_cookie_bags_count_l1755_175503

/-- Given a total number of cookies and the fact that each bag contains an equal number of cookies,
    prove that the number of bags is 14. -/
theorem cookie_bags_count (total_cookies : ℕ) (cookies_per_bag : ℕ) (total_candies : ℕ) :
  total_cookies = 28 →
  cookies_per_bag > 0 →
  total_cookies = 14 * cookies_per_bag →
  (∃ (num_bags : ℕ), num_bags = 14 ∧ num_bags * cookies_per_bag = total_cookies) :=
by sorry

end NUMINAMATH_CALUDE_cookie_bags_count_l1755_175503


namespace NUMINAMATH_CALUDE_division_problem_l1755_175514

theorem division_problem : (786 * 74) / 30 = 1938.8 := by sorry

end NUMINAMATH_CALUDE_division_problem_l1755_175514


namespace NUMINAMATH_CALUDE_morning_bikes_count_l1755_175505

/-- The number of bikes sold in the morning -/
def morning_bikes : ℕ := 19

/-- The number of bikes sold in the afternoon -/
def afternoon_bikes : ℕ := 27

/-- The number of bike clamps given with each bike -/
def clamps_per_bike : ℕ := 2

/-- The total number of bike clamps given away -/
def total_clamps : ℕ := 92

/-- Theorem stating that the number of bikes sold in the morning is 19 -/
theorem morning_bikes_count : 
  morning_bikes = 19 ∧ 
  clamps_per_bike * (morning_bikes + afternoon_bikes) = total_clamps := by
  sorry

end NUMINAMATH_CALUDE_morning_bikes_count_l1755_175505


namespace NUMINAMATH_CALUDE_gcf_lcm_problem_l1755_175529

theorem gcf_lcm_problem : Nat.gcd (Nat.lcm 18 30) (Nat.lcm 21 28) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_problem_l1755_175529


namespace NUMINAMATH_CALUDE_range_of_a_l1755_175539

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (a > 1) →
  (∀ x ≤ 2, ∀ y ≤ 2, x < y → f a y < f a x) →
  (∀ x ∈ Set.Icc 1 (a + 1), ∀ y ∈ Set.Icc 1 (a + 1), |f a x - f a y| ≤ 4) →
  a ∈ Set.Icc 2 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1755_175539


namespace NUMINAMATH_CALUDE_player5_score_l1755_175525

/-- Represents a basketball player's score breakdown -/
structure PlayerScore where
  twoPointers : Nat
  threePointers : Nat
  freeThrows : Nat

/-- Calculates the total points scored by a player -/
def totalPoints (score : PlayerScore) : Nat :=
  2 * score.twoPointers + 3 * score.threePointers + score.freeThrows

theorem player5_score 
  (teamAScore : Nat)
  (player1 : PlayerScore)
  (player2 : PlayerScore)
  (player3 : PlayerScore)
  (player4 : PlayerScore)
  (h1 : teamAScore = 75)
  (h2 : player1 = ⟨0, 5, 0⟩)
  (h3 : player2 = ⟨5, 0, 5⟩)
  (h4 : player3 = ⟨0, 3, 3⟩)
  (h5 : player4 = ⟨6, 0, 0⟩) :
  teamAScore - (totalPoints player1 + totalPoints player2 + totalPoints player3 + totalPoints player4) = 14 := by
  sorry

#eval totalPoints ⟨0, 5, 0⟩  -- Player 1
#eval totalPoints ⟨5, 0, 5⟩  -- Player 2
#eval totalPoints ⟨0, 3, 3⟩  -- Player 3
#eval totalPoints ⟨6, 0, 0⟩  -- Player 4

end NUMINAMATH_CALUDE_player5_score_l1755_175525


namespace NUMINAMATH_CALUDE_irreducible_polynomial_l1755_175526

def S : Finset ℕ := {54, 72, 36, 108}

def is_permutation (b₀ b₁ b₂ b₃ : ℕ) : Prop :=
  {b₀, b₁, b₂, b₃} = S

def polynomial (b₀ b₁ b₂ b₃ : ℕ) (x : ℤ) : ℤ :=
  x^5 + b₃*x^3 + b₂*x^2 + b₁*x + b₀

theorem irreducible_polynomial (b₀ b₁ b₂ b₃ : ℕ) :
  is_permutation b₀ b₁ b₂ b₃ →
  Irreducible (polynomial b₀ b₁ b₂ b₃) :=
by sorry

end NUMINAMATH_CALUDE_irreducible_polynomial_l1755_175526


namespace NUMINAMATH_CALUDE_dollar_to_yen_exchange_l1755_175585

/-- Given an exchange rate where 5000 yen equals 48 dollars, 
    prove that 3 dollars can be exchanged for 312.5 yen. -/
theorem dollar_to_yen_exchange : 
  ∀ (dollar_to_yen : ℝ → ℝ),
  (dollar_to_yen 48 = 5000) →  -- Exchange rate condition
  (dollar_to_yen 3 = 312.5) :=  -- What we want to prove
by
  sorry

end NUMINAMATH_CALUDE_dollar_to_yen_exchange_l1755_175585


namespace NUMINAMATH_CALUDE_function_growth_l1755_175570

theorem function_growth (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x > f x) (a : ℝ) (ha : a > 0) : 
  f a > Real.exp a * f 0 :=
sorry

end NUMINAMATH_CALUDE_function_growth_l1755_175570


namespace NUMINAMATH_CALUDE_max_x_value_l1755_175509

theorem max_x_value (x : ℝ) : 
  (((5*x - 20) / (4*x - 5))^2 + ((5*x - 20) / (4*x - 5)) = 20) → 
  x ≤ 9/5 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l1755_175509


namespace NUMINAMATH_CALUDE_initial_birds_on_fence_l1755_175582

theorem initial_birds_on_fence (initial_birds storks additional_birds final_birds : ℕ) :
  initial_birds + storks > 0 ∧ 
  storks = 46 ∧ 
  additional_birds = 6 ∧ 
  final_birds = 10 ∧ 
  initial_birds + additional_birds = final_birds →
  initial_birds = 4 := by
sorry

end NUMINAMATH_CALUDE_initial_birds_on_fence_l1755_175582


namespace NUMINAMATH_CALUDE_department_store_discount_rate_l1755_175517

/-- Represents the discount rate calculation for a department store purchase --/
theorem department_store_discount_rate : 
  -- Define the prices of items
  let shoe_price : ℚ := 74
  let sock_price : ℚ := 2
  let bag_price : ℚ := 42
  let sock_quantity : ℕ := 2
  
  -- Calculate total price before discount
  let total_before_discount : ℚ := shoe_price + sock_price * sock_quantity + bag_price
  
  -- Define the threshold for discount application
  let discount_threshold : ℚ := 100
  
  -- Define the amount paid by Jaco
  let amount_paid : ℚ := 118
  
  -- Calculate the discount amount
  let discount_amount : ℚ := total_before_discount - amount_paid
  
  -- Calculate the amount subject to discount
  let amount_subject_to_discount : ℚ := total_before_discount - discount_threshold
  
  -- Calculate the discount rate
  let discount_rate : ℚ := discount_amount / amount_subject_to_discount * 100
  
  discount_rate = 10 := by sorry

end NUMINAMATH_CALUDE_department_store_discount_rate_l1755_175517


namespace NUMINAMATH_CALUDE_equation_equivalence_and_solutions_l1755_175549

theorem equation_equivalence_and_solutions (x y z : ℤ) : 
  (x * (x - y) + y * (y - x) + z * (z - y) = 1) ↔ 
  ((x - y)^2 + (z - y)^2 = 1) ∧
  ((x = y - 1 ∧ z = y + 1) ∨ (x = y ∧ z = y + 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_and_solutions_l1755_175549


namespace NUMINAMATH_CALUDE_snail_square_exists_l1755_175501

/-- A natural number is a "snail" number if it can be formed by concatenating
    three consecutive natural numbers in some order. -/
def is_snail (n : ℕ) : Prop :=
  ∃ a b c : ℕ, b = a + 1 ∧ c = b + 1 ∧
  (n.repr = a.repr ++ b.repr ++ c.repr ∨
   n.repr = a.repr ++ c.repr ++ b.repr ∨
   n.repr = b.repr ++ a.repr ++ c.repr ∨
   n.repr = b.repr ++ c.repr ++ a.repr ∨
   n.repr = c.repr ++ a.repr ++ b.repr ∨
   n.repr = c.repr ++ b.repr ++ a.repr)

theorem snail_square_exists :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ is_snail n ∧ ∃ m : ℕ, n = m^2 :=
by
  use 1089
  sorry

end NUMINAMATH_CALUDE_snail_square_exists_l1755_175501


namespace NUMINAMATH_CALUDE_integer_solutions_inequality_l1755_175534

theorem integer_solutions_inequality (a : ℝ) (h_pos : a > 0) 
  (h_three_solutions : ∃ x y z : ℤ, x < y ∧ y < z ∧ 
    (∀ w : ℤ, 1 < w * a ∧ w * a < 2 ↔ w = x ∨ w = y ∨ w = z)) :
  ∃ p q r : ℤ, p < q ∧ q < r ∧ 
    (∀ w : ℤ, 2 < w * a ∧ w * a < 3 ↔ w = p ∨ w = q ∨ w = r) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_inequality_l1755_175534


namespace NUMINAMATH_CALUDE_town_population_problem_l1755_175581

theorem town_population_problem (p : ℕ) : 
  (p + 1500 : ℝ) * 0.8 = p + 1500 + 50 → p = 1750 := by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l1755_175581


namespace NUMINAMATH_CALUDE_octal_135_equals_binary_1011101_l1755_175511

-- Define a function to convert octal to binary
def octal_to_binary (octal : ℕ) : ℕ := sorry

-- State the theorem
theorem octal_135_equals_binary_1011101 :
  octal_to_binary 135 = 1011101 := by sorry

end NUMINAMATH_CALUDE_octal_135_equals_binary_1011101_l1755_175511


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l1755_175507

theorem max_product_sum_2000 : 
  ∀ x y : ℤ, x + y = 2000 → x * y ≤ 1000000 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l1755_175507


namespace NUMINAMATH_CALUDE_gold_silver_price_ratio_l1755_175573

/-- Proves that the ratio of gold price to silver price per ounce is 50 -/
theorem gold_silver_price_ratio :
  let silver_amount : Real := 1.5
  let gold_amount : Real := 2 * silver_amount
  let silver_price : Real := 20
  let total_spent : Real := 3030
  let gold_price := (total_spent - silver_amount * silver_price) / gold_amount
  gold_price / silver_price = 50 := by sorry

end NUMINAMATH_CALUDE_gold_silver_price_ratio_l1755_175573


namespace NUMINAMATH_CALUDE_no_dapper_numbers_l1755_175515

/-- A two-digit positive integer is 'dapper' if it equals the sum of its nonzero tens digit and the cube of its units digit. -/
def is_dapper (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ (a b : ℕ), n = 10 * a + b ∧ a ≠ 0 ∧ n = a + b^3

/-- There are no two-digit positive integers that are 'dapper'. -/
theorem no_dapper_numbers : ¬∃ (n : ℕ), is_dapper n := by
  sorry

#check no_dapper_numbers

end NUMINAMATH_CALUDE_no_dapper_numbers_l1755_175515


namespace NUMINAMATH_CALUDE_nina_money_theorem_l1755_175586

theorem nina_money_theorem (x : ℝ) 
  (h1 : 6 * x = 8 * (x - 1.5)) : 6 * x = 36 := by
  sorry

end NUMINAMATH_CALUDE_nina_money_theorem_l1755_175586


namespace NUMINAMATH_CALUDE_total_socks_is_51_l1755_175543

/-- Calculates the total number of socks John and Mary have after throwing away and buying new socks -/
def totalSocksAfterChanges (johnInitial : ℕ) (maryInitial : ℕ) (johnThrowAway : ℕ) (johnBuy : ℕ) (maryThrowAway : ℕ) (maryBuy : ℕ) : ℕ :=
  (johnInitial - johnThrowAway + johnBuy) + (maryInitial - maryThrowAway + maryBuy)

/-- Proves that John and Mary have 51 socks in total after the changes -/
theorem total_socks_is_51 :
  totalSocksAfterChanges 33 20 19 13 6 10 = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_socks_is_51_l1755_175543


namespace NUMINAMATH_CALUDE_range_of_a_l1755_175572

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc (-2) 3 ∧ 2 * x - x^2 ≥ a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1755_175572


namespace NUMINAMATH_CALUDE_correct_food_suggestion_ratio_l1755_175523

/-- The ratio of food suggestions by students -/
def food_suggestion_ratio (sushi mashed_potatoes bacon tomatoes : ℕ) : List ℕ :=
  [sushi, mashed_potatoes, bacon, tomatoes]

/-- Theorem stating the correct ratio of food suggestions -/
theorem correct_food_suggestion_ratio :
  food_suggestion_ratio 297 144 467 79 = [297, 144, 467, 79] := by
  sorry

end NUMINAMATH_CALUDE_correct_food_suggestion_ratio_l1755_175523


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_exists_l1755_175524

/-- Given a quadratic polynomial x^2 - px + q with roots r₁ and r₂ satisfying
    r₁ + r₂ = r₁² + r₂² = r₁⁴ + r₂⁴, there exists a maximum value for 1/r₁⁵ + 1/r₂⁵ -/
theorem max_reciprocal_sum_exists (p q r₁ r₂ : ℝ) : 
  (r₁ * r₁ - p * r₁ + q = 0) →
  (r₂ * r₂ - p * r₂ + q = 0) →
  (r₁ + r₂ = r₁^2 + r₂^2) →
  (r₁ + r₂ = r₁^4 + r₂^4) →
  ∃ (M : ℝ), ∀ (s₁ s₂ : ℝ), 
    (s₁ * s₁ - p * s₁ + q = 0) →
    (s₂ * s₂ - p * s₂ + q = 0) →
    (s₁ + s₂ = s₁^2 + s₂^2) →
    (s₁ + s₂ = s₁^4 + s₂^4) →
    1/s₁^5 + 1/s₂^5 ≤ M :=
by
  sorry


end NUMINAMATH_CALUDE_max_reciprocal_sum_exists_l1755_175524


namespace NUMINAMATH_CALUDE_triangle_existence_l1755_175567

theorem triangle_existence (k : ℕ) (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_k : k = 6) 
  (h_ineq : k * (x * y + y * z + z * x) > 5 * (x^2 + y^2 + z^2)) : 
  x + y > z ∧ y + z > x ∧ z + x > y := by
sorry

end NUMINAMATH_CALUDE_triangle_existence_l1755_175567


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_81_l1755_175550

theorem factor_t_squared_minus_81 (t : ℝ) : t^2 - 81 = (t - 9) * (t + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_81_l1755_175550


namespace NUMINAMATH_CALUDE_willie_stickers_l1755_175533

/-- Given Willie starts with 124 stickers and gives away 23, prove he ends up with 101 stickers. -/
theorem willie_stickers : 
  let initial_stickers : ℕ := 124
  let given_away : ℕ := 23
  initial_stickers - given_away = 101 := by
  sorry

end NUMINAMATH_CALUDE_willie_stickers_l1755_175533


namespace NUMINAMATH_CALUDE_daniels_improvement_l1755_175590

/-- Represents the jogging data for Daniel -/
structure JoggingData where
  initial_laps : ℕ
  initial_time : ℕ  -- in minutes
  final_laps : ℕ
  final_time : ℕ    -- in minutes

/-- Calculates the improvement in lap time (in seconds) given jogging data -/
def lapTimeImprovement (data : JoggingData) : ℕ :=
  let initial_lap_time := (data.initial_time * 60) / data.initial_laps
  let final_lap_time := (data.final_time * 60) / data.final_laps
  initial_lap_time - final_lap_time

/-- Theorem stating that Daniel's lap time improvement is 20 seconds -/
theorem daniels_improvement (data : JoggingData) 
  (h1 : data.initial_laps = 15) 
  (h2 : data.initial_time = 40)
  (h3 : data.final_laps = 18)
  (h4 : data.final_time = 42) : 
  lapTimeImprovement data = 20 := by
  sorry

end NUMINAMATH_CALUDE_daniels_improvement_l1755_175590


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_for_2023_l1755_175558

def is_sum_of_distinct_powers_of_two (n : ℕ) (exponents : List ℕ) : Prop :=
  n = (exponents.map (fun e => 2^e)).sum ∧ exponents.Nodup

theorem least_sum_of_exponents_for_2023 :
  ∃ (exponents : List ℕ),
    is_sum_of_distinct_powers_of_two 2023 exponents ∧
    ∀ (other_exponents : List ℕ),
      is_sum_of_distinct_powers_of_two 2023 other_exponents →
      exponents.sum ≤ other_exponents.sum ∧
      exponents.sum = 48 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_for_2023_l1755_175558


namespace NUMINAMATH_CALUDE_shooting_competition_probabilities_l1755_175516

-- Define the probabilities for A and B hitting different rings
def prob_A_8 : ℝ := 0.6
def prob_A_9 : ℝ := 0.3
def prob_A_10 : ℝ := 0.1
def prob_B_8 : ℝ := 0.4
def prob_B_9 : ℝ := 0.4
def prob_B_10 : ℝ := 0.2

-- Define the probability that A hits more rings than B in a single round
def prob_A_beats_B : ℝ := prob_A_9 * prob_B_8 + prob_A_10 * prob_B_8 + prob_A_10 * prob_B_9

-- Define the probability that A hits more rings than B in at least two out of three independent rounds
def prob_A_beats_B_twice_or_more : ℝ :=
  3 * prob_A_beats_B^2 * (1 - prob_A_beats_B) + prob_A_beats_B^3

theorem shooting_competition_probabilities :
  prob_A_beats_B = 0.2 ∧ prob_A_beats_B_twice_or_more = 0.104 := by
  sorry

end NUMINAMATH_CALUDE_shooting_competition_probabilities_l1755_175516


namespace NUMINAMATH_CALUDE_mapping_preimage_property_l1755_175508

theorem mapping_preimage_property (A B : Type) (f : A → B) :
  ∃ (b : B), ∃ (a1 a2 : A), a1 ≠ a2 ∧ f a1 = b ∧ f a2 = b :=
sorry

end NUMINAMATH_CALUDE_mapping_preimage_property_l1755_175508


namespace NUMINAMATH_CALUDE_geography_textbook_cost_l1755_175537

/-- The cost of a geography textbook given the following conditions:
  1. 35 English textbooks and 35 geography textbooks are ordered
  2. An English book costs $7.50
  3. The total amount of the order is $630
-/
theorem geography_textbook_cost :
  let num_books : ℕ := 35
  let english_book_cost : ℚ := 7.5
  let total_cost : ℚ := 630
  let geography_book_cost : ℚ := (total_cost - num_books * english_book_cost) / num_books
  geography_book_cost = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_geography_textbook_cost_l1755_175537


namespace NUMINAMATH_CALUDE_cookie_pack_cost_l1755_175527

/-- Proves the cost of each pack of cookies given Patty's chore arrangement with her siblings --/
theorem cookie_pack_cost 
  (cookies_per_chore : ℕ)
  (chores_per_week : ℕ)
  (num_siblings : ℕ)
  (num_weeks : ℕ)
  (cookies_per_pack : ℕ)
  (total_money : ℚ)
  (h1 : cookies_per_chore = 3)
  (h2 : chores_per_week = 4)
  (h3 : num_siblings = 2)
  (h4 : num_weeks = 10)
  (h5 : cookies_per_pack = 24)
  (h6 : total_money = 15) :
  (total_money / (num_siblings * chores_per_week * num_weeks * cookies_per_chore / cookies_per_pack) : ℚ) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_cookie_pack_cost_l1755_175527


namespace NUMINAMATH_CALUDE_hemisphere_base_area_l1755_175547

/-- If the surface area of a hemisphere is 9, then the area of its base is 3 -/
theorem hemisphere_base_area (r : ℝ) (h : 3 * Real.pi * r^2 = 9) : Real.pi * r^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_base_area_l1755_175547


namespace NUMINAMATH_CALUDE_set_intersection_equality_l1755_175548

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | ∃ k : ℤ, x = k}

theorem set_intersection_equality :
  (Aᶜ ∩ B ∩ {x : ℝ | -2 ≤ x ∧ x ≤ 0}) = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l1755_175548


namespace NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_a_l1755_175598

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

-- Theorem for the solution set of f(x) ≤ 6
theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = Set.Icc (-1) 2 := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∃ x, f x < |a - 1|} = {a : ℝ | a > 5 ∨ a < -3} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_a_l1755_175598


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l1755_175595

theorem square_root_of_sixteen : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l1755_175595


namespace NUMINAMATH_CALUDE_binomial_square_exists_l1755_175552

theorem binomial_square_exists : ∃ (b t u : ℝ), ∀ x : ℝ, b * x^2 + 12 * x + 9 = (t * x + u)^2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_exists_l1755_175552


namespace NUMINAMATH_CALUDE_area_under_sine_curve_l1755_175575

theorem area_under_sine_curve : 
  let lower_bound : ℝ := 0
  let upper_bound : ℝ := 2 * π / 3
  let curve (x : ℝ) := 2 * Real.sin x
  ∫ x in lower_bound..upper_bound, curve x = 3 := by
  sorry

end NUMINAMATH_CALUDE_area_under_sine_curve_l1755_175575


namespace NUMINAMATH_CALUDE_picture_area_l1755_175564

theorem picture_area (x y : ℕ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (2 * x + 5) * (y + 3) = 60) : x * y = 27 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l1755_175564


namespace NUMINAMATH_CALUDE_isotomic_lines_not_intersect_in_medial_triangle_l1755_175551

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A line in a 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The medial triangle of a given triangle --/
def medialTriangle (t : Triangle) : Triangle := sorry

/-- Checks if a point is inside or on the boundary of a triangle --/
def isInsideOrOnTriangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Checks if two lines are isotomic with respect to a triangle --/
def areIsotomicLines (l1 l2 : Line) (t : Triangle) : Prop := sorry

/-- The intersection point of two lines, if it exists --/
def lineIntersection (l1 l2 : Line) : Option (ℝ × ℝ) := sorry

theorem isotomic_lines_not_intersect_in_medial_triangle (t : Triangle) (l1 l2 : Line) :
  areIsotomicLines l1 l2 t →
  match lineIntersection l1 l2 with
  | some p => ¬isInsideOrOnTriangle p (medialTriangle t)
  | none => True
  := by sorry

end NUMINAMATH_CALUDE_isotomic_lines_not_intersect_in_medial_triangle_l1755_175551
