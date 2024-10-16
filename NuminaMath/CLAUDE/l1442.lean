import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_factorization_l1442_144243

theorem quadratic_factorization (b c d e f : ℤ) : 
  (∀ x : ℚ, 24 * x^2 + b * x + 24 = (c * x + d) * (e * x + f)) →
  c + d = 10 →
  c * e = 24 →
  d * f = 24 →
  b = 52 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1442_144243


namespace NUMINAMATH_CALUDE_salary_increase_20_percent_l1442_144216

-- Define Sharon's original weekly salary
variable (S : ℝ)

-- Define the condition that a 16% increase results in $406
axiom increase_16_percent : S * 1.16 = 406

-- Define the target salary of $420
def target_salary : ℝ := 420

-- Theorem to prove
theorem salary_increase_20_percent : 
  S * 1.20 = target_salary := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_20_percent_l1442_144216


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l1442_144202

theorem min_value_of_fraction (x a b : ℝ) (hx : 0 < x ∧ x < 1) (ha : a > 0) (hb : b > 0) :
  ∃ m : ℝ, m = (a + b)^2 ∧ ∀ y : ℝ, 0 < y ∧ y < 1 → 1 / (y^a * (1 - y)^b) ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l1442_144202


namespace NUMINAMATH_CALUDE_prob_two_consecutive_sum_four_l1442_144206

-- Define a 3-sided die
def Die := Fin 3

-- Define the sum of two dice
def diceSum (d1 d2 : Die) : ℕ := d1.val + d2.val + 2

-- Define the probability of getting a sum of 4 on a single roll
def probSumFour : ℚ := 1 / 3

-- Theorem statement
theorem prob_two_consecutive_sum_four :
  (probSumFour * probSumFour : ℚ) = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_prob_two_consecutive_sum_four_l1442_144206


namespace NUMINAMATH_CALUDE_simplify_expression_l1442_144263

theorem simplify_expression : Real.sqrt ((-2)^6) - (-1)^0 = 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1442_144263


namespace NUMINAMATH_CALUDE_new_person_weight_l1442_144272

theorem new_person_weight 
  (n : ℕ) 
  (initial_weight : ℝ) 
  (weight_increase : ℝ) 
  (replaced_weight : ℝ) :
  n = 8 →
  weight_increase = 2.5 →
  replaced_weight = 75 →
  initial_weight + (n : ℝ) * weight_increase = initial_weight - replaced_weight + 95 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1442_144272


namespace NUMINAMATH_CALUDE_remainder_3_180_mod_5_l1442_144252

theorem remainder_3_180_mod_5 : 3^180 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_180_mod_5_l1442_144252


namespace NUMINAMATH_CALUDE_probability_one_head_in_three_tosses_l1442_144268

theorem probability_one_head_in_three_tosses :
  let n : ℕ := 3  -- number of tosses
  let k : ℕ := 1  -- number of heads we want
  let p : ℚ := 1/2  -- probability of heads on a single toss
  Nat.choose n k * p^k * (1-p)^(n-k) = 3/8 := by
sorry

end NUMINAMATH_CALUDE_probability_one_head_in_three_tosses_l1442_144268


namespace NUMINAMATH_CALUDE_cloth_cost_price_l1442_144240

theorem cloth_cost_price (total_meters : ℕ) (first_part_meters : ℕ) (total_sale : ℕ) 
  (profit_first_part : ℕ) (profit_second_part : ℕ) :
  total_meters = 85 →
  first_part_meters = 50 →
  total_sale = 8925 →
  profit_first_part = 15 →
  profit_second_part = 20 →
  ∃ (cost_price : ℕ), 
    first_part_meters * (cost_price + profit_first_part) + 
    (total_meters - first_part_meters) * (cost_price + profit_second_part) = total_sale ∧
    cost_price = 88 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l1442_144240


namespace NUMINAMATH_CALUDE_nine_people_four_consecutive_l1442_144251

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def total_arrangements (n : ℕ) : ℕ := factorial n

def consecutive_arrangements (n : ℕ) (k : ℕ) : ℕ := 
  factorial (n - k + 1) * factorial k

def valid_arrangements (n : ℕ) (k : ℕ) : ℕ := 
  total_arrangements n - consecutive_arrangements n k

theorem nine_people_four_consecutive (n : ℕ) (k : ℕ) :
  n = 9 ∧ k = 4 → valid_arrangements n k = 345600 := by
  sorry

end NUMINAMATH_CALUDE_nine_people_four_consecutive_l1442_144251


namespace NUMINAMATH_CALUDE_triangle_sine_ratio_l1442_144239

/-- Given a triangle ABC where the ratio of sines of angles is 5:7:8, 
    prove the ratio of sides and the measure of angle B -/
theorem triangle_sine_ratio (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sine_ratio : ∃ k : ℝ, k > 0 ∧ Real.sin A = 5*k ∧ Real.sin B = 7*k ∧ Real.sin C = 8*k) :
  (∃ m : ℝ, m > 0 ∧ a = 5*m ∧ b = 7*m ∧ c = 8*m) ∧ B = π/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_ratio_l1442_144239


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1442_144214

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (1 + 1 / (Real.sqrt 5 + 2)) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1442_144214


namespace NUMINAMATH_CALUDE_present_age_of_A_l1442_144290

/-- Given two people A and B, their ages, and future age ratios, 
    prove that A's present age is 15 years. -/
theorem present_age_of_A (a b : ℕ) : 
  a * 3 = b * 5 →  -- Present age ratio
  (a + 6) * 5 = (b + 6) * 7 →  -- Future age ratio
  a = 15 := by
sorry

end NUMINAMATH_CALUDE_present_age_of_A_l1442_144290


namespace NUMINAMATH_CALUDE_sum_calculation_l1442_144257

def sequence_S : ℕ → ℕ
  | 0 => 0
  | (n + 1) => sequence_S n + (2 * n + 1)

def sequence_n : ℕ → ℕ
  | 0 => 1
  | (n + 1) => sequence_n n + 2

theorem sum_calculation :
  ∃ k : ℕ, sequence_n k > 50 ∧ sequence_n (k - 1) ≤ 50 ∧ sequence_S (k - 1) = 625 := by
  sorry

end NUMINAMATH_CALUDE_sum_calculation_l1442_144257


namespace NUMINAMATH_CALUDE_x_value_l1442_144273

theorem x_value (x y : ℚ) (h1 : x / y = 8 / 3) (h2 : y = 27) : x = 72 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1442_144273


namespace NUMINAMATH_CALUDE_prank_combinations_count_l1442_144224

/-- The number of choices for each day of the week-long prank --/
def prank_choices : List Nat := [1, 2, 3, 4, 2]

/-- The total number of combinations for the week-long prank --/
def total_combinations : Nat := prank_choices.prod

/-- Theorem stating that the total number of combinations is 48 --/
theorem prank_combinations_count :
  total_combinations = 48 := by sorry

end NUMINAMATH_CALUDE_prank_combinations_count_l1442_144224


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1442_144253

theorem perfect_square_trinomial (m n : ℝ) :
  (4 / 9) * m^2 + (4 / 3) * m * n + n^2 = ((2 / 3) * m + n)^2 := by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1442_144253


namespace NUMINAMATH_CALUDE_problem_statement_l1442_144271

/-- The problem statement --/
theorem problem_statement (x₀ y₀ r : ℝ) : 
  -- P(x₀, y₀) lies on both curves
  y₀ = 2 * Real.log x₀ ∧ 
  (x₀ - 3)^2 + y₀^2 = r^2 ∧ 
  -- Tangent lines are identical
  (2 / x₀ = -x₀ / y₀) ∧ 
  (2 / x₀ = x₀ * (y₀ - 2) / (9 - 3*x₀ - r^2)) ∧
  -- Quadratic function passes through (0,0), P(x₀, y₀), and (3,0)
  ∃ (a b c : ℝ), ∀ x, 
    (a*x^2 + b*x + c = 0) ∧
    (a*x₀^2 + b*x₀ + c = y₀) ∧
    (9*a + 3*b + c = 0) →
  -- The maximum value of the quadratic function is 9/8
  ∃ (f : ℝ → ℝ), (∀ x, f x ≤ 9/8) ∧ (∃ x, f x = 9/8) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1442_144271


namespace NUMINAMATH_CALUDE_pizza_distribution_l1442_144211

theorem pizza_distribution (total_pizzas : ℕ) (slices_per_pizza : ℕ) (num_students : ℕ)
  (leftover_cheese : ℕ) (leftover_onion : ℕ) (onion_per_student : ℕ)
  (h1 : total_pizzas = 6)
  (h2 : slices_per_pizza = 18)
  (h3 : num_students = 32)
  (h4 : leftover_cheese = 8)
  (h5 : leftover_onion = 4)
  (h6 : onion_per_student = 1) :
  (total_pizzas * slices_per_pizza - leftover_cheese - leftover_onion - num_students * onion_per_student) / num_students = 2 := by
  sorry

#check pizza_distribution

end NUMINAMATH_CALUDE_pizza_distribution_l1442_144211


namespace NUMINAMATH_CALUDE_agri_products_theorem_l1442_144200

/-- Represents the prices and quantities of agricultural products A and B --/
structure AgriProducts where
  price_A : ℝ
  price_B : ℝ
  quantity_A : ℝ
  quantity_B : ℝ

/-- Represents the problem constraints and conditions --/
def problem_constraints (p : AgriProducts) : Prop :=
  2 * p.price_A + 3 * p.price_B = 690 ∧
  p.price_A + 4 * p.price_B = 720 ∧
  p.quantity_A + p.quantity_B = 40 ∧
  p.price_A * p.quantity_A + p.price_B * p.quantity_B ≤ 5400 ∧
  p.quantity_A ≤ 3 * p.quantity_B

/-- Calculates the profit given the prices and quantities --/
def profit (p : AgriProducts) : ℝ :=
  (160 - p.price_A) * p.quantity_A + (200 - p.price_B) * p.quantity_B

/-- The main theorem to prove --/
theorem agri_products_theorem (p : AgriProducts) :
  problem_constraints p →
  p.price_A = 120 ∧ p.price_B = 150 ∧
  ∀ q : AgriProducts, problem_constraints q →
    profit q ≤ profit { price_A := 120, price_B := 150, quantity_A := 20, quantity_B := 20 } :=
by sorry

end NUMINAMATH_CALUDE_agri_products_theorem_l1442_144200


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l1442_144237

theorem root_sum_reciprocal (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ = 2 → 
  x₂^2 + x₂ = 2 → 
  x₁ ≠ x₂ → 
  1/x₁ + 1/x₂ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l1442_144237


namespace NUMINAMATH_CALUDE_a_four_plus_b_four_l1442_144260

theorem a_four_plus_b_four (a b : ℝ) (h1 : a^2 - b^2 = 10) (h2 : a * b = 8) : a^4 + b^4 = 548 := by
  sorry

end NUMINAMATH_CALUDE_a_four_plus_b_four_l1442_144260


namespace NUMINAMATH_CALUDE_smallest_class_size_l1442_144279

theorem smallest_class_size (total_students : ℕ) 
  (h1 : total_students ≥ 50)
  (h2 : ∃ (x : ℕ), total_students = 4 * x + (x + 2))
  (h3 : ∀ (y : ℕ), y ≥ 50 → (∃ (z : ℕ), y = 4 * z + (z + 2)) → y ≥ total_students) :
  total_students = 52 := by
sorry

end NUMINAMATH_CALUDE_smallest_class_size_l1442_144279


namespace NUMINAMATH_CALUDE_comparison_inequality_range_of_linear_combination_l1442_144274

-- Part 1
theorem comparison_inequality (x y z : ℝ) : 
  5 * x^2 + y^2 + z^2 ≥ 2 * x * y + 4 * x + 2 * z - 2 := by sorry

-- Part 2
theorem range_of_linear_combination (a b : ℝ) 
  (h1 : 1 ≤ 2 * a + b) (h2 : 2 * a + b ≤ 4) 
  (h3 : -1 ≤ a - 2 * b) (h4 : a - 2 * b ≤ 2) : 
  -1 ≤ 10 * a - 5 * b ∧ 10 * a - 5 * b ≤ 20 := by sorry

end NUMINAMATH_CALUDE_comparison_inequality_range_of_linear_combination_l1442_144274


namespace NUMINAMATH_CALUDE_percentage_red_cars_chennai_l1442_144281

/-- Percentage of red cars in the total car population -/
def percentage_red_cars (total_cars : ℕ) (honda_cars : ℕ) (honda_red_ratio : ℚ) (non_honda_red_ratio : ℚ) : ℚ :=
  let non_honda_cars := total_cars - honda_cars
  let red_honda_cars := honda_red_ratio * honda_cars
  let red_non_honda_cars := non_honda_red_ratio * non_honda_cars
  let total_red_cars := red_honda_cars + red_non_honda_cars
  (total_red_cars / total_cars) * 100

/-- The percentage of red cars in Chennai -/
theorem percentage_red_cars_chennai :
  percentage_red_cars 900 500 (90/100) (225/1000) = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_red_cars_chennai_l1442_144281


namespace NUMINAMATH_CALUDE_winter_sales_l1442_144245

/-- Proves that the number of pastries sold in winter is 3 million -/
theorem winter_sales (spring summer fall : ℕ) (total : ℝ) : 
  spring = 3 → 
  summer = 6 → 
  fall = 3 → 
  fall = (1/5 : ℝ) * total → 
  total - (spring + summer + fall : ℝ) = 3 := by
sorry

end NUMINAMATH_CALUDE_winter_sales_l1442_144245


namespace NUMINAMATH_CALUDE_f_max_value_l1442_144276

/-- The function f(x) = 6x - 2x^2 -/
def f (x : ℝ) := 6 * x - 2 * x^2

/-- The maximum value of f(x) is 9/2 -/
theorem f_max_value : ∃ (M : ℝ), M = 9/2 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l1442_144276


namespace NUMINAMATH_CALUDE_specific_combination_probability_is_one_eighth_l1442_144246

/-- A regular tetrahedron with numbers on its faces -/
structure NumberedTetrahedron :=
  (faces : Fin 4 → Fin 4)

/-- The probability of a specific face showing on a regular tetrahedron -/
def face_probability : ℚ := 1 / 4

/-- The number of ways to choose which tetrahedron shows a specific number -/
def ways_to_choose : ℕ := 2

/-- The probability of getting a specific combination of numbers when throwing two tetrahedra -/
def specific_combination_probability (t1 t2 : NumberedTetrahedron) : ℚ :=
  ↑ways_to_choose * face_probability * face_probability

theorem specific_combination_probability_is_one_eighth (t1 t2 : NumberedTetrahedron) :
  specific_combination_probability t1 t2 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_specific_combination_probability_is_one_eighth_l1442_144246


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_solution_exists_l1442_144241

theorem smallest_solution_quadratic (y : ℝ) : 
  (12 * y^2 - 56 * y + 48 = 0) → y ≥ 2 := by
  sorry

theorem solution_exists : 
  ∃ y : ℝ, 12 * y^2 - 56 * y + 48 = 0 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_solution_exists_l1442_144241


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l1442_144221

/-- Two concentric circles with center Q -/
structure ConcentricCircles where
  center : Point
  radius₁ : ℝ
  radius₂ : ℝ
  h : radius₁ < radius₂

/-- The length of an arc given its central angle and the circle's radius -/
def arcLength (angle : ℝ) (radius : ℝ) : ℝ := angle * radius

theorem concentric_circles_area_ratio 
  (circles : ConcentricCircles) 
  (h : arcLength (π/3) circles.radius₁ = arcLength (π/6) circles.radius₂) : 
  (circles.radius₁^2) / (circles.radius₂^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l1442_144221


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_of_roots_l1442_144203

/-- A quadratic function f(x) = 2x² + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := 2 * x^2 + b * x + c

theorem min_reciprocal_sum_of_roots (b c : ℝ) :
  (f b c (-10) = f b c 12) →
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ f b c x₁ = 0 ∧ f b c x₂ = 0) →
  (∃ m : ℝ, m = 2 ∧ ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → f b c x₁ = 0 → f b c x₂ = 0 → 1/x₁ + 1/x₂ ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_of_roots_l1442_144203


namespace NUMINAMATH_CALUDE_mechanic_parts_cost_l1442_144288

/-- A problem about calculating the cost of parts in a mechanic's bill -/
theorem mechanic_parts_cost
  (hourly_rate : ℝ)
  (job_duration : ℝ)
  (total_bill : ℝ)
  (h1 : hourly_rate = 45)
  (h2 : job_duration = 5)
  (h3 : total_bill = 450) :
  total_bill - hourly_rate * job_duration = 225 := by
  sorry

end NUMINAMATH_CALUDE_mechanic_parts_cost_l1442_144288


namespace NUMINAMATH_CALUDE_find_A_l1442_144291

theorem find_A : ∃ A : ℕ, A % 5 = 4 ∧ A / 5 = 6 ∧ A = 34 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l1442_144291


namespace NUMINAMATH_CALUDE_number_division_problem_l1442_144201

theorem number_division_problem (x : ℝ) : (x - 5) / 7 = 7 → (x - 2) / 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1442_144201


namespace NUMINAMATH_CALUDE_triangle_area_problem_l1442_144225

theorem triangle_area_problem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 96 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l1442_144225


namespace NUMINAMATH_CALUDE_mean_of_added_numbers_l1442_144292

theorem mean_of_added_numbers (original_mean original_count new_mean new_count : ℝ) 
  (h1 : original_mean = 65)
  (h2 : original_count = 7)
  (h3 : new_mean = 80)
  (h4 : new_count = 10) :
  let added_count := new_count - original_count
  let added_sum := new_mean * new_count - original_mean * original_count
  added_sum / added_count = 115 := by
sorry

end NUMINAMATH_CALUDE_mean_of_added_numbers_l1442_144292


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1442_144270

theorem pure_imaginary_condition (m : ℝ) : 
  (((2 : ℂ) - m * Complex.I) / (1 + Complex.I)).re = 0 → m = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1442_144270


namespace NUMINAMATH_CALUDE_power_division_l1442_144284

theorem power_division (h : 64 = 8^2) : 8^15 / 64^7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l1442_144284


namespace NUMINAMATH_CALUDE_no_ab_term_when_m_is_neg_six_l1442_144228

-- Define the polynomial as a function of a, b, and m
def polynomial (a b m : ℝ) : ℝ := 3 * (a^2 - 2*a*b - b^2) - (a^2 + m*a*b + 2*b^2)

-- Theorem stating that the polynomial has no ab term when m = -6
theorem no_ab_term_when_m_is_neg_six :
  ∀ a b : ℝ, (∀ m : ℝ, polynomial a b m = 2*a^2 - (6+m)*a*b - 5*b^2) →
  (∃! m : ℝ, ∀ a b : ℝ, polynomial a b m = 2*a^2 - 5*b^2) →
  (∃ m : ℝ, m = -6 ∧ ∀ a b : ℝ, polynomial a b m = 2*a^2 - 5*b^2) :=
by sorry

end NUMINAMATH_CALUDE_no_ab_term_when_m_is_neg_six_l1442_144228


namespace NUMINAMATH_CALUDE_scientific_notation_of_2590000_l1442_144215

theorem scientific_notation_of_2590000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 2590000 = a * (10 : ℝ) ^ n ∧ a = 2.59 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2590000_l1442_144215


namespace NUMINAMATH_CALUDE_greatest_b_for_no_minus_six_l1442_144218

theorem greatest_b_for_no_minus_six (b : ℤ) : 
  (∀ x : ℝ, x^2 + b*x + 20 ≠ -6) ↔ b ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_greatest_b_for_no_minus_six_l1442_144218


namespace NUMINAMATH_CALUDE_avery_donation_total_l1442_144227

/-- Proves that the total number of clothes Avery donates is 16 -/
theorem avery_donation_total (shirts : ℕ) (pants : ℕ) (shorts : ℕ) : 
  shirts = 4 →
  pants = 2 * shirts →
  shorts = pants / 2 →
  shirts + pants + shorts = 16 := by
  sorry

end NUMINAMATH_CALUDE_avery_donation_total_l1442_144227


namespace NUMINAMATH_CALUDE_probability_black_or_white_ball_l1442_144283

theorem probability_black_or_white_ball 
  (p_red : ℝ) 
  (p_white : ℝ) 
  (h1 : p_red = 0.45) 
  (h2 : p_white = 0.25) 
  (h3 : 0 ≤ p_red ∧ p_red ≤ 1) 
  (h4 : 0 ≤ p_white ∧ p_white ≤ 1) : 
  p_red + p_white + (1 - p_red - p_white) = 1 ∧ 1 - p_red = 0.55 := by
sorry

end NUMINAMATH_CALUDE_probability_black_or_white_ball_l1442_144283


namespace NUMINAMATH_CALUDE_gcd_78_182_l1442_144282

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := by
  sorry

end NUMINAMATH_CALUDE_gcd_78_182_l1442_144282


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1442_144293

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_abc_properties (t : Triangle) 
  (h_acute : 0 < t.C ∧ t.C < Real.pi / 2)
  (h_sine_relation : Real.sqrt 15 * t.a * Real.sin t.A = t.b * Real.sin t.B * Real.sin t.C)
  (h_b_twice_a : t.b = 2 * t.a)
  (h_a_c_sum : t.a + t.c = 6) :
  Real.tan t.C = Real.sqrt 15 ∧ 
  (1 / 2) * t.a * t.b * Real.sin t.C = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1442_144293


namespace NUMINAMATH_CALUDE_cubic_equation_roots_difference_l1442_144287

theorem cubic_equation_roots_difference (x : ℝ) : 
  (64 * x^3 - 144 * x^2 + 92 * x - 15 = 0) →
  (∃ a d : ℝ, {a - d, a, a + d} ⊆ {x | 64 * x^3 - 144 * x^2 + 92 * x - 15 = 0}) →
  (∃ r₁ r₂ r₃ : ℝ, 
    r₁ < r₂ ∧ r₂ < r₃ ∧
    {r₁, r₂, r₃} = {x | 64 * x^3 - 144 * x^2 + 92 * x - 15 = 0} ∧
    r₃ - r₁ = 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_difference_l1442_144287


namespace NUMINAMATH_CALUDE_total_investment_amount_l1442_144213

/-- Prove that the total investment is $8000 given the specified conditions --/
theorem total_investment_amount (total_income : ℝ) (rate1 rate2 : ℝ) (investment1 : ℝ) :
  total_income = 575 →
  rate1 = 0.085 →
  rate2 = 0.064 →
  investment1 = 3000 →
  ∃ (investment2 : ℝ),
    total_income = investment1 * rate1 + investment2 * rate2 ∧
    investment1 + investment2 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_total_investment_amount_l1442_144213


namespace NUMINAMATH_CALUDE_center_is_eight_l1442_144294

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Check if two positions are adjacent --/
def adjacent (p q : Fin 3 × Fin 3) : Prop :=
  (p.1 = q.1 ∧ (p.2 = q.2 + 1 ∨ p.2 + 1 = q.2)) ∨
  (p.2 = q.2 ∧ (p.1 = q.1 + 1 ∨ p.1 + 1 = q.1))

/-- Check if the grid satisfies the consecutive adjacency property --/
def consecutive_adjacent (g : Grid) : Prop :=
  ∀ n : Fin 8, ∃ p q : Fin 3 × Fin 3, 
    g p.1 p.2 = n ∧ g q.1 q.2 = n + 1 ∧ adjacent p q

/-- Sum of corner numbers in the grid --/
def corner_sum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

/-- Sum of numbers in the middle column --/
def middle_column_sum (g : Grid) : Nat :=
  g 0 1 + g 1 1 + g 2 1

theorem center_is_eight (g : Grid) 
  (h1 : ∀ n : Fin 9, ∃! p : Fin 3 × Fin 3, g p.1 p.2 = n)
  (h2 : consecutive_adjacent g)
  (h3 : corner_sum g = 20)
  (h4 : Even (middle_column_sum g)) :
  g 1 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_center_is_eight_l1442_144294


namespace NUMINAMATH_CALUDE_sum_K_floor_quotient_100_l1442_144258

/-- K(x) is the number of irreducible fractions a/b where 1 ≤ a < x and 1 ≤ b < x -/
def K (x : ℕ) : ℕ :=
  (Finset.range (x - 1)).sum (λ k => Nat.totient k)

/-- The sum of K(⌊100/k⌋) for k from 1 to 100 equals 9801 -/
theorem sum_K_floor_quotient_100 :
  (Finset.range 100).sum (λ k => K (100 / (k + 1))) = 9801 := by
  sorry

end NUMINAMATH_CALUDE_sum_K_floor_quotient_100_l1442_144258


namespace NUMINAMATH_CALUDE_sin_2010_degrees_l1442_144238

theorem sin_2010_degrees : Real.sin (2010 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2010_degrees_l1442_144238


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_196_l1442_144285

theorem factor_x_squared_minus_196 (x : ℝ) : x^2 - 196 = (x - 14) * (x + 14) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_196_l1442_144285


namespace NUMINAMATH_CALUDE_solve_class_problem_l1442_144280

def class_problem (num_girls : ℕ) (total_books : ℕ) (girls_books : ℕ) : Prop :=
  ∃ (num_boys : ℕ),
    num_boys = 10 ∧
    num_girls = 15 ∧
    total_books = 375 ∧
    girls_books = 225 ∧
    ∃ (books_per_student : ℕ),
      books_per_student * (num_girls + num_boys) = total_books ∧
      books_per_student * num_girls = girls_books

theorem solve_class_problem :
  class_problem 15 375 225 := by
  sorry

end NUMINAMATH_CALUDE_solve_class_problem_l1442_144280


namespace NUMINAMATH_CALUDE_product_of_one_fourth_and_one_half_l1442_144298

theorem product_of_one_fourth_and_one_half : (1 / 4 : ℚ) * (1 / 2 : ℚ) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_one_fourth_and_one_half_l1442_144298


namespace NUMINAMATH_CALUDE_negative_sqrt_of_square_of_negative_three_l1442_144275

theorem negative_sqrt_of_square_of_negative_three :
  -Real.sqrt ((-3)^2) = -3 := by sorry

end NUMINAMATH_CALUDE_negative_sqrt_of_square_of_negative_three_l1442_144275


namespace NUMINAMATH_CALUDE_downstream_speed_calculation_l1442_144234

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ
  downstream : ℝ

/-- 
Given a man's upstream speed and still water speed, 
calculates and proves his downstream speed
-/
theorem downstream_speed_calculation (speed : RowingSpeed) 
  (h1 : speed.upstream = 30)
  (h2 : speed.stillWater = 45) :
  speed.downstream = 60 := by
  sorry

end NUMINAMATH_CALUDE_downstream_speed_calculation_l1442_144234


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_m_leq_neg_one_l1442_144299

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x^2 - 4*m*x + 2*m + 6 = 0}
def B : Set ℝ := {x | x < 0}

-- State the theorem
theorem intersection_nonempty_implies_m_leq_neg_one (m : ℝ) :
  (A m ∩ B).Nonempty → m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_m_leq_neg_one_l1442_144299


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l1442_144223

theorem sum_of_roots_quadratic_equation :
  let a : ℝ := -3
  let b : ℝ := -27
  let c : ℝ := 81
  let equation := fun x : ℝ => a * x^2 + b * x + c
  ∃ r s : ℝ, equation r = 0 ∧ equation s = 0 ∧ r + s = -b / a :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l1442_144223


namespace NUMINAMATH_CALUDE_employee_discount_percentage_l1442_144242

theorem employee_discount_percentage
  (wholesale_cost : ℝ)
  (markup_percentage : ℝ)
  (employee_paid_price : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : markup_percentage = 20)
  (h3 : employee_paid_price = 168) :
  let retail_price := wholesale_cost * (1 + markup_percentage / 100)
  let discount_amount := retail_price - employee_paid_price
  let discount_percentage := (discount_amount / retail_price) * 100
  discount_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_employee_discount_percentage_l1442_144242


namespace NUMINAMATH_CALUDE_cloth_selling_price_l1442_144255

/-- Calculates the total selling price of cloth given the quantity, profit per meter, and cost price per meter -/
def totalSellingPrice (quantity : ℕ) (profitPerMeter : ℕ) (costPricePerMeter : ℕ) : ℕ :=
  quantity * (costPricePerMeter + profitPerMeter)

/-- Proves that the total selling price for 85 meters of cloth with a profit of Rs. 25 per meter 
    and a cost price of Rs. 80 per meter is Rs. 8925 -/
theorem cloth_selling_price :
  totalSellingPrice 85 25 80 = 8925 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l1442_144255


namespace NUMINAMATH_CALUDE_coin_identification_possible_l1442_144229

/-- Represents the expert's response, which is always an overestimate -/
structure ExpertResponse :=
  (reported : ℕ)
  (actual : ℕ)
  (overestimate : ℕ)
  (h : reported = actual + overestimate)

/-- Represents the coin identification process -/
def can_identify_counterfeit (total_coins : ℕ) (max_presentation : ℕ) : Prop :=
  ∀ (counterfeit : Finset ℕ) (overestimate : ℕ),
    counterfeit.card ≤ total_coins →
    (∀ subset : Finset ℕ, subset.card ≤ max_presentation →
      ∃ response : ExpertResponse,
        response.actual = (subset ∩ counterfeit).card ∧
        response.overestimate = overestimate) →
    ∃ process : ℕ → Bool,
      ∀ coin, coin < total_coins → (process coin ↔ coin ∈ counterfeit)

theorem coin_identification_possible :
  can_identify_counterfeit 100 20 :=
sorry

end NUMINAMATH_CALUDE_coin_identification_possible_l1442_144229


namespace NUMINAMATH_CALUDE_number_of_black_marbles_l1442_144217

/-- Given a bag of marbles with white and black marbles, prove the number of black marbles. -/
theorem number_of_black_marbles
  (total_marbles : ℕ)
  (white_marbles : ℕ)
  (h1 : total_marbles = 37)
  (h2 : white_marbles = 19) :
  total_marbles - white_marbles = 18 :=
by sorry

end NUMINAMATH_CALUDE_number_of_black_marbles_l1442_144217


namespace NUMINAMATH_CALUDE_time_until_sunset_l1442_144262

-- Define the initial sunset time in minutes past midnight
def initial_sunset : ℕ := 18 * 60

-- Define the daily sunset delay in minutes
def daily_delay : ℚ := 1.2

-- Define the number of days since March 1st
def days_passed : ℕ := 40

-- Define the current time in minutes past midnight
def current_time : ℕ := 18 * 60 + 10

-- Theorem statement
theorem time_until_sunset :
  let total_delay : ℚ := daily_delay * days_passed
  let new_sunset : ℚ := initial_sunset + total_delay
  ⌊new_sunset⌋ - current_time = 38 := by sorry

end NUMINAMATH_CALUDE_time_until_sunset_l1442_144262


namespace NUMINAMATH_CALUDE_largest_eight_digit_with_even_digits_l1442_144266

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop :=
  n ≥ 10000000 ∧ n < 100000000

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k, n / (10^k) % 10 = d

theorem largest_eight_digit_with_even_digits :
  ∀ n : Nat, is_eight_digit n → contains_all_even_digits n →
  n ≤ 99986420 :=
sorry

end NUMINAMATH_CALUDE_largest_eight_digit_with_even_digits_l1442_144266


namespace NUMINAMATH_CALUDE_image_difference_l1442_144296

/-- Define the mapping f -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.1 + p.2)

/-- Theorem statement -/
theorem image_difference (m n : ℝ) (h : (m, n) = f (2, 1)) :
  m - n = -1 := by
  sorry

end NUMINAMATH_CALUDE_image_difference_l1442_144296


namespace NUMINAMATH_CALUDE_tv_screen_height_l1442_144226

/-- The height of a rectangular TV screen given its area and width -/
theorem tv_screen_height (area width : ℝ) (h_area : area = 21) (h_width : width = 3) :
  area / width = 7 := by
  sorry

end NUMINAMATH_CALUDE_tv_screen_height_l1442_144226


namespace NUMINAMATH_CALUDE_find_S_l1442_144207

theorem find_S : ∃ S : ℝ, (1/3 * 1/8 * S = 1/4 * 1/6 * 120) ∧ (S = 120) := by
  sorry

end NUMINAMATH_CALUDE_find_S_l1442_144207


namespace NUMINAMATH_CALUDE_town_businesses_town_businesses_proof_l1442_144259

theorem town_businesses : ℕ → Prop :=
  fun total_businesses =>
    let fired := total_businesses / 2
    let quit := total_businesses / 3
    let can_apply := 12
    fired + quit + can_apply = total_businesses ∧ total_businesses = 72

-- Proof
theorem town_businesses_proof : ∃ n : ℕ, town_businesses n := by
  sorry

end NUMINAMATH_CALUDE_town_businesses_town_businesses_proof_l1442_144259


namespace NUMINAMATH_CALUDE_product_and_sum_of_squares_l1442_144219

theorem product_and_sum_of_squares (x y : ℝ) : 
  x * y = 120 → x^2 + y^2 = 289 → x + y = 22 := by
sorry

end NUMINAMATH_CALUDE_product_and_sum_of_squares_l1442_144219


namespace NUMINAMATH_CALUDE_triangle_angle_from_sides_l1442_144208

theorem triangle_angle_from_sides : 
  ∀ (a b c : ℝ), 
    a = 1 → 
    b = Real.sqrt 7 → 
    c = Real.sqrt 3 → 
    ∃ (A B C : ℝ), 
      A + B + C = π ∧ 
      0 < A ∧ A < π ∧ 
      0 < B ∧ B < π ∧ 
      0 < C ∧ C < π ∧ 
      b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧ 
      B = 5*π/6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_from_sides_l1442_144208


namespace NUMINAMATH_CALUDE_divisibility_problem_l1442_144264

theorem divisibility_problem (a b : ℕ) 
  (h1 : b ∣ (5 * a - 1))
  (h2 : b ∣ (a - 10))
  (h3 : ¬(b ∣ (3 * a + 5))) :
  b = 49 := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1442_144264


namespace NUMINAMATH_CALUDE_triangle_similarity_theorem_l1442_144233

-- Define the properties of the first triangle
def first_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ a = 15 ∧ c = 34

-- Define the similarity ratio between the two triangles
def similarity_ratio (r : ℝ) : Prop :=
  r = 102 / 34

-- Define the shortest side of the second triangle
def shortest_side (x : ℝ) : Prop :=
  x = 3 * Real.sqrt 931

-- Theorem statement
theorem triangle_similarity_theorem :
  ∀ a b c r x : ℝ,
  first_triangle a b c →
  similarity_ratio r →
  shortest_side x →
  x = r * a :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_theorem_l1442_144233


namespace NUMINAMATH_CALUDE_candy_packing_problem_l1442_144204

theorem candy_packing_problem :
  ∃! n : ℕ, 11 ≤ n ∧ n ≤ 100 ∧ 
    6 ∣ n ∧ 9 ∣ n ∧ n % 7 = 1 ∧
    n = 36 := by sorry

end NUMINAMATH_CALUDE_candy_packing_problem_l1442_144204


namespace NUMINAMATH_CALUDE_four_pencils_per_child_l1442_144254

/-- Given a group of children and pencils, calculate the number of pencils per child. -/
def pencils_per_child (num_children : ℕ) (total_pencils : ℕ) : ℕ :=
  total_pencils / num_children

/-- Theorem stating that with 8 children and 32 pencils, each child has 4 pencils. -/
theorem four_pencils_per_child :
  pencils_per_child 8 32 = 4 := by
  sorry

#eval pencils_per_child 8 32

end NUMINAMATH_CALUDE_four_pencils_per_child_l1442_144254


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_1_simplify_and_evaluate_2_l1442_144286

-- Problem 1
theorem simplify_and_evaluate_1 : 2 * Real.sqrt 3 * 31.5 * 612 = 6 := by sorry

-- Problem 2
theorem simplify_and_evaluate_2 : 
  (Real.log 3 / Real.log 4 - Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 1/4 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_1_simplify_and_evaluate_2_l1442_144286


namespace NUMINAMATH_CALUDE_childrens_vehicle_wheels_l1442_144231

theorem childrens_vehicle_wheels 
  (adult_count : ℕ) 
  (child_count : ℕ) 
  (total_wheels : ℕ) 
  (bicycle_wheels : ℕ) :
  adult_count = 6 →
  child_count = 15 →
  total_wheels = 57 →
  bicycle_wheels = 2 →
  ∃ (child_vehicle_wheels : ℕ), 
    child_vehicle_wheels = 3 ∧
    total_wheels = adult_count * bicycle_wheels + child_count * child_vehicle_wheels :=
by sorry

end NUMINAMATH_CALUDE_childrens_vehicle_wheels_l1442_144231


namespace NUMINAMATH_CALUDE_jerry_skit_first_character_lines_l1442_144235

/-- Represents the number of lines for each character in Jerry's skit script -/
structure SkitScript where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of Jerry's skit script -/
def validScript (s : SkitScript) : Prop :=
  s.first = s.second + 8 ∧
  s.third = 2 ∧
  s.second = 3 * s.third + 6

theorem jerry_skit_first_character_lines :
  ∀ s : SkitScript, validScript s → s.first = 20 := by
  sorry

#check jerry_skit_first_character_lines

end NUMINAMATH_CALUDE_jerry_skit_first_character_lines_l1442_144235


namespace NUMINAMATH_CALUDE_art_show_earnings_l1442_144232

def extra_large_price : ℕ := 150
def large_price : ℕ := 100
def medium_price : ℕ := 80
def small_price : ℕ := 60

def extra_large_sold : ℕ := 3
def large_sold : ℕ := 5
def medium_sold : ℕ := 8
def small_sold : ℕ := 10

def large_discount : ℚ := 0.1
def sales_tax : ℚ := 0.05

def total_earnings : ℚ := 2247

theorem art_show_earnings :
  let extra_large_total := extra_large_price * extra_large_sold
  let large_total := large_price * large_sold * (1 - large_discount)
  let medium_total := medium_price * medium_sold
  let small_total := small_price * small_sold
  let subtotal := extra_large_total + large_total + medium_total + small_total
  let tax := subtotal * sales_tax
  (subtotal + tax : ℚ) = total_earnings := by
sorry

end NUMINAMATH_CALUDE_art_show_earnings_l1442_144232


namespace NUMINAMATH_CALUDE_retail_price_calculation_l1442_144297

def calculate_retail_price (wholesale : ℝ) (profit_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  let intended_price := wholesale * (1 + profit_percent)
  intended_price * (1 - discount_percent)

def overall_retail_price (w1 w2 w3 : ℝ) (p1 p2 p3 : ℝ) (d1 d2 d3 : ℝ) : ℝ :=
  calculate_retail_price w1 p1 d1 +
  calculate_retail_price w2 p2 d2 +
  calculate_retail_price w3 p3 d3

theorem retail_price_calculation :
  overall_retail_price 90 120 200 0.20 0.30 0.25 0.10 0.15 0.05 = 467.30 := by
  sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l1442_144297


namespace NUMINAMATH_CALUDE_water_balloon_count_l1442_144222

-- Define the number of water balloons for each person
def sarah_balloons : ℕ := 5
def janice_balloons : ℕ := 6

-- Define the relationships between the number of water balloons
theorem water_balloon_count :
  ∀ (tim_balloons randy_balloons cynthia_balloons : ℕ),
  (tim_balloons = 2 * sarah_balloons) →
  (tim_balloons + 3 = janice_balloons) →
  (2 * randy_balloons = janice_balloons) →
  (cynthia_balloons = 4 * randy_balloons) →
  cynthia_balloons = 12 := by
  sorry

end NUMINAMATH_CALUDE_water_balloon_count_l1442_144222


namespace NUMINAMATH_CALUDE_discriminant_positive_roots_when_k_zero_l1442_144220

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 2*k*x - 1

-- Define the discriminant of the quadratic equation f(x) = 0
def discriminant (k : ℝ) : ℝ := (2*k)^2 - 4*1*(-1)

-- Theorem 1: The discriminant is always positive
theorem discriminant_positive (k : ℝ) : discriminant k > 0 := by
  sorry

-- Theorem 2: When k = 0, the roots are 1 and -1
theorem roots_when_k_zero :
  ∃ (x1 x2 : ℝ), x1 = 1 ∧ x2 = -1 ∧ f 0 x1 = 0 ∧ f 0 x2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_positive_roots_when_k_zero_l1442_144220


namespace NUMINAMATH_CALUDE_equation_solution_l1442_144289

theorem equation_solution (x y k z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0)
  (h : 1/x + 1/y = k/z) : z = x*y / (k*(y+x)) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l1442_144289


namespace NUMINAMATH_CALUDE_no_valid_labeling_l1442_144248

/-- Represents a labeling of the hexagon vertices and center -/
def Labeling := Fin 7 → Fin 7

/-- The sum of labels on a line through the center -/
def lineSum (l : Labeling) (i j : Fin 7) : ℕ :=
  l i + l 6 + l j  -- Assuming index 6 represents the center J

/-- Checks if a labeling is valid according to the problem conditions -/
def isValidLabeling (l : Labeling) : Prop :=
  (Function.Injective l) ∧ 
  (lineSum l 0 4 = lineSum l 1 5) ∧
  (lineSum l 1 5 = lineSum l 2 3) ∧
  (lineSum l 2 3 = lineSum l 0 4)

/-- The main theorem: there are no valid labelings -/
theorem no_valid_labeling : 
  ¬ ∃ l : Labeling, isValidLabeling l :=
sorry

end NUMINAMATH_CALUDE_no_valid_labeling_l1442_144248


namespace NUMINAMATH_CALUDE_inequality_proof_l1442_144236

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2)/(x^2 + y*z) + (y^2 + 2*z^2 + 2*x^2)/(y^2 + z*x) + (z^2 + 2*x^2 + 2*y^2)/(z^2 + x*y) > 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1442_144236


namespace NUMINAMATH_CALUDE_min_vertical_distance_l1442_144278

/-- The vertical distance between |x| and -x^2-4x-3 -/
def verticalDistance (x : ℝ) : ℝ := |x| - (-x^2 - 4*x - 3)

/-- The minimum vertical distance between |x| and -x^2-4x-3 is 3/4 -/
theorem min_vertical_distance :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), verticalDistance x₀ ≤ verticalDistance x ∧ verticalDistance x₀ = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_min_vertical_distance_l1442_144278


namespace NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l1442_144244

/-- A rectangular prism with given dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of pairs of parallel edges in a rectangular prism -/
def parallel_edge_pairs (prism : RectangularPrism) : ℕ :=
  3 -- Each dimension contributes 2 pairs, so 2 + 2 + 2 = 6

/-- Theorem stating that a rectangular prism with dimensions 8, 4, and 2 has 6 pairs of parallel edges -/
theorem rectangular_prism_parallel_edges :
  let prism : RectangularPrism := { length := 8, width := 4, height := 2 }
  parallel_edge_pairs prism = 6 := by
  sorry


end NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l1442_144244


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l1442_144230

theorem triangle_angle_relation (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 4)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (α β : ℝ), 0 < α ∧ 0 < β ∧ α < π ∧ β < π ∧ 2 * α + 3 * β = π := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l1442_144230


namespace NUMINAMATH_CALUDE_increasing_linear_function_l1442_144265

def linearFunction (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem increasing_linear_function (k b : ℝ) (h : k > 0) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → linearFunction k b x₁ < linearFunction k b x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_increasing_linear_function_l1442_144265


namespace NUMINAMATH_CALUDE_simplify_logarithmic_expression_l1442_144269

theorem simplify_logarithmic_expression (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  Real.log (Real.cos x * Real.tan x + 1 - 2 * Real.sin (x / 2) ^ 2) +
  Real.log (Real.sqrt 2 * Real.cos (x - Real.pi / 4)) -
  Real.log (1 + Real.sin (2 * x)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_logarithmic_expression_l1442_144269


namespace NUMINAMATH_CALUDE_gcd_154_90_l1442_144256

theorem gcd_154_90 : Nat.gcd 154 90 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_154_90_l1442_144256


namespace NUMINAMATH_CALUDE_problem_solution_l1442_144247

/-- Calculates the number of songs per album given the initial number of albums,
    the number of albums removed, and the total number of songs bought. -/
def songs_per_album (initial_albums : ℕ) (removed_albums : ℕ) (total_songs : ℕ) : ℕ :=
  total_songs / (initial_albums - removed_albums)

/-- Proves that given the specific conditions in the problem,
    the number of songs per album is 7. -/
theorem problem_solution :
  songs_per_album 8 2 42 = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1442_144247


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l1442_144250

theorem solution_set_abs_inequality :
  {x : ℝ | |2 - x| < 5} = {x : ℝ | -3 < x ∧ x < 7} := by sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l1442_144250


namespace NUMINAMATH_CALUDE_wario_field_goals_l1442_144295

/-- Given the conditions of Wario's field goal attempts, prove the number of wide right misses. -/
theorem wario_field_goals (total_attempts : ℕ) (miss_ratio : ℚ) (wide_right_ratio : ℚ) 
  (h1 : total_attempts = 60)
  (h2 : miss_ratio = 1 / 4)
  (h3 : wide_right_ratio = 1 / 5) : 
  ⌊(total_attempts : ℚ) * miss_ratio * wide_right_ratio⌋ = 3 := by
  sorry

#check wario_field_goals

end NUMINAMATH_CALUDE_wario_field_goals_l1442_144295


namespace NUMINAMATH_CALUDE_revenue_difference_l1442_144249

/-- Mr. Banks' number of investments -/
def banks_investments : ℕ := 8

/-- Revenue from each of Mr. Banks' investments -/
def banks_revenue_per_investment : ℕ := 500

/-- Ms. Elizabeth's number of investments -/
def elizabeth_investments : ℕ := 5

/-- Revenue from each of Ms. Elizabeth's investments -/
def elizabeth_revenue_per_investment : ℕ := 900

/-- The difference in total revenue between Ms. Elizabeth and Mr. Banks -/
theorem revenue_difference : 
  elizabeth_investments * elizabeth_revenue_per_investment - 
  banks_investments * banks_revenue_per_investment = 500 := by
  sorry

end NUMINAMATH_CALUDE_revenue_difference_l1442_144249


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l1442_144205

theorem quadratic_roots_transformation (K : ℝ) (α β : ℝ) : 
  (3 * α^2 + 7 * α + K = 0) →
  (3 * β^2 + 7 * β + K = 0) →
  (∃ m : ℝ, (α^2 - α)^2 + p * (α^2 - α) + m = 0 ∧ (β^2 - β)^2 + p * (β^2 - β) + m = 0) →
  p = -70/9 + 2*K/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l1442_144205


namespace NUMINAMATH_CALUDE_kendalls_quarters_l1442_144209

/-- Represents the number of coins of each type -/
structure CoinCounts where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ

/-- Calculates the total value of coins in dollars -/
def totalValue (c : CoinCounts) : ℚ :=
  c.quarters * (1/4) + c.dimes * (1/10) + c.nickels * (1/20)

theorem kendalls_quarters :
  ∃ (c : CoinCounts), c.dimes = 12 ∧ c.nickels = 6 ∧ totalValue c = 4 ∧ c.quarters = 10 := by
  sorry

end NUMINAMATH_CALUDE_kendalls_quarters_l1442_144209


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1442_144277

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 - x - 2 = 0}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1442_144277


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1442_144210

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the line
def line (x y : ℝ) : Prop :=
  y = 2*x - 4

-- Define the right focus
def right_focus (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 = a^2 + b^2 ∧ x > 0 ∧ y = 0

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ (x y : ℝ), right_focus a b x y ∧ line x y) →
  (∃! (p : ℝ × ℝ), hyperbola a b p.1 p.2 ∧ line p.1 p.2) →
  (∀ (x y : ℝ), hyperbola a b x y ↔ 5*x^2/4 - 5*y^2/16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1442_144210


namespace NUMINAMATH_CALUDE_exams_fourth_year_l1442_144261

theorem exams_fourth_year 
  (a b c d e : ℕ) 
  (h_sum : a + b + c + d + e = 31)
  (h_order : a < b ∧ b < c ∧ c < d ∧ d < e)
  (h_fifth : e = 3 * a)
  : d = 8 := by
  sorry

end NUMINAMATH_CALUDE_exams_fourth_year_l1442_144261


namespace NUMINAMATH_CALUDE_pencils_given_to_dorothy_l1442_144212

theorem pencils_given_to_dorothy (initial_pencils : ℕ) (remaining_pencils : ℕ) 
  (h1 : initial_pencils = 142) 
  (h2 : remaining_pencils = 111) : 
  initial_pencils - remaining_pencils = 31 := by
  sorry

end NUMINAMATH_CALUDE_pencils_given_to_dorothy_l1442_144212


namespace NUMINAMATH_CALUDE_average_after_12th_innings_l1442_144267

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  lastScore : ℕ
  averageIncrease : ℕ
  neverNotOut : Bool

/-- Calculates the average score after the latest innings -/
def calculateAverage (b : Batsman) : ℚ :=
  if b.innings = 0 then 0
  else (b.innings * (b.averageIncrease : ℚ) + b.lastScore) / b.innings

/-- Theorem stating the average after 12th innings -/
theorem average_after_12th_innings (b : Batsman) 
  (h1 : b.innings = 12)
  (h2 : b.lastScore = 55)
  (h3 : b.averageIncrease = 1)
  (h4 : b.neverNotOut = true) :
  calculateAverage b = 44 := by
  sorry


end NUMINAMATH_CALUDE_average_after_12th_innings_l1442_144267
