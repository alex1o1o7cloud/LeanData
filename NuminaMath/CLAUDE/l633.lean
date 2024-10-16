import Mathlib

namespace NUMINAMATH_CALUDE_equation_equivalence_l633_63357

theorem equation_equivalence (x : ℝ) : x^2 - 4*x - 4 = 0 ↔ (x - 2)^2 = 8 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l633_63357


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_l633_63354

theorem pure_imaginary_solutions : 
  let f (x : ℂ) := x^6 - 6*x^5 + 15*x^4 - 20*x^3 + 27*x^2 - 18*x - 8
  let y := Real.sqrt ((Real.sqrt 52 - 5) / 3)
  f (Complex.I * y) = 0 ∧ f (-Complex.I * y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_l633_63354


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l633_63380

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x > 0, x^2 + 1/x^2 ≥ 2) ∧
  (∃ x ≤ 0, x ≠ 0 ∧ x^2 + 1/x^2 ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l633_63380


namespace NUMINAMATH_CALUDE_min_value_of_f_l633_63349

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x - 2| + 3 * |x - 3| + 4 * |x - 4|

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = 8) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l633_63349


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l633_63382

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 + 3 * Complex.I) = 31/13 - (1/13) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l633_63382


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l633_63325

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, 1; 4, -2]

theorem inverse_as_linear_combination :
  N⁻¹ = (1 / 10 : ℚ) • N + (-1 / 10 : ℚ) • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l633_63325


namespace NUMINAMATH_CALUDE_percentage_with_neither_condition_l633_63329

/-- Given a survey of teachers, calculate the percentage who have neither high blood pressure nor heart trouble. -/
theorem percentage_with_neither_condition
  (total : ℕ)
  (high_blood_pressure : ℕ)
  (heart_trouble : ℕ)
  (both : ℕ)
  (h1 : total = 150)
  (h2 : high_blood_pressure = 80)
  (h3 : heart_trouble = 60)
  (h4 : both = 30)
  : (total - (high_blood_pressure + heart_trouble - both)) / total * 100 = 800 / 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_with_neither_condition_l633_63329


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_negative_seven_l633_63392

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x : ℝ | ∃ (a b : ℝ), x^2 + a*x + b ≤ 0}

-- State the theorem
theorem sum_of_coefficients_is_negative_seven 
  (h_union : A ∪ B = Set.univ)
  (h_intersection : A ∩ B = Set.Ioc 3 4)
  : ∃ (a b : ℝ), B = {x : ℝ | x^2 + a*x + b ≤ 0} ∧ a + b = -7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_negative_seven_l633_63392


namespace NUMINAMATH_CALUDE_solution_set_inequality_l633_63361

theorem solution_set_inequality (x : ℝ) :
  (-x^2 + 3*x - 2 ≥ 0) ↔ (1 ≤ x ∧ x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l633_63361


namespace NUMINAMATH_CALUDE_absolute_quadratic_inequality_l633_63340

/-- The set of real numbers x satisfying |x^2 - 4x + 3| ≤ 3 is equal to the closed interval [0, 4]. -/
theorem absolute_quadratic_inequality (x : ℝ) :
  |x^2 - 4*x + 3| ≤ 3 ↔ x ∈ Set.Icc 0 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_quadratic_inequality_l633_63340


namespace NUMINAMATH_CALUDE_three_collinear_points_same_color_l633_63309

-- Define a color type
inductive Color
| Black
| White

-- Define a point as a pair of real number (position) and color
structure Point where
  position : ℝ
  color : Color

-- Define a function to check if three points are collinear with one in the middle
def areCollinearWithMiddle (p1 p2 p3 : Point) : Prop :=
  p2.position = (p1.position + p3.position) / 2

-- State the theorem
theorem three_collinear_points_same_color (points : Set Point) : 
  ∃ (p1 p2 p3 : Point), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
  p1.color = p2.color ∧ p2.color = p3.color ∧
  areCollinearWithMiddle p1 p2 p3 := by
  sorry

end NUMINAMATH_CALUDE_three_collinear_points_same_color_l633_63309


namespace NUMINAMATH_CALUDE_man_walking_distance_l633_63308

theorem man_walking_distance (x t d : ℝ) : 
  (d = x * t) →                           -- distance = rate * time
  (d = (x + 1) * (3/4 * t)) →             -- faster speed condition
  (d = (x - 1) * (t + 3)) →               -- slower speed condition
  (d = 18) :=                             -- distance is 18 miles
by sorry

end NUMINAMATH_CALUDE_man_walking_distance_l633_63308


namespace NUMINAMATH_CALUDE_village_population_l633_63364

theorem village_population (initial_population : ℕ) : 
  (initial_population : ℝ) * (1 - 0.08) * (1 - 0.15) = 3553 → 
  initial_population = 4547 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l633_63364


namespace NUMINAMATH_CALUDE_prime_triplet_with_perfect_square_sum_l633_63330

theorem prime_triplet_with_perfect_square_sum (p₁ p₂ p₃ : ℕ) : 
  Prime p₁ → Prime p₂ → Prime p₃ → 
  p₂ ≠ p₃ → 
  ∃ x y : ℕ, x^2 = 4 + p₁ * p₂ ∧ y^2 = 4 + p₁ * p₃ → 
  ((p₁ = 7 ∧ p₂ = 11 ∧ p₃ = 3) ∨ (p₁ = 7 ∧ p₂ = 3 ∧ p₃ = 11)) := by
sorry

end NUMINAMATH_CALUDE_prime_triplet_with_perfect_square_sum_l633_63330


namespace NUMINAMATH_CALUDE_square_diff_squares_squared_l633_63378

theorem square_diff_squares_squared : (7^2 - 3^2)^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_squares_squared_l633_63378


namespace NUMINAMATH_CALUDE_converse_opposite_numbers_correct_l633_63343

theorem converse_opposite_numbers_correct :
  (∀ x y : ℝ, x = -y → x + y = 0) := by sorry

end NUMINAMATH_CALUDE_converse_opposite_numbers_correct_l633_63343


namespace NUMINAMATH_CALUDE_max_side_length_of_triangle_l633_63371

/-- A triangle with integer side lengths and perimeter 24 has a maximum side length of 11 -/
theorem max_side_length_of_triangle (a b c : ℕ) : 
  a < b ∧ b < c ∧ 
  a + b + c = 24 ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  c ≤ 11 := by
  sorry

end NUMINAMATH_CALUDE_max_side_length_of_triangle_l633_63371


namespace NUMINAMATH_CALUDE_problem_solution_l633_63304

theorem problem_solution (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x / 2 = y^2) 
  (h3 : x / 5 = 3*y) : 
  x = 112.5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l633_63304


namespace NUMINAMATH_CALUDE_total_marks_calculation_l633_63365

/-- Given 50 candidates in an examination with an average mark of 40,
    prove that the total marks is 2000. -/
theorem total_marks_calculation (num_candidates : ℕ) (average_mark : ℚ) :
  num_candidates = 50 →
  average_mark = 40 →
  (num_candidates : ℚ) * average_mark = 2000 := by
  sorry

end NUMINAMATH_CALUDE_total_marks_calculation_l633_63365


namespace NUMINAMATH_CALUDE_sophies_doughnuts_price_l633_63398

theorem sophies_doughnuts_price (cupcake_price cupcake_count doughnut_count
  pie_price pie_count cookie_price cookie_count total_spent : ℚ) :
  cupcake_price = 2 →
  cupcake_count = 5 →
  doughnut_count = 6 →
  pie_price = 2 →
  pie_count = 4 →
  cookie_price = 0.60 →
  cookie_count = 15 →
  total_spent = 33 →
  cupcake_price * cupcake_count + doughnut_count * 1 + pie_price * pie_count + cookie_price * cookie_count = total_spent :=
by
  sorry

#check sophies_doughnuts_price

end NUMINAMATH_CALUDE_sophies_doughnuts_price_l633_63398


namespace NUMINAMATH_CALUDE_negation_of_cube_odd_is_odd_l633_63306

theorem negation_of_cube_odd_is_odd :
  (¬ ∀ n : ℤ, Odd n → Odd (n^3)) ↔ (∃ n : ℤ, Odd n ∧ Even (n^3)) :=
sorry

end NUMINAMATH_CALUDE_negation_of_cube_odd_is_odd_l633_63306


namespace NUMINAMATH_CALUDE_faye_age_l633_63386

/-- Represents the ages of Chad, Diana, Eduardo, and Faye --/
structure Ages where
  chad : ℕ
  diana : ℕ
  eduardo : ℕ
  faye : ℕ

/-- The conditions given in the problem --/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.diana = ages.eduardo - 5 ∧
  ages.eduardo = ages.chad + 3 ∧
  ages.faye = ages.chad + 2 ∧
  ages.diana = 18

theorem faye_age (ages : Ages) :
  satisfiesConditions ages → ages.faye = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_faye_age_l633_63386


namespace NUMINAMATH_CALUDE_final_attendance_is_1166_l633_63359

/-- Calculates the final number of spectators after a series of changes in attendance at a football game. -/
def final_attendance (initial_total initial_boys initial_girls : ℕ) : ℕ :=
  let initial_adults := initial_total - (initial_boys + initial_girls)
  
  -- After first quarter
  let boys_after_q1 := initial_boys - (initial_boys / 4)
  let girls_after_q1 := initial_girls - (initial_girls / 8)
  let adults_after_q1 := initial_adults - (initial_adults / 5)
  
  -- After halftime
  let boys_after_half := boys_after_q1 + (boys_after_q1 * 5 / 100)
  let girls_after_half := girls_after_q1 + (girls_after_q1 * 7 / 100)
  let adults_after_half := adults_after_q1 + 50
  
  -- After third quarter
  let boys_after_q3 := boys_after_half - (boys_after_half * 3 / 100)
  let girls_after_q3 := girls_after_half - (girls_after_half * 4 / 100)
  let adults_after_q3 := adults_after_half + (adults_after_half * 2 / 100)
  
  -- Final numbers
  let final_boys := boys_after_q3 + 15
  let final_girls := girls_after_q3 + 25
  let final_adults := adults_after_q3 - (adults_after_q3 / 100)
  
  final_boys + final_girls + final_adults

/-- Theorem stating that given the initial conditions, the final attendance is 1166. -/
theorem final_attendance_is_1166 : final_attendance 1300 350 450 = 1166 := by
  sorry

end NUMINAMATH_CALUDE_final_attendance_is_1166_l633_63359


namespace NUMINAMATH_CALUDE_total_insects_l633_63383

theorem total_insects (leaves : ℕ) (ladybugs_per_leaf : ℕ) (stones : ℕ) (ants_per_stone : ℕ) 
  (bees : ℕ) (flowers : ℕ) : 
  leaves = 345 → 
  ladybugs_per_leaf = 267 → 
  stones = 178 → 
  ants_per_stone = 423 → 
  bees = 498 → 
  flowers = 6 → 
  leaves * ladybugs_per_leaf + stones * ants_per_stone + bees = 167967 := by
  sorry

end NUMINAMATH_CALUDE_total_insects_l633_63383


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l633_63381

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

-- Define the conditions of the problem
def sequence_conditions (a : ℕ → ℝ) : Prop :=
  arithmetic_sequence a ∧
  a 1 = 1 ∧
  a 3 = Real.sqrt (a 1 * a 9)

-- State the theorem
theorem arithmetic_sequence_theorem (a : ℕ → ℝ) :
  sequence_conditions a → ∀ n : ℕ, a n = n := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l633_63381


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_value_l633_63331

/-- Given two lines in the general form ax + by + c = 0, this function returns true if they are perpendicular --/
def are_perpendicular (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

/-- The problem statement --/
theorem perpendicular_lines_m_value :
  ∀ m : ℝ, are_perpendicular m 4 (-2) 2 (-5) 1 → m = 10 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_value_l633_63331


namespace NUMINAMATH_CALUDE_system_I_solution_system_II_solution_l633_63313

-- System I
theorem system_I_solution :
  ∃ (x y : ℝ), (y = x + 3 ∧ x - 2*y + 12 = 0) → (x = 6 ∧ y = 9) := by sorry

-- System II
theorem system_II_solution :
  ∃ (x y : ℝ), (4*(x - y - 1) = 3*(1 - y) - 2 ∧ x/2 + y/3 = 2) → (x = 2 ∧ y = 3) := by sorry

end NUMINAMATH_CALUDE_system_I_solution_system_II_solution_l633_63313


namespace NUMINAMATH_CALUDE_kolya_purchase_options_l633_63346

-- Define the store's pricing rule
def item_price (rubles : ℕ) : ℕ := 100 * rubles + 99

-- Define Kolya's total purchase amount in kopecks
def total_purchase : ℕ := 20083

-- Define the possible number of items
def possible_items : Set ℕ := {17, 117}

-- Theorem statement
theorem kolya_purchase_options :
  ∀ n : ℕ, (∃ r : ℕ, n * item_price r = total_purchase) ↔ n ∈ possible_items :=
by sorry

end NUMINAMATH_CALUDE_kolya_purchase_options_l633_63346


namespace NUMINAMATH_CALUDE_caps_lost_per_year_l633_63327

def caps_first_year : ℕ := 3 * 12
def caps_subsequent_years (years : ℕ) : ℕ := 5 * 12 * years
def christmas_caps (years : ℕ) : ℕ := 40 * years
def total_collection_years : ℕ := 5
def current_cap_count : ℕ := 401

theorem caps_lost_per_year :
  let total_caps := caps_first_year + 
                    caps_subsequent_years (total_collection_years - 1) + 
                    christmas_caps total_collection_years
  let total_lost := total_caps - current_cap_count
  (total_lost / total_collection_years : ℚ) = 15 := by sorry

end NUMINAMATH_CALUDE_caps_lost_per_year_l633_63327


namespace NUMINAMATH_CALUDE_investment_result_unique_initial_investment_l633_63373

/-- Represents the growth of an investment over time with compound interest and additional investments. -/
def investment_growth (initial_investment : ℝ) : ℝ :=
  let after_compound := initial_investment * (1 + 0.20)^3
  let after_triple := after_compound * 3
  after_triple * (1 + 0.15)

/-- Theorem stating that an initial investment of $10,000 results in $59,616 after the given growth pattern. -/
theorem investment_result : investment_growth 10000 = 59616 := by
  sorry

/-- Theorem proving the uniqueness of the initial investment that results in $59,616. -/
theorem unique_initial_investment (x : ℝ) :
  investment_growth x = 59616 → x = 10000 := by
  sorry

end NUMINAMATH_CALUDE_investment_result_unique_initial_investment_l633_63373


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l633_63379

/-- Given three points A, B, and C in a 2D plane, this function checks if they are collinear -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  (C.2 - A.2) * (B.1 - A.1) = (B.2 - A.2) * (C.1 - A.1)

/-- The theorem states that if points A(3,1), B(-2,k), and C(8,11) are collinear, then k = -9 -/
theorem collinear_points_k_value :
  collinear (3, 1) (-2, k) (8, 11) → k = -9 := by
  sorry


end NUMINAMATH_CALUDE_collinear_points_k_value_l633_63379


namespace NUMINAMATH_CALUDE_second_divisor_l633_63352

theorem second_divisor (n : ℕ) : 
  (n ≠ 12 ∧ n ≠ 18 ∧ n ≠ 21 ∧ n ≠ 28) →
  (1008 % n = 0) →
  (∀ m : ℕ, m < n → m ≠ 12 → m ≠ 18 → m ≠ 21 → m ≠ 28 → 1008 % m ≠ 0) →
  n = 14 :=
by sorry

end NUMINAMATH_CALUDE_second_divisor_l633_63352


namespace NUMINAMATH_CALUDE_multiply_add_theorem_l633_63311

theorem multiply_add_theorem : 15 * 30 + 45 * 15 + 90 = 1215 := by
  sorry

end NUMINAMATH_CALUDE_multiply_add_theorem_l633_63311


namespace NUMINAMATH_CALUDE_election_result_l633_63320

/-- Represents the total number of valid votes cast in the election -/
def total_votes : ℕ := sorry

/-- Represents the number of votes received by the winning candidate -/
def winning_votes : ℕ := 7320

/-- Represents the percentage of votes received by the winning candidate after redistribution -/
def winning_percentage : ℚ := 43 / 100

theorem election_result :
  total_votes * winning_percentage = winning_votes ∧
  total_votes ≥ 17023 ∧
  total_votes < 17024 :=
sorry

end NUMINAMATH_CALUDE_election_result_l633_63320


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l633_63324

theorem quadratic_roots_sum (p q : ℝ) : 
  p^2 - 5*p + 6 = 0 → q^2 - 5*q + 6 = 0 → p^3 + p^4*q^2 + p^2*q^4 + q^3 = 503 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l633_63324


namespace NUMINAMATH_CALUDE_order_total_parts_l633_63345

theorem order_total_parts (total_cost : ℕ) (cost_cheap : ℕ) (cost_expensive : ℕ) (num_expensive : ℕ) :
  total_cost = 2380 →
  cost_cheap = 20 →
  cost_expensive = 50 →
  num_expensive = 40 →
  ∃ (num_cheap : ℕ), num_cheap * cost_cheap + num_expensive * cost_expensive = total_cost ∧
                      num_cheap + num_expensive = 59 :=
by sorry

end NUMINAMATH_CALUDE_order_total_parts_l633_63345


namespace NUMINAMATH_CALUDE_impossible_equal_sum_configuration_l633_63377

theorem impossible_equal_sum_configuration : ¬ ∃ (a b c d e f : ℕ),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
   d ≠ e ∧ d ≠ f ∧
   e ≠ f) ∧
  (a + d + e = b + d + f) ∧
  (a + d + e = c + e + f) ∧
  (b + d + f = c + e + f) :=
by
  sorry

end NUMINAMATH_CALUDE_impossible_equal_sum_configuration_l633_63377


namespace NUMINAMATH_CALUDE_minimum_cost_for_25_apples_l633_63318

/-- Represents a group of apples with its cost -/
structure AppleGroup where
  count : Nat
  cost : Nat

/-- Calculates the total number of apples from a list of apple groups -/
def totalApples (groups : List AppleGroup) : Nat :=
  groups.foldl (fun sum group => sum + group.count) 0

/-- Calculates the total cost from a list of apple groups -/
def totalCost (groups : List AppleGroup) : Nat :=
  groups.foldl (fun sum group => sum + group.cost) 0

/-- Represents the store's apple pricing policy -/
def applePricing : List AppleGroup := [
  { count := 4, cost := 15 },
  { count := 7, cost := 25 }
]

theorem minimum_cost_for_25_apples :
  ∃ (purchase : List AppleGroup),
    totalApples purchase = 25 ∧
    purchase.length ≥ 3 ∧
    (∀ group ∈ purchase, group ∈ applePricing) ∧
    totalCost purchase = 90 ∧
    (∀ (other : List AppleGroup),
      totalApples other = 25 →
      other.length ≥ 3 →
      (∀ group ∈ other, group ∈ applePricing) →
      totalCost purchase ≤ totalCost other) :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_cost_for_25_apples_l633_63318


namespace NUMINAMATH_CALUDE_exponential_inequality_l633_63322

theorem exponential_inequality (x : ℝ) : (2 : ℝ) ^ (2 * x - 7) > (2 : ℝ) ^ (4 * x - 1) ↔ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l633_63322


namespace NUMINAMATH_CALUDE_rectangle_area_l633_63338

theorem rectangle_area (a b c : ℝ) (h : a > 0) (h' : b > 0) (h'' : c > 0) 
  (h_pythagorean : a^2 + b^2 = c^2) : a * b = a * b :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l633_63338


namespace NUMINAMATH_CALUDE_fraction_equality_l633_63319

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / ((a : ℚ) + 37) = 925 / 1000 → a = 455 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l633_63319


namespace NUMINAMATH_CALUDE_a_minus_b_values_l633_63335

theorem a_minus_b_values (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 13) (h3 : a * b > 0) :
  a - b = 10 ∨ a - b = -10 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_values_l633_63335


namespace NUMINAMATH_CALUDE_airplane_shot_down_probability_l633_63328

def probability_airplane_shot_down : ℝ :=
  let p_A : ℝ := 0.4
  let p_B : ℝ := 0.5
  let p_C : ℝ := 0.8
  let p_one_hit : ℝ := 0.4
  let p_two_hit : ℝ := 0.7
  let p_three_hit : ℝ := 1

  let p_A_miss : ℝ := 1 - p_A
  let p_B_miss : ℝ := 1 - p_B
  let p_C_miss : ℝ := 1 - p_C

  let p_one_person_hits : ℝ := 
    (p_A * p_B_miss * p_C_miss + p_A_miss * p_B * p_C_miss + p_A_miss * p_B_miss * p_C) * p_one_hit

  let p_two_people_hit : ℝ := 
    (p_A * p_B * p_C_miss + p_A * p_B_miss * p_C + p_A_miss * p_B * p_C) * p_two_hit

  let p_all_hit : ℝ := p_A * p_B * p_C * p_three_hit

  p_one_person_hits + p_two_people_hit + p_all_hit

theorem airplane_shot_down_probability : 
  probability_airplane_shot_down = 0.604 := by sorry

end NUMINAMATH_CALUDE_airplane_shot_down_probability_l633_63328


namespace NUMINAMATH_CALUDE_equal_angles_on_curve_l633_63337

/-- Curve C defined by y² = 4x -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Point on the x-axis -/
def xAxisPoint (x : ℝ) : ℝ × ℝ := (x, 0)

/-- Line passing through two points -/
def line (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, r = (1 - t) • p + t • q}

/-- Angle between two vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem equal_angles_on_curve (m n : ℝ) (hm : m > 0) (hmn : m + n = 0)
    (A B : ℝ × ℝ) (hA : A ∈ C) (hB : B ∈ C)
    (hline : A ∈ line (xAxisPoint m) B ∧ B ∈ line (xAxisPoint m) A) :
  angle (A - xAxisPoint n) (xAxisPoint m - xAxisPoint n) =
  angle (B - xAxisPoint n) (xAxisPoint m - xAxisPoint n) := by
  sorry

end NUMINAMATH_CALUDE_equal_angles_on_curve_l633_63337


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l633_63384

open Set

-- Define the sets M and N
def M : Set ℝ := {x | 2 - x < 0}
def N : Set ℝ := {x | x - 3 ≤ 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Ioo 2 3 ∪ Iic 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l633_63384


namespace NUMINAMATH_CALUDE_number_problem_l633_63317

theorem number_problem (N p q : ℝ) 
  (h1 : N / p = 4)
  (h2 : N / q = 18)
  (h3 : p - q = 0.5833333333333334) : 
  N = 3 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l633_63317


namespace NUMINAMATH_CALUDE_max_zombies_after_four_days_l633_63393

/-- The maximum number of zombies in a mall after 4 days of doubling, given initial constraints -/
theorem max_zombies_after_four_days (initial_zombies : ℕ) : 
  initial_zombies < 50 → 
  (initial_zombies * 2^4 : ℕ) ≤ 48 :=
by sorry

end NUMINAMATH_CALUDE_max_zombies_after_four_days_l633_63393


namespace NUMINAMATH_CALUDE_equation_implies_fraction_value_l633_63314

theorem equation_implies_fraction_value
  (a x y : ℝ)
  (h : x * Real.sqrt (a * (x - a)) + y * Real.sqrt (a * (y - a)) = Real.sqrt (abs (Real.log (x - a) - Real.log (a - y)))) :
  (3 * x^2 + x * y - y^2) / (x^2 - x * y + y^2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_equation_implies_fraction_value_l633_63314


namespace NUMINAMATH_CALUDE_extreme_points_imply_a_range_and_negative_min_l633_63339

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2 - x

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

-- Theorem statement
theorem extreme_points_imply_a_range_and_negative_min 
  (a : ℝ) (x₁ x₂ : ℝ) (h_extreme : x₁ < x₂ ∧ 
    f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0 ∧
    (∀ x, x₁ < x → x < x₂ → f_deriv a x ≠ 0)) :
  (0 < a ∧ a < Real.exp (-1)) ∧ f a x₁ < 0 := by
  sorry

end NUMINAMATH_CALUDE_extreme_points_imply_a_range_and_negative_min_l633_63339


namespace NUMINAMATH_CALUDE_intersection_of_sets_l633_63344

theorem intersection_of_sets : 
  let M : Set ℤ := {0, 1, 2, 3}
  let P : Set ℤ := {-1, 1, -2, 2}
  M ∩ P = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l633_63344


namespace NUMINAMATH_CALUDE_unique_y_value_l633_63342

theorem unique_y_value (x : ℝ) (h : x^2 + 4 * (x / (x + 3))^2 = 64) : 
  ((x + 3)^2 * (x - 2)) / (2 * x + 3) = 250 / 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_y_value_l633_63342


namespace NUMINAMATH_CALUDE_problem_solution_l633_63333

theorem problem_solution : 
  (2/3 - 3/4 + 1/6) / (-1/24) = -2 ∧ 
  -2^3 + 3 * (-1)^2023 - |3-7| = -15 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l633_63333


namespace NUMINAMATH_CALUDE_triangle_similarity_l633_63370

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Triangle : Type :=
  (A B C : Point)

-- Define the properties
def isAcute (t : Triangle) : Prop := sorry

def incircleTouchPoints (t : Triangle) (D E F : Point) : Prop := sorry

def isCircumcenter (P : Point) (t : Triangle) : Prop := sorry

-- Main theorem
theorem triangle_similarity (A B C D E F P Q R : Point) :
  let ABC := Triangle.mk A B C
  let AEF := Triangle.mk A E F
  let BDF := Triangle.mk B D F
  let CDE := Triangle.mk C D E
  let PQR := Triangle.mk P Q R
  isAcute ABC →
  incircleTouchPoints ABC D E F →
  isCircumcenter P AEF →
  isCircumcenter Q BDF →
  isCircumcenter R CDE →
  -- Conclusion: ABC and PQR are similar
  ∃ (k : ℝ), k > 0 ∧
    (P.x - Q.x)^2 + (P.y - Q.y)^2 = k * ((A.x - B.x)^2 + (A.y - B.y)^2) ∧
    (Q.x - R.x)^2 + (Q.y - R.y)^2 = k * ((B.x - C.x)^2 + (B.y - C.y)^2) ∧
    (R.x - P.x)^2 + (R.y - P.y)^2 = k * ((C.x - A.x)^2 + (C.y - A.y)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_similarity_l633_63370


namespace NUMINAMATH_CALUDE_dog_shampoo_time_l633_63360

theorem dog_shampoo_time (total_time hosing_time : ℕ) (shampoo_count : ℕ) : 
  total_time = 55 → 
  hosing_time = 10 → 
  shampoo_count = 3 → 
  (total_time - hosing_time) / shampoo_count = 15 := by
  sorry

end NUMINAMATH_CALUDE_dog_shampoo_time_l633_63360


namespace NUMINAMATH_CALUDE_domain_equals_range_l633_63394

-- Define the function f(x) = |x-2| - 2
def f (x : ℝ) : ℝ := |x - 2| - 2

-- Define the domain set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define the range set N
def N : Set ℝ := f '' M

-- Theorem stating that M equals N
theorem domain_equals_range : M = N := by sorry

end NUMINAMATH_CALUDE_domain_equals_range_l633_63394


namespace NUMINAMATH_CALUDE_dessert_preference_l633_63350

theorem dessert_preference (total : ℕ) (apple : ℕ) (chocolate : ℕ) (neither : ℕ)
  (h1 : total = 50)
  (h2 : apple = 22)
  (h3 : chocolate = 20)
  (h4 : neither = 15) :
  apple + chocolate - (total - neither) = 7 :=
by sorry

end NUMINAMATH_CALUDE_dessert_preference_l633_63350


namespace NUMINAMATH_CALUDE_diagonal_intersection_l633_63387

/-- A regular 18-sided polygon -/
structure RegularPolygon18 where
  vertices : Fin 18 → ℝ × ℝ
  is_regular : ∀ i j : Fin 18, 
    dist (vertices i) (vertices ((i + 1) % 18)) = 
    dist (vertices j) (vertices ((j + 1) % 18))

/-- A diagonal of the polygon -/
def diagonal (p : RegularPolygon18) (i j : Fin 18) : Set (ℝ × ℝ) :=
  {x | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = (1 - t) • p.vertices i + t • p.vertices j}

/-- The statement to be proved -/
theorem diagonal_intersection (p : RegularPolygon18) :
  ∃ x : ℝ × ℝ, x ∈ diagonal p 1 11 ∩ diagonal p 7 17 ∩ diagonal p 4 15 ∧
  (∀ i : Fin 18, x ∉ diagonal p i ((i + 9) % 18)) :=
sorry

end NUMINAMATH_CALUDE_diagonal_intersection_l633_63387


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l633_63389

/-- Given two lines l₁ and l₂ in the plane, prove that a=2 is a necessary and sufficient condition for l₁ to be parallel to l₂. -/
theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, 2*x - a*y + 1 = 0 ↔ (a-1)*x - y + a = 0) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l633_63389


namespace NUMINAMATH_CALUDE_solution_not_zero_l633_63367

theorem solution_not_zero (a : ℝ) : ∀ x : ℝ, x = a * x + 1 → x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_not_zero_l633_63367


namespace NUMINAMATH_CALUDE_cell_division_genetic_info_l633_63326

-- Define the types for cells and genetic information
variable (Cell GeneticInfo : Type)

-- Define the functions for getting genetic information from a cell
variable (genetic_info : Cell → GeneticInfo)

-- Define the cells
variable (C₁ C₂ S₁ S₂ : Cell)

-- Define the property of being daughter cells from mitosis
variable (mitosis_daughter_cells : Cell → Cell → Prop)

-- Define the property of being secondary spermatocytes from meiosis I
variable (meiosis_I_secondary_spermatocytes : Cell → Cell → Prop)

-- State the theorem
theorem cell_division_genetic_info :
  mitosis_daughter_cells C₁ C₂ →
  meiosis_I_secondary_spermatocytes S₁ S₂ →
  (genetic_info C₁ = genetic_info C₂) ∧
  (genetic_info S₁ ≠ genetic_info S₂) :=
by sorry

end NUMINAMATH_CALUDE_cell_division_genetic_info_l633_63326


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l633_63301

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := ((2 * x - y) ^ (2 / x)) ^ (1 / 2) = 2
def equation2 (x y : ℝ) : Prop := (2 * x - y) * (5 ^ (x / 4)) = 1000

-- Theorem statement
theorem solution_satisfies_system :
  ∃ (x y : ℝ), x = 12 ∧ y = 16 ∧ equation1 x y ∧ equation2 x y :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l633_63301


namespace NUMINAMATH_CALUDE_area_enclosed_by_function_and_line_l633_63363

theorem area_enclosed_by_function_and_line (c : ℝ) : 
  30 = (1/2) * (c + 2) * (c - 2) → c = 8 := by
sorry

end NUMINAMATH_CALUDE_area_enclosed_by_function_and_line_l633_63363


namespace NUMINAMATH_CALUDE_letitia_order_l633_63332

theorem letitia_order (julie_order anton_order individual_tip tip_percentage : ℚ) 
  (h1 : julie_order = 10)
  (h2 : anton_order = 30)
  (h3 : individual_tip = 4)
  (h4 : tip_percentage = 1/5)
  : ∃ letitia_order : ℚ, 
    tip_percentage * (julie_order + letitia_order + anton_order) = 3 * individual_tip ∧ 
    letitia_order = 20 := by
  sorry

end NUMINAMATH_CALUDE_letitia_order_l633_63332


namespace NUMINAMATH_CALUDE_rhombus_count_in_triangle_l633_63355

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ
  positive : sideLength > 0

/-- Represents a rhombus composed of smaller triangles -/
structure Rhombus where
  smallTriangles : ℕ

/-- The number of rhombuses in a large equilateral triangle -/
def countRhombuses (largeTriangle : EquilateralTriangle) (smallTriangleSideLength : ℝ) (rhombusSize : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem rhombus_count_in_triangle :
  let largeTriangle : EquilateralTriangle := ⟨10, by norm_num⟩
  let smallTriangleSideLength : ℝ := 1
  let rhombusSize : ℕ := 8
  countRhombuses largeTriangle smallTriangleSideLength rhombusSize = 84 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_count_in_triangle_l633_63355


namespace NUMINAMATH_CALUDE_first_candidate_percentage_l633_63303

/-- Given an election between two candidates where the total number of votes is 600
    and the second candidate received 240 votes, prove that the first candidate
    received 60% of the votes. -/
theorem first_candidate_percentage (total_votes : ℕ) (second_candidate_votes : ℕ)
  (h1 : total_votes = 600)
  (h2 : second_candidate_votes = 240) :
  (total_votes - second_candidate_votes : ℚ) / total_votes * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_first_candidate_percentage_l633_63303


namespace NUMINAMATH_CALUDE_parabola_shift_l633_63323

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := 2 * x^2

/-- The shifted parabola function -/
def shifted_parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 - 5

/-- The horizontal shift amount -/
def h_shift : ℝ := 1

/-- The vertical shift amount -/
def v_shift : ℝ := -5

theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - h_shift) + v_shift :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l633_63323


namespace NUMINAMATH_CALUDE_min_value_fraction_min_value_achievable_l633_63305

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  (b^2 + a + 1) / (a * b) ≥ 2 * Real.sqrt 10 + 6 :=
by sorry

theorem min_value_achievable (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 1 ∧
    (b₀^2 + a₀ + 1) / (a₀ * b₀) = 2 * Real.sqrt 10 + 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_min_value_achievable_l633_63305


namespace NUMINAMATH_CALUDE_one_basket_of_peaches_l633_63300

def basket_count (red_peaches green_peaches total_peaches : ℕ) : ℕ :=
  if red_peaches + green_peaches = total_peaches then 1 else 0

theorem one_basket_of_peaches (red_peaches green_peaches total_peaches : ℕ) 
  (h1 : red_peaches = 7)
  (h2 : green_peaches = 3)
  (h3 : total_peaches = 10) :
  basket_count red_peaches green_peaches total_peaches = 1 := by
sorry

end NUMINAMATH_CALUDE_one_basket_of_peaches_l633_63300


namespace NUMINAMATH_CALUDE_jeff_cabinets_l633_63368

/-- Calculates the total number of cabinets Jeff has after installations -/
def total_cabinets (initial : ℕ) (counters : ℕ) (additional : ℕ) : ℕ :=
  initial + counters * (2 * initial) + additional

/-- Proves that Jeff has 26 cabinets in total -/
theorem jeff_cabinets : total_cabinets 3 3 5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_jeff_cabinets_l633_63368


namespace NUMINAMATH_CALUDE_dark_tile_fraction_l633_63395

theorem dark_tile_fraction (block_size : ℕ) (dark_tiles : ℕ) : 
  block_size = 8 → 
  dark_tiles = 18 → 
  (dark_tiles : ℚ) / (block_size * block_size : ℚ) = 9 / 32 :=
by sorry

end NUMINAMATH_CALUDE_dark_tile_fraction_l633_63395


namespace NUMINAMATH_CALUDE_complement_union_theorem_l633_63307

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4}

-- Define set A
def A : Set Nat := {0, 1, 2}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l633_63307


namespace NUMINAMATH_CALUDE_min_value_of_function_l633_63397

theorem min_value_of_function (x : ℝ) (h : x ≥ 0) :
  (4 * x^2 + 12 * x + 25) / (6 * (1 + x)) ≥ 8/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l633_63397


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l633_63348

theorem opposite_of_negative_2023 : -(Int.neg 2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l633_63348


namespace NUMINAMATH_CALUDE_demolition_time_with_injury_l633_63399

/-- The time it takes to demolish a building given the work rates of different combinations of workers and an injury to one worker. -/
theorem demolition_time_with_injury 
  (carl_bob_rate : ℚ)
  (anne_bob_rate : ℚ)
  (anne_carl_rate : ℚ)
  (h_carl_bob : carl_bob_rate = 1 / 6)
  (h_anne_bob : anne_bob_rate = 1 / 3)
  (h_anne_carl : anne_carl_rate = 1 / 5) :
  let all_rate := carl_bob_rate + anne_bob_rate + anne_carl_rate - 1 / 2
  let work_done_day_one := all_rate
  let remaining_work := 1 - work_done_day_one
  let time_for_remainder := remaining_work / anne_bob_rate
  1 + time_for_remainder = 59 / 20 := by
sorry

end NUMINAMATH_CALUDE_demolition_time_with_injury_l633_63399


namespace NUMINAMATH_CALUDE_nicks_nacks_nocks_conversion_l633_63374

/-- Given the conversion rates between nicks, nacks, and nocks, 
    prove that 40 nocks is equal to 160/3 nicks. -/
theorem nicks_nacks_nocks_conversion 
  (h1 : (5 : ℚ) * nick = 3 * nack)
  (h2 : (4 : ℚ) * nack = 5 * nock)
  : (40 : ℚ) * nock = 160 / 3 * nick :=
by sorry

end NUMINAMATH_CALUDE_nicks_nacks_nocks_conversion_l633_63374


namespace NUMINAMATH_CALUDE_sum_le_product_plus_two_l633_63302

theorem sum_le_product_plus_two (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) :
  x + y + z ≤ x*y*z + 2 := by sorry

end NUMINAMATH_CALUDE_sum_le_product_plus_two_l633_63302


namespace NUMINAMATH_CALUDE_ones_digit_8_pow_32_l633_63351

/-- The ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- The ones digit of 8^n for any natural number n -/
def ones_digit_8_pow (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 8
  | 2 => 4
  | 3 => 2
  | _ => 0  -- This case is unreachable, but needed for exhaustiveness

theorem ones_digit_8_pow_32 :
  ones_digit (8^32) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_8_pow_32_l633_63351


namespace NUMINAMATH_CALUDE_product_of_proper_fractions_sum_of_proper_and_improper_l633_63353

-- Define a fraction as a pair of integers where the denominator is non-zero
def Fraction := { p : ℚ // p > 0 }

-- Define a proper fraction
def isProper (f : Fraction) : Prop := f.val < 1

-- Define an improper fraction
def isImproper (f : Fraction) : Prop := f.val ≥ 1

-- Statement 2
theorem product_of_proper_fractions (f g : Fraction) 
  (hf : isProper f) (hg : isProper g) : 
  isProper ⟨f.val * g.val, by sorry⟩ := by sorry

-- Statement 3
theorem sum_of_proper_and_improper (f g : Fraction) 
  (hf : isProper f) (hg : isImproper g) : 
  isImproper ⟨f.val + g.val, by sorry⟩ := by sorry

end NUMINAMATH_CALUDE_product_of_proper_fractions_sum_of_proper_and_improper_l633_63353


namespace NUMINAMATH_CALUDE_train_length_l633_63341

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) : 
  speed_kmph = 18 → time_sec = 5 → (speed_kmph * 1000 / 3600) * time_sec = 25 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l633_63341


namespace NUMINAMATH_CALUDE_number_problem_l633_63310

theorem number_problem (x : ℝ) : 0.65 * x = 0.8 * x - 21 → x = 140 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l633_63310


namespace NUMINAMATH_CALUDE_expression_equality_l633_63369

theorem expression_equality (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z - z/x ≠ 0) :
  (x^2 - 1/y^2) / (z - z/x) = x/z :=
by sorry

end NUMINAMATH_CALUDE_expression_equality_l633_63369


namespace NUMINAMATH_CALUDE_rational_equation_solution_l633_63375

theorem rational_equation_solution :
  ∃ x : ℚ, (x^2 - 7*x + 10) / (x^2 - 6*x + 5) = (x^2 - 4*x - 21) / (x^2 - 3*x - 18) ∧ x = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l633_63375


namespace NUMINAMATH_CALUDE_tangent_lines_range_l633_63362

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Define the function g(x) = 2x^3 - 3x^2
def g (x : ℝ) : ℝ := 2*x^3 - 3*x^2

-- Theorem statement
theorem tangent_lines_range (t : ℝ) :
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (∀ x : ℝ, f x - (f a + f' a * (x - a)) = 0 → x = a) ∧
    (∀ x : ℝ, f x - (f b + f' b * (x - b)) = 0 → x = b) ∧
    (∀ x : ℝ, f x - (f c + f' c * (x - c)) = 0 → x = c) ∧
    t = f a + f' a * (3 - a) ∧
    t = f b + f' b * (3 - b) ∧
    t = f c + f' c * (3 - c)) →
  -9 < t ∧ t < 8 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_range_l633_63362


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l633_63376

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmerSpeed : ℝ
  streamSpeed : ℝ

/-- Calculates the effective speed when swimming downstream. -/
def downstreamSpeed (s : SwimmerSpeed) : ℝ :=
  s.swimmerSpeed + s.streamSpeed

/-- Calculates the effective speed when swimming upstream. -/
def upstreamSpeed (s : SwimmerSpeed) : ℝ :=
  s.swimmerSpeed - s.streamSpeed

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 7 km/h. -/
theorem swimmer_speed_in_still_water :
  ∀ (s : SwimmerSpeed),
    (downstreamSpeed s * 4 = 32) →
    (upstreamSpeed s * 4 = 24) →
    s.swimmerSpeed = 7 :=
by sorry

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l633_63376


namespace NUMINAMATH_CALUDE_at_most_one_obtuse_angle_l633_63321

-- Define a triangle
def Triangle : Type := Unit

-- Define an angle in a triangle
def Angle (t : Triangle) : Type := Unit

-- Define if an angle is obtuse
def IsObtuse (t : Triangle) (a : Angle t) : Prop := sorry

-- State the theorem
theorem at_most_one_obtuse_angle (t : Triangle) :
  ¬∃ (a b : Angle t), a ≠ b ∧ IsObtuse t a ∧ IsObtuse t b :=
sorry

end NUMINAMATH_CALUDE_at_most_one_obtuse_angle_l633_63321


namespace NUMINAMATH_CALUDE_stating_n_gon_triangulation_l633_63315

/-- 
A polygon with n sides (n-gon) can be divided into triangles by non-intersecting diagonals. 
This function represents the number of such triangles.
-/
def num_triangles (n : ℕ) : ℕ := n - 2

/-- 
Theorem stating that the number of triangles into which non-intersecting diagonals 
divide an n-gon is equal to n-2, for any n ≥ 3.
-/
theorem n_gon_triangulation (n : ℕ) (h : n ≥ 3) : 
  num_triangles n = n - 2 := by
  sorry

end NUMINAMATH_CALUDE_stating_n_gon_triangulation_l633_63315


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_small_remainder_l633_63312

theorem smallest_cube_root_with_small_remainder : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ) (r : ℝ), 0 < r ∧ r < 1/1000 ∧ (m : ℝ)^(1/3) = n + r →
    ∀ (k : ℕ) (s : ℝ), 0 < s ∧ s < 1/1000 ∧ (k : ℝ)^(1/3) = (n-1) + s → k > m) ∧
  n = 19 :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_with_small_remainder_l633_63312


namespace NUMINAMATH_CALUDE_red_balls_count_l633_63336

theorem red_balls_count (total : ℕ) (p : ℚ) (h_total : total = 12) (h_p : p = 1 / 22) :
  ∃ (r : ℕ), r ≤ total ∧ 
    (r : ℚ) / total * ((r - 1) : ℚ) / (total - 1) = p ∧
    r = 3 :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l633_63336


namespace NUMINAMATH_CALUDE_triangle_problem_l633_63388

open Real

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Acute triangle condition
  (a - b + c) * (a - b - c) + a * b = 0 →  -- First given equation
  b * c * sin C = 3 * c * cos A + 3 * a * cos C →  -- Second given equation
  c = 2 * sqrt 3 ∧ 6 < a + b ∧ a + b ≤ 4 * sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l633_63388


namespace NUMINAMATH_CALUDE_geometric_series_sum_first_5_terms_l633_63366

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_first_5_terms :
  let a : ℚ := 2
  let r : ℚ := 1/4
  let n : ℕ := 5
  geometric_series_sum a r n = 341/128 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_first_5_terms_l633_63366


namespace NUMINAMATH_CALUDE_parabola_height_comparison_l633_63356

theorem parabola_height_comparison (x₁ x₂ : ℝ) (h1 : -4 < x₁ ∧ x₁ < -2) (h2 : 0 < x₂ ∧ x₂ < 2) :
  (x₁ ^ 2 : ℝ) > x₂ ^ 2 := by sorry

end NUMINAMATH_CALUDE_parabola_height_comparison_l633_63356


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l633_63358

theorem perpendicular_vectors_x_value (x : ℝ) : 
  let a : Fin 3 → ℝ := ![2, -1, x]
  let b : Fin 3 → ℝ := ![3, 2, -1]
  (∀ i : Fin 3, (a i) * (b i) = 0) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l633_63358


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l633_63390

/-- Represents the dimensions of a rectangular parallelepiped -/
structure Cuboid where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular parallelepiped -/
def surface_area (c : Cuboid) : ℝ :=
  2 * (c.width * c.length + c.width * c.height + c.length * c.height)

/-- Theorem stating that the surface area of a cuboid with given dimensions is 340 cm² -/
theorem cuboid_surface_area :
  let c : Cuboid := ⟨8, 5, 10⟩
  surface_area c = 340 := by sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_l633_63390


namespace NUMINAMATH_CALUDE_correct_calculation_l633_63347

theorem correct_calculation (x : ℝ) (h : x * 3 = 18) : x / 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l633_63347


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l633_63372

theorem trigonometric_equation_solution :
  ∀ x : ℝ, (Real.sin (3 * x) * Real.cos (3 * x) + Real.cos (3 * x) * Real.sin (3 * x) = 3 / 8) ↔
  (∃ k : ℤ, x = (7.5 * π / 180) + k * (π / 2) ∨ x = (37.5 * π / 180) + k * (π / 2)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l633_63372


namespace NUMINAMATH_CALUDE_division_with_remainder_l633_63385

theorem division_with_remainder (A : ℕ) (h : 14 = A * 3 + 2) : A = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_with_remainder_l633_63385


namespace NUMINAMATH_CALUDE_equation_describes_cone_l633_63316

/-- Cylindrical coordinates -/
structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- Predicate for points satisfying θ = 2z -/
def SatisfiesEquation (p : CylindricalCoord) : Prop :=
  p.θ = 2 * p.z

/-- Predicate for points on a cone -/
def OnCone (p : CylindricalCoord) (α : ℝ) : Prop :=
  p.r = α * p.z

theorem equation_describes_cone :
  ∃ α : ℝ, ∀ p : CylindricalCoord, SatisfiesEquation p → OnCone p α :=
sorry

end NUMINAMATH_CALUDE_equation_describes_cone_l633_63316


namespace NUMINAMATH_CALUDE_range_of_m_l633_63334

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 2 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*m*x + 1 > 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (¬(p m ∨ q m)) → (m ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l633_63334


namespace NUMINAMATH_CALUDE_letter_lock_max_attempts_l633_63391

/-- A letter lock with a given number of rings and letters per ring. -/
structure LetterLock :=
  (num_rings : ℕ)
  (letters_per_ring : ℕ)

/-- The maximum number of distinct unsuccessful attempts for a letter lock. -/
def max_unsuccessful_attempts (lock : LetterLock) : ℕ :=
  lock.letters_per_ring ^ lock.num_rings - 1

/-- Theorem: For a letter lock with 3 rings and 6 letters per ring,
    the maximum number of distinct unsuccessful attempts is 215. -/
theorem letter_lock_max_attempts :
  let lock := LetterLock.mk 3 6
  max_unsuccessful_attempts lock = 215 := by
  sorry

end NUMINAMATH_CALUDE_letter_lock_max_attempts_l633_63391


namespace NUMINAMATH_CALUDE_jacks_remaining_money_l633_63396

def remaining_money (initial_amount snack_cost ride_multiplier game_multiplier : ℝ) : ℝ :=
  initial_amount - (snack_cost + ride_multiplier * snack_cost + game_multiplier * snack_cost)

theorem jacks_remaining_money :
  remaining_money 100 15 3 1.5 = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_jacks_remaining_money_l633_63396
