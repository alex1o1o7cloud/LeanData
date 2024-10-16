import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_not_necessary_contrapositive_equivalence_negation_equivalence_l3427_342716

-- Statement ②
theorem sufficient_not_necessary :
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧
  (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) :=
sorry

-- Statement ③
theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ↔
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) :=
sorry

-- Statement ④
theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔
  (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_contrapositive_equivalence_negation_equivalence_l3427_342716


namespace NUMINAMATH_CALUDE_power_of_product_l3427_342713

theorem power_of_product (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3427_342713


namespace NUMINAMATH_CALUDE_fraction_equality_l3427_342791

theorem fraction_equality (a b : ℚ) (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3427_342791


namespace NUMINAMATH_CALUDE_hyperbola_ratio_l3427_342773

theorem hyperbola_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (3^2 / a^2 - (3 * Real.sqrt 2)^2 / b^2 = 1) →
  (Real.tan (45 * π / 360) = b / a) →
  (a / b = Real.sqrt 2 + 1) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_ratio_l3427_342773


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3427_342765

theorem fraction_subtraction : (3 + 5 + 7) / (2 + 4 + 6) - (2 - 4 + 6) / (3 - 5 + 7) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3427_342765


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3427_342735

theorem polynomial_remainder (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^4 - 8*x^3 + 15*x^2 + 12*x - 20
  let g : ℝ → ℝ := λ x => x - 2
  (f 2) = 16 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3427_342735


namespace NUMINAMATH_CALUDE_collinear_vectors_dot_product_l3427_342724

/-- Given two collinear vectors m and n, prove their dot product is -17/2 -/
theorem collinear_vectors_dot_product :
  ∀ (k : ℝ),
  let m : ℝ × ℝ := (2*k - 1, k)
  let n : ℝ × ℝ := (4, 1)
  (∃ (t : ℝ), m = t • n) →  -- collinearity condition
  m.1 * n.1 + m.2 * n.2 = -17/2 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_dot_product_l3427_342724


namespace NUMINAMATH_CALUDE_largest_five_digit_base5_l3427_342794

theorem largest_five_digit_base5 : 
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 := by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_base5_l3427_342794


namespace NUMINAMATH_CALUDE_toothpick_grid_15_8_l3427_342771

/-- Calculates the number of toothpicks needed for a rectangular grid with diagonals -/
def toothpick_count (height width : ℕ) : ℕ :=
  let horizontal := (height + 1) * width
  let vertical := (width + 1) * height
  let diagonal := height * width
  horizontal + vertical + diagonal

/-- Theorem stating the correct number of toothpicks for a 15x8 grid with diagonals -/
theorem toothpick_grid_15_8 :
  toothpick_count 15 8 = 383 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_grid_15_8_l3427_342771


namespace NUMINAMATH_CALUDE_asymptote_sum_l3427_342733

theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x^3 + A*x^2 + B*x + C = (x + 1)*(x - 3)*(x - 4)) →
  A + B + C = 11 := by
sorry

end NUMINAMATH_CALUDE_asymptote_sum_l3427_342733


namespace NUMINAMATH_CALUDE_set_intersection_equality_l3427_342779

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| ≥ 2}
def N : Set ℝ := {x | x^2 - 4*x ≥ 0}

-- State the theorem
theorem set_intersection_equality : 
  M ∩ N = {x | x ≤ -1 ∨ x ≥ 4} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l3427_342779


namespace NUMINAMATH_CALUDE_hexagon_division_divisible_by_three_l3427_342776

/-- A regular hexagon divided into congruent parallelograms -/
structure RegularHexagonDivision where
  /-- The number of congruent parallelograms -/
  N : ℕ
  /-- The hexagon is divided into N congruent parallelograms -/
  is_division : N > 0

/-- Theorem: The number of congruent parallelograms in a regular hexagon division is divisible by 3 -/
theorem hexagon_division_divisible_by_three (h : RegularHexagonDivision) : 
  ∃ k : ℕ, h.N = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_hexagon_division_divisible_by_three_l3427_342776


namespace NUMINAMATH_CALUDE_population_scientific_notation_l3427_342797

/-- Proves that 1.412 billion people is equal to 1.412 × 10^9 people. -/
theorem population_scientific_notation :
  (1.412 : ℝ) * 1000000000 = 1.412 * (10 : ℝ)^9 := by sorry

end NUMINAMATH_CALUDE_population_scientific_notation_l3427_342797


namespace NUMINAMATH_CALUDE_unique_three_digit_square_l3427_342729

theorem unique_three_digit_square (a b c : Nat) : 
  a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a^2 < 10 ∧ b^2 < 10 ∧ c^2 < 10 ∧
  (100*a + 10*b + c)^2 = 1000*100*a + 1000*10*b + 1000*c + 100*a + 10*b + c →
  a = 2 ∧ b = 3 ∧ c = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_three_digit_square_l3427_342729


namespace NUMINAMATH_CALUDE_eugene_payment_l3427_342761

def tshirt_cost : ℕ := 20
def pants_cost : ℕ := 80
def shoes_cost : ℕ := 150
def discount_rate : ℚ := 1/10

def tshirt_quantity : ℕ := 4
def pants_quantity : ℕ := 3
def shoes_quantity : ℕ := 2

def total_cost : ℕ := tshirt_cost * tshirt_quantity + pants_cost * pants_quantity + shoes_cost * shoes_quantity

def discounted_cost : ℚ := (1 - discount_rate) * total_cost

theorem eugene_payment : discounted_cost = 558 := by
  sorry

end NUMINAMATH_CALUDE_eugene_payment_l3427_342761


namespace NUMINAMATH_CALUDE_average_salary_non_officers_l3427_342704

/-- Prove that the average salary of non-officers is 110 Rs/month -/
theorem average_salary_non_officers (
  total_avg : ℝ) (officer_avg : ℝ) (num_officers : ℕ) (num_non_officers : ℕ)
  (h1 : total_avg = 120)
  (h2 : officer_avg = 420)
  (h3 : num_officers = 15)
  (h4 : num_non_officers = 450)
  : (((total_avg * (num_officers + num_non_officers : ℝ)) - 
     (officer_avg * num_officers)) / num_non_officers) = 110 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_non_officers_l3427_342704


namespace NUMINAMATH_CALUDE_integer_fraction_sum_l3427_342725

theorem integer_fraction_sum (n : ℤ) (h : n ≥ 8) :
  (∃ k : ℤ, n + 1 / (n - 7) = k) ↔ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_sum_l3427_342725


namespace NUMINAMATH_CALUDE_local_min_implies_a_eq_4_l3427_342781

/-- The function f(x) = x^3 - ax^2 + 4x - 8 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4*x - 8

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + 4

/-- Theorem: If f(x) has a local minimum at x = 2, then a = 4 -/
theorem local_min_implies_a_eq_4 (a : ℝ) :
  (∃ δ > 0, ∀ x, |x - 2| < δ → f a x ≥ f a 2) →
  f_deriv a 2 = 0 →
  a = 4 := by sorry

end NUMINAMATH_CALUDE_local_min_implies_a_eq_4_l3427_342781


namespace NUMINAMATH_CALUDE_coat_price_reduction_l3427_342793

theorem coat_price_reduction (original_price reduction_amount : ℝ) 
  (h1 : original_price = 500)
  (h2 : reduction_amount = 300) :
  (reduction_amount / original_price) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_reduction_l3427_342793


namespace NUMINAMATH_CALUDE_least_repeating_digits_seven_thirteenths_l3427_342714

/-- The least number of digits in a repeating block of 7/13 -/
def leastRepeatingDigits : ℕ := 6

/-- 7/13 is a repeating decimal -/
axiom seven_thirteenths_repeating : ∃ (n : ℕ) (k : ℕ+), (7 : ℚ) / 13 = ↑n / (10 ^ k.val - 1)

theorem least_repeating_digits_seven_thirteenths :
  leastRepeatingDigits = 6 ∧
  ∀ m : ℕ, m < leastRepeatingDigits → ¬∃ (n : ℕ) (k : ℕ+), (7 : ℚ) / 13 = ↑n / (10 ^ m - 1) :=
sorry

end NUMINAMATH_CALUDE_least_repeating_digits_seven_thirteenths_l3427_342714


namespace NUMINAMATH_CALUDE_variable_value_l3427_342755

theorem variable_value (x : ℝ) : 5 / (4 + 1 / x) = 1 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_variable_value_l3427_342755


namespace NUMINAMATH_CALUDE_equation_solutions_l3427_342708

theorem equation_solutions :
  (∀ x : ℝ, (x + 1)^2 - 16 = 0 ↔ x = 3 ∨ x = -5) ∧
  (∀ x : ℝ, -2 * (x - 1)^3 = 16 ↔ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3427_342708


namespace NUMINAMATH_CALUDE_boyGirlRatio_in_example_college_l3427_342784

/-- Represents the number of students in a college -/
structure College where
  total : ℕ
  girls : ℕ
  boys : ℕ
  total_eq : total = girls + boys

/-- The ratio of boys to girls in a college -/
def boyGirlRatio (c : College) : ℚ :=
  c.boys / c.girls

theorem boyGirlRatio_in_example_college :
  ∃ c : College, c.total = 600 ∧ c.girls = 200 ∧ boyGirlRatio c = 2 := by
  sorry

end NUMINAMATH_CALUDE_boyGirlRatio_in_example_college_l3427_342784


namespace NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l3427_342786

/-- The equation 9x^2 - 16y^2 = 0 represents a pair of straight lines -/
theorem equation_represents_pair_of_lines : 
  ∃ (m₁ m₂ : ℝ), ∀ (x y : ℝ), 9 * x^2 - 16 * y^2 = 0 ↔ (y = m₁ * x ∨ y = m₂ * x) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l3427_342786


namespace NUMINAMATH_CALUDE_triangle_tangent_circles_intersection_l3427_342772

/-- Triangle ABC with side lengths AB=8, BC=9, CA=10 -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB_length : dist A B = 8)
  (BC_length : dist B C = 9)
  (CA_length : dist C A = 10)

/-- Circle passing through a point and tangent to a line at another point -/
structure TangentCircle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (passes_through : ℝ × ℝ)
  (tangent_point : ℝ × ℝ)
  (tangent_line : ℝ × ℝ → ℝ × ℝ → Prop)

/-- The intersection point of two circles -/
def CircleIntersection (ω₁ ω₂ : TangentCircle) : ℝ × ℝ := sorry

/-- The theorem to be proved -/
theorem triangle_tangent_circles_intersection
  (abc : Triangle)
  (ω₁ : TangentCircle)
  (ω₂ : TangentCircle)
  (h₁ : ω₁.passes_through = abc.B ∧ ω₁.tangent_point = abc.A ∧ ω₁.tangent_line abc.A abc.C)
  (h₂ : ω₂.passes_through = abc.C ∧ ω₂.tangent_point = abc.A ∧ ω₂.tangent_line abc.A abc.B)
  (K : ℝ × ℝ)
  (hK : K = CircleIntersection ω₁ ω₂ ∧ K ≠ abc.A) :
  dist abc.A K = 10 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_circles_intersection_l3427_342772


namespace NUMINAMATH_CALUDE_sundae_cost_calculation_l3427_342748

-- Define constants for prices and discount thresholds
def scoop_price : ℚ := 2
def topping_a_price : ℚ := 0.5
def topping_b_price : ℚ := 0.75
def topping_c_price : ℚ := 0.6
def topping_d_price : ℚ := 0.8
def topping_e_price : ℚ := 0.9

def topping_a_discount_threshold : ℕ := 3
def topping_b_discount_threshold : ℕ := 2
def topping_c_discount_threshold : ℕ := 4

def topping_a_discount : ℚ := 0.3
def topping_b_discount : ℚ := 0.4
def topping_c_discount : ℚ := 0.5

-- Define the function to calculate the total cost
def calculate_sundae_cost (scoops topping_a topping_b topping_c topping_d topping_e : ℕ) : ℚ :=
  let ice_cream_cost := scoops * scoop_price
  let topping_a_cost := topping_a * topping_a_price - (topping_a / topping_a_discount_threshold) * topping_a_discount
  let topping_b_cost := topping_b * topping_b_price - (topping_b / topping_b_discount_threshold) * topping_b_discount
  let topping_c_cost := topping_c * topping_c_price - (topping_c / topping_c_discount_threshold) * topping_c_discount
  let topping_d_cost := topping_d * topping_d_price
  let topping_e_cost := topping_e * topping_e_price
  ice_cream_cost + topping_a_cost + topping_b_cost + topping_c_cost + topping_d_cost + topping_e_cost

-- Theorem statement
theorem sundae_cost_calculation :
  calculate_sundae_cost 3 5 3 7 2 1 = 16.25 := by
  sorry

end NUMINAMATH_CALUDE_sundae_cost_calculation_l3427_342748


namespace NUMINAMATH_CALUDE_cars_with_airbag_l3427_342796

theorem cars_with_airbag (total : ℕ) (power_windows : ℕ) (both : ℕ) (neither : ℕ) :
  total = 65 →
  power_windows = 30 →
  both = 12 →
  neither = 2 →
  total - neither = power_windows + (total - power_windows - neither) - both :=
by sorry

end NUMINAMATH_CALUDE_cars_with_airbag_l3427_342796


namespace NUMINAMATH_CALUDE_partial_week_salary_l3427_342762

/-- Calculates the salary for a partial work week --/
theorem partial_week_salary
  (usual_hours : ℝ)
  (worked_fraction : ℝ)
  (hourly_rate : ℝ)
  (h1 : usual_hours = 40)
  (h2 : worked_fraction = 4/5)
  (h3 : hourly_rate = 15) :
  worked_fraction * usual_hours * hourly_rate = 480 := by
  sorry

#check partial_week_salary

end NUMINAMATH_CALUDE_partial_week_salary_l3427_342762


namespace NUMINAMATH_CALUDE_sandy_spending_percentage_l3427_342770

/-- Given that Sandy took $310 for shopping and had $217 left after spending,
    prove that she spent 30% of her money. -/
theorem sandy_spending_percentage (money_taken : ℝ) (money_left : ℝ) : 
  money_taken = 310 → money_left = 217 → 
  (money_taken - money_left) / money_taken * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sandy_spending_percentage_l3427_342770


namespace NUMINAMATH_CALUDE_work_completion_time_l3427_342705

/-- The efficiency ratio between p and q -/
def efficiency_ratio : ℝ := 1.6

/-- The time taken by p and q working together -/
def combined_time : ℝ := 16

/-- The time taken by p working alone -/
def p_time : ℝ := 26

theorem work_completion_time :
  (efficiency_ratio * combined_time) / (efficiency_ratio + 1) = p_time := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3427_342705


namespace NUMINAMATH_CALUDE_train_clicks_theorem_l3427_342747

/-- Represents the number of clicks heard in 30 seconds for a train accelerating from 30 to 60 mph over 5 miles --/
def train_clicks : ℕ := 40

/-- Rail length in feet --/
def rail_length : ℝ := 50

/-- Initial speed in miles per hour --/
def initial_speed : ℝ := 30

/-- Final speed in miles per hour --/
def final_speed : ℝ := 60

/-- Acceleration distance in miles --/
def acceleration_distance : ℝ := 5

/-- Time period in seconds --/
def time_period : ℝ := 30

/-- Theorem stating that the number of clicks heard in 30 seconds is approximately 40 --/
theorem train_clicks_theorem : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |train_clicks - (((initial_speed + final_speed) / 2 * 5280 / 60) / rail_length * (time_period / 60))| < ε :=
sorry

end NUMINAMATH_CALUDE_train_clicks_theorem_l3427_342747


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_min_value_achieved_l3427_342717

theorem min_value_of_reciprocal_sum (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : 
  4/x + 1/y ≥ 9 := by
  sorry

theorem min_value_achieved (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 4/x₀ + 1/y₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_min_value_achieved_l3427_342717


namespace NUMINAMATH_CALUDE_angle_between_vectors_l3427_342742

/-- Given two unit vectors a and b in a real inner product space,
    prove that the angle between them is 2π/3 if |a-2b| = √7 -/
theorem angle_between_vectors 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b : V) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (h : ‖a - 2 • b‖ = Real.sqrt 7) :
  Real.arccos (inner a b) = 2 * Real.pi / 3 := by
  sorry

#check angle_between_vectors

end NUMINAMATH_CALUDE_angle_between_vectors_l3427_342742


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3427_342734

def A : Set ℝ := {y | ∃ x, y = 2^x ∧ 0 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3427_342734


namespace NUMINAMATH_CALUDE_job_completion_time_l3427_342728

/-- Given that m people can complete a job in d days, 
    prove that (m + r) people can complete the same job in md / (m + r) days. -/
theorem job_completion_time 
  (m d r : ℕ) (m_pos : m > 0) (d_pos : d > 0) (r_pos : r > 0) : 
  let n := (m * d) / (m + r)
  ∃ (W : ℝ), W > 0 ∧ W / (m * d) = W / ((m + r) * n) :=
sorry

end NUMINAMATH_CALUDE_job_completion_time_l3427_342728


namespace NUMINAMATH_CALUDE_min_value_expression_l3427_342778

theorem min_value_expression (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  (x^2 / (y - 1)) + (y^2 / (z - 1)) + (z^2 / (x - 1)) ≥ 12 ∧
  ((x^2 / (y - 1)) + (y^2 / (z - 1)) + (z^2 / (x - 1)) = 12 ↔ x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3427_342778


namespace NUMINAMATH_CALUDE_radius_of_third_circle_l3427_342756

structure Triangle :=
  (a b c : ℝ)

structure Circle :=
  (radius : ℝ)

def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b

def isInscribed (c : Circle) (t : Triangle) : Prop :=
  sorry

def isTangent (c1 c2 : Circle) (t : Triangle) : Prop :=
  sorry

theorem radius_of_third_circle (t : Triangle) (q1 q2 q3 : Circle) :
  t.a = 78 →
  t.b = 78 →
  t.c = 60 →
  isIsosceles t →
  isInscribed q1 t →
  isTangent q2 q1 t →
  isTangent q3 q2 t →
  q3.radius = 320 / 81 :=
sorry

end NUMINAMATH_CALUDE_radius_of_third_circle_l3427_342756


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l3427_342788

theorem quadratic_is_square_of_binomial (d : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + 80*x + d = (x + a)^2 + b^2) → d = 1600 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l3427_342788


namespace NUMINAMATH_CALUDE_school_boys_count_l3427_342732

theorem school_boys_count (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ) (other_count : ℕ) :
  muslim_percent = 44/100 →
  hindu_percent = 28/100 →
  sikh_percent = 10/100 →
  other_count = 117 →
  ∃ (total : ℕ), 
    (muslim_percent + hindu_percent + sikh_percent + (other_count : ℚ) / total = 1) ∧
    total = 650 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l3427_342732


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3427_342763

theorem solve_linear_equation (x : ℝ) :
  3 * x - 5 * x + 7 * x = 140 → x = 28 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3427_342763


namespace NUMINAMATH_CALUDE_equation_solution_l3427_342740

theorem equation_solution :
  let f (n : ℚ) := (2 - n) / (n + 1) + (2 * n - 4) / (2 - n)
  ∃ (n : ℚ), f n = 1 ∧ n = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3427_342740


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3427_342780

theorem max_value_sqrt_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 3) :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 3 →
    Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) + Real.sqrt (2 * c + 1) ≤
    Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1)) →
  Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) = 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3427_342780


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l3427_342736

-- Define the line l: x + y = 0
def line_l (x y : ℝ) : Prop := x + y = 0

-- Define the symmetric point of (-2, 0) with respect to line l
def symmetric_point (a b : ℝ) : Prop :=
  (b - 0) / (a + 2) = -1 ∧ (a - (-2)) / 2 + b / 2 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 2

-- Define the tangency condition
def is_tangent (x y : ℝ) : Prop := line_l x y ∧ circle_equation x y

-- Theorem statement
theorem circle_tangent_to_line :
  ∃ (x y : ℝ), is_tangent x y ∧
  ∃ (a b : ℝ), symmetric_point a b ∧
  (x - a)^2 + (y - b)^2 = ((a + b) / Real.sqrt 2)^2 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l3427_342736


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3427_342706

-- Define a and b as real numbers
variable (a b : ℝ)

-- Theorem stating that the sum of a and b is equal to a + b
theorem sum_of_a_and_b : (a + b) = (a + b) := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3427_342706


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3427_342744

/-- A quadratic function with positive leading coefficient -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0

/-- The roots of f(x) - x = 0 for a quadratic function f -/
structure QuadraticRoots (f : QuadraticFunction) where
  x₁ : ℝ
  x₂ : ℝ
  root_order : 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 / f.a

theorem quadratic_function_properties (f : QuadraticFunction) (roots : QuadraticRoots f) :
  (∀ x, 0 < x ∧ x < roots.x₁ → x < f.a * x^2 + f.b * x + f.c ∧ f.a * x^2 + f.b * x + f.c < roots.x₁) ∧
  roots.x₁ < roots.x₂ / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3427_342744


namespace NUMINAMATH_CALUDE_equivalent_propositions_l3427_342792

theorem equivalent_propositions (x y : ℝ) :
  (x > 1 ∧ y < -3 → x - y > 4) ↔ (x - y ≤ 4 → x ≤ 1 ∨ y ≥ -3) := by
sorry

end NUMINAMATH_CALUDE_equivalent_propositions_l3427_342792


namespace NUMINAMATH_CALUDE_correct_equation_transformation_l3427_342767

theorem correct_equation_transformation :
  ∀ x : ℚ, 3 * x = -7 ↔ x = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_transformation_l3427_342767


namespace NUMINAMATH_CALUDE_four_touching_circles_l3427_342739

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Returns true if the circle touches the line -/
def touches (c : Circle) (l : Line) : Prop :=
  sorry

/-- The main theorem stating that there are exactly 4 circles of a given radius
    touching two given lines -/
theorem four_touching_circles 
  (r : ℝ) 
  (l₁ l₂ : Line) 
  (h_r : r > 0) 
  (h_distinct : l₁ ≠ l₂) : 
  ∃! (s : Finset Circle), 
    s.card = 4 ∧ 
    ∀ c ∈ s, c.radius = r ∧ touches c l₁ ∧ touches c l₂ :=
sorry

end NUMINAMATH_CALUDE_four_touching_circles_l3427_342739


namespace NUMINAMATH_CALUDE_absolute_value_not_positive_l3427_342738

theorem absolute_value_not_positive (y : ℚ) : |6 * y - 8| ≤ 0 ↔ y = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_not_positive_l3427_342738


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l3427_342782

/-- Given a rectangular plot with the following properties:
  - The length is 60 meters
  - The length is 20 meters more than the breadth
  - The total cost of fencing is Rs. 5300
  Prove that the cost of fencing per meter is Rs. 26.50 -/
theorem fencing_cost_per_meter
  (length : ℝ)
  (breadth : ℝ)
  (total_cost : ℝ)
  (h1 : length = 60)
  (h2 : length = breadth + 20)
  (h3 : total_cost = 5300) :
  total_cost / (2 * length + 2 * breadth) = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l3427_342782


namespace NUMINAMATH_CALUDE_logo_shaded_area_l3427_342741

/-- Calculates the shaded area of a logo design with a rectangle and four tangent circles -/
theorem logo_shaded_area (length width : ℝ) (h1 : length = 30) (h2 : width = 15) : 
  let rectangle_area := length * width
  let circle_radius := width / 4
  let circle_area := π * circle_radius^2
  let total_circle_area := 4 * circle_area
  rectangle_area - total_circle_area = 450 - 56.25 * π := by
  sorry

end NUMINAMATH_CALUDE_logo_shaded_area_l3427_342741


namespace NUMINAMATH_CALUDE_triangle_gp_common_ratio_bounds_l3427_342766

/-- The common ratio of a geometric progression forming the sides of a triangle -/
def common_ratio_triangle_gp : Set ℝ :=
  {q : ℝ | (Real.sqrt 5 - 1) / 2 ≤ q ∧ q ≤ (Real.sqrt 5 + 1) / 2}

/-- Theorem: The common ratio of a geometric progression forming the sides of a triangle
    is bounded by (√5 - 1)/2 and (√5 + 1)/2 -/
theorem triangle_gp_common_ratio_bounds (a : ℝ) (q : ℝ) 
    (h_a : a > 0) (h_q : q ≥ 1) 
    (h_triangle : a + a*q > a*q^2 ∧ a + a*q^2 > a*q ∧ a*q + a*q^2 > a) :
  q ∈ common_ratio_triangle_gp := by
  sorry

end NUMINAMATH_CALUDE_triangle_gp_common_ratio_bounds_l3427_342766


namespace NUMINAMATH_CALUDE_ten_candies_distribution_l3427_342750

/-- The number of ways to distribute n candies over days, with at least one candy per day -/
def candy_distribution (n : ℕ) : ℕ := 2^(n - 1)

/-- Theorem: The number of ways to distribute 10 candies over days, with at least one candy per day, is 512 -/
theorem ten_candies_distribution : candy_distribution 10 = 512 := by
  sorry

end NUMINAMATH_CALUDE_ten_candies_distribution_l3427_342750


namespace NUMINAMATH_CALUDE_coords_wrt_origin_invariant_point_P_coords_l3427_342798

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin of the Cartesian coordinate system -/
def origin : Point := ⟨0, 0⟩

/-- Coordinates of a point with respect to the origin -/
def coordsWrtOrigin (p : Point) : ℝ × ℝ := (p.x, p.y)

theorem coords_wrt_origin_invariant (p : Point) :
  coordsWrtOrigin p = (p.x, p.y) := by sorry

theorem point_P_coords :
  let P : Point := ⟨-1, -3⟩
  coordsWrtOrigin P = (-1, -3) := by sorry

end NUMINAMATH_CALUDE_coords_wrt_origin_invariant_point_P_coords_l3427_342798


namespace NUMINAMATH_CALUDE_min_value_f_over_x_range_of_a_l3427_342760

-- Part 1
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 1 + a

theorem min_value_f_over_x (x : ℝ) (hx : x > 0) :
  ∃ (y : ℝ), y = (f 2 x) / x ∧ ∀ (z : ℝ), z > 0 → (f 2 z) / z ≥ y ∧ y = -2 :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f a x ≤ a) ↔ a ∈ Set.Ici (3/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_f_over_x_range_of_a_l3427_342760


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_270_l3427_342751

theorem rectangle_area (square_area : ℝ) (rectangle_length : ℝ) : ℝ :=
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_breadth := (3 / 5) * circle_radius
  rectangle_length * rectangle_breadth

theorem rectangle_area_is_270 :
  rectangle_area 2025 10 = 270 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_270_l3427_342751


namespace NUMINAMATH_CALUDE_second_square_weight_l3427_342711

/-- Represents a square piece of metal -/
structure MetalSquare where
  side_length : ℝ
  weight : ℝ

/-- The density of the metal in ounces per square inch -/
def metal_density : ℝ := 0.5

theorem second_square_weight
  (first_square : MetalSquare)
  (h1 : first_square.side_length = 4)
  (h2 : first_square.weight = 8)
  (second_square : MetalSquare)
  (h3 : second_square.side_length = 7) :
  second_square.weight = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_second_square_weight_l3427_342711


namespace NUMINAMATH_CALUDE_equidistant_point_on_leg_l3427_342774

/-- 
Given a right triangle with legs 240 and 320 rods, and hypotenuse 400 rods,
prove that the point on the longer leg equidistant from the other two vertices
is 95 rods from the right angle.
-/
theorem equidistant_point_on_leg (a b c x : ℝ) : 
  a = 240 → b = 320 → c = 400 → 
  a^2 + b^2 = c^2 →
  x^2 + a^2 = (b - x)^2 + b^2 →
  x = 95 := by
sorry

end NUMINAMATH_CALUDE_equidistant_point_on_leg_l3427_342774


namespace NUMINAMATH_CALUDE_tangent_line_problem_l3427_342700

theorem tangent_line_problem (f : ℝ → ℝ) (h : ∀ x y, x = 2 ∧ f x = y → 2*x + y - 3 = 0) :
  f 2 + deriv f 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l3427_342700


namespace NUMINAMATH_CALUDE_balloon_height_proof_l3427_342707

/-- Calculates the maximum height a helium balloon can fly given budget and costs. -/
def balloon_max_height (total_budget : ℚ) (sheet_cost rope_cost propane_cost : ℚ) 
  (helium_cost_per_oz : ℚ) (height_per_oz : ℚ) : ℚ :=
  let remaining_budget := total_budget - (sheet_cost + rope_cost + propane_cost)
  let helium_oz := remaining_budget / helium_cost_per_oz
  helium_oz * height_per_oz

/-- The maximum height of the balloon is 9,492 feet given the specified conditions. -/
theorem balloon_height_proof : 
  balloon_max_height 200 42 18 14 (3/2) 113 = 9492 := by
  sorry

end NUMINAMATH_CALUDE_balloon_height_proof_l3427_342707


namespace NUMINAMATH_CALUDE_max_value_on_circle_l3427_342715

theorem max_value_on_circle (x y : ℝ) (h : x^2 + y^2 = 9) :
  ∃ (M : ℝ), M = 9 ∧ ∀ (a b : ℝ), a^2 + b^2 = 9 → 3 * |a| + 2 * |b| ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l3427_342715


namespace NUMINAMATH_CALUDE_f_properties_l3427_342789

def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

def M : Set ℝ := {x : ℝ | f x ≥ 3}

theorem f_properties :
  (M = {x : ℝ | x ≤ -1/2 ∨ x ≥ 2}) ∧
  (∀ a ∈ M, ∀ x : ℝ, |x + a| + |x - 1/a| ≥ 5/2) := by sorry

end NUMINAMATH_CALUDE_f_properties_l3427_342789


namespace NUMINAMATH_CALUDE_original_earnings_before_raise_l3427_342719

theorem original_earnings_before_raise (new_earnings : ℝ) (increase_percentage : ℝ) :
  new_earnings = 75 ∧ increase_percentage = 0.25 →
  ∃ original_earnings : ℝ,
    original_earnings * (1 + increase_percentage) = new_earnings ∧
    original_earnings = 60 :=
by sorry

end NUMINAMATH_CALUDE_original_earnings_before_raise_l3427_342719


namespace NUMINAMATH_CALUDE_diving_survey_contradiction_l3427_342749

structure Survey where
  population : ℕ
  sample : ℕ
  topic : String

def is_sampling_survey (s : Survey) : Prop :=
  s.sample < s.population

theorem diving_survey_contradiction (s : Survey) 
  (h1 : s.population = 2000)
  (h2 : s.sample = 150)
  (h3 : s.topic = "interest in diving")
  (h4 : is_sampling_survey s) : 
  s.sample ≠ 150 := by
  sorry

end NUMINAMATH_CALUDE_diving_survey_contradiction_l3427_342749


namespace NUMINAMATH_CALUDE_differential_system_properties_l3427_342703

-- Define the system of differential equations
def system_ode (u : ℝ → ℝ) (x y : ℝ → ℝ) : Prop :=
  ∀ t, deriv x t = -2 * y t + u t ∧ deriv y t = -2 * x t + u t

-- Define the theorem
theorem differential_system_properties
  (u : ℝ → ℝ) (x y : ℝ → ℝ) (x₀ y₀ : ℝ)
  (h_cont : Continuous u)
  (h_system : system_ode u x y)
  (h_init : x 0 = x₀ ∧ y 0 = y₀) :
  (x₀ ≠ y₀ → ∀ t, x t - y t ≠ 0) ∧
  (x₀ = y₀ → ∀ T > 0, ∃ u : ℝ → ℝ, Continuous u ∧ x T = 0 ∧ y T = 0) :=
sorry

end NUMINAMATH_CALUDE_differential_system_properties_l3427_342703


namespace NUMINAMATH_CALUDE_second_month_sale_l3427_342712

def sale_month1 : ℕ := 7435
def sale_month3 : ℕ := 7855
def sale_month4 : ℕ := 8230
def sale_month5 : ℕ := 7562
def sale_month6 : ℕ := 5991
def average_sale : ℕ := 7500
def num_months : ℕ := 6

theorem second_month_sale :
  ∃ (sale_month2 : ℕ),
    (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / num_months = average_sale ∧
    sale_month2 = 7927 :=
by sorry

end NUMINAMATH_CALUDE_second_month_sale_l3427_342712


namespace NUMINAMATH_CALUDE_village_population_l3427_342787

theorem village_population (p : ℝ) : p = 939 ↔ 0.92 * p = 1.15 * p + 216 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3427_342787


namespace NUMINAMATH_CALUDE_quadratic_condition_for_x_equals_one_l3427_342701

theorem quadratic_condition_for_x_equals_one :
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧
  ¬(∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_condition_for_x_equals_one_l3427_342701


namespace NUMINAMATH_CALUDE_johns_watermelon_weight_l3427_342720

theorem johns_watermelon_weight (michael_weight : ℕ) (clay_factor : ℕ) (john_factor : ℚ) :
  michael_weight = 8 →
  clay_factor = 3 →
  john_factor = 1/2 →
  (↑michael_weight * ↑clay_factor * john_factor : ℚ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_johns_watermelon_weight_l3427_342720


namespace NUMINAMATH_CALUDE_athlete_difference_ultimate_fitness_camp_problem_l3427_342753

/-- The difference in the number of athletes at Ultimate Fitness Camp over two nights -/
theorem athlete_difference (initial_athletes : ℕ) 
                           (leaving_rate : ℕ) (leaving_hours : ℕ)
                           (arriving_rate : ℕ) (arriving_hours : ℕ) : ℕ :=
  let athletes_leaving := leaving_rate * leaving_hours
  let athletes_remaining := initial_athletes - athletes_leaving
  let athletes_arriving := arriving_rate * arriving_hours
  let final_athletes := athletes_remaining + athletes_arriving
  initial_athletes - final_athletes

/-- The specific case of the Ultimate Fitness Camp problem -/
theorem ultimate_fitness_camp_problem : 
  athlete_difference 300 28 4 15 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_athlete_difference_ultimate_fitness_camp_problem_l3427_342753


namespace NUMINAMATH_CALUDE_collinear_points_m_value_l3427_342757

/-- Given a line containing points (2, 9), (10, m), and (25, 4), prove that m = 167/23 -/
theorem collinear_points_m_value : 
  ∀ m : ℚ, 
  (∃ (line : Set (ℚ × ℚ)), 
    (2, 9) ∈ line ∧ 
    (10, m) ∈ line ∧ 
    (25, 4) ∈ line ∧ 
    (∀ (x y z : ℚ × ℚ), x ∈ line → y ∈ line → z ∈ line → 
      (z.2 - y.2) * (y.1 - x.1) = (y.2 - x.2) * (z.1 - y.1))) →
  m = 167 / 23 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_m_value_l3427_342757


namespace NUMINAMATH_CALUDE_zmod_is_field_l3427_342799

/-- Given a prime number p, (ℤ/pℤ, +, ×, 0, 1) is a commutative field -/
theorem zmod_is_field (p : ℕ) (hp : Prime p) : Field (ZMod p) := by sorry

end NUMINAMATH_CALUDE_zmod_is_field_l3427_342799


namespace NUMINAMATH_CALUDE_f_has_max_and_min_l3427_342764

/-- A cubic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m+6)*x + 1

/-- The derivative of f with respect to x -/
def f_derivative (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*m*x + (m+6)

/-- Theorem stating the condition for f to have both a maximum and a minimum -/
theorem f_has_max_and_min (m : ℝ) : 
  (∃ (a b : ℝ), ∀ x, f m x ≤ f m a ∧ f m x ≥ f m b) ↔ m < -3 ∨ m > 6 := by
  sorry

end NUMINAMATH_CALUDE_f_has_max_and_min_l3427_342764


namespace NUMINAMATH_CALUDE_students_playing_sports_l3427_342775

theorem students_playing_sports (basketball cricket both : ℕ) 
  (hb : basketball = 7)
  (hc : cricket = 8)
  (hboth : both = 3) :
  basketball + cricket - both = 12 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_sports_l3427_342775


namespace NUMINAMATH_CALUDE_quadratic_equation_constant_term_l3427_342759

theorem quadratic_equation_constant_term (m : ℝ) : 
  (∀ x, (m - 2) * x^2 + 3 * x + m^2 - 4 = 0) → 
  m^2 - 4 = 0 → 
  m - 2 ≠ 0 → 
  m = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_constant_term_l3427_342759


namespace NUMINAMATH_CALUDE_forces_equilibrium_l3427_342769

/-- A 2D vector representing a force -/
structure Force where
  x : ℝ
  y : ℝ

/-- Add two forces -/
def Force.add (f g : Force) : Force :=
  ⟨f.x + g.x, f.y + g.y⟩

instance : Add Force :=
  ⟨Force.add⟩

/-- The zero force -/
def Force.zero : Force :=
  ⟨0, 0⟩

instance : Zero Force :=
  ⟨Force.zero⟩

theorem forces_equilibrium (f₁ f₂ f₃ f₄ : Force) 
    (h₁ : f₁ = ⟨-2, -1⟩)
    (h₂ : f₂ = ⟨-3, 2⟩)
    (h₃ : f₃ = ⟨4, -3⟩)
    (h₄ : f₄ = ⟨1, 2⟩) :
    f₁ + f₂ + f₃ + f₄ = 0 := by
  sorry

end NUMINAMATH_CALUDE_forces_equilibrium_l3427_342769


namespace NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l3427_342754

/-- Represents a seven-digit number in the form 3b5n678 -/
def sevenDigitNumber (b n : ℕ) : ℕ := 3000000 + 100000 * b + 50000 + 10000 * n + 678

/-- Checks if a number is divisible by 11 -/
def isDivisibleBy11 (num : ℕ) : Prop := ∃ k : ℕ, num = 11 * k

/-- b and n are single digits -/
def isSingleDigit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

theorem seven_digit_divisible_by_11 :
  ∃ b n : ℕ, isSingleDigit b ∧ isSingleDigit n ∧ 
  isDivisibleBy11 (sevenDigitNumber b n) ∧ 
  b = 4 ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l3427_342754


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3427_342709

theorem inequality_solution_set (x : ℝ) :
  (x^2 + 1) / ((x - 3) * (x + 2)) ≥ 0 ↔ x ≤ -2 ∨ x ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3427_342709


namespace NUMINAMATH_CALUDE_robins_hair_length_l3427_342777

/-- Calculates the final hair length after growth and cutting -/
def finalHairLength (initial growth cut : ℝ) : ℝ :=
  initial + growth - cut

/-- Theorem stating that given the initial conditions, the final hair length is 2 inches -/
theorem robins_hair_length :
  finalHairLength 14 8 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robins_hair_length_l3427_342777


namespace NUMINAMATH_CALUDE_circle_center_correct_l3427_342721

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 6*y = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, -3)

/-- Theorem: The center of the circle defined by circle_equation is circle_center -/
theorem circle_center_correct :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 13 :=
sorry

end NUMINAMATH_CALUDE_circle_center_correct_l3427_342721


namespace NUMINAMATH_CALUDE_find_M_l3427_342722

theorem find_M : ∃ M : ℕ+, (15^2 * 25^2 : ℕ) = 5^2 * M^2 ∧ M = 375 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l3427_342722


namespace NUMINAMATH_CALUDE_representation_2015_l3427_342702

theorem representation_2015 : ∃ (a b c : ℤ), 
  a + b + c = 2015 ∧ 
  Nat.Prime a.natAbs ∧ 
  ∃ (k : ℤ), b = 3 * k ∧
  400 < c ∧ c < 500 ∧
  ¬∃ (m : ℤ), c = 3 * m := by
  sorry

end NUMINAMATH_CALUDE_representation_2015_l3427_342702


namespace NUMINAMATH_CALUDE_art_students_count_l3427_342723

theorem art_students_count (total : ℕ) (music : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : music = 20)
  (h3 : both = 10)
  (h4 : neither = 470) :
  ∃ art : ℕ, art = 20 ∧ 
    total = (music - both) + (art - both) + both + neither :=
by sorry

end NUMINAMATH_CALUDE_art_students_count_l3427_342723


namespace NUMINAMATH_CALUDE_ellipse_properties_l3427_342785

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (1, 0)

-- Define the line l with slope m passing through the right focus
def line_l (m : ℝ) (x y : ℝ) : Prop := y = m * (x - 1)

-- Define the area of a triangle given three points
def triangle_area (A B P : ℝ × ℝ) : ℝ := sorry

-- Define the area of the incircle of a triangle
def incircle_area (A B F : ℝ × ℝ) : ℝ := sorry

theorem ellipse_properties :
  ∃ (P₁ P₂ : ℝ × ℝ),
    ellipse P₁.1 P₁.2 ∧ 
    ellipse P₂.1 P₂.2 ∧
    P₁ ≠ P₂ ∧
    (∀ (A B : ℝ × ℝ), 
      ellipse A.1 A.2 ∧ 
      ellipse B.1 B.2 ∧ 
      line_l 1 A.1 A.2 ∧ 
      line_l 1 B.1 B.2 →
      triangle_area A B P₁ = (2 * Real.sqrt 5 - 2) / 3 ∧
      triangle_area A B P₂ = (2 * Real.sqrt 5 - 2) / 3) ∧
    (∀ (P : ℝ × ℝ),
      ellipse P.1 P.2 ∧ 
      P ≠ P₁ ∧ 
      P ≠ P₂ →
      triangle_area A B P ≠ (2 * Real.sqrt 5 - 2) / 3) ∧
    (∃ (A B : ℝ × ℝ) (m : ℝ),
      ellipse A.1 A.2 ∧
      ellipse B.1 B.2 ∧
      line_l m A.1 A.2 ∧
      line_l m B.1 B.2 ∧
      incircle_area A B (-1, 0) = π / 8 ∧
      (∀ (C D : ℝ × ℝ) (n : ℝ),
        ellipse C.1 C.2 ∧
        ellipse D.1 D.2 ∧
        line_l n C.1 C.2 ∧
        line_l n D.1 D.2 →
        incircle_area C D (-1, 0) ≤ π / 8)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3427_342785


namespace NUMINAMATH_CALUDE_min_phi_for_even_shifted_sine_l3427_342795

/-- Given a function f and its left-shifted version g, proves that the minimum φ for g to be even is π/10 -/
theorem min_phi_for_even_shifted_sine (φ : ℝ) (f g : ℝ → ℝ) : 
  (φ > 0) →
  (∀ x, f x = 2 * Real.sin (2 * x + φ)) →
  (∀ x, g x = f (x + π/5)) →
  (∀ x, g x = g (-x)) →
  (∃ k : ℤ, φ = k * π + π/10) →
  φ ≥ π/10 := by
sorry

end NUMINAMATH_CALUDE_min_phi_for_even_shifted_sine_l3427_342795


namespace NUMINAMATH_CALUDE_raphael_manny_ratio_l3427_342752

/-- Represents the number of lasagna pieces each person eats -/
structure LasagnaPieces where
  manny : ℕ
  lisa : ℕ
  raphael : ℕ
  aaron : ℕ
  kai : ℕ

/-- The properties of the lasagna distribution -/
def LasagnaDistribution (p : LasagnaPieces) : Prop :=
  p.manny = 1 ∧
  p.aaron = 0 ∧
  p.kai = 2 * p.manny ∧
  p.lisa = 2 + (p.raphael - 1) ∧
  p.manny + p.lisa + p.raphael + p.aaron + p.kai = 6

theorem raphael_manny_ratio (p : LasagnaPieces) 
  (h : LasagnaDistribution p) : p.raphael = p.manny := by
  sorry

end NUMINAMATH_CALUDE_raphael_manny_ratio_l3427_342752


namespace NUMINAMATH_CALUDE_largest_class_size_l3427_342731

/-- Represents the number of students in each class of a school --/
structure School :=
  (largest_class : ℕ)

/-- Calculates the total number of students in the school --/
def total_students (s : School) : ℕ :=
  s.largest_class + (s.largest_class - 2) + (s.largest_class - 4) + (s.largest_class - 6) + (s.largest_class - 8)

/-- Theorem stating that a school with 5 classes, where each class has 2 students less than the previous class, 
    and a total of 105 students, has 25 students in the largest class --/
theorem largest_class_size :
  ∃ (s : School), total_students s = 105 ∧ s.largest_class = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_class_size_l3427_342731


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3427_342726

-- Define the sets P and Q
def P : Set ℝ := {x | x ≤ 0 ∨ x > 3}
def Q : Set ℝ := {0, 1, 2, 3}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.compl P) ∩ Q = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3427_342726


namespace NUMINAMATH_CALUDE_stating_count_five_digit_divisible_by_12_is_72000_l3427_342730

/-- 
A function that counts the number of positive five-digit integers divisible by 12.
-/
def count_five_digit_divisible_by_12 : ℕ :=
  sorry

/-- 
Theorem stating that the count of positive five-digit integers divisible by 12 is 72000.
-/
theorem count_five_digit_divisible_by_12_is_72000 : 
  count_five_digit_divisible_by_12 = 72000 :=
by sorry

end NUMINAMATH_CALUDE_stating_count_five_digit_divisible_by_12_is_72000_l3427_342730


namespace NUMINAMATH_CALUDE_real_part_reciprocal_l3427_342783

/-- For a nonreal complex number z with |z| = 2, 
    the real part of 1/(2-z) is (2-x)/(8-4x+x^2), where x is the real part of z -/
theorem real_part_reciprocal (z : ℂ) (x : ℝ) (h1 : z.im ≠ 0) (h2 : Complex.abs z = 2) 
  (h3 : z.re = x) : 
  Complex.re (1 / (2 - z)) = (2 - x) / (8 - 4*x + x^2) := by
  sorry

end NUMINAMATH_CALUDE_real_part_reciprocal_l3427_342783


namespace NUMINAMATH_CALUDE_handshake_count_l3427_342758

/-- Represents a basketball game setup with two teams and referees -/
structure BasketballGame where
  team_size : Nat
  coach_per_team : Nat
  referee_count : Nat

/-- Calculates the total number of handshakes in a basketball game -/
def total_handshakes (game : BasketballGame) : Nat :=
  let inter_team_handshakes := game.team_size * game.team_size
  let total_team_members := game.team_size + game.coach_per_team
  let intra_team_handshakes := 2 * (total_team_members.choose 2)
  let team_referee_handshakes := 2 * total_team_members * game.referee_count
  let referee_handshakes := game.referee_count.choose 2
  inter_team_handshakes + intra_team_handshakes + team_referee_handshakes + referee_handshakes

/-- The main theorem stating the total number of handshakes in the given game setup -/
theorem handshake_count :
  let game : BasketballGame := {
    team_size := 6
    coach_per_team := 1
    referee_count := 2
  }
  total_handshakes game = 107 := by
  sorry


end NUMINAMATH_CALUDE_handshake_count_l3427_342758


namespace NUMINAMATH_CALUDE_ice_cream_consumption_l3427_342737

theorem ice_cream_consumption (friday_amount saturday_amount : Real) 
  (h1 : friday_amount = 3.25)
  (h2 : saturday_amount = 0.25) :
  friday_amount + saturday_amount = 3.50 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_consumption_l3427_342737


namespace NUMINAMATH_CALUDE_rachel_family_age_ratio_l3427_342745

/-- The ratio of Rachel's grandfather's age to Rachel's age -/
def age_ratio (rachel_age grandfather_age : ℕ) : ℚ :=
  grandfather_age / rachel_age

theorem rachel_family_age_ratio :
  ∀ (rachel_age grandfather_age mother_age father_age : ℕ),
    rachel_age = 12 →
    mother_age = grandfather_age / 2 →
    father_age = mother_age + 5 →
    father_age + (25 - rachel_age) = 60 →
    age_ratio rachel_age grandfather_age = 7 / 1 := by
  sorry

end NUMINAMATH_CALUDE_rachel_family_age_ratio_l3427_342745


namespace NUMINAMATH_CALUDE_pipe_cut_theorem_l3427_342790

theorem pipe_cut_theorem (total_length : ℝ) (difference : ℝ) (shorter_piece : ℝ) : 
  total_length = 68 →
  difference = 12 →
  shorter_piece + (shorter_piece + difference) = total_length →
  shorter_piece = 28 := by
  sorry

end NUMINAMATH_CALUDE_pipe_cut_theorem_l3427_342790


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l3427_342727

/-- The volume of a cone formed by rolling up a half-sector of a circle -/
theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let slant_height := r
  let base_circumference := r * π
  let base_radius := base_circumference / (2 * π)
  let cone_height := Real.sqrt (slant_height ^ 2 - base_radius ^ 2)
  (1 / 3) * π * base_radius ^ 2 * cone_height = 9 * π * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l3427_342727


namespace NUMINAMATH_CALUDE_luke_money_in_january_l3427_342746

/-- The amount of money Luke had in January -/
def initial_amount : ℕ := sorry

/-- The amount Luke spent -/
def spent : ℕ := 11

/-- The amount Luke received from his mom -/
def received : ℕ := 21

/-- The amount Luke has now -/
def current_amount : ℕ := 58

theorem luke_money_in_january :
  initial_amount = 48 :=
by
  have h : initial_amount - spent + received = current_amount := by sorry
  sorry

end NUMINAMATH_CALUDE_luke_money_in_january_l3427_342746


namespace NUMINAMATH_CALUDE_marble_difference_prove_marble_difference_l3427_342743

/-- The difference in marbles between Ed and Doug after a series of events -/
theorem marble_difference : ℤ → Prop :=
  fun initial_difference =>
    ∀ (doug_initial : ℤ) (doug_lost : ℤ) (susan_found : ℤ),
      initial_difference = 22 →
      doug_lost = 8 →
      susan_found = 5 →
      (doug_initial + initial_difference + susan_found) - (doug_initial - doug_lost) = 35

/-- Proof of the marble difference theorem -/
theorem prove_marble_difference : marble_difference 22 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_prove_marble_difference_l3427_342743


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_true_l3427_342768

theorem quadratic_inequality_always_true (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + (a + 1) * x + 1 ≥ 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_true_l3427_342768


namespace NUMINAMATH_CALUDE_relationship_abc_l3427_342710

theorem relationship_abc : ∀ (a b c : ℕ),
  a = 2^12 → b = 3^8 → c = 7^4 → b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l3427_342710


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3427_342718

-- Define the complex number z
def z : ℂ := (2 - Complex.I) * (1 - Complex.I)

-- Theorem statement
theorem z_in_fourth_quadrant :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3427_342718
