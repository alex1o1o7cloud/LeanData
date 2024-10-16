import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l3270_327002

theorem arithmetic_evaluation : 23 - |(-6)| - 23 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l3270_327002


namespace NUMINAMATH_CALUDE_net_population_increase_is_154800_l3270_327053

/-- Represents the number of seconds in an hour -/
def secondsPerHour : ℕ := 3600

/-- Represents the number of hours in a day -/
def hoursPerDay : ℕ := 24

/-- Represents the number of peak hours in a day -/
def peakHours : ℕ := 12

/-- Represents the number of off-peak hours in a day -/
def offPeakHours : ℕ := 12

/-- Represents the birth rate during peak hours (people per 2 seconds) -/
def peakBirthRate : ℕ := 7

/-- Represents the birth rate during off-peak hours (people per 2 seconds) -/
def offPeakBirthRate : ℕ := 3

/-- Represents the death rate during peak hours (people per 2 seconds) -/
def peakDeathRate : ℕ := 1

/-- Represents the death rate during off-peak hours (people per 2 seconds) -/
def offPeakDeathRate : ℕ := 2

/-- Represents the net migration rate during peak hours (people entering per 4 seconds) -/
def peakMigrationRate : ℕ := 1

/-- Represents the net migration rate during off-peak hours (people leaving per 6 seconds) -/
def offPeakMigrationRate : ℕ := 1

/-- Calculates the net population increase over a 24-hour period -/
def netPopulationIncrease : ℕ :=
  let peakIncrease := (peakBirthRate * 30 * secondsPerHour * peakHours) -
                      (peakDeathRate * 30 * secondsPerHour * peakHours) +
                      (peakMigrationRate * 15 * secondsPerHour * peakHours)
  let offPeakIncrease := (offPeakBirthRate * 30 * secondsPerHour * offPeakHours) -
                         (offPeakDeathRate * 30 * secondsPerHour * offPeakHours) -
                         (offPeakMigrationRate * 10 * secondsPerHour * offPeakHours)
  peakIncrease + offPeakIncrease

theorem net_population_increase_is_154800 : netPopulationIncrease = 154800 := by
  sorry

end NUMINAMATH_CALUDE_net_population_increase_is_154800_l3270_327053


namespace NUMINAMATH_CALUDE_quadratic_inequality_necessary_not_sufficient_l3270_327084

theorem quadratic_inequality_necessary_not_sufficient :
  (∀ x : ℝ, x > 2 → x^2 + 2*x - 8 > 0) ∧
  (∃ x : ℝ, x^2 + 2*x - 8 > 0 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_necessary_not_sufficient_l3270_327084


namespace NUMINAMATH_CALUDE_johns_candy_store_spending_l3270_327054

theorem johns_candy_store_spending (allowance : ℝ) (arcade_fraction : ℝ) (toy_store_fraction : ℝ)
  (h1 : allowance = 3.375)
  (h2 : arcade_fraction = 3/5)
  (h3 : toy_store_fraction = 1/3) :
  allowance * (1 - arcade_fraction) * (1 - toy_store_fraction) = 0.90 := by
  sorry

end NUMINAMATH_CALUDE_johns_candy_store_spending_l3270_327054


namespace NUMINAMATH_CALUDE_max_value_of_ab_l3270_327065

theorem max_value_of_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 8) :
  ∃ (m : ℝ), m = 8 ∧ ∀ x y, x > 0 → y > 0 → x + 2*y = 8 → x*y ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_ab_l3270_327065


namespace NUMINAMATH_CALUDE_no_rational_roots_l3270_327047

theorem no_rational_roots (p q : ℤ) 
  (hp : p % 3 = 2) 
  (hq : q % 3 = 2) : 
  ¬ ∃ (r : ℚ), r^2 + p * r + q = 0 := by
sorry

end NUMINAMATH_CALUDE_no_rational_roots_l3270_327047


namespace NUMINAMATH_CALUDE_percentage_calculation_l3270_327049

theorem percentage_calculation : (200 / 50) * 100 = 400 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3270_327049


namespace NUMINAMATH_CALUDE_special_polygon_area_l3270_327099

/-- A polygon with special properties -/
structure SpecialPolygon where
  sides : ℕ
  perimeter : ℝ
  is_perpendicular : Bool
  is_equal_length : Bool

/-- The area of a special polygon -/
def area (p : SpecialPolygon) : ℝ := sorry

/-- Theorem: The area of a special polygon with 36 sides and perimeter 72 is 144 -/
theorem special_polygon_area :
  ∀ (p : SpecialPolygon),
    p.sides = 36 ∧
    p.perimeter = 72 ∧
    p.is_perpendicular ∧
    p.is_equal_length →
    area p = 144 := by
  sorry

end NUMINAMATH_CALUDE_special_polygon_area_l3270_327099


namespace NUMINAMATH_CALUDE_vector_equation_y_axis_l3270_327010

/-- Given points O, A, and B in the plane, and a vector equation for OP,
    prove that if P is on the y-axis, then m = 2/3 -/
theorem vector_equation_y_axis (O A B P : ℝ × ℝ) (m : ℝ) :
  O = (0, 0) →
  A = (-1, 3) →
  B = (2, -4) →
  P.1 = 0 →
  P = (2 * A.1 + m * (B.1 - A.1), 2 * A.2 + m * (B.2 - A.2)) →
  m = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_y_axis_l3270_327010


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3270_327052

theorem complex_fraction_equality : 2 + 1 / (2 + 1 / (2 + 1 / 3)) = 41 / 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3270_327052


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3270_327091

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + 3*b - 1 = 0) :
  (2/a + 3/b) ≥ 25 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2*a₀ + 3*b₀ - 1 = 0 ∧ 2/a₀ + 3/b₀ = 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3270_327091


namespace NUMINAMATH_CALUDE_max_swaps_is_19_l3270_327011

/-- A permutation of the numbers 1 to 20 -/
def Permutation := Fin 20 → Fin 20

/-- The identity permutation -/
def id_perm : Permutation := fun i => i

/-- A swap operation on a permutation -/
def swap (p : Permutation) (i j : Fin 20) : Permutation :=
  fun k => if k = i then p j else if k = j then p i else p k

/-- The minimum number of swaps needed to transform a permutation into the identity permutation -/
def min_swaps (p : Permutation) : ℕ := sorry

/-- Theorem: The maximum number of swaps needed for any permutation is 19 -/
theorem max_swaps_is_19 :
  ∃ (p : Permutation), min_swaps p = 19 ∧ 
  ∀ (q : Permutation), min_swaps q ≤ 19 :=
sorry

end NUMINAMATH_CALUDE_max_swaps_is_19_l3270_327011


namespace NUMINAMATH_CALUDE_rectangular_prism_ratios_l3270_327035

/-- A rectangular prism with edges a, b, c, and free surface ratios p, q, r -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  p : ℝ
  q : ℝ
  r : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  p_pos : 0 < p
  q_pos : 0 < q
  r_pos : 0 < r

/-- The theorem stating the edge ratios and conditions for p, q, r -/
theorem rectangular_prism_ratios (prism : RectangularPrism) :
  (prism.a : ℝ) / (prism.b : ℝ) = (2 * prism.p - 3 * prism.q + 2 * prism.r) / (-3 * prism.p + 2 * prism.q + 2 * prism.r) ∧
  (prism.b : ℝ) / (prism.c : ℝ) = (2 * prism.p + 2 * prism.q - 3 * prism.r) / (2 * prism.p - 3 * prism.q + 2 * prism.r) ∧
  (prism.c : ℝ) / (prism.a : ℝ) = (-3 * prism.p + 2 * prism.q + 2 * prism.r) / (2 * prism.p + 2 * prism.q - 3 * prism.r) ∧
  2 * prism.p + 2 * prism.r > 3 * prism.q ∧
  2 * prism.p + 2 * prism.q > 3 * prism.r ∧
  2 * prism.q + 2 * prism.r > 3 * prism.p := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_ratios_l3270_327035


namespace NUMINAMATH_CALUDE_olivers_shirts_l3270_327017

theorem olivers_shirts (short_sleeve : ℕ) (washed : ℕ) (unwashed : ℕ) :
  short_sleeve = 39 →
  washed = 20 →
  unwashed = 66 →
  ∃ (long_sleeve : ℕ), long_sleeve = 7 ∧ short_sleeve + long_sleeve = washed + unwashed :=
by sorry

end NUMINAMATH_CALUDE_olivers_shirts_l3270_327017


namespace NUMINAMATH_CALUDE_min_printers_purchase_l3270_327080

def printer_cost_a : ℕ := 350
def printer_cost_b : ℕ := 200

theorem min_printers_purchase :
  ∃ (n_a n_b : ℕ),
    n_a * printer_cost_a = n_b * printer_cost_b ∧
    n_a + n_b = 11 ∧
    ∀ (m_a m_b : ℕ),
      m_a * printer_cost_a = m_b * printer_cost_b →
      m_a + m_b ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_min_printers_purchase_l3270_327080


namespace NUMINAMATH_CALUDE_reciprocal_sum_is_one_l3270_327051

theorem reciprocal_sum_is_one :
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_is_one_l3270_327051


namespace NUMINAMATH_CALUDE_inequality_proof_l3270_327000

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) + a + b + c ≤ 3 + (1 / 3) * (a * b + b * c + c * a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3270_327000


namespace NUMINAMATH_CALUDE_power_function_through_point_l3270_327059

theorem power_function_through_point (k m : ℝ) : 
  k * (2 : ℝ)^m = 1/4 → m * k = -2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3270_327059


namespace NUMINAMATH_CALUDE_mary_nancy_balloon_ratio_l3270_327093

def nancy_balloons : ℕ := 7
def mary_balloons : ℕ := 28

theorem mary_nancy_balloon_ratio :
  mary_balloons / nancy_balloons = 4 := by sorry

end NUMINAMATH_CALUDE_mary_nancy_balloon_ratio_l3270_327093


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_l3270_327061

theorem cubic_roots_sum_of_squares (α β γ : ℂ) : 
  (α^3 - 6*α^2 + 11*α - 6 = 0) → 
  (β^3 - 6*β^2 + 11*β - 6 = 0) → 
  (γ^3 - 6*γ^2 + 11*γ - 6 = 0) → 
  α^2 + β^2 + γ^2 = 14 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_l3270_327061


namespace NUMINAMATH_CALUDE_factorization_proof_l3270_327032

theorem factorization_proof (x y : ℝ) : x^2*y - y^3 = y*(x+y)*(x-y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3270_327032


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l3270_327068

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 4
  let θ : ℝ := 5 * π / 3
  let φ : ℝ := π / 2
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (2, -2 * Real.sqrt 3, 0) := by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l3270_327068


namespace NUMINAMATH_CALUDE_prob_at_least_one_of_each_l3270_327083

/-- The probability of having a boy or a girl -/
def p_boy_or_girl : ℚ := 1/2

/-- The number of children in the family -/
def num_children : ℕ := 4

/-- The probability of having all boys or all girls -/
def p_all_same : ℚ := 2 * (p_boy_or_girl ^ num_children)

/-- The probability of having at least one boy and one girl -/
def p_at_least_one_of_each : ℚ := 1 - p_all_same

theorem prob_at_least_one_of_each :
  p_at_least_one_of_each = 7/8 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_of_each_l3270_327083


namespace NUMINAMATH_CALUDE_complement_of_union_l3270_327073

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {4, 5}
def B : Set Nat := {3, 4}

theorem complement_of_union :
  (A ∪ B)ᶜ = {1, 2, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3270_327073


namespace NUMINAMATH_CALUDE_correct_product_l3270_327050

theorem correct_product (x y : ℚ) (z : ℕ) (h1 : x = 63 / 10000) (h2 : y = 385 / 100) (h3 : z = 24255) :
  x * y = 24255 / 1000000 :=
sorry

end NUMINAMATH_CALUDE_correct_product_l3270_327050


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l3270_327098

theorem quadratic_completion_of_square :
  ∀ x : ℝ, x^2 + 2*x + 3 = (x + 1)^2 + 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l3270_327098


namespace NUMINAMATH_CALUDE_noah_yearly_call_cost_l3270_327001

/-- The total cost of Noah's calls to his Grammy for a year -/
def total_cost (calls_per_week : ℕ) (minutes_per_call : ℕ) (cost_per_minute : ℚ) (weeks_per_year : ℕ) : ℚ :=
  (calls_per_week * minutes_per_call * weeks_per_year : ℕ) * cost_per_minute

/-- Theorem stating that Noah's yearly call cost to his Grammy is $78 -/
theorem noah_yearly_call_cost :
  total_cost 1 30 (5/100) 52 = 78 := by
  sorry

end NUMINAMATH_CALUDE_noah_yearly_call_cost_l3270_327001


namespace NUMINAMATH_CALUDE_y_takes_70_days_l3270_327094

-- Define the work completion rates
def mahesh_rate : ℚ := 1 / 35
def rajesh_rate : ℚ := 1 / 30

-- Define the amount of work Mahesh completes
def mahesh_work : ℚ := mahesh_rate * 20

-- Define the amount of work Rajesh completes
def rajesh_work : ℚ := 1 - mahesh_work

-- Define Y's completion time
def y_completion_time : ℚ := 70

-- Theorem statement
theorem y_takes_70_days :
  y_completion_time = 70 := by sorry

end NUMINAMATH_CALUDE_y_takes_70_days_l3270_327094


namespace NUMINAMATH_CALUDE_proposition_and_converse_l3270_327095

theorem proposition_and_converse :
  (∀ a b : ℝ, a + b ≥ 2 → a ≥ 1 ∨ b ≥ 1) ∧
  (∃ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) ∧ a + b < 2) :=
by sorry

end NUMINAMATH_CALUDE_proposition_and_converse_l3270_327095


namespace NUMINAMATH_CALUDE_x_positive_necessary_not_sufficient_l3270_327021

theorem x_positive_necessary_not_sufficient :
  (∃ x : ℝ, x > 0 ∧ ¬(|x - 1| < 1)) ∧
  (∀ x : ℝ, |x - 1| < 1 → x > 0) :=
by sorry

end NUMINAMATH_CALUDE_x_positive_necessary_not_sufficient_l3270_327021


namespace NUMINAMATH_CALUDE_negative_expression_l3270_327005

theorem negative_expression : 
  (|(-1)| - |(-7)| < 0) ∧ 
  (|(-7)| + |(-1)| ≥ 0) ∧ 
  (|(-7)| - (-1) ≥ 0) ∧ 
  (|(-1)| - (-7) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_expression_l3270_327005


namespace NUMINAMATH_CALUDE_triangle_inequality_possible_third_side_length_l3270_327056

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

theorem possible_third_side_length : 
  ∃ (x : ℝ), x > 0 ∧ 3 + 6 > x ∧ 6 + x > 3 ∧ x + 3 > 6 ∧ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_possible_third_side_length_l3270_327056


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3270_327072

-- Define the vector space
variable (V : Type) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors and their properties
variable (a b : V)
variable (h1 : a = (1 : ℝ) • (1, 0))
variable (h2 : ‖b‖ = 1)
variable (h3 : inner a b = -(1/2 : ℝ) * ‖a‖ * ‖b‖)

-- State the theorem
theorem vector_sum_magnitude :
  ‖a + 2 • b‖ = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3270_327072


namespace NUMINAMATH_CALUDE_jenna_peeled_potatoes_l3270_327040

/-- The number of potatoes Jenna peeled -/
def jenna_potatoes : ℕ := 24

/-- The total number of potatoes -/
def total_potatoes : ℕ := 60

/-- Homer's peeling rate in potatoes per minute -/
def homer_rate : ℕ := 4

/-- Jenna's peeling rate in potatoes per minute -/
def jenna_rate : ℕ := 6

/-- The time Homer peeled alone in minutes -/
def homer_alone_time : ℕ := 6

/-- The combined peeling rate of Homer and Jenna in potatoes per minute -/
def combined_rate : ℕ := homer_rate + jenna_rate

theorem jenna_peeled_potatoes :
  jenna_potatoes = total_potatoes - (homer_rate * homer_alone_time) :=
by sorry

#check jenna_peeled_potatoes

end NUMINAMATH_CALUDE_jenna_peeled_potatoes_l3270_327040


namespace NUMINAMATH_CALUDE_log_not_always_decreasing_l3270_327060

-- Define the logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_not_always_decreasing :
  ∃ (a : ℝ), a > 1 ∧ ∀ (x y : ℝ), x > y → x > 0 → y > 0 → log a x > log a y :=
sorry

end NUMINAMATH_CALUDE_log_not_always_decreasing_l3270_327060


namespace NUMINAMATH_CALUDE_millennium_running_time_l3270_327023

/-- The running time of Millennium in minutes -/
def millennium_time : ℕ := 120

/-- The running time of Alpha Epsilon in minutes -/
def alpha_epsilon_time : ℕ := millennium_time - 30

/-- The running time of Beast of War: Armoured Command in minutes -/
def beast_of_war_time : ℕ := alpha_epsilon_time + 10

/-- Theorem stating that Millennium's running time is 120 minutes -/
theorem millennium_running_time : 
  millennium_time = 120 ∧ 
  alpha_epsilon_time = millennium_time - 30 ∧
  beast_of_war_time = alpha_epsilon_time + 10 ∧
  beast_of_war_time = 100 :=
by sorry

end NUMINAMATH_CALUDE_millennium_running_time_l3270_327023


namespace NUMINAMATH_CALUDE_number_equation_l3270_327029

theorem number_equation (n : ℚ) : n / 5 + 16 = 58 → n / 15 + 74 = 88 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3270_327029


namespace NUMINAMATH_CALUDE_intersection_M_N_l3270_327009

def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x^2 - 6*x + 5 < 0}

theorem intersection_M_N : M ∩ N = Set.Icc 2 5 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3270_327009


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3270_327030

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x^2 + 1 > 2*x ∧ x ≤ 1) ∧
  (∀ x : ℝ, x > 1 → x^2 + 1 > 2*x) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3270_327030


namespace NUMINAMATH_CALUDE_probability_test_l3270_327043

def probability_at_least_3_of_4 (p : ℝ) : ℝ :=
  (4 : ℝ) * p^3 * (1 - p) + p^4

theorem probability_test (p : ℝ) (hp : p = 4/5) :
  probability_at_least_3_of_4 p = 512/625 := by
  sorry

end NUMINAMATH_CALUDE_probability_test_l3270_327043


namespace NUMINAMATH_CALUDE_problem_statement_l3270_327037

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a / (1 + a)) + (b / (1 + b)) = 1) : 
  (a / (1 + b^2)) - (b / (1 + a^2)) = a - b := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3270_327037


namespace NUMINAMATH_CALUDE_beach_trip_ratio_l3270_327078

/-- Proves that the ratio of students taken to the beach to total students is 1/2 -/
theorem beach_trip_ratio (total : ℕ) (remaining : ℕ) : 
  total = 1000 → remaining = 250 → (total - remaining * 2) / total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_beach_trip_ratio_l3270_327078


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3270_327016

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - x - 2 < 0 ↔ -1 < x ∧ x < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3270_327016


namespace NUMINAMATH_CALUDE_boys_usual_time_to_school_l3270_327036

theorem boys_usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) : 
  usual_rate > 0 → 
  usual_time > 0 → 
  usual_rate * usual_time = (6/5 * usual_rate) * (usual_time - 4) → 
  usual_time = 24 := by
sorry

end NUMINAMATH_CALUDE_boys_usual_time_to_school_l3270_327036


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l3270_327015

/-- Given two points A and B as the diameter of a circle, 
    prove that the equation of the circle is as stated. -/
theorem circle_equation_from_diameter 
  (A B : ℝ × ℝ) 
  (hA : A = (4, 9)) 
  (hB : B = (6, -3)) : 
  ∃ (C : ℝ × ℝ) (r : ℝ), 
    C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ 
    r^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4 ∧
    ∀ (x y : ℝ), (x - C.1)^2 + (y - C.2)^2 = r^2 ↔ 
      (x - 5)^2 + (y - 3)^2 = 37 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_l3270_327015


namespace NUMINAMATH_CALUDE_joystick_payment_ratio_l3270_327062

/-- Proves that the ratio of Frank's payment for the joystick to the total cost of the joystick is 1:4 -/
theorem joystick_payment_ratio :
  ∀ (computer_table computer_chair joystick frank_joystick eman_joystick : ℕ),
    computer_table = 140 →
    computer_chair = 100 →
    joystick = 20 →
    frank_joystick + eman_joystick = joystick →
    computer_table + frank_joystick = computer_chair + eman_joystick + 30 →
    frank_joystick * 4 = joystick := by
  sorry

end NUMINAMATH_CALUDE_joystick_payment_ratio_l3270_327062


namespace NUMINAMATH_CALUDE_min_value_theorem_l3270_327014

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) :
  x^2 + 8*x*y + 16*y^2 + 4*z^2 ≥ 192 ∧
  (x^2 + 8*x*y + 16*y^2 + 4*z^2 = 192 ↔ x = 8 ∧ y = 2 ∧ z = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3270_327014


namespace NUMINAMATH_CALUDE_arithmetic_sequence_and_binomial_expansion_l3270_327058

theorem arithmetic_sequence_and_binomial_expansion :
  let a : ℕ → ℤ := λ n => 3*n - 5
  let binomial_sum : ℕ → ℤ := λ k => Nat.choose 5 k + Nat.choose 6 k + Nat.choose 7 k
  a 20 = binomial_sum 4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_and_binomial_expansion_l3270_327058


namespace NUMINAMATH_CALUDE_non_tipping_customers_l3270_327075

/-- Calculates the number of non-tipping customers given the total number of customers,
    the tip amount per tipping customer, and the total tips earned. -/
theorem non_tipping_customers
  (total_customers : ℕ)
  (tip_amount : ℕ)
  (total_tips : ℕ)
  (h1 : total_customers > 0)
  (h2 : tip_amount > 0)
  (h3 : total_tips % tip_amount = 0)
  (h4 : total_tips / tip_amount ≤ total_customers) :
  total_customers - (total_tips / tip_amount) =
    total_customers - (total_tips / tip_amount) :=
by sorry

end NUMINAMATH_CALUDE_non_tipping_customers_l3270_327075


namespace NUMINAMATH_CALUDE_min_even_integers_l3270_327088

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 26 → 
  a + b + c + d = 41 → 
  a + b + c + d + e + f = 57 → 
  ∃ (n : ℕ), n ≥ 1 ∧ 
  (∀ (m : ℕ), m < n → 
    ¬∃ (evens : Finset ℤ), evens.card = m ∧ 
    (∀ x ∈ evens, Even x) ∧ 
    evens ⊆ {a, b, c, d, e, f}) :=
sorry

end NUMINAMATH_CALUDE_min_even_integers_l3270_327088


namespace NUMINAMATH_CALUDE_max_product_l3270_327004

def digits : List ℕ := [1, 3, 5, 8, 9]

def is_valid_arrangement (a b c d e : ℕ) : Prop :=
  {a, b, c, d, e} = digits.toFinset

def four_digit_num (a b c d : ℕ) : ℕ := 
  1000 * a + 100 * b + 10 * c + d

def product (a b c d e : ℕ) : ℕ :=
  (four_digit_num a b c d) * e

theorem max_product :
  ∀ a b c d e,
    is_valid_arrangement a b c d e →
    product a b c d e ≤ product 8 5 3 1 9 :=
sorry

end NUMINAMATH_CALUDE_max_product_l3270_327004


namespace NUMINAMATH_CALUDE_range_of_g_l3270_327070

/-- The function g(x) = ⌊2x⌋ - 2x has a range of [-1, 0] -/
theorem range_of_g : 
  let g : ℝ → ℝ := λ x => ⌊2 * x⌋ - 2 * x
  ∀ y : ℝ, (∃ x : ℝ, g x = y) ↔ -1 ≤ y ∧ y ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l3270_327070


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_6_l3270_327097

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ d : ℚ) : ℕ → ℚ
  | 0 => 0
  | n + 1 => a₁ + n * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * (a₁ + (a₁ + (n - 1) * d)) / 2

-- Theorem statement
theorem arithmetic_sequence_sum_6 (a₁ d : ℚ) :
  a₁ = 1/2 →
  sum_arithmetic_sequence a₁ d 4 = 20 →
  sum_arithmetic_sequence a₁ d 6 = 48 := by
  sorry

-- The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_6_l3270_327097


namespace NUMINAMATH_CALUDE_cookies_eaten_l3270_327008

theorem cookies_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 18 → remaining = 9 → eaten = initial - remaining → eaten = 9 := by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_l3270_327008


namespace NUMINAMATH_CALUDE_pair_probability_after_removal_l3270_327019

/-- Represents a deck of cards -/
structure Deck :=
  (cards : Finset ℕ)
  (count : ℕ → ℕ)
  (total : ℕ)

/-- Initial deck configuration -/
def initial_deck : Deck :=
  { cards := Finset.range 12,
    count := λ n => if n ∈ Finset.range 12 then 4 else 0,
    total := 48 }

/-- Deck after removing two pairs -/
def deck_after_removal (d : Deck) : Deck :=
  { cards := d.cards,
    count := λ n => if n ∈ d.cards then d.count n - 2 else 0,
    total := d.total - 4 }

/-- Number of ways to choose 2 cards from n cards -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Number of ways to form pairs from remaining cards -/
def pair_ways (d : Deck) : ℕ :=
  (d.cards.filter (λ n => d.count n = 4)).card * 6 +
  (d.cards.filter (λ n => d.count n = 2)).card * 1

/-- Probability of selecting a pair -/
def pair_probability (d : Deck) : ℚ :=
  pair_ways d / choose_two d.total

theorem pair_probability_after_removal :
  pair_probability (deck_after_removal initial_deck) = 31 / 473 := by
  sorry

#eval pair_probability (deck_after_removal initial_deck)

end NUMINAMATH_CALUDE_pair_probability_after_removal_l3270_327019


namespace NUMINAMATH_CALUDE_invisible_dots_sum_l3270_327007

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The total number of dice -/
def num_dice : ℕ := 4

/-- The visible numbers on the stacked dice -/
def visible_numbers : List ℕ := [1, 1, 2, 3, 3, 4, 5, 5, 6]

/-- Theorem: The total number of dots not visible is 54 -/
theorem invisible_dots_sum : 
  num_dice * die_sum - visible_numbers.sum = 54 := by sorry

end NUMINAMATH_CALUDE_invisible_dots_sum_l3270_327007


namespace NUMINAMATH_CALUDE_least_positive_integer_satisfying_congruences_l3270_327071

theorem least_positive_integer_satisfying_congruences : ∃ n : ℕ, 
  n > 0 ∧
  n % 3 = 1 ∧
  n % 5 = 3 ∧
  n % 6 = 5 ∧
  n % 7 = 2 ∧
  (∀ m : ℕ, m > 0 ∧ m % 3 = 1 ∧ m % 5 = 3 ∧ m % 6 = 5 ∧ m % 7 = 2 → m ≥ n) ∧
  n = 83 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_satisfying_congruences_l3270_327071


namespace NUMINAMATH_CALUDE_cauchy_mean_value_theorem_sine_cosine_l3270_327042

open Real

theorem cauchy_mean_value_theorem_sine_cosine :
  ∃ c : ℝ, 0 < c ∧ c < π / 2 ∧
    (cos c) / (-sin c) = (sin (π / 2) - sin 0) / (cos (π / 2) - cos 0) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_mean_value_theorem_sine_cosine_l3270_327042


namespace NUMINAMATH_CALUDE_valid_numbers_l3270_327039

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 730000 + 10000 * a + 1000 * b + 100 * c + 6 ∧
    b < 4 ∧
    n % 56 = 0 ∧
    (a % 40 = a % 61) ∧ (a % 61 = a % 810)

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ (n = 731136 ∨ n = 737016 ∨ n = 737296) :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l3270_327039


namespace NUMINAMATH_CALUDE_ratio_HB_JD_l3270_327013

-- Define the points
variable (A B C D E F G H J : ℝ × ℝ)

-- Define the conditions
axiom on_line : ∃ (t : ℝ), B = A + t • (F - A) ∧
                           C = A + (t + 1) • (F - A) ∧
                           D = A + (t + 3) • (F - A) ∧
                           E = A + (t + 4) • (F - A) ∧
                           F = A + (t + 5) • (F - A)

axiom segment_lengths : 
  dist A B = 1 ∧ dist B C = 2 ∧ dist C D = 1 ∧ dist D E = 2 ∧ dist E F = 1

axiom G_not_on_line : ∀ (t : ℝ), G ≠ A + t • (F - A)

axiom H_on_GD : ∃ (t : ℝ), H = G + t • (D - G)

axiom J_on_GE : ∃ (t : ℝ), J = G + t • (E - G)

axiom parallel_lines : 
  (H.2 - B.2) / (H.1 - B.1) = (J.2 - D.2) / (J.1 - D.1) ∧
  (J.2 - D.2) / (J.1 - D.1) = (G.2 - A.2) / (G.1 - A.1)

-- Theorem to prove
theorem ratio_HB_JD : dist H B / dist J D = 5 / 4 :=
sorry

end NUMINAMATH_CALUDE_ratio_HB_JD_l3270_327013


namespace NUMINAMATH_CALUDE_stratified_sampling_pine_saplings_l3270_327089

theorem stratified_sampling_pine_saplings 
  (total_saplings : ℕ) 
  (pine_saplings : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_saplings = 30000) 
  (h2 : pine_saplings = 4000) 
  (h3 : sample_size = 150) :
  (pine_saplings : ℚ) / total_saplings * sample_size = 20 := by
sorry


end NUMINAMATH_CALUDE_stratified_sampling_pine_saplings_l3270_327089


namespace NUMINAMATH_CALUDE_parabola_equation_l3270_327018

/-- A parabola with vertex at the origin and axis of symmetry x = 2 -/
structure Parabola where
  /-- The equation of the parabola in the form y² = -2px -/
  equation : ℝ → ℝ → Prop
  /-- The vertex of the parabola is at the origin -/
  vertex_at_origin : equation 0 0
  /-- The axis of symmetry is x = 2 -/
  axis_of_symmetry : ∀ y, equation 2 y ↔ equation 2 (-y)

/-- The equation of the parabola is y² = -8x -/
theorem parabola_equation (p : Parabola) : 
  p.equation = fun x y => y^2 = -8*x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l3270_327018


namespace NUMINAMATH_CALUDE_i_minus_one_squared_l3270_327069

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem i_minus_one_squared : (i - 1)^2 = -2*i := by
  sorry

end NUMINAMATH_CALUDE_i_minus_one_squared_l3270_327069


namespace NUMINAMATH_CALUDE_pens_purchased_correct_l3270_327066

/-- The number of pens purchased given the conditions of the problem -/
def number_of_pens : ℕ := 30

/-- The total number of pencils purchased -/
def number_of_pencils : ℕ := 75

/-- The total cost of all items purchased -/
def total_cost : ℚ := 750

/-- The average price of a pencil -/
def pencil_price : ℚ := 2

/-- The average price of a pen -/
def pen_price : ℚ := 20

/-- Theorem stating that the number of pens purchased is correct given the conditions -/
theorem pens_purchased_correct : 
  number_of_pens * pen_price + number_of_pencils * pencil_price = total_cost := by
  sorry

end NUMINAMATH_CALUDE_pens_purchased_correct_l3270_327066


namespace NUMINAMATH_CALUDE_ryan_final_tokens_l3270_327048

def initial_tokens : ℕ := 36
def pacman_fraction : ℚ := 1/3
def candy_crush_fraction : ℚ := 1/4
def skiball_tokens : ℕ := 7
def parent_multiplier : ℕ := 7

theorem ryan_final_tokens :
  let pacman_tokens := (pacman_fraction * initial_tokens).floor
  let candy_crush_tokens := (candy_crush_fraction * initial_tokens).floor
  let total_spent := pacman_tokens + candy_crush_tokens + skiball_tokens
  let tokens_left := initial_tokens - total_spent
  let parent_bought := parent_multiplier * skiball_tokens
  tokens_left + parent_bought = 57 := by
  sorry

end NUMINAMATH_CALUDE_ryan_final_tokens_l3270_327048


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_example_l3270_327055

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ aₙ d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Theorem: An arithmetic sequence starting with 2, ending with 1007, 
    and having a common difference of 5, contains 202 terms. -/
theorem arithmetic_sequence_length_example : 
  arithmetic_sequence_length 2 1007 5 = 202 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_example_l3270_327055


namespace NUMINAMATH_CALUDE_multiplication_of_squares_l3270_327041

theorem multiplication_of_squares (a b : ℝ) : 2 * a^2 * 3 * b^2 = 6 * a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_squares_l3270_327041


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l3270_327096

theorem square_sum_given_product_and_sum (x y : ℝ) 
  (h1 : x * y = 16) (h2 : x + y = 10) : x^2 + y^2 = 68 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l3270_327096


namespace NUMINAMATH_CALUDE_sphere_volume_from_parallel_planes_l3270_327081

theorem sphere_volume_from_parallel_planes (R : ℝ) :
  R > 0 →
  ∃ (h : ℝ),
    h > 0 ∧
    h < R ∧
    (h^2 + 9^2 = R^2) ∧
    ((h + 3)^2 + 12^2 = R^2) →
    (4 / 3 * Real.pi * R^3 = 4050 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_from_parallel_planes_l3270_327081


namespace NUMINAMATH_CALUDE_jerry_softball_time_l3270_327027

theorem jerry_softball_time (
  num_daughters : ℕ)
  (games_per_daughter : ℕ)
  (practice_hours_per_game : ℕ)
  (game_duration : ℕ)
  (h1 : num_daughters = 4)
  (h2 : games_per_daughter = 12)
  (h3 : practice_hours_per_game = 6)
  (h4 : game_duration = 3) :
  num_daughters * games_per_daughter * (practice_hours_per_game + game_duration) = 432 :=
by sorry

end NUMINAMATH_CALUDE_jerry_softball_time_l3270_327027


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_eight_l3270_327064

theorem ceiling_negative_three_point_eight :
  ⌈(-3.8 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_eight_l3270_327064


namespace NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_half_l3270_327067

theorem no_solution_iff_n_eq_neg_half (n : ℝ) : 
  (∀ x y z : ℝ, ¬(2*n*x + y = 2 ∧ n*y + z = 2 ∧ x + 2*n*z = 2)) ↔ n = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_half_l3270_327067


namespace NUMINAMATH_CALUDE_total_books_count_l3270_327031

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 8

/-- The number of shelves with mystery books -/
def mystery_shelves : ℕ := 5

/-- The number of shelves with picture books -/
def picture_shelves : ℕ := 4

/-- The total number of books -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem total_books_count : total_books = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l3270_327031


namespace NUMINAMATH_CALUDE_max_value_a_l3270_327006

theorem max_value_a (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b = 1 - a)
  (h4 : ∀ x : ℝ, x ∈ Set.Icc 0 1 → Real.exp x ≤ (1 + a * x) / (1 - b * x)) :
  a ≤ (1 : ℝ) / 2 ∧ ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ Real.exp x = (1 + (1/2) * x) / (1 - (1/2) * x) :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l3270_327006


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3270_327045

theorem quadratic_minimum : 
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) := by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3270_327045


namespace NUMINAMATH_CALUDE_two_sqrt_two_less_than_three_l3270_327046

theorem two_sqrt_two_less_than_three : 2 * Real.sqrt 2 < 3 := by
  sorry

end NUMINAMATH_CALUDE_two_sqrt_two_less_than_three_l3270_327046


namespace NUMINAMATH_CALUDE_solution_exists_l3270_327033

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the equation
def equation (x : ℝ) : Prop :=
  cubeRoot (24 * x + cubeRoot (24 * x + 16)) = 14

-- Theorem statement
theorem solution_exists : ∃ x : ℝ, equation x ∧ x = 114 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l3270_327033


namespace NUMINAMATH_CALUDE_platform_length_l3270_327034

/-- The length of a platform given train specifications -/
theorem platform_length
  (train_length : ℝ)
  (time_tree : ℝ)
  (time_platform : ℝ)
  (h1 : train_length = 1200)
  (h2 : time_tree = 120)
  (h3 : time_platform = 160) :
  let train_speed := train_length / time_tree
  let platform_length := train_speed * time_platform - train_length
  platform_length = 400 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l3270_327034


namespace NUMINAMATH_CALUDE_cube_root_of_4x_plus_3y_is_3_l3270_327020

theorem cube_root_of_4x_plus_3y_is_3 (x y : ℝ) : 
  y = Real.sqrt (3 - x) + Real.sqrt (x - 3) + 5 → 
  (4 * x + 3 * y) ^ (1/3 : ℝ) = 3 := by
sorry

end NUMINAMATH_CALUDE_cube_root_of_4x_plus_3y_is_3_l3270_327020


namespace NUMINAMATH_CALUDE_shirt_and_sweater_cost_l3270_327090

theorem shirt_and_sweater_cost (shirt_price sweater_price total_cost : ℝ) : 
  shirt_price = 36.46 →
  sweater_price = shirt_price + 7.43 →
  total_cost = shirt_price + sweater_price →
  total_cost = 80.35 := by
sorry

end NUMINAMATH_CALUDE_shirt_and_sweater_cost_l3270_327090


namespace NUMINAMATH_CALUDE_negation_of_forall_product_nonzero_l3270_327074

theorem negation_of_forall_product_nonzero (f g : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x * g x ≠ 0) ↔ (∃ x : ℝ, f x = 0 ∨ g x = 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_product_nonzero_l3270_327074


namespace NUMINAMATH_CALUDE_taxi_distance_is_ten_miles_l3270_327022

/-- Calculates the taxi fare distance given the total fare, initial fare, initial distance, and additional fare per unit distance -/
def taxi_fare_distance (total_fare : ℚ) (initial_fare : ℚ) (initial_distance : ℚ) (additional_fare_per_unit : ℚ) : ℚ :=
  initial_distance + (total_fare - initial_fare) / additional_fare_per_unit

/-- Theorem: Given the specified fare structure and total fare, the distance traveled is 10 miles -/
theorem taxi_distance_is_ten_miles :
  let total_fare : ℚ := 59
  let initial_fare : ℚ := 10
  let initial_distance : ℚ := 1/5
  let additional_fare_per_unit : ℚ := 1/(1/5)
  taxi_fare_distance total_fare initial_fare initial_distance additional_fare_per_unit = 10 := by
  sorry

#eval taxi_fare_distance 59 10 (1/5) 5

end NUMINAMATH_CALUDE_taxi_distance_is_ten_miles_l3270_327022


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l3270_327085

/-- The total surface area of a cuboid with dimensions in the ratio 6:5:4 and actual dimensions 90 cm, 75 cm, and 60 cm is 33300 cm². -/
theorem cuboid_surface_area : 
  let length : ℝ := 90
  let breadth : ℝ := 75
  let height : ℝ := 60
  let ratio_length : ℝ := 6
  let ratio_breadth : ℝ := 5
  let ratio_height : ℝ := 4
  -- Ensure the dimensions are in the correct ratio
  length / ratio_length = breadth / ratio_breadth ∧ 
  breadth / ratio_breadth = height / ratio_height →
  -- Calculate the total surface area
  2 * (length * breadth + breadth * height + height * length) = 33300 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_l3270_327085


namespace NUMINAMATH_CALUDE_ace_king_queen_probability_l3270_327079

-- Define the total number of cards in a standard deck
def totalCards : ℕ := 52

-- Define the number of each face card (Ace, King, Queen)
def faceCards : ℕ := 4

-- Define the probability of drawing the sequence (Ace, King, Queen)
def probAceKingQueen : ℚ := (faceCards : ℚ) / totalCards *
                            (faceCards : ℚ) / (totalCards - 1) *
                            (faceCards : ℚ) / (totalCards - 2)

-- Theorem statement
theorem ace_king_queen_probability :
  probAceKingQueen = 8 / 16575 := by sorry

end NUMINAMATH_CALUDE_ace_king_queen_probability_l3270_327079


namespace NUMINAMATH_CALUDE_milk_conversion_theorem_l3270_327076

/-- Represents the conversion between milliliters and fluid ounces -/
structure MilkConversion where
  packets : Nat
  ml_per_packet : Nat
  total_ounces : Nat

/-- Calculates the number of milliliters in one fluid ounce -/
def ml_per_ounce (conv : MilkConversion) : Rat :=
  (conv.packets * conv.ml_per_packet) / conv.total_ounces

/-- Theorem stating that under the given conditions, one fluid ounce equals 30 ml -/
theorem milk_conversion_theorem (conv : MilkConversion) 
  (h1 : conv.packets = 150)
  (h2 : conv.ml_per_packet = 250)
  (h3 : conv.total_ounces = 1250) : 
  ml_per_ounce conv = 30 := by
  sorry

end NUMINAMATH_CALUDE_milk_conversion_theorem_l3270_327076


namespace NUMINAMATH_CALUDE_sharon_salary_increase_l3270_327057

theorem sharon_salary_increase (S : ℝ) (h1 : S + 0.20 * S = 600) (h2 : S + x * S = 575) : x = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_sharon_salary_increase_l3270_327057


namespace NUMINAMATH_CALUDE_smallest_x_for_fraction_l3270_327082

theorem smallest_x_for_fraction (x : ℕ) (y : ℤ) : 
  (3 : ℚ) / 4 = y / (256 + x) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_fraction_l3270_327082


namespace NUMINAMATH_CALUDE_fence_poles_count_l3270_327026

theorem fence_poles_count (total_length bridge_length pole_spacing : ℕ) 
  (h1 : total_length = 900)
  (h2 : bridge_length = 42)
  (h3 : pole_spacing = 6) : 
  (2 * ((total_length - bridge_length) / pole_spacing)) = 286 := by
  sorry

end NUMINAMATH_CALUDE_fence_poles_count_l3270_327026


namespace NUMINAMATH_CALUDE_existence_of_sets_l3270_327028

theorem existence_of_sets : ∃ (A B C : Set ℕ),
  (A ∩ B).Nonempty ∧
  (A ∩ C).Nonempty ∧
  ((A ∩ B) \ C).Nonempty := by
  sorry

end NUMINAMATH_CALUDE_existence_of_sets_l3270_327028


namespace NUMINAMATH_CALUDE_cube_sum_divided_l3270_327012

theorem cube_sum_divided (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 9 = 219 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_divided_l3270_327012


namespace NUMINAMATH_CALUDE_stating_number_of_people_in_first_group_l3270_327025

/-- Represents the amount of work one person can do in one day -/
def work_per_person_per_day : ℝ := 1

/-- Represents the number of days given in the problem -/
def days : ℕ := 3

/-- Represents the number of people in the second group -/
def people_second_group : ℕ := 6

/-- Represents the amount of work done by the first group -/
def work_first_group : ℕ := 3

/-- Represents the amount of work done by the second group -/
def work_second_group : ℕ := 6

/-- 
Theorem stating that the number of people in the first group is 3,
given the conditions from the problem.
-/
theorem number_of_people_in_first_group : 
  ∃ (p : ℕ), 
    p * days * work_per_person_per_day = work_first_group ∧
    people_second_group * days * work_per_person_per_day = work_second_group ∧
    p = 3 := by
  sorry

end NUMINAMATH_CALUDE_stating_number_of_people_in_first_group_l3270_327025


namespace NUMINAMATH_CALUDE_no_valid_tiling_l3270_327077

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  rows : ℕ
  cols : ℕ

/-- Represents a domino with given dimensions -/
structure Domino where
  length : ℕ
  width : ℕ

/-- Represents a tiling configuration -/
structure Tiling where
  rect : Rectangle
  dominos : List Domino
  count : ℕ

def is_valid_tiling (t : Tiling) : Prop :=
  t.rect.rows = 11 ∧
  t.rect.cols = 12 ∧
  t.count = 19 ∧
  ∀ d ∈ t.dominos, (d.length = 6 ∧ d.width = 1) ∨ (d.length = 7 ∧ d.width = 1) ∨
                   (d.length = 1 ∧ d.width = 6) ∨ (d.length = 1 ∧ d.width = 7)

theorem no_valid_tiling :
  ¬ ∃ t : Tiling, is_valid_tiling t := by
  sorry

end NUMINAMATH_CALUDE_no_valid_tiling_l3270_327077


namespace NUMINAMATH_CALUDE_remainder_when_divided_by_x_plus_one_l3270_327024

def q (x : ℝ) : ℝ := 2*x^4 - 3*x^3 + 4*x^2 - 5*x + 6

theorem remainder_when_divided_by_x_plus_one :
  ∃ p : ℝ → ℝ, q = fun x ↦ (x + 1) * p x + 20 :=
sorry

end NUMINAMATH_CALUDE_remainder_when_divided_by_x_plus_one_l3270_327024


namespace NUMINAMATH_CALUDE_smallest_positive_and_largest_negative_integer_l3270_327038

theorem smallest_positive_and_largest_negative_integer :
  (∀ n : ℤ, n > 0 → n ≥ 1) ∧ (∀ m : ℤ, m < 0 → m ≤ -1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_and_largest_negative_integer_l3270_327038


namespace NUMINAMATH_CALUDE_polynomial_roots_l3270_327087

theorem polynomial_roots : ∃ (x₁ x₂ x₃ x₄ : ℂ),
  (x₁ = (7 + Real.sqrt 37) / 6) ∧
  (x₂ = (7 - Real.sqrt 37) / 6) ∧
  (x₃ = (-3 + Real.sqrt 5) / 2) ∧
  (x₄ = (-3 - Real.sqrt 5) / 2) ∧
  (∀ x : ℂ, 3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3 = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3270_327087


namespace NUMINAMATH_CALUDE_remainder_seven_twelfth_mod_hundred_l3270_327063

theorem remainder_seven_twelfth_mod_hundred : 7^12 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_twelfth_mod_hundred_l3270_327063


namespace NUMINAMATH_CALUDE_wrapping_paper_fraction_l3270_327086

theorem wrapping_paper_fraction (total_fraction : ℚ) (num_small : ℕ) (num_large : ℕ) :
  total_fraction = 3/8 →
  num_small = 4 →
  num_large = 2 →
  (∃ small_fraction : ℚ, 
    total_fraction = num_small * small_fraction + num_large * (2 * small_fraction) ∧
    small_fraction = 3/64) :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_fraction_l3270_327086


namespace NUMINAMATH_CALUDE_average_pages_is_23_l3270_327003

/-- Represents the number of pages in the book -/
def total_pages : ℕ := 161

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Calculates the average number of pages read per day -/
def average_pages_per_day : ℚ := total_pages / days_in_week

/-- Theorem stating that the average number of pages read per day is 23 -/
theorem average_pages_is_23 : average_pages_per_day = 23 := by
  sorry

end NUMINAMATH_CALUDE_average_pages_is_23_l3270_327003


namespace NUMINAMATH_CALUDE_min_value_ab_l3270_327092

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b - 2 * a - b = 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y - 2 * x - y = 0 → a * b ≤ x * y ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y - 2 * x - y = 0 ∧ x * y = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_l3270_327092


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3270_327044

theorem line_segment_endpoint (y : ℝ) : 
  y > 0 → 
  Real.sqrt ((3 - (-5))^2 + (y - 4)^2) = 12 → 
  y = 4 + 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3270_327044
