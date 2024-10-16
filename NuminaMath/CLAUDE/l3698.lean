import Mathlib

namespace NUMINAMATH_CALUDE_zero_geometric_mean_with_one_l3698_369854

def geometric_mean (list : List ℝ) : ℝ := (list.prod) ^ (1 / list.length)

theorem zero_geometric_mean_with_one {n : ℕ} (h : n > 1) :
  let list : List ℝ := 1 :: List.replicate (n - 1) 0
  geometric_mean list = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_geometric_mean_with_one_l3698_369854


namespace NUMINAMATH_CALUDE_problem_solving_probability_l3698_369802

theorem problem_solving_probability (xavier_prob yvonne_prob zelda_prob : ℚ)
  (hx : xavier_prob = 1 / 4)
  (hy : yvonne_prob = 1 / 3)
  (hz : zelda_prob = 5 / 8) :
  xavier_prob * yvonne_prob * (1 - zelda_prob) = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l3698_369802


namespace NUMINAMATH_CALUDE_cupcakes_per_package_l3698_369856

theorem cupcakes_per_package
  (initial_cupcakes : ℕ)
  (eaten_cupcakes : ℕ)
  (num_packages : ℕ)
  (h1 : initial_cupcakes = 18)
  (h2 : eaten_cupcakes = 8)
  (h3 : num_packages = 5)
  : (initial_cupcakes - eaten_cupcakes) / num_packages = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cupcakes_per_package_l3698_369856


namespace NUMINAMATH_CALUDE_sanchez_rope_theorem_l3698_369840

def rope_problem (rope_last_week : ℕ) (rope_difference : ℕ) (inches_per_foot : ℕ) : Prop :=
  let rope_this_week : ℕ := rope_last_week - rope_difference
  let total_rope_feet : ℕ := rope_last_week + rope_this_week
  let total_rope_inches : ℕ := total_rope_feet * inches_per_foot
  total_rope_inches = 96

theorem sanchez_rope_theorem : rope_problem 6 4 12 := by
  sorry

end NUMINAMATH_CALUDE_sanchez_rope_theorem_l3698_369840


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3698_369884

/-- Proves that the interest rate A lends to B is 10% given the specified conditions -/
theorem interest_rate_calculation (principal : ℝ) (rate_c : ℝ) (time : ℝ) (b_gain : ℝ) 
  (h1 : principal = 3500)
  (h2 : rate_c = 0.115)
  (h3 : time = 3)
  (h4 : b_gain = 157.5)
  (h5 : b_gain = principal * rate_c * time - principal * rate_a * time) : 
  rate_a = 0.1 := by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_interest_rate_calculation_l3698_369884


namespace NUMINAMATH_CALUDE_gcd_xyz_square_l3698_369816

theorem gcd_xyz_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, Nat.gcd x (Nat.gcd y z) * x * y * z = k ^ 2 := by
sorry

end NUMINAMATH_CALUDE_gcd_xyz_square_l3698_369816


namespace NUMINAMATH_CALUDE_line_y_intercept_l3698_369826

/-- A straight line in the xy-plane with slope 2 passing through (239, 480) has y-intercept 2 -/
theorem line_y_intercept (m : ℝ) (x₀ y₀ b : ℝ) : 
  m = 2 → 
  x₀ = 239 → 
  y₀ = 480 → 
  y₀ = m * x₀ + b → 
  b = 2 := by sorry

end NUMINAMATH_CALUDE_line_y_intercept_l3698_369826


namespace NUMINAMATH_CALUDE_inequality_solution_l3698_369843

theorem inequality_solution (m : ℝ) : 
  (∀ x : ℝ, (x + m) / 2 - 1 > 2 * m ↔ x > 5) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3698_369843


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l3698_369852

theorem root_sum_reciprocal (p q r : ℂ) : 
  (p^3 - 2*p + 3 = 0) → 
  (q^3 - 2*q + 3 = 0) → 
  (r^3 - 2*r + 3 = 0) → 
  1/(p+2) + 1/(q+2) + 1/(r+2) = -10 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l3698_369852


namespace NUMINAMATH_CALUDE_order_of_abc_l3698_369860

theorem order_of_abc (a b c : ℝ) (ha : a = 2^(4/3)) (hb : b = 3^(2/3)) (hc : c = 25^(1/3)) :
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l3698_369860


namespace NUMINAMATH_CALUDE_concert_attendance_l3698_369886

theorem concert_attendance (num_buses : ℕ) (students_per_bus : ℕ) 
  (h1 : num_buses = 8) (h2 : students_per_bus = 45) : 
  num_buses * students_per_bus = 360 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l3698_369886


namespace NUMINAMATH_CALUDE_b_n_formula_l3698_369867

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem b_n_formula (a b : ℕ → ℤ) :
  arithmetic_sequence a →
  a 3 = 2 →
  a 8 = 12 →
  b 1 = 4 →
  (∀ n : ℕ, n > 1 → a n + b n = b (n - 1)) →
  ∀ n : ℕ, b n = -n^2 + 3*n + 2 :=
sorry

end NUMINAMATH_CALUDE_b_n_formula_l3698_369867


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3698_369839

theorem quadratic_factorization (C D : ℤ) :
  (∀ y, 20 * y^2 - 122 * y + 72 = (C * y - 8) * (D * y - 9)) →
  C * D + C = 25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3698_369839


namespace NUMINAMATH_CALUDE_quadratic_sum_equality_l3698_369890

/-- A quadratic function satisfying specific conditions -/
def P : ℝ → ℝ := fun x ↦ 6 * x^2 - 3 * x + 7

/-- The theorem statement -/
theorem quadratic_sum_equality (a b c : ℤ) :
  P 0 = 7 ∧ P 1 = 10 ∧ P 2 = 25 ∧
  (∀ x : ℝ, 0 < x → x < 1 →
    (∑' n, P n * x^n) = (a * x^2 + b * x + c) / (1 - x)^3) →
  (a, b, c) = (16, -11, 7) := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_equality_l3698_369890


namespace NUMINAMATH_CALUDE_stevens_collection_group_size_l3698_369883

theorem stevens_collection_group_size :
  let skittles : ℕ := 4502
  let erasers : ℕ := 4276
  let num_groups : ℕ := 154
  let total_items : ℕ := skittles + erasers
  (total_items / num_groups : ℕ) = 57 := by
  sorry

end NUMINAMATH_CALUDE_stevens_collection_group_size_l3698_369883


namespace NUMINAMATH_CALUDE_chucks_team_score_l3698_369876

theorem chucks_team_score (yellow_team_score lead : ℕ) 
  (h1 : yellow_team_score = 55)
  (h2 : lead = 17) :
  yellow_team_score + lead = 72 := by
  sorry

end NUMINAMATH_CALUDE_chucks_team_score_l3698_369876


namespace NUMINAMATH_CALUDE_virginia_eggs_remaining_l3698_369885

theorem virginia_eggs_remaining (initial_eggs : ℕ) (eggs_taken : ℕ) : 
  initial_eggs = 200 → eggs_taken = 37 → initial_eggs - eggs_taken = 163 :=
by
  sorry

end NUMINAMATH_CALUDE_virginia_eggs_remaining_l3698_369885


namespace NUMINAMATH_CALUDE_cos_squared_pi_third_minus_x_l3698_369803

theorem cos_squared_pi_third_minus_x (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.cos (π/3 - x) ^ 2 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_pi_third_minus_x_l3698_369803


namespace NUMINAMATH_CALUDE_quarter_circles_sum_approaches_circumference_l3698_369817

/-- The sum of quarter-circle arc lengths approaches the original circle's circumference as n approaches infinity -/
theorem quarter_circles_sum_approaches_circumference (R : ℝ) (h : R > 0) :
  let C := 2 * Real.pi * R
  let quarter_circle_sum (n : ℕ) := 2 * n * (Real.pi * C) / (2 * n)
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |quarter_circle_sum n - C| < ε :=
by sorry

end NUMINAMATH_CALUDE_quarter_circles_sum_approaches_circumference_l3698_369817


namespace NUMINAMATH_CALUDE_number_line_properties_l3698_369851

-- Definition of distance between points on a number line
def distance (a b : ℚ) : ℚ := |a - b|

-- Statements to prove
theorem number_line_properties :
  -- 1. The distance between 2 and 5 is 3
  distance 2 5 = 3 ∧
  -- 2. The distance between x and -6 is |x + 6|
  ∀ x : ℚ, distance x (-6) = |x + 6| ∧
  -- 3. For -2 < x < 2, |x-2|+|x+2| = 4
  ∀ x : ℚ, -2 < x → x < 2 → |x-2|+|x+2| = 4 ∧
  -- 4. For |x-1|+|x+3| > 4, x > 1 or x < -3
  ∀ x : ℚ, |x-1|+|x+3| > 4 → x > 1 ∨ x < -3 ∧
  -- 5. The minimum value of |x-3|+|x+2|+|x+1| is 5, occurring at x = -1
  (∀ x : ℚ, |x-3|+|x+2|+|x+1| ≥ 5) ∧ (|-1-3|+|-1+2|+|-1+1| = 5) ∧
  -- 6. The maximum value of y when |x-1|+|x+2|=10-|y-3|-|y+4| is 3
  ∀ x y : ℚ, |x-1|+|x+2| = 10-|y-3|-|y+4| → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_number_line_properties_l3698_369851


namespace NUMINAMATH_CALUDE_car_growth_rates_l3698_369807

/-- The number of cars in millions at the end of 2010 -/
def cars_2010 : ℝ := 1

/-- The number of cars in millions at the end of 2012 -/
def cars_2012 : ℝ := 1.44

/-- The maximum allowed number of cars in millions at the end of 2013 -/
def max_cars_2013 : ℝ := 1.5552

/-- The proportion of cars scrapped in 2013 -/
def scrap_rate : ℝ := 0.1

/-- The average annual growth rate of cars from 2010 to 2012 -/
def growth_rate_2010_2012 : ℝ := 0.2

/-- The maximum annual growth rate from 2012 to 2013 -/
def max_growth_rate_2012_2013 : ℝ := 0.18

theorem car_growth_rates :
  (cars_2010 * (1 + growth_rate_2010_2012)^2 = cars_2012) ∧
  (cars_2012 * (1 + max_growth_rate_2012_2013) * (1 - scrap_rate) ≤ max_cars_2013) := by
  sorry

end NUMINAMATH_CALUDE_car_growth_rates_l3698_369807


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_27_l3698_369842

theorem greatest_three_digit_multiple_of_27 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 27 ∣ n → n ≤ 999 ∧ 27 ∣ 999 := by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_27_l3698_369842


namespace NUMINAMATH_CALUDE_omega_range_l3698_369865

open Real

/-- A function that is increasing on the real numbers. -/
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Definition of an acute angle. -/
def AcuteAngle (ω : ℝ) : Prop := 0 < ω ∧ ω < π / 2

/-- A function that is monotonically decreasing on an interval. -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

/-- The main theorem stating the range of ω. -/
theorem omega_range (f : ℝ → ℝ) (ω : ℝ) :
  IncreasingFunction f →
  f (sin ω) + f (-cos ω) > f (cos ω) + f (-sin ω) →
  AcuteAngle ω →
  MonotonicallyDecreasing (fun x ↦ sin (ω * x + π / 4)) (π / 2) π →
  π / 4 < ω ∧ ω ≤ 5 / 4 := by
  sorry


end NUMINAMATH_CALUDE_omega_range_l3698_369865


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l3698_369882

theorem rectangular_plot_area 
  (breadth : ℝ) 
  (length : ℝ) 
  (h1 : breadth = 12)
  (h2 : length = 3 * breadth) : 
  breadth * length = 432 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l3698_369882


namespace NUMINAMATH_CALUDE_min_value_m2_plus_n2_l3698_369895

theorem min_value_m2_plus_n2 (m n : ℝ) (hm : m ≠ 0) :
  let f := λ x : ℝ => m * x^2 + (2*n + 1) * x - m - 2
  (∃ x ∈ Set.Icc 3 4, f x = 0) →
  (∀ a b : ℝ, (∃ x ∈ Set.Icc 3 4, a * x^2 + (2*b + 1) * x - a - 2 = 0) → a^2 + b^2 ≥ 1/100) ∧
  (∃ a b : ℝ, (∃ x ∈ Set.Icc 3 4, a * x^2 + (2*b + 1) * x - a - 2 = 0) ∧ a^2 + b^2 = 1/100) :=
by sorry

end NUMINAMATH_CALUDE_min_value_m2_plus_n2_l3698_369895


namespace NUMINAMATH_CALUDE_als_original_portion_l3698_369845

theorem als_original_portion (total_initial : ℝ) (total_final : ℝ) (al_loss : ℝ) 
  (h1 : total_initial = 1200)
  (h2 : total_final = 1800)
  (h3 : al_loss = 200) :
  ∃ (al betty clare : ℝ),
    al + betty + clare = total_initial ∧
    al - al_loss + 3 * betty + 3 * clare = total_final ∧
    al = 800 := by
  sorry

end NUMINAMATH_CALUDE_als_original_portion_l3698_369845


namespace NUMINAMATH_CALUDE_tan_630_undefined_l3698_369873

theorem tan_630_undefined :
  ¬∃ (x : ℝ), Real.tan (630 * π / 180) = x :=
by
  sorry


end NUMINAMATH_CALUDE_tan_630_undefined_l3698_369873


namespace NUMINAMATH_CALUDE_max_rectangle_area_l3698_369809

/-- The maximum area of a rectangle with integer dimensions and perimeter 34 cm is 72 square cm. -/
theorem max_rectangle_area : ∀ l w : ℕ, 
  2 * l + 2 * w = 34 → 
  l * w ≤ 72 :=
by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l3698_369809


namespace NUMINAMATH_CALUDE_limit_of_exponential_sine_l3698_369827

theorem limit_of_exponential_sine (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧
  ∀ x : ℝ, 0 < |x - 3| ∧ |x - 3| < δ →
    |(2 - x / 3)^(Real.sin (π * x)) - 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_exponential_sine_l3698_369827


namespace NUMINAMATH_CALUDE_net_population_increase_per_day_l3698_369898

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate in people per second -/
def birth_rate : ℚ := 7 / 2

/-- Represents the death rate in people per second -/
def death_rate : ℚ := 2 / 2

/-- Represents the net population increase per second -/
def net_increase_per_second : ℚ := birth_rate - death_rate

/-- Theorem stating the net population increase in one day -/
theorem net_population_increase_per_day :
  ⌊(net_increase_per_second * seconds_per_day : ℚ)⌋ = 216000 := by
  sorry

end NUMINAMATH_CALUDE_net_population_increase_per_day_l3698_369898


namespace NUMINAMATH_CALUDE_tire_comparison_l3698_369811

def type_A : List ℕ := [94, 96, 99, 99, 105, 107]
def type_B : List ℕ := [95, 95, 98, 99, 104, 109]

def mode (l : List ℕ) : ℕ := sorry
def range (l : List ℕ) : ℕ := sorry
def mean (l : List ℕ) : ℚ := sorry
def variance (l : List ℕ) : ℚ := sorry

theorem tire_comparison :
  (mode type_A > mode type_B) ∧
  (range type_A < range type_B) ∧
  (mean type_A = mean type_B) ∧
  (variance type_A < variance type_B) := by sorry

end NUMINAMATH_CALUDE_tire_comparison_l3698_369811


namespace NUMINAMATH_CALUDE_sum_of_divisors_450_prime_factors_l3698_369849

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_450_prime_factors :
  ∃ (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    sum_of_divisors 450 = p * q * r ∧
    ∀ (s : ℕ), Nat.Prime s → s ∣ sum_of_divisors 450 → (s = p ∨ s = q ∨ s = r) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_450_prime_factors_l3698_369849


namespace NUMINAMATH_CALUDE_inequality_proof_l3698_369834

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3698_369834


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3698_369896

def M : ℕ := 36 * 36 * 65 * 275

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 30 := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3698_369896


namespace NUMINAMATH_CALUDE_original_scissors_count_l3698_369847

theorem original_scissors_count (initial_scissors : ℕ) (added_scissors : ℕ) (total_scissors : ℕ) : 
  added_scissors = 13 →
  total_scissors = 52 →
  total_scissors = initial_scissors + added_scissors →
  initial_scissors = 39 := by
sorry

end NUMINAMATH_CALUDE_original_scissors_count_l3698_369847


namespace NUMINAMATH_CALUDE_georges_work_hours_l3698_369821

/-- George's work problem -/
theorem georges_work_hours (hourly_rate : ℕ) (tuesday_hours : ℕ) (total_earnings : ℕ) :
  hourly_rate = 5 →
  tuesday_hours = 2 →
  total_earnings = 45 →
  ∃ (monday_hours : ℕ), monday_hours = 7 ∧ hourly_rate * (monday_hours + tuesday_hours) = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_georges_work_hours_l3698_369821


namespace NUMINAMATH_CALUDE_smallest_integer_in_consecutive_even_set_l3698_369805

theorem smallest_integer_in_consecutive_even_set (n : ℤ) : 
  n % 2 = 0 ∧ 
  (n + 8 < 3 * ((n + (n + 2) + (n + 4) + (n + 6) + (n + 8)) / 5)) →
  n = 0 ∧ ∀ m : ℤ, (m % 2 = 0 ∧ 
    m + 8 < 3 * ((m + (m + 2) + (m + 4) + (m + 6) + (m + 8)) / 5)) →
    m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_consecutive_even_set_l3698_369805


namespace NUMINAMATH_CALUDE_usamo_page_count_l3698_369869

theorem usamo_page_count (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ+) :
  (((a₁ : ℝ) + 1) / 2 + ((a₂ : ℝ) + 1) / 2 + ((a₃ : ℝ) + 1) / 2 + 
   ((a₄ : ℝ) + 1) / 2 + ((a₅ : ℝ) + 1) / 2 + ((a₆ : ℝ) + 1) / 2) = 2017 →
  (a₁ : ℕ) + a₂ + a₃ + a₄ + a₅ + a₆ = 4028 := by
  sorry

end NUMINAMATH_CALUDE_usamo_page_count_l3698_369869


namespace NUMINAMATH_CALUDE_k_value_l3698_369878

theorem k_value : ∃ k : ℝ, (24 / k = 4) ∧ (k = 6) := by
  sorry

end NUMINAMATH_CALUDE_k_value_l3698_369878


namespace NUMINAMATH_CALUDE_incenter_representation_l3698_369838

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle with vertices P, Q, R and side lengths p, q, r -/
structure Triangle where
  P : Point2D
  Q : Point2D
  R : Point2D
  p : ℝ
  q : ℝ
  r : ℝ

/-- The incenter of a triangle -/
def incenter (t : Triangle) : Point2D := sorry

/-- Theorem: The incenter of the specific triangle can be represented as a linear combination
    of its vertices with coefficients (1/3, 1/4, 5/12) -/
theorem incenter_representation (t : Triangle) 
  (h1 : t.p = 8) (h2 : t.q = 6) (h3 : t.r = 10) : 
  ∃ (J : Point2D), J = incenter t ∧ 
    J.x = (1/3) * t.P.x + (1/4) * t.Q.x + (5/12) * t.R.x ∧
    J.y = (1/3) * t.P.y + (1/4) * t.Q.y + (5/12) * t.R.y :=
sorry

end NUMINAMATH_CALUDE_incenter_representation_l3698_369838


namespace NUMINAMATH_CALUDE_max_value_of_f_l3698_369859

/-- Given a function f(x) = x^3 - 3ax + 2 where x = 2 is an extremum point,
    prove that the maximum value of f(x) is 18 -/
theorem max_value_of_f (a : ℝ) (f : ℝ → ℝ) (h1 : f = fun x ↦ x^3 - 3*a*x + 2) 
    (h2 : ∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≥ f 2 ∨ f x ≤ f 2) :
  (⨆ x, f x) = 18 := by
  sorry


end NUMINAMATH_CALUDE_max_value_of_f_l3698_369859


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l3698_369855

theorem smallest_fraction_between (p q : ℕ+) : 
  (5 : ℚ) / 9 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (4 : ℚ) / 7 ∧ 
  (∀ (p' q' : ℕ+), (5 : ℚ) / 9 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (4 : ℚ) / 7 → q ≤ q') →
  q - p = 7 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l3698_369855


namespace NUMINAMATH_CALUDE_M_geq_N_l3698_369829

theorem M_geq_N (x : ℝ) : 
  let M := 2 * x^2 - 12 * x + 15
  let N := x^2 - 8 * x + 11
  M ≥ N := by
sorry

end NUMINAMATH_CALUDE_M_geq_N_l3698_369829


namespace NUMINAMATH_CALUDE_sequence_difference_l3698_369862

theorem sequence_difference (a : ℕ → ℕ) : 
  (∀ n m : ℕ, n < m → a n < a m) →  -- strictly increasing
  (∀ n : ℕ, n ≥ 1 → a n ≥ 1) →     -- a_n ≥ 1 for n ≥ 1
  (∀ n : ℕ, n ≥ 1 → a (a n) = 3 * n) →  -- a_{a_n} = 3n for n ≥ 1
  a 2021 - a 1999 = 66 := by
sorry

end NUMINAMATH_CALUDE_sequence_difference_l3698_369862


namespace NUMINAMATH_CALUDE_remainder_6n_mod_4_l3698_369871

theorem remainder_6n_mod_4 (n : ℤ) (h : n % 4 = 1) : (6 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_6n_mod_4_l3698_369871


namespace NUMINAMATH_CALUDE_intersect_at_two_points_l3698_369853

/-- The first function representing y = 2x^2 - x + 3 --/
def f (x : ℝ) : ℝ := 2 * x^2 - x + 3

/-- The second function representing y = -x^2 + x + 5 --/
def g (x : ℝ) : ℝ := -x^2 + x + 5

/-- The difference function between f and g --/
def h (x : ℝ) : ℝ := f x - g x

/-- Theorem stating that the graphs of f and g intersect at exactly two distinct points --/
theorem intersect_at_two_points : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ h x₁ = 0 ∧ h x₂ = 0 ∧ ∀ x, h x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_intersect_at_two_points_l3698_369853


namespace NUMINAMATH_CALUDE_not_all_prime_l3698_369820

theorem not_all_prime (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_div_a : a ∣ b + c + b * c)
  (h_div_b : b ∣ c + a + c * a)
  (h_div_c : c ∣ a + b + a * b) :
  ¬(Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c) := by
  sorry

end NUMINAMATH_CALUDE_not_all_prime_l3698_369820


namespace NUMINAMATH_CALUDE_point_on_number_line_l3698_369800

/-- Given two points A and B on a number line where A represents -3 and B is 7 units to the right of A, 
    prove that B represents 4. -/
theorem point_on_number_line (A B : ℝ) : A = -3 ∧ B = A + 7 → B = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_number_line_l3698_369800


namespace NUMINAMATH_CALUDE_yuna_has_most_apples_l3698_369825

def jungkook_apples : ℚ := 6 / 3
def yoongi_apples : ℕ := 4
def yuna_apples : ℕ := 5

theorem yuna_has_most_apples : 
  (jungkook_apples : ℝ) < yuna_apples ∧ yoongi_apples < yuna_apples :=
by sorry

end NUMINAMATH_CALUDE_yuna_has_most_apples_l3698_369825


namespace NUMINAMATH_CALUDE_keyboard_printer_cost_l3698_369880

/-- The total cost of keyboards and printers -/
def total_cost (num_keyboards : ℕ) (num_printers : ℕ) (keyboard_price : ℕ) (printer_price : ℕ) : ℕ :=
  num_keyboards * keyboard_price + num_printers * printer_price

/-- Theorem stating that the total cost of 15 keyboards at $20 each and 25 printers at $70 each is $2050 -/
theorem keyboard_printer_cost : total_cost 15 25 20 70 = 2050 := by
  sorry

end NUMINAMATH_CALUDE_keyboard_printer_cost_l3698_369880


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3698_369830

/-- The eccentricity of a hyperbola passing through the focus of a specific parabola -/
theorem hyperbola_eccentricity (a : ℝ) (h_a : a > 0) :
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 = 1
  let parabola := fun (x y : ℝ) => y^2 = 8 * x
  let focus : ℝ × ℝ := (2, 0)
  hyperbola focus.1 focus.2 →
  let c := Real.sqrt (a^2 + 1)
  c / a = Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3698_369830


namespace NUMINAMATH_CALUDE_unique_valid_number_l3698_369877

/-- A function that returns the digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a 3-digit number -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The set of 3-digit numbers with digit sum 25 that are even -/
def validNumbers : Set ℕ := {n : ℕ | isThreeDigit n ∧ digitSum n = 25 ∧ Even n}

theorem unique_valid_number : ∃! n : ℕ, n ∈ validNumbers := by sorry

end NUMINAMATH_CALUDE_unique_valid_number_l3698_369877


namespace NUMINAMATH_CALUDE_scientific_notation_130_billion_l3698_369887

theorem scientific_notation_130_billion : 130000000000 = 1.3 * (10 ^ 11) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_130_billion_l3698_369887


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l3698_369812

theorem triangle_max_perimeter :
  ∀ x : ℕ,
    x > 0 →
    x < 17 →
    x + 4*x > 17 →
    x + 17 > 4*x →
    ∀ y : ℕ,
      y > 0 →
      y < 17 →
      y + 4*y > 17 →
      y + 17 > 4*y →
      x + 4*x + 17 ≥ y + 4*y + 17 →
      x + 4*x + 17 ≤ 42 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l3698_369812


namespace NUMINAMATH_CALUDE_circle_and_tangent_lines_l3698_369897

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (m : ℝ), (x - m)^2 + (y - 3*m)^2 = (1 - m)^2 + (6 - 3*m)^2 ∧
             (x - m)^2 + (y - 3*m)^2 = (-2 - m)^2 + (3 - 3*m)^2

-- Define the line 3x-y=0
def center_line (x y : ℝ) : Prop := 3*x - y = 0

-- Define the point P
def point_P : ℝ × ℝ := (4, 1)

-- Theorem statement
theorem circle_and_tangent_lines :
  ∃ (x₀ y₀ r : ℝ),
    (∀ x y, circle_C x y ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧
    center_line x₀ y₀ ∧
    ((x₀ - 1)^2 + (y₀ - 3)^2 = 9) ∧
    (∀ x y, (5*x - 12*y - 8 = 0 ∨ x = 4) →
      ((x - x₀)^2 + (y - y₀)^2 = r^2 ∧
       ((x - 4)^2 + (y - 1)^2) * r^2 = ((x - x₀)*(4 - x₀) + (y - y₀)*(1 - y₀))^2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_lines_l3698_369897


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3698_369822

variable (x y a : ℝ)

theorem simplify_expression_1 : (x + y)^2 + y * (3 * x - y) = x^2 + 5 * x * y := by sorry

theorem simplify_expression_2 (h1 : a ≠ 1) (h2 : a ≠ 4) (h3 : a ≠ -4) :
  ((4 - a^2) / (a - 1) + a) / ((a^2 - 16) / (a - 1)) = -1 / (a + 4) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3698_369822


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l3698_369837

theorem square_sum_geq_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l3698_369837


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3698_369861

theorem trigonometric_identities :
  (2 * Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1 / 2) ∧
  (Real.cos (45 * π / 180) * Real.cos (15 * π / 180) - Real.sin (45 * π / 180) * Real.sin (15 * π / 180) = 1 / 2) ∧
  ((Real.tan (77 * π / 180) - Real.tan (32 * π / 180)) / (2 * (1 + Real.tan (77 * π / 180) * Real.tan (32 * π / 180))) = 1 / 2) :=
by sorry


end NUMINAMATH_CALUDE_trigonometric_identities_l3698_369861


namespace NUMINAMATH_CALUDE_least_exponent_for_divisibility_l3698_369868

/-- The function that calculates the sum of powers for the given exponent -/
def sumOfPowers (a : ℕ+) : ℕ :=
  (1995 : ℕ) ^ a.val + (1996 : ℕ) ^ a.val + (1997 : ℕ) ^ a.val

/-- The property that the sum is divisible by 10 -/
def isDivisibleBy10 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10 * k

/-- The main theorem statement -/
theorem least_exponent_for_divisibility :
  (∀ a : ℕ+, a < 2 → ¬(isDivisibleBy10 (sumOfPowers a))) ∧
  isDivisibleBy10 (sumOfPowers 2) := by
  sorry

#check least_exponent_for_divisibility

end NUMINAMATH_CALUDE_least_exponent_for_divisibility_l3698_369868


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l3698_369814

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Theorem stating that i^1234 + i^1235 + i^1236 + i^1237 = 0 -/
theorem sum_of_powers_of_i_is_zero : i^1234 + i^1235 + i^1236 + i^1237 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l3698_369814


namespace NUMINAMATH_CALUDE_zhang_hua_cards_l3698_369844

-- Define the variables
variable (x y z : ℕ)

-- State the theorem
theorem zhang_hua_cards :
  (Nat.lcm (Nat.lcm x y) z = 60) →
  (Nat.gcd x y = 4) →
  (Nat.gcd y z = 3) →
  (x = 4 ∨ x = 20) :=
by
  sorry

end NUMINAMATH_CALUDE_zhang_hua_cards_l3698_369844


namespace NUMINAMATH_CALUDE_triangle_properties_l3698_369818

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 5 ∧
  t.b^2 + t.c^2 - Real.sqrt 2 * t.b * t.c = 25 ∧
  Real.cos t.B = 3/5

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.A = Real.pi/4 ∧ t.c = 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3698_369818


namespace NUMINAMATH_CALUDE_perpendicular_planes_parallel_perpendicular_planes_line_l3698_369833

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (subset : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)
variable (intersection : Plane → Plane → Line)

-- Theorem 1
theorem perpendicular_planes_parallel 
  (m n l : Line) (α β : Plane) :
  line_perpendicular_plane l α →
  line_perpendicular_plane m β →
  parallel l m →
  plane_parallel α β := by sorry

-- Theorem 2
theorem perpendicular_planes_line 
  (m n : Line) (α β : Plane) :
  plane_perpendicular α β →
  intersection α β = m →
  subset n β →
  perpendicular n m →
  line_perpendicular_plane n α := by sorry

end NUMINAMATH_CALUDE_perpendicular_planes_parallel_perpendicular_planes_line_l3698_369833


namespace NUMINAMATH_CALUDE_sum_inequality_l3698_369846

theorem sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x * y + y * z + z * x = 1) :
  (1 + x^2 * y^2) / (x + y)^2 + (1 + y^2 * z^2) / (y + z)^2 + (1 + z^2 * x^2) / (z + x)^2 ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3698_369846


namespace NUMINAMATH_CALUDE_parallel_transitivity_counterexample_l3698_369870

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- State the theorem
theorem parallel_transitivity_counterexample 
  (l n : Line) (α : Plane) : 
  ¬(∀ l n α, parallelLinePlane l α → parallelLinePlane n α → parallelLine l n) :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_counterexample_l3698_369870


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l3698_369894

/-- Represents the number of fish in a sample, given the total population, 
    sample size, and the count of a specific type of fish in the population -/
def stratified_sample_count (population : ℕ) (sample_size : ℕ) (fish_count : ℕ) : ℕ :=
  (fish_count * sample_size) / population

/-- Proves that in a stratified sample of size 20 drawn from a population of 200 fish, 
    where silver carp make up 20 of the population and common carp make up 40 of the population, 
    the number of silver carp and common carp together in the sample is 6 -/
theorem stratified_sample_theorem (total_population : ℕ) (sample_size : ℕ) 
  (silver_carp_count : ℕ) (common_carp_count : ℕ) 
  (h1 : total_population = 200) 
  (h2 : sample_size = 20) 
  (h3 : silver_carp_count = 20) 
  (h4 : common_carp_count = 40) : 
  stratified_sample_count total_population sample_size silver_carp_count + 
  stratified_sample_count total_population sample_size common_carp_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l3698_369894


namespace NUMINAMATH_CALUDE_factor_sum_l3698_369813

theorem factor_sum (P Q : ℝ) : 
  (∃ c d : ℝ, (X^2 - 3*X + 7) * (X^2 + c*X + d) = X^4 + P*X^2 + Q) →
  P + Q = 54 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l3698_369813


namespace NUMINAMATH_CALUDE_distance_calculation_l3698_369892

/-- Given a journey time of 8 hours and an average speed of 23 miles per hour,
    the distance traveled is 184 miles. -/
theorem distance_calculation (journey_time : ℝ) (average_speed : ℝ) 
  (h1 : journey_time = 8)
  (h2 : average_speed = 23) :
  journey_time * average_speed = 184 := by
  sorry

end NUMINAMATH_CALUDE_distance_calculation_l3698_369892


namespace NUMINAMATH_CALUDE_birthday_money_calculation_l3698_369879

/-- The amount of money Sam spent on baseball gear -/
def amount_spent : ℕ := 64

/-- The amount of money Sam had left over -/
def amount_left : ℕ := 23

/-- The total amount of money Sam received for his birthday -/
def total_amount : ℕ := amount_spent + amount_left

/-- Theorem stating that the total amount Sam received is the sum of what he spent and what he had left -/
theorem birthday_money_calculation : total_amount = 87 := by
  sorry

end NUMINAMATH_CALUDE_birthday_money_calculation_l3698_369879


namespace NUMINAMATH_CALUDE_chess_team_selection_l3698_369801

def boys : ℕ := 10
def girls : ℕ := 12
def team_boys : ℕ := 5
def team_girls : ℕ := 3

theorem chess_team_selection :
  (Nat.choose boys team_boys) * (Nat.choose girls team_girls) = 55440 :=
by sorry

end NUMINAMATH_CALUDE_chess_team_selection_l3698_369801


namespace NUMINAMATH_CALUDE_quarter_equals_point_two_five_l3698_369810

theorem quarter_equals_point_two_five : (1 : ℚ) / 4 = 0.250000000 := by
  sorry

end NUMINAMATH_CALUDE_quarter_equals_point_two_five_l3698_369810


namespace NUMINAMATH_CALUDE_rain_probability_l3698_369857

theorem rain_probability (p_saturday p_sunday : ℝ) 
  (h_saturday : p_saturday = 0.6)
  (h_sunday : p_sunday = 0.4)
  (h_independent : True) -- Assumption of independence
  : p_saturday * (1 - p_sunday) + (1 - p_saturday) * p_sunday = 0.52 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l3698_369857


namespace NUMINAMATH_CALUDE_floor_ceiling_product_l3698_369815

theorem floor_ceiling_product : ⌊(3.999 : ℝ)⌋ * ⌈(0.002 : ℝ)⌉ = 3 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_product_l3698_369815


namespace NUMINAMATH_CALUDE_displacement_increment_formula_l3698_369850

/-- The equation of motion for an object -/
def equation_of_motion (t : ℝ) : ℝ := 2 * t^2

/-- The increment of displacement -/
def displacement_increment (d : ℝ) : ℝ :=
  equation_of_motion (2 + d) - equation_of_motion 2

theorem displacement_increment_formula (d : ℝ) :
  displacement_increment d = 8 * d + 2 * d^2 := by
  sorry

end NUMINAMATH_CALUDE_displacement_increment_formula_l3698_369850


namespace NUMINAMATH_CALUDE_tan_double_angle_l3698_369875

theorem tan_double_angle (θ : Real) (h : Real.tan θ = 2) : Real.tan (2 * θ) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l3698_369875


namespace NUMINAMATH_CALUDE_problem_1_l3698_369828

theorem problem_1 : Real.sin (30 * π / 180) + |(-1)| - (Real.sqrt 3 - Real.pi)^0 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3698_369828


namespace NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l3698_369832

theorem tangent_perpendicular_to_line (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.cos x
  let f' : ℝ → ℝ := λ x ↦ -Real.sin x
  let tangent_slope : ℝ := f' (π/6)
  tangent_slope * a = -1 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l3698_369832


namespace NUMINAMATH_CALUDE_tan_690_degrees_l3698_369858

theorem tan_690_degrees : Real.tan (690 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_690_degrees_l3698_369858


namespace NUMINAMATH_CALUDE_max_value_on_circle_l3698_369889

theorem max_value_on_circle (x y : ℝ) :
  (x - 1)^2 + y^2 = 1 →
  ∃ (max : ℝ), (∀ (x' y' : ℝ), (x' - 1)^2 + y'^2 = 1 → 2*x' + y' ≤ max) ∧ max = Real.sqrt 5 + 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l3698_369889


namespace NUMINAMATH_CALUDE_complement_of_A_l3698_369899

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4}

-- Define set A
def A : Finset Nat := {1, 3}

-- Theorem statement
theorem complement_of_A :
  (U \ A) = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3698_369899


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3698_369841

def f (a x : ℝ) : ℝ := -x^2 + 2*a*x + 2*a

theorem quadratic_function_properties (a : ℝ) :
  (f a (-1) = -1 → a = 0) ∧
  (f a 3 = -1 → (∀ x ∈ Set.Icc (-2) 3, f a x ≤ 3) ∧ (∃ x ∈ Set.Icc (-2) 3, f a x = -6)) ∧
  (∃ x y : ℝ, x ≠ y ∧ f a x = -1 ∧ f a y = -1 ∧ |x - y| = |2*a + 2|) ∧
  (∃! x : ℝ, x ∈ Set.Icc (a - 1) (2*a + 3) ∧ f a x = -1 ↔ a ≥ 0 ∨ a = -1 ∨ a = -2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3698_369841


namespace NUMINAMATH_CALUDE_stars_per_jar_l3698_369819

theorem stars_per_jar (stars_made : ℕ) (bottles_to_fill : ℕ) (stars_to_make : ℕ) : 
  stars_made = 33 →
  bottles_to_fill = 4 →
  stars_to_make = 307 →
  (stars_made + stars_to_make) / bottles_to_fill = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_stars_per_jar_l3698_369819


namespace NUMINAMATH_CALUDE_composite_product_properties_l3698_369804

def first_five_composites : List Nat := [4, 6, 8, 9, 10]

def product_of_composites : Nat := first_five_composites.prod

theorem composite_product_properties :
  (product_of_composites % 10 = 0) ∧
  (Nat.digits 10 product_of_composites).sum = 18 := by
  sorry

end NUMINAMATH_CALUDE_composite_product_properties_l3698_369804


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3698_369806

theorem quadratic_one_root (m : ℝ) : 
  (∃! x : ℝ, x^2 + 6*m*x + 2*m = 0) ↔ m = 2/9 := by sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3698_369806


namespace NUMINAMATH_CALUDE_paul_sandwich_consumption_l3698_369836

def sandwiches_per_cycle : ℕ := 2 + 4 + 8

def study_days : ℕ := 6

def cycles : ℕ := study_days / 3

theorem paul_sandwich_consumption :
  cycles * sandwiches_per_cycle = 28 := by
  sorry

end NUMINAMATH_CALUDE_paul_sandwich_consumption_l3698_369836


namespace NUMINAMATH_CALUDE_horner_method_f_at_3_l3698_369848

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 + 2x^3 + 3x^2 + x + 1 -/
def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

/-- Coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [1, 0, 2, 3, 1, 1]

theorem horner_method_f_at_3 :
  horner f_coeffs 3 = 36 := by
  sorry

#eval horner f_coeffs 3  -- This should output 36
#eval f 3  -- This should also output 36

end NUMINAMATH_CALUDE_horner_method_f_at_3_l3698_369848


namespace NUMINAMATH_CALUDE_v2_value_for_f_at_2_l3698_369881

def f (x : ℝ) : ℝ := 2 * x^5 - 3 * x + 2 * x^2 - x + 5

def qin_jiushao_v2 (a b c d e : ℝ) (x : ℝ) : ℝ :=
  (a * x + b) * x + c

theorem v2_value_for_f_at_2 :
  let a := 2
  let b := 3
  let c := 0
  qin_jiushao_v2 a b c 5 (-4) 2 = 14 := by sorry

end NUMINAMATH_CALUDE_v2_value_for_f_at_2_l3698_369881


namespace NUMINAMATH_CALUDE_determinant_invariant_impossibility_of_transformation_l3698_369808

def Sequence := Fin 4 → ℤ

def initial_sequence : Sequence := ![1, 2, 3, 4]
def target_sequence : Sequence := ![3, 4, 5, 7]

def determinant (s : Sequence) : ℤ :=
  s 0 * s 3 - s 1 * s 2

def transform_1 (s : Sequence) : Sequence :=
  ![s 2, s 3, s 0, s 1]

def transform_2 (s : Sequence) : Sequence :=
  ![s 1, s 0, s 3, s 2]

def transform_3 (s : Sequence) (n : ℤ) : Sequence :=
  ![s 0 + n * s 2, s 1 + n * s 3, s 2, s 3]

def transform_4 (s : Sequence) (n : ℤ) : Sequence :=
  ![s 0 + n * s 1, s 1, s 2 + n * s 3, s 3]

theorem determinant_invariant (s : Sequence) :
  (∀ t : Sequence, t = transform_1 s ∨ t = transform_2 s ∨
   (∃ n : ℤ, t = transform_3 s n) ∨ (∃ n : ℤ, t = transform_4 s n) →
   abs (determinant t) = abs (determinant s)) :=
sorry

theorem impossibility_of_transformation :
  ¬ ∃ (steps : List (Sequence → Sequence)),
    steps.foldl (λ acc f => f acc) initial_sequence = target_sequence :=
sorry

end NUMINAMATH_CALUDE_determinant_invariant_impossibility_of_transformation_l3698_369808


namespace NUMINAMATH_CALUDE_root_equation_problem_l3698_369864

theorem root_equation_problem (b c x₁ x₂ : ℝ) (y : ℝ) : 
  x₁ ≠ x₂ →
  (x₁^2 + 5*b*x₁ + c = 0) →
  (x₂^2 + 5*b*x₂ + c = 0) →
  (y^2 + 2*x₁*y + 2*x₂ = 0) →
  (y^2 + 2*x₂*y + 2*x₁ = 0) →
  b = 1/10 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l3698_369864


namespace NUMINAMATH_CALUDE_exhibition_hall_probability_l3698_369866

/-- The probability of entering from entrance A and exiting from exit F in an exhibition hall -/
theorem exhibition_hall_probability :
  let num_entrances : ℕ := 2
  let num_exits : ℕ := 3
  let prob_entrance_A : ℚ := 1 / num_entrances
  let prob_exit_F : ℚ := 1 / num_exits
  prob_entrance_A * prob_exit_F = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_exhibition_hall_probability_l3698_369866


namespace NUMINAMATH_CALUDE_total_length_l3698_369874

def problem (rubber pen pencil marker ruler : ℝ) : Prop :=
  pen = rubber + 3 ∧
  pencil = pen + 2 ∧
  pencil = 12 ∧
  ruler = 3 * rubber ∧
  marker = (pen + rubber + pencil) / 3 ∧
  marker = ruler / 2

theorem total_length (rubber pen pencil marker ruler : ℝ) 
  (h : problem rubber pen pencil marker ruler) : 
  rubber + pen + pencil + marker + ruler = 60.5 := by
  sorry

end NUMINAMATH_CALUDE_total_length_l3698_369874


namespace NUMINAMATH_CALUDE_arithmetic_sequences_difference_l3698_369888

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) : ℕ :=
  let n := aₙ - a₁ + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequences_difference : 
  arithmetic_sum 2001 2093 - arithmetic_sum 201 293 - arithmetic_sum 1 93 = 165044 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_difference_l3698_369888


namespace NUMINAMATH_CALUDE_water_depth_is_12_feet_l3698_369891

/-- The height of Ron in feet -/
def ron_height : ℝ := 14

/-- The difference in height between Ron and Dean in feet -/
def height_difference : ℝ := 8

/-- The height of Dean in feet -/
def dean_height : ℝ := ron_height - height_difference

/-- The depth of the water as a multiple of Dean's height -/
def water_depth_factor : ℝ := 2

/-- The depth of the water in feet -/
def water_depth : ℝ := water_depth_factor * dean_height

theorem water_depth_is_12_feet : water_depth = 12 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_is_12_feet_l3698_369891


namespace NUMINAMATH_CALUDE_largest_value_l3698_369872

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  a^2 + b^2 = max (max (max (a^2 + b^2) (2*a*b)) a) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l3698_369872


namespace NUMINAMATH_CALUDE_count_pairs_20_l3698_369863

def count_pairs (n : ℕ) : ℕ :=
  (n - 11) * (n - 11 + 1) / 2

theorem count_pairs_20 :
  count_pairs 20 = 45 :=
sorry

end NUMINAMATH_CALUDE_count_pairs_20_l3698_369863


namespace NUMINAMATH_CALUDE_gcd_of_numbers_l3698_369831

theorem gcd_of_numbers : Nat.gcd 128 (Nat.gcd 144 (Nat.gcd 480 450)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_numbers_l3698_369831


namespace NUMINAMATH_CALUDE_group_size_from_shoes_l3698_369824

/-- Given a group of people where the total number of shoes is 20 and each person has 2 shoes,
    prove that the number of people in the group is 10. -/
theorem group_size_from_shoes (total_shoes : ℕ) (shoes_per_person : ℕ) 
    (h1 : total_shoes = 20) (h2 : shoes_per_person = 2) : 
    total_shoes / shoes_per_person = 10 := by
  sorry

end NUMINAMATH_CALUDE_group_size_from_shoes_l3698_369824


namespace NUMINAMATH_CALUDE_extra_bananas_proof_l3698_369835

/-- Calculates the number of extra bananas each child receives when some children are absent -/
def extra_bananas (total_children : ℕ) (absent_children : ℕ) : ℕ :=
  absent_children

theorem extra_bananas_proof (total_children : ℕ) (absent_children : ℕ) 
  (h1 : total_children = 700) 
  (h2 : absent_children = 350) :
  extra_bananas total_children absent_children = absent_children :=
by
  sorry

#eval extra_bananas 700 350

end NUMINAMATH_CALUDE_extra_bananas_proof_l3698_369835


namespace NUMINAMATH_CALUDE_fraction_simplification_l3698_369823

theorem fraction_simplification (x y z : ℝ) (h : x + y + z ≠ 0) :
  (x^2 + y^2 - z^2 + 2*x*y) / (x^2 + z^2 - y^2 + 2*x*z) = (x + y - z) / (x + z - y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3698_369823


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3698_369893

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3698_369893
