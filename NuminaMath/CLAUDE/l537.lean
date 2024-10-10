import Mathlib

namespace common_point_theorem_l537_53743

/-- Represents a line with equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the coefficients of a line form an arithmetic progression with common difference a/2 -/
def Line.isArithmeticProgression (l : Line) : Prop :=
  l.b = l.a + l.a/2 ∧ l.c = l.a + 2*(l.a/2)

/-- Checks if a point (x, y) lies on a given line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

theorem common_point_theorem :
  ∀ l : Line, l.isArithmeticProgression → l.containsPoint 0 (4/3) :=
sorry

end common_point_theorem_l537_53743


namespace quadratic_inequality_solution_l537_53740

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 + x - 12 ≤ 0 ∧ x ≥ -4 → -4 ≤ x ∧ x ≤ 3 := by
sorry

end quadratic_inequality_solution_l537_53740


namespace product_125_sum_31_l537_53741

theorem product_125_sum_31 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 125 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 31 := by
sorry

end product_125_sum_31_l537_53741


namespace cube_edge_is_60_l537_53759

-- Define the volume of the rectangular cuboid-shaped cabinet
def cuboid_volume : ℝ := 420000

-- Define the volume difference between the cabinets
def volume_difference : ℝ := 204000

-- Define the volume of the cube-shaped cabinet
def cube_volume : ℝ := cuboid_volume - volume_difference

-- Define the function to calculate the edge length of a cube given its volume
def cube_edge_length (volume : ℝ) : ℝ := volume ^ (1/3)

-- Theorem statement
theorem cube_edge_is_60 :
  cube_edge_length cube_volume = 60 := by
  sorry

end cube_edge_is_60_l537_53759


namespace least_number_satisfying_conditions_l537_53736

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def leaves_remainder_2 (n : ℕ) (d : ℕ) : Prop := n % d = 2

def satisfies_conditions (n : ℕ) : Prop :=
  is_divisible_by_9 n ∧
  (∀ d : ℕ, 3 ≤ d ∧ d ≤ 7 → leaves_remainder_2 n d)

theorem least_number_satisfying_conditions :
  satisfies_conditions 6302 ∧
  ∀ m : ℕ, m < 6302 → ¬(satisfies_conditions m) :=
sorry

end least_number_satisfying_conditions_l537_53736


namespace number_of_girls_in_school_l537_53703

/-- Proves the number of girls in a school with given conditions -/
theorem number_of_girls_in_school (total : ℕ) (girls : ℕ) (boys : ℕ) 
  (h_total : total = 300)
  (h_ratio : girls * 8 = boys * 5)
  (h_sum : girls + boys = total) : 
  girls = 116 := by sorry

end number_of_girls_in_school_l537_53703


namespace course_selection_schemes_l537_53702

/-- The number of elective courses in each category (physical education and art) -/
def n : ℕ := 4

/-- The total number of different course selection schemes -/
def total_schemes : ℕ := 
  (n.choose 1 * n.choose 1) +  -- Selecting 2 courses (1 from each category)
  (n.choose 2 * n.choose 1) +  -- Selecting 3 courses (2 PE, 1 Art)
  (n.choose 1 * n.choose 2)    -- Selecting 3 courses (1 PE, 2 Art)

theorem course_selection_schemes : total_schemes = 64 := by
  sorry

end course_selection_schemes_l537_53702


namespace closest_to_fraction_l537_53797

def options : List ℝ := [500, 1000, 2000, 2100, 4000]

theorem closest_to_fraction (options : List ℝ) :
  2100 = (options.filter (λ x => ∀ y ∈ options, |850 / 0.42 - x| ≤ |850 / 0.42 - y|)).head! :=
by sorry

end closest_to_fraction_l537_53797


namespace constant_terms_are_like_terms_l537_53758

/-- Two algebraic terms are considered "like terms" if they have the same variables with the same exponents. -/
def like_terms (term1 term2 : String) : Prop := sorry

/-- A constant term is a number without variables. -/
def is_constant_term (term : String) : Prop := sorry

theorem constant_terms_are_like_terms (a b : String) :
  is_constant_term a ∧ is_constant_term b → like_terms a b := by sorry

end constant_terms_are_like_terms_l537_53758


namespace dividing_line_ratio_l537_53794

/-- A trapezoid with given dimensions and a dividing line -/
structure Trapezoid :=
  (base1 : ℝ)
  (base2 : ℝ)
  (leg1 : ℝ)
  (leg2 : ℝ)
  (dividing_ratio : ℝ × ℝ)

/-- The condition that the dividing line creates equal perimeters -/
def equal_perimeters (t : Trapezoid) : Prop :=
  let (m, n) := t.dividing_ratio
  let x := t.base1 + (t.base2 - t.base1) * (m / (m + n))
  t.base1 + m + x + t.leg1 * (m / (m + n)) =
  t.base2 + n + x + t.leg1 * (n / (m + n))

/-- The theorem stating the ratio of the dividing line -/
theorem dividing_line_ratio (t : Trapezoid) 
    (h1 : t.base1 = 3) 
    (h2 : t.base2 = 9) 
    (h3 : t.leg1 = 4) 
    (h4 : t.leg2 = 6) 
    (h5 : equal_perimeters t) : 
    t.dividing_ratio = (4, 1) := by
  sorry


end dividing_line_ratio_l537_53794


namespace angle_sum_pi_half_l537_53792

theorem angle_sum_pi_half (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2)
  (h_eq : (Real.sin α)^4 / (Real.cos β)^2 + (Real.cos α)^4 / (Real.sin β)^2 = 1) :
  α + β = π/2 := by sorry

end angle_sum_pi_half_l537_53792


namespace equal_pairwise_products_l537_53733

theorem equal_pairwise_products (n : ℕ) : 
  (¬ ∃ n : ℕ, n > 0 ∧ n < 1000 ∧ n^2 - 1000*n + 499500 = 0) ∧
  (∃ n : ℕ, n > 0 ∧ n < 10000 ∧ n^2 - 10000*n + 49995000 = 0) := by
  sorry

end equal_pairwise_products_l537_53733


namespace marble_selection_ways_l537_53757

def total_marbles : ℕ := 15
def specific_marbles : ℕ := 4
def marbles_to_choose : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem marble_selection_ways : 
  specific_marbles * choose (total_marbles - specific_marbles) (marbles_to_choose - 1) = 1320 := by
  sorry

end marble_selection_ways_l537_53757


namespace interest_rate_calculation_l537_53788

/-- Calculates the simple interest given principal, time, and rate. -/
def simpleInterest (principal : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  principal * time * rate / 100

theorem interest_rate_calculation (loanB_principal loanC_principal totalInterest : ℚ)
  (loanB_time loanC_time : ℚ) :
  loanB_principal = 5000 →
  loanC_principal = 3000 →
  loanB_time = 2 →
  loanC_time = 4 →
  totalInterest = 3300 →
  ∃ rate : ℚ, 
    simpleInterest loanB_principal loanB_time rate +
    simpleInterest loanC_principal loanC_time rate = totalInterest ∧
    rate = 15 := by
  sorry

end interest_rate_calculation_l537_53788


namespace largest_prime_divisor_test_l537_53761

theorem largest_prime_divisor_test (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1050) :
  Prime n → ∀ p, Prime p ∧ p > 31 → ¬(p ∣ n) :=
by sorry

end largest_prime_divisor_test_l537_53761


namespace larger_integer_problem_l537_53722

theorem larger_integer_problem (x y : ℕ+) 
  (h1 : y - x = 6) 
  (h2 : x * y = 135) : 
  y = 15 := by sorry

end larger_integer_problem_l537_53722


namespace jason_shorts_expenditure_l537_53799

theorem jason_shorts_expenditure (total : ℝ) (jacket : ℝ) (shorts : ℝ) : 
  total = 19.02 → jacket = 4.74 → total = jacket + shorts → shorts = 14.28 := by
  sorry

end jason_shorts_expenditure_l537_53799


namespace bangles_per_box_l537_53791

def total_pairs : ℕ := 240
def num_boxes : ℕ := 20

theorem bangles_per_box :
  (total_pairs * 2) / num_boxes = 24 := by
  sorry

end bangles_per_box_l537_53791


namespace inequality_and_equality_conditions_l537_53700

theorem inequality_and_equality_conditions (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  (1/2 ≤ (a^3 + b^3) / (a^2 + b^2)) ∧ 
  ((a^3 + b^3) / (a^2 + b^2) ≤ 1) ∧
  ((a^3 + b^3) / (a^2 + b^2) = 1/2 ↔ a = 1/2 ∧ b = 1/2) ∧
  ((a^3 + b^3) / (a^2 + b^2) = 1 ↔ (a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 0)) :=
sorry

end inequality_and_equality_conditions_l537_53700


namespace no_integer_solutions_for_fermat_like_equation_l537_53762

theorem no_integer_solutions_for_fermat_like_equation (n : ℕ) (hn : n ≥ 2) :
  ¬∃ (x y z : ℤ), x^2 + y^2 = z^n := by
  sorry

end no_integer_solutions_for_fermat_like_equation_l537_53762


namespace shelby_heavy_rain_time_l537_53712

/-- Represents the speeds and durations of Shelby's scooter ride --/
structure ScooterRide where
  sunnySpeed : ℝ
  lightRainSpeed : ℝ
  heavyRainSpeed : ℝ
  totalDistance : ℝ
  totalTime : ℝ
  heavyRainTime : ℝ

/-- Theorem stating that given the conditions of Shelby's ride, she spent 20 minutes in heavy rain --/
theorem shelby_heavy_rain_time (ride : ScooterRide) 
  (h1 : ride.sunnySpeed = 35)
  (h2 : ride.lightRainSpeed = 25)
  (h3 : ride.heavyRainSpeed = 15)
  (h4 : ride.totalDistance = 50)
  (h5 : ride.totalTime = 100) :
  ride.heavyRainTime = 20 := by
  sorry

#check shelby_heavy_rain_time

end shelby_heavy_rain_time_l537_53712


namespace school_parade_l537_53735

theorem school_parade (a b : ℕ+) : 
  ∃ k : ℕ, a.val * b.val * (a.val^2 - b.val^2) = 3 * k := by
  sorry

end school_parade_l537_53735


namespace antoinette_weight_l537_53786

/-- Proves that Antoinette weighs 79 kilograms given the conditions of the problem -/
theorem antoinette_weight :
  ∀ (rupert antoinette charles : ℝ),
  antoinette = 2 * rupert - 7 →
  charles = (antoinette + rupert) / 2 + 5 →
  rupert + antoinette + charles = 145 →
  antoinette = 79 := by
sorry

end antoinette_weight_l537_53786


namespace one_large_pizza_sufficient_l537_53725

/-- Represents the number of slices in different pizza sizes --/
structure PizzaSizes where
  large : Nat
  medium : Nat
  small : Nat

/-- Represents the number of pizzas ordered for each dietary restriction --/
structure PizzaOrder where
  gluten_free_small : Nat
  dairy_free_medium : Nat
  large : Nat

/-- Calculates if the pizza order is sufficient for both brothers --/
def is_sufficient_order (sizes : PizzaSizes) (order : PizzaOrder) : Prop :=
  let gluten_free_slices := order.gluten_free_small * sizes.small + order.large * sizes.large
  let dairy_free_slices := order.dairy_free_medium * sizes.medium
  gluten_free_slices ≥ 15 ∧ dairy_free_slices ≥ 15

/-- Theorem stating that ordering 1 large pizza is sufficient --/
theorem one_large_pizza_sufficient 
  (sizes : PizzaSizes)
  (h_large : sizes.large = 14)
  (h_medium : sizes.medium = 10)
  (h_small : sizes.small = 8) :
  is_sufficient_order sizes { gluten_free_small := 1, dairy_free_medium := 2, large := 1 } :=
by sorry


end one_large_pizza_sufficient_l537_53725


namespace units_digit_of_sum_l537_53747

theorem units_digit_of_sum (a b : ℕ) : (24^4 + 42^4) % 10 = 2 := by
  sorry

end units_digit_of_sum_l537_53747


namespace tv_discounted_price_l537_53713

def original_price : ℝ := 500.00
def discount1 : ℝ := 0.10
def discount2 : ℝ := 0.20
def discount3 : ℝ := 0.15

def final_price : ℝ := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)

theorem tv_discounted_price : final_price = 306.00 := by
  sorry

end tv_discounted_price_l537_53713


namespace inverse_proportion_l537_53701

/-- Given that x is inversely proportional to y, prove that if x = 5 when y = 15, 
    then x = -5/2 when y = -30 -/
theorem inverse_proportion (x y : ℝ) (h : ∃ k : ℝ, ∀ x y, x * y = k) 
    (h1 : 5 * 15 = x * y) : 
  x * (-30) = 5 * 15 → x = -5/2 := by sorry

end inverse_proportion_l537_53701


namespace smiths_bakery_pies_smiths_bakery_pies_proof_l537_53784

theorem smiths_bakery_pies : ℕ → ℕ → Prop :=
  fun mcgees_pies smiths_pies =>
    mcgees_pies = 16 →
    smiths_pies = mcgees_pies^2 + mcgees_pies^2 / 2 →
    smiths_pies = 384

-- The proof would go here, but we're skipping it as requested
theorem smiths_bakery_pies_proof : smiths_bakery_pies 16 384 := by
  sorry

end smiths_bakery_pies_smiths_bakery_pies_proof_l537_53784


namespace smallest_non_negative_solution_l537_53769

theorem smallest_non_negative_solution (x : ℕ) : 
  (x + 7263 : ℤ) ≡ 3507 [ZMOD 15] ↔ x = 9 ∨ (x > 9 ∧ (x : ℤ) ≡ 9 [ZMOD 15]) := by
  sorry

end smallest_non_negative_solution_l537_53769


namespace sufficient_condition_for_quadratic_inequality_l537_53738

theorem sufficient_condition_for_quadratic_inequality (a : ℝ) :
  (a ≥ 3) →
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^2 - x - a ≤ 0) ∧
  ¬(∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^2 - x - a ≤ 0) → a ≥ 3) :=
by sorry

end sufficient_condition_for_quadratic_inequality_l537_53738


namespace shortest_tangent_length_l537_53790

-- Define the circles
def C1 (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 49
def C2 (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 64

-- Define the tangent line segment
def is_tangent (P Q : ℝ × ℝ) : Prop :=
  C1 P.1 P.2 ∧ C2 Q.1 Q.2 ∧ 
  ∀ R : ℝ × ℝ, (C1 R.1 R.2 ∨ C2 R.1 R.2) → 
    (R.1 - P.1)^2 + (R.2 - P.2)^2 ≤ (Q.1 - P.1)^2 + (Q.2 - P.2)^2

-- State the theorem
theorem shortest_tangent_length :
  ∃ P Q : ℝ × ℝ, is_tangent P Q ∧
    ∀ P' Q' : ℝ × ℝ, is_tangent P' Q' →
      Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) ≤ 
      Real.sqrt ((Q'.1 - P'.1)^2 + (Q'.2 - P'.2)^2) ∧
    Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = Real.sqrt 207 + Real.sqrt 132 :=
sorry

end shortest_tangent_length_l537_53790


namespace middle_three_average_l537_53730

theorem middle_three_average (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →  -- five different positive integers
  (a + b + c + d + e) / 5 = 5 →  -- average is 5
  e - a = 14 →  -- maximum possible difference
  (b + c + d) / 3 = 3 := by  -- average of middle three is 3
sorry

end middle_three_average_l537_53730


namespace wall_building_time_l537_53706

/-- Represents the time taken to build a wall given the number of workers -/
def build_time (workers : ℕ) : ℝ :=
  sorry

/-- The number of workers in the initial scenario -/
def initial_workers : ℕ := 20

/-- The number of days taken in the initial scenario -/
def initial_days : ℝ := 6

/-- The number of workers in the new scenario -/
def new_workers : ℕ := 30

theorem wall_building_time :
  (build_time initial_workers = initial_days) →
  (∀ w₁ w₂ : ℕ, w₁ * build_time w₁ = w₂ * build_time w₂) →
  (build_time new_workers = 4.0) :=
sorry

end wall_building_time_l537_53706


namespace johnnys_third_job_rate_l537_53763

/-- Given Johnny's work schedule and earnings, prove the hourly rate of his third job. -/
theorem johnnys_third_job_rate (hours_job1 hours_job2 hours_job3 : ℕ)
                               (rate_job1 rate_job2 : ℕ)
                               (days : ℕ)
                               (total_earnings : ℕ) :
  hours_job1 = 3 →
  hours_job2 = 2 →
  hours_job3 = 4 →
  rate_job1 = 7 →
  rate_job2 = 10 →
  days = 5 →
  total_earnings = 445 →
  ∃ (rate_job3 : ℕ), 
    rate_job3 = 12 ∧
    total_earnings = (hours_job1 * rate_job1 + hours_job2 * rate_job2 + hours_job3 * rate_job3) * days :=
by sorry

end johnnys_third_job_rate_l537_53763


namespace min_value_sqrt_plus_reciprocal_l537_53711

theorem min_value_sqrt_plus_reciprocal (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 2 / x ≥ 5 ∧
  ∃ y > 0, 3 * Real.sqrt y + 2 / y = 5 :=
sorry

end min_value_sqrt_plus_reciprocal_l537_53711


namespace x_fifth_plus_64x_l537_53734

theorem x_fifth_plus_64x (x : ℝ) (h : x^2 + 4*x = 8) : x^5 + 64*x = 768*x - 1024 := by
  sorry

end x_fifth_plus_64x_l537_53734


namespace only_eleven_not_sum_of_two_primes_l537_53793

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def is_sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p + q

def numbers_to_check : List ℕ := [5, 7, 9, 10, 11]

theorem only_eleven_not_sum_of_two_primes :
  ∀ n ∈ numbers_to_check, n ≠ 11 → is_sum_of_two_primes n ∧
  ¬(is_sum_of_two_primes 11) :=
sorry

end only_eleven_not_sum_of_two_primes_l537_53793


namespace complex_equality_theorem_l537_53787

theorem complex_equality_theorem :
  ∃ (x : ℝ), 
    (Complex.mk (Real.sin x ^ 2) (Real.cos (2 * x)) = Complex.mk (Real.sin x ^ 2) (Real.cos x)) ∧ 
    ((Complex.mk (Real.sin x ^ 2) (Real.cos x) = Complex.I) ∨ 
     (Complex.mk (Real.sin x ^ 2) (Real.cos x) = Complex.mk (3/4) (-1/2))) := by
  sorry

end complex_equality_theorem_l537_53787


namespace monotonically_decreasing_implies_second_or_third_quadrant_l537_53728

/-- A linear function f(x) = kx + b is monotonically decreasing on ℝ -/
def is_monotonically_decreasing (k b : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → k * x + b > k * y + b

/-- The point (k, b) is in the second or third quadrant -/
def is_in_second_or_third_quadrant (k b : ℝ) : Prop :=
  k < 0 ∧ (b > 0 ∨ b < 0)

/-- If a linear function y = kx + b is monotonically decreasing on ℝ,
    then the point (k, b) is in the second or third quadrant -/
theorem monotonically_decreasing_implies_second_or_third_quadrant (k b : ℝ) :
  is_monotonically_decreasing k b → is_in_second_or_third_quadrant k b :=
by sorry

end monotonically_decreasing_implies_second_or_third_quadrant_l537_53728


namespace smallest_divisible_by_72_l537_53756

/-- Concatenates the digits of all positive integers from 1 to n -/
def concatenateDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is divisible by another number -/
def isDivisibleBy (a b : ℕ) : Prop := ∃ k, a = b * k

theorem smallest_divisible_by_72 :
  ∃ (n : ℕ), n > 0 ∧ isDivisibleBy (concatenateDigits n) 72 ∧
  ∀ (m : ℕ), m > 0 ∧ m < n → ¬isDivisibleBy (concatenateDigits m) 72 :=
by sorry

end smallest_divisible_by_72_l537_53756


namespace equation_equivalence_l537_53749

theorem equation_equivalence (x : ℝ) : 6 - (x - 2) / 2 = x ↔ 12 - x + 2 = 2 * x :=
sorry

end equation_equivalence_l537_53749


namespace lucy_apples_per_week_l537_53748

/-- Given the following conditions:
  - Chandler eats 23 apples per week
  - They order 168 apples per month
  - There are 4 weeks in a month
  Prove that Lucy can eat 19 apples per week. -/
theorem lucy_apples_per_week :
  ∀ (chandler_apples_per_week : ℕ) 
    (total_apples_per_month : ℕ) 
    (weeks_per_month : ℕ),
  chandler_apples_per_week = 23 →
  total_apples_per_month = 168 →
  weeks_per_month = 4 →
  ∃ (lucy_apples_per_week : ℕ),
    lucy_apples_per_week = 19 ∧
    lucy_apples_per_week * weeks_per_month + 
    chandler_apples_per_week * weeks_per_month = 
    total_apples_per_month :=
by sorry

end lucy_apples_per_week_l537_53748


namespace least_addition_for_divisibility_l537_53789

theorem least_addition_for_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬ (9 ∣ (4499 + m))) ∧ (9 ∣ (4499 + n)) → n = 1 := by
  sorry

end least_addition_for_divisibility_l537_53789


namespace least_three_digit_multiple_of_11_l537_53724

theorem least_three_digit_multiple_of_11 : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (∃ k : ℕ, n = 11 * k) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (∃ j : ℕ, m = 11 * j) → n ≤ m) ∧
  n = 110 := by
  sorry

end least_three_digit_multiple_of_11_l537_53724


namespace smallest_number_with_remainders_l537_53753

theorem smallest_number_with_remainders : ∃ (n : ℕ), 
  (n > 0) ∧
  (n % 13 = 12) ∧
  (n % 11 = 10) ∧
  (n % 7 = 6) ∧
  (n % 5 = 4) ∧
  (n % 3 = 2) ∧
  (∀ m : ℕ, m > 0 → 
    (m % 13 = 12) ∧
    (m % 11 = 10) ∧
    (m % 7 = 6) ∧
    (m % 5 = 4) ∧
    (m % 3 = 2) → 
    n ≤ m) :=
by
  -- Proof goes here
  sorry

end smallest_number_with_remainders_l537_53753


namespace sine_sum_equality_l537_53714

theorem sine_sum_equality (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β) →
  α + β = π/2 := by
sorry

end sine_sum_equality_l537_53714


namespace tangent_property_of_sine_equation_l537_53754

theorem tangent_property_of_sine_equation (k : ℝ) (α β : ℝ) :
  (∃ (k : ℝ), k > 0 ∧
    (∀ x : ℝ, x ∈ Set.Ioo 0 Real.pi → (|Real.sin x| / x = k ↔ x = α ∨ x = β)) ∧
    α ∈ Set.Ioo 0 Real.pi ∧
    β ∈ Set.Ioo 0 Real.pi ∧
    α < β) →
  Real.tan (β + Real.pi / 4) = (1 + β) / (1 - β) :=
by sorry

end tangent_property_of_sine_equation_l537_53754


namespace triangle_properties_l537_53782

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 + c^2 - b^2 + a*c = 0 →
  (∃ (p : ℝ), p = a + b + c) →
  B = 2*π/3 ∧ (b = 2*Real.sqrt 3 → ∃ (p_max : ℝ), p_max = 4 + 2*Real.sqrt 3 ∧ ∀ p, p ≤ p_max) :=
by sorry

end triangle_properties_l537_53782


namespace parabola_has_one_x_intercept_l537_53726

-- Define the parabola function
def f (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

-- Theorem statement
theorem parabola_has_one_x_intercept :
  ∃! x : ℝ, ∃ y : ℝ, f y = x ∧ y = 0 :=
sorry

end parabola_has_one_x_intercept_l537_53726


namespace pink_notebook_cost_l537_53779

def total_notebooks : ℕ := 4
def green_notebooks : ℕ := 2
def black_notebooks : ℕ := 1
def pink_notebooks : ℕ := 1
def total_cost : ℕ := 45
def black_notebook_cost : ℕ := 15
def green_notebook_cost : ℕ := 10

theorem pink_notebook_cost :
  total_notebooks = green_notebooks + black_notebooks + pink_notebooks →
  total_cost = green_notebooks * green_notebook_cost + black_notebook_cost + pink_notebooks * 10 := by
  sorry

end pink_notebook_cost_l537_53779


namespace proportion_equality_l537_53795

theorem proportion_equality (x y : ℝ) (h1 : 2 * y = 5 * x) (h2 : x * y ≠ 0) : x / y = 2 / 5 := by
  sorry

end proportion_equality_l537_53795


namespace repeating_decimal_equals_fraction_l537_53781

/-- The repeating decimal 0.4̄67 as a real number -/
def repeating_decimal : ℚ := 0.4 + (2/3) / 100

/-- The fraction 4621/9900 as a rational number -/
def target_fraction : ℚ := 4621 / 9900

/-- Theorem stating that the repeating decimal 0.4̄67 is equal to the fraction 4621/9900 -/
theorem repeating_decimal_equals_fraction :
  repeating_decimal = target_fraction := by sorry

end repeating_decimal_equals_fraction_l537_53781


namespace first_interest_rate_is_ten_percent_l537_53745

/-- Proves that the first interest rate is 10% given the problem conditions --/
theorem first_interest_rate_is_ten_percent
  (total_amount : ℕ)
  (first_part : ℕ)
  (second_part : ℕ)
  (total_profit : ℕ)
  (second_rate : ℚ)
  (h1 : total_amount = 50000)
  (h2 : first_part = 30000)
  (h3 : second_part = total_amount - first_part)
  (h4 : total_profit = 7000)
  (h5 : second_rate = 20 / 100)
  : ∃ (r : ℚ), r = 10 / 100 ∧ 
    total_profit = (first_part * r).floor + (second_part * second_rate).floor :=
by sorry


end first_interest_rate_is_ten_percent_l537_53745


namespace unique_assignment_l537_53764

-- Define the friends and professions as enums
inductive Friend : Type
  | Ivanov | Petrenko | Sidorchuk | Grishin | Altman

inductive Profession : Type
  | Painter | Miller | Carpenter | Postman | Barber

-- Define the assignment of professions to friends
def assignment : Friend → Profession
  | Friend.Ivanov => Profession.Barber
  | Friend.Petrenko => Profession.Miller
  | Friend.Sidorchuk => Profession.Postman
  | Friend.Grishin => Profession.Carpenter
  | Friend.Altman => Profession.Painter

-- Define the conditions
def conditions (a : Friend → Profession) : Prop :=
  -- Each friend has a unique profession
  (∀ f1 f2, f1 ≠ f2 → a f1 ≠ a f2) ∧
  -- Petrenko and Grishin have never used a painter's brush
  (a Friend.Petrenko ≠ Profession.Painter ∧ a Friend.Grishin ≠ Profession.Painter) ∧
  -- Ivanov and Grishin visited the miller
  (a Friend.Ivanov ≠ Profession.Miller ∧ a Friend.Grishin ≠ Profession.Miller) ∧
  -- Petrenko and Altman live in the same house as the postman
  (a Friend.Petrenko ≠ Profession.Postman ∧ a Friend.Altman ≠ Profession.Postman) ∧
  -- Sidorchuk attended Petrenko's wedding and the wedding of his barber friend's daughter
  (a Friend.Sidorchuk ≠ Profession.Barber ∧ a Friend.Petrenko ≠ Profession.Barber) ∧
  -- Ivanov and Petrenko often play dominoes with the carpenter and the painter
  (a Friend.Ivanov ≠ Profession.Carpenter ∧ a Friend.Ivanov ≠ Profession.Painter ∧
   a Friend.Petrenko ≠ Profession.Carpenter ∧ a Friend.Petrenko ≠ Profession.Painter) ∧
  -- Grishin and Altman go to their barber friend's shop to get shaved
  (a Friend.Grishin ≠ Profession.Barber ∧ a Friend.Altman ≠ Profession.Barber) ∧
  -- The postman shaves himself
  (∀ f, a f = Profession.Postman → a f ≠ Profession.Barber)

-- Theorem statement
theorem unique_assignment : 
  ∀ a : Friend → Profession, conditions a → a = assignment :=
sorry

end unique_assignment_l537_53764


namespace albert_sequence_theorem_l537_53765

/-- Represents the sequence of positive integers starting with 1 or 2 in increasing order -/
def albert_sequence : ℕ → ℕ := sorry

/-- Returns the nth digit in Albert's sequence -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- The three-digit number formed by the 1498th, 1499th, and 1500th digits -/
def target_number : ℕ := 100 * (nth_digit 1498) + 10 * (nth_digit 1499) + (nth_digit 1500)

theorem albert_sequence_theorem : target_number = 121 := by sorry

end albert_sequence_theorem_l537_53765


namespace system_solution_l537_53729

theorem system_solution (x y z : ℝ) 
  (eq1 : y + z = 10 - 4*x)
  (eq2 : x + z = -16 - 4*y)
  (eq3 : x + y = 9 - 4*z) :
  3*x + 3*y + 3*z = 1.5 := by
sorry

end system_solution_l537_53729


namespace short_sleeve_shirts_count_proof_short_sleeve_shirts_l537_53766

theorem short_sleeve_shirts_count : ℕ → ℕ → ℕ → Prop :=
  fun total_shirts long_sleeve_shirts short_sleeve_shirts =>
    total_shirts = long_sleeve_shirts + short_sleeve_shirts →
    total_shirts = 30 →
    long_sleeve_shirts = 21 →
    short_sleeve_shirts = 8

-- The proof is omitted
theorem proof_short_sleeve_shirts : short_sleeve_shirts_count 30 21 8 := by
  sorry

end short_sleeve_shirts_count_proof_short_sleeve_shirts_l537_53766


namespace john_reading_capacity_l537_53778

/-- Represents the reading speed ratio between John and his brother -/
def johnSpeedRatio : ℝ := 1.6

/-- Time taken by John's brother to read one book (in hours) -/
def brotherReadTime : ℝ := 8

/-- Available time for John to read (in hours) -/
def availableTime : ℝ := 15

/-- Number of books John can read in the available time -/
def johnBooksRead : ℕ := 3

theorem john_reading_capacity : 
  ⌊availableTime / (brotherReadTime / johnSpeedRatio)⌋ = johnBooksRead := by
  sorry

end john_reading_capacity_l537_53778


namespace max_intersection_points_l537_53718

/-- The maximum number of intersection points given 8 planes in 3D space -/
theorem max_intersection_points (n : ℕ) (h : n = 8) : 
  (Nat.choose n 3 : ℕ) = 56 := by
  sorry

end max_intersection_points_l537_53718


namespace pies_from_36_apples_l537_53715

/-- Given that 3 pies can be made from 12 apples, this function calculates
    the number of pies that can be made from a given number of apples. -/
def pies_from_apples (apples : ℕ) : ℕ :=
  (apples * 3) / 12

theorem pies_from_36_apples :
  pies_from_apples 36 = 9 := by
  sorry

end pies_from_36_apples_l537_53715


namespace sin_double_alpha_l537_53776

/-- Given that the terminal side of angle α intersects the unit circle at point P(-√3/2, 1/2),
    prove that sin 2α = -√3/2 -/
theorem sin_double_alpha (α : Real) 
  (h : ∃ P : Real × Real, P.1 = -Real.sqrt 3 / 2 ∧ P.2 = 1 / 2 ∧ 
       P.1^2 + P.2^2 = 1 ∧ P.1 = Real.cos α ∧ P.2 = Real.sin α) : 
  Real.sin (2 * α) = -Real.sqrt 3 / 2 := by
sorry

end sin_double_alpha_l537_53776


namespace correct_equation_l537_53727

/-- Represents the situation described in the problem -/
structure Situation where
  x : ℕ  -- number of people
  total_cost : ℕ  -- total cost of the item

/-- The condition when each person contributes 8 coins -/
def condition_8 (s : Situation) : Prop :=
  8 * s.x = s.total_cost + 3

/-- The condition when each person contributes 7 coins -/
def condition_7 (s : Situation) : Prop :=
  7 * s.x + 4 = s.total_cost

/-- The theorem stating that the equation 8x - 3 = 7x + 4 correctly represents the situation -/
theorem correct_equation (s : Situation) :
  condition_8 s ∧ condition_7 s ↔ 8 * s.x - 3 = 7 * s.x + 4 := by
  sorry

end correct_equation_l537_53727


namespace danny_found_eighteen_caps_l537_53774

/-- The number of bottle caps Danny found at the park -/
def bottleCapsFound (initial : ℕ) (total : ℕ) : ℕ := total - initial

/-- Theorem: Danny found 18 bottle caps at the park -/
theorem danny_found_eighteen_caps : 
  let initial := 37
  let total := 55
  bottleCapsFound initial total = 18 := by
sorry

end danny_found_eighteen_caps_l537_53774


namespace circumcircle_radius_isosceles_triangle_l537_53742

/-- Given a triangle with two sides of length a and one side of length b,
    the radius of its circumcircle is a²/√(4a² - b²). -/
theorem circumcircle_radius_isosceles_triangle (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ R : ℝ, R = a^2 / Real.sqrt (4 * a^2 - b^2) ∧ 
  R > 0 ∧ 
  R * Real.sqrt (4 * a^2 - b^2) = a^2 := by
  sorry

end circumcircle_radius_isosceles_triangle_l537_53742


namespace choose_four_from_fifteen_l537_53777

theorem choose_four_from_fifteen (n : ℕ) (k : ℕ) : n = 15 ∧ k = 4 → Nat.choose n k = 1365 := by
  sorry

end choose_four_from_fifteen_l537_53777


namespace bold_o_lit_cells_l537_53773

/-- Represents a 5x5 grid with boolean values indicating lit (true) or unlit (false) cells. -/
def Grid := Matrix (Fin 5) (Fin 5) Bool

/-- The initial configuration of the letter 'o' on the grid. -/
def initial_o : Grid := sorry

/-- The number of lit cells in the initial 'o' configuration. -/
def initial_lit_cells : Nat := 12

/-- Makes a letter bold by lighting cells to the right of lit cells. -/
def make_bold (g : Grid) : Grid := sorry

/-- Counts the number of lit cells in a grid. -/
def count_lit_cells (g : Grid) : Nat := sorry

/-- Theorem stating that the number of lit cells in a bold 'o' is 24. -/
theorem bold_o_lit_cells :
  count_lit_cells (make_bold initial_o) = 24 := by sorry

end bold_o_lit_cells_l537_53773


namespace coconut_grove_problem_l537_53705

theorem coconut_grove_problem (x : ℝ) : 
  (60 * (x + 1) + 120 * x + 180 * (x - 1)) / (3 * x) = 100 → x = 2 := by
  sorry

end coconut_grove_problem_l537_53705


namespace boys_percentage_l537_53723

theorem boys_percentage (total : ℕ) (boys : ℕ) (girls : ℕ) (additional_boys : ℕ) : 
  total = 50 →
  boys + girls = total →
  additional_boys = 50 →
  girls = (total + additional_boys) / 20 →
  (boys : ℚ) / total = 9 / 10 := by
sorry

end boys_percentage_l537_53723


namespace geometric_sequence_product_l537_53772

/-- A sequence a : ℕ → ℝ is geometric if there exists a common ratio r such that
    a(n+1) = r * a(n) for all n ≥ 1 -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  IsGeometric a →
  (a 1)^2 - 2*(a 1) - 3 = 0 →
  (a 4)^2 - 2*(a 4) - 3 = 0 →
  a 2 * a 3 = -3 := by
  sorry

end geometric_sequence_product_l537_53772


namespace zero_not_in_positive_integers_l537_53751

theorem zero_not_in_positive_integers : 0 ∉ {n : ℕ | n > 0} := by
  sorry

end zero_not_in_positive_integers_l537_53751


namespace trigonometric_identity_l537_53783

theorem trigonometric_identity (c : ℝ) (h : c = 2 * Real.pi / 13) :
  (Real.sin (4 * c) * Real.sin (8 * c) * Real.sin (12 * c) * Real.sin (16 * c) * Real.sin (20 * c)) /
  (Real.sin c * Real.sin (2 * c) * Real.sin (3 * c) * Real.sin (4 * c) * Real.sin (5 * c)) =
  1 / Real.sin (2 * Real.pi / 13) := by
  sorry

end trigonometric_identity_l537_53783


namespace nancy_crayons_l537_53752

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 15

/-- The number of packs Nancy bought -/
def packs_bought : ℕ := 41

/-- The total number of crayons Nancy bought -/
def total_crayons : ℕ := crayons_per_pack * packs_bought

theorem nancy_crayons : total_crayons = 615 := by
  sorry

end nancy_crayons_l537_53752


namespace carla_games_won_l537_53798

theorem carla_games_won (total_games : ℕ) (frankie_games : ℕ) (carla_games : ℕ) : 
  total_games = 30 →
  frankie_games = carla_games / 2 →
  frankie_games + carla_games = total_games →
  carla_games = 20 := by
  sorry

end carla_games_won_l537_53798


namespace sum_of_roots_l537_53719

/-- Given a quadratic function f(x) = x^2 - 2016x + 2015 and two distinct points a and b
    where f(a) = f(b), prove that a + b = 2016 -/
theorem sum_of_roots (a b : ℝ) (ha : a ≠ b) :
  (a^2 - 2016*a + 2015 = b^2 - 2016*b + 2015) →
  a + b = 2016 := by
  sorry

end sum_of_roots_l537_53719


namespace min_value_of_sum_of_squares_l537_53750

theorem min_value_of_sum_of_squares (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 8) 
  (h2 : e * f * g * h = 16) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 32 ∧ 
  ∃ (a' b' c' d' e' f' g' h' : ℝ), 
    a' * b' * c' * d' = 8 ∧ 
    e' * f' * g' * h' = 16 ∧ 
    (a' * e')^2 + (b' * f')^2 + (c' * g')^2 + (d' * h')^2 = 32 :=
by sorry

end min_value_of_sum_of_squares_l537_53750


namespace greatest_multiple_of_5_and_6_less_than_1000_l537_53716

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  (∃ n : ℕ, n < 1000 ∧ 5 ∣ n ∧ 6 ∣ n) →
  (∀ m : ℕ, m < 1000 ∧ 5 ∣ m ∧ 6 ∣ m → m ≤ 990) ∧
  990 < 1000 ∧ 5 ∣ 990 ∧ 6 ∣ 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l537_53716


namespace quadratic_two_distinct_roots_l537_53704

/-- A quadratic equation (a-1)x^2 - 2x + 1 = 0 has two distinct real roots if and only if a < 2 and a ≠ 1 -/
theorem quadratic_two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (a - 1) * x^2 - 2 * x + 1 = 0 ∧ (a - 1) * y^2 - 2 * y + 1 = 0) ↔ 
  (a < 2 ∧ a ≠ 1) :=
sorry

end quadratic_two_distinct_roots_l537_53704


namespace total_pancakes_l537_53775

/-- Represents the number of pancakes in a big stack -/
def big_stack : ℕ := 5

/-- Represents the number of pancakes in a short stack -/
def short_stack : ℕ := 3

/-- Represents the number of customers who ordered short stack -/
def short_stack_orders : ℕ := 9

/-- Represents the number of customers who ordered big stack -/
def big_stack_orders : ℕ := 6

/-- Theorem stating the total number of pancakes Hank needs to make -/
theorem total_pancakes : 
  short_stack_orders * short_stack + big_stack_orders * big_stack = 57 := by
  sorry


end total_pancakes_l537_53775


namespace nested_squares_difference_l537_53785

/-- Given four nested squares with side lengths S₁ > S₂ > S₃ > S₄,
    where the differences between consecutive square side lengths are 11, 5, and 13 (from largest to smallest),
    prove that S₁ - S₄ = 29. -/
theorem nested_squares_difference (S₁ S₂ S₃ S₄ : ℝ) 
  (h₁ : S₁ = S₂ + 11)
  (h₂ : S₂ = S₃ + 5)
  (h₃ : S₃ = S₄ + 13) :
  S₁ - S₄ = 29 := by
  sorry

end nested_squares_difference_l537_53785


namespace gregs_shopping_expenditure_l537_53731

theorem gregs_shopping_expenditure (total_spent : ℕ) (shoe_price_difference : ℕ) :
  total_spent = 300 →
  shoe_price_difference = 9 →
  ∃ (shirt_price shoe_price : ℕ),
    shirt_price + shoe_price = total_spent ∧
    shoe_price = 2 * shirt_price + shoe_price_difference ∧
    shirt_price = 97 :=
by sorry

end gregs_shopping_expenditure_l537_53731


namespace fraction_problem_l537_53770

theorem fraction_problem (F : ℚ) : 
  3 + F * (1/3) * (1/5) * 90 = (1/15) * 90 → F = 1/2 := by
sorry

end fraction_problem_l537_53770


namespace only_pi_smaller_than_neg_three_l537_53720

theorem only_pi_smaller_than_neg_three : 
  (-Real.sqrt 2 > -3) ∧ (1 > -3) ∧ (0 > -3) ∧ (-Real.pi < -3) := by
  sorry

end only_pi_smaller_than_neg_three_l537_53720


namespace f_min_value_negative_reals_l537_53708

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^9 + 2

-- State the theorem
theorem f_min_value_negative_reals 
  (a b : ℝ) 
  (h_max : ∀ x > 0, f a b x ≤ 5) :
  ∀ x < 0, f a b x ≥ -1 :=
sorry

end f_min_value_negative_reals_l537_53708


namespace gcd_problem_l537_53796

theorem gcd_problem (n : ℕ) : 
  70 ≤ n ∧ n ≤ 80 → Nat.gcd 15 n = 5 → n = 70 ∨ n = 80 := by
  sorry

end gcd_problem_l537_53796


namespace paperback_cost_is_twelve_l537_53755

/-- Represents the book club's annual fee collection --/
structure BookClub where
  members : ℕ
  snackFeePerMember : ℕ
  hardcoverBooksPerMember : ℕ
  hardcoverBookPrice : ℕ
  paperbackBooksPerMember : ℕ
  totalCollected : ℕ

/-- Calculates the cost per paperback book --/
def costPerPaperback (club : BookClub) : ℚ :=
  let snackTotal := club.members * club.snackFeePerMember
  let hardcoverTotal := club.members * club.hardcoverBooksPerMember * club.hardcoverBookPrice
  let paperbackTotal := club.totalCollected - snackTotal - hardcoverTotal
  paperbackTotal / (club.members * club.paperbackBooksPerMember)

/-- Theorem stating that the cost per paperback book is $12 --/
theorem paperback_cost_is_twelve (club : BookClub) 
    (h1 : club.members = 6)
    (h2 : club.snackFeePerMember = 150)
    (h3 : club.hardcoverBooksPerMember = 6)
    (h4 : club.hardcoverBookPrice = 30)
    (h5 : club.paperbackBooksPerMember = 6)
    (h6 : club.totalCollected = 2412) :
    costPerPaperback club = 12 := by
  sorry


end paperback_cost_is_twelve_l537_53755


namespace sqrt_15_div_sqrt_5_eq_sqrt_3_l537_53737

theorem sqrt_15_div_sqrt_5_eq_sqrt_3 : Real.sqrt 15 / Real.sqrt 5 = Real.sqrt 3 := by
  sorry

end sqrt_15_div_sqrt_5_eq_sqrt_3_l537_53737


namespace arthur_hamburgers_l537_53709

/-- The number of hamburgers Arthur bought on the first day -/
def hamburgers_day1 : ℕ := 1

/-- The price of a hamburger in dollars -/
def hamburger_price : ℚ := 6

/-- The price of a hot dog in dollars -/
def hotdog_price : ℚ := 1

/-- Total cost of Arthur's purchase on day 1 in dollars -/
def total_cost_day1 : ℚ := 10

/-- Total cost of Arthur's purchase on day 2 in dollars -/
def total_cost_day2 : ℚ := 7

/-- Number of hot dogs bought on day 1 -/
def hotdogs_day1 : ℕ := 4

/-- Number of hamburgers bought on day 2 -/
def hamburgers_day2 : ℕ := 2

/-- Number of hot dogs bought on day 2 -/
def hotdogs_day2 : ℕ := 3

theorem arthur_hamburgers :
  (hamburgers_day1 : ℚ) * hamburger_price + (hotdogs_day1 : ℚ) * hotdog_price = total_cost_day1 ∧
  (hamburgers_day2 : ℚ) * hamburger_price + (hotdogs_day2 : ℚ) * hotdog_price = total_cost_day2 :=
sorry

end arthur_hamburgers_l537_53709


namespace opposite_numbers_equation_l537_53768

theorem opposite_numbers_equation (a b : ℝ) : a + b = 0 → a - (2 - b) = -2 := by
  sorry

end opposite_numbers_equation_l537_53768


namespace white_washing_cost_l537_53767

/-- Calculate the cost of white washing a room with given dimensions and openings. -/
theorem white_washing_cost
  (room_length room_width room_height : ℝ)
  (door_length door_width : ℝ)
  (window_length window_width : ℝ)
  (num_windows : ℕ)
  (cost_per_sqft : ℝ)
  (h_room_length : room_length = 25)
  (h_room_width : room_width = 15)
  (h_room_height : room_height = 12)
  (h_door_length : door_length = 6)
  (h_door_width : door_width = 3)
  (h_window_length : window_length = 4)
  (h_window_width : window_width = 3)
  (h_num_windows : num_windows = 3)
  (h_cost_per_sqft : cost_per_sqft = 10) :
  (2 * (room_length * room_height + room_width * room_height) -
   (door_length * door_width + num_windows * window_length * window_width)) * cost_per_sqft = 9060 := by
  sorry

end white_washing_cost_l537_53767


namespace bridge_length_specific_bridge_length_l537_53780

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof of the specific bridge length problem -/
theorem specific_bridge_length :
  bridge_length 140 45 30 = 235 := by
  sorry

end bridge_length_specific_bridge_length_l537_53780


namespace contrapositive_zero_product_l537_53744

theorem contrapositive_zero_product (a b : ℝ) : a ≠ 0 ∧ b ≠ 0 → a * b ≠ 0 := by
  sorry

end contrapositive_zero_product_l537_53744


namespace perpendicular_line_parallel_lines_l537_53710

-- Define the original lines
def line1 (x y : ℝ) : Prop := x + 3 * y - 5 = 0
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-1, 0)

-- Define the distance between parallel lines
def distance : ℝ := 7

-- Theorem for the perpendicular line
theorem perpendicular_line :
  ∃ (a b c : ℝ), (∀ x y, a * x + b * y + c = 0 ↔ 3 * x - y + 3 = 0) ∧
  (∀ x y, line1 x y → (a * x + b * y + c = 0 → a * 3 + b = 0)) ∧
  (a * point_P.1 + b * point_P.2 + c = 0) :=
sorry

-- Theorem for the parallel lines
theorem parallel_lines :
  ∃ (c1 c2 : ℝ), 
  (∀ x y, 3 * x + 4 * y + c1 = 0 ∨ 3 * x + 4 * y + c2 = 0 ↔ 
    (3 * x + 4 * y + 23 = 0 ∨ 3 * x + 4 * y - 47 = 0)) ∧
  (∀ x y, line2 x y → 
    (|c1 + 12| / Real.sqrt 25 = distance ∧ |c2 + 12| / Real.sqrt 25 = distance)) :=
sorry

end perpendicular_line_parallel_lines_l537_53710


namespace fourth_root_16_times_fifth_root_32_l537_53760

theorem fourth_root_16_times_fifth_root_32 : (16 : ℝ) ^ (1/4) * (32 : ℝ) ^ (1/5) = 4 := by
  sorry

end fourth_root_16_times_fifth_root_32_l537_53760


namespace quadratic_root_sum_product_l537_53707

-- Define the quadratic equation
def quadratic (x k : ℝ) : Prop := x^2 - 3*x + k = 0

-- Define the condition on the roots
def root_condition (x₁ x₂ : ℝ) : Prop := x₁*x₂ + 2*x₁ + 2*x₂ = 1

-- Theorem statement
theorem quadratic_root_sum_product 
  (k : ℝ) 
  (x₁ x₂ : ℝ) 
  (h1 : quadratic x₁ k) 
  (h2 : quadratic x₂ k) 
  (h3 : x₁ ≠ x₂) 
  (h4 : root_condition x₁ x₂) : 
  k = -5 := by
  sorry

end quadratic_root_sum_product_l537_53707


namespace sum_of_coefficients_l537_53746

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (3*x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₂ + a₄ = 136 := by
  sorry

end sum_of_coefficients_l537_53746


namespace least_sum_of_valid_pair_l537_53771

def is_valid_pair (a b : ℕ+) : Prop :=
  Nat.gcd (a + b) 330 = 1 ∧
  (a : ℕ) ^ (a : ℕ) % (b : ℕ) ^ (b : ℕ) = 0 ∧
  (a : ℕ) % (b : ℕ) ≠ 0

theorem least_sum_of_valid_pair :
  ∃ (a b : ℕ+), is_valid_pair a b ∧
    ∀ (a' b' : ℕ+), is_valid_pair a' b' → a + b ≤ a' + b' ∧
    a + b = 357 :=
sorry

end least_sum_of_valid_pair_l537_53771


namespace intersection_implies_a_zero_l537_53739

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {a^2, a+1, -1}
def B (a : ℝ) : Set ℝ := {2*a-1, |a-2|, 3*a^2+4}

-- Theorem statement
theorem intersection_implies_a_zero (a : ℝ) : A a ∩ B a = {-1} → a = 0 := by
  sorry

end intersection_implies_a_zero_l537_53739


namespace ursula_change_l537_53721

/-- Calculates the change Ursula received after buying hot dogs and salads -/
theorem ursula_change (hot_dog_price : ℚ) (salad_price : ℚ) 
  (num_hot_dogs : ℕ) (num_salads : ℕ) (bill_value : ℚ) (num_bills : ℕ) :
  hot_dog_price = 3/2 →
  salad_price = 5/2 →
  num_hot_dogs = 5 →
  num_salads = 3 →
  bill_value = 10 →
  num_bills = 2 →
  (num_bills * bill_value) - (num_hot_dogs * hot_dog_price + num_salads * salad_price) = 5 :=
by sorry

end ursula_change_l537_53721


namespace isosceles_trapezoid_area_l537_53732

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid :=
  (longer_base : ℝ)
  (base_angle : ℝ)

/-- The area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

theorem isosceles_trapezoid_area :
  ∀ t : IsoscelesTrapezoid,
    t.longer_base = 20 ∧
    t.base_angle = Real.arcsin 0.6 →
    area t = 72 :=
  sorry

end isosceles_trapezoid_area_l537_53732


namespace unique_n_satisfying_conditions_l537_53717

theorem unique_n_satisfying_conditions :
  ∃! n : ℤ,
    0 ≤ n ∧ n ≤ 8 ∧
    ∃ x : ℤ,
      x > 0 ∧
      (-4567 + x ≥ 0) ∧
      (∀ y : ℤ, y > 0 ∧ -4567 + y ≥ 0 → x ≤ y) ∧
      n ≡ -4567 + x [ZMOD 9] ∧
    n = 0 :=
by sorry

end unique_n_satisfying_conditions_l537_53717
