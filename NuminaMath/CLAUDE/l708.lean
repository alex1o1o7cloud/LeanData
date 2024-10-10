import Mathlib

namespace parallelogram_reflection_l708_70838

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the parallelogram
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the perpendicular line
def perpendicular_line (p : Parallelogram) (t : Line) : Prop :=
  -- Assuming some condition for perpendicularity
  sorry

-- Define the intersection points
def intersection_point (l1 l2 : Line) : Point :=
  -- Assuming some method to find intersection
  sorry

-- Define the reflection operation
def reflect_point (p : Point) (t : Line) : Point :=
  -- Assuming some method to reflect a point over a line
  sorry

-- The main theorem
theorem parallelogram_reflection 
  (p : Parallelogram) 
  (t : Line) 
  (h_perp : perpendicular_line p t) :
  ∃ (p' : Parallelogram),
    let K := intersection_point (Line.mk 0 1 0) t  -- Assuming AB is on y-axis for simplicity
    let L := intersection_point (Line.mk 0 1 0) t  -- Assuming CD is parallel to AB
    p'.A = reflect_point p.A t ∧
    p'.B = reflect_point p.B t ∧
    p'.C = reflect_point p.C t ∧
    p'.D = reflect_point p.D t ∧
    p'.A = Point.mk (2 * K.x - p.A.x) (2 * K.y - p.A.y) ∧
    p'.B = Point.mk (2 * K.x - p.B.x) (2 * K.y - p.B.y) ∧
    p'.C = Point.mk (2 * L.x - p.C.x) (2 * L.y - p.C.y) ∧
    p'.D = Point.mk (2 * L.x - p.D.x) (2 * L.y - p.D.y) :=
  by sorry

end parallelogram_reflection_l708_70838


namespace teachers_percentage_of_boys_l708_70885

/-- Proves that the percentage of teachers to boys is 20% given the specified conditions -/
theorem teachers_percentage_of_boys (boys girls teachers : ℕ) : 
  (boys : ℚ) / (girls : ℚ) = 3 / 4 →
  girls = 60 →
  boys + girls + teachers = 114 →
  (teachers : ℚ) / (boys : ℚ) * 100 = 20 := by
  sorry

end teachers_percentage_of_boys_l708_70885


namespace pond_algae_free_day_24_l708_70865

/-- Represents the coverage of algae in the pond on a given day -/
def algae_coverage (day : ℕ) : ℝ := sorry

/-- The algae coverage triples every two days -/
axiom triple_every_two_days (d : ℕ) : algae_coverage (d + 2) = 3 * algae_coverage d

/-- The pond is completely covered on day 28 -/
axiom full_coverage_day_28 : algae_coverage 28 = 1

/-- Theorem: The pond is 88.89% algae-free on day 24 -/
theorem pond_algae_free_day_24 : algae_coverage 24 = 1 - 0.8889 := by sorry

end pond_algae_free_day_24_l708_70865


namespace lcm_hcf_problem_l708_70834

theorem lcm_hcf_problem (a b : ℕ+) (h1 : Nat.lcm a b = 25974) (h2 : Nat.gcd a b = 107) (h3 : a = 4951) : b = 561 := by
  sorry

end lcm_hcf_problem_l708_70834


namespace cubic_sum_problem_l708_70823

theorem cubic_sum_problem (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) :
  x^3 + y^3 = 65 := by sorry

end cubic_sum_problem_l708_70823


namespace percent_within_one_std_dev_l708_70880

-- Define a symmetric distribution
structure SymmetricDistribution where
  mean : ℝ
  std_dev : ℝ
  is_symmetric : Bool
  percent_less_than_mean_plus_std : ℝ

-- Theorem statement
theorem percent_within_one_std_dev 
  (dist : SymmetricDistribution) 
  (h1 : dist.is_symmetric = true) 
  (h2 : dist.percent_less_than_mean_plus_std = 92) : 
  (100 - 2 * (100 - dist.percent_less_than_mean_plus_std)) = 84 := by
  sorry

end percent_within_one_std_dev_l708_70880


namespace dagger_example_l708_70813

-- Define the † operation
def dagger (m n p q : ℕ) (hm : m ≠ 0) : ℚ :=
  (m^2 * p * (q / n : ℚ)) + (p / m : ℚ)

-- Theorem statement
theorem dagger_example : dagger 5 9 6 2 (by norm_num) = 518 / 15 := by
  sorry

end dagger_example_l708_70813


namespace rectangle_area_theorem_l708_70891

theorem rectangle_area_theorem (x : ℝ) : 
  let large_rectangle_area := (2*x + 14) * (2*x + 10)
  let hole_area := (4*x - 6) * (2*x - 4)
  let square_area := (x + 3)^2
  large_rectangle_area - hole_area + square_area = -3*x^2 + 82*x + 125 := by
sorry

end rectangle_area_theorem_l708_70891


namespace no_permutation_sum_all_nines_l708_70847

/-- The number of 9's in the sum -/
def num_nines : ℕ := 1111

/-- Function to check if a number is composed of only 9's -/
def is_all_nines (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10^k - 1 ∧ k = num_nines

/-- Function to check if two numbers are digit permutations of each other -/
def is_permutation (x y : ℕ) : Prop :=
  ∃ σ : Fin (Nat.digits 10 x).length ≃ Fin (Nat.digits 10 y).length,
    ∀ i, (Nat.digits 10 x)[i] = (Nat.digits 10 y)[σ i]

/-- Main theorem statement -/
theorem no_permutation_sum_all_nines :
  ¬∃ (x y : ℕ), is_permutation x y ∧ is_all_nines (x + y) := by
  sorry

end no_permutation_sum_all_nines_l708_70847


namespace circle_radius_sqrt34_l708_70842

/-- Given a circle with center on the x-axis passing through points (0,5) and (2,3),
    prove that its radius is √34. -/
theorem circle_radius_sqrt34 :
  ∀ x : ℝ,
  (x^2 + 5^2 = (x-2)^2 + 3^2) →  -- condition that (x,0) is equidistant from (0,5) and (2,3)
  ∃ r : ℝ,
  r^2 = 34 ∧                    -- r is the radius
  r^2 = x^2 + 5^2               -- distance formula from center to (0,5)
  :=
by sorry

end circle_radius_sqrt34_l708_70842


namespace airplane_seats_total_l708_70859

/-- Represents the number of seats in an airplane -/
def AirplaneSeats (total : ℝ) : Prop :=
  let first_class : ℝ := 36
  let business_class : ℝ := 0.3 * total
  let economy : ℝ := 0.6 * total
  let premium_economy : ℝ := total - first_class - business_class - economy
  (first_class + business_class + economy + premium_economy = total) ∧
  (premium_economy ≥ 0)

/-- The total number of seats in the airplane is 360 -/
theorem airplane_seats_total : ∃ (total : ℝ), AirplaneSeats total ∧ total = 360 := by
  sorry

end airplane_seats_total_l708_70859


namespace sufficient_not_necessary_condition_l708_70872

/-- Given a complex number z = (a+1) - ai where a is real,
    prove that a = -1 is a sufficient but not necessary condition for |z| = 1 -/
theorem sufficient_not_necessary_condition (a : ℝ) :
  let z : ℂ := (a + 1) - a * I
  (a = -1 → Complex.abs z = 1) ∧
  ¬(Complex.abs z = 1 → a = -1) := by
  sorry

end sufficient_not_necessary_condition_l708_70872


namespace quadratic_condition_l708_70845

/-- A quadratic equation in x is of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_in_x (a b c : ℝ) : Prop := a ≠ 0

/-- The equation ax² - 2x + 3 = 0 -/
def equation (a : ℝ) (x : ℝ) : Prop := a * x^2 - 2*x + 3 = 0

theorem quadratic_condition (a : ℝ) :
  (∃ x, equation a x) ∧ is_quadratic_in_x a (-2) 3 ↔ a ≠ 0 :=
sorry

end quadratic_condition_l708_70845


namespace m_range_l708_70861

/-- Given conditions p and q, prove that the range of real numbers m is [-2, -1). -/
theorem m_range (p : ∀ x : ℝ, 2 * x > m * (x^2 + 1))
                (q : ∃ x₀ : ℝ, x₀^2 + 2 * x₀ - m - 1 = 0) :
  m ≥ -2 ∧ m < -1 :=
by sorry

end m_range_l708_70861


namespace squirrel_acorns_l708_70826

theorem squirrel_acorns (initial_acorns : ℕ) (winter_months : ℕ) (remaining_per_month : ℕ) : 
  initial_acorns = 210 →
  winter_months = 3 →
  remaining_per_month = 60 →
  initial_acorns - (winter_months * remaining_per_month) = 30 :=
by
  sorry

end squirrel_acorns_l708_70826


namespace hotel_outlets_count_l708_70817

/-- Represents the number of outlets required for different room types and the distribution of outlet types -/
structure HotelOutlets where
  standardRoomOutlets : ℕ
  suiteOutlets : ℕ
  standardRoomCount : ℕ
  suiteCount : ℕ
  typeAPercentage : ℚ
  typeBPercentage : ℚ
  typeCPercentage : ℚ

/-- Calculates the total number of outlets needed for a hotel -/
def totalOutlets (h : HotelOutlets) : ℕ :=
  h.standardRoomCount * h.standardRoomOutlets +
  h.suiteCount * h.suiteOutlets

/-- Theorem stating that the total number of outlets for the given hotel configuration is 650 -/
theorem hotel_outlets_count (h : HotelOutlets)
    (h_standard : h.standardRoomOutlets = 10)
    (h_suite : h.suiteOutlets = 15)
    (h_standard_count : h.standardRoomCount = 50)
    (h_suite_count : h.suiteCount = 10)
    (h_typeA : h.typeAPercentage = 2/5)
    (h_typeB : h.typeBPercentage = 3/5)
    (h_typeC : h.typeCPercentage = 1) :
  totalOutlets h = 650 := by
  sorry

end hotel_outlets_count_l708_70817


namespace max_tau_minus_n_max_tau_minus_n_achievable_l708_70871

/-- τ(n) is the number of positive divisors of n -/
def tau (n : ℕ+) : ℕ := sorry

/-- The theorem states that 4τ(n) - n is at most 12 for all positive integers n -/
theorem max_tau_minus_n (n : ℕ+) : 4 * (tau n) - n.val ≤ 12 := by sorry

/-- The theorem states that there exists a positive integer n for which 4τ(n) - n equals 12 -/
theorem max_tau_minus_n_achievable : ∃ n : ℕ+, 4 * (tau n) - n.val = 12 := by sorry

end max_tau_minus_n_max_tau_minus_n_achievable_l708_70871


namespace quadratic_roots_sum_l708_70879

/-- Given that α and β are the roots of x^2 + x - 1 = 0, prove that 2α^5 + β^3 = -13 ± 4√5 -/
theorem quadratic_roots_sum (α β : ℝ) : 
  α^2 + α - 1 = 0 → β^2 + β - 1 = 0 → 
  2 * α^5 + β^3 = -13 + 4 * Real.sqrt 5 ∨ 2 * α^5 + β^3 = -13 - 4 * Real.sqrt 5 := by
  sorry

end quadratic_roots_sum_l708_70879


namespace parallelepiped_dimensions_l708_70897

theorem parallelepiped_dimensions (n : ℕ) (h1 : n > 6) : 
  (n - 2) * (n - 4) * (n - 6) = (2 * n * (n - 2) * (n - 4)) / 3 → n = 18 :=
by sorry

end parallelepiped_dimensions_l708_70897


namespace profit_maximized_at_150_l708_70829

/-- The profit function for a company -/
def profit_function (a : ℝ) (x : ℝ) : ℝ := -a * x^2 + 7500 * x

/-- The derivative of the profit function -/
def profit_derivative (a : ℝ) (x : ℝ) : ℝ := -2 * a * x + 7500

theorem profit_maximized_at_150 (a : ℝ) :
  (profit_derivative a 150 = 0) → (a = 25) :=
by sorry

#check profit_maximized_at_150

end profit_maximized_at_150_l708_70829


namespace inequality_holds_l708_70828

theorem inequality_holds (a b c d : ℝ) (h1 : b < a) (h2 : d < c) : a + c > b + d := by
  sorry

end inequality_holds_l708_70828


namespace subset_of_any_implies_zero_l708_70894

theorem subset_of_any_implies_zero (a : ℝ) : 
  (∀ S : Set ℝ, {x : ℝ | a * x = 1} ⊆ S) → a = 0 :=
by sorry

end subset_of_any_implies_zero_l708_70894


namespace softball_team_size_l708_70881

/-- Proves that a co-ed softball team with 5 more women than men and a men-to-women ratio of 0.5 has 15 total players -/
theorem softball_team_size (men women : ℕ) : 
  women = men + 5 →
  men / women = 1 / 2 →
  men + women = 15 := by
sorry

end softball_team_size_l708_70881


namespace sum_of_large_prime_factors_2310_l708_70850

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem sum_of_large_prime_factors_2310 : 
  ∃ (factors : List ℕ), 
    (∀ f ∈ factors, is_prime f ∧ f > 5) ∧ 
    (factors.prod = 2310 / (2 * 3 * 5)) ∧
    (factors.sum = 18) := by
  sorry

end sum_of_large_prime_factors_2310_l708_70850


namespace nested_square_root_value_l708_70886

theorem nested_square_root_value :
  ∃ y : ℝ, y = Real.sqrt (2 + y) → y = 2 := by
  sorry

end nested_square_root_value_l708_70886


namespace total_brass_l708_70878

def brass_composition (copper zinc : ℝ) : Prop :=
  copper / zinc = 13 / 7

theorem total_brass (zinc : ℝ) (h : zinc = 35) :
  ∃ total : ℝ, brass_composition (total - zinc) zinc ∧ total = 100 :=
sorry

end total_brass_l708_70878


namespace least_number_for_divisibility_l708_70899

theorem least_number_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((5432 + y) % 5 = 0 ∧ (5432 + y) % 6 = 0 ∧ (5432 + y) % 7 = 0 ∧ (5432 + y) % 11 = 0 ∧ (5432 + y) % 13 = 0)) ∧
  ((5432 + x) % 5 = 0 ∧ (5432 + x) % 6 = 0 ∧ (5432 + x) % 7 = 0 ∧ (5432 + x) % 11 = 0 ∧ (5432 + x) % 13 = 0) →
  x = 24598 :=
by sorry

end least_number_for_divisibility_l708_70899


namespace fraction_sum_equals_point_four_l708_70884

theorem fraction_sum_equals_point_four :
  2 / 20 + 3 / 30 + 4 / 40 + 5 / 50 = 0.4 := by
  sorry

end fraction_sum_equals_point_four_l708_70884


namespace standard_time_proof_l708_70854

/-- The standard time to complete one workpiece -/
def standard_time : ℝ := 15

/-- The time taken by the first worker after innovation -/
def worker1_time (x : ℝ) : ℝ := x - 5

/-- The time taken by the second worker after innovation -/
def worker2_time (x : ℝ) : ℝ := x - 3

/-- The performance improvement factor -/
def improvement_factor : ℝ := 1.375

theorem standard_time_proof :
  ∃ (x : ℝ),
    x > 0 ∧
    worker1_time x > 0 ∧
    worker2_time x > 0 ∧
    (1 / worker1_time x + 1 / worker2_time x) = (2 / x) * improvement_factor ∧
    x = standard_time :=
by sorry

end standard_time_proof_l708_70854


namespace largest_gold_coins_max_gold_coins_l708_70868

theorem largest_gold_coins (n : ℕ) : n < 120 → n % 15 = 3 → n ≤ 105 := by
  sorry

theorem max_gold_coins : ∃ n : ℕ, n = 105 ∧ n < 120 ∧ n % 15 = 3 ∧ ∀ m : ℕ, m < 120 → m % 15 = 3 → m ≤ n := by
  sorry

end largest_gold_coins_max_gold_coins_l708_70868


namespace inequality_solution_set_l708_70898

theorem inequality_solution_set (x : ℝ) :
  (3 - x) / (2 * x - 4) < 1 ↔ x > 2 := by sorry

end inequality_solution_set_l708_70898


namespace bus_dispatch_theorem_l708_70873

/-- Represents the bus dispatch problem -/
structure BusDispatchProblem where
  initial_buses : ℕ := 15
  dispatch_interval : ℕ := 6
  entry_interval : ℕ := 8
  entry_delay : ℕ := 3
  total_time : ℕ := 840

/-- Calculates the time when the parking lot is first empty -/
def first_empty_time (problem : BusDispatchProblem) : ℕ :=
  sorry

/-- Calculates the time when buses can no longer be dispatched on time -/
def dispatch_failure_time (problem : BusDispatchProblem) : ℕ :=
  sorry

/-- Calculates the delay for the first bus that can't be dispatched on time -/
def first_delay_time (problem : BusDispatchProblem) : ℕ :=
  sorry

/-- Calculates the minimum interval for continuous dispatching -/
def min_continuous_interval (problem : BusDispatchProblem) : ℕ :=
  sorry

/-- Calculates the minimum number of additional buses needed for 6-minute interval dispatching -/
def min_additional_buses (problem : BusDispatchProblem) : ℕ :=
  sorry

theorem bus_dispatch_theorem (problem : BusDispatchProblem) :
  first_empty_time problem = 330 ∧
  dispatch_failure_time problem = 354 ∧
  first_delay_time problem = 1 ∧
  min_continuous_interval problem = 8 ∧
  min_additional_buses problem = 22 := by
  sorry

end bus_dispatch_theorem_l708_70873


namespace smallest_n_congruence_l708_70849

theorem smallest_n_congruence : 
  ∃! (n : ℕ), n > 0 ∧ (3 * n) % 24 = 1410 % 24 ∧ ∀ m : ℕ, m > 0 → (3 * m) % 24 = 1410 % 24 → n ≤ m :=
by sorry

end smallest_n_congruence_l708_70849


namespace comparison_inequality_range_of_linear_combination_l708_70856

-- Part 1
theorem comparison_inequality (x y z : ℝ) : 
  5 * x^2 + y^2 + z^2 ≥ 2 * x * y + 4 * x + 2 * z - 2 := by sorry

-- Part 2
theorem range_of_linear_combination (a b : ℝ) 
  (h1 : 1 ≤ 2 * a + b) (h2 : 2 * a + b ≤ 4) 
  (h3 : -1 ≤ a - 2 * b) (h4 : a - 2 * b ≤ 2) : 
  -1 ≤ 10 * a - 5 * b ∧ 10 * a - 5 * b ≤ 20 := by sorry

end comparison_inequality_range_of_linear_combination_l708_70856


namespace sally_balloons_l708_70863

theorem sally_balloons (sally_balloons fred_balloons : ℕ) : 
  fred_balloons = 3 * sally_balloons →
  fred_balloons = 18 →
  sally_balloons = 6 := by
sorry

end sally_balloons_l708_70863


namespace star_properties_l708_70840

/-- Custom multiplication operation -/
def star (a b : ℝ) : ℝ := (a + b)^2

/-- Theorem stating that exactly two of the given properties hold for the star operation -/
theorem star_properties :
  (∃! n : ℕ, n = 2 ∧ 
    (((∀ a b : ℝ, star a b = 0 → a = 0 ∧ b = 0) → n ≥ 1) ∧
     ((∀ a b : ℝ, star a b = star b a) → n ≥ 1) ∧
     ((∀ a b c : ℝ, star a (b + c) = star a b + star a c) → n ≥ 1) ∧
     ((∀ a b : ℝ, star a b = star (-a) (-b)) → n ≥ 1)) ∧
    n ≤ 2) :=
sorry

end star_properties_l708_70840


namespace total_decorations_handed_out_l708_70830

/-- Represents the contents of a decoration box -/
structure DecorationBox where
  tinsel : Nat
  tree : Nat
  snowGlobes : Nat

/-- Calculates the total number of decorations in a box -/
def totalDecorationsPerBox (box : DecorationBox) : Nat :=
  box.tinsel + box.tree + box.snowGlobes

/-- Theorem: The total number of decorations handed out is 120 -/
theorem total_decorations_handed_out :
  let standardBox : DecorationBox := { tinsel := 4, tree := 1, snowGlobes := 5 }
  let familyBoxes : Nat := 11
  let communityBoxes : Nat := 1
  totalDecorationsPerBox standardBox * (familyBoxes + communityBoxes) = 120 := by
  sorry

end total_decorations_handed_out_l708_70830


namespace fraction_sum_l708_70855

theorem fraction_sum (a b : ℚ) (h : a / b = 1 / 3) : (a + b) / b = 4 / 3 := by
  sorry

end fraction_sum_l708_70855


namespace percentage_of_l708_70814

theorem percentage_of (a b : ℝ) (h : b ≠ 0) :
  (a / b) * 100 = 250 → a = 150 ∧ b = 60 :=
by sorry

end percentage_of_l708_70814


namespace smallest_class_size_l708_70852

theorem smallest_class_size (total_students : ℕ) 
  (h1 : total_students ≥ 50)
  (h2 : ∃ (x : ℕ), total_students = 4 * x + (x + 2))
  (h3 : ∀ (y : ℕ), y ≥ 50 → (∃ (z : ℕ), y = 4 * z + (z + 2)) → y ≥ total_students) :
  total_students = 52 := by
sorry

end smallest_class_size_l708_70852


namespace difference_divisible_by_18_l708_70805

theorem difference_divisible_by_18 (a b : ℤ) : 
  18 ∣ ((3*a + 2)^2 - (3*b + 2)^2) := by sorry

end difference_divisible_by_18_l708_70805


namespace set_equality_l708_70875

def U : Set ℕ := Set.univ

def A : Set ℕ := {x | ∃ n : ℕ, x = 2 * n}

def B : Set ℕ := {x | ∃ n : ℕ, x = 4 * n}

theorem set_equality : U = A ∪ (U \ B) := by sorry

end set_equality_l708_70875


namespace determinant_equals_negative_two_l708_70816

-- Define the polynomial and its roots
def polynomial (p q : ℝ) (x : ℝ) : ℝ := x^3 - 3*p*x^2 + q*x - 2

-- Define the roots of the polynomial
def roots (p q : ℝ) : Set ℝ := {x | polynomial p q x = 0}

-- Assume the polynomial has exactly three roots
axiom three_roots (p q : ℝ) : ∃ (a b c : ℝ), roots p q = {a, b, c}

-- Define the determinant
def determinant (r a b c : ℝ) : ℝ :=
  (r + a) * ((r + b) * (r + c) - r^2) -
  r * (r * (r + c) - r^2) +
  r * (r * (r + b) - r^2)

-- State the theorem
theorem determinant_equals_negative_two (p q r : ℝ) :
  ∃ (a b c : ℝ), roots p q = {a, b, c} ∧ determinant r a b c = -2 := by
  sorry

end determinant_equals_negative_two_l708_70816


namespace fraction_equality_l708_70882

theorem fraction_equality (x : ℝ) : (5 + x) / (7 + x) = (3 + x) / (4 + x) → x = -1 := by
  sorry

end fraction_equality_l708_70882


namespace sum_of_pairwise_products_of_roots_l708_70802

theorem sum_of_pairwise_products_of_roots (p q r : ℂ) : 
  2 * p^3 - 4 * p^2 + 8 * p - 5 = 0 →
  2 * q^3 - 4 * q^2 + 8 * q - 5 = 0 →
  2 * r^3 - 4 * r^2 + 8 * r - 5 = 0 →
  p * q + q * r + p * r = 4 := by
  sorry

end sum_of_pairwise_products_of_roots_l708_70802


namespace line_segment_endpoint_l708_70851

/-- Given a line segment from (1, 3) to (x, -4) with length 15 and x > 0, prove x = 1 + √176 -/
theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  ((x - 1)^2 + (-4 - 3)^2).sqrt = 15 → 
  x = 1 + Real.sqrt 176 := by
  sorry

end line_segment_endpoint_l708_70851


namespace smallest_divisor_power_l708_70837

def Q (z : ℂ) : ℂ := z^10 + z^9 + z^6 + z^5 + z^4 + z + 1

theorem smallest_divisor_power : 
  ∃! k : ℕ, k > 0 ∧ 
  (∀ z : ℂ, Q z = 0 → z^k = 1) ∧
  (∀ m : ℕ, m > 0 → m < k → ∃ z : ℂ, Q z = 0 ∧ z^m ≠ 1) ∧
  k = 84 := by
sorry

end smallest_divisor_power_l708_70837


namespace x_value_from_fraction_equality_l708_70810

theorem x_value_from_fraction_equality (x y : ℝ) :
  x ≠ 2 →
  x / (x - 2) = (y^2 + 3*y - 4) / (y^2 + 3*y - 5) →
  x = 2*y^2 + 6*y - 8 := by
sorry

end x_value_from_fraction_equality_l708_70810


namespace mojave_population_increase_l708_70812

/-- Calculates the percentage increase between two populations -/
def percentageIncrease (initial : ℕ) (final : ℕ) : ℚ :=
  (final - initial : ℚ) / initial * 100

theorem mojave_population_increase : 
  let initialPopulation : ℕ := 4000
  let currentPopulation : ℕ := initialPopulation * 3
  let futurePopulation : ℕ := 16800
  percentageIncrease currentPopulation futurePopulation = 40 := by
sorry

end mojave_population_increase_l708_70812


namespace line_through_point_equal_intercepts_l708_70804

theorem line_through_point_equal_intercepts :
  ∃ (m b : ℝ), (3 = m * 2 + b) ∧ (∃ (a : ℝ), a ≠ 0 ∧ (a = -b/m ∧ a = b)) :=
by sorry

end line_through_point_equal_intercepts_l708_70804


namespace arrangements_count_l708_70831

/-- The number of students -/
def total_students : ℕ := 5

/-- The number of boys -/
def num_boys : ℕ := 2

/-- The number of girls -/
def num_girls : ℕ := 3

/-- A function that calculates the number of arrangements -/
def count_arrangements (n : ℕ) (b : ℕ) (g : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem arrangements_count :
  count_arrangements total_students num_boys num_girls = 48 :=
sorry

end arrangements_count_l708_70831


namespace tank_leak_emptying_time_l708_70836

/-- Given a tank that can be filled in 7 hours without a leak and 8 hours with a leak,
    prove that it takes 56 hours for the tank to become empty due to the leak. -/
theorem tank_leak_emptying_time :
  ∀ (fill_rate_no_leak fill_rate_with_leak leak_rate : ℚ),
    fill_rate_no_leak = 1 / 7 →
    fill_rate_with_leak = 1 / 8 →
    fill_rate_with_leak = fill_rate_no_leak - leak_rate →
    (1 : ℚ) / leak_rate = 56 := by
  sorry

end tank_leak_emptying_time_l708_70836


namespace parking_spaces_available_l708_70858

theorem parking_spaces_available (front_spaces back_spaces total_parked : ℕ) 
  (h1 : front_spaces = 52)
  (h2 : back_spaces = 38)
  (h3 : total_parked = 39)
  (h4 : total_parked = front_spaces + back_spaces / 2) : 
  front_spaces + back_spaces - total_parked = 51 := by
  sorry

end parking_spaces_available_l708_70858


namespace max_value_operation_l708_70887

theorem max_value_operation (n : ℕ) (h : 100 ≤ n ∧ n ≤ 999) :
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 → (300 - m)^2 - 10 ≤ (300 - n)^2 - 10) →
  (300 - n)^2 - 10 = 39990 :=
by sorry

end max_value_operation_l708_70887


namespace mean_steps_per_day_l708_70844

theorem mean_steps_per_day (total_steps : ℕ) (num_days : ℕ) (h1 : total_steps = 243000) (h2 : num_days = 30) :
  total_steps / num_days = 8100 := by
  sorry

end mean_steps_per_day_l708_70844


namespace negative_sqrt_of_square_of_negative_three_l708_70857

theorem negative_sqrt_of_square_of_negative_three :
  -Real.sqrt ((-3)^2) = -3 := by sorry

end negative_sqrt_of_square_of_negative_three_l708_70857


namespace closest_to_one_l708_70874

theorem closest_to_one : 
  let numbers : List ℝ := [3/4, 1.2, 0.81, 4/3, 7/10]
  ∀ x ∈ numbers, |0.81 - 1| ≤ |x - 1| := by
sorry

end closest_to_one_l708_70874


namespace field_area_diminished_l708_70803

theorem field_area_diminished (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let original_area := L * W
  let new_length := L * (1 - 0.4)
  let new_width := W * (1 - 0.4)
  let new_area := new_length * new_width
  (original_area - new_area) / original_area = 0.64 := by sorry

end field_area_diminished_l708_70803


namespace plant_branches_problem_l708_70818

theorem plant_branches_problem (x : ℕ) : 
  (1 + x + x^2 = 43) → (x = 6) :=
by
  sorry

end plant_branches_problem_l708_70818


namespace power_simplification_l708_70869

theorem power_simplification : 16^10 * 8^5 / 4^15 = 2^25 := by
  sorry

end power_simplification_l708_70869


namespace negative_x_exponent_product_l708_70832

theorem negative_x_exponent_product (x : ℝ) : (-x)^3 * (-x)^4 = -x^7 := by sorry

end negative_x_exponent_product_l708_70832


namespace digit_matching_equality_l708_70870

theorem digit_matching_equality : ∃ (a b : ℕ), 
  a ≠ b ∧ 
  a ≤ 99 ∧ 
  b ≤ 99 ∧ 
  a + b ≤ 9999 ∧ 
  (a + b)^2 = 100 * a + b :=
sorry

end digit_matching_equality_l708_70870


namespace prob_three_same_color_l708_70800

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 6

/-- The number of white marbles in the bag -/
def white_marbles : ℕ := 7

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 8

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 4

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

/-- The probability of drawing three marbles of the same color without replacement -/
theorem prob_three_same_color : 
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) + 
   white_marbles * (white_marbles - 1) * (white_marbles - 2) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) +
   green_marbles * (green_marbles - 1) * (green_marbles - 2)) / 
  (total_marbles * (total_marbles - 1) * (total_marbles - 2)) = 1 / 20 := by
  sorry

end prob_three_same_color_l708_70800


namespace gcd_of_special_powers_l708_70877

theorem gcd_of_special_powers :
  Nat.gcd (2^2020 - 1) (2^2000 - 1) = 2^20 - 1 := by
  sorry

end gcd_of_special_powers_l708_70877


namespace average_apples_per_guest_l708_70819

/-- Represents the number of apples per serving -/
def apples_per_serving : ℚ := 3/2

/-- Represents the ratio of Red Delicious to Granny Smith apples per serving -/
def apple_ratio : ℚ := 2

/-- Represents the number of guests -/
def num_guests : ℕ := 12

/-- Represents the number of pies -/
def num_pies : ℕ := 3

/-- Represents the number of servings per pie -/
def servings_per_pie : ℕ := 8

/-- Represents the number of cups of apple pieces per Red Delicious apple -/
def red_delicious_cups : ℚ := 1

/-- Represents the number of cups of apple pieces per Granny Smith apple -/
def granny_smith_cups : ℚ := 5/4

/-- Theorem stating that the average number of apples each guest eats is 3 -/
theorem average_apples_per_guest : 
  (num_pies * servings_per_pie * apples_per_serving) / num_guests = 3 := by
  sorry

end average_apples_per_guest_l708_70819


namespace van_capacity_l708_70841

theorem van_capacity (students : ℕ) (adults : ℕ) (vans : ℕ) (h1 : students = 2) (h2 : adults = 6) (h3 : vans = 2) :
  (students + adults) / vans = 4 := by
  sorry

end van_capacity_l708_70841


namespace book_sales_calculation_l708_70860

/-- Calculates the total book sales over three days given specific sales patterns. -/
theorem book_sales_calculation (day1_sales : ℕ) : 
  day1_sales = 15 →
  (day1_sales + 3 * day1_sales + (3 * day1_sales) / 5 : ℕ) = 69 := by
  sorry

end book_sales_calculation_l708_70860


namespace peace_treaty_day_l708_70827

def day_of_week : Fin 7 → String
| 0 => "Sunday"
| 1 => "Monday"
| 2 => "Tuesday"
| 3 => "Wednesday"
| 4 => "Thursday"
| 5 => "Friday"
| 6 => "Saturday"

def days_between : Nat := 919

theorem peace_treaty_day :
  let start_day : Fin 7 := 4  -- Thursday
  let end_day : Fin 7 := (start_day + days_between) % 7
  day_of_week end_day = "Saturday" := by
  sorry


end peace_treaty_day_l708_70827


namespace right_triangle_hypotenuse_l708_70843

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Ensure positive side lengths
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a^2 + b^2 + c^2 = 980 →  -- Given condition
  c = 70 := by
sorry

end right_triangle_hypotenuse_l708_70843


namespace three_prime_divisors_special_form_l708_70888

theorem three_prime_divisors_special_form (n : ℕ) (x : ℕ) : 
  x = 2^n - 32 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 2 ∧ q ≠ 2 ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ x → r = 2 ∨ r = p ∨ r = q)) →
  x = 2016 ∨ x = 16352 := by
sorry

end three_prime_divisors_special_form_l708_70888


namespace shaltaev_boltaev_inequality_l708_70876

theorem shaltaev_boltaev_inequality (S B : ℝ) 
  (h1 : S > 0) (h2 : B > 0) 
  (h3 : 175 * S > 125 * B) (h4 : 175 * S < 126 * B) : 
  3 * S + B > S := by
sorry

end shaltaev_boltaev_inequality_l708_70876


namespace tangent_circles_m_value_l708_70815

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Circle C₂ with equation x² + y² - 6x - 8y + m = 0 -/
def C₂ (m : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 - 8*p.2 + m = 0}

/-- Two circles are tangent if they intersect at exactly one point -/
def AreTangent (A B : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ A ∧ p ∈ B

/-- The main theorem: If C₁ and C₂ are tangent, then m = 9 -/
theorem tangent_circles_m_value :
  AreTangent C₁ (C₂ 9) :=
sorry

end tangent_circles_m_value_l708_70815


namespace bake_sale_fundraiser_l708_70867

/-- 
Given a bake sale that earned $400 total, prove that the amount kept for ingredients
is $100, when half of the remaining amount plus $10 equals $160.
-/
theorem bake_sale_fundraiser (total_earnings : ℝ) (donation_to_shelter : ℝ) :
  total_earnings = 400 ∧ 
  donation_to_shelter = 160 ∧
  donation_to_shelter = (total_earnings - (total_earnings - donation_to_shelter + 10)) / 2 + 10 →
  total_earnings - donation_to_shelter + 10 = 100 := by
sorry

end bake_sale_fundraiser_l708_70867


namespace no_linear_term_implies_n_eq_neg_two_l708_70839

theorem no_linear_term_implies_n_eq_neg_two (n : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + n) * (x + 2) = a * x^2 + b) → n = -2 := by
  sorry

end no_linear_term_implies_n_eq_neg_two_l708_70839


namespace quiz_probability_correct_l708_70846

/-- Represents a quiz with one MCQ and two True/False questions -/
structure Quiz where
  mcq_options : Nat
  tf_questions : Nat

/-- Calculates the probability of answering all questions correctly in a quiz -/
def probability_all_correct (q : Quiz) : ℚ :=
  (1 : ℚ) / q.mcq_options * ((1 : ℚ) / 2) ^ q.tf_questions

/-- Theorem: The probability of answering all questions correctly in the given quiz is 1/12 -/
theorem quiz_probability_correct :
  let q := Quiz.mk 3 2
  probability_all_correct q = 1 / 12 := by
  sorry


end quiz_probability_correct_l708_70846


namespace camillas_jelly_beans_l708_70895

theorem camillas_jelly_beans (b c : ℕ) : 
  b = 2 * c →                     -- Initial condition: twice as many blueberry as cherry
  b - 10 = 3 * (c - 10) →         -- Condition after eating: three times as many blueberry as cherry
  b = 40                          -- Conclusion: original number of blueberry jelly beans
:= by sorry

end camillas_jelly_beans_l708_70895


namespace olivers_bags_weight_l708_70806

theorem olivers_bags_weight (james_bag_weight : ℝ) (oliver_bag_ratio : ℝ) : 
  james_bag_weight = 18 →
  oliver_bag_ratio = 1 / 6 →
  2 * (oliver_bag_ratio * james_bag_weight) = 6 := by
  sorry

end olivers_bags_weight_l708_70806


namespace digit_difference_digit_difference_proof_l708_70801

theorem digit_difference : ℕ → Prop :=
  fun n =>
    (∀ m : ℕ, m < 1000 → m < n) ∧
    (n < 10000) ∧
    (∀ k : ℕ, k < 1000) →
    n - 999 = 1

-- The proof
theorem digit_difference_proof : digit_difference 1000 := by
  sorry

end digit_difference_digit_difference_proof_l708_70801


namespace negation_of_proposition_negation_of_exponential_inequality_l708_70866

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) := by sorry

theorem negation_of_exponential_inequality :
  (¬ ∀ x : ℝ, Real.exp x ≥ 1) ↔ (∃ x : ℝ, Real.exp x < 1) := by sorry

end negation_of_proposition_negation_of_exponential_inequality_l708_70866


namespace polynomial_form_l708_70808

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The functional equation that P must satisfy -/
def SatisfiesEquation (P : RealPolynomial) : Prop :=
  ∀ (x : ℝ), x ≠ 0 → P x + P (1/x) = (P (x + 1/x) + P (x - 1/x)) / 2

/-- The form of the polynomial we want to prove -/
def HasRequiredForm (P : RealPolynomial) : Prop :=
  ∃ (a b : ℝ), ∀ (x : ℝ), P x = a * x^4 + b * x^2 + 6 * a

theorem polynomial_form (P : RealPolynomial) :
  SatisfiesEquation P → HasRequiredForm P :=
by sorry

end polynomial_form_l708_70808


namespace era_burgers_l708_70892

theorem era_burgers (num_friends : ℕ) (slices_per_burger : ℕ) 
  (friend1_slices : ℕ) (friend2_slices : ℕ) (friend3_slices : ℕ) 
  (friend4_slices : ℕ) (era_slices : ℕ) :
  num_friends = 4 →
  slices_per_burger = 2 →
  friend1_slices = 1 →
  friend2_slices = 2 →
  friend3_slices = 3 →
  friend4_slices = 3 →
  era_slices = 1 →
  (friend1_slices + friend2_slices + friend3_slices + friend4_slices + era_slices) / slices_per_burger = 5 := by
  sorry

end era_burgers_l708_70892


namespace two_people_three_movies_l708_70809

/-- The number of ways two people can choose tickets from three movies -/
def ticket_choices (num_people : ℕ) (num_movies : ℕ) : ℕ :=
  num_movies ^ num_people

/-- Theorem: Two people choosing from three movies results in 9 different combinations -/
theorem two_people_three_movies :
  ticket_choices 2 3 = 9 := by
  sorry

end two_people_three_movies_l708_70809


namespace cube_rotation_invariance_l708_70893

-- Define a cube
structure Cube where
  position : ℕ × ℕ  -- Position on the plane
  topFace : Fin 6   -- Top face (numbered 1 to 6)
  rotation : Fin 4  -- Rotation of top face (0, 90, 180, or 270 degrees)

-- Define a roll operation
def roll (c : Cube) : Cube :=
  sorry

-- Define a sequence of rolls
def rollSequence (c : Cube) (n : ℕ) : Cube :=
  sorry

-- Theorem statement
theorem cube_rotation_invariance (c : Cube) (n : ℕ) :
  let c' := rollSequence c n
  c'.position = c.position ∧ c'.topFace = c.topFace →
  c'.rotation = c.rotation :=
sorry

end cube_rotation_invariance_l708_70893


namespace technician_avg_salary_l708_70824

def total_workers : ℕ := 24
def avg_salary_all : ℕ := 8000
def num_technicians : ℕ := 8
def avg_salary_non_tech : ℕ := 6000

theorem technician_avg_salary :
  let num_non_tech := total_workers - num_technicians
  let total_salary := avg_salary_all * total_workers
  let total_salary_non_tech := avg_salary_non_tech * num_non_tech
  let total_salary_tech := total_salary - total_salary_non_tech
  total_salary_tech / num_technicians = 12000 := by
  sorry

end technician_avg_salary_l708_70824


namespace city_fuel_efficiency_l708_70862

/-- Represents the fuel efficiency of a car in miles per gallon -/
structure FuelEfficiency where
  highway : ℝ
  city : ℝ

/-- Represents the distance a car can travel on a full tank in miles -/
structure TankDistance where
  highway : ℝ
  city : ℝ

theorem city_fuel_efficiency 
  (fe : FuelEfficiency) 
  (td : TankDistance) 
  (h1 : fe.city = fe.highway - 9)
  (h2 : td.highway = 462)
  (h3 : td.city = 336)
  (h4 : fe.highway * (td.city / fe.city) = td.highway) :
  fe.city = 24 := by
  sorry

end city_fuel_efficiency_l708_70862


namespace function_inequality_l708_70825

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x > f x) : f 2 > Real.exp 2 * f 0 := by
  sorry

end function_inequality_l708_70825


namespace pirate_loot_sum_l708_70833

def base5ToBase10 (n : List Nat) : Nat :=
  List.foldl (fun acc d => acc * 5 + d) 0 (List.reverse n)

theorem pirate_loot_sum :
  let silver := base5ToBase10 [1, 4, 3, 2]
  let spices := base5ToBase10 [2, 1, 3, 4]
  let silk := base5ToBase10 [3, 0, 2, 1]
  let books := base5ToBase10 [2, 3, 1]
  silver + spices + silk + books = 988 := by
  sorry

end pirate_loot_sum_l708_70833


namespace special_number_exists_l708_70807

def digit_product (n : ℕ) : ℕ := sorry

def digit_sum (n : ℕ) : ℕ := sorry

def is_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem special_number_exists : ∃ x : ℕ, 
  (digit_product x = 44 * x - 86868) ∧ 
  (is_cube (digit_sum x)) := by sorry

end special_number_exists_l708_70807


namespace installment_payment_installment_payment_proof_l708_70835

theorem installment_payment (cash_price : ℕ) (down_payment : ℕ) (first_four : ℕ) 
  (last_four : ℕ) (installment_difference : ℕ) : ℕ :=
  let total_installment := cash_price + installment_difference
  let first_four_total := 4 * first_four
  let last_four_total := 4 * last_four
  let middle_four_total := total_installment - down_payment - first_four_total - last_four_total
  let middle_four_monthly := middle_four_total / 4
  middle_four_monthly

#check @installment_payment

theorem installment_payment_proof 
  (h1 : installment_payment 450 100 40 30 70 = 35) : True := by
  sorry

end installment_payment_installment_payment_proof_l708_70835


namespace max_probability_zero_units_digit_l708_70864

def probability_zero_units_digit (N : ℕ+) : ℚ :=
  let q2 := (N / 2 : ℚ) / N
  let q5 := (N / 5 : ℚ) / N
  let q10 := (N / 10 : ℚ) / N
  q10 * (2 - q10) + 2 * (q2 - q10) * (q5 - q10)

theorem max_probability_zero_units_digit :
  ∀ N : ℕ+, probability_zero_units_digit N ≤ 27/100 := by
  sorry

end max_probability_zero_units_digit_l708_70864


namespace polygon_sides_from_diagonals_l708_70811

theorem polygon_sides_from_diagonals (D : ℕ) (n : ℕ) : D = n * (n - 3) / 2 → D = 44 → n = 11 := by
  sorry

end polygon_sides_from_diagonals_l708_70811


namespace total_shared_amount_l708_70853

def ken_share : ℕ := 1750
def tony_share : ℕ := 2 * ken_share

theorem total_shared_amount : ken_share + tony_share = 5250 := by
  sorry

end total_shared_amount_l708_70853


namespace trajectory_properties_line_intersection_condition_unique_k_for_dot_product_l708_70848

noncomputable def trajectory (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y - 1)^2) = Real.sqrt 2 * Real.sqrt ((x - 1)^2 + (y - 2)^2)

def line_intersects (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x - 1

theorem trajectory_properties :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, trajectory x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    center_x = 2 ∧ center_y = 3 ∧ radius = 2 :=
sorry

theorem line_intersection_condition (k : ℝ) :
  (∃ x y, trajectory x y ∧ line_intersects k x y) ↔ k > 3/4 :=
sorry

theorem unique_k_for_dot_product :
  ∃! k, k > 3/4 ∧
    ∀ x₁ y₁ x₂ y₂,
      trajectory x₁ y₁ ∧ trajectory x₂ y₂ ∧
      line_intersects k x₁ y₁ ∧ line_intersects k x₂ y₂ →
      x₁ * x₂ + y₁ * y₂ = 11 :=
sorry

end trajectory_properties_line_intersection_condition_unique_k_for_dot_product_l708_70848


namespace algebraic_expression_value_l708_70820

theorem algebraic_expression_value :
  ∀ x : ℝ, x = 2 * Real.sqrt 3 - 1 → x^2 + 2*x - 3 = 8 := by
  sorry

end algebraic_expression_value_l708_70820


namespace tangent_parallel_points_l708_70822

/-- The function f(x) = x³ + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_parallel_points :
  {x : ℝ | f' x = 4} = {1, -1} ∧
  f 1 = 0 ∧ f (-1) = -4 := by
  sorry

end tangent_parallel_points_l708_70822


namespace riverside_denial_rate_l708_70896

theorem riverside_denial_rate (total_kids : ℕ) (riverside_kids : ℕ) (westside_kids : ℕ) (mountaintop_kids : ℕ)
  (westside_denial_rate : ℚ) (mountaintop_denial_rate : ℚ) (kids_admitted : ℕ) :
  total_kids = riverside_kids + westside_kids + mountaintop_kids →
  total_kids = 260 →
  riverside_kids = 120 →
  westside_kids = 90 →
  mountaintop_kids = 50 →
  westside_denial_rate = 7/10 →
  mountaintop_denial_rate = 1/2 →
  kids_admitted = 148 →
  (riverside_kids - (total_kids - kids_admitted - (westside_denial_rate * westside_kids).num
    - (mountaintop_denial_rate * mountaintop_kids).num)) / riverside_kids = 4/5 := by
  sorry

end riverside_denial_rate_l708_70896


namespace summer_reading_challenge_l708_70821

def books_to_coupons (books : ℕ) : ℕ := books / 5

def quinn_books : ℕ := 5 * 5

def taylor_books : ℕ := 1 + 4 * 9

def jordan_books : ℕ := 3 * 10

theorem summer_reading_challenge : 
  books_to_coupons quinn_books + books_to_coupons taylor_books + books_to_coupons jordan_books = 18 := by
  sorry

end summer_reading_challenge_l708_70821


namespace shell_difference_l708_70883

theorem shell_difference (perfect_shells broken_shells non_spiral_perfect : ℕ) 
  (h1 : perfect_shells = 17)
  (h2 : broken_shells = 52)
  (h3 : non_spiral_perfect = 12) : 
  (broken_shells / 2) - (perfect_shells - non_spiral_perfect) = 21 := by
  sorry

end shell_difference_l708_70883


namespace seed_selection_correct_l708_70889

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Checks if a number is a valid seed number --/
def isValidSeed (n : Nat) : Bool :=
  0 < n && n ≤ 500

/-- Extracts the next three-digit number from the random number table --/
def nextThreeDigitNumber (table : RandomNumberTable) (row : Nat) (col : Nat) : Option Nat :=
  sorry

/-- Selects the first n valid seeds from the random number table --/
def selectValidSeeds (table : RandomNumberTable) (startRow : Nat) (startCol : Nat) (n : Nat) : List Nat :=
  sorry

/-- The given random number table --/
def givenTable : RandomNumberTable := [
  [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67],
  [21, 76, 33, 50, 25, 83, 92, 12, 06, 76],
  [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75],
  [12, 86, 73, 58, 07, 44, 39, 52, 38, 79],
  [33, 21, 12, 34, 29, 78, 64, 56, 07, 82, 52, 42, 07, 44, 38],
  [15, 51, 00, 13, 42, 99, 66, 02, 79, 54]
]

theorem seed_selection_correct :
  selectValidSeeds givenTable 7 8 5 = [331, 455, 068, 047, 447] :=
sorry

end seed_selection_correct_l708_70889


namespace correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_C_l708_70890

theorem correct_calculation : 2 * Real.sqrt 3 - Real.sqrt 3 = Real.sqrt 3 :=
by sorry

theorem incorrect_calculation_A : ¬(Real.sqrt 3 + Real.sqrt 2 = Real.sqrt 5) :=
by sorry

theorem incorrect_calculation_B : ¬(Real.sqrt 3 * Real.sqrt 5 = 15) :=
by sorry

theorem incorrect_calculation_C : ¬(Real.sqrt 32 / Real.sqrt 8 = 2 ∨ Real.sqrt 32 / Real.sqrt 8 = -2) :=
by sorry

end correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_C_l708_70890
