import Mathlib

namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1429_142930

theorem interest_rate_calculation (total_sum : ℝ) (second_part : ℝ) : 
  total_sum = 2665 →
  second_part = 1332.5 →
  let first_part := total_sum - second_part
  let interest_first := first_part * 0.03 * 5
  let interest_second := second_part * 0.03 * 3 * (5 : ℝ) / 3
  interest_first = interest_second →
  5 = 100 * interest_second / (second_part * 3) := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1429_142930


namespace NUMINAMATH_CALUDE_one_intersection_condition_tangent_lines_at_point_l1429_142986

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem for the range of m
theorem one_intersection_condition (m : ℝ) :
  (∃! x, f x = m) ↔ (m < -2 ∨ m > 2) :=
sorry

-- Theorem for the tangent lines
theorem tangent_lines_at_point :
  let P : ℝ × ℝ := (2, -6)
  ∃ (l₁ l₂ : ℝ → ℝ),
    (∀ x, l₁ x = -3*x) ∧
    (∀ x, l₂ x = 24*x - 54) ∧
    (∀ t, ∃ x, (x, f x) = (t, l₁ t) ∨ (x, f x) = (t, l₂ t)) ∧
    (l₁ 2 = -6) ∧ (l₂ 2 = -6) :=
sorry

end NUMINAMATH_CALUDE_one_intersection_condition_tangent_lines_at_point_l1429_142986


namespace NUMINAMATH_CALUDE_consecutive_pages_product_l1429_142988

theorem consecutive_pages_product (n : ℕ) : 
  n > 0 → n + (n + 1) = 217 → n * (n + 1) = 11772 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_product_l1429_142988


namespace NUMINAMATH_CALUDE_train_meeting_point_l1429_142983

theorem train_meeting_point 
  (route_length : ℝ) 
  (speed_A : ℝ) 
  (speed_B : ℝ) 
  (h1 : route_length = 75)
  (h2 : speed_A = 25)
  (h3 : speed_B = 37.5)
  : (route_length * speed_A) / (speed_A + speed_B) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_point_l1429_142983


namespace NUMINAMATH_CALUDE_expression_evaluation_l1429_142919

theorem expression_evaluation (a b : ℤ) (h1 : a = 4) (h2 : b = -3) :
  -2*a - b^2 + 2*a*b = -41 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1429_142919


namespace NUMINAMATH_CALUDE_profit_starts_third_year_average_profit_plan_more_effective_l1429_142925

/-- Represents a fishing company's financial situation -/
structure FishingCompany where
  initialCost : ℕ
  firstYearExpenses : ℕ
  annualExpenseIncrement : ℕ
  annualIncome : ℕ

/-- Calculates the cumulative expenses after n years -/
def cumulativeExpenses (company : FishingCompany) (n : ℕ) : ℕ :=
  company.initialCost + n * company.firstYearExpenses + (n * (n - 1) / 2) * company.annualExpenseIncrement

/-- Calculates the cumulative income after n years -/
def cumulativeIncome (company : FishingCompany) (n : ℕ) : ℕ :=
  n * company.annualIncome

/-- Determines if the company is profitable after n years -/
def isProfitable (company : FishingCompany) (n : ℕ) : Prop :=
  cumulativeIncome company n > cumulativeExpenses company n

/-- Represents the two selling plans -/
inductive SellingPlan
  | AverageProfit
  | TotalNetIncome

/-- Theorem: The company begins to profit in the third year -/
theorem profit_starts_third_year (company : FishingCompany) 
  (h1 : company.initialCost = 490000)
  (h2 : company.firstYearExpenses = 60000)
  (h3 : company.annualExpenseIncrement = 20000)
  (h4 : company.annualIncome = 250000) :
  isProfitable company 3 ∧ ¬isProfitable company 2 :=
sorry

/-- Theorem: The average annual profit plan is more cost-effective -/
theorem average_profit_plan_more_effective (company : FishingCompany) 
  (h1 : company.initialCost = 490000)
  (h2 : company.firstYearExpenses = 60000)
  (h3 : company.annualExpenseIncrement = 20000)
  (h4 : company.annualIncome = 250000) :
  ∃ (n m : ℕ), 
    (∀ k, cumulativeIncome company k - cumulativeExpenses company k + 180000 ≤ n) ∧
    (∀ k, cumulativeIncome company k - cumulativeExpenses company k + 90000 ≤ m) ∧
    n > m :=
sorry

end NUMINAMATH_CALUDE_profit_starts_third_year_average_profit_plan_more_effective_l1429_142925


namespace NUMINAMATH_CALUDE_coronavirus_case_increase_l1429_142952

theorem coronavirus_case_increase (initial_cases : ℕ) 
  (second_day_recoveries : ℕ) (third_day_new_cases : ℕ) 
  (third_day_recoveries : ℕ) (final_total_cases : ℕ) :
  initial_cases = 2000 →
  second_day_recoveries = 50 →
  third_day_new_cases = 1500 →
  third_day_recoveries = 200 →
  final_total_cases = 3750 →
  ∃ (second_day_increase : ℕ),
    final_total_cases = initial_cases + second_day_increase - second_day_recoveries + 
      third_day_new_cases - third_day_recoveries ∧
    second_day_increase = 750 :=
by sorry

end NUMINAMATH_CALUDE_coronavirus_case_increase_l1429_142952


namespace NUMINAMATH_CALUDE_equation_proof_l1429_142961

theorem equation_proof : 529 + 2 * 23 * 11 + 121 = 1156 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1429_142961


namespace NUMINAMATH_CALUDE_problem_solution_l1429_142916

theorem problem_solution (a b : ℝ) (h1 : 0 < b) (h2 : b < 1/2) (h3 : 1/2 < a) (h4 : a < 1) :
  (0 < a - b) ∧ (a - b < 1) ∧ (a * b < a^2) ∧ (a - 1/b < b - 1/a) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1429_142916


namespace NUMINAMATH_CALUDE_roots_cube_theorem_l1429_142927

theorem roots_cube_theorem (a b c d x₁ x₂ : ℝ) :
  (x₁^2 - (a + d)*x₁ + ad - bc = 0) →
  (x₂^2 - (a + d)*x₂ + ad - bc = 0) →
  (x₁^3)^2 - (a^3 + d^3 + 3*a*b*c + 3*b*c*d)*(x₁^3) + (a*d - b*c)^3 = 0 ∧
  (x₂^3)^2 - (a^3 + d^3 + 3*a*b*c + 3*b*c*d)*(x₂^3) + (a*d - b*c)^3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_cube_theorem_l1429_142927


namespace NUMINAMATH_CALUDE_initial_alloy_weight_l1429_142951

/-- Represents the composition of an alloy --/
structure Alloy where
  zinc : ℝ
  copper : ℝ

/-- The initial ratio of zinc to copper in the alloy --/
def initial_ratio : ℚ := 5 / 3

/-- The final ratio of zinc to copper after adding zinc --/
def final_ratio : ℚ := 3 / 1

/-- The amount of zinc added to the alloy --/
def added_zinc : ℝ := 8

/-- Theorem stating the initial weight of the alloy --/
theorem initial_alloy_weight (a : Alloy) :
  (a.zinc / a.copper = initial_ratio) →
  ((a.zinc + added_zinc) / a.copper = final_ratio) →
  (a.zinc + a.copper = 16) :=
by sorry

end NUMINAMATH_CALUDE_initial_alloy_weight_l1429_142951


namespace NUMINAMATH_CALUDE_log_equation_solution_l1429_142903

theorem log_equation_solution (t : ℝ) (h : t > 0) :
  4 * (Real.log t / Real.log 3) = Real.log (4 * t) / Real.log 3 → t = (4 : ℝ) ^ (1 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1429_142903


namespace NUMINAMATH_CALUDE_journey_average_speed_l1429_142980

/-- Calculates the average speed given distances in meters and times in minutes -/
def average_speed (distances : List Float) (times : List Float) : Float :=
  let total_distance := (distances.sum / 1000)  -- Convert to km
  let total_time := (times.sum / 60)  -- Convert to hours
  total_distance / total_time

/-- Theorem: The average speed for the given journey is 6 km/h -/
theorem journey_average_speed :
  let distances := [1000, 1500, 2000]
  let times := [10, 15, 20]
  average_speed distances times = 6 := by
sorry

#eval average_speed [1000, 1500, 2000] [10, 15, 20]

end NUMINAMATH_CALUDE_journey_average_speed_l1429_142980


namespace NUMINAMATH_CALUDE_composite_product_quotient_l1429_142970

def first_five_composites : List ℕ := [21, 22, 24, 25, 26]
def next_five_composites : List ℕ := [27, 28, 30, 32, 33]

def product_list (l : List ℕ) : ℕ := l.foldl (·*·) 1

theorem composite_product_quotient :
  (product_list first_five_composites : ℚ) / (product_list next_five_composites) = 1 / 1964 := by
  sorry

end NUMINAMATH_CALUDE_composite_product_quotient_l1429_142970


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l1429_142950

theorem lcm_gcd_problem (x y : ℕ+) : 
  Nat.lcm x y = 5940 → 
  Nat.gcd x y = 22 → 
  x = 220 → 
  y = 594 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l1429_142950


namespace NUMINAMATH_CALUDE_not_divisible_by_two_or_five_l1429_142967

def T : Set ℤ := {x | ∃ n : ℤ, x = (n - 3)^2 + (n - 1)^2 + (n + 1)^2 + (n + 3)^2}

theorem not_divisible_by_two_or_five :
  ∀ x ∈ T, ¬(∃ k : ℤ, x = 2 * k ∨ x = 5 * k) :=
by sorry

end NUMINAMATH_CALUDE_not_divisible_by_two_or_five_l1429_142967


namespace NUMINAMATH_CALUDE_circle_radius_problem_l1429_142953

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c2.radius - c1.radius)^2

/-- Checks if two circles are congruent -/
def are_congruent (c1 c2 : Circle) : Prop :=
  c1.radius = c2.radius

theorem circle_radius_problem (A B C D : Circle) :
  are_externally_tangent A B ∧
  are_externally_tangent A C ∧
  are_externally_tangent B C ∧
  is_internally_tangent A D ∧
  is_internally_tangent B D ∧
  is_internally_tangent C D ∧
  are_congruent B C ∧
  A.radius = 1 ∧
  (let (x, y) := D.center; (x - A.center.1)^2 + (y - A.center.2)^2 = A.radius^2) →
  B.radius = 8/9 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l1429_142953


namespace NUMINAMATH_CALUDE_triple_sharp_100_l1429_142957

-- Define the # function
def sharp (N : ℝ) : ℝ := 0.6 * N + 1

-- State the theorem
theorem triple_sharp_100 : sharp (sharp (sharp 100)) = 23.56 := by
  sorry

end NUMINAMATH_CALUDE_triple_sharp_100_l1429_142957


namespace NUMINAMATH_CALUDE_f_times_g_equals_one_l1429_142924

/-- The formal power series f(x) defined as an infinite geometric series -/
noncomputable def f (x : ℝ) : ℝ := ∑' n, x^n

/-- The function g(x) defined as 1 - x -/
def g (x : ℝ) : ℝ := 1 - x

/-- Theorem stating that f(x)g(x) = 1 -/
theorem f_times_g_equals_one (x : ℝ) (hx : |x| < 1) : f x * g x = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_times_g_equals_one_l1429_142924


namespace NUMINAMATH_CALUDE_loan_years_approx_eight_l1429_142991

/-- Calculates the number of years for which the first part of a loan is lent, given the total sum,
    the second part, and interest rates for both parts. -/
def calculate_years (total : ℚ) (second_part : ℚ) (rate1 : ℚ) (rate2 : ℚ) : ℚ :=
  let first_part := total - second_part
  let n := (second_part * rate2 * 3) / (first_part * rate1)
  n

/-- Proves that given the specified conditions, the number of years for which
    the first part is lent is approximately 8. -/
theorem loan_years_approx_eight :
  let total := 2691
  let second_part := 1656
  let rate1 := 3 / 100
  let rate2 := 5 / 100
  let years := calculate_years total second_part rate1 rate2
  ∃ ε > 0, abs (years - 8) < ε := by
  sorry


end NUMINAMATH_CALUDE_loan_years_approx_eight_l1429_142991


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1429_142931

-- Problem 1
theorem problem_1 : (1 - Real.sqrt 3) ^ 0 - |-(Real.sqrt 2)| + ((-27) ^ (1/3 : ℝ)) - ((-1/2) ^ (-1 : ℝ)) = -(Real.sqrt 2) := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x ≠ 2) : 
  ((x^2 - 1) / (x - 2) - x - 1) / ((x + 1) / (x^2 - 4*x + 4)) = x - 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1429_142931


namespace NUMINAMATH_CALUDE_no_more_birds_can_join_l1429_142995

/-- Represents the weight capacity of the fence in pounds -/
def fence_capacity : ℝ := 20

/-- Represents the weight of the first bird in pounds -/
def bird1_weight : ℝ := 2.5

/-- Represents the weight of the second bird in pounds -/
def bird2_weight : ℝ := 3.5

/-- Represents the number of additional birds that joined -/
def additional_birds : ℕ := 4

/-- Represents the weight of each additional bird in pounds -/
def additional_bird_weight : ℝ := 2.8

/-- Represents the weight of each new bird that might join in pounds -/
def new_bird_weight : ℝ := 3

/-- Calculates the total weight of birds currently on the fence -/
def current_weight : ℝ := bird1_weight + bird2_weight + additional_birds * additional_bird_weight

/-- Theorem stating that no more 3 lb birds can join the fence without exceeding its capacity -/
theorem no_more_birds_can_join : 
  ∀ n : ℕ, current_weight + n * new_bird_weight ≤ fence_capacity → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_more_birds_can_join_l1429_142995


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l1429_142913

theorem system_of_equations_solutions :
  -- System 1
  (∃ x y : ℚ, 2 * x + 4 * y = 5 ∧ x = 1 - y) →
  (∃ x y : ℚ, 2 * x + 4 * y = 5 ∧ x = 1 - y ∧ x = -1/2 ∧ y = 3/2) ∧
  -- System 2
  (∃ x y : ℚ, 5 * x + 6 * y = 4 ∧ 3 * x - 4 * y = 10) →
  (∃ x y : ℚ, 5 * x + 6 * y = 4 ∧ 3 * x - 4 * y = 10 ∧ x = 2 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l1429_142913


namespace NUMINAMATH_CALUDE_dance_attendance_l1429_142936

theorem dance_attendance (girls boys : ℕ) : 
  boys = 2 * girls ∧ 
  boys = (girls - 1) + 8 → 
  boys = 14 := by
sorry

end NUMINAMATH_CALUDE_dance_attendance_l1429_142936


namespace NUMINAMATH_CALUDE_father_son_age_sum_father_son_age_sum_proof_l1429_142907

/-- The sum of the present ages of a father and son, given specific age relationships -/
theorem father_son_age_sum : ℕ → ℕ → Prop :=
  fun son_age father_age =>
    (father_age - 18 = 3 * (son_age - 18)) ∧  -- 18 years ago relationship
    (father_age = 2 * son_age) →              -- current relationship
    son_age + father_age = 108                -- sum of present ages

/-- Proof of the father_son_age_sum theorem -/
theorem father_son_age_sum_proof : ∃ (son_age father_age : ℕ), father_son_age_sum son_age father_age := by
  sorry

#check father_son_age_sum
#check father_son_age_sum_proof

end NUMINAMATH_CALUDE_father_son_age_sum_father_son_age_sum_proof_l1429_142907


namespace NUMINAMATH_CALUDE_common_tangent_parabola_log_l1429_142938

theorem common_tangent_parabola_log (a s t : ℝ) : 
  a > 0 → 
  t = a * s^2 → 
  t = Real.log s → 
  (2 * a * s) = (1 / s) → 
  a = 1 / (2 * Real.exp 1) := by
sorry

end NUMINAMATH_CALUDE_common_tangent_parabola_log_l1429_142938


namespace NUMINAMATH_CALUDE_lcm_gcd_1365_910_l1429_142926

theorem lcm_gcd_1365_910 :
  (Nat.lcm 1365 910 = 2730) ∧ (Nat.gcd 1365 910 = 455) := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_1365_910_l1429_142926


namespace NUMINAMATH_CALUDE_g_range_l1429_142998

noncomputable def g (x : ℝ) : ℝ :=
  (Real.sin x ^ 3 + 4 * Real.sin x ^ 2 - 3 * Real.sin x + 3 * Real.cos x ^ 2 - 9) / (Real.sin x - 1)

theorem g_range :
  Set.range (fun x : ℝ => g x) = Set.Icc 5 9 \ {9} :=
by
  sorry

end NUMINAMATH_CALUDE_g_range_l1429_142998


namespace NUMINAMATH_CALUDE_pentagon_rectangle_angle_sum_l1429_142996

/-- The sum of an interior angle of a regular pentagon and an interior angle of a rectangle is 198°. -/
theorem pentagon_rectangle_angle_sum : 
  let pentagon_angle : ℝ := 180 * (5 - 2) / 5
  let rectangle_angle : ℝ := 90
  pentagon_angle + rectangle_angle = 198 := by sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_angle_sum_l1429_142996


namespace NUMINAMATH_CALUDE_equal_positive_numbers_l1429_142965

theorem equal_positive_numbers (a b c d : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
  (h5 : a^4 + b^4 + c^4 + d^4 = 4*a*b*c*d) : 
  a = b ∧ b = c ∧ c = d := by
sorry

end NUMINAMATH_CALUDE_equal_positive_numbers_l1429_142965


namespace NUMINAMATH_CALUDE_boric_acid_mixture_volume_l1429_142942

theorem boric_acid_mixture_volume 
  (volume_1_percent : ℝ) 
  (volume_5_percent : ℝ) 
  (h1 : volume_1_percent = 15) 
  (h2 : volume_5_percent = 15) : 
  volume_1_percent + volume_5_percent = 30 := by
  sorry

end NUMINAMATH_CALUDE_boric_acid_mixture_volume_l1429_142942


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l1429_142955

theorem imaginary_part_of_complex_division (z₁ z₂ : ℂ) :
  z₁ = 2 - I → z₂ = 1 - 3*I → Complex.im (z₂ / z₁) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l1429_142955


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1429_142964

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1429_142964


namespace NUMINAMATH_CALUDE_mean_problem_l1429_142979

theorem mean_problem (x : ℝ) : 
  (48 + 62 + 98 + 124 + x) / 5 = 78 → 
  (28 + x + 42 + 78 + 104) / 5 = 62 := by
sorry

end NUMINAMATH_CALUDE_mean_problem_l1429_142979


namespace NUMINAMATH_CALUDE_min_sum_m_n_l1429_142978

theorem min_sum_m_n (m n : ℕ+) (h : 135 * m = n^3) : 
  ∀ (k l : ℕ+), 135 * k = l^3 → m + n ≤ k + l :=
by sorry

end NUMINAMATH_CALUDE_min_sum_m_n_l1429_142978


namespace NUMINAMATH_CALUDE_line_intersection_difference_l1429_142920

/-- Given a line y = 2x - 4 intersecting the x-axis at point A(m, 0) and the y-axis at point B(0, n), prove that m - n = 6 -/
theorem line_intersection_difference (m n : ℝ) : 
  (∀ x y, y = 2 * x - 4) →  -- Line equation
  0 = 2 * m - 4 →           -- A(m, 0) satisfies the line equation
  n = -4 →                  -- B(0, n) satisfies the line equation
  m - n = 6 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_difference_l1429_142920


namespace NUMINAMATH_CALUDE_quadratic_root_range_l1429_142999

theorem quadratic_root_range (a : ℝ) (α β : ℝ) : 
  (∃ x, x^2 - 2*a*x + a + 2 = 0) ∧ 
  (α^2 - 2*a*α + a + 2 = 0) ∧ 
  (β^2 - 2*a*β + a + 2 = 0) ∧ 
  (1 < α) ∧ (α < 2) ∧ (2 < β) ∧ (β < 3) →
  (2 < a) ∧ (a < 11/5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l1429_142999


namespace NUMINAMATH_CALUDE_parallel_line_equation_l1429_142928

/-- Given a line L passing through the point (1, 0) and parallel to the line x - 2y - 2 = 0,
    prove that the equation of L is x - 2y - 1 = 0 -/
theorem parallel_line_equation :
  ∀ (L : Set (ℝ × ℝ)),
  (∀ p ∈ L, ∃ x y : ℝ, p = (x, y) ∧ x - 2*y - 1 = 0) →
  (1, 0) ∈ L →
  (∀ p q : ℝ × ℝ, p ∈ L → q ∈ L → p.1 - q.1 = 2*(p.2 - q.2)) →
  ∀ p ∈ L, ∃ x y : ℝ, p = (x, y) ∧ x - 2*y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l1429_142928


namespace NUMINAMATH_CALUDE_household_survey_l1429_142941

/-- Households survey problem -/
theorem household_survey (neither : ℕ) (only_w : ℕ) (both : ℕ) 
  (h1 : neither = 80)
  (h2 : only_w = 60)
  (h3 : both = 40) :
  neither + only_w + 3 * both + both = 300 := by
  sorry

end NUMINAMATH_CALUDE_household_survey_l1429_142941


namespace NUMINAMATH_CALUDE_expansion_properties_l1429_142922

/-- Given that for some natural number n, the expansion of (x^(1/6) + x^(-1/6))^n has
    binomial coefficients of the 2nd, 3rd, and 4th terms forming an arithmetic sequence,
    prove that n = 7 and there is no constant term in the expansion. -/
theorem expansion_properties (n : ℕ) 
  (h : 2 * (n.choose 2) = n.choose 1 + n.choose 3) : 
  (n = 7) ∧ (∀ k : ℕ, (7 : ℚ) - 2 * k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l1429_142922


namespace NUMINAMATH_CALUDE_wheel_probability_l1429_142968

theorem wheel_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 →
  p_B = 1/3 →
  p_C = 1/6 →
  p_A + p_B + p_C + p_D = 1 →
  p_D = 1/4 := by
sorry

end NUMINAMATH_CALUDE_wheel_probability_l1429_142968


namespace NUMINAMATH_CALUDE_combined_cost_theorem_l1429_142909

def wallet_cost : ℕ := 22
def purse_cost : ℕ := 4 * wallet_cost - 3

theorem combined_cost_theorem : wallet_cost + purse_cost = 107 := by
  sorry

end NUMINAMATH_CALUDE_combined_cost_theorem_l1429_142909


namespace NUMINAMATH_CALUDE_fraction_problem_l1429_142933

theorem fraction_problem (a b : ℚ) (h1 : a + b = 100) (h2 : b = 60) : 
  (3 / 10) * a = (1 / 5) * b := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1429_142933


namespace NUMINAMATH_CALUDE_chairs_per_trip_l1429_142976

theorem chairs_per_trip 
  (num_students : ℕ) 
  (trips_per_student : ℕ) 
  (total_chairs : ℕ) 
  (h1 : num_students = 5) 
  (h2 : trips_per_student = 10) 
  (h3 : total_chairs = 250) : 
  (total_chairs / (num_students * trips_per_student) : ℚ) = 5 := by
sorry

end NUMINAMATH_CALUDE_chairs_per_trip_l1429_142976


namespace NUMINAMATH_CALUDE_infinite_symmetric_subset_exists_l1429_142904

/-- A color type representing black and white --/
inductive Color
| Black
| White

/-- A point in the plane with integer coordinates --/
structure Point where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each point with integer coordinates --/
def Coloring := Point → Color

/-- A set of points is symmetric about a center point if for every point in the set,
    its reflection about the center is also in the set --/
def IsSymmetric (S : Set Point) (center : Point) : Prop :=
  ∀ p ∈ S, Point.mk (2 * center.x - p.x) (2 * center.y - p.y) ∈ S

/-- The main theorem stating the existence of an infinite symmetric subset --/
theorem infinite_symmetric_subset_exists (coloring : Coloring) :
  ∃ (S : Set Point) (c : Color) (center : Point),
    Set.Infinite S ∧ (∀ p ∈ S, coloring p = c) ∧ IsSymmetric S center :=
  sorry

end NUMINAMATH_CALUDE_infinite_symmetric_subset_exists_l1429_142904


namespace NUMINAMATH_CALUDE_ratio_proof_l1429_142946

theorem ratio_proof (a b : ℝ) (h : (a - 3*b) / (2*a - b) = 0.14285714285714285) : 
  a/b = 4 := by sorry

end NUMINAMATH_CALUDE_ratio_proof_l1429_142946


namespace NUMINAMATH_CALUDE_lizzys_final_money_l1429_142945

/-- Calculates the final amount of money Lizzy has after a series of transactions -/
def lizzys_money (
  mother_gave : ℕ)
  (father_gave : ℕ)
  (candy_cost : ℕ)
  (uncle_gave : ℕ)
  (toy_price : ℕ)
  (discount_percent : ℕ)
  (change_dollars : ℕ)
  (change_cents : ℕ) : ℕ :=
  let initial := mother_gave + father_gave
  let after_candy := initial - candy_cost + uncle_gave
  let discounted_price := toy_price - (toy_price * discount_percent / 100)
  let after_toy := after_candy - discounted_price
  let final := after_toy + change_dollars * 100 + change_cents
  final

theorem lizzys_final_money :
  lizzys_money 80 40 50 70 90 20 1 10 = 178 := by
  sorry

end NUMINAMATH_CALUDE_lizzys_final_money_l1429_142945


namespace NUMINAMATH_CALUDE_charity_event_arrangements_l1429_142906

theorem charity_event_arrangements (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 5 → k = 4 → m = 3 →
  (Nat.choose n 2) * (Nat.choose (n - 2) 1) * (Nat.choose (n - 3) 1) = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_charity_event_arrangements_l1429_142906


namespace NUMINAMATH_CALUDE_fraction_addition_l1429_142940

theorem fraction_addition (d : ℝ) : (5 + 2 * d) / 8 + 3 = (29 + 2 * d) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1429_142940


namespace NUMINAMATH_CALUDE_algebraic_simplification_l1429_142918

theorem algebraic_simplification (x y : ℝ) :
  ((-3 * x * y^2)^3 * (-6 * x^2 * y)) / (9 * x^4 * y^5) = 18 * x * y^2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l1429_142918


namespace NUMINAMATH_CALUDE_exponent_simplification_l1429_142912

theorem exponent_simplification :
  (10 ^ 0.7) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ (-0.1)) * (10 ^ 0.5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l1429_142912


namespace NUMINAMATH_CALUDE_quadratic_function_uniqueness_l1429_142944

-- Define a quadratic function
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x

-- State the theorem
theorem quadratic_function_uniqueness (g : ℝ → ℝ) :
  (∃ a b : ℝ, g = QuadraticFunction a b) →  -- g is quadratic
  g 1 = 1 →                                 -- g(1) = 1
  g (-1) = 5 →                              -- g(-1) = 5
  g 0 = 0 →                                 -- g(0) = 0 (passes through origin)
  g = QuadraticFunction 3 (-2) :=           -- g(x) = 3x^2 - 2x
by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_uniqueness_l1429_142944


namespace NUMINAMATH_CALUDE_cos_thirteen_pi_fourths_l1429_142972

theorem cos_thirteen_pi_fourths : Real.cos (13 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_thirteen_pi_fourths_l1429_142972


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l1429_142962

theorem subtraction_of_fractions : (5 : ℚ) / 9 - (1 : ℚ) / 6 = (7 : ℚ) / 18 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l1429_142962


namespace NUMINAMATH_CALUDE_complex_imaginary_condition_l1429_142929

theorem complex_imaginary_condition (a : ℝ) : 
  let z : ℂ := (1 - 3*Complex.I) * (a - Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → a = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_imaginary_condition_l1429_142929


namespace NUMINAMATH_CALUDE_A_symmetry_l1429_142974

/-- A(n, k, r) is the number of integer tuples (x₁, x₂, ..., xₖ) satisfying:
    - x₁ ≥ x₂ ≥ ... ≥ xₖ ≥ 0
    - x₁ + x₂ + ... + xₖ = n
    - x₁ - xₖ ≤ r -/
def A (n k r : ℕ+) : ℕ :=
  sorry

/-- For all positive integers m, s, t, A(m, s, t) = A(m, t, s) -/
theorem A_symmetry (m s t : ℕ+) : A m s t = A m t s := by
  sorry

end NUMINAMATH_CALUDE_A_symmetry_l1429_142974


namespace NUMINAMATH_CALUDE_intersection_point_l1429_142902

/-- The first curve -/
def curve1 (x : ℝ) : ℝ := 4 * x^2 + 3 * x - 2

/-- The second curve -/
def curve2 (x : ℝ) : ℝ := 2 * x^3 + x^2 + 7

/-- Theorem stating that (-1, -1) is the only intersection point of the two curves -/
theorem intersection_point : 
  ∃! p : ℝ × ℝ, p.1 = -1 ∧ p.2 = -1 ∧ curve1 p.1 = curve2 p.1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l1429_142902


namespace NUMINAMATH_CALUDE_lansing_elementary_students_l1429_142905

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 25

/-- The number of students in each elementary school in Lansing -/
def students_per_school : ℕ := 247

/-- The total number of elementary students in Lansing -/
def total_students : ℕ := num_schools * students_per_school

theorem lansing_elementary_students :
  total_students = 6175 :=
by sorry

end NUMINAMATH_CALUDE_lansing_elementary_students_l1429_142905


namespace NUMINAMATH_CALUDE_ellipse_sum_bounds_l1429_142984

theorem ellipse_sum_bounds (x y : ℝ) : 
  x^2 / 2 + y^2 / 3 = 1 → 
  ∃ (S : ℝ), S = x + y ∧ -Real.sqrt 5 ≤ S ∧ S ≤ Real.sqrt 5 ∧
  (∃ (x₁ y₁ : ℝ), x₁^2 / 2 + y₁^2 / 3 = 1 ∧ x₁ + y₁ = -Real.sqrt 5) ∧
  (∃ (x₂ y₂ : ℝ), x₂^2 / 2 + y₂^2 / 3 = 1 ∧ x₂ + y₂ = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_sum_bounds_l1429_142984


namespace NUMINAMATH_CALUDE_initial_girls_count_l1429_142959

theorem initial_girls_count (initial_total : ℕ) (initial_girls : ℕ) : 
  initial_girls = 12 ∧ initial_total = 24 :=
  by
  have h1 : initial_girls = initial_total / 2 := by sorry
  have h2 : (initial_girls - 2) * 100 = 40 * (initial_total + 1) := by sorry
  have h3 : initial_girls * 100 = 45 * (initial_total - 1) := by sorry
  sorry

#check initial_girls_count

end NUMINAMATH_CALUDE_initial_girls_count_l1429_142959


namespace NUMINAMATH_CALUDE_sandbox_area_calculation_l1429_142935

/-- The area of a rectangular sandbox in square centimeters -/
def sandbox_area (length_meters : ℝ) (width_cm : ℝ) : ℝ :=
  (length_meters * 100) * width_cm

/-- Theorem: The area of a rectangular sandbox with length 3.12 meters and width 146 centimeters is 45552 square centimeters -/
theorem sandbox_area_calculation :
  sandbox_area 3.12 146 = 45552 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_area_calculation_l1429_142935


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_prism_volume_l1429_142990

theorem cube_surface_area_equal_prism_volume (a b c : ℝ) (h : a = 6 ∧ b = 3 ∧ c = 36) :
  let prism_volume := a * b * c
  let cube_edge := (prism_volume) ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge ^ 2
  cube_surface_area = 216 * 3 ^ (2/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_prism_volume_l1429_142990


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l1429_142969

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -1; 4, 2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![0, 6; -2, 3]
def C : Matrix (Fin 2) (Fin 2) ℤ := !![2, 15; -4, 30]

theorem matrix_multiplication_result : A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l1429_142969


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1429_142966

theorem opposite_of_2023 : 
  (∀ x : ℤ, x + 2023 = 0 → x = -2023) ∧ (-2023 + 2023 = 0) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1429_142966


namespace NUMINAMATH_CALUDE_corner_sum_possibilities_l1429_142932

/-- Represents the color of a cell on the board -/
inductive CellColor
| Gold
| Silver

/-- Represents the board configuration -/
structure Board :=
  (rows : Nat)
  (cols : Nat)
  (cellColor : Nat → Nat → CellColor)
  (vertexValue : Nat → Nat → Fin 2)

/-- Checks if a cell satisfies the sum condition based on its color -/
def validCell (b : Board) (row col : Nat) : Prop :=
  let sum := b.vertexValue row col + b.vertexValue row (col+1) +
             b.vertexValue (row+1) col + b.vertexValue (row+1) (col+1)
  match b.cellColor row col with
  | CellColor.Gold => sum % 2 = 0
  | CellColor.Silver => sum % 2 = 1

/-- Checks if the entire board configuration is valid -/
def validBoard (b : Board) : Prop :=
  b.rows = 2016 ∧ b.cols = 2017 ∧
  (∀ row col, row < b.rows → col < b.cols → validCell b row col) ∧
  (∀ row col, (row + col) % 2 = 0 → b.cellColor row col = CellColor.Gold) ∧
  (∀ row col, (row + col) % 2 = 1 → b.cellColor row col = CellColor.Silver)

/-- The sum of the four corner vertices of the board -/
def cornerSum (b : Board) : Nat :=
  b.vertexValue 0 0 + b.vertexValue 0 b.cols +
  b.vertexValue b.rows 0 + b.vertexValue b.rows b.cols

/-- Theorem stating the possible sums of the four corner vertices -/
theorem corner_sum_possibilities (b : Board) (h : validBoard b) :
  cornerSum b = 0 ∨ cornerSum b = 2 ∨ cornerSum b = 4 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_possibilities_l1429_142932


namespace NUMINAMATH_CALUDE_largest_power_is_396_l1429_142921

def pow (n : ℕ) : ℕ :=
  sorry

def largest_divisible_power (upper_bound : ℕ) : ℕ :=
  sorry

theorem largest_power_is_396 :
  largest_divisible_power 4000 = 396 :=
sorry

end NUMINAMATH_CALUDE_largest_power_is_396_l1429_142921


namespace NUMINAMATH_CALUDE_permutations_of_33377_l1429_142937

/-- The number of permutations of a multiset with 5 elements, where 3 elements are the same and 2 elements are the same -/
def permutations_of_multiset : ℕ :=
  Nat.factorial 5 / (Nat.factorial 3 * Nat.factorial 2)

theorem permutations_of_33377 : permutations_of_multiset = 10 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_33377_l1429_142937


namespace NUMINAMATH_CALUDE_no_solution_iff_k_eq_seven_l1429_142989

theorem no_solution_iff_k_eq_seven (k : ℝ) : 
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - k) / (x - 8)) ↔ k = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_eq_seven_l1429_142989


namespace NUMINAMATH_CALUDE_test_questions_count_l1429_142949

theorem test_questions_count : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (5 * n = 45) ∧ 
  (32 > 0.70 * 45) ∧ 
  (32 < 0.77 * 45) := by
  sorry

end NUMINAMATH_CALUDE_test_questions_count_l1429_142949


namespace NUMINAMATH_CALUDE_largest_integer_square_sum_l1429_142900

theorem largest_integer_square_sum : ∃ (x y z : ℤ),
  6^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10 ∧
  ∀ (n : ℤ), n > 6 → ¬∃ (x y z : ℤ),
    n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_square_sum_l1429_142900


namespace NUMINAMATH_CALUDE_double_thrice_one_is_eight_l1429_142934

def double (n : ℕ) : ℕ := 2 * n

def iterate_double (n : ℕ) (times : ℕ) : ℕ :=
  match times with
  | 0 => n
  | k + 1 => iterate_double (double n) k

theorem double_thrice_one_is_eight :
  iterate_double 1 3 = 8 := by sorry

end NUMINAMATH_CALUDE_double_thrice_one_is_eight_l1429_142934


namespace NUMINAMATH_CALUDE_inscribed_right_isosceles_hypotenuse_l1429_142977

/-- Represents a triangle with a given base and height -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Represents a right isosceles triangle inscribed in another triangle -/
structure InscribedRightIsoscelesTriangle where
  outer : Triangle
  hypotenuse : ℝ

/-- The hypotenuse of an inscribed right isosceles triangle in a 30x10 triangle is 12 -/
theorem inscribed_right_isosceles_hypotenuse 
  (t : Triangle) 
  (i : InscribedRightIsoscelesTriangle) 
  (h1 : t.base = 30) 
  (h2 : t.height = 10) 
  (h3 : i.outer = t) : 
  i.hypotenuse = 12 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_right_isosceles_hypotenuse_l1429_142977


namespace NUMINAMATH_CALUDE_tan_forty_five_degrees_equals_one_l1429_142958

theorem tan_forty_five_degrees_equals_one : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_forty_five_degrees_equals_one_l1429_142958


namespace NUMINAMATH_CALUDE_two_props_true_l1429_142994

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x = 5 → x^2 - 8*x + 15 = 0

-- Define the converse proposition
def converse_prop (x : ℝ) : Prop := x^2 - 8*x + 15 = 0 → x = 5

-- Define the inverse proposition
def inverse_prop (x : ℝ) : Prop := x ≠ 5 → x^2 - 8*x + 15 ≠ 0

-- Define the contrapositive proposition
def contrapositive_prop (x : ℝ) : Prop := x^2 - 8*x + 15 ≠ 0 → x ≠ 5

-- Theorem stating that exactly two propositions (including the original) are true
theorem two_props_true :
  (∀ x, original_prop x) ∧
  (¬ ∀ x, converse_prop x) ∧
  (¬ ∀ x, inverse_prop x) ∧
  (∀ x, contrapositive_prop x) :=
sorry

end NUMINAMATH_CALUDE_two_props_true_l1429_142994


namespace NUMINAMATH_CALUDE_shopkeeper_decks_l1429_142993

-- Define the number of red cards the shopkeeper has
def total_red_cards : ℕ := 208

-- Define the number of cards in a standard deck
def cards_per_deck : ℕ := 52

-- Theorem stating the number of decks the shopkeeper has
theorem shopkeeper_decks : 
  total_red_cards / cards_per_deck = 4 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_decks_l1429_142993


namespace NUMINAMATH_CALUDE_min_value_theorem_l1429_142911

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 2) :
  (1 / x + 1 / (3 * y)) ≥ 4 ∧ 
  ∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ 
    Real.log 2 * x₀ + Real.log 8 * y₀ = Real.log 2 ∧
    1 / x₀ + 1 / (3 * y₀) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1429_142911


namespace NUMINAMATH_CALUDE_number_line_movement_1_number_line_movement_2_number_line_movement_general_absolute_value_equality_l1429_142910

-- Problem 1
theorem number_line_movement_1 (A B : ℝ) :
  A = -2 → B = A + 5 → B = 3 ∧ |B - A| = 5 := by sorry

-- Problem 2
theorem number_line_movement_2 (A B : ℝ) :
  A = 5 → B = A - 4 + 7 → B = 8 ∧ |B - A| = 3 := by sorry

-- Problem 3
theorem number_line_movement_general (a b c A B : ℝ) :
  A = a → B = A + b - c → B = a + b - c ∧ |B - A| = |b - c| := by sorry

-- Problem 4
theorem absolute_value_equality (x : ℝ) :
  |x + 1| = |x - 2| ↔ x = 1/2 := by sorry

end NUMINAMATH_CALUDE_number_line_movement_1_number_line_movement_2_number_line_movement_general_absolute_value_equality_l1429_142910


namespace NUMINAMATH_CALUDE_calculate_expression_l1429_142908

theorem calculate_expression : 6 * (1/3 - 1/2) - 3^2 / (-12) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1429_142908


namespace NUMINAMATH_CALUDE_simplify_expression_l1429_142985

theorem simplify_expression (x : ℚ) : 
  ((3 * x + 6) - 5 * x) / 3 = -2 * x / 3 + 2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1429_142985


namespace NUMINAMATH_CALUDE_red_tile_cost_courtyard_red_tile_cost_l1429_142981

/-- Calculates the cost of each red tile in a courtyard tiling project. -/
theorem red_tile_cost (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (tiles_per_sqft : ℝ) (green_tile_percentage : ℝ) (green_tile_cost : ℝ) 
  (total_cost : ℝ) : ℝ :=
  let total_area := courtyard_length * courtyard_width
  let total_tiles := total_area * tiles_per_sqft
  let green_tiles := green_tile_percentage * total_tiles
  let red_tiles := total_tiles - green_tiles
  let green_cost := green_tiles * green_tile_cost
  let red_cost := total_cost - green_cost
  red_cost / red_tiles

/-- The cost of each red tile in the given courtyard tiling project is $1.50. -/
theorem courtyard_red_tile_cost : 
  red_tile_cost 25 10 4 0.4 3 2100 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_red_tile_cost_courtyard_red_tile_cost_l1429_142981


namespace NUMINAMATH_CALUDE_fabric_length_l1429_142960

/-- Given a rectangular piece of fabric with width 3 cm and area 24 cm², prove its length is 8 cm. -/
theorem fabric_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 3 → area = 24 → area = length * width → length = 8 := by
sorry

end NUMINAMATH_CALUDE_fabric_length_l1429_142960


namespace NUMINAMATH_CALUDE_family_ages_solution_l1429_142971

/-- Represents the ages of a family members -/
structure FamilyAges where
  son : ℕ
  daughter : ℕ
  man : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.man = ages.son + 20 ∧
  ages.man = ages.daughter + 15 ∧
  ages.man + 2 = 2 * (ages.son + 2) ∧
  ages.man + 2 = 3 * (ages.daughter + 2)

/-- Theorem stating that the given ages satisfy the problem conditions -/
theorem family_ages_solution :
  ∃ (ages : FamilyAges), satisfiesConditions ages ∧
    ages.son = 18 ∧ ages.daughter = 23 ∧ ages.man = 38 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_solution_l1429_142971


namespace NUMINAMATH_CALUDE_doubling_base_theorem_l1429_142975

theorem doubling_base_theorem (a b x : ℝ) (h1 : b ≠ 0) :
  (2 * a) ^ b = a ^ b * x ^ b → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_doubling_base_theorem_l1429_142975


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l1429_142914

theorem modular_congruence_solution : ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l1429_142914


namespace NUMINAMATH_CALUDE_milly_extra_balloons_l1429_142956

theorem milly_extra_balloons (total_packs : ℕ) (balloons_per_pack : ℕ) (floretta_balloons : ℕ) : 
  total_packs = 5 →
  balloons_per_pack = 6 →
  floretta_balloons = 8 →
  (total_packs * balloons_per_pack) / 2 - floretta_balloons = 7 := by
  sorry

end NUMINAMATH_CALUDE_milly_extra_balloons_l1429_142956


namespace NUMINAMATH_CALUDE_simplification_proof_l1429_142948

theorem simplification_proof (a : ℝ) (h : a ≠ 1 ∧ a ≠ -1) : 
  (a - 1) / (a^2 - 1) + 1 / (a + 1) = 2 / (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplification_proof_l1429_142948


namespace NUMINAMATH_CALUDE_slope_range_for_inclination_angle_l1429_142987

theorem slope_range_for_inclination_angle (α : Real) :
  π / 4 ≤ α ∧ α ≤ 3 * π / 4 →
  ∃ k : Real, (k < -1 ∨ k = -1 ∨ k = 1 ∨ k > 1) ∧ k = Real.tan α :=
sorry

end NUMINAMATH_CALUDE_slope_range_for_inclination_angle_l1429_142987


namespace NUMINAMATH_CALUDE_sum_remainder_nine_specific_sum_remainder_l1429_142939

theorem sum_remainder_nine (n : ℕ) : (n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) % 9 = ((n % 9 + (n + 1) % 9 + (n + 2) % 9 + (n + 3) % 9 + (n + 4) % 9) % 9) := by
  sorry

theorem specific_sum_remainder :
  (9150 + 9151 + 9152 + 9153 + 9154) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_nine_specific_sum_remainder_l1429_142939


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1429_142901

theorem polynomial_factorization (x : ℝ) : 
  9 * (x + 6) * (x + 12) * (x + 5) * (x + 15) - 8 * x^2 = 
  (3 * x^2 + 52 * x + 210) * (3 * x^2 + 56 * x + 222) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1429_142901


namespace NUMINAMATH_CALUDE_remainder_problem_l1429_142997

theorem remainder_problem (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 34 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1429_142997


namespace NUMINAMATH_CALUDE_valentines_day_cards_l1429_142992

theorem valentines_day_cards (boys girls : ℕ) : 
  boys * girls = boys + girls + 18 → boys * girls = 40 := by
  sorry

end NUMINAMATH_CALUDE_valentines_day_cards_l1429_142992


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1429_142917

theorem quadratic_two_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + m = 0 ∧ y^2 + 2*y + m = 0) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1429_142917


namespace NUMINAMATH_CALUDE_trigonometric_equation_system_solution_l1429_142923

theorem trigonometric_equation_system_solution :
  ∃ (x y : ℝ),
    3 * Real.cos x + 4 * Real.sin x = -1.4 ∧
    13 * Real.cos x - 41 * Real.cos y = -45 ∧
    13 * Real.sin x + 41 * Real.sin y = 3 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_system_solution_l1429_142923


namespace NUMINAMATH_CALUDE_room_area_difference_l1429_142973

-- Define the dimensions of the rooms
def largest_room_width : ℝ := 45
def largest_room_length : ℝ := 30
def smallest_room_width : ℝ := 15
def smallest_room_length : ℝ := 8

-- Define the area calculation function
def area (width : ℝ) (length : ℝ) : ℝ := width * length

-- Theorem statement
theorem room_area_difference :
  area largest_room_width largest_room_length - area smallest_room_width smallest_room_length = 1230 := by
  sorry

end NUMINAMATH_CALUDE_room_area_difference_l1429_142973


namespace NUMINAMATH_CALUDE_min_correct_answers_for_target_score_l1429_142982

/-- Represents a quiz with specified scoring rules -/
structure Quiz where
  total_questions : ℕ
  correct_points : ℕ
  incorrect_deduction : ℕ

/-- Calculates the score for a given number of correct answers -/
def score (q : Quiz) (correct_answers : ℕ) : ℤ :=
  (q.correct_points * correct_answers : ℤ) - 
  (q.incorrect_deduction * (q.total_questions - correct_answers) : ℤ)

/-- Theorem stating the minimum number of correct answers needed to achieve a target score -/
theorem min_correct_answers_for_target_score 
  (q : Quiz) 
  (target_score : ℤ) 
  (correct_answers : ℕ) :
  q.total_questions = 20 →
  q.correct_points = 5 →
  q.incorrect_deduction = 1 →
  target_score = 88 →
  score q correct_answers ≥ target_score →
  (5 : ℤ) * correct_answers - (20 - correct_answers) ≥ 88 := by
  sorry

#check min_correct_answers_for_target_score

end NUMINAMATH_CALUDE_min_correct_answers_for_target_score_l1429_142982


namespace NUMINAMATH_CALUDE_simon_beach_treasures_l1429_142943

/-- Represents the number of treasures Simon collected on the beach. -/
def beach_treasures (sand_dollars : ℕ) (glass_multiplier : ℕ) (shell_multiplier : ℕ) : ℕ :=
  let glass := sand_dollars * glass_multiplier
  let shells := glass * shell_multiplier
  sand_dollars + glass + shells

/-- Proves that Simon collected 190 treasures on the beach. -/
theorem simon_beach_treasures :
  beach_treasures 10 3 5 = 190 := by
  sorry

end NUMINAMATH_CALUDE_simon_beach_treasures_l1429_142943


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1429_142954

theorem least_subtraction_for_divisibility : ∃ (n : ℕ), n = 8 ∧
  20 ∣ (50248 - n) ∧
  ∀ (m : ℕ), m < n → ¬(20 ∣ (50248 - m)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1429_142954


namespace NUMINAMATH_CALUDE_stock_price_fluctuation_l1429_142947

theorem stock_price_fluctuation (P : ℝ) (x : ℝ) 
  (h1 : P > 0)
  (h2 : 1.30 * (1 - x / 100) * 1.20 * P = 1.17 * P) : 
  x = 25 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_fluctuation_l1429_142947


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1429_142915

/-- An arithmetic sequence with sum Sn for first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function

/-- The common difference of the arithmetic sequence is -2 -/
theorem arithmetic_sequence_difference (seq : ArithmeticSequence) 
  (h1 : seq.S 3 = 6)
  (h2 : seq.a 3 = 0) : 
  seq.d = -2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1429_142915


namespace NUMINAMATH_CALUDE_solve_system_l1429_142963

theorem solve_system (x y : ℝ) (h1 : 3 * x - 482 = 2 * y) (h2 : 7 * x + 517 = 5 * y) :
  x = 3444 ∧ y = 4925 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1429_142963
