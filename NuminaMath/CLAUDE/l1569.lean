import Mathlib

namespace bus_passenger_count_l1569_156930

/-- The number of passengers who got on at the first stop -/
def passengers_first_stop : ℕ := 16

theorem bus_passenger_count : passengers_first_stop = 16 :=
  let initial_passengers : ℕ := 50
  let final_passengers : ℕ := 49
  let passengers_off : ℕ := 22
  let passengers_on_other_stops : ℕ := 5
  have h : initial_passengers + passengers_first_stop - (passengers_off - passengers_on_other_stops) = final_passengers :=
    by sorry
  by sorry

end bus_passenger_count_l1569_156930


namespace convergence_of_difference_series_l1569_156951

/-- Given two real sequences (a_i) and (b_i) where the series of their squares converge,
    prove that the series of |a_i - b_i|^p converges for all p ≥ 2. -/
theorem convergence_of_difference_series
  (a b : ℕ → ℝ)
  (ha : Summable (λ i => (a i)^2))
  (hb : Summable (λ i => (b i)^2))
  (p : ℝ)
  (hp : p ≥ 2) :
  Summable (λ i => |a i - b i|^p) :=
sorry

end convergence_of_difference_series_l1569_156951


namespace min_draws_for_even_product_l1569_156969

theorem min_draws_for_even_product (cards : Finset ℕ) : 
  cards = Finset.range 14 →
  ∃ (n : ℕ), n = 8 ∧ 
    ∀ (subset : Finset ℕ), subset ⊆ cards → subset.card < n → 
      ∃ (x : ℕ), x ∈ subset ∧ Even x :=
by sorry

end min_draws_for_even_product_l1569_156969


namespace problem_one_problem_two_problem_three_problem_four_l1569_156961

-- Problem 1
theorem problem_one : 6 + (-8) - (-5) = 3 := by sorry

-- Problem 2
theorem problem_two : 5 + 3/5 + (-5 - 2/3) + 4 + 2/5 + (-1/3) = 4 := by sorry

-- Problem 3
theorem problem_three : (-1/2 + 1/6 - 1/4) * 12 = -7 := by sorry

-- Problem 4
theorem problem_four : -1^2022 + 27 * (-1/3)^2 - |(-5)| = -3 := by sorry

end problem_one_problem_two_problem_three_problem_four_l1569_156961


namespace proportion_property_l1569_156906

theorem proportion_property (a b c d : ℝ) (h : a / b = c / d) : b * c - a * d = 0 := by
  sorry

end proportion_property_l1569_156906


namespace tv_sales_after_three_years_l1569_156970

def initial_sales : ℕ := 327
def yearly_increase : ℕ := 50
def years : ℕ := 3

theorem tv_sales_after_three_years :
  initial_sales + years * yearly_increase = 477 :=
by sorry

end tv_sales_after_three_years_l1569_156970


namespace fraction_simplification_positive_integer_solutions_l1569_156934

-- Problem 1
theorem fraction_simplification (x : ℝ) (h : x ≠ 1) :
  (x^2 + x) / (x^2 - 2*x + 1) / ((2 / (x - 1)) - (1 / x)) = x^2 / (x - 1) := by
  sorry

-- Problem 2
def inequality_system (x : ℝ) : Prop :=
  (2*x + 1) / 3 - (5*x - 1) / 2 < 1 ∧ 5*x - 1 < 3*(x + 2)

theorem positive_integer_solutions :
  {x : ℕ | inequality_system x} = {1, 2, 3} := by
  sorry

end fraction_simplification_positive_integer_solutions_l1569_156934


namespace basketball_tryouts_l1569_156981

/-- Given the number of girls and boys trying out for a basketball team,
    and the number of students called back, calculate the number of
    students who didn't make the cut. -/
theorem basketball_tryouts (girls boys called_back : ℕ) : 
  girls = 39 → boys = 4 → called_back = 26 → 
  girls + boys - called_back = 17 := by
  sorry

end basketball_tryouts_l1569_156981


namespace line_through_circle_center_perpendicular_to_given_line_l1569_156977

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0

-- Define the given line equation
def given_line_equation (x y : ℝ) : Prop := x + y = 0

-- Define the resulting line equation
def result_line_equation (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the center of a circle
def circle_center (c : ℝ × ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y ↔ (x - c.1)^2 + (y - c.2)^2 = 1

-- Define perpendicularity of two lines
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

theorem line_through_circle_center_perpendicular_to_given_line :
  ∃ c : ℝ × ℝ,
    circle_center c circle_equation ∧
    (∃ m₁ m₂ : ℝ,
      (∀ x y, given_line_equation x y ↔ y = m₁ * x) ∧
      (∀ x y, result_line_equation x y ↔ y = m₂ * x + c.2) ∧
      perpendicular m₁ m₂) :=
sorry

end line_through_circle_center_perpendicular_to_given_line_l1569_156977


namespace alpha_necessary_not_sufficient_for_beta_l1569_156972

theorem alpha_necessary_not_sufficient_for_beta :
  (∀ a b : ℝ, b ≠ 0 → (a / b ≥ 1 → b * (b - a) ≤ 0)) ∧
  (∃ a b : ℝ, b ≠ 0 ∧ b * (b - a) ≤ 0 ∧ a / b < 1) := by
sorry

end alpha_necessary_not_sufficient_for_beta_l1569_156972


namespace binomial_square_coeff_l1569_156983

/-- If ax^2 + 8x + 16 is the square of a binomial, then a = 1 -/
theorem binomial_square_coeff (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 8 * x + 16 = (r * x + s)^2) → a = 1 := by
  sorry

end binomial_square_coeff_l1569_156983


namespace min_value_of_function_l1569_156932

theorem min_value_of_function (x : ℝ) (h : x > 1) : 
  x + 1 / (x - 1) ≥ 3 ∧ ∃ y > 1, y + 1 / (y - 1) = 3 := by
  sorry

end min_value_of_function_l1569_156932


namespace negation_of_universal_statement_l1569_156995

theorem negation_of_universal_statement :
  (¬∀ x : ℝ, x > 2 → x^2 - 2*x > 0) ↔ (∃ x : ℝ, x > 2 ∧ x^2 - 2*x ≤ 0) := by
  sorry

end negation_of_universal_statement_l1569_156995


namespace no_bounded_figure_with_parallel_axes_exists_unbounded_figure_with_parallel_axes_l1569_156936

-- Define a type for figures on a plane
structure PlaneFigure where
  -- Add necessary fields here
  isBounded : Bool
  hasParallelAxes : Bool

-- Define a predicate for having two parallel, non-coincident symmetry axes
def hasParallelSymmetryAxes (f : PlaneFigure) : Prop :=
  f.hasParallelAxes

theorem no_bounded_figure_with_parallel_axes :
  ¬ ∃ (f : PlaneFigure), f.isBounded ∧ hasParallelSymmetryAxes f := by
  sorry

theorem exists_unbounded_figure_with_parallel_axes :
  ∃ (f : PlaneFigure), ¬f.isBounded ∧ hasParallelSymmetryAxes f := by
  sorry

end no_bounded_figure_with_parallel_axes_exists_unbounded_figure_with_parallel_axes_l1569_156936


namespace problems_per_worksheet_l1569_156963

/-- Given a set of worksheets with some graded and some problems left to grade,
    calculate the number of problems per worksheet. -/
theorem problems_per_worksheet
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (remaining_problems : ℕ)
  (h1 : total_worksheets = 16)
  (h2 : graded_worksheets = 8)
  (h3 : remaining_problems = 32)
  : (remaining_problems / (total_worksheets - graded_worksheets) : ℚ) = 4 := by
  sorry

end problems_per_worksheet_l1569_156963


namespace num_triangles_in_polygon_l1569_156913

/-- 
A polygon with n sides, where n is at least 3.
-/
structure Polygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- 
The number of triangles formed by non-intersecting diagonals in an n-gon.
-/
def num_triangles (p : Polygon) : ℕ := p.n - 2

/-- 
Theorem: The number of triangles formed by non-intersecting diagonals 
in an n-gon is equal to n-2.
-/
theorem num_triangles_in_polygon (p : Polygon) : 
  num_triangles p = p.n - 2 := by
  sorry

end num_triangles_in_polygon_l1569_156913


namespace parallel_lines_m_value_l1569_156950

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (a1 b1 a2 b2 : ℝ) : Prop := a1 / b1 = a2 / b2

/-- Definition of the first line l1 -/
def l1 (m : ℝ) (x y : ℝ) : Prop := (3 + m) * x + 4 * y = 5 - 3 * m

/-- Definition of the second line l2 -/
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (5 + m) * y = 8

/-- Additional condition for m -/
def additional_condition (m : ℝ) : Prop := (3 + m) / 2 ≠ (5 - 3 * m) / 8

theorem parallel_lines_m_value :
  ∃ (m : ℝ), parallel_lines (3 + m) 4 2 (5 + m) ∧ 
             additional_condition m ∧
             m = -7 :=
sorry

end parallel_lines_m_value_l1569_156950


namespace cars_to_double_earnings_l1569_156910

def base_salary : ℕ := 1000
def commission_per_car : ℕ := 200
def january_earnings : ℕ := 1800

theorem cars_to_double_earnings : 
  ∃ (february_cars : ℕ), 
    base_salary + february_cars * commission_per_car = 2 * january_earnings ∧ 
    february_cars = 13 :=
by sorry

end cars_to_double_earnings_l1569_156910


namespace weekly_coffee_cost_household_weekly_coffee_cost_l1569_156991

/-- Calculates the weekly cost of coffee for a household -/
theorem weekly_coffee_cost 
  (people : ℕ) 
  (cups_per_person : ℕ) 
  (ounces_per_cup : ℚ) 
  (cost_per_ounce : ℚ) : ℚ :=
  let daily_cups := people * cups_per_person
  let daily_ounces := daily_cups * ounces_per_cup
  let weekly_ounces := daily_ounces * 7
  weekly_ounces * cost_per_ounce

/-- Proves that the weekly coffee cost for the given household is $35 -/
theorem household_weekly_coffee_cost : 
  weekly_coffee_cost 4 2 (1/2) (5/4) = 35 := by
  sorry

end weekly_coffee_cost_household_weekly_coffee_cost_l1569_156991


namespace cubic_equation_solutions_l1569_156903

theorem cubic_equation_solutions : 
  let z₁ : ℂ := -3
  let z₂ : ℂ := (3/2) + (3*I*Real.sqrt 3)/2
  let z₃ : ℂ := (3/2) - (3*I*Real.sqrt 3)/2
  (z₁^3 = -27 ∧ z₂^3 = -27 ∧ z₃^3 = -27) ∧
  (∀ z : ℂ, z^3 = -27 → z = z₁ ∨ z = z₂ ∨ z = z₃) := by sorry

end cubic_equation_solutions_l1569_156903


namespace linda_borrowed_amount_l1569_156964

-- Define the pay pattern
def payPattern : List Nat := [2, 4, 6, 8, 10]

-- Function to calculate pay for a given number of hours
def calculatePay (hours : Nat) : Nat :=
  let fullCycles := hours / payPattern.length
  let remainingHours := hours % payPattern.length
  fullCycles * payPattern.sum + (payPattern.take remainingHours).sum

-- Theorem statement
theorem linda_borrowed_amount :
  calculatePay 22 = 126 := by
  sorry

end linda_borrowed_amount_l1569_156964


namespace ivan_remaining_money_l1569_156925

def initial_amount : ℚ := 10
def cupcake_fraction : ℚ := 1/5
def milkshake_cost : ℚ := 5

theorem ivan_remaining_money :
  let cupcake_cost : ℚ := initial_amount * cupcake_fraction
  let remaining_after_cupcakes : ℚ := initial_amount - cupcake_cost
  let final_remaining : ℚ := remaining_after_cupcakes - milkshake_cost
  final_remaining = 3 := by sorry

end ivan_remaining_money_l1569_156925


namespace remainder_nine_eight_mod_five_l1569_156984

theorem remainder_nine_eight_mod_five : 9^8 % 5 = 1 := by
  sorry

end remainder_nine_eight_mod_five_l1569_156984


namespace fractional_equation_m_range_l1569_156955

theorem fractional_equation_m_range :
  ∀ m x : ℝ,
  (x / (x - 3) = 2 + m / (x - 3)) →
  (x > 0) →
  (m < 6 ∧ m ≠ 3) :=
by sorry

end fractional_equation_m_range_l1569_156955


namespace range_of_a_l1569_156926

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 1}
def B : Set ℝ := {x | |x - 1| > 2}

-- Theorem statement
theorem range_of_a (a : ℝ) : (A a ∩ B = A a) ↔ (a ≤ -1 ∨ a ≥ 3) := by
  sorry

end range_of_a_l1569_156926


namespace distance_to_directrix_l1569_156974

/-- A parabola C with equation y² = 2px and a point A(1, √5) lying on it -/
structure Parabola where
  p : ℝ
  A : ℝ × ℝ
  h1 : A.1 = 1
  h2 : A.2 = Real.sqrt 5
  h3 : A.2^2 = 2 * p * A.1

/-- The distance from point A to the directrix of parabola C is 9/4 -/
theorem distance_to_directrix (C : Parabola) : 
  C.A.1 + C.p / 2 = 9 / 4 := by
  sorry

end distance_to_directrix_l1569_156974


namespace triangle_formation_theorem_l1569_156921

/-- Given three positive real numbers a, b, and c, they can form a triangle
    if and only if the sum of any two is greater than the third. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem states that among the given combinations, only (4, 5, 6)
    satisfies the triangle inequality and thus can form a triangle. -/
theorem triangle_formation_theorem :
  ¬ can_form_triangle 2 3 6 ∧
  ¬ can_form_triangle 3 3 6 ∧
  can_form_triangle 4 5 6 ∧
  ¬ can_form_triangle 4 10 6 :=
sorry

end triangle_formation_theorem_l1569_156921


namespace samuel_homework_time_l1569_156935

theorem samuel_homework_time (sarah_time : Real) (time_difference : Nat) : 
  sarah_time = 1.3 → time_difference = 48 → 
  ⌊sarah_time * 60 - time_difference⌋ = 30 := by
  sorry

end samuel_homework_time_l1569_156935


namespace isosceles_triangle_apex_angle_l1569_156902

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base_angle : ℝ
  apex_angle : ℝ
  is_isosceles : base_angle ≥ 0 ∧ apex_angle ≥ 0
  angle_sum : 2 * base_angle + apex_angle = 180

-- Theorem statement
theorem isosceles_triangle_apex_angle 
  (triangle : IsoscelesTriangle) 
  (h : triangle.base_angle = 42) : 
  triangle.apex_angle = 96 := by
sorry


end isosceles_triangle_apex_angle_l1569_156902


namespace probability_of_integer_occurrence_l1569_156994

theorem probability_of_integer_occurrence (a b : ℤ) (h : a ≤ b) :
  let range := b - a + 1
  (∀ k : ℤ, a ≤ k ∧ k ≤ b → (1 : ℚ) / range = (1 : ℚ) / range) :=
by sorry

end probability_of_integer_occurrence_l1569_156994


namespace smallest_cookie_boxes_l1569_156907

theorem smallest_cookie_boxes : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (15 * m - 1) % 11 = 0 → n ≤ m) ∧ 
  (15 * n - 1) % 11 = 0 :=
by sorry

end smallest_cookie_boxes_l1569_156907


namespace rectangle_area_is_twelve_l1569_156966

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle defined by its four vertices -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The area of a rectangle -/
def rectangleArea (rect : Rectangle) : ℝ :=
  (rect.B.x - rect.A.x) * (rect.C.y - rect.B.y)

theorem rectangle_area_is_twelve :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨3, 0⟩
  let C : Point := ⟨3, 4⟩
  let D : Point := ⟨0, 4⟩
  let rect : Rectangle := ⟨A, B, C, D⟩
  rectangleArea rect = 12 := by
  sorry

end rectangle_area_is_twelve_l1569_156966


namespace remainder_problem_l1569_156953

theorem remainder_problem (d r : ℤ) : 
  d > 1 → 
  2024 % d = r → 
  3250 % d = r → 
  4330 % d = r → 
  d - r = 2 := by
sorry

end remainder_problem_l1569_156953


namespace sum_reciprocals_bound_l1569_156933

theorem sum_reciprocals_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 ∧ ∀ M : ℝ, ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ a' + b' = 2 ∧ 1/a' + 1/b' > M :=
by sorry

end sum_reciprocals_bound_l1569_156933


namespace marble_jar_ratio_l1569_156919

/-- Proves that the ratio of marbles in the second jar to the first jar is 2:1 --/
theorem marble_jar_ratio :
  ∀ (jar1 jar2 jar3 : ℕ),
  jar1 = 80 →
  jar3 = jar1 / 4 →
  jar1 + jar2 + jar3 = 260 →
  jar2 = 2 * jar1 :=
by
  sorry

end marble_jar_ratio_l1569_156919


namespace not_diff_of_squares_2022_l1569_156980

theorem not_diff_of_squares_2022 : ∀ a b : ℤ, a^2 - b^2 ≠ 2022 := by
  sorry

end not_diff_of_squares_2022_l1569_156980


namespace total_coins_proof_l1569_156946

theorem total_coins_proof (jayden_coins jasmine_coins : ℕ) 
  (h1 : jayden_coins = 300)
  (h2 : jasmine_coins = 335)
  (h3 : ∃ jason_coins : ℕ, jason_coins = jayden_coins + 60 ∧ jason_coins = jasmine_coins + 25) :
  ∃ total_coins : ℕ, total_coins = jayden_coins + jasmine_coins + (jayden_coins + 60) ∧ total_coins = 995 := by
  sorry

end total_coins_proof_l1569_156946


namespace percentage_passed_all_topics_percentage_passed_all_topics_proof_l1569_156986

/-- The percentage of students who passed in all topics in a practice paper -/
theorem percentage_passed_all_topics : ℝ :=
  let total_students : ℕ := 2500
  let passed_three_topics : ℕ := 500
  let percent_no_pass : ℝ := 10
  let percent_one_topic : ℝ := 20
  let percent_two_topics : ℝ := 25
  let percent_four_topics : ℝ := 24
  let percent_three_topics : ℝ := (passed_three_topics : ℝ) / (total_students : ℝ) * 100

  1 -- This is the percentage we need to prove

theorem percentage_passed_all_topics_proof : percentage_passed_all_topics = 1 := by
  sorry

end percentage_passed_all_topics_percentage_passed_all_topics_proof_l1569_156986


namespace S_subset_T_l1569_156944

-- Define set S
def S : Set ℕ := {x | ∃ n : ℕ, x = 3^n}

-- Define set T
def T : Set ℕ := {x | ∃ n : ℕ, x = 3*n}

-- Theorem stating S is a subset of T
theorem S_subset_T : S ⊆ T := by
  sorry

end S_subset_T_l1569_156944


namespace problem_solution_l1569_156985

-- Define the properties of functions f and g
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

def is_inverse_proportion (g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → g x = k / x

-- Define the conditions given in the problem
def problem_conditions (f g : ℝ → ℝ) : Prop :=
  is_direct_proportion f ∧ is_inverse_proportion g ∧ f 1 = 1 ∧ g 1 = 2

-- Define what it means for a function to be odd
def is_odd_function (h : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, h (-x) = -h x

-- Theorem statement
theorem problem_solution (f g : ℝ → ℝ) (h : problem_conditions f g) :
  (∀ x : ℝ, f x = x) ∧
  (∀ x : ℝ, x ≠ 0 → g x = 2 / x) ∧
  is_odd_function (λ x => f x + g x) :=
by sorry

end problem_solution_l1569_156985


namespace least_positive_integer_with_remainders_l1569_156911

theorem least_positive_integer_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  n % 10 = 9 ∧
  n % 11 = 10 ∧
  ∀ m : ℕ, m > 0 ∧
    m % 5 = 4 ∧
    m % 6 = 5 ∧
    m % 7 = 6 ∧
    m % 8 = 7 ∧
    m % 9 = 8 ∧
    m % 10 = 9 ∧
    m % 11 = 10 → n ≤ m :=
by
  sorry

end least_positive_integer_with_remainders_l1569_156911


namespace stating_prob_served_last_independent_of_position_prob_served_last_2014_l1569_156965

/-- 
Represents a round table with n people, where food is passed randomly.
n is the number of people at the table.
-/
structure RoundTable where
  n : ℕ
  hn : n > 1

/-- 
The probability of a specific person (other than the head) being served last.
table: The round table setup
person: The index of the person we're interested in (2 ≤ person ≤ n)
-/
def probabilityServedLast (table : RoundTable) (person : ℕ) : ℚ :=
  1 / (table.n - 1)

/-- 
Theorem stating that the probability of any specific person (other than the head) 
being served last is 1/(n-1), regardless of their position.
-/
theorem prob_served_last_independent_of_position (table : RoundTable) 
    (person : ℕ) (h : 2 ≤ person ∧ person ≤ table.n) : 
    probabilityServedLast table person = 1 / (table.n - 1) := by
  sorry

/-- 
The specific case for the problem with 2014 people and the person of interest
seated 2 seats away from the head.
-/
def table2014 : RoundTable := ⟨2014, by norm_num⟩

theorem prob_served_last_2014 : 
    probabilityServedLast table2014 2 = 1 / 2013 := by
  sorry

end stating_prob_served_last_independent_of_position_prob_served_last_2014_l1569_156965


namespace odd_sum_of_squares_implies_odd_sum_l1569_156957

theorem odd_sum_of_squares_implies_odd_sum (n m : ℤ) :
  Odd (n^2 + m^2) → Odd (n + m) := by
  sorry

end odd_sum_of_squares_implies_odd_sum_l1569_156957


namespace perfect_square_trinomial_l1569_156960

theorem perfect_square_trinomial (m : ℚ) : 
  (∃ a b : ℚ, ∀ x, 4*x^2 - (2*m+1)*x + 121 = (a*x + b)^2) → 
  (m = 43/2 ∨ m = -45/2) :=
sorry

end perfect_square_trinomial_l1569_156960


namespace task_completion_probability_l1569_156952

theorem task_completion_probability (p1 p2 : ℚ) (h1 : p1 = 3/8) (h2 : p2 = 3/5) :
  p1 * (1 - p2) = 3/20 := by
  sorry

end task_completion_probability_l1569_156952


namespace martin_failed_by_200_l1569_156962

/-- Calculates the number of marks by which a student failed an exam -/
def marksFailedBy (maxMarks passingPercentage studentScore : ℕ) : ℕ :=
  let passingMark := (passingPercentage * maxMarks) / 100
  passingMark - studentScore

/-- Proves that Martin failed the exam by 200 marks -/
theorem martin_failed_by_200 :
  let maxMarks : ℕ := 500
  let passingPercentage : ℕ := 80
  let martinScore : ℕ := 200
  let passingMark := (passingPercentage * maxMarks) / 100
  martinScore < passingMark →
  marksFailedBy maxMarks passingPercentage martinScore = 200 := by
  sorry

end martin_failed_by_200_l1569_156962


namespace stone_value_proof_l1569_156912

/-- Represents the worth of a precious stone based on its weight and a proportionality constant -/
def stone_worth (weight : ℝ) (k : ℝ) : ℝ := k * weight^2

/-- Calculates the total worth of two pieces of a stone -/
def pieces_worth (weight1 : ℝ) (weight2 : ℝ) (k : ℝ) : ℝ :=
  stone_worth weight1 k + stone_worth weight2 k

theorem stone_value_proof (k : ℝ) :
  let original_weight : ℝ := 35
  let smaller_piece : ℝ := 2 * (original_weight / 7)
  let larger_piece : ℝ := 5 * (original_weight / 7)
  let loss : ℝ := 5000
  stone_worth original_weight k - pieces_worth smaller_piece larger_piece k = loss →
  stone_worth original_weight k = 12250 := by
sorry

end stone_value_proof_l1569_156912


namespace sum_squares_interior_8th_row_l1569_156901

/-- Pascal's Triangle row function -/
def pascal_row (n : ℕ) : List ℕ := sorry

/-- Function to get interior numbers of a row -/
def interior_numbers (row : List ℕ) : List ℕ := sorry

/-- Sum of squares function -/
def sum_of_squares (list : List ℕ) : ℕ := sorry

/-- Theorem: Sum of squares of interior numbers in 8th row of Pascal's Triangle is 3430 -/
theorem sum_squares_interior_8th_row : 
  sum_of_squares (interior_numbers (pascal_row 8)) = 3430 := by sorry

end sum_squares_interior_8th_row_l1569_156901


namespace miae_closer_estimate_l1569_156959

def bowl_volume : ℝ := 1000  -- in milliliters
def miae_estimate : ℝ := 1100  -- in milliliters
def hyori_estimate : ℝ := 850  -- in milliliters

theorem miae_closer_estimate :
  |miae_estimate - bowl_volume| < |hyori_estimate - bowl_volume| := by
  sorry

end miae_closer_estimate_l1569_156959


namespace quadratic_completion_square_l1569_156979

theorem quadratic_completion_square (a : ℝ) : 
  (a > 0) → 
  (∃ n : ℝ, ∀ x : ℝ, x^2 + a*x + 27 = (x + n)^2 + 3) → 
  a = 4 * Real.sqrt 6 := by
sorry

end quadratic_completion_square_l1569_156979


namespace coordinates_wrt_origin_l1569_156956

-- Define a point in a 2D Cartesian coordinate system
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the origin
def origin : Point2D := ⟨0, 0⟩

-- Define the point P
def P : Point2D := ⟨2, 4⟩

-- Theorem stating that the coordinates of P with respect to the origin are (2,4)
theorem coordinates_wrt_origin :
  (P.x - origin.x = 2) ∧ (P.y - origin.y = 4) := by
  sorry

end coordinates_wrt_origin_l1569_156956


namespace two_digit_number_transformation_l1569_156917

/-- Given a two-digit integer n = 10a + b, where n = (k+1)(a + b),
    prove that 10(a+1) + (b+1) = ((k+1)(a + b) + 11) / (a + b + 2) * (a + b + 2) -/
theorem two_digit_number_transformation (a b k : ℕ) (h1 : 10*a + b = (k+1)*(a + b)) :
  10*(a+1) + (b+1) = ((k+1)*(a + b) + 11) / (a + b + 2) * (a + b + 2) := by
  sorry

end two_digit_number_transformation_l1569_156917


namespace multiple_problem_l1569_156915

theorem multiple_problem (S L : ℝ) (h1 : S = 10) (h2 : S + L = 24) :
  ∃ M : ℝ, 7 * S = M * L ∧ M = 5 := by
  sorry

end multiple_problem_l1569_156915


namespace bacteria_increase_l1569_156940

theorem bacteria_increase (original : ℕ) (current : ℕ) (increase : ℕ) : 
  original = 600 → current = 8917 → increase = current - original → increase = 8317 := by
sorry

end bacteria_increase_l1569_156940


namespace population_change_l1569_156914

/-- Proves that given an initial population of 15000, a 12% increase in the first year,
    and a final population of 14784 after two years, the percentage decrease in the second year is 12%. -/
theorem population_change (initial_population : ℝ) (first_year_increase : ℝ) (final_population : ℝ)
  (h1 : initial_population = 15000)
  (h2 : first_year_increase = 0.12)
  (h3 : final_population = 14784) :
  let population_after_first_year := initial_population * (1 + first_year_increase)
  let second_year_decrease := (population_after_first_year - final_population) / population_after_first_year
  second_year_decrease = 0.12 := by
  sorry

end population_change_l1569_156914


namespace investment_calculation_l1569_156988

/-- Given two investors p and q, where p invested 52000 and the profit is divided in the ratio 4:5,
    prove that q invested 65000. -/
theorem investment_calculation (p q : ℕ) : 
  p = 52000 → 
  (4 : ℚ) / 5 = p / q →
  q = 65000 := by
sorry

end investment_calculation_l1569_156988


namespace airport_distance_proof_l1569_156945

/-- The distance from Victor's home to the airport -/
def airport_distance : ℝ := 150

/-- Victor's initial speed -/
def initial_speed : ℝ := 60

/-- Victor's increased speed -/
def increased_speed : ℝ := 80

/-- Time Victor drives at initial speed -/
def initial_drive_time : ℝ := 0.5

/-- Time difference if Victor continued at initial speed -/
def late_time : ℝ := 0.25

/-- Time difference after increasing speed -/
def early_time : ℝ := 0.25

theorem airport_distance_proof :
  ∃ (planned_time : ℝ),
    -- Distance covered at initial speed
    initial_speed * initial_drive_time +
    -- Remaining distance if continued at initial speed
    initial_speed * (planned_time + late_time) =
    -- Distance covered at initial speed
    initial_speed * initial_drive_time +
    -- Remaining distance at increased speed
    increased_speed * (planned_time - early_time) ∧
    -- Total distance equals airport_distance
    airport_distance = initial_speed * initial_drive_time +
                       increased_speed * (planned_time - early_time) := by
  sorry

end airport_distance_proof_l1569_156945


namespace remainder_of_198_digit_sequence_l1569_156993

/-- The sum of digits function for a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The function that generates the sequence of digits up to the nth digit -/
def sequenceUpTo (n : ℕ) : List ℕ := sorry

/-- Sum of all digits in the sequence up to the nth digit -/
def sumOfSequenceDigits (n : ℕ) : ℕ := 
  (sequenceUpTo n).map sumOfDigits |>.sum

theorem remainder_of_198_digit_sequence : 
  sumOfSequenceDigits 198 % 9 = 6 := by sorry

end remainder_of_198_digit_sequence_l1569_156993


namespace single_interval_condition_l1569_156989

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- Condition for single interval solution of [x]^2 + k[x] + l = 0 -/
theorem single_interval_condition (k l : ℤ) : 
  (∃ (a b : ℝ), ∀ x, (floor x)^2 + k * (floor x) + l = 0 ↔ a ≤ x ∧ x < b) ↔ 
  l = floor ((k^2 : ℝ) / 4) :=
sorry

end single_interval_condition_l1569_156989


namespace simulation_needed_for_exact_probability_l1569_156999

structure Player where
  money : Nat

structure GameState where
  players : List Player

def initial_state : GameState :=
  { players := [{ money := 2 }, { money := 2 }, { money := 2 }] }

def can_give_money (p : Player) : Bool :=
  p.money > 1

def ring_bell (state : GameState) : GameState :=
  sorry

def is_final_state (state : GameState) : Bool :=
  state.players.all (fun p => p.money = 2)

noncomputable def probability_of_final_state (num_rings : Nat) : ℝ :=
  sorry

theorem simulation_needed_for_exact_probability :
  ∀ (analytical_function : Nat → ℝ),
    ∃ (ε : ℝ), ε > 0 ∧
      |probability_of_final_state 2019 - analytical_function 2019| > ε :=
by sorry

end simulation_needed_for_exact_probability_l1569_156999


namespace max_a_value_l1569_156908

theorem max_a_value (x a : ℤ) : 
  (∃ x : ℤ, x^2 + a*x = -24) → 
  (a > 0) → 
  a ≤ 25 :=
by sorry

end max_a_value_l1569_156908


namespace sequence_inequality_l1569_156968

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 1/2)
  (h1 : ∀ k, k < n → a (k + 1) = a k + (1/n) * (a k)^2) :
  1 - 1/n < a n ∧ a n < 1 := by
  sorry

end sequence_inequality_l1569_156968


namespace cuboids_intersecting_diagonal_l1569_156916

/-- Represents a cuboid with integer side lengths -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cube with integer side length -/
structure Cube where
  sideLength : ℕ

/-- Counts the number of cuboids intersecting the diagonal of a cube -/
def countIntersectingCuboids (cuboid : Cuboid) (cube : Cube) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem cuboids_intersecting_diagonal
  (smallCuboid : Cuboid)
  (largeCube : Cube)
  (h1 : smallCuboid.length = 2)
  (h2 : smallCuboid.width = 3)
  (h3 : smallCuboid.height = 5)
  (h4 : largeCube.sideLength = 90)
  (h5 : largeCube.sideLength % smallCuboid.length = 0)
  (h6 : largeCube.sideLength % smallCuboid.width = 0)
  (h7 : largeCube.sideLength % smallCuboid.height = 0) :
  countIntersectingCuboids smallCuboid largeCube = 65 := by
  sorry


end cuboids_intersecting_diagonal_l1569_156916


namespace largest_two_twos_l1569_156967

def two_twos_operation : ℕ → Prop :=
  λ n => ∃ (op : ℕ → ℕ → ℕ), n = op 2 2 ∨ n = 22

theorem largest_two_twos :
  ∀ n : ℕ, two_twos_operation n → n ≤ 22 :=
by
  sorry

#check largest_two_twos

end largest_two_twos_l1569_156967


namespace sum_remainder_is_six_l1569_156971

/-- Given positive integers a, b, c less than 7 satisfying certain congruences,
    prove that their sum has a remainder of 6 when divided by 7. -/
theorem sum_remainder_is_six (a b c : ℕ) 
    (ha : a < 7) (hb : b < 7) (hc : c < 7) 
    (ha_pos : a > 0) (hb_pos : b > 0) (hc_pos : c > 0)
    (h1 : a * b * c % 7 = 2)
    (h2 : (3 * c) % 7 = 4)
    (h3 : (4 * b) % 7 = (2 + b) % 7) :
    (a + b + c) % 7 = 6 := by
  sorry

end sum_remainder_is_six_l1569_156971


namespace football_team_right_handed_count_l1569_156929

theorem football_team_right_handed_count (total_players throwers : ℕ) : 
  total_players = 70 →
  throwers = 37 →
  (total_players - throwers) % 3 = 0 →
  (throwers + (total_players - throwers) * 2 / 3 = 59) :=
by
  sorry

end football_team_right_handed_count_l1569_156929


namespace function_composition_sqrt2_l1569_156943

theorem function_composition_sqrt2 (a : ℝ) (f : ℝ → ℝ) (h1 : 0 < a) :
  (∀ x, f x = a * x^2 - Real.sqrt 2) →
  f (f (Real.sqrt 2)) = -Real.sqrt 2 →
  a = Real.sqrt 2 / 2 := by
sorry

end function_composition_sqrt2_l1569_156943


namespace expression_value_l1569_156928

theorem expression_value : (100 - (3000 - 300)) + (3000 - (300 - 100)) = 200 := by
  sorry

end expression_value_l1569_156928


namespace simplify_and_ratio_l1569_156976

theorem simplify_and_ratio (m : ℝ) : ∃ (c d : ℝ), 
  (6 * m + 12) / 3 = c * m + d ∧ c / d = 1 / 2 := by
  sorry

end simplify_and_ratio_l1569_156976


namespace power_product_equality_l1569_156904

theorem power_product_equality (a b : ℝ) : 3 * a^2 * b * (-a)^2 = 3 * a^4 * b := by
  sorry

end power_product_equality_l1569_156904


namespace four_digit_cubes_divisible_by_16_l1569_156937

theorem four_digit_cubes_divisible_by_16 : 
  (∃! (list : List ℕ), 
    (∀ n ∈ list, 1000 ≤ (4 * n)^3 ∧ (4 * n)^3 ≤ 9999 ∧ (4 * n)^3 % 16 = 0) ∧ 
    list.length = 3) := by
  sorry

end four_digit_cubes_divisible_by_16_l1569_156937


namespace terminating_decimal_of_fraction_l1569_156939

theorem terminating_decimal_of_fraction (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 65 / 1000 →
  ∃ (a b : ℕ), (n : ℚ) / d = (a : ℚ) / (10 ^ b) ∧ (a : ℚ) / (10 ^ b) = 0.065 :=
by sorry

end terminating_decimal_of_fraction_l1569_156939


namespace min_ones_is_one_l1569_156931

/-- Represents the count of squares of each size --/
structure SquareCounts where
  threes : Nat
  twos : Nat
  ones : Nat

/-- Checks if the given square counts fit within a 7x7 square --/
def fitsIn7x7 (counts : SquareCounts) : Prop :=
  9 * counts.threes + 4 * counts.twos + counts.ones = 49

/-- Defines a valid square division --/
def isValidDivision (counts : SquareCounts) : Prop :=
  fitsIn7x7 counts ∧ counts.threes ≥ 0 ∧ counts.twos ≥ 0 ∧ counts.ones ≥ 0

/-- The main theorem stating that the minimum number of 1x1 squares is 1 --/
theorem min_ones_is_one :
  ∃ (counts : SquareCounts), isValidDivision counts ∧ counts.ones = 1 ∧
  (∀ (other : SquareCounts), isValidDivision other → other.ones ≥ counts.ones) :=
sorry

end min_ones_is_one_l1569_156931


namespace kerrys_age_l1569_156990

/-- Given the conditions of Kerry's birthday candles, prove his age --/
theorem kerrys_age (num_cakes : ℕ) (candles_per_box : ℕ) (cost_per_box : ℚ) (total_cost : ℚ) :
  num_cakes = 5 →
  candles_per_box = 22 →
  cost_per_box = 9/2 →
  total_cost = 27 →
  ∃ (age : ℕ), age = 26 ∧ (num_cakes * age : ℚ) ≤ (total_cost / cost_per_box * candles_per_box) :=
by sorry

end kerrys_age_l1569_156990


namespace number_of_divisors_36_l1569_156958

theorem number_of_divisors_36 : Nat.card {d : ℕ | d > 0 ∧ 36 % d = 0} = 9 := by
  sorry

end number_of_divisors_36_l1569_156958


namespace expression_values_l1569_156900

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  expr = 5 ∨ expr = 1 ∨ expr = -3 ∨ expr = -5 := by
  sorry

end expression_values_l1569_156900


namespace probability_three_yellow_one_white_l1569_156941

/-- The probability of drawing 3 yellow balls followed by 1 white ball from a box
    containing 5 yellow balls and 4 white balls, where yellow balls are returned
    after being drawn. -/
theorem probability_three_yellow_one_white (yellow_balls : ℕ) (white_balls : ℕ)
    (h_yellow : yellow_balls = 5) (h_white : white_balls = 4) :
    (yellow_balls / (yellow_balls + white_balls : ℚ))^3 *
    (white_balls / (yellow_balls + white_balls : ℚ)) =
    (5/9 : ℚ)^3 * (4/9 : ℚ) :=
by sorry

end probability_three_yellow_one_white_l1569_156941


namespace lillians_candies_l1569_156924

theorem lillians_candies (initial_candies final_candies : ℕ) 
  (h1 : initial_candies = 88)
  (h2 : final_candies = 93) :
  final_candies - initial_candies = 5 := by
  sorry

end lillians_candies_l1569_156924


namespace parallel_linear_functions_min_value_l1569_156996

/-- Two linear functions with parallel graphs not parallel to coordinate axes -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x + b) ∧ (∀ x, g x = a * x + c)

/-- The minimum value of a quadratic function -/
def quadratic_min (h : ℝ → ℝ) : ℝ := sorry

theorem parallel_linear_functions_min_value 
  (funcs : ParallelLinearFunctions)
  (h_min : quadratic_min (λ x => (funcs.f x)^2 + 5 * funcs.g x) = -17) :
  quadratic_min (λ x => (funcs.g x)^2 + 5 * funcs.f x) = 9/2 := by
  sorry

end parallel_linear_functions_min_value_l1569_156996


namespace factoring_expression_l1569_156978

theorem factoring_expression (a b : ℝ) : 6 * a^2 * b + 2 * a = 2 * a * (3 * a * b + 1) := by
  sorry

end factoring_expression_l1569_156978


namespace bug_flower_problem_l1569_156923

theorem bug_flower_problem (total_bugs : ℕ) (total_flowers : ℕ) (flowers_per_bug : ℕ) :
  total_bugs = 3 →
  total_flowers = 6 →
  total_flowers = total_bugs * flowers_per_bug →
  flowers_per_bug = 2 :=
by
  sorry

end bug_flower_problem_l1569_156923


namespace polynomial_equality_l1569_156909

theorem polynomial_equality (a b c d e : ℝ) :
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = (2*x - 1)^4) →
  a + c = 40 := by
  sorry

end polynomial_equality_l1569_156909


namespace isosceles_triangle_perimeter_l1569_156992

/-- An isosceles triangle with side lengths 5 and 10 has a perimeter of 25 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 10 → b = 10 → c = 5 →
  a + b > c → b + c > a → c + a > b →
  a + b + c = 25 := by
  sorry

end isosceles_triangle_perimeter_l1569_156992


namespace parabola_shift_theorem_l1569_156927

/-- Represents a parabola in the form y = a(x-h)² + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    h := p.h - dx
    k := p.k + dy }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 3 ∧ p.h = 4 ∧ p.k = 3 →
  let p' := shift p 4 (-4)
  p'.a * X ^ 2 + p'.a * p'.h ^ 2 - 2 * p'.a * p'.h * X + p'.k = 3 * X ^ 2 - 1 := by
  sorry

#check parabola_shift_theorem

end parabola_shift_theorem_l1569_156927


namespace isosceles_triangle_perimeter_l1569_156998

/-- An isosceles triangle with side lengths 4 and 8 has a perimeter of 20. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 8 → b = 8 → c = 4 →
  (a = b ∨ a = c ∨ b = c) →  -- isosceles condition
  a + b + c = 20 := by sorry

end isosceles_triangle_perimeter_l1569_156998


namespace lap_time_calculation_l1569_156947

/-- Represents the field and boy's running conditions -/
structure FieldConditions where
  side_length : ℝ
  normal_speed : ℝ
  sandy_length : ℝ
  sandy_speed_reduction : ℝ
  hurdle_count_low : ℕ
  hurdle_count_high : ℕ
  hurdle_time_low : ℝ
  hurdle_time_high : ℝ
  corner_slowdown : ℝ

/-- Calculates the total time to complete one lap around the field -/
def total_lap_time (conditions : FieldConditions) : ℝ :=
  sorry

/-- Theorem stating the total time to complete one lap -/
theorem lap_time_calculation (conditions : FieldConditions) 
  (h1 : conditions.side_length = 50)
  (h2 : conditions.normal_speed = 9 * 1000 / 3600)
  (h3 : conditions.sandy_length = 20)
  (h4 : conditions.sandy_speed_reduction = 0.25)
  (h5 : conditions.hurdle_count_low = 2)
  (h6 : conditions.hurdle_count_high = 2)
  (h7 : conditions.hurdle_time_low = 2)
  (h8 : conditions.hurdle_time_high = 3)
  (h9 : conditions.corner_slowdown = 2) :
  total_lap_time conditions = 138.68 := by
  sorry

end lap_time_calculation_l1569_156947


namespace solve_apple_problem_l1569_156997

def apple_problem (initial_apples : ℕ) (pears_difference : ℕ) (pears_bought : ℕ) (final_total : ℕ) : Prop :=
  let initial_pears : ℕ := initial_apples + pears_difference
  let new_pears : ℕ := initial_pears + pears_bought
  let apples_sold : ℕ := initial_apples + new_pears - final_total
  apples_sold = 599

theorem solve_apple_problem :
  apple_problem 1238 374 276 2527 :=
by sorry

end solve_apple_problem_l1569_156997


namespace revenue_calculation_l1569_156954

-- Define the initial and final counts of phones
def samsung_start : ℕ := 14
def samsung_end : ℕ := 10
def iphone_start : ℕ := 8
def iphone_end : ℕ := 5

-- Define the number of damaged phones
def samsung_damaged : ℕ := 2
def iphone_damaged : ℕ := 1

-- Define the retail prices
def samsung_price : ℚ := 800
def iphone_price : ℚ := 1000

-- Define the discount and tax rates
def samsung_discount : ℚ := 0.10
def samsung_tax : ℚ := 0.12
def iphone_discount : ℚ := 0.15
def iphone_tax : ℚ := 0.10

-- Calculate the number of phones sold
def samsung_sold : ℕ := samsung_start - samsung_end - samsung_damaged
def iphone_sold : ℕ := iphone_start - iphone_end - iphone_damaged

-- Calculate the final price for each phone type after discount and tax
def samsung_final_price : ℚ := samsung_price * (1 - samsung_discount) * (1 + samsung_tax)
def iphone_final_price : ℚ := iphone_price * (1 - iphone_discount) * (1 + iphone_tax)

-- Calculate the total revenue
def total_revenue : ℚ := samsung_final_price * samsung_sold + iphone_final_price * iphone_sold

-- Theorem to prove
theorem revenue_calculation : total_revenue = 3482.80 := by
  sorry

end revenue_calculation_l1569_156954


namespace a_greater_than_b_greater_than_one_l1569_156975

theorem a_greater_than_b_greater_than_one
  (n : ℕ) (a b : ℝ) 
  (h_n : n ≥ 2)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_eq : a^n = a + 1)
  (h_b_eq : b^(2*n) = b + 3*a) :
  a > b ∧ b > 1 := by
sorry

end a_greater_than_b_greater_than_one_l1569_156975


namespace min_sum_of_squares_with_diff_l1569_156987

theorem min_sum_of_squares_with_diff (x y : ℤ) (h : x^2 - y^2 = 165) :
  ∃ (a b : ℤ), a^2 - b^2 = 165 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 173 :=
sorry

end min_sum_of_squares_with_diff_l1569_156987


namespace quilt_cost_calculation_l1569_156905

/-- The cost of a rectangular quilt -/
def quilt_cost (length width cost_per_sqft : ℝ) : ℝ :=
  length * width * cost_per_sqft

/-- Theorem: The cost of a 7ft by 8ft quilt at $40 per square foot is $2240 -/
theorem quilt_cost_calculation :
  quilt_cost 7 8 40 = 2240 := by
  sorry

end quilt_cost_calculation_l1569_156905


namespace sum_of_coefficients_zero_l1569_156949

theorem sum_of_coefficients_zero 
  (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ a₁₄ : ℝ) :
  (∀ x : ℝ, (1 + x - x^2)^3 * (1 - 2*x^2)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + 
    a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12 + a₁₃*x^13 + a₁₄*x^14) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ + a₁₂ + a₁₃ + a₁₄ = 0 :=
by
  sorry

end sum_of_coefficients_zero_l1569_156949


namespace consecutive_even_integers_sum_l1569_156918

theorem consecutive_even_integers_sum (x : ℕ) (h1 : x > 4) : 
  (x - 4) * (x - 2) * x * (x + 2) = 48 * (4 * x) → 
  (x - 4) + (x - 2) + x + (x + 2) = 28 := by
  sorry

end consecutive_even_integers_sum_l1569_156918


namespace cube_inscribed_in_sphere_surface_area_l1569_156942

/-- The surface area of a cube inscribed in a sphere with radius 5 units is 200 square units. -/
theorem cube_inscribed_in_sphere_surface_area :
  let r : ℝ := 5  -- radius of the sphere
  let s : ℝ := 10 * Real.sqrt 3 / 3  -- edge length of the cube
  6 * s^2 = 200 := by sorry

end cube_inscribed_in_sphere_surface_area_l1569_156942


namespace salary_change_percentage_l1569_156973

theorem salary_change_percentage (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 64 / 100 → x = 6 := by
  sorry

end salary_change_percentage_l1569_156973


namespace angelinas_speed_to_gym_l1569_156938

-- Define the distances and time difference
def distance_home_to_grocery : ℝ := 1200
def distance_grocery_to_gym : ℝ := 480
def time_difference : ℝ := 40

-- Define the relationship between speeds
def speed_grocery_to_gym (v : ℝ) : ℝ := 2 * v

-- Theorem statement
theorem angelinas_speed_to_gym :
  ∃ v : ℝ, v > 0 ∧
  distance_home_to_grocery / v - distance_grocery_to_gym / (speed_grocery_to_gym v) = time_difference ∧
  speed_grocery_to_gym v = 48 := by
  sorry

end angelinas_speed_to_gym_l1569_156938


namespace middle_number_is_twelve_l1569_156922

/-- Given three distinct integers x, y, z satisfying the given conditions,
    prove that the middle number y equals 12. -/
theorem middle_number_is_twelve (x y z : ℤ)
  (h_distinct : x < y ∧ y < z)
  (h_sum1 : x + y = 21)
  (h_sum2 : x + z = 25)
  (h_sum3 : y + z = 28) :
  y = 12 := by sorry

end middle_number_is_twelve_l1569_156922


namespace solve_equation_l1569_156982

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 4 / 3 → x = -27 / 11 := by
sorry

end solve_equation_l1569_156982


namespace comic_arrangement_count_l1569_156920

/-- The number of ways to arrange comics as described in the problem -/
def comic_arrangements (batman : ℕ) (xmen : ℕ) (calvin_hobbes : ℕ) : ℕ :=
  (Nat.factorial (batman + xmen)) * (Nat.factorial calvin_hobbes) * 2

/-- Theorem stating the correct number of arrangements for the given comic counts -/
theorem comic_arrangement_count :
  comic_arrangements 7 6 5 = 1494084992000 := by
  sorry

end comic_arrangement_count_l1569_156920


namespace dog_catch_ball_time_l1569_156948

/-- The time it takes for a dog to catch up to a thrown ball -/
theorem dog_catch_ball_time (ball_speed : ℝ) (ball_time : ℝ) (dog_speed : ℝ) :
  ball_speed = 20 →
  ball_time = 8 →
  dog_speed = 5 →
  (ball_speed * ball_time) / dog_speed = 32 := by
  sorry

#check dog_catch_ball_time

end dog_catch_ball_time_l1569_156948
