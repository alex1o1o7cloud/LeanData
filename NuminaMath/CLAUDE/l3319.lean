import Mathlib

namespace unique_total_prices_l3319_331968

def gift_prices : Finset ℕ := {2, 5, 8, 11, 14}
def box_prices : Finset ℕ := {3, 6, 9, 12, 15}

def total_prices : Finset ℕ := 
  Finset.image (λ (p : ℕ × ℕ) => p.1 + p.2) (gift_prices.product box_prices)

theorem unique_total_prices : Finset.card total_prices = 9 := by
  sorry

end unique_total_prices_l3319_331968


namespace units_digit_problem_l3319_331998

theorem units_digit_problem (n : ℤ) : n = (30 * 31 * 32 * 33 * 34 * 35) / 2500 → n % 10 = 1 := by
  sorry

end units_digit_problem_l3319_331998


namespace inequality_preservation_l3319_331995

theorem inequality_preservation (a b : ℝ) (h : a < b) : a / 3 < b / 3 := by
  sorry

end inequality_preservation_l3319_331995


namespace train_distance_problem_l3319_331967

/-- Theorem: Train Distance Problem
Given:
- A passenger train travels from A to B at 60 km/h for 2/3 of the journey, then at 30 km/h for the rest.
- A high-speed train travels at 120 km/h and catches up with the passenger train 80 km before B.
Prove that the distance from A to B is 360 km. -/
theorem train_distance_problem (D : ℝ) 
  (h1 : D > 0)  -- Distance is positive
  (h2 : ∃ t : ℝ, t > 0 ∧ (2/3 * D) / 60 + (1/3 * D) / 30 = (D - 80) / 120 + t)
  : D = 360 := by
  sorry

end train_distance_problem_l3319_331967


namespace largest_value_problem_l3319_331926

theorem largest_value_problem : 
  let a := 12345 + 1/5678
  let b := 12345 - 1/5678
  let c := 12345 * 1/5678
  let d := 12345 / (1/5678)
  let e := 12345.5678
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end largest_value_problem_l3319_331926


namespace function_domain_implies_m_range_l3319_331932

/-- If f(x) = √(mx² - 6mx + m + 8) has a domain of ℝ, then m ∈ [0, 1] -/
theorem function_domain_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, mx^2 - 6*m*x + m + 8 ≥ 0) → 0 ≤ m ∧ m ≤ 1 := by
sorry

end function_domain_implies_m_range_l3319_331932


namespace rotated_line_x_intercept_l3319_331957

/-- The x-coordinate of the x-intercept of a rotated line -/
theorem rotated_line_x_intercept 
  (m : Real → Real → Prop) -- Original line
  (θ : Real) -- Rotation angle
  (p : Real × Real) -- Point of rotation
  (n : Real → Real → Prop) -- Rotated line
  (h1 : ∀ x y, m x y ↔ 4 * x - 3 * y + 20 = 0) -- Equation of line m
  (h2 : θ = π / 3) -- 60° in radians
  (h3 : p = (10, 10)) -- Point of rotation
  (h4 : ∀ x y, n x y ↔ 
    y - p.2 = ((24 + 25 * Real.sqrt 3) / (-39)) * (x - p.1)) -- Equation of line n
  (C : Real) -- Constant C
  (h5 : C = 10 - (240 + 250 * Real.sqrt 3) / (-39)) -- Definition of C
  : ∃ x_intercept : Real, 
    x_intercept = -39 * C / (24 + 25 * Real.sqrt 3) ∧ 
    n x_intercept 0 := by sorry

end rotated_line_x_intercept_l3319_331957


namespace floor_sum_example_l3319_331956

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by sorry

end floor_sum_example_l3319_331956


namespace circular_garden_ratio_l3319_331940

theorem circular_garden_ratio (r : ℝ) (h : r = 6) : 
  (2 * Real.pi * r) / (Real.pi * r^2) = 1/3 := by
  sorry

end circular_garden_ratio_l3319_331940


namespace cone_volume_from_cylinder_l3319_331953

/-- The volume of a cone with the same radius and height as a cylinder with volume 54π cm³ is 18π cm³ -/
theorem cone_volume_from_cylinder (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  π * r^2 * h = 54 * π → (1/3) * π * r^2 * h = 18 * π := by
  sorry

end cone_volume_from_cylinder_l3319_331953


namespace power_sum_difference_l3319_331925

theorem power_sum_difference : 2^1 + 1^0 - 3^2 = -6 := by
  sorry

end power_sum_difference_l3319_331925


namespace max_k_value_l3319_331962

theorem max_k_value (k : ℝ) : 
  (k > 0 ∧ ∀ x > 0, k * Real.log (k * x) - Real.exp x ≤ 0) →
  k ≤ Real.exp 1 :=
sorry

end max_k_value_l3319_331962


namespace hyperbola_real_axis_length_l3319_331944

/-- The hyperbola and parabola intersect at two points A and B -/
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The common focus of the hyperbola and parabola -/
def CommonFocus : ℝ × ℝ := (1, 2)

/-- The hyperbola equation -/
def isOnHyperbola (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (p.1^2 / a^2) - (p.2^2 / b^2) = 1

/-- The parabola equation -/
def isOnParabola (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

/-- Line AB passes through the common focus -/
def lineABThroughFocus (points : IntersectionPoints) : Prop :=
  ∃ (t : ℝ), (1 - t) * points.A.1 + t * points.B.1 = CommonFocus.1 ∧
             (1 - t) * points.A.2 + t * points.B.2 = CommonFocus.2

/-- Theorem: The length of the real axis of the hyperbola is 2√2 - 2 -/
theorem hyperbola_real_axis_length (a b : ℝ) (points : IntersectionPoints) :
  isOnHyperbola a b CommonFocus →
  isOnParabola CommonFocus →
  lineABThroughFocus points →
  2 * a = 2 * Real.sqrt 2 - 2 := by
  sorry

end hyperbola_real_axis_length_l3319_331944


namespace parking_lot_width_l3319_331977

/-- Calculates the width of a parking lot given its specifications -/
theorem parking_lot_width
  (total_length : ℝ)
  (usable_percentage : ℝ)
  (area_per_car : ℝ)
  (total_cars : ℕ)
  (h1 : total_length = 500)
  (h2 : usable_percentage = 0.8)
  (h3 : area_per_car = 10)
  (h4 : total_cars = 16000) :
  (total_length * usable_percentage * (total_cars : ℝ) * area_per_car) / (total_length * usable_percentage) = 400 := by
  sorry

#check parking_lot_width

end parking_lot_width_l3319_331977


namespace zoo_population_increase_l3319_331982

theorem zoo_population_increase (c p : ℕ) (h1 : c * 3 = p) (h2 : (c + 2) * 3 = p + 6) : True :=
by sorry

end zoo_population_increase_l3319_331982


namespace go_stones_count_l3319_331955

theorem go_stones_count (n : ℕ) (h1 : n^2 + 3 + 44 = (n + 2)^2) : n^2 + 3 = 103 := by
  sorry

#check go_stones_count

end go_stones_count_l3319_331955


namespace square_sum_reciprocal_l3319_331910

theorem square_sum_reciprocal (x : ℝ) (h : x + (1/x) = 5) : x^2 + (1/x)^2 = 23 := by
  sorry

end square_sum_reciprocal_l3319_331910


namespace problem_statement_l3319_331984

theorem problem_statement (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = {0, a^2, a+b} → a^2016 + b^2016 = 1 := by
sorry

end problem_statement_l3319_331984


namespace pure_imaginary_implies_m_eq_neg_three_l3319_331912

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Given that i is the imaginary unit, if the complex number z=(m^2+2m-3)+(m-1)i
    is a pure imaginary number, then m = -3. -/
theorem pure_imaginary_implies_m_eq_neg_three (m : ℝ) :
  IsPureImaginary (Complex.mk (m^2 + 2*m - 3) (m - 1)) → m = -3 := by
  sorry

end pure_imaginary_implies_m_eq_neg_three_l3319_331912


namespace seating_arrangements_with_restriction_l3319_331920

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def circularArrangements (n : ℕ) : ℕ := factorial (n - 1)

def adjacentPairArrangements (n : ℕ) : ℕ := 2 * factorial (n - 2)

theorem seating_arrangements_with_restriction (total : ℕ) (restricted_pair : ℕ) :
  total = 12 →
  restricted_pair = 2 →
  circularArrangements total - adjacentPairArrangements total = 32659200 := by
  sorry

#eval circularArrangements 12 - adjacentPairArrangements 12

end seating_arrangements_with_restriction_l3319_331920


namespace janice_age_l3319_331970

theorem janice_age (current_year : ℕ) (mark_birth_year : ℕ) (graham_age_difference : ℕ) :
  current_year = 2021 →
  mark_birth_year = 1976 →
  graham_age_difference = 3 →
  (current_year - mark_birth_year - graham_age_difference) / 2 = 21 :=
by sorry

end janice_age_l3319_331970


namespace cube_sum_reciprocal_l3319_331976

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 4) : x^3 + 1/x^3 = 52 := by
  sorry

end cube_sum_reciprocal_l3319_331976


namespace necessary_not_sufficient_condition_l3319_331980

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ a b, a > b → a + 1 > b) ∧
  (∃ a b, a + 1 > b ∧ ¬(a > b)) :=
by sorry

end necessary_not_sufficient_condition_l3319_331980


namespace previous_year_300th_day_is_monday_l3319_331937

/-- Represents days of the week -/
inductive DayOfWeek
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Represents a year -/
structure Year where
  isLeapYear : Bool

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.monday => DayOfWeek.tuesday
  | DayOfWeek.tuesday => DayOfWeek.wednesday
  | DayOfWeek.wednesday => DayOfWeek.thursday
  | DayOfWeek.thursday => DayOfWeek.friday
  | DayOfWeek.friday => DayOfWeek.saturday
  | DayOfWeek.saturday => DayOfWeek.sunday
  | DayOfWeek.sunday => DayOfWeek.monday

/-- Calculates the day of the week after a given number of days -/
def advanceDays (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => advanceDays (nextDay start) n

/-- Main theorem -/
theorem previous_year_300th_day_is_monday 
  (currentYear : Year)
  (nextYear : Year)
  (h1 : advanceDays DayOfWeek.sunday 200 = DayOfWeek.sunday)
  (h2 : advanceDays DayOfWeek.sunday 100 = DayOfWeek.sunday) :
  advanceDays DayOfWeek.monday 300 = DayOfWeek.sunday :=
sorry

end previous_year_300th_day_is_monday_l3319_331937


namespace imaginary_power_sum_l3319_331978

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Statement of the problem -/
theorem imaginary_power_sum : i^23 + i^52 + i^103 = 1 - 2*i := by sorry

end imaginary_power_sum_l3319_331978


namespace angle_side_inequality_l3319_331902

-- Define a structure for triangles
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the property that larger angles are opposite longer sides
axiom larger_angle_longer_side {t : Triangle} : 
  ∀ (x y : Real), (x = t.A ∧ y = t.a) ∨ (x = t.B ∧ y = t.b) ∨ (x = t.C ∧ y = t.c) →
  ∀ (p q : Real), (p = t.A ∧ q = t.a) ∨ (p = t.B ∧ q = t.b) ∨ (p = t.C ∧ q = t.c) →
  x > p → y > q

-- Theorem statement
theorem angle_side_inequality (t : Triangle) : t.A < t.B → t.a < t.b := by
  sorry

end angle_side_inequality_l3319_331902


namespace bus_arrival_probabilities_l3319_331999

def prob_bus_A : ℝ := 0.7
def prob_bus_B : ℝ := 0.75

theorem bus_arrival_probabilities :
  (3 * prob_bus_A^2 * (1 - prob_bus_A) = 0.441) ∧
  (1 - (1 - prob_bus_A) * (1 - prob_bus_B) = 0.925) :=
by sorry

end bus_arrival_probabilities_l3319_331999


namespace loan_interest_rate_calculation_l3319_331942

/-- The interest rate for the second part of a loan, given specific conditions -/
theorem loan_interest_rate_calculation (total : ℝ) (second_part : ℝ) : 
  total = 2743 →
  second_part = 1688 →
  let first_part := total - second_part
  let interest_rate_first := 0.03
  let time_first := 8
  let time_second := 3
  let interest_first := first_part * interest_rate_first * time_first
  let interest_second := second_part * time_second
  ∃ (r : ℝ), interest_first = r * interest_second ∧ 
             r ≥ 0.0499 ∧ r ≤ 0.05 := by
  sorry

#check loan_interest_rate_calculation

end loan_interest_rate_calculation_l3319_331942


namespace fraction_simplification_l3319_331930

theorem fraction_simplification : (4 * 5) / 10 = 2 := by
  sorry

end fraction_simplification_l3319_331930


namespace binomial_10_choose_3_l3319_331996

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_choose_3_l3319_331996


namespace triangle_case1_triangle_case2_l3319_331985

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem 1: In a triangle ABC with c = 2, C = π/3, and area = √3, a = 2 and b = 2 -/
theorem triangle_case1 (t : Triangle) 
  (h1 : t.c = 2)
  (h2 : t.C = π / 3)
  (h3 : (1 / 2) * t.a * t.b * Real.sin t.C = Real.sqrt 3) :
  t.a = 2 ∧ t.b = 2 := by
  sorry

/-- Theorem 2: In a triangle ABC with c = 2, C = π/3, and sin C + sin(B-A) = sin 2A,
    either (a = 4√3/3 and b = 2√3/3) or (a = 2 and b = 2) -/
theorem triangle_case2 (t : Triangle)
  (h1 : t.c = 2)
  (h2 : t.C = π / 3)
  (h3 : Real.sin t.C + Real.sin (t.B - t.A) = Real.sin (2 * t.A)) :
  (t.a = (4 * Real.sqrt 3) / 3 ∧ t.b = (2 * Real.sqrt 3) / 3) ∨ 
  (t.a = 2 ∧ t.b = 2) := by
  sorry

end triangle_case1_triangle_case2_l3319_331985


namespace apps_added_minus_deleted_l3319_331929

theorem apps_added_minus_deleted (initial_apps added_apps final_apps : ℕ) :
  initial_apps = 115 →
  added_apps = 235 →
  final_apps = 178 →
  added_apps - (initial_apps + added_apps - final_apps) = 63 :=
by
  sorry

end apps_added_minus_deleted_l3319_331929


namespace perfect_square_iff_divisibility_l3319_331946

theorem perfect_square_iff_divisibility (A : ℕ+) :
  (∃ d : ℕ+, A = d^2) ↔
  (∀ n : ℕ+, ∃ j : ℕ+, j ≤ n ∧ n ∣ ((A + j)^2 - A)) :=
sorry

end perfect_square_iff_divisibility_l3319_331946


namespace eighth_root_of_256289062500_l3319_331993

theorem eighth_root_of_256289062500 : (256289062500 : ℝ) ^ (1/8 : ℝ) = 52 := by
  sorry

end eighth_root_of_256289062500_l3319_331993


namespace min_width_proof_l3319_331966

/-- The minimum width of a rectangular area satisfying the given conditions -/
def min_width : ℝ := 5

/-- The length of the rectangular area in terms of its width -/
def length (w : ℝ) : ℝ := w + 15

/-- The area of the rectangular enclosure -/
def area (w : ℝ) : ℝ := w * length w

theorem min_width_proof :
  (∀ w : ℝ, w > 0 → area w ≥ 100 → w ≥ min_width) ∧
  (area min_width ≥ 100) ∧
  (min_width > 0) :=
sorry

end min_width_proof_l3319_331966


namespace equation_solutions_l3319_331931

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 9 = 0 ↔ x = 3/2 ∨ x = -3/2) ∧
  (∀ x : ℝ, 64 * (x + 1)^3 = -125 ↔ x = -9/4) := by
  sorry

end equation_solutions_l3319_331931


namespace reflection_property_l3319_331951

/-- A reflection in R² --/
structure Reflection where
  line : ℝ × ℝ  -- Vector representing the line of reflection

/-- Apply a reflection to a point --/
def apply_reflection (r : Reflection) (p : ℝ × ℝ) : ℝ × ℝ := sorry

theorem reflection_property :
  ∃ (r : Reflection),
    apply_reflection r (3, 5) = (7, 1) ∧
    apply_reflection r (2, 7) = (-7, -2) := by
  sorry

end reflection_property_l3319_331951


namespace journey_time_calculation_l3319_331974

theorem journey_time_calculation (total_distance : ℝ) (initial_fraction : ℝ) (initial_time : ℝ) (lunch_time : ℝ) :
  total_distance = 200 →
  initial_fraction = 1/4 →
  initial_time = 1 →
  lunch_time = 1 →
  ∃ (total_time : ℝ), total_time = 5 := by
  sorry

end journey_time_calculation_l3319_331974


namespace shopping_money_l3319_331963

theorem shopping_money (initial_amount remaining_amount : ℝ) : 
  remaining_amount = initial_amount * (1 - 0.3) ∧ remaining_amount = 840 →
  initial_amount = 1200 := by
sorry

end shopping_money_l3319_331963


namespace odd_divisors_of_factorial_20_l3319_331903

/-- The factorial of 20 -/
def factorial_20 : ℕ := 2432902008176640000

/-- The total number of natural divisors of 20! -/
def total_divisors : ℕ := 41040

/-- Theorem: The number of odd natural divisors of 20! is 2160 -/
theorem odd_divisors_of_factorial_20 : 
  (Finset.filter (fun d => d % 2 = 1) (Nat.divisors factorial_20)).card = 2160 := by
  sorry

end odd_divisors_of_factorial_20_l3319_331903


namespace decreasing_function_positive_l3319_331981

/-- A decreasing function satisfying the given condition is always positive -/
theorem decreasing_function_positive (f : ℝ → ℝ) (hf : Monotone (fun x ↦ -f x)) 
    (h : ∀ x, f x / (deriv f x) + x < 1) : ∀ x, f x > 0 := by
  sorry

end decreasing_function_positive_l3319_331981


namespace min_reciprocal_sum_min_reciprocal_sum_equality_l3319_331950

theorem min_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 3) :
  (1 / x + 1 / y + 1 / z) ≥ 3 := by
  sorry

theorem min_reciprocal_sum_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 3) :
  (1 / x + 1 / y + 1 / z = 3) ↔ (x = 1 ∧ y = 1 ∧ z = 1) := by
  sorry

end min_reciprocal_sum_min_reciprocal_sum_equality_l3319_331950


namespace difference_of_squares_and_perfect_squares_l3319_331901

theorem difference_of_squares_and_perfect_squares : 
  (102^2 - 98^2 = 800) ∧ 
  (¬ ∃ n : ℕ, n^2 = 102) ∧ 
  (¬ ∃ m : ℕ, m^2 = 98) := by
  sorry

end difference_of_squares_and_perfect_squares_l3319_331901


namespace largest_multiple_of_seven_under_hundred_l3319_331941

theorem largest_multiple_of_seven_under_hundred : 
  ∀ n : ℕ, n * 7 < 100 → n * 7 ≤ 98 :=
by
  sorry

end largest_multiple_of_seven_under_hundred_l3319_331941


namespace set_operation_result_l3319_331989

def A : Set ℕ := {0, 1, 2, 4, 5, 7, 8}
def B : Set ℕ := {1, 3, 6, 7, 9}
def C : Set ℕ := {3, 4, 7, 8}

theorem set_operation_result : (A ∩ B) ∪ C = {1, 3, 4, 7, 8} := by
  sorry

end set_operation_result_l3319_331989


namespace range_of_a_for_two_distinct_roots_l3319_331936

theorem range_of_a_for_two_distinct_roots : 
  ∀ a : ℝ, (∃! x y : ℝ, x ≠ y ∧ |x^2 - 5*x| = a) → (a = 0 ∨ a > 25/4) :=
by sorry

end range_of_a_for_two_distinct_roots_l3319_331936


namespace game_show_probability_l3319_331971

def num_questions : ℕ := 4
def num_options : ℕ := 4
def min_correct : ℕ := 3

def prob_correct_one : ℚ := 1 / num_options

def prob_all_correct : ℚ := prob_correct_one ^ num_questions

def prob_exactly_three_correct : ℚ := num_questions * (prob_correct_one ^ 3) * (1 - prob_correct_one)

theorem game_show_probability :
  prob_all_correct + prob_exactly_three_correct = 13 / 256 := by
  sorry

end game_show_probability_l3319_331971


namespace union_equals_B_implies_m_leq_neg_three_l3319_331939

def A : Set ℝ := {x | x^2 - 3*x - 10 < 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 1 - 3*m}

theorem union_equals_B_implies_m_leq_neg_three (m : ℝ) : A ∪ B m = B m → m ≤ -3 := by
  sorry

end union_equals_B_implies_m_leq_neg_three_l3319_331939


namespace subset_P_l3319_331907

def P : Set ℝ := {x | x ≤ 3}

theorem subset_P : {-1} ⊆ P := by sorry

end subset_P_l3319_331907


namespace problem_1_l3319_331934

theorem problem_1 : 
  (-1.75) - 6.3333333333 - 2.25 + (10/3) = -7 := by sorry

end problem_1_l3319_331934


namespace min_sum_dimensions_2310_l3319_331947

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- Calculates the volume of a box given its dimensions -/
def volume (d : BoxDimensions) : ℕ :=
  d.length.val * d.width.val * d.height.val

/-- Calculates the sum of dimensions of a box -/
def sumDimensions (d : BoxDimensions) : ℕ :=
  d.length.val + d.width.val + d.height.val

/-- Theorem: The minimum sum of dimensions for a box with volume 2310 is 42 -/
theorem min_sum_dimensions_2310 :
  (∃ d : BoxDimensions, volume d = 2310) →
  (∀ d : BoxDimensions, volume d = 2310 → sumDimensions d ≥ 42) ∧
  (∃ d : BoxDimensions, volume d = 2310 ∧ sumDimensions d = 42) :=
sorry

end min_sum_dimensions_2310_l3319_331947


namespace ellipse_and_triangle_area_l3319_331933

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the inscribed circle
def inscribed_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 2

-- Define the parabola E
def parabola_E (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ p > 0

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y = x + m ∧ 0 ≤ m ∧ m ≤ 1

-- State the theorem
theorem ellipse_and_triangle_area :
  ∀ (a b p m : ℝ) (x y : ℝ),
  ellipse_C a b x y →
  inscribed_circle x y →
  parabola_E p x y →
  line_l m x y →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    parabola_E p x₁ y₁ ∧ 
    parabola_E p x₂ y₂ ∧ 
    line_l m x₁ y₁ ∧ 
    line_l m x₂ y₂ ∧ 
    x₁ ≠ x₂) →
  (a^2 = 8 ∧ b^2 = 4) ∧
  (∃ (S : ℝ), S = (32 * Real.sqrt 6) / 9 ∧
    ∀ (S' : ℝ), S' ≤ S) :=
by sorry

end ellipse_and_triangle_area_l3319_331933


namespace no_solution_system_l3319_331943

theorem no_solution_system :
  ¬∃ (x y : ℝ), (3 * x - 4 * y = 12) ∧ (9 * x - 12 * y = 15) := by
  sorry

end no_solution_system_l3319_331943


namespace mirasol_account_balance_l3319_331949

def remaining_amount (initial : ℕ) (expense1 : ℕ) (expense2 : ℕ) : ℕ :=
  initial - (expense1 + expense2)

theorem mirasol_account_balance : remaining_amount 50 10 30 = 10 := by
  sorry

end mirasol_account_balance_l3319_331949


namespace expression_evaluation_l3319_331900

theorem expression_evaluation :
  let x : ℚ := -1
  let y : ℚ := 2
  (2*x + y) * (2*x - y) - (8*x^3*y - 2*x*y^3 - x^2*y^2) / (2*x*y) = -1 :=
by sorry

end expression_evaluation_l3319_331900


namespace two_color_theorem_l3319_331908

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A region in the plane --/
inductive Region
  | Inside (n : ℕ) -- Inside n circles
  | Outside        -- Outside all circles

/-- The type of coloring function --/
def Coloring := Region → Fin 2

/-- Two regions are adjacent if they differ by crossing one circle boundary --/
def adjacent (r1 r2 : Region) : Prop :=
  match r1, r2 with
  | Region.Inside n, Region.Inside m => n + 1 = m ∨ m + 1 = n
  | Region.Inside 1, Region.Outside => True
  | Region.Outside, Region.Inside 1 => True
  | _, _ => False

/-- A coloring is valid if adjacent regions have different colors --/
def valid_coloring (c : Coloring) : Prop :=
  ∀ r1 r2, adjacent r1 r2 → c r1 ≠ c r2

theorem two_color_theorem (circles : List Circle) :
  ∃ c : Coloring, valid_coloring c :=
sorry

end two_color_theorem_l3319_331908


namespace f_monotone_decreasing_l3319_331960

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem f_monotone_decreasing : 
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y :=
sorry

end f_monotone_decreasing_l3319_331960


namespace bella_steps_l3319_331994

-- Define the constants
def total_distance_miles : ℝ := 3
def speed_ratio : ℝ := 4
def feet_per_step : ℝ := 3
def feet_per_mile : ℝ := 5280

-- Define the theorem
theorem bella_steps : 
  ∀ (bella_speed : ℝ),
  bella_speed > 0 →
  (bella_speed * (total_distance_miles * feet_per_mile / (bella_speed * (1 + speed_ratio)))) / feet_per_step = 1056 :=
by
  sorry


end bella_steps_l3319_331994


namespace sin_cos_product_l3319_331979

theorem sin_cos_product (θ : Real) 
  (h : (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 2) : 
  Real.sin θ * Real.cos θ = 3/10 := by
  sorry

end sin_cos_product_l3319_331979


namespace rational_equation_power_l3319_331954

theorem rational_equation_power (x y : ℚ) 
  (h : |x + 5| + (y - 5)^2 = 0) : (x / y)^2023 = -1 := by
  sorry

end rational_equation_power_l3319_331954


namespace academic_year_school_days_l3319_331973

/-- The number of school days in the academic year -/
def school_days : ℕ := sorry

/-- The number of days Aliyah packs lunch -/
def aliyah_lunch_days : ℕ := sorry

/-- The number of days Becky packs lunch -/
def becky_lunch_days : ℕ := 45

theorem academic_year_school_days :
  (aliyah_lunch_days = 2 * becky_lunch_days) →
  (school_days = 2 * aliyah_lunch_days) →
  (school_days = 180) :=
by sorry

end academic_year_school_days_l3319_331973


namespace parallel_lines_probability_l3319_331983

/-- The number of points (centers of cube faces) -/
def num_points : ℕ := 6

/-- The number of ways to select 2 points from num_points -/
def num_lines : ℕ := num_points.choose 2

/-- The total number of ways for two people to each select a line -/
def total_selections : ℕ := num_lines * num_lines

/-- The number of pairs of lines that are parallel but not coincident -/
def parallel_pairs : ℕ := 12

/-- The probability of selecting two parallel but not coincident lines -/
def probability : ℚ := parallel_pairs / total_selections

theorem parallel_lines_probability :
  probability = 4 / 75 := by sorry

end parallel_lines_probability_l3319_331983


namespace pizza_theorem_l3319_331921

def pizza_eaten (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 1/3
  else pizza_eaten (n-1) + (1 - pizza_eaten (n-1)) / 2

theorem pizza_theorem :
  pizza_eaten 4 = 11/12 ∧ (1 - pizza_eaten 4) = 1/12 := by
  sorry

end pizza_theorem_l3319_331921


namespace female_managers_count_l3319_331986

/-- Represents a company with employees and managers -/
structure Company where
  total_employees : ℕ
  total_managers : ℕ
  male_employees : ℕ
  male_managers : ℕ
  female_employees : ℕ
  female_managers : ℕ

/-- The conditions of the company as described in the problem -/
def company_conditions (c : Company) : Prop :=
  c.female_employees = 500 ∧
  c.total_managers = (2 * c.total_employees) / 5 ∧
  c.male_managers = (2 * c.male_employees) / 5 ∧
  c.total_employees = c.male_employees + c.female_employees ∧
  c.total_managers = c.male_managers + c.female_managers

/-- The theorem to be proved -/
theorem female_managers_count (c : Company) :
  company_conditions c → c.female_managers = 200 := by
  sorry


end female_managers_count_l3319_331986


namespace min_value_sin_cos_min_value_achievable_l3319_331906

theorem min_value_sin_cos (x : ℝ) : 
  Real.sin x ^ 6 + 2 * Real.cos x ^ 6 ≥ 2/3 :=
sorry

theorem min_value_achievable : 
  ∃ x : ℝ, Real.sin x ^ 6 + 2 * Real.cos x ^ 6 = 2/3 :=
sorry

end min_value_sin_cos_min_value_achievable_l3319_331906


namespace triangle_theorem_l3319_331945

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
  t.b * Real.tan t.A = 2 * t.a * Real.sin t.B ∧
  t.a = Real.sqrt 7 ∧
  2 * t.b - t.c = 4

-- Define the theorem
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.A = π / 3 ∧ (1/2 * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 3 / 2) :=
sorry

end triangle_theorem_l3319_331945


namespace log_product_equality_l3319_331919

open Real

theorem log_product_equality (A m n p : ℝ) (hA : A > 0) (hm : m > 0) (hn : n > 0) (hp : p > 0) :
  (log A / log m) * (log A / log n) + (log A / log n) * (log A / log p) + (log A / log p) * (log A / log m) =
  (log (m * n * p) / log A) * (log A / log p) * (log A / log n) * (log A / log m) :=
by sorry

#check log_product_equality

end log_product_equality_l3319_331919


namespace even_function_composition_l3319_331991

def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

theorem even_function_composition (g : ℝ → ℝ) (h : is_even_function g) :
  is_even_function (g ∘ g) :=
by
  sorry

end even_function_composition_l3319_331991


namespace sum_equals_two_n_cubed_l3319_331927

/-- The sum of numbers in the nth group of positive integers -/
def A (n : ℕ+) : ℕ := sorry

/-- The difference between the latter and former number in the nth group of cubes of natural numbers -/
def B (n : ℕ+) : ℕ := sorry

/-- The theorem stating that A_n + B_n = 2n^3 for all positive integers n -/
theorem sum_equals_two_n_cubed (n : ℕ+) : A n + B n = 2 * n.val ^ 3 := by sorry

end sum_equals_two_n_cubed_l3319_331927


namespace tan_150_and_pythagorean_identity_l3319_331958

theorem tan_150_and_pythagorean_identity :
  (Real.tan (150 * π / 180) = -Real.sqrt 3 / 3) ∧
  (Real.sin (150 * π / 180))^2 + (Real.cos (150 * π / 180))^2 = 1 := by
  sorry

end tan_150_and_pythagorean_identity_l3319_331958


namespace xyz_value_l3319_331915

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) :
  x * y * z = 10 := by sorry

end xyz_value_l3319_331915


namespace linear_equation_power_l3319_331948

theorem linear_equation_power (n m : ℕ) :
  (∃ a b c : ℝ, ∀ x y : ℝ, a * x + b * y = c ↔ 2 * x^(n - 3) - (1/3) * y^(2*m + 1) = 0) →
  n^m = 1 := by
  sorry

end linear_equation_power_l3319_331948


namespace square_roots_of_2011_sum_l3319_331992

theorem square_roots_of_2011_sum (x y : ℝ) : 
  x^2 = 2011 → y^2 = 2011 → x + y = 0 := by
sorry

end square_roots_of_2011_sum_l3319_331992


namespace melanies_dimes_l3319_331917

theorem melanies_dimes (initial_dimes : ℕ) (mother_dimes : ℕ) (total_dimes : ℕ) (dad_dimes : ℕ) :
  initial_dimes = 7 →
  mother_dimes = 4 →
  total_dimes = 19 →
  total_dimes = initial_dimes + mother_dimes + dad_dimes →
  dad_dimes = 8 := by
sorry

end melanies_dimes_l3319_331917


namespace point_on_transformed_graph_l3319_331916

-- Define the function g
variable (g : ℝ → ℝ)

-- State the theorem
theorem point_on_transformed_graph (h : g 3 = 10) :
  ∃ (x y : ℝ), 3 * y = 4 * g (3 * x) + 6 ∧ x = 1 ∧ y = 46 / 3 ∧ x + y = 49 / 3 := by
  sorry

end point_on_transformed_graph_l3319_331916


namespace i_to_2016_l3319_331987

theorem i_to_2016 (i : ℂ) (h : i^2 = -1) : i^2016 = 1 := by
  sorry

end i_to_2016_l3319_331987


namespace intersection_with_complement_l3319_331975

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 ≤ x ∧ x < 3}

-- State the theorem
theorem intersection_with_complement :
  A ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

end intersection_with_complement_l3319_331975


namespace pure_imaginary_complex_l3319_331909

theorem pure_imaginary_complex (m : ℝ) : 
  (m * (m - 1) : ℂ) + m * Complex.I = (0 : ℝ) + (b : ℝ) * Complex.I ∧ b ≠ 0 → m = 1 := by
  sorry

end pure_imaginary_complex_l3319_331909


namespace inequality_proof_l3319_331938

theorem inequality_proof (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum_squares : a^2 + b^2 + c^2 = 3) : 
  1/(4-a^2) + 1/(4-b^2) + 1/(4-c^2) ≤ 9/((a+b+c)^2) := by
sorry

end inequality_proof_l3319_331938


namespace regular_polygon_with_150_degree_angles_l3319_331959

theorem regular_polygon_with_150_degree_angles (n : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : (n : ℝ) * 150 = 180 * (n - 2)) : n = 12 := by
  sorry

end regular_polygon_with_150_degree_angles_l3319_331959


namespace triangle_properties_l3319_331905

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The cosine rule for triangle ABC -/
axiom cosine_rule (t : Triangle) : t.a^2 + t.b^2 - t.c^2 = 2 * t.a * t.b * Real.cos t.C

/-- The area formula for triangle ABC -/
axiom area_formula (t : Triangle) (S : ℝ) : S = 1/2 * t.a * t.b * Real.sin t.C

theorem triangle_properties (t : Triangle) (S : ℝ) :
  (t.a^2 + t.b^2 - t.c^2 = t.a * t.b → t.C = π/3) ∧
  (t.a^2 + t.b^2 - t.c^2 = t.a * t.b ∧ t.c = Real.sqrt 7 ∧ S = 3 * Real.sqrt 3 / 2 → t.a + t.b = 5) :=
by sorry

end triangle_properties_l3319_331905


namespace bobbys_candy_consumption_l3319_331928

/-- Represents the number of candies Bobby takes during each of the remaining days of the week. -/
def candies_on_remaining_days (
  packets : ℕ)  -- Number of candy packets
  (candies_per_packet : ℕ)  -- Number of candies in each packet
  (candies_per_weekday : ℕ)  -- Number of candies eaten per weekday
  (weekdays : ℕ)  -- Number of weekdays per week
  (weeks : ℕ)  -- Number of weeks to finish all candies
  : ℕ :=
  let total_candies := packets * candies_per_packet
  let weekday_candies := candies_per_weekday * weekdays * weeks
  let remaining_candies := total_candies - weekday_candies
  let remaining_days := (7 - weekdays) * weeks
  remaining_candies / remaining_days

/-- Theorem stating that Bobby takes 1 candy during each of the remaining days of the week. -/
theorem bobbys_candy_consumption :
  candies_on_remaining_days 2 18 2 5 3 = 1 := by
  sorry

end bobbys_candy_consumption_l3319_331928


namespace only_36_satisfies_conditions_l3319_331969

/-- A two-digit integer is represented by 10a + b, where a and b are single digits -/
def TwoDigitInteger (a b : ℕ) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9

/-- The sum of digits of a two-digit integer -/
def SumOfDigits (a b : ℕ) : ℕ := a + b

/-- Twice the product of digits of a two-digit integer -/
def TwiceProductOfDigits (a b : ℕ) : ℕ := 2 * a * b

/-- The value of a two-digit integer -/
def IntegerValue (a b : ℕ) : ℕ := 10 * a + b

theorem only_36_satisfies_conditions :
  ∀ a b : ℕ,
    TwoDigitInteger a b →
    (IntegerValue a b % SumOfDigits a b = 0 ∧
     IntegerValue a b % TwiceProductOfDigits a b = 0) →
    IntegerValue a b = 36 :=
by sorry

end only_36_satisfies_conditions_l3319_331969


namespace car_speed_calculation_l3319_331972

/-- Calculates the speed of a car given distance and time -/
theorem car_speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 360 ∧ time = 4.5 → speed = distance / time → speed = 80 := by
  sorry

end car_speed_calculation_l3319_331972


namespace coat_drive_l3319_331952

theorem coat_drive (total_coats : ℕ) (high_school_coats : ℕ) (elementary_coats : ℕ) :
  total_coats = 9437 →
  high_school_coats = 6922 →
  elementary_coats = total_coats - high_school_coats →
  elementary_coats = 2515 := by
  sorry

end coat_drive_l3319_331952


namespace cannot_form_square_l3319_331918

/-- Represents the number of sticks of each length --/
structure Sticks :=
  (length1 : ℕ)
  (length2 : ℕ)
  (length3 : ℕ)
  (length4 : ℕ)

/-- Calculates the total length of all sticks --/
def totalLength (s : Sticks) : ℕ :=
  s.length1 * 1 + s.length2 * 2 + s.length3 * 3 + s.length4 * 4

/-- Represents the given set of sticks --/
def givenSticks : Sticks :=
  { length1 := 6
  , length2 := 3
  , length3 := 6
  , length4 := 5 }

/-- Theorem stating that it's impossible to form a square with the given sticks --/
theorem cannot_form_square (s : Sticks) (h : s = givenSticks) :
  ¬ ∃ (side : ℕ), side > 0 ∧ 4 * side = totalLength s :=
by sorry


end cannot_form_square_l3319_331918


namespace log_equation_solution_l3319_331904

theorem log_equation_solution (k x : ℝ) (h : k > 0) (h2 : x > 0) :
  (Real.log x / Real.log k) * (Real.log (k^2) / Real.log 5) = 3 →
  x = 5 * Real.sqrt 5 :=
sorry

end log_equation_solution_l3319_331904


namespace even_increasing_function_inequality_l3319_331913

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem even_increasing_function_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_incr : increasing_on f (Set.Ici 0)) : 
  f (-2) < f (-3) ∧ f (-3) < f π := by
  sorry

end even_increasing_function_inequality_l3319_331913


namespace intersection_implies_a_value_l3319_331961

/-- A point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Curve C₁ in polar coordinates -/
def C₁ (p : PolarPoint) : Prop :=
  p.ρ * (Real.sqrt 2 * Real.cos p.θ + Real.sin p.θ) = 1

/-- Curve C₂ in polar coordinates -/
def C₂ (a : ℝ) (p : PolarPoint) : Prop :=
  p.ρ = a

/-- A point is on the polar axis if its θ coordinate is 0 or π -/
def onPolarAxis (p : PolarPoint) : Prop :=
  p.θ = 0 ∨ p.θ = Real.pi

theorem intersection_implies_a_value (a : ℝ) (h_a_pos : a > 0) :
  (∃ p : PolarPoint, C₁ p ∧ C₂ a p ∧ onPolarAxis p) →
  a = Real.sqrt 2 / 2 := by
  sorry

end intersection_implies_a_value_l3319_331961


namespace committee_probability_l3319_331965

def total_members : ℕ := 30
def num_boys : ℕ := 12
def num_girls : ℕ := 18
def committee_size : ℕ := 6

def prob_at_least_one_boy_and_girl : ℚ :=
  1 - (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size) / Nat.choose total_members committee_size

theorem committee_probability :
  prob_at_least_one_boy_and_girl = 574287 / 593775 :=
sorry

end committee_probability_l3319_331965


namespace money_sum_l3319_331923

/-- Given two people a and b with some amount of money, 
    if 2/3 of a's amount equals 1/2 of b's amount, 
    and b has 484 rupees, then their total amount is 847 rupees. -/
theorem money_sum (a b : ℕ) (h1 : 2 * a = 3 * (b / 2)) (h2 : b = 484) : 
  a + b = 847 := by
  sorry

end money_sum_l3319_331923


namespace seating_theorem_l3319_331988

/-- Represents a taxi with 4 seats -/
structure Taxi :=
  (front_seat : Fin 1)
  (back_seats : Fin 3)

/-- Represents the number of window seats in a taxi -/
def window_seats : Nat := 2

/-- Represents the total number of passengers -/
def total_passengers : Nat := 4

/-- Calculates the number of seating arrangements in a taxi -/
def seating_arrangements (t : Taxi) (w : Nat) (p : Nat) : Nat :=
  w * (p - 1) * (p - 2) * (p - 3)

/-- Theorem stating that the number of seating arrangements is 12 -/
theorem seating_theorem (t : Taxi) :
  seating_arrangements t window_seats total_passengers = 12 := by
  sorry

end seating_theorem_l3319_331988


namespace total_questions_l3319_331935

/-- Represents the examination structure -/
structure Examination where
  typeA : Nat
  typeB : Nat
  totalTime : Nat
  typeATime : Nat

/-- The given examination parameters -/
def givenExam : Examination where
  typeA := 25
  typeB := 0  -- We don't know this value yet
  totalTime := 180  -- 3 hours * 60 minutes
  typeATime := 40

/-- Theorem stating the total number of questions in the examination -/
theorem total_questions (e : Examination) (h1 : e.typeA = givenExam.typeA)
    (h2 : e.totalTime = givenExam.totalTime) (h3 : e.typeATime = givenExam.typeATime)
    (h4 : 2 * (e.totalTime - e.typeATime) = 7 * e.typeATime) :
    e.typeA + e.typeB = 200 := by
  sorry

#check total_questions

end total_questions_l3319_331935


namespace simplify_expression_calculate_expression_l3319_331997

-- Part 1
theorem simplify_expression (x y : ℝ) :
  3 * (2 * x - y) - 2 * (4 * x + 1/2 * y) = -2 * x - 4 * y := by sorry

-- Part 2
theorem calculate_expression (x y : ℝ) (h1 : x * y = 4) (h2 : x - y = -7.5) :
  3 * (x * y - 2/3 * y) - 1/2 * (2 * x + 4 * x * y) - (-2 * x - y) = -7/2 := by sorry

end simplify_expression_calculate_expression_l3319_331997


namespace constant_term_is_60_l3319_331990

/-- The constant term in the expansion of (√x - 2/x)^6 -/
def constantTerm : ℕ :=
  -- We define the constant term without using the solution steps
  -- This definition should be completed in the proof
  sorry

/-- Proof that the constant term in the expansion of (√x - 2/x)^6 is 60 -/
theorem constant_term_is_60 : constantTerm = 60 := by
  sorry

end constant_term_is_60_l3319_331990


namespace circle_sum_puzzle_l3319_331922

/-- A solution is a 6-tuple of natural numbers representing the values in circles A, B, C, D, E, F --/
def Solution := (Nat × Nat × Nat × Nat × Nat × Nat)

/-- Check if a solution satisfies all conditions --/
def is_valid_solution (s : Solution) : Prop :=
  let (a, b, c, d, e, f) := s
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  b + c + a = 22 ∧
  d + c + f = 11 ∧
  e + b + d = 19 ∧
  a + e + c = 22

theorem circle_sum_puzzle :
  ∃! (s1 s2 : Solution),
    is_valid_solution s1 ∧
    is_valid_solution s2 ∧
    (∀ s, is_valid_solution s → (s = s1 ∨ s = s2)) :=
sorry

end circle_sum_puzzle_l3319_331922


namespace circle_area_when_eight_times_reciprocal_circumference_equals_diameter_l3319_331924

theorem circle_area_when_eight_times_reciprocal_circumference_equals_diameter :
  ∀ (r : ℝ), r > 0 → (8 * (1 / (2 * Real.pi * r)) = 2 * r) → (Real.pi * r^2 = 2) :=
by sorry

end circle_area_when_eight_times_reciprocal_circumference_equals_diameter_l3319_331924


namespace algebraic_expression_symmetry_l3319_331911

theorem algebraic_expression_symmetry (a b c : ℝ) : 
  a * (-5)^4 + b * (-5)^2 + c = 3 → a * 5^4 + b * 5^2 + c = 3 := by
  sorry

end algebraic_expression_symmetry_l3319_331911


namespace ratio_percentage_difference_l3319_331964

theorem ratio_percentage_difference (A B : ℝ) (hA : A > 0) (hB : B > 0) (h_ratio : A / B = 1/8 * 7) :
  (B - A) / A * 100 = 100/7 := by
sorry

end ratio_percentage_difference_l3319_331964


namespace same_constant_term_similar_structure_l3319_331914

-- Define a polynomial with distinct positive real coefficients
def P (x : ℝ) : ℝ := sorry

-- Define the median of the coefficients of P
def median_coeff : ℝ := sorry

-- Define Q using the median of coefficients of P
def Q (x : ℝ) : ℝ := sorry

-- Theorem stating that P and Q have the same constant term
theorem same_constant_term : P 0 = Q 0 := by sorry

-- Theorem stating that P and Q have similar structure
-- (We can't precisely define "similar structure" without more information,
-- so we'll use a placeholder property)
theorem similar_structure : ∃ (k : ℝ), k > 0 ∧ ∀ x, |P x - Q x| ≤ k := by sorry

end same_constant_term_similar_structure_l3319_331914
