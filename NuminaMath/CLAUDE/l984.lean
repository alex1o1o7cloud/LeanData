import Mathlib

namespace two_trains_meeting_time_l984_98429

/-- Two trains problem -/
theorem two_trains_meeting_time 
  (distance : ℝ) 
  (fast_speed slow_speed : ℝ) 
  (head_start : ℝ) 
  (h_distance : distance = 270) 
  (h_fast_speed : fast_speed = 120) 
  (h_slow_speed : slow_speed = 75) 
  (h_head_start : head_start = 1) :
  ∃ x : ℝ, slow_speed * head_start + (fast_speed + slow_speed) * x = distance :=
by sorry

end two_trains_meeting_time_l984_98429


namespace intersection_parallel_line_l984_98457

/-- The equation of a line passing through the intersection of two given lines and parallel to a third line -/
theorem intersection_parallel_line (x y : ℝ) : 
  (2 * x - 3 * y + 2 = 0) →  -- l₁
  (3 * x - 4 * y - 2 = 0) →  -- l₂
  ∃ (k : ℝ), (4 * x - 2 * y + k = 0) ∧  -- parallel line
  (2 * x - y - 18 = 0) :=  -- result
by sorry

end intersection_parallel_line_l984_98457


namespace quadratic_inequality_solution_l984_98402

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a + b = -14 := by
sorry

end quadratic_inequality_solution_l984_98402


namespace expression_value_l984_98414

theorem expression_value : (3^2 - 2 * 3) - (5^2 - 2 * 5) + (7^2 - 2 * 7) = 23 := by
  sorry

end expression_value_l984_98414


namespace triangle_properties_l984_98458

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  2 * t.a * Real.cos t.A = t.b * Real.cos t.C + t.c * Real.cos t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : given_condition t) : 
  t.A = π / 3 ∧ 
  ∀ x, x ∈ Set.Icc (-1 : ℝ) (-1/2) ↔ 
    ∃ (B C : ℝ), t.B = B ∧ t.C = C ∧ x = Real.cos B - Real.sqrt 3 * Real.sin C :=
sorry


end triangle_properties_l984_98458


namespace rectangle_enumeration_l984_98425

/-- Represents a rectangle in the Cartesian plane with sides parallel to the axes. -/
structure Rectangle where
  x_min : ℝ
  y_min : ℝ
  x_max : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- Defines when one rectangle is below another. -/
def is_below (r1 r2 : Rectangle) : Prop :=
  r1.y_max < r2.y_min

/-- Defines when one rectangle is to the right of another. -/
def is_right_of (r1 r2 : Rectangle) : Prop :=
  r1.x_min > r2.x_max

/-- Defines when two rectangles are disjoint. -/
def are_disjoint (r1 r2 : Rectangle) : Prop :=
  r1.x_max ≤ r2.x_min ∨ r2.x_max ≤ r1.x_min ∨
  r1.y_max ≤ r2.y_min ∨ r2.y_max ≤ r1.y_min

/-- The main theorem stating that any finite set of pairwise disjoint rectangles
    can be enumerated such that each rectangle is to the right of or below all
    subsequent rectangles in the enumeration. -/
theorem rectangle_enumeration (n : ℕ) (rectangles : Fin n → Rectangle)
    (h_disjoint : ∀ i j : Fin n, i ≠ j → are_disjoint (rectangles i) (rectangles j)) :
    ∃ σ : Equiv.Perm (Fin n),
      ∀ i j : Fin n, i < j →
        is_right_of (rectangles (σ i)) (rectangles (σ j)) ∨
        is_below (rectangles (σ i)) (rectangles (σ j)) :=
  sorry

end rectangle_enumeration_l984_98425


namespace angle_with_supplement_four_times_complement_l984_98472

theorem angle_with_supplement_four_times_complement : ∃ (x : ℝ), 
  x = 60 ∧ 
  (180 - x) = 4 * (90 - x) := by
  sorry

end angle_with_supplement_four_times_complement_l984_98472


namespace quadratic_root_form_l984_98401

theorem quadratic_root_form (d : ℝ) : 
  (∀ x : ℝ, x^2 + 6*x + d = 0 ↔ x = (-6 + Real.sqrt d) / 2 ∨ x = (-6 - Real.sqrt d) / 2) →
  d = 36 / 5 := by
sorry

end quadratic_root_form_l984_98401


namespace inequality_and_factorial_l984_98410

theorem inequality_and_factorial (n : ℕ) : 2 ≤ (1 + 1 / n : ℝ) ^ n ∧ (1 + 1 / n : ℝ) ^ n < 3 ∧ (n / 3 : ℝ) ^ n < n! := by
  sorry

end inequality_and_factorial_l984_98410


namespace glenn_total_expenditure_l984_98449

/-- Represents the cost of movie tickets and concessions -/
structure MovieCosts where
  monday_ticket : ℕ
  wednesday_ticket : ℕ
  saturday_ticket : ℕ
  concession : ℕ

/-- Represents discount percentages -/
structure Discounts where
  wednesday : ℕ
  group : ℕ

/-- Represents the number of people in Glenn's group for each day -/
structure GroupSize where
  wednesday : ℕ
  saturday : ℕ

/-- Calculates the total cost of Glenn's movie outings -/
def calculate_total_cost (costs : MovieCosts) (discounts : Discounts) (group : GroupSize) : ℕ :=
  let wednesday_cost := costs.wednesday_ticket * (100 - discounts.wednesday) / 100 * group.wednesday
  let saturday_cost := costs.saturday_ticket * group.saturday + costs.concession
  wednesday_cost + saturday_cost

/-- Theorem stating that Glenn's total expenditure is $93 -/
theorem glenn_total_expenditure (costs : MovieCosts) (discounts : Discounts) (group : GroupSize) :
  costs.monday_ticket = 5 →
  costs.wednesday_ticket = 2 * costs.monday_ticket →
  costs.saturday_ticket = 5 * costs.monday_ticket →
  costs.concession = 7 →
  discounts.wednesday = 10 →
  discounts.group = 20 →
  group.wednesday = 4 →
  group.saturday = 2 →
  calculate_total_cost costs discounts group = 93 := by
  sorry


end glenn_total_expenditure_l984_98449


namespace initial_pencils_count_l984_98446

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := sorry

/-- The number of pencils Tim added to the drawer -/
def pencils_added : ℕ := 3

/-- The total number of pencils after Tim's addition -/
def total_pencils : ℕ := 5

theorem initial_pencils_count : initial_pencils = 2 :=
  by sorry

end initial_pencils_count_l984_98446


namespace work_completion_time_l984_98432

/-- 
Given that:
- A does 20% less work than B
- A completes the work in 15/2 hours
Prove that B will complete the work in 6 hours
-/
theorem work_completion_time (work_rate_A work_rate_B : ℝ) 
  (h1 : work_rate_A = 0.8 * work_rate_B) 
  (h2 : work_rate_A * (15/2) = 1) : 
  work_rate_B * 6 = 1 := by
  sorry

end work_completion_time_l984_98432


namespace not_even_and_composite_two_l984_98441

/-- Definition of an even number -/
def IsEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- Definition of a composite number -/
def IsComposite (n : ℕ) : Prop := ∃ a b : ℕ, a ≠ 1 ∧ b ≠ 1 ∧ n = a * b

/-- Theorem: It is false that 2 is both an even number and a composite number -/
theorem not_even_and_composite_two : ¬(IsEven 2 ∧ IsComposite 2) := by
  sorry

end not_even_and_composite_two_l984_98441


namespace minimum_gloves_needed_l984_98493

theorem minimum_gloves_needed (participants : ℕ) (gloves_per_participant : ℕ) : 
  participants = 43 → gloves_per_participant = 2 → participants * gloves_per_participant = 86 := by
  sorry

end minimum_gloves_needed_l984_98493


namespace cubic_root_sum_l984_98431

theorem cubic_root_sum (a b : ℝ) : 
  (Complex.I + 2 : ℂ) ^ 3 + a * (Complex.I + 2) + b = 0 → a + b = 9 := by
  sorry

end cubic_root_sum_l984_98431


namespace population_average_age_l984_98418

theorem population_average_age
  (ratio_women_men : ℚ)
  (avg_age_women : ℚ)
  (avg_age_men : ℚ)
  (h_ratio : ratio_women_men = 10 / 9)
  (h_women_age : avg_age_women = 36)
  (h_men_age : avg_age_men = 33) :
  (ratio_women_men * avg_age_women + avg_age_men) / (ratio_women_men + 1) = 34 + 13 / 19 :=
by sorry

end population_average_age_l984_98418


namespace all_propositions_false_l984_98475

-- Define the correlation coefficient
def correlation_coefficient : ℝ → ℝ := sorry

-- Define the degree of linear correlation
def linear_correlation_degree : ℝ → ℝ := sorry

-- Define the cubic function
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

-- Define what it means for f to have an extreme value at x = -1
def has_extreme_value_at_neg_one (a b : ℝ) : Prop :=
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x + 1| ∧ |x + 1| < ε →
    (f a b (-1) - f a b x) * (f a b (-1) - f a b (-1 - (x + 1))) > 0

theorem all_propositions_false :
  -- Proposition 1
  (∀ r₁ r₂ : ℝ, |r₁| < |r₂| → linear_correlation_degree r₁ < linear_correlation_degree r₂) ∧
  -- Proposition 2
  (¬(∃ x : ℝ, x^2 + x + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 > 0) ∧
  -- Proposition 3
  (∀ p q : Prop, (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) ∧
  -- Proposition 4
  (∀ a b : ℝ, has_extreme_value_at_neg_one a b → (a = 1 ∧ b = 9))
  → False := by sorry

end all_propositions_false_l984_98475


namespace remaining_student_l984_98440

theorem remaining_student (n : ℕ) (hn : n ≤ 2002) : n % 1331 = 0 ↔ n = 1331 :=
by sorry

#check remaining_student

end remaining_student_l984_98440


namespace complex_equation_solution_l984_98447

/-- Given a complex number z and a real number a, if |z| = 2 and (z - a)² = a, then a = 2 -/
theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : Complex.abs z = 2) 
  (h2 : (z - a)^2 = a) : 
  a = 2 := by
  sorry

end complex_equation_solution_l984_98447


namespace max_distance_for_given_car_l984_98404

/-- Represents a car with front and rear tires that can be switched --/
structure Car where
  frontTireLife : ℕ
  rearTireLife : ℕ

/-- Calculates the maximum distance a car can travel by switching tires once --/
def maxDistanceWithSwitch (car : Car) : ℕ :=
  let switchPoint := min car.frontTireLife car.rearTireLife / 2
  switchPoint + (car.frontTireLife - switchPoint) + (car.rearTireLife - switchPoint)

/-- Theorem stating the maximum distance for the given car specifications --/
theorem max_distance_for_given_car :
  let car := { frontTireLife := 24000, rearTireLife := 36000 : Car }
  maxDistanceWithSwitch car = 48000 := by
  sorry

#eval maxDistanceWithSwitch { frontTireLife := 24000, rearTireLife := 36000 }

end max_distance_for_given_car_l984_98404


namespace teacher_age_proof_l984_98430

def teacher_age (num_students : ℕ) (student_avg_age : ℕ) (new_avg_age : ℕ) (total_people : ℕ) : ℕ :=
  (new_avg_age * total_people) - (student_avg_age * num_students)

theorem teacher_age_proof :
  teacher_age 23 22 23 24 = 46 := by
  sorry

end teacher_age_proof_l984_98430


namespace sqrt_inequality_at_least_one_positive_l984_98484

-- Problem 1
theorem sqrt_inequality (a : ℝ) (h : a > 0) :
  Real.sqrt (a + 5) - Real.sqrt (a + 3) > Real.sqrt (a + 6) - Real.sqrt (a + 4) := by
  sorry

-- Problem 2
theorem at_least_one_positive (x y z : ℝ) :
  let a := x^2 - 2*y + π/2
  let b := y^2 - 2*z + π/3
  let c := z^2 - 2*x + π/6
  max a (max b c) > 0 := by
  sorry

end sqrt_inequality_at_least_one_positive_l984_98484


namespace absolute_value_plus_exponent_l984_98483

theorem absolute_value_plus_exponent : |-4| + (3 - Real.pi)^0 = 5 := by sorry

end absolute_value_plus_exponent_l984_98483


namespace decreasing_interval_of_quadratic_b_range_for_decreasing_l984_98497

/-- A quadratic function f(x) = ax^2 + bx -/
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := 2 * a * x + b

theorem decreasing_interval_of_quadratic (a b : ℝ) :
  (f_derivative a b 3 = 24) →  -- Tangent at x=3 is parallel to 24x-y+1=0
  (f_derivative a b 1 = 0) →   -- Extreme value at x=1
  ∀ x > 1, f_derivative a b x < 0 := by sorry

theorem b_range_for_decreasing (b : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f_derivative 1 b x ≤ 0) →
  b ≤ -2 := by sorry

end decreasing_interval_of_quadratic_b_range_for_decreasing_l984_98497


namespace third_digit_is_one_l984_98477

/-- A self-descriptive 7-digit number -/
structure SelfDescriptiveNumber where
  digits : Fin 7 → Fin 7
  sum_is_seven : (Finset.sum Finset.univ (λ i => digits i)) = 7
  first_digit : digits 0 = 3
  second_digit : digits 1 = 2
  fourth_digit : digits 3 = 1
  fifth_digit : digits 4 = 0

/-- The third digit of a self-descriptive number is 1 -/
theorem third_digit_is_one (n : SelfDescriptiveNumber) : n.digits 2 = 1 := by
  sorry

end third_digit_is_one_l984_98477


namespace speed_train_B_is_25_l984_98421

/-- Represents the distance between two stations in kilometers -/
def distance_between_stations : ℝ := 155

/-- Represents the speed of the train from station A in km/h -/
def speed_train_A : ℝ := 20

/-- Represents the time difference between the starts of the two trains in hours -/
def time_difference : ℝ := 1

/-- Represents the total time until the trains meet in hours -/
def total_time : ℝ := 4

/-- Represents the time the train from B travels in hours -/
def time_train_B : ℝ := 3

/-- Theorem stating that the speed of the train from station B is 25 km/h -/
theorem speed_train_B_is_25 : 
  ∃ (speed_B : ℝ), 
    speed_B * time_train_B = distance_between_stations - speed_train_A * total_time ∧ 
    speed_B = 25 := by
  sorry

end speed_train_B_is_25_l984_98421


namespace distinct_configurations_correct_l984_98403

/-- Represents the number of distinct configurations of n coins arranged in a circle
    that cannot be transformed into one another by flipping adjacent pairs of coins
    with the same orientation. -/
def distinctConfigurations (n : ℕ) : ℕ :=
  if n % 2 = 0 then n + 1 else 2

theorem distinct_configurations_correct (n : ℕ) :
  distinctConfigurations n = if n % 2 = 0 then n + 1 else 2 := by
  sorry

end distinct_configurations_correct_l984_98403


namespace sum_lent_is_300_l984_98434

/-- Proves that the sum lent is 300, given the conditions of the problem -/
theorem sum_lent_is_300 
  (interest_rate : ℝ) 
  (loan_duration : ℕ) 
  (interest_difference : ℝ) 
  (h1 : interest_rate = 0.04)
  (h2 : loan_duration = 8)
  (h3 : interest_difference = 204) :
  ∃ (principal : ℝ), 
    principal * interest_rate * loan_duration = principal - interest_difference ∧ 
    principal = 300 := by
sorry


end sum_lent_is_300_l984_98434


namespace lee_apple_harvest_l984_98406

/-- The number of baskets Mr. Lee used to pack apples -/
def num_baskets : ℕ := 19

/-- The number of apples in each basket -/
def apples_per_basket : ℕ := 25

/-- The total number of apples harvested by Mr. Lee -/
def total_apples : ℕ := num_baskets * apples_per_basket

theorem lee_apple_harvest : total_apples = 475 := by
  sorry

end lee_apple_harvest_l984_98406


namespace bethany_age_proof_l984_98423

/-- Bethany's current age -/
def bethanys_current_age : ℕ := 19

/-- Bethany's younger sister's current age -/
def sisters_current_age : ℕ := 11

/-- Bethany's age three years ago -/
def bethanys_age_three_years_ago : ℕ := bethanys_current_age - 3

/-- Bethany's younger sister's age three years ago -/
def sisters_age_three_years_ago : ℕ := sisters_current_age - 3

theorem bethany_age_proof :
  (bethanys_age_three_years_ago = 2 * sisters_age_three_years_ago) ∧
  (sisters_current_age + 5 = 16) →
  bethanys_current_age = 19 := by
  sorry

end bethany_age_proof_l984_98423


namespace solve_inequality_find_a_range_l984_98481

-- Define the functions f and g
def f (x a : ℝ) : ℝ := |x - 1| + |x + a|
def g (a : ℝ) : ℝ := a^2 - a - 2

-- Theorem for the first part of the problem
theorem solve_inequality (x : ℝ) :
  f x 3 > g 3 + 2 ↔ x < -4 ∨ x > 2 := by sorry

-- Theorem for the second part of the problem
theorem find_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc (-a) 1, f x a ≤ g a) → a ≥ 3 := by sorry

end solve_inequality_find_a_range_l984_98481


namespace min_value_expression_l984_98455

theorem min_value_expression (x y z : ℝ) 
  (hx : -1/2 ≤ x ∧ x ≤ 1/2) 
  (hy : -1/2 ≤ y ∧ y ≤ 1/2) 
  (hz : -1/2 ≤ z ∧ z ≤ 1/2) : 
  (1/((1 - x^2)*(1 - y^2)*(1 - z^2))) + (1/((1 + x^2)*(1 + y^2)*(1 + z^2))) ≥ 2 ∧
  (1/((1 - 0^2)*(1 - 0^2)*(1 - 0^2))) + (1/((1 + 0^2)*(1 + 0^2)*(1 + 0^2))) = 2 :=
by sorry

end min_value_expression_l984_98455


namespace sin_cos_fourth_power_difference_l984_98492

theorem sin_cos_fourth_power_difference (α : ℝ) :
  Real.sin (π / 2 - 2 * α) = 3 / 5 →
  Real.sin α ^ 4 - Real.cos α ^ 4 = -(3 / 5) := by
  sorry

end sin_cos_fourth_power_difference_l984_98492


namespace teds_age_l984_98487

theorem teds_age (s : ℝ) (t : ℝ) (a : ℝ) 
  (h1 : t = 2 * s + 17)
  (h2 : a = s / 2)
  (h3 : t + s + a = 72) : 
  ⌊t⌋ = 48 := by
sorry

end teds_age_l984_98487


namespace chess_tournament_participants_l984_98485

theorem chess_tournament_participants : ∃ n : ℕ, 
  n > 0 ∧ 
  (n * (n - 1)) / 2 = 190 ∧ 
  n = 20 := by
sorry

end chess_tournament_participants_l984_98485


namespace trapezoid_area_l984_98454

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a trapezoid PQRS with diagonals intersecting at T -/
structure Trapezoid :=
  (P Q R S T : Point)

/-- Calculates the area of a triangle given three points -/
def triangleArea (A B C : Point) : ℝ := sorry

/-- Checks if two line segments are parallel -/
def isParallel (A B C D : Point) : Prop := sorry

theorem trapezoid_area (PQRS : Trapezoid) : 
  isParallel PQRS.P PQRS.Q PQRS.R PQRS.S →
  triangleArea PQRS.P PQRS.Q PQRS.T = 40 →
  triangleArea PQRS.P PQRS.R PQRS.T = 25 →
  triangleArea PQRS.P PQRS.Q PQRS.R + 
  triangleArea PQRS.P PQRS.R PQRS.S + 
  triangleArea PQRS.P PQRS.S PQRS.Q = 105.625 := by
  sorry

end trapezoid_area_l984_98454


namespace hawks_percentage_l984_98479

/-- Represents the percentages of different bird types in a nature reserve -/
structure BirdReserve where
  hawks : ℝ
  paddyfieldWarblers : ℝ
  kingfishers : ℝ
  others : ℝ

/-- The conditions of the bird reserve problem -/
def validBirdReserve (b : BirdReserve) : Prop :=
  b.paddyfieldWarblers = 0.4 * (100 - b.hawks) ∧
  b.kingfishers = 0.25 * b.paddyfieldWarblers ∧
  b.others = 35 ∧
  b.hawks + b.paddyfieldWarblers + b.kingfishers + b.others = 100

/-- The theorem stating that hawks make up 30% of the birds in a valid bird reserve -/
theorem hawks_percentage (b : BirdReserve) (h : validBirdReserve b) : b.hawks = 30 := by
  sorry

end hawks_percentage_l984_98479


namespace min_value_problem_l984_98486

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  (4*x/(x-1)) + (9*y/(y-1)) ≥ 25 ∧ ∃ (x₀ y₀ : ℝ), (4*x₀/(x₀-1)) + (9*y₀/(y₀-1)) = 25 := by
  sorry

end min_value_problem_l984_98486


namespace factorization_equality_l984_98456

theorem factorization_equality (x y : ℝ) : (x + 2) * (x - 2) - 4 * y * (x - y) = (x - 2*y + 2) * (x - 2*y - 2) := by
  sorry

end factorization_equality_l984_98456


namespace smallest_number_divisible_by_multiple_l984_98461

theorem smallest_number_divisible_by_multiple (x : ℕ) : x = 34 ↔ 
  (∀ y : ℕ, y < x → ¬(∃ k : ℕ, y - 10 = 2 * k ∧ y - 10 = 6 * k ∧ y - 10 = 12 * k ∧ y - 10 = 24 * k)) ∧
  (∃ k : ℕ, x - 10 = 2 * k ∧ x - 10 = 6 * k ∧ x - 10 = 12 * k ∧ x - 10 = 24 * k) :=
by sorry

#check smallest_number_divisible_by_multiple

end smallest_number_divisible_by_multiple_l984_98461


namespace sin_1_lt_log_3_sqrt_7_l984_98465

theorem sin_1_lt_log_3_sqrt_7 :
  ∀ (sin : ℝ → ℝ) (log : ℝ → ℝ → ℝ),
  (0 < 1 ∧ 1 < π/3 ∧ π/3 < π/2) →
  sin (π/3) = Real.sqrt 3 / 2 →
  3^7 < 7^4 →
  sin 1 < log 3 (Real.sqrt 7) :=
by sorry

end sin_1_lt_log_3_sqrt_7_l984_98465


namespace probability_of_specific_sequence_l984_98427

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Number of Jacks in a standard deck -/
def NumJacks : ℕ := 4

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of Queens in a standard deck -/
def NumQueens : ℕ := 4

/-- Calculates the probability of drawing the specified sequence of cards -/
def probability_of_sequence : ℚ :=
  (NumKings : ℚ) / StandardDeck *
  (NumHearts - 1) / (StandardDeck - 1) *
  NumJacks / (StandardDeck - 2) *
  (NumSpades - 1) / (StandardDeck - 3) *
  NumQueens / (StandardDeck - 4)

theorem probability_of_specific_sequence :
  probability_of_sequence = 3 / 10125 := by sorry

end probability_of_specific_sequence_l984_98427


namespace expand_expression_l984_98442

theorem expand_expression (x : ℝ) : (1 + x^2) * (1 - x^4) = 1 + x^2 - x^4 - x^6 := by
  sorry

end expand_expression_l984_98442


namespace jason_toy_count_l984_98400

/-- The number of toys each person has -/
structure ToyCount where
  rachel : ℝ
  john : ℝ
  jason : ℝ

/-- The conditions of the problem -/
def toy_problem (t : ToyCount) : Prop :=
  t.rachel = 1 ∧
  t.john = t.rachel + 6.5 ∧
  t.jason = 3 * t.john

/-- Theorem stating that Jason has 22.5 toys -/
theorem jason_toy_count (t : ToyCount) (h : toy_problem t) : t.jason = 22.5 := by
  sorry

end jason_toy_count_l984_98400


namespace sum_modulo_thirteen_l984_98459

theorem sum_modulo_thirteen : (9245 + 9246 + 9247 + 9248 + 9249 + 9250) % 13 = 1 := by
  sorry

end sum_modulo_thirteen_l984_98459


namespace inequality_proof_l984_98409

theorem inequality_proof (a b : ℝ) (n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n-1) := by
  sorry

end inequality_proof_l984_98409


namespace hansel_salary_l984_98460

theorem hansel_salary (hansel_initial : ℝ) (gretel_initial : ℝ) :
  hansel_initial = gretel_initial →
  hansel_initial * 1.10 + 1500 = gretel_initial * 1.15 →
  hansel_initial = 30000 := by
  sorry

end hansel_salary_l984_98460


namespace harmonious_number_properties_l984_98462

/-- A harmonious number is a three-digit number where the tens digit 
    is equal to the sum of its units digit and hundreds digit. -/
def is_harmonious (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ 
  (n / 10 % 10 = n % 10 + n / 100)

/-- The smallest harmonious number -/
def smallest_harmonious : ℕ := 110

/-- The largest harmonious number -/
def largest_harmonious : ℕ := 990

/-- Algebraic expression for a harmonious number -/
def harmonious_expression (a b : ℕ) : ℕ := 110 * b - 99 * a

theorem harmonious_number_properties :
  (∀ n : ℕ, is_harmonious n → smallest_harmonious ≤ n ∧ n ≤ largest_harmonious) ∧
  (∀ n : ℕ, is_harmonious n → 
    ∃ a b : ℕ, a ≥ 0 ∧ b ≥ 1 ∧ b > a ∧ 
    n = harmonious_expression a b) :=
sorry

end harmonious_number_properties_l984_98462


namespace quadratic_function_properties_quadratic_function_max_value_l984_98419

/-- A quadratic function f(x) = ax^2 + bx + c where the solution set of f(x) > -2x is {x | 1 < x < 3} -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_properties (a b c : ℝ) 
  (h_solution_set : ∀ x, 1 < x ∧ x < 3 ↔ QuadraticFunction a b c x > -2 * x) :
  (∃ x₀ > 0, QuadraticFunction a b c x₀ = 2 * a ∧ 
   ∀ x, QuadraticFunction a b c x = 2 * a → x = x₀) →
  QuadraticFunction a b c = fun x ↦ -x^2 + 2*x - 3 :=
sorry

theorem quadratic_function_max_value (a b c : ℝ) 
  (h_solution_set : ∀ x, 1 < x ∧ x < 3 ↔ QuadraticFunction a b c x > -2 * x) :
  (∃ x₀, ∀ x, QuadraticFunction a b c x ≤ QuadraticFunction a b c x₀ ∧ 
   QuadraticFunction a b c x₀ > 0) →
  (-2 - Real.sqrt 3 < a ∧ a < 0) ∨ (-2 + Real.sqrt 3 < a ∧ a < 0) :=
sorry

end quadratic_function_properties_quadratic_function_max_value_l984_98419


namespace stratified_sampling_theorem_l984_98464

/-- Represents the number of exam papers checked in a school -/
structure SchoolSample where
  total : ℕ
  sampled : ℕ

/-- Calculates the total number of exam papers checked across all schools -/
def totalSampled (schools : List SchoolSample) : ℕ :=
  schools.map (fun s => s.sampled) |>.sum

theorem stratified_sampling_theorem (schoolA schoolB schoolC : SchoolSample) :
  schoolA.total = 1260 →
  schoolB.total = 720 →
  schoolC.total = 900 →
  schoolC.sampled = 45 →
  schoolA.sampled = schoolA.total / (schoolC.total / schoolC.sampled) →
  schoolB.sampled = schoolB.total / (schoolC.total / schoolC.sampled) →
  totalSampled [schoolA, schoolB, schoolC] = 144 := by
  sorry

#check stratified_sampling_theorem

end stratified_sampling_theorem_l984_98464


namespace technician_count_l984_98439

/-- Proves the number of technicians in a workshop given specific salary and worker information --/
theorem technician_count (total_workers : ℕ) (avg_salary_all : ℚ) (avg_salary_tech : ℚ) (avg_salary_non_tech : ℚ) 
  (h1 : total_workers = 12)
  (h2 : avg_salary_all = 9500)
  (h3 : avg_salary_tech = 12000)
  (h4 : avg_salary_non_tech = 6000) :
  ∃ (tech_count : ℕ), tech_count = 7 ∧ tech_count ≤ total_workers :=
by sorry

end technician_count_l984_98439


namespace painted_cube_problem_l984_98499

theorem painted_cube_problem (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 4 → n = 4 := by
  sorry

end painted_cube_problem_l984_98499


namespace sign_sum_zero_l984_98473

theorem sign_sum_zero (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_sum : a + b + c = 0) :
  a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| = 0 := by
  sorry

end sign_sum_zero_l984_98473


namespace min_sum_with_reciprocal_constraint_l984_98478

theorem min_sum_with_reciprocal_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 1 / a + 2 / b = 2) : 
  a + b ≥ (3 + 2 * Real.sqrt 2) / 2 ∧ 
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 1 / a₀ + 2 / b₀ = 2 ∧ a₀ + b₀ = (3 + 2 * Real.sqrt 2) / 2 :=
by sorry

end min_sum_with_reciprocal_constraint_l984_98478


namespace function_product_l984_98428

theorem function_product (f : ℕ → ℝ) 
  (h₁ : ∀ n : ℕ, n > 0 → f (n + 3) = (f n - 1) / (f n + 1))
  (h₂ : f 1 ≠ 0)
  (h₃ : f 1 ≠ 1 ∧ f 1 ≠ -1) :
  f 8 * f 2018 = -1 := by
  sorry

end function_product_l984_98428


namespace negation_of_absolute_value_statement_l984_98415

theorem negation_of_absolute_value_statement (x : ℝ) :
  ¬(abs x ≤ 3 ∨ abs x > 5) ↔ (abs x > 3 ∧ abs x ≤ 5) := by
  sorry

end negation_of_absolute_value_statement_l984_98415


namespace composition_ratio_l984_98435

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x - 1
def g (x : ℝ) : ℝ := 2 * x + 5

-- State the theorem
theorem composition_ratio :
  (g (f (g 3))) / (f (g (f 3))) = 69 / 206 := by
  sorry

end composition_ratio_l984_98435


namespace lcm_18_30_l984_98437

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l984_98437


namespace third_month_sale_l984_98405

def sales_problem (m1 m2 m4 m5 m6 average : ℕ) : Prop :=
  ∃ m3 : ℕ,
    m3 = 6 * average - (m1 + m2 + m4 + m5 + m6) ∧
    (m1 + m2 + m3 + m4 + m5 + m6) / 6 = average

theorem third_month_sale :
  sales_problem 5420 5660 6350 6500 8270 6400 →
  ∃ m3 : ℕ, m3 = 6200
:= by sorry

end third_month_sale_l984_98405


namespace quadratic_real_equal_roots_l984_98470

/-- 
For a quadratic equation of the form 3x^2 + 6kx + 9 = 0, 
the roots are real and equal if and only if k = ± √3.
-/
theorem quadratic_real_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + 6 * k * x + 9 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 + 6 * k * y + 9 = 0 → y = x) ↔ 
  k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
by sorry

end quadratic_real_equal_roots_l984_98470


namespace second_number_proof_l984_98420

theorem second_number_proof : ∃ x : ℕ, 
  (1657 % 1 = 10) ∧ 
  (x % 1 = 7) ∧ 
  (∀ y : ℕ, y > x → ¬(y % 1 = 7)) ∧ 
  (x = 1655) := by
sorry

end second_number_proof_l984_98420


namespace bella_steps_to_meet_l984_98463

/-- The number of steps Bella takes when meeting Ella -/
def steps_to_meet (distance : ℕ) (speed_ratio : ℕ) (step_length : ℕ) : ℕ :=
  (distance * 2) / ((speed_ratio + 1) * step_length)

/-- Theorem stating that Bella takes 1056 steps to meet Ella under given conditions -/
theorem bella_steps_to_meet :
  steps_to_meet 15840 4 3 = 1056 :=
by sorry

end bella_steps_to_meet_l984_98463


namespace solve_equation_l984_98490

theorem solve_equation (y : ℚ) (h : (2 / 7) * (1 / 5) * y = 4) : y = 70 := by
  sorry

end solve_equation_l984_98490


namespace snow_leopard_arrangement_l984_98476

theorem snow_leopard_arrangement (n : ℕ) (h : n = 9) : 
  (2 : ℕ) * (Nat.factorial (n - 3)) = 1440 := by
  sorry

end snow_leopard_arrangement_l984_98476


namespace inverse_of_M_l984_98489

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, -1]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![4, 1; 2, 3]
def M : Matrix (Fin 2) (Fin 2) ℝ := B * A

theorem inverse_of_M : 
  M⁻¹ = !![3/10, -1/10; 1/5, -2/5] :=
sorry

end inverse_of_M_l984_98489


namespace geometric_sequence_common_ratio_l984_98413

/-- Given a geometric sequence {a_n} with common ratio q, 
    if a_1 + a_4 = 10 and a_2 + a_5 = 20, then q = 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = q * a n) 
  (h_sum1 : a 1 + a 4 = 10) 
  (h_sum2 : a 2 + a 5 = 20) : 
  q = 2 := by
sorry

end geometric_sequence_common_ratio_l984_98413


namespace intersection_point_is_correct_l984_98468

-- Define the slope of the first line
def m₁ : ℚ := 2

-- Define the first line: y = 2x + 3
def line₁ (x y : ℚ) : Prop := y = m₁ * x + 3

-- Define the slope of the perpendicular line
def m₂ : ℚ := -1 / m₁

-- Define the point that the perpendicular line passes through
def point : ℚ × ℚ := (3, 8)

-- Define the perpendicular line passing through (3, 8)
def line₂ (x y : ℚ) : Prop :=
  y - point.2 = m₂ * (x - point.1)

-- Define the intersection point
def intersection_point : ℚ × ℚ := (13/5, 41/5)

-- Theorem statement
theorem intersection_point_is_correct :
  line₁ intersection_point.1 intersection_point.2 ∧
  line₂ intersection_point.1 intersection_point.2 := by
  sorry

end intersection_point_is_correct_l984_98468


namespace base_three_20121_equals_178_l984_98443

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base_three_20121_equals_178 :
  base_three_to_decimal [2, 0, 1, 2, 1] = 178 := by
  sorry

end base_three_20121_equals_178_l984_98443


namespace wilson_family_seating_arrangements_l984_98469

/-- The number of ways to seat a family with the given constraints -/
def seatingArrangements (numBoys numGirls : ℕ) : ℕ :=
  let numAdjacentBoys := 3
  let totalSeats := numBoys + numGirls
  let numRemainingBoys := numBoys - numAdjacentBoys
  let numEntities := numRemainingBoys + numGirls + 1  -- +1 for the block of 3 boys
  (numBoys.choose numAdjacentBoys) * (Nat.factorial numAdjacentBoys) *
  (Nat.factorial numEntities) * (Nat.factorial numRemainingBoys) *
  (Nat.factorial numGirls)

/-- Theorem stating that the number of seating arrangements for the Wilson family is 5760 -/
theorem wilson_family_seating_arrangements :
  seatingArrangements 5 2 = 5760 := by
  sorry

#eval seatingArrangements 5 2

end wilson_family_seating_arrangements_l984_98469


namespace solution_system1_solution_system2_l984_98452

-- Define the systems of equations
def system1 (x y : ℝ) : Prop := (3 * x + 2 * y = 5) ∧ (y = 2 * x - 8)
def system2 (x y : ℝ) : Prop := (2 * x - y = 10) ∧ (2 * x + 3 * y = 2)

-- Theorem for System 1
theorem solution_system1 : ∃ x y : ℝ, system1 x y ∧ x = 3 ∧ y = -2 := by
  sorry

-- Theorem for System 2
theorem solution_system2 : ∃ x y : ℝ, system2 x y ∧ x = 4 ∧ y = -2 := by
  sorry

end solution_system1_solution_system2_l984_98452


namespace angle_identity_l984_98407

theorem angle_identity (A B C : ℝ) (h : A + B + C = π) :
  Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 - 2 * Real.cos A * Real.cos B * Real.cos C = 2 := by
  sorry

end angle_identity_l984_98407


namespace opposite_of_abs_neg_five_l984_98417

theorem opposite_of_abs_neg_five : -(|-5|) = -5 := by
  sorry

end opposite_of_abs_neg_five_l984_98417


namespace distance_between_foci_l984_98466

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 3)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 25

/-- The first focus of the ellipse -/
def F1 : ℝ × ℝ := (4, 3)

/-- The second focus of the ellipse -/
def F2 : ℝ × ℝ := (-6, 9)

/-- The theorem stating the distance between the foci -/
theorem distance_between_foci :
  Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2) = 2 * Real.sqrt 34 := by
  sorry

end distance_between_foci_l984_98466


namespace aiguo_seashells_l984_98445

/-- The number of seashells collected by Aiguo, Vail, and Stefan satisfies the given conditions -/
def seashell_collection (aiguo vail stefan : ℕ) : Prop :=
  stefan = vail + 16 ∧ 
  vail + 5 = aiguo ∧ 
  aiguo + vail + stefan = 66

/-- Aiguo had 20 seashells -/
theorem aiguo_seashells :
  ∃ (vail stefan : ℕ), seashell_collection 20 vail stefan := by
  sorry

end aiguo_seashells_l984_98445


namespace pasture_problem_l984_98448

/-- The number of horses c put in the pasture -/
def c_horses : ℕ := 18

/-- The total cost of the pasture in Rs -/
def total_cost : ℕ := 870

/-- b's payment for the pasture in Rs -/
def b_payment : ℕ := 360

/-- a's horses -/
def a_horses : ℕ := 12

/-- b's horses -/
def b_horses : ℕ := 16

/-- a's months -/
def a_months : ℕ := 8

/-- b's months -/
def b_months : ℕ := 9

/-- c's months -/
def c_months : ℕ := 6

theorem pasture_problem :
  c_horses * c_months * total_cost = 
    b_payment * (a_horses * a_months + b_horses * b_months + c_horses * c_months) - 
    b_horses * b_months * total_cost := by
  sorry

end pasture_problem_l984_98448


namespace total_muffins_after_baking_l984_98498

def initial_muffins : ℕ := 35
def additional_muffins : ℕ := 48

theorem total_muffins_after_baking :
  initial_muffins + additional_muffins = 83 := by
  sorry

end total_muffins_after_baking_l984_98498


namespace more_pups_than_adults_l984_98416

def num_huskies : ℕ := 5
def num_pitbulls : ℕ := 2
def num_golden_retrievers : ℕ := 4

def pups_per_husky : ℕ := 3
def pups_per_pitbull : ℕ := 3
def pups_per_golden_retriever : ℕ := pups_per_husky + 2

def total_adult_dogs : ℕ := num_huskies + num_pitbulls + num_golden_retrievers

def total_pups : ℕ := 
  num_huskies * pups_per_husky + 
  num_pitbulls * pups_per_pitbull + 
  num_golden_retrievers * pups_per_golden_retriever

theorem more_pups_than_adults : total_pups - total_adult_dogs = 30 := by
  sorry

end more_pups_than_adults_l984_98416


namespace det_2x2_matrix_l984_98411

theorem det_2x2_matrix : 
  Matrix.det !![4, 3; 2, 1] = -2 := by
  sorry

end det_2x2_matrix_l984_98411


namespace function_divides_property_l984_98433

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem function_divides_property 
  (f : ℤ → ℕ+) 
  (h : ∀ m n : ℤ, divides (f (m - n)) (f m - f n)) :
  ∀ n m : ℤ, f n ≤ f m → divides (f n) (f m) := by
  sorry

end function_divides_property_l984_98433


namespace pirate_game_solution_l984_98482

def pirate_game (initial_coins : ℕ) : (ℕ × ℕ) :=
  let after_first_transfer := (initial_coins / 2, initial_coins + initial_coins / 2)
  let after_second_transfer := (after_first_transfer.1 + after_first_transfer.2 / 2, after_first_transfer.2 / 2)
  (after_second_transfer.1 / 2, after_second_transfer.2 + after_second_transfer.1 / 2)

theorem pirate_game_solution :
  ∃ (x : ℕ), pirate_game x = (15, 33) ∧ x = 24 :=
sorry

end pirate_game_solution_l984_98482


namespace line_through_coefficient_points_l984_98426

/-- Given two lines that intersect at (2, 3), prove the equation of the line
    passing through the points formed by their coefficients. -/
theorem line_through_coefficient_points
  (A₁ B₁ A₂ B₂ : ℝ) 
  (h₁ : A₁ * 2 + B₁ * 3 = 1)
  (h₂ : A₂ * 2 + B₂ * 3 = 1) :
  ∀ x y : ℝ, (x = A₁ ∧ y = B₁) ∨ (x = A₂ ∧ y = B₂) → 2*x + 3*y = 1 :=
sorry

end line_through_coefficient_points_l984_98426


namespace secret_reaches_1093_on_sunday_l984_98474

def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

theorem secret_reaches_1093_on_sunday : 
  ∃ n : ℕ, secret_spread n = 1093 ∧ n = 6 :=
sorry

end secret_reaches_1093_on_sunday_l984_98474


namespace probability_at_least_one_woman_l984_98494

theorem probability_at_least_one_woman (total_people : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) :
  total_people = men + women →
  men = 10 →
  women = 5 →
  selected = 4 →
  (1 - (Nat.choose men selected : ℚ) / (Nat.choose total_people selected : ℚ)) = 77 / 91 :=
by sorry

end probability_at_least_one_woman_l984_98494


namespace f_monotonicity_and_extrema_l984_98480

noncomputable def f (x : ℝ) := Real.sin x - Real.cos x + x + 1

theorem f_monotonicity_and_extrema :
  ∀ x : ℝ, 0 < x → x < 2 * Real.pi →
  (∀ y : ℝ, 0 < y → y < Real.pi → HasDerivAt f (Real.cos y + Real.sin y + 1) y) ∧
  (∀ y : ℝ, Real.pi < y → y < 3 * Real.pi / 2 → HasDerivAt f (Real.cos y + Real.sin y + 1) y) ∧
  (∀ y : ℝ, 3 * Real.pi / 2 < y → y < 2 * Real.pi → HasDerivAt f (Real.cos y + Real.sin y + 1) y) ∧
  (f (3 * Real.pi / 2) = 3 * Real.pi / 2) ∧
  (f Real.pi = Real.pi + 2) ∧
  (∀ y : ℝ, 0 < y → y < 2 * Real.pi → f y ≥ 3 * Real.pi / 2) ∧
  (∀ y : ℝ, 0 < y → y < 2 * Real.pi → f y ≤ Real.pi + 2) :=
by sorry

end f_monotonicity_and_extrema_l984_98480


namespace cos_315_degrees_l984_98412

theorem cos_315_degrees : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end cos_315_degrees_l984_98412


namespace mountain_paths_l984_98450

/-- Given a mountain with paths from east and west sides, calculate the total number of ways to ascend and descend -/
theorem mountain_paths (east_paths west_paths : ℕ) : 
  east_paths = 3 → west_paths = 2 → (east_paths + west_paths) * (east_paths + west_paths) = 25 := by
  sorry

#check mountain_paths

end mountain_paths_l984_98450


namespace shaded_squares_percentage_l984_98467

/-- Given a 5x5 grid with 9 shaded squares, the percentage of shaded squares is 36%. -/
theorem shaded_squares_percentage :
  ∀ (total_squares shaded_squares : ℕ),
    total_squares = 5 * 5 →
    shaded_squares = 9 →
    (shaded_squares : ℚ) / total_squares * 100 = 36 := by
  sorry

end shaded_squares_percentage_l984_98467


namespace tangency_quadrilateral_area_is_1_6_l984_98436

/-- An isosceles trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  /-- Radius of the inscribed circle -/
  radius : ℝ
  /-- Area of the trapezoid -/
  trapezoidArea : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : Bool
  /-- The circle is inscribed in the trapezoid -/
  isInscribed : Bool

/-- The area of the quadrilateral formed by the points of tangency -/
def tangencyQuadrilateralArea (t : InscribedCircleTrapezoid) : ℝ := sorry

/-- Theorem: The area of the tangency quadrilateral is 1.6 -/
theorem tangency_quadrilateral_area_is_1_6 (t : InscribedCircleTrapezoid) 
  (h1 : t.radius = 1) 
  (h2 : t.trapezoidArea = 5) 
  (h3 : t.isIsosceles = true) 
  (h4 : t.isInscribed = true) : 
  tangencyQuadrilateralArea t = 1.6 := by sorry

end tangency_quadrilateral_area_is_1_6_l984_98436


namespace mass_percentage_h_in_water_l984_98471

/-- The mass percentage of hydrogen in water, considering isotopic composition --/
theorem mass_percentage_h_in_water (h1_abundance : Real) (h2_abundance : Real)
  (h1_mass : Real) (h2_mass : Real) (o_mass : Real)
  (h1_abundance_val : h1_abundance = 0.9998)
  (h2_abundance_val : h2_abundance = 0.0002)
  (h1_mass_val : h1_mass = 1)
  (h2_mass_val : h2_mass = 2)
  (o_mass_val : o_mass = 16) :
  let avg_h_mass := h1_abundance * h1_mass + h2_abundance * h2_mass
  let water_mass := 2 * avg_h_mass + o_mass
  let mass_percentage := (2 * avg_h_mass) / water_mass * 100
  ∃ ε > 0, |mass_percentage - 11.113| < ε :=
sorry

end mass_percentage_h_in_water_l984_98471


namespace ln_range_l984_98453

open Real

theorem ln_range (f : ℝ → ℝ) (x : ℝ) :
  (∀ y, f y = log y) →
  f (x - 1) < 1 →
  1 < x ∧ x < exp 1 + 1 := by
  sorry

end ln_range_l984_98453


namespace constant_function_if_arithmetic_mean_l984_98438

def IsArithmeticMean (f : ℤ × ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (x, y) = (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) / 4

theorem constant_function_if_arithmetic_mean (f : ℤ × ℤ → ℤ) 
  (h1 : ∀ x y : ℤ, f (x, y) > 0)
  (h2 : IsArithmeticMean f) :
  ∃ c : ℤ, ∀ x y : ℤ, f (x, y) = c := by
  sorry

end constant_function_if_arithmetic_mean_l984_98438


namespace nearest_fraction_sum_l984_98422

theorem nearest_fraction_sum : ∃ (x : ℕ), 
  (2007 : ℝ) / 2999 + (8001 : ℝ) / x + (2001 : ℝ) / 3999 = 3.0035428163476343 ∧ 
  x = 4362 := by
sorry

end nearest_fraction_sum_l984_98422


namespace square_circumference_l984_98424

/-- Given a square with an area of 324 square meters, its circumference is 72 meters. -/
theorem square_circumference (s : Real) (area : Real) (h1 : area = 324) (h2 : s^2 = area) :
  4 * s = 72 := by
  sorry

end square_circumference_l984_98424


namespace problem_1_problem_2_l984_98444

-- Problem 1
theorem problem_1 (x : ℝ) : x * (x + 6) + (x - 3)^2 = 2 * x^2 + 9 := by
  sorry

-- Problem 2
theorem problem_2 (m n : ℝ) (hm : m ≠ 0) (hmn : 3 * m ≠ n) :
  (3 + n / m) / ((9 * m^2 - n^2) / m) = 1 / (3 * m - n) := by
  sorry

end problem_1_problem_2_l984_98444


namespace digit_cube_equals_square_l984_98495

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem digit_cube_equals_square (n : Nat) : 
  n ∈ Finset.range 1000 → (n^2 = (sum_of_digits n)^3 ↔ n = 1 ∨ n = 27) := by
  sorry

end digit_cube_equals_square_l984_98495


namespace cosine_one_third_irrational_l984_98451

theorem cosine_one_third_irrational (a : ℝ) (h : Real.cos (π * a) = (1 : ℝ) / 3) : 
  Irrational a := by sorry

end cosine_one_third_irrational_l984_98451


namespace negative_eight_interpretations_l984_98408

theorem negative_eight_interpretations :
  (-(- 8) = -(-8)) ∧
  (-(- 8) = (-1) * (-8)) ∧
  (-(- 8) = |(-8)|) ∧
  (-(- 8) = 8) := by
  sorry

end negative_eight_interpretations_l984_98408


namespace triangular_sum_perfect_squares_l984_98488

def triangular_sum (K : ℕ) : ℕ := K * (K + 1) / 2

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem triangular_sum_perfect_squares :
  {K : ℕ | K > 0 ∧ K < 50 ∧ is_perfect_square (triangular_sum K)} = {1, 8, 49} := by
sorry

end triangular_sum_perfect_squares_l984_98488


namespace find_divisor_l984_98496

theorem find_divisor (N : ℝ) (D : ℝ) (h1 : N = 95) (h2 : N / D + 23 = 42) : D = 5 := by
  sorry

end find_divisor_l984_98496


namespace L_intersects_C_twice_L_min_chord_correct_l984_98491

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line L
def L (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Statement 1: L always intersects C at two points for any real m
theorem L_intersects_C_twice : ∀ m : ℝ, ∃! (p q : ℝ × ℝ), 
  p ≠ q ∧ C p.1 p.2 ∧ C q.1 q.2 ∧ L m p.1 p.2 ∧ L m q.1 q.2 :=
sorry

-- Statement 2: Equation of L with minimum chord length
def L_min_chord (x y : ℝ) : Prop := 2*x - y - 5 = 0

theorem L_min_chord_correct : 
  (∀ m : ℝ, ∃ (p q : ℝ × ℝ), p ≠ q ∧ C p.1 p.2 ∧ C q.1 q.2 ∧ L m p.1 p.2 ∧ L m q.1 q.2 ∧ 
    ∀ (r s : ℝ × ℝ), r ≠ s ∧ C r.1 r.2 ∧ C s.1 s.2 ∧ L_min_chord r.1 r.2 ∧ L_min_chord s.1 s.2 →
      (p.1 - q.1)^2 + (p.2 - q.2)^2 ≥ (r.1 - s.1)^2 + (r.2 - s.2)^2) ∧
  (∃ (p q : ℝ × ℝ), p ≠ q ∧ C p.1 p.2 ∧ C q.1 q.2 ∧ L_min_chord p.1 p.2 ∧ L_min_chord q.1 q.2) :=
sorry

end L_intersects_C_twice_L_min_chord_correct_l984_98491
