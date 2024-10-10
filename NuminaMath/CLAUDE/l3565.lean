import Mathlib

namespace annual_increase_rate_l3565_356542

theorem annual_increase_rate (initial_value final_value : ℝ) 
  (h1 : initial_value = 6400)
  (h2 : final_value = 8100) :
  ∃ r : ℝ, initial_value * (1 + r)^2 = final_value ∧ r = 0.125 := by
sorry

end annual_increase_rate_l3565_356542


namespace min_tries_for_given_counts_l3565_356578

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  purple : Nat
  blue : Nat
  yellow : Nat
  green : Nat

/-- The minimum number of tries required to get at least two blue, two yellow, and one green ball -/
def minTriesRequired (counts : BallCounts) : Nat :=
  counts.purple + (counts.yellow - 1) + (counts.green - 1) + 2 + 2

/-- Theorem stating the minimum number of tries required for the given ball counts -/
theorem min_tries_for_given_counts :
  let counts : BallCounts := ⟨9, 7, 13, 6⟩
  minTriesRequired counts = 30 := by sorry

end min_tries_for_given_counts_l3565_356578


namespace triangle_side_and_area_l3565_356521

theorem triangle_side_and_area (a b c : ℝ) (B : ℝ) (h_a : a = 8) (h_b : b = 7) (h_B : B = Real.pi / 3) :
  c^2 - 4*c - 25 = 0 ∧ 
  ∃ S : ℝ, S = (1/2) * a * c * Real.sin B :=
by sorry

end triangle_side_and_area_l3565_356521


namespace one_third_of_recipe_l3565_356503

theorem one_third_of_recipe (original_amount : ℚ) (reduced_amount : ℚ) : 
  original_amount = 5 + 3/4 → reduced_amount = (1/3) * original_amount → 
  reduced_amount = 1 + 11/12 := by
sorry

end one_third_of_recipe_l3565_356503


namespace circle_projection_bodies_l3565_356576

/-- A type representing geometric bodies -/
inductive GeometricBody
  | Cone
  | Cylinder
  | Sphere
  | Other

/-- A predicate that determines if a geometric body appears as a circle from a certain perspective -/
def appearsAsCircle (body : GeometricBody) : Prop :=
  sorry

/-- The theorem stating that cones, cylinders, and spheres appear as circles from certain perspectives -/
theorem circle_projection_bodies :
  ∃ (cone cylinder sphere : GeometricBody),
    cone = GeometricBody.Cone ∧
    cylinder = GeometricBody.Cylinder ∧
    sphere = GeometricBody.Sphere ∧
    appearsAsCircle cone ∧
    appearsAsCircle cylinder ∧
    appearsAsCircle sphere :=
  sorry

end circle_projection_bodies_l3565_356576


namespace hyperbola_eccentricity_l3565_356550

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  let focal_distance := 2 * c
  let asymptote_slope := b / a
  let focus_to_asymptote_distance := b * c / Real.sqrt (a^2 + b^2)
  focus_to_asymptote_distance = (1/4) * focal_distance →
  c / a = 2 * Real.sqrt 3 / 3 := by
  sorry

end hyperbola_eccentricity_l3565_356550


namespace painter_work_days_l3565_356527

/-- Represents the number of work-days required for a given number of painters to complete a job -/
def work_days (painters : ℕ) (days : ℚ) : Prop :=
  painters * days = 6 * 2

theorem painter_work_days :
  work_days 6 2 → work_days 4 3 := by
  sorry

end painter_work_days_l3565_356527


namespace max_sum_of_squares_l3565_356565

/-- Given a system of equations, prove that the maximum value of a^2 + b^2 + c^2 + d^2 is 714 -/
theorem max_sum_of_squares (a b c d : ℝ) 
  (eq1 : a + b = 18)
  (eq2 : a * b + c + d = 95)
  (eq3 : a * d + b * c = 195)
  (eq4 : c * d = 120) :
  a^2 + b^2 + c^2 + d^2 ≤ 714 ∧ 
  ∃ (a₀ b₀ c₀ d₀ : ℝ), 
    a₀ + b₀ = 18 ∧
    a₀ * b₀ + c₀ + d₀ = 95 ∧
    a₀ * d₀ + b₀ * c₀ = 195 ∧
    c₀ * d₀ = 120 ∧
    a₀^2 + b₀^2 + c₀^2 + d₀^2 = 714 :=
by sorry

end max_sum_of_squares_l3565_356565


namespace pizza_slice_count_l3565_356592

/-- Given a number of pizzas and slices per pizza, calculates the total number of slices -/
def total_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) : ℕ :=
  num_pizzas * slices_per_pizza

/-- Proves that 21 pizzas with 8 slices each results in 168 total slices -/
theorem pizza_slice_count : total_slices 21 8 = 168 := by
  sorry

end pizza_slice_count_l3565_356592


namespace staircase_cutting_count_l3565_356544

/-- Represents a staircase with a given number of steps -/
structure Staircase :=
  (steps : ℕ)

/-- Represents a cutting of the staircase into rectangles and a square -/
structure StaircaseCutting :=
  (staircase : Staircase)
  (rectangles : ℕ)
  (squares : ℕ)

/-- Counts the number of ways to cut a staircase -/
def countCuttings (s : Staircase) (r : ℕ) (sq : ℕ) : ℕ :=
  sorry

/-- The main theorem: there are 32 ways to cut a 6-step staircase into 5 rectangles and one square -/
theorem staircase_cutting_count :
  countCuttings (Staircase.mk 6) 5 1 = 32 := by
  sorry

end staircase_cutting_count_l3565_356544


namespace min_radios_problem_l3565_356590

/-- Represents the problem of finding the minimum number of radios. -/
theorem min_radios_problem (n d : ℕ) : 
  n > 0 → -- n is positive
  d > 0 → -- d is positive
  (45 : ℚ) - (d + 90 : ℚ) / n = -105 → -- profit equation
  n ≥ 2 := by
  sorry

#check min_radios_problem

end min_radios_problem_l3565_356590


namespace quadratic_equal_roots_l3565_356531

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*k*x + 6 = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*k*y + 6 = 0 → y = x) ↔ 
  k = Real.sqrt 6 ∨ k = -Real.sqrt 6 :=
by sorry

end quadratic_equal_roots_l3565_356531


namespace insurance_company_expenses_percentage_l3565_356530

/-- Proves that given the conditions from the problem, the expenses in 2006 were 55.2% of the revenue in 2006 -/
theorem insurance_company_expenses_percentage (revenue2005 expenses2005 : ℝ) 
  (h1 : revenue2005 > 0)
  (h2 : expenses2005 > 0)
  (h3 : revenue2005 > expenses2005)
  (h4 : (1.25 * revenue2005 - 1.15 * expenses2005) = 1.4 * (revenue2005 - expenses2005)) :
  (1.15 * expenses2005) / (1.25 * revenue2005) = 0.552 := by
sorry

end insurance_company_expenses_percentage_l3565_356530


namespace smallest_solution_congruence_l3565_356570

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 17 % 31 ∧
  ∀ (y : ℕ), y > 0 ∧ (5 * y) % 31 = 17 % 31 → x ≤ y :=
by
  -- The proof goes here
  sorry

end smallest_solution_congruence_l3565_356570


namespace expression_simplification_l3565_356529

theorem expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let expr := ((a^(3/4) - b^(3/4)) * (a^(3/4) + b^(3/4)) / (a^(1/2) - b^(1/2)) - Real.sqrt (a * b)) *
               (2 * Real.sqrt 2.5 * (a + b)⁻¹) / (Real.sqrt 1000)^(1/3)
  expr = 1 := by
  sorry

end expression_simplification_l3565_356529


namespace floor_sum_equals_140_l3565_356506

theorem floor_sum_equals_140 (p q r s : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
  (h1 : p^2 + q^2 = 2500) (h2 : r^2 + s^2 = 2500) (h3 : p * r = 1200) (h4 : q * s = 1200) :
  ⌊p + q + r + s⌋ = 140 := by
  sorry

end floor_sum_equals_140_l3565_356506


namespace rectangle_height_decrease_l3565_356573

theorem rectangle_height_decrease (b h : ℝ) (h_pos : 0 < b) (h_pos' : 0 < h) :
  let new_base := 1.1 * b
  let new_height := h * (1 - 9 / 11 / 100)
  b * h = new_base * new_height := by
  sorry

end rectangle_height_decrease_l3565_356573


namespace our_system_is_linear_l3565_356500

/-- A linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  toFun : ℝ → ℝ → Prop := λ x y => a * x + b * y = c

/-- A system of two equations -/
structure SystemOfTwoEquations where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- The specific system we want to prove is linear -/
def ourSystem : SystemOfTwoEquations where
  eq1 := { a := 1, b := 1, c := 2 }
  eq2 := { a := 1, b := -1, c := 4 }

/-- A predicate to check if a system is linear -/
def isLinearSystem (s : SystemOfTwoEquations) : Prop :=
  s.eq1.a ≠ 0 ∨ s.eq1.b ≠ 0 ∧
  s.eq2.a ≠ 0 ∨ s.eq2.b ≠ 0

theorem our_system_is_linear : isLinearSystem ourSystem := by
  sorry

end our_system_is_linear_l3565_356500


namespace first_day_exceeding_500_l3565_356588

def bacterial_population (n : ℕ) : ℕ := 4 * 3^n

theorem first_day_exceeding_500 :
  ∃ n : ℕ, bacterial_population n > 500 ∧ ∀ m : ℕ, m < n → bacterial_population m ≤ 500 :=
by
  use 6
  sorry

end first_day_exceeding_500_l3565_356588


namespace log_sum_equals_two_l3565_356547

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end log_sum_equals_two_l3565_356547


namespace spherical_coordinate_transformation_l3565_356582

/-- Given a point with rectangular coordinates (-3, -4, 5) and spherical coordinates (ρ, θ, φ),
    the point with spherical coordinates (ρ, θ + π, -φ) has rectangular coordinates (3, 4, 5). -/
theorem spherical_coordinate_transformation (ρ θ φ : ℝ) :
  ρ * Real.sin φ * Real.cos θ = -3 →
  ρ * Real.sin φ * Real.sin θ = -4 →
  ρ * Real.cos φ = 5 →
  ρ * Real.sin (-φ) * Real.cos (θ + π) = 3 ∧
  ρ * Real.sin (-φ) * Real.sin (θ + π) = 4 ∧
  ρ * Real.cos (-φ) = 5 :=
by sorry

end spherical_coordinate_transformation_l3565_356582


namespace d_value_l3565_356585

-- Define the function f(x) = x⋅(4x-3)
def f (x : ℝ) : ℝ := x * (4 * x - 3)

-- Define the interval (-9/4, 3/2)
def interval : Set ℝ := { x | -9/4 < x ∧ x < 3/2 }

-- State the theorem
theorem d_value : 
  ∃ d : ℝ, (∀ x : ℝ, f x < d ↔ x ∈ interval) → d = 27/2 := by sorry

end d_value_l3565_356585


namespace triangle_area_l3565_356538

theorem triangle_area (a b c A B C : Real) : 
  a = 7 →
  2 * Real.sin A = Real.sqrt 3 →
  Real.sin B + Real.sin C = 13 * Real.sqrt 3 / 14 →
  (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3 :=
by sorry

end triangle_area_l3565_356538


namespace systematic_sampling_probabilities_l3565_356563

theorem systematic_sampling_probabilities 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (excluded_count : ℕ) 
  (h1 : total_students = 1005)
  (h2 : sample_size = 50)
  (h3 : excluded_count = 5) :
  (excluded_count : ℚ) / total_students = 5 / 1005 ∧
  (sample_size : ℚ) / total_students = 50 / 1005 := by
  sorry

end systematic_sampling_probabilities_l3565_356563


namespace desired_depth_is_50_l3565_356511

/-- Represents the digging scenario with initial and new conditions -/
structure DiggingScenario where
  initial_men : ℕ
  initial_hours : ℕ
  initial_depth : ℕ
  new_hours : ℕ
  extra_men : ℕ

/-- Calculates the desired depth given a digging scenario -/
def desired_depth (scenario : DiggingScenario) : ℕ :=
  let initial_work := scenario.initial_men * scenario.initial_hours
  let new_men := scenario.initial_men + scenario.extra_men
  let new_work := new_men * scenario.new_hours
  (new_work * scenario.initial_depth) / initial_work

/-- The main theorem stating that the desired depth is 50 meters -/
theorem desired_depth_is_50 (scenario : DiggingScenario)
  (h1 : scenario.initial_men = 18)
  (h2 : scenario.initial_hours = 8)
  (h3 : scenario.initial_depth = 30)
  (h4 : scenario.new_hours = 6)
  (h5 : scenario.extra_men = 22) :
  desired_depth scenario = 50 := by
  sorry

end desired_depth_is_50_l3565_356511


namespace consecutive_products_not_end_2019_l3565_356599

theorem consecutive_products_not_end_2019 (n : ℤ) : 
  ∃ k : ℕ, ((n - 1) * (n + 1) + n * (n - 1) + n * (n + 1)) % 10000 ≠ 2019 + 10000 * k := by
  sorry

end consecutive_products_not_end_2019_l3565_356599


namespace minimum_value_range_l3565_356525

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The theorem stating the range of m for which f(x) has a minimum on (m, 6-m^2) --/
theorem minimum_value_range (m : ℝ) : 
  (∃ (c : ℝ), c ∈ Set.Ioo m (6 - m^2) ∧ 
    (∀ x ∈ Set.Ioo m (6 - m^2), f c ≤ f x)) ↔ 
  m ∈ Set.Icc (-2) 1 := by sorry

end minimum_value_range_l3565_356525


namespace original_fraction_l3565_356548

theorem original_fraction (x y : ℚ) 
  (h1 : x / (y + 1) = 1 / 2) 
  (h2 : (x + 1) / y = 1) : 
  x / y = 2 / 3 := by
  sorry

end original_fraction_l3565_356548


namespace course_length_proof_l3565_356510

/-- Proves that the length of a course is 45 miles given the conditions of two cyclists --/
theorem course_length_proof (speed1 speed2 time : ℝ) 
  (h1 : speed1 = 14)
  (h2 : speed2 = 16)
  (h3 : time = 1.5)
  : speed1 * time + speed2 * time = 45 := by
  sorry

#check course_length_proof

end course_length_proof_l3565_356510


namespace binomial_20_10_l3565_356509

theorem binomial_20_10 (h1 : Nat.choose 18 8 = 31824) 
                        (h2 : Nat.choose 18 9 = 48620) 
                        (h3 : Nat.choose 18 10 = 43758) : 
  Nat.choose 20 10 = 172822 := by
  sorry

end binomial_20_10_l3565_356509


namespace even_sum_problem_l3565_356571

theorem even_sum_problem (n : ℕ) (h1 : Odd n) 
  (h2 : (n^2 - 1) / 4 = 95 * 96) : n = 191 := by
  sorry

end even_sum_problem_l3565_356571


namespace equidistant_point_on_x_axis_l3565_356560

/-- Given two points M₁(x₁, y₁, z₁) and M₂(x₂, y₂, z₂), this theorem proves that the x-coordinate
    of the point P(x, 0, 0) on the Ox axis that is equidistant from M₁ and M₂ is given by
    x = (x₂² - x₁² + y₂² - y₁² + z₂² - z₁²) / (2(x₂ - x₁)) -/
theorem equidistant_point_on_x_axis 
  (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) 
  (h : x₁ ≠ x₂) : 
  ∃ x : ℝ, x = (x₂^2 - x₁^2 + y₂^2 - y₁^2 + z₂^2 - z₁^2) / (2 * (x₂ - x₁)) ∧ 
  (x - x₁)^2 + y₁^2 + z₁^2 = (x - x₂)^2 + y₂^2 + z₂^2 :=
by sorry

end equidistant_point_on_x_axis_l3565_356560


namespace circle_radius_l3565_356545

theorem circle_radius (x y : ℝ) :
  x^2 + y^2 - 2*x + 6*y + 1 = 0 → ∃ (h k r : ℝ), r = 3 ∧ (x - h)^2 + (y - k)^2 = r^2 := by
  sorry

end circle_radius_l3565_356545


namespace sqrt_fraction_equals_two_l3565_356569

theorem sqrt_fraction_equals_two (a b : ℝ) (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 7) :
  Real.sqrt ((14 * a^2) / b^2) = 2 := by
  sorry

end sqrt_fraction_equals_two_l3565_356569


namespace complex_magnitude_equation_l3565_356561

theorem complex_magnitude_equation : 
  ∃ (t : ℝ), t > 0 ∧ Complex.abs (8 + 3 * t * Complex.I) = 13 ↔ t = Real.sqrt (105 / 3) :=
by sorry

end complex_magnitude_equation_l3565_356561


namespace solution_set_part1_range_of_a_part2_l3565_356507

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Part 1
theorem solution_set_part1 (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ 2 < x ∧ x < 3) →
  (∀ x, b*x^2 - a*x + 1 > 0 ↔ x < 1/3 ∨ x > 1/2) :=
sorry

-- Part 2
theorem range_of_a_part2 (a : ℝ) :
  (∀ x, x ∈ Set.Ioc (-1) 0 → f a (3-a) x ≥ 0) →
  a ≤ 3 :=
sorry

end solution_set_part1_range_of_a_part2_l3565_356507


namespace cookies_left_after_week_l3565_356555

/-- The number of cookies left after a week -/
def cookiesLeftAfterWeek (initialCookies : ℕ) (cookiesTakenInFourDays : ℕ) : ℕ :=
  initialCookies - 7 * (cookiesTakenInFourDays / 4)

/-- Theorem: The number of cookies left after a week is 28 -/
theorem cookies_left_after_week :
  cookiesLeftAfterWeek 70 24 = 28 := by
  sorry

end cookies_left_after_week_l3565_356555


namespace sin_cos_sum_equals_sqrt2_over_2_l3565_356558

theorem sin_cos_sum_equals_sqrt2_over_2 :
  Real.sin (187 * π / 180) * Real.cos (52 * π / 180) +
  Real.cos (7 * π / 180) * Real.sin (52 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_cos_sum_equals_sqrt2_over_2_l3565_356558


namespace field_length_calculation_l3565_356516

theorem field_length_calculation (width : ℝ) (pond_area : ℝ) (pond_percentage : ℝ) : 
  pond_area = 150 →
  pond_percentage = 0.4 →
  let length := 3 * width
  let field_area := length * width
  pond_area = pond_percentage * field_area →
  length = 15 * Real.sqrt 5 := by
  sorry

end field_length_calculation_l3565_356516


namespace excess_purchase_l3565_356595

/-- Calculates the excess amount of Chinese herbal medicine purchased given the planned amount and completion percentages -/
theorem excess_purchase (planned_amount : ℝ) (first_half_percent : ℝ) (second_half_percent : ℝ) 
  (h1 : planned_amount = 1500)
  (h2 : first_half_percent = 55)
  (h3 : second_half_percent = 65) :
  (first_half_percent + second_half_percent - 100) / 100 * planned_amount = 300 := by
  sorry

end excess_purchase_l3565_356595


namespace line_parameterization_l3565_356584

/-- Given a line y = 3x + 2 parameterized as (x, y) = (5, r) + t(m, 6),
    prove that r = 17 and m = 2 -/
theorem line_parameterization (r m : ℝ) : 
  (∀ t : ℝ, ∀ x y : ℝ, 
    (x = 5 + t * m ∧ y = r + t * 6) → y = 3 * x + 2) →
  r = 17 ∧ m = 2 :=
by sorry

end line_parameterization_l3565_356584


namespace cheryl_unused_material_l3565_356591

-- Define the amount of material Cheryl bought of each type
def material1 : ℚ := 3 / 8
def material2 : ℚ := 1 / 3

-- Define the total amount of material Cheryl bought
def total_bought : ℚ := material1 + material2

-- Define the amount of material Cheryl used
def material_used : ℚ := 0.33333333333333326

-- Define the amount of material left unused
def material_left : ℚ := total_bought - material_used

-- Theorem statement
theorem cheryl_unused_material : material_left = 0.375 := by sorry

end cheryl_unused_material_l3565_356591


namespace cos_225_degrees_l3565_356536

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_degrees_l3565_356536


namespace refusing_managers_pair_l3565_356524

/-- The number of managers to choose from -/
def total_managers : ℕ := 8

/-- The number of managers needed for the meeting -/
def meeting_size : ℕ := 4

/-- The number of ways to select managers for the meeting -/
def selection_ways : ℕ := 55

/-- Calculates the number of combinations -/
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The theorem to prove -/
theorem refusing_managers_pair : 
  ∃! (refusing_pairs : ℕ), 
    combinations total_managers meeting_size - 
    refusing_pairs * combinations (total_managers - 2) (meeting_size - 2) = 
    selection_ways :=
sorry

end refusing_managers_pair_l3565_356524


namespace youngest_age_l3565_356552

/-- Proves the age of the youngest person given the conditions of the problem -/
theorem youngest_age (n : ℕ) (current_avg : ℚ) (birth_avg : ℚ) 
  (h1 : n = 7)
  (h2 : current_avg = 30)
  (h3 : birth_avg = 22) :
  (n * current_avg - (n - 1) * birth_avg) / n = 78 / 7 := by
  sorry

end youngest_age_l3565_356552


namespace absolute_value_inequality_solution_l3565_356562

theorem absolute_value_inequality_solution :
  {x : ℤ | |7 * x - 5| ≤ 9} = {0, 1, 2} := by
  sorry

end absolute_value_inequality_solution_l3565_356562


namespace trig_product_equals_one_l3565_356517

theorem trig_product_equals_one :
  let x : Real := 30 * π / 180  -- 30 degrees in radians
  let y : Real := 60 * π / 180  -- 60 degrees in radians
  (1 - 1 / Real.cos x) * (1 + 1 / Real.sin y) * (1 - 1 / Real.sin x) * (1 + 1 / Real.cos y) = 1 := by
  sorry

end trig_product_equals_one_l3565_356517


namespace divisible_by_five_l3565_356546

theorem divisible_by_five (n : ℕ) : ∃ k : ℤ, (k = 2^n - 1 ∨ k = 2^n + 1 ∨ k = 2^(2*n) + 1) ∧ k % 5 = 0 := by
  sorry

end divisible_by_five_l3565_356546


namespace large_bucket_capacity_l3565_356553

theorem large_bucket_capacity (small : ℝ) (large : ℝ) 
  (h1 : large = 2 * small + 3)
  (h2 : 2 * small + 5 * large = 63) :
  large = 11 := by
sorry

end large_bucket_capacity_l3565_356553


namespace factor_expression_l3565_356513

theorem factor_expression (x : ℝ) : 54 * x^5 - 135 * x^9 = 27 * x^5 * (2 - 5 * x^4) := by
  sorry

end factor_expression_l3565_356513


namespace intersection_condition_l3565_356518

def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}

def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

theorem intersection_condition (a : ℝ) : 
  (A ∩ B a = B a) ↔ (a < -4 ∨ a ≥ 4 ∨ a = -2) :=
sorry

end intersection_condition_l3565_356518


namespace mismatched_boots_count_l3565_356557

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to arrange n distinct items --/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of pairs of boots --/
def num_pairs : ℕ := 6

/-- The number of ways two people can wear mismatched boots --/
def mismatched_boots_ways : ℕ :=
  -- Case 1: Using boots from two pairs
  choose num_pairs 2 * 4 +
  -- Case 2: Using boots from three pairs
  choose num_pairs 3 * 4 * 4 +
  -- Case 3: Using boots from four pairs
  choose num_pairs 4 * factorial 4

theorem mismatched_boots_count :
  mismatched_boots_ways = 740 := by sorry

end mismatched_boots_count_l3565_356557


namespace race_finishing_orders_l3565_356502

/-- The number of possible finishing orders for a race with n participants and no ties -/
def racePermutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of racers -/
def numRacers : ℕ := 4

theorem race_finishing_orders :
  racePermutations numRacers = 24 :=
by sorry

end race_finishing_orders_l3565_356502


namespace average_speed_tony_l3565_356534

def rollercoaster_speeds : List ℝ := [50, 62, 73, 70, 40]

theorem average_speed_tony (speeds := rollercoaster_speeds) : 
  (speeds.sum / speeds.length : ℝ) = 59 := by
  sorry

end average_speed_tony_l3565_356534


namespace sin_2alpha_minus_pi_6_l3565_356512

theorem sin_2alpha_minus_pi_6 (α : Real) :
  (∃ P : Real × Real, P.1 = -3/5 ∧ P.2 = 4/5 ∧ P.1^2 + P.2^2 = 1 ∧
    P.1 = Real.cos (α + π/6) ∧ P.2 = Real.sin (α + π/6)) →
  Real.sin (2*α - π/6) = 7/25 := by
sorry

end sin_2alpha_minus_pi_6_l3565_356512


namespace circle_k_range_l3565_356501

/-- The equation of a potential circle -/
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + 5*k = 0

/-- The condition for the equation to represent a circle -/
def is_circle (k : ℝ) : Prop :=
  ∃ (x₀ y₀ r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y k ↔ (x - x₀)^2 + (y - y₀)^2 = r^2

/-- The theorem stating the range of k for which the equation represents a circle -/
theorem circle_k_range :
  ∀ k : ℝ, is_circle k ↔ k < 1 :=
sorry

end circle_k_range_l3565_356501


namespace vector_dot_product_problem_l3565_356515

-- Define the type for 2D vectors
def Vector2D := ℝ × ℝ

-- Define vector addition
def add (v w : Vector2D) : Vector2D :=
  (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication
def smul (r : ℝ) (v : Vector2D) : Vector2D :=
  (r * v.1, r * v.2)

-- Define dot product
def dot (v w : Vector2D) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_dot_product_problem (a b : Vector2D) 
  (h1 : add (smul 2 a) b = (1, 6)) 
  (h2 : add a (smul 2 b) = (-4, 9)) : 
  dot a b = -2 := by
  sorry

end vector_dot_product_problem_l3565_356515


namespace probability_of_color_change_is_three_seventeenths_l3565_356567

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the probability of observing a color change in a traffic light -/
def probabilityOfColorChange (cycle : TrafficLightCycle) (observationDuration : ℕ) : ℚ :=
  let totalCycleDuration := cycle.green + cycle.yellow + cycle.red
  let changeWindows := 3 * observationDuration
  ↑changeWindows / ↑totalCycleDuration

/-- The main theorem stating the probability of observing a color change -/
theorem probability_of_color_change_is_three_seventeenths :
  let cycle := TrafficLightCycle.mk 45 5 35
  let observationDuration := 5
  probabilityOfColorChange cycle observationDuration = 3 / 17 := by
  sorry

#eval probabilityOfColorChange (TrafficLightCycle.mk 45 5 35) 5

end probability_of_color_change_is_three_seventeenths_l3565_356567


namespace power_minus_one_rational_l3565_356523

/-- A complex number with rational real and imaginary parts and unit modulus -/
structure UnitRationalComplex where
  re : ℚ
  im : ℚ
  unit_modulus : re^2 + im^2 = 1

/-- The result of z^(2n) - 1 is rational for any integer n -/
theorem power_minus_one_rational (z : UnitRationalComplex) (n : ℤ) :
  ∃ (q : ℚ), (z.re + z.im * Complex.I)^(2*n) - 1 = q := by
  sorry

end power_minus_one_rational_l3565_356523


namespace rectangle_width_l3565_356566

/-- Given a rectangle with length 18 cm and a largest inscribed circle with area 153.93804002589985 square cm, the width of the rectangle is 14 cm. -/
theorem rectangle_width (length : ℝ) (circle_area : ℝ) (width : ℝ) : 
  length = 18 → 
  circle_area = 153.93804002589985 → 
  circle_area = Real.pi * (width / 2)^2 → 
  width = 14 := by
sorry

end rectangle_width_l3565_356566


namespace student_committee_size_l3565_356568

theorem student_committee_size (ways : ℕ) (h : ways = 30) : 
  (∃ n : ℕ, n * (n - 1) = ways) → 
  (∃! n : ℕ, n > 0 ∧ n * (n - 1) = ways) ∧ 
  (∃ n : ℕ, n > 0 ∧ n * (n - 1) = ways ∧ n = 6) :=
by sorry

end student_committee_size_l3565_356568


namespace sum_of_squares_of_roots_l3565_356540

theorem sum_of_squares_of_roots (p q r : ℂ) : 
  (3 * p^3 - 4 * p^2 + 3 * p + 7 = 0) →
  (3 * q^3 - 4 * q^2 + 3 * q + 7 = 0) →
  (3 * r^3 - 4 * r^2 + 3 * r + 7 = 0) →
  p^2 + q^2 + r^2 = -2/9 := by
sorry

end sum_of_squares_of_roots_l3565_356540


namespace actual_daily_production_l3565_356543

/-- The actual daily production of TVs given the planned production and early completion. -/
theorem actual_daily_production
  (planned_production : ℕ)
  (planned_days : ℕ)
  (days_ahead : ℕ)
  (h1 : planned_production = 560)
  (h2 : planned_days = 16)
  (h3 : days_ahead = 2)
  : (planned_production : ℚ) / (planned_days - days_ahead) = 40 := by
  sorry

end actual_daily_production_l3565_356543


namespace prob_intersects_inner_is_one_third_l3565_356514

/-- Two concentric circles with radii 1 and 2 -/
structure ConcentricCircles where
  inner_radius : ℝ := 1
  outer_radius : ℝ := 2

/-- A chord on the outer circle -/
structure Chord where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Function to determine if a chord intersects the inner circle -/
def intersects_inner_circle (c : ConcentricCircles) (ch : Chord) : Prop :=
  sorry

/-- Function to calculate the probability of a random chord intersecting the inner circle -/
noncomputable def probability_intersects_inner (c : ConcentricCircles) : ℝ :=
  sorry

/-- Theorem stating that the probability of a random chord intersecting the inner circle is 1/3 -/
theorem prob_intersects_inner_is_one_third (c : ConcentricCircles) :
  probability_intersects_inner c = 1/3 :=
sorry

end prob_intersects_inner_is_one_third_l3565_356514


namespace max_gift_sets_l3565_356537

theorem max_gift_sets (total_chocolates total_candies left_chocolates left_candies : ℕ)
  (h1 : total_chocolates = 69)
  (h2 : total_candies = 86)
  (h3 : left_chocolates = 5)
  (h4 : left_candies = 6) :
  Nat.gcd (total_chocolates - left_chocolates) (total_candies - left_candies) = 16 :=
by sorry

end max_gift_sets_l3565_356537


namespace min_value_reciprocal_sum_l3565_356505

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 3) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + 2*b = 3 → 1/x + 1/y ≤ 1/a + 1/b) ∧
  (1/x + 1/y = 1 + 2*Real.sqrt 2/3) :=
sorry

end min_value_reciprocal_sum_l3565_356505


namespace max_cookies_eaten_24_l3565_356572

/-- Given two siblings sharing cookies, where one eats a positive multiple
    of the other's cookies, this function calculates the maximum number
    of cookies the first sibling could have eaten. -/
def max_cookies_eaten (total_cookies : ℕ) : ℕ :=
  total_cookies / 2

/-- Theorem stating that given 24 cookies shared between two siblings,
    where one sibling eats a positive multiple of the other's cookies,
    the maximum number of cookies the first sibling could have eaten is 12. -/
theorem max_cookies_eaten_24 :
  max_cookies_eaten 24 = 12 := by
  sorry

#eval max_cookies_eaten 24

end max_cookies_eaten_24_l3565_356572


namespace largest_n_multiple_of_three_n_99998_is_solution_n_99998_is_largest_l3565_356532

theorem largest_n_multiple_of_three (n : ℕ) : 
  n < 100000 → 
  (∃ k : ℤ, (n - 3)^5 - n^2 + 10*n - 30 = 3*k) → 
  n ≤ 99998 :=
sorry

theorem n_99998_is_solution : 
  ∃ k : ℤ, (99998 - 3)^5 - 99998^2 + 10*99998 - 30 = 3*k :=
sorry

theorem n_99998_is_largest : 
  ¬∃ n : ℕ, n > 99998 ∧ n < 100000 ∧ 
  (∃ k : ℤ, (n - 3)^5 - n^2 + 10*n - 30 = 3*k) :=
sorry

end largest_n_multiple_of_three_n_99998_is_solution_n_99998_is_largest_l3565_356532


namespace solution_set_l3565_356533

def f (x : ℝ) := abs x + x^2 + 2

theorem solution_set (x : ℝ) :
  f (2*x - 1) > f (3 - x) ↔ x < -2 ∨ x > 4/3 :=
by sorry

end solution_set_l3565_356533


namespace problem_statement_l3565_356556

theorem problem_statement : (-12 : ℚ) * ((2 : ℚ) / 3 - (1 : ℚ) / 4 + (1 : ℚ) / 6) = -7 := by
  sorry

end problem_statement_l3565_356556


namespace college_students_count_l3565_356520

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 210) :
  boys + girls = 546 := by
sorry

end college_students_count_l3565_356520


namespace eraser_cost_l3565_356593

def total_money : ℕ := 100
def heaven_spent : ℕ := 30
def brother_highlighters : ℕ := 30
def num_erasers : ℕ := 10

theorem eraser_cost :
  (total_money - heaven_spent - brother_highlighters) / num_erasers = 4 := by
  sorry

end eraser_cost_l3565_356593


namespace expression_value_l3565_356583

theorem expression_value : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 10) / (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2) = 360 := by
  sorry

end expression_value_l3565_356583


namespace segments_in_proportion_l3565_356575

-- Define the set of line segments
def segments : List ℝ := [2, 3, 4, 6]

-- Define what it means for a list of four numbers to be in proportion
def isInProportion (l : List ℝ) : Prop :=
  l.length = 4 ∧ l[0]! * l[3]! = l[1]! * l[2]!

-- Theorem statement
theorem segments_in_proportion : isInProportion segments := by
  sorry

end segments_in_proportion_l3565_356575


namespace infinite_nested_sqrt_l3565_356559

/-- Given that y is a non-negative real number satisfying y = √(2 - y), prove that y = 1 -/
theorem infinite_nested_sqrt (y : ℝ) (hy : y ≥ 0) (h : y = Real.sqrt (2 - y)) : y = 1 := by
  sorry

end infinite_nested_sqrt_l3565_356559


namespace exists_quadratic_function_with_conditions_l3565_356564

/-- A quadratic function with coefficient a, b, and c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The vertex of a quadratic function is on the negative half of the y-axis -/
def VertexOnNegativeYAxis (a b c : ℝ) : Prop :=
  b = 0 ∧ c < 0

/-- The part of the quadratic function to the left of its axis of symmetry is rising -/
def LeftPartRising (a b c : ℝ) : Prop :=
  a < 0

/-- Theorem stating the existence of a quadratic function satisfying the given conditions -/
theorem exists_quadratic_function_with_conditions : ∃ a b c : ℝ,
  VertexOnNegativeYAxis a b c ∧
  LeftPartRising a b c ∧
  QuadraticFunction a b c = QuadraticFunction (-1) 0 (-1) :=
sorry

end exists_quadratic_function_with_conditions_l3565_356564


namespace unique_integer_property_l3565_356589

theorem unique_integer_property (a : ℕ+) : 
  let b := 2 * a ^ 2
  let c := 2 * b ^ 2
  let d := 2 * c ^ 2
  (∃ n k : ℕ, a * 10^(n+k) + b * 10^k + c = d) → a = 1 := by
sorry

end unique_integer_property_l3565_356589


namespace smallest_positive_multiple_of_45_l3565_356526

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 := by
sorry

end smallest_positive_multiple_of_45_l3565_356526


namespace davids_purchase_cost_l3565_356581

/-- The minimum cost to buy a given number of bottles, given the price of individual bottles and packs --/
def min_cost (single_price : ℚ) (pack_price : ℚ) (pack_size : ℕ) (total_bottles : ℕ) : ℚ :=
  let num_packs := total_bottles / pack_size
  let remaining_bottles := total_bottles % pack_size
  num_packs * pack_price + remaining_bottles * single_price

/-- Theorem stating the minimum cost for David's purchase --/
theorem davids_purchase_cost :
  let single_price : ℚ := 280 / 100  -- $2.80
  let pack_price : ℚ := 1500 / 100   -- $15.00
  let pack_size : ℕ := 6
  let total_bottles : ℕ := 22
  min_cost single_price pack_price pack_size total_bottles = 5620 / 100 := by
  sorry


end davids_purchase_cost_l3565_356581


namespace min_draws_for_eighteen_balls_l3565_356597

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to draw to guarantee at least n balls of a single color -/
def minDrawsForColor (counts : BallCounts) (n : Nat) : Nat :=
  sorry

theorem min_draws_for_eighteen_balls (counts : BallCounts) 
  (h_red : counts.red = 30)
  (h_green : counts.green = 23)
  (h_yellow : counts.yellow = 21)
  (h_blue : counts.blue = 17)
  (h_white : counts.white = 14)
  (h_black : counts.black = 12) :
  minDrawsForColor counts 18 = 95 := by
  sorry

end min_draws_for_eighteen_balls_l3565_356597


namespace equation_solution_difference_l3565_356554

theorem equation_solution_difference : ∃ (a b : ℝ),
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ -6 → ((5 * x - 20) / (x^2 + 3*x - 18) = x + 3 ↔ (x = a ∨ x = b))) ∧
  a > b ∧
  a - b = Real.sqrt 29 := by
  sorry

end equation_solution_difference_l3565_356554


namespace scientific_notation_correct_l3565_356539

/-- The scientific notation of 15.6 billion -/
def scientific_notation_15_6_billion : ℝ := 1.56 * (10 ^ 9)

/-- 15.6 billion as a real number -/
def fifteen_point_six_billion : ℝ := 15600000000

/-- Theorem stating that the scientific notation of 15.6 billion is correct -/
theorem scientific_notation_correct : 
  scientific_notation_15_6_billion = fifteen_point_six_billion := by
  sorry

end scientific_notation_correct_l3565_356539


namespace min_value_polynomial_l3565_356579

theorem min_value_polynomial (x : ℝ) : 
  (13 - x) * (11 - x) * (13 + x) * (11 + x) + 1000 ≥ 424 :=
by sorry

end min_value_polynomial_l3565_356579


namespace tony_winnings_l3565_356574

/-- Calculates the total winnings for lottery tickets with identical numbers -/
def totalWinnings (numTickets : ℕ) (winningNumbersPerTicket : ℕ) (valuePerWinningNumber : ℕ) : ℕ :=
  numTickets * winningNumbersPerTicket * valuePerWinningNumber

/-- Theorem: Tony's total winnings are $300 -/
theorem tony_winnings :
  totalWinnings 3 5 20 = 300 := by
  sorry

end tony_winnings_l3565_356574


namespace multiply_nine_negative_three_l3565_356528

theorem multiply_nine_negative_three : 9 * (-3) = -27 := by
  sorry

end multiply_nine_negative_three_l3565_356528


namespace pencil_distribution_l3565_356508

theorem pencil_distribution (num_pens : ℕ) (num_pencils : ℕ) (max_students : ℕ) :
  num_pens = 2010 →
  max_students = 30 →
  num_pens % max_students = 0 →
  num_pencils % max_students = 0 →
  ∃ k : ℕ, num_pencils = 30 * k :=
by sorry

end pencil_distribution_l3565_356508


namespace last_two_nonzero_digits_of_70_factorial_l3565_356596

-- Define 70!
def factorial_70 : ℕ := Nat.factorial 70

-- Define the function to get the last two nonzero digits
def last_two_nonzero_digits (n : ℕ) : ℕ :=
  n % 100

-- Theorem statement
theorem last_two_nonzero_digits_of_70_factorial :
  last_two_nonzero_digits factorial_70 = 48 := by
  sorry

end last_two_nonzero_digits_of_70_factorial_l3565_356596


namespace light_flashes_l3565_356504

/-- A light flashes every 15 seconds. This theorem proves that it will flash 180 times in ¾ of an hour. -/
theorem light_flashes (flash_interval : ℕ) (hour_fraction : ℚ) (flashes : ℕ) : 
  flash_interval = 15 → hour_fraction = 3/4 → flashes = 180 → 
  (hour_fraction * 3600) / flash_interval = flashes := by
  sorry

end light_flashes_l3565_356504


namespace unit_vector_same_direction_l3565_356577

def b : Fin 2 → ℝ := ![(-3), 4]

theorem unit_vector_same_direction (a : Fin 2 → ℝ) : 
  (∀ i, a i * a i = 1) →  -- a is a unit vector
  (∃ c : ℝ, c ≠ 0 ∧ ∀ i, a i = c * b i) →  -- a is in the same direction as b
  a = ![(-3/5), 4/5] := by
sorry

end unit_vector_same_direction_l3565_356577


namespace sequence_difference_proof_l3565_356549

def arithmetic_sequence_sum (a1 n d : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2

theorem sequence_difference_proof : 
  let n1 := (2298 - 2204) / 2 + 1
  let n2 := (400 - 306) / 2 + 1
  arithmetic_sequence_sum 2204 n1 2 - arithmetic_sequence_sum 306 n2 2 = 91056 := by
  sorry

end sequence_difference_proof_l3565_356549


namespace system_range_of_a_l3565_356551

/-- Given a system of linear equations in x and y, prove the range of a -/
theorem system_range_of_a (x y a : ℝ) 
  (eq1 : x + 3*y = 2 + a) 
  (eq2 : 3*x + y = -4*a) 
  (h : x + y > 2) : 
  a < -2 := by
sorry

end system_range_of_a_l3565_356551


namespace no_nonzero_solution_l3565_356522

theorem no_nonzero_solution :
  ¬∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ (1 / x + 2 / y = 1 / (x + y)) := by
  sorry

end no_nonzero_solution_l3565_356522


namespace point_in_first_quadrant_l3565_356541

theorem point_in_first_quadrant (a : ℝ) (h : a < 0) :
  let P : ℝ × ℝ := (-a, a^2)
  P.1 > 0 ∧ P.2 > 0 :=
sorry

end point_in_first_quadrant_l3565_356541


namespace total_pears_picked_l3565_356598

/-- Represents a person who picks pears -/
structure PearPicker where
  name : String
  morning : Bool

/-- Calculates the number of pears picked on Day 2 -/
def day2Amount (day1 : ℕ) (morning : Bool) : ℕ :=
  if morning then day1 / 2 else day1 * 2

/-- Calculates the number of pears picked on Day 3 -/
def day3Amount (day1 day2 : ℕ) : ℕ :=
  (day1 + day2 + 1) / 2  -- Adding 1 for rounding up

/-- Calculates the total pears picked by a person over three days -/
def totalPears (day1 : ℕ) (morning : Bool) : ℕ :=
  let day2 := day2Amount day1 morning
  let day3 := day3Amount day1 day2
  day1 + day2 + day3

/-- The main theorem stating the total number of pears picked -/
theorem total_pears_picked : 
  let jason := PearPicker.mk "Jason" true
  let keith := PearPicker.mk "Keith" true
  let mike := PearPicker.mk "Mike" true
  let alicia := PearPicker.mk "Alicia" false
  let tina := PearPicker.mk "Tina" false
  let nicola := PearPicker.mk "Nicola" false
  totalPears 46 jason.morning +
  totalPears 47 keith.morning +
  totalPears 12 mike.morning +
  totalPears 28 alicia.morning +
  totalPears 33 tina.morning +
  totalPears 52 nicola.morning = 747 := by
  sorry

end total_pears_picked_l3565_356598


namespace cube_difference_positive_l3565_356535

theorem cube_difference_positive {a b : ℝ} (h : a > b) : a^3 - b^3 > 0 := by
  sorry

end cube_difference_positive_l3565_356535


namespace sum_of_three_integers_with_product_625_l3565_356519

theorem sum_of_three_integers_with_product_625 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 625 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 51 := by
sorry

end sum_of_three_integers_with_product_625_l3565_356519


namespace gcd_A_B_eq_one_l3565_356580

def A : ℤ := 1989^1990 - 1988^1990
def B : ℤ := 1989^1989 - 1988^1989

theorem gcd_A_B_eq_one : Int.gcd A B = 1 := by sorry

end gcd_A_B_eq_one_l3565_356580


namespace buying_more_can_cost_less_buying_101_is_cheaper_l3565_356594

/-- The cost function for notebooks -/
def notebook_cost (n : ℕ) : ℝ :=
  if n ≤ 100 then 2.3 * n else 2.2 * n

theorem buying_more_can_cost_less :
  ∃ (n₁ n₂ : ℕ), n₁ < n₂ ∧ notebook_cost n₁ > notebook_cost n₂ :=
sorry

theorem buying_101_is_cheaper :
  notebook_cost 101 < notebook_cost 100 :=
sorry

end buying_more_can_cost_less_buying_101_is_cheaper_l3565_356594


namespace range_of_a_l3565_356586

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 3 ≥ 0
def q (x a : ℝ) : Prop := x^2 - (2*a - 1)*x + a*(a - 1) ≥ 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ (0 ≤ a ∧ a ≤ 3) :=
sorry

end range_of_a_l3565_356586


namespace line_through_coefficient_points_l3565_356587

/-- Given two lines passing through a common point, prove that the line
    passing through the points defined by their coefficients has a specific equation. -/
theorem line_through_coefficient_points
  (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : 2 * a₁ + 3 * b₁ + 1 = 0)
  (h₂ : 2 * a₂ + 3 * b₂ + 1 = 0) :
  (fun x y : ℝ => 2 * x + 3 * y + 1 = 0) a₁ b₁ ∧ 
  (fun x y : ℝ => 2 * x + 3 * y + 1 = 0) a₂ b₂ := by
  sorry

#check line_through_coefficient_points

end line_through_coefficient_points_l3565_356587
