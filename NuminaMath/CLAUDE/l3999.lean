import Mathlib

namespace NUMINAMATH_CALUDE_evaporation_weight_theorem_l3999_399977

/-- Represents the weight of a glass containing a solution --/
structure GlassSolution where
  total_weight : ℝ
  water_percentage : ℝ
  glass_weight : ℝ

/-- Calculates the final weight of a glass solution after water evaporation --/
def final_weight (initial : GlassSolution) (final_water_percentage : ℝ) : ℝ :=
  sorry

/-- Theorem stating that given the initial conditions and final water percentage,
    the final weight of the glass with solution is 400 grams --/
theorem evaporation_weight_theorem (initial : GlassSolution) 
    (h1 : initial.total_weight = 500)
    (h2 : initial.water_percentage = 0.99)
    (h3 : initial.glass_weight = 300)
    (final_water_percentage : ℝ)
    (h4 : final_water_percentage = 0.98) :
    final_weight initial final_water_percentage = 400 := by
  sorry

end NUMINAMATH_CALUDE_evaporation_weight_theorem_l3999_399977


namespace NUMINAMATH_CALUDE_not_divisible_by_1955_l3999_399979

theorem not_divisible_by_1955 : ∀ n : ℕ, ¬(1955 ∣ (n^2 + n + 1)) := by sorry

end NUMINAMATH_CALUDE_not_divisible_by_1955_l3999_399979


namespace NUMINAMATH_CALUDE_magnitude_of_linear_combination_l3999_399906

/-- Given two plane vectors α and β, prove that |2α + β| = √10 -/
theorem magnitude_of_linear_combination (α β : ℝ × ℝ) 
  (h1 : ‖α‖ = 1) 
  (h2 : ‖β‖ = 2) 
  (h3 : α • (α - 2 • β) = 0) : 
  ‖2 • α + β‖ = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_linear_combination_l3999_399906


namespace NUMINAMATH_CALUDE_earliest_meeting_time_l3999_399949

def kelly_lap_time : ℕ := 5
def rachel_lap_time : ℕ := 8
def mike_lap_time : ℕ := 10

theorem earliest_meeting_time :
  let lap_times := [kelly_lap_time, rachel_lap_time, mike_lap_time]
  Nat.lcm (Nat.lcm kelly_lap_time rachel_lap_time) mike_lap_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_l3999_399949


namespace NUMINAMATH_CALUDE_unique_prime_sum_diff_l3999_399902

theorem unique_prime_sum_diff : ∃! p : ℕ, 
  Prime p ∧ 
  (∃ q r : ℕ, Prime q ∧ Prime r ∧ p = q + r) ∧ 
  (∃ s t : ℕ, Prime s ∧ Prime t ∧ p = s - t) :=
by
  use 5
  sorry

end NUMINAMATH_CALUDE_unique_prime_sum_diff_l3999_399902


namespace NUMINAMATH_CALUDE_proposition_implication_l3999_399912

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 7) : 
  ¬ P 6 := by
sorry

end NUMINAMATH_CALUDE_proposition_implication_l3999_399912


namespace NUMINAMATH_CALUDE_work_completion_days_l3999_399958

/-- The number of days it takes person A to complete the work -/
def days_A : ℝ := 20

/-- The number of days it takes person B to complete the work -/
def days_B : ℝ := 30

/-- The number of days A worked before leaving -/
def days_A_worked : ℝ := 10

/-- The number of days B worked to finish the remaining work -/
def days_B_worked : ℝ := 15

/-- Theorem stating that A can complete the work in 20 days -/
theorem work_completion_days :
  (days_A_worked / days_A) + (days_B_worked / days_B) = 1 :=
sorry

end NUMINAMATH_CALUDE_work_completion_days_l3999_399958


namespace NUMINAMATH_CALUDE_f_minimum_l3999_399908

/-- The quadratic function f(x) = x^2 - 14x + 40 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 40

/-- The value of x that minimizes f(x) -/
def x_min : ℝ := 7

theorem f_minimum :
  ∀ x : ℝ, f x ≥ f x_min :=
sorry

end NUMINAMATH_CALUDE_f_minimum_l3999_399908


namespace NUMINAMATH_CALUDE_rational_roots_count_l3999_399910

/-- The number of distinct possible rational roots for a polynomial of the form
    8x^4 + a₃x³ + a₂x² + a₁x + 16 = 0, where a₃, a₂, and a₁ are integers. -/
def num_rational_roots (a₃ a₂ a₁ : ℤ) : ℕ :=
  16

/-- Theorem stating that the number of distinct possible rational roots for the given polynomial
    is always 16, regardless of the values of a₃, a₂, and a₁. -/
theorem rational_roots_count (a₃ a₂ a₁ : ℤ) :
  num_rational_roots a₃ a₂ a₁ = 16 := by
  sorry

end NUMINAMATH_CALUDE_rational_roots_count_l3999_399910


namespace NUMINAMATH_CALUDE_spherical_coordinate_symmetry_l3999_399997

/-- Given a point with rectangular coordinates (3, -4, 2) and corresponding
    spherical coordinates (ρ, θ, φ), prove that the point with spherical
    coordinates (ρ, -θ, φ) has rectangular coordinates (3, -4, 2). -/
theorem spherical_coordinate_symmetry (ρ θ φ : ℝ) :
  ρ * Real.sin φ * Real.cos θ = 3 ∧
  ρ * Real.sin φ * Real.sin θ = -4 ∧
  ρ * Real.cos φ = 2 →
  ρ * Real.sin φ * Real.cos (-θ) = 3 ∧
  ρ * Real.sin φ * Real.sin (-θ) = -4 ∧
  ρ * Real.cos φ = 2 :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_symmetry_l3999_399997


namespace NUMINAMATH_CALUDE_coefficient_of_x_is_10_l3999_399915

/-- The coefficient of x in the expansion of (x^2 + 1/x)^5 -/
def coefficient_of_x : ℕ :=
  (Nat.choose 5 3)

theorem coefficient_of_x_is_10 : coefficient_of_x = 10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_is_10_l3999_399915


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3999_399973

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r

theorem geometric_sequence_product (x y z : ℝ) :
  is_geometric_sequence (-1) x y z (-2) →
  x * y * z = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3999_399973


namespace NUMINAMATH_CALUDE_mechanic_work_hours_l3999_399933

/-- Calculates the number of hours a mechanic worked given the total cost, part costs, and labor rate. -/
theorem mechanic_work_hours 
  (total_cost : ℝ) 
  (part_cost : ℝ) 
  (labor_rate_per_minute : ℝ) 
  (h1 : total_cost = 220) 
  (h2 : part_cost = 20) 
  (h3 : labor_rate_per_minute = 0.5) : 
  (total_cost - 2 * part_cost) / (labor_rate_per_minute * 60) = 6 := by
  sorry

end NUMINAMATH_CALUDE_mechanic_work_hours_l3999_399933


namespace NUMINAMATH_CALUDE_product_approximation_l3999_399996

def is_approximately_equal (x y : ℕ) (tolerance : ℕ) : Prop :=
  (x ≤ y + tolerance) ∧ (y ≤ x + tolerance)

theorem product_approximation (tolerance : ℕ) :
  (is_approximately_equal (4 * 896) 3600 tolerance) ∧
  (is_approximately_equal (405 * 9) 3600 tolerance) ∧
  ¬(is_approximately_equal (6 * 689) 3600 tolerance) ∧
  ¬(is_approximately_equal (398 * 8) 3600 tolerance) :=
by sorry

end NUMINAMATH_CALUDE_product_approximation_l3999_399996


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3999_399966

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line_l (x y b : ℝ) : Prop := y = (1/2) * x + b

-- Theorem statement
theorem ellipse_and_line_intersection :
  -- Given conditions
  let left_focus : ℝ × ℝ := (-Real.sqrt 3, 0)
  let right_vertex : ℝ × ℝ := (2, 0)
  
  -- Prove the following
  -- 1. Standard equation of ellipse C
  ∀ x y : ℝ, ellipse_C x y ↔ x^2 / 4 + y^2 = 1
  
  -- 2. Maximum chord length and corresponding line equation
  ∧ ∃ max_length : ℝ,
    (max_length = Real.sqrt 10) ∧
    (∀ b : ℝ, ∃ A B : ℝ × ℝ,
      ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
      line_l A.1 A.2 b ∧ line_l B.1 B.2 b ∧
      (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ max_length)) ∧
    (∃ A B : ℝ × ℝ,
      ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
      line_l A.1 A.2 0 ∧ line_l B.1 B.2 0 ∧
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = max_length) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3999_399966


namespace NUMINAMATH_CALUDE_power_four_five_l3999_399961

theorem power_four_five : (4 : ℕ) ^ 4 * (5 : ℕ) ^ 4 = 160000 := by sorry

end NUMINAMATH_CALUDE_power_four_five_l3999_399961


namespace NUMINAMATH_CALUDE_jack_reading_pages_l3999_399923

theorem jack_reading_pages (pages_per_booklet : ℕ) (number_of_booklets : ℕ) (total_pages : ℕ) :
  pages_per_booklet = 9 →
  number_of_booklets = 49 →
  total_pages = pages_per_booklet * number_of_booklets →
  total_pages = 441 :=
by sorry

end NUMINAMATH_CALUDE_jack_reading_pages_l3999_399923


namespace NUMINAMATH_CALUDE_problem_solution_l3999_399988

theorem problem_solution (x y : ℝ) 
  (h1 : 1 / x + 1 / y = 4)
  (h2 : x * y + x + y = 5) :
  x^2 * y + x * y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3999_399988


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3999_399956

theorem solve_exponential_equation :
  ∃ x : ℝ, 16 = 4 * (4 : ℝ) ^ (x - 1) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3999_399956


namespace NUMINAMATH_CALUDE_sams_age_l3999_399936

theorem sams_age (sam masc : ℕ) 
  (h1 : masc = sam + 7)
  (h2 : sam + masc = 27) : 
  sam = 10 := by
sorry

end NUMINAMATH_CALUDE_sams_age_l3999_399936


namespace NUMINAMATH_CALUDE_simplified_expression_ratio_l3999_399998

theorem simplified_expression_ratio (m : ℤ) : 
  let simplified := (6 * m + 12) / 3
  ∃ (c d : ℤ), simplified = c * m + d ∧ c / d = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_ratio_l3999_399998


namespace NUMINAMATH_CALUDE_max_planes_eq_combinations_l3999_399940

/-- The number of points in space -/
def num_points : ℕ := 15

/-- A function that calculates the number of combinations of k items from n items -/
def combinations (n k : ℕ) : ℕ := Nat.choose n k

/-- The maximum number of planes determined by the points -/
def max_planes : ℕ := combinations num_points 3

/-- Theorem stating that the maximum number of planes is equal to the number of combinations of 3 points from 15 points -/
theorem max_planes_eq_combinations : 
  max_planes = combinations num_points 3 := by sorry

end NUMINAMATH_CALUDE_max_planes_eq_combinations_l3999_399940


namespace NUMINAMATH_CALUDE_sum_of_inscribed_squares_l3999_399907

/-- The sum of areas of an infinite series of inscribed squares -/
theorem sum_of_inscribed_squares (a : ℝ) (h : a > 0) :
  ∃ S : ℝ, S = (4 * a^2) / 3 ∧ 
  S = a^2 + ∑' n, (a^2 / 4^n) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_inscribed_squares_l3999_399907


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3999_399993

/-- Given a square with side length y that is divided into a central square
    with side length (y - z) and four congruent rectangles, prove that
    the perimeter of one of these rectangles is 2y. -/
theorem rectangle_perimeter (y z : ℝ) (hz : z < y) :
  let central_side := y - z
  let rect_long_side := y - z
  let rect_short_side := y - (y - z)
  2 * rect_long_side + 2 * rect_short_side = 2 * y :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3999_399993


namespace NUMINAMATH_CALUDE_max_dishes_l3999_399914

theorem max_dishes (main_ingredients : Nat) (secondary_ingredients : Nat) (cooking_methods : Nat)
  (select_main : Nat) (select_secondary : Nat) :
  main_ingredients = 5 →
  secondary_ingredients = 8 →
  cooking_methods = 5 →
  select_main = 2 →
  select_secondary = 3 →
  (Nat.choose main_ingredients select_main) *
  (Nat.choose secondary_ingredients select_secondary) *
  cooking_methods = 2800 :=
by sorry

end NUMINAMATH_CALUDE_max_dishes_l3999_399914


namespace NUMINAMATH_CALUDE_nail_decoration_theorem_l3999_399995

/-- The time it takes to decorate nails with three coats -/
def nail_decoration_time (application_time dry_time number_of_coats : ℕ) : ℕ :=
  (application_time + dry_time) * number_of_coats

/-- Theorem: The total time to apply and dry three coats on nails is 120 minutes -/
theorem nail_decoration_theorem :
  nail_decoration_time 20 20 3 = 120 :=
by sorry

end NUMINAMATH_CALUDE_nail_decoration_theorem_l3999_399995


namespace NUMINAMATH_CALUDE_linear_function_problem_l3999_399942

/-- A linear function satisfying specific conditions -/
def f (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x + b

/-- The theorem statement -/
theorem linear_function_problem (a b : ℝ) :
  (∀ x, f a b x = 3 * (f a b).invFun x ^ 2 + 5) →
  f a b 0 = 2 →
  f a b 3 = 3 * Real.sqrt 5 + 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_problem_l3999_399942


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3999_399900

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (2/3) * x = (144/216) * (1/x) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3999_399900


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3999_399999

theorem complex_equation_solution (z : ℂ) :
  z / (z - Complex.I) = Complex.I → z = (1 : ℂ) / 2 + Complex.I / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3999_399999


namespace NUMINAMATH_CALUDE_common_factor_proof_l3999_399994

theorem common_factor_proof (a b : ℕ) : 
  (4 * a^2 * b^3).gcd (6 * a^3 * b) = 2 * a^2 * b :=
by sorry

end NUMINAMATH_CALUDE_common_factor_proof_l3999_399994


namespace NUMINAMATH_CALUDE_square_decomposition_l3999_399909

theorem square_decomposition (a : ℤ) :
  a^2 + 5*a + 7 = (a + 3) * (a + 2)^2 + (a + 2) * 1^2 := by
  sorry

end NUMINAMATH_CALUDE_square_decomposition_l3999_399909


namespace NUMINAMATH_CALUDE_johnny_signature_dish_count_l3999_399929

/-- Represents the number of times Johnny makes his signature crab dish in a day -/
def signature_dish_count : ℕ := sorry

/-- The amount of crab meat used in each signature dish (in pounds) -/
def crab_meat_per_dish : ℚ := 3/2

/-- The price of crab meat per pound (in dollars) -/
def crab_meat_price : ℕ := 8

/-- The total amount Johnny spends on crab meat in a week (in dollars) -/
def weekly_spending : ℕ := 1920

/-- The number of days Johnny's restaurant is closed in a week -/
def closed_days : ℕ := 3

/-- The number of days Johnny's restaurant is open in a week -/
def open_days : ℕ := 7 - closed_days

theorem johnny_signature_dish_count :
  signature_dish_count = 40 :=
by sorry

end NUMINAMATH_CALUDE_johnny_signature_dish_count_l3999_399929


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l3999_399911

-- Problem 1
theorem problem_one (a : ℝ) (α : ℝ) (h1 : a < 0) 
  (h2 : Real.cos α = -3 * a / (5 * (-a))) 
  (h3 : Real.sin α = 4 * a / (5 * (-a))) : 
  Real.sin α + 2 * Real.cos α = 2/5 := by sorry

-- Problem 2
theorem problem_two (β : ℝ) (h : Real.tan β = 2) : 
  Real.sin β ^ 2 + 2 * Real.sin β * Real.cos β = 8/5 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l3999_399911


namespace NUMINAMATH_CALUDE_sandy_marks_l3999_399983

theorem sandy_marks (correct_marks : ℕ) (incorrect_marks : ℕ) (total_sums : ℕ) (correct_sums : ℕ) 
  (h1 : correct_marks = 3)
  (h2 : incorrect_marks = 2)
  (h3 : total_sums = 30)
  (h4 : correct_sums = 22) :
  correct_marks * correct_sums - incorrect_marks * (total_sums - correct_sums) = 50 := by
  sorry

end NUMINAMATH_CALUDE_sandy_marks_l3999_399983


namespace NUMINAMATH_CALUDE_kenneth_distance_past_finish_line_l3999_399934

/-- A proof that Kenneth will be 10 yards past the finish line when Biff crosses it in a 500-yard race -/
theorem kenneth_distance_past_finish_line 
  (race_distance : ℝ) 
  (biff_speed : ℝ) 
  (kenneth_speed : ℝ) 
  (h1 : race_distance = 500)
  (h2 : biff_speed = 50)
  (h3 : kenneth_speed = 51) :
  kenneth_speed * (race_distance / biff_speed) - race_distance = 10 :=
by
  sorry

#check kenneth_distance_past_finish_line

end NUMINAMATH_CALUDE_kenneth_distance_past_finish_line_l3999_399934


namespace NUMINAMATH_CALUDE_infinite_fraction_equals_sqrt_15_l3999_399903

theorem infinite_fraction_equals_sqrt_15 :
  ∃ x : ℝ, x > 0 ∧ x = 3 + 5 / (2 + 5 / x) → x = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_infinite_fraction_equals_sqrt_15_l3999_399903


namespace NUMINAMATH_CALUDE_workshop_total_workers_l3999_399963

/-- Represents the workshop scenario with workers and their salaries -/
structure Workshop where
  avgSalary : ℕ
  technicianCount : ℕ
  technicianAvgSalary : ℕ
  supervisorAvgSalary : ℕ
  laborerAvgSalary : ℕ
  supervisorLaborerTotalSalary : ℕ

/-- Theorem stating that the total number of workers in the workshop is 38 -/
theorem workshop_total_workers (w : Workshop)
  (h1 : w.avgSalary = 9000)
  (h2 : w.technicianCount = 6)
  (h3 : w.technicianAvgSalary = 12000)
  (h4 : w.supervisorAvgSalary = 15000)
  (h5 : w.laborerAvgSalary = 6000)
  (h6 : w.supervisorLaborerTotalSalary = 270000) :
  ∃ (supervisorCount laborerCount : ℕ),
    w.technicianCount + supervisorCount + laborerCount = 38 :=
by sorry

end NUMINAMATH_CALUDE_workshop_total_workers_l3999_399963


namespace NUMINAMATH_CALUDE_long_tennis_players_l3999_399926

theorem long_tennis_players (total : ℕ) (football : ℕ) (both : ℕ) (neither : ℕ) :
  total = 40 →
  football = 26 →
  both = 17 →
  neither = 11 →
  ∃ long_tennis : ℕ,
    long_tennis = 20 ∧
    total = football + long_tennis - both + neither :=
by
  sorry

end NUMINAMATH_CALUDE_long_tennis_players_l3999_399926


namespace NUMINAMATH_CALUDE_chemistry_textbook_weight_l3999_399928

/-- The weight of Kelly's chemistry textbook in pounds -/
def chemistry_weight : ℝ := sorry

/-- The weight of Kelly's geometry textbook in pounds -/
def geometry_weight : ℝ := 0.625

theorem chemistry_textbook_weight :
  chemistry_weight = geometry_weight + 6.5 ∧ chemistry_weight = 7.125 := by sorry

end NUMINAMATH_CALUDE_chemistry_textbook_weight_l3999_399928


namespace NUMINAMATH_CALUDE_increasing_iff_a_gt_two_l3999_399971

-- Define the linear function
def f (a x : ℝ) : ℝ := (2*a - 4)*x + 3

-- State the theorem
theorem increasing_iff_a_gt_two :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a > 2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_iff_a_gt_two_l3999_399971


namespace NUMINAMATH_CALUDE_only_negative_number_l3999_399989

theorem only_negative_number (a b c d : ℝ) : 
  a = |(-2)| ∧ b = Real.sqrt 3 ∧ c = 0 ∧ d = -5 →
  (d < 0 ∧ a ≥ 0 ∧ b > 0 ∧ c = 0) := by sorry

end NUMINAMATH_CALUDE_only_negative_number_l3999_399989


namespace NUMINAMATH_CALUDE_right_triangle_complex_count_l3999_399965

/-- A complex number z satisfies the right triangle property if 0, z, and z^2 form a right triangle
    with the right angle at z. -/
def has_right_triangle_property (z : ℂ) : Prop :=
  z ≠ 0 ∧ 
  (0 : ℂ) ≠ z ∧ 
  z ≠ z^2 ∧
  (z - 0) * (z^2 - z) = 0

/-- There are exactly two complex numbers that satisfy the right triangle property. -/
theorem right_triangle_complex_count : 
  ∃! (s : Finset ℂ), (∀ z ∈ s, has_right_triangle_property z) ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_complex_count_l3999_399965


namespace NUMINAMATH_CALUDE_weight_of_new_person_l3999_399992

/-- The weight of the new person in a group where the average weight has increased --/
def new_person_weight (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) : ℝ :=
  old_weight + n * avg_increase

/-- Theorem: The weight of the new person in the given scenario is 78.5 kg --/
theorem weight_of_new_person :
  new_person_weight 9 1.5 65 = 78.5 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l3999_399992


namespace NUMINAMATH_CALUDE_ratio_equality_l3999_399925

variables {a b c : ℝ}

theorem ratio_equality (h1 : 7 * a = 8 * b) (h2 : 4 * a + 3 * c = 11 * b) (h3 : 2 * c - b = 5 * a) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : a / 8 = b / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3999_399925


namespace NUMINAMATH_CALUDE_f_seven_plus_f_nine_l3999_399980

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_seven_plus_f_nine (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 4)
  (h_odd : is_odd (fun x ↦ f (x - 1)))
  (h_f_one : f 1 = 1) : 
  f 7 + f 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_seven_plus_f_nine_l3999_399980


namespace NUMINAMATH_CALUDE_functional_relationship_max_profit_remaining_profit_range_l3999_399970

-- Define the constants and variables
def cost_price : ℝ := 40
def min_selling_price : ℝ := 44
def max_selling_price : ℝ := 52
def initial_sales : ℝ := 300
def price_increase : ℝ := 1
def sales_decrease : ℝ := 10
def donation : ℝ := 200
def min_remaining_profit : ℝ := 2200

-- Define the functional relationship
def sales (x : ℝ) : ℝ := -10 * x + 740

-- Define the profit function
def profit (x : ℝ) : ℝ := (sales x) * (x - cost_price)

-- State the theorems to be proved
theorem functional_relationship (x : ℝ) (h : min_selling_price ≤ x ∧ x ≤ max_selling_price) :
  sales x = -10 * x + 740 := by sorry

theorem max_profit :
  ∃ (max_x : ℝ), max_x = max_selling_price ∧
  ∀ (x : ℝ), min_selling_price ≤ x ∧ x ≤ max_selling_price →
  profit x ≤ profit max_x ∧ profit max_x = 2640 := by sorry

theorem remaining_profit_range :
  ∀ (x : ℝ), 50 ≤ x ∧ x ≤ 52 ↔ profit x - donation ≥ min_remaining_profit := by sorry

end NUMINAMATH_CALUDE_functional_relationship_max_profit_remaining_profit_range_l3999_399970


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l3999_399941

/-- Converts a number from base 11 to base 10 -/
def base11ToBase10 (n : ℕ) : ℤ := sorry

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (n : ℕ) : ℤ := sorry

/-- Represents the digit E in base 14 -/
def E : ℕ := 14

theorem base_conversion_subtraction :
  base11ToBase10 373 - base14ToBase10 (4 * 14 * 14 + E * 14 + 5) = -542 := by sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l3999_399941


namespace NUMINAMATH_CALUDE_prob_sum_18_correct_l3999_399918

/-- The number of faces on each die -/
def num_faces : ℕ := 7

/-- The target sum we're aiming for -/
def target_sum : ℕ := 18

/-- The number of dice being rolled -/
def num_dice : ℕ := 3

/-- The probability of rolling a sum of 18 with three 7-faced dice -/
def prob_sum_18 : ℚ := 4 / 343

/-- Theorem stating that the probability of rolling a sum of 18 
    with three 7-faced dice is 4/343 -/
theorem prob_sum_18_correct :
  prob_sum_18 = (num_favorable_outcomes : ℚ) / (num_faces ^ num_dice) :=
sorry

end NUMINAMATH_CALUDE_prob_sum_18_correct_l3999_399918


namespace NUMINAMATH_CALUDE_smallest_divisor_k_divisibility_at_126_smallest_k_is_126_l3999_399930

def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_divisor_k : ∀ k : ℕ, k > 0 → (∀ z : ℂ, f z ∣ (z^k - 1)) → k ≥ 126 :=
by sorry

theorem divisibility_at_126 : ∀ z : ℂ, f z ∣ (z^126 - 1) :=
by sorry

theorem smallest_k_is_126 : (∀ z : ℂ, f z ∣ (z^126 - 1)) ∧ 
  (∀ k : ℕ, k > 0 → k < 126 → ∃ z : ℂ, ¬(f z ∣ (z^k - 1))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_k_divisibility_at_126_smallest_k_is_126_l3999_399930


namespace NUMINAMATH_CALUDE_incircle_radius_of_special_triangle_l3999_399916

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the radius of the incircle is 2 under the following conditions:
    - a, b, c form an arithmetic sequence
    - c = 10
    - a cos A = b cos B
    - A ≠ B -/
theorem incircle_radius_of_special_triangle 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_arithmetic : ∃ (d : ℝ), b = a + d ∧ c = a + 2*d)
  (h_c : c = 10)
  (h_cos : a * Real.cos A = b * Real.cos B)
  (h_angle_neq : A ≠ B) :
  let s := (a + b + c) / 2
  (s - a) * (s - b) * (s - c) / s = 4 :=
by sorry

end NUMINAMATH_CALUDE_incircle_radius_of_special_triangle_l3999_399916


namespace NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l3999_399944

theorem cubic_polynomial_integer_root 
  (b c : ℚ) 
  (h1 : ∃ x : ℝ, x^3 + b*x + c = 0 ∧ x = 5 - Real.sqrt 11)
  (h2 : ∃ n : ℤ, (n : ℝ)^3 + b*(n : ℝ) + c = 0) :
  ∃ n : ℤ, (n : ℝ)^3 + b*(n : ℝ) + c = 0 ∧ n = -10 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l3999_399944


namespace NUMINAMATH_CALUDE_cube_diagonal_length_l3999_399924

theorem cube_diagonal_length (S : ℝ) (h : S = 864) :
  ∃ (d : ℝ), d = 12 * Real.sqrt 3 ∧ d^2 = 3 * (S / 6) :=
by sorry

end NUMINAMATH_CALUDE_cube_diagonal_length_l3999_399924


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3999_399955

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 5) - (x^6 + 2 * x^5 + x^4 - x^3 + 7) = 
  x^6 - x^5 + 2 * x^4 + 2 * x^3 - 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3999_399955


namespace NUMINAMATH_CALUDE_remaining_pictures_to_color_l3999_399932

/-- The number of pictures in each coloring book -/
def pictures_per_book : ℕ := 44

/-- The number of coloring books -/
def num_books : ℕ := 2

/-- The number of pictures already colored -/
def colored_pictures : ℕ := 20

/-- Theorem: Given two coloring books with 44 pictures each, and 20 pictures already colored,
    the number of pictures left to color is 68. -/
theorem remaining_pictures_to_color :
  (num_books * pictures_per_book) - colored_pictures = 68 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pictures_to_color_l3999_399932


namespace NUMINAMATH_CALUDE_bird_families_count_l3999_399954

/-- The number of bird families that flew to Africa -/
def families_to_africa : ℕ := 47

/-- The number of bird families that flew to Asia -/
def families_to_asia : ℕ := 94

/-- The difference between families that flew to Asia and Africa -/
def difference : ℕ := 47

/-- The total number of bird families before migration -/
def total_families : ℕ := families_to_africa + families_to_asia

theorem bird_families_count : 
  (families_to_asia = families_to_africa + difference) → 
  (total_families = 141) := by
  sorry

end NUMINAMATH_CALUDE_bird_families_count_l3999_399954


namespace NUMINAMATH_CALUDE_root_in_interval_l3999_399984

-- Define the function f(x) = x^3 + 3x - 1
def f (x : ℝ) : ℝ := x^3 + 3*x - 1

-- State the theorem
theorem root_in_interval :
  (f 0 < 0) → (f 1 > 0) → ∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l3999_399984


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l3999_399957

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- The original point -/
def original_point : ℝ × ℝ := (-2, 3)

/-- The reflected point -/
def reflected_point : ℝ × ℝ := (2, 3)

theorem reflection_across_y_axis :
  reflect_y original_point = reflected_point := by sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l3999_399957


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3999_399917

theorem triangle_abc_properties (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  4 * Real.sin A * Real.sin B - 4 * (Real.cos ((A - B) / 2))^2 = Real.sqrt 2 - 2 ∧
  a * Real.sin B / Real.sin A = 4 ∧
  1/2 * a * b * Real.sin C = 8 →
  C = π/4 ∧ c = 4 := by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3999_399917


namespace NUMINAMATH_CALUDE_square_vertex_C_l3999_399938

def square (A B C D : ℂ) : Prop :=
  (B - A) * Complex.I = C - B ∧
  (C - B) * Complex.I = D - C ∧
  (D - C) * Complex.I = A - D ∧
  (A - D) * Complex.I = B - A

theorem square_vertex_C (A B C D : ℂ) :
  square A B C D →
  A = 1 + 2*Complex.I →
  B = 3 - 5*Complex.I →
  C = 10 - 3*Complex.I :=
by
  sorry

#check square_vertex_C

end NUMINAMATH_CALUDE_square_vertex_C_l3999_399938


namespace NUMINAMATH_CALUDE_triangle_properties_l3999_399935

/-- Triangle ABC with given properties -/
structure TriangleABC where
  /-- Point B has coordinates (4,4) -/
  B : ℝ × ℝ
  B_coord : B = (4, 4)
  
  /-- The angle bisector of angle A lies on the line y=0 -/
  angle_bisector_A : ℝ → ℝ
  angle_bisector_A_eq : ∀ x, angle_bisector_A x = 0
  
  /-- The altitude from B to side AC lies on the line x-2y+2=0 -/
  altitude_B : ℝ → ℝ
  altitude_B_eq : ∀ x, altitude_B x = (x + 2) / 2

/-- The main theorem stating the properties of the triangle -/
theorem triangle_properties (t : TriangleABC) :
  ∃ (C : ℝ × ℝ) (area : ℝ),
    C = (10, -8) ∧ 
    area = 48 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3999_399935


namespace NUMINAMATH_CALUDE_train_passing_time_l3999_399960

/-- Proves that a train with given length and speed takes the calculated time to pass a fixed point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 300 ∧ train_speed_kmh = 90 → 
  (train_length / (train_speed_kmh * (5/18))) = 12 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l3999_399960


namespace NUMINAMATH_CALUDE_min_packs_needed_l3999_399972

def pack_sizes : List Nat := [8, 15, 30]

/-- The target number of cans to be purchased -/
def target_cans : Nat := 120

/-- A function to check if a combination of packs can achieve the target number of cans -/
def achieves_target (x y z : Nat) : Prop :=
  8 * x + 15 * y + 30 * z = target_cans

/-- The minimum number of packs needed -/
def min_packs : Nat := 4

theorem min_packs_needed : 
  (∃ x y z : Nat, achieves_target x y z) ∧ 
  (∀ x y z : Nat, achieves_target x y z → x + y + z ≥ min_packs) ∧
  (∃ x y z : Nat, achieves_target x y z ∧ x + y + z = min_packs) :=
sorry

end NUMINAMATH_CALUDE_min_packs_needed_l3999_399972


namespace NUMINAMATH_CALUDE_percentage_equation_solution_l3999_399967

theorem percentage_equation_solution : ∃ x : ℝ, 
  (x / 100 * 1442 - 36 / 100 * 1412) + 63 = 252 ∧ 
  abs (x - 33.52) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_solution_l3999_399967


namespace NUMINAMATH_CALUDE_reachable_region_characterization_l3999_399919

/-- A particle's position in a 2D plane -/
structure Particle where
  x : ℝ
  y : ℝ

/-- The speed of the particle along the x-axis -/
def x_speed : ℝ := 2

/-- The speed of the particle elsewhere -/
def other_speed : ℝ := 1

/-- The time limit for the particle's movement -/
def time_limit : ℝ := 1

/-- Check if a point is within the reachable region -/
def is_reachable (p : Particle) : Prop :=
  let o := Particle.mk 0 0
  let a := Particle.mk (1/2) (Real.sqrt 3 / 2)
  let b := Particle.mk 2 0
  let c := Particle.mk 1 0
  (p.x ≥ 0 ∧ p.y ≥ 0) ∧  -- First quadrant
  ((p.x ≤ 2 ∧ p.y ≤ (Real.sqrt 3 * (1 - p.x/2))) ∨  -- Triangle OAB
   (p.x^2 + p.y^2 ≤ 1 ∧ p.y ≥ 0 ∧ p.x ≥ p.y/Real.sqrt 3))  -- Sector OAC

/-- The main theorem stating that a point is reachable if and only if it's in the defined region -/
theorem reachable_region_characterization (p : Particle) :
  (∃ (path : ℝ → Particle), path 0 = Particle.mk 0 0 ∧
    (∀ t, 0 ≤ t ∧ t ≤ time_limit →
      (path t).x^2 + (path t).y^2 ≤ (x_speed * t)^2 ∨
      (path t).x^2 + (path t).y^2 ≤ (other_speed * t)^2) ∧
    path time_limit = p) ↔
  is_reachable p :=
sorry

end NUMINAMATH_CALUDE_reachable_region_characterization_l3999_399919


namespace NUMINAMATH_CALUDE_alyssa_soccer_spending_l3999_399901

/-- Calculates the total amount Alyssa spends on soccer games over three years -/
def total_soccer_spending (
  year1_games : ℕ)
  (year2_in_person : ℕ)
  (year2_missed : ℕ)
  (year2_online : ℕ)
  (year2_streaming_cost : ℕ)
  (year3_in_person : ℕ)
  (year3_online : ℕ)
  (year3_friends_games : ℕ)
  (year3_streaming_cost : ℕ)
  (ticket_price : ℕ) : ℕ :=
  let year1_cost := year1_games * ticket_price
  let year2_cost := year2_in_person * ticket_price + year2_streaming_cost
  let year3_cost := year3_in_person * ticket_price + year3_streaming_cost
  let friends_payback := year3_friends_games * 2 * ticket_price
  year1_cost + year2_cost + year3_cost - friends_payback

/-- Theorem stating that Alyssa's total spending on soccer games over three years is $850 -/
theorem alyssa_soccer_spending :
  total_soccer_spending 13 11 12 8 120 15 10 5 150 20 = 850 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_soccer_spending_l3999_399901


namespace NUMINAMATH_CALUDE_problem_proof_l3999_399987

theorem problem_proof (a b : ℝ) (h : (a - 5)^2 + |b^3 - 27| = 0) : a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l3999_399987


namespace NUMINAMATH_CALUDE_expression_value_l3999_399953

theorem expression_value : 
  let c : ℚ := 1
  (1 + c + 1/1) * (1 + c + 1/2) * (1 + c + 1/3) * (1 + c + 1/4) * (1 + c + 1/5) = 133/20 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3999_399953


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3999_399990

theorem polynomial_expansion (z : ℝ) : 
  (2 * z^2 + 5 * z - 6) * (3 * z^3 - 2 * z + 1) = 
  6 * z^5 + 15 * z^4 - 22 * z^3 - 8 * z^2 + 17 * z - 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3999_399990


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l3999_399931

/-- A circle C in the xy-plane -/
structure Circle where
  m : ℝ
  equation : ℝ → ℝ → Prop :=
    fun x y => x^2 + y^2 + m*x - 4 = 0

/-- A line in the xy-plane -/
def symmetry_line : ℝ → ℝ → Prop :=
  fun x y => x - y + 4 = 0

/-- Two points are symmetric with respect to a line -/
def symmetric (p1 p2 : ℝ × ℝ) (L : ℝ → ℝ → Prop) : Prop := sorry

theorem circle_symmetry_line (C : Circle) :
  (∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ 
    C.equation p1.1 p1.2 ∧ 
    C.equation p2.1 p2.2 ∧ 
    symmetric p1 p2 symmetry_line) →
  C.m = 8 := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l3999_399931


namespace NUMINAMATH_CALUDE_tobias_shoe_purchase_l3999_399905

/-- Tobias's shoe purchase problem -/
theorem tobias_shoe_purchase (shoe_cost : ℕ) (saving_months : ℕ) (monthly_allowance : ℕ)
  (lawn_charge : ℕ) (lawns_mowed : ℕ) (driveways_shoveled : ℕ) (change : ℕ)
  (h1 : shoe_cost = 95)
  (h2 : saving_months = 3)
  (h3 : monthly_allowance = 5)
  (h4 : lawn_charge = 15)
  (h5 : lawns_mowed = 4)
  (h6 : driveways_shoveled = 5)
  (h7 : change = 15) :
  ∃ (driveway_charge : ℕ),
    shoe_cost + change =
      saving_months * monthly_allowance +
      lawns_mowed * lawn_charge +
      driveways_shoveled * driveway_charge ∧
    driveway_charge = 7 :=
by sorry

end NUMINAMATH_CALUDE_tobias_shoe_purchase_l3999_399905


namespace NUMINAMATH_CALUDE_complex_moduli_equality_l3999_399921

theorem complex_moduli_equality (a : ℝ) : 
  let z₁ : ℂ := a + 2 * Complex.I
  let z₂ : ℂ := 2 - Complex.I
  Complex.abs z₁ = Complex.abs z₂ → a^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_moduli_equality_l3999_399921


namespace NUMINAMATH_CALUDE_sock_pair_count_l3999_399991

/-- The number of ways to choose a pair of socks with different colors -/
def different_color_pairs (white brown blue red : ℕ) : ℕ :=
  white * brown + white * blue + white * red +
  brown * blue + brown * red +
  blue * red

/-- Theorem: The number of ways to choose a pair of socks with different colors
    from a drawer containing 5 white socks, 5 brown socks, 3 blue socks,
    and 2 red socks is equal to 81. -/
theorem sock_pair_count :
  different_color_pairs 5 5 3 2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l3999_399991


namespace NUMINAMATH_CALUDE_surface_area_difference_l3999_399913

/-- The difference in surface area between 8 unit cubes and a cube with volume 8 -/
theorem surface_area_difference (large_cube_volume : ℝ) (small_cube_volume : ℝ) 
  (num_small_cubes : ℕ) (h1 : large_cube_volume = 8) (h2 : small_cube_volume = 1) 
  (h3 : num_small_cubes = 8) : 
  (num_small_cubes : ℝ) * (6 * small_cube_volume ^ (2/3)) - 
  (6 * large_cube_volume ^ (2/3)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_difference_l3999_399913


namespace NUMINAMATH_CALUDE_relative_minimum_condition_l3999_399920

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 - x^3 - x^2 + a*x + 1

/-- The first derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 4*x^3 - 3*x^2 - 2*x + a

/-- The second derivative of f(x) -/
def f'' (x : ℝ) : ℝ := 12*x^2 - 6*x - 2

/-- Theorem stating that f(a) = a is a relative minimum iff a = 1 -/
theorem relative_minimum_condition (a : ℝ) :
  (f a a = a ∧ ∀ x, x ≠ a → f a x ≥ f a a) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_relative_minimum_condition_l3999_399920


namespace NUMINAMATH_CALUDE_equation_represents_three_lines_lines_do_not_all_intersect_l3999_399981

-- Define the equation
def equation (x y : ℝ) : Prop :=
  x^2 * (x - y - 2) = y^2 * (x - y - 2)

-- Define the three lines
def line1 (x y : ℝ) : Prop := y = x
def line2 (x y : ℝ) : Prop := y = -x
def line3 (x y : ℝ) : Prop := y = x - 2

-- Theorem statement
theorem equation_represents_three_lines :
  ∀ x y : ℝ, equation x y ↔ (line1 x y ∨ line2 x y ∨ line3 x y) :=
by sorry

-- Theorem stating that the three lines do not all intersect at a common point
theorem lines_do_not_all_intersect :
  ¬∃ x y : ℝ, line1 x y ∧ line2 x y ∧ line3 x y :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_three_lines_lines_do_not_all_intersect_l3999_399981


namespace NUMINAMATH_CALUDE_down_payment_correct_l3999_399978

/-- Represents the down payment problem for a car purchase. -/
structure DownPayment where
  total : ℕ
  contributionA : ℕ
  contributionB : ℕ
  contributionC : ℕ
  contributionD : ℕ

/-- Theorem stating that the given contributions satisfy the problem conditions. -/
theorem down_payment_correct (dp : DownPayment) : 
  dp.total = 3500 ∧
  dp.contributionA = 1225 ∧
  dp.contributionB = 875 ∧
  dp.contributionC = 700 ∧
  dp.contributionD = 700 ∧
  dp.contributionA + dp.contributionB + dp.contributionC + dp.contributionD = dp.total ∧
  dp.contributionA = (35 * dp.total) / 100 ∧
  dp.contributionB = (25 * dp.total) / 100 ∧
  dp.contributionC = (20 * dp.total) / 100 ∧
  dp.contributionD = dp.total - (dp.contributionA + dp.contributionB + dp.contributionC) :=
by sorry


end NUMINAMATH_CALUDE_down_payment_correct_l3999_399978


namespace NUMINAMATH_CALUDE_second_to_first_ratio_l3999_399948

theorem second_to_first_ratio (x y z : ℝ) : 
  y = 90 →
  z = 4 * y →
  (x + y + z) / 3 = 165 →
  y / x = 2 := by
sorry

end NUMINAMATH_CALUDE_second_to_first_ratio_l3999_399948


namespace NUMINAMATH_CALUDE_purple_greater_than_green_less_than_triple_l3999_399959

-- Define the probability space
def prob_space : Type := Unit

-- Define the random variables
def X : prob_space → ℝ := sorry
def Y : prob_space → ℝ := sorry

-- Define the probability measure
def P : Set prob_space → ℝ := sorry

-- State the theorem
theorem purple_greater_than_green_less_than_triple (ω : prob_space) : 
  P {ω | X ω < Y ω ∧ Y ω < min (3 * X ω) 1} = 1/3 := by sorry

end NUMINAMATH_CALUDE_purple_greater_than_green_less_than_triple_l3999_399959


namespace NUMINAMATH_CALUDE_decimal_equals_fraction_l3999_399945

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ
  repeatLength : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (d : RepeatingDecimal) : ℚ :=
  d.integerPart + (d.repeatingPart : ℚ) / ((10 ^ d.repeatLength - 1) : ℚ)

/-- The repeating decimal 0.3045045045... -/
def decimal : RepeatingDecimal :=
  { integerPart := 0
  , repeatingPart := 3045
  , repeatLength := 4 }

theorem decimal_equals_fraction : toRational decimal = 383 / 1110 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equals_fraction_l3999_399945


namespace NUMINAMATH_CALUDE_cos105_cos45_plus_sin45_sin105_eq_half_l3999_399975

theorem cos105_cos45_plus_sin45_sin105_eq_half :
  Real.cos (105 * π / 180) * Real.cos (45 * π / 180) +
  Real.sin (45 * π / 180) * Real.sin (105 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos105_cos45_plus_sin45_sin105_eq_half_l3999_399975


namespace NUMINAMATH_CALUDE_dot_path_length_on_rolling_cube_l3999_399904

/-- The length of the path followed by a dot on a rolling cube -/
theorem dot_path_length_on_rolling_cube :
  ∀ (cube_edge : ℝ) (dot_path : ℝ),
  cube_edge = 2 →
  dot_path = 2 * Real.sqrt 2 * Real.pi →
  dot_path = (cube_edge * Real.sqrt 2) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_dot_path_length_on_rolling_cube_l3999_399904


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l3999_399974

theorem right_triangle_perimeter 
  (area : ℝ) 
  (leg : ℝ) 
  (h1 : area = 150) 
  (h2 : leg = 30) : 
  ∃ (other_leg hypotenuse : ℝ),
    area = (1/2) * leg * other_leg ∧
    hypotenuse^2 = leg^2 + other_leg^2 ∧
    leg + other_leg + hypotenuse = 40 + 10 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l3999_399974


namespace NUMINAMATH_CALUDE_root_value_theorem_l3999_399943

theorem root_value_theorem (a : ℝ) : 
  a^2 - 4*a + 3 = 0 → -2*a^2 + 8*a - 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l3999_399943


namespace NUMINAMATH_CALUDE_find_divisor_l3999_399969

theorem find_divisor (dividend : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 95 ∧ quotient = 6 ∧ remainder = 5 →
  ∃ divisor : Nat, dividend = divisor * quotient + remainder ∧ divisor = 15 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l3999_399969


namespace NUMINAMATH_CALUDE_number_of_girls_in_class_l3999_399951

/-- Proves the number of girls in a class given a specific ratio and total number of students -/
theorem number_of_girls_in_class (total : ℕ) (boy_ratio girl_ratio : ℕ) (h_total : total = 260) (h_ratio : boy_ratio = 5 ∧ girl_ratio = 8) :
  (girl_ratio * total) / (boy_ratio + girl_ratio) = 160 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_class_l3999_399951


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l3999_399952

/-- Given that the solution set of ax² + bx + c > 0 is {x | -3 < x < 2}, prove the following statements -/
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : ∀ x : ℝ, ax^2 + b*x + c > 0 ↔ -3 < x ∧ x < 2) :
  (a < 0) ∧
  (a + b + c > 0) ∧
  (∀ x : ℝ, c*x^2 + b*x + a < 0 ↔ -1/3 < x ∧ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l3999_399952


namespace NUMINAMATH_CALUDE_interest_rate_difference_l3999_399927

/-- Proves that the difference in interest rates is 1% given the problem conditions --/
theorem interest_rate_difference 
  (principal : ℝ) 
  (time : ℝ) 
  (additional_interest : ℝ) 
  (h1 : principal = 2400) 
  (h2 : time = 3) 
  (h3 : additional_interest = 72) : 
  ∃ (r dr : ℝ), 
    principal * ((r + dr) / 100) * time - principal * (r / 100) * time = additional_interest ∧ 
    dr = 1 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l3999_399927


namespace NUMINAMATH_CALUDE_juice_bottles_count_l3999_399976

theorem juice_bottles_count : ∃ x : ℕ, 
  let day0_remaining := x / 2 + 1
  let day1_remaining := day0_remaining / 2
  let day2_remaining := day1_remaining / 2 - 1
  x > 0 ∧ day2_remaining = 2 → x = 22 := by
  sorry

end NUMINAMATH_CALUDE_juice_bottles_count_l3999_399976


namespace NUMINAMATH_CALUDE_milk_price_decrease_is_60_percent_l3999_399962

/-- Represents the price change of milk powder and coffee from June to July -/
structure PriceChange where
  june_price : ℝ  -- Price of both milk powder and coffee in June
  coffee_increase : ℝ  -- Percentage increase in coffee price
  july_mixture_price : ℝ  -- Price of 3 lbs mixture in July
  july_milk_price : ℝ  -- Price of milk powder per pound in July

/-- Calculates the percentage decrease in milk powder price -/
def milk_price_decrease (pc : PriceChange) : ℝ :=
  -- We'll implement the calculation here
  sorry

/-- Theorem stating that given the conditions, the milk price decrease is 60% -/
theorem milk_price_decrease_is_60_percent (pc : PriceChange) 
  (h1 : pc.coffee_increase = 200)
  (h2 : pc.july_mixture_price = 5.1)
  (h3 : pc.july_milk_price = 0.4) : 
  milk_price_decrease pc = 60 := by
  sorry

end NUMINAMATH_CALUDE_milk_price_decrease_is_60_percent_l3999_399962


namespace NUMINAMATH_CALUDE_least_common_denominator_l3999_399985

theorem least_common_denominator (a b c d e f g h : ℕ) 
  (ha : a = 2) (hb : b = 3) (hc : c = 4) (hd : d = 5) 
  (he : e = 6) (hf : f = 7) (hg : g = 9) (hh : h = 10) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d (Nat.lcm e (Nat.lcm f (Nat.lcm g h)))))) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_l3999_399985


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_ratio_l3999_399946

/-- The ratio of areas between an inscribed hexagon with side length s/2 
    and an outer hexagon with side length s is 1/4 -/
theorem inscribed_hexagon_area_ratio (s : ℝ) (h : s > 0) : 
  (3 * Real.sqrt 3 * (s/2)^2 / 2) / (3 * Real.sqrt 3 * s^2 / 2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_ratio_l3999_399946


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l3999_399982

theorem tangent_point_coordinates :
  ∀ x y : ℝ,
  y = x^2 →
  (2 : ℝ) = 2*x →
  x = 1 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l3999_399982


namespace NUMINAMATH_CALUDE_a_range_l3999_399950

theorem a_range (a : ℝ) : a < 9 * a^3 - 11 * a ∧ 9 * a^3 - 11 * a < |a| → -2 * Real.sqrt 3 / 3 < a ∧ a < -Real.sqrt 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l3999_399950


namespace NUMINAMATH_CALUDE_male_students_count_l3999_399968

theorem male_students_count (total_students : ℕ) (sample_size : ℕ) (female_in_sample : ℕ) :
  total_students = 800 →
  sample_size = 40 →
  female_in_sample = 11 →
  (sample_size - female_in_sample) * total_students = 580 * sample_size :=
by sorry

end NUMINAMATH_CALUDE_male_students_count_l3999_399968


namespace NUMINAMATH_CALUDE_triangle_max_side_length_l3999_399937

/-- A triangle with three different integer side lengths and a perimeter of 24 units has a maximum side length of 11 units. -/
theorem triangle_max_side_length :
  ∀ a b c : ℕ,
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a + b + c = 24 →
  a ≤ 11 ∧ b ≤ 11 ∧ c ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_side_length_l3999_399937


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3999_399947

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 5 ∧ x^2 + a*x - 2 > 0) → a > -23/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3999_399947


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3999_399922

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  a 1 * a 3 = 4 →
  a 4 = 8 →
  (a 1 + q = 3 ∨ a 1 + q = -3) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3999_399922


namespace NUMINAMATH_CALUDE_bill_problem_count_bill_composes_twenty_l3999_399939

theorem bill_problem_count : ℕ → Prop :=
  fun b : ℕ =>
    let r := 2 * b  -- Ryan's problem count
    let f := 3 * r  -- Frank's problem count
    let types := 4  -- Number of problem types
    let frank_per_type := 30  -- Frank's problems per type
    f = types * frank_per_type → b = 20

-- Proof
theorem bill_composes_twenty : ∃ b : ℕ, bill_problem_count b :=
  sorry

end NUMINAMATH_CALUDE_bill_problem_count_bill_composes_twenty_l3999_399939


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3999_399964

theorem max_value_sqrt_sum (x : ℝ) (h : -36 ≤ x ∧ x ≤ 36) :
  Real.sqrt (36 + x) + Real.sqrt (36 - x) + x / 6 ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3999_399964


namespace NUMINAMATH_CALUDE_structure_has_112_cubes_l3999_399986

/-- A structure made of cubes with 5 layers -/
structure CubeStructure where
  middle_layer : ℕ
  other_layers : ℕ
  total_layers : ℕ
  h_middle : middle_layer = 16
  h_other : other_layers = 24
  h_total : total_layers = 5

/-- The total number of cubes in the structure -/
def total_cubes (s : CubeStructure) : ℕ :=
  s.middle_layer + (s.total_layers - 1) * s.other_layers

/-- Theorem stating that the structure contains 112 cubes -/
theorem structure_has_112_cubes (s : CubeStructure) : total_cubes s = 112 := by
  sorry


end NUMINAMATH_CALUDE_structure_has_112_cubes_l3999_399986
