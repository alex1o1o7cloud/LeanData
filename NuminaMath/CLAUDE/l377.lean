import Mathlib

namespace NUMINAMATH_CALUDE_somu_age_problem_l377_37727

/-- Proves that Somu was one-fifth of his father's age 6 years ago -/
theorem somu_age_problem (somu_age : ℕ) (father_age : ℕ) (years_ago : ℕ) : 
  somu_age = 12 →
  somu_age = father_age / 3 →
  somu_age - years_ago = (father_age - years_ago) / 5 →
  years_ago = 6 := by
sorry

end NUMINAMATH_CALUDE_somu_age_problem_l377_37727


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l377_37756

def M : Set ℝ := {x | |x| < 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l377_37756


namespace NUMINAMATH_CALUDE_max_value_x_plus_2y_l377_37770

theorem max_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2*x^2 + 8*y^2 + x*y = 2) : x + 2*y ≤ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_2y_l377_37770


namespace NUMINAMATH_CALUDE_tv_watching_time_l377_37769

/-- Given children watch 6 hours of television in 2 weeks and are allowed to watch 4 days a week,
    prove they spend 45 minutes each day watching television. -/
theorem tv_watching_time (hours_per_two_weeks : ℕ) (days_per_week : ℕ) 
    (h1 : hours_per_two_weeks = 6) 
    (h2 : days_per_week = 4) : 
  (hours_per_two_weeks * 60) / (days_per_week * 2) = 45 := by
  sorry

end NUMINAMATH_CALUDE_tv_watching_time_l377_37769


namespace NUMINAMATH_CALUDE_parabola_no_intersection_l377_37793

/-- A parabola is defined by the equation y = -x^2 - 6x + m -/
def parabola (x m : ℝ) : ℝ := -x^2 - 6*x + m

/-- The parabola does not intersect the x-axis if it has no real roots -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x : ℝ, parabola x m ≠ 0

/-- If the parabola does not intersect the x-axis, then m < -9 -/
theorem parabola_no_intersection (m : ℝ) :
  no_intersection m → m < -9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_no_intersection_l377_37793


namespace NUMINAMATH_CALUDE_numerator_increase_l377_37722

theorem numerator_increase (x y a : ℝ) : 
  x / y = 2 / 5 → 
  x + y = 5.25 → 
  (x + a) / (2 * y) = 1 / 3 → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_numerator_increase_l377_37722


namespace NUMINAMATH_CALUDE_soccer_camp_afternoon_count_l377_37758

theorem soccer_camp_afternoon_count (total_kids : ℕ) 
  (h1 : total_kids = 2000)
  (h2 : total_kids / 2 = total_kids / 2) -- Half of the kids are going to soccer camp
  (h3 : (total_kids / 2) / 4 = (total_kids / 2) / 4) -- 1/4 of the kids going to soccer camp are going in the morning
  : total_kids / 2 - (total_kids / 2) / 4 = 750 := by
  sorry

end NUMINAMATH_CALUDE_soccer_camp_afternoon_count_l377_37758


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_11_l377_37724

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

theorem smallest_positive_integer_ending_in_3_divisible_by_11 :
  ∃ (n : ℕ), n > 0 ∧ ends_in_3 n ∧ n % 11 = 0 ∧
  ∀ (m : ℕ), m > 0 → ends_in_3 m → m % 11 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_11_l377_37724


namespace NUMINAMATH_CALUDE_initial_walking_speed_l377_37760

theorem initial_walking_speed 
  (distance : Real) 
  (initial_speed : Real) 
  (fast_speed : Real) 
  (miss_time : Real) 
  (early_time : Real) 
  (h1 : distance = 2.2)
  (h2 : fast_speed = 6)
  (h3 : miss_time = 12 / 60)
  (h4 : early_time = 10 / 60)
  (h5 : distance / initial_speed - miss_time = distance / fast_speed + early_time) :
  initial_speed = 3 := by
sorry

end NUMINAMATH_CALUDE_initial_walking_speed_l377_37760


namespace NUMINAMATH_CALUDE_non_increasing_iff_exists_greater_l377_37735

open Set

-- Define the property of being an increasing function
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

-- Define the property of being a non-increasing function
def IsNonIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a ≤ x ∧ x < y ∧ y ≤ b ∧ f x > f y

-- Theorem statement
theorem non_increasing_iff_exists_greater (f : ℝ → ℝ) (a b : ℝ) :
  IsNonIncreasing f a b ↔ ¬(IsIncreasing f a b) :=
sorry

end NUMINAMATH_CALUDE_non_increasing_iff_exists_greater_l377_37735


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l377_37737

/-- Simple interest rate calculation -/
theorem simple_interest_rate_calculation
  (principal : ℝ)
  (time : ℝ)
  (final_amount : ℝ)
  (h1 : principal = 1000)
  (h2 : time = 3)
  (h3 : final_amount = 1300) :
  (final_amount - principal) / (principal * time) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l377_37737


namespace NUMINAMATH_CALUDE_product_evaluation_l377_37742

theorem product_evaluation (n : ℤ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l377_37742


namespace NUMINAMATH_CALUDE_sum_of_cubes_l377_37798

theorem sum_of_cubes (a b c : ℝ) 
  (sum_eq : a + b + c = 7)
  (sum_products_eq : a * b + a * c + b * c = 11)
  (product_eq : a * b * c = -18) :
  a^3 + b^3 + c^3 = 151 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l377_37798


namespace NUMINAMATH_CALUDE_a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq_l377_37752

theorem a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq :
  (∃ a b : ℝ, a = b → a^2 = b^2) ∧
  (∃ a b : ℝ, a^2 = b^2 ∧ a ≠ b) :=
by sorry

end NUMINAMATH_CALUDE_a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq_l377_37752


namespace NUMINAMATH_CALUDE_constant_function_proof_l377_37781

theorem constant_function_proof (g : ℝ → ℝ) 
  (h1 : ∃ x, g x ≠ 0)
  (h2 : ∀ a b : ℝ, g (a + b) + g (a - b) = g a + g b) :
  ∃ k : ℝ, ∀ x : ℝ, g x = k := by
  sorry

end NUMINAMATH_CALUDE_constant_function_proof_l377_37781


namespace NUMINAMATH_CALUDE_unique_correct_answers_l377_37723

/-- Scoring rules for the Intermediate Maths Challenge -/
structure ScoringRules where
  totalQuestions : Nat
  easyQuestions : Nat
  hardQuestions : Nat
  easyMarks : Nat
  hardMarks : Nat
  easyPenalty : Nat
  hardPenalty : Nat

/-- Calculate the total score based on the number of correct answers -/
def calculateScore (rules : ScoringRules) (correctAnswers : Nat) : Int :=
  sorry

/-- Theorem stating that given the scoring rules and a total score of 80,
    the only possible number of correct answers is 16 -/
theorem unique_correct_answers (rules : ScoringRules) :
  rules.totalQuestions = 25 →
  rules.easyQuestions = 15 →
  rules.hardQuestions = 10 →
  rules.easyMarks = 5 →
  rules.hardMarks = 6 →
  rules.easyPenalty = 1 →
  rules.hardPenalty = 2 →
  ∃! (correctAnswers : Nat), calculateScore rules correctAnswers = 80 ∧ correctAnswers = 16 :=
sorry

end NUMINAMATH_CALUDE_unique_correct_answers_l377_37723


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l377_37771

theorem purely_imaginary_complex_number (a : ℝ) : 
  (∀ z : ℂ, z = (a^2 - 3*a + 2 : ℝ) + (a - 1 : ℝ)*I → z.re = 0 ∧ z.im ≠ 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l377_37771


namespace NUMINAMATH_CALUDE_perimeter_of_square_III_l377_37720

/-- Given three squares I, II, and III, prove that the perimeter of III is 36. -/
theorem perimeter_of_square_III (I II III : Real) : 
  (I > 0) →  -- I is positive (side length of a square)
  (II > 0) → -- II is positive (side length of a square)
  (4 * I = 12) → -- Perimeter of I is 12
  (4 * II = 24) → -- Perimeter of II is 24
  (III = I + II) → -- Side length of III is sum of side lengths of I and II
  (4 * III = 36) := by -- Perimeter of III is 36
sorry

end NUMINAMATH_CALUDE_perimeter_of_square_III_l377_37720


namespace NUMINAMATH_CALUDE_greg_travel_distance_l377_37714

/-- Greg's travel problem -/
theorem greg_travel_distance :
  let workplace_to_market : ℝ := 30  -- Distance in miles
  let market_to_home_time : ℝ := 0.5  -- Time in hours
  let market_to_home_speed : ℝ := 20  -- Speed in miles per hour
  let market_to_home : ℝ := market_to_home_speed * market_to_home_time
  let total_distance : ℝ := workplace_to_market + market_to_home
  total_distance = 40 := by
sorry


end NUMINAMATH_CALUDE_greg_travel_distance_l377_37714


namespace NUMINAMATH_CALUDE_collinear_vectors_l377_37729

def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![-2, -1]
def c : Fin 2 → ℝ := ![1, 2]

def is_collinear (v w : Fin 2 → ℝ) : Prop :=
  ∃ t : ℝ, v 0 * w 1 = t * v 1 * w 0

theorem collinear_vectors (k : ℝ) :
  is_collinear (fun i => a i + k * b i) c ↔ k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_l377_37729


namespace NUMINAMATH_CALUDE_triangle_area_l377_37734

theorem triangle_area (a b c : ℝ) (h_ratio : (a, b, c) = (5 * k, 12 * k, 13 * k) → k > 0) 
  (h_perimeter : a + b + c = 60) : (a * b : ℝ) / 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l377_37734


namespace NUMINAMATH_CALUDE_triangle_problem_l377_37703

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (S : ℝ) (CM : ℝ) :
  b * (3 * b - c) * Real.cos A = b * a * Real.cos C →
  S = 2 * Real.sqrt 2 →
  CM = Real.sqrt 17 / 2 →
  (Real.cos A = 1 / 3) ∧
  ((b = 2 ∧ c = 3) ∨ (b = 3 / 2 ∧ c = 4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l377_37703


namespace NUMINAMATH_CALUDE_notebooks_cost_20_dollars_l377_37784

/-- Represents the purchase of notebooks and pens --/
structure Purchase where
  num_pens : ℕ
  cost_per_pen : ℕ
  total_paid : ℕ

/-- Calculates the amount spent on notebooks --/
def amount_spent_on_notebooks (p : Purchase) : ℕ :=
  p.total_paid - (p.num_pens * p.cost_per_pen)

/-- Theorem: The amount spent on notebooks is 20 dollars --/
theorem notebooks_cost_20_dollars (p : Purchase) 
  (h1 : p.num_pens = 5)
  (h2 : p.cost_per_pen = 2)
  (h3 : p.total_paid = 30) :
  amount_spent_on_notebooks p = 20 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_cost_20_dollars_l377_37784


namespace NUMINAMATH_CALUDE_highest_power_of_three_dividing_N_l377_37721

def N : ℕ := sorry

theorem highest_power_of_three_dividing_N : 
  (∃ m : ℕ, N = 3 * m) ∧ ¬(∃ m : ℕ, N = 9 * m) := by sorry

end NUMINAMATH_CALUDE_highest_power_of_three_dividing_N_l377_37721


namespace NUMINAMATH_CALUDE_average_of_first_16_even_divisible_by_5_l377_37706

def first_16_even_divisible_by_5 : List Nat :=
  List.range 16 |> List.map (fun n => 10 * (n + 1))

theorem average_of_first_16_even_divisible_by_5 :
  (List.sum first_16_even_divisible_by_5) / first_16_even_divisible_by_5.length = 85 := by
  sorry

end NUMINAMATH_CALUDE_average_of_first_16_even_divisible_by_5_l377_37706


namespace NUMINAMATH_CALUDE_divisibility_property_l377_37788

theorem divisibility_property (n : ℕ) : (n - 1) ∣ (n^2 + n - 2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l377_37788


namespace NUMINAMATH_CALUDE_cone_base_radius_l377_37774

/-- Given a cone whose lateral surface is formed by a sector with radius 6cm and central angle 120°,
    the radius of the base of the cone is 2cm. -/
theorem cone_base_radius (r : ℝ) : r > 0 → 2 * π * r = 120 * π * 6 / 180 → r = 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l377_37774


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l377_37777

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : ∀ x y, x < y → f x > f y) 
  (h_inequality : f (1 - a) < f (2 * a - 1)) : 
  a < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l377_37777


namespace NUMINAMATH_CALUDE_total_accidents_across_highways_l377_37712

/-- Represents the accident rate and traffic data for a highway -/
structure HighwayData where
  accidents : ℕ
  vehicles : ℕ
  totalTraffic : ℕ

/-- Calculates the number of accidents for a given highway -/
def calculateAccidents (data : HighwayData) : ℕ :=
  (data.accidents * data.totalTraffic) / data.vehicles

/-- The data for Highway A -/
def highwayA : HighwayData :=
  { accidents := 75, vehicles := 100000000, totalTraffic := 2500000000 }

/-- The data for Highway B -/
def highwayB : HighwayData :=
  { accidents := 50, vehicles := 80000000, totalTraffic := 1600000000 }

/-- The data for Highway C -/
def highwayC : HighwayData :=
  { accidents := 90, vehicles := 200000000, totalTraffic := 1900000000 }

/-- Theorem stating that the total number of accidents across all three highways is 3730 -/
theorem total_accidents_across_highways :
  calculateAccidents highwayA + calculateAccidents highwayB + calculateAccidents highwayC = 3730 :=
by
  sorry

end NUMINAMATH_CALUDE_total_accidents_across_highways_l377_37712


namespace NUMINAMATH_CALUDE_basketball_volleyball_problem_l377_37757

/-- Given the conditions of the basketball and volleyball purchase problem,
    prove the prices of the balls and the minimum total cost. -/
theorem basketball_volleyball_problem
  (basketball_price volleyball_price : ℕ)
  (total_balls min_cost : ℕ) :
  (3 * basketball_price + volleyball_price = 360) →
  (5 * basketball_price + 3 * volleyball_price = 680) →
  (total_balls = 100) →
  (∀ x y, x + y = total_balls → x ≥ 3 * y → 
    basketball_price * x + volleyball_price * y ≥ min_cost) →
  (basketball_price = 100 ∧ 
   volleyball_price = 60 ∧
   min_cost = 9000) :=
by sorry

end NUMINAMATH_CALUDE_basketball_volleyball_problem_l377_37757


namespace NUMINAMATH_CALUDE_not_red_ball_percentage_is_52_5_percent_l377_37741

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  cube_percentage : ℝ
  red_ball_percentage : ℝ

/-- Calculates the percentage of objects in the urn that are not red balls -/
def not_red_ball_percentage (urn : UrnComposition) : ℝ :=
  (1 - urn.cube_percentage) * (1 - urn.red_ball_percentage)

/-- Theorem stating that the percentage of objects in the urn that are not red balls is 52.5% -/
theorem not_red_ball_percentage_is_52_5_percent (urn : UrnComposition)
  (h1 : urn.cube_percentage = 0.3)
  (h2 : urn.red_ball_percentage = 0.25) :
  not_red_ball_percentage urn = 0.525 := by
  sorry

end NUMINAMATH_CALUDE_not_red_ball_percentage_is_52_5_percent_l377_37741


namespace NUMINAMATH_CALUDE_simplify_expression_l377_37749

theorem simplify_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (sum_eq_one : a + b + c = 1) :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l377_37749


namespace NUMINAMATH_CALUDE_train_length_l377_37787

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (crossing_time : ℝ) : 
  speed_kmh = 144 → crossing_time = 5 → 
  speed_kmh * (1000 / 3600) * crossing_time = 200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l377_37787


namespace NUMINAMATH_CALUDE_diplomats_not_speaking_russian_count_l377_37799

/-- Represents the diplomats at a summit conference -/
structure DiplomatGroup where
  total : Nat
  latin_speakers : Nat
  neither_latin_nor_russian : Nat
  both_latin_and_russian : Nat

/-- Calculates the number of diplomats who did not speak Russian -/
def diplomats_not_speaking_russian (d : DiplomatGroup) : Nat :=
  d.total - (d.total - d.neither_latin_nor_russian - d.latin_speakers + d.both_latin_and_russian)

/-- Theorem stating the number of diplomats who did not speak Russian -/
theorem diplomats_not_speaking_russian_count :
  ∃ (d : DiplomatGroup),
    d.total = 120 ∧
    d.latin_speakers = 20 ∧
    d.neither_latin_nor_russian = (20 * d.total) / 100 ∧
    d.both_latin_and_russian = (10 * d.total) / 100 ∧
    diplomats_not_speaking_russian d = 20 := by
  sorry

end NUMINAMATH_CALUDE_diplomats_not_speaking_russian_count_l377_37799


namespace NUMINAMATH_CALUDE_complex_magnitude_l377_37708

theorem complex_magnitude (z : ℂ) : z * (1 - Complex.I) = Complex.I → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l377_37708


namespace NUMINAMATH_CALUDE_function_and_tangent_line_properties_l377_37789

noncomputable section

-- Define the constant e
def e : ℝ := Real.exp 1

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2

-- Define the tangent line function
def tangentLine (b : ℝ) (x : ℝ) : ℝ := (e - 2) * x + b

theorem function_and_tangent_line_properties :
  ∃ (a b : ℝ),
    (∀ x : ℝ, tangentLine b x = (Real.exp 1 - f a 1) + (Real.exp 1 - 2 * a) * (x - 1)) ∧
    a = 1 ∧
    b = 1 ∧
    (∀ x : ℝ, x ≥ 0 → f a x > x^2 + 4*x - 14) :=
sorry

end NUMINAMATH_CALUDE_function_and_tangent_line_properties_l377_37789


namespace NUMINAMATH_CALUDE_probability_monotonic_increasing_l377_37719

def cube_faces : Finset ℤ := {-2, -1, 0, 1, 2, 3}

def is_monotonic_increasing (a b : ℤ) : Prop :=
  a ≥ 0 ∧ b ≥ 0

def favorable_outcomes : Finset (ℤ × ℤ) :=
  (cube_faces.filter (λ x => x ≥ 0)).product (cube_faces.filter (λ x => x ≥ 0))

def total_outcomes : Finset (ℤ × ℤ) :=
  cube_faces.product cube_faces

theorem probability_monotonic_increasing :
  (favorable_outcomes.card : ℚ) / (total_outcomes.card : ℚ) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_monotonic_increasing_l377_37719


namespace NUMINAMATH_CALUDE_lottery_probability_l377_37747

/-- The number of people participating in the lottery drawing event -/
def num_people : ℕ := 5

/-- The total number of tickets in the box -/
def total_tickets : ℕ := 5

/-- The number of winning tickets -/
def winning_tickets : ℕ := 3

/-- The probability of drawing exactly 2 winning tickets in the first 3 draws
    and the last winning ticket on the 4th draw -/
def event_probability : ℚ := 3 / 10

/-- Theorem stating that the probability of the event ending exactly after
    the 4th person has drawn is 3/10 -/
theorem lottery_probability :
  (num_people = 5) →
  (total_tickets = 5) →
  (winning_tickets = 3) →
  (event_probability = 3 / 10) :=
by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_l377_37747


namespace NUMINAMATH_CALUDE_tu_yuan_yuan_theorem_l377_37739

/-- Represents the purchase and sale of "Tu Yuan Yuan" toys -/
structure ToyPurchase where
  first_cost : ℕ
  second_cost : ℕ
  price_increase : ℕ
  min_profit : ℕ

/-- Calculates the quantity of the first purchase -/
def first_quantity (tp : ToyPurchase) : ℕ :=
  sorry

/-- Calculates the minimum selling price -/
def min_selling_price (tp : ToyPurchase) : ℕ :=
  sorry

/-- Theorem stating the correct quantity and minimum selling price -/
theorem tu_yuan_yuan_theorem (tp : ToyPurchase) 
  (h1 : tp.first_cost = 1500)
  (h2 : tp.second_cost = 3500)
  (h3 : tp.price_increase = 5)
  (h4 : tp.min_profit = 1150) :
  first_quantity tp = 50 ∧ min_selling_price tp = 41 := by
  sorry

end NUMINAMATH_CALUDE_tu_yuan_yuan_theorem_l377_37739


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l377_37763

theorem system_of_equations_solution (x y m : ℝ) : 
  (3 * x - y = 4 * m + 1) → 
  (x + y = 2 * m - 5) → 
  (x - y = 4) → 
  (m = 1) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l377_37763


namespace NUMINAMATH_CALUDE_inequality_solution_set_l377_37740

theorem inequality_solution_set : 
  {x : ℝ | x / (x - 1) + (x + 2) / (2 * x) ≥ 3} = 
  {x : ℝ | (0 < x ∧ x ≤ 1/3) ∨ (1 < x ∧ x ≤ 2)} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l377_37740


namespace NUMINAMATH_CALUDE_sum_equals_222_l377_37762

theorem sum_equals_222 : 148 + 35 + 17 + 13 + 9 = 222 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_222_l377_37762


namespace NUMINAMATH_CALUDE_functional_equation_solution_l377_37725

/-- A function satisfying the given functional equation for all integers -/
def SatisfiesFunctionalEq (f : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, f (f m + n) + f m = f n + f (3 * m) + 2014

/-- The theorem stating that any function satisfying the functional equation
    must be of the form f(n) = 2n + 1007 -/
theorem functional_equation_solution :
  ∀ f : ℤ → ℤ, SatisfiesFunctionalEq f → ∀ n : ℤ, f n = 2 * n + 1007 :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l377_37725


namespace NUMINAMATH_CALUDE_abigail_lost_money_l377_37700

def money_lost (initial_amount spent_amount remaining_amount : ℕ) : ℕ :=
  initial_amount - spent_amount - remaining_amount

theorem abigail_lost_money : money_lost 11 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_abigail_lost_money_l377_37700


namespace NUMINAMATH_CALUDE_diagonal_length_range_l377_37755

/-- Represents a quadrilateral with given side lengths and an integer diagonal -/
structure Quadrilateral :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (HE : ℝ)
  (EG : ℤ)

/-- The theorem stating the possible values for the diagonal EG -/
theorem diagonal_length_range (q : Quadrilateral)
  (h1 : q.EF = 7)
  (h2 : q.FG = 12)
  (h3 : q.GH = 7)
  (h4 : q.HE = 15) :
  9 ≤ q.EG ∧ q.EG ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_diagonal_length_range_l377_37755


namespace NUMINAMATH_CALUDE_inequality_solution_l377_37711

theorem inequality_solution (x : ℝ) : 
  (x^2 - 3*x + 3)^(4*x^3 + 5*x^2) ≤ (x^2 - 3*x + 3)^(2*x^3 + 18*x) ↔ 
  x ≤ -9/2 ∨ (0 ≤ x ∧ x ≤ 1) ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l377_37711


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l377_37716

/-- An arithmetic sequence with sum of first n terms Sn = n^2 + bn + c -/
structure ArithmeticSequence where
  b : ℝ
  c : ℝ
  sum : ℕ+ → ℝ
  sum_eq : ∀ n : ℕ+, sum n = n.val ^ 2 + b * n.val + c

/-- The second term of the arithmetic sequence -/
def ArithmeticSequence.a2 (seq : ArithmeticSequence) : ℝ :=
  seq.sum 2 - seq.sum 1

/-- The third term of the arithmetic sequence -/
def ArithmeticSequence.a3 (seq : ArithmeticSequence) : ℝ :=
  seq.sum 3 - seq.sum 2

theorem arithmetic_sequence_theorem (seq : ArithmeticSequence) 
  (h : seq.a2 + seq.a3 = 4) : 
  seq.c = 0 ∧ seq.b = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l377_37716


namespace NUMINAMATH_CALUDE_triangle_numbers_exist_l377_37732

theorem triangle_numbers_exist : 
  ∃ (a b c d e f g : ℕ), 
    (b = c * d) ∧ 
    (e - f = a + c * d - a * c) ∧ 
    (e - f = g) ∧ 
    (g = a + d) ∧ 
    (c > 0) ∧
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧
    (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧
    (d ≠ e) ∧ (d ≠ f) ∧ (d ≠ g) ∧
    (e ≠ f) ∧ (e ≠ g) ∧
    (f ≠ g) := by
  sorry

end NUMINAMATH_CALUDE_triangle_numbers_exist_l377_37732


namespace NUMINAMATH_CALUDE_white_balls_count_l377_37795

theorem white_balls_count (red blue white : ℕ) : 
  red = 80 → blue = 40 → red = blue + white - 12 → white = 52 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l377_37795


namespace NUMINAMATH_CALUDE_max_sum_four_numbers_l377_37733

theorem max_sum_four_numbers (a b c d : ℕ) :
  a < b → b < c → c < d →
  (b + d) + (c + d) + (a + b + c) + (a + b + d) = 2017 →
  a + b + c + d ≤ 806 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_four_numbers_l377_37733


namespace NUMINAMATH_CALUDE_samantha_routes_l377_37765

/-- The number of ways to arrange n blocks in two directions --/
def arrangeBlocks (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- The number of diagonal paths through the park --/
def diagonalPaths : ℕ := 2

/-- The total number of routes Samantha can take --/
def totalRoutes : ℕ := arrangeBlocks 3 * diagonalPaths * arrangeBlocks 3

theorem samantha_routes :
  totalRoutes = 800 := by
  sorry

end NUMINAMATH_CALUDE_samantha_routes_l377_37765


namespace NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_equals_two_l377_37713

theorem sqrt_eight_div_sqrt_two_equals_two :
  Real.sqrt 8 / Real.sqrt 2 = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_equals_two_l377_37713


namespace NUMINAMATH_CALUDE_sandals_sold_l377_37764

theorem sandals_sold (shoes : ℕ) (sandals : ℕ) : 
  (9 : ℚ) / 5 = shoes / sandals → shoes = 72 → sandals = 40 := by
  sorry

end NUMINAMATH_CALUDE_sandals_sold_l377_37764


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l377_37783

-- Define a parallelogram
def Parallelogram : Type := sorry

-- Define the property of having equal diagonals
def has_equal_diagonals (p : Parallelogram) : Prop := sorry

-- Define the property of diagonals bisecting each other
def diagonals_bisect_each_other (p : Parallelogram) : Prop := sorry

-- State the theorem
theorem negation_of_universal_proposition :
  (¬ ∀ p : Parallelogram, has_equal_diagonals p ∧ diagonals_bisect_each_other p) ↔
  (∃ p : Parallelogram, ¬(has_equal_diagonals p ∧ diagonals_bisect_each_other p)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l377_37783


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l377_37748

/-- An arithmetic sequence is determined by its first term and common difference -/
def arithmeticSequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem eighth_term_of_sequence (a : ℝ) (d : ℝ) :
  arithmeticSequence a d 4 = 23 →
  arithmeticSequence a d 6 = 47 →
  arithmeticSequence a d 8 = 71 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l377_37748


namespace NUMINAMATH_CALUDE_simplify_expression_l377_37779

theorem simplify_expression (b : ℝ) : 3*b*(3*b^3 - 2*b) - 4*b^2 = 9*b^4 - 10*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l377_37779


namespace NUMINAMATH_CALUDE_value_of_a_l377_37782

theorem value_of_a (S T : Set ℕ) (a : ℕ) 
  (h1 : S = {1, 2})
  (h2 : T = {a})
  (h3 : S ∪ T = S) : 
  a = 1 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_value_of_a_l377_37782


namespace NUMINAMATH_CALUDE_non_holiday_rate_correct_l377_37786

/-- The number of customers per hour during the non-holiday season -/
def non_holiday_rate : ℕ := 175

/-- The number of customers per hour during the holiday season -/
def holiday_rate : ℕ := non_holiday_rate * 2

/-- The total number of customers during the holiday season -/
def total_customers : ℕ := 2800

/-- The number of hours observed during the holiday season -/
def observation_hours : ℕ := 8

/-- Theorem stating that the non-holiday rate is correct given the conditions -/
theorem non_holiday_rate_correct : 
  holiday_rate * observation_hours = total_customers ∧
  non_holiday_rate = 175 := by
  sorry

end NUMINAMATH_CALUDE_non_holiday_rate_correct_l377_37786


namespace NUMINAMATH_CALUDE_family_age_theorem_l377_37796

/-- Calculates the average age of a family given initial conditions --/
def average_family_age (initial_average_age : ℚ) (years_passed : ℕ) (child_age : ℕ) : ℚ :=
  let initial_total_age := initial_average_age * 2
  let current_total_age := initial_total_age + years_passed * 2 + child_age
  current_total_age / 3

/-- Proves that the average age of the family is 19 years --/
theorem family_age_theorem (initial_average_age : ℚ) (years_passed : ℕ) (child_age : ℕ)
  (h1 : initial_average_age = 23)
  (h2 : years_passed = 5)
  (h3 : child_age = 1) :
  average_family_age initial_average_age years_passed child_age = 19 := by
  sorry

#eval average_family_age 23 5 1

end NUMINAMATH_CALUDE_family_age_theorem_l377_37796


namespace NUMINAMATH_CALUDE_perpendicular_lines_l377_37766

/-- The slope of the line 2x - 3y + 5 = 0 -/
def m₁ : ℚ := 2 / 3

/-- The slope of the line bx - 3y + 1 = 0 -/
def m₂ (b : ℚ) : ℚ := b / 3

/-- The condition for perpendicular lines -/
def perpendicular (m₁ m₂ : ℚ) : Prop := m₁ * m₂ = -1

theorem perpendicular_lines (b : ℚ) : 
  perpendicular m₁ (m₂ b) → b = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l377_37766


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l377_37780

def A : Set Int := {-1, 2, 4}
def B : Set Int := {-1, 0, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l377_37780


namespace NUMINAMATH_CALUDE_tile_problem_l377_37753

theorem tile_problem (n : ℕ) (total_tiles : ℕ) : 
  (total_tiles = n^2 + 64) ∧ (total_tiles = (n+1)^2 - 25) → total_tiles = 2000 := by
sorry

end NUMINAMATH_CALUDE_tile_problem_l377_37753


namespace NUMINAMATH_CALUDE_symmetric_point_coords_l377_37792

/-- A point in a 2D plane represented by its x and y coordinates. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the y-axis. -/
def symmetricYAxis (a b : Point2D) : Prop :=
  b.x = -a.x ∧ b.y = a.y

/-- Theorem: If point B is symmetric to point A(2, -1) with respect to the y-axis,
    then the coordinates of point B are (-2, -1). -/
theorem symmetric_point_coords :
  let a : Point2D := ⟨2, -1⟩
  let b : Point2D := ⟨-2, -1⟩
  symmetricYAxis a b → b = ⟨-2, -1⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coords_l377_37792


namespace NUMINAMATH_CALUDE_cookie_problem_l377_37767

/-- The number of cookies in each box -/
def cookies_per_box : ℕ := 12

/-- The number of boxes -/
def num_boxes : ℕ := 8

/-- The number of bags -/
def num_bags : ℕ := 9

/-- The difference in cookies between boxes and bags -/
def cookie_difference : ℕ := 33

/-- The number of cookies in each bag -/
def cookies_per_bag : ℕ := 7

theorem cookie_problem :
  cookies_per_box * num_boxes = cookies_per_bag * num_bags + cookie_difference :=
by sorry

end NUMINAMATH_CALUDE_cookie_problem_l377_37767


namespace NUMINAMATH_CALUDE_agno3_mass_fraction_l377_37772

/-- Given the number of moles, molar mass, and total solution mass of AgNO₃,
    prove that its mass fraction in the solution is 8%. -/
theorem agno3_mass_fraction :
  ∀ (n M m_total : ℝ),
  n = 0.12 →
  M = 170 →
  m_total = 255 →
  let m := n * M
  let ω := m * 100 / m_total
  ω = 8 := by
sorry

end NUMINAMATH_CALUDE_agno3_mass_fraction_l377_37772


namespace NUMINAMATH_CALUDE_f_2_neg3_neg1_eq_half_l377_37751

def f (a b c : ℚ) : ℚ := (c + a) / (c - b)

theorem f_2_neg3_neg1_eq_half : f 2 (-3) (-1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_2_neg3_neg1_eq_half_l377_37751


namespace NUMINAMATH_CALUDE_field_trip_students_l377_37710

theorem field_trip_students (van_capacity : ℕ) (num_vans : ℕ) (num_adults : ℕ) : 
  van_capacity = 4 → num_vans = 2 → num_adults = 6 → 
  van_capacity * num_vans - num_adults = 2 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_students_l377_37710


namespace NUMINAMATH_CALUDE_symmetry_about_yOz_plane_l377_37730

/-- The symmetry of a point about the yOz plane in a rectangular coordinate system -/
theorem symmetry_about_yOz_plane (x y z : ℝ) : 
  let original_point := (x, y, z)
  let symmetric_point := (-x, y, z)
  symmetric_point = (- (x : ℝ), y, z) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_about_yOz_plane_l377_37730


namespace NUMINAMATH_CALUDE_min_guesses_bound_one_guess_sufficient_two_guesses_necessary_l377_37746

/-- Given positive integers n and k with n > k, this function returns the minimum number
    of guesses required to determine a binary string of length n, given all binary strings
    that differ from it in exactly k positions. -/
def min_guesses (n k : ℕ) : ℕ :=
  if n = 2 * k then 2 else 1

/-- Theorem stating that the minimum number of guesses is at most 2 and at least 1. -/
theorem min_guesses_bound (n k : ℕ) (h : n > k) :
  min_guesses n k = max 1 2 := by
  sorry

/-- Theorem stating that when n ≠ 2k, one guess is sufficient. -/
theorem one_guess_sufficient (n k : ℕ) (h1 : n > k) (h2 : n ≠ 2 * k) :
  min_guesses n k = 1 := by
  sorry

/-- Theorem stating that when n = 2k, two guesses are necessary and sufficient. -/
theorem two_guesses_necessary (n k : ℕ) (h1 : n > k) (h2 : n = 2 * k) :
  min_guesses n k = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_guesses_bound_one_guess_sufficient_two_guesses_necessary_l377_37746


namespace NUMINAMATH_CALUDE_teacher_assignment_schemes_l377_37705

theorem teacher_assignment_schemes (male_teachers : Nat) (female_teachers : Nat) : 
  male_teachers = 5 → 
  female_teachers = 4 → 
  (Nat.factorial 9 / Nat.factorial 6) - 
  (Nat.factorial 5 / Nat.factorial 2 + Nat.factorial 4 / Nat.factorial 1) = 420 := by
  sorry

end NUMINAMATH_CALUDE_teacher_assignment_schemes_l377_37705


namespace NUMINAMATH_CALUDE_quadratic_roots_real_distinct_l377_37707

theorem quadratic_roots_real_distinct (d : ℝ) : 
  let a : ℝ := 3
  let b : ℝ := -4 * Real.sqrt 3
  let c : ℝ := d
  let discriminant : ℝ := b^2 - 4*a*c
  discriminant = 12 →
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_real_distinct_l377_37707


namespace NUMINAMATH_CALUDE_kola_age_is_16_l377_37717

/-- Kola's current age -/
def kola_age : ℕ := sorry

/-- Ola's current age -/
def ola_age : ℕ := sorry

/-- Kola's age is twice Ola's age when Kola was Ola's current age -/
axiom condition1 : kola_age = 2 * (ola_age - (kola_age - ola_age))

/-- Sum of their ages when Ola reaches Kola's current age is 36 -/
axiom condition2 : kola_age + (kola_age + (kola_age - ola_age)) = 36

/-- Theorem stating Kola's current age is 16 -/
theorem kola_age_is_16 : kola_age = 16 := by sorry

end NUMINAMATH_CALUDE_kola_age_is_16_l377_37717


namespace NUMINAMATH_CALUDE_multiply_fractions_l377_37718

theorem multiply_fractions : 8 * (1 / 11) * 33 = 24 := by sorry

end NUMINAMATH_CALUDE_multiply_fractions_l377_37718


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_sum_l377_37775

theorem quadratic_equation_solution_sum : ∃ (a b : ℝ), 
  (a^2 - 6*a + 15 = 24) ∧ 
  (b^2 - 6*b + 15 = 24) ∧ 
  (a ≥ b) ∧ 
  (3*a + 2*b = 15 + 3*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_sum_l377_37775


namespace NUMINAMATH_CALUDE_alice_shoe_probability_l377_37704

/-- Represents the number of pairs for each color of shoes --/
structure ShoePairs where
  black : Nat
  brown : Nat
  white : Nat
  gray : Nat

/-- Calculates the probability of picking two shoes of the same color
    with one being left and one being right --/
def probability_same_color_different_feet (pairs : ShoePairs) : Rat :=
  let total_shoes := 2 * (pairs.black + pairs.brown + pairs.white + pairs.gray)
  let prob_black := (2 * pairs.black) * pairs.black / (total_shoes * (total_shoes - 1))
  let prob_brown := (2 * pairs.brown) * pairs.brown / (total_shoes * (total_shoes - 1))
  let prob_white := (2 * pairs.white) * pairs.white / (total_shoes * (total_shoes - 1))
  let prob_gray := (2 * pairs.gray) * pairs.gray / (total_shoes * (total_shoes - 1))
  prob_black + prob_brown + prob_white + prob_gray

theorem alice_shoe_probability :
  probability_same_color_different_feet ⟨7, 4, 3, 1⟩ = 25 / 145 := by
  sorry

end NUMINAMATH_CALUDE_alice_shoe_probability_l377_37704


namespace NUMINAMATH_CALUDE_sixth_diagram_shaded_fraction_l377_37736

/-- Represents the number of shaded triangles in the nth diagram -/
def shaded_triangles (n : ℕ) : ℕ := (n - 1) ^ 2

/-- Represents the total number of triangles in the nth diagram -/
def total_triangles (n : ℕ) : ℕ := n ^ 2

/-- The fraction of shaded triangles in the nth diagram -/
def shaded_fraction (n : ℕ) : ℚ := shaded_triangles n / total_triangles n

theorem sixth_diagram_shaded_fraction :
  shaded_fraction 6 = 25 / 36 := by sorry

end NUMINAMATH_CALUDE_sixth_diagram_shaded_fraction_l377_37736


namespace NUMINAMATH_CALUDE_g_definition_l377_37743

-- Define the function f
def f (x : ℝ) : ℝ := 5 - 2*x

-- Define the function g
def g (x : ℝ) : ℝ := 4 - 3*x

-- Theorem statement
theorem g_definition (x : ℝ) : 
  (∀ y, f (y + 1) = 3 - 2*y) ∧ (f (g x) = 6*x - 3) → g x = 4 - 3*x :=
by
  sorry

end NUMINAMATH_CALUDE_g_definition_l377_37743


namespace NUMINAMATH_CALUDE_omega_real_iff_m_eq_4_or_neg_3_omega_in_fourth_quadrant_iff_3_lt_m_lt_4_l377_37750

-- Define the complex number ω as a function of m
def ω (m : ℝ) : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 - m - 12)

-- Theorem 1: ω is real iff m = 4 or m = -3
theorem omega_real_iff_m_eq_4_or_neg_3 (m : ℝ) :
  ω m ∈ Set.range Complex.ofReal ↔ m = 4 ∨ m = -3 :=
sorry

-- Theorem 2: ω is in the fourth quadrant iff 3 < m < 4
theorem omega_in_fourth_quadrant_iff_3_lt_m_lt_4 (m : ℝ) :
  (Complex.re (ω m) > 0 ∧ Complex.im (ω m) < 0) ↔ 3 < m ∧ m < 4 :=
sorry

end NUMINAMATH_CALUDE_omega_real_iff_m_eq_4_or_neg_3_omega_in_fourth_quadrant_iff_3_lt_m_lt_4_l377_37750


namespace NUMINAMATH_CALUDE_complex_multiplication_l377_37794

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (2 * i - 1) = -2 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l377_37794


namespace NUMINAMATH_CALUDE_binary_1010_is_10_l377_37715

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1010_is_10 :
  binary_to_decimal [false, true, false, true] = 10 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010_is_10_l377_37715


namespace NUMINAMATH_CALUDE_water_evaporation_rate_l377_37744

/-- Given a glass with 10 ounces of water, with 6% evaporating over 20 days,
    the amount of water evaporating each day is 0.03 ounces. -/
theorem water_evaporation_rate (initial_water : ℝ) (days : ℕ) (evaporation_percentage : ℝ) :
  initial_water = 10 →
  days = 20 →
  evaporation_percentage = 0.06 →
  (initial_water * evaporation_percentage) / days = 0.03 :=
by sorry

end NUMINAMATH_CALUDE_water_evaporation_rate_l377_37744


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l377_37709

theorem complex_purely_imaginary (a : ℝ) :
  (a = 1 → ∃ (z : ℂ), z = (a - 1) * (a + 2) + (a + 3) * I ∧ z.re = 0) ∧
  (∃ (b : ℝ), b ≠ 1 ∧ ∃ (z : ℂ), z = (b - 1) * (b + 2) + (b + 3) * I ∧ z.re = 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l377_37709


namespace NUMINAMATH_CALUDE_overlap_area_63_l377_37790

/-- Represents the geometric shapes and their movement --/
structure GeometricSetup where
  square_side : ℝ
  triangle_hypotenuse : ℝ
  initial_distance : ℝ
  relative_speed : ℝ

/-- Calculates the overlapping area at a given time --/
def overlapping_area (setup : GeometricSetup) (t : ℝ) : ℝ :=
  sorry

/-- The main theorem stating when the overlapping area is 63 square centimeters --/
theorem overlap_area_63 (setup : GeometricSetup) 
  (h1 : setup.square_side = 12)
  (h2 : setup.triangle_hypotenuse = 18)
  (h3 : setup.initial_distance = 13)
  (h4 : setup.relative_speed = 5) :
  (∃ t : ℝ, t = 5 ∨ t = 6.2) ∧ (overlapping_area setup t = 63) :=
sorry

end NUMINAMATH_CALUDE_overlap_area_63_l377_37790


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_l377_37778

/-- The sum of the infinite geometric series 1 - (1/4) + (1/4)^2 - (1/4)^3 + ... -/
def infiniteGeometricSeriesSum : ℚ := 4/5

/-- The first term of the series -/
def a : ℚ := 1

/-- The common ratio of the series -/
def r : ℚ := -1/4

/-- Theorem: The sum of the infinite geometric series 1 - (1/4) + (1/4)^2 - (1/4)^3 + ... is 4/5 -/
theorem infinite_geometric_series_sum :
  infiniteGeometricSeriesSum = a / (1 - r) :=
by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_l377_37778


namespace NUMINAMATH_CALUDE_power_values_l377_37773

-- Define variables
variable (a m n : ℝ)

-- State the theorem
theorem power_values (h1 : a^m = 2) (h2 : a^n = 3) :
  a^(4*m + 3*n) = 432 ∧ a^(5*m - 2*n) = 32/9 := by
  sorry

end NUMINAMATH_CALUDE_power_values_l377_37773


namespace NUMINAMATH_CALUDE_linear_function_k_value_l377_37791

/-- Given a linear function y = kx - 2 that passes through the point (-1, 3), prove that k = -5 -/
theorem linear_function_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = k * x - 2) → -- The function is y = kx - 2
  (3 : ℝ) = k * (-1 : ℝ) - 2 → -- The function passes through the point (-1, 3)
  k = -5 := by
sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l377_37791


namespace NUMINAMATH_CALUDE_brenda_skittles_l377_37731

theorem brenda_skittles (initial : ℕ) (bought : ℕ) (final : ℕ) : 
  initial = 7 → bought = 8 → final = initial + bought → final = 15 := by
  sorry

end NUMINAMATH_CALUDE_brenda_skittles_l377_37731


namespace NUMINAMATH_CALUDE_cos_A_value_c_value_l377_37768

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.C = 2 * t.A ∧ Real.cos t.C = 1/8

-- Theorem 1: Prove cos A = 3/4
theorem cos_A_value (t : Triangle) (h : triangle_conditions t) : Real.cos t.A = 3/4 := by
  sorry

-- Theorem 2: Prove c = 6 when a = 4
theorem c_value (t : Triangle) (h1 : triangle_conditions t) (h2 : t.a = 4) : t.c = 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_A_value_c_value_l377_37768


namespace NUMINAMATH_CALUDE_justin_tim_games_count_l377_37745

/-- The number of players in the four-square league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- The number of players to choose after Justin and Tim are already selected -/
def players_to_choose : ℕ := players_per_game - 2

/-- The number of remaining players after Justin and Tim are excluded -/
def remaining_players : ℕ := total_players - 2

/-- Theorem stating that the number of games Justin and Tim play together
    is equal to the number of ways to choose the remaining players -/
theorem justin_tim_games_count :
  Nat.choose remaining_players players_to_choose = 210 := by
  sorry

end NUMINAMATH_CALUDE_justin_tim_games_count_l377_37745


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l377_37702

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}
def N : Set ℝ := {x | (2 : ℝ) ^ (x * (x - 2)) < 1}

-- State the theorem
theorem complement_M_intersect_N : 
  (Mᶜ ∩ N : Set ℝ) = {x | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l377_37702


namespace NUMINAMATH_CALUDE_set_intersection_equality_l377_37797

def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x ≤ 3}

theorem set_intersection_equality : A ∩ B = A := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l377_37797


namespace NUMINAMATH_CALUDE_b_value_l377_37738

-- Define the functions p and q
def p (x : ℝ) : ℝ := 2 * x - 7
def q (x b : ℝ) : ℝ := 3 * x - b

-- State the theorem
theorem b_value (b : ℝ) : p (q 3 b) = 3 → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_b_value_l377_37738


namespace NUMINAMATH_CALUDE_line_inclination_cosine_l377_37785

/-- Given a line with parametric equations x = 1 + 3t and y = 2 - 4t,
    the cosine of its inclination angle is -3/5 -/
theorem line_inclination_cosine :
  ∀ (t : ℝ), 
  let x := 1 + 3*t
  let y := 2 - 4*t
  let slope := (y - 2) / (x - 1)
  let inclination_angle := Real.arctan slope
  Real.cos inclination_angle = -3/5 := by sorry

end NUMINAMATH_CALUDE_line_inclination_cosine_l377_37785


namespace NUMINAMATH_CALUDE_sqrt_product_equals_product_l377_37754

theorem sqrt_product_equals_product : Real.sqrt (4 * 9) = 2 * 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_product_l377_37754


namespace NUMINAMATH_CALUDE_la_retail_women_ratio_l377_37776

/-- The ratio of women working in retail to the total number of women in Los Angeles -/
def retail_women_ratio (total_population : ℕ) (women_population : ℕ) (retail_women : ℕ) : ℚ :=
  retail_women / women_population

theorem la_retail_women_ratio :
  let total_population : ℕ := 6000000
  let women_population : ℕ := total_population / 2
  let retail_women : ℕ := 1000000
  retail_women_ratio total_population women_population retail_women = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_la_retail_women_ratio_l377_37776


namespace NUMINAMATH_CALUDE_interior_triangle_area_l377_37759

/-- Given three squares with areas 36, 64, and 100, where the largest square is diagonal to the other two squares, the area of the interior triangle formed by the sides of these squares is 24. -/
theorem interior_triangle_area (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100) 
  (hdiag : c = max a b) : (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_interior_triangle_area_l377_37759


namespace NUMINAMATH_CALUDE_min_value_of_f_l377_37761

/-- The function f(x) = -x^3 + 3x^2 + 9x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

/-- Theorem: Given f(x) = -x^3 + 3x^2 + 9x + a, where a is a constant,
    and the maximum value of f(x) in the interval [-2, 2] is 20,
    the minimum value of f(x) in the interval [-2, 2] is -7. -/
theorem min_value_of_f (a : ℝ) (h : ∃ x ∈ Set.Icc (-2) 2, f a x = 20 ∧ ∀ y ∈ Set.Icc (-2) 2, f a y ≤ 20) :
  ∃ x ∈ Set.Icc (-2) 2, f a x = -7 ∧ ∀ y ∈ Set.Icc (-2) 2, f a y ≥ -7 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l377_37761


namespace NUMINAMATH_CALUDE_binomial_arithmetic_sequence_l377_37701

theorem binomial_arithmetic_sequence (n k : ℕ) :
  (∃ (u : ℕ), u > 2 ∧ n = u^2 - 2 ∧ (k = u.choose 2 - 1 ∨ k = (u + 1).choose 2 - 1)) ↔
  (Nat.choose n (k - 1) - 2 * Nat.choose n k + Nat.choose n (k + 1) = 0) := by
  sorry

end NUMINAMATH_CALUDE_binomial_arithmetic_sequence_l377_37701


namespace NUMINAMATH_CALUDE_town_literacy_distribution_l377_37726

theorem town_literacy_distribution :
  ∀ (T : ℝ) (M F : ℝ),
    T > 0 →
    M + F = 100 →
    0.20 * M * T + 0.325 * F * T = 0.25 * T →
    M = 60 ∧ F = 40 := by
  sorry

end NUMINAMATH_CALUDE_town_literacy_distribution_l377_37726


namespace NUMINAMATH_CALUDE_min_additional_cells_for_symmetry_l377_37728

/-- Represents a cell in the rectangle --/
structure Cell where
  x : ℤ
  y : ℤ

/-- Represents the rectangle --/
structure Rectangle where
  width : ℕ
  height : ℕ
  center : Cell

/-- The set of initially colored cells --/
def initialColoredCells : Finset Cell := sorry

/-- Function to determine if two cells are symmetric about the center --/
def isSymmetric (c1 c2 : Cell) (center : Cell) : Prop := sorry

/-- Function to count the number of additional cells needed for symmetry --/
def additionalCellsForSymmetry (rect : Rectangle) (initial : Finset Cell) : ℕ := sorry

/-- Theorem stating that the minimum number of additional cells to color is 7 --/
theorem min_additional_cells_for_symmetry (rect : Rectangle) : 
  additionalCellsForSymmetry rect initialColoredCells = 7 := by sorry

end NUMINAMATH_CALUDE_min_additional_cells_for_symmetry_l377_37728
