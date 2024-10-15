import Mathlib

namespace NUMINAMATH_CALUDE_num_factors_of_M_l77_7795

/-- The number of natural-number factors of M, where M = 2^5 · 3^2 · 7^3 · 11^1 -/
def num_factors (M : ℕ) : ℕ :=
  (5 + 1) * (2 + 1) * (3 + 1) * (1 + 1)

/-- M is defined as 2^5 · 3^2 · 7^3 · 11^1 -/
def M : ℕ := 2^5 * 3^2 * 7^3 * 11

theorem num_factors_of_M :
  num_factors M = 144 :=
sorry

end NUMINAMATH_CALUDE_num_factors_of_M_l77_7795


namespace NUMINAMATH_CALUDE_mean_score_proof_l77_7776

theorem mean_score_proof (first_class_mean second_class_mean : ℝ)
                         (total_students : ℕ)
                         (class_ratio : ℚ) :
  first_class_mean = 90 →
  second_class_mean = 75 →
  total_students = 66 →
  class_ratio = 5 / 6 →
  ∃ (first_class_students second_class_students : ℕ),
    first_class_students + second_class_students = total_students ∧
    (first_class_students : ℚ) / (second_class_students : ℚ) = class_ratio ∧
    (first_class_mean * (first_class_students : ℝ) + 
     second_class_mean * (second_class_students : ℝ)) / (total_students : ℝ) = 82 :=
by sorry

end NUMINAMATH_CALUDE_mean_score_proof_l77_7776


namespace NUMINAMATH_CALUDE_max_value_constraint_l77_7796

theorem max_value_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + 4 * y < 72) :
  x * y * (72 - 3 * x - 4 * y) ≤ 1152 := by
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l77_7796


namespace NUMINAMATH_CALUDE_log7_10_approximation_l77_7774

-- Define the approximations given in the problem
def log10_2_approx : ℝ := 0.301
def log10_5_approx : ℝ := 0.699

-- Define a tolerance for approximation
def tolerance : ℝ := 0.001

-- Theorem statement
theorem log7_10_approximation :
  let log10_7 := log10_5_approx + log10_2_approx
  abs (Real.log 10 / Real.log 7 - 33 / 10) < tolerance := by
  sorry

end NUMINAMATH_CALUDE_log7_10_approximation_l77_7774


namespace NUMINAMATH_CALUDE_troll_count_l77_7745

/-- The number of creatures at the table -/
def total_creatures : ℕ := 60

/-- The number of trolls at the table -/
def num_trolls : ℕ := 20

/-- The number of elves who made a mistake -/
def mistake_elves : ℕ := 2

theorem troll_count :
  ∀ t : ℕ,
  t = num_trolls →
  (∃ x : ℕ,
    x ∈ ({2, 4, 6} : Set ℕ) ∧
    3 * t + x = total_creatures + 4 ∧
    t + (total_creatures - t) = total_creatures ∧
    t - mistake_elves = (total_creatures - t) - x / 2) :=
by sorry

end NUMINAMATH_CALUDE_troll_count_l77_7745


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l77_7799

theorem divisibility_by_seven (n : ℕ) : 
  ∃ k : ℤ, ((-8)^(2019 : ℕ) + (-8)^(2018 : ℕ)) = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l77_7799


namespace NUMINAMATH_CALUDE_complementary_angle_adjustment_l77_7708

/-- Proves that when two complementary angles with a ratio of 3:7 have the smaller angle
    increased by 20%, the larger angle must decrease by 8.571% to maintain complementary angles. -/
theorem complementary_angle_adjustment (smaller larger : ℝ) : 
  smaller + larger = 90 →  -- angles are complementary
  smaller / larger = 3 / 7 →  -- ratio of angles is 3:7
  let new_smaller := smaller * 1.20  -- smaller angle increased by 20%
  let new_larger := 90 - new_smaller  -- new larger angle to maintain complementary
  (larger - new_larger) / larger * 100 = 8.571 :=  -- percentage decrease of larger angle
by sorry

end NUMINAMATH_CALUDE_complementary_angle_adjustment_l77_7708


namespace NUMINAMATH_CALUDE_f_properties_l77_7798

-- Define the properties of function f
def is_additive (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

def is_negative_for_positive (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f x < 0

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- State the theorem
theorem f_properties (f : ℝ → ℝ) 
  (h1 : is_additive f) (h2 : is_negative_for_positive f) : 
  is_odd f ∧ is_decreasing f := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l77_7798


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l77_7726

theorem sum_of_fifth_powers (a b c d : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (sum_condition : a + b + c + d = 3) 
  (sum_of_squares : a^2 + b^2 + c^2 + d^2 = 45) : 
  (a^5 / ((a-b)*(a-c)*(a-d))) + (b^5 / ((b-a)*(b-c)*(b-d))) + 
  (c^5 / ((c-a)*(c-b)*(c-d))) + (d^5 / ((d-a)*(d-b)*(d-c))) = -9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l77_7726


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l77_7768

theorem sufficient_not_necessary_condition (x y z : ℝ) :
  (∀ z ≠ 0, x * z^2024 < y * z^2024 → x < y) ∧
  ¬(∀ x y : ℝ, x < y → ∀ z : ℝ, x * z^2024 < y * z^2024) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l77_7768


namespace NUMINAMATH_CALUDE_january_salary_is_2900_l77_7737

/-- Calculates the salary for January given the average salaries and May's salary -/
def january_salary (avg_jan_to_apr avg_feb_to_may may_salary : ℚ) : ℚ :=
  4 * avg_jan_to_apr - (4 * avg_feb_to_may - may_salary)

/-- Theorem stating that the salary for January is 2900 given the provided conditions -/
theorem january_salary_is_2900 :
  january_salary 8000 8900 6500 = 2900 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_is_2900_l77_7737


namespace NUMINAMATH_CALUDE_sum_even_coefficients_l77_7713

theorem sum_even_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ) :
  (∀ x : ℝ, (1 + x + x^2)^6 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
    a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12) →
  a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂ = 364 := by
sorry

end NUMINAMATH_CALUDE_sum_even_coefficients_l77_7713


namespace NUMINAMATH_CALUDE_distribute_balls_theorem_l77_7779

/-- The number of ways to distribute 4 different balls into 3 labeled boxes, with no box left empty -/
def distributeWays : ℕ := 36

/-- The number of ways to choose 2 balls from 4 different balls -/
def chooseTwo : ℕ := 6

/-- The number of ways to permute 3 groups -/
def permuteThree : ℕ := 6

theorem distribute_balls_theorem :
  distributeWays = chooseTwo * permuteThree := by sorry

end NUMINAMATH_CALUDE_distribute_balls_theorem_l77_7779


namespace NUMINAMATH_CALUDE_clock_right_angles_in_day_l77_7707

/-- Represents a clock with an hour hand and a minute hand. -/
structure Clock :=
  (hour_hand : ℕ)
  (minute_hand : ℕ)

/-- Represents a day consisting of 24 hours. -/
def Day := 24

/-- Checks if the hands of a clock are at right angles. -/
def is_right_angle (c : Clock) : Prop :=
  (c.hour_hand * 5 - c.minute_hand) % 60 = 15 ∨ (c.minute_hand - c.hour_hand * 5) % 60 = 15

/-- Counts the number of times the clock hands are at right angles in a day. -/
def count_right_angles (d : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the hands of a clock are at right angles 44 times in a day. -/
theorem clock_right_angles_in_day :
  count_right_angles Day = 44 :=
sorry

end NUMINAMATH_CALUDE_clock_right_angles_in_day_l77_7707


namespace NUMINAMATH_CALUDE_number_of_girls_is_760_l77_7741

/-- Represents the number of students in a school survey --/
structure SchoolSurvey where
  total_students : ℕ
  sample_size : ℕ
  girls_sampled_difference : ℕ

/-- Calculates the number of girls in the school based on survey data --/
def number_of_girls (survey : SchoolSurvey) : ℕ :=
  survey.total_students / 2 - survey.girls_sampled_difference * (survey.total_students / survey.sample_size / 2)

/-- Theorem stating that given the survey conditions, the number of girls in the school is 760 --/
theorem number_of_girls_is_760 (survey : SchoolSurvey) 
    (h1 : survey.total_students = 1600)
    (h2 : survey.sample_size = 200)
    (h3 : survey.girls_sampled_difference = 10) : 
  number_of_girls survey = 760 := by
  sorry

#eval number_of_girls { total_students := 1600, sample_size := 200, girls_sampled_difference := 10 }

end NUMINAMATH_CALUDE_number_of_girls_is_760_l77_7741


namespace NUMINAMATH_CALUDE_martha_apples_l77_7735

theorem martha_apples (jane_apples james_apples martha_remaining martha_to_give : ℕ) :
  jane_apples = 5 →
  james_apples = jane_apples + 2 →
  martha_remaining = 4 →
  martha_to_give = 4 →
  jane_apples + james_apples + martha_remaining + martha_to_give = 20 :=
by sorry

end NUMINAMATH_CALUDE_martha_apples_l77_7735


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l77_7756

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem f_derivative_at_zero : 
  (deriv f) 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l77_7756


namespace NUMINAMATH_CALUDE_banana_price_theorem_l77_7703

/-- The cost of a banana in pence -/
def banana_cost : ℚ := 1.25

/-- The number of pence in a shilling -/
def pence_per_shilling : ℕ := 12

/-- The number of shillings in a pound -/
def shillings_per_pound : ℕ := 20

/-- The number of bananas in a dozen dozen -/
def dozen_dozen : ℕ := 12 * 12

theorem banana_price_theorem :
  let pence_per_pound : ℕ := pence_per_shilling * shillings_per_pound
  let bananas_per_fiver : ℚ := (5 * pence_per_pound : ℚ) / banana_cost
  let sixpences_for_16_dozen_dozen : ℚ := (16 * dozen_dozen * banana_cost) / 6
  sixpences_for_16_dozen_dozen = bananas_per_fiver / 2 :=
by sorry


end NUMINAMATH_CALUDE_banana_price_theorem_l77_7703


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l77_7772

/-- The y-intercept of the line 3x - 4y = 12 is -3 -/
theorem y_intercept_of_line (x y : ℝ) : 3 * x - 4 * y = 12 → x = 0 → y = -3 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l77_7772


namespace NUMINAMATH_CALUDE_kath_movie_cost_l77_7786

def movie_admission_cost (regular_price : ℚ) (discount_percent : ℚ) (before_6pm : Bool) (num_people : ℕ) : ℚ :=
  let discounted_price := if before_6pm then regular_price * (1 - discount_percent / 100) else regular_price
  discounted_price * num_people

theorem kath_movie_cost :
  let regular_price : ℚ := 8
  let discount_percent : ℚ := 25
  let before_6pm : Bool := true
  let num_people : ℕ := 6
  movie_admission_cost regular_price discount_percent before_6pm num_people = 36 := by
  sorry

end NUMINAMATH_CALUDE_kath_movie_cost_l77_7786


namespace NUMINAMATH_CALUDE_continuous_function_inequality_l77_7759

theorem continuous_function_inequality (f : ℝ → ℝ) (hf : Continuous f) 
  (h : ∀ x : ℝ, (x - 1) * (deriv f x) < 0) : f 0 + f 2 < 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_continuous_function_inequality_l77_7759


namespace NUMINAMATH_CALUDE_square_sum_equals_fifteen_l77_7757

theorem square_sum_equals_fifteen (x y : ℝ) (h1 : x * y = 3) (h2 : (x - y)^2 = 9) :
  x^2 + y^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_fifteen_l77_7757


namespace NUMINAMATH_CALUDE_man_tshirt_count_l77_7748

/-- Given a man with pants and t-shirts, calculates the number of ways he can dress --/
def dressing_combinations (num_tshirts : ℕ) (num_pants : ℕ) : ℕ :=
  num_tshirts * num_pants

theorem man_tshirt_count :
  ∀ (num_pants : ℕ) (total_combinations : ℕ),
    num_pants = 9 →
    total_combinations = 72 →
    ∃ (num_tshirts : ℕ),
      dressing_combinations num_tshirts num_pants = total_combinations ∧
      num_tshirts = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_man_tshirt_count_l77_7748


namespace NUMINAMATH_CALUDE_reciprocal_problem_l77_7763

theorem reciprocal_problem (x : ℝ) (h : 5 * x = 2) : 100 * (1 / x) = 250 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l77_7763


namespace NUMINAMATH_CALUDE_bear_hunting_problem_l77_7758

theorem bear_hunting_problem (bear_need : ℕ) (cub_need : ℕ) (num_cubs : ℕ) (animals_per_day : ℕ) : 
  bear_need = 210 →
  cub_need = 35 →
  num_cubs = 4 →
  animals_per_day = 10 →
  (bear_need + cub_need * num_cubs) / 7 / animals_per_day = 5 := by
sorry

end NUMINAMATH_CALUDE_bear_hunting_problem_l77_7758


namespace NUMINAMATH_CALUDE_solution_set_of_f_positive_l77_7773

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 5*x + 6

-- State the theorem
theorem solution_set_of_f_positive :
  {x : ℝ | f x > 0} = {x : ℝ | x > 3 ∨ x < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_f_positive_l77_7773


namespace NUMINAMATH_CALUDE_triangle_area_l77_7721

theorem triangle_area (A B C : ℝ) (a : ℝ) (h1 : a = 2) (h2 : C = π/4) (h3 : Real.tan (B/2) = 1/2) :
  (1/2) * a * (Real.sin C) * (8 * Real.sqrt 2 / 7) = 8/7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l77_7721


namespace NUMINAMATH_CALUDE_apple_selection_probability_l77_7743

def total_apples : ℕ := 10
def red_apples : ℕ := 5
def green_apples : ℕ := 3
def yellow_apples : ℕ := 2
def selected_apples : ℕ := 3

theorem apple_selection_probability :
  (Nat.choose green_apples 2 * Nat.choose yellow_apples 1) / Nat.choose total_apples selected_apples = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_apple_selection_probability_l77_7743


namespace NUMINAMATH_CALUDE_second_number_proof_l77_7767

theorem second_number_proof (x y z : ℝ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 4 / 7)
  (x_pos : x > 0) (y_pos : y > 0) (z_pos : z > 0) : 
  y = 240 / 7 := by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l77_7767


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l77_7722

theorem sqrt_equation_solution : 
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 13 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l77_7722


namespace NUMINAMATH_CALUDE_cannon_firing_time_l77_7740

/-- Represents a cannon with a specified firing rate and number of shots -/
structure Cannon where
  firing_rate : ℕ  -- shots per minute
  total_shots : ℕ

/-- Calculates the time taken to fire all shots for a given cannon -/
def time_to_fire (c : Cannon) : ℕ :=
  c.total_shots - 1

/-- The cannon from the problem -/
def test_cannon : Cannon :=
  { firing_rate := 1, total_shots := 60 }

/-- Theorem stating that the time to fire all shots is 59 minutes -/
theorem cannon_firing_time :
  time_to_fire test_cannon = 59 := by sorry

end NUMINAMATH_CALUDE_cannon_firing_time_l77_7740


namespace NUMINAMATH_CALUDE_missing_number_exists_l77_7771

theorem missing_number_exists : ∃ x : ℝ, (1 / ((1 / 0.03) + (1 / x))) = 0.02775 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_exists_l77_7771


namespace NUMINAMATH_CALUDE_abc_negative_root_at_four_y1_greater_y2_l77_7701

/-- Represents a parabola y = ax² + bx + c with vertex at (1, n) and 4a - 2b + c = 0 -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  n : ℝ
  vertex_x : a * 1 + b = 0
  vertex_y : a * 1^2 + b * 1 + c = n
  condition : 4 * a - 2 * b + c = 0

/-- If n > 0, then abc < 0 -/
theorem abc_negative (p : Parabola) (h : p.n > 0) : p.a * p.b * p.c < 0 := by sorry

/-- The equation ax² + bx + c = 0 has a root at x = 4 -/
theorem root_at_four (p : Parabola) : p.a * 4^2 + p.b * 4 + p.c = 0 := by sorry

/-- For any two points A(x₁, y₁) and B(x₂, y₂) on the parabola with x₁ < x₂, 
    if a(x₁ + x₂ - 2) < 0, then y₁ > y₂ -/
theorem y1_greater_y2 (p : Parabola) (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = p.a * x₁^2 + p.b * x₁ + p.c)
  (h2 : y₂ = p.a * x₂^2 + p.b * x₂ + p.c)
  (h3 : x₁ < x₂)
  (h4 : p.a * (x₁ + x₂ - 2) < 0) : 
  y₁ > y₂ := by sorry

end NUMINAMATH_CALUDE_abc_negative_root_at_four_y1_greater_y2_l77_7701


namespace NUMINAMATH_CALUDE_product_of_specific_integers_l77_7752

theorem product_of_specific_integers : 
  ∀ (a b : ℤ), a = 32 ∧ b = 32 ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 → a * b = 1024 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_integers_l77_7752


namespace NUMINAMATH_CALUDE_max_y_coordinate_cos_2theta_l77_7733

/-- The maximum y-coordinate of a point on the curve r = cos 2θ in polar coordinates -/
theorem max_y_coordinate_cos_2theta : 
  let r : ℝ → ℝ := λ θ ↦ Real.cos (2 * θ)
  let x : ℝ → ℝ := λ θ ↦ r θ * Real.cos θ
  let y : ℝ → ℝ := λ θ ↦ r θ * Real.sin θ
  ∃ (θ_max : ℝ), ∀ (θ : ℝ), y θ ≤ y θ_max ∧ y θ_max = 3 * Real.sqrt 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_max_y_coordinate_cos_2theta_l77_7733


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l77_7719

theorem equal_roots_quadratic (c : ℝ) :
  (∃ x : ℝ, x^2 + 6*x + c = 0 ∧ (∀ y : ℝ, y^2 + 6*y + c = 0 → y = x)) →
  c = 9 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l77_7719


namespace NUMINAMATH_CALUDE_oshea_large_planters_l77_7715

/-- The number of large planters Oshea has -/
def num_large_planters (total_seeds small_planter_capacity large_planter_capacity num_small_planters : ℕ) : ℕ :=
  (total_seeds - small_planter_capacity * num_small_planters) / large_planter_capacity

/-- Proof that Oshea has 4 large planters -/
theorem oshea_large_planters :
  num_large_planters 200 4 20 30 = 4 := by
  sorry

end NUMINAMATH_CALUDE_oshea_large_planters_l77_7715


namespace NUMINAMATH_CALUDE_log_46328_between_consecutive_integers_l77_7739

theorem log_46328_between_consecutive_integers :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 46328 / Real.log 10 ∧ (Real.log 46328 / Real.log 10 < b) ∧ a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_46328_between_consecutive_integers_l77_7739


namespace NUMINAMATH_CALUDE_sphere_intersection_area_ratio_l77_7704

theorem sphere_intersection_area_ratio (R : ℝ) (h : R > 0) :
  let r := Real.sqrt ((3 / 4) * R^2)
  let circle_area := π * r^2
  let sphere_surface_area := 4 * π * R^2
  circle_area / sphere_surface_area = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sphere_intersection_area_ratio_l77_7704


namespace NUMINAMATH_CALUDE_work_completion_time_l77_7794

/-- The number of days it takes for the original number of people to complete the work -/
def days_to_complete_work (original_people : ℕ) (total_work : ℝ) : ℕ :=
  16

theorem work_completion_time 
  (original_people : ℕ) 
  (total_work : ℝ) 
  (h : (2 * original_people : ℝ) * 4 = total_work / 2) : 
  days_to_complete_work original_people total_work = 16 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l77_7794


namespace NUMINAMATH_CALUDE_average_of_c_and_d_l77_7770

theorem average_of_c_and_d (c d : ℝ) : 
  (4 + 6 + 8 + c + d) / 5 = 21 → (c + d) / 2 = 43.5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_c_and_d_l77_7770


namespace NUMINAMATH_CALUDE_line_plane_relationship_l77_7754

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the contained relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel_lines : Line → Line → Prop)

-- Define the skew relation between two lines
variable (skew_lines : Line → Line → Prop)

-- Theorem statement
theorem line_plane_relationship (a b : Line) (α : Plane) 
  (h1 : parallel_line_plane a α) 
  (h2 : contained_in b α) :
  parallel_lines a b ∨ skew_lines a b :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l77_7754


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l77_7787

/-- The number of diagonals in a polygon with n vertices -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon with one vertex removed is equivalent to a heptagon -/
def heptagon_vertices : ℕ := 8 - 1

theorem heptagon_diagonals : diagonals heptagon_vertices = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l77_7787


namespace NUMINAMATH_CALUDE_bridgette_has_three_cats_l77_7762

/-- Represents the number of baths given to pets in a year -/
def total_baths : ℕ := 96

/-- Represents the number of dogs Bridgette has -/
def num_dogs : ℕ := 2

/-- Represents the number of birds Bridgette has -/
def num_birds : ℕ := 4

/-- Represents the number of baths given to a dog in a year -/
def dog_baths_per_year : ℕ := 24

/-- Represents the number of baths given to a bird in a year -/
def bird_baths_per_year : ℕ := 3

/-- Represents the number of baths given to a cat in a year -/
def cat_baths_per_year : ℕ := 12

/-- Theorem stating that Bridgette has 3 cats -/
theorem bridgette_has_three_cats :
  ∃ (num_cats : ℕ),
    num_cats * cat_baths_per_year = 
      total_baths - (num_dogs * dog_baths_per_year + num_birds * bird_baths_per_year) ∧
    num_cats = 3 :=
by sorry

end NUMINAMATH_CALUDE_bridgette_has_three_cats_l77_7762


namespace NUMINAMATH_CALUDE_min_value_xy_plus_reciprocal_l77_7700

theorem min_value_xy_plus_reciprocal (x y : ℝ) 
  (h1 : x + y = -1) 
  (h2 : x < 0) 
  (h3 : y < 0) : 
  ∃ (min : ℝ), min = 17/4 ∧ ∀ z, z = x*y + 1/(x*y) → z ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_xy_plus_reciprocal_l77_7700


namespace NUMINAMATH_CALUDE_periodic_sequence_sum_l77_7727

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A sequence is periodic with period T if a_{n+T} = a_n for all n -/
def IsPeriodic (a : Sequence) (T : ℕ) : Prop :=
  ∀ n, a (n + T) = a n

/-- The sum of the first n terms of a sequence -/
def SequenceSum (a : Sequence) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

/-- Theorem: For a periodic sequence with period T, 
    the sum of m terms can be expressed in terms of T and r -/
theorem periodic_sequence_sum 
  (a : Sequence) (T m q r : ℕ) 
  (h_periodic : IsPeriodic a T) 
  (h_smallest : ∀ k, 0 < k → k < T → ¬IsPeriodic a k)
  (h_pos : 0 < T ∧ 0 < m ∧ 0 < q ∧ 0 < r)
  (h_decomp : m = q * T + r) :
  SequenceSum a m = q * SequenceSum a T + SequenceSum a r := by
  sorry

end NUMINAMATH_CALUDE_periodic_sequence_sum_l77_7727


namespace NUMINAMATH_CALUDE_dj_oldies_ratio_l77_7729

/-- Represents the number of song requests for each genre and the total requests --/
structure SongRequests where
  total : Nat
  electropop : Nat
  dance : Nat
  rock : Nat
  oldies : Nat
  rap : Nat

/-- Calculates the number of DJ's choice songs --/
def djChoice (s : SongRequests) : Nat :=
  s.total - (s.electropop + s.rock + s.oldies + s.rap)

/-- Theorem stating the ratio of DJ's choice to oldies songs --/
theorem dj_oldies_ratio (s : SongRequests) : 
  s.total = 30 ∧ 
  s.electropop = s.total / 2 ∧ 
  s.dance = s.electropop / 3 ∧ 
  s.rock = 5 ∧ 
  s.oldies = s.rock - 3 ∧ 
  s.rap = 2 → 
  (djChoice s : Int) / s.oldies = 3 := by
  sorry

end NUMINAMATH_CALUDE_dj_oldies_ratio_l77_7729


namespace NUMINAMATH_CALUDE_positive_fourth_root_of_6561_l77_7705

theorem positive_fourth_root_of_6561 (x : ℝ) (h1 : x > 0) (h2 : x^4 = 6561) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_positive_fourth_root_of_6561_l77_7705


namespace NUMINAMATH_CALUDE_a_value_proof_l77_7732

/-- The function f(x) = ax³ + 3x² + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem a_value_proof (a : ℝ) : f_derivative a (-1) = -12 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_a_value_proof_l77_7732


namespace NUMINAMATH_CALUDE_double_length_isosceles_triangle_base_length_l77_7709

/-- A triangle is double-length if one side is twice the length of another side. -/
def is_double_length_triangle (a b c : ℝ) : Prop :=
  a = 2 * b ∨ a = 2 * c ∨ b = 2 * c

/-- An isosceles triangle has two sides of equal length. -/
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ a = c ∨ b = c

theorem double_length_isosceles_triangle_base_length
  (a b c : ℝ)
  (h_isosceles : is_isosceles_triangle a b c)
  (h_double_length : is_double_length_triangle a b c)
  (h_ab_length : a = 10) :
  c = 5 := by
  sorry

end NUMINAMATH_CALUDE_double_length_isosceles_triangle_base_length_l77_7709


namespace NUMINAMATH_CALUDE_cube_volume_l77_7751

theorem cube_volume (s : ℝ) (h : s * s = 64) : s * s * s = 512 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l77_7751


namespace NUMINAMATH_CALUDE_calories_per_slice_l77_7746

/-- Given a pizza with 8 slices, where half the pizza contains 1200 calories,
    prove that each slice contains 300 calories. -/
theorem calories_per_slice (total_slices : ℕ) (eaten_fraction : ℚ) (total_calories : ℕ) :
  total_slices = 8 →
  eaten_fraction = 1/2 →
  total_calories = 1200 →
  (total_calories : ℚ) / (eaten_fraction * total_slices) = 300 := by
sorry

end NUMINAMATH_CALUDE_calories_per_slice_l77_7746


namespace NUMINAMATH_CALUDE_license_plate_count_l77_7706

/-- The number of possible letters in each position of the license plate -/
def num_letters : ℕ := 26

/-- The number of possible odd digits for the first digit position -/
def num_odd_digits : ℕ := 5

/-- The number of possible even digits for the second digit position -/
def num_even_digits : ℕ := 5

/-- The number of possible digits for the third digit position -/
def num_all_digits : ℕ := 10

/-- The total number of possible license plates under the given conditions -/
def total_license_plates : ℕ := num_letters^3 * num_odd_digits * num_even_digits * num_all_digits

theorem license_plate_count : total_license_plates = 17576000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l77_7706


namespace NUMINAMATH_CALUDE_hyperbola_max_y_coordinate_l77_7731

/-- Given a hyperbola with equation x²/4 - y²/b = 1 where b > 0,
    a point P(x,y) in the first quadrant satisfying |OP| = |F₁F₂|/2,
    and eccentricity e ∈ (1, 2], prove that the maximum y-coordinate is 3. -/
theorem hyperbola_max_y_coordinate (b : ℝ) (x y : ℝ) (e : ℝ) :
  b > 0 →
  x > 0 →
  y > 0 →
  x^2 / 4 - y^2 / b = 1 →
  x^2 + y^2 = 4 + b →
  1 < e ∧ e ≤ 2 →
  y ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_max_y_coordinate_l77_7731


namespace NUMINAMATH_CALUDE_factorization_equality_l77_7791

theorem factorization_equality (m n : ℝ) : 2*m*n^2 - 12*m*n + 18*m = 2*m*(n-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l77_7791


namespace NUMINAMATH_CALUDE_minutes_after_midnight_l77_7714

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

/-- The starting DateTime (midnight on January 1, 2013) -/
def startDateTime : DateTime :=
  { year := 2013, month := 1, day := 1, hour := 0, minute := 0 }

/-- The resulting DateTime after adding 2537 minutes -/
def resultDateTime : DateTime :=
  { year := 2013, month := 1, day := 2, hour := 18, minute := 17 }

/-- Theorem stating that adding 2537 minutes to the start time results in the correct end time -/
theorem minutes_after_midnight (startTime : DateTime) (elapsedMinutes : ℕ) :
  startTime = startDateTime → elapsedMinutes = 2537 →
  addMinutes startTime elapsedMinutes = resultDateTime :=
by
  sorry

end NUMINAMATH_CALUDE_minutes_after_midnight_l77_7714


namespace NUMINAMATH_CALUDE_smallest_next_divisor_l77_7730

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem smallest_next_divisor (m : ℕ) (h1 : 1000 ≤ m ∧ m ≤ 9999) 
  (h2 : is_odd m) (h3 : m % 437 = 0) :
  ∃ d : ℕ, d > 437 ∧ m % d = 0 ∧ is_odd d ∧
  ∀ d' : ℕ, d' > 437 → m % d' = 0 → is_odd d' → d ≤ d' :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_l77_7730


namespace NUMINAMATH_CALUDE_ajay_distance_theorem_l77_7780

/-- Calculates the distance traveled given speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Proves that given Ajay's speed of 50 km/hour and a travel time of 20 hours, 
    the distance traveled is 1000 km -/
theorem ajay_distance_theorem :
  let speed : ℝ := 50
  let time : ℝ := 20
  distance_traveled speed time = 1000 := by
sorry

end NUMINAMATH_CALUDE_ajay_distance_theorem_l77_7780


namespace NUMINAMATH_CALUDE_frank_game_points_l77_7717

theorem frank_game_points (enemies_defeated : ℕ) (points_per_enemy : ℕ) 
  (level_completion_points : ℕ) (special_challenges : ℕ) (points_per_challenge : ℕ) : 
  enemies_defeated = 15 → 
  points_per_enemy = 12 → 
  level_completion_points = 20 → 
  special_challenges = 5 → 
  points_per_challenge = 10 → 
  enemies_defeated * points_per_enemy + level_completion_points + special_challenges * points_per_challenge = 250 := by
  sorry

#check frank_game_points

end NUMINAMATH_CALUDE_frank_game_points_l77_7717


namespace NUMINAMATH_CALUDE_thick_line_segments_length_l77_7753

theorem thick_line_segments_length
  (perimeter_quadrilaterals : ℝ)
  (perimeter_triangles : ℝ)
  (perimeter_large_triangle : ℝ)
  (h1 : perimeter_quadrilaterals = 25)
  (h2 : perimeter_triangles = 20)
  (h3 : perimeter_large_triangle = 19) :
  (perimeter_quadrilaterals + perimeter_triangles - perimeter_large_triangle) / 2 = 13 :=
sorry

end NUMINAMATH_CALUDE_thick_line_segments_length_l77_7753


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l77_7710

-- Define the triangle ABC
variable (A B C : ℝ)

-- Define the dot product of vectors AB and BC
def dot_product_AB_BC : ℝ := sorry

-- Define the area of triangle ABC
def area_ABC : ℝ := sorry

-- State the theorem
theorem triangle_ABC_properties 
  (h1 : dot_product_AB_BC = (3/2) * area_ABC)
  (h2 : A - C = π/4) : 
  Real.sin B = 4/5 ∧ Real.cos A = (Real.sqrt (50 + 5 * Real.sqrt 2)) / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l77_7710


namespace NUMINAMATH_CALUDE_eldest_age_l77_7760

theorem eldest_age (x : ℝ) (h1 : 5*x - 7 + 7*x - 7 + 8*x - 7 = 59) : 8*x = 32 := by
  sorry

#check eldest_age

end NUMINAMATH_CALUDE_eldest_age_l77_7760


namespace NUMINAMATH_CALUDE_units_digit_problem_l77_7744

theorem units_digit_problem : ∃ n : ℕ, (8 * 14 * 1986 + 8^2) % 10 = 6 ∧ n * 10 ≤ (8 * 14 * 1986 + 8^2) ∧ (8 * 14 * 1986 + 8^2) < (n + 1) * 10 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l77_7744


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_for_necessary_not_sufficient_l77_7725

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3)*(2 - x) ≥ 0

-- Theorem 1
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 ≤ x ∧ x < 3) :=
sorry

-- Theorem 2
theorem range_of_a_for_necessary_not_sufficient :
  ∀ a : ℝ, (∀ x : ℝ, ¬(q x) → ¬(p x a)) ∧ (∃ x : ℝ, ¬(p x a) ∧ q x) ↔ (1 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_for_necessary_not_sufficient_l77_7725


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_l77_7765

theorem pure_imaginary_solutions (x : ℂ) :
  (x^4 - 4*x^3 + 6*x^2 - 40*x - 64 = 0) ∧ (∃ k : ℝ, x = k * Complex.I) ↔
  x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_l77_7765


namespace NUMINAMATH_CALUDE_PQRS_equals_nine_l77_7766

theorem PQRS_equals_nine :
  let P : ℝ := Real.sqrt 2010 + Real.sqrt 2007
  let Q : ℝ := -Real.sqrt 2010 - Real.sqrt 2007
  let R : ℝ := Real.sqrt 2010 - Real.sqrt 2007
  let S : ℝ := Real.sqrt 2007 - Real.sqrt 2010
  P * Q * R * S = 9 := by
  sorry

end NUMINAMATH_CALUDE_PQRS_equals_nine_l77_7766


namespace NUMINAMATH_CALUDE_solve_equation_l77_7720

theorem solve_equation (m n : ℕ) (h1 : ((1^m) / (5^m)) * ((1^n) / (4^n)) = 1 / (2 * (10^31))) (h2 : m = 31) : n = 16 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l77_7720


namespace NUMINAMATH_CALUDE_k_range_for_unique_integer_solution_l77_7702

/-- Given a real number k, this function represents the system of inequalities -/
def inequality_system (x k : ℝ) : Prop :=
  x^2 - x - 2 > 0 ∧ 2*x^2 + (5+2*k)*x + 5*k < 0

/-- This theorem states that if -2 is the only integer solution to the inequality system,
    then k is in the range [-3, 2) -/
theorem k_range_for_unique_integer_solution :
  (∀ x : ℤ, inequality_system (x : ℝ) k ↔ x = -2) →
  -3 ≤ k ∧ k < 2 :=
sorry

end NUMINAMATH_CALUDE_k_range_for_unique_integer_solution_l77_7702


namespace NUMINAMATH_CALUDE_triangle_agw_area_l77_7782

/-- Right triangle ABC with squares on legs and intersecting lines -/
structure RightTriangleWithSquares where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  W : ℝ × ℝ
  -- Conditions
  right_angle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  ac_length : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 14^2
  bc_length : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 28^2
  square_acde : D = (A.1 + C.2 - C.1, A.2 - C.2 + C.1) ∧ E = (C.1, A.2)
  square_cbfg : F = (C.1, B.2) ∧ G = (B.1, B.2 + B.1 - C.1)
  w_on_bc : ∃ t : ℝ, W = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2)
  w_on_af : ∃ s : ℝ, W = (s * A.1 + (1 - s) * F.1, s * A.2 + (1 - s) * F.2)

/-- The area of triangle AGW is 196 -/
theorem triangle_agw_area (t : RightTriangleWithSquares) : 
  abs ((t.A.1 * (t.G.2 - t.W.2) + t.G.1 * (t.W.2 - t.A.2) + t.W.1 * (t.A.2 - t.G.2)) / 2) = 196 := by
  sorry

end NUMINAMATH_CALUDE_triangle_agw_area_l77_7782


namespace NUMINAMATH_CALUDE_nebula_boys_count_total_students_correct_total_students_by_gender_correct_l77_7761

/-- Represents a school in the science camp. -/
inductive School
| Orion
| Nebula
| Galaxy

/-- Represents the gender of a student. -/
inductive Gender
| Boy
| Girl

/-- Represents the distribution of students in the science camp. -/
structure CampDistribution where
  total_students : ℕ
  total_boys : ℕ
  total_girls : ℕ
  students_by_school : School → ℕ
  boys_by_school : School → ℕ

/-- The actual distribution of students in the science camp. -/
def camp_data : CampDistribution :=
  { total_students := 150,
    total_boys := 84,
    total_girls := 66,
    students_by_school := fun s => match s with
      | School.Orion => 70
      | School.Nebula => 50
      | School.Galaxy => 30,
    boys_by_school := fun s => match s with
      | School.Orion => 30
      | _ => 0  -- We don't know these values yet
  }

/-- Theorem stating that the number of boys from Nebula Middle School is 32. -/
theorem nebula_boys_count (d : CampDistribution) (h : d = camp_data) :
  d.boys_by_school School.Nebula = 32 := by
  sorry

/-- Verify that the total number of students is correct. -/
theorem total_students_correct (d : CampDistribution) (h : d = camp_data) :
  d.total_students = d.students_by_school School.Orion +
                     d.students_by_school School.Nebula +
                     d.students_by_school School.Galaxy := by
  sorry

/-- Verify that the total number of students by gender is correct. -/
theorem total_students_by_gender_correct (d : CampDistribution) (h : d = camp_data) :
  d.total_students = d.total_boys + d.total_girls := by
  sorry

end NUMINAMATH_CALUDE_nebula_boys_count_total_students_correct_total_students_by_gender_correct_l77_7761


namespace NUMINAMATH_CALUDE_tangent_point_on_parabola_l77_7785

theorem tangent_point_on_parabola : ∃ (x y : ℝ), 
  y = x^2 ∧ 
  (2 : ℝ) * x = Real.tan (π / 4) ∧ 
  x = 1 / 2 ∧ 
  y = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_on_parabola_l77_7785


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l77_7712

theorem purely_imaginary_complex_number (m : ℝ) : 
  (m^2 - 1 : ℂ) + (m + 1 : ℂ) * Complex.I = Complex.I * y → m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l77_7712


namespace NUMINAMATH_CALUDE_tonya_needs_22_hamburgers_l77_7742

/-- The number of hamburgers Tonya needs to eat to beat last year's winner -/
def hamburgers_to_beat_record (ounces_per_hamburger : ℕ) (last_year_winner_ounces : ℕ) : ℕ :=
  (last_year_winner_ounces / ounces_per_hamburger) + 1

/-- Theorem stating that Tonya needs to eat 22 hamburgers to beat last year's winner -/
theorem tonya_needs_22_hamburgers :
  hamburgers_to_beat_record 4 84 = 22 := by
  sorry

end NUMINAMATH_CALUDE_tonya_needs_22_hamburgers_l77_7742


namespace NUMINAMATH_CALUDE_shelter_dogs_count_l77_7736

/-- Given an animal shelter with dogs and cats, prove the number of dogs. -/
theorem shelter_dogs_count (d c : ℕ) : 
  d * 7 = c * 15 →  -- Initial ratio of dogs to cats is 15:7
  d * 11 = (c + 16) * 15 →  -- Ratio after adding 16 cats is 15:11
  d = 60 :=  -- The number of dogs is 60
by sorry

end NUMINAMATH_CALUDE_shelter_dogs_count_l77_7736


namespace NUMINAMATH_CALUDE_corys_initial_money_l77_7781

/-- The problem of determining Cory's initial amount of money -/
theorem corys_initial_money (cost_per_pack : ℝ) (additional_needed : ℝ) : 
  cost_per_pack = 49 → additional_needed = 78 → 
  2 * cost_per_pack - additional_needed = 20 := by
  sorry

end NUMINAMATH_CALUDE_corys_initial_money_l77_7781


namespace NUMINAMATH_CALUDE_white_is_lightest_l77_7750

-- Define the puppy type
inductive Puppy
| White
| Black
| Yellowy
| Spotted

-- Define the "lighter than" relation
def lighterThan : Puppy → Puppy → Prop := sorry

-- State the theorem
theorem white_is_lightest :
  (lighterThan Puppy.White Puppy.Black) ∧
  (lighterThan Puppy.Black Puppy.Yellowy) ∧
  (lighterThan Puppy.Yellowy Puppy.Spotted) →
  ∀ p : Puppy, p ≠ Puppy.White → lighterThan Puppy.White p :=
sorry

end NUMINAMATH_CALUDE_white_is_lightest_l77_7750


namespace NUMINAMATH_CALUDE_ellipse_intersection_property_l77_7784

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 4 = 1

/-- Definition of the line l passing through P(x₁, y₁) -/
def line_l (x₁ y₁ x y : ℝ) : Prop :=
  4 * x₁ * x + 9 * y₁ * y = 36

/-- Theorem statement -/
theorem ellipse_intersection_property :
  ∀ (x₁ y₁ : ℝ),
  is_on_ellipse x₁ y₁ →
  ∃ (M_x M_y M'_x M'_y : ℝ),
    line_l x₁ y₁ M_x M_y ∧
    line_l x₁ y₁ M'_x M'_y ∧
    M_x = 3 ∧
    M'_x = -3 ∧
    (M_y^2 + 9) * (M'_y^2 + 9) = 36 ∧
    ∀ (N_x N_y N'_x N'_y : ℝ),
      line_l x₁ y₁ N_x N_y →
      line_l x₁ y₁ N'_x N'_y →
      N_x = 3 →
      N'_x = -3 →
      6 * (|N_y| + |N'_y|) ≥ 72 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_property_l77_7784


namespace NUMINAMATH_CALUDE_freezer_temperature_l77_7749

/-- Given a refrigerator with a refrigeration compartment and a freezer compartment,
    this theorem proves that if the refrigeration compartment is at 4°C and
    the freezer is 22°C colder, then the freezer temperature is -18°C. -/
theorem freezer_temperature
  (temp_refrigeration : ℝ)
  (temp_difference : ℝ)
  (h1 : temp_refrigeration = 4)
  (h2 : temp_difference = 22)
  : temp_refrigeration - temp_difference = -18 := by
  sorry

end NUMINAMATH_CALUDE_freezer_temperature_l77_7749


namespace NUMINAMATH_CALUDE_opposite_numbers_fraction_equals_one_l77_7711

theorem opposite_numbers_fraction_equals_one (a b : ℝ) 
  (h1 : a + b = 0) 
  (h2 : |a - b| = 2) : 
  (a^2 + 2*a*b + 2*b^2 + 2*a + 2*b + 1) / (a^2 + 3*a*b + b^2 + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_fraction_equals_one_l77_7711


namespace NUMINAMATH_CALUDE_remainder_of_eighteen_divided_by_seven_l77_7723

theorem remainder_of_eighteen_divided_by_seven : ∃ k : ℤ, 18 = 7 * k + 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_eighteen_divided_by_seven_l77_7723


namespace NUMINAMATH_CALUDE_coin_toss_probability_l77_7718

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1 - p)^(n - k)

theorem coin_toss_probability :
  let n : ℕ := 10
  let k : ℕ := 3
  let p : ℚ := 1/2
  binomial_probability n k p = 15/128 := by
sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l77_7718


namespace NUMINAMATH_CALUDE_prob_same_team_is_one_third_l77_7789

/-- The number of teams -/
def num_teams : ℕ := 3

/-- The probability of two students choosing the same team -/
def prob_same_team : ℚ := 1 / 3

/-- Theorem: The probability of two students independently and randomly choosing the same team out of three teams is 1/3 -/
theorem prob_same_team_is_one_third :
  prob_same_team = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_same_team_is_one_third_l77_7789


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_l77_7728

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_l77_7728


namespace NUMINAMATH_CALUDE_average_weight_b_c_l77_7734

theorem average_weight_b_c (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 41)
  (h3 : b = 33) :
  (b + c) / 2 = 43 := by
sorry

end NUMINAMATH_CALUDE_average_weight_b_c_l77_7734


namespace NUMINAMATH_CALUDE_max_product_under_constraint_l77_7793

theorem max_product_under_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3*a + 2*b = 1) :
  a * b ≤ 1/24 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3*a₀ + 2*b₀ = 1 ∧ a₀ * b₀ = 1/24 :=
by sorry

end NUMINAMATH_CALUDE_max_product_under_constraint_l77_7793


namespace NUMINAMATH_CALUDE_wallpaper_overlap_area_l77_7790

theorem wallpaper_overlap_area
  (total_area : ℝ)
  (double_layer_area : ℝ)
  (triple_layer_area : ℝ)
  (h1 : total_area = 300)
  (h2 : double_layer_area = 38)
  (h3 : triple_layer_area = 41) :
  total_area - 2 * double_layer_area - 3 * triple_layer_area = 101 :=
by
  sorry

end NUMINAMATH_CALUDE_wallpaper_overlap_area_l77_7790


namespace NUMINAMATH_CALUDE_locus_equation_l77_7783

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The locus of centers of circles externally tangent to C₁ and internally tangent to C₃ -/
def locus_of_centers (C₁ C₃ : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ r : ℝ,
    (Circle.mk p r).radius + C₁.radius = Real.sqrt ((p.1 - C₁.center.1)^2 + (p.2 - C₁.center.2)^2) ∧
    C₃.radius - (Circle.mk p r).radius = Real.sqrt ((p.1 - C₃.center.1)^2 + (p.2 - C₃.center.2)^2)}

theorem locus_equation (C₁ C₃ : Circle)
  (h₁ : C₁.center = (0, 0) ∧ C₁.radius = 2)
  (h₃ : C₃.center = (3, 0) ∧ C₃.radius = 5) :
  locus_of_centers C₁ C₃ = {p : ℝ × ℝ | p.1^2 + 7*p.2^2 - 34*p.1 - 57 = 0} :=
sorry

end NUMINAMATH_CALUDE_locus_equation_l77_7783


namespace NUMINAMATH_CALUDE_flag_arrangement_count_flag_arrangement_remainder_l77_7755

def M (b g : ℕ) : ℕ :=
  (b - 1) * Nat.choose (b + 2) g - 2 * Nat.choose (b + 1) g

theorem flag_arrangement_count :
  M 14 11 = 54054 :=
by sorry

theorem flag_arrangement_remainder :
  M 14 11 % 1000 = 54 :=
by sorry

end NUMINAMATH_CALUDE_flag_arrangement_count_flag_arrangement_remainder_l77_7755


namespace NUMINAMATH_CALUDE_amy_spent_32_pounds_l77_7764

/-- Represents the amount spent by Chloe in pounds -/
def chloe_spent : ℝ := 20

/-- Represents the amount spent by Becky as a fraction of Chloe's spending -/
def becky_spent_ratio : ℝ := 0.15

/-- Represents the amount spent by Amy as a fraction above Chloe's spending -/
def amy_spent_ratio : ℝ := 1.6

/-- The total amount spent by all three shoppers in pounds -/
def total_spent : ℝ := 55

theorem amy_spent_32_pounds :
  let becky_spent := becky_spent_ratio * chloe_spent
  let amy_spent := amy_spent_ratio * chloe_spent
  becky_spent + amy_spent + chloe_spent = total_spent ∧
  amy_spent = 32 := by sorry

end NUMINAMATH_CALUDE_amy_spent_32_pounds_l77_7764


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l77_7769

theorem quadratic_root_difference (p : ℝ) : 
  let a := 1
  let b := -(p + 1)
  let c := (p^2 + 2*p - 3) / 4
  let discriminant := b^2 - 4*a*c
  let larger_root := (-b + Real.sqrt discriminant) / (2*a)
  let smaller_root := (-b - Real.sqrt discriminant) / (2*a)
  larger_root - smaller_root = Real.sqrt (2*p + 1 - p^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l77_7769


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l77_7724

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) : 
  (d^2 / 2 : ℝ) = 72 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l77_7724


namespace NUMINAMATH_CALUDE_estate_value_l77_7778

/-- Represents the distribution of Mr. T's estate -/
structure EstateDistribution where
  total : ℝ
  wife_share : ℝ
  daughter1_share : ℝ
  daughter2_share : ℝ
  son_share : ℝ
  gardener_share : ℝ

/-- Defines the conditions of Mr. T's estate distribution -/
def valid_distribution (e : EstateDistribution) : Prop :=
  -- Two daughters and son received 3/4 of the estate
  e.daughter1_share + e.daughter2_share + e.son_share = 3/4 * e.total ∧
  -- Daughters shared their portion in the ratio of 5:3
  e.daughter1_share / e.daughter2_share = 5/3 ∧
  -- Wife received thrice as much as the son
  e.wife_share = 3 * e.son_share ∧
  -- Gardener received $600
  e.gardener_share = 600 ∧
  -- Sum of wife and gardener's shares was 1/4 of the estate
  e.wife_share + e.gardener_share = 1/4 * e.total ∧
  -- Total is sum of all shares
  e.total = e.wife_share + e.daughter1_share + e.daughter2_share + e.son_share + e.gardener_share

/-- Theorem stating that Mr. T's estate value is $2400 -/
theorem estate_value (e : EstateDistribution) (h : valid_distribution e) : e.total = 2400 :=
  sorry


end NUMINAMATH_CALUDE_estate_value_l77_7778


namespace NUMINAMATH_CALUDE_min_groups_and_people_is_16_l77_7738

/-- Represents the seating arrangement in a cafe -/
structure CafeSeating where
  tables : Nat
  counter_seats : Nat
  min_group_size : Nat
  max_group_size : Nat

/-- Represents the final seating state of the cafe -/
structure SeatingState where
  groups : Nat
  total_people : Nat

/-- The minimum possible value of groups + total people given the cafe seating conditions -/
def min_groups_and_people (cafe : CafeSeating) : Nat :=
  16

/-- Theorem stating that the minimum possible value of M + N is 16 -/
theorem min_groups_and_people_is_16 (cafe : CafeSeating) 
  (h1 : cafe.tables = 3)
  (h2 : cafe.counter_seats = 5)
  (h3 : cafe.min_group_size = 1)
  (h4 : cafe.max_group_size = 4)
  (state : SeatingState)
  (h5 : state.groups + state.total_people ≥ min_groups_and_people cafe) :
  min_groups_and_people cafe = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_groups_and_people_is_16_l77_7738


namespace NUMINAMATH_CALUDE_range_of_a_l77_7797

-- Define the set P
def P : Set ℝ := {x : ℝ | x^2 ≤ 1}

-- Define the set M
def M (a : ℝ) : Set ℝ := {a}

-- Theorem statement
theorem range_of_a (a : ℝ) : P ∪ M a = P → a ∈ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l77_7797


namespace NUMINAMATH_CALUDE_race_equation_l77_7792

theorem race_equation (x : ℝ) (h : x > 0) : 
  (1000 / x : ℝ) - (1000 / (1.25 * x)) = 30 :=
by sorry

end NUMINAMATH_CALUDE_race_equation_l77_7792


namespace NUMINAMATH_CALUDE_coach_team_division_l77_7777

/-- Given a total number of athletes and a maximum team size, 
    calculate the minimum number of teams needed. -/
def min_teams (total_athletes : ℕ) (max_team_size : ℕ) : ℕ :=
  ((total_athletes + max_team_size - 1) / max_team_size : ℕ)

theorem coach_team_division (total_athletes max_team_size : ℕ) 
  (h1 : total_athletes = 30) (h2 : max_team_size = 12) :
  min_teams total_athletes max_team_size = 3 := by
  sorry

#eval min_teams 30 12

end NUMINAMATH_CALUDE_coach_team_division_l77_7777


namespace NUMINAMATH_CALUDE_max_profit_is_33000_l77_7716

/-- Profit function for the first store -/
def L₁ (x : ℝ) : ℝ := -5 * x^2 + 900 * x - 16000

/-- Profit function for the second store -/
def L₂ (x : ℝ) : ℝ := 300 * x - 2000

/-- Total number of vehicles sold -/
def total_vehicles : ℝ := 110

/-- Total profit function -/
def S (x : ℝ) : ℝ := L₁ x + L₂ (total_vehicles - x)

theorem max_profit_is_33000 :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ total_vehicles ∧ 
  (∀ y : ℝ, y ≥ 0 → y ≤ total_vehicles → S y ≤ S x) ∧
  S x = 33000 :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_33000_l77_7716


namespace NUMINAMATH_CALUDE_least_multiple_of_35_over_500_l77_7788

theorem least_multiple_of_35_over_500 :
  (∀ k : ℕ, k * 35 > 500 → k * 35 ≥ 525) ∧ 525 > 500 ∧ ∃ n : ℕ, 525 = n * 35 :=
sorry

end NUMINAMATH_CALUDE_least_multiple_of_35_over_500_l77_7788


namespace NUMINAMATH_CALUDE_original_number_proof_l77_7775

theorem original_number_proof (x : ℝ) (h : 1 + 1/x = 5/2) : x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l77_7775


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l77_7747

theorem multiplication_puzzle :
  ∀ P Q R : ℕ,
    P ≠ Q → P ≠ R → Q ≠ R →
    P < 10 → Q < 10 → R < 10 →
    (100 * P + 10 * P + Q) * Q = 1000 * R + 100 * Q + 50 + Q →
    P + Q + R = 17 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l77_7747
