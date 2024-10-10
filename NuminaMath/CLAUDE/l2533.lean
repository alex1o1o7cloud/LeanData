import Mathlib

namespace inverse_proportion_function_l2533_253385

/-- Given that y is inversely proportional to x and y = 1 when x = 2,
    prove that the function expression of y with respect to x is y = 2/x. -/
theorem inverse_proportion_function (x : ℝ) (y : ℝ → ℝ) (k : ℝ) :
  (∀ x ≠ 0, y x = k / x) →  -- y is inversely proportional to x
  y 2 = 1 →                 -- when x = 2, y = 1
  ∀ x ≠ 0, y x = 2 / x :=   -- the function expression is y = 2/x
by
  sorry


end inverse_proportion_function_l2533_253385


namespace small_circle_area_l2533_253321

/-- Configuration of circles -/
structure CircleConfiguration where
  large_circle_area : ℝ
  small_circle_count : ℕ
  small_circles_inscribed : Prop

/-- Theorem: In a configuration where 6 small circles of equal radius are inscribed 
    in a large circle with an area of 120, the area of each small circle is 40 -/
theorem small_circle_area 
  (config : CircleConfiguration) 
  (h1 : config.large_circle_area = 120)
  (h2 : config.small_circle_count = 6)
  (h3 : config.small_circles_inscribed) :
  ∃ (small_circle_area : ℝ), small_circle_area = 40 ∧ 
    config.small_circle_count * small_circle_area = config.large_circle_area :=
by
  sorry


end small_circle_area_l2533_253321


namespace can_form_triangle_l2533_253382

/-- Triangle inequality theorem: the sum of the lengths of any two sides 
    of a triangle must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: The line segments 5, 6, and 10 can form a triangle -/
theorem can_form_triangle : triangle_inequality 5 6 10 := by
  sorry

end can_form_triangle_l2533_253382


namespace mason_savings_l2533_253336

theorem mason_savings (savings : ℝ) (total_books : ℕ) (book_price : ℝ) 
  (h1 : savings > 0) 
  (h2 : total_books > 0) 
  (h3 : book_price > 0) 
  (h4 : (1/4) * savings = (2/5) * total_books * book_price) : 
  savings - total_books * book_price = (3/8) * savings := by
sorry

end mason_savings_l2533_253336


namespace rolling_circle_traces_line_l2533_253315

/-- A circle with radius R -/
structure SmallCircle (R : ℝ) where
  center : ℝ × ℝ
  radius : ℝ
  radius_eq : radius = R

/-- A circle with radius 2R -/
structure LargeCircle (R : ℝ) where
  center : ℝ × ℝ
  radius : ℝ
  radius_eq : radius = 2 * R

/-- A point on the circumference of the small circle -/
def PointOnSmallCircle (R : ℝ) (sc : SmallCircle R) : Type :=
  { p : ℝ × ℝ // (p.1 - sc.center.1)^2 + (p.2 - sc.center.2)^2 = R^2 }

/-- The path traced by a point on the small circle as it rolls inside the large circle -/
def TracedPath (R : ℝ) (sc : SmallCircle R) (lc : LargeCircle R) (p : PointOnSmallCircle R sc) : Set (ℝ × ℝ) :=
  sorry

/-- The statement that the traced path is a straight line -/
def IsStraitLine (path : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- The main theorem -/
theorem rolling_circle_traces_line (R : ℝ) (sc : SmallCircle R) (lc : LargeCircle R) 
  (p : PointOnSmallCircle R sc) : 
  IsStraitLine (TracedPath R sc lc p) :=
sorry

end rolling_circle_traces_line_l2533_253315


namespace negative_integer_equation_solution_l2533_253337

theorem negative_integer_equation_solution :
  ∃ (N : ℤ), N < 0 ∧ 3 * N^2 + N = 15 → N = -3 :=
by sorry

end negative_integer_equation_solution_l2533_253337


namespace sum_of_four_cubes_1944_l2533_253388

theorem sum_of_four_cubes_1944 : ∃ (a b c d : ℤ), 1944 = a^3 + b^3 + c^3 + d^3 := by
  sorry

end sum_of_four_cubes_1944_l2533_253388


namespace function_monotonicity_l2533_253390

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a + a^x else 3 + (a - 1) * x

-- State the theorem
theorem function_monotonicity (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) →
  a ≥ 2 :=
sorry

end function_monotonicity_l2533_253390


namespace power_of_product_l2533_253393

theorem power_of_product (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end power_of_product_l2533_253393


namespace marlas_driving_time_l2533_253387

/-- The time Marla spends driving one way to her son's school -/
def driving_time : ℕ := sorry

/-- The total time Marla spends on the errand -/
def total_time : ℕ := 110

/-- The time Marla spends at parent-teacher night -/
def parent_teacher_time : ℕ := 70

/-- Theorem stating that the driving time is 20 minutes -/
theorem marlas_driving_time : driving_time = 20 :=
by
  have h1 : total_time = driving_time + parent_teacher_time + driving_time :=
    sorry
  sorry

end marlas_driving_time_l2533_253387


namespace min_value_of_reciprocal_sum_l2533_253367

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem min_value_of_reciprocal_sum (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + a 2014 = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 2) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 2) :=
sorry

end min_value_of_reciprocal_sum_l2533_253367


namespace age_problem_l2533_253355

theorem age_problem (x : ℝ) : (1/2) * (8 * (x + 8) - 8 * (x - 8)) = x ↔ x = 64 := by
  sorry

end age_problem_l2533_253355


namespace tangent_line_and_inequality_l2533_253389

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) * (Real.exp x + x + 2)

theorem tangent_line_and_inequality 
  (a b : ℝ) 
  (h1 : f a b 0 = 0) 
  (h2 : (deriv (f a b)) 0 = 6) :
  (a = 2 ∧ b = 0) ∧ 
  ∀ x > 0, f 2 0 x > 2 * Real.log x + 2 * x + 3 :=
sorry

end tangent_line_and_inequality_l2533_253389


namespace cube_root_of_negative_27_l2533_253319

theorem cube_root_of_negative_27 : ∃ x : ℝ, x^3 = -27 ∧ x = -3 := by sorry

end cube_root_of_negative_27_l2533_253319


namespace nico_reading_proof_l2533_253333

/-- The number of pages Nico read on Monday -/
def pages_monday : ℕ := 39

/-- The number of pages Nico read on Tuesday -/
def pages_tuesday : ℕ := 12

/-- The total number of pages Nico read over three days -/
def total_pages : ℕ := 51

/-- The number of books Nico borrowed -/
def num_books : ℕ := 3

/-- The number of days Nico read -/
def num_days : ℕ := 3

theorem nico_reading_proof :
  pages_monday = total_pages - pages_tuesday ∧
  pages_monday + pages_tuesday ≤ total_pages ∧
  num_books = num_days := by sorry

end nico_reading_proof_l2533_253333


namespace units_digit_of_n_l2533_253399

/-- Given two natural numbers m and n, where mn = 17^6 and m has a units digit of 8,
    prove that the units digit of n is 2. -/
theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 17^6) (h2 : m % 10 = 8) : n % 10 = 2 := by
  sorry

end units_digit_of_n_l2533_253399


namespace lcm_gcd_product_l2533_253360

theorem lcm_gcd_product (a b : ℕ) (ha : a = 240) (hb : b = 360) :
  Nat.lcm a b * Nat.gcd a b = 17280 := by
  sorry

end lcm_gcd_product_l2533_253360


namespace employee_pay_l2533_253356

/-- Proves that employee y is paid 268.18 per week given the conditions -/
theorem employee_pay (x y : ℝ) (h1 : x + y = 590) (h2 : x = 1.2 * y) : y = 268.18 := by
  sorry

end employee_pay_l2533_253356


namespace merchant_problem_l2533_253370

theorem merchant_problem (n : ℕ) (C : ℕ) : 
  (8 * n = C + 3) → 
  (7 * n = C - 4) → 
  (n = 7 ∧ C = 53) := by
  sorry

end merchant_problem_l2533_253370


namespace place_value_ratio_l2533_253372

-- Define the number
def number : ℚ := 53674.9281

-- Define the place value of a digit at a specific position
def place_value (n : ℚ) (pos : ℤ) : ℚ := 10 ^ pos

-- Define the position of digit 6 (counting from right, with decimal point at 0)
def pos_6 : ℤ := 3

-- Define the position of digit 8 (counting from right, with decimal point at 0)
def pos_8 : ℤ := -1

-- Theorem to prove
theorem place_value_ratio :
  (place_value number pos_6) / (place_value number pos_8) = 10000 := by
  sorry

end place_value_ratio_l2533_253372


namespace outstanding_consumer_credit_l2533_253391

/-- The total outstanding consumer installment credit in billions of dollars -/
def total_credit : ℝ := 855

/-- The automobile installment credit in billions of dollars -/
def auto_credit : ℝ := total_credit * 0.2

/-- The credit extended by automobile finance companies in billions of dollars -/
def finance_company_credit : ℝ := 57

theorem outstanding_consumer_credit :
  (auto_credit = total_credit * 0.2) ∧
  (finance_company_credit = 57) ∧
  (finance_company_credit = auto_credit / 3) →
  total_credit = 855 :=
by sorry

end outstanding_consumer_credit_l2533_253391


namespace average_of_a_and_b_l2533_253398

theorem average_of_a_and_b (a b c : ℝ) : 
  (4 + 6 + 8 + 12 + a + b + c) / 7 = 20 →
  a + b + c = 3 * ((4 + 6 + 8) / 3) →
  (a + b) / 2 = (18 - c) / 2 := by
sorry

end average_of_a_and_b_l2533_253398


namespace spending_ratio_l2533_253331

def monthly_allowance : ℚ := 12

def spending_scenario (first_week_spending : ℚ) : Prop :=
  let remaining_after_first_week := monthly_allowance - first_week_spending
  let second_week_spending := (1 / 4) * remaining_after_first_week
  monthly_allowance - first_week_spending - second_week_spending = 6

theorem spending_ratio :
  ∃ (first_week_spending : ℚ),
    spending_scenario first_week_spending ∧
    first_week_spending / monthly_allowance = 1 / 3 := by
  sorry

end spending_ratio_l2533_253331


namespace problem_statement_l2533_253394

theorem problem_statement (a b c : ℝ) 
  (h1 : a^2 + a*b = c) 
  (h2 : a*b + b^2 = c + 5) : 
  (2*c + 5 ≥ 0) ∧ 
  (a^2 - b^2 = -5) ∧ 
  (a ≠ b ∧ a ≠ -b) := by
sorry

end problem_statement_l2533_253394


namespace middle_circle_radius_l2533_253327

/-- A sequence of five circles tangent to two parallel lines and to each other -/
structure CircleSequence where
  radii : Fin 5 → ℝ
  tangent_to_lines : Bool
  sequentially_tangent : Bool

/-- The property that the radii form a geometric sequence -/
def is_geometric_sequence (cs : CircleSequence) : Prop :=
  ∃ r : ℝ, ∀ i : Fin 4, cs.radii i.succ = cs.radii i * r

theorem middle_circle_radius 
  (cs : CircleSequence)
  (h_tangent : cs.tangent_to_lines = true)
  (h_seq_tangent : cs.sequentially_tangent = true)
  (h_geometric : is_geometric_sequence cs)
  (h_smallest : cs.radii 0 = 8)
  (h_largest : cs.radii 4 = 18) :
  cs.radii 2 = 12 := by
  sorry

end middle_circle_radius_l2533_253327


namespace duty_arrangements_count_l2533_253373

/- Define the number of days -/
def num_days : ℕ := 7

/- Define the number of people -/
def num_people : ℕ := 4

/- Define the possible work days for each person -/
def work_days : Set ℕ := {1, 2}

/- Define the function to calculate the number of duty arrangements -/
def duty_arrangements (days : ℕ) (people : ℕ) (work_options : Set ℕ) : ℕ :=
  sorry  -- The actual calculation would go here

/- Theorem stating that the number of duty arrangements is 2520 -/
theorem duty_arrangements_count :
  duty_arrangements num_days num_people work_days = 2520 :=
sorry

end duty_arrangements_count_l2533_253373


namespace modulus_of_z_equals_one_l2533_253347

open Complex

theorem modulus_of_z_equals_one (z : ℂ) (h : z * (1 + I) = 1 - I) : abs z = 1 := by
  sorry

end modulus_of_z_equals_one_l2533_253347


namespace original_jeans_price_l2533_253312

/-- Proves that the original price of jeans is $49.00 given the discount conditions --/
theorem original_jeans_price (x : ℝ) : 
  (0.5 * x - 10 = 14.5) → x = 49 := by
  sorry

end original_jeans_price_l2533_253312


namespace min_value_expression_equality_condition_l2533_253353

theorem min_value_expression (x : ℝ) (h : x > 0) : x^2 + 8*x + 64/x^3 ≥ 28 := by
  sorry

theorem equality_condition : ∃ x > 0, x^2 + 8*x + 64/x^3 = 28 := by
  sorry

end min_value_expression_equality_condition_l2533_253353


namespace rain_probability_l2533_253307

theorem rain_probability (p : ℝ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 4) :
  1 - (1 - p)^n = 255/256 := by
  sorry

end rain_probability_l2533_253307


namespace canvas_area_l2533_253306

/-- The area of a rectangular canvas inside a decorative border -/
theorem canvas_area (outer_width outer_height border_width : ℝ) : 
  outer_width = 100 →
  outer_height = 140 →
  border_width = 15 →
  (outer_width - 2 * border_width) * (outer_height - 2 * border_width) = 7700 := by
sorry

end canvas_area_l2533_253306


namespace max_distance_sparkling_points_l2533_253304

theorem max_distance_sparkling_points :
  ∀ (a₁ b₁ a₂ b₂ : ℝ),
    a₁^2 + b₁^2 = 1 →
    a₂^2 + b₂^2 = 1 →
    ∀ (d : ℝ),
      d = Real.sqrt ((a₂ - a₁)^2 + (b₂ - b₁)^2) →
      d ≤ 2 ∧ ∃ (a₁' b₁' a₂' b₂' : ℝ),
        a₁'^2 + b₁'^2 = 1 ∧
        a₂'^2 + b₂'^2 = 1 ∧
        Real.sqrt ((a₂' - a₁')^2 + (b₂' - b₁')^2) = 2 :=
by sorry

end max_distance_sparkling_points_l2533_253304


namespace square_of_1009_l2533_253366

theorem square_of_1009 : 1009 ^ 2 = 1018081 := by
  sorry

end square_of_1009_l2533_253366


namespace arithmetic_calculation_l2533_253348

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 30 / 3 + 2^3 = 77 := by
  sorry

end arithmetic_calculation_l2533_253348


namespace max_log_sum_l2533_253349

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  ∃ (max : ℝ), max = Real.log 4 ∧ ∀ z w : ℝ, z > 0 → w > 0 → z + w = 4 → Real.log z + Real.log w ≤ max :=
by sorry

end max_log_sum_l2533_253349


namespace minimum_score_for_eligibility_l2533_253383

def minimum_score (q1 q2 q3 : ℚ) (target_average : ℚ) : ℚ :=
  4 * target_average - (q1 + q2 + q3)

theorem minimum_score_for_eligibility 
  (q1 q2 q3 : ℚ) 
  (target_average : ℚ) 
  (h1 : q1 = 80) 
  (h2 : q2 = 85) 
  (h3 : q3 = 78) 
  (h4 : target_average = 85) :
  minimum_score q1 q2 q3 target_average = 97 := by
sorry

end minimum_score_for_eligibility_l2533_253383


namespace percentage_green_shirts_l2533_253392

theorem percentage_green_shirts (total_students : ℕ) (blue_percent red_percent : ℚ) (other_students : ℕ) :
  total_students = 600 →
  blue_percent = 45/100 →
  red_percent = 23/100 →
  other_students = 102 →
  (total_students - (blue_percent * total_students + red_percent * total_students + other_students)) / total_students = 15/100 := by
  sorry

end percentage_green_shirts_l2533_253392


namespace car_repair_cost_johns_car_repair_cost_l2533_253332

/-- Calculates the total cost of car repairs given labor rate, hours worked, and part cost -/
theorem car_repair_cost (labor_rate : ℕ) (hours : ℕ) (part_cost : ℕ) : 
  labor_rate * hours + part_cost = 2400 :=
by
  sorry

/-- Proves the specific case of John's car repair cost -/
theorem johns_car_repair_cost : 
  75 * 16 + 1200 = 2400 :=
by
  sorry

end car_repair_cost_johns_car_repair_cost_l2533_253332


namespace infinitely_many_pairs_dividing_powers_l2533_253363

theorem infinitely_many_pairs_dividing_powers (d : ℤ) 
  (h1 : d > 1) (h2 : d % 4 = 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (a + b : ℤ) ∣ (a^b + b^a : ℤ) := by
  sorry

#check infinitely_many_pairs_dividing_powers

end infinitely_many_pairs_dividing_powers_l2533_253363


namespace tan_strictly_increasing_interval_l2533_253386

open Real

noncomputable def f (x : ℝ) : ℝ := tan (x - π / 4)

theorem tan_strictly_increasing_interval (k : ℤ) :
  StrictMonoOn f (Set.Ioo (k * π - π / 4) (k * π + 3 * π / 4)) := by
  sorry

end tan_strictly_increasing_interval_l2533_253386


namespace fixed_point_values_l2533_253358

/-- A function has exactly one fixed point if and only if
    the equation f(x) = x has exactly one solution. -/
def has_exactly_one_fixed_point (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = x

/-- The quadratic function f(x) = ax² + (2a-3)x + 1 -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + (2*a - 3) * x + 1

/-- The set of values for a such that f has exactly one fixed point -/
def A : Set ℝ := {a | has_exactly_one_fixed_point (f a)}

theorem fixed_point_values :
  A = {0, 1, 4} := by sorry

end fixed_point_values_l2533_253358


namespace system_solution_l2533_253324

/-- The solution to the system of equations:
     4x - 3y = -2.4
     5x + 6y = 7.5
-/
theorem system_solution :
  ∃ (x y : ℝ), 
    (4 * x - 3 * y = -2.4) ∧
    (5 * x + 6 * y = 7.5) ∧
    (x = 2.7 / 13) ∧
    (y = 1.0769) := by
  sorry

end system_solution_l2533_253324


namespace points_collinear_collinear_vectors_l2533_253314

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Non-zero vectors e₁ and e₂ are not collinear -/
def not_collinear (e₁ e₂ : V) : Prop :=
  e₁ ≠ 0 ∧ e₂ ≠ 0 ∧ ∀ (r : ℝ), e₁ ≠ r • e₂

variable (e₁ e₂ : V) (h : not_collinear e₁ e₂)

/-- Vector AB -/
def AB : V := e₁ + e₂

/-- Vector BC -/
def BC : V := 2 • e₁ + 8 • e₂

/-- Vector CD -/
def CD : V := 3 • (e₁ - e₂)

/-- Three points are collinear if the vector between any two is a scalar multiple of the vector between the other two -/
def collinear (A B D : V) : Prop :=
  ∃ (r : ℝ), B - A = r • (D - B) ∨ D - B = r • (B - A)

theorem points_collinear :
  collinear (0 : V) (AB e₁ e₂) ((AB e₁ e₂) + (BC e₁ e₂) + (CD e₁ e₂)) :=
sorry

theorem collinear_vectors (k : ℝ) :
  (∃ (r : ℝ), k • e₁ + e₂ = r • (e₁ + k • e₂)) ↔ k = 1 ∨ k = -1 :=
sorry

end points_collinear_collinear_vectors_l2533_253314


namespace worker_y_defective_rate_l2533_253338

-- Define the fractions and percentages
def worker_x_fraction : ℝ := 1 - 0.1666666666666668
def worker_y_fraction : ℝ := 0.1666666666666668
def worker_x_defective_rate : ℝ := 0.005
def total_defective_rate : ℝ := 0.0055

-- Theorem statement
theorem worker_y_defective_rate :
  ∃ (y_rate : ℝ),
    y_rate = 0.008 ∧
    total_defective_rate = worker_x_fraction * worker_x_defective_rate + worker_y_fraction * y_rate :=
by sorry

end worker_y_defective_rate_l2533_253338


namespace sum_powers_l2533_253301

theorem sum_powers (a b c d : ℝ) 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  (a^5 + b^5 = c^5 + d^5) ∧ 
  (∃ (a b c d : ℝ), (a + b = c + d) ∧ (a^3 + b^3 = c^3 + d^3) ∧ (a^4 + b^4 ≠ c^4 + d^4)) :=
by sorry

end sum_powers_l2533_253301


namespace probability_x_plus_y_leq_6_l2533_253379

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 5}

-- Define the region where x + y ≤ 6
def region : Set (ℝ × ℝ) := {p ∈ rectangle | p.1 + p.2 ≤ 6}

-- Define the probability measure on the rectangle
noncomputable def prob : MeasureTheory.Measure (ℝ × ℝ) := sorry

-- State the theorem
theorem probability_x_plus_y_leq_6 : 
  prob region / prob rectangle = 1 / 2 := by sorry

end probability_x_plus_y_leq_6_l2533_253379


namespace fraction_equality_l2533_253362

theorem fraction_equality (x y z : ℝ) 
  (hx : x ≠ 1) (hy : y ≠ 1) (hxy : x ≠ y) 
  (h : (y * z - x^2) / (1 - x) = (x * z - y^2) / (1 - y)) : 
  (y * z - x^2) / (1 - x) = x + y + z ∧ (x * z - y^2) / (1 - y) = x + y + z := by
  sorry

end fraction_equality_l2533_253362


namespace A_work_time_l2533_253329

/-- The number of days it takes B to complete the work alone -/
def B_days : ℝ := 8

/-- The total payment for the work -/
def total_payment : ℝ := 3600

/-- The payment to C -/
def C_payment : ℝ := 450

/-- The number of days it takes A, B, and C to complete the work together -/
def combined_days : ℝ := 3

/-- The number of days it takes A to complete the work alone -/
def A_days : ℝ := 56

theorem A_work_time :
  ∃ (C_rate : ℝ),
    (1 / A_days + 1 / B_days + C_rate = 1 / combined_days) ∧
    (1 / A_days : ℝ) / (1 / B_days) = (total_payment - C_payment) / C_payment :=
by sorry

end A_work_time_l2533_253329


namespace R_squared_eq_one_when_no_error_l2533_253359

/-- A structure representing a set of observations in a linear regression model. -/
structure LinearRegressionData (n : ℕ) where
  x : Fin n → ℝ
  y : Fin n → ℝ
  a : ℝ
  b : ℝ
  e : Fin n → ℝ

/-- The coefficient of determination (R-squared) for a linear regression model. -/
def R_squared (data : LinearRegressionData n) : ℝ :=
  sorry

/-- Theorem stating that if all error terms are zero, then R-squared equals 1. -/
theorem R_squared_eq_one_when_no_error (n : ℕ) (data : LinearRegressionData n)
  (h1 : ∀ i, data.y i = data.b * data.x i + data.a + data.e i)
  (h2 : ∀ i, data.e i = 0) :
  R_squared data = 1 :=
sorry

end R_squared_eq_one_when_no_error_l2533_253359


namespace max_perfect_squares_l2533_253374

theorem max_perfect_squares (n : ℕ) : 
  (∃ (S : Finset ℕ), 
    (∀ k ∈ S, 1 ≤ k ∧ k ≤ 2015 ∧ ∃ m : ℕ, 240 * k = m^2) ∧ 
    S.card = n) →
  n ≤ 11 :=
sorry

end max_perfect_squares_l2533_253374


namespace f_derivative_at_zero_l2533_253334

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem f_derivative_at_zero : 
  deriv f 0 = 1 := by sorry

end f_derivative_at_zero_l2533_253334


namespace ticket_price_possibilities_l2533_253318

theorem ticket_price_possibilities (x : ℕ) : 
  (∃ n m : ℕ, n * x = 72 ∧ m * x = 108 ∧ Even x) ↔ 
  x ∈ ({2, 4, 6, 12, 18, 36} : Set ℕ) :=
sorry

end ticket_price_possibilities_l2533_253318


namespace power_function_increasing_l2533_253380

theorem power_function_increasing (m : ℝ) : 
  (∀ x > 0, ∀ h > 0, (m^2 - 4*m + 1) * x^(m^2 - 2*m - 3) < (m^2 - 4*m + 1) * (x + h)^(m^2 - 2*m - 3)) ↔ 
  m = 4 := by
sorry

end power_function_increasing_l2533_253380


namespace scalene_triangle_two_angles_less_than_60_l2533_253341

/-- A scalene triangle with side lengths in arithmetic progression has two angles less than 60 degrees. -/
theorem scalene_triangle_two_angles_less_than_60 (a d : ℝ) 
  (h_d_pos : d > 0) 
  (h_scalene : a - d ≠ a ∧ a ≠ a + d ∧ a - d ≠ a + d) :
  ∃ (α β : ℝ), α + β + (180 - α - β) = 180 ∧ 
               0 < α ∧ α < 60 ∧ 
               0 < β ∧ β < 60 := by
  sorry


end scalene_triangle_two_angles_less_than_60_l2533_253341


namespace cranberries_left_l2533_253310

/-- The number of cranberries left in a bog after harvesting and elk consumption -/
theorem cranberries_left (total : ℕ) (harvest_percent : ℚ) (elk_eaten : ℕ) 
  (h1 : total = 60000)
  (h2 : harvest_percent = 40 / 100)
  (h3 : elk_eaten = 20000) :
  total - (total * harvest_percent).floor - elk_eaten = 16000 := by
  sorry

#check cranberries_left

end cranberries_left_l2533_253310


namespace purely_imaginary_implies_m_eq_neg_three_l2533_253328

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of a real parameter m. -/
def z (m : ℝ) : ℂ :=
  Complex.mk (m^2 + m - 6) (m - 2)

/-- If z(m) is purely imaginary, then m = -3. -/
theorem purely_imaginary_implies_m_eq_neg_three :
  ∀ m : ℝ, is_purely_imaginary (z m) → m = -3 := by
  sorry

end purely_imaginary_implies_m_eq_neg_three_l2533_253328


namespace cost_increase_percentage_l2533_253316

theorem cost_increase_percentage (initial_cost final_cost : ℝ) 
  (h1 : initial_cost = 75)
  (h2 : final_cost = 72)
  (h3 : ∃ x : ℝ, final_cost = (initial_cost + (x / 100) * initial_cost) * 0.8) :
  ∃ x : ℝ, x = 20 ∧ final_cost = (initial_cost + (x / 100) * initial_cost) * 0.8 := by
  sorry

end cost_increase_percentage_l2533_253316


namespace misha_phone_number_l2533_253302

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.reverse = digits

def is_consecutive (a b c : ℕ) : Prop :=
  b = a + 1 ∧ c = b + 1

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0))

theorem misha_phone_number :
  ∃! n : ℕ,
    n ≥ 1000000 ∧ n < 10000000 ∧
    is_palindrome (n / 100) ∧
    is_consecutive (n % 10) ((n / 10) % 10) ((n / 100) % 10) ∧
    (n / 10000) % 9 = 0 ∧
    ∃ i : ℕ, i < 5 → (n / (10^i)) % 1000 = 111 ∧
    (is_prime ((n / 100) % 100) ∨ is_prime (n % 100)) ∧
    n = 7111765 :=
  sorry

end misha_phone_number_l2533_253302


namespace gcd_problem_l2533_253376

theorem gcd_problem :
  ∃! n : ℕ, 30 ≤ n ∧ n ≤ 40 ∧ Nat.gcd n 15 = 5 :=
by
  -- The proof would go here
  sorry

end gcd_problem_l2533_253376


namespace shaded_area_is_six_l2533_253313

/-- Represents a quadrilateral divided into four smaller quadrilaterals -/
structure DividedQuadrilateral where
  /-- Areas of the four smaller quadrilaterals -/
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ
  /-- The sum of areas of two opposite quadrilaterals is 28 -/
  sum_opposite : area1 + area3 = 28
  /-- One of the quadrilaterals has an area of 8 -/
  known_area : area2 = 8

/-- The theorem to be proved -/
theorem shaded_area_is_six (q : DividedQuadrilateral) : q.area4 = 6 := by
  sorry

end shaded_area_is_six_l2533_253313


namespace intersection_single_point_l2533_253300

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
def B (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

-- State the theorem
theorem intersection_single_point (r : ℝ) (h_r : r > 0) 
  (h_intersection : ∃! p, p ∈ A ∩ B r) : 
  r = 3 ∨ r = 7 := by sorry

end intersection_single_point_l2533_253300


namespace carnations_in_third_bouquet_l2533_253325

theorem carnations_in_third_bouquet 
  (total_bouquets : ℕ)
  (first_bouquet : ℕ)
  (second_bouquet : ℕ)
  (average_carnations : ℕ)
  (h1 : total_bouquets = 3)
  (h2 : first_bouquet = 9)
  (h3 : second_bouquet = 14)
  (h4 : average_carnations = 12) :
  average_carnations * total_bouquets - (first_bouquet + second_bouquet) = 13 :=
by
  sorry

#check carnations_in_third_bouquet

end carnations_in_third_bouquet_l2533_253325


namespace equation_equivalence_l2533_253339

theorem equation_equivalence (x y : ℝ) :
  x^2 * (y + y^2) = y^3 + x^4 ↔ y = x ∨ y = -x ∨ y = x^2 := by
  sorry

end equation_equivalence_l2533_253339


namespace john_took_six_pink_l2533_253340

def initial_pink : ℕ := 26
def initial_green : ℕ := 15
def initial_yellow : ℕ := 24
def carl_took : ℕ := 4
def total_remaining : ℕ := 43

def john_took_pink (p : ℕ) : Prop :=
  initial_pink - carl_took - p +
  initial_green - 2 * p +
  initial_yellow = total_remaining

theorem john_took_six_pink : john_took_pink 6 := by sorry

end john_took_six_pink_l2533_253340


namespace right_triangle_hypotenuse_l2533_253384

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 6 → 
    b = 8 → 
    c^2 = a^2 + b^2 → 
    c = 10 := by
  sorry

end right_triangle_hypotenuse_l2533_253384


namespace max_valid_ltrominos_eighteen_is_achievable_l2533_253330

-- Define the colors
inductive Color
  | Red
  | Green
  | Blue

-- Define the grid
def Grid := Fin 4 → Fin 4 → Color

-- Define an L-tromino
structure LTromino where
  x : Fin 4
  y : Fin 4
  orientation : Fin 4

-- Function to check if an L-tromino has one square of each color
def hasOneOfEachColor (g : Grid) (l : LTromino) : Bool := sorry

-- Function to count valid L-trominos in a grid
def countValidLTrominos (g : Grid) : Nat := sorry

-- Theorem statement
theorem max_valid_ltrominos (g : Grid) : 
  countValidLTrominos g ≤ 18 := sorry

-- Theorem stating that 18 is achievable
theorem eighteen_is_achievable : 
  ∃ g : Grid, countValidLTrominos g = 18 := sorry

end max_valid_ltrominos_eighteen_is_achievable_l2533_253330


namespace soccer_ball_max_height_l2533_253342

/-- The height function of a soccer ball kicked vertically -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

/-- The maximum height achieved by the soccer ball -/
def max_height : ℝ := 40

/-- Theorem stating that the maximum height of the soccer ball is 40 feet -/
theorem soccer_ball_max_height :
  ∀ t : ℝ, h t ≤ max_height :=
by sorry

end soccer_ball_max_height_l2533_253342


namespace distance_to_y_axis_angle_bisector_and_line_l2533_253309

-- Define the point M
def M (m : ℝ) : ℝ × ℝ := (2 - m, 1 + 2*m)

-- Part 1
theorem distance_to_y_axis (m : ℝ) :
  abs (2 - m) = 2 → (M m = (2, 1) ∨ M m = (-2, 9)) :=
sorry

-- Part 2
theorem angle_bisector_and_line (m k b : ℝ) :
  (2 - m = 1 + 2*m) →  -- M lies on angle bisector
  ((2 - m) = k*(2 - m) + b) →  -- Line passes through M
  (0 = k*0 + b) →  -- Line passes through (0,5)
  (5 = k*0 + b) →
  (k = -2 ∧ b = 5) :=  -- Line equation is y = -2x + 5
sorry

end distance_to_y_axis_angle_bisector_and_line_l2533_253309


namespace intersection_of_A_and_B_l2533_253357

def A : Set ℕ := {0, 1, 2, 3, 4, 5}
def B : Set ℕ := {x | x^2 < 10}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2, 3} := by sorry

end intersection_of_A_and_B_l2533_253357


namespace cost_calculation_l2533_253308

/-- The cost of items given their quantities and price ratios -/
def cost_of_items (pen_price pencil_price eraser_price : ℚ) : ℚ :=
  4 * pen_price + 6 * pencil_price + 2 * eraser_price

/-- The cost of a dozen pens and half a dozen erasers -/
def cost_of_dozen_pens_and_half_dozen_erasers (pen_price eraser_price : ℚ) : ℚ :=
  12 * pen_price + 6 * eraser_price

theorem cost_calculation :
  ∀ (x : ℚ),
    cost_of_items (4*x) (2*x) x = 360 →
    cost_of_dozen_pens_and_half_dozen_erasers (4*x) x = 648 :=
by
  sorry

end cost_calculation_l2533_253308


namespace parabola_focus_coordinates_l2533_253371

/-- Given a quadratic function f(x) = ax^2 + bx + 2 where a ≠ 0 and |f(x)| ≥ 2 for all real x,
    the focus of the parabola has coordinates (0, 1/(4a) + 2). -/
theorem parabola_focus_coordinates (a b : ℝ) (ha : a ≠ 0) 
    (hf : ∀ x : ℝ, |a * x^2 + b * x + 2| ≥ 2) :
    ∃ (focus : ℝ × ℝ), focus = (0, 1 / (4 * a) + 2) := by
  sorry

end parabola_focus_coordinates_l2533_253371


namespace wire_length_ratio_l2533_253322

-- Define the given constants
def bonnie_wire_pieces : ℕ := 12
def bonnie_wire_length : ℕ := 8
def roark_wire_length : ℕ := 2

-- Define Bonnie's prism
def bonnie_prism_volume : ℕ := (bonnie_wire_length / 2) ^ 3

-- Define Roark's unit prism
def roark_unit_prism_volume : ℕ := roark_wire_length ^ 3

-- Define the number of Roark's prisms
def roark_prism_count : ℕ := bonnie_prism_volume / roark_unit_prism_volume

-- Define the total wire lengths
def bonnie_total_wire : ℕ := bonnie_wire_pieces * bonnie_wire_length
def roark_total_wire : ℕ := roark_prism_count * (12 * roark_wire_length)

-- Theorem to prove
theorem wire_length_ratio :
  bonnie_total_wire / roark_total_wire = 1 / 16 :=
by sorry

end wire_length_ratio_l2533_253322


namespace floor_of_4_7_l2533_253335

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end floor_of_4_7_l2533_253335


namespace sqrt_ratio_equality_implies_y_value_l2533_253396

theorem sqrt_ratio_equality_implies_y_value (y : ℝ) :
  y > 2 →
  (Real.sqrt (7 * y)) / (Real.sqrt (4 * (y - 2))) = 3 →
  y = 72 / 29 := by
  sorry

end sqrt_ratio_equality_implies_y_value_l2533_253396


namespace apartment_units_per_floor_l2533_253344

theorem apartment_units_per_floor (total_units : ℕ) (first_floor_units : ℕ) (num_buildings : ℕ) (num_floors : ℕ) :
  total_units = 34 →
  first_floor_units = 2 →
  num_buildings = 2 →
  num_floors = 4 →
  ∃ (other_floor_units : ℕ),
    total_units = num_buildings * (first_floor_units + (num_floors - 1) * other_floor_units) ∧
    other_floor_units = 5 :=
by sorry

end apartment_units_per_floor_l2533_253344


namespace wildcats_score_is_36_l2533_253368

/-- The score of the Panthers -/
def panthers_score : ℕ := 17

/-- The difference between the Wildcats' and Panthers' scores -/
def score_difference : ℕ := 19

/-- The score of the Wildcats -/
def wildcats_score : ℕ := panthers_score + score_difference

theorem wildcats_score_is_36 : wildcats_score = 36 := by
  sorry

end wildcats_score_is_36_l2533_253368


namespace circular_track_length_l2533_253346

theorem circular_track_length :
  ∀ (track_length : ℝ) (brenda_speed sally_speed : ℝ),
    brenda_speed > 0 →
    sally_speed > 0 →
    track_length / 2 - 120 = 120 * sally_speed / brenda_speed →
    track_length / 2 + 40 = (track_length / 2 - 80) * sally_speed / brenda_speed →
    track_length = 480 := by
  sorry

end circular_track_length_l2533_253346


namespace coin_game_theorem_l2533_253378

/-- Represents the result of a coin-taking game -/
inductive GameResult
| FirstPlayerWins
| SecondPlayerWins

/-- Defines the coin-taking game and determines the winner -/
def coinGameWinner (n : ℕ) : GameResult :=
  if n = 7 then GameResult.FirstPlayerWins
  else if n = 12 then GameResult.SecondPlayerWins
  else sorry -- For other cases

/-- Theorem stating the winner for specific game configurations -/
theorem coin_game_theorem :
  (coinGameWinner 7 = GameResult.FirstPlayerWins) ∧
  (coinGameWinner 12 = GameResult.SecondPlayerWins) := by
  sorry

/-- The maximum value of coins a player can take in one turn -/
def maxTakeValue : ℕ := 3

/-- The value of a two-pound coin -/
def twoPoundValue : ℕ := 2

/-- The value of a one-pound coin -/
def onePoundValue : ℕ := 1

end coin_game_theorem_l2533_253378


namespace investment_interest_rate_l2533_253303

theorem investment_interest_rate 
  (total_investment : ℝ) 
  (investment_at_r : ℝ) 
  (total_interest : ℝ) 
  (known_rate : ℝ) :
  total_investment = 10000 →
  investment_at_r = 7200 →
  known_rate = 0.09 →
  total_interest = 684 →
  ∃ r : ℝ, 
    r * investment_at_r + known_rate * (total_investment - investment_at_r) = total_interest ∧
    r = 0.06 :=
by sorry

end investment_interest_rate_l2533_253303


namespace intersection_and_sufficient_condition_l2533_253395

def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}
def B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 3}

theorem intersection_and_sufficient_condition :
  (A (-2) ∩ B = {x | -3 ≤ x ∧ x ≤ -2}) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ A a → x ∈ B) ∧ (∃ x : ℝ, x ∈ B ∧ x ∉ A a) ↔ a ≤ -3 ∨ a ≥ 4) :=
by sorry

end intersection_and_sufficient_condition_l2533_253395


namespace geometric_sequence_problem_l2533_253317

/-- A geometric sequence with a positive common ratio -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 * a 6 = 8 * a 4 →
  a 2 = 2 →
  a 1 = 1 := by
  sorry

end geometric_sequence_problem_l2533_253317


namespace regular_pay_is_three_dollars_l2533_253375

/-- Represents a worker's pay structure and hours worked -/
structure WorkerPay where
  regularPayPerHour : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalPay : ℝ

/-- Calculates the total pay for a worker given their pay structure and hours worked -/
def calculateTotalPay (w : WorkerPay) : ℝ :=
  w.regularPayPerHour * w.regularHours + 2 * w.regularPayPerHour * w.overtimeHours

/-- Theorem stating that under given conditions, the regular pay per hour is $3 -/
theorem regular_pay_is_three_dollars
  (w : WorkerPay)
  (h1 : w.regularHours = 40)
  (h2 : w.overtimeHours = 11)
  (h3 : w.totalPay = 186)
  (h4 : calculateTotalPay w = w.totalPay) :
  w.regularPayPerHour = 3 := by
  sorry

end regular_pay_is_three_dollars_l2533_253375


namespace james_writing_time_l2533_253377

/-- James' writing scenario -/
structure WritingScenario where
  pages_per_hour : ℕ
  pages_per_day_per_person : ℕ
  people_per_day : ℕ

/-- Calculate the hours spent writing per week -/
def hours_per_week (scenario : WritingScenario) : ℕ :=
  let pages_per_day := scenario.pages_per_day_per_person * scenario.people_per_day
  let pages_per_week := pages_per_day * 7
  pages_per_week / scenario.pages_per_hour

/-- Theorem stating James spends 7 hours a week writing -/
theorem james_writing_time (james : WritingScenario)
  (h1 : james.pages_per_hour = 10)
  (h2 : james.pages_per_day_per_person = 5)
  (h3 : james.people_per_day = 2) :
  hours_per_week james = 7 := by
  sorry

end james_writing_time_l2533_253377


namespace integer_roots_of_polynomial_l2533_253345

def polynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^3 + b₂*x^2 + b₁*x + 18

def divisors_of_18 : Set ℤ := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  {x : ℤ | polynomial b₂ b₁ x = 0} = divisors_of_18 :=
sorry

end integer_roots_of_polynomial_l2533_253345


namespace total_paintable_area_is_876_l2533_253354

/-- The number of bedrooms in Isabella's house -/
def num_bedrooms : ℕ := 3

/-- The length of each bedroom in feet -/
def bedroom_length : ℕ := 12

/-- The width of each bedroom in feet -/
def bedroom_width : ℕ := 10

/-- The height of each bedroom in feet -/
def bedroom_height : ℕ := 8

/-- The area occupied by doorways and windows in each bedroom in square feet -/
def unpaintable_area : ℕ := 60

/-- The total area of walls to be painted in all bedrooms -/
def total_paintable_area : ℕ :=
  num_bedrooms * (
    2 * (bedroom_length * bedroom_height + bedroom_width * bedroom_height) - unpaintable_area
  )

/-- Theorem stating that the total area to be painted is 876 square feet -/
theorem total_paintable_area_is_876 : total_paintable_area = 876 := by
  sorry

end total_paintable_area_is_876_l2533_253354


namespace rectangle_width_is_five_l2533_253326

/-- A rectangle with specific properties -/
structure Rectangle where
  length : ℝ
  width : ℝ
  width_longer : width = length + 2
  perimeter : length * 2 + width * 2 = 16

/-- The width of the rectangle is 5 -/
theorem rectangle_width_is_five (r : Rectangle) : r.width = 5 := by
  sorry

end rectangle_width_is_five_l2533_253326


namespace camping_trip_percentage_l2533_253351

theorem camping_trip_percentage (total_students : ℕ) 
  (h1 : (16 : ℚ) / 100 * total_students = (25 : ℚ) / 100 * camping_students)
  (h2 : (75 : ℚ) / 100 * camping_students + (16 : ℚ) / 100 * total_students = camping_students)
  (camping_students : ℕ) :
  (camping_students : ℚ) / total_students = 64 / 100 :=
by
  sorry


end camping_trip_percentage_l2533_253351


namespace pizza_promotion_savings_l2533_253369

/-- The regular price of a medium pizza in dollars -/
def regular_price : ℝ := 18

/-- The promotional price of a medium pizza in dollars -/
def promo_price : ℝ := 5

/-- The number of pizzas eligible for the promotion -/
def num_pizzas : ℕ := 3

/-- The total savings when buying the promotional pizzas -/
def total_savings : ℝ := num_pizzas * (regular_price - promo_price)

theorem pizza_promotion_savings : total_savings = 39 := by
  sorry

end pizza_promotion_savings_l2533_253369


namespace train_crossing_time_specific_train_crossing_time_l2533_253352

/-- Proves that a train with given length, crossing its own length in a certain time,
    takes the calculated time to cross a platform of given length. -/
theorem train_crossing_time (train_length platform_length cross_own_length_time : ℝ) 
    (train_length_pos : 0 < train_length)
    (platform_length_pos : 0 < platform_length)
    (cross_own_length_time_pos : 0 < cross_own_length_time) :
  let train_speed := train_length / cross_own_length_time
  let total_distance := train_length + platform_length
  let crossing_time := total_distance / train_speed
  crossing_time = 45 :=
by
  sorry

/-- Specific instance of the train crossing problem -/
theorem specific_train_crossing_time :
  let train_length := 300
  let platform_length := 450
  let cross_own_length_time := 18
  let train_speed := train_length / cross_own_length_time
  let total_distance := train_length + platform_length
  let crossing_time := total_distance / train_speed
  crossing_time = 45 :=
by
  sorry

end train_crossing_time_specific_train_crossing_time_l2533_253352


namespace fred_book_purchase_l2533_253323

theorem fred_book_purchase (initial_amount remaining_amount cost_per_book : ℕ) 
  (h1 : initial_amount = 236)
  (h2 : remaining_amount = 14)
  (h3 : cost_per_book = 37) :
  (initial_amount - remaining_amount) / cost_per_book = 6 := by
  sorry

end fred_book_purchase_l2533_253323


namespace soda_difference_l2533_253365

/-- The number of liters of soda in each bottle -/
def liters_per_bottle : ℕ := 2

/-- The number of orange soda bottles Julio has -/
def julio_orange : ℕ := 4

/-- The number of grape soda bottles Julio has -/
def julio_grape : ℕ := 7

/-- The number of orange soda bottles Mateo has -/
def mateo_orange : ℕ := 1

/-- The number of grape soda bottles Mateo has -/
def mateo_grape : ℕ := 3

/-- The difference in total liters of soda between Julio and Mateo -/
theorem soda_difference : 
  (julio_orange + julio_grape) * liters_per_bottle - 
  (mateo_orange + mateo_grape) * liters_per_bottle = 14 := by
sorry

end soda_difference_l2533_253365


namespace dolls_in_big_box_l2533_253311

/-- Given information about big and small boxes containing dolls, 
    prove that each big box contains 7 dolls. -/
theorem dolls_in_big_box 
  (num_big_boxes : ℕ) 
  (num_small_boxes : ℕ) 
  (dolls_per_small_box : ℕ) 
  (total_dolls : ℕ) 
  (h1 : num_big_boxes = 5)
  (h2 : num_small_boxes = 9)
  (h3 : dolls_per_small_box = 4)
  (h4 : total_dolls = 71) :
  ∃ (dolls_per_big_box : ℕ), 
    dolls_per_big_box * num_big_boxes + 
    dolls_per_small_box * num_small_boxes = total_dolls ∧ 
    dolls_per_big_box = 7 :=
by sorry

end dolls_in_big_box_l2533_253311


namespace hyperbola_asymptotes_l2533_253361

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the line
def line (x y t : ℝ) : Prop := x - 3*y + t = 0

-- Define point M
def point_M (t : ℝ) : ℝ × ℝ := (t, 0)

-- Define the asymptotes
def asymptotes (k : ℝ) : Set (ℝ × ℝ) := {(x, y) | y = k*x ∨ y = -k*x}

-- Theorem statement
theorem hyperbola_asymptotes 
  (a b t : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (ht : t ≠ 0) :
  ∃ (A B : ℝ × ℝ),
    (∀ x y, hyperbola a b x y → line x y t → (x, y) ∈ asymptotes (1/2)) ∧
    (A ∈ asymptotes (1/2) ∧ B ∈ asymptotes (1/2)) ∧
    (line A.1 A.2 t ∧ line B.1 B.2 t) ∧
    (dist (point_M t) A = dist (point_M t) B) :=
sorry

end hyperbola_asymptotes_l2533_253361


namespace childrens_buffet_price_l2533_253305

def adult_price : ℚ := 30
def senior_discount : ℚ := 1/10
def num_adults : ℕ := 2
def num_seniors : ℕ := 2
def num_children : ℕ := 3
def total_spent : ℚ := 159

theorem childrens_buffet_price :
  ∃ (child_price : ℚ),
    child_price * num_children +
    adult_price * num_adults +
    adult_price * (1 - senior_discount) * num_seniors = total_spent ∧
    child_price = 15 := by
  sorry

end childrens_buffet_price_l2533_253305


namespace triangle_side_count_l2533_253350

theorem triangle_side_count : ∃! n : ℕ, n = (Finset.filter (fun x => x > 3 ∧ x < 11) (Finset.range 11)).card := by sorry

end triangle_side_count_l2533_253350


namespace simplify_fraction_l2533_253364

theorem simplify_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^(3/2) * b^(5/2)) / (a*b)^(1/2) = a * b^2 := by
  sorry

end simplify_fraction_l2533_253364


namespace bridge_length_is_two_km_l2533_253343

/-- The length of a bridge crossed by a man -/
def bridge_length (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: The length of a bridge is 2 km when crossed by a man walking at 8 km/hr in 15 minutes -/
theorem bridge_length_is_two_km :
  let speed := 8 -- km/hr
  let time := 15 / 60 -- 15 minutes converted to hours
  bridge_length speed time = 2 := by
  sorry

end bridge_length_is_two_km_l2533_253343


namespace correct_answer_after_resolving_errors_l2533_253397

theorem correct_answer_after_resolving_errors 
  (incorrect_divisor : ℝ)
  (correct_divisor : ℝ)
  (incorrect_answer : ℝ)
  (subtracted_value : ℝ)
  (should_add_value : ℝ)
  (h1 : incorrect_divisor = 63.5)
  (h2 : correct_divisor = 36.2)
  (h3 : incorrect_answer = 24)
  (h4 : subtracted_value = 12)
  (h5 : should_add_value = 8) :
  ∃ (correct_answer : ℝ), abs (correct_answer - 42.98) < 0.01 := by
sorry

end correct_answer_after_resolving_errors_l2533_253397


namespace logarithm_product_theorem_l2533_253381

theorem logarithm_product_theorem (c d : ℕ+) : 
  (d - c + 2 = 1000) →
  (Real.log (d + 1) / Real.log c = 3) →
  (c + d : ℕ) = 1009 := by
sorry

end logarithm_product_theorem_l2533_253381


namespace solution_equation1_solution_equation2_l2533_253320

-- Define the equations
def equation1 (x : ℝ) : Prop := 6 * x - 7 = 4 * x - 5
def equation2 (x : ℝ) : Prop := 4 / 3 - 8 * x = 3 - 11 / 2 * x

-- Theorem for the first equation
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1 := by sorry

-- Theorem for the second equation
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = -2/3 := by sorry

end solution_equation1_solution_equation2_l2533_253320
