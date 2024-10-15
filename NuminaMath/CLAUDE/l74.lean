import Mathlib

namespace NUMINAMATH_CALUDE_distance_between_points_l74_7420

/-- The Euclidean distance between two points (7, 0) and (-2, 12) is 15 -/
theorem distance_between_points : Real.sqrt ((7 - (-2))^2 + (0 - 12)^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l74_7420


namespace NUMINAMATH_CALUDE_tan_105_degrees_l74_7407

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l74_7407


namespace NUMINAMATH_CALUDE_quadratic_sum_l74_7473

/-- 
Given a quadratic function f(x) = -3x^2 + 18x + 108, 
there exist constants a, b, and c such that 
f(x) = a(x+b)^2 + c for all x, 
and a + b + c = 129
-/
theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (-3 * x^2 + 18 * x + 108 = a * (x + b)^2 + c) ∧ 
  (a + b + c = 129) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l74_7473


namespace NUMINAMATH_CALUDE_john_illustration_time_l74_7438

/-- Calculates the total time spent on John's illustration project -/
def total_illustration_time (
  num_landscapes : ℕ)
  (num_portraits : ℕ)
  (landscape_draw_time : ℝ)
  (landscape_color_time_ratio : ℝ)
  (portrait_draw_time : ℝ)
  (portrait_color_time_ratio : ℝ)
  (landscape_enhance_time : ℝ)
  (portrait_enhance_time : ℝ) : ℝ :=
  let landscape_time := 
    num_landscapes * (landscape_draw_time + landscape_color_time_ratio * landscape_draw_time + landscape_enhance_time)
  let portrait_time := 
    num_portraits * (portrait_draw_time + portrait_color_time_ratio * portrait_draw_time + portrait_enhance_time)
  landscape_time + portrait_time

/-- Theorem stating the total time John spends on his illustration project -/
theorem john_illustration_time : 
  total_illustration_time 10 15 2 0.7 3 0.75 0.75 1 = 135.25 := by
  sorry

end NUMINAMATH_CALUDE_john_illustration_time_l74_7438


namespace NUMINAMATH_CALUDE_ferris_wheel_seats_l74_7489

theorem ferris_wheel_seats (people_per_seat : ℕ) (total_people : ℕ) (h1 : people_per_seat = 9) (h2 : total_people = 18) :
  total_people / people_per_seat = 2 :=
by sorry

end NUMINAMATH_CALUDE_ferris_wheel_seats_l74_7489


namespace NUMINAMATH_CALUDE_calculate_expression_l74_7485

theorem calculate_expression : (Real.pi - Real.sqrt 3) ^ 0 - 2 * Real.sin (π / 4) + |-Real.sqrt 2| + Real.sqrt 8 = 1 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l74_7485


namespace NUMINAMATH_CALUDE_dodecahedron_triangle_count_l74_7499

/-- The number of vertices in a regular dodecahedron -/
def dodecahedron_vertices : ℕ := 12

/-- The number of distinct triangles that can be formed by connecting three
    different vertices of a regular dodecahedron -/
def dodecahedron_triangles : ℕ := Nat.choose dodecahedron_vertices 3

theorem dodecahedron_triangle_count :
  dodecahedron_triangles = 220 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_triangle_count_l74_7499


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l74_7411

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 32 → area = (perimeter / 4) ^ 2 → area = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l74_7411


namespace NUMINAMATH_CALUDE_inverse_sum_theorem_l74_7444

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the inverse of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- State the given condition
axiom condition : ∀ x, f (x + 1) + f (-x - 3) = 2

-- State the theorem to be proved
theorem inverse_sum_theorem : 
  ∀ x, f_inv (2009 - x) + f_inv (x - 2007) = -2 := by sorry

end NUMINAMATH_CALUDE_inverse_sum_theorem_l74_7444


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l74_7406

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) ↔ |a| ≤ 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l74_7406


namespace NUMINAMATH_CALUDE_walking_time_calculation_l74_7453

theorem walking_time_calculation (walking_speed run_speed : ℝ) (run_time : ℝ) (h1 : walking_speed = 5) (h2 : run_speed = 15) (h3 : run_time = 36 / 60) :
  let distance := run_speed * run_time
  walking_speed * (distance / walking_speed) = 1.8 := by sorry

end NUMINAMATH_CALUDE_walking_time_calculation_l74_7453


namespace NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l74_7403

-- Define a complex number
def complex (a b : ℝ) := a + b * Complex.I

-- Define what it means for a complex number to be purely imaginary
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem a_zero_necessary_not_sufficient :
  (∀ z : ℂ, is_purely_imaginary z → z.re = 0) ∧
  ¬(∀ z : ℂ, z.re = 0 → is_purely_imaginary z) :=
by sorry

end NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l74_7403


namespace NUMINAMATH_CALUDE_product_of_differences_l74_7488

theorem product_of_differences (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2005) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2004)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2005) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2004)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2005) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2004) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/1002 := by
  sorry

end NUMINAMATH_CALUDE_product_of_differences_l74_7488


namespace NUMINAMATH_CALUDE_florist_roses_problem_l74_7417

/-- Proves that the initial number of roses was 37 given the conditions of the problem. -/
theorem florist_roses_problem (initial_roses : ℕ) : 
  (initial_roses - 16 + 19 = 40) → initial_roses = 37 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_problem_l74_7417


namespace NUMINAMATH_CALUDE_fraction_of_fraction_one_eighth_of_one_third_l74_7424

theorem fraction_of_fraction (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem one_eighth_of_one_third :
  (1 / 8 : ℚ) / (1 / 3 : ℚ) = 3 / 8 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_one_eighth_of_one_third_l74_7424


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l74_7419

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y^2

def is_perfect_cube (x : ℕ) : Prop := ∃ y : ℕ, x = y^3

theorem smallest_n_square_and_cube :
  let n := 54
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(is_perfect_square (3*m) ∧ is_perfect_cube (4*m))) ∧
  is_perfect_square (3*n) ∧ is_perfect_cube (4*n) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l74_7419


namespace NUMINAMATH_CALUDE_palindromic_not_end_zero_two_digit_palindromic_count_three_digit_palindromic_count_four_digit_palindromic_count_ten_digit_palindromic_count_l74_7464

/-- A number is palindromic if it reads the same backward as forward. -/
def IsPalindromic (n : ℕ) : Prop := sorry

/-- The count of palindromic numbers with a given number of digits. -/
def PalindromicCount (digits : ℕ) : ℕ := sorry

/-- Palindromic numbers with more than two digits cannot end in 0. -/
theorem palindromic_not_end_zero (n : ℕ) (h : n > 99) (h_pal : IsPalindromic n) : n % 10 ≠ 0 := sorry

/-- There are 9 two-digit palindromic numbers. -/
theorem two_digit_palindromic_count : PalindromicCount 2 = 9 := sorry

/-- There are 90 three-digit palindromic numbers. -/
theorem three_digit_palindromic_count : PalindromicCount 3 = 90 := sorry

/-- There are 90 four-digit palindromic numbers. -/
theorem four_digit_palindromic_count : PalindromicCount 4 = 90 := sorry

/-- The main theorem: There are 90000 ten-digit palindromic numbers. -/
theorem ten_digit_palindromic_count : PalindromicCount 10 = 90000 := sorry

end NUMINAMATH_CALUDE_palindromic_not_end_zero_two_digit_palindromic_count_three_digit_palindromic_count_four_digit_palindromic_count_ten_digit_palindromic_count_l74_7464


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l74_7451

theorem billion_to_scientific_notation :
  ∀ (x : ℝ), x = 26.62 * 1000000000 → x = 2.662 * (10 ^ 9) := by
  sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l74_7451


namespace NUMINAMATH_CALUDE_remaining_distance_l74_7447

theorem remaining_distance (total_distance driven_distance : ℕ) : 
  total_distance = 1200 → driven_distance = 642 → total_distance - driven_distance = 558 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_l74_7447


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_implies_a_value_l74_7410

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a - 1)*x + 2

-- State the theorem
theorem monotonic_decreasing_interval_implies_a_value (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y ≤ 4 → f a x > f a y) →
  a = -3 :=
sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_implies_a_value_l74_7410


namespace NUMINAMATH_CALUDE_money_exchange_solution_l74_7414

/-- Represents the money exchange scenario between A, B, and C -/
def MoneyExchange (a b c : ℕ) : Prop :=
  let a₁ := a - 3*b - 3*c
  let b₁ := 4*b
  let c₁ := 4*c
  let a₂ := 4*a₁
  let b₂ := b₁ - 3*a₁ - 3*c₁
  let c₂ := 4*c₁
  let a₃ := 4*a₂
  let b₃ := 4*b₂
  let c₃ := c₂ - 3*a₂ - 3*b₂
  a₃ = 27 ∧ b₃ = 27 ∧ c₃ = 27 ∧ a + b + c = 81

theorem money_exchange_solution :
  ∃ (b c : ℕ), MoneyExchange 52 b c :=
sorry

end NUMINAMATH_CALUDE_money_exchange_solution_l74_7414


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l74_7400

-- Problem 1
theorem simplify_expression_1 : 
  (3 * Real.sqrt 8 - 12 * Real.sqrt (1/2) + Real.sqrt 18) * 2 * Real.sqrt 3 = 6 * Real.sqrt 6 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) (hx : x > 0) : 
  (6 * Real.sqrt (x/4) - 2*x * Real.sqrt (1/x)) / (3 * Real.sqrt x) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l74_7400


namespace NUMINAMATH_CALUDE_area_is_two_side_a_value_l74_7468

/-- Triangle ABC with given properties -/
structure TriangleABC where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Condition: cos A = 3/5
  cos_A : a^2 = b^2 + c^2 - 2*b*c*(3/5)
  -- Condition: AB · AC = 3
  dot_product : b*c*(3/5) = 3
  -- Condition: b - c = 3
  side_diff : b - c = 3

/-- The area of triangle ABC is 2 -/
theorem area_is_two (t : TriangleABC) : (1/2) * t.b * t.c * (4/5) = 2 := by sorry

/-- The value of side a is √13 -/
theorem side_a_value (t : TriangleABC) : t.a = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_area_is_two_side_a_value_l74_7468


namespace NUMINAMATH_CALUDE_average_net_income_is_399_50_l74_7429

/-- Represents the daily income and expense for a cab driver --/
structure DailyFinance where
  income : ℝ
  expense : ℝ

/-- Calculates the net income for a single day --/
def netIncome (df : DailyFinance) : ℝ := df.income - df.expense

/-- The cab driver's finances for 10 days --/
def tenDaysFinances : List DailyFinance := [
  ⟨600, 50⟩,
  ⟨250, 70⟩,
  ⟨450, 100⟩,
  ⟨400, 30⟩,
  ⟨800, 60⟩,
  ⟨450, 40⟩,
  ⟨350, 0⟩,
  ⟨600, 55⟩,
  ⟨270, 80⟩,
  ⟨500, 90⟩
]

/-- Theorem: The average daily net income for the cab driver over 10 days is $399.50 --/
theorem average_net_income_is_399_50 :
  (tenDaysFinances.map netIncome).sum / 10 = 399.50 := by
  sorry


end NUMINAMATH_CALUDE_average_net_income_is_399_50_l74_7429


namespace NUMINAMATH_CALUDE_cube_root_64_square_root_4_power_l74_7467

theorem cube_root_64_square_root_4_power (x y : ℝ) : 
  x^3 = 64 → y^2 = 4 → x^y = 16 := by
sorry

end NUMINAMATH_CALUDE_cube_root_64_square_root_4_power_l74_7467


namespace NUMINAMATH_CALUDE_quad_sum_is_six_l74_7441

/-- A quadrilateral with given properties --/
structure Quadrilateral where
  a : ℤ
  c : ℤ
  a_pos : 0 < a
  c_pos : 0 < c
  a_gt_c : c < a
  symmetric : True  -- Represents symmetry about origin
  equal_diagonals : True  -- Represents equal diagonal lengths
  area : (2 * (a - c).natAbs * (a + c).natAbs : ℤ) = 24

/-- The sum of a and c in a quadrilateral with given properties is 6 --/
theorem quad_sum_is_six (q : Quadrilateral) : q.a + q.c = 6 := by
  sorry

end NUMINAMATH_CALUDE_quad_sum_is_six_l74_7441


namespace NUMINAMATH_CALUDE_triangular_array_digit_sum_l74_7494

/-- The number of coins in a triangular array with n rows -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem triangular_array_digit_sum :
  ∃ (n : ℕ), triangular_sum n = 2145 ∧ sum_of_digits n = 11 := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_digit_sum_l74_7494


namespace NUMINAMATH_CALUDE_trey_decorations_l74_7415

theorem trey_decorations (total : ℕ) (nails thumbtacks sticky : ℕ) : 
  (nails = (2 * total) / 3) →
  (thumbtacks = (2 * (total - nails)) / 5) →
  (sticky = total - nails - thumbtacks) →
  (sticky = 15) →
  (nails = 50) := by
  sorry

end NUMINAMATH_CALUDE_trey_decorations_l74_7415


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l74_7463

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x + 1 ≥ 0 ∧ x - 2 < 0) ↔ (-1 ≤ x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l74_7463


namespace NUMINAMATH_CALUDE_fraction_problem_l74_7456

theorem fraction_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 16 → 
  (40/100 : ℝ) * N = 384 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l74_7456


namespace NUMINAMATH_CALUDE_adults_at_ball_game_l74_7460

theorem adults_at_ball_game :
  let num_children : ℕ := 11
  let adult_ticket_price : ℕ := 8
  let child_ticket_price : ℕ := 4
  let total_bill : ℕ := 124
  let num_adults : ℕ := (total_bill - num_children * child_ticket_price) / adult_ticket_price
  num_adults = 10 := by
sorry

end NUMINAMATH_CALUDE_adults_at_ball_game_l74_7460


namespace NUMINAMATH_CALUDE_exp_two_pi_third_in_second_quadrant_l74_7461

-- Define the complex exponential function
noncomputable def complex_exp (z : ℂ) : ℂ := Real.exp z.re * (Real.cos z.im + Complex.I * Real.sin z.im)

-- Define the second quadrant of the complex plane
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem exp_two_pi_third_in_second_quadrant :
  second_quadrant (complex_exp (Complex.I * (2 * Real.pi / 3))) :=
sorry

end NUMINAMATH_CALUDE_exp_two_pi_third_in_second_quadrant_l74_7461


namespace NUMINAMATH_CALUDE_mean_salary_calculation_l74_7471

def total_employees : ℕ := 100
def salary_group_1 : ℕ := 6000
def salary_group_2 : ℕ := 4000
def salary_group_3 : ℕ := 2500
def employees_group_1 : ℕ := 5
def employees_group_2 : ℕ := 15
def employees_group_3 : ℕ := 80

theorem mean_salary_calculation :
  (salary_group_1 * employees_group_1 + salary_group_2 * employees_group_2 + salary_group_3 * employees_group_3) / total_employees = 2900 := by
  sorry

end NUMINAMATH_CALUDE_mean_salary_calculation_l74_7471


namespace NUMINAMATH_CALUDE_intersection_condition_l74_7440

/-- Given functions f, g, f₁, g₁ and their coefficients, prove that if their graphs intersect
    at a single point with a negative x-coordinate and ac ≠ 0, then bc = ad. -/
theorem intersection_condition (a b c d : ℝ) (x₀ : ℝ) :
  let f := fun x : ℝ => a * x^2 + b * x
  let g := fun x : ℝ => c * x^2 + d * x
  let f₁ := fun x : ℝ => a * x + b
  let g₁ := fun x : ℝ => c * x + d
  (∀ x ≠ x₀, f x ≠ g x ∧ f x ≠ f₁ x ∧ f x ≠ g₁ x ∧
             g x ≠ f₁ x ∧ g x ≠ g₁ x ∧ f₁ x ≠ g₁ x) →
  (f x₀ = g x₀ ∧ f x₀ = f₁ x₀ ∧ f x₀ = g₁ x₀) →
  x₀ < 0 →
  a * c ≠ 0 →
  b * c = a * d :=
by sorry

end NUMINAMATH_CALUDE_intersection_condition_l74_7440


namespace NUMINAMATH_CALUDE_discounted_soda_price_l74_7472

/-- Calculate the price of discounted soda cans -/
theorem discounted_soda_price
  (regular_price : ℝ)
  (discount_percent : ℝ)
  (num_cans : ℕ)
  (h1 : regular_price = 0.60)
  (h2 : discount_percent = 20)
  (h3 : num_cans = 72) :
  let discounted_price := regular_price * (1 - discount_percent / 100)
  num_cans * discounted_price = 34.56 :=
by sorry

end NUMINAMATH_CALUDE_discounted_soda_price_l74_7472


namespace NUMINAMATH_CALUDE_man_speed_calculation_man_speed_proof_l74_7426

/-- Calculates the speed of a man given the parameters of a train passing him. -/
theorem man_speed_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / crossing_time
  train_speed_ms - relative_speed

/-- Given the specific parameters, proves that the man's speed is approximately 0.832 m/s. -/
theorem man_speed_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |man_speed_calculation 700 63 41.9966402687785 - 0.832| < ε :=
sorry

end NUMINAMATH_CALUDE_man_speed_calculation_man_speed_proof_l74_7426


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l74_7493

/-- An arithmetic sequence with second term -5 and common difference 3 has first term -8 -/
theorem arithmetic_sequence_first_term (a : ℕ → ℤ) :
  (∀ n, a (n + 1) = a n + 3) →  -- arithmetic sequence condition
  a 2 = -5 →                    -- given second term
  a 1 = -8 :=                   -- conclusion: first term
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l74_7493


namespace NUMINAMATH_CALUDE_total_rulers_l74_7423

/-- Given an initial number of rulers and a number of rulers added, 
    the total number of rulers is equal to their sum. -/
theorem total_rulers (initial_rulers added_rulers : ℕ) :
  initial_rulers + added_rulers = initial_rulers + added_rulers :=
by sorry

end NUMINAMATH_CALUDE_total_rulers_l74_7423


namespace NUMINAMATH_CALUDE_equation_equivalence_l74_7479

theorem equation_equivalence (x z : ℝ) 
  (h1 : 3 * x^2 + 4 * x + 6 * z + 2 = 0)
  (h2 : x - 2 * z + 1 = 0) :
  12 * z^2 + 2 * z + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l74_7479


namespace NUMINAMATH_CALUDE_triangle_inequality_from_condition_l74_7466

theorem triangle_inequality_from_condition 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : ∀ (A B C : ℝ), A > 0 → B > 0 → C > 0 → 
    A * a * (B * b + C * c) + B * b * (C * c + A * a) + C * c * (A * a + B * b) > 
    (1/2) * (A * B * c^2 + B * C * a^2 + C * A * b^2)) :
  a + b > c ∧ b + c > a ∧ c + a > b := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_from_condition_l74_7466


namespace NUMINAMATH_CALUDE_smallest_valid_graph_size_l74_7437

/-- A graph representing acquaintances among n people -/
def AcquaintanceGraph (n : ℕ) := Fin n → Fin n → Prop

/-- The property that any two acquainted people have no common acquaintances -/
def NoCommonAcquaintances (n : ℕ) (g : AcquaintanceGraph n) : Prop :=
  ∀ a b c : Fin n, g a b → g a c → g b c → a = b ∨ a = c ∨ b = c

/-- The property that any two non-acquainted people have exactly two common acquaintances -/
def TwoCommonAcquaintances (n : ℕ) (g : AcquaintanceGraph n) : Prop :=
  ∀ a b : Fin n, ¬g a b → ∃! (c d : Fin n), c ≠ d ∧ g a c ∧ g a d ∧ g b c ∧ g b d

/-- The main theorem stating that 11 is the smallest number satisfying the conditions -/
theorem smallest_valid_graph_size :
  (∃ (g : AcquaintanceGraph 11), NoCommonAcquaintances 11 g ∧ TwoCommonAcquaintances 11 g) ∧
  (∀ n : ℕ, 5 ≤ n → n < 11 →
    ¬∃ (g : AcquaintanceGraph n), NoCommonAcquaintances n g ∧ TwoCommonAcquaintances n g) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_graph_size_l74_7437


namespace NUMINAMATH_CALUDE_max_F_value_l74_7450

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  units : Nat
  is_four_digit : thousands ≥ 1 ∧ thousands ≤ 9 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Defines an eternal number -/
def is_eternal (m : FourDigitNumber) : Prop :=
  m.hundreds + m.tens + m.units = 12

/-- Swaps digits to create N -/
def swap_digits (m : FourDigitNumber) : FourDigitNumber :=
  { thousands := m.hundreds,
    hundreds := m.thousands,
    tens := m.units,
    units := m.tens,
    is_four_digit := by sorry }

/-- Defines the function F(M) -/
def F (m : FourDigitNumber) : Int :=
  let n := swap_digits m
  let m_value := 1000 * m.thousands + 100 * m.hundreds + 10 * m.tens + m.units
  let n_value := 1000 * n.thousands + 100 * n.hundreds + 10 * n.tens + n.units
  (m_value - n_value) / 9

/-- Main theorem -/
theorem max_F_value (m : FourDigitNumber) 
  (h_eternal : is_eternal m)
  (h_diff : m.hundreds - m.units = m.thousands)
  (h_div : (F m) % 9 = 0) :
  F m ≤ 9 ∧ ∃ (m' : FourDigitNumber), is_eternal m' ∧ m'.hundreds - m'.units = m'.thousands ∧ (F m') % 9 = 0 ∧ F m' = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_F_value_l74_7450


namespace NUMINAMATH_CALUDE_min_square_sum_l74_7427

theorem min_square_sum (y₁ y₂ y₃ : ℝ) 
  (h_pos : y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0) 
  (h_sum : 2*y₁ + 3*y₂ + 4*y₃ = 120) : 
  y₁^2 + y₂^2 + y₃^2 ≥ 14400/29 := by
  sorry

end NUMINAMATH_CALUDE_min_square_sum_l74_7427


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l74_7413

/-- Given a circle with center (4, 6) and one endpoint of a diameter at (1, 2),
    the other endpoint of the diameter is at (7, 10). -/
theorem circle_diameter_endpoint (P : Set (ℝ × ℝ)) : 
  (∃ (r : ℝ), P = {p | ∃ (x y : ℝ), p = (x, y) ∧ (x - 4)^2 + (y - 6)^2 = r^2}) →
  ((1, 2) ∈ P) →
  ((7, 10) ∈ P) ∧ 
  (∀ (x y : ℝ), (x, y) ∈ P → (x - 4)^2 + (y - 6)^2 = (7 - 4)^2 + (10 - 6)^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l74_7413


namespace NUMINAMATH_CALUDE_cost_price_calculation_cost_price_is_15000_l74_7455

theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : ℝ :=
  let discounted_price := selling_price * (1 - discount_rate)
  let cost_price := discounted_price / (1 + profit_rate)
  cost_price

theorem cost_price_is_15000 : 
  cost_price_calculation 18000 0.1 0.08 = 15000 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_cost_price_is_15000_l74_7455


namespace NUMINAMATH_CALUDE_relay_race_probability_l74_7448

-- Define the set of students
inductive Student : Type
  | A | B | C | D

-- Define the events
def event_A (s : Student) : Prop := s = Student.A
def event_B (s : Student) : Prop := s = Student.B

-- Define the conditional probability
def conditional_probability (A B : Student → Prop) : ℚ :=
  1 / 3

-- Theorem statement
theorem relay_race_probability :
  conditional_probability event_A event_B = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_probability_l74_7448


namespace NUMINAMATH_CALUDE_largest_m_satisfying_inequality_l74_7477

theorem largest_m_satisfying_inequality :
  ∃ m : ℕ, (((1 : ℚ) / 4 + m / 9 < 5 / 2) ∧
            ∀ n : ℕ, (n > m → (1 : ℚ) / 4 + n / 9 ≥ 5 / 2)) ∧
            m = 10 := by
  sorry

end NUMINAMATH_CALUDE_largest_m_satisfying_inequality_l74_7477


namespace NUMINAMATH_CALUDE_new_person_weight_l74_7404

theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 45 →
  avg_increase = 2.5 →
  ∃ (new_weight : ℝ), new_weight = replaced_weight + (initial_count * avg_increase) :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l74_7404


namespace NUMINAMATH_CALUDE_f_properties_l74_7431

def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

theorem f_properties (a : ℝ) :
  (∀ x, f a x = f a (-x) ↔ a = 0) ∧
  (∀ x, f a x ≥ 
    if a ≤ -1/2 then 3/4 - a
    else if a ≤ 1/2 then a^2 + 1
    else 3/4 + a) ∧
  (∃ x, f a x = 
    if a ≤ -1/2 then 3/4 - a
    else if a ≤ 1/2 then a^2 + 1
    else 3/4 + a) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l74_7431


namespace NUMINAMATH_CALUDE_box_2_neg1_3_neg2_l74_7418

/-- Definition of the box operation for integers a, b, c, d -/
def box (a b c d : ℤ) : ℚ := a^b - b^c + c^a + d^a

/-- Theorem stating that box(2,-1,3,-2) = 12.5 -/
theorem box_2_neg1_3_neg2 : box 2 (-1) 3 (-2) = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_box_2_neg1_3_neg2_l74_7418


namespace NUMINAMATH_CALUDE_triangle_angles_from_exterior_ratio_l74_7495

/-- Proves that a triangle with exterior angles in the ratio 12:13:15 has interior angles of 45°, 63°, and 72° -/
theorem triangle_angles_from_exterior_ratio :
  ∀ (E₁ E₂ E₃ : ℝ),
  E₁ > 0 ∧ E₂ > 0 ∧ E₃ > 0 →
  E₁ / 12 = E₂ / 13 ∧ E₂ / 13 = E₃ / 15 →
  E₁ + E₂ + E₃ = 360 →
  ∃ (I₁ I₂ I₃ : ℝ),
    I₁ = 180 - E₁ ∧
    I₂ = 180 - E₂ ∧
    I₃ = 180 - E₃ ∧
    I₁ + I₂ + I₃ = 180 ∧
    I₁ = 45 ∧ I₂ = 63 ∧ I₃ = 72 :=
by sorry


end NUMINAMATH_CALUDE_triangle_angles_from_exterior_ratio_l74_7495


namespace NUMINAMATH_CALUDE_hello_arrangements_l74_7470

theorem hello_arrangements : ℕ := by
  -- Define the word length
  let word_length : ℕ := 5

  -- Define the number of repeated letters
  let repeated_letters : ℕ := 1

  -- Define the number of repetitions of the repeated letter
  let repetitions : ℕ := 2

  -- Calculate total permutations
  let total_permutations : ℕ := Nat.factorial word_length

  -- Calculate unique permutations
  let unique_permutations : ℕ := total_permutations / Nat.factorial repeated_letters

  -- Calculate incorrect arrangements
  let incorrect_arrangements : ℕ := unique_permutations - 1

  -- Prove that the number of incorrect arrangements is 59
  sorry

end NUMINAMATH_CALUDE_hello_arrangements_l74_7470


namespace NUMINAMATH_CALUDE_smallest_N_for_Q_condition_l74_7434

def Q (N : ℕ) : ℚ := ((2 * N + 3) / 3 : ℚ) / (N + 1 : ℚ)

theorem smallest_N_for_Q_condition : 
  ∀ N : ℕ, 
    N > 0 → 
    N % 6 = 0 → 
    (∀ k : ℕ, k > 0 → k % 6 = 0 → k < N → Q k ≥ 7/10) → 
    Q N < 7/10 → 
    N = 12 := by sorry

end NUMINAMATH_CALUDE_smallest_N_for_Q_condition_l74_7434


namespace NUMINAMATH_CALUDE_line_translation_down_4_units_l74_7483

/-- Represents a line in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically -/
def translateLine (l : Line) (units : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - units }

theorem line_translation_down_4_units :
  let original_line : Line := { slope := -2, intercept := 3 }
  let translated_line := translateLine original_line 4
  translated_line = { slope := -2, intercept := -1 } := by sorry

end NUMINAMATH_CALUDE_line_translation_down_4_units_l74_7483


namespace NUMINAMATH_CALUDE_cone_base_radius_l74_7408

/-- Given a sector paper with radius 24 cm and area 120π cm², 
    prove that the radius of the circular base of the cone formed by this sector is 5 cm -/
theorem cone_base_radius (sector_radius : ℝ) (sector_area : ℝ) (base_radius : ℝ) : 
  sector_radius = 24 →
  sector_area = 120 * Real.pi →
  sector_area = Real.pi * base_radius * sector_radius →
  base_radius = 5 := by
sorry

end NUMINAMATH_CALUDE_cone_base_radius_l74_7408


namespace NUMINAMATH_CALUDE_distance_between_points_l74_7412

/-- The distance between two points (3, 3) and (9, 10) is √85 -/
theorem distance_between_points : Real.sqrt 85 = Real.sqrt ((9 - 3)^2 + (10 - 3)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l74_7412


namespace NUMINAMATH_CALUDE_polygon_sides_from_triangles_l74_7480

/-- Represents a polygon with n sides -/
structure Polygon (n : ℕ) where
  -- Add any necessary fields here

/-- Represents a point on a side of a polygon -/
structure PointOnSide (p : Polygon n) where
  -- Add any necessary fields here

/-- The number of triangles formed when connecting a point on a side to all vertices -/
def numTriangles (p : Polygon n) (point : PointOnSide p) : ℕ :=
  n - 1

theorem polygon_sides_from_triangles
  (p : Polygon n) (point : PointOnSide p)
  (h : numTriangles p point = 8) :
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_triangles_l74_7480


namespace NUMINAMATH_CALUDE_jogging_challenge_l74_7475

theorem jogging_challenge (monday_distance : Real) (daily_increase : Real) 
  (saturday_multiplier : Real) (weekly_goal : Real) :
  let tuesday_distance := monday_distance * (1 + daily_increase)
  let thursday_distance := tuesday_distance * (1 + daily_increase)
  let saturday_distance := thursday_distance * saturday_multiplier
  let sunday_distance := weekly_goal - (monday_distance + tuesday_distance + thursday_distance + saturday_distance)
  monday_distance = 3 ∧ 
  daily_increase = 0.1 ∧ 
  saturday_multiplier = 2.5 ∧ 
  weekly_goal = 40 →
  tuesday_distance = 3.3 ∧ 
  thursday_distance = 3.63 ∧ 
  saturday_distance = 9.075 ∧ 
  sunday_distance = 21.995 := by
  sorry

end NUMINAMATH_CALUDE_jogging_challenge_l74_7475


namespace NUMINAMATH_CALUDE_total_loaves_served_l74_7422

theorem total_loaves_served (wheat_bread : Real) (white_bread : Real) :
  wheat_bread = 0.2 → white_bread = 0.4 → wheat_bread + white_bread = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_total_loaves_served_l74_7422


namespace NUMINAMATH_CALUDE_perpendicular_line_to_plane_and_contained_line_l74_7409

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem perpendicular_line_to_plane_and_contained_line 
  (m n : Line) (α : Plane) 
  (h1 : perpendicular m α) 
  (h2 : contained_in n α) : 
  perpendicular_lines m n := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_plane_and_contained_line_l74_7409


namespace NUMINAMATH_CALUDE_girls_at_picnic_l74_7478

theorem girls_at_picnic (total_students : ℕ) (picnic_attendees : ℕ) 
  (h1 : total_students = 1200)
  (h2 : picnic_attendees = 730)
  (h3 : ∃ (girls boys : ℕ), girls + boys = total_students ∧ 
    2 * girls / 3 + boys / 2 = picnic_attendees) :
  ∃ (girls : ℕ), 2 * girls / 3 = 520 := by
sorry

end NUMINAMATH_CALUDE_girls_at_picnic_l74_7478


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l74_7484

theorem inequality_and_equality_condition (a b c : ℝ) : 
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) ≥ Real.sqrt (3*a^2 + (a + b + c)^2) ∧
  (Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) = Real.sqrt (3*a^2 + (a + b + c)^2) ↔ 
    (b = c ∨ a = 0) ∧ b*c ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l74_7484


namespace NUMINAMATH_CALUDE_no_integer_solution_for_divisibility_l74_7439

theorem no_integer_solution_for_divisibility : ¬∃ (x y : ℤ), (x^2 + y^2 + x + y) ∣ 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_divisibility_l74_7439


namespace NUMINAMATH_CALUDE_sphere_surface_area_l74_7469

theorem sphere_surface_area (cube_surface_area : ℝ) (sphere_radius : ℝ) : 
  cube_surface_area = 24 →
  (2 * sphere_radius) ^ 2 = 3 * (cube_surface_area / 6) →
  4 * Real.pi * sphere_radius ^ 2 = 12 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l74_7469


namespace NUMINAMATH_CALUDE_regular_poly15_distance_sum_l74_7442

/-- Regular 15-sided polygon -/
structure RegularPoly15 where
  vertices : Fin 15 → ℝ × ℝ
  is_regular : ∀ i j : Fin 15, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)

/-- Distance between two vertices -/
def dist_vertices (p : RegularPoly15) (i j : Fin 15) : ℝ :=
  dist (p.vertices i) (p.vertices j)

/-- Theorem statement -/
theorem regular_poly15_distance_sum (p : RegularPoly15) :
  1 / dist_vertices p 0 2 + 1 / dist_vertices p 0 4 + 1 / dist_vertices p 0 7 =
  1 / dist_vertices p 0 1 := by
  sorry

end NUMINAMATH_CALUDE_regular_poly15_distance_sum_l74_7442


namespace NUMINAMATH_CALUDE_binomial_coefficient_properties_l74_7452

theorem binomial_coefficient_properties (p : ℕ) (hp : Nat.Prime p) :
  (∀ k, p ∣ (Nat.choose (p - 1) k ^ 2 - 1)) ∧
  (∀ s, Even s → p ∣ (Finset.sum (Finset.range p) (λ k => Nat.choose (p - 1) k ^ s))) ∧
  (∀ s, Odd s → (Finset.sum (Finset.range p) (λ k => Nat.choose (p - 1) k ^ s)) % p = 1) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_properties_l74_7452


namespace NUMINAMATH_CALUDE_min_red_balls_l74_7490

/-- The total number of balls in the circle -/
def total_balls : ℕ := 58

/-- A type representing the color of a ball -/
inductive Color
| Red
| Blue

/-- A function that counts the number of consecutive triplets with a majority of a given color -/
def count_majority_triplets (balls : List Color) (color : Color) : ℕ := sorry

/-- A function that counts the total number of balls of a given color -/
def count_color (balls : List Color) (color : Color) : ℕ := sorry

/-- The main theorem stating the minimum number of red balls -/
theorem min_red_balls (balls : List Color) :
  balls.length = total_balls →
  count_majority_triplets balls Color.Red = count_majority_triplets balls Color.Blue →
  count_color balls Color.Red ≥ 20 := by sorry

end NUMINAMATH_CALUDE_min_red_balls_l74_7490


namespace NUMINAMATH_CALUDE_remaining_area_of_19x11_rectangle_l74_7462

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of the remaining rectangle after placing the largest possible squares inside a given rectangle -/
def remainingArea (rect : Rectangle) : ℝ :=
  sorry

/-- Theorem stating that the remaining area of a 19x11 rectangle after placing four squares is 6 -/
theorem remaining_area_of_19x11_rectangle : 
  remainingArea ⟨19, 11⟩ = 6 := by sorry

end NUMINAMATH_CALUDE_remaining_area_of_19x11_rectangle_l74_7462


namespace NUMINAMATH_CALUDE_power_of_two_representation_l74_7457

theorem power_of_two_representation (n : ℕ) (h : n ≥ 3) :
  ∃ (x y : ℕ), 2^n = 7*x^2 + y^2 ∧ Odd x ∧ Odd y :=
sorry

end NUMINAMATH_CALUDE_power_of_two_representation_l74_7457


namespace NUMINAMATH_CALUDE_find_number_l74_7432

theorem find_number : ∃! x : ℝ, 22 * (x - 36) = 748 ∧ x = 70 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l74_7432


namespace NUMINAMATH_CALUDE_donny_savings_l74_7425

theorem donny_savings (monday : ℕ) (wednesday : ℕ) (thursday_spent : ℕ) :
  monday = 15 →
  wednesday = 13 →
  thursday_spent = 28 →
  ∃ tuesday : ℕ, 
    tuesday = 28 ∧ 
    monday + tuesday + wednesday = 2 * thursday_spent :=
by sorry

end NUMINAMATH_CALUDE_donny_savings_l74_7425


namespace NUMINAMATH_CALUDE_inequality_solution_range_of_a_l74_7421

-- Define the functions f and g
def f (x : ℝ) := |x - 4|
def g (x : ℝ) := |2*x + 1|

-- Theorem for the first part of the problem
theorem inequality_solution :
  ∀ x : ℝ, f x < g x ↔ x < -5 ∨ x > 1 := by sorry

-- Theorem for the second part of the problem
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, 2 * f x + g x > a * x) ↔ -4 ≤ a ∧ a < 9/4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_of_a_l74_7421


namespace NUMINAMATH_CALUDE_second_half_speed_l74_7498

/-- Proves that given a journey of 224 km completed in 10 hours, where the first half is traveled at 21 km/hr, the speed for the second half of the journey is 24 km/hr. -/
theorem second_half_speed (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) :
  total_distance = 224 →
  total_time = 10 →
  first_half_speed = 21 →
  let first_half_distance := total_distance / 2
  let first_half_time := first_half_distance / first_half_speed
  let second_half_time := total_time - first_half_time
  let second_half_speed := first_half_distance / second_half_time
  second_half_speed = 24 := by
  sorry

end NUMINAMATH_CALUDE_second_half_speed_l74_7498


namespace NUMINAMATH_CALUDE_min_value_quadratic_l74_7435

theorem min_value_quadratic (x : ℝ) : x^2 + 10*x ≥ -25 ∧ ∃ y : ℝ, y^2 + 10*y = -25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l74_7435


namespace NUMINAMATH_CALUDE_greatest_divisor_three_consecutive_integers_l74_7459

theorem greatest_divisor_three_consecutive_integers :
  ∃ (d : ℕ), d > 0 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (n * (n + 1) * (n + 2))) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (m * (m + 1) * (m + 2)))) ∧
  d = 6 :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_three_consecutive_integers_l74_7459


namespace NUMINAMATH_CALUDE_consecutive_product_not_perfect_power_l74_7476

theorem consecutive_product_not_perfect_power (a : ℤ) :
  ¬ ∃ (n : ℕ) (k : ℤ), n > 1 ∧ a * (a^2 - 1) = k^n := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_not_perfect_power_l74_7476


namespace NUMINAMATH_CALUDE_natasha_dimes_l74_7445

theorem natasha_dimes (n : ℕ) 
  (h1 : 100 < n ∧ n < 200)
  (h2 : n % 6 = 2)
  (h3 : n % 7 = 2)
  (h4 : n % 8 = 2) : 
  n = 170 := by sorry

end NUMINAMATH_CALUDE_natasha_dimes_l74_7445


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l74_7487

/-- Given a cube with a sphere inscribed within it, and another cube inscribed within that sphere,
    this theorem relates the surface area of the outer cube to the surface area of the inner cube. -/
theorem inscribed_cube_surface_area
  (outer_cube_surface_area : ℝ)
  (h_outer_surface : outer_cube_surface_area = 54)
  : ∃ (inner_cube_surface_area : ℝ),
    inner_cube_surface_area = 18 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l74_7487


namespace NUMINAMATH_CALUDE_juice_transfer_difference_l74_7458

/-- Represents a barrel with a certain volume of juice -/
structure Barrel where
  volume : ℝ

/-- Represents the state of two barrels -/
structure TwoBarrels where
  barrel1 : Barrel
  barrel2 : Barrel

/-- Transfers a given volume from one barrel to another -/
def transfer (barrels : TwoBarrels) (amount : ℝ) : TwoBarrels :=
  { barrel1 := { volume := barrels.barrel1.volume + amount },
    barrel2 := { volume := barrels.barrel2.volume - amount } }

/-- Calculates the difference in volume between two barrels -/
def volumeDifference (barrels : TwoBarrels) : ℝ :=
  barrels.barrel1.volume - barrels.barrel2.volume

/-- Theorem stating that after transferring 3 L from the 8 L barrel to the 10 L barrel,
    the difference in volume between the two barrels is 8 L -/
theorem juice_transfer_difference :
  let initialBarrels : TwoBarrels := { barrel1 := { volume := 10 }, barrel2 := { volume := 8 } }
  let finalBarrels := transfer initialBarrels 3
  volumeDifference finalBarrels = 8 := by
  sorry


end NUMINAMATH_CALUDE_juice_transfer_difference_l74_7458


namespace NUMINAMATH_CALUDE_max_sides_touched_l74_7416

/-- Represents a regular hexagon -/
structure RegularHexagon where
  -- Add any necessary fields

/-- Represents a circle -/
structure Circle where
  -- Add any necessary fields

/-- Predicate to check if a circle is entirely contained within a hexagon -/
def is_contained (c : Circle) (h : RegularHexagon) : Prop :=
  sorry

/-- Predicate to check if a circle touches a side of a hexagon -/
def touches_side (c : Circle) (h : RegularHexagon) (side : Nat) : Prop :=
  sorry

/-- Predicate to check if a circle touches all sides of a hexagon -/
def touches_all_sides (c : Circle) (h : RegularHexagon) : Prop :=
  sorry

/-- The main theorem -/
theorem max_sides_touched (h : RegularHexagon) :
  ∃ (c : Circle), is_contained c h ∧ ¬touches_all_sides c h ∧
  (∃ (n : Nat), n = 2 ∧ 
    (∀ (m : Nat), (∃ (sides : Finset Nat), sides.card = m ∧ 
      (∀ (side : Nat), side ∈ sides → touches_side c h side)) → m ≤ n)) :=
sorry

end NUMINAMATH_CALUDE_max_sides_touched_l74_7416


namespace NUMINAMATH_CALUDE_time_difference_to_halfway_point_l74_7405

/-- Given that Danny can reach Steve's house in 29 minutes and Steve takes twice as long to reach Danny's house,
    prove that Steve takes 14.5 minutes longer than Danny to reach the halfway point between their houses. -/
theorem time_difference_to_halfway_point (danny_time : ℝ) (steve_time : ℝ) : 
  danny_time = 29 → steve_time = 2 * danny_time → steve_time / 2 - danny_time / 2 = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_to_halfway_point_l74_7405


namespace NUMINAMATH_CALUDE_smallest_upper_bound_sum_reciprocals_l74_7491

theorem smallest_upper_bound_sum_reciprocals :
  ∃ (r s : ℕ), r ≠ 0 ∧ s ≠ 0 ∧
  (∀ (k m n : ℕ), k ≠ 0 → m ≠ 0 → n ≠ 0 →
    (1 : ℚ) / k + (1 : ℚ) / m + (1 : ℚ) / n < 1 →
    (1 : ℚ) / k + (1 : ℚ) / m + (1 : ℚ) / n ≤ r / s) ∧
  (∀ (p q : ℕ), p ≠ 0 → q ≠ 0 →
    (∀ (k m n : ℕ), k ≠ 0 → m ≠ 0 → n ≠ 0 →
      (1 : ℚ) / k + (1 : ℚ) / m + (1 : ℚ) / n < 1 →
      (1 : ℚ) / k + (1 : ℚ) / m + (1 : ℚ) / n ≤ p / q) →
    r / s ≤ p / q) ∧
  r / s = 41 / 42 := by
sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_sum_reciprocals_l74_7491


namespace NUMINAMATH_CALUDE_student_count_l74_7497

/-- If a student is ranked 17th from the right and 5th from the left in a line of students,
    then the total number of students is 21. -/
theorem student_count (n : ℕ) (rank_right rank_left : ℕ) 
  (h1 : rank_right = 17)
  (h2 : rank_left = 5)
  (h3 : n = rank_right + rank_left - 1) :
  n = 21 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l74_7497


namespace NUMINAMATH_CALUDE_correct_transformation_l74_7436

theorem correct_transformation (a b m : ℝ) : a * (m^2 + 1) = b * (m^2 + 1) → a = b := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l74_7436


namespace NUMINAMATH_CALUDE_correct_algorithm_l74_7454

theorem correct_algorithm : 
  ((-8) / (-4) = 8 / 4) ∧ 
  ((-5) + 9 ≠ -(9 - 5)) ∧ 
  (7 - (-10) ≠ 7 - 10) ∧ 
  ((-5) * 0 ≠ -5) := by
  sorry

end NUMINAMATH_CALUDE_correct_algorithm_l74_7454


namespace NUMINAMATH_CALUDE_simplify_complex_root_expression_l74_7443

theorem simplify_complex_root_expression (x : ℝ) (h : x ≥ 0) :
  (4 * x * (11 + 4 * Real.sqrt 6)) ^ (1/6) *
  (4 * Real.sqrt (2 * x) - 2 * Real.sqrt (3 * x)) ^ (1/3) =
  (20 * x) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_root_expression_l74_7443


namespace NUMINAMATH_CALUDE_seashells_remaining_l74_7465

theorem seashells_remaining (initial : ℕ) (given_joan : ℕ) (given_ali : ℕ) (given_lee : ℕ) :
  initial = 200 →
  given_joan = 43 →
  given_ali = 27 →
  given_lee = 59 →
  initial - given_joan - given_ali - given_lee = 71 :=
by sorry

end NUMINAMATH_CALUDE_seashells_remaining_l74_7465


namespace NUMINAMATH_CALUDE_correct_league_members_l74_7474

/-- The number of members in the Valleyball Soccer League --/
def league_members : ℕ := 110

/-- The cost of a pair of socks in dollars --/
def sock_cost : ℕ := 6

/-- The additional cost of a T-shirt compared to a pair of socks in dollars --/
def tshirt_additional_cost : ℕ := 8

/-- The total expenditure of the league in dollars --/
def total_expenditure : ℕ := 3740

/-- Theorem stating that the number of members in the league is correct given the conditions --/
theorem correct_league_members :
  let tshirt_cost : ℕ := sock_cost + tshirt_additional_cost
  let member_cost : ℕ := sock_cost + 2 * tshirt_cost
  total_expenditure = league_members * member_cost :=
by sorry

end NUMINAMATH_CALUDE_correct_league_members_l74_7474


namespace NUMINAMATH_CALUDE_polar_coordinates_of_point_M_l74_7481

theorem polar_coordinates_of_point_M : 
  let x : ℝ := -1
  let y : ℝ := Real.sqrt 3
  let ρ : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arccos (x / ρ)
  (ρ = 2) ∧ (θ = 2 * Real.pi / 3) := by sorry

end NUMINAMATH_CALUDE_polar_coordinates_of_point_M_l74_7481


namespace NUMINAMATH_CALUDE_hash_computation_l74_7482

def hash (a b : ℤ) : ℤ := a * b - a - 3

theorem hash_computation : hash (hash 2 0) (hash 1 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_hash_computation_l74_7482


namespace NUMINAMATH_CALUDE_produce_worth_is_630_l74_7446

/-- The total worth of produce Gary stocked -/
def total_worth (asparagus_bundles asparagus_price grape_boxes grape_price apple_count apple_price : ℝ) : ℝ :=
  asparagus_bundles * asparagus_price + grape_boxes * grape_price + apple_count * apple_price

/-- Proof that the total worth of produce Gary stocked is $630 -/
theorem produce_worth_is_630 :
  total_worth 60 3 40 2.5 700 0.5 = 630 := by
  sorry

#eval total_worth 60 3 40 2.5 700 0.5

end NUMINAMATH_CALUDE_produce_worth_is_630_l74_7446


namespace NUMINAMATH_CALUDE_olympic_photo_arrangements_l74_7401

/-- Represents the number of athletes -/
def num_athletes : ℕ := 5

/-- Represents the number of athletes that can occupy the leftmost position -/
def num_leftmost_athletes : ℕ := 2

/-- Represents whether athlete A can occupy the rightmost position -/
def a_can_be_rightmost : Bool := false

/-- The total number of different arrangement possibilities -/
def total_arrangements : ℕ := 42

/-- Theorem stating that the total number of arrangements is 42 -/
theorem olympic_photo_arrangements :
  (num_athletes = 5) →
  (num_leftmost_athletes = 2) →
  (a_can_be_rightmost = false) →
  (total_arrangements = 42) := by
  sorry

end NUMINAMATH_CALUDE_olympic_photo_arrangements_l74_7401


namespace NUMINAMATH_CALUDE_division_remainder_problem_l74_7402

theorem division_remainder_problem (x y u v : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x = u * y + v) (h4 : v < y) : 
  (x + 3 * u * y + 4 * v) % y = 5 * v % y := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l74_7402


namespace NUMINAMATH_CALUDE_expand_expression_l74_7496

theorem expand_expression (x : ℝ) : (11 * x + 17) * (3 * x) + 5 = 33 * x^2 + 51 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l74_7496


namespace NUMINAMATH_CALUDE_problem_statement_l74_7492

theorem problem_statement (a b c d : ℕ+) 
  (h1 : a ^ 5 = b ^ 4)
  (h2 : c ^ 3 = d ^ 2)
  (h3 : c - a = 19) :
  d - b = 757 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l74_7492


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l74_7428

/-- An isosceles triangle with centroid on the inscribed circle -/
structure IsoscelesTriangleWithCentroidOnIncircle where
  -- The lengths of the sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The triangle is isosceles with sides a and b equal
  isosceles : a = b
  -- The perimeter of the triangle is 60
  perimeter : a + b + c = 60
  -- The centroid lies on the inscribed circle
  centroid_on_incircle : True  -- We represent this condition as always true for simplicity

/-- The theorem stating the side lengths of the triangle -/
theorem isosceles_triangle_side_lengths 
  (t : IsoscelesTriangleWithCentroidOnIncircle) : 
  t.a = 25 ∧ t.b = 25 ∧ t.c = 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l74_7428


namespace NUMINAMATH_CALUDE_part_one_part_two_l74_7449

/-- Given positive numbers a, b, c, d such that ad = bc and a + d > b + c, 
    then |a - d| > |b - c| -/
theorem part_one (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_ad_bc : a * d = b * c) (h_sum : a + d > b + c) :
  |a - d| > |b - c| := by sorry

/-- Given positive numbers a, b, c, d and a real number t such that 
    t * √(a² + b²) * √(c² + d²) = √(a⁴ + c⁴) + √(b⁴ + d⁴), then t ≥ √2 -/
theorem part_two (a b c d t : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_eq : t * Real.sqrt (a^2 + b^2) * Real.sqrt (c^2 + d^2) = 
          Real.sqrt (a^4 + c^4) + Real.sqrt (b^4 + d^4)) :
  t ≥ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l74_7449


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l74_7486

/-- The number of seconds in the experiment -/
def experiment_duration : ℕ := 240

/-- The number of seconds it takes for the bacteria population to double -/
def doubling_time : ℕ := 30

/-- The number of bacteria after the experiment duration -/
def final_population : ℕ := 524288

/-- The number of times the population doubles during the experiment -/
def doubling_count : ℕ := experiment_duration / doubling_time

theorem initial_bacteria_count :
  ∃ (initial_count : ℕ), initial_count * (2 ^ doubling_count) = final_population ∧ initial_count = 2048 :=
sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l74_7486


namespace NUMINAMATH_CALUDE_simplify_expression_l74_7430

theorem simplify_expression (a b : ℝ) : (1:ℝ)*(2*b)*(3*a)*(4*a^2)*(5*b^2)*(6*a^3) = 720*a^6*b^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l74_7430


namespace NUMINAMATH_CALUDE_total_earnings_theorem_l74_7433

/-- Represents the investment and return ratios for three investors -/
structure InvestmentData where
  a_invest : ℝ
  b_invest : ℝ
  c_invest : ℝ
  a_return : ℝ
  b_return : ℝ
  c_return : ℝ

/-- Calculates the total earnings given investment data -/
def totalEarnings (data : InvestmentData) : ℝ :=
  data.a_invest * data.a_return +
  data.b_invest * data.b_return +
  data.c_invest * data.c_return

/-- Theorem stating the total earnings under given conditions -/
theorem total_earnings_theorem (data : InvestmentData) :
  data.a_invest = 3 ∧
  data.b_invest = 4 ∧
  data.c_invest = 5 ∧
  data.a_return = 6 ∧
  data.b_return = 5 ∧
  data.c_return = 4 ∧
  data.b_invest * data.b_return = data.a_invest * data.a_return + 200 →
  totalEarnings data = 58000 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_theorem_l74_7433
