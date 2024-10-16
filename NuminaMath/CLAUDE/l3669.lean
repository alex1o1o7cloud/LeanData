import Mathlib

namespace NUMINAMATH_CALUDE_equation_roots_properties_l3669_366921

open Real

theorem equation_roots_properties (m : ℝ) (θ : ℝ) :
  θ ∈ Set.Ioo 0 π →
  (∀ x, 2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0 ↔ x = sin θ ∨ x = cos θ) →
  (m = Real.sqrt 3 / 2) ∧
  ((tan θ * sin θ) / (tan θ - 1) + cos θ / (1 - tan θ) = (Real.sqrt 3 + 1) / 2) ∧
  ((sin θ = Real.sqrt 3 / 2 ∧ cos θ = 1 / 2) ∨ (sin θ = 1 / 2 ∧ cos θ = Real.sqrt 3 / 2)) ∧
  (θ = π / 3 ∨ θ = π / 6) := by
  sorry

#check equation_roots_properties

end NUMINAMATH_CALUDE_equation_roots_properties_l3669_366921


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l3669_366990

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 6) ((Nat.factorial 9) / (Nat.factorial 4)) = 720 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l3669_366990


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_7_25_plus_13_25_l3669_366925

theorem sum_of_last_two_digits_of_7_25_plus_13_25 : 
  (7^25 + 13^25) % 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_7_25_plus_13_25_l3669_366925


namespace NUMINAMATH_CALUDE_polynomial_value_at_three_l3669_366911

theorem polynomial_value_at_three : 
  let x : ℤ := 3
  (x^5 : ℤ) - 5*x + 7*(x^3) = 417 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_three_l3669_366911


namespace NUMINAMATH_CALUDE_parabola_circle_tangency_l3669_366940

/-- Given a parabola y² = 8x and a circle x² + y² + 6x + m = 0, 
    if the directrix of the parabola is tangent to the circle, then m = 8 -/
theorem parabola_circle_tangency (m : ℝ) : 
  (∀ y : ℝ, (∃! x : ℝ, x = -2 ∧ x^2 + y^2 + 6*x + m = 0)) → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_circle_tangency_l3669_366940


namespace NUMINAMATH_CALUDE_middle_number_divisible_by_four_l3669_366912

theorem middle_number_divisible_by_four (x : ℕ) :
  (∃ y : ℕ, (x - 1)^3 + x^3 + (x + 1)^3 = y^3) →
  4 ∣ x :=
by sorry

end NUMINAMATH_CALUDE_middle_number_divisible_by_four_l3669_366912


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3669_366920

theorem quadratic_roots_relation (m p q : ℝ) (hm : m ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0) :
  (∃ r₁ r₂ : ℝ, (r₁ + r₂ = -q ∧ r₁ * r₂ = m) ∧
               (3*r₁ + 3*r₂ = -m ∧ 9*r₁*r₂ = p)) →
  p / q = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3669_366920


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l3669_366961

theorem complex_exponential_sum (θ φ : ℝ) :
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (-2/3 : ℂ) + (1/9 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (-2/3 : ℂ) - (1/9 : ℂ) * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l3669_366961


namespace NUMINAMATH_CALUDE_enrollment_system_correct_l3669_366944

/-- Represents the enrollment plan and actual enrollment of a school -/
structure EnrollmentPlan where
  planned_total : ℕ
  actual_total : ℕ
  boys_exceed_percent : ℚ
  girls_exceed_percent : ℚ

/-- The correct system of equations for the enrollment plan -/
def correct_system (plan : EnrollmentPlan) (x y : ℚ) : Prop :=
  x + y = plan.planned_total ∧
  (1 + plan.boys_exceed_percent) * x + (1 + plan.girls_exceed_percent) * y = plan.actual_total

/-- Theorem stating that the given system of equations is correct for the enrollment plan -/
theorem enrollment_system_correct (plan : EnrollmentPlan)
  (h1 : plan.planned_total = 1000)
  (h2 : plan.actual_total = 1240)
  (h3 : plan.boys_exceed_percent = 1/5)
  (h4 : plan.girls_exceed_percent = 3/10) :
  ∀ x y : ℚ, correct_system plan x y ↔ 
    (x + y = 1000 ∧ 6/5 * x + 13/10 * y = 1240) :=
by sorry

end NUMINAMATH_CALUDE_enrollment_system_correct_l3669_366944


namespace NUMINAMATH_CALUDE_bryans_skittles_count_l3669_366975

theorem bryans_skittles_count (ben_mm : ℕ) (bryan_extra : ℕ) 
  (h1 : ben_mm = 20) 
  (h2 : bryan_extra = 30) : 
  ben_mm + bryan_extra = 50 := by
  sorry

end NUMINAMATH_CALUDE_bryans_skittles_count_l3669_366975


namespace NUMINAMATH_CALUDE_abs_func_differentiable_l3669_366964

-- Define the absolute value function
def abs_func (x : ℝ) : ℝ := |x|

-- State the theorem
theorem abs_func_differentiable :
  ∀ x : ℝ, x ≠ 0 →
    (DifferentiableAt ℝ abs_func x) ∧
    (deriv abs_func x = if x > 0 then 1 else -1) :=
by sorry

end NUMINAMATH_CALUDE_abs_func_differentiable_l3669_366964


namespace NUMINAMATH_CALUDE_necklace_price_l3669_366916

def total_cost : ℕ := 240000
def necklace_count : ℕ := 3

theorem necklace_price (necklace_price : ℕ) 
  (h1 : necklace_count * necklace_price + 3 * necklace_price = total_cost) :
  necklace_price = 40000 := by
  sorry

end NUMINAMATH_CALUDE_necklace_price_l3669_366916


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3669_366970

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a where a₂ = 5 and a₅ = 33,
    prove that a₃ + a₄ = 38. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_a2 : a 2 = 5)
  (h_a5 : a 5 = 33) :
  a 3 + a 4 = 38 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3669_366970


namespace NUMINAMATH_CALUDE_problem_statement_l3669_366955

theorem problem_statement (x y : ℝ) (h1 : x * y = 4) (h2 : x - y = 5) :
  x^2 + 5*x*y + y^2 = 53 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3669_366955


namespace NUMINAMATH_CALUDE_reciprocal_equation_solution_l3669_366954

theorem reciprocal_equation_solution (x : ℝ) : 
  3 - 1 / (2 + x) = 2 * (1 / (2 + x)) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_solution_l3669_366954


namespace NUMINAMATH_CALUDE_f_composed_with_g_l3669_366933

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 1
def g (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem f_composed_with_g : f (1 + g 4) = 11 := by
  sorry

end NUMINAMATH_CALUDE_f_composed_with_g_l3669_366933


namespace NUMINAMATH_CALUDE_custom_operation_result_l3669_366996

/-- The custom operation ⊕ -/
def circle_plus (a b : ℤ) : ℤ := (a + b) * (a - b)

/-- The main theorem -/
theorem custom_operation_result :
  ((circle_plus 7 4) - 12) * 5 = 105 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_result_l3669_366996


namespace NUMINAMATH_CALUDE_house_selling_price_l3669_366936

def commission_rate : ℝ := 0.06
def commission_amount : ℝ := 8880

theorem house_selling_price :
  ∃ (selling_price : ℝ),
    selling_price * commission_rate = commission_amount ∧
    selling_price = 148000 := by
  sorry

end NUMINAMATH_CALUDE_house_selling_price_l3669_366936


namespace NUMINAMATH_CALUDE_volunteer_arrangements_l3669_366930

theorem volunteer_arrangements (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 7 ∧ k = 3 ∧ m = 4 →
  (n.choose k) * (m.choose k) = 140 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_arrangements_l3669_366930


namespace NUMINAMATH_CALUDE_age_difference_l3669_366980

/-- Proves that z is 1.2 decades younger than x given the condition on ages -/
theorem age_difference (x y z : ℝ) (h : x + y = y + z + 12) : (x - z) / 10 = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3669_366980


namespace NUMINAMATH_CALUDE_min_words_for_spanish_exam_l3669_366905

/-- Represents the Spanish vocabulary exam scenario -/
structure SpanishExam where
  total_words : ℕ
  min_score_percent : ℕ

/-- Calculates the minimum number of words needed to achieve the desired score -/
def min_words_needed (exam : SpanishExam) : ℕ :=
  (exam.min_score_percent * exam.total_words + 99) / 100

/-- Theorem stating the minimum number of words needed for the given exam conditions -/
theorem min_words_for_spanish_exam :
  let exam : SpanishExam := { total_words := 500, min_score_percent := 85 }
  min_words_needed exam = 425 := by
  sorry

#eval min_words_needed { total_words := 500, min_score_percent := 85 }

end NUMINAMATH_CALUDE_min_words_for_spanish_exam_l3669_366905


namespace NUMINAMATH_CALUDE_a_2000_mod_9_l3669_366966

-- Define the sequence
def a : ℕ → ℕ
  | 0 => 1995
  | n + 1 => (n + 1) * a n + 1

-- State the theorem
theorem a_2000_mod_9 : a 2000 % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_a_2000_mod_9_l3669_366966


namespace NUMINAMATH_CALUDE_remainder_13_pow_21_mod_1000_l3669_366948

theorem remainder_13_pow_21_mod_1000 : 13^21 % 1000 = 413 := by
  sorry

end NUMINAMATH_CALUDE_remainder_13_pow_21_mod_1000_l3669_366948


namespace NUMINAMATH_CALUDE_min_value_theorem_l3669_366919

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2 * y = 5) :
  ∃ (min_val : ℝ), min_val = 4 * Real.sqrt 3 ∧
  ∀ (z : ℝ), z = (x + 1) * (2 * y + 1) / Real.sqrt (x * y) → z ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3669_366919


namespace NUMINAMATH_CALUDE_proportion_solution_l3669_366918

theorem proportion_solution (x : ℝ) : (x / 5 = 1.05 / 7) → x = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l3669_366918


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l3669_366989

theorem trigonometric_expression_equals_one :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l3669_366989


namespace NUMINAMATH_CALUDE_school_student_count_l3669_366953

theorem school_student_count :
  ∀ (total_students : ℕ),
  (∃ (girls boys : ℕ),
    girls = boys ∧
    girls + boys = total_students ∧
    (girls : ℚ) * (1/5) + (boys : ℚ) * (1/10) = 15) →
  total_students = 100 := by
sorry

end NUMINAMATH_CALUDE_school_student_count_l3669_366953


namespace NUMINAMATH_CALUDE_pasture_problem_l3669_366992

/-- The number of horses b put in the pasture -/
def b_horses : ℕ := 16

/-- The total cost of the pasture in Rs -/
def total_cost : ℕ := 435

/-- The amount b should pay in Rs -/
def b_payment : ℕ := 180

/-- The number of horses a put in -/
def a_horses : ℕ := 12

/-- The number of months a's horses stayed -/
def a_months : ℕ := 8

/-- The number of months b's horses stayed -/
def b_months : ℕ := 9

/-- The number of horses c put in -/
def c_horses : ℕ := 18

/-- The number of months c's horses stayed -/
def c_months : ℕ := 6

theorem pasture_problem :
  b_horses = 16 ∧
  (b_horses * b_months : ℚ) / (a_horses * a_months + b_horses * b_months + c_horses * c_months : ℚ) =
  b_payment / total_cost := by
  sorry

end NUMINAMATH_CALUDE_pasture_problem_l3669_366992


namespace NUMINAMATH_CALUDE_new_person_weight_l3669_366923

theorem new_person_weight (n : ℕ) (initial_weight replaced_weight avg_increase : ℝ) :
  n = 8 ∧ 
  replaced_weight = 65 ∧ 
  avg_increase = 2.5 →
  replaced_weight + n * avg_increase = 85 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l3669_366923


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3669_366907

/-- Given vectors a and b in ℝ², if k*a + b is parallel to a + 3*b, then k = 1/3 -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h_parallel : ∃ (t : ℝ), t ≠ 0 ∧ k • a + b = t • (a + 3 • b)) :
  k = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3669_366907


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l3669_366901

theorem consecutive_even_numbers_sum (a b c : ℤ) : 
  (∃ n : ℤ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4) →  -- a, b, c are consecutive even numbers
  a + b + c = 246 →                                -- their sum is 246
  b = 82                                           -- the second number is 82
:= by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l3669_366901


namespace NUMINAMATH_CALUDE_seating_theorem_l3669_366913

/-- The number of ways to arrange 3 people in a row of 6 seats with exactly two adjacent empty seats -/
def seating_arrangements (total_seats : Nat) (people : Nat) (adjacent_empty : Nat) : Nat :=
  24 * 3

theorem seating_theorem :
  seating_arrangements 6 3 2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l3669_366913


namespace NUMINAMATH_CALUDE_board_cut_ratio_l3669_366927

/-- Given a board of length 69 inches cut into two pieces,
    where one piece is a multiple of the other and the longer piece is 46 inches,
    prove that the ratio of the longer piece to the shorter piece is 2:1 -/
theorem board_cut_ratio (total_length shorter_length longer_length : ℝ)
  (h1 : total_length = 69)
  (h2 : total_length = shorter_length + longer_length)
  (h3 : ∃ (m : ℝ), longer_length = m * shorter_length)
  (h4 : longer_length = 46) :
  longer_length / shorter_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_board_cut_ratio_l3669_366927


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3669_366977

theorem triangle_perimeter : ∀ (a b c : ℝ),
  a = 4 ∧ b = 8 ∧ c^2 - 14*c + 40 = 0 ∧
  a + b > c ∧ b + c > a ∧ a + c > b →
  a + b + c = 22 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3669_366977


namespace NUMINAMATH_CALUDE_sugar_consumption_reduction_l3669_366934

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (h_initial : initial_price = 10)
  (h_new : new_price = 13) :
  let reduction_percentage := (new_price - initial_price) / initial_price * 100
  reduction_percentage = 30 := by
  sorry

end NUMINAMATH_CALUDE_sugar_consumption_reduction_l3669_366934


namespace NUMINAMATH_CALUDE_cloth_length_proof_l3669_366956

/-- The length of a piece of cloth satisfying given cost conditions -/
def cloth_length : ℝ := 10

/-- The cost of the cloth -/
def total_cost : ℝ := 35

/-- The additional length in the hypothetical scenario -/
def additional_length : ℝ := 4

/-- The price reduction per meter in the hypothetical scenario -/
def price_reduction : ℝ := 1

theorem cloth_length_proof :
  cloth_length > 0 ∧
  total_cost = cloth_length * (total_cost / cloth_length) ∧
  total_cost = (cloth_length + additional_length) * (total_cost / cloth_length - price_reduction) :=
by sorry

end NUMINAMATH_CALUDE_cloth_length_proof_l3669_366956


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3669_366981

theorem triangle_angle_measure (A B C : Real) (a b c : Real) :
  -- Given conditions
  (4 * (Real.cos A)^2 + 4 * Real.cos B * Real.cos C + 1 = 4 * Real.sin B * Real.sin C) →
  (A < B) →
  (a = 2 * Real.sqrt 3) →
  (a / Real.sin A = 4) →  -- Circumradius condition
  -- Conclusion
  A = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3669_366981


namespace NUMINAMATH_CALUDE_add_self_eq_two_mul_l3669_366902

theorem add_self_eq_two_mul (a : ℝ) : a + a = 2 * a := by sorry

end NUMINAMATH_CALUDE_add_self_eq_two_mul_l3669_366902


namespace NUMINAMATH_CALUDE_average_speed_ratio_l3669_366929

/-- Represents the average speed ratio problem -/
theorem average_speed_ratio 
  (distance_eddy : ℝ) 
  (distance_freddy : ℝ) 
  (time_eddy : ℝ) 
  (time_freddy : ℝ) 
  (h1 : distance_eddy = 600) 
  (h2 : distance_freddy = 460) 
  (h3 : time_eddy = 3) 
  (h4 : time_freddy = 4) : 
  (distance_eddy / time_eddy) / (distance_freddy / time_freddy) = 200 / 115 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_ratio_l3669_366929


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3669_366969

theorem adult_ticket_cost 
  (total_spent : ℕ) 
  (family_size : ℕ) 
  (child_ticket_cost : ℕ) 
  (adult_tickets : ℕ) 
  (h1 : total_spent = 119)
  (h2 : family_size = 7)
  (h3 : child_ticket_cost = 14)
  (h4 : adult_tickets = 4) :
  ∃ (adult_ticket_cost : ℕ), 
    adult_ticket_cost * adult_tickets + 
    child_ticket_cost * (family_size - adult_tickets) = total_spent ∧ 
    adult_ticket_cost = 14 :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l3669_366969


namespace NUMINAMATH_CALUDE_insertion_methods_eq_336_l3669_366971

/- Given 5 books originally and 3 books to insert -/
def original_books : ℕ := 5
def books_to_insert : ℕ := 3

/- The number of gaps increases after each insertion -/
def gaps (n : ℕ) : ℕ := n + 1

/- The total number of insertion methods -/
def insertion_methods : ℕ :=
  (gaps original_books) * (gaps (original_books + 1)) * (gaps (original_books + 2))

/- Theorem stating that the number of insertion methods is 336 -/
theorem insertion_methods_eq_336 : insertion_methods = 336 := by
  sorry

end NUMINAMATH_CALUDE_insertion_methods_eq_336_l3669_366971


namespace NUMINAMATH_CALUDE_group_average_score_l3669_366968

def class_size : ℕ := 14
def class_average : ℝ := 85
def score_differences : List ℝ := [2, 3, -3, -5, 12, 12, 8, 2, -1, 4, -10, -2, 5, 5]

theorem group_average_score :
  let total_score := class_size * class_average + score_differences.sum
  total_score / class_size = 87.29 := by sorry

end NUMINAMATH_CALUDE_group_average_score_l3669_366968


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3669_366900

theorem quadratic_no_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - a ≠ 0) → a < -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3669_366900


namespace NUMINAMATH_CALUDE_digging_project_breadth_l3669_366978

/-- The breadth of the first digging project -/
def breadth_project1 : ℝ := 30

/-- The depth of the first digging project in meters -/
def depth_project1 : ℝ := 100

/-- The length of the first digging project in meters -/
def length_project1 : ℝ := 25

/-- The depth of the second digging project in meters -/
def depth_project2 : ℝ := 75

/-- The length of the second digging project in meters -/
def length_project2 : ℝ := 20

/-- The breadth of the second digging project in meters -/
def breadth_project2 : ℝ := 50

/-- The number of days to complete each project -/
def days_to_complete : ℝ := 12

theorem digging_project_breadth :
  depth_project1 * length_project1 * breadth_project1 =
  depth_project2 * length_project2 * breadth_project2 :=
by sorry

end NUMINAMATH_CALUDE_digging_project_breadth_l3669_366978


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l3669_366935

/-- Given a function f: ℝ → ℝ satisfying the specified condition,
    proves that the tangent line to y = f(x) at (1, f(1)) has the equation x - y - 2 = 0 -/
theorem tangent_line_at_one (f : ℝ → ℝ) 
    (h : ∀ x, f (1 + x) = 2 * f (1 - x) - x^2 + 3*x + 1) : 
    ∃ m b, (∀ x, m * (x - 1) + f 1 = m * x + b) ∧ m = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l3669_366935


namespace NUMINAMATH_CALUDE_color_combination_count_l3669_366946

/-- The number of colors available -/
def total_colors : ℕ := 9

/-- The number of colors to be chosen -/
def colors_to_choose : ℕ := 2

/-- The number of forbidden combinations (red and pink) -/
def forbidden_combinations : ℕ := 1

/-- The number of ways to choose colors, excluding forbidden combinations -/
def valid_combinations : ℕ := (total_colors.choose colors_to_choose) - forbidden_combinations

theorem color_combination_count : valid_combinations = 35 := by
  sorry

end NUMINAMATH_CALUDE_color_combination_count_l3669_366946


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2010th_term_l3669_366987

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_2010th_term 
  (p q : ℝ) 
  (h1 : 9 = arithmetic_sequence p (2 * q) 2)
  (h2 : 3 * p - q = arithmetic_sequence p (2 * q) 3)
  (h3 : 3 * p + q = arithmetic_sequence p (2 * q) 4) :
  arithmetic_sequence p (2 * q) 2010 = 8041 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2010th_term_l3669_366987


namespace NUMINAMATH_CALUDE_aiyannas_cookie_count_l3669_366958

/-- The number of cookies Alyssa has -/
def alyssas_cookies : ℕ := 129

/-- The number of additional cookies Aiyanna has compared to Alyssa -/
def additional_cookies : ℕ := 11

/-- The number of cookies Aiyanna has -/
def aiyannas_cookies : ℕ := alyssas_cookies + additional_cookies

theorem aiyannas_cookie_count : aiyannas_cookies = 140 := by
  sorry

end NUMINAMATH_CALUDE_aiyannas_cookie_count_l3669_366958


namespace NUMINAMATH_CALUDE_disjunction_true_l3669_366952

theorem disjunction_true : 
  (∃ α : ℝ, Real.cos (π - α) = Real.cos α) ∨ 
  (∀ x : ℝ, x^2 + 1 > 0) := by
sorry

end NUMINAMATH_CALUDE_disjunction_true_l3669_366952


namespace NUMINAMATH_CALUDE_total_animals_legoland_animals_l3669_366988

/-- Given a ratio of kangaroos to koalas and the total number of kangaroos,
    calculate the total number of animals (koalas and kangaroos). -/
theorem total_animals (ratio : ℕ) (num_kangaroos : ℕ) : ℕ :=
  let num_koalas := num_kangaroos / ratio
  num_koalas + num_kangaroos

/-- Prove that given 5 kangaroos for each koala and 180 kangaroos in total,
    the total number of koalas and kangaroos is 216. -/
theorem legoland_animals : total_animals 5 180 = 216 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_legoland_animals_l3669_366988


namespace NUMINAMATH_CALUDE_quadratic_equation_1_l3669_366945

theorem quadratic_equation_1 : 
  ∀ x : ℝ, x^2 - 2*x + 1 = 0 → x = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_1_l3669_366945


namespace NUMINAMATH_CALUDE_parabola_focus_l3669_366993

/-- The focus of a parabola y^2 = -4x is at (-1, 0) -/
theorem parabola_focus (x y : ℝ) :
  y^2 = -4*x → (x + 1)^2 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l3669_366993


namespace NUMINAMATH_CALUDE_triangle_side_range_l3669_366995

theorem triangle_side_range (m : ℝ) : 
  (3 : ℝ) > 0 ∧ (1 - 2*m : ℝ) > 0 ∧ (8 : ℝ) > 0 ∧
  (3 + (1 - 2*m) > 8) ∧ (3 + 8 > (1 - 2*m)) ∧ ((1 - 2*m) + 8 > 3) →
  -5 < m ∧ m < -2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_range_l3669_366995


namespace NUMINAMATH_CALUDE_max_delta_ratio_l3669_366986

/-- Represents a contestant's score in a two-day competition -/
structure ContestantScore where
  day1_score : ℕ
  day1_total : ℕ
  day2_score : ℕ
  day2_total : ℕ

/-- Calculate the two-day success ratio -/
def two_day_ratio (score : ContestantScore) : ℚ :=
  (score.day1_score + score.day2_score : ℚ) / (score.day1_total + score.day2_total)

/-- Charlie's score in the competition -/
def charlie : ContestantScore :=
  { day1_score := 210, day1_total := 400, day2_score := 150, day2_total := 200 }

theorem max_delta_ratio :
  ∀ delta : ContestantScore,
    delta.day1_score > 0 ∧ 
    delta.day2_score > 0 ∧
    delta.day1_total + delta.day2_total = 600 ∧
    (delta.day1_score : ℚ) / delta.day1_total < 210 / 400 ∧
    (delta.day2_score : ℚ) / delta.day2_total < 3 / 4 →
    two_day_ratio delta ≤ 349 / 600 :=
  sorry

end NUMINAMATH_CALUDE_max_delta_ratio_l3669_366986


namespace NUMINAMATH_CALUDE_product_divisible_by_all_product_prime_factorization_divisibility_condition_l3669_366960

theorem product_divisible_by_all : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 10 → (45 * 56) % n = 0 := by sorry

theorem product_prime_factorization : 
  ∃ (k : ℕ), 45 * 56 = 2^3 * 3^2 * 5 * 7 * k ∧ k ≥ 1 := by sorry

theorem divisibility_condition (a b c d : ℕ) : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 10 → (2^a * 3^b * 5^c * 7^d) % n = 0 → a ≥ 3 ∧ b ≥ 2 ∧ c ≥ 1 ∧ d ≥ 1 := by sorry

end NUMINAMATH_CALUDE_product_divisible_by_all_product_prime_factorization_divisibility_condition_l3669_366960


namespace NUMINAMATH_CALUDE_least_possible_FG_l3669_366962

-- Define the triangle EFG
structure TriangleEFG where
  EF : ℝ
  EG : ℝ
  FG : ℝ

-- Define the triangle HFG
structure TriangleHFG where
  HF : ℝ
  HG : ℝ
  FG : ℝ

-- Define the shared triangle configuration
def SharedTriangles (t1 : TriangleEFG) (t2 : TriangleHFG) : Prop :=
  t1.FG = t2.FG ∧
  t1.EF = 7 ∧
  t1.EG = 15 ∧
  t2.HG = 10 ∧
  t2.HF = 25

-- Theorem statement
theorem least_possible_FG (t1 : TriangleEFG) (t2 : TriangleHFG) 
  (h : SharedTriangles t1 t2) : 
  ∃ (n : ℕ), n = 15 ∧ t1.FG = n ∧ 
  (∀ (m : ℕ), m < n → ¬(∃ (t1' : TriangleEFG) (t2' : TriangleHFG), 
    SharedTriangles t1' t2' ∧ t1'.FG = m)) :=
  sorry

end NUMINAMATH_CALUDE_least_possible_FG_l3669_366962


namespace NUMINAMATH_CALUDE_tan_c_in_triangle_l3669_366982

theorem tan_c_in_triangle (A B C : Real) : 
  -- Triangle condition
  A + B + C = π → 
  -- tan A and tan B are roots of 3x^2 - 7x + 2 = 0
  (∃ (x y : Real), x ≠ y ∧ 
    3 * x^2 - 7 * x + 2 = 0 ∧ 
    3 * y^2 - 7 * y + 2 = 0 ∧ 
    x = Real.tan A ∧ 
    y = Real.tan B) → 
  -- Conclusion
  Real.tan C = -7 :=
by sorry

end NUMINAMATH_CALUDE_tan_c_in_triangle_l3669_366982


namespace NUMINAMATH_CALUDE_min_value_of_f_over_x_range_of_a_for_inequality_l3669_366947

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - 2*a*x - 1 + a

-- Theorem for part I
theorem min_value_of_f_over_x (x : ℝ) (h : x > 0) :
  ∃ (min_val : ℝ), min_val = -2 ∧ ∀ (y : ℝ), y > 0 → f 2 y / y ≥ min_val :=
sorry

-- Theorem for part II
theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f a x ≤ a) ↔ a ≥ -2 + Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_over_x_range_of_a_for_inequality_l3669_366947


namespace NUMINAMATH_CALUDE_expression_simplification_l3669_366904

theorem expression_simplification (x : ℝ) : 
  (((x+1)^3*(x^2-x+1)^3)/(x^3+1)^3)^3 * (((x-1)^3*(x^2+x+1)^3)/(x^3-1)^3)^3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3669_366904


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3669_366999

theorem perpendicular_lines_a_value (a : ℝ) :
  let l1 : ℝ → ℝ → Prop := λ x y => a * x - y - 2 = 0
  let l2 : ℝ → ℝ → Prop := λ x y => (a + 2) * x - y + 1 = 0
  (∀ x1 y1 x2 y2, l1 x1 y1 → l2 x2 y2 → (x2 - x1) * (y2 - y1) = 0) →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3669_366999


namespace NUMINAMATH_CALUDE_money_distribution_l3669_366997

theorem money_distribution (A B C : ℤ) 
  (total : A + B + C = 300)
  (AC_sum : A + C = 200)
  (BC_sum : B + C = 350) :
  C = 250 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l3669_366997


namespace NUMINAMATH_CALUDE_square_difference_formula_l3669_366922

-- Define the expressions
def expr_A (x : ℝ) := (x + 1) * (x - 1)
def expr_B (x : ℝ) := (-x + 1) * (-x - 1)
def expr_C (x : ℝ) := (x + 1) * (-x + 1)
def expr_D (x : ℝ) := (x + 1) * (1 + x)

-- Define a predicate for expressions that can be written as a difference of squares
def is_diff_of_squares (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ → ℝ), ∀ x, f x = (a x)^2 - (b x)^2

-- State the theorem
theorem square_difference_formula :
  (is_diff_of_squares expr_A) ∧
  (is_diff_of_squares expr_B) ∧
  (is_diff_of_squares expr_C) ∧
  ¬(is_diff_of_squares expr_D) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l3669_366922


namespace NUMINAMATH_CALUDE_initial_nickels_l3669_366991

/-- Given the current number of nickels and the number of borrowed nickels,
    prove that the initial number of nickels is their sum. -/
theorem initial_nickels (current_nickels borrowed_nickels : ℕ) :
  let initial_nickels := current_nickels + borrowed_nickels
  initial_nickels = current_nickels + borrowed_nickels :=
by sorry

end NUMINAMATH_CALUDE_initial_nickels_l3669_366991


namespace NUMINAMATH_CALUDE_inequality_proof_l3669_366979

theorem inequality_proof (x y z : ℝ) : x^2 + y^2 + z^2 ≥ Real.sqrt 2 * (x*y + y*z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3669_366979


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l3669_366909

theorem triangle_angle_inequality (f : ℝ → ℝ) (α β : ℝ) : 
  (∀ x y, x ∈ [-1, 1] → y ∈ [-1, 1] → x < y → f x > f y) →  -- f is decreasing on [-1,1]
  0 < α →                                                   -- α is positive
  0 < β →                                                   -- β is positive
  α < π / 2 →                                               -- α is less than π/2
  β < π / 2 →                                               -- β is less than π/2
  α + β > π / 2 →                                           -- sum of α and β is greater than π/2
  α ≠ β →                                                   -- α and β are distinct
  f (Real.cos α) > f (Real.sin β) :=                        -- prove this inequality
by sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l3669_366909


namespace NUMINAMATH_CALUDE_candidates_per_state_l3669_366965

theorem candidates_per_state : 
  ∀ x : ℝ,
  (x * 0.07 = x * 0.06 + 79) →
  x = 7900 := by
sorry

end NUMINAMATH_CALUDE_candidates_per_state_l3669_366965


namespace NUMINAMATH_CALUDE_mascs_age_l3669_366984

/-- Given that Masc is 7 years older than Sam and the sum of their ages is 27,
    prove that Masc's age is 17 years old. -/
theorem mascs_age (sam : ℕ) (masc : ℕ) 
    (h1 : masc = sam + 7)
    (h2 : sam + masc = 27) : 
  masc = 17 := by
  sorry

end NUMINAMATH_CALUDE_mascs_age_l3669_366984


namespace NUMINAMATH_CALUDE_concrete_blocks_theorem_l3669_366906

/-- Calculates the number of concrete blocks per section in a hedge. -/
def concrete_blocks_per_section (total_sections : ℕ) (total_cost : ℕ) (cost_per_piece : ℕ) : ℕ :=
  (total_cost / cost_per_piece) / total_sections

/-- Proves that the number of concrete blocks per section is 30 given the specified conditions. -/
theorem concrete_blocks_theorem :
  concrete_blocks_per_section 8 480 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_concrete_blocks_theorem_l3669_366906


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_sum_l3669_366974

/-- The quadratic function f(x) = 4x^2 - 8x + 6 -/
def f (x : ℝ) : ℝ := 4 * x^2 - 8 * x + 6

/-- The vertex form of the quadratic function -/
def vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

theorem quadratic_vertex_form_sum :
  ∃ (a h k : ℝ), (∀ x, f x = vertex_form a h k x) ∧ (a + h + k = 7) := by
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_sum_l3669_366974


namespace NUMINAMATH_CALUDE_max_quad_area_l3669_366917

/-- The ellipse defined by x²/8 + y²/4 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

/-- The foci of the ellipse -/
def foci : ℝ × ℝ × ℝ × ℝ := sorry

/-- A point on the ellipse -/
def point_on_ellipse (p : ℝ × ℝ) : Prop := ellipse p.1 p.2

/-- Segment AB passes through the center of the ellipse -/
def segment_through_center (a b : ℝ × ℝ) : Prop := sorry

/-- The area of quadrilateral F₁AF₂B -/
def quad_area (a b : ℝ × ℝ) : ℝ := sorry

theorem max_quad_area :
  ∀ (a b : ℝ × ℝ),
    point_on_ellipse a →
    point_on_ellipse b →
    segment_through_center a b →
    quad_area a b ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_quad_area_l3669_366917


namespace NUMINAMATH_CALUDE_fresh_grapes_weight_l3669_366926

/-- The weight of fresh grapes required to produce a given weight of dried grapes -/
theorem fresh_grapes_weight (fresh_water_content : ℝ) (dried_water_content : ℝ) (dried_weight : ℝ) :
  fresh_water_content = 0.8 →
  dried_water_content = 0.2 →
  dried_weight = 10 →
  (1 - fresh_water_content) * (dried_weight / (1 - dried_water_content)) = 40 :=
by sorry

end NUMINAMATH_CALUDE_fresh_grapes_weight_l3669_366926


namespace NUMINAMATH_CALUDE_probability_of_two_distinct_roots_l3669_366957

/-- Represents the outcome of rolling two dice where at least one die shows 4 -/
inductive DiceRoll
  | first_four (second : Nat)
  | second_four (first : Nat)
  | both_four

/-- The set of all possible outcomes when rolling two dice and at least one is 4 -/
def all_outcomes : Finset DiceRoll :=
  sorry

/-- Checks if the quadratic equation x^2 + mx + n = 0 has two distinct real roots -/
def has_two_distinct_roots (roll : DiceRoll) : Bool :=
  sorry

/-- The set of outcomes where the equation has two distinct real roots -/
def favorable_outcomes : Finset DiceRoll :=
  sorry

theorem probability_of_two_distinct_roots :
  (Finset.card favorable_outcomes) / (Finset.card all_outcomes) = 5 / 11 :=
sorry

end NUMINAMATH_CALUDE_probability_of_two_distinct_roots_l3669_366957


namespace NUMINAMATH_CALUDE_parabola_equation_from_conditions_l3669_366939

/-- A parabola is defined by its focus-directrix distance and a point it passes through. -/
structure Parabola where
  focus_directrix_distance : ℝ
  point : ℝ × ℝ

/-- The equation of a parabola in the form y^2 = ax, where a is a real number. -/
def parabola_equation (a : ℝ) : ℝ → ℝ → Prop :=
  fun x y => y^2 = a * x

theorem parabola_equation_from_conditions (p : Parabola) 
  (h1 : p.focus_directrix_distance = 2)
  (h2 : p.point = (1, 2)) :
  parabola_equation 4 = fun x y => y^2 = 4 * x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_from_conditions_l3669_366939


namespace NUMINAMATH_CALUDE_triangle_area_from_rectangle_ratio_l3669_366973

/-- Given a rectangle with length 6 cm and width 4 cm, and a triangle whose area is in a 5:2 ratio
    with the rectangle's area, prove that the area of the triangle is 60 cm². -/
theorem triangle_area_from_rectangle_ratio :
  ∀ (rectangle_length rectangle_width triangle_area : ℝ),
  rectangle_length = 6 →
  rectangle_width = 4 →
  5 * (rectangle_length * rectangle_width) = 2 * triangle_area →
  triangle_area = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_rectangle_ratio_l3669_366973


namespace NUMINAMATH_CALUDE_product_increase_l3669_366983

theorem product_increase : ∃ n : ℤ, 
  53 * n = 1585 ∧ 
  53 * n - 35 * n = 535 :=
by sorry

end NUMINAMATH_CALUDE_product_increase_l3669_366983


namespace NUMINAMATH_CALUDE_rook_in_subrectangle_l3669_366943

/-- Represents a chessboard with rooks -/
structure Chessboard :=
  (rooks : Finset (Nat × Nat))
  (valid : rooks.card = 8)
  (no_attack : ∀ (r1 r2 : Nat × Nat), r1 ∈ rooks → r2 ∈ rooks → r1 ≠ r2 → 
    (r1.1 ≠ r2.1 ∧ r1.2 ≠ r2.2))

/-- Represents a 4x5 subrectangle on the chessboard -/
structure Subrectangle :=
  (top_left : Nat × Nat)
  (valid : top_left.1 ≤ 4 ∧ top_left.2 ≤ 3)

/-- Main theorem: Any 4x5 subrectangle contains at least one rook -/
theorem rook_in_subrectangle (board : Chessboard) (sub : Subrectangle) : 
  ∃ (rook : Nat × Nat), rook ∈ board.rooks ∧ 
    sub.top_left.1 ≤ rook.1 ∧ rook.1 < sub.top_left.1 + 4 ∧
    sub.top_left.2 ≤ rook.2 ∧ rook.2 < sub.top_left.2 + 5 := by
  sorry

end NUMINAMATH_CALUDE_rook_in_subrectangle_l3669_366943


namespace NUMINAMATH_CALUDE_division_result_l3669_366915

theorem division_result (h : 144 * 177 = 25488) : 254.88 / 0.177 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l3669_366915


namespace NUMINAMATH_CALUDE_min_value_and_relationship_l3669_366951

theorem min_value_and_relationship (a b : ℝ) : 
  (4 + (a + b)^2 ≥ 4) ∧ (4 + (a + b)^2 = 4 ↔ a + b = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_relationship_l3669_366951


namespace NUMINAMATH_CALUDE_triangle_tangent_identity_l3669_366959

theorem triangle_tangent_identity (α β γ : Real) (h : α + β + γ = PI) :
  Real.tan (α/2) * Real.tan (β/2) + Real.tan (β/2) * Real.tan (γ/2) + Real.tan (γ/2) * Real.tan (α/2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_identity_l3669_366959


namespace NUMINAMATH_CALUDE_black_car_speed_l3669_366942

/-- Proves that given the conditions of the car problem, the black car's speed is 50 mph -/
theorem black_car_speed (red_speed : ℝ) (initial_gap : ℝ) (overtake_time : ℝ) : ℝ :=
  let black_speed := (initial_gap + red_speed * overtake_time) / overtake_time
  by
    sorry

#check black_car_speed 40 30 3 = 50

end NUMINAMATH_CALUDE_black_car_speed_l3669_366942


namespace NUMINAMATH_CALUDE_unique_n_l3669_366938

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def digit_sum (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

def digit_product (n : ℕ) : ℕ := 
  (n / 100) * ((n / 10) % 10) * (n % 10)

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem unique_n : 
  ∃! n : ℕ, 
    is_three_digit n ∧ 
    is_perfect_square n ∧ 
    is_two_digit (digit_sum n) ∧
    (∀ m : ℕ, is_three_digit m ∧ is_perfect_square m ∧ digit_product m = digit_product n → m = n) ∧
    (∃ m : ℕ, is_three_digit m ∧ is_perfect_square m ∧ m ≠ n ∧ digit_sum m = digit_sum n) ∧
    (∀ m : ℕ, is_three_digit m ∧ is_perfect_square m ∧ digit_sum m = digit_sum n →
      (∀ k : ℕ, is_three_digit k ∧ is_perfect_square k ∧ digit_product k = digit_product m → k = m)) ∧
    n = 841 :=
by sorry

end NUMINAMATH_CALUDE_unique_n_l3669_366938


namespace NUMINAMATH_CALUDE_equation_solution_l3669_366932

theorem equation_solution (x : ℝ) :
  (1 / (Real.sqrt x + Real.sqrt (x - 2)) + 1 / (Real.sqrt x + Real.sqrt (x + 2)) = 1 / 4) →
  x = 257 / 16 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3669_366932


namespace NUMINAMATH_CALUDE_koolaid_percentage_is_four_percent_l3669_366931

def koolaid_percentage (initial_powder initial_water evaporation water_multiplier : ℕ) : ℚ :=
  let remaining_water := initial_water - evaporation
  let final_water := remaining_water * water_multiplier
  let total_liquid := initial_powder + final_water
  (initial_powder : ℚ) / total_liquid * 100

theorem koolaid_percentage_is_four_percent :
  koolaid_percentage 2 16 4 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_koolaid_percentage_is_four_percent_l3669_366931


namespace NUMINAMATH_CALUDE_point_transformation_l3669_366963

/-- Given a point B(5, -1) moved 3 units upwards to point A(a+1, 1-b), prove that a = 4 and b = -1 -/
theorem point_transformation (a b : ℝ) : 
  (5 : ℝ) = a + 1 ∧ 
  (1 : ℝ) - b = -1 + 3 → 
  a = 4 ∧ b = -1 := by sorry

end NUMINAMATH_CALUDE_point_transformation_l3669_366963


namespace NUMINAMATH_CALUDE_probability_of_selection_X_l3669_366985

theorem probability_of_selection_X 
  (prob_Y : ℝ) 
  (prob_X_and_Y : ℝ) 
  (h1 : prob_Y = 2/5) 
  (h2 : prob_X_and_Y = 0.13333333333333333) :
  ∃ (prob_X : ℝ), prob_X = 1/3 ∧ prob_X_and_Y = prob_X * prob_Y :=
by
  sorry

end NUMINAMATH_CALUDE_probability_of_selection_X_l3669_366985


namespace NUMINAMATH_CALUDE_final_time_sum_l3669_366924

-- Define a structure for time
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

-- Define a function to add time
def addTime (start : Time) (elapsedHours elapsedMinutes elapsedSeconds : Nat) : Time :=
  sorry

-- Define a function to calculate the sum of time components
def sumTimeComponents (t : Time) : Nat :=
  t.hours + t.minutes + t.seconds

-- Theorem statement
theorem final_time_sum (startTime : Time) 
  (h1 : startTime.hours = 3) 
  (h2 : startTime.minutes = 0) 
  (h3 : startTime.seconds = 0) : 
  let finalTime := addTime startTime 240 58 30
  sumTimeComponents finalTime = 91 :=
sorry

end NUMINAMATH_CALUDE_final_time_sum_l3669_366924


namespace NUMINAMATH_CALUDE_tims_total_expense_l3669_366937

/-- Calculates Tim's total out-of-pocket expense for medical visits -/
theorem tims_total_expense (tims_visit_cost : ℝ) (tims_insurance_coverage : ℝ) 
  (cats_visit_cost : ℝ) (cats_insurance_coverage : ℝ) 
  (h1 : tims_visit_cost = 300)
  (h2 : tims_insurance_coverage = 0.75 * tims_visit_cost)
  (h3 : cats_visit_cost = 120)
  (h4 : cats_insurance_coverage = 60) : 
  tims_visit_cost - tims_insurance_coverage + cats_visit_cost - cats_insurance_coverage = 135 := by
  sorry


end NUMINAMATH_CALUDE_tims_total_expense_l3669_366937


namespace NUMINAMATH_CALUDE_sweeties_remainder_l3669_366949

theorem sweeties_remainder (m : ℕ) (h : m % 6 = 4) : (2 * m) % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sweeties_remainder_l3669_366949


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_half_l3669_366994

theorem opposite_of_negative_one_half : 
  (-(-(1/2 : ℚ))) = (1/2 : ℚ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_half_l3669_366994


namespace NUMINAMATH_CALUDE_sally_tuesday_shirts_l3669_366972

/-- The number of shirts Sally sewed on Monday -/
def monday_shirts : ℕ := 4

/-- The number of shirts Sally sewed on Wednesday -/
def wednesday_shirts : ℕ := 2

/-- The number of buttons required for each shirt -/
def buttons_per_shirt : ℕ := 5

/-- The total number of buttons needed for all shirts -/
def total_buttons : ℕ := 45

/-- The number of shirts Sally sewed on Tuesday -/
def tuesday_shirts : ℕ := 3

theorem sally_tuesday_shirts :
  tuesday_shirts = (total_buttons - (monday_shirts + wednesday_shirts) * buttons_per_shirt) / buttons_per_shirt :=
by sorry

end NUMINAMATH_CALUDE_sally_tuesday_shirts_l3669_366972


namespace NUMINAMATH_CALUDE_f_prime_zero_l3669_366998

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x) * Real.exp x

theorem f_prime_zero : (deriv f) 0 = -2 := by sorry

end NUMINAMATH_CALUDE_f_prime_zero_l3669_366998


namespace NUMINAMATH_CALUDE_binop_commutative_l3669_366928

-- Define a binary operation on a type
def BinOp (α : Type) := α → α → α

-- Define the properties of the binary operation
class MyBinOp (α : Type) (op : BinOp α) where
  left_cancel : ∀ a b : α, op a (op a b) = b
  right_cancel : ∀ a b : α, op (op a b) b = a

-- State the theorem
theorem binop_commutative {α : Type} (op : BinOp α) [MyBinOp α op] :
  ∀ a b : α, op a b = op b a := by
  sorry

end NUMINAMATH_CALUDE_binop_commutative_l3669_366928


namespace NUMINAMATH_CALUDE_archaeopteryx_humerus_estimate_l3669_366910

/-- Represents the linear regression equation for Archaeopteryx fossil specimens -/
def archaeopteryx_regression (x : ℝ) : ℝ := 1.197 * x - 3.660

/-- Theorem stating the estimated humerus length for a given femur length -/
theorem archaeopteryx_humerus_estimate :
  archaeopteryx_regression 50 = 56.19 := by
  sorry

end NUMINAMATH_CALUDE_archaeopteryx_humerus_estimate_l3669_366910


namespace NUMINAMATH_CALUDE_inequality_proof_l3669_366914

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) + (a + b + c)^2 ≥ 
  4 * Real.sqrt (3 * a * b * c * (a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3669_366914


namespace NUMINAMATH_CALUDE_harry_last_mile_water_l3669_366976

/-- Represents the water consumption during Harry's hike --/
structure HikeWater where
  total_distance : ℝ
  total_time : ℝ
  initial_water : ℝ
  final_water : ℝ
  leak_rate : ℝ
  drink_rate_first_6_miles : ℝ

/-- Calculates the amount of water Harry drank during the last mile of the hike --/
def water_last_mile (h : HikeWater) : ℝ :=
  h.initial_water - h.final_water - (h.leak_rate * h.total_time) - (h.drink_rate_first_6_miles * 6)

/-- Theorem stating that Harry drank 3 cups of water during the last mile --/
theorem harry_last_mile_water :
  let h : HikeWater := {
    total_distance := 7,
    total_time := 3,
    initial_water := 11,
    final_water := 2,
    leak_rate := 1,
    drink_rate_first_6_miles := 0.5
  }
  water_last_mile h = 3 := by sorry

end NUMINAMATH_CALUDE_harry_last_mile_water_l3669_366976


namespace NUMINAMATH_CALUDE_max_viewership_l3669_366903

structure Series where
  runtime : ℕ
  commercials : ℕ
  viewers : ℕ

def seriesA : Series := { runtime := 80, commercials := 1, viewers := 600000 }
def seriesB : Series := { runtime := 40, commercials := 1, viewers := 200000 }

def totalProgramTime : ℕ := 320
def minCommercials : ℕ := 6

def Schedule := ℕ × ℕ  -- (number of A episodes, number of B episodes)

def isValidSchedule (s : Schedule) : Prop :=
  s.1 * seriesA.runtime + s.2 * seriesB.runtime ≤ totalProgramTime ∧
  s.1 * seriesA.commercials + s.2 * seriesB.commercials ≥ minCommercials

def viewership (s : Schedule) : ℕ :=
  s.1 * seriesA.viewers + s.2 * seriesB.viewers

theorem max_viewership :
  ∃ (s : Schedule), isValidSchedule s ∧
    ∀ (s' : Schedule), isValidSchedule s' → viewership s' ≤ viewership s ∧
    viewership s = 2000000 :=
  sorry

end NUMINAMATH_CALUDE_max_viewership_l3669_366903


namespace NUMINAMATH_CALUDE_abc_inequality_l3669_366967

/-- Given a = 2/ln(4), b = ln(3)/ln(2), c = 3/2, prove that b > c > a -/
theorem abc_inequality (a b c : ℝ) (ha : a = 2 / Real.log 4) (hb : b = Real.log 3 / Real.log 2) (hc : c = 3 / 2) :
  b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3669_366967


namespace NUMINAMATH_CALUDE_choral_group_max_size_l3669_366950

theorem choral_group_max_size :
  ∀ (n s : ℕ),
  (∃ (m : ℕ),
    m < 150 ∧
    n * s + 4 = m ∧
    (s - 3) * (n + 2) = m) →
  (∀ (m : ℕ),
    m < 150 ∧
    (∃ (x y : ℕ),
      x * y + 4 = m ∧
      (y - 3) * (x + 2) = m) →
    m ≤ 144) :=
by sorry

end NUMINAMATH_CALUDE_choral_group_max_size_l3669_366950


namespace NUMINAMATH_CALUDE_sin_even_function_phi_l3669_366941

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem sin_even_function_phi (φ : ℝ) 
  (h1 : is_even_function (fun x ↦ Real.sin (x + φ)))
  (h2 : 0 ≤ φ ∧ φ ≤ π) :
  φ = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_even_function_phi_l3669_366941


namespace NUMINAMATH_CALUDE_inequality_proof_l3669_366908

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + a * c ≤ 1 / 3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3669_366908
