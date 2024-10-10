import Mathlib

namespace brenda_remaining_mice_l795_79529

def total_baby_mice : ℕ := 3 * 8

def mice_given_to_robbie : ℕ := total_baby_mice / 6

def mice_sold_to_pet_store : ℕ := 3 * mice_given_to_robbie

def remaining_after_pet_store : ℕ := total_baby_mice - (mice_given_to_robbie + mice_sold_to_pet_store)

def mice_sold_as_feeder : ℕ := remaining_after_pet_store / 2

theorem brenda_remaining_mice :
  total_baby_mice - (mice_given_to_robbie + mice_sold_to_pet_store + mice_sold_as_feeder) = 4 := by
  sorry

end brenda_remaining_mice_l795_79529


namespace expand_binomials_l795_79587

theorem expand_binomials (a : ℝ) : (a + 2) * (a - 3) = a^2 - a - 6 := by
  sorry

end expand_binomials_l795_79587


namespace weight_of_A_l795_79593

/-- Given the weights of five people A, B, C, D, and E, prove that A weighs 77 kg -/
theorem weight_of_A (A B C D E : ℝ) : 
  (A + B + C) / 3 = 84 →
  (A + B + C + D) / 4 = 80 →
  E = D + 5 →
  (B + C + D + E) / 4 = 79 →
  A = 77 := by
sorry

end weight_of_A_l795_79593


namespace negative_comparison_l795_79533

theorem negative_comparison : -0.5 > -0.7 := by
  sorry

end negative_comparison_l795_79533


namespace pascal_triangle_50th_row_third_number_l795_79555

theorem pascal_triangle_50th_row_third_number :
  Nat.choose 50 2 = 1225 := by
  sorry

end pascal_triangle_50th_row_third_number_l795_79555


namespace gcf_360_150_l795_79563

theorem gcf_360_150 : Nat.gcd 360 150 = 30 := by
  sorry

end gcf_360_150_l795_79563


namespace remainder_problem_l795_79592

theorem remainder_problem (n : ℤ) (h : n ≡ 3 [ZMOD 4]) : 7 * n ≡ 1 [ZMOD 5] := by
  sorry

end remainder_problem_l795_79592


namespace cos_42_cos_18_minus_cos_48_sin_18_l795_79564

theorem cos_42_cos_18_minus_cos_48_sin_18 :
  Real.cos (42 * π / 180) * Real.cos (18 * π / 180) -
  Real.cos (48 * π / 180) * Real.sin (18 * π / 180) = 1 / 2 := by
  sorry

end cos_42_cos_18_minus_cos_48_sin_18_l795_79564


namespace power_and_arithmetic_equality_l795_79549

theorem power_and_arithmetic_equality : (-1)^100 * 5 + (-2)^3 / 4 = 3 := by
  sorry

end power_and_arithmetic_equality_l795_79549


namespace largest_consecutive_sum_achievable_486_largest_k_is_486_l795_79537

theorem largest_consecutive_sum (k : ℕ) : 
  (∃ a : ℕ, (k * (2 * a + k - 1)) / 2 = 3^11) → k ≤ 486 :=
by
  sorry

theorem achievable_486 : 
  ∃ a : ℕ, (486 * (2 * a + 486 - 1)) / 2 = 3^11 :=
by
  sorry

theorem largest_k_is_486 : 
  (∃ k : ℕ, k > 486 ∧ ∃ a : ℕ, (k * (2 * a + k - 1)) / 2 = 3^11) → False :=
by
  sorry

end largest_consecutive_sum_achievable_486_largest_k_is_486_l795_79537


namespace fifth_root_of_x_fourth_root_of_x_l795_79597

theorem fifth_root_of_x_fourth_root_of_x (x : ℝ) (hx : x > 0) :
  (x * x^(1/4))^(1/5) = x^(1/4) := by
  sorry

end fifth_root_of_x_fourth_root_of_x_l795_79597


namespace negation_equivalence_l795_79599

theorem negation_equivalence :
  (¬ ∀ (a b : ℝ), a + b = 0 → a^2 + b^2 = 0) ↔
  (∃ (a b : ℝ), a + b = 0 ∧ a^2 + b^2 ≠ 0) :=
by sorry

end negation_equivalence_l795_79599


namespace car_trip_average_speed_l795_79521

/-- Calculates the average speed of a car trip given specific conditions -/
theorem car_trip_average_speed :
  let total_time : ℝ := 6
  let first_part_time : ℝ := 4
  let first_part_speed : ℝ := 35
  let second_part_speed : ℝ := 44
  let total_distance : ℝ := first_part_speed * first_part_time + 
                             second_part_speed * (total_time - first_part_time)
  total_distance / total_time = 38 := by
sorry

end car_trip_average_speed_l795_79521


namespace b_equals_one_l795_79543

-- Define the variables
variable (a b y : ℝ)

-- Define the conditions
def condition1 : Prop := |b - y| = b + y - a
def condition2 : Prop := |b + y| = b + a

-- State the theorem
theorem b_equals_one (h1 : condition1 a b y) (h2 : condition2 a b y) : b = 1 := by
  sorry

end b_equals_one_l795_79543


namespace square_sum_seventeen_l795_79523

theorem square_sum_seventeen (x y : ℝ) 
  (h1 : y + 7 = (x - 3)^2) 
  (h2 : x + 7 = (y - 3)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 17 := by
sorry

end square_sum_seventeen_l795_79523


namespace negation_of_positive_quadratic_l795_79590

theorem negation_of_positive_quadratic (x : ℝ) :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) :=
by sorry

end negation_of_positive_quadratic_l795_79590


namespace newton_6_years_or_more_percentage_l795_79509

/-- Represents the number of marks for each year range on the graph --/
structure EmployeeDistribution :=
  (less_than_1_year : ℕ)
  (one_to_2_years : ℕ)
  (two_to_3_years : ℕ)
  (three_to_4_years : ℕ)
  (four_to_5_years : ℕ)
  (five_to_6_years : ℕ)
  (six_to_7_years : ℕ)
  (seven_to_8_years : ℕ)
  (eight_to_9_years : ℕ)
  (nine_to_10_years : ℕ)

/-- Calculates the percentage of employees who have worked for 6 years or more --/
def percentage_6_years_or_more (dist : EmployeeDistribution) : ℚ :=
  let total_marks := dist.less_than_1_year + dist.one_to_2_years + dist.two_to_3_years +
                     dist.three_to_4_years + dist.four_to_5_years + dist.five_to_6_years +
                     dist.six_to_7_years + dist.seven_to_8_years + dist.eight_to_9_years +
                     dist.nine_to_10_years
  let marks_6_plus := dist.six_to_7_years + dist.seven_to_8_years + dist.eight_to_9_years +
                      dist.nine_to_10_years
  (marks_6_plus : ℚ) / (total_marks : ℚ) * 100

/-- The given distribution of marks on the graph --/
def newton_distribution : EmployeeDistribution :=
  { less_than_1_year := 6,
    one_to_2_years := 6,
    two_to_3_years := 7,
    three_to_4_years := 4,
    four_to_5_years := 3,
    five_to_6_years := 3,
    six_to_7_years := 3,
    seven_to_8_years := 1,
    eight_to_9_years := 1,
    nine_to_10_years := 1 }

theorem newton_6_years_or_more_percentage :
  percentage_6_years_or_more newton_distribution = 17.14 := by
  sorry

end newton_6_years_or_more_percentage_l795_79509


namespace unique_two_digit_integer_l795_79517

theorem unique_two_digit_integer (u : ℕ) : 
  (u ≥ 10 ∧ u ≤ 99) ∧ (17 * u) % 100 = 45 ↔ u = 85 := by
  sorry

end unique_two_digit_integer_l795_79517


namespace three_cyclic_equations_l795_79553

theorem three_cyclic_equations (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
    a = x + 1/y ∧ a = y + 1/z ∧ a = z + 1/x) ↔ 
  (a = 1 ∨ a = -1) :=
sorry

end three_cyclic_equations_l795_79553


namespace multiplication_mistake_difference_l795_79574

theorem multiplication_mistake_difference : 
  let correct_number : ℕ := 139
  let correct_multiplier : ℕ := 43
  let mistaken_multiplier : ℕ := 34
  (correct_number * correct_multiplier) - (correct_number * mistaken_multiplier) = 1251 := by
  sorry

end multiplication_mistake_difference_l795_79574


namespace polynomial_coefficient_identity_l795_79501

theorem polynomial_coefficient_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x + Real.sqrt 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
sorry

end polynomial_coefficient_identity_l795_79501


namespace total_tasters_l795_79520

/-- Represents the number of apple pies Sedrach has -/
def num_pies : ℕ := 13

/-- Represents the number of halves each pie can be divided into -/
def halves_per_pie : ℕ := 2

/-- Represents the number of bite-size samples each half can be split into -/
def samples_per_half : ℕ := 5

/-- Theorem stating the total number of people who can taste Sedrach's apple pies -/
theorem total_tasters : num_pies * halves_per_pie * samples_per_half = 130 := by
  sorry

end total_tasters_l795_79520


namespace unique_stamp_denomination_l795_79503

/-- A function that determines if a postage value can be formed with given stamp denominations -/
def can_form_postage (n : ℕ) (postage : ℕ) : Prop :=
  ∃ (a b c : ℕ), postage = 10 * a + n * b + (n + 1) * c

/-- The main theorem stating that 16 is the unique positive integer satisfying the conditions -/
theorem unique_stamp_denomination : 
  ∃! (n : ℕ), n > 0 ∧ 
    (¬ can_form_postage n 120) ∧ 
    (∀ m > 120, can_form_postage n m) ∧
    n = 16 :=
sorry

end unique_stamp_denomination_l795_79503


namespace three_numbers_sum_l795_79519

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c →
  b = 10 →
  (a + b + c) / 3 = a + 20 →
  (a + b + c) / 3 = c - 30 →
  a + b + c = 60 := by
sorry

end three_numbers_sum_l795_79519


namespace jerry_action_figures_l795_79583

/-- Given an initial count of action figures, a number removed, and a final count,
    this function calculates how many action figures were added. -/
def actionFiguresAdded (initial final removed : ℕ) : ℕ :=
  final + removed - initial

/-- Theorem stating that given the specific conditions in the problem,
    the number of action figures added must be 11. -/
theorem jerry_action_figures :
  actionFiguresAdded 7 8 10 = 11 := by
  sorry

end jerry_action_figures_l795_79583


namespace rectangle_area_change_l795_79513

theorem rectangle_area_change (initial_area : ℝ) : 
  initial_area = 540 →
  (0.8 * 1.15) * initial_area = 496.8 :=
by
  sorry

end rectangle_area_change_l795_79513


namespace special_nine_digit_numbers_exist_l795_79580

/-- Represents a nine-digit number in the specified format -/
structure NineDigitNumber where
  a₁ : ℕ
  a₂ : ℕ
  a₃ : ℕ
  b₁ : ℕ
  b₂ : ℕ
  b₃ : ℕ
  h₁ : a₁ ≠ 0
  h₂ : b₁ * 100 + b₂ * 10 + b₃ = 2 * (a₁ * 100 + a₂ * 10 + a₃)

/-- The value of the nine-digit number -/
def NineDigitNumber.value (n : NineDigitNumber) : ℕ :=
  n.a₁ * 100000000 + n.a₂ * 10000000 + n.a₃ * 1000000 +
  n.b₁ * 100000 + n.b₂ * 10000 + n.b₃ * 1000 +
  n.a₁ * 100 + n.a₂ * 10 + n.a₃

/-- Theorem stating the existence of the special nine-digit numbers -/
theorem special_nine_digit_numbers_exist : ∃ (n : NineDigitNumber),
  (∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧
    n.value = (p₁ * p₂ * p₃ * p₄ * p₅)^2) ∧
  (n.value = 100200100 ∨ n.value = 225450225) :=
sorry

end special_nine_digit_numbers_exist_l795_79580


namespace min_value_of_f_l795_79547

-- Define the function f
def f (x : ℝ) : ℝ := (x - 2024)^2

-- State the theorem
theorem min_value_of_f :
  (∀ x : ℝ, f (x + 2023) = x^2 - 2*x + 1) →
  (∃ x₀ : ℝ, f x₀ = 0 ∧ ∀ x : ℝ, f x ≥ 0) :=
by sorry

end min_value_of_f_l795_79547


namespace radar_coverage_l795_79578

noncomputable def n : ℕ := 7
def r : ℝ := 41
def w : ℝ := 18

theorem radar_coverage (n : ℕ) (r w : ℝ) 
  (h_n : n = 7) 
  (h_r : r = 41) 
  (h_w : w = 18) :
  ∃ (max_distance area : ℝ),
    max_distance = 40 / Real.sin (180 / n * π / 180) ∧
    area = 1440 * π / Real.tan (180 / n * π / 180) := by
  sorry

end radar_coverage_l795_79578


namespace largest_integer_k_for_distinct_roots_l795_79581

theorem largest_integer_k_for_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (k - 2) * x^2 - 4 * x + 4 = 0 ∧ 
   (k - 2) * y^2 - 4 * y + 4 = 0) →
  (∀ m : ℤ, m > 1 → (m : ℝ) > k) :=
by sorry

#check largest_integer_k_for_distinct_roots

end largest_integer_k_for_distinct_roots_l795_79581


namespace vasya_gift_choices_l795_79528

theorem vasya_gift_choices (n_cars : ℕ) (n_sets : ℕ) : 
  n_cars = 7 → n_sets = 5 → (n_cars.choose 2) + (n_sets.choose 2) + n_cars * n_sets = 66 :=
by
  sorry

end vasya_gift_choices_l795_79528


namespace vector_operation_proof_l795_79556

/-- Prove that the vector operation (3, -8) - 3(2, -5) + (-1, 4) equals (-4, 11) -/
theorem vector_operation_proof :
  let v1 : Fin 2 → ℝ := ![3, -8]
  let v2 : Fin 2 → ℝ := ![2, -5]
  let v3 : Fin 2 → ℝ := ![-1, 4]
  v1 - 3 • v2 + v3 = ![-4, 11] := by
  sorry

end vector_operation_proof_l795_79556


namespace planar_cube_area_is_600_l795_79568

/-- The side length of each square in centimeters -/
def side_length : ℝ := 10

/-- The number of faces in a cube -/
def cube_faces : ℕ := 6

/-- The area of the planar figure of a cube in square centimeters -/
def planar_cube_area : ℝ := side_length^2 * cube_faces

/-- Theorem: The area of a planar figure representing a cube, 
    made up of squares with side length 10 cm, is 600 cm² -/
theorem planar_cube_area_is_600 : planar_cube_area = 600 := by
  sorry

end planar_cube_area_is_600_l795_79568


namespace truth_values_equivalence_l795_79561

theorem truth_values_equivalence (p q : Prop) 
  (h1 : p ∨ q) (h2 : ¬(p ∧ q)) : p ↔ ¬q := by
  sorry

end truth_values_equivalence_l795_79561


namespace complex_equation_solution_l795_79557

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 2 * Complex.I → z = 1 - Complex.I := by
  sorry

end complex_equation_solution_l795_79557


namespace average_and_difference_l795_79584

theorem average_and_difference (y : ℝ) : 
  (35 + y) / 2 = 44 → |35 - y| = 18 := by
  sorry

end average_and_difference_l795_79584


namespace quadratic_factorization_l795_79518

theorem quadratic_factorization (x : ℝ) :
  -x^2 + 4*x - 4 = -(x - 2)^2 := by sorry

end quadratic_factorization_l795_79518


namespace distribute_18_balls_5_boxes_l795_79558

/-- The number of ways to distribute n identical balls into k distinct boxes,
    with each box containing at least m balls. -/
def distribute_balls (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

theorem distribute_18_balls_5_boxes :
  distribute_balls 18 5 3 = 35 := by
  sorry

end distribute_18_balls_5_boxes_l795_79558


namespace expression_evaluation_l795_79512

theorem expression_evaluation : 80 + 5 * 12 / (180 / 3) = 81 := by
  sorry

end expression_evaluation_l795_79512


namespace water_scooped_out_l795_79562

theorem water_scooped_out (total_weight : ℝ) (alcohol_concentration : ℝ) :
  total_weight = 10 ∧ alcohol_concentration = 0.75 →
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ 10 ∧ x / total_weight = alcohol_concentration ∧ x = 7.5 :=
by sorry

end water_scooped_out_l795_79562


namespace typing_time_proof_l795_79585

def original_speed : ℕ := 212
def speed_reduction : ℕ := 40
def document_length : ℕ := 3440

theorem typing_time_proof :
  let new_speed := original_speed - speed_reduction
  document_length / new_speed = 20 := by sorry

end typing_time_proof_l795_79585


namespace undefined_values_count_l795_79586

theorem undefined_values_count : 
  ∃! (s : Finset ℝ), (∀ x ∈ s, (x^2 + 2*x - 3) * (x - 3) = 0) ∧ Finset.card s = 3 := by
  sorry

end undefined_values_count_l795_79586


namespace angle_equality_in_right_triangle_l795_79534

theorem angle_equality_in_right_triangle (D E F : Real) (angle_D angle_E angle_3 angle_4 : Real) :
  angle_E = 90 →
  angle_D = 70 →
  angle_3 = angle_4 →
  angle_3 + angle_4 = 180 - angle_E - angle_D →
  angle_4 = 45 := by
sorry

end angle_equality_in_right_triangle_l795_79534


namespace max_value_on_triangle_vertices_l795_79579

-- Define a 2D point
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a triangle in 2D space
structure Triangle where
  P : Point2D
  Q : Point2D
  R : Point2D

-- Define a linear function f(x, y) = ax + by + c
def linearFunction (a b c : ℝ) (p : Point2D) : ℝ :=
  a * p.x + b * p.y + c

-- Define a predicate to check if a point is in or on a triangle
def isInOrOnTriangle (t : Triangle) (p : Point2D) : Prop :=
  sorry -- The actual implementation is not needed for the theorem statement

-- Theorem statement
theorem max_value_on_triangle_vertices 
  (t : Triangle) (a b c : ℝ) (p : Point2D) 
  (h : isInOrOnTriangle t p) : 
  linearFunction a b c p ≤ max 
    (linearFunction a b c t.P) 
    (max (linearFunction a b c t.Q) (linearFunction a b c t.R)) := by
  sorry


end max_value_on_triangle_vertices_l795_79579


namespace twelfth_even_multiple_of_5_l795_79596

/-- The nth positive integer that is both even and a multiple of 5 -/
def evenMultipleOf5 (n : ℕ) : ℕ := 10 * n

/-- Proof that the 12th positive integer that is both even and a multiple of 5 is 120 -/
theorem twelfth_even_multiple_of_5 : evenMultipleOf5 12 = 120 := by
  sorry

end twelfth_even_multiple_of_5_l795_79596


namespace mrs_hilt_daily_reading_l795_79546

/-- The number of books Mrs. Hilt read in one week -/
def books_per_week : ℕ := 14

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of books Mrs. Hilt read per day -/
def books_per_day : ℚ := books_per_week / days_in_week

theorem mrs_hilt_daily_reading : books_per_day = 2 := by
  sorry

end mrs_hilt_daily_reading_l795_79546


namespace carol_meets_alice_l795_79515

/-- Alice's speed in miles per hour -/
def alice_speed : ℝ := 4

/-- Carol's speed in miles per hour -/
def carol_speed : ℝ := 6

/-- Initial distance between Carol and Alice in miles -/
def initial_distance : ℝ := 5

/-- Time in minutes for Carol to meet Alice -/
def meeting_time : ℝ := 30

theorem carol_meets_alice : 
  initial_distance / (alice_speed + carol_speed) * 60 = meeting_time := by
  sorry

end carol_meets_alice_l795_79515


namespace equation_describes_ellipse_l795_79559

def is_ellipse (f₁ f₂ : ℝ × ℝ) (c : ℝ) : Prop :=
  ∀ p : ℝ × ℝ, Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) + 
               Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = c

theorem equation_describes_ellipse :
  is_ellipse (0, 2) (6, -4) 12 := by sorry

end equation_describes_ellipse_l795_79559


namespace new_rectangle_area_l795_79566

/-- Given a rectangle with sides a and b, construct a new rectangle and calculate its area -/
theorem new_rectangle_area (a b : ℝ) (ha : a = 3) (hb : b = 4) : 
  let d := Real.sqrt (a^2 + b^2)
  let new_length := d + min a b
  let new_breadth := d - max a b
  new_length * new_breadth = 8 := by sorry

end new_rectangle_area_l795_79566


namespace tangent_line_implies_b_minus_a_equals_three_l795_79506

-- Define the function f(x) = ax³ + bx - 1
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 1

-- Define the derivative of f
def f_derivative (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + b

theorem tangent_line_implies_b_minus_a_equals_three (a b : ℝ) :
  f_derivative a b 1 = 1 ∧ f a b 1 = 1 → b - a = 3 := by
  sorry

end tangent_line_implies_b_minus_a_equals_three_l795_79506


namespace cab_cost_for_week_long_event_l795_79594

/-- Calculates the total cost of cab rides for a week-long event -/
def total_cab_cost (days : ℕ) (distance : ℝ) (cost_per_mile : ℝ) : ℝ :=
  2 * days * distance * cost_per_mile

/-- Proves that the total cost of cab rides for the given conditions is $7000 -/
theorem cab_cost_for_week_long_event :
  total_cab_cost 7 200 2.5 = 7000 := by
  sorry

end cab_cost_for_week_long_event_l795_79594


namespace range_of_a_minus_b_l795_79582

theorem range_of_a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 4) :
  ∃ (x : ℝ), -3 < x ∧ x < 6 ∧ ∃ (a' b' : ℝ), 1 < a' ∧ a' < 4 ∧ -2 < b' ∧ b' < 4 ∧ x = a' - b' :=
by sorry

end range_of_a_minus_b_l795_79582


namespace product_of_two_digit_numbers_l795_79527

theorem product_of_two_digit_numbers (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) → 
  (10 ≤ b ∧ b < 100) → 
  a * b = 4680 → 
  min a b = 40 := by
sorry

end product_of_two_digit_numbers_l795_79527


namespace indeterminate_equation_solution_l795_79530

theorem indeterminate_equation_solution (a b : ℤ) :
  ∃ (x y z u v w t : ℤ),
    x^4 + y^4 + z^4 = u^2 + v^2 + w^2 + t^2 ∧
    x = a ∧
    y = b ∧
    z = a + b ∧
    u = a^2 + a*b + b^2 ∧
    v = a*b ∧
    w = a*b*(a + b) ∧
    t = b*(a + b) := by
  sorry

end indeterminate_equation_solution_l795_79530


namespace inscribed_quadrilateral_fourth_side_l795_79575

/-- A quadrilateral inscribed in a circle with given properties -/
structure InscribedQuadrilateral where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The lengths of the four sides of the quadrilateral -/
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- The theorem stating the properties of the specific inscribed quadrilateral -/
theorem inscribed_quadrilateral_fourth_side
  (q : InscribedQuadrilateral)
  (h_radius : q.radius = 100 * Real.sqrt 6)
  (h_side1 : q.side1 = 100)
  (h_side2 : q.side2 = 200)
  (h_side3 : q.side3 = 200) :
  q.side4 = 100 * Real.sqrt 2 := by
  sorry

end inscribed_quadrilateral_fourth_side_l795_79575


namespace difference_of_squares_65_35_l795_79522

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end difference_of_squares_65_35_l795_79522


namespace emily_furniture_assembly_time_l795_79525

/-- Calculates the total assembly time for furniture -/
def total_assembly_time (
  num_chairs : ℕ) (chair_time : ℕ)
  (num_tables : ℕ) (table_time : ℕ)
  (num_shelves : ℕ) (shelf_time : ℕ)
  (num_wardrobes : ℕ) (wardrobe_time : ℕ) : ℕ :=
  num_chairs * chair_time +
  num_tables * table_time +
  num_shelves * shelf_time +
  num_wardrobes * wardrobe_time

/-- Proves that the total assembly time for Emily's furniture is 137 minutes -/
theorem emily_furniture_assembly_time :
  total_assembly_time 4 8 2 15 3 10 1 45 = 137 := by
  sorry


end emily_furniture_assembly_time_l795_79525


namespace min_employees_needed_l795_79500

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of working days for each employee per week -/
def working_days : ℕ := 5

/-- The number of rest days for each employee per week -/
def rest_days : ℕ := 2

/-- The minimum number of employees required on duty each day -/
def min_employees_per_day : ℕ := 45

/-- The minimum number of employees needed by the company -/
def min_total_employees : ℕ := 63

theorem min_employees_needed :
  ∀ (total_employees : ℕ),
    (∀ (day : Fin days_in_week),
      (total_employees * working_days) / days_in_week ≥ min_employees_per_day) →
    total_employees ≥ min_total_employees :=
by sorry

end min_employees_needed_l795_79500


namespace inverse_variation_doubling_inverse_variation_example_l795_79569

/-- Given two quantities that vary inversely, if one quantity doubles, the other halves -/
theorem inverse_variation_doubling (a b c d : ℝ) (h1 : a * b = c * d) (h2 : c = 2 * a) :
  d = b / 2 := by
  sorry

/-- When a and b vary inversely, if b = 0.5 when a = 800, then b = 0.25 when a = 1600 -/
theorem inverse_variation_example :
  ∃ (k : ℝ), (800 * 0.5 = k) ∧ (1600 * 0.25 = k) := by
  sorry

end inverse_variation_doubling_inverse_variation_example_l795_79569


namespace math_team_selection_count_l795_79532

theorem math_team_selection_count : 
  let total_boys : ℕ := 7
  let total_girls : ℕ := 10
  let team_size : ℕ := 5
  let boys_in_team : ℕ := 2
  let girls_in_team : ℕ := 3
  (Nat.choose total_boys boys_in_team) * (Nat.choose total_girls girls_in_team) = 2520 :=
by sorry

end math_team_selection_count_l795_79532


namespace at_least_one_not_less_than_six_l795_79576

theorem at_least_one_not_less_than_six (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ¬(a + 4 / b < 6 ∧ b + 9 / c < 6 ∧ c + 16 / a < 6) := by
  sorry

end at_least_one_not_less_than_six_l795_79576


namespace archery_score_distribution_l795_79573

theorem archery_score_distribution :
  ∃! (a b c d : ℕ),
    a + b + c + d = 10 ∧
    a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ d ≥ 1 ∧
    8*a + 12*b + 14*c + 18*d = 110 :=
by
  sorry

end archery_score_distribution_l795_79573


namespace frenchBulldogRatioIsTwo_l795_79526

/-- The ratio of French Bulldogs Peter wants to Sam's -/
def frenchBulldogRatio (samGermanShepherds samFrenchBulldogs peterTotalDogs : ℕ) : ℚ :=
  let peterGermanShepherds := 3 * samGermanShepherds
  let peterFrenchBulldogs := peterTotalDogs - peterGermanShepherds
  (peterFrenchBulldogs : ℚ) / samFrenchBulldogs

/-- The ratio of French Bulldogs Peter wants to Sam's is 2:1 -/
theorem frenchBulldogRatioIsTwo :
  frenchBulldogRatio 3 4 17 = 2 := by
  sorry

end frenchBulldogRatioIsTwo_l795_79526


namespace problem_solution_l795_79552

open Real

-- Define the given condition
def alpha_condition (α : ℝ) : Prop := 2 * sin α = cos α

-- Define that α is in the third quadrant
def third_quadrant (α : ℝ) : Prop := π < α ∧ α < 3 * π / 2

theorem problem_solution (α : ℝ) 
  (h1 : alpha_condition α) 
  (h2 : third_quadrant α) : 
  (cos (π - α) = 2 * sqrt 5 / 5) ∧ 
  ((1 + 2 * sin α * sin (π / 2 - α)) / (sin α ^ 2 - cos α ^ 2) = -3) := by
  sorry

end problem_solution_l795_79552


namespace baker_donuts_l795_79577

theorem baker_donuts (total_donuts : ℕ) (boxes : ℕ) (extra_donuts : ℕ) : 
  boxes = 7 → 
  extra_donuts = 6 → 
  ∃ (n : ℕ), n > 0 ∧ total_donuts = 7 * n + 6 := by
  sorry

end baker_donuts_l795_79577


namespace gcd_90_450_l795_79567

theorem gcd_90_450 : Nat.gcd 90 450 = 90 := by
  sorry

end gcd_90_450_l795_79567


namespace mean_proportional_of_segments_l795_79589

theorem mean_proportional_of_segments (a b : ℝ) (ha : a = 2) (hb : b = 6) :
  ∃ c : ℝ, c ^ 2 = a * b ∧ c = 2 * Real.sqrt 3 := by
  sorry

end mean_proportional_of_segments_l795_79589


namespace elberta_amount_l795_79505

/-- The amount of money Granny Smith has -/
def granny_smith : ℕ := 120

/-- The amount of money Anjou has -/
def anjou : ℕ := granny_smith / 2

/-- The amount of money Elberta has -/
def elberta : ℕ := anjou + 5

/-- Theorem stating that Elberta has $65 -/
theorem elberta_amount : elberta = 65 := by
  sorry

end elberta_amount_l795_79505


namespace nonagon_diagonal_intersection_probability_l795_79554

/-- A regular nonagon is a 9-sided regular polygon -/
def RegularNonagon : Type := Unit

/-- The number of diagonals in a regular nonagon -/
def num_diagonals (n : RegularNonagon) : ℕ := 27

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def total_diagonal_pairs (n : RegularNonagon) : ℕ := 351

/-- The number of ways to choose 4 vertices from the nonagon, 
    which correspond to intersecting diagonals -/
def intersecting_diagonal_pairs (n : RegularNonagon) : ℕ := 126

/-- The probability that two randomly chosen diagonals in a regular nonagon intersect -/
def intersection_probability (n : RegularNonagon) : ℚ :=
  intersecting_diagonal_pairs n / total_diagonal_pairs n

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  intersection_probability n = 14 / 39 := by
  sorry


end nonagon_diagonal_intersection_probability_l795_79554


namespace unknown_number_is_nine_l795_79507

def first_number : ℝ := 4.2

def second_number : ℝ := first_number + 2

def third_number : ℝ := first_number + 4

def unknown_number : ℝ := 9 * first_number - 2 * third_number - 2 * second_number

theorem unknown_number_is_nine : unknown_number = 9 := by
  sorry

end unknown_number_is_nine_l795_79507


namespace correct_sunset_time_l795_79545

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := totalMinutes / 60
  let newMinutes := totalMinutes % 60
  ⟨newHours % 24, newMinutes, by sorry, by sorry⟩

theorem correct_sunset_time :
  let sunrise : Time := ⟨7, 12, by sorry, by sorry⟩
  let incorrectDaylight : Nat := 11 * 60 + 15 -- 11 hours and 15 minutes in minutes
  let calculatedSunset := addMinutes sunrise incorrectDaylight
  calculatedSunset.hours = 18 ∧ calculatedSunset.minutes = 27 :=
by sorry

end correct_sunset_time_l795_79545


namespace inequality_chain_l795_79595

theorem inequality_chain (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a^2 + b^2)/(2*c) + (a^2 + c^2)/(2*b) + (b^2 + c^2)/(2*a) ∧
  (a^2 + b^2)/(2*c) + (a^2 + c^2)/(2*b) + (b^2 + c^2)/(2*a) ≤ a^3/(b*c) + b^3/(a*c) + c^3/(a*b) :=
by sorry

end inequality_chain_l795_79595


namespace wheel_marking_theorem_l795_79598

theorem wheel_marking_theorem :
  ∃ (R : ℝ), R > 0 ∧
    ∀ (θ : ℝ), 0 ≤ θ ∧ θ < 360 →
      ∃ (n : ℕ), ∃ (k : ℤ),
        n / (2 * π * R) = θ / 360 + k ∧
        0 ≤ n / (2 * π * R) - k ∧
        n / (2 * π * R) - k < 1 / 360 :=
by sorry

end wheel_marking_theorem_l795_79598


namespace system_solution_l795_79511

theorem system_solution : 
  ∀ x y z : ℕ, 
    (2 * x^2 + 30 * y^2 + 3 * z^2 + 12 * x * y + 12 * y * z = 308 ∧
     2 * x^2 + 6 * y^2 - 3 * z^2 + 12 * x * y - 12 * y * z = 92) →
    ((x = 7 ∧ y = 1 ∧ z = 4) ∨ (x = 4 ∧ y = 2 ∧ z = 2)) := by
  sorry

end system_solution_l795_79511


namespace supplementary_angle_of_60_degrees_l795_79510

theorem supplementary_angle_of_60_degrees (α : Real) : 
  α = 60 → 180 - α = 120 := by sorry

end supplementary_angle_of_60_degrees_l795_79510


namespace skater_race_solution_l795_79524

/-- Represents the speeds and times of two speed skaters in a race --/
structure SkaterRace where
  v : ℝ  -- Speed of the second skater in m/s
  t1 : ℝ  -- Time for the first skater to complete 10000 m in seconds
  t2 : ℝ  -- Time for the second skater to complete 10000 m in seconds

/-- The speeds and times of the skaters satisfy the race conditions --/
def satisfies_conditions (race : SkaterRace) : Prop :=
  let v1 := race.v + 1/3  -- Speed of the first skater
  (v1 * 600 - race.v * 600 = 200) ∧  -- Overtaking condition
  (400 / race.v - 400 / v1 = 2) ∧  -- Lap time difference
  (10000 / v1 = race.t1) ∧  -- First skater's total time
  (10000 / race.v = race.t2)  -- Second skater's total time

/-- The theorem stating the correct speeds and times for the skaters --/
theorem skater_race_solution :
  ∃ (race : SkaterRace),
    satisfies_conditions race ∧
    race.v = 8 ∧
    race.t1 = 1200 ∧
    race.t2 = 1250 :=
sorry

end skater_race_solution_l795_79524


namespace money_problem_l795_79516

theorem money_problem (a b : ℝ) 
  (eq_condition : 6 * a + b = 66)
  (ineq_condition : 4 * a + b < 48) :
  a > 9 ∧ b = 6 := by
sorry

end money_problem_l795_79516


namespace function_value_at_negative_half_l795_79535

theorem function_value_at_negative_half (a : ℝ) (f : ℝ → ℝ) :
  0 < a →
  a ≠ 1 →
  (∀ x, f x = a^x) →
  f 2 = 81 →
  f (-1/2) = 1/3 := by
sorry

end function_value_at_negative_half_l795_79535


namespace cross_in_square_side_length_l795_79541

/-- Represents a cross shape inside a square -/
structure CrossInSquare where
  a : ℝ  -- Side length of the largest square
  area_cross : ℝ -- Area of the cross

/-- The area of the cross is equal to the sum of areas of its component squares -/
def cross_area_equation (c : CrossInSquare) : Prop :=
  c.area_cross = 2 * (c.a / 2)^2 + 2 * (c.a / 4)^2

/-- Theorem stating that if the area of the cross is 810 cm², then the side length of the largest square is 36 cm -/
theorem cross_in_square_side_length 
  (c : CrossInSquare) 
  (h1 : c.area_cross = 810) 
  (h2 : cross_area_equation c) : 
  c.a = 36 := by
sorry

end cross_in_square_side_length_l795_79541


namespace june_christopher_difference_l795_79508

/-- The length of Christopher's sword in inches -/
def christopher_sword : ℕ := 15

/-- The length of Jameson's sword in inches -/
def jameson_sword : ℕ := 2 * christopher_sword + 3

/-- The length of June's sword in inches -/
def june_sword : ℕ := jameson_sword + 5

/-- Theorem: June's sword is 23 inches longer than Christopher's sword -/
theorem june_christopher_difference : june_sword - christopher_sword = 23 := by
  sorry

end june_christopher_difference_l795_79508


namespace equal_distances_l795_79591

/-- Represents a right triangle with squares on its sides -/
structure RightTriangleWithSquares where
  -- The lengths of the sides of the right triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The acute angle α
  α : ℝ
  -- Conditions
  right_triangle : c^2 = a^2 + b^2
  acute_angle : 0 < α ∧ α < π / 2
  angle_sum : α + (π / 2 - α) = π / 2

/-- The theorem stating that the distances O₁O₂ and CO₃ are equal -/
theorem equal_distances (t : RightTriangleWithSquares) : 
  (t.a + t.b) / Real.sqrt 2 = t.c / Real.sqrt 2 :=
sorry

end equal_distances_l795_79591


namespace largest_gold_coins_distribution_l795_79504

theorem largest_gold_coins_distribution (n : ℕ) : 
  n > 50 ∧ n < 150 ∧ 
  ∃ (k : ℕ), n = 7 * k + 2 ∧
  (∀ m : ℕ, m > n → ¬(∃ j : ℕ, m = 7 * j + 2)) →
  n = 149 := by
sorry

end largest_gold_coins_distribution_l795_79504


namespace min_growth_rate_doubles_coverage_l795_79551

-- Define the initial forest coverage area
variable (a : ℝ)
-- Define the natural growth rate
def natural_growth_rate : ℝ := 0.02
-- Define the time period in years
def years : ℕ := 10
-- Define the target multiplier for forest coverage
def target_multiplier : ℝ := 2

-- Define the function for forest coverage area after x years with natural growth
def forest_coverage (x : ℕ) : ℝ := a * (1 + natural_growth_rate) ^ x

-- Define the minimum required growth rate
def min_growth_rate : ℝ := 0.072

-- Theorem statement
theorem min_growth_rate_doubles_coverage :
  ∀ p : ℝ, p ≥ min_growth_rate →
  a * (1 + p) ^ years ≥ target_multiplier * a :=
by sorry

end min_growth_rate_doubles_coverage_l795_79551


namespace point_difference_l795_79571

-- Define the value of a touchdown
def touchdown_value : ℕ := 7

-- Define the number of touchdowns for each team
def brayden_gavin_touchdowns : ℕ := 7
def cole_freddy_touchdowns : ℕ := 9

-- Calculate the points for each team
def brayden_gavin_points : ℕ := brayden_gavin_touchdowns * touchdown_value
def cole_freddy_points : ℕ := cole_freddy_touchdowns * touchdown_value

-- State the theorem
theorem point_difference : cole_freddy_points - brayden_gavin_points = 14 := by
  sorry

end point_difference_l795_79571


namespace knights_problem_l795_79565

/-- Represents the arrangement of knights -/
structure KnightArrangement where
  total : ℕ
  rows : ℕ
  cols : ℕ
  knights_per_row : ℕ
  knights_per_col : ℕ

/-- The conditions of the problem -/
def problem_conditions (k : KnightArrangement) : Prop :=
  k.total = k.rows * k.cols ∧
  k.total - 2 * k.knights_per_row = 24 ∧
  k.total - 2 * k.knights_per_col = 18

/-- The theorem to be proved -/
theorem knights_problem :
  ∀ k : KnightArrangement, problem_conditions k → k.total = 40 :=
by
  sorry


end knights_problem_l795_79565


namespace angle_B_is_140_degrees_l795_79531

/-- A quadrilateral with angles A, B, C, and D -/
structure Quadrilateral where
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  angleD : ℝ

/-- The theorem stating that if the sum of angles A, B, and C in a quadrilateral is 220°, 
    then angle B is 140° -/
theorem angle_B_is_140_degrees (q : Quadrilateral) 
    (h : q.angleA + q.angleB + q.angleC = 220) : q.angleB = 140 := by
  sorry


end angle_B_is_140_degrees_l795_79531


namespace increasing_function_property_l795_79570

-- Define a function f on positive real numbers
variable (f : ℝ → ℝ)

-- Define the property of being increasing for positive real numbers
def IncreasingOnPositive (g : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → g x < g y

-- State the theorem
theorem increasing_function_property
  (h1 : IncreasingOnPositive (fun x => f x - x))
  (h2 : IncreasingOnPositive (fun x => f (x^2) - x^6)) :
  IncreasingOnPositive (fun x => f (x^3) - (Real.sqrt 3 / 2) * x^6) :=
sorry

end increasing_function_property_l795_79570


namespace min_value_x_plus_2y_l795_79544

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 2*y + 2*x*y = 8) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + 2*b + 2*a*b = 8 → x + 2*y ≤ a + 2*b ∧ 
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ + 2*x₀*y₀ = 8 ∧ x₀ + 2*y₀ = 4 := by
  sorry

end min_value_x_plus_2y_l795_79544


namespace cersei_cotton_candies_l795_79548

/-- The number of cotton candies Cersei initially bought -/
def initial_candies : ℕ := 40

/-- The number of cotton candies given to brother and sister -/
def given_to_siblings : ℕ := 10

/-- The fraction of remaining candies given to cousin -/
def fraction_to_cousin : ℚ := 1/4

/-- The number of cotton candies Cersei ate -/
def eaten_candies : ℕ := 12

/-- The number of cotton candies left at the end -/
def remaining_candies : ℕ := 18

theorem cersei_cotton_candies : 
  initial_candies = 40 ∧
  (initial_candies - given_to_siblings) * (1 - fraction_to_cousin) - eaten_candies = remaining_candies :=
by sorry

end cersei_cotton_candies_l795_79548


namespace bijection_image_l795_79502

def B : Set ℤ := {-3, 3, 5}

def f (x : ℤ) : ℤ := 2 * x - 1

theorem bijection_image (A : Set ℤ) :
  (Function.Bijective f) → (f '' A = B) → A = {-1, 2, 3} := by
  sorry

end bijection_image_l795_79502


namespace hyperbola_properties_l795_79538

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define eccentricity
def eccentricity (e : ℝ) : Prop := e = 5/4

-- Define distance from focus to asymptote
def distance_focus_asymptote (d : ℝ) : Prop := d = 3

-- Theorem statement
theorem hyperbola_properties :
  ∃ (e d : ℝ), hyperbola x y ∧ eccentricity e ∧ distance_focus_asymptote d :=
sorry

end hyperbola_properties_l795_79538


namespace zero_of_f_l795_79572

/-- The function f(x) = 4x - 2 -/
def f (x : ℝ) : ℝ := 4 * x - 2

/-- Theorem: The zero of the function f(x) = 4x - 2 is 1/2 -/
theorem zero_of_f : ∃ x : ℝ, f x = 0 ∧ x = 1/2 := by
  sorry

end zero_of_f_l795_79572


namespace triangle_area_determines_p_l795_79540

/-- Given a triangle ABC with vertices A(3, 15), B(15, 0), and C(0, p),
    if the area of triangle ABC is 36, then p = 12.75 -/
theorem triangle_area_determines_p :
  ∀ p : ℝ,
  let A : ℝ × ℝ := (3, 15)
  let B : ℝ × ℝ := (15, 0)
  let C : ℝ × ℝ := (0, p)
  let triangle_area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  triangle_area = 36 → p = 12.75 := by
sorry

end triangle_area_determines_p_l795_79540


namespace pauls_new_books_l795_79514

theorem pauls_new_books (initial_books sold_books current_books : ℕ) : 
  initial_books = 2 → 
  sold_books = 94 → 
  current_books = 58 → 
  current_books = initial_books - sold_books + (sold_books - initial_books + current_books) :=
by
  sorry

end pauls_new_books_l795_79514


namespace polygon_with_20_diagonals_is_octagon_l795_79539

theorem polygon_with_20_diagonals_is_octagon :
  ∀ n : ℕ, n > 2 → (n * (n - 3)) / 2 = 20 → n = 8 := by
sorry

end polygon_with_20_diagonals_is_octagon_l795_79539


namespace our_system_is_linear_l795_79588

/-- Represents a linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  -- ax + by = c

/-- Represents a system of two equations -/
structure EquationSystem where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- Checks if an equation is linear -/
def isLinear (eq : LinearEquation) : Prop :=
  eq.a ≠ 0 ∨ eq.b ≠ 0

/-- Checks if a system consists of two linear equations -/
def isSystemOfTwoLinearEquations (sys : EquationSystem) : Prop :=
  isLinear sys.eq1 ∧ isLinear sys.eq2

/-- The specific system we want to prove is a system of two linear equations -/
def ourSystem : EquationSystem :=
  { eq1 := { a := 1, b := 1, c := 5 }  -- x + y = 5
    eq2 := { a := 0, b := 1, c := 2 }  -- y = 2
  }

/-- Theorem stating that our system is a system of two linear equations -/
theorem our_system_is_linear : isSystemOfTwoLinearEquations ourSystem := by
  sorry


end our_system_is_linear_l795_79588


namespace other_pencil_length_is_12_l795_79536

/-- The length of Isha's pencil in cubes -/
def ishas_pencil_length : ℕ := 12

/-- The total length of both pencils in cubes -/
def total_length : ℕ := 24

/-- The length of the other pencil in cubes -/
def other_pencil_length : ℕ := total_length - ishas_pencil_length

theorem other_pencil_length_is_12 : other_pencil_length = 12 := by
  sorry

end other_pencil_length_is_12_l795_79536


namespace six_ronna_scientific_notation_l795_79542

/-- Represents the number of zeros after the number for the "ronna" prefix -/
def ronna_zeros : ℕ := 27

/-- Converts a number with the "ronna" prefix to scientific notation -/
def ronna_to_scientific (n : ℝ) : ℝ := n * (10 ^ ronna_zeros)

/-- Theorem stating that 6 ronna is equal to 6 × 10^27 -/
theorem six_ronna_scientific_notation : ronna_to_scientific 6 = 6 * (10 ^ 27) := by
  sorry

end six_ronna_scientific_notation_l795_79542


namespace kates_wand_cost_l795_79560

/-- Proves that the original cost of each wand is $60 given the conditions of Kate's wand purchase and sale. -/
theorem kates_wand_cost (total_wands : ℕ) (kept_wands : ℕ) (sold_wands : ℕ) 
  (price_increase : ℕ) (total_collected : ℕ) : ℕ :=
  by
  have h1 : total_wands = 3 := by sorry
  have h2 : kept_wands = 1 := by sorry
  have h3 : sold_wands = 2 := by sorry
  have h4 : price_increase = 5 := by sorry
  have h5 : total_collected = 130 := by sorry
  
  have h6 : sold_wands = total_wands - kept_wands := by sorry
  
  have h7 : total_collected / sold_wands - price_increase = 60 := by sorry
  
  exact 60

end kates_wand_cost_l795_79560


namespace tournament_handshakes_count_l795_79550

/-- The number of unique handshakes in a tournament with 4 teams of 2 players each,
    where each player shakes hands once with every other player except their partner. -/
def tournament_handshakes : ℕ :=
  let total_players : ℕ := 4 * 2
  let handshakes_per_player : ℕ := total_players - 2
  (total_players * handshakes_per_player) / 2

theorem tournament_handshakes_count : tournament_handshakes = 24 := by
  sorry

end tournament_handshakes_count_l795_79550
