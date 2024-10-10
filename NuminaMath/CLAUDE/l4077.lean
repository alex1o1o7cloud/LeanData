import Mathlib

namespace train_length_l4077_407775

/-- Calculates the length of a train given its speed, the speed of a bus moving in the opposite direction, and the time it takes for the train to pass the bus. -/
theorem train_length (train_speed : ℝ) (bus_speed : ℝ) (passing_time : ℝ) :
  train_speed = 90 →
  bus_speed = 60 →
  passing_time = 5.279577633789296 →
  let relative_speed := (train_speed + bus_speed) * (5 / 18)
  let train_length := relative_speed * passing_time
  train_length = 41.663147 := by
  sorry

end train_length_l4077_407775


namespace reciprocal_of_sum_l4077_407788

theorem reciprocal_of_sum : (1 / (1/3 + 1/4) : ℚ) = 12/7 := by sorry

end reciprocal_of_sum_l4077_407788


namespace domain_of_sqrt_plus_fraction_l4077_407724

theorem domain_of_sqrt_plus_fraction (x : ℝ) :
  (x + 3 ≥ 0 ∧ x + 2 ≠ 0) ↔ (x ≥ -3 ∧ x ≠ -2) := by sorry

end domain_of_sqrt_plus_fraction_l4077_407724


namespace parallelogram_rotation_volume_ratio_l4077_407746

/-- Given a parallelogram with adjacent sides a and b, the ratio of the volume of the cylinder
    formed by rotating the parallelogram around side a to the volume of the cylinder formed by
    rotating the parallelogram around side b is equal to a/b. -/
theorem parallelogram_rotation_volume_ratio
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (π * (a/2)^2 * b) / (π * (b/2)^2 * a) = a / b :=
sorry

end parallelogram_rotation_volume_ratio_l4077_407746


namespace remainder_problem_l4077_407774

theorem remainder_problem (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 34 = 20 := by
  sorry

end remainder_problem_l4077_407774


namespace figurine_cost_calculation_l4077_407768

def brand_a_price : ℝ := 65
def brand_b_price : ℝ := 75
def num_brand_a : ℕ := 3
def num_brand_b : ℕ := 2
def num_figurines : ℕ := 8
def figurine_total_cost : ℝ := brand_b_price + 40

theorem figurine_cost_calculation :
  (figurine_total_cost / num_figurines : ℝ) = 14.375 := by sorry

end figurine_cost_calculation_l4077_407768


namespace symmetry_about_point_period_four_l4077_407758

-- Define the function f
variable (f : ℝ → ℝ)

-- Statement ②
theorem symmetry_about_point (h : ∀ x, f (x + 1) + f (1 - x) = 0) :
  ∀ x, f (2 - x) = -f x :=
sorry

-- Statement ④
theorem period_four (h : ∀ x, f (1 + x) + f (x - 1) = 0) :
  ∀ x, f (x + 4) = f x :=
sorry

end symmetry_about_point_period_four_l4077_407758


namespace terms_before_negative_three_l4077_407717

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem terms_before_negative_three (a₁ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = 105 ∧ d = -6 →
  (∀ k < n, arithmetic_sequence a₁ d k > -3) ∧
  arithmetic_sequence a₁ d n = -3 →
  n - 1 = 18 := by
  sorry

end terms_before_negative_three_l4077_407717


namespace bob_cannot_win_and_prevent_alice_l4077_407750

def game_number : Set ℕ := {19, 20}
def start_number : Set ℕ := {9, 10}

theorem bob_cannot_win_and_prevent_alice (s : ℕ) (a : ℕ) :
  s ∈ start_number →
  a ∈ game_number →
  (∀ n : ℤ, s + 39 * n ≠ 2019) ∧
  (s = 9 → ∀ n : ℤ, s + 39 * n + a ≠ 2019) :=
by sorry

end bob_cannot_win_and_prevent_alice_l4077_407750


namespace ellipse_focal_length_specific_ellipse_focal_length_l4077_407727

/-- The focal length of an ellipse with equation x²/a² + y²/b² = 1 is 2c, where c² = a² - b² -/
theorem ellipse_focal_length (a b : ℝ) (h : 0 < b ∧ b < a) :
  let c := Real.sqrt (a^2 - b^2)
  let focal_length := 2 * c
  focal_length = 2 → a^2 = 2 ∧ b^2 = 1 := by sorry

/-- The focal length of the ellipse x²/2 + y² = 1 is 2 -/
theorem specific_ellipse_focal_length :
  let a := Real.sqrt 2
  let b := 1
  let c := Real.sqrt (a^2 - b^2)
  let focal_length := 2 * c
  focal_length = 2 := by sorry

end ellipse_focal_length_specific_ellipse_focal_length_l4077_407727


namespace bird_watching_ratio_l4077_407739

theorem bird_watching_ratio (cardinals robins blue_jays sparrows : ℕ) : 
  cardinals = 3 →
  robins = 4 * cardinals →
  sparrows = 3 * cardinals + 1 →
  cardinals + robins + blue_jays + sparrows = 31 →
  blue_jays = 2 * cardinals :=
by
  sorry

end bird_watching_ratio_l4077_407739


namespace average_marks_math_biology_l4077_407741

theorem average_marks_math_biology 
  (P C M B : ℕ) -- Marks in Physics, Chemistry, Mathematics, and Biology
  (h : P + C + M + B = P + C + 200) -- Total marks condition
  : (M + B) / 2 = 100 := by
sorry

end average_marks_math_biology_l4077_407741


namespace pasta_bins_l4077_407779

theorem pasta_bins (soup_bins vegetables_bins total_bins : ℝ)
  (h1 : soup_bins = 0.125)
  (h2 : vegetables_bins = 0.125)
  (h3 : total_bins = 0.75) :
  total_bins - (soup_bins + vegetables_bins) = 0.5 := by
  sorry

end pasta_bins_l4077_407779


namespace blanket_thickness_proof_l4077_407789

-- Define the initial thickness of the blanket
def initial_thickness : ℝ := 3

-- Define a function that calculates the thickness after n foldings
def thickness_after_foldings (n : ℕ) : ℝ :=
  initial_thickness * (2 ^ n)

-- Theorem statement
theorem blanket_thickness_proof :
  thickness_after_foldings 4 = 48 :=
by
  sorry


end blanket_thickness_proof_l4077_407789


namespace root_range_l4077_407719

theorem root_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁ ∈ Set.Icc (k - 1) (k + 1) ∧ 
    x₂ ∈ Set.Icc (k - 1) (k + 1) ∧
    Real.sqrt 2 * |x₁ - k| = k * Real.sqrt x₁ ∧
    Real.sqrt 2 * |x₂ - k| = k * Real.sqrt x₂) 
  ↔ 
  (0 < k ∧ k ≤ 1) :=
by sorry

end root_range_l4077_407719


namespace laran_weekly_profit_l4077_407733

/-- Calculates the profit for Laran's poster business over a 5-day school week --/
def calculate_profit (
  total_posters_per_day : ℕ)
  (large_posters_per_day : ℕ)
  (large_poster_price : ℚ)
  (large_poster_tax_rate : ℚ)
  (large_poster_cost : ℚ)
  (small_poster_price : ℚ)
  (small_poster_tax_rate : ℚ)
  (small_poster_cost : ℚ)
  (fixed_weekly_expense : ℚ)
  (days_per_week : ℕ) : ℚ :=
  sorry

/-- Theorem stating that Laran's weekly profit is $98.50 --/
theorem laran_weekly_profit :
  calculate_profit 5 2 10 (1/10) 5 6 (3/20) 3 20 5 = 197/2 :=
  sorry

end laran_weekly_profit_l4077_407733


namespace max_triangle_area_ellipse_circle_intersection_l4077_407778

/-- Given an ellipse E and a line x = t intersecting it, this theorem proves
    the maximum area of triangle ABC formed by the intersection of a circle
    with the y-axis, where the circle's diameter is the chord of the ellipse. -/
theorem max_triangle_area_ellipse_circle_intersection
  (a : ℝ) (t : ℝ) 
  (ha : a > Real.sqrt 3) 
  (ht : t > 0) 
  (he : Real.sqrt (a^2 - 3) / a = 1/2) :
  let E := {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / 3 = 1}
  let M := (t, Real.sqrt ((1 - t^2 / a^2) * 3))
  let N := (t, -Real.sqrt ((1 - t^2 / a^2) * 3))
  let C := {p : ℝ × ℝ | (p.1 - t)^2 + p.2^2 = ((M.2 - N.2) / 2)^2}
  let A := (0, Real.sqrt ((M.2 - N.2)^2 / 4 - t^2))
  let B := (0, -Real.sqrt ((M.2 - N.2)^2 / 4 - t^2))
  ∃ (tmax : ℝ), tmax > 0 ∧ 
    (∀ t' > 0, t' * Real.sqrt (12 - 7 * t'^2) / 2 ≤ tmax * Real.sqrt (12 - 7 * tmax^2) / 2) ∧
    tmax * Real.sqrt (12 - 7 * tmax^2) / 2 = 3 * Real.sqrt 7 / 7 :=
by sorry

end max_triangle_area_ellipse_circle_intersection_l4077_407778


namespace fraction_sum_equality_l4077_407745

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (36 - a) + b / (49 - b) + c / (81 - c) = 8) :
  6 / (36 - a) + 7 / (49 - b) + 9 / (81 - c) = 66 / 36 + 77 / 49 + 99 / 81 := by
sorry

end fraction_sum_equality_l4077_407745


namespace quadratic_sum_abc_l4077_407795

theorem quadratic_sum_abc : ∃ (a b c : ℝ), 
  (∀ x, 15 * x^2 + 75 * x + 375 = a * (x + b)^2 + c) ∧ 
  (a + b + c = 298.75) := by
  sorry

end quadratic_sum_abc_l4077_407795


namespace sixth_grade_count_l4077_407726

/-- The number of students in the sixth grade -/
def sixth_grade_students : ℕ := 108

/-- The total number of students in fifth and sixth grades -/
def total_students : ℕ := 200

/-- The number of fifth grade students who went to the celebration -/
def fifth_grade_celebration : ℕ := 11

/-- The percentage of sixth grade students who went to the celebration -/
def sixth_grade_celebration_percent : ℚ := 1/4

theorem sixth_grade_count : 
  sixth_grade_students = 108 ∧
  total_students = 200 ∧
  fifth_grade_celebration = 11 ∧
  sixth_grade_celebration_percent = 1/4 ∧
  (total_students - sixth_grade_students - fifth_grade_celebration) = 
  (sixth_grade_students * (1 - sixth_grade_celebration_percent)) :=
by sorry

end sixth_grade_count_l4077_407726


namespace sphere_volume_after_radius_increase_l4077_407738

theorem sphere_volume_after_radius_increase (initial_surface_area : ℝ) (radius_increase : ℝ) : 
  initial_surface_area = 256 * Real.pi → 
  radius_increase = 2 → 
  (4 / 3) * Real.pi * ((initial_surface_area / (4 * Real.pi))^(1/2) + radius_increase)^3 = (4000 / 3) * Real.pi := by
  sorry

end sphere_volume_after_radius_increase_l4077_407738


namespace not_all_odd_have_all_five_multiple_l4077_407734

theorem not_all_odd_have_all_five_multiple : ∃ n : ℕ, Odd n ∧ ∀ k : ℕ, ∃ d : ℕ, d ≠ 5 ∧ d ∈ (k * n).digits 10 := by
  sorry

end not_all_odd_have_all_five_multiple_l4077_407734


namespace avg_percent_grades_5_6_midville_easton_l4077_407705

/-- Represents a school with its total number of students and percentages for each grade --/
structure School where
  total_students : ℕ
  grade_k_percent : ℚ
  grade_1_percent : ℚ
  grade_2_percent : ℚ
  grade_3_percent : ℚ
  grade_4_percent : ℚ
  grade_5_percent : ℚ
  grade_6_percent : ℚ

def midville : School := {
  total_students := 150,
  grade_k_percent := 18/100,
  grade_1_percent := 14/100,
  grade_2_percent := 15/100,
  grade_3_percent := 12/100,
  grade_4_percent := 16/100,
  grade_5_percent := 12/100,
  grade_6_percent := 13/100
}

def easton : School := {
  total_students := 250,
  grade_k_percent := 10/100,
  grade_1_percent := 14/100,
  grade_2_percent := 17/100,
  grade_3_percent := 18/100,
  grade_4_percent := 13/100,
  grade_5_percent := 15/100,
  grade_6_percent := 13/100
}

/-- Calculates the average percentage of students in grades 5 and 6 for two schools combined --/
def avg_percent_grades_5_6 (s1 s2 : School) : ℚ :=
  let total_students := s1.total_students + s2.total_students
  let students_5_6 := s1.total_students * (s1.grade_5_percent + s1.grade_6_percent) +
                      s2.total_students * (s2.grade_5_percent + s2.grade_6_percent)
  students_5_6 / total_students

theorem avg_percent_grades_5_6_midville_easton :
  avg_percent_grades_5_6 midville easton = 2725/10000 := by
  sorry

end avg_percent_grades_5_6_midville_easton_l4077_407705


namespace katie_homework_problem_l4077_407794

/-- The number of math problems Katie finished on the bus ride home. -/
def finished_problems : ℕ := 5

/-- The number of math problems Katie had left to do. -/
def remaining_problems : ℕ := 4

/-- The total number of math problems Katie had for homework. -/
def total_problems : ℕ := finished_problems + remaining_problems

theorem katie_homework_problem :
  total_problems = 9 := by sorry

end katie_homework_problem_l4077_407794


namespace alternating_arithmetic_series_sum_l4077_407743

def arithmetic_series (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (fun i => a₁ - i * d)

def alternating_sign (n : ℕ) : List ℤ :=
  List.range n |>.map (fun i => if i % 2 == 0 then 1 else -1)

def series_sum (series : List ℤ) : ℤ :=
  series.sum

theorem alternating_arithmetic_series_sum :
  let a₁ : ℤ := 2005
  let d : ℤ := 10
  let n : ℕ := 200
  let series := List.zip (arithmetic_series a₁ d n) (alternating_sign n) |>.map (fun (x, y) => x * y)
  series_sum series = 1000 := by
  sorry

end alternating_arithmetic_series_sum_l4077_407743


namespace min_value_expression_equality_condition_l4077_407731

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  6 * a^3 + 9 * b^3 + 32 * c^3 + 1 / (4 * a * b * c) ≥ 6 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  6 * a^3 + 9 * b^3 + 32 * c^3 + 1 / (4 * a * b * c) = 6 ↔
  a = (1 : ℝ) / (6 : ℝ)^(1/3) ∧ b = (1 : ℝ) / (9 : ℝ)^(1/3) ∧ c = (1 : ℝ) / (32 : ℝ)^(1/3) :=
by sorry

end min_value_expression_equality_condition_l4077_407731


namespace portias_high_school_students_portias_high_school_students_proof_l4077_407710

theorem portias_high_school_students : ℕ → ℕ → Prop :=
  fun (portia_students lara_students : ℕ) =>
    (portia_students = 3 * lara_students) →
    (portia_students + lara_students = 2600) →
    (portia_students = 1950)

-- Proof
theorem portias_high_school_students_proof : 
  ∃ (portia_students lara_students : ℕ), 
    portias_high_school_students portia_students lara_students :=
by
  sorry

end portias_high_school_students_portias_high_school_students_proof_l4077_407710


namespace laundry_drying_time_l4077_407728

theorem laundry_drying_time 
  (num_loads : ℕ) 
  (wash_time_per_load : ℕ) 
  (total_laundry_time : ℕ) 
  (h1 : num_loads = 2) 
  (h2 : wash_time_per_load = 45) 
  (h3 : total_laundry_time = 165) : 
  total_laundry_time - (num_loads * wash_time_per_load) = 75 := by
sorry

end laundry_drying_time_l4077_407728


namespace supermarket_spending_l4077_407748

theorem supermarket_spending (total : ℝ) : 
  (1/2 : ℝ) * total + (1/3 : ℝ) * total + (1/10 : ℝ) * total + 10 = total → 
  total = 150 := by
sorry

end supermarket_spending_l4077_407748


namespace salary_change_l4077_407790

theorem salary_change (S : ℝ) : 
  S * (1 + 0.25) * (1 - 0.15) * (1 + 0.10) * (1 - 0.20) = S * 0.935 := by
  sorry

end salary_change_l4077_407790


namespace inequality_proof_l4077_407714

theorem inequality_proof (a b x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : x / a < y / b) :
  (1 / 2) * (x / a + y / b) > (x + y) / (a + b) := by
  sorry

end inequality_proof_l4077_407714


namespace hyperbola_t_squared_l4077_407772

/-- A hyperbola centered at the origin, opening horizontally -/
structure Hyperbola where
  /-- The equation of the hyperbola: x²/a² - y²/b² = 1 -/
  equation : ℝ → ℝ → Prop

/-- The hyperbola passes through the given points -/
def passes_through (h : Hyperbola) (x y : ℝ) : Prop :=
  h.equation x y

theorem hyperbola_t_squared (h : Hyperbola) :
  passes_through h 2 3 →
  passes_through h 3 0 →
  passes_through h t 5 →
  t^2 = 1854/81 :=
sorry

end hyperbola_t_squared_l4077_407772


namespace find_M_l4077_407722

theorem find_M : ∃ M : ℝ, (0.2 * M = 0.6 * 1500) ∧ (M = 4500) := by
  sorry

end find_M_l4077_407722


namespace geometric_sequence_product_l4077_407725

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 * a 1 - 10 * a 1 + 16 = 0) →
  (a 19 * a 19 - 10 * a 19 + 16 = 0) →
  a 8 * a 10 * a 12 = 64 := by
  sorry

end geometric_sequence_product_l4077_407725


namespace least_integer_greater_than_sqrt_500_l4077_407723

theorem least_integer_greater_than_sqrt_500 : 
  (∀ n : ℕ, n ≤ 22 → n ^ 2 ≤ 500) ∧ 23 ^ 2 > 500 :=
by sorry

end least_integer_greater_than_sqrt_500_l4077_407723


namespace inequality_equivalence_l4077_407762

theorem inequality_equivalence :
  ∀ a : ℝ, a > 0 →
  ((∀ t₁ t₂ t₃ t₄ : ℝ, t₁ > 0 → t₂ > 0 → t₃ > 0 → t₄ > 0 → 
    t₁ * t₂ * t₃ * t₄ = a^4 →
    (1 / Real.sqrt (1 + t₁)) + (1 / Real.sqrt (1 + t₂)) + 
    (1 / Real.sqrt (1 + t₃)) + (1 / Real.sqrt (1 + t₄)) ≤ 
    4 / Real.sqrt (1 + a))
  ↔ 
  (0 < a ∧ a ≤ 7/9)) := by
sorry

end inequality_equivalence_l4077_407762


namespace sandwich_meal_combinations_l4077_407753

theorem sandwich_meal_combinations : 
  ∃! n : ℕ, n = (Finset.filter 
    (λ (pair : ℕ × ℕ) => 5 * pair.1 + 7 * pair.2 = 90) 
    (Finset.product (Finset.range 19) (Finset.range 13))).card := by
  sorry

end sandwich_meal_combinations_l4077_407753


namespace tangent_line_slope_l4077_407700

open Real

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x * (-2)

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 2 * x + a * (-2)

-- Theorem statement
theorem tangent_line_slope (a : ℝ) :
  f' a 1 = -2 → a = 2 := by
  sorry

end tangent_line_slope_l4077_407700


namespace max_x_value_l4077_407713

theorem max_x_value (x : ℝ) : 
  ((5 * x - 20) / (4 * x - 5))^2 + (5 * x - 20) / (4 * x - 5) = 18 → x ≤ 50 / 29 := by
  sorry

end max_x_value_l4077_407713


namespace inequality_solution_l4077_407792

theorem inequality_solution (x : ℝ) : 3 - 2 / (3 * x + 4) < 5 ↔ x < -4/3 ∨ x > -5/3 :=
by sorry

end inequality_solution_l4077_407792


namespace infinitely_many_triples_l4077_407799

theorem infinitely_many_triples :
  ∀ n : ℕ, ∃ (a b p : ℕ),
    Prime p ∧
    0 < a ∧ a ≤ b ∧ b < p ∧
    (p^5 ∣ (a + b)^p - a^p - b^p) ∧
    p > n :=
by sorry

end infinitely_many_triples_l4077_407799


namespace certain_number_is_four_l4077_407707

theorem certain_number_is_four (k : ℝ) (certain_number : ℝ) 
  (h1 : 64 / k = certain_number) 
  (h2 : k = 16) : 
  certain_number = 4 := by
sorry

end certain_number_is_four_l4077_407707


namespace four_Y_three_equals_negative_twentythree_l4077_407754

-- Define the Y operation
def Y (a b : ℤ) : ℤ := a^2 - 2 * a * b * 2 + b^2

-- Theorem statement
theorem four_Y_three_equals_negative_twentythree :
  Y 4 3 = -23 := by
  sorry

end four_Y_three_equals_negative_twentythree_l4077_407754


namespace complement_of_A_l4077_407701

theorem complement_of_A (U A : Set ℕ) : 
  U = {1, 2, 3, 4, 5, 6} → 
  A = {3, 4, 5} → 
  Aᶜ = {1, 2, 6} := by
sorry

end complement_of_A_l4077_407701


namespace orange_distribution_l4077_407752

theorem orange_distribution (total_oranges : ℕ) (bad_oranges : ℕ) (difference : ℕ) :
  total_oranges = 108 →
  bad_oranges = 36 →
  difference = 3 →
  (total_oranges : ℚ) / (total_oranges / difference - bad_oranges / difference) - 
  ((total_oranges - bad_oranges) : ℚ) / (total_oranges / difference - bad_oranges / difference) = difference →
  total_oranges / difference - bad_oranges / difference = 12 :=
by sorry

end orange_distribution_l4077_407752


namespace inequality_proof_l4077_407718

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 + c^2 = 14) : 
  a^5 + (1/8)*b^5 + (1/27)*c^5 ≥ 14 := by
  sorry

end inequality_proof_l4077_407718


namespace unique_stamp_arrangements_l4077_407787

/-- Represents the number of stamps of each denomination -/
def stamp_counts : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- Represents the value of each stamp denomination -/
def stamp_values : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- A type to represent a stamp arrangement -/
structure StampArrangement where
  stamps : List Nat
  sum_to_ten : (stamps.sum = 10)

/-- Function to count unique arrangements -/
def count_unique_arrangements (stamps : List Nat) (values : List Nat) : Nat :=
  sorry

/-- The main theorem stating that there are 88 unique arrangements -/
theorem unique_stamp_arrangements :
  count_unique_arrangements stamp_counts stamp_values = 88 := by
  sorry

end unique_stamp_arrangements_l4077_407787


namespace two_numbers_problem_l4077_407742

theorem two_numbers_problem (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : 
  a * b = 875 ∧ a^2 + b^2 = 1850 := by
sorry

end two_numbers_problem_l4077_407742


namespace largest_non_composite_sum_l4077_407761

def is_composite (n : ℕ) : Prop :=
  ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

theorem largest_non_composite_sum : 
  (∀ n : ℕ, n > 11 → ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) ∧
  ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ 11 = a + b) :=
sorry

end largest_non_composite_sum_l4077_407761


namespace jovanas_shells_l4077_407773

/-- The total amount of shells Jovana has after her friends add to her collection -/
def total_shells (initial : ℕ) (friend1 : ℕ) (friend2 : ℕ) : ℕ :=
  initial + friend1 + friend2

/-- Theorem stating that Jovana's total shells equal 37 pounds -/
theorem jovanas_shells :
  total_shells 5 15 17 = 37 := by
  sorry

end jovanas_shells_l4077_407773


namespace rhombus_perimeter_l4077_407766

/-- Given a rhombus with diagonals of 14 inches and 48 inches, its perimeter is 100 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 48) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 100 := by
  sorry

end rhombus_perimeter_l4077_407766


namespace parabola_tangent_min_area_l4077_407703

/-- The parabola equation -/
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

/-- The point M -/
def M (y₀ : ℝ) : ℝ × ℝ := (-1, y₀)

/-- The area of triangle MAB -/
noncomputable def triangleArea (p : ℝ) (y₀ : ℝ) : ℝ :=
  2 * Real.sqrt (y₀^2 + 2*p)

/-- The main theorem -/
theorem parabola_tangent_min_area (p : ℝ) :
  p > 0 →
  (∀ y₀ : ℝ, triangleArea p y₀ ≥ 4) →
  (∃ y₀ : ℝ, triangleArea p y₀ = 4) →
  p = 2 := by sorry

end parabola_tangent_min_area_l4077_407703


namespace sum_of_angles_in_figure_l4077_407782

-- Define the angles
def angle_A : ℝ := 34
def angle_B : ℝ := 80
def angle_C : ℝ := 24

-- Define x and y as real numbers (measures of angles)
variable (x y : ℝ)

-- Define the theorem
theorem sum_of_angles_in_figure (h1 : 0 ≤ x) (h2 : 0 ≤ y) : x + y = 132 := by
  sorry


end sum_of_angles_in_figure_l4077_407782


namespace equivalent_equations_product_l4077_407711

/-- Given that the equation a^8xy - 2a^7y - 3a^6x = 2a^5(b^5 - 2) is equivalent to 
    (a^m*x - 2a^n)(a^p*y - 3a^3) = 2a^5*b^5 for some integers m, n, and p, 
    prove that m*n*p = 60 -/
theorem equivalent_equations_product (a b x y : ℝ) (m n p : ℤ) 
  (h1 : a^8*x*y - 2*a^7*y - 3*a^6*x = 2*a^5*(b^5 - 2))
  (h2 : (a^m*x - 2*a^n)*(a^p*y - 3*a^3) = 2*a^5*b^5) :
  m * n * p = 60 := by
  sorry

end equivalent_equations_product_l4077_407711


namespace original_trees_eq_sum_l4077_407791

/-- The number of trees Haley originally grew in her backyard -/
def original_trees : ℕ := 20

/-- The number of trees left after the typhoon -/
def trees_left : ℕ := 4

/-- The number of trees that died in the typhoon -/
def trees_died : ℕ := 16

/-- Theorem stating that the original number of trees equals the sum of trees left and trees that died -/
theorem original_trees_eq_sum : original_trees = trees_left + trees_died := by
  sorry

end original_trees_eq_sum_l4077_407791


namespace greatest_integer_c_for_domain_all_reals_l4077_407755

theorem greatest_integer_c_for_domain_all_reals : 
  (∃ c : ℤ, (∀ x : ℝ, (x^2 + c*x + 10 ≠ 0)) ∧ 
   (∀ d : ℤ, d > c → ∃ x : ℝ, x^2 + d*x + 10 = 0)) → 
  (∃ c : ℤ, c = 6 ∧ 
   (∀ x : ℝ, (x^2 + c*x + 10 ≠ 0)) ∧ 
   (∀ d : ℤ, d > c → ∃ x : ℝ, x^2 + d*x + 10 = 0)) :=
by sorry

end greatest_integer_c_for_domain_all_reals_l4077_407755


namespace car_lot_power_windows_l4077_407765

theorem car_lot_power_windows 
  (total : ℕ) 
  (air_bags : ℕ) 
  (both : ℕ) 
  (neither : ℕ) 
  (h1 : total = 65)
  (h2 : air_bags = 45)
  (h3 : both = 12)
  (h4 : neither = 2) :
  ∃ power_windows : ℕ, power_windows = 30 ∧ 
    total = air_bags + power_windows - both + neither :=
by sorry

end car_lot_power_windows_l4077_407765


namespace M_eq_real_l4077_407759

/-- The set of complex numbers Z satisfying (Z-1)^2 = |Z-1|^2 -/
def M : Set ℂ := {Z | (Z - 1)^2 = Complex.abs (Z - 1)^2}

/-- Theorem stating that M is equal to the set of real numbers -/
theorem M_eq_real : M = {Z : ℂ | Z.im = 0} := by sorry

end M_eq_real_l4077_407759


namespace quadratic_inequality_range_l4077_407747

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := by
  sorry

end quadratic_inequality_range_l4077_407747


namespace gcf_of_360_and_150_l4077_407769

theorem gcf_of_360_and_150 : Nat.gcd 360 150 = 30 := by
  sorry

end gcf_of_360_and_150_l4077_407769


namespace rhombus_perimeter_l4077_407780

/-- A rhombus with diagonals of 6 and 8 units has a perimeter of 20 units. -/
theorem rhombus_perimeter (d₁ d₂ : ℝ) (h₁ : d₁ = 6) (h₂ : d₂ = 8) :
  let side := Real.sqrt ((d₁/2)^2 + (d₂/2)^2)
  4 * side = 20 :=
by sorry

end rhombus_perimeter_l4077_407780


namespace min_stamps_for_60_cents_l4077_407781

theorem min_stamps_for_60_cents : ∃ (s t : ℕ), 
  5 * s + 6 * t = 60 ∧ 
  s + t = 11 ∧
  ∀ (s' t' : ℕ), 5 * s' + 6 * t' = 60 → s + t ≤ s' + t' := by
  sorry

end min_stamps_for_60_cents_l4077_407781


namespace product_simplification_l4077_407767

theorem product_simplification (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end product_simplification_l4077_407767


namespace cubic_function_property_l4077_407763

/-- Given a cubic function y = ax³ + bx² + cx + d, if (2, y₁) and (-2, y₂) lie on its graph
    and y₁ - y₂ = 12, then c = 3 - 4a. -/
theorem cubic_function_property (a b c d y₁ y₂ : ℝ) :
  y₁ = 8*a + 4*b + 2*c + d →
  y₂ = -8*a + 4*b - 2*c + d →
  y₁ - y₂ = 12 →
  c = 3 - 4*a :=
by sorry

end cubic_function_property_l4077_407763


namespace b_work_days_l4077_407770

/-- Represents the number of days it takes for a person to complete the work alone -/
structure WorkDays where
  days : ℕ

/-- Represents the rate at which a person completes the work per day -/
def workRate (w : WorkDays) : ℚ :=
  1 / w.days

theorem b_work_days (total_payment : ℕ) (a_work : WorkDays) (abc_work : WorkDays) (c_share : ℕ) :
  total_payment = 1200 →
  a_work.days = 6 →
  abc_work.days = 3 →
  c_share = 150 →
  ∃ b_work : WorkDays,
    b_work.days = 24 ∧
    workRate a_work + workRate b_work + (c_share : ℚ) / total_payment = workRate abc_work :=
by sorry

end b_work_days_l4077_407770


namespace intersection_of_lines_l4077_407721

/-- Given two lines m and n that intersect at (1, 6), where
    m has equation y = 4x + 2 and n has equation y = kx + 3,
    prove that k = 3. -/
theorem intersection_of_lines (k : ℝ) : 
  (∀ x y : ℝ, y = 4*x + 2 → (x = 1 ∧ y = 6)) →  -- line m passes through (1, 6)
  (∀ x y : ℝ, y = k*x + 3 → (x = 1 ∧ y = 6)) →  -- line n passes through (1, 6)
  k = 3 := by
  sorry

end intersection_of_lines_l4077_407721


namespace partial_fraction_decomposition_constant_l4077_407720

theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 5 ∧ x ≠ -3 → 
    1 / (x^3 - 7*x^2 + 11*x + 45) = A / (x - 5) + B / (x + 3) + C / (x + 3)^2) →
  B = -1 / 64 := by
sorry

end partial_fraction_decomposition_constant_l4077_407720


namespace tan_floor_eq_two_cos_sq_iff_pi_quarter_plus_two_pi_k_l4077_407704

/-- The floor function, which returns the greatest integer less than or equal to a real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Theorem stating that the equation [tan x] = 2 cos^2 x is satisfied if and only if x = π/4 + 2kπ, where k is an integer -/
theorem tan_floor_eq_two_cos_sq_iff_pi_quarter_plus_two_pi_k (x : ℝ) :
  floor (Real.tan x) = (2 : ℝ) * (Real.cos x)^2 ↔ ∃ k : ℤ, x = π/4 + 2*k*π :=
sorry

end tan_floor_eq_two_cos_sq_iff_pi_quarter_plus_two_pi_k_l4077_407704


namespace purely_imaginary_complex_number_l4077_407751

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) = (a^2 - 3*a + 2) + Complex.I * (a - 1)) → a = 2 := by
  sorry

end purely_imaginary_complex_number_l4077_407751


namespace or_necessary_not_sufficient_for_and_l4077_407716

theorem or_necessary_not_sufficient_for_and (p q : Prop) :
  (∀ (p q : Prop), p ∧ q → p ∨ q) ∧
  (∃ (p q : Prop), p ∨ q ∧ ¬(p ∧ q)) :=
by sorry

end or_necessary_not_sufficient_for_and_l4077_407716


namespace coin_flip_probability_l4077_407729

theorem coin_flip_probability (n m k l : ℕ) (h1 : n = 11) (h2 : m = 5) (h3 : k = 7) (h4 : l = 3) :
  let p := (1 : ℚ) / 2
  let total_success_prob := (n.choose k : ℚ) * p^k * (1 - p)^(n - k)
  let monday_success_prob := (m.choose l : ℚ) * p^l * (1 - p)^(m - l)
  let tuesday_success_prob := ((n - m).choose (k - l) : ℚ) * p^(k - l) * (1 - p)^(n - m - (k - l))
  (monday_success_prob * tuesday_success_prob) / total_success_prob = 5 / 11 :=
by sorry

end coin_flip_probability_l4077_407729


namespace midpoint_trajectory_l4077_407712

/-- The trajectory of the midpoint of a line segment connecting a point on a parabola and a fixed point -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ (P : ℝ × ℝ), 
    P.2 = 2 * P.1^2 + 1 ∧ 
    P.1 = 2 * x ∧ 
    P.2 = 2 * y + 1) →
  y = 4 * x^2 := by
  sorry

end midpoint_trajectory_l4077_407712


namespace clover_field_count_l4077_407798

theorem clover_field_count : ∀ (total : ℕ),
  (total : ℝ) * (20 / 100) * (25 / 100) = 25 →
  total = 500 := by
  sorry

end clover_field_count_l4077_407798


namespace imaginary_part_of_x_l4077_407732

theorem imaginary_part_of_x (x : ℂ) (h : (3 + 4*I)*x = Complex.abs (4 + 3*I)) : 
  x.im = -4/5 := by
  sorry

end imaginary_part_of_x_l4077_407732


namespace files_deleted_l4077_407735

theorem files_deleted (initial_music : ℕ) (initial_video : ℕ) (remaining : ℕ) : 
  initial_music = 26 → initial_video = 36 → remaining = 14 →
  initial_music + initial_video - remaining = 48 := by
  sorry

end files_deleted_l4077_407735


namespace lattice_points_sum_l4077_407776

/-- Number of lattice points in a plane region -/
noncomputable def N (D : Set (ℝ × ℝ)) : ℕ := sorry

/-- Region A -/
def A : Set (ℝ × ℝ) := {(x, y) | y = x^2 ∧ x ≤ 0 ∧ x ≥ -10 ∧ y ≤ 1}

/-- Region B -/
def B : Set (ℝ × ℝ) := {(x, y) | y = x^2 ∧ x ≥ 0 ∧ x ≤ 1 ∧ y ≤ 100}

/-- Theorem: The sum of lattice points in the union and intersection of A and B is 1010 -/
theorem lattice_points_sum : N (A ∪ B) + N (A ∩ B) = 1010 := by sorry

end lattice_points_sum_l4077_407776


namespace hueys_pizza_size_proof_l4077_407797

theorem hueys_pizza_size_proof (small_side : ℝ) (small_cost : ℝ) (large_cost : ℝ) 
  (individual_budget : ℝ) (extra_area : ℝ) :
  small_side = 12 →
  small_cost = 10 →
  large_cost = 20 →
  individual_budget = 30 →
  extra_area = 36 →
  ∃ (large_side : ℝ),
    large_side = 10 * Real.sqrt 3 ∧
    3 * (large_side ^ 2) = 2 * (3 * small_side ^ 2) + extra_area :=
by
  sorry

end hueys_pizza_size_proof_l4077_407797


namespace lines_coplanar_iff_k_eq_three_halves_l4077_407740

/-- Parametric equation of the first line -/
def line1 (r : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (2 + r, -1 - 2*k*r, 3 + k*r)

/-- Parametric equation of the second line -/
def line2 (t : ℝ) : ℝ × ℝ × ℝ := (1 + 3*t, 2 - t, 1 + 2*t)

/-- Direction vector of the first line -/
def dir1 (k : ℝ) : ℝ × ℝ × ℝ := (1, -2*k, k)

/-- Direction vector of the second line -/
def dir2 : ℝ × ℝ × ℝ := (3, -1, 2)

/-- Two lines are coplanar if their direction vectors are proportional -/
def coplanar (k : ℝ) : Prop :=
  ∃ (c : ℝ), c ≠ 0 ∧ dir1 k = (c • dir2)

theorem lines_coplanar_iff_k_eq_three_halves :
  ∃ (k : ℝ), coplanar k ↔ k = 3/2 :=
sorry

end lines_coplanar_iff_k_eq_three_halves_l4077_407740


namespace expected_weekly_rainfall_l4077_407706

/-- The number of days in the week --/
def days : ℕ := 7

/-- The probability of sun (0 inches of rain) --/
def prob_sun : ℝ := 0.3

/-- The probability of 3 inches of rain --/
def prob_rain_3 : ℝ := 0.4

/-- The probability of 7 inches of rain --/
def prob_rain_7 : ℝ := 0.3

/-- The amount of rain in inches for the sunny scenario --/
def rain_sun : ℝ := 0

/-- The amount of rain in inches for the 3-inch rain scenario --/
def rain_3 : ℝ := 3

/-- The amount of rain in inches for the 7-inch rain scenario --/
def rain_7 : ℝ := 7

/-- The expected value of rainfall for a single day --/
def expected_daily_rainfall : ℝ :=
  prob_sun * rain_sun + prob_rain_3 * rain_3 + prob_rain_7 * rain_7

theorem expected_weekly_rainfall :
  days * expected_daily_rainfall = 23.1 := by
  sorry

end expected_weekly_rainfall_l4077_407706


namespace sufficient_not_necessary_condition_l4077_407702

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → abs a > abs b) ∧
  (∃ a b : ℝ, abs a > abs b ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end sufficient_not_necessary_condition_l4077_407702


namespace difference_number_and_fraction_difference_150_and_its_three_fifths_l4077_407784

theorem difference_number_and_fraction (n : ℚ) : n - (3 / 5) * n = (2 / 5) * n := by sorry

theorem difference_150_and_its_three_fifths : 150 - (3 / 5) * 150 = 60 := by sorry

end difference_number_and_fraction_difference_150_and_its_three_fifths_l4077_407784


namespace polynomial_factorization_l4077_407756

theorem polynomial_factorization :
  ∀ x : ℝ, (x^2 + 2*x + 1) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 7*x + 1) * (x^2 + 3*x + 7) := by
  sorry

end polynomial_factorization_l4077_407756


namespace third_week_cases_new_york_coronavirus_cases_l4077_407708

/-- Proves the number of new coronavirus cases in the third week --/
theorem third_week_cases (first_week : ℕ) (total_cases : ℕ) : ℕ :=
  let second_week := first_week / 2
  let first_two_weeks := first_week + second_week
  total_cases - first_two_weeks

/-- The main theorem that proves the number of new cases in the third week is 2000 --/
theorem new_york_coronavirus_cases : third_week_cases 5000 9500 = 2000 := by
  sorry

end third_week_cases_new_york_coronavirus_cases_l4077_407708


namespace sequence_sum_formula_l4077_407730

theorem sequence_sum_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = (2/3) * (a n) + 1/3) →
  (∀ n : ℕ, a n = (-2)^(n-1)) :=
sorry

end sequence_sum_formula_l4077_407730


namespace value_of_y_l4077_407757

theorem value_of_y (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 21) : y = 84 := by
  sorry

end value_of_y_l4077_407757


namespace dataset_groups_l4077_407783

/-- Calculate the number of groups for a dataset given its maximum value, minimum value, and class interval. -/
def number_of_groups (max_value min_value class_interval : ℕ) : ℕ :=
  (max_value - min_value) / class_interval + 1

/-- Theorem: For a dataset with maximum value 140, minimum value 50, and class interval 10, 
    the number of groups is 10. -/
theorem dataset_groups :
  number_of_groups 140 50 10 = 10 := by
  sorry

end dataset_groups_l4077_407783


namespace sock_pair_count_l4077_407760

/-- The number of ways to choose a pair of socks of different colors -/
def differentColorPairs (white brown blue red : ℕ) : ℕ :=
  white * brown + white * blue + white * red +
  brown * blue + brown * red +
  blue * red

/-- Theorem: There are 93 ways to choose a pair of socks of different colors
    from a drawer containing 5 white, 5 brown, 4 blue, and 2 red socks -/
theorem sock_pair_count :
  differentColorPairs 5 5 4 2 = 93 := by
  sorry

end sock_pair_count_l4077_407760


namespace units_digit_of_M_M_10_l4077_407744

-- Define the sequence M_n
def M : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | n + 2 => M (n + 1) + M n

-- Function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_M_M_10 : unitsDigit (M (M 10)) = 1 := by sorry

end units_digit_of_M_M_10_l4077_407744


namespace integer_solutions_cubic_equation_l4077_407771

theorem integer_solutions_cubic_equation :
  ∀ x y : ℤ, x^3 - y^3 = 2*x*y + 8 ↔ (x = 0 ∧ y = -2) ∨ (x = 2 ∧ y = 0) :=
by sorry

end integer_solutions_cubic_equation_l4077_407771


namespace opposite_values_theorem_l4077_407785

theorem opposite_values_theorem (a b : ℝ) 
  (h : |a - 2| + (b + 1)^2 = 0) : 
  b^a = 1 ∧ a^3 + b^15 = 7 := by sorry

end opposite_values_theorem_l4077_407785


namespace probability_different_colors_is_two_thirds_l4077_407709

def total_balls : ℕ := 4
def red_balls : ℕ := 2
def white_balls : ℕ := 2
def balls_drawn : ℕ := 2

def total_ways : ℕ := Nat.choose total_balls balls_drawn
def different_color_ways : ℕ := red_balls * white_balls

def probability_different_colors : ℚ := different_color_ways / total_ways

theorem probability_different_colors_is_two_thirds :
  probability_different_colors = 2 / 3 := by
  sorry

end probability_different_colors_is_two_thirds_l4077_407709


namespace x_positive_sufficient_not_necessary_l4077_407777

theorem x_positive_sufficient_not_necessary :
  (∀ x : ℝ, x > 0 → |x - 1| - |x| ≤ 1) ∧
  (∃ x : ℝ, x ≤ 0 ∧ |x - 1| - |x| ≤ 1) :=
by sorry

end x_positive_sufficient_not_necessary_l4077_407777


namespace factorization_equality_l4077_407793

theorem factorization_equality (x : ℝ) : 
  (x^4 + x^2 - 4) * (x^4 + x^2 + 3) + 10 = 
  (x^2 + x + 1) * (x^2 - x + 1) * (x^2 + 2) * (x + 1) * (x - 1) := by
  sorry

end factorization_equality_l4077_407793


namespace least_number_of_grapes_l4077_407764

theorem least_number_of_grapes : ∃ n : ℕ, n > 0 ∧ 
  n % 19 = 1 ∧ n % 23 = 1 ∧ n % 29 = 1 ∧ 
  ∀ m : ℕ, m > 0 → m % 19 = 1 → m % 23 = 1 → m % 29 = 1 → n ≤ m :=
by
  use 12209
  sorry

end least_number_of_grapes_l4077_407764


namespace fake_coin_strategy_exists_find_fake_coin_correct_l4077_407715

/-- Represents a strategy to find a fake coin among 2^(2^k) coins using dogs -/
structure FakeCoinStrategy (k : ℕ) :=
  (num_tests : ℕ)
  (find_fake_coin : Unit → ℕ)

/-- Theorem stating the existence of a strategy to find the fake coin -/
theorem fake_coin_strategy_exists (k : ℕ) :
  ∃ (strategy : FakeCoinStrategy k),
    strategy.num_tests ≤ 2^k + k + 2 ∧
    strategy.find_fake_coin () < 2^(2^k) :=
by sorry

/-- Function to perform a test with selected coins and a dog -/
def perform_test (selected_coins : Finset ℕ) (dog : ℕ) : Bool :=
sorry

/-- Function to select a dog for testing -/
def select_dog : ℕ :=
sorry

/-- Function to implement the strategy and find the fake coin -/
def find_fake_coin (k : ℕ) : ℕ :=
sorry

/-- Theorem proving the correctness of the find_fake_coin function -/
theorem find_fake_coin_correct (k : ℕ) :
  ∃ (num_tests : ℕ),
    num_tests ≤ 2^k + k + 2 ∧
    find_fake_coin k < 2^(2^k) :=
by sorry

end fake_coin_strategy_exists_find_fake_coin_correct_l4077_407715


namespace geometric_sequence_product_l4077_407749

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence where a_4 = 3, a_2 * a_6 = 9 -/
theorem geometric_sequence_product (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) (h_a4 : a 4 = 3) : 
  a 2 * a 6 = 9 := by
  sorry

end geometric_sequence_product_l4077_407749


namespace determine_c_absolute_value_l4077_407736

/-- The polynomial g(x) = ax^4 + bx^3 + cx^2 + bx + a -/
def g (a b c : ℤ) (x : ℂ) : ℂ := a * x^4 + b * x^3 + c * x^2 + b * x + a

/-- The theorem statement -/
theorem determine_c_absolute_value (a b c : ℤ) : 
  g a b c (3 + Complex.I) = 0 ∧ 
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 111 := by
  sorry

end determine_c_absolute_value_l4077_407736


namespace range_of_z_l4077_407796

theorem range_of_z (x y z : ℝ) 
  (hx : -1 ≤ x ∧ x ≤ 2) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : z = 2*x - y) : 
  -3 ≤ z ∧ z ≤ 4 := by
sorry

end range_of_z_l4077_407796


namespace equation_solution_l4077_407786

theorem equation_solution : 
  ∃! x : ℝ, 5 + 3.5 * x = 2.5 * x - 25 ∧ x = -30 := by sorry

end equation_solution_l4077_407786


namespace unique_m_solution_l4077_407737

theorem unique_m_solution : ∃! m : ℝ, (1 - m)^4 + 6*(1 - m)^3 + 8*(1 - m) = 16*m^2 := by
  sorry

end unique_m_solution_l4077_407737
