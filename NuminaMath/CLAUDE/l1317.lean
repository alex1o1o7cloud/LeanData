import Mathlib

namespace intersection_M_N_l1317_131758

def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x ≤ 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end intersection_M_N_l1317_131758


namespace max_playground_area_l1317_131788

/-- Represents a rectangular playground --/
structure Playground where
  length : ℝ
  width : ℝ

/-- The perimeter of the playground is 400 feet --/
def perimeterConstraint (p : Playground) : Prop :=
  2 * p.length + 2 * p.width = 400

/-- The length of the playground is at least 100 feet --/
def lengthConstraint (p : Playground) : Prop :=
  p.length ≥ 100

/-- The width of the playground is at least 50 feet --/
def widthConstraint (p : Playground) : Prop :=
  p.width ≥ 50

/-- The area of the playground --/
def area (p : Playground) : ℝ :=
  p.length * p.width

/-- The maximum area of the playground satisfying all constraints is 10000 square feet --/
theorem max_playground_area :
  ∃ (p : Playground),
    perimeterConstraint p ∧
    lengthConstraint p ∧
    widthConstraint p ∧
    area p = 10000 ∧
    ∀ (q : Playground),
      perimeterConstraint q →
      lengthConstraint q →
      widthConstraint q →
      area q ≤ area p :=
by
  sorry


end max_playground_area_l1317_131788


namespace smallest_x_with_given_remainders_l1317_131768

theorem smallest_x_with_given_remainders :
  ∃ x : ℕ,
    x > 0 ∧
    x % 6 = 5 ∧
    x % 7 = 6 ∧
    x % 8 = 7 ∧
    ∀ y : ℕ, y > 0 → y % 6 = 5 → y % 7 = 6 → y % 8 = 7 → x ≤ y ∧
    x = 167 :=
by sorry

end smallest_x_with_given_remainders_l1317_131768


namespace election_invalid_votes_percentage_l1317_131798

theorem election_invalid_votes_percentage 
  (total_votes : ℕ) 
  (votes_B : ℕ) 
  (h_total : total_votes = 9720)
  (h_B : votes_B = 3159)
  (h_difference : ∃ (votes_A : ℕ), votes_A = votes_B + (15 * total_votes) / 100) :
  (total_votes - (votes_B + (votes_B + (15 * total_votes) / 100))) * 100 / total_votes = 20 := by
sorry

end election_invalid_votes_percentage_l1317_131798


namespace pine_cones_on_roof_l1317_131708

/-- The number of pine trees in Alan's backyard -/
def num_trees : ℕ := 8

/-- The number of pine cones dropped by each tree -/
def cones_per_tree : ℕ := 200

/-- The weight of each pine cone in ounces -/
def cone_weight : ℕ := 4

/-- The total weight of pine cones on Alan's roof in ounces -/
def roof_weight : ℕ := 1920

/-- The percentage of pine cones that fall on Alan's roof -/
def roof_percentage : ℚ := 30 / 100

theorem pine_cones_on_roof :
  (roof_weight / cone_weight) / (num_trees * cones_per_tree) = roof_percentage := by
  sorry

end pine_cones_on_roof_l1317_131708


namespace remainder_11_pow_2023_mod_7_l1317_131754

theorem remainder_11_pow_2023_mod_7 : 11^2023 % 7 = 4 := by
  sorry

end remainder_11_pow_2023_mod_7_l1317_131754


namespace all_digits_appear_as_cube_units_l1317_131725

theorem all_digits_appear_as_cube_units : ∀ d : Nat, d < 10 → ∃ n : Nat, n^3 % 10 = d := by
  sorry

end all_digits_appear_as_cube_units_l1317_131725


namespace infinitely_many_real_roots_l1317_131700

theorem infinitely_many_real_roots : Set.Infinite {x : ℝ | ∃ y : ℝ, y^2 = -(x+1)^3} := by
  sorry

end infinitely_many_real_roots_l1317_131700


namespace geometric_sequence_product_l1317_131701

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence where a_4 = 5 and a_8 = 6, a_2 * a_10 = 30 -/
theorem geometric_sequence_product (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) 
    (h_a4 : a 4 = 5) 
    (h_a8 : a 8 = 6) : 
  a 2 * a 10 = 30 := by
  sorry


end geometric_sequence_product_l1317_131701


namespace complex_magnitude_problem_l1317_131756

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) :
  Complex.abs z = 1 := by
  sorry

end complex_magnitude_problem_l1317_131756


namespace ball_selection_count_l1317_131728

def num_colors : ℕ := 4
def balls_per_color : ℕ := 6
def balls_to_select : ℕ := 3

def valid_number_combinations : List (List ℕ) :=
  [[1, 3, 5], [1, 3, 6], [1, 4, 6], [2, 4, 6]]

theorem ball_selection_count :
  (num_colors.choose balls_to_select) *
  (valid_number_combinations.length) *
  (balls_to_select.factorial) = 96 := by
sorry

end ball_selection_count_l1317_131728


namespace min_value_quadratic_l1317_131785

theorem min_value_quadratic :
  ∃ (min_z : ℝ), min_z = -44 ∧ ∀ (x : ℝ), x^2 + 16*x + 20 ≥ min_z :=
by sorry

end min_value_quadratic_l1317_131785


namespace triangle_height_ratio_l1317_131753

theorem triangle_height_ratio (a b c : ℝ) (ha hb hc : a > 0 ∧ b > 0 ∧ c > 0) 
  (side_ratio : a / 3 = b / 4 ∧ b / 4 = c / 5) :
  ∃ (h₁ h₂ h₃ : ℝ), h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧ 
    (a * h₁ = b * h₂) ∧ (b * h₂ = c * h₃) ∧
    h₁ / 20 = h₂ / 15 ∧ h₂ / 15 = h₃ / 12 :=
sorry

end triangle_height_ratio_l1317_131753


namespace adult_ticket_price_l1317_131715

theorem adult_ticket_price 
  (total_tickets : ℕ) 
  (total_profit : ℕ) 
  (kid_tickets : ℕ) 
  (kid_price : ℕ) :
  total_tickets = 175 →
  total_profit = 750 →
  kid_tickets = 75 →
  kid_price = 2 →
  (total_tickets - kid_tickets) * 
    ((total_profit - kid_tickets * kid_price) / (total_tickets - kid_tickets)) = 600 :=
by sorry

end adult_ticket_price_l1317_131715


namespace geometric_sequence_ratio_l1317_131734

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∃ q : ℝ, ∀ n, a (n + 1) = q * a n) →
  (a 2 - (1/2 * a 3) = (1/2 * a 3) - a 1) →
  ((a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2) :=
by sorry

end geometric_sequence_ratio_l1317_131734


namespace smallest_integer_power_l1317_131743

theorem smallest_integer_power (x : ℕ) (h : x = 9 * 3) :
  (∀ c : ℕ, x^c > 3^24 → c ≥ 9) ∧ x^9 > 3^24 := by
  sorry

end smallest_integer_power_l1317_131743


namespace circle_center_parabola_focus_l1317_131761

/-- The value of p for which the center of the circle x^2 + y^2 - 6x = 0 
    is exactly the focus of the parabola y^2 = 2px (p > 0) -/
theorem circle_center_parabola_focus (p : ℝ) : p > 0 → 
  (∃ (x y : ℝ), x^2 + y^2 - 6*x = 0 ∧ y^2 = 2*p*x) →
  (∀ (x y : ℝ), x^2 + y^2 - 6*x = 0 → x = 3 ∧ y = 0) →
  (∀ (x y : ℝ), y^2 = 2*p*x → x = p/2 ∧ y = 0) →
  p = 6 := by sorry

end circle_center_parabola_focus_l1317_131761


namespace abs_cubic_inequality_l1317_131704

theorem abs_cubic_inequality (x : ℝ) : |x| ≤ 2 → |3*x - x^3| ≤ 2 := by
  sorry

end abs_cubic_inequality_l1317_131704


namespace unique_a_value_l1317_131713

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + 4 = 0}

-- State the theorem
theorem unique_a_value :
  ∃! a : ℝ, (B a).Nonempty ∧ B a ⊆ A := by sorry

end unique_a_value_l1317_131713


namespace solution_to_equation_l1317_131774

theorem solution_to_equation : ∃ x : ℝ, (2 / (x + 5) = 1 / x) ∧ x = 5 := by sorry

end solution_to_equation_l1317_131774


namespace calculation_proof_l1317_131790

theorem calculation_proof : 2325 + 300 / 75 - 425 * 2 = 1479 := by
  sorry

end calculation_proof_l1317_131790


namespace fraction_calculation_l1317_131777

theorem fraction_calculation : (0.5^3) / (0.05^2) = 50 := by sorry

end fraction_calculation_l1317_131777


namespace negation_of_universal_proposition_l1317_131717

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + |x| ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 + |x₀| < 0) := by
  sorry

end negation_of_universal_proposition_l1317_131717


namespace factor_expression_l1317_131712

/-- The expression a^3 (b^2 - c^2) + b^3 (c^2 - a^2) + c^3 (a^2 - b^2) 
    can be factored as (a - b)(b - c)(c - a) * (-(ab + ac + bc)) -/
theorem factor_expression (a b c : ℝ) :
  a^3 * (b^2 - c^2) + b^3 * (c^2 - a^2) + c^3 * (a^2 - b^2) =
  (a - b) * (b - c) * (c - a) * (-(a*b + a*c + b*c)) := by
  sorry

end factor_expression_l1317_131712


namespace max_area_rectangle_with_fixed_perimeter_l1317_131748

theorem max_area_rectangle_with_fixed_perimeter :
  ∀ (width height : ℝ),
  width > 0 → height > 0 →
  width + height = 50 →
  width * height ≤ 625 :=
by
  sorry

end max_area_rectangle_with_fixed_perimeter_l1317_131748


namespace division_problem_l1317_131793

theorem division_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 100 →
  divisor = 11 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 9 := by
sorry

end division_problem_l1317_131793


namespace natasha_exercise_time_l1317_131797

theorem natasha_exercise_time :
  -- Define variables
  ∀ (natasha_daily_minutes : ℕ) 
    (natasha_days : ℕ) 
    (esteban_daily_minutes : ℕ) 
    (esteban_days : ℕ) 
    (total_minutes : ℕ),
  -- Set conditions
  natasha_days = 7 →
  esteban_daily_minutes = 10 →
  esteban_days = 9 →
  total_minutes = 5 * 60 →
  natasha_daily_minutes * natasha_days + esteban_daily_minutes * esteban_days = total_minutes →
  -- Conclusion
  natasha_daily_minutes = 30 := by
sorry

end natasha_exercise_time_l1317_131797


namespace sufficient_not_imply_necessary_l1317_131781

-- Define the propositions A and B
variable (A B : Prop)

-- Define what it means for B to be a sufficient condition for A
def sufficient (B A : Prop) : Prop := B → A

-- Define what it means for A to be a necessary condition for B
def necessary (A B : Prop) : Prop := B → A

-- Theorem: If B is sufficient for A, it doesn't necessarily mean A is necessary for B
theorem sufficient_not_imply_necessary (h : sufficient B A) : 
  ¬ (∀ A B, sufficient B A → necessary A B) :=
sorry

end sufficient_not_imply_necessary_l1317_131781


namespace two_distinct_roots_condition_l1317_131735

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : ℝ := x^2 - 4*x + 2*m

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m = 0 ∧ quadratic_equation x₂ m = 0

-- Theorem statement
theorem two_distinct_roots_condition (m : ℝ) :
  has_two_distinct_real_roots m ↔ m < 2 :=
sorry

end two_distinct_roots_condition_l1317_131735


namespace largest_prime_divisor_test_l1317_131722

theorem largest_prime_divisor_test (n : ℕ) : 
  1000 ≤ n → n ≤ 1100 → 
  (∀ p : ℕ, p ≤ 31 → Nat.Prime p → ¬(p ∣ n)) → 
  Nat.Prime n :=
sorry

end largest_prime_divisor_test_l1317_131722


namespace f_is_even_and_decreasing_l1317_131796

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 1

-- State the theorem
theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) := by
sorry

end f_is_even_and_decreasing_l1317_131796


namespace arithmetic_mean_problem_l1317_131714

theorem arithmetic_mean_problem (y b : ℝ) (h : y ≠ 0) :
  (((y + b) / y + (2 * y - b) / y) / 2) = 1.5 := by
  sorry

end arithmetic_mean_problem_l1317_131714


namespace largest_multiple_of_15_less_than_500_l1317_131738

theorem largest_multiple_of_15_less_than_500 :
  ∀ n : ℕ, n * 15 < 500 → n * 15 ≤ 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l1317_131738


namespace three_percent_difference_l1317_131727

theorem three_percent_difference (x y : ℝ) : 
  3 = 0.15 * x → 3 = 0.25 * y → x - y = 8 := by
sorry

end three_percent_difference_l1317_131727


namespace sum_nonpositive_implies_one_nonpositive_l1317_131705

theorem sum_nonpositive_implies_one_nonpositive (x y : ℝ) : 
  x + y ≤ 0 → x ≤ 0 ∨ y ≤ 0 := by
  sorry

end sum_nonpositive_implies_one_nonpositive_l1317_131705


namespace certain_number_proof_l1317_131737

theorem certain_number_proof : ∃! N : ℕ, 
  N % 101 = 8 ∧ 
  5161 % 101 = 10 ∧ 
  N = 5159 := by
  sorry

end certain_number_proof_l1317_131737


namespace g_derivative_l1317_131732

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2 + 3^x

theorem g_derivative (x : ℝ) (h : x > 0) :
  deriv g x = 1 / (x * Real.log 2) + 3^x * Real.log 3 := by
  sorry

end g_derivative_l1317_131732


namespace evans_needed_amount_l1317_131739

/-- The amount Evan still needs to buy the watch -/
def amount_needed (david_found : ℕ) (evan_initial : ℕ) (watch_cost : ℕ) : ℕ :=
  watch_cost - (evan_initial + david_found)

/-- Theorem stating the amount Evan still needs -/
theorem evans_needed_amount :
  amount_needed 12 1 20 = 7 := by
  sorry

end evans_needed_amount_l1317_131739


namespace grade_students_ratio_l1317_131742

theorem grade_students_ratio (sixth_grade seventh_grade : ℕ) : 
  (sixth_grade : ℚ) / seventh_grade = 3 / 4 →
  seventh_grade - sixth_grade = 13 →
  sixth_grade = 39 ∧ seventh_grade = 52 :=
by
  sorry

end grade_students_ratio_l1317_131742


namespace correct_equation_representation_l1317_131740

/-- Represents the boat distribution problem from the ancient Chinese text --/
def boat_distribution_problem (total_boats : ℕ) (large_boat_capacity : ℕ) (small_boat_capacity : ℕ) (total_students : ℕ) : Prop :=
  ∃ (small_boats : ℕ),
    small_boats ≤ total_boats ∧
    (small_boats * small_boat_capacity + (total_boats - small_boats) * large_boat_capacity = total_students)

/-- Theorem stating that the equation 4x + 6(8 - x) = 38 correctly represents the boat distribution problem --/
theorem correct_equation_representation :
  boat_distribution_problem 8 6 4 38 ↔ ∃ x : ℕ, 4 * x + 6 * (8 - x) = 38 :=
sorry

end correct_equation_representation_l1317_131740


namespace reciprocal_inequality_l1317_131716

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : (1 : ℝ) / a > (1 : ℝ) / b := by
  sorry

end reciprocal_inequality_l1317_131716


namespace problem_statement_l1317_131767

theorem problem_statement (x y z : ℝ) (h : x^4 + y^4 + z^4 + x*y*z = 4) :
  x ≤ 2 ∧ Real.sqrt (2 - x) ≥ (y + z) / 2 := by
  sorry

end problem_statement_l1317_131767


namespace coefficient_of_y_squared_l1317_131724

def polynomial (x : ℝ) : ℝ :=
  (1 - x + x^2 - x^3 + x^4 - x^5 + x^6 - x^7 + x^8 - x^9 + x^10 - x^11 + x^12 - x^13 + x^14 - x^15 + x^16 - x^17)

def y (x : ℝ) : ℝ := x + 1

theorem coefficient_of_y_squared (x : ℝ) : 
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ a₁₄ a₁₅ a₁₆ a₁₇ : ℝ), 
    polynomial x = a₀ + a₁ * y x + a₂ * (y x)^2 + a₃ * (y x)^3 + a₄ * (y x)^4 + 
                   a₅ * (y x)^5 + a₆ * (y x)^6 + a₇ * (y x)^7 + a₈ * (y x)^8 + 
                   a₉ * (y x)^9 + a₁₀ * (y x)^10 + a₁₁ * (y x)^11 + a₁₂ * (y x)^12 + 
                   a₁₃ * (y x)^13 + a₁₄ * (y x)^14 + a₁₅ * (y x)^15 + a₁₆ * (y x)^16 + 
                   a₁₇ * (y x)^17 ∧
    a₂ = 816 := by
  sorry

end coefficient_of_y_squared_l1317_131724


namespace roots_sum_square_value_l1317_131775

theorem roots_sum_square_value (m n : ℝ) : 
  m^2 + 3*m - 1 = 0 → n^2 + 3*n - 1 = 0 → m^2 + 4*m + n = -2 := by
  sorry

end roots_sum_square_value_l1317_131775


namespace triangle_side_length_l1317_131765

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 4 → b = 6 → C = 2 * π / 3 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = 2 * Real.sqrt 19 := by
sorry

end triangle_side_length_l1317_131765


namespace pizza_combinations_l1317_131745

theorem pizza_combinations (n k : ℕ) (h1 : n = 8) (h2 : k = 5) :
  Nat.choose n k = 56 := by
  sorry

end pizza_combinations_l1317_131745


namespace perpendicular_vectors_x_value_l1317_131721

/-- Given two vectors a and b in ℝ², where a = (-√3, 1) and b = (1, x),
    if a and b are perpendicular, then x = √3. -/
theorem perpendicular_vectors_x_value :
  let a : ℝ × ℝ := (-Real.sqrt 3, 1)
  let b : ℝ × ℝ := (1, x)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = Real.sqrt 3 :=
by
  sorry

end perpendicular_vectors_x_value_l1317_131721


namespace sufficient_not_necessary_l1317_131729

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m1 m2 : ℝ) : Prop := m1 = m2

/-- The condition "a = -1" -/
def condition (a : ℝ) : Prop := a = -1

/-- The line ax + y - 1 = 0 is parallel to x + ay + 5 = 0 -/
def lines_are_parallel (a : ℝ) : Prop := parallel_lines (-a) (1/a)

/-- "a = -1" is a sufficient but not necessary condition for the lines to be parallel -/
theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, condition a → lines_are_parallel a) ∧
  (∃ a, lines_are_parallel a ∧ ¬condition a) :=
sorry

end sufficient_not_necessary_l1317_131729


namespace y48y_divisible_by_24_l1317_131751

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

def four_digit_number (y : ℕ) : ℕ := y * 1000 + 480 + y

theorem y48y_divisible_by_24 :
  ∃! (y : ℕ), y < 10 ∧ is_divisible_by (four_digit_number y) 24 :=
sorry

end y48y_divisible_by_24_l1317_131751


namespace find_number_l1317_131792

theorem find_number : ∃ x : ℝ, (x / 18) - 29 = 6 ∧ x = 630 := by
  sorry

end find_number_l1317_131792


namespace problem_statement_l1317_131730

/-- Given positive real numbers a, b, c, and a function f with minimum value 1, 
    prove that a + b + c = 1 and a² + b² + c² ≥ 1/3 -/
theorem problem_statement (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (hf : ∀ x, |x - a| + |x + b| + c ≥ 1) : 
    (a + b + c = 1) ∧ (a^2 + b^2 + c^2 ≥ 1/3) := by
  sorry

end problem_statement_l1317_131730


namespace rosie_pie_production_l1317_131766

/-- Given that Rosie can make 3 pies out of 12 apples, prove that she can make 9 pies with 36 apples. -/
theorem rosie_pie_production (apples_per_batch : ℕ) (pies_per_batch : ℕ) (total_apples : ℕ) 
  (h1 : apples_per_batch = 12) 
  (h2 : pies_per_batch = 3) 
  (h3 : total_apples = 36) :
  (total_apples / (apples_per_batch / pies_per_batch)) = 9 := by
  sorry

end rosie_pie_production_l1317_131766


namespace toms_vaccines_l1317_131723

theorem toms_vaccines (total_payment : ℕ) (trip_cost : ℕ) (vaccine_cost : ℕ) (doctor_visit : ℕ) 
  (insurance_coverage : ℚ) :
  total_payment = 1340 →
  trip_cost = 1200 →
  vaccine_cost = 45 →
  doctor_visit = 250 →
  insurance_coverage = 4/5 →
  ∃ (num_vaccines : ℕ), 
    (total_payment : ℚ) = trip_cost + (1 - insurance_coverage) * (doctor_visit + num_vaccines * vaccine_cost) ∧
    num_vaccines = 10 :=
by sorry

end toms_vaccines_l1317_131723


namespace g_f_neg_3_l1317_131733

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 7

-- Define g(f(3)) = 15 as a hypothesis
axiom g_f_3 : ∃ g : ℝ → ℝ, g (f 3) = 15

-- Theorem to prove
theorem g_f_neg_3 : ∃ g : ℝ → ℝ, g (f (-3)) = 15 := by
  sorry

end g_f_neg_3_l1317_131733


namespace largest_inscribed_triangle_area_l1317_131784

theorem largest_inscribed_triangle_area (r : ℝ) (hr : r = 8) :
  let circle_area := π * r^2
  let diameter := 2 * r
  let max_triangle_area := (1/2) * diameter * r
  max_triangle_area = 64 := by
  sorry

end largest_inscribed_triangle_area_l1317_131784


namespace triangle_angle_from_sides_and_area_l1317_131719

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively,
    prove that if a = 2√3, b = 2, and the area S = √3, then C = π/6 -/
theorem triangle_angle_from_sides_and_area 
  (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  a = 2 * Real.sqrt 3 →
  b = 2 →
  S = Real.sqrt 3 →
  S = 1/2 * a * b * Real.sin C →
  C = π/6 := by
  sorry


end triangle_angle_from_sides_and_area_l1317_131719


namespace equation_solution_l1317_131759

theorem equation_solution :
  ∃ y : ℝ, (y = 18 / 7 ∧ (Real.sqrt (8 * y) / Real.sqrt (4 * (y - 2)) = 3)) := by
  sorry

end equation_solution_l1317_131759


namespace magnitude_v_l1317_131783

/-- Given complex numbers u and v, prove that |v| = 5.2 under the given conditions -/
theorem magnitude_v (u v : ℂ) : 
  u * v = 24 - 10 * Complex.I → Complex.abs u = 5 → Complex.abs v = 5.2 := by
  sorry

end magnitude_v_l1317_131783


namespace actual_score_calculation_l1317_131769

/-- Given the following conditions:
  * The passing threshold is 30% of the maximum score
  * The maximum possible score is 790
  * The actual score falls short of the passing threshold by 25 marks
  Prove that the actual score is 212 marks -/
theorem actual_score_calculation (passing_threshold : Real) (max_score : Nat) (shortfall : Nat) :
  passing_threshold = 0.30 →
  max_score = 790 →
  shortfall = 25 →
  ⌊passing_threshold * max_score⌋ - shortfall = 212 := by
  sorry

end actual_score_calculation_l1317_131769


namespace box_fits_40_blocks_l1317_131776

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℕ
  width : ℕ
  length : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ :=
  d.height * d.width * d.length

/-- Calculates how many smaller objects can fit into a larger object -/
def fitCount (larger smaller : Dimensions) : ℕ :=
  (volume larger) / (volume smaller)

theorem box_fits_40_blocks : 
  let box := Dimensions.mk 8 10 12
  let block := Dimensions.mk 3 2 4
  fitCount box block = 40 := by
  sorry

#eval fitCount (Dimensions.mk 8 10 12) (Dimensions.mk 3 2 4)

end box_fits_40_blocks_l1317_131776


namespace newberg_airport_passengers_l1317_131778

theorem newberg_airport_passengers :
  let on_time : ℕ := 14507
  let late : ℕ := 213
  on_time + late = 14720 :=
by sorry

end newberg_airport_passengers_l1317_131778


namespace pictures_on_front_l1317_131757

theorem pictures_on_front (total : ℕ) (on_back : ℕ) (h1 : total = 15) (h2 : on_back = 9) :
  total - on_back = 6 := by
  sorry

end pictures_on_front_l1317_131757


namespace rectangle_max_area_l1317_131718

/-- A rectangle with integer dimensions and perimeter 30 has a maximum area of 56 -/
theorem rectangle_max_area :
  ∀ l w : ℕ,
  l + w = 15 →
  ∀ a b : ℕ,
  a + b = 15 →
  l * w ≤ 56 :=
by sorry

end rectangle_max_area_l1317_131718


namespace distance_point_to_line_l1317_131741

/-- The distance from the point (1, 0) to the line x - y + 1 = 0 is √2 -/
theorem distance_point_to_line : 
  let point : ℝ × ℝ := (1, 0)
  let line (x y : ℝ) : Prop := x - y + 1 = 0
  Real.sqrt 2 = (|1 - 0 + 1|) / Real.sqrt (1^2 + (-1)^2) := by sorry

end distance_point_to_line_l1317_131741


namespace joan_grilled_cheese_sandwiches_l1317_131726

/-- Represents the number of cheese slices required for one ham sandwich. -/
def ham_cheese_slices : ℕ := 2

/-- Represents the number of cheese slices required for one grilled cheese sandwich. -/
def grilled_cheese_slices : ℕ := 3

/-- Represents the total number of cheese slices Joan uses. -/
def total_cheese_slices : ℕ := 50

/-- Represents the number of ham sandwiches Joan makes. -/
def ham_sandwiches : ℕ := 10

/-- Proves that Joan makes 10 grilled cheese sandwiches. -/
theorem joan_grilled_cheese_sandwiches : 
  (total_cheese_slices - ham_cheese_slices * ham_sandwiches) / grilled_cheese_slices = 10 := by
  sorry

end joan_grilled_cheese_sandwiches_l1317_131726


namespace distance_inequality_l1317_131782

-- Define the space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

-- Define the vertices of the quadrilateral
variable (A B C D : V)

-- Define the condition that all sides are equal
variable (h : ‖A - B‖ = ‖B - C‖ ∧ ‖B - C‖ = ‖C - D‖ ∧ ‖C - D‖ = ‖D - A‖)

-- State the theorem
theorem distance_inequality (P : V) : 
  ‖P - A‖ < ‖P - B‖ + ‖P - C‖ + ‖P - D‖ := by sorry

end distance_inequality_l1317_131782


namespace least_months_to_triple_l1317_131709

def interest_rate : ℝ := 1.06

theorem least_months_to_triple (t : ℕ) : t = 19 ↔ 
  (∀ n : ℕ, n < 19 → interest_rate ^ n ≤ 3) ∧ 
  interest_rate ^ 19 > 3 := by
  sorry

end least_months_to_triple_l1317_131709


namespace tangent_line_equation_l1317_131794

/-- The equation of a line tangent to a unit circle that intersects a specific ellipse -/
theorem tangent_line_equation (k b : ℝ) (h_b_pos : b > 0) 
  (h_tangent : b^2 = k^2 + 1)
  (h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ + b ∧ 
    y₂ = k * x₂ + b ∧ 
    x₁^2 / 2 + y₁^2 = 1 ∧ 
    x₂^2 / 2 + y₂^2 = 1)
  (h_dot_product : 
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      y₁ = k * x₁ + b → 
      y₂ = k * x₂ + b → 
      x₁^2 / 2 + y₁^2 = 1 → 
      x₂^2 / 2 + y₂^2 = 1 → 
      x₁ * x₂ + y₁ * y₂ = 2/3) :
  (k = 1 ∧ b = Real.sqrt 2) ∨ (k = -1 ∧ b = Real.sqrt 2) :=
sorry

end tangent_line_equation_l1317_131794


namespace students_with_b_in_smith_class_l1317_131760

/-- Calculates the number of students who received a B in Ms. Smith's class -/
theorem students_with_b_in_smith_class 
  (johnson_total : ℕ) 
  (johnson_b : ℕ) 
  (smith_total : ℕ) 
  (h1 : johnson_total = 30)
  (h2 : johnson_b = 18)
  (h3 : smith_total = 45)
  (h4 : johnson_b * smith_total = johnson_total * (smith_total * johnson_b / johnson_total)) :
  smith_total * johnson_b / johnson_total = 27 := by
  sorry

#check students_with_b_in_smith_class

end students_with_b_in_smith_class_l1317_131760


namespace expression_range_l1317_131795

theorem expression_range (x a b c : ℝ) (h : a^2 + b^2 + c^2 ≠ 0) :
  ∃ y ∈ Set.Icc (-Real.sqrt 5) (Real.sqrt 5),
    y = (a * Real.cos x - b * Real.sin x + 2 * c) / Real.sqrt (a^2 + b^2 + c^2) := by
  sorry

end expression_range_l1317_131795


namespace triangle_properties_l1317_131707

/-- Given a triangle ABC with sides a, b, and c, prove the following properties -/
theorem triangle_properties (A B C : ℝ × ℝ) (a b c : ℝ) :
  let AB := B - A
  let BC := C - B
  let CA := A - C
  -- Given condition
  AB • AC + 2 * (-AB) • BC = 3 * (-CA) • (-BC) →
  -- Side lengths
  ‖BC‖ = a ∧ ‖CA‖ = b ∧ ‖AB‖ = c →
  -- Prove these properties
  a^2 + 2*b^2 = 3*c^2 ∧ 
  ∀ (cos_C : ℝ), cos_C = (a^2 + b^2 - c^2) / (2*a*b) → cos_C ≥ Real.sqrt 2 / 3 :=
by sorry

end triangle_properties_l1317_131707


namespace expression_evaluation_l1317_131752

theorem expression_evaluation :
  let x : ℚ := -1/2
  (x - 3)^2 + (x + 3)*(x - 3) - 2*x*(x - 2) + 1 = 2 := by
  sorry

end expression_evaluation_l1317_131752


namespace p_sufficient_not_necessary_for_q_l1317_131747

-- Define the conditions p and q
def p (x : ℝ) : Prop := (x - 2)^2 ≤ 1
def q (x : ℝ) : Prop := 2 / (x - 1) ≥ 1

-- Define the set of x that satisfy p
def p_set : Set ℝ := {x | p x}

-- Define the set of x that satisfy q
def q_set : Set ℝ := {x | q x}

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (q_set ⊆ p_set) ∧ ¬(p_set ⊆ q_set) := by sorry

end p_sufficient_not_necessary_for_q_l1317_131747


namespace max_triangle_area_l1317_131789

def parabola (x : ℝ) : ℝ := x^2 - 6*x + 9

theorem max_triangle_area :
  let A : ℝ × ℝ := (0, 9)
  let B : ℝ × ℝ := (6, 9)
  ∀ p q : ℝ,
    1 ≤ p → p ≤ 6 →
    q = parabola p →
    let C : ℝ × ℝ := (p, q)
    let area := abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1)) / 2
    area ≤ 27 :=
by sorry

end max_triangle_area_l1317_131789


namespace age_difference_proof_l1317_131749

theorem age_difference_proof (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → 
  (10 * a + b + 3 = 3 * (10 * b + a + 3)) → 
  (10 * a + b) - (10 * b + a) = 36 := by
  sorry

end age_difference_proof_l1317_131749


namespace common_root_equations_l1317_131755

theorem common_root_equations (p : ℤ) (x : ℚ) : 
  (3 * x^2 - 4 * x + p - 2 = 0 ∧ x^2 - 2 * p * x + 5 = 0) ↔ (p = 3 ∧ x = 1) :=
by sorry

#check common_root_equations

end common_root_equations_l1317_131755


namespace product_of_roots_l1317_131762

theorem product_of_roots (x : ℝ) : (x + 4) * (x - 5) = 22 → ∃ y : ℝ, (x + 4) * (x - 5) = 22 ∧ (x * y = -42) := by
  sorry

end product_of_roots_l1317_131762


namespace f_sum_equals_half_point_five_l1317_131799

/-- A function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- f is an odd function -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- f(x+1) = -f(x) for all x -/
axiom f_period (x : ℝ) : f (x + 1) = -f x

/-- f(x) = x for x in (-1, 1) -/
axiom f_identity (x : ℝ) (h : x > -1 ∧ x < 1) : f x = x

/-- The main theorem to prove -/
theorem f_sum_equals_half_point_five : f 3 + f (-7.5) = 0.5 := by sorry

end f_sum_equals_half_point_five_l1317_131799


namespace divisor_sum_five_l1317_131791

/-- d(m) is the number of positive divisors of m -/
def d (m : ℕ+) : ℕ := sorry

/-- Theorem: For a positive integer n, d(n) + d(n+1) = 5 if and only if n = 3 or n = 4 -/
theorem divisor_sum_five (n : ℕ+) : d n + d (n + 1) = 5 ↔ n = 3 ∨ n = 4 := by sorry

end divisor_sum_five_l1317_131791


namespace cos_ninety_degrees_l1317_131720

theorem cos_ninety_degrees : Real.cos (π / 2) = 0 := by
  sorry

end cos_ninety_degrees_l1317_131720


namespace travel_time_calculation_l1317_131772

theorem travel_time_calculation (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_distance = 300 →
  speed1 = 30 →
  speed2 = 25 →
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 11 := by
  sorry

end travel_time_calculation_l1317_131772


namespace arithmetic_sequence_second_term_l1317_131771

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_second_term
  (a : ℕ → ℤ)
  (h_arithmetic : ArithmeticSequence a)
  (h_10th : a 10 = 15)
  (h_11th : a 11 = 18) :
  a 2 = -9 := by
sorry

end arithmetic_sequence_second_term_l1317_131771


namespace maria_friends_money_l1317_131703

def problem (maria_total rene_amount : ℚ) : Prop :=
  let isha_amount := maria_total / 4
  let florence_amount := isha_amount / 2
  let john_amount := florence_amount / 3
  florence_amount = 4 * rene_amount ∧
  rene_amount = 450 ∧
  isha_amount + florence_amount + rene_amount + john_amount = 6450

theorem maria_friends_money :
  ∃ maria_total : ℚ, problem maria_total 450 :=
sorry

end maria_friends_money_l1317_131703


namespace cinema_systematic_sampling_l1317_131763

/-- Represents a sampling method --/
inductive SamplingMethod
  | LotteryMethod
  | RandomNumberMethod
  | StratifiedSampling
  | SystematicSampling

/-- Represents a cinema with rows and seats --/
structure Cinema where
  rows : Nat
  seatsPerRow : Nat

/-- Represents a selection of audience members --/
structure AudienceSelection where
  seatNumber : Nat
  count : Nat

/-- Determines the sampling method based on the cinema layout and audience selection --/
def determineSamplingMethod (c : Cinema) (a : AudienceSelection) : SamplingMethod :=
  sorry

/-- Theorem stating that the given scenario results in systematic sampling --/
theorem cinema_systematic_sampling (c : Cinema) (a : AudienceSelection) :
  c.rows = 30 ∧ c.seatsPerRow = 25 ∧ a.seatNumber = 18 ∧ a.count = 30 →
  determineSamplingMethod c a = SamplingMethod.SystematicSampling :=
  sorry

end cinema_systematic_sampling_l1317_131763


namespace both_products_not_qualified_l1317_131710

-- Define the qualification rates for Factory A and Factory B
def qualification_rate_A : ℝ := 0.9
def qualification_rate_B : ℝ := 0.8

-- Define the probability that both products are not qualified
def both_not_qualified : ℝ := (1 - qualification_rate_A) * (1 - qualification_rate_B)

-- Theorem statement
theorem both_products_not_qualified :
  both_not_qualified = 0.02 :=
sorry

end both_products_not_qualified_l1317_131710


namespace complex_power_problem_l1317_131773

theorem complex_power_problem : (((1 - Complex.I) / (1 + Complex.I)) ^ 10 : ℂ) = -1 := by
  sorry

end complex_power_problem_l1317_131773


namespace complex_expression_equals_one_l1317_131746

theorem complex_expression_equals_one : 
  (((4.5 * (1 + 2/3) - 6.75) * (2/3)) / 
   ((3 + 1/3) * 0.3 + (5 + 1/3) * (1/8)) / (2 + 2/3)) + 
  ((1 + 4/11) * 0.22 / 0.3 - 0.96) / 
   ((0.2 - 3/40) * 1.6) = 1 := by sorry

end complex_expression_equals_one_l1317_131746


namespace intersection_equals_interval_l1317_131736

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x^2 - 2*x ≥ 3}
def Q : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

-- Define the intersection of P and Q
def PQ_intersection : Set ℝ := P ∩ Q

-- Define the half-open interval [3,4)
def interval_3_4 : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 4}

-- Theorem statement
theorem intersection_equals_interval : PQ_intersection = interval_3_4 := by
  sorry

end intersection_equals_interval_l1317_131736


namespace five_letter_words_with_vowels_l1317_131706

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def consonants : Finset Char := alphabet \ vowels

def word_length : Nat := 5

theorem five_letter_words_with_vowels :
  (alphabet.card ^ word_length) - (consonants.card ^ word_length) = 6752 := by
  sorry

end five_letter_words_with_vowels_l1317_131706


namespace sum_of_digits_of_square_1111_l1317_131770

def repeat_digit (d : Nat) (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | m + 1 => d + 10 * (repeat_digit d m)

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n
  else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_square_1111 :
  sum_of_digits ((repeat_digit 1 4) ^ 2) = 16 := by
  sorry

end sum_of_digits_of_square_1111_l1317_131770


namespace five_point_thirty_five_million_equals_scientific_notation_l1317_131764

-- Define 5.35 million
def five_point_thirty_five_million : ℝ := 5.35 * 1000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 5.35 * (10 ^ 6)

-- Theorem to prove equality
theorem five_point_thirty_five_million_equals_scientific_notation : 
  five_point_thirty_five_million = scientific_notation := by
  sorry

end five_point_thirty_five_million_equals_scientific_notation_l1317_131764


namespace regression_change_l1317_131779

/-- Represents a linear regression equation of the form y = a + bx -/
structure LinearRegression where
  a : ℝ  -- y-intercept
  b : ℝ  -- slope

/-- Calculates the change in y given a change in x for a linear regression -/
def changeInY (reg : LinearRegression) (dx : ℝ) : ℝ :=
  reg.b * dx

theorem regression_change 
  (reg : LinearRegression) 
  (h1 : reg.a = 2)
  (h2 : reg.b = -2.5) : 
  changeInY reg 2 = -5 := by
  sorry

end regression_change_l1317_131779


namespace four_digit_number_remainder_l1317_131750

theorem four_digit_number_remainder (a b c d : Nat) 
  (h1 : a ≠ 0) 
  (h2 : d ≠ 0) 
  (h3 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) 
  (h4 : ∃ k : Int, (1000 * a + 100 * b + 10 * c + d) + (1000 * a + 100 * c + 10 * b + d) = 900 * k) : 
  (1000 * a + 100 * b + 10 * c + d) % 90 = 45 := by
sorry

end four_digit_number_remainder_l1317_131750


namespace at_least_one_intersection_l1317_131702

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of a line lying in a plane
variable (lies_in : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the property of a line intersecting another line
variable (intersects : Line → Line → Prop)

-- Theorem statement
theorem at_least_one_intersection 
  (a b c : Line) (α β : Plane)
  (h1 : skew a b)
  (h2 : lies_in a α)
  (h3 : lies_in b β)
  (h4 : c = intersect α β) :
  intersects c a ∨ intersects c b :=
sorry

end at_least_one_intersection_l1317_131702


namespace wall_bricks_l1317_131731

/-- Represents the time taken by the first bricklayer to build the wall alone -/
def time1 : ℝ := 8

/-- Represents the time taken by the second bricklayer to build the wall alone -/
def time2 : ℝ := 12

/-- Represents the reduction in productivity when working together (in bricks per hour) -/
def reduction : ℝ := 15

/-- Represents the time taken by both bricklayers working together to build the wall -/
def timeJoint : ℝ := 6

/-- Represents the total number of bricks in the wall -/
def totalBricks : ℝ := 360

theorem wall_bricks : 
  timeJoint * (totalBricks / time1 + totalBricks / time2 - reduction) = totalBricks := by
  sorry

end wall_bricks_l1317_131731


namespace largest_three_digit_product_l1317_131786

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem largest_three_digit_product :
  ∀ n x y : ℕ,
    n ≥ 100 ∧ n < 1000 →
    is_prime x ∧ is_prime y ∧ is_prime (10 * y + x) →
    x < 10 ∧ y < 10 →
    x ≠ y ∧ x ≠ (10 * y + x) ∧ y ≠ (10 * y + x) →
    n = x * y * (10 * y + x) →
    n ≤ 777 :=
by sorry

end largest_three_digit_product_l1317_131786


namespace function_composition_equality_l1317_131780

/-- Given two functions p and q, where p(x) = 5x - 4 and q(x) = 4x - b,
    prove that if p(q(5)) = 16, then b = 16. -/
theorem function_composition_equality (b : ℝ) : 
  (let p : ℝ → ℝ := λ x => 5 * x - 4
   let q : ℝ → ℝ := λ x => 4 * x - b
   p (q 5) = 16) → b = 16 := by
  sorry

end function_composition_equality_l1317_131780


namespace largest_four_digit_divisible_by_smallest_primes_l1317_131787

def smallest_primes : List Nat := [2, 3, 5, 7, 11]

def is_divisible_by_all (n : Nat) (lst : List Nat) : Prop :=
  ∀ m ∈ lst, n % m = 0

theorem largest_four_digit_divisible_by_smallest_primes :
  ∀ n : Nat, n ≤ 9999 → n ≥ 1000 →
  is_divisible_by_all n smallest_primes →
  n ≤ 9240 :=
sorry

end largest_four_digit_divisible_by_smallest_primes_l1317_131787


namespace range_of_a_l1317_131711

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 + 4 * (a - 2) * x + 1 ≠ 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 > 0

-- Define the theorem
theorem range_of_a (a : ℝ) (h : p a ∨ q a) : -2 < a ∧ a < 3 := by
  sorry

end range_of_a_l1317_131711


namespace percentage_of_amount_twenty_five_percent_of_500_l1317_131744

theorem percentage_of_amount (amount : ℝ) (percentage : ℝ) :
  (percentage / 100) * amount = (percentage * amount) / 100 := by sorry

theorem twenty_five_percent_of_500 :
  (25 : ℝ) / 100 * 500 = 125 := by sorry

end percentage_of_amount_twenty_five_percent_of_500_l1317_131744
