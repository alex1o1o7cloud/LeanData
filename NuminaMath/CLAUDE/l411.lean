import Mathlib

namespace NUMINAMATH_CALUDE_max_sum_at_11_l411_41109

/-- Arithmetic sequence with first term 21 and common difference -2 -/
def arithmetic_sequence (n : ℕ) : ℚ := 21 - 2 * (n - 1)

/-- Sum of first n terms of the arithmetic sequence -/
def sequence_sum (n : ℕ) : ℚ := (n : ℚ) * (21 + arithmetic_sequence n) / 2

/-- The sum reaches its maximum value when n = 11 -/
theorem max_sum_at_11 : 
  ∀ k : ℕ, k ≠ 0 → sequence_sum 11 ≥ sequence_sum k :=
sorry

end NUMINAMATH_CALUDE_max_sum_at_11_l411_41109


namespace NUMINAMATH_CALUDE_zoo_tickets_cost_l411_41156

/-- The total cost of zoo tickets for a group of children and adults. -/
def total_cost (num_children num_adults child_ticket_price adult_ticket_price : ℕ) : ℕ :=
  num_children * child_ticket_price + num_adults * adult_ticket_price

/-- Theorem: The total cost of zoo tickets for a group of 6 children and 10 adults is $220,
    given that child tickets cost $10 each and adult tickets cost $16 each. -/
theorem zoo_tickets_cost :
  total_cost 6 10 10 16 = 220 := by
sorry

end NUMINAMATH_CALUDE_zoo_tickets_cost_l411_41156


namespace NUMINAMATH_CALUDE_problem_solution_l411_41154

theorem problem_solution (x y : ℝ) (h1 : x + y = 4) (h2 : x - y = 6) :
  2 * x^2 - 2 * y^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l411_41154


namespace NUMINAMATH_CALUDE_num_biology_books_is_14_l411_41157

/-- The number of ways to choose 2 books from n books -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of chemistry books -/
def num_chemistry_books : ℕ := 8

/-- The total number of ways to choose 2 biology and 2 chemistry books -/
def total_ways : ℕ := 2548

/-- The number of biology books satisfies the given conditions -/
theorem num_biology_books_is_14 : 
  ∃ (n : ℕ), n > 0 ∧ choose_two n * choose_two num_chemistry_books = total_ways ∧ n = 14 :=
sorry

end NUMINAMATH_CALUDE_num_biology_books_is_14_l411_41157


namespace NUMINAMATH_CALUDE_student_competition_numbers_l411_41188

theorem student_competition_numbers (n : ℕ) : 
  (100 < n ∧ n < 200) ∧ 
  (∃ k : ℕ, n = 4 * k - 2) ∧
  (∃ l : ℕ, n = 5 * l - 3) ∧
  (∃ m : ℕ, n = 6 * m - 4) →
  n = 122 ∨ n = 182 := by
sorry

end NUMINAMATH_CALUDE_student_competition_numbers_l411_41188


namespace NUMINAMATH_CALUDE_count_odd_coefficients_l411_41110

/-- The number of odd coefficients in (x^2 + x + 1)^n -/
def odd_coefficients (n : ℕ+) : ℕ :=
  (2^n.val - 1) / 3 * 4 + 1

/-- Theorem stating the number of odd coefficients in (x^2 + x + 1)^n -/
theorem count_odd_coefficients (n : ℕ+) :
  odd_coefficients n = (2^n.val - 1) / 3 * 4 + 1 :=
by sorry

end NUMINAMATH_CALUDE_count_odd_coefficients_l411_41110


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l411_41125

-- Problem 1
theorem problem_1 : (-16) - 25 + (-43) - (-39) = -45 := by sorry

-- Problem 2
theorem problem_2 : (-3/4)^2 * (-8 + 1/3) = -69/16 := by sorry

-- Problem 3
theorem problem_3 : 16 / (-1/2) * 3/8 - |(-45)| / 9 = -17 := by sorry

-- Problem 4
theorem problem_4 : -1^2024 - (2 - 0.75) * 2/7 * (4 - (-5)^2) = 13/2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l411_41125


namespace NUMINAMATH_CALUDE_suzanna_bike_ride_l411_41178

/-- Suzanna's bike ride problem -/
theorem suzanna_bike_ride (speed : ℝ) (total_time : ℝ) (break_time : ℝ) (distance : ℝ) : 
  speed = 2 / 10 → 
  total_time = 30 → 
  break_time = 5 → 
  distance = speed * (total_time - break_time) → 
  distance = 5 := by
  sorry

end NUMINAMATH_CALUDE_suzanna_bike_ride_l411_41178


namespace NUMINAMATH_CALUDE_some_number_value_l411_41194

theorem some_number_value : ∀ some_number : ℝ, 
  (54 / some_number) * (54 / 162) = 1 → some_number = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l411_41194


namespace NUMINAMATH_CALUDE_find_x_value_l411_41162

theorem find_x_value (x : ℝ) : (15 : ℝ)^x * 8^3 / 256 = 450 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l411_41162


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l411_41151

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 1}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- Define the interval (1, 2]
def interval_one_two : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_equals_interval : A ∩ B = interval_one_two := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l411_41151


namespace NUMINAMATH_CALUDE_tylers_age_l411_41140

theorem tylers_age (clay jessica alex tyler : ℕ) : 
  tyler = 3 * clay + 1 →
  jessica = 2 * tyler - 4 →
  alex = (clay + jessica) / 2 →
  clay + jessica + alex + tyler = 52 →
  tyler = 13 := by
sorry

end NUMINAMATH_CALUDE_tylers_age_l411_41140


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l411_41152

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 2*x = -1) : 5 + x*(x + 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l411_41152


namespace NUMINAMATH_CALUDE_largest_integer_with_two_digit_square_l411_41103

theorem largest_integer_with_two_digit_square : ∃ M : ℕ, 
  (∀ n : ℕ, n^2 ≥ 10 ∧ n^2 < 100 → n ≤ M) ∧ 
  M^2 ≥ 10 ∧ M^2 < 100 ∧ 
  M = 9 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_two_digit_square_l411_41103


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l411_41193

theorem sqrt_expression_equality : 
  (Real.sqrt 3 + Real.sqrt 2)^2 - (Real.sqrt 3 - Real.sqrt 2) * (Real.sqrt 3 + Real.sqrt 2) = 4 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l411_41193


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l411_41145

/-- Given a rectangle with an inscribed circle of radius 7 and a length-to-width ratio of 3:1,
    prove that the area of the rectangle is 588. -/
theorem inscribed_circle_rectangle_area :
  ∀ (length width radius : ℝ),
    radius = 7 →
    length = 3 * width →
    width = 2 * radius →
    length * width = 588 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l411_41145


namespace NUMINAMATH_CALUDE_function_inequality_l411_41173

theorem function_inequality (f : ℝ → ℝ) (h₁ : Differentiable ℝ f) 
  (h₂ : ∀ x, deriv f x < f x) : 
  f 1 < Real.exp 1 * f 0 ∧ f 2017 < Real.exp 2017 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l411_41173


namespace NUMINAMATH_CALUDE_phi_bounded_by_one_l411_41171

/-- The functional equation satisfied by f and φ -/
def FunctionalEquation (f φ : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) + f (x - y) = 2 * φ y * f x

/-- f is not identically zero -/
def NotIdenticallyZero (f : ℝ → ℝ) : Prop :=
  ∃ x, f x ≠ 0

/-- The absolute value of f is bounded by 1 -/
def BoundedByOne (f : ℝ → ℝ) : Prop :=
  ∀ x, |f x| ≤ 1

/-- The main theorem -/
theorem phi_bounded_by_one
    (f φ : ℝ → ℝ)
    (h_eq : FunctionalEquation f φ)
    (h_nz : NotIdenticallyZero f)
    (h_bound : BoundedByOne f) :
    BoundedByOne φ := by
  sorry

end NUMINAMATH_CALUDE_phi_bounded_by_one_l411_41171


namespace NUMINAMATH_CALUDE_symmetry_of_points_l411_41141

/-- The line of symmetry --/
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

/-- Check if two points are symmetric with respect to a line --/
def is_symmetric (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  line_of_symmetry midpoint_x midpoint_y ∧
  (y₂ - y₁) / (x₂ - x₁) = -1

theorem symmetry_of_points :
  is_symmetric 2 2 3 1 :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_points_l411_41141


namespace NUMINAMATH_CALUDE_unique_solution_l411_41163

def pizza_problem (boys girls : ℕ) : Prop :=
  let day1_consumption := 7 * boys + 3 * girls
  let day2_consumption := 6 * boys + 2 * girls
  (49 ≤ day1_consumption) ∧ (day1_consumption ≤ 59) ∧
  (49 ≤ day2_consumption) ∧ (day2_consumption ≤ 59)

theorem unique_solution : ∃! (b g : ℕ), pizza_problem b g ∧ b = 8 ∧ g = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l411_41163


namespace NUMINAMATH_CALUDE_intersection_empty_implies_m_range_l411_41101

theorem intersection_empty_implies_m_range (m : ℝ) : 
  let A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
  let B : Set ℝ := {x | x*(x-m) > 0}
  (A ∩ B = ∅) → m ≥ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_m_range_l411_41101


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l411_41155

theorem stratified_sampling_theorem (elementary middle high : ℕ) 
  (h1 : elementary = 126) 
  (h2 : middle = 280) 
  (h3 : high = 95) 
  (sample_size : ℕ) 
  (h4 : sample_size = 100) : 
  ∃ (adjusted_elementary : ℕ), 
    adjusted_elementary = elementary - 1 ∧ 
    (adjusted_elementary + middle + high) % sample_size = 0 ∧
    (adjusted_elementary / 5 + middle / 5 + high / 5 = sample_size) :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l411_41155


namespace NUMINAMATH_CALUDE_find_A_l411_41116

theorem find_A : ∃ A : ℕ, A ≥ 1 ∧ A ≤ 9 ∧ (10 * A + 72) - 23 = 549 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l411_41116


namespace NUMINAMATH_CALUDE_range_of_alpha_minus_half_beta_l411_41105

theorem range_of_alpha_minus_half_beta (α β : Real) 
  (h_α : 0 ≤ α ∧ α ≤ π/2) 
  (h_β : π/2 ≤ β ∧ β ≤ π) : 
  ∃ (x : Real), x = α - β/2 ∧ -π/2 ≤ x ∧ x ≤ π/4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_alpha_minus_half_beta_l411_41105


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_inscribed_circle_theorem_l411_41131

/-- An isosceles right triangle with an inscribed circle -/
structure IsoscelesRightTriangleWithCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The height from the right angle to the hypotenuse -/
  h : ℝ
  /-- The height is twice the radius -/
  height_radius_relation : h = 2 * r
  /-- The radius is √2/4 -/
  radius_value : r = Real.sqrt 2 / 4

/-- The theorem to be proved -/
theorem isosceles_right_triangle_inscribed_circle_theorem 
  (triangle : IsoscelesRightTriangleWithCircle) : 
  triangle.h - triangle.r = Real.sqrt 2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_right_triangle_inscribed_circle_theorem_l411_41131


namespace NUMINAMATH_CALUDE_hawks_first_half_score_l411_41170

/-- Represents the score of a basketball team in a game with two halves -/
structure TeamScore where
  first_half : ℕ
  second_half : ℕ

/-- Represents the final scores of two teams in a basketball game -/
structure GameScore where
  eagles : TeamScore
  hawks : TeamScore

/-- The conditions of the basketball game -/
def game_conditions (game : GameScore) : Prop :=
  let eagles_total := game.eagles.first_half + game.eagles.second_half
  let hawks_total := game.hawks.first_half + game.hawks.second_half
  eagles_total + hawks_total = 120 ∧
  eagles_total = hawks_total + 16 ∧
  game.hawks.second_half = game.hawks.first_half + 8

theorem hawks_first_half_score (game : GameScore) :
  game_conditions game → game.hawks.first_half = 22 := by
  sorry

#check hawks_first_half_score

end NUMINAMATH_CALUDE_hawks_first_half_score_l411_41170


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l411_41153

/-- Proves that if in a triangle with sides a, b, c and opposite angles α, β, γ,
    the equation a + b = tan(γ/2) * (a * tan(α) + b * tan(β)) holds, then α = β. -/
theorem isosceles_triangle_condition 
  (a b c : ℝ) 
  (α β γ : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < α ∧ 0 < β ∧ 0 < γ)
  (h_angles : α + β + γ = Real.pi)
  (h_condition : a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β)) :
  α = β := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_condition_l411_41153


namespace NUMINAMATH_CALUDE_equation_solution_l411_41192

theorem equation_solution (x : ℝ) : 
  (3 * x + 25 ≠ 0) → 
  ((8 * x^2 + 75 * x - 3) / (3 * x + 25) = 2 * x + 5 ↔ x = -16 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l411_41192


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l411_41142

theorem least_subtraction_for_divisibility : 
  ∃! x : ℕ, x ≤ 29 ∧ (10154 - x) % 30 = 0 ∧ ∀ y : ℕ, y < x → (10154 - y) % 30 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l411_41142


namespace NUMINAMATH_CALUDE_open_box_volume_l411_41177

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet -/
theorem open_box_volume
  (sheet_length : ℝ)
  (sheet_width : ℝ)
  (cut_length : ℝ)
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_length = 3)
  : (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 3780 :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l411_41177


namespace NUMINAMATH_CALUDE_family_earning_members_l411_41111

theorem family_earning_members (initial_avg : ℚ) (new_avg : ℚ) (deceased_income : ℚ) :
  initial_avg = 735 →
  new_avg = 650 →
  deceased_income = 905 →
  ∃ n : ℕ, n * initial_avg - (n - 1) * new_avg = deceased_income ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_family_earning_members_l411_41111


namespace NUMINAMATH_CALUDE_base_equality_l411_41183

theorem base_equality : ∃ (n k : ℕ), n > 1 ∧ k > 1 ∧ n^2 + 1 = k^4 + k^3 + k + 1 := by
  sorry

end NUMINAMATH_CALUDE_base_equality_l411_41183


namespace NUMINAMATH_CALUDE_function_f_linear_positive_l411_41149

/-- A function satisfying the given conditions -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∀ x, x * (f (x + 1) - f x) = f x) ∧
  (∀ x y, |f x - f y| ≤ |x - y|)

/-- The main theorem -/
theorem function_f_linear_positive (f : ℝ → ℝ) (hf : FunctionF f) :
  ∃ k : ℝ, ∀ x > 0, f x = k * x :=
sorry

end NUMINAMATH_CALUDE_function_f_linear_positive_l411_41149


namespace NUMINAMATH_CALUDE_parallelogram_product_l411_41184

/-- Given a parallelogram EFGH with side lengths as specified, 
    prove that the product of x and y is 57√2 -/
theorem parallelogram_product (x y : ℝ) : 
  58 = 3 * x + 1 →   -- EF = GH
  2 * y^2 = 36 →     -- FG = HE
  x * y = 57 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_product_l411_41184


namespace NUMINAMATH_CALUDE_johns_final_push_time_l411_41122

/-- The time of John's final push in a speed walking race --/
theorem johns_final_push_time (john_initial_distance_behind : ℝ)
                               (john_speed : ℝ)
                               (steve_speed : ℝ)
                               (john_final_distance_ahead : ℝ)
                               (h1 : john_initial_distance_behind = 16)
                               (h2 : john_speed = 4.2)
                               (h3 : steve_speed = 3.7)
                               (h4 : john_final_distance_ahead = 2) :
  let t : ℝ := (john_initial_distance_behind + john_final_distance_ahead) / (john_speed - steve_speed)
  t = 36 := by
  sorry

end NUMINAMATH_CALUDE_johns_final_push_time_l411_41122


namespace NUMINAMATH_CALUDE_cost_for_holly_fence_l411_41108

/-- The total cost to plant trees along a fence --/
def total_cost (fence_length_yards : ℕ) (tree_width_feet : ℚ) (cost_per_tree : ℚ) : ℚ :=
  let fence_length_feet := fence_length_yards * 3
  let num_trees := fence_length_feet / tree_width_feet
  num_trees * cost_per_tree

/-- Proof that the total cost to plant trees along a 25-yard fence,
    where each tree is 1.5 feet wide and costs $8.00, is $400.00 --/
theorem cost_for_holly_fence :
  total_cost 25 (3/2) 8 = 400 := by
  sorry

end NUMINAMATH_CALUDE_cost_for_holly_fence_l411_41108


namespace NUMINAMATH_CALUDE_range_of_m_l411_41199

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 1) * x + 2

-- Define the condition that the solution set is ℝ
def solution_set_is_real (m : ℝ) : Prop :=
  ∀ x, f m x > 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  solution_set_is_real m ↔ (1 ≤ m ∧ m < 9) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l411_41199


namespace NUMINAMATH_CALUDE_b_eq_one_sufficient_not_necessary_l411_41196

/-- The condition for the line and curve to have common points -/
def has_common_points (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ x^2 + (y - 1)^2 = 1

/-- The statement that b = 1 is sufficient but not necessary for common points -/
theorem b_eq_one_sufficient_not_necessary :
  (∀ k : ℝ, has_common_points k 1) ∧
  (∃ k b : ℝ, b ≠ 1 ∧ has_common_points k b) :=
sorry

end NUMINAMATH_CALUDE_b_eq_one_sufficient_not_necessary_l411_41196


namespace NUMINAMATH_CALUDE_equation_equivalence_l411_41191

theorem equation_equivalence (p q : ℝ) 
  (hp_nonzero : p ≠ 0) (hp_not_five : p ≠ 5) 
  (hq_nonzero : q ≠ 0) (hq_not_seven : q ≠ 7) :
  (3 / p + 5 / q = 1 / 3) ↔ (p = 9 * q / (q - 15)) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l411_41191


namespace NUMINAMATH_CALUDE_polynomial_factorization_l411_41168

theorem polynomial_factorization (x : ℝ) : 
  x^4 + 2*x^3 - 9*x^2 - 2*x + 8 = (x + 4)*(x - 2)*(x + 1)*(x - 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l411_41168


namespace NUMINAMATH_CALUDE_sum_of_roots_l411_41179

theorem sum_of_roots (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 6 * x₁ - 1 = 0) → 
  (2 * x₂^2 + 6 * x₂ - 1 = 0) → 
  x₁ + x₂ = -3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l411_41179


namespace NUMINAMATH_CALUDE_equation_solution_l411_41127

theorem equation_solution :
  ∃ x : ℝ, 45 - 5 = 3 * x + 10 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l411_41127


namespace NUMINAMATH_CALUDE_medication_price_reduction_l411_41106

theorem medication_price_reduction (P : ℝ) (r : ℝ) : 
  P * (1 - r)^2 = 100 →
  P * (1 - r)^2 = P * 0.81 →
  0 < r →
  r < 1 →
  P * (1 - r)^3 = 90 := by
sorry

end NUMINAMATH_CALUDE_medication_price_reduction_l411_41106


namespace NUMINAMATH_CALUDE_power_of_seven_mod_thousand_l411_41150

theorem power_of_seven_mod_thousand : 7^2011 ≡ 7 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_thousand_l411_41150


namespace NUMINAMATH_CALUDE_power_2020_l411_41164

theorem power_2020 (m n : ℕ) (h1 : 3^m = 4) (h2 : 3^(m-4*n) = 4/81) : 
  2020^n = 2020 := by
  sorry

end NUMINAMATH_CALUDE_power_2020_l411_41164


namespace NUMINAMATH_CALUDE_kelly_apples_l411_41136

theorem kelly_apples (initial_apples target_apples : ℕ) 
  (h1 : initial_apples = 128) 
  (h2 : target_apples = 250) : 
  target_apples - initial_apples = 122 := by
  sorry

end NUMINAMATH_CALUDE_kelly_apples_l411_41136


namespace NUMINAMATH_CALUDE_milk_calculation_l411_41158

/-- The amount of milk Yuna's family drank as a fraction of the total -/
def milk_drunk : ℝ := 0.4

/-- The amount of leftover milk in liters -/
def leftover_milk : ℝ := 0.69

/-- The initial amount of milk in liters -/
def initial_milk : ℝ := 1.15

theorem milk_calculation (milk_drunk : ℝ) (leftover_milk : ℝ) (initial_milk : ℝ) :
  milk_drunk = 0.4 →
  leftover_milk = 0.69 →
  initial_milk = 1.15 →
  initial_milk * (1 - milk_drunk) = leftover_milk :=
by sorry

end NUMINAMATH_CALUDE_milk_calculation_l411_41158


namespace NUMINAMATH_CALUDE_pool_width_is_eight_feet_l411_41129

/-- Proves that the width of a rectangular pool is 8 feet given its length, depth, and volume. -/
theorem pool_width_is_eight_feet (length : ℝ) (depth : ℝ) (volume : ℝ) 
  (h_length : length = 10)
  (h_depth : depth = 6)
  (h_volume : volume = 480) :
  volume / (length * depth) = 8 := by
  sorry

#check pool_width_is_eight_feet

end NUMINAMATH_CALUDE_pool_width_is_eight_feet_l411_41129


namespace NUMINAMATH_CALUDE_election_votes_calculation_l411_41165

theorem election_votes_calculation (total_votes : ℕ) : 
  (∃ (candidate_votes rival_votes : ℕ),
    candidate_votes = (34 * total_votes) / 100 ∧ 
    rival_votes = candidate_votes + 640 ∧
    candidate_votes + rival_votes = total_votes) →
  total_votes = 2000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l411_41165


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l411_41167

/-- A positive term geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  geometric : ∀ n, a (n + 1) / a n = a 2 / a 1

/-- The theorem statement -/
theorem geometric_sequence_product (seq : GeometricSequence) 
  (h1 : 2 * (seq.a 1)^2 - 7 * (seq.a 1) + 6 = 0)
  (h2 : 2 * (seq.a 48)^2 - 7 * (seq.a 48) + 6 = 0) :
  seq.a 1 * seq.a 2 * seq.a 25 * seq.a 48 * seq.a 49 = 9 * Real.sqrt 3 := by
  sorry

#check geometric_sequence_product

end NUMINAMATH_CALUDE_geometric_sequence_product_l411_41167


namespace NUMINAMATH_CALUDE_election_winner_votes_l411_41133

theorem election_winner_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (58 : ℚ) / 100 * total_votes - (42 : ℚ) / 100 * total_votes = 288) :
  ⌊(58 : ℚ) / 100 * total_votes⌋ = 1044 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l411_41133


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l411_41195

/-- Given a triangle ABC with sides a, b, c, this structure represents the triangle and related points --/
structure TriangleWithIntersections where
  -- The lengths of the sides of triangle ABC
  a : ℝ
  b : ℝ
  c : ℝ
  -- The area of triangle ABC
  S_ABC : ℝ
  -- The area of hexagon PQRSTF
  S_PQRSTF : ℝ
  -- Assumption that a, b, c form a valid triangle
  triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem stating the relationship between the areas --/
theorem area_ratio_theorem (t : TriangleWithIntersections) :
  t.S_PQRSTF / t.S_ABC = 1 - (t.a * t.b + t.b * t.c + t.c * t.a) / (t.a + t.b + t.c)^2 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l411_41195


namespace NUMINAMATH_CALUDE_gcf_36_54_81_l411_41121

theorem gcf_36_54_81 : Nat.gcd 36 (Nat.gcd 54 81) = 9 := by sorry

end NUMINAMATH_CALUDE_gcf_36_54_81_l411_41121


namespace NUMINAMATH_CALUDE_order_of_variables_l411_41198

theorem order_of_variables (a b c d : ℝ) 
  (h1 : a > b) 
  (h2 : d > c) 
  (h3 : (c - a) * (c - b) < 0) 
  (h4 : (d - a) * (d - b) > 0) : 
  b < c ∧ c < a ∧ a < d :=
sorry

end NUMINAMATH_CALUDE_order_of_variables_l411_41198


namespace NUMINAMATH_CALUDE_jumping_contest_l411_41143

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) : 
  grasshopper_jump = 25 →
  frog_jump = grasshopper_jump + 32 →
  mouse_jump = frog_jump - 26 →
  mouse_jump = 31 := by
  sorry

end NUMINAMATH_CALUDE_jumping_contest_l411_41143


namespace NUMINAMATH_CALUDE_prob_first_class_is_072_l411_41169

/-- Represents a batch of products -/
structure ProductBatch where
  defectiveRate : ℝ
  firstClassRateAmongQualified : ℝ

/-- Calculates the probability of selecting a first-class product from a batch -/
def probabilityFirstClass (batch : ProductBatch) : ℝ :=
  (1 - batch.defectiveRate) * batch.firstClassRateAmongQualified

/-- Theorem: The probability of selecting a first-class product from the given batch is 0.72 -/
theorem prob_first_class_is_072 (batch : ProductBatch) 
    (h1 : batch.defectiveRate = 0.04)
    (h2 : batch.firstClassRateAmongQualified = 0.75) : 
    probabilityFirstClass batch = 0.72 := by
  sorry

#eval probabilityFirstClass { defectiveRate := 0.04, firstClassRateAmongQualified := 0.75 }

end NUMINAMATH_CALUDE_prob_first_class_is_072_l411_41169


namespace NUMINAMATH_CALUDE_optimal_strategy_minimizes_cost_l411_41159

/-- Represents the bookstore's ordering strategy --/
structure OrderStrategy where
  numOrders : ℕ
  copiesPerOrder : ℕ

/-- Calculates the total cost for a given order strategy --/
def totalCost (s : OrderStrategy) : ℝ :=
  let handlingCost := 30 * s.numOrders
  let storageCost := 40 * (s.copiesPerOrder / 1000) * s.numOrders / 2
  handlingCost + storageCost

/-- The optimal order strategy --/
def optimalStrategy : OrderStrategy :=
  { numOrders := 10, copiesPerOrder := 15000 }

/-- Theorem stating that the optimal strategy minimizes total cost --/
theorem optimal_strategy_minimizes_cost :
  ∀ s : OrderStrategy,
    s.numOrders * s.copiesPerOrder = 150000 →
    totalCost optimalStrategy ≤ totalCost s :=
by sorry

#check optimal_strategy_minimizes_cost

end NUMINAMATH_CALUDE_optimal_strategy_minimizes_cost_l411_41159


namespace NUMINAMATH_CALUDE_mad_hatter_march_hare_meeting_time_difference_l411_41176

/-- Represents a clock with a specific rate of time change per hour -/
structure Clock where
  rate : ℚ

/-- Calculates the actual time passed for a given clock time -/
def actual_time (c : Clock) (clock_time : ℚ) : ℚ :=
  clock_time * c.rate

theorem mad_hatter_march_hare_meeting_time_difference : 
  let mad_hatter_clock : Clock := ⟨60 / 75⟩
  let march_hare_clock : Clock := ⟨60 / 50⟩
  let meeting_time : ℚ := 5

  actual_time march_hare_clock meeting_time - actual_time mad_hatter_clock meeting_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_mad_hatter_march_hare_meeting_time_difference_l411_41176


namespace NUMINAMATH_CALUDE_sums_are_equal_l411_41189

def sum1 : ℕ :=
  1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888 + 999999999

def sum2 : ℕ :=
  9 + 98 + 987 + 9876 + 98765 + 987654 + 9876543 + 98765432 + 987654321

theorem sums_are_equal : sum1 = sum2 := by
  sorry

end NUMINAMATH_CALUDE_sums_are_equal_l411_41189


namespace NUMINAMATH_CALUDE_mod_17_equivalence_l411_41102

theorem mod_17_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ 42762 % 17 = n % 17 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_mod_17_equivalence_l411_41102


namespace NUMINAMATH_CALUDE_max_value_fraction_l411_41104

theorem max_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 3) :
  (x + y) / (x - 1) ≤ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_fraction_l411_41104


namespace NUMINAMATH_CALUDE_two_translations_complex_plane_l411_41181

theorem two_translations_complex_plane :
  let first_translation : ℂ → ℂ := λ z => z + ((-7 - Complex.I) - (-3 + 2 * Complex.I))
  let second_translation : ℂ → ℂ := λ z => z + (-10 - (-7 - Complex.I))
  let combined_translation := second_translation ∘ first_translation
  combined_translation (-4 + 5 * Complex.I) = -11 + 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_two_translations_complex_plane_l411_41181


namespace NUMINAMATH_CALUDE_K33_not_planar_l411_41146

/-- A bipartite graph with two sets of three vertices each --/
structure BipartiteGraph :=
  (left : Finset ℕ)
  (right : Finset ℕ)
  (edges : Set (ℕ × ℕ))

/-- The K₃,₃ graph --/
def K33 : BipartiteGraph :=
  { left := {1, 2, 3},
    right := {4, 5, 6},
    edges := {(1,4), (1,5), (1,6), (2,4), (2,5), (2,6), (3,4), (3,5), (3,6)} }

/-- A graph is planar if it can be drawn on a plane without edge crossings --/
def isPlanar (G : BipartiteGraph) : Prop := sorry

/-- Theorem: K₃,₃ is not planar --/
theorem K33_not_planar : ¬ isPlanar K33 := by
  sorry

end NUMINAMATH_CALUDE_K33_not_planar_l411_41146


namespace NUMINAMATH_CALUDE_amoeba_population_day_10_l411_41186

/-- The number of amoebas on day n, given an initial population of 3 and daily doubling. -/
def amoeba_population (n : ℕ) : ℕ := 3 * 2^n

/-- Theorem stating that after 10 days, the amoeba population is 3072. -/
theorem amoeba_population_day_10 : amoeba_population 10 = 3072 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_population_day_10_l411_41186


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l411_41119

theorem polynomial_factor_implies_coefficients 
  (a b : ℚ) 
  (h : ∃ (c d k : ℚ), ax^4 + bx^3 + 38*x^2 - 12*x + 15 = (3*x^2 - 2*x + 2)*(c*x^2 + d*x + k)) :
  a = -75/2 ∧ b = 59/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l411_41119


namespace NUMINAMATH_CALUDE_peanuts_added_l411_41123

theorem peanuts_added (initial_peanuts final_peanuts : ℕ) : 
  initial_peanuts = 4 →
  final_peanuts = 16 →
  final_peanuts - initial_peanuts = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_peanuts_added_l411_41123


namespace NUMINAMATH_CALUDE_total_cantaloupes_l411_41187

def fred_cantaloupes : ℕ := 38
def tim_cantaloupes : ℕ := 44
def susan_cantaloupes : ℕ := 57
def nancy_cantaloupes : ℕ := 25

theorem total_cantaloupes : 
  fred_cantaloupes + tim_cantaloupes + susan_cantaloupes + nancy_cantaloupes = 164 := by
  sorry

end NUMINAMATH_CALUDE_total_cantaloupes_l411_41187


namespace NUMINAMATH_CALUDE_trig_expression_equality_l411_41190

theorem trig_expression_equality : 
  (Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + Real.cos (160 * π / 180) * Real.cos (110 * π / 180)) /
  (Real.sin (24 * π / 180) * Real.cos (6 * π / 180) + Real.cos (156 * π / 180) * Real.cos (96 * π / 180)) =
  (1 - Real.sin (40 * π / 180)) / (1 - Real.sin (48 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l411_41190


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l411_41161

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q) →
  (a 1 + (2 * a 2 - a 1) / 2 = a 3 / 2) →
  (a 10 + a 11) / (a 8 + a 9) = 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l411_41161


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l411_41172

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def B : Set ℕ := {x | 2 ≤ x ∧ x < 6}

theorem intersection_of_A_and_B : A ∩ B = {2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l411_41172


namespace NUMINAMATH_CALUDE_pet_count_theorem_l411_41128

/-- Represents the pet ownership distribution in a community -/
structure PetOwnership where
  total_families : ℕ
  dog_cat_families : ℕ
  dog_cat_bird_families : ℕ
  dog_cat_bird_families2 : ℕ
  cat_fish_families : ℕ
  horse_share_families : ℕ

/-- Calculates the total number of each type of pet in the community -/
def count_pets (p : PetOwnership) : ℕ × ℕ × ℕ × ℕ × ℕ :=
  let dogs := p.dog_cat_families * 4 + p.dog_cat_bird_families * 3 + p.dog_cat_bird_families2 * 2
  let cats := p.dog_cat_families * 2 + p.dog_cat_bird_families * 2 + p.dog_cat_bird_families2 * 1 + p.cat_fish_families * 3
  let birds := p.dog_cat_bird_families * 1 + p.dog_cat_bird_families2 * 3
  let fish := p.cat_fish_families * 4
  let horses := p.horse_share_families / 3
  (dogs, cats, birds, fish, horses)

theorem pet_count_theorem (p : PetOwnership) 
  (h1 : p.total_families = 120)
  (h2 : p.dog_cat_families = 25)
  (h3 : p.dog_cat_bird_families = 30)
  (h4 : p.dog_cat_bird_families2 = 20)
  (h5 : p.cat_fish_families = p.total_families - p.dog_cat_families - p.dog_cat_bird_families - p.dog_cat_bird_families2)
  (h6 : p.horse_share_families = 15) :
  count_pets p = (230, 265, 90, 180, 5) := by
  sorry

#check pet_count_theorem

end NUMINAMATH_CALUDE_pet_count_theorem_l411_41128


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l411_41115

theorem expression_simplification_and_evaluation :
  let a : ℚ := -2
  let b : ℚ := 1/3
  2 * (a^2 * b - 2 * a * b) - 3 * (a^2 * b - 3 * a * b) + a^2 * b = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l411_41115


namespace NUMINAMATH_CALUDE_garden_length_l411_41197

/-- Proves that a rectangular garden with perimeter 80 meters and width 15 meters has a length of 25 meters -/
theorem garden_length (perimeter width : ℝ) (h1 : perimeter = 80) (h2 : width = 15) :
  let length := (perimeter / 2) - width
  length = 25 := by
sorry

end NUMINAMATH_CALUDE_garden_length_l411_41197


namespace NUMINAMATH_CALUDE_inverse_g_84_l411_41112

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 3

-- State the theorem
theorem inverse_g_84 : g⁻¹ 84 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_84_l411_41112


namespace NUMINAMATH_CALUDE_chord_equation_l411_41130

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the point M
def M : ℝ × ℝ := (2, 1)

-- Define a chord of the ellipse
def is_chord (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2

-- Define M as the midpoint of the chord
def M_bisects_chord (A B : ℝ × ℝ) : Prop :=
  M = ((A.1 + B.1)/2, (A.2 + B.2)/2)

-- The theorem to prove
theorem chord_equation :
  ∀ A B : ℝ × ℝ,
  is_chord A B →
  M_bisects_chord A B →
  ∃ k m : ℝ, k = -1/2 ∧ m = 4 ∧ 
    ∀ x y : ℝ, (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) → y = k*x + m :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_l411_41130


namespace NUMINAMATH_CALUDE_divisibility_theorem_l411_41126

theorem divisibility_theorem (a b c x y z : ℝ) :
  (a * y - b * x)^2 + (b * z - c * y)^2 + (c * x - a * z)^2 + (a * x + b * y + c * z)^2 =
  (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l411_41126


namespace NUMINAMATH_CALUDE_isosceles_triangle_cosine_l411_41182

def IsoscelesTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ b = c

def LargestAngleThreeTimesSmallest (a b c : ℝ) : Prop :=
  let cosSmallest := (b^2 + c^2 - a^2) / (2 * b * c)
  let cosLargest := (a^2 + b^2 - c^2) / (2 * a * b)
  cosLargest = 4 * cosSmallest^3 - 3 * cosSmallest

theorem isosceles_triangle_cosine (n : ℕ) :
  IsoscelesTriangle n (n + 1) (n + 1) →
  LargestAngleThreeTimesSmallest n (n + 1) (n + 1) →
  let cosSmallest := ((n + 1)^2 + (n + 1)^2 - n^2) / (2 * (n + 1) * (n + 1))
  cosSmallest = 7 / 9 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_cosine_l411_41182


namespace NUMINAMATH_CALUDE_product_of_half_and_two_thirds_l411_41132

theorem product_of_half_and_two_thirds (x y : ℚ) : 
  x = 1/2 → y = 2/3 → x * y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_half_and_two_thirds_l411_41132


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l411_41117

theorem ratio_of_numbers (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : a + b = 7 * (a - b)) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l411_41117


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l411_41118

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  ¬(∀ a b, a^2 > b^2 → a > b ∧ b > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l411_41118


namespace NUMINAMATH_CALUDE_counterexamples_exist_l411_41185

theorem counterexamples_exist : ∃ (a b c : ℝ),
  -- Statement A is not always true
  (a * b ≠ 0 ∧ a < b ∧ (1 / a) ≤ (1 / b)) ∧
  -- Statement C is not always true
  (a > b ∧ b > 0 ∧ ((b + 1) / (a + 1)) ≥ (b / a)) ∧
  -- Statement D is not always true
  (c < b ∧ b < a ∧ a * c < 0 ∧ c * b^2 ≥ a * b^2) :=
sorry

end NUMINAMATH_CALUDE_counterexamples_exist_l411_41185


namespace NUMINAMATH_CALUDE_james_weekly_earnings_l411_41100

/-- Calculates the weekly earnings from car rental -/
def weekly_earnings (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hourly_rate * hours_per_day * days_per_week

/-- Proves that James' weekly earnings from car rental are $640 -/
theorem james_weekly_earnings :
  weekly_earnings 20 8 4 = 640 := by
  sorry

end NUMINAMATH_CALUDE_james_weekly_earnings_l411_41100


namespace NUMINAMATH_CALUDE_work_completion_equality_prove_men_first_group_l411_41147

/-- The number of days it takes the first group to complete the work -/
def days_first_group : ℕ := 30

/-- The number of men in the second group -/
def men_second_group : ℕ := 10

/-- The number of days it takes the second group to complete the work -/
def days_second_group : ℕ := 36

/-- The number of men in the first group -/
def men_first_group : ℕ := 12

theorem work_completion_equality :
  men_first_group * days_first_group = men_second_group * days_second_group :=
by sorry

theorem prove_men_first_group :
  men_first_group = (men_second_group * days_second_group) / days_first_group :=
by sorry

end NUMINAMATH_CALUDE_work_completion_equality_prove_men_first_group_l411_41147


namespace NUMINAMATH_CALUDE_ceiling_negative_five_thirds_squared_l411_41139

theorem ceiling_negative_five_thirds_squared : ⌈(-5/3)^2⌉ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_five_thirds_squared_l411_41139


namespace NUMINAMATH_CALUDE_power_product_equals_four_l411_41174

theorem power_product_equals_four (a b : ℕ+) (h : (3 ^ a.val) ^ b.val = 3 ^ 3) :
  3 ^ a.val * 3 ^ b.val = 3 ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_four_l411_41174


namespace NUMINAMATH_CALUDE_half_AB_equals_neg_two_two_l411_41148

def OA : Fin 2 → ℝ := ![1, -2]
def OB : Fin 2 → ℝ := ![-3, 2]

theorem half_AB_equals_neg_two_two : 
  (1 / 2 : ℝ) • (OB - OA) = ![(-2 : ℝ), (2 : ℝ)] := by sorry

end NUMINAMATH_CALUDE_half_AB_equals_neg_two_two_l411_41148


namespace NUMINAMATH_CALUDE_square_root_of_nine_l411_41175

theorem square_root_of_nine : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l411_41175


namespace NUMINAMATH_CALUDE_age_problem_l411_41160

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 42 → 
  b = 16 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l411_41160


namespace NUMINAMATH_CALUDE_largest_n_with_conditions_l411_41144

theorem largest_n_with_conditions : 
  ∃ (m : ℤ), 139^2 = m^3 - 1 ∧ 
  ∃ (a : ℤ), 2 * 139 + 83 = a^2 ∧
  ∀ (n : ℤ), n > 139 → 
    (∀ (m : ℤ), n^2 ≠ m^3 - 1 ∨ ¬∃ (a : ℤ), 2 * n + 83 = a^2) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_conditions_l411_41144


namespace NUMINAMATH_CALUDE_dice_probability_l411_41120

/-- The number of dice --/
def n : ℕ := 8

/-- The number of sides on each die --/
def sides : ℕ := 8

/-- The number of favorable outcomes (dice showing a number less than 5) --/
def k : ℕ := 4

/-- The probability of a single die showing a number less than 5 --/
def p : ℚ := 1/2

/-- The probability of exactly k out of n dice showing a number less than 5 --/
def probability : ℚ := (n.choose k) * p^k * (1-p)^(n-k)

theorem dice_probability : probability = 35/128 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l411_41120


namespace NUMINAMATH_CALUDE_soap_boxes_in_carton_l411_41138

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the maximum number of smaller boxes that can fit in a larger box -/
def maxBoxesFit (largeBox smallBox : BoxDimensions) : ℕ :=
  (boxVolume largeBox) / (boxVolume smallBox)

theorem soap_boxes_in_carton :
  let carton : BoxDimensions := ⟨25, 42, 60⟩
  let soapBox : BoxDimensions := ⟨7, 12, 5⟩
  maxBoxesFit carton soapBox = 150 := by
  sorry

end NUMINAMATH_CALUDE_soap_boxes_in_carton_l411_41138


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l411_41180

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if two points are symmetric with respect to the x-axis. -/
def symmetricXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- The theorem stating that the point N(-2, -1) is symmetric to M(-2, 1) with respect to the x-axis. -/
theorem symmetric_point_x_axis : 
  let M : Point := ⟨-2, 1⟩
  let N : Point := ⟨-2, -1⟩
  symmetricXAxis M N := by
  sorry

#check symmetric_point_x_axis

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l411_41180


namespace NUMINAMATH_CALUDE_aunt_may_milk_problem_l411_41137

theorem aunt_may_milk_problem (morning_milk evening_milk sold_milk leftover_milk : ℕ) 
  (h1 : morning_milk = 365)
  (h2 : evening_milk = 380)
  (h3 : sold_milk = 612)
  (h4 : leftover_milk = 15) :
  morning_milk + evening_milk - sold_milk + leftover_milk = 148 :=
by sorry

end NUMINAMATH_CALUDE_aunt_may_milk_problem_l411_41137


namespace NUMINAMATH_CALUDE_rectangle_opposite_sides_l411_41113

/-- A parallelogram is a quadrilateral with opposite sides parallel and equal. -/
structure Parallelogram where
  opposite_sides_parallel : Bool
  opposite_sides_equal : Bool

/-- A rectangle is a special case of a parallelogram with right angles. -/
structure Rectangle extends Parallelogram where
  right_angles : Bool

/-- Deductive reasoning is a method of logical reasoning that uses general rules to reach a specific conclusion. -/
def DeductiveReasoning : Prop := True

/-- The reasoning method used in the given statement. -/
def reasoning_method : Prop := DeductiveReasoning

theorem rectangle_opposite_sides (p : Parallelogram) (r : Rectangle) :
  p.opposite_sides_parallel ∧ p.opposite_sides_equal →
  r.opposite_sides_parallel ∧ r.opposite_sides_equal →
  reasoning_method := by sorry

end NUMINAMATH_CALUDE_rectangle_opposite_sides_l411_41113


namespace NUMINAMATH_CALUDE_trees_planted_l411_41114

theorem trees_planted (initial_trees final_trees : ℕ) (h1 : initial_trees = 13) (h2 : final_trees = 25) :
  final_trees - initial_trees = 12 := by
  sorry

end NUMINAMATH_CALUDE_trees_planted_l411_41114


namespace NUMINAMATH_CALUDE_min_point_of_translated_graph_l411_41107

-- Define the function
def f (x : ℝ) : ℝ := |x - 1|^2 - 7

-- State the theorem
theorem min_point_of_translated_graph :
  ∃! p : ℝ × ℝ, p.1 = 1 ∧ p.2 = -7 ∧ ∀ x : ℝ, f x ≥ f p.1 :=
sorry

end NUMINAMATH_CALUDE_min_point_of_translated_graph_l411_41107


namespace NUMINAMATH_CALUDE_median_same_variance_decreases_l411_41135

def original_data : List ℝ := [2, 2, 4, 4]
def new_data : List ℝ := [2, 2, 3, 4, 4]

def median (l : List ℝ) : ℝ := sorry
def variance (l : List ℝ) : ℝ := sorry

theorem median_same_variance_decreases :
  median original_data = median new_data ∧
  variance new_data < variance original_data := by sorry

end NUMINAMATH_CALUDE_median_same_variance_decreases_l411_41135


namespace NUMINAMATH_CALUDE_line_equation_l411_41134

/-- Circle C with equation x^2 + (y-1)^2 = 5 -/
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

/-- Line l with equation mx - y + 1 - m = 0 -/
def line_l (m x y : ℝ) : Prop := m*x - y + 1 - m = 0

/-- Point P(1,1) -/
def point_P : ℝ × ℝ := (1, 1)

/-- Chord AB of circle C -/
def chord_AB (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  circle_C x₁ y₁ ∧ circle_C x₂ y₂

/-- Point P divides chord AB with ratio AP:PB = 1:2 -/
def divides_chord (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  chord_AB x₁ y₁ x₂ y₂ ∧ 2*(1 - x₁) = x₂ - 1 ∧ 2*(1 - y₁) = y₂ - 1

theorem line_equation (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, line_l m 1 1 ∧ divides_chord x₁ y₁ x₂ y₂) →
  (line_l 1 x y ∨ line_l (-1) x y) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l411_41134


namespace NUMINAMATH_CALUDE_cloth_sale_problem_l411_41166

/-- Given a shopkeeper's cloth sale scenario, calculate the number of metres sold. -/
theorem cloth_sale_problem (total_selling_price : ℕ) (loss_per_metre : ℕ) (cost_price_per_metre : ℕ) : 
  total_selling_price = 9000 →
  loss_per_metre = 6 →
  cost_price_per_metre = 36 →
  (total_selling_price / (cost_price_per_metre - loss_per_metre) : ℕ) = 300 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_problem_l411_41166


namespace NUMINAMATH_CALUDE_average_of_abc_is_three_l411_41124

theorem average_of_abc_is_three (A B C : ℝ) 
  (eq1 : 2012 * C - 4024 * A = 8048)
  (eq2 : 2012 * B + 6036 * A = 10010) :
  (A + B + C) / 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_abc_is_three_l411_41124
