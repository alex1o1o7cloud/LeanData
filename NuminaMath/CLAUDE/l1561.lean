import Mathlib

namespace f_shifted_up_is_g_l1561_156167

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the shifted function g
def g : ℝ → ℝ := sorry

-- Theorem stating that g is f shifted up by 1
theorem f_shifted_up_is_g : ∀ x : ℝ, g x = f x + 1 := by sorry

end f_shifted_up_is_g_l1561_156167


namespace linear_equation_solution_l1561_156189

theorem linear_equation_solution (a b m : ℝ) : 
  (∀ y, (a + b) * y^2 - y^((1/3)*a + 2) + 5 = 0 → (a + b = 0 ∧ (1/3)*a + 2 = 1)) →
  ((a + 2)/6 - (a - 1)/2 + 3 = a - (2*a - m)/6) →
  |a - b| - |b - m| = -32 := by
  sorry

end linear_equation_solution_l1561_156189


namespace oranges_taken_l1561_156114

theorem oranges_taken (initial : ℕ) (remaining : ℕ) (taken : ℕ) 
  (h1 : initial = 60)
  (h2 : remaining = 25)
  (h3 : initial = remaining + taken) : 
  taken = 35 := by
  sorry

end oranges_taken_l1561_156114


namespace binary_sum_equals_669_l1561_156155

/-- Represents a binary number as a list of booleans, where true represents 1 and false represents 0 -/
def BinaryNumber := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNumber) : ℕ :=
  b.foldl (fun acc digit => 2 * acc + if digit then 1 else 0) 0

/-- The binary number 111111111₂ -/
def b1 : BinaryNumber := [true, true, true, true, true, true, true, true, true]

/-- The binary number 1111111₂ -/
def b2 : BinaryNumber := [true, true, true, true, true, true, true]

/-- The binary number 11111₂ -/
def b3 : BinaryNumber := [true, true, true, true, true]

theorem binary_sum_equals_669 :
  binary_to_decimal b1 + binary_to_decimal b2 + binary_to_decimal b3 = 669 := by
  sorry

end binary_sum_equals_669_l1561_156155


namespace cos_minus_sin_seventeen_fourths_pi_equals_sqrt_two_l1561_156156

theorem cos_minus_sin_seventeen_fourths_pi_equals_sqrt_two :
  Real.cos (-17/4 * Real.pi) - Real.sin (-17/4 * Real.pi) = Real.sqrt 2 := by
  sorry

end cos_minus_sin_seventeen_fourths_pi_equals_sqrt_two_l1561_156156


namespace largest_divisor_of_consecutive_even_integers_l1561_156154

theorem largest_divisor_of_consecutive_even_integers (n : ℕ) : 
  ∃ k : ℕ, (2*n) * (2*n + 2) * (2*n + 4) = 48 * k :=
sorry

end largest_divisor_of_consecutive_even_integers_l1561_156154


namespace abc_area_is_sqrt3_over_12_l1561_156150

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the points M, P, and O
def M (t : Triangle) : ℝ × ℝ := sorry
def P (t : Triangle) : ℝ × ℝ := sorry
def O (t : Triangle) : ℝ × ℝ := sorry

-- Define the similarity of triangles BOM and AOP
def triangles_similar (t : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    dist (t.B) (O t) / dist (t.A) (O t) = k ∧
    dist (O t) (M t) / dist (O t) (P t) = k ∧
    dist (t.B) (M t) / dist (t.A) (P t) = k

-- Define the condition BO = (1 + √3) OP
def bo_op_relation (t : Triangle) : Prop :=
  dist (t.B) (O t) = (1 + Real.sqrt 3) * dist (O t) (P t)

-- Define the condition BC = 1
def bc_length (t : Triangle) : Prop :=
  dist (t.B) (t.C) = 1

-- Define the area of the triangle
def triangle_area (t : Triangle) : ℝ := sorry

-- State the theorem
theorem abc_area_is_sqrt3_over_12 (t : Triangle) 
  (h1 : triangles_similar t) 
  (h2 : bo_op_relation t) 
  (h3 : bc_length t) : 
  triangle_area t = Real.sqrt 3 / 12 := by sorry

end abc_area_is_sqrt3_over_12_l1561_156150


namespace always_two_real_roots_root_less_than_one_implies_k_negative_l1561_156103

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 - (k+3)*x + 2*k + 2

-- Theorem 1: The equation always has two real roots
theorem always_two_real_roots (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic x₁ k = 0 ∧ quadratic x₂ k = 0 :=
sorry

-- Theorem 2: When one root is less than 1, k < 0
theorem root_less_than_one_implies_k_negative (k : ℝ) :
  (∃ x : ℝ, x < 1 ∧ quadratic x k = 0) → k < 0 :=
sorry

end always_two_real_roots_root_less_than_one_implies_k_negative_l1561_156103


namespace jerry_age_l1561_156123

/-- Given that Mickey's age is 18 and Mickey's age is 2 years less than 400% of Jerry's age,
    prove that Jerry's age is 5. -/
theorem jerry_age (mickey_age jerry_age : ℕ) 
  (h1 : mickey_age = 18)
  (h2 : mickey_age = 4 * jerry_age - 2) : 
  jerry_age = 5 := by
sorry

end jerry_age_l1561_156123


namespace min_value_3x_4y_l1561_156176

theorem min_value_3x_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = 5 * x₀ * y₀ ∧ 3 * x₀ + 4 * y₀ = 5 :=
sorry

end min_value_3x_4y_l1561_156176


namespace complex_fraction_simplification_l1561_156159

theorem complex_fraction_simplification :
  (2 - Complex.I) / (1 + 2 * Complex.I) = -Complex.I := by
  sorry

end complex_fraction_simplification_l1561_156159


namespace seashells_given_to_joan_l1561_156194

/-- Given that Sam initially found 35 seashells and now has 17 seashells,
    prove that the number of seashells he gave to Joan is 18. -/
theorem seashells_given_to_joan 
  (initial_seashells : ℕ) 
  (current_seashells : ℕ) 
  (h1 : initial_seashells = 35) 
  (h2 : current_seashells = 17) : 
  initial_seashells - current_seashells = 18 := by
sorry

end seashells_given_to_joan_l1561_156194


namespace pure_imaginary_complex_number_l1561_156111

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := m * (m - 1) + (m - 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = 0 := by
  sorry

end pure_imaginary_complex_number_l1561_156111


namespace max_value_problem_l1561_156102

theorem max_value_problem (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  (∃ (x y z : ℝ), 9 * x^2 + 4 * y^2 + 25 * z^2 = 1 ∧ 3 * x + 4 * y + 5 * z > 3 * a + 4 * b + 5 * c) →
  3 * a + 4 * b + 5 * c ≤ Real.sqrt 6 :=
by sorry

end max_value_problem_l1561_156102


namespace afternoon_emails_l1561_156129

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 9

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 7

/-- The difference between morning and evening emails -/
def email_difference : ℕ := 2

/-- Theorem stating that Jack received 7 emails in the afternoon -/
theorem afternoon_emails : ℕ := by
  sorry

end afternoon_emails_l1561_156129


namespace replaced_girl_weight_l1561_156190

/-- Given a group of girls where replacing one with a heavier girl increases the average weight, 
    this theorem proves the weight of the replaced girl. -/
theorem replaced_girl_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (new_girl_weight : ℝ) 
  (average_increase : ℝ) 
  (h1 : n = 10)
  (h2 : new_girl_weight = 100)
  (h3 : average_increase = 5) :
  initial_average * n + new_girl_weight - (initial_average * n + n * average_increase) = 50 :=
by
  sorry

#check replaced_girl_weight

end replaced_girl_weight_l1561_156190


namespace factorial_sum_not_end_1990_l1561_156181

theorem factorial_sum_not_end_1990 (m n : ℕ) : (m.factorial + n.factorial) % 10000 ≠ 1990 := by
  sorry

end factorial_sum_not_end_1990_l1561_156181


namespace complex_number_in_first_quadrant_l1561_156169

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (1 + 3*I) / (3 + I)
  0 < z.re ∧ 0 < z.im :=
by
  sorry

end complex_number_in_first_quadrant_l1561_156169


namespace area_bounded_by_curves_l1561_156177

/-- The area of the region bounded by x = √(e^y - 1), x = 0, and y = ln 2 -/
theorem area_bounded_by_curves : ∃ (S : ℝ),
  (∀ x y : ℝ, x = Real.sqrt (Real.exp y - 1) → 
    0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ Real.log 2) →
  S = ∫ x in (0)..(1), (Real.log 2 - Real.log (x^2 + 1)) →
  S = 2 - π / 2 := by
  sorry

end area_bounded_by_curves_l1561_156177


namespace expression_simplification_l1561_156125

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x + 2) / (x^2 - 2*x) / ((8*x) / (x - 2) + x - 2) = 1 := by
  sorry

end expression_simplification_l1561_156125


namespace ones_digit_of_3_to_53_l1561_156179

theorem ones_digit_of_3_to_53 : (3^53 : ℕ) % 10 = 3 := by sorry

end ones_digit_of_3_to_53_l1561_156179


namespace five_digit_palindrome_digits_l1561_156191

/-- A function that calculates the number of 5-digit palindromes that can be formed
    using n distinct digits -/
def palindrome_count (n : ℕ) : ℕ := n * n * n

/-- The theorem stating that if there are 125 possible 5-digit palindromes formed
    using some distinct digits, then the number of distinct digits is 5 -/
theorem five_digit_palindrome_digits :
  (∃ (n : ℕ), n > 0 ∧ palindrome_count n = 125) →
  (∃ (n : ℕ), n > 0 ∧ palindrome_count n = 125 ∧ n = 5) :=
by sorry

end five_digit_palindrome_digits_l1561_156191


namespace equation_has_solution_equation_has_unique_solution_l1561_156144

-- Define the equation
def equation (a x : ℝ) : Prop :=
  (Real.log x / Real.log a) / (Real.log 2 / Real.log a) +
  (Real.log (2*a - x) / Real.log x) / (Real.log 2 / Real.log x) =
  1 / (Real.log 2 / Real.log (a^2 - 1))

-- Theorem for the first question
theorem equation_has_solution (a : ℝ) :
  (∃ x, equation a x) ↔ (a > 1 ∧ a ≠ Real.sqrt 2) :=
sorry

-- Theorem for the second question
theorem equation_has_unique_solution (a : ℝ) :
  (∃! x, equation a x) ↔ a = 2 :=
sorry

end equation_has_solution_equation_has_unique_solution_l1561_156144


namespace quadratic_equation_properties_l1561_156116

theorem quadratic_equation_properties (m : ℝ) :
  let f := fun x => m * x^2 - 4 * x + 1
  (∃ x : ℝ, f x = 0) →
  (f 1 = 0 → m = 3) ∧
  (m ≠ 0 → m ≤ 4) :=
by sorry

end quadratic_equation_properties_l1561_156116


namespace car_wash_rate_l1561_156146

def babysitting_families : ℕ := 4
def babysitting_rate : ℕ := 30
def cars_washed : ℕ := 5
def total_raised : ℕ := 180

theorem car_wash_rate :
  (total_raised - babysitting_families * babysitting_rate) / cars_washed = 12 := by
  sorry

end car_wash_rate_l1561_156146


namespace range_of_a_l1561_156185

/-- A decreasing function defined on (-∞, 3] -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x > f y) ∧ (∀ x, f x ≤ 3)

theorem range_of_a (f : ℝ → ℝ) (h_f : DecreasingFunction f)
    (h_ineq : ∀ x a : ℝ, f (a^2 - Real.sin x) ≤ f (a + 1 + Real.cos x ^ 2)) :
    ∀ a : ℝ, a ∈ Set.Icc (-Real.sqrt 2) ((1 - Real.sqrt 10) / 2) :=
  sorry

end range_of_a_l1561_156185


namespace equation_solutions_l1561_156135

theorem equation_solutions : 
  {x : ℝ | (x - 1) * (x - 3) * (x - 5) * (x - 6) * (x - 3) * (x - 1) / 
           ((x - 3) * (x - 6) * (x - 3)) = 2 ∧ 
           x ≠ 3 ∧ x ≠ 6} = 
  {2 + Real.sqrt 2, 2 - Real.sqrt 2} := by
sorry

end equation_solutions_l1561_156135


namespace circular_garden_radius_l1561_156182

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1 / 5) * π * r^2 → r = 10 := by
  sorry

end circular_garden_radius_l1561_156182


namespace solution_set_when_m_zero_solution_set_all_reals_l1561_156145

/-- The quadratic inequality in question -/
def quadratic_inequality (m : ℝ) (x : ℝ) : Prop :=
  (m - 1) * x^2 + (m - 1) * x + 2 > 0

/-- The solution set when m = 0 -/
theorem solution_set_when_m_zero :
  {x : ℝ | quadratic_inequality 0 x} = Set.Ioo (-2) 1 := by sorry

/-- The condition for the solution set to be all real numbers -/
theorem solution_set_all_reals (m : ℝ) :
  ({x : ℝ | quadratic_inequality m x} = Set.univ) ↔ (m ∈ Set.Icc 1 9) := by sorry

end solution_set_when_m_zero_solution_set_all_reals_l1561_156145


namespace chain_store_max_profit_l1561_156148

/-- Annual profit function for a chain store -/
def L (x a : ℝ) : ℝ := (x - 4 - a) * (10 - x)^2

/-- Maximum annual profit for the chain store -/
theorem chain_store_max_profit (a : ℝ) (ha : 1 ≤ a ∧ a ≤ 3) :
  ∃ (L_max : ℝ),
    (∀ x, 7 ≤ x → x ≤ 9 → L x a ≤ L_max) ∧
    ((1 ≤ a ∧ a ≤ 3/2 → L_max = 27 - 9*a) ∧
     (3/2 < a ∧ a ≤ 3 → L_max = 4*(2 - a/3)^3)) :=
sorry

end chain_store_max_profit_l1561_156148


namespace fishing_loss_fraction_l1561_156127

theorem fishing_loss_fraction (jordan_catch : ℕ) (perry_catch : ℕ) (remaining : ℕ) : 
  jordan_catch = 4 →
  perry_catch = 2 * jordan_catch →
  remaining = 9 →
  (jordan_catch + perry_catch - remaining : ℚ) / (jordan_catch + perry_catch) = 1/4 :=
by sorry

end fishing_loss_fraction_l1561_156127


namespace eating_contest_l1561_156128

/-- Eating contest problem -/
theorem eating_contest (hot_dog_weight burger_weight pie_weight : ℕ)
  (noah_burgers jacob_pies mason_hotdog_weight : ℕ) :
  hot_dog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  jacob_pies = noah_burgers - 3 →
  noah_burgers = 8 →
  mason_hotdog_weight = 30 →
  mason_hotdog_weight / hot_dog_weight = 15 :=
by sorry

end eating_contest_l1561_156128


namespace fourth_root_equation_solutions_l1561_156137

theorem fourth_root_equation_solutions : 
  {x : ℝ | x > 0 ∧ x^(1/4) = 15 / (8 - x^(1/4))} = {81, 625} := by sorry

end fourth_root_equation_solutions_l1561_156137


namespace solution_satisfies_system_l1561_156132

open Real

noncomputable def x (t : ℝ) : ℝ := exp (2 * t) * (-2 * cos t + sin t) + 2

noncomputable def y (t : ℝ) : ℝ := exp (2 * t) * (-cos t + 3 * sin t) + 3

theorem solution_satisfies_system :
  (∀ t, deriv x t = x t + y t - 3) ∧
  (∀ t, deriv y t = -2 * x t + 3 * y t + 1) ∧
  x 0 = 0 ∧
  y 0 = 0 :=
sorry

end solution_satisfies_system_l1561_156132


namespace simplify_expression_l1561_156106

theorem simplify_expression (x : ℝ) (h1 : x ≠ -3) (h2 : x ≠ 1) :
  (x - 5 + 16 / (x + 3)) / ((x - 1) / (x^2 - 9)) = x^2 - 4*x + 3 := by
  sorry

end simplify_expression_l1561_156106


namespace y_derivative_l1561_156199

noncomputable def y (x : ℝ) : ℝ := 
  (3 * x^2 - 4 * x + 2) * Real.sqrt (9 * x^2 - 12 * x + 3) + 
  (3 * x - 2)^4 * Real.arcsin (1 / (3 * x - 2))

theorem y_derivative (x : ℝ) (h : 3 * x - 2 > 0) : 
  deriv y x = 12 * (3 * x - 2)^3 * Real.arcsin (1 / (3 * x - 2)) := by
  sorry

end y_derivative_l1561_156199


namespace correct_number_of_children_l1561_156161

/-- The number of crayons each child has -/
def crayons_per_child : ℕ := 5

/-- The total number of crayons -/
def total_crayons : ℕ := 50

/-- The number of children -/
def number_of_children : ℕ := total_crayons / crayons_per_child

theorem correct_number_of_children : number_of_children = 10 := by
  sorry

end correct_number_of_children_l1561_156161


namespace tournament_max_wins_l1561_156193

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Minimum number of participants required for n wins -/
def f (n : ℕ) : ℕ := fib (n + 2)

/-- Tournament properties -/
structure Tournament :=
  (participants : ℕ)
  (one_match_at_a_time : Bool)
  (loser_drops_out : Bool)
  (max_win_diff : ℕ)

/-- Main theorem -/
theorem tournament_max_wins (t : Tournament) (h1 : t.participants = 55) 
  (h2 : t.one_match_at_a_time = true) (h3 : t.loser_drops_out = true) 
  (h4 : t.max_win_diff = 1) : 
  (∃ (n : ℕ), f n ≤ t.participants ∧ f (n + 1) > t.participants ∧ n = 8) :=
sorry

end tournament_max_wins_l1561_156193


namespace simplify_sqrt_expression_l1561_156164

theorem simplify_sqrt_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end simplify_sqrt_expression_l1561_156164


namespace noelle_homework_assignments_l1561_156171

/-- The number of homework points Noelle needs to earn -/
def total_points : ℕ := 30

/-- The number of points for which one assignment is required per point -/
def first_tier_points : ℕ := 5

/-- The number of points for which two assignments are required per point -/
def second_tier_points : ℕ := 10

/-- The number of assignments required for each point in the first tier -/
def first_tier_assignments_per_point : ℕ := 1

/-- The number of assignments required for each point in the second tier -/
def second_tier_assignments_per_point : ℕ := 2

/-- The number of assignments required for each point after the first and second tiers -/
def third_tier_assignments_per_point : ℕ := 3

/-- The total number of assignments Noelle needs to complete -/
def total_assignments : ℕ := 
  first_tier_points * first_tier_assignments_per_point +
  second_tier_points * second_tier_assignments_per_point +
  (total_points - first_tier_points - second_tier_points) * third_tier_assignments_per_point

theorem noelle_homework_assignments : total_assignments = 70 := by
  sorry

end noelle_homework_assignments_l1561_156171


namespace joan_seashells_l1561_156184

/-- The number of seashells Joan gave to Sam -/
def seashells_given : ℕ := 43

/-- The number of seashells Joan has left -/
def seashells_left : ℕ := 27

/-- The total number of seashells Joan found originally -/
def total_seashells : ℕ := seashells_given + seashells_left

theorem joan_seashells : total_seashells = 70 := by
  sorry

end joan_seashells_l1561_156184


namespace stratified_sample_composition_l1561_156122

def total_students : ℕ := 2700
def freshmen : ℕ := 900
def sophomores : ℕ := 1200
def juniors : ℕ := 600
def sample_size : ℕ := 135

theorem stratified_sample_composition :
  let freshmen_sample := (freshmen * sample_size) / total_students
  let sophomores_sample := (sophomores * sample_size) / total_students
  let juniors_sample := (juniors * sample_size) / total_students
  freshmen_sample = 45 ∧ sophomores_sample = 60 ∧ juniors_sample = 30 :=
by sorry

end stratified_sample_composition_l1561_156122


namespace simplify_and_evaluate_l1561_156124

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -4) (h2 : b = 1/2) :
  b * (a + b) + (-a + b) * (-a - b) - a^2 = -2 := by
  sorry

end simplify_and_evaluate_l1561_156124


namespace square_sum_from_difference_and_product_l1561_156136

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 := by
  sorry

end square_sum_from_difference_and_product_l1561_156136


namespace playground_total_l1561_156195

/-- The number of children on the playground at recess -/
def total_children (soccer_boys soccer_girls swings_boys swings_girls snacks_boys snacks_girls : ℕ) : ℕ :=
  soccer_boys + soccer_girls + swings_boys + swings_girls + snacks_boys + snacks_girls

/-- Theorem stating the total number of children on the playground -/
theorem playground_total :
  total_children 27 35 15 20 10 5 = 112 := by
  sorry

end playground_total_l1561_156195


namespace polynomial_coefficient_G_l1561_156158

-- Define the polynomial p(z)
def p (z E F G H I : ℤ) : ℤ := z^7 - 13*z^6 + E*z^5 + F*z^4 + G*z^3 + H*z^2 + I*z + 36

-- Define the property that all roots are positive integers
def all_roots_positive_integers (p : ℤ → ℤ) : Prop :=
  ∀ z : ℤ, p z = 0 → z > 0

-- Theorem statement
theorem polynomial_coefficient_G (E F G H I : ℤ) :
  all_roots_positive_integers (p · E F G H I) →
  G = -82 := by
  sorry


end polynomial_coefficient_G_l1561_156158


namespace min_value_theorem_l1561_156117

/-- The function f(x) = |x + a| + |x - b| -/
def f (a b x : ℝ) : ℝ := |x + a| + |x - b|

/-- The theorem statement -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 4) (hmin_exists : ∃ x, f a b x = 4) :
  (a + b = 4) ∧ 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 4 → 1/4 * x^2 + 1/9 * y^2 ≥ 16/13) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 4 ∧ 1/4 * x^2 + 1/9 * y^2 = 16/13) :=
by sorry

end min_value_theorem_l1561_156117


namespace equal_probability_l1561_156162

def num_dice : ℕ := 8
def min_value : ℕ := 2
def max_value : ℕ := 7

def sum_probability (sum : ℕ) : ℝ :=
  sorry

theorem equal_probability : sum_probability 20 = sum_probability 52 := by
  sorry

end equal_probability_l1561_156162


namespace bicycle_cost_price_l1561_156160

/-- The cost price of a bicycle for seller A, given the following conditions:
  - A sells the bicycle to B at a profit of 20%
  - B sells it to C at a profit of 25%
  - C pays Rs. 225 for the bicycle
-/
theorem bicycle_cost_price (profit_A_to_B : ℝ) (profit_B_to_C : ℝ) (price_C : ℝ) :
  profit_A_to_B = 0.20 →
  profit_B_to_C = 0.25 →
  price_C = 225 →
  ∃ (cost_price_A : ℝ), cost_price_A = 150 ∧
    price_C = cost_price_A * (1 + profit_A_to_B) * (1 + profit_B_to_C) :=
by sorry

end bicycle_cost_price_l1561_156160


namespace certain_number_problem_l1561_156157

theorem certain_number_problem (n x : ℝ) (h1 : 4 / (n + 3 / x) = 1) (h2 : x = 1) : n = 1 := by
  sorry

end certain_number_problem_l1561_156157


namespace y_greater_than_x_l1561_156140

theorem y_greater_than_x (x y : ℝ) (h1 : x + y > 2*x) (h2 : x - y < 2*y) : y > x := by
  sorry

end y_greater_than_x_l1561_156140


namespace eggs_taken_away_l1561_156153

/-- Proof that the number of eggs Amy took away is the difference between Virginia's initial and final number of eggs -/
theorem eggs_taken_away (initial_eggs final_eggs : ℕ) (h1 : initial_eggs = 96) (h2 : final_eggs = 93) :
  initial_eggs - final_eggs = 3 := by
  sorry

end eggs_taken_away_l1561_156153


namespace investment_income_l1561_156141

/-- Proves that an investment of $6800 in a 60% stock at a price of 136 yields an annual income of $3000 -/
theorem investment_income (investment : ℝ) (stock_percentage : ℝ) (stock_price : ℝ) (annual_income : ℝ) : 
  investment = 6800 ∧ 
  stock_percentage = 0.60 ∧ 
  stock_price = 136 ∧ 
  annual_income = 3000 → 
  investment * (stock_percentage / stock_price) = annual_income :=
by sorry

end investment_income_l1561_156141


namespace negation_equivalence_l1561_156180

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 2*x + 4 > 0) ↔ (∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) := by
  sorry

end negation_equivalence_l1561_156180


namespace smallest_positive_period_l1561_156100

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the symmetric points
variable (a b y₀ : ℝ)

-- Define the symmetry property
def isSymmetric (f : ℝ → ℝ) (x₁ x₂ y : ℝ) : Prop :=
  ∀ t, f (x₁ - t) = 2 * y - f (x₂ + t)

-- State the theorem
theorem smallest_positive_period
  (h₁ : isSymmetric f a a y₀)
  (h₂ : isSymmetric f b b y₀)
  (h₃ : ∀ x, a < x → x < b → ¬ isSymmetric f x x y₀)
  (h₄ : a < b) :
  ∃ T, T > 0 ∧ (∀ x, f (x + T) = f x) ∧
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = 2 * (b - a) :=
sorry

end smallest_positive_period_l1561_156100


namespace probability_different_colors_is_seven_ninths_l1561_156101

/-- The number of color options for socks -/
def sock_colors : ℕ := 3

/-- The number of color options for headband -/
def headband_colors : ℕ := 3

/-- The number of colors shared between socks and headband options -/
def shared_colors : ℕ := 1

/-- The total number of possible combinations -/
def total_combinations : ℕ := sock_colors * headband_colors

/-- The number of combinations where socks and headband have different colors -/
def different_color_combinations : ℕ := 
  sock_colors * headband_colors - sock_colors * shared_colors

/-- The probability of selecting different colors for socks and headband -/
def probability_different_colors : ℚ := 
  different_color_combinations / total_combinations

theorem probability_different_colors_is_seven_ninths : 
  probability_different_colors = 7 / 9 := by
  sorry

end probability_different_colors_is_seven_ninths_l1561_156101


namespace min_distance_to_2i_l1561_156192

theorem min_distance_to_2i (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z - 2*Complex.I) ≥ 1 ∧ ∃ w : ℂ, Complex.abs w = 1 ∧ Complex.abs (w - 2*Complex.I) = 1 :=
by sorry

end min_distance_to_2i_l1561_156192


namespace students_in_general_hall_l1561_156113

theorem students_in_general_hall (general : ℕ) (biology : ℕ) (math : ℕ) : 
  biology = 2 * general →
  math = (3 * (general + biology)) / 5 →
  general + biology + math = 144 →
  general = 30 := by
sorry

end students_in_general_hall_l1561_156113


namespace number_of_valid_paths_l1561_156168

-- Define the grid dimensions
def columns : ℕ := 10
def rows : ℕ := 4

-- Define the forbidden segment
def forbidden_column : ℕ := 6
def forbidden_row_start : ℕ := 2
def forbidden_row_end : ℕ := 3

-- Define the total number of steps
def total_steps : ℕ := columns + rows

-- Function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Function to calculate the number of paths between two points
def paths_between (col_diff row_diff : ℕ) : ℕ := 
  binomial (col_diff + row_diff) row_diff

-- Theorem statement
theorem number_of_valid_paths : 
  paths_between columns rows - 
  (paths_between forbidden_column (rows - forbidden_row_end) * 
   paths_between (columns - forbidden_column) (forbidden_row_end)) = 861 := by
  sorry

end number_of_valid_paths_l1561_156168


namespace smallest_x_for_equation_l1561_156165

theorem smallest_x_for_equation : 
  ∃ (x : ℝ), x > 0 ∧ 
  (⌊x^2⌋ : ℝ) - x * (⌊x⌋ : ℝ) = 10 ∧ 
  (∀ y : ℝ, y > 0 ∧ (⌊y^2⌋ : ℝ) - y * (⌊y⌋ : ℝ) = 10 → y ≥ x) ∧
  x = 131 / 11 := by
sorry

end smallest_x_for_equation_l1561_156165


namespace cubic_expression_value_l1561_156152

theorem cubic_expression_value (p q : ℝ) : 
  3 * p^2 - 7 * p + 1 = 0 →
  3 * q^2 - 7 * q + 1 = 0 →
  (9 * p^3 - 9 * q^3) / (p - q) = 46 := by
sorry

end cubic_expression_value_l1561_156152


namespace equal_digit_prob_is_three_eighths_l1561_156133

/-- Represents a die with a given number of sides -/
structure Die :=
  (sides : ℕ)

/-- Probability of rolling a one-digit number on a given die -/
def prob_one_digit (d : Die) : ℚ :=
  if d.sides ≤ 9 then 1 else (9 : ℚ) / d.sides

/-- Probability of rolling a two-digit number on a given die -/
def prob_two_digit (d : Die) : ℚ :=
  1 - prob_one_digit d

/-- The set of dice used in the game -/
def game_dice : List Die :=
  [⟨6⟩, ⟨6⟩, ⟨6⟩, ⟨12⟩, ⟨12⟩]

/-- The probability of having an equal number of dice showing two-digit and one-digit numbers -/
def equal_digit_prob : ℚ :=
  2 * (prob_two_digit ⟨12⟩ * prob_one_digit ⟨12⟩)

theorem equal_digit_prob_is_three_eighths :
  equal_digit_prob = 3/8 := by
  sorry

end equal_digit_prob_is_three_eighths_l1561_156133


namespace probability_two_black_balls_l1561_156119

/-- The probability of drawing two black balls from a box containing white and black balls. -/
theorem probability_two_black_balls (white_balls black_balls : ℕ) 
  (h_white : white_balls = 7) (h_black : black_balls = 8) : 
  (black_balls.choose 2 : ℚ) / ((white_balls + black_balls).choose 2) = 4 / 15 := by
  sorry

end probability_two_black_balls_l1561_156119


namespace max_black_pens_l1561_156139

/-- The maximum number of pens in the basket -/
def max_pens : ℕ := 2500

/-- The probability of selecting two pens of the same color -/
def same_color_prob : ℚ := 1 / 3

/-- The function that calculates the probability of selecting two pens of the same color
    given the number of black pens and total pens -/
def calc_prob (black_pens total_pens : ℕ) : ℚ :=
  let red_pens := total_pens - black_pens
  (black_pens * (black_pens - 1) + red_pens * (red_pens - 1)) / (total_pens * (total_pens - 1))

theorem max_black_pens :
  ∃ (total_pens : ℕ) (black_pens : ℕ),
    total_pens ≤ max_pens ∧
    calc_prob black_pens total_pens = same_color_prob ∧
    black_pens = 1275 ∧
    ∀ (t : ℕ) (b : ℕ),
      t ≤ max_pens →
      calc_prob b t = same_color_prob →
      b ≤ 1275 :=
by sorry

end max_black_pens_l1561_156139


namespace volume_ratio_l1561_156143

-- Define the vertices of the larger pyramid
def large_pyramid_vertices : List (Fin 4 → ℚ) := [
  (λ i => if i = 0 then 1 else 0),
  (λ i => if i = 1 then 1 else 0),
  (λ i => if i = 2 then 1 else 0),
  (λ i => if i = 3 then 1 else 0),
  (λ _ => 0)
]

-- Define the center of the base of the larger pyramid
def base_center : Fin 4 → ℚ := λ _ => 1/4

-- Define the vertices of the smaller pyramid
def small_pyramid_vertices : List (Fin 4 → ℚ) := 
  base_center :: (List.range 4).map (λ i => λ j => if i = j then 1/2 else 0)

-- Define a function to calculate the volume of a pyramid
def pyramid_volume (vertices : List (Fin 4 → ℚ)) : ℚ := sorry

-- Theorem stating the volume ratio
theorem volume_ratio : 
  (pyramid_volume small_pyramid_vertices) / (pyramid_volume large_pyramid_vertices) = 3/64 := by
  sorry

end volume_ratio_l1561_156143


namespace z_in_fourth_quadrant_l1561_156112

def z : ℂ := Complex.I * (-2 - Complex.I)

theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end z_in_fourth_quadrant_l1561_156112


namespace inscribed_circle_radius_isosceles_triangle_l1561_156198

/-- The radius of the inscribed circle in an isosceles triangle --/
theorem inscribed_circle_radius_isosceles_triangle 
  (A B C : EuclideanSpace ℝ (Fin 2)) 
  (h_isosceles : dist A B = dist A C) 
  (h_AB : dist A B = 7)
  (h_BC : dist B C = 6) :
  let s := (dist A B + dist A C + dist B C) / 2
  let area := Real.sqrt (s * (s - dist A B) * (s - dist A C) * (s - dist B C))
  area / s = (3 * Real.sqrt 10) / 5 := by
sorry

end inscribed_circle_radius_isosceles_triangle_l1561_156198


namespace quadratic_equation_solution_l1561_156196

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = 1/2 ∧ x₂ = 1 ∧ 
  2 * x₁^2 - 3 * x₁ + 1 = 0 ∧ 
  2 * x₂^2 - 3 * x₂ + 1 = 0 :=
by
  sorry

end quadratic_equation_solution_l1561_156196


namespace circle_properties_l1561_156183

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line1 (x y : ℝ) : Prop := x - 6 * y - 10 = 0
def line2 (x y : ℝ) : Prop := 5 * x - 3 * y = 0

-- Define the given conditions
axiom circle_intersects_line1 : ∃ (c : Circle), line1 4 (-1)
axiom center_on_line2 : ∀ (c : Circle), line2 c.center.1 c.center.2

-- Define the theorem to prove
theorem circle_properties (c : Circle) :
  (∀ (x y : ℝ), (x - 3)^2 + (y - 5)^2 = 37 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
  (∃ (chord : ℝ), chord = 2 * Real.sqrt 3 ∧
    ∀ (l : ℝ → ℝ → Prop),
      (∀ x y, l x y → x = 0 ∨ y = 0) →
      (∃ x₁ y₁ x₂ y₂, l x₁ y₁ ∧ l x₂ y₂ ∧
        (x₁ - c.center.1)^2 + (y₁ - c.center.2)^2 = c.radius^2 ∧
        (x₂ - c.center.1)^2 + (y₂ - c.center.2)^2 = c.radius^2 ∧
        (x₂ - x₁)^2 + (y₂ - y₁)^2 ≤ chord^2)) :=
by
  sorry


end circle_properties_l1561_156183


namespace not_p_sufficient_not_necessary_for_not_q_l1561_156166

-- Define the conditions
def p (a : ℝ) : Prop := a ≤ 2
def q (a : ℝ) : Prop := a * (a - 2) ≤ 0

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ a : ℝ, ¬(p a) → ¬(q a)) ∧
  ¬(∀ a : ℝ, ¬(q a) → ¬(p a)) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l1561_156166


namespace gold_coins_percentage_l1561_156120

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  total : ℝ
  beads : ℝ
  coins : ℝ
  silver_coins : ℝ
  gold_coins : ℝ

/-- The conditions of the urn as given in the problem -/
def urn_conditions (u : UrnComposition) : Prop :=
  u.total > 0 ∧
  u.beads + u.coins = u.total ∧
  u.silver_coins + u.gold_coins = u.coins ∧
  u.beads = 0.3 * u.total ∧
  u.silver_coins = 0.3 * u.coins

/-- The theorem stating that 49% of the objects in the urn are gold coins -/
theorem gold_coins_percentage (u : UrnComposition) 
  (h : urn_conditions u) : u.gold_coins / u.total = 0.49 := by
  sorry


end gold_coins_percentage_l1561_156120


namespace fraction_equals_sqrt_two_l1561_156104

theorem fraction_equals_sqrt_two (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : a^2 + b^2 = 6*a*b) : 
  (a + b) / (a - b) = Real.sqrt 2 := by
sorry

end fraction_equals_sqrt_two_l1561_156104


namespace solution_set_implies_a_eq_one_solution_set_varies_with_a_l1561_156178

/-- The quadratic function f(x) = ax^2 + (1-2a)x - 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (1 - 2*a) * x - 2

/-- The solution set of f(x) > 0 when a = 1 -/
def solution_set_a1 : Set ℝ := {x | x < -1 ∨ x > 2}

/-- Theorem: When the solution set of f(x) > 0 is {x | x < -1 or x > 2}, a = 1 -/
theorem solution_set_implies_a_eq_one :
  (∀ x, f 1 x > 0 ↔ x ∈ solution_set_a1) → 1 = 1 := by sorry

/-- The solution set of f(x) > 0 for a > 0 -/
def solution_set_a_pos (a : ℝ) : Set ℝ := {x | x < -1/a ∨ x > 2}

/-- The solution set of f(x) > 0 for a = 0 -/
def solution_set_a_zero : Set ℝ := {x | x > 2}

/-- The solution set of f(x) > 0 for -1/2 < a < 0 -/
def solution_set_a_neg_small (a : ℝ) : Set ℝ := {x | 2 < x ∧ x < -1/a}

/-- The solution set of f(x) > 0 for a = -1/2 -/
def solution_set_a_neg_half : Set ℝ := ∅

/-- The solution set of f(x) > 0 for a < -1/2 -/
def solution_set_a_neg_large (a : ℝ) : Set ℝ := {x | -1/a < x ∧ x < 2}

/-- Theorem: The solution set of f(x) > 0 varies for different ranges of a ∈ ℝ -/
theorem solution_set_varies_with_a (a : ℝ) :
  (∀ x, f a x > 0 ↔ 
    (a > 0 ∧ x ∈ solution_set_a_pos a) ∨
    (a = 0 ∧ x ∈ solution_set_a_zero) ∨
    (-1/2 < a ∧ a < 0 ∧ x ∈ solution_set_a_neg_small a) ∨
    (a = -1/2 ∧ x ∈ solution_set_a_neg_half) ∨
    (a < -1/2 ∧ x ∈ solution_set_a_neg_large a)) := by sorry

end solution_set_implies_a_eq_one_solution_set_varies_with_a_l1561_156178


namespace xy_positive_iff_fraction_positive_and_am_gm_inequality_l1561_156197

theorem xy_positive_iff_fraction_positive_and_am_gm_inequality :
  (∀ x y : ℝ, x * y > 0 ↔ x / y > 0) ∧
  (∀ a b : ℝ, a * b ≤ ((a + b) / 2)^2) := by
  sorry

end xy_positive_iff_fraction_positive_and_am_gm_inequality_l1561_156197


namespace circle_equation_radius_five_l1561_156174

/-- A circle equation in the form x^2 + 8x + y^2 + 4y - k = 0 -/
def CircleEquation (x y k : ℝ) : Prop :=
  x^2 + 8*x + y^2 + 4*y - k = 0

/-- The standard form of a circle equation with center (h, j) and radius r -/
def StandardCircleEquation (x y h j r : ℝ) : Prop :=
  (x - h)^2 + (y - j)^2 = r^2

theorem circle_equation_radius_five (k : ℝ) :
  (∀ x y, CircleEquation x y k ↔ StandardCircleEquation x y (-4) (-2) 5) ↔ k = 5 := by
  sorry

end circle_equation_radius_five_l1561_156174


namespace intersection_in_second_quadrant_l1561_156147

theorem intersection_in_second_quadrant (k : ℝ) :
  (∃ x y : ℝ, k * x - y = k - 1 ∧ k * y = x + 2 * k ∧ x < 0 ∧ y > 0) ↔ 0 < k ∧ k < 1/2 :=
by sorry

end intersection_in_second_quadrant_l1561_156147


namespace max_value_of_expression_l1561_156107

theorem max_value_of_expression (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum : x + y + z = 1) :
  x + y^2 + z^3 ≤ 1 ∧ ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ ≥ 0 ∧ y₀ ≥ 0 ∧ z₀ ≥ 0 ∧ x₀ + y₀ + z₀ = 1 ∧ x₀ + y₀^2 + z₀^3 = 1 :=
by sorry

end max_value_of_expression_l1561_156107


namespace bird_stork_difference_l1561_156170

theorem bird_stork_difference : 
  ∀ (initial_storks initial_birds joining_birds : ℕ),
    initial_storks = 5 →
    initial_birds = 3 →
    joining_birds = 4 →
    (initial_birds + joining_birds) - initial_storks = 2 := by
  sorry

end bird_stork_difference_l1561_156170


namespace expression_evaluation_l1561_156142

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- The expression to be evaluated -/
def expression : ℂ := 2 * i^13 - 3 * i^18 + 4 * i^23 - 5 * i^28 + 6 * i^33

/-- The theorem stating the equality of the expression and its simplified form -/
theorem expression_evaluation : expression = 4 * i - 2 := by
  sorry

end expression_evaluation_l1561_156142


namespace jane_max_tickets_l1561_156108

/-- Represents the maximum number of tickets Jane can buy given the conditions. -/
def max_tickets (regular_price : ℕ) (discount_price : ℕ) (budget : ℕ) (discount_threshold : ℕ) : ℕ :=
  let regular_tickets := min discount_threshold (budget / regular_price)
  let remaining_budget := budget - regular_tickets * regular_price
  let extra_tickets := remaining_budget / discount_price
  regular_tickets + extra_tickets

/-- Theorem stating that the maximum number of tickets Jane can buy is 19. -/
theorem jane_max_tickets :
  max_tickets 15 12 135 8 = 19 := by
  sorry

end jane_max_tickets_l1561_156108


namespace smallest_with_12_divisors_l1561_156138

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- n has exactly 12 positive integer divisors -/
def has_12_divisors (n : ℕ+) : Prop := num_divisors n = 12

theorem smallest_with_12_divisors :
  ∃ (n : ℕ+), has_12_divisors n ∧ ∀ (m : ℕ+), has_12_divisors m → n ≤ m := by
  use 288
  sorry

end smallest_with_12_divisors_l1561_156138


namespace cylinder_radius_determination_l1561_156134

/-- Given a cylinder with height 4 units, if increasing its radius by 3 units
    and increasing its height by 3 units both result in the same volume increase,
    then the original radius of the cylinder is 12 units. -/
theorem cylinder_radius_determination (r : ℝ) (y : ℝ) : 
  (4 * π * ((r + 3)^2 - r^2) = y) →
  (3 * π * r^2 = y) →
  r = 12 := by sorry

end cylinder_radius_determination_l1561_156134


namespace pirate_treasure_probability_l1561_156110

def num_islands : ℕ := 8
def num_treasure_islands : ℕ := 4
def prob_treasure : ℚ := 1/5
def prob_trap : ℚ := 1/10
def prob_neither : ℚ := 7/10

theorem pirate_treasure_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  prob_treasure ^ num_treasure_islands *
  prob_neither ^ (num_islands - num_treasure_islands) =
  33614/1250000 := by sorry

end pirate_treasure_probability_l1561_156110


namespace problem_statement_l1561_156105

theorem problem_statement (a : ℝ) (h : a^2 - 2*a = 1) : 3*a^2 - 6*a - 4 = -1 := by
  sorry

end problem_statement_l1561_156105


namespace tetrahedron_volume_and_height_l1561_156175

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the volume of a tetrahedron given its four vertices -/
def tetrahedronVolume (a b c d : Point3D) : ℝ := sorry

/-- Calculates the height of a tetrahedron from a vertex to the opposite face -/
def tetrahedronHeight (a b c d : Point3D) : ℝ := sorry

theorem tetrahedron_volume_and_height :
  let a₁ : Point3D := ⟨1, -1, 2⟩
  let a₂ : Point3D := ⟨2, 1, 2⟩
  let a₃ : Point3D := ⟨1, 1, 4⟩
  let a₄ : Point3D := ⟨6, -3, 8⟩
  (tetrahedronVolume a₁ a₂ a₃ a₄ = 6) ∧
  (tetrahedronHeight a₄ a₁ a₂ a₃ = 3 * Real.sqrt 6) := by
  sorry

end tetrahedron_volume_and_height_l1561_156175


namespace chemistry_alone_count_l1561_156151

/-- Represents the number of students in a school with chemistry and biology classes -/
structure School where
  total : ℕ
  chemistry : ℕ
  biology : ℕ
  both : ℕ

/-- The conditions of the school -/
def school_conditions (s : School) : Prop :=
  s.total = 100 ∧
  s.chemistry + s.biology - s.both = s.total ∧
  s.chemistry = 4 * s.biology ∧
  s.both = 10

/-- The theorem stating that under the given conditions, 
    the number of students in chemistry class alone is 80 -/
theorem chemistry_alone_count (s : School) 
  (h : school_conditions s) : s.chemistry - s.both = 80 := by
  sorry

end chemistry_alone_count_l1561_156151


namespace range_of_t_l1561_156131

theorem range_of_t (a b : ℝ) (h : a^2 + a*b + b^2 = 1) :
  let t := a*b - a^2 - b^2
  ∀ x, (∃ a b : ℝ, a^2 + a*b + b^2 = 1 ∧ t = a*b - a^2 - b^2) → -3 ≤ x ∧ x ≤ -1/3 :=
by sorry

end range_of_t_l1561_156131


namespace tenth_number_value_l1561_156173

def known_numbers : List ℕ := [744, 745, 747, 748, 749, 752, 752, 753, 755]

theorem tenth_number_value (x : ℕ) :
  (known_numbers.sum + x) / 10 = 750 →
  x = 1555 := by
  sorry

end tenth_number_value_l1561_156173


namespace star_commutative_iff_three_lines_l1561_156187

/-- The ⋆ operation -/
def star (a b : ℝ) : ℝ := a^2 * b - 2 * a * b^2

/-- The set of points (x, y) where x ⋆ y = y ⋆ x -/
def star_commutative_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

/-- The union of three lines: x = 0, y = 0, and x = y -/
def three_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2}

theorem star_commutative_iff_three_lines :
  star_commutative_set = three_lines := by sorry

end star_commutative_iff_three_lines_l1561_156187


namespace roses_cut_is_difference_jessica_roses_problem_l1561_156172

/-- The number of roses Jessica cut from her flower garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Theorem stating that the number of roses Jessica cut is the difference between the final and initial number of roses -/
theorem roses_cut_is_difference (initial_roses final_roses : ℕ) 
  (h : final_roses ≥ initial_roses) : 
  roses_cut initial_roses final_roses = final_roses - initial_roses :=
by
  sorry

/-- The specific problem instance -/
theorem jessica_roses_problem :
  roses_cut 10 18 = 8 :=
by
  sorry

end roses_cut_is_difference_jessica_roses_problem_l1561_156172


namespace problem_1_problem_2_problem_3_l1561_156188

-- Problem 1
theorem problem_1 (x : ℝ) (h : x = 2 - Real.sqrt 7) : x^2 - 4*x + 5 = 8 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : 2*x = Real.sqrt 5 + 1) : x^3 - 2*x^2 = -1 := by
  sorry

-- Problem 3
theorem problem_3 (a : ℝ) (h : a^2 = Real.sqrt (a^2 + 10) + 3) : a^2 + 1/a^2 = Real.sqrt 53 := by
  sorry

end problem_1_problem_2_problem_3_l1561_156188


namespace decimal_difference_equals_fraction_l1561_156186

/-- The repeating decimal 0.2̅3̅ -/
def repeating_decimal : ℚ := 23 / 99

/-- The terminating decimal 0.23 -/
def terminating_decimal : ℚ := 23 / 100

/-- The difference between the repeating decimal 0.2̅3̅ and the terminating decimal 0.23 -/
def decimal_difference : ℚ := repeating_decimal - terminating_decimal

theorem decimal_difference_equals_fraction : decimal_difference = 23 / 9900 := by
  sorry

end decimal_difference_equals_fraction_l1561_156186


namespace speed_range_correct_l1561_156163

/-- Represents a road roller with its properties and the road to be compressed -/
structure RoadRoller where
  strip_width : Real
  overlap_ratio : Real
  road_length : Real
  road_width : Real
  compression_count : Nat
  min_time : Real
  max_time : Real

/-- Calculates the range of speeds for the road roller to complete the task -/
def calculate_speed_range (roller : RoadRoller) : Set Real :=
  let effective_width := roller.strip_width * (1 - roller.overlap_ratio)
  let passes := Nat.ceil (roller.road_width / effective_width)
  let total_distance := passes * roller.compression_count * 2 * roller.road_length
  let min_speed := total_distance / (roller.max_time * 1000)
  let max_speed := total_distance / (roller.min_time * 1000)
  {x | min_speed ≤ x ∧ x ≤ max_speed}

/-- Theorem stating that the calculated speed range is correct -/
theorem speed_range_correct (roller : RoadRoller) :
  roller.strip_width = 0.85 ∧
  roller.overlap_ratio = 1/4 ∧
  roller.road_length = 750 ∧
  roller.road_width = 6.5 ∧
  roller.compression_count = 2 ∧
  roller.min_time = 5 ∧
  roller.max_time = 6 →
  ∀ x ∈ calculate_speed_range roller, 2.75 ≤ x ∧ x ≤ 3.3 :=
by sorry

end speed_range_correct_l1561_156163


namespace equation_one_solution_equation_two_solution_quadratic_function_solution_l1561_156149

-- Equation 1
theorem equation_one_solution (x : ℝ) : 
  x^2 - 6*x + 3 = 0 ↔ x = 3 + Real.sqrt 6 ∨ x = 3 - Real.sqrt 6 :=
sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  x*(x+2) = 3*(x+2) ↔ x = -2 ∨ x = 3 :=
sorry

-- Equation 3
def quadratic_function (x : ℝ) : ℝ := 4*x^2 + 5*x

theorem quadratic_function_solution :
  (quadratic_function 0 = 0) ∧ 
  (quadratic_function (-1) = -1) ∧ 
  (quadratic_function 1 = 9) :=
sorry

end equation_one_solution_equation_two_solution_quadratic_function_solution_l1561_156149


namespace total_area_is_1800_l1561_156115

/-- Calculates the total area of rooms given initial dimensions and modifications -/
def total_area (length width increase_amount : ℕ) : ℕ :=
  let new_length := length + increase_amount
  let new_width := width + increase_amount
  let single_room_area := new_length * new_width
  let four_rooms_area := 4 * single_room_area
  let double_room_area := 2 * single_room_area
  four_rooms_area + double_room_area

/-- Theorem stating that the total area of rooms is 1800 square feet -/
theorem total_area_is_1800 :
  total_area 13 18 2 = 1800 := by sorry

end total_area_is_1800_l1561_156115


namespace integer_roots_of_polynomial_l1561_156109

def polynomial (x : ℤ) : ℤ := x^3 + 3*x^2 - 4*x - 13

theorem integer_roots_of_polynomial :
  {x : ℤ | polynomial x = 0} = {-13, -1, 1, 13} := by sorry

end integer_roots_of_polynomial_l1561_156109


namespace coefficient_x4_equals_180_l1561_156130

/-- The coefficient of x^4 in the expansion of (2 + √x - 1/x^2016)^10 -/
def coefficient_x4 (x : ℝ) : ℕ :=
  -- We define this as a natural number since coefficients in polynomial expansions are typically integers
  -- The actual computation is not implemented here
  sorry

/-- The main theorem stating that the coefficient of x^4 is 180 -/
theorem coefficient_x4_equals_180 :
  ∀ x : ℝ, coefficient_x4 x = 180 := by
  sorry

end coefficient_x4_equals_180_l1561_156130


namespace chris_money_before_birthday_l1561_156126

def chris_current_money : ℕ := 279
def grandmother_gift : ℕ := 25
def aunt_uncle_gift : ℕ := 20
def parents_gift : ℕ := 75

def total_birthday_gifts : ℕ := grandmother_gift + aunt_uncle_gift + parents_gift

theorem chris_money_before_birthday :
  chris_current_money - total_birthday_gifts = 159 := by
  sorry

end chris_money_before_birthday_l1561_156126


namespace expression_evaluation_l1561_156118

theorem expression_evaluation : (1/8)^(1/3) - Real.log 2 / Real.log 3 * Real.log 27 / Real.log 4 + 2018^0 = 0 := by
  sorry

end expression_evaluation_l1561_156118


namespace trig_expression_equals_one_l1561_156121

theorem trig_expression_equals_one : 
  let tan_30 : ℝ := 1 / Real.sqrt 3
  let sin_30 : ℝ := 1 / 2
  (tan_30^2 - sin_30^2) / (tan_30^2 * sin_30^2) = 1 := by sorry

end trig_expression_equals_one_l1561_156121
