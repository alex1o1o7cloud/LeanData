import Mathlib

namespace odd_function_properties_l340_34027

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_properties (f : ℝ → ℝ) 
    (h_odd : is_odd f) 
    (h_shift : ∀ x, f (x - 2) = -f x) : 
    (f 2 = 0) ∧ 
    (periodic f 4) ∧ 
    (∀ x, f (x + 2) = f (-x)) := by
  sorry

end odd_function_properties_l340_34027


namespace intersection_A_B_l340_34036

def A : Set ℝ := {x | x * (x - 3) < 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end intersection_A_B_l340_34036


namespace sin_570_degrees_l340_34060

theorem sin_570_degrees : 2 * Real.sin (570 * π / 180) = -1 := by
  sorry

end sin_570_degrees_l340_34060


namespace polynomial_multiplication_l340_34014

theorem polynomial_multiplication (t : ℝ) : 
  (3*t^3 + 2*t^2 - 4*t + 3) * (-2*t^2 + 3*t - 4) = 
  -6*t^5 + 5*t^4 + 2*t^3 - 26*t^2 + 25*t - 12 := by
sorry

end polynomial_multiplication_l340_34014


namespace tangent_line_sum_l340_34080

open Real

/-- Given a function f: ℝ → ℝ with a tangent line at x = 2 described by the equation 2x - y - 3 = 0,
    prove that f(2) + f'(2) = 3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x y, y = f 2 → 2 * x - y - 3 = 0 ↔ y = 2 * x - 3) :
  f 2 + deriv f 2 = 3 := by
  sorry

end tangent_line_sum_l340_34080


namespace square_sum_division_theorem_l340_34064

theorem square_sum_division_theorem : (56^2 + 56^2) / 28^2 = 8 := by
  sorry

end square_sum_division_theorem_l340_34064


namespace fraction_simplification_l340_34029

theorem fraction_simplification : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end fraction_simplification_l340_34029


namespace binomial_coefficient_equality_l340_34009

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 25 (2 * x) = Nat.choose 25 (x + 4)) ↔ (x = 4 ∨ x = 7) :=
by sorry

end binomial_coefficient_equality_l340_34009


namespace factor_w4_minus_16_l340_34093

theorem factor_w4_minus_16 (w : ℝ) : w^4 - 16 = (w-2)*(w+2)*(w^2+4) := by sorry

end factor_w4_minus_16_l340_34093


namespace max_value_of_a_l340_34062

-- Define the condition function
def condition (x : ℝ) : Prop := x^2 - 2*x - 3 > 0

-- Define the main theorem
theorem max_value_of_a :
  (∀ x, x < a → condition x) ∧ 
  (∃ x, condition x ∧ x ≥ a) →
  ∀ b, (∀ x, x < b → condition x) ∧ 
       (∃ x, condition x ∧ x ≥ b) →
  b ≤ -1 :=
sorry

end max_value_of_a_l340_34062


namespace quadratic_condition_l340_34092

theorem quadratic_condition (m : ℝ) : 
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, m * x^2 - 4*x + 3 = a * x^2 + b * x + c) → m ≠ 0 := by
  sorry

end quadratic_condition_l340_34092


namespace intersection_complement_equality_l340_34056

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set Nat := {1, 3, 5, 7, 9}
def B : Set Nat := {1, 2, 5, 6, 8}

theorem intersection_complement_equality : A ∩ (U \ B) = {3, 7, 9} := by
  sorry

end intersection_complement_equality_l340_34056


namespace xyz_inequality_l340_34081

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  x^3 + y^3 + z^3 ≥ x*y + y*z + x*z := by
sorry

end xyz_inequality_l340_34081


namespace symmetric_complex_product_l340_34039

theorem symmetric_complex_product :
  ∀ (z₁ z₂ : ℂ),
  (Complex.re z₁ = -Complex.re z₂) →
  (Complex.im z₁ = Complex.im z₂) →
  (z₁ = 3 + Complex.I) →
  z₁ * z₂ = -10 := by
sorry

end symmetric_complex_product_l340_34039


namespace units_digit_17_to_17_l340_34046

theorem units_digit_17_to_17 : (17^17 : ℕ) % 10 = 7 := by sorry

end units_digit_17_to_17_l340_34046


namespace real_roots_iff_k_leq_5_root_one_implies_k_values_l340_34051

-- Define the quadratic equation
def quadratic (k x : ℝ) : ℝ := x^2 - 2*(k-3)*x + k^2 - 4*k - 1

-- Theorem 1: The equation has real roots iff k ≤ 5
theorem real_roots_iff_k_leq_5 (k : ℝ) : 
  (∃ x : ℝ, quadratic k x = 0) ↔ k ≤ 5 := by sorry

-- Theorem 2: If 1 is a root, then k = 3 + √3 or k = 3 - √3
theorem root_one_implies_k_values (k : ℝ) : 
  quadratic k 1 = 0 → k = 3 + Real.sqrt 3 ∨ k = 3 - Real.sqrt 3 := by sorry

end real_roots_iff_k_leq_5_root_one_implies_k_values_l340_34051


namespace expected_consecutive_reds_l340_34066

/-- A bag containing one red, one yellow, and one blue ball -/
inductive Ball : Type
| Red : Ball
| Yellow : Ball
| Blue : Ball

/-- The process of drawing balls with replacement -/
def DrawProcess : Type := ℕ → Ball

/-- The probability of drawing each color is equal -/
axiom equal_probability (b : Ball) : ℝ

/-- The sum of probabilities is 1 -/
axiom prob_sum : equal_probability Ball.Red + equal_probability Ball.Yellow + equal_probability Ball.Blue = 1

/-- ξ is the number of draws until two consecutive red balls are drawn -/
def ξ (process : DrawProcess) : ℕ := sorry

/-- The expected value of ξ -/
def expected_ξ : ℝ := sorry

/-- Theorem: The expected value of ξ is 12 -/
theorem expected_consecutive_reds : expected_ξ = 12 := by sorry

end expected_consecutive_reds_l340_34066


namespace tangent_line_to_circle_l340_34016

/-- Given a positive real number r, prove that if the line x - y = r is tangent to the circle x^2 + y^2 = r, then r = 2 -/
theorem tangent_line_to_circle (r : ℝ) (hr : r > 0) : 
  (∀ x y : ℝ, x - y = r → x^2 + y^2 ≤ r) ∧ 
  (∃ x y : ℝ, x - y = r ∧ x^2 + y^2 = r) → 
  r = 2 := by sorry

end tangent_line_to_circle_l340_34016


namespace one_integer_is_seventeen_l340_34071

theorem one_integer_is_seventeen (a b c d : ℕ+) 
  (eq1 : (b.val + c.val + d.val) / 3 + 2 * a.val = 54)
  (eq2 : (a.val + c.val + d.val) / 3 + 2 * b.val = 50)
  (eq3 : (a.val + b.val + d.val) / 3 + 2 * c.val = 42)
  (eq4 : (a.val + b.val + c.val) / 3 + 2 * d.val = 30) :
  a = 17 ∨ b = 17 ∨ c = 17 ∨ d = 17 := by
sorry

end one_integer_is_seventeen_l340_34071


namespace theme_parks_calculation_l340_34038

/-- The number of theme parks in three towns -/
def total_theme_parks (jamestown venice marina_del_ray : ℕ) : ℕ :=
  jamestown + venice + marina_del_ray

/-- Theorem stating the total number of theme parks in the three towns -/
theorem theme_parks_calculation :
  ∃ (jamestown venice marina_del_ray : ℕ),
    jamestown = 20 ∧
    venice = jamestown + 25 ∧
    marina_del_ray = jamestown + 50 ∧
    total_theme_parks jamestown venice marina_del_ray = 135 :=
by
  sorry

end theme_parks_calculation_l340_34038


namespace additional_amount_for_free_shipping_l340_34061

/-- The problem of calculating the additional amount needed for free shipping -/
def free_shipping_problem (free_shipping_threshold : ℚ) 
                          (shampoo_price : ℚ) 
                          (conditioner_price : ℚ) 
                          (lotion_price : ℚ) 
                          (lotion_quantity : ℕ) : ℚ :=
  let total_spent := shampoo_price + conditioner_price + lotion_price * (lotion_quantity : ℚ)
  max (free_shipping_threshold - total_spent) 0

/-- Theorem stating the correct additional amount needed for free shipping -/
theorem additional_amount_for_free_shipping :
  free_shipping_problem 50 10 10 6 3 = 12 :=
sorry

end additional_amount_for_free_shipping_l340_34061


namespace jones_trip_time_comparison_l340_34068

theorem jones_trip_time_comparison 
  (distance1 : ℝ) 
  (distance2 : ℝ) 
  (speed_multiplier : ℝ) 
  (h1 : distance1 = 50) 
  (h2 : distance2 = 300) 
  (h3 : speed_multiplier = 3) :
  let time1 := distance1 / (distance1 / time1)
  let time2 := distance2 / (speed_multiplier * (distance1 / time1))
  time2 = 2 * time1 := by
sorry

end jones_trip_time_comparison_l340_34068


namespace sequence_convergence_l340_34033

theorem sequence_convergence (a : ℕ → ℤ) 
  (h : ∀ n : ℕ, (a (n + 2))^2 + a (n + 1) * a n ≤ a (n + 2) * (a (n + 1) + a n)) :
  ∃ N : ℕ, ∀ n ≥ N, a (n + 2) = a n :=
sorry

end sequence_convergence_l340_34033


namespace total_animals_theorem_l340_34024

/-- Calculates the total number of animals seen given initial counts and changes --/
def total_animals_seen (initial_beavers initial_chipmunks : ℕ) : ℕ :=
  let morning_total := initial_beavers + initial_chipmunks
  let afternoon_beavers := 4 * initial_beavers
  let afternoon_chipmunks := initial_chipmunks - 20
  let afternoon_total := afternoon_beavers + afternoon_chipmunks
  morning_total + afternoon_total

/-- Theorem stating that given the specific initial counts and changes, the total animals seen is 410 --/
theorem total_animals_theorem : total_animals_seen 50 90 = 410 := by
  sorry

end total_animals_theorem_l340_34024


namespace exercise_239_theorem_existence_not_implied_l340_34067

-- Define a property A for functions
def PropertyA (f : ℝ → ℝ) : Prop := sorry

-- Define periodicity for functions
def Periodic (f : ℝ → ℝ) : Prop := ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x

-- The theorem from exercise 239
theorem exercise_239_theorem : ∀ f : ℝ → ℝ, PropertyA f → Periodic f := sorry

-- The statement we want to prove
theorem existence_not_implied :
  (∀ f : ℝ → ℝ, PropertyA f → Periodic f) →
  ¬(∃ f : ℝ → ℝ, PropertyA f) := sorry

end exercise_239_theorem_existence_not_implied_l340_34067


namespace sum_expression_l340_34003

-- Define the variables
variable (x y z : ℝ)

-- State the theorem
theorem sum_expression (h1 : y = 3 * x + 1) (h2 : z = y - x) : 
  x + y + z = 6 * x + 2 := by
  sorry

end sum_expression_l340_34003


namespace circle_containment_l340_34018

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point is inside a circle if its distance from the center is less than the radius --/
def is_inside (p : ℝ × ℝ) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) < c.radius

theorem circle_containment (circles : Fin 6 → Circle) 
  (O : ℝ × ℝ) (h : ∀ i, is_inside O (circles i)) :
  ∃ i j, i ≠ j ∧ is_inside (circles j).center (circles i) := by
  sorry

end circle_containment_l340_34018


namespace square_sum_reciprocal_l340_34035

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 := by
  sorry

end square_sum_reciprocal_l340_34035


namespace homework_difference_l340_34053

theorem homework_difference (total : ℕ) (math : ℕ) (reading : ℕ)
  (h1 : total = 13)
  (h2 : math = 8)
  (h3 : total = math + reading) :
  math - reading = 3 :=
by sorry

end homework_difference_l340_34053


namespace range_of_a_l340_34026

theorem range_of_a (a : ℝ) (n : ℕ) (h1 : a > 1) (h2 : n ≥ 2) 
  (h3 : ∃! (s : Finset ℤ), s.card = n ∧ ∀ x ∈ s, ⌊a * x⌋ = x) :
  1 + 1 / n ≤ a ∧ a < 1 + 1 / (n - 1) := by
  sorry

end range_of_a_l340_34026


namespace test_total_points_l340_34055

-- Define the test structure
structure Test where
  total_questions : ℕ
  two_point_questions : ℕ
  four_point_questions : ℕ

-- Define the function to calculate total points
def calculateTotalPoints (test : Test) : ℕ :=
  2 * test.two_point_questions + 4 * test.four_point_questions

-- Theorem statement
theorem test_total_points (test : Test) 
  (h1 : test.total_questions = 40)
  (h2 : test.two_point_questions = 30)
  (h3 : test.four_point_questions = 10)
  (h4 : test.total_questions = test.two_point_questions + test.four_point_questions) :
  calculateTotalPoints test = 100 := by
  sorry

-- Example usage
def exampleTest : Test := {
  total_questions := 40,
  two_point_questions := 30,
  four_point_questions := 10
}

#eval calculateTotalPoints exampleTest

end test_total_points_l340_34055


namespace roots_of_equation_l340_34057

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => (x^3 - 6*x^2 + 11*x - 6)*(x - 2)
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3 := by
sorry

end roots_of_equation_l340_34057


namespace equal_share_ratio_l340_34091

def total_amount : ℕ := 5400
def num_children : ℕ := 3
def b_share : ℕ := 1800

theorem equal_share_ratio :
  ∃ (a_share c_share : ℕ),
    a_share + b_share + c_share = total_amount ∧
    a_share = c_share ∧
    a_share = b_share :=
by sorry

end equal_share_ratio_l340_34091


namespace abc_sum_sqrt_l340_34006

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 20)
  (h2 : c + a = 22)
  (h3 : a + b = 24) :
  Real.sqrt (a * b * c * (a + b + c)) = 206.1 := by
  sorry

end abc_sum_sqrt_l340_34006


namespace pen_cost_l340_34087

theorem pen_cost (pen_price : ℝ) (briefcase_price : ℝ) : 
  briefcase_price = 5 * pen_price →
  pen_price + briefcase_price = 24 →
  pen_price = 4 := by
sorry

end pen_cost_l340_34087


namespace division_remainder_problem_l340_34005

theorem division_remainder_problem : ∃ (A : ℕ), 17 = 5 * 3 + A ∧ A < 5 := by
  sorry

end division_remainder_problem_l340_34005


namespace work_completion_time_l340_34094

theorem work_completion_time 
  (a_rate : ℝ) (b_rate : ℝ) (work_left : ℝ) (days_worked : ℝ) : 
  a_rate = 1 / 15 →
  b_rate = 1 / 20 →
  work_left = 0.41666666666666663 →
  (1 - work_left) = (a_rate + b_rate) * days_worked →
  days_worked = 5 := by
sorry

end work_completion_time_l340_34094


namespace rahul_deepak_age_ratio_l340_34002

def rahul_future_age : ℕ := 32
def years_to_future : ℕ := 4
def deepak_age : ℕ := 21

theorem rahul_deepak_age_ratio :
  let rahul_age := rahul_future_age - years_to_future
  (rahul_age : ℚ) / deepak_age = 4 / 3 := by
  sorry

end rahul_deepak_age_ratio_l340_34002


namespace tan_and_expression_values_l340_34025

theorem tan_and_expression_values (α : Real) 
  (h_acute : 0 < α ∧ α < π / 2)
  (h_tan : Real.tan (π / 4 + α) = 2) :
  Real.tan α = 1 / 3 ∧ 
  (Real.sqrt 2 * Real.sin (2 * α + π / 4) * Real.cos α - Real.sin α) / Real.cos (2 * α) = (2 / 5) * Real.sqrt 10 := by
  sorry

end tan_and_expression_values_l340_34025


namespace gcd_12012_21021_l340_34030

theorem gcd_12012_21021 : Nat.gcd 12012 21021 = 1001 := by
  sorry

end gcd_12012_21021_l340_34030


namespace inequality_proof_l340_34085

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  (a^4 + b^4)/(a^2 + b^2) + (b^4 + c^4)/(b^2 + c^2) + (c^4 + d^4)/(c^2 + d^2) + (d^4 + a^4)/(d^2 + a^2) ≥ 4 := by
  sorry

end inequality_proof_l340_34085


namespace retirement_total_is_70_l340_34013

/-- The retirement eligibility rule for a company -/
structure RetirementRule where
  hire_year : ℕ
  hire_age : ℕ
  eligible_year : ℕ

/-- Calculate the required total of age and years of employment for retirement -/
def retirement_total (rule : RetirementRule) : ℕ :=
  let years_of_employment := rule.eligible_year - rule.hire_year
  let age_at_eligibility := rule.hire_age + years_of_employment
  age_at_eligibility + years_of_employment

/-- Theorem stating the required total for retirement -/
theorem retirement_total_is_70 (rule : RetirementRule) 
  (h1 : rule.hire_year = 1990)
  (h2 : rule.hire_age = 32)
  (h3 : rule.eligible_year = 2009) :
  retirement_total rule = 70 := by
  sorry

#eval retirement_total ⟨1990, 32, 2009⟩

end retirement_total_is_70_l340_34013


namespace survey_III_participants_l340_34031

/-- Represents the systematic sampling method for a school survey. -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  group_size : ℕ
  first_selected : ℕ
  survey_III_start : ℕ
  survey_III_end : ℕ

/-- The number of students participating in Survey III. -/
def students_in_survey_III (s : SystematicSampling) : ℕ :=
  let n_start := ((s.survey_III_start + s.first_selected - 1) + s.group_size - 1) / s.group_size
  let n_end := (s.survey_III_end + s.first_selected - 1) / s.group_size
  n_end - n_start + 1

/-- Theorem stating the number of students in Survey III for the given conditions. -/
theorem survey_III_participants (s : SystematicSampling) 
  (h1 : s.total_students = 1080)
  (h2 : s.sample_size = 90)
  (h3 : s.group_size = s.total_students / s.sample_size)
  (h4 : s.first_selected = 5)
  (h5 : s.survey_III_start = 847)
  (h6 : s.survey_III_end = 1080) :
  students_in_survey_III s = 19 := by
  sorry

#eval students_in_survey_III {
  total_students := 1080,
  sample_size := 90,
  group_size := 12,
  first_selected := 5,
  survey_III_start := 847,
  survey_III_end := 1080
}

end survey_III_participants_l340_34031


namespace units_digit_G_1000_l340_34077

def G (n : ℕ) : ℕ := 3^(2^n) + 2

theorem units_digit_G_1000 : G 1000 % 10 = 3 := by
  sorry

end units_digit_G_1000_l340_34077


namespace triangle_cosine_difference_l340_34048

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if 4b * sin A = √7 * a and a, b, c are in arithmetic progression
    with positive common difference, then cos A - cos C = √7/2 -/
theorem triangle_cosine_difference (a b c : ℝ) (A B C : ℝ) (h1 : 4 * b * Real.sin A = Real.sqrt 7 * a)
  (h2 : ∃ (d : ℝ), d > 0 ∧ b = a + d ∧ c = b + d) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 := by
  sorry

end triangle_cosine_difference_l340_34048


namespace total_legs_in_group_l340_34065

/-- The number of legs a human has -/
def human_legs : ℕ := 2

/-- The number of legs a dog has -/
def dog_legs : ℕ := 4

/-- The number of humans in the group -/
def num_humans : ℕ := 2

/-- The number of dogs in the group -/
def num_dogs : ℕ := 2

/-- Theorem stating that the total number of legs in the group is 12 -/
theorem total_legs_in_group : 
  num_humans * human_legs + num_dogs * dog_legs = 12 := by
  sorry

end total_legs_in_group_l340_34065


namespace discount_percentage_l340_34063

/-- Proves that the discount percentage is 10% given the costs and final paid amount -/
theorem discount_percentage (couch_cost sectional_cost other_cost paid : ℚ)
  (h1 : couch_cost = 2500)
  (h2 : sectional_cost = 3500)
  (h3 : other_cost = 2000)
  (h4 : paid = 7200) :
  (1 - paid / (couch_cost + sectional_cost + other_cost)) * 100 = 10 := by
  sorry

end discount_percentage_l340_34063


namespace root_transformation_l340_34019

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + 5 = 0) ∧ 
  (r₂^3 - 4*r₂^2 + 5 = 0) ∧ 
  (r₃^3 - 4*r₃^2 + 5 = 0) → 
  ((3*r₁)^3 - 12*(3*r₁)^2 + 135 = 0) ∧ 
  ((3*r₂)^3 - 12*(3*r₂)^2 + 135 = 0) ∧ 
  ((3*r₃)^3 - 12*(3*r₃)^2 + 135 = 0) := by
sorry

end root_transformation_l340_34019


namespace parallel_line_through_point_l340_34097

/-- Given two lines in the form ax + by + c = 0, they are parallel if and only if they have the same slope (a/b ratio) -/
def parallel_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ / b₁ = a₂ / b₂

/-- A point (x, y) lies on a line ax + by + c = 0 if and only if the equation is satisfied -/
def point_on_line (a b c x y : ℝ) : Prop :=
  a * x + b * y + c = 0

/-- The theorem states that the line 3x + 4y - 11 = 0 is parallel to 3x + 4y + 1 = 0 and passes through (1, 2) -/
theorem parallel_line_through_point :
  parallel_lines 3 4 (-11) 3 4 1 ∧ point_on_line 3 4 (-11) 1 2 :=
by sorry

end parallel_line_through_point_l340_34097


namespace ladder_problem_l340_34095

/-- Given a right triangle with hypotenuse 13 meters and one leg 12 meters,
    prove that the other leg is 5 meters. -/
theorem ladder_problem (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : b = 12) :
  a = 5 := by
sorry

end ladder_problem_l340_34095


namespace possible_AC_values_l340_34049

/-- Three points on a line with given distances between them -/
structure ThreePointsOnLine where
  A : ℝ
  B : ℝ
  C : ℝ
  AB_eq : |A - B| = 3
  BC_eq : |B - C| = 5

/-- The possible values for AC given AB = 3 and BC = 5 -/
theorem possible_AC_values (p : ThreePointsOnLine) : 
  |p.A - p.C| = 2 ∨ |p.A - p.C| = 8 :=
by sorry

end possible_AC_values_l340_34049


namespace martha_points_l340_34028

/-- Represents Martha's shopping trip and point system. -/
structure ShoppingTrip where
  /-- Points earned per $10 spent -/
  pointsPerTen : ℕ
  /-- Bonus points for spending over $100 -/
  overHundredBonus : ℕ
  /-- Bonus points for 5th visit -/
  fifthVisitBonus : ℕ
  /-- Price of beef per pound -/
  beefPrice : ℚ
  /-- Quantity of beef in pounds -/
  beefQuantity : ℕ
  /-- Discount on beef as a percentage -/
  beefDiscount : ℚ
  /-- Price of fruits and vegetables per pound -/
  fruitVegPrice : ℚ
  /-- Quantity of fruits and vegetables in pounds -/
  fruitVegQuantity : ℕ
  /-- Discount on fruits and vegetables as a percentage -/
  fruitVegDiscount : ℚ
  /-- Price of spices per jar -/
  spicePrice : ℚ
  /-- Quantity of spice jars -/
  spiceQuantity : ℕ
  /-- Discount on spices as a percentage -/
  spiceDiscount : ℚ
  /-- Price of other groceries before coupon -/
  otherGroceriesPrice : ℚ
  /-- Coupon value for other groceries -/
  otherGroceriesCoupon : ℚ

/-- Calculates the total points earned during the shopping trip. -/
def calculatePoints (trip : ShoppingTrip) : ℕ :=
  sorry

/-- Theorem stating that Martha earns 850 points given the specific shopping conditions. -/
theorem martha_points : ∃ (trip : ShoppingTrip),
  trip.pointsPerTen = 50 ∧
  trip.overHundredBonus = 250 ∧
  trip.fifthVisitBonus = 100 ∧
  trip.beefPrice = 11 ∧
  trip.beefQuantity = 3 ∧
  trip.beefDiscount = 1/10 ∧
  trip.fruitVegPrice = 4 ∧
  trip.fruitVegQuantity = 8 ∧
  trip.fruitVegDiscount = 2/25 ∧
  trip.spicePrice = 6 ∧
  trip.spiceQuantity = 3 ∧
  trip.spiceDiscount = 1/20 ∧
  trip.otherGroceriesPrice = 37 ∧
  trip.otherGroceriesCoupon = 3 ∧
  calculatePoints trip = 850 :=
sorry

end martha_points_l340_34028


namespace scooter_travel_time_l340_34079

/-- The time it takes for a scooter to travel a given distance, given the following conditions:
  * The distance between two points A and B is 50 miles
  * A bicycle travels 1/2 mile per hour slower than the scooter
  * The bicycle takes 45 minutes (3/4 hour) more than the scooter to make the trip
  * x is the scooter's rate of speed in miles per hour
-/
theorem scooter_travel_time (x : ℝ) : 
  (∃ y : ℝ, y > 0 ∧ y = x - 1/2) →  -- Bicycle speed exists and is positive
  50 / (x - 1/2) - 50 / x = 3/4 →   -- Time difference equation
  50 / x = 50 / x :=                -- Conclusion (trivial here, but represents the result)
by sorry

end scooter_travel_time_l340_34079


namespace curve_points_difference_l340_34078

theorem curve_points_difference (a b : ℝ) : 
  (a ≠ b) → 
  ((a : ℝ)^2 + (Real.sqrt π)^4 = 2 * (Real.sqrt π)^2 * a + 1) → 
  ((b : ℝ)^2 + (Real.sqrt π)^4 = 2 * (Real.sqrt π)^2 * b + 1) → 
  |a - b| = 2 := by
  sorry

end curve_points_difference_l340_34078


namespace real_part_of_complex_number_l340_34021

theorem real_part_of_complex_number (z : ℂ) 
  (h1 : Complex.abs z = 1)
  (h2 : Complex.abs (z - 1.45) = 1.05) :
  z.re = 20 / 29 := by
  sorry

end real_part_of_complex_number_l340_34021


namespace mixed_number_multiplication_l340_34037

theorem mixed_number_multiplication : 
  (39 + 18 / 19) * (18 + 19 / 20) = 757 + 1 / 380 := by sorry

end mixed_number_multiplication_l340_34037


namespace function_properties_l340_34045

noncomputable def f (x m : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 + m

theorem function_properties :
  ∃ m : ℝ,
    (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x m ≤ 1) ∧
    (∃ x ∈ Set.Icc 0 (Real.pi / 4), f x m = 1) ∧
    m = -2 ∧
    (∀ x : ℝ, f x m ≥ -3) ∧
    (∀ k : ℤ, f ((2 * Real.pi / 3) + k * Real.pi) m = -3) :=
by sorry

end function_properties_l340_34045


namespace distance_to_reflection_l340_34074

/-- The distance between a point (2, -4) and its reflection over the y-axis is 4. -/
theorem distance_to_reflection : Real.sqrt ((2 - (-2))^2 + (-4 - (-4))^2) = 4 := by sorry

end distance_to_reflection_l340_34074


namespace amount_in_scientific_notation_l340_34059

-- Define the amount in yuan
def amount : ℕ := 25000000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 2.5 * (10 ^ 10)

-- Theorem statement
theorem amount_in_scientific_notation :
  (amount : ℝ) = scientific_notation := by sorry

end amount_in_scientific_notation_l340_34059


namespace power_relation_l340_34072

theorem power_relation (x m n : ℝ) (h1 : x^m = 6) (h2 : x^n = 9) : x^(2*m - n) = 4 := by
  sorry

end power_relation_l340_34072


namespace bella_roses_count_l340_34090

/-- The number of roses in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of roses Bella received from her parents -/
def roses_from_parents_dozens : ℕ := 2

/-- The number of Bella's dancer friends -/
def number_of_friends : ℕ := 10

/-- The number of roses Bella received from each friend -/
def roses_per_friend : ℕ := 2

/-- The total number of roses Bella received -/
def total_roses : ℕ := roses_from_parents_dozens * dozen + number_of_friends * roses_per_friend

theorem bella_roses_count : total_roses = 44 := by
  sorry

end bella_roses_count_l340_34090


namespace gina_purse_value_l340_34069

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes : ℕ) : ℕ :=
  pennies * coin_value "penny" +
  nickels * coin_value "nickel" +
  dimes * coin_value "dime"

/-- Converts cents to percentage of a dollar -/
def cents_to_percentage (cents : ℕ) : ℚ :=
  (cents : ℚ) / 100

theorem gina_purse_value :
  cents_to_percentage (total_value 2 3 2) = 37 / 100 := by
  sorry

end gina_purse_value_l340_34069


namespace no_polynomial_satisfies_conditions_l340_34022

/-- A polynomial function over real numbers. -/
def PolynomialFunction := ℝ → ℝ

/-- The degree of a polynomial function. -/
noncomputable def degree (f : PolynomialFunction) : ℕ := sorry

/-- Predicate for a function satisfying the given conditions. -/
def satisfiesConditions (f : PolynomialFunction) : Prop :=
  ∀ x : ℝ, f (x + 1) = (f x)^2 ∧ (f x)^2 = f (f x)

theorem no_polynomial_satisfies_conditions :
  ¬ ∃ f : PolynomialFunction, degree f ≥ 1 ∧ satisfiesConditions f := by sorry

end no_polynomial_satisfies_conditions_l340_34022


namespace sum_of_squares_l340_34010

theorem sum_of_squares (x y z a b : ℝ) 
  (sum_eq : x + y + z = a) 
  (sum_prod_eq : x*y + y*z + x*z = b) : 
  x^2 + y^2 + z^2 = a^2 - 2*b := by
  sorry

end sum_of_squares_l340_34010


namespace surface_area_change_after_cube_removal_l340_34089

/-- Represents a rectangular solid with length, width, and height -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Calculates the change in surface area after removing a cube from the center -/
def surfaceAreaChange (solid : RectangularSolid) (cubeSize : ℝ) : ℝ :=
  6 * cubeSize^2

/-- The theorem to be proved -/
theorem surface_area_change_after_cube_removal :
  let original := RectangularSolid.mk 4 3 2
  let cubeSize := 1
  surfaceAreaChange original cubeSize = 6 := by sorry

end surface_area_change_after_cube_removal_l340_34089


namespace sphere_chords_theorem_l340_34098

/-- Represents a sphere with a point inside and three perpendicular chords -/
structure SphereWithChords where
  R : ℝ  -- radius of the sphere
  a : ℝ  -- distance of point A from the center
  h : 0 < R ∧ 0 ≤ a ∧ a < R  -- constraints on R and a

/-- The sum of squares of three mutually perpendicular chords through a point in a sphere -/
def sum_of_squares_chords (s : SphereWithChords) : ℝ := 12 * s.R^2 - 4 * s.a^2

/-- The sum of squares of the segments of three mutually perpendicular chords created by a point in a sphere -/
def sum_of_squares_segments (s : SphereWithChords) : ℝ := 6 * s.R^2 - 2 * s.a^2

/-- Theorem stating the properties of chords in a sphere -/
theorem sphere_chords_theorem (s : SphereWithChords) :
  (sum_of_squares_chords s = 12 * s.R^2 - 4 * s.a^2) ∧
  (sum_of_squares_segments s = 6 * s.R^2 - 2 * s.a^2) := by
  sorry

end sphere_chords_theorem_l340_34098


namespace clown_balloons_l340_34011

/-- The number of additional balloons blown up by the clown -/
def additional_balloons (initial : ℕ) (total : ℕ) : ℕ :=
  total - initial

theorem clown_balloons : additional_balloons 47 60 = 13 := by
  sorry

end clown_balloons_l340_34011


namespace bens_baseball_card_boxes_l340_34015

theorem bens_baseball_card_boxes (basketball_boxes : ℕ) (basketball_cards_per_box : ℕ)
  (baseball_cards_per_box : ℕ) (cards_given_away : ℕ) (cards_left : ℕ) :
  basketball_boxes = 4 →
  basketball_cards_per_box = 10 →
  baseball_cards_per_box = 8 →
  cards_given_away = 58 →
  cards_left = 22 →
  (basketball_boxes * basketball_cards_per_box +
    baseball_cards_per_box * ((cards_given_away + cards_left - basketball_boxes * basketball_cards_per_box) / baseball_cards_per_box)) =
  cards_given_away + cards_left →
  (cards_given_away + cards_left - basketball_boxes * basketball_cards_per_box) / baseball_cards_per_box = 5 :=
by sorry

end bens_baseball_card_boxes_l340_34015


namespace basketball_cards_cost_l340_34040

/-- The cost of Mary's sunglasses -/
def sunglasses_cost : ℕ := 50

/-- The number of sunglasses Mary bought -/
def num_sunglasses : ℕ := 2

/-- The cost of Mary's jeans -/
def jeans_cost : ℕ := 100

/-- The cost of Rose's shoes -/
def shoes_cost : ℕ := 150

/-- The number of basketball card decks Rose bought -/
def num_card_decks : ℕ := 2

/-- Mary's total spending -/
def mary_total : ℕ := num_sunglasses * sunglasses_cost + jeans_cost

/-- Rose's total spending -/
def rose_total : ℕ := shoes_cost + num_card_decks * (mary_total - shoes_cost) / num_card_decks

theorem basketball_cards_cost (h : mary_total = rose_total) : 
  (mary_total - shoes_cost) / num_card_decks = 25 := by
  sorry

end basketball_cards_cost_l340_34040


namespace fibonacci_tetrahedron_volume_zero_l340_34041

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def tetrahedron_vertex (n : ℕ) : ℕ × ℕ × ℕ :=
  (fibonacci n, fibonacci (n + 1), fibonacci (n + 2))

def tetrahedron_volume (n : ℕ) : ℝ :=
  let v1 := tetrahedron_vertex n
  let v2 := tetrahedron_vertex (n + 3)
  let v3 := tetrahedron_vertex (n + 6)
  let v4 := tetrahedron_vertex (n + 9)
  -- Volume calculation would go here
  0  -- Placeholder for the actual volume calculation

theorem fibonacci_tetrahedron_volume_zero (n : ℕ) :
  tetrahedron_volume n = 0 := by
  sorry

#check fibonacci_tetrahedron_volume_zero

end fibonacci_tetrahedron_volume_zero_l340_34041


namespace fourth_term_in_geometric_sequence_l340_34000

/-- Given a geometric sequence of 6 terms where the first term is 5 and the sixth term is 20,
    prove that the fourth term is approximately 6.6. -/
theorem fourth_term_in_geometric_sequence (a : ℕ → ℝ) (h1 : a 1 = 5) (h6 : a 6 = 20)
  (h_geometric : ∀ n ∈ Finset.range 5, a (n + 2) / a (n + 1) = a (n + 1) / a n) :
  ∃ ε > 0, |a 4 - 6.6| < ε :=
sorry

end fourth_term_in_geometric_sequence_l340_34000


namespace candy_problem_l340_34052

theorem candy_problem (x : ℕ) : 
  (x % 12 = 0) →
  (∃ c : ℕ, c ≥ 1 ∧ c ≤ 3 ∧ 
   ((3 * x / 4) * 2 / 3 - 20 - c = 5)) →
  (x = 52 ∨ x = 56) := by
sorry

end candy_problem_l340_34052


namespace eliza_initial_rings_l340_34076

/-- The number of ornamental rings Eliza initially bought -/
def initial_rings : ℕ := 100

/-- The total stock after Eliza's purchase -/
def total_stock : ℕ := 3 * initial_rings

/-- The remaining stock after selling 3/4 of the total -/
def remaining_after_sale : ℕ := (total_stock * 1) / 4

/-- The stock after mother's purchase -/
def stock_after_mother_purchase : ℕ := remaining_after_sale + 300

/-- The final stock -/
def final_stock : ℕ := stock_after_mother_purchase - 150

theorem eliza_initial_rings :
  final_stock = 225 :=
by sorry

end eliza_initial_rings_l340_34076


namespace class_average_theorem_l340_34008

theorem class_average_theorem (total_students : ℕ) (boys_percentage : ℚ) (girls_percentage : ℚ)
  (boys_score : ℚ) (girls_score : ℚ) :
  boys_percentage = 2/5 →
  girls_percentage = 3/5 →
  boys_score = 4/5 →
  girls_score = 9/10 →
  (boys_percentage * boys_score + girls_percentage * girls_score : ℚ) = 43/50 :=
by sorry

end class_average_theorem_l340_34008


namespace intersection_in_fourth_quadrant_l340_34032

/-- The intersection point of f(x) = log_a(x) and g(x) = (1-a)x is in the fourth quadrant when a > 1 -/
theorem intersection_in_fourth_quadrant (a : ℝ) (h : a > 1) :
  ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ Real.log x / Real.log a = (1 - a) * x := by
  sorry

end intersection_in_fourth_quadrant_l340_34032


namespace photo_arrangements_l340_34086

def team_size : ℕ := 6

theorem photo_arrangements (captain_positions : ℕ) (ab_arrangements : ℕ) (remaining_arrangements : ℕ) :
  captain_positions = 2 →
  ab_arrangements = 2 →
  remaining_arrangements = 24 →
  captain_positions * ab_arrangements * remaining_arrangements = 96 :=
by sorry

end photo_arrangements_l340_34086


namespace man_speed_in_still_water_l340_34088

/-- The speed of the man in still water -/
def man_speed : ℝ := 7

/-- The speed of the stream -/
def stream_speed : ℝ := 1

/-- The distance traveled downstream -/
def downstream_distance : ℝ := 40

/-- The distance traveled upstream -/
def upstream_distance : ℝ := 30

/-- The time taken for each journey -/
def journey_time : ℝ := 5

theorem man_speed_in_still_water :
  (downstream_distance / journey_time = man_speed + stream_speed) ∧
  (upstream_distance / journey_time = man_speed - stream_speed) →
  man_speed = 7 := by
  sorry

end man_speed_in_still_water_l340_34088


namespace negation_equivalence_l340_34050

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (3 : ℝ) ^ x + x < 0) ↔ (∀ x : ℝ, (3 : ℝ) ^ x + x ≥ 0) := by sorry

end negation_equivalence_l340_34050


namespace probability_log_base_3_is_integer_l340_34034

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_power_of_three (n : ℕ) : Prop := ∃ k : ℕ, n = 3^k

def count_three_digit_powers_of_three : ℕ := 2

def total_three_digit_numbers : ℕ := 900

theorem probability_log_base_3_is_integer :
  (count_three_digit_powers_of_three : ℚ) / (total_three_digit_numbers : ℚ) = 1 / 450 := by
  sorry

#check probability_log_base_3_is_integer

end probability_log_base_3_is_integer_l340_34034


namespace bookstore_sales_l340_34020

theorem bookstore_sales (tuesday : ℕ) (total : ℕ) : 
  total = tuesday + 3 * tuesday + 9 * tuesday → 
  total = 91 → 
  tuesday = 7 := by
sorry

end bookstore_sales_l340_34020


namespace salt_solution_dilution_l340_34012

/-- Given a salt solution, prove that adding a specific amount of water yields the target concentration -/
theorem salt_solution_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (target_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 40 →
  initial_concentration = 0.25 →
  target_concentration = 0.15 →
  water_added = 400 / 15 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = target_concentration := by
  sorry

#eval (400 : ℚ) / 15

end salt_solution_dilution_l340_34012


namespace ellipse_major_axis_length_l340_34075

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  x_tangent : Point
  has_y_tangent : Bool

/-- Calculate the length of the major axis of an ellipse -/
def majorAxisLength (e : Ellipse) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem: The length of the major axis of the given ellipse is 4 -/
theorem ellipse_major_axis_length :
  let e : Ellipse := {
    focus1 := { x := 4, y := 2 + 2 * Real.sqrt 2 },
    focus2 := { x := 4, y := 2 - 2 * Real.sqrt 2 },
    x_tangent := { x := 4, y := 0 },
    has_y_tangent := true
  }
  majorAxisLength e = 4 := by sorry


end ellipse_major_axis_length_l340_34075


namespace train_distance_theorem_l340_34082

/-- Represents the distance traveled by the second train -/
def x : ℝ := 400

/-- The speed of the first train in km/hr -/
def speed1 : ℝ := 50

/-- The speed of the second train in km/hr -/
def speed2 : ℝ := 40

/-- The additional distance traveled by the first train compared to the second train -/
def additional_distance : ℝ := 100

/-- The total distance between the starting points of the two trains -/
def total_distance : ℝ := x + (x + additional_distance)

theorem train_distance_theorem :
  speed1 > 0 ∧ speed2 > 0 ∧ 
  x / speed2 = (x + additional_distance) / speed1 →
  total_distance = 900 :=
sorry

end train_distance_theorem_l340_34082


namespace negative_square_two_l340_34099

theorem negative_square_two : -2^2 = -4 := by
  sorry

end negative_square_two_l340_34099


namespace total_vegetables_l340_34058

def vegetable_garden_problem (potatoes cucumbers tomatoes peppers carrots : ℕ) : Prop :=
  potatoes = 560 ∧
  cucumbers = potatoes - 132 ∧
  tomatoes = 3 * cucumbers ∧
  peppers = tomatoes / 2 ∧
  carrots = cucumbers + tomatoes

theorem total_vegetables (potatoes cucumbers tomatoes peppers carrots : ℕ) :
  vegetable_garden_problem potatoes cucumbers tomatoes peppers carrots →
  potatoes + cucumbers + tomatoes + peppers + carrots = 4626 := by
  sorry

end total_vegetables_l340_34058


namespace tangent_line_sum_l340_34004

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the theorem
theorem tangent_line_sum (h : ∀ y, y = f 5 ↔ y = -5 + 8) : f 5 + deriv f 5 = 2 := by
  sorry

end tangent_line_sum_l340_34004


namespace michelle_sandwiches_l340_34044

/-- The number of sandwiches Michelle gave to the first co-worker -/
def sandwiches_given : ℕ := sorry

/-- The total number of sandwiches Michelle originally made -/
def total_sandwiches : ℕ := 20

/-- The number of sandwiches left for other co-workers -/
def sandwiches_left : ℕ := 8

/-- The number of sandwiches Michelle kept for herself -/
def sandwiches_kept : ℕ := 2 * sandwiches_given

theorem michelle_sandwiches :
  sandwiches_given + sandwiches_kept + sandwiches_left = total_sandwiches ∧
  sandwiches_given = 4 := by
  sorry

end michelle_sandwiches_l340_34044


namespace odometer_problem_l340_34047

theorem odometer_problem (a b c : ℕ) (ha : a ≥ 1) (hsum : a + b + c = 9)
  (hx : ∃ x : ℕ, x > 0 ∧ 60 * x = 100 * c + 10 * a + b - (100 * a + 10 * b + c)) :
  a^2 + b^2 + c^2 = 51 := by
sorry

end odometer_problem_l340_34047


namespace speed_conversion_l340_34083

theorem speed_conversion (speed_ms : ℝ) (speed_kmh : ℝ) : 
  speed_ms = 0.2790697674418605 ∧ speed_kmh = 1.0046511627906978 → 
  speed_ms = speed_kmh / 3.6 :=
by
  sorry

end speed_conversion_l340_34083


namespace total_kids_signed_up_l340_34017

/-- The number of girls signed up for the talent show. -/
def num_girls : ℕ := 28

/-- The difference between the number of girls and boys signed up. -/
def girl_boy_difference : ℕ := 22

/-- Theorem: The total number of kids signed up for the talent show is 34. -/
theorem total_kids_signed_up : 
  num_girls + (num_girls - girl_boy_difference) = 34 := by
  sorry

end total_kids_signed_up_l340_34017


namespace eightieth_number_is_eighty_l340_34042

def game_sequence (n : ℕ) : ℕ := n

theorem eightieth_number_is_eighty : game_sequence 80 = 80 := by sorry

end eightieth_number_is_eighty_l340_34042


namespace problem_solution_l340_34001

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * 2^(x+1) + (k-3) * 2^(-x)

theorem problem_solution (k : ℝ) (t : ℝ) :
  (∀ x, f k (-x) = -(f k x)) →
  (∀ x ∈ Set.Icc 1 3, f k (x^2 - x) + f k (t*x + 4) > 0) →
  (k = 1 ∧
   (∀ x₁ x₂, x₁ < x₂ → f k x₁ < f k x₂) ∧
   t > -3) :=
sorry

end problem_solution_l340_34001


namespace gain_percent_calculation_l340_34023

theorem gain_percent_calculation (MP : ℝ) (MP_pos : MP > 0) : 
  let CP := 0.64 * MP
  let SP := 0.86 * MP
  let gain := SP - CP
  let gain_percent := (gain / CP) * 100
  gain_percent = 34.375 := by
sorry

end gain_percent_calculation_l340_34023


namespace arithmetic_sequence_partial_sum_l340_34007

-- Define the arithmetic sequence and its partial sums
def arithmetic_sequence (n : ℕ) : ℝ := sorry
def S (n : ℕ) : ℝ := sorry

-- State the theorem
theorem arithmetic_sequence_partial_sum :
  S 3 = 6 ∧ S 9 = 27 → S 6 = 15 := by sorry

end arithmetic_sequence_partial_sum_l340_34007


namespace greatest_five_digit_with_product_90_l340_34054

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def digit_product (n : ℕ) : ℕ :=
  (n / 10000) * ((n / 1000) % 10) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def digit_sum (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem greatest_five_digit_with_product_90 :
  ∃ M : ℕ, is_five_digit M ∧ 
    digit_product M = 90 ∧ 
    (∀ n : ℕ, is_five_digit n → digit_product n = 90 → n ≤ M) ∧
    digit_sum M = 17 :=
sorry

end greatest_five_digit_with_product_90_l340_34054


namespace minimum_balls_in_box_l340_34043

theorem minimum_balls_in_box (blue : ℕ) (white : ℕ) (total : ℕ) : 
  white = 8 * blue →
  total = blue + white →
  (∀ drawn : ℕ, drawn = 100 → drawn > white) →
  total ≥ 108 := by
sorry

end minimum_balls_in_box_l340_34043


namespace function_identity_l340_34073

def NatPos := {n : ℕ // n > 0}

theorem function_identity (f : NatPos → NatPos) 
  (h : ∀ m n : NatPos, (m.val ^ 2 + (f n).val) ∣ (m.val * (f m).val + n.val)) :
  ∀ n : NatPos, f n = n := by
  sorry

end function_identity_l340_34073


namespace maddie_total_cost_l340_34084

/-- Calculates the total cost of Maddie's beauty products purchase --/
def total_cost (palette_price : ℚ) (palette_quantity : ℕ) 
               (lipstick_price : ℚ) (lipstick_quantity : ℕ)
               (hair_color_price : ℚ) (hair_color_quantity : ℕ) : ℚ :=
  palette_price * palette_quantity + 
  lipstick_price * lipstick_quantity + 
  hair_color_price * hair_color_quantity

/-- Theorem stating that Maddie's total cost is $67 --/
theorem maddie_total_cost : 
  total_cost 15 3 (5/2) 4 4 3 = 67 := by
  sorry

end maddie_total_cost_l340_34084


namespace elizabeth_pencil_purchase_l340_34070

/-- Given Elizabeth's shopping scenario, prove she can buy exactly 5 pencils. -/
theorem elizabeth_pencil_purchase (
  initial_money : ℚ)
  (pen_cost : ℚ)
  (pencil_cost : ℚ)
  (pens_to_buy : ℕ)
  (h1 : initial_money = 20)
  (h2 : pen_cost = 2)
  (h3 : pencil_cost = 1.6)
  (h4 : pens_to_buy = 6) :
  (initial_money - pens_to_buy * pen_cost) / pencil_cost = 5 := by
sorry

end elizabeth_pencil_purchase_l340_34070


namespace age_ratio_problem_l340_34096

/-- Given Tom's current age t and Lily's current age l, prove that the smallest positive integer x
    that satisfies (t + x) / (l + x) = 3 is 22, where t and l satisfy the given conditions. -/
theorem age_ratio_problem (t l : ℕ) (h1 : t - 3 = 5 * (l - 3)) (h2 : t - 8 = 6 * (l - 8)) :
  (∃ x : ℕ, x > 0 ∧ (t + x : ℚ) / (l + x) = 3 ∧ ∀ y : ℕ, y > 0 → (t + y : ℚ) / (l + y) = 3 → x ≤ y) →
  (∃ x : ℕ, x = 22 ∧ x > 0 ∧ (t + x : ℚ) / (l + x) = 3 ∧ ∀ y : ℕ, y > 0 → (t + y : ℚ) / (l + y) = 3 → x ≤ y) :=
by
  sorry


end age_ratio_problem_l340_34096
