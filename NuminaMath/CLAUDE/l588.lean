import Mathlib

namespace unused_types_count_l588_58816

/-- The number of natural resources --/
def num_resources : ℕ := 6

/-- The number of types of nature use developed --/
def types_developed : ℕ := 23

/-- The total number of possible combinations of resource usage --/
def total_combinations : ℕ := 2^num_resources

/-- The number of valid combinations (excluding the all-zero combination) --/
def valid_combinations : ℕ := total_combinations - 1

/-- The number of unused types of nature use --/
def unused_types : ℕ := valid_combinations - types_developed

theorem unused_types_count : unused_types = 40 := by
  sorry

end unused_types_count_l588_58816


namespace monotonicity_condition_l588_58875

theorem monotonicity_condition (ω : ℝ) (h_ω_pos : ω > 0) :
  (∀ x ∈ Set.Ioo 0 (π / 3), Monotone (fun x => Real.sin (ω * x + π / 6))) ↔ ω ∈ Set.Ioc 0 1 := by
  sorry

end monotonicity_condition_l588_58875


namespace mikails_age_correct_l588_58833

/-- Mikail's age on his birthday -/
def mikails_age : ℕ := 9

/-- Amount of money Mikail receives per year of age -/
def money_per_year : ℕ := 5

/-- Total amount of money Mikail receives on his birthday -/
def total_money : ℕ := 45

/-- Theorem: Mikail's age is correct given the money he receives -/
theorem mikails_age_correct : mikails_age = total_money / money_per_year := by
  sorry

end mikails_age_correct_l588_58833


namespace club_truncator_probability_l588_58842

/-- The number of matches played by Club Truncator -/
def num_matches : ℕ := 8

/-- The probability of winning, losing, or tying a single match -/
def single_match_prob : ℚ := 1/3

/-- The probability of finishing with more wins than losses -/
def more_wins_prob : ℚ := 5483/13122

theorem club_truncator_probability :
  (num_matches = 8) →
  (single_match_prob = 1/3) →
  (more_wins_prob = 5483/13122) :=
by sorry

end club_truncator_probability_l588_58842


namespace young_in_sample_is_seven_l588_58838

/-- Represents the number of employees in each age group and the sample size --/
structure EmployeeData where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ
  sampleSize : ℕ

/-- Calculates the number of young employees in a stratified sample --/
def youngInSample (data : EmployeeData) : ℕ :=
  (data.young * data.sampleSize) / data.total

/-- Theorem stating that for the given employee data, the number of young employees in the sample is 7 --/
theorem young_in_sample_is_seven (data : EmployeeData)
  (h1 : data.total = 750)
  (h2 : data.young = 350)
  (h3 : data.middleAged = 250)
  (h4 : data.elderly = 150)
  (h5 : data.sampleSize = 15) :
  youngInSample data = 7 := by
  sorry


end young_in_sample_is_seven_l588_58838


namespace least_positive_integer_congruence_l588_58891

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 3742) % 17 = 1578 % 17 ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 3742) % 17 = 1578 % 17 → x ≤ y :=
by sorry

end least_positive_integer_congruence_l588_58891


namespace sufficient_not_necessary_l588_58822

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 2 → x > 0) ∧ 
  (∃ x : ℝ, x > 0 ∧ ¬(x > 2)) := by
sorry

end sufficient_not_necessary_l588_58822


namespace afternoon_and_evening_emails_l588_58820

def morning_emails : ℕ := 4
def afternoon_emails : ℕ := 5
def evening_emails : ℕ := 8

theorem afternoon_and_evening_emails :
  afternoon_emails + evening_emails = 13 :=
by sorry

end afternoon_and_evening_emails_l588_58820


namespace wills_initial_amount_l588_58882

/-- The amount of money Will's mom gave him initially -/
def initial_amount : ℕ := 74

/-- The cost of the sweater Will bought -/
def sweater_cost : ℕ := 9

/-- The cost of the T-shirt Will bought -/
def tshirt_cost : ℕ := 11

/-- The cost of the shoes Will bought -/
def shoes_cost : ℕ := 30

/-- The refund percentage for the returned shoes -/
def refund_percentage : ℚ := 90 / 100

/-- The amount of money Will has left after all transactions -/
def money_left : ℕ := 51

theorem wills_initial_amount :
  initial_amount = 
    money_left + 
    sweater_cost + 
    tshirt_cost + 
    shoes_cost - 
    (↑shoes_cost * refund_percentage).floor :=
by sorry

end wills_initial_amount_l588_58882


namespace classroom_gpa_l588_58886

theorem classroom_gpa (N : ℝ) (h : N > 0) :
  let gpa_one_third := 54
  let gpa_whole := 48
  let gpa_rest := (3 * gpa_whole - gpa_one_third) / 2
  gpa_rest = 45 := by sorry

end classroom_gpa_l588_58886


namespace solve_exponential_equation_l588_58832

theorem solve_exponential_equation :
  ∃ x : ℝ, (5 : ℝ) ^ (3 * x) = Real.sqrt 125 ∧ x = (1 : ℝ) / 2 := by
  sorry

end solve_exponential_equation_l588_58832


namespace probability_two_defective_out_of_ten_l588_58899

/-- Given a set of products with some defective ones, this function calculates
    the probability of randomly selecting a defective product. -/
def probability_defective (total : ℕ) (defective : ℕ) : ℚ :=
  defective / total

/-- Theorem stating that for 10 products with 2 defective ones,
    the probability of randomly selecting a defective product is 1/5. -/
theorem probability_two_defective_out_of_ten :
  probability_defective 10 2 = 1 / 5 := by
  sorry

end probability_two_defective_out_of_ten_l588_58899


namespace max_sum_on_circle_max_sum_achievable_l588_58888

theorem max_sum_on_circle (x y : ℤ) : x^2 + y^2 = 100 → x + y ≤ 14 := by
  sorry

theorem max_sum_achievable : ∃ x y : ℤ, x^2 + y^2 = 100 ∧ x + y = 14 := by
  sorry

end max_sum_on_circle_max_sum_achievable_l588_58888


namespace number_equation_solution_l588_58830

theorem number_equation_solution : 
  ∀ x : ℝ, (2/5 : ℝ) * x - 3 * ((1/4 : ℝ) * x) + 7 = 14 → x = -20 := by
sorry

end number_equation_solution_l588_58830


namespace peters_leaf_raking_l588_58887

/-- Given that Peter rakes 3 bags of leaves in 15 minutes at a constant rate,
    prove that it will take him 40 minutes to rake 8 bags of leaves. -/
theorem peters_leaf_raking (rate : ℚ) : 
  (rate * 15 = 3) → (rate * 40 = 8) :=
by sorry

end peters_leaf_raking_l588_58887


namespace olympiad_solution_l588_58810

def olympiad_problem (N_a N_b N_c N_ab N_ac N_bc N_abc : ℕ) : Prop :=
  let total := N_a + N_b + N_c + N_ab + N_ac + N_bc + N_abc
  let B_not_A := N_b + N_bc
  let C_not_A := N_c + N_bc
  let A_and_others := N_ab + N_ac + N_abc
  let only_one := N_a + N_b + N_c
  total = 25 ∧
  B_not_A = 2 * C_not_A ∧
  N_a = A_and_others + 1 ∧
  2 * N_a = only_one

theorem olympiad_solution :
  ∀ N_a N_b N_c N_ab N_ac N_bc N_abc,
  olympiad_problem N_a N_b N_c N_ab N_ac N_bc N_abc →
  N_b = 6 := by
sorry

end olympiad_solution_l588_58810


namespace second_car_distance_l588_58871

/-- Calculates the distance traveled by the second car given the initial separation,
    the distance traveled by the first car, and the final distance between the cars. -/
def distance_traveled_by_second_car (initial_separation : ℝ) (distance_first_car : ℝ) (final_distance : ℝ) : ℝ :=
  initial_separation - (distance_first_car + final_distance)

/-- Theorem stating that given the conditions of the problem, 
    the second car must have traveled 87 km. -/
theorem second_car_distance : 
  let initial_separation : ℝ := 150
  let distance_first_car : ℝ := 25
  let final_distance : ℝ := 38
  distance_traveled_by_second_car initial_separation distance_first_car final_distance = 87 := by
  sorry

#eval distance_traveled_by_second_car 150 25 38

end second_car_distance_l588_58871


namespace intersection_A_complement_B_l588_58817

def U : Finset Int := {-2, -1, 0, 1, 2}
def A : Finset Int := {1, 2}
def B : Finset Int := {-2, -1, 2}

theorem intersection_A_complement_B : A ∩ (U \ B) = {1} := by
  sorry

end intersection_A_complement_B_l588_58817


namespace simplify_expression_l588_58890

theorem simplify_expression (y : ℝ) : 
  3 * y - 7 * y^2 + 4 - (5 + 3 * y - 7 * y^2) = 0 * y^2 + 0 * y - 1 := by
  sorry

end simplify_expression_l588_58890


namespace square_difference_formula_l588_58857

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 8/15) (h2 : x - y = 2/15) : x^2 - y^2 = 16/225 := by
  sorry

end square_difference_formula_l588_58857


namespace instantaneous_rate_of_change_l588_58873

/-- Given a curve y = x^2 + 2x, prove that if the instantaneous rate of change
    at a point M is 6, then the coordinates of point M are (2, 8). -/
theorem instantaneous_rate_of_change (x y : ℝ) : 
  y = x^2 + 2*x →                             -- Curve equation
  (2*x + 2 : ℝ) = 6 →                         -- Instantaneous rate of change is 6
  (x, y) = (2, 8) :=                          -- Coordinates of point M
by sorry

end instantaneous_rate_of_change_l588_58873


namespace linear_function_equation_l588_58884

/-- A linear function passing through (3, 2) and intersecting positive x and y axes --/
structure LinearFunctionWithConstraints where
  f : ℝ → ℝ
  is_linear : ∀ x y c : ℝ, f (x + y) = f x + f y ∧ f (c * x) = c * f x
  passes_through : f 3 = 2
  intersects_x_axis : ∃ a : ℝ, a > 0 ∧ f a = 0
  intersects_y_axis : ∃ b : ℝ, b > 0 ∧ f 0 = b
  sum_of_intersects : let a := Classical.choose (intersects_x_axis)
                      let b := Classical.choose (intersects_y_axis)
                      a + b = 12

/-- The equation of the linear function satisfies the given constraints --/
theorem linear_function_equation (l : LinearFunctionWithConstraints) :
  (∀ x, l.f x = -2 * x + 8) ∨ (∀ x, l.f x = -1/3 * x + 3) := by
  sorry

end linear_function_equation_l588_58884


namespace swim_club_prep_course_count_l588_58827

/-- Represents a swim club with members, some of whom have passed a lifesaving test
    and some of whom have taken a preparatory course. -/
structure SwimClub where
  totalMembers : ℕ
  passedTest : ℕ
  notPassedNotTakenCourse : ℕ

/-- Calculates the number of members who have taken the preparatory course
    but not passed the test in a given swim club. -/
def membersInPreparatoryNotPassed (club : SwimClub) : ℕ :=
  club.totalMembers - club.passedTest - club.notPassedNotTakenCourse

/-- Theorem stating that in a swim club with 50 members, where 30% have passed
    the lifesaving test and 30 of those who haven't passed haven't taken the
    preparatory course, the number of members who have taken the preparatory
    course but not passed the test is 5. -/
theorem swim_club_prep_course_count :
  let club : SwimClub := {
    totalMembers := 50,
    passedTest := 15,  -- 30% of 50
    notPassedNotTakenCourse := 30
  }
  membersInPreparatoryNotPassed club = 5 := by
  sorry


end swim_club_prep_course_count_l588_58827


namespace second_caterer_cheaper_l588_58895

/-- Represents a caterer's pricing structure -/
structure Caterer where
  basic_fee : ℕ
  per_person : ℕ

/-- Calculates the total cost for a given number of people -/
def total_cost (c : Caterer) (people : ℕ) : ℕ :=
  c.basic_fee + c.per_person * people

/-- The first caterer's pricing structure -/
def caterer1 : Caterer :=
  { basic_fee := 150, per_person := 18 }

/-- The second caterer's pricing structure -/
def caterer2 : Caterer :=
  { basic_fee := 250, per_person := 15 }

/-- Theorem stating the minimum number of people for which the second caterer is cheaper -/
theorem second_caterer_cheaper : 
  (∀ n : ℕ, n ≥ 34 → total_cost caterer2 n < total_cost caterer1 n) ∧
  (∀ n : ℕ, n < 34 → total_cost caterer2 n ≥ total_cost caterer1 n) :=
sorry

end second_caterer_cheaper_l588_58895


namespace additional_investment_rate_l588_58811

/-- Proves that the interest rate of an additional investment is 10% given specific conditions --/
theorem additional_investment_rate (initial_investment : ℝ) (initial_rate : ℝ) 
  (additional_investment : ℝ) (total_rate : ℝ) : 
  initial_investment = 2400 →
  initial_rate = 0.05 →
  additional_investment = 600 →
  total_rate = 0.06 →
  (initial_investment * initial_rate + additional_investment * 0.1) / 
    (initial_investment + additional_investment) = total_rate := by
  sorry

#check additional_investment_rate

end additional_investment_rate_l588_58811


namespace esperanza_salary_l588_58858

/-- Calculates the gross monthly salary given the specified expenses and savings. -/
def gross_monthly_salary (rent food_ratio mortgage_ratio savings tax_ratio : ℝ) : ℝ :=
  let food := food_ratio * rent
  let mortgage := mortgage_ratio * food
  let taxes := tax_ratio * savings
  rent + food + mortgage + savings + taxes

/-- Theorem stating the gross monthly salary under given conditions. -/
theorem esperanza_salary : 
  gross_monthly_salary 600 (3/5) 3 2000 (2/5) = 4840 := by
  sorry

end esperanza_salary_l588_58858


namespace floor_equation_equivalence_l588_58840

def solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, (k + 1/5 ≤ x ∧ x < k + 1/3) ∨
           (k + 2/5 ≤ x ∧ x < k + 3/5) ∨
           (k + 2/3 ≤ x ∧ x < k + 4/5)

theorem floor_equation_equivalence (x : ℝ) :
  ⌊(5 : ℝ) * x⌋ = ⌊(3 : ℝ) * x⌋ + 2 * ⌊x⌋ + 1 ↔ solution_set x :=
by sorry

end floor_equation_equivalence_l588_58840


namespace hyperbola_equation_l588_58825

/-- Given a hyperbola with the standard equation (x²/a² - y²/b² = 1),
    one focus at (-2, 0), and the angle between asymptotes is 60°,
    prove that its equation is either x² - y²/3 = 1 or x²/3 - y² = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (a^2 + b^2 = 4) →
  (b / a = Real.sqrt 3 ∨ b / a = Real.sqrt 3 / 3) →
  ((∀ x y : ℝ, x^2 - y^2 / 3 = 1) ∨ (∀ x y : ℝ, x^2 / 3 - y^2 = 1)) :=
by sorry

end hyperbola_equation_l588_58825


namespace completing_square_equivalence_l588_58855

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 4*x - 7 = 0 ↔ (x - 2)^2 = 11 := by
  sorry

end completing_square_equivalence_l588_58855


namespace max_product_of_functions_l588_58815

theorem max_product_of_functions (f h : ℝ → ℝ) 
  (hf : ∀ x, f x ∈ Set.Icc (-5) 3) 
  (hh : ∀ x, h x ∈ Set.Icc (-3) 4) : 
  (⨆ x, f x * h x) = 20 := by
  sorry

end max_product_of_functions_l588_58815


namespace toy_store_revenue_ratio_l588_58852

theorem toy_store_revenue_ratio : 
  ∀ (N D J : ℝ),
  J = (1 / 2) * N →
  D = (10 / 3) * ((N + J) / 2) →
  N / D = 2 / 5 := by
sorry

end toy_store_revenue_ratio_l588_58852


namespace hyperbola_standard_equation_l588_58889

/-- The standard equation of a hyperbola passing through specific points and sharing asymptotes with another hyperbola -/
theorem hyperbola_standard_equation :
  ∀ (x y : ℝ → ℝ),
  (∃ (t : ℝ), x t = -3 ∧ y t = 2 * Real.sqrt 7) →
  (∃ (t : ℝ), x t = 6 * Real.sqrt 2 ∧ y t = -7) →
  (∃ (t : ℝ), x t = 2 ∧ y t = 2 * Real.sqrt 3) →
  (∀ (t : ℝ), (x t)^2 / 4 - (y t)^2 / 3 = 1 ↔ ∃ (k : ℝ), k * ((x t)^2 / 4 - (y t)^2 / 3) = k) →
  ∀ (t : ℝ), (y t)^2 / 9 - (x t)^2 / 12 = 1 :=
by sorry

end hyperbola_standard_equation_l588_58889


namespace trig_problem_l588_58806

theorem trig_problem (x : ℝ) 
  (h1 : -π/2 < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x - Real.cos x = -7/5) ∧ 
  (4 * Real.sin x * Real.cos x - Real.cos x^2 = -64/25) := by
sorry

end trig_problem_l588_58806


namespace parabola_point_relationship_l588_58819

/-- A parabola with equation y = x² - 4x - m -/
structure Parabola where
  m : ℝ

/-- A point on the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on a parabola -/
def lies_on (p : Point) (para : Parabola) : Prop :=
  p.y = p.x^2 - 4*p.x - para.m

theorem parabola_point_relationship (para : Parabola) (A B C : Point)
    (hA : A.x = 2) (hB : B.x = -3) (hC : C.x = -1)
    (onA : lies_on A para) (onB : lies_on B para) (onC : lies_on C para) :
    A.y < C.y ∧ C.y < B.y := by
  sorry

end parabola_point_relationship_l588_58819


namespace rectangle_side_multiple_of_6_l588_58851

/-- A rectangle constructed from 1 x 6 rectangles -/
structure Rectangle :=
  (length : ℕ)
  (width : ℕ)
  (area : ℕ)
  (area_eq : area = length * width)
  (divisible_by_6 : 6 ∣ area)

/-- Theorem: One side of a rectangle constructed from 1 x 6 rectangles is a multiple of 6 -/
theorem rectangle_side_multiple_of_6 (r : Rectangle) : 
  6 ∣ r.length ∨ 6 ∣ r.width :=
sorry

end rectangle_side_multiple_of_6_l588_58851


namespace proportional_function_decreasing_l588_58831

theorem proportional_function_decreasing (k : ℝ) (h1 : k ≠ 0) :
  (∀ x₁ x₂ y₁ y₂ : ℝ, x₁ < x₂ → k * x₁ = y₁ → k * x₂ = y₂ → y₁ > y₂) → k < 0 :=
by sorry

end proportional_function_decreasing_l588_58831


namespace pyramid_height_is_two_main_theorem_l588_58826

/-- A right square pyramid with given properties -/
structure RightSquarePyramid where
  top_side : ℝ
  bottom_side : ℝ
  lateral_area : ℝ
  height : ℝ

/-- The theorem stating the height of the pyramid is 2 -/
theorem pyramid_height_is_two (p : RightSquarePyramid) : p.height = 2 :=
  by
  have h1 : p.top_side = 3 := by sorry
  have h2 : p.bottom_side = 6 := by sorry
  have h3 : p.lateral_area = p.top_side ^ 2 + p.bottom_side ^ 2 := by sorry
  sorry

/-- Main theorem combining all conditions and the result -/
theorem main_theorem : ∃ (p : RightSquarePyramid), 
  p.top_side = 3 ∧ 
  p.bottom_side = 6 ∧ 
  p.lateral_area = p.top_side ^ 2 + p.bottom_side ^ 2 ∧
  p.height = 2 :=
by
  sorry

end pyramid_height_is_two_main_theorem_l588_58826


namespace mushroom_price_per_unit_l588_58800

theorem mushroom_price_per_unit (total_mushrooms day2_mushrooms day1_revenue : ℕ) : 
  total_mushrooms = 65 →
  day2_mushrooms = 12 →
  day1_revenue = 58 →
  (total_mushrooms - day2_mushrooms - 2 * day2_mushrooms) * 2 = day1_revenue :=
by
  sorry

end mushroom_price_per_unit_l588_58800


namespace even_operations_l588_58879

theorem even_operations (a b : ℤ) (ha : Even a) (hb : Odd b) : 
  Even (a * b) ∧ Even (a * a) := by
  sorry

end even_operations_l588_58879


namespace platform_length_l588_58849

/-- Given a train of length 1200 m that crosses a tree in 120 sec and passes a platform in 230 sec,
    the length of the platform is 1100 m. -/
theorem platform_length (train_length : ℝ) (tree_crossing_time : ℝ) (platform_passing_time : ℝ) :
  train_length = 1200 →
  tree_crossing_time = 120 →
  platform_passing_time = 230 →
  let train_speed := train_length / tree_crossing_time
  let platform_length := train_speed * platform_passing_time - train_length
  platform_length = 1100 := by
  sorry

end platform_length_l588_58849


namespace fraction_equality_l588_58856

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (2 * x - 8 * y) = 3) : 
  (x + 4 * y) / (4 * x - y) = 1 / 3 := by
sorry

end fraction_equality_l588_58856


namespace lollipop_sequence_l588_58859

theorem lollipop_sequence (a b c d e : ℕ) : 
  a + b + c + d + e = 100 →
  b = a + 6 →
  c = b + 6 →
  d = c + 6 →
  e = d + 6 →
  c = 20 := by sorry

end lollipop_sequence_l588_58859


namespace sequence_properties_l588_58847

def S (n : ℕ) : ℝ := -n^2 + 7*n + 1

def a (n : ℕ) : ℝ :=
  if n = 1 then 7
  else -2*n + 8

theorem sequence_properties :
  (∀ n > 4, a n < 0) ∧
  (∀ n : ℕ, n ≠ 0 → S n ≤ S 3 ∧ S n ≤ S 4) :=
sorry

end sequence_properties_l588_58847


namespace complex_fraction_simplification_l588_58876

theorem complex_fraction_simplification (x y : ℚ) 
  (hx : x = 3) 
  (hy : y = 4) : 
  (1 / (y + 1)) / (1 / (x - 1)) = 2/5 := by
  sorry

end complex_fraction_simplification_l588_58876


namespace hardware_store_lcm_l588_58809

theorem hardware_store_lcm : Nat.lcm 13 (Nat.lcm 19 (Nat.lcm 8 (Nat.lcm 11 (Nat.lcm 17 23)))) = 772616 := by
  sorry

end hardware_store_lcm_l588_58809


namespace income_expenditure_ratio_l588_58880

def income : ℕ := 21000
def savings : ℕ := 7000
def expenditure : ℕ := income - savings

theorem income_expenditure_ratio : 
  (income : ℚ) / (expenditure : ℚ) = 3 / 2 := by sorry

end income_expenditure_ratio_l588_58880


namespace car_speed_problem_l588_58860

/-- Given a car traveling for two hours, prove that if its speed in the second hour
    is 45 km/h and its average speed over the two hours is 55 km/h, then its speed
    in the first hour must be 65 km/h. -/
theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ)
    (h1 : speed_second_hour = 45)
    (h2 : average_speed = 55) :
    let speed_first_hour := 2 * average_speed - speed_second_hour
    speed_first_hour = 65 := by
  sorry

end car_speed_problem_l588_58860


namespace science_fair_sophomores_fraction_l588_58804

theorem science_fair_sophomores_fraction (s j n : ℕ) : 
  s > 0 → -- Ensure s is positive to avoid division by zero
  s = j → -- Equal number of sophomores and juniors
  j = n → -- Number of juniors equals number of seniors
  (4 * s / 5 : ℚ) / ((4 * s / 5 : ℚ) + (3 * j / 4 : ℚ) + (n / 3 : ℚ)) = 240 / 565 := by
  sorry

#check science_fair_sophomores_fraction

end science_fair_sophomores_fraction_l588_58804


namespace range_of_a_l588_58878

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l588_58878


namespace max_value_of_f_l588_58853

def S : Finset ℕ := {0, 1, 2, 3, 4}

def f (a b c d e : ℕ) : ℕ := e * c^a + b - d

theorem max_value_of_f :
  ∃ (a b c d e : ℕ),
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    f a b c d e = 39 ∧
    ∀ (a' b' c' d' e' : ℕ),
      a' ∈ S → b' ∈ S → c' ∈ S → d' ∈ S → e' ∈ S →
      a' ≠ b' → a' ≠ c' → a' ≠ d' → a' ≠ e' →
      b' ≠ c' → b' ≠ d' → b' ≠ e' →
      c' ≠ d' → c' ≠ e' →
      d' ≠ e' →
      f a' b' c' d' e' ≤ 39 :=
by sorry

end max_value_of_f_l588_58853


namespace men_count_l588_58823

/-- The number of women in the arrangement -/
def num_women : ℕ := 2

/-- The number of distinct alternating arrangements -/
def num_arrangements : ℕ := 4

/-- A function that calculates the number of distinct alternating arrangements
    given the number of men and women -/
def calc_arrangements (men women : ℕ) : ℕ := sorry

theorem men_count :
  ∃ (men : ℕ), men > 0 ∧ calc_arrangements men num_women = num_arrangements :=
sorry

end men_count_l588_58823


namespace birds_in_marsh_l588_58839

theorem birds_in_marsh (geese ducks : ℕ) (h1 : geese = 58) (h2 : ducks = 37) :
  geese + ducks = 95 := by
  sorry

end birds_in_marsh_l588_58839


namespace repeating_decimal_division_l588_58870

theorem repeating_decimal_division : 
  let a := (36 : ℚ) / 99
  let b := (12 : ℚ) / 99
  a / b = 3 := by sorry

end repeating_decimal_division_l588_58870


namespace percent_relation_l588_58803

theorem percent_relation (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : b = 0.5 * a) :
  c = 0.5 * b := by
  sorry

end percent_relation_l588_58803


namespace baseball_team_score_l588_58801

theorem baseball_team_score :
  let total_players : ℕ := 9
  let high_scorers : ℕ := 5
  let high_scorer_average : ℕ := 50
  let low_scorer_average : ℕ := 5
  let low_scorers : ℕ := total_players - high_scorers
  let total_score : ℕ := high_scorers * high_scorer_average + low_scorers * low_scorer_average
  total_score = 270 := by sorry

end baseball_team_score_l588_58801


namespace factorial_sum_mod_30_l588_58805

theorem factorial_sum_mod_30 : (1 + 2 + 6 + 24 + 120) % 30 = 3 := by sorry

end factorial_sum_mod_30_l588_58805


namespace correct_seating_count_l588_58885

/-- Number of Democrats in the Senate committee -/
def num_democrats : ℕ := 6

/-- Number of Republicans in the Senate committee -/
def num_republicans : ℕ := 6

/-- Number of Independents in the Senate committee -/
def num_independents : ℕ := 2

/-- Total number of committee members -/
def total_members : ℕ := num_democrats + num_republicans + num_independents

/-- Function to calculate the number of valid seating arrangements -/
def seating_arrangements : ℕ :=
  12 * (Nat.factorial 10) / 2

/-- Theorem stating the number of valid seating arrangements -/
theorem correct_seating_count :
  seating_arrangements = 21772800 := by sorry

end correct_seating_count_l588_58885


namespace diamond_four_three_l588_58821

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := 4 * a + 3 * b - 2 * a * b

-- Theorem statement
theorem diamond_four_three : diamond 4 3 = 1 := by
  sorry

end diamond_four_three_l588_58821


namespace interest_problem_l588_58850

/-- Given a sum of money put at simple interest for 3 years, if increasing the
    interest rate by 2% results in Rs. 360 more interest, then the sum is Rs. 6000. -/
theorem interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 2) * 3 / 100 - P * R * 3 / 100 = 360) → P = 6000 := by
  sorry

end interest_problem_l588_58850


namespace max_product_sum_2024_l588_58872

theorem max_product_sum_2024 :
  (∃ (a b : ℤ), a + b = 2024 ∧ a * b = 1024144) ∧
  (∀ (x y : ℤ), x + y = 2024 → x * y ≤ 1024144) := by
  sorry

end max_product_sum_2024_l588_58872


namespace polynomial_divisibility_l588_58843

theorem polynomial_divisibility (a b c d m : ℤ) 
  (h1 : (5 : ℤ) ∣ (a * m^3 + b * m^2 + c * m + d))
  (h2 : ¬((5 : ℤ) ∣ d)) :
  ∃ n : ℤ, (5 : ℤ) ∣ (d * n^3 + c * n^2 + b * n + a) := by
  sorry

end polynomial_divisibility_l588_58843


namespace common_chord_equation_l588_58807

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 8 = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

/-- The equation of the line on which the common chord lies -/
def common_chord (x y : ℝ) : Prop := x - y + 1 = 0

/-- Theorem stating that the common chord of the two circles lies on the line x - y + 1 = 0 -/
theorem common_chord_equation :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → common_chord x y :=
sorry

end common_chord_equation_l588_58807


namespace function_positive_range_l588_58836

-- Define the function f(x) = -x^2 + 2x + 3
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- State the theorem
theorem function_positive_range :
  ∀ x : ℝ, f x > 0 ↔ -1 < x ∧ x < 3 := by
sorry

end function_positive_range_l588_58836


namespace train_speed_l588_58862

/-- A train passes a pole in 5 seconds and crosses a 360-meter long stationary train in 25 seconds. -/
theorem train_speed (pole_passing_time : ℝ) (stationary_train_length : ℝ) (crossing_time : ℝ)
  (h1 : pole_passing_time = 5)
  (h2 : stationary_train_length = 360)
  (h3 : crossing_time = 25) :
  ∃ (speed : ℝ), speed = 18 ∧ 
    speed * pole_passing_time = speed * crossing_time - stationary_train_length :=
by sorry

end train_speed_l588_58862


namespace min_value_one_iff_k_eq_two_ninths_l588_58808

/-- The expression as a function of x, y, and k -/
def f (x y k : ℝ) : ℝ := 9*x^2 - 8*k*x*y + (4*k^2 + 3)*y^2 - 6*x - 6*y + 9

/-- The theorem stating the minimum value of f is 1 iff k = 2/9 -/
theorem min_value_one_iff_k_eq_two_ninths :
  (∀ x y : ℝ, f x y (2/9 : ℝ) ≥ 1) ∧ (∃ x y : ℝ, f x y (2/9 : ℝ) = 1) ↔
  ∀ k : ℝ, (∀ x y : ℝ, f x y k ≥ 1) ∧ (∃ x y : ℝ, f x y k = 1) → k = 2/9 :=
sorry

end min_value_one_iff_k_eq_two_ninths_l588_58808


namespace percentage_equation_solution_l588_58874

theorem percentage_equation_solution : 
  ∃ x : ℝ, 45 * x = (35 / 100) * 900 ∧ x = 7 := by sorry

end percentage_equation_solution_l588_58874


namespace bricks_per_square_meter_l588_58837

-- Define the parameters
def num_rooms : ℕ := 5
def room_length : ℝ := 4
def room_width : ℝ := 5
def room_height : ℝ := 2
def bricks_per_room : ℕ := 340

-- Define the theorem
theorem bricks_per_square_meter :
  let room_area : ℝ := room_length * room_width
  let bricks_per_sq_meter : ℝ := bricks_per_room / room_area
  bricks_per_sq_meter = 17 := by sorry

end bricks_per_square_meter_l588_58837


namespace circle_area_relation_l588_58898

/-- Two circles are tangent if they touch at exactly one point. -/
def CirclesTangent (A B : Set ℝ × ℝ) : Prop := sorry

/-- A circle passes through a point if the point lies on the circle's circumference. -/
def CirclePassesThrough (C : Set ℝ × ℝ) (p : ℝ × ℝ) : Prop := sorry

/-- The center of a circle. -/
def CircleCenter (C : Set ℝ × ℝ) : ℝ × ℝ := sorry

/-- The area of a circle. -/
def CircleArea (C : Set ℝ × ℝ) : ℝ := sorry

/-- Theorem: Given two circles A and B, where A is tangent to B and passes through B's center,
    if the area of A is 16π, then the area of B is 64π. -/
theorem circle_area_relation (A B : Set ℝ × ℝ) :
  CirclesTangent A B →
  CirclePassesThrough A (CircleCenter B) →
  CircleArea A = 16 * Real.pi →
  CircleArea B = 64 * Real.pi := by
  sorry

end circle_area_relation_l588_58898


namespace distance_between_points_l588_58896

/-- The distance between the points (2, -1) and (-3, 6) is √74. -/
theorem distance_between_points : Real.sqrt 74 = Real.sqrt ((2 - (-3))^2 + ((-1) - 6)^2) := by
  sorry

end distance_between_points_l588_58896


namespace semicircle_problem_l588_58802

/-- Given a large semicircle with diameter D and N congruent small semicircles
    fitting exactly on its diameter, if the ratio of the combined area of the
    small semicircles to the area of the large semicircle not covered by the
    small semicircles is 1:10, then N = 11. -/
theorem semicircle_problem (D : ℝ) (N : ℕ) (h : N > 0) :
  let r := D / (2 * N)
  let A := N * π * r^2 / 2
  let B := π * (N * r)^2 / 2 - A
  A / B = 1 / 10 → N = 11 := by
  sorry

end semicircle_problem_l588_58802


namespace eggs_remaining_l588_58864

theorem eggs_remaining (original : ℝ) (removed : ℝ) (remaining : ℝ) : 
  original = 35.3 → removed = 4.5 → remaining = original - removed → remaining = 30.8 := by
  sorry

end eggs_remaining_l588_58864


namespace competition_sequences_count_l588_58834

/-- The number of possible competition sequences for two teams with 7 members each -/
def competition_sequences : ℕ :=
  Nat.choose 14 7

/-- Theorem stating that the number of competition sequences is 3432 -/
theorem competition_sequences_count : competition_sequences = 3432 := by
  sorry

end competition_sequences_count_l588_58834


namespace find_z_when_y_is_6_l588_58846

-- Define the direct variation relationship
def varies_directly (y z : ℝ) : Prop := ∃ k : ℝ, y^3 = k * z^(1/3)

-- State the theorem
theorem find_z_when_y_is_6 (y z : ℝ) (h1 : varies_directly y z) (h2 : y = 3 ∧ z = 8) :
  y = 6 → z = 4096 := by
  sorry

end find_z_when_y_is_6_l588_58846


namespace smallest_upper_bound_l588_58867

theorem smallest_upper_bound (x : ℤ) 
  (h1 : 5 < x)
  (h2 : 7 < x ∧ x < 18)
  (h3 : 2 < x ∧ x < 13)
  (h4 : 9 < x ∧ x < 12)
  (h5 : x + 1 < 13) :
  ∃ (y : ℤ), x < y ∧ (∀ (z : ℤ), x < z → y ≤ z) ∧ y = 12 := by
  sorry

end smallest_upper_bound_l588_58867


namespace three_digit_sum_27_l588_58835

theorem three_digit_sum_27 : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n / 100 + (n / 10) % 10 + n % 10 = 27) :=
by sorry

end three_digit_sum_27_l588_58835


namespace two_valid_selections_l588_58881

def numbers : List ℕ := [1, 2, 3, 4, 5]

def average (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

def validSelection (a b : ℕ) : Prop :=
  a ∈ numbers ∧ b ∈ numbers ∧ a ≠ b ∧
  average (numbers.filter (λ x => x ≠ a ∧ x ≠ b)) = average numbers

theorem two_valid_selections :
  (∃! (pair : ℕ × ℕ), validSelection pair.1 pair.2) ∨
  (∃! (pair1 pair2 : ℕ × ℕ), 
    validSelection pair1.1 pair1.2 ∧ 
    validSelection pair2.1 pair2.2 ∧ 
    pair1 ≠ pair2) :=
  sorry

end two_valid_selections_l588_58881


namespace monomial_degree_5_l588_58877

/-- The degree of a monomial of the form 3a^2b^n -/
def monomialDegree (n : ℕ) : ℕ := 2 + n

theorem monomial_degree_5 (n : ℕ) : monomialDegree n = 5 → n = 3 := by
  sorry

end monomial_degree_5_l588_58877


namespace sum_of_fourth_powers_squared_l588_58869

theorem sum_of_fourth_powers_squared (x y z : ℤ) (h : x + y + z = 0) :
  ∃ (n : ℤ), 2 * (x^4 + y^4 + z^4) = n^2 := by
sorry

end sum_of_fourth_powers_squared_l588_58869


namespace ratio_shoes_to_total_earned_l588_58866

def rate_per_hour : ℕ := 14
def hours_per_day : ℕ := 2
def days_worked : ℕ := 7
def money_left : ℕ := 49

def total_hours : ℕ := hours_per_day * days_worked
def total_earned : ℕ := total_hours * rate_per_hour
def money_before_mom : ℕ := money_left * 2
def money_spent_shoes : ℕ := total_earned - money_before_mom

theorem ratio_shoes_to_total_earned :
  (money_spent_shoes : ℚ) / total_earned = 1 / 2 := by
  sorry

end ratio_shoes_to_total_earned_l588_58866


namespace cube_inequality_l588_58897

theorem cube_inequality (x y a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x < a^y) : x^3 > y^3 := by
  sorry

end cube_inequality_l588_58897


namespace bridge_length_l588_58824

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) :
  train_length = 148 →
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time_s
  let bridge_length := total_distance - train_length
  bridge_length = 227 := by
  sorry

end bridge_length_l588_58824


namespace last_locker_theorem_l588_58861

/-- The number of lockers in the hall -/
def num_lockers : ℕ := 2048

/-- The pattern of opening lockers -/
def open_pattern (n : ℕ) : Bool :=
  if n % 3 = 1 then true  -- opened in first pass
  else if n % 3 = 2 then true  -- opened in second pass
  else false  -- opened in third pass

/-- The last locker opened is the largest multiple of 3 not exceeding the number of lockers -/
def last_locker_opened (total : ℕ) : ℕ :=
  total - (total % 3)

theorem last_locker_theorem :
  last_locker_opened num_lockers = 2046 ∧
  ∀ n, n > last_locker_opened num_lockers → n ≤ num_lockers → open_pattern n = false :=
by sorry

end last_locker_theorem_l588_58861


namespace earthquake_relief_donation_scientific_notation_l588_58892

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem earthquake_relief_donation_scientific_notation :
  toScientificNotation 3990000000 = ScientificNotation.mk 3.99 9 (by sorry) :=
sorry

end earthquake_relief_donation_scientific_notation_l588_58892


namespace no_equality_under_condition_l588_58812

theorem no_equality_under_condition :
  ¬∃ (a b c : ℝ), (a^2 + b*c = (a + b)*(a + c)) ∧ (a + b + c = 2) :=
sorry

end no_equality_under_condition_l588_58812


namespace number_equation_l588_58883

theorem number_equation (x : ℝ) : (0.5 * x = (3/5) * x - 10) ↔ (x = 100) := by
  sorry

end number_equation_l588_58883


namespace combined_age_proof_l588_58818

/-- Given that Hezekiah is 4 years old and Ryanne is 7 years older than Hezekiah,
    prove that their combined age is 15 years. -/
theorem combined_age_proof (hezekiah_age : ℕ) (ryanne_age : ℕ) : 
  hezekiah_age = 4 → 
  ryanne_age = hezekiah_age + 7 → 
  hezekiah_age + ryanne_age = 15 := by
sorry

end combined_age_proof_l588_58818


namespace smallest_four_digit_2_mod_5_l588_58828

theorem smallest_four_digit_2_mod_5 : 
  ∀ n : ℕ, n ≥ 1000 ∧ n % 5 = 2 → n ≥ 1002 :=
by sorry

end smallest_four_digit_2_mod_5_l588_58828


namespace arithmetic_sequence_sum_l588_58845

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence {a_n}, if a_3 + a_4 + a_5 + a_6 + a_7 = 25, then a_2 + a_8 = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  (a 3 + a 4 + a 5 + a 6 + a 7 = 25) → (a 2 + a 8 = 10) := by
  sorry

end arithmetic_sequence_sum_l588_58845


namespace days_without_calls_is_244_l588_58829

/-- The number of days in the year --/
def year_days : ℕ := 365

/-- The intervals at which the nephews call --/
def call_intervals : List ℕ := [4, 6, 8]

/-- Calculate the number of days without calls --/
def days_without_calls (total_days : ℕ) (intervals : List ℕ) : ℕ :=
  total_days - (total_days / intervals.head! + total_days / intervals.tail.head! + total_days / intervals.tail.tail.head! -
    total_days / (intervals.head!.lcm intervals.tail.head!) - 
    total_days / (intervals.head!.lcm intervals.tail.tail.head!) - 
    total_days / (intervals.tail.head!.lcm intervals.tail.tail.head!) +
    total_days / (intervals.head!.lcm intervals.tail.head!).lcm intervals.tail.tail.head!)

theorem days_without_calls_is_244 :
  days_without_calls year_days call_intervals = 244 := by
  sorry

end days_without_calls_is_244_l588_58829


namespace cody_spent_25_tickets_on_beanie_l588_58893

/-- The number of tickets Cody spent on the beanie -/
def tickets_spent_on_beanie (initial_tickets : ℕ) (additional_tickets : ℕ) (remaining_tickets : ℕ) : ℕ :=
  initial_tickets + additional_tickets - remaining_tickets

/-- Proof that Cody spent 25 tickets on the beanie -/
theorem cody_spent_25_tickets_on_beanie :
  tickets_spent_on_beanie 49 6 30 = 25 := by
  sorry

end cody_spent_25_tickets_on_beanie_l588_58893


namespace abc_inequality_l588_58813

theorem abc_inequality (a b c : ℝ) (h : (a + b + c) * c < 0) : b^2 > 4*a*c := by
  sorry

end abc_inequality_l588_58813


namespace cos_sin_pi_eighth_difference_l588_58844

theorem cos_sin_pi_eighth_difference (π : Real) : 
  (Real.cos (π / 8))^4 - (Real.sin (π / 8))^4 = Real.sqrt 2 / 2 := by
sorry

end cos_sin_pi_eighth_difference_l588_58844


namespace book_problem_solution_l588_58868

def book_problem (cost_loss : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : Prop :=
  let selling_price := cost_loss * (1 - loss_percent)
  let cost_gain := selling_price / (1 + gain_percent)
  cost_loss + cost_gain = 360

theorem book_problem_solution :
  book_problem 210 0.15 0.19 :=
sorry

end book_problem_solution_l588_58868


namespace cubic_roots_existence_l588_58841

theorem cubic_roots_existence (a b c : ℝ) : 
  (a + b + c = 6 ∧ a * b + b * c + c * a = 9) →
  (¬ (a^4 + b^4 + c^4 = 260) ∧ ∃ (x y z : ℝ), x + y + z = 6 ∧ x * y + y * z + z * x = 9 ∧ x^4 + y^4 + z^4 = 210) :=
by sorry

end cubic_roots_existence_l588_58841


namespace multiples_of_three_imply_F_equals_six_l588_58814

def first_number (D E : ℕ) : ℕ := 8000000 + D * 100000 + 70000 + 3000 + E * 10 + 2

def second_number (D E F : ℕ) : ℕ := 4000000 + 100000 + 70000 + D * 1000 + E * 100 + 60 + F

theorem multiples_of_three_imply_F_equals_six (D E : ℕ) 
  (h1 : D < 10) (h2 : E < 10) 
  (h3 : ∃ k : ℕ, first_number D E = 3 * k) 
  (h4 : ∃ m : ℕ, second_number D E 6 = 3 * m) : 
  ∃ F : ℕ, F = 6 ∧ F < 10 ∧ ∃ n : ℕ, second_number D E F = 3 * n :=
sorry

end multiples_of_three_imply_F_equals_six_l588_58814


namespace harvester_problem_l588_58865

/-- Represents the number of harvesters of each type -/
structure HarvesterCount where
  typeA : ℕ
  typeB : ℕ

/-- Represents a plan for introducing additional harvesters -/
structure IntroductionPlan where
  additionalTypeA : ℕ
  additionalTypeB : ℕ

/-- The problem statement -/
theorem harvester_problem 
  (total_harvesters : ℕ)
  (typeA_capacity : ℕ)
  (typeB_capacity : ℕ)
  (total_daily_harvest : ℕ)
  (new_target : ℕ)
  (additional_harvesters : ℕ)
  (h1 : total_harvesters = 20)
  (h2 : typeA_capacity = 80)
  (h3 : typeB_capacity = 120)
  (h4 : total_daily_harvest = 2080)
  (h5 : new_target > 2900)
  (h6 : additional_harvesters = 8) :
  ∃ (initial : HarvesterCount) (plans : List IntroductionPlan),
    initial.typeA + initial.typeB = total_harvesters ∧
    initial.typeA * typeA_capacity + initial.typeB * typeB_capacity = total_daily_harvest ∧
    initial.typeA = 8 ∧
    initial.typeB = 12 ∧
    plans.length = 3 ∧
    ∀ plan ∈ plans, 
      plan.additionalTypeA + plan.additionalTypeB = additional_harvesters ∧
      (initial.typeA + plan.additionalTypeA) * typeA_capacity + 
      (initial.typeB + plan.additionalTypeB) * typeB_capacity > new_target :=
by sorry

end harvester_problem_l588_58865


namespace parabola_intersection_l588_58863

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Line structure -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Intersection points -/
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Main theorem -/
theorem parabola_intersection (C : Parabola) (l : Line) (I : IntersectionPoints) :
  l.slope = 2 ∧
  l.point = (C.p / 2, 0) ∧
  (I.A.1 - C.p / 2) * (I.A.1 - C.p / 2) + I.A.2 * I.A.2 = 20 ∧
  (I.B.1 - C.p / 2) * (I.B.1 - C.p / 2) + I.B.2 * I.B.2 = 20 ∧
  I.A.2 * I.A.2 = 2 * C.p * I.A.1 ∧
  I.B.2 * I.B.2 = 2 * C.p * I.B.1 →
  C.p = 4 := by
  sorry

end parabola_intersection_l588_58863


namespace girls_count_l588_58848

/-- The number of boys in the school -/
def num_boys : ℕ := 841

/-- The difference between the number of boys and girls -/
def boy_girl_diff : ℕ := 807

/-- The number of girls in the school -/
def num_girls : ℕ := num_boys - boy_girl_diff

theorem girls_count : num_girls = 34 := by
  sorry

end girls_count_l588_58848


namespace product_of_ratios_l588_58854

theorem product_of_ratios (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 4*x₁*y₁^2 = 3003)
  (h₂ : y₁^3 - 4*x₁^2*y₁ = 3002)
  (h₃ : x₂^3 - 4*x₂*y₂^2 = 3003)
  (h₄ : y₂^3 - 4*x₂^2*y₂ = 3002)
  (h₅ : x₃^3 - 4*x₃*y₃^2 = 3003)
  (h₆ : y₃^3 - 4*x₃^2*y₃ = 3002) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 3/3002 := by
sorry

end product_of_ratios_l588_58854


namespace watch_sale_loss_percentage_l588_58894

/-- Proves that the loss percentage is 10% given the conditions of the watch sale problem -/
theorem watch_sale_loss_percentage (cost_price : ℝ) (additional_amount : ℝ) (gain_percentage : ℝ) :
  cost_price = 3000 →
  additional_amount = 540 →
  gain_percentage = 8 →
  ∃ (loss_percentage : ℝ),
    loss_percentage = 10 ∧
    cost_price * (1 + gain_percentage / 100) = 
    cost_price * (1 - loss_percentage / 100) + additional_amount :=
by
  sorry

end watch_sale_loss_percentage_l588_58894
