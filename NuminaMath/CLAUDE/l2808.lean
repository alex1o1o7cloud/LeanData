import Mathlib

namespace student_calculation_error_l2808_280890

theorem student_calculation_error (N : ℝ) : (5/4)*N - (4/5)*N = 36 → N = 80 := by
  sorry

end student_calculation_error_l2808_280890


namespace total_waiting_after_changes_l2808_280837

/-- Represents the number of people waiting at each entrance of SFL -/
structure EntranceCount where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- Calculates the total number of people waiting at all entrances -/
def total_waiting (count : EntranceCount) : ℕ :=
  count.a + count.b + count.c + count.d + count.e

/-- Initial count of people waiting at each entrance -/
def initial_count : EntranceCount :=
  { a := 283, b := 356, c := 412, d := 179, e := 389 }

/-- Final count of people waiting at each entrance after changes -/
def final_count : EntranceCount :=
  { a := initial_count.a - 15,
    b := initial_count.b,
    c := initial_count.c + 10,
    d := initial_count.d,
    e := initial_count.e - 20 }

/-- Theorem stating that the total number of people waiting after changes is 1594 -/
theorem total_waiting_after_changes :
  total_waiting final_count = 1594 := by sorry

end total_waiting_after_changes_l2808_280837


namespace smaller_number_proof_l2808_280894

theorem smaller_number_proof (x y : ℤ) 
  (sum_condition : x + y = 84)
  (ratio_condition : y = 3 * x) :
  x = 21 := by
sorry

end smaller_number_proof_l2808_280894


namespace complex_absolute_value_product_l2808_280849

theorem complex_absolute_value_product : 
  Complex.abs ((7 - 4 * Complex.I) * (5 + 12 * Complex.I)) = 13 * Real.sqrt 65 := by
  sorry

end complex_absolute_value_product_l2808_280849


namespace frog_grasshopper_difference_l2808_280807

/-- Represents the jumping distances in the contest -/
structure JumpDistances where
  grasshopper : ℕ
  frog : ℕ
  mouse : ℕ

/-- The conditions of the jumping contest -/
def contest_conditions (j : JumpDistances) : Prop :=
  j.grasshopper = 19 ∧
  j.frog > j.grasshopper ∧
  j.mouse = j.frog + 20 ∧
  j.mouse = j.grasshopper + 30

/-- The theorem stating the difference between the frog's and grasshopper's jump distances -/
theorem frog_grasshopper_difference (j : JumpDistances) 
  (h : contest_conditions j) : j.frog - j.grasshopper = 10 := by
  sorry


end frog_grasshopper_difference_l2808_280807


namespace tenth_row_third_element_l2808_280878

/-- Represents the exponent of 2 for an element in the triangular array --/
def triangularArrayExponent (row : ℕ) (position : ℕ) : ℕ :=
  (row - 1) * row / 2 + position

/-- The theorem stating that the third element from the left in the 10th row is 2^47 --/
theorem tenth_row_third_element :
  triangularArrayExponent 10 2 = 47 := by
  sorry

end tenth_row_third_element_l2808_280878


namespace division_properties_7529_l2808_280869

theorem division_properties_7529 : 
  (7529 % 9 = 5) ∧ ¬(11 ∣ 7529) := by
  sorry

end division_properties_7529_l2808_280869


namespace lawn_mowing_payment_l2808_280831

theorem lawn_mowing_payment (rate : ℚ) (lawns_mowed : ℚ) : 
  rate = 15 / 4 → lawns_mowed = 5 / 2 → rate * lawns_mowed = 75 / 8 := by
  sorry

end lawn_mowing_payment_l2808_280831


namespace race_distance_proof_l2808_280825

/-- The distance John was behind Steve when he began his final push -/
def initial_distance : ℝ := 16

/-- John's speed in meters per second -/
def john_speed : ℝ := 4.2

/-- Steve's speed in meters per second -/
def steve_speed : ℝ := 3.7

/-- Duration of the final push in seconds -/
def final_push_duration : ℝ := 36

/-- The distance John finishes ahead of Steve -/
def final_distance_ahead : ℝ := 2

theorem race_distance_proof :
  john_speed * final_push_duration = 
  steve_speed * final_push_duration + initial_distance + final_distance_ahead :=
by sorry

end race_distance_proof_l2808_280825


namespace johns_hat_cost_l2808_280844

/-- The total cost of John's hats -/
def total_cost (weeks : ℕ) (odd_cost even_cost : ℕ) : ℕ :=
  let total_days := weeks * 7
  let odd_days := total_days / 2
  let even_days := total_days / 2
  odd_days * odd_cost + even_days * even_cost

/-- Theorem stating that the total cost of John's hats is $7350 -/
theorem johns_hat_cost :
  total_cost 20 45 60 = 7350 := by
  sorry

end johns_hat_cost_l2808_280844


namespace exists_primitive_root_mod_2p_alpha_l2808_280867

/-- Given an odd prime p and a natural number α, there exists a primitive root modulo 2p^α -/
theorem exists_primitive_root_mod_2p_alpha (p : Nat) (α : Nat) 
  (h_prime : Nat.Prime p) (h_odd : Odd p) : 
  ∃ x : Nat, IsPrimitiveRoot x (2 * p^α) := by
  sorry

end exists_primitive_root_mod_2p_alpha_l2808_280867


namespace car_speed_difference_l2808_280868

/-- Prove that given two cars P and R traveling 300 miles, where car R's speed is 34.05124837953327 mph
    and car P takes 2 hours less than car R, the difference in their average speeds is 10 mph. -/
theorem car_speed_difference (distance : ℝ) (speed_R : ℝ) (time_difference : ℝ) :
  distance = 300 →
  speed_R = 34.05124837953327 →
  time_difference = 2 →
  let time_R := distance / speed_R
  let time_P := time_R - time_difference
  let speed_P := distance / time_P
  speed_P - speed_R = 10 := by sorry

end car_speed_difference_l2808_280868


namespace necessary_but_not_sufficient_l2808_280871

theorem necessary_but_not_sufficient (p q : Prop) :
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) :=
sorry

end necessary_but_not_sufficient_l2808_280871


namespace problem_solution_l2808_280893

-- Define x as the solution to the equation x = 1 + √3 / x
noncomputable def x : ℝ := Real.sqrt 3 + 1

-- State the theorem
theorem problem_solution : 
  1 / ((x + 1) * (x - 2)) = -(Real.sqrt 3 + 2) := by
  sorry

end problem_solution_l2808_280893


namespace tiffany_towels_l2808_280895

theorem tiffany_towels (packs : ℕ) (towels_per_pack : ℕ) (h1 : packs = 9) (h2 : towels_per_pack = 3) :
  packs * towels_per_pack = 27 := by
  sorry

end tiffany_towels_l2808_280895


namespace number_manipulation_l2808_280842

theorem number_manipulation (x : ℚ) (h : (7/8) * x = 28) : (x + 16) * (5/16) = 15 := by
  sorry

end number_manipulation_l2808_280842


namespace hyperbola_eccentricity_l2808_280848

/-- The hyperbola C: mx² + ny² = 1 -/
structure Hyperbola where
  m : ℝ
  n : ℝ
  h_mn : m * n < 0

/-- The circle x² + y² - 6x - 2y + 9 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 - 2*p.2 + 9 = 0}

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The asymptotes of a hyperbola -/
def asymptotes (h : Hyperbola) : Set (Set (ℝ × ℝ)) := sorry

/-- Predicate to check if a line is tangent to a circle -/
def is_tangent_to (line : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)) : Prop := sorry

theorem hyperbola_eccentricity (h : Hyperbola) :
  (∃ a ∈ asymptotes h, is_tangent_to a Circle) →
  (eccentricity h = 5/3 ∨ eccentricity h = 5/4) := by
  sorry

end hyperbola_eccentricity_l2808_280848


namespace car_distance_theorem_l2808_280860

/-- Theorem: A car traveling at 208 km/h for 3 hours covers a distance of 624 km. -/
theorem car_distance_theorem (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 208 → time = 3 → distance = speed * time → distance = 624 := by
  sorry

end car_distance_theorem_l2808_280860


namespace cubic_roots_sum_l2808_280816

theorem cubic_roots_sum (p q r : ℝ) : 
  (3 * p^3 - 4 * p^2 + 200 * p - 5 = 0) →
  (3 * q^3 - 4 * q^2 + 200 * q - 5 = 0) →
  (3 * r^3 - 4 * r^2 + 200 * r - 5 = 0) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 184/9 := by
sorry

end cubic_roots_sum_l2808_280816


namespace twenty_one_in_fibonacci_l2808_280813

def fibonacci : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

theorem twenty_one_in_fibonacci : ∃ n : ℕ, fibonacci n = 21 := by
  sorry

end twenty_one_in_fibonacci_l2808_280813


namespace painting_class_combinations_l2808_280802

theorem painting_class_combinations : Nat.choose 10 4 = 210 := by
  sorry

end painting_class_combinations_l2808_280802


namespace group_b_sample_size_l2808_280809

/-- Represents the number of cities in each group and the total sample size -/
structure CityGroups where
  total : Nat
  groupA : Nat
  groupB : Nat
  groupC : Nat
  sampleSize : Nat

/-- Calculates the number of cities to be selected from a specific group in stratified sampling -/
def stratifiedSampleSize (cg : CityGroups) (groupSize : Nat) : Nat :=
  (groupSize * cg.sampleSize) / cg.total

/-- Theorem stating that for the given city groups, the stratified sample size for Group B is 3 -/
theorem group_b_sample_size (cg : CityGroups) 
  (h1 : cg.total = 24)
  (h2 : cg.groupA = 4)
  (h3 : cg.groupB = 12)
  (h4 : cg.groupC = 8)
  (h5 : cg.sampleSize = 6)
  : stratifiedSampleSize cg cg.groupB = 3 := by
  sorry

end group_b_sample_size_l2808_280809


namespace not_divisible_five_power_minus_one_by_four_power_minus_one_l2808_280881

theorem not_divisible_five_power_minus_one_by_four_power_minus_one (n : ℕ) :
  ¬(∃ k : ℕ, 5^n - 1 = k * (4^n - 1)) := by
  sorry

end not_divisible_five_power_minus_one_by_four_power_minus_one_l2808_280881


namespace max_cross_sum_l2808_280826

def CrossNumbers : Finset ℕ := {2, 5, 8, 11, 14}

theorem max_cross_sum :
  ∃ (a b c d e : ℕ),
    a ∈ CrossNumbers ∧ b ∈ CrossNumbers ∧ c ∈ CrossNumbers ∧ d ∈ CrossNumbers ∧ e ∈ CrossNumbers ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    a + b + e = b + d + e ∧
    a + c + e = a + b + e ∧
    a + b + e = 36 ∧
    ∀ (x y z : ℕ),
      x ∈ CrossNumbers → y ∈ CrossNumbers → z ∈ CrossNumbers →
      x + y + z ≤ 36 :=
by sorry

end max_cross_sum_l2808_280826


namespace number_problem_l2808_280884

theorem number_problem : ∃! x : ℝ, x + (2/3) * x + 1 = 10 ∧ x = 27/5 := by
  sorry

end number_problem_l2808_280884


namespace egg_roll_count_l2808_280823

/-- The number of egg rolls Omar rolled -/
def omar_rolls : ℕ := 219

/-- The number of egg rolls Karen rolled -/
def karen_rolls : ℕ := 229

/-- The total number of egg rolls rolled by Omar and Karen -/
def total_rolls : ℕ := omar_rolls + karen_rolls

theorem egg_roll_count : total_rolls = 448 := by sorry

end egg_roll_count_l2808_280823


namespace power_of_seven_fraction_l2808_280886

theorem power_of_seven_fraction (a b : ℕ) : 
  (2^a : ℕ) = Nat.gcd (2^a) 196 → 
  (7^b : ℕ) = Nat.gcd (7^b) 196 → 
  (1/7 : ℚ)^(b - a) = 1 := by sorry

end power_of_seven_fraction_l2808_280886


namespace solution_count_l2808_280835

/-- The number of distinct divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The number of positive integer solutions (x, y) to the equation 1/n = 1/x + 1/y where x ≠ y -/
def num_solutions (n : ℕ+) : ℕ := sorry

theorem solution_count (n : ℕ+) : num_solutions n = num_divisors (n^2) - 1 := by sorry

end solution_count_l2808_280835


namespace sprint_competition_correct_l2808_280865

def sprint_competition (total_sprinters : ℕ) (byes : ℕ) (first_round_lanes : ℕ) (subsequent_lanes : ℕ) : ℕ :=
  let first_round := (total_sprinters - byes + first_round_lanes - 1) / first_round_lanes
  let second_round := (first_round + byes + subsequent_lanes - 1) / subsequent_lanes
  let third_round := (second_round + subsequent_lanes - 1) / subsequent_lanes
  let final_round := 1
  first_round + second_round + third_round + final_round

theorem sprint_competition_correct :
  sprint_competition 300 16 8 6 = 48 := by
  sorry

end sprint_competition_correct_l2808_280865


namespace evaluate_64_to_two_thirds_l2808_280864

theorem evaluate_64_to_two_thirds : (64 : ℝ) ^ (2/3) = 16 := by sorry

end evaluate_64_to_two_thirds_l2808_280864


namespace distribute_six_tasks_three_people_l2808_280805

/-- The number of ways to distribute tasks among people -/
def distribute_tasks (num_tasks : ℕ) (num_people : ℕ) : ℕ :=
  num_people^num_tasks - num_people * (num_people - 1)^num_tasks + num_people

/-- Theorem stating the correct number of ways to distribute 6 tasks among 3 people -/
theorem distribute_six_tasks_three_people :
  distribute_tasks 6 3 = 540 := by
  sorry


end distribute_six_tasks_three_people_l2808_280805


namespace distribution_of_distinct_objects_l2808_280898

theorem distribution_of_distinct_objects (n : ℕ) (m : ℕ) :
  n = 6 → m = 12 → n^m = 2985984 := by
  sorry

end distribution_of_distinct_objects_l2808_280898


namespace copper_weights_problem_l2808_280812

theorem copper_weights_problem :
  ∃ (x y z u : ℕ+),
    (x : ℤ) + y + z + u = 40 ∧
    ∀ W : ℤ, 1 ≤ W ∧ W ≤ 40 →
      ∃ (a b c d : ℤ),
        (a = -1 ∨ a = 0 ∨ a = 1) ∧
        (b = -1 ∨ b = 0 ∨ b = 1) ∧
        (c = -1 ∨ c = 0 ∨ c = 1) ∧
        (d = -1 ∨ d = 0 ∨ d = 1) ∧
        W = a * x + b * y + c * z + d * u :=
by sorry

end copper_weights_problem_l2808_280812


namespace no_integer_solutions_l2808_280845

/-- The system of equations has no integer solutions -/
theorem no_integer_solutions : ¬ ∃ (x y z : ℤ), 
  (x^2 - 2*x*y + y^2 - z^2 = 17) ∧ 
  (-x^2 + 3*y*z + 3*z^2 = 27) ∧ 
  (x^2 - x*y + 5*z^2 = 50) := by
  sorry

end no_integer_solutions_l2808_280845


namespace sphere_surface_area_circumscribing_cylinder_l2808_280879

/-- The surface area of a sphere circumscribing a right circular cylinder with edge length 6 -/
theorem sphere_surface_area_circumscribing_cylinder (r : ℝ) : r^2 = 21 → 4 * Real.pi * r^2 = 84 * Real.pi := by
  sorry

end sphere_surface_area_circumscribing_cylinder_l2808_280879


namespace dennis_teaching_years_l2808_280861

def teaching_problem (v a d e n : ℕ) : Prop :=
  v + a + d + e + n = 225 ∧
  v = a + 9 ∧
  v = d - 15 ∧
  e = a - 3 ∧
  e = n + 7

theorem dennis_teaching_years :
  ∀ v a d e n : ℕ, teaching_problem v a d e n → d = 65 :=
by
  sorry

end dennis_teaching_years_l2808_280861


namespace tan_alpha_and_expression_l2808_280858

theorem tan_alpha_and_expression (α : Real) 
  (h : Real.tan (π / 4 + α) = 1 / 2) : 
  Real.tan α = -1 / 3 ∧ 
  (Real.sin (2 * α + 2 * π) - Real.sin (π / 2 - α) ^ 2) / 
  (1 - Real.cos (π - 2 * α) + Real.sin α ^ 2) = -15 / 19 := by
  sorry

end tan_alpha_and_expression_l2808_280858


namespace interest_rate_calculation_l2808_280830

/-- Proves that given a loan of 1000 for 5 years, where the interest amount is 750 less than the sum lent, the interest rate per annum is 5% -/
theorem interest_rate_calculation (sum_lent : ℝ) (time_period : ℝ) (interest_amount : ℝ) 
  (h1 : sum_lent = 1000)
  (h2 : time_period = 5)
  (h3 : interest_amount = sum_lent - 750) :
  (interest_amount * 100) / (sum_lent * time_period) = 5 := by
  sorry

end interest_rate_calculation_l2808_280830


namespace custom_op_result_l2808_280843

-- Define the custom operation
def custom_op (a b : ℚ) : ℚ := (a + b) / (a - b - 1)

-- State the theorem
theorem custom_op_result : custom_op (custom_op 7 5) 2 = 14 / 9 := by
  sorry

end custom_op_result_l2808_280843


namespace min_cost_verification_l2808_280811

/-- Represents a set of weights -/
def WeightSet := List Nat

/-- Cost of using a weight once -/
def weighing_cost : Nat := 100

/-- The range of possible diamond masses -/
def diamond_range : Set Nat := Finset.range 15

/-- Checks if a set of weights can measure all masses in the given range -/
def can_measure_all (weights : WeightSet) (range : Set Nat) : Prop :=
  ∀ n ∈ range, ∃ subset : List Nat, subset.toFinset ⊆ weights.toFinset ∧ subset.sum = n

/-- Calculates the minimum number of weighings needed for a given set of weights -/
def min_weighings (weights : WeightSet) : Nat :=
  weights.length + 1

/-- Calculates the total cost for a given number of weighings -/
def total_cost (num_weighings : Nat) : Nat :=
  num_weighings * weighing_cost

/-- The optimal set of weights for measuring masses from 1 to 15 -/
def optimal_weights : WeightSet := [1, 2, 4, 8]

theorem min_cost_verification :
  can_measure_all optimal_weights diamond_range →
  total_cost (min_weighings optimal_weights) = 800 := by
  sorry

end min_cost_verification_l2808_280811


namespace intersection_of_A_and_B_l2808_280852

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end intersection_of_A_and_B_l2808_280852


namespace domain_of_f_with_restricted_range_l2808_280815

def f (x : ℝ) : ℝ := x^2

def domain : Set ℝ := {-2, -1, 1, 2}
def range : Set ℝ := {1, 4}

theorem domain_of_f_with_restricted_range :
  ∀ y ∈ range, ∃ x ∈ domain, f x = y ∧
  ∀ x : ℝ, f x ∈ range → x ∈ domain :=
by
  sorry

end domain_of_f_with_restricted_range_l2808_280815


namespace quadratic_inequality_solution_l2808_280853

theorem quadratic_inequality_solution (c : ℝ) : 
  (∃ x ∈ Set.Ioo (-2 : ℝ) 1, x^2 + x - c < 0) → c = 2 := by
  sorry

end quadratic_inequality_solution_l2808_280853


namespace g_range_l2808_280855

noncomputable def g (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 11 * Real.sin x ^ 2 + 3 * Real.sin x + 4 * Real.cos x ^ 2 - 10) / (Real.sin x - 2)

theorem g_range : 
  ∀ y ∈ Set.range g, 1 ≤ y ∧ y ≤ 19 ∧ 
  ∃ x : ℝ, g x = 1 ∧ 
  ∃ x : ℝ, g x = 19 :=
sorry

end g_range_l2808_280855


namespace range_of_a_l2808_280851

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, a^x > 1 ↔ x < 0

def q (a : ℝ) : Prop := ∀ x : ℝ, ∃ y : ℝ, y = Real.log (a*x^2 - x + 2)

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, 
    (a > 0) → 
    (a ≠ 1) → 
    ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → 
    ((0 < a ∧ a ≤ 1/8) ∨ (a ≥ 1)) :=
sorry

end range_of_a_l2808_280851


namespace szilveszter_age_l2808_280841

def birth_year (a b : ℕ) := 1900 + 10 * a + b

def grandfather_birth_year (a b : ℕ) := 1910 + a + b

def current_year := 1999

theorem szilveszter_age (a b : ℕ) 
  (h1 : a < 10 ∧ b < 10) 
  (h2 : 1 + 9 + a + b = current_year - grandfather_birth_year a b) 
  (h3 : 10 * a + b = current_year - grandfather_birth_year a b) :
  current_year - birth_year a b = 23 := by
sorry

end szilveszter_age_l2808_280841


namespace last_day_vases_proof_l2808_280863

/-- The number of vases Jane can arrange in a day -/
def vases_per_day : ℕ := 16

/-- The total number of vases to be arranged -/
def total_vases : ℕ := 248

/-- The number of vases Jane will arrange on the last day -/
def last_day_vases : ℕ := total_vases - (vases_per_day * (total_vases / vases_per_day))

theorem last_day_vases_proof :
  last_day_vases = 8 :=
by sorry

end last_day_vases_proof_l2808_280863


namespace largest_digit_change_l2808_280854

def incorrect_sum : ℕ := 2456
def num1 : ℕ := 641
def num2 : ℕ := 852
def num3 : ℕ := 973

theorem largest_digit_change :
  ∃ (d : ℕ), d ≤ 9 ∧
  (num1 + num2 + (num3 - 10) = incorrect_sum) ∧
  (∀ (d' : ℕ), d' ≤ 9 → 
    (num1 - 10 * d' + num2 + num3 = incorrect_sum ∨
     num1 + (num2 - 10 * d') + num3 = incorrect_sum) → 
    d' ≤ d) ∧
  d = 7 :=
sorry

end largest_digit_change_l2808_280854


namespace expression_bounds_l2808_280899

theorem expression_bounds (a b c d e : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) (he : 0 ≤ e ∧ e ≤ 1) :
  4 * Real.sqrt (2/3) ≤ 
    Real.sqrt (a^2 + (1-b)^2 + e^2) + Real.sqrt (b^2 + (1-c)^2 + e^2) + 
    Real.sqrt (c^2 + (1-d)^2 + e^2) + Real.sqrt (d^2 + (1-a)^2 + e^2) ∧
  Real.sqrt (a^2 + (1-b)^2 + e^2) + Real.sqrt (b^2 + (1-c)^2 + e^2) + 
  Real.sqrt (c^2 + (1-d)^2 + e^2) + Real.sqrt (d^2 + (1-a)^2 + e^2) ≤ 8 ∧
  ∃ (a b c d e : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 
    0 ≤ d ∧ d ≤ 1 ∧ 0 ≤ e ∧ e ≤ 1 ∧
    Real.sqrt (a^2 + (1-b)^2 + e^2) + Real.sqrt (b^2 + (1-c)^2 + e^2) + 
    Real.sqrt (c^2 + (1-d)^2 + e^2) + Real.sqrt (d^2 + (1-a)^2 + e^2) = 4 * Real.sqrt (2/3) ∧
  ∃ (a b c d e : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 
    0 ≤ d ∧ d ≤ 1 ∧ 0 ≤ e ∧ e ≤ 1 ∧
    Real.sqrt (a^2 + (1-b)^2 + e^2) + Real.sqrt (b^2 + (1-c)^2 + e^2) + 
    Real.sqrt (c^2 + (1-d)^2 + e^2) + Real.sqrt (d^2 + (1-a)^2 + e^2) = 8 :=
by sorry

end expression_bounds_l2808_280899


namespace floor_equation_iff_solution_set_l2808_280840

def floor_equation (x : ℝ) : Prop :=
  ⌊x^2 - 2*x⌋ + 2*⌊x⌋ = ⌊x⌋^2

def solution_set (x : ℝ) : Prop :=
  (∃ (n : ℤ), n < 0 ∧ x = n) ∨
  x = 0 ∨
  (∃ (n : ℕ), n ≥ 1 ∧ n ≤ x ∧ x < Real.sqrt (n^2 - 2*n + 2) + 1)

theorem floor_equation_iff_solution_set :
  ∀ x : ℝ, floor_equation x ↔ solution_set x :=
sorry

end floor_equation_iff_solution_set_l2808_280840


namespace undefined_fraction_l2808_280827

theorem undefined_fraction (a : ℝ) : 
  ¬ (∃ x : ℝ, x = (a + 3) / (a^2 - 9)) ↔ a = -3 ∨ a = 3 := by
  sorry

end undefined_fraction_l2808_280827


namespace min_value_of_fraction_sum_l2808_280829

/-- Given a quadratic function f(x) = ax^2 - 4x + c with range [0,+∞),
    prove that the minimum value of 1/c + 9/a is 3 -/
theorem min_value_of_fraction_sum (a c : ℝ) (h₁ : a > 0) (h₂ : c > 0)
    (h₃ : ∀ x, ax^2 - 4*x + c ≥ 0) : 
    ∃ (m : ℝ), m = 3 ∧ ∀ a c, a > 0 → c > 0 → (∀ x, ax^2 - 4*x + c ≥ 0) → 1/c + 9/a ≥ m := by
  sorry

end min_value_of_fraction_sum_l2808_280829


namespace franks_age_l2808_280824

/-- Represents the ages of Dave, Ella, and Frank -/
structure Ages where
  dave : ℕ
  ella : ℕ
  frank : ℕ

/-- The conditions from the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- The average of their ages is 10
  (ages.dave + ages.ella + ages.frank) / 3 = 10 ∧
  -- Five years ago, Frank was the same age as Dave is now
  ages.frank - 5 = ages.dave ∧
  -- In 2 years, Ella's age will be 3/4 of Dave's age at that time
  ages.ella + 2 = (3 * (ages.dave + 2)) / 4

/-- The theorem to prove -/
theorem franks_age (ages : Ages) (h : satisfies_conditions ages) : ages.frank = 14 := by
  sorry


end franks_age_l2808_280824


namespace new_persons_weight_l2808_280896

theorem new_persons_weight (W : ℝ) (X Y : ℝ) :
  (∀ (T : ℝ), T = 8 * W) →
  (∀ (new_total : ℝ), new_total = 8 * W - 140 + X + Y) →
  (∀ (new_avg : ℝ), new_avg = W + 5) →
  (∀ (new_total : ℝ), new_total = 8 * new_avg) →
  X + Y = 180 := by
sorry

end new_persons_weight_l2808_280896


namespace bianca_drawing_time_l2808_280839

/-- The number of minutes Bianca spent drawing at school -/
def minutes_at_school : ℕ := sorry

/-- The number of minutes Bianca spent drawing at home -/
def minutes_at_home : ℕ := 19

/-- The total number of minutes Bianca spent drawing -/
def total_minutes : ℕ := 41

/-- Theorem stating that Bianca spent 22 minutes drawing at school -/
theorem bianca_drawing_time : minutes_at_school = 22 := by
  sorry

end bianca_drawing_time_l2808_280839


namespace password_decryption_probability_l2808_280862

theorem password_decryption_probability :
  let p₁ : ℚ := 1/5
  let p₂ : ℚ := 2/5
  let p₃ : ℚ := 1/2
  let prob_at_least_one_success : ℚ := 1 - (1 - p₁) * (1 - p₂) * (1 - p₃)
  prob_at_least_one_success = 19/25 :=
by sorry

end password_decryption_probability_l2808_280862


namespace exist_non_congruent_equal_perimeter_area_cyclic_quadrilaterals_l2808_280810

/-- A cyclic quadrilateral is a quadrilateral that can be inscribed in a circle. -/
structure CyclicQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_cyclic : ∃ (center : ℝ × ℝ) (radius : ℝ), ∀ i, dist center (vertices i) = radius

/-- The perimeter of a quadrilateral. -/
def perimeter (q : CyclicQuadrilateral) : ℝ :=
  (dist (q.vertices 0) (q.vertices 1)) +
  (dist (q.vertices 1) (q.vertices 2)) +
  (dist (q.vertices 2) (q.vertices 3)) +
  (dist (q.vertices 3) (q.vertices 0))

/-- The area of a quadrilateral. -/
def area (q : CyclicQuadrilateral) : ℝ := sorry

/-- Two quadrilaterals are congruent if there exists a rigid transformation that maps one to the other. -/
def congruent (q1 q2 : CyclicQuadrilateral) : Prop := sorry

theorem exist_non_congruent_equal_perimeter_area_cyclic_quadrilaterals :
  ∃ (q1 q2 : CyclicQuadrilateral),
    perimeter q1 = perimeter q2 ∧
    area q1 = area q2 ∧
    ¬congruent q1 q2 := by
  sorry

end exist_non_congruent_equal_perimeter_area_cyclic_quadrilaterals_l2808_280810


namespace monogram_count_is_300_l2808_280838

/-- The number of letters in the alphabet before 'A' --/
def n : ℕ := 25

/-- The number of initials to choose (first and middle) --/
def k : ℕ := 2

/-- The number of ways to choose k distinct letters from n letters in alphabetical order --/
def monogram_count : ℕ := Nat.choose n k

/-- Theorem stating that the number of possible monograms is 300 --/
theorem monogram_count_is_300 : monogram_count = 300 := by
  sorry

end monogram_count_is_300_l2808_280838


namespace negative_five_times_three_l2808_280870

theorem negative_five_times_three : -5 * 3 = -15 := by
  sorry

end negative_five_times_three_l2808_280870


namespace ellipse_hyperbola_equations_l2808_280803

/-- The equations of an ellipse and a hyperbola with shared foci -/
theorem ellipse_hyperbola_equations :
  ∀ (a b m n : ℝ),
  a > b ∧ b > 0 ∧ m > 0 ∧ n > 0 →
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/m^2 - y^2/n^2 = 1) →
  (a^2 - b^2 = 4 ∧ m^2 + n^2 = 4) →
  (∃ k : ℝ, k > 0 ∧ ∀ x y : ℝ, y = k*x → x^2/m^2 - y^2/n^2 = 1) →
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    (x₁ - 2)^2 + y₁^2 = (x₁ + 2)^2 + y₁^2 ∧
    x₁^2/m^2 - y₁^2/n^2 = 1 ∧
    x₂^2/a^2 + y₂^2/b^2 = 1 ∧
    x₂^2/m^2 - y₂^2/n^2 = 1) →
  (∀ x y : ℝ, 11*x^2/60 + 11*y^2/16 = 1 ↔ x^2/a^2 + y^2/b^2 = 1) ∧
  (∀ x y : ℝ, 5*x^2/4 - 5*y^2/16 = 1 ↔ x^2/m^2 - y^2/n^2 = 1) :=
by
  sorry

end ellipse_hyperbola_equations_l2808_280803


namespace infinitely_many_amiable_squares_l2808_280832

/-- A number is amiable if the set {1,2,...,N} can be partitioned into pairs
    of elements, each pair having the sum of its elements a perfect square. -/
def IsAmiable (N : ℕ) : Prop :=
  ∃ (partition : List (ℕ × ℕ)),
    (∀ (pair : ℕ × ℕ), pair ∈ partition → pair.1 ≤ N ∧ pair.2 ≤ N) ∧
    (∀ n : ℕ, n ≤ N → ∃ (pair : ℕ × ℕ), pair ∈ partition ∧ (n = pair.1 ∨ n = pair.2)) ∧
    (∀ (pair : ℕ × ℕ), pair ∈ partition → ∃ m : ℕ, pair.1 + pair.2 = m^2)

/-- There exist infinitely many amiable numbers which are themselves perfect squares. -/
theorem infinitely_many_amiable_squares :
  ∀ k : ℕ, ∃ N : ℕ, N > k ∧ ∃ m : ℕ, N = m^2 ∧ IsAmiable N :=
sorry

end infinitely_many_amiable_squares_l2808_280832


namespace farmer_budget_distribution_l2808_280874

theorem farmer_budget_distribution (g sh : ℕ) : 
  g > 0 ∧ sh > 0 ∧ 24 * g + 27 * sh = 1200 → g = 5 ∧ sh = 40 :=
by
  sorry

end farmer_budget_distribution_l2808_280874


namespace bottle_weight_difference_l2808_280866

/-- The weight difference between a glass bottle and a plastic bottle -/
def weight_difference : ℝ := by sorry

theorem bottle_weight_difference :
  let glass_bottle_weight : ℝ := 600 / 3
  let plastic_bottle_weight : ℝ := (1050 - 4 * glass_bottle_weight) / 5
  weight_difference = glass_bottle_weight - plastic_bottle_weight :=
by sorry

end bottle_weight_difference_l2808_280866


namespace max_value_linear_program_l2808_280873

theorem max_value_linear_program (x y : ℝ) 
  (h1 : x - y ≥ 0) 
  (h2 : x + 2*y ≤ 4) 
  (h3 : x - 2*y ≤ 2) : 
  ∃ (z : ℝ), z = x + 3*y ∧ z ≤ 16/3 ∧ 
  (∀ (x' y' : ℝ), x' - y' ≥ 0 → x' + 2*y' ≤ 4 → x' - 2*y' ≤ 2 → x' + 3*y' ≤ z) :=
by sorry

end max_value_linear_program_l2808_280873


namespace joan_egg_count_l2808_280806

-- Define the number of dozens Joan bought
def dozen_count : ℕ := 6

-- Define the number of eggs in a dozen
def eggs_per_dozen : ℕ := 12

-- Theorem to prove
theorem joan_egg_count : dozen_count * eggs_per_dozen = 72 := by
  sorry

end joan_egg_count_l2808_280806


namespace geometric_series_sum_l2808_280888

/-- The sum of an infinite geometric series with first term 1 and common ratio 2/3 is 3 -/
theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 2/3
  let S := ∑' n, a * r^n
  S = 3 := by sorry

end geometric_series_sum_l2808_280888


namespace consumption_increase_l2808_280857

theorem consumption_increase (original_tax original_consumption : ℝ) 
  (h1 : original_tax > 0) (h2 : original_consumption > 0) : 
  let new_tax := 0.76 * original_tax
  let revenue_decrease := 0.1488
  let new_revenue := (1 - revenue_decrease) * (original_tax * original_consumption)
  ∃ (consumption_increase : ℝ), 
    new_tax * (original_consumption * (1 + consumption_increase)) = new_revenue ∧ 
    consumption_increase = 0.12 := by
  sorry

end consumption_increase_l2808_280857


namespace parking_probability_l2808_280833

/-- Represents a parking lot configuration -/
structure ParkingLot :=
  (total_spaces : ℕ)
  (occupied_spaces : ℕ)

/-- Calculates the probability of finding two adjacent empty spaces in a parking lot -/
def probability_of_two_adjacent_empty_spaces (p : ParkingLot) : ℚ :=
  1 - (Nat.choose (p.total_spaces - p.occupied_spaces + 1) 5 : ℚ) / (Nat.choose p.total_spaces (p.total_spaces - p.occupied_spaces) : ℚ)

/-- Theorem stating the probability of finding two adjacent empty spaces in the given scenario -/
theorem parking_probability (p : ParkingLot) 
  (h1 : p.total_spaces = 20) 
  (h2 : p.occupied_spaces = 15) : 
  probability_of_two_adjacent_empty_spaces p = 232 / 323 := by
  sorry

end parking_probability_l2808_280833


namespace power_sum_equality_l2808_280808

theorem power_sum_equality (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  (a^5 + b^5 = c^5 + d^5) ∧ 
  ∃ (a b c d : ℝ), (a + b = c + d) ∧ (a^3 + b^3 = c^3 + d^3) ∧ (a^4 + b^4 ≠ c^4 + d^4) :=
by sorry

end power_sum_equality_l2808_280808


namespace german_students_count_l2808_280859

theorem german_students_count (total : ℕ) (french : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 78 → french = 41 → both = 9 → neither = 24 → 
  ∃ german : ℕ, german = 22 ∧ total = french + german - both + neither :=
by sorry

end german_students_count_l2808_280859


namespace trig_problem_l2808_280876

theorem trig_problem (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : Real.tan α - 1 / Real.tan α = -3/2) : 
  Real.tan α = -2 ∧ 
  (Real.cos (3*π/2 + α) - Real.cos (π - α)) / Real.sin (π/2 - α) = -1 := by
  sorry

end trig_problem_l2808_280876


namespace M_intersect_N_l2808_280818

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem M_intersect_N : M ∩ N = {0, 1} := by
  sorry

end M_intersect_N_l2808_280818


namespace ones_digit_of_13_power_power_cycle_of_3_main_theorem_l2808_280822

theorem ones_digit_of_13_power (n : ℕ) : n > 0 → (13^n) % 10 = (3^n) % 10 := by sorry

theorem power_cycle_of_3 (n : ℕ) : (3^n) % 10 = (3^(n % 4)) % 10 := by sorry

theorem main_theorem : (13^(13 * (12^12))) % 10 = 9 := by sorry

end ones_digit_of_13_power_power_cycle_of_3_main_theorem_l2808_280822


namespace dan_added_sixteen_pencils_l2808_280834

/-- The number of pencils Dan placed on the desk -/
def pencils_added (drawer : ℕ) (desk : ℕ) (total : ℕ) : ℕ :=
  total - (drawer + desk)

/-- Proof that Dan placed 16 pencils on the desk -/
theorem dan_added_sixteen_pencils :
  pencils_added 43 19 78 = 16 := by
  sorry

end dan_added_sixteen_pencils_l2808_280834


namespace hidden_primes_average_l2808_280883

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem hidden_primes_average (visible1 visible2 visible3 hidden1 hidden2 hidden3 : ℕ) :
  visible1 = 42 →
  visible2 = 59 →
  visible3 = 36 →
  is_prime hidden1 →
  is_prime hidden2 →
  is_prime hidden3 →
  visible1 + hidden1 = visible2 + hidden2 →
  visible2 + hidden2 = visible3 + hidden3 →
  visible1 ≠ visible2 ∧ visible2 ≠ visible3 ∧ visible1 ≠ visible3 →
  hidden1 ≠ hidden2 ∧ hidden2 ≠ hidden3 ∧ hidden1 ≠ hidden3 →
  (hidden1 + hidden2 + hidden3) / 3 = 56 / 3 := by
sorry

end hidden_primes_average_l2808_280883


namespace prob_monochromatic_triangle_l2808_280897

/-- A complete graph K6 where each edge is colored red or blue -/
def ColoredK6 := Fin 15 → Bool

/-- The probability of an edge being red (or blue) -/
def p : ℚ := 1/2

/-- The set of all possible colorings of K6 -/
def allColorings : Set ColoredK6 := Set.univ

/-- A triangle in K6 -/
structure Triangle :=
  (a b c : Fin 6)
  (ha : a < b)
  (hb : b < c)

/-- The set of all triangles in K6 -/
def allTriangles : Set Triangle := sorry

/-- A coloring has a monochromatic triangle -/
def hasMonochromaticTriangle (coloring : ColoredK6) : Prop := sorry

/-- The probability of having at least one monochromatic triangle -/
noncomputable def probMonochromaticTriangle : ℚ := sorry

theorem prob_monochromatic_triangle :
  probMonochromaticTriangle = 1048575/1048576 := by sorry

end prob_monochromatic_triangle_l2808_280897


namespace union_of_A_and_B_l2808_280804

def A : Set ℝ := {x | -5 ≤ x ∧ x < 1}
def B : Set ℝ := {x | x ≤ 2}

theorem union_of_A_and_B : A ∪ B = {x | x ≤ 2} := by sorry

end union_of_A_and_B_l2808_280804


namespace hyperbola_asymptote_l2808_280877

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

-- Theorem statement
theorem hyperbola_asymptote :
  ∀ x y : ℝ, hyperbola_equation x y → (∃ X Y : ℝ, X ≠ 0 ∧ asymptote_equation X Y) :=
sorry

end hyperbola_asymptote_l2808_280877


namespace jean_trips_l2808_280801

theorem jean_trips (total : ℕ) (extra : ℕ) (h1 : total = 40) (h2 : extra = 6) :
  ∃ (bill : ℕ) (jean : ℕ), bill + jean = total ∧ jean = bill + extra ∧ jean = 23 :=
by sorry

end jean_trips_l2808_280801


namespace shirt_price_markdown_l2808_280889

/-- Given a shirt price that goes through two markdowns, prove that the initial sale price
    was 70% of the original price if the second markdown is 10% and the final price
    is 63% of the original price. -/
theorem shirt_price_markdown (original_price : ℝ) (initial_sale_price : ℝ) :
  initial_sale_price > 0 →
  original_price > 0 →
  initial_sale_price * 0.9 = original_price * 0.63 →
  initial_sale_price / original_price = 0.7 := by
sorry

end shirt_price_markdown_l2808_280889


namespace intersection_of_A_and_B_l2808_280820

def A : Set ℚ := {x | ∃ k : ℕ, x = 3 * k + 1}
def B : Set ℚ := {x | x ≤ 7}

theorem intersection_of_A_and_B : A ∩ B = {1, 4, 7} := by sorry

end intersection_of_A_and_B_l2808_280820


namespace absolute_value_sum_zero_l2808_280814

theorem absolute_value_sum_zero (x y : ℝ) :
  |x - 2| + |4 - y| = 0 → x = 2 ∧ y = 4 := by
sorry

end absolute_value_sum_zero_l2808_280814


namespace project_hours_difference_l2808_280875

theorem project_hours_difference (total_hours : ℕ) (kate_hours : ℕ) : 
  total_hours = 153 →
  kate_hours + 2 * kate_hours + 6 * kate_hours = total_hours →
  6 * kate_hours - kate_hours = 85 :=
by
  sorry

end project_hours_difference_l2808_280875


namespace tan_roots_sum_angles_l2808_280817

theorem tan_roots_sum_angles (α β : Real) : 
  (∃ (x y : Real), x^2 + Real.sqrt 3 * x - 2 = 0 ∧ y^2 + Real.sqrt 3 * y - 2 = 0 ∧ 
   x = Real.tan α ∧ y = Real.tan β) →
  -π/2 < α ∧ α < π/2 →
  -π/2 < β ∧ β < π/2 →
  α + β = π/6 ∨ α + β = -5*π/6 :=
by sorry

end tan_roots_sum_angles_l2808_280817


namespace blue_paint_gallons_l2808_280847

theorem blue_paint_gallons (total : ℕ) (white : ℕ) (blue : ℕ) :
  total = 6689 →
  white + blue = total →
  8 * white = 5 * blue →
  blue = 4116 := by
sorry

end blue_paint_gallons_l2808_280847


namespace vector_computation_l2808_280892

theorem vector_computation : 
  4 • !![3, -5] - 3 • !![2, -6] + 2 • !![0, 3] = !![6, 4] := by sorry

end vector_computation_l2808_280892


namespace problem_solution_l2808_280828

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | (x - m + 2)*(x - m - 2) ≤ 0}

theorem problem_solution :
  (∀ m : ℝ, (A ∩ B m = {x | 0 ≤ x ∧ x ≤ 3}) → m = 2) ∧
  (∀ m : ℝ, (A ⊆ (Set.univ \ B m)) → (m < -3 ∨ m > 5)) :=
sorry

end problem_solution_l2808_280828


namespace reflection_of_point_A_l2808_280885

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The initial point A -/
def point_A : ℝ × ℝ := (1, 2)

theorem reflection_of_point_A :
  reflect_y_axis point_A = (-1, 2) := by
  sorry

end reflection_of_point_A_l2808_280885


namespace undefined_fraction_l2808_280821

theorem undefined_fraction (x : ℝ) : x = 1 → ¬∃y : ℝ, y = x / (x - 1) := by
  sorry

end undefined_fraction_l2808_280821


namespace greatest_integer_with_prime_absolute_value_l2808_280819

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_integer_with_prime_absolute_value :
  ∀ x : ℤ, (is_prime (Int.natAbs (8 * x^2 - 66 * x + 21))) →
    x ≤ 2 ∧ is_prime (Int.natAbs (8 * 2^2 - 66 * 2 + 21)) :=
by sorry

end greatest_integer_with_prime_absolute_value_l2808_280819


namespace registered_students_calculation_l2808_280856

/-- The number of students registered for a science course. -/
def registered_students (students_yesterday : ℕ) (students_absent_today : ℕ) : ℕ :=
  let students_today := (2 * students_yesterday) - (2 * students_yesterday / 10)
  students_today + students_absent_today

/-- Theorem stating the number of registered students given the problem conditions. -/
theorem registered_students_calculation :
  registered_students 70 30 = 156 := by
  sorry

#eval registered_students 70 30

end registered_students_calculation_l2808_280856


namespace robins_hair_length_l2808_280850

/-- Given Robin's initial hair length and the length he cut off, calculate his final hair length -/
theorem robins_hair_length (initial_length cut_length : ℕ) 
  (h1 : initial_length = 14)
  (h2 : cut_length = 13) :
  initial_length - cut_length = 1 := by
  sorry

end robins_hair_length_l2808_280850


namespace average_speed_calculation_l2808_280887

theorem average_speed_calculation (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) 
  (second_part_distance : ℝ) (second_part_speed : ℝ) : 
  total_distance = 850 ∧ 
  first_part_distance = 400 ∧ 
  first_part_speed = 20 ∧
  second_part_distance = 450 ∧
  second_part_speed = 15 →
  (total_distance / ((first_part_distance / first_part_speed) + (second_part_distance / second_part_speed))) = 17 := by
  sorry

end average_speed_calculation_l2808_280887


namespace ellipse_theorem_l2808_280882

/-- Given an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The equation of a line y = mx + c -/
structure Line where
  m : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def distance_to_line (p : Point) (l : Line) : ℝ := sorry

def eccentricity (e : Ellipse) : ℝ := sorry

theorem ellipse_theorem (e : Ellipse) 
  (h_distance : distance_to_line ⟨0, 0⟩ 
    {m := -e.b / e.a, c := e.a * e.b / (e.a^2 + e.b^2)} = 2 * Real.sqrt 5 / 5)
  (h_eccentricity : eccentricity e = Real.sqrt 3 / 2) :
  ∃ (l : Line), 
    (e.a = 2 ∧ e.b = 1) ∧ 
    (l.c = 5/3) ∧ 
    (l.m = 3 * Real.sqrt 14 / 14 ∨ l.m = -3 * Real.sqrt 14 / 14) ∧
    (∃ (m n : Point), 
      m.x^2 / 4 + m.y^2 = 1 ∧ 
      n.x^2 / 4 + n.y^2 = 1 ∧ 
      m.y = l.m * m.x + l.c ∧ 
      n.y = l.m * n.x + l.c ∧ 
      (m.x - 0)^2 + (m.y - 5/3)^2 = 4 * ((n.x - 0)^2 + (n.y - 5/3)^2)) := by
  sorry

end ellipse_theorem_l2808_280882


namespace m_minus_n_values_l2808_280800

theorem m_minus_n_values (m n : ℤ) 
  (hm : |m| = 4)
  (hn : |n| = 6)
  (hmn : |m + n| = m + n) :
  m - n = -2 ∨ m - n = -10 := by
sorry

end m_minus_n_values_l2808_280800


namespace valid_sequences_count_l2808_280891

/-- The number of distinct coin flip sequences of length n -/
def total_sequences (n : ℕ) : ℕ := 2^n

/-- The number of distinct coin flip sequences of length n starting with two heads -/
def sequences_starting_with_two_heads (n : ℕ) : ℕ := 2^(n-2)

/-- The number of valid coin flip sequences of length 10, excluding those starting with two heads -/
def valid_sequences : ℕ := total_sequences 10 - sequences_starting_with_two_heads 10

theorem valid_sequences_count : valid_sequences = 768 := by sorry

end valid_sequences_count_l2808_280891


namespace fifth_root_of_eight_to_fifteen_l2808_280872

theorem fifth_root_of_eight_to_fifteen (x : ℝ) : x = (8 ^ (1 / 5 : ℝ)) → x^15 = 512 := by
  sorry

end fifth_root_of_eight_to_fifteen_l2808_280872


namespace smallest_lcm_with_gcd_five_l2808_280880

theorem smallest_lcm_with_gcd_five (m n : ℕ) : 
  10000 ≤ m ∧ m < 100000 ∧ 
  10000 ≤ n ∧ n < 100000 ∧ 
  Nat.gcd m n = 5 →
  20030010 ≤ Nat.lcm m n :=
by sorry

end smallest_lcm_with_gcd_five_l2808_280880


namespace cookies_left_in_scenario_l2808_280836

/-- Represents the cookie-making scenario -/
structure CookieScenario where
  cookies_per_batch : ℕ
  flour_per_batch : ℕ
  flour_bags : ℕ
  flour_per_bag : ℕ
  cookies_eaten : ℕ

/-- Calculates the number of cookies left after baking and eating -/
def cookies_left (scenario : CookieScenario) : ℕ :=
  let total_flour := scenario.flour_bags * scenario.flour_per_bag
  let total_cookies := (total_flour / scenario.flour_per_batch) * scenario.cookies_per_batch
  total_cookies - scenario.cookies_eaten

/-- Theorem stating the number of cookies left in the given scenario -/
theorem cookies_left_in_scenario : 
  let scenario : CookieScenario := {
    cookies_per_batch := 12,
    flour_per_batch := 2,
    flour_bags := 4,
    flour_per_bag := 5,
    cookies_eaten := 15
  }
  cookies_left scenario = 105 := by
  sorry


end cookies_left_in_scenario_l2808_280836


namespace checkers_placement_divisibility_l2808_280846

/-- Given a prime p ≥ 5, r(p) is the number of ways to place p identical checkers 
    on a p × p checkerboard such that not all checkers are in the same row. -/
def r (p : ℕ) : ℕ := sorry

theorem checkers_placement_divisibility (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) : 
  p^5 ∣ r p := by
  sorry

end checkers_placement_divisibility_l2808_280846
