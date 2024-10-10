import Mathlib

namespace postman_speeds_theorem_l2603_260350

/-- Represents the speeds of the postman on different terrains -/
structure PostmanSpeeds where
  uphill : ℝ
  flat : ℝ
  downhill : ℝ

/-- Checks if the given speeds satisfy the journey conditions -/
def satisfiesConditions (speeds : PostmanSpeeds) : Prop :=
  let uphill := speeds.uphill
  let flat := speeds.flat
  let downhill := speeds.downhill
  (2 / uphill + 4 / flat + 3 / downhill = 2.267) ∧
  (3 / uphill + 4 / flat + 2 / downhill = 2.4) ∧
  (1 / uphill + 2 / flat + 1.5 / downhill = 1.158)

/-- Theorem stating that the specific speeds satisfy the journey conditions -/
theorem postman_speeds_theorem :
  satisfiesConditions { uphill := 3, flat := 4, downhill := 5 } := by
  sorry

#check postman_speeds_theorem

end postman_speeds_theorem_l2603_260350


namespace sin_negative_150_degrees_l2603_260354

theorem sin_negative_150_degrees :
  Real.sin (-(150 * π / 180)) = -1/2 := by sorry

end sin_negative_150_degrees_l2603_260354


namespace line_through_point_parallel_to_line_l2603_260317

/-- Given two lines in the form ax + by + c = 0, this function returns true if they are parallel -/
def are_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 * b2 = a2 * b1

/-- Given a line ax + by + c = 0 and a point (x0, y0), this function returns true if the point lies on the line -/
def point_on_line (a b c x0 y0 : ℝ) : Prop :=
  a * x0 + b * y0 + c = 0

theorem line_through_point_parallel_to_line :
  are_parallel 2 (-3) 12 2 (-3) 4 ∧
  point_on_line 2 (-3) 12 (-3) 2 := by
  sorry

end line_through_point_parallel_to_line_l2603_260317


namespace odd_prime_congruence_l2603_260312

theorem odd_prime_congruence (p : Nat) (c : Int) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ a : Int, (a^((p+1)/2) + (a+c)^((p+1)/2)) % p = c % p := by
  sorry

end odd_prime_congruence_l2603_260312


namespace binomial_coefficient_identity_l2603_260364

theorem binomial_coefficient_identity (n k : ℕ) (h : k ≤ n) :
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := by
  sorry

end binomial_coefficient_identity_l2603_260364


namespace last_student_number_l2603_260358

def skip_pattern (n : ℕ) : ℕ := 3 * n - 1

def student_number (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => skip_pattern (student_number n)

theorem last_student_number :
  ∃ (k : ℕ), student_number k = 242 ∧ 
  ∀ (m : ℕ), m > k → student_number m > 500 := by
sorry

end last_student_number_l2603_260358


namespace units_digit_of_seven_to_six_to_five_l2603_260334

theorem units_digit_of_seven_to_six_to_five (n : ℕ) :
  7^(6^5) ≡ 1 [MOD 10] :=
by sorry

end units_digit_of_seven_to_six_to_five_l2603_260334


namespace max_xy_value_l2603_260376

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  xy ≤ 1/8 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ x * y = 1/8 :=
sorry

end max_xy_value_l2603_260376


namespace expansion_coefficient_equals_negative_eighty_l2603_260385

/-- The coefficient of the term containing x in the expansion of (2√x - 1/x)^n -/
def coefficient (n : ℕ) : ℤ :=
  (-1)^((n-2)/3) * 2^((2*n+2)/3) * (n.choose ((n-2)/3))

theorem expansion_coefficient_equals_negative_eighty (n : ℕ) :
  coefficient n = -80 → n = 5 := by sorry

end expansion_coefficient_equals_negative_eighty_l2603_260385


namespace multiply_power_rule_l2603_260304

theorem multiply_power_rule (x : ℝ) : x * x^4 = x^5 := by
  sorry

end multiply_power_rule_l2603_260304


namespace inequality_equivalence_l2603_260370

theorem inequality_equivalence (x : ℝ) : 
  (12 * x^2 + 24 * x - 75) / ((3 * x - 5) * (x + 5)) < 4 ↔ -5 < x ∧ x < 5/3 :=
by sorry

end inequality_equivalence_l2603_260370


namespace angle_120_degrees_is_200_vens_l2603_260300

/-- Represents the number of vens in a full circle -/
def vens_in_full_circle : ℕ := 600

/-- Represents the number of degrees in a full circle -/
def degrees_in_full_circle : ℕ := 360

/-- Represents the angle in degrees we want to convert to vens -/
def angle_in_degrees : ℕ := 120

/-- Theorem stating that 120 degrees is equivalent to 200 vens -/
theorem angle_120_degrees_is_200_vens :
  (angle_in_degrees : ℚ) * vens_in_full_circle / degrees_in_full_circle = 200 := by
  sorry


end angle_120_degrees_is_200_vens_l2603_260300


namespace percentage_calculation_l2603_260339

theorem percentage_calculation : 
  (0.2 * (0.75 * 800)) / 4 = 30 := by
  sorry

end percentage_calculation_l2603_260339


namespace problem_solution_l2603_260345

-- Define the variables
def x : ℝ := 12 * (1 + 0.2)
def y : ℝ := 0.75 * x^2
def z : ℝ := 3 * y + 16
def w : ℝ := 2 * z - y
def v : ℝ := z^3 - 0.5 * y

-- State the theorem
theorem problem_solution :
  v = 112394885.1456 ∧ w = 809.6 := by
  sorry

end problem_solution_l2603_260345


namespace number_in_set_l2603_260393

theorem number_in_set (initial_avg : ℝ) (wrong_num : ℝ) (correct_num : ℝ) (correct_avg : ℝ) :
  initial_avg = 23 →
  wrong_num = 26 →
  correct_num = 36 →
  correct_avg = 24 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * initial_avg - wrong_num = (n : ℝ) * correct_avg - correct_num ∧
    n = 10 :=
by sorry

end number_in_set_l2603_260393


namespace rational_function_sum_l2603_260362

/-- Given rational functions p(x) and q(x) satisfying certain conditions,
    prove that their sum has a specific form. -/
theorem rational_function_sum (p q : ℝ → ℝ) : 
  (∀ x, ∃ y, q x = y * (x + 1) * (x - 2) * (x - 3)) →  -- q(x) is cubic with specific factors
  (∀ x, ∃ y, p x = y * (x + 1) * (x - 2)) →  -- p(x) is quadratic with specific factors
  p 2 = 2 →  -- p(2) = 2
  q (-1) = -1 →  -- q(-1) = -1
  ∀ x, p x + q x = x^3 - 3*x^2 + 4*x + 4 := by
sorry

end rational_function_sum_l2603_260362


namespace product_remainder_mod_seven_l2603_260380

theorem product_remainder_mod_seven : ((-1234 * 1984 * -1460 * 2008) % 7 = 0) := by
  sorry

end product_remainder_mod_seven_l2603_260380


namespace circle_transformation_l2603_260309

theorem circle_transformation (x₀ y₀ x y : ℝ) :
  x₀^2 + y₀^2 = 9 → x = x₀ → y = 4*y₀ → x^2/9 + y^2/144 = 1 := by
  sorry

end circle_transformation_l2603_260309


namespace equation_solution_l2603_260311

theorem equation_solution : 
  {x : ℝ | (16:ℝ)^x - (5/2) * (2:ℝ)^(2*x+1) + 4 = 0} = {0, 1} := by
  sorry

end equation_solution_l2603_260311


namespace square_area_from_perimeter_l2603_260344

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) :
  perimeter = 40 →
  area = (perimeter / 4) ^ 2 →
  area = 100 := by
sorry

end square_area_from_perimeter_l2603_260344


namespace total_jumps_l2603_260396

/-- Given that Ronald jumped 157 times and Rupert jumped 86 more times than Ronald,
    prove that the total number of jumps by both is 400. -/
theorem total_jumps (ronald_jumps : ℕ) (rupert_extra_jumps : ℕ) 
  (h1 : ronald_jumps = 157)
  (h2 : rupert_extra_jumps = 86) : 
  ronald_jumps + (ronald_jumps + rupert_extra_jumps) = 400 := by
  sorry

end total_jumps_l2603_260396


namespace youngest_age_in_office_l2603_260348

/-- Proves that in a group of 4 people whose ages form an arithmetic sequence,
    if the oldest person is 50 years old and the sum of their ages is 158 years,
    then the youngest person is 29 years old. -/
theorem youngest_age_in_office (ages : Fin 4 → ℕ) 
  (arithmetic_sequence : ∀ i j k : Fin 4, i < j → j < k → 
    ages j - ages i = ages k - ages j)
  (oldest_age : ages 3 = 50)
  (sum_of_ages : (Finset.univ.sum ages) = 158) :
  ages 0 = 29 := by
sorry

end youngest_age_in_office_l2603_260348


namespace decimal_to_fraction_l2603_260327

theorem decimal_to_fraction :
  (3.68 : ℚ) = 92 / 25 := by sorry

end decimal_to_fraction_l2603_260327


namespace parabola_intersection_l2603_260331

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 4
def parabola2 (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 8

theorem parabola_intersection :
  ∀ x y : ℝ, parabola1 x = parabola2 x ∧ y = parabola1 x ↔ (x = 3 ∧ y = 20) ∨ (x = 4 ∧ y = 32) :=
by sorry

end parabola_intersection_l2603_260331


namespace janet_number_problem_l2603_260343

theorem janet_number_problem (x : ℤ) : 
  2 * (x + 7) - 4 = 28 → x = 9 := by
  sorry

end janet_number_problem_l2603_260343


namespace geometric_sum_mod_500_l2603_260387

theorem geometric_sum_mod_500 : (Finset.sum (Finset.range 1001) (fun i => 3^i)) % 500 = 1 := by
  sorry

end geometric_sum_mod_500_l2603_260387


namespace remaining_salary_l2603_260388

theorem remaining_salary (salary : ℝ) (food_fraction house_rent_fraction clothes_fraction : ℝ) 
  (h1 : salary = 140000)
  (h2 : food_fraction = 1/5)
  (h3 : house_rent_fraction = 1/10)
  (h4 : clothes_fraction = 3/5)
  (h5 : food_fraction + house_rent_fraction + clothes_fraction < 1) :
  salary * (1 - (food_fraction + house_rent_fraction + clothes_fraction)) = 14000 :=
by sorry

end remaining_salary_l2603_260388


namespace max_true_statements_l2603_260359

theorem max_true_statements : ∃ x : ℝ, 
  (-1 < x ∧ x < 1) ∧ 
  (-1 < x^3 ∧ x^3 < 1) ∧ 
  (0 < x ∧ x < 1) ∧ 
  (0 < x^2 ∧ x^2 < 1) ∧ 
  (0 < x^3 - x^2 ∧ x^3 - x^2 < 1) := by
  sorry

end max_true_statements_l2603_260359


namespace unique_solution_l2603_260365

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem unique_solution :
  ∃! n : ℕ, n + S n = 1964 ∧ n = 1945 := by sorry

end unique_solution_l2603_260365


namespace ham_bread_percentage_l2603_260363

theorem ham_bread_percentage (bread_cost ham_cost cake_cost : ℚ) 
  (h1 : bread_cost = 50)
  (h2 : ham_cost = 150)
  (h3 : cake_cost = 200) :
  (bread_cost + ham_cost) / (bread_cost + ham_cost + cake_cost) = 1/2 := by
  sorry

end ham_bread_percentage_l2603_260363


namespace abc_sum_product_bound_l2603_260322

theorem abc_sum_product_bound (a b c : ℝ) (h : a + b + c = 3) :
  ∃ (M : ℝ), ∀ (x : ℝ), x ≤ M ∧ (∃ (a' b' c' : ℝ), a' + b' + c' = 3 ∧ a' * b' + a' * c' + b' * c' = x) :=
sorry

end abc_sum_product_bound_l2603_260322


namespace dave_tickets_proof_l2603_260399

/-- The number of tickets Dave won initially -/
def initial_tickets : ℕ := 11

/-- The number of tickets Dave spent on a beanie -/
def spent_tickets : ℕ := 5

/-- The number of additional tickets Dave won later -/
def additional_tickets : ℕ := 10

/-- The number of tickets Dave has now -/
def current_tickets : ℕ := 16

/-- Theorem stating that the initial number of tickets is correct given the conditions -/
theorem dave_tickets_proof :
  initial_tickets - spent_tickets + additional_tickets = current_tickets :=
by sorry

end dave_tickets_proof_l2603_260399


namespace scientific_notation_508_billion_yuan_l2603_260340

theorem scientific_notation_508_billion_yuan :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧
    508 * (10 ^ 9) = a * (10 ^ n) ∧
    a = 5.08 ∧ n = 11 := by
  sorry

end scientific_notation_508_billion_yuan_l2603_260340


namespace y_percent_of_x_l2603_260384

theorem y_percent_of_x (x y : ℝ) (h : 0.6 * (x - y) = 0.3 * (x + y)) : y / x = 1 / 3 := by
  sorry

end y_percent_of_x_l2603_260384


namespace events_independent_prob_A_or_B_l2603_260324

/-- The total number of balls -/
def total_balls : ℕ := 8

/-- The set of all ball numbers -/
def ball_numbers : Finset ℕ := Finset.range total_balls

/-- Event A: selecting a ball with an odd number -/
def event_A : Finset ℕ := ball_numbers.filter (λ n => n % 2 = 1)

/-- Event B: selecting a ball with a number that is a multiple of 3 -/
def event_B : Finset ℕ := ball_numbers.filter (λ n => n % 3 = 0)

/-- The probability of an event occurring -/
def prob (event : Finset ℕ) : ℚ := (event.card : ℚ) / total_balls

/-- The intersection of events A and B -/
def event_AB : Finset ℕ := event_A ∩ event_B

/-- Theorem: Events A and B are independent -/
theorem events_independent : prob event_AB = prob event_A * prob event_B := by sorry

/-- Theorem: The probability of A or B occurring is 5/8 -/
theorem prob_A_or_B : prob (event_A ∪ event_B) = 5 / 8 := by sorry

end events_independent_prob_A_or_B_l2603_260324


namespace unique_quadratic_solution_l2603_260315

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 6 * x + c = 0) →
  a + c = 12 →
  a < c →
  (a, c) = (6 - 3 * Real.sqrt 3, 6 + 3 * Real.sqrt 3) :=
by sorry

end unique_quadratic_solution_l2603_260315


namespace new_ratio_after_subtraction_l2603_260336

theorem new_ratio_after_subtraction :
  let a : ℚ := 72
  let b : ℚ := 192
  let subtrahend : ℚ := 24
  (a / b = 3 / 8) →
  ((a - subtrahend) / (b - subtrahend) = 1 / (7/2)) :=
by sorry

end new_ratio_after_subtraction_l2603_260336


namespace num_tangent_circles_bounds_l2603_260329

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a circle in 2D space --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The number of solutions for circles tangent to a line and another circle --/
def num_tangent_circles (r : ℝ) (L : Line) (C : Circle) : ℕ :=
  sorry

/-- Theorem stating the bounds on the number of tangent circles --/
theorem num_tangent_circles_bounds (r : ℝ) (L : Line) (C : Circle) :
  0 ≤ num_tangent_circles r L C ∧ num_tangent_circles r L C ≤ 8 :=
by sorry

end num_tangent_circles_bounds_l2603_260329


namespace factorial_difference_l2603_260357

theorem factorial_difference : Nat.factorial 9 - Nat.factorial 8 = 322560 := by
  sorry

end factorial_difference_l2603_260357


namespace determinant_transformation_l2603_260307

theorem determinant_transformation (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = -3 →
  Matrix.det !![p, 5*p + 2*q; r, 5*r + 2*s] = -6 := by
  sorry

end determinant_transformation_l2603_260307


namespace tangent_line_at_one_zero_l2603_260356

/-- The equation of the tangent line to y = x^3 - 2x + 1 at (1, 0) is y = x - 1 -/
theorem tangent_line_at_one_zero (x y : ℝ) : 
  (y = x^3 - 2*x + 1) → -- curve equation
  (1^3 - 2*1 + 1 = 0) → -- point (1, 0) lies on the curve
  (∀ t, (t - 1) * (3*1^2 - 2) = y - 0) → -- point-slope form of tangent line
  (y = x - 1) -- equation of tangent line
  := by sorry

end tangent_line_at_one_zero_l2603_260356


namespace cans_recycling_l2603_260308

theorem cans_recycling (total_cans : ℕ) (saturday_bags : ℕ) (cans_per_bag : ℕ) : 
  total_cans = 42 →
  saturday_bags = 4 →
  cans_per_bag = 6 →
  (total_cans - saturday_bags * cans_per_bag) / cans_per_bag = 3 :=
by sorry

end cans_recycling_l2603_260308


namespace cotton_collection_rate_l2603_260366

/-- The amount of cotton (in kg) that can be collected by a given number of workers in 2 days -/
def cotton_collected (w : ℕ) : ℝ := w * 8

theorem cotton_collection_rate 
  (h1 : 3 * (48 / 4) = 3 * 12)  -- 3 workers collect 48 kg in 4 days
  (h2 : 9 * 8 = 72) :  -- 9 workers collect 72 kg in 2 days
  ∀ w : ℕ, cotton_collected w = w * 8 := by
  sorry

#check cotton_collection_rate

end cotton_collection_rate_l2603_260366


namespace sum_of_cubes_difference_l2603_260310

theorem sum_of_cubes_difference (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 2700 → a + b + c = 11 := by
  sorry

end sum_of_cubes_difference_l2603_260310


namespace parabola_circle_intersection_l2603_260360

/-- Parabola M: y^2 = 4x -/
def parabola_M (x y : ℝ) : Prop := y^2 = 4*x

/-- Circle N: (x-1)^2 + y^2 = r^2 -/
def circle_N (x y r : ℝ) : Prop := (x - 1)^2 + y^2 = r^2

/-- Line l passing through (1, 0) -/
def line_l (m x y : ℝ) : Prop := x = m * y + 1

/-- Condition for |AC| = |BD| -/
def equal_distances (y₁ y₂ y₃ y₄ : ℝ) : Prop := |y₁ - y₃| = |y₂ - y₄|

/-- Main theorem -/
theorem parabola_circle_intersection (r : ℝ) :
  (r > 0) →
  (∃ (m₁ m₂ m₃ : ℝ),
    (∀ (m : ℝ), m ≠ m₁ ∧ m ≠ m₂ ∧ m ≠ m₃ →
      ¬(∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
        parabola_M x₁ y₁ ∧ parabola_M x₂ y₂ ∧
        circle_N x₃ y₃ r ∧ circle_N x₄ y₄ r ∧
        line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧
        line_l m x₃ y₃ ∧ line_l m x₄ y₄ ∧
        equal_distances y₁ y₂ y₃ y₄))) →
  r ≥ 3/2 :=
sorry

end parabola_circle_intersection_l2603_260360


namespace sphere_enclosed_by_truncated_cone_l2603_260303

theorem sphere_enclosed_by_truncated_cone (R r r' ζ : ℝ) 
  (h_positive : R > 0)
  (h_volume : (4/3) * π * R^3 * 2 = (4/3) * π * (r^2 + r * r' + r'^2) * R)
  (h_generator : (r + r')^2 = 4 * R^2 + (r - r')^2)
  (h_contact : ζ = (2 * r * r') / (r + r')) :
  r = (R/2) * (Real.sqrt 5 + 1) ∧ 
  r' = (R/2) * (Real.sqrt 5 - 1) ∧ 
  ζ = (2 * R * Real.sqrt 5) / 5 := by
sorry

end sphere_enclosed_by_truncated_cone_l2603_260303


namespace max_value_problem_l2603_260377

theorem max_value_problem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) :
  (2 * a * b * Real.sqrt 3 + 2 * a * c) ≤ Real.sqrt 3 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ a₀^2 + b₀^2 + c₀^2 = 1 ∧
    2 * a₀ * b₀ * Real.sqrt 3 + 2 * a₀ * c₀ = Real.sqrt 3 := by
  sorry

end max_value_problem_l2603_260377


namespace magic_sum_order_8_l2603_260361

def magic_sum (n : ℕ) : ℕ :=
  let total_sum := n^2 * (n^2 + 1) / 2
  total_sum / n

theorem magic_sum_order_8 :
  magic_sum 8 = 260 :=
by sorry

end magic_sum_order_8_l2603_260361


namespace cubic_identity_l2603_260323

theorem cubic_identity : ∀ x : ℝ, (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end cubic_identity_l2603_260323


namespace double_acute_angle_range_l2603_260386

/-- If θ is an acute angle, then 2θ is a positive angle less than 180°. -/
theorem double_acute_angle_range (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  0 < 2 * θ ∧ 2 * θ < Real.pi := by sorry

end double_acute_angle_range_l2603_260386


namespace largest_prime_divisor_l2603_260379

def base_8_number : ℕ := 201021022

theorem largest_prime_divisor :
  let decimal_number := 35661062
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ decimal_number ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ decimal_number → q ≤ p ∧ p = 17830531 := by
  sorry

#eval base_8_number

end largest_prime_divisor_l2603_260379


namespace tables_needed_l2603_260394

theorem tables_needed (total_children : ℕ) (children_per_table : ℕ) (h1 : total_children = 152) (h2 : children_per_table = 7) :
  ∃ (tables : ℕ), tables = 22 ∧ tables * children_per_table ≥ total_children ∧ (tables - 1) * children_per_table < total_children :=
by sorry

end tables_needed_l2603_260394


namespace tangent_circles_parallelism_l2603_260391

-- Define the types for our points and circles
variable (Point : Type) (Circle : Type)

-- Define the basic geometric relations
variable (on_circle : Point → Circle → Prop)
variable (on_line : Point → Point → Point → Prop)
variable (between : Point → Point → Point → Prop)
variable (tangent : Point → Point → Circle → Prop)
variable (intersect : Circle → Circle → Point → Point → Prop)
variable (cuts : Point → Point → Circle → Point → Prop)
variable (parallel : Point → Point → Point → Point → Prop)

-- Define our specific points and circles
variable (A B C P Q R S X Y Z : Point)
variable (C1 C2 : Circle)

-- State the theorem
theorem tangent_circles_parallelism 
  (h1 : intersect C1 C2 A B)
  (h2 : on_line A B C ∧ between A B C)
  (h3 : on_circle P C1 ∧ on_circle Q C2)
  (h4 : tangent C P C1 ∧ tangent C Q C2)
  (h5 : ¬on_circle P C2 ∧ ¬on_circle Q C1)
  (h6 : cuts P Q C1 R ∧ cuts P Q C2 S)
  (h7 : R ≠ P ∧ R ≠ Q ∧ R ≠ B ∧ S ≠ P ∧ S ≠ Q ∧ S ≠ B)
  (h8 : cuts C R C1 X ∧ cuts C S C2 Y)
  (h9 : on_line X Y Z) :
  parallel S Z Q X ↔ parallel P Z R X :=
sorry

end tangent_circles_parallelism_l2603_260391


namespace x_squared_inequality_l2603_260390

theorem x_squared_inequality (x : ℝ) (h : x^2 + x < 0) : x < x^2 ∧ x^2 < -x := by
  sorry

end x_squared_inequality_l2603_260390


namespace triangle_point_distance_inequality_triangle_point_distance_equality_condition_l2603_260381

-- Define a triangle ABC
variable (A B C : ℝ × ℝ)

-- Define a point P inside or on the boundary of triangle ABC
variable (P : ℝ × ℝ)

-- Define distances from P to sides of the triangle
def da : ℝ := sorry
def db : ℝ := sorry
def dc : ℝ := sorry

-- Define distances from P to vertices of the triangle
def AP : ℝ := sorry
def BP : ℝ := sorry
def CP : ℝ := sorry

-- Theorem statement
theorem triangle_point_distance_inequality :
  (max AP (max BP CP)) ≥ Real.sqrt (da^2 + db^2 + dc^2) :=
sorry

-- Equality condition
theorem triangle_point_distance_equality_condition :
  (max AP (max BP CP)) = Real.sqrt (da^2 + db^2 + dc^2) ↔
  (A = B ∧ B = C) ∧ P = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) :=
sorry

end triangle_point_distance_inequality_triangle_point_distance_equality_condition_l2603_260381


namespace square_ratio_side_length_sum_l2603_260368

theorem square_ratio_side_length_sum (area_ratio : ℚ) :
  area_ratio = 135 / 45 →
  ∃ (a b c : ℕ), 
    (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧
    (Real.sqrt (area_ratio) = (a * Real.sqrt b) / c) ∧
    (a + b + c = 5) := by
  sorry

end square_ratio_side_length_sum_l2603_260368


namespace no_solutions_factorial_equation_l2603_260397

theorem no_solutions_factorial_equation (n m : ℕ) (h : m ≥ 2) :
  n.factorial ≠ 2^m * m.factorial :=
sorry

end no_solutions_factorial_equation_l2603_260397


namespace hyperbola_properties_l2603_260373

/-- The hyperbola defined by the equation x^2 - y^2 = 1 passes through (1, 0) and has asymptotes x ± y = 0 -/
theorem hyperbola_properties :
  ∃ (x y : ℝ), 
    (x^2 - y^2 = 1) ∧ 
    (x = 1 ∧ y = 0) ∧
    (∀ (t : ℝ), (x = t ∧ y = t) ∨ (x = t ∧ y = -t)) :=
by sorry

end hyperbola_properties_l2603_260373


namespace contrapositive_equivalence_l2603_260325

theorem contrapositive_equivalence :
  (∀ x : ℝ, x < 3 → x^2 ≤ 9) ↔ (∀ x : ℝ, x^2 > 9 → x ≥ 3) :=
by sorry

end contrapositive_equivalence_l2603_260325


namespace probability_log3_is_integer_l2603_260335

/-- A three-digit number is a natural number between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The count of three-digit numbers that are powers of 3. -/
def CountPowersOfThree : ℕ := 2

/-- The total count of three-digit numbers. -/
def TotalThreeDigitNumbers : ℕ := 900

/-- The probability of a randomly chosen three-digit number being a power of 3. -/
def ProbabilityPowerOfThree : ℚ := CountPowersOfThree / TotalThreeDigitNumbers

theorem probability_log3_is_integer :
  ProbabilityPowerOfThree = 1 / 450 := by sorry

end probability_log3_is_integer_l2603_260335


namespace sqrt_neg_two_squared_l2603_260341

theorem sqrt_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by sorry

end sqrt_neg_two_squared_l2603_260341


namespace M_intersect_N_l2603_260321

-- Define set M
def M : Set ℝ := {0, 1, 2}

-- Define set N
def N : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}

-- Theorem statement
theorem M_intersect_N : M ∩ N = {1, 2} := by
  sorry

end M_intersect_N_l2603_260321


namespace saxbridge_parade_max_members_l2603_260346

theorem saxbridge_parade_max_members :
  ∀ n : ℕ,
  (15 * n < 1200) →
  (15 * n) % 24 = 3 →
  (∀ m : ℕ, (15 * m < 1200) ∧ (15 * m) % 24 = 3 → 15 * m ≤ 15 * n) →
  15 * n = 1155 :=
sorry

end saxbridge_parade_max_members_l2603_260346


namespace vertex_of_quadratic_l2603_260319

def f (x : ℝ) : ℝ := (x - 2)^2 - 3

theorem vertex_of_quadratic :
  ∃ (a b c : ℝ), f x = a * (x - b)^2 + c ∧ f b = c ∧ b = 2 ∧ c = -3 :=
sorry

end vertex_of_quadratic_l2603_260319


namespace log_inequality_l2603_260395

theorem log_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b := by
  sorry

end log_inequality_l2603_260395


namespace hexagon_quadrilateral_areas_l2603_260332

/-- The area of a regular hexagon -/
def hexagon_area : ℝ := 156

/-- The number of distinct quadrilateral shapes possible -/
def num_distinct_quadrilaterals : ℕ := 3

/-- The areas of the distinct quadrilaterals -/
def quadrilateral_areas : Set ℝ := {78, 104}

/-- Theorem: Given a regular hexagon with area 156 cm², the areas of all possible
    distinct quadrilaterals formed by its vertices are 78 cm² and 104 cm² -/
theorem hexagon_quadrilateral_areas :
  ∀ (area : ℝ), area ∈ quadrilateral_areas →
  ∃ (vertices : Finset (Fin 6)), vertices.card = 4 ∧
  (area = hexagon_area / 2 ∨ area = hexagon_area * 2 / 3) :=
sorry

end hexagon_quadrilateral_areas_l2603_260332


namespace factor_z6_minus_64_l2603_260353

theorem factor_z6_minus_64 (z : ℂ) : 
  z^6 - 64 = (z - 2) * (z^2 + 2*z + 4) * (z + 2) * (z^2 - 2*z + 4) := by
  sorry

#check factor_z6_minus_64

end factor_z6_minus_64_l2603_260353


namespace pigeon_hole_problem_l2603_260352

theorem pigeon_hole_problem (pigeonholes : ℕ) (pigeons : ℕ) : 
  (pigeons = 6 * pigeonholes + 3) →
  (pigeons + 5 = 8 * pigeonholes) →
  (pigeons = 27 ∧ pigeonholes = 4) := by
  sorry

end pigeon_hole_problem_l2603_260352


namespace employee_pay_l2603_260351

theorem employee_pay (x y z : ℝ) : 
  x + y + z = 900 →
  x = 1.2 * y →
  z = 0.8 * y →
  y = 300 := by
sorry

end employee_pay_l2603_260351


namespace bracket_6_times_3_l2603_260392

-- Define the custom bracket operation
def bracket (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 + 1 else 2 * x + 1

-- Theorem statement
theorem bracket_6_times_3 : bracket 6 * bracket 3 = 28 := by
  sorry

end bracket_6_times_3_l2603_260392


namespace frog_climb_days_l2603_260355

/-- The number of days required for a frog to climb out of a well -/
def days_to_climb (well_depth : ℕ) (climb_distance : ℕ) (slide_distance : ℕ) : ℕ :=
  (well_depth + climb_distance - slide_distance - 1) / (climb_distance - slide_distance) + 1

/-- Theorem: A frog in a 50-meter well, climbing 5 meters up and sliding 2 meters down daily, 
    takes at least 16 days to reach the top -/
theorem frog_climb_days :
  days_to_climb 50 5 2 ≥ 16 := by
  sorry

#eval days_to_climb 50 5 2

end frog_climb_days_l2603_260355


namespace fraction_simplification_l2603_260382

theorem fraction_simplification :
  (1/2 - 1/3) / ((3/7) * (2/8)) = 14/9 := by
  sorry

end fraction_simplification_l2603_260382


namespace smallest_nat_greater_than_12_l2603_260313

theorem smallest_nat_greater_than_12 :
  ∀ n : ℕ, n > 12 → n ≥ 13 :=
by
  sorry

end smallest_nat_greater_than_12_l2603_260313


namespace mans_running_speed_l2603_260318

/-- A proof that calculates a man's running speed given his walking speed and times. -/
theorem mans_running_speed (walking_speed : ℝ) (walking_time : ℝ) (running_time : ℝ) :
  walking_speed = 8 →
  walking_time = 3 →
  running_time = 1 →
  walking_speed * walking_time / running_time = 24 := by
  sorry

#check mans_running_speed

end mans_running_speed_l2603_260318


namespace edward_initial_money_l2603_260326

def books_cost : ℕ := 6
def pens_cost : ℕ := 16
def notebook_cost : ℕ := 5
def pencil_case_cost : ℕ := 3
def money_left : ℕ := 19

theorem edward_initial_money :
  books_cost + pens_cost + notebook_cost + pencil_case_cost + money_left = 49 := by
  sorry

end edward_initial_money_l2603_260326


namespace book_difference_l2603_260333

theorem book_difference (total : ℕ) (fiction : ℕ) (picture : ℕ)
  (h_total : total = 35)
  (h_fiction : fiction = 5)
  (h_picture : picture = 11)
  (h_autobio : ∃ autobio : ℕ, autobio = 2 * fiction) :
  ∃ nonfiction : ℕ, nonfiction - fiction = 4 :=
by sorry

end book_difference_l2603_260333


namespace alien_eggs_conversion_l2603_260375

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

/-- The number of eggs laid by the alien creature in base 7 -/
def alienEggsBase7 : ℕ := 215

theorem alien_eggs_conversion :
  base7ToBase10 alienEggsBase7 = 110 := by
  sorry

end alien_eggs_conversion_l2603_260375


namespace min_xy_min_x_plus_y_l2603_260371

-- Define the conditions
def condition (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ 2 * x + 8 * y - x * y = 0

-- Theorem for the minimum value of xy
theorem min_xy (x y : ℝ) (h : condition x y) :
  x * y ≥ 64 ∧ ∃ x y, condition x y ∧ x * y = 64 :=
sorry

-- Theorem for the minimum value of x + y
theorem min_x_plus_y (x y : ℝ) (h : condition x y) :
  x + y ≥ 18 ∧ ∃ x y, condition x y ∧ x + y = 18 :=
sorry

end min_xy_min_x_plus_y_l2603_260371


namespace bipartite_ramsey_theorem_l2603_260374

/-- A bipartite graph -/
structure BipartiteGraph where
  X : Type
  Y : Type
  E : X → Y → Prop

/-- An edge coloring of a bipartite graph -/
def EdgeColoring (G : BipartiteGraph) := G.X → G.Y → Bool

/-- A homomorphism between bipartite graphs -/
structure BipartiteHomomorphism (G H : BipartiteGraph) where
  φX : G.X → H.X
  φY : G.Y → H.Y
  preserves_edges : ∀ x y, G.E x y → H.E (φX x) (φY y)

/-- The main theorem -/
theorem bipartite_ramsey_theorem :
  ∀ P : BipartiteGraph, ∃ P' : BipartiteGraph,
    ∀ c : EdgeColoring P',
      ∃ φ : BipartiteHomomorphism P P',
        ∃ color : Bool,
          ∀ x y, P.E x y → c (φ.φX x) (φ.φY y) = color :=
sorry

end bipartite_ramsey_theorem_l2603_260374


namespace boxes_per_case_l2603_260372

/-- Given that Shirley sold 10 boxes of trefoils and needs to deliver 5 cases of boxes,
    prove that there are 2 boxes in each case. -/
theorem boxes_per_case (total_boxes : ℕ) (num_cases : ℕ) 
    (h1 : total_boxes = 10) (h2 : num_cases = 5) :
  total_boxes / num_cases = 2 := by
  sorry

end boxes_per_case_l2603_260372


namespace f_properties_l2603_260349

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 2 + x) * Real.cos (Real.pi / 2 - x)

theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ = -x₂ → f x₁ = -f x₂) ∧
  (∃ T : ℝ, T > 0 ∧ T < 2 * Real.pi ∧ ∀ x : ℝ, f (x + T) = f x) ∧
  (∀ x y : ℝ, -Real.pi/4 ≤ x ∧ x < y ∧ y ≤ Real.pi/4 → f x < f y) ∧
  (∀ x : ℝ, f (3 * Real.pi / 2 - x) = f (3 * Real.pi / 2 + x)) :=
by sorry

end f_properties_l2603_260349


namespace expression_equal_to_five_l2603_260314

theorem expression_equal_to_five : 3^2 - 2^2 = 5 := by
  sorry

#check expression_equal_to_five

end expression_equal_to_five_l2603_260314


namespace odd_function_negative_domain_l2603_260342

/-- An odd function defined on ℝ -/
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_positive : ∀ x ≥ 0, f x = x^2 - 2*x) :
  ∀ x < 0, f x = -x^2 - 2*x :=
by sorry

end odd_function_negative_domain_l2603_260342


namespace correct_product_with_decimals_l2603_260369

theorem correct_product_with_decimals (x y : ℚ) (z : ℕ) : 
  x = 0.035 → y = 3.84 → z = 13440 → x * y = 0.1344 := by
  sorry

end correct_product_with_decimals_l2603_260369


namespace cubic_roots_inequality_l2603_260328

theorem cubic_roots_inequality (a b c r s t : ℝ) :
  (∀ x, x^3 + a*x^2 + b*x + c = 0 ↔ x = r ∨ x = s ∨ x = t) →
  r ≥ s →
  s ≥ t →
  (a^2 - 3*b ≥ 0) ∧ (Real.sqrt (a^2 - 3*b) ≤ r - t) := by
  sorry

end cubic_roots_inequality_l2603_260328


namespace average_price_per_book_l2603_260367

theorem average_price_per_book (books1 books2 : ℕ) (price1 price2 : ℚ) :
  books1 = 40 →
  books2 = 20 →
  price1 = 600 →
  price2 = 240 →
  (price1 + price2) / (books1 + books2 : ℚ) = 14 :=
by sorry

end average_price_per_book_l2603_260367


namespace factorization_example_l2603_260337

/-- Represents a factorization from left to right -/
def is_factorization (f : ℝ → ℝ → ℝ) (g : ℝ → ℝ → ℝ) (h : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b, f a b = g a b * h a b

theorem factorization_example :
  is_factorization (λ a b => a^2*b + a*b^3) (λ a b => a*b) (λ a b => a + b^2) ∧
  ¬is_factorization (λ x _ => x^2 - 1) (λ x _ => x) (λ x _ => x - 1) ∧
  ¬is_factorization (λ x y => x^2 + 2*y + 1) (λ x y => x) (λ x y => x + 2*y) ∧
  ¬is_factorization (λ x y => x*(x+y)) (λ x _ => x^2) (λ _ y => y) :=
by sorry

end factorization_example_l2603_260337


namespace none_always_true_l2603_260305

/-- Given r > 0 and x^2 + y^2 > x^2y^2 for x, y ≠ 0, none of the following statements are true for all x and y -/
theorem none_always_true (r : ℝ) (x y : ℝ) (hr : r > 0) (hxy : x ≠ 0 ∧ y ≠ 0) (h : x^2 + y^2 > x^2 * y^2) :
  ¬(∀ x y : ℝ, -x > -y) ∧
  ¬(∀ x y : ℝ, -x > y) ∧
  ¬(∀ x y : ℝ, 1 > -y/x) ∧
  ¬(∀ x y : ℝ, 1 < x/y) :=
by sorry

end none_always_true_l2603_260305


namespace sin_sq_plus_4cos_max_value_l2603_260398

theorem sin_sq_plus_4cos_max_value (x : ℝ) : 
  Real.sin x ^ 2 + 4 * Real.cos x ≤ 4 := by
sorry

end sin_sq_plus_4cos_max_value_l2603_260398


namespace stationery_cost_theorem_l2603_260383

/-- Calculates the total cost of stationery given the number of pencil boxes, pencils per box,
    pencil cost, pen cost, and additional pens ordered. -/
def total_stationery_cost (pencil_boxes : ℕ) (pencils_per_box : ℕ) (pencil_cost : ℕ) 
                          (pen_cost : ℕ) (additional_pens : ℕ) : ℕ :=
  let total_pencils := pencil_boxes * pencils_per_box
  let total_pens := 2 * total_pencils + additional_pens
  let pencil_total_cost := total_pencils * pencil_cost
  let pen_total_cost := total_pens * pen_cost
  pencil_total_cost + pen_total_cost

/-- Theorem stating that the total cost of stationery for the given conditions is $18300. -/
theorem stationery_cost_theorem : 
  total_stationery_cost 15 80 4 5 300 = 18300 := by
  sorry

end stationery_cost_theorem_l2603_260383


namespace second_number_proof_l2603_260389

theorem second_number_proof (x : ℝ) : 3 + x + 333 + 3.33 = 369.63 → x = 30.3 := by
  sorry

end second_number_proof_l2603_260389


namespace quadratic_form_simplification_l2603_260330

theorem quadratic_form_simplification 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ K : ℝ, ∀ x : ℝ, 
    (x + a)^2 / ((a - b) * (a - c)) + 
    (x + b)^2 / ((b - a) * (b - c + 2)) + 
    (x + c)^2 / ((c - a) * (c - b)) = 
    x^2 - (a + b + c) * x + K :=
by sorry

end quadratic_form_simplification_l2603_260330


namespace divisibility_condition_l2603_260320

theorem divisibility_condition (x y : ℕ+) :
  (∃ k : ℤ, (2 * x * y^2 - y^3 + 1 : ℤ) = k * x^2) ↔
  (∃ t : ℕ+, (x = 2 * t ∧ y = 1) ∨
             (x = t ∧ y = 2 * t) ∨
             (x = 8 * t^4 - t ∧ y = 2 * t)) :=
sorry

end divisibility_condition_l2603_260320


namespace grain_production_scientific_notation_l2603_260316

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) (h : x > 0) : ScientificNotation :=
  sorry

theorem grain_production_scientific_notation :
  toScientificNotation 686530000 (by norm_num) =
    ScientificNotation.mk 6.8653 8 (by norm_num) :=
  sorry

end grain_production_scientific_notation_l2603_260316


namespace fencing_cost_per_meter_l2603_260301

/-- Given a rectangular plot with specified dimensions and total fencing cost,
    calculate the cost per meter of fencing. -/
theorem fencing_cost_per_meter
  (length : ℝ) (breadth : ℝ) (total_cost : ℝ)
  (h1 : length = 60)
  (h2 : breadth = 40)
  (h3 : total_cost = 5300) :
  total_cost / (2 * (length + breadth)) = 26.5 := by
  sorry

end fencing_cost_per_meter_l2603_260301


namespace boy_running_speed_l2603_260347

/-- Calculates the speed of a boy running around a square field -/
theorem boy_running_speed (side : ℝ) (time : ℝ) (speed_kmh : ℝ) : 
  side = 40 → 
  time = 64 → 
  speed_kmh = (4 * side / time) * 3.6 →
  speed_kmh = 9 := by
  sorry

#check boy_running_speed

end boy_running_speed_l2603_260347


namespace g_in_terms_of_f_l2603_260306

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the theorem
theorem g_in_terms_of_f : ∀ x : ℝ, g x = f (6 - x) := by sorry

end g_in_terms_of_f_l2603_260306


namespace closest_point_sum_l2603_260378

/-- The point (a, b) on the line y = -3x + 10 that is closest to (16, 8) satisfies a + b = 8.8 -/
theorem closest_point_sum (a b : ℝ) : 
  (b = -3 * a + 10) →  -- Mouse path equation
  (∀ x y : ℝ, y = -3 * x + 10 → (x - 16)^2 + (y - 8)^2 ≥ (a - 16)^2 + (b - 8)^2) →  -- (a, b) is closest to (16, 8)
  a + b = 8.8 := by
  sorry

end closest_point_sum_l2603_260378


namespace regular_polygon_perimeter_l2603_260338

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) : 
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  exterior_angle = 360 / n →
  n * side_length = 28 := by
  sorry

end regular_polygon_perimeter_l2603_260338


namespace yuna_position_l2603_260302

/-- Given Eunji's position and Yuna's relative position after Eunji, 
    calculate Yuna's absolute position on the train. -/
theorem yuna_position (eunji_pos yuna_after : ℕ) : 
  eunji_pos = 100 → yuna_after = 11 → eunji_pos + yuna_after = 111 := by
  sorry

#check yuna_position

end yuna_position_l2603_260302
