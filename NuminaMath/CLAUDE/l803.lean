import Mathlib

namespace kelly_cheese_packages_l803_80345

/-- The number of packages of string cheese needed for school lunches --/
def string_cheese_packages (days_per_week : ℕ) (oldest_daily : ℕ) (youngest_daily : ℕ) 
  (cheeses_per_package : ℕ) (num_weeks : ℕ) : ℕ :=
  let total_cheeses := (oldest_daily + youngest_daily) * days_per_week * num_weeks
  (total_cheeses + cheeses_per_package - 1) / cheeses_per_package

/-- Theorem: Kelly needs 2 packages of string cheese for 4 weeks of school lunches --/
theorem kelly_cheese_packages : 
  string_cheese_packages 5 2 1 30 4 = 2 := by
  sorry

end kelly_cheese_packages_l803_80345


namespace dihedral_angle_measure_l803_80342

/-- Represents a dihedral angle formed by two plane mirrors -/
structure DihedralAngle where
  angle : ℝ

/-- Represents a light ray in the context of the problem -/
structure LightRay where
  perpendicular_to_edge : Bool
  parallel_to_first_mirror : Bool

/-- Represents the reflection pattern of the light ray -/
inductive ReflectionPattern
  | Alternating : ReflectionPattern

/-- Represents the result of the light ray's path -/
inductive PathResult
  | ReturnsAlongSamePath : PathResult

theorem dihedral_angle_measure 
  (d : DihedralAngle) 
  (l : LightRay) 
  (r : ReflectionPattern) 
  (p : PathResult) :
  l.perpendicular_to_edge = true →
  l.parallel_to_first_mirror = true →
  r = ReflectionPattern.Alternating →
  p = PathResult.ReturnsAlongSamePath →
  d.angle = 30 := by
  sorry

end dihedral_angle_measure_l803_80342


namespace binomial_expansion_coefficient_l803_80314

theorem binomial_expansion_coefficient (n : ℕ) : 
  (9 : ℕ) * (n.choose 2) = 54 → n = 4 := by
  sorry

end binomial_expansion_coefficient_l803_80314


namespace max_parts_three_planes_is_eight_l803_80357

/-- The maximum number of parts that three planes can divide space into -/
def max_parts_three_planes : ℕ := 8

/-- Theorem stating that the maximum number of parts that three planes can divide space into is 8 -/
theorem max_parts_three_planes_is_eight :
  max_parts_three_planes = 8 := by sorry

end max_parts_three_planes_is_eight_l803_80357


namespace fencing_problem_l803_80382

theorem fencing_problem (area : ℝ) (uncovered_side : ℝ) :
  area = 600 ∧ uncovered_side = 10 →
  ∃ width : ℝ, 
    area = uncovered_side * width ∧
    uncovered_side + 2 * width = 130 :=
by sorry

end fencing_problem_l803_80382


namespace subtraction_problem_l803_80385

theorem subtraction_problem : 4444444444444 - 2222222222222 - 444444444444 = 1777777777778 := by
  sorry

end subtraction_problem_l803_80385


namespace f_properties_l803_80368

noncomputable def f (x : ℝ) := Real.exp x - x + (1/2) * x^2

theorem f_properties :
  (∃ (x₀ : ℝ), f x₀ = 1 ∧ ∀ (x : ℝ), f x ≥ f x₀) ∧  -- Minimum value is 1
  (∀ (M : ℝ), ∃ (x : ℝ), f x > M) ∧                -- No maximum value
  (∀ (a b : ℝ), (∀ (x : ℝ), (1/2) * x^2 - f x ≤ a * x + b) →
    (1 - a) * b ≥ -Real.exp 1 / 2) ∧               -- Minimum value of (1-a)b
  (∃ (a b : ℝ), (∀ (x : ℝ), (1/2) * x^2 - f x ≤ a * x + b) ∧
    (1 - a) * b = -Real.exp 1 / 2) :=               -- Minimum is attained
by sorry

end f_properties_l803_80368


namespace average_of_five_quantities_l803_80390

theorem average_of_five_quantities (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : (q1 + q2 + q3) / 3 = 4)
  (h2 : (q4 + q5) / 2 = 33) : 
  (q1 + q2 + q3 + q4 + q5) / 5 = 15.6 := by
  sorry

end average_of_five_quantities_l803_80390


namespace complex_fraction_equality_l803_80313

theorem complex_fraction_equality : ((-1 : ℂ) + 3*I) / (1 + I) = 1 + 2*I := by
  sorry

end complex_fraction_equality_l803_80313


namespace power_function_domain_and_oddness_l803_80388

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem power_function_domain_and_oddness (a : ℤ) :
  a ∈ ({-1, 1, 3} : Set ℤ) →
  (∀ x : ℝ, ∃ y : ℝ, y = x^a) ∧ is_odd_function (λ x : ℝ ↦ x^a) ↔
  a ∈ ({1, 3} : Set ℤ) := by
  sorry

end power_function_domain_and_oddness_l803_80388


namespace x_plus_y_equals_two_l803_80339

theorem x_plus_y_equals_two (x y : ℝ) (h : |x - 6| + (y + 4)^2 = 0) : x + y = 2 := by
  sorry

end x_plus_y_equals_two_l803_80339


namespace daria_concert_friends_l803_80344

def ticket_cost : ℕ := 90
def current_money : ℕ := 189
def additional_money_needed : ℕ := 171

def total_cost : ℕ := current_money + additional_money_needed

def total_tickets : ℕ := total_cost / ticket_cost

def number_of_friends : ℕ := total_tickets - 1

theorem daria_concert_friends : number_of_friends = 3 := by
  sorry

end daria_concert_friends_l803_80344


namespace line_intersects_circle_l803_80354

/-- The line 4x - 3y = 0 intersects the circle x^2 + y^2 = 36 -/
theorem line_intersects_circle :
  ∃ (x y : ℝ), 4 * x - 3 * y = 0 ∧ x^2 + y^2 = 36 := by
  sorry

end line_intersects_circle_l803_80354


namespace pasture_perimeter_difference_l803_80398

/-- Calculates the perimeter of a pasture given the number of stakes and the interval between stakes -/
def pasture_perimeter (stakes : ℕ) (interval : ℕ) : ℕ := stakes * interval

/-- The difference between the perimeters of two pastures -/
theorem pasture_perimeter_difference : 
  pasture_perimeter 82 20 - pasture_perimeter 96 10 = 680 := by
  sorry

end pasture_perimeter_difference_l803_80398


namespace max_value_sin_cos_product_l803_80362

theorem max_value_sin_cos_product (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) *
  (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) ≤ 4.5 ∧
  ∃ x y z : ℝ, (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) *
              (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) = 4.5 :=
by sorry

end max_value_sin_cos_product_l803_80362


namespace point_on_line_l803_80358

theorem point_on_line (k : ℝ) : 
  (1 + 3 * k * (-1/3) = -4 * 4) → k = 17 := by
sorry

end point_on_line_l803_80358


namespace largest_fraction_l803_80310

def fraction_set : Set ℚ := {1/2, 1/3, 1/4, 1/5, 1/10}

theorem largest_fraction :
  ∀ x ∈ fraction_set, (1/2 : ℚ) ≥ x :=
by sorry

end largest_fraction_l803_80310


namespace rice_purchase_comparison_l803_80355

theorem rice_purchase_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (200 / (100 / a + 100 / b)) ≤ ((100 * a + 100 * b) / 200) := by
  sorry

#check rice_purchase_comparison

end rice_purchase_comparison_l803_80355


namespace intersection_range_l803_80300

theorem intersection_range (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    x₁^2 - y₁^2 = 6 ∧ 
    x₂^2 - y₂^2 = 6 ∧ 
    y₁ = k * x₁ + 2 ∧ 
    y₂ = k * x₂ + 2 ∧ 
    x₁ > 0 ∧ 
    x₂ > 0) → 
  -Real.sqrt 15 / 3 < k ∧ k < -1 := by
sorry

end intersection_range_l803_80300


namespace function_domain_l803_80374

/-- The function y = √(x-1) / (x-2) is defined for x ≥ 1 and x ≠ 2 -/
theorem function_domain (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (x - 1) / (x - 2)) ↔ (x ≥ 1 ∧ x ≠ 2) := by
  sorry

end function_domain_l803_80374


namespace undefined_expression_l803_80366

theorem undefined_expression (x : ℝ) : 
  (x^2 - 18*x + 81 = 0) ↔ (x = 9) := by
  sorry

#check undefined_expression

end undefined_expression_l803_80366


namespace choose_three_from_nine_l803_80341

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end choose_three_from_nine_l803_80341


namespace prob_three_correct_is_five_twelfths_l803_80336

-- Define the probability of A and B guessing correctly
def prob_A_correct : ℚ := 3/4
def prob_B_correct : ℚ := 2/3

-- Define the function to calculate the probability of exactly three correct guesses
def prob_three_correct : ℚ :=
  let p_A := prob_A_correct
  let p_B := prob_B_correct
  let q_A := 1 - p_A
  let q_B := 1 - p_B
  
  -- Calculate the probability of each scenario
  let scenario1 := p_A * p_A * p_A * p_B * q_B * q_B * q_B
  let scenario2 := p_A * p_A * p_A * p_B * q_B * p_B * q_B
  let scenario3 := p_A * p_A * p_A * p_B * p_B * q_B * q_B
  let scenario4 := p_A * p_A * p_A * q_B * p_B * p_B * q_B
  
  -- Sum up all scenarios
  scenario1 + scenario2 + scenario3 + scenario4

-- Theorem statement
theorem prob_three_correct_is_five_twelfths :
  prob_three_correct = 5/12 := by
  sorry

end prob_three_correct_is_five_twelfths_l803_80336


namespace intersection_of_A_and_B_l803_80346

def A : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def B : Set ℝ := {-2, -1, 0, 1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2, 3} := by
  sorry

end intersection_of_A_and_B_l803_80346


namespace gcd_problem_l803_80316

theorem gcd_problem (n : ℕ) : 
  75 ≤ n ∧ n ≤ 90 ∧ Nat.gcd n 15 = 5 → n = 80 ∨ n = 85 := by
  sorry

end gcd_problem_l803_80316


namespace string_average_length_l803_80369

theorem string_average_length (s1 s2 s3 : ℝ) 
  (h1 : s1 = 2) (h2 : s2 = 3) (h3 : s3 = 7) : 
  (s1 + s2 + s3) / 3 = 4 := by
  sorry

end string_average_length_l803_80369


namespace functional_equation_solutions_l803_80340

/-- A function satisfying the given functional equation is either constantly zero or f(x) = x - 1. -/
theorem functional_equation_solutions (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (x - 2) * f y + f (y + 2 * f x) = f (x + y * f x)) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x - 1) := by
  sorry

end functional_equation_solutions_l803_80340


namespace line_through_point_parallel_to_line_l803_80360

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (given_line : Line) 
  (point : ℝ × ℝ) : 
  ∃ (result_line : Line), 
    result_line.contains point.1 point.2 ∧ 
    result_line.parallel given_line ∧
    result_line.a = 1 ∧ 
    result_line.b = 2 ∧ 
    result_line.c = -3 :=
by sorry

end line_through_point_parallel_to_line_l803_80360


namespace equation_solution_l803_80363

theorem equation_solution :
  let S : Set ℂ := {x | (x - 4)^4 + (x - 6)^4 = 16}
  S = {5 + Complex.I * Real.sqrt 7, 5 - Complex.I * Real.sqrt 7, 6, 4} := by
  sorry

end equation_solution_l803_80363


namespace vacation_speed_problem_l803_80334

theorem vacation_speed_problem (distance1 distance2 time_diff : ℝ) 
  (h1 : distance1 = 100)
  (h2 : distance2 = 175)
  (h3 : time_diff = 3)
  (h4 : distance2 / speed = distance1 / speed + time_diff)
  (speed : ℝ) :
  speed = 25 := by
sorry

end vacation_speed_problem_l803_80334


namespace f_of_i_eq_zero_l803_80356

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the function f
def f (x : ℂ) : ℂ := x^3 - x^2 + x - 1

-- Theorem statement
theorem f_of_i_eq_zero : f i = 0 := by sorry

end f_of_i_eq_zero_l803_80356


namespace smallest_a_value_l803_80380

-- Define the arithmetic sequence
def is_arithmetic_sequence (a b c : ℕ) : Prop := b - a = c - b

-- Define the function f
def f (a b c : ℕ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem smallest_a_value (a b c : ℕ) (r s : ℝ) :
  is_arithmetic_sequence a b c →
  a < b →
  b < c →
  f a b c r = s →
  f a b c s = r →
  r * s = 2017 →
  ∃ (min_a : ℕ), min_a = 1 ∧ ∀ (a' : ℕ), (∃ (b' c' : ℕ) (r' s' : ℝ),
    is_arithmetic_sequence a' b' c' ∧
    a' < b' ∧
    b' < c' ∧
    f a' b' c' r' = s' ∧
    f a' b' c' s' = r' ∧
    r' * s' = 2017) → a' ≥ min_a :=
sorry

end smallest_a_value_l803_80380


namespace pi_half_irrational_l803_80347

theorem pi_half_irrational : Irrational (π / 2) :=
by
  sorry

end pi_half_irrational_l803_80347


namespace pop_survey_l803_80375

theorem pop_survey (total : ℕ) (pop_angle : ℕ) (pop_count : ℕ) : 
  total = 472 →
  pop_angle = 251 →
  (pop_count : ℝ) / total * 360 ≥ pop_angle.pred →
  (pop_count : ℝ) / total * 360 < pop_angle.succ →
  pop_count = 329 := by
sorry

end pop_survey_l803_80375


namespace tailor_buttons_count_l803_80393

/-- The number of green buttons purchased by the tailor -/
def green_buttons : ℕ := 90

/-- The number of yellow buttons purchased by the tailor -/
def yellow_buttons : ℕ := green_buttons + 10

/-- The number of blue buttons purchased by the tailor -/
def blue_buttons : ℕ := green_buttons - 5

/-- The total number of buttons purchased by the tailor -/
def total_buttons : ℕ := green_buttons + yellow_buttons + blue_buttons

theorem tailor_buttons_count : total_buttons = 275 := by
  sorry

end tailor_buttons_count_l803_80393


namespace line_through_circle_center_l803_80335

/-- The value of 'a' when the line 3x + y + a = 0 passes through the center of the circle x^2 + y^2 + 2x - 4y = 0 -/
theorem line_through_circle_center (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0 ∧ 3*x + y + a = 0 ∧ 
   ∀ x' y' : ℝ, x'^2 + y'^2 + 2*x' - 4*y' = 0 → (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2) →
  a = 1 :=
by sorry

end line_through_circle_center_l803_80335


namespace vaccine_effectiveness_l803_80321

-- Define the contingency table data
def a : ℕ := 10  -- Injected and Infected
def b : ℕ := 40  -- Injected and Not Infected
def c : ℕ := 20  -- Not Injected and Infected
def d : ℕ := 30  -- Not Injected and Not Infected
def n : ℕ := 100 -- Total number of observations

-- Define the K² formula
def K_squared : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the thresholds
def lower_threshold : ℚ := 3841 / 1000
def upper_threshold : ℚ := 5024 / 1000

-- Theorem statement
theorem vaccine_effectiveness :
  lower_threshold < K_squared ∧ K_squared < upper_threshold :=
sorry

end vaccine_effectiveness_l803_80321


namespace inequality_solution_l803_80330

def solution_set (m : ℝ) : Set ℝ :=
  if m < -4 then {x | -1 < x ∧ x < 1 / (m + 3)}
  else if m = -4 then ∅
  else if m > -4 ∧ m < -3 then {x | 1 / (m + 3) < x ∧ x < -1}
  else if m = -3 then {x | x > -1}
  else {x | x < -1 ∨ x > 1 / (m + 3)}

theorem inequality_solution (m : ℝ) :
  {x : ℝ | ((m + 3) * x - 1) * (x + 1) > 0} = solution_set m := by
  sorry

end inequality_solution_l803_80330


namespace passing_marks_l803_80315

/-- The passing marks problem -/
theorem passing_marks (T P : ℝ) 
  (h1 : 0.40 * T = P - 40)
  (h2 : 0.60 * T = P + 20)
  (h3 : 0.45 * T = P - 10) : 
  P = 160 := by sorry

end passing_marks_l803_80315


namespace investment_problem_l803_80372

theorem investment_problem (P : ℝ) : 
  let A1 := 1.02 * P - 100
  let A2 := 1.03 * A1 + 200
  let A3 := 1.04 * A2
  let A4 := 1.05 * A3
  let A5 := 1.06 * A4
  A5 = 750 →
  1.19304696 * P + 112.27824 = 750 :=
by sorry

end investment_problem_l803_80372


namespace fraction_simplification_l803_80370

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) :
  (7 * a + 7 * b) / (a^2 - b^2) = 7 / (a - b) := by
  sorry

end fraction_simplification_l803_80370


namespace bicycle_problem_l803_80332

/-- Prove that given the conditions of the bicycle problem, student B's speed is 12 km/h -/
theorem bicycle_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) 
  (h1 : distance = 12)
  (h2 : speed_ratio = 1.2)
  (h3 : time_difference = 1/6) : 
  let speed_B := 
    distance * (speed_ratio - 1) / (distance * speed_ratio * time_difference - distance * time_difference)
  speed_B = 12 := by
  sorry

end bicycle_problem_l803_80332


namespace boys_to_girls_ratio_l803_80323

/-- Given a family with the following properties:
  * The total number of children is 180
  * Boys are given $3900 to share
  * Each boy receives $52
  Prove that the ratio of boys to girls is 5:7 -/
theorem boys_to_girls_ratio (total_children : ℕ) (boys_money : ℕ) (boy_share : ℕ)
  (h_total : total_children = 180)
  (h_money : boys_money = 3900)
  (h_share : boy_share = 52)
  : ∃ (boys girls : ℕ), boys + girls = total_children ∧ 
    boys * boy_share = boys_money ∧
    boys * 7 = girls * 5 := by
  sorry

end boys_to_girls_ratio_l803_80323


namespace smallest_x_prime_factorization_l803_80322

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2
def is_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3
def is_fifth_power (n : ℕ) : Prop := ∃ m : ℕ, n = m^5

def satisfies_conditions (x : ℕ) : Prop :=
  is_square (2 * x) ∧ is_cube (3 * x) ∧ is_fifth_power (5 * x)

theorem smallest_x_prime_factorization :
  ∃ x : ℕ, 
    satisfies_conditions x ∧ 
    (∀ y : ℕ, satisfies_conditions y → x ≤ y) ∧
    x = 2^15 * 3^20 * 5^24 :=
sorry

end smallest_x_prime_factorization_l803_80322


namespace omega_double_omega_8n_plus_5_omega_2_pow_n_minus_1_l803_80328

-- Define a function to represent the binary expansion of a non-negative integer
def binaryExpansion (n : ℕ) : List (Fin 2) := sorry

-- Define the ω function
def ω (n : ℕ) : ℕ := (binaryExpansion n).sum

-- Theorem 1
theorem omega_double (n : ℕ) : ω (2 * n) = ω n := by sorry

-- Theorem 2
theorem omega_8n_plus_5 (n : ℕ) : ω (8 * n + 5) = ω (4 * n + 3) := by sorry

-- Theorem 3
theorem omega_2_pow_n_minus_1 (n : ℕ) : ω (2^n - 1) = n := by sorry

end omega_double_omega_8n_plus_5_omega_2_pow_n_minus_1_l803_80328


namespace smallest_possible_value_l803_80309

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = 13 * a (n - 1) - 2 * n

/-- The sequence is positive -/
def PositiveSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

theorem smallest_possible_value (a : ℕ → ℝ) 
    (h_recurrence : RecurrenceSequence a) 
    (h_positive : PositiveSequence a) :
    (∀ a₁ : ℝ, a 1 ≥ a₁ → a₁ ≥ 13/36) :=
  sorry

end smallest_possible_value_l803_80309


namespace drop_1m_l803_80359

def water_level_change (change : ℝ) : ℝ := change

axiom rise_positive (x : ℝ) : x > 0 → water_level_change x > 0
axiom rise_4m : water_level_change 4 = 4

theorem drop_1m : water_level_change (-1) = -1 := by sorry

end drop_1m_l803_80359


namespace factorial_equation_l803_80343

/-- Definition of factorial for positive integers -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- Theorem stating the equality of 7! * 11! and 15 * 12! -/
theorem factorial_equation : factorial 7 * factorial 11 = 15 * factorial 12 := by
  sorry

end factorial_equation_l803_80343


namespace equation_solutions_l803_80311

/-- The set of solutions to the equation (x^3 + 3x^2√3 + 9x + 3√3) + (x + √3) = 0 -/
def solution_set : Set ℂ :=
  {z : ℂ | z = -Real.sqrt 3 ∨ z = -Real.sqrt 3 + Complex.I ∨ z = -Real.sqrt 3 - Complex.I}

/-- The equation (x^3 + 3x^2√3 + 9x + 3√3) + (x + √3) = 0 -/
def equation (x : ℂ) : Prop :=
  (x^3 + 3*x^2*Real.sqrt 3 + 9*x + 3*Real.sqrt 3) + (x + Real.sqrt 3) = 0

theorem equation_solutions :
  ∀ x : ℂ, equation x ↔ x ∈ solution_set := by
  sorry

end equation_solutions_l803_80311


namespace tangent_circles_count_l803_80303

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Determines if a circle is tangent to two other circles -/
def is_tangent_to_both (c : Circle) (c1 c2 : Circle) : Prop :=
  are_externally_tangent c c1 ∧ are_externally_tangent c c2

/-- The main theorem to be proven -/
theorem tangent_circles_count 
  (O1 O2 : Circle) 
  (h_tangent : are_externally_tangent O1 O2) 
  (h_radius1 : O1.radius = 2) 
  (h_radius2 : O2.radius = 4) : 
  ∃! (s : Finset Circle), 
    Finset.card s = 5 ∧ 
    ∀ c ∈ s, c.radius = 6 ∧ is_tangent_to_both c O1 O2 :=
sorry

end tangent_circles_count_l803_80303


namespace intersection_point_exists_l803_80349

/-- Square with side length 6 -/
def square_side : ℝ := 6

/-- Point P on side AB -/
def P : ℝ × ℝ := (3, square_side)

/-- Point D at origin -/
def D : ℝ × ℝ := (0, 0)

/-- Radius of circle centered at P -/
def r_P : ℝ := 3

/-- Radius of circle centered at D -/
def r_D : ℝ := 5

/-- Definition of circle centered at P -/
def circle_P (x y : ℝ) : Prop :=
  (x - P.1)^2 + (y - P.2)^2 = r_P^2

/-- Definition of circle centered at D -/
def circle_D (x y : ℝ) : Prop :=
  x^2 + y^2 = r_D^2

/-- Theorem stating the existence of intersection point Q and its distance from BC -/
theorem intersection_point_exists : ∃ Q : ℝ × ℝ,
  circle_P Q.1 Q.2 ∧ circle_D Q.1 Q.2 ∧ 
  (∃ d : ℝ, d = Q.2 ∧ d ≥ 0 ∧ d ≤ square_side) :=
sorry

end intersection_point_exists_l803_80349


namespace negation_of_existence_is_forall_not_l803_80389

theorem negation_of_existence_is_forall_not :
  (¬ ∃ x : ℚ, x^2 - 2 = 0) ↔ (∀ x : ℚ, x^2 - 2 ≠ 0) := by sorry

end negation_of_existence_is_forall_not_l803_80389


namespace tangent_lines_intersection_l803_80306

/-- Given a circle and four tangent points, proves that the diagonals of the
    trapezoid formed by the tangent lines intersect on the y-axis and that
    the line connecting two specific tangent points passes through this
    intersection point. -/
theorem tangent_lines_intersection
  (ξ η : ℝ)
  (h_ξ_pos : 0 < ξ)
  (h_ξ_lt_1 : ξ < 1)
  (h_circle_eq : ξ^2 + η^2 = 1) :
  ∃ y : ℝ,
    (∀ x : ℝ, x ≠ 0 →
      (y = -((2 * ξ) / (1 + η + ξ)) * x + (1 - η - ξ) / (1 + η + ξ) ↔
       y = ((2 * ξ) / (1 - η + ξ)) * x + (1 + η - ξ) / (1 - η + ξ))) ∧
    y = η / (ξ + 1) :=
sorry

end tangent_lines_intersection_l803_80306


namespace collins_savings_l803_80364

/-- The amount earned per aluminum can in dollars -/
def earnings_per_can : ℚ := 25 / 100

/-- The number of cans found at home -/
def cans_at_home : ℕ := 12

/-- The number of cans found at grandparents' house -/
def cans_at_grandparents : ℕ := 3 * cans_at_home

/-- The number of cans given by the neighbor -/
def cans_from_neighbor : ℕ := 46

/-- The number of cans brought by dad from the office -/
def cans_from_dad : ℕ := 250

/-- The total number of cans collected -/
def total_cans : ℕ := cans_at_home + cans_at_grandparents + cans_from_neighbor + cans_from_dad

/-- The total earnings from recycling all cans -/
def total_earnings : ℚ := earnings_per_can * total_cans

/-- The amount Collin needs to put into savings -/
def savings_amount : ℚ := total_earnings / 2

/-- Theorem stating that the amount Collin needs to put into savings is $43.00 -/
theorem collins_savings : savings_amount = 43 := by sorry

end collins_savings_l803_80364


namespace regular_triangular_pyramid_volume_l803_80326

/-- The volume of a regular triangular pyramid -/
theorem regular_triangular_pyramid_volume 
  (l : ℝ) (α : ℝ) (h_l : l > 0) (h_α : 0 < α ∧ α < π / 2) :
  let volume := (l^3 * Real.sqrt 3 * Real.sin (2 * α) * Real.cos α) / 8
  ∃ (V : ℝ), V = volume ∧ V > 0 := by
  sorry

end regular_triangular_pyramid_volume_l803_80326


namespace marble_ratio_l803_80399

/-- Represents the number of marbles each person has -/
structure Marbles where
  you : ℕ
  brother : ℕ
  friend : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : Marbles) : Prop :=
  m.you = 16 ∧
  m.you + m.brother + m.friend = 63 ∧
  m.you - 2 = 2 * (m.brother + 2) ∧
  ∃ k : ℕ, m.friend = k * m.you

/-- The theorem to prove -/
theorem marble_ratio (m : Marbles) (h : marble_problem m) :
  m.friend * 8 = m.you * 21 := by
  sorry


end marble_ratio_l803_80399


namespace number_difference_l803_80304

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 21352)
  (b_div_9 : ∃ k, b = 9 * k)
  (relation : 10 * a + 1 = b) : 
  b - a = 17470 := by sorry

end number_difference_l803_80304


namespace committee_probability_l803_80312

def total_members : ℕ := 30
def num_boys : ℕ := 12
def num_girls : ℕ := 18
def committee_size : ℕ := 6

def probability_at_least_two_of_each : ℚ :=
  1 - (Nat.choose num_girls committee_size +
       num_boys * Nat.choose num_girls (committee_size - 1) +
       Nat.choose num_boys committee_size +
       num_girls * Nat.choose num_boys (committee_size - 1)) /
      Nat.choose total_members committee_size

theorem committee_probability :
  probability_at_least_two_of_each = 457215 / 593775 :=
by sorry

end committee_probability_l803_80312


namespace certain_number_problem_l803_80376

theorem certain_number_problem : ∃ x : ℝ, (0.45 * x = 0.35 * 40 + 13) ∧ (x = 60) := by
  sorry

end certain_number_problem_l803_80376


namespace linear_function_value_l803_80361

/-- A function that is linear in both arguments -/
def LinearInBoth (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ y₁ y₂ a b : ℝ, 
    f (a*x₁ + b*x₂) y₁ = a*(f x₁ y₁) + b*(f x₂ y₁) ∧
    f x₁ (a*y₁ + b*y₂) = a*(f x₁ y₁) + b*(f x₁ y₂)

/-- The main theorem -/
theorem linear_function_value (f : ℝ → ℝ → ℝ) 
  (h_linear : LinearInBoth f)
  (h_3_3 : f 3 3 = 1/(3*3))
  (h_3_4 : f 3 4 = 1/(3*4))
  (h_4_3 : f 4 3 = 1/(4*3))
  (h_4_4 : f 4 4 = 1/(4*4)) :
  f 5 5 = 1/36 := by
  sorry

end linear_function_value_l803_80361


namespace add_decimals_l803_80381

theorem add_decimals : 5.47 + 4.26 = 9.73 := by
  sorry

end add_decimals_l803_80381


namespace minimum_value_implies_c_l803_80378

def f (c : ℝ) (x : ℝ) : ℝ := x^4 - 8*x^2 + c

theorem minimum_value_implies_c (c : ℝ) :
  (∃ x₀ ∈ Set.Icc (-1) 3, f c x₀ = -14 ∧ ∀ x ∈ Set.Icc (-1) 3, f c x ≥ -14) →
  c = 2 := by
  sorry

end minimum_value_implies_c_l803_80378


namespace lines_parallel_to_same_line_are_parallel_l803_80338

-- Define a type for lines in space
variable (Line : Type)

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Axiom: If two lines are parallel to the same line, they are parallel to each other
axiom parallel_transitivity :
  ∀ (l1 l2 l3 : Line), parallel l1 l3 → parallel l2 l3 → parallel l1 l2

-- Theorem: Two lines parallel to the same line are parallel to each other
theorem lines_parallel_to_same_line_are_parallel
  (l1 l2 l3 : Line) (h1 : parallel l1 l3) (h2 : parallel l2 l3) :
  parallel l1 l2 :=
sorry

end lines_parallel_to_same_line_are_parallel_l803_80338


namespace william_window_wash_time_l803_80305

/-- The time William spends washing vehicles -/
def william_car_wash (window_time : ℕ) : Prop :=
  let normal_car_time := window_time + 7 + 4 + 9
  let suv_time := 2 * normal_car_time
  let total_time := 2 * normal_car_time + suv_time
  total_time = 96

theorem william_window_wash_time :
  ∃ (w : ℕ), william_car_wash w ∧ w = 4 := by
  sorry

end william_window_wash_time_l803_80305


namespace hyperbola_focus_on_y_axis_range_l803_80377

/-- Represents the equation (m+1)x^2 + (2-m)y^2 = 1 -/
def hyperbola_equation (m x y : ℝ) : Prop :=
  (m + 1) * x^2 + (2 - m) * y^2 = 1

/-- Condition for the equation to represent a hyperbola with focus on y-axis -/
def is_hyperbola_on_y_axis (m : ℝ) : Prop :=
  m + 1 < 0 ∧ 2 - m > 0

/-- The theorem stating the range of m for which the equation represents
    a hyperbola with focus on the y-axis -/
theorem hyperbola_focus_on_y_axis_range :
  ∀ m : ℝ, is_hyperbola_on_y_axis m ↔ m < -1 :=
by sorry

end hyperbola_focus_on_y_axis_range_l803_80377


namespace floor_sqrt_24_squared_l803_80350

theorem floor_sqrt_24_squared : ⌊Real.sqrt 24⌋^2 = 16 := by
  sorry

end floor_sqrt_24_squared_l803_80350


namespace sum_of_geometric_sequences_indeterminate_l803_80395

/-- A sequence is geometric if there exists a non-zero constant r such that each term is r times the previous term. -/
def IsGeometricSequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, s (n + 1) = r * s n

/-- The sum of two sequences -/
def SequenceSum (s t : ℕ → ℝ) : ℕ → ℝ :=
  λ n => s n + t n

/-- Statement: Given two geometric sequences, their sum sequence may or may not be geometric or arithmetic. -/
theorem sum_of_geometric_sequences_indeterminate (a b : ℕ → ℝ)
    (ha : IsGeometricSequence a) (hb : IsGeometricSequence b) :
    ¬ (∀ a b : ℕ → ℝ, IsGeometricSequence a → IsGeometricSequence b →
      (IsGeometricSequence (SequenceSum a b) ∨
       ∃ d : ℝ, ∀ n : ℕ, SequenceSum a b (n + 1) = SequenceSum a b n + d)) :=
by sorry

end sum_of_geometric_sequences_indeterminate_l803_80395


namespace johns_house_wall_planks_l803_80365

/-- The number of planks needed for a house wall --/
def total_planks (large_planks small_planks : ℕ) : ℕ :=
  large_planks + small_planks

/-- Theorem stating the total number of planks needed for John's house wall --/
theorem johns_house_wall_planks :
  total_planks 37 42 = 79 := by
  sorry

end johns_house_wall_planks_l803_80365


namespace parabola_intersection_area_l803_80394

/-- Parabola represented by y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- Line passing through the focus at a 45° angle -/
def Line (x y : ℝ) : Prop := y = x - 1

/-- Intersection points of the line with the parabola -/
def IntersectionPoints (A B : ℝ × ℝ) : Prop :=
  A ∈ Parabola ∧ B ∈ Parabola ∧ Line A.1 A.2 ∧ Line B.1 B.2

/-- Origin point -/
def Origin : ℝ × ℝ := (0, 0)

/-- Area of a triangle given three points -/
def TriangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem parabola_intersection_area :
  ∀ A B : ℝ × ℝ, IntersectionPoints A B →
  TriangleArea Origin A B = 2 * Real.sqrt 2 := by
  sorry

end parabola_intersection_area_l803_80394


namespace polynomial_simplification_l803_80384

theorem polynomial_simplification (x : ℝ) :
  (2 * x^2 + 5 * x - 3) - (2 * x^2 + 9 * x - 6) = -4 * x + 3 := by
  sorry

end polynomial_simplification_l803_80384


namespace remaining_money_after_gifts_l803_80333

def initial_budget : ℚ := 999
def shoes_cost : ℚ := 165
def yoga_mat_cost : ℚ := 85
def sports_watch_cost : ℚ := 215
def hand_weights_cost : ℚ := 60

theorem remaining_money_after_gifts :
  initial_budget - (shoes_cost + yoga_mat_cost + sports_watch_cost + hand_weights_cost) = 474 := by
  sorry

end remaining_money_after_gifts_l803_80333


namespace marys_birthday_money_l803_80397

theorem marys_birthday_money (M : ℚ) : 
  (3/4 : ℚ) * M - (1/5 : ℚ) * ((3/4 : ℚ) * M) = 60 → M = 100 := by
  sorry

end marys_birthday_money_l803_80397


namespace first_rectangle_height_l803_80307

/-- Proves that the height of the first rectangle is 5 inches -/
theorem first_rectangle_height : 
  ∀ (h : ℝ), -- height of the first rectangle
  (4 * h = 3 * 6 + 2) → -- area of first = area of second + 2
  h = 5 := by
  sorry

end first_rectangle_height_l803_80307


namespace yoki_cans_count_l803_80391

/-- Given a scenario where:
  - The total number of cans collected is 85
  - LaDonna picked up 25 cans
  - Prikya picked up twice as many cans as LaDonna
  - Yoki picked up the rest of the cans
This theorem proves that Yoki picked up 10 cans. -/
theorem yoki_cans_count (total : ℕ) (ladonna : ℕ) (prikya : ℕ) (yoki : ℕ) 
  (h1 : total = 85)
  (h2 : ladonna = 25)
  (h3 : prikya = 2 * ladonna)
  (h4 : total = ladonna + prikya + yoki) :
  yoki = 10 := by
  sorry

end yoki_cans_count_l803_80391


namespace chips_probability_and_count_l803_80348

def total_bags : ℕ := 9
def bbq_bags : ℕ := 5

def prob_three_bbq : ℚ := 10 / 84

theorem chips_probability_and_count :
  (total_bags = 9) →
  (bbq_bags = 5) →
  (prob_three_bbq = 10 / 84) →
  (Nat.choose bbq_bags 3 * Nat.choose (total_bags - bbq_bags) 0) / Nat.choose total_bags 3 = prob_three_bbq ∧
  total_bags - bbq_bags = 4 := by
  sorry

end chips_probability_and_count_l803_80348


namespace equation_solution_l803_80351

theorem equation_solution : ∃! x : ℝ, x + (x + 1) + (x + 2) + (x + 3) = 18 ∧ x = 3 := by
  sorry

end equation_solution_l803_80351


namespace julio_lost_fish_l803_80317

/-- Proves that Julio lost 15 fish given the fishing conditions -/
theorem julio_lost_fish (fish_per_hour : ℕ) (fishing_hours : ℕ) (final_fish_count : ℕ) : 
  fish_per_hour = 7 →
  fishing_hours = 9 →
  final_fish_count = 48 →
  fish_per_hour * fishing_hours - final_fish_count = 15 := by
sorry

end julio_lost_fish_l803_80317


namespace prob_limit_theorem_l803_80386

/-- The probability that every boy chooses a different number than every girl
    when n boys and n girls choose numbers uniformly from {1, 2, 3, 4, 5} -/
def p (n : ℕ) : ℝ := sorry

/-- The limit of the nth root of p_n as n approaches infinity -/
def limit_p : ℝ := sorry

theorem prob_limit_theorem : 
  limit_p = 6 / 25 := by sorry

end prob_limit_theorem_l803_80386


namespace power_23_mod_25_l803_80308

theorem power_23_mod_25 : 23^2057 % 25 = 16 := by sorry

end power_23_mod_25_l803_80308


namespace midpoint_locus_l803_80331

/-- Given a circle C with center (3,6) and radius 2√5, and a fixed point Q(-3,-6),
    the locus of the midpoint M of any point P on C and Q is described by the equation x^2 + y^2 = 5. -/
theorem midpoint_locus (P : ℝ × ℝ) (M : ℝ × ℝ) :
  (P.1 - 3)^2 + (P.2 - 6)^2 = 20 →
  M.1 = (P.1 + (-3)) / 2 →
  M.2 = (P.2 + (-6)) / 2 →
  M.1^2 + M.2^2 = 5 :=
by sorry

end midpoint_locus_l803_80331


namespace root_implies_coefficients_l803_80352

theorem root_implies_coefficients (p q : ℝ) : 
  (2 * (Complex.I * 2 - 3)^2 + p * (Complex.I * 2 - 3) + q = 0) → 
  (p = 12 ∧ q = 26) := by
  sorry

end root_implies_coefficients_l803_80352


namespace sum_of_fractions_integer_l803_80379

theorem sum_of_fractions_integer (n : ℕ+) :
  (1/2 + 1/3 + 1/5 + 1/n.val : ℚ).isInt → n = 30 := by
  sorry

end sum_of_fractions_integer_l803_80379


namespace triangle_area_l803_80392

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 1 ∧ 
  t.b = Real.sqrt 3 ∧ 
  t.A + t.C = 2 * t.B

-- Theorem statement
theorem triangle_area (t : Triangle) 
  (h : triangle_conditions t) : 
  (1/2 : Real) * t.a * t.c * Real.sin t.B = Real.sqrt 3 / 2 := by
  sorry


end triangle_area_l803_80392


namespace composition_difference_l803_80325

/-- Given two functions f and g, prove that their composition difference
    equals a specific polynomial. -/
theorem composition_difference (x : ℝ) : 
  let f (x : ℝ) := 3 * x^2 + 4 * x - 5
  let g (x : ℝ) := 2 * x + 1
  (f (g x) - g (f x)) = 6 * x^2 + 12 * x + 11 := by
  sorry

end composition_difference_l803_80325


namespace smallest_factor_for_perfect_square_l803_80301

theorem smallest_factor_for_perfect_square (n : ℕ) (h : n = 31360) : 
  (∃ (y : ℕ), y > 0 ∧ ∃ (k : ℕ), n * y = k^2) ∧ 
  (∀ (z : ℕ), z > 0 → z < 623 → ¬∃ (k : ℕ), n * z = k^2) ∧
  (∃ (k : ℕ), n * 623 = k^2) := by
sorry

end smallest_factor_for_perfect_square_l803_80301


namespace laptop_price_l803_80387

theorem laptop_price (upfront_percentage : ℚ) (upfront_payment : ℚ) :
  upfront_percentage = 20 / 100 →
  upfront_payment = 240 →
  upfront_percentage * 1200 = upfront_payment :=
by sorry

end laptop_price_l803_80387


namespace clock_strike_times_l803_80324

/-- Represents the time taken for a given number of clock strikes -/
def strike_time (n : ℕ) : ℚ :=
  (n - 1) * (10 : ℚ) / 9

/-- The clock takes 10 seconds to strike 10 times at 10:00 o'clock -/
axiom ten_strikes_time : strike_time 10 = 10

/-- The strikes are uniformly spaced -/
axiom uniform_strikes : ∀ (n m : ℕ), n > 0 → m > 0 → 
  strike_time n / (n - 1) = strike_time m / (m - 1)

theorem clock_strike_times :
  strike_time 8 = 70 / 9 ∧ strike_time 15 = 140 / 9 := by
  sorry

end clock_strike_times_l803_80324


namespace alcohol_concentration_l803_80373

theorem alcohol_concentration (original_volume : ℝ) (added_water : ℝ) (final_concentration : ℝ) :
  original_volume = 9 →
  added_water = 3 →
  final_concentration = 42.75 →
  (original_volume * (57 / 100)) = ((original_volume + added_water) * (final_concentration / 100)) :=
by sorry

end alcohol_concentration_l803_80373


namespace monkey_travel_distance_l803_80318

/-- Represents the speed and time of movement for a monkey --/
structure MonkeyMovement where
  swingingSpeed : ℝ
  runningSpeed : ℝ
  runningTime : ℝ
  swingingTime : ℝ

/-- Calculates the total distance traveled by the monkey --/
def totalDistance (m : MonkeyMovement) : ℝ :=
  m.runningSpeed * m.runningTime + m.swingingSpeed * m.swingingTime

/-- Theorem stating the total distance traveled by the monkey --/
theorem monkey_travel_distance :
  ∀ (m : MonkeyMovement),
  m.swingingSpeed = 10 ∧
  m.runningSpeed = 15 ∧
  m.runningTime = 5 ∧
  m.swingingTime = 10 →
  totalDistance m = 175 := by
  sorry

end monkey_travel_distance_l803_80318


namespace count_parallelepipeds_l803_80367

/-- The number of parallelepipeds formed in a rectangular parallelepiped -/
def num_parallelepipeds (m n k : ℕ) : ℚ :=
  (m * n * k * (m + 1) * (n + 1) * (k + 1) : ℚ) / 8

/-- Theorem: The number of parallelepipeds formed in a rectangular parallelepiped
    with dimensions m × n × k, divided into unit cubes, is equal to
    (m * n * k * (m+1) * (n+1) * (k+1)) / 8 -/
theorem count_parallelepipeds (m n k : ℕ) :
  num_parallelepipeds m n k = (m * n * k * (m + 1) * (n + 1) * (k + 1) : ℚ) / 8 :=
by sorry

end count_parallelepipeds_l803_80367


namespace max_r_value_l803_80329

theorem max_r_value (r : ℕ) (m n : ℕ → ℤ) 
  (h1 : r ≥ 2)
  (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ r → |m i * n j - m j * n i| = 1) :
  r ≤ 3 :=
sorry

end max_r_value_l803_80329


namespace chicken_surprise_servings_l803_80337

/-- Calculates the number of servings for Chicken Surprise recipe -/
theorem chicken_surprise_servings 
  (chicken_pounds : ℝ) 
  (stuffing_ounces : ℝ) 
  (serving_size_ounces : ℝ) : 
  chicken_pounds = 4.5 ∧ 
  stuffing_ounces = 24 ∧ 
  serving_size_ounces = 8 → 
  (chicken_pounds * 16 + stuffing_ounces) / serving_size_ounces = 12 := by
sorry


end chicken_surprise_servings_l803_80337


namespace petyas_friends_count_l803_80353

/-- The number of friends Petya has -/
def num_friends : ℕ := 19

/-- The number of stickers Petya has -/
def total_stickers : ℕ := num_friends * 5 + 8

theorem petyas_friends_count :
  (num_friends * 5 + 8 = total_stickers) ∧
  (num_friends * 6 = total_stickers + 11) :=
by sorry

end petyas_friends_count_l803_80353


namespace students_not_in_biology_l803_80320

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 880)
  (h2 : biology_percentage = 325 / 1000) :
  total_students - (total_students * biology_percentage).floor = 594 := by
  sorry

end students_not_in_biology_l803_80320


namespace successive_discounts_theorem_l803_80371

/-- The original price of the gadget -/
def original_price : ℝ := 350.00

/-- The first discount rate -/
def first_discount : ℝ := 0.10

/-- The second discount rate -/
def second_discount : ℝ := 0.12

/-- The final sale price as a percentage of the original price -/
def final_sale_percentage : ℝ := 0.792

theorem successive_discounts_theorem :
  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  (final_price / original_price) = final_sale_percentage := by sorry

end successive_discounts_theorem_l803_80371


namespace roots_sum_reciprocal_l803_80327

theorem roots_sum_reciprocal (x₁ x₂ : ℝ) : 
  (5 * x₁^2 - 3 * x₁ - 2 = 0) → 
  (5 * x₂^2 - 3 * x₂ - 2 = 0) → 
  x₁ ≠ x₂ →
  (1 / x₁ + 1 / x₂ = -3 / 2) :=
by sorry

end roots_sum_reciprocal_l803_80327


namespace orphanage_flowers_l803_80396

theorem orphanage_flowers (flower_types : ℕ) (flowers_per_type : ℕ) : 
  flower_types = 4 → flowers_per_type = 40 → flower_types * flowers_per_type = 160 := by
  sorry

end orphanage_flowers_l803_80396


namespace cylinder_cone_base_radii_equal_l803_80302

/-- Given a cylinder and a cone with the same height and base radius, 
    if the ratio of their volumes is 3, then their base radii are equal -/
theorem cylinder_cone_base_radii_equal 
  (h : ℝ) -- height of both cylinder and cone
  (r_cylinder : ℝ) -- radius of cylinder base
  (r_cone : ℝ) -- radius of cone base
  (h_positive : h > 0)
  (r_cylinder_positive : r_cylinder > 0)
  (r_cone_positive : r_cone > 0)
  (same_radius : r_cylinder = r_cone)
  (volume_ratio : π * r_cylinder^2 * h / ((1/3) * π * r_cone^2 * h) = 3) :
  r_cylinder = r_cone :=
sorry

end cylinder_cone_base_radii_equal_l803_80302


namespace number_multiplied_by_7000_l803_80383

theorem number_multiplied_by_7000 : ∃ x : ℝ, x * 7000 = (28000 : ℝ) * (100 : ℝ)^1 ∧ x = 400 := by
  sorry

end number_multiplied_by_7000_l803_80383


namespace smallest_k_for_four_color_rectangle_l803_80319

/-- Represents a coloring of an n × n board -/
def Coloring (n : ℕ) (k : ℕ) := Fin n → Fin n → Fin k

/-- Predicate that checks if four cells form a rectangle with different colors -/
def hasFourColorRectangle (n : ℕ) (k : ℕ) (c : Coloring n k) : Prop :=
  ∃ (r1 r2 c1 c2 : Fin n), r1 ≠ r2 ∧ c1 ≠ c2 ∧
    c r1 c1 ≠ c r1 c2 ∧ c r1 c1 ≠ c r2 c1 ∧ c r1 c1 ≠ c r2 c2 ∧
    c r1 c2 ≠ c r2 c1 ∧ c r1 c2 ≠ c r2 c2 ∧
    c r2 c1 ≠ c r2 c2

/-- Main theorem stating the smallest k that guarantees a four-color rectangle -/
theorem smallest_k_for_four_color_rectangle (n : ℕ) (h : n ≥ 2) :
  (∀ k : ℕ, k ≥ 2*n → ∀ c : Coloring n k, hasFourColorRectangle n k c) ∧
  (∃ c : Coloring n (2*n - 1), ¬hasFourColorRectangle n (2*n - 1) c) :=
sorry

end smallest_k_for_four_color_rectangle_l803_80319
