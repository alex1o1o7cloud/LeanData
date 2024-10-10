import Mathlib

namespace conveyance_percentage_l2787_278794

def salary : ℝ := 5000
def food_percent : ℝ := 40
def rent_percent : ℝ := 20
def entertainment_percent : ℝ := 10
def savings : ℝ := 1000

theorem conveyance_percentage :
  let other_expenses := (food_percent + rent_percent + entertainment_percent) / 100 * salary
  let conveyance := salary - savings - other_expenses
  conveyance / salary * 100 = 10 := by sorry

end conveyance_percentage_l2787_278794


namespace inequality_solution_l2787_278727

theorem inequality_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (x + 1) / (x - 2) + (x - 3) / (3 * x) ≥ 2 ↔ x ≤ -1/3 ∨ x ≥ 3 := by
  sorry

end inequality_solution_l2787_278727


namespace selection_schemes_count_l2787_278716

/-- The number of boys in the selection pool -/
def num_boys : ℕ := 4

/-- The number of girls in the selection pool -/
def num_girls : ℕ := 3

/-- The total number of volunteers to be selected -/
def num_volunteers : ℕ := 4

/-- Function to calculate the number of ways to select volunteers -/
def selection_schemes : ℕ := sorry

/-- Theorem stating that the number of selection schemes is 25 -/
theorem selection_schemes_count : selection_schemes = 25 := by sorry

end selection_schemes_count_l2787_278716


namespace x4_plus_y4_l2787_278719

theorem x4_plus_y4 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : x^4 + y^4 = 175 := by
  sorry

end x4_plus_y4_l2787_278719


namespace work_completion_time_l2787_278743

/-- Given workers A, B, and C, where:
    - A can complete the work in 6 days
    - C can complete the work in 7.5 days
    - A, B, and C together complete the work in 2 days
    Prove that B can complete the work alone in 5 days -/
theorem work_completion_time (A B C : ℝ) 
  (hA : A = 1 / 6)  -- A's work rate per day
  (hC : C = 1 / 7.5)  -- C's work rate per day
  (hABC : A + B + C = 1 / 2)  -- Combined work rate of A, B, and C
  : B = 1 / 5 := by  -- B's work rate per day
sorry

end work_completion_time_l2787_278743


namespace computer_contract_probability_l2787_278791

theorem computer_contract_probability 
  (p_hardware : ℝ) 
  (p_not_software : ℝ) 
  (p_at_least_one : ℝ) 
  (h1 : p_hardware = 4/5)
  (h2 : p_not_software = 3/5)
  (h3 : p_at_least_one = 9/10) :
  p_hardware + (1 - p_not_software) - p_at_least_one = 7/10 := by
sorry

end computer_contract_probability_l2787_278791


namespace matrix_is_own_inverse_l2787_278753

def A (x y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; x, y]

theorem matrix_is_own_inverse (x y : ℝ) :
  A x y * A x y = 1 ↔ x = 15/2 ∧ y = -4 := by
  sorry

end matrix_is_own_inverse_l2787_278753


namespace sheepdog_catch_time_l2787_278710

theorem sheepdog_catch_time (sheep_speed dog_speed initial_distance : ℝ) 
  (h1 : sheep_speed = 16)
  (h2 : dog_speed = 28)
  (h3 : initial_distance = 240) : 
  initial_distance / (dog_speed - sheep_speed) = 20 := by
  sorry

#check sheepdog_catch_time

end sheepdog_catch_time_l2787_278710


namespace min_value_expression_l2787_278790

open Real

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (min_val : ℝ), min_val = Real.sqrt 10 / 5 ∧
  ∀ (x y : ℝ) (hx : x > 0) (hy : y > 0),
    (abs (x + 3*y - y*(x + 9*y)) + abs (3*y - x + 3*y*(x - y))) / Real.sqrt (x^2 + 9*y^2) ≥ min_val :=
by sorry

end min_value_expression_l2787_278790


namespace modulus_of_complex_number_l2787_278767

theorem modulus_of_complex_number (z : ℂ) (h : z = 3 + 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end modulus_of_complex_number_l2787_278767


namespace equation_solution_l2787_278742

theorem equation_solution :
  ∃! x : ℚ, (x ≠ 3 ∧ x ≠ -2) ∧ (3 - x) / (x + 2) + (3*x - 9) / (3 - x) = 2 :=
by
  use (-7/6)
  sorry

end equation_solution_l2787_278742


namespace machine_fill_time_l2787_278769

theorem machine_fill_time (time_A time_AB : ℝ) (time_A_pos : time_A > 0) (time_AB_pos : time_AB > 0) :
  time_A = 20 → time_AB = 12 → ∃ time_B : ℝ, time_B > 0 ∧ 1 / time_A + 1 / time_B = 1 / time_AB ∧ time_B = 30 :=
by sorry

end machine_fill_time_l2787_278769


namespace soccer_team_selection_l2787_278745

theorem soccer_team_selection (n m k : ℕ) (h1 : n = 18) (h2 : m = 7) (h3 : k = 2) :
  (Nat.choose (n - k) (m - k)) = (Nat.choose 16 5) :=
by sorry

end soccer_team_selection_l2787_278745


namespace cosine_square_root_pi_eighths_l2787_278782

theorem cosine_square_root_pi_eighths :
  Real.sqrt ((3 - Real.cos (π / 8) ^ 2) * (3 - Real.cos (3 * π / 8) ^ 2)) = 3 * Real.sqrt 5 / 4 := by
  sorry

end cosine_square_root_pi_eighths_l2787_278782


namespace average_marks_chemistry_mathematics_l2787_278726

/-- Given that the total marks in physics, chemistry, and mathematics is 180 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 90. -/
theorem average_marks_chemistry_mathematics 
  (P C M : ℕ) -- P: marks in physics, C: marks in chemistry, M: marks in mathematics
  (h : P + C + M = P + 180) -- Given condition
  : (C + M) / 2 = 90 := by
  sorry

end average_marks_chemistry_mathematics_l2787_278726


namespace train_distance_problem_l2787_278796

/-- Given two trains with specified lengths, speeds, and crossing time, 
    calculate the initial distance between them. -/
theorem train_distance_problem (length1 length2 speed1 speed2 crossing_time : ℝ) 
  (h1 : length1 = 100)
  (h2 : length2 = 150)
  (h3 : speed1 = 10)
  (h4 : speed2 = 15)
  (h5 : crossing_time = 60)
  (h6 : speed2 > speed1) : 
  (speed2 - speed1) * crossing_time = length1 + length2 + 50 := by
  sorry

end train_distance_problem_l2787_278796


namespace min_m_for_24m_eq_n4_l2787_278705

theorem min_m_for_24m_eq_n4 (m n : ℕ+) (h : 24 * m = n ^ 4) :
  ∀ k : ℕ+, 24 * k = (k : ℕ+) ^ 4 → m ≤ k :=
sorry

end min_m_for_24m_eq_n4_l2787_278705


namespace emily_contribution_l2787_278757

/-- Proves that Emily needs to contribute 3 more euros to buy the pie -/
theorem emily_contribution (pie_cost : ℝ) (emily_usd : ℝ) (berengere_euro : ℝ) (exchange_rate : ℝ) :
  pie_cost = 15 →
  emily_usd = 10 →
  berengere_euro = 3 →
  exchange_rate = 1.1 →
  ∃ (emily_extra : ℝ), emily_extra = 3 ∧ 
    pie_cost = berengere_euro + (emily_usd / exchange_rate) + emily_extra :=
by sorry

end emily_contribution_l2787_278757


namespace union_of_sets_l2787_278761

theorem union_of_sets : 
  let A : Set ℕ := {2, 5, 6}
  let B : Set ℕ := {3, 5}
  A ∪ B = {2, 3, 5, 6} := by
sorry

end union_of_sets_l2787_278761


namespace rocket_coaster_capacity_l2787_278736

/-- Represents a roller coaster with two types of cars -/
structure RollerCoaster where
  total_cars : ℕ
  four_passenger_cars : ℕ
  six_passenger_cars : ℕ

/-- Calculates the total capacity of a roller coaster -/
def total_capacity (rc : RollerCoaster) : ℕ :=
  rc.four_passenger_cars * 4 + rc.six_passenger_cars * 6

/-- The Rocket Coaster specification -/
def rocket_coaster : RollerCoaster := {
  total_cars := 15,
  four_passenger_cars := 9,
  six_passenger_cars := 15 - 9
}

theorem rocket_coaster_capacity :
  total_capacity rocket_coaster = 72 := by
  sorry

end rocket_coaster_capacity_l2787_278736


namespace minimum_interval_for_f_l2787_278720

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem minimum_interval_for_f (t s : ℝ) (h : f t = g s) :
  ∃ (a : ℝ), (a > (1/2) ∧ a < Real.log 2) ∧
  (∀ (t' s' : ℝ), f t' = g s' → s' - t' ≥ s - t → f t = a) :=
sorry

end minimum_interval_for_f_l2787_278720


namespace angle_between_vectors_l2787_278730

/-- Given vectors a and b, if the angle between them is π/6, then the second component of b is √3. -/
theorem angle_between_vectors (a b : ℝ × ℝ) :
  a = (1, Real.sqrt 3) →
  b.1 = 3 →
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = Real.cos (π / 6) →
  b.2 = Real.sqrt 3 := by
  sorry

end angle_between_vectors_l2787_278730


namespace inequality_system_solution_l2787_278703

theorem inequality_system_solution (x : ℝ) : 
  (3 * x + 1) / 2 > x ∧ 4 * (x - 2) ≤ x - 5 → -1 < x ∧ x ≤ 1 := by
  sorry

end inequality_system_solution_l2787_278703


namespace base_difference_l2787_278763

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

/-- The problem statement -/
theorem base_difference : 
  let base_9_number := [1, 2, 3]  -- 321 in base 9, least significant digit first
  let base_6_number := [5, 6, 1]  -- 165 in base 6, least significant digit first
  (to_base_10 base_9_number 9) - (to_base_10 base_6_number 6) = 221 := by
  sorry


end base_difference_l2787_278763


namespace bill_work_hours_l2787_278708

/-- Calculates the total pay for a given number of hours worked, 
    with a base rate for the first 40 hours and a double rate thereafter. -/
def calculatePay (baseRate : ℕ) (hours : ℕ) : ℕ :=
  if hours ≤ 40 then
    baseRate * hours
  else
    baseRate * 40 + baseRate * 2 * (hours - 40)

/-- Proves that working 50 hours results in a total pay of $1200, 
    given the specified pay rates. -/
theorem bill_work_hours (baseRate : ℕ) (totalPay : ℕ) :
  baseRate = 20 → totalPay = 1200 → ∃ hours, calculatePay baseRate hours = totalPay ∧ hours = 50 :=
by
  sorry

end bill_work_hours_l2787_278708


namespace solve_pretzel_problem_l2787_278706

def pretzel_problem (barry_pretzels : ℕ) : Prop :=
  let shelly_pretzels : ℕ := barry_pretzels / 2
  let angie_pretzels : ℕ := 3 * shelly_pretzels
  let dave_pretzels : ℕ := (angie_pretzels + shelly_pretzels) / 4
  let total_pretzels : ℕ := barry_pretzels + shelly_pretzels + angie_pretzels + dave_pretzels
  let price_per_pretzel : ℕ := 1
  let total_cost : ℕ := total_pretzels * price_per_pretzel
  (barry_pretzels = 12) →
  (total_cost = 42)

theorem solve_pretzel_problem :
  pretzel_problem 12 :=
by
  sorry

end solve_pretzel_problem_l2787_278706


namespace simplify_and_rationalize_l2787_278729

theorem simplify_and_rationalize :
  (Real.sqrt 5 / Real.sqrt 7) * (Real.sqrt 9 / Real.sqrt 11) * (Real.sqrt 13 / Real.sqrt 15) = 
  (3 * Real.sqrt 3003) / 231 := by
  sorry

end simplify_and_rationalize_l2787_278729


namespace min_p_minus_q_equals_zero_l2787_278702

theorem min_p_minus_q_equals_zero
  (x y p q : ℤ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (eq1 : (3 : ℚ) / (x * p) = 8)
  (eq2 : (5 : ℚ) / (y * q) = 18)
  (hmin : ∀ x' y' p' q' : ℤ,
    x' ≠ 0 → y' ≠ 0 → p' ≠ 0 → q' ≠ 0 →
    (3 : ℚ) / (x' * p') = 8 →
    (5 : ℚ) / (y' * q') = 18 →
    (x' ≤ x ∧ y' ≤ y)) :
  p - q = 0 := by
sorry

end min_p_minus_q_equals_zero_l2787_278702


namespace product_of_roots_l2787_278740

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 18 → ∃ (x₁ x₂ : ℝ), x₁ * x₂ = -30 ∧ (x₁ + 3) * (x₁ - 4) = 18 ∧ (x₂ + 3) * (x₂ - 4) = 18 := by
  sorry

end product_of_roots_l2787_278740


namespace olivia_wallet_proof_l2787_278773

def initial_wallet_amount (amount_spent : ℕ) (amount_left : ℕ) : ℕ :=
  amount_spent + amount_left

theorem olivia_wallet_proof (amount_spent : ℕ) (amount_left : ℕ) 
  (h1 : amount_spent = 38) (h2 : amount_left = 90) :
  initial_wallet_amount amount_spent amount_left = 128 := by
  sorry

end olivia_wallet_proof_l2787_278773


namespace five_consecutive_integers_product_not_square_l2787_278772

theorem five_consecutive_integers_product_not_square (n : ℕ+) :
  ∃ m : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) : ℕ) ≠ m^2 := by
  sorry

end five_consecutive_integers_product_not_square_l2787_278772


namespace largest_gcd_of_sum_1008_l2787_278784

theorem largest_gcd_of_sum_1008 :
  ∃ (a b : ℕ+), a + b = 1008 ∧ 
  ∀ (c d : ℕ+), c + d = 1008 → Nat.gcd a b ≥ Nat.gcd c d ∧
  Nat.gcd a b = 504 :=
by sorry

end largest_gcd_of_sum_1008_l2787_278784


namespace inequality_proof_l2787_278752

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3*b^2 + 5*c^2 ≤ 1 := by
  sorry

end inequality_proof_l2787_278752


namespace sum_interior_angles_increases_l2787_278737

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: The sum of interior angles increases as the number of sides increases from 3 to n -/
theorem sum_interior_angles_increases (n : ℕ) (h : n > 3) :
  sum_interior_angles n > sum_interior_angles 3 := by
  sorry


end sum_interior_angles_increases_l2787_278737


namespace perfect_square_trinomial_m_value_l2787_278700

-- Define a perfect square trinomial
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (x + k)^2

theorem perfect_square_trinomial_m_value (m : ℝ) :
  is_perfect_square_trinomial 1 (2*m) 9 → m = 3 ∨ m = -3 := by
  sorry

end perfect_square_trinomial_m_value_l2787_278700


namespace triangle_trig_identity_l2787_278731

theorem triangle_trig_identity (A B C : ℝ) 
  (h1 : A + B + C = Real.pi)
  (h2 : (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) = 1) :
  (Real.cos (2*A) + Real.cos (2*B) + Real.cos (2*C)) / (Real.cos A + Real.cos B + Real.cos C) = 2 := by
  sorry

end triangle_trig_identity_l2787_278731


namespace opposite_of_three_l2787_278783

theorem opposite_of_three : -(3 : ℤ) = -3 := by sorry

end opposite_of_three_l2787_278783


namespace quadratic_equation_solution_l2787_278718

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = (1 : ℝ) / 2 ∧ x₂ = 1 ∧ 
  (2 * x₁^2 - 3 * x₁ + 1 = 0) ∧ (2 * x₂^2 - 3 * x₂ + 1 = 0) :=
by
  sorry

end quadratic_equation_solution_l2787_278718


namespace customers_left_l2787_278744

/-- A problem about customers leaving a waiter's section. -/
theorem customers_left (initial_customers : ℕ) (remaining_tables : ℕ) (people_per_table : ℕ) : 
  initial_customers = 22 → remaining_tables = 2 → people_per_table = 4 →
  initial_customers - (remaining_tables * people_per_table) = 14 := by
sorry

end customers_left_l2787_278744


namespace field_completion_time_l2787_278798

theorem field_completion_time (team1_time team2_time initial_days joint_days : ℝ) : 
  team1_time = 12 →
  team2_time = 0.75 * team1_time →
  initial_days = 5 →
  (initial_days / team1_time) + joint_days * (1 / team1_time + 1 / team2_time) = 1 →
  joint_days = 3 := by
  sorry

end field_completion_time_l2787_278798


namespace units_digit_sum_l2787_278732

/-- The base of the number system -/
def base : ℕ := 8

/-- The first number in base 8 -/
def num1 : ℕ := 63

/-- The second number in base 8 -/
def num2 : ℕ := 74

/-- The units digit of the first number -/
def units_digit1 : ℕ := 3

/-- The units digit of the second number -/
def units_digit2 : ℕ := 4

/-- Theorem: The units digit of the sum of num1 and num2 in base 8 is 7 -/
theorem units_digit_sum : (num1 + num2) % base = 7 := by sorry

end units_digit_sum_l2787_278732


namespace inscribed_circle_area_ratio_l2787_278723

/-- The ratio of the area of the inscribed circle to the area of a right triangle -/
theorem inscribed_circle_area_ratio (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let a := (3 / 5) * h
  let b := (4 / 5) * h
  let triangle_area := (1 / 2) * a * b
  let circle_area := π * r^2
  circle_area / triangle_area = (5 * π * r) / (12 * h) := by
  sorry

end inscribed_circle_area_ratio_l2787_278723


namespace right_triangle_side_values_l2787_278786

theorem right_triangle_side_values (a b x : ℝ) : 
  a = 6 → b = 8 → (x^2 = a^2 + b^2 ∨ b^2 = a^2 + x^2) → (x = 10 ∨ x = 2 * Real.sqrt 7) := by
  sorry

end right_triangle_side_values_l2787_278786


namespace rhombus_other_diagonal_length_l2787_278714

-- Define the rhombus properties
def diagonal1 : ℝ := 7.4
def area : ℝ := 21.46

-- Theorem to prove
theorem rhombus_other_diagonal_length :
  let diagonal2 := (2 * area) / diagonal1
  ∃ ε > 0, abs (diagonal2 - 5.8) < ε :=
by sorry

end rhombus_other_diagonal_length_l2787_278714


namespace right_triangle_segment_ratio_l2787_278759

theorem right_triangle_segment_ratio (a b c r s : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ s > 0 →  -- Positive lengths
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  r * s = a^2 →  -- Geometric mean theorem for r
  r * s = b^2 →  -- Geometric mean theorem for s
  r + s = c →  -- r and s form the hypotenuse
  a / b = 1 / 4 →  -- Given ratio
  r / s = 1 / 16 :=
by sorry

end right_triangle_segment_ratio_l2787_278759


namespace sixth_quiz_score_l2787_278750

def existing_scores : List ℕ := [86, 91, 83, 88, 97]
def target_mean : ℕ := 90
def num_quizzes : ℕ := 6

theorem sixth_quiz_score (x : ℕ) :
  (existing_scores.sum + x) / num_quizzes = target_mean ↔ x = 95 := by
  sorry

end sixth_quiz_score_l2787_278750


namespace always_true_inequality_l2787_278747

theorem always_true_inequality (x : ℝ) : x + 2 < x + 3 := by
  sorry

end always_true_inequality_l2787_278747


namespace coupon_value_l2787_278774

theorem coupon_value (total_spent peaches_after_coupon cherries : ℚ) : 
  total_spent = 23.86 →
  peaches_after_coupon = 12.32 →
  cherries = 11.54 →
  total_spent = peaches_after_coupon + cherries →
  0 = total_spent - (peaches_after_coupon + cherries) := by
sorry

end coupon_value_l2787_278774


namespace brady_hours_june_l2787_278741

def hours_per_day_april : ℝ := 6
def hours_per_day_september : ℝ := 8
def average_hours_per_month : ℝ := 190
def days_per_month : ℕ := 30

theorem brady_hours_june :
  ∃ (hours_per_day_june : ℝ),
    hours_per_day_june * days_per_month +
    hours_per_day_april * days_per_month +
    hours_per_day_september * days_per_month =
    average_hours_per_month * 3 ∧
    hours_per_day_june = 5 := by
  sorry

end brady_hours_june_l2787_278741


namespace fourth_roll_max_probability_l2787_278748

-- Define the dice
structure Die :=
  (sides : ℕ)
  (max_prob : ℚ)
  (other_prob : ℚ)

-- Define the three dice
def six_sided_die : Die := ⟨6, 1/6, 1/6⟩
def eight_sided_die : Die := ⟨8, 3/4, 1/28⟩
def ten_sided_die : Die := ⟨10, 4/5, 1/45⟩

-- Define the probability of choosing each die
def choose_prob : ℚ := 1/3

-- Define the event of rolling maximum value three times for a given die
def max_three_times (d : Die) : ℚ := d.max_prob^3

-- Define the total probability of rolling maximum value three times
def total_max_three_times : ℚ :=
  choose_prob * (max_three_times six_sided_die + 
                 max_three_times eight_sided_die + 
                 max_three_times ten_sided_die)

-- Define the conditional probability of using each die given three max rolls
def cond_prob (d : Die) : ℚ :=
  (choose_prob * max_three_times d) / total_max_three_times

-- Define the probability of fourth roll being max given three max rolls
def fourth_max_prob : ℚ :=
  cond_prob six_sided_die * six_sided_die.max_prob +
  cond_prob eight_sided_die * eight_sided_die.max_prob +
  cond_prob ten_sided_die * ten_sided_die.max_prob

-- The theorem to prove
theorem fourth_roll_max_probability : 
  fourth_max_prob = 1443 / 2943 := by
  sorry

end fourth_roll_max_probability_l2787_278748


namespace sin_shift_l2787_278795

theorem sin_shift (x : ℝ) : 
  Real.sin (2 * x + π / 3) = Real.sin (2 * (x + π / 6)) := by
  sorry

end sin_shift_l2787_278795


namespace right_triangle_side_length_l2787_278749

theorem right_triangle_side_length 
  (P Q R : ℝ × ℝ) 
  (right_angle : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0) 
  (cos_R : Real.cos (Real.arccos ((3 * Real.sqrt 65) / 65)) = (3 * Real.sqrt 65) / 65) 
  (hypotenuse : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = Real.sqrt 169) :
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = (3 * Real.sqrt 65) / 5 := by
  sorry


end right_triangle_side_length_l2787_278749


namespace simplify_sqrt_expression_l2787_278775

theorem simplify_sqrt_expression : 
  Real.sqrt (28 - 12 * Real.sqrt 2) = 6 - Real.sqrt 2 := by
  sorry

end simplify_sqrt_expression_l2787_278775


namespace geometric_sequence_common_ratio_l2787_278715

/-- Given a geometric sequence starting with 25, -50, 100, -200, prove that its common ratio is -2 -/
theorem geometric_sequence_common_ratio :
  let a₁ : ℝ := 25
  let a₂ : ℝ := -50
  let a₃ : ℝ := 100
  let a₄ : ℝ := -200
  ∀ r : ℝ, (a₂ = r * a₁ ∧ a₃ = r * a₂ ∧ a₄ = r * a₃) → r = -2 := by
  sorry

end geometric_sequence_common_ratio_l2787_278715


namespace angle_sum_when_product_is_four_l2787_278760

theorem angle_sum_when_product_is_four (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) 
  (h3 : (1 + Real.tan α) * (1 + Real.tan β) = 4) : α + β = π * 3/4 := by
  sorry

end angle_sum_when_product_is_four_l2787_278760


namespace special_ellipse_equation_l2787_278738

/-- An ellipse with center at the origin, foci at (±√2, 0), intersected by the line y = x + 1
    such that the x-coordinate of the midpoint of the chord is -2/3 -/
def special_ellipse (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (x^2 / a^2 + y^2 / b^2 = 1) ∧
  (a^2 = b^2 + 2) ∧
  (∃ (x₁ x₂ y₁ y₂ : ℝ),
    (x₁^2 / a^2 + y₁^2 / b^2 = 1) ∧
    (x₂^2 / a^2 + y₂^2 / b^2 = 1) ∧
    (y₁ = x₁ + 1) ∧ (y₂ = x₂ + 1) ∧
    ((x₁ + x₂) / 2 = -2/3))

/-- The equation of the special ellipse is x²/4 + y²/2 = 1 -/
theorem special_ellipse_equation :
  ∀ x y : ℝ, special_ellipse x y ↔ x^2/4 + y^2/2 = 1 := by
  sorry

end special_ellipse_equation_l2787_278738


namespace no_three_digit_odd_divisible_by_six_l2787_278771

theorem no_three_digit_odd_divisible_by_six : 
  ¬ ∃ n : ℕ, 
    (100 ≤ n ∧ n ≤ 999) ∧ 
    (∀ d, d ∈ n.digits 10 → d % 2 = 1 ∧ d > 4) ∧ 
    n % 6 = 0 :=
by sorry

end no_three_digit_odd_divisible_by_six_l2787_278771


namespace davids_english_marks_l2787_278739

def davidsMaths : ℕ := 89
def davidsPhysics : ℕ := 82
def davidsChemistry : ℕ := 87
def davidsBiology : ℕ := 81
def averageMarks : ℕ := 85
def numberOfSubjects : ℕ := 5

theorem davids_english_marks :
  ∃ (englishMarks : ℕ),
    (englishMarks + davidsMaths + davidsPhysics + davidsChemistry + davidsBiology) / numberOfSubjects = averageMarks ∧
    englishMarks = 86 := by
  sorry

end davids_english_marks_l2787_278739


namespace constant_function_from_square_plus_k_l2787_278735

/-- A continuous function satisfying f(x) = f(x² + k) for non-negative k is constant. -/
theorem constant_function_from_square_plus_k 
  (f : ℝ → ℝ) (hf : Continuous f) (k : ℝ) (hk : k ≥ 0) 
  (h : ∀ x, f x = f (x^2 + k)) : 
  ∃ C, ∀ x, f x = C :=
sorry

end constant_function_from_square_plus_k_l2787_278735


namespace sphere_volume_hexagonal_prism_l2787_278704

/-- The volume of a sphere circumscribing a hexagonal prism -/
theorem sphere_volume_hexagonal_prism (h : ℝ) (p : ℝ) : 
  h = Real.sqrt 3 →
  p = 3 →
  (4 / 3 * Real.pi : ℝ) = (4 / 3 * Real.pi * (((h^2 + (p / 6)^2) / 4)^(3/2))) :=
by sorry

end sphere_volume_hexagonal_prism_l2787_278704


namespace range_of_a_l2787_278792

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) →
  (a ∈ Set.Icc (-1) 1) ∨ (a > 3) :=
by sorry

end range_of_a_l2787_278792


namespace complex_inequality_nonexistence_l2787_278756

theorem complex_inequality_nonexistence : 
  ∀ (a b c : ℂ) (h : ℕ), a ≠ 0 → b ≠ 0 → c ≠ 0 → 
  ∃ (k l m : ℤ), (abs k + abs l + abs m ≥ 1996) ∧ 
  (Complex.abs (1 + k * a + l * b + m * c) ≤ 1 / h) := by
  sorry

end complex_inequality_nonexistence_l2787_278756


namespace james_total_score_l2787_278793

-- Define the number of field goals and shots
def field_goals : ℕ := 13
def shots : ℕ := 20

-- Define the point values for field goals and shots
def field_goal_points : ℕ := 3
def shot_points : ℕ := 2

-- Define the total points scored
def total_points : ℕ := field_goals * field_goal_points + shots * shot_points

-- Theorem stating that the total points scored is 79
theorem james_total_score : total_points = 79 := by
  sorry

end james_total_score_l2787_278793


namespace impossibleSquareConstruction_l2787_278755

/-- Represents a square constructed on a chord of a unit circle -/
structure SquareOnChord where
  sideLength : ℝ
  twoVerticesOnChord : Bool
  twoVerticesOnCircumference : Bool

/-- Represents a chord of a unit circle -/
structure Chord where
  length : ℝ
  inUnitCircle : length > 0 ∧ length ≤ 2

theorem impossibleSquareConstruction (c : Chord) :
  ¬∃ (s1 s2 : SquareOnChord),
    s1.twoVerticesOnChord ∧
    s1.twoVerticesOnCircumference ∧
    s2.twoVerticesOnChord ∧
    s2.twoVerticesOnCircumference ∧
    s1.sideLength - s2.sideLength = 1 ∧
    s1.sideLength = c.length / Real.sqrt 2 ∧
    s2.sideLength = (c.length - Real.sqrt 2) / Real.sqrt 2 := by
  sorry

end impossibleSquareConstruction_l2787_278755


namespace dice_stack_top_bottom_sum_l2787_278711

/-- Represents a standard die -/
structure StandardDie :=
  (faces : Fin 6 → Nat)
  (opposite_sum : ∀ (f : Fin 6), faces f + faces (5 - f) = 7)

/-- Represents a stack of two standard dice -/
structure DiceStack :=
  (top : StandardDie)
  (bottom : StandardDie)
  (touching_sum : ∃ (f1 f2 : Fin 6), top.faces f1 + bottom.faces f2 = 5)

/-- Theorem: The sum of pips on the top and bottom faces of a dice stack is 9 -/
theorem dice_stack_top_bottom_sum (stack : DiceStack) : 
  ∃ (f1 f2 : Fin 6), stack.top.faces f1 + stack.bottom.faces f2 = 9 :=
sorry

end dice_stack_top_bottom_sum_l2787_278711


namespace evening_ticket_price_l2787_278776

/-- The cost of an evening movie ticket --/
def evening_ticket_cost : ℝ := 10

/-- The cost of a large popcorn & drink combo --/
def combo_cost : ℝ := 10

/-- The discount rate for tickets during the special offer --/
def ticket_discount_rate : ℝ := 0.2

/-- The discount rate for food combos during the special offer --/
def combo_discount_rate : ℝ := 0.5

/-- The amount saved by going to the earlier movie --/
def savings : ℝ := 7

theorem evening_ticket_price :
  evening_ticket_cost = 10 ∧
  combo_cost = 10 ∧
  ticket_discount_rate = 0.2 ∧
  combo_discount_rate = 0.5 ∧
  savings = 7 →
  evening_ticket_cost + combo_cost - 
  (evening_ticket_cost * (1 - ticket_discount_rate) + combo_cost * (1 - combo_discount_rate)) = savings :=
by sorry

end evening_ticket_price_l2787_278776


namespace equation_solutions_l2787_278762

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ > 0 ∧ x₂ > 0) ∧
    (∀ x : ℝ, x > 0 → 
      ((1/2) * (4*x^2 - 1) = (x^2 - 60*x - 20) * (x^2 + 30*x + 10)) ↔ 
      (x = x₁ ∨ x = x₂)) ∧
    x₁ = 30 + Real.sqrt 919 ∧
    x₂ = -15 + Real.sqrt 216 := by
  sorry

end equation_solutions_l2787_278762


namespace expression_evaluation_l2787_278709

theorem expression_evaluation (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (3 * x + y / 3 + 3 * z)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹ + (3 * z)⁻¹) = (9 * x * y * z)⁻¹ := by
  sorry

end expression_evaluation_l2787_278709


namespace floor_ceiling_sum_seven_l2787_278746

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℝ) + (⌈x⌉ : ℝ) = 7 ↔ 3 < x ∧ x < 4 := by sorry

end floor_ceiling_sum_seven_l2787_278746


namespace range_of_x_l2787_278722

theorem range_of_x (x : ℝ) : 4 * x - 12 ≥ 0 → x ≥ 3 := by
  sorry

end range_of_x_l2787_278722


namespace dress_price_inconsistency_l2787_278724

theorem dress_price_inconsistency :
  ¬∃ (D : ℝ), D > 0 ∧ 7 * D + 4 * 5 + 8 * 15 + 6 * 20 = 250 := by
  sorry

end dress_price_inconsistency_l2787_278724


namespace arun_speed_l2787_278785

theorem arun_speed (arun_speed : ℝ) (anil_speed : ℝ) : 
  (30 / arun_speed = 30 / anil_speed + 2) →
  (30 / (2 * arun_speed) = 30 / anil_speed - 1) →
  arun_speed = Real.sqrt 15 := by
  sorry

end arun_speed_l2787_278785


namespace parabola_focus_distance_l2787_278778

/-- Theorem: For a parabola y^2 = 2px (p > 0) with vertex at origin, passing through (x₀, 2),
    if the distance from A to focus is 3 times the distance from origin to focus, then p = √2 -/
theorem parabola_focus_distance (p : ℝ) (x₀ : ℝ) (h_p_pos : p > 0) :
  (2 : ℝ)^2 = 2 * p * x₀ →  -- parabola passes through (x₀, 2)
  x₀ + p / 2 = 3 * (p / 2) →  -- |AF| = 3|OF|
  p = Real.sqrt 2 := by
  sorry

end parabola_focus_distance_l2787_278778


namespace pencil_count_difference_l2787_278764

theorem pencil_count_difference (D J M E : ℕ) : 
  D = J + 15 → 
  J = 2 * M → 
  E = (J - M) / 2 → 
  J = 20 → 
  D - (M + E) = 20 := by
sorry

end pencil_count_difference_l2787_278764


namespace investment_rate_proof_l2787_278799

/-- Proves that the remaining investment rate is 7% given the specified conditions --/
theorem investment_rate_proof (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ)
  (first_rate : ℝ) (second_rate : ℝ) (desired_income : ℝ) :
  total_investment = 12000 →
  first_investment = 5000 →
  second_investment = 4000 →
  first_rate = 0.05 →
  second_rate = 0.035 →
  desired_income = 600 →
  let remaining_investment := total_investment - first_investment - second_investment
  let first_income := first_investment * first_rate
  let second_income := second_investment * second_rate
  let remaining_income := desired_income - first_income - second_income
  remaining_income / remaining_investment = 0.07 :=
by sorry

end investment_rate_proof_l2787_278799


namespace typing_difference_l2787_278797

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Micah's typing speed in words per minute -/
def micah_speed : ℕ := 20

/-- Isaiah's typing speed in words per minute -/
def isaiah_speed : ℕ := 40

/-- The difference in words typed per hour between Isaiah and Micah -/
theorem typing_difference : 
  (isaiah_speed * minutes_per_hour) - (micah_speed * minutes_per_hour) = 1200 := by
  sorry

end typing_difference_l2787_278797


namespace max_pages_copied_l2787_278779

/-- The number of pages that can be copied given a budget and copying costs -/
def pages_copied (cost_per_4_pages : ℕ) (flat_fee : ℕ) (budget : ℕ) : ℕ :=
  ((budget - flat_fee) * 4) / cost_per_4_pages

/-- Theorem stating the maximum number of pages that can be copied under given conditions -/
theorem max_pages_copied : 
  pages_copied 7 100 3000 = 1657 := by
  sorry

end max_pages_copied_l2787_278779


namespace f_increasing_on_interval_l2787_278717

-- Define the function f(x) = (x-1)^2 - 2
def f (x : ℝ) : ℝ := (x - 1)^2 - 2

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x ≤ y → f x ≤ f y :=
by sorry

end f_increasing_on_interval_l2787_278717


namespace ellipse_and_line_equation_l2787_278788

noncomputable section

-- Define the ellipse E
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the hyperbola
def hyperbola (m n : ℝ) (x y : ℝ) : Prop := x^2 / m - y^2 / n = 1

-- Define eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

theorem ellipse_and_line_equation 
  (a b m n e₁ e₂ : ℝ)
  (h₁ : a > b)
  (h₂ : b > 0)
  (h₃ : m > 0)
  (h₄ : n > 0)
  (h₅ : b = Real.sqrt 3)
  (h₆ : n / m = 3)
  (h₇ : e₁ * e₂ = 1)
  (h₈ : e₂ = Real.sqrt 4)
  (h₉ : e₁ = eccentricity a b)
  (P : ℝ × ℝ)
  (h₁₀ : P = (-1, 3/2))
  (S₁ S₂ : ℝ)
  (h₁₁ : S₁ = 6 * S₂) :
  (∀ x y, ellipse a b x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∃ k, k = Real.sqrt 6 / 2 ∧ 
    (∀ x y, y = k * x + 1 ∨ y = -k * x + 1)) :=
by sorry

end

end ellipse_and_line_equation_l2787_278788


namespace initial_money_l2787_278712

theorem initial_money (x : ℝ) : x + 13 + 3 = 18 → x = 2 := by
  sorry

end initial_money_l2787_278712


namespace exactly_one_greater_than_one_l2787_278770

theorem exactly_one_greater_than_one 
  (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (prod_one : a * b * c = 1)
  (sum_greater : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ 
  (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ 
  (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
by sorry

end exactly_one_greater_than_one_l2787_278770


namespace sock_combinations_proof_l2787_278701

/-- The number of ways to choose 4 socks out of 7, with at least one red sock -/
def sockCombinations : ℕ := 20

/-- The total number of socks -/
def totalSocks : ℕ := 7

/-- The number of socks to be chosen -/
def chosenSocks : ℕ := 4

/-- The number of non-red socks -/
def nonRedSocks : ℕ := 6

theorem sock_combinations_proof :
  sockCombinations = Nat.choose totalSocks chosenSocks - Nat.choose nonRedSocks chosenSocks :=
by sorry

end sock_combinations_proof_l2787_278701


namespace problem_2023_squared_minus_2024_times_2022_l2787_278768

theorem problem_2023_squared_minus_2024_times_2022 : 2023^2 - 2024 * 2022 = 1 := by
  sorry

end problem_2023_squared_minus_2024_times_2022_l2787_278768


namespace chocolate_division_l2787_278780

theorem chocolate_division (total : ℚ) (piles : ℕ) (h1 : total = 60 / 7) (h2 : piles = 5) :
  let pile_weight := total / piles
  let received := pile_weight
  let given_back := received / 2
  received - given_back = 6 / 7 := by
  sorry

end chocolate_division_l2787_278780


namespace min_distinct_values_l2787_278707

theorem min_distinct_values (list_size : ℕ) (mode_count : ℕ) (min_distinct : ℕ) : 
  list_size = 3045 →
  mode_count = 15 →
  min_distinct = 218 →
  (∀ n : ℕ, n < min_distinct → 
    n * (mode_count - 1) + mode_count < list_size) ∧
  min_distinct * (mode_count - 1) + mode_count ≥ list_size :=
by sorry

end min_distinct_values_l2787_278707


namespace contrapositive_theorem_l2787_278725

theorem contrapositive_theorem (a b : ℝ) :
  (∀ a b, a > b → 2^a > 2^b - 1) ↔
  (∀ a b, 2^a ≤ 2^b - 1 → a ≤ b) :=
by sorry

end contrapositive_theorem_l2787_278725


namespace triangle_3_4_5_l2787_278777

/-- A function that checks if three numbers can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that the line segments 3, 4, and 5 can form a triangle -/
theorem triangle_3_4_5 : can_form_triangle 3 4 5 := by
  sorry

end triangle_3_4_5_l2787_278777


namespace arcade_change_machine_l2787_278781

theorem arcade_change_machine (total_value : ℕ) (one_dollar_bills : ℕ) : 
  total_value = 300 → one_dollar_bills = 175 → 
  ∃ (five_dollar_bills : ℕ), 
    one_dollar_bills + five_dollar_bills = 200 ∧ 
    one_dollar_bills + 5 * five_dollar_bills = total_value :=
by sorry

end arcade_change_machine_l2787_278781


namespace binomial_26_6_l2787_278787

theorem binomial_26_6 (h1 : Nat.choose 24 4 = 10626)
                      (h2 : Nat.choose 24 5 = 42504)
                      (h3 : Nat.choose 24 6 = 53130) :
  Nat.choose 26 6 = 148764 := by
  sorry

end binomial_26_6_l2787_278787


namespace expand_expression_l2787_278728

theorem expand_expression (x : ℝ) : (x - 2) * (x + 2) * (x^2 + x + 6) = x^4 + x^3 + 2*x^2 - 4*x - 24 := by
  sorry

end expand_expression_l2787_278728


namespace smallest_prime_sum_of_five_primes_l2787_278721

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def isSumOfFiveDifferentPrimes (n : ℕ) : Prop :=
  ∃ p₁ p₂ p₃ p₄ p₅ : ℕ,
    isPrime p₁ ∧ isPrime p₂ ∧ isPrime p₃ ∧ isPrime p₄ ∧ isPrime p₅ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    n = p₁ + p₂ + p₃ + p₄ + p₅

theorem smallest_prime_sum_of_five_primes :
  isPrime 43 ∧
  isSumOfFiveDifferentPrimes 43 ∧
  ∀ n : ℕ, n < 43 → ¬(isPrime n ∧ isSumOfFiveDifferentPrimes n) :=
sorry

end smallest_prime_sum_of_five_primes_l2787_278721


namespace manuscript_pages_count_l2787_278734

/-- Represents the cost structure and revision information for a manuscript --/
structure ManuscriptInfo where
  firstTypingCost : ℕ
  revisionCost : ℕ
  pagesRevisedOnce : ℕ
  pagesRevisedTwice : ℕ
  totalCost : ℕ

/-- Calculates the total number of pages in a manuscript given its cost information --/
def calculateTotalPages (info : ManuscriptInfo) : ℕ :=
  sorry

/-- Theorem stating that for the given manuscript information, the total number of pages is 100 --/
theorem manuscript_pages_count (info : ManuscriptInfo) 
  (h1 : info.firstTypingCost = 10)
  (h2 : info.revisionCost = 5)
  (h3 : info.pagesRevisedOnce = 30)
  (h4 : info.pagesRevisedTwice = 20)
  (h5 : info.totalCost = 1350) :
  calculateTotalPages info = 100 := by
  sorry

end manuscript_pages_count_l2787_278734


namespace first_eligible_retirement_year_l2787_278713

/-- Rule of 70 retirement eligibility -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- Year of hire -/
def hire_year : ℕ := 1990

/-- Age at hire -/
def age_at_hire : ℕ := 32

/-- First year of retirement eligibility -/
def retirement_year : ℕ := 2009

/-- Theorem: The employee is first eligible to retire in 2009 -/
theorem first_eligible_retirement_year :
  rule_of_70 (age_at_hire + (retirement_year - hire_year)) (retirement_year - hire_year) ∧
  ∀ (year : ℕ), year < retirement_year → 
    ¬rule_of_70 (age_at_hire + (year - hire_year)) (year - hire_year) :=
by sorry

end first_eligible_retirement_year_l2787_278713


namespace triangle_pentagon_side_ratio_l2787_278733

theorem triangle_pentagon_side_ratio : ∀ (t p : ℝ),
  (3 * t = 24) →  -- Perimeter of equilateral triangle
  (5 * p = 24) →  -- Perimeter of regular pentagon
  (t / p = 5 / 3) :=
by
  sorry

end triangle_pentagon_side_ratio_l2787_278733


namespace game_probability_l2787_278754

/-- A game with 8 rounds where one person wins each round -/
structure Game where
  rounds : Nat
  alex_prob : ℚ
  mel_prob : ℚ
  chelsea_prob : ℚ

/-- The probability of a specific outcome in the game -/
def outcome_probability (g : Game) (alex_wins mel_wins chelsea_wins : Nat) : ℚ :=
  (g.alex_prob ^ alex_wins) * (g.mel_prob ^ mel_wins) * (g.chelsea_prob ^ chelsea_wins) *
  (Nat.choose g.rounds alex_wins).choose mel_wins

/-- The theorem to be proved -/
theorem game_probability (g : Game) :
  g.rounds = 8 →
  g.alex_prob = 1/2 →
  g.mel_prob = g.chelsea_prob →
  g.alex_prob + g.mel_prob + g.chelsea_prob = 1 →
  outcome_probability g 4 3 1 = 35/512 := by
  sorry

end game_probability_l2787_278754


namespace large_planter_capacity_l2787_278766

/-- Proves that each large planter can hold 20 seeds given the problem conditions -/
theorem large_planter_capacity
  (total_seeds : ℕ)
  (num_large_planters : ℕ)
  (small_planter_capacity : ℕ)
  (num_small_planters : ℕ)
  (h1 : total_seeds = 200)
  (h2 : num_large_planters = 4)
  (h3 : small_planter_capacity = 4)
  (h4 : num_small_planters = 30)
  : (total_seeds - num_small_planters * small_planter_capacity) / num_large_planters = 20 := by
  sorry

end large_planter_capacity_l2787_278766


namespace triangle_properties_l2787_278765

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given triangle satisfies the condition b² + c² - a² = 2bc sin(B+C) -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.b^2 + t.c^2 - t.a^2 = 2 * t.b * t.c * Real.sin (t.B + t.C)

/-- Theorem about the angle A and area of the triangle -/
theorem triangle_properties (t : Triangle) 
    (h1 : satisfiesCondition t) 
    (h2 : t.a = 2) 
    (h3 : t.B = π/3) : 
    t.A = π/4 ∧ 
    (1/2 * t.a * t.b * Real.sin t.C = (3 + Real.sqrt 3) / 2) := by
  sorry


end triangle_properties_l2787_278765


namespace smallest_non_odd_units_digit_l2787_278758

def OddUnitsDigits : Set Nat := {1, 3, 5, 7, 9}

def Digits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem smallest_non_odd_units_digit :
  ∃ (d : Nat), d ∈ Digits ∧ d ∉ OddUnitsDigits ∧ ∀ (x : Nat), x ∈ Digits ∧ x ∉ OddUnitsDigits → d ≤ x :=
by
  sorry

end smallest_non_odd_units_digit_l2787_278758


namespace fibonacci_mod_13_not_4_l2787_278751

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem fibonacci_mod_13_not_4 (n : ℕ) : fibonacci n % 13 ≠ 4 := by
  sorry

end fibonacci_mod_13_not_4_l2787_278751


namespace sum_of_three_consecutive_cubes_divisible_by_nine_l2787_278789

theorem sum_of_three_consecutive_cubes_divisible_by_nine (n : ℤ) :
  ∃ k : ℤ, n^3 + (n+1)^3 + (n+2)^3 = 9 * k := by
  sorry

end sum_of_three_consecutive_cubes_divisible_by_nine_l2787_278789
