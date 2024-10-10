import Mathlib

namespace tangent_points_parallel_to_line_tangent_points_on_curve_unique_tangent_points_main_theorem_l1516_151681

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_points_parallel_to_line (x : ℝ) :
  (f_derivative x = 4) ↔ (x = 1 ∨ x = -1) := by sorry

theorem tangent_points_on_curve :
  f 1 = 0 ∧ f (-1) = -4 := by sorry

theorem unique_tangent_points :
  ∀ x : ℝ, f_derivative x = 4 → (x = 1 ∨ x = -1) := by sorry

-- Main theorem
theorem main_theorem :
  ∃! s : Set (ℝ × ℝ), s = {(1, 0), (-1, -4)} ∧
  (∀ (x y : ℝ), (x, y) ∈ s ↔ f x = y ∧ f_derivative x = 4) := by sorry

end tangent_points_parallel_to_line_tangent_points_on_curve_unique_tangent_points_main_theorem_l1516_151681


namespace x_2021_minus_one_values_l1516_151649

theorem x_2021_minus_one_values (x : ℝ) :
  (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 →
  x^2021 - 1 = 0 ∨ x^2021 - 1 = -2 := by
sorry

end x_2021_minus_one_values_l1516_151649


namespace adams_earnings_l1516_151665

/-- Adam's daily earnings problem -/
theorem adams_earnings (daily_earnings : ℝ) : 
  (daily_earnings * 0.9 * 30 = 1080) → daily_earnings = 40 := by
  sorry

end adams_earnings_l1516_151665


namespace pen_cost_l1516_151600

theorem pen_cost (cost : ℝ) (has : ℝ) (needs : ℝ) : 
  has = cost / 3 → needs = 20 → has + needs = cost → cost = 30 := by
  sorry

end pen_cost_l1516_151600


namespace smaller_k_implies_smaller_certainty_l1516_151601

/-- Represents the observed value of the random variable K² -/
def observed_value (k : ℝ) : Prop := k ≥ 0

/-- Represents the certainty of the relationship between categorical variables -/
def relationship_certainty (c : ℝ) : Prop := c ≥ 0 ∧ c ≤ 1

/-- Theorem stating the relationship between observed K² value and relationship certainty -/
theorem smaller_k_implies_smaller_certainty 
  (X Y : Type) [Finite X] [Finite Y] 
  (k₁ k₂ c₁ c₂ : ℝ) 
  (hk₁ : observed_value k₁) 
  (hk₂ : observed_value k₂) 
  (hc₁ : relationship_certainty c₁) 
  (hc₂ : relationship_certainty c₂) :
  k₁ < k₂ → c₁ < c₂ :=
sorry

end smaller_k_implies_smaller_certainty_l1516_151601


namespace cricket_bat_price_l1516_151637

theorem cricket_bat_price (profit_A_to_B profit_B_to_C profit_C_to_D final_price : ℝ)
  (h1 : profit_A_to_B = 0.2)
  (h2 : profit_B_to_C = 0.25)
  (h3 : profit_C_to_D = 0.3)
  (h4 : final_price = 400) :
  ∃ (original_price : ℝ),
    original_price = final_price / ((1 + profit_A_to_B) * (1 + profit_B_to_C) * (1 + profit_C_to_D)) :=
by sorry

end cricket_bat_price_l1516_151637


namespace midpoint_property_l1516_151617

/-- Given two points A and B in ℝ², prove that if C is their midpoint,
    then 2x - 4y = -22 where (x, y) are the coordinates of C. -/
theorem midpoint_property (A B : ℝ × ℝ) (h1 : A = (15, 10)) (h2 : B = (-5, 6)) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  2 * C.1 - 4 * C.2 = -22 := by
sorry

end midpoint_property_l1516_151617


namespace possible_values_for_e_l1516_151667

def is_digit (n : ℕ) : Prop := n < 10

def distinct (a b c e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ e ∧ b ≠ c ∧ b ≠ e ∧ c ≠ e

def subtraction_equation (a b c e : ℕ) : Prop :=
  10000 * a + 1000 * b + 100 * b + 10 * c + b -
  (10000 * b + 1000 * c + 100 * a + 10 * c + b) =
  10000 * c + 1000 * e + 100 * b + 10 * e + e

theorem possible_values_for_e :
  ∃ (s : Finset ℕ),
    (∀ e ∈ s, is_digit e) ∧
    (∀ e ∈ s, ∃ (a b c : ℕ),
      is_digit a ∧ is_digit b ∧ is_digit c ∧
      distinct a b c e ∧
      subtraction_equation a b c e) ∧
    s.card = 10 :=
sorry

end possible_values_for_e_l1516_151667


namespace math_marks_proof_l1516_151644

/-- Calculates the marks in Mathematics given marks in other subjects and the average -/
def calculate_math_marks (english physics chemistry biology average : ℕ) : ℕ :=
  5 * average - (english + physics + chemistry + biology)

theorem math_marks_proof (english physics chemistry biology average : ℕ) 
  (h_english : english = 96)
  (h_physics : physics = 82)
  (h_chemistry : chemistry = 97)
  (h_biology : biology = 95)
  (h_average : average = 93) :
  calculate_math_marks english physics chemistry biology average = 95 := by
  sorry

end math_marks_proof_l1516_151644


namespace range_of_a_for_increasing_f_l1516_151636

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

-- State the theorem
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, 
    a > 0 ∧ 
    a ≠ 1 ∧ 
    (∀ x y : ℝ, x < y → f a x < f a y) →
    a ∈ Set.Icc (3/2) 2 ∧ a < 2 :=
sorry

end range_of_a_for_increasing_f_l1516_151636


namespace points_per_correct_answer_l1516_151640

theorem points_per_correct_answer 
  (total_problems : ℕ) 
  (total_score : ℕ) 
  (wrong_answers : ℕ) 
  (points_per_wrong : ℕ) 
  (h1 : total_problems = 25)
  (h2 : total_score = 85)
  (h3 : wrong_answers = 3)
  (h4 : points_per_wrong = 1) :
  (total_score + wrong_answers * points_per_wrong) / (total_problems - wrong_answers) = 4 := by
sorry

end points_per_correct_answer_l1516_151640


namespace smallest_positive_integer_with_remainders_l1516_151684

theorem smallest_positive_integer_with_remainders : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a % 5 = 4) ∧ 
  (a % 7 = 6) ∧ 
  (∀ b : ℕ, b > 0 ∧ b % 5 = 4 ∧ b % 7 = 6 → a ≤ b) ∧
  (a = 34) := by
sorry

end smallest_positive_integer_with_remainders_l1516_151684


namespace nondecreasing_function_l1516_151693

-- Define the property that a sequence is nondecreasing
def IsNondecreasingSeq (s : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n ≤ m → s n ≤ s m

-- State the theorem
theorem nondecreasing_function 
  (f : ℝ → ℝ) 
  (hf_dom : ∀ x, 0 < x → f x ≠ 0) 
  (hf_cont : Continuous f) 
  (h_seq : ∀ x > 0, IsNondecreasingSeq (fun n ↦ f (n * x))) : 
  ∀ x y, 0 < x → 0 < y → x ≤ y → f x ≤ f y :=
sorry

end nondecreasing_function_l1516_151693


namespace diamond_value_l1516_151635

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- Converts a number in base 9 to base 10 -/
def base9To10 (d : Digit) : ℕ :=
  9 * d.val + 5

/-- Converts a number in base 10 to itself -/
def base10To10 (d : Digit) : ℕ :=
  10 * d.val + 2

theorem diamond_value :
  ∃ (d : Digit), base9To10 d = base10To10 d ∧ d.val = 3 := by sorry

end diamond_value_l1516_151635


namespace stratified_sampling_third_year_count_l1516_151609

theorem stratified_sampling_third_year_count 
  (total_sample : ℕ) 
  (first_year : ℕ) 
  (second_year : ℕ) 
  (third_year : ℕ) 
  (h1 : total_sample = 200)
  (h2 : first_year = 1300)
  (h3 : second_year = 1200)
  (h4 : third_year = 1500) :
  (third_year * total_sample) / (first_year + second_year + third_year) = 75 := by
sorry

end stratified_sampling_third_year_count_l1516_151609


namespace simultaneous_sequence_probability_l1516_151695

-- Define the probabilities for each coin
def coin_a_heads : ℝ := 0.3
def coin_a_tails : ℝ := 0.7
def coin_b_heads : ℝ := 0.4
def coin_b_tails : ℝ := 0.6

-- Define the number of consecutive flips
def num_flips : ℕ := 6

-- Define the probability of the desired sequence for each coin
def prob_a_sequence : ℝ := coin_a_tails * coin_a_tails * coin_a_heads
def prob_b_sequence : ℝ := coin_b_heads * coin_b_tails * coin_b_tails

-- Theorem to prove
theorem simultaneous_sequence_probability :
  prob_a_sequence * prob_b_sequence = 0.021168 :=
sorry

end simultaneous_sequence_probability_l1516_151695


namespace intersection_when_m_is_5_intersection_equals_B_iff_l1516_151633

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 9}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Statement 1: When m = 5, A ∩ B = {x | 6 ≤ x < 9}
theorem intersection_when_m_is_5 : 
  A ∩ B 5 = {x : ℝ | 6 ≤ x ∧ x < 9} := by sorry

-- Statement 2: A ∩ B = B if and only if m ∈ (-∞, 5)
theorem intersection_equals_B_iff :
  ∀ m : ℝ, A ∩ B m = B m ↔ m < 5 := by sorry

end intersection_when_m_is_5_intersection_equals_B_iff_l1516_151633


namespace definite_integral_tan_trig_l1516_151626

theorem definite_integral_tan_trig : 
  ∃ (f : ℝ → ℝ), (∀ x ∈ Set.Icc (π / 4) (Real.arcsin (Real.sqrt (2 / 3))), 
    HasDerivAt f ((8 * Real.tan x) / (3 * (Real.cos x)^2 + 8 * Real.sin (2 * x) - 7)) x) ∧ 
  (f (Real.arcsin (Real.sqrt (2 / 3))) - f (π / 4) = 
    (4 / 21) * Real.log (abs ((7 * Real.sqrt 2 - 2) / 5)) - 
    (4 / 3) * Real.log (abs (2 - Real.sqrt 2))) := by
  sorry

end definite_integral_tan_trig_l1516_151626


namespace geometric_sequence_common_ratio_sum_l1516_151616

theorem geometric_sequence_common_ratio_sum 
  (k p r : ℝ) 
  (h_nonconstant_p : p ≠ 1) 
  (h_nonconstant_r : r ≠ 1) 
  (h_different_ratios : p ≠ r) 
  (h_relation : k * p^2 - k * r^2 = 5 * (k * p - k * r)) :
  p + r = 5 := by sorry

end geometric_sequence_common_ratio_sum_l1516_151616


namespace fractional_equation_solution_l1516_151692

theorem fractional_equation_solution : 
  ∃ x : ℝ, (2 / (x - 3) = 1 / x) ∧ (x = -3) := by
  sorry

end fractional_equation_solution_l1516_151692


namespace unique_charming_number_l1516_151675

def is_charming (n : ℕ) : Prop :=
  ∃ a b : ℕ, 
    n = 10 * a + b ∧ 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧
    n = 2 * a + b^3

theorem unique_charming_number : 
  ∃! n : ℕ, is_charming n := by sorry

end unique_charming_number_l1516_151675


namespace x_intercept_of_line_l1516_151666

/-- The x-intercept of the line -4x + 6y = 24 is (-6, 0) -/
theorem x_intercept_of_line (x y : ℝ) : 
  -4 * x + 6 * y = 24 → y = 0 → x = -6 := by
  sorry

end x_intercept_of_line_l1516_151666


namespace truck_rental_theorem_l1516_151687

-- Define the capacities of small and large trucks
def small_truck_capacity : ℕ := 300
def large_truck_capacity : ℕ := 400

-- Define the conditions from the problem
axiom condition1 : 2 * small_truck_capacity + 3 * large_truck_capacity = 1800
axiom condition2 : 3 * small_truck_capacity + 4 * large_truck_capacity = 2500

-- Define the total items to be transported
def total_items : ℕ := 3100

-- Define a rental plan as a pair of natural numbers (small trucks, large trucks)
def RentalPlan := ℕ × ℕ

-- Define a function to check if a rental plan is valid
def is_valid_plan (plan : RentalPlan) : Prop :=
  plan.1 * small_truck_capacity + plan.2 * large_truck_capacity = total_items

-- Define the set of all valid rental plans
def valid_plans : Set RentalPlan :=
  {plan | is_valid_plan plan}

-- Theorem stating the main result
theorem truck_rental_theorem :
  (small_truck_capacity = 300 ∧ large_truck_capacity = 400) ∧
  (valid_plans = {(9, 1), (5, 4), (1, 7)}) := by
  sorry


end truck_rental_theorem_l1516_151687


namespace polynomial_root_problem_l1516_151657

/-- Given a polynomial g(x) with three distinct roots that are also roots of f(x),
    prove that f(2) = -16342.5 -/
theorem polynomial_root_problem (p q d : ℝ) : 
  let g : ℝ → ℝ := λ x => x^3 + p*x^2 + 2*x + 15
  let f : ℝ → ℝ := λ x => x^4 + 2*x^3 + q*x^2 + 150*x + d
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    g r₁ = 0 ∧ g r₂ = 0 ∧ g r₃ = 0 ∧
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  f 2 = -16342.5 := by
sorry

end polynomial_root_problem_l1516_151657


namespace roller_coaster_rides_l1516_151628

/-- Given a roller coaster that costs 5 tickets per ride and a person with 10 tickets,
    prove that the number of possible rides is 2. -/
theorem roller_coaster_rides (total_tickets : ℕ) (cost_per_ride : ℕ) (h1 : total_tickets = 10) (h2 : cost_per_ride = 5) :
  total_tickets / cost_per_ride = 2 := by
  sorry

end roller_coaster_rides_l1516_151628


namespace greg_and_earl_final_amount_l1516_151670

/-- Represents the financial state of three individuals and their debts --/
structure FinancialState where
  earl_initial : ℕ
  fred_initial : ℕ
  greg_initial : ℕ
  earl_owes_fred : ℕ
  fred_owes_greg : ℕ
  greg_owes_earl : ℕ

/-- Calculates the final amount Greg and Earl have together after all debts are paid --/
def final_amount (state : FinancialState) : ℕ :=
  (state.earl_initial - state.earl_owes_fred + state.greg_owes_earl) +
  (state.greg_initial + state.fred_owes_greg - state.greg_owes_earl)

/-- Theorem stating that Greg and Earl will have $130 together after all debts are paid --/
theorem greg_and_earl_final_amount (state : FinancialState)
  (h1 : state.earl_initial = 90)
  (h2 : state.fred_initial = 48)
  (h3 : state.greg_initial = 36)
  (h4 : state.earl_owes_fred = 28)
  (h5 : state.fred_owes_greg = 32)
  (h6 : state.greg_owes_earl = 40) :
  final_amount state = 130 := by
  sorry

end greg_and_earl_final_amount_l1516_151670


namespace bacon_strips_for_fourteen_customers_l1516_151698

/-- Breakfast plate configuration at a cafe -/
structure BreakfastPlate where
  eggs : ℕ
  bacon_multiplier : ℕ

/-- Calculate total bacon strips needed for multiple breakfast plates -/
def total_bacon_strips (plate : BreakfastPlate) (num_customers : ℕ) : ℕ :=
  num_customers * (plate.eggs * plate.bacon_multiplier)

/-- Theorem: The cook needs to fry 56 bacon strips for 14 customers -/
theorem bacon_strips_for_fourteen_customers :
  ∃ (plate : BreakfastPlate),
    plate.eggs = 2 ∧
    plate.bacon_multiplier = 2 ∧
    total_bacon_strips plate 14 = 56 := by
  sorry

end bacon_strips_for_fourteen_customers_l1516_151698


namespace sequence_bounds_l1516_151662

def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => (2 * a n + 1) / (a n + 2)

theorem sequence_bounds : ∀ n : ℕ, n ≥ 1 → 1 < a n ∧ a n < 1 + 1 / 3^n := by
  sorry

end sequence_bounds_l1516_151662


namespace distributive_property_l1516_151668

theorem distributive_property (m : ℝ) : m * (m - 1) = m^2 - m := by
  sorry

end distributive_property_l1516_151668


namespace shortest_path_between_circles_l1516_151688

theorem shortest_path_between_circles (center_distance : Real) 
  (radius_large : Real) (radius_small : Real) : Real :=
by
  -- Define the conditions
  have h1 : center_distance = 51 := by sorry
  have h2 : radius_large = 12 := by sorry
  have h3 : radius_small = 7 := by sorry

  -- Calculate the length of the external tangent
  let total_distance := center_distance + radius_large + radius_small
  let tangent_length := Real.sqrt (total_distance^2 - radius_large^2)

  -- Prove that the tangent length is 69 feet
  have h4 : tangent_length = 69 := by sorry

  -- Return the result
  exact tangent_length

end shortest_path_between_circles_l1516_151688


namespace average_of_solutions_eq_neg_two_thirds_l1516_151677

theorem average_of_solutions_eq_neg_two_thirds : 
  let f (x : ℝ) := 3 * x^2 + 4 * x + 1
  let solutions := {x : ℝ | f x = 28}
  ∃ (x₁ x₂ : ℝ), solutions = {x₁, x₂} ∧ (x₁ + x₂) / 2 = -2/3 := by
  sorry

end average_of_solutions_eq_neg_two_thirds_l1516_151677


namespace oranges_sum_l1516_151680

/-- The number of oranges Janet has -/
def janet_oranges : ℕ := 9

/-- The number of oranges Sharon has -/
def sharon_oranges : ℕ := 7

/-- The total number of oranges Janet and Sharon have together -/
def total_oranges : ℕ := janet_oranges + sharon_oranges

theorem oranges_sum : total_oranges = 16 := by
  sorry

end oranges_sum_l1516_151680


namespace triangle_inradius_inequality_l1516_151611

/-- 
For any triangle ABC with side lengths a, b, c and inradius r, 
the inequality 24√3 r³ ≤ (-a+b+c)(a-b+c)(a+b-c) holds.
-/
theorem triangle_inradius_inequality (a b c r : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_inradius : r = (a + b - c) * (b + c - a) * (c + a - b) / (4 * (a + b + c))) :
  24 * Real.sqrt 3 * r^3 ≤ (-a + b + c) * (a - b + c) * (a + b - c) := by
  sorry

end triangle_inradius_inequality_l1516_151611


namespace inequality_implies_interval_bound_l1516_151624

theorem inequality_implies_interval_bound 
  (k m a b : ℝ) 
  (h : ∀ x ∈ Set.Icc a b, |x^2 - k*x - m| ≤ 1) : 
  b - a ≤ 2 * Real.sqrt 2 := by
sorry

end inequality_implies_interval_bound_l1516_151624


namespace external_diagonals_theorem_l1516_151685

/-- Checks if a set of three numbers could be the lengths of external diagonals of a right regular prism -/
def valid_external_diagonals (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + b^2 ≥ c^2 ∧
  b^2 + c^2 ≥ a^2 ∧
  a^2 + c^2 ≥ b^2

theorem external_diagonals_theorem :
  ¬(valid_external_diagonals 3 4 6) ∧
  valid_external_diagonals 3 4 5 ∧
  valid_external_diagonals 5 6 8 ∧
  valid_external_diagonals 5 8 9 ∧
  valid_external_diagonals 6 8 10 :=
by sorry

end external_diagonals_theorem_l1516_151685


namespace light_travel_distance_l1516_151630

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- The number of years we want to calculate the light travel distance for -/
def years : ℕ := 50

/-- Theorem stating that the distance light travels in 50 years
    is equal to 293.5 × 10^12 miles -/
theorem light_travel_distance : 
  (light_year_distance * years : ℝ) = 293.5 * (10 ^ 12) := by
  sorry

end light_travel_distance_l1516_151630


namespace total_stuffed_animals_l1516_151686

def stuffed_animals (mckenna kenley tenly : ℕ) : Prop :=
  (kenley = 2 * mckenna) ∧ 
  (tenly = kenley + 5) ∧ 
  (mckenna + kenley + tenly = 175)

theorem total_stuffed_animals :
  ∃ (mckenna kenley tenly : ℕ), 
    mckenna = 34 ∧ 
    stuffed_animals mckenna kenley tenly :=
by
  sorry

end total_stuffed_animals_l1516_151686


namespace volume_error_percentage_l1516_151625

theorem volume_error_percentage (L W H : ℝ) (L_meas W_meas H_meas : ℝ)
  (h_L : L_meas = 1.08 * L)
  (h_W : W_meas = 1.12 * W)
  (h_H : H_meas = 1.05 * H) :
  let V_true := L * W * H
  let V_calc := L_meas * W_meas * H_meas
  (V_calc - V_true) / V_true * 100 = 25.424 := by
sorry

end volume_error_percentage_l1516_151625


namespace equation_solutions_l1516_151661

theorem equation_solutions : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 3 ∧ x₂ = 5) ∧ 
  (∀ x : ℝ, (x - 2)^6 + (x - 6)^6 = 64 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end equation_solutions_l1516_151661


namespace problem_solution_l1516_151632

theorem problem_solution (x y : ℝ) (h : 3 * x - y ≤ Real.log (x + 2 * y - 3) + Real.log (2 * x - 3 * y + 5)) : 
  x + y = 16 / 7 := by
sorry

end problem_solution_l1516_151632


namespace complex_exp_thirteen_pi_over_two_l1516_151645

theorem complex_exp_thirteen_pi_over_two : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end complex_exp_thirteen_pi_over_two_l1516_151645


namespace consecutive_integers_square_sum_odd_l1516_151671

theorem consecutive_integers_square_sum_odd (a b c M : ℤ) : 
  (a = b + 1 ∨ b = a + 1) →  -- a and b are consecutive integers
  c = a * b →               -- c = ab
  M^2 = a^2 + b^2 + c^2 →   -- M^2 = a^2 + b^2 + c^2
  Odd (M^2) :=              -- M^2 is an odd number
by
  sorry

end consecutive_integers_square_sum_odd_l1516_151671


namespace pet_shop_stock_worth_l1516_151613

/-- The total worth of the stock in a pet shop -/
def stock_worth (num_puppies num_kittens puppy_price kitten_price : ℕ) : ℕ :=
  num_puppies * puppy_price + num_kittens * kitten_price

/-- Theorem stating that the stock worth is 100 given the specific conditions -/
theorem pet_shop_stock_worth :
  stock_worth 2 4 20 15 = 100 := by
  sorry

end pet_shop_stock_worth_l1516_151613


namespace x_value_proof_l1516_151634

theorem x_value_proof (x : ℝ) (h : 0.65 * x = 0.20 * 487.50) : x = 150 := by
  sorry

end x_value_proof_l1516_151634


namespace parabola_translation_up_one_unit_l1516_151610

/-- Represents a parabola in the form y = ax² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Translates a parabola vertically -/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b + d }

theorem parabola_translation_up_one_unit :
  let original := Parabola.mk 3 0
  let translated := translate_vertical original 1
  translated = Parabola.mk 3 1 := by sorry

end parabola_translation_up_one_unit_l1516_151610


namespace consecutive_days_sum_l1516_151648

theorem consecutive_days_sum (x : ℕ) : 
  x + (x + 1) + (x + 2) = 33 → x = 10 ∧ x + 1 = 11 ∧ x + 2 = 12 := by
  sorry

end consecutive_days_sum_l1516_151648


namespace dogs_not_doing_anything_l1516_151646

/-- Proves that the number of dogs not doing anything is 10, given the total number of dogs and the number of dogs engaged in each activity. -/
theorem dogs_not_doing_anything (total : ℕ) (running : ℕ) (playing : ℕ) (barking : ℕ) : 
  total = 88 → 
  running = 12 → 
  playing = total / 2 → 
  barking = total / 4 → 
  total - (running + playing + barking) = 10 := by
sorry

end dogs_not_doing_anything_l1516_151646


namespace joe_fruit_probability_l1516_151664

/-- The number of meals in a day -/
def num_meals : ℕ := 3

/-- The number of fruit types available -/
def num_fruits : ℕ := 4

/-- The probability of choosing a specific fruit at any meal -/
def prob_fruit : ℚ := 1 / num_fruits

/-- The probability of eating the same fruit for all meals -/
def prob_same_fruit : ℚ := num_fruits * (prob_fruit ^ num_meals)

theorem joe_fruit_probability :
  1 - prob_same_fruit = 15 / 16 :=
sorry

end joe_fruit_probability_l1516_151664


namespace rectangular_prism_volume_error_l1516_151619

theorem rectangular_prism_volume_error (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let actual_volume := a * b * c
  let measured_volume := (a * 1.08) * (b * 0.90) * (c * 0.94)
  let error_percentage := (measured_volume - actual_volume) / actual_volume * 100
  error_percentage = -2.728 := by
sorry

end rectangular_prism_volume_error_l1516_151619


namespace c_profit_share_is_400_l1516_151653

/-- Represents an investor in the business --/
structure Investor where
  name : String
  investment : ℕ

/-- Represents the business venture --/
structure Business where
  investors : List Investor
  duration : ℕ
  total_profit : ℕ

/-- Calculates an investor's share of the profit --/
def profit_share (b : Business) (i : Investor) : ℕ :=
  let total_investment := b.investors.map (·.investment) |>.sum
  (i.investment * b.total_profit) / total_investment

theorem c_profit_share_is_400 (b : Business) (c : Investor) :
  b.investors = [⟨"a", 800⟩, ⟨"b", 1000⟩, c] →
  b.duration = 2 →
  b.total_profit = 1000 →
  c.investment = 1200 →
  profit_share b c = 400 := by
  sorry

#eval profit_share
  ⟨[⟨"a", 800⟩, ⟨"b", 1000⟩, ⟨"c", 1200⟩], 2, 1000⟩
  ⟨"c", 1200⟩

end c_profit_share_is_400_l1516_151653


namespace not_all_data_sets_have_regression_equation_l1516_151654

-- Define a type for data sets
structure DataSet where
  -- Add necessary fields for a data set
  nonEmpty : Bool

-- Define a predicate for whether a regression equation exists for a data set
def hasRegressionEquation (d : DataSet) : Prop := sorry

-- Theorem stating that not every data set has a regression equation
theorem not_all_data_sets_have_regression_equation :
  ¬ ∀ (d : DataSet), hasRegressionEquation d := by
  sorry


end not_all_data_sets_have_regression_equation_l1516_151654


namespace passing_mark_is_200_l1516_151622

/-- Represents an exam with a total number of marks and a passing mark. -/
structure Exam where
  total_marks : ℕ
  passing_mark : ℕ

/-- Defines the conditions of the exam as described in the problem. -/
def exam_conditions (e : Exam) : Prop :=
  (e.total_marks * 30 / 100 + 50 = e.passing_mark) ∧
  (e.total_marks * 45 / 100 = e.passing_mark + 25)

/-- Theorem stating that under the given conditions, the passing mark is 200. -/
theorem passing_mark_is_200 :
  ∃ e : Exam, exam_conditions e ∧ e.passing_mark = 200 := by
  sorry


end passing_mark_is_200_l1516_151622


namespace division_problem_l1516_151663

theorem division_problem (divisor quotient remainder : ℕ) (h1 : divisor = 21) (h2 : quotient = 8) (h3 : remainder = 3) :
  divisor * quotient + remainder = 171 := by
  sorry

end division_problem_l1516_151663


namespace negative_p_exponent_product_l1516_151659

theorem negative_p_exponent_product (p : ℝ) : (-p)^2 * (-p)^3 = -p^5 := by sorry

end negative_p_exponent_product_l1516_151659


namespace cistern_leak_emptying_time_l1516_151655

theorem cistern_leak_emptying_time 
  (normal_fill_time : ℝ) 
  (leak_fill_time : ℝ) 
  (h1 : normal_fill_time = 8) 
  (h2 : leak_fill_time = 10) : 
  (1 / (1 / leak_fill_time - 1 / normal_fill_time)) = 40 := by
  sorry

end cistern_leak_emptying_time_l1516_151655


namespace tim_score_l1516_151612

def single_line_score : ℕ := 1000
def tetris_multiplier : ℕ := 8
def consecutive_multiplier : ℕ := 2

def calculate_score (singles : ℕ) (regular_tetrises : ℕ) (consecutive_tetrises : ℕ) : ℕ :=
  singles * single_line_score +
  regular_tetrises * (single_line_score * tetris_multiplier) +
  consecutive_tetrises * (single_line_score * tetris_multiplier * consecutive_multiplier)

theorem tim_score : calculate_score 6 2 2 = 54000 := by
  sorry

end tim_score_l1516_151612


namespace irreducible_fractions_count_l1516_151623

/-- A rational number between 0 and 1 in irreducible fraction form -/
structure IrreducibleFraction :=
  (numerator : ℕ)
  (denominator : ℕ)
  (is_between_0_and_1 : numerator < denominator)
  (is_irreducible : Nat.gcd numerator denominator = 1)
  (product_is_20 : numerator * denominator = 20)

/-- The count of irreducible fractions between 0 and 1 with numerator-denominator product of 20 -/
def count_irreducible_fractions : ℕ := sorry

/-- The main theorem stating there are 128 such fractions -/
theorem irreducible_fractions_count :
  count_irreducible_fractions = 128 := by sorry

end irreducible_fractions_count_l1516_151623


namespace peach_trees_count_l1516_151696

/-- The number of peach trees in an orchard. -/
def number_of_peach_trees (apple_trees : ℕ) (apple_yield : ℕ) (peach_yield : ℕ) (total_yield : ℕ) : ℕ :=
  (total_yield - apple_trees * apple_yield) / peach_yield

/-- Theorem stating the number of peach trees in the orchard. -/
theorem peach_trees_count : number_of_peach_trees 30 150 65 7425 = 45 := by
  sorry

end peach_trees_count_l1516_151696


namespace coopers_age_l1516_151638

/-- Given the ages of four people with specific relationships, prove Cooper's age --/
theorem coopers_age (cooper dante maria emily : ℕ) : 
  cooper + dante + maria + emily = 62 →
  dante = 2 * cooper →
  maria = dante + 1 →
  emily = 3 * cooper →
  cooper = 8 :=
by sorry

end coopers_age_l1516_151638


namespace olivias_papers_l1516_151682

/-- Given an initial number of papers and a number of papers used,
    calculate the remaining number of papers. -/
def remaining_papers (initial : ℕ) (used : ℕ) : ℕ :=
  initial - used

/-- Theorem stating that given 81 initial papers and 56 used papers,
    the remaining number is 25. -/
theorem olivias_papers :
  remaining_papers 81 56 = 25 := by
  sorry

end olivias_papers_l1516_151682


namespace inequality_properties_l1516_151689

theorem inequality_properties (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a + c > b + c) ∧ (1 / a < 1 / b) := by sorry

end inequality_properties_l1516_151689


namespace sarah_candy_consumption_l1516_151631

theorem sarah_candy_consumption 
  (candy_from_neighbors : ℕ)
  (candy_from_sister : ℕ)
  (days_lasted : ℕ)
  (h1 : candy_from_neighbors = 66)
  (h2 : candy_from_sister = 15)
  (h3 : days_lasted = 9)
  (h4 : days_lasted > 0) :
  (candy_from_neighbors + candy_from_sister) / days_lasted = 9 :=
by sorry

end sarah_candy_consumption_l1516_151631


namespace seans_sandwiches_l1516_151604

/-- Calculates the number of sandwiches Sean bought given the costs of items and total cost -/
theorem seans_sandwiches
  (soda_cost : ℕ)
  (soup_cost : ℕ)
  (sandwich_cost : ℕ)
  (total_cost : ℕ)
  (h1 : soda_cost = 3)
  (h2 : soup_cost = 6)
  (h3 : sandwich_cost = 9)
  (h4 : total_cost = 18)
  : (total_cost - soda_cost - soup_cost) / sandwich_cost = 1 := by
  sorry

#check seans_sandwiches

end seans_sandwiches_l1516_151604


namespace longest_segment_in_cylinder_l1516_151606

/-- The longest segment that can fit inside a cylinder -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = 10 * Real.sqrt 2 := by
  sorry

end longest_segment_in_cylinder_l1516_151606


namespace quadratic_linear_intersection_l1516_151647

/-- Given a quadratic function f(x) = ax² + bx + c and a linear function g(x) = -bx,
    where a > b > c and f(1) = 0, prove that f and g intersect at two distinct points. -/
theorem quadratic_linear_intersection
  (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a * 1^2 + b * 1 + c = 0) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    a * x₁^2 + b * x₁ + c = -b * x₁ ∧
    a * x₂^2 + b * x₂ + c = -b * x₂ :=
by sorry

end quadratic_linear_intersection_l1516_151647


namespace part_one_part_two_l1516_151678

-- Define propositions P and Q
def P (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def Q (x : ℝ) : Prop := |x - 3| ≤ 1

-- Part 1
theorem part_one (x : ℝ) (h : P x 1 ∨ Q x) : 1 < x ∧ x ≤ 4 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h : ∀ x, ¬(P x a) → ¬(Q x)) 
  (h_not_sufficient : ∃ x, ¬(P x a) ∧ Q x) : 4/3 ≤ a ∧ a ≤ 2 := by sorry

end part_one_part_two_l1516_151678


namespace geometric_sequence_n_l1516_151697

/-- For a geometric sequence {a_n} with a₁ = 9/8, q = 2/3, and aₙ = 1/3, n = 4 -/
theorem geometric_sequence_n (a : ℕ → ℚ) :
  (∀ k, a (k + 1) = a k * (2/3)) →  -- geometric sequence condition
  a 1 = 9/8 →                      -- first term condition
  (∃ n, a n = 1/3) →               -- nth term condition
  ∃ n, n = 4 ∧ a n = 1/3 :=
by sorry

end geometric_sequence_n_l1516_151697


namespace constant_term_binomial_expansion_l1516_151674

theorem constant_term_binomial_expansion :
  ∃ (c : ℝ), ∀ (x : ℝ),
    (λ x => (4^x - 2^(-x))^6) x = c + (λ x => (4^x - 2^(-x))^6 - c) x ∧ c = 15 := by
  sorry

end constant_term_binomial_expansion_l1516_151674


namespace cubic_values_quadratic_polynomial_l1516_151627

theorem cubic_values_quadratic_polynomial 
  (a b c : ℕ) (ha : a ≠ b) (hb : b ≠ c) (hc : a ≠ c) :
  ∃ (p q r : ℤ) (x₁ x₂ x₃ : ℤ), 
    p > 0 ∧ 
    (p * x₁^2 + q * x₁ + r = a^3) ∧
    (p * x₂^2 + q * x₂ + r = b^3) ∧
    (p * x₃^2 + q * x₃ + r = c^3) :=
sorry

end cubic_values_quadratic_polynomial_l1516_151627


namespace quadratic_roots_property_l1516_151620

theorem quadratic_roots_property : ∀ x₁ x₂ : ℝ, 
  (∀ x : ℝ, x^2 - 3*x + 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁^2 - 2*x₁ + x₂ = 2 := by
  sorry

end quadratic_roots_property_l1516_151620


namespace average_weight_increase_l1516_151615

/-- Proves that replacing a person weighing 60 kg with a person weighing 80 kg
    in a group of 8 people increases the average weight by 2.5 kg -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 8 * initial_average
  let new_total := initial_total - 60 + 80
  let new_average := new_total / 8
  new_average - initial_average = 2.5 := by sorry

end average_weight_increase_l1516_151615


namespace base_number_proof_l1516_151621

theorem base_number_proof (x : ℝ) : 16^7 = x^14 → x = 4 := by
  sorry

end base_number_proof_l1516_151621


namespace function_equation_solution_l1516_151690

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x : ℝ, 2 * f x + f (1 - x) = x^2) →
  (∀ x : ℝ, f x = (1/3) * (x^2 + 2*x - 1)) := by
sorry

end function_equation_solution_l1516_151690


namespace bills_weights_theorem_l1516_151656

/-- The total weight of sand in two jugs filled partially with sand -/
def total_weight (jug_capacity : ℝ) (fill_percentage : ℝ) (num_jugs : ℕ) (sand_density : ℝ) : ℝ :=
  jug_capacity * fill_percentage * (num_jugs : ℝ) * sand_density

/-- Theorem stating the total weight of sand in Bill's improvised weights -/
theorem bills_weights_theorem :
  total_weight 2 0.7 2 5 = 14 := by
  sorry

end bills_weights_theorem_l1516_151656


namespace perfect_square_condition_l1516_151614

theorem perfect_square_condition (m : ℝ) : 
  (∃ x : ℝ, ∃ k : ℝ, x^2 + 2*(m-3)*x + 16 = k^2) → (m = 7 ∨ m = -1) := by
  sorry

end perfect_square_condition_l1516_151614


namespace downstream_distance_84km_l1516_151618

/-- Represents the swimming scenario --/
structure SwimmingScenario where
  current_speed : ℝ
  upstream_distance : ℝ
  upstream_time : ℝ
  downstream_time : ℝ

/-- Calculates the downstream distance given a swimming scenario --/
def downstream_distance (s : SwimmingScenario) : ℝ :=
  sorry

/-- Theorem stating the downstream distance for the given scenario --/
theorem downstream_distance_84km (s : SwimmingScenario) 
  (h1 : s.current_speed = 2.5)
  (h2 : s.upstream_distance = 24)
  (h3 : s.upstream_time = 8)
  (h4 : s.downstream_time = 8) :
  downstream_distance s = 84 := by
  sorry

end downstream_distance_84km_l1516_151618


namespace age_difference_l1516_151602

theorem age_difference (a b : ℕ) : 
  (a > 0) → 
  (b > 0) → 
  (a < 100) → 
  (b < 100) → 
  (a = 10 * (b % 10) + (a / 10)) → 
  (b = 10 * (a % 10) + (b / 10)) → 
  (a + 7 = 3 * (b + 7)) → 
  (a - b = 45) := by
sorry

end age_difference_l1516_151602


namespace min_minutes_for_plan_b_cheaper_l1516_151660

/-- Cost function for Plan A -/
def costA (x : ℕ) : ℕ :=
  if x ≤ 300 then 8 * x else 2400 + 7 * (x - 300)

/-- Cost function for Plan B -/
def costB (x : ℕ) : ℕ := 2500 + 4 * x

/-- Theorem stating the minimum number of minutes for Plan B to be cheaper -/
theorem min_minutes_for_plan_b_cheaper :
  ∀ x : ℕ, x < 301 → costA x ≤ costB x ∧
  ∀ y : ℕ, y ≥ 301 → costB y < costA y := by
  sorry

#check min_minutes_for_plan_b_cheaper

end min_minutes_for_plan_b_cheaper_l1516_151660


namespace no_solution_iff_parallel_equation_no_solution_iff_l1516_151643

/-- Two 2D vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v.1 = c * w.1 ∧ v.2 = c * w.2

/-- The equation has no solution if and only if the direction vectors are parallel -/
theorem no_solution_iff_parallel (m : ℝ) : Prop :=
  parallel (5, 2) (-2, m)

/-- The main theorem: the equation has no solution if and only if m = -4/5 -/
theorem equation_no_solution_iff (m : ℝ) : 
  no_solution_iff_parallel m ↔ m = -4/5 := by sorry

end no_solution_iff_parallel_equation_no_solution_iff_l1516_151643


namespace compound_carbon_atoms_l1516_151699

/-- Represents a chemical compound --/
structure Compound where
  hydrogen : ℕ
  oxygen : ℕ
  molecular_weight : ℕ

/-- Atomic weights of elements --/
def atomic_weight : String → ℕ
  | "C" => 12
  | "H" => 1
  | "O" => 16
  | _ => 0

/-- Calculate the number of Carbon atoms in a compound --/
def carbon_atoms (c : Compound) : ℕ :=
  (c.molecular_weight - (c.hydrogen * atomic_weight "H" + c.oxygen * atomic_weight "O")) / atomic_weight "C"

/-- Theorem: The given compound has 4 Carbon atoms --/
theorem compound_carbon_atoms :
  let c : Compound := { hydrogen := 8, oxygen := 2, molecular_weight := 88 }
  carbon_atoms c = 4 := by
  sorry

end compound_carbon_atoms_l1516_151699


namespace twentyseven_eighths_two_thirds_power_l1516_151673

theorem twentyseven_eighths_two_thirds_power :
  (27 / 8 : ℝ) ^ (2 / 3) = 9 / 4 := by
  sorry

end twentyseven_eighths_two_thirds_power_l1516_151673


namespace matrix_equation_solution_l1516_151691

theorem matrix_equation_solution : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![0, 5; 0, 10]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![2, 5; 4, 3]
  let C : Matrix (Fin 2) (Fin 2) ℚ := !![10, 15; 20, 6]
  A * B = C := by sorry

end matrix_equation_solution_l1516_151691


namespace gold_alloy_ratio_l1516_151652

/-- Proves that adding 62.5 ounces of pure gold to an alloy of 100 ounces that is 35% gold 
    will result in a new alloy that is 60% gold -/
theorem gold_alloy_ratio (original_weight : ℝ) (original_gold_ratio : ℝ) 
    (new_gold_ratio : ℝ) (added_gold : ℝ) : 
    original_weight = 100 →
    original_gold_ratio = 0.35 →
    new_gold_ratio = 0.60 →
    added_gold = 62.5 →
    (original_weight * original_gold_ratio + added_gold) / (original_weight + added_gold) = new_gold_ratio := by
  sorry

#check gold_alloy_ratio

end gold_alloy_ratio_l1516_151652


namespace swallow_flock_capacity_l1516_151605

/-- Represents the carrying capacity of different types of swallows -/
structure SwallowCapacity where
  american : ℕ
  european : ℕ
  african : ℕ

/-- Represents the composition of a flock of swallows -/
structure SwallowFlock where
  american : ℕ
  european : ℕ
  african : ℕ

/-- Calculates the total number of swallows in a flock -/
def totalSwallows (flock : SwallowFlock) : ℕ :=
  flock.american + flock.european + flock.african

/-- Calculates the maximum weight a flock can carry -/
def maxCarryWeight (capacity : SwallowCapacity) (flock : SwallowFlock) : ℕ :=
  flock.american * capacity.american +
  flock.european * capacity.european +
  flock.african * capacity.african

/-- Theorem stating the maximum carrying capacity of a specific flock of swallows -/
theorem swallow_flock_capacity
  (capacity : SwallowCapacity)
  (flock : SwallowFlock)
  (h1 : capacity.american = 5)
  (h2 : capacity.european = 10)
  (h3 : capacity.african = 15)
  (h4 : flock.american = 45)
  (h5 : flock.european = 30)
  (h6 : flock.african = 75)
  (h7 : totalSwallows flock = 150)
  (h8 : flock.american * 2 = flock.european * 3)
  (h9 : flock.american * 5 = flock.african * 3) :
  maxCarryWeight capacity flock = 1650 := by
  sorry


end swallow_flock_capacity_l1516_151605


namespace max_plate_valid_l1516_151607

-- Define a custom type for characters that can be on a number plate
inductive PlateChar
| Zero
| Six
| Nine
| H
| O

-- Define a function to check if a character looks the same upside down
def looks_same_upside_down (c : PlateChar) : Bool :=
  match c with
  | PlateChar.Zero => true
  | PlateChar.H => true
  | PlateChar.O => true
  | _ => false

-- Define a function to get the upside down version of a character
def upside_down (c : PlateChar) : PlateChar :=
  match c with
  | PlateChar.Six => PlateChar.Nine
  | PlateChar.Nine => PlateChar.Six
  | c => c

-- Define a number plate as a list of PlateChar
def NumberPlate := List PlateChar

-- Define the specific number plate we want to check
def max_plate : NumberPlate :=
  [PlateChar.Six, PlateChar.Zero, PlateChar.H, PlateChar.O, PlateChar.H, PlateChar.Zero, PlateChar.Nine]

-- Theorem: Max's plate is valid when turned upside down
theorem max_plate_valid : 
  max_plate.reverse.map upside_down = max_plate :=
by sorry

end max_plate_valid_l1516_151607


namespace perpendicular_condition_l1516_151658

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of perpendicularity for two lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The first line (a+2)x+3ay+1=0 -/
def line1 (a : ℝ) : Line :=
  { a := a + 2, b := 3 * a, c := 1 }

/-- The second line (a-2)x+(a+2)y-3=0 -/
def line2 (a : ℝ) : Line :=
  { a := a - 2, b := a + 2, c := -3 }

theorem perpendicular_condition (a : ℝ) :
  (a = -2 → perpendicular (line1 a) (line2 a)) ∧
  (∃ b : ℝ, b ≠ -2 ∧ perpendicular (line1 b) (line2 b)) := by
  sorry

end perpendicular_condition_l1516_151658


namespace bond_return_rate_l1516_151650

-- Define the total investment
def total_investment : ℝ := 10000

-- Define the bank interest rate
def bank_interest_rate : ℝ := 0.05

-- Define the total annual income
def total_annual_income : ℝ := 660

-- Define the amount invested in each method
def investment_per_method : ℝ := 6000

-- Theorem statement
theorem bond_return_rate :
  let bank_income := investment_per_method * bank_interest_rate
  let bond_income := total_annual_income - bank_income
  bond_income / investment_per_method = 0.06 := by
sorry


end bond_return_rate_l1516_151650


namespace road_graveling_cost_l1516_151641

/-- Calculate the cost of graveling two intersecting roads on a rectangular lawn -/
theorem road_graveling_cost (lawn_length lawn_width road_width gravel_cost : ℝ) 
  (h1 : lawn_length = 90)
  (h2 : lawn_width = 60)
  (h3 : road_width = 10)
  (h4 : gravel_cost = 3) : 
  (lawn_length * road_width + lawn_width * road_width - road_width * road_width) * gravel_cost = 4200 := by
  sorry

#check road_graveling_cost

end road_graveling_cost_l1516_151641


namespace g_is_self_inverse_l1516_151679

-- Define a function f that is symmetric about y=x-1
def f : ℝ → ℝ := sorry

-- Define the property of f being symmetric about y=x-1
axiom f_symmetric : ∀ x y : ℝ, f x = y ↔ f (y + 1) = x + 1

-- Define g in terms of f
def g : ℝ → ℝ := λ x => f (x + 1)

-- State the theorem
theorem g_is_self_inverse : ∀ x : ℝ, g (g x) = x := by sorry

end g_is_self_inverse_l1516_151679


namespace expression_equals_three_l1516_151651

theorem expression_equals_three : 
  3⁻¹ + (Real.sqrt 2 - 1)^0 + 2 * Real.sin (30 * π / 180) - (-2/3) = 3 := by
  sorry

end expression_equals_three_l1516_151651


namespace log_cutting_theorem_l1516_151669

/-- The number of pieces of wood after cutting a log -/
def num_pieces (initial_logs : ℕ) (num_cuts : ℕ) : ℕ :=
  initial_logs + num_cuts

/-- Theorem: Cutting a single log 10 times results in 11 pieces -/
theorem log_cutting_theorem :
  num_pieces 1 10 = 11 := by
  sorry

end log_cutting_theorem_l1516_151669


namespace intersection_range_distance_when_b_is_one_l1516_151603

/-- The line y = x + b intersects the ellipse x^2/2 + y^2 = 1 at two distinct points -/
def intersects_at_two_points (b : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    y₁ = x₁ + b ∧ y₂ = x₂ + b ∧
    x₁^2/2 + y₁^2 = 1 ∧ x₂^2/2 + y₂^2 = 1

/-- The range of b for which the line intersects the ellipse at two distinct points -/
theorem intersection_range :
  ∀ b : ℝ, intersects_at_two_points b ↔ -Real.sqrt 3 < b ∧ b < Real.sqrt 3 :=
sorry

/-- The distance between intersection points when b = 1 -/
theorem distance_when_b_is_one :
  ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    y₁ = x₁ + 1 ∧ y₂ = x₂ + 1 ∧
    x₁^2/2 + y₁^2 = 1 ∧ x₂^2/2 + y₂^2 = 1 ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 4 * Real.sqrt 2 / 3 :=
sorry

end intersection_range_distance_when_b_is_one_l1516_151603


namespace overlap_area_of_specific_triangles_l1516_151642

/-- A point in a 2D grid. -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- A triangle defined by three points in a 2D grid. -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- Calculate the area of a triangle given its base and height. -/
def triangleArea (base height : ℕ) : ℚ :=
  (base * height : ℚ) / 2

/-- The main theorem stating the area of overlap between two specific triangles. -/
theorem overlap_area_of_specific_triangles :
  let triangleA : GridTriangle := ⟨⟨0, 0⟩, ⟨2, 0⟩, ⟨2, 2⟩⟩
  let triangleB : GridTriangle := ⟨⟨0, 2⟩, ⟨2, 2⟩, ⟨0, 0⟩⟩
  triangleArea 2 2 = 2 := by sorry

end overlap_area_of_specific_triangles_l1516_151642


namespace burn_all_bridges_probability_l1516_151683

/-- The number of islands in the lake -/
def num_islands : ℕ := 2013

/-- The probability of choosing a new bridge at each step -/
def prob_new_bridge : ℚ := 2/3

/-- The probability of burning all bridges -/
def prob_burn_all : ℚ := num_islands * prob_new_bridge ^ (num_islands - 1)

/-- Theorem stating the probability of burning all bridges -/
theorem burn_all_bridges_probability :
  prob_burn_all = num_islands * (2/3) ^ (num_islands - 1) := by sorry

end burn_all_bridges_probability_l1516_151683


namespace jonas_tshirts_l1516_151694

/-- Represents the number of items in Jonas' wardrobe -/
structure Wardrobe where
  socks : ℕ
  shoes : ℕ
  pants : ℕ
  tshirts : ℕ

/-- Calculates the total number of individual items in the wardrobe -/
def totalItems (w : Wardrobe) : ℕ :=
  2 * w.socks + 2 * w.shoes + 2 * w.pants + w.tshirts

/-- The theorem to prove -/
theorem jonas_tshirts : 
  ∀ w : Wardrobe, 
    w.socks = 20 → 
    w.shoes = 5 → 
    w.pants = 10 → 
    totalItems w + 2 * 35 = 2 * totalItems w → 
    w.tshirts = 70 := by
  sorry


end jonas_tshirts_l1516_151694


namespace pizza_slices_per_person_l1516_151672

/-- Given a pizza with 12 slices shared equally among Ron and his 2 friends, 
    prove that each person ate 4 slices. -/
theorem pizza_slices_per_person (total_slices : Nat) (num_friends : Nat) :
  total_slices = 12 →
  num_friends = 2 →
  total_slices / (num_friends + 1) = 4 :=
by sorry

end pizza_slices_per_person_l1516_151672


namespace volume_of_inscribed_sphere_l1516_151639

/-- The volume of a sphere inscribed in a cube with edge length 12 inches is 288π cubic inches. -/
theorem volume_of_inscribed_sphere (π : ℝ) :
  let cube_edge : ℝ := 12
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3
  sphere_volume = 288 * π := by sorry

end volume_of_inscribed_sphere_l1516_151639


namespace system_solution_l1516_151608

theorem system_solution (m : ℤ) : 
  (∃ (x y : ℝ), 
    x - 2*y = m ∧ 
    2*x + 3*y = 2*m - 3 ∧ 
    3*x + y ≥ 0 ∧ 
    x + 5*y < 0) ↔ 
  (m = 1 ∨ m = 2) := by sorry

end system_solution_l1516_151608


namespace trajectory_of_point_m_l1516_151676

/-- The trajectory of point M given a circle and specific conditions -/
theorem trajectory_of_point_m (x y : ℝ) : 
  (∃ m n : ℝ, 
    m^2 + n^2 = 9 ∧  -- P(m, n) is on the circle
    (x - m)^2 + y^2 = ((m - x)^2 + y^2) / 4) -- PM = 2MP'
  → x^2 / 9 + y^2 = 1 := by
  sorry

end trajectory_of_point_m_l1516_151676


namespace brother_statement_contradiction_l1516_151629

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

-- Define the brother's behavior
structure Brother where
  lying_days : Set Day
  today : Day

-- Define the brother's statement
def brother_statement (b : Brother) : Prop :=
  b.today ∈ b.lying_days

-- Theorem: The brother's statement leads to a contradiction
theorem brother_statement_contradiction (b : Brother) :
  ¬(brother_statement b ↔ ¬(brother_statement b)) :=
sorry

end brother_statement_contradiction_l1516_151629
