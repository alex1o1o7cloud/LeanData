import Mathlib

namespace NUMINAMATH_GPT_percent_sold_second_day_l1936_193651

-- Defining the problem conditions
def initial_pears (x : ℕ) : ℕ := x
def pears_sold_first_day (x : ℕ) : ℕ := (20 * x) / 100
def pears_remaining_after_first_sale (x : ℕ) : ℕ := x - pears_sold_first_day x
def pears_thrown_away_first_day (x : ℕ) : ℕ := (50 * pears_remaining_after_first_sale x) / 100
def pears_remaining_after_first_day (x : ℕ) : ℕ := pears_remaining_after_first_sale x - pears_thrown_away_first_day x
def total_pears_thrown_away (x : ℕ) : ℕ := (72 * x) / 100
def pears_thrown_away_second_day (x : ℕ) : ℕ := total_pears_thrown_away x - pears_thrown_away_first_day x
def pears_remaining_after_second_day (x : ℕ) : ℕ := pears_remaining_after_first_day x - pears_thrown_away_second_day x

-- Prove that the vendor sold 20% of the remaining pears on the second day
theorem percent_sold_second_day (x : ℕ) (h : x > 0) :
  ((pears_remaining_after_second_day x * 100) / pears_remaining_after_first_day x) = 20 :=
by 
  sorry

end NUMINAMATH_GPT_percent_sold_second_day_l1936_193651


namespace NUMINAMATH_GPT_solve_equation_l1936_193646

-- Define the equation to be proven
def equation (x : ℚ) : Prop :=
  (x + 4) / (x - 3) = (x - 2) / (x + 2)

-- State the theorem
theorem solve_equation : equation (-2 / 11) :=
by
  -- Introduce the equation and the solution to be proven
  unfold equation

  -- Simplify the equation to verify the solution
  sorry


end NUMINAMATH_GPT_solve_equation_l1936_193646


namespace NUMINAMATH_GPT_original_solution_sugar_percentage_l1936_193667

theorem original_solution_sugar_percentage :
  ∃ x : ℚ, (∀ (y : ℚ), (y = 14) → (∃ (z : ℚ), (z = 26) → (3 / 4 * x + 1 / 4 * z = y))) → x = 10 := 
  sorry

end NUMINAMATH_GPT_original_solution_sugar_percentage_l1936_193667


namespace NUMINAMATH_GPT_greatest_possible_x_max_possible_x_l1936_193672

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end NUMINAMATH_GPT_greatest_possible_x_max_possible_x_l1936_193672


namespace NUMINAMATH_GPT_tyler_saltwater_animals_l1936_193666

/-- Tyler had 56 aquariums for saltwater animals and each aquarium has 39 animals in it. 
    We need to prove that the total number of saltwater animals Tyler has is 2184. --/
theorem tyler_saltwater_animals : (56 * 39) = 2184 := by
  sorry

end NUMINAMATH_GPT_tyler_saltwater_animals_l1936_193666


namespace NUMINAMATH_GPT_time_to_cross_first_platform_l1936_193676

noncomputable section

def train_length : ℝ := 310
def platform_1_length : ℝ := 110
def platform_2_length : ℝ := 250
def crossing_time_platform_2 : ℝ := 20

def total_distance_2 (train_length platform_2_length : ℝ) : ℝ :=
  train_length + platform_2_length

def train_speed (total_distance_2 crossing_time_platform_2 : ℝ) : ℝ :=
  total_distance_2 / crossing_time_platform_2

def total_distance_1 (train_length platform_1_length : ℝ) : ℝ :=
  train_length + platform_1_length

def crossing_time_platform_1 (total_distance_1 train_speed : ℝ) : ℝ :=
  total_distance_1 / train_speed

theorem time_to_cross_first_platform :
  crossing_time_platform_1 (total_distance_1 train_length platform_1_length)
                           (train_speed (total_distance_2 train_length platform_2_length)
                                        crossing_time_platform_2) 
  = 15 :=
by
  -- We would prove this in a detailed proof which is omitted here.
  sorry

end NUMINAMATH_GPT_time_to_cross_first_platform_l1936_193676


namespace NUMINAMATH_GPT_solve_system_eqns_l1936_193650

theorem solve_system_eqns (x y z a : ℝ)
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = a^2)
  (h3 : x^3 + y^3 + z^3 = a^3) :
  (x = a ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = a ∧ z = 0) ∨
  (x = 0 ∧ y = 0 ∧ z = a) := 
by
  sorry

end NUMINAMATH_GPT_solve_system_eqns_l1936_193650


namespace NUMINAMATH_GPT_no_prime_divisor_of_form_8k_minus_1_l1936_193603

theorem no_prime_divisor_of_form_8k_minus_1 (n : ℕ) (h : 0 < n) :
  ¬ ∃ p k : ℕ, Nat.Prime p ∧ p = 8 * k - 1 ∧ p ∣ (2^n + 1) :=
by
  sorry

end NUMINAMATH_GPT_no_prime_divisor_of_form_8k_minus_1_l1936_193603


namespace NUMINAMATH_GPT_contrapositive_proof_l1936_193613

theorem contrapositive_proof (a b : ℝ) : 
  (a > b → a - 1 > b - 1) ↔ (a - 1 ≤ b - 1 → a ≤ b) :=
sorry

end NUMINAMATH_GPT_contrapositive_proof_l1936_193613


namespace NUMINAMATH_GPT_find_n_l1936_193674

theorem find_n (k : ℤ) : 
  ∃ n : ℤ, (n = 35 * k + 24) ∧ (5 ∣ (3 * n - 2)) ∧ (7 ∣ (2 * n + 1)) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_n_l1936_193674


namespace NUMINAMATH_GPT_total_visitors_three_days_l1936_193611

def V_Rachel := 92
def V_prev_day := 419
def V_day_before_prev := 103

theorem total_visitors_three_days : V_Rachel + V_prev_day + V_day_before_prev = 614 := 
by sorry

end NUMINAMATH_GPT_total_visitors_three_days_l1936_193611


namespace NUMINAMATH_GPT_max_value_f_min_value_a_l1936_193671

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - 4 * Real.pi / 3) + 2 * (Real.cos x)^2

theorem max_value_f :
  ∀ x, f x ≤ 2 ∧ (∃ k : ℤ, x = k * Real.pi - Real.pi / 6) → f x = 2 :=
by { sorry }

variables {A B C a b c : ℝ}

noncomputable def f' (x : ℝ) : ℝ := Real.cos (2 * x +  Real.pi / 3) + 1

theorem min_value_a
  (h1 : f' (B + C) = 3/2)
  (h2 : b + c = 2)
  (h3 : A + B + C = Real.pi)
  (h4 : Real.cos A = 1/2) :
  ∃ a, ∀ b c, a^2 = b^2 + c^2 - 2 * b * c * Real.cos A ∧ a ≥ 1 :=
by { sorry }

end NUMINAMATH_GPT_max_value_f_min_value_a_l1936_193671


namespace NUMINAMATH_GPT_square_of_other_leg_l1936_193601

-- Conditions
variable (a b c : ℝ)
variable (h₁ : c = a + 2)
variable (h₂ : a^2 + b^2 = c^2)

-- The theorem statement
theorem square_of_other_leg (a b c : ℝ) (h₁ : c = a + 2) (h₂ : a^2 + b^2 = c^2) : b^2 = 4 * a + 4 :=
by
  sorry

end NUMINAMATH_GPT_square_of_other_leg_l1936_193601


namespace NUMINAMATH_GPT_benny_money_l1936_193607

-- Conditions
def cost_per_apple (cost : ℕ) := cost = 4
def apples_needed (apples : ℕ) := apples = 5 * 18

-- The proof problem
theorem benny_money (cost : ℕ) (apples : ℕ) (total_money : ℕ) :
  cost_per_apple cost → apples_needed apples → total_money = apples * cost → total_money = 360 :=
by
  intros h_cost h_apples h_total
  rw [h_cost, h_apples] at h_total
  exact h_total

end NUMINAMATH_GPT_benny_money_l1936_193607


namespace NUMINAMATH_GPT_who_threw_at_third_child_l1936_193647

-- Definitions based on conditions
def children_count : ℕ := 43

def threw_snowball (i j : ℕ) : Prop :=
∃ k, i = (k % children_count).succ ∧ j = ((k + 1) % children_count).succ

-- Conditions
axiom cond_1 : threw_snowball 1 (1 + 1) -- child 1 threw a snowball at the child who threw a snowball at child 2
axiom cond_2 : threw_snowball 2 (2 + 1) -- child 2 threw a snowball at the child who threw a snowball at child 3
axiom cond_3 : threw_snowball 43 1 -- child 43 threw a snowball at the child who threw a snowball at the first child

-- Question to prove
theorem who_threw_at_third_child : threw_snowball 24 3 :=
sorry

end NUMINAMATH_GPT_who_threw_at_third_child_l1936_193647


namespace NUMINAMATH_GPT_expression_evaluation_l1936_193688

theorem expression_evaluation : 
  (50 - (2210 - 251)) + (2210 - (251 - 50)) = 100 := 
  by sorry

end NUMINAMATH_GPT_expression_evaluation_l1936_193688


namespace NUMINAMATH_GPT_length_of_platform_is_correct_l1936_193614

-- Given conditions:
def length_of_train : ℕ := 250
def speed_of_train_kmph : ℕ := 72
def time_to_cross_platform : ℕ := 20

-- Convert speed from kmph to m/s
def speed_of_train_mps : ℕ := speed_of_train_kmph * 1000 / 3600

-- Distance covered in 20 seconds
def distance_covered : ℕ := speed_of_train_mps * time_to_cross_platform

-- Length of the platform
def length_of_platform : ℕ := distance_covered - length_of_train

-- The proof statement
theorem length_of_platform_is_correct :
  length_of_platform = 150 := by
  -- This proof would involve the detailed calculations and verifications as laid out in the solution steps.
  sorry

end NUMINAMATH_GPT_length_of_platform_is_correct_l1936_193614


namespace NUMINAMATH_GPT_triangles_with_positive_area_l1936_193657

-- Define the set of points in the coordinate grid
def points := { p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 4 ∧ 1 ≤ p.2 ∧ p.2 ≤ 4 }

-- Number of ways to choose 3 points from the grid
def total_triples := Nat.choose 16 3

-- Number of collinear triples
def collinear_triples := 32 + 8 + 4

-- Number of triangles with positive area
theorem triangles_with_positive_area :
  (total_triples - collinear_triples) = 516 :=
by
  -- Definitions for total_triples and collinear_triples.
  -- Proof steps would go here.
  sorry

end NUMINAMATH_GPT_triangles_with_positive_area_l1936_193657


namespace NUMINAMATH_GPT_arrange_abc_l1936_193638

theorem arrange_abc (a b c : ℝ) (h1 : a = Real.log 2 / Real.log 2)
                               (h2 : b = Real.sqrt 2)
                               (h3 : c = Real.cos ((3 / 4) * Real.pi)) :
  c < a ∧ a < b :=
by
  sorry

end NUMINAMATH_GPT_arrange_abc_l1936_193638


namespace NUMINAMATH_GPT_inequality_holds_equality_condition_l1936_193624

variables {x y z : ℝ}
-- Assuming positive real numbers and the given condition
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom pos_z : 0 < z
axiom h : x * y + y * z + z * x = x + y + z

theorem inequality_holds : 
  (1 / (x^2 + y + 1)) + (1 / (y^2 + z + 1)) + (1 / (z^2 + x + 1)) ≤ 1 :=
by
  sorry

theorem equality_condition : 
  (1 / (x^2 + y + 1)) + (1 / (y^2 + z + 1)) + (1 / (z^2 + x + 1)) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_equality_condition_l1936_193624


namespace NUMINAMATH_GPT_train_speed_correct_l1936_193656

/-- Define the length of the train in meters -/
def length_train : ℝ := 120

/-- Define the length of the bridge in meters -/
def length_bridge : ℝ := 160

/-- Define the time taken to pass the bridge in seconds -/
def time_taken : ℝ := 25.2

/-- Define the expected speed of the train in meters per second -/
def expected_speed : ℝ := 11.1111

/-- Prove that the speed of the train is 11.1111 meters per second given conditions -/
theorem train_speed_correct :
  (length_train + length_bridge) / time_taken = expected_speed :=
by
  sorry

end NUMINAMATH_GPT_train_speed_correct_l1936_193656


namespace NUMINAMATH_GPT_hyperbola_asymptote_slope_l1936_193683

theorem hyperbola_asymptote_slope :
  (∀ x y : ℝ, (x^2 / 100 - y^2 / 64 = 1) → y = (4/5) * x ∨ y = -(4/5) * x) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_slope_l1936_193683


namespace NUMINAMATH_GPT_train_john_arrival_probability_l1936_193659

-- Define the probability of independent uniform distributions on the interval [0, 120]
noncomputable def probability_train_present_when_john_arrives : ℝ :=
  let total_square_area := (120 : ℝ) * 120
  let triangle_area := (1 / 2) * 90 * 30
  let trapezoid_area := (1 / 2) * (30 + 0) * 30
  let total_shaded_area := triangle_area + trapezoid_area
  total_shaded_area / total_square_area

theorem train_john_arrival_probability :
  probability_train_present_when_john_arrives = 1 / 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_train_john_arrival_probability_l1936_193659


namespace NUMINAMATH_GPT_incorrect_statement_B_is_wrong_l1936_193623

variable (number_of_students : ℕ) (sample_size : ℕ) (population : Set ℕ) (sample : Set ℕ)

-- Conditions
def school_population_is_4000 := number_of_students = 4000
def sample_selected_is_400 := sample_size = 400
def valid_population := population = { x | x < 4000 }
def valid_sample := sample = { x | x < 400 }

-- Incorrect statement (as per given solution)
def incorrect_statement_B := ¬(∀ student ∈ population, true)

theorem incorrect_statement_B_is_wrong 
  (h1 : school_population_is_4000 number_of_students)
  (h2 : sample_selected_is_400 sample_size)
  (h3 : valid_population population)
  (h4 : valid_sample sample)
  : incorrect_statement_B population :=
sorry

end NUMINAMATH_GPT_incorrect_statement_B_is_wrong_l1936_193623


namespace NUMINAMATH_GPT_solve_rational_numbers_l1936_193686

theorem solve_rational_numbers:
  ∃ (a b c d : ℚ),
    8 * a^2 - 3 * b^2 + 5 * c^2 + 16 * d^2 - 10 * a * b + 42 * c * d + 18 * a + 22 * b - 2 * c - 54 * d = 42 ∧
    15 * a^2 - 3 * b^2 + 21 * c^2 - 5 * d^2 + 4 * a * b + 32 * c * d - 28 * a + 14 * b - 54 * c - 52 * d = -22 ∧
    a = 4 / 7 ∧ b = 19 / 7 ∧ c = 29 / 19 ∧ d = -6 / 19 :=
  sorry

end NUMINAMATH_GPT_solve_rational_numbers_l1936_193686


namespace NUMINAMATH_GPT_trigonometric_identity_l1936_193680

theorem trigonometric_identity (x y z : ℝ)
  (h1 : Real.cos x + Real.cos y + Real.cos z = 1)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 1) :
  Real.cos (2 * x) + Real.cos (2 * y) + 2 * Real.cos (2 * z) = 2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1936_193680


namespace NUMINAMATH_GPT_find_a_b_sum_pos_solution_l1936_193612

theorem find_a_b_sum_pos_solution :
  ∃ (a b : ℕ), (∃ (x : ℝ), x^2 + 16 * x = 100 ∧ x = Real.sqrt a - b) ∧ a + b = 172 :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_sum_pos_solution_l1936_193612


namespace NUMINAMATH_GPT_repair_cost_is_5000_l1936_193687

-- Define the initial cost of the machine
def initial_cost : ℝ := 9000

-- Define the transportation charges
def transportation_charges : ℝ := 1000

-- Define the selling price
def selling_price : ℝ := 22500

-- Define the profit percentage as a decimal
def profit_percentage : ℝ := 0.5

-- Define the total cost including repairs
def total_cost (repair_cost : ℝ) : ℝ :=
  initial_cost + transportation_charges + repair_cost

-- Define the equation for selling price with 50% profit
def selling_price_equation (repair_cost : ℝ) : Prop :=
  selling_price = (1 + profit_percentage) * total_cost repair_cost

-- State the proof problem in Lean
theorem repair_cost_is_5000 : selling_price_equation 5000 :=
by 
  sorry

end NUMINAMATH_GPT_repair_cost_is_5000_l1936_193687


namespace NUMINAMATH_GPT_diamond_eight_five_l1936_193678

def diamond (a b : ℕ) : ℕ := (a + b) * ((a - b) * (a - b))

theorem diamond_eight_five : diamond 8 5 = 117 := by
  sorry

end NUMINAMATH_GPT_diamond_eight_five_l1936_193678


namespace NUMINAMATH_GPT_ironed_clothing_l1936_193658

theorem ironed_clothing (shirts_rate pants_rate shirts_hours pants_hours : ℕ)
    (h1 : shirts_rate = 4)
    (h2 : pants_rate = 3)
    (h3 : shirts_hours = 3)
    (h4 : pants_hours = 5) :
    shirts_rate * shirts_hours + pants_rate * pants_hours = 27 := by
  sorry

end NUMINAMATH_GPT_ironed_clothing_l1936_193658


namespace NUMINAMATH_GPT_find_m_value_l1936_193649

theorem find_m_value :
  let x_values := [8, 9.5, m, 10.5, 12]
  let y_values := [16, 10, 8, 6, 5]
  let regression_eq (x : ℝ) := -3.5 * x + 44
  let avg (l : List ℝ) := l.sum / l.length
  avg y_values = 9 →
  avg x_values = (40 + m) / 5 →
  9 = regression_eq (avg x_values) →
  m = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l1936_193649


namespace NUMINAMATH_GPT_empty_plane_speed_l1936_193696

variable (V : ℝ)

def speed_first_plane (V : ℝ) : ℝ := V - 2 * 50
def speed_second_plane (V : ℝ) : ℝ := V - 2 * 60
def speed_third_plane (V : ℝ) : ℝ := V - 2 * 40

theorem empty_plane_speed (V : ℝ) (h : (speed_first_plane V + speed_second_plane V + speed_third_plane V) / 3 = 500) : V = 600 :=
by 
  sorry

end NUMINAMATH_GPT_empty_plane_speed_l1936_193696


namespace NUMINAMATH_GPT_irrational_number_problem_l1936_193627

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_number_problem :
  ∀ x ∈ ({(0.4 : ℝ), (2 / 3 : ℝ), (2 : ℝ), - (Real.sqrt 5)} : Set ℝ), 
  is_irrational x ↔ x = - (Real.sqrt 5) :=
by
  intros x hx
  -- Other proof steps can go here
  sorry

end NUMINAMATH_GPT_irrational_number_problem_l1936_193627


namespace NUMINAMATH_GPT_total_amount_invested_l1936_193653

-- Define the problem details: given conditions
def interest_rate_share1 : ℚ := 9 / 100
def interest_rate_share2 : ℚ := 11 / 100
def total_interest_rate : ℚ := 39 / 400
def amount_invested_share2 : ℚ := 3750

-- Define the total amount invested (A), the amount invested at the 9% share (x)
variable (A x : ℚ)

-- Conditions
axiom condition1 : x + amount_invested_share2 = A
axiom condition2 : interest_rate_share1 * x + interest_rate_share2 * amount_invested_share2 = total_interest_rate * A

-- Prove that the total amount invested in both types of shares is Rs. 10,000
theorem total_amount_invested : A = 10000 :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_total_amount_invested_l1936_193653


namespace NUMINAMATH_GPT_alice_minimum_speed_exceed_l1936_193690

-- Define the conditions

def distance_ab : ℕ := 30  -- Distance from city A to city B is 30 miles
def speed_bob : ℕ := 40    -- Bob's constant speed is 40 miles per hour
def bob_travel_time := distance_ab / speed_bob  -- Bob's travel time in hours
def alice_travel_time := bob_travel_time - (1 / 2)  -- Alice leaves 0.5 hours after Bob

-- Theorem stating the minimum speed Alice must exceed
theorem alice_minimum_speed_exceed : ∃ v : Real, v > 60 ∧ distance_ab / alice_travel_time ≤ v := sorry

end NUMINAMATH_GPT_alice_minimum_speed_exceed_l1936_193690


namespace NUMINAMATH_GPT_gas_usage_correct_l1936_193609

def starting_gas : ℝ := 0.5
def ending_gas : ℝ := 0.16666666666666666

theorem gas_usage_correct : starting_gas - ending_gas = 0.33333333333333334 := by
  sorry

end NUMINAMATH_GPT_gas_usage_correct_l1936_193609


namespace NUMINAMATH_GPT_elena_meeting_percentage_l1936_193610

noncomputable def workday_hours : ℕ := 10
noncomputable def first_meeting_duration_minutes : ℕ := 60
noncomputable def second_meeting_duration_minutes : ℕ := 3 * first_meeting_duration_minutes
noncomputable def total_workday_minutes := workday_hours * 60
noncomputable def total_meeting_minutes := first_meeting_duration_minutes + second_meeting_duration_minutes
noncomputable def percent_time_in_meetings := (total_meeting_minutes * 100) / total_workday_minutes

theorem elena_meeting_percentage : percent_time_in_meetings = 40 := by 
  sorry

end NUMINAMATH_GPT_elena_meeting_percentage_l1936_193610


namespace NUMINAMATH_GPT_truck_distance_l1936_193668

theorem truck_distance :
  let a1 := 8
  let d := 9
  let n := 40
  let an := a1 + (n - 1) * d
  let S_n := n / 2 * (a1 + an)
  S_n = 7340 :=
by
  sorry

end NUMINAMATH_GPT_truck_distance_l1936_193668


namespace NUMINAMATH_GPT_arithmetic_evaluation_l1936_193630

theorem arithmetic_evaluation : 6 * 2 - 3 = 9 := by
  sorry

end NUMINAMATH_GPT_arithmetic_evaluation_l1936_193630


namespace NUMINAMATH_GPT_solve_for_y_l1936_193608

-- Define the given condition as a Lean definition
def equation (y : ℝ) : Prop :=
  (2 / y) + ((3 / y) / (6 / y)) = 1.2

-- Theorem statement proving the solution given the condition
theorem solve_for_y (y : ℝ) (h : equation y) : y = 20 / 7 := by
  sorry

-- Example usage to instantiate and make use of the definition
example : equation (20 / 7) := by
  unfold equation
  sorry

end NUMINAMATH_GPT_solve_for_y_l1936_193608


namespace NUMINAMATH_GPT_minimum_value_expression_l1936_193606

theorem minimum_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ m, (∀ x y, x > 0 ∧ y > 0 → (x + y) * (1/x + 4/y) ≥ m) ∧ m = 9 :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l1936_193606


namespace NUMINAMATH_GPT_axis_of_symmetry_parabola_l1936_193632

theorem axis_of_symmetry_parabola (a b : ℝ) (h₁ : a = -3) (h₂ : b = 6) :
  -b / (2 * a) = 1 :=
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_parabola_l1936_193632


namespace NUMINAMATH_GPT_ellipse_area_l1936_193645

/-- 
In a certain ellipse, the endpoints of the major axis are (1, 6) and (21, 6). 
Also, the ellipse passes through the point (19, 9). Prove that the area of the ellipse is 50π. 
-/
theorem ellipse_area : 
  let a := 10
  let b := 5 
  let center := (11, 6)
  let endpoints_major := [(1, 6), (21, 6)]
  let point_on_ellipse := (19, 9)
  ∀ x y, ((x - 11)^2 / a^2) + ((y - 6)^2 / b^2) = 1 → 
    (x, y) = (19, 9) →  -- given point on the ellipse
    (endpoints_major = [(1, 6), (21, 6)]) →  -- given endpoints of the major axis
    50 * Real.pi = π * a * b := 
by
  sorry

end NUMINAMATH_GPT_ellipse_area_l1936_193645


namespace NUMINAMATH_GPT_correct_choice_C_l1936_193681

def geometric_sequence (n : ℕ) : ℕ := 
  2^(n - 1)

def sum_geometric_sequence (n : ℕ) : ℕ := 
  2^n - 1

theorem correct_choice_C (n : ℕ) (h : 0 < n) : sum_geometric_sequence n < geometric_sequence (n + 1) := by
  sorry

end NUMINAMATH_GPT_correct_choice_C_l1936_193681


namespace NUMINAMATH_GPT_calculate_sum_l1936_193617

theorem calculate_sum : (2 / 20) + (3 / 50 * 5 / 100) + (4 / 1000) + (6 / 10000) = 0.1076 := 
by
  sorry

end NUMINAMATH_GPT_calculate_sum_l1936_193617


namespace NUMINAMATH_GPT_stella_profit_l1936_193639

def price_of_doll := 5
def price_of_clock := 15
def price_of_glass := 4

def number_of_dolls := 3
def number_of_clocks := 2
def number_of_glasses := 5

def cost := 40

def dolls_sales := number_of_dolls * price_of_doll
def clocks_sales := number_of_clocks * price_of_clock
def glasses_sales := number_of_glasses * price_of_glass

def total_sales := dolls_sales + clocks_sales + glasses_sales

def profit := total_sales - cost

theorem stella_profit : profit = 25 :=
by 
  sorry

end NUMINAMATH_GPT_stella_profit_l1936_193639


namespace NUMINAMATH_GPT_base_b_three_digit_count_l1936_193652

-- Define the condition that counts the valid three-digit numbers in base b
def num_three_digit_numbers (b : ℕ) : ℕ :=
  (b - 1) ^ 2 * b

-- Define the specific problem statement
theorem base_b_three_digit_count :
  num_three_digit_numbers 4 = 72 :=
by
  -- Proof skipped as per the instruction
  sorry

end NUMINAMATH_GPT_base_b_three_digit_count_l1936_193652


namespace NUMINAMATH_GPT_opposite_number_of_neg_two_reciprocal_of_three_abs_val_three_eq_l1936_193682

theorem opposite_number_of_neg_two (a : Int) (h : a = -2) :
  -a = 2 := by
  sorry

theorem reciprocal_of_three (x y : Real) (hx : x = 3) (hy : y = 1 / 3) : 
  x * y = 1 := by
  sorry

theorem abs_val_three_eq (x : Real) (hx : abs x = 3) :
  x = -3 ∨ x = 3 := by
  sorry

end NUMINAMATH_GPT_opposite_number_of_neg_two_reciprocal_of_three_abs_val_three_eq_l1936_193682


namespace NUMINAMATH_GPT_general_term_arithmetic_sequence_sum_terms_sequence_l1936_193637

noncomputable def a_n (n : ℕ) : ℤ := 
  2 * (n : ℤ) - 1

theorem general_term_arithmetic_sequence :
  ∀ n : ℕ, a_n n = 2 * (n : ℤ) - 1 :=
by sorry

noncomputable def c (n : ℕ) : ℚ := 
  1 / ((2 * (n : ℤ) - 1) * (2 * (n + 1) - 1))

noncomputable def T_n (n : ℕ) : ℚ :=
  (1 / 2 : ℚ) * (1 - (1 / (2 * (n : ℤ) + 1)))

theorem sum_terms_sequence :
  ∀ n : ℕ, T_n n = (n : ℚ) / (2 * (n : ℤ) + 1) :=
by sorry

end NUMINAMATH_GPT_general_term_arithmetic_sequence_sum_terms_sequence_l1936_193637


namespace NUMINAMATH_GPT_time_spent_moving_l1936_193673

noncomputable def time_per_trip_filling : ℝ := 15
noncomputable def time_per_trip_driving : ℝ := 30
noncomputable def time_per_trip_unloading : ℝ := 20
noncomputable def number_of_trips : ℕ := 10

theorem time_spent_moving :
  10.83 = (time_per_trip_filling + time_per_trip_driving + time_per_trip_unloading) * number_of_trips / 60 :=
by
  sorry

end NUMINAMATH_GPT_time_spent_moving_l1936_193673


namespace NUMINAMATH_GPT_number_of_numbers_is_11_l1936_193677

noncomputable def total_number_of_numbers 
  (avg_all : ℝ) (avg_first_6 : ℝ) (avg_last_6 : ℝ) (num_6th : ℝ) : ℝ :=
if h : avg_all = 60 ∧ avg_first_6 = 58 ∧ avg_last_6 = 65 ∧ num_6th = 78 
then 11 else 0 

-- The theorem statement assuming the problem conditions
theorem number_of_numbers_is_11
  {n S : ℝ}
  (avg_all : ℝ) (avg_first_6 : ℝ) (avg_last_6 : ℝ) (num_6th : ℝ) 
  (h1 : avg_all = 60) 
  (h2 : avg_first_6 = 58)
  (h3 : avg_last_6 = 65)
  (h4 : num_6th = 78) 
  (h5 : S = 6 * avg_first_6 + 6 * avg_last_6 - num_6th)
  (h6 : S = avg_all * n) : 
  n = 11 := sorry

end NUMINAMATH_GPT_number_of_numbers_is_11_l1936_193677


namespace NUMINAMATH_GPT_graduating_class_total_l1936_193629

theorem graduating_class_total (boys girls : ℕ) 
  (h_boys : boys = 138)
  (h_more_girls : girls = boys + 69) :
  boys + girls = 345 :=
sorry

end NUMINAMATH_GPT_graduating_class_total_l1936_193629


namespace NUMINAMATH_GPT_factorization_of_cubic_polynomial_l1936_193664

theorem factorization_of_cubic_polynomial (x y z : ℝ) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = (x + y + z) * (x^2 + y^2 + z^2 - x * y - y * z - z * x) := 
by sorry

end NUMINAMATH_GPT_factorization_of_cubic_polynomial_l1936_193664


namespace NUMINAMATH_GPT_david_chemistry_marks_l1936_193698

theorem david_chemistry_marks :
  let english := 96
  let mathematics := 95
  let physics := 82
  let biology := 95
  let average_all := 93
  let total_subjects := 5
  let total_other_subjects := english + mathematics + physics + biology
  let total_all_subjects := average_all * total_subjects
  let chemistry := total_all_subjects - total_other_subjects
  chemistry = 97 :=
by
  -- Definition of variables
  let english := 96
  let mathematics := 95
  let physics := 82
  let biology := 95
  let average_all := 93
  let total_subjects := 5
  let total_other_subjects := english + mathematics + physics + biology
  let total_all_subjects := average_all * total_subjects
  let chemistry := total_all_subjects - total_other_subjects

  -- Assert the final value
  show chemistry = 97
  sorry

end NUMINAMATH_GPT_david_chemistry_marks_l1936_193698


namespace NUMINAMATH_GPT_sam_spent_136_96_l1936_193654

def glove_original : Real := 35
def glove_discount : Real := 0.20
def baseball_price : Real := 15
def bat_original : Real := 50
def bat_discount : Real := 0.10
def cleats_price : Real := 30
def cap_price : Real := 10
def tax_rate : Real := 0.07

def total_spent (glove_original : Real) (glove_discount : Real) (baseball_price : Real) (bat_original : Real) (bat_discount : Real) (cleats_price : Real) (cap_price : Real) (tax_rate : Real) : Real :=
  let glove_price := glove_original - (glove_discount * glove_original)
  let bat_price := bat_original - (bat_discount * bat_original)
  let total_before_tax := glove_price + baseball_price + bat_price + cleats_price + cap_price
  let tax_amount := total_before_tax * tax_rate
  total_before_tax + tax_amount

theorem sam_spent_136_96 :
  total_spent glove_original glove_discount baseball_price bat_original bat_discount cleats_price cap_price tax_rate = 136.96 :=
sorry

end NUMINAMATH_GPT_sam_spent_136_96_l1936_193654


namespace NUMINAMATH_GPT_luggage_max_length_l1936_193633

theorem luggage_max_length
  (l w h : ℕ)
  (h_eq : h = 30)
  (ratio_l_w : l = 3 * w / 2)
  (sum_leq : l + w + h ≤ 160) :
  l ≤ 78 := sorry

end NUMINAMATH_GPT_luggage_max_length_l1936_193633


namespace NUMINAMATH_GPT_journey_total_distance_l1936_193636

-- Define the conditions
def miles_already_driven : ℕ := 642
def miles_to_drive : ℕ := 558

-- The total distance of the journey
def total_distance : ℕ := miles_already_driven + miles_to_drive

-- Prove that the total distance of the journey equals 1200 miles
theorem journey_total_distance : total_distance = 1200 := 
by
  -- here the proof would go
  sorry

end NUMINAMATH_GPT_journey_total_distance_l1936_193636


namespace NUMINAMATH_GPT_ratio_apps_optimal_l1936_193615

theorem ratio_apps_optimal (max_apps : ℕ) (recommended_apps : ℕ) (apps_to_delete : ℕ) (current_apps : ℕ)
  (h_max_apps : max_apps = 50)
  (h_recommended_apps : recommended_apps = 35)
  (h_apps_to_delete : apps_to_delete = 20)
  (h_current_apps : current_apps = max_apps + apps_to_delete) :
  current_apps / recommended_apps = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_apps_optimal_l1936_193615


namespace NUMINAMATH_GPT_sum_sequence_eq_l1936_193626

noncomputable def S (n : ℕ) : ℝ := Real.log (1 + n) / Real.log 0.1

theorem sum_sequence_eq :
  (S 99 - S 9) = -1 := by
  sorry

end NUMINAMATH_GPT_sum_sequence_eq_l1936_193626


namespace NUMINAMATH_GPT_find_sale4_l1936_193670

variable (sale1 sale2 sale3 sale5 sale6 avg : ℕ)
variable (total_sales : ℕ := 6 * avg)
variable (known_sales : ℕ := sale1 + sale2 + sale3 + sale5 + sale6)
variable (sale4 : ℕ := total_sales - known_sales)

theorem find_sale4 (h1 : sale1 = 6235) (h2 : sale2 = 6927) (h3 : sale3 = 6855)
                   (h5 : sale5 = 6562) (h6 : sale6 = 5191) (h_avg : avg = 6500) :
  sale4 = 7225 :=
by 
  sorry

end NUMINAMATH_GPT_find_sale4_l1936_193670


namespace NUMINAMATH_GPT_side_length_of_square_base_l1936_193618

theorem side_length_of_square_base (area slant_height : ℝ) (h_area : area = 120) (h_slant_height : slant_height = 40) :
  ∃ s : ℝ, area = 0.5 * s * slant_height ∧ s = 6 :=
by
  sorry

end NUMINAMATH_GPT_side_length_of_square_base_l1936_193618


namespace NUMINAMATH_GPT_evaluate_f_of_f_of_3_l1936_193669

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 2

theorem evaluate_f_of_f_of_3 :
  f (f 3) = 2943 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_f_of_f_of_3_l1936_193669


namespace NUMINAMATH_GPT_maria_profit_disks_l1936_193694

theorem maria_profit_disks (cost_price_per_5 : ℝ) (sell_price_per_4 : ℝ) (desired_profit : ℝ) : 
  (cost_price_per_5 = 6) → (sell_price_per_4 = 8) → (desired_profit = 120) →
  (150 : ℝ) = desired_profit / ((sell_price_per_4 / 4) - (cost_price_per_5 / 5)) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  done

end NUMINAMATH_GPT_maria_profit_disks_l1936_193694


namespace NUMINAMATH_GPT_remainder_when_sum_divided_by_7_l1936_193642

theorem remainder_when_sum_divided_by_7 (a b c : ℕ) (ha : a < 7) (hb : b < 7) (hc : c < 7)
  (h1 : a * b * c ≡ 1 [MOD 7])
  (h2 : 4 * c ≡ 3 [MOD 7])
  (h3 : 5 * b ≡ 4 + b [MOD 7]) :
  (a + b + c) % 7 = 6 := by
  sorry

end NUMINAMATH_GPT_remainder_when_sum_divided_by_7_l1936_193642


namespace NUMINAMATH_GPT_total_fare_for_20km_l1936_193648

def base_fare : ℝ := 8
def fare_per_km_from_3_to_10 : ℝ := 1.5
def fare_per_km_beyond_10 : ℝ := 0.8

def fare_for_first_3km : ℝ := base_fare
def fare_for_3_to_10_km : ℝ := 7 * fare_per_km_from_3_to_10
def fare_for_beyond_10_km : ℝ := 10 * fare_per_km_beyond_10

theorem total_fare_for_20km : fare_for_first_3km + fare_for_3_to_10_km + fare_for_beyond_10_km = 26.5 :=
by
  sorry

end NUMINAMATH_GPT_total_fare_for_20km_l1936_193648


namespace NUMINAMATH_GPT_question1_question2_question3_l1936_193699

def f : Nat → Nat → Nat := sorry

axiom condition1 : f 1 1 = 1
axiom condition2 : ∀ m n, f m (n + 1) = f m n + 2
axiom condition3 : ∀ m, f (m + 1) 1 = 2 * f m 1

theorem question1 (n : Nat) : f 1 n = 2 * n - 1 :=
sorry

theorem question2 (m : Nat) : f m 1 = 2 ^ (m - 1) :=
sorry

theorem question3 : f 2002 9 = 2 ^ 2001 + 16 :=
sorry

end NUMINAMATH_GPT_question1_question2_question3_l1936_193699


namespace NUMINAMATH_GPT_cyclic_inequality_l1936_193631

theorem cyclic_inequality
    (x1 x2 x3 x4 x5 : ℝ)
    (h1 : 0 < x1)
    (h2 : 0 < x2)
    (h3 : 0 < x3)
    (h4 : 0 < x4)
    (h5 : 0 < x5) :
    (x1 + x2 + x3 + x4 + x5)^2 > 4 * (x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x1) :=
by
  sorry

end NUMINAMATH_GPT_cyclic_inequality_l1936_193631


namespace NUMINAMATH_GPT_problem_equivalent_l1936_193679

-- Define the problem conditions
def an (n : ℕ) : ℤ := -4 * n + 2

-- Arithmetic sequence: given conditions
axiom arith_seq_cond1 : an 2 + an 7 = -32
axiom arith_seq_cond2 : an 3 + an 8 = -40

-- Suppose the sequence {an + bn} is geometric with first term 1 and common ratio 2
def geom_seq (n : ℕ) : ℤ := 2 ^ (n - 1)
def bn (n : ℕ) : ℤ := geom_seq n - an n

-- To prove: sum of the first n terms of {bn}, denoted as Sn
def Sn (n : ℕ) : ℤ := (n * (2 + 4 * n - 2)) / 2 + (1 - 2 ^ n) / (1 - 2)

theorem problem_equivalent (n : ℕ) :
  an 2 + an 7 = -32 ∧
  an 3 + an 8 = -40 ∧
  (∀ n : ℕ, an n + bn n = geom_seq n) →
  Sn n = 2 * n ^ 2 + 2 ^ n - 1 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_problem_equivalent_l1936_193679


namespace NUMINAMATH_GPT_total_animals_count_l1936_193616

theorem total_animals_count (a m : ℕ) (h1 : a = 35) (h2 : a + 7 = m) : a + m = 77 :=
by
  sorry

end NUMINAMATH_GPT_total_animals_count_l1936_193616


namespace NUMINAMATH_GPT_range_of_a_l1936_193634

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ (a < -4 ∨ a > 4) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1936_193634


namespace NUMINAMATH_GPT_solve_inequality_l1936_193625

-- Define the inequality
def inequality (x : ℝ) : Prop :=
  -3 * (x^2 - 4 * x + 16) * (x^2 + 6 * x + 8) / ((x^3 + 64) * (Real.sqrt (x^2 + 4 * x + 4))) ≤ x^2 + x - 3

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  x ∈ Set.Iic (-4) ∪ {x : ℝ | -4 < x ∧ x ≤ -3} ∪ {x : ℝ | -2 < x ∧ x ≤ -1} ∪ Set.Ici 0

-- The theorem statement, which we need to prove
theorem solve_inequality : ∀ x : ℝ, inequality x ↔ solution_set x :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_inequality_l1936_193625


namespace NUMINAMATH_GPT_area_of_PQRS_l1936_193643

noncomputable def length_EF := 6
noncomputable def width_EF := 4

noncomputable def area_PQRS := (length_EF + 6 * Real.sqrt 3) * (width_EF + 4 * Real.sqrt 3)

theorem area_of_PQRS :
  area_PQRS = 60 + 48 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_area_of_PQRS_l1936_193643


namespace NUMINAMATH_GPT_sugar_cups_l1936_193662

theorem sugar_cups (S : ℕ) (h1 : 21 = S + 8) : S = 13 := 
by { sorry }

end NUMINAMATH_GPT_sugar_cups_l1936_193662


namespace NUMINAMATH_GPT_perpendicular_bisector_l1936_193641

theorem perpendicular_bisector (x y : ℝ) (h : -1 ≤ x ∧ x ≤ 3) (h_line : x - 2 * y + 1 = 0) : 
  2 * x - y - 1 = 0 :=
sorry

end NUMINAMATH_GPT_perpendicular_bisector_l1936_193641


namespace NUMINAMATH_GPT_cylinder_volume_ratio_l1936_193660

theorem cylinder_volume_ratio (s : ℝ) :
  let r := s / 2
  let h := s
  let V_cylinder := π * r^2 * h
  let V_cube := s^3
  V_cylinder / V_cube = π / 4 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_ratio_l1936_193660


namespace NUMINAMATH_GPT_liquid_levels_proof_l1936_193640

noncomputable def liquid_levels (H : ℝ) : ℝ × ℝ :=
  let ρ_water := 1000
  let ρ_gasoline := 600
  -- x = level drop in the left vessel
  let x := (3 / 14) * H
  let h_left := 0.9 * H - x
  let h_right := H
  (h_left, h_right)

theorem liquid_levels_proof (H : ℝ) (h : ℝ) :
  H > 0 →
  h = 0.9 * H →
  liquid_levels H = (0.69 * H, H) :=
by
  intros
  sorry

end NUMINAMATH_GPT_liquid_levels_proof_l1936_193640


namespace NUMINAMATH_GPT_sum_and_product_of_conjugates_l1936_193675

theorem sum_and_product_of_conjugates (c d : ℚ) 
  (h1 : 2 * c = 6)
  (h2 : c^2 - 4 * d = 4) :
  c + d = 17 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_and_product_of_conjugates_l1936_193675


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1936_193689

theorem sufficient_but_not_necessary (x : ℝ) : (x < -1) → (x < -1 ∨ x > 1) ∧ ¬((x < -1 ∨ x > 1) → (x < -1)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1936_193689


namespace NUMINAMATH_GPT_rectangle_perimeter_l1936_193605

theorem rectangle_perimeter (s : ℝ) (h1 : 4 * s = 180) :
    let length := s
    let width := s / 3
    2 * (length + width) = 120 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1936_193605


namespace NUMINAMATH_GPT_pond_eye_count_l1936_193619

def total_animal_eyes (snakes alligators spiders snails : ℕ) 
    (snake_eyes alligator_eyes spider_eyes snail_eyes: ℕ) : ℕ :=
  snakes * snake_eyes + alligators * alligator_eyes + spiders * spider_eyes + snails * snail_eyes

theorem pond_eye_count : total_animal_eyes 18 10 5 15 2 2 8 2 = 126 := 
by
  sorry

end NUMINAMATH_GPT_pond_eye_count_l1936_193619


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1936_193692

noncomputable def expression (x : ℤ) : ℤ :=
  ( (-2 * x^3 - 6 * x) / (-2 * x) - 2 * (3 * x + 1) * (3 * x - 1) + 7 * x * (x - 1) )

theorem simplify_and_evaluate : 
  (expression (-3) = -64) := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1936_193692


namespace NUMINAMATH_GPT_subset_definition_l1936_193663

variable {α : Type} {A B : Set α}

theorem subset_definition :
  A ⊆ B ↔ ∀ a ∈ A, a ∈ B :=
by sorry

end NUMINAMATH_GPT_subset_definition_l1936_193663


namespace NUMINAMATH_GPT_time_to_walk_against_walkway_150_l1936_193655

def v_p := 4 / 3
def v_w := 2 - v_p
def distance := 100
def time_against_walkway := distance / (v_p - v_w)

theorem time_to_walk_against_walkway_150 :
  time_against_walkway = 150 := by
  -- Note: Proof goes here (not required)
  sorry

end NUMINAMATH_GPT_time_to_walk_against_walkway_150_l1936_193655


namespace NUMINAMATH_GPT_solution_set_of_f_greater_than_one_l1936_193685

theorem solution_set_of_f_greater_than_one (f : ℝ → ℝ) (h_inv : ∀ x, f (x / (x + 3)) = x) :
  {x | f x > 1} = {x | 1 / 4 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_f_greater_than_one_l1936_193685


namespace NUMINAMATH_GPT_quadrilateral_area_lemma_l1936_193695

-- Define the coordinates of the vertices
structure Point where
  x : ℤ
  y : ℤ

def A : Point := ⟨1, 3⟩
def B : Point := ⟨1, 1⟩
def C : Point := ⟨2, 1⟩
def D : Point := ⟨2006, 2007⟩

-- Function to calculate the area of a quadrilateral given its vertices
def quadrilateral_area (A B C D : Point) : ℤ := 
  let triangle_area (P Q R : Point) : ℤ :=
    (Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x) / 2
  triangle_area A B C + triangle_area A C D

-- The statement to be proved
theorem quadrilateral_area_lemma : quadrilateral_area A B C D = 3008 := 
  sorry

end NUMINAMATH_GPT_quadrilateral_area_lemma_l1936_193695


namespace NUMINAMATH_GPT_range_of_sqrt_x_minus_1_meaningful_l1936_193628

theorem range_of_sqrt_x_minus_1_meaningful (x : ℝ) (h : 0 ≤ x - 1) : 1 ≤ x := 
sorry

end NUMINAMATH_GPT_range_of_sqrt_x_minus_1_meaningful_l1936_193628


namespace NUMINAMATH_GPT_rate_percent_correct_l1936_193604

noncomputable def findRatePercent (P A T : ℕ) : ℚ :=
  let SI := A - P
  (SI * 100 : ℚ) / (P * T)

theorem rate_percent_correct :
  findRatePercent 12000 19500 7 = 8.93 := by
  sorry

end NUMINAMATH_GPT_rate_percent_correct_l1936_193604


namespace NUMINAMATH_GPT_smallest_number_condition_l1936_193620

def smallest_number := 1621432330
def primes := [29, 53, 37, 41, 47, 61]
def lcm_of_primes := primes.prod

theorem smallest_number_condition :
  ∃ k : ℕ, 5 * (smallest_number + 11) = k * lcm_of_primes ∧
          (∀ y, (∃ m : ℕ, 5 * (y + 11) = m * lcm_of_primes) → smallest_number ≤ y) :=
by
  -- The proof goes here
  sorry

#print smallest_number_condition

end NUMINAMATH_GPT_smallest_number_condition_l1936_193620


namespace NUMINAMATH_GPT_find_a_given_integer_roots_l1936_193635

-- Given polynomial equation and the condition of integer roots
theorem find_a_given_integer_roots (a : ℤ) :
    (∃ x y : ℤ, x ≠ y ∧ (x^2 - (a+8)*x + 8*a - 1 = 0) ∧ (y^2 - (a+8)*y + 8*a - 1 = 0)) → 
    a = 8 := 
by
  sorry

end NUMINAMATH_GPT_find_a_given_integer_roots_l1936_193635


namespace NUMINAMATH_GPT_total_games_played_l1936_193697

-- Define the number of teams and games per matchup condition
def num_teams : ℕ := 10
def games_per_matchup : ℕ := 5

-- Calculate total games played during the season
theorem total_games_played : 
  5 * ((num_teams * (num_teams - 1)) / 2) = 225 := by 
  sorry

end NUMINAMATH_GPT_total_games_played_l1936_193697


namespace NUMINAMATH_GPT_prime_numbers_count_and_sum_l1936_193602

-- Definition of prime numbers less than or equal to 20
def prime_numbers_leq_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Proposition stating the number of prime numbers and their sum within 20
theorem prime_numbers_count_and_sum :
  (prime_numbers_leq_20.length = 8) ∧ (prime_numbers_leq_20.sum = 77) := by
  sorry

end NUMINAMATH_GPT_prime_numbers_count_and_sum_l1936_193602


namespace NUMINAMATH_GPT_y_intercept_of_line_l1936_193622

theorem y_intercept_of_line : 
  (∃ t : ℝ, 4 - 4 * t = 0) → (∃ y : ℝ, y = -2 + 3 * 1) := 
by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1936_193622


namespace NUMINAMATH_GPT_find_x_l1936_193644

theorem find_x (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y + x * y = 152) : x = 16 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l1936_193644


namespace NUMINAMATH_GPT_fraction_power_seven_l1936_193661

theorem fraction_power_seven : (5 / 3 : ℚ) ^ 7 = 78125 / 2187 := 
by
  sorry

end NUMINAMATH_GPT_fraction_power_seven_l1936_193661


namespace NUMINAMATH_GPT_alice_commission_percentage_l1936_193693

-- Definitions from the given problem
def basic_salary : ℝ := 240
def total_sales : ℝ := 2500
def savings : ℝ := 29
def savings_percentage : ℝ := 0.10

-- The target percentage we want to prove
def commission_percentage : ℝ := 0.02

-- The statement we aim to prove
theorem alice_commission_percentage :
  commission_percentage =
  (savings / savings_percentage - basic_salary) / total_sales := 
sorry

end NUMINAMATH_GPT_alice_commission_percentage_l1936_193693


namespace NUMINAMATH_GPT_x_intercept_of_line_l1936_193600

variables (x₁ y₁ x₂ y₂ : ℝ) (m : ℝ)

/-- The line passing through the points (-1, 1) and (3, 9) has an x-intercept of -3/2. -/
theorem x_intercept_of_line : 
  let x₁ := -1
  let y₁ := 1
  let x₂ := 3
  let y₂ := 9
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (0 : ℝ) = m * (x : ℝ) + b → x = (-3 / 2) := 
by 
  sorry

end NUMINAMATH_GPT_x_intercept_of_line_l1936_193600


namespace NUMINAMATH_GPT_graph1_higher_than_graph2_l1936_193621

theorem graph1_higher_than_graph2 :
  ∀ (x : ℝ), (-x^2 + 2 * x + 3) ≥ (x^2 - 2 * x + 3) :=
by
  intros x
  sorry

end NUMINAMATH_GPT_graph1_higher_than_graph2_l1936_193621


namespace NUMINAMATH_GPT_boys_running_speed_l1936_193684
-- Import the necessary libraries

-- Define the input conditions:
def side_length : ℝ := 50
def time_seconds : ℝ := 80
def conversion_factor_meters_to_kilometers : ℝ := 1000
def conversion_factor_seconds_to_hours : ℝ := 3600

-- Define the theorem:
theorem boys_running_speed :
  let perimeter := 4 * side_length
  let distance_kilometers := perimeter / conversion_factor_meters_to_kilometers
  let time_hours := time_seconds / conversion_factor_seconds_to_hours
  distance_kilometers / time_hours = 9 :=
by
  sorry

end NUMINAMATH_GPT_boys_running_speed_l1936_193684


namespace NUMINAMATH_GPT_best_store_is_A_l1936_193665

/-- Problem conditions -/
def price_per_ball : Nat := 25
def balls_to_buy : Nat := 58

/-- Store A conditions -/
def balls_bought_per_offer_A : Nat := 10
def balls_free_per_offer_A : Nat := 3

/-- Store B conditions -/
def discount_per_ball_B : Nat := 5

/-- Store C conditions -/
def cashback_rate_C : Nat := 40
def cashback_threshold_C : Nat := 200

/-- Cost calculations -/
def cost_store_A (total_balls : Nat) (price : Nat) : Nat :=
  let full_offers := total_balls / balls_bought_per_offer_A
  let remaining_balls := total_balls % balls_bought_per_offer_A
  let balls_paid_for := full_offers * (balls_bought_per_offer_A - balls_free_per_offer_A) + remaining_balls
  balls_paid_for * price

def cost_store_B (total_balls : Nat) (price : Nat) (discount : Nat) : Nat :=
  total_balls * (price - discount)

def cost_store_C (total_balls : Nat) (price : Nat) (cashback_rate : Nat) (threshold : Nat) : Nat :=
  let cost_before_cashback := total_balls * price
  let full_cashbacks := cost_before_cashback / threshold
  let cashback_amount := full_cashbacks * cashback_rate
  cost_before_cashback - cashback_amount

theorem best_store_is_A :
  cost_store_A balls_to_buy price_per_ball = 1075 ∧
  cost_store_B balls_to_buy price_per_ball discount_per_ball_B = 1160 ∧
  cost_store_C balls_to_buy price_per_ball cashback_rate_C cashback_threshold_C = 1170 ∧
  cost_store_A balls_to_buy price_per_ball < cost_store_B balls_to_buy price_per_ball discount_per_ball_B ∧
  cost_store_A balls_to_buy price_per_ball < cost_store_C balls_to_buy price_per_ball cashback_rate_C cashback_threshold_C :=
by {
  -- placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_best_store_is_A_l1936_193665


namespace NUMINAMATH_GPT_congruent_rectangle_perimeter_l1936_193691

theorem congruent_rectangle_perimeter (x y w l P : ℝ) 
  (h1 : x + 2 * w = 2 * y) 
  (h2 : x + 2 * l = y) 
  (hP : P = 2 * l + 2 * w) : 
  P = 3 * y - 2 * x :=
by sorry

end NUMINAMATH_GPT_congruent_rectangle_perimeter_l1936_193691
