import Mathlib

namespace concert_ticket_cost_l2037_203743

-- Definitions based on the conditions
def hourlyWage : ℝ := 18
def hoursPerWeek : ℝ := 30
def drinkTicketCost : ℝ := 7
def numberOfDrinkTickets : ℝ := 5
def outingPercentage : ℝ := 0.10
def weeksPerMonth : ℝ := 4

-- Proof statement
theorem concert_ticket_cost (hourlyWage hoursPerWeek drinkTicketCost numberOfDrinkTickets outingPercentage weeksPerMonth : ℝ)
  (monthlySalary := weeksPerMonth * (hoursPerWeek * hourlyWage))
  (outingAmount := outingPercentage * monthlySalary)
  (costOfDrinkTickets := numberOfDrinkTickets * drinkTicketCost)
  (costOfConcertTicket := outingAmount - costOfDrinkTickets)
  : costOfConcertTicket = 181 := 
sorry

end concert_ticket_cost_l2037_203743


namespace max_quadratic_function_l2037_203786

theorem max_quadratic_function :
  ∃ M, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → (x^2 - 2*x - 1 ≤ M)) ∧
       (∀ y : ℝ, y = (x : ℝ) ^ 2 - 2 * x - 1 → x = 3 → y = M) :=
by
  use 2
  sorry

end max_quadratic_function_l2037_203786


namespace problem_solution_l2037_203746

noncomputable def otimes (a b : ℝ) : ℝ := (a^3) / b

theorem problem_solution :
  (otimes (otimes 2 3) 4) - (otimes 2 (otimes 3 4)) = (32/9) :=
by
  sorry

end problem_solution_l2037_203746


namespace difference_divisible_by_18_l2037_203780

theorem difference_divisible_by_18 (a b : ℤ) : 18 ∣ ((3 * a + 2) ^ 2 - (3 * b + 2) ^ 2) :=
by
  sorry

end difference_divisible_by_18_l2037_203780


namespace ben_points_l2037_203721

theorem ben_points (zach_points : ℝ) (total_points : ℝ) (ben_points : ℝ) 
  (h1 : zach_points = 42.0) 
  (h2 : total_points = 63) 
  (h3 : total_points = zach_points + ben_points) : 
  ben_points = 21 :=
by
  sorry

end ben_points_l2037_203721


namespace value_of_Priyanka_l2037_203732

-- Defining the context with the conditions
variables (X : ℕ) (Neha : ℕ) (Sonali Priyanka Sadaf Tanu : ℕ)
-- The conditions given in the problem
axiom h1 : Neha = X
axiom h2 : Sonali = 15
axiom h3 : Priyanka = 15
axiom h4 : Sadaf = Neha
axiom h5 : Tanu = Neha

-- Stating the theorem we need to prove
theorem value_of_Priyanka : Priyanka = 15 :=
by
  sorry

end value_of_Priyanka_l2037_203732


namespace solution_set_inequality_l2037_203770

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition1 : f 1 = 1
axiom f_condition2 : ∀ x : ℝ, deriv f x < 1 / 2

theorem solution_set_inequality : {x : ℝ | 0 < x ∧ x < 2} = {x : ℝ | f (Real.log x / Real.log 2) > (Real.log x / Real.log 2 + 1) / 2} :=
by
  sorry

end solution_set_inequality_l2037_203770


namespace expression_value_l2037_203767

theorem expression_value (x : ℝ) (h : x = 3) : x^4 - 4 * x^2 = 45 := by
  sorry

end expression_value_l2037_203767


namespace value_of_expression_l2037_203741

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) : 
  a * b / (c * d) = 180 :=
by
  sorry

end value_of_expression_l2037_203741


namespace unique_three_digit_numbers_unique_three_digit_odd_numbers_l2037_203793

def num_digits: ℕ := 10

theorem unique_three_digit_numbers (num_digits : ℕ) :
  ∃ n : ℕ, n = 648 ∧ n = (num_digits - 1) * (num_digits - 1) * (num_digits - 2) + 2 * (num_digits - 1) * (num_digits - 1) :=
  sorry

theorem unique_three_digit_odd_numbers (num_digits : ℕ) :
  ∃ n : ℕ, n = 320 ∧ ∀ odd_digit_nums : ℕ, odd_digit_nums ≥ 1 → odd_digit_nums = 5 → 
  n = odd_digit_nums * (num_digits - 2) * (num_digits - 2) :=
  sorry

end unique_three_digit_numbers_unique_three_digit_odd_numbers_l2037_203793


namespace find_s_l2037_203733

def is_monic_cubic (p : Polynomial ℝ) : Prop :=
  p.degree = 3 ∧ p.leadingCoeff = 1

def has_roots (p : Polynomial ℝ) (roots : Set ℝ) : Prop :=
  ∀ x ∈ roots, p.eval x = 0

def poly_condition (f g : Polynomial ℝ) (s : ℝ) : Prop :=
  ∀ x : ℝ, f.eval x - g.eval x = 2 * s

theorem find_s (s : ℝ)
  (f g : Polynomial ℝ)
  (hf_monic : is_monic_cubic f)
  (hg_monic : is_monic_cubic g)
  (hf_roots : has_roots f {s + 2, s + 6})
  (hg_roots : has_roots g {s + 4, s + 10})
  (h_condition : poly_condition f g s) :
  s = 10.67 :=
sorry

end find_s_l2037_203733


namespace train_length_l2037_203772

noncomputable def length_of_train (time_sec : ℕ) (speed_kmh : ℝ) : ℝ :=
  (speed_kmh * 1000 / 3600) * time_sec

theorem train_length (h_time : 21 = 21) (h_speed : 75.6 = 75.6) :
  length_of_train 21 75.6 = 441 :=
by
  sorry

end train_length_l2037_203772


namespace cost_per_box_of_cookies_l2037_203766

-- Given conditions
def initial_money : ℝ := 20
def mother_gift : ℝ := 2 * initial_money
def total_money : ℝ := initial_money + mother_gift
def cupcake_price : ℝ := 1.50
def num_cupcakes : ℝ := 10
def cost_cupcakes : ℝ := num_cupcakes * cupcake_price
def money_after_cupcakes : ℝ := total_money - cost_cupcakes
def remaining_money : ℝ := 30
def num_boxes_cookies : ℝ := 5
def money_spent_on_cookies : ℝ := money_after_cupcakes - remaining_money

-- Theorem: Calculate the cost per box of cookies
theorem cost_per_box_of_cookies : (money_spent_on_cookies / num_boxes_cookies) = 3 :=
by
  sorry

end cost_per_box_of_cookies_l2037_203766


namespace calculate_percentage_passed_l2037_203750

theorem calculate_percentage_passed (F_H F_E F_HE : ℝ) (h1 : F_H = 0.32) (h2 : F_E = 0.56) (h3 : F_HE = 0.12) :
  1 - (F_H + F_E - F_HE) = 0.24 := by
  sorry

end calculate_percentage_passed_l2037_203750


namespace circle_equation_standard_l2037_203713

def center : ℝ × ℝ := (-1, 1)
def radius : ℝ := 2

theorem circle_equation_standard:
  (∀ x y : ℝ, ((x + 1)^2 + (y - 1)^2 = 4) ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2)) :=
by 
  intros x y
  rw [center, radius]
  simp
  sorry

end circle_equation_standard_l2037_203713


namespace toys_per_hour_computation_l2037_203765

noncomputable def total_toys : ℕ := 20500
noncomputable def monday_hours : ℕ := 8
noncomputable def tuesday_hours : ℕ := 7
noncomputable def wednesday_hours : ℕ := 9
noncomputable def thursday_hours : ℕ := 6

noncomputable def total_hours_worked : ℕ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours
noncomputable def toys_produced_each_hour : ℚ := total_toys / total_hours_worked

theorem toys_per_hour_computation :
  toys_produced_each_hour = 20500 / (8 + 7 + 9 + 6) :=
by
  -- Proof goes here
  sorry

end toys_per_hour_computation_l2037_203765


namespace inequality_problem_l2037_203727

theorem inequality_problem (a b c : ℝ) (h : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by 
  sorry

end inequality_problem_l2037_203727


namespace minimum_value_8_l2037_203762

noncomputable def minimum_value (x : ℝ) : ℝ :=
  3 * x + 2 / x^5 + 3 / x

theorem minimum_value_8 (x : ℝ) (hx : x > 0) :
  ∃ y : ℝ, (∀ z > 0, minimum_value z ≥ y) ∧ (y = 8) :=
by
  sorry

end minimum_value_8_l2037_203762


namespace car_initial_speed_l2037_203720

theorem car_initial_speed (s t : ℝ) (h₁ : t = 15 * s^2) (h₂ : t = 3) :
  s = (Real.sqrt 2) / 5 :=
by
  sorry

end car_initial_speed_l2037_203720


namespace ratio_of_areas_l2037_203771

theorem ratio_of_areas (R_A R_B : ℝ) 
  (h1 : (1 / 6) * 2 * Real.pi * R_A = (1 / 9) * 2 * Real.pi * R_B) :
  (Real.pi * R_A^2) / (Real.pi * R_B^2) = (4 : ℝ) / 9 :=
by 
  sorry

end ratio_of_areas_l2037_203771


namespace avg_temp_l2037_203744

theorem avg_temp (M T W Th F : ℝ) (h1 : M = 41) (h2 : F = 33) (h3 : (T + W + Th + F) / 4 = 46) : 
  (M + T + W + Th) / 4 = 48 :=
by
  -- insert proof steps here
  sorry

end avg_temp_l2037_203744


namespace total_distance_biked_l2037_203768

-- Definitions of the given conditions
def biking_time_to_park : ℕ := 15
def biking_time_return : ℕ := 25
def average_speed : ℚ := 6 -- miles per hour

-- Total biking time in minutes, then converted to hours
def total_biking_time_minutes : ℕ := biking_time_to_park + biking_time_return
def total_biking_time_hours : ℚ := total_biking_time_minutes / 60

-- Prove that the total distance biked is 4 miles
theorem total_distance_biked : total_biking_time_hours * average_speed = 4 := 
by
  -- proof will be here
  sorry

end total_distance_biked_l2037_203768


namespace no_integer_solutions_l2037_203791

theorem no_integer_solutions (m n : ℤ) : ¬ (5 * m^2 - 6 * m * n + 7 * n^2 = 2011) :=
by sorry

end no_integer_solutions_l2037_203791


namespace five_coins_not_155_l2037_203702

def coin_values : List ℕ := [5, 25, 50]

def can_sum_to (n : ℕ) (count : ℕ) : Prop :=
  ∃ (a b c : ℕ), a + b + c = count ∧ a * 5 + b * 25 + c * 50 = n

theorem five_coins_not_155 : ¬ can_sum_to 155 5 :=
  sorry

end five_coins_not_155_l2037_203702


namespace money_made_l2037_203711

def initial_amount : ℕ := 26
def final_amount : ℕ := 52

theorem money_made : (final_amount - initial_amount) = 26 :=
by sorry

end money_made_l2037_203711


namespace solve_for_x_l2037_203777

theorem solve_for_x : ∃ x : ℝ, (2010 + x)^3 = -x^3 ∧ x = -1005 := 
by
  use -1005
  sorry

end solve_for_x_l2037_203777


namespace probability_painted_faces_l2037_203751

theorem probability_painted_faces (total_cubes : ℕ) (corner_cubes : ℕ) (no_painted_face_cubes : ℕ) (successful_outcomes : ℕ) (total_outcomes : ℕ) 
  (probability : ℚ) : 
  total_cubes = 125 ∧ corner_cubes = 8 ∧ no_painted_face_cubes = 27 ∧ successful_outcomes = 216 ∧ total_outcomes = 7750 ∧ 
  probability = 72 / 2583 :=
by
  sorry

end probability_painted_faces_l2037_203751


namespace simplify_expression_l2037_203714

theorem simplify_expression : 4 * (12 / 9) * (36 / -45) = -12 / 5 :=
by
  sorry

end simplify_expression_l2037_203714


namespace addition_of_two_odds_is_even_subtraction_of_two_odds_is_even_l2037_203722

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem addition_of_two_odds_is_even (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_even (a + b) :=
sorry

theorem subtraction_of_two_odds_is_even (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_even (a - b) :=
sorry

end addition_of_two_odds_is_even_subtraction_of_two_odds_is_even_l2037_203722


namespace snow_leopards_arrangement_l2037_203787

theorem snow_leopards_arrangement :
  let leopards := 8
  let end_positions := 2
  let remaining_leopards := 6
  let factorial_six := Nat.factorial remaining_leopards
  end_positions * factorial_six = 1440 :=
by
  let leopards := 8
  let end_positions := 2
  let remaining_leopards := 6
  let factorial_six := Nat.factorial remaining_leopards
  show end_positions * factorial_six = 1440
  sorry

end snow_leopards_arrangement_l2037_203787


namespace percent_decrease_l2037_203796

def original_price : ℝ := 100
def sale_price : ℝ := 60

theorem percent_decrease : (original_price - sale_price) / original_price * 100 = 40 := by
  sorry

end percent_decrease_l2037_203796


namespace fraction_of_walls_not_illuminated_l2037_203795

-- Define given conditions
def point_light_source : Prop := true
def rectangular_room : Prop := true
def flat_mirror_on_wall : Prop := true
def full_height_of_room : Prop := true

-- Define the fraction not illuminated
def fraction_not_illuminated := 17 / 32

-- State the theorem to prove
theorem fraction_of_walls_not_illuminated :
  point_light_source ∧ rectangular_room ∧ flat_mirror_on_wall ∧ full_height_of_room →
  fraction_not_illuminated = 17 / 32 :=
by
  intros h
  sorry

end fraction_of_walls_not_illuminated_l2037_203795


namespace expand_product_l2037_203725

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12 * x + 27 := 
by 
  sorry

end expand_product_l2037_203725


namespace simplify_expression_l2037_203712

theorem simplify_expression : 
  (1 - 1 / (2 + Real.sqrt 5) + 1 / (2 - Real.sqrt 5) = 1) :=
by
  sorry

end simplify_expression_l2037_203712


namespace max_value_of_6_f_x_plus_2012_l2037_203775

noncomputable def f (x : ℝ) : ℝ :=
  min (min (4*x + 1) (x + 2)) (-2*x + 4)

theorem max_value_of_6_f_x_plus_2012 : ∃ x : ℝ, 6 * f x + 2012 = 2028 :=
sorry

end max_value_of_6_f_x_plus_2012_l2037_203775


namespace no_integer_root_quadratic_trinomials_l2037_203705

theorem no_integer_root_quadratic_trinomials :
  ¬ ∃ (a b c : ℤ),
    (∃ r1 r2 : ℤ, a * r1^2 + b * r1 + c = 0 ∧ a * r2^2 + b * r2 + c = 0 ∧ r1 ≠ r2) ∧
    (∃ s1 s2 : ℤ, (a + 1) * s1^2 + (b + 1) * s1 + (c + 1) = 0 ∧ (a + 1) * s2^2 + (b + 1) * s2 + (c + 1) = 0 ∧ s1 ≠ s2) :=
by
  sorry

end no_integer_root_quadratic_trinomials_l2037_203705


namespace ellipse_major_minor_axis_l2037_203788

theorem ellipse_major_minor_axis (m : ℝ) :
  (∀ x y : ℝ, x^2 + m*y^2 = 1) ∧
  (∃ a b : ℝ, a = 2 * b ∧ b^2 = 1 ∧ a^2 = 1/m) →
  m = 1/4 :=
by {
  sorry
}

end ellipse_major_minor_axis_l2037_203788


namespace sin_of_2000_deg_l2037_203740

theorem sin_of_2000_deg (a : ℝ) (h : Real.tan (160 * Real.pi / 180) = a) : 
  Real.sin (2000 * Real.pi / 180) = -a / Real.sqrt (1 + a^2) := 
by
  sorry

end sin_of_2000_deg_l2037_203740


namespace emma_missing_coins_l2037_203764

theorem emma_missing_coins (x : ℤ) (h₁ : x > 0) :
  let lost := (1 / 3 : ℚ) * x
  let found := (2 / 3 : ℚ) * lost
  let remaining := x - lost + found
  let missing := x - remaining
  missing / x = 1 / 9 :=
by
  sorry

end emma_missing_coins_l2037_203764


namespace system_of_equations_solution_l2037_203709

theorem system_of_equations_solution:
  ∀ (x y : ℝ), 
    x^2 + y^2 + x + y = 42 ∧ x * y = 15 → 
      (x = 3 ∧ y = 5) ∨ (x = 5 ∧ y = 3) ∨ 
      (x = (-9 + Real.sqrt 21) / 2 ∧ y = (-9 - Real.sqrt 21) / 2) ∨ 
      (x = (-9 - Real.sqrt 21) / 2 ∧ y = (-9 + Real.sqrt 21) / 2) := 
by
  sorry

end system_of_equations_solution_l2037_203709


namespace largest_possible_difference_l2037_203706

theorem largest_possible_difference (A_est : ℕ) (B_est : ℕ) (A : ℝ) (B : ℝ)
(hA_est : A_est = 40000) (hB_est : B_est = 70000)
(hA_range : 36000 ≤ A ∧ A ≤ 44000)
(hB_range : 60870 ≤ B ∧ B ≤ 82353) :
  abs (B - A) = 46000 :=
by sorry

end largest_possible_difference_l2037_203706


namespace roman_remy_gallons_l2037_203757

theorem roman_remy_gallons (R : ℕ) (Remy_uses : 3 * R + 1 = 25) :
  R + (3 * R + 1) = 33 :=
by
  sorry

end roman_remy_gallons_l2037_203757


namespace normal_line_at_point_l2037_203763

noncomputable def curve (x : ℝ) : ℝ := (4 * x - x ^ 2) / 4

theorem normal_line_at_point (x0 : ℝ) (h : x0 = 2) :
  ∃ (L : ℝ → ℝ), ∀ (x : ℝ), L x = (2 : ℝ) :=
by
  sorry

end normal_line_at_point_l2037_203763


namespace different_rhetorical_device_in_optionA_l2037_203797

def optionA_uses_metaphor : Prop :=
  -- Here, define the condition explaining that Option A uses metaphor
  true -- This will denote that Option A uses metaphor 

def optionsBCD_use_personification : Prop :=
  -- Here, define the condition explaining that Options B, C, and D use personification
  true -- This will denote that Options B, C, and D use personification

theorem different_rhetorical_device_in_optionA :
  optionA_uses_metaphor ∧ optionsBCD_use_personification → 
  (∃ (A P : Prop), A ≠ P) :=
by
  -- No proof is required as per instructions
  intro h
  exact Exists.intro optionA_uses_metaphor (Exists.intro optionsBCD_use_personification sorry)

end different_rhetorical_device_in_optionA_l2037_203797


namespace tan_double_angle_sum_l2037_203729

theorem tan_double_angle_sum (α : ℝ) (h : Real.tan α = 3 / 2) :
  Real.tan (2 * α + Real.pi / 4) = -7 / 17 := 
sorry

end tan_double_angle_sum_l2037_203729


namespace simplify_expr1_simplify_expr2_l2037_203701

variable {a b : ℝ} -- Assume a and b are arbitrary real numbers

-- Part 1: Prove that 2a - [-3b - 3(3a - b)] = 11a
theorem simplify_expr1 : (2 * a - (-3 * b - 3 * (3 * a - b))) = 11 * a :=
by
  sorry

-- Part 2: Prove that 12ab^2 - [7a^2b - (ab^2 - 3a^2b)] = 13ab^2 - 10a^2b
theorem simplify_expr2 : (12 * a * b^2 - (7 * a^2 * b - (a * b^2 - 3 * a^2 * b))) = (13 * a * b^2 - 10 * a^2 * b) :=
by
  sorry

end simplify_expr1_simplify_expr2_l2037_203701


namespace average_speed_stan_l2037_203724

theorem average_speed_stan (d1 d2 : ℝ) (h1 h2 rest : ℝ) (total_distance total_time : ℝ) (avg_speed : ℝ) :
  d1 = 350 → 
  d2 = 400 → 
  h1 = 6 → 
  h2 = 7 → 
  rest = 0.5 → 
  total_distance = d1 + d2 → 
  total_time = h1 + h2 + rest → 
  avg_speed = total_distance / total_time → 
  avg_speed = 55.56 :=
by 
  intros h_d1 h_d2 h_h1 h_h2 h_rest h_total_distance h_total_time h_avg_speed
  sorry

end average_speed_stan_l2037_203724


namespace knitting_time_total_l2037_203789

-- Define knitting times for each item
def hat_knitting_time : ℕ := 2
def scarf_knitting_time : ℕ := 3
def mitten_knitting_time : ℕ := 1
def sock_knitting_time : ℕ := 3 / 2
def sweater_knitting_time : ℕ := 6

-- Define the number of grandchildren
def grandchildren_count : ℕ := 3

-- Total knitting time calculation
theorem knitting_time_total : 
  hat_knitting_time + scarf_knitting_time + 2 * mitten_knitting_time + 2 * sock_knitting_time + sweater_knitting_time = 16 ∧ 
  (hat_knitting_time + scarf_knitting_time + 2 * mitten_knitting_time + 2 * sock_knitting_time + sweater_knitting_time) * grandchildren_count = 48 :=
by 
  sorry

end knitting_time_total_l2037_203789


namespace triangle_area_l2037_203776

variables {A B C D M N: Type}

-- Define the conditions and the proof 
theorem triangle_area
  (α β : ℝ)
  (CD : ℝ)
  (sin_Ratio : ℝ)
  (C_angle : ℝ)
  (MCN_Area : ℝ)
  (M_distance : ℝ)
  (N_distance : ℝ)
  (hCD : CD = Real.sqrt 13)
  (hSinRatio : (Real.sin α) / (Real.sin β) = 4 / 3)
  (hC_angle : C_angle = 120)
  (hMCN_Area : MCN_Area = 3 * Real.sqrt 3)
  (hDistance : M_distance = 2 * N_distance)
  : ∃ ABC_Area, ABC_Area = 27 * Real.sqrt 3 / 2 :=
sorry

end triangle_area_l2037_203776


namespace total_journey_distance_l2037_203753

theorem total_journey_distance (D : ℝ)
  (h1 : (D / 2) / 21 + (D / 2) / 24 = 25) : D = 560 := by
  sorry

end total_journey_distance_l2037_203753


namespace percent_employed_females_in_employed_population_l2037_203774

def percent_employed (population: ℝ) : ℝ := 0.64 * population
def percent_employed_males (population: ℝ) : ℝ := 0.50 * population
def percent_employed_females (population: ℝ) : ℝ := percent_employed population - percent_employed_males population

theorem percent_employed_females_in_employed_population (population: ℝ) : 
  (percent_employed_females population / percent_employed population) * 100 = 21.875 :=
by
  sorry

end percent_employed_females_in_employed_population_l2037_203774


namespace log_addition_property_l2037_203758

noncomputable def logFunction (x : ℝ) : ℝ := Real.log x

theorem log_addition_property (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : logFunction (a * b) = 1) :
  logFunction (a^2) + logFunction (b^2) = 2 :=
by
  sorry

end log_addition_property_l2037_203758


namespace triangle_area_l2037_203707

theorem triangle_area {a b m : ℝ} (h1 : a = 27) (h2 : b = 29) (h3 : m = 26) : 
  ∃ (area : ℝ), area = 270 :=
by
  sorry

end triangle_area_l2037_203707


namespace weight_box_plate_cups_l2037_203745

theorem weight_box_plate_cups (b p c : ℝ) 
  (h₁ : b + 20 * p + 30 * c = 4.8)
  (h₂ : b + 40 * p + 50 * c = 8.4) : 
  b + 10 * p + 20 * c = 3 :=
sorry

end weight_box_plate_cups_l2037_203745


namespace theta_in_second_quadrant_l2037_203739

theorem theta_in_second_quadrant (θ : ℝ) (h₁ : Real.sin θ > 0) (h₂ : Real.cos θ < 0) : 
  π / 2 < θ ∧ θ < π := 
sorry

end theta_in_second_quadrant_l2037_203739


namespace find_a4_l2037_203778

noncomputable def a (n : ℕ) : ℕ := sorry -- Define the arithmetic sequence
def S (n : ℕ) : ℕ := sorry -- Define the sum function for the sequence

theorem find_a4 (h1 : S 5 = 25) (h2 : a 2 = 3) : a 4 = 7 := by
  sorry

end find_a4_l2037_203778


namespace num_ways_to_divide_friends_l2037_203784

theorem num_ways_to_divide_friends :
  (4 : ℕ) ^ 8 = 65536 := by
  sorry

end num_ways_to_divide_friends_l2037_203784


namespace find_y_l2037_203748

theorem find_y (y : ℕ) : y = (12 ^ 3 * 6 ^ 4) / 432 → y = 5184 :=
by
  intro h
  rw [h]
  sorry

end find_y_l2037_203748


namespace difference_in_tiles_l2037_203734

-- Definition of side length of the nth square
def side_length (n : ℕ) : ℕ := n

-- Theorem stating the difference in tiles between the 10th and 9th squares
theorem difference_in_tiles : (side_length 10) ^ 2 - (side_length 9) ^ 2 = 19 := 
by {
  sorry
}

end difference_in_tiles_l2037_203734


namespace fraction_of_airing_time_spent_on_commercials_l2037_203736

theorem fraction_of_airing_time_spent_on_commercials 
  (num_programs : ℕ) (minutes_per_program : ℕ) (total_commercial_time : ℕ) 
  (h1 : num_programs = 6) (h2 : minutes_per_program = 30) (h3 : total_commercial_time = 45) : 
  (total_commercial_time : ℚ) / (num_programs * minutes_per_program : ℚ) = 1 / 4 :=
by {
  -- The proof is omitted here as only the statement is required according to the instruction.
  sorry
}

end fraction_of_airing_time_spent_on_commercials_l2037_203736


namespace integer_combination_zero_l2037_203760

theorem integer_combination_zero (a b c : ℤ) (h : a * Real.sqrt 2 + b * Real.sqrt 3 + c = 0) : 
  a = 0 ∧ b = 0 ∧ c = 0 :=
sorry

end integer_combination_zero_l2037_203760


namespace team_A_wins_exactly_4_of_7_l2037_203708

noncomputable def probability_team_A_wins_4_of_7 : ℚ :=
  (Nat.choose 7 4) * ((1/2)^4) * ((1/2)^3)

theorem team_A_wins_exactly_4_of_7 :
  probability_team_A_wins_4_of_7 = 35 / 128 := by
sorry

end team_A_wins_exactly_4_of_7_l2037_203708


namespace investment_change_l2037_203737

theorem investment_change (x : ℝ) :
  (1 : ℝ) > (0 : ℝ) → 
  1.05 * x / x - 1 * 100 = 5 :=
by
  sorry

end investment_change_l2037_203737


namespace acute_triangle_tangent_sum_range_l2037_203735

theorem acute_triangle_tangent_sum_range
  (a b c : ℝ) (A B C : ℝ)
  (triangle_ABC_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (opposite_sides : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (side_relation : b^2 - a^2 = a * c)
  (angle_relation : A + B + C = π)
  (angles_in_radians : 0 < A ∧ A < π)
  (angles_positive : A > 0 ∧ B > 0 ∧ C > 0) :
  1 < (1 / Real.tan A + 1 / Real.tan B) ∧ (1 / Real.tan A + 1 / Real.tan B) < 2 * Real.sqrt 3 / 3 :=
sorry 

end acute_triangle_tangent_sum_range_l2037_203735


namespace geometric_sequence_product_l2037_203755

-- Defining a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Given data
def a := fun n => (4 : ℝ) * (2 : ℝ)^(n-4)

-- Main proof problem
theorem geometric_sequence_product (a : ℕ → ℝ) (h : is_geometric_sequence a) (h₁ : a 4 = 4) :
  a 2 * a 6 = 16 :=
by
  sorry

end geometric_sequence_product_l2037_203755


namespace shaded_region_area_is_correct_l2037_203794

noncomputable def area_of_shaded_region : ℝ :=
  let R := 6 -- radius of the larger circle
  let r := R / 2 -- radius of each smaller circle
  let area_large_circle := Real.pi * R^2
  let area_two_small_circles := 2 * Real.pi * r^2
  area_large_circle - area_two_small_circles

theorem shaded_region_area_is_correct :
  area_of_shaded_region = 18 * Real.pi :=
sorry

end shaded_region_area_is_correct_l2037_203794


namespace inequality_holds_l2037_203719

theorem inequality_holds (a b : ℝ) : 
  a^2 + a * b + b^2 ≥ 3 * (a + b - 1) :=
sorry

end inequality_holds_l2037_203719


namespace intersection_point_l2037_203747

variable (x y : ℚ)

theorem intersection_point :
  (8 * x - 5 * y = 40) ∧ (10 * x + 2 * y = 14) → 
  (x = 25 / 11) ∧ (y = 48 / 11) :=
by
  sorry

end intersection_point_l2037_203747


namespace possible_values_of_a_l2037_203718

theorem possible_values_of_a :
  ∃ (a b c : ℤ), ∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c) → a = 3 ∨ a = 7 :=
by
  sorry

end possible_values_of_a_l2037_203718


namespace mean_reciprocals_first_three_composites_l2037_203700

theorem mean_reciprocals_first_three_composites :
  (1 / 4 + 1 / 6 + 1 / 8) / 3 = (13 : ℚ) / 72 := 
by
  sorry

end mean_reciprocals_first_three_composites_l2037_203700


namespace model_x_computers_used_l2037_203716

theorem model_x_computers_used
    (x_rate : ℝ)
    (y_rate : ℝ)
    (combined_rate : ℝ)
    (num_computers : ℝ) :
    x_rate = 1 / 72 →
    y_rate = 1 / 36 →
    combined_rate = num_computers * (x_rate + y_rate) →
    combined_rate = 1 →
    num_computers = 24 := by
  intros h1 h2 h3 h4
  sorry

end model_x_computers_used_l2037_203716


namespace cricketer_new_average_l2037_203782

variable (A : ℕ) (runs_19th_inning : ℕ) (avg_increase : ℕ)
variable (total_runs_after_18 : ℕ)

theorem cricketer_new_average
  (h1 : runs_19th_inning = 98)
  (h2 : avg_increase = 4)
  (h3 : total_runs_after_18 = 18 * A)
  (h4 : 18 * A + 98 = 19 * (A + 4)) :
  A + 4 = 26 :=
by sorry

end cricketer_new_average_l2037_203782


namespace proof_problem_l2037_203781

def f (x : ℝ) : ℝ := x^2 - 6 * x + 5

-- The two conditions
def condition1 (x y : ℝ) : Prop := f x + f y ≤ 0
def condition2 (x y : ℝ) : Prop := f x - f y ≥ 0

-- Equivalent description
def circle_condition (x y : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 ≤ 8
def region1 (x y : ℝ) : Prop := y ≤ x ∧ y ≥ 6 - x
def region2 (x y : ℝ) : Prop := y ≥ x ∧ y ≤ 6 - x

-- The proof statement
theorem proof_problem (x y : ℝ) :
  (condition1 x y ∧ condition2 x y) ↔ 
  (circle_condition x y ∧ (region1 x y ∨ region2 x y)) :=
sorry

end proof_problem_l2037_203781


namespace barbara_weekly_allowance_l2037_203752

theorem barbara_weekly_allowance (W C S : ℕ) (H : W = 100) (A : S = 20) (N : C = 16) :
  (W - S) / C = 5 :=
by
  -- definitions to match conditions
  have W_def : W = 100 := H
  have S_def : S = 20 := A
  have C_def : C = 16 := N
  sorry

end barbara_weekly_allowance_l2037_203752


namespace sum_of_ages_is_14_l2037_203779

/-- Kiana has two older twin brothers and the product of their three ages is 72.
    Prove that the sum of their three ages is 14. -/
theorem sum_of_ages_is_14 (kiana_age twin_age : ℕ) (htwins : twin_age > kiana_age) (h_product : kiana_age * twin_age * twin_age = 72) :
  kiana_age + twin_age + twin_age = 14 :=
sorry

end sum_of_ages_is_14_l2037_203779


namespace dice_sum_probability_l2037_203754

theorem dice_sum_probability
  (a b c d : ℕ)
  (cond1 : 1 ≤ a ∧ a ≤ 6)
  (cond2 : 1 ≤ b ∧ b ≤ 6)
  (cond3 : 1 ≤ c ∧ c ≤ 6)
  (cond4 : 1 ≤ d ∧ d ≤ 6)
  (sum_cond : a + b + c + d = 5) :
  (∃ p, p = 1 / 324) :=
sorry

end dice_sum_probability_l2037_203754


namespace maximum_value_of_objective_function_l2037_203790

variables (x y : ℝ)

def objective_function (x y : ℝ) := 3 * x + 2 * y

theorem maximum_value_of_objective_function : 
  (∀ x y, (x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4) → objective_function x y ≤ 12) 
  ∧ 
  (∃ x y, x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4 ∧ objective_function x y = 12) :=
sorry

end maximum_value_of_objective_function_l2037_203790


namespace parabola_equation_line_intersection_proof_l2037_203761

-- Define the parabola and its properties
def parabola (p x y : ℝ) := y^2 = 2 * p * x

-- Define point A
def A_point (x y₀ : ℝ) := (x, y₀)

-- Define the conditions
axiom p_pos (p : ℝ) : p > 0
axiom passes_A (y₀ : ℝ) (p : ℝ) : parabola p 2 y₀
axiom distance_A_axis (p : ℝ) : 2 + p / 2 = 4

-- Prove the equation of the parabola given the conditions
theorem parabola_equation : ∃ p, parabola p x y ∧ p = 4 := sorry

-- Define line l and its intersection properties
def line_l (m x y : ℝ) := y = x + m
def intersection_PQ (m x₁ x₂ y₁ y₂ : ℝ) := 
  line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧ y₁^2 = 8 * x₁ ∧ y₂^2 = 8 * x₂ ∧ 
  x₁ + x₂ = 8 - 2 * m ∧ x₁ * x₂ = m^2 ∧ y₁ + y₂ = 8 ∧ y₁ * y₂ = 8 * m ∧ 
  x₁ * x₂ + y₁ * y₂ = 0

-- Prove the value of m
theorem line_intersection_proof : ∃ m, ∀ (x₁ x₂ y₁ y₂ : ℝ), 
  intersection_PQ m x₁ x₂ y₁ y₂ -> m = -8 := sorry

end parabola_equation_line_intersection_proof_l2037_203761


namespace food_for_elephants_l2037_203756

theorem food_for_elephants (t : ℕ) : 
  (∀ (food_per_day : ℕ), (12 * food_per_day) * 1 = (1000 * food_per_day) * 600) →
  (∀ (food_per_day : ℕ), (t * food_per_day) * 1 = (100 * food_per_day) * d) →
  d = 500 * t :=
by
  sorry

end food_for_elephants_l2037_203756


namespace largest_is_B_l2037_203785

noncomputable def A : ℚ := ((2023:ℚ) / 2022) + ((2023:ℚ) / 2024)
noncomputable def B : ℚ := ((2024:ℚ) / 2023) + ((2026:ℚ) / 2023)
noncomputable def C : ℚ := ((2025:ℚ) / 2024) + ((2025:ℚ) / 2026)

theorem largest_is_B : B > A ∧ B > C := by
  sorry

end largest_is_B_l2037_203785


namespace graph_does_not_pass_through_fourth_quadrant_l2037_203799

def linear_function (x : ℝ) : ℝ := x + 1

theorem graph_does_not_pass_through_fourth_quadrant : 
  ¬ ∃ x : ℝ, x > 0 ∧ linear_function x < 0 :=
sorry

end graph_does_not_pass_through_fourth_quadrant_l2037_203799


namespace original_average_l2037_203742

theorem original_average (A : ℝ) (h : (10 * A = 70)) : A = 7 :=
sorry

end original_average_l2037_203742


namespace square_ratio_short_to_long_side_l2037_203710

theorem square_ratio_short_to_long_side (a b : ℝ) (h : a / b + 1 / 2 = b / (Real.sqrt (a^2 + b^2))) : (a / b)^2 = (3 - Real.sqrt 5) / 2 := by
  sorry

end square_ratio_short_to_long_side_l2037_203710


namespace frac_subtraction_l2037_203715

theorem frac_subtraction : (18 / 42) - (3 / 8) = (3 / 56) := by
  -- Conditions
  have h1 : 18 / 42 = 3 / 7 := by sorry
  have h2 : 3 / 7 = 24 / 56 := by sorry
  have h3 : 3 / 8 = 21 / 56 := by sorry
  -- Proof using the conditions
  sorry

end frac_subtraction_l2037_203715


namespace stratified_sampling_girls_count_l2037_203731

theorem stratified_sampling_girls_count :
  (boys girls sampleSize totalSample : ℕ) →
  boys = 36 →
  girls = 18 →
  sampleSize = 6 →
  totalSample = boys + girls →
  (sampleSize * girls) / totalSample = 2 :=
by
  intros boys girls sampleSize totalSample h_boys h_girls h_sampleSize h_totalSample
  sorry

end stratified_sampling_girls_count_l2037_203731


namespace integer_sum_of_squares_power_l2037_203783

theorem integer_sum_of_squares_power (a p q : ℤ) (k : ℕ) (h : a = p^2 + q^2) : 
  ∃ c d : ℤ, a^k = c^2 + d^2 := 
sorry

end integer_sum_of_squares_power_l2037_203783


namespace greatest_int_lt_neg_31_div_6_l2037_203717

theorem greatest_int_lt_neg_31_div_6 : ∃ (n : ℤ), n < -31 / 6 ∧ ∀ m : ℤ, m < -31 / 6 → m ≤ n := 
sorry

end greatest_int_lt_neg_31_div_6_l2037_203717


namespace rectangle_area_l2037_203773

theorem rectangle_area (length diagonal : ℝ) (h_length : length = 16) (h_diagonal : diagonal = 20) : 
  ∃ width : ℝ, (length * width = 192) :=
by 
  sorry

end rectangle_area_l2037_203773


namespace modulus_problem_l2037_203749

theorem modulus_problem : (13 ^ 13 + 13) % 14 = 12 :=
by
  sorry

end modulus_problem_l2037_203749


namespace sum_of_coefficients_l2037_203792

theorem sum_of_coefficients 
  (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℕ)
  (h : (3 * x - 1) ^ 10 = a + a_1 * x + a_2 * x ^ 2 + a_3 * x ^ 3 + a_4 * x ^ 4 + a_5 * x ^ 5 + a_6 * x ^ 6 + a_7 * x ^ 7 + a_8 * x ^ 8 + a_9 * x ^ 9 + a_10 * x ^ 10) :
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 = 1023 := 
sorry

end sum_of_coefficients_l2037_203792


namespace range_of_a_l2037_203728

noncomputable def range_a (a : ℝ) : Prop :=
  ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ (x^3 * Real.exp (y / x) = a * y^3)

theorem range_of_a (a : ℝ) : range_a a → a ≥ Real.exp 3 / 27 :=
by
  sorry

end range_of_a_l2037_203728


namespace problem_statement_l2037_203704

theorem problem_statement (x y z : ℝ) 
  (h1 : x + y + z = 6) 
  (h2 : x * y + y * z + z * x = 11) 
  (h3 : x * y * z = 6) : 
  x / (y * z) + y / (z * x) + z / (x * y) = 7 / 3 := 
sorry

end problem_statement_l2037_203704


namespace no_yarn_earnings_l2037_203723

noncomputable def yarn_cost : Prop :=
  let monday_yards := 20
  let tuesday_yards := 2 * monday_yards
  let wednesday_yards := (1 / 4) * tuesday_yards
  let total_yards := monday_yards + tuesday_yards + wednesday_yards
  let fabric_cost_per_yard := 2
  let total_fabric_earnings := total_yards * fabric_cost_per_yard
  let total_earnings := 140
  total_fabric_earnings = total_earnings

theorem no_yarn_earnings:
  yarn_cost :=
sorry

end no_yarn_earnings_l2037_203723


namespace projection_multiplier_l2037_203738

noncomputable def a : ℝ × ℝ := (3, 6)
noncomputable def b : ℝ × ℝ := (-1, 0)

theorem projection_multiplier :
  let dot_product := a.1 * b.1 + a.2 * b.2
  let b_norm_sq := b.1 * b.1 + b.2 * b.2
  let proj := (dot_product / b_norm_sq) * 2
  (proj * b.1, proj * b.2) = (6, 0) :=
by 
  sorry

end projection_multiplier_l2037_203738


namespace probability_correct_l2037_203798

-- Define the problem conditions.
def num_balls : ℕ := 8
def possible_colors : ℕ := 2

-- Probability calculation for a specific arrangement (either configuration of colors).
def probability_per_arrangement : ℚ := (1/2) ^ num_balls

-- Number of favorable arrangements with 4 black and 4 white balls.
def favorable_arrangements : ℕ := Nat.choose num_balls 4

-- The required probability for the solution.
def desired_probability : ℚ := favorable_arrangements * probability_per_arrangement

-- The proof statement to be provided.
theorem probability_correct :
  desired_probability = 35 / 128 := 
by
  sorry

end probability_correct_l2037_203798


namespace find_number_of_tails_l2037_203703

-- Definitions based on conditions
variables (T H : ℕ)
axiom total_coins : T + H = 1250
axiom heads_more_than_tails : H = T + 124

-- The goal is to prove T = 563
theorem find_number_of_tails : T = 563 :=
sorry

end find_number_of_tails_l2037_203703


namespace Jill_age_l2037_203759

theorem Jill_age 
  (G H I J : ℕ)
  (h1 : G = H - 4)
  (h2 : H = I + 5)
  (h3 : I + 2 = J)
  (h4 : G = 18) : 
  J = 19 := 
sorry

end Jill_age_l2037_203759


namespace fraction_decomposition_l2037_203769
noncomputable def A := (48 : ℚ) / 17
noncomputable def B := (-(25 : ℚ) / 17)

theorem fraction_decomposition (A : ℚ) (B : ℚ) :
  ( ∀ x : ℚ, x ≠ -5 ∧ x ≠ 2/3 →
    (7 * x - 13) / (3 * x^2 + 13 * x - 10) = A / (x + 5) + B / (3 * x - 2) ) ↔ 
    (A = (48 : ℚ) / 17 ∧ B = (-(25 : ℚ) / 17)) :=
by
  sorry

end fraction_decomposition_l2037_203769


namespace find_K_l2037_203730

theorem find_K 
  (Z K : ℤ) 
  (hZ_range : 1000 < Z ∧ Z < 2000)
  (hZ_eq : Z = K^4)
  (hK_pos : K > 0) :
  K = 6 :=
by {
  sorry -- Proof to be filled in
}

end find_K_l2037_203730


namespace line_slope_through_origin_intersects_parabola_l2037_203726

theorem line_slope_through_origin_intersects_parabola (k : ℝ) :
  (∃ x1 x2 : ℝ, 5 * (kx1) = 2 * x1 ^ 2 - 9 * x1 + 10 ∧ 5 * (kx2) = 2 * x2 ^ 2 - 9 * x2 + 10 ∧ x1 + x2 = 77) → k = 29 :=
by
  intro h
  sorry

end line_slope_through_origin_intersects_parabola_l2037_203726
