import Mathlib

namespace mistaken_divisor_problem_l3863_386339

theorem mistaken_divisor_problem (dividend : ℕ) (correct_divisor : ℕ) (correct_quotient : ℕ) (mistaken_quotient : ℕ) 
  (h1 : dividend = correct_divisor * correct_quotient)
  (h2 : correct_divisor = 21)
  (h3 : correct_quotient = 40)
  (h4 : mistaken_quotient = 70)
  (h5 : ∃ (mistaken_divisor : ℕ), dividend = mistaken_divisor * mistaken_quotient) :
  ∃ (mistaken_divisor : ℕ), mistaken_divisor = 12 ∧ dividend = mistaken_divisor * mistaken_quotient := by
  sorry

end mistaken_divisor_problem_l3863_386339


namespace oliver_card_arrangement_l3863_386394

/-- Calculates the minimum number of pages required to arrange Oliver's baseball cards --/
def min_pages_for_cards : ℕ :=
  let cards_per_page : ℕ := 3
  let new_cards : ℕ := 2
  let old_cards : ℕ := 10
  let rare_cards : ℕ := 3
  let pages_for_new_cards : ℕ := 1
  let pages_for_rare_cards : ℕ := 1
  let remaining_old_cards : ℕ := old_cards - rare_cards
  let pages_for_remaining_old_cards : ℕ := (remaining_old_cards + cards_per_page - 1) / cards_per_page

  pages_for_new_cards + pages_for_rare_cards + pages_for_remaining_old_cards

theorem oliver_card_arrangement :
  min_pages_for_cards = 5 := by
  sorry

end oliver_card_arrangement_l3863_386394


namespace friday_temperature_l3863_386370

/-- Given the average temperatures for two sets of four days and the temperature on Monday,
    prove that the temperature on Friday is 36 degrees. -/
theorem friday_temperature
  (avg_mon_to_thu : (mon + tue + wed + thu) / 4 = 48)
  (avg_tue_to_fri : (tue + wed + thu + fri) / 4 = 46)
  (monday_temp : mon = 44)
  : fri = 36 := by
  sorry

end friday_temperature_l3863_386370


namespace chess_game_probability_l3863_386356

theorem chess_game_probability (prob_A_win prob_A_not_lose : ℝ) 
  (h1 : prob_A_win = 0.4)
  (h2 : prob_A_not_lose = 0.8) : 
  1 - prob_A_not_lose = 0.6 :=
by
  sorry


end chess_game_probability_l3863_386356


namespace division_equals_500_l3863_386399

theorem division_equals_500 : (35 : ℝ) / 0.07 = 500 := by
  sorry

end division_equals_500_l3863_386399


namespace fraction_of_number_l3863_386311

theorem fraction_of_number (N : ℝ) (h : N = 180) : 
  6 + (1/2) * (1/3) * (1/5) * N = (1/25) * N := by
  sorry

end fraction_of_number_l3863_386311


namespace union_of_M_and_N_l3863_386352

open Set

def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

theorem union_of_M_and_N :
  M ∪ N = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end union_of_M_and_N_l3863_386352


namespace binomial_coefficient_equality_l3863_386375

theorem binomial_coefficient_equality (n : ℕ+) :
  (Nat.choose 9 (n + 1) = Nat.choose 9 (2 * n - 1)) → (n = 2 ∨ n = 3) := by
  sorry

end binomial_coefficient_equality_l3863_386375


namespace angle_bisection_limit_l3863_386357

/-- The limit of an alternating series of angle bisections in a 60° angle -/
theorem angle_bisection_limit (θ : Real) (h : θ = 60) : 
  (∑' n, (-1)^n * (1/2)^(n+1)) * θ = 20 := by
  sorry

end angle_bisection_limit_l3863_386357


namespace orchids_in_vase_l3863_386366

/-- Represents the number of roses initially in the vase -/
def initial_roses : ℕ := 9

/-- Represents the number of orchids initially in the vase -/
def initial_orchids : ℕ := 6

/-- Represents the number of roses in the vase now -/
def current_roses : ℕ := 3

/-- Represents the difference between the number of orchids and roses in the vase now -/
def orchid_rose_difference : ℕ := 10

/-- Represents the number of orchids in the vase now -/
def current_orchids : ℕ := current_roses + orchid_rose_difference

theorem orchids_in_vase : current_orchids = 13 := by sorry

end orchids_in_vase_l3863_386366


namespace cycle_price_proof_l3863_386340

theorem cycle_price_proof (selling_price : ℝ) (gain_percent : ℝ) (original_price : ℝ) : 
  selling_price = 1620 → 
  gain_percent = 8 → 
  selling_price = original_price * (1 + gain_percent / 100) → 
  original_price = 1500 := by
  sorry

#check cycle_price_proof

end cycle_price_proof_l3863_386340


namespace mechanic_rate_is_75_l3863_386363

/-- Calculates the mechanic's hourly rate given the total work time, part cost, and total amount paid -/
def mechanicHourlyRate (workTime : ℕ) (partCost : ℕ) (totalPaid : ℕ) : ℕ :=
  (totalPaid - partCost) / workTime

/-- Proves that the mechanic's hourly rate is $75 given the problem conditions -/
theorem mechanic_rate_is_75 :
  mechanicHourlyRate 2 150 300 = 75 := by
  sorry

end mechanic_rate_is_75_l3863_386363


namespace union_when_k_neg_one_intersection_condition_l3863_386373

-- Define sets A and B
def A : Set ℝ := {x | (x - 2) / (x + 1) < 0}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 - k}

-- Theorem 1: When k = -1, A ∪ B = (-1, 3)
theorem union_when_k_neg_one :
  A ∪ B (-1) = Set.Ioo (-1) 3 := by sorry

-- Theorem 2: A ∩ B = B if and only if k ∈ [0, +∞)
theorem intersection_condition (k : ℝ) :
  A ∩ B k = B k ↔ k ∈ Set.Ici 0 := by sorry

end union_when_k_neg_one_intersection_condition_l3863_386373


namespace discriminant_of_specific_quadratic_l3863_386396

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 5x^2 - 11x + 4 is 41 -/
theorem discriminant_of_specific_quadratic : discriminant 5 (-11) 4 = 41 := by
  sorry

end discriminant_of_specific_quadratic_l3863_386396


namespace right_triangle_parity_l3863_386350

theorem right_triangle_parity (a b c : ℕ) (h_right : a^2 + b^2 = c^2) :
  (Even a ∧ Even b ∧ Even c) ∨
  (Even a ∧ Odd b ∧ Odd c) ∨
  (Odd a ∧ Even b ∧ Odd c) ∨
  (Odd a ∧ Odd b ∧ Even c) :=
sorry

end right_triangle_parity_l3863_386350


namespace F_range_l3863_386374

noncomputable def F (x : ℝ) : ℝ := |2 * x + 4| - |x - 2|

theorem F_range :
  Set.range F = Set.Ici (-4) :=
sorry

end F_range_l3863_386374


namespace rectangle_measurement_error_l3863_386320

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) 
  (h1 : L > 0) (h2 : W > 0) (h3 : x > 0) :
  (1.12 * L) * ((1 - 0.01 * x) * W) = 1.064 * (L * W) → x = 5 := by
  sorry

end rectangle_measurement_error_l3863_386320


namespace max_value_quadratic_l3863_386312

theorem max_value_quadratic (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  x * (1 - 2*x) ≤ 1/8 :=
sorry

end max_value_quadratic_l3863_386312


namespace conference_hall_tables_l3863_386371

/-- Represents the setup of a conference hall --/
structure ConferenceHall where
  tables : ℕ
  chairs_per_table : ℕ
  chair_legs : ℕ
  table_legs : ℕ
  sofa_legs : ℕ
  total_legs : ℕ

/-- The conference hall setup satisfies the given conditions --/
def valid_setup (hall : ConferenceHall) : Prop :=
  hall.chairs_per_table = 8 ∧
  hall.chair_legs = 4 ∧
  hall.table_legs = 5 ∧
  hall.sofa_legs = 6 ∧
  hall.total_legs = 760

/-- The number of sofas is half the number of tables --/
def sofa_table_relation (hall : ConferenceHall) : Prop :=
  2 * (hall.tables / 2) = hall.tables

/-- The total number of legs is correctly calculated --/
def correct_leg_count (hall : ConferenceHall) : Prop :=
  hall.total_legs = 
    hall.chair_legs * (hall.chairs_per_table * hall.tables) +
    hall.table_legs * hall.tables +
    hall.sofa_legs * (hall.tables / 2)

/-- Theorem stating that given the conditions, there are 19 tables in the hall --/
theorem conference_hall_tables (hall : ConferenceHall) :
  valid_setup hall → sofa_table_relation hall → correct_leg_count hall → hall.tables = 19 := by
  sorry


end conference_hall_tables_l3863_386371


namespace shells_added_calculation_l3863_386355

/-- Calculates the amount of shells added given initial weight, percentage increase, and final weight -/
def shells_added (initial_weight : ℝ) (percent_increase : ℝ) (final_weight : ℝ) : ℝ :=
  final_weight - initial_weight

/-- Theorem stating that given the problem conditions, the amount of shells added is 23 pounds -/
theorem shells_added_calculation (initial_weight : ℝ) (percent_increase : ℝ) (final_weight : ℝ)
  (h1 : initial_weight = 5)
  (h2 : percent_increase = 150)
  (h3 : final_weight = 28) :
  shells_added initial_weight percent_increase final_weight = 23 := by
  sorry

#eval shells_added 5 150 28

end shells_added_calculation_l3863_386355


namespace p_necessary_not_sufficient_for_q_l3863_386318

-- Define the propositions
variable (f : ℝ → ℝ)
def p (f : ℝ → ℝ) : Prop := ∃ c : ℝ, ∀ x : ℝ, deriv f x = c
def q (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ f : ℝ → ℝ, q f → p f) ∧ (∃ f : ℝ → ℝ, p f ∧ ¬q f) :=
sorry

end p_necessary_not_sufficient_for_q_l3863_386318


namespace intersection_of_M_and_N_l3863_386364

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x | x^2 - x - 2 > 0}

-- State the theorem
theorem intersection_of_M_and_N : 
  M ∩ N = {x : ℝ | (-4 ≤ x ∧ x < -1) ∨ (2 < x ∧ x ≤ 7)} := by
  sorry

end intersection_of_M_and_N_l3863_386364


namespace girls_count_l3863_386313

/-- Represents the number of students in a college -/
structure College where
  boys : ℕ
  girls : ℕ

/-- The properties of the college in the problem -/
def ProblemCollege : Prop :=
  ∃ (c : College),
    (c.boys : ℚ) / c.girls = 8 / 5 ∧
    c.boys + c.girls = 780

/-- The theorem to be proved -/
theorem girls_count (h : ProblemCollege) : ∃ (c : College), c.girls = 300 ∧ ProblemCollege := by
  sorry

end girls_count_l3863_386313


namespace max_value_a_plus_2b_l3863_386310

theorem max_value_a_plus_2b (a b : ℝ) (h : a^2 + 2*b^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 3 ∧ ∀ (x y : ℝ), x^2 + 2*y^2 = 1 → x + 2*y ≤ max :=
sorry

end max_value_a_plus_2b_l3863_386310


namespace decimal_to_fraction_l3863_386369

theorem decimal_to_fraction (x : ℚ) : x = 0.38 → x = 19/50 := by
  sorry

end decimal_to_fraction_l3863_386369


namespace circle_symmetry_axis_l3863_386309

theorem circle_symmetry_axis (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 1 = 0 → 
    (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 4*x₀ + 2*y₀ + 1 = 0 ∧
      m*x₀ + y₀ - 1 = 0 ∧
      ∀ x' y' : ℝ, x'^2 + y'^2 - 4*x' + 2*y' + 1 = 0 →
        (m*x' + y' - 1 = 0 ↔ m*(2*x₀ - x') + (2*y₀ - y') - 1 = 0))) →
  m = 1 := by sorry

end circle_symmetry_axis_l3863_386309


namespace new_person_weight_l3863_386314

/-- The weight of a new person joining a group, given the initial group size,
    average weight increase, and weight of the replaced person. -/
theorem new_person_weight
  (initial_group_size : ℕ)
  (average_weight_increase : ℝ)
  (replaced_person_weight : ℝ)
  (h1 : initial_group_size = 8)
  (h2 : average_weight_increase = 6)
  (h3 : replaced_person_weight = 40) :
  replaced_person_weight + initial_group_size * average_weight_increase = 88 :=
sorry

end new_person_weight_l3863_386314


namespace F_minimum_value_G_two_zeros_range_inequality_for_positive_x_l3863_386345

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := (1/2) * x^2
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.log x
def F (a : ℝ) (x : ℝ) : ℝ := f x * g a x
def G (a : ℝ) (x : ℝ) : ℝ := f x - g a x + (a - 1) * x

-- Theorem 1: Minimum value of F(x)
theorem F_minimum_value (a : ℝ) (h : a > 0) :
  ∃ x₀ : ℝ, x₀ > 0 ∧ F a x₀ = -a / (4 * Real.exp 1) ∧ ∀ x > 0, F a x ≥ -a / (4 * Real.exp 1) :=
sorry

-- Theorem 2: Range of a for G(x) to have two zeros
theorem G_two_zeros_range :
  ∃ a₁ a₂ : ℝ, a₁ = (2 * Real.exp 1 - 1) / (2 * Real.exp 1^2 + 2 * Real.exp 1) ∧
               a₂ = 1/2 ∧
               ∀ a : ℝ, (∃ x₁ x₂ : ℝ, 1/Real.exp 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp 1 ∧
                                      G a x₁ = 0 ∧ G a x₂ = 0) ↔
                        (a₁ < a ∧ a < a₂) :=
sorry

-- Theorem 3: Inequality for x > 0
theorem inequality_for_positive_x (x : ℝ) (h : x > 0) :
  Real.log x + 3 / (4 * x^2) - 1 / Real.exp x > 0 :=
sorry

end F_minimum_value_G_two_zeros_range_inequality_for_positive_x_l3863_386345


namespace power_inequality_l3863_386330

theorem power_inequality (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 3) :
  a^b + 1 ≥ b * (a + 1) := by
  sorry

end power_inequality_l3863_386330


namespace water_cube_product_l3863_386329

/-- Definition of a water cube number -/
def V (a b c : ℝ) : ℝ := a^3 + b^3 + c^3 - 3*a*b*c

/-- Theorem: The product of two water cube numbers is a water cube number -/
theorem water_cube_product (a b c x y z : ℝ) :
  V a b c * V x y z = V (a*x + b*y + c*z) (b*x + c*y + a*z) (c*x + a*y + b*z) := by
  sorry

end water_cube_product_l3863_386329


namespace probability_no_university_in_further_analysis_l3863_386301

/-- Represents the types of schools in the region -/
inductive SchoolType
  | Elementary
  | Middle
  | University

/-- Represents the total number of schools of each type -/
def totalSchools : SchoolType → Nat
  | SchoolType.Elementary => 21
  | SchoolType.Middle => 14
  | SchoolType.University => 7

/-- The total number of schools in the region -/
def totalAllSchools : Nat := 
  totalSchools SchoolType.Elementary + 
  totalSchools SchoolType.Middle + 
  totalSchools SchoolType.University

/-- The number of schools selected in the stratified sample -/
def sampleSize : Nat := 6

/-- The number of schools of each type in the stratified sample -/
def stratifiedSample : SchoolType → Nat
  | SchoolType.Elementary => 3
  | SchoolType.Middle => 2
  | SchoolType.University => 1

/-- The number of schools selected for further analysis -/
def furtherAnalysisSize : Nat := 2

theorem probability_no_university_in_further_analysis : 
  (Nat.choose (stratifiedSample SchoolType.Elementary + stratifiedSample SchoolType.Middle) furtherAnalysisSize : ℚ) / 
  (Nat.choose sampleSize furtherAnalysisSize : ℚ) = 2 / 3 := by
  sorry

end probability_no_university_in_further_analysis_l3863_386301


namespace range_of_a_l3863_386386

theorem range_of_a (x : ℝ) (a : ℝ) : 
  x ∈ Set.Ioo 0 π → 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 2 * Real.sin (x₁ + π/3) = a ∧ 2 * Real.sin (x₂ + π/3) = a) → 
  a ∈ Set.Ioo (Real.sqrt 3) 2 :=
sorry

end range_of_a_l3863_386386


namespace number_in_mind_l3863_386306

theorem number_in_mind (x : ℝ) : (x - 6) / 13 = 2 → x = 32 := by
  sorry

end number_in_mind_l3863_386306


namespace lcm_of_ratio_and_hcf_l3863_386351

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 5 / 13 → 
  Nat.gcd a b = 19 → 
  Nat.lcm a b = 1235 := by
sorry

end lcm_of_ratio_and_hcf_l3863_386351


namespace students_above_120_l3863_386349

/-- Represents the probability density function of a normal distribution -/
noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ := sorry

/-- Represents the cumulative distribution function of a normal distribution -/
noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ := sorry

/-- The math scores follow a normal distribution with mean 110 and some standard deviation σ -/
axiom score_distribution (σ : ℝ) (x : ℝ) : 
  normal_pdf 110 σ x = normal_pdf 110 σ x

/-- The probability of scoring between 100 and 110 is 0.2 -/
axiom prob_100_to_110 (σ : ℝ) : 
  normal_cdf 110 σ 110 - normal_cdf 110 σ 100 = 0.2

/-- The total number of students is 800 -/
def total_students : ℕ := 800

/-- Theorem: Given the conditions, 240 students will score above 120 -/
theorem students_above_120 (σ : ℝ) : 
  (1 - normal_cdf 110 σ 120) * total_students = 240 := by sorry

end students_above_120_l3863_386349


namespace weight_difference_l3863_386379

/-- Given Mildred weighs 59 pounds and Carol weighs 9 pounds, 
    prove that Mildred is 50 pounds heavier than Carol. -/
theorem weight_difference (mildred_weight carol_weight : ℕ) 
  (h1 : mildred_weight = 59) 
  (h2 : carol_weight = 9) : 
  mildred_weight - carol_weight = 50 := by
  sorry

end weight_difference_l3863_386379


namespace fuel_tank_capacity_l3863_386367

/-- Proves that the capacity of a fuel tank is 212 gallons given specific conditions about fuel composition and volume. -/
theorem fuel_tank_capacity : ∃ (C : ℝ), 
  (0.12 * 98 + 0.16 * (C - 98) = 30) ∧ 
  C = 212 := by
  sorry

end fuel_tank_capacity_l3863_386367


namespace local_value_in_product_l3863_386348

/-- The face value of a digit is the digit itself -/
def faceValue (d : ℕ) : ℕ := d

/-- The local value of a digit in a number is the digit multiplied by its place value -/
def localValue (d : ℕ) (placeValue : ℕ) : ℕ := d * placeValue

/-- The product of two numbers -/
def product (a b : ℕ) : ℕ := a * b

/-- Theorem: In the product of the face value of 7 and the local value of 6 in 7098060,
    the local value of 6 is 6000 -/
theorem local_value_in_product :
  let number : ℕ := 7098060
  let fv7 : ℕ := faceValue 7
  let lv6 : ℕ := localValue 6 1000
  let prod : ℕ := product fv7 lv6
  localValue 6 1000 = 6000 := by sorry

end local_value_in_product_l3863_386348


namespace red_balls_count_l3863_386381

/-- The number of red balls in a bag with given conditions -/
theorem red_balls_count (total : ℕ) (white green yellow purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 60 →
  white = 22 →
  green = 18 →
  yellow = 8 →
  purple = 7 →
  prob_not_red_purple = 4/5 →
  (white + green + yellow : ℚ) / total = prob_not_red_purple →
  total - (white + green + yellow + purple) = 5 := by
  sorry

end red_balls_count_l3863_386381


namespace min_cost_container_l3863_386359

/-- Represents the dimensions and costs of a rectangular container. -/
structure Container where
  volume : ℝ
  height : ℝ
  baseCost : ℝ
  sideCost : ℝ

/-- Calculates the total cost of constructing the container. -/
def totalCost (c : Container) (length width : ℝ) : ℝ :=
  c.baseCost * length * width + c.sideCost * 2 * (length + width) * c.height

/-- Theorem stating that the minimum cost to construct the given container is 1600 yuan. -/
theorem min_cost_container (c : Container) 
  (h_volume : c.volume = 4)
  (h_height : c.height = 1)
  (h_baseCost : c.baseCost = 200)
  (h_sideCost : c.sideCost = 100) :
  ∃ (cost : ℝ), cost = 1600 ∧ ∀ (length width : ℝ), length * width * c.height = c.volume → 
    totalCost c length width ≥ cost := by
  sorry

#check min_cost_container

end min_cost_container_l3863_386359


namespace factorize_3x_minus_12x_squared_factorize_negative_x_squared_plus_6xy_minus_9y_squared_factorize_n_squared_m_minus_2_plus_2_minus_m_factorize_a_squared_plus_4b_squared_squared_minus_16a_squared_b_squared_l3863_386398

-- Problem 1
theorem factorize_3x_minus_12x_squared (x : ℝ) :
  3*x - 12*x^2 = 3*x*(1-4*x) := by sorry

-- Problem 2
theorem factorize_negative_x_squared_plus_6xy_minus_9y_squared (x y : ℝ) :
  -x^2 + 6*x*y - 9*y^2 = -(x-3*y)^2 := by sorry

-- Problem 3
theorem factorize_n_squared_m_minus_2_plus_2_minus_m (m n : ℝ) :
  n^2*(m-2) + (2-m) = (m-2)*(n+1)*(n-1) := by sorry

-- Problem 4
theorem factorize_a_squared_plus_4b_squared_squared_minus_16a_squared_b_squared (a b : ℝ) :
  (a^2 + 4*b^2)^2 - 16*a^2*b^2 = (a+2*b)^2 * (a-2*b)^2 := by sorry

end factorize_3x_minus_12x_squared_factorize_negative_x_squared_plus_6xy_minus_9y_squared_factorize_n_squared_m_minus_2_plus_2_minus_m_factorize_a_squared_plus_4b_squared_squared_minus_16a_squared_b_squared_l3863_386398


namespace unique_quadratic_solution_l3863_386316

theorem unique_quadratic_solution (c : ℝ) : 
  (c = -1 ∧ 
   ∃! b : ℝ, b ≠ 0 ∧ 
   ∃! x : ℝ, x^2 + (b^2 + 1/b^2) * x + c = 0) ↔ 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b ≠ 0 ∧ 
   ∃! x : ℝ, x^2 + (b^2 + 1/b^2) * x + c = 0) :=
by sorry

end unique_quadratic_solution_l3863_386316


namespace complex_fraction_equality_l3863_386354

theorem complex_fraction_equality (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : a + b * i = i * (1 - i)) :
  (a + b * i) / (a - b * i) = i := by sorry

end complex_fraction_equality_l3863_386354


namespace min_sum_with_gcd_and_divisibility_l3863_386321

theorem min_sum_with_gcd_and_divisibility (a b : ℕ+) :
  (Nat.gcd a b = 2015) →
  ((a + b) ∣ ((a - b)^2016 + b^2016)) →
  (∀ c d : ℕ+, (Nat.gcd c d = 2015) → ((c + d) ∣ ((c - d)^2016 + d^2016)) → (a + b ≤ c + d)) →
  a + b = 10075 := by
sorry

end min_sum_with_gcd_and_divisibility_l3863_386321


namespace time_between_periods_l3863_386362

theorem time_between_periods 
  (total_time : ℕ)
  (num_periods : ℕ)
  (period_duration : ℕ)
  (h1 : total_time = 220)
  (h2 : num_periods = 5)
  (h3 : period_duration = 40) :
  (total_time - num_periods * period_duration) / (num_periods - 1) = 5 :=
by sorry

end time_between_periods_l3863_386362


namespace quadratic_two_distinct_roots_l3863_386344

theorem quadratic_two_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 5*x₁ + 6 = 0 ∧ x₂^2 - 5*x₂ + 6 = 0 := by
  sorry

end quadratic_two_distinct_roots_l3863_386344


namespace combined_completion_time_l3863_386307

/-- Given the time taken by X, Y, and Z to complete a task individually,
    calculate the time taken when they work together. -/
theorem combined_completion_time
  (x_time y_time z_time : ℝ)
  (hx : x_time = 15)
  (hy : y_time = 30)
  (hz : z_time = 20)
  : (1 : ℝ) / ((1 / x_time) + (1 / y_time) + (1 / z_time)) = 20 / 3 := by
  sorry

#check combined_completion_time

end combined_completion_time_l3863_386307


namespace units_digit_of_sum_of_squares_of_first_1013_odd_integers_l3863_386319

def first_n_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def sum_of_squares (list : List ℕ) : ℕ :=
  list.map (fun x => x * x) |> List.sum

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_sum_of_squares_of_first_1013_odd_integers :
  units_digit (sum_of_squares (first_n_odd_integers 1013)) = 5 := by
  sorry

end units_digit_of_sum_of_squares_of_first_1013_odd_integers_l3863_386319


namespace expression_evaluation_l3863_386308

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := 1/3
  5 * (3 * a^2 * b - a * b^2) - (a * b^2 + 3 * a^2 * b) = 2/3 := by
  sorry

end expression_evaluation_l3863_386308


namespace pencil_distribution_l3863_386325

/-- Given an initial number of pencils, number of containers, and additional pencils,
    calculate the number of pencils that can be evenly distributed per container. -/
def evenlyDistributedPencils (initialPencils : ℕ) (containers : ℕ) (additionalPencils : ℕ) : ℕ :=
  (initialPencils + additionalPencils) / containers

theorem pencil_distribution (initialPencils : ℕ) (containers : ℕ) (additionalPencils : ℕ) 
    (h1 : initialPencils = 150)
    (h2 : containers = 5)
    (h3 : additionalPencils = 30) :
  evenlyDistributedPencils initialPencils containers additionalPencils = 36 := by
  sorry

end pencil_distribution_l3863_386325


namespace probability_three_quarters_l3863_386378

/-- A diamond-shaped checkerboard formed by an 8x8 grid -/
structure DiamondCheckerboard where
  total_squares : ℕ
  squares_per_vertex : ℕ
  num_vertices : ℕ

/-- The probability that a randomly chosen unit square does not touch a vertex of the diamond -/
def probability_not_touching_vertex (board : DiamondCheckerboard) : ℚ :=
  1 - (board.squares_per_vertex * board.num_vertices : ℚ) / board.total_squares

/-- Theorem stating that the probability of not touching a vertex is 3/4 -/
theorem probability_three_quarters (board : DiamondCheckerboard) 
  (h1 : board.total_squares = 64)
  (h2 : board.squares_per_vertex = 4)
  (h3 : board.num_vertices = 4) : 
  probability_not_touching_vertex board = 3/4 := by
  sorry

end probability_three_quarters_l3863_386378


namespace system_solution_l3863_386326

-- Define the system of equations
def equation1 (x y p : ℝ) : Prop := (x - p)^2 = 16 * (y - 3 + p)
def equation2 (x y : ℝ) : Prop := y^2 + ((x - 3) / (|x| - 3))^2 = 1

-- Define the solution set
def is_solution (x y p : ℝ) : Prop :=
  equation1 x y p ∧ equation2 x y

-- Define the valid range for p
def valid_p (p : ℝ) : Prop :=
  (p > 3 ∧ p ≤ 4) ∨ (p > 12 ∧ p ≠ 19) ∨ (p > 19)

-- Theorem statement
theorem system_solution :
  ∀ p : ℝ, valid_p p →
    ∃ x y : ℝ, is_solution x y p ∧
      x = p + 4 * Real.sqrt (p - 3) ∧
      y = 0 :=
sorry

end system_solution_l3863_386326


namespace sector_area_l3863_386333

/-- The area of a sector with radius 2 and perimeter equal to the circumference of its circle is 4π - 2 -/
theorem sector_area (r : ℝ) (θ : ℝ) : 
  r = 2 → 
  2 * r + r * θ = 2 * π * r → 
  (1/2) * r^2 * θ = 4 * π - 2 := by
  sorry

end sector_area_l3863_386333


namespace prob_diff_suits_one_heart_correct_l3863_386324

/-- The number of cards in a standard deck --/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck --/
def num_suits : ℕ := 4

/-- The number of cards of each suit in a standard deck --/
def cards_per_suit : ℕ := standard_deck_size / num_suits

/-- The total number of cards in two combined standard decks --/
def total_cards : ℕ := 2 * standard_deck_size

/-- The probability of selecting two cards from two combined standard 52-card decks,
    where the cards are of different suits and at least one is a heart --/
def prob_diff_suits_one_heart : ℚ := 91467 / 276044

theorem prob_diff_suits_one_heart_correct :
  let total_combinations := total_cards.choose 2
  let diff_suit_prob := (total_cards - cards_per_suit) / (total_cards - 1)
  let at_least_one_heart := total_combinations - (total_cards - 2 * cards_per_suit).choose 2
  diff_suit_prob * (at_least_one_heart / total_combinations) = prob_diff_suits_one_heart := by
  sorry

end prob_diff_suits_one_heart_correct_l3863_386324


namespace division_multiplication_order_l3863_386337

theorem division_multiplication_order : (120 / 6) / 2 * 3 = 30 := by
  sorry

end division_multiplication_order_l3863_386337


namespace gift_box_wrapping_l3863_386303

theorem gift_box_wrapping (total_ribbon : ℝ) (ribbon_per_box : ℝ) :
  total_ribbon = 25 →
  ribbon_per_box = 1.6 →
  ⌊total_ribbon / ribbon_per_box⌋ = 15 := by
  sorry

end gift_box_wrapping_l3863_386303


namespace binomial_20_choose_6_l3863_386358

theorem binomial_20_choose_6 : Nat.choose 20 6 = 19380 := by sorry

end binomial_20_choose_6_l3863_386358


namespace quadratic_function_range_l3863_386323

def quadratic_function (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * a * x - 6

theorem quadratic_function_range (a : ℝ) :
  (∃ y₁ y₂ y₃ y₄ : ℝ,
    quadratic_function a (-4) = y₁ ∧
    quadratic_function a (-3) = y₂ ∧
    quadratic_function a 0 = y₃ ∧
    quadratic_function a 2 = y₄ ∧
    (y₁ ≤ 0 ∧ y₂ ≤ 0 ∧ y₃ ≤ 0 ∧ y₄ > 0) ∨
    (y₁ ≤ 0 ∧ y₂ > 0 ∧ y₃ ≤ 0 ∧ y₄ ≤ 0) ∨
    (y₁ ≤ 0 ∧ y₂ ≤ 0 ∧ y₃ > 0 ∧ y₄ ≤ 0) ∨
    (y₁ > 0 ∧ y₂ ≤ 0 ∧ y₃ ≤ 0 ∧ y₄ ≤ 0)) →
  a < -2 ∨ a > 1/2 := by
sorry

end quadratic_function_range_l3863_386323


namespace identify_genuine_coins_l3863_386365

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal : WeighResult
  | Unequal : WeighResult

/-- Represents a set of coins -/
structure CoinSet where
  total : Nat
  fake : Nat
  h_fake_count : fake ≤ 1
  h_total : total = 101

/-- Represents a weighing action -/
def weighing (left right : Nat) : WeighResult :=
  sorry

/-- The main theorem to prove -/
theorem identify_genuine_coins (coins : CoinSet) :
  ∃ (genuine : Nat), genuine ≥ 50 ∧
    ∀ (left right : Nat),
      left + right ≤ coins.total →
      (weighing left right = WeighResult.Equal →
        genuine = left + right) ∧
      (weighing left right = WeighResult.Unequal →
        genuine = coins.total - (left + right)) :=
  sorry

end identify_genuine_coins_l3863_386365


namespace exponential_inequality_l3863_386317

theorem exponential_inequality (m n : ℝ) (a b : ℝ) 
  (h1 : a = (0.2 : ℝ) ^ m) 
  (h2 : b = (0.2 : ℝ) ^ n) 
  (h3 : m > n) : 
  a < b := by
sorry

end exponential_inequality_l3863_386317


namespace sum_of_roots_l3863_386390

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 :=
by sorry

end sum_of_roots_l3863_386390


namespace solve_for_a_l3863_386372

theorem solve_for_a (x a : ℚ) (h1 : x - 2 * a + 5 = 0) (h2 : x = -2) : a = 3 / 2 := by
  sorry

end solve_for_a_l3863_386372


namespace mutually_exclusive_events_l3863_386384

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Black

/-- Represents the bag of balls -/
def Bag : Multiset BallColor :=
  Multiset.replicate 3 BallColor.Red + Multiset.replicate 2 BallColor.Black

/-- Represents a draw of two balls from the bag -/
def Draw : Type := Fin 2 → BallColor

/-- The event of drawing at least one black ball -/
def AtLeastOneBlack (draw : Draw) : Prop :=
  ∃ i : Fin 2, draw i = BallColor.Black

/-- The event of drawing all red balls -/
def AllRed (draw : Draw) : Prop :=
  ∀ i : Fin 2, draw i = BallColor.Red

/-- The theorem stating that AtLeastOneBlack and AllRed are mutually exclusive -/
theorem mutually_exclusive_events :
  ∀ (draw : Draw), ¬(AtLeastOneBlack draw ∧ AllRed draw) :=
by sorry

end mutually_exclusive_events_l3863_386384


namespace top_field_is_nine_l3863_386391

/-- Represents a labelling of the figure -/
def Labelling := Fin 9 → Fin 9

/-- Check if a labelling is valid -/
def is_valid (l : Labelling) : Prop :=
  let s := l 0 + l 1 + l 2 -- sum of top row
  (l 0 + l 1 + l 2 = s) ∧
  (l 3 + l 4 + l 5 = s) ∧
  (l 6 + l 7 + l 8 = s) ∧
  (l 0 + l 3 + l 6 = s) ∧
  (l 1 + l 4 + l 7 = s) ∧
  (l 2 + l 5 + l 8 = s) ∧
  (l 0 + l 4 + l 8 = s) ∧
  (l 2 + l 4 + l 6 = s) ∧
  Function.Injective l

theorem top_field_is_nine (l : Labelling) (h : is_valid l) : l 0 = 9 := by
  sorry

#check top_field_is_nine

end top_field_is_nine_l3863_386391


namespace sum_of_squares_of_roots_l3863_386376

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 3 * x₁ - 5 = 0) →
  (2 * x₂^2 + 3 * x₂ - 5 = 0) →
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 29/4) := by
sorry

end sum_of_squares_of_roots_l3863_386376


namespace P_proper_subset_Q_l3863_386336

def P : Set ℕ := {1, 2, 4}
def Q : Set ℕ := {1, 2, 4, 8}

theorem P_proper_subset_Q : P ⊂ Q := by sorry

end P_proper_subset_Q_l3863_386336


namespace circle_center_coordinate_product_l3863_386360

/-- Given two points as endpoints of a circle's diameter, 
    calculate the product of the coordinates of the circle's center -/
theorem circle_center_coordinate_product 
  (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) 
  (h1 : p1 = (7, -8)) 
  (h2 : p2 = (-2, 3)) : 
  let center := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (center.1 * center.2) = -25/4 := by
sorry

end circle_center_coordinate_product_l3863_386360


namespace man_pants_count_l3863_386383

theorem man_pants_count (t_shirts : ℕ) (total_ways : ℕ) (pants : ℕ) : 
  t_shirts = 8 → 
  total_ways = 72 → 
  total_ways = t_shirts * pants → 
  pants = 9 := by sorry

end man_pants_count_l3863_386383


namespace team_a_prefers_best_of_five_l3863_386353

/-- Represents the probability of Team A winning a non-deciding game -/
def team_a_win_prob : ℝ := 0.6

/-- Represents the probability of Team B winning a non-deciding game -/
def team_b_win_prob : ℝ := 0.4

/-- Represents the probability of either team winning a deciding game -/
def deciding_game_win_prob : ℝ := 0.5

/-- Calculates the probability of Team A winning a best-of-three series -/
def best_of_three_win_prob : ℝ := 
  team_a_win_prob^2 + 2 * team_a_win_prob * team_b_win_prob * deciding_game_win_prob

/-- Calculates the probability of Team A winning a best-of-five series -/
def best_of_five_win_prob : ℝ := 
  team_a_win_prob^3 + 
  3 * team_a_win_prob^2 * team_b_win_prob + 
  6 * team_a_win_prob^2 * team_b_win_prob^2 * deciding_game_win_prob

/-- Theorem stating that Team A has a higher probability of winning in a best-of-five series -/
theorem team_a_prefers_best_of_five : best_of_five_win_prob > best_of_three_win_prob := by
  sorry

end team_a_prefers_best_of_five_l3863_386353


namespace quadratic_roots_imply_a_value_l3863_386361

theorem quadratic_roots_imply_a_value (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 3) * x^2 + 5 * x - 2 = 0 ↔ (x = 1/2 ∨ x = 2)) →
  (a = 1 ∨ a = -1) :=
by sorry

end quadratic_roots_imply_a_value_l3863_386361


namespace abs_two_minus_sqrt_five_l3863_386341

theorem abs_two_minus_sqrt_five : 
  |2 - Real.sqrt 5| = Real.sqrt 5 - 2 := by sorry

end abs_two_minus_sqrt_five_l3863_386341


namespace power_function_through_point_l3863_386335

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2) : 
  f 27 = 3 * Real.sqrt 3 := by
  sorry

end power_function_through_point_l3863_386335


namespace new_students_average_age_l3863_386382

/-- Proves that the average age of new students is 32 years given the conditions of the problem -/
theorem new_students_average_age
  (original_average : ℝ)
  (original_strength : ℕ)
  (new_students : ℕ)
  (average_decrease : ℝ)
  (h1 : original_average = 40)
  (h2 : original_strength = 15)
  (h3 : new_students = 15)
  (h4 : average_decrease = 4) :
  let new_average := original_average - average_decrease
  let total_original := original_strength * original_average
  let total_new := (original_strength + new_students) * new_average - total_original
  total_new / new_students = 32 := by
  sorry

#check new_students_average_age

end new_students_average_age_l3863_386382


namespace cost_price_percentage_l3863_386300

-- Define the discount rate
def discount_rate : ℝ := 0.25

-- Define the gain percent after discount
def gain_percent : ℝ := 0.171875

-- Define the relationship between cost price and marked price
theorem cost_price_percentage (marked_price cost_price : ℝ) 
  (h1 : marked_price > 0) 
  (h2 : cost_price > 0) 
  (h3 : marked_price * (1 - discount_rate) = cost_price * (1 + gain_percent)) : 
  cost_price / marked_price = 0.64 := by
  sorry


end cost_price_percentage_l3863_386300


namespace max_element_bound_l3863_386346

/-- A set of 5 different positive integers -/
def IntegerSet : Type := Fin 5 → ℕ+

/-- The mean of the set is 20 -/
def hasMean20 (s : IntegerSet) : Prop :=
  (s 0 + s 1 + s 2 + s 3 + s 4 : ℚ) / 5 = 20

/-- The median of the set is 18 -/
def hasMedian18 (s : IntegerSet) : Prop :=
  s 2 = 18

/-- The elements of the set are distinct -/
def isDistinct (s : IntegerSet) : Prop :=
  ∀ i j, i ≠ j → s i ≠ s j

/-- The elements are in ascending order -/
def isAscending (s : IntegerSet) : Prop :=
  ∀ i j, i < j → s i < s j

theorem max_element_bound (s : IntegerSet) 
  (h_mean : hasMean20 s)
  (h_median : hasMedian18 s)
  (h_distinct : isDistinct s)
  (h_ascending : isAscending s) :
  s 4 ≤ 60 :=
sorry

end max_element_bound_l3863_386346


namespace divisor_problem_l3863_386342

theorem divisor_problem (n : ℕ) (h : n = 13294) : 
  ∃ (d : ℕ), d > 1 ∧ (n - 5) % d = 0 ∧ d = 13289 := by
  sorry

end divisor_problem_l3863_386342


namespace intersection_of_A_and_B_l3863_386304

def A : Set ℝ := {y | ∃ x, y = 2^x ∧ 0 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l3863_386304


namespace polynomial_simplification_l3863_386389

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^2 + 4 * x + 6) * (x - 2) - (x - 2) * (2 * x^2 + 5 * x - 72) + (2 * x - 15) * (x - 2) * (x + 4) = 
  3 * x^3 - 14 * x^2 + 34 * x - 36 := by
sorry

end polynomial_simplification_l3863_386389


namespace sequence_sum_l3863_386347

/-- Given a sequence {a_n} where a_1 = 1 and S_n = n^2 * a_n for all positive integers n,
    prove that the sum of the first n terms (S_n) is equal to 2n / (n+1). -/
theorem sequence_sum (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) :
  a 1 = 1 →
  (∀ n : ℕ+, S n = n^2 * a n) →
  ∀ n : ℕ+, S n = (2 * n : ℝ) / (n + 1) :=
sorry

end sequence_sum_l3863_386347


namespace number_system_existence_l3863_386392

/-- Represents a number in a given base --/
def BaseNumber (base : ℕ) (value : ℕ) : Prop :=
  value < base

/-- Addition in a given base --/
def BaseAdd (base : ℕ) (a b c : ℕ) : Prop :=
  BaseNumber base a ∧ BaseNumber base b ∧ BaseNumber base c ∧
  (a + b) % base = c

/-- Multiplication in a given base --/
def BaseMult (base : ℕ) (a b c : ℕ) : Prop :=
  BaseNumber base a ∧ BaseNumber base b ∧ BaseNumber base c ∧
  (a * b) % base = c

theorem number_system_existence :
  (∃ b : ℕ, BaseAdd b 3 4 10 ∧ BaseMult b 3 4 15) ∧
  (¬ ∃ b : ℕ, BaseAdd b 2 3 5 ∧ BaseMult b 2 3 11) := by
  sorry

end number_system_existence_l3863_386392


namespace circle_tangent_to_x_axis_l3863_386393

-- Define the center of the circle
def center : ℝ × ℝ := (-3, 4)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x + 3)^2 + (y - 4)^2 = 16

-- Theorem statement
theorem circle_tangent_to_x_axis :
  -- The circle has center at (-3, 4)
  ∃ (x y : ℝ), circle_equation x y ∧ (x, y) = center ∧
  -- The circle is tangent to the x-axis
  ∃ (x : ℝ), circle_equation x 0 ∧
  -- The equation represents a circle
  ∀ (p : ℝ × ℝ), p ∈ {p | circle_equation p.1 p.2} ↔ 
    (p.1 - center.1)^2 + (p.2 - center.2)^2 = 4^2 :=
sorry

end circle_tangent_to_x_axis_l3863_386393


namespace perpendicular_lines_parallel_l3863_386327

-- Define the types for lines and planes in 3D space
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perp m α → perp n α → para m n :=
sorry

end perpendicular_lines_parallel_l3863_386327


namespace set_intersection_range_l3863_386385

theorem set_intersection_range (a : ℝ) : 
  let A : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 1}
  let B : Set ℝ := {x | 0 < x ∧ x < 1}
  A ∩ B = ∅ → (a ≤ -1/2 ∨ a ≥ 2) :=
by sorry

end set_intersection_range_l3863_386385


namespace average_price_of_cow_l3863_386368

/-- Given the total price for 2 cows and 8 goats, and the average price of a goat,
    prove that the average price of a cow is 460 rupees. -/
theorem average_price_of_cow (total_price : ℕ) (goat_price : ℕ) (cow_count : ℕ) (goat_count : ℕ) :
  total_price = 1400 →
  goat_price = 60 →
  cow_count = 2 →
  goat_count = 8 →
  (total_price - goat_count * goat_price) / cow_count = 460 := by
  sorry

end average_price_of_cow_l3863_386368


namespace largest_x_satisfying_equation_l3863_386387

theorem largest_x_satisfying_equation : 
  ∃ (x : ℚ), x = 3/25 ∧ 
  (∀ y : ℚ, y ≥ 0 → Real.sqrt (3 * y) = 5 * y → y ≤ x) ∧
  Real.sqrt (3 * x) = 5 * x := by
  sorry

end largest_x_satisfying_equation_l3863_386387


namespace partnership_investment_ratio_l3863_386328

/-- Given a partnership business where:
  * A's investment is k times B's investment
  * A's investment period is twice B's investment period
  * B's profit is 7000
  * Total profit is 49000
  Prove that the ratio of A's investment to B's investment is 3:1 -/
theorem partnership_investment_ratio 
  (k : ℚ) 
  (b_profit : ℚ) 
  (total_profit : ℚ) 
  (h1 : b_profit = 7000)
  (h2 : total_profit = 49000)
  (h3 : k * b_profit * 2 + b_profit = total_profit) : 
  k = 3 := by
sorry

end partnership_investment_ratio_l3863_386328


namespace range_of_f_l3863_386334

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 5

-- Define the domain of x
def domain : Set ℝ := { x | -3 ≤ x ∧ x < 2 }

-- State the theorem
theorem range_of_f :
  ∃ (y_min y_max : ℝ), y_min = -7 ∧ y_max = 11 ∧
  ∀ y, (∃ x ∈ domain, f x = y) ↔ y_min ≤ y ∧ y < y_max :=
sorry

end range_of_f_l3863_386334


namespace exponent_multiplication_l3863_386397

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by sorry

end exponent_multiplication_l3863_386397


namespace restricted_arrangements_five_students_l3863_386322

/-- The number of ways to arrange n students in a row. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n students in a row with one specific student not in the front. -/
def arrangementsWithRestriction (n : ℕ) : ℕ := (n - 1) * arrangements (n - 1)

/-- Theorem stating that for 5 students, there are 96 ways to arrange them with one specific student not in the front. -/
theorem restricted_arrangements_five_students :
  arrangementsWithRestriction 5 = 96 := by
  sorry

#eval arrangementsWithRestriction 5  -- This should output 96

end restricted_arrangements_five_students_l3863_386322


namespace race_end_people_count_l3863_386305

/-- The number of people in cars at the end of a race -/
def people_at_race_end (num_cars : ℕ) (initial_people_per_car : ℕ) (additional_passengers : ℕ) : ℕ :=
  num_cars * (initial_people_per_car + additional_passengers)

/-- Theorem stating the number of people at the end of the race -/
theorem race_end_people_count : 
  people_at_race_end 20 3 1 = 80 := by sorry

end race_end_people_count_l3863_386305


namespace mutually_exclusive_not_contradictory_l3863_386395

/-- Represents the total number of products -/
def total_products : ℕ := 5

/-- Represents the number of genuine products -/
def genuine_products : ℕ := 3

/-- Represents the number of defective products -/
def defective_products : ℕ := 2

/-- Represents the number of products selected -/
def selected_products : ℕ := 2

/-- Represents the event of selecting exactly one defective product -/
def event_one_defective (selected : ℕ) : Prop :=
  selected = 1

/-- Represents the event of selecting exactly two genuine products -/
def event_two_genuine (selected : ℕ) : Prop :=
  selected = 2

/-- Theorem stating that the events are mutually exclusive and not contradictory -/
theorem mutually_exclusive_not_contradictory :
  (¬ (event_one_defective selected_products ∧ event_two_genuine selected_products)) ∧
  (∃ (x : ℕ), ¬ (event_one_defective x ∨ event_two_genuine x)) :=
sorry

end mutually_exclusive_not_contradictory_l3863_386395


namespace inverse_variation_problem_l3863_386380

/-- Two real numbers vary inversely if their product is constant. -/
def VaryInversely (p q : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ x : ℝ, p x * q x = k

theorem inverse_variation_problem (p q : ℝ → ℝ) 
    (h_inverse : VaryInversely p q)
    (h_initial : p 1 = 800 ∧ q 1 = 0.5) :
    p 2 = 1600 → q 2 = 0.25 := by
  sorry

end inverse_variation_problem_l3863_386380


namespace ball_placement_theorem_l3863_386302

/-- The number of ways to place 5 numbered balls into 5 numbered boxes, leaving one box empty -/
def ball_placement_count : ℕ := 1200

/-- The number of balls -/
def num_balls : ℕ := 5

/-- The number of boxes -/
def num_boxes : ℕ := 5

theorem ball_placement_theorem : 
  ball_placement_count = 1200 ∧ 
  num_balls = 5 ∧ 
  num_boxes = 5 := by sorry

end ball_placement_theorem_l3863_386302


namespace tooth_permutations_l3863_386331

def word_length : ℕ := 5
def t_occurrences : ℕ := 3
def o_occurrences : ℕ := 2

theorem tooth_permutations : 
  (word_length.factorial) / (t_occurrences.factorial * o_occurrences.factorial) = 10 := by
  sorry

end tooth_permutations_l3863_386331


namespace simplify_fraction_l3863_386377

theorem simplify_fraction : 20 * (9 / 14) * (1 / 18) = 5 / 7 := by
  sorry

end simplify_fraction_l3863_386377


namespace isabellas_hair_length_l3863_386338

/-- Isabella's hair length problem -/
theorem isabellas_hair_length :
  ∀ (current_length future_length growth : ℕ),
  future_length = 22 →
  growth = 4 →
  future_length = current_length + growth →
  current_length = 18 := by
  sorry

end isabellas_hair_length_l3863_386338


namespace no_solution_fractional_equation_l3863_386332

theorem no_solution_fractional_equation :
  ∀ x : ℝ, x ≠ 2 → ¬ (1 / (x - 2) = (1 - x) / (2 - x) - 3) := by
  sorry

end no_solution_fractional_equation_l3863_386332


namespace art_earnings_l3863_386388

/-- The total money earned from an art contest prize and selling paintings -/
def total_money_earned (prize : ℕ) (num_paintings : ℕ) (price_per_painting : ℕ) : ℕ :=
  prize + num_paintings * price_per_painting

/-- Theorem: Given a prize of $150 and selling 3 paintings for $50 each, the total money earned is $300 -/
theorem art_earnings : total_money_earned 150 3 50 = 300 := by
  sorry

end art_earnings_l3863_386388


namespace farmer_land_calculation_l3863_386315

/-- The total land owned by a farmer, given the following conditions:
  * 90% of the total land is cleared for planting
  * 10% of the cleared land is planted with grapes
  * 80% of the cleared land is planted with potatoes
  * The remaining cleared land (450 acres) is planted with tomatoes
-/
def farmer_land : ℝ := 1666.67

/-- The proportion of total land that is cleared for planting -/
def cleared_proportion : ℝ := 0.90

/-- The proportion of cleared land planted with grapes -/
def grape_proportion : ℝ := 0.10

/-- The proportion of cleared land planted with potatoes -/
def potato_proportion : ℝ := 0.80

/-- The amount of cleared land planted with tomatoes (in acres) -/
def tomato_acres : ℝ := 450

/-- Theorem stating that the farmer's land calculation is correct -/
theorem farmer_land_calculation (ε : ℝ) (hε : ε > 0) :
  ∃ (land : ℝ), abs (land - farmer_land) < ε ∧
  cleared_proportion * land * (1 - grape_proportion - potato_proportion) = tomato_acres :=
sorry

end farmer_land_calculation_l3863_386315


namespace min_ratio_logarithmic_intersections_l3863_386343

theorem min_ratio_logarithmic_intersections (m : ℝ) (h : m > 0) :
  let f (m : ℝ) := (2^m - 2^(8/(2*m+1))) / (2^(-m) - 2^(-8/(2*m+1)))
  ∀ x > 0, f m ≥ 8 * Real.sqrt 2 ∧ ∃ m₀ > 0, f m₀ = 8 * Real.sqrt 2 := by
  sorry

end min_ratio_logarithmic_intersections_l3863_386343
