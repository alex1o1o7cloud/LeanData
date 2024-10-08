import Mathlib

namespace june_earnings_l144_144505

theorem june_earnings
  (total_clovers : ℕ)
  (clover_3_petals_percentage : ℝ)
  (clover_2_petals_percentage : ℝ)
  (clover_4_petals_percentage : ℝ)
  (earnings_per_clover : ℝ) :
  total_clovers = 200 →
  clover_3_petals_percentage = 0.75 →
  clover_2_petals_percentage = 0.24 →
  clover_4_petals_percentage = 0.01 →
  earnings_per_clover = 1 →
  (total_clovers * earnings_per_clover) = 200 := by
  sorry

end june_earnings_l144_144505


namespace total_flowers_sold_l144_144012

/-
Ginger owns a flower shop, where she sells roses, lilacs, and gardenias.
On Tuesday, she sold three times more roses than lilacs, and half as many gardenias as lilacs.
If she sold 10 lilacs, prove that the total number of flowers sold on Tuesday is 45.
-/

theorem total_flowers_sold
    (lilacs roses gardenias : ℕ)
    (h_lilacs : lilacs = 10)
    (h_roses : roses = 3 * lilacs)
    (h_gardenias : gardenias = lilacs / 2)
    (ht : lilacs + roses + gardenias = 45) :
    lilacs + roses + gardenias = 45 :=
by sorry

end total_flowers_sold_l144_144012


namespace binom_odd_n_eq_2_pow_m_minus_1_l144_144050

open Nat

/-- For which n will binom n k be odd for every 0 ≤ k ≤ n?
    Prove that n = 2^m - 1 for some m ≥ 1. -/
theorem binom_odd_n_eq_2_pow_m_minus_1 (n : ℕ) :
  (∀ k : ℕ, k ≤ n → Nat.choose n k % 2 = 1) ↔ (∃ m : ℕ, m ≥ 1 ∧ n = 2^m - 1) :=
by
  sorry

end binom_odd_n_eq_2_pow_m_minus_1_l144_144050


namespace magic_square_expression_l144_144215

theorem magic_square_expression : 
  let a := 8
  let b := 6
  let c := 14
  let d := 10
  let e := 11
  let f := 5
  let g := 3
  a - b - c + d + e + f - g = 11 :=
by
  sorry

end magic_square_expression_l144_144215


namespace mass_percentage_O_in_Al2_CO3_3_l144_144495

-- Define the atomic masses
def atomic_mass_Al : Float := 26.98
def atomic_mass_C : Float := 12.01
def atomic_mass_O : Float := 16.00

-- Define the formula of aluminum carbonate
def Al_count : Nat := 2
def C_count : Nat := 3
def O_count : Nat := 9

-- Define the molar mass calculation
def molar_mass_Al2_CO3_3 : Float :=
  (Al_count.toFloat * atomic_mass_Al) + 
  (C_count.toFloat * atomic_mass_C) + 
  (O_count.toFloat * atomic_mass_O)

-- Define the mass of oxygen in aluminum carbonate
def mass_O_in_Al2_CO3_3 : Float := O_count.toFloat * atomic_mass_O

-- Define the mass percentage of oxygen in aluminum carbonate
def mass_percentage_O : Float := (mass_O_in_Al2_CO3_3 / molar_mass_Al2_CO3_3) * 100

-- Proof statement
theorem mass_percentage_O_in_Al2_CO3_3 :
  mass_percentage_O = 61.54 := by
  sorry

end mass_percentage_O_in_Al2_CO3_3_l144_144495


namespace abba_divisible_by_11_aaabbb_divisible_by_37_ababab_divisible_by_7_abab_baba_divisible_by_9_and_101_l144_144873

/-- Part 1: Prove that the number \overline{abba} is divisible by 11 -/
theorem abba_divisible_by_11 (a b : ℕ) : 11 ∣ (1000 * a + 100 * b + 10 * b + a) :=
sorry

/-- Part 2: Prove that the number \overline{aaabbb} is divisible by 37 -/
theorem aaabbb_divisible_by_37 (a b : ℕ) : 37 ∣ (1000 * 111 * a + 111 * b) :=
sorry

/-- Part 3: Prove that the number \overline{ababab} is divisible by 7 -/
theorem ababab_divisible_by_7 (a b : ℕ) : 7 ∣ (100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b) :=
sorry

/-- Part 4: Prove that the number \overline{abab} - \overline{baba} is divisible by 9 and 101 -/
theorem abab_baba_divisible_by_9_and_101 (a b : ℕ) :
  9 ∣ (1000 * a + 100 * b + 10 * a + b - (1000 * b + 100 * a + 10 * b + a)) ∧
  101 ∣ (1000 * a + 100 * b + 10 * a + b - (1000 * b + 100 * a + 10 * b + a)) :=
sorry

end abba_divisible_by_11_aaabbb_divisible_by_37_ababab_divisible_by_7_abab_baba_divisible_by_9_and_101_l144_144873


namespace gigi_initial_batches_l144_144831

-- Define the conditions
def flour_per_batch := 2 
def initial_flour := 20 
def remaining_flour := 14 
def future_batches := 7

-- Prove the number of batches initially baked is 3
theorem gigi_initial_batches :
  (initial_flour - remaining_flour) / flour_per_batch = 3 :=
by
  sorry

end gigi_initial_batches_l144_144831


namespace find_cost_of_book_sold_at_loss_l144_144330

-- Definitions from the conditions
def total_cost (C1 C2 : ℝ) : Prop := C1 + C2 = 540
def selling_price_loss (C1 : ℝ) : ℝ := 0.85 * C1
def selling_price_gain (C2 : ℝ) : ℝ := 1.19 * C2
def same_selling_price (SP1 SP2 : ℝ) : Prop := SP1 = SP2

theorem find_cost_of_book_sold_at_loss (C1 C2 : ℝ) 
  (h1 : total_cost C1 C2) 
  (h2 : same_selling_price (selling_price_loss C1) (selling_price_gain C2)) :
  C1 = 315 :=
by {
   sorry
}

end find_cost_of_book_sold_at_loss_l144_144330


namespace geometric_sequence_condition_l144_144481

variable (a b c : ℝ)

-- Condition: For a, b, c to form a geometric sequence.
def is_geometric_sequence (a b c : ℝ) : Prop :=
  (b ≠ 0) ∧ (b^2 = a * c)

-- Given that a, b, c are real numbers
-- Prove that ac = b^2 is a necessary but not sufficient condition for a, b, c to form a geometric sequence.
theorem geometric_sequence_condition (a b c : ℝ) (h : a * c = b^2) :
  ¬ (∃ b : ℝ, b^2 = a * c → (is_geometric_sequence a b c)) :=
sorry

end geometric_sequence_condition_l144_144481


namespace question_1_question_2_question_3_question_4_l144_144643

-- Define each condition as a theorem
theorem question_1 (explanation: String) : explanation = "providing for the living" :=
  sorry

theorem question_2 (usage: String) : usage = "structural auxiliary word, placed between subject and predicate, negating sentence independence" :=
  sorry

theorem question_3 (explanation: String) : explanation = "The Shang dynasty called it 'Xu,' and the Zhou dynasty called it 'Xiang.'" :=
  sorry

theorem question_4 (analysis: String) : analysis = "The statement about the 'ultimate ideal' is incorrect; the original text states that 'enabling people to live and die without regret' is 'the beginning of the King's Way.'" :=
  sorry

end question_1_question_2_question_3_question_4_l144_144643


namespace expression_evaluation_l144_144604

theorem expression_evaluation (x : ℤ) (hx : x = 4) : 5 * x + 3 - x^2 = 7 :=
by
  sorry

end expression_evaluation_l144_144604


namespace total_savings_l144_144825

def weekly_savings : ℕ := 15
def weeks_per_cycle : ℕ := 60
def number_of_cycles : ℕ := 5

theorem total_savings :
  (weekly_savings * weeks_per_cycle) * number_of_cycles = 4500 := 
sorry

end total_savings_l144_144825


namespace distinct_zeros_arithmetic_geometric_sequence_l144_144656

theorem distinct_zeros_arithmetic_geometric_sequence 
  (a b p q : ℝ)
  (h1 : a ≠ b)
  (h2 : a + b = p)
  (h3 : ab = q)
  (h4 : p > 0)
  (h5 : q > 0)
  (h6 : (a = 4 ∧ b = 1) ∨ (a = 1 ∧ b = 4))
  : p + q = 9 := 
sorry

end distinct_zeros_arithmetic_geometric_sequence_l144_144656


namespace part1_part2_part3_l144_144603

-- Part 1
theorem part1 (x : ℝ) (h : abs (x + 2) = abs (x - 4)) : x = 1 :=
by
  sorry

-- Part 2
theorem part2 (x : ℝ) (h : abs (x + 2) + abs (x - 4) = 8) : x = -3 ∨ x = 5 :=
by
  sorry

-- Part 3
theorem part3 (t : ℝ) :
  let M := -2 - t
  let N := 4 - 3 * t
  (abs M = abs (M - N) → t = 1/2) ∧ 
  (N = 0 → t = 4/3) ∧
  (abs N = abs (N - M) → t = 2) ∧
  (M = N → t = 3) ∧
  (abs (M - N) = abs (2 * M) → t = 8) :=
by
  sorry

end part1_part2_part3_l144_144603


namespace boys_at_beginning_is_15_l144_144174

noncomputable def number_of_boys_at_beginning (B : ℝ) : Prop :=
  let girls_start := 1.20 * B
  let girls_end := 2 * girls_start
  let total_students := B + girls_end
  total_students = 51 

theorem boys_at_beginning_is_15 : number_of_boys_at_beginning 15 := 
  by
  -- Sorry is added to skip the proof
  sorry

end boys_at_beginning_is_15_l144_144174


namespace desired_alcohol_percentage_is_18_l144_144971

noncomputable def final_alcohol_percentage (volume_x volume_y : ℕ) (percentage_x percentage_y : ℚ) : ℚ :=
  let total_volume := (volume_x + volume_y)
  let total_alcohol := (percentage_x * volume_x + percentage_y * volume_y)
  total_alcohol / total_volume * 100

theorem desired_alcohol_percentage_is_18 : 
  final_alcohol_percentage 300 200 0.10 0.30 = 18 := 
  sorry

end desired_alcohol_percentage_is_18_l144_144971


namespace S10_value_l144_144906

def sequence_sum (n : ℕ) : ℕ :=
  (2^(n+1)) - 2 - n

theorem S10_value : sequence_sum 10 = 2036 := by
  sorry

end S10_value_l144_144906


namespace fraction_decimal_representation_l144_144016

noncomputable def fraction_as_term_dec : ℚ := 47 / (2^3 * 5^4)

theorem fraction_decimal_representation : fraction_as_term_dec = 0.0094 :=
by
  sorry

end fraction_decimal_representation_l144_144016


namespace simplify_root_exponentiation_l144_144007

theorem simplify_root_exponentiation : (7 ^ (1 / 3) : ℝ) ^ 6 = 49 := by
  sorry

end simplify_root_exponentiation_l144_144007


namespace largest_two_digit_number_l144_144668

-- Define the conditions and the theorem to be proven
theorem largest_two_digit_number (n : ℕ) : 
  (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 5 = 4) ∧ (10 ≤ n) ∧ (n < 100) → n = 84 := by
  sorry

end largest_two_digit_number_l144_144668


namespace extra_yellow_balls_dispatched_eq_49_l144_144639

-- Define the given conditions
def ordered_balls : ℕ := 114
def white_balls : ℕ := ordered_balls / 2
def yellow_balls := ordered_balls / 2

-- Define the additional yellow balls dispatched and the ratio condition
def dispatch_error_ratio : ℚ := 8 / 15

-- The statement to prove the number of extra yellow balls dispatched
theorem extra_yellow_balls_dispatched_eq_49
  (ordered_balls_rounded : ordered_balls = 114)
  (white_balls_57 : white_balls = 57)
  (yellow_balls_57 : yellow_balls = 57)
  (ratio_condition : white_balls / (yellow_balls + x) = dispatch_error_ratio) :
  x = 49 :=
  sorry

end extra_yellow_balls_dispatched_eq_49_l144_144639


namespace cost_per_package_l144_144686

theorem cost_per_package
  (parents : ℕ)
  (brothers : ℕ)
  (spouses_per_brother : ℕ)
  (children_per_brother : ℕ)
  (total_cost : ℕ)
  (num_packages : ℕ)
  (h1 : parents = 2)
  (h2 : brothers = 3)
  (h3 : spouses_per_brother = 1)
  (h4 : children_per_brother = 2)
  (h5 : total_cost = 70)
  (h6 : num_packages = parents + brothers + brothers * spouses_per_brother + brothers * children_per_brother) :
  total_cost / num_packages = 5 :=
by
  -- Proof goes here
  sorry

end cost_per_package_l144_144686


namespace Miss_Darlington_total_blueberries_l144_144564

-- Conditions
def initial_basket := 20
def additional_baskets := 9

-- Definition and statement to be proved
theorem Miss_Darlington_total_blueberries :
  initial_basket + additional_baskets * initial_basket = 200 :=
by
  sorry

end Miss_Darlington_total_blueberries_l144_144564


namespace speed_increase_71_6_percent_l144_144472

theorem speed_increase_71_6_percent (S : ℝ) (hS : 0 < S) : 
    let S₁ := S * 1.30
    let S₂ := S₁ * 1.10
    let S₃ := S₂ * 1.20
    (S₃ - S) / S * 100 = 71.6 :=
by
  let S₁ := S * 1.30
  let S₂ := S₁ * 1.10
  let S₃ := S₂ * 1.20
  sorry

end speed_increase_71_6_percent_l144_144472


namespace volume_of_inscribed_sphere_l144_144209

theorem volume_of_inscribed_sphere (a : ℝ) (π : ℝ) (h : a = 6) : 
  (4 / 3 * π * (a / 2) ^ 3) = 36 * π :=
by
  sorry

end volume_of_inscribed_sphere_l144_144209


namespace charity_donation_ratio_l144_144111

theorem charity_donation_ratio :
  let total_winnings := 114
  let hot_dog_cost := 2
  let remaining_amount := 55
  let donation_amount := 114 - (remaining_amount + hot_dog_cost)
  donation_amount = 55 :=
by
  sorry

end charity_donation_ratio_l144_144111


namespace percentage_of_alcohol_in_second_vessel_l144_144625

-- Define the problem conditions
def capacity1 : ℝ := 2
def percentage1 : ℝ := 0.35
def alcohol1 := capacity1 * percentage1

def capacity2 : ℝ := 6 
def percentage2 (x : ℝ) : ℝ := 0.01 * x
def alcohol2 (x : ℝ) := capacity2 * percentage2 x

def total_capacity : ℝ := 8
def final_percentage : ℝ := 0.37
def total_alcohol := total_capacity * final_percentage

theorem percentage_of_alcohol_in_second_vessel (x : ℝ) :
  alcohol1 + alcohol2 x = total_alcohol → x = 37.67 :=
by sorry

end percentage_of_alcohol_in_second_vessel_l144_144625


namespace combinations_eight_choose_three_l144_144965

theorem combinations_eight_choose_three : Nat.choose 8 3 = 56 := by
  sorry

end combinations_eight_choose_three_l144_144965


namespace percentage_increase_salary_l144_144728

theorem percentage_increase_salary (S : ℝ) (P : ℝ) (h1 : 1.16 * S = 348) (h2 : S + P * S = 375) : P = 0.25 :=
by
  sorry

end percentage_increase_salary_l144_144728


namespace income_percent_greater_l144_144611

variable (A B : ℝ)

-- Condition: A's income is 25% less than B's income
def income_condition (A B : ℝ) : Prop :=
  A = 0.75 * B

-- Statement: B's income is 33.33% greater than A's income
theorem income_percent_greater (A B : ℝ) (h : income_condition A B) :
  B = A * (4 / 3) := by
sorry

end income_percent_greater_l144_144611


namespace Eva_is_6_l144_144389

def ages : Set ℕ := {2, 4, 6, 8, 10}

def conditions : Prop :=
  ∃ a b, a ∈ ages ∧ b ∈ ages ∧ a + b = 12 ∧
  b ≠ 2 ∧ b ≠ 10 ∧ a ≠ 2 ∧ a ≠ 10 ∧
  (∃ c d, c ∈ ages ∧ d ∈ ages ∧ c = 2 ∧ d = 10 ∧
           (∃ e, e ∈ ages ∧ e = 4 ∧
           ∃ eva, eva ∈ ages ∧ eva ≠ 2 ∧ eva ≠ 4 ∧ eva ≠ 8 ∧ eva ≠ 10 ∧ eva = 6))

theorem Eva_is_6 (h : conditions) : ∃ eva, eva ∈ ages ∧ eva = 6 := sorry

end Eva_is_6_l144_144389


namespace parabola_problem_l144_144272

noncomputable def p_value_satisfy_all_conditions (p : ℝ) : Prop :=
  ∃ (F : ℝ × ℝ) (A B : ℝ × ℝ),
    F = (p / 2, 0) ∧
    (A.2 = A.1 - p / 2 ∧ (A.2)^2 = 2 * p * A.1) ∧
    (B.2 = B.1 - p / 2 ∧ (B.2)^2 = 2 * p * B.1) ∧
    (A.1 + B.1) / 2 = 3 * p / 2 ∧
    (A.2 + B.2) / 2 = p ∧
    (p - 2 = -3 * p / 2)

theorem parabola_problem : ∃ (p : ℝ), p_value_satisfy_all_conditions p ∧ p = 4 / 5 :=
by
  sorry

end parabola_problem_l144_144272


namespace find_difference_l144_144013

variables (x y : ℝ)

theorem find_difference (h1 : x * (y + 2) = 100) (h2 : y * (x + 2) = 60) : x - y = 20 :=
sorry

end find_difference_l144_144013


namespace sum_of_circumferences_eq_28pi_l144_144920

theorem sum_of_circumferences_eq_28pi (R r : ℝ) (h1 : r = (1:ℝ)/3 * R) (h2 : R - r = 7) : 
  2 * Real.pi * R + 2 * Real.pi * r = 28 * Real.pi :=
by
  sorry

end sum_of_circumferences_eq_28pi_l144_144920


namespace range_of_a_l144_144101

noncomputable def satisfies_inequality (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → (x^2 + 1) * Real.exp x ≥ a * x^2

theorem range_of_a (a : ℝ) : satisfies_inequality a ↔ a ≤ 2 * Real.exp 1 :=
by
  sorry

end range_of_a_l144_144101


namespace problem_statement_l144_144521

theorem problem_statement (h1 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2) :
  (Real.pi / (Real.sqrt 3 - 1))^0 - (Real.cos (Real.pi / 6))^2 = 1 / 4 := by
  sorry

end problem_statement_l144_144521


namespace circle_positional_relationship_l144_144211

noncomputable def r1 : ℝ := 2
noncomputable def r2 : ℝ := 3
noncomputable def d : ℝ := 5

theorem circle_positional_relationship :
  d = r1 + r2 → "externally tangent" = "externally tangent" := by
  intro h
  exact rfl

end circle_positional_relationship_l144_144211


namespace min_a1_value_l144_144892

theorem min_a1_value (a : ℕ → ℝ) :
  (∀ n > 1, a n = 9 * a (n-1) - 2 * n) →
  (∀ n, a n > 0) →
  (∀ x, (∀ n > 1, a n = 9 * a (n-1) - 2 * n) → (∀ n, a n > 0) → x ≥ a 1) →
  a 1 = 499.25 / 648 :=
sorry

end min_a1_value_l144_144892


namespace part1_part2_l144_144784

theorem part1 (A B : ℝ) (c : ℝ) (cos_A : ℝ) (tan_half_B_add_cot_half_B: ℝ) 
  (h1: cos_A = 5 / 13) 
  (h2: tan_half_B_add_cot_half_B = 10 / 3) 
  (pos_c: c = 21) :
  ∃ (cos_AB: ℝ), cos_AB = 56 / 65 :=
by {
  sorry
}

theorem part2 (A B : ℝ) (c : ℝ) (cos_A : ℝ) (tan_half_B_add_cot_half_B: ℝ) 
  (h1: cos_A = 5 / 13) 
  (h2: tan_half_B_add_cot_half_B = 10 / 3) 
  (pos_c: c = 21) :
  ∃ (area: ℝ), area = 126 :=
by {
  sorry
}

end part1_part2_l144_144784


namespace complex_number_solution_l144_144909

def imaginary_unit : ℂ := Complex.I -- defining the imaginary unit

theorem complex_number_solution (z : ℂ) (h : z / (z - imaginary_unit) = imaginary_unit) :
  z = (1 / 2 : ℂ) + (1 / 2 : ℂ) * imaginary_unit :=
sorry

end complex_number_solution_l144_144909


namespace speed_of_water_l144_144132

variable (v : ℝ)
variable (swimming_speed_still_water : ℝ := 10)
variable (time_against_current : ℝ := 8)
variable (distance_against_current : ℝ := 16)

theorem speed_of_water :
  distance_against_current = (swimming_speed_still_water - v) * time_against_current ↔ v = 8 := by
  sorry

end speed_of_water_l144_144132


namespace sphere_surface_area_l144_144271

-- Define the conditions
def points_on_sphere (A B C : Type) := 
  ∃ (AB BC AC : Real), AB = 6 ∧ BC = 8 ∧ AC = 10

-- Define the distance condition
def distance_condition (R : Real) := 
  ∃ (d : Real), d = R / 2

-- Define the main theorem
theorem sphere_surface_area 
  (A B C : Type) 
  (h_points : points_on_sphere A B C) 
  (h_distance : ∃ R : Real, distance_condition R) : 
  4 * Real.pi * (10 / 3 * Real.sqrt 3) ^ 2 = 400 / 3 * Real.pi := 
by 
  sorry

end sphere_surface_area_l144_144271


namespace quadratic_rewrite_ab_l144_144044

theorem quadratic_rewrite_ab : 
  ∃ (a b c : ℤ), (16*(x:ℝ)^2 - 40*x + 24 = (a*x + b)^2 + c) ∧ (a * b = -20) :=
by {
  sorry
}

end quadratic_rewrite_ab_l144_144044


namespace sin_45_eq_sqrt2_div_2_l144_144327

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l144_144327


namespace total_cost_of_books_l144_144256

theorem total_cost_of_books (total_children : ℕ) (n : ℕ) (extra_payment_per_child : ℕ) (cost : ℕ) :
  total_children = 12 →
  n = 2 →
  extra_payment_per_child = 10 →
  (total_children - n) * extra_payment_per_child = 100 →
  cost = 600 :=
by
  intros h1 h2 h3 h4
  sorry

end total_cost_of_books_l144_144256


namespace speed_ratio_l144_144156

theorem speed_ratio (v1 v2 : ℝ) 
  (h1 : v1 > 0) 
  (h2 : v2 > 0) 
  (h : v2 / v1 - v1 / v2 = 35 / 60) : v1 / v2 = 3 / 4 := 
sorry

end speed_ratio_l144_144156


namespace Paul_work_time_l144_144863

def work_completed (rate: ℚ) (time: ℚ) : ℚ := rate * time

noncomputable def George_work_rate : ℚ := 3 / 5 / 9

noncomputable def combined_work_rate : ℚ := 2 / 5 / 4

noncomputable def Paul_work_rate : ℚ := combined_work_rate - George_work_rate

theorem Paul_work_time :
  (work_completed Paul_work_rate 30) = 1 :=
by
  have h_george_rate : George_work_rate = 1 / 15 :=
    by norm_num [George_work_rate]
  have h_combined_rate : combined_work_rate = 1 / 10 :=
    by norm_num [combined_work_rate]
  have h_paul_rate : Paul_work_rate = 1 / 30 :=
    by norm_num [Paul_work_rate, h_combined_rate, h_george_rate]
  sorry -- Complete proof statement here

end Paul_work_time_l144_144863


namespace train_length_l144_144660

theorem train_length (L V : ℝ) (h1 : L = V * 120) (h2 : L + 1000 = V * 220) : L = 1200 := 
by
  sorry

end train_length_l144_144660


namespace avg_age_9_proof_l144_144851

-- Definitions of the given conditions
def total_persons := 16
def avg_age_all := 15
def total_age_all := total_persons * avg_age_all -- 240
def persons_5 := 5
def avg_age_5 := 14
def total_age_5 := persons_5 * avg_age_5 -- 70
def age_15th_person := 26
def persons_9 := 9

-- The theorem to prove the average age of the remaining 9 persons
theorem avg_age_9_proof : 
  total_age_all - total_age_5 - age_15th_person = persons_9 * 16 :=
by
  sorry

end avg_age_9_proof_l144_144851


namespace ratio_of_votes_l144_144351

theorem ratio_of_votes (randy_votes : ℕ) (shaun_votes : ℕ) (eliot_votes : ℕ)
  (h1 : randy_votes = 16)
  (h2 : shaun_votes = 5 * randy_votes)
  (h3 : eliot_votes = 160) :
  eliot_votes / shaun_votes = 2 :=
by
  sorry

end ratio_of_votes_l144_144351


namespace triangle_angle_bisector_l144_144290

theorem triangle_angle_bisector 
  (a b l : ℝ) (h1: a > 0) (h2: b > 0) (h3: l > 0) :
  ∃ α : ℝ, α = 2 * Real.arccos (l * (a + b) / (2 * a * b)) :=
by
  sorry

end triangle_angle_bisector_l144_144290


namespace simplify_fraction_l144_144538

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l144_144538


namespace min_5a2_plus_6a3_l144_144847

theorem min_5a2_plus_6a3 (a_1 a_2 a_3 : ℝ) (r : ℝ)
  (h1 : a_1 = 2)
  (h2 : a_2 = a_1 * r)
  (h3 : a_3 = a_1 * r^2) :
  5 * a_2 + 6 * a_3 ≥ -25 / 12 :=
by
  sorry

end min_5a2_plus_6a3_l144_144847


namespace polygons_ratio_four_three_l144_144881

theorem polygons_ratio_four_three : 
  ∃ (r k : ℕ), 3 ≤ r ∧ 3 ≤ k ∧ 
  (180 - (360 / r : ℝ)) / (180 - (360 / k : ℝ)) = 4 / 3 
  ∧ ((r, k) = (42,7) ∨ (r, k) = (18,6) ∨ (r, k) = (10,5) ∨ (r, k) = (6,4)) :=
sorry

end polygons_ratio_four_three_l144_144881


namespace cost_of_eight_CDs_l144_144512

theorem cost_of_eight_CDs (cost_of_two_CDs : ℕ) (h : cost_of_two_CDs = 36) : 8 * (cost_of_two_CDs / 2) = 144 := by
  sorry

end cost_of_eight_CDs_l144_144512


namespace total_baseball_fans_l144_144575

variable (Y M R : ℕ)

open Nat

theorem total_baseball_fans (h1 : 3 * M = 2 * Y) 
    (h2 : 4 * R = 5 * M) 
    (h3 : M = 96) : Y + M + R = 360 := by
  sorry

end total_baseball_fans_l144_144575


namespace required_hours_for_fifth_week_l144_144142

def typical_hours_needed (week1 week2 week3 week4 week5 add_hours total_weeks target_avg : ℕ) : ℕ :=
  if (week1 + week2 + week3 + week4 + week5 + add_hours) / total_weeks = target_avg then 
    week5 
  else 
    0

theorem required_hours_for_fifth_week :
  typical_hours_needed 10 14 11 9 x 1 5 12 = 15 :=
by
  sorry

end required_hours_for_fifth_week_l144_144142


namespace either_p_or_q_false_suff_not_p_true_l144_144577

theorem either_p_or_q_false_suff_not_p_true (p q : Prop) : (p ∨ q = false) → (¬p = true) :=
by
  sorry

end either_p_or_q_false_suff_not_p_true_l144_144577


namespace terminal_zeros_of_product_l144_144386

noncomputable def prime_factors (n : ℕ) : List (ℕ × ℕ) := sorry

theorem terminal_zeros_of_product (n m : ℕ) (hn : prime_factors n = [(2, 1), (5, 2)])
 (hm : prime_factors m = [(2, 3), (3, 2), (5, 1)]) : 
  (∃ k, n * m = 10^k) ∧ k = 3 :=
by {
  sorry
}

end terminal_zeros_of_product_l144_144386


namespace choose_7_starters_with_at_least_one_quadruplet_l144_144414

-- Given conditions
variable (n : ℕ := 18) -- total players
variable (k : ℕ := 7)  -- number of starters
variable (q : ℕ := 4)  -- number of quadruplets

-- Lean statement
theorem choose_7_starters_with_at_least_one_quadruplet 
  (h : n = 18) 
  (h1 : k = 7) 
  (h2 : q = 4) :
  (Nat.choose 18 7 - Nat.choose 14 7) = 28392 :=
by
  sorry

end choose_7_starters_with_at_least_one_quadruplet_l144_144414


namespace expression_value_l144_144703

-- Step c: Definitions based on conditions
def base1 : ℤ := -2
def exponent1 : ℕ := 4^2
def base2 : ℕ := 1
def exponent2 : ℕ := 3^3

-- The Lean statement for the problem
theorem expression_value :
  base1 ^ exponent1 + base2 ^ exponent2 = 65537 := by
  sorry

end expression_value_l144_144703


namespace sum_of_y_for_f_l144_144736

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x + 5

theorem sum_of_y_for_f (y1 y2 y3 : ℝ) :
  (∀ y, 64 * y^3 - 8 * y + 5 = 7) →
  y1 + y2 + y3 = 0 :=
by
  -- placeholder for actual proof
  sorry

end sum_of_y_for_f_l144_144736


namespace participants_count_l144_144543

theorem participants_count (F M : ℕ)
  (hF2 : F / 2 = 110)
  (hM4 : M / 4 = 330 - F - M / 3)
  (hFm : (F + M) / 3 = F / 2 + M / 4) :
  F + M = 330 :=
sorry

end participants_count_l144_144543


namespace hieu_catches_up_beatrice_in_5_minutes_probability_beatrice_hieu_same_place_range_of_x_for_meeting_probability_l144_144032

namespace CatchUpProblem

-- Part (a)
theorem hieu_catches_up_beatrice_in_5_minutes :
  ∀ (d_b_walked : ℕ) (relative_speed : ℕ) (catch_up_time : ℕ),
  d_b_walked = 5 / 6 ∧ relative_speed = 10 ∧ catch_up_time = 5 :=
sorry

-- Part (b)(i)
theorem probability_beatrice_hieu_same_place :
  ∀ (total_pairs : ℕ) (valid_pairs : ℕ) (probability : Rat),
  total_pairs = 3600 ∧ valid_pairs = 884 ∧ probability = 221 / 900 :=
sorry

-- Part (b)(ii)
theorem range_of_x_for_meeting_probability :
  ∀ (probability : Rat) (valid_pairs : ℕ) (total_pairs : ℕ) (lower_bound : ℕ) (upper_bound : ℕ),
  probability = 13 / 200 ∧ valid_pairs = 234 ∧ total_pairs = 3600 ∧ 
  lower_bound = 10 ∧ upper_bound = 120 / 11 :=
sorry

end CatchUpProblem

end hieu_catches_up_beatrice_in_5_minutes_probability_beatrice_hieu_same_place_range_of_x_for_meeting_probability_l144_144032


namespace tony_lift_ratio_l144_144765

noncomputable def curl_weight := 90
noncomputable def military_press_weight := 2 * curl_weight
noncomputable def squat_weight := 900

theorem tony_lift_ratio : 
  squat_weight / military_press_weight = 5 :=
by
  sorry

end tony_lift_ratio_l144_144765


namespace AlbertTookAwayCandies_l144_144948

-- Define the parameters and conditions given in the problem
def PatriciaStartCandies : ℕ := 76
def PatriciaEndCandies : ℕ := 71

-- Define the statement that proves the number of candies Albert took away
theorem AlbertTookAwayCandies :
  PatriciaStartCandies - PatriciaEndCandies = 5 := by
  sorry

end AlbertTookAwayCandies_l144_144948


namespace digit_difference_l144_144352

theorem digit_difference (x y : ℕ) (h : 10 * x + y - (10 * y + x) = 45) : x - y = 5 :=
sorry

end digit_difference_l144_144352


namespace pentagon_area_proof_l144_144323

noncomputable def area_of_pentagon : ℕ :=
  let side1 := 18
  let side2 := 25
  let side3 := 30
  let side4 := 28
  let side5 := 25
  -- Assuming the total area calculated from problem's conditions
  950

theorem pentagon_area_proof : area_of_pentagon = 950 := by
  sorry

end pentagon_area_proof_l144_144323


namespace books_taken_off_l144_144741

def books_initially : ℝ := 38.0
def books_remaining : ℝ := 28.0

theorem books_taken_off : books_initially - books_remaining = 10 := by
  sorry

end books_taken_off_l144_144741


namespace exists_k_simplifies_expression_to_5x_squared_l144_144903

theorem exists_k_simplifies_expression_to_5x_squared :
  ∃ k : ℝ, (∀ x : ℝ, (x - k * x) * (2 * x - k * x) - 3 * x * (2 * x - k * x) = 5 * x^2) :=
by
  sorry

end exists_k_simplifies_expression_to_5x_squared_l144_144903


namespace symmetric_points_x_axis_l144_144220

theorem symmetric_points_x_axis (a b : ℤ) 
  (h1 : a - 1 = 2) (h2 : 5 = -(b - 1)) : (a + b) ^ 2023 = -1 := 
by
  -- The proof steps will go here.
  sorry

end symmetric_points_x_axis_l144_144220


namespace quadratic_touches_x_axis_l144_144342

theorem quadratic_touches_x_axis (a : ℝ) : 
  (∃ x : ℝ, 2 * x ^ 2 - 8 * x + a = 0) ∧ (∀ y : ℝ, y^2 - 4 * a = 0 → y = 0) → a = 8 := 
by 
  sorry

end quadratic_touches_x_axis_l144_144342


namespace track_width_eight_l144_144416

theorem track_width_eight (r1 r2 : ℝ) (h : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 16 * Real.pi) : r1 - r2 = 8 := 
sorry

end track_width_eight_l144_144416


namespace part1_part2_l144_144265

variables {a m n : ℝ}

theorem part1 (h1 : a^m = 2) (h2 : a^n = 3) : a^(4*m + 3*n) = 432 :=
by sorry

theorem part2 (h1 : a^m = 2) (h2 : a^n = 3) : a^(5*m - 2*n) = 32 / 9 :=
by sorry

end part1_part2_l144_144265


namespace compute_expr_l144_144956

theorem compute_expr {x : ℝ} (h : x = 5) : (x^6 - 2 * x^3 + 1) / (x^3 - 1) = 124 :=
by
  sorry

end compute_expr_l144_144956


namespace solution_set_l144_144344

-- Definitions representing the given conditions
def cond1 (x : ℝ) := x - 3 < 0
def cond2 (x : ℝ) := x + 1 ≥ 0

-- The problem: Prove the solution set is as given
theorem solution_set (x : ℝ) :
  (cond1 x) ∧ (cond2 x) ↔ -1 ≤ x ∧ x < 3 :=
by
  sorry

end solution_set_l144_144344


namespace max_k_mono_incr_binom_l144_144950

theorem max_k_mono_incr_binom :
  ∀ (k : ℕ), (k ≤ 11) → 
  (∀ i j : ℕ, 1 ≤ i → i < j → j ≤ k → (Nat.choose 10 (i - 1) < Nat.choose 10 (j - 1))) →
  k = 6 :=
by sorry

end max_k_mono_incr_binom_l144_144950


namespace total_time_spent_l144_144104

-- Definition of the problem conditions
def warm_up_time : ℕ := 10
def additional_puzzles : ℕ := 2
def multiplier : ℕ := 3

-- Statement to prove the total time spent solving puzzles
theorem total_time_spent : warm_up_time + (additional_puzzles * (multiplier * warm_up_time)) = 70 :=
by
  sorry

end total_time_spent_l144_144104


namespace g_of_3_equals_5_l144_144221

def g (x : ℝ) : ℝ := 2 * (x - 2) + 3

theorem g_of_3_equals_5 :
  g 3 = 5 :=
by
  sorry

end g_of_3_equals_5_l144_144221


namespace domain_of_log_function_l144_144502

open Real

noncomputable def domain_of_function : Set ℝ :=
  {x | x > 2 ∨ x < -1}

theorem domain_of_log_function :
  ∀ x : ℝ, (x^2 - x - 2 > 0) ↔ (x > 2 ∨ x < -1) :=
by
  intro x
  exact sorry

end domain_of_log_function_l144_144502


namespace number_of_real_solutions_l144_144385

noncomputable def greatest_integer (x: ℝ) : ℤ :=
  ⌊x⌋

def equation (x: ℝ) :=
  4 * x^2 - 40 * (greatest_integer x : ℝ) + 51 = 0

theorem number_of_real_solutions : 
  ∃ (x1 x2 x3 x4: ℝ), 
  equation x1 ∧ equation x2 ∧ equation x3 ∧ equation x4 ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 := 
sorry

end number_of_real_solutions_l144_144385


namespace quiz_score_of_dropped_student_l144_144437

theorem quiz_score_of_dropped_student 
    (avg_all : ℝ) (num_all : ℕ) (new_avg_remaining : ℝ) (num_remaining : ℕ)
    (total_all : ℝ := num_all * avg_all) (total_remaining : ℝ := num_remaining * new_avg_remaining) :
    avg_all = 61.5 → num_all = 16 → new_avg_remaining = 64 → num_remaining = 15 → (total_all - total_remaining = 24) :=
by
  intros h_avg_all h_num_all h_new_avg_remaining h_num_remaining
  rw [h_avg_all, h_new_avg_remaining, h_num_all, h_num_remaining]
  sorry

end quiz_score_of_dropped_student_l144_144437


namespace sum_of_quotient_and_remainder_is_184_l144_144186

theorem sum_of_quotient_and_remainder_is_184 
  (q r : ℕ)
  (h1 : 23 * 17 + 19 = q)
  (h2 : q * 10 = r)
  (h3 : r / 23 = 178)
  (h4 : r % 23 = 6) :
  178 + 6 = 184 :=
by
  -- Inform Lean that we are skipping the proof
  sorry

end sum_of_quotient_and_remainder_is_184_l144_144186


namespace minimum_value_y_l144_144249

noncomputable def y (x : ℝ) : ℝ := x + 1 / (x - 1)

theorem minimum_value_y (x : ℝ) (hx : x > 1) : ∃ A, (A = 3) ∧ (∀ y', y' = y x → y' ≥ A) := sorry

end minimum_value_y_l144_144249


namespace smallest_multiple_of_6_and_15_l144_144662

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ b = 30 := 
by 
  use 30 
  sorry

end smallest_multiple_of_6_and_15_l144_144662


namespace loss_percent_l144_144264

theorem loss_percent (C S : ℝ) (h : 100 * S = 40 * C) : ((C - S) / C) * 100 = 60 :=
by
  sorry

end loss_percent_l144_144264


namespace max_value_f_l144_144034

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 2 + Real.sqrt 3 * Real.cos x - 3 / 4

theorem max_value_f :
  ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x ≤ 1 :=
sorry

end max_value_f_l144_144034


namespace problem_statement_l144_144292

noncomputable def a : ℝ := 13 / 2
noncomputable def b : ℝ := -4

theorem problem_statement :
  ∀ k : ℝ, ∃ x : ℝ, (2 * k * x + a) / 3 = 2 + (x - b * k) / 6 ↔ x = 1 :=
by
  sorry

end problem_statement_l144_144292


namespace sum_underlined_numbers_non_negative_l144_144293

def sum_underlined_numbers (seq : Fin 100 → Int) : Bool :=
  let underlined_indices : List (Fin 100) :=
    List.range 100 |>.filter (λ i =>
      seq i > 0 ∨ (i < 99 ∧ seq i + seq (i + 1) > 0) ∨ (i < 98 ∧ seq i + seq (i + 1) + seq (i + 2) > 0))
  let underlined_sum : Int := underlined_indices.map (λ i => seq i) |>.sum
  underlined_sum ≤ 0

theorem sum_underlined_numbers_non_negative {seq : Fin 100 → Int} :
  ¬ sum_underlined_numbers seq :=
sorry

end sum_underlined_numbers_non_negative_l144_144293


namespace physics_majors_consecutive_probability_l144_144501

open Nat

-- Define the total number of seats and the specific majors
def totalSeats : ℕ := 10
def mathMajors : ℕ := 4
def physicsMajors : ℕ := 3
def chemistryMajors : ℕ := 2
def biologyMajors : ℕ := 1

-- Assuming a round table configuration
def probabilityPhysicsMajorsConsecutive : ℚ :=
  (3 * (Nat.factorial (totalSeats - physicsMajors))) / (Nat.factorial (totalSeats - 1))

-- Declare the theorem
theorem physics_majors_consecutive_probability : 
  probabilityPhysicsMajorsConsecutive = 1 / 24 :=
by
  sorry

end physics_majors_consecutive_probability_l144_144501


namespace boys_left_is_31_l144_144697

def initial_children : ℕ := 85
def girls_came_in : ℕ := 24
def final_children : ℕ := 78

noncomputable def compute_boys_left (initial : ℕ) (girls_in : ℕ) (final : ℕ) : ℕ :=
  (initial + girls_in) - final

theorem boys_left_is_31 :
  compute_boys_left initial_children girls_came_in final_children = 31 :=
by
  sorry

end boys_left_is_31_l144_144697


namespace budget_equality_year_l144_144571

theorem budget_equality_year :
  let budget_q_1990 := 540000
  let budget_v_1990 := 780000
  let annual_increase_q := 30000
  let annual_decrease_v := 10000

  let budget_q (n : ℕ) := budget_q_1990 + n * annual_increase_q
  let budget_v (n : ℕ) := budget_v_1990 - n * annual_decrease_v

  (∃ n : ℕ, budget_q n = budget_v n ∧ 1990 + n = 1996) :=
by
  sorry

end budget_equality_year_l144_144571


namespace simplify_expr1_simplify_expr2_l144_144601

-- Define the first problem with necessary conditions
theorem simplify_expr1 (a b : ℝ) (h : a ≠ b) : 
  (a / (a - b)) - (b / (b - a)) = (a + b) / (a - b) :=
by
  sorry

-- Define the second problem with necessary conditions
theorem simplify_expr2 (x : ℝ) (hx1 : x ≠ -3) (hx2 : x ≠ 4) (hx3 : x ≠ -4) :
  ((x - 4) / (x + 3)) / (x - 3 - (7 / (x + 3))) = 1 / (x + 4) :=
by
  sorry

end simplify_expr1_simplify_expr2_l144_144601


namespace gray_area_is_50pi_l144_144623

noncomputable section

-- Define the radii of the inner and outer circles
def R_inner : ℝ := 2.5
def R_outer : ℝ := 3 * R_inner

-- Area of circles
def A_inner : ℝ := Real.pi * R_inner^2
def A_outer : ℝ := Real.pi * R_outer^2

-- Define width of the gray region
def gray_width : ℝ := R_outer - R_inner

-- Gray area calculation
def A_gray : ℝ := A_outer - A_inner

-- The theorem stating the area of the gray region
theorem gray_area_is_50pi :
  gray_width = 5 → A_gray = 50 * Real.pi := by
  -- Here we assume the proof continues
  sorry

end gray_area_is_50pi_l144_144623


namespace partial_fraction_sum_zero_l144_144442

theorem partial_fraction_sum_zero (A B C D E : ℚ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5)) →
  A + B + C + D + E = 0 :=
by
  sorry

end partial_fraction_sum_zero_l144_144442


namespace rectangular_prism_diagonal_inequality_l144_144857

theorem rectangular_prism_diagonal_inequality 
  (a b c l : ℝ) 
  (h : l^2 = a^2 + b^2 + c^2) :
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := 
by sorry

end rectangular_prism_diagonal_inequality_l144_144857


namespace ms_warren_walking_speed_correct_l144_144746

noncomputable def walking_speed_proof : Prop :=
  let running_speed := 6 -- mph
  let running_time := 20 / 60 -- hours
  let total_distance := 3 -- miles
  let distance_ran := running_speed * running_time
  let distance_walked := total_distance - distance_ran
  let walking_time := 30 / 60 -- hours
  let walking_speed := distance_walked / walking_time
  walking_speed = 2

theorem ms_warren_walking_speed_correct (walking_speed_proof : Prop) : walking_speed_proof :=
by sorry

end ms_warren_walking_speed_correct_l144_144746


namespace bunnies_out_of_burrow_l144_144967

theorem bunnies_out_of_burrow:
  (3 * 60 * 10 * 20) = 36000 :=
by 
  sorry

end bunnies_out_of_burrow_l144_144967


namespace fractional_sum_equals_015025_l144_144638

theorem fractional_sum_equals_015025 :
  (2 / 20) + (8 / 200) + (3 / 300) + (5 / 40000) * 2 = 0.15025 := 
by
  sorry

end fractional_sum_equals_015025_l144_144638


namespace book_arrangement_count_l144_144663

theorem book_arrangement_count :
  let n := 6
  let identical_pairs := 2
  let total_arrangements_if_unique := n.factorial
  let ident_pair_correction := (identical_pairs.factorial * identical_pairs.factorial)
  (total_arrangements_if_unique / ident_pair_correction) = 180 := by
  sorry

end book_arrangement_count_l144_144663


namespace find_x_values_l144_144880

theorem find_x_values (x : ℝ) :
  (3 * x + 2 < (x - 1) ^ 2 ∧ (x - 1) ^ 2 < 9 * x + 1) ↔
  (x > (5 + Real.sqrt 29) / 2 ∧ x < 11) := 
by
  sorry

end find_x_values_l144_144880


namespace find_y_l144_144326

variable (a b c x : ℝ) (p q r y : ℝ)
variable (log : ℝ → ℝ) -- represents the logarithm function

-- Conditions as hypotheses
axiom log_eq : (log a) / p = (log b) / q
axiom log_eq' : (log b) / q = (log c) / r
axiom log_eq'' : (log c) / r = log x
axiom x_ne_one : x ≠ 1
axiom eq_exp : (b^3) / (a^2 * c) = x^y

-- Statement to be proven
theorem find_y : y = 3 * q - 2 * p - r := by
  sorry

end find_y_l144_144326


namespace xiao_ming_completion_days_l144_144410

/-
  Conditions:
  1. The total number of pages is 960.
  2. The planned number of days to finish the book is 20.
  3. Xiao Ming actually read 12 more pages per day than planned.

  Question:
  How many days did it actually take Xiao Ming to finish the book?

  Answer:
  The actual number of days to finish the book is 16 days.
-/

open Nat

theorem xiao_ming_completion_days :
  let total_pages := 960
  let planned_days := 20
  let additional_pages_per_day := 12
  let planned_pages_per_day := total_pages / planned_days
  let actual_pages_per_day := planned_pages_per_day + additional_pages_per_day
  let actual_days := total_pages / actual_pages_per_day
  actual_days = 16 :=
by
  let total_pages := 960
  let planned_days := 20
  let additional_pages_per_day := 12
  let planned_pages_per_day := total_pages / planned_days
  let actual_pages_per_day := planned_pages_per_day + additional_pages_per_day
  let actual_days := total_pages / actual_pages_per_day
  show actual_days = 16
  sorry

end xiao_ming_completion_days_l144_144410


namespace cube_inequality_contradiction_l144_144158

theorem cube_inequality_contradiction (a b : Real) (h : a > b) : ¬(a^3 <= b^3) := by
  sorry

end cube_inequality_contradiction_l144_144158


namespace arrangements_with_AB_together_l144_144706

theorem arrangements_with_AB_together (students : Finset α) (A B : α) (hA : A ∈ students) (hB : B ∈ students) (h_students : students.card = 5) : 
  ∃ n, n = 48 :=
sorry

end arrangements_with_AB_together_l144_144706


namespace Trumpington_marching_band_max_l144_144335

theorem Trumpington_marching_band_max (n : ℕ) (k : ℕ) 
  (h1 : 20 * n % 26 = 4)
  (h2 : n = 8 + 13 * k)
  (h3 : 20 * n < 1000) 
  : 20 * (8 + 13 * 3) = 940 := 
by
  sorry

end Trumpington_marching_band_max_l144_144335


namespace spending_difference_is_65_l144_144328

-- Definitions based on conditions
def ice_cream_cones : ℕ := 15
def pudding_cups : ℕ := 5
def ice_cream_cost_per_unit : ℝ := 5
def pudding_cost_per_unit : ℝ := 2

-- The solution requires the calculation of the total cost and the difference
def total_ice_cream_cost : ℝ := ice_cream_cones * ice_cream_cost_per_unit
def total_pudding_cost : ℝ := pudding_cups * pudding_cost_per_unit
def spending_difference : ℝ := total_ice_cream_cost - total_pudding_cost

-- Theorem statement proving the difference is 65
theorem spending_difference_is_65 : spending_difference = 65 := by
  -- The proof is omitted with sorry
  sorry

end spending_difference_is_65_l144_144328


namespace green_team_final_score_l144_144841

theorem green_team_final_score (G : ℕ) :
  (∀ G : ℕ, 68 = G + 29 → G = 39) :=
by
  sorry

end green_team_final_score_l144_144841


namespace utility_bills_total_l144_144106

-- Define the conditions
def fifty_bills := 3
def ten_dollar_bills := 2
def fifty_dollar_value := 50
def ten_dollar_value := 10

-- Prove the total utility bills amount
theorem utility_bills_total : (fifty_bills * fifty_dollar_value + ten_dollar_bills * ten_dollar_value) = 170 := by
  sorry

end utility_bills_total_l144_144106


namespace page_copy_cost_l144_144690

theorem page_copy_cost (cost_per_4_pages : ℕ) (page_count : ℕ) (dollar_to_cents : ℕ) : cost_per_4_pages = 8 → page_count = 4 → dollar_to_cents = 100 → (1500 * (page_count / cost_per_4_pages) = 750) :=
by
  intros
  sorry

end page_copy_cost_l144_144690


namespace solve_inequality_l144_144412

theorem solve_inequality (x : ℝ) :
  (abs ((6 - x) / 4) < 3) ∧ (2 ≤ x) ↔ (2 ≤ x) ∧ (x < 18) := 
by
  sorry

end solve_inequality_l144_144412


namespace calculate_x_minus_y_l144_144732

theorem calculate_x_minus_y (x y z : ℝ) 
    (h1 : x - y + z = 23) 
    (h2 : x - y - z = 7) : 
    x - y = 15 :=
by
  sorry

end calculate_x_minus_y_l144_144732


namespace find_n_l144_144891

theorem find_n (n : ℝ) (h1 : (n ≠ 0)) (h2 : ∃ (n' : ℝ), n = n' ∧ -n' = -9 / n') (h3 : ∀ x : ℝ, x > 0 → -n * x < 0) : n = 3 :=
sorry

end find_n_l144_144891


namespace find_number_l144_144257

theorem find_number (X a b : ℕ) (hX : X = 10 * a + b) 
  (h1 : a * b = 24) (h2 : 10 * b + a = X + 18) : X = 46 :=
by
  sorry

end find_number_l144_144257


namespace seq_2016_2017_l144_144392

-- Define the sequence a_n
def seq (n : ℕ) : ℚ := sorry

-- Given conditions
axiom a1_cond : seq 1 = 1/2
axiom a2_cond : seq 2 = 1/3
axiom seq_rec : ∀ n : ℕ, seq n * seq (n + 2) = 1

-- The main goal
theorem seq_2016_2017 : seq 2016 + seq 2017 = 7/2 := sorry

end seq_2016_2017_l144_144392


namespace probability_53_sundays_in_leap_year_l144_144107

-- Define the conditions
def num_days_in_leap_year : ℕ := 366
def num_weeks_in_leap_year : ℕ := 52
def extra_days_in_leap_year : ℕ := 2
def num_combinations : ℕ := 7
def num_sunday_combinations : ℕ := 2

-- Define the problem statement
theorem probability_53_sundays_in_leap_year (hdays : num_days_in_leap_year = 52 * 7 + extra_days_in_leap_year) :
  (num_sunday_combinations / num_combinations : ℚ) = 2 / 7 :=
by
  sorry

end probability_53_sundays_in_leap_year_l144_144107


namespace part_a_l144_144465

theorem part_a (a b c : Int) (h1 : a + b + c = 0) : 
  ¬(a ^ 1999 + b ^ 1999 + c ^ 1999 = 2) :=
by
  sorry

end part_a_l144_144465


namespace terminal_side_in_third_quadrant_l144_144684

theorem terminal_side_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  (∃ k : ℤ, α = k * π + π / 2 + π) := sorry

end terminal_side_in_third_quadrant_l144_144684


namespace complement_intersection_l144_144716

open Set

variable {U : Set ℝ} (A B : Set ℝ)

def A_def : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B_def : Set ℝ := { x | 2 < x ∧ x < 10 }

theorem complement_intersection :
  (U = univ ∧ A = A_def ∧ B = B_def) →
  (compl (A ∩ B) = {x | x < 3 ∨ x ≥ 7}) :=
by
  sorry

end complement_intersection_l144_144716


namespace Laura_pays_more_l144_144097

theorem Laura_pays_more 
  (slices : ℕ) 
  (cost_plain : ℝ) 
  (cost_mushrooms : ℝ) 
  (laura_mushroom_slices : ℕ) 
  (laura_plain_slices : ℕ) 
  (jessica_plain_slices: ℕ) :
  slices = 12 →
  cost_plain = 12 →
  cost_mushrooms = 3 →
  laura_mushroom_slices = 4 →
  laura_plain_slices = 2 →
  jessica_plain_slices = 6 →
  15 / 12 * (laura_mushroom_slices + laura_plain_slices) - 
  (cost_plain / 12 * jessica_plain_slices) = 1.5 :=
by
  intro slices_eq
  intro cost_plain_eq
  intro cost_mushrooms_eq
  intro laura_mushroom_slices_eq
  intro laura_plain_slices_eq
  intro jessica_plain_slices_eq
  sorry

end Laura_pays_more_l144_144097


namespace midpoint_product_coordinates_l144_144666

theorem midpoint_product_coordinates :
  ∃ (x y : ℝ), (4 : ℝ) = (-2 + x) / 2 ∧ (-3 : ℝ) = (-7 + y) / 2 ∧ x * y = 10 := by
  sorry

end midpoint_product_coordinates_l144_144666


namespace train_length_is_95_l144_144586

noncomputable def train_length (time_seconds : ℝ) (speed_kmh : ℝ) : ℝ := 
  let speed_ms := speed_kmh * 1000 / 3600 
  speed_ms * time_seconds

theorem train_length_is_95 : train_length 1.5980030008814248 214 = 95 := by
  sorry

end train_length_is_95_l144_144586


namespace carnival_activity_order_l144_144015

theorem carnival_activity_order :
  let dodgeball := 3 / 8
  let magic_show := 9 / 24
  let petting_zoo := 1 / 3
  let face_painting := 5 / 12
  let ordered_activities := ["Face Painting", "Dodgeball", "Magic Show", "Petting Zoo"]
  (face_painting > dodgeball) ∧ (dodgeball = magic_show) ∧ (magic_show > petting_zoo) →
  ordered_activities = ["Face Painting", "Dodgeball", "Magic Show", "Petting Zoo"] :=
by {
  sorry
}

end carnival_activity_order_l144_144015


namespace frequency_of_group5_l144_144008

-- Define the total number of students and the frequencies of each group
def total_students : ℕ := 40
def freq_group1 : ℕ := 12
def freq_group2 : ℕ := 10
def freq_group3 : ℕ := 6
def freq_group4 : ℕ := 8

-- Define the frequency of the fifth group in terms of the above frequencies
def freq_group5 : ℕ := total_students - (freq_group1 + freq_group2 + freq_group3 + freq_group4)

-- The theorem to be proven
theorem frequency_of_group5 : freq_group5 = 4 := by
  -- Proof goes here, skipped with sorry
  sorry

end frequency_of_group5_l144_144008


namespace multiple_6_9_statements_false_l144_144852

theorem multiple_6_9_statements_false
    (a b : ℤ)
    (h₁ : ∃ m : ℤ, a = 6 * m)
    (h₂ : ∃ n : ℤ, b = 9 * n) :
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → ((a + b) % 2 = 0)) ∧
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → (a + b) % 6 = 0) ∧
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → (a + b) % 9 = 0) ∧
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → (a + b) % 9 ≠ 0) :=
by
  sorry

end multiple_6_9_statements_false_l144_144852


namespace no_third_quadrant_l144_144491

def quadratic_no_real_roots (b : ℝ) : Prop :=
  16 - 4 * b < 0

def passes_through_third_quadrant (b : ℝ) : Prop :=
  ∃ x y : ℝ, y = -2 * x + b ∧ x < 0 ∧ y < 0

theorem no_third_quadrant (b : ℝ) (h : quadratic_no_real_roots b) : ¬ passes_through_third_quadrant b := 
by {
  sorry
}

end no_third_quadrant_l144_144491


namespace cone_surface_area_l144_144630

-- Define the surface area formula for a cone with radius r and slant height l
theorem cone_surface_area (r l : ℝ) : 
  let S := π * r^2 + π * r * l
  S = π * r^2 + π * r * l :=
by sorry

end cone_surface_area_l144_144630


namespace polynomial_roots_sum_l144_144170

theorem polynomial_roots_sum (a b c d e : ℝ) (h₀ : a ≠ 0)
  (h1 : a * 5^4 + b * 5^3 + c * 5^2 + d * 5 + e = 0)
  (h2 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
  (h3 : a * 2^4 + b * 2^3 + c * 2^2 + d * 2 + e = 0) :
  (b + d) / a = -2677 := 
sorry

end polynomial_roots_sum_l144_144170


namespace area_of_triangle_ABC_l144_144482

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 2 }
def B : Point := { x := 6, y := 0 }
def C : Point := { x := 4, y := 7 }

def triangle_area (P1 P2 P3 : Point) : ℝ :=
  0.5 * abs (P1.x * (P2.y - P3.y) +
             P2.x * (P3.y - P1.y) +
             P3.x * (P1.y - P2.y))

theorem area_of_triangle_ABC : triangle_area A B C = 19 :=
by
  sorry

end area_of_triangle_ABC_l144_144482


namespace parabola_sequence_l144_144407

theorem parabola_sequence (m: ℝ) (n: ℕ):
  (∀ t s: ℝ, t * s = -1/4) →
  (∀ x y: ℝ, y^2 = (1/(3^n)) * m * (x - (m / 4) * (1 - (1/(3^n))))) :=
sorry

end parabola_sequence_l144_144407


namespace factorize_x2y_minus_4y_l144_144242

variable {x y : ℝ}

theorem factorize_x2y_minus_4y : x^2 * y - 4 * y = y * (x + 2) * (x - 2) :=
sorry

end factorize_x2y_minus_4y_l144_144242


namespace card_paiting_modulus_l144_144756

theorem card_paiting_modulus (cards : Finset ℕ) (H : cards = Finset.range 61 \ {0}) :
  ∃ d : ℕ, ∀ n ∈ cards, ∃! k, (∀ x ∈ cards, (x + n ≡ k [MOD d])) ∧ (d ∣ 30) ∧ (∃! n : ℕ, 1 ≤ n ∧ n ≤ 8) :=
sorry

end card_paiting_modulus_l144_144756


namespace both_buyers_correct_l144_144100

-- Define the total number of buyers
def total_buyers : ℕ := 100

-- Define the number of buyers who purchase cake mix
def cake_mix_buyers : ℕ := 50

-- Define the number of buyers who purchase muffin mix
def muffin_mix_buyers : ℕ := 40

-- Define the number of buyers who purchase neither cake mix nor muffin mix
def neither_buyers : ℕ := 29

-- Define the number of buyers who purchase both cake and muffin mix
def both_buyers : ℕ := 19

-- The assertion to be proved
theorem both_buyers_correct :
  neither_buyers = total_buyers - (cake_mix_buyers + muffin_mix_buyers - both_buyers) :=
sorry

end both_buyers_correct_l144_144100


namespace exists_separating_line_l144_144457

noncomputable def f1 (x : ℝ) (a1 b1 c1 : ℝ) : ℝ := a1 * x^2 + b1 * x + c1
noncomputable def f2 (x : ℝ) (a2 b2 c2 : ℝ) : ℝ := a2 * x^2 + b2 * x + c2

theorem exists_separating_line (a1 b1 c1 a2 b2 c2 : ℝ) (h_intersect : ∀ x, f1 x a1 b1 c1 ≠ f2 x a2 b2 c2)
  (h_neg : a1 * a2 < 0) : ∃ α β : ℝ, ∀ x, f1 x a1 b1 c1 < α * x + β ∧ α * x + β < f2 x a2 b2 c2 :=
sorry

end exists_separating_line_l144_144457


namespace toys_produced_each_day_l144_144282

theorem toys_produced_each_day (weekly_production : ℕ) (days_worked : ℕ) (h₁ : weekly_production = 4340) (h₂ : days_worked = 2) : weekly_production / days_worked = 2170 :=
by {
  -- Proof can be filled in here
  sorry
}

end toys_produced_each_day_l144_144282


namespace cats_weigh_more_by_5_kg_l144_144424

def puppies_weight (num_puppies : ℕ) (weight_per_puppy : ℝ) : ℝ :=
  num_puppies * weight_per_puppy

def cats_weight (num_cats : ℕ) (weight_per_cat : ℝ) : ℝ :=
  num_cats * weight_per_cat

theorem cats_weigh_more_by_5_kg :
  puppies_weight 4 7.5  = 30 ∧ cats_weight 14 2.5 = 35 → (cats_weight 14 2.5 - puppies_weight 4 7.5 = 5) := 
by
  intro h
  sorry

end cats_weigh_more_by_5_kg_l144_144424


namespace sodium_chloride_moles_produced_l144_144829

theorem sodium_chloride_moles_produced (NaOH HCl NaCl : ℕ) : 
    (NaOH = 3) → (HCl = 3) → NaCl = 3 :=
by
  intro hNaOH hHCl
  -- Placeholder for actual proof
  sorry

end sodium_chloride_moles_produced_l144_144829


namespace eliminate_denominators_l144_144598

variable {x : ℝ}

theorem eliminate_denominators (h : 3 / (2 * x) = 1 / (x - 1)) :
  3 * x - 3 = 2 * x := 
by
  sorry

end eliminate_denominators_l144_144598


namespace sum_of_g_49_l144_144589

def f (x : ℝ) := 4 * x^2 - 3
def g (y : ℝ) := y^2 + 2 * y + 2

theorem sum_of_g_49 : (g 49) = 30 :=
  sorry

end sum_of_g_49_l144_144589


namespace lattice_points_on_hyperbola_l144_144982

theorem lattice_points_on_hyperbola : 
  ∃ n, (∀ x y : ℤ, x^2 - y^2 = 1800^2 → (x, y) ∈ {p : ℤ × ℤ | 
  ∃ a b : ℤ, x = 2 * a + b ∧ y = 2 * a - b}) ∧ n = 250 := 
by {
  sorry
}

end lattice_points_on_hyperbola_l144_144982


namespace cyclic_sum_inequality_l144_144319

open BigOperators

theorem cyclic_sum_inequality {n : ℕ} (h : 0 < n) (a : ℕ → ℝ)
  (hpos : ∀ i, 0 < a i) :
  (∑ k in Finset.range n, a k / (a (k+1) + a (k+2))) > n / 4 := by
  sorry

end cyclic_sum_inequality_l144_144319


namespace bill_apples_left_l144_144600

-- Definitions based on the conditions
def total_apples : Nat := 50
def apples_per_child : Nat := 3
def number_of_children : Nat := 2
def apples_per_pie : Nat := 10
def number_of_pies : Nat := 2

-- The main statement to prove
theorem bill_apples_left : total_apples - ((apples_per_child * number_of_children) + (apples_per_pie * number_of_pies)) = 24 := by
sorry

end bill_apples_left_l144_144600


namespace library_shelves_l144_144490

theorem library_shelves (S : ℕ) (h_books : 4305 + 11 = 4316) :
  4316 % S = 0 ↔ S = 11 :=
by 
  have h_total_books := h_books
  sorry

end library_shelves_l144_144490


namespace paper_sufficient_to_cover_cube_l144_144764

noncomputable def edge_length_cube : ℝ := 1
noncomputable def side_length_sheet : ℝ := 2.5

noncomputable def surface_area_cube : ℝ := 6
noncomputable def area_sheet : ℝ := 6.25

theorem paper_sufficient_to_cover_cube : area_sheet ≥ surface_area_cube :=
  by
    sorry

end paper_sufficient_to_cover_cube_l144_144764


namespace max_value_a4_b4_c4_d4_l144_144796

theorem max_value_a4_b4_c4_d4 (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 16) :
  a^4 + b^4 + c^4 + d^4 ≤ 64 :=
sorry

end max_value_a4_b4_c4_d4_l144_144796


namespace jamies_father_days_to_lose_weight_l144_144877

def calories_per_pound : ℕ := 3500
def pounds_to_lose : ℕ := 5
def calories_burned_per_day : ℕ := 2500
def calories_consumed_per_day : ℕ := 2000
def net_calories_burned_per_day : ℕ := calories_burned_per_day - calories_consumed_per_day
def total_calories_to_burn : ℕ := pounds_to_lose * calories_per_pound
def days_to_burn_calories := total_calories_to_burn / net_calories_burned_per_day

theorem jamies_father_days_to_lose_weight : days_to_burn_calories = 35 := by
  sorry

end jamies_father_days_to_lose_weight_l144_144877


namespace compare_a_b_l144_144224

theorem compare_a_b (a b : ℝ) (h1 : a = 2 * Real.sqrt 7) (h2 : b = 3 * Real.sqrt 5) : a < b :=
by {
  sorry -- We'll leave the proof as a placeholder.
}

end compare_a_b_l144_144224


namespace cauliflower_difference_is_401_l144_144733

-- Definitions using conditions from part a)
def garden_area_this_year : ℕ := 40401
def side_length_this_year : ℕ := Nat.sqrt garden_area_this_year
def side_length_last_year : ℕ := side_length_this_year - 1
def garden_area_last_year : ℕ := side_length_last_year ^ 2
def cauliflowers_difference : ℕ := garden_area_this_year - garden_area_last_year

-- Problem statement claiming that the difference in cauliflowers produced is 401
theorem cauliflower_difference_is_401 :
  garden_area_this_year = 40401 →
  side_length_this_year = 201 →
  side_length_last_year = 200 →
  garden_area_last_year = 40000 →
  cauliflowers_difference = 401 :=
by
  intros
  sorry

end cauliflower_difference_is_401_l144_144733


namespace squares_sum_l144_144337

theorem squares_sum (a b c : ℝ) 
  (h1 : 36 - 4 * Real.sqrt 2 - 6 * Real.sqrt 3 + 12 * Real.sqrt 6 = (a * Real.sqrt 2 + b * Real.sqrt 3 + c) ^ 2) : 
  a^2 + b^2 + c^2 = 14 := 
by
  sorry

end squares_sum_l144_144337


namespace neil_baked_cookies_l144_144372

theorem neil_baked_cookies (total_cookies : ℕ) (given_to_friend : ℕ) (cookies_left : ℕ)
    (h1 : given_to_friend = (2 / 5) * total_cookies)
    (h2 : cookies_left = (3 / 5) * total_cookies)
    (h3 : cookies_left = 12) : total_cookies = 20 :=
by
  sorry

end neil_baked_cookies_l144_144372


namespace jill_basket_total_weight_l144_144304

def jill_basket_capacity : ℕ := 24
def type_a_weight : ℕ := 150
def type_b_weight : ℕ := 170
def jill_basket_type_a_count : ℕ := 12
def jill_basket_type_b_count : ℕ := 12

theorem jill_basket_total_weight :
  (jill_basket_type_a_count * type_a_weight + jill_basket_type_b_count * type_b_weight) = 3840 :=
by
  -- We provide the calculations for clarification; not essential to the theorem statement
  -- (12 * 150) + (12 * 170) = 1800 + 2040 = 3840
  -- Started proof to provide context; actual proof steps are omitted
  sorry

end jill_basket_total_weight_l144_144304


namespace exists_q_lt_1_l144_144470

variable {a : ℕ → ℝ}

theorem exists_q_lt_1 (h_nonneg : ∀ n, 0 ≤ a n)
  (h_rec : ∀ k m, a (k + m) ≤ a (k + m + 1) + a k * a m)
  (h_large_n : ∃ n₀, ∀ n ≥ n₀, n * a n < 0.2499) :
  ∃ q, 0 < q ∧ q < 1 ∧ (∃ n₀, ∀ n ≥ n₀, a n < q ^ n) :=
by
  sorry

end exists_q_lt_1_l144_144470


namespace problem_part1_problem_part2_l144_144768

-- Define what we need to prove
theorem problem_part1 (x : ℝ) (a b : ℤ) 
  (h : (2*x - 21)*(3*x - 7) - (3*x - 7)*(x - 13) = (3*x + a)*(x + b)): 
  a + 3*b = -31 := 
by {
  -- We know from the problem that h holds,
  -- thus the values of a and b must satisfy the condition.
  sorry
}

theorem problem_part2 (x : ℝ) : 
  (x^2 - 3*x + 2) = (x - 1)*(x - 2) := 
by {
  sorry
}

end problem_part1_problem_part2_l144_144768


namespace radius_of_third_circle_l144_144861

theorem radius_of_third_circle (r₁ r₂ : ℝ) (r₁_val : r₁ = 23) (r₂_val : r₂ = 37) : 
  ∃ r : ℝ, r = 2 * Real.sqrt 210 :=
by
  sorry

end radius_of_third_circle_l144_144861


namespace min_value_expression_l144_144294

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  (∃ c : ℝ, c = (1 / (2 * x) + x / (y + 1)) ∧ c = 5 / 4) :=
sorry

end min_value_expression_l144_144294


namespace price_reduction_relationship_l144_144557

variable (a : ℝ) -- original price a in yuan
variable (b : ℝ) -- final price b in yuan

-- condition: price decreased by 10% first
def priceAfterFirstReduction := a * (1 - 0.10)

-- condition: price decreased by 20% on the result of the first reduction
def finalPrice := priceAfterFirstReduction a * (1 - 0.20)

-- theorem: relationship between original price a and final price b
theorem price_reduction_relationship (h : b = finalPrice a) : 
  b = a * (1 - 0.10) * (1 - 0.20) :=
by
  -- proof would go here
  sorry

end price_reduction_relationship_l144_144557


namespace solve_textbook_by_12th_l144_144358

/-!
# Problem Statement
There are 91 problems in a textbook. Yura started solving them on September 6 and solves one problem less each subsequent morning. By the evening of September 8, there are 46 problems left to solve.

We need to prove that Yura finishes solving all the problems by September 12.
-/

def initial_problems : ℕ := 91
def problems_left_by_evening_of_8th : ℕ := 46

def problems_solved_by_evening_of_8th : ℕ :=
  initial_problems - problems_left_by_evening_of_8th

def z : ℕ := (problems_solved_by_evening_of_8th / 3)

theorem solve_textbook_by_12th 
    (total_problems : ℕ)
    (problems_left : ℕ)
    (solved_by_evening_8th : ℕ)
    (daily_problem_count : ℕ) :
    (total_problems = initial_problems) →
    (problems_left = problems_left_by_evening_of_8th) →
    (solved_by_evening_8th = problems_solved_by_evening_of_8th) →
    (daily_problem_count = z) →
    ∃ (finishing_date : ℕ), finishing_date = 12 :=
  by
    intros _ _ _ _
    sorry

end solve_textbook_by_12th_l144_144358


namespace total_players_l144_144102

def kabaddi (K : ℕ) (Kho_only : ℕ) (Both : ℕ) : ℕ :=
  K - Both + Kho_only + Both

theorem total_players (K : ℕ) (Kho_only : ℕ) (Both : ℕ)
  (hK : K = 10)
  (hKho_only : Kho_only = 35)
  (hBoth : Both = 5) :
  kabaddi K Kho_only Both = 45 :=
by
  rw [hK, hKho_only, hBoth]
  unfold kabaddi
  norm_num

end total_players_l144_144102


namespace john_total_spent_l144_144350

noncomputable def total_spent (computer_cost : ℝ) (peripheral_ratio : ℝ) (base_video_cost : ℝ) : ℝ :=
  let peripheral_cost := computer_cost * peripheral_ratio
  let upgraded_video_cost := base_video_cost * 2
  computer_cost + peripheral_cost + (upgraded_video_cost - base_video_cost)

theorem john_total_spent :
  total_spent 1500 0.2 300 = 2100 :=
by
  sorry

end john_total_spent_l144_144350


namespace Jane_exercises_days_per_week_l144_144150

theorem Jane_exercises_days_per_week 
  (goal_hours_per_day : ℕ)
  (weeks : ℕ)
  (total_hours : ℕ)
  (exercise_days_per_week : ℕ) 
  (h_goal : goal_hours_per_day = 1)
  (h_weeks : weeks = 8)
  (h_total_hours : total_hours = 40)
  (h_exercise_hours_weekly : total_hours / weeks = exercise_days_per_week) :
  exercise_days_per_week = 5 :=
by
  sorry

end Jane_exercises_days_per_week_l144_144150


namespace mr_williams_land_percentage_l144_144876

-- Given conditions
def farm_tax_percent : ℝ := 60
def total_tax_collected : ℝ := 5000
def mr_williams_tax_paid : ℝ := 480

-- Theorem statement
theorem mr_williams_land_percentage :
  (mr_williams_tax_paid / total_tax_collected) * 100 = 9.6 := by
  sorry

end mr_williams_land_percentage_l144_144876


namespace find_two_numbers_l144_144978

theorem find_two_numbers :
  ∃ (x y : ℝ), 
  (2 * (x + y) = x^2 - y^2 ∧ 2 * (x + y) = (x * y) / 4 - 56) ∧ 
  ((x = 26 ∧ y = 24) ∨ (x = -8 ∧ y = -10)) := 
sorry

end find_two_numbers_l144_144978


namespace largest_n_unique_k_l144_144177

theorem largest_n_unique_k : ∃ n : ℕ, n = 24 ∧ (∃! k : ℕ, 
  3 / 7 < n / (n + k: ℤ) ∧ n / (n + k: ℤ) < 8 / 19) :=
by
  sorry

end largest_n_unique_k_l144_144177


namespace find_triples_l144_144588

theorem find_triples 
  (x y z : ℝ)
  (h1 : x + y * z = 2)
  (h2 : y + z * x = 2)
  (h3 : z + x * y = 2)
 : (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -2 ∧ y = -2 ∧ z = -2) :=
sorry

end find_triples_l144_144588


namespace Apollonian_Circle_Range_l144_144226

def range_of_m := Set.Icc (Real.sqrt 5 / 2) (Real.sqrt 21 / 2)

theorem Apollonian_Circle_Range :
  ∃ P : ℝ × ℝ, ∃ m > 0, ((P.1 - 2) ^ 2 + (P.2 - m) ^ 2 = 1 / 4) ∧ 
            (Real.sqrt ((P.1 + 1) ^ 2 + P.2 ^ 2) = 2 * Real.sqrt ((P.1 - 2) ^ 2 + P.2 ^ 2)) →
            m ∈ range_of_m :=
  sorry

end Apollonian_Circle_Range_l144_144226


namespace integer_solutions_are_zero_l144_144886

-- Definitions for integers and the given equation
def satisfies_equation (a b : ℤ) : Prop :=
  a^2 * b^2 = a^2 + b^2

-- The main statement to prove
theorem integer_solutions_are_zero :
  ∀ (a b : ℤ), satisfies_equation a b → (a = 0 ∧ b = 0) :=
sorry

end integer_solutions_are_zero_l144_144886


namespace point_M_quadrant_l144_144039

theorem point_M_quadrant (θ : ℝ) (h1 : π / 2 < θ) (h2 : θ < π) :
  (0 < Real.sin θ) ∧ (Real.cos θ < 0) :=
by
  sorry

end point_M_quadrant_l144_144039


namespace solution_set_of_inequality_l144_144708

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / (3 - x) < 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 3} :=
by
  sorry

end solution_set_of_inequality_l144_144708


namespace move_line_up_l144_144824

theorem move_line_up (x : ℝ) :
  let y_initial := 4 * x - 1
  let y_moved := y_initial + 2
  y_moved = 4 * x + 1 :=
by
  let y_initial := 4 * x - 1
  let y_moved := y_initial + 2
  show y_moved = 4 * x + 1
  sorry

end move_line_up_l144_144824


namespace range_of_k_for_positivity_l144_144474

theorem range_of_k_for_positivity (k x : ℝ) (h1 : -1 ≤ x) (h2 : x ≤ 2) :
  ((k - 2) * x + 2 * |k| - 1 > 0) → (k > 5 / 4) :=
sorry

end range_of_k_for_positivity_l144_144474


namespace quadrilateral_segments_l144_144925

theorem quadrilateral_segments {a b c d : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a + b + c + d = 2) (h6 : 1/4 < a) (h7 : a < 1) (h8 : 1/4 < b) (h9 : b < 1)
  (h10 : 1/4 < c) (h11 : c < 1) (h12 : 1/4 < d) (h13 : d < 1) : 
  (a + b > d) ∧ (a + c > d) ∧ (a + d > c) ∧ (b + c > d) ∧ 
  (b + d > c) ∧ (c + d > a) ∧ (a + b + c > d) ∧ (a + b + d > c) ∧
  (a + c + d > b) ∧ (b + c + d > a) :=
sorry

end quadrilateral_segments_l144_144925


namespace cole_round_trip_time_l144_144618

-- Define the relevant quantities
def speed_to_work : ℝ := 70 -- km/h
def speed_to_home : ℝ := 105 -- km/h
def time_to_work_mins : ℝ := 72 -- minutes

-- Define the theorem to be proved
theorem cole_round_trip_time : 
  (time_to_work_mins / 60 + (speed_to_work * time_to_work_mins / 60) / speed_to_home) = 2 :=
by
  sorry

end cole_round_trip_time_l144_144618


namespace number_of_chickens_free_ranging_l144_144391

-- Defining the conditions
def chickens_in_coop : ℕ := 14
def chickens_in_run (coop_chickens : ℕ) : ℕ := 2 * coop_chickens
def chickens_free_ranging (run_chickens : ℕ) : ℕ := 2 * run_chickens - 4

-- Proving the number of chickens free ranging
theorem number_of_chickens_free_ranging : chickens_free_ranging (chickens_in_run chickens_in_coop) = 52 := by
  -- Lean will be able to infer
  sorry  -- proof is not required

end number_of_chickens_free_ranging_l144_144391


namespace exponents_divisible_by_8_l144_144147

theorem exponents_divisible_by_8 (n : ℕ) : 8 ∣ (3^(4 * n + 1) + 5^(2 * n + 1)) :=
by
-- Base case and inductive step will be defined here.
sorry

end exponents_divisible_by_8_l144_144147


namespace handshakes_correct_l144_144520

-- Definitions based on conditions
def num_gremlins : ℕ := 25
def num_imps : ℕ := 20
def num_imps_shaking_hands_among_themselves : ℕ := num_imps / 2
def comb (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- Function to calculate the total handshakes
def total_handshakes : ℕ :=
  (comb num_gremlins 2) + -- Handshakes among gremlins
  (comb num_imps_shaking_hands_among_themselves 2) + -- Handshakes among half the imps
  (num_gremlins * num_imps) -- Handshakes between all gremlins and all imps

-- The theorem to be proved
theorem handshakes_correct : total_handshakes = 845 := by
  sorry

end handshakes_correct_l144_144520


namespace price_of_n_kilograms_l144_144449

theorem price_of_n_kilograms (m n : ℕ) (hm : m ≠ 0) (h : 9 = m) : (9 * n) / m = (9 * n) / m :=
by
  sorry

end price_of_n_kilograms_l144_144449


namespace find_a5_geometric_sequence_l144_144483

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ r > 0, ∀ n ≥ 1, a (n + 1) = r * a n

theorem find_a5_geometric_sequence :
  ∀ (a : ℕ → ℝ),
  geometric_sequence a ∧ 
  (∀ n, a n > 0) ∧ 
  (a 3 * a 11 = 16) 
  → a 5 = 1 :=
by
  sorry

end find_a5_geometric_sequence_l144_144483


namespace printing_machine_completion_time_l144_144729

-- Definitions of times in hours
def start_time : ℕ := 9 -- 9:00 AM
def half_job_time : ℕ := 12 -- 12:00 PM
def completion_time : ℕ := 15 -- 3:00 PM

-- Time taken to complete half the job
def half_job_duration : ℕ := half_job_time - start_time

-- Total time to complete the entire job
def total_job_duration : ℕ := 2 * half_job_duration

-- Proof that the machine will complete the job at 3:00 PM
theorem printing_machine_completion_time : 
    start_time + total_job_duration = completion_time :=
sorry

end printing_machine_completion_time_l144_144729


namespace smallest_n_with_square_ending_in_2016_l144_144834

theorem smallest_n_with_square_ending_in_2016 : 
  ∃ n : ℕ, (n^2 % 10000 = 2016) ∧ (n = 996) :=
by
  sorry

end smallest_n_with_square_ending_in_2016_l144_144834


namespace find_power_l144_144160

theorem find_power (x y : ℕ) (h1 : 2^x - 2^y = 3 * 2^11) (h2 : x = 13) : y = 11 :=
sorry

end find_power_l144_144160


namespace distance_AB_l144_144381

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l144_144381


namespace reciprocal_inequality_reciprocal_inequality_opposite_l144_144453

theorem reciprocal_inequality (a b : ℝ) (h1 : a > b) (h2 : ab > 0) : (1 / a < 1 / b) := 
sorry

theorem reciprocal_inequality_opposite (a b : ℝ) (h1 : a > b) (h2 : ab < 0) : (1 / a > 1 / b) := 
sorry

end reciprocal_inequality_reciprocal_inequality_opposite_l144_144453


namespace range_of_a_l144_144334

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
if x < 1 then -x + 2 else a / x

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x < 1 ∧ (0 < -x + 2)) ∧ (∀ x : ℝ, x ≥ 1 → (0 < a / x)) → a ≥ 1 :=
by
  sorry

end range_of_a_l144_144334


namespace ratio_proof_l144_144178

variables {F : Type*} [Field F] 
variables (w x y z : F)

theorem ratio_proof 
  (h1 : w / x = 4 / 3) 
  (h2 : y / z = 3 / 2) 
  (h3 : z / x = 1 / 6) : 
  w / y = 16 / 3 :=
by sorry

end ratio_proof_l144_144178


namespace solve_equation_l144_144213

theorem solve_equation (x : ℝ) (h : x ≠ -2) : (x = -1/2) ↔ (x / (x + 2) + 1 = 1 / (x + 2)) :=
by
  sorry

end solve_equation_l144_144213


namespace min_rounds_for_expected_value_l144_144629

theorem min_rounds_for_expected_value 
  (p1 p2 : ℝ) (h0 : 0 ≤ p1 ∧ p1 ≤ 1) (h1 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (h2 : p1 + p2 = 3 / 2)
  (indep : true) -- Assuming independence implicitly
  (X : ℕ → ℕ) (n : ℕ)
  (E_X_eq_24 : (n : ℕ) * (3 * p1 * p2 * (1 - p1 * p2)) = 24) :
  n = 32 := 
sorry

end min_rounds_for_expected_value_l144_144629


namespace cutting_wire_random_event_l144_144620

noncomputable def length : ℝ := sorry

def is_random_event (a : ℝ) : Prop := sorry

theorem cutting_wire_random_event (a : ℝ) (h : a > 0) :
  is_random_event a := 
by
  sorry

end cutting_wire_random_event_l144_144620


namespace simplify_expression_l144_144655

theorem simplify_expression:
  (a = 2) ∧ (b = 1) →
  - (1 / 3 : ℚ) * (a^3 * b - a * b) 
  + a * b^3 
  - (a * b - b) / 2 
  - b / 2 
  + (1 / 3 : ℚ) * (a^3 * b) 
  = (5 / 3 : ℚ) := by 
  intros h
  simp [h.1, h.2]
  sorry

end simplify_expression_l144_144655


namespace intersection_of_P_with_complement_Q_l144_144707

-- Define the universal set U, and sets P and Q
def U : List ℕ := [1, 2, 3, 4]
def P : List ℕ := [1, 2]
def Q : List ℕ := [2, 3]

-- Define the complement of Q with respect to U
def complement (U Q : List ℕ) : List ℕ := U.filter (λ x => x ∉ Q)

-- Define the intersection of two sets
def intersection (A B : List ℕ) : List ℕ := A.filter (λ x => x ∈ B)

-- The proof statement we need to show
theorem intersection_of_P_with_complement_Q : intersection P (complement U Q) = [1] := by
  sorry

end intersection_of_P_with_complement_Q_l144_144707


namespace volume_of_smaller_cube_l144_144268

noncomputable def volume_of_larger_cube : ℝ := 343
noncomputable def number_of_smaller_cubes : ℝ := 343
noncomputable def surface_area_difference : ℝ := 1764

theorem volume_of_smaller_cube (v_lc : ℝ) (n_sc : ℝ) (sa_diff : ℝ) :
  v_lc = volume_of_larger_cube →
  n_sc = number_of_smaller_cubes →
  sa_diff = surface_area_difference →
  ∃ (v_sc : ℝ), v_sc = 1 :=
by sorry

end volume_of_smaller_cube_l144_144268


namespace sum_of_squares_first_28_l144_144766

theorem sum_of_squares_first_28 : 
  (28 * (28 + 1) * (2 * 28 + 1)) / 6 = 7722 := by
  sorry

end sum_of_squares_first_28_l144_144766


namespace find_fourth_number_in_proportion_l144_144425

-- Define the given conditions
def x : ℝ := 0.39999999999999997
def proportion (y : ℝ) := 0.60 / x = 6 / y

-- State the theorem to be proven
theorem find_fourth_number_in_proportion :
  proportion y → y = 4 :=
by
  intro h
  sorry

end find_fourth_number_in_proportion_l144_144425


namespace geometric_progressions_sum_eq_l144_144823

variable {a q b : ℝ}
variable {n : ℕ}
variable (h1 : q ≠ 1)

/-- The given statement in Lean 4 -/
theorem geometric_progressions_sum_eq (h : a * (q^(3*n) - 1) / (q - 1) = b * (q^(3*n) - 1) / (q^3 - 1)) : 
  b = a * (1 + q + q^2) := 
by
  sorry

end geometric_progressions_sum_eq_l144_144823


namespace tiger_distance_proof_l144_144363

-- Declare the problem conditions
def tiger_initial_speed : ℝ := 25
def tiger_initial_time : ℝ := 3
def tiger_slow_speed : ℝ := 10
def tiger_slow_time : ℝ := 4
def tiger_chase_speed : ℝ := 50
def tiger_chase_time : ℝ := 0.5

-- Compute individual distances
def distance1 := tiger_initial_speed * tiger_initial_time
def distance2 := tiger_slow_speed * tiger_slow_time
def distance3 := tiger_chase_speed * tiger_chase_time

-- Compute the total distance
def total_distance := distance1 + distance2 + distance3

-- The final theorem to prove
theorem tiger_distance_proof : total_distance = 140 := by
  sorry

end tiger_distance_proof_l144_144363


namespace evaluate_expression_l144_144702

theorem evaluate_expression :
  let x := (1/4 : ℚ)
  let y := (4/5 : ℚ)
  let z := (-2 : ℚ)
  x^3 * y^2 * z^2 = 1/25 :=
by
  sorry

end evaluate_expression_l144_144702


namespace min_b_for_factorization_l144_144910

theorem min_b_for_factorization : 
  ∃ b : ℕ, (∀ p q : ℤ, (p + q = b) ∧ (p * q = 1764) → x^2 + b * x + 1764 = (x + p) * (x + q)) 
  ∧ b = 84 :=
sorry

end min_b_for_factorization_l144_144910


namespace max_volume_prism_l144_144121

theorem max_volume_prism (V : ℝ) (h l w : ℝ) 
  (h_eq_2h : l = 2 * h ∧ w = 2 * h) 
  (surface_area_eq : l * h + w * h + l * w = 36) : 
  V = 27 * Real.sqrt 2 := 
  sorry

end max_volume_prism_l144_144121


namespace fg_eq_gf_condition_l144_144429

/-- Definitions of the functions f and g --/
def f (m n c x : ℝ) : ℝ := m * x + n + c
def g (p q c x : ℝ) : ℝ := p * x + q + c

/-- The main theorem stating the equivalence of the condition for f(g(x)) = g(f(x)) --/
theorem fg_eq_gf_condition (m n p q c x : ℝ) :
  f m n c (g p q c x) = g p q c (f m n c x) ↔ n * (1 - p) - q * (1 - m) + c * (m - p) = 0 := by
  sorry

end fg_eq_gf_condition_l144_144429


namespace Parkway_Elementary_girls_not_playing_soccer_l144_144779

/-
  In the fifth grade at Parkway Elementary School, there are 500 students. 
  350 students are boys and 250 students are playing soccer.
  86% of the students that play soccer are boys.
  Prove that the number of girl students that are not playing soccer is 115.
-/
theorem Parkway_Elementary_girls_not_playing_soccer
  (total_students : ℕ)
  (boys : ℕ)
  (playing_soccer : ℕ)
  (percentage_boys_playing_soccer : ℝ)
  (H1 : total_students = 500)
  (H2 : boys = 350)
  (H3 : playing_soccer = 250)
  (H4 : percentage_boys_playing_soccer = 0.86) :
  ∃ (girls_not_playing_soccer : ℕ), girls_not_playing_soccer = 115 :=
by
  sorry

end Parkway_Elementary_girls_not_playing_soccer_l144_144779


namespace evaluate_expression_l144_144488

-- Define the condition b = 2
def b : ℕ := 2

-- Theorem statement
theorem evaluate_expression : (b^3 * b^4 = 128) := 
by
  sorry

end evaluate_expression_l144_144488


namespace problem_I4_1_l144_144210

theorem problem_I4_1 (a : ℝ) : ((∃ y : ℝ, x + 2 * y + 3 = 0) ∧ (∃ y : ℝ, 4 * x - a * y + 5 = 0) ∧ 
  (∃ m1 m2 : ℝ, m1 = -(1 / 2) ∧ m2 = 4 / a ∧ m1 * m2 = -1)) → a = 2 :=
sorry

end problem_I4_1_l144_144210


namespace part1_part2_l144_144657

-- Part 1: Prove that the range of values for k is k ≤ 1/4
theorem part1 (f : ℝ → ℝ) (k : ℝ) 
  (h1 : ∀ x0 : ℝ, f x0 ≥ |k+3| - |k-2|)
  (h2 : ∀ x : ℝ, f x = |2*x - 1| + |x - 2| ) : 
  k ≤ 1/4 := 
sorry

-- Part 2: Show that the minimum value of m+n is 8/3
theorem part2 (f : ℝ → ℝ) (m n : ℝ) 
  (h1 : ∀ x : ℝ, f x ≥ 1/m + 1/n)
  (h2 : ∀ x : ℝ, f x = |2*x - 1| + |x - 2| ) : 
  m + n ≥ 8/3 := 
sorry

end part1_part2_l144_144657


namespace part1_part2_l144_144185

noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.sin x + Real.cos x)

theorem part1 : f (Real.pi / 4) = 2 := sorry

theorem part2 : ∀ k : ℤ, ∀ x : ℝ, k * Real.pi - Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 3 * Real.pi / 8 → 
  (2 * Real.sqrt 2 * Real.cos (2 * x - Real.pi / 4) > 0) := sorry

end part1_part2_l144_144185


namespace isosceles_triangle_sides_l144_144172

theorem isosceles_triangle_sides (a b c : ℝ) (h_iso : a = b ∨ b = c ∨ c = a) (h_perimeter : a + b + c = 14) (h_side : a = 4 ∨ b = 4 ∨ c = 4) : 
  (a = 4 ∧ b = 5 ∧ c = 5) ∨ (a = 5 ∧ b = 5 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 6) ∨ (a = 4 ∧ b = 6 ∧ c = 4) :=
  sorry

end isosceles_triangle_sides_l144_144172


namespace lisa_time_to_complete_l144_144717

theorem lisa_time_to_complete 
  (hotdogs_record : ℕ) 
  (eaten_so_far : ℕ) 
  (rate_per_minute : ℕ) 
  (remaining_hotdogs : ℕ) 
  (time_to_complete : ℕ) 
  (h1 : hotdogs_record = 75) 
  (h2 : eaten_so_far = 20) 
  (h3 : rate_per_minute = 11) 
  (h4 : remaining_hotdogs = hotdogs_record - eaten_so_far)
  (h5 : time_to_complete = remaining_hotdogs / rate_per_minute) :
  time_to_complete = 5 :=
sorry

end lisa_time_to_complete_l144_144717


namespace range_of_f_l144_144940

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^4 + 6 * x^2 + 9

-- Define the domain as [0, ∞)
def domain (x : ℝ) : Prop := x ≥ 0

-- State the theorem which asserts the range of f(x) is [9, ∞)
theorem range_of_f : ∀ y : ℝ, (∃ x : ℝ, domain x ∧ f x = y) ↔ y ≥ 9 := by
  sorry

end range_of_f_l144_144940


namespace chessboard_overlap_area_l144_144082

theorem chessboard_overlap_area :
  let n := 8
  let cell_area := 1
  let side_length := 8
  let overlap_area := 32 * (Real.sqrt 2 - 1)
  (∃ black_overlap_area : ℝ, black_overlap_area = overlap_area) :=
by
  sorry

end chessboard_overlap_area_l144_144082


namespace tan_ratio_l144_144730

open Real

variable (x y : ℝ)

-- Conditions
def sin_add_eq : sin (x + y) = 5 / 8 := sorry
def sin_sub_eq : sin (x - y) = 1 / 4 := sorry

-- Proof problem statement
theorem tan_ratio : sin (x + y) = 5 / 8 → sin (x - y) = 1 / 4 → (tan x) / (tan y) = 2 := sorry

end tan_ratio_l144_144730


namespace final_position_is_negative_one_total_revenue_is_118_yuan_l144_144010

-- Define the distances
def distances : List Int := [9, -3, -6, 4, -8, 6, -3, -6, -4, 10]

-- Define the taxi price per kilometer
def price_per_km : Int := 2

-- Theorem to prove the final position of the taxi relative to Wu Zhong
theorem final_position_is_negative_one : 
  List.sum distances = -1 :=
by 
  sorry -- Proof omitted

-- Theorem to prove the total revenue for the afternoon
theorem total_revenue_is_118_yuan : 
  price_per_km * List.sum (List.map Int.natAbs distances) = 118 :=
by
  sorry -- Proof omitted

end final_position_is_negative_one_total_revenue_is_118_yuan_l144_144010


namespace no_n_repeats_stock_price_l144_144731

-- Problem statement translation
theorem no_n_repeats_stock_price (n : ℕ) (h1 : n < 100) : ¬ ∃ k l : ℕ, (100 + n) ^ k * (100 - n) ^ l = 100 ^ (k + l) :=
by
  sorry

end no_n_repeats_stock_price_l144_144731


namespace horse_problem_l144_144492

theorem horse_problem (x : ℕ) :
  150 * (x + 12) = 240 * x :=
sorry

end horse_problem_l144_144492


namespace find_the_number_l144_144583

theorem find_the_number :
  ∃ x : ℕ, (x + 720) / 125 = 7392 / 462 ∧ x = 1280 := 
  sorry

end find_the_number_l144_144583


namespace median_to_hypotenuse_of_right_triangle_l144_144584

theorem median_to_hypotenuse_of_right_triangle (DE DF : ℝ) (h₁ : DE = 6) (h₂ : DF = 8) :
  let EF := Real.sqrt (DE^2 + DF^2)
  let N := EF / 2
  N = 5 :=
by
  let EF := Real.sqrt (DE^2 + DF^2)
  let N := EF / 2
  have h : N = 5 :=
    by
      sorry
  exact h

end median_to_hypotenuse_of_right_triangle_l144_144584


namespace solve_for_q_l144_144627

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : q = -25 / 11 :=
by
  sorry

end solve_for_q_l144_144627


namespace num_students_l144_144514

theorem num_students (x : ℕ) (h1 : ∃ z : ℕ, z = 10 * x + 6) (h2 : ∃ z : ℕ, z = 12 * x - 6) : x = 6 :=
by
  sorry

end num_students_l144_144514


namespace contrapositive_example_l144_144704

theorem contrapositive_example (x : ℝ) (h : x = 1 → x^2 - 3 * x + 2 = 0) :
  x^2 - 3 * x + 2 ≠ 0 → x ≠ 1 :=
by
  intro h₀
  intro h₁
  have h₂ := h h₁
  contradiction

end contrapositive_example_l144_144704


namespace union_of_A_and_B_l144_144883

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} :=
by
  sorry

end union_of_A_and_B_l144_144883


namespace final_price_after_adjustments_l144_144927

theorem final_price_after_adjustments (p : ℝ) :
  let increased_price := p * 1.30
  let discounted_price := increased_price * 0.75
  let final_price := discounted_price * 1.10
  final_price = 1.0725 * p :=
by
  sorry

end final_price_after_adjustments_l144_144927


namespace geometric_series_sum_eq_l144_144297

theorem geometric_series_sum_eq :
  let a := (5 : ℚ)
  let r := (-1/2 : ℚ)
  (∑' n : ℕ, a * r^n) = (10 / 3 : ℚ) :=
by
  sorry

end geometric_series_sum_eq_l144_144297


namespace find_other_x_intercept_l144_144874

theorem find_other_x_intercept (a b c : ℝ) (h1 : ∀ x, a * x^2 + b * x + c = a * (x - 4)^2 + 9)
  (h2 : a * 0^2 + b * 0 + c = 0) : ∃ x, x ≠ 0 ∧ a * x^2 + b * x + c = 0 ∧ x = 8 :=
by
  sorry

end find_other_x_intercept_l144_144874


namespace asterisk_replacement_l144_144098

theorem asterisk_replacement (x : ℝ) : 
  (x / 20) * (x / 80) = 1 ↔ x = 40 :=
by sorry

end asterisk_replacement_l144_144098


namespace option_C_correct_l144_144800

theorem option_C_correct (m : ℝ) : (-m + 2) * (-m - 2) = m ^ 2 - 4 :=
by sorry

end option_C_correct_l144_144800


namespace solution_set_of_inequality_l144_144955

theorem solution_set_of_inequality :
  {x : ℝ | 1 / x < 1 / 3} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 3} :=
by
  sorry

end solution_set_of_inequality_l144_144955


namespace y_relationship_l144_144484

theorem y_relationship (x1 x2 x3 y1 y2 y3 : ℝ) 
  (hA : y1 = -7 * x1 + 14) 
  (hB : y2 = -7 * x2 + 14) 
  (hC : y3 = -7 * x3 + 14) 
  (hx : x1 > x3 ∧ x3 > x2) : y1 < y3 ∧ y3 < y2 :=
by
  sorry

end y_relationship_l144_144484


namespace kaleb_can_buy_toys_l144_144939

def kaleb_initial_money : ℕ := 12
def money_spent_on_game : ℕ := 8
def money_saved : ℕ := 2
def toy_cost : ℕ := 2

theorem kaleb_can_buy_toys :
  (kaleb_initial_money - money_spent_on_game - money_saved) / toy_cost = 1 :=
by
  sorry

end kaleb_can_buy_toys_l144_144939


namespace sum_final_numbers_l144_144596

theorem sum_final_numbers (x y S : ℝ) (h : x + y = S) : 
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 :=
by
  sorry

end sum_final_numbers_l144_144596


namespace g_minus3_is_correct_l144_144074

theorem g_minus3_is_correct (g : ℚ → ℚ) (h : ∀ x : ℚ, x ≠ 0 → 4 * g (1 / x) + 3 * g x / x = 3 * x^2) : 
  g (-3) = 247 / 39 :=
by
  sorry

end g_minus3_is_correct_l144_144074


namespace how_many_children_l144_144367

-- Definitions based on conditions
def total_spectators : ℕ := 10000
def men : ℕ := 7000
def others : ℕ := total_spectators - men -- women + children
def children_per_woman : ℕ := 5

-- Variables
variable (W C : ℕ)

-- Conditions as Lean equalities
def condition1 : W + C = others := by sorry
def condition2 : C = children_per_woman * W := by sorry

-- Theorem statement to prove the number of children
theorem how_many_children (h1 : W + C = others) (h2 : C = children_per_woman * W) : C = 2500 :=
by sorry

end how_many_children_l144_144367


namespace rational_operation_example_l144_144270

def rational_operation (a b : ℚ) : ℚ := a^3 - 2 * a * b + 4

theorem rational_operation_example : rational_operation 4 (-9) = 140 := 
by
  sorry

end rational_operation_example_l144_144270


namespace primes_solution_l144_144284

theorem primes_solution (p q : ℕ) (m n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hm : m ≥ 2) (hn : n ≥ 2) :
    p^n = q^m + 1 ∨ p^n = q^m - 1 → (p = 2 ∧ n = 3 ∧ q = 3 ∧ m = 2) :=
by
  sorry

end primes_solution_l144_144284


namespace factorization_correct_l144_144641

theorem factorization_correct (x : ℝ) : 
  (hxA : x^2 + 2*x + 1 ≠ x*(x + 2) + 1) → 
  (hxB : x^2 + 2*x + 1 ≠ (x + 1)*(x - 1)) → 
  (hxC : x^2 + x ≠ (x + 1/2)^2 - 1/4) →
  x^2 + x = x * (x + 1) := 
by sorry

end factorization_correct_l144_144641


namespace percentage_increase_in_yield_after_every_harvest_is_20_l144_144136

theorem percentage_increase_in_yield_after_every_harvest_is_20
  (P : ℝ)
  (h1 : ∀ n : ℕ, n = 1 → 20 * n = 20)
  (h2 : 20 + 20 * (1 + P / 100) = 44) :
  P = 20 := 
sorry

end percentage_increase_in_yield_after_every_harvest_is_20_l144_144136


namespace largest_number_of_minerals_per_shelf_l144_144454

theorem largest_number_of_minerals_per_shelf (d : ℕ) :
  d ∣ 924 ∧ d ∣ 1386 ∧ d ∣ 462 ↔ d = 462 :=
by
  sorry

end largest_number_of_minerals_per_shelf_l144_144454


namespace sum_of_possible_M_l144_144118

theorem sum_of_possible_M (M : ℝ) (h : M * (M - 8) = -8) : M = 4 ∨ M = 4 := 
by sorry

end sum_of_possible_M_l144_144118


namespace third_pipe_empty_time_l144_144277

theorem third_pipe_empty_time (x : ℝ) :
  (1 / 60 : ℝ) + (1 / 120) - (1 / x) = (1 / 60) →
  x = 120 :=
by
  intros h
  sorry

end third_pipe_empty_time_l144_144277


namespace necess_suff_cond_odd_function_l144_144090

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) := Real.sin (ω * x + ϕ)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def P (ω ϕ : ℝ) : Prop := f ω ϕ 0 = 0
def Q (ω ϕ : ℝ) : Prop := is_odd (f ω ϕ)

theorem necess_suff_cond_odd_function (ω ϕ : ℝ) : P ω ϕ ↔ Q ω ϕ := by
  sorry

end necess_suff_cond_odd_function_l144_144090


namespace abs_f_at_1_eq_20_l144_144063

noncomputable def fourth_degree_polynomial (f : ℝ → ℝ) : Prop :=
  ∃ p : Polynomial ℝ, p.degree = 4 ∧ ∀ x, f x = p.eval x

theorem abs_f_at_1_eq_20 
  (f : ℝ → ℝ)
  (h_f_poly : fourth_degree_polynomial f)
  (h_f_neg2 : |f (-2)| = 10)
  (h_f_0 : |f 0| = 10)
  (h_f_3 : |f 3| = 10)
  (h_f_7 : |f 7| = 10) :
  |f 1| = 20 := 
sorry

end abs_f_at_1_eq_20_l144_144063


namespace Bert_total_profit_is_14_90_l144_144809

-- Define the sales price for each item
def sales_price_barrel : ℝ := 90
def sales_price_tools : ℝ := 50
def sales_price_fertilizer : ℝ := 30

-- Define the tax rates for each item
def tax_rate_barrel : ℝ := 0.10
def tax_rate_tools : ℝ := 0.05
def tax_rate_fertilizer : ℝ := 0.12

-- Define the profit added per item
def profit_per_item : ℝ := 10

-- Define the tax amount for each item
def tax_barrel : ℝ := tax_rate_barrel * sales_price_barrel
def tax_tools : ℝ := tax_rate_tools * sales_price_tools
def tax_fertilizer : ℝ := tax_rate_fertilizer * sales_price_fertilizer

-- Define the cost price for each item
def cost_price_barrel : ℝ := sales_price_barrel - profit_per_item
def cost_price_tools : ℝ := sales_price_tools - profit_per_item
def cost_price_fertilizer : ℝ := sales_price_fertilizer - profit_per_item

-- Define the profit for each item
def profit_barrel : ℝ := sales_price_barrel - tax_barrel - cost_price_barrel
def profit_tools : ℝ := sales_price_tools - tax_tools - cost_price_tools
def profit_fertilizer : ℝ := sales_price_fertilizer - tax_fertilizer - cost_price_fertilizer

-- Define the total profit
def total_profit : ℝ := profit_barrel + profit_tools + profit_fertilizer

-- Assert the total profit is $14.90
theorem Bert_total_profit_is_14_90 : total_profit = 14.90 :=
by
  -- Omitted proof
  sorry

end Bert_total_profit_is_14_90_l144_144809


namespace correct_value_l144_144253

theorem correct_value (x : ℝ) (h : x + 2.95 = 9.28) : x - 2.95 = 3.38 :=
by
  sorry

end correct_value_l144_144253


namespace find_quantities_l144_144058

variables {a b x y : ℝ}

-- Original total expenditure condition
axiom h1 : a * x + b * y = 1500

-- New prices and quantities for the first scenario
axiom h2 : (a + 1.5) * (x - 10) + (b + 1) * y = 1529

-- New prices and quantities for the second scenario
axiom h3 : (a + 1) * (x - 5) + (b + 1) * y = 1563.5

-- Inequality constraint
axiom h4 : 205 < 2 * x + y ∧ 2 * x + y < 210

-- Range for 'a'
axiom h5 : 17.5 < a ∧ a < 18.5

-- Proving x and y are specific values.
theorem find_quantities :
  x = 76 ∧ y = 55 :=
sorry

end find_quantities_l144_144058


namespace determine_n_l144_144897

theorem determine_n (n : ℕ) (h : n ≥ 2)
    (condition : ∀ i j : ℕ, i ≤ n → j ≤ n → (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) :
    ∃ k : ℕ, k ≥ 2 ∧ n = 2^k - 2 := 
sorry

end determine_n_l144_144897


namespace remainder_when_divided_by_7_l144_144362

theorem remainder_when_divided_by_7 (k : ℕ) (h1 : k % 5 = 2) (h2 : k % 6 = 5) : k % 7 = 3 :=
sorry

end remainder_when_divided_by_7_l144_144362


namespace problem_l144_144025

theorem problem (x y : ℝ) : 
  2 * x + y = 11 → x + 2 * y = 13 → 10 * x^2 - 6 * x * y + y^2 = 530 :=
by
  sorry

end problem_l144_144025


namespace book_surface_area_l144_144739

variables (L : ℕ) (T : ℕ) (A1 : ℕ) (A2 : ℕ) (W : ℕ) (S : ℕ)

theorem book_surface_area (hL : L = 5) (hT : T = 2) 
                         (hA1 : A1 = L * W) (hA1_val : A1 = 50)
                         (hA2 : A2 = T * W) (hA2_val : A2 = 10) :
  S = 2 * A1 + A2 + 2 * (L * T) :=
sorry

end book_surface_area_l144_144739


namespace germination_percentage_l144_144365

theorem germination_percentage (seeds_plot1 seeds_plot2 : ℕ) (percent_germ_plot1 : ℕ) (total_percent_germ : ℕ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  percent_germ_plot1 = 20 →
  total_percent_germ = 26 →
  ∃ (percent_germ_plot2 : ℕ), percent_germ_plot2 = 35 :=
by
  sorry

end germination_percentage_l144_144365


namespace angle_ABC_measure_l144_144569

theorem angle_ABC_measure
  (CBD : ℝ)
  (ABC ABD : ℝ)
  (h1 : CBD = 90)
  (h2 : ABC + ABD + CBD = 270)
  (h3 : ABD = 100) : 
  ABC = 80 :=
by
  -- Given:
  -- CBD = 90
  -- ABC + ABD + CBD = 270
  -- ABD = 100
  sorry

end angle_ABC_measure_l144_144569


namespace number_of_algebra_textbooks_l144_144867

theorem number_of_algebra_textbooks
  (x y n : ℕ)
  (h₁ : x * n + y = 2015)
  (h₂ : y * n + x = 1580) :
  y = 287 := 
sorry

end number_of_algebra_textbooks_l144_144867


namespace total_distance_yards_remaining_yards_l144_144307

structure Distance where
  miles : Nat
  yards : Nat

def marathon_distance : Distance :=
  { miles := 26, yards := 385 }

def miles_to_yards (miles : Nat) : Nat :=
  miles * 1760

def total_yards_in_marathon (d : Distance) : Nat :=
  miles_to_yards d.miles + d.yards

def total_distance_in_yards (d : Distance) (n : Nat) : Nat :=
  n * total_yards_in_marathon d

def remaining_yards (total_yards : Nat) (yards_in_mile : Nat) : Nat :=
  total_yards % yards_in_mile

theorem total_distance_yards_remaining_yards :
    let total_yards := total_distance_in_yards marathon_distance 15
    remaining_yards total_yards 1760 = 495 :=
by
  sorry

end total_distance_yards_remaining_yards_l144_144307


namespace find_son_l144_144202

variable (SonAge ManAge : ℕ)

def age_relationship (SonAge ManAge : ℕ) : Prop :=
  ManAge = SonAge + 20 ∧ ManAge + 2 = 2 * (SonAge + 2)

theorem find_son's_age (S M : ℕ) (h : age_relationship S M) : S = 18 :=
by
  unfold age_relationship at h
  obtain ⟨h1, h2⟩ := h
  sorry

end find_son_l144_144202


namespace basic_astrophysics_degrees_l144_144030

def percentages : List ℚ := [12, 22, 14, 27, 7, 5, 3, 4]

def total_budget_percentage : ℚ := 100

def degrees_in_circle : ℚ := 360

def remaining_percentage (lst : List ℚ) (total : ℚ) : ℚ :=
  total - lst.sum / 100  -- convert sum to percentage

def degrees_of_percentage (percent : ℚ) (circle_degrees : ℚ) : ℚ :=
  percent * (circle_degrees / total_budget_percentage) -- conversion rate per percentage point

theorem basic_astrophysics_degrees :
  degrees_of_percentage (remaining_percentage percentages total_budget_percentage) degrees_in_circle = 21.6 :=
by
  sorry

end basic_astrophysics_degrees_l144_144030


namespace log_inequality_l144_144266

theorem log_inequality
  (a : ℝ := Real.log 4 / Real.log 5)
  (b : ℝ := (Real.log 3 / Real.log 5)^2)
  (c : ℝ := Real.log 5 / Real.log 4) :
  b < a ∧ a < c :=
by
  sorry

end log_inequality_l144_144266


namespace geometric_sum_first_six_terms_l144_144311

theorem geometric_sum_first_six_terms : 
  let a := (1 : ℚ) / 2
  let r := (1 : ℚ) / 4
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 4095 / 6144 :=
by
  -- Definitions and properties of geometric series
  sorry

end geometric_sum_first_six_terms_l144_144311


namespace sum_of_ai_powers_l144_144031

theorem sum_of_ai_powers :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ),
  (∀ x : ℝ, (1 + x) * (1 - 2 * x)^8 = 
            a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + 
            a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + 
            a_7 * x^7 + a_8 * x^8 + a_9 * x^9) →
  a_1 * 2 + a_2 * 2^2 + a_3 * 2^3 + 
  a_4 * 2^4 + a_5 * 2^5 + a_6 * 2^6 + 
  a_7 * 2^7 + a_8 * 2^8 + a_9 * 2^9 = 3^9 - 1 :=
by
  sorry

end sum_of_ai_powers_l144_144031


namespace scientific_notation_correct_l144_144028

def number_in_scientific_notation : ℝ := 1600000
def expected_scientific_notation : ℝ := 1.6 * 10^6

theorem scientific_notation_correct :
  number_in_scientific_notation = expected_scientific_notation := by
  sorry

end scientific_notation_correct_l144_144028


namespace current_in_circuit_l144_144320

open Complex

theorem current_in_circuit
  (V : ℂ := 2 + 3 * I)
  (Z : ℂ := 4 - 2 * I) :
  (V / Z) = (1 / 10 + 4 / 5 * I) :=
  sorry

end current_in_circuit_l144_144320


namespace no_positive_integer_solution_exists_l144_144541

theorem no_positive_integer_solution_exists :
  ¬ ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 3 * x^2 + 2 * x + 2 = y^2 :=
by
  -- The proof steps will go here.
  sorry

end no_positive_integer_solution_exists_l144_144541


namespace solve_abs_inequality_l144_144818

theorem solve_abs_inequality (x : ℝ) :
  |x + 2| + |x - 2| < x + 7 ↔ -7 / 3 < x ∧ x < 7 :=
sorry

end solve_abs_inequality_l144_144818


namespace triangle_angle_and_side_l144_144275

theorem triangle_angle_and_side (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C)
  (h2 : a + b = 6)
  (h3 : 1 / 2 * a * b * Real.sin C = 2 * Real.sqrt 3)
  : C = 2 * Real.pi / 3 ∧ c = 2 * Real.sqrt 7 := by
  -- proof omitted
  sorry

end triangle_angle_and_side_l144_144275


namespace segment_length_at_1_point_5_l144_144451

-- Definitions for the conditions
def Point := ℝ × ℝ
def Triangle (A B C : Point) := ∃ a b c : ℝ, a = 4 ∧ b = 3 ∧ c = 5 ∧ (A = (0, 0)) ∧ (B = (4, 0)) ∧ (C = (0, 3)) ∧ (c^2 = a^2 + b^2)

noncomputable def length_l (x : ℝ) : ℝ := (4 * (abs ((3/4) * x + 3))) / 5

theorem segment_length_at_1_point_5 (A B C : Point) (h : Triangle A B C) : 
  length_l 1.5 = 3.3 := by 
  sorry

end segment_length_at_1_point_5_l144_144451


namespace paint_can_distribution_l144_144836

-- Definitions based on conditions provided in the problem.
def ratio_red := 3
def ratio_white := 2
def ratio_blue := 1
def total_paint := 60
def ratio_sum := ratio_red + ratio_white + ratio_blue

-- Definition of the problem to be proved.
theorem paint_can_distribution :
  (ratio_red * total_paint) / ratio_sum = 30 ∧
  (ratio_white * total_paint) / ratio_sum = 20 ∧
  (ratio_blue * total_paint) / ratio_sum = 10 := 
by
  sorry

end paint_can_distribution_l144_144836


namespace min_coins_for_any_amount_below_dollar_l144_144066

-- Definitions of coin values
def penny := 1
def nickel := 5
def dime := 10
def half_dollar := 50

-- Statement: The minimum number of coins required to pay any amount less than a dollar
theorem min_coins_for_any_amount_below_dollar :
  ∃ (n : ℕ), n = 11 ∧
  (∀ (amount : ℕ), 1 ≤ amount ∧ amount < 100 →
   ∃ (a b c d : ℕ), amount = a * penny + b * nickel + c * dime + d * half_dollar ∧ 
   a + b + c + d ≤ n) :=
sorry

end min_coins_for_any_amount_below_dollar_l144_144066


namespace isosceles_right_triangle_area_l144_144914

theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) :
  (h = 5 * Real.sqrt 2) →
  (A = 12.5) →
  ∃ (leg : ℝ), (leg = 5) ∧ (A = 1 / 2 * leg^2) := by
  sorry

end isosceles_right_triangle_area_l144_144914


namespace hyperbola_equation_l144_144062

theorem hyperbola_equation (a b : ℝ) (h₁ : a^2 + b^2 = 25) (h₂ : 2 * b / a = 1) : 
  a = 2 * Real.sqrt 5 ∧ b = Real.sqrt 5 ∧ (∀ x y : ℝ, x^2 / (a^2) - y^2 / (b^2) = 1 ↔ x^2 / 20 - y^2 / 5 = 1) :=
by
  sorry

end hyperbola_equation_l144_144062


namespace solve_for_y_l144_144325

theorem solve_for_y (y : ℝ)
  (h1 : 9 * y^2 + 8 * y - 1 = 0)
  (h2 : 27 * y^2 + 44 * y - 7 = 0) : 
  y = 1 / 9 :=
sorry

end solve_for_y_l144_144325


namespace pens_exceed_500_on_saturday_l144_144913

theorem pens_exceed_500_on_saturday :
  ∃ k : ℕ, (5 * 3 ^ k > 500) ∧ k = 6 :=
by 
  sorry   -- Skipping the actual proof here

end pens_exceed_500_on_saturday_l144_144913


namespace cone_shape_in_spherical_coordinates_l144_144154

-- Define the conditions as given in the problem
def spherical_coordinates (rho theta phi c : ℝ) : Prop := 
  rho = c * Real.sin phi

-- Define the main statement to prove
theorem cone_shape_in_spherical_coordinates (rho theta phi c : ℝ) (hpos : 0 < c) :
  spherical_coordinates rho theta phi c → 
  ∃ cone : Prop, cone :=
sorry

end cone_shape_in_spherical_coordinates_l144_144154


namespace find_orig_denominator_l144_144559

-- Definitions as per the conditions
def orig_numer : ℕ := 2
def mod_numer : ℕ := orig_numer + 3

-- The modified fraction yields 1/3
def new_fraction (d : ℕ) : Prop :=
  (mod_numer : ℚ) / (d + 4) = 1 / 3

-- Proof Problem Statement
theorem find_orig_denominator (d : ℕ) : new_fraction d → d = 11 :=
  sorry

end find_orig_denominator_l144_144559


namespace paulina_convertibles_l144_144141

-- Definitions for conditions
def total_cars : ℕ := 125
def percentage_regular_cars : ℚ := 64 / 100
def percentage_trucks : ℚ := 8 / 100
def percentage_convertibles : ℚ := 1 - (percentage_regular_cars + percentage_trucks)

-- Theorem to prove the number of convertibles
theorem paulina_convertibles : (percentage_convertibles * total_cars) = 35 := by
  sorry

end paulina_convertibles_l144_144141


namespace tan_beta_formula_l144_144361

theorem tan_beta_formula (α β : ℝ) 
  (h1 : Real.tan α = -2/3)
  (h2 : Real.tan (α + β) = 1/2) :
  Real.tan β = 7/4 :=
sorry

end tan_beta_formula_l144_144361


namespace hour_hand_degrees_noon_to_2_30_l144_144133

def degrees_moved (hours: ℕ) : ℝ := (hours * 30)

theorem hour_hand_degrees_noon_to_2_30 :
  degrees_moved 2 + degrees_moved 1 / 2 = 75 :=
sorry

end hour_hand_degrees_noon_to_2_30_l144_144133


namespace real_solution_l144_144218

theorem real_solution (x : ℝ) (h : x ≠ 3) :
  (x * (x + 2)) / ((x - 3)^2) ≥ 8 ↔ (2 ≤ x ∧ x < 3) ∨ (3 < x ∧ x ≤ 48) :=
by
  sorry

end real_solution_l144_144218


namespace max_sum_n_value_l144_144207

open Nat

-- Definitions for the problem
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a 0 + (n - 1) * (a 1 - a 0))) / 2

-- Statement of the theorem
theorem max_sum_n_value (a : ℕ → ℤ) (d : ℤ) (h_arith_seq : arithmetic_sequence a d) 
  (h_initial : a 0 > 0) (h_condition : 8 * a 4 = 13 * a 10) : 
  ∃ n, sum_of_first_n_terms a n = max (sum_of_first_n_terms a n) ∧ n = 20 :=
sorry

end max_sum_n_value_l144_144207


namespace bart_earned_14_l144_144695

variable (questions_per_survey money_per_question surveys_monday surveys_tuesday : ℕ → ℝ)
variable (total_surveys total_questions money_earned : ℕ → ℝ)

noncomputable def conditions :=
  let questions_per_survey := 10
  let money_per_question := 0.2
  let surveys_monday := 3
  let surveys_tuesday := 4
  let total_surveys := surveys_monday + surveys_tuesday
  let total_questions := questions_per_survey * total_surveys
  let money_earned := total_questions * money_per_question
  money_earned = 14

theorem bart_earned_14 : conditions :=
by
  -- proof steps
  sorry

end bart_earned_14_l144_144695


namespace percentage_mike_has_l144_144204
-- Definitions and conditions
variables (phone_cost : ℝ) (additional_needed : ℝ)
def amount_mike_has := phone_cost - additional_needed

-- Main statement
theorem percentage_mike_has (phone_cost : ℝ) (additional_needed : ℝ) (h1 : phone_cost = 1300) (h2 : additional_needed = 780) : 
  (amount_mike_has phone_cost additional_needed) * 100 / phone_cost = 40 :=
by
  sorry

end percentage_mike_has_l144_144204


namespace average_gas_mileage_round_trip_l144_144724

noncomputable def average_gas_mileage
  (d1 d2 : ℕ) (m1 m2 : ℕ) : ℚ :=
  let total_distance := d1 + d2
  let total_fuel := (d1 / m1) + (d2 / m2)
  total_distance / total_fuel

theorem average_gas_mileage_round_trip :
  average_gas_mileage 150 180 25 15 = 18.3 := by
  sorry

end average_gas_mileage_round_trip_l144_144724


namespace theresa_needs_15_hours_l144_144195

theorem theresa_needs_15_hours 
  (h1 : ℕ) (h2 : ℕ) (h3 : ℕ) (h4 : ℕ) (h5 : ℕ) (average : ℕ) (weeks : ℕ) (total_hours_first_5 : ℕ) :
  h1 = 10 → h2 = 13 → h3 = 9 → h4 = 14 → h5 = 11 → average = 12 → weeks = 6 → 
  total_hours_first_5 = h1 + h2 + h3 + h4 + h5 → 
  (total_hours_first_5 + x) / weeks = average → x = 15 :=
by
  intros h1_eq h2_eq h3_eq h4_eq h5_eq avg_eq weeks_eq sum_eq avg_eqn
  sorry

end theresa_needs_15_hours_l144_144195


namespace total_heartbeats_during_race_l144_144119

-- Definitions for the conditions
def heart_beats_per_minute : ℕ := 160
def pace_in_minutes_per_mile : ℕ := 6
def total_distance_in_miles : ℕ := 30

-- Main theorem statement
theorem total_heartbeats_during_race : 
  heart_beats_per_minute * pace_in_minutes_per_mile * total_distance_in_miles = 28800 :=
by
  -- Place the proof here
  sorry

end total_heartbeats_during_race_l144_144119


namespace length_of_first_two_CDs_l144_144924

theorem length_of_first_two_CDs
  (x : ℝ)
  (h1 : x + x + 2 * x = 6) :
  x = 1.5 := 
sorry

end length_of_first_two_CDs_l144_144924


namespace Mandy_older_than_Jackson_l144_144212

variable (M J A : ℕ)

-- Given conditions
variables (h1 : J = 20)
variables (h2 : A = (3 * J) / 4)
variables (h3 : (M + 10) + (J + 10) + (A + 10) = 95)

-- Prove that Mandy is 10 years older than Jackson
theorem Mandy_older_than_Jackson : M - J = 10 :=
by
  sorry

end Mandy_older_than_Jackson_l144_144212


namespace calculate_brick_height_cm_l144_144143

noncomputable def wall_length_cm : ℕ := 1000  -- 10 m converted to cm
noncomputable def wall_width_cm : ℕ := 800   -- 8 m converted to cm
noncomputable def wall_height_cm : ℕ := 2450 -- 24.5 m converted to cm

noncomputable def wall_volume_cm3 : ℕ := wall_length_cm * wall_width_cm * wall_height_cm

noncomputable def brick_length_cm : ℕ := 20
noncomputable def brick_width_cm : ℕ := 10
noncomputable def number_of_bricks : ℕ := 12250

noncomputable def brick_area_cm2 : ℕ := brick_length_cm * brick_width_cm

theorem calculate_brick_height_cm (h : ℕ) : brick_area_cm2 * h * number_of_bricks = wall_volume_cm3 → 
  h = wall_volume_cm3 / (brick_area_cm2 * number_of_bricks) := by
  sorry

end calculate_brick_height_cm_l144_144143


namespace algebraic_expression_l144_144169

theorem algebraic_expression (m : ℝ) (hm : m^2 + m - 1 = 0) : 
  m^3 + 2 * m^2 + 2014 = 2015 := 
by
  sorry

end algebraic_expression_l144_144169


namespace percent_increase_bike_helmet_l144_144400

theorem percent_increase_bike_helmet :
  let old_bike_cost := 160
  let old_helmet_cost := 40
  let bike_increase_rate := 0.05
  let helmet_increase_rate := 0.10
  let new_bike_cost := old_bike_cost * (1 + bike_increase_rate)
  let new_helmet_cost := old_helmet_cost * (1 + helmet_increase_rate)
  let old_total_cost := old_bike_cost + old_helmet_cost
  let new_total_cost := new_bike_cost + new_helmet_cost
  let increase_amount := new_total_cost - old_total_cost
  let percent_increase := (increase_amount / old_total_cost) * 100
  percent_increase = 6 :=
by
  sorry

end percent_increase_bike_helmet_l144_144400


namespace gcd_subtraction_result_l144_144694

theorem gcd_subtraction_result : gcd 8100 270 - 8 = 262 := by
  sorry

end gcd_subtraction_result_l144_144694


namespace fraction_of_earth_surface_inhabitable_l144_144384

theorem fraction_of_earth_surface_inhabitable (f_land : ℚ) (f_inhabitable_land : ℚ)
  (h1 : f_land = 1 / 3)
  (h2 : f_inhabitable_land = 2 / 3) :
  f_land * f_inhabitable_land = 2 / 9 :=
by
  sorry

end fraction_of_earth_surface_inhabitable_l144_144384


namespace total_cost_of_rolls_l144_144176

-- Defining the conditions
def price_per_dozen : ℕ := 5
def total_rolls_bought : ℕ := 36
def rolls_per_dozen : ℕ := 12

-- Prove the total cost calculation
theorem total_cost_of_rolls : (total_rolls_bought / rolls_per_dozen) * price_per_dozen = 15 :=
by
  sorry

end total_cost_of_rolls_l144_144176


namespace productivity_increase_l144_144973

variable (a b : ℝ)
variable (h₁ : a > 0) (h₂ : b > 0) (h₃ : (7/8) * b * (1 + x / 100) = 1.05 * b)

theorem productivity_increase (x : ℝ) : x = 20 := sorry

end productivity_increase_l144_144973


namespace find_angle_B_find_triangle_area_l144_144263

open Real

theorem find_angle_B (B : ℝ) (h : sqrt 3 * sin (2 * B) = 1 - cos (2 * B)) : B = π / 3 :=
sorry

theorem find_triangle_area (BC A B : ℝ) (hBC : BC = 2) (hA : A = π / 4) (hB : B = π / 3) :
  let AC := BC * (sin B / sin A)
  let C := π - A - B
  let area := (1 / 2) * AC * BC * sin C
  area = (3 + sqrt 3) / 2 :=
sorry


end find_angle_B_find_triangle_area_l144_144263


namespace line_properties_l144_144308

theorem line_properties : 
  ∃ (m b : ℝ), 
  (∀ x : ℝ, ∀ y : ℝ, (x = 1 ∧ y = 3) ∨ (x = 3 ∧ y = 7) → y = m * x + b) ∧
  m + b = 3 ∧
  (∀ x : ℝ, ∀ y : ℝ, (x = 0 ∧ y = 1) → y = m * x + b) :=
sorry

end line_properties_l144_144308


namespace lena_more_than_nicole_l144_144148

theorem lena_more_than_nicole (L K N : ℕ) 
  (h1 : L = 23)
  (h2 : 4 * K = L + 7)
  (h3 : K = N - 6) : L - N = 10 := sorry

end lena_more_than_nicole_l144_144148


namespace number_of_routes_l144_144116

open Nat

theorem number_of_routes (south_cities north_cities : ℕ) 
  (connections : south_cities = 4 ∧ north_cities = 5) : 
  ∃ routes, routes = (factorial 3) * (5 ^ 4) := 
by
  sorry

end number_of_routes_l144_144116


namespace percentage_import_tax_l144_144070

theorem percentage_import_tax (total_value import_paid excess_amount taxable_amount : ℝ) 
  (h1 : total_value = 2570) 
  (h2 : import_paid = 109.90) 
  (h3 : excess_amount = 1000) 
  (h4 : taxable_amount = total_value - excess_amount) : 
  taxable_amount = 1570 →
  (import_paid / taxable_amount) * 100 = 7 := 
by
  intros h_taxable_amount
  simp [h1, h2, h3, h4, h_taxable_amount]
  sorry -- Proof goes here

end percentage_import_tax_l144_144070


namespace B_finish_work_alone_in_12_days_l144_144309

theorem B_finish_work_alone_in_12_days (A_days B_days both_days : ℕ) :
  A_days = 6 →
  both_days = 4 →
  (1 / A_days + 1 / B_days = 1 / both_days) →
  B_days = 12 :=
by
  intros hA hBoth hRate
  sorry

end B_finish_work_alone_in_12_days_l144_144309


namespace unit_prices_possible_combinations_l144_144821

-- Part 1: Unit Prices
theorem unit_prices (x y : ℕ) (h1 : x = y - 20) (h2 : 3 * x + 2 * y = 340) : x = 60 ∧ y = 80 := 
by 
  sorry

-- Part 2: Possible Combinations
theorem possible_combinations (a : ℕ) (h3 : 60 * a + 80 * (150 - a) ≤ 10840) (h4 : 150 - a ≥ 3 * a / 2) : 
  a = 58 ∨ a = 59 ∨ a = 60 := 
by 
  sorry

end unit_prices_possible_combinations_l144_144821


namespace greatest_prime_factor_of_5_pow_7_plus_6_pow_6_l144_144734

def greatest_prime_factor (n : ℕ) : ℕ :=
  sorry -- Implementation of finding greatest prime factor goes here

theorem greatest_prime_factor_of_5_pow_7_plus_6_pow_6 : 
  greatest_prime_factor (5^7 + 6^6) = 211 := 
by 
  sorry -- Proof of the theorem goes here

end greatest_prime_factor_of_5_pow_7_plus_6_pow_6_l144_144734


namespace tens_digit_of_8_pow_2023_l144_144895

theorem tens_digit_of_8_pow_2023 :
    ∃ d, 0 ≤ d ∧ d < 10 ∧ (8^2023 % 100) / 10 = d ∧ d = 1 :=
by
  sorry

end tens_digit_of_8_pow_2023_l144_144895


namespace johns_pieces_of_gum_l144_144832

theorem johns_pieces_of_gum : 
  (∃ (john cole aubrey : ℕ), 
    cole = 45 ∧ 
    aubrey = 0 ∧ 
    (john + cole + aubrey) = 3 * 33) → 
  ∃ john : ℕ, john = 54 :=
by 
  sorry

end johns_pieces_of_gum_l144_144832


namespace symmetric_circle_eqn_l144_144533

theorem symmetric_circle_eqn :
  ∀ (x y : ℝ),
  ((x + 1)^2 + (y - 1)^2 = 1) ∧ (x - y - 1 = 0) →
  (∀ (x' y' : ℝ), (x' = y + 1) ∧ (y' = x - 1) → (x' + 1)^2 + (y' - 1)^2 = 1) →
  (x - 2)^2 + (y + 2)^2 = 1 :=
by
  intros x y h h_sym
  sorry

end symmetric_circle_eqn_l144_144533


namespace sphere_ratios_l144_144691

theorem sphere_ratios (r1 r2 : ℝ) (h : r1 / r2 = 1 / 3) :
  (4 * π * r1^2) / (4 * π * r2^2) = 1 / 9 ∧ (4 / 3 * π * r1^3) / (4 / 3 * π * r2^3) = 1 / 27 :=
by
  sorry

end sphere_ratios_l144_144691


namespace divisibility_by_six_l144_144149

theorem divisibility_by_six (a x: ℤ) : ∃ t: ℤ, x = 3 * t ∨ x = 3 * t - a^2 → 6 ∣ a * (x^3 + a^2 * x^2 + a^2 - 1) :=
by
  sorry

end divisibility_by_six_l144_144149


namespace Nicky_profit_l144_144681

-- Definitions for conditions
def card1_value : ℕ := 8
def card2_value : ℕ := 8
def received_card_value : ℕ := 21

-- The statement to be proven
theorem Nicky_profit : (received_card_value - (card1_value + card2_value)) = 5 :=
by
  sorry

end Nicky_profit_l144_144681


namespace joe_paint_left_after_third_week_l144_144302

def initial_paint : ℕ := 360

def paint_used_first_week (initial_paint : ℕ) : ℕ := initial_paint / 4

def paint_left_after_first_week (initial_paint : ℕ) : ℕ := initial_paint - paint_used_first_week initial_paint

def paint_used_second_week (paint_left_after_first_week : ℕ) : ℕ := paint_left_after_first_week / 2

def paint_left_after_second_week (paint_left_after_first_week : ℕ) : ℕ := paint_left_after_first_week - paint_used_second_week paint_left_after_first_week

def paint_used_third_week (paint_left_after_second_week : ℕ) : ℕ := paint_left_after_second_week * 2 / 3

def paint_left_after_third_week (paint_left_after_second_week : ℕ) : ℕ := paint_left_after_second_week - paint_used_third_week paint_left_after_second_week

theorem joe_paint_left_after_third_week : 
  paint_left_after_third_week (paint_left_after_second_week (paint_left_after_first_week initial_paint)) = 45 :=
by 
  sorry

end joe_paint_left_after_third_week_l144_144302


namespace abs_c_eq_181_l144_144747

theorem abs_c_eq_181
  (a b c : ℤ)
  (h_gcd : Int.gcd a (Int.gcd b c) = 1)
  (h_eq : a * (Complex.mk 3 2)^4 + b * (Complex.mk 3 2)^3 + c * (Complex.mk 3 2)^2 + b * (Complex.mk 3 2) + a = 0) :
  |c| = 181 :=
sorry

end abs_c_eq_181_l144_144747


namespace total_swordfish_caught_correct_l144_144443

-- Define the parameters and the catch per trip
def shelly_catch_per_trip : ℕ := 5 - 2
def sam_catch_per_trip (shelly_catch : ℕ) : ℕ := shelly_catch - 1
def total_catch_per_trip (shelly_catch sam_catch : ℕ) : ℕ := shelly_catch + sam_catch
def total_trips : ℕ := 5

-- Define the total number of swordfish caught over the trips
def total_swordfish_caught : ℕ :=
  let shelly_catch := shelly_catch_per_trip
  let sam_catch := sam_catch_per_trip shelly_catch
  let total_catch := total_catch_per_trip shelly_catch sam_catch
  total_catch * total_trips

-- The theorem states that the total swordfish caught is 25
theorem total_swordfish_caught_correct : total_swordfish_caught = 25 := by
  sorry

end total_swordfish_caught_correct_l144_144443


namespace stability_comparison_l144_144332

-- Definitions of conditions
def variance_A : ℝ := 3
def variance_B : ℝ := 1.2

-- Definition of the stability metric
def more_stable (performance_A performance_B : ℝ) : Prop :=
  performance_B < performance_A

-- Target Proposition
theorem stability_comparison (h_variance_A : variance_A = 3)
                            (h_variance_B : variance_B = 1.2) :
  more_stable variance_A variance_B = true :=
by
  sorry

end stability_comparison_l144_144332


namespace find_weight_of_sausages_l144_144680

variable (packages : ℕ) (cost_per_pound : ℕ) (total_cost : ℕ) (total_weight : ℕ) (weight_per_package : ℕ)

-- Defining the given conditions
def jake_buys_packages (packages : ℕ) : Prop := packages = 3
def cost_of_sausages (cost_per_pound : ℕ) : Prop := cost_per_pound = 4
def amount_paid (total_cost : ℕ) : Prop := total_cost = 24

-- Derived condition to find total weight
def total_weight_of_sausages (total_cost : ℕ) (cost_per_pound : ℕ) : ℕ := total_cost / cost_per_pound

-- Derived condition to find weight per package
def weight_of_each_package (total_weight : ℕ) (packages : ℕ) : ℕ := total_weight / packages

-- The theorem statement
theorem find_weight_of_sausages
  (h1 : jake_buys_packages packages)
  (h2 : cost_of_sausages cost_per_pound)
  (h3 : amount_paid total_cost) :
  weight_of_each_package (total_weight_of_sausages total_cost cost_per_pound) packages = 2 :=
by
  sorry  -- Proof placeholder

end find_weight_of_sausages_l144_144680


namespace exponent_equality_l144_144696

theorem exponent_equality (s m : ℕ) (h : (2^16) * (25^s) = 5 * (10^m)) : m = 16 :=
by sorry

end exponent_equality_l144_144696


namespace parabola_with_given_focus_l144_144076

-- Defining the given condition of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1

-- Defining the focus coordinates
def focus_coords : ℝ × ℝ := (-3, 0)

-- Proving that the standard equation of the parabola with the left focus of the hyperbola as its focus is y^2 = -12x
theorem parabola_with_given_focus :
  ∃ p : ℝ, (∃ focus : ℝ × ℝ, focus = focus_coords) → 
  ∀ y x : ℝ, y^2 = 4 * p * x → y^2 = -12 * x :=
by
  -- placeholder for proof
  sorry

end parabola_with_given_focus_l144_144076


namespace marriage_year_proof_l144_144536

-- Definitions based on conditions
def marriage_year : ℕ := sorry
def child1_birth_year : ℕ := 1982
def child2_birth_year : ℕ := 1984
def reference_year : ℕ := 1986

-- Age calculations based on reference year
def age_in_1986 (birth_year : ℕ) : ℕ := reference_year - birth_year

-- Combined ages in the reference year
def combined_ages_in_1986 : ℕ := age_in_1986 child1_birth_year + age_in_1986 child2_birth_year

-- The main theorem to prove
theorem marriage_year_proof :
  combined_ages_in_1986 = reference_year - marriage_year →
  marriage_year = 1980 := by
  sorry

end marriage_year_proof_l144_144536


namespace find_x_ceil_mul_l144_144984

theorem find_x_ceil_mul (x : ℝ) (h : ⌈x⌉ * x = 75) : x = 8.333 := by
  sorry

end find_x_ceil_mul_l144_144984


namespace equilateral_triangle_sum_l144_144964

theorem equilateral_triangle_sum (a u v w : ℝ)
  (h1: u^2 + v^2 = w^2):
  w^2 + Real.sqrt 3 * u * v = a^2 := 
sorry

end equilateral_triangle_sum_l144_144964


namespace tammy_driving_rate_l144_144331

-- Define the conditions given in the problem
def total_miles : ℕ := 1980
def total_hours : ℕ := 36

-- Define the desired rate to prove
def expected_rate : ℕ := 55

-- The theorem stating that given the conditions, Tammy's driving rate is correct
theorem tammy_driving_rate :
  total_miles / total_hours = expected_rate :=
by
  -- Detailed proof would go here
  sorry

end tammy_driving_rate_l144_144331


namespace a_minus_2_values_l144_144146

theorem a_minus_2_values (a : ℝ) (h : |a| = 3) : a - 2 = 1 ∨ a - 2 = -5 :=
by {
  -- the theorem states that given the absolute value condition, a - 2 can be 1 or -5
  sorry
}

end a_minus_2_values_l144_144146


namespace product_of_ages_l144_144700

theorem product_of_ages (O Y : ℕ) (h1 : O - Y = 12) (h2 : O + Y = (O - Y) + 40) : O * Y = 640 := by
  sorry

end product_of_ages_l144_144700


namespace patty_coins_value_l144_144356

theorem patty_coins_value (n d q : ℕ) (h₁ : n + d + q = 30) (h₂ : 5 * n + 15 * d - 20 * q = 120) : 
  5 * n + 10 * d + 25 * q = 315 := by
sorry

end patty_coins_value_l144_144356


namespace james_jail_time_l144_144893

-- Definitions based on the conditions
def arson_sentence := 6
def arson_count := 2
def total_arson_sentence := arson_sentence * arson_count

def explosives_sentence := 2 * total_arson_sentence
def terrorism_sentence := 20

-- Total sentence calculation
def total_jail_time := total_arson_sentence + explosives_sentence + terrorism_sentence

-- The theorem we want to prove
theorem james_jail_time : total_jail_time = 56 := by
  sorry

end james_jail_time_l144_144893


namespace part1_part2_l144_144654

open Nat

variable {a : ℕ → ℝ} -- Defining the arithmetic sequence
variable {S : ℕ → ℝ} -- Defining the sum of the first n terms
variable {m n p q : ℕ} -- Defining the positive integers m, n, p, q
variable {d : ℝ} -- The common difference

-- Conditions
axiom arithmetic_sequence_pos_terms : (∀ k, a k = a 1 + (k - 1) * d) ∧ ∀ k, a k > 0
axiom sum_of_first_n_terms : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2
axiom positive_common_difference : d > 0
axiom constraints_on_mnpq : n < p ∧ q < m ∧ m + n = p + q

-- Parts to prove
theorem part1 : a m * a n < a p * a q :=
by sorry

theorem part2 : S m + S n > S p + S q :=
by sorry

end part1_part2_l144_144654


namespace cube_root_sum_is_integer_l144_144554

theorem cube_root_sum_is_integer :
  let a := (2 + (10 / 9) * Real.sqrt 3)^(1/3)
  let b := (2 - (10 / 9) * Real.sqrt 3)^(1/3)
  a + b = 2 := by
  sorry

end cube_root_sum_is_integer_l144_144554


namespace quadruple_perimeter_l144_144983

variable (s : ℝ) -- side length of the original square
variable (x : ℝ) -- perimeter of the original square
variable (P_new : ℝ) -- new perimeter after side length is quadrupled

theorem quadruple_perimeter (h1 : x = 4 * s) (h2 : P_new = 4 * (4 * s)) : P_new = 4 * x := 
by sorry

end quadruple_perimeter_l144_144983


namespace units_digit_diff_is_seven_l144_144080

noncomputable def units_digit_resulting_difference (a b c : ℕ) (h1 : a = c - 3) :=
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  let difference := original - reversed
  difference % 10

theorem units_digit_diff_is_seven (a b c : ℕ) (h1 : a = c - 3) :
  units_digit_resulting_difference a b c h1 = 7 :=
by sorry

end units_digit_diff_is_seven_l144_144080


namespace train_length_l144_144900

theorem train_length (time : ℝ) (speed_in_kmph : ℝ) (speed_in_mps : ℝ) (length_of_train : ℝ) :
  (time = 6) →
  (speed_in_kmph = 96) →
  (speed_in_mps = speed_in_kmph * (5 / 18)) →
  length_of_train = speed_in_mps * time →
  length_of_train = 480 := by
  sorry

end train_length_l144_144900


namespace point_in_third_quadrant_l144_144755

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 1 + 2 * m < 0) : m < -1 / 2 := 
by 
  sorry

end point_in_third_quadrant_l144_144755


namespace units_digit_of_A_is_1_l144_144345

-- Definition of A
def A : ℕ := 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) + 1

-- Main theorem stating that the units digit of A is 1
theorem units_digit_of_A_is_1 : (A % 10) = 1 :=
by 
  -- Given conditions about powers of 3 and their properties in modulo 10
  sorry

end units_digit_of_A_is_1_l144_144345


namespace part_one_part_two_l144_144092

-- First part: Prove that \( (1)(-1)^{2017}+(\frac{1}{2})^{-2}+(3.14-\pi)^{0} = 4\)
theorem part_one : (1 * (-1:ℤ)^2017 + (1/2)^(-2:ℤ) + (3.14 - Real.pi)^0 : ℝ) = 4 := 
  sorry

-- Second part: Prove that \( ((-2*x^2)^3 + 4*x^3*x^3) = -4*x^6 \)
theorem part_two (x : ℝ) : ((-2*x^2)^3 + 4*x^3*x^3) = -4*x^6 := 
  sorry

end part_one_part_two_l144_144092


namespace parabola_directrix_l144_144839

theorem parabola_directrix :
  ∀ (p : ℝ), (y^2 = 6 * x) → (x = -3/2) :=
by
  sorry

end parabola_directrix_l144_144839


namespace election_votes_l144_144735

theorem election_votes (V : ℝ) 
    (h1 : ∃ c1 c2 : ℝ, c1 + c2 = V ∧ c1 = 0.60 * V ∧ c2 = 0.40 * V)
    (h2 : ∃ m : ℝ, m = 280 ∧ 0.60 * V - 0.40 * V = m) : 
    V = 1400 :=
by
  sorry

end election_votes_l144_144735


namespace product_of_values_of_x_l144_144003

theorem product_of_values_of_x : 
  (∃ x : ℝ, |x^2 - 7| - 3 = -1) → 
  (∀ x1 x2 x3 x4 : ℝ, 
    (|x1^2 - 7| - 3 = -1) ∧
    (|x2^2 - 7| - 3 = -1) ∧
    (|x3^2 - 7| - 3 = -1) ∧
    (|x4^2 - 7| - 3 = -1) 
    → x1 * x2 * x3 * x4 = 45) :=
sorry

end product_of_values_of_x_l144_144003


namespace second_option_feasible_l144_144088

def Individual : Type := String
def M : Individual := "M"
def I : Individual := "I"
def P : Individual := "P"
def A : Individual := "A"

variable (is_sitting : Individual → Prop)

-- Given conditions
axiom fact1 : ¬ is_sitting M
axiom fact2 : ¬ is_sitting A
axiom fact3 : ¬ is_sitting M → is_sitting I
axiom fact4 : is_sitting I → is_sitting P

theorem second_option_feasible :
  is_sitting I ∧ is_sitting P ∧ ¬ is_sitting M ∧ ¬ is_sitting A :=
by
  sorry

end second_option_feasible_l144_144088


namespace no_more_than_one_100_l144_144814

-- Define the score variables and the conditions
variables (R P M : ℕ)

-- Given conditions: R = P - 3 and P = M - 7
def score_conditions : Prop := R = P - 3 ∧ P = M - 7

-- The maximum score condition
def max_score_condition : Prop := R ≤ 100 ∧ P ≤ 100 ∧ M ≤ 100

-- The goal: it is impossible for Vanya to have scored 100 in more than one exam
theorem no_more_than_one_100 (R P M : ℕ) (h1 : score_conditions R P M) (h2 : max_score_condition R P M) :
  (R = 100 ∧ P = 100) ∨ (P = 100 ∧ M = 100) ∨ (M = 100 ∧ R = 100) → false :=
sorry

end no_more_than_one_100_l144_144814


namespace R_depends_on_d_and_n_l144_144922

-- Define the given properties of the arithmetic progression sums
def s1 (a d n : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2
def s3 (a d n : ℕ) : ℕ := (3 * n * (2 * a + (3 * n - 1) * d)) / 2
def s5 (a d n : ℕ) : ℕ := (5 * n * (2 * a + (5 * n - 1) * d)) / 2

-- Define R in terms of s1, s3, and s5
def R (a d n : ℕ) : ℕ := s5 a d n - s3 a d n - s1 a d n

-- The main theorem to prove the statement about R's dependency
theorem R_depends_on_d_and_n (a d n : ℕ) : R a d n = 7 * d * n^2 := by 
  sorry

end R_depends_on_d_and_n_l144_144922


namespace sum_of_first_ten_nice_numbers_is_182_l144_144134

def is_proper_divisor (n d : ℕ) : Prop :=
  d > 1 ∧ d < n ∧ n % d = 0

def is_nice (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, is_proper_divisor n m → ∃ p q, n = p * q ∧ p ≠ q

def first_ten_nice_numbers : List ℕ := [6, 8, 10, 14, 15, 21, 22, 26, 27, 33]

def sum_first_ten_nice_numbers : ℕ := first_ten_nice_numbers.sum

theorem sum_of_first_ten_nice_numbers_is_182 :
  sum_first_ten_nice_numbers = 182 :=
by
  sorry

end sum_of_first_ten_nice_numbers_is_182_l144_144134


namespace B_pow_101_l144_144198

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]
  ]

theorem B_pow_101 :
  B ^ 101 = ![
    ![0, 0, 1],
    ![1, 0, 0],
    ![0, 1, 0]
  ] :=
  sorry

end B_pow_101_l144_144198


namespace solve_abs_equation_l144_144871

-- Define the condition for the equation
def condition (x : ℝ) : Prop := 3 * x + 5 ≥ 0

-- The main theorem to prove that x = 1/5 is the only solution
theorem solve_abs_equation (x : ℝ) (h : condition x) : |2 * x - 6| = 3 * x + 5 ↔ x = 1 / 5 := by
  sorry

end solve_abs_equation_l144_144871


namespace exists_complex_on_line_y_eq_neg_x_l144_144633

open Complex

theorem exists_complex_on_line_y_eq_neg_x :
  ∃ (z : ℂ), ∃ (a b : ℝ), z = a + b * I ∧ b = -a :=
by
  use 1 - I
  use 1, -1
  sorry

end exists_complex_on_line_y_eq_neg_x_l144_144633


namespace eating_time_l144_144553

-- Defining the terms based on the conditions provided
def rate_mr_swift := 1 / 15 -- Mr. Swift eats 1 pound in 15 minutes
def rate_mr_slow := 1 / 45  -- Mr. Slow eats 1 pound in 45 minutes

-- Combined eating rate of Mr. Swift and Mr. Slow
def combined_rate := rate_mr_swift + rate_mr_slow

-- Total amount of cereal to be consumed
def total_cereal := 4 -- pounds

-- Proving the total time to eat the cereal
theorem eating_time :
  (total_cereal / combined_rate) = 45 :=
by
  sorry

end eating_time_l144_144553


namespace find_common_ratio_l144_144316

noncomputable def geometric_seq_sum (a₁ q : ℂ) (n : ℕ) :=
if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem find_common_ratio (a₁ q : ℂ) :
(geometric_seq_sum a₁ q 8) / (geometric_seq_sum a₁ q 4) = 2 → q = 1 :=
by
  intro h
  sorry

end find_common_ratio_l144_144316


namespace part1_l144_144281

theorem part1 (m n p : ℝ) (h1 : m > n) (h2 : n > 0) (h3 : p > 0) : 
  (n / m) < (n + p) / (m + p) := 
sorry

end part1_l144_144281


namespace tamara_diff_3kim_height_l144_144515

variables (K T X : ℕ) -- Kim's height, Tamara's height, and the difference inches respectively

-- Conditions
axiom ht_Tamara : T = 68
axiom combined_ht : T + K = 92
axiom diff_eqn : T = 3 * K - X

theorem tamara_diff_3kim_height (h₁ : T = 68) (h₂ : T + K = 92) (h₃ : T = 3 * K - X) : X = 4 :=
by
  sorry

end tamara_diff_3kim_height_l144_144515


namespace geometric_sequence_sum_point_on_line_l144_144726

theorem geometric_sequence_sum_point_on_line
  (S : ℕ → ℝ) (a : ℕ → ℝ) (t : ℝ) (r : ℝ)
  (h1 : a 1 = t)
  (h2 : ∀ n : ℕ, a (n + 1) = t * r ^ n)
  (h3 : ∀ n : ℕ, S n = t * (1 - r ^ n) / (1 - r))
  (h4 : ∀ n : ℕ, (S n, a (n + 1)) ∈ {p : ℝ × ℝ | p.2 = 2 * p.1 + 1})
  : t = 1 :=
by
  sorry

end geometric_sequence_sum_point_on_line_l144_144726


namespace sin_value_l144_144432

theorem sin_value (x : ℝ) (h : Real.sin (x + π / 3) = Real.sqrt 3 / 3) :
  Real.sin (2 * π / 3 - x) = Real.sqrt 3 / 3 :=
by
  sorry

end sin_value_l144_144432


namespace possible_values_of_b_l144_144115

theorem possible_values_of_b (r s : ℝ) (t t' : ℝ)
  (hp : ∀ x, x^3 + a * x + b = 0 → (x = r ∨ x = s ∨ x = t))
  (hq : ∀ x, x^3 + a * x + b + 240 = 0 → (x = r + 4 ∨ x = s - 3 ∨ x = t'))
  (h_sum_p : r + s + t = 0)
  (h_sum_q : (r + 4) + (s - 3) + t' = 0)
  (ha_p : a = r * s + r * t + s * t)
  (ha_q : a = (r + 4) * (s - 3) + (r + 4) * (t' - 1) + (s - 3) * (t' - 1))
  (ht'_def : t' = t - 1)
  : b = -330 ∨ b = 90 :=
by
  sorry

end possible_values_of_b_l144_144115


namespace equilateral_prism_lateral_edge_length_l144_144017

theorem equilateral_prism_lateral_edge_length
  (base_side_length : ℝ)
  (h_base : base_side_length = 1)
  (perpendicular_diagonals : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = base_side_length ∧ b = lateral_edge ∧ c = some_diagonal_length ∧ lateral_edge ≠ 0)
  : ∀ lateral_edge : ℝ, lateral_edge = (Real.sqrt 2) / 2 := sorry

end equilateral_prism_lateral_edge_length_l144_144017


namespace trajectory_of_center_l144_144494

theorem trajectory_of_center :
  ∃ (x y : ℝ), (x + 1) ^ 2 + y ^ 2 = 49 / 4 ∧ (x - 1) ^ 2 + y ^ 2 = 1 / 4 ∧ ( ∀ P, (P = (x, y) → (P.1^2) / 4 + (P.2^2) / 3 = 1) ) := sorry

end trajectory_of_center_l144_144494


namespace min_distance_eq_3_l144_144945

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (Real.pi / 3 * x + Real.pi / 4)

theorem min_distance_eq_3 (x₁ x₂ : ℝ) 
  (h₁ : f x₁ ≤ f x) (h₂ : f x ≤ f x₂) 
  (x : ℝ) :
  |x₁ - x₂| = 3 :=
by
  -- Sorry placeholder for proof.
  sorry

end min_distance_eq_3_l144_144945


namespace Yan_distance_ratio_l144_144976

theorem Yan_distance_ratio (d x : ℝ) (v : ℝ) (h1 : d > 0) (h2 : x > 0) (h3 : x < d)
  (h4 : 7 * (d - x) = x + d) : 
  x / (d - x) = 3 / 4 :=
by
  sorry

end Yan_distance_ratio_l144_144976


namespace determinant_nonnegative_of_skew_symmetric_matrix_l144_144581

theorem determinant_nonnegative_of_skew_symmetric_matrix
  (a b c d e f : ℝ)
  (A : Matrix (Fin 4) (Fin 4) ℝ)
  (hA : A = ![
    ![0, a, b, c],
    ![-a, 0, d, e],
    ![-b, -d, 0, f],
    ![-c, -e, -f, 0]]) :
  0 ≤ Matrix.det A := by
  sorry

end determinant_nonnegative_of_skew_symmetric_matrix_l144_144581


namespace hockeyPlayers_count_l144_144124

def numPlayers := 50
def cricketPlayers := 12
def footballPlayers := 11
def softballPlayers := 10

theorem hockeyPlayers_count : 
  let hockeyPlayers := numPlayers - (cricketPlayers + footballPlayers + softballPlayers)
  hockeyPlayers = 17 :=
by
  sorry

end hockeyPlayers_count_l144_144124


namespace max_xy_l144_144287

theorem max_xy (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_eq : 2 * x + 3 * y = 6) : 
  xy ≤ (3/2) :=
sorry

end max_xy_l144_144287


namespace taxi_ride_distance_l144_144180

theorem taxi_ride_distance
  (initial_charge : ℝ) (additional_charge : ℝ) 
  (total_charge : ℝ) (initial_increment : ℝ) (distance_increment : ℝ)
  (initial_charge_eq : initial_charge = 2.10) 
  (additional_charge_eq : additional_charge = 0.40) 
  (total_charge_eq : total_charge = 17.70) 
  (initial_increment_eq : initial_increment = 1/5) 
  (distance_increment_eq : distance_increment = 1/5) : 
  (distance : ℝ) = 8 :=
by sorry

end taxi_ride_distance_l144_144180


namespace father_age_when_sum_100_l144_144369

/-- Given the current ages of the mother and father, prove that the father's age will be 51 years old when the sum of their ages is 100. -/
theorem father_age_when_sum_100 (M F : ℕ) (hM : M = 42) (hF : F = 44) :
  ∃ X : ℕ, (M + X) + (F + X) = 100 ∧ F + X = 51 :=
by
  sorry

end father_age_when_sum_100_l144_144369


namespace moles_NaCl_formed_in_reaction_l144_144649

noncomputable def moles_of_NaCl_formed (moles_NaOH moles_HCl : ℕ) : ℕ :=
  if moles_NaOH = 1 ∧ moles_HCl = 1 then 1 else 0

theorem moles_NaCl_formed_in_reaction : moles_of_NaCl_formed 1 1 = 1 := 
by
  sorry

end moles_NaCl_formed_in_reaction_l144_144649


namespace tracy_dog_food_l144_144972

theorem tracy_dog_food
(f : ℕ) (c : ℝ) (m : ℕ) (d : ℕ)
(hf : f = 4) (hc : c = 2.25) (hm : m = 3) (hd : d = 2) :
  (f * c / m) / d = 1.5 :=
by
  sorry

end tracy_dog_food_l144_144972


namespace average_of_seven_starting_with_d_l144_144619

theorem average_of_seven_starting_with_d (c d : ℕ) (h : d = (c + 3)) : 
  (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = c + 6 :=
by
  sorry

end average_of_seven_starting_with_d_l144_144619


namespace committee_problem_solution_l144_144301

def committee_problem : Prop :=
  let total_committees := Nat.choose 15 5
  let zero_profs_committees := Nat.choose 8 5
  let one_prof_committees := (Nat.choose 7 1) * (Nat.choose 8 4)
  let undesirable_committees := zero_profs_committees + one_prof_committees
  let desired_committees := total_committees - undesirable_committees
  desired_committees = 2457

theorem committee_problem_solution : committee_problem :=
by
  sorry

end committee_problem_solution_l144_144301


namespace infinite_series_eq_5_over_16_l144_144709

noncomputable def infinite_series_sum : ℝ :=
  ∑' (n : ℕ), (n + 1 : ℝ) / (5 ^ (n + 1))

theorem infinite_series_eq_5_over_16 :
  infinite_series_sum = 5 / 16 :=
sorry

end infinite_series_eq_5_over_16_l144_144709


namespace determine_range_of_m_l144_144817

variable {m : ℝ}

-- Condition (p) for all x in ℝ, x^2 - mx + 3/2 > 0
def condition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - m * x + (3 / 2) > 0

-- Condition (q) the foci of the ellipse lie on the x-axis, implying 2 < m < 3
def condition_q (m : ℝ) : Prop :=
  (m - 1 > 0) ∧ ((3 - m) > 0) ∧ ((m - 1) > (3 - m))

theorem determine_range_of_m (h1 : condition_p m) (h2 : condition_q m) : 2 < m ∧ m < Real.sqrt 6 :=
  sorry

end determine_range_of_m_l144_144817


namespace geometric_seq_condition_l144_144579

variable (n : ℕ) (a : ℕ → ℝ)

-- The definition of a geometric sequence
def is_geometric_seq (a : ℕ → ℝ) (n : ℕ) : Prop :=
  a (n + 1) * a (n + 1) = a n * a (n + 2)

-- The main theorem statement
theorem geometric_seq_condition :
  (is_geometric_seq a n → ∀ n, |a n| ≥ 0) →
  ∃ (a : ℕ → ℝ), (∀ n, a n * a (n + 2) = a (n + 1) * a (n + 1)) →
  (∀ m, a m = 0 → ¬(is_geometric_seq a n)) :=
sorry

end geometric_seq_condition_l144_144579


namespace find_number_is_9_l144_144045

noncomputable def number (y : ℕ) : ℕ := 3^(12 / y)

theorem find_number_is_9 (y : ℕ) (h_y : y = 6) (h_eq : (number y)^y = 3^12) : number y = 9 :=
by
  sorry

end find_number_is_9_l144_144045


namespace exp_value_l144_144992

theorem exp_value (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + 2 * n) = 18 := 
by
  sorry

end exp_value_l144_144992


namespace relay_race_order_count_l144_144303

-- Definitions based on the given conditions
def team_members : List String := ["Sam", "Priya", "Jordan", "Luis"]
def first_runner := "Sam"
def last_runner := "Jordan"

-- Theorem stating the number of different possible orders
theorem relay_race_order_count {team_members first_runner last_runner} :
  (team_members = ["Sam", "Priya", "Jordan", "Luis"]) →
  (first_runner = "Sam") →
  (last_runner = "Jordan") →
  (2 = 2) :=
by
  intros _ _ _
  sorry

end relay_race_order_count_l144_144303


namespace problem_solution_l144_144248

theorem problem_solution (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : a + b = 1) :
  (a + 1 / b) ^ 2 + (b + 1 / a) ^ 2 ≥ 25 / 2 :=
sorry

end problem_solution_l144_144248


namespace find_S_l144_144689

noncomputable def S : ℕ+ → ℝ := sorry
noncomputable def a : ℕ+ → ℝ := sorry

axiom h : ∀ n : ℕ+, 2 * S n = 3 * a n + 4

theorem find_S : ∀ n : ℕ+, S n = 2 - 2 * 3 ^ (n : ℕ) :=
  sorry

end find_S_l144_144689


namespace find_number_l144_144203

theorem find_number (x : ℕ) (h : x - 263 + 419 = 725) : x = 569 :=
sorry

end find_number_l144_144203


namespace solve_for_a_l144_144035

noncomputable def a := 3.6

theorem solve_for_a (h : 4 * ((a * 0.48 * 2.5) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005) : 
    a = 3.6 :=
by
  sorry

end solve_for_a_l144_144035


namespace wildcats_points_l144_144188

theorem wildcats_points (panthers_points wildcats_additional_points wildcats_points : ℕ)
  (h_panthers : panthers_points = 17)
  (h_wildcats : wildcats_additional_points = 19)
  (h_wildcats_points : wildcats_points = panthers_points + wildcats_additional_points) :
  wildcats_points = 36 :=
by
  have h1 : panthers_points = 17 := h_panthers
  have h2 : wildcats_additional_points = 19 := h_wildcats
  have h3 : wildcats_points = panthers_points + wildcats_additional_points := h_wildcats_points
  sorry

end wildcats_points_l144_144188


namespace simplify_expr_l144_144979

theorem simplify_expr (x : ℝ) : 1 - (1 - (1 - (1 + (1 - (1 - x))))) = 2 - x :=
by
  sorry

end simplify_expr_l144_144979


namespace third_player_games_l144_144989

theorem third_player_games (p1 p2 p3 : ℕ) (h1 : p1 = 21) (h2 : p2 = 10)
  (total_games : p1 = p2 + p3) : p3 = 11 :=
by
  sorry

end third_player_games_l144_144989


namespace forty_percent_of_n_l144_144813

theorem forty_percent_of_n (N : ℝ) (h : (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 10) : 0.40 * N = 120 := by
  sorry

end forty_percent_of_n_l144_144813


namespace original_price_vase_l144_144498

-- Definitions based on the conditions and problem elements
def original_price (P : ℝ) : Prop :=
  0.825 * P = 165

-- Statement to prove equivalence
theorem original_price_vase : ∃ P : ℝ, original_price P ∧ P = 200 :=
  by
    sorry

end original_price_vase_l144_144498


namespace even_function_m_value_l144_144652

theorem even_function_m_value {m : ℤ} (h : ∀ (x : ℝ), (m^2 - m - 1) * (-x)^m = (m^2 - m - 1) * x^m) : m = 2 := 
by
  sorry

end even_function_m_value_l144_144652


namespace pair_comparison_l144_144261

theorem pair_comparison :
  (∀ (a b : ℤ), (a, b) = (-2^4, (-2)^4) → a ≠ b) ∧
  (∀ (a b : ℤ), (a, b) = (5^3, 3^5) → a ≠ b) ∧
  (∀ (a b : ℤ), (a, b) = (-(-3), -|-3|) → a ≠ b) ∧
  (∀ (a b : ℤ), (a, b) = ((-1)^2, (-1)^2008) → a = b) :=
by
  sorry

end pair_comparison_l144_144261


namespace age_difference_l144_144738

def age1 : ℕ := 10
def age2 : ℕ := age1 - 2
def age3 : ℕ := age2 + 4
def age4 : ℕ := age3 / 2
def age5 : ℕ := age4 + 20
def avg : ℕ := (age1 + age5) / 2

theorem age_difference :
  (age3 - age2) = 4 ∧ avg = 18 := by
  sorry

end age_difference_l144_144738


namespace probability_two_hearts_is_one_seventeenth_l144_144452

-- Define the problem parameters
def totalCards : ℕ := 52
def hearts : ℕ := 13
def drawCount : ℕ := 2

-- Define function to calculate combinations
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Define the probability calculation
def probability_drawing_two_hearts : ℚ :=
  (combination hearts drawCount) / (combination totalCards drawCount)

-- State the theorem to be proved
theorem probability_two_hearts_is_one_seventeenth :
  probability_drawing_two_hearts = 1 / 17 :=
by
  -- Proof not required, so provide sorry
  sorry

end probability_two_hearts_is_one_seventeenth_l144_144452


namespace determine_constants_l144_144189

theorem determine_constants :
  ∃ (a b c p : ℝ), (a = -1) ∧ (b = -1) ∧ (c = -1) ∧ (p = 3) ∧
  (∀ x : ℝ, x^3 + p*x^2 + 3*x - 10 = 0 ↔ (x = a ∨ x = b ∨ x = c)) ∧ 
  c - b = b - a ∧ c - b > 0 :=
by
  sorry

end determine_constants_l144_144189


namespace sequence_inequality_l144_144842

open Real

def seq (F : ℕ → ℝ) : Prop :=
  F 1 = 1 ∧ F 2 = 2 ∧ ∀ n ≥ 1, F (n + 2) = F (n + 1) + F n

theorem sequence_inequality (F : ℕ → ℝ) (h : seq F) (n : ℕ) : 
  sqrt (F (n+1))^(1/(n:ℝ)) ≥ 1 + 1 / sqrt (F n)^(1/(n:ℝ)) :=
sorry

end sequence_inequality_l144_144842


namespace fraction_division_l144_144197

theorem fraction_division (a b c d e : ℚ)
  (h1 : a = 3 / 7)
  (h2 : b = 1 / 3)
  (h3 : d = 2 / 5)
  (h4 : c = a + b)
  (h5 : e = c / d):
  e = 40 / 21 := by
  sorry

end fraction_division_l144_144197


namespace total_money_calculation_l144_144433

theorem total_money_calculation (N50 N500 Total_money : ℕ) 
( h₁ : N50 = 37 ) 
( h₂ : N50 + N500 = 54 ) :
Total_money = N50 * 50 + N500 * 500 ↔ Total_money = 10350 := 
by 
  sorry

end total_money_calculation_l144_144433


namespace cube_surface_area_is_24_l144_144845

def edge_length : ℝ := 2

def surface_area_of_cube (a : ℝ) : ℝ := 6 * a * a

theorem cube_surface_area_is_24 : surface_area_of_cube edge_length = 24 := 
by 
  -- Compute the surface area of the cube with given edge length
  -- surface_area_of_cube 2 = 6 * 2 * 2 = 24
  sorry

end cube_surface_area_is_24_l144_144845


namespace grace_pennies_l144_144315

theorem grace_pennies (dime_value nickel_value : ℕ) (dimes nickels : ℕ) 
  (h₁ : dime_value = 10) (h₂ : nickel_value = 5) (h₃ : dimes = 10) (h₄ : nickels = 10) : 
  dimes * dime_value + nickels * nickel_value = 150 := 
by 
  sorry

end grace_pennies_l144_144315


namespace poly_value_at_two_l144_144500

def f (x : ℝ) : ℝ := x^5 + 2 * x^4 + 3 * x^3 + 4 * x^2 + 5 * x + 6

theorem poly_value_at_two : f 2 = 216 :=
by
  unfold f
  norm_num
  sorry

end poly_value_at_two_l144_144500


namespace smallest_value_of_a_l144_144000

theorem smallest_value_of_a (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : 2 * b = a + c) (h4 : c^2 = a * b) : a = -4 :=
by
  sorry

end smallest_value_of_a_l144_144000


namespace solve_rectangular_field_problem_l144_144725

-- Define the problem
def f (L W : ℝ) := L * W = 80 ∧ 2 * W + L = 28

-- Define the length of the uncovered side
def length_of_uncovered_side (L: ℝ) := L = 20

-- The statement we need to prove
theorem solve_rectangular_field_problem (L W : ℝ) (h : f L W) : length_of_uncovered_side L :=
by
  sorry

end solve_rectangular_field_problem_l144_144725


namespace smallest_positive_integer_solution_l144_144727

theorem smallest_positive_integer_solution (x : ℕ) (h : 5 * x ≡ 17 [MOD 29]) : x = 15 :=
sorry

end smallest_positive_integer_solution_l144_144727


namespace fraction_second_year_not_third_year_l144_144938

theorem fraction_second_year_not_third_year (N T S : ℕ) (hN : N = 100) (hT : T = N / 2) (hS : S = N * 3 / 10) :
  (S / (N - T) : ℚ) = 3 / 5 :=
by
  rw [hN, hT, hS]
  norm_num
  sorry

end fraction_second_year_not_third_year_l144_144938


namespace speed_in_still_water_l144_144721

-- Given conditions
def upstream_speed : ℝ := 60
def downstream_speed : ℝ := 90

-- Proof that the speed of the man in still water is 75 kmph
theorem speed_in_still_water :
  (upstream_speed + downstream_speed) / 2 = 75 := 
by
  sorry

end speed_in_still_water_l144_144721


namespace gcd_max_value_l144_144645

theorem gcd_max_value : ∀ (n : ℕ), n > 0 → ∃ (d : ℕ), d = 9 ∧ d ∣ gcd (13 * n + 4) (8 * n + 3) :=
by
  sorry

end gcd_max_value_l144_144645


namespace plane_equation_of_points_l144_144431

theorem plane_equation_of_points :
  ∃ A B C D : ℤ, A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1 ∧
  ∀ x y z : ℤ, (15 * x + 7 * y + 17 * z - 26 = 0) ↔
  (A * x + B * y + C * z + D = 0) :=
by
  sorry

end plane_equation_of_points_l144_144431


namespace general_formula_for_a_n_l144_144953

-- Given conditions
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
variable (h1 : ∀ n : ℕ, a n > 0)
variable (h2 : ∀ n : ℕ, 4 * S n = (a n - 1) * (a n + 3))

theorem general_formula_for_a_n :
  ∀ n : ℕ, a n = 2 * n + 1 :=
sorry

end general_formula_for_a_n_l144_144953


namespace hyperbola_params_l144_144669

theorem hyperbola_params (a b h k : ℝ) (h_positivity : a > 0 ∧ b > 0)
  (asymptote_1 : ∀ x : ℝ, ∃ y : ℝ, y = (3/2) * x + 4)
  (asymptote_2 : ∀ x : ℝ, ∃ y : ℝ, y = -(3/2) * x + 2)
  (passes_through : ∃ x y : ℝ, x = 2 ∧ y = 8 ∧ (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1) 
  (standard_form : ∀ x y : ℝ, ((y - k)^2 / a^2 - (x - h)^2 / b^2 = 1)) : 
  a + h = 7/3 := sorry

end hyperbola_params_l144_144669


namespace mod_graph_sum_l144_144640

theorem mod_graph_sum (x₀ y₀ : ℕ) (h₁ : 2 * x₀ ≡ 1 [MOD 11]) (h₂ : 3 * y₀ ≡ 10 [MOD 11]) : x₀ + y₀ = 13 :=
by
  sorry

end mod_graph_sum_l144_144640


namespace dot_product_focus_hyperbola_l144_144374

-- Definitions related to the problem of the hyperbola
def hyperbola (x y : ℝ) : Prop := (x^2 / 3) - y^2 = 1

def is_focus (c : ℝ) (x y : ℝ) : Prop := (x = c ∧ y = 0) ∨ (x = -c ∧ y = 0)

-- Problem conditions
def point_on_hyperbola (p : ℝ × ℝ) : Prop := hyperbola p.1 p.2

def triangle_area (a b c : ℝ × ℝ) (area : ℝ) : Prop :=
  0.5 * (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2)) = area

def foci_of_hyperbola : (ℝ × ℝ) × (ℝ × ℝ) := ((2, 0), (-2, 0))

-- Main statement to prove
theorem dot_product_focus_hyperbola
  (m n : ℝ)
  (hP : point_on_hyperbola (m, n))
  (hArea : triangle_area (2, 0) (m, n) (-2, 0) 2) :
  ((-2 - m) * (2 - m) + (-n) * (-n)) = 3 :=
sorry

end dot_product_focus_hyperbola_l144_144374


namespace grandmother_total_payment_l144_144095

theorem grandmother_total_payment
  (senior_discount : Real := 0.30)
  (children_discount : Real := 0.40)
  (num_seniors : Nat := 2)
  (num_children : Nat := 2)
  (num_regular : Nat := 2)
  (senior_ticket_price : Real := 7.50)
  (regular_ticket_price : Real := senior_ticket_price / (1 - senior_discount))
  (children_ticket_price : Real := regular_ticket_price * (1 - children_discount))
  : (num_seniors * senior_ticket_price + num_regular * regular_ticket_price + num_children * children_ticket_price) = 49.27 := 
by
  sorry

end grandmother_total_payment_l144_144095


namespace equal_playing_time_for_each_player_l144_144120

-- Defining the conditions
def num_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45
def total_field_time : ℕ := players_on_field * match_duration
def equal_playing_time : ℕ := total_field_time / num_players

-- Stating the question and the proof problem
theorem equal_playing_time_for_each_player : equal_playing_time = 36 := 
  by sorry

end equal_playing_time_for_each_player_l144_144120


namespace tom_has_hours_to_spare_l144_144789

theorem tom_has_hours_to_spare 
  (num_walls : ℕ) 
  (wall_length wall_height : ℕ) 
  (painting_rate : ℕ) 
  (total_hours : ℕ) 
  (num_walls_eq : num_walls = 5) 
  (wall_length_eq : wall_length = 2) 
  (wall_height_eq : wall_height = 3) 
  (painting_rate_eq : painting_rate = 10) 
  (total_hours_eq : total_hours = 10)
  : total_hours - (num_walls * wall_length * wall_height * painting_rate) / 60 = 5 := 
sorry

end tom_has_hours_to_spare_l144_144789


namespace find_line_equation_l144_144980

-- Define the point (2, -1) which the line passes through
def point : ℝ × ℝ := (2, -1)

-- Define the line perpendicular to 2x - 3y = 1
def perpendicular_line (x y : ℝ) : Prop := 2 * x - 3 * y - 1 = 0

-- The equation of the line we are supposed to find
def equation_of_line (x y : ℝ) : Prop := 3 * x + 2 * y - 4 = 0

-- Proof problem: prove the equation satisfies given the conditions
theorem find_line_equation :
  (equation_of_line point.1 point.2) ∧ 
  (∃ (a b c : ℝ), ∀ (x y : ℝ), perpendicular_line x y → equation_of_line x y) := sorry

end find_line_equation_l144_144980


namespace infinite_product_equals_nine_l144_144068

noncomputable def infinite_product : ℝ :=
  ∏' n : ℕ, ite (n = 0) 1 (3^(n * (1 / 2^n)))

theorem infinite_product_equals_nine : infinite_product = 9 := sorry

end infinite_product_equals_nine_l144_144068


namespace triangle_count_l144_144199

-- Define the function to compute the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  if k > n then 0 else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the number of points on each side
def pointsAB : ℕ := 6
def pointsBC : ℕ := 7

-- Compute the number of triangles that can be formed
theorem triangle_count (h₁ : pointsAB = 6) (h₂ : pointsBC = 7) : 
  (binom pointsAB 2) * (binom pointsBC 1) + (binom pointsBC 2) * (binom pointsAB 1) = 231 := by
  sorry

end triangle_count_l144_144199


namespace determine_peter_and_liar_l144_144403

structure Brothers where
  names : Fin 2 → String
  tells_truth : Fin 2 → Bool -- true if the brother tells the truth, false if lies
  (unique_truth_teller : ∃! (i : Fin 2), tells_truth i)
  (one_is_peter : ∃ (i : Fin 2), names i = "Péter")

theorem determine_peter_and_liar (B : Brothers) : 
  ∃ (peter liar : Fin 2), B.names peter = "Péter" ∧ B.tells_truth liar = false ∧
    ∀ (p q : Fin 2), B.names p = "Péter" → B.tells_truth q = false → p = peter ∧ q = liar :=
by
  sorry

end determine_peter_and_liar_l144_144403


namespace mary_money_left_l144_144574

variable (p : ℝ)

theorem mary_money_left :
  have cost_drinks := 3 * p
  have cost_medium_pizza := 2 * p
  have cost_large_pizza := 3 * p
  let total_cost := cost_drinks + cost_medium_pizza + cost_large_pizza
  30 - total_cost = 30 - 8 * p := by {
    sorry
  }

end mary_money_left_l144_144574


namespace boys_tried_out_l144_144597

theorem boys_tried_out (G B C N : ℕ) (hG : G = 9) (hC : C = 2) (hN : N = 21) (h : G + B - C = N) : B = 14 :=
by
  -- The proof is omitted, focusing only on stating the theorem
  sorry

end boys_tried_out_l144_144597


namespace rectangle_area_l144_144441

theorem rectangle_area (a w l : ℝ) (h_square_area : a = 36) 
    (h_rect_width : w * w = a) 
    (h_rect_length : l = 3 * w) : w * l = 108 := 
sorry

end rectangle_area_l144_144441


namespace find_remainder_l144_144125

theorem find_remainder (P Q R D D' Q' R' C : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R') :
  (P % (D * D')) = (D * R' + R + C) :=
sorry

end find_remainder_l144_144125


namespace cards_sum_l144_144375

theorem cards_sum (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) :
  a + b + c + d + e + f + g + h = 21 :=
by
  sorry

end cards_sum_l144_144375


namespace amount_paid_to_Y_l144_144822

-- Definition of the conditions.
def total_payment (X Y : ℕ) : Prop := X + Y = 330
def payment_relation (X Y : ℕ) : Prop := X = 12 * Y / 10

-- The theorem we want to prove.
theorem amount_paid_to_Y (X Y : ℕ) (h1 : total_payment X Y) (h2 : payment_relation X Y) : Y = 150 := 
by 
  sorry

end amount_paid_to_Y_l144_144822


namespace winner_for_2023_winner_for_2024_l144_144475

-- Definitions for the game conditions
def barbara_moves : List ℕ := [3, 5]
def jenna_moves : List ℕ := [1, 4, 5]

-- Lean theorem statement proving the required answers
theorem winner_for_2023 (coins : ℕ) (barbara_moves : List ℕ) (jenna_moves : List ℕ) :
  coins = 2023 →  -- Specifying that the game starts with 2023 coins
  (∀n, n ∈ barbara_moves → n ≤ 2023) ∧ (∀n, n ∈ jenna_moves → n ≤ 2023) → 
  -- Specifying valid moves for both players
  ∃ winner : String, winner = "Barbara" := 
sorry

theorem winner_for_2024 (coins : ℕ) (barbara_moves : List ℕ) (jenna_moves : List ℕ) :
  coins = 2024 →  -- Specifying that the game starts with 2024 coins
  (∀n, n ∈ barbara_moves → n ≤ 2024) ∧ (∀n, n ∈ jenna_moves → n ≤ 2024) → 
  -- Specifying valid moves for both players
  ∃ winner : String, winner = "Whoever starts" :=
sorry

end winner_for_2023_winner_for_2024_l144_144475


namespace non_subset_condition_l144_144653

theorem non_subset_condition (M P : Set α) (non_empty : M ≠ ∅) : 
  ¬(M ⊆ P) ↔ ∃ x ∈ M, x ∉ P := 
sorry

end non_subset_condition_l144_144653


namespace lcm_of_three_numbers_l144_144112

theorem lcm_of_three_numbers (x : ℕ) :
  (Nat.gcd (3 * x) (Nat.gcd (4 * x) (5 * x)) = 40) →
  Nat.lcm (3 * x) (Nat.lcm (4 * x) (5 * x)) = 2400 :=
by
  sorry

end lcm_of_three_numbers_l144_144112


namespace hyperbola_properties_l144_144896

open Real

def is_asymptote (y x : ℝ) : Prop :=
  y = (1/2) * x ∨ y = -(1/2) * x

noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_properties :
  ∀ x y : ℝ,
  (x^2 / 4 - y^2 = 1) →
  ∀ (a b c : ℝ), 
  (a = 2) →
  (b = 1) →
  (c = sqrt (a^2 + b^2)) →
  (∀ y x : ℝ, (is_asymptote y x)) ∧ (eccentricity a (sqrt (a^2 + b^2)) = sqrt 5 / 2) :=
by
  intros x y h a b c ha hb hc
  sorry

end hyperbola_properties_l144_144896


namespace solution_set_inequality_l144_144499

theorem solution_set_inequality (a x : ℝ) :
  (12 * x^2 - a * x > a^2) ↔
  ((a > 0 ∧ (x < -a / 4 ∨ x > a / 3)) ∨
   (a = 0 ∧ x ≠ 0) ∨
   (a < 0 ∧ (x > -a / 4 ∨ x < a / 3))) :=
sorry


end solution_set_inequality_l144_144499


namespace brian_commission_rate_l144_144968

noncomputable def commission_rate (sale1 sale2 sale3 commission : ℝ) : ℝ :=
  (commission / (sale1 + sale2 + sale3)) * 100

theorem brian_commission_rate :
  commission_rate 157000 499000 125000 15620 = 2 :=
by
  unfold commission_rate
  sorry

end brian_commission_rate_l144_144968


namespace maximum_utilization_rate_80_l144_144019

noncomputable def maximum_utilization_rate (side_length : ℝ) (AF : ℝ) (BF : ℝ) : ℝ :=
  let area_square := side_length * side_length
  let length_rectangle := side_length
  let width_rectangle := AF / 2
  let area_rectangle := length_rectangle * width_rectangle
  (area_rectangle / area_square) * 100

theorem maximum_utilization_rate_80:
  maximum_utilization_rate 4 2 1 = 80 := by
  sorry

end maximum_utilization_rate_80_l144_144019


namespace inverse_proportion_relationship_l144_144997

variable {x1 x2 y1 y2 : ℝ}

theorem inverse_proportion_relationship (h1 : x1 < 0) (h2 : 0 < x2) 
  (hy1 : y1 = 3 / x1) (hy2 : y2 = 3 / x2) : y1 < 0 ∧ 0 < y2 :=
by
  sorry

end inverse_proportion_relationship_l144_144997


namespace sufficient_but_not_necessary_condition_l144_144458

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : a < b) : 
  ((a - b) * a^2 < 0) ↔ (a < b) :=
sorry

end sufficient_but_not_necessary_condition_l144_144458


namespace sum_all_possible_values_l144_144159

theorem sum_all_possible_values (x : ℝ) (h : x^2 = 16) :
  (x = 4 ∨ x = -4) → (4 + (-4) = 0) :=
by
  intro h1
  have : 4 + (-4) = 0 := by norm_num
  exact this

end sum_all_possible_values_l144_144159


namespace initial_bottle_caps_l144_144214

theorem initial_bottle_caps (X : ℕ) (h1 : X - 60 + 58 = 67) : X = 69 := by
  sorry

end initial_bottle_caps_l144_144214


namespace inequality_range_of_a_l144_144637

theorem inequality_range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Icc (-2: ℝ) 2 :=
by
  sorry

end inequality_range_of_a_l144_144637


namespace reflect_over_y_axis_matrix_l144_144943

theorem reflect_over_y_axis_matrix :
  ∃ M : Matrix (Fin 2) (Fin 2) ℝ, M = ![![ -1, 0], ![0, 1]] :=
  -- Proof
  sorry

end reflect_over_y_axis_matrix_l144_144943


namespace usual_time_to_school_l144_144279

theorem usual_time_to_school (R T : ℕ) (h : 7 * R * (T - 4) = 6 * R * T) : T = 28 :=
sorry

end usual_time_to_school_l144_144279


namespace sum_x_y_z_l144_144436
open Real

theorem sum_x_y_z (a b : ℝ) (h1 : a / b = 98 / 63) (x y z : ℕ) (h2 : (sqrt a) / (sqrt b) = (x * sqrt y) / z) : x + y + z = 18 := 
by
  sorry

end sum_x_y_z_l144_144436


namespace arithmetic_geometric_inequality_l144_144510

theorem arithmetic_geometric_inequality (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧ ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) :=
by
  sorry

end arithmetic_geometric_inequality_l144_144510


namespace length_after_5th_cut_l144_144644

theorem length_after_5th_cut (initial_length : ℝ) (n : ℕ) (h1 : initial_length = 1) (h2 : n = 5) :
  initial_length / 2^n = 1 / 2^5 := by
  sorry

end length_after_5th_cut_l144_144644


namespace servings_required_l144_144235

/-- Each serving of cereal is 2.0 cups, and 36 cups are needed. Prove that the number of servings required is 18. -/
theorem servings_required (cups_per_serving : ℝ) (total_cups : ℝ) (h1 : cups_per_serving = 2.0) (h2 : total_cups = 36.0) :
  total_cups / cups_per_serving = 18 :=
by
  sorry

end servings_required_l144_144235


namespace max_area_house_l144_144951

def price_colored := 450
def price_composite := 200
def cost_limit := 32000

def material_cost (x y : ℝ) : ℝ := 900 * x + 400 * y + 200 * x * y

theorem max_area_house : 
  ∃ (x y S : ℝ), 
    (S = x * y) ∧ 
    (material_cost x y ≤ cost_limit) ∧ 
    (0 < S ∧ S ≤ 100) ∧ 
    (S = 100 → x = 20 / 3) := 
by
  sorry

end max_area_house_l144_144951


namespace sum_even_numbers_from_2_to_60_l144_144341

noncomputable def sum_even_numbers_seq : ℕ :=
  let a₁ := 2
  let d := 2
  let aₙ := 60
  let n := (aₙ - a₁) / d + 1
  n / 2 * (a₁ + aₙ)

theorem sum_even_numbers_from_2_to_60:
  sum_even_numbers_seq = 930 :=
by
  sorry

end sum_even_numbers_from_2_to_60_l144_144341


namespace div_by_eight_l144_144804

theorem div_by_eight (n : ℕ) : (5^n + 2 * 3^(n-1) + 1) % 8 = 0 :=
by
  sorry

end div_by_eight_l144_144804


namespace domain_of_f_l144_144238

theorem domain_of_f (m : ℝ) : (∀ x : ℝ, mx^2 + mx + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
by
  sorry

end domain_of_f_l144_144238


namespace circle_diameter_from_area_l144_144343

theorem circle_diameter_from_area (A : ℝ) (hA : A = 400 * Real.pi) :
    ∃ D : ℝ, D = 40 := 
by
  -- Consider the formula for the area of a circle with radius r.
  -- The area is given as A = π * r^2.
  let r := Real.sqrt 400 -- Solve for radius r.
  have hr : r = 20 := by sorry
  -- The diameter D is twice the radius.
  let D := 2 * r 
  existsi D
  have hD : D = 40 := by sorry
  exact hD

end circle_diameter_from_area_l144_144343


namespace both_owners_count_l144_144692

-- Define the sets and counts as given in the conditions
variable (total_students : ℕ) (rabbit_owners : ℕ) (guinea_pig_owners : ℕ) (both_owners : ℕ)

-- Assume the values given in the problem
axiom total : total_students = 50
axiom rabbits : rabbit_owners = 35
axiom guinea_pigs : guinea_pig_owners = 40

-- The theorem to prove
theorem both_owners_count : both_owners = rabbit_owners + guinea_pig_owners - total_students := by
  sorry

end both_owners_count_l144_144692


namespace greatest_x_value_l144_144793

theorem greatest_x_value : 
  (∃ x : ℝ, 2 * x^2 + 7 * x + 3 = 5 ∧ ∀ y : ℝ, (2 * y^2 + 7 * y + 3 = 5) → y ≤ x) → x = 1 / 2 :=
by
  sorry

end greatest_x_value_l144_144793


namespace sphere_volume_proof_l144_144086

noncomputable def sphereVolume (d : ℝ) (S : ℝ) : ℝ :=
  let r := Real.sqrt (S / Real.pi)
  let R := Real.sqrt (r^2 + d^2)
  (4 / 3) * Real.pi * R^3

theorem sphere_volume_proof : sphereVolume 1 (2 * Real.pi) = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end sphere_volume_proof_l144_144086


namespace balls_to_boxes_l144_144534

theorem balls_to_boxes (balls boxes : ℕ) (h1 : balls = 5) (h2 : boxes = 3) :
  ∃ ways : ℕ, ways = 150 := by
  sorry

end balls_to_boxes_l144_144534


namespace rectangles_equal_area_implies_value_l144_144745

theorem rectangles_equal_area_implies_value (x y : ℝ) (h1 : x < 9) (h2 : y < 4)
  (h3 : x * (4 - y) = y * (9 - x)) : 360 * x / y = 810 :=
by
  -- We only need to state the theorem, the proof is not required.
  sorry

end rectangles_equal_area_implies_value_l144_144745


namespace students_enrolled_for_german_l144_144812

theorem students_enrolled_for_german 
  (total_students : ℕ)
  (both_english_german : ℕ)
  (only_english : ℕ)
  (at_least_one_subject : total_students = 32 ∧ both_english_german = 12 ∧ only_english = 10) :
  ∃ G : ℕ, G = 22 :=
by
  -- Lean proof steps will go here.
  sorry

end students_enrolled_for_german_l144_144812


namespace sum_even_integers_between_200_and_600_is_80200_l144_144679

noncomputable def sum_even_integers_between_200_and_600 (a d n : ℕ) : ℕ :=
  n / 2 * (a + (a + (n - 1) * d))

theorem sum_even_integers_between_200_and_600_is_80200 :
  sum_even_integers_between_200_and_600 202 2 200 = 80200 :=
by
  -- proof would go here
  sorry

end sum_even_integers_between_200_and_600_is_80200_l144_144679


namespace find_PA_values_l144_144083

theorem find_PA_values :
  ∃ P A : ℕ, 10 ≤ P * 10 + A ∧ P * 10 + A < 100 ∧
            (P * 10 + A) ^ 2 / 1000 = P ∧ (P * 10 + A) ^ 2 % 10 = A ∧
            ((P = 9 ∧ A = 5) ∨ (P = 9 ∧ A = 6)) := by
  sorry

end find_PA_values_l144_144083


namespace innings_count_l144_144856

-- Definitions of the problem conditions
def total_runs (n : ℕ) : ℕ := 63 * n
def highest_score : ℕ := 248
def lowest_score : ℕ := 98

theorem innings_count (n : ℕ) (h : total_runs n - highest_score - lowest_score = 58 * (n - 2)) : n = 46 :=
  sorry

end innings_count_l144_144856


namespace slip_2_5_in_A_or_C_l144_144870

-- Define the slips and their values
def slips : List ℚ := [1, 1.5, 2, 2, 2.5, 3, 3, 3.5, 3.5, 4, 4.5, 4.5, 5, 5.5, 6]

-- Define the cups
inductive Cup
| A | B | C | D | E | F

open Cup

-- Define the given cups constraints
def sum_constraints : Cup → ℚ
| A => 6
| B => 7
| C => 8
| D => 9
| E => 10
| F => 10

-- Initial conditions for slips placement
def slips_in_cups (c : Cup) : List ℚ :=
match c with
| F => [1.5]
| B => [4]
| _ => []

-- We'd like to prove that:
def slip_2_5_can_go_into : Prop :=
  (slips_in_cups A = [2.5] ∧ slips_in_cups C = [2.5])

theorem slip_2_5_in_A_or_C : slip_2_5_can_go_into :=
sorry

end slip_2_5_in_A_or_C_l144_144870


namespace parabola_coeff_sum_l144_144572

def parabola_vertex_form (a b c : ℚ) : Prop :=
  (∀ y : ℚ, y = 2 → (-3) = a * (y - 2)^2 + b * (y - 2) + c) ∧
  (∀ x y : ℚ, x = 1 ∧ y = -1 → x = a * y^2 + b * y + c) ∧
  (a < 0)  -- Since the parabola opens to the left, implying the coefficient 'a' is positive.

theorem parabola_coeff_sum (a b c : ℚ) :
  parabola_vertex_form a b c → a + b + c = -23 / 9 :=
by
  sorry

end parabola_coeff_sum_l144_144572


namespace average_of_primes_less_than_twenty_l144_144826

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]
def sum_primes : ℕ := 77
def count_primes : ℕ := 8
def average_primes : ℚ := 77 / 8

theorem average_of_primes_less_than_twenty : (primes_less_than_twenty.sum / count_primes : ℚ) = 9.625 := by
  sorry

end average_of_primes_less_than_twenty_l144_144826


namespace solve_for_nabla_l144_144887

theorem solve_for_nabla (nabla : ℤ) (h : 4 * (-3) = nabla + 3) : nabla = -15 :=
by
  sorry

end solve_for_nabla_l144_144887


namespace max_c_value_for_f_x_range_l144_144348

theorem max_c_value_for_f_x_range:
  (∀ c : ℝ, (∃ x : ℝ, x^2 + 4 * x + c = -2) → c ≤ 2) ∧ (∃ (x : ℝ), x^2 + 4 * x + 2 = -2) :=
sorry

end max_c_value_for_f_x_range_l144_144348


namespace stadium_length_l144_144473

theorem stadium_length
  (W : ℝ) (H : ℝ) (P : ℝ) (L : ℝ)
  (h1 : W = 18)
  (h2 : H = 16)
  (h3 : P = 34)
  (h4 : P^2 = L^2 + W^2 + H^2) :
  L = 24 :=
by
  sorry

end stadium_length_l144_144473


namespace find_base_l144_144969

theorem find_base (b : ℕ) (h : (3 * b + 5) ^ 2 = b ^ 3 + 3 * b ^ 2 + 2 * b + 5) : b = 7 :=
sorry

end find_base_l144_144969


namespace geom_seq_a4_l144_144005

theorem geom_seq_a4 (a : ℕ → ℝ) (r : ℝ)
  (h : ∀ n, a (n + 1) = a n * r)
  (h2 : a 3 = 9)
  (h3 : a 5 = 1) :
  a 4 = 3 ∨ a 4 = -3 :=
by {
  sorry
}

end geom_seq_a4_l144_144005


namespace midpoint_distance_l144_144455

theorem midpoint_distance (a b c d : ℝ) :
  let m := (a + c) / 2
  let n := (b + d) / 2
  let m' := m - 0.5
  let n' := n - 0.5
  dist (m, n) (m', n') = (Real.sqrt 2) / 2 := 
by 
  sorry

end midpoint_distance_l144_144455


namespace find_principal_l144_144067

theorem find_principal (SI : ℝ) (R : ℝ) (T : ℝ) (hSI : SI = 4025.25) (hR : R = 9) (hT : T = 5) :
    let P := SI / (R * T / 100)
    P = 8950 :=
by
  -- we will put proof steps here
  sorry

end find_principal_l144_144067


namespace curve_intersects_self_at_6_6_l144_144869

-- Definitions for the given conditions
def x (t : ℝ) : ℝ := t^2 - 3
def y (t : ℝ) : ℝ := t^4 - t^2 - 9 * t + 6

-- Lean statement stating that the curve intersects itself at the coordinate (6, 6)
theorem curve_intersects_self_at_6_6 :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ x t1 = x t2 ∧ y t1 = y t2 ∧ x t1 = 6 ∧ y t1 = 6 :=
by
  sorry

end curve_intersects_self_at_6_6_l144_144869


namespace product_of_repeating_decimal_l144_144127

-- Define the repeating decimal 0.3
def repeating_decimal : ℚ := 1 / 3
-- Define the question
def product (a b : ℚ) := a * b

-- State the theorem to be proved
theorem product_of_repeating_decimal :
  product repeating_decimal 8 = 8 / 3 :=
sorry

end product_of_repeating_decimal_l144_144127


namespace polynomial_factorization_l144_144673

theorem polynomial_factorization : 
  (x : ℤ) → (x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1)) := 
by
  sorry

end polynomial_factorization_l144_144673


namespace ratio_lt_one_l144_144223

def product_sequence (k j : ℕ) := List.prod (List.range' k j)

theorem ratio_lt_one :
  let a := product_sequence 2020 4
  let b := product_sequence 2120 4
  a / b < 1 :=
by
  sorry

end ratio_lt_one_l144_144223


namespace sufficient_but_not_necessary_condition_l144_144230

theorem sufficient_but_not_necessary_condition (f : ℝ → ℝ) (h : ∀ x, f x = x⁻¹) :
  ∀ x, (x > 1 → f (x + 2) > f (2*x + 1)) ∧ (¬ (x > 1) → ¬ (f (x + 2) > f (2*x + 1))) :=
by
  sorry

end sufficient_but_not_necessary_condition_l144_144230


namespace evaluate_expression_at_3_l144_144605

-- Define the expression
def expression (x : ℕ) : ℕ := x^2 - 3*x + 2

-- Statement of the problem
theorem evaluate_expression_at_3 : expression 3 = 2 := by
    sorry -- Proof is omitted

end evaluate_expression_at_3_l144_144605


namespace speed_of_water_current_l144_144916

theorem speed_of_water_current (v : ℝ) 
  (swimmer_speed_still_water : ℝ := 4) 
  (distance : ℝ := 3) 
  (time : ℝ := 1.5)
  (effective_speed_against_current : ℝ := swimmer_speed_still_water - v) :
  effective_speed_against_current = distance / time → v = 2 := 
by
  -- Proof
  sorry

end speed_of_water_current_l144_144916


namespace solve_for_x_l144_144504

noncomputable def n : ℝ := Real.sqrt (7^2 + 24^2)
noncomputable def d : ℝ := Real.sqrt (49 + 16)
noncomputable def x : ℝ := n / d

theorem solve_for_x : x = 5 * Real.sqrt 65 / 13 := by
  sorry

end solve_for_x_l144_144504


namespace initial_people_count_l144_144239

theorem initial_people_count (left remaining total : ℕ) (h1 : left = 6) (h2 : remaining = 5) : total = 11 :=
  by
  sorry

end initial_people_count_l144_144239


namespace first_day_exceeding_100_paperclips_l144_144366

def paperclips_day (k : ℕ) : ℕ := 3 * 2^k

theorem first_day_exceeding_100_paperclips :
  ∃ (k : ℕ), paperclips_day k > 100 ∧ k = 6 := by
  sorry

end first_day_exceeding_100_paperclips_l144_144366


namespace work_together_days_l144_144885

theorem work_together_days
  (a_days : ℝ) (ha : a_days = 18)
  (b_days : ℝ) (hb : b_days = 30)
  (c_days : ℝ) (hc : c_days = 45)
  (combined_days : ℝ) :
  (combined_days = 1 / ((1 / a_days) + (1 / b_days) + (1 / c_days))) → combined_days = 9 := 
by
  sorry

end work_together_days_l144_144885


namespace kim_boxes_sold_on_tuesday_l144_144278

theorem kim_boxes_sold_on_tuesday :
  ∀ (T W Th F : ℕ),
  (T = 3 * W) →
  (W = 2 * Th) →
  (Th = 3 / 2 * F) →
  (F = 600) →
  T = 5400 :=
by
  intros T W Th F h1 h2 h3 h4
  sorry

end kim_boxes_sold_on_tuesday_l144_144278


namespace martha_clothes_total_l144_144168

-- Given conditions
def jackets_bought : Nat := 4
def t_shirts_bought : Nat := 9
def free_jacket_ratio : Nat := 2
def free_t_shirt_ratio : Nat := 3

-- Problem statement to prove
theorem martha_clothes_total :
  (jackets_bought + jackets_bought / free_jacket_ratio) + 
  (t_shirts_bought + t_shirts_bought / free_t_shirt_ratio) = 18 := 
by 
  sorry

end martha_clothes_total_l144_144168


namespace total_goals_l144_144743

-- Define constants for goals scored in respective seasons
def goalsLastSeason : ℕ := 156
def goalsThisSeason : ℕ := 187

-- Define the theorem for the total number of goals
theorem total_goals : goalsLastSeason + goalsThisSeason = 343 :=
by
  -- Proof is omitted
  sorry

end total_goals_l144_144743


namespace chemical_x_percentage_l144_144828

-- Define the initial volume of the mixture
def initial_volume : ℕ := 80

-- Define the percentage of chemical x in the initial mixture
def percentage_x_initial : ℚ := 0.30

-- Define the volume of chemical x added to the mixture
def added_volume_x : ℕ := 20

-- Define the calculation of the amount of chemical x in the initial mixture
def initial_amount_x : ℚ := percentage_x_initial * initial_volume

-- Define the calculation of the total amount of chemical x after adding more
def total_amount_x : ℚ := initial_amount_x + added_volume_x

-- Define the calculation of the total volume after adding 20 liters of chemical x
def total_volume : ℚ := initial_volume + added_volume_x

-- Define the percentage of chemical x in the final mixture
def percentage_x_final : ℚ := (total_amount_x / total_volume) * 100

-- The proof goal
theorem chemical_x_percentage : percentage_x_final = 44 := 
by
  sorry

end chemical_x_percentage_l144_144828


namespace initial_passengers_l144_144528

theorem initial_passengers (P : ℝ) :
  (1/2 * (2/3 * P + 280) + 12 = 242) → P = 270 :=
by
  sorry

end initial_passengers_l144_144528


namespace power_modulo_l144_144539

theorem power_modulo (a b c n : ℕ) (h1 : a = 17) (h2 : b = 1999) (h3 : c = 29) (h4 : n = a^b % c) : 
  n = 17 := 
by
  -- Note: Additional assumptions and intermediate calculations could be provided as needed
  sorry

end power_modulo_l144_144539


namespace triangle_area_l144_144346

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem triangle_area (a b c : ℕ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) (h4 : is_right_triangle a b c) :
  (1 / 2 : ℝ) * a * b = 180 :=
by sorry

end triangle_area_l144_144346


namespace total_valid_votes_l144_144379

theorem total_valid_votes (V : ℝ) (h1 : 0.70 * V - 0.30 * V = 176) : V = 440 :=
by sorry

end total_valid_votes_l144_144379


namespace no_integers_satisfy_l144_144628

def P (x a b c d : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_integers_satisfy :
  ∀ a b c d : ℤ, ¬ (P 19 a b c d = 1 ∧ P 62 a b c d = 2) :=
by
  intro a b c d
  sorry

end no_integers_satisfy_l144_144628


namespace fisherman_caught_total_fish_l144_144359

noncomputable def number_of_boxes : ℕ := 15
noncomputable def fish_per_box : ℕ := 20
noncomputable def fish_outside_boxes : ℕ := 6

theorem fisherman_caught_total_fish :
  number_of_boxes * fish_per_box + fish_outside_boxes = 306 :=
by
  sorry

end fisherman_caught_total_fish_l144_144359


namespace not_right_triangle_l144_144808

theorem not_right_triangle (A B C : ℝ) (hA : A + B = 180 - C) 
  (hB : A = B / 2 ∧ A = C / 3) 
  (hC : A = B / 2 ∧ B = C / 1.5) 
  (hD : A = 2 * B ∧ A = 3 * C):
  (C ≠ 90) :=
by {
  sorry
}

end not_right_triangle_l144_144808


namespace no_positive_integer_solutions_l144_144027

theorem no_positive_integer_solutions (m : ℕ) (h_pos : m > 0) :
  ¬ ∃ x : ℚ, m * x^2 + 40 * x + m = 0 :=
by {
  -- the proof goes here
  sorry
}

end no_positive_integer_solutions_l144_144027


namespace function_identity_l144_144109

-- Definitions of the problem
def f (n : ℕ) : ℕ := sorry

-- Main theorem to prove
theorem function_identity (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, m > 0 → n > 0 → f (m + n) * f (m - n) = f (m * m)) : 
  ∀ n : ℕ, n > 0 → f n = 1 := 
sorry

end function_identity_l144_144109


namespace handshaking_remainder_l144_144318

noncomputable def num_handshaking_arrangements_modulo (n : ℕ) : ℕ := sorry

theorem handshaking_remainder (N : ℕ) (h : num_handshaking_arrangements_modulo 9 = N) :
  N % 1000 = 16 :=
sorry

end handshaking_remainder_l144_144318


namespace no_real_solutions_l144_144676

-- Define the equation
def equation (x : ℝ) : Prop :=
  (2 * x ^ 2 - 6 * x + 5) ^ 2 + 1 = -|x|

-- Declare the theorem which states there are no real solutions to the given equation
theorem no_real_solutions : ∀ x : ℝ, ¬ equation x :=
by
  intro x
  sorry

end no_real_solutions_l144_144676


namespace cake_pieces_l144_144957

theorem cake_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_length : ℕ) (piece_width : ℕ) 
  (pan_dim : pan_length = 24 ∧ pan_width = 15) 
  (piece_dim : piece_length = 3 ∧ piece_width = 2) : 
  (pan_length * pan_width) / (piece_length * piece_width) = 60 :=
sorry

end cake_pieces_l144_144957


namespace num_divisors_360_l144_144298

theorem num_divisors_360 :
  ∀ n : ℕ, n = 360 → (∀ (p q r : ℕ), p = 2 ∧ q = 3 ∧ r = 5 →
    (∃ (a b c : ℕ), 360 = p^a * q^b * r^c ∧ a = 3 ∧ b = 2 ∧ c = 1) →
    (3+1) * (2+1) * (1+1) = 24) :=
  sorry

end num_divisors_360_l144_144298


namespace smallest_integer_in_set_l144_144889

theorem smallest_integer_in_set : 
  ∀ (n : ℤ), (n + 6 < 2 * (n + 3)) → n ≥ 1 :=
by 
  sorry

end smallest_integer_in_set_l144_144889


namespace jameson_badminton_medals_l144_144659

theorem jameson_badminton_medals (total_medals track_medals : ℕ) (swimming_medals : ℕ) :
  total_medals = 20 →
  track_medals = 5 →
  swimming_medals = 2 * track_medals →
  total_medals - (track_medals + swimming_medals) = 5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end jameson_badminton_medals_l144_144659


namespace average_weight_estimate_l144_144722

noncomputable def average_weight (female_students male_students : ℕ) (avg_weight_female avg_weight_male : ℕ) : ℝ :=
  (female_students / (female_students + male_students) : ℝ) * avg_weight_female +
  (male_students / (female_students + male_students) : ℝ) * avg_weight_male

theorem average_weight_estimate :
  average_weight 504 596 49 57 = (504 / 1100 : ℝ) * 49 + (596 / 1100 : ℝ) * 57 :=
by
  sorry

end average_weight_estimate_l144_144722


namespace sqrt_16_l144_144710

theorem sqrt_16 : {x : ℝ | x^2 = 16} = {4, -4} :=
by
  sorry

end sqrt_16_l144_144710


namespace average_mark_of_remaining_students_l144_144682

theorem average_mark_of_remaining_students
  (n : ℕ) (A : ℕ) (m : ℕ) (B : ℕ) (total_students : n = 10)
  (avg_class : A = 80) (excluded_students : m = 5) (avg_excluded : B = 70) :
  (A * n - B * m) / (n - m) = 90 :=
by
  sorry

end average_mark_of_remaining_students_l144_144682


namespace chloe_points_first_round_l144_144977

theorem chloe_points_first_round 
  (P : ℕ)
  (second_round_points : ℕ := 50)
  (lost_points : ℕ := 4)
  (total_points : ℕ := 86)
  (h : P + second_round_points - lost_points = total_points) : 
  P = 40 := 
by 
  sorry

end chloe_points_first_round_l144_144977


namespace solve_quadratic_l144_144970

   theorem solve_quadratic (x : ℝ) (h1 : x > 0) (h2 : 5 * x^2 + 8 * x - 24 = 0) : x = 6 / 5 :=
   sorry
   
end solve_quadratic_l144_144970


namespace point_B_value_l144_144222

theorem point_B_value :
  ∃ B : ℝ, (|B + 1| = 4) ∧ (B = 3 ∨ B = -5) := 
by
  sorry

end point_B_value_l144_144222


namespace main_l144_144357

def M (x : ℝ) : Prop := x^2 - 5 * x ≤ 0
def N (x : ℝ) (p : ℝ) : Prop := p < x ∧ x < 6
def intersection (x : ℝ) (q : ℝ) : Prop := 2 < x ∧ x ≤ q

theorem main (p q : ℝ) (hM : ∀ x, M x → 0 ≤ x ∧ x ≤ 5) (hN : ∀ x, N x p → p < x ∧ x < 6) (hMN : ∀ x, (M x ∧ N x p) ↔ intersection x q) :
  p + q = 7 :=
by
  sorry

end main_l144_144357


namespace henrietta_paint_gallons_l144_144758

-- Define the conditions
def living_room_area : Nat := 600
def bedrooms_count : Nat := 3
def bedroom_area : Nat := 400
def coverage_per_gallon : Nat := 600

-- The theorem we want to prove
theorem henrietta_paint_gallons :
  (bedrooms_count * bedroom_area + living_room_area) / coverage_per_gallon = 3 :=
by
  sorry

end henrietta_paint_gallons_l144_144758


namespace distance_3D_l144_144052

theorem distance_3D : 
  let A := (2, -2, 0)
  let B := (8, 8, 3)
  let d := Real.sqrt ((8 - 2) ^ 2 + (8 - (-2)) ^ 2 + (3 - 0) ^ 2)
  d = Real.sqrt 145 :=
by
  let A := (2, -2, 0)
  let B := (8, 8, 3)
  let d := Real.sqrt ((8 - 2) ^ 2 + (8 - (-2)) ^ 2 + (3 - 0) ^ 2)
  dsimp [A, B, d]
  sorry

end distance_3D_l144_144052


namespace uncle_bradley_money_l144_144129

-- Definitions of the variables and conditions
variables (F H M : ℝ)
variables (h1 : F + H = 13)
variables (h2 : 50 * F = (3 / 10) * M)
variables (h3 : 100 * H = (7 / 10) * M)

-- The theorem statement
theorem uncle_bradley_money : M = 1300 :=
by
  sorry

end uncle_bradley_money_l144_144129


namespace length_of_crate_l144_144772

theorem length_of_crate (h crate_dim : ℕ) (radius : ℕ) (h_radius : radius = 8) 
  (h_dims : crate_dim = 18) (h_fit : 2 * radius = 16)
  : h = 18 := 
sorry

end length_of_crate_l144_144772


namespace boats_solution_l144_144780

theorem boats_solution (x y : ℕ) (h1 : x + y = 42) (h2 : 6 * x = 8 * y) : x = 24 ∧ y = 18 :=
by
  sorry

end boats_solution_l144_144780


namespace cone_base_circumference_l144_144547

-- Definitions of the problem
def radius : ℝ := 5
def angle_sector_degree : ℝ := 120
def full_circle_degree : ℝ := 360

-- Proof statement
theorem cone_base_circumference 
  (r : ℝ) (angle_sector : ℝ) (full_angle : ℝ) 
  (h1 : r = radius) 
  (h2 : angle_sector = angle_sector_degree) 
  (h3 : full_angle = full_circle_degree) : 
  (angle_sector / full_angle) * (2 * π * r) = (10 * π) / 3 := 
by sorry

end cone_base_circumference_l144_144547


namespace solve_for_x_l144_144426

theorem solve_for_x (x : ℝ) (h : (x / 5) + 3 = 4) : x = 5 :=
by
  sorry

end solve_for_x_l144_144426


namespace custom_op_example_l144_144096

def custom_op (a b : ℤ) : ℤ := a + 2 * b^2

theorem custom_op_example : custom_op (-4) 6 = 68 :=
by
  sorry

end custom_op_example_l144_144096


namespace triangle_inequality_circumradius_l144_144364

theorem triangle_inequality_circumradius (a b c R : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) 
  (circumradius_def : R = (a * b * c) / (4 * (Real.sqrt ((a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c))))) :
  (1 / (a * b)) + (1 / (b * c)) + (1 / (c * a)) ≥ (1 / (R ^ 2)) :=
sorry

end triangle_inequality_circumradius_l144_144364


namespace find_x_plus_y_l144_144688

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 2010) (h2 : x + 2010 * Real.sin y = 2009) (hy : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 :=
by
  sorry

end find_x_plus_y_l144_144688


namespace hardcover_volumes_l144_144489

theorem hardcover_volumes (h p : ℕ) (h_condition : h + p = 12) (cost_condition : 25 * h + 15 * p = 240) : h = 6 :=
by
  -- omitted proof steps for brevity
  sorry

end hardcover_volumes_l144_144489


namespace fraction_power_computation_l144_144123

theorem fraction_power_computation : (5 / 6) ^ 4 = 625 / 1296 :=
by
  -- Normally we'd provide the proof here, but it's omitted as per instructions
  sorry

end fraction_power_computation_l144_144123


namespace triangle_area_from_curve_l144_144749

-- Definition of the curve
def curve (x : ℝ) : ℝ := (x - 2)^2 * (x + 3)

-- Area calculation based on intercepts
theorem triangle_area_from_curve : 
  (1 / 2) * (2 - (-3)) * (curve 0) = 30 :=
by
  sorry

end triangle_area_from_curve_l144_144749


namespace riley_outside_fraction_l144_144631

theorem riley_outside_fraction
  (awake_jonsey : ℚ := 2 / 3)
  (jonsey_outside_fraction : ℚ := 1 / 2)
  (awake_riley : ℚ := 3 / 4)
  (total_inside_time : ℚ := 10)
  (hours_per_day : ℕ := 24) :
  let jonsey_inside_time := 1 / 3 * hours_per_day
  let riley_inside_time := (1 - (8 / 9)) * (3 / 4) * hours_per_day
  jonsey_inside_time + riley_inside_time = total_inside_time :=
by
  sorry

end riley_outside_fraction_l144_144631


namespace gcd_lcm_divisible_l144_144360

theorem gcd_lcm_divisible (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b + Nat.lcm a b = a + b) : a % b = 0 ∨ b % a = 0 := 
sorry

end gcd_lcm_divisible_l144_144360


namespace count_two_digit_integers_sum_seven_l144_144748

theorem count_two_digit_integers_sum_seven : 
  ∃ n : ℕ, (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a + b = 7 → n = 7) := 
by
  sorry

end count_two_digit_integers_sum_seven_l144_144748


namespace no_triangle_formed_l144_144540

def line1 (x y : ℝ) := 2 * x - 3 * y + 1 = 0
def line2 (x y : ℝ) := 4 * x + 3 * y + 5 = 0
def line3 (m : ℝ) (x y : ℝ) := m * x - y - 1 = 0

theorem no_triangle_formed (m : ℝ) :
  (∀ x y, line1 x y → line3 m x y) ∨
  (∀ x y, line2 x y → line3 m x y) ∨
  (∃ x y, line1 x y ∧ line2 x y ∧ line3 m x y) ↔
  (m = -4/3 ∨ m = 2/3 ∨ m = 4/3) :=
sorry -- Proof to be provided

end no_triangle_formed_l144_144540


namespace solve_equation_l144_144621

theorem solve_equation (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end solve_equation_l144_144621


namespace probability_of_seeing_red_light_l144_144237

def red_light_duration : ℝ := 30
def yellow_light_duration : ℝ := 5
def green_light_duration : ℝ := 40

def total_cycle_duration : ℝ := red_light_duration + yellow_light_duration + green_light_duration

theorem probability_of_seeing_red_light :
  (red_light_duration / total_cycle_duration) = 30 / 75 := by
  sorry

end probability_of_seeing_red_light_l144_144237


namespace radius_of_circle_tangent_to_xaxis_l144_144077

theorem radius_of_circle_tangent_to_xaxis
  (Ω : Set (ℝ × ℝ)) (Γ : Set (ℝ × ℝ))
  (hΓ : ∀ x y : ℝ, (x, y) ∈ Γ ↔ y^2 = 4 * x)
  (F : ℝ × ℝ) (hF : F = (1, 0))
  (hΩ_tangent : ∃ r : ℝ, ∀ x y : ℝ, (x - 1)^2 + (y - r)^2 = r^2 ∧ (1, 0) ∈ Ω)
  (hΩ_intersect : ∀ x y : ℝ, (x, y) ∈ Ω → (x, y) ∈ Γ → (x, y) = (1, 0)) :
  ∃ r : ℝ, r = 4 * Real.sqrt 3 / 9 :=
sorry

end radius_of_circle_tangent_to_xaxis_l144_144077


namespace simplify_expression_l144_144795

variables {a b : ℝ}

theorem simplify_expression (a b : ℝ) : (2 * a^2 * b)^3 = 8 * a^6 * b^3 := 
by
  sorry

end simplify_expression_l144_144795


namespace candidate_final_score_l144_144928

/- Given conditions -/
def interview_score : ℤ := 80
def written_test_score : ℤ := 90
def interview_weight : ℤ := 3
def written_test_weight : ℤ := 2

/- Final score computation -/
noncomputable def final_score : ℤ :=
  (interview_score * interview_weight + written_test_score * written_test_weight) / (interview_weight + written_test_weight)

theorem candidate_final_score : final_score = 84 := 
by
  sorry

end candidate_final_score_l144_144928


namespace sequence_value_l144_144217

theorem sequence_value (a : ℕ → ℕ) (h₁ : ∀ n, a (2 * n) = a (2 * n - 1) + (-1 : ℤ)^n) 
                        (h₂ : ∀ n, a (2 * n + 1) = a (2 * n) + n)
                        (h₃ : a 1 = 1) : a 20 = 46 :=
by 
  sorry

end sequence_value_l144_144217


namespace cost_of_paper_l144_144837

noncomputable def cost_of_paper_per_kg (edge_length : ℕ) (coverage_per_kg : ℕ) (expenditure : ℕ) : ℕ :=
  let surface_area := 6 * edge_length * edge_length
  let paper_needed := surface_area / coverage_per_kg
  expenditure / paper_needed

theorem cost_of_paper (h1 : edge_length = 10) (h2 : coverage_per_kg = 20) (h3 : expenditure = 1800) : 
  cost_of_paper_per_kg 10 20 1800 = 60 :=
by
  -- Using the hypothesis to directly derive the result.
  unfold cost_of_paper_per_kg
  sorry

end cost_of_paper_l144_144837


namespace trigonometric_identity_l144_144273

theorem trigonometric_identity : 
  let sin := Real.sin
  let cos := Real.cos
  sin 18 * cos 63 - sin 72 * sin 117 = - (Real.sqrt 2 / 2) :=
by
  -- The proof would go here
  sorry

end trigonometric_identity_l144_144273


namespace time_for_A_l144_144289

noncomputable def work_days (A B C D E : ℝ) : Prop :=
  (1/A + 1/B + 1/C + 1/D = 1/8) ∧
  (1/B + 1/C + 1/D + 1/E = 1/6) ∧
  (1/A + 1/E = 1/12)

theorem time_for_A (A B C D E : ℝ) (h : work_days A B C D E) : A = 48 :=
  by
    sorry

end time_for_A_l144_144289


namespace polynomial_negativity_l144_144996

theorem polynomial_negativity (a x : ℝ) (h₀ : 0 < x) (h₁ : x < a) (h₂ : 0 < a) : 
  (a - x)^6 - 3 * a * (a - x)^5 + (5 / 2) * a^2 * (a - x)^4 - (1 / 2) * a^4 * (a - x)^2 < 0 := 
by
  sorry

end polynomial_negativity_l144_144996


namespace find_number_of_flowers_l144_144434
open Nat

theorem find_number_of_flowers (F : ℕ) (h_candles : choose 4 2 = 6) (h_groupings : 6 * choose F 8 = 54) : F = 9 :=
sorry

end find_number_of_flowers_l144_144434


namespace x_divisible_by_5_l144_144069

theorem x_divisible_by_5 (x y : ℕ) (hx : x > 1) (h : 2 * x^2 - 1 = y^15) : 5 ∣ x := 
sorry

end x_divisible_by_5_l144_144069


namespace joan_balloons_l144_144310

theorem joan_balloons (m t j : ℕ) (h1 : m = 41) (h2 : t = 81) : j = t - m → j = 40 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

end joan_balloons_l144_144310


namespace least_number_to_add_l144_144606

theorem least_number_to_add (n : ℕ) : (3457 + n) % 103 = 0 ↔ n = 45 :=
by sorry

end least_number_to_add_l144_144606


namespace problem_statement_l144_144079

noncomputable def α : ℝ := 3 + Real.sqrt 8
noncomputable def β : ℝ := 3 - Real.sqrt 8
noncomputable def x : ℝ := α ^ 500
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := 
by
  sorry

end problem_statement_l144_144079


namespace perfect_square_trinomial_coeff_l144_144612

theorem perfect_square_trinomial_coeff (m : ℝ) : (∃ a b : ℝ, (a ≠ 0) ∧ ((a * x + b)^2 = x^2 - m * x + 25)) ↔ (m = 10 ∨ m = -10) :=
by sorry

end perfect_square_trinomial_coeff_l144_144612


namespace find_A_from_complement_l144_144961

-- Define the universal set U
def U : Set ℕ := {0, 1, 2}

-- Define the complement of set A in U
variable (A : Set ℕ)
def complement_U_A : Set ℕ := {n | n ∈ U ∧ n ∉ A}

-- Define the condition given in the problem
axiom h : complement_U_A A = {2}

-- State the theorem to be proven
theorem find_A_from_complement : A = {0, 1} :=
sorry

end find_A_from_complement_l144_144961


namespace sum_of_first_50_digits_of_one_over_1234_l144_144114

def first_n_digits_sum (x : ℚ) (n : ℕ) : ℕ :=
  sorry  -- This function should compute the sum of the first n digits after the decimal point of x

theorem sum_of_first_50_digits_of_one_over_1234 :
  first_n_digits_sum (1/1234) 50 = 275 :=
sorry

end sum_of_first_50_digits_of_one_over_1234_l144_144114


namespace problem_l144_144312

theorem problem (h : ℤ) : (∃ x : ℤ, x = -2 ∧ x^3 + h * x - 12 = 0) → h = -10 := by
  sorry

end problem_l144_144312


namespace find_N_l144_144152

def consecutive_product_sum_condition (a : ℕ) : Prop :=
  a*(a + 1)*(a + 2) = 8*(a + (a + 1) + (a + 2))

theorem find_N : ∃ (N : ℕ), N = 120 ∧ ∃ (a : ℕ), a > 0 ∧ consecutive_product_sum_condition a := by
  sorry

end find_N_l144_144152


namespace solution_for_system_of_inequalities_l144_144336

theorem solution_for_system_of_inequalities (p : ℝ) : 
  (19 * p < 10) ∧ (p > 0.5) ↔ (0.5 < p ∧ p < 10/19) := 
by {
  sorry
}

end solution_for_system_of_inequalities_l144_144336


namespace simplify_fraction_l144_144545

theorem simplify_fraction : 
  (1 - (1 / 4)) / (1 - (1 / 3)) = 9 / 8 :=
by
  sorry

end simplify_fraction_l144_144545


namespace composite_function_increasing_l144_144830

variable {F : ℝ → ℝ}

/-- An odd function is a function that satisfies f(-x) = -f(x) for all x. -/
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is strictly increasing on negative values if it satisfies the given conditions. -/
def strictly_increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2, x1 < x2 → x2 < 0 → f x1 < f x2

/-- Combining properties of an odd function and strictly increasing for negative inputs:
  We need to prove that the composite function is strictly increasing for positive inputs. -/
theorem composite_function_increasing (hf_odd : odd_function F)
    (hf_strict_inc_neg : strictly_increasing_on_neg F)
    : ∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → F (F x1) < F (F x2) :=
  sorry

end composite_function_increasing_l144_144830


namespace part1_part2_l144_144923

def P (a : ℝ) := ∀ x : ℝ, x^2 - a * x + a + 5 / 4 > 0
def Q (a : ℝ) := 4 * a + 7 ≠ 0 ∧ a - 3 ≠ 0 ∧ (4 * a + 7) * (a - 3) < 0

theorem part1 (h : Q a) : -7 / 4 < a ∧ a < 3 := sorry

theorem part2 (h : (P a ∨ Q a) ∧ ¬(P a ∧ Q a)) :
  (-7 / 4 < a ∧ a ≤ -1) ∨ (3 ≤ a ∧ a < 5) := sorry

end part1_part2_l144_144923


namespace area_of_smallest_square_that_encloses_circle_l144_144041

def radius : ℕ := 5

def diameter (r : ℕ) : ℕ := 2 * r

def side_length (d : ℕ) : ℕ := d

def area_of_square (s : ℕ) : ℕ := s * s

theorem area_of_smallest_square_that_encloses_circle :
  area_of_square (side_length (diameter radius)) = 100 := by
  sorry

end area_of_smallest_square_that_encloses_circle_l144_144041


namespace perpendicular_condition_l144_144340

def line1 (a : ℝ) (x y : ℝ) : ℝ := a * x + 2 * y - 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := 3 * x - a * y + 1

def perpendicular_lines (a : ℝ) : Prop := 
  ∀ (x y : ℝ), line1 a x y = 0 → line2 a x y = 0 → 3 * a - 2 * a = 0 

theorem perpendicular_condition (a : ℝ) (h : perpendicular_lines a) : a = 0 := sorry

end perpendicular_condition_l144_144340


namespace charges_equal_at_x_4_cost_effectiveness_l144_144624

-- Defining the conditions
def full_price : ℕ := 240

def yA (x : ℕ) : ℕ := 120 * x + 240
def yB (x : ℕ) : ℕ := 144 * x + 144

-- (Ⅰ) Establishing the expressions for the charges is already encapsulated in the definitions.

-- (Ⅱ) Proving the equivalence of the two charges for a specific number of students x.
theorem charges_equal_at_x_4 : ∀ x : ℕ, yA x = yB x ↔ x = 4 := 
by {
  sorry
}

-- (Ⅲ) Discussing which travel agency is more cost-effective based on the number of students x.
theorem cost_effectiveness (x : ℕ) :
  (x < 4 → yA x > yB x) ∧ (x > 4 → yA x < yB x) :=
by {
  sorry
}

end charges_equal_at_x_4_cost_effectiveness_l144_144624


namespace common_tangents_count_l144_144026

def circleC1 : Prop := ∃ (x y : ℝ), x^2 + y^2 + 2 * x - 6 * y - 15 = 0
def circleC2 : Prop := ∃ (x y : ℝ), x^2 + y^2 - 4 * x + 2 * y + 4 = 0

theorem common_tangents_count (C1 : circleC1) (C2 : circleC2) : 
  ∃ (n : ℕ), n = 2 := 
sorry

end common_tangents_count_l144_144026


namespace elder_three_times_younger_l144_144719

-- Definitions based on conditions
def age_difference := 16
def elder_present_age := 30
def younger_present_age := elder_present_age - age_difference

-- The problem statement to prove the correct value of n (years ago)
theorem elder_three_times_younger (n : ℕ) 
  (h1 : elder_present_age = younger_present_age + age_difference)
  (h2 : elder_present_age - n = 3 * (younger_present_age - n)) : 
  n = 6 := 
sorry

end elder_three_times_younger_l144_144719


namespace f_96_l144_144763

noncomputable def f : ℕ → ℝ := sorry -- assume f is defined somewhere

axiom f_property (a b k : ℕ) (h : a + b = 3 * 2^k) : f a + f b = 2 * k^2

theorem f_96 : f 96 = 20 :=
by
  -- Here we should provide the proof, but for now we use sorry
  sorry

end f_96_l144_144763


namespace f_zero_f_odd_f_range_l144_144075

noncomputable def f : ℝ → ℝ := sorry

-- Add the hypothesis for the conditions
axiom f_domain : ∀ x : ℝ, ∃ y : ℝ, f y = f x
axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_value_one_third : f (1 / 3) = 1
axiom f_positive : ∀ x : ℝ, x > 0 → f x > 0

-- (1) Prove that f(0) = 0
theorem f_zero : f 0 = 0 := sorry

-- (2) Prove that f(x) is odd
theorem f_odd (x : ℝ) : f (-x) = -f x := sorry

-- (3) Given f(x) + f(2 + x) < 2, find the range of x
theorem f_range (x : ℝ) : f x + f (2 + x) < 2 → x < -2 / 3 := sorry

end f_zero_f_odd_f_range_l144_144075


namespace total_number_of_employees_l144_144802

theorem total_number_of_employees (n : ℕ) (hm : ℕ) (hd : ℕ) 
  (h_ratio : 4 * hd = hm)
  (h_diff : hm = hd + 72) : n = 120 :=
by
  -- proof steps would go here
  sorry

end total_number_of_employees_l144_144802


namespace units_digit_k_squared_plus_three_to_the_k_mod_10_l144_144723

def k := 2025^2 + 3^2025

theorem units_digit_k_squared_plus_three_to_the_k_mod_10 : 
  (k^2 + 3^k) % 10 = 5 := by
sorry

end units_digit_k_squared_plus_three_to_the_k_mod_10_l144_144723


namespace theater_cost_per_square_foot_l144_144276

theorem theater_cost_per_square_foot
    (n_seats : ℕ)
    (space_per_seat : ℕ)
    (cost_ratio : ℕ)
    (partner_coverage : ℕ)
    (tom_expense : ℕ)
    (total_seats := 500)
    (square_footage := total_seats * space_per_seat)
    (construction_cost := cost_ratio * land_cost)
    (total_cost := land_cost + construction_cost)
    (partner_expense := total_cost * partner_coverage / 100)
    (tom_expense_ratio := 100 - partner_coverage)
    (cost_equation := tom_expense = total_cost * tom_expense_ratio / 100)
    (land_cost := 30000) :
    tom_expense = 54000 → 
    space_per_seat = 12 → 
    cost_ratio = 2 →
    partner_coverage = 40 → 
    tom_expense_ratio = 60 → 
    total_cost = 90000 → 
    total_cost / 3 = land_cost →
    land_cost / square_footage = 5 :=
    sorry

end theater_cost_per_square_foot_l144_144276


namespace circle_diameter_l144_144200

theorem circle_diameter (r : ℝ) (h : π * r^2 = 16 * π) : 2 * r = 8 :=
by
  sorry

end circle_diameter_l144_144200


namespace find_slope_l144_144254

theorem find_slope (x y : ℝ) (h : 4 * x + 7 * y = 28) : ∃ m : ℝ, m = -4/7 ∧ (∀ x, y = m * x + 4) := 
by
  sorry

end find_slope_l144_144254


namespace least_positive_integer_satisfying_congruences_l144_144636

theorem least_positive_integer_satisfying_congruences :
  ∃ b : ℕ, b > 0 ∧
    (b % 6 = 5) ∧
    (b % 7 = 6) ∧
    (b % 8 = 7) ∧
    (b % 9 = 8) ∧
    ∀ n : ℕ, (n > 0 → (n % 6 = 5) ∧ (n % 7 = 6) ∧ (n % 8 = 7) ∧ (n % 9 = 8) → n ≥ b) ∧
    b = 503 :=
by
  sorry

end least_positive_integer_satisfying_congruences_l144_144636


namespace simplify_fraction_l144_144037

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := 
by sorry

end simplify_fraction_l144_144037


namespace find_b_l144_144775

theorem find_b (a b : ℝ) (h1 : 3 * a - 2 = 1) (h2 : 2 * b - 3 * a = 2) : b = 5 / 2 := 
by 
  sorry

end find_b_l144_144775


namespace part_1_part_2_l144_144300

noncomputable def f (a x : ℝ) : ℝ := a - 1 / (1 + 2^x)

theorem part_1 (a : ℝ) (h1 : f a 1 + f a (-1) = 0) : a = 1 / 2 :=
by sorry

theorem part_2 : ∃ a : ℝ, ∀ x : ℝ, f a (-x) + f a x = 0 :=
by sorry

end part_1_part_2_l144_144300


namespace sin_alpha_sqrt5_div5_and_sin_beta_sqrt10_div10_acute_sum_pi_div4_l144_144859

theorem sin_alpha_sqrt5_div5_and_sin_beta_sqrt10_div10_acute_sum_pi_div4
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = Real.sqrt 5 / 5)
  (h_sin_β : Real.sin β = Real.sqrt 10 / 10) :
  α + β = π / 4 := sorry

end sin_alpha_sqrt5_div5_and_sin_beta_sqrt10_div10_acute_sum_pi_div4_l144_144859


namespace jake_split_shots_l144_144402

theorem jake_split_shots (shot_volume : ℝ) (purity : ℝ) (alcohol_consumed : ℝ) 
    (h1 : shot_volume = 1.5) (h2 : purity = 0.50) (h3 : alcohol_consumed = 3) : 
    2 * (alcohol_consumed / (purity * shot_volume)) = 8 :=
by
  sorry

end jake_split_shots_l144_144402


namespace dvds_still_fit_in_book_l144_144587

def total_capacity : ℕ := 126
def dvds_already_in_book : ℕ := 81

theorem dvds_still_fit_in_book : (total_capacity - dvds_already_in_book = 45) :=
by
  sorry

end dvds_still_fit_in_book_l144_144587


namespace triangle_XYZ_PQZ_lengths_l144_144317

theorem triangle_XYZ_PQZ_lengths :
  ∀ (X Y Z P Q : Type) (d_XZ d_YZ d_PQ : ℝ),
  d_XZ = 9 → d_YZ = 12 → d_PQ = 3 →
  ∀ (XY YP : ℝ),
  XY = Real.sqrt (d_XZ^2 + d_YZ^2) →
  YP = (d_PQ / d_XZ) * d_YZ →
  YP = 4 :=
by
  intros X Y Z P Q d_XZ d_YZ d_PQ hXZ hYZ hPQ XY YP hXY hYP
  -- Skipping detailed proof
  sorry

end triangle_XYZ_PQZ_lengths_l144_144317


namespace regular_polygon_property_l144_144774

variables {n : ℕ}
variables {r : ℝ} -- r is the radius of the circumscribed circle
variables {t_2n : ℝ} -- t_2n is the area of the 2n-gon
variables {k_n : ℝ} -- k_n is the perimeter of the n-gon

theorem regular_polygon_property
  (h1 : t_2n = (n * k_n * r) / 2)
  (h2 : k_n = n * a_n) :
  (t_2n / r^2) = (k_n / (2 * r)) :=
by sorry

end regular_polygon_property_l144_144774


namespace negation_of_p_is_neg_p_l144_144718

-- Define proposition p
def p : Prop := ∃ x : ℝ, x^2 + x - 1 ≥ 0

-- Define the negation of p
def neg_p : Prop := ∀ x : ℝ, x^2 + x - 1 < 0

theorem negation_of_p_is_neg_p : ¬p = neg_p := by
  -- The proof is omitted as per the instruction
  sorry

end negation_of_p_is_neg_p_l144_144718


namespace candle_problem_l144_144167

theorem candle_problem :
  ∃ x : ℚ,
    (1 - x / 6 = 3 * (1 - x / 5)) ∧
    x = 60 / 13 :=
by
  -- let initial_height_first_candle be 1
  -- let rate_first_burns be 1 / 6
  -- let initial_height_second_candle be 1
  -- let rate_second_burns be 1 / 5
  -- We want to prove:
  -- 1 - x / 6 = 3 * (1 - x / 5) ∧ x = 60 / 13
  sorry

end candle_problem_l144_144167


namespace correct_option_l144_144558

def inverse_proportion (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ y = k / x

theorem correct_option :
  inverse_proportion x y → 
  (y = x + 3 ∨ y = x / 3 ∨ y = 3 / (x ^ 2) ∨ y = 3 / x) → 
  y = 3 / x :=
by
  sorry

end correct_option_l144_144558


namespace differential_savings_l144_144803

def original_tax_rate : ℝ := 0.45
def new_tax_rate : ℝ := 0.30
def annual_income : ℝ := 48000

theorem differential_savings : (original_tax_rate * annual_income) - (new_tax_rate * annual_income) = 7200 := by
  sorry

end differential_savings_l144_144803


namespace exists_sequence_of_ten_numbers_l144_144393

theorem exists_sequence_of_ten_numbers :
  ∃ a : Fin 10 → ℝ,
    (∀ i : Fin 6,    a i + a ⟨i.1 + 1, sorry⟩ + a ⟨i.1 + 2, sorry⟩ + a ⟨i.1 + 3, sorry⟩ + a ⟨i.1 + 4, sorry⟩ > 0) ∧
    (∀ j : Fin 4, a j + a ⟨j.1 + 1, sorry⟩ + a ⟨j.1 + 2, sorry⟩ + a ⟨j.1 + 3, sorry⟩ + a ⟨j.1 + 4, sorry⟩ + a ⟨j.1 + 5, sorry⟩ + a ⟨j.1 + 6, sorry⟩ < 0) :=
sorry

end exists_sequence_of_ten_numbers_l144_144393


namespace monkey_hop_distance_l144_144781

theorem monkey_hop_distance
    (total_height : ℕ)
    (slip_back : ℕ)
    (hours : ℕ)
    (reach_time : ℕ)
    (hop : ℕ)
    (H1 : total_height = 19)
    (H2 : slip_back = 2)
    (H3 : hours = 17)
    (H4 : reach_time = 16 * (hop - slip_back) + hop)
    (H5 : total_height = reach_time) :
    hop = 3 := by
  sorry

end monkey_hop_distance_l144_144781


namespace find_a_l144_144632

noncomputable def P (a : ℚ) (k : ℕ) : ℚ := a * (1 / 2)^(k)

theorem find_a (a : ℚ) : (P a 1 + P a 2 + P a 3 = 1) → (a = 8 / 7) :=
by
  sorry

end find_a_l144_144632


namespace fraction_not_collapsing_l144_144975

variable (total_homes : ℕ)
variable (termite_ridden_fraction collapsing_fraction : ℚ)
variable (h : termite_ridden_fraction = 1 / 3)
variable (c : collapsing_fraction = 7 / 10)

theorem fraction_not_collapsing : 
  (termite_ridden_fraction - (termite_ridden_fraction * collapsing_fraction)) = 1 / 10 := 
by 
  rw [h, c]
  sorry

end fraction_not_collapsing_l144_144975


namespace total_cost_over_8_weeks_l144_144714

def cost_per_weekday_edition : ℝ := 0.50
def cost_per_sunday_edition : ℝ := 2.00
def num_weekday_editions_per_week : ℕ := 3
def duration_in_weeks : ℕ := 8

theorem total_cost_over_8_weeks :
  (num_weekday_editions_per_week * cost_per_weekday_edition + cost_per_sunday_edition) * duration_in_weeks = 28.00 := by
  sorry

end total_cost_over_8_weeks_l144_144714


namespace like_terms_exponent_l144_144884

theorem like_terms_exponent (x y : ℝ) (n : ℕ) : 
  (∀ (a b : ℝ), a * x ^ 3 * y ^ (n - 1) = b * x ^ 3 * y ^ 1 → n = 2) :=
by
  sorry

end like_terms_exponent_l144_144884


namespace factorize_expression_l144_144960

variable {x y : ℝ}

theorem factorize_expression :
  3 * x^2 - 27 * y^2 = 3 * (x + 3 * y) * (x - 3 * y) :=
by
  sorry

end factorize_expression_l144_144960


namespace minimum_value_of_a_l144_144552

theorem minimum_value_of_a (x : ℝ) (a : ℝ) (hx : 0 ≤ x) (hx2 : x ≤ 20) (ha : 0 < a) (h : (20 - x) / 4 + a / 2 * Real.sqrt x ≥ 5) : 
  a ≥ Real.sqrt 5 := 
sorry

end minimum_value_of_a_l144_144552


namespace f_satisfies_conditions_l144_144024

def g (n : Int) : Int :=
  if n >= 1 then 1 else 0

def f (n m : Int) : Int :=
  if m = 0 then n
  else n % m

theorem f_satisfies_conditions (n m : Int) : 
  (f 0 m = 0) ∧ 
  (f (n + 1) m = (1 - g m + g m * g (m - 1 - f n m)) * (1 + f n m)) := by
  sorry

end f_satisfies_conditions_l144_144024


namespace henry_socks_l144_144151

theorem henry_socks : 
  ∃ a b c : ℕ, 
    a + b + c = 15 ∧ 
    2 * a + 3 * b + 5 * c = 36 ∧ 
    a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ 
    a = 11 :=
by
  sorry

end henry_socks_l144_144151


namespace smallest_n_for_candy_distribution_l144_144349

theorem smallest_n_for_candy_distribution : ∃ (n : ℕ), (∀ (a : ℕ), ∃ (x : ℕ), (x * (x + 1)) / 2 % n = a % n) ∧ n = 2 :=
sorry

end smallest_n_for_candy_distribution_l144_144349


namespace second_point_x_coord_l144_144055

open Function

variable (n : ℝ)

def line_eq (y : ℝ) : ℝ := 2 * y + 5

theorem second_point_x_coord (h₁ : ∀ (x y : ℝ), x = line_eq y → True) :
  ∃ m : ℝ, ∀ n : ℝ, m = 2 * n + 5 → (m + 1 = line_eq (n + 0.5)) :=
by
  sorry

end second_point_x_coord_l144_144055


namespace temperature_decrease_2C_l144_144693

variable (increase_3 : ℤ := 3)
variable (decrease_2 : ℤ := -2)

theorem temperature_decrease_2C :
  decrease_2 = -2 :=
by
  -- This is where the proof would go
  sorry

end temperature_decrease_2C_l144_144693


namespace factorization_check_l144_144040

theorem factorization_check 
  (A : 4 - x^2 + 3 * x ≠ (2 - x) * (2 + x) + 3)
  (B : -x^2 + 3 * x + 4 ≠ -(x + 4) * (x - 1))
  (D : x^2 * y - x * y + x^3 * y ≠ x * (x * y - y + x^2 * y)) :
  1 - 2 * x + x^2 = (1 - x) ^ 2 :=
by
  sorry

end factorization_check_l144_144040


namespace range_of_a_l144_144064

def p (x : ℝ) : Prop := x ≤ 1/2 ∨ x ≥ 1

def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

def not_q (x a : ℝ) : Prop := x < a ∨ x > a + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, not_q x a → p x) ∧ (∃ x : ℝ, ¬ (p x → not_q x a)) →
  0 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l144_144064


namespace grid_sum_21_proof_l144_144396

-- Define the condition that the sum of the horizontal and vertical lines are 21
def valid_grid (nums : List ℕ) (x : ℕ) : Prop :=
  nums ≠ [] ∧ (((nums.sum + x) = 42) ∧ (21 + 21 = 42))

-- Define the main theorem to prove x = 7
theorem grid_sum_21_proof (nums : List ℕ) (h : valid_grid nums 7) : 7 ∈ nums :=
  sorry

end grid_sum_21_proof_l144_144396


namespace equal_opposite_roots_eq_m_l144_144578

theorem equal_opposite_roots_eq_m (a b c : ℝ) (m : ℝ) (h : (∃ x : ℝ, (a * x - c ≠ 0) ∧ (((x^2 - b * x) / (a * x - c)) = ((m - 1) / (m + 1)))) ∧
(∀ x : ℝ, ((x^2 - b * x) = 0 → x = 0) ∧ (∃ t : ℝ, t > 0 ∧ ((x = t) ∨ (x = -t))))):
  m = (a - b) / (a + b) :=
by
  sorry

end equal_opposite_roots_eq_m_l144_144578


namespace no_square_number_divisible_by_six_in_range_l144_144525

theorem no_square_number_divisible_by_six_in_range :
  ¬ ∃ (x : ℕ), (x ^ 2) % 6 = 0 ∧ 39 < x ^ 2 ∧ x ^ 2 < 120 :=
by
  sorry

end no_square_number_divisible_by_six_in_range_l144_144525


namespace goods_train_length_is_470_l144_144477

noncomputable section

def speed_kmph := 72
def platform_length := 250
def crossing_time := 36

def speed_mps := speed_kmph * 5 / 18
def distance_covered := speed_mps * crossing_time

def length_of_train := distance_covered - platform_length

theorem goods_train_length_is_470 :
  length_of_train = 470 :=
by
  sorry

end goods_train_length_is_470_l144_144477


namespace investment_in_scheme_B_l144_144252

theorem investment_in_scheme_B 
    (yieldA : ℝ) (yieldB : ℝ) (investmentA : ℝ) (difference : ℝ) (totalA : ℝ) (totalB : ℝ):
    yieldA = 0.30 → yieldB = 0.50 → investmentA = 300 → difference = 90 
    → totalA = investmentA + (yieldA * investmentA) 
    → totalB = (1 + yieldB) * totalB 
    → totalA = totalB + difference 
    → totalB = 200 :=
by sorry

end investment_in_scheme_B_l144_144252


namespace mixture_contains_pecans_l144_144912

theorem mixture_contains_pecans 
  (price_per_cashew_per_pound : ℝ)
  (cashews_weight : ℝ)
  (price_per_mixture_per_pound : ℝ)
  (price_of_cashews : ℝ)
  (mixture_weight : ℝ)
  (pecans_weight : ℝ)
  (price_per_pecan_per_pound : ℝ)
  (pecans_price : ℝ)
  (total_cost_of_mixture : ℝ)
  
  (h1 : price_per_cashew_per_pound = 3.50) 
  (h2 : cashews_weight = 2)
  (h3 : price_per_mixture_per_pound = 4.34) 
  (h4 : pecans_weight = 1.33333333333)
  (h5 : price_per_pecan_per_pound = 5.60)
  
  (h6 : price_of_cashews = cashews_weight * price_per_cashew_per_pound)
  (h7 : mixture_weight = cashews_weight + pecans_weight)
  (h8 : pecans_price = pecans_weight * price_per_pecan_per_pound)
  (h9 : total_cost_of_mixture = price_of_cashews + pecans_price)

  (h10 : price_per_mixture_per_pound = total_cost_of_mixture / mixture_weight)
  
  : pecans_weight = 1.33333333333 :=
sorry

end mixture_contains_pecans_l144_144912


namespace initial_men_in_garrison_l144_144548

variable (x : ℕ)

theorem initial_men_in_garrison (h1 : x * 65 = x * 50 + (x + 3000) * 20) : x = 2000 :=
  sorry

end initial_men_in_garrison_l144_144548


namespace simplify_and_evaluate_expression_l144_144387

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 1/2) : x^2 * (x - 1) - x * (x^2 + x - 1) = 0 := by
  sorry

end simplify_and_evaluate_expression_l144_144387


namespace withdraw_representation_l144_144447

-- Define the concept of depositing and withdrawing money.
def deposit (amount : ℕ) : ℤ := amount
def withdraw (amount : ℕ) : ℤ := - amount

-- Define the given condition: depositing $30,000 is represented as $+30,000.
def deposit_condition : deposit 30000 = 30000 := by rfl

-- The statement to be proved: withdrawing $40,000 is represented as $-40,000
theorem withdraw_representation (deposit_condition : deposit 30000 = 30000) : withdraw 40000 = -40000 :=
by
  sorry

end withdraw_representation_l144_144447


namespace solve_for_x_add_y_l144_144508

theorem solve_for_x_add_y (x y : ℤ) 
  (h1 : y = 245) 
  (h2 : x - y = 200) : 
  x + y = 690 :=
by {
  -- Here we would provide the proof if needed
  sorry
}

end solve_for_x_add_y_l144_144508


namespace capacity_ratio_proof_l144_144561

noncomputable def capacity_ratio :=
  ∀ (C_X C_Y : ℝ), 
    (1 / 2) * C_X + (2 / 5) * C_Y = (65 / 100) * C_Y →
    (C_Y / C_X) = (1 / 2)

-- includes a statement without proof
theorem capacity_ratio_proof (C_X C_Y : ℝ) (h : (1 / 2) * C_X + (2 / 5) * C_Y = (65 / 100) * C_Y) : 
  (C_Y / C_X) = (1 / 2) :=
  by
    sorry

end capacity_ratio_proof_l144_144561


namespace three_digit_cubes_divisible_by_4_l144_144720

-- Let's define the conditions in Lean
def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Let's combine these conditions to define the target predicate in Lean
def is_target_number (n : ℕ) : Prop := is_three_digit n ∧ is_perfect_cube n ∧ is_divisible_by_4 n

-- The statement to be proven: that there is only one such number
theorem three_digit_cubes_divisible_by_4 : 
  (∃! n, is_target_number n) :=
sorry

end three_digit_cubes_divisible_by_4_l144_144720


namespace find_max_problems_l144_144267

def max_problems_in_7_days (P : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, i ∈ Finset.range 7 → P i ≤ 10) ∧
  (∀ i : ℕ, i ∈ Finset.range 5 → (P i > 7) → (P (i + 1) ≤ 5 ∧ P (i + 2) ≤ 5))

theorem find_max_problems : ∃ P : ℕ → ℕ, max_problems_in_7_days P ∧ (Finset.range 7).sum P = 50 :=
by
  sorry

end find_max_problems_l144_144267


namespace graph_intersects_x_axis_once_l144_144183

noncomputable def f (m x : ℝ) : ℝ := (m - 1) * x^2 - 6 * x + (3 / 2) * m

theorem graph_intersects_x_axis_once (m : ℝ) :
  (∃ x : ℝ, f m x = 0 ∧ ∀ y : ℝ, f m y = 0 → y = x) ↔ (m = 1 ∨ m = 3 ∨ m = -2) :=
by
  sorry

end graph_intersects_x_axis_once_l144_144183


namespace option_C_correct_l144_144497

theorem option_C_correct (m n : ℝ) (h : m > n) : (1/5) * m > (1/5) * n := 
by
  sorry

end option_C_correct_l144_144497


namespace initial_music_files_eq_sixteen_l144_144233

theorem initial_music_files_eq_sixteen (M : ℕ) :
  (M + 48 - 30 = 34) → (M = 16) :=
by
  sorry

end initial_music_files_eq_sixteen_l144_144233


namespace Gyeongyeon_cookies_l144_144476

def initial_cookies : ℕ := 20
def cookies_given : ℕ := 7
def cookies_received : ℕ := 5

def final_cookies (initial : ℕ) (given : ℕ) (received : ℕ) : ℕ :=
  initial - given + received

theorem Gyeongyeon_cookies :
  final_cookies initial_cookies cookies_given cookies_received = 18 :=
by
  sorry

end Gyeongyeon_cookies_l144_144476


namespace divisibility_condition_l144_144527

theorem divisibility_condition (M C D U A q1 q2 q3 r1 r2 r3 : ℕ)
  (h1 : 10 = A * q1 + r1)
  (h2 : 10 * r1 = A * q2 + r2)
  (h3 : 10 * r2 = A * q3 + r3) :
  (U + D * r1 + C * r2 + M * r3) % A = 0 ↔ (1000 * M + 100 * C + 10 * D + U) % A = 0 :=
sorry

end divisibility_condition_l144_144527


namespace andrew_worked_days_l144_144608

-- Definitions per given conditions
def vacation_days_per_work_days (W : ℕ) : ℕ := W / 10
def days_taken_off_in_march := 5
def days_taken_off_in_september := 2 * days_taken_off_in_march
def total_days_off_taken := days_taken_off_in_march + days_taken_off_in_september
def remaining_vacation_days := 15
def total_vacation_days := total_days_off_taken + remaining_vacation_days

theorem andrew_worked_days (W : ℕ) :
  vacation_days_per_work_days W = total_vacation_days → W = 300 := by
  sorry

end andrew_worked_days_l144_144608


namespace intersection_points_form_line_l144_144701

theorem intersection_points_form_line :
  ∀ (x y : ℝ), ((x * y = 12) ∧ ((x^2 / 16) + (y^2 / 36) = 1)) →
  ∃ (x1 x2 : ℝ) (y1 y2 : ℝ), (x, y) = (x1, y1) ∨ (x, y) = (x2, y2) ∧ (x2 - x1) * (y2 - y1) = x1 * y1 - x2 * y2 :=
by
  sorry

end intersection_points_form_line_l144_144701


namespace greatest_integer_x_l144_144901

theorem greatest_integer_x (x : ℤ) (h : (5 : ℚ) / 8 > (x : ℚ) / 17) : x ≤ 10 :=
by
  sorry

end greatest_integer_x_l144_144901


namespace find_value_of_triangle_l144_144872

theorem find_value_of_triangle (p : ℕ) (triangle : ℕ) 
  (h1 : triangle + p = 47) 
  (h2 : 3 * (triangle + p) - p = 133) :
  triangle = 39 :=
by 
  sorry

end find_value_of_triangle_l144_144872


namespace john_twice_james_l144_144415

def john_age : ℕ := 39
def years_ago : ℕ := 3
def years_future : ℕ := 6
def age_difference : ℕ := 4

theorem john_twice_james {J : ℕ} (h : 39 - years_ago = 2 * (J + years_future)) : 
  (J + age_difference = 16) :=
by
  sorry  -- Proof steps here

end john_twice_james_l144_144415


namespace principal_sum_l144_144036

/-!
# Problem Statement
Given:
1. The difference between compound interest (CI) and simple interest (SI) on a sum at 10% per annum for 2 years is 65.
2. The rate of interest \( R \) is 10%.
3. The time \( T \) is 2 years.

We need to prove that the principal sum \( P \) is 6500.
-/

theorem principal_sum (P : ℝ) (R : ℝ) (T : ℕ) (H : (P * (1 + R / 100)^T - P) - (P * R * T / 100) = 65) 
                      (HR : R = 10) (HT : T = 2) : P = 6500 := 
by 
  sorry

end principal_sum_l144_144036


namespace whitney_spent_179_l144_144460

def total_cost (books_whales books_fish magazines book_cost magazine_cost : ℕ) : ℕ :=
  (books_whales + books_fish) * book_cost + magazines * magazine_cost

theorem whitney_spent_179 :
  total_cost 9 7 3 11 1 = 179 :=
by
  sorry

end whitney_spent_179_l144_144460


namespace smallest_five_digit_multiple_of_18_l144_144799

theorem smallest_five_digit_multiple_of_18 : ∃ n : ℕ, n = 10008 ∧ (n ≥ 10000 ∧ n < 100000) ∧ n % 18 = 0 ∧ (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ m % 18 = 0 → n ≤ m) := sorry

end smallest_five_digit_multiple_of_18_l144_144799


namespace find_m_solve_inequality_l144_144524

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := m - |x - 2|

theorem find_m (m : ℝ) : (∀ x : ℝ, m - |x| ≥ 0 ↔ x ∈ [-1, 1]) → m = 1 :=
by
  sorry

theorem solve_inequality (x : ℝ) : |x + 1| + |x - 2| > 4 * 1 ↔ x < -3 / 2 ∨ x > 5 / 2 :=
by
  sorry

end find_m_solve_inequality_l144_144524


namespace triangle_is_isosceles_range_of_expression_l144_144936

variable {a b c A B C : ℝ}
variable (triangle_ABC : 0 < A ∧ A < π ∧ 0 < B ∧ B < π)
variable (opposite_sides : a = 1 ∧ b = 1 ∧ c = 1)
variable (cos_condition : a * Real.cos B = b * Real.cos A)

theorem triangle_is_isosceles (h : a * Real.cos B = b * Real.cos A) : A = B := sorry

theorem range_of_expression 
  (h1 : 0 < A ∧ A < π/2) 
  (h2 : a * Real.cos B = b * Real.cos A) : 
  -3/2 < Real.sin (2 * A + π / 6) - 2 * Real.cos B ^ 2 ∧ Real.sin (2 * A + π / 6) - 2 * Real.cos B ^ 2 < 0 := 
sorry

end triangle_is_isosceles_range_of_expression_l144_144936


namespace max_cards_from_poster_board_l144_144762

theorem max_cards_from_poster_board (card_length card_width poster_length : ℕ) (h1 : card_length = 2) (h2 : card_width = 3) (h3 : poster_length = 12) : 
  (poster_length / card_length) * (poster_length / card_width) = 24 :=
by
  sorry

end max_cards_from_poster_board_l144_144762


namespace second_container_clay_l144_144463

theorem second_container_clay :
  let h1 := 3
  let w1 := 5
  let l1 := 7
  let clay1 := 105
  let h2 := 3 * h1
  let w2 := 2 * w1
  let l2 := l1
  let V1 := h1 * w1 * l1
  let V2 := h2 * w2 * l2
  V1 = clay1 →
  V2 = 6 * V1 →
  V2 = 630 :=
by
  intros
  sorry

end second_container_clay_l144_144463


namespace chord_length_of_intersection_l144_144827

def ellipse (x y : ℝ) := x^2 + 4 * y^2 = 16
def line (x y : ℝ) := y = (1/2) * x + 1

theorem chord_length_of_intersection :
  ∃ A B : ℝ × ℝ, ellipse A.fst A.snd ∧ ellipse B.fst B.snd ∧ line A.fst A.snd ∧ line B.fst B.snd ∧
  dist A B = Real.sqrt 35 :=
sorry

end chord_length_of_intersection_l144_144827


namespace count_positive_integers_m_l144_144602

theorem count_positive_integers_m :
  ∃ m_values : Finset ℕ, m_values.card = 4 ∧ ∀ m ∈ m_values, 
    ∃ k : ℕ, k > 0 ∧ (7 * m + 2 = m * k + 2 * m) := 
sorry

end count_positive_integers_m_l144_144602


namespace equivalent_lemons_l144_144864

theorem equivalent_lemons 
  (lemons_per_apple_approx : ∀ apples : ℝ, 3/4 * 14 = 9 → 1 = 9 / (3/4 * 14))
  (apples_to_lemons : ℝ) :
  5 / 7 * 7 = 30 / 7 :=
by
  sorry

end equivalent_lemons_l144_144864


namespace water_used_for_plates_and_clothes_is_48_l144_144144

noncomputable def waterUsedToWashPlatesAndClothes : ℕ := 
  let barrel1 := 65 
  let barrel2 := (75 * 80) / 100 
  let barrel3 := (45 * 60) / 100 
  let totalCollected := barrel1 + barrel2 + barrel3
  let usedForCars := 7 * 2
  let usedForPlants := 15
  let usedForDog := 10
  let usedForCooking := 5
  let usedForBathing := 12
  let totalUsed := usedForCars + usedForPlants + usedForDog + usedForCooking + usedForBathing
  let remainingWater := totalCollected - totalUsed
  remainingWater / 2

theorem water_used_for_plates_and_clothes_is_48 : 
  waterUsedToWashPlatesAndClothes = 48 :=
by
  sorry

end water_used_for_plates_and_clothes_is_48_l144_144144


namespace positively_correlated_variables_l144_144759

-- Define all conditions given in the problem
def weightOfCarVar1 : Type := ℝ
def avgDistPerLiter : Type := ℝ
def avgStudyTime : Type := ℝ
def avgAcademicPerformance : Type := ℝ
def dailySmokingAmount : Type := ℝ
def healthCondition : Type := ℝ
def sideLength : Type := ℝ
def areaOfSquare : Type := ℝ
def fuelConsumptionPerHundredKm : Type := ℝ

-- Define the relationship status between variables
def isPositivelyCorrelated (x y : Type) : Prop := sorry
def isFunctionallyRelated (x y : Type) : Prop := sorry

axiom weight_car_distance_neg : ¬ isPositivelyCorrelated weightOfCarVar1 avgDistPerLiter
axiom study_time_performance_pos : isPositivelyCorrelated avgStudyTime avgAcademicPerformance
axiom smoking_health_neg : ¬ isPositivelyCorrelated dailySmokingAmount healthCondition
axiom side_area_func : isFunctionallyRelated sideLength areaOfSquare
axiom car_weight_fuel_pos : isPositivelyCorrelated weightOfCarVar1 fuelConsumptionPerHundredKm

-- The proof statement to prove C is the correct answer
theorem positively_correlated_variables:
  isPositivelyCorrelated avgStudyTime avgAcademicPerformance ∧
  isPositivelyCorrelated weightOfCarVar1 fuelConsumptionPerHundredKm :=
by
  sorry

end positively_correlated_variables_l144_144759


namespace event_eq_conds_l144_144786

-- Definitions based on conditions
def Die := { n : ℕ // 1 ≤ n ∧ n ≤ 6 }
def sum_points (d1 d2 : Die) : ℕ := d1.val + d2.val

def event_xi_eq_4 (d1 d2 : Die) : Prop := 
  sum_points d1 d2 = 4

def condition_a (d1 d2 : Die) : Prop := 
  d1.val = 2 ∧ d2.val = 2

def condition_b (d1 d2 : Die) : Prop := 
  (d1.val = 3 ∧ d2.val = 1) ∨ (d1.val = 1 ∧ d2.val = 3)

def event_condition (d1 d2 : Die) : Prop :=
  condition_a d1 d2 ∨ condition_b d1 d2

-- The main Lean statement
theorem event_eq_conds (d1 d2 : Die) : 
  event_xi_eq_4 d1 d2 ↔ event_condition d1 d2 := 
by
  sorry

end event_eq_conds_l144_144786


namespace solve_inequality_l144_144435

theorem solve_inequality (x : ℝ) : 2 * x + 6 > 5 * x - 3 → x < 3 :=
by
  -- Proof steps would go here
  sorry

end solve_inequality_l144_144435


namespace caleb_caught_trouts_l144_144563

theorem caleb_caught_trouts (C : ℕ) (h1 : 3 * C = C + 4) : C = 2 :=
by {
  sorry
}

end caleb_caught_trouts_l144_144563


namespace candy_left_l144_144428

-- Define the given conditions
def KatieCandy : ℕ := 8
def SisterCandy : ℕ := 23
def AteCandy : ℕ := 8

-- The theorem stating the total number of candy left
theorem candy_left (k : ℕ) (s : ℕ) (e : ℕ) (hk : k = KatieCandy) (hs : s = SisterCandy) (he : e = AteCandy) : 
  (k + s) - e = 23 :=
by
  -- (Proof will be inserted here, but we include a placeholder "sorry" for now)
  sorry

end candy_left_l144_144428


namespace integer_points_on_segment_l144_144742

noncomputable def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 else 0

theorem integer_points_on_segment (n : ℕ) (h : 0 < n) :
  f n = if n % 3 = 0 then 2 else 0 :=
by
  sorry

end integer_points_on_segment_l144_144742


namespace prime_square_mod_180_l144_144057

theorem prime_square_mod_180 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : 5 < p) :
  ∃ (s : Finset ℕ), s.card = 2 ∧ ∀ r ∈ s, ∃ n : ℕ, p^2 % 180 = r :=
sorry

end prime_square_mod_180_l144_144057


namespace population_decreases_l144_144865

theorem population_decreases (P_0 : ℝ) (k : ℝ) (n : ℕ) (hP0 : P_0 > 0) (hk : -1 < k ∧ k < 0) : 
  P_0 * (1 + k)^n * k < 0 → P_0 * (1 + k)^(n + 1) < P_0 * (1 + k)^n := by
  sorry

end population_decreases_l144_144865


namespace bus_speed_including_stoppages_l144_144994

theorem bus_speed_including_stoppages 
  (speed_without_stoppages : ℕ) 
  (stoppage_time_per_hour : ℕ) 
  (correct_speed_including_stoppages : ℕ) :
  speed_without_stoppages = 54 →
  stoppage_time_per_hour = 10 →
  correct_speed_including_stoppages = 45 :=
by
sorry

end bus_speed_including_stoppages_l144_144994


namespace second_person_avg_pages_per_day_l144_144999

theorem second_person_avg_pages_per_day
  (summer_days : ℕ) 
  (deshaun_books : ℕ) 
  (avg_pages_per_book : ℕ) 
  (percentage_read_by_second_person : ℚ) :
  -- Given conditions
  summer_days = 80 →
  deshaun_books = 60 →
  avg_pages_per_book = 320 →
  percentage_read_by_second_person = 0.75 →
  -- Prove
  (percentage_read_by_second_person * (deshaun_books * avg_pages_per_book) / summer_days) = 180 :=
by
  intros h1 h2 h3 h4
  sorry

end second_person_avg_pages_per_day_l144_144999


namespace find_b_l144_144006

variable {a b c : ℚ}

theorem find_b (h1 : a + b + c = 117) (h2 : a + 8 = 4 * c) (h3 : b - 10 = 4 * c) : b = 550 / 9 := by
  sorry

end find_b_l144_144006


namespace common_difference_divisible_by_6_l144_144651

theorem common_difference_divisible_by_6 (p q r d : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hp3 : p > 3) (hq3 : q > 3) (hr3 : r > 3) (h1 : q = p + d) (h2 : r = p + 2 * d) : d % 6 = 0 := 
sorry

end common_difference_divisible_by_6_l144_144651


namespace part1_part2_l144_144072

namespace Proof

def A (a b : ℝ) : ℝ := 3 * a ^ 2 - 4 * a * b
def B (a b : ℝ) : ℝ := a ^ 2 + 2 * a * b

theorem part1 (a b : ℝ) : 2 * A a b - 3 * B a b = 3 * a ^ 2 - 14 * a * b := by
  sorry
  
theorem part2 (a b : ℝ) (h : |3 * a + 1| + (2 - 3 * b) ^ 2 = 0) : A a b - 2 * B a b = 5 / 3 := by
  have ha : a = -1 / 3 := by
    sorry
  have hb : b = 2 / 3 := by
    sorry
  rw [ha, hb]
  sorry

end Proof

end part1_part2_l144_144072


namespace combined_area_of_four_removed_triangles_l144_144380

noncomputable def combined_area_of_removed_triangles (s x y: ℝ) : Prop :=
  x + y = s ∧ s - 2 * x = 15 ∧ s - 2 * y = 9 ∧
  4 * (1 / 2 * x * y) = 67.5

-- Statement of the problem
theorem combined_area_of_four_removed_triangles (s x y: ℝ) :
  combined_area_of_removed_triangles s x y :=
  by
    sorry

end combined_area_of_four_removed_triangles_l144_144380


namespace solve_inequality_l144_144053

theorem solve_inequality (x : ℝ) : |x - 2| > 2 - x ↔ x > 2 :=
sorry

end solve_inequality_l144_144053


namespace length_O_D1_l144_144894

-- Definitions for the setup of the cube and its faces, the center of the sphere, and the intersecting circles
def O : Point := sorry -- Center of the sphere and cube
def radius : ℝ := 10 -- Radius of the sphere

-- Intersection circles with given radii on specific faces of the cube
def r_ADA1D1 : ℝ := 1 -- Radius of the intersection circle on face ADA1D1
def r_A1B1C1D1 : ℝ := 1 -- Radius of the intersection circle on face A1B1C1D1
def r_CDD1C1 : ℝ := 3 -- Radius of the intersection circle on face CDD1C1

-- Distances derived from the problem
def OX1_sq : ℝ := radius^2 - r_ADA1D1^2
def OX2_sq : ℝ := radius^2 - r_A1B1C1D1^2
def OX_sq : ℝ := radius^2 - r_CDD1C1^2

-- To simplify, replace OX1, OX2, and OX with their squared values directly
def OX1_sq_calc : ℝ := 99
def OX2_sq_calc : ℝ := 99
def OX_sq_calc : ℝ := 91

theorem length_O_D1 : (OX1_sq_calc + OX2_sq_calc + OX_sq_calc) = 289 ↔ OD1 = 17 := by
  sorry

end length_O_D1_l144_144894


namespace diameter_of_inscribed_circle_l144_144093

theorem diameter_of_inscribed_circle (a b c r : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_radius : r = (a + b - c) / 2) : 
  2 * r = a + b - c :=
by
  sorry

end diameter_of_inscribed_circle_l144_144093


namespace manfred_total_paychecks_l144_144685

-- Define the conditions
def first_paychecks : ℕ := 6
def first_paycheck_amount : ℕ := 750
def remaining_paycheck_amount : ℕ := first_paycheck_amount + 20
def average_amount : ℝ := 765.38

-- Main theorem statement
theorem manfred_total_paychecks (x : ℕ) (h : (first_paychecks * first_paycheck_amount + x * remaining_paycheck_amount) / (first_paychecks + x) = average_amount) : first_paychecks + x = 26 :=
by
  sorry

end manfred_total_paychecks_l144_144685


namespace xiao_ming_excellent_score_probability_l144_144771

theorem xiao_ming_excellent_score_probability :
  let P_M : ℝ := 0.5
  let P_L : ℝ := 0.3
  let P_E := 1 - P_M - P_L
  P_E = 0.2 :=
by
  let P_M : ℝ := 0.5
  let P_L : ℝ := 0.3
  let P_E := 1 - P_M - P_L
  sorry

end xiao_ming_excellent_score_probability_l144_144771


namespace bags_with_chocolate_hearts_l144_144875

-- Definitions for given conditions
def total_candies : ℕ := 63
def total_bags : ℕ := 9
def candies_per_bag : ℕ := total_candies / total_bags
def chocolate_kiss_bags : ℕ := 3
def not_chocolate_candies : ℕ := 28
def bags_not_chocolate : ℕ := not_chocolate_candies / candies_per_bag
def remaining_bags : ℕ := total_bags - chocolate_kiss_bags - bags_not_chocolate

-- Statement to be proved
theorem bags_with_chocolate_hearts :
  remaining_bags = 2 := by 
  sorry

end bags_with_chocolate_hearts_l144_144875


namespace functional_equation_divisibility_l144_144339

theorem functional_equation_divisibility (f : ℕ+ → ℕ+) :
  (∀ x y : ℕ+, (f x)^2 + y ∣ f y + x^2) → (∀ x : ℕ+, f x = x) :=
by
  sorry

end functional_equation_divisibility_l144_144339


namespace smart_charging_piles_growth_l144_144551

-- Define the conditions
variables {x : ℝ}

-- First month charging piles
def first_month_piles : ℝ := 301

-- Third month charging piles
def third_month_piles : ℝ := 500

-- The theorem stating the relationship between the first and third month
theorem smart_charging_piles_growth : 
  first_month_piles * (1 + x) ^ 2 = third_month_piles :=
by
  sorry

end smart_charging_piles_growth_l144_144551


namespace geometric_sequence_sum_l144_144519

theorem geometric_sequence_sum (q : ℝ) (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_a1 : a 1 = 1)
  (h_sum : a 1 + a 3 + a 5 = 21) :
  a 2 + a 4 + a 6 = 42 :=
sorry

end geometric_sequence_sum_l144_144519


namespace problem_1_problem_2_l144_144949

open Real
open Set

noncomputable def y (x : ℝ) : ℝ := (2 * sin x - cos x ^ 2) / (1 + sin x)

theorem problem_1 :
  { x : ℝ | y x = 1 ∧ sin x ≠ -1 } = { x | ∃ (k : ℤ), x = 2 * k * π + (π / 2) } :=
by
  sorry

theorem problem_2 : 
  ∃ x, y x = 1 ∧ ∀ x', y x' ≤ 1 :=
by
  sorry

end problem_1_problem_2_l144_144949


namespace other_root_of_quadratic_l144_144503

theorem other_root_of_quadratic (m : ℝ) (h : ∀ x : ℝ, x^2 + m*x - 20 = 0 → (x = -4)) 
: ∃ t : ℝ, t = 5 := 
by
  existsi 5
  sorry

end other_root_of_quadratic_l144_144503


namespace sugar_amount_l144_144406

variables (S F B : ℝ)

-- Conditions
def condition1 : Prop := S / F = 5 / 2
def condition2 : Prop := F / B = 10 / 1
def condition3 : Prop := F / (B + 60) = 8 / 1

-- Theorem to prove
theorem sugar_amount (h1 : condition1 S F) (h2 : condition2 F B) (h3 : condition3 F B) : S = 6000 :=
sorry

end sugar_amount_l144_144406


namespace canal_depth_l144_144291

theorem canal_depth (A : ℝ) (W_top : ℝ) (W_bottom : ℝ) (d : ℝ) (h: ℝ)
  (h₁ : A = 840) 
  (h₂ : W_top = 12) 
  (h₃ : W_bottom = 8)
  (h₄ : A = (1/2) * (W_top + W_bottom) * d) : 
  d = 84 :=
by 
  sorry

end canal_depth_l144_144291


namespace dropouts_correct_l144_144981

/-- Definition for initial racers, racers joining after 20 minutes, and racers at finish line. -/
def initial_racers : ℕ := 50
def joining_racers : ℕ := 30
def finishers : ℕ := 130

/-- Total racers after initial join and doubling. -/
def total_racers : ℕ := (initial_racers + joining_racers) * 2

/-- The number of people who dropped out before finishing the race. -/
def dropped_out : ℕ := total_racers - finishers

/-- Proof statement to show the number of people who dropped out before finishing is 30. -/
theorem dropouts_correct : dropped_out = 30 := by
  sorry

end dropouts_correct_l144_144981


namespace remainder_relation_l144_144448

theorem remainder_relation (P P' D R R' : ℕ) (hP : P > P') (h1 : P % D = R) (h2 : P' % D = R') :
  ∃ C : ℕ, ((P + C) * P') % D ≠ (P * P') % D ∧ ∃ C : ℕ, ((P + C) * P') % D = (P * P') % D :=
by sorry

end remainder_relation_l144_144448


namespace find_n_l144_144438

theorem find_n (x n : ℝ) (h : x > 0) 
  (h_eq : x / 10 + x / n = 0.14000000000000002 * x) : 
  n = 25 :=
by
  sorry

end find_n_l144_144438


namespace value_of_f_sum_l144_144245

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (h_odd : ∀ x, f (-x) = -f x) : Prop
axiom period_9 (h_period : ∀ x, f (x + 9) = f x) : Prop
axiom f_one (h_f1 : f 1 = 5) : Prop

theorem value_of_f_sum (h_odd : ∀ x, f (-x) = -f x)
                       (h_period : ∀ x, f (x + 9) = f x)
                       (h_f1 : f 1 = 5) :
  f 2007 + f 2008 = 5 :=
sorry

end value_of_f_sum_l144_144245


namespace train_average_speed_l144_144614

theorem train_average_speed (speed : ℕ) (stop_time : ℕ) (running_time : ℕ) (total_time : ℕ)
  (h1 : speed = 60)
  (h2 : stop_time = 24)
  (h3 : running_time = total_time - stop_time)
  (h4 : running_time = 36)
  (h5 : total_time = 60) :
  (speed * running_time / total_time = 36) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end train_average_speed_l144_144614


namespace train_passes_jogger_time_l144_144699

theorem train_passes_jogger_time (speed_jogger_kmph : ℝ) 
                                (speed_train_kmph : ℝ) 
                                (distance_ahead_m : ℝ) 
                                (length_train_m : ℝ) : 
  speed_jogger_kmph = 9 → 
  speed_train_kmph = 45 →
  distance_ahead_m = 250 →
  length_train_m = 120 →
  (distance_ahead_m + length_train_m) / (speed_train_kmph - speed_jogger_kmph) * (1000 / 3600) = 37 :=
by
  intros h1 h2 h3 h4
  sorry

end train_passes_jogger_time_l144_144699


namespace optimal_discount_order_l144_144194

variables (p : ℝ) (d1 : ℝ) (d2 : ℝ)

-- Original price of "Stars Beyond" is 30 dollars
def original_price : ℝ := 30

-- Fixed discount is 5 dollars
def discount_5 : ℝ := 5

-- 25% discount represented as a multiplier
def discount_25 : ℝ := 0.75

-- Applying $5 discount first and then 25% discount
def price_after_5_then_25_discount := discount_25 * (original_price - discount_5)

-- Applying 25% discount first and then $5 discount
def price_after_25_then_5_discount := (discount_25 * original_price) - discount_5

-- The additional savings when applying 25% discount first
def additional_savings := price_after_5_then_25_discount - price_after_25_then_5_discount

theorem optimal_discount_order : 
  additional_savings = 1.25 :=
sorry

end optimal_discount_order_l144_144194


namespace youngest_person_age_l144_144677

noncomputable def avg_age_seven_people := 30
noncomputable def avg_age_six_people_when_youngest_born := 25
noncomputable def num_people := 7
noncomputable def num_people_minus_one := 6

theorem youngest_person_age :
  let total_age_seven_people := num_people * avg_age_seven_people
  let total_age_six_people := num_people_minus_one * avg_age_six_people_when_youngest_born
  total_age_seven_people - total_age_six_people = 60 :=
by
  let total_age_seven_people := num_people * avg_age_seven_people
  let total_age_six_people := num_people_minus_one * avg_age_six_people_when_youngest_born
  sorry

end youngest_person_age_l144_144677


namespace find_x_plus_2y_squared_l144_144647

theorem find_x_plus_2y_squared (x y : ℝ) (h1 : x * (x + 2 * y) = 48) (h2 : y * (x + 2 * y) = 72) :
  (x + 2 * y) ^ 2 = 96 := 
sorry

end find_x_plus_2y_squared_l144_144647


namespace john_initial_pairs_9_l144_144153

-- Definitions based on the conditions in the problem

def john_initial_pairs (x : ℕ) := 2 * x   -- Each pair consists of 2 socks

def john_remaining_socks (x : ℕ) := john_initial_pairs x - 5   -- John loses 5 individual socks

def john_max_pairs_left := 7
def john_minimum_socks_required := john_max_pairs_left * 2  -- 7 pairs mean he needs 14 socks

-- Theorem statement proving John initially had 9 pairs of socks
theorem john_initial_pairs_9 : 
  ∀ (x : ℕ), john_remaining_socks x ≥ john_minimum_socks_required → x = 9 := by
  sorry

end john_initial_pairs_9_l144_144153


namespace more_people_needed_to_paint_fence_l144_144286

theorem more_people_needed_to_paint_fence :
  ∀ (n t m t' : ℕ), n = 8 → t = 3 → t' = 2 → (n * t = m * t') → m - n = 4 :=
by
  intros n t m t'
  intro h1
  intro h2
  intro h3
  intro h4
  sorry

end more_people_needed_to_paint_fence_l144_144286


namespace jackson_grade_l144_144683

open Function

theorem jackson_grade :
  ∃ (grade : ℕ), 
  ∀ (hours_playing hours_studying : ℕ), 
    (hours_playing = 9) ∧ 
    (hours_studying = hours_playing / 3) ∧ 
    (grade = hours_studying * 15) →
    grade = 45 := 
by {
  sorry
}

end jackson_grade_l144_144683


namespace jakes_class_boys_count_l144_144607

theorem jakes_class_boys_count 
    (ratio_girls_boys : ℕ → ℕ → Prop)
    (students_total : ℕ)
    (ratio_condition : ratio_girls_boys 3 4)
    (total_condition : students_total = 35) :
    ∃ boys : ℕ, boys = 20 :=
by
  sorry

end jakes_class_boys_count_l144_144607


namespace lcm_20_45_36_l144_144485

-- Definitions from the problem
def num1 : ℕ := 20
def num2 : ℕ := 45
def num3 : ℕ := 36

-- Statement of the proof problem
theorem lcm_20_45_36 : Nat.lcm (Nat.lcm num1 num2) num3 = 180 := by
  sorry

end lcm_20_45_36_l144_144485


namespace compare_magnitudes_l144_144917

noncomputable
def f (x : ℝ) : ℝ := Real.cos (Real.cos x)

noncomputable
def g (x : ℝ) : ℝ := Real.sin (Real.sin x)

theorem compare_magnitudes : ∀ x : ℝ, f x > g x :=
by
  sorry

end compare_magnitudes_l144_144917


namespace find_g2_l144_144018

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def even_function (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = g x

theorem find_g2 {f g : ℝ → ℝ}
  (h1 : odd_function f)
  (h2 : even_function g)
  (h3 : ∀ x : ℝ, f x + g x = 2^x) :
  g 2 = 17 / 8 :=
sorry

end find_g2_l144_144018


namespace candy_distribution_l144_144283

theorem candy_distribution (n k : ℕ) (h1 : 3 < n) (h2 : n < 15) (h3 : 195 - n * k = 8) : k = 17 :=
  by
    sorry

end candy_distribution_l144_144283


namespace find_constants_and_calculate_result_l144_144353

theorem find_constants_and_calculate_result :
  ∃ (a b : ℤ), 
    (∀ (x : ℤ), (x + a) * (x + 6) = x^2 + 8 * x + 12) ∧ 
    (∀ (x : ℤ), (x - a) * (x + b) = x^2 + x - 6) ∧ 
    (∀ (x : ℤ), (x + a) * (x + b) = x^2 + 5 * x + 6) :=
by
  sorry

end find_constants_and_calculate_result_l144_144353


namespace NewYearSeasonMarkup_theorem_l144_144678

def NewYearSeasonMarkup (C N : ℝ) : Prop :=
    (0.90 * (1.20 * C * (1 + N)) = 1.35 * C) -> N = 0.25

theorem NewYearSeasonMarkup_theorem (C : ℝ) (h₀ : C > 0) : ∃ (N : ℝ), NewYearSeasonMarkup C N :=
by
  use 0.25
  sorry

end NewYearSeasonMarkup_theorem_l144_144678


namespace pipe_filling_time_l144_144919

-- Definitions for the conditions
variables (A : ℝ) (h : 1 / A - 1 / 24 = 1 / 12)

-- The statement of the problem
theorem pipe_filling_time : A = 8 :=
by
  sorry

end pipe_filling_time_l144_144919


namespace wall_building_time_l144_144191

variables (f b c y : ℕ) 

theorem wall_building_time :
  (y = 2 * f * c / b) 
  ↔ 
  (f > 0 ∧ b > 0 ∧ c > 0 ∧ (f * b * y = 2 * b * c)) := 
sorry

end wall_building_time_l144_144191


namespace vasya_can_interfere_with_petya_goal_l144_144305

theorem vasya_can_interfere_with_petya_goal :
  ∃ (evens odds : ℕ), evens + odds = 50 ∧ (evens + odds) % 2 = 1 :=
sorry

end vasya_can_interfere_with_petya_goal_l144_144305


namespace combinations_x_eq_2_or_8_l144_144091

theorem combinations_x_eq_2_or_8 (x : ℕ) (h_pos : 0 < x) (h_comb : Nat.choose 10 x = Nat.choose 10 2) : x = 2 ∨ x = 8 :=
sorry

end combinations_x_eq_2_or_8_l144_144091


namespace regular_soda_count_l144_144844

theorem regular_soda_count 
  (diet_soda : ℕ) 
  (additional_soda : ℕ) 
  (h1 : diet_soda = 19) 
  (h2 : additional_soda = 41) 
  : diet_soda + additional_soda = 60 :=
by
  sorry

end regular_soda_count_l144_144844


namespace overlap_32_l144_144866

section
variables (t : ℝ)
def position_A : ℝ := 120 - 50 * t
def position_B : ℝ := 220 - 50 * t
def position_N : ℝ := 30 * t - 30
def position_M : ℝ := 30 * t + 10

theorem overlap_32 :
  (∃ t : ℝ, (30 * t + 10 - (120 - 50 * t) = 32) ∨ 
            (-50 * t + 220 - (30 * t - 30) = 32)) ↔
  (t = 71 / 40 ∨ t = 109 / 40) :=
sorry
end

end overlap_32_l144_144866


namespace A_alone_work_days_l144_144770

noncomputable def A_and_B_together : ℕ := 40
noncomputable def A_and_B_worked_together_days : ℕ := 10
noncomputable def B_left_and_C_joined_after_days : ℕ := 6
noncomputable def A_and_C_finish_remaining_work_days : ℕ := 15
noncomputable def C_alone_work_days : ℕ := 60

theorem A_alone_work_days (h1 : A_and_B_together = 40)
                          (h2 : A_and_B_worked_together_days = 10)
                          (h3 : B_left_and_C_joined_after_days = 6)
                          (h4 : A_and_C_finish_remaining_work_days = 15)
                          (h5 : C_alone_work_days = 60) : ∃ (n : ℕ), n = 30 :=
by {
  sorry -- Proof goes here
}

end A_alone_work_days_l144_144770


namespace neon_signs_blink_together_l144_144059

theorem neon_signs_blink_together (a b : ℕ) (ha : a = 9) (hb : b = 15) : Nat.lcm a b = 45 := by
  rw [ha, hb]
  have : Nat.lcm 9 15 = 45 := by sorry
  exact this

end neon_signs_blink_together_l144_144059


namespace line_eq_l144_144011

theorem line_eq (x y : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ), x1 = 5 ∧ y1 = 0 ∧ x2 = 2 ∧ y2 = -5 ∧
    (y - y1) / (x - x1) = (y2 - y1) / (x2 - x1)) →
  5 * x - 3 * y - 25 = 0 :=
sorry

end line_eq_l144_144011


namespace inverse_proportion_l144_144231

variable {x y x1 x2 y1 y2 : ℝ}
variable {k : ℝ}

theorem inverse_proportion {h1 : x1 ≠ 0} {h2 : x2 ≠ 0} {h3 : y1 ≠ 0} {h4 : y2 ≠ 0}
  (h5 : (∃ k, ∀ (x y : ℝ), x * y = k))
  (h6 : x1 / x2 = 4 / 5) : 
  y1 / y2 = 5 / 4 :=
sorry

end inverse_proportion_l144_144231


namespace satisfy_eq_pairs_l144_144110

theorem satisfy_eq_pairs (x y : ℤ) : (x^2 = y^2 + 2 * y + 13) ↔ (x = 4 ∧ (y = 1 ∨ y = -3) ∨ x = -4 ∧ (y = 1 ∨ y = -3)) :=
by
  sorry

end satisfy_eq_pairs_l144_144110


namespace frog_jumps_further_l144_144227

-- Given conditions
def grasshopper_jump : ℕ := 9 -- The grasshopper jumped 9 inches
def frog_jump : ℕ := 12 -- The frog jumped 12 inches

-- Proof statement
theorem frog_jumps_further : frog_jump - grasshopper_jump = 3 := by
  sorry

end frog_jumps_further_l144_144227


namespace current_job_wage_l144_144807

variable (W : ℝ) -- Maisy's wage per hour at her current job

-- Define the conditions
def current_job_hours : ℝ := 8
def new_job_hours : ℝ := 4
def new_job_wage_per_hour : ℝ := 15
def new_job_bonus : ℝ := 35
def additional_new_job_earnings : ℝ := 15

-- Assert the given condition
axiom job_earnings_condition : 
  new_job_hours * new_job_wage_per_hour + new_job_bonus 
  = current_job_hours * W + additional_new_job_earnings

-- The theorem we want to prove
theorem current_job_wage : W = 10 := by
  sorry

end current_job_wage_l144_144807


namespace fraction_of_students_who_walk_home_l144_144966

theorem fraction_of_students_who_walk_home (bus auto bikes scooters : ℚ) 
  (hbus : bus = 2/5) (hauto : auto = 1/5) 
  (hbikes : bikes = 1/10) (hscooters : scooters = 1/10) : 
  1 - (bus + auto + bikes + scooters) = 1/5 :=
by 
  rw [hbus, hauto, hbikes, hscooters]
  sorry

end fraction_of_students_who_walk_home_l144_144966


namespace probability_A_will_receive_2_awards_l144_144296

def classes := Fin 4
def awards := 8

-- The number of ways to distribute 4 remaining awards to 4 classes
noncomputable def total_distributions : ℕ :=
  Nat.choose (awards - 4 + 4 - 1) (4 - 1)

-- The number of ways when class A receives exactly 2 awards
noncomputable def favorable_distributions : ℕ :=
  Nat.choose (2 + 3 - 1) (4 - 1)

-- The probability that class A receives exactly 2 out of 8 awards
noncomputable def probability_A_receives_2_awards : ℚ :=
  favorable_distributions / total_distributions

theorem probability_A_will_receive_2_awards :
  probability_A_receives_2_awards = 2 / 7 := by
  sorry

end probability_A_will_receive_2_awards_l144_144296


namespace largest_possible_sum_l144_144810

theorem largest_possible_sum :
  let a := 12
  let b := 6
  let c := 6
  let d := 12
  a + b = c + d ∧ a + b + 15 = 33 :=
by
  have h1 : 12 + 6 = 6 + 12 := by norm_num
  have h2 : 12 + 6 + 15 = 33 := by norm_num
  exact ⟨h1, h2⟩

end largest_possible_sum_l144_144810


namespace initial_geese_count_l144_144615

-- Define the number of geese that flew away
def geese_flew_away : ℕ := 28

-- Define the number of geese left in the field
def geese_left : ℕ := 23

-- Prove that the initial number of geese in the field was 51
theorem initial_geese_count : geese_left + geese_flew_away = 51 := by
  sorry

end initial_geese_count_l144_144615


namespace gcd_45_75_l144_144269

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l144_144269


namespace sum_ages_l144_144609

variables (uncle_age eunji_age yuna_age : ℕ)

def EunjiAge (uncle_age : ℕ) := uncle_age - 25
def YunaAge (eunji_age : ℕ) := eunji_age + 3

theorem sum_ages (h_uncle : uncle_age = 41) (h_eunji : EunjiAge uncle_age = eunji_age) (h_yuna : YunaAge eunji_age = yuna_age) :
  eunji_age + yuna_age = 35 :=
sorry

end sum_ages_l144_144609


namespace geometric_seq_not_sufficient_necessary_l144_144390

theorem geometric_seq_not_sufficient_necessary (a_n : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a_n (n+1) = a_n n * q) : 
  ¬ ((∃ q > 1, ∀ n, a_n (n+1) > a_n n) ∧ (∀ q > 1, ∀ n, a_n (n+1) > a_n n)) := 
sorry

end geometric_seq_not_sufficient_necessary_l144_144390


namespace maximum_n_for_positive_S_l144_144985

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
n * (a 1 + a n) / 2

theorem maximum_n_for_positive_S
  (a : ℕ → ℝ)
  (d : ℝ)
  (S : ℕ → ℝ)
  (a1_pos : a 1 > 0)
  (d_neg : d < 0)
  (S4_eq_S8 : S 4 = S 8)
  (h1 : is_arithmetic_sequence a d)
  (h2 : ∀ n, S n = sum_of_first_n_terms a n) :
  ∃ n, ∀ m, m ≤ n → S m > 0 ∧ ∀ k, k > n → S k ≤ 0 ∧ n = 11 :=
sorry

end maximum_n_for_positive_S_l144_144985


namespace tank_volume_ratio_l144_144931

theorem tank_volume_ratio (A B : ℝ) 
    (h : (3 / 4) * A = (5 / 8) * B) : A / B = 6 / 5 := 
by 
  sorry

end tank_volume_ratio_l144_144931


namespace geom_seq_m_equals_11_l144_144929

noncomputable def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) :=
  ∀ (n : ℕ), a n = a1 * q ^ n

theorem geom_seq_m_equals_11 {a : ℕ → ℝ} {q : ℝ} (hq : q ≠ 1) 
  (h : geometric_sequence a 1 q) : 
  a 11 = a 1 * a 2 * a 3 * a 4 * a 5 := 
by sorry

end geom_seq_m_equals_11_l144_144929


namespace positive_difference_sum_even_odd_l144_144368

theorem positive_difference_sum_even_odd :
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1)) / 2
  let sum_first_n_odd (n : ℕ) := n * n
  let sum_30_even := sum_first_n_even 30
  let sum_25_odd := sum_first_n_odd 25
  sum_30_even - sum_25_odd = 305 :=
by
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1)) / 2
  let sum_first_n_odd (n : ℕ) := n * n
  let sum_30_even := sum_first_n_even 30
  let sum_25_odd := sum_first_n_odd 25
  show sum_30_even - sum_25_odd = 305
  sorry

end positive_difference_sum_even_odd_l144_144368


namespace estimate_correctness_l144_144753

noncomputable def total_species_estimate (A B C : ℕ) : Prop :=
  A = 2400 ∧ B = 1440 ∧ C = 3600

theorem estimate_correctness (A B C taggedA taggedB taggedC caught : ℕ) 
  (h1 : taggedA = 40) 
  (h2 : taggedB = 40) 
  (h3 : taggedC = 40)
  (h4 : caught = 180)
  (h5 : 3 * A = taggedA * caught) 
  (h6 : 5 * B = taggedB * caught) 
  (h7 : 2 * C = taggedC * caught) 
  : total_species_estimate A B C := 
by
  sorry

end estimate_correctness_l144_144753


namespace max_m_for_factored_polynomial_l144_144243

theorem max_m_for_factored_polynomial :
  ∃ m, (∀ A B : ℤ, (5 * x ^ 2 + m * x + 45 = (5 * x + A) * (x + B) → AB = 45) → 
    m = 226) :=
sorry

end max_m_for_factored_polynomial_l144_144243


namespace simplify_polynomial_l144_144760

-- Define the original polynomial
def original_expr (x : ℝ) : ℝ := 3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 + 2 * x^3

-- Define the simplified version of the polynomial
def simplified_expr (x : ℝ) : ℝ := 2 * x^3 - x^2 + 23 * x - 3

-- State the theorem that the original expression is equal to the simplified one
theorem simplify_polynomial (x : ℝ) : original_expr x = simplified_expr x := 
by 
  sorry

end simplify_polynomial_l144_144760


namespace pam_bags_count_l144_144754

noncomputable def geralds_bag_apples : ℕ := 40

noncomputable def pams_bag_apples := 3 * geralds_bag_apples

noncomputable def pams_total_apples : ℕ := 1200

theorem pam_bags_count : pams_total_apples / pams_bag_apples = 10 := by 
  sorry

end pam_bags_count_l144_144754


namespace diophantine_solution_l144_144542

theorem diophantine_solution (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (n : ℕ) (h_n : n > a * b) :
  ∃ x y : ℕ, n = a * x + b * y :=
by
  sorry

end diophantine_solution_l144_144542


namespace evaluate_expression_l144_144355

theorem evaluate_expression :
  2 * 7^(-1/3 : ℝ) + (1/2 : ℝ) * Real.log (1/64) / Real.log 2 = -3 := 
  sorry

end evaluate_expression_l144_144355


namespace total_chickens_l144_144122

theorem total_chickens (ducks geese : ℕ) (hens roosters chickens: ℕ) :
  ducks = 45 → geese = 28 →
  hens = ducks - 13 → roosters = geese + 9 →
  chickens = hens + roosters →
  chickens = 69 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_chickens_l144_144122


namespace tan_alpha_plus_pi_div4_sin2alpha_over_expr_l144_144087

variables (α : ℝ) (h : Real.tan α = 3)

-- Problem 1
theorem tan_alpha_plus_pi_div4 : Real.tan (α + π / 4) = -2 :=
by
  sorry

-- Problem 2
theorem sin2alpha_over_expr : (Real.sin (2 * α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2 * α) - 1) = 3 / 5 :=
by
  sorry

end tan_alpha_plus_pi_div4_sin2alpha_over_expr_l144_144087


namespace problem1_solution_set_problem2_min_value_l144_144135

-- For Problem (1)
def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

theorem problem1_solution_set (x : ℝ) (h : f x 1 1 ≤ 4) : 
  -2 ≤ x ∧ x ≤ 2 :=
sorry

-- For Problem (2)
theorem problem2_min_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : ∀ x : ℝ, f x a b ≥ 2) : 
  (1 / a) + (2 / b) = 3 :=
sorry

end problem1_solution_set_problem2_min_value_l144_144135


namespace resistor_value_l144_144333

-- Definitions based on given conditions
def U : ℝ := 9 -- Volt reading by the voltmeter
def I : ℝ := 2 -- Current reading by the ammeter
def U_total : ℝ := 2 * U -- Total voltage in the series circuit

-- Stating the theorem
theorem resistor_value (R₀ : ℝ) :
  (U_total = I * (2 * R₀)) → R₀ = 9 :=
by
  intro h
  sorry

end resistor_value_l144_144333


namespace fixed_monthly_fee_l144_144405

theorem fixed_monthly_fee (x y : ℝ)
  (h1 : x + y = 15.80)
  (h2 : x + 3 * y = 28.62) :
  x = 9.39 :=
sorry

end fixed_monthly_fee_l144_144405


namespace total_marks_more_than_physics_l144_144988

-- Definitions of variables for marks in different subjects
variables (P C M : ℕ)

-- Conditions provided in the problem
def total_marks_condition (P : ℕ) (C : ℕ) (M : ℕ) : Prop := P + C + M > P
def average_chemistry_math_marks (C : ℕ) (M : ℕ) : Prop := (C + M) / 2 = 55

-- The main proof statement: Proving the difference in total marks and physics marks
theorem total_marks_more_than_physics 
    (h1 : total_marks_condition P C M)
    (h2 : average_chemistry_math_marks C M) :
  (P + C + M) - P = 110 := 
sorry

end total_marks_more_than_physics_l144_144988


namespace fraction_equality_l144_144585

theorem fraction_equality 
  (a b c d : ℝ)
  (h1 : a + c = 2 * b)
  (h2 : 2 * b * d = c * (b + d))
  (hb : b ≠ 0)
  (hd : d ≠ 0) :
  a / b = c / d :=
sorry

end fraction_equality_l144_144585


namespace travel_cost_AB_l144_144251

theorem travel_cost_AB
  (distance_AB : ℕ)
  (booking_fee : ℕ)
  (cost_per_km_flight : ℝ)
  (correct_total_cost : ℝ)
  (h1 : distance_AB = 4000)
  (h2 : booking_fee = 150)
  (h3 : cost_per_km_flight = 0.12) :
  correct_total_cost = 630 :=
by
  sorry

end travel_cost_AB_l144_144251


namespace cameron_speed_ratio_l144_144002

variables (C Ch : ℝ)
-- Danielle's speed is three times Cameron's speed
def Danielle_speed := 3 * C
-- Danielle's travel time from Granville to Salisbury is 30 minutes
def Danielle_time := 30
-- Chase's travel time from Granville to Salisbury is 180 minutes
def Chase_time := 180

-- Prove the ratio of Cameron's speed to Chase's speed is 2
theorem cameron_speed_ratio :
  (Danielle_speed C / Ch) = (Chase_time / Danielle_time) → (C / Ch) = 2 :=
by {
  sorry
}

end cameron_speed_ratio_l144_144002


namespace max_value_inequality_l144_144517

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5 * x + 3 * y < 90) : 
  x * y * (90 - 5 * x - 3 * y) ≤ 1800 := 
sorry

end max_value_inequality_l144_144517


namespace plywood_cut_difference_l144_144046

theorem plywood_cut_difference :
  let original_width := 6
  let original_height := 9
  let total_area := original_width * original_height
  let num_pieces := 6
  let area_per_piece := total_area / num_pieces
  -- Let possible perimeters based on given conditions
  let max_perimeter := 20
  let min_perimeter := 15
  max_perimeter - min_perimeter = 5 :=
by
  sorry

end plywood_cut_difference_l144_144046


namespace complement_correct_l144_144378

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}

theorem complement_correct : (U \ A) = {2, 4} := by
  sorry

end complement_correct_l144_144378


namespace min_people_liking_both_l144_144853

theorem min_people_liking_both (A B C V : ℕ) (hA : A = 200) (hB : B = 150) (hC : C = 120) (hV : V = 80) :
  ∃ D, D = 80 ∧ D ≤ min B (A - C + V) :=
by {
  sorry
}

end min_people_liking_both_l144_144853


namespace length_of_other_leg_l144_144761

theorem length_of_other_leg (c a b : ℕ) (h1 : c = 10) (h2 : a = 6) (h3 : c^2 = a^2 + b^2) : b = 8 :=
by
  sorry

end length_of_other_leg_l144_144761


namespace length_of_platform_l144_144306

-- Definitions for the given conditions
def speed_of_train_kmph : ℕ := 54
def speed_of_train_mps : ℕ := 15
def time_to_pass_platform : ℕ := 16
def time_to_pass_man : ℕ := 10

-- Main statement of the problem
theorem length_of_platform (v_kmph : ℕ) (v_mps : ℕ) (t_p : ℕ) (t_m : ℕ) 
    (h1 : v_kmph = 54) 
    (h2 : v_mps = 15) 
    (h3 : t_p = 16) 
    (h4 : t_m = 10) : 
    v_mps * t_p - v_mps * t_m = 90 := 
sorry

end length_of_platform_l144_144306


namespace cone_cylinder_volume_ratio_l144_144383

theorem cone_cylinder_volume_ratio (h_cyl r_cyl: ℝ) (h_cone: ℝ) :
  h_cyl = 10 → r_cyl = 5 → h_cone = 5 →
  (1/3 * (Real.pi * r_cyl^2 * h_cone)) / (Real.pi * r_cyl^2 * h_cyl) = 1/6 :=
by
  intros h_cyl_eq r_cyl_eq h_cone_eq
  rw [h_cyl_eq, r_cyl_eq, h_cone_eq]
  sorry

end cone_cylinder_volume_ratio_l144_144383


namespace clock_hand_overlaps_in_24_hours_l144_144926

-- Define the number of revolutions of the hour hand in 24 hours.
def hour_hand_revolutions_24_hours : ℕ := 2

-- Define the number of revolutions of the minute hand in 24 hours.
def minute_hand_revolutions_24_hours : ℕ := 24

-- Define the number of overlaps as a constant.
def number_of_overlaps (hour_rev : ℕ) (minute_rev : ℕ) : ℕ :=
  minute_rev - hour_rev

-- The theorem we want to prove:
theorem clock_hand_overlaps_in_24_hours :
  number_of_overlaps hour_hand_revolutions_24_hours minute_hand_revolutions_24_hours = 22 :=
sorry

end clock_hand_overlaps_in_24_hours_l144_144926


namespace volume_eq_eight_space_diagonal_eq_sqrt21_surface_area_neq_24_circumscribed_sphere_area_eq_21pi_l144_144782

namespace RectangularPrism

def length := 4
def width := 2
def height := 1

theorem volume_eq_eight : length * width * height = 8 := sorry

theorem space_diagonal_eq_sqrt21 :
  Real.sqrt (length ^ 2 + width ^ 2 + height ^ 2) = Real.sqrt 21 := sorry

theorem surface_area_neq_24 :
  2 * (length * width + width * height + height * length) ≠ 24 := sorry

theorem circumscribed_sphere_area_eq_21pi :
  4 * Real.pi * ((Real.sqrt (length ^ 2 + width ^ 2 + height ^ 2) / 2) ^ 2) = 21 * Real.pi := sorry

end RectangularPrism

end volume_eq_eight_space_diagonal_eq_sqrt21_surface_area_neq_24_circumscribed_sphere_area_eq_21pi_l144_144782


namespace eccentricity_of_ellipse_l144_144085

-- Definitions
variable (a b c : ℝ)  -- semi-major axis, semi-minor axis, and distance from center to a focus
variable (h_c_eq_b : c = b)  -- given condition focal length equals length of minor axis
variable (h_a_eq_sqrt_sum : a = Real.sqrt (c^2 + b^2))  -- relationship in ellipse

-- Question: Prove the eccentricity of the ellipse e = √2 / 2
theorem eccentricity_of_ellipse : (c = b) → (a = Real.sqrt (c^2 + b^2)) → (c / a = Real.sqrt 2 / 2) :=
by
  intros h_c_eq_b h_a_eq_sqrt_sum
  sorry

end eccentricity_of_ellipse_l144_144085


namespace function_monotonically_increasing_on_interval_l144_144908

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem function_monotonically_increasing_on_interval (e : ℝ) (h_e_pos : 0 < e) (h_ln_e_pos : 0 < Real.log e) :
  ∀ x : ℝ, e < x → 0 < Real.log x - 1 := 
sorry

end function_monotonically_increasing_on_interval_l144_144908


namespace geometric_sequence_a6_l144_144042

theorem geometric_sequence_a6 (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 + a 3 = 5 / 2) (h2 : a 2 + a 4 = 5 / 4) 
  (h3 : ∀ n, a (n + 1) = a n * q) : a 6 = 1 / 16 :=
by
  sorry

end geometric_sequence_a6_l144_144042


namespace minimum_value_ineq_l144_144430

variable (m n : ℝ)

noncomputable def minimum_value := (1 / (2 * m)) + (1 / n)

theorem minimum_value_ineq (h1 : m > 0) (h2 : n > 0) (h3 : m + 2 * n = 1) : minimum_value m n = 9 / 2 := 
sorry

end minimum_value_ineq_l144_144430


namespace probability_red_or_green_l144_144769

variable (P_brown P_purple P_green P_red P_yellow : ℝ)

def conditions : Prop :=
  P_brown = 0.3 ∧
  P_brown = 3 * P_purple ∧
  P_green = P_purple ∧
  P_red = P_yellow ∧
  P_brown + P_purple + P_green + P_red + P_yellow = 1

theorem probability_red_or_green (h : conditions P_brown P_purple P_green P_red P_yellow) :
  P_red + P_green = 0.35 :=
by
  sorry

end probability_red_or_green_l144_144769


namespace arithmetic_sequence_first_term_l144_144794

theorem arithmetic_sequence_first_term
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (k : ℕ) (hk : k ≥ 2)
  (hS : S k = 5)
  (ha_k2_p1 : a (k^2 + 1) = -45)
  (ha_sum : (Finset.range (2 * k + 1) \ Finset.range (k + 1)).sum a = -45) :
  a 1 = 5 := 
sorry

end arithmetic_sequence_first_term_l144_144794


namespace percentage_running_wickets_l144_144840

-- Conditions provided as definitions and assumptions in Lean
def total_runs : ℕ := 120
def boundaries : ℕ := 3
def sixes : ℕ := 8
def boundary_runs (b : ℕ) := b * 4
def six_runs (s : ℕ) := s * 6

-- Calculate runs from boundaries and sixes
def runs_from_boundaries := boundary_runs boundaries
def runs_from_sixes := six_runs sixes
def runs_not_from_boundaries_and_sixes := total_runs - (runs_from_boundaries + runs_from_sixes)

-- Proof that the percentage of the total score by running between the wickets is 50%
theorem percentage_running_wickets :
  (runs_not_from_boundaries_and_sixes : ℝ) / (total_runs : ℝ) * 100 = 50 :=
by
  sorry

end percentage_running_wickets_l144_144840


namespace horizontal_distance_is_0_65_l144_144244

def parabola (x : ℝ) : ℝ := 2 * x^2 - 3 * x - 4

-- Calculate the horizontal distance between two points on the parabola given their y-coordinates and prove it equals to 0.65
theorem horizontal_distance_is_0_65 :
  ∃ (x1 x2 : ℝ), 
    parabola x1 = 10 ∧ parabola x2 = 0 ∧ abs (x1 - x2) = 0.65 :=
sorry

end horizontal_distance_is_0_65_l144_144244


namespace problem_1_problem_2_l144_144590

-- Problem 1 statement
theorem problem_1 (a x : ℝ) (m : ℝ) (h_pos_a : a > 0) (h_cond_a : a = 1/4) (h_cond_q : (1 : ℝ) / 2 < x ∧ x < 1) (h_cond_p : a < x ∧ x < 3 * a): 1 / 2 < x ∧ x < 3 / 4 :=
by sorry

-- Problem 2 statement
theorem problem_2 (a x : ℝ) (m : ℝ) (h_pos_a : a > 0) (h_neg_p : ¬(a < x ∧ x < 3 * a)) (h_neg_q : ¬((1 / (2 : ℝ))^(m - 1) < x ∧ x < 1)): 1 / 3 ≤ a ∧ a ≤ 1 / 2 :=
by sorry

end problem_1_problem_2_l144_144590


namespace fraction_in_range_l144_144798

theorem fraction_in_range : 
  (2:ℝ) / 5 < (4:ℝ) / 7 ∧ (4:ℝ) / 7 < 3 / 4 := by
  sorry

end fraction_in_range_l144_144798


namespace sum_of_roots_l144_144658

theorem sum_of_roots (f : ℝ → ℝ) (h_symmetric : ∀ x, f (3 + x) = f (3 - x)) (h_roots : ∃ (roots : Finset ℝ), roots.card = 6 ∧ ∀ r ∈ roots, f r = 0) : 
  ∃ S, S = 18 :=
by
  sorry

end sum_of_roots_l144_144658


namespace max_area_of_rectangular_fence_l144_144848

theorem max_area_of_rectangular_fence (x y : ℕ) (h : x + y = 75) : 
  (x * (75 - x) ≤ 1406) ∧ (∀ x' y', x' + y' = 75 → x' * y' ≤ 1406) :=
by
  sorry

end max_area_of_rectangular_fence_l144_144848


namespace chocolates_difference_l144_144401

/-!
We are given that:
- Robert ate 10 chocolates
- Nickel ate 5 chocolates

We need to prove that Robert ate 5 more chocolates than Nickel.
-/

def robert_chocolates := 10
def nickel_chocolates := 5

theorem chocolates_difference : robert_chocolates - nickel_chocolates = 5 :=
by
  -- Proof is omitted as per instructions
  sorry

end chocolates_difference_l144_144401


namespace number_of_ways_to_choose_students_l144_144205

theorem number_of_ways_to_choose_students :
  let female_students := 4
  let male_students := 3
  (female_students * male_students) = 12 :=
by
  sorry

end number_of_ways_to_choose_students_l144_144205


namespace total_job_applications_l144_144888

theorem total_job_applications (apps_in_state : ℕ) (apps_other_states : ℕ) 
  (h1 : apps_in_state = 200)
  (h2 : apps_other_states = 2 * apps_in_state) :
  apps_in_state + apps_other_states = 600 :=
by
  sorry

end total_job_applications_l144_144888


namespace deepak_present_age_l144_144860

def rahul_age (x : ℕ) : ℕ := 4 * x
def deepak_age (x : ℕ) : ℕ := 3 * x

theorem deepak_present_age (x : ℕ) (h1 : rahul_age x + 10 = 26) : deepak_age x = 12 :=
by sorry

end deepak_present_age_l144_144860


namespace smallest_sum_symmetrical_dice_l144_144171

theorem smallest_sum_symmetrical_dice (p : ℝ) (N : ℕ) (h₁ : p > 0) (h₂ : 6 * N = 2022) : N = 337 := 
by
  -- Proof can be filled in here
  sorry

end smallest_sum_symmetrical_dice_l144_144171


namespace m_squared_minus_n_squared_plus_one_is_perfect_square_l144_144935

theorem m_squared_minus_n_squared_plus_one_is_perfect_square (m n : ℤ)
  (hm : m % 2 = 1) (hn : n % 2 = 1)
  (h : m^2 - n^2 + 1 ∣ n^2 - 1) :
  ∃ k : ℤ, k^2 = m^2 - n^2 + 1 :=
sorry

end m_squared_minus_n_squared_plus_one_is_perfect_square_l144_144935


namespace evaluate_expression_l144_144486

theorem evaluate_expression :
  let a := 3^1005
  let b := 4^1006
  (a + b)^2 - (a - b)^2 = 160 * 10^1004 :=
by
  sorry

end evaluate_expression_l144_144486


namespace smallest_k_square_divisible_l144_144526

theorem smallest_k_square_divisible (k : ℤ) (n : ℤ) (h1 : k = 60)
    (h2 : ∀ m : ℤ, m < k → ∃ d : ℤ, d ∣ (k^2) → m = d ) : n = 3600 :=
sorry

end smallest_k_square_divisible_l144_144526


namespace age_difference_l144_144530

theorem age_difference (A B : ℕ) (h1 : B = 39) (h2 : A + 10 = 2 * (B - 10)) : A - B = 9 := by
  sorry

end age_difference_l144_144530


namespace find_d1_l144_144855

noncomputable def E (n : ℕ) : ℕ := sorry

theorem find_d1 :
  ∃ (d4 d3 d2 d0 : ℤ), 
  (∀ (n : ℕ), n ≥ 4 ∧ n % 2 = 0 → 
     E n = d4 * n^4 + d3 * n^3 + d2 * n^2 + (12 : ℤ) * n + d0) :=
sorry

end find_d1_l144_144855


namespace molecular_weight_AlPO4_correct_l144_144783

-- Noncomputable because we are working with specific numerical values.
noncomputable def atomic_weight_Al : ℝ := 26.98
noncomputable def atomic_weight_P : ℝ := 30.97
noncomputable def atomic_weight_O : ℝ := 16.00

noncomputable def molecular_weight_AlPO4 : ℝ := 
  (1 * atomic_weight_Al) + (1 * atomic_weight_P) + (4 * atomic_weight_O)

theorem molecular_weight_AlPO4_correct : molecular_weight_AlPO4 = 121.95 := by
  sorry

end molecular_weight_AlPO4_correct_l144_144783


namespace cylinder_height_relation_l144_144496

variables (r1 h1 r2 h2 : ℝ)
variables (V1_eq_V2 : π * r1^2 * h1 = π * r2^2 * h2) (r2_eq_1_2_r1 : r2 = 1.2 * r1)

theorem cylinder_height_relation : h1 = 1.44 * h2 :=
by
  sorry

end cylinder_height_relation_l144_144496


namespace perpendicularity_proof_l144_144858

-- Definitions of geometric entities and properties
variable (Plane Line : Type)
variable (α β : Plane) -- α and β are planes
variable (m n : Line) -- m and n are lines

-- Geometric properties and relations
variable (subset : Line → Plane → Prop) -- Line is subset of plane
variable (perpendicular : Line → Plane → Prop) -- Line is perpendicular to plane
variable (line_perpendicular : Line → Line → Prop) -- Line is perpendicular to another line

-- Conditions
axiom planes_different : α ≠ β
axiom lines_different : m ≠ n
axiom m_in_beta : subset m β
axiom n_in_beta : subset n β

-- Proof problem statement
theorem perpendicularity_proof :
  (subset m α) → (perpendicular n α) → (line_perpendicular n m) :=
by
  sorry

end perpendicularity_proof_l144_144858


namespace ratio_of_areas_l144_144409

theorem ratio_of_areas (b : ℝ) (h1 : 0 < b) (h2 : b < 4) 
  (h3 : (9 : ℝ) / 25 = (4 - b) / b * (4 : ℝ)) : b = 2.5 := 
sorry

end ratio_of_areas_l144_144409


namespace dwarfs_truthful_count_l144_144954

theorem dwarfs_truthful_count : ∃ (x y : ℕ), x + y = 10 ∧ x + 2 * y = 16 ∧ x = 4 := by
  sorry

end dwarfs_truthful_count_l144_144954


namespace grid_values_equal_l144_144394

theorem grid_values_equal (f : ℤ × ℤ → ℕ) (h : ∀ (i j : ℤ), 
  f (i, j) = 1 / 4 * (f (i + 1, j) + f (i - 1, j) + f (i, j + 1) + f (i, j - 1))) :
  ∀ (i j i' j' : ℤ), f (i, j) = f (i', j') :=
by
  sorry

end grid_values_equal_l144_144394


namespace factorization_correct_l144_144299

theorem factorization_correct (x : ℝ) :
  16 * x ^ 2 + 8 * x - 24 = 8 * (2 * x ^ 2 + x - 3) ∧ (2 * x ^ 2 + x - 3) = (2 * x + 3) * (x - 1) :=
by
  sorry

end factorization_correct_l144_144299


namespace Q_neither_necessary_nor_sufficient_l144_144103

-- Define the propositions P and Q
def PropositionP (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  ∀ x : ℝ, (a1*x^2 + b1*x + c1 > 0) ↔ (a2*x^2 + b2*x + c2 > 0)

def PropositionQ (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (a1 / a2 = b1 / b2) ∧ (b1 / b2 = c1 / c2)

-- The final statement to prove that Q is neither necessary nor sufficient for P
theorem Q_neither_necessary_nor_sufficient (a1 b1 c1 a2 b2 c2 : ℝ) :
  ¬ ((PropositionQ a1 b1 c1 a2 b2 c2) ↔ (PropositionP a1 b1 c1 a2 b2 c2)) := sorry

end Q_neither_necessary_nor_sufficient_l144_144103


namespace point_in_third_quadrant_l144_144635

theorem point_in_third_quadrant :
  let sin2018 := Real.sin (2018 * Real.pi / 180)
  let tan117 := Real.tan (117 * Real.pi / 180)
  sin2018 < 0 ∧ tan117 < 0 → 
  (sin2018 < 0 ∧ tan117 < 0) :=
by
  intros
  sorry

end point_in_third_quadrant_l144_144635


namespace complement_union_l144_144507

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {1, 2, 4}
def N : Set ℕ := {2, 3}

theorem complement_union (U : Set ℕ) (M : Set ℕ) (N : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hM : M = {1, 2, 4}) (hN : N = {2, 3}) :
  (U \ M) ∪ N = {0, 2, 3} :=
by
  rw [hU, hM, hN] -- Substitute U, M, N definitions
  sorry -- Proof omitted

end complement_union_l144_144507


namespace calculate_group_A_B_C_and_total_is_correct_l144_144776

def groupA_1week : Int := 175000
def groupA_2week : Int := 107000
def groupA_3week : Int := 35000
def groupB_1week : Int := 100000
def groupB_2week : Int := 70350
def groupB_3week : Int := 19500
def groupC_1week : Int := 45000
def groupC_2week : Int := 87419
def groupC_3week : Int := 14425
def kids_staying_home : Int := 590796
def kids_outside_county : Int := 22

def total_kids_in_A := groupA_1week + groupA_2week + groupA_3week
def total_kids_in_B := groupB_1week + groupB_2week + groupB_3week
def total_kids_in_C := groupC_1week + groupC_2week + groupC_3week
def total_kids_in_camp := total_kids_in_A + total_kids_in_B + total_kids_in_C
def total_kids := total_kids_in_camp + kids_staying_home + kids_outside_county

theorem calculate_group_A_B_C_and_total_is_correct :
  total_kids_in_A = 317000 ∧
  total_kids_in_B = 189850 ∧
  total_kids_in_C = 146844 ∧
  total_kids = 1244512 := by
  sorry

end calculate_group_A_B_C_and_total_is_correct_l144_144776


namespace rectangle_width_is_nine_l144_144791

theorem rectangle_width_is_nine (w l : ℝ) (h1 : l = 2 * w)
  (h2 : l * w = 3 * 2 * (l + w)) : 
  w = 9 :=
by
  sorry

end rectangle_width_is_nine_l144_144791


namespace interest_first_year_l144_144187
-- Import the necessary math library

-- Define the conditions and proof the interest accrued in the first year
theorem interest_first_year :
  ∀ (P B₁ : ℝ) (r₂ increase_ratio: ℝ),
    P = 1000 →
    B₁ = 1100 →
    r₂ = 0.20 →
    increase_ratio = 0.32 →
    (B₁ - P) = 100 :=
by
  intros P B₁ r₂ increase_ratio P_def B₁_def r₂_def increase_ratio_def
  sorry

end interest_first_year_l144_144187


namespace amitabh_avg_expenditure_feb_to_jul_l144_144466

variable (expenditure_avg_jan_to_jun expenditure_jan expenditure_jul : ℕ)

theorem amitabh_avg_expenditure_feb_to_jul (h1 : expenditure_avg_jan_to_jun = 4200) 
  (h2 : expenditure_jan = 1200) (h3 : expenditure_jul = 1500) :
  (expenditure_avg_jan_to_jun * 6 - expenditure_jan + expenditure_jul) / 6 = 4250 := by
  -- Using the given conditions
  sorry

end amitabh_avg_expenditure_feb_to_jul_l144_144466


namespace naomi_drives_to_parlor_l144_144260

theorem naomi_drives_to_parlor (d v t t_back : ℝ)
  (ht : t = d / v)
  (ht_back : t_back = 2 * d / v)
  (h_total : 2 * (t + t_back) = 6) : 
  t = 1 :=
by sorry

end naomi_drives_to_parlor_l144_144260


namespace men_at_conference_l144_144744

theorem men_at_conference (M : ℕ) 
  (num_women : ℕ) (num_children : ℕ)
  (indian_men_fraction : ℚ) (indian_women_fraction : ℚ)
  (indian_children_fraction : ℚ) (non_indian_fraction : ℚ)
  (num_women_eq : num_women = 300)
  (num_children_eq : num_children = 500)
  (indian_men_fraction_eq : indian_men_fraction = 0.10)
  (indian_women_fraction_eq : indian_women_fraction = 0.60)
  (indian_children_fraction_eq : indian_children_fraction = 0.70)
  (non_indian_fraction_eq : non_indian_fraction = 0.5538461538461539) :
  M = 500 :=
by
  sorry

end men_at_conference_l144_144744


namespace lcm_factor_l144_144423

-- Define the variables and conditions
variables (A B H L x : ℕ)
variable (hcf_23 : Nat.gcd A B = 23)
variable (larger_number_391 : A = 391)
variable (lcm_hcf_mult_factors : L = Nat.lcm A B)
variable (lcm_factors : L = 23 * x * 17)

-- The proof statement
theorem lcm_factor (hcf_23 : Nat.gcd A B = 23) (larger_number_391 : A = 391) (lcm_hcf_mult_factors : L = Nat.lcm A B) (lcm_factors : L = 23 * x * 17) :
  x = 17 :=
sorry

end lcm_factor_l144_144423


namespace conference_total_duration_is_715_l144_144108

structure ConferenceSession where
  hours : ℕ
  minutes : ℕ

def totalDuration (s1 s2 : ConferenceSession): ℕ :=
  (s1.hours * 60 + s1.minutes) + (s2.hours * 60 + s2.minutes)

def session1 : ConferenceSession := { hours := 8, minutes := 15 }
def session2 : ConferenceSession := { hours := 3, minutes := 40 }

theorem conference_total_duration_is_715 :
  totalDuration session1 session2 = 715 := 
sorry

end conference_total_duration_is_715_l144_144108


namespace initial_meals_is_70_l144_144060

-- Define variables and conditions
variables (A : ℕ)
def initial_meals_for_adults := A

-- Given conditions
def condition_1 := true  -- Group of 55 adults and some children (not directly used in proving A)
def condition_2 := true  -- Either a certain number of adults or 90 children (implicitly used in equation)
def condition_3 := (A - 21) * (90 / A) = 63  -- 21 adults have their meal, remaining food serves 63 children

-- The proof statement
theorem initial_meals_is_70 (h : (A - 21) * (90 / A) = 63) : A = 70 :=
sorry

end initial_meals_is_70_l144_144060


namespace num_three_digit_integers_with_odd_factors_l144_144004

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l144_144004


namespace percentage_of_respondents_l144_144993

variables {X Y : ℝ}
variable (h₁ : 23 <= 100 - X)

theorem percentage_of_respondents 
  (h₁ : 0 ≤ X) 
  (h₂ : X ≤ 100) 
  (h₃ : 0 ≤ 23) 
  (h₄ : 23 ≤ 23) : 
  Y = 100 - X := 
by
  sorry

end percentage_of_respondents_l144_144993


namespace paula_bought_two_shirts_l144_144274

-- Define the conditions
def total_money : Int := 109
def shirt_cost : Int := 11
def pants_cost : Int := 13
def remaining_money : Int := 74

-- Calculate the expenditure on shirts and pants
def expenditure : Int := total_money - remaining_money

-- Define the number of shirts bought
def number_of_shirts (S : Int) : Prop := expenditure = shirt_cost * S + pants_cost

-- The theorem stating that Paula bought 2 shirts
theorem paula_bought_two_shirts : number_of_shirts 2 :=
by
  -- The proof is omitted as per instructions
  sorry

end paula_bought_two_shirts_l144_144274


namespace smallest_class_number_selected_l144_144179

theorem smallest_class_number_selected
  {n k : ℕ} (hn : n = 30) (hk : k = 5) (h_sum : ∃ x : ℕ, x + (x + 6) + (x + 12) + (x + 18) + (x + 24) = 75) :
  ∃ x : ℕ, x = 3 := 
sorry

end smallest_class_number_selected_l144_144179


namespace chords_even_arcs_even_l144_144550

theorem chords_even_arcs_even (N : ℕ) 
  (h1 : ∀ k : ℕ, k ≤ N → ¬ ((k : ℤ) % 2 = 1)) : 
  N % 2 = 0 := 
sorry

end chords_even_arcs_even_l144_144550


namespace meadow_to_campsite_distance_l144_144014

variable (d1 d2 d_total d_meadow_to_campsite : ℝ)

theorem meadow_to_campsite_distance
  (h1 : d1 = 0.2)
  (h2 : d2 = 0.4)
  (h_total : d_total = 0.7)
  (h_before_meadow : d_before_meadow = d1 + d2)
  (h_distance : d_meadow_to_campsite = d_total - d_before_meadow) :
  d_meadow_to_campsite = 0.1 :=
by 
  sorry

end meadow_to_campsite_distance_l144_144014


namespace eval_complex_fraction_expr_l144_144139

def complex_fraction_expr : ℚ :=
  2 + (3 / (4 + (5 / (6 + (7 / 8)))))

theorem eval_complex_fraction_expr : complex_fraction_expr = 137 / 52 :=
by
  -- we skip the actual proof but ensure it can build successfully.
  sorry

end eval_complex_fraction_expr_l144_144139


namespace pow_15_1234_mod_19_l144_144934

theorem pow_15_1234_mod_19 : (15^1234) % 19 = 6 := 
by sorry

end pow_15_1234_mod_19_l144_144934


namespace a9_proof_l144_144258

variable {a : ℕ → ℝ}

-- Conditions
axiom a1 : a 1 = 1
axiom an_recurrence : ∀ n > 1, a n = (a (n - 1)) * 2^(n - 1)

-- Goal
theorem a9_proof : a 9 = 2^36 := 
by 
  sorry

end a9_proof_l144_144258


namespace inequality_am_gm_l144_144117

variable {a b c : ℝ}

theorem inequality_am_gm (habc_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_abc_eq_1 : a * b * c = 1) : 
  a^3 + b^3 + c^3 + (a * b / (a^2 + b^2) + b * c / (b^2 + c^2) + c * a / (c^2 + a^2)) ≥ 9 / 2 := 
by
  sorry

end inequality_am_gm_l144_144117


namespace statement_c_false_l144_144421

theorem statement_c_false : ¬ ∃ (x y : ℝ), x^2 + y^2 < 0 := by
  sorry

end statement_c_false_l144_144421


namespace abs_neg_five_l144_144439

theorem abs_neg_five : abs (-5) = 5 :=
by
  sorry

end abs_neg_five_l144_144439


namespace total_balls_estimation_l144_144329

theorem total_balls_estimation
  (n : ℕ)  -- Let n be the total number of balls in the bag
  (yellow_balls : ℕ)  -- Let yellow_balls be the number of yellow balls
  (frequency : ℝ)  -- Let frequency be the stabilized frequency of drawing a yellow ball
  (h1 : yellow_balls = 6)
  (h2 : frequency = 0.3)
  (h3 : (yellow_balls : ℝ) / (n : ℝ) = frequency) :
  n = 20 :=
by
  sorry

end total_balls_estimation_l144_144329


namespace jane_mean_score_l144_144815

-- Define the six quiz scores Jane took
def score1 : ℕ := 86
def score2 : ℕ := 91
def score3 : ℕ := 89
def score4 : ℕ := 95
def score5 : ℕ := 88
def score6 : ℕ := 94

-- The number of quizzes
def num_quizzes : ℕ := 6

-- The sum of all quiz scores
def total_score : ℕ := score1 + score2 + score3 + score4 + score5 + score6 

-- The expected mean score
def mean_score : ℚ := 90.5

-- The proof statement
theorem jane_mean_score (h : total_score = 543) : total_score / num_quizzes = mean_score := 
by sorry

end jane_mean_score_l144_144815


namespace ratio_of_red_to_total_simplified_l144_144255

def number_of_red_haired_children := 9
def total_number_of_children := 48

theorem ratio_of_red_to_total_simplified:
  (number_of_red_haired_children: ℚ) / (total_number_of_children: ℚ) = (3 : ℚ) / (16 : ℚ) := 
by
  sorry

end ratio_of_red_to_total_simplified_l144_144255


namespace sequence_term_4_l144_144687

noncomputable def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2 * sequence n

theorem sequence_term_4 : sequence 3 = 8 := 
by
  sorry

end sequence_term_4_l144_144687


namespace inequality_abc_geq_36_l144_144173

theorem inequality_abc_geq_36 (a b c : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) (h_prod : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^6 + 5) ≥ 36 :=
by
  sorry

end inequality_abc_geq_36_l144_144173


namespace sugar_cone_count_l144_144398

theorem sugar_cone_count (ratio_sugar_waffle : ℕ → ℕ → Prop) (sugar_waffle_ratio : ratio_sugar_waffle 5 4) 
(w : ℕ) (h_w : w = 36) : ∃ s : ℕ, ratio_sugar_waffle s w ∧ s = 45 :=
by
  sorry

end sugar_cone_count_l144_144398


namespace jessica_milk_problem_l144_144234

theorem jessica_milk_problem (gallons_owned : ℝ) (gallons_given : ℝ) : gallons_owned = 5 → gallons_given = 16 / 3 → gallons_owned - gallons_given = -(1 / 3) :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  -- sorry

end jessica_milk_problem_l144_144234


namespace vectors_parallel_perpendicular_l144_144193

theorem vectors_parallel_perpendicular (t t1 t2 : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) 
    (h_a : a = (2, t)) (h_b : b = (1, 2)) :
    ((2 * 2 = t * 1) → t1 = 4) ∧ ((2 * 1 + 2 * t = 0) → t2 = -1) :=
by 
  sorry

end vectors_parallel_perpendicular_l144_144193


namespace jewel_price_reduction_l144_144661

theorem jewel_price_reduction (P x : ℝ) (P1 : ℝ) (hx : x ≠ 0) 
  (hP1 : P1 = P * (1 - (x / 100) ^ 2))
  (h_final : P1 * (1 - (x / 100) ^ 2) = 2304) : 
  P1 = 2304 / (1 - (x / 100) ^ 2) :=
by
  sorry

end jewel_price_reduction_l144_144661


namespace sequence_a2002_l144_144446

theorem sequence_a2002 :
  ∀ (a : ℕ → ℕ), (a 1 = 1) → (a 2 = 2) → 
  (∀ n, 2 ≤ n → a (n + 1) = 3 * a n - 2 * a (n - 1)) → 
  a 2002 = 2 ^ 2001 :=
by
  intros a ha1 ha2 hrecur
  sorry

end sequence_a2002_l144_144446


namespace swapped_digits_greater_by_18_l144_144599

theorem swapped_digits_greater_by_18 (x : ℕ) : 
  (10 * x + 1) - (10 + x) = 18 :=
  sorry

end swapped_digits_greater_by_18_l144_144599


namespace binomial_expansion_b_value_l144_144562

theorem binomial_expansion_b_value (a b x : ℝ) (h : (1 + a * x) ^ 5 = 1 + 10 * x + b * x ^ 2 + a^5 * x ^ 5) : b = 40 := 
sorry

end binomial_expansion_b_value_l144_144562


namespace new_group_size_l144_144467

theorem new_group_size (N : ℕ) (h1 : 20 < N) (h2 : N < 50) (h3 : (N - 5) % 6 = 0) (h4 : (N - 5) % 7 = 0) (h5 : (N % (N - 7)) = 7) : (N - 7).gcd (N) = 8 :=
by
  sorry

end new_group_size_l144_144467


namespace max_discount_l144_144879

theorem max_discount (cost_price selling_price : ℝ) (min_profit_margin : ℝ) (x : ℝ) : 
  cost_price = 400 → selling_price = 500 → min_profit_margin = 0.0625 → 
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin * cost_price) → x ≤ 15 :=
by
  intros h1 h2 h3 h4
  sorry

end max_discount_l144_144879


namespace min_n_plus_d_l144_144833

theorem min_n_plus_d (a : ℕ → ℕ) (n d : ℕ) (h1 : a 1 = 1) (h2 : a n = 51)
  (h3 : ∀ i, a i = a 1 + (i-1) * d) : n + d = 16 :=
by
  sorry

end min_n_plus_d_l144_144833


namespace cos_360_eq_one_l144_144479

theorem cos_360_eq_one : Real.cos (2 * Real.pi) = 1 :=
by sorry

end cos_360_eq_one_l144_144479


namespace abc_le_one_eighth_l144_144905

theorem abc_le_one_eighth (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : a * b * c ≤ 1 / 8 :=
by
  sorry

end abc_le_one_eighth_l144_144905


namespace half_vectorAB_is_2_1_l144_144942

def point := ℝ × ℝ -- Define a point as a pair of real numbers
def vector := ℝ × ℝ -- Define a vector as a pair of real numbers

def A : point := (-1, 0) -- Define point A
def B : point := (3, 2) -- Define point B

noncomputable def vectorAB : vector := (B.1 - A.1, B.2 - A.2) -- Define vector AB as B - A

noncomputable def half_vectorAB : vector := (1 / 2 * vectorAB.1, 1 / 2 * vectorAB.2) -- Define half of vector AB

theorem half_vectorAB_is_2_1 : half_vectorAB = (2, 1) := by
  -- Sorry is a placeholder for the proof
  sorry

end half_vectorAB_is_2_1_l144_144942


namespace polynomial_remainder_l144_144930

theorem polynomial_remainder (x : ℂ) : (x^1500) % (x^3 - 1) = 1 := 
sorry

end polynomial_remainder_l144_144930


namespace sol_earnings_in_a_week_l144_144314

-- Define the number of candy bars sold each day using recurrence relation
def candies_sold (n : ℕ) : ℕ :=
  match n with
  | 0     => 10  -- Day 1
  | (n+1) => candies_sold n + 4  -- Each subsequent day

-- Define the total candies sold in a week and total earnings in dollars
def total_candies_sold_in_a_week : ℕ :=
  List.sum (List.map candies_sold [0, 1, 2, 3, 4, 5])

def total_earnings_in_dollars : ℕ :=
  (total_candies_sold_in_a_week * 10) / 100

-- Proving that Sol will earn 12 dollars in a week
theorem sol_earnings_in_a_week : total_earnings_in_dollars = 12 := by
  sorry

end sol_earnings_in_a_week_l144_144314


namespace hyperbola_representation_l144_144522

variable (x y : ℝ)

/--
Given the equation (x - y)^2 = 3(x^2 - y^2), we prove that
the resulting graph represents a hyperbola.
-/
theorem hyperbola_representation :
  (x - y)^2 = 3 * (x^2 - y^2) →
  ∃ A B C : ℝ, A ≠ 0 ∧ (x^2 + x * y - 2 * y^2 = 0) ∧ (A = 1) ∧ (B = 1) ∧ (C = -2) ∧ (B^2 - 4*A*C > 0) :=
by
  sorry

end hyperbola_representation_l144_144522


namespace swap_original_x_y_l144_144130

variables (x y z : ℕ)

theorem swap_original_x_y (x_original y_original : ℕ) 
  (step1 : z = x_original)
  (step2 : x = y_original)
  (step3 : y = z) :
  x = y_original ∧ y = x_original :=
sorry

end swap_original_x_y_l144_144130


namespace Area_S_inequality_l144_144958

def S (t : ℝ) (x y : ℝ) : Prop :=
  let T := Real.sin (Real.pi * t)
  |x - T| + |y - T| ≤ T

theorem Area_S_inequality (t : ℝ) :
  let T := Real.sin (Real.pi * t)
  0 ≤ 2 * T^2 := by
  sorry

end Area_S_inequality_l144_144958


namespace sub_from_square_l144_144751

theorem sub_from_square (n : ℕ) (h : n = 17) : (n * n - n) = 272 :=
by 
  -- Proof goes here
  sorry

end sub_from_square_l144_144751


namespace evaluate_expression_l144_144388

def x : ℚ := 1 / 4
def y : ℚ := 1 / 3
def z : ℚ := 12

theorem evaluate_expression : x^3 * y^4 * z = 1 / 432 := 
by
  sorry

end evaluate_expression_l144_144388


namespace tangent_intersection_locus_l144_144549

theorem tangent_intersection_locus :
  ∀ (l : ℝ → ℝ) (C : ℝ → ℝ), 
  (∀ x > 0, C x = x + 1/x) →
  (∃ k : ℝ, ∀ x, l x = k * x + 1) →
  ∃ (P : ℝ × ℝ), (P = (2, 2)) ∨ (P = (2, 5/2)) :=
by sorry

end tangent_intersection_locus_l144_144549


namespace angle_C_magnitude_area_triangle_l144_144236

variable {a b c A B C : ℝ}

namespace triangle

-- Conditions and variable declarations
axiom condition1 : 2 * b * Real.cos C = a * Real.cos C + c * Real.cos A
axiom triangle_sides : a = 3 ∧ b = 2 ∧ c = Real.sqrt 7

-- Prove the magnitude of angle C is π/3
theorem angle_C_magnitude : C = Real.pi / 3 :=
by sorry

-- Prove that given b = 2 and c = sqrt(7), a = 3 and the area of triangle ABC is 3*sqrt(3)/2
theorem area_triangle :
  (b = 2 ∧ c = Real.sqrt 7 ∧ C = Real.pi / 3) → 
  (a = 3 ∧ (1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2)) :=
by sorry

end triangle

end angle_C_magnitude_area_triangle_l144_144236


namespace ordered_triple_unique_l144_144347

variable (a b c : ℝ)

theorem ordered_triple_unique
  (h_pos_a : a > 4)
  (h_pos_b : b > 4)
  (h_pos_c : c > 4)
  (h_eq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  (a, b, c) = (12, 10, 8) := 
sorry

end ordered_triple_unique_l144_144347


namespace boys_on_soccer_team_l144_144532

theorem boys_on_soccer_team (B G : ℕ) (h1 : B + G = 30) (h2 : 1 / 3 * G + B = 20) : B = 15 :=
sorry

end boys_on_soccer_team_l144_144532


namespace golden_apples_first_six_months_l144_144338

-- Use appropriate namespaces
namespace ApolloProblem

-- Define the given conditions
def total_cost : ℕ := 54
def months_in_half_year : ℕ := 6

-- Prove that the number of golden apples charged for the first six months is 18
theorem golden_apples_first_six_months (X : ℕ) 
  (h1 : 6 * X + 6 * (2 * X) = total_cost) : 
  6 * X = 18 := 
sorry

end ApolloProblem

end golden_apples_first_six_months_l144_144338


namespace projectile_max_height_l144_144986

theorem projectile_max_height :
  ∀ (t : ℝ), -12 * t^2 + 72 * t + 45 ≤ 153 :=
by
  sorry

end projectile_max_height_l144_144986


namespace susan_gave_sean_8_apples_l144_144382

theorem susan_gave_sean_8_apples (initial_apples total_apples apples_given : ℕ) 
  (h1 : initial_apples = 9)
  (h2 : total_apples = 17)
  (h3 : apples_given = total_apples - initial_apples) : 
  apples_given = 8 :=
by
  sorry

end susan_gave_sean_8_apples_l144_144382


namespace simplify_expression_l144_144206

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l144_144206


namespace Pablo_is_70_cm_taller_than_Charlene_l144_144617

variable (Ruby Pablo Charlene Janet : ℕ)

-- Conditions
axiom h1 : Ruby + 2 = Pablo
axiom h2 : Charlene = 2 * Janet
axiom h3 : Janet = 62
axiom h4 : Ruby = 192

-- The statement to prove
theorem Pablo_is_70_cm_taller_than_Charlene : Pablo - Charlene = 70 :=
by
  -- Formalizing the proof
  sorry

end Pablo_is_70_cm_taller_than_Charlene_l144_144617


namespace tangent_line_problem_l144_144417

theorem tangent_line_problem 
  (x1 x2 : ℝ)
  (h1 : (1 / x1) = Real.exp x2)
  (h2 : Real.log x1 - 1 = Real.exp x2 * (1 - x2)) :
  (2 / (x1 - 1) + x2 = -1) :=
by 
  sorry

end tangent_line_problem_l144_144417


namespace find_nsatisfy_l144_144567

-- Define the function S(n) that denotes the sum of the digits of n
def S (n : ℕ) : ℕ := n.digits 10 |>.sum

-- State the main theorem
theorem find_nsatisfy {n : ℕ} : n = 2 * (S n)^2 → n = 50 ∨ n = 162 ∨ n = 392 ∨ n = 648 := 
sorry

end find_nsatisfy_l144_144567


namespace set_of_points_l144_144622

theorem set_of_points : {p : ℝ × ℝ | (2 * p.1 - p.2 = 1) ∧ (p.1 + 4 * p.2 = 5)} = { (1, 1) } :=
by
  sorry

end set_of_points_l144_144622


namespace intersection_of_A_and_B_l144_144555

-- Define the sets A and B
def A := {x : ℝ | x ≥ 1}
def B := {x : ℝ | -1 < x ∧ x < 2}

-- Define the expected intersection
def expected_intersection := {x : ℝ | 1 ≤ x ∧ x < 2}

-- The proof problem statement
theorem intersection_of_A_and_B :
  A ∩ B = expected_intersection := by
  sorry

end intersection_of_A_and_B_l144_144555


namespace magic_coin_l144_144959

theorem magic_coin (m n : ℕ) (h_m_prime: Nat.gcd m n = 1)
  (h_prob : (m : ℚ) / n = 1 / 158760): m + n = 158761 := by
  sorry

end magic_coin_l144_144959


namespace percentage_salt_l144_144854

-- Variables
variables {S1 S2 R : ℝ}

-- Conditions
def first_solution := S1
def second_solution := (25 / 100) * 19.000000000000007
def resulting_solution := 16

theorem percentage_salt (S1 S2 : ℝ) (H1: S2 = 19.000000000000007) 
(H2: (75 / 100) * S1 + (25 / 100) * S2 = 16) : 
S1 = 15 :=
by
    rw [H1] at H2
    sorry

end percentage_salt_l144_144854


namespace passenger_catches_bus_l144_144113

-- Definitions based on conditions from part a)
def P_route3 := 0.20
def P_route6 := 0.60

-- Statement to prove based on part c)
theorem passenger_catches_bus : 
  P_route3 + P_route6 = 0.80 := 
by
  sorry

end passenger_catches_bus_l144_144113


namespace mass_percentage_O_mixture_l144_144878

noncomputable def molar_mass_Al2O3 : ℝ := (2 * 26.98) + (3 * 16.00)
noncomputable def molar_mass_Cr2O3 : ℝ := (2 * 51.99) + (3 * 16.00)
noncomputable def mass_of_O_in_Al2O3 : ℝ := 3 * 16.00
noncomputable def mass_of_O_in_Cr2O3 : ℝ := 3 * 16.00
noncomputable def mass_percentage_O_in_Al2O3 : ℝ := (mass_of_O_in_Al2O3 / molar_mass_Al2O3) * 100
noncomputable def mass_percentage_O_in_Cr2O3 : ℝ := (mass_of_O_in_Cr2O3 / molar_mass_Cr2O3) * 100
noncomputable def mass_percentage_O_in_mixture : ℝ := (0.50 * mass_percentage_O_in_Al2O3) + (0.50 * mass_percentage_O_in_Cr2O3)

theorem mass_percentage_O_mixture : mass_percentage_O_in_mixture = 39.325 := by
  sorry

end mass_percentage_O_mixture_l144_144878


namespace pipe_length_l144_144819

theorem pipe_length (S L : ℕ) (h1: S = 28) (h2: L = S + 12) : S + L = 68 := 
by
  sorry

end pipe_length_l144_144819


namespace boys_chairs_problem_l144_144672

theorem boys_chairs_problem :
  ∃ (n k : ℕ), n * k = 123 ∧ (∀ p q : ℕ, p * q = 123 → p = n ∧ q = k ∨ p = k ∧ q = n) :=
by
  sorry

end boys_chairs_problem_l144_144672


namespace molecular_weight_Dinitrogen_pentoxide_l144_144038

theorem molecular_weight_Dinitrogen_pentoxide :
  let atomic_weight_N := 14.01
  let atomic_weight_O := 16.00
  let molecular_formula := (2 * atomic_weight_N) + (5 * atomic_weight_O)
  molecular_formula = 108.02 :=
by
  sorry

end molecular_weight_Dinitrogen_pentoxide_l144_144038


namespace ratio_pr_l144_144987

variable (p q r s : ℚ)

def ratio_pq (p q : ℚ) : Prop := p / q = 5 / 4
def ratio_rs (r s : ℚ) : Prop := r / s = 4 / 3
def ratio_sq (s q : ℚ) : Prop := s / q = 1 / 5

theorem ratio_pr (hpq : ratio_pq p q) (hrs : ratio_rs r s) (hsq : ratio_sq s q) : p / r = 75 / 16 := by
  sorry

end ratio_pr_l144_144987


namespace average_score_remaining_students_l144_144576

theorem average_score_remaining_students (n : ℕ) (h : n > 15) (avg_all : ℚ) (avg_15 : ℚ) :
  avg_all = 12 → avg_15 = 20 →
  (∃ avg_remaining : ℚ, avg_remaining = (12 * n - 300) / (n - 15)) :=
by
  sorry

end average_score_remaining_students_l144_144576


namespace inequality_min_value_l144_144126

theorem inequality_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  ∃ m : ℝ, (x + 2 * y) * (2 / x + 1 / y) ≥ m ∧ m ≤ 8 :=
by
  sorry

end inequality_min_value_l144_144126


namespace other_asymptote_l144_144219

-- Define the conditions
def C1 := ∀ x y, y = -2 * x
def C2 := ∀ x, x = -3

-- Formulate the problem
theorem other_asymptote :
  (∃ y m b, y = m * x + b ∧ m = 2 ∧ b = 12) :=
by
  sorry

end other_asymptote_l144_144219


namespace simplify_expression_l144_144373

theorem simplify_expression (a c d x y z : ℝ) :
  (cx * (a^3 * x^3 + 3 * a^3 * y^3 + c^3 * z^3) + dz * (a^3 * x^3 + 3 * c^3 * x^3 + c^3 * z^3)) / (cx + dz) =
  a^3 * x^3 + c^3 * z^3 + (3 * cx * a^3 * y^3 / (cx + dz)) + (3 * dz * c^3 * x^3 / (cx + dz)) :=
by
  sorry

end simplify_expression_l144_144373


namespace number_of_solution_pairs_l144_144778

def integer_solutions_on_circle : Set (Int × Int) := {
  (1, 7), (1, -7), (-1, 7), (-1, -7),
  (5, 5), (5, -5), (-5, 5), (-5, -5),
  (7, 1), (7, -1), (-7, 1), (-7, -1) 
}

def system_of_equations_has_integer_solution (a b : ℝ) : Prop :=
  ∃ (x y : ℤ), a * ↑x + b * ↑y = 1 ∧ (↑x ^ 2 + ↑y ^ 2 = 50)

theorem number_of_solution_pairs : ∃ (n : ℕ), n = 72 ∧
  (∀ (a b : ℝ), system_of_equations_has_integer_solution a b → n = 72) := 
sorry

end number_of_solution_pairs_l144_144778


namespace algebraic_expression_value_l144_144250

variables {m n : ℝ}

theorem algebraic_expression_value (h : n = 3 - 5 * m) : 10 * m + 2 * n - 3 = 3 :=
by sorry

end algebraic_expression_value_l144_144250


namespace functional_equation_solution_l144_144033

theorem functional_equation_solution (f : ℚ → ℚ) (H : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ k : ℚ, ∀ x : ℚ, f x = k * x := 
sorry

end functional_equation_solution_l144_144033


namespace maria_sister_drank_l144_144184

-- Define the conditions
def initial_bottles : ℝ := 45.0
def maria_drank : ℝ := 14.0
def remaining_bottles : ℝ := 23.0

-- Define the problem statement to prove the number of bottles Maria's sister drank
theorem maria_sister_drank (initial_bottles maria_drank remaining_bottles : ℝ) : 
    (initial_bottles - maria_drank) - remaining_bottles = 8.0 :=
by
  sorry

end maria_sister_drank_l144_144184


namespace geometric_sequence_sum_l144_144713

noncomputable def geometric_sequence (a : ℕ → ℝ) (r: ℝ): Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (r: ℝ)
  (h_geometric : geometric_sequence a r)
  (h_ratio : r = 2)
  (h_sum_condition : a 1 + a 4 + a 7 = 10) :
  a 3 + a 6 + a 9 = 20 := 
sorry

end geometric_sequence_sum_l144_144713


namespace tangent_line_circle_p_l144_144418

theorem tangent_line_circle_p (p : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 6 * x + 8 = 0 → (x = -p/2 ∨ y = 0)) → 
  (p = 4 ∨ p = 8) :=
by
  sorry

end tangent_line_circle_p_l144_144418


namespace intersection_of_A_and_B_eq_C_l144_144529

noncomputable def A (x : ℝ) : Prop := x^2 - 4*x + 3 < 0
noncomputable def B (x : ℝ) : Prop := 2 - x > 0
noncomputable def A_inter_B (x : ℝ) : Prop := A x ∧ B x

theorem intersection_of_A_and_B_eq_C :
  {x : ℝ | A_inter_B x} = {x : ℝ | 1 < x ∧ x < 2} :=
by sorry

end intersection_of_A_and_B_eq_C_l144_144529


namespace percent_students_with_pets_l144_144998

theorem percent_students_with_pets 
  (total_students : ℕ) (students_with_cats : ℕ) (students_with_dogs : ℕ) (students_with_both : ℕ) (h_total : total_students = 500)
  (h_cats : students_with_cats = 150) (h_dogs : students_with_dogs = 100) (h_both : students_with_both = 40) :
  (students_with_cats + students_with_dogs - students_with_both) * 100 / total_students = 42 := 
by
  sorry

end percent_students_with_pets_l144_144998


namespace factor_polynomial_l144_144182

theorem factor_polynomial (x : ℝ) : 
    54 * x^4 - 135 * x^8 = -27 * x^4 * (5 * x^4 - 2) := 
by 
  sorry

end factor_polynomial_l144_144182


namespace curve_is_line_l144_144145

theorem curve_is_line (r θ x y : ℝ) (h : r = 1 / (Real.sin θ + Real.cos θ)) (hx : x = r * Real.cos θ) (hy : y = r * Real.sin θ) :
  x + y = 1 := by
  sorry

end curve_is_line_l144_144145


namespace max_unique_rankings_l144_144646

theorem max_unique_rankings (n : ℕ) : 
  ∃ (contestants : ℕ), 
    (∀ (scores : ℕ → ℕ), 
      (∀ i, 0 ≤ scores i ∧ scores i ≤ contestants) ∧
      (∀ i j, i ≠ j → scores i ≠ scores j)) 
    → contestants = 2^n := 
sorry

end max_unique_rankings_l144_144646


namespace total_animal_legs_l144_144061

theorem total_animal_legs (total_animals : ℕ) (sheep : ℕ) (chickens : ℕ) : 
  total_animals = 20 ∧ sheep = 10 ∧ chickens = 10 ∧ 
  2 * chickens + 4 * sheep = 60 :=
by 
  sorry

end total_animal_legs_l144_144061


namespace real_root_exists_l144_144667

theorem real_root_exists (p1 p2 q1 q2 : ℝ) 
(h : p1 * p2 = 2 * (q1 + q2)) : 
  (∃ x : ℝ, x^2 + p1 * x + q1 = 0) ∨ (∃ x : ℝ, x^2 + p2 * x + q2 = 0) :=
by
  sorry

end real_root_exists_l144_144667


namespace angle_A_condition_area_range_condition_l144_144413

/-- Given a triangle ABC with sides opposite to internal angles A, B, and C labeled as a, b, and c respectively. 
Given the condition a * cos C + sqrt 3 * a * sin C = b + c.
Prove that angle A = π / 3.
-/
theorem angle_A_condition
  (a b c : ℝ) (C : ℝ) (h : a * Real.cos C + Real.sqrt 3 * a * Real.sin C = b + c) :
  A = Real.pi / 3 := sorry
  
/-- Given an acute triangle ABC with b = 2 and angle A = π / 3,
find the range of possible values for the area of the triangle ABC.
-/
theorem area_range_condition
  (a c : ℝ) (A : ℝ) (b : ℝ) (C B : ℝ)
  (h1 : b = 2)
  (h2 : A = Real.pi / 3)
  (h3 : 0 < B) (h4 : B < Real.pi / 2)
  (h5 : 0 < C) (h6 : C < Real.pi / 2)
  (h7 : A + C = 2 * Real.pi / 3) :
  Real.sqrt 3 / 2 < (1 / 2) * a * b * Real.sin C ∧
  (1 / 2) * a * b * Real.sin C < 2 * Real.sqrt 3 := sorry

end angle_A_condition_area_range_condition_l144_144413


namespace reduced_price_after_exchange_rate_fluctuation_l144_144767

-- Definitions based on conditions
variables (P : ℝ) -- Original price per kg

def reduced_price_per_kg : ℝ := 0.9 * P

axiom six_kg_costs_900 : 6 * reduced_price_per_kg P = 900

-- Additional conditions
def exchange_rate_factor : ℝ := 1.02

-- Question restated as the theorem to prove
theorem reduced_price_after_exchange_rate_fluctuation : 
  ∃ P : ℝ, reduced_price_per_kg P * exchange_rate_factor = 153 :=
sorry

end reduced_price_after_exchange_rate_fluctuation_l144_144767


namespace find_value_l144_144757

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom symmetric_about_one : ∀ x, f (x - 1) = f (1 - x)
axiom equation_on_interval : ∀ x, 0 < x ∧ x < 1 → f x = 9^x

theorem find_value : f (5 / 2) + f 2 = -3 := 
by sorry

end find_value_l144_144757


namespace range_of_a_l144_144902

-- Define the inequality condition
def inequality (x a : ℝ) : Prop :=
  2 * x^2 + a * x - a^2 > 0

-- State the main problem
theorem range_of_a (a: ℝ) : 
  inequality 2 a -> (-2 < a) ∧ (a < 4) :=
by
  sorry

end range_of_a_l144_144902


namespace salmon_trip_l144_144427

theorem salmon_trip (male_female_sum : 712261 + 259378 = 971639) : 
  712261 + 259378 = 971639 := 
by 
  exact male_female_sum

end salmon_trip_l144_144427


namespace angle_sum_in_hexagon_l144_144890

theorem angle_sum_in_hexagon (P Q R s t : ℝ) 
    (hP: P = 40) (hQ: Q = 88) (hR: R = 30)
    (hex_sum: 6 * 180 - 720 = 0): 
    s + t = 312 :=
by
  have hex_interior_sum: 6 * 180 - 720 = 0 := hex_sum
  sorry

end angle_sum_in_hexagon_l144_144890


namespace abs_neg_three_l144_144377

theorem abs_neg_three : abs (-3) = 3 :=
sorry

end abs_neg_three_l144_144377


namespace extreme_value_0_at_minus_1_l144_144480

theorem extreme_value_0_at_minus_1 (m n : ℝ)
  (h1 : (-1) + 3 * m - n + m^2 = 0)
  (h2 : 3 - 6 * m + n = 0) :
  m + n = 11 :=
sorry

end extreme_value_0_at_minus_1_l144_144480


namespace total_profit_l144_144241

variable (A_s B_s C_s : ℝ)
variable (A_p : ℝ := 14700)
variable (P : ℝ)

theorem total_profit
  (h1 : A_s + B_s + C_s = 50000)
  (h2 : A_s = B_s + 4000)
  (h3 : B_s = C_s + 5000)
  (h4 : A_p = 14700) :
  P = 35000 :=
sorry

end total_profit_l144_144241


namespace rate_percent_simple_interest_l144_144420

theorem rate_percent_simple_interest:
  ∀ (P SI T R : ℝ), SI = 400 → P = 1000 → T = 4 → (SI = P * R * T / 100) → R = 10 :=
by
  intros P SI T R h_si h_p h_t h_formula
  -- Proof skipped
  sorry

end rate_percent_simple_interest_l144_144420


namespace units_digit_product_l144_144592

theorem units_digit_product (a b : ℕ) (ha : a % 10 = 7) (hb : b % 10 = 4) :
  (a * b) % 10 = 8 := 
by
  sorry

end units_digit_product_l144_144592


namespace prime_factors_of_difference_l144_144246

theorem prime_factors_of_difference (A B : ℕ) (h_neq : A ≠ B) : 
  ∃ p, Nat.Prime p ∧ p ∣ (Nat.gcd (9 * A - 9 * B + 10) (9 * B - 9 * A - 10)) :=
by
  sorry

end prime_factors_of_difference_l144_144246


namespace express_x2_y2_z2_in_terms_of_sigma1_sigma2_l144_144712

variable (x y z : ℝ)
def sigma1 := x + y + z
def sigma2 := x * y + y * z + z * x

theorem express_x2_y2_z2_in_terms_of_sigma1_sigma2 :
  x^2 + y^2 + z^2 = sigma1 x y z ^ 2 - 2 * sigma2 x y z := by
  sorry

end express_x2_y2_z2_in_terms_of_sigma1_sigma2_l144_144712


namespace min_bought_chocolates_l144_144582

variable (a b : ℕ)

theorem min_bought_chocolates :
    ∃ a : ℕ, 
        ∃ b : ℕ, 
            b = a + 41 
            ∧ (376 - a - b = 3 * a) 
            ∧ a = 67 :=
by
  sorry

end min_bought_chocolates_l144_144582


namespace cost_per_mile_l144_144288

theorem cost_per_mile (m x : ℝ) (h_cost_eq : 2.50 + x * m = 2.50 + 5.00 + x * 14) : 
  x = 5 / 14 :=
by
  sorry

end cost_per_mile_l144_144288


namespace spent_on_music_l144_144674

variable (total_allowance : ℝ) (fraction_music : ℝ)

-- Assuming the conditions
def conditions : Prop :=
  total_allowance = 50 ∧ fraction_music = 3 / 10

-- The proof problem
theorem spent_on_music (h : conditions total_allowance fraction_music) : 
  total_allowance * fraction_music = 15 := by
  cases h with
  | intro h_total h_fraction =>
  sorry

end spent_on_music_l144_144674


namespace probability_jammed_l144_144843

theorem probability_jammed (T τ : ℝ) (h : τ < T) : 
    (2 * τ / T - (τ / T) ^ 2) = (T^2 - (T - τ)^2) / T^2 := 
by
  sorry

end probability_jammed_l144_144843


namespace find_acute_angle_l144_144991

theorem find_acute_angle (α : ℝ) (h1 : 0 < α ∧ α < 90) (h2 : ∃ k : ℤ, 10 * α = α + k * 360) :
  α = 40 ∨ α = 80 :=
by
  sorry

end find_acute_angle_l144_144991


namespace volume_of_sphere_l144_144593

theorem volume_of_sphere (V : ℝ) (r : ℝ) : r = 1 / 3 → (2 * r) = (16 / 9 * V)^(1/3) → V = 1 / 6 :=
by
  intro h_radius h_diameter
  sorry

end volume_of_sphere_l144_144593


namespace angle_same_terminal_side_l144_144084

theorem angle_same_terminal_side (θ : ℝ) (α : ℝ) 
  (hθ : θ = -950) 
  (hα_range : 0 ≤ α ∧ α ≤ 180) 
  (h_terminal_side : ∃ k : ℤ, θ = α + k * 360) : 
  α = 130 := by
  sorry

end angle_same_terminal_side_l144_144084


namespace eric_has_more_than_500_paperclips_on_saturday_l144_144595

theorem eric_has_more_than_500_paperclips_on_saturday :
  ∃ k : ℕ, (4 * 3 ^ k > 500) ∧ (∀ m : ℕ, m < k → 4 * 3 ^ m ≤ 500) ∧ ((k + 1) % 7 = 6) :=
by
  sorry

end eric_has_more_than_500_paperclips_on_saturday_l144_144595


namespace preceding_integer_binary_l144_144321

theorem preceding_integer_binary (M : ℕ) (h : M = 0b110101) : 
  (M - 1) = 0b110100 :=
by
  sorry

end preceding_integer_binary_l144_144321


namespace binomial_10_3_l144_144128

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l144_144128


namespace f_max_a_zero_f_zero_range_l144_144990

-- Part 1: Proving the maximum value when a = 0
theorem f_max_a_zero : ∀ (f : ℝ → ℝ) (x : ℝ),
  (f x = (-1 / x) - Real.log x) ∧ x > 0 → 
  ∃ x_max : ℝ, f x_max = -1 ∧ 
  (∀ x > 0, f x ≤ -1) := 
sorry

-- Part 2: Proving the range of a for exactly one zero of f(x)
theorem f_zero_range (a : ℝ) : (0 < a) → 
  ∀ (f : ℝ → ℝ) (x : ℝ), 
  (f x = a * x - 1 / x - (a + 1) * Real.log x) ∧ x > 0 →
  ∃! (x_zero : ℝ), f x_zero = 0 :=
sorry

end f_max_a_zero_f_zero_range_l144_144990


namespace cos_lt_sin3_div_x3_l144_144963

open Real

theorem cos_lt_sin3_div_x3 (x : ℝ) (h1 : 0 < x) (h2 : x < pi / 2) : 
  cos x < (sin x / x) ^ 3 := 
  sorry

end cos_lt_sin3_div_x3_l144_144963


namespace permits_cost_l144_144594

-- Definitions based on conditions
def total_cost : ℕ := 2950
def contractor_hourly_rate : ℕ := 150
def contractor_hours_per_day : ℕ := 5
def contractor_days : ℕ := 3
def inspector_discount_rate : ℕ := 80

-- Proving the cost of permits
theorem permits_cost : ∃ (permits_cost : ℕ), permits_cost = 250 :=
by
  let contractor_hours := contractor_days * contractor_hours_per_day
  let contractor_cost := contractor_hours * contractor_hourly_rate
  let inspector_hourly_rate := contractor_hourly_rate - (contractor_hourly_rate * inspector_discount_rate / 100)
  let inspector_cost := contractor_hours * inspector_hourly_rate
  let total_cost_without_permits := contractor_cost + inspector_cost
  let permits_cost := total_cost - total_cost_without_permits
  use permits_cost
  sorry

end permits_cost_l144_144594


namespace ratio_of_doctors_to_lawyers_l144_144570

variable (d l : ℕ) -- number of doctors and lawyers
variable (h1 : (40 * d + 55 * l) / (d + l) = 45) -- overall average age condition

theorem ratio_of_doctors_to_lawyers : d = 2 * l :=
by
  sorry

end ratio_of_doctors_to_lawyers_l144_144570


namespace student_history_mark_l144_144846

theorem student_history_mark
  (math_score : ℕ)
  (desired_average : ℕ)
  (third_subject_score : ℕ)
  (history_score : ℕ) :
  math_score = 74 →
  desired_average = 75 →
  third_subject_score = 70 →
  (math_score + history_score + third_subject_score) / 3 = desired_average →
  history_score = 81 :=
by
  intros h_math h_avg h_third h_equiv
  sorry

end student_history_mark_l144_144846


namespace sum_of_arithmetic_sequence_l144_144642

noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

theorem sum_of_arithmetic_sequence (a1 : ℤ) (d : ℤ)
  (h1 : a1 = -2010)
  (h2 : (S 2011 a1 d) / 2011 - (S 2009 a1 d) / 2009 = 2) :
  S 2010 a1 d = -2010 := 
sorry

end sum_of_arithmetic_sequence_l144_144642


namespace y_values_l144_144787

noncomputable def y (x : ℝ) : ℝ :=
  (Real.sin x / |Real.sin x|) + (|Real.cos x| / Real.cos x) + (Real.tan x / |Real.tan x|)

theorem y_values (x : ℝ) (h1 : 0 < x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x ≠ 0) (h4 : Real.cos x ≠ 0) (h5 : Real.tan x ≠ 0) :
  y x = 3 ∨ y x = -1 :=
sorry

end y_values_l144_144787


namespace probability_of_drawing_2_black_and_2_white_l144_144816

def total_balls : ℕ := 17
def black_balls : ℕ := 9
def white_balls : ℕ := 8
def balls_drawn : ℕ := 4
def favorable_outcomes := (Nat.choose 9 2) * (Nat.choose 8 2)
def total_outcomes := Nat.choose 17 4
def probability_draw : ℚ := favorable_outcomes / total_outcomes

theorem probability_of_drawing_2_black_and_2_white :
  probability_draw = 168 / 397 :=
by
  sorry

end probability_of_drawing_2_black_and_2_white_l144_144816


namespace subset_proper_l144_144105

def M : Set ℝ := {x | x^2 - x ≤ 0}

def N : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem subset_proper : N ⊂ M := by
  sorry

end subset_proper_l144_144105


namespace heavy_operators_earn_129_dollars_per_day_l144_144181

noncomputable def heavy_operator_daily_wage (H : ℕ) : Prop :=
  let laborer_wage := 82
  let total_people := 31
  let total_payroll := 3952
  let laborers_count := 1
  let heavy_operators_count := total_people - laborers_count
  let heavy_operators_payroll := total_payroll - (laborer_wage * laborers_count)
  H = heavy_operators_payroll / heavy_operators_count

theorem heavy_operators_earn_129_dollars_per_day : heavy_operator_daily_wage 129 :=
by
  unfold heavy_operator_daily_wage
  sorry

end heavy_operators_earn_129_dollars_per_day_l144_144181


namespace triangle_shape_l144_144560

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) (h1 : a * Real.cos A = b * Real.cos B) :
  (a = b ∨ c = a ∨ c = b ∨ A = Real.pi / 2 ∨ B = Real.pi / 2 ∨ C = Real.pi / 2) :=
sorry

end triangle_shape_l144_144560


namespace sequence_an_sequence_Tn_l144_144850

theorem sequence_an (a : ℕ → ℕ) (S : ℕ → ℕ) (h : ∀ n, 2 * S n = a n ^ 2 + a n):
  ∀ n, a n = n :=
sorry

theorem sequence_Tn (b : ℕ → ℕ) (T : ℕ → ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, 2 * S n = a n ^ 2 + a n) (h2 : ∀ n, a n = n) (h3 : ∀ n, b n = 2^n * a n):
  ∀ n, T n = (n - 1) * 2^(n + 1) + 2 :=
sorry

end sequence_an_sequence_Tn_l144_144850


namespace average_age_of_family_l144_144208

theorem average_age_of_family :
  let num_grandparents := 2
  let num_parents := 2
  let num_grandchildren := 3
  let avg_age_grandparents := 64
  let avg_age_parents := 39
  let avg_age_grandchildren := 6
  let total_age_grandparents := avg_age_grandparents * num_grandparents
  let total_age_parents := avg_age_parents * num_parents
  let total_age_grandchildren := avg_age_grandchildren * num_grandchildren
  let total_age_family := total_age_grandparents + total_age_parents + total_age_grandchildren
  let num_family_members := num_grandparents + num_parents + num_grandchildren
  let avg_age_family := total_age_family / num_family_members
  avg_age_family = 32 := 
  by 
  repeat { sorry }

end average_age_of_family_l144_144208


namespace find_x_l144_144444

def vec (x y : ℝ) := (x, y)

def a := vec 1 (-4)
def b (x : ℝ) := vec (-1) x
def c (x : ℝ) := (a.1 + 3 * (b x).1, a.2 + 3 * (b x).2)

theorem find_x (x : ℝ) : a.1 * (c x).2 = (c x).1 * a.2 → x = 4 :=
by
  sorry

end find_x_l144_144444


namespace max_catch_up_distance_l144_144511

/-- 
Given:
  - The total length of the race is 5000 feet.
  - Alex and Max are even for the first 200 feet, so the initial distance between them is 0 feet.
  - On the uphill slope, Alex gets ahead by 300 feet.
  - On the downhill slope, Max gains a lead of 170 feet over Alex, reducing Alex's lead.
  - On the flat section, Alex pulls ahead by 440 feet.

Prove:
  - The distance left for Max to catch up to Alex is 4430 feet.
--/
theorem max_catch_up_distance :
  let total_distance := 5000
  let initial_distance := 0
  let alex_uphill_lead := 300
  let max_downhill_gain := 170
  let alex_flat_gain := 440
  let final_distance := initial_distance + alex_uphill_lead - max_downhill_gain + alex_flat_gain
  total_distance - final_distance = 4430 :=
by
  let total_distance := 5000
  let initial_distance := 0
  let alex_uphill_lead := 300
  let max_downhill_gain := 170
  let alex_flat_gain := 440
  let final_distance := initial_distance + alex_uphill_lead - max_downhill_gain + alex_flat_gain
  have final_distance_calc : final_distance = 570
  sorry
  show total_distance - final_distance = 4430
  sorry

end max_catch_up_distance_l144_144511


namespace find_a_plus_b_l144_144811

def f (x : ℝ) : ℝ := x^3 + 3*x - 1

theorem find_a_plus_b (a b : ℝ) (h1 : f (a - 3) = -3) (h2 : f (b - 3) = 1) :
  a + b = 6 :=
sorry

end find_a_plus_b_l144_144811


namespace inequality_hold_l144_144805

theorem inequality_hold (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  0 < (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ∧ 
  (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ≤ 1/8 :=
sorry

end inequality_hold_l144_144805


namespace arithmetic_sequence_geometric_sequence_l144_144196

-- Arithmetic sequence proof
theorem arithmetic_sequence (d n : ℕ) (a_n a_1 : ℤ) (s_n : ℤ) :
  d = 2 → n = 15 → a_n = -10 →
  a_1 = -38 ∧ s_n = -360 :=
sorry

-- Geometric sequence proof
theorem geometric_sequence (a_1 a_4 q s_3 : ℤ) :
  a_1 = -1 → a_4 = 64 →
  q = -4 ∧ s_3 = -13 :=
sorry

end arithmetic_sequence_geometric_sequence_l144_144196


namespace retailer_profit_percentage_l144_144422

theorem retailer_profit_percentage
  (cost_price : ℝ)
  (marked_percent : ℝ)
  (discount_percent : ℝ)
  (selling_price : ℝ)
  (marked_price : ℝ)
  (profit_percent : ℝ) :
  marked_percent = 60 →
  discount_percent = 25 →
  marked_price = cost_price * (1 + marked_percent / 100) →
  selling_price = marked_price * (1 - discount_percent / 100) →
  profit_percent = ((selling_price - cost_price) / cost_price) * 100 →
  profit_percent = 20 :=
by
  sorry

end retailer_profit_percentage_l144_144422


namespace arrange_balls_l144_144921

/-- Given 4 yellow balls and 3 red balls, we want to prove that there are 35 different ways to arrange these balls in a row. -/
theorem arrange_balls : (Nat.choose 7 4) = 35 := by
  sorry

end arrange_balls_l144_144921


namespace jane_average_speed_correct_l144_144838

noncomputable def jane_average_speed : ℝ :=
  let total_distance : ℝ := 250
  let total_time : ℝ := 6
  total_distance / total_time

theorem jane_average_speed_correct : jane_average_speed = 41.67 := by
  sorry

end jane_average_speed_correct_l144_144838


namespace combined_rate_is_29_l144_144094

def combined_rate_of_mpg (miles_ray : ℕ) (mpg_ray : ℕ) (miles_tom : ℕ) (mpg_tom : ℕ) (miles_jerry : ℕ) (mpg_jerry : ℕ) : ℕ :=
  let gallons_ray := miles_ray / mpg_ray
  let gallons_tom := miles_tom / mpg_tom
  let gallons_jerry := miles_jerry / mpg_jerry
  let total_gallons := gallons_ray + gallons_tom + gallons_jerry
  let total_miles := miles_ray + miles_tom + miles_jerry
  total_miles / total_gallons

theorem combined_rate_is_29 :
  combined_rate_of_mpg 60 50 60 20 60 30 = 29 :=
by
  sorry

end combined_rate_is_29_l144_144094


namespace equilateral_triangle_BJ_l144_144192

-- Define points G, F, H, J and their respective lengths on sides AB and BC
def equilateral_triangle_AG_GF_HJ_FC (AG GF HJ FC BJ : ℕ) : Prop :=
  AG = 3 ∧ GF = 11 ∧ HJ = 5 ∧ FC = 4 ∧ 
    (∀ (side_length : ℕ), side_length = AG + GF + HJ + FC → 
    (∀ (length_J : ℕ), length_J = side_length - (AG + HJ) → BJ = length_J))

-- Example usage statement
theorem equilateral_triangle_BJ : 
  ∃ BJ, equilateral_triangle_AG_GF_HJ_FC 3 11 5 4 BJ ∧ BJ = 15 :=
by
  use 15
  sorry

end equilateral_triangle_BJ_l144_144192


namespace prob_three_heads_is_one_eighth_l144_144138

-- Define the probability of heads in a fair coin
def fair_coin_prob_heads : ℚ := 1 / 2

-- Define the probability of three consecutive heads
def prob_three_heads (p : ℚ) : ℚ := p * p * p

-- Theorem statement
theorem prob_three_heads_is_one_eighth :
  prob_three_heads fair_coin_prob_heads = 1 / 8 := 
sorry

end prob_three_heads_is_one_eighth_l144_144138


namespace combined_cost_of_apples_and_strawberries_l144_144616

theorem combined_cost_of_apples_and_strawberries :
  let cost_of_apples := 15
  let cost_of_strawberries := 26
  cost_of_apples + cost_of_strawberries = 41 :=
by
  sorry

end combined_cost_of_apples_and_strawberries_l144_144616


namespace thabo_number_of_hardcover_nonfiction_books_l144_144750

variables (P_f H_f P_nf H_nf A : ℕ)

theorem thabo_number_of_hardcover_nonfiction_books
  (h1 : P_nf = H_nf + 15)
  (h2 : H_f = P_f + 10)
  (h3 : P_f = 3 * A)
  (h4 : A + H_f = 70)
  (h5 : P_f + H_f + P_nf + H_nf + A = 250) :
  H_nf = 30 :=
by {
  sorry
}

end thabo_number_of_hardcover_nonfiction_books_l144_144750


namespace annulus_area_l144_144456

theorem annulus_area (r_inner r_outer : ℝ) (h_inner : r_inner = 8) (h_outer : r_outer = 2 * r_inner) :
  π * r_outer ^ 2 - π * r_inner ^ 2 = 192 * π :=
by
  sorry

end annulus_area_l144_144456


namespace infinite_non_congruent_right_triangles_l144_144048

noncomputable def right_triangle_equal_perimeter_area : Prop :=
  ∃ (a b c : ℝ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + b^2 = c^2) ∧ 
  (a + b + c = (1/2) * a * b)

theorem infinite_non_congruent_right_triangles :
  ∃ (k : ℕ), right_triangle_equal_perimeter_area :=
sorry

end infinite_non_congruent_right_triangles_l144_144048


namespace race_time_difference_l144_144464

-- Define Malcolm's speed, Joshua's speed, and the distance
def malcolm_speed := 6 -- minutes per mile
def joshua_speed := 7 -- minutes per mile
def race_distance := 15 -- miles

-- Statement of the theorem
theorem race_time_difference :
  (joshua_speed * race_distance) - (malcolm_speed * race_distance) = 15 :=
by sorry

end race_time_difference_l144_144464


namespace probability_of_three_white_balls_equals_8_over_65_l144_144161

noncomputable def probability_three_white_balls (n_white n_black : ℕ) (draws : ℕ) : ℚ :=
  (Nat.choose n_white draws : ℚ) / Nat.choose (n_white + n_black) draws

theorem probability_of_three_white_balls_equals_8_over_65 :
  probability_three_white_balls 8 7 3 = 8 / 65 :=
by
  sorry

end probability_of_three_white_balls_equals_8_over_65_l144_144161


namespace local_minimum_at_neg_one_l144_144478

noncomputable def f (x : ℝ) := x * Real.exp x

theorem local_minimum_at_neg_one : (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x + 1) < δ → f x > f (-1)) :=
sorry

end local_minimum_at_neg_one_l144_144478


namespace shares_of_stocks_they_can_buy_l144_144671

def weekly_savings_wife : ℕ := 100
def monthly_savings_husband : ℕ := 225
def months_of_savings : ℕ := 4
def cost_per_share : ℕ := 50

theorem shares_of_stocks_they_can_buy :
  (((weekly_savings_wife * 4) + monthly_savings_husband) * months_of_savings / 2) / cost_per_share = 25 :=
by
  -- sorry for the implementation
  sorry

end shares_of_stocks_they_can_buy_l144_144671


namespace range_of_f_l144_144946

noncomputable def f (x : ℝ) : ℝ := 2^(2*x) + 2^(x+1) + 3

theorem range_of_f : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y > 3 :=
by
  sorry

end range_of_f_l144_144946


namespace total_toothpicks_correct_l144_144399

def number_of_horizontal_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
(height + 1) * width

def number_of_vertical_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
(height) * (width + 1)

def total_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
number_of_horizontal_toothpicks height width + number_of_vertical_toothpicks height width

theorem total_toothpicks_correct:
  total_toothpicks 30 15 = 945 :=
by
  sorry

end total_toothpicks_correct_l144_144399


namespace problem_statement_l144_144228

variable (x P : ℝ)

theorem problem_statement
  (h1 : x^2 - 5 * x + 6 < 0)
  (h2 : P = x^2 + 5 * x + 6) :
  (20 < P) ∧ (P < 30) :=
sorry

end problem_statement_l144_144228


namespace math_problem_l144_144029

theorem math_problem 
  (a : Int) (b : Int) (c : Int)
  (h_a : a = -1)
  (h_b : b = 1)
  (h_c : c = 0) :
  a + c - b = -2 := 
by
  sorry

end math_problem_l144_144029


namespace problem_statement_l144_144419

variable (x : ℝ)
def A := ({-3, x^2, x + 1} : Set ℝ)
def B := ({x - 3, 2 * x - 1, x^2 + 1} : Set ℝ)

theorem problem_statement (hx : A x ∩ B x = {-3}) : 
  x = -1 ∧ A x ∪ B x = ({-4, -3, 0, 1, 2} : Set ℝ) :=
by
  sorry

end problem_statement_l144_144419


namespace part1_part2_l144_144806

open Set

variable {α : Type*} [PartialOrder α]

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem part1 : A ∩ B = {x | 2 < x ∧ x < 3} :=
by
  sorry

theorem part2 : (compl B) = {x | x ≤ 1 ∨ x ≥ 3} :=
by
  sorry

end part1_part2_l144_144806


namespace tamara_is_17_over_6_times_taller_than_kim_l144_144785

theorem tamara_is_17_over_6_times_taller_than_kim :
  ∀ (T K : ℕ), T = 68 → T + K = 92 → (T : ℚ) / K = 17 / 6 :=
by
  intros T K hT hSum
  -- proof steps go here, but we use sorry to skip the proof
  sorry

end tamara_is_17_over_6_times_taller_than_kim_l144_144785


namespace student_scores_marks_per_correct_answer_l144_144670

theorem student_scores_marks_per_correct_answer
  (total_questions : ℕ) (total_marks : ℤ) (correct_questions : ℕ)
  (wrong_questions : ℕ) (marks_wrong_answer : ℤ)
  (x : ℤ) (h1 : total_questions = 60) (h2 : total_marks = 110)
  (h3 : correct_questions = 34) (h4 : wrong_questions = total_questions - correct_questions)
  (h5 : marks_wrong_answer = -1) :
  34 * x - 26 = 110 → x = 4 := by
  sorry

end student_scores_marks_per_correct_answer_l144_144670


namespace psychology_majors_percentage_in_liberal_arts_l144_144175

theorem psychology_majors_percentage_in_liberal_arts 
  (total_students : ℕ) 
  (percent_freshmen : ℝ) 
  (percent_freshmen_liberal_arts : ℝ) 
  (percent_freshmen_psych_majors_liberal_arts : ℝ) 
  (h1: percent_freshmen = 0.40) 
  (h2: percent_freshmen_liberal_arts = 0.50)
  (h3: percent_freshmen_psych_majors_liberal_arts = 0.10) :
  ((percent_freshmen_psych_majors_liberal_arts / (percent_freshmen * percent_freshmen_liberal_arts)) * 100 = 50) :=
by
  sorry

end psychology_majors_percentage_in_liberal_arts_l144_144175


namespace transformed_parabola_is_correct_l144_144164

-- Definitions based on conditions
def original_parabola (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 3
def shifted_left (x : ℝ) : ℝ := original_parabola (x - 2)
def shifted_up (y : ℝ) : ℝ := y + 2

-- Theorem statement
theorem transformed_parabola_is_correct :
  ∀ x : ℝ, shifted_up (shifted_left x) = 3 * x^2 + 6 * x - 1 :=
by 
  -- Proof will be filled in here
  sorry

end transformed_parabola_is_correct_l144_144164


namespace reflection_matrix_correct_l144_144043

-- Definitions based on the conditions given in the problem
def reflect_over_line_y_eq_x_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 1], ![1, 0]]

-- The main theorem to state the equivalence
theorem reflection_matrix_correct :
  reflect_over_line_y_eq_x_matrix = ![![0, 1], ![1, 0]] :=
by
  sorry

end reflection_matrix_correct_l144_144043


namespace cost_per_bundle_l144_144546

-- Condition: each rose costs 500 won
def rose_price := 500

-- Condition: total number of roses
def total_roses := 200

-- Condition: number of bundles
def bundles := 25

-- Question: Prove the cost per bundle
theorem cost_per_bundle (rp : ℕ) (tr : ℕ) (b : ℕ) : rp = 500 → tr = 200 → b = 25 → (rp * tr) / b = 4000 :=
by
  intros h0 h1 h2
  sorry

end cost_per_bundle_l144_144546


namespace original_amount_in_cookie_jar_l144_144995

theorem original_amount_in_cookie_jar (doris_spent martha_spent money_left_in_jar original_amount : ℕ)
  (h1 : doris_spent = 6)
  (h2 : martha_spent = doris_spent / 2)
  (h3 : money_left_in_jar = 15)
  (h4 : original_amount = money_left_in_jar + doris_spent + martha_spent) :
  original_amount = 24 := 
sorry

end original_amount_in_cookie_jar_l144_144995


namespace smallest_value_of_M_l144_144535

theorem smallest_value_of_M :
  ∀ (a b c d e f g M : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 → g > 0 →
  a + b + c + d + e + f + g = 2024 →
  M = max (a + b) (max (b + c) (max (c + d) (max (d + e) (max (e + f) (f + g))))) →
  M = 338 :=
by
  intro a b c d e f g M ha hb hc hd he hf hg hsum hmax
  sorry

end smallest_value_of_M_l144_144535


namespace book_price_distribution_l144_144933

theorem book_price_distribution :
  ∃ (x y z: ℤ), 
  x + y + z = 109 ∧
  (34 * x + 27.5 * y + 17.5 * z : ℝ) = 2845 ∧
  (x - y : ℤ).natAbs ≤ 2 ∧ (y - z).natAbs ≤ 2 := 
sorry

end book_price_distribution_l144_144933


namespace veranda_width_l144_144054

def room_length : ℕ := 17
def room_width : ℕ := 12
def veranda_area : ℤ := 132

theorem veranda_width :
  ∃ (w : ℝ), (17 + 2 * w) * (12 + 2 * w) - 17 * 12 = 132 ∧ w = 2 :=
by
  use 2
  sorry

end veranda_width_l144_144054


namespace flower_combinations_l144_144081

theorem flower_combinations (t l : ℕ) (h : 4 * t + 3 * l = 60) : 
  ∃ (t_values : Finset ℕ), (∀ x ∈ t_values, 0 ≤ x ∧ x ≤ 15 ∧ x % 3 = 0) ∧
  t_values.card = 6 :=
sorry

end flower_combinations_l144_144081


namespace length_AB_l144_144487

noncomputable def parabola_p := 3
def x1_x2_sum := 6

theorem length_AB (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (h1 : x1 + x2 = x1_x2_sum)
  (h2 : (y1^2 = 6 * x1) ∧ (y2^2 = 6 * x2))
  : abs (x1 + parabola_p / 2 - (x2 + parabola_p / 2)) = 9 := by
  sorry

end length_AB_l144_144487


namespace weights_identical_l144_144468

theorem weights_identical (w : Fin 13 → ℤ) 
  (h : ∀ i, ∃ (A B : Finset (Fin 13)), A.card = 6 ∧ B.card = 6 ∧ A ∪ B = Finset.univ.erase i ∧ (A.sum w) = (B.sum w)) :
  ∀ i j, w i = w j :=
by
  sorry

end weights_identical_l144_144468


namespace divisibility_by_120_l144_144882

theorem divisibility_by_120 (n : ℕ) : 120 ∣ (n^7 - n^3) :=
sorry

end divisibility_by_120_l144_144882


namespace unit_prices_and_purchasing_schemes_l144_144022

theorem unit_prices_and_purchasing_schemes :
  ∃ (x y : ℕ),
    (14 * x + 8 * y = 1600) ∧
    (3 * x = 4 * y) ∧
    (x = 80) ∧ 
    (y = 60) ∧
    ∃ (m : ℕ), 
      (m ≥ 29) ∧ 
      (m ≤ 30) ∧ 
      (80 * m + 60 * (50 - m) ≤ 3600) ∧
      (m = 29 ∨ m = 30) := 
sorry

end unit_prices_and_purchasing_schemes_l144_144022


namespace balls_sold_l144_144650

theorem balls_sold (CP SP_total : ℕ) (loss : ℕ) (n : ℕ) :
  CP = 60 →
  SP_total = 720 →
  loss = 5 * CP →
  loss = n * CP - SP_total →
  n = 17 :=
by
  intros hCP hSP_total hloss htotal
  -- Your proof here
  sorry

end balls_sold_l144_144650


namespace compare_fractions_l144_144899

theorem compare_fractions : - (1 + 3 / 5) < -1.5 := 
by
  sorry

end compare_fractions_l144_144899


namespace largest_number_is_B_l144_144904

noncomputable def numA : ℝ := 7.196533
noncomputable def numB : ℝ := 7.19655555555555555555555555555555555555 -- 7.196\overline{5}
noncomputable def numC : ℝ := 7.1965656565656565656565656565656565 -- 7.19\overline{65}
noncomputable def numD : ℝ := 7.196596596596596596596596596596596 -- 7.1\overline{965}
noncomputable def numE : ℝ := 7.196519651965196519651965196519651 -- 7.\overline{1965}

theorem largest_number_is_B : 
  numB > numA ∧ numB > numC ∧ numB > numD ∧ numB > numE :=
by
  sorry

end largest_number_is_B_l144_144904


namespace sin_330_eq_neg_one_half_l144_144915

theorem sin_330_eq_neg_one_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l144_144915


namespace otimes_calc_1_otimes_calc_2_otimes_calc_3_l144_144408

def otimes (a b : Int) : Int :=
  a^2 - Int.natAbs b

theorem otimes_calc_1 : otimes (-2) 3 = 1 :=
by
  sorry

theorem otimes_calc_2 : otimes 5 (-4) = 21 :=
by
  sorry

theorem otimes_calc_3 : otimes (-3) (-1) = 8 :=
by
  sorry

end otimes_calc_1_otimes_calc_2_otimes_calc_3_l144_144408


namespace roots_inverse_sum_eq_two_thirds_l144_144648

theorem roots_inverse_sum_eq_two_thirds {x₁ x₂ : ℝ} (h1 : x₁ ^ 2 + 2 * x₁ - 3 = 0) (h2 : x₂ ^ 2 + 2 * x₂ - 3 = 0) : 
  (1 / x₁) + (1 / x₂) = 2 / 3 :=
sorry

end roots_inverse_sum_eq_two_thirds_l144_144648


namespace max_eggs_l144_144009

theorem max_eggs (x : ℕ) 
  (h1 : x < 200) 
  (h2 : x % 3 = 2) 
  (h3 : x % 4 = 3) 
  (h4 : x % 5 = 4) : 
  x = 179 := 
by
  sorry

end max_eggs_l144_144009


namespace extreme_points_l144_144056

noncomputable def f (x : ℝ) : ℝ :=
  x + 4 / x

theorem extreme_points (P : ℝ × ℝ) :
  (P = (2, f 2) ∨ P = (-2, f (-2))) ↔ 
  ∃ x : ℝ, x ≠ 0 ∧ (P = (x, f x)) ∧ 
    (∀ ε > 0, f (x - ε) < f x ∧ f x > f (x + ε) ∨ f (x - ε) > f x ∧ f x < f (x + ε)) := 
sorry

end extreme_points_l144_144056


namespace sum_infinite_series_l144_144613

theorem sum_infinite_series : ∑' n : ℕ, (4 * (n + 1) - 3) / (3 ^ (n + 1)) = 3 / 2 := by
    sorry

end sum_infinite_series_l144_144613


namespace baseball_games_per_month_l144_144240

theorem baseball_games_per_month (total_games : ℕ) (season_length : ℕ) (games_per_month : ℕ) :
  total_games = 14 → season_length = 2 → games_per_month = total_games / season_length → games_per_month = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end baseball_games_per_month_l144_144240


namespace avg_age_across_rooms_l144_144354

namespace AverageAgeProof

def Room := Type

-- Conditions
def people_in_room_a : ℕ := 8
def avg_age_room_a : ℕ := 35

def people_in_room_b : ℕ := 5
def avg_age_room_b : ℕ := 30

def people_in_room_c : ℕ := 7
def avg_age_room_c : ℕ := 25

-- Combined Calculations
def total_people := people_in_room_a + people_in_room_b + people_in_room_c
def total_age := (people_in_room_a * avg_age_room_a) + (people_in_room_b * avg_age_room_b) + (people_in_room_c * avg_age_room_c)

noncomputable def average_age : ℚ := total_age / total_people

-- Proof that the average age of all the people across the three rooms is 30.25
theorem avg_age_across_rooms : average_age = 30.25 := 
sorry

end AverageAgeProof

end avg_age_across_rooms_l144_144354


namespace nancy_weight_l144_144047

theorem nancy_weight (w : ℕ) (h : (60 * w) / 100 = 54) : w = 90 :=
by
  sorry

end nancy_weight_l144_144047


namespace min_abs_val_sum_l144_144065

noncomputable def abs_val_sum_min : ℝ := (4:ℝ)^(1/3)

theorem min_abs_val_sum (a b c : ℝ) (h : |(a - b) * (b - c) * (c - a)| = 1) :
  |a| + |b| + |c| >= abs_val_sum_min :=
sorry

end min_abs_val_sum_l144_144065


namespace coefficient_of_x_100_l144_144544

-- Define the polynomial P
noncomputable def P : Polynomial ℤ :=
  (Polynomial.C (-1) + Polynomial.X) *
  (Polynomial.C (-2) + Polynomial.X^2) *
  (Polynomial.C (-3) + Polynomial.X^3) *
  (Polynomial.C (-4) + Polynomial.X^4) *
  (Polynomial.C (-5) + Polynomial.X^5) *
  (Polynomial.C (-6) + Polynomial.X^6) *
  (Polynomial.C (-7) + Polynomial.X^7) *
  (Polynomial.C (-8) + Polynomial.X^8) *
  (Polynomial.C (-9) + Polynomial.X^9) *
  (Polynomial.C (-10) + Polynomial.X^10) *
  (Polynomial.C (-11) + Polynomial.X^11) *
  (Polynomial.C (-12) + Polynomial.X^12) *
  (Polynomial.C (-13) + Polynomial.X^13) *
  (Polynomial.C (-14) + Polynomial.X^14) *
  (Polynomial.C (-15) + Polynomial.X^15)

-- State the theorem
theorem coefficient_of_x_100 : P.coeff 100 = 445 :=
  by sorry

end coefficient_of_x_100_l144_144544


namespace distinct_solutions_eq_l144_144862

theorem distinct_solutions_eq : ∃! x : ℝ, abs (x - 5) = abs (x + 3) :=
by
  sorry

end distinct_solutions_eq_l144_144862


namespace cos4_x_minus_sin4_x_l144_144918

theorem cos4_x_minus_sin4_x (x : ℝ) (h : x = π / 12) : (Real.cos x) ^ 4 - (Real.sin x) ^ 4 = (Real.sqrt 3) / 2 := by
  sorry

end cos4_x_minus_sin4_x_l144_144918


namespace cubic_diff_l144_144911

theorem cubic_diff (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 40) : a^3 - b^3 = 208 :=
by
  sorry

end cubic_diff_l144_144911


namespace height_of_square_pyramid_is_13_l144_144166

noncomputable def square_pyramid_height (base_edge : ℝ) (adjacent_face_angle : ℝ) : ℝ :=
  let half_diagonal := base_edge * (Real.sqrt 2) / 2
  let sin_angle := Real.sin (adjacent_face_angle / 2 : ℝ)
  let opp_side := half_diagonal * sin_angle
  let height := half_diagonal * sin_angle / (Real.sqrt 3)
  height

theorem height_of_square_pyramid_is_13 :
  ∀ (base_edge : ℝ) (adjacent_face_angle : ℝ), 
  base_edge = 26 → 
  adjacent_face_angle = 120 → 
  square_pyramid_height base_edge adjacent_face_angle = 13 :=
by
  intros base_edge adjacent_face_angle h_base_edge h_adj_face_angle
  rw [h_base_edge, h_adj_face_angle]
  have half_diagonal := 26 * (Real.sqrt 2) / 2
  have sin_angle := Real.sin (120 / 2 : ℝ) -- sin 60 degrees
  have sqrt_three := Real.sqrt 3
  have height := (half_diagonal * sin_angle) / sqrt_three
  sorry

end height_of_square_pyramid_is_13_l144_144166


namespace find_fraction_l144_144140

def f (x : ℤ) : ℤ := 3 * x + 4
def g (x : ℤ) : ℤ := 4 * x - 3

theorem find_fraction :
  (f (g (f 2)):ℚ) / (g (f (g 2)):ℚ) = 115 / 73 := by
  sorry

end find_fraction_l144_144140


namespace wheel_moves_distance_in_one_hour_l144_144216

-- Definition of the given conditions
def rotations_per_minute : ℕ := 10
def distance_per_rotation : ℕ := 20
def minutes_per_hour : ℕ := 60

-- Theorem statement to prove the wheel moves 12000 cm in one hour
theorem wheel_moves_distance_in_one_hour : 
  rotations_per_minute * minutes_per_hour * distance_per_rotation = 12000 := 
by
  sorry

end wheel_moves_distance_in_one_hour_l144_144216


namespace is_factorization_l144_144190

-- given an equation A,
-- Prove A is factorization: 
-- i.e., x^3 - x = x * (x + 1) * (x - 1)

theorem is_factorization (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by sorry

end is_factorization_l144_144190


namespace question_1_question_2_l144_144131

open Real

theorem question_1 (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : ab < m / 2 → m > 2 := sorry

theorem question_2 (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) (h4 : 9 / a + 1 / b ≥ |x - 1| + |x + 2|) :
  -9/2 ≤ x ∧ x ≤ 7/2 := sorry

end question_1_question_2_l144_144131


namespace tina_sells_more_than_katya_l144_144023

noncomputable def katya_rev : ℝ := 8 * 1.5
noncomputable def ricky_rev : ℝ := 9 * 2.0
noncomputable def combined_rev : ℝ := katya_rev + ricky_rev
noncomputable def tina_target : ℝ := 2 * combined_rev
noncomputable def tina_glasses : ℝ := tina_target / 3.0
noncomputable def difference_glasses : ℝ := tina_glasses - 8

theorem tina_sells_more_than_katya :
  difference_glasses = 12 := by
  sorry

end tina_sells_more_than_katya_l144_144023


namespace find_a_equals_two_l144_144941

noncomputable def a := ((7 + 4 * Real.sqrt 3) ^ (1 / 2) - (7 - 4 * Real.sqrt 3) ^ (1 / 2)) / Real.sqrt 3

theorem find_a_equals_two : a = 2 := 
sorry

end find_a_equals_two_l144_144941


namespace second_half_takes_200_percent_longer_l144_144397

noncomputable def time_take (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

variable (total_distance : ℕ := 640)
variable (first_half_speed : ℕ := 80)
variable (average_speed : ℕ := 40)

theorem second_half_takes_200_percent_longer :
  let first_half_distance := total_distance / 2;
  let first_half_time := time_take first_half_distance first_half_speed;
  let total_time := time_take total_distance average_speed;
  let second_half_time := total_time - first_half_time;
  let time_increase := second_half_time - first_half_time;
  let percentage_increase := (time_increase * 100) / first_half_time;
  percentage_increase = 200 :=
by
  sorry

end second_half_takes_200_percent_longer_l144_144397


namespace show_revenue_l144_144051

theorem show_revenue (tickets_first_showing : ℕ) 
                     (tickets_second_showing : ℕ) 
                     (ticket_price : ℕ) :
                      tickets_first_showing = 200 →
                      tickets_second_showing = 3 * tickets_first_showing →
                      ticket_price = 25 →
                      (tickets_first_showing + tickets_second_showing) * ticket_price = 20000 :=
by
  intros h1 h2 h3
  have h4 : tickets_first_showing + tickets_second_showing = 800 := sorry -- Calculation step
  have h5 : (tickets_first_showing + tickets_second_showing) * ticket_price = 20000 := sorry -- Calculation step
  exact h5

end show_revenue_l144_144051


namespace square_area_and_diagonal_ratio_l144_144665

theorem square_area_and_diagonal_ratio
    (a b : ℕ)
    (h_perimeter : 4 * a = 16 * b) :
    (a = 4 * b) ∧ ((a^2) / (b^2) = 16) ∧ ((a * Real.sqrt 2) / (b * Real.sqrt 2) = 4) :=
  by
  sorry

end square_area_and_diagonal_ratio_l144_144665


namespace factor_x4_minus_81_l144_144162

theorem factor_x4_minus_81 :
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  intros x
  sorry

end factor_x4_minus_81_l144_144162


namespace factor_64_minus_16y_squared_l144_144565

theorem factor_64_minus_16y_squared (y : ℝ) : 
  64 - 16 * y^2 = 16 * (2 - y) * (2 + y) :=
by
  -- skipping the actual proof steps
  sorry

end factor_64_minus_16y_squared_l144_144565


namespace evaluate_fraction_l144_144509

noncomputable def evaluate_expression : ℚ := 
  1 / (2 - (1 / (2 - (1 / (2 - (1 / 3))))))
  
theorem evaluate_fraction :
  evaluate_expression = 5 / 7 :=
sorry

end evaluate_fraction_l144_144509


namespace base_of_minus4_pow3_l144_144493

theorem base_of_minus4_pow3 : ∀ (x : ℤ) (n : ℤ), (x, n) = (-4, 3) → x = -4 :=
by intros x n h
   cases h
   rfl

end base_of_minus4_pow3_l144_144493


namespace part_a_l144_144201

theorem part_a (a b c : ℕ) (h : (a + b + c) % 6 = 0) : (a^5 + b^3 + c) % 6 = 0 := 
by sorry

end part_a_l144_144201


namespace solve_for_x_l144_144322

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ -1 then x + 2 
  else if x < 2 then x^2 
  else 2 * x

theorem solve_for_x (x : ℝ) : f x = 3 ↔ x = Real.sqrt 3 := by
  sorry

end solve_for_x_l144_144322


namespace fourth_and_fifth_suppliers_cars_equal_l144_144937

-- Define the conditions
def total_cars : ℕ := 5650000
def cars_supplier_1 : ℕ := 1000000
def cars_supplier_2 : ℕ := cars_supplier_1 + 500000
def cars_supplier_3 : ℕ := cars_supplier_1 + cars_supplier_2
def cars_distributed_first_three : ℕ := cars_supplier_1 + cars_supplier_2 + cars_supplier_3
def cars_remaining : ℕ := total_cars - cars_distributed_first_three

-- Theorem stating the question and answer
theorem fourth_and_fifth_suppliers_cars_equal 
  : (cars_remaining / 2) = 325000 := by
  sorry

end fourth_and_fifth_suppliers_cars_equal_l144_144937


namespace sum_of_reciprocals_of_roots_l144_144370

theorem sum_of_reciprocals_of_roots {r1 r2 : ℚ} (h1 : r1 + r2 = 15) (h2 : r1 * r2 = 6) :
  (1 / r1 + 1 / r2) = 5 / 2 := 
by sorry

end sum_of_reciprocals_of_roots_l144_144370


namespace second_month_sales_l144_144610

def sales_first_month : ℝ := 7435
def sales_third_month : ℝ := 7855
def sales_fourth_month : ℝ := 8230
def sales_fifth_month : ℝ := 7562
def sales_sixth_month : ℝ := 5991
def average_sales : ℝ := 7500

theorem second_month_sales : 
  ∃ (second_month_sale : ℝ), 
    (sales_first_month + second_month_sale + sales_third_month + sales_fourth_month + sales_fifth_month + sales_sixth_month) / 6 = average_sales ∧
    second_month_sale = 7927 := by
  sorry

end second_month_sales_l144_144610


namespace quadruple_equation_solution_count_l144_144715

theorem quadruple_equation_solution_count (
    a b c d : ℕ
) (h_pos: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_order: a < b ∧ b < c ∧ c < d) 
  (h_equation: 2 * a + 2 * b + 2 * c + 2 * d = d^2 - c^2 + b^2 - a^2) : 
  num_correct_statements = 2 :=
sorry

end quadruple_equation_solution_count_l144_144715


namespace problem_solution_l144_144462

noncomputable def solution_set : Set ℝ :=
  { x : ℝ | x ∈ (Set.Ioo 0 (5 - Real.sqrt 10)) ∨ x ∈ (Set.Ioi (5 + Real.sqrt 10)) }

theorem problem_solution (x : ℝ) : (x^3 - 10*x^2 + 15*x > 0) ↔ x ∈ solution_set :=
by
  sorry

end problem_solution_l144_144462


namespace value_of_A_l144_144792

theorem value_of_A (G F L: ℤ) (H1 : G = 15) (H2 : F + L + 15 = 50) (H3 : F + L + 37 + 15 = 65) (H4 : F + ((58 - F - L) / 2) + ((58 - F - L) / 2) + L = 58) : 
  37 = 37 := 
by 
  sorry

end value_of_A_l144_144792


namespace sandra_fathers_contribution_ratio_l144_144469

theorem sandra_fathers_contribution_ratio :
  let saved := 10
  let mother := 4
  let candy_cost := 0.5
  let jellybean_cost := 0.2
  let candies := 14
  let jellybeans := 20
  let remaining := 11
  let total_cost := candies * candy_cost + jellybeans * jellybean_cost
  let total_amount := total_cost + remaining
  let amount_without_father := saved + mother
  let father := total_amount - amount_without_father
  (father / mother) = 2 := by 
  sorry

end sandra_fathers_contribution_ratio_l144_144469


namespace min_value_frac_inverse_l144_144573

theorem min_value_frac_inverse (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : 
  (1 / a + 1 / b) >= 2 :=
by
  sorry

end min_value_frac_inverse_l144_144573


namespace andy_incorrect_l144_144137

theorem andy_incorrect (a b c d : ℕ) (h1 : a + b = c + d) (h2 : a + d = b + c + 6) (h3 : c = 8) : a = 14 :=
by
  sorry

end andy_incorrect_l144_144137


namespace circle_m_condition_l144_144849

theorem circle_m_condition (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2*x + 4*y + m = 0) → m < 5 :=
by
  sorry

end circle_m_condition_l144_144849


namespace find_a_l144_144740

theorem find_a (a : ℝ) (h1 : a^2 + 2 * a - 15 = 0) (h2 : a^2 + 4 * a - 5 ≠ 0) :
  a = 3 :=
by
sorry

end find_a_l144_144740


namespace ginger_size_l144_144773

theorem ginger_size (anna_size : ℕ) (becky_size : ℕ) (ginger_size : ℕ) 
  (h1 : anna_size = 2) 
  (h2 : becky_size = 3 * anna_size) 
  (h3 : ginger_size = 2 * becky_size - 4) : 
  ginger_size = 8 :=
by
  -- The proof is omitted, just the theorem statement is required.
  sorry

end ginger_size_l144_144773


namespace cookies_per_batch_l144_144099

theorem cookies_per_batch
  (bag_chips : ℕ)
  (batches : ℕ)
  (chips_per_cookie : ℕ)
  (total_chips : ℕ)
  (h1 : bag_chips = total_chips)
  (h2 : batches = 3)
  (h3 : chips_per_cookie = 9)
  (h4 : total_chips = 81) :
  (bag_chips / batches) / chips_per_cookie = 3 := 
by
  sorry

end cookies_per_batch_l144_144099


namespace union_A_B_l144_144411

def setA : Set ℝ := {x | (x + 1) * (x - 2) < 0}
def setB : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem union_A_B : setA ∪ setB = {x | -1 < x ∧ x ≤ 3} :=
by
  sorry

end union_A_B_l144_144411


namespace password_probability_l144_144932

theorem password_probability 
  (password : Fin 6 → Fin 10) 
  (attempts : ℕ) 
  (correct_digit : Fin 10) 
  (probability_first_try : ℚ := 1 / 10)
  (probability_second_try : ℚ := (9 / 10) * (1 / 9)) : 
  ((password 5 = correct_digit) ∧ attempts ≤ 2) →
  (probability_first_try + probability_second_try = 1 / 5) :=
sorry

end password_probability_l144_144932


namespace ten_percent_markup_and_markdown_l144_144664

theorem ten_percent_markup_and_markdown (x : ℝ) (hx : x > 0) : 0.99 * x < x :=
by 
  sorry

end ten_percent_markup_and_markdown_l144_144664


namespace tim_prank_combinations_l144_144229

def number_of_combinations (monday_choices : ℕ) (tuesday_choices : ℕ) (wednesday_choices : ℕ) (thursday_choices : ℕ) (friday_choices : ℕ) : ℕ :=
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices

theorem tim_prank_combinations : number_of_combinations 2 3 0 6 1 = 0 :=
by
  -- Calculation yields 2 * 3 * 0 * 6 * 1 = 0
  sorry

end tim_prank_combinations_l144_144229


namespace chess_team_girls_l144_144001

theorem chess_team_girls (B G : ℕ) (h1 : B + G = 26) (h2 : (G / 2) + B = 16) : G = 20 := by
  sorry

end chess_team_girls_l144_144001


namespace min_value_ineq_l144_144247

theorem min_value_ineq (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_point_on_chord : ∃ x y : ℝ, x = 4 * a ∧ y = 2 * b ∧ (x + y = 2) ∧ (x^2 + y^2 = 4) ∧ ((x - 2)^2 + (y - 2)^2 = 4)) :
  1 / a + 2 / b ≥ 8 :=
by
  sorry

end min_value_ineq_l144_144247


namespace weights_in_pile_l144_144395

theorem weights_in_pile (a b c : ℕ) (h1 : a + b + c = 100) (h2 : a + 10 * b + 50 * c = 500) : 
  a = 60 ∧ b = 39 ∧ c = 1 :=
sorry

end weights_in_pile_l144_144395


namespace problem_l144_144518

theorem problem 
  (a b A B : ℝ)
  (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 :=
by sorry

end problem_l144_144518


namespace find_original_price_of_petrol_l144_144078

open Real

noncomputable def original_price_of_petrol (P : ℝ) : Prop :=
  ∀ G : ℝ, 
  (G * P = 300) ∧ 
  ((G + 7) * 0.85 * P = 300) → 
  P = 7.56

-- Theorems should ideally be defined within certain scopes or namespaces
theorem find_original_price_of_petrol (P : ℝ) : original_price_of_petrol P :=
  sorry

end find_original_price_of_petrol_l144_144078


namespace highest_wave_height_l144_144952

-- Definitions of surfboard length and shortest wave conditions
def surfboard_length : ℕ := 7
def shortest_wave_height (H : ℕ) : ℕ := H + 4

-- Lean statement to be proved
theorem highest_wave_height (H : ℕ) (condition1 : H + 4 = surfboard_length + 3) : 
  4 * H + 2 = 26 :=
sorry

end highest_wave_height_l144_144952


namespace linear_regression_equation_demand_prediction_l144_144801

def data_x : List ℝ := [12, 11, 10, 9, 8]
def data_y : List ℝ := [5, 6, 8, 10, 11]

noncomputable def mean_x : ℝ := (12 + 11 + 10 + 9 + 8) / 5
noncomputable def mean_y : ℝ := (5 + 6 + 8 + 10 + 11) / 5

noncomputable def numerator : ℝ := 
  (12 - mean_x) * (5 - mean_y) + 
  (11 - mean_x) * (6 - mean_y) +
  (10 - mean_x) * (8 - mean_y) +
  (9 - mean_x) * (10 - mean_y) +
  (8 - mean_x) * (11 - mean_y)

noncomputable def denominator : ℝ := 
  (12 - mean_x)^2 + 
  (11 - mean_x)^2 +
  (10 - mean_x)^2 +
  (9 - mean_x)^2 +
  (8 - mean_x)^2

noncomputable def slope_b : ℝ := numerator / denominator
noncomputable def intercept_a : ℝ := mean_y - slope_b * mean_x

theorem linear_regression_equation :
  (slope_b = -1.6) ∧ (intercept_a = 24) :=
by
  sorry

noncomputable def predicted_y (x : ℝ) : ℝ :=
  slope_b * x + intercept_a

theorem demand_prediction :
  predicted_y 6 = 14.4 ∧ (predicted_y 6 < 15) :=
by
  sorry

end linear_regression_equation_demand_prediction_l144_144801


namespace probability_snow_first_week_l144_144974

theorem probability_snow_first_week :
  let p1 := 1/4
  let p2 := 1/3
  let no_snow := (3/4)^4 * (2/3)^3
  let snows_at_least_once := 1 - no_snow
  snows_at_least_once = 29 / 32 := by
  sorry

end probability_snow_first_week_l144_144974


namespace min_total_cost_minimize_cost_l144_144445

theorem min_total_cost (x : ℝ) (h₀ : x > 0) :
  (900 / x * 3 + 3 * x) ≥ 180 :=
by sorry

theorem minimize_cost (x : ℝ) (h₀ : x > 0) :
  x = 30 ↔ (900 / x * 3 + 3 * x) = 180 :=
by sorry

end min_total_cost_minimize_cost_l144_144445


namespace evaluate_expression_l144_144089

theorem evaluate_expression : 
  (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3) :=
by
  sorry

end evaluate_expression_l144_144089


namespace tangent_circle_line_l144_144777

theorem tangent_circle_line (a : ℝ) :
  (∀ x y : ℝ, (x - y + 3 = 0) → (x^2 + y^2 - 2 * x + 2 - a = 0)) →
  a = 9 :=
by
  sorry

end tangent_circle_line_l144_144777


namespace molly_age_l144_144440

theorem molly_age
  (S M : ℕ)
  (h1 : S / M = 4 / 3)
  (h2 : S + 6 = 30) :
  M = 18 :=
sorry

end molly_age_l144_144440


namespace scientific_notation_11580000_l144_144371

theorem scientific_notation_11580000 :
  11580000 = 1.158 * 10^7 :=
sorry

end scientific_notation_11580000_l144_144371


namespace sqrt_condition_sqrt_not_meaningful_2_l144_144752

theorem sqrt_condition (x : ℝ) : 1 - x ≥ 0 ↔ x ≤ 1 := 
by
  sorry

theorem sqrt_not_meaningful_2 : ¬(1 - 2 ≥ 0) :=
by
  sorry

end sqrt_condition_sqrt_not_meaningful_2_l144_144752


namespace shaded_area_possible_values_l144_144461

variable (AB BC PQ SC : ℕ)

-- Conditions:
def dimensions_correct : Prop := AB * BC = 33 ∧ AB < 7 ∧ BC < 7
def length_constraint : Prop := PQ < SC

-- Theorem statement
theorem shaded_area_possible_values (h1 : dimensions_correct AB BC) (h2 : length_constraint PQ SC) :
  (AB = 3 ∧ BC = 11 ∧ (PQ = 1 ∧ SC = 6 ∧ (33 - 1 * 4 - 2 * 6 = 17) ∨
                      (33 - 2 * 3 - 1 * 6 = 21) ∨
                      (33 - 2 * 4 - 1 * 5 = 20))) ∨ 
  (AB = 11 ∧ BC = 3 ∧ (PQ = 1 ∧ SC = 6 ∧ (33 - 1 * 4 - 2 * 6 = 17))) :=
sorry

end shaded_area_possible_values_l144_144461


namespace stuffed_animal_sales_l144_144049

theorem stuffed_animal_sales (Q T J : ℕ) 
  (h1 : Q = 100 * T) 
  (h2 : J = T + 15) 
  (h3 : Q = 2000) : 
  Q - J = 1965 := 
by
  sorry

end stuffed_animal_sales_l144_144049


namespace find_a4_plus_b4_l144_144071

theorem find_a4_plus_b4 (a b : ℝ)
  (h1 : (a^2 - b^2)^2 = 100)
  (h2 : a^3 * b^3 = 512) :
  a^4 + b^4 = 228 :=
by
  sorry

end find_a4_plus_b4_l144_144071


namespace find_OH_squared_l144_144259

theorem find_OH_squared (R a b c : ℝ) (hR : R = 10) (hsum : a^2 + b^2 + c^2 = 50) : 
  9 * R^2 - (a^2 + b^2 + c^2) = 850 :=
by
  sorry

end find_OH_squared_l144_144259


namespace blue_marbles_difference_l144_144516

theorem blue_marbles_difference  (a b : ℚ) 
  (h1 : 3 * a + 2 * b = 80)
  (h2 : 2 * a = b) :
  (7 * a - 3 * b) = 80 / 7 := by
  sorry

end blue_marbles_difference_l144_144516


namespace probability_at_least_two_students_succeeding_l144_144566

-- The probabilities of each student succeeding
def p1 : ℚ := 1 / 2
def p2 : ℚ := 1 / 4
def p3 : ℚ := 1 / 5

/-- Calculation of the total probability that at least two out of the three students succeed -/
theorem probability_at_least_two_students_succeeding : 
  (p1 * p2 * (1 - p3)) + (p1 * (1 - p2) * p3) + ((1 - p1) * p2 * p3) + (p1 * p2 * p3) = 9 / 40 :=
  sorry

end probability_at_least_two_students_succeeding_l144_144566


namespace students_walk_fraction_l144_144523

theorem students_walk_fraction :
  (1 - (1/3 + 1/5 + 1/10 + 1/15)) = 3/10 :=
by sorry

end students_walk_fraction_l144_144523


namespace largest_integer_x_l144_144737

theorem largest_integer_x (x : ℤ) (h : 3 - 5 * x > 22) : x ≤ -4 :=
by
  sorry

end largest_integer_x_l144_144737


namespace eggs_total_l144_144962

-- Definitions based on the conditions
def breakfast_eggs : Nat := 2
def lunch_eggs : Nat := 3
def dinner_eggs : Nat := 1

-- Theorem statement
theorem eggs_total : breakfast_eggs + lunch_eggs + dinner_eggs = 6 :=
by
  -- This part will be filled in with proof steps, but it's omitted here
  sorry

end eggs_total_l144_144962


namespace smallest_number_of_cubes_l144_144506

theorem smallest_number_of_cubes (l w d : ℕ) (hl : l = 36) (hw : w = 45) (hd : d = 18) : 
  ∃ n : ℕ, n = 40 ∧ (∃ s : ℕ, l % s = 0 ∧ w % s = 0 ∧ d % s = 0 ∧ (l / s) * (w / s) * (d / s) = n) := 
by
  sorry

end smallest_number_of_cubes_l144_144506


namespace rows_of_pies_l144_144634

theorem rows_of_pies (baked_pecan_pies : ℕ) (baked_apple_pies : ℕ) (pies_per_row : ℕ) : 
  baked_pecan_pies = 16 ∧ baked_apple_pies = 14 ∧ pies_per_row = 5 → 
  (baked_pecan_pies + baked_apple_pies) / pies_per_row = 6 :=
by
  sorry

end rows_of_pies_l144_144634


namespace tangent_sum_half_angles_l144_144163

-- Lean statement for the proof problem
theorem tangent_sum_half_angles (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.tan (A / 2) * Real.tan (B / 2) + 
  Real.tan (B / 2) * Real.tan (C / 2) + 
  Real.tan (C / 2) * Real.tan (A / 2) = 1 := 
by
  sorry

end tangent_sum_half_angles_l144_144163


namespace necklace_wire_length_l144_144531

theorem necklace_wire_length
  (spools : ℕ)
  (feet_per_spool : ℕ)
  (total_necklaces : ℕ)
  (h1 : spools = 3)
  (h2 : feet_per_spool = 20)
  (h3 : total_necklaces = 15) :
  (spools * feet_per_spool) / total_necklaces = 4 := by
  sorry

end necklace_wire_length_l144_144531


namespace decreasing_on_negative_interval_and_max_value_l144_144790

open Classical

noncomputable def f : ℝ → ℝ := sorry  -- Define f later

variables {f : ℝ → ℝ}

-- Hypotheses
axiom h_even : ∀ x, f x = f (-x)
axiom h_increasing_0_7 : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x ≤ y → y ≤ 7 → f x ≤ f y
axiom h_decreasing_7_inf : ∀ ⦃x y : ℝ⦄, 7 ≤ x → x ≤ y → f x ≥ f y
axiom h_f_7_6 : f 7 = 6

-- Theorem Statement
theorem decreasing_on_negative_interval_and_max_value :
  (∀ ⦃x y : ℝ⦄, -7 ≤ x → x ≤ y → y ≤ 0 → f x ≥ f y) ∧ (∀ x, -7 ≤ x → x ≤ 0 → f x ≤ 6) :=
by
  sorry

end decreasing_on_negative_interval_and_max_value_l144_144790


namespace quadratic_expression_and_intersections_l144_144868

noncomputable def quadratic_eq_expression (a b c : ℝ) : Prop :=
  ∃ a b c : ℝ, (a * (1:ℝ) ^ 2 + b * (1:ℝ) + c = -3) ∧ (4 * a + 2 * b + c = - 5 / 2) ∧ (b = -2 * a) ∧ (c = -5 / 2) ∧ (a = 1 / 2)

noncomputable def find_m (a b c : ℝ) : Prop :=
  ∀ x m : ℝ, (a * (-2:ℝ)^2 + b * (-2:ℝ) + c = m) → (a * (4:ℝ) + b * (4:ℝ) + c = m) → (6:ℝ) = abs (x - (-2:ℝ)) → m = 3 / 2

noncomputable def y_range (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, 
  (x^2 * a + x * b + c >= -3) ∧ 
  (x^2 * a + x * b + c < 5) ↔ (-3 < x ∧ x < 3)

theorem quadratic_expression_and_intersections 
  (a b c : ℝ) (h1 : quadratic_eq_expression a b c) (h2 : find_m a b c) : y_range a b c :=
  sorry

end quadratic_expression_and_intersections_l144_144868


namespace odd_numbers_in_A_even_4k_minus_2_not_in_A_product_in_A_l144_144313

def is_in_A (a : ℤ) : Prop := ∃ (x y : ℤ), a = x^2 - y^2

theorem odd_numbers_in_A :
  ∀ (n : ℤ), n % 2 = 1 → is_in_A n :=
sorry

theorem even_4k_minus_2_not_in_A :
  ∀ (k : ℤ), ¬ is_in_A (4 * k - 2) :=
sorry

theorem product_in_A :
  ∀ (a b : ℤ), is_in_A a → is_in_A b → is_in_A (a * b) :=
sorry

end odd_numbers_in_A_even_4k_minus_2_not_in_A_product_in_A_l144_144313


namespace total_distance_travelled_l144_144698

/-- Proving that the total horizontal distance traveled by the centers of two wheels with radii 1 m and 2 m 
    after one complete revolution is 6π meters. -/
theorem total_distance_travelled (R1 R2 : ℝ) (h1 : R1 = 1) (h2 : R2 = 2) : 
    2 * Real.pi * R1 + 2 * Real.pi * R2 = 6 * Real.pi :=
by
  sorry

end total_distance_travelled_l144_144698


namespace problem_proof_l144_144450

theorem problem_proof (M N : ℕ) 
  (h1 : 4 * 63 = 7 * M) 
  (h2 : 4 * N = 7 * 84) : 
  M + N = 183 :=
sorry

end problem_proof_l144_144450


namespace solution_set_inequality_l144_144471

open Real

theorem solution_set_inequality (k : ℤ) (x : ℝ) :
  (x ∈ Set.Ioo (-π/4 + k * π) (k * π)) ↔ cos (4 * x) - 2 * sin (2 * x) - sin (4 * x) - 1 > 0 :=
by
  sorry

end solution_set_inequality_l144_144471


namespace ratio_of_wages_l144_144591

def hours_per_day_josh : ℕ := 8
def days_per_week : ℕ := 5
def weeks_per_month : ℕ := 4
def wage_per_hour_josh : ℕ := 9
def monthly_total_payment : ℚ := 1980

def hours_per_day_carl : ℕ := hours_per_day_josh - 2

def monthly_hours_josh : ℕ := hours_per_day_josh * days_per_week * weeks_per_month
def monthly_hours_carl : ℕ := hours_per_day_carl * days_per_week * weeks_per_month

def monthly_earnings_josh : ℚ := wage_per_hour_josh * monthly_hours_josh
def monthly_earnings_carl : ℚ := monthly_total_payment - monthly_earnings_josh

def hourly_wage_carl : ℚ := monthly_earnings_carl / monthly_hours_carl

theorem ratio_of_wages : hourly_wage_carl / wage_per_hour_josh = 1 / 2 := by
  sorry

end ratio_of_wages_l144_144591


namespace hcf_of_two_numbers_l144_144898

theorem hcf_of_two_numbers (H : ℕ) 
(lcm_def : lcm a b = H * 13 * 14) 
(h : a = 280 ∨ b = 280) 
(is_factor_h : H ∣ 280) : 
H = 5 :=
sorry

end hcf_of_two_numbers_l144_144898


namespace increase_in_circumference_l144_144797

theorem increase_in_circumference (d e : ℝ) : (fun d e => let C := π * d; let C_new := π * (d + e); C_new - C) d e = π * e :=
by sorry

end increase_in_circumference_l144_144797


namespace arithmetic_sequence_geometric_l144_144459

theorem arithmetic_sequence_geometric (a : ℕ → ℤ) (d : ℤ) (m n : ℕ)
  (h1 : ∀ n, a (n+1) = a 1 + n * d)
  (h2 : a 1 = 1)
  (h3 : (a 3 - 2)^2 = a 1 * a 5)
  (h_d_nonzero : d ≠ 0)
  (h_mn : m - n = 10) :
  a m - a n = 30 := 
by
  sorry

end arithmetic_sequence_geometric_l144_144459


namespace sum_a_b_l144_144944

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem sum_a_b (a b : ℝ) 
  (H : ∀ x, 2 < x ∧ x < 3 → otimes (x - a) (x - b) > 0) : a + b = 4 :=
by
  sorry

end sum_a_b_l144_144944


namespace triangle_ABC_properties_l144_144711

theorem triangle_ABC_properties 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : 2 * Real.sin B * Real.sin C * Real.cos A + Real.cos A = 3 * Real.sin A ^ 2 - Real.cos (B - C)) : 
  (2 * a = b + c) ∧ 
  (b + c = 2) →
  (Real.cos A = 3/5) → 
  (1 / 2 * b * c * Real.sin A = 3 / 8) :=
by
  sorry

end triangle_ABC_properties_l144_144711


namespace f_triple_application_l144_144157

-- Define the function f : ℕ → ℕ such that f(x) = 3x + 2
def f (x : ℕ) : ℕ := 3 * x + 2

-- Theorem statement to prove f(f(f(1))) = 53
theorem f_triple_application : f (f (f 1)) = 53 := 
by 
  sorry

end f_triple_application_l144_144157


namespace subway_length_in_meters_l144_144675

noncomputable def subway_speed : ℝ := 1.6 -- km per minute
noncomputable def crossing_time : ℝ := 3 + 15 / 60 -- minutes
noncomputable def bridge_length : ℝ := 4.85 -- km

theorem subway_length_in_meters :
  let total_distance_traveled := subway_speed * crossing_time
  let subway_length_km := total_distance_traveled - bridge_length
  let subway_length_m := subway_length_km * 1000
  subway_length_m = 350 :=
by
  sorry

end subway_length_in_meters_l144_144675


namespace cindy_marbles_problem_l144_144626

theorem cindy_marbles_problem
  (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ)
  (h1 : initial_marbles = 500) (h2 : friends = 4) (h3 : marbles_per_friend = 80) :
  4 * (initial_marbles - (marbles_per_friend * friends)) = 720 :=
by
  sorry

end cindy_marbles_problem_l144_144626


namespace deepak_speed_proof_l144_144376

noncomputable def deepak_speed (circumference : ℝ) (meeting_time : ℝ) (wife_speed_kmh : ℝ) : ℝ :=
  let wife_speed_mpm := wife_speed_kmh * 1000 / 60
  let wife_distance := wife_speed_mpm * meeting_time
  let deepak_speed_mpm := ((circumference - wife_distance) / meeting_time)
  deepak_speed_mpm * 60 / 1000

theorem deepak_speed_proof :
  deepak_speed 726 5.28 3.75 = 4.5054 :=
by
  -- The functions and definitions used here come from the problem statement
  -- Conditions:
  -- circumference = 726
  -- meeting_time = 5.28 minutes
  -- wife_speed_kmh = 3.75 km/hr
  sorry

end deepak_speed_proof_l144_144376


namespace part1_solution_set_part2_range_of_a_l144_144225

-- Defining the function f(x) under given conditions
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

-- Part 1: Determine the solution set for the inequality when a = 2
theorem part1_solution_set (x : ℝ) : (f x 2) ≥ 4 ↔ x ≤ 3/2 ∨ x ≥ 11/2 := by
  sorry

-- Part 2: Determine the range of values for a given the inequality
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end part1_solution_set_part2_range_of_a_l144_144225


namespace difference_between_twice_smaller_and_larger_is_three_l144_144021

theorem difference_between_twice_smaller_and_larger_is_three
(S L x : ℕ) 
(h1 : L = 2 * S - x) 
(h2 : S + L = 39)
(h3 : S = 14) : 
2 * S - L = 3 := 
sorry

end difference_between_twice_smaller_and_larger_is_three_l144_144021


namespace open_door_within_time_l144_144513

-- Define the initial conditions
def device := ℕ → ℕ

-- Constraint: Each device has 5 toggle switches ("0" or "1") and a three-digit display.
def valid_configuration (d : device) (k : ℕ) : Prop :=
  d k < 32 ∧ d k <= 999

def system_configuration (A B : device) (k : ℕ) : Prop :=
  A k = B k

-- Constraint: The devices can be synchronized to display the same number simultaneously to open the door.
def open_door (A B : device) : Prop :=
  ∃ k, system_configuration A B k

-- The main theorem: Devices A and B can be synchronized within the given time constraints to open the door.
theorem open_door_within_time (A B : device) (notebook : ℕ) : 
  (∀ k, valid_configuration A k ∧ valid_configuration B k) →
  open_door A B :=
by sorry

end open_door_within_time_l144_144513


namespace divisor_is_five_l144_144580

theorem divisor_is_five (n d : ℕ) (h1 : ∃ k, n = k * d + 3) (h2 : ∃ l, n^2 = l * d + 4) : d = 5 :=
sorry

end divisor_is_five_l144_144580


namespace newOp_of_M_and_N_l144_144155

def newOp (A B : Set ℕ) : Set ℕ :=
  {x | x ∈ A ∨ x ∈ B ∧ x ∉ (A ∩ B)}

theorem newOp_of_M_and_N (M N : Set ℕ) :
  M = {0, 2, 4, 6, 8, 10} →
  N = {0, 3, 6, 9, 12, 15} →
  newOp (newOp M N) M = N :=
by
  intros hM hN
  sorry

end newOp_of_M_and_N_l144_144155


namespace total_tiles_l144_144568

theorem total_tiles (n : ℕ) (h : 2 * n - 1 = 133) : n^2 = 4489 :=
by
  sorry

end total_tiles_l144_144568


namespace most_stable_scores_l144_144907

structure StudentScores :=
  (average : ℝ)
  (variance : ℝ)

def studentA : StudentScores := { average := 132, variance := 38 }
def studentB : StudentScores := { average := 132, variance := 10 }
def studentC : StudentScores := { average := 132, variance := 26 }

theorem most_stable_scores :
  studentB.variance < studentA.variance ∧ studentB.variance < studentC.variance :=
by 
  sorry

end most_stable_scores_l144_144907


namespace product_of_sum_positive_and_quotient_negative_l144_144295

-- Definitions based on conditions in the problem
def sum_positive (a b : ℝ) : Prop := a + b > 0
def quotient_negative (a b : ℝ) : Prop := a / b < 0

-- Problem statement as a theorem
theorem product_of_sum_positive_and_quotient_negative (a b : ℝ)
  (h1 : sum_positive a b)
  (h2 : quotient_negative a b) :
  a * b < 0 := by
  sorry

end product_of_sum_positive_and_quotient_negative_l144_144295


namespace inequality_proof_l144_144232

variable {x : ℝ}
variable {n : ℕ}
variable {a : ℝ}

theorem inequality_proof (h1 : x > 0) (h2 : n > 0) (h3 : x + a / x^n ≥ n + 1) : a = n^n := 
sorry

end inequality_proof_l144_144232


namespace tip_percentage_l144_144165

theorem tip_percentage (cost_of_crown : ℕ) (total_paid : ℕ) (h1 : cost_of_crown = 20000) (h2 : total_paid = 22000) :
  (total_paid - cost_of_crown) * 100 / cost_of_crown = 10 :=
by
  sorry

end tip_percentage_l144_144165


namespace complement_M_l144_144280

noncomputable def U : Set ℝ := Set.univ

def M : Set ℝ := { x | x^2 - 4 ≤ 0 }

theorem complement_M : U \ M = { x | x < -2 ∨ x > 2 } :=
by 
  sorry

end complement_M_l144_144280


namespace total_fish_correct_l144_144705

def Leo_fish := 40
def Agrey_fish := Leo_fish + 20
def Sierra_fish := Agrey_fish + 15
def total_fish := Leo_fish + Agrey_fish + Sierra_fish

theorem total_fish_correct : total_fish = 175 := by
  sorry


end total_fish_correct_l144_144705


namespace solution_x_chemical_b_l144_144285

theorem solution_x_chemical_b (percentage_x_a percentage_y_a percentage_y_b : ℝ) :
  percentage_x_a = 0.3 →
  percentage_y_a = 0.4 →
  percentage_y_b = 0.6 →
  (0.8 * percentage_x_a + 0.2 * percentage_y_a = 0.32) →
  (100 * (1 - percentage_x_a) = 70) :=
by {
  sorry
}

end solution_x_chemical_b_l144_144285


namespace rectangle_area_is_180_l144_144020

def area_of_square (side : ℕ) : ℕ := side * side
def length_of_rectangle (radius : ℕ) : ℕ := (2 * radius) / 5
def area_of_rectangle (length breadth : ℕ) : ℕ := length * breadth

theorem rectangle_area_is_180 :
  ∀ (side breadth : ℕ), 
    area_of_square side = 2025 → 
    breadth = 10 → 
    area_of_rectangle (length_of_rectangle side) breadth = 180 :=
by
  intros side breadth h_area h_breadth
  sorry

end rectangle_area_is_180_l144_144020


namespace alex_jamie_casey_probability_l144_144820

-- Probability definitions and conditions
def alex_win_prob := 1/3
def casey_win_prob := 1/6
def jamie_win_prob := 1/2

def total_rounds := 8
def alex_wins := 4
def jamie_wins := 3
def casey_wins := 1

-- The probability computation
theorem alex_jamie_casey_probability : 
  alex_win_prob ^ alex_wins * jamie_win_prob ^ jamie_wins * casey_win_prob ^ casey_wins * (Nat.choose total_rounds (alex_wins + jamie_wins + casey_wins)) = 35 / 486 := 
sorry

end alex_jamie_casey_probability_l144_144820


namespace max_value_of_seq_l144_144073

theorem max_value_of_seq (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n, S n = -n^2 + 6 * n + 7)
  (h_a_def : ∀ n, a n = S n - S (n - 1)) : ∃ max_val, max_val = 12 ∧ ∀ n, a n ≤ max_val :=
by
  sorry

end max_value_of_seq_l144_144073


namespace common_difference_l144_144324

-- Definitions
variable (a₁ d : ℝ) -- First term and common difference of the arithmetic sequence

-- Conditions
def mean_nine_terms (a₁ d : ℝ) : Prop :=
  (1 / 2 * (2 * a₁ + 8 * d)) = 10

def mean_ten_terms (a₁ d : ℝ) : Prop :=
  (1 / 2 * (2 * a₁ + 9 * d)) = 13

-- Theorem to prove the common difference is 6
theorem common_difference (a₁ d : ℝ) :
  mean_nine_terms a₁ d → 
  mean_ten_terms a₁ d → 
  d = 6 := by
  intros
  sorry

end common_difference_l144_144324


namespace problem_statement_l144_144835

variable (x y : ℝ)

theorem problem_statement
  (h1 : 4 * x + y = 9)
  (h2 : x + 4 * y = 16) :
  18 * x^2 + 20 * x * y + 18 * y^2 = 337 :=
sorry

end problem_statement_l144_144835


namespace old_barbell_cost_l144_144788

theorem old_barbell_cost (x : ℝ) (new_barbell_cost : ℝ) (h1 : new_barbell_cost = 1.30 * x) (h2 : new_barbell_cost = 325) : x = 250 :=
by
  sorry

end old_barbell_cost_l144_144788


namespace pats_and_mats_numbers_l144_144556

theorem pats_and_mats_numbers (x y : ℕ) (hxy : x ≠ y) (hx_gt_hy : x > y) 
    (h_sum : (x + y) + (x - y) + x * y + (x / y) = 98) : x = 12 ∧ y = 6 :=
by
  sorry

end pats_and_mats_numbers_l144_144556


namespace yogurt_combinations_l144_144537

theorem yogurt_combinations : (4 * Nat.choose 8 3) = 224 := by
  sorry

end yogurt_combinations_l144_144537


namespace find_tricksters_in_16_questions_l144_144262

-- Definitions
def Inhabitant := {i : Nat // i < 65}
def isKnight (i : Inhabitant) : Prop := sorry  -- placeholder for actual definition
def isTrickster (i : Inhabitant) : Prop := sorry -- placeholder for actual definition

-- Conditions
def condition1 : ∀ i : Inhabitant, isKnight i ∨ isTrickster i := sorry
def condition2 : ∃ t1 t2 : Inhabitant, t1 ≠ t2 ∧ isTrickster t1 ∧ isTrickster t2 :=
  sorry
def condition3 : ∀ t1 t2 : Inhabitant, t1 ≠ t2 → ¬(isTrickster t1 ∧ isTrickster t2) → isKnight t1 ∧ isKnight t2 := 
  sorry 
def question (i : Inhabitant) (group : List Inhabitant) : Prop :=
  ∀ j ∈ group, isKnight j

-- Theorem statement
theorem find_tricksters_in_16_questions : ∃ (knight : Inhabitant) (knaves : (Inhabitant × Inhabitant)), 
  isKnight knight ∧ isTrickster knaves.fst ∧ isTrickster knaves.snd ∧ knaves.fst ≠ knaves.snd ∧
  (∀ questionsAsked ≤ 30, sorry) :=
  sorry

end find_tricksters_in_16_questions_l144_144262


namespace stickers_after_birthday_l144_144404

-- Definitions based on conditions
def initial_stickers : Nat := 39
def birthday_stickers : Nat := 22

-- Theorem stating the problem we aim to prove
theorem stickers_after_birthday : initial_stickers + birthday_stickers = 61 :=
by 
  sorry

end stickers_after_birthday_l144_144404


namespace even_function_phi_l144_144947

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + φ)

noncomputable def f' (x φ : ℝ) : ℝ := -Real.sqrt 3 * Real.sin (Real.sqrt 3 * x + φ)

noncomputable def y (x φ : ℝ) : ℝ := f x φ + f' x φ

def is_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = g (-x)

theorem even_function_phi :
  (∀ x : ℝ, y x φ = y (-x) φ) → ∃ k : ℤ, φ = -Real.pi / 3 + k * Real.pi :=
by
  sorry

end even_function_phi_l144_144947
