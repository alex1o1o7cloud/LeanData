import Mathlib

namespace yoongi_more_points_l45_45083

def yoongiPoints : ℕ := 4
def jungkookPoints : ℕ := 6 - 3

theorem yoongi_more_points : yoongiPoints > jungkookPoints := by
  sorry

end yoongi_more_points_l45_45083


namespace movie_theater_charge_l45_45750

theorem movie_theater_charge 
    (charge_adult : ℝ) 
    (children : ℕ) 
    (adults : ℕ) 
    (total_receipts : ℝ) 
    (charge_child : ℝ) 
    (condition1 : charge_adult = 6.75) 
    (condition2 : children = adults + 20) 
    (condition3 : total_receipts = 405) 
    (condition4 : children = 48) 
    : charge_child = 4.5 :=
sorry

end movie_theater_charge_l45_45750


namespace Jake_needs_to_lose_12_pounds_l45_45168

theorem Jake_needs_to_lose_12_pounds (J S : ℕ) (h1 : J + S = 156) (h2 : J = 108) : J - 2 * S = 12 := by
  sorry

end Jake_needs_to_lose_12_pounds_l45_45168


namespace circle_radius_seven_l45_45925

theorem circle_radius_seven (k : ℝ) :
  (∃ x y : ℝ, (x^2 + 12 * x + y^2 + 8 * y - k = 0)) ↔ (k = -3) :=
by
  sorry

end circle_radius_seven_l45_45925


namespace mean_sharpening_instances_l45_45988

def pencil_sharpening_instances : List ℕ :=
  [13, 8, 13, 21, 7, 23, 15, 19, 12, 9, 28, 6, 17, 29, 31, 10, 4, 20, 16, 12, 2, 18, 27, 22, 5, 14, 31, 29, 8, 25]

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem mean_sharpening_instances :
  mean pencil_sharpening_instances = 18.1 := by
  sorry

end mean_sharpening_instances_l45_45988


namespace rhombus_perimeter_and_radius_l45_45715

-- Define the rhombus with given diagonals
structure Rhombus where
  d1 : ℝ -- diagonal 1
  d2 : ℝ -- diagonal 2
  h : d1 = 20 ∧ d2 = 16

-- Define the proof problem
theorem rhombus_perimeter_and_radius (r : Rhombus) : 
  let side_length := Real.sqrt ((r.d1 / 2) ^ 2 + (r.d2 / 2) ^ 2)
  let perimeter := 4 * side_length
  let radius := r.d1 / 2
  perimeter = 16 * Real.sqrt 41 ∧ radius = 10 :=
by
  sorry

end rhombus_perimeter_and_radius_l45_45715


namespace parabola_axis_of_symmetry_is_x_eq_1_l45_45636

theorem parabola_axis_of_symmetry_is_x_eq_1 :
  ∀ x : ℝ, ∀ y : ℝ, y = -2 * (x - 1)^2 + 3 → (∀ c : ℝ, c = 1 → ∃ x1 x2 : ℝ, x1 = c ∧ x2 = c) := 
by
  sorry

end parabola_axis_of_symmetry_is_x_eq_1_l45_45636


namespace geometric_sequence_a3_l45_45929

theorem geometric_sequence_a3 (a : ℕ → ℝ) (q : ℝ) (h1 : a 4 = a 1 * q ^ 3) (h2 : a 2 = a 1 * q) (h3 : a 5 = a 1 * q ^ 4) 
    (h4 : a 4 - a 2 = 6) (h5 : a 5 - a 1 = 15) : a 3 = 4 ∨ a 3 = -4 :=
by
  sorry

end geometric_sequence_a3_l45_45929


namespace monotonicity_and_k_range_l45_45333

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x + (1 / 2 : ℝ) * x^2 - x

theorem monotonicity_and_k_range :
  (∀ x : ℝ, x ≥ 0 → f x ≥ k * x - 2) ↔ k ∈ Set.Iic (-2) := sorry

end monotonicity_and_k_range_l45_45333


namespace matrix_inverse_correct_l45_45048

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, -2], ![5, 3]]

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![3/22, 1/11], ![-5/22, 2/11]]

theorem matrix_inverse_correct : A⁻¹ = A_inv :=
  by
    sorry

end matrix_inverse_correct_l45_45048


namespace arithmetic_geometric_sequence_l45_45509

theorem arithmetic_geometric_sequence (d : ℤ) (a_1 a_2 a_5 : ℤ)
  (h1 : d ≠ 0)
  (h2 : a_2 = a_1 + d)
  (h3 : a_5 = a_1 + 4 * d)
  (h4 : a_2 ^ 2 = a_1 * a_5) :
  a_5 = 9 * a_1 := 
sorry

end arithmetic_geometric_sequence_l45_45509


namespace inequality_proof_l45_45521

theorem inequality_proof (a b x : ℝ) (h : a > b) : a * 2 ^ x > b * 2 ^ x :=
sorry

end inequality_proof_l45_45521


namespace complement_intersection_l45_45280

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2 * x > 0}

-- Define complement of A in U
def C_U_A : Set ℝ := U \ A

-- Define set B
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem complement_intersection (x : ℝ) : x ∈ C_U_A ∩ B ↔ 1 < x ∧ x ≤ 2 :=
by
   sorry

end complement_intersection_l45_45280


namespace converted_land_eqn_l45_45268

theorem converted_land_eqn (forest_land dry_land converted_dry_land : ℝ)
  (h1 : forest_land = 108)
  (h2 : dry_land = 54)
  (h3 : converted_dry_land = x) :
  (dry_land - converted_dry_land = 0.2 * (forest_land + converted_dry_land)) :=
by
  simp [h1, h2, h3]
  sorry

end converted_land_eqn_l45_45268


namespace january_first_is_tuesday_l45_45243

-- Define the days of the week for convenience
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define the problem conditions
def daysInJanuary : Nat := 31
def weeksInJanuary : Nat := daysInJanuary / 7   -- This is 4 weeks
def extraDays : Nat := daysInJanuary % 7         -- This leaves 3 extra days

-- Define the problem as proving January 1st is a Tuesday
theorem january_first_is_tuesday (fridaysInJanuary : Nat) (mondaysInJanuary : Nat)
    (h_friday : fridaysInJanuary = 4) (h_monday: mondaysInJanuary = 4) : Weekday :=
  -- Avoid specific proof steps from the solution; assume conditions and directly prove the result
  sorry

end january_first_is_tuesday_l45_45243


namespace james_spends_90_dollars_per_week_l45_45546

structure PistachioPurchasing where
  can_cost : ℕ  -- cost in dollars per can
  can_weight : ℕ -- weight in ounces per can
  consumption_oz_per_5days : ℕ -- consumption in ounces per 5 days

def cost_per_week (p : PistachioPurchasing) : ℕ :=
  let daily_consumption := p.consumption_oz_per_5days / 5
  let weekly_consumption := daily_consumption * 7
  let cans_needed := (weekly_consumption + p.can_weight - 1) / p.can_weight -- round up
  cans_needed * p.can_cost

theorem james_spends_90_dollars_per_week :
  cost_per_week ⟨10, 5, 30⟩ = 90 :=
by
  sorry

end james_spends_90_dollars_per_week_l45_45546


namespace true_statement_count_l45_45169

def reciprocal (n : ℕ) : ℚ := 1 / n

def statement_i := (reciprocal 4 + reciprocal 8 = reciprocal 12)
def statement_ii := (reciprocal 9 - reciprocal 3 = reciprocal 6)
def statement_iii := (reciprocal 3 * reciprocal 9 = reciprocal 27)
def statement_iv := (reciprocal 15 / reciprocal 3 = reciprocal 5)

theorem true_statement_count :
  (¬statement_i ∧ ¬statement_ii ∧ statement_iii ∧ statement_iv) ↔ (2 = 2) :=
by sorry

end true_statement_count_l45_45169


namespace stratified_sample_l45_45279

theorem stratified_sample 
  (total_households : ℕ) 
  (high_income_households : ℕ) 
  (middle_income_households : ℕ) 
  (low_income_households : ℕ) 
  (sample_size : ℕ)
  (H1 : total_households = 600) 
  (H2 : high_income_households = 150)
  (H3 : middle_income_households = 360)
  (H4 : low_income_households = 90)
  (H5 : sample_size = 100) : 
  (middle_income_households * sample_size / total_households = 60) := 
by 
  sorry

end stratified_sample_l45_45279


namespace extreme_points_of_f_range_of_a_l45_45937

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x ≥ -1 then Real.log (x + 1) + a * (x^2 - x) 
  else 0

theorem extreme_points_of_f (a : ℝ) :
  (a < 0 → ∃ x, f a x = 0) ∧
  (0 ≤ a ∧ a ≤ 8/9 → ∃! x, f a x = 0) ∧
  (a > 8/9 → ∃ x₁ x₂, x₁ < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end extreme_points_of_f_range_of_a_l45_45937


namespace probability_same_color_given_first_red_l45_45236

-- Definitions of events
def event_A (draw1 : ℕ) : Prop := draw1 = 1 -- Event A: the first ball drawn is red (drawing 1 means the first ball is red)

def event_B (draw1 draw2 : ℕ) : Prop := -- Event B: the two balls drawn are of the same color
  (draw1 = 1 ∧ draw2 = 1) ∨ (draw1 = 2 ∧ draw2 = 2)

-- Given probabilities
def P_A : ℚ := 2 / 5
def P_AB : ℚ := (2 / 5) * (1 / 4)

-- The conditional probability P(B|A)
def P_B_given_A : ℚ := P_AB / P_A

theorem probability_same_color_given_first_red : P_B_given_A = 1 / 4 := 
by 
  unfold P_B_given_A P_A P_AB
  sorry

end probability_same_color_given_first_red_l45_45236


namespace batsman_average_after_17th_l45_45495

theorem batsman_average_after_17th (A : ℤ) (h1 : 86 + 16 * A = 17 * (A + 3)) : A + 3 = 38 :=
by
  sorry

end batsman_average_after_17th_l45_45495


namespace pairs_xy_solution_sum_l45_45209

theorem pairs_xy_solution_sum :
  ∃ (x y : ℝ) (a b c d : ℕ), 
    x + y = 5 ∧ 2 * x * y = 5 ∧ 
    (x = (5 + Real.sqrt 15) / 2 ∨ x = (5 - Real.sqrt 15) / 2) ∧ 
    a = 5 ∧ b = 1 ∧ c = 15 ∧ d = 2 ∧ a + b + c + d = 23 :=
by
  sorry

end pairs_xy_solution_sum_l45_45209


namespace determine_n_l45_45023

variable (x a n : ℕ)

def binomial_term (n k : ℕ) (x a : ℤ) : ℤ :=
  Nat.choose n k * x ^ (n - k) * a ^ k

theorem determine_n (hx : 0 < x) (ha : 0 < a)
  (h4 : binomial_term n 3 x a = 330)
  (h5 : binomial_term n 4 x a = 792)
  (h6 : binomial_term n 5 x a = 1716) :
  n = 7 :=
sorry

end determine_n_l45_45023


namespace sum_of_primes_less_than_10_is_17_l45_45229

-- Definition of prime numbers less than 10
def primes_less_than_10 : List ℕ := [2, 3, 5, 7]

-- Sum of the prime numbers less than 10
def sum_primes_less_than_10 : ℕ := List.sum primes_less_than_10

theorem sum_of_primes_less_than_10_is_17 : sum_primes_less_than_10 = 17 := 
by
  sorry

end sum_of_primes_less_than_10_is_17_l45_45229


namespace rice_price_per_kg_l45_45021

theorem rice_price_per_kg (price1 price2 : ℝ) (amount1 amount2 : ℝ) (total_cost total_weight : ℝ) (P : ℝ)
  (h1 : price1 = 6.60)
  (h2 : amount1 = 49)
  (h3 : price2 = 9.60)
  (h4 : amount2 = 56)
  (h5 : total_cost = price1 * amount1 + price2 * amount2)
  (h6 : total_weight = amount1 + amount2)
  (h7 : P = total_cost / total_weight) :
  P = 8.20 := 
by sorry

end rice_price_per_kg_l45_45021


namespace find_x_l45_45261

theorem find_x (x : ℝ) (h : x * 1.6 - (2 * 1.4) / 1.3 = 4) : x = 3.846154 :=
sorry

end find_x_l45_45261


namespace eggs_needed_per_month_l45_45384

def saly_needs : ℕ := 10
def ben_needs : ℕ := 14
def ked_needs : ℕ := ben_needs / 2
def weeks_in_month : ℕ := 4

def total_weekly_need : ℕ := saly_needs + ben_needs + ked_needs
def total_monthly_need : ℕ := total_weekly_need * weeks_in_month

theorem eggs_needed_per_month : total_monthly_need = 124 := by
  sorry

end eggs_needed_per_month_l45_45384


namespace pencils_added_by_mike_l45_45015

-- Definitions and assumptions based on conditions
def initial_pencils : ℕ := 41
def final_pencils : ℕ := 71

-- Statement of the problem
theorem pencils_added_by_mike : final_pencils - initial_pencils = 30 := 
by 
  sorry

end pencils_added_by_mike_l45_45015


namespace number_of_children_l45_45114

theorem number_of_children (A V S : ℕ) (x : ℕ → ℕ) (n : ℕ) 
  (h1 : (A / 2) + V = (A + V + S + (Finset.range (n - 3)).sum x) / n)
  (h2 : S + A = V + (Finset.range (n - 3)).sum x) : 
  n = 6 :=
sorry

end number_of_children_l45_45114


namespace length_A_l45_45473

open Real

theorem length_A'B'_correct {A B C A' B' : ℝ × ℝ} :
  A = (0, 10) →
  B = (0, 15) →
  C = (3, 9) →
  (A'.1 = A'.2) →
  (B'.1 = B'.2) →
  (C.2 - A.2) / (C.1 - A.1) = ((B.2 - C.2) / (B.1 - C.1)) →
  (dist A' B') = 2.5 * sqrt 2 :=
by
  intros
  sorry

end length_A_l45_45473


namespace charge_per_kilo_l45_45329

variable (x : ℝ)

theorem charge_per_kilo (h : 5 * x + 10 * x + 20 * x = 70) : x = 2 := by
  -- Proof goes here
  sorry

end charge_per_kilo_l45_45329


namespace arithmetic_sequence_length_l45_45142

theorem arithmetic_sequence_length 
  (a₁ : ℕ) (d : ℤ) (x : ℤ) (n : ℕ) 
  (h_start : a₁ = 20)
  (h_diff : d = -2)
  (h_eq : x = 10)
  (h_term : x = a₁ + (n - 1) * d) :
  n = 6 :=
by
  sorry

end arithmetic_sequence_length_l45_45142


namespace group_8_extracted_number_is_72_l45_45933

-- Definitions related to the problem setup
def individ_to_group (n : ℕ) : ℕ := n / 10 + 1
def unit_digit (n : ℕ) : ℕ := n % 10
def extraction_rule (k m : ℕ) : ℕ := (k + m - 1) % 10

-- Given condition: total individuals split into sequential groups and m = 5
def total_individuals : ℕ := 100
def total_groups : ℕ := 10
def m : ℕ := 5
def k_8 : ℕ := 8

-- The final theorem statement
theorem group_8_extracted_number_is_72 : ∃ n : ℕ, individ_to_group n = k_8 ∧ unit_digit n = extraction_rule k_8 m := by
  sorry

end group_8_extracted_number_is_72_l45_45933


namespace distance_sum_is_ten_l45_45623

noncomputable def angle_sum_distance (C A B : ℝ) (d : ℝ) (k : ℝ) : ℝ := 
  let h_A : ℝ := sorry -- replace with expression for h_A based on conditions
  let h_B : ℝ := sorry -- replace with expression for h_B based on conditions
  h_A + h_B

theorem distance_sum_is_ten 
  (A B C : ℝ) 
  (h : ℝ) 
  (k : ℝ) 
  (h_pos : h = 4) 
  (ratio_condition : h_A = 4 * h_B)
  : angle_sum_distance C A B h k = 10 := 
  sorry

end distance_sum_is_ten_l45_45623


namespace work_problem_l45_45586

theorem work_problem (x : ℕ) (b_work : ℕ) (a_b_together_work : ℕ) (h1: b_work = 24) (h2: a_b_together_work = 8) :
  (1 / x) + (1 / b_work) = (1 / a_b_together_work) → x = 12 :=
by 
  intros h_eq
  have h_b : b_work = 24 := h1
  have h_ab : a_b_together_work = 8 := h2
  -- Full proof is omitted
  sorry

end work_problem_l45_45586


namespace range_of_m_if_not_p_and_q_l45_45393

def p (m : ℝ) : Prop := 2 < m

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem range_of_m_if_not_p_and_q (m : ℝ) : ¬ p m ∧ q m → 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_if_not_p_and_q_l45_45393


namespace books_already_read_l45_45203

def total_books : ℕ := 20
def unread_books : ℕ := 5

theorem books_already_read : (total_books - unread_books = 15) :=
by
 -- Proof goes here
 sorry

end books_already_read_l45_45203


namespace find_a8_l45_45678

theorem find_a8 (a : ℕ → ℝ) 
  (h_arith_seq : ∀ n : ℕ, (1 / (a n + 1)) = (1 / (a 0 + 1)) + n * ((1 / (a 1 + 1 - 1)) / 3)) 
  (h2 : a 2 = 3) 
  (h5 : a 5 = 1) : 
  a 8 = 1 / 3 :=
by
  sorry

end find_a8_l45_45678


namespace num_three_digit_powers_of_three_l45_45620

theorem num_three_digit_powers_of_three : 
  ∃ n1 n2 : ℕ, 100 ≤ 3^n1 ∧ 3^n1 ≤ 999 ∧ 100 ≤ 3^n2 ∧ 3^n2 ≤ 999 ∧ n1 ≠ n2 ∧ 
  (∀ n : ℕ, 100 ≤ 3^n ∧ 3^n ≤ 999 → n = n1 ∨ n = n2) :=
sorry

end num_three_digit_powers_of_three_l45_45620


namespace total_sales_first_three_days_total_earnings_seven_days_l45_45992

def planned_daily_sales : Int := 100

def deviation : List Int := [4, -3, -5, 14, -8, 21, -6]

def selling_price_per_pound : Int := 8
def freight_cost_per_pound : Int := 3

-- Part (1): Proof statement for the total amount sold in the first three days
theorem total_sales_first_three_days :
  let monday_sales := planned_daily_sales + deviation.head!
  let tuesday_sales := planned_daily_sales + (deviation.drop 1).head!
  let wednesday_sales := planned_daily_sales + (deviation.drop 2).head!
  monday_sales + tuesday_sales + wednesday_sales = 296 := by
  sorry

-- Part (2): Proof statement for Xiaoming's total earnings for the seven days
theorem total_earnings_seven_days :
  let total_sales := (List.sum (deviation.map (λ x => planned_daily_sales + x)))
  total_sales * (selling_price_per_pound - freight_cost_per_pound) = 3585 := by
  sorry

end total_sales_first_three_days_total_earnings_seven_days_l45_45992


namespace part1_part2_l45_45215

-- Define the coordinates of point P as functions of n
def pointP (n : ℝ) : ℝ × ℝ := (n + 3, 2 - 3 * n)

-- Condition 1: Point P is in the fourth quadrant
def inFourthQuadrant (n : ℝ) : Prop :=
  let point := pointP n
  point.1 > 0 ∧ point.2 < 0

-- Condition 2: Distance from P to the x-axis is 1 greater than the distance to the y-axis
def distancesCondition (n : ℝ) : Prop :=
  abs (2 - 3 * n) + 1 = abs (n + 3)

-- Definition of point Q
def pointQ (n : ℝ) : ℝ × ℝ := (n, -4)

-- Condition 3: PQ is parallel to the x-axis
def pqParallelX (n : ℝ) : Prop :=
  (pointP n).2 = (pointQ n).2

-- Theorems to prove the coordinates of point P and the length of PQ
theorem part1 (n : ℝ) (h1 : inFourthQuadrant n) (h2 : distancesCondition n) : pointP n = (6, -7) :=
sorry

theorem part2 (n : ℝ) (h1 : pqParallelX n) : abs ((pointP n).1 - (pointQ n).1) = 3 :=
sorry

end part1_part2_l45_45215


namespace smallest_n_divisibility_problem_l45_45862

theorem smallest_n_divisibility_problem :
  ∃ n : ℕ, (∀ k : ℕ, (1 ≤ k ∧ k ≤ n → ¬(n^2 + n) % k = 0)) ∧ n = 4 :=
by
  sorry

end smallest_n_divisibility_problem_l45_45862


namespace relationship_p_q_l45_45130

noncomputable def p (α β : ℝ) : ℝ := Real.cos α * Real.cos β
noncomputable def q (α β : ℝ) : ℝ := Real.cos ((α + β) / 2) ^ 2

theorem relationship_p_q (α β : ℝ) : p α β ≤ q α β :=
by
  sorry

end relationship_p_q_l45_45130


namespace find_number_satisfies_l45_45828

noncomputable def find_number (m : ℤ) (n : ℤ) : Prop :=
  (m % n = 2) ∧ (3 * m % n = 1)

theorem find_number_satisfies (m : ℤ) : ∃ n : ℤ, find_number m n ∧ n = 5 :=
by
  sorry

end find_number_satisfies_l45_45828


namespace parking_space_area_l45_45526

theorem parking_space_area (L W : ℕ) (h1 : L = 9) (h2 : L + 2 * W = 37) : L * W = 126 := by
  sorry

end parking_space_area_l45_45526


namespace halfway_between_l45_45998

theorem halfway_between (a b : ℚ) (h₁ : a = 1/8) (h₂ : b = 1/3) : (a + b) / 2 = 11 / 48 := 
by
  sorry

end halfway_between_l45_45998


namespace hours_per_trainer_l45_45976

-- Define the conditions from part (a)
def number_of_dolphins : ℕ := 4
def hours_per_dolphin : ℕ := 3
def number_of_trainers : ℕ := 2

-- Define the theorem we want to prove using the answer from part (b)
theorem hours_per_trainer : (number_of_dolphins * hours_per_dolphin) / number_of_trainers = 6 :=
by
  -- Proof goes here
  sorry

end hours_per_trainer_l45_45976


namespace annie_extracurricular_hours_l45_45909

-- Definitions based on conditions
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3
def weeks_per_semester : ℕ := 12
def weeks_off_sick : ℕ := 2

-- Total hours of extracurricular activities per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Number of active weeks before midterms
def active_weeks_before_midterms : ℕ := weeks_per_semester - weeks_off_sick

-- Total hours of extracurricular activities before midterms
def total_hours_before_midterms : ℕ := total_hours_per_week * active_weeks_before_midterms

-- Proof statement
theorem annie_extracurricular_hours : total_hours_before_midterms = 130 := by
  sorry

end annie_extracurricular_hours_l45_45909


namespace pond_volume_extraction_l45_45287

/--
  Let length (l), width (w), and depth (h) be dimensions of a pond.
  Given:
  l = 20,
  w = 10,
  h = 5,
  Prove that the volume of the soil extracted from the pond is 1000 cubic meters.
-/
theorem pond_volume_extraction (l w h : ℕ) (hl : l = 20) (hw : w = 10) (hh : h = 5) :
  l * w * h = 1000 :=
  by
    sorry

end pond_volume_extraction_l45_45287


namespace find_t_l45_45251

theorem find_t (t a b : ℝ) :
  (∀ x : ℝ, (3 * x^2 - 4 * x + 5) * (5 * x^2 + t * x + 12) = 15 * x^4 - 47 * x^3 + a * x^2 + b * x + 60) →
  t = -9 :=
by
  intros h
  -- We'll skip the proof part
  sorry

end find_t_l45_45251


namespace apples_from_C_to_D_l45_45648

theorem apples_from_C_to_D (n m : ℕ)
  (h_tree_ratio : ∀ (P V : ℕ), P = 2 * V)
  (h_apple_ratio : ∀ (P V : ℕ), P = 7 * V)
  (trees_CD_Petya trees_CD_Vasya : ℕ)
  (h_trees_CD : trees_CD_Petya = 2 * trees_CD_Vasya)
  (apples_CD_Petya apples_CD_Vasya: ℕ)
  (h_apples_CD : apples_CD_Petya = (m / 4) ∧ apples_CD_Vasya = (3 * m / 4)) : 
  apples_CD_Vasya = 3 * apples_CD_Petya := by
  sorry

end apples_from_C_to_D_l45_45648


namespace molecular_weight_N2O5_correct_l45_45858

noncomputable def atomic_weight_N : ℝ := 14.01
noncomputable def atomic_weight_O : ℝ := 16.00
def molecular_formula_N2O5 : (ℕ × ℕ) := (2, 5)

theorem molecular_weight_N2O5_correct :
  let weight := (2 * atomic_weight_N) + (5 * atomic_weight_O)
  weight = 108.02 :=
by
  sorry

end molecular_weight_N2O5_correct_l45_45858


namespace part_a_part_b_l45_45719

namespace TrihedralAngle

-- Part (a)
theorem part_a (α β γ : ℝ) (h1 : β = 70) (h2 : γ = 100) (h3 : α < β + γ) (h4 : β < α + γ) (h5 : γ < α + β) (h6 : α + β + γ < 360) : 
    30 < α ∧ α < 170 := 
sorry

-- Part (b)
theorem part_b (α β γ : ℝ) (h1 : β = 130) (h2 : γ = 150) (h3 : α < β + γ) (h4 : β < α + γ) (h5 : γ < α + β) (h6 : α + β + γ < 360) : 
    20 < α ∧ α < 80 := 
sorry

end TrihedralAngle

end part_a_part_b_l45_45719


namespace avg_growth_rate_selling_price_reduction_l45_45957

open Real

-- Define the conditions for the first question
def sales_volume_aug : ℝ := 50000
def sales_volume_oct : ℝ := 72000

-- Define the conditions for the second question
def cost_price_per_unit : ℝ := 40
def initial_selling_price_per_unit : ℝ := 80
def initial_sales_volume_per_day : ℝ := 20
def additional_units_per_half_dollar_decrease : ℝ := 4
def desired_daily_profit : ℝ := 1400

-- First proof: monthly average growth rate
theorem avg_growth_rate (x : ℝ) :
  sales_volume_aug * (1 + x)^2 = sales_volume_oct → x = 0.2 :=
by {
  sorry
}

-- Second proof: reduction in selling price for daily profit
theorem selling_price_reduction (y : ℝ) :
  (initial_selling_price_per_unit - y - cost_price_per_unit) * (initial_sales_volume_per_day + additional_units_per_half_dollar_decrease * y / 0.5) = desired_daily_profit → y = 30 :=
by {
  sorry
}

end avg_growth_rate_selling_price_reduction_l45_45957


namespace calculate_power_of_fractions_l45_45942

-- Defining the fractions
def a : ℚ := 5 / 6
def b : ℚ := 3 / 5

-- The main statement to prove the given question
theorem calculate_power_of_fractions : a^3 + b^3 = (21457 : ℚ) / 27000 := by 
  sorry

end calculate_power_of_fractions_l45_45942


namespace average_of_three_marbles_l45_45668

-- Define the conditions as hypotheses
theorem average_of_three_marbles (R Y B : ℕ) 
  (h1 : R + Y = 53)
  (h2 : B + Y = 69)
  (h3 : R + B = 58) :
  (R + Y + B) / 3 = 30 :=
by
  sorry

end average_of_three_marbles_l45_45668


namespace abc_mod_n_l45_45986

theorem abc_mod_n (n : ℕ) (a b c : ℤ) (hn : 0 < n)
  (h1 : a * b ≡ 1 [ZMOD n])
  (h2 : c ≡ b [ZMOD n]) : (a * b * c) ≡ 1 [ZMOD n] := sorry

end abc_mod_n_l45_45986


namespace total_operations_in_one_hour_l45_45881

theorem total_operations_in_one_hour :
  let additions_per_second := 12000
  let multiplications_per_second := 8000
  (additions_per_second + multiplications_per_second) * 3600 = 72000000 :=
by
  sorry

end total_operations_in_one_hour_l45_45881


namespace symmetric_points_addition_l45_45852

theorem symmetric_points_addition (m n : ℤ) (h₁ : m = 2) (h₂ : n = -3) : m + n = -1 := by
  rw [h₁, h₂]
  norm_num

end symmetric_points_addition_l45_45852


namespace parabola_vertex_position_l45_45964

def f (x : ℝ) : ℝ := x^2 - 2 * x + 5
def g (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem parabola_vertex_position (x y : ℝ) :
  (∃ a b : ℝ, f a = y ∧ g b = y ∧ a = 1 ∧ b = -1)
  → (1 > -1) ∧ (f 1 > g (-1)) :=
by
  sorry

end parabola_vertex_position_l45_45964


namespace evaluate_expression_at_3_l45_45081

theorem evaluate_expression_at_3 :
  (∀ x ≠ 2, (x = 3) → (x^2 - 5 * x + 6) / (x - 2) = 0) :=
by
  sorry

end evaluate_expression_at_3_l45_45081


namespace simplify_expression_l45_45707

theorem simplify_expression :
  (↑(Real.sqrt 648) / ↑(Real.sqrt 81) - ↑(Real.sqrt 245) / ↑(Real.sqrt 49)) = 2 * Real.sqrt 2 - Real.sqrt 5 := by
  -- proof omitted
  sorry

end simplify_expression_l45_45707


namespace fraction_to_zero_power_l45_45073

theorem fraction_to_zero_power :
  756321948 ≠ 0 ∧ -3958672103 ≠ 0 →
  (756321948 / -3958672103 : ℝ) ^ 0 = 1 :=
by
  intro h
  have numerator_nonzero : 756321948 ≠ 0 := h.left
  have denominator_nonzero : -3958672103 ≠ 0 := h.right
  -- Skipping the rest of the proof.
  sorry

end fraction_to_zero_power_l45_45073


namespace train_distance_problem_l45_45644

theorem train_distance_problem
  (Vx : ℝ) (Vy : ℝ) (t : ℝ) (distanceX : ℝ) 
  (h1 : Vx = 32) 
  (h2 : Vy = 160 / 3) 
  (h3 : 32 * t + (160 / 3) * t = 160) :
  distanceX = Vx * t → distanceX = 60 :=
by {
  sorry
}

end train_distance_problem_l45_45644


namespace min_n_satisfies_inequality_l45_45867

theorem min_n_satisfies_inequality :
  ∃ n : ℕ, 0 < n ∧ -3 * (n : ℤ) ^ 4 + 5 * (n : ℤ) ^ 2 - 199 < 0 ∧ (∀ m : ℕ, 0 < m ∧ -3 * (m : ℤ) ^ 4 + 5 * (m : ℤ) ^ 2 - 199 < 0 → 2 ≤ m) := 
  sorry

end min_n_satisfies_inequality_l45_45867


namespace average_speed_l45_45980

theorem average_speed (v : ℝ) (v_pos : 0 < v) (v_pos_10 : 0 < v + 10):
  420 / v - 420 / (v + 10) = 2 → v = 42 :=
by
  sorry

end average_speed_l45_45980


namespace calculate_expression_l45_45230

def f (x : ℝ) := 2 * x^2 - 3 * x + 1
def g (x : ℝ) := x + 2

theorem calculate_expression : f (1 + g 3) = 55 := 
by
  sorry

end calculate_expression_l45_45230


namespace min_value_of_b_minus_2c_plus_1_over_a_l45_45295

theorem min_value_of_b_minus_2c_plus_1_over_a
  (a b c : ℝ)
  (h₁ : (a ≠ 0))
  (h₂ : ∀ x, -1 < x ∧ x < 3 → ax^2 + bx + c < 0) :
  b - 2 * c + (1 / a) = 4 :=
sorry

end min_value_of_b_minus_2c_plus_1_over_a_l45_45295


namespace sum_of_g_of_nine_values_l45_45940

def f (x : ℝ) : ℝ := x^2 - 9 * x + 20
def g (y : ℝ) : ℝ := 3 * y - 4

theorem sum_of_g_of_nine_values : (g 9) = 19 := by
  sorry

end sum_of_g_of_nine_values_l45_45940


namespace total_detergent_used_l45_45971

-- Define the parameters of the problem
def total_pounds_of_clothes : ℝ := 9
def pounds_of_cotton : ℝ := 4
def pounds_of_woolen : ℝ := 5
def detergent_per_pound_cotton : ℝ := 2
def detergent_per_pound_woolen : ℝ := 1.5

-- Main theorem statement
theorem total_detergent_used : 
  (pounds_of_cotton * detergent_per_pound_cotton) + (pounds_of_woolen * detergent_per_pound_woolen) = 15.5 :=
by
  sorry

end total_detergent_used_l45_45971


namespace taxi_fare_l45_45486

-- Define the necessary values and functions based on the problem conditions
def starting_price : ℝ := 6
def additional_charge_per_km : ℝ := 1.5
def distance (P : ℝ) : Prop := P > 6

-- Lean proposition to state the problem
theorem taxi_fare (P : ℝ) (hP : distance P) : 
  (starting_price + additional_charge_per_km * (P - 6)) = 1.5 * P - 3 := 
by 
  sorry

end taxi_fare_l45_45486


namespace xiaoming_money_l45_45376

open Real

noncomputable def verify_money_left (M P_L : ℝ) : Prop := M = 12 * P_L

noncomputable def verify_money_right (M P_R : ℝ) : Prop := M = 14 * P_R

noncomputable def price_relationship (P_L P_R : ℝ) : Prop := P_R = P_L - 1

theorem xiaoming_money (M P_L P_R : ℝ) 
  (h1 : verify_money_left M P_L) 
  (h2 : verify_money_right M P_R) 
  (h3 : price_relationship P_L P_R) : 
  M = 84 := 
  by
  sorry

end xiaoming_money_l45_45376


namespace value_of_q_l45_45520

theorem value_of_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 1) (h4 : p * q = 12) : q = 6 + 2 * Real.sqrt 6 :=
by
  sorry

end value_of_q_l45_45520


namespace games_played_so_far_l45_45233

-- Definitions based on conditions
def total_matches := 20
def points_for_victory := 3
def points_for_draw := 1
def points_for_defeat := 0
def points_scored_so_far := 14
def points_needed := 40
def required_wins := 6

-- The proof problem
theorem games_played_so_far : 
  ∃ W D L : ℕ, 3 * W + D + 0 * L = points_scored_so_far ∧ 
  ∃ W' D' L' : ℕ, 3 * W' + D' + 0 * L' + 3 * required_wins = points_needed ∧ 
  (total_matches - required_wins = 14) :=
by 
  sorry

end games_played_so_far_l45_45233


namespace dave_added_apps_l45_45675

-- Define the conditions as a set of given facts
def initial_apps : Nat := 10
def deleted_apps : Nat := 17
def remaining_apps : Nat := 4

-- The statement to prove
theorem dave_added_apps : ∃ x : Nat, initial_apps + x - deleted_apps = remaining_apps ∧ x = 11 :=
by
  use 11
  sorry

end dave_added_apps_l45_45675


namespace matthew_and_zac_strawberries_l45_45284

theorem matthew_and_zac_strawberries (total_strawberries jonathan_and_matthew_strawberries zac_strawberries : ℕ) (h1 : total_strawberries = 550) (h2 : jonathan_and_matthew_strawberries = 350) (h3 : zac_strawberries = 200) : (total_strawberries - (jonathan_and_matthew_strawberries - zac_strawberries) = 400) :=
by { sorry }

end matthew_and_zac_strawberries_l45_45284


namespace find_y_l45_45186

theorem find_y (x y : ℝ) (h1 : x = 100) (h2 : x^3 * y - 3 * x^2 * y + 3 * x * y = 3000000) : 
  y = 3000000 / (100^3 - 3 * 100^2 + 3 * 100 * 1) :=
by sorry

end find_y_l45_45186


namespace scrap_rate_independence_l45_45366

theorem scrap_rate_independence (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  (1 - (1 - a) * (1 - b)) = 1 - (1 - a) * (1 - b) :=
by
  sorry

end scrap_rate_independence_l45_45366


namespace quadratic_no_real_roots_l45_45399

theorem quadratic_no_real_roots (c : ℝ) : 
  (∀ x : ℝ, ¬(x^2 + x - c = 0)) ↔ c < -1/4 := 
sorry

end quadratic_no_real_roots_l45_45399


namespace avg_reading_time_l45_45778

theorem avg_reading_time (emery_book_time serena_book_time emery_article_time serena_article_time : ℕ)
    (h1 : emery_book_time = 20)
    (h2 : emery_article_time = 2)
    (h3 : emery_book_time * 5 = serena_book_time)
    (h4 : emery_article_time * 3 = serena_article_time) :
    (emery_book_time + emery_article_time + serena_book_time + serena_article_time) / 2 = 64 := by
  sorry

end avg_reading_time_l45_45778


namespace num_divisors_of_m_cubed_l45_45307

theorem num_divisors_of_m_cubed (m : ℕ) (h : ∃ p : ℕ, Nat.Prime p ∧ m = p ^ 4) :
    Nat.totient (m ^ 3) = 13 := 
sorry

end num_divisors_of_m_cubed_l45_45307


namespace smallest_divisible_by_1_to_10_l45_45662

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l45_45662


namespace solve_for_A_in_terms_of_B_l45_45561

noncomputable def f (A B x : ℝ) := A * x - 2 * B^2
noncomputable def g (B x : ℝ) := B * x

theorem solve_for_A_in_terms_of_B (A B : ℝ) (hB : B ≠ 0) (h : f A B (g B 1) = 0) : A = 2 * B := by
  sorry

end solve_for_A_in_terms_of_B_l45_45561


namespace ratio_of_ages_l45_45659

noncomputable def ratio_4th_to_3rd (age1 age2 age3 age4 age5 : ℕ) : ℚ :=
  age4 / age3

theorem ratio_of_ages
  (age1 age2 age3 age4 age5 : ℕ)
  (h1 : (age1 + age5) / 2 = 18)
  (h2 : age1 = 10)
  (h3 : age2 = age1 - 2)
  (h4 : age3 = age2 + 4)
  (h5 : age4 = age3 / 2)
  (h6 : age5 = age4 + 20) :
  ratio_4th_to_3rd age1 age2 age3 age4 age5 = 1 / 2 :=
by
  sorry

end ratio_of_ages_l45_45659


namespace quadratic_equation_formulation_l45_45036

theorem quadratic_equation_formulation (a b c : ℝ) (x₁ x₂ : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : a * x₁^2 + b * x₁ + c = 0)
  (h₃ : a * x₂^2 + b * x₂ + c = 0)
  (h₄ : x₁ + x₂ = -b / a)
  (h₅ : x₁ * x₂ = c / a) :
  ∃ (y : ℝ), a^2 * y^2 + a * (b - c) * y - b * c = 0 :=
by
  sorry

end quadratic_equation_formulation_l45_45036


namespace purely_imaginary_subtraction_l45_45206

-- Definition of the complex number z.
def z : ℂ := Complex.mk 2 (-1)

-- Statement to prove
theorem purely_imaginary_subtraction (h: z = Complex.mk 2 (-1)) : ∃ (b : ℝ), z - 2 = Complex.im b :=
by {
    sorry
}

end purely_imaginary_subtraction_l45_45206


namespace smallest_pos_int_mult_4410_sq_l45_45717

noncomputable def smallest_y : ℤ := 10

theorem smallest_pos_int_mult_4410_sq (y : ℕ) (hy : y > 0) :
  (∃ z : ℕ, 4410 * y = z^2) ↔ y = smallest_y :=
sorry

end smallest_pos_int_mult_4410_sq_l45_45717


namespace original_price_of_trouser_l45_45993

theorem original_price_of_trouser (P : ℝ) (sale_price : ℝ) (percent_decrease : ℝ) 
  (h1 : sale_price = 40) (h2 : percent_decrease = 0.60) 
  (h3 : sale_price = P * (1 - percent_decrease)) : P = 100 :=
by
  sorry

end original_price_of_trouser_l45_45993


namespace true_proposition_l45_45598

-- Definitions based on the conditions
def p (x : ℝ) := x * (x - 1) ≠ 0 → x ≠ 0 ∧ x ≠ 1
def q (a b c : ℝ) := a > b → c > 0 → a * c > b * c

-- The theorem based on the question and the conditions
theorem true_proposition (x a b c : ℝ) (hp : p x) (hq_false : ¬ q a b c) : p x ∨ q a b c :=
by
  sorry

end true_proposition_l45_45598


namespace bennett_brothers_count_l45_45337

theorem bennett_brothers_count :
  ∃ B, B = 2 * 4 - 2 ∧ B = 6 :=
by
  sorry

end bennett_brothers_count_l45_45337


namespace tickets_won_whack_a_mole_l45_45363

variable (t : ℕ)

def tickets_from_skee_ball : ℕ := 9
def cost_per_candy : ℕ := 6
def number_of_candies : ℕ := 7
def total_tickets_needed : ℕ := cost_per_candy * number_of_candies

theorem tickets_won_whack_a_mole : t + tickets_from_skee_ball = total_tickets_needed → t = 33 :=
by
  intro h
  have h1 : total_tickets_needed = 42 := by sorry
  have h2 : tickets_from_skee_ball = 9 := by rfl
  rw [h2, h1] at h
  sorry

end tickets_won_whack_a_mole_l45_45363


namespace master_wang_resting_on_sunday_again_l45_45299

theorem master_wang_resting_on_sunday_again (n : ℕ) 
  (works_days := 8) 
  (rest_days := 2) 
  (week_days := 7) 
  (cycle_days := works_days + rest_days) 
  (initial_rest_saturday_sunday : Prop) : 
  (initial_rest_saturday_sunday → ∃ n : ℕ, (week_days * n) % cycle_days = rest_days) → 
  (∃ n : ℕ, n = 7) :=
by
  sorry

end master_wang_resting_on_sunday_again_l45_45299


namespace seeds_per_packet_l45_45866

theorem seeds_per_packet (total_seedlings packets : ℕ) (h1 : total_seedlings = 420) (h2 : packets = 60) : total_seedlings / packets = 7 :=
by 
  sorry

end seeds_per_packet_l45_45866


namespace esther_commute_distance_l45_45736

theorem esther_commute_distance (D : ℕ) :
  (D / 45 + D / 30 = 1) → D = 18 :=
by
  sorry

end esther_commute_distance_l45_45736


namespace barry_sotter_length_increase_l45_45550

theorem barry_sotter_length_increase (n : ℕ) : (n + 3) / 3 = 50 → n = 147 :=
by
  intro h
  sorry

end barry_sotter_length_increase_l45_45550


namespace value_range_of_2_sin_x_minus_1_l45_45192

theorem value_range_of_2_sin_x_minus_1 :
  (∀ x : ℝ, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1) →
  (∀ y : ℝ, y = 2 * Real.sin y - 1 → -3 ≤ y ∧ y ≤ 1) :=
by
  sorry

end value_range_of_2_sin_x_minus_1_l45_45192


namespace find_c_d_l45_45839

noncomputable def g (c d x : ℝ) : ℝ := c * x^3 + 5 * x^2 + d * x + 7

theorem find_c_d : ∃ (c d : ℝ), 
  (g c d 2 = 11) ∧ (g c d (-3) = 134) ∧ c = -35 / 13 ∧ d = 16 / 13 :=
  by
  sorry

end find_c_d_l45_45839


namespace stones_required_correct_l45_45387

/- 
Given:
- The hall measures 36 meters long and 15 meters broad.
- Each stone measures 6 decimeters by 5 decimeters.

We need to prove that the number of stones required to pave the hall is 1800.
-/
noncomputable def stones_required 
  (hall_length_m : ℕ) 
  (hall_breadth_m : ℕ) 
  (stone_length_dm : ℕ) 
  (stone_breadth_dm : ℕ) : ℕ :=
  (hall_length_m * 10) * (hall_breadth_m * 10) / (stone_length_dm * stone_breadth_dm)

theorem stones_required_correct : 
  stones_required 36 15 6 5 = 1800 :=
by 
  -- Placeholder for proof
  sorry

end stones_required_correct_l45_45387


namespace speed_comparison_l45_45666

theorem speed_comparison (v v2 : ℝ) (h1 : v2 > 0) (h2 : v = 5 * v2) : v = 5 * v2 :=
by
  exact h2 

end speed_comparison_l45_45666


namespace bacteria_growth_rate_l45_45613

theorem bacteria_growth_rate (r : ℝ) 
  (h1 : ∀ n : ℕ, n = 22 → ∃ c : ℝ, c * r^n = c) 
  (h2 : ∀ n : ℕ, n = 21 → ∃ c : ℝ, 2 * c * r^n = c) : 
  r = 2 := 
by
  sorry

end bacteria_growth_rate_l45_45613


namespace shelley_weight_l45_45523

theorem shelley_weight (p s r : ℕ) (h1 : p + s = 151) (h2 : s + r = 132) (h3 : p + r = 115) : s = 84 := 
  sorry

end shelley_weight_l45_45523


namespace prove_a2_l45_45188

def arithmetic_seq (a d : ℕ → ℝ) : Prop :=
  ∀ n m, a n + d (n - m) = a m

theorem prove_a2 (a : ℕ → ℝ) (d : ℕ → ℝ) :
  (∀ n, a n = a 0 + (n - 1) * 2) → 
  (a 1 + 4) / a 1 = (a 1 + 6) / (a 1 + 4) →
  (d 1 = 2) →
  a 2 = -6 :=
by
  intros h_seq h_geo h_common_diff
  sorry

end prove_a2_l45_45188


namespace max_valid_words_for_AU_language_l45_45821

noncomputable def maxValidWords : ℕ :=
  2^14 - 128

theorem max_valid_words_for_AU_language 
  (letters : Finset (String)) (validLengths : Set ℕ) (noConcatenation : Prop) :
  letters = {"a", "u"} ∧ validLengths = {n | 1 ≤ n ∧ n ≤ 13} ∧ noConcatenation →
  maxValidWords = 16256 :=
by
  sorry

end max_valid_words_for_AU_language_l45_45821


namespace max_papers_l45_45043

theorem max_papers (p c r : ℕ) (h1 : p ≥ 2) (h2 : c ≥ 1) (h3 : 3 * p + 5 * c + 9 * r = 72) : r ≤ 6 :=
sorry

end max_papers_l45_45043


namespace sprinkler_system_days_l45_45764

theorem sprinkler_system_days 
  (morning_water : ℕ) (evening_water : ℕ) (total_water : ℕ) 
  (h_morning : morning_water = 4) 
  (h_evening : evening_water = 6) 
  (h_total : total_water = 50) :
  total_water / (morning_water + evening_water) = 5 := 
by 
  sorry

end sprinkler_system_days_l45_45764


namespace positive_difference_proof_l45_45180

noncomputable def solve_system : Prop :=
  ∃ (x y : ℝ), 
  (x + y = 40) ∧ 
  (3 * y - 4 * x = 10) ∧ 
  abs (y - x) = 8.58

theorem positive_difference_proof : solve_system := 
  sorry

end positive_difference_proof_l45_45180


namespace remainder_of_num_five_element_subsets_with_two_consecutive_l45_45877

-- Define the set and the problem
noncomputable def num_five_element_subsets_with_two_consecutive (n : ℕ) : ℕ := 
  Nat.choose 14 5 - Nat.choose 10 5

-- Main Lean statement: prove the final condition
theorem remainder_of_num_five_element_subsets_with_two_consecutive :
  (num_five_element_subsets_with_two_consecutive 14) % 1000 = 750 :=
by
  -- Proof goes here
  sorry

end remainder_of_num_five_element_subsets_with_two_consecutive_l45_45877


namespace range_of_a_l45_45926

theorem range_of_a (M : Set ℝ) (a : ℝ) :
  (M = {x | x^2 - 4 * x + 4 * a < 0}) →
  ¬(2 ∈ M) →
  (1 ≤ a) :=
by
  -- Given assumptions
  intros hM h2_notin_M
  -- Convert h2_notin_M to an inequality and prove the desired result
  sorry

end range_of_a_l45_45926


namespace interior_angle_regular_octagon_exterior_angle_regular_octagon_l45_45709

-- Definitions
def sumInteriorAngles (n : ℕ) : ℕ := 180 * (n - 2)
def oneInteriorAngle (n : ℕ) (sumInterior : ℕ) : ℕ := sumInterior / n
def sumExteriorAngles : ℕ := 360
def oneExteriorAngle (n : ℕ) (sumExterior : ℕ) : ℕ := sumExterior / n

-- Theorem statements
theorem interior_angle_regular_octagon : oneInteriorAngle 8 (sumInteriorAngles 8) = 135 := by sorry

theorem exterior_angle_regular_octagon : oneExteriorAngle 8 sumExteriorAngles = 45 := by sorry

end interior_angle_regular_octagon_exterior_angle_regular_octagon_l45_45709


namespace similar_triangle_legs_l45_45645

theorem similar_triangle_legs {y : ℝ} 
  (h₁ : 12 / y = 9 / 6) : y = 8 :=
sorry

end similar_triangle_legs_l45_45645


namespace kenneth_initial_money_l45_45467

-- Define the costs of the items
def cost_baguette := 2
def cost_water := 1

-- Define the quantities bought
def baguettes_bought := 2
def water_bought := 2

-- Define the amount left after buying the items
def money_left := 44

-- Calculate the total cost
def total_cost := (baguettes_bought * cost_baguette) + (water_bought * cost_water)

-- Define the initial money Kenneth had
def initial_money := total_cost + money_left

-- Prove the initial money is $50
theorem kenneth_initial_money : initial_money = 50 := 
by 
  -- The proof part is omitted because it is not required.
  sorry

end kenneth_initial_money_l45_45467


namespace fraction_videocassette_recorders_l45_45334

variable (H : ℝ) (F : ℝ)

-- Conditions
variable (cable_TV_frac : ℝ := 1 / 5)
variable (both_frac : ℝ := 1 / 20)
variable (neither_frac : ℝ := 0.75)

-- Main theorem statement
theorem fraction_videocassette_recorders (H_pos : 0 < H) 
  (cable_tv : cable_TV_frac * H > 0)
  (both : both_frac * H > 0) 
  (neither : neither_frac * H > 0) :
  F = 1 / 10 :=
by
  sorry

end fraction_videocassette_recorders_l45_45334


namespace range_of_a_l45_45253

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a + Real.cos (2 * x) < 5 - 4 * Real.sin x + Real.sqrt (5 * a - 4)) :
  a ∈ Set.Icc (4 / 5) 8 :=
sorry

end range_of_a_l45_45253


namespace car_rental_cost_l45_45275

def day1_cost (base_rate : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) : ℝ :=
  base_rate + miles_driven * cost_per_mile

def day2_cost (base_rate : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) : ℝ :=
  base_rate + miles_driven * cost_per_mile

def day3_cost (base_rate : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) : ℝ :=
  base_rate + miles_driven * cost_per_mile

def total_cost (day1 : ℝ) (day2 : ℝ) (day3 : ℝ) : ℝ :=
  day1 + day2 + day3

theorem car_rental_cost :
  let day1_base_rate := 150
  let day2_base_rate := 100
  let day3_base_rate := 75
  let day1_miles_driven := 620
  let day2_miles_driven := 744
  let day3_miles_driven := 510
  let day1_cost_per_mile := 0.50
  let day2_cost_per_mile := 0.40
  let day3_cost_per_mile := 0.30
  day1_cost day1_base_rate day1_miles_driven day1_cost_per_mile +
  day2_cost day2_base_rate day2_miles_driven day2_cost_per_mile +
  day3_cost day3_base_rate day3_miles_driven day3_cost_per_mile = 1085.60 :=
by
  let day1 := day1_cost 150 620 0.50
  let day2 := day2_cost 100 744 0.40
  let day3 := day3_cost 75 510 0.30
  let total := total_cost day1 day2 day3
  show total = 1085.60
  sorry

end car_rental_cost_l45_45275


namespace icing_cubes_count_l45_45568

theorem icing_cubes_count :
  let n := 5
  let total_cubes := n * n * n
  let side_faces := 4
  let cubes_per_edge_per_face := (n - 2) * (n - 1)
  let shared_edges := 4
  let icing_cubes := (side_faces * cubes_per_edge_per_face) / 2
  icing_cubes = 32 := sorry

end icing_cubes_count_l45_45568


namespace smaller_number_is_22_l45_45723

noncomputable def smaller_number (x y : ℕ) : ℕ := 
x

theorem smaller_number_is_22 (x y : ℕ) (h1 : x + y = 56) (h2 : y = x + 12) : x = 22 :=
by
  sorry

end smaller_number_is_22_l45_45723


namespace initial_price_of_iphone_l45_45512

variable (P : ℝ)

def initial_price_conditions : Prop :=
  (P > 0) ∧ (0.72 * P = 720)

theorem initial_price_of_iphone (h : initial_price_conditions P) : P = 1000 :=
by
  sorry

end initial_price_of_iphone_l45_45512


namespace union_A_B_l45_45582

open Set

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | x < 1}

theorem union_A_B : A ∪ B = {x | x < 2} := 
by sorry

end union_A_B_l45_45582


namespace sum_of_digits_least_N_l45_45099

-- Define the function P(N)
def P (N : ℕ) : ℚ := (Nat.ceil (3 * N / 5 + 1) : ℕ) / (N + 1)

-- Define the predicate that checks if P(N) is less than 321/400
def P_lt_321_over_400 (N : ℕ) : Prop := P N < (321 / 400 : ℚ)

-- Define a function that sums the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- The main statement: we claim the least multiple of 5 satisfying the condition
-- That the sum of its digits is 12
theorem sum_of_digits_least_N :
  ∃ N : ℕ, 
    (N % 5 = 0) ∧ 
    P_lt_321_over_400 N ∧ 
    (∀ N' : ℕ, (N' % 5 = 0) → P_lt_321_over_400 N' → N' ≥ N) ∧ 
    sum_of_digits N = 12 := 
sorry

end sum_of_digits_least_N_l45_45099


namespace prove_range_of_a_l45_45182

noncomputable def f (x a : ℝ) : ℝ := (x + a - 1) * Real.exp x

def problem_condition1 (x a : ℝ) : Prop := 
  f x a ≥ (x^2 / 2 + a * x)

def problem_condition2 (x : ℝ) : Prop := 
  x ∈ Set.Ici 0 -- equivalent to [0, +∞)

theorem prove_range_of_a (a : ℝ) :
  (∀ x : ℝ, problem_condition2 x → problem_condition1 x a) → a ∈ Set.Ici 1 :=
sorry

end prove_range_of_a_l45_45182


namespace max_value_of_x_times_one_minus_2x_l45_45133

theorem max_value_of_x_times_one_minus_2x : 
  ∀ x : ℝ, 0 < x ∧ x < 1 / 2 → x * (1 - 2 * x) ≤ 1 / 8 :=
by
  intro x 
  intro hx
  sorry

end max_value_of_x_times_one_minus_2x_l45_45133


namespace percentage_of_Luccas_balls_are_basketballs_l45_45157

-- Defining the variables and their conditions 
variables (P : ℝ) (Lucca_Balls : ℕ := 100) (Lucien_Balls : ℕ := 200)
variable (Total_Basketballs : ℕ := 50)

-- Condition that Lucien has 20% basketballs
def Lucien_Basketballs := (20 / 100) * Lucien_Balls

-- We need to prove that percentage of Lucca's balls that are basketballs is 10%
theorem percentage_of_Luccas_balls_are_basketballs :
  (P / 100) * Lucca_Balls + Lucien_Basketballs = Total_Basketballs → P = 10 :=
by
  sorry

end percentage_of_Luccas_balls_are_basketballs_l45_45157


namespace no_solution_for_vectors_l45_45541

theorem no_solution_for_vectors {t s k : ℝ} :
  (∃ t s : ℝ, (1 + 6 * t = -1 + 3 * s) ∧ (3 + 1 * t = 4 + k * s)) ↔ k ≠ 0.5 :=
sorry

end no_solution_for_vectors_l45_45541


namespace value_of_x_l45_45869

def x : ℚ :=
  (320 / 2) / 3

theorem value_of_x : x = 160 / 3 := 
by
  unfold x
  sorry

end value_of_x_l45_45869


namespace constants_solution_l45_45482

theorem constants_solution (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 → 
    (5 * x^2 / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2)) ↔ 
    (A = 20 ∧ B = -15 ∧ C = -10) :=
by
  sorry

end constants_solution_l45_45482


namespace ratio_second_to_first_l45_45853

theorem ratio_second_to_first (F S T : ℕ) 
  (hT : T = 2 * F)
  (havg : (F + S + T) / 3 = 77)
  (hmin : F = 33) :
  S / F = 4 :=
by
  sorry

end ratio_second_to_first_l45_45853


namespace no_x_intersections_geometric_sequence_l45_45265

theorem no_x_intersections_geometric_sequence (a b c : ℝ) 
  (h1 : b^2 = a * c)
  (h2 : a * c > 0) : 
  (∃ x : ℝ, a * x^2 + b * x + c = 0) = false :=
by
  sorry

end no_x_intersections_geometric_sequence_l45_45265


namespace compute_value_l45_45535

def diamond_op (x y : ℕ) : ℕ := 3 * x + 5 * y
def heart_op (z x : ℕ) : ℕ := 4 * z + 2 * x

theorem compute_value : heart_op (diamond_op 4 3) 8 = 124 := by
  sorry

end compute_value_l45_45535


namespace calculate_f3_minus_f4_l45_45369

-- Defining the function f and the given conditions
variables (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = -f x)
variable (periodic_f : ∀ x, f (x + 2) = -f x)
variable (f1 : f 1 = 1)

-- Proving the required equality
theorem calculate_f3_minus_f4 : f 3 - f 4 = -1 :=
by
  sorry

end calculate_f3_minus_f4_l45_45369


namespace no_odd_prime_pn_plus_1_eq_2m_no_odd_prime_pn_minus_1_eq_2m_l45_45876

open Nat

theorem no_odd_prime_pn_plus_1_eq_2m (n p m : ℕ)
  (hn : n > 1) (hp : p.Prime) (hp_odd : Odd p) (hm : m > 0) :
  p^n + 1 ≠ 2^m := by
  sorry

theorem no_odd_prime_pn_minus_1_eq_2m (n p m : ℕ)
  (hn : n > 2) (hp : p.Prime) (hp_odd : Odd p) (hm : m > 0) :
  p^n - 1 ≠ 2^m := by
  sorry

end no_odd_prime_pn_plus_1_eq_2m_no_odd_prime_pn_minus_1_eq_2m_l45_45876


namespace eq_solutions_a2_eq_b_times_b_plus_7_l45_45459

theorem eq_solutions_a2_eq_b_times_b_plus_7 (a b : ℤ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h : a^2 = b * (b + 7)) :
  (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
sorry

end eq_solutions_a2_eq_b_times_b_plus_7_l45_45459


namespace man_swim_downstream_distance_l45_45103

-- Define the given conditions
def t_d : ℝ := 6
def t_u : ℝ := 6
def d_u : ℝ := 18
def V_m : ℝ := 4.5

-- The distance the man swam downstream
def distance_downstream : ℝ := 36

-- Prove that given the conditions, the man swam 36 km downstream
theorem man_swim_downstream_distance (V_c : ℝ) :
  (d_u / (V_m - V_c) = t_u) →
  (distance_downstream / (V_m + V_c) = t_d) →
  distance_downstream = 36 :=
by
  sorry

end man_swim_downstream_distance_l45_45103


namespace cos_B_in_triangle_l45_45050

theorem cos_B_in_triangle
  (A B C a b c : ℝ)
  (h1 : Real.sin A = 2 * Real.sin C)
  (h2 : b^2 = a * c)
  (h3 : 0 < b)
  (h4 : 0 < c)
  (h5 : a = 2 * c)
  : Real.cos B = 3 / 4 := 
sorry

end cos_B_in_triangle_l45_45050


namespace find_m_eq_4_l45_45315

theorem find_m_eq_4 (m : ℝ) (h₁ : ∃ (A B C : ℝ × ℝ), A = (m, -m+3) ∧ B = (2, m-1) ∧ C = (-1, 4)) (h₂ : (4 - (-m+3)) / (-1-m) = 3 * ((m-1) - 4) / (2 - (-1))) : m = 4 :=
sorry

end find_m_eq_4_l45_45315


namespace solve_for_x_l45_45841

theorem solve_for_x (x : ℝ) : 
  x - 3 * x + 5 * x = 150 → x = 50 :=
by
  intro h
  -- sorry to skip the proof
  sorry

end solve_for_x_l45_45841


namespace fraction_sum_to_decimal_l45_45618

theorem fraction_sum_to_decimal : 
  (3 / 10 : Rat) + (5 / 100) - (1 / 1000) = 349 / 1000 := 
by
  sorry

end fraction_sum_to_decimal_l45_45618


namespace janine_test_score_l45_45158

theorem janine_test_score :
  let num_mc := 10
  let p_mc := 0.80
  let num_sa := 30
  let p_sa := 0.70
  let total_questions := 40
  let correct_mc := p_mc * num_mc
  let correct_sa := p_sa * num_sa
  let total_correct := correct_mc + correct_sa
  (total_correct / total_questions) * 100 = 72.5 := 
by
  sorry

end janine_test_score_l45_45158


namespace a37_b37_sum_l45_45724

-- Declare the sequences as functions from natural numbers to real numbers
variables {a b : ℕ → ℝ}

-- State the hypotheses based on the conditions
variables (h1 : ∀ n, a (n + 1) = a n + a 2 - a 1)
variables (h2 : ∀ n, b (n + 1) = b n + b 2 - b 1)
variables (h3 : a 1 = 25)
variables (h4 : b 1 = 75)
variables (h5 : a 2 + b 2 = 100)

-- State the theorem to be proved
theorem a37_b37_sum : a 37 + b 37 = 100 := 
by 
  sorry

end a37_b37_sum_l45_45724


namespace find_marks_in_biology_l45_45664

-- Definitions based on conditions in a)
def marks_english : ℕ := 76
def marks_math : ℕ := 60
def marks_physics : ℕ := 72
def marks_chemistry : ℕ := 65
def num_subjects : ℕ := 5
def average_marks : ℕ := 71

-- The theorem that needs to be proved
theorem find_marks_in_biology : 
  let total_marks := marks_english + marks_math + marks_physics + marks_chemistry 
  let total_marks_all := average_marks * num_subjects
  let marks_biology := total_marks_all - total_marks
  marks_biology = 82 := 
by
  sorry

end find_marks_in_biology_l45_45664


namespace abs_sum_zero_eq_neg_one_l45_45037

theorem abs_sum_zero_eq_neg_one (a b : ℝ) (h : |3 + a| + |b - 2| = 0) : a + b = -1 :=
sorry

end abs_sum_zero_eq_neg_one_l45_45037


namespace fraction_addition_l45_45955

theorem fraction_addition (d : ℝ) : (6 + 5 * d) / 9 + 3 = (33 + 5 * d) / 9 := 
by 
  sorry

end fraction_addition_l45_45955


namespace symmetric_point_origin_l45_45075

-- Define the notion of symmetry with respect to the origin
def symmetric_with_origin (p : ℤ × ℤ) : ℤ × ℤ :=
  (-p.1, -p.2)

-- Define the given point
def given_point : ℤ × ℤ :=
  (-2, 5)

-- State the theorem to be proven
theorem symmetric_point_origin : 
  symmetric_with_origin given_point = (2, -5) :=
by 
  -- The proof will go here, use sorry for now
  sorry

end symmetric_point_origin_l45_45075


namespace and_or_distrib_left_or_and_distrib_right_l45_45611

theorem and_or_distrib_left (A B C : Prop) : A ∧ (B ∨ C) ↔ (A ∧ B) ∨ (A ∧ C) :=
sorry

theorem or_and_distrib_right (A B C : Prop) : A ∨ (B ∧ C) ↔ (A ∨ B) ∧ (A ∨ C) :=
sorry

end and_or_distrib_left_or_and_distrib_right_l45_45611


namespace greatest_product_from_sum_2004_l45_45343

theorem greatest_product_from_sum_2004 : ∃ (x y : ℤ), x + y = 2004 ∧ x * y = 1004004 :=
by
  sorry

end greatest_product_from_sum_2004_l45_45343


namespace range_of_a_l45_45817

noncomputable def setA : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def setB (a : ℝ) : Set ℝ := {x | x^2 + a * x + a = 0}

theorem range_of_a :
  ∀ a : ℝ, (setA ∪ setB a) = setA ↔ 0 ≤ a ∧ a < 4 :=
by sorry

end range_of_a_l45_45817


namespace no_analytic_roots_l45_45978

theorem no_analytic_roots : ¬∃ x : ℝ, (x - 2) * (x + 5)^3 * (5 - x) = 8 := 
sorry

end no_analytic_roots_l45_45978


namespace family_ages_sum_today_l45_45883

theorem family_ages_sum_today (A B C D E : ℕ) (h1 : A + B + C + D = 114) (h2 : E = D - 14) :
    (A + 5) + (B + 5) + (C + 5) + (E + 5) = 120 :=
by
  sorry

end family_ages_sum_today_l45_45883


namespace focus_of_parabola_l45_45596

theorem focus_of_parabola :
  (∃ f : ℝ, ∀ y : ℝ, (x = -1 / 4 * y^2) = (x = (y^2 / 4 + f)) -> f = -1) :=
by
  sorry

end focus_of_parabola_l45_45596


namespace max_value_g_l45_45701

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * (100 - x)) + Real.sqrt (x * (4 - x))

theorem max_value_g : ∃ (x₁ N : ℝ), (0 ≤ x₁ ∧ x₁ ≤ 4) ∧ (N = 16) ∧ (x₁ = 2) ∧ (∀ x, 0 ≤ x ∧ x ≤ 4 → g x ≤ N) :=
by
  sorry

end max_value_g_l45_45701


namespace min_n_for_circuit_l45_45167

theorem min_n_for_circuit
  (n : ℕ) 
  (p_success_component : ℝ)
  (p_work_circuit : ℝ) 
  (h1 : p_success_component = 0.5)
  (h2 : p_work_circuit = 1 - p_success_component ^ n) 
  (h3 : p_work_circuit ≥ 0.95) :
  n ≥ 5 := 
sorry

end min_n_for_circuit_l45_45167


namespace increasing_f_for_x_ge_1_f_gt_1_for_x_gt_0_l45_45155

noncomputable def f (x : ℝ) := x ^ 2 * Real.exp x - Real.log x

theorem increasing_f_for_x_ge_1 : ∀ (x : ℝ), x ≥ 1 → ∀ y > x, f y > f x :=
by
  sorry

theorem f_gt_1_for_x_gt_0 : ∀ (x : ℝ), x > 0 → f x > 1 :=
by
  sorry

end increasing_f_for_x_ge_1_f_gt_1_for_x_gt_0_l45_45155


namespace probability_heads_9_of_12_is_correct_l45_45979

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l45_45979


namespace max_value_2cosx_3sinx_l45_45734

open Real 

theorem max_value_2cosx_3sinx : ∀ x : ℝ, 2 * cos x + 3 * sin x ≤ sqrt 13 :=
by sorry

end max_value_2cosx_3sinx_l45_45734


namespace expression_value_l45_45228

theorem expression_value :
  let x := (3 + 1 : ℚ)⁻¹ * 2
  let y := x⁻¹ * 2
  let z := y⁻¹ * 2
  z = (1 / 2 : ℚ) :=
by
  sorry

end expression_value_l45_45228


namespace Kyle_is_25_l45_45457

variable (Tyson_age : ℕ := 20)
variable (Frederick_age : ℕ := 2 * Tyson_age)
variable (Julian_age : ℕ := Frederick_age - 20)
variable (Kyle_age : ℕ := Julian_age + 5)

theorem Kyle_is_25 : Kyle_age = 25 := by
  sorry

end Kyle_is_25_l45_45457


namespace sufficient_but_not_necessary_l45_45190

theorem sufficient_but_not_necessary (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : (a^2 + b^2 < 1) → (ab + 1 > a + b) ∧ ¬(ab + 1 > a + b ↔ a^2 + b^2 < 1) := 
sorry

end sufficient_but_not_necessary_l45_45190


namespace ticket_cost_correct_l45_45547

noncomputable def calculate_ticket_cost : ℝ :=
  let x : ℝ := 5  -- price of an adult ticket
  let child_price := x / 2  -- price of a child ticket
  let senior_price := 0.75 * x  -- price of a senior ticket
  10 * x + 8 * child_price + 5 * senior_price

theorem ticket_cost_correct :
  let x : ℝ := 5  -- price of an adult ticket
  let child_price := x / 2  -- price of a child ticket
  let senior_price := 0.75 * x  -- price of a senior ticket
  (4 * x + 3 * child_price + 2 * senior_price = 35) →
  (10 * x + 8 * child_price + 5 * senior_price = 88.75) :=
by
  intros
  sorry

end ticket_cost_correct_l45_45547


namespace find_C_and_D_l45_45593

theorem find_C_and_D :
  (∀ x, x^2 - 3 * x - 10 ≠ 0 → (4 * x - 3) / (x^2 - 3 * x - 10) = (17 / 7) / (x - 5) + (11 / 7) / (x + 2)) :=
by
  sorry

end find_C_and_D_l45_45593


namespace zero_function_is_uniq_l45_45398

theorem zero_function_is_uniq (f : ℝ → ℝ) :
  (∀ (x : ℝ) (hx : x ≠ 0) (y : ℝ), f (x^2 + y) ≥ (1/x + 1) * f y) → 
  (∀ x, f x = 0) :=
by
  sorry

end zero_function_is_uniq_l45_45398


namespace sarah_shampoo_and_conditioner_usage_l45_45171

-- Condition Definitions
def shampoo_daily_oz := 1
def conditioner_daily_oz := shampoo_daily_oz / 2
def total_daily_usage := shampoo_daily_oz + conditioner_daily_oz

def days_in_two_weeks := 14

-- Assertion: Total volume used in two weeks.
theorem sarah_shampoo_and_conditioner_usage :
  (days_in_two_weeks * total_daily_usage) = 21 := by
  sorry

end sarah_shampoo_and_conditioner_usage_l45_45171


namespace postage_cost_correct_l45_45594

-- Conditions
def base_rate : ℕ := 35
def additional_rate_per_ounce : ℕ := 25
def weight_in_ounces : ℚ := 5.25
def first_ounce : ℚ := 1
def fraction_weight : ℚ := weight_in_ounces - first_ounce
def num_additional_charges : ℕ := Nat.ceil (fraction_weight)

-- Question and correct answer
def total_postage_cost : ℕ := base_rate + (num_additional_charges * additional_rate_per_ounce)
def answer_in_cents : ℕ := 160

theorem postage_cost_correct : total_postage_cost = answer_in_cents := by sorry

end postage_cost_correct_l45_45594


namespace x_squared_eq_y_squared_iff_x_eq_y_or_neg_y_x_squared_eq_y_squared_necessary_but_not_sufficient_l45_45164

theorem x_squared_eq_y_squared_iff_x_eq_y_or_neg_y (x y : ℝ) : 
  (x^2 = y^2) ↔ (x = y ∨ x = -y) := by
  sorry

theorem x_squared_eq_y_squared_necessary_but_not_sufficient (x y : ℝ) :
  (x^2 = y^2 → x = y) ↔ false := by
  sorry

end x_squared_eq_y_squared_iff_x_eq_y_or_neg_y_x_squared_eq_y_squared_necessary_but_not_sufficient_l45_45164


namespace remainder_when_divided_by_29_l45_45524

theorem remainder_when_divided_by_29 (k N : ℤ) (h : N = 761 * k + 173) : N % 29 = 28 :=
by
  sorry

end remainder_when_divided_by_29_l45_45524


namespace negation_of_universal_proposition_l45_45604

open Real

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → (x+1) * exp x > 1) ↔ ∃ x : ℝ, x > 0 ∧ (x+1) * exp x ≤ 1 :=
by sorry

end negation_of_universal_proposition_l45_45604


namespace star_3_4_equals_8_l45_45349

def star (a b : ℕ) : ℕ := 4 * a + 5 * b - 2 * a * b

theorem star_3_4_equals_8 : star 3 4 = 8 := by
  sorry

end star_3_4_equals_8_l45_45349


namespace evaluate_expression_l45_45352

theorem evaluate_expression (c d : ℝ) (h_c : c = 3) (h_d : d = 2) : 
  (c^2 + d + 1)^2 - (c^2 - d - 1)^2 = 80 := by 
  sorry

end evaluate_expression_l45_45352


namespace completing_square_l45_45600

theorem completing_square (x : ℝ) (h : x^2 - 6 * x - 7 = 0) : (x - 3)^2 = 16 := 
sorry

end completing_square_l45_45600


namespace hash_hash_hash_45_l45_45020

def hash (N : ℝ) : ℝ := 0.4 * N + 3

theorem hash_hash_hash_45 : hash (hash (hash 45)) = 7.56 :=
by
  sorry

end hash_hash_hash_45_l45_45020


namespace Linda_purchase_cost_l45_45317

def price_peanuts : ℝ := sorry
def price_berries : ℝ := sorry
def price_coconut : ℝ := sorry
def price_dates : ℝ := sorry

theorem Linda_purchase_cost:
  ∃ (p b c d : ℝ), 
    (p + b + c + d = 30) ∧ 
    (3 * p = d) ∧
    ((p + b) / 2 = c) ∧
    (b + c = 65 / 9) :=
sorry

end Linda_purchase_cost_l45_45317


namespace burger_share_l45_45436

theorem burger_share (burger_length : ℝ) (brother_share : ℝ) (first_friend_share : ℝ) (second_friend_share : ℝ) (valentina_share : ℝ) :
  burger_length = 12 →
  brother_share = burger_length / 3 →
  first_friend_share = (burger_length - brother_share) / 4 →
  second_friend_share = (burger_length - brother_share - first_friend_share) / 2 →
  valentina_share = burger_length - (brother_share + first_friend_share + second_friend_share) →
  brother_share = 4 ∧ first_friend_share = 2 ∧ second_friend_share = 3 ∧ valentina_share = 3 :=
by
  intros
  sorry

end burger_share_l45_45436


namespace total_flowers_eaten_l45_45183

theorem total_flowers_eaten (bugs : ℕ) (flowers_per_bug : ℕ) (h_bugs : bugs = 3) (h_flowers_per_bug : flowers_per_bug = 2) :
  (bugs * flowers_per_bug) = 6 :=
by
  sorry

end total_flowers_eaten_l45_45183


namespace average_salary_of_laborers_l45_45963

-- Define the main statement as a theorem
theorem average_salary_of_laborers 
  (total_workers : ℕ)
  (total_salary_all : ℕ)
  (supervisors : ℕ)
  (supervisor_salary : ℕ)
  (laborers : ℕ)
  (expected_laborer_salary : ℝ) :
  total_workers = 48 → 
  total_salary_all = 60000 →
  supervisors = 6 →
  supervisor_salary = 2450 →
  laborers = 42 →
  expected_laborer_salary = 1078.57 :=
sorry

end average_salary_of_laborers_l45_45963


namespace last_number_aryana_counts_l45_45795

theorem last_number_aryana_counts (a d : ℤ) (h_start : a = 72) (h_diff : d = -11) :
  ∃ n : ℕ, (a + n * d > 0) ∧ (a + (n + 1) * d ≤ 0) ∧ a + n * d = 6 := by
  sorry

end last_number_aryana_counts_l45_45795


namespace problem_I_problem_II_l45_45762

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := |x - 2| + |2 * x + a|

-- Problem (I): Inequality solution when a = 1
theorem problem_I (x : ℝ) : f x 1 ≥ 5 ↔ x ∈ (Set.Iic (-4 / 3) ∪ Set.Ici 2) :=
sorry

-- Problem (II): Range of a given the conditions
theorem problem_II (x₀ : ℝ) (a : ℝ) (h : f x₀ a + |x₀ - 2| < 3) : -7 < a ∧ a < -1 :=
sorry

end problem_I_problem_II_l45_45762


namespace matrix_det_eq_l45_45079

open Matrix

def matrix3x3 (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![x + 1, x, x],
    ![x, x + 2, x],
    ![x, x, x + 3]
  ]

theorem matrix_det_eq (x : ℝ) : det (matrix3x3 x) = 2 * x^2 + 11 * x + 6 :=
  sorry

end matrix_det_eq_l45_45079


namespace alcohol_percentage_first_solution_l45_45787

theorem alcohol_percentage_first_solution
  (x : ℝ)
  (h1 : 0 ≤ x ∧ x ≤ 1) -- since percentage in decimal form is between 0 and 1
  (h2 : 75 * x + 0.12 * 125 = 0.15 * 200) :
  x = 0.20 :=
by
  sorry

end alcohol_percentage_first_solution_l45_45787


namespace sum_of_squares_of_roots_of_quadratic_l45_45375

noncomputable def sum_of_squares_of_roots (p q : ℝ) (a b : ℝ) : Prop :=
  a^2 + b^2 = 4 * p^2 - 6 * q

theorem sum_of_squares_of_roots_of_quadratic
  (p q a b : ℝ)
  (h1 : a + b = 2 * p / 3)
  (h2 : a * b = q / 3)
  (h3 : a * a + b * b = 4 * p^2 - 6 * q) :
  sum_of_squares_of_roots p q a b :=
by
  sorry

end sum_of_squares_of_roots_of_quadratic_l45_45375


namespace vinegar_final_percentage_l45_45816

def vinegar_percentage (volume1 volume2 : ℕ) (percent1 percent2 : ℚ) : ℚ :=
  let vinegar1 := volume1 * percent1 / 100
  let vinegar2 := volume2 * percent2 / 100
  (vinegar1 + vinegar2) / (volume1 + volume2) * 100

theorem vinegar_final_percentage:
  vinegar_percentage 128 128 8 13 = 10.5 :=
  sorry

end vinegar_final_percentage_l45_45816


namespace find_integer_n_l45_45478

theorem find_integer_n (n : ℤ) (hn : -150 < n ∧ n < 150) : (n = 80 ∨ n = -100) ↔ (Real.tan (n * Real.pi / 180) = Real.tan (1340 * Real.pi / 180)) :=
by 
  sorry

end find_integer_n_l45_45478


namespace a8_value_l45_45408

def sequence_sum (n : ℕ) : ℕ := 2^n - 1

def nth_term (S : ℕ → ℕ) (n : ℕ) : ℕ :=
  S n - S (n - 1)

theorem a8_value : nth_term sequence_sum 8 = 128 :=
by
  -- Proof goes here
  sorry

end a8_value_l45_45408


namespace min_sum_of_diagonals_l45_45034

theorem min_sum_of_diagonals (x y : ℝ) (α : ℝ) (hx : 0 < x) (hy : 0 < y) (hα : 0 < α ∧ α < π) (h_area : x * y * Real.sin α = 2) : x + y ≥ 2 * Real.sqrt 2 :=
sorry

end min_sum_of_diagonals_l45_45034


namespace comparison_l45_45194

open Real

noncomputable def a := 5 * log (2 ^ exp 1)
noncomputable def b := 2 * log (5 ^ exp 1)
noncomputable def c := 10

theorem comparison : c > a ∧ a > b :=
by
  have a_def : a = 5 * log (2 ^ exp 1) := rfl
  have b_def : b = 2 * log (5 ^ exp 1) := rfl
  have c_def : c = 10 := rfl
  sorry -- Proof goes here

end comparison_l45_45194


namespace MinTransportCost_l45_45191

noncomputable def TruckTransportOptimization :=
  ∃ (x y : ℕ), x + y = 6 ∧ 45 * x + 30 * y ≥ 240 ∧ 400 * x + 300 * y ≤ 2300 ∧ (∃ (min_cost : ℕ), min_cost = 2200 ∧ x = 4 ∧ y = 2)
  
theorem MinTransportCost : TruckTransportOptimization :=
sorry

end MinTransportCost_l45_45191


namespace Q_investment_l45_45121

-- Given conditions
variables (P Q : Nat) (P_investment : P = 30000) (profit_ratio : 2 / 3 = P / Q)

-- Target statement
theorem Q_investment : Q = 45000 :=
by 
  sorry

end Q_investment_l45_45121


namespace cos_evaluation_l45_45894

open Real

noncomputable def a (n : ℕ) : ℝ := sorry  -- since it's an arithmetic sequence

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m k : ℕ, a n + a k = 2 * a ((n + k) / 2)

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 6 + a 9 = 3 * a 6 ∧ a 6 = π / 4

theorem cos_evaluation :
  is_arithmetic_sequence a →
  satisfies_condition a →
  cos (a 2 + a 10 + π / 4) = - (sqrt 2 / 2) :=
by
  intros
  sorry

end cos_evaluation_l45_45894


namespace find_g_of_3_l45_45744

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_of_3 (h : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = 3) : g 3 = 0 :=
by sorry

end find_g_of_3_l45_45744


namespace molecular_weight_of_one_mole_l45_45954

theorem molecular_weight_of_one_mole (molecular_weight_3_moles : ℕ) (h : molecular_weight_3_moles = 222) : (molecular_weight_3_moles / 3) = 74 := 
by
  sorry

end molecular_weight_of_one_mole_l45_45954


namespace percentage_of_25_of_fifty_percent_of_500_l45_45463

-- Define the constants involved
def fifty_percent_of_500 := 0.50 * 500  -- 50% of 500

-- Prove the equivalence
theorem percentage_of_25_of_fifty_percent_of_500 : (25 / fifty_percent_of_500) * 100 = 10 := by
  -- Place proof steps here
  sorry

end percentage_of_25_of_fifty_percent_of_500_l45_45463


namespace boiling_point_fahrenheit_l45_45438

-- Define the conditions as hypotheses
def boils_celsius : ℝ := 100
def melts_celsius : ℝ := 0
def melts_fahrenheit : ℝ := 32
def pot_temp_celsius : ℝ := 55
def pot_temp_fahrenheit : ℝ := 131

-- Theorem to prove the boiling point in Fahrenheit
theorem boiling_point_fahrenheit : ∀ (boils_celsius : ℝ) (melts_celsius : ℝ) (melts_fahrenheit : ℝ) 
                                    (pot_temp_celsius : ℝ) (pot_temp_fahrenheit : ℝ),
  boils_celsius = 100 →
  melts_celsius = 0 →
  melts_fahrenheit = 32 →
  pot_temp_celsius = 55 →
  pot_temp_fahrenheit = 131 →
  ∃ boils_fahrenheit : ℝ, boils_fahrenheit = 212 :=
by
  intros
  existsi 212
  sorry

end boiling_point_fahrenheit_l45_45438


namespace difference_between_roots_l45_45969

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := -7
noncomputable def c : ℝ := 11

noncomputable def quadratic_roots (a b c : ℝ) : (ℝ × ℝ) :=
  let discriminant := b ^ 2 - 4 * a * c
  ((-b + Real.sqrt discriminant) / (2 * a), (-b - Real.sqrt discriminant) / (2 * a))

-- Extract the roots from the equation
noncomputable def r1_r2 := quadratic_roots a b c

noncomputable def r1 : ℝ := r1_r2.1
noncomputable def r2 : ℝ := r1_r2.2

-- Theorem statement: the difference between the roots is sqrt(5)
theorem difference_between_roots :
  |r1 - r2| = Real.sqrt 5 :=
  sorry

end difference_between_roots_l45_45969


namespace isosceles_triangle_angles_l45_45080

theorem isosceles_triangle_angles (A B C : ℝ) (h_iso: (A = B) ∨ (B = C) ∨ (A = C)) (angle_A : A = 50) :
  (B = 50) ∨ (B = 65) ∨ (B = 80) :=
by
  sorry

end isosceles_triangle_angles_l45_45080


namespace langsley_commute_time_l45_45311

theorem langsley_commute_time (first_bus: ℕ) (first_wait: ℕ) (second_bus: ℕ) (second_wait: ℕ) (third_bus: ℕ) (total_time: ℕ)
  (h1: first_bus = 40)
  (h2: first_wait = 10)
  (h3: second_bus = 50)
  (h4: second_wait = 15)
  (h5: third_bus = 95)
  (h6: total_time = first_bus + first_wait + second_bus + second_wait + third_bus) :
  total_time = 210 := 
by 
  sorry

end langsley_commute_time_l45_45311


namespace less_sum_mult_l45_45255

theorem less_sum_mult {a b : ℝ} (h1 : a < 1) (h2 : b > 1) : a * b < a + b :=
sorry

end less_sum_mult_l45_45255


namespace solve_system_of_equations_l45_45899

theorem solve_system_of_equations (x y : ℝ) (h1 : y^2 + x * y = 15) (h2 : x^2 + x * y = 10) :
  (x = 2 ∧ y = 3) ∨ (x = -2 ∧ y = -3) :=
sorry

end solve_system_of_equations_l45_45899


namespace remainder_when_divided_by_11_l45_45728

theorem remainder_when_divided_by_11 :
  (7 * 10^20 + 2^20) % 11 = 8 := by
sorry

end remainder_when_divided_by_11_l45_45728


namespace arccos_one_eq_zero_l45_45647

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l45_45647


namespace find_counterfeit_10_l45_45742

theorem find_counterfeit_10 (coins : Fin 10 → ℕ) (h_counterfeit : ∃ k, ∀ i, i ≠ k → coins i < coins k) : 
  ∃ w : ℕ → ℕ → Prop, (∀ g1 g2, g1 ≠ g2 → w g1 g2 ∨ w g2 g1) → 
  ∃ k, ∀ i, i ≠ k → coins i < coins k :=
sorry

end find_counterfeit_10_l45_45742


namespace copper_sheet_area_l45_45726

noncomputable def area_of_copper_sheet (l w h : ℝ) (thickness_mm : ℝ) : ℝ :=
  let volume := l * w * h
  let thickness_cm := thickness_mm / 10
  (volume / thickness_cm) / 10000

theorem copper_sheet_area :
  ∀ (l w h thickness_mm : ℝ), 
  l = 80 → w = 20 → h = 5 → thickness_mm = 1 → 
  area_of_copper_sheet l w h thickness_mm = 8 := 
by
  intros l w h thickness_mm hl hw hh hthickness_mm
  rw [hl, hw, hh, hthickness_mm]
  simp [area_of_copper_sheet]
  sorry

end copper_sheet_area_l45_45726


namespace triangle_angle_sum_l45_45879

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by
  sorry

end triangle_angle_sum_l45_45879


namespace smallest_nat_divisible_by_48_squared_l45_45693

theorem smallest_nat_divisible_by_48_squared :
  ∃ n : ℕ, (n % (48^2) = 0) ∧ 
           (∀ (d : ℕ), d ∈ (Nat.digits n 10) → d = 0 ∨ d = 1) ∧ 
           (n = 11111111100000000) := sorry

end smallest_nat_divisible_by_48_squared_l45_45693


namespace inequality_to_prove_l45_45615

variable {r r1 r2 r3 m : ℝ}
variable {A B C : ℝ}

-- Conditions
-- r is the radius of an inscribed circle in a triangle
-- r1, r2, r3 are radii of circles each touching two sides of the triangle and the inscribed circle
-- m is a real number such that m >= 1/2

axiom r_radii_condition : r > 0
axiom r1_radii_condition : r1 > 0
axiom r2_radii_condition : r2 > 0
axiom r3_radii_condition : r3 > 0
axiom m_condition : m ≥ 1/2

-- Inequality to prove
theorem inequality_to_prove : 
  (r1 * r2) ^ m + (r2 * r3) ^ m + (r3 * r1) ^ m ≥ 3 * (r / 3) ^ (2 * m) := 
sorry

end inequality_to_prove_l45_45615


namespace problem_equation_has_solution_l45_45663

noncomputable def x (real_number : ℚ) : ℚ := 210 / 23

theorem problem_equation_has_solution (x_value : ℚ) : 
  (3 / 7) + (7 / x_value) = (10 / x_value) + (1 / 10) → 
  x_value = 210 / 23 :=
by
  intro h
  sorry

end problem_equation_has_solution_l45_45663


namespace range_of_a_l45_45477

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, x^2 + a * x + 3 ≥ a) ↔ -7 ≤ a ∧ a ≤ 2 :=
by sorry

end range_of_a_l45_45477


namespace crafts_club_necklaces_l45_45272

theorem crafts_club_necklaces (members : ℕ) (total_beads : ℕ) (beads_per_necklace : ℕ)
  (h1 : members = 9) (h2 : total_beads = 900) (h3 : beads_per_necklace = 50) :
  (total_beads / beads_per_necklace) / members = 2 :=
by
  sorry

end crafts_club_necklaces_l45_45272


namespace arithmetic_sequence_sum_equality_l45_45475

variables {a_n : ℕ → ℝ} -- the arithmetic sequence
variables (S_n : ℕ → ℝ) -- the sum of the first n terms of the sequence

-- Define the conditions as hypotheses
def condition_1 (S_n : ℕ → ℝ) : Prop := S_n 3 = 3
def condition_2 (S_n : ℕ → ℝ) : Prop := S_n 6 = 15

-- Theorem statement
theorem arithmetic_sequence_sum_equality
  (h1 : condition_1 S_n)
  (h2 : condition_2 S_n)
  (a_n_formula : ∀ n, a_n n = a_n 0 + n * (a_n 1 - a_n 0))
  (S_n_formula : ∀ n, S_n n = n * (a_n 0 + (n - 1) * (a_n 1 - a_n 0) / 2)) :
  a_n 10 + a_n 11 + a_n 12 = 30 := sorry

end arithmetic_sequence_sum_equality_l45_45475


namespace find_ordered_pair_l45_45846

theorem find_ordered_pair (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0)
  (h₃ : (x : ℝ) → x^2 + 2 * a * x + b = 0 → x = a ∨ x = b) :
  (a, b) = (1, -3) :=
sorry

end find_ordered_pair_l45_45846


namespace find_base_l45_45712

theorem find_base (b : ℕ) : (b^3 ≤ 64 ∧ 64 < b^4) ↔ b = 4 := 
by
  sorry

end find_base_l45_45712


namespace solve_equation_l45_45829

noncomputable def equation (x : ℝ) : Prop := x * (x - 2) + x - 2 = 0

theorem solve_equation : ∀ x, equation x ↔ (x = 2 ∨ x = -1) :=
by sorry

end solve_equation_l45_45829


namespace product_of_xyz_is_correct_l45_45890

theorem product_of_xyz_is_correct : 
  ∃ x y z : ℤ, 
    (-3 * x + 4 * y - z = 28) ∧ 
    (3 * x - 2 * y + z = 8) ∧ 
    (x + y - z = 2) ∧ 
    (x * y * z = 2898) :=
by
  sorry

end product_of_xyz_is_correct_l45_45890


namespace rectangle_dimensions_exist_l45_45864

theorem rectangle_dimensions_exist :
  ∃ (a b c d : ℕ), (a * b + c * d = 81) ∧ (2 * (a + b) = 2 * 2 * (c + d) ∨ 2 * (c + d) = 2 * 2 * (a + b)) :=
by sorry

end rectangle_dimensions_exist_l45_45864


namespace derivative_at_pi_div_3_l45_45798

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem derivative_at_pi_div_3 : 
  deriv f (Real.pi / 3) = - (Real.sqrt 3 * Real.pi / 6) :=
by
  sorry

end derivative_at_pi_div_3_l45_45798


namespace intersection_of_A_and_B_l45_45471

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5}

-- State the theorem about the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {2, 4} :=
  sorry

end intersection_of_A_and_B_l45_45471


namespace option_d_is_right_triangle_l45_45374

noncomputable def right_triangle (a b c : ℝ) : Prop :=
  a^2 + c^2 = b^2

theorem option_d_is_right_triangle (a b c : ℝ) (h : a^2 = b^2 - c^2) :
  right_triangle a b c :=
by
  sorry

end option_d_is_right_triangle_l45_45374


namespace units_digit_divisible_by_18_l45_45875

theorem units_digit_divisible_by_18 : ∃ n : ℕ, (3150 ≤ 315 * n) ∧ (315 * n < 3160) ∧ (n % 2 = 0) ∧ (315 * n % 18 = 0) ∧ (n = 0) :=
by
  use 0
  sorry

end units_digit_divisible_by_18_l45_45875


namespace arithmetic_seq_sum_l45_45320

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h2 : a 2 + a 5 + a 8 = 15) : a 3 + a 7 = 10 :=
sorry

end arithmetic_seq_sum_l45_45320


namespace net_effect_on_sale_l45_45150

theorem net_effect_on_sale (P Q : ℝ) :
  let new_price := 0.65 * P
  let new_quantity := 1.8 * Q
  let original_revenue := P * Q
  let new_revenue := new_price * new_quantity
  new_revenue - original_revenue = 0.17 * original_revenue :=
by
  sorry

end net_effect_on_sale_l45_45150


namespace geometric_sequence_sum_l45_45260

variable (a : ℕ → ℝ)
variable (q : ℝ)

axiom h1 : a 1 + a 2 = 20
axiom h2 : a 3 + a 4 = 40
axiom h3 : q^2 = 2

theorem geometric_sequence_sum : a 5 + a 6 = 80 :=
by
  sorry

end geometric_sequence_sum_l45_45260


namespace inequality_holds_l45_45294

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : x * y + y * z + z * x = 1) :
  (1 / (x + y)) + (1 / (y + z)) + (1 / (z + x)) ≥ 5 / 2 :=
by
  sorry

end inequality_holds_l45_45294


namespace find_natural_number_l45_45386

theorem find_natural_number (n : ℕ) (h : ∃ k : ℕ, n^2 - 19 * n + 95 = k^2) : n = 5 ∨ n = 14 := by
  sorry

end find_natural_number_l45_45386


namespace smallest_b_for_factors_l45_45159

theorem smallest_b_for_factors (b : ℕ) (h : ∃ r s : ℤ, (x : ℤ) → (x + r) * (x + s) = x^2 + ↑b * x + 2016 ∧ r * s = 2016) :
  b = 90 :=
by
  sorry

end smallest_b_for_factors_l45_45159


namespace maxwell_distance_traveled_l45_45805

theorem maxwell_distance_traveled
  (distance_between_homes : ℕ)
  (maxwell_speed : ℕ)
  (brad_speed : ℕ)
  (meeting_time : ℕ)
  (h1 : distance_between_homes = 72)
  (h2 : maxwell_speed = 6)
  (h3 : brad_speed = 12)
  (h4 : meeting_time = distance_between_homes / (maxwell_speed + brad_speed)) :
  maxwell_speed * meeting_time = 24 :=
by
  sorry

end maxwell_distance_traveled_l45_45805


namespace fencing_cost_is_correct_l45_45124

def length : ℕ := 60
def cost_per_meter : ℕ := 27 -- using the closest integer value to 26.50
def breadth (l : ℕ) : ℕ := l - 20
def perimeter (l b : ℕ) : ℕ := 2 * l + 2 * b
def total_cost (P : ℕ) (c : ℕ) : ℕ := P * c

theorem fencing_cost_is_correct :
  total_cost (perimeter length (breadth length)) cost_per_meter = 5300 :=
  sorry

end fencing_cost_is_correct_l45_45124


namespace min_project_time_l45_45273

theorem min_project_time (A B C : ℝ) (D : ℝ := 12) :
  (1 / B + 1 / C) = 1 / 2 →
  (1 / A + 1 / C) = 1 / 3 →
  (1 / A + 1 / B) = 1 / 4 →
  (1 / D) = 1 / 12 →
  ∃ x : ℝ, x = 8 / 5 ∧ 1 / x = 1 / A + 1 / B + 1 / C + 1 / (12:ℝ) :=
by
  intros h1 h2 h3 h4
  -- Combination of given hypotheses to prove the goal
  sorry

end min_project_time_l45_45273


namespace labor_cost_calculation_l45_45905

def num_men : Nat := 5
def num_women : Nat := 8
def num_boys : Nat := 10

def base_wage_man : Nat := 100
def base_wage_woman : Nat := 80
def base_wage_boy : Nat := 50

def efficiency_man_woman_ratio : Nat := 2
def efficiency_man_boy_ratio : Nat := 3

def overtime_rate_multiplier : Nat := 3 / 2 -- 1.5 as a ratio
def holiday_rate_multiplier : Nat := 2

def num_men_working_overtime : Nat := 3
def hours_worked_overtime : Nat := 10
def regular_workday_hours : Nat := 8

def is_holiday : Bool := true

theorem labor_cost_calculation : 
  (num_men * base_wage_man * holiday_rate_multiplier
    + num_women * base_wage_woman * holiday_rate_multiplier
    + num_boys * base_wage_boy * holiday_rate_multiplier
    + num_men_working_overtime * (hours_worked_overtime - regular_workday_hours) * (base_wage_man * overtime_rate_multiplier)) 
  = 4180 :=
by
  sorry

end labor_cost_calculation_l45_45905


namespace product_modulo_l45_45556

theorem product_modulo (n : ℕ) (h : 93 * 68 * 105 ≡ n [MOD 20]) (h_range : 0 ≤ n ∧ n < 20) : n = 0 := 
by
  sorry

end product_modulo_l45_45556


namespace remaining_distance_l45_45793

theorem remaining_distance (S u : ℝ) (h1 : S / (2 * u) + 24 = S) (h2 : S * u / 2 + 15 = S) : ∃ x : ℝ, x = 8 :=
by
  -- Proof steps would go here
  sorry

end remaining_distance_l45_45793


namespace remaining_gift_card_value_correct_l45_45924

def initial_best_buy := 5
def initial_target := 3
def initial_walmart := 7
def initial_amazon := 2

def value_best_buy := 500
def value_target := 250
def value_walmart := 100
def value_amazon := 1000

def sent_best_buy := 1
def sent_walmart := 2
def sent_amazon := 1

def remaining_dollars : Nat :=
  (initial_best_buy - sent_best_buy) * value_best_buy +
  initial_target * value_target +
  (initial_walmart - sent_walmart) * value_walmart +
  (initial_amazon - sent_amazon) * value_amazon

theorem remaining_gift_card_value_correct : remaining_dollars = 4250 :=
  sorry

end remaining_gift_card_value_correct_l45_45924


namespace math_problem_l45_45364

theorem math_problem :
  2537 + 240 * 3 / 60 - 347 = 2202 :=
by
  sorry

end math_problem_l45_45364


namespace problem_l45_45699

variable (x : ℝ) (Q : ℝ)

theorem problem (h : 2 * (5 * x + 3 * Real.pi) = Q) : 4 * (10 * x + 6 * Real.pi + 2) = 4 * Q + 8 :=
by
  sorry

end problem_l45_45699


namespace find_b_l45_45238

noncomputable def g (b x : ℝ) : ℝ := b * x^2 - Real.cos (Real.pi * x)

theorem find_b (b : ℝ) (hb : 0 < b) (h : g b (g b 1) = -Real.cos Real.pi) : b = 1 :=
by
  sorry

end find_b_l45_45238


namespace crocodiles_count_l45_45404

-- Definitions of constants
def alligators : Nat := 23
def vipers : Nat := 5
def total_dangerous_animals : Nat := 50

-- Theorem statement
theorem crocodiles_count :
  total_dangerous_animals - alligators - vipers = 22 :=
by
  sorry

end crocodiles_count_l45_45404


namespace quadratic_interval_inequality_l45_45908

theorem quadratic_interval_inequality (a b c : ℝ) :
  (∀ x : ℝ, -1 / 2 < x ∧ x < 2 → a * x^2 + b * x + c > 0) →
  a < 0 ∧ c > 0 :=
sorry

end quadratic_interval_inequality_l45_45908


namespace Steve_pencils_left_l45_45428

-- Define the initial number of boxes and pencils per box
def boxes := 2
def pencils_per_box := 12
def initial_pencils := boxes * pencils_per_box

-- Define the number of pencils given to Lauren and the additional pencils given to Matt
def pencils_to_Lauren := 6
def diff_Lauren_Matt := 3
def pencils_to_Matt := pencils_to_Lauren + diff_Lauren_Matt

-- Calculate the total pencils given away
def pencils_given_away := pencils_to_Lauren + pencils_to_Matt

-- Number of pencils left with Steve
def pencils_left := initial_pencils - pencils_given_away

-- The statement to prove
theorem Steve_pencils_left : pencils_left = 9 := by
  sorry

end Steve_pencils_left_l45_45428


namespace factor_expression_l45_45269

variable (y : ℝ)

theorem factor_expression : 64 - 16 * y ^ 3 = 16 * (2 - y) * (4 + 2 * y + y ^ 2) := by
  sorry

end factor_expression_l45_45269


namespace abs_difference_extrema_l45_45177

theorem abs_difference_extrema (x : ℝ) (h : 2 ≤ x ∧ x < 3) :
  max (|x-2| + |x-3| - |x-1|) = 0 ∧ min (|x-2| + |x-3| - |x-1|) = -1 :=
by
  sorry

end abs_difference_extrema_l45_45177


namespace find_c_l45_45094

theorem find_c (c : ℕ) (h : 111111222222 = c * (c + 1)) : c = 333333 :=
by
  -- proof goes here
  sorry

end find_c_l45_45094


namespace football_team_practice_hours_l45_45694

-- Definitions for each day's practice adjusted for weather events
def monday_hours : ℕ := 4
def tuesday_hours : ℕ := 5 - 1
def wednesday_hours : ℕ := 0
def thursday_hours : ℕ := 5
def friday_hours : ℕ := 3 + 2
def saturday_hours : ℕ := 4
def sunday_hours : ℕ := 0

-- Total practice hours calculation
def total_practice_hours : ℕ := 
  monday_hours + tuesday_hours + wednesday_hours + 
  thursday_hours + friday_hours + saturday_hours + 
  sunday_hours

-- Statement to prove
theorem football_team_practice_hours : total_practice_hours = 22 := by
  sorry

end football_team_practice_hours_l45_45694


namespace Scruffy_weight_l45_45860

variable {Muffy Puffy Scruffy : ℝ}

def Puffy_weight_condition (Muffy Puffy : ℝ) : Prop := Puffy = Muffy + 5
def Scruffy_weight_condition (Muffy Scruffy : ℝ) : Prop := Scruffy = Muffy + 3
def Combined_weight_condition (Muffy Puffy : ℝ) : Prop := Muffy + Puffy = 23

theorem Scruffy_weight (h1 : Puffy_weight_condition Muffy Puffy) (h2 : Scruffy_weight_condition Muffy Scruffy) (h3 : Combined_weight_condition Muffy Puffy) : Scruffy = 12 := by
  sorry

end Scruffy_weight_l45_45860


namespace least_positive_integer_l45_45758

theorem least_positive_integer (n : ℕ) : 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 6 = 3) → n = 57 := by
sorry

end least_positive_integer_l45_45758


namespace quadratic_roots_sum_squares_l45_45429

theorem quadratic_roots_sum_squares {a b : ℝ} 
  (h₁ : a + b = -1) 
  (h₂ : a * b = -5) : 
  2 * a^2 + a + b^2 = 16 :=
by sorry

end quadratic_roots_sum_squares_l45_45429


namespace area_enclosed_by_curves_l45_45748

noncomputable def areaBetweenCurves : ℝ :=
  ∫ x in (0 : ℝ)..(4 : ℝ), (x - (x^2 - 3*x))

theorem area_enclosed_by_curves :
  (∫ x in (0 : ℝ)..(4 : ℝ), (x - (x^2 - 3*x))) = (32 / 3 : ℝ) :=
by
  sorry

end area_enclosed_by_curves_l45_45748


namespace second_athlete_triple_jump_l45_45887

theorem second_athlete_triple_jump
  (long_jump1 triple_jump1 high_jump1 : ℕ) 
  (long_jump2 high_jump2 : ℕ)
  (average_winner : ℕ) 
  (H1 : long_jump1 = 26) (H2 : triple_jump1 = 30) (H3 : high_jump1 = 7)
  (H4 : long_jump2 = 24) (H5 : high_jump2 = 8) (H6 : average_winner = 22)
  : ∃ x : ℕ, (24 + x + 8) / 3 = 22 ∧ x = 34 := 
by
  sorry

end second_athlete_triple_jump_l45_45887


namespace polygon_max_sides_l45_45120

theorem polygon_max_sides (n : ℕ) (h : (n - 2) * 180 < 2005) : n ≤ 13 :=
by {
  sorry
}

end polygon_max_sides_l45_45120


namespace javier_savings_l45_45222

theorem javier_savings (regular_price : ℕ) (discount1 : ℕ) (discount2 : ℕ) : 
  (regular_price = 50) 
  ∧ (discount1 = 40)
  ∧ (discount2 = 50) 
  → (30 = (100 * (regular_price * 3 - (regular_price + (regular_price * (100 - discount1) / 100) + regular_price / 2)) / (regular_price * 3))) :=
by
  intros h
  sorry

end javier_savings_l45_45222


namespace initial_pennies_in_each_compartment_l45_45832

theorem initial_pennies_in_each_compartment (x : ℕ) (h : 12 * (x + 6) = 96) : x = 2 :=
by sorry

end initial_pennies_in_each_compartment_l45_45832


namespace minute_hand_length_l45_45353

theorem minute_hand_length 
  (arc_length : ℝ) (r : ℝ) (h : arc_length = 20 * (2 * Real.pi / 60) * r) :
  r = 1/2 :=
  sorry

end minute_hand_length_l45_45353


namespace percentage_workday_in_meetings_l45_45797

theorem percentage_workday_in_meetings :
  let workday_minutes := 10 * 60
  let first_meeting := 30
  let second_meeting := 2 * first_meeting
  let third_meeting := first_meeting + second_meeting
  let total_meeting_minutes := first_meeting + second_meeting + third_meeting
  (total_meeting_minutes * 100) / workday_minutes = 30 :=
by
  sorry

end percentage_workday_in_meetings_l45_45797


namespace nat_power_of_p_iff_only_prime_factor_l45_45198

theorem nat_power_of_p_iff_only_prime_factor (p n : ℕ) (hp : Nat.Prime p) :
  (∃ k : ℕ, n = p^k) ↔ (∀ q : ℕ, Nat.Prime q → q ∣ n → q = p) := 
sorry

end nat_power_of_p_iff_only_prime_factor_l45_45198


namespace age_difference_l45_45345

theorem age_difference (A B C : ℕ) (h1 : A + B > B + C) (h2 : C = A - 17) : (A + B) - (B + C) = 17 :=
by
  sorry

end age_difference_l45_45345


namespace sq_in_scientific_notation_l45_45055

theorem sq_in_scientific_notation (a : Real) (h : a = 25000) (h_scientific : a = 2.5 * 10^4) : a^2 = 6.25 * 10^8 :=
sorry

end sq_in_scientific_notation_l45_45055


namespace inequality_solution_l45_45695

theorem inequality_solution (x : ℝ) (h : ∀ (a b : ℝ) (ha : 0 < a) (hb : 0 < b), x^2 + x < a / b + b / a) : x ∈ Set.Ioo (-2 : ℝ) 1 := 
sorry

end inequality_solution_l45_45695


namespace jeff_makes_donuts_for_days_l45_45673

variable (d : ℕ) (boxes donuts_per_box : ℕ) (donuts_per_day eaten_per_day : ℕ) (chris_eaten total_donuts : ℕ)

theorem jeff_makes_donuts_for_days :
  (donuts_per_day = 10) →
  (eaten_per_day = 1) →
  (chris_eaten = 8) →
  (boxes = 10) →
  (donuts_per_box = 10) →
  (total_donuts = boxes * donuts_per_box) →
  (9 * d - chris_eaten = total_donuts) →
  d = 12 :=
  by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end jeff_makes_donuts_for_days_l45_45673


namespace triangle_side_length_l45_45514

theorem triangle_side_length (a b c : ℝ)
  (h1 : 1/2 * a * c * (Real.sin (60 * Real.pi / 180)) = Real.sqrt 3)
  (h2 : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
by
  sorry

end triangle_side_length_l45_45514


namespace pipes_fill_tank_l45_45706

theorem pipes_fill_tank (T : ℝ) (h1 : T > 0)
  (h2 : (1/4 : ℝ) + 1/T - 1/20 = 1/2.5) : T = 5 := by
  sorry

end pipes_fill_tank_l45_45706


namespace most_cost_effective_way_cost_is_860_l45_45935

-- Definitions based on the problem conditions
def adult_cost := 150
def child_cost := 60
def group_cost_per_person := 100
def group_min_size := 5

-- Number of adults and children
def num_adults := 4
def num_children := 7

-- Calculate the total cost for the most cost-effective way
noncomputable def most_cost_effective_way_cost :=
  let group_tickets_count := 5  -- 4 adults + 1 child
  let remaining_children := num_children - 1
  group_tickets_count * group_cost_per_person + remaining_children * child_cost

-- Theorem to state the cost for the most cost-effective way
theorem most_cost_effective_way_cost_is_860 : most_cost_effective_way_cost = 860 := by
  sorry

end most_cost_effective_way_cost_is_860_l45_45935


namespace cheddar_cheese_slices_l45_45173

-- Define the conditions
def cheddar_slices (C : ℕ) := ∃ (packages : ℕ), packages * C = 84
def swiss_slices := 28
def randy_bought_same_slices (C : ℕ) := swiss_slices = 28 ∧ 84 = 84

-- Lean theorem statement to prove the number of slices per package of cheddar cheese equals 28.
theorem cheddar_cheese_slices {C : ℕ} (h1 : cheddar_slices C) (h2 : randy_bought_same_slices C) : C = 28 :=
sorry

end cheddar_cheese_slices_l45_45173


namespace line_intersects_x_axis_at_point_l45_45579

-- Define the conditions and required proof
theorem line_intersects_x_axis_at_point :
  (∃ x : ℝ, ∃ y : ℝ, 5 * y - 7 * x = 35 ∧ y = 0 ∧ (x, y) = (-5, 0)) :=
by
  -- The proof is omitted according to the steps
  sorry

end line_intersects_x_axis_at_point_l45_45579


namespace banana_price_l45_45385

theorem banana_price (b : ℝ) : 
    (∃ x : ℕ, 0.70 * x + b * (9 - x) = 5.60 ∧ x + (9 - x) = 9) → b = 0.60 :=
by
  intro h
  obtain ⟨x, hx1, hx2⟩ := h
  -- equations to work with:
  -- 0.70 * x + b * (9 - x) = 5.60
  -- x + (9 - x) = 9
  sorry

end banana_price_l45_45385


namespace system_inconsistent_l45_45595

theorem system_inconsistent :
  ¬(∃ (x1 x2 x3 x4 : ℝ), 
    (5 * x1 + 12 * x2 + 19 * x3 + 25 * x4 = 25) ∧
    (10 * x1 + 22 * x2 + 16 * x3 + 39 * x4 = 25) ∧
    (5 * x1 + 12 * x2 + 9 * x3 + 25 * x4 = 30) ∧
    (20 * x1 + 46 * x2 + 34 * x3 + 89 * x4 = 70)) := 
by
  sorry

end system_inconsistent_l45_45595


namespace rectangle_length_l45_45491

theorem rectangle_length
  (side_length_square : ℝ)
  (width_rectangle : ℝ)
  (area_equiv : side_length_square ^ 2 = width_rectangle * l)
  : l = 24 := by
  sorry

end rectangle_length_l45_45491


namespace square_field_side_length_l45_45610

theorem square_field_side_length (time_sec : ℕ) (speed_kmh : ℕ) (perimeter : ℕ) (side_length : ℕ)
  (h1 : time_sec = 96)
  (h2 : speed_kmh = 9)
  (h3 : perimeter = (9 * 1000 / 3600 : ℕ) * 96)
  (h4 : perimeter = 4 * side_length) :
  side_length = 60 :=
by
  sorry

end square_field_side_length_l45_45610


namespace difference_quotient_correct_l45_45956

theorem difference_quotient_correct (a b : ℝ) :
  abs (3 * a - b) / abs (a + 2 * b) = abs (3 * a - b) / abs (a + 2 * b) :=
by
  sorry

end difference_quotient_correct_l45_45956


namespace value_b15_l45_45140

def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d
def geometric_sequence (b : ℕ → ℤ) := ∃ q : ℤ, ∀ n : ℕ, b (n+1) = q * b n

theorem value_b15 
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (b : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : ∀ n : ℕ, S n = (n * (a 0 + a (n-1)) / 2))
  (h3 : S 9 = -18)
  (h4 : S 13 = -52)
  (h5 : geometric_sequence b)
  (h6 : b 5 = a 5)
  (h7 : b 7 = a 7) : 
  b 15 = -64 :=
sorry

end value_b15_l45_45140


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l45_45466

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l45_45466


namespace max_value_y_l45_45868

variable (x : ℝ)
def y : ℝ := -3 * x^2 + 6

theorem max_value_y : ∃ M, ∀ x : ℝ, y x ≤ M ∧ (∀ x : ℝ, y x = M → x = 0) :=
by
  use 6
  sorry

end max_value_y_l45_45868


namespace solve_equation1_solve_equation2_l45_45024

-- Define the first equation (x-3)^2 + 2x(x-3) = 0
def equation1 (x : ℝ) : Prop := (x - 3)^2 + 2 * x * (x - 3) = 0

-- Define the second equation x^2 - 4x + 1 = 0
def equation2 (x : ℝ) : Prop := x^2 - 4 * x + 1 = 0

-- Theorem stating the solutions for the first equation
theorem solve_equation1 : ∀ (x : ℝ), equation1 x ↔ x = 3 ∨ x = 1 :=
by
  intro x
  sorry  -- Proof is omitted

-- Theorem stating the solutions for the second equation
theorem solve_equation2 : ∀ (x : ℝ), equation2 x ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
by
  intro x
  sorry  -- Proof is omitted

end solve_equation1_solve_equation2_l45_45024


namespace solve_for_pairs_l45_45612
-- Import necessary libraries

-- Define the operation
def diamond (a b c d : ℤ) : ℤ × ℤ :=
  (a * c - b * d, a * d + b * c)

theorem solve_for_pairs : ∃! (x y : ℤ), diamond x 3 x y = (6, 0) ∧ (x, y) = (0, -2) := by
  sorry

end solve_for_pairs_l45_45612


namespace rectangular_prism_sum_of_dimensions_l45_45500

theorem rectangular_prism_sum_of_dimensions (a b c : ℕ) (h_volume : a * b * c = 21) 
(h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) : 
a + b + c = 11 :=
sorry

end rectangular_prism_sum_of_dimensions_l45_45500


namespace eddy_time_to_B_l45_45782

-- Definitions
def distance_A_to_B : ℝ := 570
def distance_A_to_C : ℝ := 300
def time_C : ℝ := 4
def speed_ratio : ℝ := 2.5333333333333333

-- Theorem Statement
theorem eddy_time_to_B : 
  (distance_A_to_B / (distance_A_to_C / time_C * speed_ratio)) = 3 := 
by
  sorry

end eddy_time_to_B_l45_45782


namespace Lucas_identity_l45_45573

def Lucas (L : ℕ → ℤ) (F : ℕ → ℤ) : Prop :=
  ∀ n, L n = F (n + 1) + F (n - 1)

def Fib_identity1 (F : ℕ → ℤ) : Prop :=
  ∀ n, F (2 * n + 1) = F (n + 1) ^ 2 + F n ^ 2

def Fib_identity2 (F : ℕ → ℤ) : Prop :=
  ∀ n, F n ^ 2 = F (n + 1) * F (n - 1) - (-1) ^ n

theorem Lucas_identity {L F : ℕ → ℤ} (hL : Lucas L F) (hF1 : Fib_identity1 F) (hF2 : Fib_identity2 F) :
  ∀ n, L (2 * n) = L n ^ 2 - 2 * (-1) ^ n := 
sorry

end Lucas_identity_l45_45573


namespace fifth_term_sequence_l45_45518

theorem fifth_term_sequence 
  (a : ℕ → ℤ)
  (h1 : a 1 = 3)
  (h2 : a 2 = 6)
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n) :
  a 5 = -6 := 
by
  sorry

end fifth_term_sequence_l45_45518


namespace acute_angle_vector_range_l45_45831

theorem acute_angle_vector_range (m : ℝ) (a b : ℝ × ℝ) 
  (h1 : a = (1, 2)) 
  (h2 : b = (4, m)) 
  (acute : (a.1 * b.1 + a.2 * b.2) > 0) : 
  (m > -2) ∧ (m ≠ 8) := 
by 
  sorry

end acute_angle_vector_range_l45_45831


namespace circle_area_approx_error_exceeds_one_l45_45456

theorem circle_area_approx_error_exceeds_one (r : ℝ) : 
  (3.14159 < Real.pi ∧ Real.pi < 3.14160) → 
  2 * r > 25 →  
  |(r * r * Real.pi - r * r * 3.14)| > 1 → 
  2 * r = 51 := 
by 
  sorry

end circle_area_approx_error_exceeds_one_l45_45456


namespace roots_equal_when_m_l45_45350

noncomputable def equal_roots_condition (k n m : ℝ) : Prop :=
  1 + 4 * m^2 * k + 4 * m * n = 0

theorem roots_equal_when_m :
  equal_roots_condition 1 3 (-1.5 + Real.sqrt 2) ∧ 
  equal_roots_condition 1 3 (-1.5 - Real.sqrt 2) :=
by 
  sorry

end roots_equal_when_m_l45_45350


namespace quadratic_inequality_ab_l45_45884

theorem quadratic_inequality_ab (a b : ℝ) :
  (∀ x : ℝ, (x > -1 ∧ x < 1 / 3) → a * x^2 + b * x + 1 > 0) →
  a * b = 6 :=
sorry

end quadratic_inequality_ab_l45_45884


namespace abs_neg_eight_l45_45892

theorem abs_neg_eight : abs (-8) = 8 := by
  sorry

end abs_neg_eight_l45_45892


namespace union_A_B_eq_intersection_A_B_complement_eq_l45_45127

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 - 4 * x ≤ 0}
def B_complement : Set ℝ := {x | x < 0 ∨ x > 4}

theorem union_A_B_eq : A ∪ B = {x | -1 ≤ x ∧ x ≤ 4} := by
  sorry

theorem intersection_A_B_complement_eq : A ∩ B_complement = {x | -1 ≤ x ∧ x < 0} := by
  sorry

end union_A_B_eq_intersection_A_B_complement_eq_l45_45127


namespace Rebecca_tips_calculation_l45_45426

def price_haircut : ℤ := 30
def price_perm : ℤ := 40
def price_dye_job : ℤ := 60
def cost_hair_dye_box : ℤ := 10
def num_haircuts : ℕ := 4
def num_perms : ℕ := 1
def num_dye_jobs : ℕ := 2
def total_end_day : ℤ := 310

noncomputable def total_service_earnings : ℤ := 
  num_haircuts * price_haircut + num_perms * price_perm + num_dye_jobs * price_dye_job

noncomputable def total_hair_dye_cost : ℤ := 
  num_dye_jobs * cost_hair_dye_box

noncomputable def earnings_after_cost : ℤ := 
  total_service_earnings - total_hair_dye_cost

noncomputable def tips : ℤ := 
  total_end_day - earnings_after_cost

theorem Rebecca_tips_calculation : tips = 50 := by
  sorry

end Rebecca_tips_calculation_l45_45426


namespace total_bill_l45_45642

/-
Ten friends dined at a restaurant and split the bill equally.
One friend, Chris, forgets his money.
Each of the remaining nine friends agreed to pay an extra $3 to cover Chris's share.
How much was the total bill?

Correct answer: 270
-/

theorem total_bill (t : ℕ) (h1 : ∀ x, t = 10 * x) (h2 : ∀ x, t = 9 * (x + 3)) : t = 270 := by
  sorry

end total_bill_l45_45642


namespace find_y_value_l45_45982

theorem find_y_value : (12 : ℕ)^3 * (6 : ℕ)^2 / 432 = 144 := by
  -- assumptions and computations are not displayed in the statement
  sorry

end find_y_value_l45_45982


namespace happy_boys_count_l45_45401

def total_children := 60
def happy_children := 30
def sad_children := 10
def neither_happy_nor_sad_children := total_children - happy_children - sad_children

def total_boys := 19
def total_girls := 41
def sad_girls := 4
def neither_happy_nor_sad_boys := 7

def sad_boys := sad_children - sad_girls

theorem happy_boys_count :
  total_boys - sad_boys - neither_happy_nor_sad_boys = 6 :=
by
  sorry

end happy_boys_count_l45_45401


namespace solution_set_f_l45_45840

noncomputable def f (x : ℝ) : ℝ := sorry -- The differentiable function f

axiom f_deriv_lt (x : ℝ) : deriv f x < x -- Condition on the derivative of f
axiom f_at_2 : f 2 = 1 -- Given f(2) = 1

theorem solution_set_f : ∀ x : ℝ, f x < (1 / 2) * x^2 - 1 ↔ x > 2 :=
by sorry

end solution_set_f_l45_45840


namespace samia_walked_distance_l45_45670

theorem samia_walked_distance :
  ∀ (total_distance cycling_speed walking_speed total_time : ℝ), 
  total_distance = 18 → 
  cycling_speed = 20 → 
  walking_speed = 4 → 
  total_time = 1 + 10 / 60 → 
  2 / 3 * total_distance / cycling_speed + 1 / 3 * total_distance / walking_speed = total_time → 
  1 / 3 * total_distance = 6 := 
by
  intros total_distance cycling_speed walking_speed total_time h1 h2 h3 h4 h5
  sorry

end samia_walked_distance_l45_45670


namespace lunks_needed_for_apples_l45_45796

theorem lunks_needed_for_apples :
  (∀ l k a : ℕ, (4 * k = 2 * l) ∧ (3 * a = 5 * k ) → ∃ l', l' = (24 * l / 4)) :=
by
  intros l k a h
  obtain ⟨h1, h2⟩ := h
  have k_for_apples := 3 * a / 5
  have l_for_kunks := 4 * k / 2
  sorry

end lunks_needed_for_apples_l45_45796


namespace M_ends_in_two_zeros_iff_l45_45009

theorem M_ends_in_two_zeros_iff (n : ℕ) (h : n > 0) : 
  (1^n + 2^n + 3^n + 4^n) % 100 = 0 ↔ n % 4 = 3 :=
by sorry

end M_ends_in_two_zeros_iff_l45_45009


namespace ratio_areas_l45_45160

theorem ratio_areas (H : ℝ) (L : ℝ) (r : ℝ) (A_rectangle : ℝ) (A_circle : ℝ) :
  H = 45 ∧ (L / H = 4 / 3) ∧ r = H / 2 ∧ A_rectangle = L * H ∧ A_circle = π * r^2 →
  (A_rectangle / A_circle = 17 / π) :=
by
  sorry

end ratio_areas_l45_45160


namespace final_selling_price_l45_45559

-- Define the conditions as constants
def CP := 750
def loss_percentage := 20 / 100
def sales_tax_percentage := 10 / 100

-- Define the final selling price after loss and adding sales tax
theorem final_selling_price 
  (CP : ℝ) 
  (loss_percentage : ℝ)
  (sales_tax_percentage : ℝ) 
  : 750 = CP ∧ 20 / 100 = loss_percentage ∧ 10 / 100 = sales_tax_percentage → 
    (CP - (loss_percentage * CP) + (sales_tax_percentage * CP) = 675) := 
by
  intros
  sorry

end final_selling_price_l45_45559


namespace tiled_board_remainder_l45_45999

def num_ways_to_tile_9x1 : Nat := -- hypothetical function to calculate the number of ways
  sorry

def N : Nat :=
  num_ways_to_tile_9x1 -- placeholder for N, should be computed using correct formula

theorem tiled_board_remainder : N % 1000 = 561 :=
  sorry

end tiled_board_remainder_l45_45999


namespace initial_bushes_count_l45_45174

theorem initial_bushes_count (n : ℕ) (h : 2 * (27 * n - 26) + 26 = 190 + 26) : n = 8 :=
by
  sorry

end initial_bushes_count_l45_45174


namespace evaluate_f_2x_l45_45557

def f (x : ℝ) : ℝ := x^2 - 1

theorem evaluate_f_2x (x : ℝ) : f (2 * x) = 4 * x^2 - 1 :=
by
  sorry

end evaluate_f_2x_l45_45557


namespace decreasing_cubic_function_l45_45010

theorem decreasing_cubic_function (a : ℝ) :
  (∀ x : ℝ, 3 * a * x^2 - 1 ≤ 0) → a ≤ 0 :=
sorry

end decreasing_cubic_function_l45_45010


namespace sum_of_arithmetic_sequence_is_54_l45_45415

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence_is_54 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h_condition : 2 * a 8 = 6 + a 11) : 
  S 9 = 54 :=
sorry

end sum_of_arithmetic_sequence_is_54_l45_45415


namespace josiah_hans_age_ratio_l45_45465

theorem josiah_hans_age_ratio (H : ℕ) (J : ℕ) (hH : H = 15) (hSum : (J + 3) + (H + 3) = 66) : J / H = 3 :=
by
  sorry

end josiah_hans_age_ratio_l45_45465


namespace intersection_A_B_eq_C_l45_45090

def A : Set ℝ := { x | 4 - x^2 ≥ 0 }
def B : Set ℝ := { x | x > -1 }
def C : Set ℝ := { x | -1 < x ∧ x ≤ 2 }

theorem intersection_A_B_eq_C : A ∩ B = C := 
by {
  sorry
}

end intersection_A_B_eq_C_l45_45090


namespace area_of_ABCD_proof_l45_45440

noncomputable def point := ℝ × ℝ

structure Rectangle :=
  (A B C D : point)
  (angle_C_trisected_by_CE_CF : Prop)
  (E_on_AB : Prop)
  (F_on_AD : Prop)
  (AF : ℝ)
  (BE : ℝ)

def area_of_rectangle (rect : Rectangle) : ℝ :=
  let (x1, y1) := rect.A
  let (x2, y2) := rect.C
  (x2 - x1) * (y2 - y1)

theorem area_of_ABCD_proof :
  ∀ (ABCD : Rectangle),
    ABCD.angle_C_trisected_by_CE_CF →
    ABCD.E_on_AB →
    ABCD.F_on_AD →
    ABCD.AF = 2 →
    ABCD.BE = 6 →
    abs (area_of_rectangle ABCD - 150) < 1 :=
by
  sorry

end area_of_ABCD_proof_l45_45440


namespace survey_households_selected_l45_45231

theorem survey_households_selected 
    (total_households : ℕ) 
    (middle_income_families : ℕ) 
    (low_income_families : ℕ) 
    (high_income_selected : ℕ)
    (total_high_income_families : ℕ)
    (total_selected_households : ℕ) 
    (H1 : total_households = 480)
    (H2 : middle_income_families = 200)
    (H3 : low_income_families = 160)
    (H4 : high_income_selected = 6)
    (H5 : total_high_income_families = total_households - (middle_income_families + low_income_families))
    (H6 : total_selected_households * total_high_income_families = high_income_selected * total_households) :
    total_selected_households = 24 :=
by
  -- The actual proof will go here:
  sorry

end survey_households_selected_l45_45231


namespace max_value_expression_l45_45833

noncomputable def target_expr (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (y^2 - y * z + z^2) * (z^2 - z * x + x^2)

theorem max_value_expression (x y z : ℝ) (h : x + y + z = 3) (hxy : x = y) (hxz : 0 ≤ x) (hyz : 0 ≤ y) (hzz : 0 ≤ z) :
  target_expr x y z ≤ 9 / 4 := by
  sorry

end max_value_expression_l45_45833


namespace minimize_potato_cost_l45_45802

def potatoes_distribution (x1 x2 x3 : ℚ) : Prop :=
  x1 ≥ 0 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧
  x1 + x2 + x3 = 12 ∧
  x1 + 4 * x2 + 3 * x3 ≤ 40 ∧
  x1 ≤ 10 ∧ x2 ≤ 8 ∧ x3 ≤ 6 ∧
  4 * x1 + 3 * x2 + 1 * x3 = (74 / 3)

theorem minimize_potato_cost :
  ∃ x1 x2 x3 : ℚ, potatoes_distribution x1 x2 x3 ∧ x1 = (2/3) ∧ x2 = (16/3) ∧ x3 = 6 :=
by
  sorry

end minimize_potato_cost_l45_45802


namespace toothpicks_for_10_squares_l45_45270

theorem toothpicks_for_10_squares : (4 + 3 * (10 - 1)) = 31 :=
by 
  sorry

end toothpicks_for_10_squares_l45_45270


namespace new_person_weight_l45_45729

theorem new_person_weight (avg_increase : Real) (n : Nat) (old_weight : Real) (W_new : Real) :
  avg_increase = 2.5 → n = 8 → old_weight = 67 → W_new = old_weight + n * avg_increase → W_new = 87 :=
by
  intros avg_increase_eq n_eq old_weight_eq calc_eq
  sorry

end new_person_weight_l45_45729


namespace calculate_expression_l45_45704

theorem calculate_expression : 200 * 39.96 * 3.996 * 500 = (3996)^2 :=
by
  sorry

end calculate_expression_l45_45704


namespace hans_room_count_l45_45234

theorem hans_room_count :
  let total_floors := 10
  let rooms_per_floor := 10
  let unavailable_floors := 1
  let available_floors := total_floors - unavailable_floors
  available_floors * rooms_per_floor = 90 := by
  let total_floors := 10
  let rooms_per_floor := 10
  let unavailable_floors := 1
  let available_floors := total_floors - unavailable_floors
  show available_floors * rooms_per_floor = 90
  sorry

end hans_room_count_l45_45234


namespace average_of_two_intermediate_numbers_l45_45431

theorem average_of_two_intermediate_numbers (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
(h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
(h_average : (a + b + c + d) / 4 = 5)
(h_max_diff: (max (max a b) (max c d) - min (min a b) (min c d) = 19)) :
  (a + b + c + d) - (max (max a b) (max c d)) - (min (min a b) (min c d)) = 5 :=
by
  -- The proof goes here
  sorry

end average_of_two_intermediate_numbers_l45_45431


namespace hypotenuse_square_l45_45286

theorem hypotenuse_square (a : ℕ) : (a + 1)^2 + a^2 = 2 * a^2 + 2 * a + 1 := 
by sorry

end hypotenuse_square_l45_45286


namespace greatest_integer_difference_l45_45959

theorem greatest_integer_difference (x y : ℤ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 7) : y - x = 3 :=
sorry

end greatest_integer_difference_l45_45959


namespace girls_boys_difference_l45_45501

variables (B G : ℕ) (x : ℕ)

-- Condition that relates boys and girls with a ratio
def ratio_condition : Prop := 3 * x = B ∧ 4 * x = G

-- Condition that the total number of students is 42
def total_students_condition : Prop := B + G = 42

-- We want to prove that the difference between the number of girls and boys is 6
theorem girls_boys_difference (h_ratio : ratio_condition B G x) (h_total : total_students_condition B G) : 
  G - B = 6 :=
sorry

end girls_boys_difference_l45_45501


namespace find_n_l45_45822

theorem find_n (n : ℕ) : (8 : ℝ)^(1/3) = (2 : ℝ)^n → n = 1 := by
  sorry

end find_n_l45_45822


namespace rational_powers_imply_integers_l45_45754

theorem rational_powers_imply_integers (a b : ℚ) (h_distinct : a ≠ b)
  (h_infinitely_many_n : ∃ᶠ (n : ℕ) in Filter.atTop, (n * (a^n - b^n) : ℚ).den = 1) :
  ∃ (a_int b_int : ℤ), a = a_int ∧ b = b_int := 
sorry

end rational_powers_imply_integers_l45_45754


namespace no_n_divisible_by_1955_l45_45857

theorem no_n_divisible_by_1955 : ∀ n : ℕ, ¬ (1955 ∣ (n^2 + n + 1)) := by
  sorry

end no_n_divisible_by_1955_l45_45857


namespace overall_percent_change_l45_45389

theorem overall_percent_change (x : ℝ) : 
  (0.85 * x * 1.25 * 0.9 / x - 1) * 100 = -4.375 := 
by 
  sorry

end overall_percent_change_l45_45389


namespace largest_divisor_of_n_pow4_minus_n_for_composites_l45_45069

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬(∀ k : ℕ, 1 < k ∧ k < n → ¬(k ∣ n))

theorem largest_divisor_of_n_pow4_minus_n_for_composites : ∀ n : ℕ, is_composite n → 6 ∣ (n^4 - n) :=
by
  intro n
  intro hn
  -- we would add proof steps here if necessary
  sorry

end largest_divisor_of_n_pow4_minus_n_for_composites_l45_45069


namespace average_speed_is_37_5_l45_45865

-- Define the conditions
def distance_local : ℕ := 60
def speed_local : ℕ := 30
def distance_gravel : ℕ := 10
def speed_gravel : ℕ := 20
def distance_highway : ℕ := 105
def speed_highway : ℕ := 60
def traffic_delay : ℚ := 15 / 60
def obstruction_delay : ℚ := 10 / 60

-- Define the total distance
def total_distance : ℕ := distance_local + distance_gravel + distance_highway

-- Define the total time
def total_time : ℚ :=
  (distance_local / speed_local) +
  (distance_gravel / speed_gravel) +
  (distance_highway / speed_highway) +
  traffic_delay +
  obstruction_delay

-- Define the average speed as distance divided by time
def average_speed : ℚ := total_distance / total_time

theorem average_speed_is_37_5 :
  average_speed = 37.5 := by sorry

end average_speed_is_37_5_l45_45865


namespace range_of_a_l45_45825

theorem range_of_a : (∀ x : ℝ, x^2 + (a-1)*x + 1 > 0) ↔ (-1 < a ∧ a < 3) := by
  sorry

end range_of_a_l45_45825


namespace number_of_new_students_l45_45918

theorem number_of_new_students (initial_students left_students final_students new_students : ℕ) 
  (h_initial : initial_students = 4) 
  (h_left : left_students = 3) 
  (h_final : final_students = 43) : 
  new_students = final_students - (initial_students - left_students) :=
by 
  sorry

end number_of_new_students_l45_45918


namespace cycle_selling_price_l45_45915

noncomputable def selling_price (cost_price : ℝ) (gain_percent : ℝ) : ℝ :=
  let gain_amount := (gain_percent / 100) * cost_price
  cost_price + gain_amount

theorem cycle_selling_price :
  selling_price 450 15.56 = 520.02 :=
by
  sorry

end cycle_selling_price_l45_45915


namespace prime_remainder_l45_45487

theorem prime_remainder (p : ℕ) (k : ℕ) (h1 : Prime p) (h2 : p > 3) :
  (∃ k, p = 6 * k + 1 ∧ (p^3 + 17) % 24 = 18) ∨
  (∃ k, p = 6 * k - 1 ∧ (p^3 + 17) % 24 = 16) :=
by
  sorry

end prime_remainder_l45_45487


namespace least_number_subtracted_l45_45074

theorem least_number_subtracted (n : ℕ) (h : n = 427398) : ∃ x, x = 8 ∧ (n - x) % 10 = 0 :=
by
  sorry

end least_number_subtracted_l45_45074


namespace sport_flavoring_to_water_ratio_l45_45844

/-- The ratio by volume of flavoring to corn syrup to water in the 
standard formulation is 1:12:30. The sport formulation has a ratio 
of flavoring to corn syrup three times as great as in the standard formulation. 
A large bottle of the sport formulation contains 4 ounces of corn syrup and 
60 ounces of water. Prove that the ratio of the amount of flavoring to water 
in the sport formulation compared to the standard formulation is 1:2. -/
theorem sport_flavoring_to_water_ratio 
    (standard_flavoring : ℝ) 
    (standard_corn_syrup : ℝ) 
    (standard_water : ℝ) : 
  standard_flavoring = 1 → standard_corn_syrup = 12 → 
  standard_water = 30 → 
  ∃ sport_flavoring : ℝ, 
  ∃ sport_corn_syrup : ℝ, 
  ∃ sport_water : ℝ, 
  sport_corn_syrup = 4 ∧ 
  sport_water = 60 ∧ 
  (sport_flavoring / sport_water) = (standard_flavoring / standard_water) / 2 :=
by
  sorry

end sport_flavoring_to_water_ratio_l45_45844


namespace age_of_new_person_l45_45400

theorem age_of_new_person (T A : ℤ) (h : (T / 10 - 3) = (T - 40 + A) / 10) : A = 10 :=
sorry

end age_of_new_person_l45_45400


namespace inverse_proportion_relation_l45_45824

theorem inverse_proportion_relation (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 2 / x₁) 
  (h2 : y₂ = 2 / x₂) 
  (h3 : x₁ < x₂) 
  (h4 : x₂ < 0) : 
  y₂ < y₁ ∧ y₁ < 0 := 
sorry

end inverse_proportion_relation_l45_45824


namespace haley_laundry_loads_l45_45237

theorem haley_laundry_loads (shirts sweaters pants socks : ℕ) 
    (machine_capacity total_pieces : ℕ)
    (sum_of_clothing : 6 + 28 + 10 + 9 = total_pieces)
    (machine_capacity_eq : machine_capacity = 5) :
  ⌈(total_pieces:ℚ) / machine_capacity⌉ = 11 :=
by
  sorry

end haley_laundry_loads_l45_45237


namespace a2_value_is_42_l45_45753

noncomputable def a₂_value (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :=
  a_2

theorem a2_value_is_42 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :
  (x^3 + x^10 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + a_4 * (x + 1)^4 +
                a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + a_7 * (x + 1)^7 + a_8 * (x + 1)^8 + 
                a_9 * (x + 1)^9 + a_10 * (x + 1)^10) →
  a₂_value a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 = 42 :=
by
  sorry

end a2_value_is_42_l45_45753


namespace percentage_is_40_l45_45248

variables (num : ℕ) (perc : ℕ)

-- Conditions
def ten_percent_eq_40 : Prop := 10 * num = 400
def certain_percentage_eq_160 : Prop := perc * num = 160 * 100

-- Statement to prove
theorem percentage_is_40 (h1 : ten_percent_eq_40 num) (h2 : certain_percentage_eq_160 num perc) : perc = 40 :=
sorry

end percentage_is_40_l45_45248


namespace product_of_real_values_eq_4_l45_45045

theorem product_of_real_values_eq_4 : ∀ s : ℝ, 
  (∃ x : ℝ, x ≠ 0 ∧ (1/(3*x) = (s - x)/9) → 
  (∀ x : ℝ, x ≠ 0 → (1/(3*x) = (s - x)/9 → x = s - 3))) → s = 4 :=
by
  sorry

end product_of_real_values_eq_4_l45_45045


namespace bike_travel_distance_l45_45932

-- Declaring the conditions as definitions
def speed : ℝ := 50 -- Speed in meters per second
def time : ℝ := 7 -- Time in seconds

-- Declaring the question and expected answer
def expected_distance : ℝ := 350 -- Expected distance in meters

-- The proof statement that needs to be proved
theorem bike_travel_distance : (speed * time = expected_distance) :=
by
  sorry

end bike_travel_distance_l45_45932


namespace bottles_needed_l45_45213

theorem bottles_needed (runners : ℕ) (bottles_needed_per_runner : ℕ) (bottles_available : ℕ)
  (h_runners : runners = 14)
  (h_bottles_needed_per_runner : bottles_needed_per_runner = 5)
  (h_bottles_available : bottles_available = 68) :
  runners * bottles_needed_per_runner - bottles_available = 2 :=
by
  sorry

end bottles_needed_l45_45213


namespace evaluate_fraction_sum_l45_45282

variable (a b c : ℝ)

theorem evaluate_fraction_sum
  (h : (a / (30 - a)) + (b / (70 - b)) + (c / (80 - c)) = 9) :
  (6 / (30 - a)) + (14 / (70 - b)) + (16 / (80 - c)) = 2.4 :=
by
  sorry

end evaluate_fraction_sum_l45_45282


namespace students_not_taking_either_l45_45938

-- Definitions of the conditions
def total_students : ℕ := 28
def students_taking_french : ℕ := 5
def students_taking_spanish : ℕ := 10
def students_taking_both : ℕ := 4

-- Theorem stating the mathematical problem
theorem students_not_taking_either :
  total_students - (students_taking_french + students_taking_spanish + students_taking_both) = 9 :=
sorry

end students_not_taking_either_l45_45938


namespace algebraic_expression_value_l45_45379

variable (x y A B : ℤ)
variable (x_val : x = -1)
variable (y_val : y = 2)
variable (A_def : A = 2*x + y)
variable (B_def : B = 2*x - y)

theorem algebraic_expression_value : 
  (A^2 - B^2) * (x - 2*y) = 80 := 
by
  rw [x_val, y_val, A_def, B_def]
  sorry

end algebraic_expression_value_l45_45379


namespace find_x_l45_45242

def delta (x : ℝ) : ℝ := 4 * x + 9
def phi (x : ℝ) : ℝ := 9 * x + 6

theorem find_x (x : ℝ) (h : delta (phi x) = 10) : x = -23 / 36 := 
by 
  sorry

end find_x_l45_45242


namespace find_Y_l45_45321

-- Definition of the problem.
def arithmetic_sequence (a d n : ℕ) : ℕ := a + d * (n - 1)

-- Conditions provided in the problem.
-- Conditions of the first row
def first_row (a₁ a₄ : ℕ) : Prop :=
  a₁ = 4 ∧ a₄ = 16

-- Conditions of the last row
def last_row (a₁' a₄' : ℕ) : Prop :=
  a₁' = 10 ∧ a₄' = 40

-- Value of Y (the second element of the second row from the second column)
def center_top_element (Y : ℕ) : Prop :=
  Y = 12

-- The theorem to prove.
theorem find_Y (a₁ a₄ a₁' a₄' Y : ℕ) (h1 : first_row a₁ a₄) (h2 : last_row a₁' a₄') (h3 : center_top_element Y) : Y = 12 := 
by 
  sorry -- proof to be provided.

end find_Y_l45_45321


namespace range_of_m_l45_45247

theorem range_of_m (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) →
  (1 < m) :=
by
  sorry

end range_of_m_l45_45247


namespace highest_throw_christine_janice_l45_45714

theorem highest_throw_christine_janice
  (c1 : ℕ) -- Christine's first throw
  (j1 : ℕ) -- Janice's first throw
  (c2 : ℕ) -- Christine's second throw
  (j2 : ℕ) -- Janice's second throw
  (c3 : ℕ) -- Christine's third throw
  (j3 : ℕ) -- Janice's third throw
  (h1 : c1 = 20)
  (h2 : j1 = c1 - 4)
  (h3 : c2 = c1 + 10)
  (h4 : j2 = j1 * 2)
  (h5 : c3 = c2 + 4)
  (h6 : j3 = c1 + 17) :
  max c1 (max c2 (max c3 (max j1 (max j2 j3)))) = 37 := by
  sorry

end highest_throw_christine_janice_l45_45714


namespace value_of_g_3_l45_45624

def g (x : ℚ) : ℚ := (x^2 + x + 1) / (5*x - 3)

theorem value_of_g_3 : g 3 = 13 / 12 :=
by
  -- Proof goes here
  sorry

end value_of_g_3_l45_45624


namespace intersection_eq_l45_45944

def M : Set ℝ := {x | ∃ y, y = Real.log (2 - x) / Real.log 3}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_eq : M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
sorry

end intersection_eq_l45_45944


namespace dot_product_result_l45_45773

def a : ℝ × ℝ := (-1, 2)
def b (m : ℝ) : ℝ × ℝ := (2, m)
def c : ℝ × ℝ := (7, 1)

def are_parallel (a b : ℝ × ℝ) : Prop := 
  a.1 * b.2 = a.2 * b.1

def dot_product (u v : ℝ × ℝ) : ℝ := 
  u.1 * v.1 + u.2 * v.2

theorem dot_product_result : 
  ∀ m : ℝ, are_parallel a (b m) → dot_product (b m) c = 10 := 
by
  sorry

end dot_product_result_l45_45773


namespace even_function_a_value_l45_45189

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (x^2 + (a^2 - 1) * x + (a - 1)) = ((-x)^2 + (a^2 - 1) * (-x) + (a - 1))) → (a = 1 ∨ a = -1) :=
by
  sorry

end even_function_a_value_l45_45189


namespace find_x_for_g_inv_l45_45148

def g (x : ℝ) : ℝ := 5 * x ^ 3 - 4 * x + 1

theorem find_x_for_g_inv (x : ℝ) (h : g 3 = x) : g⁻¹ 3 = 3 :=
by
  sorry

end find_x_for_g_inv_l45_45148


namespace stratified_sampling_numbers_l45_45088

-- Definitions of the conditions
def total_teachers : ℕ := 300
def senior_teachers : ℕ := 90
def intermediate_teachers : ℕ := 150
def junior_teachers : ℕ := 60
def sample_size : ℕ := 40

-- Hypothesis of proportions
def proportion_senior := senior_teachers / total_teachers
def proportion_intermediate := intermediate_teachers / total_teachers
def proportion_junior := junior_teachers / total_teachers

-- Expected sample counts using stratified sampling method
def expected_senior_drawn := proportion_senior * sample_size
def expected_intermediate_drawn := proportion_intermediate * sample_size
def expected_junior_drawn := proportion_junior * sample_size

-- Proof goal
theorem stratified_sampling_numbers :
  (expected_senior_drawn = 12) ∧ 
  (expected_intermediate_drawn = 20) ∧ 
  (expected_junior_drawn = 8) :=
by
  sorry

end stratified_sampling_numbers_l45_45088


namespace relationship_bx_x2_a2_l45_45390

theorem relationship_bx_x2_a2 {a b x : ℝ} (h1 : b < x) (h2 : x < a) (h3 : 0 < a) (h4 : 0 < b) : 
  b * x < x^2 ∧ x^2 < a^2 :=
by sorry

end relationship_bx_x2_a2_l45_45390


namespace meal_cost_with_tip_l45_45327

theorem meal_cost_with_tip 
  (cost_samosas : ℕ := 3 * 2)
  (cost_pakoras : ℕ := 4 * 3)
  (cost_lassi : ℕ := 2)
  (total_cost_before_tip := cost_samosas + cost_pakoras + cost_lassi)
  (tip : ℝ := 0.25 * total_cost_before_tip) :
  (total_cost_before_tip + tip = 25) :=
sorry

end meal_cost_with_tip_l45_45327


namespace FerrisWheelCostIsSix_l45_45458

structure AmusementPark where
  roller_coaster_cost : ℕ
  log_ride_cost : ℕ
  initial_tickets : ℕ
  additional_tickets_needed : ℕ

def ferris_wheel_cost (a : AmusementPark) : ℕ :=
  let total_needed := a.initial_tickets + a.additional_tickets_needed
  let total_ride_cost := a.roller_coaster_cost + a.log_ride_cost
  total_needed - total_ride_cost

theorem FerrisWheelCostIsSix (a : AmusementPark) 
  (h₁ : a.roller_coaster_cost = 5)
  (h₂ : a.log_ride_cost = 7)
  (h₃ : a.initial_tickets = 2)
  (h₄ : a.additional_tickets_needed = 16) :
  ferris_wheel_cost a = 6 :=
by
  -- proof omitted
  sorry

end FerrisWheelCostIsSix_l45_45458


namespace ratio_of_areas_l45_45377

theorem ratio_of_areas (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
    let S₁ := (1 - p * q * r) * (1 - p * q * r)
    let S₂ := (1 + p + p * q) * (1 + q + q * r) * (1 + r + r * p)
    S₁ / S₂ = (S₁ / S₂) := sorry

end ratio_of_areas_l45_45377


namespace intersection_point_of_lines_l45_45537

theorem intersection_point_of_lines :
  ∃ x y : ℝ, 3 * x + 4 * y - 2 = 0 ∧ 2 * x + y + 2 = 0 := 
by
  sorry

end intersection_point_of_lines_l45_45537


namespace solve_for_x_l45_45826

theorem solve_for_x (x : ℝ) (hx : x^(1/10) * (x^(3/2))^(1/10) = 3) : x = 9 :=
sorry

end solve_for_x_l45_45826


namespace total_distance_travelled_l45_45409

theorem total_distance_travelled (total_time hours_foot hours_bicycle speed_foot speed_bicycle distance_foot : ℕ)
  (h1 : total_time = 7)
  (h2 : speed_foot = 8)
  (h3 : speed_bicycle = 16)
  (h4 : distance_foot = 32)
  (h5 : hours_foot = distance_foot / speed_foot)
  (h6 : hours_bicycle = total_time - hours_foot)
  (distance_bicycle := speed_bicycle * hours_bicycle) :
  distance_foot + distance_bicycle = 80 := 
by
  sorry

end total_distance_travelled_l45_45409


namespace ratio_of_canoes_to_kayaks_l45_45335

theorem ratio_of_canoes_to_kayaks 
    (canoe_cost kayak_cost total_revenue : ℕ) 
    (canoe_to_kayak_ratio extra_canoes : ℕ)
    (h1 : canoe_cost = 14)
    (h2 : kayak_cost = 15)
    (h3 : total_revenue = 288)
    (h4 : extra_canoes = 4)
    (h5 : canoe_to_kayak_ratio = 3) 
    (c k : ℕ)
    (h6 : c = k + extra_canoes)
    (h7 : c = canoe_to_kayak_ratio * k)
    (h8 : canoe_cost * c + kayak_cost * k = total_revenue) :
    c / k = 3 := 
sorry

end ratio_of_canoes_to_kayaks_l45_45335


namespace flagpole_height_l45_45562

theorem flagpole_height
  (AB : ℝ) (AD : ℝ) (BC : ℝ)
  (h1 : AB = 10)
  (h2 : BC = 3)
  (h3 : 2 * AD^2 = AB^2 + BC^2) :
  AD = Real.sqrt 54.5 :=
by 
  -- Proof omitted
  sorry

end flagpole_height_l45_45562


namespace problem1_problem2_problem3_problem4_l45_45912

-- Problem 1
theorem problem1 : (2 / 19) * (8 / 25) + (17 / 25) / (19 / 2) = 2 / 19 := 
by sorry

-- Problem 2
theorem problem2 : (1 / 4) * 125 * (1 / 25) * 8 = 10 := 
by sorry

-- Problem 3
theorem problem3 : ((1 / 3) + (1 / 4)) / ((1 / 2) - (1 / 3)) = 7 / 2 := 
by sorry

-- Problem 4
theorem problem4 : ((1 / 6) + (1 / 8)) * 24 * (1 / 9) = 7 / 9 := 
by sorry

end problem1_problem2_problem3_problem4_l45_45912


namespace solution_set_of_inequality_l45_45371

theorem solution_set_of_inequality : 
  { x : ℝ | (3 - 2 * x) * (x + 1) ≤ 0 } = { x : ℝ | -1 ≤ x ∧ x ≤ 3 / 2 } :=
sorry

end solution_set_of_inequality_l45_45371


namespace speed_of_boat_in_still_water_l45_45339

-- Define a structure for the conditions
structure BoatConditions where
  V_b : ℝ    -- Speed of the boat in still water
  V_s : ℝ    -- Speed of the stream
  goes_along_stream : V_b + V_s = 11
  goes_against_stream : V_b - V_s = 5

-- Define the target theorem
theorem speed_of_boat_in_still_water (c : BoatConditions) : c.V_b = 8 :=
by
  sorry

end speed_of_boat_in_still_water_l45_45339


namespace fraction_value_l45_45987

theorem fraction_value (x y z : ℝ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) : (x + y + z) / (2 * z) = 9 / 8 :=
by
  sorry

end fraction_value_l45_45987


namespace smallest_number_to_add_for_divisibility_l45_45105

theorem smallest_number_to_add_for_divisibility :
  ∃ x : ℕ, 1275890 + x ≡ 0 [MOD 2375] ∧ x = 1360 :=
by sorry

end smallest_number_to_add_for_divisibility_l45_45105


namespace sum_of_squares_l45_45388

theorem sum_of_squares (a b c : ℝ) :
  a + b + c = 4 → ab + ac + bc = 4 → a^2 + b^2 + c^2 = 8 :=
by
  sorry

end sum_of_squares_l45_45388


namespace numberOfBooks_correct_l45_45445

variable (totalWeight : ℕ) (weightPerBook : ℕ)

def numberOfBooks (totalWeight weightPerBook : ℕ) : ℕ :=
  totalWeight / weightPerBook

theorem numberOfBooks_correct (h1 : totalWeight = 42) (h2 : weightPerBook = 3) :
  numberOfBooks totalWeight weightPerBook = 14 := by
  sorry

end numberOfBooks_correct_l45_45445


namespace total_pages_in_scifi_section_l45_45847

theorem total_pages_in_scifi_section : 
  let books := 8
  let pages_per_book := 478
  books * pages_per_book = 3824 := 
by
  sorry

end total_pages_in_scifi_section_l45_45847


namespace slips_with_3_count_l45_45949

def number_of_slips_with_3 (x : ℕ) : Prop :=
  let total_slips := 15
  let expected_value := 4.6
  let prob_3 := (x : ℚ) / total_slips
  let prob_8 := (total_slips - x : ℚ) / total_slips
  let E := prob_3 * 3 + prob_8 * 8
  E = expected_value

theorem slips_with_3_count : ∃ x : ℕ, number_of_slips_with_3 x ∧ x = 10 :=
by
  sorry

end slips_with_3_count_l45_45949


namespace total_pears_picked_l45_45354

def mikes_pears : Nat := 8
def jasons_pears : Nat := 7
def freds_apples : Nat := 6

theorem total_pears_picked : (mikes_pears + jasons_pears) = 15 :=
by
  sorry

end total_pears_picked_l45_45354


namespace total_number_of_workers_l45_45769

theorem total_number_of_workers 
    (W N : ℕ) 
    (h1 : 8000 * W = 12000 * 8 + 6000 * N) 
    (h2 : W = 8 + N) : 
    W = 24 :=
by
  sorry

end total_number_of_workers_l45_45769


namespace probability_of_interval_l45_45119

-- Define the random variable ξ and its probability distribution P(ξ = k)
variables (ξ : ℕ → ℝ) (P : ℕ → ℝ)

-- Define a constant a
noncomputable def a : ℝ := 5/4

-- Given conditions
axiom condition1 : ∀ k, k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 → P k = a / (k * (k + 1))
axiom condition2 : P 1 + P 2 + P 3 + P 4 = 1

-- Statement to prove
theorem probability_of_interval : P 1 + P 2 = 5/6 :=
by sorry

end probability_of_interval_l45_45119


namespace evaluate_fraction_l45_45425

open Complex

theorem evaluate_fraction (a b : ℂ) (h : a ≠ 0 ∧ b ≠ 0) (h_eq : a^2 - a*b + b^2 = 0) : 
  (a^6 + b^6) / (a + b)^6 = 1 / 18 := by
  sorry

end evaluate_fraction_l45_45425


namespace equal_charges_at_4_hours_l45_45631

-- Define the charges for both companies
def PaulsPlumbingCharge (h : ℝ) : ℝ := 55 + 35 * h
def ReliablePlumbingCharge (h : ℝ) : ℝ := 75 + 30 * h

-- Prove that for 4 hours of labor, the charges are equal
theorem equal_charges_at_4_hours : PaulsPlumbingCharge 4 = ReliablePlumbingCharge 4 :=
by
  sorry

end equal_charges_at_4_hours_l45_45631


namespace find_x_in_sequence_l45_45481

theorem find_x_in_sequence :
  ∃ x : ℕ, x = 32 ∧
    2 + 3 = 5 ∧
    5 + 6 = 11 ∧
    11 + 9 = 20 ∧
    20 + (9 + 3) = x ∧
    x + (9 + 3 + 3) = 47 :=
by
  sorry

end find_x_in_sequence_l45_45481


namespace inequality_transformation_incorrect_l45_45507

theorem inequality_transformation_incorrect (a b : ℝ) (h : a > b) : (3 - a > 3 - b) -> false :=
by
  intros h1
  simp at h1
  sorry

end inequality_transformation_incorrect_l45_45507


namespace talia_total_distance_l45_45064

-- Definitions from the conditions
def distance_house_to_park : ℝ := 5
def distance_park_to_store : ℝ := 3
def distance_store_to_house : ℝ := 8

-- The theorem to be proven
theorem talia_total_distance : distance_house_to_park + distance_park_to_store + distance_store_to_house = 16 := by
  sorry

end talia_total_distance_l45_45064


namespace log_6_15_expression_l45_45447

theorem log_6_15_expression (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 3 = b) :
  Real.log 15 / Real.log 6 = (b + 1 - a) / (a + b) :=
sorry

end log_6_15_expression_l45_45447


namespace system1_solution_system2_solution_l45_45342

theorem system1_solution (x y : ℤ) : 
  (x - y = 3) ∧ (x = 3 * y - 1) → (x = 5) ∧ (y = 2) :=
by
  sorry

theorem system2_solution (x y : ℤ) : 
  (2 * x + 3 * y = -1) ∧ (3 * x - 2 * y = 18) → (x = 4) ∧ (y = -3) :=
by
  sorry

end system1_solution_system2_solution_l45_45342


namespace total_number_of_people_l45_45911

-- Conditions
def number_of_parents : ℕ := 105
def number_of_pupils : ℕ := 698

-- Theorem stating the total number of people is 803 given the conditions
theorem total_number_of_people : 
  number_of_parents + number_of_pupils = 803 :=
by
  sorry

end total_number_of_people_l45_45911


namespace people_counted_on_second_day_l45_45564

theorem people_counted_on_second_day (x : ℕ) (H1 : 2 * x + x = 1500) : x = 500 :=
by {
  sorry -- Proof goes here
}

end people_counted_on_second_day_l45_45564


namespace older_brother_is_14_l45_45151

theorem older_brother_is_14 {Y O : ℕ} (h1 : Y + O = 26) (h2 : O = Y + 2) : O = 14 :=
by
  sorry

end older_brother_is_14_l45_45151


namespace area_of_rectangle_EFGH_l45_45947

theorem area_of_rectangle_EFGH :
  ∀ (a b c : ℕ), 
    a = 7 → 
    b = 3 * a → 
    c = 2 * a → 
    (area : ℕ) = b * c → 
    area = 294 := 
by
  sorry

end area_of_rectangle_EFGH_l45_45947


namespace incorrect_expression_l45_45522

theorem incorrect_expression : 
  ∀ (x y : ℚ), (x / y = 2 / 5) → (x + 3 * y) / x ≠ 17 / 2 :=
by
  intros x y h
  sorry

end incorrect_expression_l45_45522


namespace sunzi_classic_l45_45439

noncomputable def length_of_rope : ℝ := sorry
noncomputable def length_of_wood : ℝ := sorry
axiom first_condition : length_of_rope - length_of_wood = 4.5
axiom second_condition : length_of_wood - (1 / 2) * length_of_rope = 1

theorem sunzi_classic : 
  (length_of_rope - length_of_wood = 4.5) ∧ (length_of_wood - (1 / 2) * length_of_rope = 1) := 
by 
  exact ⟨first_condition, second_condition⟩

end sunzi_classic_l45_45439


namespace Linda_needs_15_hours_to_cover_fees_l45_45686

def wage : ℝ := 10
def fee_per_college : ℝ := 25
def number_of_colleges : ℝ := 6

theorem Linda_needs_15_hours_to_cover_fees :
  (number_of_colleges * fee_per_college) / wage = 15 := by
  sorry

end Linda_needs_15_hours_to_cover_fees_l45_45686


namespace problem_statement_l45_45101

noncomputable def f (x : ℝ) (b c : ℝ) := x^2 + b * x + c

theorem problem_statement (b c : ℝ) (h : ∀ x : ℝ, f (x - 1) b c = f (3 - x) b c) : f 0 b c < f (-2) b c ∧ f (-2) b c < f 5 b c := 
by sorry

end problem_statement_l45_45101


namespace max_quotient_l45_45012

theorem max_quotient (a b : ℕ) (h₁ : 100 ≤ a) (h₂ : a ≤ 300) (h₃ : 1200 ≤ b) (h₄ : b ≤ 2400) :
  b / a ≤ 24 :=
sorry

end max_quotient_l45_45012


namespace ark5_ensures_metabolic_energy_l45_45585

-- Define conditions
def inhibits_ark5_activity (inhibits: Bool) (balance: Bool): Prop :=
  if inhibits then ¬balance else balance

def cancer_cells_proliferate_without_energy (proliferate: Bool) (die_due_to_insufficient_energy: Bool) : Prop :=
  proliferate → die_due_to_insufficient_energy

-- Define the hypothesis based on conditions
def hypothesis (inhibits: Bool) (balance: Bool) (proliferate: Bool) (die_due_to_insufficient_energy: Bool): Prop :=
  inhibits_ark5_activity inhibits balance ∧ cancer_cells_proliferate_without_energy proliferate die_due_to_insufficient_energy

-- Define the theorem to be proved
theorem ark5_ensures_metabolic_energy
  (inhibits : Bool)
  (balance : Bool)
  (proliferate : Bool)
  (die_due_to_insufficient_energy : Bool)
  (h : hypothesis inhibits balance proliferate die_due_to_insufficient_energy) :
  ensures_metabolic_energy :=
  sorry

end ark5_ensures_metabolic_energy_l45_45585


namespace arithmetic_geometric_sequence_a1_l45_45834

theorem arithmetic_geometric_sequence_a1 (a : ℕ → ℚ)
  (h1 : a 1 + a 6 = 11)
  (h2 : a 3 * a 4 = 32 / 9) :
  a 1 = 32 / 3 ∨ a 1 = 1 / 3 :=
sorry

end arithmetic_geometric_sequence_a1_l45_45834


namespace curve_not_parabola_l45_45934

theorem curve_not_parabola (k : ℝ) : ¬(∃ (a b c d e f : ℝ), k * x^2 + y^2 = a * x^2 + b * x * y + c * y^2 + d * x + e * y + f ∧ b^2 = 4*a*c ∧ (a = 0 ∨ c = 0)) := sorry

end curve_not_parabola_l45_45934


namespace point_inside_circle_l45_45759

theorem point_inside_circle (m : ℝ) : (1 - 2)^2 + (-3 + 1)^2 < m → m > 5 :=
by
  sorry

end point_inside_circle_l45_45759


namespace percent_problem_l45_45392

theorem percent_problem (x : ℝ) (hx : 0.60 * 600 = 0.50 * x) : x = 720 :=
by
  sorry

end percent_problem_l45_45392


namespace pamTotalApples_l45_45461

-- Define the given conditions
def applesPerGeraldBag : Nat := 40
def applesPerPamBag := 3 * applesPerGeraldBag
def pamBags : Nat := 10

-- Statement to prove
theorem pamTotalApples : pamBags * applesPerPamBag = 1200 :=
by
  sorry

end pamTotalApples_l45_45461


namespace highest_score_not_necessarily_at_least_12_l45_45084

section

-- Define the number of teams
def teams : ℕ := 12

-- Define the number of games each team plays
def games_per_team : ℕ := teams - 1

-- Define the total number of games
def total_games : ℕ := (teams * games_per_team) / 2

-- Define the points system
def points_for_win : ℕ := 2
def points_for_draw : ℕ := 1

-- Define the total points in the tournament
def total_points : ℕ := total_games * points_for_win

-- The highest score possible statement
def highest_score_must_be_at_least_12_statement : Prop :=
  ∀ (scores : Fin teams → ℕ), (∃ i, scores i ≥ 12)

-- Theorem stating that the statement "The highest score must be at least 12" is false
theorem highest_score_not_necessarily_at_least_12 (h : ∀ (scores : Fin teams → ℕ), (∃ i, scores i ≥ 12)) : False :=
  sorry

end

end highest_score_not_necessarily_at_least_12_l45_45084


namespace max_plus_min_value_of_y_eq_neg4_l45_45843

noncomputable def y (x : ℝ) : ℝ := (2 * (Real.sin x) ^ 2 + Real.sin (3 * x / 2) - 4) / ((Real.sin x) ^ 2 + 2 * (Real.cos x) ^ 2)

theorem max_plus_min_value_of_y_eq_neg4 (M m : ℝ) (hM : ∃ x : ℝ, y x = M) (hm : ∃ x : ℝ, y x = m) :
  M + m = -4 := sorry

end max_plus_min_value_of_y_eq_neg4_l45_45843


namespace value_of_a_for_positive_root_l45_45873

theorem value_of_a_for_positive_root :
  ∀ a : ℝ, (∃ x : ℝ, x > 0 ∧ (3*x - 1)/(x - 3) = a/(3 - x) - 1) → a = -8 :=
by
  sorry

end value_of_a_for_positive_root_l45_45873


namespace range_of_x_inequality_l45_45720

theorem range_of_x_inequality (x : ℝ) (h : |2 * x - 1| + x + 3 ≤ 5) : -1 ≤ x ∧ x ≤ 1 :=
by
  sorry

end range_of_x_inequality_l45_45720


namespace number_of_students_third_l45_45622

-- Define the ratio and the total number of samples.
def ratio_first : ℕ := 3
def ratio_second : ℕ := 3
def ratio_third : ℕ := 4
def total_sample : ℕ := 50

-- Define the condition that the sum of ratios equals the total proportion numerator.
def sum_ratios : ℕ := ratio_first + ratio_second + ratio_third

-- Final proposition: the number of students to be sampled from the third grade.
theorem number_of_students_third :
  (ratio_third * total_sample) / sum_ratios = 20 := by
  sorry

end number_of_students_third_l45_45622


namespace benny_gave_seashells_l45_45689

theorem benny_gave_seashells (original_seashells : ℕ) (remaining_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : original_seashells = 66) 
  (h2 : remaining_seashells = 14) 
  (h3 : original_seashells - remaining_seashells = given_seashells) : 
  given_seashells = 52 := 
by
  sorry

end benny_gave_seashells_l45_45689


namespace four_thirds_of_twelve_fifths_l45_45733

theorem four_thirds_of_twelve_fifths : (4 / 3) * (12 / 5) = 16 / 5 := 
by sorry

end four_thirds_of_twelve_fifths_l45_45733


namespace calculate_ff2_l45_45654

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x + 4

theorem calculate_ff2 : f (f 2) = 5450 := by
  sorry

end calculate_ff2_l45_45654


namespace mandy_yoga_time_l45_45300

-- Define the conditions
def ratio_swimming := 1
def ratio_running := 2
def ratio_gym := 3
def ratio_biking := 5
def ratio_yoga := 4

def time_biking := 30

-- Define the Lean 4 statement to prove
theorem mandy_yoga_time : (time_biking / ratio_biking) * ratio_yoga = 24 :=
by
  sorry

end mandy_yoga_time_l45_45300


namespace find_r_over_s_at_2_l45_45589

noncomputable def r (x : ℝ) := 6 * x
noncomputable def s (x : ℝ) := (x + 4) * (x - 1)

theorem find_r_over_s_at_2 :
  r 2 / s 2 = 2 :=
by
  -- The corresponding steps to show this theorem.
  sorry

end find_r_over_s_at_2_l45_45589


namespace symmetrical_circle_equation_l45_45761

theorem symmetrical_circle_equation :
  ∀ (x y : ℝ), (x^2 + y^2 - 2 * x - 1 = 0) ∧ (2 * x - y + 1 = 0) →
  ((x + 7/5)^2 + (y - 6/5)^2 = 2) :=
sorry

end symmetrical_circle_equation_l45_45761


namespace natural_numbers_solution_l45_45488

theorem natural_numbers_solution :
  ∃ (a b c d : ℕ), 
    ab = c + d ∧ a + b = cd ∧
    ((a, b, c, d) = (2, 2, 2, 2) ∨ (a, b, c, d) = (2, 3, 5, 1) ∨ 
     (a, b, c, d) = (3, 2, 5, 1) ∨ (a, b, c, d) = (2, 2, 1, 5) ∨ 
     (a, b, c, d) = (3, 2, 1, 5) ∨ (a, b, c, d) = (2, 3, 1, 5)) :=
by
  sorry

end natural_numbers_solution_l45_45488


namespace initial_coloring_books_l45_45533

theorem initial_coloring_books
  (x : ℝ)
  (h1 : x - 20 = 80 / 4) :
  x = 40 :=
by
  sorry

end initial_coloring_books_l45_45533


namespace sequence_general_term_l45_45049

theorem sequence_general_term (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3)
  (h3 : a 3 = 5)
  (h4 : a 4 = 7) :
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

end sequence_general_term_l45_45049


namespace highest_geometric_frequency_count_l45_45111

-- Define the problem conditions and the statement to be proved
theorem highest_geometric_frequency_count :
  ∀ (vol : ℕ) (num_groups : ℕ) (cum_freq_first_seven : ℝ)
  (remaining_freqs : List ℕ) (total_freq_remaining : ℕ)
  (r : ℕ) (a : ℕ),
  vol = 100 → 
  num_groups = 10 → 
  cum_freq_first_seven = 0.79 → 
  total_freq_remaining = 21 → 
  r > 1 →
  remaining_freqs = [a, a * r, a * r ^ 2] → 
  a * (1 + r + r ^ 2) = total_freq_remaining → 
  ∃ max_freq, max_freq ∈ remaining_freqs ∧ max_freq = 12 :=
by
  intro vol num_groups cum_freq_first_seven remaining_freqs total_freq_remaining r a
  intros h_vol h_num_groups h_cum_freq_first h_total_freq_remaining h_r_pos h_geom_seq h_freq_sum
  use 12
  sorry

end highest_geometric_frequency_count_l45_45111


namespace find_x_perpendicular_l45_45684

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 3)
def b (x : ℝ) : ℝ × ℝ := (-3, x)

-- Define the condition that the dot product of vectors a and b is zero
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- Statement we need to prove
theorem find_x_perpendicular (x : ℝ) (h : perpendicular a (b x)) : x = -1 :=
by sorry

end find_x_perpendicular_l45_45684


namespace garden_perimeter_l45_45394

-- Definitions for length and breadth
def length := 150
def breadth := 100

-- Theorem that states the perimeter of the rectangular garden
theorem garden_perimeter : (2 * (length + breadth)) = 500 :=
by sorry

end garden_perimeter_l45_45394


namespace discrim_of_quad_l45_45781

-- Definition of the quadratic equation coefficients
def a : ℤ := 5
def b : ℤ := -9
def c : ℤ := 4

-- Definition of the discriminant formula which needs to be proved as 1
def discriminant (a b c : ℤ) : ℤ :=
  b^2 - 4 * a * c

-- The proof problem statement
theorem discrim_of_quad : discriminant a b c = 1 := by
  sorry

end discrim_of_quad_l45_45781


namespace base_six_equals_base_b_l45_45591

theorem base_six_equals_base_b (b : ℕ) (h1 : 3 * 6 ^ 1 + 4 * 6 ^ 0 = 22)
  (h2 : b ^ 2 + 2 * b + 1 = 22) : b = 3 :=
sorry

end base_six_equals_base_b_l45_45591


namespace complex_norm_wz_l45_45691

open Complex

theorem complex_norm_wz (w z : ℂ) (h₁ : ‖w + z‖ = 2) (h₂ : ‖w^2 + z^2‖ = 8) : 
  ‖w^4 + z^4‖ = 56 := 
  sorry

end complex_norm_wz_l45_45691


namespace min_value_of_quadratic_expression_l45_45928

theorem min_value_of_quadratic_expression : ∃ x : ℝ, (∀ y : ℝ, x^2 + 6 * x + 3 ≤ y) ∧ x^2 + 6 * x + 3 = -6 :=
sorry

end min_value_of_quadratic_expression_l45_45928


namespace third_number_in_list_l45_45076

theorem third_number_in_list :
  let nums : List ℕ := [201, 202, 205, 206, 209, 209, 210, 212, 212]
  nums.nthLe 2 (by simp [List.length]) = 205 :=
sorry

end third_number_in_list_l45_45076


namespace selection_methods_including_both_boys_and_girls_l45_45571

def boys : ℕ := 4
def girls : ℕ := 3
def total_people : ℕ := boys + girls
def select : ℕ := 4

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_methods_including_both_boys_and_girls :
  combination 7 4 - combination boys 4 = 34 :=
by
  sorry

end selection_methods_including_both_boys_and_girls_l45_45571


namespace pipe_length_difference_l45_45811

theorem pipe_length_difference (total_length shorter_piece : ℕ) (h1 : total_length = 68) (h2 : shorter_piece = 28) : 
  total_length - shorter_piece * 2 = 12 := 
sorry

end pipe_length_difference_l45_45811


namespace max_min_cos_sin_product_l45_45952

theorem max_min_cos_sin_product (x y z : ℝ) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π / 12) (h4 : x + y + z = π / 2) :
  ∃ (maximum minimum : ℝ), maximum = (2 + Real.sqrt 3) / 8 ∧ minimum = 1 / 8 := by
  sorry

end max_min_cos_sin_product_l45_45952


namespace baseball_team_wins_more_than_three_times_losses_l45_45097

theorem baseball_team_wins_more_than_three_times_losses
    (total_games : ℕ)
    (wins : ℕ)
    (losses : ℕ)
    (h1 : total_games = 130)
    (h2 : wins = 101)
    (h3 : wins + losses = total_games) :
    wins - 3 * losses = 14 :=
by
    -- Proof goes here
    sorry

end baseball_team_wins_more_than_three_times_losses_l45_45097


namespace eight_sided_dice_theorem_l45_45574
open Nat

noncomputable def eight_sided_dice_probability : ℚ :=
  let total_outcomes := 8^8
  let favorable_outcomes := 8!
  let probability_all_different := favorable_outcomes / total_outcomes
  let probability_at_least_two_same := 1 - probability_all_different
  probability_at_least_two_same

theorem eight_sided_dice_theorem :
  eight_sided_dice_probability = 16736996 / 16777216 := by
    sorry

end eight_sided_dice_theorem_l45_45574


namespace regular_pentagonal_prism_diagonal_count_l45_45027

noncomputable def diagonal_count (n : ℕ) : ℕ := 
  if n = 5 then 10 else 0

theorem regular_pentagonal_prism_diagonal_count :
  diagonal_count 5 = 10 := 
  by
    sorry

end regular_pentagonal_prism_diagonal_count_l45_45027


namespace smallest_odd_m_satisfying_inequality_l45_45029

theorem smallest_odd_m_satisfying_inequality : ∃ m : ℤ, m^2 - 11 * m + 24 ≥ 0 ∧ (m % 2 = 1) ∧ ∀ n : ℤ, n^2 - 11 * n + 24 ≥ 0 ∧ (n % 2 = 1) → m ≤ n → m = 3 :=
by
  sorry

end smallest_odd_m_satisfying_inequality_l45_45029


namespace algebraic_expression_value_l45_45165

variables (x y : ℝ)

theorem algebraic_expression_value :
  x^2 - 4 * x - 1 = 0 →
  (2 * x - 3)^2 - (x + y) * (x - y) - y^2 = 12 :=
by
  intro h
  sorry

end algebraic_expression_value_l45_45165


namespace remainder_of_2n_div_9_l45_45900

theorem remainder_of_2n_div_9
  (n : ℤ) (h : ∃ k : ℤ, n = 18 * k + 10) : (2 * n) % 9 = 2 := 
by
  sorry

end remainder_of_2n_div_9_l45_45900


namespace prism_diagonal_length_l45_45916

theorem prism_diagonal_length (x y z : ℝ) (h1 : 4 * x + 4 * y + 4 * z = 24) (h2 : 2 * x * y + 2 * x * z + 2 * y * z = 11) : Real.sqrt (x^2 + y^2 + z^2) = 5 :=
  by
  sorry

end prism_diagonal_length_l45_45916


namespace boat_distance_along_stream_l45_45837

theorem boat_distance_along_stream
  (distance_against_stream : ℝ)
  (speed_still_water : ℝ)
  (time : ℝ)
  (v_s : ℝ)
  (H1 : distance_against_stream = 5)
  (H2 : speed_still_water = 6)
  (H3 : time = 1)
  (H4 : speed_still_water - v_s = distance_against_stream / time) :
  (speed_still_water + v_s) * time = 7 :=
by
  -- Sorry to skip proof
  sorry

end boat_distance_along_stream_l45_45837


namespace total_marbles_l45_45751

variable (r b g : ℝ)
variable (h1 : r = 1.3 * b)
variable (h2 : g = 1.7 * r)

theorem total_marbles (r b g : ℝ) (h1 : r = 1.3 * b) (h2 : g = 1.7 * r) :
  r + b + g = 3.469 * r :=
by
  sorry

end total_marbles_l45_45751


namespace hayley_initial_meatballs_l45_45078

theorem hayley_initial_meatballs (x : ℕ) (stolen : ℕ) (left : ℕ) (h1 : stolen = 14) (h2 : left = 11) (h3 : x - stolen = left) : x = 25 := 
by 
  sorry

end hayley_initial_meatballs_l45_45078


namespace max_min_values_l45_45000

def f (x a : ℝ) : ℝ := -x^2 + 2*x + a

theorem max_min_values (a : ℝ) (h : a ≠ 0) :
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≤ 1 + a) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ f x a = 1 + a) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 3 → -3 + a ≤ f x a) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ f x a = -3 + a) := 
sorry

end max_min_values_l45_45000


namespace calc_expression_l45_45276

theorem calc_expression :
  (-(1 / 2))⁻¹ - 4 * Real.cos (Real.pi / 6) - (Real.pi + 2013)^0 + Real.sqrt 12 = -3 :=
by
  sorry

end calc_expression_l45_45276


namespace part1_part2_l45_45346

noncomputable def f (x : ℝ) : ℝ :=
  abs (2 * x - 3) + abs (x - 5)

theorem part1 : { x : ℝ | f x ≥ 4 } = { x : ℝ | x ≥ 2 ∨ x ≤ 4 / 3 } :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x : ℝ, f x < a) ↔ a > 7 / 2 :=
by
  sorry

end part1_part2_l45_45346


namespace sufficient_but_not_necessary_condition_l45_45914

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x^2 - 1 = 0 → x^3 - x = 0) ∧ ¬ (x^3 - x = 0 → x^2 - 1 = 0) := by
  sorry

end sufficient_but_not_necessary_condition_l45_45914


namespace average_episodes_per_year_l45_45634

theorem average_episodes_per_year (total_years : ℕ) (n1 n2 n3 e1 e2 e3 : ℕ) 
  (h1 : total_years = 14)
  (h2 : n1 = 8) (h3 : e1 = 15)
  (h4 : n2 = 4) (h5 : e2 = 20)
  (h6 : n3 = 2) (h7 : e3 = 12) :
  (n1 * e1 + n2 * e2 + n3 * e3) / total_years = 16 := by
  -- Skip the proof steps
  sorry

end average_episodes_per_year_l45_45634


namespace find_x_l45_45731

-- Definitions based on provided conditions

def rectangle_length (x : ℝ) : ℝ := 4 * x
def rectangle_width (x : ℝ) : ℝ := x + 7
def rectangle_area (x : ℝ) : ℝ := rectangle_length x * rectangle_width x
def rectangle_perimeter (x : ℝ) : ℝ := 2 * rectangle_length x + 2 * rectangle_width x

-- Theorem statement
theorem find_x (x : ℝ) (h : rectangle_area x = 2 * rectangle_perimeter x) : x = 1 := 
sorry

end find_x_l45_45731


namespace february_sales_increase_l45_45395

theorem february_sales_increase (Slast : ℝ) (r : ℝ) (Sthis : ℝ) 
  (h_last_year_sales : Slast = 320) 
  (h_percent_increase : r = 0.25) : 
  Sthis = 400 :=
by
  have h1 : Sthis = Slast * (1 + r) := sorry
  sorry

end february_sales_increase_l45_45395


namespace time_both_pipes_opened_l45_45920

def fill_rate_p := 1 / 10
def fill_rate_q := 1 / 15
def total_fill_rate := fill_rate_p + fill_rate_q -- Combined fill rate of both pipes

def remaining_fill_rate := 10 * fill_rate_q -- Fill rate of pipe q in 10 minutes

theorem time_both_pipes_opened (t : ℝ) :
  (t / 6) + (2 / 3) = 1 → t = 2 :=
by
  sorry

end time_both_pipes_opened_l45_45920


namespace find_constants_PQR_l45_45946

theorem find_constants_PQR :
  ∃ P Q R : ℝ, 
    (6 * x + 2) / ((x - 4) * (x - 2) ^ 3) = P / (x - 4) + Q / (x - 2) + R / (x - 2) ^ 3 :=
by
  use 13 / 4
  use -6.5
  use -7
  sorry

end find_constants_PQR_l45_45946


namespace average_speed_l45_45990

/--
On the first day of her vacation, Louisa traveled 100 miles.
On the second day, traveling at the same average speed, she traveled 175 miles.
If the 100-mile trip took 3 hours less than the 175-mile trip,
prove that her average speed (in miles per hour) was 25.
-/
theorem average_speed (v : ℝ) (h1 : 100 / v + 3 = 175 / v) : v = 25 :=
by 
  sorry

end average_speed_l45_45990


namespace average_of_w_and_x_is_one_half_l45_45767

noncomputable def average_of_w_and_x (w x y : ℝ) : ℝ :=
  (w + x) / 2

theorem average_of_w_and_x_is_one_half (w x y : ℝ)
  (h1 : 2 / w + 2 / x = 2 / y)
  (h2 : w * x = y) : average_of_w_and_x w x y = 1 / 2 :=
by
  sorry

end average_of_w_and_x_is_one_half_l45_45767


namespace find_x_values_l45_45181

theorem find_x_values (x : ℝ) (h : x ≠ 5) : x + 36 / (x - 5) = -12 ↔ x = -8 ∨ x = 3 :=
by sorry

end find_x_values_l45_45181


namespace count_scalene_triangles_under_16_l45_45056

theorem count_scalene_triangles_under_16 : 
  ∃ (n : ℕ), n = 6 ∧ ∀ (a b c : ℕ), 
  a < b ∧ b < c ∧ a + b + c < 16 ∧ a + b > c ∧ a + c > b ∧ b + c > a ↔ 
  (a, b, c) ∈ [(2, 3, 4), (2, 4, 5), (2, 5, 6), (3, 4, 5), (3, 5, 6), (4, 5, 6)] :=
by sorry

end count_scalene_triangles_under_16_l45_45056


namespace time_fraction_l45_45536

variable (t₅ t₁₅ : ℝ)

def total_distance (t₅ t₁₅ : ℝ) : ℝ :=
  5 * t₅ + 15 * t₁₅

def total_time (t₅ t₁₅ : ℝ) : ℝ :=
  t₅ + t₁₅

def average_speed_eq (t₅ t₁₅ : ℝ) : Prop :=
  10 * (t₅ + t₁₅) = 5 * t₅ + 15 * t₁₅

theorem time_fraction (t₅ t₁₅ : ℝ) (h : average_speed_eq t₅ t₁₅) :
  (t₁₅ / (t₅ + t₁₅)) = 1 / 2 := by
  sorry

end time_fraction_l45_45536


namespace aubree_animals_total_l45_45743

theorem aubree_animals_total (b_go c_go b_return c_return : ℕ) 
    (h1 : b_go = 20) (h2 : c_go = 40) 
    (h3 : b_return = b_go * 2) 
    (h4 : c_return = c_go - 10) : 
    b_go + c_go + b_return + c_return = 130 := by 
  sorry

end aubree_animals_total_l45_45743


namespace points_lie_on_circle_l45_45485

theorem points_lie_on_circle (s : ℝ) :
  ( (2 - s^2) / (2 + s^2) )^2 + ( 3 * s / (2 + s^2) )^2 = 1 :=
by sorry

end points_lie_on_circle_l45_45485


namespace sum_of_exponents_l45_45763

theorem sum_of_exponents : 
  (-1)^(2010) + (-1)^(2013) + 1^(2014) + (-1)^(2016) = 0 := 
by
  sorry

end sum_of_exponents_l45_45763


namespace triangle_exists_among_single_color_sticks_l45_45474

theorem triangle_exists_among_single_color_sticks
  (red yellow green : ℕ)
  (k y g K Y G : ℕ)
  (hk : k + y > G)
  (hy : y + g > K)
  (hg : g + k > Y)
  (hred : red = 100)
  (hyellow : yellow = 100)
  (hgreen : green = 100) :
  ∃ color : string, ∀ a b c : ℕ, (a = k ∨ a = K) → (b = k ∨ b = K) → (c = k ∨ c = K) → a + b > c :=
sorry

end triangle_exists_among_single_color_sticks_l45_45474


namespace product_not_divisible_by_prime_l45_45848

theorem product_not_divisible_by_prime (p a b : ℕ) (hp : Prime p) (ha : 1 ≤ a) (hpa : a < p) (hb : 1 ≤ b) (hpb : b < p) : ¬ (p ∣ (a * b)) :=
by
  sorry

end product_not_divisible_by_prime_l45_45848


namespace smallest_n_modulo_l45_45004

theorem smallest_n_modulo :
  ∃ n : ℕ, 0 < n ∧ 5 * n % 26 = 1846 % 26 ∧ n = 26 :=
by
  sorry

end smallest_n_modulo_l45_45004


namespace find_min_value_l45_45316

variable (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1)

theorem find_min_value :
  (1 / (2 * a + 3 * b) + 1 / (2 * b + 3 * c) + 1 / (2 * c + 3 * a)) ≥ (9 / 5) :=
sorry

end find_min_value_l45_45316


namespace find_number_l45_45939

theorem find_number (x : ℝ) (h : 5020 - (1004 / x) = 4970) : x = 20.08 := 
by
  sorry

end find_number_l45_45939


namespace percentage_brand_A_l45_45014

theorem percentage_brand_A
  (A B : ℝ)
  (h1 : 0.6 * A + 0.65 * B = 0.5 * (A + B))
  : (A / (A + B)) * 100 = 60 :=
by
  sorry

end percentage_brand_A_l45_45014


namespace possible_values_of_m_plus_n_l45_45614

theorem possible_values_of_m_plus_n (m n : ℕ) (hmn_pos : 0 < m ∧ 0 < n) 
  (cond : Nat.lcm m n - Nat.gcd m n = 103) : m + n = 21 ∨ m + n = 105 ∨ m + n = 309 := by
  sorry

end possible_values_of_m_plus_n_l45_45614


namespace intersection_is_one_l45_45232

def M : Set ℝ := {x | x - 1 = 0}
def N : Set ℝ := {x | x^2 - 3 * x + 2 = 0}

theorem intersection_is_one : M ∩ N = {1} :=
by
  sorry

end intersection_is_one_l45_45232


namespace value_of_expression_l45_45794

theorem value_of_expression : 2 - (-2 : ℝ) ^ (-2 : ℝ) = 7 / 4 := 
by 
  sorry

end value_of_expression_l45_45794


namespace values_of_k_real_equal_roots_l45_45830

theorem values_of_k_real_equal_roots (k : ℝ) : 
  (∃ k, (3 - 2 * k)^2 - 4 * 3 * 12 = 0 ∧ (k = -9 / 2 ∨ k = 15 / 2)) :=
by
  sorry

end values_of_k_real_equal_roots_l45_45830


namespace solve_proof_problem_l45_45096

noncomputable def proof_problem : Prop :=
  let short_videos_per_day := 2
  let short_video_time := 2
  let longer_videos_per_day := 1
  let week_days := 7
  let total_weekly_video_time := 112
  let total_short_video_time_per_week := short_videos_per_day * short_video_time * week_days
  let total_longer_video_time_per_week := total_weekly_video_time - total_short_video_time_per_week
  let longer_video_multiple := total_longer_video_time_per_week / short_video_time
  longer_video_multiple = 42

theorem solve_proof_problem : proof_problem :=
by
  /- Proof goes here -/
  sorry

end solve_proof_problem_l45_45096


namespace number_of_throwers_l45_45639

theorem number_of_throwers (T N : ℕ) :
  (T + N = 61) ∧ ((2 * N) / 3 = 53 - T) → T = 37 :=
by 
  sorry

end number_of_throwers_l45_45639


namespace french_students_l45_45525

theorem french_students 
  (T : ℕ) (G : ℕ) (B : ℕ) (N : ℕ) (F : ℕ)
  (hT : T = 78) (hG : G = 22) (hB : B = 9) (hN : N = 24)
  (h_eq : F + G - B = T - N) :
  F = 41 :=
by
  sorry

end french_students_l45_45525


namespace tailor_time_l45_45442

theorem tailor_time (x : ℝ) 
  (t_shirt : ℝ := x) 
  (t_pants : ℝ := 2 * x) 
  (t_jacket : ℝ := 3 * x) 
  (h_capacity : 2 * t_shirt + 3 * t_pants + 4 * t_jacket = 10) : 
  14 * t_shirt + 10 * t_pants + 2 * t_jacket = 20 :=
by
  sorry

end tailor_time_l45_45442


namespace johns_drive_distance_l45_45072

/-- John's driving problem -/
theorem johns_drive_distance
  (d t : ℝ)
  (h1 : d = 25 * (t + 1.5))
  (h2 : d = 25 + 45 * (t - 1.25)) :
  d = 123.4375 := 
sorry

end johns_drive_distance_l45_45072


namespace rectangle_width_l45_45608

theorem rectangle_width
  (l w : ℕ)
  (h1 : l * w = 1638)
  (h2 : 10 * l = 390) :
  w = 42 :=
by
  sorry

end rectangle_width_l45_45608


namespace convex_pentagon_probability_l45_45214

-- Defining the number of chords and the probability calculation as per the problem's conditions
def number_of_chords (n : ℕ) : ℕ := (n * (n - 1)) / 2
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem conditions
def eight_points_on_circle : ℕ := 8
def chords_chosen : ℕ := 5

-- Total number of chords from eight points
def total_chords : ℕ := number_of_chords eight_points_on_circle

-- The probability calculation
def probability_convex_pentagon :=
  binom 8 5 / binom total_chords chords_chosen

-- Statement to be proven
theorem convex_pentagon_probability :
  probability_convex_pentagon = 1 / 1755 := sorry

end convex_pentagon_probability_l45_45214


namespace sin_arcsin_plus_arctan_l45_45016

theorem sin_arcsin_plus_arctan :
  let a := Real.arcsin (4/5)
  let b := Real.arctan 1
  Real.sin (a + b) = (7 * Real.sqrt 2) / 10 := by
  sorry

end sin_arcsin_plus_arctan_l45_45016


namespace N_is_necessary_but_not_sufficient_l45_45008

-- Define sets M and N
def M := { x : ℝ | 0 < x ∧ x < 1 }
def N := { x : ℝ | -2 < x ∧ x < 1 }

-- State the theorem to prove that "a belongs to N" is necessary but not sufficient for "a belongs to M"
theorem N_is_necessary_but_not_sufficient (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (a ∈ N → a ∈ M → False) :=
by sorry

end N_is_necessary_but_not_sufficient_l45_45008


namespace age_difference_l45_45919

-- Define the present age of the son.
def S : ℕ := 22

-- Define the present age of the man.
variable (M : ℕ)

-- Given condition: In two years, the man's age will be twice the age of his son.
axiom condition : M + 2 = 2 * (S + 2)

-- Prove that the difference in present ages of the man and his son is 24 years.
theorem age_difference : M - S = 24 :=
by 
  -- We will fill in the proof here
  sorry

end age_difference_l45_45919


namespace ab_plus_cd_is_composite_l45_45025

theorem ab_plus_cd_is_composite 
  (a b c d : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_order : a > b ∧ b > c ∧ c > d)
  (h_eq : a^2 + a * c - c^2 = b^2 + b * d - d^2) : 
  ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ ab + cd = p * q :=
by
  sorry

end ab_plus_cd_is_composite_l45_45025


namespace weight_7_moles_AlI3_l45_45553

-- Definitions from the conditions
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_I : ℝ := 126.90
def molecular_weight_AlI3 : ℝ := atomic_weight_Al + 3 * atomic_weight_I
def weight_of_compound (moles : ℝ) (molecular_weight : ℝ) : ℝ := moles * molecular_weight

-- Theorem stating the weight of 7 moles of AlI3
theorem weight_7_moles_AlI3 : 
  weight_of_compound 7 molecular_weight_AlI3 = 2853.76 :=
by
  -- Proof will be added here
  sorry

end weight_7_moles_AlI3_l45_45553


namespace S7_is_28_l45_45380

variables {a_n : ℕ → ℤ} -- Sequence definition
variables {S_n : ℕ → ℤ} -- Sum of the first n terms

-- Define an arithmetic sequence condition
def is_arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

-- Given conditions
axiom sum_condition : a_n 2 + a_n 4 + a_n 6 = 12
axiom sum_formula (n : ℕ) : S_n n = n * (a_n 1 + a_n n) / 2
axiom arith_seq : is_arithmetic_sequence a_n

-- The statement to be proven
theorem S7_is_28 : S_n 7 = 28 :=
sorry

end S7_is_28_l45_45380


namespace average_age_correct_l45_45066

def ratio (m w : ℕ) : Prop := w * 8 = m * 9

def average_age_of_group (m w : ℕ) (avg_men avg_women : ℕ) : ℚ :=
  (avg_men * m + avg_women * w) / (m + w)

/-- The average age of the group is 32 14/17 given that the ratio of the number of women to the number of men is 9 to 8, 
    the average age of the women is 30 years, and the average age of the men is 36 years. -/
theorem average_age_correct
  (m w : ℕ)
  (h_ratio : ratio m w)
  (h_avg_women : avg_age_women = 30)
  (h_avg_men : avg_age_men = 36) :
  average_age_of_group m w avg_age_men avg_age_women = 32 + (14 / 17) := 
by
  sorry

end average_age_correct_l45_45066


namespace age_difference_is_100_l45_45619

-- Definition of the ages
variables {X Y Z : ℕ}

-- Conditions from the problem statement
axiom age_condition1 : X + Y > Y + Z
axiom age_condition2 : Z = X - 100

-- Proof to show the difference is 100 years
theorem age_difference_is_100 : (X + Y) - (Y + Z) = 100 :=
by sorry

end age_difference_is_100_l45_45619


namespace lucy_money_l45_45878

variable (L : ℕ) -- Value for Lucy's original amount of money

theorem lucy_money (h1 : ∀ (L : ℕ), L - 5 = 10 + 5 → L = 20) : L = 20 :=
by sorry

end lucy_money_l45_45878


namespace meaningful_fraction_l45_45783

theorem meaningful_fraction (x : ℝ) : (x ≠ -2) ↔ (∃ y : ℝ, y = 1 / (x + 2)) :=
by sorry

end meaningful_fraction_l45_45783


namespace mph_to_fps_l45_45184

theorem mph_to_fps (C G : ℝ) (x : ℝ) (hC : C = 60 * x) (hG : G = 40 * x) (h1 : 7 * C - 7 * G = 210) :
  x = 1.5 :=
by {
  -- Math proof here, but we insert sorry for now
  sorry
}

end mph_to_fps_l45_45184


namespace train_length_is_correct_l45_45529

variable (speed_km_hr : ℕ) (time_sec : ℕ)
def convert_speed (speed_km_hr : ℕ) : ℚ :=
  (speed_km_hr * 1000 : ℚ) / 3600

noncomputable def length_of_train (speed_km_hr time_sec : ℕ) : ℚ :=
  convert_speed speed_km_hr * time_sec

theorem train_length_is_correct (speed_km_hr : ℕ) (time_sec : ℕ) (h₁ : speed_km_hr = 300) (h₂ : time_sec = 33) :
  length_of_train speed_km_hr time_sec = 2750 := by
  sorry

end train_length_is_correct_l45_45529


namespace inequality_proof_l45_45966

variable {x y z : ℝ}

theorem inequality_proof 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hxyz : x + y + z = 1) : 
  (1 / x^2 + x) * (1 / y^2 + y) * (1 / z^2 + z) ≥ (28 / 3)^3 :=
sorry

end inequality_proof_l45_45966


namespace calculate_expression_l45_45249

theorem calculate_expression : 6^3 - 5 * 7 + 2^4 = 197 := 
by
  -- Generally, we would provide the proof here, but it's not required.
  sorry

end calculate_expression_l45_45249


namespace max_log_expression_l45_45943

noncomputable def log_base (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem max_log_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : x > y) :
  log_base x (x^2 / y^3) + log_base y (y^2 / x^3) = -2 :=
by
  sorry

end max_log_expression_l45_45943


namespace smallest_positive_period_minimum_value_of_f_center_of_symmetry_interval_of_increasing_l45_45667

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x - Real.pi / 6)

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

theorem minimum_value_of_f :
  ∃ x, f x = -3 :=
sorry

theorem center_of_symmetry (k : ℤ) :
  ∃ p, (∀ x, f (p + x) = f (p - x)) ∧ p = (Real.pi / 12) + (k * Real.pi / 2) :=
sorry

theorem interval_of_increasing (k : ℤ) :
  ∃ a b, a = -(Real.pi / 6) + k * Real.pi ∧ b = (Real.pi / 3) + k * Real.pi ∧
  ∀ x, (a <= x ∧ x <= b) → StrictMonoOn f (Set.Icc a b) :=
sorry

end smallest_positive_period_minimum_value_of_f_center_of_symmetry_interval_of_increasing_l45_45667


namespace sally_and_mary_picked_16_lemons_l45_45122

theorem sally_and_mary_picked_16_lemons (sally_lemons mary_lemons : ℕ) (sally_picked : sally_lemons = 7) (mary_picked : mary_lemons = 9) :
  sally_lemons + mary_lemons = 16 :=
by {
  sorry
}

end sally_and_mary_picked_16_lemons_l45_45122


namespace solve_equation_l45_45087

noncomputable def solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, x = Real.arcsin (3/4) + 2 * k * Real.pi ∨ x = Real.pi - Real.arcsin (3/4) + 2 * k * Real.pi

theorem solve_equation (x : ℝ) :
  (5 * Real.sin x = 4 + 2 * Real.cos (2 * x)) ↔ solution_set x := 
sorry

end solve_equation_l45_45087


namespace lice_checks_l45_45365

theorem lice_checks (t_first t_second t_third t_total t_per_check n_first n_second n_third n_total n_per_check n_kg : ℕ) 
 (h1 : t_first = 19 * t_per_check)
 (h2 : t_second = 20 * t_per_check)
 (h3 : t_third = 25 * t_per_check)
 (h4 : t_total = 3 * 60)
 (h5 : t_per_check = 2)
 (h6 : n_first = t_first / t_per_check)
 (h7 : n_second = t_second / t_per_check)
 (h8 : n_third = t_third / t_per_check)
 (h9 : n_total = (t_total - (t_first + t_second + t_third)) / t_per_check) :
 n_total = 26 :=
sorry

end lice_checks_l45_45365


namespace vance_family_stamp_cost_difference_l45_45143

theorem vance_family_stamp_cost_difference :
    let cost_rooster := 2 * 1.50
    let cost_daffodil := 5 * 0.75
    cost_daffodil - cost_rooster = 0.75 :=
by
    let cost_rooster := 2 * 1.50
    let cost_daffodil := 5 * 0.75
    show cost_daffodil - cost_rooster = 0.75
    sorry

end vance_family_stamp_cost_difference_l45_45143


namespace kevin_age_l45_45257

theorem kevin_age (x : ℕ) :
  (∃ n : ℕ, x - 2 = n^2) ∧ (∃ m : ℕ, x + 2 = m^3) → x = 6 :=
by
  sorry

end kevin_age_l45_45257


namespace inequality_solutions_l45_45741

theorem inequality_solutions (y : ℝ) :
  (2 / (y + 2) + 4 / (y + 8) ≥ 1 ↔ (y > -8 ∧ y ≤ -4) ∨ (y ≥ -2 ∧ y ≤ 2)) :=
by
  sorry

end inequality_solutions_l45_45741


namespace average_disk_space_per_hour_l45_45950

theorem average_disk_space_per_hour :
  let days : ℕ := 15
  let total_mb : ℕ := 20000
  let hours_per_day : ℕ := 24
  let total_hours := days * hours_per_day
  total_mb / total_hours = 56 :=
by
  let days := 15
  let total_mb := 20000
  let hours_per_day := 24
  let total_hours := days * hours_per_day
  have h : total_mb / total_hours = 56 := sorry
  exact h

end average_disk_space_per_hour_l45_45950


namespace complement_of_M_is_34_l45_45288

open Set

noncomputable def U : Set ℝ := univ
def M : Set ℝ := {x | (x - 3) / (4 - x) < 0}
def complement_M (U : Set ℝ) (M : Set ℝ) : Set ℝ := U \ M

theorem complement_of_M_is_34 : complement_M U M = {x | 3 ≤ x ∧ x ≤ 4} := 
by sorry

end complement_of_M_is_34_l45_45288


namespace diagonal_splits_odd_vertices_l45_45054

theorem diagonal_splits_odd_vertices (n : ℕ) (H : n^2 ≤ (2 * n + 2) * (2 * n + 1) / 2) :
  ∃ (x y : ℕ), x < y ∧ x ≤ 2 * n + 1 ∧ y ≤ 2 * n + 2 ∧ (y - x) % 2 = 0 :=
sorry

end diagonal_splits_odd_vertices_l45_45054


namespace adam_and_simon_50_miles_apart_l45_45578

noncomputable def time_when_50_miles_apart (x : ℝ) : Prop :=
  let adam_distance := 10 * x
  let simon_distance := 8 * x
  (adam_distance^2 + simon_distance^2 = 50^2) 

theorem adam_and_simon_50_miles_apart : 
  ∃ x : ℝ, time_when_50_miles_apart x ∧ x = 50 / 12.8 := 
sorry

end adam_and_simon_50_miles_apart_l45_45578


namespace transformation_l45_45784

noncomputable def Q (a b c x y : ℝ) : ℝ := a * x^2 + 2 * b * x * y + c * y^2

theorem transformation 
  (a b c d e f x y x₀ y₀ x' y' : ℝ)
  (h : a * c - b^2 ≠ 0)
  (hQ : Q a b c x y + 2 * d * x + 2 * e * y = f)
  (hx : x' = x + x₀)
  (hy : y' = y + y₀) :
  ∃ f' : ℝ, (a * x'^2 + 2 * b * x' * y' + c * y'^2 = f' ∧ 
             f' = f - Q a b c x₀ y₀ + 2 * (d * x₀ + e * y₀)) :=
sorry

end transformation_l45_45784


namespace work_fraction_left_l45_45372

theorem work_fraction_left (A_days B_days : ℕ) (work_days : ℕ)
  (hA : A_days = 15) (hB : B_days = 20) (h_work : work_days = 3) :
  1 - (work_days * ((1 / A_days) + (1 / B_days))) = 13 / 20 :=
by
  rw [hA, hB, h_work]
  simp
  sorry

end work_fraction_left_l45_45372


namespace symmetric_polynomial_evaluation_l45_45984

theorem symmetric_polynomial_evaluation :
  ∃ (a b : ℝ), (∀ x : ℝ, (x^2 + 3 * x) * (x^2 + a * x + b) = ((2 - x)^2 + 3 * (2 - x)) * ((2 - x)^2 + a * (2 - x) + b)) ∧
  ((3^2 + 3 * 3) * (3^2 + (-6) * 3 + 8) = -18) :=
sorry

end symmetric_polynomial_evaluation_l45_45984


namespace exist_line_l1_exist_line_l2_l45_45006

noncomputable def P : ℝ × ℝ := ⟨3, 2⟩

def line1_eq (x y : ℝ) : Prop := 2 * x - y - 4 = 0
def line2_eq (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def perpend_line_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 15 = 0

def line_l1 (x y : ℝ) : Prop := 4 * x - 3 * y - 6 = 0
def line_l2_case1 (x y : ℝ) : Prop := 2 * x - 3 * y = 0
def line_l2_case2 (x y : ℝ) : Prop := x + y - 5 = 0

theorem exist_line_l1 : ∃ (x y : ℝ), line1_eq x y ∧ line2_eq x y ∧ perpend_line_eq x y → line_l1 x y :=
by
  sorry

theorem exist_line_l2 : ∃ (x y : ℝ), line1_eq x y ∧ line2_eq x y ∧ ((line_l2_case1 x y) ∨ (line_l2_case2 x y)) :=
by
  sorry

end exist_line_l1_exist_line_l2_l45_45006


namespace complement_set_U_A_l45_45679

-- Definitions of U and A
def U : Set ℝ := { x : ℝ | x^2 ≤ 4 }
def A : Set ℝ := { x : ℝ | |x - 1| ≤ 1 }

-- Theorem statement
theorem complement_set_U_A : (U \ A) = { x : ℝ | -2 ≤ x ∧ x < 0 } := 
by
  sorry

end complement_set_U_A_l45_45679


namespace supplement_of_complement_65_degrees_l45_45780

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_65_degrees : 
  supplement (complement 65) = 155 :=
by
  sorry

end supplement_of_complement_65_degrees_l45_45780


namespace men_handshakes_l45_45931

theorem men_handshakes (n : ℕ) (h : n * (n - 1) / 2 = 435) : n = 30 :=
sorry

end men_handshakes_l45_45931


namespace total_area_of_colored_paper_l45_45527

-- Definitions
def num_pieces : ℝ := 3.2
def side_length : ℝ := 8.5

-- Theorem statement
theorem total_area_of_colored_paper : 
  let area_one_piece := side_length * side_length
  let total_area := area_one_piece * num_pieces
  total_area = 231.2 := by
  sorry

end total_area_of_colored_paper_l45_45527


namespace equal_cost_per_copy_l45_45633

theorem equal_cost_per_copy 
    (x : ℕ) 
    (h₁ : 2000 % x = 0) 
    (h₂ : 3000 % (x + 50) = 0) 
    (h₃ : 2000 / x = 3000 / (x + 50)) :
    (2000 : ℕ) / x = (3000 : ℕ) / (x + 50) :=
by
  sorry

end equal_cost_per_copy_l45_45633


namespace son_present_age_l45_45727

variable (S F : ℕ)

-- Define the conditions
def fatherAgeCondition := F = S + 35
def twoYearsCondition := F + 2 = 2 * (S + 2)

-- The proof theorem
theorem son_present_age : 
  fatherAgeCondition S F → 
  twoYearsCondition S F → 
  S = 33 :=
by
  intros h1 h2
  sorry

end son_present_age_l45_45727


namespace division_value_l45_45298

theorem division_value (x : ℝ) (h : 800 / x - 154 = 6) : x = 5 := by
  sorry

end division_value_l45_45298


namespace Bobby_candy_l45_45592

theorem Bobby_candy (initial_candy remaining_candy1 remaining_candy2 : ℕ)
  (H1 : initial_candy = 21)
  (H2 : remaining_candy1 = initial_candy - 5)
  (H3 : remaining_candy2 = remaining_candy1 - 9):
  remaining_candy2 = 7 :=
by
  sorry

end Bobby_candy_l45_45592


namespace directrix_of_parabola_l45_45410

-- Define the given conditions
def parabola_focus_on_line (p : ℝ) := ∃ (x y : ℝ), y^2 = 2 * p * x ∧ 2 * x + 3 * y - 8 = 0

-- Define the statement to be proven
theorem directrix_of_parabola (p : ℝ) (h: parabola_focus_on_line p) : 
   ∃ (d : ℝ), d = -4 := 
sorry

end directrix_of_parabola_l45_45410


namespace max_blocks_fit_l45_45506

-- Define the dimensions of the block
def block_length : ℕ := 3
def block_width : ℕ := 1
def block_height : ℕ := 1

-- Define the dimensions of the box
def box_length : ℕ := 5
def box_width : ℕ := 3
def box_height : ℕ := 2

-- Theorem stating the maximum number of blocks that can fit in the box
theorem max_blocks_fit :
  (box_length * box_width * box_height) / (block_length * block_width * block_height) = 15 := sorry

end max_blocks_fit_l45_45506


namespace sale_in_fifth_month_l45_45424

-- Define the sales in the first, second, third, fourth, and sixth months
def a1 : ℕ := 7435
def a2 : ℕ := 7927
def a3 : ℕ := 7855
def a4 : ℕ := 8230
def a6 : ℕ := 5991

-- Define the average sale
def avg_sale : ℕ := 7500

-- Define the number of months
def months : ℕ := 6

-- The total sales required for the average sale to be 7500 over 6 months.
def total_sales : ℕ := avg_sale * months

-- Calculate the sales in the first four months
def sales_first_four_months : ℕ := a1 + a2 + a3 + a4

-- Calculate the total sales for the first four months plus the sixth month.
def sales_first_four_and_sixth : ℕ := sales_first_four_months + a6

-- Prove the sale in the fifth month
theorem sale_in_fifth_month : ∃ a5 : ℕ, total_sales = sales_first_four_and_sixth + a5 ∧ a5 = 7562 :=
by
  sorry


end sale_in_fifth_month_l45_45424


namespace union_of_A_and_B_l45_45413

-- Define the sets A and B
def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}

-- Prove that the union of A and B is {-1, 0, 1}
theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} :=
  by sorry

end union_of_A_and_B_l45_45413


namespace stones_required_to_pave_hall_l45_45690

theorem stones_required_to_pave_hall :
  ∀ (hall_length_m hall_breadth_m stone_length_dm stone_breadth_dm: ℕ),
  hall_length_m = 72 →
  hall_breadth_m = 30 →
  stone_length_dm = 6 →
  stone_breadth_dm = 8 →
  (hall_length_m * 10 * hall_breadth_m * 10) / (stone_length_dm * stone_breadth_dm) = 4500 := by
  intros _ _ _ _ h_length h_breadth h_slength h_sbreadth
  sorry

end stones_required_to_pave_hall_l45_45690


namespace value_of_n_l45_45089

theorem value_of_n {k n : ℕ} (h1 : k = 71 * n + 11) (h2 : (k : ℝ) / (n : ℝ) = 71.2) : n = 55 :=
sorry

end value_of_n_l45_45089


namespace race_distance_l45_45060

theorem race_distance (a b c d : ℝ) 
  (h₁ : d / a = (d - 25) / b)
  (h₂ : d / b = (d - 15) / c)
  (h₃ : d / a = (d - 37) / c) : 
  d = 125 :=
by
  sorry

end race_distance_l45_45060


namespace greatest_ABCBA_divisible_by_13_l45_45629

theorem greatest_ABCBA_divisible_by_13 :
  ∃ A B C : ℕ, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  1 ≤ A ∧ A < 10 ∧ 0 ≤ B ∧ B < 10 ∧ 0 ≤ C ∧ C < 10 ∧
  (10000 * A + 1000 * B + 100 * C + 10 * B + A) % 13 = 0 ∧
  (10000 * A + 1000 * B + 100 * C + 10 * B + A) = 95159 :=
by
  sorry

end greatest_ABCBA_divisible_by_13_l45_45629


namespace distance_between_A_and_B_is_750_l45_45861

def original_speed := 150 -- derived from the solution

def distance (S D : ℝ) :=
  (D / S) - (D / ((5 / 4) * S)) = 1 ∧
  ((D - 150) / S) - ((5 * (D - 150)) / (6 * S)) = 2 / 3

theorem distance_between_A_and_B_is_750 :
  ∃ D : ℝ, distance original_speed D ∧ D = 750 :=
by
  sorry

end distance_between_A_and_B_is_750_l45_45861


namespace f_cos_x_l45_45293

theorem f_cos_x (f : ℝ → ℝ) (x : ℝ) (h₁ : -1 ≤ x) (h₂ : x ≤ 1) (hx : f (Real.sin x) = 2 - Real.cos (2 * x)) :
  f (Real.cos x) = 2 + (Real.cos x)^2 :=
sorry

end f_cos_x_l45_45293


namespace magician_earning_correct_l45_45153

def magician_earning (initial_decks : ℕ) (remaining_decks : ℕ) (price_per_deck : ℕ) : ℕ :=
  (initial_decks - remaining_decks) * price_per_deck

theorem magician_earning_correct :
  magician_earning 5 3 2 = 4 :=
by
  sorry

end magician_earning_correct_l45_45153


namespace Matt_received_more_pencils_than_Lauren_l45_45484

-- Definitions based on conditions
def total_pencils := 2 * 12
def pencils_to_Lauren := 6
def pencils_after_Lauren := total_pencils - pencils_to_Lauren
def pencils_left := 9
def pencils_to_Matt := pencils_after_Lauren - pencils_left

-- Formulate the problem statement
theorem Matt_received_more_pencils_than_Lauren (total_pencils := 24) (pencils_to_Lauren := 6) (pencils_after_Lauren := 18) (pencils_left := 9) (correct_answer := 3) :
  pencils_to_Matt - pencils_to_Lauren = correct_answer := 
by 
  sorry

end Matt_received_more_pencils_than_Lauren_l45_45484


namespace smallest_number_divisible_by_11_and_conditional_modulus_l45_45451

theorem smallest_number_divisible_by_11_and_conditional_modulus :
  ∃ n : ℕ, (n % 11 = 0) ∧ (n % 3 = 2) ∧ (n % 4 = 2) ∧ (n % 5 = 2) ∧ (n % 6 = 2) ∧ (n % 7 = 2) ∧ n = 2102 :=
by
  sorry

end smallest_number_divisible_by_11_and_conditional_modulus_l45_45451


namespace problem_solution_l45_45134

theorem problem_solution (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -5) :
  (1 / a) + (1 / b) = -3 / 5 :=
by
  sorry

end problem_solution_l45_45134


namespace geom_sequence_ratio_and_fifth_term_l45_45199

theorem geom_sequence_ratio_and_fifth_term 
  (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ = 10) 
  (h₂ : a₂ = -15) 
  (h₃ : a₃ = 22.5) 
  (h₄ : a₄ = -33.75) : 
  ∃ r a₅, r = -1.5 ∧ a₅ = 50.625 ∧ (a₂ = r * a₁) ∧ (a₃ = r * a₂) ∧ (a₄ = r * a₃) ∧ (a₅ = r * a₄) := 
by
  sorry

end geom_sequence_ratio_and_fifth_term_l45_45199


namespace sum_reciprocal_factors_of_12_l45_45462

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l45_45462


namespace find_integer_pairs_l45_45118

theorem find_integer_pairs : 
  ∀ (x y : Int), x^3 = y^3 + 2 * y^2 + 1 ↔ (x, y) = (1, 0) ∨ (x, y) = (1, -2) ∨ (x, y) = (-2, -3) :=
by
  intros x y
  sorry

end find_integer_pairs_l45_45118


namespace circle_area_ratio_l45_45147

theorem circle_area_ratio (O X P : ℝ) (rOx rOp : ℝ) (h1 : rOx = rOp / 3) :
  (π * rOx^2) / (π * rOp^2) = 1 / 9 :=
by 
  -- Import required theorems and add assumptions as necessary
  -- Continue the proof based on Lean syntax
  sorry

end circle_area_ratio_l45_45147


namespace mean_study_hours_l45_45791

theorem mean_study_hours :
  let students := [3, 6, 8, 5, 4, 2, 2]
  let hours := [0, 2, 4, 6, 8, 10, 12]
  (0 * 3 + 2 * 6 + 4 * 8 + 6 * 5 + 8 * 4 + 10 * 2 + 12 * 2) / (3 + 6 + 8 + 5 + 4 + 2 + 2) = 5 :=
by
  sorry

end mean_study_hours_l45_45791


namespace polygon_sides_l45_45145

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
by
  sorry

end polygon_sides_l45_45145


namespace jose_internet_speed_l45_45356

-- Define the given conditions
def file_size : ℕ := 160
def upload_time : ℕ := 20

-- Define the statement we need to prove
theorem jose_internet_speed : file_size / upload_time = 8 :=
by
  -- Proof should be provided here
  sorry

end jose_internet_speed_l45_45356


namespace complex_modulus_squared_l45_45454

open Complex

theorem complex_modulus_squared (w : ℂ) (h : w^2 + abs w ^ 2 = 7 + 2 * I) : abs w ^ 2 = 53 / 14 :=
sorry

end complex_modulus_squared_l45_45454


namespace inequality_correct_l45_45129

variable {a b : ℝ}

theorem inequality_correct (h₁ : a < 1) (h₂ : b > 1) : ab < a + b :=
sorry

end inequality_correct_l45_45129


namespace not_divisible_by_97_l45_45498

theorem not_divisible_by_97 (k : ℤ) (h : k ∣ (99^3 - 99)) : k ≠ 97 :=
sorry

end not_divisible_by_97_l45_45498


namespace side_length_of_square_IJKL_l45_45721

theorem side_length_of_square_IJKL 
  (x y : ℝ) (hypotenuse : ℝ) 
  (h1 : x - y = 3) 
  (h2 : x + y = 9) 
  (h3 : hypotenuse = Real.sqrt (x^2 + y^2)) : 
  hypotenuse = 3 * Real.sqrt 5 :=
by
  sorry

end side_length_of_square_IJKL_l45_45721


namespace brendan_cuts_yards_l45_45818

theorem brendan_cuts_yards (x : ℝ) (h : 7 * 1.5 * x = 84) : x = 8 :=
sorry

end brendan_cuts_yards_l45_45818


namespace probability_entire_grid_black_l45_45683

-- Definitions of the problem in terms of conditions.
def grid_size : Nat := 4

def prob_black_initial : ℚ := 1 / 2

def middle_squares : List (Nat × Nat) := [(2, 2), (2, 3), (3, 2), (3, 3)]

def edge_squares : List (Nat × Nat) := 
  [ (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 0), (1, 3),
    (2, 0), (2, 3),
    (3, 0), (3, 1), (3, 2), (3, 3) ]

-- The probability that each of these squares is black independently.
def prob_all_middle_black : ℚ := (1 / 2) ^ 4

def prob_all_edge_black : ℚ := (1 / 2) ^ 12

-- The combined probability that the entire grid is black.
def prob_grid_black := prob_all_middle_black * prob_all_edge_black

-- Statement of the proof problem.
theorem probability_entire_grid_black :
  prob_grid_black = 1 / 65536 := by
  sorry

end probability_entire_grid_black_l45_45683


namespace discount_percentage_l45_45338

theorem discount_percentage (C M A : ℝ) (h1 : M = 1.40 * C) (h2 : A = 1.05 * C) :
    (M - A) / M * 100 = 25 :=
by
  sorry

end discount_percentage_l45_45338


namespace francine_leave_time_earlier_l45_45698

-- Definitions for the conditions in the problem
def leave_time := "noon"  -- Francine and her father leave at noon every day.
def father_meet_time_shorten := 10  -- They arrived home 10 minutes earlier than usual.
def francine_walk_duration := 15  -- Francine walked for 15 minutes.

-- Premises based on the conditions
def usual_meet_time := 12 * 60  -- Meeting time in minutes from midnight (noon = 720 minutes)
def special_day_meet_time := usual_meet_time - father_meet_time_shorten / 2  -- 5 minutes earlier

-- The main theorem to prove: Francine leaves at 11:40 AM (700 minutes from midnight)
theorem francine_leave_time_earlier :
  usual_meet_time - (father_meet_time_shorten / 2 + francine_walk_duration) = (11 * 60 + 40) := by
  sorry

end francine_leave_time_earlier_l45_45698


namespace rectangular_box_inscribed_in_sphere_l45_45508

noncomputable def problem_statement : Prop :=
  ∃ (a b c s : ℝ), (4 * (a + b + c) = 72) ∧ (2 * (a * b + b * c + c * a) = 216) ∧
  (a^2 + b^2 + c^2 = 108) ∧ (4 * s^2 = 108) ∧ (s = 3 * Real.sqrt 3)

theorem rectangular_box_inscribed_in_sphere : problem_statement := 
  sorry

end rectangular_box_inscribed_in_sphere_l45_45508


namespace sum_G_correct_l45_45927

def G (n : ℕ) : ℕ :=
  if n % 2 = 0 then n^2 + 1 else n^2

def sum_G (a b : ℕ) : ℕ :=
  List.sum (List.map G (List.range' a (b - a + 1)))

theorem sum_G_correct :
  sum_G 2 2007 = 8546520 := by
  sorry

end sum_G_correct_l45_45927


namespace smallest_n_l45_45653

theorem smallest_n (n : ℕ) (h₁ : ∃ k1 : ℕ, 4 * n = k1 ^ 2) (h₂ : ∃ k2 : ℕ, 3 * n = k2 ^ 3) : n = 18 :=
sorry

end smallest_n_l45_45653


namespace complex_addition_zero_l45_45358

theorem complex_addition_zero (a b : ℝ) (i : ℂ) (h1 : (1 + i) * i = a + b * i) (h2 : i * i = -1) : a + b = 0 :=
sorry

end complex_addition_zero_l45_45358


namespace monotonicity_of_f_solve_inequality_range_of_m_l45_45511

variable {f : ℝ → ℝ}
variable {a b m : ℝ}

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def in_interval (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1
def f_at_one (f : ℝ → ℝ) : Prop := f 1 = 1
def positivity_condition (f : ℝ → ℝ) (a b : ℝ) : Prop := (a + b ≠ 0) → ((f a + f b) / (a + b) > 0)

-- Proof problems
theorem monotonicity_of_f 
  (h_odd : is_odd_function f) 
  (h_interval : ∀ x, in_interval x → in_interval (f x)) 
  (h_f_one : f_at_one f) 
  (h_pos : ∀ a b, in_interval a → in_interval b → positivity_condition f a b) :
  ∀ x1 x2, in_interval x1 → in_interval x2 → x1 < x2 → f x1 < f x2 :=
sorry

theorem solve_inequality 
  (h_odd : is_odd_function f) 
  (h_interval : ∀ x, in_interval x → in_interval (f x)) 
  (h_f_increasing : ∀ x1 x2, in_interval x1 → in_interval x2 → x1 < x2 → f x1 < f x2) 
  (h_f_one : f_at_one f) 
  (h_pos : ∀ a b, in_interval a → in_interval b → positivity_condition f a b) :
  ∀ x, in_interval (x + 1/2) → in_interval (1 / (x - 1)) → f (x + 1/2) < f (1 / (x - 1)) → -3/2 ≤ x ∧ x < -1 :=
sorry

theorem range_of_m 
  (h_odd : is_odd_function f) 
  (h_interval : ∀ x, in_interval x → in_interval (f x)) 
  (h_f_one : f_at_one f) 
  (h_pos : ∀ a b, in_interval a → in_interval b → positivity_condition f a b) 
  (h_f_increasing : ∀ x1 x2, in_interval x1 → in_interval x2 → x1 < x2 → f x1 < f x2) :
  (∀ a, in_interval a → f a ≤ m^2 - 2 * a * m + 1) → (m = 0 ∨ m ≤ -2 ∨ m ≥ 2) :=
sorry

end monotonicity_of_f_solve_inequality_range_of_m_l45_45511


namespace james_faster_than_john_l45_45777

theorem james_faster_than_john :
  let john_time := 13
  let john_distance := 100
  let john_first_second := 4
  let john_remaining_seconds := john_time - 1
  let john_remaining_distance := john_distance - john_first_second
  let john_top_speed := john_remaining_distance / john_remaining_seconds

  let james_time := 11
  let james_first_two_seconds := 10
  let james_remaining_seconds := james_time - 2
  let james_remaining_distance := john_distance - james_first_two_seconds
  let james_top_speed := james_remaining_distance / james_remaining_seconds
  
  james_top_speed - john_top_speed = 2 :=
by
  let john_time := 13
  let john_distance := 100
  let john_first_second := 4
  let john_remaining_seconds := john_time - 1
  let john_remaining_distance := john_distance - john_first_second
  let john_top_speed := john_remaining_distance / john_remaining_seconds

  let james_time := 11
  let james_first_two_seconds := 10
  let james_remaining_seconds := james_time - 2
  let james_remaining_distance := john_distance - james_first_two_seconds
  let james_top_speed := james_remaining_distance / james_remaining_seconds

  sorry

end james_faster_than_john_l45_45777


namespace largest_integer_solving_inequality_l45_45278

theorem largest_integer_solving_inequality :
  ∃ (x : ℤ), (7 - 5 * x > 22) ∧ ∀ (y : ℤ), (7 - 5 * y > 22) → x ≥ y ∧ x = -4 :=
by
  sorry

end largest_integer_solving_inequality_l45_45278


namespace simplify_expression_l45_45309

theorem simplify_expression : (1 / (1 + Real.sqrt 2)) * (1 / (1 - Real.sqrt 2)) = -1 := by
  sorry

end simplify_expression_l45_45309


namespace cost_flying_X_to_Y_l45_45391

def distance_XY : ℝ := 4500 -- Distance from X to Y in km
def cost_per_km_flying : ℝ := 0.12 -- Cost per km for flying in dollars
def booking_fee_flying : ℝ := 120 -- Booking fee for flying in dollars

theorem cost_flying_X_to_Y : 
    distance_XY * cost_per_km_flying + booking_fee_flying = 660 := by
  sorry

end cost_flying_X_to_Y_l45_45391


namespace area_of_figure_M_l45_45019

noncomputable def figure_M_area : Real :=
  sorry

theorem area_of_figure_M :
  figure_M_area = 3 :=
  sorry

end area_of_figure_M_l45_45019


namespace possible_k_values_l45_45702

def triangle_right_k_values (AB AC : ℝ × ℝ) (k : ℝ) : Prop :=
  let BC := (AC.1 - AB.1, AC.2 - AB.2)
  let angle_A := AB.1 * AC.1 + AB.2 * AC.2 = 0   -- Condition for ∠A = 90°
  let angle_B := AB.1 * BC.1 + AB.2 * BC.2 = 0   -- Condition for ∠B = 90°
  let angle_C := BC.1 * AC.1 + BC.2 * AC.2 = 0   -- Condition for ∠C = 90°
  (angle_A ∨ angle_B ∨ angle_C)

theorem possible_k_values (k : ℝ) :
  triangle_right_k_values (2, 3) (1, k) k ↔
  k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 13) / 2 ∨ k = (3 - Real.sqrt 13) / 2 :=
by
  sorry

end possible_k_values_l45_45702


namespace cream_ratio_l45_45528

def joe_ends_with_cream (start_coffee : ℕ) (drank_coffee : ℕ) (added_cream : ℕ) : ℕ :=
  added_cream

def joann_cream_left (start_coffee : ℕ) (added_cream : ℕ) (drank_mix : ℕ) : ℚ :=
  added_cream - drank_mix * (added_cream / (start_coffee + added_cream))

theorem cream_ratio (start_coffee : ℕ) (joe_drinks : ℕ) (joe_adds : ℕ)
                    (joann_adds : ℕ) (joann_drinks : ℕ) :
  joe_ends_with_cream start_coffee joe_drinks joe_adds / 
  joann_cream_left start_coffee joann_adds joann_drinks = (9 : ℚ) / (7 : ℚ) :=
by
  sorry

end cream_ratio_l45_45528


namespace birds_landed_l45_45551

theorem birds_landed (original_birds total_birds : ℕ) (h : original_birds = 12) (h2 : total_birds = 20) :
  total_birds - original_birds = 8 :=
by {
  sorry
}

end birds_landed_l45_45551


namespace factor_theorem_q_value_l45_45888

theorem factor_theorem_q_value (q : ℤ) (m : ℤ) :
  (∀ m, (m - 8) ∣ (m^2 - q * m - 24)) → q = 5 :=
by
  sorry

end factor_theorem_q_value_l45_45888


namespace exists_A_for_sqrt_d_l45_45807

def is_not_perfect_square (d : ℕ) : Prop := ∀ m : ℕ, m * m ≠ d

def s (d n : ℕ) : ℕ := 
  -- count number of 1's in the first n digits of binary representation of √d
  sorry 

theorem exists_A_for_sqrt_d (d : ℕ) (h : is_not_perfect_square d) :
  ∃ A : ℕ, ∀ n ≥ A, s d n > Int.sqrt (2 * n) - 2 :=
sorry

end exists_A_for_sqrt_d_l45_45807


namespace pencils_total_l45_45003

theorem pencils_total (p1 p2 : ℕ) (h1 : p1 = 3) (h2 : p2 = 7) : p1 + p2 = 10 := by
  sorry

end pencils_total_l45_45003


namespace cube_faces_consecutive_sum_l45_45669

noncomputable def cube_face_sum (n : ℕ) : ℕ :=
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5)

theorem cube_faces_consecutive_sum (n : ℕ) (h1 : ∀ i, i ∈ [0, 5] -> (2 * n + 5 + n + 5 - 6) = 6) (h2 : n = 12) :
  cube_face_sum n = 87 :=
  sorry

end cube_faces_consecutive_sum_l45_45669


namespace bob_buys_nose_sprays_l45_45018

theorem bob_buys_nose_sprays (cost_per_spray : ℕ) (promotion : ℕ → ℕ) (total_paid : ℕ)
  (h1 : cost_per_spray = 3)
  (h2 : ∀ n, promotion n = 2 * n)
  (h3 : total_paid = 15) : (total_paid / cost_per_spray) * 2 = 10 :=
by
  sorry

end bob_buys_nose_sprays_l45_45018


namespace geometric_sequence_sum_l45_45117

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 0 + a 1 = 1) (h2 : a 1 + a 2 = 2) : a 5 + a 6 = 32 :=
sorry

end geometric_sequence_sum_l45_45117


namespace file_size_l45_45809

-- Definitions based on conditions
def upload_speed : ℕ := 8 -- megabytes per minute
def upload_time : ℕ := 20 -- minutes

-- Goal to prove
theorem file_size:
  (upload_speed * upload_time = 160) :=
by sorry

end file_size_l45_45809


namespace polynomial_never_33_l45_45205

theorem polynomial_never_33 (x y : ℤ) : 
  x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 :=
by
  sorry

end polynomial_never_33_l45_45205


namespace find_f_neg_2_l45_45497

theorem find_f_neg_2 (f : ℝ → ℝ) (b x : ℝ) (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, x ≥ 0 → f x = x^2 - 3*x + b) (h3 : f 0 = 0) : f (-2) = 2 := by
sorry

end find_f_neg_2_l45_45497


namespace speed_of_ferry_P_l45_45519

variable (v_P v_Q : ℝ)

noncomputable def condition1 : Prop := v_Q = v_P + 4
noncomputable def condition2 : Prop := (6 * v_P) / v_Q = 4
noncomputable def condition3 : Prop := 2 + 2 = 4

theorem speed_of_ferry_P
    (h1 : condition1 v_P v_Q)
    (h2 : condition2 v_P v_Q)
    (h3 : condition3) :
    v_P = 8 := 
by 
    sorry

end speed_of_ferry_P_l45_45519


namespace solve_length_BF_l45_45254

-- Define the problem conditions
def rectangular_paper (short_side long_side : ℝ) : Prop :=
  short_side = 12 ∧ long_side > short_side

def vertex_touch_midpoint (vmp mid : ℝ) : Prop :=
  vmp = mid / 2

def congruent_triangles (triangle1 triangle2 : ℝ) : Prop :=
  triangle1 = triangle2

-- Theorem to prove the length of BF
theorem solve_length_BF (short_side long_side vmp mid triangle1 triangle2 : ℝ) 
  (h1 : rectangular_paper short_side long_side)
  (h2 : vertex_touch_midpoint vmp mid)
  (h3 : congruent_triangles triangle1 triangle2) :
  -- The length of BF is 10
  mid = 6 → 18 - 6 = 12 + 6 - 10 → 10 = 12 - (18 - 10) → vmp = 6 → 6 * 2 = 12 →
  sorry :=
sorry

end solve_length_BF_l45_45254


namespace infinite_common_divisor_l45_45901

theorem infinite_common_divisor (n : ℕ) : ∃ᶠ n in at_top, Nat.gcd (2 * n - 3) (3 * n - 2) > 1 := 
sorry

end infinite_common_divisor_l45_45901


namespace original_price_of_article_l45_45362

theorem original_price_of_article (selling_price : ℝ) (loss_percent : ℝ) (P : ℝ) 
  (h1 : selling_price = 450)
  (h2 : loss_percent = 25)
  : selling_price = (1 - loss_percent / 100) * P → P = 600 :=
by
  sorry

end original_price_of_article_l45_45362


namespace maximum_value_expression_l45_45534

theorem maximum_value_expression (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h_sum : a + b + c = 3) : 
  (a^2 - a * b + b^2) * (a^2 - a * c + c^2) * (b^2 - b * c + c^2) ≤ 1 :=
sorry

end maximum_value_expression_l45_45534


namespace inequality_proof_l45_45040

theorem inequality_proof
  (a b c d : ℝ) (h0 : a ≥ 0) (h1 : b ≥ 0) (h2 : c ≥ 0) (h3 : d ≥ 0) (h4 : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c)) ≥ 1 / 3 :=
sorry

end inequality_proof_l45_45040


namespace determine_properties_range_of_m_l45_45430

noncomputable def f (a x : ℝ) : ℝ := (a / (a - 1)) * (2^x - 2^(-x))

theorem determine_properties (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) :
  (∀ x : ℝ, f a (-x) = -f a x) ∧
  ((0 < a ∧ a < 1) → ∀ x1 x2 : ℝ, x1 < x2 → f a x1 > f a x2) ∧
  (a > 1 → ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2) := 
sorry

theorem range_of_m (a m : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) (h_m_in_I : -1 < m ∧ m < 1) :
  f a (m - 1) + f a m < 0 ↔ 
  ((0 < a ∧ a < 1 → (1 / 2) < m ∧ m < 1) ∧
  (a > 1 → 0 < m ∧ m < (1 / 2))) := 
sorry

end determine_properties_range_of_m_l45_45430


namespace num_positive_k_for_solution_to_kx_minus_18_eq_3k_l45_45597

theorem num_positive_k_for_solution_to_kx_minus_18_eq_3k : 
  ∃ (k_vals : Finset ℕ), 
  (∀ k ∈ k_vals, ∃ x : ℤ, k * x - 18 = 3 * k) ∧ 
  k_vals.card = 6 :=
by
  sorry

end num_positive_k_for_solution_to_kx_minus_18_eq_3k_l45_45597


namespace probability_difference_l45_45893

-- Definitions for probabilities
def P_plane : ℚ := 7 / 10
def P_train : ℚ := 3 / 10
def P_on_time_plane : ℚ := 8 / 10
def P_on_time_train : ℚ := 9 / 10

-- Events definitions
def P_arrive_on_time : ℚ := (7 / 10) * (8 / 10) + (3 / 10) * (9 / 10)
def P_plane_and_on_time : ℚ := (7 / 10) * (8 / 10)
def P_train_and_on_time : ℚ := (3 / 10) * (9 / 10)
def P_conditional_plane_given_on_time : ℚ := P_plane_and_on_time / P_arrive_on_time
def P_conditional_train_given_on_time : ℚ := P_train_and_on_time / P_arrive_on_time

theorem probability_difference :
  P_conditional_plane_given_on_time - P_conditional_train_given_on_time = 29 / 83 :=
by sorry

end probability_difference_l45_45893


namespace largest_w_exists_l45_45460

theorem largest_w_exists (w x y z : ℝ) (h1 : w + x + y + z = 25) (h2 : w * x + w * y + w * z + x * y + x * z + y * z = 2 * y + 2 * z + 193) :
  ∃ (w1 w2 : ℤ), w1 > 0 ∧ w2 > 0 ∧ ((w = w1 / w2) ∧ (w1 + w2 = 27)) :=
sorry

end largest_w_exists_l45_45460


namespace manufacturers_price_l45_45505

theorem manufacturers_price (M : ℝ) 
  (h1 : 0.1 ≤ 0.3) 
  (h2 : 0.2 = 0.2) 
  (h3 : 0.56 * M = 25.2) : 
  M = 45 := 
sorry

end manufacturers_price_l45_45505


namespace union_A_B_l45_45808

def set_A : Set ℝ := { x | 1 / x ≤ 0 }
def set_B : Set ℝ := { x | x^2 - 1 < 0 }

theorem union_A_B : set_A ∪ set_B = { x | x < 1 } :=
by
  sorry

end union_A_B_l45_45808


namespace bat_wings_area_l45_45144

structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨0, 0⟩
def Q : Point := ⟨5, 0⟩
def R : Point := ⟨5, 2⟩
def S : Point := ⟨0, 2⟩
def A : Point := ⟨5, 1⟩
def T : Point := ⟨3, 2⟩

def area_triangle (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x))

theorem bat_wings_area :
  area_triangle P A T = 5.5 :=
sorry

end bat_wings_area_l45_45144


namespace probability_not_cash_l45_45765

theorem probability_not_cash (h₁ : 0.45 + 0.15 + pnc = 1) : pnc = 0.4 :=
by
  sorry

end probability_not_cash_l45_45765


namespace cycling_journey_l45_45760

theorem cycling_journey :
  ∃ y : ℚ, 0 < y ∧ y <= 12 ∧ (15 * y + 10 * (12 - y) = 150) ∧ y = 6 :=
by
  sorry

end cycling_journey_l45_45760


namespace roots_polynomial_sum_cubes_l45_45057

theorem roots_polynomial_sum_cubes (u v w : ℂ) (h : (∀ x, (x = u ∨ x = v ∨ x = w) → 5 * x ^ 3 + 500 * x + 1005 = 0)) :
  (u + v) ^ 3 + (v + w) ^ 3 + (w + u) ^ 3 = 603 := sorry

end roots_polynomial_sum_cubes_l45_45057


namespace value_of_x_l45_45187

theorem value_of_x (a x y : ℝ) 
  (h1 : a^(x - y) = 343) 
  (h2 : a^(x + y) = 16807) : x = 4 :=
by
  sorry

end value_of_x_l45_45187


namespace flower_problem_l45_45845

def totalFlowers (n_rows n_per_row : Nat) : Nat :=
  n_rows * n_per_row

def flowersCut (total percent_cut : Nat) : Nat :=
  total * percent_cut / 100

def flowersRemaining (total cut : Nat) : Nat :=
  total - cut

theorem flower_problem :
  let n_rows := 50
  let n_per_row := 400
  let percent_cut := 60
  let total := totalFlowers n_rows n_per_row
  let cut := flowersCut total percent_cut
  flowersRemaining total cut = 8000 :=
by
  sorry

end flower_problem_l45_45845


namespace gummy_vitamins_cost_l45_45863

def bottle_discounted_price (P D_s : ℝ) : ℝ :=
  P * (1 - D_s)

def normal_purchase_discounted_price (discounted_price D_n : ℝ) : ℝ :=
  discounted_price * (1 - D_n)

def bulk_purchase_discounted_price (discounted_price D_b : ℝ) : ℝ :=
  discounted_price * (1 - D_b)

def total_cost (normal_bottles bulk_bottles normal_price bulk_price : ℝ) : ℝ :=
  (normal_bottles * normal_price) + (bulk_bottles * bulk_price)

def apply_coupons (total_cost N_c C : ℝ) : ℝ :=
  total_cost - (N_c * C)

theorem gummy_vitamins_cost 
  (P N_c C D_s D_n D_b : ℝ) 
  (normal_bottles bulk_bottles : ℕ) :
  bottle_discounted_price P D_s = 12.45 → 
  normal_purchase_discounted_price 12.45 D_n = 11.33 → 
  bulk_purchase_discounted_price 12.45 D_b = 11.83 → 
  total_cost 4 3 11.33 11.83 = 80.81 → 
  apply_coupons 80.81 N_c C = 70.81 :=
sorry

end gummy_vitamins_cost_l45_45863


namespace shortest_third_stick_length_l45_45609

-- Definitions of the stick lengths
def length1 := 6
def length2 := 9

-- Statement: The shortest length of the third stick that forms a triangle with lengths 6 and 9 should be 4
theorem shortest_third_stick_length : ∃ length3, length3 = 4 ∧
  (length1 + length2 > length3) ∧ (length1 + length3 > length2) ∧ (length2 + length3 > length1) :=
sorry

end shortest_third_stick_length_l45_45609


namespace nth_term_formula_l45_45555

theorem nth_term_formula (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : ∀ n, S n = 2 * n^2 + n)
  (h2 : a 1 = S 1)
  (h3 : ∀ n ≥ 2, a n = S n - S (n - 1))
  : ∀ n, a n = 4 * n - 1 := by
  sorry

end nth_term_formula_l45_45555


namespace juniors_in_program_l45_45166

theorem juniors_in_program (J S x y : ℕ) (h1 : J + S = 40) 
                           (h2 : x = y) 
                           (h3 : J / 5 = x) 
                           (h4 : S / 10 = y) : J = 12 :=
by
  sorry

end juniors_in_program_l45_45166


namespace quadratic_function_value_l45_45328

theorem quadratic_function_value (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a - b + c = 9) :
  a + 3 * b + c = 1 := 
by 
  sorry

end quadratic_function_value_l45_45328


namespace dan_spent_more_on_chocolates_l45_45510

def price_candy_bar : ℝ := 4
def number_of_candy_bars : ℕ := 5
def candy_discount : ℝ := 0.20
def discount_threshold : ℕ := 3
def price_chocolate : ℝ := 6
def number_of_chocolates : ℕ := 4
def chocolate_tax_rate : ℝ := 0.05

def candy_cost_total : ℝ :=
  let cost_without_discount := number_of_candy_bars * price_candy_bar
  if number_of_candy_bars >= discount_threshold
  then cost_without_discount * (1 - candy_discount)
  else cost_without_discount

def chocolate_cost_total : ℝ :=
  let cost_without_tax := number_of_chocolates * price_chocolate
  cost_without_tax * (1 + chocolate_tax_rate)

def difference_in_spending : ℝ :=
  chocolate_cost_total - candy_cost_total

theorem dan_spent_more_on_chocolates :
  difference_in_spending = 9.20 :=
by
  sorry

end dan_spent_more_on_chocolates_l45_45510


namespace proof_solution_l45_45951

noncomputable def proof_problem (x : ℝ) : Prop :=
  (⌈2 * x⌉₊ : ℝ) - (⌊2 * x⌋₊ : ℝ) = 0 → (⌈2 * x⌉₊ : ℝ) - 2 * x = 0

theorem proof_solution (x : ℝ) : proof_problem x :=
by
  sorry

end proof_solution_l45_45951


namespace soul_inequality_phi_inequality_iff_t_one_l45_45030

noncomputable def e : ℝ := Real.exp 1

theorem soul_inequality (x : ℝ) : e^x ≥ x + 1 ↔ x = 0 :=
by sorry

theorem phi_inequality_iff_t_one (x t : ℝ) : (∀ x, e^x - t*x - 1 ≥ 0) ↔ t = 1 :=
by sorry

end soul_inequality_phi_inequality_iff_t_one_l45_45030


namespace sum_of_reciprocals_l45_45468

variable {x y : ℝ}

theorem sum_of_reciprocals (h1 : x + y = 4 * x * y) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1 / x + 1 / y) = 4 :=
by
  sorry

end sum_of_reciprocals_l45_45468


namespace sheets_of_paper_l45_45046

theorem sheets_of_paper (S E : ℕ) (h1 : S - E = 100) (h2 : E = S / 3 - 25) : S = 120 :=
sorry

end sheets_of_paper_l45_45046


namespace percentage_decrease_correct_l45_45267

theorem percentage_decrease_correct :
  ∀ (p : ℝ), (1 + 0.25) * (1 - p) = 1 → p = 0.20 :=
by
  intro p
  intro h
  sorry

end percentage_decrease_correct_l45_45267


namespace speed_A_correct_l45_45326

noncomputable def speed_A : ℝ :=
  200 / (19.99840012798976 * 60)

theorem speed_A_correct :
  speed_A = 0.16668 :=
sorry

end speed_A_correct_l45_45326


namespace find_a22_l45_45566

-- Definitions and conditions
noncomputable def seq (n : ℕ) : ℝ := if n = 0 then 0 else sorry

axiom seq_conditions
  (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) : True

theorem find_a22 (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 :=
sorry

end find_a22_l45_45566


namespace percentage_increase_in_length_l45_45154

theorem percentage_increase_in_length (L B : ℝ) (hB : 0 < B) (hL : 0 < L) :
  (1 + x / 100) * 1.22 = 1.3542 -> x = 11.016393 :=
by
  sorry

end percentage_increase_in_length_l45_45154


namespace number_of_real_a_l45_45637

open Int

-- Define the quadratic equation with integer roots
def quadratic_eq_with_integer_roots (a : ℝ) : Prop :=
  ∃ (r s : ℤ), r + s = -a ∧ r * s = 12 * a

-- Prove there are exactly 9 values of a such that the quadratic equation has only integer roots
theorem number_of_real_a (n : ℕ) : n = 9 ↔ ∃ (as : Finset ℝ), as.card = n ∧ ∀ a ∈ as, quadratic_eq_with_integer_roots a :=
by
  -- We can skip the proof with "sorry"
  sorry

end number_of_real_a_l45_45637


namespace predicted_customers_on_Saturday_l45_45548

theorem predicted_customers_on_Saturday 
  (breakfast_customers : ℕ)
  (lunch_customers : ℕ)
  (dinner_customers : ℕ)
  (prediction_factor : ℕ)
  (h1 : breakfast_customers = 73)
  (h2 : lunch_customers = 127)
  (h3 : dinner_customers = 87)
  (h4 : prediction_factor = 2) :
  prediction_factor * (breakfast_customers + lunch_customers + dinner_customers) = 574 :=  
by 
  sorry 

end predicted_customers_on_Saturday_l45_45548


namespace spider_total_distance_l45_45602

-- Define points where spider starts and moves
def start_position : ℤ := 3
def first_move : ℤ := -4
def second_move : ℤ := 8
def final_move : ℤ := 2

-- Define the total distance the spider crawls
def total_distance : ℤ :=
  |first_move - start_position| +
  |second_move - first_move| +
  |final_move - second_move|

-- Theorem statement
theorem spider_total_distance : total_distance = 25 :=
sorry

end spider_total_distance_l45_45602


namespace maximum_profit_and_price_range_l45_45031

-- Definitions
def cost_per_item : ℝ := 60
def max_profit_percentage : ℝ := 0.45
def sales_volume (x : ℝ) : ℝ := -x + 120
def profit (x : ℝ) : ℝ := sales_volume x * (x - cost_per_item)

-- The main theorem
theorem maximum_profit_and_price_range :
  (∃ x : ℝ, x = 87 ∧ profit x = 891) ∧
  (∀ x : ℝ, profit x ≥ 500 ↔ (70 ≤ x ∧ x ≤ 110)) :=
by
  sorry

end maximum_profit_and_price_range_l45_45031


namespace max_score_exam_l45_45128

theorem max_score_exam (Gibi_percent Jigi_percent Mike_percent Lizzy_percent : ℝ)
  (avg_score total_score M : ℝ) :
  Gibi_percent = 0.59 →
  Jigi_percent = 0.55 →
  Mike_percent = 0.99 →
  Lizzy_percent = 0.67 →
  avg_score = 490 →
  total_score = avg_score * 4 →
  total_score = (Gibi_percent + Jigi_percent + Mike_percent + Lizzy_percent) * M →
  M = 700 :=
by
  intros hGibi hJigi hMike hLizzy hAvg hTotalScore hEq
  sorry

end max_score_exam_l45_45128


namespace find_h_l45_45606

theorem find_h (h : ℝ) (j k : ℝ) 
  (y_eq1 : ∀ x : ℝ, (4 * (x - h)^2 + j) = 2030)
  (y_eq2 : ∀ x : ℝ, (5 * (x - h)^2 + k) = 2040)
  (int_xint1 : ∀ x1 x2 : ℝ, x1 > 0 → x2 > 0 → x1 ≠ x2 → (4 * x1 * x2 = 2032) )
  (int_xint2 : ∀ x1 x2 : ℝ, x1 > 0 → x2 > 0 → x1 ≠ x2 → (5 * x1 * x2 = 2040) ) :
  h = 20.5 :=
by
  sorry

end find_h_l45_45606


namespace smallest_n_exceeds_15_l45_45306

noncomputable def g (n : ℕ) : ℕ :=
  sorry  -- Define the sum of the digits of 1 / 3^n to the right of the decimal point

theorem smallest_n_exceeds_15 : ∃ n : ℕ, n > 0 ∧ g n > 15 ∧ ∀ m : ℕ, m > 0 ∧ g m > 15 → n ≤ m :=
  sorry  -- Prove the smallest n such that g(n) > 15

end smallest_n_exceeds_15_l45_45306


namespace number_exceeds_its_part_l45_45397

theorem number_exceeds_its_part (x : ℝ) (h : x = 3/8 * x + 25) : x = 40 :=
by sorry

end number_exceeds_its_part_l45_45397


namespace range_of_b_l45_45842

noncomputable def f : ℝ → ℝ
| x => if x < -1/2 then (2*x + 1) / (x^2) else x + 1

def g (x : ℝ) : ℝ := x^2 - 4*x - 4

theorem range_of_b (a b : ℝ) (h : f a + g b = 0) : -1 <= b ∧ b <= 5 :=
sorry

end range_of_b_l45_45842


namespace number_of_birds_seen_l45_45792

theorem number_of_birds_seen (dozens_seen : ℕ) (birds_per_dozen : ℕ) (h₀ : dozens_seen = 8) (h₁ : birds_per_dozen = 12) : dozens_seen * birds_per_dozen = 96 :=
by sorry

end number_of_birds_seen_l45_45792


namespace smallest_digit_for_divisibility_by_3_l45_45033

theorem smallest_digit_for_divisibility_by_3 : ∃ x : ℕ, x < 10 ∧ (5 + 2 + 6 + x + 1 + 8) % 3 = 0 ∧ ∀ y : ℕ, y < 10 ∧ (5 + 2 + 6 + y + 1 + 8) % 3 = 0 → x ≤ y := by
  sorry

end smallest_digit_for_divisibility_by_3_l45_45033


namespace parametric_eq_to_ordinary_l45_45607

theorem parametric_eq_to_ordinary (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) :
    let x := abs (Real.sin (θ / 2) + Real.cos (θ / 2))
    let y := 1 + Real.sin θ
    x ^ 2 = y := by sorry

end parametric_eq_to_ordinary_l45_45607


namespace FG_length_of_trapezoid_l45_45007

-- Define the dimensions and properties of trapezoid EFGH.
def EFGH_trapezoid (area : ℝ) (altitude : ℝ) (EF : ℝ) (GH : ℝ) : Prop :=
  area = 180 ∧ altitude = 9 ∧ EF = 12 ∧ GH = 20

-- State the theorem to prove the length of FG.
theorem FG_length_of_trapezoid : 
  ∀ {E F G H : Type} (area EF GH fg : ℝ) (altitude : ℝ),
  EFGH_trapezoid area altitude EF GH → fg = 6.57 :=
by sorry

end FG_length_of_trapezoid_l45_45007


namespace slower_pump_time_l45_45682

def pool_problem (R : ℝ) :=
  (∀ t : ℝ, (2.5 * R * t = 1) → (t = 5))
  ∧ (∀ R1 R2 : ℝ, (R1 = 1.5 * R) → (R1 + R = 2.5 * R))
  ∧ (∀ t : ℝ, (R * t = 1) → (t = 12.5))

theorem slower_pump_time (R : ℝ) : pool_problem R :=
by
  -- Assume that the combined rates take 5 hours to fill the pool
  sorry

end slower_pump_time_l45_45682


namespace football_team_goal_l45_45303

-- Definitions of the conditions
def L1 : ℤ := -5
def G2 : ℤ := 13
def L3 : ℤ := -(L1 ^ 2)
def G4 : ℚ := - (L3 : ℚ) / 2

def total_yardage : ℚ := L1 + G2 + L3 + G4

-- The statement to be proved
theorem football_team_goal : total_yardage < 30 := by
  -- sorry for now since no proof is needed
  sorry

end football_team_goal_l45_45303


namespace smallest_three_digit_integer_l45_45871

theorem smallest_three_digit_integer (n : ℕ) : 
  100 ≤ n ∧ n < 1000 ∧ ¬ (n - 1 ∣ (n!)) ↔ n = 1004 := 
by
  sorry

end smallest_three_digit_integer_l45_45871


namespace brownies_count_l45_45136

theorem brownies_count (pan_length : ℕ) (pan_width : ℕ) (piece_side : ℕ) 
  (h1 : pan_length = 24) (h2 : pan_width = 15) (h3 : piece_side = 3) : 
  (pan_length * pan_width) / (piece_side * piece_side) = 40 :=
by {
  sorry
}

end brownies_count_l45_45136


namespace general_term_formula_of_sequence_l45_45437

theorem general_term_formula_of_sequence {a : ℕ → ℝ} (S : ℕ → ℝ)
  (hS : ∀ n, S n = (2 / 3) * a n + 1 / 3) :
  (∀ n, a n = (-2) ^ (n - 1)) :=
by
  sorry

end general_term_formula_of_sequence_l45_45437


namespace number_of_pupils_l45_45449

theorem number_of_pupils (n : ℕ) (h1 : 79 - 45 = 34)
  (h2 : 34 = 1 / 2 * n) : n = 68 :=
by
  sorry

end number_of_pupils_l45_45449


namespace exponential_rule_l45_45396

theorem exponential_rule (a : ℝ) : (a ^ 3) ^ 2 = a ^ 6 :=  
  sorry

end exponential_rule_l45_45396


namespace triangle_interior_angle_l45_45688

-- Define the given values and equations
variables (x : ℝ) 
def arc_DE := x + 80
def arc_EF := 2 * x + 30
def arc_FD := 3 * x - 25

-- The main proof statement
theorem triangle_interior_angle :
  arc_DE x + arc_EF x + arc_FD x = 360 →
  0.5 * (arc_EF x) = 60.83 :=
by sorry

end triangle_interior_angle_l45_45688


namespace total_people_l45_45545

-- Given definitions
def students : ℕ := 37500
def ratio_students_professors : ℕ := 15
def professors : ℕ := students / ratio_students_professors

-- The statement to prove
theorem total_people : students + professors = 40000 := by
  sorry

end total_people_l45_45545


namespace inequality_solution_l45_45749

theorem inequality_solution (x : ℝ) :
  (x - 3) / (x^2 + 4 * x + 13) ≥ 0 ↔ x ∈ Set.Ici 3 :=
by
  sorry

end inequality_solution_l45_45749


namespace hh_two_eq_902_l45_45098

def h (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

theorem hh_two_eq_902 : h (h 2) = 902 := 
by
  sorry

end hh_two_eq_902_l45_45098


namespace second_term_is_correct_l45_45799

noncomputable def arithmetic_sequence_second_term (a d : ℤ) (h1 : a + 9 * d = 15) (h2 : a + 10 * d = 18) : ℤ :=
  a + d

theorem second_term_is_correct (a d : ℤ) (h1 : a + 9 * d = 15) (h2 : a + 10 * d = 18) :
  arithmetic_sequence_second_term a d h1 h2 = -9 :=
sorry

end second_term_is_correct_l45_45799


namespace remainder_103_107_div_11_l45_45116

theorem remainder_103_107_div_11 :
  (103 * 107) % 11 = 10 :=
by
  sorry

end remainder_103_107_div_11_l45_45116


namespace middle_rungs_widths_l45_45910

theorem middle_rungs_widths (a : ℕ → ℝ) (d : ℝ) :
  a 1 = 33 ∧ a 12 = 110 ∧ (∀ n, a (n + 1) = a n + 7) →
  (a 2 = 40 ∧ a 3 = 47 ∧ a 4 = 54 ∧ a 5 = 61 ∧
   a 6 = 68 ∧ a 7 = 75 ∧ a 8 = 82 ∧ a 9 = 89 ∧
   a 10 = 96 ∧ a 11 = 103) :=
by
  sorry

end middle_rungs_widths_l45_45910


namespace factorization_problem_l45_45125

theorem factorization_problem (a b c : ℝ) :
  let E := a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3)
  let P := -(a^2 + ab + b^2 + bc + c^2 + ac)
  E = (a - b) * (b - c) * (c - a) * P :=
by
  sorry

end factorization_problem_l45_45125


namespace set_intersection_A_B_l45_45880

theorem set_intersection_A_B :
  (A : Set ℤ) ∩ (B : Set ℤ) = { -1, 0, 1, 2 } :=
by
  let A := { x : ℤ | x^2 - x - 2 ≤ 0 }
  let B := {x : ℤ | x ∈ Set.univ}
  sorry

end set_intersection_A_B_l45_45880


namespace parallel_line_equation_perpendicular_line_equation_l45_45576

theorem parallel_line_equation {x y : ℝ} (P : ∃ x y, 2 * x + y - 5 = 0 ∧ x - 2 * y = 0) :
  (∃ (l : ℝ), ∀ x y, 4 * x - y - 7 = 0) :=
sorry

theorem perpendicular_line_equation {x y : ℝ} (P : ∃ x y, 2 * x + y - 5 = 0 ∧ x - 2 * y = 0) :
  (∃ (l : ℝ), ∀ x y, x + 4 * y - 6 = 0) :=
sorry

end parallel_line_equation_perpendicular_line_equation_l45_45576


namespace max_students_gave_away_balls_more_l45_45544

theorem max_students_gave_away_balls_more (N : ℕ) (hN : N ≤ 13) : 
  ∃(students : ℕ), students = 27 ∧ (students = 27 ∧ N ≤ students - N) :=
by
  sorry

end max_students_gave_away_balls_more_l45_45544


namespace solve_eq1_solve_eq2_l45_45274

-- Define the first problem statement and the correct answers
theorem solve_eq1 (x : ℝ) (h : (x - 2) ^ 2 = 169) : x = 15 ∨ x = -11 := 
  by sorry

-- Define the second problem statement and the correct answer
theorem solve_eq2 (x : ℝ) (h : 3 * (x - 3) ^ 3 - 24 = 0) : x = 5 := 
  by sorry

end solve_eq1_solve_eq2_l45_45274


namespace minimum_value_fraction_l45_45204

theorem minimum_value_fraction (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) :
  ∃ (x : ℝ), (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 3 → x ≤ (a + b) / (a * b * c)) ∧ x = 16 / 9 := 
sorry

end minimum_value_fraction_l45_45204


namespace A_alone_days_l45_45318

variable (r_A r_B r_C : ℝ)

-- Given conditions:
axiom cond1 : r_A + r_B = 1 / 3
axiom cond2 : r_B + r_C = 1 / 6
axiom cond3 : r_A + r_C = 4 / 15

-- Proposition stating the required proof, that A alone can do the job in 60/13 days:
theorem A_alone_days : r_A ≠ 0 → 1 / r_A = 60 / 13 :=
by
  intro h
  sorry

end A_alone_days_l45_45318


namespace perfect_square_polynomial_l45_45368

theorem perfect_square_polynomial (m : ℝ) :
  (∃ f : ℝ → ℝ, ∀ x, x^2 - (m + 1) * x + 1 = (f x) * (f x)) → (m = 1 ∨ m = -3) :=
by
  sorry

end perfect_square_polynomial_l45_45368


namespace apple_count_l45_45581

-- Definitions of initial conditions and calculations.
def B_0 : Int := 5  -- initial number of blue apples
def R_0 : Int := 3  -- initial number of red apples
def Y : Int := 2 * B_0  -- number of yellow apples given by neighbor
def R : Int := R_0 - 2  -- number of red apples after giving away to a friend
def B : Int := B_0 - 3  -- number of blue apples after 3 rot
def G : Int := (B + Y) / 3  -- number of green apples received
def Y' : Int := Y - 2  -- number of yellow apples after eating 2
def R' : Int := R - 1  -- number of red apples after eating 1

-- Lean theorem statement
theorem apple_count (B_0 R_0 Y R B G Y' R' : ℤ)
  (h1 : B_0 = 5)
  (h2 : R_0 = 3)
  (h3 : Y = 2 * B_0)
  (h4 : R = R_0 - 2)
  (h5 : B = B_0 - 3)
  (h6 : G = (B + Y) / 3)
  (h7 : Y' = Y - 2)
  (h8 : R' = R - 1)
  : B + Y' + G + R' = 14 := 
by
  sorry

end apple_count_l45_45581


namespace percentage_y_less_than_x_l45_45617

theorem percentage_y_less_than_x (x y : ℝ) (h : x = 11 * y) : 
  ((x - y) / x) * 100 = 90.91 := 
by 
  sorry -- proof to be provided separately

end percentage_y_less_than_x_l45_45617


namespace seventh_observation_is_4_l45_45757

def avg_six := 11 -- Average of the first six observations
def sum_six := 6 * avg_six -- Total sum of the first six observations
def new_avg := avg_six - 1 -- New average after including the new observation
def new_sum := 7 * new_avg -- Total sum after including the new observation

theorem seventh_observation_is_4 : 
  (new_sum - sum_six) = 4 :=
by
  sorry

end seventh_observation_is_4_l45_45757


namespace charity_tickets_solution_l45_45419

theorem charity_tickets_solution (f h d p : ℕ) (ticket_count : f + h + d = 200)
  (revenue : f * p + h * (p / 2) + d * (2 * p) = 3600) : f = 80 := by
  sorry

end charity_tickets_solution_l45_45419


namespace surface_area_original_cube_l45_45420

theorem surface_area_original_cube
  (n : ℕ)
  (edge_length_smaller : ℕ)
  (smaller_cubes : ℕ)
  (original_surface_area : ℕ)
  (h1 : n = 3)
  (h2 : edge_length_smaller = 4)
  (h3 : smaller_cubes = 27)
  (h4 : 6 * (n * edge_length_smaller) ^ 2 = original_surface_area) :
  original_surface_area = 864 := by
  sorry

end surface_area_original_cube_l45_45420


namespace exactly_one_defective_l45_45643

theorem exactly_one_defective (p_A p_B : ℝ) (hA : p_A = 0.04) (hB : p_B = 0.05) :
  ((p_A * (1 - p_B)) + ((1 - p_A) * p_B)) = 0.086 :=
by
  sorry

end exactly_one_defective_l45_45643


namespace tim_score_in_math_l45_45264

def even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14]

def sum_even_numbers (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem tim_score_in_math : sum_even_numbers even_numbers = 56 := by
  -- Proof steps would be here
  sorry

end tim_score_in_math_l45_45264


namespace integer_solution_for_equation_l45_45917

theorem integer_solution_for_equation :
  ∃ (M : ℤ), 14^2 * 35^2 = 10^2 * (M - 10)^2 ∧ M = 59 :=
by
  sorry

end integer_solution_for_equation_l45_45917


namespace perpendicular_line_through_point_l45_45085

theorem perpendicular_line_through_point (x y : ℝ) (c : ℝ) (P : ℝ × ℝ) :
  P = (-1, 2) →
  (∀ x y c : ℝ, (2*x - y + c = 0) ↔ (x+2*y-1=0) → (x+2*y-1=0)) →
  ∃ c : ℝ, 2*(-1) - 2 + c = 0 ∧ (2*x - y + c = 0) :=
by
  sorry

end perpendicular_line_through_point_l45_45085


namespace remainder_division_l45_45035

theorem remainder_division
  (P E M S F N T : ℕ)
  (h1 : P = E * M + S)
  (h2 : M = N * F + T) :
  (∃ r, P = (EF + 1) * (P / (EF + 1)) + r ∧ r = ET + S - N) :=
sorry

end remainder_division_l45_45035


namespace cookie_radius_l45_45407

theorem cookie_radius (x y : ℝ) : x^2 + y^2 + 28 = 6*x + 20*y → ∃ r, r = 9 :=
by
  sorry

end cookie_radius_l45_45407


namespace abs_neg_2023_l45_45738

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l45_45738


namespace bill_has_6_less_pieces_than_mary_l45_45756

-- Definitions based on the conditions
def total_candy : ℕ := 20
def candy_kate : ℕ := 4
def candy_robert : ℕ := candy_kate + 2
def candy_mary : ℕ := candy_robert + 2
def candy_bill : ℕ := candy_kate - 2

-- Statement of the theorem
theorem bill_has_6_less_pieces_than_mary :
  candy_mary - candy_bill = 6 :=
sorry

end bill_has_6_less_pieces_than_mary_l45_45756


namespace proof_problem_l45_45493

-- Definitions of the conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) + f x = 0

def decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f y < f x

def satisfies_neq_point (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f a = 0

-- Main problem statement to prove (with conditions)
theorem proof_problem (f : ℝ → ℝ)
  (Hodd : odd_function f)
  (Hdec : decreasing_on f {y | 0 < y})
  (Hpt : satisfies_neq_point f (-2)) :
  {x : ℝ | (x - 1) * f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 1 < x ∧ x < 2} :=
sorry

end proof_problem_l45_45493


namespace series_sum_equals_four_l45_45245

/-- 
  Proof of the sum of the series: 
  ∑ (n=1 to ∞) (6n² - n + 1) / (n⁵ - n⁴ + n³ - n² + n) = 4 
--/
theorem series_sum_equals_four :
  (∑' n : ℕ, (if n > 0 then (6 * n^2 - n + 1 : ℝ) / (n^5 - n^4 + n^3 - n^2 + n) else 0)) = 4 :=
by
  sorry

end series_sum_equals_four_l45_45245


namespace conversion_base10_to_base7_l45_45995

-- Define the base-10 number
def num_base10 : ℕ := 1023

-- Define the conversion base
def base : ℕ := 7

-- Define the expected base-7 representation as a function of the base
def expected_base7 (b : ℕ) : ℕ := 2 * b^3 + 6 * b^2 + 6 * b^1 + 1 * b^0

-- Statement to prove
theorem conversion_base10_to_base7 : expected_base7 base = num_base10 :=
by 
  -- Sorry is a placeholder for the proof
  sorry

end conversion_base10_to_base7_l45_45995


namespace weight_problem_l45_45913

theorem weight_problem (w1 w2 w3 : ℝ) (h1 : w1 + w2 + w3 = 100)
  (h2 : w1 + 2 * w2 + w3 = 101) (h3 : w1 + w2 + 2 * w3 = 102) : 
  w1 ≥ 90 ∨ w2 ≥ 90 ∨ w3 ≥ 90 :=
by
  sorry

end weight_problem_l45_45913


namespace number_of_outfits_l45_45472

-- Definitions based on conditions a)
def trousers : ℕ := 5
def shirts : ℕ := 7
def jackets : ℕ := 3
def specific_trousers : ℕ := 2
def specific_jackets : ℕ := 2

-- Lean 4 theorem statement to prove the number of outfits
theorem number_of_outfits (trousers shirts jackets specific_trousers specific_jackets : ℕ) :
  (3 * jackets + specific_trousers * specific_jackets) * shirts = 91 :=
by
  sorry

end number_of_outfits_l45_45472


namespace range_x1_x2_l45_45569

theorem range_x1_x2
  (x1 x2 x3 : ℝ)
  (hx3_le_x2 : x3 ≤ x2)
  (hx2_le_x1 : x2 ≤ x1)
  (hx_sum : x1 + x2 + x3 = 1)
  (hfx_sum : (x1^2) + (x2^2) + (x3^2) = 1) :
  (2 / 3 : ℝ) ≤ x1 + x2 ∧ x1 + x2 ≤ (4 / 3 : ℝ) :=
sorry

end range_x1_x2_l45_45569


namespace trig_expression_value_l45_45906

open Real

theorem trig_expression_value : 
  (2 * cos (10 * (π / 180)) - sin (20 * (π / 180))) / cos (20 * (π / 180)) = sqrt 3 :=
by
  -- Proof should go here
  sorry

end trig_expression_value_l45_45906


namespace mason_hotdogs_proof_mason_ate_15_hotdogs_l45_45960

-- Define the weights of the items.
def weight_hotdog := 2 -- in ounces
def weight_burger := 5 -- in ounces
def weight_pie := 10 -- in ounces

-- Define Noah's consumption
def noah_burgers := 8

-- Define the total weight of hotdogs Mason ate
def mason_hotdogs_weight := 30

-- Calculate the number of hotdogs Mason ate
def hotdogs_mason_ate := mason_hotdogs_weight / weight_hotdog

-- Calculate the number of pies Jacob ate
def jacob_pies := noah_burgers - 3

-- Given conditions
theorem mason_hotdogs_proof :
  mason_hotdogs_weight / weight_hotdog = 3 * (noah_burgers - 3) :=
by
  sorry

-- Proving the number of hotdogs Mason ate equals 15
theorem mason_ate_15_hotdogs :
  hotdogs_mason_ate = 15 :=
by
  sorry

end mason_hotdogs_proof_mason_ate_15_hotdogs_l45_45960


namespace speed_in_terms_of_time_l45_45281

variable (a b x : ℝ)

-- Conditions
def condition1 : Prop := 1000 = a * x
def condition2 : Prop := 833 = b * x

-- The theorem to prove
theorem speed_in_terms_of_time (h1 : condition1 a x) (h2 : condition2 b x) :
  a = 1000 / x ∧ b = 833 / x :=
by
  sorry

end speed_in_terms_of_time_l45_45281


namespace stream_speed_l45_45402

-- Define the conditions
def still_water_speed : ℝ := 15
def upstream_time_factor : ℕ := 2

-- Define the theorem
theorem stream_speed (t v : ℝ) (h : (still_water_speed + v) * t = (still_water_speed - v) * (upstream_time_factor * t)) : v = 5 :=
by
  sorry

end stream_speed_l45_45402


namespace perpendicular_lines_slope_l45_45630

theorem perpendicular_lines_slope :
  ∀ (a : ℚ), (∀ x y : ℚ, y = 3 * x + 5) 
  ∧ (∀ x y : ℚ, 4 * y + a * x = 8) →
  a = 4 / 3 :=
by
  intro a
  intro h
  sorry

end perpendicular_lines_slope_l45_45630


namespace degree_to_radian_l45_45416

theorem degree_to_radian (deg : ℝ) (h : deg = 50) : deg * (Real.pi / 180) = (5 / 18) * Real.pi :=
by
  -- placeholder for the proof
  sorry

end degree_to_radian_l45_45416


namespace total_money_from_selling_watermelons_l45_45740

-- Given conditions
def weight_of_one_watermelon : ℝ := 23
def price_per_pound : ℝ := 2
def number_of_watermelons : ℝ := 18

-- Statement to be proved
theorem total_money_from_selling_watermelons : 
  (weight_of_one_watermelon * price_per_pound) * number_of_watermelons = 828 := 
by 
  sorry

end total_money_from_selling_watermelons_l45_45740


namespace probability_of_green_ball_l45_45219

def total_balls : ℕ := 3 + 3 + 6
def green_balls : ℕ := 3

theorem probability_of_green_ball : (green_balls : ℚ) / total_balls = 1 / 4 :=
by
  sorry

end probability_of_green_ball_l45_45219


namespace geometric_series_sum_l45_45981

theorem geometric_series_sum (a r : ℚ) (h_a : a = 1) (h_r : r = 1 / 3) : 
  (∑' n : ℕ, a * r ^ n) = 3 / 2 := 
by
  sorry

end geometric_series_sum_l45_45981


namespace combined_water_leak_l45_45872

theorem combined_water_leak
  (largest_rate : ℕ)
  (medium_rate : ℕ)
  (smallest_rate : ℕ)
  (time_minutes : ℕ)
  (h1 : largest_rate = 3)
  (h2 : medium_rate = largest_rate / 2)
  (h3 : smallest_rate = medium_rate / 3)
  (h4 : time_minutes = 120) :
  largest_rate * time_minutes + medium_rate * time_minutes + smallest_rate * time_minutes = 600 := by
  sorry

end combined_water_leak_l45_45872


namespace calculate_expression_value_l45_45235

theorem calculate_expression_value (x y : ℚ) (hx : x = 4 / 7) (hy : y = 5 / 8) :
  (7 * x + 5 * y) / (70 * x * y) = 57 / 400 := by
  sorry

end calculate_expression_value_l45_45235


namespace neg_abs_neg_three_l45_45804

theorem neg_abs_neg_three : -|(-3)| = -3 := 
by
  sorry

end neg_abs_neg_three_l45_45804


namespace percentage_of_other_sales_l45_45341

theorem percentage_of_other_sales :
  let pensPercentage := 20
  let pencilsPercentage := 15
  let notebooksPercentage := 30
  let totalPercentage := 100
  totalPercentage - (pensPercentage + pencilsPercentage + notebooksPercentage) = 35 :=
by
  sorry

end percentage_of_other_sales_l45_45341


namespace fruit_count_l45_45755

theorem fruit_count :
  let limes_mike : ℝ := 32.5
  let limes_alyssa : ℝ := 8.25
  let limes_jenny_picked : ℝ := 10.8
  let limes_jenny_ate := limes_jenny_picked / 2
  let limes_jenny := limes_jenny_picked - limes_jenny_ate
  let plums_tom : ℝ := 14.5
  let plums_tom_ate : ℝ := 2.5
  let X := (limes_mike - limes_alyssa) + limes_jenny
  let Y := plums_tom - plums_tom_ate
  X = 29.65 ∧ Y = 12 :=
by {
  sorry
}

end fruit_count_l45_45755


namespace solve_system_of_equations_l45_45185

theorem solve_system_of_equations
  {a b c d x y z : ℝ}
  (h1 : x + y + z = 1)
  (h2 : a * x + b * y + c * z = d)
  (h3 : a^2 * x + b^2 * y + c^2 * z = d^2)
  (hne1 : a ≠ b)
  (hne2 : a ≠ c)
  (hne3 : b ≠ c) :
  x = (d - b) * (d - c) / ((a - b) * (a - c)) ∧
  y = (d - a) * (d - c) / ((b - a) * (b - c)) ∧
  z = (d - a) * (d - b) / ((c - a) * (c - b)) :=
sorry

end solve_system_of_equations_l45_45185


namespace solution_x_y_l45_45745

noncomputable def eq_values (x y : ℝ) := (
  x ≠ 0 ∧ x ≠ 1 ∧ y ≠ 0 ∧ y ≠ 3 ∧ (3/x + 2/y = 1/3)
)

theorem solution_x_y (x y : ℝ) (h : eq_values x y) : x = 9 * y / (y - 6) :=
sorry

end solution_x_y_l45_45745


namespace decreasing_function_range_l45_45785

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 1) * x + 7 * a - 2 else a ^ x

theorem decreasing_function_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (3 / 8 ≤ a ∧ a < 1 / 2) := 
by
  intro a
  sorry

end decreasing_function_range_l45_45785


namespace binomial_product_l45_45383

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := 
by 
  sorry

end binomial_product_l45_45383


namespace teachers_quit_before_lunch_percentage_l45_45503

variables (n_initial n_after_one_hour n_after_lunch n_quit_before_lunch : ℕ)

def initial_teachers : ℕ := 60
def teachers_after_one_hour (n_initial : ℕ) : ℕ := n_initial / 2
def teachers_after_lunch : ℕ := 21
def quit_before_lunch (n_after_one_hour n_after_lunch : ℕ) : ℕ := n_after_one_hour - n_after_lunch
def percentage_quit (n_quit_before_lunch n_after_one_hour : ℕ) : ℕ := (n_quit_before_lunch * 100) / n_after_one_hour

theorem teachers_quit_before_lunch_percentage :
  ∀ n_initial n_after_one_hour n_after_lunch n_quit_before_lunch,
  n_initial = initial_teachers →
  n_after_one_hour = teachers_after_one_hour n_initial →
  n_after_lunch = teachers_after_lunch →
  n_quit_before_lunch = quit_before_lunch n_after_one_hour n_after_lunch →
  percentage_quit n_quit_before_lunch n_after_one_hour = 30 := by 
    sorry

end teachers_quit_before_lunch_percentage_l45_45503


namespace number_of_shelves_l45_45850

-- Given conditions
def booksBeforeTrip : ℕ := 56
def booksBought : ℕ := 26
def avgBooksPerShelf : ℕ := 20
def booksLeftOver : ℕ := 2
def totalBooks : ℕ := booksBeforeTrip + booksBought

-- Statement to prove
theorem number_of_shelves :
  totalBooks - booksLeftOver = 80 →
  80 / avgBooksPerShelf = 4 := by
  intros h
  sorry

end number_of_shelves_l45_45850


namespace one_person_remains_dry_l45_45775

theorem one_person_remains_dry (n : ℕ) :
  ∃ (person_dry : ℕ -> Bool), (∀ i : ℕ, i < 2 * n + 1 -> person_dry i = tt) := 
sorry

end one_person_remains_dry_l45_45775


namespace total_chocolates_l45_45923

-- Definitions based on conditions
def chocolates_per_bag := 156
def number_of_bags := 20

-- Statement to prove
theorem total_chocolates : chocolates_per_bag * number_of_bags = 3120 :=
by
  -- skip the proof
  sorry

end total_chocolates_l45_45923


namespace arithmetic_sequence_sum_l45_45658

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
  (∀ n, S n = n * (a 1 + a n) / 2) →
  a 4 + a 8 = 4 →
  S 11 + a 6 = 24 :=
by
  intros a S h1 h2
  sorry

end arithmetic_sequence_sum_l45_45658


namespace gcd_lcm_ordering_l45_45583

theorem gcd_lcm_ordering (a b p q : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_a_gt_b : a > b) 
    (h_p_gcd : p = Nat.gcd a b) (h_q_lcm : q = Nat.lcm a b) : q ≥ a ∧ a > b ∧ b ≥ p :=
by
  sorry

end gcd_lcm_ordering_l45_45583


namespace vector_calculation_l45_45870

def vector_a : ℝ × ℝ := (1, -1)
def vector_b : ℝ × ℝ := (-1, 2)

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1 + v1.2 * v2.2)

theorem vector_calculation :
  (dot_product (vector_add (scalar_mult 2 vector_a) vector_b) vector_a) = 1 :=
by
  sorry

end vector_calculation_l45_45870


namespace sin_15_mul_sin_75_l45_45193

theorem sin_15_mul_sin_75 : Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 4 :=
by
  sorry

end sin_15_mul_sin_75_l45_45193


namespace isosceles_triangle_perimeter_l45_45737

theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 5) (h₂ : b = 10) :
  (a + b + b = 25) ∧ (a + a + b ≤ b → False) :=
by
  sorry

end isosceles_triangle_perimeter_l45_45737


namespace smallest_pos_int_greater_than_one_rel_prime_multiple_of_7_l45_45052

theorem smallest_pos_int_greater_than_one_rel_prime_multiple_of_7 (x : ℕ) :
  (x > 1) ∧ (gcd x 210 = 7) ∧ (7 ∣ x) → x = 49 :=
by {
  sorry
}

end smallest_pos_int_greater_than_one_rel_prime_multiple_of_7_l45_45052


namespace barley_percentage_is_80_l45_45635

variables (T C : ℝ) -- Total land and cleared land
variables (B : ℝ) -- Percentage of cleared land planted with barley

-- Given conditions
def cleared_land (T : ℝ) : ℝ := 0.9 * T
def total_land_approx : ℝ := 1000
def potato_land (C : ℝ) : ℝ := 0.1 * C
def tomato_land : ℝ := 90
def barley_percentage (C : ℝ) (B : ℝ) : Prop := C - (potato_land C) - tomato_land = (B / 100) * C

-- Theorem statement to prove
theorem barley_percentage_is_80 :
  cleared_land total_land_approx = 900 → barley_percentage 900 80 :=
by
  intros hC
  rw [cleared_land, total_land_approx] at hC
  simp [barley_percentage, potato_land, tomato_land]
  sorry

end barley_percentage_is_80_l45_45635


namespace snowfall_difference_l45_45011

def baldMountainSnowfallMeters : ℝ := 1.5
def billyMountainSnowfallMeters : ℝ := 3.5
def mountPilotSnowfallCentimeters : ℝ := 126
def cmPerMeter : ℝ := 100

theorem snowfall_difference :
  billyMountainSnowfallMeters * cmPerMeter + mountPilotSnowfallCentimeters - baldMountainSnowfallMeters * cmPerMeter = 326 :=
by
  sorry

end snowfall_difference_l45_45011


namespace correct_decision_box_l45_45220

theorem correct_decision_box (a b c : ℝ) (x : ℝ) : 
  x = a ∨ x = b → (x = b → b > a) →
  (c > x) ↔ (max (max a b) c = c) :=
by sorry

end correct_decision_box_l45_45220


namespace words_added_to_removed_ratio_l45_45357

-- Conditions in the problem
def Yvonnes_words : ℕ := 400
def Jannas_extra_words : ℕ := 150
def words_removed : ℕ := 20
def words_needed : ℕ := 1000 - 930

-- Definitions derived from the conditions
def Jannas_words : ℕ := Yvonnes_words + Jannas_extra_words
def total_words_before_editing : ℕ := Yvonnes_words + Jannas_words
def total_words_after_removal : ℕ := total_words_before_editing - words_removed
def words_added : ℕ := words_needed

-- The theorem we need to prove
theorem words_added_to_removed_ratio :
  (words_added : ℚ) / words_removed = 7 / 2 :=
sorry

end words_added_to_removed_ratio_l45_45357


namespace product_correct_l45_45432

/-- Define the number and the digit we're interested in -/
def num : ℕ := 564823
def digit : ℕ := 4

/-- Define a function to calculate the local value of the digit 4 in the number 564823 -/
def local_value (n : ℕ) (d : ℕ) := if d = 4 then 40000 else 0

/-- Define a function to calculate the absolute value, although it is trivial for natural numbers -/
def abs_value (d : ℕ) := d

/-- Define the product of local value and absolute value of 4 in 564823 -/
def product := local_value num digit * abs_value digit

/-- Theorem stating that the product is as specified in the problem -/
theorem product_correct : product = 160000 :=
by
  sorry

end product_correct_l45_45432


namespace value_of_expression_l45_45176

theorem value_of_expression (a b c d m : ℝ) (h1 : a = -b) (h2 : a ≠ 0) (h3 : c * d = 1) (h4 : |m| = 3) :
  m^2 - (-1) + |a + b| - c * d * m = 7 ∨ m^2 - (-1) + |a + b| - c * d * m = 13 :=
by
  sorry

end value_of_expression_l45_45176


namespace largest_possible_b_l45_45650

theorem largest_possible_b (b : ℝ) (h : (3 * b + 6) * (b - 2) = 9 * b) : b ≤ 4 := 
by {
  -- leaving the proof as an exercise, using 'sorry' to complete the statement
  sorry
}

end largest_possible_b_l45_45650


namespace functional_inequality_solution_l45_45941

theorem functional_inequality_solution {f : ℝ → ℝ} 
  (h : ∀ x y : ℝ, f (x * y) ≤ y * f (x) + f (y)) : 
  ∀ x : ℝ, f x = 0 :=
sorry

end functional_inequality_solution_l45_45941


namespace find_speed_of_stream_l45_45676

def distance : ℝ := 24
def total_time : ℝ := 5
def rowing_speed : ℝ := 10

def speed_of_stream (v : ℝ) : Prop :=
  distance / (rowing_speed - v) + distance / (rowing_speed + v) = total_time

theorem find_speed_of_stream : ∃ v : ℝ, speed_of_stream v ∧ v = 2 :=
by
  exists 2
  unfold speed_of_stream
  simp
  sorry -- This would be the proof part which is not required here

end find_speed_of_stream_l45_45676


namespace three_digit_number_mul_seven_results_638_l45_45059

theorem three_digit_number_mul_seven_results_638 (N : ℕ) 
  (hN1 : 100 ≤ N) 
  (hN2 : N < 1000)
  (hN3 : ∃ (x : ℕ), 7 * N = 1000 * x + 638) : N = 234 := 
sorry

end three_digit_number_mul_seven_results_638_l45_45059


namespace area_of_square_on_PS_l45_45627

-- Given parameters as conditions in the form of hypotheses
variables (PQ QR RS PS PR : ℝ)

-- Hypotheses based on problem conditions
def hypothesis1 : PQ^2 = 25 := sorry
def hypothesis2 : QR^2 = 49 := sorry
def hypothesis3 : RS^2 = 64 := sorry
def hypothesis4 : PR^2 = PQ^2 + QR^2 := sorry
def hypothesis5 : PS^2 = PR^2 - RS^2 := sorry

-- The main theorem we need to prove
theorem area_of_square_on_PS :
  PS^2 = 10 := 
by {
  sorry
}

end area_of_square_on_PS_l45_45627


namespace amount_borrowed_eq_4137_84_l45_45803

noncomputable def compound_interest (initial : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  initial * (1 + rate/100) ^ time

theorem amount_borrowed_eq_4137_84 :
  ∃ P : ℝ, 
    (compound_interest (compound_interest (compound_interest P 6 3) 8 4) 10 2 = 8110) 
    ∧ (P = 4137.84) :=
by
  sorry

end amount_borrowed_eq_4137_84_l45_45803


namespace a_values_condition_l45_45539

def is_subset (A B : Set ℝ) : Prop := ∀ x, x ∈ A → x ∈ B

theorem a_values_condition (a : ℝ) : 
  (2 * a + 1 ≤ 3 ∧ 3 * a - 5 ≤ 22 ∧ 2 * a + 1 ≤ 3 * a - 5) 
  ↔ (6 ≤ a ∧ a ≤ 9) :=
by 
  sorry

end a_values_condition_l45_45539


namespace minimize_blue_surface_l45_45053

noncomputable def fraction_blue_surface_area : ℚ := 1 / 8

theorem minimize_blue_surface
  (total_cubes : ℕ)
  (blue_cubes : ℕ)
  (green_cubes : ℕ)
  (edge_length : ℕ)
  (surface_area : ℕ)
  (blue_surface_area : ℕ)
  (fraction_blue : ℚ)
  (h1 : total_cubes = 64)
  (h2 : blue_cubes = 20)
  (h3 : green_cubes = 44)
  (h4 : edge_length = 4)
  (h5 : surface_area = 6 * edge_length^2)
  (h6 : blue_surface_area = 12)
  (h7 : fraction_blue = blue_surface_area / surface_area) :
  fraction_blue = fraction_blue_surface_area :=
by
  sorry

end minimize_blue_surface_l45_45053


namespace original_number_of_cards_l45_45700

-- Declare variables r and b as naturals representing the number of red and black cards, respectively.
variable (r b : ℕ)

-- Assume the probabilities given in the problem.
axiom prob_red : (r : ℝ) / (r + b) = 1 / 3
axiom prob_red_after_add : (r : ℝ) / (r + b + 4) = 1 / 4

-- Define the statement we need to prove.
theorem original_number_of_cards : r + b = 12 :=
by
  -- The proof steps would be here, but we'll use sorry to avoid implementing them.
  sorry

end original_number_of_cards_l45_45700


namespace common_rational_root_is_negative_non_integer_l45_45226

theorem common_rational_root_is_negative_non_integer 
    (a b c d e f g : ℤ)
    (p : ℚ)
    (h1 : 90 * p^4 + a * p^3 + b * p^2 + c * p + 15 = 0)
    (h2 : 15 * p^5 + d * p^4 + e * p^3 + f * p^2 + g * p + 90 = 0)
    (h3 : ¬ (∃ k : ℤ, p = k))
    (h4 : p < 0) : 
  p = -1 / 3 := 
sorry

end common_rational_root_is_negative_non_integer_l45_45226


namespace max_leap_years_in_200_years_l45_45859

theorem max_leap_years_in_200_years (leap_year_interval: ℕ) (span: ℕ) 
  (h1: leap_year_interval = 4) 
  (h2: span = 200) : 
  (span / leap_year_interval) = 50 := 
sorry

end max_leap_years_in_200_years_l45_45859


namespace amount_per_person_is_correct_l45_45496

-- Define the total amount and the number of people
def total_amount : ℕ := 2400
def number_of_people : ℕ := 9

-- State the main theorem to be proved
theorem amount_per_person_is_correct : total_amount / number_of_people = 266 := 
by sorry

end amount_per_person_is_correct_l45_45496


namespace inequality_of_weighted_squares_equality_conditions_of_weighted_squares_l45_45411

theorem inequality_of_weighted_squares
  (x y a b : ℝ)
  (h_sum : a + b = 1)
  (h_nonneg_a : a ≥ 0)
  (h_nonneg_b : b ≥ 0) :
  (a * x + b * y)^2 ≤ a * x^2 + b * y^2 :=
by sorry

theorem equality_conditions_of_weighted_squares
  (x y a b : ℝ)
  (h_sum : a + b = 1)
  (h_nonneg_a : a ≥ 0)
  (h_nonneg_b : b ≥ 0) :
  (a * x + b * y)^2 = a * x^2 + b * y^2
  ↔ (a = 0 ∨ b = 0 ∨ x = y) :=
by sorry

end inequality_of_weighted_squares_equality_conditions_of_weighted_squares_l45_45411


namespace remaining_balance_on_phone_card_l45_45953

theorem remaining_balance_on_phone_card (original_balance : ℝ) (cost_per_minute : ℝ) (call_duration : ℕ) :
  original_balance = 30 → cost_per_minute = 0.16 → call_duration = 22 →
  original_balance - (cost_per_minute * call_duration) = 26.48 :=
by
  intros
  sorry

end remaining_balance_on_phone_card_l45_45953


namespace consecutive_integers_divisible_product_l45_45558

theorem consecutive_integers_divisible_product (m n : ℕ) (h : m < n) :
  ∀ k : ℕ, ∃ i j : ℕ, i ≠ j ∧ k + i < k + n ∧ k + j < k + n ∧ (k + i) * (k + j) % (m * n) = 0 :=
by sorry

end consecutive_integers_divisible_product_l45_45558


namespace find_value_of_fraction_l45_45479

variable {x y : ℝ}

theorem find_value_of_fraction (h1 : x > 0) (h2 : y > x) (h3 : y > 0) (h4 : x / y + y / x = 3) : 
  (x + y) / (y - x) = Real.sqrt 5 := 
by sorry

end find_value_of_fraction_l45_45479


namespace time_to_cross_bridge_l45_45470

theorem time_to_cross_bridge (speed_km_hr : ℝ) (length_m : ℝ) (time_min : ℝ) :
  speed_km_hr = 5 → length_m = 1250 → time_min = length_m / (speed_km_hr * 1000 / 60) → time_min = 15 :=
by
  intros h_speed h_length h_time
  rw [h_speed, h_length] at h_time
  -- Since 5 km/hr * 1000 / 60 = 83.33 m/min,
  -- substituting into equation gives us 1250 / 83.33 ≈ 15.
  sorry

end time_to_cross_bridge_l45_45470


namespace consecutive_odd_integers_sum_l45_45708

theorem consecutive_odd_integers_sum (n : ℤ) (h : (n - 2) + (n + 2) = 150) : n = 75 := 
by
  sorry

end consecutive_odd_integers_sum_l45_45708


namespace Nina_now_l45_45093

def Lisa_age (l m n : ℝ) := l + m + n = 36
def Nina_age (l n : ℝ) := n - 5 = 2 * l
def Mike_age (l m : ℝ) := m + 2 = (l + 2) / 2

theorem Nina_now (l m n : ℝ) (h1 : Lisa_age l m n) (h2 : Nina_age l n) (h3 : Mike_age l m) : n = 34.6 := by
  sorry

end Nina_now_l45_45093


namespace prob_a_prob_b_l45_45746

def A (a : ℝ) := {x : ℝ | 0 < x + a ∧ x + a ≤ 5}
def B := {x : ℝ | -1/2 ≤ x ∧ x < 6}

theorem prob_a (a : ℝ) : (A a ⊆ B) → (-1 < a ∧ a ≤ 1/2) :=
sorry

theorem prob_b (a : ℝ) : (∃ x, A a ∩ B = {x}) → a = 11/2 :=
sorry

end prob_a_prob_b_l45_45746


namespace measure_of_angle_E_l45_45289

theorem measure_of_angle_E
    (A B C D E F : ℝ)
    (h1 : A = B)
    (h2 : B = C)
    (h3 : C = D)
    (h4 : E = F)
    (h5 : A = E - 30)
    (h6 : A + B + C + D + E + F = 720) :
  E = 140 :=
by
  -- Proof goes here
  sorry

end measure_of_angle_E_l45_45289


namespace simplify_expression_correct_l45_45515

variable (a b x y : ℝ) (i : ℂ)

noncomputable def simplify_expression (a b x y : ℝ) (i : ℂ) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0) ∧ (i^2 = -1) → (a * x + b * i * y) * (a * x - b * i * y) = a^2 * x^2 + b^2 * y^2

theorem simplify_expression_correct (a b x y : ℝ) (i : ℂ) :
  simplify_expression a b x y i := by
  sorry

end simplify_expression_correct_l45_45515


namespace driving_speed_ratio_l45_45601

theorem driving_speed_ratio
  (x : ℝ) (y : ℝ)
  (h1 : y = 2 * x) :
  y / x = 2 := by
  sorry

end driving_speed_ratio_l45_45601


namespace linda_spent_total_l45_45590

noncomputable def total_spent (notebooks_price_euro : ℝ) (notebooks_count : ℕ) 
    (pencils_price_pound : ℝ) (pencils_gift_card_pound : ℝ)
    (pens_price_yen : ℝ) (pens_points : ℝ) 
    (markers_price_dollar : ℝ) (calculator_price_dollar : ℝ)
    (marker_discount : ℝ) (coupon_discount : ℝ) (sales_tax : ℝ)
    (euro_to_dollar : ℝ) (pound_to_dollar : ℝ) (yen_to_dollar : ℝ) : ℝ :=
  let notebooks_cost := (notebooks_price_euro * notebooks_count) * euro_to_dollar
  let pencils_cost := 0
  let pens_cost := 0
  let marked_price := markers_price_dollar * (1 - marker_discount)
  let us_total_before_tax := (marked_price + calculator_price_dollar) * (1 - coupon_discount)
  let us_total_after_tax := us_total_before_tax * (1 + sales_tax)
  notebooks_cost + pencils_cost + pens_cost + us_total_after_tax

theorem linda_spent_total : 
  total_spent 1.2 3 1.5 5 170 200 2.8 12.5 0.15 0.10 0.05 1.1 1.25 0.009 = 18.0216 := 
  by
  sorry

end linda_spent_total_l45_45590


namespace lcm_100_40_is_200_l45_45570

theorem lcm_100_40_is_200 : Nat.lcm 100 40 = 200 := by
  sorry

end lcm_100_40_is_200_l45_45570


namespace find_x0_l45_45638

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3
noncomputable def g (x : ℝ) : ℝ := 3 * x - 5

theorem find_x0 :
  (∃ x0 : ℝ, f (g x0) = 1) → (∃ x0 : ℝ, x0 = 4/3) :=
by
  sorry

end find_x0_l45_45638


namespace rectangle_area_diff_l45_45838

theorem rectangle_area_diff :
  ∀ (l w : ℕ), (2 * l + 2 * w = 60) → (∃ A_max A_min : ℕ, 
    A_max = (l * (30 - l)) ∧ A_min = (min (1 * (30 - 1)) (29 * (30 - 29))) ∧ (A_max - A_min = 196)) :=
by
  intros l w h
  use 15 * 15, min (1 * 29) (29 * 1)
  sorry

end rectangle_area_diff_l45_45838


namespace cost_of_goods_l45_45082

theorem cost_of_goods
  (x y z : ℝ)
  (h1 : 3 * x + 7 * y + z = 315)
  (h2 : 4 * x + 10 * y + z = 420) :
  x + y + z = 105 :=
by
  sorry

end cost_of_goods_l45_45082


namespace domain_of_w_l45_45813

theorem domain_of_w :
  {x : ℝ | x + (x - 1)^(1/3) + (8 - x)^(1/3) ≥ 0} = {x : ℝ | x ≥ 0} :=
by {
  sorry
}

end domain_of_w_l45_45813


namespace width_of_metallic_sheet_l45_45660

-- Define the given conditions
def length_of_sheet : ℝ := 48
def side_of_square_cut : ℝ := 7
def volume_of_box : ℝ := 5236

-- Define the question as a Lean theorem
theorem width_of_metallic_sheet : ∃ (w : ℝ), w = 36 ∧
  volume_of_box = (length_of_sheet - 2 * side_of_square_cut) * (w - 2 * side_of_square_cut) * side_of_square_cut := by
  sorry

end width_of_metallic_sheet_l45_45660


namespace simplify_expression_calculate_difference_of_squares_l45_45687

section Problem1
variable (a b : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0)

theorem simplify_expression : ((-2 * a^2) ^ 2 * (-b^2)) / (4 * a^3 * b^2) = -a :=
by sorry
end Problem1

section Problem2

theorem calculate_difference_of_squares : 2023^2 - 2021 * 2025 = 4 :=
by sorry
end Problem2

end simplify_expression_calculate_difference_of_squares_l45_45687


namespace range_of_expression_l45_45305

theorem range_of_expression (x y : ℝ) 
  (h1 : x - 2 * y + 2 ≥ 0) 
  (h2 : x ≤ 1) 
  (h3 : x + y - 1 ≥ 0) : 
  3 / 2 ≤ (x + y + 2) / (x + 1) ∧ (x + y + 2) / (x + 1) ≤ 3 :=
by
  sorry

end range_of_expression_l45_45305


namespace smallest_n_with_divisors_2020_l45_45137

theorem smallest_n_with_divisors_2020 :
  ∃ n : ℕ, (∃ α1 α2 α3 : ℕ, 
  n = 2^α1 * 3^α2 * 5^α3 ∧
  (α1 + 1) * (α2 + 1) * (α3 + 1) = 2020) ∧
  n = 2^100 * 3^4 * 5 * 7 := by
  sorry

end smallest_n_with_divisors_2020_l45_45137


namespace quadratic_equation_iff_non_zero_coefficient_l45_45490

theorem quadratic_equation_iff_non_zero_coefficient (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + a * x - 3 = 0 → (a - 2) ≠ 0) ↔ a ≠ 2 :=
by
  sorry

end quadratic_equation_iff_non_zero_coefficient_l45_45490


namespace max_value_of_f_symmetric_about_point_concave_inequality_l45_45077

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 - x)

theorem max_value_of_f : ∃ x, f x = -4 :=
by
  sorry

theorem symmetric_about_point : ∀ x, f (1 - x) + f (1 + x) = -4 :=
by
  sorry

theorem concave_inequality (x1 x2 : ℝ) (h1 : x1 > 1) (h2 : x2 > 1) : 
  f ((x1 + x2) / 2) ≥ (f x1 + f x2) / 2 :=
by
  sorry

end max_value_of_f_symmetric_about_point_concave_inequality_l45_45077


namespace find_principal_l45_45996

theorem find_principal (A : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) (h1 : A = 1456) (h2 : R = 0.05) (h3 : T = 2.4) :
  A = P + P * R * T → P = 1300 :=
by {
  sorry
}

end find_principal_l45_45996


namespace scientists_speculation_reasonable_uranus_will_not_affect_earth_next_observation_l45_45739

-- Define the given conditions as constants and theorems in Lean
theorem scientists_speculation_reasonable : 
  ∃ (a b c : ℝ), 
  (64*a - 8*b + c = 32) ∧ 
  (36*a - 6*b + c = 28.5) ∧ 
  (16*a - 4*b + c = 26) ∧ 
  (∀ (x y : ℝ), (y = a*x^2 + b*x + c) → (x = 0) → y < 24.5) :=
by -- sorry is a placeholder for the proof
sorry

theorem uranus_will_not_affect_earth_next_observation : 
  ∃ (a b c : ℝ), 
  (64*a - 8*b + c = 32) ∧ 
  (36*a - 6*b + c = 28.5) ∧ 
  (16*a - 4*b + c = 26) ∧ 
  (∀ (x y : ℝ), (y = a*x^2 + b*x + c) → (x = 2) → y ≥ 24.5) :=
by -- sorry is a placeholder for the proof
sorry

end scientists_speculation_reasonable_uranus_will_not_affect_earth_next_observation_l45_45739


namespace min_buses_needed_l45_45973

theorem min_buses_needed (n : ℕ) (h1 : 45 * n ≥ 500) (h2 : n ≥ 2) : n = 12 :=
sorry

end min_buses_needed_l45_45973


namespace land_to_water_time_ratio_l45_45106

-- Define the conditions
def distance_water : ℕ := 50
def distance_land : ℕ := 300
def speed_ratio : ℕ := 3

-- Define the Lean theorem statement
theorem land_to_water_time_ratio (x : ℝ) (hx : x > 0) : 
  (distance_land / (speed_ratio * x)) / (distance_water / x) = 2 := by
  sorry

end land_to_water_time_ratio_l45_45106


namespace factor_expression_l45_45302

theorem factor_expression (x : ℝ) : (45 * x^3 - 135 * x^7) = 45 * x^3 * (1 - 3 * x^4) :=
by
  sorry

end factor_expression_l45_45302


namespace three_digit_number_second_digit_l45_45885

theorem three_digit_number_second_digit (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : a < 10) (h₂ : b < 10) (h₃ : c < 10) :
  (100 * a + 10 * b + c) - (a + b + c) = 261 → b = 7 :=
by sorry

end three_digit_number_second_digit_l45_45885


namespace fourth_grade_students_l45_45958

theorem fourth_grade_students:
  (initial_students = 35) →
  (first_semester_left = 6) →
  (first_semester_joined = 4) →
  (first_semester_transfers = 2) →
  (second_semester_left = 3) →
  (second_semester_joined = 7) →
  (second_semester_transfers = 2) →
  final_students = initial_students - first_semester_left + first_semester_joined - second_semester_left + second_semester_joined :=
  sorry

end fourth_grade_students_l45_45958


namespace simplify_expression_l45_45071

-- Defining the original expression
def original_expr (y : ℝ) : ℝ := 3 * y^3 - 7 * y^2 + 12 * y + 5 - (2 * y^3 - 4 + 3 * y^2 - 9 * y)

-- Defining the simplified expression
def simplified_expr (y : ℝ) : ℝ := y^3 - 10 * y^2 + 21 * y + 9

-- The statement to prove
theorem simplify_expression (y : ℝ) : original_expr y = simplified_expr y :=
by sorry

end simplify_expression_l45_45071


namespace line_tangent_to_ellipse_l45_45827

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = mx + 2 → x^2 + 9 * y^2 = 9 → ∃ u, y = u) → m^2 = 1 / 3 := 
by
  intro h
  sorry

end line_tangent_to_ellipse_l45_45827


namespace sum_of_powers_seven_l45_45768

theorem sum_of_powers_seven (α1 α2 α3 : ℂ)
  (h1 : α1 + α2 + α3 = 2)
  (h2 : α1^2 + α2^2 + α3^2 = 6)
  (h3 : α1^3 + α2^3 + α3^3 = 14) :
  α1^7 + α2^7 + α3^7 = 478 := by
  sorry

end sum_of_powers_seven_l45_45768


namespace monotonic_increasing_on_interval_l45_45657

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x - Real.pi / 6)

theorem monotonic_increasing_on_interval (ω : ℝ) (h1 : ω > 0) (h2 : 2 * Real.pi / (2 * ω) = 4 * Real.pi) :
  ∀ x y : ℝ, (x ∈ Set.Icc (Real.pi / 2) Real.pi) → (y ∈ Set.Icc (Real.pi / 2) Real.pi) → x ≤ y → f ω x ≤ f ω y := 
by
  sorry

end monotonic_increasing_on_interval_l45_45657


namespace sin_lt_alpha_lt_tan_l45_45001

open Real

theorem sin_lt_alpha_lt_tan {α : ℝ} (hα1 : 0 < α) (hα2 : α < π / 2) : sin α < α ∧ α < tan α := by
  sorry

end sin_lt_alpha_lt_tan_l45_45001


namespace correct_average_marks_l45_45217

theorem correct_average_marks :
  let num_students := 40
  let reported_avg := 65
  let incorrect_marks := [100, 85, 15]
  let correct_marks := [20, 50, 55]
  let incorrect_total_sum := num_students * reported_avg
  let wrong_sum := List.sum incorrect_marks
  let correct_sum := List.sum correct_marks
  let correct_total_sum := incorrect_total_sum - wrong_sum + correct_sum
  let correct_avg := (correct_total_sum : ℚ) / num_students
  correct_avg = 63.125 :=
by
  let num_students := 40
  let reported_avg := 65
  let incorrect_marks := [100, 85, 15]
  let correct_marks := [20, 50, 55]
  let incorrect_total_sum := num_students * reported_avg
  let wrong_sum := List.sum incorrect_marks
  let correct_sum := List.sum correct_marks
  let correct_total_sum := incorrect_total_sum - wrong_sum + correct_sum
  let correct_avg := (correct_total_sum : ℚ) / num_students
  sorry

end correct_average_marks_l45_45217


namespace evaluate_expression_l45_45172

theorem evaluate_expression :
  (2 / 10 + 3 / 100 + 5 / 1000 + 7 / 10000)^2 = 0.05555649 :=
by
  sorry

end evaluate_expression_l45_45172


namespace distance_between_islands_l45_45314

theorem distance_between_islands (AB : ℝ) (angle_BAC angle_ABC : ℝ) : 
  AB = 20 ∧ angle_BAC = 60 ∧ angle_ABC = 75 → 
  (∃ BC : ℝ, BC = 10 * Real.sqrt 6) := by
  intro h
  sorry

end distance_between_islands_l45_45314


namespace rest_area_milepost_l45_45835

theorem rest_area_milepost : 
  let fifth_exit := 30
  let fifteenth_exit := 210
  (3 / 5) * (fifteenth_exit - fifth_exit) + fifth_exit = 138 := 
by 
  let fifth_exit := 30
  let fifteenth_exit := 210
  sorry

end rest_area_milepost_l45_45835


namespace common_ratio_q_is_one_l45_45360

-- Define the geometric sequence {a_n}, and the third term a_3 and sum of first three terms S_3
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = a n * a 1

variables {a : ℕ → ℝ}
variable (q : ℝ)

-- Given conditions
axiom a_3 : a 3 = 3 / 2
axiom S_3 : a 1 * (1 + q + q^2) = 9 / 2

-- We need to prove q = 1
theorem common_ratio_q_is_one (h1 : is_geometric_sequence a) : q = 1 := sorry

end common_ratio_q_is_one_l45_45360


namespace Susan_total_peaches_l45_45513

-- Define the number of peaches in the knapsack
def peaches_in_knapsack : ℕ := 12

-- Define the condition that the number of peaches in the knapsack is half the number of peaches in each cloth bag
def peaches_per_cloth_bag (x : ℕ) : Prop := peaches_in_knapsack * 2 = x

-- Define the total number of peaches Susan bought
def total_peaches (x : ℕ) : ℕ := x + 2 * x

-- Theorem statement: Prove that the total number of peaches Susan bought is 60
theorem Susan_total_peaches (x : ℕ) (h : peaches_per_cloth_bag x) : total_peaches peaches_in_knapsack = 60 := by
  sorry

end Susan_total_peaches_l45_45513


namespace value_of_a_l45_45554

theorem value_of_a (a : ℝ) (A : ℝ × ℝ) (h : A = (1, 0)) : (a * A.1 + 3 * A.2 - 2 = 0) → a = 2 :=
by
  intro h1
  rw [h] at h1
  sorry

end value_of_a_l45_45554


namespace h_at_7_over_5_eq_0_l45_45373

def h (x : ℝ) : ℝ := 5 * x - 7

theorem h_at_7_over_5_eq_0 : h (7 / 5) = 0 := 
by 
  sorry

end h_at_7_over_5_eq_0_l45_45373


namespace polynomial_abs_sum_l45_45259

theorem polynomial_abs_sum (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) :
  (1 - (2:ℝ) * x) ^ 8 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 →
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| = (3:ℝ) ^ 8 :=
sorry

end polynomial_abs_sum_l45_45259


namespace boats_distance_one_minute_before_collision_l45_45123

theorem boats_distance_one_minute_before_collision :
  let speedA := 5  -- miles/hr
  let speedB := 21 -- miles/hr
  let initial_distance := 20 -- miles
  let combined_speed := speedA + speedB -- combined speed in miles/hr
  let speed_per_minute := combined_speed / 60 -- convert to miles/minute
  let time_to_collision := initial_distance / speed_per_minute -- time in minutes until collision
  initial_distance - (time_to_collision - 1) * speed_per_minute = 0.4333 :=
by
  sorry

end boats_distance_one_minute_before_collision_l45_45123


namespace cabinets_ratio_proof_l45_45882

-- Definitions for the conditions
def initial_cabinets : ℕ := 3
def total_cabinets : ℕ := 26
def additional_cabinets : ℕ := 5
def number_of_counters : ℕ := 3

-- Definition for the unknown cabinets installed per counter
def cabinets_per_counter : ℕ := (total_cabinets - additional_cabinets - initial_cabinets) / number_of_counters

-- The ratio to be proven
theorem cabinets_ratio_proof : (cabinets_per_counter : ℚ) / initial_cabinets = 2 / 1 :=
by
  -- Proof goes here
  sorry

end cabinets_ratio_proof_l45_45882


namespace scheduling_arrangements_l45_45405

-- We want to express this as a problem to prove the number of scheduling arrangements.

theorem scheduling_arrangements (n : ℕ) (h : n = 6) :
  (Nat.choose 6 1) * (Nat.choose 5 1) * (Nat.choose 4 2) = 180 := by
  sorry

end scheduling_arrangements_l45_45405


namespace minimum_value_of_f_l45_45175

noncomputable def f (x y z : ℝ) : ℝ := x^2 + 2 * y^2 + 3 * z^2 + 2 * x * y + 4 * y * z + 2 * z * x - 6 * x - 10 * y - 12 * z

theorem minimum_value_of_f : ∃ x y z : ℝ, f x y z = -14 :=
by
  sorry

end minimum_value_of_f_l45_45175


namespace ratio_bananas_dates_l45_45178

theorem ratio_bananas_dates (s c b d a : ℕ)
  (h1 : s = 780)
  (h2 : c = 60)
  (h3 : b = 3 * c)
  (h4 : a = 2 * d)
  (h5 : s = a + b + c + d) :
  b / d = 1 :=
by sorry

end ratio_bananas_dates_l45_45178


namespace bathroom_area_is_50_square_feet_l45_45681

/-- A bathroom has 10 6-inch tiles along its width and 20 6-inch tiles along its length. --/
def bathroom_width_inches := 10 * 6
def bathroom_length_inches := 20 * 6

/-- Convert width and length from inches to feet. --/
def bathroom_width_feet := bathroom_width_inches / 12
def bathroom_length_feet := bathroom_length_inches / 12

/-- Calculate the square footage of the bathroom. --/
def bathroom_square_footage := bathroom_width_feet * bathroom_length_feet

/-- The square footage of the bathroom is 50 square feet. --/
theorem bathroom_area_is_50_square_feet : bathroom_square_footage = 50 := by
  sorry

end bathroom_area_is_50_square_feet_l45_45681


namespace composite_sum_l45_45453

theorem composite_sum (x y n : ℕ) (hx : x > 1) (hy : y > 1) (h : x^2 + x * y - y = n^2) :
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = x + y + 1 :=
sorry

end composite_sum_l45_45453


namespace find_d_l45_45930

theorem find_d 
  (d : ℝ)
  (d_gt_zero : d > 0)
  (line_eq : ∀ x : ℝ, (2 * x - 6 = 0) → x = 3)
  (y_intercept : ∀ y : ℝ, (2 * 0 - 6 = y) → y = -6)
  (area_condition : (1/2 * 3 * 6 = 9) → (1/2 * (d - 3) * (2 * d - 6) = 36)) :
  d = 9 :=
sorry

end find_d_l45_45930


namespace equal_expression_exists_l45_45854

-- lean statement for the mathematical problem
theorem equal_expression_exists (a b : ℤ) :
  ∃ (expr : ℤ), expr = 20 * a - 18 * b := by
  sorry

end equal_expression_exists_l45_45854


namespace derivative_at_0_eq_6_l45_45776

-- Definition of the function
def f (x : ℝ) : ℝ := (2 * x + 1)^3

-- Theorem statement indicating the derivative at x = 0 is 6
theorem derivative_at_0_eq_6 : (deriv f 0) = 6 := 
by 
  sorry -- The proof is omitted as per the instructions

end derivative_at_0_eq_6_l45_45776


namespace ink_percentage_left_l45_45790

def area_of_square (side: ℕ) := side * side
def area_of_rectangle (length: ℕ) (width: ℕ) := length * width
def total_area_marker_can_paint (num_squares: ℕ) (square_side: ℕ) :=
  num_squares * area_of_square square_side
def total_area_colored (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ) :=
  num_rectangles * area_of_rectangle rect_length rect_width

def fraction_of_ink_used (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ)
  (num_squares: ℕ) (square_side: ℕ) :=
  (total_area_colored num_rectangles rect_length rect_width : ℚ)
    / (total_area_marker_can_paint num_squares square_side : ℚ)

def percentage_ink_left (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ)
  (num_squares: ℕ) (square_side: ℕ) :=
  100 * (1 - fraction_of_ink_used num_rectangles rect_length rect_width num_squares square_side)

theorem ink_percentage_left :
  percentage_ink_left 2 6 2 3 4 = 50 := by
  sorry

end ink_percentage_left_l45_45790


namespace problem_statement_l45_45856

noncomputable def increase_and_subtract (x p y : ℝ) : ℝ :=
  (x + p * x) - y

theorem problem_statement : increase_and_subtract 75 1.5 40 = 147.5 := by
  sorry

end problem_statement_l45_45856


namespace seating_arrangement_correct_l45_45677

-- Define the number of seating arrangements based on the given conditions

def seatingArrangements : Nat := 
  2 * 4 * 6

theorem seating_arrangement_correct :
  seatingArrangements = 48 := by
  sorry

end seating_arrangement_correct_l45_45677


namespace train_takes_longer_l45_45162

-- Definitions for the conditions
def train_speed : ℝ := 48
def ship_speed : ℝ := 60
def distance : ℝ := 480

-- Theorem statement for the proof
theorem train_takes_longer : (distance / train_speed) - (distance / ship_speed) = 2 := by
  sorry

end train_takes_longer_l45_45162


namespace derivative_at_minus_one_l45_45814
open Real

def f (x : ℝ) : ℝ := (1 + x) * (2 + x^2)^(1 / 2) * (3 + x^3)^(1 / 3)

theorem derivative_at_minus_one : deriv f (-1) = sqrt 3 * 2^(1 / 3) :=
by sorry

end derivative_at_minus_one_l45_45814


namespace total_ladybugs_l45_45936

theorem total_ladybugs (ladybugs_with_spots ladybugs_without_spots : ℕ) 
  (h1 : ladybugs_with_spots = 12170) 
  (h2 : ladybugs_without_spots = 54912) : 
  ladybugs_with_spots + ladybugs_without_spots = 67082 := 
by
  sorry

end total_ladybugs_l45_45936


namespace sqrt_expression_eq_neg_one_l45_45543

theorem sqrt_expression_eq_neg_one : 
  Real.sqrt ((-2)^2) + (Real.sqrt 3)^2 - (Real.sqrt 12 * Real.sqrt 3) = -1 :=
sorry

end sqrt_expression_eq_neg_one_l45_45543


namespace profit_divided_equally_l45_45516

noncomputable def Mary_investment : ℝ := 800
noncomputable def Mike_investment : ℝ := 200
noncomputable def total_profit : ℝ := 2999.9999999999995
noncomputable def Mary_extra : ℝ := 1200

theorem profit_divided_equally (E : ℝ) : 
  (E / 2 + 4 / 5 * (total_profit - E)) - (E / 2 + 1 / 5 * (total_profit - E)) = Mary_extra →
  E = 1000 :=
  by sorry

end profit_divided_equally_l45_45516


namespace correct_quotient_is_243_l45_45308

-- Define the given conditions
def mistaken_divisor : ℕ := 121
def mistaken_quotient : ℕ := 432
def correct_divisor : ℕ := 215
def remainder : ℕ := 0

-- Calculate the dividend based on mistaken values
def dividend : ℕ := mistaken_divisor * mistaken_quotient + remainder

-- State the theorem for the correct quotient
theorem correct_quotient_is_243
  (h_dividend : dividend = mistaken_divisor * mistaken_quotient + remainder)
  (h_divisible : dividend % correct_divisor = remainder) :
  dividend / correct_divisor = 243 :=
sorry

end correct_quotient_is_243_l45_45308


namespace find_real_numbers_l45_45725

theorem find_real_numbers (x1 x2 x3 x4 : ℝ) :
  x1 + x2 * x3 * x4 = 2 →
  x2 + x1 * x3 * x4 = 2 →
  x3 + x1 * x2 * x4 = 2 →
  x4 + x1 * x2 * x3 = 2 →
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨ 
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = 3 ∧ x4 = -1) ∨
  (x1 = -1 ∧ x2 = 3 ∧ x3 = -1 ∧ x4 = -1) ∨
  (x1 = 3 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = -1) :=
by sorry

end find_real_numbers_l45_45725


namespace reduced_travel_time_l45_45216

-- Definition of conditions as given in part a)
def initial_speed := 48 -- km/h
def initial_time := 50/60 -- hours (50 minutes)
def required_speed := 60 -- km/h
def reduced_time := 40/60 -- hours (40 minutes)

-- Problem statement
theorem reduced_travel_time :
  ∃ t2, (initial_speed * initial_time = required_speed * t2) ∧ (t2 = reduced_time) :=
by
  sorry

end reduced_travel_time_l45_45216


namespace married_fraction_l45_45897

variable (total_people : ℕ) (fraction_women : ℚ) (max_unmarried_women : ℕ)
variable (fraction_married : ℚ)

theorem married_fraction (h1 : total_people = 80)
                         (h2 : fraction_women = 1/4)
                         (h3 : max_unmarried_women = 20)
                         : fraction_married = 3/4 :=
by
  sorry

end married_fraction_l45_45897


namespace no_third_quadrant_l45_45532

theorem no_third_quadrant {a b : ℝ} (h1 : 0 < a) (h2 : a < 1) (h3 : -1 < b) : ∀ x y : ℝ, (y = a^x + b) → ¬ (x < 0 ∧ y < 0) :=
by
  intro x y h
  sorry

end no_third_quadrant_l45_45532


namespace tan_7pi_over_4_eq_neg1_l45_45641

theorem tan_7pi_over_4_eq_neg1 : Real.tan (7 * Real.pi / 4) = -1 :=
  sorry

end tan_7pi_over_4_eq_neg1_l45_45641


namespace option_A_option_B_option_D_l45_45665

-- Given real numbers a, b, c such that a > b > 1 and c > 0,
-- prove the following inequalities.
variables {a b c : ℝ}

-- Assume the conditions
axiom H1 : a > b
axiom H2 : b > 1
axiom H3 : c > 0

-- Statements to prove
theorem option_A (H1: a > b) (H2: b > 1) (H3: c > 0) : a^2 - bc > b^2 - ac := sorry
theorem option_B (H1: a > b) (H2: b > 1) : a^3 > b^2 := sorry
theorem option_D (H1: a > b) (H2: b > 1) : a + (1/a) > b + (1/b) := sorry
  
end option_A_option_B_option_D_l45_45665


namespace right_triangle_hypotenuse_l45_45502

theorem right_triangle_hypotenuse {a b c : ℝ} 
  (h1: a + b + c = 60) 
  (h2: a * b = 96) 
  (h3: a^2 + b^2 = c^2) : 
  c = 28.4 := 
sorry

end right_triangle_hypotenuse_l45_45502


namespace total_genuine_purses_and_handbags_l45_45146

def TirzahPurses : ℕ := 26
def TirzahHandbags : ℕ := 24
def FakePurses : ℕ := TirzahPurses / 2
def FakeHandbags : ℕ := TirzahHandbags / 4
def GenuinePurses : ℕ := TirzahPurses - FakePurses
def GenuineHandbags : ℕ := TirzahHandbags - FakeHandbags

theorem total_genuine_purses_and_handbags : GenuinePurses + GenuineHandbags = 31 := by
  sorry

end total_genuine_purses_and_handbags_l45_45146


namespace find_number_l45_45227

variable (x : ℝ)

theorem find_number (h : 2 * x - 6 = (1/4) * x + 8) : x = 8 :=
sorry

end find_number_l45_45227


namespace production_line_B_units_l45_45965

theorem production_line_B_units {x y z : ℕ} (h1 : x + y + z = 24000) (h2 : 2 * y = x + z) : y = 8000 :=
sorry

end production_line_B_units_l45_45965


namespace village_male_population_l45_45851

theorem village_male_population (total_population parts male_parts : ℕ) (h1 : total_population = 600) (h2 : parts = 4) (h3 : male_parts = 2) :
  male_parts * (total_population / parts) = 300 :=
by
  -- We are stating the problem as per the given conditions
  sorry

end village_male_population_l45_45851


namespace find_n_l45_45332

theorem find_n : ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 7615 [MOD 15] ∧ n = 10 := by
  use 10
  repeat { sorry }

end find_n_l45_45332


namespace shahrazad_stories_not_power_of_two_l45_45994

theorem shahrazad_stories_not_power_of_two :
  ∀ (a b c : ℕ) (k : ℕ),
  a + b + c = 1001 → 27 * a + 14 * b + c = 2^k → False :=
by {
  sorry
}

end shahrazad_stories_not_power_of_two_l45_45994


namespace find_constant_a_range_of_f_l45_45801

noncomputable def f (a x : ℝ) : ℝ :=
  2 * a * (Real.sin x)^2 + 2 * (Real.sin x) * (Real.cos x) - a

theorem find_constant_a (h : f a 0 = -Real.sqrt 3) : a = Real.sqrt 3 := by
  sorry

theorem range_of_f (a : ℝ) (h : a = Real.sqrt 3) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  f a x ∈ Set.Icc (-Real.sqrt 3) 2 := by
  sorry

end find_constant_a_range_of_f_l45_45801


namespace school_trip_seat_count_l45_45469

theorem school_trip_seat_count :
  ∀ (classrooms students_per_classroom seats_per_bus : ℕ),
  classrooms = 87 →
  students_per_classroom = 58 →
  seats_per_bus = 29 →
  ∀ (total_students total_buses_needed : ℕ),
  total_students = classrooms * students_per_classroom →
  total_buses_needed = (total_students + seats_per_bus - 1) / seats_per_bus →
  seats_per_bus = 29 := by
  intros classrooms students_per_classroom seats_per_bus
  intros h1 h2 h3
  intros total_students total_buses_needed
  intros h4 h5
  sorry

end school_trip_seat_count_l45_45469


namespace determinant_scaled_l45_45135

theorem determinant_scaled
  (x y z w : ℝ)
  (h : x * w - y * z = 10) :
  (3 * x) * (3 * w) - (3 * y) * (3 * z) = 90 :=
by sorry

end determinant_scaled_l45_45135


namespace axis_of_symmetry_range_of_m_l45_45067

/-- The conditions given in the original mathematical problem -/
noncomputable def f (x : ℝ) : ℝ :=
  let OA := (2 * Real.cos x, Real.sqrt 3)
  let OB := (Real.sin x + Real.sqrt 3 * Real.cos x, -1)
  (OA.1 * OB.1 + OA.2 * OB.2) + 2

/-- Question 1: The axis of symmetry for the function f(x) -/
theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, (2 * x + Real.pi / 3 = Real.pi / 2 + k * Real.pi) ↔ (x = k * Real.pi / 2 + Real.pi / 12) :=
sorry

/-- Question 2: The range of m such that g(x) = f(x) + m has zero points for x in (0, π/2) -/
theorem range_of_m (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (∃ c : ℝ, (f x + c = 0)) ↔ ( -4 ≤ c ∧ c < Real.sqrt 3 - 2) :=
sorry

end axis_of_symmetry_range_of_m_l45_45067


namespace problem_solution_l45_45152

open Set

-- Define the universal set U
def U : Set ℝ := univ

-- Define the set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define the set N using the given condition
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- Define the complement of N in U
def complement_N : Set ℝ := U \ N

-- Define the intersection of M and the complement of N
def result_set : Set ℝ := M ∩ complement_N

-- Prove the desired result
theorem problem_solution : result_set = {x | -2 ≤ x ∧ x < 0} :=
sorry

end problem_solution_l45_45152


namespace projectile_hits_ground_at_5_over_2_l45_45058

theorem projectile_hits_ground_at_5_over_2 :
  ∃ t : ℚ, (-20) * t ^ 2 + 26 * t + 60 = 0 ∧ t = 5 / 2 :=
sorry

end projectile_hits_ground_at_5_over_2_l45_45058


namespace third_generation_tail_length_is_25_l45_45985

def first_generation_tail_length : ℝ := 16
def growth_rate : ℝ := 0.25

def second_generation_tail_length : ℝ := first_generation_tail_length * (1 + growth_rate)
def third_generation_tail_length : ℝ := second_generation_tail_length * (1 + growth_rate)

theorem third_generation_tail_length_is_25 :
  third_generation_tail_length = 25 := by
  sorry

end third_generation_tail_length_is_25_l45_45985


namespace suzhou_metro_scientific_notation_l45_45100

theorem suzhou_metro_scientific_notation : 
  (∃(a : ℝ) (n : ℤ), 
    1 ≤ abs a ∧ abs a < 10 ∧ 15.6 * 10^9 = a * 10^n) → 
    (a = 1.56 ∧ n = 9) := 
by
  sorry

end suzhou_metro_scientific_notation_l45_45100


namespace half_angle_in_first_or_third_quadrant_l45_45710

noncomputable 
def angle_in_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < (2 * k + 1) * Real.pi / 2

noncomputable 
def angle_in_first_or_third_quadrant (β : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi < β ∧ β < (k + 1/4) * Real.pi ∨
  ∃ i : ℤ, (2 * i + 1) * Real.pi < β ∧ β < (2 * i + 5/4) * Real.pi 

theorem half_angle_in_first_or_third_quadrant (α : ℝ) (h : angle_in_first_quadrant α) :
  angle_in_first_or_third_quadrant (α / 2) :=
  sorry

end half_angle_in_first_or_third_quadrant_l45_45710


namespace carrie_hours_per_day_l45_45091

theorem carrie_hours_per_day (h : ℕ) 
  (worked_4_days : ∀ n, n = 4 * h) 
  (paid_per_hour : ℕ := 22)
  (cost_of_supplies : ℕ := 54)
  (profit : ℕ := 122) :
  88 * h - cost_of_supplies = profit → h = 2 := 
by 
  -- Assume problem conditions and solve
  sorry

end carrie_hours_per_day_l45_45091


namespace smallest_possible_value_of_M_l45_45823

theorem smallest_possible_value_of_M :
  ∀ (a b c d e f : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 →
  a + b + c + d + e + f = 4020 →
  (∃ M : ℕ, M = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) ∧
    (∀ (M' : ℕ), (∀ (a b c d e f : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 →
      a + b + c + d + e + f = 4020 →
      M' = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) → M' ≥ 804) → M = 804)) := by
  sorry

end smallest_possible_value_of_M_l45_45823


namespace arthur_walked_total_miles_l45_45849

def blocks_east := 8
def blocks_north := 15
def blocks_west := 3
def block_length := 1/2

def total_blocks := blocks_east + blocks_north + blocks_west
def total_miles := total_blocks * block_length

theorem arthur_walked_total_miles : total_miles = 13 := by
  sorry

end arthur_walked_total_miles_l45_45849


namespace arithmetic_seq_a10_l45_45312

variable (a : ℕ → ℚ)
variable (S : ℕ → ℚ)
variable (d : ℚ := 1)

def is_arithmetic_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = d

def sum_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem arithmetic_seq_a10 (h_seq : is_arithmetic_seq a d)
                          (h_sum : sum_first_n_terms a S)
                          (h_condition : S 8 = 4 * S 4) :
  a 10 = 19/2 := 
sorry

end arithmetic_seq_a10_l45_45312


namespace system_of_equations_is_B_l45_45336

-- Define the given conditions and correct answer
def condition1 (x y : ℝ) : Prop := 5 * x + y = 3
def condition2 (x y : ℝ) : Prop := x + 5 * y = 2
def correctAnswer (x y : ℝ) : Prop := 5 * x + y = 3 ∧ x + 5 * y = 2

theorem system_of_equations_is_B (x y : ℝ) : condition1 x y ∧ condition2 x y ↔ correctAnswer x y := by
  -- Proof goes here
  sorry

end system_of_equations_is_B_l45_45336


namespace monotonic_intervals_l45_45652

noncomputable def f (a x : ℝ) : ℝ := x^2 * Real.exp (a * x)

theorem monotonic_intervals (a : ℝ) :
  (a = 0 → (∀ x : ℝ, (x < 0 → f a x < f a (-1)) ∧ (x > 0 → f a x > f a 1))) ∧
  (a > 0 → (∀ x : ℝ, (x < -2 / a → f a x < f a (-2 / a - 1)) ∧ (x > 0 → f a x > f a 1) ∧ 
                  ((-2 / a) < x ∧ x < 0 → f a x < f a (-2 / a + 1)))) ∧
  (a < 0 → (∀ x : ℝ, (x < 0 → f a x < f a (-1)) ∧ (x > -2 / a → f a x < f a (-2 / a - 1)) ∧
                  (0 < x ∧ x < -2 / a → f a x > f a (-2 / a + 1))))
:= sorry

end monotonic_intervals_l45_45652


namespace memory_efficiency_problem_l45_45406

theorem memory_efficiency_problem (x : ℝ) (hx : x ≠ 0) :
  (100 / x - 100 / (1.2 * x) = 5 / 12) ↔ (100 / x - 100 / ((1 + 0.20) * x) = 5 / 12) :=
by sorry

end memory_efficiency_problem_l45_45406


namespace non_talking_birds_count_l45_45179

def total_birds : ℕ := 77
def talking_birds : ℕ := 64

theorem non_talking_birds_count : total_birds - talking_birds = 13 := by
  sorry

end non_talking_birds_count_l45_45179


namespace monotonically_increasing_sequence_l45_45200

theorem monotonically_increasing_sequence (k : ℝ) : (∀ n : ℕ+, n^2 + k * n < (n + 1)^2 + k * (n + 1)) ↔ k > -3 := by
  sorry

end monotonically_increasing_sequence_l45_45200


namespace monthly_production_increase_l45_45061

/-- A salt manufacturing company produced 3000 tonnes in January and increased its
    production by some tonnes every month over the previous month until the end
    of the year. Given that the average daily production was 116.71232876712328 tonnes,
    determine the monthly production increase. -/
theorem monthly_production_increase :
  let initial_production := 3000
  let daily_average_production := 116.71232876712328
  let days_per_year := 365
  let total_yearly_production := daily_average_production * days_per_year
  let months_per_year := 12
  ∃ (x : ℝ), total_yearly_production = (months_per_year / 2) * (2 * initial_production + (months_per_year - 1) * x) → x = 100 :=
sorry

end monthly_production_increase_l45_45061


namespace sum_of_remainders_l45_45646

theorem sum_of_remainders (a b c d : ℤ) (h1 : a % 53 = 33) (h2 : b % 53 = 25) (h3 : c % 53 = 6) (h4 : d % 53 = 12) : 
  (a + b + c + d) % 53 = 23 :=
by {
  sorry
}

end sum_of_remainders_l45_45646


namespace problem_statement_l45_45240

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x
noncomputable def F (x : ℝ) : ℝ := f x - g x
noncomputable def m (x x₀ : ℝ) : ℝ := if x ≤ x₀ then f x else g x

-- Statement of the theorem
theorem problem_statement (x₀ x₁ x₂ n : ℝ) (hx₀ : x₀ ∈ Set.Ioo 1 2)
  (hF_root : F x₀ = 0)
  (hm_roots : m x₁ x₀ = n ∧ m x₂ x₀ = n ∧ 1 < x₁ ∧ x₁ < x₀ ∧ x₀ < x₂) :
  x₁ + x₂ > 2 * x₀ :=
sorry

end problem_statement_l45_45240


namespace first_number_is_105_percent_of_second_kilograms_reduced_by_10_percent_l45_45113

-- Proof problem 1: Given a number is 5% more than another number
theorem first_number_is_105_percent_of_second (x y : ℚ) (h : x = y * 1.05) : x = y * (1 + 0.05) :=
by {
  -- proof here
  sorry
}

-- Proof problem 2: 10 kilograms reduced by 10%
theorem kilograms_reduced_by_10_percent (kg : ℚ) (h : kg = 10) : kg * (1 - 0.1) = 9 :=
by {
  -- proof here
  sorry
}

end first_number_is_105_percent_of_second_kilograms_reduced_by_10_percent_l45_45113


namespace scientific_notation_of_50000_l45_45297

theorem scientific_notation_of_50000 :
  50000 = 5 * 10^4 :=
sorry

end scientific_notation_of_50000_l45_45297


namespace perimeter_of_garden_l45_45041

-- Definitions based on conditions
def length : ℕ := 150
def breadth : ℕ := 150
def is_square (l b : ℕ) := l = b

-- Theorem statement proving the perimeter given conditions
theorem perimeter_of_garden : is_square length breadth → 4 * length = 600 :=
by
  intro h
  rw [h]
  norm_num
  sorry

end perimeter_of_garden_l45_45041


namespace part1_part2_l45_45131

variable (a : ℝ)
variable (x y : ℝ)
variable (P Q : ℝ × ℝ)

-- Part (1)
theorem part1 (hP : P = (2 * a - 2, a + 5)) (h_y : y = 0) : P = (-12, 0) :=
sorry

-- Part (2)
theorem part2 (hP : P = (2 * a - 2, a + 5)) (hQ : Q = (4, 5)) 
    (h_parallel : 2 * a - 2 = 4) : P = (4, 8) ∧ quadrant = "first" :=
sorry

end part1_part2_l45_45131


namespace weight_of_new_boy_l45_45414

theorem weight_of_new_boy (W : ℕ) (original_weight : ℕ) (total_new_weight : ℕ)
  (h_original_avg : original_weight = 5 * 35)
  (h_new_avg : total_new_weight = 6 * 36)
  (h_new_weight : total_new_weight = original_weight + W) :
  W = 41 := by
  sorry

end weight_of_new_boy_l45_45414


namespace students_in_each_class_l45_45063

theorem students_in_each_class (S : ℕ) 
  (h1 : 10 * S * 5 = 1750) : 
  S = 35 := 
by 
  sorry

end students_in_each_class_l45_45063


namespace marginal_cost_per_product_calculation_l45_45038

def fixed_cost : ℝ := 12000
def total_cost : ℝ := 16000
def num_products : ℕ := 20

theorem marginal_cost_per_product_calculation :
  (total_cost - fixed_cost) / num_products = 200 := by
  sorry

end marginal_cost_per_product_calculation_l45_45038


namespace John_cycles_distance_l45_45771

-- Define the rate and time as per the conditions in the problem
def rate : ℝ := 8 -- miles per hour
def time : ℝ := 2.25 -- hours

-- The mathematical statement to prove: distance = rate * time
theorem John_cycles_distance : rate * time = 18 := by
  sorry

end John_cycles_distance_l45_45771


namespace range_of_m_l45_45716

noncomputable def set_A (x : ℝ) : ℝ := x^2 - (3 / 2) * x + 1

def A : Set ℝ := {y | ∃ (x : ℝ), x ∈ (Set.Icc (-1/2 : ℝ) 2) ∧ y = set_A x}
def B (m : ℝ) : Set ℝ := {x : ℝ | x ≥ m + 1 ∨ x ≤ m - 1}

def sufficient_condition (m : ℝ) : Prop := A ⊆ B m

theorem range_of_m :
  {m : ℝ | sufficient_condition m} = {m | m ≤ -(9 / 16) ∨ m ≥ 3} :=
sorry

end range_of_m_l45_45716


namespace total_games_won_l45_45221

-- Define the number of games won by the Chicago Bulls
def bulls_games : ℕ := 70

-- Define the number of games won by the Miami Heat
def heat_games : ℕ := bulls_games + 5

-- Define the total number of games won by both the Bulls and the Heat
def total_games : ℕ := bulls_games + heat_games

-- The theorem stating that the total number of games won by both teams is 145
theorem total_games_won : total_games = 145 := by
  -- Proof is omitted
  sorry

end total_games_won_l45_45221


namespace percentage_deficit_for_second_side_l45_45722

-- Defining the given conditions and the problem statement
def side1_excess : ℚ := 0.14
def area_error : ℚ := 0.083
def original_length (L : ℚ) := L
def original_width (W : ℚ) := W
def measured_length_side1 (L : ℚ) := (1 + side1_excess) * L
def measured_width_side2 (W : ℚ) (x : ℚ) := W * (1 - 0.01 * x)
def original_area (L W : ℚ) := L * W
def calculated_area (L W x : ℚ) := 
  measured_length_side1 L * measured_width_side2 W x

theorem percentage_deficit_for_second_side (L W : ℚ) :
  (calculated_area L W 5) / (original_area L W) = 1 + area_error :=
by
  sorry

end percentage_deficit_for_second_side_l45_45722


namespace min_value_of_angle_function_l45_45266

theorem min_value_of_angle_function (α β γ : ℝ) (h1 : α + β + γ = Real.pi) (h2 : 0 < α) (h3 : α < Real.pi) :
  ∃ α, α = (2 * Real.pi / 3) ∧ (4 / α + 1 / (Real.pi - α)) = (9 / Real.pi) := by
  sorry

end min_value_of_angle_function_l45_45266


namespace binary_to_base5_l45_45256

theorem binary_to_base5 : Nat.digits 5 (Nat.ofDigits 2 [1, 0, 1, 1, 0, 0, 1]) = [4, 2, 3] :=
by
  sorry

end binary_to_base5_l45_45256


namespace adam_room_shelves_l45_45989

def action_figures_per_shelf : ℕ := 15
def total_action_figures : ℕ := 120
def total_shelves (total_figures shelves_capacity : ℕ) : ℕ := total_figures / shelves_capacity

theorem adam_room_shelves :
  total_shelves total_action_figures action_figures_per_shelf = 8 :=
by
  sorry

end adam_room_shelves_l45_45989


namespace ratio_expression_l45_45874

variable (a b c : ℚ)
variable (h1 : a / b = 6 / 5)
variable (h2 : b / c = 8 / 7)

theorem ratio_expression (a b c : ℚ) (h1 : a / b = 6 / 5) (h2 : b / c = 8 / 7) :
  (7 * a + 6 * b + 5 * c) / (7 * a - 6 * b + 5 * c) = 751 / 271 := by
  sorry

end ratio_expression_l45_45874


namespace number_of_quarters_l45_45494

-- Defining constants for the problem
def value_dime : ℝ := 0.10
def value_nickel : ℝ := 0.05
def value_penny : ℝ := 0.01
def value_quarter : ℝ := 0.25

-- Given conditions
def total_dimes : ℝ := 3
def total_nickels : ℝ := 4
def total_pennies : ℝ := 200
def total_amount : ℝ := 5.00

-- Theorem stating the number of quarters found
theorem number_of_quarters :
  (total_amount - (total_dimes * value_dime + total_nickels * value_nickel + total_pennies * value_penny)) / value_quarter = 10 :=
by
  sorry

end number_of_quarters_l45_45494


namespace bounded_f_l45_45889

theorem bounded_f (f : ℝ → ℝ) (h1 : ∀ x1 x2, |x1 - x2| ≤ 1 → |f x2 - f x1| ≤ 1)
  (h2 : f 0 = 1) : ∀ x, -|x| ≤ f x ∧ f x ≤ |x| + 2 := by
  sorry

end bounded_f_l45_45889


namespace sequence_sum_l45_45732

theorem sequence_sum (r z w : ℝ) (h1 : 4 * r = 1) (h2 : 256 * r = z) (h3 : z * r = w) : z + w = 80 :=
by
  -- Proceed with your proof here.
  -- sorry for skipping the proof part.
  sorry

end sequence_sum_l45_45732


namespace unique_pair_solution_l45_45051

theorem unique_pair_solution:
  ∃! (a n : ℕ) (h_pos_a : 0 < a) (h_pos_n : 0 < n), a^2 = 2^n + 15 ∧ a = 4 ∧ n = 0 := sorry

end unique_pair_solution_l45_45051


namespace ratio_of_speeds_is_2_l45_45674

-- Definitions based on conditions
def rate_of_machine_B : ℕ := 100 / 40 -- Rate of Machine B (parts per minute)
def rate_of_machine_A : ℕ := 50 / 10 -- Rate of Machine A (parts per minute)
def ratio_of_speeds (rate_A rate_B : ℕ) : ℕ := rate_A / rate_B -- Ratio of speeds

-- Proof statement
theorem ratio_of_speeds_is_2 : ratio_of_speeds rate_of_machine_A rate_of_machine_B = 2 := by
  sorry

end ratio_of_speeds_is_2_l45_45674


namespace least_boxes_l45_45263
-- Definitions and conditions
def isPerfectCube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

def isFactor (a b : ℕ) : Prop := ∃ k, a * k = b

def numBoxes (N boxSize : ℕ) : ℕ := N / boxSize

-- Specific conditions for our problem
theorem least_boxes (N : ℕ) (boxSize : ℕ) 
  (h1 : N ≠ 0) 
  (h2 : isPerfectCube N)
  (h3 : isFactor boxSize N)
  (h4 : boxSize = 45): 
  numBoxes N boxSize = 75 :=
by
  sorry

end least_boxes_l45_45263


namespace total_balls_l45_45705

theorem total_balls (colors : ℕ) (balls_per_color : ℕ) (h_colors : colors = 10) (h_balls_per_color : balls_per_color = 35) : 
    colors * balls_per_color = 350 :=
by
  -- Import necessary libraries
  sorry

end total_balls_l45_45705


namespace solve_quadratic_eq_l45_45304

theorem solve_quadratic_eq (x : ℝ) : x^2 = 4 * x → x = 0 ∨ x = 4 :=
by
  intro h
  sorry

end solve_quadratic_eq_l45_45304


namespace intersection_points_of_quadratic_minimum_value_of_quadratic_in_range_range_of_m_for_intersection_with_segment_PQ_l45_45455

-- Define the quadratic function
def quadratic (m x : ℝ) : ℝ := m * x^2 - 4 * m * x + 3 * m

-- Define the conditions
variables (m : ℝ)
theorem intersection_points_of_quadratic :
    (quadratic m 1 = 0) ∧ (quadratic m 3 = 0) ↔ m ≠ 0 :=
sorry

theorem minimum_value_of_quadratic_in_range :
    ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 4 → quadratic (-2) x ≥ -6 :=
sorry

theorem range_of_m_for_intersection_with_segment_PQ :
    ∀ (m : ℝ), (∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ quadratic m x = (m + 4) / 2) ↔ 
    m ≤ -4 / 3 ∨ m ≥ 4 / 5 :=
sorry

end intersection_points_of_quadratic_minimum_value_of_quadratic_in_range_range_of_m_for_intersection_with_segment_PQ_l45_45455


namespace temperature_conversion_l45_45107

theorem temperature_conversion :
  ∀ (k t : ℝ),
    (t = (5 / 9) * (k - 32) ∧ k = 95) →
    t = 35 := by
  sorry

end temperature_conversion_l45_45107


namespace smallest_two_digit_multiple_of_17_l45_45382

theorem smallest_two_digit_multiple_of_17 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ n % 17 = 0 ∧ ∀ m, (10 ≤ m ∧ m < n ∧ m % 17 = 0) → false := sorry

end smallest_two_digit_multiple_of_17_l45_45382


namespace percentage_increase_on_friday_l45_45443

theorem percentage_increase_on_friday (avg_books_per_day : ℕ) (friday_books : ℕ) (total_books_per_week : ℕ) (days_open : ℕ)
  (h1 : avg_books_per_day = 40)
  (h2 : total_books_per_week = 216)
  (h3 : days_open = 5)
  (h4 : friday_books > avg_books_per_day) :
  (((friday_books - avg_books_per_day) * 100) / avg_books_per_day) = 40 :=
sorry

end percentage_increase_on_friday_l45_45443


namespace system_solution_exists_l45_45367

theorem system_solution_exists (x y: ℝ) :
    (y^2 = (x + 8) * (x^2 + 2) ∧ y^2 - (8 + 4 * x) * y + (16 + 16 * x - 5 * x^2) = 0) → 
    ((x = 0 ∧ (y = 4 ∨ y = -4)) ∨ (x = -2 ∧ (y = 6 ∨ y = -6)) ∨ (x = 19 ∧ (y = 99 ∨ y = -99))) :=
    sorry

end system_solution_exists_l45_45367


namespace intersect_single_point_l45_45421

theorem intersect_single_point (k : ℝ) :
  (∃ x : ℝ, (x^2 + k * x + 1 = 0) ∧
   ∀ x y : ℝ, (x^2 + k * x + 1 = 0 → y^2 + k * y + 1 = 0 → x = y))
  ↔ (k = 2 ∨ k = -2) :=
by
  sorry

end intersect_single_point_l45_45421


namespace find_r_minus_p_l45_45044

-- Define the variables and conditions
variables (p q r A1 A2 : ℝ)
noncomputable def arithmetic_mean (x y : ℝ) := (x + y) / 2

-- Given conditions in the problem
axiom hA1 : arithmetic_mean p q = 10
axiom hA2 : arithmetic_mean q r = 25

-- Statement to prove
theorem find_r_minus_p : r - p = 30 :=
by {
  -- write the necessary proof steps here
  sorry
}

end find_r_minus_p_l45_45044


namespace combined_mean_score_l45_45605

-- Definitions based on the conditions
def mean_score_class1 : ℕ := 90
def mean_score_class2 : ℕ := 80
def ratio_students (n1 n2 : ℕ) : Prop := n1 / n2 = 2 / 3

-- Proof statement
theorem combined_mean_score (n1 n2 : ℕ) 
  (h1 : ratio_students n1 n2) 
  (h2 : mean_score_class1 = 90) 
  (h3 : mean_score_class2 = 80) : 
  ((mean_score_class1 * n1) + (mean_score_class2 * n2)) / (n1 + n2) = 84 := 
by
  sorry

end combined_mean_score_l45_45605


namespace pipes_fill_tank_in_8_hours_l45_45640

theorem pipes_fill_tank_in_8_hours (A B C : ℝ) (hA : A = 1 / 56) (hB : B = 2 * A) (hC : C = 2 * B) :
  1 / (A + B + C) = 8 :=
by
  sorry

end pipes_fill_tank_in_8_hours_l45_45640


namespace total_bike_clamps_given_away_l45_45138

-- Definitions for conditions
def bike_clamps_per_bike := 2
def bikes_sold_morning := 19
def bikes_sold_afternoon := 27

-- Theorem statement to be proven
theorem total_bike_clamps_given_away :
  bike_clamps_per_bike * bikes_sold_morning +
  bike_clamps_per_bike * bikes_sold_afternoon = 92 :=
by
  sorry -- Proof is to be filled in later

end total_bike_clamps_given_away_l45_45138


namespace truck_capacity_solution_l45_45504

variable (x y : ℝ)

theorem truck_capacity_solution (h1 : 3 * x + 4 * y = 22) (h2 : 2 * x + 6 * y = 23) :
  x + y = 6.5 := sorry

end truck_capacity_solution_l45_45504


namespace sum_of_roots_quadratic_specific_sum_of_roots_l45_45530

theorem sum_of_roots_quadratic:
  ∀ a b c : ℚ, a ≠ 0 → 
  ∀ x1 x2 : ℚ, (a * x1^2 + b * x1 + c = 0) ∧ 
               (a * x2^2 + b * x2 + c = 0) → 
               x1 + x2 = -b / a := 
by
  sorry

theorem specific_sum_of_roots:
  ∀ x1 x2 : ℚ, (12 * x1^2 + 19 * x1 - 21 = 0) ∧ 
               (12 * x2^2 + 19 * x2 - 21 = 0) → 
               x1 + x2 = -19 / 12 := 
by
  sorry

end sum_of_roots_quadratic_specific_sum_of_roots_l45_45530


namespace geometric_sequence_eighth_term_l45_45163

noncomputable def a_8 : ℕ :=
  let a₁ := 8
  let r := 2
  a₁ * r^(8-1)

theorem geometric_sequence_eighth_term : a_8 = 1024 := by
  sorry

end geometric_sequence_eighth_term_l45_45163


namespace repeat_decimal_to_fraction_l45_45588

theorem repeat_decimal_to_fraction : 0.36666 = 11 / 30 :=
by {
    sorry
}

end repeat_decimal_to_fraction_l45_45588


namespace more_girls_than_boys_l45_45806

variables (boys girls : ℕ)

def ratio_condition : Prop := (3 * girls = 4 * boys)
def total_students_condition : Prop := (boys + girls = 42)

theorem more_girls_than_boys (h1 : ratio_condition boys girls) (h2 : total_students_condition boys girls) :
  (girls - boys = 6) :=
sorry

end more_girls_than_boys_l45_45806


namespace pieces_per_pizza_is_five_l45_45752

-- Definitions based on the conditions
def cost_per_pizza (total_cost : ℕ) (number_of_pizzas : ℕ) : ℕ :=
  total_cost / number_of_pizzas

def number_of_pieces_per_pizza (cost_per_pizza : ℕ) (cost_per_piece : ℕ) : ℕ :=
  cost_per_pizza / cost_per_piece

-- Given conditions
def total_cost : ℕ := 80
def number_of_pizzas : ℕ := 4
def cost_per_piece : ℕ := 4

-- Prove
theorem pieces_per_pizza_is_five : number_of_pieces_per_pizza (cost_per_pizza total_cost number_of_pizzas) cost_per_piece = 5 :=
by sorry

end pieces_per_pizza_is_five_l45_45752


namespace find_income_l45_45330

-- Define the condition for savings
def savings_formula (income expenditure savings : ℝ) : Prop :=
  income - expenditure = savings

-- Define the ratio between income and expenditure
def ratio_condition (income expenditure : ℝ) : Prop :=
  income = 5 / 4 * expenditure

-- Given:
-- savings: Rs. 3400
-- We need to prove the income is Rs. 17000
theorem find_income (savings : ℝ) (income expenditure : ℝ) :
  savings_formula income expenditure savings →
  ratio_condition income expenditure →
  savings = 3400 →
  income = 17000 :=
sorry

end find_income_l45_45330


namespace distance_from_origin_is_correct_l45_45616

noncomputable def is_distance_8_from_x_axis (x y : ℝ) := y = 8
noncomputable def is_distance_12_from_point (x y : ℝ) := (x - 1)^2 + (y - 6)^2 = 144
noncomputable def x_greater_than_1 (x : ℝ) := x > 1
noncomputable def distance_from_origin (x y : ℝ) := Real.sqrt (x^2 + y^2)

theorem distance_from_origin_is_correct (x y : ℝ)
  (h1 : is_distance_8_from_x_axis x y)
  (h2 : is_distance_12_from_point x y)
  (h3 : x_greater_than_1 x) :
  distance_from_origin x y = Real.sqrt (205 + 2 * Real.sqrt 140) :=
by
  sorry

end distance_from_origin_is_correct_l45_45616


namespace minimum_value_expression_l45_45812

noncomputable def expression (a b c d : ℝ) : ℝ :=
  (a + b) / c + (a + c) / d + (b + d) / a + (c + d) / b

theorem minimum_value_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  expression a b c d ≥ 8 :=
by
  -- Proof goes here
  sorry

end minimum_value_expression_l45_45812


namespace meet_at_35_l45_45685

def walking_distance_A (t : ℕ) := 5 * t

def walking_distance_B (t : ℕ) := (t * (7 + t)) / 2

def total_distance (t : ℕ) := walking_distance_A t + walking_distance_B t

theorem meet_at_35 : ∃ (t : ℕ), total_distance t = 100 ∧ walking_distance_A t - walking_distance_B t = 35 := by
  sorry

end meet_at_35_l45_45685


namespace bridge_length_calculation_l45_45766

def length_of_bridge (train_length : ℕ) (train_speed_kmph : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_mps := (train_speed_kmph * 1000) / 3600
  let distance_covered := speed_mps * time_seconds
  distance_covered - train_length

theorem bridge_length_calculation :
  length_of_bridge 140 45 30 = 235 :=
by
  unfold length_of_bridge
  norm_num
  sorry

end bridge_length_calculation_l45_45766


namespace prove_f_neg_a_l45_45170

noncomputable def f (x : ℝ) : ℝ := x + 1/x - 1

theorem prove_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = -4 :=
by
  sorry

end prove_f_neg_a_l45_45170


namespace unique_real_root_count_l45_45283

theorem unique_real_root_count :
  ∃! x : ℝ, (x^12 + 1) * (x^10 + x^8 + x^6 + x^4 + x^2 + 1) = 12 * x^11 := by
  sorry

end unique_real_root_count_l45_45283


namespace number_of_ordered_triples_l45_45412

theorem number_of_ordered_triples :
  ∃ n, (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ Nat.lcm a b = 12 ∧ Nat.gcd b c = 6 ∧ Nat.lcm c a = 24) ∧ n = 4 :=
sorry

end number_of_ordered_triples_l45_45412


namespace blocks_left_l45_45730

theorem blocks_left (initial_blocks used_blocks : ℕ) (h_initial : initial_blocks = 59) (h_used : used_blocks = 36) : initial_blocks - used_blocks = 23 :=
by
  -- proof here
  sorry

end blocks_left_l45_45730


namespace absolute_value_half_angle_cosine_l45_45324

theorem absolute_value_half_angle_cosine (x : ℝ) (h1 : Real.sin x = -5 / 13) (h2 : ∀ n : ℤ, (2 * n) * Real.pi < x ∧ x < (2 * n + 1) * Real.pi) :
  |Real.cos (x / 2)| = Real.sqrt 26 / 26 :=
sorry

end absolute_value_half_angle_cosine_l45_45324


namespace area_of_triangle_is_correct_l45_45651

def vector := (ℝ × ℝ)

def a : vector := (7, 3)
def b : vector := (-1, 5)

noncomputable def det2x2 (v1 v2 : vector) : ℝ :=
  (v1.1 * v2.2) - (v1.2 * v2.1)

theorem area_of_triangle_is_correct :
  let area := (det2x2 a b) / 2
  area = 19 := by
  -- defintions and conditions are set here, proof skipped
  sorry

end area_of_triangle_is_correct_l45_45651


namespace min_blue_edges_l45_45224

def tetrahedron_min_blue_edges : ℕ := sorry

theorem min_blue_edges (edges_colored : ℕ → Bool) (face_has_blue_edge : ℕ → Bool) 
    (H1 : ∀ face, face_has_blue_edge face)
    (H2 : ∀ edge, face_has_blue_edge edge = True → edges_colored edge = True) : 
    tetrahedron_min_blue_edges = 2 := 
sorry

end min_blue_edges_l45_45224


namespace maximize_revenue_at_175_l45_45351

def price (x : ℕ) : ℕ :=
  if x ≤ 150 then 200 else 200 - (x - 150)

def revenue (x : ℕ) : ℕ :=
  price x * x

theorem maximize_revenue_at_175 :
  ∀ x : ℕ, revenue 175 ≥ revenue x := 
sorry

end maximize_revenue_at_175_l45_45351


namespace quadratic_roots_l45_45584

theorem quadratic_roots (a b : ℝ) (h : a^2 - 4*a*b + 5*b^2 - 2*b + 1 = 0) :
  ∃ (p q : ℝ), (∀ (x : ℝ), x^2 - p*x + q = 0 ↔ (x = a ∨ x = b)) ∧
               p = 3 ∧ q = 2 :=
by {
  sorry
}

end quadratic_roots_l45_45584


namespace isosceles_triangle_properties_l45_45672

/--
  An isosceles triangle has a base of 6 units and legs of 5 units each.
  Prove:
  1. The area of the triangle is 12 square units.
  2. The radius of the inscribed circle is 1.5 units.
-/
theorem isosceles_triangle_properties (base : ℝ) (legs : ℝ) 
  (h_base : base = 6) (h_legs : legs = 5) : 
  ∃ (area : ℝ) (inradius : ℝ), 
  area = 12 ∧ inradius = 1.5 
  :=
by
  sorry

end isosceles_triangle_properties_l45_45672


namespace actual_time_between_two_and_three_l45_45340

theorem actual_time_between_two_and_three (x y : ℕ) 
  (h1 : 2 ≤ x ∧ x < 3)
  (h2 : 60 * y + x = 60 * x + y - 55) : 
  x = 2 ∧ y = 5 + 5 / 11 := 
sorry

end actual_time_between_two_and_three_l45_45340


namespace exists_coprime_linear_combination_l45_45104

theorem exists_coprime_linear_combination (a b p : ℤ) :
  ∃ k l : ℤ, Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) :=
  sorry

end exists_coprime_linear_combination_l45_45104


namespace work_completion_l45_45271

theorem work_completion (p q : ℝ) (h1 : p = 1.60 * q) (h2 : (1 / p + 1 / q) = 1 / 16) : p = 1 / 26 := 
by {
  -- This will be followed by the proof steps, but we add sorry since only the statement is required
  sorry
}

end work_completion_l45_45271


namespace rainfall_march_correct_l45_45110

def rainfall_march : ℝ :=
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let average := 4
  let total_expected := 5 * average
  let total_april_to_july := april + may + june + july
  total_expected - total_april_to_july

theorem rainfall_march_correct (march_rainfall : ℝ) :
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let average := 4
  let total_expected := 5 * average
  let total_april_to_july := april + may + june + july
  march_rainfall = total_expected - total_april_to_july :=
by
  sorry

end rainfall_march_correct_l45_45110


namespace count_logical_propositions_l45_45086

def proposition_1 : Prop := ∃ d : ℕ, d = 1
def proposition_2 : Prop := ∀ n : ℕ, n % 10 = 0 → n % 5 = 0
def proposition_3 : Prop := ∀ t : Prop, t → ¬t

theorem count_logical_propositions :
  (proposition_1 ∧ proposition_3) →
  (proposition_1 ∧ proposition_2 ∧ proposition_3) →
  (∃ (n : ℕ), n = 10 ∧ n % 5 = 0) ∧ n = 2 :=
sorry

end count_logical_propositions_l45_45086


namespace simplify_fraction_when_b_equals_4_l45_45441

theorem simplify_fraction_when_b_equals_4 (b : ℕ) (h : b = 4) : (18 * b^4) / (27 * b^3) = 8 / 3 :=
by {
  -- we use the provided condition to state our theorem goals.
  sorry
}

end simplify_fraction_when_b_equals_4_l45_45441


namespace no_integer_solution_for_euler_conjecture_l45_45047

theorem no_integer_solution_for_euler_conjecture :
  ¬(∃ n : ℕ, 5^4 + 12^4 + 9^4 + 8^4 = n^4) :=
by
  -- Sum of the given fourth powers
  have lhs : ℕ := 5^4 + 12^4 + 9^4 + 8^4
  -- Direct proof skipped with sorry
  sorry

end no_integer_solution_for_euler_conjecture_l45_45047


namespace axis_of_symmetry_range_of_t_l45_45632

section
variables (a b m n p t : ℝ)

-- Assume the given conditions
def parabola (x : ℝ) : ℝ := a * x ^ 2 + b * x

-- Part (1): Find the axis of symmetry
theorem axis_of_symmetry (h_a_pos : a > 0) 
    (hM : parabola a b 2 = m) 
    (hN : parabola a b 4 = n) 
    (hmn : m = n) : 
    -b / (2 * a) = 3 := 
  sorry

-- Part (2): Find the range of values for t
theorem range_of_t (h_a_pos : a > 0) 
    (hP : parabola a b (-1) = p)
    (axis : -b / (2 * a) = t) 
    (hmn_neg : m * n < 0) 
    (hmpn : m < p ∧ p < n) :
    1 < t ∧ t < 3 / 2 := 
  sorry
end

end axis_of_symmetry_range_of_t_l45_45632


namespace feed_cost_l45_45991

theorem feed_cost (total_birds ducks_fraction chicken_feed_cost : ℕ) (h1 : total_birds = 15) (h2 : ducks_fraction = 1/3) (h3 : chicken_feed_cost = 2) :
  15 * (1 - 1/3) * 2 = 20 :=
by
  sorry

end feed_cost_l45_45991


namespace water_bottle_capacity_l45_45549

theorem water_bottle_capacity :
  (20 * 250 + 13 * 600) / 1000 = 12.8 := 
by
  sorry

end water_bottle_capacity_l45_45549


namespace jacob_find_more_l45_45418

theorem jacob_find_more :
  let initial_shells := 2
  let ed_limpet_shells := 7
  let ed_oyster_shells := 2
  let ed_conch_shells := 4
  let total_shells := 30
  let ed_shells := ed_limpet_shells + ed_oyster_shells + ed_conch_shells + initial_shells
  let jacob_shells := total_shells - ed_shells
  (jacob_shells - ed_limpet_shells - ed_oyster_shells - ed_conch_shells = 2) := 
by 
  sorry

end jacob_find_more_l45_45418


namespace range_of_a_l45_45450

def A : Set ℝ := { x | x^2 - 3 * x + 2 ≤ 0 }
def B (a : ℝ) : Set ℝ := { x | 1 / (x - 3) < a }

theorem range_of_a (a : ℝ) : A ⊆ B a ↔ a > -1/2 :=
by sorry

end range_of_a_l45_45450


namespace mona_drives_125_miles_l45_45446

/-- Mona can drive 125 miles with $25 worth of gas, given the car mileage
    and the cost per gallon of gas. -/
theorem mona_drives_125_miles (miles_per_gallon : ℕ) (cost_per_gallon : ℕ) (total_money : ℕ)
  (h_miles_per_gallon : miles_per_gallon = 25) (h_cost_per_gallon : cost_per_gallon = 5)
  (h_total_money : total_money = 25) :
  (total_money / cost_per_gallon) * miles_per_gallon = 125 :=
by
  sorry

end mona_drives_125_miles_l45_45446


namespace area_of_polygon_DEFG_l45_45095

-- Given conditions
def isosceles_triangle (A B C : Type) (AB AC BC : ℝ) : Prop :=
  AB = AC ∧ AB = 2 ∧ AC = 2 ∧ BC = 1

def square (side : ℝ) : ℝ :=
  side * side

def constructed_square_areas_equal (AB AC : ℝ) (D E F G : Type) : Prop :=
  square AB = square AC ∧ square AB = 4 ∧ square AC = 4

-- Question to prove
theorem area_of_polygon_DEFG (A B C D E F G : Type) (AB AC BC : ℝ) 
  (h1 : isosceles_triangle A B C AB AC BC) 
  (h2 : constructed_square_areas_equal AB AC D E F G) : 
  square AB + square AC = 8 :=
by
  sorry

end area_of_polygon_DEFG_l45_45095


namespace tan_alpha_sub_beta_l45_45322

theorem tan_alpha_sub_beta
  (α β : ℝ)
  (h1 : Real.tan (α + Real.pi / 5) = 2)
  (h2 : Real.tan (β - 4 * Real.pi / 5) = -3) :
  Real.tan (α - β) = -1 := 
sorry

end tan_alpha_sub_beta_l45_45322


namespace ratio_side_length_to_brush_width_l45_45625

theorem ratio_side_length_to_brush_width (s w : ℝ) (h1 : w = s / 4) (h2 : s^2 / 3 = w^2 + ((s - w)^2) / 2) :
    s / w = 4 := by
  sorry

end ratio_side_length_to_brush_width_l45_45625


namespace current_price_after_increase_and_decrease_l45_45323

-- Define constants and conditions
def initial_price_RAM : ℝ := 50
def percent_increase : ℝ := 0.30
def percent_decrease : ℝ := 0.20

-- Define intermediate and final values based on conditions
def increased_price_RAM : ℝ := initial_price_RAM * (1 + percent_increase)
def final_price_RAM : ℝ := increased_price_RAM * (1 - percent_decrease)

-- Theorem stating the final result
theorem current_price_after_increase_and_decrease 
  (init_price : ℝ) 
  (inc : ℝ) 
  (dec : ℝ) 
  (final_price : ℝ) :
  init_price = 50 ∧ inc = 0.30 ∧ dec = 0.20 → final_price = 52 := 
  sorry

end current_price_after_increase_and_decrease_l45_45323


namespace smallest_of_5_consecutive_natural_numbers_sum_100_l45_45972

theorem smallest_of_5_consecutive_natural_numbers_sum_100
  (n : ℕ)
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) :
  n = 18 := sorry

end smallest_of_5_consecutive_natural_numbers_sum_100_l45_45972


namespace domain_of_f_l45_45968

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1)

theorem domain_of_f : { x : ℝ | x > 1 } = { x : ℝ | ∃ y, f y = f x } :=
by sorry

end domain_of_f_l45_45968


namespace trisha_dogs_food_expense_l45_45126

theorem trisha_dogs_food_expense :
  ∀ (meat chicken veggies eggs initial remaining final: ℤ),
    meat = 17 → 
    chicken = 22 → 
    veggies = 43 → 
    eggs = 5 → 
    remaining = 35 → 
    initial = 167 →
    final = initial - (meat + chicken + veggies + eggs) - remaining →
    final = 45 := 
by
  intros meat chicken veggies eggs initial remaining final h_meat h_chicken h_veggies h_eggs h_remaining h_initial h_final
  sorry

end trisha_dogs_food_expense_l45_45126


namespace time_spent_cleaning_bathroom_l45_45696

-- Define the times spent on each task
def laundry_time : ℕ := 30
def room_cleaning_time : ℕ := 35
def homework_time : ℕ := 40
def total_time : ℕ := 120

-- Let b be the time spent cleaning the bathroom
variable (b : ℕ)

-- Total time spent on all tasks is the sum of individual times
def total_task_time := laundry_time + b + room_cleaning_time + homework_time

-- Proof that b = 15 given the total time
theorem time_spent_cleaning_bathroom (h : total_task_time = total_time) : b = 15 :=
by
  sorry

end time_spent_cleaning_bathroom_l45_45696


namespace scatter_plot_role_regression_analysis_l45_45241

theorem scatter_plot_role_regression_analysis :
  ∀ (role : String), 
  (role = "Finding the number of individuals" ∨ 
   role = "Comparing the size relationship of individual data" ∨ 
   role = "Exploring individual classification" ∨ 
   role = "Roughly judging whether variables are linearly related")
  → role = "Roughly judging whether variables are linearly related" :=
by
  intros role h
  sorry

end scatter_plot_role_regression_analysis_l45_45241


namespace total_yellow_balloons_l45_45855

theorem total_yellow_balloons (n_tom : ℕ) (n_sara : ℕ) (h_tom : n_tom = 9) (h_sara : n_sara = 8) : n_tom + n_sara = 17 :=
by
  sorry

end total_yellow_balloons_l45_45855


namespace sum_of_cubes_l45_45563

variable {R : Type} [OrderedRing R] [Field R] [DecidableEq R]

theorem sum_of_cubes (a b c : R) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a)
    (h₄ : (a^3 + 12) / a = (b^3 + 12) / b) (h₅ : (b^3 + 12) / b = (c^3 + 12) / c) :
    a^3 + b^3 + c^3 = -36 := by
  sorry

end sum_of_cubes_l45_45563


namespace pirate_coins_l45_45028

theorem pirate_coins (x : ℕ) : 
  (x * (x + 1)) / 2 = 3 * x → 4 * x = 20 := by
  sorry

end pirate_coins_l45_45028


namespace factor_is_two_l45_45207

theorem factor_is_two (n f : ℤ) (h1 : n = 121) (h2 : n * f - 140 = 102) : f = 2 :=
by
  sorry

end factor_is_two_l45_45207


namespace geometric_sequences_identical_l45_45068

theorem geometric_sequences_identical
  (a_0 q r : ℝ)
  (a_n b_n c_n : ℕ → ℝ)
  (H₁ : ∀ n, a_n n = a_0 * q ^ n)
  (H₂ : ∀ n, b_n n = a_0 * r ^ n)
  (H₃ : ∀ n, c_n n = a_n n + b_n n)
  (H₄ : ∃ s : ℝ, ∀ n, c_n n = c_n 0 * s ^ n):
  ∀ n, a_n n = b_n n := sorry

end geometric_sequences_identical_l45_45068


namespace office_speed_l45_45975

variable (d v : ℝ)

theorem office_speed (h1 : v > 0) (h2 : ∀ t : ℕ, t = 30) (h3 : (2 * d) / (d / v + d / 30) = 24) : v = 20 := 
sorry

end office_speed_l45_45975


namespace loss_percentage_is_10_l45_45250

-- Define the conditions
def cost_price (CP : ℝ) : Prop :=
  (550 : ℝ) = 1.1 * CP

def selling_price (SP : ℝ) : Prop :=
  SP = 450

-- Define the main proof statement
theorem loss_percentage_is_10 (CP SP : ℝ) (HCP : cost_price CP) (HSP : selling_price SP) :
  ((CP - SP) / CP) * 100 = 10 :=
by
  -- Translation of the condition into Lean statement
  sorry

end loss_percentage_is_10_l45_45250


namespace at_least_one_ge_two_l45_45711

theorem at_least_one_ge_two (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 3) :
  (a + 1/b >= 2) ∨ (b + 1/c >= 2) ∨ (c + 1/a >= 2) :=
sorry

end at_least_one_ge_two_l45_45711


namespace negation_of_proposition_l45_45109

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, 0 < x → (x^2 + x > 0)) ↔ ∃ x : ℝ, 0 < x ∧ (x^2 + x ≤ 0) :=
sorry

end negation_of_proposition_l45_45109


namespace part_I_part_II_l45_45313

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := log x - k * x

theorem part_I (k : ℝ) (hk : k = 1) :
  (∀ x, 0 < x ∧ x < 1 → 0 < f 1 x - f 1 1)
  ∧ (∀ x, 1 < x → f 1 1 > f 1 x)
  ∧ f 1 1 = 0 :=
by
  sorry

theorem part_II (k : ℝ) (h_no_zeros : ∀ x, f k x ≠ 0) :
  k > 1 / exp 1 :=
by
  sorry

end part_I_part_II_l45_45313


namespace coin_loading_impossible_l45_45435

theorem coin_loading_impossible (p q : ℝ) (h₁ : p ≠ 1 - p) (h₂ : q ≠ 1 - q)
  (h₃ : p * q = 1 / 4) (h₄ : p * (1 - q) = 1 / 4) (h₅ : (1 - p) * q = 1 / 4) (h₆ : (1 - p) * (1 - q) = 1 / 4) :
  false :=
by { sorry }

end coin_loading_impossible_l45_45435


namespace inequality_proof_l45_45252

theorem inequality_proof (x : ℝ) (n : ℕ) (hx : 0 < x) : 
  1 + x^(n+1) ≥ (2*x)^n / (1 + x)^(n-1) := 
by
  sorry

end inequality_proof_l45_45252


namespace chosen_number_l45_45945

theorem chosen_number (x : ℝ) (h1 : x / 9 - 100 = 10) : x = 990 :=
  sorry

end chosen_number_l45_45945


namespace problem_l45_45422

variable (m : ℝ)

def p (m : ℝ) : Prop := m ≤ 2
def q (m : ℝ) : Prop := 0 < m ∧ m < 1

theorem problem (hpq : ¬ (p m ∧ q m)) (hlpq : p m ∨ q m) : m ≤ 0 ∨ (1 ≤ m ∧ m ≤ 2) := 
sorry

end problem_l45_45422


namespace ice_cream_sandwiches_l45_45983

theorem ice_cream_sandwiches (n : ℕ) (x : ℕ) (h1 : n = 11) (h2 : x = 13) : (n * x = 143) := 
by
  sorry

end ice_cream_sandwiches_l45_45983


namespace ternary_to_decimal_l45_45239

theorem ternary_to_decimal (n : ℕ) (h : n = 121) : 
  (1 * 3^2 + 2 * 3^1 + 1 * 3^0) = 16 :=
by sorry

end ternary_to_decimal_l45_45239


namespace increasing_function_cond_l45_45065

theorem increasing_function_cond (f : ℝ → ℝ)
  (h : ∀ a b : ℝ, a ≠ b → (f a - f b) / (a - b) > 0) :
  ∀ x y : ℝ, x < y → f x < f y :=
by
  sorry

end increasing_function_cond_l45_45065


namespace volume_is_correct_l45_45417

def volume_of_box (x : ℝ) : ℝ :=
  (14 - 2 * x) * (10 - 2 * x) * x

theorem volume_is_correct (x : ℝ) :
  volume_of_box x = 140 * x - 48 * x^2 + 4 * x^3 :=
by
  sorry

end volume_is_correct_l45_45417


namespace bus_interval_duration_l45_45347

-- Definition of the conditions
def total_minutes : ℕ := 60
def total_buses : ℕ := 11
def intervals : ℕ := total_buses - 1

-- Theorem stating the interval between each bus departure
theorem bus_interval_duration : total_minutes / intervals = 6 := 
by
  -- The proof is omitted. 
  sorry

end bus_interval_duration_l45_45347


namespace remaining_cube_edge_length_l45_45962

theorem remaining_cube_edge_length (a b : ℕ) (h : a^3 = 98 + b^3) : b = 3 :=
sorry

end remaining_cube_edge_length_l45_45962


namespace problem_solution_l45_45747

variables (x y : ℝ)

def cond1 : Prop := 4 * x + y = 12
def cond2 : Prop := x + 4 * y = 18

theorem problem_solution (h1 : cond1 x y) (h2 : cond2 x y) : 20 * x^2 + 24 * x * y + 20 * y^2 = 468 :=
by
  -- Proof would go here
  sorry

end problem_solution_l45_45747


namespace determine_n_l45_45244

noncomputable def average_value (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1) : ℚ) / (6 * (n * (n + 1) / 2))

theorem determine_n :
  ∃ n : ℕ, average_value n = 2020 ∧ n = 3029 :=
sorry

end determine_n_l45_45244


namespace probability_two_white_balls_l45_45820

noncomputable def probability_of_two_white_balls (total_balls white_balls black_balls: ℕ) : ℚ :=
  if white_balls + black_balls = total_balls ∧ total_balls = 15 ∧ white_balls = 7 ∧ black_balls = 8 then
    (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))
  else 0

theorem probability_two_white_balls : 
  probability_of_two_white_balls 15 7 8 = 1/5
:= sorry

end probability_two_white_balls_l45_45820


namespace johnnys_hourly_wage_l45_45517

def totalEarnings : ℝ := 26
def totalHours : ℝ := 8
def hourlyWage : ℝ := 3.25

theorem johnnys_hourly_wage : totalEarnings / totalHours = hourlyWage :=
by
  sorry

end johnnys_hourly_wage_l45_45517


namespace negation_of_exists_gt0_and_poly_gt0_l45_45810

theorem negation_of_exists_gt0_and_poly_gt0 :
  (¬ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 5 * x₀ + 6 > 0)) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0) :=
by sorry

end negation_of_exists_gt0_and_poly_gt0_l45_45810


namespace meadowbrook_total_not_74_l45_45580

theorem meadowbrook_total_not_74 (h c : ℕ) : 
  21 * h + 6 * c ≠ 74 := sorry

end meadowbrook_total_not_74_l45_45580


namespace find_y_l45_45560

theorem find_y (y : ℝ) (h : (y - 8) / (5 - (-3)) = -5 / 4) : y = -2 :=
by sorry

end find_y_l45_45560


namespace positive_difference_is_30_l45_45552

-- Define the absolute value equation condition
def abs_condition (x : ℝ) : Prop := abs (x - 3) = 15

-- Define the solutions to the absolute value equation
def solution1 : ℝ := 18
def solution2 : ℝ := -12

-- Define the positive difference of the solutions
def positive_difference : ℝ := abs (solution1 - solution2)

-- Theorem statement: the positive difference is 30
theorem positive_difference_is_30 : positive_difference = 30 :=
by
  sorry

end positive_difference_is_30_l45_45552


namespace width_of_jesses_room_l45_45212

theorem width_of_jesses_room (length : ℝ) (tile_area : ℝ) (num_tiles : ℕ) (total_area : ℝ) (width : ℝ) :
  length = 2 → tile_area = 4 → num_tiles = 6 → total_area = (num_tiles * tile_area : ℝ) → (length * width) = total_area → width = 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end width_of_jesses_room_l45_45212


namespace mod_z_range_l45_45039

noncomputable def z (t : ℝ) : ℂ := Complex.ofReal (1/t) + Complex.I * t

noncomputable def mod_z (t : ℝ) : ℝ := Complex.abs (z t)

theorem mod_z_range : 
  ∀ (t : ℝ), t ≠ 0 → ∃ (r : ℝ), r = mod_z t ∧ r ≥ Real.sqrt 2 :=
  by sorry

end mod_z_range_l45_45039


namespace min_pq_sq_min_value_l45_45196

noncomputable def min_pq_sq (α : ℝ) : ℝ :=
  let p := α - 2
  let q := -(α + 1)
  (p + q)^2 - 2 * (p * q)

theorem min_pq_sq_min_value : 
  (∃ (α : ℝ), ∀ p q : ℝ, 
    p^2 + q^2 = (p + q)^2 - 2 * p * q ∧ 
    (p + q = α - 2 ∧ p * q = -(α + 1))) → 
  (min_pq_sq 1) = 5 :=
by
  sorry

end min_pq_sq_min_value_l45_45196


namespace cannot_use_diff_of_squares_l45_45680

def diff_of_squares (a b : ℤ) : ℤ := a^2 - b^2

theorem cannot_use_diff_of_squares (x y : ℤ) : 
  ¬ ( ((-x + y) * (x - y)) = diff_of_squares (x - y) (0) ) :=
by {
  sorry
}

end cannot_use_diff_of_squares_l45_45680


namespace solve_linear_equation_l45_45974

theorem solve_linear_equation :
  ∀ x : ℝ, (2 * x + 1) / 3 - (5 * x - 1) / 6 = 1 → x = -3 :=
by
  sorry

end solve_linear_equation_l45_45974


namespace jugglers_count_l45_45907

-- Define the conditions
def num_balls_each_juggler := 6
def total_balls := 2268

-- Define the theorem to prove the number of jugglers
theorem jugglers_count : (total_balls / num_balls_each_juggler) = 378 :=
by
  sorry

end jugglers_count_l45_45907


namespace convert_536_oct_to_base7_l45_45599

def octal_to_decimal (n : ℕ) : ℕ :=
  n % 10 + (n / 10 % 10) * 8 + (n / 100 % 10) * 64

def decimal_to_base7 (n : ℕ) : ℕ :=
  n % 7 + (n / 7 % 7) * 10 + (n / 49 % 7) * 100 + (n / 343 % 7) * 1000

theorem convert_536_oct_to_base7 : 
  decimal_to_base7 (octal_to_decimal 536) = 1010 :=
by
  sorry

end convert_536_oct_to_base7_l45_45599


namespace expand_product_l45_45896

theorem expand_product (y : ℝ) : 4 * (y - 3) * (y + 2) = 4 * y^2 - 4 * y - 24 := 
by
  sorry

end expand_product_l45_45896


namespace employee_age_when_hired_l45_45961

theorem employee_age_when_hired
    (hire_year retire_year : ℕ)
    (rule_of_70 : ∀ A Y, A + Y = 70)
    (years_worked : ∀ hire_year retire_year, retire_year - hire_year = 19)
    (hire_year_eqn : hire_year = 1987)
    (retire_year_eqn : retire_year = 2006) :
  ∃ A : ℕ, A = 51 :=
by
  have Y := 19
  have A := 70 - Y
  use A
  sorry

end employee_age_when_hired_l45_45961


namespace train_length_180_l45_45891

noncomputable def train_length (time_seconds : ℕ) (speed_kmh : ℕ) : ℕ :=
  (speed_kmh * 1000 / 3600) * time_seconds

theorem train_length_180 :
  train_length 6 108 = 180 :=
sorry

end train_length_180_l45_45891


namespace find_values_of_x_and_y_l45_45655

-- Define the conditions
def first_condition (x : ℝ) : Prop := 0.75 / x = 5 / 7
def second_condition (y : ℝ) : Prop := y / 19 = 11 / 3

-- Define the main theorem to prove
theorem find_values_of_x_and_y (x y : ℝ) (h1 : first_condition x) (h2 : second_condition y) :
  x = 1.05 ∧ y = 209 / 3 := 
by 
  sorry

end find_values_of_x_and_y_l45_45655


namespace wholesale_prices_l45_45195

-- Definitions for the problem conditions
variable (p1 p2 d k : ℝ)
variable (h_d : d > 0)
variable (h_k : k > 1)
variable (prices : Finset ℝ)
variable (h_prices : prices = {64, 64, 70, 72})

-- The theorem statement to prove
theorem wholesale_prices :
  ∃ p1 p2, (p1 + d ∈ prices ∧ k * p1 ∈ prices) ∧ 
           (p2 + d ∈ prices ∧ k * p2 ∈ prices) ∧ 
           p1 ≠ p2
:= sorry

end wholesale_prices_l45_45195


namespace cos_half_angle_inequality_1_cos_half_angle_inequality_2_l45_45005

open Real

variable {A B C : ℝ} (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hA_sum : A + B + C = π)

theorem cos_half_angle_inequality_1 :
  cos (A / 2) < cos (B / 2) + cos (C / 2) :=
by sorry

theorem cos_half_angle_inequality_2 :
  cos (A / 2) < sin (B / 2) + sin (C / 2) :=
by sorry

end cos_half_angle_inequality_1_cos_half_angle_inequality_2_l45_45005


namespace raman_salary_loss_l45_45290

theorem raman_salary_loss : 
  ∀ (S : ℝ), S > 0 →
  let decreased_salary := S - (0.5 * S) 
  let final_salary := decreased_salary + (0.5 * decreased_salary) 
  let loss := S - final_salary 
  let percentage_loss := (loss / S) * 100
  percentage_loss = 25 := 
by
  intros S hS
  let decreased_salary := S - (0.5 * S)
  let final_salary := decreased_salary + (0.5 * decreased_salary)
  let loss := S - final_salary
  let percentage_loss := (loss / S) * 100
  have h1 : decreased_salary = 0.5 * S := by sorry
  have h2 : final_salary = 0.75 * S := by sorry
  have h3 : loss = 0.25 * S := by sorry
  have h4 : percentage_loss = 25 := by sorry
  exact h4

end raman_salary_loss_l45_45290


namespace find_triples_l45_45779

theorem find_triples (a m n : ℕ) (k : ℕ):
  a ≥ 2 ∧ m ≥ 2 ∧ a^n + 203 ≡ 0 [MOD a^m + 1] ↔ 
  (a = 2 ∧ ((n = 4 * k + 1 ∧ m = 2) ∨ (n = 6 * k + 2 ∧ m = 3) ∨ (n = 8 * k + 8 ∧ m = 4) ∨ (n = 12 * k + 9 ∧ m = 6))) ∨
  (a = 3 ∧ n = 4 * k + 3 ∧ m = 2) ∨
  (a = 4 ∧ n = 4 * k + 4 ∧ m = 2) ∨
  (a = 5 ∧ n = 4 * k + 1 ∧ m = 2) ∨
  (a = 8 ∧ n = 4 * k + 3 ∧ m = 2) ∨
  (a = 10 ∧ n = 4 * k + 2 ∧ m = 2) ∨
  (a = 203 ∧ n = (2 * k + 1) * m + 1 ∧ m ≥ 2) := by sorry

end find_triples_l45_45779


namespace volume_of_cut_out_box_l45_45671

theorem volume_of_cut_out_box (x : ℝ) : 
  let l := 16
  let w := 12
  let new_l := l - 2 * x
  let new_w := w - 2 * x
  let height := x
  let V := new_l * new_w * height
  V = 4 * x^3 - 56 * x^2 + 192 * x :=
by
  sorry

end volume_of_cut_out_box_l45_45671


namespace jeffreys_total_steps_l45_45444

-- Define the conditions
def effective_steps_per_pattern : ℕ := 1
def total_effective_distance : ℕ := 66
def steps_per_pattern : ℕ := 5

-- Define the proof problem
theorem jeffreys_total_steps : ∀ (N : ℕ), 
  N = (total_effective_distance * steps_per_pattern) := 
sorry

end jeffreys_total_steps_l45_45444


namespace speed_limit_correct_l45_45538

def speed_limit (distance : ℕ) (time : ℕ) (over_limit : ℕ) : ℕ :=
  let speed := distance / time
  speed - over_limit

theorem speed_limit_correct :
  speed_limit 60 1 10 = 50 :=
by
  sorry

end speed_limit_correct_l45_45538


namespace circle_equation_exists_l45_45370

def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, -2)
def l (p : ℝ × ℝ) : Prop := p.1 - p.2 + 1 = 0
def is_on_circle (C : ℝ × ℝ) (p : ℝ × ℝ) (r : ℝ) : Prop :=
  (p.1 - C.1)^2 + (p.2 - C.2)^2 = r^2

theorem circle_equation_exists :
  ∃ C : ℝ × ℝ, C.1 - C.2 + 1 = 0 ∧
  (is_on_circle C A 5) ∧
  (is_on_circle C B 5) ∧
  is_on_circle C (-3, -2) 5 :=
sorry

end circle_equation_exists_l45_45370


namespace k_range_proof_l45_45042

/- Define points in the Cartesian plane as ordered pairs. -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/- Define two points P and Q. -/
def P : Point := { x := -1, y := 1 }
def Q : Point := { x := 2, y := 2 }

/- Define the line equation. -/
def line_equation (k : ℝ) (x : ℝ) : ℝ :=
  k * x - 1

/- Define the range of k. -/
def k_range (k : ℝ) : Prop :=
  1 / 3 < k ∧ k < 3 / 2

/- Theorem statement. -/
theorem k_range_proof (k : ℝ) (intersects_PQ_extension : ∀ k : ℝ, ∀ x : ℝ, ((P.y ≤ line_equation k x ∧ line_equation k x ≤ Q.y) ∧ line_equation k x ≠ Q.y) → k_range k) :
  ∀ k, k_range k :=
by
  sorry

end k_range_proof_l45_45042


namespace divisor_is_22_l45_45948

theorem divisor_is_22 (n d : ℤ) (h1 : n % d = 12) (h2 : (2 * n) % 11 = 2) : d = 22 :=
by
  sorry

end divisor_is_22_l45_45948


namespace sum_of_squares_first_15_l45_45815

def sum_of_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem sum_of_squares_first_15 : sum_of_squares 15 = 3720 :=
by
  sorry

end sum_of_squares_first_15_l45_45815


namespace extrema_of_f_l45_45967

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x + 1

theorem extrema_of_f :
  (∃ x : ℝ, f x = -1 ∧ ∀ y : ℝ, f y ≥ f x) ∧
  (∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≤ f x) :=
by
  sorry

end extrema_of_f_l45_45967


namespace y_is_75_percent_of_x_l45_45423

variable (x y z : ℝ)

-- Conditions
def condition1 : Prop := 0.45 * z = 0.72 * y
def condition2 : Prop := z = 1.20 * x

-- Theorem to prove y = 0.75 * x
theorem y_is_75_percent_of_x (h1 : condition1 z y) (h2 : condition2 x z) : y = 0.75 * x :=
by sorry

end y_is_75_percent_of_x_l45_45423


namespace feasible_measures_l45_45531

-- Conditions for the problem
def condition1 := "Replace iron filings with iron pieces"
def condition2 := "Use excess zinc pieces instead of iron pieces"
def condition3 := "Add a small amount of CuSO₄ solution to the dilute hydrochloric acid"
def condition4 := "Add CH₃COONa solid to the dilute hydrochloric acid"
def condition5 := "Add sulfuric acid of the same molar concentration to the dilute hydrochloric acid"
def condition6 := "Add potassium sulfate solution to the dilute hydrochloric acid"
def condition7 := "Slightly heat (without considering the volatilization of HCl)"
def condition8 := "Add NaNO₃ solid to the dilute hydrochloric acid"

-- The criteria for the problem
def isFeasible (cond : String) : Prop :=
  cond = condition1 ∨ cond = condition2 ∨ cond = condition3 ∨ cond = condition7

theorem feasible_measures :
  ∀ cond, 
  cond ≠ condition4 →
  cond ≠ condition5 →
  cond ≠ condition6 →
  cond ≠ condition8 →
  isFeasible cond :=
by
  intros
  sorry

end feasible_measures_l45_45531


namespace book_total_pages_eq_90_l45_45141

theorem book_total_pages_eq_90 {P : ℕ} (h1 : (2 / 3 : ℚ) * P = (1 / 3 : ℚ) * P + 30) : P = 90 :=
sorry

end book_total_pages_eq_90_l45_45141


namespace ratio_Sachin_Rahul_l45_45836

-- Definitions: Sachin's age (S) is 63, and Sachin is younger than Rahul by 18 years.
def Sachin_age : ℕ := 63
def Rahul_age : ℕ := Sachin_age + 18

-- The problem: Prove the ratio of Sachin's age to Rahul's age is 7/9.
theorem ratio_Sachin_Rahul : (Sachin_age : ℚ) / (Rahul_age : ℚ) = 7 / 9 :=
by 
  -- The proof will go here, but we are skipping the proof as per the instructions.
  sorry

end ratio_Sachin_Rahul_l45_45836


namespace complement_of_P_in_U_l45_45092

/-- Definitions of sets U and P -/
def U := { y : ℝ | ∃ x : ℝ, x > 1 ∧ y = Real.log x / Real.log 2 }
def P := { y : ℝ | ∃ x : ℝ, x > 2 ∧ y = 1 / x }

/-- The complement of P in U -/
def complement_U_P := { y : ℝ | y = 0 ∨ y ≥ 1 / 2 }

/-- Proving the complement of P in U is as expected -/
theorem complement_of_P_in_U : { y : ℝ | y ∈ U ∧ y ∉ P } = complement_U_P := by
  sorry

end complement_of_P_in_U_l45_45092


namespace remainder_of_3_pow_600_mod_19_l45_45800

theorem remainder_of_3_pow_600_mod_19 :
  (3 ^ 600) % 19 = 11 :=
sorry

end remainder_of_3_pow_600_mod_19_l45_45800


namespace find_initial_men_l45_45132

def men_employed (M : ℕ) : Prop :=
  let total_hours := 50 * 8
  let completed_hours := 25 * 8
  let remaining_hours := total_hours - completed_hours
  let new_hours := 25 * 10
  let completed_work := 1 / 3
  let remaining_work := 2 / 3
  let total_work := 2 -- Total work in terms of "work units", assuming 2 km = 2 work units
  let first_eq := M * 25 * 8 = total_work * completed_work
  let second_eq := (M + 60) * 25 * 10 = total_work * remaining_work
  (M = 300 → first_eq ∧ second_eq)

theorem find_initial_men : ∃ M : ℕ, men_employed M := sorry

end find_initial_men_l45_45132


namespace ratio_of_numbers_l45_45139

theorem ratio_of_numbers (A B : ℕ) (HCF_AB : Nat.gcd A B = 3) (LCM_AB : Nat.lcm A B = 36) : 
  A / B = 3 / 4 :=
sorry

end ratio_of_numbers_l45_45139


namespace puppy_ratios_l45_45262

theorem puppy_ratios :
  ∀(total_puppies : ℕ)(golden_retriever_females golden_retriever_males : ℕ)
   (labrador_females labrador_males : ℕ)(poodle_females poodle_males : ℕ)
   (beagle_females beagle_males : ℕ),
  total_puppies = golden_retriever_females + golden_retriever_males +
                  labrador_females + labrador_males +
                  poodle_females + poodle_males +
                  beagle_females + beagle_males →
  golden_retriever_females = 2 →
  golden_retriever_males = 4 →
  labrador_females = 1 →
  labrador_males = 3 →
  poodle_females = 3 →
  poodle_males = 2 →
  beagle_females = 1 →
  beagle_males = 2 →
  (golden_retriever_females / golden_retriever_males = 1 / 2) ∧
  (labrador_females / labrador_males = 1 / 3) ∧
  (poodle_females / poodle_males = 3 / 2) ∧
  (beagle_females / beagle_males = 1 / 2) ∧
  (7 / 11 = (golden_retriever_females + labrador_females + poodle_females + beagle_females) / 
            (golden_retriever_males + labrador_males + poodle_males + beagle_males)) :=
by intros;
   sorry

end puppy_ratios_l45_45262


namespace milk_for_9_cookies_l45_45661

def quarts_to_pints (q : ℕ) : ℕ := q * 2

def milk_for_cookies (cookies : ℕ) (milk_in_quarts : ℕ) : ℕ :=
  quarts_to_pints milk_in_quarts * cookies / 18

theorem milk_for_9_cookies :
  milk_for_cookies 9 3 = 3 :=
by
  -- We define the conversion and proportional conditions explicitly here.
  unfold milk_for_cookies
  unfold quarts_to_pints
  sorry

end milk_for_9_cookies_l45_45661


namespace surface_area_of_solid_l45_45032

noncomputable def solid_surface_area (r : ℝ) (h : ℝ) : ℝ :=
  2 * Real.pi * r * h

theorem surface_area_of_solid : solid_surface_area 1 3 = 6 * Real.pi := by
  sorry

end surface_area_of_solid_l45_45032


namespace fill_tank_without_leak_l45_45359

theorem fill_tank_without_leak (T : ℕ) : 
  (1 / T - 1 / 110 = 1 / 11) ↔ T = 10 :=
by 
  sorry

end fill_tank_without_leak_l45_45359


namespace diameter_of_large_circle_l45_45922

-- Given conditions
def small_radius : ℝ := 3
def num_small_circles : ℕ := 6

-- Problem statement: Prove the diameter of the large circle
theorem diameter_of_large_circle (r : ℝ) (n : ℕ) (h_radius : r = small_radius) (h_num : n = num_small_circles) :
  ∃ (R : ℝ), R = 9 * 2 := 
sorry

end diameter_of_large_circle_l45_45922


namespace find_m_l45_45433

open Nat

def is_arithmetic (a : ℕ → ℤ) (n : ℕ) : Prop := ∀ i < n - 1, a (i + 2) - a (i + 1) = a (i + 1) - a i
def is_geometric (a : ℕ → ℤ) (n : ℕ) : Prop := ∀ i ≥ n, a (i + 1) * a n = a i * a (n + 1)
def sum_prod_condition (a : ℕ → ℤ) (m : ℕ) : Prop := a m + a (m + 1) + a (m + 2) = a m * a (m + 1) * a (m + 2)

theorem find_m (a : ℕ → ℤ)
  (h1 : a 3 = -1)
  (h2 : a 7 = 4)
  (h3 : is_arithmetic a 6)
  (h4 : is_geometric a 5) :
  ∃ m : ℕ, m = 1 ∨ m = 3 ∧ sum_prod_condition a m := sorry

end find_m_l45_45433


namespace equation_solution_l45_45483

theorem equation_solution (t : ℤ) : 
  ∃ y : ℤ, (21 * t + 2)^3 + 2 * (21 * t + 2)^2 + 5 = 21 * y :=
sorry

end equation_solution_l45_45483


namespace max_value_min_value_l45_45904

noncomputable def y (x : ℝ) : ℝ := 2 * Real.sin (3 * x + (Real.pi / 3))

theorem max_value (x : ℝ) : (∃ k : ℤ, x = (2 * k * Real.pi) / 3 + Real.pi / 18) ↔ y x = 2 :=
sorry

theorem min_value (x : ℝ) : (∃ k : ℤ, x = (2 * k * Real.pi) / 3 - 5 * Real.pi / 18) ↔ y x = -2 :=
sorry

end max_value_min_value_l45_45904


namespace fib_math_competition_l45_45361

theorem fib_math_competition :
  ∃ (n9 n8 n7 : ℕ), 
    n9 * 4 = n8 * 7 ∧ 
    n9 * 3 = n7 * 10 ∧ 
    n9 + n8 + n7 = 131 :=
sorry

end fib_math_competition_l45_45361


namespace part1_part2_l45_45291

-- Definitions and conditions
def a : ℕ := 60
def b : ℕ := 40
def c : ℕ := 80
def d : ℕ := 20
def n : ℕ := a + b + c + d

-- Given critical value for 99% certainty
def critical_value_99 : ℝ := 6.635

-- Calculate K^2 using the given formula
noncomputable def K_squared : ℝ := (n * ((a * d - b * c) ^ 2)) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Calculation of probability of selecting 2 qualified products from 5 before renovation
def total_sampled : ℕ := 5
def qualified_before_renovation : ℕ := 3
def total_combinations (n k : ℕ) : ℕ := Nat.choose n k
def prob_selecting_2_qualified : ℚ := (total_combinations qualified_before_renovation 2 : ℚ) / 
                                      (total_combinations total_sampled 2 : ℚ)

-- Proof statements
theorem part1 : K_squared > critical_value_99 := by
  sorry

theorem part2 : prob_selecting_2_qualified = 3 / 10 := by
  sorry

end part1_part2_l45_45291


namespace weight_differences_correct_l45_45277

-- Define the weights of Heather, Emily, Elizabeth, and Emma
def H : ℕ := 87
def E1 : ℕ := 58
def E2 : ℕ := 56
def E3 : ℕ := 64

-- Proof problem statement
theorem weight_differences_correct :
  (H - E1 = 29) ∧ (H - E2 = 31) ∧ (H - E3 = 23) :=
by
  -- Note: 'sorry' is used to skip the proof itself
  sorry

end weight_differences_correct_l45_45277


namespace smallest_difference_l45_45656

variable (DE EF FD : ℕ)

def is_valid_triangle (DE EF FD : ℕ) : Prop :=
  DE + EF > FD ∧ EF + FD > DE ∧ FD + DE > EF

theorem smallest_difference (h1 : DE < EF)
                           (h2 : EF ≤ FD)
                           (h3 : DE + EF + FD = 1024)
                           (h4 : is_valid_triangle DE EF FD) :
  ∃ d, d = EF - DE ∧ d = 1 :=
by
  sorry

end smallest_difference_l45_45656


namespace no_infinite_pos_sequence_l45_45921

theorem no_infinite_pos_sequence (α : ℝ) (hα : 0 < α ∧ α < 1) :
  ¬(∃ a : ℕ → ℝ, (∀ n : ℕ, a n > 0) ∧ (∀ n : ℕ, 1 + a (n + 1) ≤ a n + (α / n) * a n)) :=
sorry

end no_infinite_pos_sequence_l45_45921


namespace unique_solution_iff_a_values_l45_45886

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 5 * a

theorem unique_solution_iff_a_values (a : ℝ) :
  (∃! x : ℝ, |f x a| ≤ 3) ↔ (a = 3 / 4 ∨ a = -3 / 4) :=
by
  sorry

end unique_solution_iff_a_values_l45_45886


namespace y_pow_one_div_x_neq_x_pow_y_l45_45208

theorem y_pow_one_div_x_neq_x_pow_y (t : ℝ) (ht : t > 1) : 
  let x := t ^ (2 / (t - 1))
  let y := t ^ (3 / (t - 1))
  (y ^ (1 / x) ≠ x ^ y) :=
by
  let x := t ^ (2 / (t - 1))
  let y := t ^ (3 / (t - 1))
  sorry

end y_pow_one_div_x_neq_x_pow_y_l45_45208


namespace sign_of_f_based_on_C_l45_45197

def is_triangle (a b c : ℝ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem sign_of_f_based_on_C (a b c : ℝ) (R r : ℝ) (A B C : ℝ)
  (h1 : a = 2 * R * Real.sin A) 
  (h2 : b = 2 * R * Real.sin B) 
  (h3 : c = 2 * R * Real.sin C)
  (h4 : r = 4 * R * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2))
  (h5 : A + B + C = Real.pi)
  (h_triangle : is_triangle a b c)
  : (a + b - 2 * R - 2 * r > 0 ↔ C < Real.pi / 2) ∧
    (a + b - 2 * R - 2 * r = 0 ↔ C = Real.pi / 2) ∧
    (a + b - 2 * R - 2 * r < 0 ↔ C > Real.pi / 2) :=
sorry

end sign_of_f_based_on_C_l45_45197


namespace determine_x_l45_45258

theorem determine_x (x : ℚ) (n : ℤ) (d : ℚ) 
  (h_cond : x = n + d)
  (h_floor : n = ⌊x⌋)
  (h_d : 0 ≤ d ∧ d < 1)
  (h_eq : ⌊x⌋ + x = 17 / 4) :
  x = 9 / 4 := sorry

end determine_x_l45_45258


namespace real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l45_45895

variables {m : ℝ}

-- (1) For z to be a real number
theorem real_number_condition : (m^2 - 3 * m = 0) ↔ (m = 0 ∨ m = 3) :=
by sorry

-- (2) For z to be an imaginary number
theorem imaginary_number_condition : (m^2 - 3 * m ≠ 0) ↔ (m ≠ 0 ∧ m ≠ 3) :=
by sorry

-- (3) For z to be a purely imaginary number
theorem pure_imaginary_number_condition : (m^2 - 5 * m + 6 = 0 ∧ m^2 - 3 * m ≠ 0) ↔ (m = 2) :=
by sorry

end real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l45_45895


namespace intersection_point_of_given_lines_l45_45292

theorem intersection_point_of_given_lines :
  ∃ (x y : ℚ), 2 * y = -x + 3 ∧ -y = 5 * x + 1 ∧ x = -5 / 9 ∧ y = 16 / 9 :=
by
  sorry

end intersection_point_of_given_lines_l45_45292


namespace percent_calculation_l45_45149

theorem percent_calculation (x : ℝ) : 
  (∃ y : ℝ, y / 100 * x = 0.3 * 0.7 * x) → ∃ y : ℝ, y = 21 :=
by
  sorry

end percent_calculation_l45_45149


namespace profit_percent_l45_45112

variable {P C : ℝ}

theorem profit_percent (h1: 2 / 3 * P = 0.82 * C) : ((P - C) / C) * 100 = 23 := by
  have h2 : C = (2 / 3 * P) / 0.82 := by sorry
  have h3 : (P - C) / C = (P - (2 / 3 * P) / 0.82) / ((2 / 3 * P) / 0.82) := by sorry
  have h4 : (P - (2 / 3 * P) / 0.82) / ((2 / 3 * P) / 0.82) = (0.82 * P - 2 / 3 * P) / (2 / 3 * P) := by sorry
  have h5 : (0.82 * P - 2 / 3 * P) / (2 / 3 * P) = 0.1533 := by sorry
  have h6 : 0.1533 * 100 = 23 := by sorry
  sorry

end profit_percent_l45_45112


namespace point_on_y_axis_l45_45540

theorem point_on_y_axis (x y : ℝ) (h : x = 0 ∧ y = -1) : y = -1 := by
  -- Using the conditions directly
  cases h with
  | intro hx hy =>
    -- The proof would typically follow, but we include sorry to complete the statement
    sorry

end point_on_y_axis_l45_45540


namespace find_value_l45_45218

theorem find_value :
  3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800 :=
by
  sorry

end find_value_l45_45218


namespace fraction_to_decimal_l45_45649

theorem fraction_to_decimal :
  (7 : ℝ) / (16 : ℝ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l45_45649


namespace find_parcera_triples_l45_45567

noncomputable def is_prime (n : ℕ) : Prop := sorry
noncomputable def parcera_triple (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧
  p ∣ q^2 - 4 ∧ q ∣ r^2 - 4 ∧ r ∣ p^2 - 4

theorem find_parcera_triples : 
  {t : ℕ × ℕ × ℕ | parcera_triple t.1 t.2.1 t.2.2} = 
  {(2, 2, 2), (5, 3, 7), (7, 5, 3), (3, 7, 5)} :=
sorry

end find_parcera_triples_l45_45567


namespace tiling_rect_divisible_by_4_l45_45348

theorem tiling_rect_divisible_by_4 (m n : ℕ) (h : ∃ k l : ℕ, m = 4 * k ∧ n = 4 * l) : 
  (∃ a : ℕ, m = 4 * a) ∧ (∃ b : ℕ, n = 4 * b) :=
by 
  sorry

end tiling_rect_divisible_by_4_l45_45348


namespace number_of_solutions_l45_45772

noncomputable def system_of_equations (a b c : ℕ) : Prop :=
  a * b + b * c = 44 ∧ a * c + b * c = 23

theorem number_of_solutions : ∃! (a b c : ℕ), system_of_equations a b c :=
by
  sorry

end number_of_solutions_l45_45772


namespace q_is_false_l45_45002

theorem q_is_false (p q : Prop) (h1 : ¬(p ∧ q) = false) (h2 : ¬p = false) : q = false :=
by
  sorry

end q_is_false_l45_45002


namespace marble_color_197_l45_45108

-- Define the types and properties of the marbles
inductive Color where
  | red | blue | green

-- Define a function to find the color of the nth marble in the cycle pattern
def colorOfMarble (n : Nat) : Color :=
  let cycleLength := 15
  let positionInCycle := n % cycleLength
  if positionInCycle < 6 then Color.red  -- first 6 marbles are red
  else if positionInCycle < 11 then Color.blue  -- next 5 marbles are blue
  else Color.green  -- last 4 marbles are green

-- The theorem asserting the color of the 197th marble
theorem marble_color_197 : colorOfMarble 197 = Color.red :=
sorry

end marble_color_197_l45_45108


namespace solve_for_y_l45_45246

theorem solve_for_y (x y : ℝ) (h : 2 * x - 3 * y = 4) : y = (2 * x - 4) / 3 :=
sorry

end solve_for_y_l45_45246


namespace cone_to_sphere_surface_area_ratio_l45_45492

noncomputable def sphere_radius (r : ℝ) := r
noncomputable def cone_height (r : ℝ) := 3 * r
noncomputable def side_length_of_triangle (r : ℝ) := 2 * Real.sqrt 3 * r
noncomputable def surface_area_of_sphere (r : ℝ) := 4 * Real.pi * r^2
noncomputable def surface_area_of_cone (r : ℝ) := 9 * Real.pi * r^2
noncomputable def ratio_of_areas (cone_surface : ℝ) (sphere_surface : ℝ) := cone_surface / sphere_surface

theorem cone_to_sphere_surface_area_ratio (r : ℝ) :
    ratio_of_areas (surface_area_of_cone r) (surface_area_of_sphere r) = 9 / 4 := sorry

end cone_to_sphere_surface_area_ratio_l45_45492


namespace albert_complete_laps_l45_45210

theorem albert_complete_laps (D L : ℝ) (I : ℕ) (hD : D = 256.5) (hL : L = 9.7) (hI : I = 6) :
  ⌊(D - I * L) / L⌋ = 20 :=
by
  sorry

end albert_complete_laps_l45_45210


namespace molecular_weight_of_3_moles_l45_45621

namespace AscorbicAcid

def molecular_form : List (String × ℕ) := [("C", 6), ("H", 8), ("O", 6)]

def atomic_weight : String → ℝ
| "C" => 12.01
| "H" => 1.008
| "O" => 16.00
| _ => 0

noncomputable def molecular_weight (molecular_form : List (String × ℕ)) : ℝ :=
molecular_form.foldr (λ (x : (String × ℕ)) acc => acc + (x.snd * atomic_weight x.fst)) 0

noncomputable def weight_of_3_moles (mw : ℝ) : ℝ := mw * 3

theorem molecular_weight_of_3_moles :
  weight_of_3_moles (molecular_weight molecular_form) = 528.372 :=
by
  sorry

end AscorbicAcid

end molecular_weight_of_3_moles_l45_45621


namespace ROI_difference_l45_45202

-- Definitions based on the conditions
def Emma_investment : ℝ := 300
def Briana_investment : ℝ := 500
def Emma_yield : ℝ := 0.15
def Briana_yield : ℝ := 0.10
def years : ℕ := 2

-- The goal is to prove that the difference between their 2-year ROI is $10
theorem ROI_difference :
  let Emma_ROI := Emma_investment * Emma_yield * years
  let Briana_ROI := Briana_investment * Briana_yield * years
  (Briana_ROI - Emma_ROI) = 10 :=
by
  sorry

end ROI_difference_l45_45202


namespace zero_if_sum_of_squares_eq_zero_l45_45225

theorem zero_if_sum_of_squares_eq_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by
  sorry

end zero_if_sum_of_squares_eq_zero_l45_45225


namespace true_statement_D_l45_45115

-- Definitions related to the problem conditions
def supplementary_angles (a b : ℝ) : Prop := a + b = 180

def exterior_angle_sum_of_polygon (n : ℕ) : ℝ := 360

def acute_angle (a : ℝ) : Prop := a < 90

def triangle_inequality (a b c : ℝ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

-- The theorem to be proven based on the correct evaluation
theorem true_statement_D (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0):
  triangle_inequality a b c :=
by 
  sorry

end true_statement_D_l45_45115


namespace hire_charges_paid_by_b_l45_45223

theorem hire_charges_paid_by_b (total_cost : ℕ) (hours_a : ℕ) (hours_b : ℕ) (hours_c : ℕ) 
  (total_hours : ℕ) (cost_per_hour : ℕ) : 
  total_cost = 520 → hours_a = 7 → hours_b = 8 → hours_c = 11 → total_hours = hours_a + hours_b + hours_c 
  → cost_per_hour = total_cost / total_hours → 
  (hours_b * cost_per_hour) = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end hire_charges_paid_by_b_l45_45223


namespace sin_mul_cos_eq_quarter_l45_45565

open Real

theorem sin_mul_cos_eq_quarter (α : ℝ) (h : sin α - cos α = sqrt 2 / 2) : sin α * cos α = 1 / 4 :=
by
  sorry

end sin_mul_cos_eq_quarter_l45_45565


namespace find_multiplier_l45_45378

theorem find_multiplier (x y : ℝ) (hx : x = 0.42857142857142855) (hx_nonzero : x ≠ 0) (h_eq : (x * y) / 7 = x^2) : y = 3 :=
sorry

end find_multiplier_l45_45378


namespace nonWhiteHomesWithoutFireplace_l45_45102

-- Definitions based on the conditions
def totalHomes : ℕ := 400
def whiteHomes (h : ℕ) : ℕ := h / 4
def nonWhiteHomes (h w : ℕ) : ℕ := h - w
def nonWhiteHomesWithFireplace (nh : ℕ) : ℕ := nh / 5

-- Theorem statement to prove the required result
theorem nonWhiteHomesWithoutFireplace : 
  let h := totalHomes
  let w := whiteHomes h
  let nh := nonWhiteHomes h w
  let nf := nonWhiteHomesWithFireplace nh
  nh - nf = 240 :=
by
  let h := totalHomes
  let w := whiteHomes h
  let nh := nonWhiteHomes h w
  let nf := nonWhiteHomesWithFireplace nh
  show nh - nf = 240
  sorry

end nonWhiteHomesWithoutFireplace_l45_45102


namespace line_through_two_points_l45_45434

theorem line_through_two_points (x_1 y_1 x_2 y_2 x y : ℝ) :
  (x - x_1) * (y_2 - y_1) = (y - y_1) * (x_2 - x_1) :=
sorry

end line_through_two_points_l45_45434


namespace polynomial_has_real_root_l45_45575

open Real

theorem polynomial_has_real_root (a : ℝ) : 
  ∃ x : ℝ, x^5 + a * x^4 - x^3 + a * x^2 - x + a = 0 :=
sorry

end polynomial_has_real_root_l45_45575


namespace calculate_expression_l45_45499

theorem calculate_expression :
  (121^2 - 110^2 + 11) / 10 = 255.2 := 
sorry

end calculate_expression_l45_45499


namespace intersect_sets_l45_45013

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {0, 1, 2}

theorem intersect_sets : M ∩ N = {1, 2} :=
by
  sorry

end intersect_sets_l45_45013


namespace frankie_pets_total_l45_45789

theorem frankie_pets_total
  (C S P D : ℕ)
  (h_snakes : S = C + 6)
  (h_parrots : P = C - 1)
  (h_dogs : D = 2)
  (h_total : C + S + P + D = 19) :
  C + (C + 6) + (C - 1) + 2 = 19 := by
  sorry

end frankie_pets_total_l45_45789


namespace machine_value_correct_l45_45903

-- The present value of the machine
def present_value : ℝ := 1200

-- The depreciation rate function based on the year
def depreciation_rate (year : ℕ) : ℝ :=
  match year with
  | 1 => 0.10
  | 2 => 0.12
  | n => if n > 2 then 0.10 + 0.02 * (n - 1) else 0

-- The repair rate
def repair_rate : ℝ := 0.03

-- Value of the machine after n years
noncomputable def machine_value_after_n_years (initial_value : ℝ) (n : ℕ) : ℝ :=
  let value_first_year := (initial_value - (depreciation_rate 1 * initial_value)) + (repair_rate * initial_value)
  let value_second_year := (value_first_year - (depreciation_rate 2 * value_first_year)) + (repair_rate * value_first_year)
  match n with
  | 1 => value_first_year
  | 2 => value_second_year
  | _ => sorry -- Further generalization would be required for n > 2

-- Theorem statement
theorem machine_value_correct (initial_value : ℝ) :
  machine_value_after_n_years initial_value 2 = 1015.56 := by
  sorry

end machine_value_correct_l45_45903


namespace ferry_time_difference_l45_45703

-- Definitions for the given conditions
def speed_p := 8
def time_p := 3
def distance_p := speed_p * time_p
def distance_q := 3 * distance_p
def speed_q := speed_p + 1
def time_q := distance_q / speed_q

-- Theorem to be proven
theorem ferry_time_difference : (time_q - time_p) = 5 := 
by
  let speed_p := 8
  let time_p := 3
  let distance_p := speed_p * time_p
  let distance_q := 3 * distance_p
  let speed_q := speed_p + 1
  let time_q := distance_q / speed_q
  sorry

end ferry_time_difference_l45_45703


namespace fireflies_joined_l45_45344

theorem fireflies_joined (x : ℕ) : 
  let initial_fireflies := 3
  let flew_away := 2
  let remaining_fireflies := 9
  initial_fireflies + x - flew_away = remaining_fireflies → x = 8 := by
  sorry

end fireflies_joined_l45_45344


namespace ron_total_tax_l45_45319

def car_price : ℝ := 30000
def first_tier_level : ℝ := 10000
def first_tier_rate : ℝ := 0.25
def second_tier_rate : ℝ := 0.15

def first_tier_tax : ℝ := first_tier_level * first_tier_rate
def second_tier_tax : ℝ := (car_price - first_tier_level) * second_tier_rate
def total_tax : ℝ := first_tier_tax + second_tier_tax

theorem ron_total_tax : 
  total_tax = 5500 := by
  -- Proof will be provided here
  sorry

end ron_total_tax_l45_45319


namespace units_digit_p2_plus_3p_l45_45325

-- Define p
def p : ℕ := 2017^3 + 3^2017

-- Define the theorem to be proved
theorem units_digit_p2_plus_3p : (p^2 + 3^p) % 10 = 5 :=
by
  sorry -- Proof goes here

end units_digit_p2_plus_3p_l45_45325


namespace intersection_right_complement_l45_45603

open Set

def A := {x : ℝ | x - 1 ≥ 0}
def B := {x : ℝ | 3 / x ≤ 1}

theorem intersection_right_complement :
  A ∩ (compl B) = {x : ℝ | 1 ≤ x ∧ x < 3} :=
by
  sorry

end intersection_right_complement_l45_45603


namespace probability_neither_prime_nor_composite_l45_45476

/-- Definition of prime number: A number is prime if it has exactly two distinct positive divisors -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Definition of composite number: A number is composite if it has more than two positive divisors -/
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

/-- Given the number in the range 1 to 98 -/
def neither_prime_nor_composite (n : ℕ) : Prop := n = 1

/-- Probability function for uniform probability in a discrete sample space -/
def probability (event_occurrences total_possibilities : ℕ) : ℚ := event_occurrences / total_possibilities

theorem probability_neither_prime_nor_composite :
    probability 1 98 = 1 / 98 := by
  sorry

end probability_neither_prime_nor_composite_l45_45476


namespace max_value_of_f_l45_45692

-- Define the quadratic function
def f (x : ℝ) : ℝ := 9 * x - 4 * x^2

-- Define a proof problem to show that the maximum value of f(x) is 81/16
theorem max_value_of_f : ∃ x : ℝ, f x = 81 / 16 :=
by
  -- The vertex of the quadratic function gives the maximum value since the parabola opens downward
  let x := 9 / (2 * 4)
  use x
  -- sorry to skip the proof steps
  sorry

end max_value_of_f_l45_45692


namespace find_k_l45_45697

theorem find_k (k : ℝ) (d : ℝ) (h : d = 4) :
  -x^2 - (k + 10) * x - 8 = -(x - 2) * (x - d) → k = -16 :=
by
  intros
  rw [h] at *
  sorry

end find_k_l45_45697


namespace problem_statement_l45_45452

-- Define the function g and specify its properties
def g : ℕ → ℕ := sorry

axiom g_property (a b : ℕ) : g (a^2 + b^2) + g (a + b) = (g a)^2 + (g b)^2

-- Define the values of m and t that arise from the constraints on g(49)
def m : ℕ := 2
def t : ℕ := 106

-- Prove that the product m * t is 212
theorem problem_statement : m * t = 212 :=
by {
  -- Since g_property is an axiom, we use it to derive that
  -- g(49) can only take possible values 0 and 106,
  -- thus m = 2 and t = 106.
  sorry
}

end problem_statement_l45_45452


namespace rectangle_area_in_ellipse_l45_45770

theorem rectangle_area_in_ellipse :
  ∃ a b : ℝ, 2 * a = b ∧ (a^2 / 4 + b^2 / 8 = 1) ∧ 2 * a * b = 16 :=
by
  sorry

end rectangle_area_in_ellipse_l45_45770


namespace max_positive_integers_l45_45902

theorem max_positive_integers (a b c d e f : ℤ) (h : (a * b + c * d * e * f) < 0) :
  ∃ n, n ≤ 5 ∧ (∀x ∈ [a, b, c, d, e, f], 0 < x → x ≤ 5) :=
by
  sorry

end max_positive_integers_l45_45902


namespace prove_inequality_l45_45448

theorem prove_inequality
  (a b c d : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : c > 0)
  (h₃ : d > 0)
  (h₄ : a ≤ b)
  (h₅ : b ≤ c)
  (h₆ : c ≤ d)
  (h₇ : a + b + c + d ≥ 1) :
  a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 ≥ 1 :=
by
  sorry

end prove_inequality_l45_45448


namespace find_interest_rate_l45_45628

-- Definitions for conditions
def principal : ℝ := 12500
def interest : ℝ := 1500
def time : ℝ := 1

-- Interest rate to prove
def interest_rate : ℝ := 0.12

-- Formal statement to prove
theorem find_interest_rate (P I T : ℝ) (hP : P = principal) (hI : I = interest) (hT : T = time) : I = P * interest_rate * T :=
by
  sorry

end find_interest_rate_l45_45628


namespace four_leaf_area_l45_45713

theorem four_leaf_area (a : ℝ) : 
  let radius := a / 2
  let semicircle_area := (π * radius ^ 2) / 2
  let triangle_area := (a / 2) * (a / 2) / 2
  let half_leaf_area := semicircle_area - triangle_area
  let leaf_area := 2 * half_leaf_area
  let total_area := 4 * leaf_area
  total_area = a ^ 2 / 2 * (π - 2) := 
by
  sorry

end four_leaf_area_l45_45713


namespace largest_positive_integer_n_l45_45572

theorem largest_positive_integer_n :
  ∃ n : ℕ, (∀ x : ℝ, (Real.sin x) ^ n + (Real.cos x) ^ n ≥ 2 / n) ∧ (∀ m : ℕ, m > n → ∃ x : ℝ, (Real.sin x) ^ m + (Real.cos x) ^ m < 2 / m) :=
sorry

end largest_positive_integer_n_l45_45572


namespace James_wait_weeks_l45_45403

def JamesExercising (daysPainSubside : ℕ) (healingMultiplier : ℕ) (delayAfterHealing : ℕ) (totalDaysUntilHeavyLift : ℕ) : ℕ :=
  let healingTime := daysPainSubside * healingMultiplier
  let startWorkingOut := healingTime + delayAfterHealing
  let waitingPeriodDays := totalDaysUntilHeavyLift - startWorkingOut
  waitingPeriodDays / 7

theorem James_wait_weeks : 
  JamesExercising 3 5 3 39 = 3 :=
by
  sorry

end James_wait_weeks_l45_45403


namespace teammates_of_oliver_l45_45427

-- Define the player characteristics
structure Player :=
  (name   : String)
  (eyes   : String)
  (hair   : String)

-- Define the list of players with their given characteristics
def players : List Player := [
  {name := "Daniel", eyes := "Green", hair := "Red"},
  {name := "Oliver", eyes := "Gray", hair := "Brown"},
  {name := "Mia", eyes := "Gray", hair := "Red"},
  {name := "Ella", eyes := "Green", hair := "Brown"},
  {name := "Leo", eyes := "Green", hair := "Red"},
  {name := "Zoe", eyes := "Green", hair := "Brown"}
]

-- Define the condition for being on the same team
def same_team (p1 p2 : Player) : Bool :=
  (p1.eyes = p2.eyes && p1.hair ≠ p2.hair) || (p1.eyes ≠ p2.eyes && p1.hair = p2.hair)

-- Define the criterion to check if two players are on the same team as Oliver
def is_teammate_of_oliver (p : Player) : Bool :=
  let oliver := players[1] -- Oliver is the second player in the list
  same_team oliver p

-- Formal proof statement
theorem teammates_of_oliver : 
  is_teammate_of_oliver players[2] = true ∧ is_teammate_of_oliver players[3] = true :=
by
  -- Provide the intended proof here
  sorry

end teammates_of_oliver_l45_45427


namespace find_k_percent_l45_45331

theorem find_k_percent (k : ℝ) : 0.2 * 30 = 6 → (k / 100) * 25 = 6 → k = 24 := by
  intros h1 h2
  sorry

end find_k_percent_l45_45331


namespace base6_addition_correct_l45_45201

-- Define a function to convert a base 6 digit to its base 10 equivalent
def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 4
  | 5 => 5
  | d => 0 -- for illegal digits, fallback to 0

-- Define a function to convert a number in base 6 to base 10
def convert_base6_to_base10 (n : Nat) : Nat :=
  let units := base6_to_base10 (n % 10)
  let tens := base6_to_base10 ((n / 10) % 10)
  let hundreds := base6_to_base10 ((n / 100) % 10)
  units + 6 * tens + 6 * 6 * hundreds

-- Define a function to convert a base 10 number to a base 6 number
def base10_to_base6 (n : Nat) : Nat :=
  (n % 6) + 10 * ((n / 6) % 6) + 100 * ((n / (6 * 6)) % 6)

theorem base6_addition_correct : base10_to_base6 (convert_base6_to_base10 35 + convert_base6_to_base10 25) = 104 := by
  sorry

end base6_addition_correct_l45_45201


namespace positive_difference_even_odd_sums_l45_45161

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l45_45161


namespace books_taken_out_on_monday_l45_45355

-- Define total number of books initially
def total_books_init := 336

-- Define books taken out on Monday
variable (x : ℕ)

-- Define books brought back on Tuesday
def books_brought_back := 22

-- Define books present after Tuesday
def books_after_tuesday := 234

-- Theorem statement
theorem books_taken_out_on_monday :
  total_books_init - x + books_brought_back = books_after_tuesday → x = 124 :=
by sorry

end books_taken_out_on_monday_l45_45355


namespace no_triangle_possible_l45_45017

-- Define the lengths of the sticks
def stick_lengths : List ℕ := [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

-- The theorem stating the impossibility of forming a triangle with any combination of these lengths
theorem no_triangle_possible : ¬ ∃ (a b c : ℕ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (a + b > c ∧ a + c > b ∧ b + c > a) := 
by
  sorry

end no_triangle_possible_l45_45017


namespace initial_students_l45_45070

def students_got_off : ℕ := 3
def students_left : ℕ := 7

theorem initial_students (h1 : students_got_off = 3) (h2 : students_left = 7) :
    students_got_off + students_left = 10 :=
by
  sorry

end initial_students_l45_45070


namespace spherical_to_rectangular_conversion_l45_45587

/-- Convert a point in spherical coordinates to rectangular coordinates given specific angles and distance -/
theorem spherical_to_rectangular_conversion :
  ∀ (ρ θ φ : ℝ) (x y z : ℝ), 
  ρ = 15 → θ = 225 * (Real.pi / 180) → φ = 45 * (Real.pi / 180) →
  x = ρ * Real.sin φ * Real.cos θ → y = ρ * Real.sin φ * Real.sin θ → z = ρ * Real.cos φ →
  x = -15 / 2 ∧ y = -15 / 2 ∧ z = 15 * Real.sqrt 2 / 2 := by
  sorry

end spherical_to_rectangular_conversion_l45_45587


namespace figure_at_1000th_position_position_of_1000th_diamond_l45_45489

-- Define the repeating sequence
def repeating_sequence : List String := ["△", "Λ", "◇", "Λ", "⊙", "□"]

-- Lean 4 statement for (a)
theorem figure_at_1000th_position :
  repeating_sequence[(1000 % repeating_sequence.length) - 1] = "Λ" :=
by sorry

-- Define the arithmetic sequence for diamond positions
def diamond_position (n : Nat) : Nat :=
  3 + (n - 1) * 6

-- Lean 4 statement for (b)
theorem position_of_1000th_diamond :
  diamond_position 1000 = 5997 :=
by sorry

end figure_at_1000th_position_position_of_1000th_diamond_l45_45489


namespace largest_angle_in_triangle_PQR_is_75_degrees_l45_45381

noncomputable def largest_angle (p q r : ℝ) : ℝ :=
  if p + q + 2 * r = p^2 ∧ p + q - 2 * r = -1 then 
    Real.arccos ((p^2 + q^2 - (p^2 + p*q + (1/2)*q^2)/2) / (2 * p * q)) * (180/Real.pi)
  else 
    0

theorem largest_angle_in_triangle_PQR_is_75_degrees (p q r : ℝ) (h1 : p + q + 2 * r = p^2) (h2 : p + q - 2 * r = -1) :
  largest_angle p q r = 75 :=
by sorry

end largest_angle_in_triangle_PQR_is_75_degrees_l45_45381


namespace last_locker_opened_2046_l45_45977

def last_locker_opened (n : ℕ) : ℕ :=
  n - (n % 3)

theorem last_locker_opened_2046 : last_locker_opened 2048 = 2046 := by
  sorry

end last_locker_opened_2046_l45_45977


namespace problem1_problem2_l45_45464

-- Problem 1
theorem problem1 (a b : ℝ) : (a + 2 * b)^2 - a * (a + 4 * b) = 4 * b^2 :=
by
  sorry

-- Problem 2
theorem problem2 (m : ℝ) (h : m ≠ 1) : 
  (2 / (m - 1) + 1) / (2 * (m + 1) / (m^2 - 2 * m + 1)) = (m - 1) / 2 :=
by
  sorry

end problem1_problem2_l45_45464


namespace least_positive_x_l45_45301

theorem least_positive_x (x : ℕ) : ((2 * x) ^ 2 + 2 * 41 * 2 * x + 41 ^ 2) % 53 = 0 ↔ x = 6 := 
sorry

end least_positive_x_l45_45301


namespace find_number_l45_45718

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def XiaoQian_statements (n : ℕ) : Prop :=
  is_perfect_square n ∧ n < 5

def XiaoLu_statements (n : ℕ) : Prop :=
  n < 7 ∧ 10 ≤ n ∧ n < 100

def XiaoDai_statements (n : ℕ) : Prop :=
  is_perfect_square n ∧ ¬ (n < 5)

theorem find_number :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 99 ∧ 
    ( (XiaoQian_statements n ∧ ¬XiaoLu_statements n ∧ ¬XiaoDai_statements n) ∨
      (¬XiaoQian_statements n ∧ XiaoLu_statements n ∧ ¬XiaoDai_statements n) ∨
      (¬XiaoQian_statements n ∧ ¬XiaoLu_statements n ∧ XiaoDai_statements n) ) ∧
    n = 9 :=
sorry

end find_number_l45_45718


namespace t_shirt_jersey_price_difference_l45_45997

theorem t_shirt_jersey_price_difference :
  ∀ (T J : ℝ), (0.9 * T = 192) → (0.9 * J = 34) → (T - J = 175.55) :=
by
  intros T J hT hJ
  sorry

end t_shirt_jersey_price_difference_l45_45997


namespace focus_of_parabola_l45_45735

theorem focus_of_parabola (x y : ℝ) : x^2 = 4 * y → (0, 1) = (0, (4 / 4)) :=
by
  sorry

end focus_of_parabola_l45_45735


namespace product_of_469111_and_9999_l45_45310

theorem product_of_469111_and_9999 : 469111 * 9999 = 4690418889 := 
by 
  sorry

end product_of_469111_and_9999_l45_45310


namespace bricks_required_l45_45296

   -- Definitions from the conditions
   def courtyard_length_meters : ℝ := 42
   def courtyard_width_meters : ℝ := 22
   def brick_length_cm : ℝ := 16
   def brick_width_cm : ℝ := 10

   -- The Lean statement to prove
   theorem bricks_required : (courtyard_length_meters * courtyard_width_meters * 10000) / (brick_length_cm * brick_width_cm) = 57750 :=
   by 
       sorry
   
end bricks_required_l45_45296


namespace inverse_variation_with_constant_l45_45774

theorem inverse_variation_with_constant
  (k : ℝ)
  (x y : ℝ)
  (h1 : y = (3 * k) / x)
  (h2 : x = 4)
  (h3 : y = 8) :
  (y = (3 * (32 / 3)) / -16) := by
sorry

end inverse_variation_with_constant_l45_45774


namespace multiplication_result_l45_45062

theorem multiplication_result
  (h : 16 * 21.3 = 340.8) :
  213 * 16 = 3408 :=
sorry

end multiplication_result_l45_45062


namespace inequalities_sufficient_but_not_necessary_l45_45788

theorem inequalities_sufficient_but_not_necessary (a b c d : ℝ) :
  (a > b ∧ c > d) → (a + c > b + d) ∧ ¬((a + c > b + d) → (a > b ∧ c > d)) :=
by
  sorry

end inequalities_sufficient_but_not_necessary_l45_45788


namespace distance_from_center_to_point_l45_45022

theorem distance_from_center_to_point :
  let circle_center := (5, -7)
  let point := (3, -4)
  let distance := Real.sqrt ((3 - 5)^2 + (-4 + 7)^2)
  distance = Real.sqrt 13 := sorry

end distance_from_center_to_point_l45_45022


namespace oranges_in_box_l45_45626

theorem oranges_in_box :
  ∃ (A P O : ℕ), A + P + O = 60 ∧ A = 3 * (P + O) ∧ P = (A + O) / 5 ∧ O = 5 :=
by
  sorry

end oranges_in_box_l45_45626


namespace John_l45_45898

theorem John's_net_profit 
  (gross_income : ℕ)
  (car_purchase_cost : ℕ)
  (car_maintenance : ℕ → ℕ → ℕ)
  (car_insurance : ℕ)
  (car_tire_replacement : ℕ)
  (trade_in_value : ℕ)
  (tax_rate : ℚ)
  (total_taxes : ℕ)
  (monthly_maintenance_cost : ℕ)
  (months : ℕ)
  (net_profit : ℕ) :
  gross_income = 30000 →
  car_purchase_cost = 20000 →
  car_maintenance monthly_maintenance_cost months = 3600 →
  car_insurance = 1200 →
  car_tire_replacement = 400 →
  trade_in_value = 6000 →
  tax_rate = 15/100 →
  total_taxes = 4500 →
  monthly_maintenance_cost = 300 →
  months = 12 →
  net_profit = gross_income - (car_purchase_cost + car_maintenance monthly_maintenance_cost months + car_insurance + car_tire_replacement + total_taxes) + trade_in_value →
  net_profit = 6300 := 
by 
  sorry -- Proof to be provided

end John_l45_45898


namespace sylvia_buttons_l45_45577

theorem sylvia_buttons (n : ℕ) (h₁: n % 10 = 0) (h₂: n ≥ 80):
  (∃ w : ℕ, w = (n - (n / 2) - (n / 5) - 8)) ∧ (n - (n / 2) - (n / 5) - 8 = 1) :=
by
  sorry

end sylvia_buttons_l45_45577


namespace brownies_pieces_count_l45_45480

-- Definitions of the conditions
def pan_length : ℕ := 24
def pan_width : ℕ := 15
def pan_area : ℕ := pan_length * pan_width -- pan_area = 360

def piece_length : ℕ := 3
def piece_width : ℕ := 2
def piece_area : ℕ := piece_length * piece_width -- piece_area = 6

-- Definition of the question and proving the expected answer
theorem brownies_pieces_count : (pan_area / piece_area) = 60 := by
  sorry

end brownies_pieces_count_l45_45480


namespace parabola_equation_l45_45970

/--
Given a point P (4, -2) on a parabola, prove that the equation of the parabola is either:
1) y^2 = x or
2) x^2 = -8y.
-/
theorem parabola_equation (p : ℝ) (x y : ℝ) (h1 : (4 : ℝ) = 4) (h2 : (-2 : ℝ) = -2) :
  (∃ p : ℝ, y^2 = 2 * p * x ∧ 4 = 4 ∧ y = -2) ∨ (∃ p : ℝ, x^2 = 2 * p * y ∧ 4 = 4 ∧ x = 4) :=
sorry

end parabola_equation_l45_45970


namespace area_comparison_l45_45285

namespace Quadrilaterals

open Real

-- Define the vertices of both quadrilaterals
def quadrilateral_I_vertices : List (ℝ × ℝ) := [(0, 0), (2, 0), (2, 2), (0, 1)]
def quadrilateral_II_vertices : List (ℝ × ℝ) := [(0, 0), (3, 0), (3, 1), (0, 2)]

-- Area calculation function (example function for clarity)
def area_of_quadrilateral (vertices : List (ℝ × ℝ)) : ℝ :=
  -- This would use the actual geometry to compute the area
  2.5 -- placeholder for the area of quadrilateral I
  -- 4.5 -- placeholder for the area of quadrilateral II

theorem area_comparison :
  (area_of_quadrilateral quadrilateral_I_vertices) < (area_of_quadrilateral quadrilateral_II_vertices) :=
  sorry

end Quadrilaterals

end area_comparison_l45_45285


namespace ascending_order_perimeters_l45_45026

noncomputable def hypotenuse (r : ℝ) : ℝ := r * Real.sqrt 2

noncomputable def perimeter_P (r : ℝ) : ℝ := (2 + 3 * Real.sqrt 2) * r
noncomputable def perimeter_Q (r : ℝ) : ℝ := (6 + Real.sqrt 2) * r
noncomputable def perimeter_R (r : ℝ) : ℝ := (4 + 3 * Real.sqrt 2) * r

theorem ascending_order_perimeters (r : ℝ) (h_r_pos : 0 < r) : 
  perimeter_P r < perimeter_Q r ∧ perimeter_Q r < perimeter_R r := by
  sorry

end ascending_order_perimeters_l45_45026


namespace g_ten_l45_45211

-- Define the function g and its properties
def g : ℝ → ℝ := sorry

axiom g_property1 : ∀ x y : ℝ, g (x * y) = 2 * g x * g y
axiom g_property2 : g 0 = 2

-- Prove that g 10 = 1 / 2
theorem g_ten : g 10 = 1 / 2 :=
by
  sorry

end g_ten_l45_45211


namespace true_prop_count_l45_45819

-- Define the propositions
def original_prop (x : ℝ) : Prop := x > -3 → x > -6
def converse (x : ℝ) : Prop := x > -6 → x > -3
def inverse (x : ℝ) : Prop := x ≤ -3 → x ≤ -6
def contrapositive (x : ℝ) : Prop := x ≤ -6 → x ≤ -3

-- The statement to prove
theorem true_prop_count (x : ℝ) : 
  (original_prop x → true) ∧ (contrapositive x → true) ∧ ¬(converse x) ∧ ¬(inverse x) → 
  (count_true_propositions = 2) :=
sorry

end true_prop_count_l45_45819


namespace plants_given_away_l45_45786

-- Define the conditions as constants
def initial_plants : ℕ := 3
def final_plants : ℕ := 20
def months : ℕ := 3

-- Function to calculate the number of plants after n months
def plants_after_months (initial: ℕ) (months: ℕ) : ℕ := initial * (2 ^ months)

-- The proof problem statement
theorem plants_given_away : (plants_after_months initial_plants months - final_plants) = 4 :=
by
  sorry

end plants_given_away_l45_45786


namespace number_of_ways_to_choose_roles_l45_45542

-- Define the problem setup
def friends := Fin 6
def cooks (maria : Fin 1) := {f : Fin 6 | f ≠ maria}
def cleaners (cooks : Fin 6 → Prop) := {f : Fin 6 | ¬cooks f}

-- The number of ways to select one additional cook from the remaining friends
def chooseSecondCook : ℕ := Nat.choose 5 1  -- 5 ways

-- The number of ways to select two cleaners from the remaining friends
def chooseCleaners : ℕ := Nat.choose 4 2  -- 6 ways

-- The final number of ways to choose roles
theorem number_of_ways_to_choose_roles (maria : Fin 1) : 
  let total_ways : ℕ := chooseSecondCook * chooseCleaners
  total_ways = 30 := sorry

end number_of_ways_to_choose_roles_l45_45542


namespace range_of_m_l45_45156

def p (m : ℝ) : Prop := m > 3
def q (m : ℝ) : Prop := m > (1 / 4)

theorem range_of_m (m : ℝ) (h1 : ¬p m) (h2 : p m ∨ q m) : (1 / 4) < m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l45_45156
