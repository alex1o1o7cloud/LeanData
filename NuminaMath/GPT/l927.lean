import Mathlib

namespace area_of_parallelogram_l927_92712

def parallelogram_base : ℝ := 26
def parallelogram_height : ℝ := 14

theorem area_of_parallelogram : parallelogram_base * parallelogram_height = 364 := by
  sorry

end area_of_parallelogram_l927_92712


namespace find_number_l927_92728

theorem find_number (x : ℝ) : 
  10 * ((2 * (x * x + 2) + 3) / 5) = 50 → x = 3 := 
by
  sorry

end find_number_l927_92728


namespace h_at_neg_one_l927_92709

-- Definitions based on the conditions
def f (x : ℝ) : ℝ := 3 * x + 6
def g (x : ℝ) : ℝ := x ^ 3
def h (x : ℝ) : ℝ := f (g x)

-- The main statement to prove
theorem h_at_neg_one : h (-1) = 3 := by
  sorry

end h_at_neg_one_l927_92709


namespace k_value_range_l927_92780

-- Definitions
def f (x : ℝ) (k : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- The theorem we are interested in
theorem k_value_range (k : ℝ) (h : ∀ x₁ x₂ : ℝ, (x₁ > 5 → x₂ > 5 → f x₁ k ≤ f x₂ k) ∨ (x₁ > 5 → x₂ > 5 → f x₁ k ≥ f x₂ k)) :
  k ≥ 40 :=
sorry

end k_value_range_l927_92780


namespace amusing_permutations_formula_l927_92745

-- Definition of amusing permutations and their count
def amusing_permutations_count (n : ℕ) : ℕ :=
  2^(n-1)

-- Theorem statement: The number of amusing permutations of the set {1, 2, ..., n} is 2^(n-1)
theorem amusing_permutations_formula (n : ℕ) : 
  -- The number of amusing permutations should be equal to 2^(n-1)
  amusing_permutations_count n = 2^(n-1) :=
by
  sorry

end amusing_permutations_formula_l927_92745


namespace estimate_less_Exact_l927_92726

variables (a b c d : ℕ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)

def round_up (x : ℕ) : ℕ := x + 1
def round_down (x : ℕ) : ℕ := x - 1

theorem estimate_less_Exact
  (h₁ : round_down a = a - 1)
  (h₂ : round_down b = b - 1)
  (h₃ : round_down c = c - 1)
  (h₄ : round_up d = d + 1) :
  (round_down a + round_down b) / round_down c - round_up d < 
  (a + b) / c - d :=
sorry

end estimate_less_Exact_l927_92726


namespace fraction_uninterested_students_interested_l927_92748

theorem fraction_uninterested_students_interested 
  (students : Nat)
  (interest_ratio : ℚ)
  (express_interest_ratio_if_interested : ℚ)
  (express_disinterest_ratio_if_not_interested : ℚ) 
  (h1 : students > 0)
  (h2 : interest_ratio = 0.70)
  (h3 : express_interest_ratio_if_interested = 0.75)
  (h4 : express_disinterest_ratio_if_not_interested = 0.85) :
  let interested_students := students * interest_ratio
  let not_interested_students := students * (1 - interest_ratio)
  let express_interest_and_interested := interested_students * express_interest_ratio_if_interested
  let not_express_interest_and_interested := interested_students * (1 - express_interest_ratio_if_interested)
  let express_disinterest_and_not_interested := not_interested_students * express_disinterest_ratio_if_not_interested
  let express_interest_and_not_interested := not_interested_students * (1 - express_disinterest_ratio_if_not_interested)
  let not_express_interest_total := not_express_interest_and_interested + express_disinterest_and_not_interested
  let fraction := not_express_interest_and_interested / not_express_interest_total
  fraction = 0.407 := 
by
  sorry

end fraction_uninterested_students_interested_l927_92748


namespace planted_area_ratio_l927_92765

noncomputable def ratio_of_planted_area_to_total_area : ℚ := 145 / 147

theorem planted_area_ratio (h : ∃ (S : ℚ), 
  (∃ (x y : ℚ), x * x + y * y ≤ S * S) ∧
  (∃ (a b : ℚ), 3 * a + 4 * b = 12 ∧ (3 * x + 4 * y - 12) / 5 = 2)) :
  ratio_of_planted_area_to_total_area = 145 / 147 :=
sorry

end planted_area_ratio_l927_92765


namespace common_difference_l927_92789

variable (a : ℕ → ℝ)

def arithmetic (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∃ a1, ∀ n, a n = a1 + (n - 1) * d

def geometric_sequence (a1 a2 a5 : ℝ) : Prop :=
  a1 * (a1 + 4 * (a2 - a1)) = (a2 - a1)^2

theorem common_difference {d : ℝ} (hd : d ≠ 0)
  (h_arith : arithmetic a d)
  (h_sum : a 1 + a 2 + a 5 = 13)
  (h_geom : geometric_sequence (a 1) (a 2) (a 5)) :
  d = 2 :=
sorry

end common_difference_l927_92789


namespace length_of_first_leg_of_triangle_l927_92787

theorem length_of_first_leg_of_triangle 
  (a b c : ℝ) 
  (h1 : b = 8) 
  (h2 : c = 10) 
  (h3 : c^2 = a^2 + b^2) : 
  a = 6 :=
by
  sorry

end length_of_first_leg_of_triangle_l927_92787


namespace p_sufficient_not_necessary_for_q_neg_s_sufficient_not_necessary_for_neg_q_l927_92757

-- Define conditions
def p (x : ℝ) : Prop := -x^2 + 2 * x + 8 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0
def s (x : ℝ) : Prop := -x^2 + 8 * x + 20 ≥ 0

variable {x m : ℝ}

-- Question 1
theorem p_sufficient_not_necessary_for_q (hp : ∀ x, p x → q x m) : m ≥ 3 :=
sorry

-- Defining negation of s and q
def neg_s (x : ℝ) : Prop := ¬s x
def neg_q (x m : ℝ) : Prop := ¬q x m

-- Question 2
theorem neg_s_sufficient_not_necessary_for_neg_q (hp : ∀ x, neg_s x → neg_q x m) : false :=
sorry

end p_sufficient_not_necessary_for_q_neg_s_sufficient_not_necessary_for_neg_q_l927_92757


namespace value_of_a_l927_92781

theorem value_of_a (a b c : ℕ) (h1 : a + b = 12) (h2 : b + c = 16) (h3 : c = 7) : a = 3 := by
  sorry

end value_of_a_l927_92781


namespace parallel_lines_m_value_l927_92761

theorem parallel_lines_m_value (x y m : ℝ) (h₁ : 2 * x + m * y - 7 = 0) (h₂ : m * x + 8 * y - 14 = 0) (parallel : (2 / m = m / 8)) : m = -4 := 
sorry

end parallel_lines_m_value_l927_92761


namespace problem_solution_l927_92714

theorem problem_solution
  (P Q R S : ℕ)
  (h1 : 2 * Q = P + R)
  (h2 : R * R = Q * S)
  (h3 : R = 4 * Q / 3) :
  P + Q + R + S = 171 :=
by sorry

end problem_solution_l927_92714


namespace map_length_l927_92749

theorem map_length 
  (width : ℝ) (area : ℝ) 
  (h_width : width = 10) (h_area : area = 20) : 
  ∃ length : ℝ, area = width * length ∧ length = 2 :=
by 
  sorry

end map_length_l927_92749


namespace fraction_multiplication_l927_92773

theorem fraction_multiplication :
  ((3 : ℚ) / 4) ^ 3 * ((2 : ℚ) / 5) ^ 3 = (27 : ℚ) / 1000 := sorry

end fraction_multiplication_l927_92773


namespace find_distinct_prime_triples_l927_92754

noncomputable def areDistinctPrimes (p q r : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r

def satisfiesConditions (p q r : ℕ) : Prop :=
  p ∣ (q + r) ∧ q ∣ (r + 2 * p) ∧ r ∣ (p + 3 * q)

theorem find_distinct_prime_triples :
  { (p, q, r) : ℕ × ℕ × ℕ | areDistinctPrimes p q r ∧ satisfiesConditions p q r } =
  { (5, 3, 2), (2, 11, 7), (2, 3, 11) } :=
by
  sorry

end find_distinct_prime_triples_l927_92754


namespace triangle_inequality_for_n6_l927_92746

variables {a b c : ℝ} {n : ℕ}
open Real

-- Define the main statement as a theorem
theorem triangle_inequality_for_n6 (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end triangle_inequality_for_n6_l927_92746


namespace pigs_remaining_l927_92704

def initial_pigs : ℕ := 364
def pigs_joined : ℕ := 145
def pigs_moved : ℕ := 78

theorem pigs_remaining : initial_pigs + pigs_joined - pigs_moved = 431 := by
  sorry

end pigs_remaining_l927_92704


namespace cubic_roots_c_div_d_l927_92722

theorem cubic_roots_c_div_d (a b c d : ℚ) :
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = 1/2 ∨ x = 4) →
  (c / d = 9 / 4) :=
by
  intros h
  -- Proof would go here
  sorry

end cubic_roots_c_div_d_l927_92722


namespace total_water_carried_l927_92703

/-- Define the capacities of the four tanks in each truck -/
def tank1_capacity : ℝ := 200
def tank2_capacity : ℝ := 250
def tank3_capacity : ℝ := 300
def tank4_capacity : ℝ := 350

/-- The total capacity of one truck -/
def total_truck_capacity : ℝ := tank1_capacity + tank2_capacity + tank3_capacity + tank4_capacity

/-- Define the fill percentages for each truck -/
def fill_percentage (truck_number : ℕ) : ℝ :=
if truck_number = 1 then 1
else if truck_number = 2 then 0.75
else if truck_number = 3 then 0.5
else if truck_number = 4 then 0.25
else 0

/-- Define the amounts of water each truck carries -/
def water_carried_by_truck (truck_number : ℕ) : ℝ :=
(fill_percentage truck_number) * total_truck_capacity

/-- Prove that the total amount of water the farmer can carry in his trucks is 2750 liters -/
theorem total_water_carried : 
  water_carried_by_truck 1 + water_carried_by_truck 2 + water_carried_by_truck 3 +
  water_carried_by_truck 4 + water_carried_by_truck 5 = 2750 :=
by sorry

end total_water_carried_l927_92703


namespace area_of_triangle_l927_92767

theorem area_of_triangle (base : ℝ) (height : ℝ) (h_base : base = 3.6) (h_height : height = 2.5 * base) : 
  (base * height) / 2 = 16.2 :=
by {
  sorry
}

end area_of_triangle_l927_92767


namespace coby_travel_time_l927_92763

theorem coby_travel_time :
  let d1 := 640
  let d2 := 400
  let d3 := 250
  let d4 := 380
  let s1 := 80
  let s2 := 65
  let s3 := 75
  let s4 := 50
  let time1 := d1 / s1
  let time2 := d2 / s2
  let time3 := d3 / s3
  let time4 := d4 / s4
  let total_time := time1 + time2 + time3 + time4
  total_time = 25.08 :=
by
  sorry

end coby_travel_time_l927_92763


namespace fraction_of_percent_l927_92759

theorem fraction_of_percent (h : (1 / 8 * (1 / 100)) * 800 = 1) : true :=
by
  trivial

end fraction_of_percent_l927_92759


namespace find_x_values_l927_92782

theorem find_x_values (x1 x2 x3 x4 : ℝ)
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) :
  x1 = 4 / 5 ∧ x2 = 3 / 5 ∧ x3 = 2 / 5 ∧ x4 = 1 / 5 :=
by
  sorry

end find_x_values_l927_92782


namespace function_identity_l927_92751

theorem function_identity (f : ℕ → ℕ) 
  (h_pos : f 1 > 0) 
  (h_property : ∀ m n : ℕ, f (m^2 + n^2) = f m^2 + f n^2) : 
  ∀ n : ℕ, f n = n :=
by
  sorry

end function_identity_l927_92751


namespace simplify_expression_l927_92724

variable (m : ℝ)

theorem simplify_expression (h₁ : m ≠ 2) (h₂ : m ≠ 3) :
  (m - (4 * m - 9) / (m - 2)) / ((m ^ 2 - 9) / (m - 2)) = (m - 3) / (m + 3) := 
sorry

end simplify_expression_l927_92724


namespace range_m_l927_92741

theorem range_m (m : ℝ) :
  (∀ x : ℝ, (1 / 3 < x ∧ x < 1 / 2) ↔ abs (x - m) < 1) →
  -1 / 2 ≤ m ∧ m ≤ 4 / 3 :=
by
  intro h
  sorry

end range_m_l927_92741


namespace find_m_l927_92731

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x + α / x + Real.log x

theorem find_m (α : ℝ) (m : ℝ) (l e : ℝ) (hα_range : α ∈ Set.Icc (1 / Real.exp 1) (2 * Real.exp 2))
(h1 : f 1 α < m) (he : f (Real.exp 1) α < m) :
m > 1 + 2 * Real.exp 2 := by
  sorry

end find_m_l927_92731


namespace main_theorem_l927_92788

-- defining the conditions
def cost_ratio_pen_pencil (x : ℕ) : Prop :=
  ∀ (pen pencil : ℕ), pen = 5 * pencil ∧ x = pencil

def cost_3_pens_pencils (pen pencil total_cost : ℕ) : Prop :=
  total_cost = 3 * pen + 7 * pencil  -- assuming "some pencils" translates to 7 pencils for this demonstration

def total_cost_dozen_pens (pen total_cost : ℕ) : Prop :=
  total_cost = 12 * pen

-- proving the main statement from conditions
theorem main_theorem (pen pencil total_cost : ℕ) (x : ℕ) 
  (h1 : cost_ratio_pen_pencil x)
  (h2 : cost_3_pens_pencils (5 * x) x 100)
  (h3 : total_cost_dozen_pens (5 * x) 300) :
  total_cost = 300 :=
by
  sorry

end main_theorem_l927_92788


namespace pascal_triangle_10_to_30_l927_92750

-- Definitions
def pascal_row_numbers (n : ℕ) : ℕ := n + 1

def total_numbers_up_to (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

-- Proof Statement
theorem pascal_triangle_10_to_30 :
  total_numbers_up_to 29 - total_numbers_up_to 9 = 400 := by
  sorry

end pascal_triangle_10_to_30_l927_92750


namespace remainder_div_7_l927_92706

theorem remainder_div_7 (k : ℕ) (h1 : k % 5 = 2) (h2 : k % 6 = 5) (h3 : k < 39) : k % 7 = 3 :=
sorry

end remainder_div_7_l927_92706


namespace pig_farm_fence_l927_92725

theorem pig_farm_fence (fenced_side : ℝ) (area : ℝ) 
  (h1 : fenced_side * 2 * fenced_side = area) 
  (h2 : area = 1250) :
  4 * fenced_side = 100 :=
by {
  sorry
}

end pig_farm_fence_l927_92725


namespace no_real_solution_arctan_eqn_l927_92717

theorem no_real_solution_arctan_eqn :
  ¬∃ x : ℝ, 0 < x ∧ (Real.arctan (1 / x ^ 2) + Real.arctan (1 / x ^ 4) = (Real.pi / 4)) :=
by
  sorry

end no_real_solution_arctan_eqn_l927_92717


namespace geometric_quadratic_root_l927_92764

theorem geometric_quadratic_root (a b c : ℝ) (h1 : a > 0) (h2 : b = a * (1 / 4)) (h3 : c = a * (1 / 16)) (h4 : a * a * (1 / 4)^2 = 4 * a * a * (1 / 16)) : 
    -b / (2 * a) = -1 / 8 :=
by 
    sorry

end geometric_quadratic_root_l927_92764


namespace find_f_neg2016_l927_92798

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f_neg2016 (a b k : ℝ) (h : f a b 2016 = k) (h_ab : a * b ≠ 0) : f a b (-2016) = 2 - k :=
by
  sorry

end find_f_neg2016_l927_92798


namespace initial_population_l927_92705

theorem initial_population (P : ℝ) (h : P * (1.24 : ℝ)^2 = 18451.2) : P = 12000 :=
by
  sorry

end initial_population_l927_92705


namespace Maggie_earnings_l927_92735

theorem Maggie_earnings
    (price_per_subscription : ℕ)
    (subscriptions_parents : ℕ)
    (subscriptions_grandfather : ℕ)
    (subscriptions_nextdoor : ℕ)
    (subscriptions_another : ℕ)
    (total_subscriptions : ℕ)
    (total_earnings : ℕ) :
    subscriptions_parents = 4 →
    subscriptions_grandfather = 1 →
    subscriptions_nextdoor = 2 →
    subscriptions_another = 2 * subscriptions_nextdoor →
    total_subscriptions = subscriptions_parents + subscriptions_grandfather + subscriptions_nextdoor + subscriptions_another →
    price_per_subscription = 5 →
    total_earnings = price_per_subscription * total_subscriptions →
    total_earnings = 55 :=
by
  intros
  sorry

end Maggie_earnings_l927_92735


namespace students_only_in_math_l927_92713

-- Define the sets and their cardinalities according to the problem conditions
def total_students : ℕ := 120
def math_students : ℕ := 85
def foreign_language_students : ℕ := 65
def sport_students : ℕ := 50
def all_three_classes : ℕ := 10

-- Define the Lean theorem to prove the number of students taking only a math class
theorem students_only_in_math (total : ℕ) (M F S : ℕ) (MFS : ℕ)
  (H_total : total = 120)
  (H_M : M = 85)
  (H_F : F = 65)
  (H_S : S = 50)
  (H_MFS : MFS = 10) :
  (M - (MFS + MFS - MFS) = 35) :=
sorry

end students_only_in_math_l927_92713


namespace calculation_result_l927_92777

theorem calculation_result : 7 * (9 + 2 / 5) + 3 = 68.8 :=
by
  sorry

end calculation_result_l927_92777


namespace no_same_distribution_of_silver_as_gold_l927_92771

theorem no_same_distribution_of_silver_as_gold (n m : ℕ) 
  (hn : n ≡ 5 [MOD 10]) 
  (hm : m = 2 * n) 
  : ∀ (f : Fin 10 → ℕ), (∀ i j : Fin 10, i ≠ j → ¬ (f i - f j ≡ 0 [MOD 10])) 
  → ∀ (g : Fin 10 → ℕ), ¬ (∀ i j : Fin 10, i ≠ j → ¬ (g i - g j ≡ 0 [MOD 10])) :=
sorry

end no_same_distribution_of_silver_as_gold_l927_92771


namespace at_least_one_fraction_lt_two_l927_92769

theorem at_least_one_fraction_lt_two 
  (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_sum : 2 < x + y) : 
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end at_least_one_fraction_lt_two_l927_92769


namespace remainder_base12_2543_div_9_l927_92710

theorem remainder_base12_2543_div_9 : 
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0
  (n % 9) = 8 :=
by
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0
  sorry

end remainder_base12_2543_div_9_l927_92710


namespace find_x_l927_92715

theorem find_x (x : ℝ) (h : 0.009 / x = 0.05) : x = 0.18 :=
sorry

end find_x_l927_92715


namespace acute_angle_at_9_35_is_77_5_degrees_l927_92778

def degrees_in_acute_angle_formed_by_hands_of_clock_9_35 : ℝ := 77.5

theorem acute_angle_at_9_35_is_77_5_degrees 
  (hour_angle : ℝ := 270 + (35/60 * 30))
  (minute_angle : ℝ := 35/60 * 360) : 
  |hour_angle - minute_angle| < 180 → |hour_angle - minute_angle| = degrees_in_acute_angle_formed_by_hands_of_clock_9_35 := 
by 
  sorry

end acute_angle_at_9_35_is_77_5_degrees_l927_92778


namespace proposition_1_proposition_2_proposition_3_proposition_4_l927_92799

theorem proposition_1 : ∀ x : ℝ, 2 * x^2 - 3 * x + 4 > 0 := sorry

theorem proposition_2 : ¬ (∀ x ∈ ({-1, 0, 1} : Set ℤ), 2 * x + 1 > 0) := sorry

theorem proposition_3 : ∃ x : ℕ, x^2 ≤ x := sorry

theorem proposition_4 : ∃ x : ℕ, x ∣ 29 := sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l927_92799


namespace prism_volume_l927_92711

theorem prism_volume 
    (x y z : ℝ) 
    (h_xy : x * y = 18) 
    (h_yz : y * z = 12) 
    (h_xz : x * z = 8) 
    (h_longest_shortest : max x (max y z) = 2 * min x (min y z)) : 
    x * y * z = 16 := 
  sorry

end prism_volume_l927_92711


namespace number_of_boys_is_12500_l927_92760

-- Define the number of boys and girls in the school
def numberOfBoys (B : ℕ) : ℕ := B
def numberOfGirls : ℕ := 5000

-- Define the total attendance
def totalAttendance (B : ℕ) : ℕ := B + numberOfGirls

-- Define the condition for the percentage increase from boys to total attendance
def percentageIncreaseCondition (B : ℕ) : Prop :=
  totalAttendance B = B + Int.ofNat numberOfGirls

-- Statement to prove
theorem number_of_boys_is_12500 (B : ℕ) (h : totalAttendance B = B + numberOfGirls) : B = 12500 :=
sorry

end number_of_boys_is_12500_l927_92760


namespace polygon_sides_count_l927_92752

theorem polygon_sides_count :
    ∀ (n1 n2 n3 n4 n5 n6 : ℕ),
    n1 = 3 ∧ n2 = 4 ∧ n3 = 5 ∧ n4 = 6 ∧ n5 = 7 ∧ n6 = 8 →
    (n1 - 2) + (n2 - 2) + (n3 - 2) + (n4 - 2) + (n5 - 2) + (n6 - 1) + 3 = 24 :=
by
  intros n1 n2 n3 n4 n5 n6 h
  sorry

end polygon_sides_count_l927_92752


namespace cyclic_quadrilateral_AC_plus_BD_l927_92727

theorem cyclic_quadrilateral_AC_plus_BD (AB BC CD DA : ℝ) (AC BD : ℝ) (h1 : AB = 5) (h2 : BC = 10) (h3 : CD = 11) (h4 : DA = 14)
  (h5 : AC = Real.sqrt 221) (h6 : BD = 195 / Real.sqrt 221) :
  AC + BD = 416 / Real.sqrt (13 * 17) ∧ (AC = Real.sqrt 221 ∧ BD = 195 / Real.sqrt 221) →
  (AC + BD = 416 / Real.sqrt (13 * 17)) ∧ (AC + BD = 446) :=
by
  sorry

end cyclic_quadrilateral_AC_plus_BD_l927_92727


namespace calculate_spadesuit_l927_92794

def spadesuit (x y : ℝ) : ℝ :=
  (x + y) * (x - y)

theorem calculate_spadesuit : spadesuit 3 (spadesuit 5 6) = -112 := by
  sorry

end calculate_spadesuit_l927_92794


namespace divides_five_iff_l927_92700

theorem divides_five_iff (a : ℤ) : (5 ∣ a^2) ↔ (5 ∣ a) := sorry

end divides_five_iff_l927_92700


namespace Ruby_math_homework_l927_92730

theorem Ruby_math_homework : 
  ∃ M : ℕ, ∃ R : ℕ, R = 2 ∧ 5 * M + 9 * R = 48 ∧ M = 6 := by
  sorry

end Ruby_math_homework_l927_92730


namespace min_total_trees_l927_92701

theorem min_total_trees (L X : ℕ) (h1: 13 * L < 100 * X) (h2: 100 * X < 14 * L) : L ≥ 15 :=
  sorry

end min_total_trees_l927_92701


namespace exist_end_2015_l927_92739

def in_sequence (n : Nat) : Nat :=
  90 * n + 75

theorem exist_end_2015 :
  ∃ n : Nat, in_sequence n % 10000 = 2015 :=
by
  sorry

end exist_end_2015_l927_92739


namespace simplify_and_evaluate_l927_92768

theorem simplify_and_evaluate (a : ℤ) (h : a = 0) : 
  ((a / (a - 1) : ℚ) + ((a + 1) / (a^2 - 1) : ℚ)) = (-1 : ℚ) := by
  have ha_ne1 : a ≠ 1 := by norm_num [h]
  have ha_ne_neg1 : a ≠ -1 := by norm_num [h]
  have h1 : (a^2 - 1) ≠ 0 := by
    rw [sub_ne_zero]
    norm_num [h]
  sorry

end simplify_and_evaluate_l927_92768


namespace negation_example_l927_92734

theorem negation_example :
  (¬ (∀ x: ℝ, x > 0 → x^2 + x + 1 > 0)) ↔ (∃ x: ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by
  sorry

end negation_example_l927_92734


namespace find_three_xsq_ysq_l927_92723

theorem find_three_xsq_ysq (x y : ℤ) (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 3*x^2*y^2 = 588 :=
sorry

end find_three_xsq_ysq_l927_92723


namespace compare_exponents_l927_92775

noncomputable def exp_of_log (a : ℝ) (b : ℝ) : ℝ :=
  Real.exp ((1 / b) * Real.log a)

theorem compare_exponents :
  let a := exp_of_log 4 4
  let b := exp_of_log 5 5
  let c := exp_of_log 16 16
  let d := exp_of_log 25 25
  a = max a (max b (max c d)) ∧
  b = max (min a (max b (max c d))) (max (min b (max c d)) (max (min c d) (min d (min a b))))
  :=
  by
    sorry

end compare_exponents_l927_92775


namespace odometer_reading_at_lunch_l927_92707

axiom odometer_start : ℝ
axiom miles_traveled : ℝ
axiom odometer_at_lunch : ℝ
axiom starting_reading : odometer_start = 212.3
axiom travel_distance : miles_traveled = 159.7
axiom at_lunch_reading : odometer_at_lunch = odometer_start + miles_traveled

theorem odometer_reading_at_lunch :
  odometer_at_lunch = 372.0 :=
  by
  sorry

end odometer_reading_at_lunch_l927_92707


namespace exists_x0_gt_0_f_x0_lt_0_implies_m_lt_neg_2_l927_92793

variable (m : ℝ)
def f (x : ℝ) : ℝ := x^2 + m*x + 1

theorem exists_x0_gt_0_f_x0_lt_0_implies_m_lt_neg_2 :
  (∃ x0 : ℝ, x0 > 0 ∧ f m x0 < 0) → m < -2 := by
  sorry

end exists_x0_gt_0_f_x0_lt_0_implies_m_lt_neg_2_l927_92793


namespace count_squares_and_cubes_l927_92716

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l927_92716


namespace angle_invariant_under_magnification_l927_92766

theorem angle_invariant_under_magnification :
  ∀ (angle magnification : ℝ), angle = 10 → magnification = 5 → angle = 10 := by
  intros angle magnification h_angle h_magnification
  exact h_angle

end angle_invariant_under_magnification_l927_92766


namespace housewife_spending_l927_92732

theorem housewife_spending (P R A : ℝ) (h1 : R = 34.2) (h2 : R = 0.8 * P) (h3 : A / R - A / P = 4) :
  A = 683.45 :=
by
  sorry

end housewife_spending_l927_92732


namespace trajectory_eq_range_of_k_l927_92776

-- definitions based on the conditions:
def fixed_circle (x y : ℝ) := (x + 1)^2 + y^2 = 16
def moving_circle_passing_through_B (M : ℝ × ℝ) (B : ℝ × ℝ) := 
    B = (1, 0) ∧ M.1^2 / 4 + M.2^2 / 3 = 1 -- the ellipse trajectory equation

-- question 1: prove the equation of the ellipse
theorem trajectory_eq :
    ∀ M : ℝ × ℝ, (∃ B : ℝ × ℝ, moving_circle_passing_through_B M B)
    → (M.1^2 / 4 + M.2^2 / 3 = 1) :=
sorry

-- question 2: find the range of k which satisfies given area condition
theorem range_of_k (k : ℝ) :
    (∃ M : ℝ × ℝ, ∃ B : ℝ × ℝ, moving_circle_passing_through_B M B) → 
    (0 < k) → (¬ (k = 0)) →
    ((∃ m : ℝ, (4 * k^2 + 3 - m^2 > 0) ∧ 
    (1 / 2) * (|k| * m^2 / (4 * k^2 + 3)^2) = 1 / 14) → (3 / 4 < k ∧ k < 1) 
    ∨ (-1 < k ∧ k < -3 / 4)) :=
sorry

end trajectory_eq_range_of_k_l927_92776


namespace min_total_balls_l927_92797

theorem min_total_balls (R G B : Nat) (hG : G = 12) (hRG : R + G < 24) : 23 ≤ R + G + B :=
by {
  sorry
}

end min_total_balls_l927_92797


namespace total_passengers_landed_l927_92783

theorem total_passengers_landed 
  (passengers_on_time : ℕ) 
  (passengers_late : ℕ) 
  (passengers_connecting : ℕ) 
  (passengers_changed_plans : ℕ)
  (H1 : passengers_on_time = 14507)
  (H2 : passengers_late = 213)
  (H3 : passengers_connecting = 320)
  (H4 : passengers_changed_plans = 95) : 
  passengers_on_time + passengers_late + passengers_connecting = 15040 :=
by 
  sorry

end total_passengers_landed_l927_92783


namespace problem_statement_l927_92744

open Classical

variable (a_n : ℕ → ℝ) (a1 d : ℝ)

-- Condition: Arithmetic sequence with first term a1 and common difference d
def arithmetic_sequence (a_n : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ (n : ℕ), a_n (n + 1) = a1 + n * d 

-- Condition: Geometric relationship between a1, a3, and a9
def geometric_relation (a1 a3 a9 : ℝ) : Prop :=
  a3 / a1 = a9 / a3

-- Given conditions for the arithmetic sequence and geometric relation
axiom arith : arithmetic_sequence a_n a1 d
axiom geom : geometric_relation a1 (a1 + 2 * d) (a1 + 8 * d)

theorem problem_statement : d ≠ 0 → (∃ (a1 d : ℝ), d ≠ 0 ∧ arithmetic_sequence a_n a1 d ∧ geometric_relation a1 (a1 + 2 * d) (a1 + 8 * d)) → (a1 + 2 * d) / a1 = 3 := by
  sorry

end problem_statement_l927_92744


namespace solution_set_inequality_l927_92737

theorem solution_set_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f)
  (h_ineq : ∀ x, f x + (deriv^[2] f) x < 1) (h_f0 : f 0 = 2018) :
  ∀ x, x > 0 → f x < 2017 * Real.exp (-x) + 1 :=
by
  sorry

end solution_set_inequality_l927_92737


namespace coins_problem_l927_92720

theorem coins_problem : ∃ n : ℕ, (n % 8 = 6) ∧ (n % 9 = 7) ∧ (n % 11 = 8) :=
by {
  sorry
}

end coins_problem_l927_92720


namespace tallest_is_first_l927_92791

variable (P : Type) -- representing people
variable (line : Fin 9 → P) -- original line order (0 = shortest, 8 = tallest)
variable (Hoseok : P) -- Hoseok

-- Conditions
axiom tallest_person : line 8 = Hoseok

-- Theorem
theorem tallest_is_first :
  ∃ line' : Fin 9 → P, (∀ i : Fin 9, line' i = line (8 - i)) → line' 0 = Hoseok :=
  by
  sorry

end tallest_is_first_l927_92791


namespace find_square_side_length_l927_92708

noncomputable def square_side_length (a : ℝ) : Prop :=
  let angle_deg := 30
  let a_sqr_minus_1 := Real.sqrt (a ^ 2 - 1)
  let a_sqr_minus_4 := Real.sqrt (a ^ 2 - 4)
  let dihedral_cos := Real.cos (Real.pi / 6)  -- 30 degrees in radians
  let dihedral_sin := Real.sin (Real.pi / 6)
  let area_1 := 0.5 * a_sqr_minus_1 * a_sqr_minus_4 * dihedral_sin
  let area_2 := 0.5 * Real.sqrt (a ^ 4 - 5 * a ^ 2)
  dihedral_cos = (Real.sqrt 3 / 2) -- Using the provided angle
  ∧ dihedral_sin = 0.5
  ∧ area_1 = area_2
  ∧ a = 2 * Real.sqrt 5

-- The theorem stating that the side length of the square is 2\sqrt{5}
theorem find_square_side_length (a : ℝ) (H : square_side_length a) : a = 2 * Real.sqrt 5 := by
  sorry

end find_square_side_length_l927_92708


namespace value_of_a_b_l927_92785

theorem value_of_a_b (a b : ℕ) (ha : 2 * 100 + a * 10 + 3 + 326 = 5 * 100 + b * 10 + 9) (hb : (5 + b + 9) % 9 = 0): 
  a + b = 6 := 
sorry

end value_of_a_b_l927_92785


namespace f_f_2_l927_92756

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 2 then 2 * Real.exp (x - 1) else Real.log (2^x - 1) / Real.log 3

theorem f_f_2 : f (f 2) = 2 :=
by
  sorry

end f_f_2_l927_92756


namespace x_cubed_plus_y_cubed_l927_92740

variable (x y : ℝ)
variable (h₁ : x + y = 5)
variable (h₂ : x^2 + y^2 = 17)

theorem x_cubed_plus_y_cubed :
  x^3 + y^3 = 65 :=
by sorry

end x_cubed_plus_y_cubed_l927_92740


namespace fill_half_jar_in_18_days_l927_92796

-- Define the doubling condition and the days required to fill half the jar
variable (area : ℕ → ℕ)
variable (doubling : ∀ t, area (t + 1) = 2 * area t)
variable (full_jar : area 19 = 2^19)
variable (half_jar : area 18 = 2^18)

theorem fill_half_jar_in_18_days :
  ∃ n, n = 18 ∧ area n = 2^18 :=
by {
  -- The proof is omitted, but we state the goal
  sorry
}

end fill_half_jar_in_18_days_l927_92796


namespace cost_of_calf_l927_92762

theorem cost_of_calf (C : ℝ) (total_cost : ℝ) (cow_to_calf_ratio : ℝ) :
  total_cost = 990 ∧ cow_to_calf_ratio = 8 ∧ total_cost = C + 8 * C → C = 110 := by
  sorry

end cost_of_calf_l927_92762


namespace deceased_member_income_l927_92719

theorem deceased_member_income (a b c d : ℝ)
    (h1 : a = 735) 
    (h2 : b = 650)
    (h3 : c = 4 * 735)
    (h4 : d = 3 * 650) :
    c - d = 990 := by
  sorry

end deceased_member_income_l927_92719


namespace minimum_route_length_l927_92702

/-- 
Given a city with the shape of a 5 × 5 square grid,
prove that the minimum length of a route that covers each street exactly once and 
returns to the starting point is 68, considering each street can be walked any number of times. 
-/
theorem minimum_route_length (n : ℕ) (h1 : n = 5) : 
  ∃ route_length : ℕ, route_length = 68 := 
sorry

end minimum_route_length_l927_92702


namespace trig_identity_l927_92718

theorem trig_identity (α : ℝ) (h1 : (-Real.pi / 2) < α ∧ α < 0)
  (h2 : Real.sin α + Real.cos α = 1 / 5) :
  1 / (Real.cos α ^ 2 - Real.sin α ^ 2) = 25 / 7 := 
by 
  sorry

end trig_identity_l927_92718


namespace find_m_value_l927_92786

def symmetric_inverse (g : ℝ → ℝ) (h : ℝ → ℝ) :=
  ∀ x, g (h x) = x ∧ h (g x) = x

def symmetric_y_axis (f : ℝ → ℝ) (g : ℝ → ℝ) :=
  ∀ x, f x = g (-x)

theorem find_m_value :
  (∀ g, symmetric_inverse g (Real.exp) → (∀ f, symmetric_y_axis f g → (∀ m, f m = -1 → m = - (1 / Real.exp 1)))) := by
  sorry

end find_m_value_l927_92786


namespace find_unknown_number_l927_92770

theorem find_unknown_number (x : ℝ) (h : (2 / 3) * x + 6 = 10) : x = 6 :=
  sorry

end find_unknown_number_l927_92770


namespace more_white_animals_than_cats_l927_92742

theorem more_white_animals_than_cats (C W WC : ℕ) 
  (h1 : WC = C / 3) 
  (h2 : WC = W / 6) : W = 2 * C :=
by {
  sorry
}

end more_white_animals_than_cats_l927_92742


namespace ethanol_percentage_in_fuel_A_l927_92747

variable {capacity_A fuel_A : ℝ}
variable (ethanol_A ethanol_B total_ethanol : ℝ)
variable (E : ℝ)

def fuelTank (capacity_A fuel_A ethanol_A ethanol_B total_ethanol : ℝ) (E : ℝ) : Prop := 
  (ethanol_A / fuel_A = E) ∧
  (capacity_A - fuel_A = 200 - 99.99999999999999) ∧
  (ethanol_B = 0.16 * (200 - 99.99999999999999)) ∧
  (total_ethanol = ethanol_A + ethanol_B) ∧
  (total_ethanol = 28)

theorem ethanol_percentage_in_fuel_A : 
  ∃ E, fuelTank 99.99999999999999 99.99999999999999 ethanol_A ethanol_B 28 E ∧ E = 0.12 := 
sorry

end ethanol_percentage_in_fuel_A_l927_92747


namespace max_volume_of_sphere_in_cube_l927_92795

theorem max_volume_of_sphere_in_cube (a : ℝ) (h : a = 1) : 
  ∃ V, V = π / 6 ∧ 
        ∀ (r : ℝ), r = a / 2 →
        V = (4 / 3) * π * r^3 :=
by
  sorry

end max_volume_of_sphere_in_cube_l927_92795


namespace magic_shop_change_l927_92733

theorem magic_shop_change :
  (∀ (cloak : Type), ∃ price_gold price_silver1 change_gold1 price_silver2 change_gold2, 
  price_silver1 = 20 ∧ change_gold1 = 4 ∧ 
  price_silver2 = 15 ∧ change_gold2 = 1 ∧ 
  price_gold = 14 ∧ 
  ∀ change_silver, 
    (20 - 4) * change_silver = 15 - 1 → -- Relation derived from the conditions
    (14 - (15 - 1) * change_silver / (20 - 4)) * change_silver = 10) := 
sorry

end magic_shop_change_l927_92733


namespace multiple_of_totient_l927_92784

theorem multiple_of_totient (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ (a : ℕ), ∀ (i : ℕ), 0 ≤ i ∧ i ≤ n → m ∣ Nat.totient (a + i) :=
by
sorry

end multiple_of_totient_l927_92784


namespace capital_of_z_l927_92743

theorem capital_of_z (x y z : ℕ) (annual_profit z_share : ℕ) (months_x months_y months_z : ℕ) 
    (rx ry : ℕ) (r : ℚ) :
  x = 20000 →
  y = 25000 →
  z_share = 14000 →
  annual_profit = 50000 →
  rx = 240000 →
  ry = 300000 →
  months_x = 12 →
  months_y = 12 →
  months_z = 7 →
  r = 7 / 25 →
  z * months_z * r = z_share / (rx + ry + z * months_z) →
  z = 30000 := 
by intros; sorry

end capital_of_z_l927_92743


namespace solution_set_of_inequality_l927_92736

theorem solution_set_of_inequality (x : ℝ) : -2 * x - 1 < 3 ↔ x > -2 := 
by 
  sorry

end solution_set_of_inequality_l927_92736


namespace greatest_k_dividing_n_l927_92772

theorem greatest_k_dividing_n (n : ℕ) 
  (h1 : Nat.totient n = 72) 
  (h2 : Nat.totient (3 * n) = 96) : ∃ k : ℕ, 3^k ∣ n ∧ ∀ j : ℕ, 3^j ∣ n → j ≤ 2 := 
by {
  sorry
}

end greatest_k_dividing_n_l927_92772


namespace sequence_satisfies_n_squared_l927_92721

theorem sequence_satisfies_n_squared (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 2 → a n = a (n - 1) + 2 * n - 1) :
  ∀ n, a n = n^2 :=
by
  -- sorry
  sorry

end sequence_satisfies_n_squared_l927_92721


namespace second_daily_rate_l927_92755

noncomputable def daily_rate_sunshine : ℝ := 17.99
noncomputable def mileage_cost_sunshine : ℝ := 0.18
noncomputable def mileage_cost_second : ℝ := 0.16
noncomputable def distance : ℝ := 48.0

theorem second_daily_rate (daily_rate_second : ℝ) : 
  daily_rate_sunshine + (mileage_cost_sunshine * distance) = 
  daily_rate_second + (mileage_cost_second * distance) → 
  daily_rate_second = 18.95 :=
by 
  sorry

end second_daily_rate_l927_92755


namespace ratio_of_volumes_l927_92753

-- Define the edge lengths
def edge_length_cube1 : ℝ := 9
def edge_length_cube2 : ℝ := 24

-- Theorem stating the ratio of the volumes
theorem ratio_of_volumes :
  (edge_length_cube1 / edge_length_cube2) ^ 3 = 27 / 512 :=
by
  sorry

end ratio_of_volumes_l927_92753


namespace gear_q_revolutions_per_minute_l927_92790

noncomputable def gear_p_revolutions_per_minute : ℕ := 10

noncomputable def additional_revolutions : ℕ := 15

noncomputable def calculate_q_revolutions_per_minute
  (p_rev_per_min : ℕ) (additional_rev : ℕ) : ℕ :=
  2 * (p_rev_per_min / 2 + additional_rev)

theorem gear_q_revolutions_per_minute :
  calculate_q_revolutions_per_minute gear_p_revolutions_per_minute additional_revolutions = 40 :=
by
  sorry

end gear_q_revolutions_per_minute_l927_92790


namespace ratio_b4_b3_a2_a1_l927_92738

variables {x y d d' : ℝ}
variables {a1 a2 a3 b1 b2 b3 b4 : ℝ}
-- Conditions
variables (h1 : x ≠ y)
variables (h2 : a1 = x + d)
variables (h3 : a2 = x + 2 * d)
variables (h4 : a3 = x + 3 * d)
variables (h5 : y = x + 4 * d)
variables (h6 : b2 = x + d')
variables (h7 : b3 = x + 2 * d')
variables (h8 : y = x + 3 * d')
variables (h9 : b4 = x + 4 * d')

theorem ratio_b4_b3_a2_a1 :
  (b4 - b3) / (a2 - a1) = 8 / 3 :=
by sorry

end ratio_b4_b3_a2_a1_l927_92738


namespace ammonium_chloride_reaction_l927_92774

/-- 
  Given the reaction NH4Cl + H2O → NH4OH + HCl, 
  if 1 mole of NH4Cl reacts with 1 mole of H2O to produce 1 mole of NH4OH, 
  then 1 mole of HCl is formed.
-/
theorem ammonium_chloride_reaction :
  (∀ (NH4Cl H2O NH4OH HCl : ℕ), NH4Cl = 1 ∧ H2O = 1 ∧ NH4OH = 1 → HCl = 1) :=
by
  sorry

end ammonium_chloride_reaction_l927_92774


namespace find_b_l927_92792

-- Definitions based on the given conditions
def good_point (a b : ℝ) (φ : ℝ) : Prop :=
  a + (b - a) * φ = 2.382 ∨ b - (b - a) * φ = 2.382

theorem find_b (b : ℝ) (φ : ℝ := 0.618) :
  good_point 2 b φ → b = 2.618 ∨ b = 3 :=
by
  sorry

end find_b_l927_92792


namespace problem_solution_l927_92729

noncomputable def f (x : ℝ) (p : ℝ) (q : ℝ) : ℝ := x^2 - p * x + q

theorem problem_solution
  (a b p q : ℝ)
  (h1 : a ≠ b)
  (h2 : p > 0)
  (h3 : q > 0)
  (h4 : f a p q = 0)
  (h5 : f b p q = 0)
  (h6 : ∃ k : ℝ, (a = -2 + k ∧ b = -2 - k) ∨ (a = -2 - k ∧ b = -2 + k))
  (h7 : ∃ l : ℝ, (a = -2 * l ∧ b = 4 * l) ∨ (a = 4 * l ∧ b = -2 * l))
  : p + q = 9 :=
sorry

end problem_solution_l927_92729


namespace ab_operation_l927_92779

theorem ab_operation (a b : ℤ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h1 : a + b = 10) (h2 : a * b = 24) : 
  (1 / a + 1 / b) = 5 / 12 :=
by
  sorry

end ab_operation_l927_92779


namespace number_of_unit_distance_pairs_lt_bound_l927_92758

/-- Given n distinct points in the plane, the number of pairs of points with a unit distance between them is less than n / 4 + (1 / sqrt 2) * n^(3 / 2). -/
theorem number_of_unit_distance_pairs_lt_bound (n : ℕ) (hn : 0 < n) :
  ∃ E : ℕ, E < n / 4 + (1 / Real.sqrt 2) * n^(3 / 2) :=
by
  sorry

end number_of_unit_distance_pairs_lt_bound_l927_92758
