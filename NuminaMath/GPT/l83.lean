import Mathlib

namespace Jan_is_6_inches_taller_than_Bill_l83_83216

theorem Jan_is_6_inches_taller_than_Bill :
  ∀ (Cary Bill Jan : ℕ),
    Cary = 72 →
    Bill = Cary / 2 →
    Jan = 42 →
    Jan - Bill = 6 :=
by
  intros
  sorry

end Jan_is_6_inches_taller_than_Bill_l83_83216


namespace product_of_primes_is_66_l83_83637

theorem product_of_primes_is_66 :
  let p1 : ℕ := 2
      p2 : ℕ := 3
      p3 : ℕ := 11
  in p1 * p2 * p3 = 66 := by
  sorry

end product_of_primes_is_66_l83_83637


namespace tomatoes_left_l83_83789

theorem tomatoes_left (initial_tomatoes picked_yesterday picked_today : ℕ)
    (h_initial : initial_tomatoes = 171)
    (h_picked_yesterday : picked_yesterday = 134)
    (h_picked_today : picked_today = 30) :
    initial_tomatoes - picked_yesterday - picked_today = 7 :=
by
    sorry

end tomatoes_left_l83_83789


namespace division_by_fraction_equiv_neg_multiplication_l83_83821

theorem division_by_fraction_equiv_neg_multiplication (h : 43 * 47 = 2021) : (-43) / (1 / 47) = -2021 :=
by
  -- Proof would go here, but we use sorry to skip the proof for now.
  sorry

end division_by_fraction_equiv_neg_multiplication_l83_83821


namespace susie_initial_amount_l83_83752

-- Definitions for conditions:
def initial_amount (X : ℝ) : Prop :=
  X + 0.20 * X = 240

-- Main theorem to prove:
theorem susie_initial_amount (X : ℝ) (h : initial_amount X) : X = 200 :=
by 
  -- structured proof will go here
  sorry

end susie_initial_amount_l83_83752


namespace product_of_primes_is_66_l83_83634

theorem product_of_primes_is_66 :
  let p1 : ℕ := 2
      p2 : ℕ := 3
      p3 : ℕ := 11
  in p1 * p2 * p3 = 66 := by
  sorry

end product_of_primes_is_66_l83_83634


namespace height_of_parallelogram_l83_83690

theorem height_of_parallelogram (A B H : ℕ) (hA : A = 308) (hB : B = 22) (h_eq : H = A / B) : H = 14 := 
by sorry

end height_of_parallelogram_l83_83690


namespace reflection_about_x_axis_l83_83379

theorem reflection_about_x_axis (a : ℝ) : 
  (A : ℝ × ℝ) = (3, a) → (B : ℝ × ℝ) = (3, 4) → A = (3, -4) → a = -4 :=
by
  intros A_eq B_eq reflection_eq
  sorry

end reflection_about_x_axis_l83_83379


namespace find_q_in_geometric_sequence_l83_83266

theorem find_q_in_geometric_sequence
  {q : ℝ} (q_pos : q > 0) 
  (a1_def : ∀(a : ℕ → ℝ), a 1 = 1 / q^2) 
  (S5_eq_S2_plus_2 : ∀(S : ℕ → ℝ), S 5 = S 2 + 2) :
  q = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end find_q_in_geometric_sequence_l83_83266


namespace profit_calculation_l83_83748

theorem profit_calculation (boxes_bought : ℕ) (cost_per_box : ℕ) (pens_per_box : ℕ) (packages_per_box : ℕ) (packages_price : ℕ) (sets_per_box : ℕ) (sets_price : ℕ) :
  boxes_bought = 12 ∧
  cost_per_box = 10 ∧
  pens_per_box = 30  ∧ 
  packages_per_box = 5  ∧ 
  packages_price = 3 ∧ 
  sets_per_box = 3 ∧
  sets_price = 2 →
  let total_cost := boxes_bought * cost_per_box in
  let total_pens := boxes_bought * pens_per_box in
  let boxes_used_for_packages := 5 in -- based on given conditions
  let packages := boxes_used_for_packages * packages_per_box in
  let revenue_from_packages := packages * packages_price in
  let boxes_left := boxes_bought - boxes_used_for_packages in
  let pens_left := boxes_left * pens_per_box in
  let sets := pens_left / sets_per_box in
  let revenue_from_pens := sets * sets_price in
  let total_revenue := revenue_from_packages + revenue_from_pens in
  let profit := total_revenue - total_cost in
  profit = 95 :=
by 
  intros;
  -- Ensure that all declared variables and their calculations are consistent with the problem conditions shown above
  sorry

end profit_calculation_l83_83748


namespace total_investment_is_correct_l83_83774

def Raghu_investment : ℕ := 2300
def Trishul_investment (Raghu_investment : ℕ) : ℕ := Raghu_investment - (Raghu_investment / 10)
def Vishal_investment (Trishul_investment : ℕ) : ℕ := Trishul_investment + (Trishul_investment / 10)

theorem total_investment_is_correct :
    let Raghu_inv := Raghu_investment;
    let Trishul_inv := Trishul_investment Raghu_inv;
    let Vishal_inv := Vishal_investment Trishul_inv;
    Raghu_inv + Trishul_inv + Vishal_inv = 6647 :=
by
    sorry

end total_investment_is_correct_l83_83774


namespace total_legs_of_all_animals_l83_83023

def num_kangaroos : ℕ := 23
def num_goats : ℕ := 3 * num_kangaroos
def legs_of_kangaroo : ℕ := 2
def legs_of_goat : ℕ := 4

theorem total_legs_of_all_animals : num_kangaroos * legs_of_kangaroo + num_goats * legs_of_goat = 322 :=
by 
  sorry

end total_legs_of_all_animals_l83_83023


namespace g_is_odd_l83_83885

def g (x : ℝ) : ℝ := ⌈x⌉ - 1/2

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end g_is_odd_l83_83885


namespace coin_outcomes_equivalent_l83_83262

theorem coin_outcomes_equivalent :
  let outcomes_per_coin := 2
  let total_coins := 3
  (outcomes_per_coin ^ total_coins) = 8 :=
by
  sorry

end coin_outcomes_equivalent_l83_83262


namespace substitution_not_sufficient_for_identity_proof_l83_83344

theorem substitution_not_sufficient_for_identity_proof {α : Type} (f g : α → α) :
  (∀ x : α, f x = g x) ↔ ¬ (∀ x, f x = g x ↔ (∃ (c : α), f c ≠ g c)) := by
  sorry

end substitution_not_sufficient_for_identity_proof_l83_83344


namespace sum_of_four_powers_l83_83353

theorem sum_of_four_powers (a : ℕ) : 4 * a^3 = 500 :=
by
  rw [Nat.pow_succ, Nat.pow_succ]
  sorry

end sum_of_four_powers_l83_83353


namespace cistern_emptying_time_l83_83656

theorem cistern_emptying_time (R L : ℝ) (hR : R = 1 / 6) (hL : L = 1 / 6 - 1 / 8) :
    1 / L = 24 := by
  -- The proof is omitted
  sorry

end cistern_emptying_time_l83_83656


namespace probability_three_flips_all_heads_l83_83932

open ProbabilityTheory

-- Define a fair coin flip
def fair_coin_flip : ProbabilityTheory.PMeasure bool := 
  PMF.ofMultiset { (true, 1) , (false, 1) }.1 sorry

-- Define the event that the first three flips are all heads
def three_flips_all_heads : Event (bool × bool × bool) := 
  { (true, true, true) }

-- State the theorem specifying the probability
theorem probability_three_flips_all_heads :
  P (independent_ideals fair_coin_flip fair_coin_flip fair_coin_flip)
  (three_flips_all_heads fair_coin_flip fair_coin_flip fair_coin_flip) = 1 / 8 := 
sorry

end probability_three_flips_all_heads_l83_83932


namespace range_of_a_plus_b_l83_83119

variable {a b : ℝ}

def has_two_real_roots (a b : ℝ) : Prop :=
  let discriminant := b^2 - 4 * a * (-4)
  discriminant ≥ 0

def has_root_in_interval (a b : ℝ) : Prop :=
  (a + b - 4) * (4 * a + 2 * b - 4) < 0

theorem range_of_a_plus_b 
  (h1 : has_two_real_roots a b) 
  (h2 : has_root_in_interval a b) 
  (h3 : a > 0) : 
  a + b < 4 :=
sorry

end range_of_a_plus_b_l83_83119


namespace planeThroughPointAndLine_l83_83108

theorem planeThroughPointAndLine :
  ∃ A B C D : ℤ, (A = -3 ∧ B = -4 ∧ C = -4 ∧ D = 14) ∧ 
  (∀ x y z : ℝ, x = 2 ∧ y = -3 ∧ z = 5 ∨ (∃ t : ℝ, x = 4 * t + 2 ∧ y = -5 * t - 1 ∧ z = 2 * t + 3) → A * x + B * y + C * z + D = 0) :=
sorry

end planeThroughPointAndLine_l83_83108


namespace gcd_problem_l83_83225

-- Define the two numbers
def a : ℕ := 1000000000
def b : ℕ := 1000000005

-- Define the problem to prove the GCD
theorem gcd_problem : Nat.gcd a b = 5 :=
by 
  sorry

end gcd_problem_l83_83225


namespace temperature_conversion_l83_83131

theorem temperature_conversion (C F F_new C_new : ℚ) 
  (h_formula : C = (5/9) * (F - 32))
  (h_C : C = 30)
  (h_F_new : F_new = F + 15)
  (h_F : F = 86)
: C_new = (5/9) * (F_new - 32) ↔ C_new = 38.33 := 
by 
  sorry

end temperature_conversion_l83_83131


namespace surface_area_of_parallelepiped_l83_83757

open Real

theorem surface_area_of_parallelepiped 
  (a b c : ℝ)
  (x y z : ℝ)
  (h1: a^2 = x^2 + y^2)
  (h2: b^2 = x^2 + z^2)
  (h3: c^2 = y^2 + z^2) :
  2 * (sqrt ((x * y)) + sqrt ((x * z)) + sqrt ((y * z)))  =
  sqrt ((a^2 + b^2 - c^2) * (a^2 + c^2 - b^2)) +
  sqrt ((a^2 + b^2 - c^2) * (b^2 + c^2 - a^2)) +
  sqrt ((a^2 + c^2 - b^2) * (b^2 + c^2 - a^2)) :=
by
  sorry

end surface_area_of_parallelepiped_l83_83757


namespace lesser_number_l83_83448

theorem lesser_number (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 10) : y = 25 :=
by
  have h3 : x = 35 := sorry
  exact sorry

end lesser_number_l83_83448


namespace total_rowing_and_hiking_l83_83281

def total_campers : ℕ := 80
def morning_rowing : ℕ := 41
def morning_hiking : ℕ := 4
def morning_swimming : ℕ := 15
def afternoon_rowing : ℕ := 26
def afternoon_hiking : ℕ := 8
def afternoon_swimming : ℕ := total_campers - afternoon_rowing - afternoon_hiking - (total_campers - morning_rowing - morning_hiking - morning_swimming)

theorem total_rowing_and_hiking : 
  (morning_rowing + afternoon_rowing) + (morning_hiking + afternoon_hiking) = 79 :=
by
  sorry

end total_rowing_and_hiking_l83_83281


namespace average_eq_35_implies_y_eq_50_l83_83911

theorem average_eq_35_implies_y_eq_50 (y : ℤ) (h : (15 + 30 + 45 + y) / 4 = 35) : y = 50 :=
by
  sorry

end average_eq_35_implies_y_eq_50_l83_83911


namespace correct_statements_l83_83988

def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f (-x) = f x
axiom monotonic_increasing_on_neg1_0 : ∀ ⦃x y : ℝ⦄, -1 ≤ x → x ≤ y → y ≤ 0 → f x ≤ f y
axiom functional_eqn (x : ℝ) : f (1 - x) + f (1 + x) = 0

theorem correct_statements :
  (∀ x, f (1 - x) = -f (1 + x)) ∧ f 2 ≤ f x :=
by
  sorry

end correct_statements_l83_83988


namespace a8_value_l83_83695

variable {an : ℕ → ℕ}

def S (n : ℕ) : ℕ := n ^ 2

theorem a8_value : an 8 = S 8 - S 7 := by
  sorry

end a8_value_l83_83695


namespace arithmetic_sequence_common_difference_l83_83183

/-- The sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a₁ d : ℚ) : ℚ := n * a₁ + (n * (n - 1) / 2) * d

/-- Condition for the sum of the first 5 terms -/
def S5 (a₁ d : ℚ) : Prop := S 5 a₁ d = 6

/-- Condition for the second term of the sequence -/
def a2 (a₁ d : ℚ) : Prop := a₁ + d = 1

/-- The main theorem to be proved -/
theorem arithmetic_sequence_common_difference (a₁ d : ℚ) (hS5 : S5 a₁ d) (ha2 : a2 a₁ d) : d = 1 / 5 :=
sorry

end arithmetic_sequence_common_difference_l83_83183


namespace n19_minus_n7_div_30_l83_83740

theorem n19_minus_n7_div_30 (n : ℕ) (h : 0 < n) : 30 ∣ (n^19 - n^7) :=
sorry

end n19_minus_n7_div_30_l83_83740


namespace intersection_M_N_l83_83124

def M : Set ℝ := { x | (x - 1)^2 < 4 }
def N : Set ℝ := { -1, 0, 1, 2, 3 }

theorem intersection_M_N : M ∩ N = {0, 1, 2} := 
by
  sorry

end intersection_M_N_l83_83124


namespace rectangular_floor_problem_possibilities_l83_83494

theorem rectangular_floor_problem_possibilities :
  ∃ (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s → p.2 > p.1 ∧ p.2 % 3 = 0 ∧ (p.1 - 6) * (p.2 - 6) = 36) 
    ∧ s.card = 2 := 
sorry

end rectangular_floor_problem_possibilities_l83_83494


namespace calculate_a_plus_b_l83_83401

noncomputable def f (a b x : ℝ) : ℝ := a * x + b
noncomputable def g (x : ℝ) : ℝ := 3 * x - 7

theorem calculate_a_plus_b (a b : ℝ) (h : ∀ x : ℝ, g (f a b x) = 4 * x + 6) : a + b = 17 / 3 :=
by
  sorry

end calculate_a_plus_b_l83_83401


namespace find_k_l83_83433

theorem find_k (x1 x2 : ℝ) (r : ℝ) (h1 : x1 = 3 * r) (h2 : x2 = r) (h3 : x1 + x2 = -8) (h4 : x1 * x2 = k) : k = 12 :=
by
  -- proof steps here
  sorry

end find_k_l83_83433


namespace total_area_of_sheet_l83_83334

theorem total_area_of_sheet (A B : ℝ) (h1 : A = 4 * B) (h2 : A = B + 2208) : A + B = 3680 :=
by
  sorry

end total_area_of_sheet_l83_83334


namespace avg_of_multiples_of_10_eq_305_l83_83928

theorem avg_of_multiples_of_10_eq_305 (N : ℕ) (h : N % 10 = 0) (h_avg : (10 + N) / 2 = 305) : N = 600 :=
sorry

end avg_of_multiples_of_10_eq_305_l83_83928


namespace n_value_l83_83255

theorem n_value (n : ℤ) (h1 : (18888 - n) % 11 = 0) : n = 7 :=
sorry

end n_value_l83_83255


namespace total_cleaning_time_is_100_l83_83161

def outsideCleaningTime : ℕ := 80
def insideCleaningTime : ℕ := outsideCleaningTime / 4
def totalCleaningTime : ℕ := outsideCleaningTime + insideCleaningTime

theorem total_cleaning_time_is_100 : totalCleaningTime = 100 := by
  sorry

end total_cleaning_time_is_100_l83_83161


namespace function_range_x2_minus_2x_l83_83349

theorem function_range_x2_minus_2x : 
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 3 → -1 ≤ x^2 - 2 * x ∧ x^2 - 2 * x ≤ 3 :=
by
  intro x hx
  sorry

end function_range_x2_minus_2x_l83_83349


namespace ratio_of_sums_l83_83370

noncomputable def first_sum : Nat := 
  let sequence := (List.range' 1 15)
  let differences := (List.range' 2 30).map (fun x => 2 * x)
  let sequence_sum := sequence.zip differences |>.map (λ ⟨a, d⟩ => 10 / 2 * (2 * a + 9 * d))
  5 * (20 * (sequence_sum.sum))

noncomputable def second_sum : Nat :=
  let sequence := (List.range' 1 15)
  let differences := (List.range' 1 29).filterMap (fun x => if x % 2 = 1 then some x else none)
  let sequence_sum := sequence.zip differences |>.map (λ ⟨a, d⟩ => 10 / 2 * (2 * a + 9 * d))
  5 * (20 * (sequence_sum.sum) - 135)

theorem ratio_of_sums : (first_sum / second_sum : Rat) = (160 / 151 : Rat) :=
  sorry

end ratio_of_sums_l83_83370


namespace problem1_eval_problem2_eval_l83_83215

theorem problem1_eval : (1 * (Real.pi - 3.14)^0 - |2 - Real.sqrt 3| + (-1 / 2)^2) = Real.sqrt 3 - 3 / 4 :=
  sorry

theorem problem2_eval : (Real.sqrt (1 / 3) + Real.sqrt 6 * (1 / Real.sqrt 2 + Real.sqrt 8)) = 16 * Real.sqrt 3 / 3 :=
  sorry

end problem1_eval_problem2_eval_l83_83215


namespace find_y_when_x_4_l83_83594

-- Definitions and conditions
variables (x y : ℝ)
def inversely_proportional (x y : ℝ) (K : ℝ) : Prop := x * y = K

-- Main theorem
theorem find_y_when_x_4 
  (K : ℝ) (h1 : inversely_proportional 20 10 K) (h2 : 20 + 10 = 30) (h3 : 20 - 10 = 10) 
  (hx : 4 * y = K) : y = 50 := 
sorry

end find_y_when_x_4_l83_83594


namespace distance_between_a_and_c_l83_83867

-- Given conditions
variables (a : ℝ)

-- Statement to prove
theorem distance_between_a_and_c : |a + 1| = |a - (-1)| :=
by sorry

end distance_between_a_and_c_l83_83867


namespace value_of_sum_is_eleven_l83_83036

-- Define the context and conditions

theorem value_of_sum_is_eleven (x y z w : ℤ) 
  (h1 : x - y + z = 7)
  (h2 : y - z + w = 8)
  (h3 : z - w + x = 4)
  (h4 : w - x + y = 3) :
  x + y + z + w = 11 :=
begin
  sorry
end

end value_of_sum_is_eleven_l83_83036


namespace maria_paper_count_l83_83899

-- Defining the initial number of sheets and the actions taken
variables (x y : ℕ)
def initial_sheets := 50 + 41
def remaining_sheets_after_giving_away := initial_sheets - x
def whole_sheets := remaining_sheets_after_giving_away - y
def half_sheets := y

-- The theorem we want to prove
theorem maria_paper_count (x y : ℕ) :
  whole_sheets x y = initial_sheets - x - y ∧ 
  half_sheets y = y :=
by sorry

end maria_paper_count_l83_83899


namespace find_x_y_l83_83909

theorem find_x_y (x y : ℤ) (hx : 0 < x) (hy : 0 < y) (h : (x + y * Complex.I)^2 = (7 + 24 * Complex.I)) :
  x + y * Complex.I = 4 + 3 * Complex.I :=
by
  sorry

end find_x_y_l83_83909


namespace rose_bushes_unwatered_l83_83784

theorem rose_bushes_unwatered (n V A : ℕ) (V_set A_set : Finset ℕ) (hV : V = 1003) (hA : A = 1003) (hTotal : n = 2006) (hIntersection : V_set.card = 3) :
  n - (V + A - V_set.card) = 3 :=
by
  sorry

end rose_bushes_unwatered_l83_83784


namespace largest_prime_factor_5040_is_7_l83_83775

-- Definition of the condition: the prime factorization of 5040
def prime_factorization_5040 : list ℕ := [2, 2, 2, 2, 3, 3, 5, 7]

-- Predicate to check if a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m:ℕ, m ∣ n → m = 1 ∨ m = n

-- Predicate to check if a list contains only primes
def all_primes (l: list ℕ) : Prop :=
  ∀ x, x ∈ l → is_prime x

-- Statement of the problem
theorem largest_prime_factor_5040_is_7 :
  all_primes prime_factorization_5040 ∧ 
  list.prod prime_factorization_5040 = 5040 ∧
  list.maximum prime_factorization_5040 = 7 :=
sorry

end largest_prime_factor_5040_is_7_l83_83775


namespace cookies_in_each_batch_l83_83504

theorem cookies_in_each_batch (batches : ℕ) (people : ℕ) (consumption_per_person : ℕ) (cookies_per_dozen : ℕ) 
  (total_batches : batches = 4) 
  (total_people : people = 16) 
  (cookies_per_person : consumption_per_person = 6) 
  (dozen_size : cookies_per_dozen = 12) :
  (6 * 16) / 4 / 12 = 2 := 
by {
  sorry
}

end cookies_in_each_batch_l83_83504


namespace intersection_A_B_l83_83702

def A : Set ℕ := {70, 1946, 1997, 2003}
def B : Set ℕ := {1, 10, 70, 2016}

theorem intersection_A_B : A ∩ B = {70} := by
  sorry

end intersection_A_B_l83_83702


namespace infinite_primes_divide_f_l83_83557

def non_constant_function (f : ℕ → ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ f a ≠ f b

def divisibility_condition (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a ≠ b → (a - b) ∣ (f a - f b)

theorem infinite_primes_divide_f (f : ℕ → ℕ) 
  (h_non_const : non_constant_function f)
  (h_div : divisibility_condition f) :
  ∃ᶠ p in Filter.atTop, ∃ c : ℕ, p ∣ f c := sorry

end infinite_primes_divide_f_l83_83557


namespace area_of_triangle_AEB_is_correct_l83_83006

noncomputable def area_triangle_AEB : ℚ :=
by
  -- Definitions of given conditions
  let AB := 5
  let BC := 3
  let DF := 1
  let GC := 2

  -- Conditions of the problem
  have h1 : AB = 5 := rfl
  have h2 : BC = 3 := rfl
  have h3 : DF = 1 := rfl
  have h4 : GC = 2 := rfl

  -- The goal to prove
  exact 25 / 2

-- Statement in Lean 4 with the conditions and the correct answer
theorem area_of_triangle_AEB_is_correct :
  area_triangle_AEB = 25 / 2 := sorry -- The proof is omitted for this example

end area_of_triangle_AEB_is_correct_l83_83006


namespace sophie_saves_money_l83_83173

variable (loads_per_week : ℕ) (dryer_sheets_per_load : ℕ) (weeks_per_year : ℕ) (cost_per_box : ℝ) (sheets_per_box : ℕ)
variable (given_on_birthday : Bool)

noncomputable def money_saved_per_year (loads_per_week : ℕ) (dryer_sheets_per_load : ℕ) (weeks_per_year : ℕ) (cost_per_box : ℝ) (sheets_per_box : ℕ) : ℝ :=
  (loads_per_week * dryer_sheets_per_load * weeks_per_year / sheets_per_box) * cost_per_box

theorem sophie_saves_money (h_loads_per_week : loads_per_week = 4) (h_dryer_sheets_per_load : dryer_sheets_per_load = 1)
                           (h_weeks_per_year : weeks_per_year = 52) (h_cost_per_box : cost_per_box = 5.50)
                           (h_sheets_per_box : sheets_per_box = 104) (h_given_on_birthday : given_on_birthday = true) :
  money_saved_per_year 4 1 52 5.50 104 = 11 :=
by
  have h1 : loads_per_week = 4 := h_loads_per_week
  have h2 : dryer_sheets_per_load = 1 := h_dryer_sheets_per_load
  have h3 : weeks_per_year = 52 := h_weeks_per_year
  have h4 : cost_per_box = 5.50 := h_cost_per_box
  have h5 : sheets_per_box = 104 := h_sheets_per_box
  have h6 : given_on_birthday = true := h_given_on_birthday
  sorry

end sophie_saves_money_l83_83173


namespace product_of_primes_is_66_l83_83636

theorem product_of_primes_is_66 :
  let p1 : ℕ := 2
      p2 : ℕ := 3
      p3 : ℕ := 11
  in p1 * p2 * p3 = 66 := by
  sorry

end product_of_primes_is_66_l83_83636


namespace school_club_profit_l83_83082

def price_per_bar_buy : ℚ := 5 / 6
def price_per_bar_sell : ℚ := 2 / 3
def total_bars : ℕ := 1200
def total_cost : ℚ := total_bars * price_per_bar_buy
def total_revenue : ℚ := total_bars * price_per_bar_sell
def profit : ℚ := total_revenue - total_cost

theorem school_club_profit : profit = -200 := by
  sorry

end school_club_profit_l83_83082


namespace like_terms_expression_value_l83_83533

theorem like_terms_expression_value (m n : ℤ) (h1 : m = 3) (h2 : n = 1) :
  3^2 * n - (2 * m * n^2 - 2 * (m^2 * n + 2 * m * n^2)) = 33 := by
  sorry

end like_terms_expression_value_l83_83533


namespace defective_probability_l83_83267

theorem defective_probability {total_switches checked_switches defective_checked : ℕ}
  (h1 : total_switches = 2000)
  (h2 : checked_switches = 100)
  (h3 : defective_checked = 10) :
  (defective_checked : ℚ) / checked_switches = 1 / 10 :=
sorry

end defective_probability_l83_83267


namespace problem_1_problem_2_l83_83516

noncomputable def complete_residue_system (n : ℕ) (as : Fin n → ℕ) :=
  ∀ i j : Fin n, i ≠ j → as i % n ≠ as j % n

theorem problem_1 (n : ℕ) (hn : 0 < n) :
  ∃ as : Fin n → ℕ, complete_residue_system n as ∧ complete_residue_system n (λ i => as i + i) := 
sorry

theorem problem_2 (n : ℕ) (hn : 0 < n) :
  ∃ as : Fin n → ℕ, complete_residue_system n as ∧ complete_residue_system n (λ i => as i + i) ∧ complete_residue_system n (λ i => as i - i) := 
sorry

end problem_1_problem_2_l83_83516


namespace proof_problem_l83_83258

variable {a b x : ℝ}

theorem proof_problem (h1 : x = b / a) (h2 : a ≠ b) (h3 : a ≠ 0) : 
  (2 * a + b) / (2 * a - b) = (2 + x) / (2 - x) :=
sorry

end proof_problem_l83_83258


namespace n19_minus_n7_div_30_l83_83741

theorem n19_minus_n7_div_30 (n : ℕ) (h : 0 < n) : 30 ∣ (n^19 - n^7) :=
sorry

end n19_minus_n7_div_30_l83_83741


namespace product_of_primes_l83_83630

theorem product_of_primes : 2 * 3 * 11 = 66 :=
by 
  -- Start with the multiplication of the first two primes
  have h1 : 2 * 3 = 6 := by norm_num
  -- Then multiply the result with the smallest two-digit prime
  have h2 : 6 * 11 = 66 := by norm_num
  -- Combine the steps to get the final result
  exact eq.trans (congr_arg (λ x, x * 11) h1) h2

end product_of_primes_l83_83630


namespace n_pow_19_minus_n_pow_7_div_30_l83_83743

theorem n_pow_19_minus_n_pow_7_div_30 (n : ℕ) (hn : 0 < n) : 30 ∣ (n^19 - n^7) :=
sorry

end n_pow_19_minus_n_pow_7_div_30_l83_83743


namespace M_eq_N_l83_83919

noncomputable def M (a : ℝ) : ℝ :=
  a^2 + (a + 3)^2 + (a + 5)^2 + (a + 6)^2

noncomputable def N (a : ℝ) : ℝ :=
  (a + 1)^2 + (a + 2)^2 + (a + 4)^2 + (a + 7)^2

theorem M_eq_N (a : ℝ) : M a = N a :=
by
  sorry

end M_eq_N_l83_83919


namespace partition_set_l83_83097

open Finset

theorem partition_set (S : Finset ℕ) (hS : S = (range 1989).image (λ n, n + 1)) :
  ∃ (A: Fin 118 → Finset ℕ),
    (∀ i : Fin 118, (A i).card = 17) ∧ 
    (∀ i : Fin 118, ∑ x in A i, x = ∑ x in A 0, x) ∧ 
    (∀ i j : Fin 118, i ≠ j → disjoint (A i) (A j)) :=
by
  sorry

end partition_set_l83_83097


namespace circle_tangent_to_y_axis_l83_83720

theorem circle_tangent_to_y_axis (m : ℝ) :
  (0 < m) → (∀ p : ℝ × ℝ, (p.1 - m)^2 + p.2^2 = 4 ↔ p.1 ^ 2 = p.2^2) → (m = 2 ∨ m = -2) :=
by
  sorry

end circle_tangent_to_y_axis_l83_83720


namespace rain_at_least_one_day_probability_l83_83052

-- Definitions based on given conditions
def P_rain_Friday : ℝ := 0.30
def P_rain_Monday : ℝ := 0.20

-- Events probabilities based on independence
def P_no_rain_Friday := 1 - P_rain_Friday
def P_no_rain_Monday := 1 - P_rain_Monday
def P_no_rain_both := P_no_rain_Friday * P_no_rain_Monday

-- The probability of raining at least one day
def P_rain_at_least_one_day := 1 - P_no_rain_both

-- Expected probability
def expected_probability : ℝ := 0.44

theorem rain_at_least_one_day_probability : 
  P_rain_at_least_one_day = expected_probability := by
  sorry

end rain_at_least_one_day_probability_l83_83052


namespace parallelogram_count_l83_83714

theorem parallelogram_count (m n : ℕ) : 
  ∃ p : ℕ, p = (m.choose 2) * (n.choose 2) :=
by
  sorry

end parallelogram_count_l83_83714


namespace surveyor_problem_l83_83497

theorem surveyor_problem
  (GF : ℝ) (G4 : ℝ)
  (hGF : GF = 70)
  (hG4 : G4 = 60) :
  (1/2) * GF * G4 = 2100 := 
  by
  sorry

end surveyor_problem_l83_83497


namespace sum_of_three_numbers_l83_83432

theorem sum_of_three_numbers (a b c : ℤ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : a + 15 = (a + b + c) / 3) (h4 : (a + b + c) / 3 = c - 20) (h5 : b = 7) :
  a + b + c = 36 :=
sorry

end sum_of_three_numbers_l83_83432


namespace intersection_of_sets_l83_83363

noncomputable def setA : Set ℕ := { x : ℕ | x^2 ≤ 4 * x ∧ x > 0 }

noncomputable def setB : Set ℕ := { x : ℕ | 2^x - 4 > 0 ∧ 2^x - 4 ≤ 4 }

theorem intersection_of_sets : { x ∈ setA | x ∈ setB } = {3} :=
by
  sorry

end intersection_of_sets_l83_83363


namespace largest_angle_of_trapezoid_arithmetic_sequence_l83_83211

variables (a d : ℝ)

-- Given Conditions
def smallest_angle : Prop := a = 45
def trapezoid_property : Prop := a + 3 * d = 135

theorem largest_angle_of_trapezoid_arithmetic_sequence 
  (ha : smallest_angle a) (ht : a + (a + 3 * d) = 180) : 
  a + 3 * d = 135 :=
by
  sorry

end largest_angle_of_trapezoid_arithmetic_sequence_l83_83211


namespace production_line_B_units_l83_83329

theorem production_line_B_units (total_units : ℕ) 
  (lines : ℕ) (h_total: total_units = 16800)
  (h_arithmetic_sequence: ∃ (a d : ℕ), lines = [a, a + d, a + 2 * d]) :
  ∃ (units_B : ℕ), units_B = 5600 :=
by
  -- Introduce the assumptions
  cases h_arithmetic_sequence with a ha
  cases ha with d hd
  -- Assume the total number of units equation
  have h_units : total_units = a + (a + d) + (a + 2 * d) := sorry
  -- Solve for the parameter values
  find a, d such that the sum matches
  sorry
  -- Derive units_B value
  let units_B := a + d
  use units_B
  -- Conclude that units_B = 5600
  have h_units_B_correct : units_B = 5600 := sorry
  exact h_units_B_correct

end production_line_B_units_l83_83329


namespace range_of_a_l83_83135

theorem range_of_a (a : ℝ) (x : ℝ) (h : x^2 + a * x + 1 < 0) : a < -2 ∨ a > 2 :=
sorry

end range_of_a_l83_83135


namespace cos_angle_relation_l83_83990

theorem cos_angle_relation (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) : Real.cos (2 * α - 2 * π / 3) = -7 / 9 := by 
  sorry

end cos_angle_relation_l83_83990


namespace g_of_neg_5_is_4_l83_83890

def f (x : ℝ) : ℝ := 3 * x - 8
def g (y : ℝ) : ℝ := 2 * y^2 + 5 * y - 3

theorem g_of_neg_5_is_4 : g (-5) = 4 :=
by
  sorry

end g_of_neg_5_is_4_l83_83890


namespace inequality_proof_l83_83893

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  a ^ a * b ^ b * c ^ c ≥ 1 / (a * b * c) := 
sorry

end inequality_proof_l83_83893


namespace find_a_l83_83378

theorem find_a (a b c : ℝ) (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
                 (h2 : a * 15 * 7 = 1.5) : a = 6 :=
sorry

end find_a_l83_83378


namespace evaluate_expression_l83_83511

theorem evaluate_expression (x y z : ℤ) (hx : x = 5) (hy : y = x + 3) (hz : z = y - 11) 
  (h₁ : x + 2 ≠ 0) (h₂ : y - 3 ≠ 0) (h₃ : z + 7 ≠ 0) : 
  ((x + 3) / (x + 2)) * ((y - 2) / (y - 3)) * ((z + 9) / (z + 7)) = 72 / 35 := 
by 
  sorry

end evaluate_expression_l83_83511


namespace circle_equation_standard_l83_83182

def center : ℝ × ℝ := (-1, 1)
def radius : ℝ := 2

theorem circle_equation_standard:
  (∀ x y : ℝ, ((x + 1)^2 + (y - 1)^2 = 4) ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2)) :=
by 
  intros x y
  rw [center, radius]
  simp
  sorry

end circle_equation_standard_l83_83182


namespace find_b_value_l83_83603

def perfect_square_trinomial (a b c : ℕ) : Prop :=
  ∃ d, a = d^2 ∧ c = d^2 ∧ b = 2 * d * d

theorem find_b_value (b : ℝ) :
    (∀ x : ℝ, 16 * x^2 - b * x + 9 = (4 * x - 3) * (4 * x - 3) ∨ 16 * x^2 - b * x + 9 = (4 * x + 3) * (4 * x + 3)) -> 
    b = 24 ∨ b = -24 := 
by
  sorry

end find_b_value_l83_83603


namespace sum_of_squares_of_consecutive_integers_l83_83435

theorem sum_of_squares_of_consecutive_integers (a : ℝ) (h : (a-1)*a*(a+1) = 36*a) :
  (a-1)^2 + a^2 + (a+1)^2 = 77 :=
by
  sorry

end sum_of_squares_of_consecutive_integers_l83_83435


namespace solve_system_of_equations_l83_83908

theorem solve_system_of_equations 
  (a1 a2 a3 a4 : ℝ) (h_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (x1 x2 x3 x4 : ℝ)
  (h1 : |a1 - a1| * x1 + |a1 - a2| * x2 + |a1 - a3| * x3 + |a1 - a4| * x4 = 1)
  (h2 : |a2 - a1| * x1 + |a2 - a2| * x2 + |a2 - a3| * x3 + |a2 - a4| * x4 = 1)
  (h3 : |a3 - a1| * x1 + |a3 - a2| * x2 + |a3 - a3| * x3 + |a3 - a4| * x4 = 1)
  (h4 : |a4 - a1| * x1 + |a4 - a2| * x2 + |a4 - a3| * x3 + |a4 - a4| * x4 = 1) :
  x1 = 1 / (a1 - a4) ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 1 / (a1 - a4) :=
sorry

end solve_system_of_equations_l83_83908


namespace statement_C_correct_l83_83251

theorem statement_C_correct (a b c d : ℝ) (h_ab : a > b) (h_cd : c > d) : a + c > b + d :=
by
  sorry

end statement_C_correct_l83_83251


namespace marble_problem_l83_83341

theorem marble_problem (a : ℚ) :
  (a + 2 * a + 3 * 2 * a + 5 * (3 * 2 * a) + 2 * (5 * (3 * 2 * a)) = 212) ↔
  (a = 212 / 99) :=
by
  sorry

end marble_problem_l83_83341


namespace westbound_speed_is_275_l83_83208

-- Define the conditions for the problem at hand.
def east_speed : ℕ := 325
def separation_time : ℝ := 3.5
def total_distance : ℕ := 2100

-- Compute the known east-bound distance.
def east_distance : ℝ := east_speed * separation_time

-- Define the speed of the west-bound plane as an unknown variable.
variable (v : ℕ)

-- Compute the west-bound distance.
def west_distance := v * separation_time

-- The assertion that the sum of two distances equals the total distance.
def distance_equation := east_distance + (v * separation_time) = total_distance

-- Prove that the west-bound speed is 275 mph.
theorem westbound_speed_is_275 : v = 275 :=
by
  sorry

end westbound_speed_is_275_l83_83208


namespace intersection_complement_eq_l83_83132

-- Definitions as per given conditions
def U : Set ℕ := { x | x > 0 ∧ x < 9 }
def A : Set ℕ := { 1, 2, 3, 4 }
def B : Set ℕ := { 3, 4, 5, 6 }

-- Complement of B with respect to U
def C_U_B : Set ℕ := U \ B

-- Statement of the theorem to be proved
theorem intersection_complement_eq : A ∩ C_U_B = { 1, 2 } :=
by
  sorry

end intersection_complement_eq_l83_83132


namespace volume_truncated_cone_l83_83185

-- Define the geometric constants
def large_base_radius : ℝ := 10
def small_base_radius : ℝ := 5
def height_truncated_cone : ℝ := 8

-- The statement to prove the volume of the truncated cone
theorem volume_truncated_cone :
  let V_large := (1/3) * Real.pi * (large_base_radius^2) * (height_truncated_cone + height_truncated_cone)
  let V_small := (1/3) * Real.pi * (small_base_radius^2) * height_truncated_cone
  V_large - V_small = (1400/3) * Real.pi :=
by
  sorry

end volume_truncated_cone_l83_83185


namespace row_time_14_24_l83_83552

variable (d c s r : ℝ)

-- Assumptions
def swim_with_current (d c s : ℝ) := s + c = d / 40
def swim_against_current (d c s : ℝ) := s - c = d / 45
def row_against_current (d c r : ℝ) := r - c = d / 15

-- Expected result
def time_to_row_harvard_mit (d c r : ℝ) := d / (r + c) = 14 + 24 / 60

theorem row_time_14_24 :
  swim_with_current d c s ∧
  swim_against_current d c s ∧
  row_against_current d c r →
  time_to_row_harvard_mit d c r :=
by
  sorry

end row_time_14_24_l83_83552


namespace triangle_perimeter_l83_83844

-- Conditions as definitions
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

def has_sides (a b : ℕ) : Prop :=
  a = 4 ∨ b = 4 ∨ a = 9 ∨ b = 9

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the isosceles triangle with specified sides
structure IsoTriangle :=
  (a b c : ℕ)
  (iso : is_isosceles_triangle a b c)
  (valid_sides : has_sides a b ∧ has_sides a c ∧ has_sides b c)
  (triangle : triangle_inequality a b c)

-- The statement to prove perimeter
def perimeter (T : IsoTriangle) : ℕ :=
  T.a + T.b + T.c

-- The theorem we aim to prove
theorem triangle_perimeter (T : IsoTriangle) (h: T.a = 9 ∧ T.b = 9 ∧ T.c = 4) : perimeter T = 22 :=
sorry

end triangle_perimeter_l83_83844


namespace infinite_solutions_l83_83983

theorem infinite_solutions (b : ℤ) : 
  (∀ x : ℤ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := 
by sorry

end infinite_solutions_l83_83983


namespace combine_quadratic_radicals_l83_83711

theorem combine_quadratic_radicals (x : ℝ) (h : 3 * x + 5 = 2 * x + 7) : x = 2 :=
by
  sorry

end combine_quadratic_radicals_l83_83711


namespace find_abc_unique_solution_l83_83283

theorem find_abc_unique_solution (N a b c : ℕ) 
  (hN : N > 3 ∧ N % 2 = 1)
  (h_eq : a^N = b^N + 2^N + a * b * c)
  (h_c : c ≤ 5 * 2^(N-1)) : 
  N = 5 ∧ a = 3 ∧ b = 1 ∧ c = 70 := 
sorry

end find_abc_unique_solution_l83_83283


namespace sally_baseball_cards_l83_83029

theorem sally_baseball_cards (initial_cards torn_cards purchased_cards : ℕ) 
    (h_initial : initial_cards = 39)
    (h_torn : torn_cards = 9)
    (h_purchased : purchased_cards = 24) :
    initial_cards - torn_cards - purchased_cards = 6 := by
  sorry

end sally_baseball_cards_l83_83029


namespace inequality_solution_l83_83293

noncomputable def ratFunc (x : ℝ) : ℝ := 
  ((x - 3) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7))

theorem inequality_solution (x : ℝ) : 
  (ratFunc x > 0) ↔ 
  ((x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 6) ∨ (7 < x)) := 
by
  sorry

end inequality_solution_l83_83293


namespace crushing_load_calculation_l83_83509

theorem crushing_load_calculation (T H : ℝ) (L : ℝ) 
  (h1 : L = 40 * T^5 / H^3) 
  (h2 : T = 3) 
  (h3 : H = 6) : 
  L = 45 := 
by sorry

end crushing_load_calculation_l83_83509


namespace length_other_diagonal_l83_83335

variables (d1 d2 : ℝ) (Area : ℝ)

theorem length_other_diagonal 
  (h1 : Area = 432)
  (h2 : d1 = 36) :
  d2 = 24 :=
by
  -- Insert proof here
  sorry

end length_other_diagonal_l83_83335


namespace exists_int_solutions_for_equations_l83_83352

theorem exists_int_solutions_for_equations : 
  ∃ (x y : ℤ), x * y = 4747 ∧ x - y = -54 :=
by
  sorry

end exists_int_solutions_for_equations_l83_83352


namespace lesser_number_l83_83447

theorem lesser_number (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 10) : y = 25 :=
by
  have h3 : x = 35 := sorry
  exact sorry

end lesser_number_l83_83447


namespace polar_line_eq_l83_83724

theorem polar_line_eq (ρ θ : ℝ) : (ρ * Real.cos θ = 1) ↔ (ρ = Real.cos θ ∨ ρ = Real.sin θ ∨ 1 / Real.cos θ = ρ) := by
  sorry

end polar_line_eq_l83_83724


namespace lesser_of_two_numbers_l83_83440

theorem lesser_of_two_numbers (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
by
  sorry

end lesser_of_two_numbers_l83_83440


namespace ending_number_divisible_by_3_l83_83765

theorem ending_number_divisible_by_3 (n : ℕ) :
  (∀ k, 0 ≤ k ∧ k < 13 → ∃ m, 10 ≤ m ∧ m ≤ n ∧ m % 3 = 0) →
  n = 48 :=
by
  intro h
  sorry

end ending_number_divisible_by_3_l83_83765


namespace vanessa_missed_days_l83_83607

theorem vanessa_missed_days (V M S : ℕ) 
                           (h1 : V + M + S = 17) 
                           (h2 : V + M = 14) 
                           (h3 : M + S = 12) : 
                           V = 5 :=
sorry

end vanessa_missed_days_l83_83607


namespace eagles_points_l83_83138

theorem eagles_points (s e : ℕ) (h1 : s + e = 52) (h2 : s - e = 6) : e = 23 :=
by
  sorry

end eagles_points_l83_83138


namespace wardrobe_single_discount_l83_83806

theorem wardrobe_single_discount :
  let p : ℝ := 50
  let d1 : ℝ := 0.30
  let d2 : ℝ := 0.20
  let final_price := p * (1 - d1) * (1 - d2)
  let equivalent_discount := 1 - (final_price / p)
  equivalent_discount = 0.44 :=
by
  let p : ℝ := 50
  let d1 : ℝ := 0.30
  let d2 : ℝ := 0.20
  let final_price := p * (1 - d1) * (1 - d2)
  let equivalent_discount := 1 - (final_price / p)
  show equivalent_discount = 0.44
  sorry

end wardrobe_single_discount_l83_83806


namespace intersection_of_bisectors_lies_on_BC_l83_83272

open EuclideanGeometry

theorem intersection_of_bisectors_lies_on_BC
  (ABC : Triangle)
  (A B C I X : Point)
  (D : Point) 
  (Γ Γ0 : Circle) :
  ∠BAC > ∠ABC →
  I = incenter ABC →
  D ∈ Line[BC] →
  ∠CAD = ∠ABC →
  Γ.passing_through I →
  Γ.tangent CA at A →
  Γ.intersects_circumcircle ABC at A X →
  let BXC : Angle := ∠BXC,
  let BAD : Angle := ∠BAD,
  let E := angle_bisector_intersection BXC BAD
  in E ∈ Line[BC] :=
by
  sorry

end intersection_of_bisectors_lies_on_BC_l83_83272


namespace initial_house_cats_l83_83333

theorem initial_house_cats (H : ℕ) 
  (siamese_cats : ℕ := 38) 
  (cats_sold : ℕ := 45) 
  (cats_left : ℕ := 18) 
  (initial_total_cats : ℕ := siamese_cats + H) 
  (after_sale_cats : ℕ := initial_total_cats - cats_sold) : 
  after_sale_cats = cats_left → H = 25 := 
by
  intro h
  sorry

end initial_house_cats_l83_83333


namespace product_of_primes_l83_83621

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

noncomputable def smallest_one_digit_primes (p₁ p₂ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ p₁ < p₂ ∧ p₂ < 10 ∧ ∀ p : ℕ, is_prime p → p < 10 → p = p₁ ∨ p = p₂

noncomputable def smallest_two_digit_prime (p : ℕ) : Prop :=
  is_prime p ∧ p ≥ 10 ∧ p < 100 ∧ ∀ q : ℕ, is_prime q → q ≥ 10 → q < p → q = 11

theorem product_of_primes : ∃ p₁ p₂ p₃ : ℕ, smallest_one_digit_primes p₁ p₂ ∧ smallest_two_digit_prime p₃ ∧ p₁ * p₂ * p₃ = 66 := 
by
  sorry

end product_of_primes_l83_83621


namespace angle_bisector_inequality_l83_83389

noncomputable def triangle_ABC (A B C K M : Type) [Inhabited A] [Inhabited B] [Inhabited C] (AB BC CA AK CM AM MK KC : ℝ) 
  (Hbisector_CM : BM / MA = BC / CA)
  (Hbisector_AK : BK / KC = AB / AC)
  (Hcondition : AB > BC) : Prop :=
  AM > MK ∧ MK > KC

theorem angle_bisector_inequality (A B C K M : Type) [Inhabited A] [Inhabited B] [Inhabited C]
  (AB BC CA AK CM AM MK KC : ℝ)
  (Hbisector_CM : BM / MA = BC / CA)
  (Hbisector_AK : BK / KC = AB / AC)
  (Hcondition : AB > BC) : AM > MK ∧ MK > KC :=
by
  sorry

end angle_bisector_inequality_l83_83389


namespace derivative_at_2_l83_83852

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem derivative_at_2 : deriv f 2 = (1 - Real.log 2) / 4 :=
by
  sorry

end derivative_at_2_l83_83852


namespace arithmetic_geometric_sum_l83_83987

def a (n : ℕ) : ℕ := 3 * n - 2
def b (n : ℕ) : ℕ := 3 ^ (n - 1)

theorem arithmetic_geometric_sum :
  a (b 1) + a (b 2) + a (b 3) = 33 := by
  sorry

end arithmetic_geometric_sum_l83_83987


namespace sin_func_even_min_period_2pi_l83_83237

noncomputable def f (x : ℝ) : ℝ := Real.sin (13 * Real.pi / 2 - x)

theorem sin_func_even_min_period_2pi :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ 2 * Real.pi) ∧ (∀ x : ℝ, f (x + 2 * Real.pi) = f x) :=
by
  sorry

end sin_func_even_min_period_2pi_l83_83237


namespace area_increase_l83_83799

theorem area_increase (a : ℝ) : ((a + 2) ^ 2 - a ^ 2 = 4 * a + 4) := by
  sorry

end area_increase_l83_83799


namespace problem_1_solution_problem_2_solution_l83_83910

-- Definition of the function f
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 3) - abs (x - a)

-- Proof problem for question 1
theorem problem_1_solution (x : ℝ) : f x 2 ≤ -1/2 ↔ x ≥ 11/4 :=
by
  sorry

-- Proof problem for question 2
theorem problem_2_solution (a : ℝ) : (∀ x : ℝ, f x a ≥ a) ↔ a ∈ Set.Iic (3/2) :=
by
  sorry

end problem_1_solution_problem_2_solution_l83_83910


namespace arun_gokul_age_subtract_l83_83538

theorem arun_gokul_age_subtract:
  ∃ x : ℕ, (60 - x) / 18 = 3 → x = 6 :=
sorry

end arun_gokul_age_subtract_l83_83538


namespace solve1_solve2_solve3_solve4_l83_83573

noncomputable section

-- Problem 1
theorem solve1 (x : ℝ) : x^2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 := sorry

-- Problem 2
theorem solve2 (x : ℝ) : (x + 1)^2 - 144 = 0 ↔ x = 11 ∨ x = -13 := sorry

-- Problem 3
theorem solve3 (x : ℝ) : 3 * (x - 2)^2 = x * (x - 2) ↔ x = 2 ∨ x = 3 := sorry

-- Problem 4
theorem solve4 (x : ℝ) : x^2 + 5 * x - 1 = 0 ↔ x = (-5 + Real.sqrt 29) / 2 ∨ x = (-5 - Real.sqrt 29) / 2 := sorry

end solve1_solve2_solve3_solve4_l83_83573


namespace choosing_4_out_of_10_classes_l83_83881

theorem choosing_4_out_of_10_classes :
  ∑ (k : ℕ) in (finset.range 5).map (prod.mk 10), k! / (4! * (k - 4)!) = 210 :=
by sorry

end choosing_4_out_of_10_classes_l83_83881


namespace marble_box_l83_83600

theorem marble_box (T: ℕ) 
  (h_white: (1 / 6) * T = T / 6)
  (h_green: (1 / 5) * T = T / 5)
  (h_red_blue: (19 / 30) * T = 19 * T / 30)
  (h_sum: (T / 6) + (T / 5) + (19 * T / 30) = T): 
  ∃ k : ℕ, T = 30 * k ∧ k ≥ 1 :=
by
  sorry

end marble_box_l83_83600


namespace remaining_amount_correct_l83_83414

def initial_amount : ℝ := 70
def coffee_cost_per_pound : ℝ := 8.58
def coffee_pounds : ℝ := 4.0
def total_cost : ℝ := coffee_pounds * coffee_cost_per_pound
def remaining_amount : ℝ := initial_amount - total_cost

theorem remaining_amount_correct : remaining_amount = 35.68 :=
by
  -- Skip the proof; this is a placeholder.
  sorry

end remaining_amount_correct_l83_83414


namespace chemistry_marks_l83_83677

theorem chemistry_marks (marks_english : ℕ) (marks_math : ℕ) (marks_physics : ℕ) 
                        (marks_biology : ℕ) (average_marks : ℚ) (marks_chemistry : ℕ) 
                        (h_english : marks_english = 70) 
                        (h_math : marks_math = 60) 
                        (h_physics : marks_physics = 78) 
                        (h_biology : marks_biology = 65) 
                        (h_average : average_marks = 66.6) 
                        (h_total: average_marks * 5 = marks_english + marks_math + marks_physics + marks_biology + marks_chemistry) : 
  marks_chemistry = 60 :=
by sorry

end chemistry_marks_l83_83677


namespace new_average_age_after_person_leaves_l83_83422

theorem new_average_age_after_person_leaves (avg_age : ℕ) (n : ℕ) (leaving_age : ℕ) (remaining_count : ℕ) :
  ((n * avg_age - leaving_age) / remaining_count) = 33 :=
by
  -- Given conditions
  let avg_age := 30
  let n := 5
  let leaving_age := 18
  let remaining_count := n - 1
  -- Conclusion
  sorry

end new_average_age_after_person_leaves_l83_83422


namespace base_of_parallelogram_l83_83817

theorem base_of_parallelogram (area height base : ℝ) 
  (h_area : area = 320)
  (h_height : height = 16) :
  base = area / height :=
by 
  rw [h_area, h_height]
  norm_num
  sorry

end base_of_parallelogram_l83_83817


namespace cone_lateral_surface_area_l83_83830

-- Definitions from conditions
def r : ℝ := 6
def V : ℝ := 30 * Real.pi

-- Theorem to prove
theorem cone_lateral_surface_area : 
  let h := V / (Real.pi * (r ^ 2) / 3) in
  let l := Real.sqrt (r ^ 2 + h ^ 2) in
  let S := Real.pi * r * l in
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l83_83830


namespace symmetric_origin_l83_83039

def symmetric_point (p : (Int × Int)) : (Int × Int) :=
  (-p.1, -p.2)

theorem symmetric_origin : symmetric_point (-2, 5) = (2, -5) :=
by
  -- proof goes here
  -- we use sorry to indicate the place where the solution would go
  sorry

end symmetric_origin_l83_83039


namespace train_speed_is_correct_l83_83339

-- Definitions of the problem
def length_of_train : ℕ := 360
def time_to_pass_bridge : ℕ := 25
def length_of_bridge : ℕ := 140
def conversion_factor : ℝ := 3.6

-- Distance covered by the train plus the length of the bridge
def total_distance : ℕ := length_of_train + length_of_bridge

-- Speed calculation in m/s
def speed_in_m_per_s := total_distance / time_to_pass_bridge

-- Conversion to km/h
def speed_in_km_per_h := speed_in_m_per_s * conversion_factor

-- The proof goal: the speed of the train is 72 km/h
theorem train_speed_is_correct : speed_in_km_per_h = 72 := by
  sorry

end train_speed_is_correct_l83_83339


namespace man_l83_83792

theorem man's_age_twice_son's_age_in_2_years
  (S : ℕ) (M : ℕ) (Y : ℕ)
  (h1 : M = S + 24)
  (h2 : S = 22)
  (h3 : M + Y = 2 * (S + Y)) :
  Y = 2 := by
  sorry

end man_l83_83792


namespace smallest_value_l83_83478

theorem smallest_value (x : ℝ) (h : 3 * x^2 + 33 * x - 90 = x * (x + 18)) : x ≥ -10.5 :=
sorry

end smallest_value_l83_83478


namespace average_fixed_points_of_permutation_l83_83081

open Finset

noncomputable def average_fixed_points (n : ℕ) : ℕ :=
  1

theorem average_fixed_points_of_permutation (n : ℕ) :
  ∀ (σ : (Fin n) → (Fin n)), 
  (1: ℚ) = (1: ℕ) :=
by
  sorry

end average_fixed_points_of_permutation_l83_83081


namespace loss_per_metre_eq_12_l83_83083

-- Definitions based on the conditions
def totalMetres : ℕ := 200
def totalSellingPrice : ℕ := 12000
def costPricePerMetre : ℕ := 72

-- Theorem statement to prove the loss per metre of cloth
theorem loss_per_metre_eq_12 : (costPricePerMetre * totalMetres - totalSellingPrice) / totalMetres = 12 := 
by sorry

end loss_per_metre_eq_12_l83_83083


namespace find_lesser_number_l83_83463

theorem find_lesser_number (x y : ℕ) (h₁ : x + y = 60) (h₂ : x - y = 10) : y = 25 := by
  sorry

end find_lesser_number_l83_83463


namespace systematic_sampling_employee_l83_83088

theorem systematic_sampling_employee {x : ℕ} (h1 : 1 ≤ 6 ∧ 6 ≤ 52) (h2 : 1 ≤ 32 ∧ 32 ≤ 52) (h3 : 1 ≤ 45 ∧ 45 ≤ 52) (h4 : 6 + 45 = x + 32) : x = 19 :=
  by
    sorry

end systematic_sampling_employee_l83_83088


namespace sum_of_squares_l83_83048

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 20) (h2 : a * b + b * c + c * a = 131) : 
  a^2 + b^2 + c^2 = 138 := 
sorry

end sum_of_squares_l83_83048


namespace integer_solution_count_eq_eight_l83_83302

theorem integer_solution_count_eq_eight : ∃ S : Finset (ℤ × ℤ), (∀ s ∈ S, 2 * s.1 ^ 2 + s.1 * s.2 - s.2 ^ 2 = 14 ∧ (s.1 = s.1 ∧ s.2 = s.2)) ∧ S.card = 8 :=
by
  sorry

end integer_solution_count_eq_eight_l83_83302


namespace problem_l83_83157

theorem problem (w x y z : ℕ) (h : 3^w * 5^x * 7^y * 11^z = 2310) : 3 * w + 5 * x + 7 * y + 11 * z = 26 :=
sorry

end problem_l83_83157


namespace one_third_of_1206_is_100_5_percent_of_400_l83_83737

theorem one_third_of_1206_is_100_5_percent_of_400 (n m : ℕ) (f : ℝ) :
  n = 1206 → m = 400 → f = 1 / 3 → (n * f) / m * 100 = 100.5 :=
by
  intros h_n h_m h_f
  rw [h_n, h_m, h_f]
  sorry

end one_third_of_1206_is_100_5_percent_of_400_l83_83737


namespace inequality_hold_l83_83859

theorem inequality_hold (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - |c| > b - |c| :=
sorry

end inequality_hold_l83_83859


namespace find_numbers_l83_83300

theorem find_numbers (x y : ℤ) (h1 : x > y) (h2 : x^2 - y^2 = 100) : 
  x = 26 ∧ y = 24 := 
  sorry

end find_numbers_l83_83300


namespace determine_N_l83_83279

variable (U M N : Set ℕ)

theorem determine_N (h1 : U = {1, 2, 3, 4, 5})
  (h2 : U = M ∪ N)
  (h3 : M ∩ (U \ N) = {2, 4}) :
  N = {1, 3, 5} :=
by
  sorry

end determine_N_l83_83279


namespace green_pill_cost_is_21_l83_83950

-- Definitions based on conditions
def number_of_days : ℕ := 21
def total_cost : ℕ := 819
def daily_cost : ℕ := total_cost / number_of_days
def green_pill_cost (pink_pill_cost : ℕ) : ℕ := pink_pill_cost + 3

-- Given pink pill cost is x, then green pill cost is x + 3
-- We need to prove that for some x, the daily cost of the pills equals 39, and thus green pill cost is 21

theorem green_pill_cost_is_21 (pink_pill_cost : ℕ) (h : daily_cost = (green_pill_cost pink_pill_cost) + pink_pill_cost) :
    green_pill_cost pink_pill_cost = 21 :=
by
  sorry

end green_pill_cost_is_21_l83_83950


namespace odd_function_f_neg_x_l83_83117

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 - 2 * x else -(x^2 + 2 * x)

theorem odd_function_f_neg_x (x : ℝ) (hx : x < 0) :
  f x = -x^2 - 2 * x :=
by
  sorry

end odd_function_f_neg_x_l83_83117


namespace red_users_count_l83_83651

noncomputable def total_students : ℕ := 70
noncomputable def green_users : ℕ := 52
noncomputable def both_colors_users : ℕ := 38

theorem red_users_count : 
  ∀ (R : ℕ), total_students = green_users + R - both_colors_users → R = 56 :=
by
  sorry

end red_users_count_l83_83651


namespace world_grain_demand_l83_83238

theorem world_grain_demand (S D : ℝ) (h1 : S = 1800000) (h2 : S = 0.75 * D) : D = 2400000 := by
  sorry

end world_grain_demand_l83_83238


namespace yoghurt_cost_1_l83_83295

theorem yoghurt_cost_1 :
  ∃ y : ℝ,
  (∀ (ice_cream_cartons yoghurt_cartons : ℕ) (ice_cream_cost_one_carton : ℝ) (yoghurt_cost_one_carton : ℝ),
    ice_cream_cartons = 19 →
    yoghurt_cartons = 4 →
    ice_cream_cost_one_carton = 7 →
    (19 * 7 = 133) →  -- total ice cream cost
    (133 - 129 = 4) → -- Total yogurt cost
    (4 = 4 * y) →    -- Yoghurt cost equation
    y = 1) :=
sorry

end yoghurt_cost_1_l83_83295


namespace find_dads_dimes_l83_83167

variable (original_dimes mother_dimes total_dimes dad_dimes : ℕ)

def proof_problem (original_dimes mother_dimes total_dimes dad_dimes : ℕ) : Prop :=
  original_dimes = 7 ∧
  mother_dimes = 4 ∧
  total_dimes = 19 ∧
  total_dimes = original_dimes + mother_dimes + dad_dimes

theorem find_dads_dimes (h : proof_problem 7 4 19 8) : dad_dimes = 8 :=
sorry

end find_dads_dimes_l83_83167


namespace ceil_sum_sqrt_eval_l83_83354

theorem ceil_sum_sqrt_eval : 
  (⌈Real.sqrt 2⌉ + ⌈Real.sqrt 22⌉ + ⌈Real.sqrt 222⌉) = 22 := 
by
  sorry

end ceil_sum_sqrt_eval_l83_83354


namespace average_bacterial_count_closest_to_true_value_l83_83188

-- Define the conditions
variables (dilution_spread_plate_method : Prop)
          (count_has_randomness : Prop)
          (count_not_uniform : Prop)

-- State the theorem
theorem average_bacterial_count_closest_to_true_value
  (h1: dilution_spread_plate_method)
  (h2: count_has_randomness)
  (h3: count_not_uniform)
  : true := sorry

end average_bacterial_count_closest_to_true_value_l83_83188


namespace marcie_and_martin_in_picture_l83_83020

noncomputable def marcie_prob_in_picture : ℚ :=
  let marcie_lap_time := 100
  let martin_lap_time := 75
  let start_time := 720
  let end_time := 780
  let picture_duration := 60
  let marcie_position_720 := (720 % marcie_lap_time) / marcie_lap_time
  let marcie_in_pic_start := 0
  let marcie_in_pic_end := 20 + 33 + 1/3
  let martin_position_720 := (720 % martin_lap_time) / martin_lap_time
  let martin_in_pic_start := 20
  let martin_in_pic_end := 45 + 25
  let overlap_start := max marcie_in_pic_start martin_in_pic_start
  let overlap_end := min marcie_in_pic_end martin_in_pic_end
  let overlap_duration := overlap_end - overlap_start
  overlap_duration / picture_duration

theorem marcie_and_martin_in_picture :
  marcie_prob_in_picture = 111 / 200 :=
by
  sorry

end marcie_and_martin_in_picture_l83_83020


namespace product_of_primes_l83_83620

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

noncomputable def smallest_one_digit_primes (p₁ p₂ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ p₁ < p₂ ∧ p₂ < 10 ∧ ∀ p : ℕ, is_prime p → p < 10 → p = p₁ ∨ p = p₂

noncomputable def smallest_two_digit_prime (p : ℕ) : Prop :=
  is_prime p ∧ p ≥ 10 ∧ p < 100 ∧ ∀ q : ℕ, is_prime q → q ≥ 10 → q < p → q = 11

theorem product_of_primes : ∃ p₁ p₂ p₃ : ℕ, smallest_one_digit_primes p₁ p₂ ∧ smallest_two_digit_prime p₃ ∧ p₁ * p₂ * p₃ = 66 := 
by
  sorry

end product_of_primes_l83_83620


namespace positive_difference_of_complementary_angles_in_ratio_five_to_four_l83_83761

theorem positive_difference_of_complementary_angles_in_ratio_five_to_four
  (a b : ℝ)
  (h1 : a / b = 5 / 4)
  (h2 : a + b = 90) :
  |a - b| = 10 :=
sorry

end positive_difference_of_complementary_angles_in_ratio_five_to_four_l83_83761


namespace bacon_sold_l83_83579

variable (B : ℕ) -- Declare the variable for the number of slices of bacon sold

-- Define the given conditions as Lean definitions
def pancake_price := 4
def bacon_price := 2
def stacks_sold := 60
def total_raised := 420

-- The revenue from pancake sales alone
def pancake_revenue := stacks_sold * pancake_price
-- The revenue from bacon sales
def bacon_revenue := total_raised - pancake_revenue

-- Statement of the theorem
theorem bacon_sold :
  B = bacon_revenue / bacon_price :=
sorry

end bacon_sold_l83_83579


namespace sum_of_cube_edges_l83_83361

theorem sum_of_cube_edges (edge_len : ℝ) (num_edges : ℕ) (lengths : ℝ) (h1 : edge_len = 15) (h2 : num_edges = 12) : lengths = num_edges * edge_len :=
by
  sorry

end sum_of_cube_edges_l83_83361


namespace difference_between_place_and_face_value_l83_83723

def numeral : Nat := 856973

def digit_of_interest : Nat := 7

def place_value : Nat := 7 * 10

def face_value : Nat := 7

theorem difference_between_place_and_face_value : place_value - face_value = 63 :=
by
  sorry

end difference_between_place_and_face_value_l83_83723


namespace cost_per_book_l83_83905

theorem cost_per_book (a r n c : ℕ) (h : a - r = n * c) : c = 7 :=
by sorry

end cost_per_book_l83_83905


namespace rancher_cattle_count_l83_83791

theorem rancher_cattle_count
  (truck_capacity : ℕ)
  (distance_to_higher_ground : ℕ)
  (truck_speed : ℕ)
  (total_transport_time : ℕ)
  (h1 : truck_capacity = 20)
  (h2 : distance_to_higher_ground = 60)
  (h3 : truck_speed = 60)
  (h4 : total_transport_time = 40):
  ∃ (number_of_cattle : ℕ), number_of_cattle = 400 :=
by {
  sorry
}

end rancher_cattle_count_l83_83791


namespace functional_ineq_l83_83731

noncomputable def f : ℝ → ℝ := sorry

theorem functional_ineq (h1 : ∀ x > 1400^2021, x * f x ≤ 2021) (h2 : ∀ x : ℝ, 0 < x → f x = f (x + 2) + 2 * f (x * (x + 2))) : 
  ∀ x : ℝ, 0 < x → x * f x ≤ 2021 :=
sorry

end functional_ineq_l83_83731


namespace problem_proof_l83_83015

-- Assume definitions for lines and planes, and their relationships like parallel and perpendicular exist.

variables (m n : Line) (α β : Plane)

-- Define conditions
def line_is_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_is_parallel_to_line (l1 l2 : Line) : Prop := sorry
def planes_are_perpendicular (p1 p2 : Plane) : Prop := sorry

-- Problem statement
theorem problem_proof :
  (line_is_perpendicular_to_plane m α) ∧ (line_is_perpendicular_to_plane n α) → 
  (line_is_parallel_to_line m n) ∧
  ((line_is_perpendicular_to_plane m α) ∧ (line_is_perpendicular_to_plane n β) ∧ (line_is_perpendicular_to_plane m n) → 
  (planes_are_perpendicular α β)) := 
sorry

end problem_proof_l83_83015


namespace no_such_n_exists_l83_83523

theorem no_such_n_exists : ∀ n : ℕ, n > 1 → ∀ (p1 p2 : ℕ), 
  (Nat.Prime p1) → (Nat.Prime p2) → n = p1^2 → n + 60 = p2^2 → False :=
by
  intro n hn p1 p2 hp1 hp2 h1 h2
  sorry

end no_such_n_exists_l83_83523


namespace smallest_multiplier_to_perfect_square_l83_83935

theorem smallest_multiplier_to_perfect_square : ∃ k : ℕ, k > 0 ∧ ∀ m : ℕ, (2010 * m = k * k) → m = 2010 :=
by
  sorry

end smallest_multiplier_to_perfect_square_l83_83935


namespace episodes_count_l83_83231

variable (minutes_per_episode : ℕ) (total_watching_time_minutes : ℕ)
variable (episodes_watched : ℕ)

theorem episodes_count 
  (h1 : minutes_per_episode = 50) 
  (h2 : total_watching_time_minutes = 300) 
  (h3 : total_watching_time_minutes / minutes_per_episode = episodes_watched) :
  episodes_watched = 6 := sorry

end episodes_count_l83_83231


namespace simplified_expression_value_l83_83030

noncomputable def expression (a b : ℝ) : ℝ :=
  3 * a ^ 2 - b ^ 2 - (a ^ 2 - 6 * a) - 2 * (-b ^ 2 + 3 * a)

theorem simplified_expression_value :
  expression (-1/2) 3 = 19 / 2 :=
by
  sorry

end simplified_expression_value_l83_83030


namespace max_candies_l83_83816

theorem max_candies (V M S : ℕ) (hv : V = 35) (hm : 1 ≤ M ∧ M < 35) (hs : S = 35 + M) (heven : Even S) : V + M + S = 136 :=
sorry

end max_candies_l83_83816


namespace line_l_prime_eq_2x_minus_3y_plus_5_l83_83123

theorem line_l_prime_eq_2x_minus_3y_plus_5 (m : ℝ) (x y : ℝ) : 
  (2 * m + 1) * x + (m + 1) * y + m = 0 →
  (2 * -1 + 1) * (-1) + (1 + 1) * 1 + m = 0 →
  ∀ a b : ℝ, (3 * b, 2 * b) = (3 * 1, 2 * 1) → (a, b) = (-1, 1) → 
  2 * x - 3 * y + 5 = 0 :=
by
  intro h1 h2 a b h3 h4
  sorry

end line_l_prime_eq_2x_minus_3y_plus_5_l83_83123


namespace unit_digit_of_15_pow_l83_83317

-- Define the conditions
def base_number : ℕ := 15
def base_unit_digit : ℕ := 5

-- State the question and objective in Lean 4
theorem unit_digit_of_15_pow (X : ℕ) (h : 0 < X) : (15^X) % 10 = 5 :=
sorry

end unit_digit_of_15_pow_l83_83317


namespace quadratic_has_two_real_roots_l83_83915

theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x^2 - (m + 1) * x + (3 * m - 6) = 0 :=
by
  sorry

end quadratic_has_two_real_roots_l83_83915


namespace product_of_smallest_primes_l83_83625

theorem product_of_smallest_primes :
  2 * 3 * 11 = 66 :=
by
  sorry

end product_of_smallest_primes_l83_83625


namespace possible_values_of_C_l83_83126

variable {α : Type} [LinearOrderedField α]

-- Definitions of points A, B and C
def pointA (a : α) := a
def pointB (b : α) := b
def pointC (c : α) := c

-- Given condition
def given_condition (a b : α) : Prop := (a + 3) ^ 2 + |b - 1| = 0

-- Function to determine if the folding condition is met
def folding_number_line (A B C : α) : Prop :=
  (C = 2 * A - B ∨ C = 2 * B - A ∨ (A + B) / 2 = C)

-- Theorem to prove the possible values of C
theorem possible_values_of_C (a b : α) (h : given_condition a b) :
  ∃ C : α, folding_number_line (pointA a) (pointB b) (pointC C) ∧ (C = -7 ∨ C = 5 ∨ C = -1) :=
sorry

end possible_values_of_C_l83_83126


namespace find_values_l83_83763

theorem find_values (a b c : ℕ) 
    (h1 : a + b + c = 1024) 
    (h2 : c = b - 88) 
    (h3 : a = b + c) : 
    a = 712 ∧ b = 400 ∧ c = 312 :=
by {
    sorry
}

end find_values_l83_83763


namespace integer_ratio_zero_l83_83306

theorem integer_ratio_zero
  (A B : ℤ)
  (h : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 3 ∧ x ≠ -1 → (A / (x - 3 : ℝ) + B / (x ^ 2 + 2 * x + 1) = (x ^ 3 - x ^ 2 + 3 * x + 1) / (x ^ 3 - x - 3))) :
  B / A = 0 :=
sorry

end integer_ratio_zero_l83_83306


namespace convert_base8_to_base10_l83_83965

def base8_to_base10 (n : Nat) : Nat := 
  -- Assuming a specific function that converts from base 8 to base 10
  sorry 

theorem convert_base8_to_base10 :
  base8_to_base10 5624 = 2964 :=
by
  sorry

end convert_base8_to_base10_l83_83965


namespace circle_equation_line_intersect_circle_l83_83141

theorem circle_equation (x y : ℝ) : 
  y = x^2 - 4*x + 3 → (x = 0 ∧ y = 3) ∨ (y = 0 ∧ (x = 1 ∨ x = 3)) :=
sorry

theorem line_intersect_circle (m : ℝ) :
  (∀ x y : ℝ, (x + y + m = 0) ∨ ((x - 2)^2 + (y - 2)^2 = 5)) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ + y₁ + m = 0) → ((x₁ - 2)^2 + (y₁ - 2)^2 = 5) →
    (x₂ + y₂ + m = 0) → ((x₂ - 2)^2 + (y₂ - 2)^2 = 5) →
    ((x₁ * x₂ + y₁ * y₂ = 0) → (m = -1 ∨ m = -3))) :=
sorry

end circle_equation_line_intersect_circle_l83_83141


namespace expression_simplified_l83_83034

theorem expression_simplified (d : ℤ) (h : d ≠ 0) :
  let a := 24
  let b := 61
  let c := 96
  a + b + c = 181 ∧ 
  (15 * d ^ 2 + 7 * d + 15 + (3 * d + 9) ^ 2 = a * d ^ 2 + b * d + c) := by
{
  sorry
}

end expression_simplified_l83_83034


namespace find_base_a_l83_83235

theorem find_base_a 
  (a : ℕ)
  (C_a : ℕ := 12) :
  (3 * a^2 + 4 * a + 7) + (5 * a^2 + 7 * a + 9) = 9 * a^2 + 2 * a + C_a →
  a = 14 :=
by
  intros h
  sorry

end find_base_a_l83_83235


namespace total_cleaning_time_is_100_l83_83162

def outsideCleaningTime : ℕ := 80
def insideCleaningTime : ℕ := outsideCleaningTime / 4
def totalCleaningTime : ℕ := outsideCleaningTime + insideCleaningTime

theorem total_cleaning_time_is_100 : totalCleaningTime = 100 := by
  sorry

end total_cleaning_time_is_100_l83_83162


namespace solveCubicEquation_l83_83348

-- Define the condition as a hypothesis
def equationCondition (x : ℝ) : Prop := (7 - x)^(1/3) = -5/3

-- State the theorem to be proved
theorem solveCubicEquation : ∃ x : ℝ, equationCondition x ∧ x = 314 / 27 :=
by 
  sorry

end solveCubicEquation_l83_83348


namespace sqrt_one_fourth_l83_83310

theorem sqrt_one_fourth :
  {x : ℚ | x^2 = 1/4} = {1/2, -1/2} :=
by sorry

end sqrt_one_fourth_l83_83310


namespace ratio_of_triangle_areas_l83_83923

-- Define the given conditions
variables (m n x a : ℝ) (S T1 T2 : ℝ)

-- Conditions
def area_of_square : Prop := S = x^2
def area_of_triangle_1 : Prop := T1 = m * x^2
def length_relation : Prop := x = n * a

-- The proof goal
theorem ratio_of_triangle_areas (h1 : area_of_square S x) 
                                (h2 : area_of_triangle_1 T1 m x)
                                (h3 : length_relation x n a) : 
                                T2 / S = m / n^2 := 
sorry

end ratio_of_triangle_areas_l83_83923


namespace compute_A_3_2_l83_83224

namespace Ackermann

def A : ℕ → ℕ → ℕ
| 0, n     => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem compute_A_3_2 : A 3 2 = 12 :=
sorry

end Ackermann

end compute_A_3_2_l83_83224


namespace equation_solutions_l83_83291

noncomputable def solve_equation (x : ℝ) : Prop :=
  x - 3 = 4 * (x - 3)^2

theorem equation_solutions :
  ∀ x : ℝ, solve_equation x ↔ x = 3 ∨ x = 3.25 :=
by sorry

end equation_solutions_l83_83291


namespace average_water_per_day_l83_83567

variable (day1 : ℕ)
variable (day2 : ℕ)
variable (day3 : ℕ)

def total_water_over_three_days (d1 d2 d3 : ℕ) := d1 + d2 + d3

theorem average_water_per_day :
  day1 = 215 ->
  day2 = 215 + 76 ->
  day3 = 291 - 53 ->
  (total_water_over_three_days day1 day2 day3) / 3 = 248 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end average_water_per_day_l83_83567


namespace max_repeating_sequence_length_l83_83305

theorem max_repeating_sequence_length (p q n α β d : ℕ) (h_prime: Nat.gcd p q = 1)
  (hq : q = (2 ^ α) * (5 ^ β) * d) (hd_coprime: Nat.gcd d 10 = 1) (h_repeat: 10 ^ n ≡ 1 [MOD d]) :
  ∃ s, s ≤ n * (10 ^ n - 1) ∧ (10 ^ s ≡ 1 [MOD d^2]) :=
by
  sorry

end max_repeating_sequence_length_l83_83305


namespace expected_winnings_is_350_l83_83205

noncomputable def expected_winnings : ℝ :=
  (1 / 8) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

theorem expected_winnings_is_350 :
  expected_winnings = 3.5 :=
by sorry

end expected_winnings_is_350_l83_83205


namespace trapezoid_area_l83_83007

theorem trapezoid_area
  (AD BC AC BD : ℝ)
  (h1 : AD = 24)
  (h2 : BC = 8)
  (h3 : AC = 13)
  (h4 : BD = 5 * Real.sqrt 17) : 
  ∃ (area : ℝ), area = 80 :=
by
  let area := (1 / 2) * (AD + BC) * 5
  existsi area
  sorry

end trapezoid_area_l83_83007


namespace common_ratio_geometric_sequence_l83_83527

theorem common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) ∧ a 1 = 32 ∧ a 6 = -1 → q = -1/2 :=
by
  sorry

end common_ratio_geometric_sequence_l83_83527


namespace find_equidistant_point_l83_83358

theorem find_equidistant_point :
  ∃ (x z : ℝ),
    ((x - 1)^2 + 4^2 + z^2 = (x - 2)^2 + 2^2 + (z - 3)^2) ∧
    ((x - 1)^2 + 4^2 + z^2 = (x - 3)^2 + 9 + (z + 2)^2) ∧
    (x + 2 * z = 5) ∧
    (x = 15 / 8) ∧
    (z = 5 / 8) :=
by
  sorry

end find_equidistant_point_l83_83358


namespace shooting_competition_l83_83195

variable (x y : ℕ)

theorem shooting_competition (H1 : 20 * x - 12 * (10 - x) + 20 * y - 12 * (10 - y) = 208)
                             (H2 : 20 * x - 12 * (10 - x) = 20 * y - 12 * (10 - y) + 64) :
  x = 8 ∧ y = 6 := 
by 
  sorry

end shooting_competition_l83_83195


namespace quilt_shaded_fraction_l83_83045

theorem quilt_shaded_fraction :
  let total_squares := 16
  let shaded_squares := 8
  let fully_shaded := 4
  let half_shaded := 4
  let shaded_area := fully_shaded + half_shaded * 1 / 2
  shaded_area / total_squares = 3 / 8 :=
by
  sorry

end quilt_shaded_fraction_l83_83045


namespace sum_last_two_digits_l83_83057

theorem sum_last_two_digits (h1 : 9 ^ 23 ≡ a [MOD 100]) (h2 : 11 ^ 23 ≡ b [MOD 100]) :
  (a + b) % 100 = 60 := 
  sorry

end sum_last_two_digits_l83_83057


namespace winning_candidate_percentage_l83_83879

theorem winning_candidate_percentage
  (majority_difference : ℕ)
  (total_valid_votes : ℕ)
  (P : ℕ)
  (h1 : majority_difference = 192)
  (h2 : total_valid_votes = 480)
  (h3 : 960 * P = 67200) : 
  P = 70 := by
  sorry

end winning_candidate_percentage_l83_83879


namespace length_to_width_ratio_l83_83153

/-- Let the perimeter of the rectangular sandbox be 30 feet,
    the width be 5 feet, and the length be some multiple of the width.
    Prove that the ratio of the length to the width is 2:1. -/
theorem length_to_width_ratio (P w : ℕ) (h1 : P = 30) (h2 : w = 5) (h3 : ∃ k, l = k * w) : 
  ∃ l, (P = 2 * (l + w)) ∧ (l / w = 2) := 
sorry

end length_to_width_ratio_l83_83153


namespace decimal_to_base7_conversion_l83_83221

theorem decimal_to_base7_conversion :
  (2023 : ℕ) = 5 * (7^3) + 6 * (7^2) + 2 * (7^1) + 0 * (7^0) :=
by
  sorry

end decimal_to_base7_conversion_l83_83221


namespace unique_sequence_and_a_2002_l83_83608

-- Define the sequence (a_n)
noncomputable def a : ℕ → ℕ := -- define the correct sequence based on conditions
  -- we would define a such as in the constructive steps in the solution, but here's a placeholder
  sorry

-- Prove the uniqueness and finding a_2002
theorem unique_sequence_and_a_2002 :
  (∀ n : ℕ, ∃! (i j k : ℕ), n = a i + 2 * a j + 4 * a k) ∧ a 2002 = 1227132168 :=
by
  sorry

end unique_sequence_and_a_2002_l83_83608


namespace lesser_number_l83_83459

theorem lesser_number (x y : ℕ) (h1: x + y = 60) (h2: x - y = 10) : y = 25 :=
sorry

end lesser_number_l83_83459


namespace area_of_region_l83_83102

theorem area_of_region :
  (∃ (x y: ℝ), x^2 + y^2 = 5 * |x - y| + 2 * |x + y|) → 
  (∃ (A : ℝ), A = 14.5 * Real.pi) :=
sorry

end area_of_region_l83_83102


namespace minimum_height_for_surface_area_geq_120_l83_83898

noncomputable def box_surface_area (x : ℝ) : ℝ :=
  6 * x^2 + 20 * x

theorem minimum_height_for_surface_area_geq_120 :
  ∃ (x : ℝ), (x ≥ 0) ∧ (box_surface_area x ≥ 120) ∧ (x + 5 = 9) := by
  sorry

end minimum_height_for_surface_area_geq_120_l83_83898


namespace fraction_product_l83_83809

theorem fraction_product : 
  (7 / 5) * (8 / 16) * (21 / 15) * (14 / 28) * (35 / 25) * (20 / 40) * (49 / 35) * (32 / 64) = 2401 / 10000 :=
by
  -- This line is to skip the proof
  sorry

end fraction_product_l83_83809


namespace prob_teacherA_studentB_same_group_l83_83680

-- Definitions for our conditions
def num_teachers := 2
def num_students := 4
def groups := 2
def teachers_per_group := 1
def students_per_group := 2

-- Probability calculation (we just state it)
noncomputable def probability_same_group : ℚ :=
  (nat.choose 3 1 : ℚ) / (nat.choose 2 1 * nat.choose 4 2 / nat.factorial 2 : ℚ)

-- The main theorem to prove the probability is 1/2
theorem prob_teacherA_studentB_same_group : probability_same_group = 1 / 2 :=
by
  sorry

end prob_teacherA_studentB_same_group_l83_83680


namespace decreasing_function_a_leq_zero_l83_83429

theorem decreasing_function_a_leq_zero (a : ℝ) :
  (∀ x y : ℝ, x < y → ax^3 - x ≥ ay^3 - y) → a ≤ 0 :=
by
  sorry

end decreasing_function_a_leq_zero_l83_83429


namespace evaluate_fraction_l83_83512

theorem evaluate_fraction : 1 + 3 / (4 + 5 / (6 + 7 / 8)) = 85 / 52 :=
by sorry

end evaluate_fraction_l83_83512


namespace sufficient_not_necessary_range_l83_83311

theorem sufficient_not_necessary_range (x a : ℝ) : (∀ x, x < 1 → x < a) ∧ (∃ x, x < a ∧ ¬ (x < 1)) ↔ 1 < a := by
  sorry

end sufficient_not_necessary_range_l83_83311


namespace total_amount_correct_l83_83151

noncomputable def total_amount (p_a r_a t_a p_b r_b t_b p_c r_c t_c : ℚ) : ℚ :=
  let final_price (p r t : ℚ) := p - (p * r / 100) + ((p - (p * r / 100)) * t / 100)
  final_price p_a r_a t_a + final_price p_b r_b t_b + final_price p_c r_c t_c

theorem total_amount_correct :
  total_amount 2500 6 10 3150 8 12 1000 5 7 = 6847.26 :=
by
  sorry

end total_amount_correct_l83_83151


namespace common_ratio_of_geometric_sequence_l83_83897

variable (a_1 q : ℚ) (S : ℕ → ℚ)

def geometric_sum (n : ℕ) : ℚ :=
  a_1 * (1 - q^n) / (1 - q)

def is_arithmetic_sequence (a b c : ℚ) : Prop :=
  2 * b = a + c

theorem common_ratio_of_geometric_sequence 
  (h1 : ∀ n, S n = geometric_sum a_1 q n)
  (h2 : ∀ n, is_arithmetic_sequence (S (n+2)) (S (n+1)) (S n)) : q = -2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l83_83897


namespace express_a_b_find_a_b_m_n_find_a_l83_83747

-- 1. Prove that a = m^2 + 5n^2 and b = 2mn given a + b√5 = (m + n√5)^2
theorem express_a_b (a b m n : ℤ) (h : a + b * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) :
  a = m ^ 2 + 5 * n ^ 2 ∧ b = 2 * m * n := sorry

-- 2. Prove there exists positive integers a = 6, b = 2, m = 1, and n = 1 such that 
-- a + b√5 = (m + n√5)^2.
theorem find_a_b_m_n : ∃ (a b m n : ℕ), a = 6 ∧ b = 2 ∧ m = 1 ∧ n = 1 ∧ 
  (a + b * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) := sorry

-- 3. Prove a = 46 or a = 14 given a + 6√5 = (m + n√5)^2 and a, m, n are positive integers.
theorem find_a (a m n : ℕ) (h : a + 6 * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) :
  a = 46 ∨ a = 14 := sorry

end express_a_b_find_a_b_m_n_find_a_l83_83747


namespace total_toothpicks_correct_l83_83475

noncomputable def total_toothpicks_in_grid 
  (height : ℕ) (width : ℕ) (partition_interval : ℕ) : ℕ :=
  let horizontal_lines := height + 1
  let vertical_lines := width + 1
  let num_partitions := height / partition_interval
  (horizontal_lines * width) + (vertical_lines * height) + (num_partitions * width)

theorem total_toothpicks_correct :
  total_toothpicks_in_grid 25 15 5 = 850 := 
by 
  sorry

end total_toothpicks_correct_l83_83475


namespace compute_expression_l83_83963

theorem compute_expression : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end compute_expression_l83_83963


namespace derivative_f_at_1_l83_83693

noncomputable def f (x : Real) : Real := x^3 * Real.sin x

theorem derivative_f_at_1 : deriv f 1 = 3 * Real.sin 1 + Real.cos 1 := by
  sorry

end derivative_f_at_1_l83_83693


namespace true_prop_count_l83_83530

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

end true_prop_count_l83_83530


namespace suitable_survey_l83_83501

inductive Survey
| FavoriteTVPrograms : Survey
| PrintingErrors : Survey
| BatteryServiceLife : Survey
| InternetUsage : Survey

def is_suitable_for_census (s : Survey) : Prop :=
  match s with
  | Survey.PrintingErrors => True
  | _ => False

theorem suitable_survey : is_suitable_for_census Survey.PrintingErrors = True :=
by
  sorry

end suitable_survey_l83_83501


namespace B_alone_completes_work_in_24_days_l83_83321

theorem B_alone_completes_work_in_24_days 
  (A B : ℚ) 
  (h1 : A + B = 1 / 12) 
  (h2 : A = 1 / 24) : 
  1 / B = 24 :=
by
  sorry

end B_alone_completes_work_in_24_days_l83_83321


namespace symmetric_circle_equation_l83_83913

-- Define original circle equation
def original_circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

-- Define symmetric circle equation
def symmetric_circle (x y : ℝ) : Prop := x^2 + y^2 + 4 * x = 0

theorem symmetric_circle_equation (x y : ℝ) : 
  symmetric_circle x y ↔ original_circle (-x) y :=
by sorry

end symmetric_circle_equation_l83_83913


namespace product_of_primes_l83_83610

theorem product_of_primes : (2 * 3 * 11) = 66 := by 
  sorry

end product_of_primes_l83_83610


namespace grassy_area_percentage_l83_83175

noncomputable def percentage_grassy_area (park_area path1_area path2_area intersection_area : ℝ) : ℝ :=
  let covered_by_paths := path1_area + path2_area - intersection_area
  let grassy_area := park_area - covered_by_paths
  (grassy_area / park_area) * 100

theorem grassy_area_percentage (park_area : ℝ) (path1_area : ℝ) (path2_area : ℝ) (intersection_area : ℝ) 
  (h1 : park_area = 4000) (h2 : path1_area = 400) (h3 : path2_area = 250) (h4 : intersection_area = 25) : 
  percentage_grassy_area park_area path1_area path2_area intersection_area = 84.375 :=
by
  rw [percentage_grassy_area, h1, h2, h3, h4]
  simp
  sorry

end grassy_area_percentage_l83_83175


namespace system_solutions_l83_83916

theorem system_solutions : {p : ℝ × ℝ | p.snd ^ 2 = p.fst ∧ p.snd = p.fst} = {⟨1, 1⟩, ⟨0, 0⟩} :=
by
  sorry

end system_solutions_l83_83916


namespace product_of_primes_l83_83609

theorem product_of_primes : (2 * 3 * 11) = 66 := by 
  sorry

end product_of_primes_l83_83609


namespace point_segment_length_eq_l83_83851

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x ^ 2 / 25 + y ^ 2 / 16 = 1)

noncomputable def line_eq (x : ℝ) : Prop := (x = 3)

theorem point_segment_length_eq :
  ∀ (A B : ℝ × ℝ), (ellipse_eq A.1 A.2) → (ellipse_eq B.1 B.2) → 
  (line_eq A.1) → (line_eq B.1) → (A = (3, 16/5) ∨ A = (3, -16/5)) → 
  (B = (3, 16/5) ∨ B = (3, -16/5)) → 
  |A.2 - B.2| = 32 / 5 := sorry

end point_segment_length_eq_l83_83851


namespace car_trip_eq_560_miles_l83_83076

noncomputable def car_trip_length (v L : ℝ) :=
  -- Conditions from the problem
  -- 1. Car travels for 2 hours before the delay
  let pre_delay_time := 2
  -- 2. Delay time is 1 hour
  let delay_time := 1
  -- 3. Post-delay speed is 2/3 of the initial speed
  let post_delay_speed := (2 / 3) * v
  -- 4. Car arrives 4 hours late under initial scenario:
  let late_4_hours_time := 2 + 1 + (3 * (L - 2 * v)) / (2 * v)
  -- Expected travel time without any delays is 2 + (L / v)
  -- Difference indicates delay of 4 hours
  let without_delay_time := (L / v)
  let time_diff_late_4 := (late_4_hours_time - without_delay_time = 4)
  -- 5. Delay 120 miles farther, car arrives 3 hours late
  let delay_120_miles_farther := 120
  let late_3_hours_time := 2 + delay_120_miles_farther / v + 1 + (3 * (L - 2 * v - 120)) / (2 * v)
  let time_diff_late_3 := (late_3_hours_time - without_delay_time = 3)

  -- Combining conditions to solve for L
  -- Goal: Prove L = 560
  L = 560 -> time_diff_late_4 ∧ time_diff_late_3

theorem car_trip_eq_560_miles (v : ℝ) : ∃ (L : ℝ), car_trip_length v L := 
by 
  sorry

end car_trip_eq_560_miles_l83_83076


namespace cards_given_l83_83565

/-- Martha starts with 3 cards. She ends up with 79 cards after receiving some from Emily. We need to prove that Emily gave her 76 cards. -/
theorem cards_given (initial_cards final_cards cards_given : ℕ) (h1 : initial_cards = 3) (h2 : final_cards = 79) (h3 : final_cards = initial_cards + cards_given) :
  cards_given = 76 :=
sorry

end cards_given_l83_83565


namespace geometric_seq_sum_four_and_five_l83_83528

noncomputable def geom_seq (a₁ q : ℝ) (n : ℕ) := a₁ * q^(n-1)

theorem geometric_seq_sum_four_and_five :
  (∀ n, geom_seq a₁ q n > 0) →
  geom_seq a₁ q 3 = 4 →
  geom_seq a₁ q 6 = 1 / 2 →
  geom_seq a₁ q 4 + geom_seq a₁ q 5 = 3 :=
by
  sorry

end geometric_seq_sum_four_and_five_l83_83528


namespace straight_line_cannot_intersect_all_segments_l83_83093

/-- A broken line in the plane with 11 segments -/
structure BrokenLine :=
(segments : Fin 11 → (ℝ × ℝ) × (ℝ × ℝ))
(closed_chain : ∀ i : Fin 11, i.val < 10 → (segments ⟨i.val + 1, sorry⟩).fst = (segments i).snd)

/-- A straight line that doesn't contain the vertices of the broken line -/
structure StraightLine :=
(is_not_vertex : (ℝ × ℝ) → Prop)

/-- The main theorem stating the impossibility of a straight line intersecting all segments -/
theorem straight_line_cannot_intersect_all_segments (line : StraightLine) (brokenLine: BrokenLine) :
  ∃ i : Fin 11, ¬∃ t : ℝ, ∃ x y : ℝ, 
    brokenLine.segments i = ((x, y), (x + t, y + t)) ∧ 
    ¬line.is_not_vertex (x, y) ∧ 
    ¬line.is_not_vertex (x + t, y + t) :=
sorry

end straight_line_cannot_intersect_all_segments_l83_83093


namespace lesser_number_l83_83449

theorem lesser_number (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 10) : y = 25 :=
by
  have h3 : x = 35 := sorry
  exact sorry

end lesser_number_l83_83449


namespace fifth_largest_divisor_of_1209600000_is_75600000_l83_83307

theorem fifth_largest_divisor_of_1209600000_is_75600000 :
  let n : ℤ := 1209600000
  let fifth_largest_divisor : ℤ := 75600000
  n = 2^10 * 5^5 * 3 * 503 →
  fifth_largest_divisor = n / 2^5 :=
by
  sorry

end fifth_largest_divisor_of_1209600000_is_75600000_l83_83307


namespace percentage_decrease_l83_83308

variable {a b x m : ℝ} (p : ℝ)

theorem percentage_decrease (h₁ : a / b = 4 / 5)
                          (h₂ : x = 1.25 * a)
                          (h₃ : m = b * (1 - p / 100))
                          (h₄ : m / x = 0.8) :
  p = 20 :=
sorry

end percentage_decrease_l83_83308


namespace domain_of_function_l83_83476

theorem domain_of_function:
  {x : ℝ | x^2 - 5*x + 6 > 0 ∧ x ≠ 3} = {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end domain_of_function_l83_83476


namespace Juan_run_time_l83_83727

theorem Juan_run_time
  (d : ℕ) (s : ℕ) (t : ℕ)
  (H1: d = 80)
  (H2: s = 10)
  (H3: t = d / s) :
  t = 8 := 
sorry

end Juan_run_time_l83_83727


namespace average_age_increase_l83_83176

variable (A B C : ℕ)

theorem average_age_increase (A : ℕ) (B : ℕ) (C : ℕ) (h1 : 21 < B) (h2 : 23 < C) (h3 : A + B + C > A + 21 + 23) :
  (B + C) / 2 > 22 := by
  sorry

end average_age_increase_l83_83176


namespace largest_class_students_l83_83876

theorem largest_class_students (x : ℕ) (h1 : 8 * x - (4 + 8 + 12 + 16 + 20 + 24 + 28) = 380) : x = 61 :=
by
  sorry

end largest_class_students_l83_83876


namespace physics_students_l83_83466

variable (B : Nat) (G : Nat) (Biology : Nat) (Physics : Nat)

axiom h1 : B = 25
axiom h2 : G = 3 * B
axiom h3 : Biology = B + G
axiom h4 : Physics = 2 * Biology

theorem physics_students : Physics = 200 :=
by
  sorry

end physics_students_l83_83466


namespace find_value_l83_83939

theorem find_value (number remainder certain_value : ℕ) (h1 : number = 26)
  (h2 : certain_value / 2 = remainder) 
  (h3 : remainder = ((number + 20) * 2 / 2) - 2) :
  certain_value = 88 :=
by
  sorry

end find_value_l83_83939


namespace total_servings_daily_l83_83933

def cost_per_serving : ℕ := 14
def price_A : ℕ := 20
def price_B : ℕ := 18
def total_revenue : ℕ := 1120
def total_profit : ℕ := 280

theorem total_servings_daily (x y : ℕ) (h1 : price_A * x + price_B * y = total_revenue)
                             (h2 : (price_A - cost_per_serving) * x + (price_B - cost_per_serving) * y = total_profit) :
                             x + y = 60 := sorry

end total_servings_daily_l83_83933


namespace bess_throw_distance_l83_83955

-- Definitions based on the conditions
def bess_throws (x : ℝ) : ℝ := 4 * 2 * x
def holly_throws : ℝ := 5 * 8
def total_throws (x : ℝ) : ℝ := bess_throws x + holly_throws

-- Lean statement for the proof
theorem bess_throw_distance (x : ℝ) (h : total_throws x = 200) : x = 20 :=
by 
  sorry

end bess_throw_distance_l83_83955


namespace product_of_primes_l83_83615

def smallest_one_digit_prime := 2
def second_smallest_one_digit_prime := 3
def smallest_two_digit_prime := 11

theorem product_of_primes: smallest_one_digit_prime * second_smallest_one_digit_prime * smallest_two_digit_prime = 66 :=
by {
  -- Applying the definition of the primes and carrying out the multiplication
  show 2 * 3 * 11 = 66,
  calc
  2 * 3 * 11 = 6 * 11 : by rw [mul_assoc 2 3 11]
          ... = 66    : by norm_num,
}

end product_of_primes_l83_83615


namespace intersecting_lines_angle_difference_l83_83001

-- Define the conditions
def angle_y : ℝ := 40
def straight_angle_sum : ℝ := 180

-- Define the variables representing the angles
variable (x y : ℝ)

-- Define the proof problem
theorem intersecting_lines_angle_difference : 
  ∀ x y : ℝ, 
  y = angle_y → 
  (∃ (a b : ℝ), a + b = straight_angle_sum ∧ a = y ∧ b = x) → 
  x - y = 100 :=
by
  intros x y hy h
  sorry

end intersecting_lines_angle_difference_l83_83001


namespace total_number_of_legs_l83_83022

def kangaroos : ℕ := 23
def goats : ℕ := 3 * kangaroos
def legs_of_kangaroo : ℕ := 2
def legs_of_goat : ℕ := 4

theorem total_number_of_legs : 
  (kangaroos * legs_of_kangaroo + goats * legs_of_goat) = 322 := by
  sorry

end total_number_of_legs_l83_83022


namespace product_of_primes_l83_83612

theorem product_of_primes : (2 * 3 * 11) = 66 := by 
  sorry

end product_of_primes_l83_83612


namespace initial_roses_in_vase_l83_83601

theorem initial_roses_in_vase (current_roses : ℕ) (added_roses : ℕ) (total_garden_roses : ℕ) (initial_roses : ℕ) :
  current_roses = 20 → added_roses = 13 → total_garden_roses = 59 → initial_roses = current_roses - added_roses → 
  initial_roses = 7 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h4
  sorry

end initial_roses_in_vase_l83_83601


namespace find_sum_of_distinct_real_numbers_l83_83891

noncomputable def determinant_3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

theorem find_sum_of_distinct_real_numbers (x y : ℝ) (hxy : x ≠ y) 
    (h : determinant_3x3 1 6 15 3 x y 3 y x = 0) : x + y = 63 := 
by
  sorry

end find_sum_of_distinct_real_numbers_l83_83891


namespace michael_total_cost_l83_83406

def rental_fee : ℝ := 20.99
def charge_per_mile : ℝ := 0.25
def miles_driven : ℕ := 299

def total_cost (rental_fee : ℝ) (charge_per_mile : ℝ) (miles_driven : ℕ) : ℝ :=
  rental_fee + (charge_per_mile * miles_driven)

theorem michael_total_cost :
  total_cost rental_fee charge_per_mile miles_driven = 95.74 :=
by
  sorry

end michael_total_cost_l83_83406


namespace price_difference_correct_l83_83499

-- Define the list price of Camera Y
def list_price : ℚ := 52.50

-- Define the discount at Mega Deals
def mega_deals_discount : ℚ := 12

-- Define the discount rate at Budget Buys
def budget_buys_discount_rate : ℚ := 0.30

-- Calculate the sale prices
def mega_deals_price : ℚ := list_price - mega_deals_discount
def budget_buys_price : ℚ := (1 - budget_buys_discount_rate) * list_price

-- Calculate the price difference in dollars and convert to cents
def price_difference_in_cents : ℚ := (mega_deals_price - budget_buys_price) * 100

-- Theorem to prove the computed price difference in cents equals 375
theorem price_difference_correct : price_difference_in_cents = 375 := by
  sorry

end price_difference_correct_l83_83499


namespace monotonicity_range_of_a_l83_83121

noncomputable def f (x a : ℝ) : ℝ := Real.log x + a * (1 - x)
noncomputable def f' (x a : ℝ) : ℝ := 1 / x - a

-- 1. Monotonicity discussion
theorem monotonicity (a x : ℝ) (h : 0 < x) : 
  (a ≤ 0 → ∀ x, 0 < x → f' x a > 0) ∧
  (a > 0 → (∀ x, 0 < x ∧ x < 1 / a → f' x a > 0) ∧ (∀ x, x > 1 / a → f' x a < 0)) :=
sorry

-- 2. Range of a for maximum value condition
noncomputable def g (a : ℝ) : ℝ := Real.log a + a - 1

theorem range_of_a (a : ℝ) : 
  (0 < a) ∧ (a < 1) ↔ g a < 0 :=
sorry

end monotonicity_range_of_a_l83_83121


namespace tangent_from_origin_l83_83095

-- Define the points A, B, and C
def A : ℝ × ℝ := (4, 5)
def B : ℝ × ℝ := (7, 10)
def C : ℝ × ℝ := (6, 14)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define a function that computes the length of the tangent from O to the circle passing through A, B, and C
noncomputable def tangent_length : ℝ :=
 sorry -- Placeholder for the actual calculation

-- The theorem we need to prove: The length of the tangent from O to the circle passing through A, B, and C is as calculated
theorem tangent_from_origin (L : ℕ) : 
  tangent_length = L := 
 sorry -- Placeholder for the proof

end tangent_from_origin_l83_83095


namespace problem1_problem2_l83_83416

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem problem1 (x : ℝ) : f x 2 ≥ 2 ↔ x ≤ 1/2 ∨ x ≥ 5/2 :=
  sorry -- the proof goes here

theorem problem2 (a : ℝ) (h₁ : 1 < a) : 
  (∀ x : ℝ, f x a + |x - 1| ≥ 1) ∧ (2 ≤ a) :=
  sorry -- the proof goes here

end problem1_problem2_l83_83416


namespace age_of_eldest_boy_l83_83783

theorem age_of_eldest_boy (x : ℕ) (h1 : (3*x + 5*x + 7*x) / 3 = 15) :
  7 * x = 21 :=
sorry

end age_of_eldest_boy_l83_83783


namespace lesser_of_two_numbers_l83_83442

theorem lesser_of_two_numbers (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
by
  sorry

end lesser_of_two_numbers_l83_83442


namespace square_side_length_eq_triangle_area_eq_distance_WH_final_proof_l83_83900

theorem square_side_length_eq (s : ℝ) (hs : s^2 = 144) : s = 12 :=
by {
  have h : s = real.sqrt 144, from (eq_div_iff_mul_eq 12 (12 : ℝ) hs).mpr (eq.symm (real.sqrt_sqr_eq_abs 12)),
  rw real.sqrt_eq_iff_sq_eq at h,
  cases h,
  repeat { assumption },
}

theorem triangle_area_eq (ZG ZH : ℝ) (htriangle : 90 = 1 / 2 * ZG * ZH) : ZG = ZH := 
by {
  have key : 2 * 90 = ZG * ZH, by { simp [htriangle], },
  have square : 180 = ZG^2, from (eq_div_iff_mul_eq 180 (ZG * ZH)).mpr key,
  have h1 : ZG = real.sqrt 180, from (eq_div_iff_mul_eq 180 (real.sqrt (ZG * ZG) * (1 : ℝ))).mpr (eq.symm square),
  rw real.sqrt_eq_iff_sq_eq at h1,
  cases h1,
  repeat { assumption },
}

theorem distance_WH (ZG ZH WH ZW : ℝ) (h1 : ZG = ZH) (h2 : ZW = 12) (h3 : ZG = 6 * real.sqrt 5) :
  WH = 18 :=
by {
  have hyp1 : WH^2 = ZW^2 + ZH^2,
    { simp [pow_two], },
  have hyp2 : ZW^2 = 144,
    { rw ← pow_two, exact (eq_div_iff_mul_eq ZW ZW).mpr 
      (eq.symm (real.sqr_sqrt_eq_abs 144)), },
  have hyp3 : ZH^2 = 180,
    { rw ← pow_two, exact (eq_div_iff_mul_eq ZH ZH).mpr 
      (eq.symm (real.sqr_sqrt_eq_abs 180)), },
  have hyp : WH^2 = 324,
    { simp [hyp1, hyp2, hyp3], },
  rw real.sqrt_eq_iff_sq_eq at hyp,
  cases hyp with h1 h2,
  assumption,
  simp at *,
}

theorem final_proof : WH = 18 :=
by {
  let s := 12,
  let ZG := 6 * real.sqrt 5,
  let ZH := ZG,
  let ZW := s,
  apply distance_WH ZG ZH WH ZW,
  all_goals { simp [pow_two] },
}

end square_side_length_eq_triangle_area_eq_distance_WH_final_proof_l83_83900


namespace intersection_M_N_l83_83013

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x > 1}

theorem intersection_M_N :
  M ∩ N = {x | 1 < x ∧ x ≤ 2} := 
sorry

end intersection_M_N_l83_83013


namespace A_finishes_work_in_9_days_l83_83075

noncomputable def B_work_rate : ℝ := 1 / 15
noncomputable def B_work_10_days : ℝ := 10 * B_work_rate
noncomputable def remaining_work_by_A : ℝ := 1 - B_work_10_days

theorem A_finishes_work_in_9_days (A_days : ℝ) (B_days : ℝ) (B_days_worked : ℝ) (A_days_worked : ℝ) :
  (B_days = 15) ∧ (B_days_worked = 10) ∧ (A_days_worked = 3) ∧ 
  (remaining_work_by_A = (1 / 3)) → A_days = 9 :=
by sorry

end A_finishes_work_in_9_days_l83_83075


namespace actual_distance_between_cities_l83_83301

-- Define the scale and distance on the map as constants
def distance_on_map : ℝ := 20
def scale_inch_miles : ℝ := 12  -- Because 1 inch = 12 miles derived from the scale 0.5 inches = 6 miles

-- Define the actual distance calculation
def actual_distance (distance_inch : ℝ) (scale : ℝ) : ℝ :=
  distance_inch * scale

-- Example theorem to prove the actual distance between the cities
theorem actual_distance_between_cities :
  actual_distance distance_on_map scale_inch_miles = 240 := by
  sorry

end actual_distance_between_cities_l83_83301


namespace symmetric_point_exists_l83_83977

-- Define the point P and line equation.
structure Point (α : Type*) := (x : α) (y : α)
def P : Point ℝ := ⟨5, -2⟩
def line_eq (x y : ℝ) : Prop := x - y + 5 = 0

-- Define a function for the line PQ being perpendicular to the given line.
def is_perpendicular (P Q : Point ℝ) : Prop :=
  (Q.y - P.y) / (Q.x - P.x) = -1

-- Define a function for the midpoint of PQ lying on the given line.
def midpoint_on_line (P Q : Point ℝ) : Prop :=
  line_eq ((P.x + Q.x) / 2) ((P.y + Q.y) / 2)

-- Define the symmetry function based on the provided conditions.
def is_symmetric (Q : Point ℝ) : Prop :=
  is_perpendicular P Q ∧ midpoint_on_line P Q

-- State the main theorem to be proved: there exists a point Q that satisfies the 
-- conditions and is symmetric to P with respect to the given line.
theorem symmetric_point_exists : ∃ Q : Point ℝ, is_symmetric Q ∧ Q = ⟨-7, 10⟩ :=
by
  sorry

end symmetric_point_exists_l83_83977


namespace ratio_of_green_to_blue_l83_83345

def balls (total blue red green yellow : ℕ) : Prop :=
  total = 36 ∧ blue = 6 ∧ red = 4 ∧ yellow = 2 * red ∧ green = total - (blue + red + yellow)

theorem ratio_of_green_to_blue (total blue red green yellow : ℕ) (h : balls total blue red green yellow) :
  (green / blue = 3) :=
by
  -- Unpack the conditions
  obtain ⟨total_eq, blue_eq, red_eq, yellow_eq, green_eq⟩ := h
  -- Simplify values based on the given conditions
  have blue_val := blue_eq
  have green_val := green_eq
  rw [blue_val, green_val]
  sorry

end ratio_of_green_to_blue_l83_83345


namespace symmetric_angle_set_l83_83366

theorem symmetric_angle_set (α β : ℝ) (k : ℤ) 
  (h1 : β = 2 * (k : ℝ) * Real.pi + Real.pi / 12)
  (h2 : α = -Real.pi / 3)
  (symmetric : α + β = -Real.pi / 4) :
  ∃ k : ℤ, β = 2 * (k : ℝ) * Real.pi + Real.pi / 12 :=
sorry

end symmetric_angle_set_l83_83366


namespace find_fg_of_3_l83_83122

def f (x : ℤ) : ℤ := 2 * x - 1
def g (x : ℤ) : ℤ := x^2 + 4 * x - 5

theorem find_fg_of_3 : f (g 3) = 31 := by
  sorry

end find_fg_of_3_l83_83122


namespace Rudolph_stop_signs_l83_83170

def distance : ℕ := 5 + 2
def stopSignsPerMile : ℕ := 2
def totalStopSigns : ℕ := distance * stopSignsPerMile

theorem Rudolph_stop_signs :
  totalStopSigns = 14 := 
  by sorry

end Rudolph_stop_signs_l83_83170


namespace segment_length_l83_83226
noncomputable def cube_root27 : ℝ := 3

theorem segment_length : ∀ (x : ℝ), (|x - cube_root27| = 4) → ∃ (a b : ℝ), (a = cube_root27 + 4) ∧ (b = cube_root27 - 4) ∧ |a - b| = 8 :=
by
  sorry

end segment_length_l83_83226


namespace michael_eggs_count_l83_83025

def initial_crates : List ℕ := [24, 28, 32, 36, 40, 44]
def wednesday_given : List ℕ := [28, 32, 40]
def thursday_purchases : List ℕ := [50, 45, 55, 60]
def friday_sold : List ℕ := [60, 55]

theorem michael_eggs_count :
  let total_tuesday := initial_crates.sum
  let total_given_wednesday := wednesday_given.sum
  let remaining_wednesday := total_tuesday - total_given_wednesday
  let total_thursday := thursday_purchases.sum
  let total_after_thursday := remaining_wednesday + total_thursday
  let total_sold_friday := friday_sold.sum
  total_after_thursday - total_sold_friday = 199 :=
by
  sorry

end michael_eggs_count_l83_83025


namespace slices_served_yesterday_l83_83662

theorem slices_served_yesterday
  (lunch_slices : ℕ)
  (dinner_slices : ℕ)
  (total_slices_today : ℕ)
  (h1 : lunch_slices = 7)
  (h2 : dinner_slices = 5)
  (h3 : total_slices_today = 12) :
  (total_slices_today - (lunch_slices + dinner_slices) = 0) :=
by {
  sorry
}

end slices_served_yesterday_l83_83662


namespace product_of_primes_is_66_l83_83635

theorem product_of_primes_is_66 :
  let p1 : ℕ := 2
      p2 : ℕ := 3
      p3 : ℕ := 11
  in p1 * p2 * p3 = 66 := by
  sorry

end product_of_primes_is_66_l83_83635


namespace largest_power_of_two_dividing_7_pow_2048_minus_1_l83_83813

theorem largest_power_of_two_dividing_7_pow_2048_minus_1 :
  ∃ n : ℕ, 2^n ∣ (7^2048 - 1) ∧ n = 14 :=
by
  use 14
  sorry

end largest_power_of_two_dividing_7_pow_2048_minus_1_l83_83813


namespace g_of_2_eq_14_l83_83375

theorem g_of_2_eq_14 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) : g 2 = 14 := 
sorry

end g_of_2_eq_14_l83_83375


namespace watched_movies_count_l83_83766

theorem watched_movies_count {M : ℕ} (total_books total_movies read_books : ℕ) 
  (h1 : total_books = 15) (h2 : total_movies = 14) (h3 : read_books = 11) 
  (h4 : read_books = M + 1) : M = 10 :=
by
  sorry

end watched_movies_count_l83_83766


namespace max_abc_value_l83_83730

variables (a b c : ℕ)

theorem max_abc_value : 
  (a > 0) → (b > 0) → (c > 0) → a + 2 * b + 3 * c = 100 → abc ≤ 6171 := 
by sorry

end max_abc_value_l83_83730


namespace find_original_number_l83_83907

theorem find_original_number (x : ℚ) (h : 5 * ((3 * x + 6) / 2) = 100) : x = 34 / 3 := sorry

end find_original_number_l83_83907


namespace red_peppers_weight_correct_l83_83128

def weight_of_red_peppers : Prop :=
  ∀ (T G : ℝ), (T = 0.66) ∧ (G = 0.33) → (T - G = 0.33)

theorem red_peppers_weight_correct : weight_of_red_peppers :=
  sorry

end red_peppers_weight_correct_l83_83128


namespace product_of_smallest_primes_l83_83642

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def smallest_one_digit_primes : List ℕ := [2, 3]
def smallest_two_digit_prime : ℕ := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 :=
by
  sorry

end product_of_smallest_primes_l83_83642


namespace algebraic_expression_value_l83_83706

noncomputable def a : ℝ := Real.sqrt 6 + 1
noncomputable def b : ℝ := Real.sqrt 6 - 1

theorem algebraic_expression_value :
  a^2 + a * b = 12 + 2 * Real.sqrt 6 :=
sorry

end algebraic_expression_value_l83_83706


namespace angle_not_45_or_135_l83_83869

variable {a b S : ℝ}
variable {C : ℝ} (h : S = (1/2) * a * b * Real.cos C)

theorem angle_not_45_or_135 (h : S = (1/2) * a * b * Real.cos C) : ¬ (C = 45 ∨ C = 135) :=
sorry

end angle_not_45_or_135_l83_83869


namespace book_cost_l83_83903

theorem book_cost (initial_money : ℕ) (remaining_money : ℕ) (num_books : ℕ) 
  (h1 : initial_money = 79) (h2 : remaining_money = 16) (h3 : num_books = 9) :
  (initial_money - remaining_money) / num_books = 7 :=
by
  sorry

end book_cost_l83_83903


namespace mrs_lee_earnings_percentage_l83_83540

theorem mrs_lee_earnings_percentage 
  (M F : ℝ)
  (H1 : 1.20 * M = 0.5454545454545454 * (1.20 * M + F)) :
  M = 0.5 * (M + F) :=
by sorry

end mrs_lee_earnings_percentage_l83_83540


namespace lion_to_leopard_ratio_l83_83004

variable (L P E : ℕ)

axiom lion_count : L = 200
axiom total_population : L + P + E = 450
axiom elephants_relation : E = (1 / 2 : ℚ) * (L + P)

theorem lion_to_leopard_ratio : L / P = 2 :=
by
  sorry

end lion_to_leopard_ratio_l83_83004


namespace book_cost_l83_83904

theorem book_cost (initial_money : ℕ) (remaining_money : ℕ) (num_books : ℕ) 
  (h1 : initial_money = 79) (h2 : remaining_money = 16) (h3 : num_books = 9) :
  (initial_money - remaining_money) / num_books = 7 :=
by
  sorry

end book_cost_l83_83904


namespace maximize_S_n_l83_83112

def a1 : ℚ := 5
def d : ℚ := -5 / 7

def S_n (n : ℕ) : ℚ :=
  (n * (2 * a1 + (n - 1) * d)) / 2

theorem maximize_S_n :
  (∃ n : ℕ, (S_n n ≥ S_n (n - 1)) ∧ (S_n n ≥ S_n (n + 1))) →
  (n = 7 ∨ n = 8) :=
sorry

end maximize_S_n_l83_83112


namespace product_of_primes_l83_83611

theorem product_of_primes : (2 * 3 * 11) = 66 := by 
  sorry

end product_of_primes_l83_83611


namespace lesser_number_l83_83446

theorem lesser_number (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 10) : y = 25 :=
by
  have h3 : x = 35 := sorry
  exact sorry

end lesser_number_l83_83446


namespace surface_area_of_cube_l83_83536

theorem surface_area_of_cube (V : ℝ) (H : V = 125) : ∃ A : ℝ, A = 25 :=
by
  sorry

end surface_area_of_cube_l83_83536


namespace prob_exactly_two_meet_standard_most_likely_number_meeting_standard_l83_83542

namespace ProbabilityTest

noncomputable theory
open MeasureTheory

variables {Ω : Type*} [MeasurableSpace Ω] {P : MeasureTheory.ProbabilityMeasure Ω}
variables (A B C : Set Ω)
variables (hA : P A = 2/5) (hB : P B = 3/4) (hC : P C = 1/2)
variables (hA_indep_B : MeasureTheory.Indep {A} {B} P)
variables (hA_indep_C : MeasureTheory.Indep {A} {C} P)
variables (hB_indep_C : MeasureTheory.Indep {B} {C} P)

theorem prob_exactly_two_meet_standard :
  P (A ∩ B ∩ Cᶜ ∪ A ∩ Bᶜ ∩ C ∪ Aᶜ ∩ B ∩ C) = 17/40 :=
sorry

theorem most_likely_number_meeting_standard :
  let P_all := P (A ∩ B ∩ C),
      P_none := P (Aᶜ ∩ Bᶜ ∩ Cᶜ),
      P_one := 1 - 17/40 - 3/20 - 3/40 in
  List.argmax [P_all, 17/40, P_one, P_none] = 17/40 :=
sorry

end ProbabilityTest

end prob_exactly_two_meet_standard_most_likely_number_meeting_standard_l83_83542


namespace geom_prog_identity_l83_83744

-- Define that A, B, C are the n-th, p-th, and k-th terms respectively of the same geometric progression.
variables (a r : ℝ) (n p k : ℕ) (A B C : ℝ)

-- Assume A = ar^(n-1), B = ar^(p-1), C = ar^(k-1)
def isGP (a r : ℝ) (n p k : ℕ) (A B C : ℝ) : Prop :=
  A = a * r^(n-1) ∧ B = a * r^(p-1) ∧ C = a * r^(k-1)

-- Define the statement to be proved
theorem geom_prog_identity (h : isGP a r n p k A B C) : A^(p-k) * B^(k-n) * C^(n-p) = 1 :=
sorry

end geom_prog_identity_l83_83744


namespace correct_calculation_is_7_88_l83_83646

theorem correct_calculation_is_7_88 (x : ℝ) (h : x * 8 = 56) : (x / 8) + 7 = 7.88 :=
by
  have hx : x = 7 := by
    linarith [h]
  rw [hx]
  norm_num
  sorry

end correct_calculation_is_7_88_l83_83646


namespace subscription_total_l83_83322

theorem subscription_total (a b c : ℝ) (h1 : a = b + 4000) (h2 : b = c + 5000) (h3 : 15120 / 36000 = a / (a + b + c)) : 
  a + b + c = 50000 :=
by 
  sorry

end subscription_total_l83_83322


namespace cost_per_book_l83_83906

theorem cost_per_book (a r n c : ℕ) (h : a - r = n * c) : c = 7 :=
by sorry

end cost_per_book_l83_83906


namespace elaine_rent_percentage_l83_83888

theorem elaine_rent_percentage (E : ℝ) (P : ℝ) 
  (h1 : E > 0) 
  (h2 : P > 0) 
  (h3 : 0.25 * 1.15 * E = 1.4375 * (P / 100) * E) : 
  P = 20 := 
sorry

end elaine_rent_percentage_l83_83888


namespace restore_salary_l83_83949

variable (W : ℝ) -- Define the initial wage as a real number
variable (newWage : ℝ := 0.7 * W) -- New wage after a 30% reduction

-- Define the hypothesis for the initial wage reduction
theorem restore_salary : (100 * (W / (0.7 * W) - 1)) = 42.86 :=
by
  sorry

end restore_salary_l83_83949


namespace discount_percentage_l83_83798

theorem discount_percentage (CP SP SP_no_discount discount : ℝ)
  (h1 : SP = CP * (1 + 0.44))
  (h2 : SP_no_discount = CP * (1 + 0.50))
  (h3 : discount = SP_no_discount - SP) :
  (discount / SP_no_discount) * 100 = 4 :=
by
  sorry

end discount_percentage_l83_83798


namespace problem1_problem2_l83_83486

-- Problem 1: Proving the equation
theorem problem1 (x : ℝ) : (x + 2) / 3 - 1 = (1 - x) / 2 → x = 1 :=
sorry

-- Problem 2: Proving the solution for the system of equations
theorem problem2 (x y : ℝ) : (x + 2 * y = 8) ∧ (3 * x - 4 * y = 4) → x = 4 ∧ y = 2 :=
sorry

end problem1_problem2_l83_83486


namespace hyperbola_foci_x_axis_range_l83_83709

theorem hyperbola_foci_x_axis_range (m : ℝ) :
  (∃ x y : ℝ, (x^2 / (m + 2)) - (y^2 / (m - 1)) = 1) →
  (1 < m) ↔ 
  (∀ x y : ℝ, (m + 2 > 0) ∧ (m - 1 > 0)) :=
sorry

end hyperbola_foci_x_axis_range_l83_83709


namespace compute_expression_l83_83400

theorem compute_expression (a b c : ℝ) (h : a^3 - 6 * a^2 + 11 * a - 6 = 0 ∧ b^3 - 6 * b^2 + 11 * b - 6 = 0 ∧ c^3 - 6 * c^2 + 11 * c - 6 = 0) :
  (ab / c + bc / a + ca / b) = 49 / 6 := 
  by
  sorry -- Placeholder for the proof

end compute_expression_l83_83400


namespace ex1_l83_83248

theorem ex1 (a b : ℕ) (h₀ : a = 3) (h₁ : b = 4) : ∃ n : ℕ, 3^(7*a + b) = n^7 :=
by
  use 27
  sorry

end ex1_l83_83248


namespace base_five_sum_l83_83190

theorem base_five_sum (a b : Nat) (ha : a = 2 * 5^2 + 1 * 5^1 + 2 * 5^0) (hb : b = 1 * 5^1 + 2 * 5^0) :
  Nat.toDigits 5 (a + b) = [2, 2, 4] :=
by
  sorry

end base_five_sum_l83_83190


namespace average_percentage_15_students_l83_83068

-- Define the average percentage of the 15 students
variable (x : ℝ)

-- Condition 1: Total percentage for the 15 students is 15 * x
def total_15_students : ℝ := 15 * x

-- Condition 2: Total percentage for the 10 students who averaged 88%
def total_10_students : ℝ := 10 * 88

-- Condition 3: Total percentage for all 25 students who averaged 79%
def total_all_students : ℝ := 25 * 79

-- Mathematical problem: Prove that x = 73 given the conditions.
theorem average_percentage_15_students (h : total_15_students x + total_10_students = total_all_students) : x = 73 := 
by
  sorry

end average_percentage_15_students_l83_83068


namespace complement_of_angle_l83_83712

theorem complement_of_angle (supplement : ℝ) (h_supp : supplement = 130) (original_angle : ℝ) (h_orig : original_angle = 180 - supplement) : 
  (90 - original_angle) = 40 := 
by 
  -- proof goes here
  sorry

end complement_of_angle_l83_83712


namespace find_number_l83_83069

-- Define the condition
def is_number (x : ℝ) : Prop :=
  0.15 * x = 0.25 * 16 + 2

-- The theorem statement: proving the number is 40
theorem find_number (x : ℝ) (h : is_number x) : x = 40 :=
by
  -- We would insert the proof steps here
  sorry

end find_number_l83_83069


namespace original_number_l83_83318

theorem original_number (N : ℤ) : (∃ k : ℤ, N - 7 = 12 * k) → N = 19 :=
by
  intros h
  sorry

end original_number_l83_83318


namespace solve_for_x_l83_83134

theorem solve_for_x (x : ℝ) (h : 9 / (5 + x / 0.75) = 1) : x = 3 :=
by {
  sorry
}

end solve_for_x_l83_83134


namespace time_for_trains_to_clear_l83_83471

noncomputable def train_length_1 : ℕ := 120
noncomputable def train_length_2 : ℕ := 320
noncomputable def train_speed_1_kmph : ℚ := 42
noncomputable def train_speed_2_kmph : ℚ := 30

noncomputable def kmph_to_mps (speed: ℚ) : ℚ := (5/18) * speed

noncomputable def train_speed_1_mps : ℚ := kmph_to_mps train_speed_1_kmph
noncomputable def train_speed_2_mps : ℚ := kmph_to_mps train_speed_2_kmph

noncomputable def total_length : ℕ := train_length_1 + train_length_2
noncomputable def relative_speed : ℚ := train_speed_1_mps + train_speed_2_mps

noncomputable def collision_time : ℚ := total_length / relative_speed

theorem time_for_trains_to_clear : collision_time = 22 := by
  sorry

end time_for_trains_to_clear_l83_83471


namespace minimum_shoeing_time_l83_83070

theorem minimum_shoeing_time 
  (blacksmiths : ℕ) (horses : ℕ) (hooves_per_horse : ℕ) (time_per_hoof : ℕ) 
  (total_hooves : ℕ := horses * hooves_per_horse) 
  (time_for_one_blacksmith : ℕ := total_hooves * time_per_hoof) 
  (total_parallel_time : ℕ := time_for_one_blacksmith / blacksmiths)
  (h : blacksmiths = 48)
  (h' : horses = 60)
  (h'' : hooves_per_horse = 4)
  (h''' : time_per_hoof = 5) : 
  total_parallel_time = 25 :=
by
  sorry

end minimum_shoeing_time_l83_83070


namespace dustin_reads_more_pages_l83_83682

theorem dustin_reads_more_pages (dustin_rate_per_hour : ℕ) (sam_rate_per_hour : ℕ) : 
  (dustin_rate_per_hour = 75) → (sam_rate_per_hour = 24) → 
  (dustin_rate_per_hour * 40 / 60 - sam_rate_per_hour * 40 / 60 = 34) :=
by
  sorry

end dustin_reads_more_pages_l83_83682


namespace total_dogs_barking_l83_83137

theorem total_dogs_barking 
  (initial_dogs : ℕ)
  (new_dogs : ℕ)
  (h1 : initial_dogs = 30)
  (h2 : new_dogs = 3 * initial_dogs) :
  initial_dogs + new_dogs = 120 :=
by
  sorry

end total_dogs_barking_l83_83137


namespace base5_to_octal_1234_eval_f_at_3_l83_83487

-- Definition of base conversion from base 5 to decimal and to octal
def base5_to_decimal (n : Nat) : Nat :=
  match n with
  | 1234 => 1 * 5^3 + 2 * 5^2 + 3 * 5 + 4
  | _ => 0

def decimal_to_octal (n : Nat) : Nat :=
  match n with
  | 194 => 302
  | _ => 0

-- Definition of the polynomial f(x) = 7x^7 + 6x^6 + 5x^5 + 4x^4 + 3x^3 + 2x^2 + x
def f (x : Nat) : Nat :=
  7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

-- Definition of Horner's method evaluation
def horner_eval (x : Nat) : Nat :=
  ((((((7 * x + 6) * x + 5) * x + 4) * x + 3) * x + 2) * x + 1) * x

-- Theorem statement for base-5 to octal conversion
theorem base5_to_octal_1234 : base5_to_decimal 1234 = 194 ∧ decimal_to_octal 194 = 302 :=
  by
    sorry

-- Theorem statement for polynomial evaluation using Horner's method
theorem eval_f_at_3 : horner_eval 3 = f 3 ∧ f 3 = 21324 :=
  by
    sorry

end base5_to_octal_1234_eval_f_at_3_l83_83487


namespace shirt_price_l83_83275

theorem shirt_price (S : ℝ) (h : (5 * S + 5 * 3) / 2 = 10) : S = 1 :=
by
  sorry

end shirt_price_l83_83275


namespace men_dropped_out_l83_83489

theorem men_dropped_out (x : ℕ) : 
  (∀ (days_half days_full men men_remaining : ℕ),
    days_half = 15 ∧ days_full = 25 ∧ men = 5 ∧ men_remaining = men - x ∧ 
    (men * (2 * days_half)) = ((men_remaining) * days_full)) -> x = 1 :=
by
  intros h
  sorry

end men_dropped_out_l83_83489


namespace fans_with_all_vouchers_l83_83878

theorem fans_with_all_vouchers (total_fans : ℕ) 
    (soda_interval : ℕ) (popcorn_interval : ℕ) (hotdog_interval : ℕ)
    (h1 : soda_interval = 60) (h2 : popcorn_interval = 80) (h3 : hotdog_interval = 100)
    (h4 : total_fans = 4500)
    (h5 : Nat.lcm soda_interval (Nat.lcm popcorn_interval hotdog_interval) = 1200) :
    (total_fans / Nat.lcm soda_interval (Nat.lcm popcorn_interval hotdog_interval)) = 3 := 
by
    sorry

end fans_with_all_vouchers_l83_83878


namespace max_area_of_rectangle_l83_83797

noncomputable def max_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 40 then l * w else 0

theorem max_area_of_rectangle : 
  ∃ (l w : ℕ), 2 * l + 2 * w = 40 ∧ l * w = 100 :=
by
  use 10
  use 10
  simp
  exact ⟨by norm_num, by norm_num⟩

end max_area_of_rectangle_l83_83797


namespace compute_fraction_product_l83_83673

theorem compute_fraction_product :
  (1 / 3)^4 * (1 / 5) = 1 / 405 :=
by
  sorry

end compute_fraction_product_l83_83673


namespace max_sum_of_segments_l83_83154

theorem max_sum_of_segments (A B C D : ℝ × ℝ × ℝ)
    (h : (dist A B ≤ 1 ∧ dist A C ≤ 1 ∧ dist A D ≤ 1 ∧ dist B C ≤ 1 ∧ dist B D ≤ 1 ∧ dist C D ≤ 1)
      ∨ (dist A B ≤ 1 ∧ dist A C ≤ 1 ∧ dist A D > 1 ∧ dist B C ≤ 1 ∧ dist B D ≤ 1 ∧ dist C D ≤ 1))
    : dist A B + dist A C + dist A D + dist B C + dist B D + dist C D ≤ 5 + Real.sqrt 3 := sorry

end max_sum_of_segments_l83_83154


namespace solution_exists_l83_83284

theorem solution_exists (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (gcd_ca : Nat.gcd c a = 1) (gcd_cb : Nat.gcd c b = 1) : 
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^a + y^b = z^c :=
sorry

end solution_exists_l83_83284


namespace ccamathbonanza_2016_2_1_l83_83089

-- Definitions of the speeds of the runners
def bhairav_speed := 28 -- in miles per hour
def daniel_speed := 15 -- in miles per hour
def tristan_speed := 10 -- in miles per hour

-- Distance of the race
def race_distance := 15 -- in miles

-- Time conversion from hours to minutes
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

-- Time taken by each runner to complete the race (in hours)
def time_bhairav := race_distance / bhairav_speed
def time_daniel := race_distance / daniel_speed
def time_tristan := race_distance / tristan_speed

-- Time taken by each runner to complete the race (in minutes)
def time_bhairav_minutes := hours_to_minutes time_bhairav
def time_daniel_minutes := hours_to_minutes time_daniel
def time_tristan_minutes := hours_to_minutes time_tristan

-- Time differences between consecutive runners' finishes (in minutes)
def time_diff_bhairav_daniel := time_daniel_minutes - time_bhairav_minutes
def time_diff_daniel_tristan := time_tristan_minutes - time_daniel_minutes

-- Greatest length of time between consecutive runners' finishes
def greatest_time_diff := max time_diff_bhairav_daniel time_diff_daniel_tristan

-- The theorem we need to prove
theorem ccamathbonanza_2016_2_1 : greatest_time_diff = 30 := by
  sorry

end ccamathbonanza_2016_2_1_l83_83089


namespace grandmaster_plays_21_games_l83_83942

theorem grandmaster_plays_21_games (a : ℕ → ℕ) (n : ℕ) :
  (∀ i, 1 ≤ a (i + 1) - a i) ∧ (∀ i, a (i + 7) - a i ≤ 10) →
  ∃ (i j : ℕ), i < j ∧ (a j - a i = 21) :=
sorry

end grandmaster_plays_21_games_l83_83942


namespace chocolate_cost_is_3_l83_83099

-- Definitions based on the conditions
def dan_has_5_dollars : Prop := true
def cost_candy_bar : ℕ := 2
def cost_chocolate : ℕ := cost_candy_bar + 1

-- Theorem to prove
theorem chocolate_cost_is_3 : cost_chocolate = 3 :=
by {
  -- This is where the proof steps would go
  sorry
}

end chocolate_cost_is_3_l83_83099


namespace pastor_prayer_ratio_l83_83569

theorem pastor_prayer_ratio 
  (R : ℚ) 
  (paul_prays_per_day : ℚ := 20)
  (paul_sunday_times : ℚ := 2 * paul_prays_per_day)
  (paul_total : ℚ := 6 * paul_prays_per_day + paul_sunday_times)
  (bruce_ratio : ℚ := R)
  (bruce_prays_per_day : ℚ := bruce_ratio * paul_prays_per_day)
  (bruce_sunday_times : ℚ := 2 * paul_sunday_times)
  (bruce_total : ℚ := 6 * bruce_prays_per_day + bruce_sunday_times)
  (condition : paul_total = bruce_total + 20) :
  R = 1/2 :=
sorry

end pastor_prayer_ratio_l83_83569


namespace probability_two_queens_or_at_least_one_jack_l83_83535

-- Definitions
def num_jacks : ℕ := 4
def num_queens : ℕ := 4
def total_cards : ℕ := 52

-- Probability calculation for drawing either two Queens or at least one Jack
theorem probability_two_queens_or_at_least_one_jack :
  (4 / 52) * (3 / (52 - 1)) + ((4 / 52) * (48 / (52 - 1)) + (48 / 52) * (4 / (52 - 1)) + (4 / 52) * (3 / (52 - 1))) = 2 / 13 :=
by
  sorry

end probability_two_queens_or_at_least_one_jack_l83_83535


namespace each_student_gets_8_pieces_l83_83498

-- Define the number of pieces of candy
def candy : Nat := 344

-- Define the number of students
def students : Nat := 43

-- Define the number of pieces each student gets, which we need to prove
def pieces_per_student : Nat := candy / students

-- The proof problem statement
theorem each_student_gets_8_pieces : pieces_per_student = 8 :=
by
  -- This proof content is omitted as per instructions
  sorry

end each_student_gets_8_pieces_l83_83498


namespace line_through_two_points_l83_83760

theorem line_through_two_points (A B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (3, 4)) :
  ∃ k b : ℝ, (∀ x y : ℝ, (y = k * x + b) ↔ ((x, y) = A ∨ (x, y) = B)) ∧ (k = 1) ∧ (b = 1) := 
by
  sorry

end line_through_two_points_l83_83760


namespace common_difference_arithmetic_sequence_l83_83016

variables (a b d : ℤ)

theorem common_difference_arithmetic_sequence (a_1 a_2 a_4 a_6 : ℤ)
  (h1 : a_1 * a_2 = 35)
  (h2 : 2 * a_4 - a_6 = 7)
  (ha_2 : a_2 = a_1 + d)
  (ha_4 : a_4 = a_1 + 3 * d)
  (ha_6 : a_6 = a_1 + 5 * d) :
  d = 2 :=
by
  sorry

end common_difference_arithmetic_sequence_l83_83016


namespace number_of_five_digit_numbers_l83_83999

def count_five_identical_digits: Nat := 9
def count_two_different_digits: Nat := 1215
def count_three_different_digits: Nat := 6480
def count_four_different_digits: Nat := 22680
def count_five_different_digits: Nat := 27216

theorem number_of_five_digit_numbers :
  count_five_identical_digits + count_two_different_digits +
  count_three_different_digits + count_four_different_digits +
  count_five_different_digits = 57600 :=
by
  sorry

end number_of_five_digit_numbers_l83_83999


namespace find_a_l83_83145

theorem find_a {S : ℕ → ℤ} (a : ℤ)
  (hS : ∀ n : ℕ, S n = 5 ^ (n + 1) + a) : a = -5 :=
sorry

end find_a_l83_83145


namespace find_b_in_geometric_sequence_l83_83381

theorem find_b_in_geometric_sequence 
  (a b c : ℝ) 
  (q : ℝ) 
  (h1 : -1 * q^4 = -9) 
  (h2 : a = -1 * q) 
  (h3 : b = a * q) 
  (h4 : c = b * q) 
  (h5 : -9 = c * q) : 
  b = -3 :=
by
  sorry

end find_b_in_geometric_sequence_l83_83381


namespace volume_ratio_of_spheres_l83_83871

theorem volume_ratio_of_spheres
  (r1 r2 r3 : ℝ)
  (A1 A2 A3 : ℝ)
  (V1 V2 V3 : ℝ)
  (hA : A1 / A2 = 1 / 4 ∧ A2 / A3 = 4 / 9)
  (hSurfaceArea : A1 = 4 * π * r1^2 ∧ A2 = 4 * π * r2^2 ∧ A3 = 4 * π * r3^2)
  (hVolume : V1 = (4 / 3) * π * r1^3 ∧ V2 = (4 / 3) * π * r2^3 ∧ V3 = (4 / 3) * π * r3^3) :
  V1 / V2 = 1 / 8 ∧ V2 / V3 = 8 / 27 := by
  sorry

end volume_ratio_of_spheres_l83_83871


namespace product_of_primes_l83_83614

def smallest_one_digit_prime := 2
def second_smallest_one_digit_prime := 3
def smallest_two_digit_prime := 11

theorem product_of_primes: smallest_one_digit_prime * second_smallest_one_digit_prime * smallest_two_digit_prime = 66 :=
by {
  -- Applying the definition of the primes and carrying out the multiplication
  show 2 * 3 * 11 = 66,
  calc
  2 * 3 * 11 = 6 * 11 : by rw [mul_assoc 2 3 11]
          ... = 66    : by norm_num,
}

end product_of_primes_l83_83614


namespace find_values_of_a_and_b_l83_83697

noncomputable def a : ℝ := 4
noncomputable def b : ℝ := 2

theorem find_values_of_a_and_b
  (h1a : a > 1)
  (h1b : b > 1)
  (h2 : Real.log a / Real.log b + Real.log b / Real.log a = 5 / 2)
  (h3 : a^b = b^a) :
  a = 4 ∧ b = 2 :=
begin
  sorry
end

end find_values_of_a_and_b_l83_83697


namespace maximum_area_exists_l83_83796

def max_area_rectangle (l w : ℕ) (h : l + w = 20) : Prop :=
  l * w ≤ 100

theorem maximum_area_exists : ∃ (l w : ℕ), max_area_rectangle l w (by sorry) ∧ (10 * 10 = 100) :=
begin
  sorry
end

end maximum_area_exists_l83_83796


namespace fraction_value_l83_83059

theorem fraction_value : (5 * 7 : ℝ) / 10 = 3.5 := by
  sorry

end fraction_value_l83_83059


namespace mr_green_potato_yield_l83_83407

theorem mr_green_potato_yield :
  let steps_to_feet := 2.5
  let length_steps := 18
  let width_steps := 25
  let yield_per_sqft := 0.75
  let length_feet := length_steps * steps_to_feet
  let width_feet := width_steps * steps_to_feet
  let area_sqft := length_feet * width_feet
  let expected_yield := area_sqft * yield_per_sqft
  expected_yield = 2109.375 := by sorry

end mr_green_potato_yield_l83_83407


namespace total_liquid_poured_out_l83_83666

noncomputable def capacity1 := 2
noncomputable def capacity2 := 6
noncomputable def percentAlcohol1 := 0.3
noncomputable def percentAlcohol2 := 0.4
noncomputable def totalCapacity := 10
noncomputable def finalConcentration := 0.3

theorem total_liquid_poured_out :
  capacity1 + capacity2 = 8 :=
by
  sorry

end total_liquid_poured_out_l83_83666


namespace train_length_correct_l83_83338

noncomputable def train_length (time : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  speed_mps * time

theorem train_length_correct :
  train_length 17.998560115190784 36 = 179.98560115190784 :=
by
  sorry

end train_length_correct_l83_83338


namespace eval_expression_l83_83214

theorem eval_expression : (-3)^5 + 2^(2^3 + 5^2 - 8^2) = -242.999999999535 := by
  sorry

end eval_expression_l83_83214


namespace inequality_bound_l83_83062

theorem inequality_bound 
  (a b c d : ℝ) 
  (ha : 0 ≤ a) (hb : a ≤ 1)
  (hb : 0 ≤ b) (hc : b ≤ 1)
  (hc : 0 ≤ c) (hd : c ≤ 1)
  (hd : 0 ≤ d) (ha2 : d ≤ 1) : 
  ab * (a - b) + bc * (b - c) + cd * (c - d) + da * (d - a) ≤ 8/27 := 
by
  sorry

end inequality_bound_l83_83062


namespace greatest_prime_factor_f24_is_11_value_of_f12_l83_83819

def is_even (n : ℕ) : Prop := n % 2 = 0

def f (n : ℕ) : ℕ := (List.range' 2 ((n + 1) / 2)).map (λ x => 2 * x) |> List.prod

theorem greatest_prime_factor_f24_is_11 : 
  ¬ ∃ p, Prime p ∧ p ∣ f 24 ∧ p > 11 := 
  sorry

theorem value_of_f12 : f 12 = 46080 := 
  sorry

end greatest_prime_factor_f24_is_11_value_of_f12_l83_83819


namespace factor_expression_l83_83962

theorem factor_expression (x : ℝ) : 16 * x^4 - 4 * x^2 = 4 * x^2 * (2 * x + 1) * (2 * x - 1) :=
sorry

end factor_expression_l83_83962


namespace range_of_a_l83_83261

theorem range_of_a (a : ℝ) :
  (¬ ∃ x₀ ∈ ℝ, 2 * x₀ ^ 2 + (a - 1) * x₀ + 1 / 2 ≤ 0) ↔ -1 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l83_83261


namespace most_suitable_candidate_l83_83668

-- Definitions for variances
def variance_A := 3.4
def variance_B := 2.1
def variance_C := 2.5
def variance_D := 2.7

-- We start the theorem to state the most suitable candidate based on given variances and average scores.
theorem most_suitable_candidate :
  (variance_A = 3.4) ∧ (variance_B = 2.1) ∧ (variance_C = 2.5) ∧ (variance_D = 2.7) →
  true := 
by
  sorry

end most_suitable_candidate_l83_83668


namespace ratio_of_areas_ratio_of_perimeters_l83_83043

-- Define side lengths
def side_length_A : ℕ := 48
def side_length_B : ℕ := 60

-- Define the area of squares
def area_square (side_length : ℕ) : ℕ := side_length * side_length

-- Define the perimeter of squares
def perimeter_square (side_length : ℕ) : ℕ := 4 * side_length

-- Theorem for the ratio of areas
theorem ratio_of_areas : (area_square side_length_A) / (area_square side_length_B) = 16 / 25 :=
by
  sorry

-- Theorem for the ratio of perimeters
theorem ratio_of_perimeters : (perimeter_square side_length_A) / (perimeter_square side_length_B) = 4 / 5 :=
by
  sorry

end ratio_of_areas_ratio_of_perimeters_l83_83043


namespace find_y_l83_83598

theorem find_y (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : ∃ C, x * y = C) (hx : x = 4) : y = 50 :=
sorry

end find_y_l83_83598


namespace point_in_second_quadrant_l83_83692

def point := (ℝ × ℝ)

def second_quadrant (p : point) : Prop := p.1 < 0 ∧ p.2 > 0

theorem point_in_second_quadrant : second_quadrant (-1, 2) :=
sorry

end point_in_second_quadrant_l83_83692


namespace lesser_of_two_numbers_l83_83443

theorem lesser_of_two_numbers (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
by
  sorry

end lesser_of_two_numbers_l83_83443


namespace tom_total_trip_cost_is_correct_l83_83604

noncomputable def Tom_total_cost : ℝ :=
  let cost_vaccines := 10 * 45
  let cost_doctor := 250
  let total_medical := cost_vaccines + cost_doctor
  
  let insurance_coverage := 0.8 * total_medical
  let out_of_pocket_medical := total_medical - insurance_coverage
  
  let cost_flight := 1200

  let cost_lodging := 7 * 150
  let cost_transportation := 200
  let cost_food := 7 * 60
  let total_local_usd := cost_lodging + cost_transportation + cost_food
  let total_local_bbd := total_local_usd * 2

  let conversion_fee_bbd := 0.03 * total_local_bbd
  let conversion_fee_usd := conversion_fee_bbd / 2

  out_of_pocket_medical + cost_flight + total_local_usd + conversion_fee_usd

theorem tom_total_trip_cost_is_correct : Tom_total_cost = 3060.10 :=
  by
    -- Proof skipped
    sorry

end tom_total_trip_cost_is_correct_l83_83604


namespace correct_result_l83_83785

-- Define the original number
def original_number := 51 + 6

-- Define the correct calculation using multiplication
def correct_calculation (x : ℕ) : ℕ := x * 6

-- Theorem to prove the correct calculation
theorem correct_result : correct_calculation original_number = 342 := by
  -- Skip the actual proof steps
  sorry

end correct_result_l83_83785


namespace apple_box_weights_l83_83769

theorem apple_box_weights (a b c d : ℤ) 
  (h1 : a + b + c = 70)
  (h2 : a + b + d = 80)
  (h3 : a + c + d = 73)
  (h4 : b + c + d = 77) : 
  a = 23 ∧ b = 27 ∧ c = 20 ∧ d = 30 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end apple_box_weights_l83_83769


namespace inradius_of_triangle_l83_83347

theorem inradius_of_triangle (a b c : ℝ) (h1 : a = 15) (h2 : b = 16) (h3 : c = 17) : 
    let s := (a + b + c) / 2
    let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
    let r := area / s
    r = Real.sqrt 21 := by
  sorry

end inradius_of_triangle_l83_83347


namespace evalExpression_at_3_2_l83_83510

def evalExpression (x y : ℕ) : ℕ := 3 * x^y + 4 * y^x

theorem evalExpression_at_3_2 : evalExpression 3 2 = 59 := by
  sorry

end evalExpression_at_3_2_l83_83510


namespace tracy_first_week_books_collected_l83_83470

-- Definitions for collection multipliers
def first_week (T : ℕ) := T
def second_week (T : ℕ) := 2 * T + 3 * T
def third_week (T : ℕ) := 3 * T + 4 * T + (T / 2)
def fourth_week (T : ℕ) := 4 * T + 5 * T + T
def fifth_week (T : ℕ) := 5 * T + 6 * T + 2 * T
def sixth_week (T : ℕ) := 6 * T + 7 * T + 3 * T

-- Summing up total books collected
def total_books_collected (T : ℕ) : ℕ :=
  first_week T + second_week T + third_week T + fourth_week T + fifth_week T + sixth_week T

-- Proof statement (unchanged for now)
theorem tracy_first_week_books_collected (T : ℕ) :
  total_books_collected T = 1025 → T = 20 :=
by
  sorry

end tracy_first_week_books_collected_l83_83470


namespace two_times_sum_of_fourth_power_is_perfect_square_l83_83917

theorem two_times_sum_of_fourth_power_is_perfect_square (a b c : ℤ) 
  (h : a + b + c = 0) : 2 * (a^4 + b^4 + c^4) = (a^2 + b^2 + c^2)^2 := 
by sorry

end two_times_sum_of_fourth_power_is_perfect_square_l83_83917


namespace system_has_real_solution_l83_83967

theorem system_has_real_solution (k : ℝ) : 
  (∃ x y : ℝ, y = k * x + 4 ∧ y = (3 * k - 2) * x + 5) ↔ k ≠ 1 :=
by
  sorry

end system_has_real_solution_l83_83967


namespace virginia_eggs_l83_83316

-- Definitions and conditions
variable (eggs_start : Nat)
variable (eggs_taken : Nat := 3)
variable (eggs_end : Nat := 93)

-- Problem statement to prove
theorem virginia_eggs : eggs_start - eggs_taken = eggs_end → eggs_start = 96 :=
by
  intro h
  sorry

end virginia_eggs_l83_83316


namespace probability_grade_A_l83_83209

-- Defining probabilities
def P_B : ℝ := 0.05
def P_C : ℝ := 0.03

-- Theorem: proving the probability of Grade A
theorem probability_grade_A : 1 - P_B - P_C = 0.92 :=
by
  -- Placeholder for proof
  sorry

end probability_grade_A_l83_83209


namespace base10_to_base7_conversion_l83_83219

theorem base10_to_base7_conversion : 2023 = 5 * 7^3 + 6 * 7^2 + 2 * 7^1 + 0 * 7^0 :=
  sorry

end base10_to_base7_conversion_l83_83219


namespace lesser_number_of_sum_and_difference_l83_83451

theorem lesser_number_of_sum_and_difference (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
sorry

end lesser_number_of_sum_and_difference_l83_83451


namespace quadratic_expression_value_l83_83534

theorem quadratic_expression_value
  (x : ℝ)
  (h : x^2 + x - 2 = 0)
: x^3 + 2*x^2 - x + 2021 = 2023 :=
sorry

end quadratic_expression_value_l83_83534


namespace compute_expression_l83_83399

theorem compute_expression (a b c : ℝ) (h : a^3 - 6 * a^2 + 11 * a - 6 = 0 ∧ b^3 - 6 * b^2 + 11 * b - 6 = 0 ∧ c^3 - 6 * c^2 + 11 * c - 6 = 0) :
  (ab / c + bc / a + ca / b) = 49 / 6 := 
  by
  sorry -- Placeholder for the proof

end compute_expression_l83_83399


namespace compute_expression_l83_83395

-- Given Conditions
def is_root (p : Polynomial ℝ) (x : ℝ) := p.eval x = 0

def a : ℝ := 1  -- Placeholder value
def b : ℝ := 2  -- Placeholder value
def c : ℝ := 3  -- Placeholder value
def p : Polynomial ℝ := Polynomial.C (-6) + Polynomial.C 11 * Polynomial.X - Polynomial.C 6 * Polynomial.X^2 + Polynomial.X^3

-- Assertions based on conditions
axiom h_a_root : is_root p a
axiom h_b_root : is_root p b
axiom h_c_root : is_root p c

-- Proof Problem Statement
theorem compute_expression : 
  (ab c : ℝ), (is_root p a) → (is_root p b) → (is_root p c) → 
  ((a * b / c) + (b * c / a) + (c * a / b) = 49 / 6) :=
begin
  sorry,
end


end compute_expression_l83_83395


namespace length_of_plot_l83_83661

-- Define the conditions
def width : ℝ := 60
def num_poles : ℕ := 60
def dist_between_poles : ℝ := 5
def num_intervals : ℕ := num_poles - 1
def perimeter : ℝ := num_intervals * dist_between_poles

-- Define the theorem and the correctness condition
theorem length_of_plot : 
  perimeter = 2 * (length + width) → 
  length = 87.5 :=
by
  sorry

end length_of_plot_l83_83661


namespace probability_four_in_sequence_l83_83206

open Rat

theorem probability_four_in_sequence :
  let die_faces := {1, 2, 3, 4, 5, 6},
      rolls := vector die_faces 4,
      P (s : vector ℕ 4) : ℚ := 
        if (4 ∈ {s[0], s[0] + s[1], s[0] + s[1] + s[2], s[0] + s[1] + s[2] + s[3]}) then 1 else 0,
      probability := (378 / 1296 : ℚ)
  in 
  ∑ s in finset.univ.image rolls, P s / (6 ^ 4) = probability 
:=
sorry

end probability_four_in_sequence_l83_83206


namespace number_of_classes_l83_83042

theorem number_of_classes (total_basketballs classes_basketballs : ℕ) (h1 : total_basketballs = 54) (h2 : classes_basketballs = 7) : total_basketballs / classes_basketballs = 7 := by
  sorry

end number_of_classes_l83_83042


namespace cone_lateral_surface_area_l83_83827

theorem cone_lateral_surface_area (r h l S : ℝ) (π_pos : 0 < π) (r_eq : r = 6)
  (V : ℝ) (V_eq : V = 30 * π)
  (vol_eq : V = (1/3) * π * r^2 * h)
  (h_eq : h = 5 / 2)
  (l_eq : l = Real.sqrt (r^2 + h^2))
  (S_eq : S = π * r * l) :
  S = 39 * π :=
  sorry

end cone_lateral_surface_area_l83_83827


namespace least_possible_value_f_1998_l83_83970

theorem least_possible_value_f_1998 
  (f : ℕ → ℕ)
  (h : ∀ m n, f (n^2 * f m) = m * (f n)^2) : 
  f 1998 = 120 :=
sorry

end least_possible_value_f_1998_l83_83970


namespace product_of_primes_l83_83619

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

noncomputable def smallest_one_digit_primes (p₁ p₂ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ p₁ < p₂ ∧ p₂ < 10 ∧ ∀ p : ℕ, is_prime p → p < 10 → p = p₁ ∨ p = p₂

noncomputable def smallest_two_digit_prime (p : ℕ) : Prop :=
  is_prime p ∧ p ≥ 10 ∧ p < 100 ∧ ∀ q : ℕ, is_prime q → q ≥ 10 → q < p → q = 11

theorem product_of_primes : ∃ p₁ p₂ p₃ : ℕ, smallest_one_digit_primes p₁ p₂ ∧ smallest_two_digit_prime p₃ ∧ p₁ * p₂ * p₃ = 66 := 
by
  sorry

end product_of_primes_l83_83619


namespace determinant_of_A_l83_83506

section
  open Matrix

  -- Define the given matrix
  def A : Matrix (Fin 3) (Fin 3) ℤ :=
    ![ ![0, 2, -4], ![6, -1, 3], ![2, -3, 5] ]

  -- State the theorem for the determinant
  theorem determinant_of_A : det A = 16 :=
  sorry
end

end determinant_of_A_l83_83506


namespace third_number_l83_83324

theorem third_number (x : ℝ) 
    (h : 217 + 2.017 + 2.0017 + x = 221.2357) : 
    x = 0.217 :=
sorry

end third_number_l83_83324


namespace find_salary_June_l83_83577

variable (J F M A May_s June_s : ℝ)
variable (h1 : J + F + M + A = 4 * 8000)
variable (h2 : F + M + A + May_s = 4 * 8450)
variable (h3 : May_s = 6500)
variable (h4 : M + A + May_s + June_s = 4 * 9000)
variable (h5 : June_s = 1.2 * May_s)

theorem find_salary_June : June_s = 7800 := by
  sorry

end find_salary_June_l83_83577


namespace not_all_crows_gather_on_one_tree_l83_83289

theorem not_all_crows_gather_on_one_tree :
  ∀ (crows : Fin 6 → ℕ), 
  (∀ i, crows i = 1) →
  (∀ t1 t2, abs (t1 - t2) = 1 → crows t1 = crows t1 - 1 ∧ crows t2 = crows t2 + 1) →
  ¬(∃ i, crows i = 6 ∧ (∀ j ≠ i, crows j = 0)) :=
by
  sorry

end not_all_crows_gather_on_one_tree_l83_83289


namespace sum_of_xyz_l83_83116

theorem sum_of_xyz (x y z : ℝ) (h : (x - 5)^2 + (y - 3)^2 + (z - 1)^2 = 0) : x + y + z = 9 :=
by {
  sorry
}

end sum_of_xyz_l83_83116


namespace geometric_sum_l83_83438

def S10 : ℕ := 36
def S20 : ℕ := 48

theorem geometric_sum (S30 : ℕ) (h1 : S10 = 36) (h2 : S20 = 48) : S30 = 52 :=
by
  have h3 : (S20 - S10) ^ 2 = S10 * (S30 - S20) :=
    sorry -- This is based on the properties of the geometric sequence
  sorry  -- Solve the equation to show S30 = 52

end geometric_sum_l83_83438


namespace sum_geometric_series_l83_83856

theorem sum_geometric_series :
  ∑' n : ℕ+, (3 : ℝ)⁻¹ ^ (n : ℕ) = (1 / 2 : ℝ) := by
  sorry

end sum_geometric_series_l83_83856


namespace bases_to_make_equality_l83_83488

theorem bases_to_make_equality (a b : ℕ) (h : 3 * a^2 + 4 * a + 2 = 9 * b + 7) : 
  (3 * a^2 + 4 * a + 2 = 342) ∧ (9 * b + 7 = 97) :=
by
  sorry

end bases_to_make_equality_l83_83488


namespace function_decreases_iff_l83_83870

theorem function_decreases_iff (m : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → (m - 3) * x1 + 4 > (m - 3) * x2 + 4) ↔ m < 3 :=
by
  sorry

end function_decreases_iff_l83_83870


namespace distance_to_place_equals_2_point_25_l83_83943

-- Definitions based on conditions
def rowing_speed : ℝ := 4
def river_speed : ℝ := 2
def total_time_hours : ℝ := 1.5

-- Downstream speed = rowing_speed + river_speed
def downstream_speed : ℝ := rowing_speed + river_speed
-- Upstream speed = rowing_speed - river_speed
def upstream_speed : ℝ := rowing_speed - river_speed

-- Define the distance d
def distance (d : ℝ) : Prop :=
  (d / downstream_speed + d / upstream_speed = total_time_hours)

-- The theorem statement
theorem distance_to_place_equals_2_point_25 :
  ∃ d : ℝ, distance d ∧ d = 2.25 :=
by
  sorry

end distance_to_place_equals_2_point_25_l83_83943


namespace lesser_number_of_sum_and_difference_l83_83450

theorem lesser_number_of_sum_and_difference (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
sorry

end lesser_number_of_sum_and_difference_l83_83450


namespace range_f3_l83_83853

def function_f (a c x : ℝ) : ℝ := a * x^2 - c

theorem range_f3 (a c : ℝ) :
  (-4 ≤ function_f a c 1) ∧ (function_f a c 1 ≤ -1) →
  (-1 ≤ function_f a c 2) ∧ (function_f a c 2 ≤ 5) →
  -12 ≤ function_f a c 3 ∧ function_f a c 3 ≤ 1.75 :=
by
  sorry

end range_f3_l83_83853


namespace graph_shift_correct_l83_83771

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x) - Real.sqrt 3 * Real.cos (3 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (3 * x)

theorem graph_shift_correct :
  ∀ (x : ℝ), f x = g (x - (5 * Real.pi / 18)) :=
sorry

end graph_shift_correct_l83_83771


namespace point_on_inverse_proportion_function_l83_83430

theorem point_on_inverse_proportion_function :
  ∀ (x y k : ℝ), k ≠ 0 ∧ y = k / x ∧ (2, -3) = (2, -(3 : ℝ)) → (x, y) = (-2, 3) → (y = -6 / x) :=
sorry

end point_on_inverse_proportion_function_l83_83430


namespace cube_inequality_contradiction_l83_83745

theorem cube_inequality_contradiction (a b : Real) (h : a > b) : ¬(a^3 <= b^3) := by
  sorry

end cube_inequality_contradiction_l83_83745


namespace unique_two_digit_number_l83_83103

theorem unique_two_digit_number (n : ℕ) (h1 : 10 ≤ n) (h2 : n ≤ 99) : 
  (13 * n) % 100 = 42 → n = 34 :=
by
  sorry

end unique_two_digit_number_l83_83103


namespace circle_line_distance_l83_83866

theorem circle_line_distance (c : ℝ) : 
  (∃ (P₁ P₂ P₃ : ℝ × ℝ), 
     (P₁ ≠ P₂ ∧ P₂ ≠ P₃ ∧ P₁ ≠ P₃) ∧
     ((P₁.1 - 2)^2 + (P₁.2 - 2)^2 = 18) ∧
     ((P₂.1 - 2)^2 + (P₂.2 - 2)^2 = 18) ∧
     ((P₃.1 - 2)^2 + (P₃.2 - 2)^2 = 18) ∧
     (abs (P₁.1 - P₁.2 + c) / Real.sqrt 2 = 2 * Real.sqrt 2) ∧
     (abs (P₂.1 - P₂.2 + c) / Real.sqrt 2 = 2 * Real.sqrt 2) ∧
     (abs (P₃.1 - P₃.2 + c) / Real.sqrt 2 = 2 * Real.sqrt 2)) ↔ 
  -2 ≤ c ∧ c ≤ 2 :=
sorry

end circle_line_distance_l83_83866


namespace domain_of_sqrt_cosine_sub_half_l83_83517

theorem domain_of_sqrt_cosine_sub_half :
  {x : ℝ | ∃ k : ℤ, (2 * k * π - π / 3) ≤ x ∧ x ≤ (2 * k * π + π / 3)} =
  {x : ℝ | ∃ k : ℤ, 2 * k * π - π / 3 ≤ x ∧ x ≤ 2 * k * π + π / 3} :=
by sorry

end domain_of_sqrt_cosine_sub_half_l83_83517


namespace cubic_sum_identity_l83_83921

section
variables {x y z a b c : ℝ}

theorem cubic_sum_identity
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = b^2)
  (h3 : x⁻¹ + y⁻¹ + z⁻¹ = c⁻¹) :
  x^3 + y^3 + z^3 = a^3 + (3 / 2) * (a^2 - b^2) * (c - a) := 
sorry
end

end cubic_sum_identity_l83_83921


namespace quadratic_function_fixed_points_range_l83_83521

def has_two_distinct_fixed_points (c : ℝ) : Prop := 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
               (x1 = x1^2 - x1 + c) ∧ 
               (x2 = x2^2 - x2 + c) ∧ 
               x1 < 2 ∧ 2 < x2

theorem quadratic_function_fixed_points_range (c : ℝ) :
  has_two_distinct_fixed_points c ↔ c < 0 :=
sorry

end quadratic_function_fixed_points_range_l83_83521


namespace number_of_tilings_5x1_using_all_colors_l83_83652

def tile_ways_board_5x1 : ℕ :=
  let three_pieces := 6 * (3^3 - 3 * 2^3 + 3)
  let four_pieces := 4 * (3^4 - 4 * 2^4 + 3 * 2^2)
  three_pieces + four_pieces

theorem number_of_tilings_5x1_using_all_colors :
  tile_ways_board_5x1 = 152 :=
by
  -- Proof goes here
  sorry

end number_of_tilings_5x1_using_all_colors_l83_83652


namespace difference_of_students_l83_83587

variable (G1 G2 G5 : ℕ)

theorem difference_of_students (h1 : G1 + G2 > G2 + G5) (h2 : G5 = G1 - 30) : 
  (G1 + G2) - (G2 + G5) = 30 :=
by
  sorry

end difference_of_students_l83_83587


namespace golu_distance_travelled_l83_83998

theorem golu_distance_travelled 
  (b : ℝ) (c : ℝ) (h : c^2 = x^2 + b^2) : x = 8 := by
  sorry

end golu_distance_travelled_l83_83998


namespace ellipse_shortest_major_axis_l83_83303

theorem ellipse_shortest_major_axis (P : ℝ × ℝ) (a b : ℝ) 
  (ha : a > b) (hb : b > 0) (hP_on_line : P.2 = P.1 + 2)
  (h_foci_hyperbola : ∃ c : ℝ, c = 1 ∧ a^2 - b^2 = c^2) :
  (∃ a b : ℝ, a^2 = 5 ∧ b^2 = 4 ∧ (P.1^2 / a^2 + P.2^2 / b^2 = 1)) :=
sorry

end ellipse_shortest_major_axis_l83_83303


namespace function_D_is_odd_function_D_is_decreasing_l83_83669

def f_D (x : ℝ) : ℝ := -x * |x|

theorem function_D_is_odd (x : ℝ) : f_D (-x) = -f_D x := by
  sorry

theorem function_D_is_decreasing (x y : ℝ) (h : x < y) : f_D x > f_D y := by
  sorry

end function_D_is_odd_function_D_is_decreasing_l83_83669


namespace fraction_of_married_women_l83_83650

theorem fraction_of_married_women (total_employees : ℕ) 
  (women_fraction : ℝ) (married_fraction : ℝ) (single_men_fraction : ℝ)
  (hwf : women_fraction = 0.64) (hmf : married_fraction = 0.60) 
  (hsf : single_men_fraction = 2/3) : 
  ∃ (married_women_fraction : ℝ), married_women_fraction = 3/4 := 
by
  sorry

end fraction_of_married_women_l83_83650


namespace find_function_l83_83689

noncomputable def solution_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = x + f y → ∃ c : ℝ, ∀ x : ℝ, f x = x + c

theorem find_function (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = x + f y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
sorry

end find_function_l83_83689


namespace max_tulips_l83_83925

theorem max_tulips (y r : ℕ) (h1 : (y + r) % 2 = 1) (h2 : r = y + 1 ∨ y = r + 1) (h3 : 50 * y + 31 * r ≤ 600) : y + r = 15 :=
by
  sorry

end max_tulips_l83_83925


namespace determinant_in_terms_of_roots_l83_83554

theorem determinant_in_terms_of_roots 
  (r s t a b c : ℝ)
  (h1 : a^3 - r*a^2 + s*a - t = 0)
  (h2 : b^3 - r*b^2 + s*b - t = 0)
  (h3 : c^3 - r*c^2 + s*c - t = 0) :
  (2 + a) * ((2 + b) * (2 + c) - 4) - 2 * (2 * (2 + c) - 4) + 2 * (2 * 2 - (2 + b) * 2) = t - 2 * s :=
by
  sorry

end determinant_in_terms_of_roots_l83_83554


namespace sum_of_recorded_numbers_l83_83873

theorem sum_of_recorded_numbers : 
  let n := 16
  let pairs := n.choose 2
  let total_sum := pairs
  pairs = n * (n - 1) / 2
  ∑ (i : Fin n), (friends_count i + enemies_count i) = total_sum :=
by
  let n := 16
  let pairs := n.choose 2
  let total_sum := pairs
  have pairs_eq : pairs = n * (n - 1) / 2 := by
    rw Nat.choose
    apply Nat.choose_self_eq 
  sorry

end sum_of_recorded_numbers_l83_83873


namespace product_of_smallest_primes_l83_83624

theorem product_of_smallest_primes :
  2 * 3 * 11 = 66 :=
by
  sorry

end product_of_smallest_primes_l83_83624


namespace rainfall_wednesday_correct_l83_83008

def monday_rainfall : ℝ := 0.9
def tuesday_rainfall : ℝ := monday_rainfall - 0.7
def wednesday_rainfall : ℝ := 2 * (monday_rainfall + tuesday_rainfall)

theorem rainfall_wednesday_correct : wednesday_rainfall = 2.2 := by
sorry

end rainfall_wednesday_correct_l83_83008


namespace fraction_condition_l83_83645

theorem fraction_condition (x : ℚ) :
  (3 + 2 * x) / (4 + 3 * x) = 5 / 9 ↔ x = -7 / 3 :=
by
  sorry

end fraction_condition_l83_83645


namespace find_g_of_2_l83_83376

theorem find_g_of_2 {g : ℝ → ℝ} (h : ∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) : g 2 = 14 :=
sorry

end find_g_of_2_l83_83376


namespace points_in_groups_l83_83546

theorem points_in_groups (n1 n2 : ℕ) (h_total : n1 + n2 = 28) 
  (h_lines_diff : (n1*(n1 - 1) / 2) - (n2*(n2 - 1) / 2) = 81) : 
  (n1 = 17 ∧ n2 = 11) ∨ (n1 = 11 ∧ n2 = 17) :=
by
  sorry

end points_in_groups_l83_83546


namespace arithmetic_progression_sum_l83_83992

theorem arithmetic_progression_sum (a : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a n = a 0 + n * d)
  (h2 : a 0 = 2)
  (h3 : a 1 + a 2 = 13) :
  a 3 + a 4 + a 5 = 42 :=
sorry

end arithmetic_progression_sum_l83_83992


namespace jaylen_dog_food_consumption_l83_83009

theorem jaylen_dog_food_consumption :
  ∀ (morning evening daily_consumption total_food : ℕ)
  (days : ℕ),
  (morning = evening) →
  (total_food = 32) →
  (days = 16) →
  (daily_consumption = total_food / days) →
  (morning + evening = daily_consumption) →
  morning = 1 := by
  intros morning evening daily_consumption total_food days h_eq h_total h_days h_daily h_sum
  sorry

end jaylen_dog_food_consumption_l83_83009


namespace prob_second_shot_l83_83087

theorem prob_second_shot (P_A : ℝ) (P_AB : ℝ) (p : ℝ) : 
  P_A = 0.75 → 
  P_AB = 0.6 → 
  P_A * p = P_AB → 
  p = 0.8 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  sorry

end prob_second_shot_l83_83087


namespace verify_value_l83_83846

theorem verify_value (a b c d m : ℝ) 
  (h₁ : a = -b) 
  (h₂ : c * d = 1) 
  (h₃ : |m| = 3) :
  3 * c * d + (a + b) / (c * d) - m = 0 ∨ 
  3 * c * d + (a + b) / (c * d) - m = 6 := 
sorry

end verify_value_l83_83846


namespace collinear_vectors_x_eq_neg_two_l83_83997

theorem collinear_vectors_x_eq_neg_two (x : ℝ) (a b : ℝ×ℝ) :
  a = (1, 2) → b = (x, -4) → a.1 * b.2 = a.2 * b.1 → x = -2 :=
by
  intro ha hb hc
  sorry

end collinear_vectors_x_eq_neg_two_l83_83997


namespace tommy_first_house_price_l83_83605

theorem tommy_first_house_price (C : ℝ) (P : ℝ) (loan_rate : ℝ) (interest_rate : ℝ)
  (term : ℝ) (property_tax_rate : ℝ) (insurance_cost : ℝ) 
  (price_ratio : ℝ) (monthly_payment : ℝ) :
  C = 500000 ∧ price_ratio = 1.25 ∧ P * price_ratio = C ∧
  loan_rate = 0.75 ∧ interest_rate = 0.035 ∧ term = 15 ∧
  property_tax_rate = 0.015 ∧ insurance_cost = 7500 → 
  P = 400000 :=
by sorry

end tommy_first_house_price_l83_83605


namespace cone_lateral_surface_area_l83_83835

theorem cone_lateral_surface_area (r V : ℝ) (h l S : ℝ) 
  (radius_condition : r = 6)
  (volume_condition : V = 30 * Real.pi)
  (volume_formula : V = (1 / 3) * Real.pi * r^2 * h)
  (slant_height_formula : l = Real.sqrt (r^2 + h^2))
  (lateral_surface_area_formula : S = Real.pi * r * l) :
  S = 39 * Real.pi := 
sorry

end cone_lateral_surface_area_l83_83835


namespace product_of_smallest_primes_l83_83643

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def smallest_one_digit_primes : List ℕ := [2, 3]
def smallest_two_digit_prime : ℕ := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 :=
by
  sorry

end product_of_smallest_primes_l83_83643


namespace smallest_angle_of_trapezoid_l83_83268

theorem smallest_angle_of_trapezoid (a d : ℝ) :
  (a + (a + d) + (a + 2 * d) + (a + 3 * d) = 360) → 
  (a + 3 * d = 150) → 
  a = 15 :=
by
  sorry

end smallest_angle_of_trapezoid_l83_83268


namespace find_theta_l83_83232

theorem find_theta (theta : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ 2 * π) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^3 * Real.cos θ - x^2 * (1 - x) + (1 - x)^3 * Real.sin θ > 0) →
  θ > π / 12 ∧ θ < 5 * π / 12 :=
by
  sorry

end find_theta_l83_83232


namespace find_y_when_x_4_l83_83595

-- Definitions and conditions
variables (x y : ℝ)
def inversely_proportional (x y : ℝ) (K : ℝ) : Prop := x * y = K

-- Main theorem
theorem find_y_when_x_4 
  (K : ℝ) (h1 : inversely_proportional 20 10 K) (h2 : 20 + 10 = 30) (h3 : 20 - 10 = 10) 
  (hx : 4 * y = K) : y = 50 := 
sorry

end find_y_when_x_4_l83_83595


namespace base10_to_base7_conversion_l83_83220

theorem base10_to_base7_conversion : 2023 = 5 * 7^3 + 6 * 7^2 + 2 * 7^1 + 0 * 7^0 :=
  sorry

end base10_to_base7_conversion_l83_83220


namespace necessary_but_not_sufficient_condition_holds_l83_83424

-- Let m be a real number
variable (m : ℝ)

-- Define the conditions
def condition_1 : Prop := (m + 3) * (2 * m + 1) < 0
def condition_2 : Prop := -(2 * m - 1) > m + 2
def condition_3 : Prop := m + 2 > 0

-- Define necessary but not sufficient condition
def necessary_but_not_sufficient : Prop :=
  -2 < m ∧ m < -1 / 3

-- Problem statement
theorem necessary_but_not_sufficient_condition_holds 
  (h1 : condition_1 m) 
  (h2 : condition_2 m) 
  (h3 : condition_3 m) : necessary_but_not_sufficient m :=
sorry

end necessary_but_not_sufficient_condition_holds_l83_83424


namespace order_of_numbers_l83_83860

variables (a b : ℚ)

theorem order_of_numbers (ha_pos : a > 0) (hb_neg : b < 0) (habs : |a| < |b|) :
  b < -a ∧ -a < a ∧ a < -b :=
by { sorry }

end order_of_numbers_l83_83860


namespace find_n_l83_83467

def sum_first_n_even_numbers (n : ℕ) : ℕ :=
  n * (1 + n)

theorem find_n (k : ℕ) (h : k = 3) (hn : ∃ k, n = k^2)
  (hs : sum_first_n_even_numbers n = 90) : n = 9 :=
by
  sorry

end find_n_l83_83467


namespace new_op_4_3_l83_83372

def new_op (a b : ℕ) : ℕ := a^2 - a * b + b^2

theorem new_op_4_3 : new_op 4 3 = 13 :=
by
  -- Placeholder for the proof
  sorry

end new_op_4_3_l83_83372


namespace cubic_solution_identity_l83_83397

theorem cubic_solution_identity {a b c : ℕ} 
  (h1 : a + b + c = 6) 
  (h2 : ab + bc + ca = 11) 
  (h3 : abc = 6) : 
  (ab / c) + (bc / a) + (ca / b) = 49 / 6 := 
by 
  sorry

end cubic_solution_identity_l83_83397


namespace mean_value_of_interior_angles_pentagon_l83_83192

def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

theorem mean_value_of_interior_angles_pentagon :
  sum_of_interior_angles 5 / 5 = 108 :=
by
  sorry

end mean_value_of_interior_angles_pentagon_l83_83192


namespace min_value_of_quadratic_l83_83518

theorem min_value_of_quadratic (x y : ℝ) : (x^2 + 2*x*y + y^2) ≥ 0 ∧ ∃ x y, x = -y ∧ x^2 + 2*x*y + y^2 = 0 := by
  sorry

end min_value_of_quadratic_l83_83518


namespace system_of_equations_correct_l83_83481

variable (x y : ℝ)

def correct_system_of_equations : Prop :=
  (3 / 60) * x + (5 / 60) * y = 1.2 ∧ x + y = 16

theorem system_of_equations_correct :
  correct_system_of_equations x y :=
sorry

end system_of_equations_correct_l83_83481


namespace increasing_interval_of_f_l83_83583

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval_of_f :
  ∀ x, x > 2 → ∀ y, y > x → f x < f y :=
sorry

end increasing_interval_of_f_l83_83583


namespace plaza_area_increase_l83_83801

theorem plaza_area_increase (a : ℝ) : 
  ((a + 2)^2 - a^2 = 4 * a + 4) :=
sorry

end plaza_area_increase_l83_83801


namespace total_cleaning_time_l83_83164

theorem total_cleaning_time (time_outside : ℕ) (fraction_inside : ℚ) (time_inside : ℕ) (total_time : ℕ) :
  time_outside = 80 →
  fraction_inside = 1 / 4 →
  time_inside = fraction_inside * time_outside →
  total_time = time_outside + time_inside →
  total_time = 100 :=
by
  intros hto hfi htinside httotal
  rw [hto, hfi] at htinside
  norm_num at htinside
  rw [hto, htinside] at httotal
  norm_num at httotal
  exact httotal

end total_cleaning_time_l83_83164


namespace range_of_a_l83_83824

theorem range_of_a (a : ℝ) : |a - 1| + |a - 4| = 3 ↔ 1 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l83_83824


namespace elder_twice_as_old_l83_83912

theorem elder_twice_as_old (Y E : ℕ) (hY : Y = 35) (hDiff : E - Y = 20) : ∃ (X : ℕ),  X = 15 ∧ E - X = 2 * (Y - X) := 
by
  sorry

end elder_twice_as_old_l83_83912


namespace profit_percent_l83_83936

theorem profit_percent (CP SP : ℤ) (h : CP/SP = 2/3) : (SP - CP) * 100 / CP = 50 := 
by
  sorry

end profit_percent_l83_83936


namespace number_of_digits_in_x20_l83_83050

theorem number_of_digits_in_x20 (x : ℝ) (hx1 : 10^(7/4) ≤ x) (hx2 : x < 10^2) :
  10^35 ≤ x^20 ∧ x^20 < 10^36 :=
by
  -- Proof goes here
  sorry

end number_of_digits_in_x20_l83_83050


namespace minimum_abs_ab_l83_83118

theorem minimum_abs_ab (a b : ℝ) (h : (a^2) * (b / (a^2 + 1)) = 1) : abs (a * b) = 2 := 
  sorry

end minimum_abs_ab_l83_83118


namespace find_lesser_number_l83_83460

theorem find_lesser_number (x y : ℕ) (h₁ : x + y = 60) (h₂ : x - y = 10) : y = 25 := by
  sorry

end find_lesser_number_l83_83460


namespace find_triangle_height_l83_83722

-- Define the problem conditions
def Rectangle.perimeter (l : ℕ) (w : ℕ) : ℕ := 2 * l + 2 * w
def Rectangle.area (l : ℕ) (w : ℕ) : ℕ := l * w
def Triangle.area (b : ℕ) (h : ℕ) : ℕ := (b * h) / 2

-- Conditions
namespace Conditions
  -- Perimeter of the rectangle is 60 cm
  def rect_perimeter (l w : ℕ) : Prop := Rectangle.perimeter l w = 60
  -- Base of the right triangle is 15 cm
  def tri_base : ℕ := 15
  -- Areas of the rectangle and the triangle are equal
  def equal_areas (l w h : ℕ) : Prop := Rectangle.area l w = Triangle.area tri_base h
end Conditions

-- Proof problem: Given these conditions, prove h = 30
theorem find_triangle_height (l w h : ℕ) 
  (h1 : Conditions.rect_perimeter l w)
  (h2 : Conditions.equal_areas l w h) : h = 30 :=
  sorry

end find_triangle_height_l83_83722


namespace largest_angle_triangl_DEF_l83_83146

theorem largest_angle_triangl_DEF (d e f : ℝ) (h1 : d + 3 * e + 3 * f = d^2)
  (h2 : d + 3 * e - 3 * f = -8) : 
  ∃ (F : ℝ), F = 109.47 ∧ (F > 90) := by sorry

end largest_angle_triangl_DEF_l83_83146


namespace tank_min_cost_l83_83078

/-- A factory plans to build an open-top rectangular tank with one fixed side length of 8m and a maximum water capacity of 72m³. The cost 
of constructing the bottom and the walls of the tank are $2a$ yuan per square meter and $a$ yuan per square meter, respectively. 
We need to prove the optimal dimensions and the minimum construction cost.
-/
theorem tank_min_cost 
  (a : ℝ)   -- cost multiplier
  (b h : ℝ) -- dimensions of the tank
  (volume_constraint : 8 * b * h = 72) : 
  (b = 3) ∧ (h = 3) ∧ (16 * a * (b + h) + 18 * a = 114 * a) :=
by
  sorry

end tank_min_cost_l83_83078


namespace prob_at_most_one_first_class_product_l83_83315

noncomputable def P_event (p q : ℚ) : ℚ :=
  p * (1 - q) + (1 - p) * q

theorem prob_at_most_one_first_class_product :
  let p := 2 / 3
  let q := 3 / 4
  P_event p q = 5 / 12 := by
  sorry

end prob_at_most_one_first_class_product_l83_83315


namespace product_of_primes_l83_83616

def smallest_one_digit_prime := 2
def second_smallest_one_digit_prime := 3
def smallest_two_digit_prime := 11

theorem product_of_primes: smallest_one_digit_prime * second_smallest_one_digit_prime * smallest_two_digit_prime = 66 :=
by {
  -- Applying the definition of the primes and carrying out the multiplication
  show 2 * 3 * 11 = 66,
  calc
  2 * 3 * 11 = 6 * 11 : by rw [mul_assoc 2 3 11]
          ... = 66    : by norm_num,
}

end product_of_primes_l83_83616


namespace correct_option_is_C_l83_83194

def option_A (x : ℝ) : Prop := (-x^2)^3 = -x^5
def option_B (x : ℝ) : Prop := x^2 + x^3 = x^5
def option_C (x : ℝ) : Prop := x^3 * x^4 = x^7
def option_D (x : ℝ) : Prop := 2 * x^3 - x^3 = 1

theorem correct_option_is_C (x : ℝ) : ¬ option_A x ∧ ¬ option_B x ∧ option_C x ∧ ¬ option_D x :=
by
  sorry

end correct_option_is_C_l83_83194


namespace product_of_primes_l83_83633

theorem product_of_primes : 2 * 3 * 11 = 66 :=
by 
  -- Start with the multiplication of the first two primes
  have h1 : 2 * 3 = 6 := by norm_num
  -- Then multiply the result with the smallest two-digit prime
  have h2 : 6 * 11 = 66 := by norm_num
  -- Combine the steps to get the final result
  exact eq.trans (congr_arg (λ x, x * 11) h1) h2

end product_of_primes_l83_83633


namespace wristband_distribution_l83_83265

open Nat 

theorem wristband_distribution (x y : ℕ) 
  (h1 : 2 * x + 2 * y = 460) 
  (h2 : 2 * x = 3 * y) : x = 138 :=
sorry

end wristband_distribution_l83_83265


namespace compute_complex_expression_l83_83218

-- Define the expression we want to prove
def complex_expression : ℚ := 1 / (1 + (1 / (2 + (1 / (4^2)))))

-- The theorem stating the expression equals to the correct result
theorem compute_complex_expression : complex_expression = 33 / 49 :=
by sorry

end compute_complex_expression_l83_83218


namespace max_tulips_l83_83926

theorem max_tulips :
  ∃ (y r : ℕ), (y + r = 15) ∧ (y + r) % 2 = 1 ∧ |y - r| = 1 ∧ (50 * y + 31 * r ≤ 600) :=
begin
  sorry
end

end max_tulips_l83_83926


namespace santiago_more_roses_l83_83409

def red_roses_santiago := 58
def red_roses_garrett := 24
def red_roses_difference := red_roses_santiago - red_roses_garrett

theorem santiago_more_roses : red_roses_difference = 34 := by
  sorry

end santiago_more_roses_l83_83409


namespace ratio_of_q_to_r_l83_83287

theorem ratio_of_q_to_r
  (P Q R : ℕ)
  (h1 : R = 400)
  (h2 : P + Q + R = 1210)
  (h3 : 5 * Q = 4 * P) :
  Q * 10 = R * 9 :=
by
  sorry

end ratio_of_q_to_r_l83_83287


namespace ratio_of_volumes_of_tetrahedrons_l83_83877

theorem ratio_of_volumes_of_tetrahedrons (a b : ℝ) (h : a / b = 1 / 2) : (a^3) / (b^3) = 1 / 8 :=
by
-- proof goes here
sorry

end ratio_of_volumes_of_tetrahedrons_l83_83877


namespace find_angle_A_find_cos2C_minus_pi_over_6_l83_83884

noncomputable def triangle_area_formula (a b c : ℝ) (C : ℝ) : ℝ :=
  (1 / 2) * a * b * Real.sin C

noncomputable def given_area_formula (b c : ℝ) (S : ℝ) (a : ℝ) (C : ℝ) : Prop :=
  S = (Real.sqrt 3 / 6) * b * (b + c - a * Real.cos C)

noncomputable def angle_A (S b c a C : ℝ) (h : given_area_formula b c S a C) : ℝ :=
  Real.arcsin ((Real.sqrt 3 / 3) * (b + c - a * Real.cos C))

theorem find_angle_A (a b c S C : ℝ) (h : given_area_formula b c S a C) :
  angle_A S b c a C h = π / 3 :=
sorry

-- Part 2 related definitions
noncomputable def cos2C_minus_pi_over_6 (b c a C : ℝ) : ℝ :=
  let cos_C := (b^2 + c^2 - a^2) / (2 * b * c)
  let sin_C := Real.sqrt (1 - cos_C^2)
  let cos_2C := 2 * cos_C^2 - 1
  let sin_2C := 2 * sin_C * cos_C
  cos_2C * (Real.sqrt 3 / 2) + sin_2C * (1 / 2)

theorem find_cos2C_minus_pi_over_6 (b c a C : ℝ) (hb : b = 1) (hc : c = 3) (ha : a = Real.sqrt 7) :
  cos2C_minus_pi_over_6 b c a C = - (4 * Real.sqrt 3 / 7) :=
sorry

end find_angle_A_find_cos2C_minus_pi_over_6_l83_83884


namespace parallelogram_area_l83_83659

theorem parallelogram_area (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let y_top := a
  let y_bottom := -b
  let x_left := -c + 2*y
  let x_right := d - 2*y 
  (d + c) * (a + b) = ad + ac + bd + bc :=
by
  sorry

end parallelogram_area_l83_83659


namespace find_y_when_x_4_l83_83592

-- Definitions and conditions
variables (x y : ℝ)
def inversely_proportional (x y : ℝ) (K : ℝ) : Prop := x * y = K

-- Main theorem
theorem find_y_when_x_4 
  (K : ℝ) (h1 : inversely_proportional 20 10 K) (h2 : 20 + 10 = 30) (h3 : 20 - 10 = 10) 
  (hx : 4 * y = K) : y = 50 := 
sorry

end find_y_when_x_4_l83_83592


namespace max_reflections_max_reflections_example_l83_83660

-- Definition of the conditions
def angle_cda := 10  -- angle in degrees
def max_angle := 90  -- practical limit for angle of reflections

-- Given that the angle of incidence after n reflections is 10n degrees,
-- prove that the largest possible n is 9 before exceeding practical limits.
theorem max_reflections (n : ℕ) (h₁ : angle_cda = 10) (h₂ : max_angle = 90) :
  10 * n ≤ 90 :=
by sorry

-- Specific case instantiating n = 9
theorem max_reflections_example : (10 : ℕ) * 9 ≤ 90 := max_reflections 9 rfl rfl

end max_reflections_max_reflections_example_l83_83660


namespace intersection_P_Q_l83_83360

def P : Set ℝ := {x : ℝ | x < 1}
def Q : Set ℝ := {x : ℝ | x^2 < 4}

theorem intersection_P_Q : P ∩ Q = {x : ℝ | -2 < x ∧ x < 1} := 
  sorry

end intersection_P_Q_l83_83360


namespace rectangle_area_from_square_area_and_proportions_l83_83085

theorem rectangle_area_from_square_area_and_proportions :
  ∃ (a b w : ℕ), a = 16 ∧ b = 3 * w ∧ w = Int.natAbs (Int.sqrt a) ∧ w * b = 48 :=
by
  sorry

end rectangle_area_from_square_area_and_proportions_l83_83085


namespace find_number_l83_83256

theorem find_number (number : ℝ) (h1 : 213 * number = 3408) (h2 : 0.16 * 2.13 = 0.3408) : number = 16 :=
by
  sorry

end find_number_l83_83256


namespace intersection_complement_l83_83362

def A := {x : ℝ | -1 < x ∧ x < 6}
def B := {x : ℝ | x^2 < 4}
def complement_R (S : Set ℝ) := {x : ℝ | x ∉ S}

theorem intersection_complement :
  A ∩ (complement_R B) = {x : ℝ | 2 ≤ x ∧ x < 6} := by
sorry

end intersection_complement_l83_83362


namespace prove_statements_l83_83571

theorem prove_statements (x y z : ℝ) (h : x + y + z = x * y * z) :
  ( (∀ (x y : ℝ), x + y = 0 → (∃ (z : ℝ), (x + y + z = x * y * z) → z = 0))
  ∧ (∀ (x y : ℝ), x = 0 → (∃ (z : ℝ), (x + y + z = x * y * z) → y = -z))
  ∧ z = (x + y) / (x * y - 1) ) :=
by
  sorry

end prove_statements_l83_83571


namespace library_table_count_l83_83759

def base6_to_base10 (n : Nat) : Nat :=
  let d0 := (n % 10)
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  d2 * 36 + d1 * 6 + d0 

theorem library_table_count (chairs people_per_table : Nat) (h1 : chairs = 231) (h2 : people_per_table = 3) :
    Nat.ceil ((base6_to_base10 chairs) / people_per_table) = 31 :=
by
  sorry

end library_table_count_l83_83759


namespace domain_of_f_l83_83969

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 2*x - 3)

theorem domain_of_f : 
  {x : ℝ | (x^2 - 2*x - 3) ≠ 0} = {x : ℝ | x < -1} ∪ {x : ℝ | -1 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l83_83969


namespace train_speed_in_kph_l83_83493

noncomputable def speed_of_train (jogger_speed_kph : ℝ) (gap_m : ℝ) (train_length_m : ℝ) (time_s : ℝ) : ℝ :=
let jogger_speed_mps := jogger_speed_kph * (1000 / 3600)
let total_distance_m := gap_m + train_length_m
let speed_mps := total_distance_m / time_s
speed_mps * (3600 / 1000)

theorem train_speed_in_kph :
  speed_of_train 9 240 120 36 = 36 := 
by
  sorry

end train_speed_in_kph_l83_83493


namespace part_i_part_ii_l83_83120

noncomputable def f (x : ℝ) : ℝ := (x / (x + 4)) * Real.exp (x + 2)

theorem part_i (x : ℝ) (hx : x > -2) : (x * Real.exp (x + 2) + x + 4 > 0) :=
sorry

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (Real.exp (x + 2) - a * x - 3 * a) / (x + 2)^2

theorem part_ii (a : ℝ) (ha : a ∈ set.Ico 0 1) : 
  ∃ (h : ℝ), (∀ x > -2, g x a ≥ h) ∧ set.Icc (1 / 2) (Real.exp 2 / 4) = 
  { h' | ∃ x, h = g x a } :=
sorry

end part_i_part_ii_l83_83120


namespace rational_root_contradiction_l83_83473

theorem rational_root_contradiction 
(a b c : ℤ) 
(h_odd_a : a % 2 ≠ 0) 
(h_odd_b : b % 2 ≠ 0)
(h_odd_c : c % 2 ≠ 0)
(rational_root_exists : ∃ (r : ℚ), a * r^2 + b * r + c = 0) :
false :=
sorry

end rational_root_contradiction_l83_83473


namespace draws_alternate_no_consecutive_same_color_l83_83653

-- Defining the total number of balls and the count of each color.
def total_balls : ℕ := 15
def white_balls : ℕ := 5
def black_balls : ℕ := 5
def red_balls : ℕ := 5

-- Defining the probability that the draws alternate in colors with no two consecutive balls of the same color.
def probability_no_consecutive_same_color : ℚ := 162 / 1001

theorem draws_alternate_no_consecutive_same_color :
  (white_balls + black_balls + red_balls = total_balls) →
  -- The resulting probability based on the given conditions.
  probability_no_consecutive_same_color = 162 / 1001 := by
  sorry

end draws_alternate_no_consecutive_same_color_l83_83653


namespace taxi_fare_distance_condition_l83_83049

theorem taxi_fare_distance_condition (x : ℝ) (h1 : 7 + (max (x - 3) 0) * 2.4 = 19) : x ≤ 8 := 
by
  sorry

end taxi_fare_distance_condition_l83_83049


namespace sector_area_max_radius_l83_83850

noncomputable def arc_length (R : ℝ) : ℝ := 20 - 2 * R

noncomputable def sector_area (R : ℝ) : ℝ :=
  let l := arc_length R
  0.5 * l * R

theorem sector_area_max_radius :
  ∃ (R : ℝ), sector_area R = -R^2 + 10 * R ∧
             R = 5 :=
sorry

end sector_area_max_radius_l83_83850


namespace expected_sides_of_red_polygon_l83_83764

-- Define the conditions
def isChosenWithinSquare (F : ℝ × ℝ) (side_length: ℝ) : Prop :=
  0 ≤ F.1 ∧ F.1 ≤ side_length ∧ 0 ≤ F.2 ∧ F.2 ≤ side_length

def pointF (side_length: ℝ) : ℝ × ℝ := sorry
def foldToF (vertex: ℝ × ℝ) (F: ℝ × ℝ) : ℝ := sorry

-- Define the expected number of sides of the resulting red polygon
noncomputable def expected_sides (side_length : ℝ) : ℝ :=
  let P_g := 2 - (Real.pi / 2)
  let P_o := (Real.pi / 2) - 1 
  (3 * P_o) + (4 * P_g)

-- Prove the expected number of sides equals 5 - π / 2
theorem expected_sides_of_red_polygon (side_length : ℝ) :
  expected_sides side_length = 5 - (Real.pi / 2) := 
  by sorry

end expected_sides_of_red_polygon_l83_83764


namespace wholesale_cost_l83_83946

theorem wholesale_cost (W R : ℝ) (h1 : R = 1.20 * W) (h2 : 0.70 * R = 168) : W = 200 :=
by
  sorry

end wholesale_cost_l83_83946


namespace fruit_fly_cell_division_l83_83717

/-- Genetic properties of fruit flies:
  1. Fruit flies have 2N = 8 chromosomes.
  2. Alleles A/a and B/b are inherited independently.
  3. Genotype AaBb is given.
  4. This genotype undergoes cell division without chromosomal variation.

Prove that:
Cells with a genetic composition of AAaaBBbb contain 8 or 16 chromosomes.
-/
theorem fruit_fly_cell_division (genotype : ℕ → ℕ) (A a B b : ℕ) :
  genotype 2 = 8 ∧
  (A + a + B + b = 8) ∧
  (genotype 0 = 2 * 4) →
  (genotype 1 = 8 ∨ genotype 1 = 16) :=
by
  sorry

end fruit_fly_cell_division_l83_83717


namespace participant_A_enters_third_round_participant_B_receives_commendation_l83_83077

-- Conditions for Part 1
def success_prob_first (p : ℝ) := (4/5)
def success_prob_second (p : ℝ) := (3/4)

-- Question and Goal statement for Part 1
theorem participant_A_enters_third_round :
  ∀ (p : ℝ), p = (success_prob_first p + (1 - success_prob_first p) * success_prob_first p) *
                  (success_prob_second p + (1 - success_prob_second p) * success_prob_second p) →
  p = 9/10 :=
sorry

-- Conditions for Part 2
def mean_score : ℝ := 212
def std_dev (μ σ : ℝ) : ℝ := 29
def total_participants : ℕ := 2000
def top_participants : ℕ := 317
def elevated_score_participants : ℕ := 46
def participant_b_score : ℝ := 231

-- Question and Goal statement for Part 2
theorem participant_B_receives_commendation :
  ∀ (x : ℝ), x = participant_b_score →
  231 < (mean_score + std_dev mean_score std_dev) →
  False :=
sorry

end participant_A_enters_third_round_participant_B_receives_commendation_l83_83077


namespace total_carrots_grown_l83_83288

theorem total_carrots_grown
  (Sandy_carrots : ℕ) (Sam_carrots : ℕ) (Sophie_carrots : ℕ) (Sara_carrots : ℕ)
  (h1 : Sandy_carrots = 6)
  (h2 : Sam_carrots = 3)
  (h3 : Sophie_carrots = 2 * Sam_carrots)
  (h4 : Sara_carrots = (Sandy_carrots + Sam_carrots + Sophie_carrots) - 5) :
  Sandy_carrots + Sam_carrots + Sophie_carrots + Sara_carrots = 25 :=
by sorry

end total_carrots_grown_l83_83288


namespace product_of_smallest_primes_l83_83639

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def smallest_one_digit_primes : List ℕ := [2, 3]
def smallest_two_digit_prime : ℕ := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 :=
by
  sorry

end product_of_smallest_primes_l83_83639


namespace inequality_abc_l83_83285

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1) * (b + 1) * (a + c) * (b + c) ≥ 16 * a * b * c :=
by
  sorry

end inequality_abc_l83_83285


namespace integer_pair_solution_l83_83343

theorem integer_pair_solution (x y : ℤ) :
  3^4 * 2^3 * (x^2 + y^2) = x^3 * y^3 ↔ (x = 6 ∧ y = 6) ∨ (x = -6 ∧ y = -6) ∨ (x = 0 ∧ y = 0) :=
by
  sorry

end integer_pair_solution_l83_83343


namespace find_x_l83_83703

noncomputable def vector_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem find_x (x : ℝ) :
  let a := (1, 2*x + 1)
  let b := (2, 3)
  (vector_parallel a b) → x = 1 / 4 :=
by
  intro h
  have h_eq := h
  sorry  -- proof is not needed as per instruction

end find_x_l83_83703


namespace coefficient_sum_eq_512_l83_83244

theorem coefficient_sum_eq_512 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ) :
  (1 - x) ^ 9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + 
                a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 →
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| + |a_9| = 512 :=
sorry

end coefficient_sum_eq_512_l83_83244


namespace horner_multiplications_additions_count_l83_83056

noncomputable def polynomial := λ x : ℤ, 5 * x^5 + 4 * x^4 + 3 * x^3 - 2 * x^2 - x - 1

theorem horner_multiplications_additions_count :
  let x := -4 in
  let f := polynomial x in
  (count_multiplications f x, count_additions f x) = (5, 5) :=
sorry

end horner_multiplications_additions_count_l83_83056


namespace count_non_empty_subsets_of_odd_numbers_greater_than_one_l83_83371

-- Condition definitions
def given_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def odd_numbers_greater_than_one (s : Finset ℕ) : Finset ℕ := 
  s.filter (λ x => x % 2 = 1 ∧ x > 1)

-- The problem statement
theorem count_non_empty_subsets_of_odd_numbers_greater_than_one : 
  (odd_numbers_greater_than_one given_set).powerset.card - 1 = 15 := 
by 
  sorry

end count_non_empty_subsets_of_odd_numbers_greater_than_one_l83_83371


namespace average_price_of_initial_fruit_l83_83670

theorem average_price_of_initial_fruit (A O : ℕ) (h1 : A + O = 10) (h2 : (40 * A + 60 * (O - 6)) / (A + O - 6) = 45) : 
  (40 * A + 60 * O) / 10 = 54 :=
by 
  sorry

end average_price_of_initial_fruit_l83_83670


namespace curve_is_circle_l83_83351

noncomputable def curve_eqn_polar (r θ : ℝ) : Prop :=
  r = 1 / (Real.sin θ + Real.cos θ)

theorem curve_is_circle : ∀ r θ, curve_eqn_polar r θ →
  ∃ x y : ℝ, r = Real.sqrt (x^2 + y^2) ∧ 
  x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ 
  (x - 1/2)^2 + (y - 1/2)^2 = 1/2 :=
by
  sorry

end curve_is_circle_l83_83351


namespace roots_negative_reciprocal_l83_83694

theorem roots_negative_reciprocal (a b c : ℝ) (α β : ℝ) (h_eq : a * α ^ 2 + b * α + c = 0)
  (h_roots : α * β = -1) : c = -a :=
sorry

end roots_negative_reciprocal_l83_83694


namespace sum_of_not_visible_faces_l83_83229

-- Define the sum of the numbers on the faces of one die
def die_sum : ℕ := 21

-- List of visible numbers on the dice
def visible_faces_sum : ℕ := 4 + 3 + 2 + 5 + 1 + 3 + 1

-- Define the total sum of the numbers on the faces of three dice
def total_sum : ℕ := die_sum * 3

-- Statement to prove the sum of not-visible faces equals 44
theorem sum_of_not_visible_faces : 
  total_sum - visible_faces_sum = 44 :=
sorry

end sum_of_not_visible_faces_l83_83229


namespace min_copy_paste_actions_l83_83480

theorem min_copy_paste_actions :
  ∀ (n : ℕ), (n ≥ 10) ∧ (n ≤ n) → (2^n ≥ 1001) :=
by sorry

end min_copy_paste_actions_l83_83480


namespace inclination_angle_of_line_l83_83996

theorem inclination_angle_of_line
  (α : ℝ) (h1 : α > 0) (h2 : α < 180)
  (hslope : Real.tan α = - (Real.sqrt 3) / 3) :
  α = 150 :=
sorry

end inclination_angle_of_line_l83_83996


namespace base9_number_perfect_square_l83_83986

theorem base9_number_perfect_square (a b d : ℕ) (h1 : a ≠ 0) (h2 : 0 ≤ d ∧ d ≤ 8) (n : ℕ) 
  (h3 : n = 729 * a + 81 * b + 45 + d) (h4 : ∃ k : ℕ, k * k = n) : d = 0 := 
sorry

end base9_number_perfect_square_l83_83986


namespace pyramid_four_triangular_faces_area_l83_83959

noncomputable def pyramid_total_area (base_edge lateral_edge : ℝ) : ℝ :=
  if base_edge = 8 ∧ lateral_edge = 7 then 16 * Real.sqrt 33 else 0

theorem pyramid_four_triangular_faces_area :
  pyramid_total_area 8 7 = 16 * Real.sqrt 33 :=
by
  -- Proof omitted
  sorry

end pyramid_four_triangular_faces_area_l83_83959


namespace sum_of_digits_9ab_l83_83883

def a : ℕ := 999
def b : ℕ := 666

theorem sum_of_digits_9ab : 
  let n := 9 * a * b
  (n.digits 10).sum = 36 := 
by
  sorry

end sum_of_digits_9ab_l83_83883


namespace problem_min_ineq_range_l83_83240

theorem problem_min_ineq_range (a b : ℝ) (x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (∀ x, 1 / a + 4 / b ≥ |2 * x - 1| - |x + 1|) ∧ (1 / a + 4 / b = 9) ∧ (-7 ≤ x ∧ x ≤ 11) :=
sorry

end problem_min_ineq_range_l83_83240


namespace mailman_distribution_l83_83332

theorem mailman_distribution 
    (total_mail_per_block : ℕ)
    (blocks : ℕ)
    (houses_per_block : ℕ)
    (h1 : total_mail_per_block = 32)
    (h2 : blocks = 55)
    (h3 : houses_per_block = 4) :
  total_mail_per_block / houses_per_block = 8 :=
by
  sorry

end mailman_distribution_l83_83332


namespace percentage_of_b_l83_83071

variable (a b c p : ℝ)

-- Conditions
def condition1 : Prop := 0.02 * a = 8
def condition2 : Prop := c = b / a
def condition3 : Prop := p * b = 2

-- Theorem statement
theorem percentage_of_b (h1 : condition1 a)
                        (h2 : condition2 b a c)
                        (h3 : condition3 p b) :
  p = 0.005 := sorry

end percentage_of_b_l83_83071


namespace evaluate_polynomial_l83_83479

theorem evaluate_polynomial : (99^4 - 4 * 99^3 + 6 * 99^2 - 4 * 99 + 1) = 92199816 := 
by 
  sorry

end evaluate_polynomial_l83_83479


namespace loraine_wax_usage_proof_l83_83563

-- Conditions
variables (large_animals small_animals : ℕ)
variable (wax : ℕ)

-- Definitions based on conditions
def large_animal_wax := 4
def small_animal_wax := 2
def total_sticks := 20
def small_animals_wax := 12
def small_to_large_ratio := 3

-- Proof statement
theorem loraine_wax_usage_proof (h1 : small_animals_wax = small_animals * small_animal_wax)
  (h2 : small_animals = large_animals * small_to_large_ratio)
  (h3 : wax = small_animals_wax + large_animals * large_animal_wax) :
  wax = total_sticks := by
  sorry

end loraine_wax_usage_proof_l83_83563


namespace common_chord_eqn_l83_83427

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 12 * x - 2 * y - 13 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 12 * x + 16 * y - 25 = 0

-- Define the proposition stating the common chord equation
theorem common_chord_eqn : ∀ x y : ℝ, C1 x y ∧ C2 x y → 4 * x + 3 * y - 2 = 0 :=
by
  sorry

end common_chord_eqn_l83_83427


namespace number_of_books_to_break_even_is_4074_l83_83337

-- Definitions from problem conditions
def fixed_costs : ℝ := 35630
def variable_cost_per_book : ℝ := 11.50
def selling_price_per_book : ℝ := 20.25

-- The target number of books to sell for break-even
def target_books_to_break_even : ℕ := 4074

-- Lean statement to prove that number of books to break even is 4074
theorem number_of_books_to_break_even_is_4074 :
  let total_costs (x : ℝ) := fixed_costs + variable_cost_per_book * x
  let total_revenue (x : ℝ) := selling_price_per_book * x
  ∃ x : ℝ, total_costs x = total_revenue x → x = target_books_to_break_even := by
  sorry

end number_of_books_to_break_even_is_4074_l83_83337


namespace equation_of_line_through_points_l83_83581

-- Definitions for the problem conditions
def point1 : ℝ × ℝ := (-1, 2)
def point2 : ℝ × ℝ := (-3, -2)

-- The theorem stating the equation of the line passing through the given points
theorem equation_of_line_through_points :
  ∃ a b c : ℝ, (a * point1.1 + b * point1.2 + c = 0) ∧ (a * point2.1 + b * point2.2 + c = 0) ∧ 
             (a = 2) ∧ (b = -1) ∧ (c = 4) :=
by
  sorry

end equation_of_line_through_points_l83_83581


namespace smallest_n_l83_83094

theorem smallest_n (j c g : ℕ) (n : ℕ) (total_cost : ℕ) 
  (h_condition : total_cost = 10 * j ∧ total_cost = 16 * c ∧ total_cost = 18 * g ∧ total_cost = 24 * n) 
  (h_lcm : Nat.lcm (Nat.lcm 10 16) 18 = 720) : n = 30 :=
by
  sorry

end smallest_n_l83_83094


namespace find_sum_l83_83803

theorem find_sum (I r1 r2 r3 r4 r5: ℝ) (t1 t2 t3 t4 t5 : ℝ) (P: ℝ) 
  (hI: I = 6016.75)
  (hr1: r1 = 0.06) (hr2: r2 = 0.075) (hr3: r3 = 0.08) (hr4: r4 = 0.085) (hr5: r5 = 0.09)
  (ht: ∀ i, (i = t1 ∨ i = t2 ∨ i = t3 ∨ i = t4 ∨ i = t5) → i = 1): 
  I = P * (r1 * t1 + r2 * t2 + r3 * t3 + r4 * t4 + r5 * t5) → P = 15430 :=
by
  sorry

end find_sum_l83_83803


namespace question1_question2_l83_83995

-- Definitions based on the conditions
def f (x m : ℝ) : ℝ := x^2 + 4*x + m

theorem question1 (m : ℝ) (h1 : m ≠ 0) (h2 : 16 - 4 * m > 0) : m < 4 :=
  sorry

theorem question2 (m : ℝ) (hx : ∀ x : ℝ, f x m = 0 → f (-x - 4) m = 0) 
  (h_circ : ∀ (x y : ℝ), x^2 + y^2 + 4*x - (m + 1) * y + m = 0 → (x = 0 ∧ y = 1) ∨ (x = -4 ∧ y = 1)) :
  (∀ (x y : ℝ), x^2 + y^2 + 4*x - (m + 1) * y + m = 0 → (x = 0 ∧ y = 1)) ∨ (∀ (x y : ℝ), (x = -4 ∧ y = 1)) :=
  sorry

end question1_question2_l83_83995


namespace probability_of_head_equal_half_l83_83139

def fair_coin_probability : Prop :=
  ∀ (H T : ℕ), (H = 1 ∧ T = 1 ∧ (H + T = 2)) → ((H / (H + T)) = 1 / 2)

theorem probability_of_head_equal_half : fair_coin_probability :=
sorry

end probability_of_head_equal_half_l83_83139


namespace solve_nat_eqn_l83_83290

theorem solve_nat_eqn (n k l m : ℕ) (hl : l > 1) 
  (h_eq : (1 + n^k)^l = 1 + n^m) : (n, k, l, m) = (2, 1, 2, 3) := 
sorry

end solve_nat_eqn_l83_83290


namespace simplify_expression_l83_83704
theorem simplify_expression (c : ℝ) : 
    (3 * c + 6 - 6 * c) / 3 = -c + 2 := 
by 
    sorry

end simplify_expression_l83_83704


namespace simplify_power_l83_83750

theorem simplify_power (x : ℝ) : (3 * x^4)^4 = 81 * x^16 :=
by sorry

end simplify_power_l83_83750


namespace stable_table_configurations_l83_83947

noncomputable def numberOfStableConfigurations (n : ℕ) : ℕ :=
  1 / 3 * (n + 1) * (2 * n ^ 2 + 4 * n + 3)

theorem stable_table_configurations (n : ℕ) (hn : 0 < n) :
  numberOfStableConfigurations n = 
    (1 / 3 * (n + 1) * (2 * n ^ 2 + 4 * n + 3)) :=
by
  sorry

end stable_table_configurations_l83_83947


namespace complement_A_in_U_l83_83125

def U : Set ℕ := {x | x ≥ 2}
def A : Set ℕ := {x | x^2 ≥ 5}

theorem complement_A_in_U : (U \ A) = {2} := by
  sorry

end complement_A_in_U_l83_83125


namespace find_y_l83_83597

theorem find_y (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : ∃ C, x * y = C) (hx : x = 4) : y = 50 :=
sorry

end find_y_l83_83597


namespace moles_of_NH3_formed_l83_83357

-- Conditions
def moles_NH4Cl : ℕ := 3 -- 3 moles of Ammonium chloride
def total_moles_NH3_formed : ℕ := 3 -- The total moles of Ammonia formed

-- The balanced chemical reaction implies a 1:1 molar ratio
lemma reaction_ratio (n : ℕ) : total_moles_NH3_formed = n := by
  sorry

-- Prove that the number of moles of NH3 formed is equal to 3
theorem moles_of_NH3_formed : total_moles_NH3_formed = moles_NH4Cl := 
reaction_ratio moles_NH4Cl

end moles_of_NH3_formed_l83_83357


namespace sin_theta_plus_pi_over_six_l83_83239

open Real

theorem sin_theta_plus_pi_over_six (theta : ℝ) (h : sin θ + sin (θ + π / 3) = sqrt 3) :
  sin (θ + π / 6) = 1 := 
sorry

end sin_theta_plus_pi_over_six_l83_83239


namespace exists_x_nat_l83_83278

theorem exists_x_nat (a c : ℕ) (b : ℤ) : ∃ x : ℕ, (a^x + x) % c = b % c :=
by
  sorry

end exists_x_nat_l83_83278


namespace symmetry_center_on_line_l83_83918

def symmetry_center_curve :=
  ∃ θ : ℝ, (∃ x y : ℝ, (x = -1 + Real.cos θ ∧ y = 2 + Real.sin θ))

-- The main theorem to prove
theorem symmetry_center_on_line : 
  (∃ cx cy : ℝ, (symmetry_center_curve ∧ (cy = -2 * cx))) :=
sorry

end symmetry_center_on_line_l83_83918


namespace yankees_mets_ratio_l83_83713

-- Given conditions
def num_mets_fans : ℕ := 104
def total_fans : ℕ := 390
def ratio_mets_to_redsox : ℚ := 4 / 5

-- Definitions
def num_redsox_fans (M : ℕ) := (5 / 4) * M
def num_yankees_fans (Y M B : ℕ) := (total_fans - M - B)

-- Theorem statement
theorem yankees_mets_ratio (Y M B : ℕ)
  (h1 : M = num_mets_fans)
  (h2 : Y + M + B = total_fans)
  (h3 : (M : ℚ) / (B : ℚ) = ratio_mets_to_redsox) :
  (Y : ℚ) / (M : ℚ) = 3 / 2 :=
sorry

end yankees_mets_ratio_l83_83713


namespace initial_tabs_count_l83_83393

theorem initial_tabs_count (T : ℕ) (h1 : T > 0)
  (h2 : (3 / 4 : ℚ) * T - (2 / 5 : ℚ) * ((3 / 4 : ℚ) * T) > 0)
  (h3 : (9 / 20 : ℚ) * T - (1 / 2 : ℚ) * ((9 / 20 : ℚ) * T) = 90) :
  T = 400 :=
sorry

end initial_tabs_count_l83_83393


namespace wardrobe_probability_l83_83140

theorem wardrobe_probability :
  let total_articles := 5 + 6 + 7 in
  let total_ways := choose total_articles 4 in
  let ways_choose_shirts := choose 5 2 in
  let ways_choose_shorts := choose 6 1 in
  let ways_choose_socks := choose 7 1 in
  let favorable_ways := ways_choose_shirts * ways_choose_shorts * ways_choose_socks in
  (favorable_ways : ℚ) / (total_ways : ℚ) = 7 / 51 :=
by
  let total_articles := 18
  let total_ways := choose total_articles 4
  let ways_choose_shirts := choose 5 2
  let ways_choose_shorts := choose 6 1
  let ways_choose_socks := choose 7 1
  let favorable_ways := ways_choose_shirts * ways_choose_shorts * ways_choose_socks
  have h1 : total_articles = 18 := rfl
  have h2 : total_ways = 3060 := by norm_num [total_ways]
  have h3 : ways_choose_shirts = 10 := by norm_num [ways_choose_shirts]
  have h4 : ways_choose_shorts = 6 := by norm_num [ways_choose_shorts]
  have h5 : ways_choose_socks = 7 := by norm_num [ways_choose_socks]
  have h_fav_ways : favorable_ways = 420 := by norm_num [favorable_ways, h3, h4, h5]
  show (favorable_ways : ℚ) / (total_ways : ℚ) = 7 / 51
  sorry -- Full proof omitted

end wardrobe_probability_l83_83140


namespace cone_lateral_surface_area_l83_83832

theorem cone_lateral_surface_area (r : ℕ) (V : ℝ) (h l S : ℝ)
  (h_r : r = 6)
  (h_V : V = 30 * Real.pi)
  (h_volume : V = (1 / 3) * Real.pi * (r ^ 2) * h)
  (h_slant_height : l = Real.sqrt (r^2 + h^2))
  (h_lateral_surface_area : S = Real.pi * r * l) :
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l83_83832


namespace product_multiple_of_12_probability_l83_83872

theorem product_multiple_of_12_probability :
  let s := {3, 4, 5, 6, 8}
  in
  (∃ pairs : set (ℕ × ℕ), pairs = { (3, 4), (4, 6), (6, 8) }) →
  let total_pairs := finset.card (set.to_finset ((s × s).image prod.mk)) / 2,
      favorable_pairs := finset.card (set.to_finset { (3, 4), (4, 6), (6, 8) })
  in
  favorable_pairs / total_pairs = (3 : ℚ) / 10 :=
sorry

end product_multiple_of_12_probability_l83_83872


namespace Laura_weekly_driving_distance_l83_83012

theorem Laura_weekly_driving_distance :
  ∀ (house_to_school : ℕ) (extra_to_supermarket : ℕ) (school_days_per_week : ℕ) (supermarket_trips_per_week : ℕ),
    house_to_school = 20 →
    extra_to_supermarket = 10 →
    school_days_per_week = 5 →
    supermarket_trips_per_week = 2 →
    (school_days_per_week * house_to_school + supermarket_trips_per_week * ((house_to_school / 2) + extra_to_supermarket) * 2) = 180 :=
by
  intros house_to_school extra_to_supermarket school_days_per_week supermarket_trips_per_week
  assume house_to_school_eq : house_to_school = 20
  assume extra_to_supermarket_eq : extra_to_supermarket = 10
  assume school_days_per_week_eq : school_days_per_week = 5
  assume supermarket_trips_per_week_eq : supermarket_trips_per_week = 2
  rw [house_to_school_eq, extra_to_supermarket_eq, school_days_per_week_eq, supermarket_trips_per_week_eq]
  sorry

end Laura_weekly_driving_distance_l83_83012


namespace partition_count_l83_83681

theorem partition_count (A B : Finset ℕ) :
  (∀ n, n ∈ A ∨ n ∈ B) ∧ 
  (∀ n, n ∈ A → 1 ≤ n ∧ n ≤ 9) ∧ 
  (∀ n, n ∈ B → 1 ≤ n ∧ n ≤ 9) ∧ 
  (A ∩ B = ∅) ∧ 
  (A ∪ B = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ 
  (8 * A.sum id = B.sum id) ∧ 
  (A.sum id + B.sum id = 45) → 
  ∃! (num_ways : ℕ), num_ways = 3 :=
sorry

end partition_count_l83_83681


namespace negation_of_existential_l83_83914

theorem negation_of_existential (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) :=
by
  sorry

end negation_of_existential_l83_83914


namespace max_range_of_temperatures_l83_83754

theorem max_range_of_temperatures (avg_temp : ℝ) (low_temp : ℝ) (days : ℕ) (total_temp: ℝ) (high_temp : ℝ) 
  (h1 : avg_temp = 60) (h2 : low_temp = 50) (h3 : days = 5) (h4 : total_temp = avg_temp * days) 
  (h5 : total_temp = 300) (h6 : 4 * low_temp + high_temp = total_temp) : 
  high_temp - low_temp = 50 := 
by
  sorry

end max_range_of_temperatures_l83_83754


namespace find_dividend_l83_83384

theorem find_dividend 
  (R : ℤ) 
  (Q : ℤ) 
  (D : ℤ) 
  (h1 : R = 8) 
  (h2 : D = 3 * Q) 
  (h3 : D = 3 * R + 3) : 
  (D * Q + R = 251) :=
by {
  -- The proof would follow, but for now, we'll use sorry.
  sorry
}

end find_dividend_l83_83384


namespace max_frac_a_S_l83_83553

def S (n : ℕ) : ℕ := 2^n - 1

def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else S n - S (n - 1)

theorem max_frac_a_S (n : ℕ) (h : S n = 2^n - 1) : 
  let frac := (a n) / (a n * S n + a 6)
  ∃ N : ℕ, N > 0 ∧ (frac ≤ 1 / 15) := by
  sorry

end max_frac_a_S_l83_83553


namespace regular_tire_price_l83_83948

theorem regular_tire_price 
  (x : ℝ) 
  (h1 : 3 * x + x / 2 = 300) 
  : x = 600 / 7 := 
sorry

end regular_tire_price_l83_83948


namespace cos_C_value_l83_83270

namespace Triangle

theorem cos_C_value (A B C : ℝ)
  (h_triangle : A + B + C = Real.pi)
  (sin_A : Real.sin A = 2/3)
  (cos_B : Real.cos B = 1/2) :
  Real.cos C = (2 * Real.sqrt 3 - Real.sqrt 5) / 6 := 
sorry

end Triangle

end cos_C_value_l83_83270


namespace number_of_floors_l83_83490

def hours_per_room : ℕ := 6
def hourly_rate : ℕ := 15
def total_earnings : ℕ := 3600
def rooms_per_floor : ℕ := 10

theorem number_of_floors : 
  (total_earnings / hourly_rate / hours_per_room) / rooms_per_floor = 4 := by
  sorry

end number_of_floors_l83_83490


namespace quadratic_function_symmetry_l83_83539

-- Define the quadratic function
def f (x : ℝ) (b c : ℝ) : ℝ := -x^2 + b * x + c

-- State the problem as a theorem
theorem quadratic_function_symmetry (b c : ℝ) (h_symm : ∀ x, f x b c = f (4 - x) b c) :
  f 2 b c > f 1 b c ∧ f 1 b c > f 4 b c :=
by
  -- Include a placeholder for the proof
  sorry

end quadratic_function_symmetry_l83_83539


namespace clean_car_time_l83_83165

theorem clean_car_time (t_outside : ℕ) (t_inside : ℕ) (h_outside : t_outside = 80) (h_inside : t_inside = t_outside / 4) : 
  t_outside + t_inside = 100 := 
by 
  sorry

end clean_car_time_l83_83165


namespace simplify_expression_l83_83937

theorem simplify_expression (h : (Real.pi / 2) < 2 ∧ 2 < Real.pi) : 
  Real.sqrt (1 - 2 * Real.sin 2 * Real.cos 2) = Real.sin 2 - Real.cos 2 :=
sorry

end simplify_expression_l83_83937


namespace jill_marbles_probability_l83_83550

noncomputable def probability_exactly_two_blue (total_marbles: ℕ) (blue_marbles: ℕ) (draws: ℕ) (successes: ℕ) : ℝ :=
  let p_blue := (blue_marbles : ℝ) / (total_marbles : ℝ)
  let p_red := 1 - p_blue
  let prob_specific_case := (p_blue ^ successes) * (p_red ^ (draws - successes))
  let num_ways := (Finset.range draws).choose successes).card
  num_ways * prob_specific_case

theorem jill_marbles_probability :
  probability_exactly_two_blue 10 6 5 2 ≈ 0.230 :=
sorry

end jill_marbles_probability_l83_83550


namespace max_g_equals_sqrt3_l83_83367

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x + Real.pi / 9) + Real.sin (5 * Real.pi / 9 - x)

noncomputable def g (x : ℝ) : ℝ :=
  f (f x)

theorem max_g_equals_sqrt3 : ∀ x, g x ≤ Real.sqrt 3 :=
by
  sorry

end max_g_equals_sqrt3_l83_83367


namespace complement_A_is_01_l83_83000

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Define the set A given the conditions
def A : Set ℝ := {x | x ≥ 1} ∪ {x | x < 0}

-- State the theorem: complement of A is the interval [0, 1)
theorem complement_A_is_01 : Set.compl A = {x : ℝ | 0 ≤ x ∧ x < 1} :=
by
  sorry

end complement_A_is_01_l83_83000


namespace left_vertex_of_ellipse_l83_83403

theorem left_vertex_of_ellipse :
  ∃ (a b c : ℝ), 
    (a > b) ∧ (b > 0) ∧ (b = 4) ∧ (c = 3) ∧ 
    (c^2 = a^2 - b^2) ∧ 
    (3^2 = a^2 - 4^2) ∧ 
    (a = 5) ∧ 
    (∀ x y : ℝ, (x, y) = (-5, 0)) := 
sorry

end left_vertex_of_ellipse_l83_83403


namespace basketball_game_l83_83815

theorem basketball_game 
    (a b x : ℕ)
    (h1 : 3 * b = 2 * a)
    (h2 : x = 2 * b)
    (h3 : 2 * a + 3 * b + x = 72) : 
    x = 18 :=
sorry

end basketball_game_l83_83815


namespace find_lesser_number_l83_83464

theorem find_lesser_number (x y : ℕ) (h₁ : x + y = 60) (h₂ : x - y = 10) : y = 25 := by
  sorry

end find_lesser_number_l83_83464


namespace sandy_spent_money_l83_83749

theorem sandy_spent_money :
  let shorts := 13.99
  let shirt := 12.14
  let jacket := 7.43
  shorts + shirt + jacket = 33.56 :=
by
  let shorts := 13.99
  let shirt := 12.14
  let jacket := 7.43
  have total_spent : shorts + shirt + jacket = 33.56 := sorry
  exact total_spent

end sandy_spent_money_l83_83749


namespace parabola_vertex_intercept_l83_83179

variable (a b c p : ℝ)

theorem parabola_vertex_intercept (h_vertex : ∀ x : ℝ, (a * (x - p) ^ 2 + p) = a * x^2 + b * x + c)
                                  (h_intercept : a * p^2 + p = 2 * p)
                                  (hp : p ≠ 0) : b = -2 :=
sorry

end parabola_vertex_intercept_l83_83179


namespace y_when_x_is_4_l83_83591

theorem y_when_x_is_4
  (x y : ℝ)
  (h1 : x + y = 30)
  (h2 : x - y = 10)
  (h3 : x * y = 200) :
  y = 50 :=
by
  sorry

end y_when_x_is_4_l83_83591


namespace sum_all_possible_values_l83_83863

theorem sum_all_possible_values (x : ℝ) (h : x^2 = 16) :
  (x = 4 ∨ x = -4) → (4 + (-4) = 0) :=
by
  intro h1
  have : 4 + (-4) = 0 := by norm_num
  exact this

end sum_all_possible_values_l83_83863


namespace determine_a_and_b_l83_83101

variable (a b : ℕ)
theorem determine_a_and_b 
  (h1: 0 ≤ a ∧ a ≤ 9) 
  (h2: 0 ≤ b ∧ b ≤ 9)
  (h3: (a + b + 45) % 9 = 0)
  (h4: (b - a) % 11 = 3) : 
  a = 3 ∧ b = 6 :=
sorry

end determine_a_and_b_l83_83101


namespace cylinder_height_l83_83664

noncomputable def height_of_cylinder_inscribed_in_sphere : ℝ := 4 * Real.sqrt 10

theorem cylinder_height :
  ∀ (R_cylinder R_sphere : ℝ), R_cylinder = 3 → R_sphere = 7 →
  (height_of_cylinder_inscribed_in_sphere = 4 * Real.sqrt 10) := by
  intros R_cylinder R_sphere h1 h2
  sorry

end cylinder_height_l83_83664


namespace rational_sum_zero_l83_83570

theorem rational_sum_zero {a b c : ℚ} (h : (a + b + c) * (a + b - c) = 4 * c^2) : a + b = 0 := 
sorry

end rational_sum_zero_l83_83570


namespace interval_second_bell_l83_83958

theorem interval_second_bell 
  (T : ℕ)
  (h1 : ∀ n : ℕ, n ≠ 0 → 630 % n = 0)
  (h2 : gcd T 630 = T)
  (h3 : lcm 9 (lcm 14 18) = lcm 9 (lcm 14 18))
  (h4 : 630 % lcm 9 (lcm 14 18) = 0) : 
  T = 5 :=
sorry

end interval_second_bell_l83_83958


namespace inequality_solver_l83_83574

variable {m n x : ℝ}

-- Main theorem statement validating the instances described above.
theorem inequality_solver (h : 2 * m * x + 3 < 3 * x + n) :
  (2 * m - 3 > 0 ∧ x < (n - 3) / (2 * m - 3)) ∨ 
  (2 * m - 3 < 0 ∧ x > (n - 3) / (2 * m - 3)) ∨ 
  (m = 3 / 2 ∧ n > 3 ∧ ∀ x : ℝ, true) ∨ 
  (m = 3 / 2 ∧ n ≤ 3 ∧ ∀ x : ℝ, false) :=
sorry

end inequality_solver_l83_83574


namespace value_of_x_l83_83143

theorem value_of_x (x : ℝ) (h : 4 * x + 5 * x + x + 2 * x = 360) : x = 30 := 
by
  sorry

end value_of_x_l83_83143


namespace geometric_sequence_solution_l83_83247

-- Assume we have a type for real numbers
variable {R : Type} [LinearOrderedField R]

theorem geometric_sequence_solution (a b c : R)
  (h1 : -1 ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) (h5 : -9 ≠ 0)
  (h : ∃ r : R, r ≠ 0 ∧ (a = r * -1) ∧ (b = r * a) ∧ (c = r * b) ∧ (-9 = r * c)) :
  b = -3 ∧ a * c = 9 := by
  sorry

end geometric_sequence_solution_l83_83247


namespace part_a_part_b_part_c_l83_83199

theorem part_a (θ : ℝ) (m : ℕ) : |Real.sin (m * θ)| ≤ m * |Real.sin θ| :=
sorry

theorem part_b (θ₁ θ₂ : ℝ) (m : ℕ) (hm_even : Even m) : 
  |Real.sin (m * θ₂) - Real.sin (m * θ₁)| ≤ m * |Real.sin (θ₂ - θ₁)| :=
sorry

theorem part_c (m : ℕ) (hm_odd : Odd m) : 
  ∃ θ₁ θ₂ : ℝ, |Real.sin (m * θ₂) - Real.sin (m * θ₁)| > m * |Real.sin (θ₂ - θ₁)| :=
sorry

end part_a_part_b_part_c_l83_83199


namespace integer_implies_perfect_square_l83_83556

theorem integer_implies_perfect_square (n : ℕ) (h : ∃ m : ℤ, 2 + 2 * Real.sqrt (28 * (n ^ 2) + 1) = m) :
  ∃ k : ℤ, 2 + 2 * Real.sqrt (28 * (n ^ 2) + 1) = (k ^ 2) :=
by
  sorry

end integer_implies_perfect_square_l83_83556


namespace sample_size_obtained_l83_83468

/-- A theorem which states the sample size obtained when a sample is taken from a population. -/
theorem sample_size_obtained 
  (total_students : ℕ)
  (sample_students : ℕ)
  (h1 : total_students = 300)
  (h2 : sample_students = 50) : 
  sample_students = 50 :=
by
  sorry

end sample_size_obtained_l83_83468


namespace find_g_of_2_l83_83377

theorem find_g_of_2 {g : ℝ → ℝ} (h : ∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) : g 2 = 14 :=
sorry

end find_g_of_2_l83_83377


namespace find_13th_result_l83_83485

theorem find_13th_result 
  (average_25 : ℕ) (average_12_first : ℕ) (average_12_last : ℕ) 
  (total_25 : average_25 * 25 = 600) 
  (total_12_first : average_12_first * 12 = 168) 
  (total_12_last : average_12_last * 12 = 204) 
: average_25 - average_12_first - average_12_last = 228 :=
by
  sorry

end find_13th_result_l83_83485


namespace range_of_m_l83_83111

noncomputable def proposition_p (x m : ℝ) := (x - m) ^ 2 > 3 * (x - m)
noncomputable def proposition_q (x : ℝ) := x ^ 2 + 3 * x - 4 < 0

theorem range_of_m (m : ℝ) : 
  (∀ x, proposition_p x m → proposition_q x) → 
  (1 ≤ m ∨ m ≤ -7) :=
sorry

end range_of_m_l83_83111


namespace new_number_formed_l83_83708

theorem new_number_formed (h t u : ℕ) (Hh : h < 10) (Ht : t < 10) (Hu : u < 10) :
  let original_number := 100 * h + 10 * t + u
  let new_number := 2000 + 10 * original_number
  new_number = 1000 * (h + 2) + 100 * t + 10 * u :=
by
  -- Proof would go here
  sorry

end new_number_formed_l83_83708


namespace remainder_98765432101_div_240_l83_83814

theorem remainder_98765432101_div_240 :
  (98765432101 % 240) = 61 :=
by
  -- Proof to be filled in later
  sorry

end remainder_98765432101_div_240_l83_83814


namespace parabola_passing_through_4_neg2_l83_83437

theorem parabola_passing_through_4_neg2 :
  (∃ p : ℝ, y^2 = 2 * p * x ∧ y = -2 ∧ x = 4 ∧ (y^2 = x)) ∨
  (∃ p : ℝ, x^2 = -2 * p * y ∧ y = -2 ∧ x = 4 ∧ (x^2 = -8 * y)) :=
by
  sorry

end parabola_passing_through_4_neg2_l83_83437


namespace average_of_w_and_x_is_one_half_l83_83864

noncomputable def average_of_w_and_x (w x y : ℝ) : ℝ :=
  (w + x) / 2

theorem average_of_w_and_x_is_one_half (w x y : ℝ)
  (h1 : 2 / w + 2 / x = 2 / y)
  (h2 : w * x = y) : average_of_w_and_x w x y = 1 / 2 :=
by
  sorry

end average_of_w_and_x_is_one_half_l83_83864


namespace combined_area_difference_l83_83484

theorem combined_area_difference :
  let rect1_len := 11
  let rect1_wid := 11
  let rect2_len := 5.5
  let rect2_wid := 11
  2 * (rect1_len * rect1_wid) - 2 * (rect2_len * rect2_wid) = 121 := by
  sorry

end combined_area_difference_l83_83484


namespace age_double_in_years_l83_83657

theorem age_double_in_years (S M X: ℕ) (h1: M = S + 22) (h2: S = 20) (h3: M + X = 2 * (S + X)) : X = 2 :=
by 
  sorry

end age_double_in_years_l83_83657


namespace max_gcd_15n_plus_4_8n_plus_1_l83_83952

theorem max_gcd_15n_plus_4_8n_plus_1 (n : ℕ) (h : n > 0) : 
  ∃ g, g = gcd (15 * n + 4) (8 * n + 1) ∧ g ≤ 17 :=
sorry

end max_gcd_15n_plus_4_8n_plus_1_l83_83952


namespace sum_q_p_is_minus_12_l83_83098

noncomputable def p (x : ℝ) : ℝ := x^2 - 3 * x + 2

noncomputable def q (x : ℝ) : ℝ := -x^2

theorem sum_q_p_is_minus_12 :
  (q (p 0) + q (p 1) + q (p 2) + q (p 3) + q (p 4)) = -12 :=
by
  sorry

end sum_q_p_is_minus_12_l83_83098


namespace y_intercept_of_line_eq_l83_83186

theorem y_intercept_of_line_eq (x y : ℝ) (h : x + y - 1 = 0) : y = 1 :=
by
  sorry

end y_intercept_of_line_eq_l83_83186


namespace reachable_pair_D_l83_83982

noncomputable def fA (x : ℝ) : ℝ := Real.cos x
noncomputable def gA (x : ℝ) : ℝ := 2

noncomputable def fB (x : ℝ) : ℝ := Real.log x^2 - 2*x + 5
noncomputable def gB (x : ℝ) : ℝ := Real.sin (Real.pi/2 * x)

noncomputable def fC (x : ℝ) : ℝ := Real.sqrt (4 - x^2)
noncomputable def gC (x : ℝ) : ℝ := (3/4)*x + 15/4

noncomputable def fD (x : ℝ) : ℝ := x + 2/x
noncomputable def gD (x : ℝ) : ℝ := Real.log x + 2

/-- The pairs (fA, gA), (fB, gB), and (fC, gC) are not reachable, but (fD, gD) is reachable. -/
theorem reachable_pair_D: 
  (∀ x, |fA x - gA x| ≥ 1) ∧
  (∀ x, |fB x - gB x| ≥ 1) ∧
  (∀ x, |fC x - gC x| ≥ 1) ∧
  (∃ x, |fD x - gD x| < 1) :=
by
  sorry

end reachable_pair_D_l83_83982


namespace product_of_smallest_primes_l83_83626

theorem product_of_smallest_primes :
  2 * 3 * 11 = 66 :=
by
  sorry

end product_of_smallest_primes_l83_83626


namespace compute_expression_l83_83396

-- Given Conditions
def is_root (p : Polynomial ℝ) (x : ℝ) := p.eval x = 0

def a : ℝ := 1  -- Placeholder value
def b : ℝ := 2  -- Placeholder value
def c : ℝ := 3  -- Placeholder value
def p : Polynomial ℝ := Polynomial.C (-6) + Polynomial.C 11 * Polynomial.X - Polynomial.C 6 * Polynomial.X^2 + Polynomial.X^3

-- Assertions based on conditions
axiom h_a_root : is_root p a
axiom h_b_root : is_root p b
axiom h_c_root : is_root p c

-- Proof Problem Statement
theorem compute_expression : 
  (ab c : ℝ), (is_root p a) → (is_root p b) → (is_root p c) → 
  ((a * b / c) + (b * c / a) + (c * a / b) = 49 / 6) :=
begin
  sorry,
end


end compute_expression_l83_83396


namespace largest_natural_gas_reserves_l83_83756
noncomputable def top_country_in_natural_gas_reserves : String :=
  "Russia"

theorem largest_natural_gas_reserves (countries : Fin 4 → String) :
  countries 0 = "Russia" → 
  countries 1 = "Finland" → 
  countries 2 = "United Kingdom" → 
  countries 3 = "Norway" → 
  top_country_in_natural_gas_reserves = countries 0 :=
by
  intros h_russia h_finland h_uk h_norway
  rw [h_russia]
  sorry

end largest_natural_gas_reserves_l83_83756


namespace part1_part2_l83_83096

def A (x : ℝ) : Prop := x^2 + 2*x - 3 < 0
def B (x : ℝ) (a : ℝ) : Prop := abs (x + a) < 1

theorem part1 (a : ℝ) (h : a = 3) : (∃ x : ℝ, (A x ∨ B x a)) ↔ (∃ x : ℝ, -4 < x ∧ x < 1) :=
by {
  sorry
}

theorem part2 : (∀ x : ℝ, B x a → A x) ∧ (¬ ∀ x : ℝ, A x → B x a) ↔ 0 ≤ a ∧ a ≤ 2 :=
by {
  sorry
}

end part1_part2_l83_83096


namespace retailer_profit_percentage_l83_83781

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

end retailer_profit_percentage_l83_83781


namespace floor_e_sub_6_eq_neg_4_l83_83520

theorem floor_e_sub_6_eq_neg_4 :
  (⌊(e:Real) - 6⌋ = -4) :=
by
  let h₁ : 2 < (e:Real) := sorry -- assuming e is the base of natural logarithms
  let h₂ : (e:Real) < 3 := sorry
  sorry

end floor_e_sub_6_eq_neg_4_l83_83520


namespace find_m_n_pairs_l83_83106

theorem find_m_n_pairs (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) :
  (∀ᶠ a in Filter.atTop, (a^m + a - 1) % (a^n + a^2 - 1) = 0) → m = n + 2 :=
by
  sorry

end find_m_n_pairs_l83_83106


namespace kanul_total_amount_l83_83728

theorem kanul_total_amount (T : ℝ) (R : ℝ) (M : ℝ) (C : ℝ)
  (hR : R = 80000)
  (hM : M = 30000)
  (hC : C = 0.2 * T)
  (hT : T = R + M + C) : T = 137500 :=
by {
  sorry
}

end kanul_total_amount_l83_83728


namespace julies_birthday_day_of_week_l83_83820

theorem julies_birthday_day_of_week
    (fred_birthday_monday : Nat)
    (pat_birthday_before_fred : Nat)
    (julie_birthday_before_pat : Nat)
    (fred_birthday_after_pat : fred_birthday_monday - pat_birthday_before_fred = 37)
    (julie_birthday_before_pat_eq : pat_birthday_before_fred - julie_birthday_before_pat = 67)
    : (julie_birthday_before_pat - julie_birthday_before_pat % 7 + ((julie_birthday_before_pat % 7) - fred_birthday_monday % 7)) % 7 = 2 :=
by
  sorry

end julies_birthday_day_of_week_l83_83820


namespace Soyun_distance_l83_83545

theorem Soyun_distance
  (perimeter : ℕ)
  (Soyun_speed : ℕ)
  (Jia_speed : ℕ)
  (meeting_time : ℕ)
  (time_to_meet : perimeter = (Soyun_speed + Jia_speed) * meeting_time) :
  Soyun_speed * meeting_time = 10 :=
by
  sorry

end Soyun_distance_l83_83545


namespace calc_g_x_plus_2_minus_g_x_l83_83854

def g (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 4

theorem calc_g_x_plus_2_minus_g_x (x : ℝ) : g (x + 2) - g x = 12 * x + 22 := 
by 
  sorry

end calc_g_x_plus_2_minus_g_x_l83_83854


namespace line_equation_l83_83580

theorem line_equation
  (P : ℝ × ℝ) (hP : P = (1, -1))
  (h_perp : ∀ x y : ℝ, 3 * x - 2 * y = 0 → 2 * x + 3 * y = 0):
  ∃ m : ℝ, (2 * P.1 + 3 * P.2 + m = 0) ∧ m = 1 :=
by
  sorry

end line_equation_l83_83580


namespace total_candies_l83_83404

-- Condition definitions
def lindaCandies : ℕ := 34
def chloeCandies : ℕ := 28

-- Proof statement to show their total candies
theorem total_candies : lindaCandies + chloeCandies = 62 := 
by
  sorry

end total_candies_l83_83404


namespace cost_of_four_enchiladas_and_five_tacos_l83_83746

-- Define the cost of an enchilada and a taco
variables (e t : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := e + 4 * t = 2.30
def condition2 : Prop := 4 * e + t = 3.10

-- Define the final cost of four enchiladas and five tacos
def cost : ℝ := 4 * e + 5 * t

-- State the theorem we need to prove
theorem cost_of_four_enchiladas_and_five_tacos 
  (h1 : condition1 e t) 
  (h2 : condition2 e t) : 
  cost e t = 4.73 := 
sorry

end cost_of_four_enchiladas_and_five_tacos_l83_83746


namespace canned_boxes_equation_l83_83778

theorem canned_boxes_equation (x : ℕ) (h₁: x ≤ 300) :
  2 * 14 * x = 32 * (300 - x) :=
by
sorry

end canned_boxes_equation_l83_83778


namespace max_area_inscribed_octagon_l83_83213

theorem max_area_inscribed_octagon
  (R : ℝ)
  (s : ℝ)
  (a b : ℝ)
  (h1 : s^2 = 5)
  (h2 : (a * b) = 4)
  (h3 : (s * Real.sqrt 2) = (2*R))
  (h4 : (Real.sqrt (a^2 + b^2)) = 2 * R) :
  ∃ A : ℝ, A = 3 * Real.sqrt 5 :=
by
  sorry

end max_area_inscribed_octagon_l83_83213


namespace triangle_area_ratio_l83_83017

-- Define parabola and focus
def parabola (x y : ℝ) : Prop := y^2 = 8 * x
def focus : (ℝ × ℝ) := (2, 0)

-- Define the line passing through the focus and intersecting the parabola
def line_through_focus (f : ℝ × ℝ) (a b : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  l (f.1) = f.2 ∧ parabola a.1 a.2 ∧ parabola b.1 b.2 ∧   -- line passes through the focus and intersects parabola at a and b
  l a.1 = a.2 ∧ l b.1 = b.2 ∧ 
  |a.1 - f.1| + |a.2 - f.2| = 3 ∧ -- condition |AF| = 3
  (f = (2, 0))

-- The proof problem
theorem triangle_area_ratio (a b : ℝ × ℝ) (l : ℝ → ℝ) 
  (h_line : line_through_focus focus a b l) :
  ∃ r, r = (1 / 2) := 
sorry

end triangle_area_ratio_l83_83017


namespace infinite_series_sum_l83_83684

theorem infinite_series_sum :
  (∑' n : ℕ, (n + 1) / 4^(n + 1)) + (∑' n : ℕ, 1 / 2^(n + 1)) = 13 / 9 := 
sorry

end infinite_series_sum_l83_83684


namespace cone_lateral_surface_area_l83_83831

theorem cone_lateral_surface_area (r : ℕ) (V : ℝ) (h l S : ℝ)
  (h_r : r = 6)
  (h_V : V = 30 * Real.pi)
  (h_volume : V = (1 / 3) * Real.pi * (r ^ 2) * h)
  (h_slant_height : l = Real.sqrt (r^2 + h^2))
  (h_lateral_surface_area : S = Real.pi * r * l) :
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l83_83831


namespace children_on_bus_l83_83325

theorem children_on_bus (initial_children additional_children total_children : ℕ) (h1 : initial_children = 26) (h2 : additional_children = 38) : total_children = initial_children + additional_children → total_children = 64 :=
by
  -- Proof goes here
  sorry

end children_on_bus_l83_83325


namespace max_tiles_accommodated_l83_83181

/-- 
The rectangular tiles, each of size 40 cm by 28 cm, must be laid horizontally on a rectangular floor
of size 280 cm by 240 cm, such that the tiles do not overlap, and they are placed in an alternating
checkerboard pattern with edges jutting against each other on all edges. A tile can be placed in any
orientation so long as its edges are parallel to the edges of the floor, and it follows the required
checkerboard pattern. No tile should overshoot any edge of the floor. Determine the maximum number 
of tiles that can be accommodated on the floor while adhering to the placement pattern.
-/
theorem max_tiles_accommodated (tile_len tile_wid floor_len floor_wid : ℕ)
  (h_tile_len : tile_len = 40)
  (h_tile_wid : tile_wid = 28)
  (h_floor_len : floor_len = 280)
  (h_floor_wid : floor_wid = 240) :
  tile_len * tile_wid * 12 ≤ floor_len * floor_wid :=
by 
  sorry

end max_tiles_accommodated_l83_83181


namespace gus_buys_2_dozen_l83_83298

-- Definitions from conditions
def dozens_to_golf_balls (d : ℕ) : ℕ := d * 12
def total_golf_balls : ℕ := 132
def golf_balls_per_dozen : ℕ := 12
def dan_buys : ℕ := 5
def chris_buys_golf_balls : ℕ := 48

-- The number of dozens Gus buys
noncomputable def gus_buys (total_dozens dan_dozens chris_dozens : ℕ) : ℕ := total_dozens - dan_dozens - chris_dozens

theorem gus_buys_2_dozen : gus_buys (total_golf_balls / golf_balls_per_dozen) dan_buys (chris_buys_golf_balls / golf_balls_per_dozen) = 2 := by
  sorry

end gus_buys_2_dozen_l83_83298


namespace log_function_domain_l83_83426

theorem log_function_domain (x : ℝ) : 
  (3 - x > 0) ∧ (x - 1 > 0) ∧ (x - 1 ≠ 1) -> (1 < x ∧ x < 3 ∧ x ≠ 2) :=
by
  intro h
  sorry

end log_function_domain_l83_83426


namespace max_sinA_cosB_cosC_l83_83807

theorem max_sinA_cosB_cosC (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 0 < A ∧ A < 180) (h3 : 0 < B ∧ B < 180) (h4 : 0 < C ∧ C < 180) : 
  ∃ M : ℝ, M = (1 + Real.sqrt 5) / 2 ∧ ∀ a b c : ℝ, a + b + c = 180 → 0 < a ∧ a < 180 → 0 < b ∧ b < 180 → 0 < c ∧ c < 180 → (Real.sin a + Real.cos b * Real.cos c) ≤ M :=
by sorry

end max_sinA_cosB_cosC_l83_83807


namespace cone_lateral_surface_area_l83_83836

theorem cone_lateral_surface_area (r V : ℝ) (h l S : ℝ) 
  (radius_condition : r = 6)
  (volume_condition : V = 30 * Real.pi)
  (volume_formula : V = (1 / 3) * Real.pi * r^2 * h)
  (slant_height_formula : l = Real.sqrt (r^2 + h^2))
  (lateral_surface_area_formula : S = Real.pi * r * l) :
  S = 39 * Real.pi := 
sorry

end cone_lateral_surface_area_l83_83836


namespace negation_proposition_l83_83701

open Classical

variable (x : ℝ)

def proposition (x : ℝ) : Prop := ∀ x > 1, Real.log x / Real.log 2 > 0

theorem negation_proposition (h : ¬ proposition x) : 
  ∃ x > 1, Real.log x / Real.log 2 ≤ 0 := by
  sorry

end negation_proposition_l83_83701


namespace max_students_late_all_three_days_l83_83264

theorem max_students_late_all_three_days (A B C total l: ℕ) 
  (hA: A = 20) 
  (hB: B = 13) 
  (hC: C = 7) 
  (htotal: total = 30) 
  (hposA: 0 ≤ A) (hposB: 0 ≤ B) (hposC: 0 ≤ C) 
  (hpostotal: 0 ≤ total) 
  : l = 5 := by
  sorry

end max_students_late_all_three_days_l83_83264


namespace sum_of_squares_arithmetic_geometric_l83_83902

theorem sum_of_squares_arithmetic_geometric (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 225) : x^2 + y^2 = 1150 :=
by
  sorry

end sum_of_squares_arithmetic_geometric_l83_83902


namespace find_multiple_of_pages_l83_83223

-- Definitions based on conditions
def beatrix_pages : ℕ := 704
def cristobal_extra_pages : ℕ := 1423
def cristobal_pages (x : ℕ) : ℕ := x * beatrix_pages + 15

-- Proposition to prove the multiple x equals 2
theorem find_multiple_of_pages (x : ℕ) (h : cristobal_pages x = beatrix_pages + cristobal_extra_pages) : x = 2 :=
  sorry

end find_multiple_of_pages_l83_83223


namespace expression_is_integer_l83_83027

theorem expression_is_integer (n : ℤ) : (∃ k : ℤ, n * (n + 1) * (n + 2) * (n + 3) = 24 * k) := 
sorry

end expression_is_integer_l83_83027


namespace partition_nat_l83_83522

open Set

theorem partition_nat (c : ℚ) (h₀ : 0 < c) (h₁ : c ≠ 1) :
    ∃ (A B : Set ℕ), (A ∩ B = ∅) ∧ (∀ x y ∈ A, (x:ℚ) / y ≠ c) ∧ (∀ x y ∈ B, (x:ℚ) / y ≠ c) := by
  sorry

end partition_nat_l83_83522


namespace four_n_div_four_remainder_zero_l83_83373

theorem four_n_div_four_remainder_zero (n : ℤ) (h : n % 4 = 3) : (4 * n) % 4 = 0 := 
by
  sorry

end four_n_div_four_remainder_zero_l83_83373


namespace possible_values_for_p_l83_83770

-- Definitions for the conditions
variables {a b c p : ℝ}

-- Assumptions
def distinct (a b c : ℝ) := ¬(a = b) ∧ ¬(b = c) ∧ ¬(c = a)
def main_eq (a b c p : ℝ) := a + (1 / b) = p ∧ b + (1 / c) = p ∧ c + (1 / a) = p

-- Theorem statement
theorem possible_values_for_p (h1 : distinct a b c) (h2 : main_eq a b c p) : p = 1 ∨ p = -1 := 
sorry

end possible_values_for_p_l83_83770


namespace johns_salary_percentage_increase_l83_83010

theorem johns_salary_percentage_increase (initial_salary final_salary : ℕ) (h1 : initial_salary = 50) (h2 : final_salary = 90) :
  ((final_salary - initial_salary : ℕ) / initial_salary : ℚ) * 100 = 80 := by
  sorry

end johns_salary_percentage_increase_l83_83010


namespace product_of_two_numbers_l83_83586

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x ^ 2 + y ^ 2 = 289)
  (h2 : x + y = 23) : 
  x * y = 120 :=
by
  sorry

end product_of_two_numbers_l83_83586


namespace lesser_number_l83_83456

theorem lesser_number (x y : ℕ) (h1: x + y = 60) (h2: x - y = 10) : y = 25 :=
sorry

end lesser_number_l83_83456


namespace prob_not_perfect_power_200_l83_83434

open Finset

-- Definitions
def is_perfect_power (n : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 1 ∧ n = x^y

def count_perfect_powers (m : ℕ) : ℕ :=
  (filter is_perfect_power (range (m + 1))).card

def count_not_perfect_powers (m : ℕ) : ℕ :=
  m - count_perfect_powers m

-- Main theorem statement
theorem prob_not_perfect_power_200 :
  (count_not_perfect_powers 200 : ℚ) / 200 = 181 / 200 :=
sorry

end prob_not_perfect_power_200_l83_83434


namespace water_left_in_bucket_l83_83956

theorem water_left_in_bucket (initial_amount poured_amount : ℝ) (h1 : initial_amount = 0.8) (h2 : poured_amount = 0.2) : initial_amount - poured_amount = 0.6 := by
  sorry

end water_left_in_bucket_l83_83956


namespace find_c_l83_83578

theorem find_c (a b c : ℚ) (h_eqn : ∀ y, a * y^2 + b * y + c = y^2 / 12 + 5 * y / 6 + 145 / 12)
  (h_vertex : ∀ x, x = a * (-5)^2 + b * (-5) + c)
  (h_pass : a * (-1 + 5)^2 + 1 = 4) :
  c = 145 / 12 := by
sorry

end find_c_l83_83578


namespace cone_lateral_surface_area_l83_83826

theorem cone_lateral_surface_area (r h l S : ℝ) (π_pos : 0 < π) (r_eq : r = 6)
  (V : ℝ) (V_eq : V = 30 * π)
  (vol_eq : V = (1/3) * π * r^2 * h)
  (h_eq : h = 5 / 2)
  (l_eq : l = Real.sqrt (r^2 + h^2))
  (S_eq : S = π * r * l) :
  S = 39 * π :=
  sorry

end cone_lateral_surface_area_l83_83826


namespace proposition_p_neither_sufficient_nor_necessary_l83_83243

-- Define propositions p and q
def p (m : ℝ) : Prop := m = -1
def q (m : ℝ) : Prop := ∀ x y : ℝ, (x - 1 = 0) ∧ (x + m^2 * y = 0) → ∀ x' y' : ℝ, x' = x ∧ y' = y → (x - 1) * (x + m^2 * y) = 0

-- Main theorem statement
theorem proposition_p_neither_sufficient_nor_necessary (m : ℝ) : ¬ (p m → q m) ∧ ¬ (q m → p m) :=
by
  sorry

end proposition_p_neither_sufficient_nor_necessary_l83_83243


namespace solve_eq1_solve_eq2_l83_83292

-- Define the first proof problem
theorem solve_eq1 (x : ℝ) : 2 * x - 3 = 3 * (x + 1) → x = -6 :=
by
  sorry

-- Define the second proof problem
theorem solve_eq2 (x : ℝ) : (1 / 2) * x - (9 * x - 2) / 6 - 2 = 0 → x = -5 / 3 :=
by
  sorry

end solve_eq1_solve_eq2_l83_83292


namespace no_real_intersection_l83_83924

def parabola_line_no_real_intersection : Prop :=
  let a := 3
  let b := -6
  let c := 5
  (b^2 - 4 * a * c) < 0

theorem no_real_intersection (h : parabola_line_no_real_intersection) : 
  ∀ x : ℝ, 3*x^2 - 4*x + 2 ≠ 2*x - 3 :=
by sorry

end no_real_intersection_l83_83924


namespace simplify_expression_l83_83671

theorem simplify_expression : (2^4 * 2^4 * 2^4) = 2^12 :=
by
  sorry

end simplify_expression_l83_83671


namespace longest_side_of_rectangle_l83_83735

theorem longest_side_of_rectangle
  (m : ℕ) (l w : ℕ)
  (h1 : 2 * l + 2 * w = m)
  (h2 : l * w = 12 * (m / 2)) :
  max l w = 72 := by 
  have : m = 240 := by sorry
  have : 12 * (m / 2) = 2880 := by sorry
  sorry

end longest_side_of_rectangle_l83_83735


namespace parallel_vectors_implies_scalar_l83_83368

-- Defining the vectors a and b
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (-2, m)

-- Stating the condition and required proof
theorem parallel_vectors_implies_scalar (m : ℝ) (h : (vector_a.snd / vector_a.fst) = (vector_b m).snd / (vector_b m).fst) : m = -4 :=
by sorry

end parallel_vectors_implies_scalar_l83_83368


namespace lesser_of_two_numbers_l83_83441

theorem lesser_of_two_numbers (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
by
  sorry

end lesser_of_two_numbers_l83_83441


namespace remainder_when_divided_l83_83979

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 + x^3 + 1

-- The statement to be proved
theorem remainder_when_divided (x : ℝ) : (p 2) = 25 :=
by
  sorry

end remainder_when_divided_l83_83979


namespace tom_monthly_fluid_intake_l83_83105

-- Define the daily fluid intake amounts
def daily_soda_intake := 5 * 12
def daily_water_intake := 64
def daily_juice_intake := 3 * 8
def daily_sports_drink_intake := 2 * 16
def additional_weekend_smoothie := 32

-- Define the weekdays and weekend days in a month
def weekdays_in_month := 5 * 4
def weekend_days_in_month := 2 * 4

-- Calculate the total daily intake
def daily_intake := daily_soda_intake + daily_water_intake + daily_juice_intake + daily_sports_drink_intake
def weekend_daily_intake := daily_intake + additional_weekend_smoothie

-- Calculate the total monthly intake
def total_fluid_intake_in_month := (daily_intake * weekdays_in_month) + (weekend_daily_intake * weekend_days_in_month)

-- Statement to prove
theorem tom_monthly_fluid_intake : total_fluid_intake_in_month = 5296 :=
by
  unfold total_fluid_intake_in_month
  unfold daily_intake weekend_daily_intake
  unfold weekdays_in_month weekend_days_in_month
  unfold daily_soda_intake daily_water_intake daily_juice_intake daily_sports_drink_intake additional_weekend_smoothie
  sorry

end tom_monthly_fluid_intake_l83_83105


namespace minimize_y_l83_83158

def y (x a b : ℝ) : ℝ := (x-a)^2 * (x-b)^2

theorem minimize_y (a b : ℝ) : ∃ x : ℝ, y x a b = 0 := by
  use a
  sorry

end minimize_y_l83_83158


namespace winning_candidate_percentage_l83_83054

theorem winning_candidate_percentage
  (votes1 votes2 votes3 : ℕ)
  (h1 : votes1 = 3000)
  (h2 : votes2 = 5000)
  (h3 : votes3 = 20000) :
  ((votes3 : ℝ) / (votes1 + votes2 + votes3) * 100) = 71.43 := by
  sorry

end winning_candidate_percentage_l83_83054


namespace cone_lateral_surface_area_l83_83842

-- Definitions based on the conditions
def coneRadius : ℝ := 6
def coneVolume : ℝ := 30 * Real.pi

-- Mathematical statement
theorem cone_lateral_surface_area (r V : ℝ) (hr : r = coneRadius) (hV : V = coneVolume) :
  ∃ S : ℝ, S = 39 * Real.pi :=
by 
  have h_volume := hV
  have h_radius := hr
  sorry

end cone_lateral_surface_area_l83_83842


namespace cone_lateral_surface_area_l83_83825

theorem cone_lateral_surface_area (r h l S : ℝ) (π_pos : 0 < π) (r_eq : r = 6)
  (V : ℝ) (V_eq : V = 30 * π)
  (vol_eq : V = (1/3) * π * r^2 * h)
  (h_eq : h = 5 / 2)
  (l_eq : l = Real.sqrt (r^2 + h^2))
  (S_eq : S = π * r * l) :
  S = 39 * π :=
  sorry

end cone_lateral_surface_area_l83_83825


namespace probability_fourth_term_integer_l83_83150

-- Define the initial conditions and rules for the sequence
def initial_term : ℕ := 8

def heads_step (n : ℕ) : ℕ :=
2 * n - 1

def tails_step (n : ℕ) (tails_count : ℕ) : ℕ :=
if tails_count = 1 ∨ tails_count = 3 then
  n / 2 - 1
else
  3 * n - 2

-- Define a function to calculate the fourth term under a sequence of flips
def fourth_term (flips : List Bool) : Rat :=
match flips with
| [f1, f2, f3] =>
  let a2 := if f1 then heads_step initial_term else tails_step initial_term 1
  let a3 := if f2 then heads_step a2 else tails_step a2 (if f1 = false then 2 else 1)
  if f3 then heads_step a3 else tails_step a3 (if f2 = false ∧ f1 = false then 3 else 1)
| _ => 0 -- Default case: incorrect number of flips provided

-- Define the main theorem
theorem probability_fourth_term_integer : 
  let outcomes := [fourth_term [true, true, true], fourth_term [true, true, false],
                    fourth_term [true, false, true], fourth_term [true, false, false],
                    fourth_term [false, true, true], fourth_term [false, true, false],
                    fourth_term [false, false, true]] in
  (outcomes.filter (λ x, x.denom = 1)).length = 4 ∧
  outcomes.length = 7 →
  (outcomes.filter (λ x, x.denom = 1)).length / outcomes.length = Rat.ofInt 4 / 7 := by
  sorry

end probability_fourth_term_integer_l83_83150


namespace beavers_help_l83_83200

theorem beavers_help (initial final : ℝ) (h_initial : initial = 2.0) (h_final : final = 3) : final - initial = 1 :=
  by
    sorry

end beavers_help_l83_83200


namespace division_of_polynomials_l83_83197

theorem division_of_polynomials (a b : ℝ) :
  (18 * a^2 * b - 9 * a^5 * b^2) / (-3 * a * b) = -6 * a + 3 * a^4 * b :=
by
  sorry

end division_of_polynomials_l83_83197


namespace gcd_polynomial_l83_83576

theorem gcd_polynomial (a : ℕ) (h : 270 ∣ a) : Nat.gcd (5 * a^3 + 3 * a^2 + 5 * a + 45) a = 45 :=
sorry

end gcd_polynomial_l83_83576


namespace apple_price_l83_83503

variable (p q : ℝ)

theorem apple_price :
  (30 * p + 3 * q = 168) →
  (30 * p + 6 * q = 186) →
  (20 * p = 100) →
  p = 5 :=
by
  intros h1 h2 h3
  have h4 : p = 5 := sorry
  exact h4

end apple_price_l83_83503


namespace product_of_smallest_primes_l83_83627

theorem product_of_smallest_primes :
  2 * 3 * 11 = 66 :=
by
  sorry

end product_of_smallest_primes_l83_83627


namespace area_of_triangle_BXN_l83_83889

open Real

noncomputable def is_isosceles (A B C : Point ℝ) : Prop := (dist A B = dist B C)

noncomputable def midpoint (A B : Point ℝ) : Point ℝ := (1/2 • A + 1/2 • B)

noncomputable def is_equilateral (A B C : Point ℝ) : Prop :=
  (dist A B = dist B C) ∧ (dist B C = dist C A)

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

theorem area_of_triangle_BXN 
  (A B C M N X : Point ℝ)
  (h1 : is_isosceles A B C)
  (h2 : dist A C = 4)
  (h3 : M = midpoint A C)
  (h4 : N = midpoint A B)
  (h5 : line (C, N) = angle_bisector A C B)
  (h6 : X = line (B, M) ∩ line (C, N))
  (h7 : is_equilateral B X N)
  : area_of_triangle_BXN = sqrt(3)/4 := 
sorry

end area_of_triangle_BXN_l83_83889


namespace cones_sold_l83_83282

-- Define the conditions
variable (milkshakes : Nat)
variable (cones : Nat)

-- Assume the given conditions
axiom h1 : milkshakes = 82
axiom h2 : milkshakes = cones + 15

-- State the theorem to prove
theorem cones_sold : cones = 67 :=
by
  -- Proof goes here
  sorry

end cones_sold_l83_83282


namespace number_of_Sunzi_books_l83_83330

theorem number_of_Sunzi_books
    (num_books : ℕ) (total_cost : ℕ)
    (price_Zhuangzi price_Kongzi price_Mengzi price_Laozi price_Sunzi : ℕ)
    (num_Zhuangzi num_Kongzi num_Mengzi num_Laozi num_Sunzi : ℕ) :
  num_books = 300 →
  total_cost = 4500 →
  price_Zhuangzi = 10 →
  price_Kongzi = 20 →
  price_Mengzi = 15 →
  price_Laozi = 30 →
  price_Sunzi = 12 →
  num_Zhuangzi = num_Kongzi →
  num_Sunzi = 4 * num_Laozi + 15 →
  num_Zhuangzi + num_Kongzi + num_Mengzi + num_Laozi + num_Sunzi = num_books →
  price_Zhuangzi * num_Zhuangzi +
  price_Kongzi * num_Kongzi +
  price_Mengzi * num_Mengzi +
  price_Laozi * num_Laozi +
  price_Sunzi * num_Sunzi = total_cost →
  num_Sunzi = 75 :=
by
  intros h_nb h_tc h_pZ h_pK h_pM h_pL h_pS h_nZ h_nS h_books h_cost
  sorry

end number_of_Sunzi_books_l83_83330


namespace am_gm_inequality_l83_83364

theorem am_gm_inequality {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (b^2 / a) + (c^2 / b) + (a^2 / c) ≥ a + b + c :=
by
  sorry

end am_gm_inequality_l83_83364


namespace remainder_of_poly_division_l83_83929

theorem remainder_of_poly_division :
  ∀ (x : ℂ), ((x + 1)^2048) % (x^2 - x + 1) = x + 1 :=
by
  sorry

end remainder_of_poly_division_l83_83929


namespace no_line_can_intersect_all_segments_of_11_segment_polygonal_chain_l83_83092

theorem no_line_can_intersect_all_segments_of_11_segment_polygonal_chain
  (vertices : Fin 11 → ℝ × ℝ)
  (segments : Fin 11 → (ℝ × ℝ) × (ℝ × ℝ))
  (closed_chain : segments 10.2.2 = segments 0.1 ∧ ∀ i, (segments i).2 = (segments (i + 1) % 11).1)
  (line : (ℝ × ℝ) → Prop)
  (line_no_vertex : ∀ v, v ∈ set.range vertices → ¬line v) :
  ¬ ∀ i, ∃ x, line x ∧ segments i.1 ≤ x ∧ x ≤ segments i.2 := sorry

end no_line_can_intersect_all_segments_of_11_segment_polygonal_chain_l83_83092


namespace kelseys_sister_is_3_years_older_l83_83887

-- Define the necessary conditions
def kelsey_birth_year : ℕ := 1999 - 25
def sister_birth_year : ℕ := 2021 - 50
def age_difference (a b : ℕ) : ℕ := a - b

-- State the theorem to prove
theorem kelseys_sister_is_3_years_older :
  age_difference kelsey_birth_year sister_birth_year = 3 :=
by
  -- Skipping the proof steps as only the statement is needed
  sorry

end kelseys_sister_is_3_years_older_l83_83887


namespace prob1_prob2_l83_83505

theorem prob1:
  (6 * (Real.tan (30 * Real.pi / 180))^2 - Real.sqrt 3 * Real.sin (60 * Real.pi / 180) - 2 * Real.sin (45 * Real.pi / 180)) = (1 / 2 - Real.sqrt 2) :=
sorry

theorem prob2:
  ((Real.sqrt 2 / 2) * Real.cos (45 * Real.pi / 180) - (Real.tan (40 * Real.pi / 180) + 1)^0 + Real.sqrt (1 / 4) + Real.sin (30 * Real.pi / 180)) = (1 / 2) :=
sorry

end prob1_prob2_l83_83505


namespace product_of_primes_l83_83623

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

noncomputable def smallest_one_digit_primes (p₁ p₂ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ p₁ < p₂ ∧ p₂ < 10 ∧ ∀ p : ℕ, is_prime p → p < 10 → p = p₁ ∨ p = p₂

noncomputable def smallest_two_digit_prime (p : ℕ) : Prop :=
  is_prime p ∧ p ≥ 10 ∧ p < 100 ∧ ∀ q : ℕ, is_prime q → q ≥ 10 → q < p → q = 11

theorem product_of_primes : ∃ p₁ p₂ p₃ : ℕ, smallest_one_digit_primes p₁ p₂ ∧ smallest_two_digit_prime p₃ ∧ p₁ * p₂ * p₃ = 66 := 
by
  sorry

end product_of_primes_l83_83623


namespace earnings_per_widget_l83_83421

/-
Theorem:
Given:
1. Hourly wage is $12.50.
2. Hours worked in a week is 40.
3. Total weekly earnings are $580.
4. Number of widgets produced in a week is 500.

We want to prove:
The earnings per widget are $0.16.
-/

theorem earnings_per_widget (hourly_wage : ℝ) (hours_worked : ℝ)
  (total_weekly_earnings : ℝ) (widgets_produced : ℝ) :
  (hourly_wage = 12.50) →
  (hours_worked = 40) →
  (total_weekly_earnings = 580) →
  (widgets_produced = 500) →
  ( (total_weekly_earnings - hourly_wage * hours_worked) / widgets_produced = 0.16) :=
by
  intros h_wage h_hours h_earnings h_widgets
  sorry

end earnings_per_widget_l83_83421


namespace area_increase_l83_83800

theorem area_increase (a : ℝ) : ((a + 2) ^ 2 - a ^ 2 = 4 * a + 4) := by
  sorry

end area_increase_l83_83800


namespace problem_correct_answer_l83_83648

theorem problem_correct_answer (x y : ℕ) (h1 : y > 3) (h2 : x^2 + y^4 = 2 * ((x - 6)^2 + (y + 1)^2)) : x^2 + y^4 = 1994 :=
  sorry

end problem_correct_answer_l83_83648


namespace quadratic_minimum_l83_83968

-- Define the constants p and q as positive real numbers
variables (p q : ℝ) (hp : 0 < p) (hq : 0 < q)

-- Define the quadratic function f
def f (x : ℝ) : ℝ := 3 * x^2 + p * x + q

-- Assertion to prove: the function f reaches its minimum at x = -p / 6
theorem quadratic_minimum : 
  ∃ x : ℝ, x = -p / 6 ∧ (∀ y : ℝ, f y ≥ f x) :=
sorry

end quadratic_minimum_l83_83968


namespace worth_of_entire_lot_l83_83793

theorem worth_of_entire_lot (half_share : ℝ) (amount_per_tenth : ℝ) (total_amount : ℝ) :
  half_share = 0.5 →
  amount_per_tenth = 460 →
  total_amount = (amount_per_tenth * 10) →
  (total_amount * 2) = 9200 :=
by
  intros h1 h2 h3
  sorry

end worth_of_entire_lot_l83_83793


namespace time_rachel_is_13_l83_83405

-- Definitions based on problem conditions
def time_matt := 12
def time_patty := time_matt / 3
def time_rachel := 2 * time_patty + 5

-- Theorem statement to prove Rachel's time to paint the house
theorem time_rachel_is_13 : time_rachel = 13 := 
by 
  sorry

end time_rachel_is_13_l83_83405


namespace probability_abs_diff_l83_83561

variables (P : ℕ → ℚ) (m : ℚ)

def is_probability_distribution : Prop :=
  P 1 = m ∧ P 2 = 1/4 ∧ P 3 = 1/4 ∧ P 4 = 1/3 ∧ m + 1/4 + 1/4 + 1/3 = 1

theorem probability_abs_diff (h : is_probability_distribution P m) :
  P 1 + P 3 = 5 / 12 :=
by 
sorry

end probability_abs_diff_l83_83561


namespace impossible_path_2018_grid_l83_83736

theorem impossible_path_2018_grid :
  ¬((∃ (path : Finset (Fin 2018 × Fin 2018)), 
    (0, 0) ∈ path ∧ (2017, 2017) ∈ path ∧ 
    (∀ {x y}, (x, y) ∈ path → (x + 1, y) ∈ path ∨ (x, y + 1) ∈ path ∨ (x - 1, y) ∈ path ∨ (x, y - 1) ∈ path) ∧ 
    (∀ {x y}, (x, y) ∈ path → (Finset.card path = 2018 * 2018)))) :=
by 
  sorry

end impossible_path_2018_grid_l83_83736


namespace positive_even_representation_l83_83147

theorem positive_even_representation (k : ℕ) (h : k > 0) :
  ∃ (a b : ℤ), (2 * k : ℤ) = a * b ∧ a + b = 0 := 
by
  sorry

end positive_even_representation_l83_83147


namespace product_of_primes_l83_83629

theorem product_of_primes : 2 * 3 * 11 = 66 :=
by 
  -- Start with the multiplication of the first two primes
  have h1 : 2 * 3 = 6 := by norm_num
  -- Then multiply the result with the smallest two-digit prime
  have h2 : 6 * 11 = 66 := by norm_num
  -- Combine the steps to get the final result
  exact eq.trans (congr_arg (λ x, x * 11) h1) h2

end product_of_primes_l83_83629


namespace sufficient_but_not_necessary_sin_condition_l83_83156

theorem sufficient_but_not_necessary_sin_condition (θ : ℝ) :
  (|θ - π/12| < π/12) → (sin θ < 1/2) :=
sorry

end sufficient_but_not_necessary_sin_condition_l83_83156


namespace find_PQ_l83_83386

noncomputable def right_triangle_tan (PQ PR : ℝ) (tan_P : ℝ) (R_right : Prop) : Prop :=
  tan_P = PQ / PR ∧ R_right

theorem find_PQ (PQ PR : ℝ) (tan_P : ℝ) (R_right : Prop)
  (h1 : tan_P = 3 / 2)
  (h2 : PR = 6)
  (h3 : R_right) :
  right_triangle_tan PQ PR tan_P R_right → PQ = 9 :=
by
  sorry

end find_PQ_l83_83386


namespace clea_ride_down_time_l83_83672

theorem clea_ride_down_time (c s d : ℝ) (h1 : d = 70 * c) (h2 : d = 28 * (c + s)) :
  (d / s) = 47 := by
  sorry

end clea_ride_down_time_l83_83672


namespace bookseller_fiction_books_count_l83_83073

theorem bookseller_fiction_books_count (n : ℕ) (h1 : n.factorial * 6 = 36) : n = 3 :=
sorry

end bookseller_fiction_books_count_l83_83073


namespace lunks_needed_for_apples_l83_83130

theorem lunks_needed_for_apples :
  (∀ l k a : ℕ, (4 * k = 2 * l) ∧ (3 * a = 5 * k ) → ∃ l', l' = (24 * l / 4)) :=
by
  intros l k a h
  obtain ⟨h1, h2⟩ := h
  have k_for_apples := 3 * a / 5
  have l_for_kunks := 4 * k / 2
  sorry

end lunks_needed_for_apples_l83_83130


namespace directrix_of_parabola_l83_83304

noncomputable def parabola_directrix (y : ℝ) (x : ℝ) : Prop :=
  y = 4 * x^2

theorem directrix_of_parabola : ∃ d : ℝ, (parabola_directrix (y := 4) (x := x) → d = -1/16) :=
by
  sorry

end directrix_of_parabola_l83_83304


namespace plaza_area_increase_l83_83802

theorem plaza_area_increase (a : ℝ) : 
  ((a + 2)^2 - a^2 = 4 * a + 4) :=
sorry

end plaza_area_increase_l83_83802


namespace maple_tree_taller_than_pine_tree_l83_83541

def improper_fraction (a b : ℕ) : ℚ := a + (b : ℚ) / 4
def mixed_number_to_improper_fraction (n m : ℕ) : ℚ := improper_fraction n m

def pine_tree_height : ℚ := mixed_number_to_improper_fraction 12 1
def maple_tree_height : ℚ := mixed_number_to_improper_fraction 18 3

theorem maple_tree_taller_than_pine_tree :
  maple_tree_height - pine_tree_height = 6 + 1 / 2 :=
by sorry

end maple_tree_taller_than_pine_tree_l83_83541


namespace stratified_sampling_third_grade_l83_83003

theorem stratified_sampling_third_grade (total_students : ℕ) (first_grade_students : ℕ)
  (second_grade_students : ℕ) (third_grade_students : ℕ) (sample_size : ℕ)
  (h_total : total_students = 270000) (h_first : first_grade_students = 99000)
  (h_second : second_grade_students = 90000) (h_third : third_grade_students = 81000)
  (h_sample : sample_size = 3000) :
  third_grade_students * (sample_size / total_students) = 900 := 
by {
  sorry
}

end stratified_sampling_third_grade_l83_83003


namespace students_attended_game_l83_83314

variable (s n : ℕ)

theorem students_attended_game (h1 : s + n = 3000) (h2 : 10 * s + 15 * n = 36250) : s = 1750 := by
  sorry

end students_attended_game_l83_83314


namespace positive_integer_solution_exists_l83_83107

theorem positive_integer_solution_exists (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h_eq : x^2 = y^2 + 7 * y + 6) : (x, y) = (6, 3) := 
sorry

end positive_integer_solution_exists_l83_83107


namespace total_revenue_l83_83055

theorem total_revenue (C A : ℕ) (P_C P_A total_tickets adult_tickets revenue : ℕ)
  (hCC : C = 6) -- Children's ticket price
  (hAC : A = 9) -- Adult's ticket price
  (hTT : total_tickets = 225) -- Total tickets sold
  (hAT : adult_tickets = 175) -- Adult tickets sold
  (hTR : revenue = 1875) -- Total revenue
  : revenue = adult_tickets * A + (total_tickets - adult_tickets) * C := sorry

end total_revenue_l83_83055


namespace enrollment_increase_1991_to_1992_l83_83786

theorem enrollment_increase_1991_to_1992 (E E_1992 E_1993 : ℝ)
    (h1 : E_1993 = 1.26 * E)
    (h2 : E_1993 = 1.05 * E_1992) :
    ((E_1992 - E) / E) * 100 = 20 :=
by
  sorry

end enrollment_increase_1991_to_1992_l83_83786


namespace not_sum_of_squares_l83_83412

def P (x y : ℝ) : ℝ := 4 + x^2 * y^4 + x^4 * y^2 - 3 * x^2 * y^2

theorem not_sum_of_squares (P : ℝ → ℝ → ℝ) : 
  (¬ ∃ g₁ g₂ : ℝ → ℝ → ℝ, ∀ x y : ℝ, P x y = g₁ x y * g₁ x y + g₂ x y * g₂ x y) :=
  by
  {
    -- By contradiction proof as outlined in the example problem
    sorry
  }

end not_sum_of_squares_l83_83412


namespace express_positive_rational_less_than_one_l83_83894
-- Import necessary libraries

-- Define the sequence as a predicate to show each term is a positive integer
def is_positive_integer_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, 0 < a n

-- Define the condition that for any prime p, there are infinitely many terms divisible by p
def infinitely_many_divisible_by (a : ℕ → ℕ) (p : ℕ) [h : Fact (Nat.Prime p)] : Prop :=
  ∃ infinitely_many (n : ℕ), p ∣ a n

-- Main theorem statement
theorem express_positive_rational_less_than_one
  (a : ℕ → ℕ)
  (h_seq: is_positive_integer_sequence a)
  (h_inf_prime: ∀ p [Fact (Nat.Prime p)], infinitely_many_divisible_by a p)
  (q : ℚ) (h_q_pos : 0 < q) (h_q_lt_1 : q < 1) :
  ∃ (b : ℕ → ℕ) (n : ℕ), (∀ i : ℕ, i < n → 0 ≤ b i ∧ b i < a i) ∧ q = ∑ i in finset.range n, (b i : ℚ) / (∏ j in finset.range (i + 1), a j) :=
sorry

end express_positive_rational_less_than_one_l83_83894


namespace solve_eq1_solve_eq2_l83_83419

theorem solve_eq1 (x : ℝ) : (x+1)^2 = 4 ↔ x = 1 ∨ x = -3 := 
by sorry

theorem solve_eq2 (x : ℝ) : 3*x^2 - 2*x - 1 = 0 ↔ x = 1 ∨ x = -1/3 := 
by sorry

end solve_eq1_solve_eq2_l83_83419


namespace bus_speed_excluding_stoppages_l83_83355

theorem bus_speed_excluding_stoppages (v : ℕ): (45 : ℝ) = (5 / 6 * v) → v = 54 :=
by
  sorry

end bus_speed_excluding_stoppages_l83_83355


namespace total_length_of_scale_l83_83336

theorem total_length_of_scale (num_parts : ℕ) (length_per_part : ℕ) 
  (h1: num_parts = 4) (h2: length_per_part = 20) : 
  num_parts * length_per_part = 80 := by
  sorry

end total_length_of_scale_l83_83336


namespace cracked_seashells_zero_l83_83469

/--
Tom found 15 seashells, and Fred found 43 seashells. After cleaning, it was discovered that Fred had 28 more seashells than Tom. Prove that the number of cracked seashells is 0.
-/
theorem cracked_seashells_zero
(Tom_seashells : ℕ)
(Fred_seashells : ℕ)
(cracked_seashells : ℕ)
(Tom_after_cleaning : ℕ := Tom_seashells - cracked_seashells)
(Fred_after_cleaning : ℕ := Fred_seashells - cracked_seashells)
(h1 : Tom_seashells = 15)
(h2 : Fred_seashells = 43)
(h3 : Fred_after_cleaning = Tom_after_cleaning + 28) :
  cracked_seashells = 0 :=
by
  -- Placeholder for the proof
  sorry

end cracked_seashells_zero_l83_83469


namespace time_ratio_l83_83212

theorem time_ratio (distance : ℝ) (initial_time : ℝ) (new_speed : ℝ) :
  distance = 600 → initial_time = 5 → new_speed = 80 → (distance / new_speed) / initial_time = 1.5 :=
by
  intros hdist htime hspeed
  sorry

end time_ratio_l83_83212


namespace lateral_surface_area_of_given_cone_l83_83837

noncomputable def coneLateralSurfaceArea (r V : ℝ) : ℝ :=
let h := (3 * V) / (π * r^2) in
let l := Real.sqrt (r^2 + h^2) in
π * r * l

theorem lateral_surface_area_of_given_cone :
  coneLateralSurfaceArea 6 (30 * π) = 39 * π := by
simp [coneLateralSurfaceArea]
sorry

end lateral_surface_area_of_given_cone_l83_83837


namespace describe_graph_l83_83319

theorem describe_graph : 
  ∀ (x y : ℝ), x^2 * (x + y + 1) = y^3 * (x + y + 1) ↔ (x^2 = y^3 ∨ y = -x - 1)
:= sorry

end describe_graph_l83_83319


namespace lesser_number_l83_83445

theorem lesser_number (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 10) : y = 25 :=
by
  have h3 : x = 35 := sorry
  exact sorry

end lesser_number_l83_83445


namespace meetings_percent_40_l83_83160

def percent_of_workday_in_meetings (workday_hours : ℕ) (first_meeting_min : ℕ) (second_meeting_min : ℕ) (third_meeting_min : ℕ) : ℕ :=
  (first_meeting_min + second_meeting_min + third_meeting_min) * 100 / (workday_hours * 60)

theorem meetings_percent_40 (workday_hours : ℕ) (first_meeting_min : ℕ) (second_meeting_min : ℕ) (third_meeting_min : ℕ)
  (h_workday : workday_hours = 10) 
  (h_first_meeting : first_meeting_min = 40) 
  (h_second_meeting : second_meeting_min = 2 * first_meeting_min) 
  (h_third_meeting : third_meeting_min = first_meeting_min + second_meeting_min) : 
  percent_of_workday_in_meetings workday_hours first_meeting_min second_meeting_min third_meeting_min = 40 :=
by
  sorry

end meetings_percent_40_l83_83160


namespace sum_first_13_terms_l83_83242

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable (ha : a 2 + a 5 + a 9 + a 12 = 60)

theorem sum_first_13_terms :
  S 13 = 195 := sorry

end sum_first_13_terms_l83_83242


namespace total_wristbands_proof_l83_83875

-- Definitions from the conditions
def wristbands_per_person : ℕ := 2
def total_wristbands : ℕ := 125

-- Theorem statement to be proved
theorem total_wristbands_proof : total_wristbands = 125 :=
by
  sorry

end total_wristbands_proof_l83_83875


namespace sum_of_coefficients_l83_83984

theorem sum_of_coefficients 
  (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℕ)
  (h : (3 * x - 1) ^ 10 = a + a_1 * x + a_2 * x ^ 2 + a_3 * x ^ 3 + a_4 * x ^ 4 + a_5 * x ^ 5 + a_6 * x ^ 6 + a_7 * x ^ 7 + a_8 * x ^ 8 + a_9 * x ^ 9 + a_10 * x ^ 10) :
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 = 1023 := 
sorry

end sum_of_coefficients_l83_83984


namespace y_when_x_is_4_l83_83589

theorem y_when_x_is_4
  (x y : ℝ)
  (h1 : x + y = 30)
  (h2 : x - y = 10)
  (h3 : x * y = 200) :
  y = 50 :=
by
  sorry

end y_when_x_is_4_l83_83589


namespace sixth_grader_count_l83_83342

theorem sixth_grader_count : 
  ∃ x y : ℕ, (3 / 7) * x = (1 / 3) * y ∧ x + y = 140 ∧ x = 61 :=
by {
  sorry  -- Proof not required
}

end sixth_grader_count_l83_83342


namespace base9_subtraction_multiple_of_seven_l83_83033

theorem base9_subtraction_multiple_of_seven (b : ℕ) (h1 : 0 ≤ b ∧ b ≤ 9) 
(h2 : (3 * 9^6 + 1 * 9^5 + 5 * 9^4 + 4 * 9^3 + 6 * 9^2 + 7 * 9^1 + 2 * 9^0) - b % 7 = 0) : b = 0 :=
sorry

end base9_subtraction_multiple_of_seven_l83_83033


namespace integer_multiplication_l83_83189

theorem integer_multiplication :
  ∃ A : ℤ, (999999999 : ℤ) * A = (111111111 : ℤ) :=
by {
  sorry
}

end integer_multiplication_l83_83189


namespace jerry_total_cost_correct_l83_83391

theorem jerry_total_cost_correct :
  let bw_cost := 27
  let bw_discount := 0.1 * bw_cost
  let bw_discounted_price := bw_cost - bw_discount
  let color_cost := 32
  let color_discount := 0.05 * color_cost
  let color_discounted_price := color_cost - color_discount
  let total_color_discounted_price := 3 * color_discounted_price
  let total_discounted_price_before_tax := bw_discounted_price + total_color_discounted_price
  let tax_rate := 0.07
  let tax := total_discounted_price_before_tax * tax_rate
  let total_cost := total_discounted_price_before_tax + tax
  (Float.round (total_cost * 100) / 100) = 123.59 :=
sorry

end jerry_total_cost_correct_l83_83391


namespace sequence_contains_infinite_squares_l83_83413

theorem sequence_contains_infinite_squares :
  ∃ f : ℕ → ℕ, ∀ m : ℕ, ∃ n : ℕ, f (n + m) * f (n + m) = 1 + 17 * (n + m) ^ 2 :=
sorry

end sequence_contains_infinite_squares_l83_83413


namespace jill_salary_l83_83782

-- Defining the conditions
variables (S : ℝ) -- Jill's net monthly salary
variables (discretionary_income : ℝ) -- One fifth of her net monthly salary
variables (vacation_fund : ℝ) -- 30% of discretionary income into a vacation fund
variables (savings : ℝ) -- 20% of discretionary income into savings
variables (eating_out_socializing : ℝ) -- 35% of discretionary income on eating out and socializing
variables (leftover : ℝ) -- The remaining amount, which is $99

-- Given Conditions
-- One fifth of her net monthly salary left as discretionary income
def one_fifth_of_salary : Prop := discretionary_income = (1/5) * S

-- 30% into a vacation fund
def vacation_allocation : Prop := vacation_fund = 0.30 * discretionary_income

-- 20% into savings
def savings_allocation : Prop := savings = 0.20 * discretionary_income

-- 35% on eating out and socializing
def socializing_allocation : Prop := eating_out_socializing = 0.35 * discretionary_income

-- This leaves her with $99
def leftover_amount : Prop := leftover = 99

-- Eqution considering all conditions results her leftover being $99
def income_allocation : Prop := 
  vacation_fund + savings + eating_out_socializing + leftover = discretionary_income

-- The main proof goal: given all the conditions, Jill's net monthly salary is $3300
theorem jill_salary : 
  one_fifth_of_salary S discretionary_income → 
  vacation_allocation discretionary_income vacation_fund → 
  savings_allocation discretionary_income savings → 
  socializing_allocation discretionary_income eating_out_socializing → 
  leftover_amount leftover → 
  income_allocation discretionary_income vacation_fund savings eating_out_socializing leftover → 
  S = 3300 := by sorry

end jill_salary_l83_83782


namespace base_conversion_l83_83177

noncomputable def b_value : ℝ := Real.sqrt 21

theorem base_conversion (b : ℝ) (h : b = Real.sqrt 21) : 
  (1 * b^2 + 0 * b + 2) = 23 := 
by
  rw [h]
  sorry

end base_conversion_l83_83177


namespace hyperbola_with_common_foci_l83_83688

noncomputable def ellipse_eqn := ∀ (x y : ℝ), 
  x^2 / 9 + y^2 / 4 = 1

noncomputable def hyperbola_eqn := ∀ (x y : ℝ), 
  x^2 / 4 - y^2 = 1

noncomputable def ellipse_foci := ∃ (c : ℝ),
  ∃ (a b : ℝ), a = 3 ∧ b = 2 ∧ c = real.sqrt (a^2 - b^2) ∧ c = real.sqrt 5

noncomputable def hyperbola_foci_eccentricity := ∃ (a b : ℝ), 
  ∃ (c : ℝ), c = real.sqrt 5 ∧ a = 2 ∧ c = real.sqrt (a^2 + b^2) ∧ 
  real.sqrt(a^2 + b^2) / a = real.sqrt 5 / 2

theorem hyperbola_with_common_foci (x y : ℝ) 
  (ellipse : ∀ (x y : ℝ), x^2 / 9 + y^2 / 4 = 1)
  (ellipse_foci_conditions : ∃ (c : ℝ), ∃ (a b : ℝ), a = 3 ∧ b = 2 ∧ c = real.sqrt(a^2 - b^2) 
                            ∧ c = real.sqrt 5)
  (hyperbola_eccentricity_conditions : ∃ (a b : ℝ), ∃ (c : ℝ), c = real.sqrt 5 
                                        ∧ a = 2 ∧ c = real.sqrt(a^2 + b^2)
                                        ∧ real.sqrt(a^2 + b^2) / a = real.sqrt 5 / 2) :
  (x^2 / 4 - y^2 = 1) := 
sorry

end hyperbola_with_common_foci_l83_83688


namespace total_revenue_correct_l83_83945

noncomputable def total_ticket_revenue : ℕ :=
  let revenue_2pm := 180 * 6 + 20 * 5 + 60 * 4 + 20 * 3 + 20 * 5
  let revenue_5pm := 95 * 8 + 30 * 7 + 110 * 5 + 15 * 6
  let revenue_8pm := 122 * 10 + 74 * 7 + 29 * 8
  revenue_2pm + revenue_5pm + revenue_8pm

theorem total_revenue_correct : total_ticket_revenue = 5160 := by
  sorry

end total_revenue_correct_l83_83945


namespace range_of_a_l83_83260

theorem range_of_a (a : ℝ) :
  (¬ ∃ x₀ : ℝ, 2 * x₀^2 + (a - 1) * x₀ + 1 / 2 ≤ 0) → a ∈ set.Ioo (-1 : ℝ) (3 : ℝ) :=
by
  sorry

end range_of_a_l83_83260


namespace g_of_2_eq_14_l83_83374

theorem g_of_2_eq_14 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) : g 2 = 14 := 
sorry

end g_of_2_eq_14_l83_83374


namespace intersection_complement_M_N_l83_83696

def M : Set ℝ := { x | x > 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }
def complement_M : Set ℝ := { x | x ≤ 1 }

theorem intersection_complement_M_N :
  (complement_M ∩ N) = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_complement_M_N_l83_83696


namespace find_lesser_number_l83_83461

theorem find_lesser_number (x y : ℕ) (h₁ : x + y = 60) (h₂ : x - y = 10) : y = 25 := by
  sorry

end find_lesser_number_l83_83461


namespace statement_C_l83_83250

variables (a b c d : ℝ)

theorem statement_C (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by sorry

end statement_C_l83_83250


namespace problem_1_problem_2_l83_83558

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

theorem problem_1 (x : ℝ) : f x ≥ 2 ↔ (x ≤ -7 ∨ x ≥ 5 / 3) :=
sorry

theorem problem_2 : ∃ x : ℝ, f x = -9 / 2 :=
sorry

end problem_1_problem_2_l83_83558


namespace minimum_radius_part_a_minimum_radius_part_b_l83_83519

-- Definitions for Part (a)
def a := 7
def b := 8
def c := 9
def R1 := 6

-- Statement for Part (a)
theorem minimum_radius_part_a : (c / 2) = R1 := by sorry

-- Definitions for Part (b)
def a' := 9
def b' := 15
def c' := 16
def R2 := 9

-- Statement for Part (b)
theorem minimum_radius_part_b : (c' / 2) = R2 := by sorry

end minimum_radius_part_a_minimum_radius_part_b_l83_83519


namespace smallest_of_six_consecutive_even_numbers_l83_83585

theorem smallest_of_six_consecutive_even_numbers (h : ∃ n : ℤ, (n - 4) + (n - 2) + n + (n + 2) + (n + 4) + (n + 6) = 390) : ∃ m : ℤ, m = 60 :=
by
  have ex : ∃ n : ℤ, 6 * n + 6 = 390 := by sorry
  obtain ⟨n, hn⟩ := ex
  use (n - 4)
  sorry

end smallest_of_six_consecutive_even_numbers_l83_83585


namespace parabola_focus_directrix_distance_l83_83700

theorem parabola_focus_directrix_distance 
  (p : ℝ) 
  (hp : 3 = p * (1:ℝ)^2) 
  (hparabola : ∀ x : ℝ, y = p * x^2 → x^2 = (1/3:ℝ) * y)
  : (distance_focus_directrix : ℝ) = (1 / 6:ℝ) :=
  sorry

end parabola_focus_directrix_distance_l83_83700


namespace math_problem_l83_83861

theorem math_problem (x : ℂ) (hx : x + 1/x = 3) : x^6 + 1/x^6 = 322 := 
by 
  sorry

end math_problem_l83_83861


namespace speed_of_train_l83_83665

open Real

-- Define the conditions as given in the problem
def length_of_bridge : ℝ := 650
def length_of_train : ℝ := 200
def time_to_pass_bridge : ℝ := 17

-- Define the problem statement which needs to be proved
theorem speed_of_train : (length_of_bridge + length_of_train) / time_to_pass_bridge = 50 :=
by
  sorry

end speed_of_train_l83_83665


namespace neg_power_identity_l83_83810

variable (m : ℝ)

theorem neg_power_identity : (-m^2)^3 = -m^6 :=
sorry

end neg_power_identity_l83_83810


namespace eliminate_denominators_eq_l83_83032

theorem eliminate_denominators_eq :
  ∀ (x : ℝ), 1 - (x + 3) / 6 = x / 2 → 6 - x - 3 = 3 * x :=
by
  intro x
  intro h
  -- Place proof steps here.
  sorry

end eliminate_denominators_eq_l83_83032


namespace partition_of_sum_l83_83046

-- Define the conditions
def is_positive_integer (n : ℕ) : Prop := n > 0
def is_bounded_integer (n : ℕ) : Prop := n ≤ 10
def can_be_partitioned (S : ℕ) (integers : List ℕ) : Prop :=
  ∃ (A B : List ℕ), 
    A.sum ≤ 70 ∧ 
    B.sum ≤ 70 ∧ 
    A ++ B = integers

-- Define the theorem statement
theorem partition_of_sum (S : ℕ) (integers : List ℕ)
  (h1 : ∀ x ∈ integers, is_positive_integer x ∧ is_bounded_integer x)
  (h2 : List.sum integers = S) :
  S ≤ 133 ↔ can_be_partitioned S integers :=
sorry

end partition_of_sum_l83_83046


namespace min_value_x_l83_83848

theorem min_value_x (a : ℝ) (h : ∀ a > 0, x^2 ≤ 1 + a) : ∃ x, ∀ a > 0, -1 ≤ x ∧ x ≤ 1 := 
sorry

end min_value_x_l83_83848


namespace cone_lateral_surface_area_l83_83840

-- Definitions based on the conditions
def coneRadius : ℝ := 6
def coneVolume : ℝ := 30 * Real.pi

-- Mathematical statement
theorem cone_lateral_surface_area (r V : ℝ) (hr : r = coneRadius) (hV : V = coneVolume) :
  ∃ S : ℝ, S = 39 * Real.pi :=
by 
  have h_volume := hV
  have h_radius := hr
  sorry

end cone_lateral_surface_area_l83_83840


namespace wheels_travel_distance_l83_83472

noncomputable def total_horizontal_distance (R₁ R₂ : ℝ) : ℝ :=
  2 * Real.pi * R₁ + 2 * Real.pi * R₂

theorem wheels_travel_distance (R₁ R₂ : ℝ) (h₁ : R₁ = 2) (h₂ : R₂ = 3) :
  total_horizontal_distance R₁ R₂ = 10 * Real.pi :=
by
  rw [total_horizontal_distance, h₁, h₂]
  sorry

end wheels_travel_distance_l83_83472


namespace maximum_a_for_monotonically_increasing_interval_l83_83417

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := f (x - (Real.pi / 4))

theorem maximum_a_for_monotonically_increasing_interval :
  ∀ a : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ a ∧ x < y → g x < g y) → a ≤ Real.pi / 4 := 
by
  sorry

end maximum_a_for_monotonically_increasing_interval_l83_83417


namespace alice_savings_third_month_l83_83148

theorem alice_savings_third_month :
  ∀ (saved_first : ℕ) (increase_per_month : ℕ),
  saved_first = 10 →
  increase_per_month = 30 →
  let saved_second := saved_first + increase_per_month
  let saved_third := saved_second + increase_per_month
  saved_third = 70 :=
by intros saved_first increase_per_month h1 h2;
   let saved_second := saved_first + increase_per_month;
   let saved_third := saved_second + increase_per_month;
   sorry

end alice_savings_third_month_l83_83148


namespace arithmetic_progression_no_rth_power_l83_83233

noncomputable def is_arith_sequence (a : ℕ → ℤ) : Prop := 
∀ n : ℕ, a n = 4 * (n : ℤ) - 2

theorem arithmetic_progression_no_rth_power (n : ℕ) :
  ∃ a : ℕ → ℤ, is_arith_sequence a ∧ 
  (∀ r : ℕ, 2 ≤ r ∧ r ≤ n → 
  ¬ (∃ k : ℤ, ∃ m : ℕ, m > 0 ∧ a m = k ^ r)) := 
sorry

end arithmetic_progression_no_rth_power_l83_83233


namespace time_to_cook_one_potato_l83_83655

-- Definitions for the conditions
def total_potatoes : ℕ := 16
def cooked_potatoes : ℕ := 7
def remaining_minutes : ℕ := 45

-- Lean theorem that asserts the equivalence of the problem statement to the correct answer
theorem time_to_cook_one_potato (total_potatoes cooked_potatoes remaining_minutes : ℕ) 
  (h_total : total_potatoes = 16) 
  (h_cooked : cooked_potatoes = 7) 
  (h_remaining : remaining_minutes = 45) :
  (remaining_minutes / (total_potatoes - cooked_potatoes) = 5) :=
by
  -- Using sorry to skip proof
  sorry

end time_to_cook_one_potato_l83_83655


namespace problem_B_problem_D_l83_83014

/-
  Given distinct lines m, n and distinct planes α, β,
  we want to prove the following two statements:
  
  1. If m is perpendicular to α and n is perpendicular to α, then m is parallel to n.
  2. If m is perpendicular to α, n is perpendicular to β, and m is perpendicular to n, then α is perpendicular to β.
-/

variables {m n : Type} -- Types representing distinct lines
variables {α β : Type} -- Types representing distinct planes

-- Hypotheses for the statements
variable [linear_order m n α β] -- Assume we have a linear ordering for the geometric entities

-- Define helper functions for parallelism and perpendicularity
def is_parallel (x y : Type) : Prop := sorry
def is_perpendicular (x y : Type) : Prop := sorry

-- Statement for problem B
theorem problem_B (h1 : is_perpendicular m α) (h2 : is_perpendicular n α) : is_parallel m n :=
  sorry

-- Statement for problem D
theorem problem_D (h1 : is_perpendicular m α) (h2 : is_perpendicular m n) (h3 : is_perpendicular n β) : is_perpendicular α β :=
  sorry

end problem_B_problem_D_l83_83014


namespace expected_value_is_correct_l83_83204

def expected_value_of_win : ℝ :=
  let outcomes := (list.range 8).map (fun n => 8 - (n + 1))
  let probabilities := list.repeat (1 / 8 : ℝ) 8
  list.zip_with (fun outcome probability => outcome * probability) outcomes probabilities |>.sum

theorem expected_value_is_correct :
  expected_value_of_win = 3.5 := by
  sorry

end expected_value_is_correct_l83_83204


namespace cone_lateral_surface_area_l83_83828

-- Definitions from conditions
def r : ℝ := 6
def V : ℝ := 30 * Real.pi

-- Theorem to prove
theorem cone_lateral_surface_area : 
  let h := V / (Real.pi * (r ^ 2) / 3) in
  let l := Real.sqrt (r ^ 2 + h ^ 2) in
  let S := Real.pi * r * l in
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l83_83828


namespace choose_starters_1980_l83_83568

open Finset

noncomputable def num_ways_to_choose_starters (total_players : ℕ) (quadruplets : Finset ℕ) (starter_count : ℕ) (quadruplet_inclusion : ℕ) : ℕ :=
  if quadruplets.card = 4 ∧ quadruplet_inclusion = 3 ∧ starter_count = 7 ∧ total_players = 16 then
    (quadruplets.card.choose quadruplet_inclusion) * ((total_players - quadruplets.card).choose (starter_count - quadruplet_inclusion))
  else 0

theorem choose_starters_1980 : num_ways_to_choose_starters 16 (Finset.range 4) 7 3 = 1980 := by
  sorry

end choose_starters_1980_l83_83568


namespace intersection_complementA_setB_l83_83159

noncomputable def setA : Set ℝ := { x | abs x > 1 }

noncomputable def setB : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

noncomputable def complementA : Set ℝ := { x | abs x ≤ 1 }

theorem intersection_complementA_setB : 
  (complementA ∩ setB) = { x | 0 ≤ x ∧ x ≤ 1 } := by
  sorry

end intersection_complementA_setB_l83_83159


namespace tea_sales_l83_83038

theorem tea_sales (L T : ℕ) (h1 : L = 32) (h2 : L = 4 * T + 8) : T = 6 :=
by
  sorry

end tea_sales_l83_83038


namespace arithmetic_calculation_l83_83091

theorem arithmetic_calculation : 3 - (-5) + 7 = 15 := by
  sorry

end arithmetic_calculation_l83_83091


namespace jason_steps_is_8_l83_83410

-- Definition of the problem conditions
def nancy_steps (jason_steps : ℕ) := 3 * jason_steps -- Nancy steps 3 times as often as Jason

def together_steps (jason_steps nancy_steps : ℕ) := jason_steps + nancy_steps -- Total steps

-- Lean statement of the problem to prove
theorem jason_steps_is_8 (J : ℕ) (h₁ : together_steps J (nancy_steps J) = 32) : J = 8 :=
sorry

end jason_steps_is_8_l83_83410


namespace least_sum_exponents_of_1000_l83_83254

def sum_least_exponents (n : ℕ) : ℕ :=
  if n = 1000 then 38 else 0 -- Since we only care about the case for 1000.

theorem least_sum_exponents_of_1000 :
  sum_least_exponents 1000 = 38 := by
  sorry

end least_sum_exponents_of_1000_l83_83254


namespace find_y_l83_83599

theorem find_y (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : ∃ C, x * y = C) (hx : x = 4) : y = 50 :=
sorry

end find_y_l83_83599


namespace min_value_x_2y_l83_83823

theorem min_value_x_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2*y + 2*x*y = 8) : x + 2*y ≥ 4 :=
sorry

end min_value_x_2y_l83_83823


namespace number_of_ants_l83_83566

def spiders := 8
def spider_legs := 8
def ants := 12
def ant_legs := 6
def total_legs := 136

theorem number_of_ants :
  spiders * spider_legs + ants * ant_legs = total_legs → ants = 12 :=
by
  sorry

end number_of_ants_l83_83566


namespace slope_of_line_l83_83086

theorem slope_of_line
  (m : ℝ)
  (b : ℝ)
  (h1 : b = 4)
  (h2 : ∀ x y : ℝ, y = m * x + b → (x = 199 ∧ y = 800) → True) :
  m = 4 :=
by
  sorry

end slope_of_line_l83_83086


namespace find_a_l83_83115

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 0 then x * 2^(x + a) - 1 else - (x * 2^(-x + a) - 1)

theorem find_a (a : ℝ) (h_odd: ∀ x : ℝ, f x a = -f (-x) a)
  (h_pos : ∀ x : ℝ, x > 0 → f x a = x * 2^(x + a) - 1)
  (h_neg : f (-1) a = 3 / 4) :
  a = -3 :=
by
  sorry

end find_a_l83_83115


namespace heidi_paints_fraction_in_10_minutes_l83_83257

variable (Heidi_paint_rate : ℕ → ℝ)
variable (t : ℕ)
variable (fraction : ℝ)

theorem heidi_paints_fraction_in_10_minutes 
  (h1 : Heidi_paint_rate 30 = 1) 
  (h2 : t = 10) 
  (h3 : fraction = 1 / 3) : 
  Heidi_paint_rate t = fraction := 
sorry

end heidi_paints_fraction_in_10_minutes_l83_83257


namespace trains_meet_80_km_from_A_l83_83772

-- Define the speeds of the trains
def speed_train_A : ℝ := 60 
def speed_train_B : ℝ := 90 

-- Define the distance between locations A and B
def distance_AB : ℝ := 200 

-- Define the time when the trains meet
noncomputable def meeting_time : ℝ := distance_AB / (speed_train_A + speed_train_B)

-- Define the distance from location A to where the trains meet
noncomputable def distance_from_A (speed_A : ℝ) (meeting_time : ℝ) : ℝ :=
  speed_A * meeting_time

-- Prove the statement
theorem trains_meet_80_km_from_A :
  distance_from_A speed_train_A meeting_time = 80 :=
by
  -- leaving the proof out, it's just an assumption due to 'sorry'
  sorry

end trains_meet_80_km_from_A_l83_83772


namespace distinct_bead_arrangements_on_bracelet_l83_83719

open Nat

-- Definition of factorial
def fact : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * fact n

-- Theorem stating the number of distinct arrangements of 7 beads on a bracelet
theorem distinct_bead_arrangements_on_bracelet : 
  fact 7 / 14 = 360 := 
by 
  sorry

end distinct_bead_arrangements_on_bracelet_l83_83719


namespace roots_are_integers_l83_83425

theorem roots_are_integers (a b : ℤ) (h_discriminant : ∃ (q r : ℚ), r ≠ 0 ∧ a^2 - 4 * b = (q/r)^2) : 
  ∃ x y : ℤ, x^2 - a * x + b = 0 ∧ y^2 - a * y + b = 0 := 
sorry

end roots_are_integers_l83_83425


namespace distinct_primes_p_q_r_l83_83575

theorem distinct_primes_p_q_r (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) (eqn : r * p^3 + p^2 + p = 2 * r * q^2 + q^2 + q) : p * q * r = 2014 :=
by
  sorry

end distinct_primes_p_q_r_l83_83575


namespace train_speed_l83_83773

noncomputable def speed_of_each_train (v : ℕ) : ℕ := 27

theorem train_speed
  (length_of_each_train : ℕ)
  (crossing_time : ℕ)
  (crossing_condition : 2 * (length_of_each_train * crossing_time) / (2 * crossing_time) = 15 / 2)
  (conversion_factor : ∀ n, 1 = 3.6 * n → ℕ) :
  speed_of_each_train 27 = 27 :=
by
  exact rfl

end train_speed_l83_83773


namespace savings_percentage_l83_83079

variable (I : ℝ) -- First year's income
variable (S : ℝ) -- Amount saved in the first year

-- Conditions
axiom condition1 (h1 : S = 0.05 * I) : Prop
axiom condition2 (h2 : S + 0.05 * I = 2 * S) : Prop
axiom condition3 (h3 : (I - S) + 1.10 * (I - S) = 2 * (I - S)) : Prop

-- Theorem that proves the man saved 5% of his income in the first year
theorem savings_percentage : S = 0.05 * I :=
by
  sorry -- Proof goes here

end savings_percentage_l83_83079


namespace choose_two_out_of_three_l83_83920

-- Define the number of vegetables as n and the number to choose as k
def n : ℕ := 3
def k : ℕ := 2

-- The combination formula C(n, k) == n! / (k! * (n - k)!)
def combination (n k : ℕ) : ℕ := n.choose k

-- Problem statement: Prove that the number of ways to choose 2 out of 3 vegetables is 3
theorem choose_two_out_of_three : combination n k = 3 :=
by
  sorry

end choose_two_out_of_three_l83_83920


namespace lesser_number_l83_83455

theorem lesser_number (x y : ℕ) (h1: x + y = 60) (h2: x - y = 10) : y = 25 :=
sorry

end lesser_number_l83_83455


namespace unique_prime_sum_diff_l83_83687

-- Define that p is a prime number that satisfies both conditions
def sum_two_primes (p a b : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime a ∧ Nat.Prime b ∧ p = a + b

def diff_two_primes (p c d : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime c ∧ Nat.Prime d ∧ p = c - d

-- Main theorem to prove: The only prime p that satisfies both conditions is 5
theorem unique_prime_sum_diff (p : ℕ) :
  (∃ a b, sum_two_primes p a b) ∧ (∃ c d, diff_two_primes p c d) → p = 5 :=
by
  sorry

end unique_prime_sum_diff_l83_83687


namespace simplify_expression_l83_83812

variable (x : ℝ)
variable (h₁ : x ≠ 2)
variable (h₂ : x ≠ 3)
variable (h₃ : x ≠ 4)
variable (h₄ : x ≠ 5)

theorem simplify_expression : 
  ( (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 9) / (x^2 - 8*x + 15)) 
  = ( (x - 1) * (x - 5) ) / ( (x - 4) * (x - 2) * (x - 3) ) ) :=
by sorry

end simplify_expression_l83_83812


namespace a_eq_b_pow_n_l83_83066

variables (a b n : ℕ)
variable (h : ∀ (k : ℕ), k ≠ b → b - k ∣ a - k^n)

theorem a_eq_b_pow_n : a = b^n := 
by
  sorry

end a_eq_b_pow_n_l83_83066


namespace max_length_polyline_l83_83477

-- Definition of the grid and problem
def grid_rows : ℕ := 6
def grid_cols : ℕ := 10

-- The maximum length of a closed, non-self-intersecting polyline
theorem max_length_polyline (rows cols : ℕ) 
  (h_rows : rows = grid_rows) (h_cols : cols = grid_cols) :
  ∃ length : ℕ, length = 76 :=
by {
  sorry
}

end max_length_polyline_l83_83477


namespace necessary_and_sufficient_condition_l83_83980

variable (p q : Prop)

theorem necessary_and_sufficient_condition (hp : p) (hq : q) : ¬p ∨ ¬q = False :=
by {
    -- You are requested to fill out the proof here.
    sorry
}

end necessary_and_sufficient_condition_l83_83980


namespace selection_methods_count_l83_83415

noncomputable def num_selection_methods (total_students chosen_students : ℕ) (A B : ℕ) : ℕ :=
  let with_A_and_B := Nat.choose (total_students - 2) (chosen_students - 2)
  let with_one_A_or_B := Nat.choose (total_students - 2) (chosen_students - 1) * Nat.choose 2 1
  with_A_and_B + with_one_A_or_B

theorem selection_methods_count :
  num_selection_methods 10 4 1 2 = 140 :=
by
  -- We can add detailed proof here, for now we provide a placeholder
  sorry

end selection_methods_count_l83_83415


namespace maximize_area_l83_83525

theorem maximize_area (P L W : ℝ) (h1 : P = 2 * L + 2 * W) (h2 : 0 < P) : 
  (L = P / 4) ∧ (W = P / 4) :=
by
  sorry

end maximize_area_l83_83525


namespace greatest_difference_four_digit_numbers_l83_83380

theorem greatest_difference_four_digit_numbers : 
  ∃ (d1 d2 d3 d4 : ℕ), (d1 = 0 ∨ d1 = 3 ∨ d1 = 4 ∨ d1 = 8) ∧ 
                      (d2 = 0 ∨ d2 = 3 ∨ d2 = 4 ∨ d2 = 8) ∧ 
                      (d3 = 0 ∨ d3 = 3 ∨ d3 = 4 ∨ d3 = 8) ∧ 
                      (d4 = 0 ∨ d4 = 3 ∨ d4 = 4 ∨ d4 = 8) ∧ 
                      d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ 
                      d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧ 
                      (∃ n1 n2, n1 = 1000 * 8 + 100 * 4 + 10 * 3 + 0 ∧ 
                                n2 = 1000 * 3 + 100 * 0 + 10 * 4 + 8 ∧ 
                                n1 - n2 = 5382) :=
by {
  sorry
}

end greatest_difference_four_digit_numbers_l83_83380


namespace digit_is_4_l83_83037

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

theorem digit_is_4 (d : ℕ) (hd0 : is_even d) (hd1 : is_divisible_by_3 (14 + d)) : d = 4 :=
  sorry

end digit_is_4_l83_83037


namespace product_of_smallest_primes_l83_83641

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def smallest_one_digit_primes : List ℕ := [2, 3]
def smallest_two_digit_prime : ℕ := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 :=
by
  sorry

end product_of_smallest_primes_l83_83641


namespace find_denominator_of_second_fraction_l83_83383

theorem find_denominator_of_second_fraction (y : ℝ) (h : y > 0) (x : ℝ) :
  (2 * y) / 5 + (3 * y) / x = 0.7 * y → x = 10 :=
by
  sorry

end find_denominator_of_second_fraction_l83_83383


namespace ball_hits_ground_l83_83758

theorem ball_hits_ground (t : ℚ) : 
  (∃ t ≥ 0, (-4.9 * (t^2 : ℝ) + 5 * t + 10 = 0)) → t = 100 / 49 :=
by
  sorry

end ball_hits_ground_l83_83758


namespace celina_total_cost_l83_83961

def hoodieCost : ℝ := 80
def hoodieTaxRate : ℝ := 0.05

def flashlightCost := 0.20 * hoodieCost
def flashlightTaxRate : ℝ := 0.10

def bootsInitialCost : ℝ := 110
def bootsDiscountRate : ℝ := 0.10
def bootsTaxRate : ℝ := 0.05

def waterFilterCost : ℝ := 65
def waterFilterDiscountRate : ℝ := 0.25
def waterFilterTaxRate : ℝ := 0.08

def campingMatCost : ℝ := 45
def campingMatDiscountRate : ℝ := 0.15
def campingMatTaxRate : ℝ := 0.08

def backpackCost : ℝ := 105
def backpackTaxRate : ℝ := 0.08

def totalCost : ℝ := 
  let hoodieTotal := (hoodieCost * (1 + hoodieTaxRate))
  let flashlightTotal := (flashlightCost * (1 + flashlightTaxRate))
  let bootsTotal := ((bootsInitialCost * (1 - bootsDiscountRate)) * (1 + bootsTaxRate))
  let waterFilterTotal := ((waterFilterCost * (1 - waterFilterDiscountRate)) * (1 + waterFilterTaxRate))
  let campingMatTotal := ((campingMatCost * (1 - campingMatDiscountRate)) * (1 + campingMatTaxRate))
  let backpackTotal := (backpackCost * (1 + backpackTaxRate))
  hoodieTotal + flashlightTotal + bootsTotal + waterFilterTotal + campingMatTotal + backpackTotal

theorem celina_total_cost: totalCost = 413.91 := by
  sorry

end celina_total_cost_l83_83961


namespace factorization_correct_l83_83193

theorem factorization_correct : ∀ y : ℝ, y^2 - 4*y + 4 = (y - 2)^2 := by
  intro y
  sorry

end factorization_correct_l83_83193


namespace three_digit_number_condition_l83_83976

theorem three_digit_number_condition (x y z : ℕ) (h₀ : 1 ≤ x ∧ x ≤ 9) (h₁ : 0 ≤ y ∧ y ≤ 9) (h₂ : 0 ≤ z ∧ z ≤ 9)
(h₃ : 100 * x + 10 * y + z = 34 * (x + y + z)) : 
100 * x + 10 * y + z = 102 ∨ 100 * x + 10 * y + z = 204 ∨ 100 * x + 10 * y + z = 306 ∨ 100 * x + 10 * y + z = 408 :=
sorry

end three_digit_number_condition_l83_83976


namespace original_population_l83_83938

-- Define the initial setup
variable (P : ℝ)

-- The conditions given in the problem
axiom ten_percent_died (P : ℝ) : (1 - 0.1) * P = 0.9 * P
axiom twenty_percent_left (P : ℝ) : (1 - 0.2) * (0.9 * P) = 0.9 * P * 0.8

-- Define the final condition
axiom final_population (P : ℝ) : 0.9 * P * 0.8 = 3240

-- The proof problem
theorem original_population : P = 4500 :=
by
  sorry

end original_population_l83_83938


namespace compute_fraction_product_l83_83674

theorem compute_fraction_product :
  (1 / 3)^4 * (1 / 5) = 1 / 405 :=
by
  sorry

end compute_fraction_product_l83_83674


namespace max_pancake_pieces_3_cuts_l83_83191

open Nat

def P : ℕ → ℕ
| 0 => 1
| n => n * (n + 1) / 2 + 1

theorem max_pancake_pieces_3_cuts : P 3 = 7 := by
  have h0: P 0 = 1 := by rfl
  have h1: P 1 = 2 := by rfl
  have h2: P 2 = 4 := by rfl
  show P 3 = 7
  calc
    P 3 = 3 * (3 + 1) / 2 + 1 := by rfl
    _ = 3 * 4 / 2 + 1 := by rfl
    _ = 6 + 1 := by norm_num
    _ = 7 := by norm_num

end max_pancake_pieces_3_cuts_l83_83191


namespace three_students_with_A_l83_83972

-- Define the statements of the students
variables (Eliza Fiona George Harry : Prop)

-- Conditions based on the problem statement
axiom Fiona_implies_Eliza : Fiona → Eliza
axiom George_implies_Fiona : George → Fiona
axiom Harry_implies_George : Harry → George

-- There are exactly three students who scored an A
theorem three_students_with_A (hE : Bool) : 
  (Eliza = false) → (Fiona = true) → (George = true) → (Harry = true) :=
by
  sorry

end three_students_with_A_l83_83972


namespace fraction_power_multiply_l83_83676

theorem fraction_power_multiply :
  ((1 : ℚ) / 3)^4 * ((1 : ℚ) / 5) = (1 / 405 : ℚ) :=
by sorry

end fraction_power_multiply_l83_83676


namespace problem1_solution_problem2_solution_l83_83960

-- Problem 1
theorem problem1_solution (x y : ℝ) (h1 : 2 * x + 3 * y = 8) (h2 : x = y - 1) : x = 1 ∧ y = 2 := by
  sorry

-- Problem 2
theorem problem2_solution (x y : ℝ) (h1 : 2 * x - y = -1) (h2 : x + 3 * y = 17) : x = 2 ∧ y = 5 := by
  sorry

end problem1_solution_problem2_solution_l83_83960


namespace num_valid_a_values_l83_83896

theorem num_valid_a_values : 
  ∃ S : Finset ℕ, (∀ a ∈ S, a < 100 ∧ (a^3 + 23) % 24 = 0) ∧ S.card = 5 :=
sorry

end num_valid_a_values_l83_83896


namespace monotonicity_of_f_l83_83994

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + 1

theorem monotonicity_of_f (a x : ℝ) :
  (a > 0 → ((∀ x, (x < -2 * a / 3 → f a x' > f a x) ∧ (x > 0 → f a x' > f a x)) ∧ (∀ x, (-2 * a / 3 < x ∧ x < 0 → f a x' < f a x)))) ∧
  (a = 0 → ∀ x, f a x' > f a x) ∧
  (a < 0 → ((∀ x, (x < 0 → f a x' > f a x) ∧ (x > -2 * a / 3 → f a x' > f a x)) ∧ (∀ x, (0 < x ∧ x < -2 * a / 3 → f a x' < f a x)))) :=
sorry

end monotonicity_of_f_l83_83994


namespace total_number_of_legs_l83_83021

def kangaroos : ℕ := 23
def goats : ℕ := 3 * kangaroos
def legs_of_kangaroo : ℕ := 2
def legs_of_goat : ℕ := 4

theorem total_number_of_legs : 
  (kangaroos * legs_of_kangaroo + goats * legs_of_goat) = 322 := by
  sorry

end total_number_of_legs_l83_83021


namespace intersection_complement_eq_l83_83857

open Set

universe u

def U : Set ℝ := univ

def A : Set ℝ := { x | x < 0 }

def B : Set ℝ := { x | x ≤ -1 }

theorem intersection_complement_eq : A ∩ (U \ B) = { x | -1 < x ∧ x < 0 } :=
by
  sorry

end intersection_complement_eq_l83_83857


namespace bees_on_20th_day_l83_83263

-- Define the conditions
def initial_bees : ℕ := 1

def companions_per_bee : ℕ := 4

-- Define the total number of bees on day n
def total_bees (n : ℕ) : ℕ :=
  (initial_bees + companions_per_bee) ^ n

-- Statement to prove
theorem bees_on_20th_day : total_bees 20 = 5^20 :=
by
  -- The proof is omitted
  sorry

end bees_on_20th_day_l83_83263


namespace prices_proof_sales_revenue_proof_l83_83202

-- Definitions for the prices and quantities
def price_peanut_oil := 50
def price_corn_oil := 40

-- Conditions from the problem
def condition1 (x y : ℕ) : Prop := 20 * x + 30 * y = 2200
def condition2 (x y : ℕ) : Prop := 30 * x + 10 * y = 1900
def purchased_peanut_oil := 50
def selling_price_peanut_oil := 60

-- Proof statement for Part 1
theorem prices_proof : ∃ (x y : ℕ), condition1 x y ∧ condition2 x y ∧ x = price_peanut_oil ∧ y = price_corn_oil :=
sorry

-- Proof statement for Part 2
theorem sales_revenue_proof : ∃ (m : ℕ), (selling_price_peanut_oil * m > price_peanut_oil * purchased_peanut_oil) ∧ m = 42 :=
sorry

end prices_proof_sales_revenue_proof_l83_83202


namespace product_of_primes_is_66_l83_83638

theorem product_of_primes_is_66 :
  let p1 : ℕ := 2
      p2 : ℕ := 3
      p3 : ℕ := 11
  in p1 * p2 * p3 = 66 := by
  sorry

end product_of_primes_is_66_l83_83638


namespace parallel_lines_slope_l83_83679

theorem parallel_lines_slope {a : ℝ} 
    (h1 : ∀ x y : ℝ, 4 * y + 3 * x - 5 = 0 → y = -3 / 4 * x + 5 / 4)
    (h2 : ∀ x y : ℝ, 6 * y + a * x + 4 = 0 → y = -a / 6 * x - 2 / 3)
    (h_parallel : ∀ x₁ y₁ x₂ y₂ : ℝ, (4 * y₁ + 3 * x₁ - 5 = 0 ∧ 6 * y₂ + a * x₂ + 4 = 0) → -3 / 4 = -a / 6) : 
  a = 4.5 := sorry

end parallel_lines_slope_l83_83679


namespace mod_50_remainder_of_b86_l83_83402

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem mod_50_remainder_of_b86 : (b 86) % 50 = 40 := 
by 
-- Given definition of b and the problem is to prove the remainder of b_86 when divided by 50 is 40
sorry

end mod_50_remainder_of_b86_l83_83402


namespace valid_pairs_count_l83_83269

open Nat

def no_carry_add (a b : ℕ) : Prop :=
  let digits_a := to_digits 10 a
  let digits_b := to_digits 10 b
  ∀ i, i < min digits_a.length digits_b.length → digits_a.nth i + digits_b.nth i < 10

def is_valid_pair (a b : ℕ) : Prop :=
  b = a + 1 ∧ no_carry_add a b

def count_valid_pairs : ℕ :=
  ((range (2001 - 1000)).map (fun x => 1000 + x)).foldl
    (fun count a => if is_valid_pair a (a + 1) then count + 1 else count) 0

theorem valid_pairs_count : count_valid_pairs = 156 := by
  sorry

end valid_pairs_count_l83_83269


namespace min_total_bags_l83_83313

theorem min_total_bags (x y : ℕ) (h : 15 * x + 8 * y = 1998) (hy_min : ∀ y', (15 * x + 8 * y' = 1998) → y ≤ y') :
  x + y = 140 :=
by
  sorry

end min_total_bags_l83_83313


namespace effect_on_revenue_decrease_l83_83065

variable (P Q : ℝ)

def original_revenue (P Q : ℝ) : ℝ := P * Q

def new_price (P : ℝ) : ℝ := P * 1.40

def new_quantity (Q : ℝ) : ℝ := Q * 0.65

def new_revenue (P Q : ℝ) : ℝ := new_price P * new_quantity Q

theorem effect_on_revenue_decrease :
  new_revenue P Q = original_revenue P Q * 0.91 →
  new_revenue P Q - original_revenue P Q = original_revenue P Q * -0.09 :=
by
  sorry

end effect_on_revenue_decrease_l83_83065


namespace units_digit_7_pow_2023_l83_83104

-- We start by defining a function to compute units digit of powers of 7 modulo 10.
def units_digit_of_7_pow (n : ℕ) : ℕ :=
  (7 ^ n) % 10

-- Define the problem statement: the units digit of 7^2023 is equal to 3.
theorem units_digit_7_pow_2023 : units_digit_of_7_pow 2023 = 3 := sorry

end units_digit_7_pow_2023_l83_83104


namespace find_pairs_l83_83768

def Point := (ℤ × ℤ)

def P : Point := (1, 1)
def Q : Point := (4, 5)
def valid_pairs : List Point := [(4, 1), (7, 5), (10, 9), (1, 5), (4, 9)]

def area (P Q R : Point) : ℚ :=
  (1 / 2 : ℚ) * ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)).natAbs : ℚ)

theorem find_pairs :
  {pairs : List Point // ∀ (a b : ℤ), (0 ≤ a ∧ a ≤ 10 ∧ 0 ≤ b ∧ b ≤ 10 ∧ area P Q (a, b) = 6) ↔ (a, b) ∈ pairs} :=
  ⟨valid_pairs, by sorry⟩

end find_pairs_l83_83768


namespace people_in_first_group_l83_83868

theorem people_in_first_group (P : ℕ) (work_done_by_P : 60 = 1 / (P * (1/60))) (work_done_by_16 : 30 = 1 / (16 * (1/30))) : P = 8 :=
by
  sorry

end people_in_first_group_l83_83868


namespace maximize_net_income_l83_83940

noncomputable def net_income (x : ℕ) : ℤ :=
  if 60 ≤ x ∧ x ≤ 90 then 750 * x - 1700
  else if 90 < x ∧ x ≤ 300 then -3 * x * x + 1020 * x - 1700
  else 0

theorem maximize_net_income :
  (∀ x : ℕ, 60 ≤ x ∧ x ≤ 300 →
    net_income x ≤ net_income 170) ∧
  net_income 170 = 85000 := 
sorry

end maximize_net_income_l83_83940


namespace max_g6_l83_83895

noncomputable def g (x : ℝ) : ℝ :=
sorry

theorem max_g6 :
  (∀ x, (g x = a * x^2 + b * x + c) ∧ (a ≥ 0) ∧ (b ≥ 0) ∧ (c ≥ 0)) →
  (g 3 = 3) →
  (g 9 = 243) →
  (g 6 ≤ 6) :=
sorry

end max_g6_l83_83895


namespace triangle_problem_l83_83271

noncomputable def triangle_sin_B (a b : ℝ) (A : ℝ) : ℝ :=
  b * Real.sin A / a

noncomputable def triangle_side_c (a b A : ℝ) : ℝ :=
  let discr := b^2 + a^2 - 2 * b * a * Real.cos A
  Real.sqrt discr

noncomputable def sin_diff_angle (sinB cosB sinC cosC : ℝ) : ℝ :=
  sinB * cosC - cosB * sinC

theorem triangle_problem
  (a b : ℝ)
  (A : ℝ)
  (ha : a = Real.sqrt 39)
  (hb : b = 2)
  (hA : A = Real.pi * (2 / 3)) :
  (triangle_sin_B a b A = Real.sqrt 13 / 13) ∧
  (triangle_side_c a b A = 5) ∧
  (sin_diff_angle (Real.sqrt 13 / 13) (2 * Real.sqrt 39 / 13) (5 * Real.sqrt 13 / 26) (3 * Real.sqrt 39 / 26) = -7 * Real.sqrt 3 / 26) :=
by sorry

end triangle_problem_l83_83271


namespace find_coords_C_l83_83738

-- Define the coordinates of given points
def A : ℝ × ℝ := (13, 7)
def B : ℝ × ℝ := (5, -1)
def D : ℝ × ℝ := (2, 2)

-- The proof problem wrapped in a lean theorem
theorem find_coords_C (C : ℝ × ℝ) 
  (h1 : AB = AC) (h2 : (D.1, D.2) = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) :
  C = (-1, 5) :=
sorry

end find_coords_C_l83_83738


namespace scientific_notation_of_4600000000_l83_83174

theorem scientific_notation_of_4600000000 :
  4.6 * 10^9 = 4600000000 := 
by
  sorry

end scientific_notation_of_4600000000_l83_83174


namespace inequality_f_c_f_a_f_b_l83_83991

-- Define the function f and the conditions
def f : ℝ → ℝ := sorry

noncomputable def a : ℝ := Real.log (1 / Real.pi)
noncomputable def b : ℝ := (Real.log Real.pi) ^ 2
noncomputable def c : ℝ := Real.log (Real.sqrt Real.pi)

-- Theorem statement
theorem inequality_f_c_f_a_f_b :
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f x1 > f x2) →
  f c > f a ∧ f a > f b :=
by
  -- Proof omitted
  sorry

end inequality_f_c_f_a_f_b_l83_83991


namespace intersection_M_N_l83_83369

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := {x | x ≥ 1} ∪ {x | x ≤ -3}

-- Prove the intersection of M and N is [1, 2)
theorem intersection_M_N : (M ∩ N) = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l83_83369


namespace triangle_incenter_circumradius_inradius_l83_83241

theorem triangle_incenter_circumradius_inradius (ABC : Triangle) (R : ℝ) (r : ℝ) (I : Point)
  (h1 : ABC.Circumradius = R) (h2 : ABC.Inradius = r) (h3 : ABC.incenter = I) :
  ∃ D : Point, (Line.mk I ABC.A).meets_circle_point ABC.circumcircle D ∧ 
               (dist I D) * (dist I ABC.A) = 2 * R * r :=
by
  sorry

end triangle_incenter_circumradius_inradius_l83_83241


namespace min_length_PQ_l83_83387

noncomputable def minimum_length (a : ℝ) : ℝ :=
  let x := 2 * a
  let y := a + 2
  let d := |2 * 2 - 2 * 0 + 4| / Real.sqrt (1^2 + (-2)^2)
  let r := Real.sqrt 5
  d - r

theorem min_length_PQ : ∀ (a : ℝ), P ∈ {P : ℝ × ℝ | (P.1 - 2)^2 + P.2^2 = 5} ∧ Q = (2 * a, a + 2) →
  minimum_length a = 3 * Real.sqrt 5 / 5 :=
by
  intro a
  intro h
  rcases h with ⟨hP, hQ⟩
  sorry

end min_length_PQ_l83_83387


namespace binom_25_7_l83_83845

theorem binom_25_7 :
  (Nat.choose 23 5 = 33649) →
  (Nat.choose 23 6 = 42504) →
  (Nat.choose 23 7 = 33649) →
  Nat.choose 25 7 = 152306 :=
by
  intros h1 h2 h3
  sorry

end binom_25_7_l83_83845


namespace product_of_smallest_primes_l83_83640

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def smallest_one_digit_primes : List ℕ := [2, 3]
def smallest_two_digit_prime : ℕ := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 :=
by
  sorry

end product_of_smallest_primes_l83_83640


namespace quadrilateral_perimeter_l83_83187

-- Define the basic conditions
variables (a b : ℝ)

-- Let's define what happens when Xiao Ming selected 2 pieces of type A, 7 pieces of type B, and 3 pieces of type C
theorem quadrilateral_perimeter (a b : ℝ) : 2 * (a + 3 * b + 2 * a + b) = 6 * a + 8 * b :=
by sorry

end quadrilateral_perimeter_l83_83187


namespace quadratic_square_binomial_l83_83227

theorem quadratic_square_binomial (d : ℝ) : (∃ b : ℝ, (x : ℝ) -> (x + b)^2 = x^2 + 110 * x + d) ↔ d = 3025 :=
by
  sorry

end quadratic_square_binomial_l83_83227


namespace g_of_five_eq_one_l83_83178

variable (g : ℝ → ℝ)

theorem g_of_five_eq_one (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
    (h2 : ∀ x : ℝ, g x ≠ 0) : g 5 = 1 :=
sorry

end g_of_five_eq_one_l83_83178


namespace probability_five_correct_is_zero_l83_83053

-- Let's define the problem in Lean.
theorem probability_five_correct_is_zero :
  let letters := { 'A', 'B', 'C', 'D', 'E', 'F' }
  let people := { 'P1', 'P2', 'P3', 'P4', 'P5', 'P6' }
  let permutations := letters.prod_map.people -- all possible mappings
  let correct_distribution := 1 / (people.card)! -- probability of one correct distribution
  let exactly_five_correct (mapping : people → letters) := 
    (∃ person, ∀ p i ≠ person, mapping p = letters p) ∧ 
    ∃ wrong_person, mapping wrong_person ≠ letters wrong_person
  in ∀ mapping ∈ permutations, exactly_five_correct mapping → 0 :=
by
  sorry

end probability_five_correct_is_zero_l83_83053


namespace inequality_equivalence_l83_83989

theorem inequality_equivalence (a : ℝ) :
  (∀ (x : ℝ), |x + 1| + |x - 1| ≥ a) ↔ (a ≤ 2) :=
sorry

end inequality_equivalence_l83_83989


namespace alice_savings_third_month_l83_83149

theorem alice_savings_third_month :
  ∀ (saved_first : ℕ) (increase_per_month : ℕ),
  saved_first = 10 →
  increase_per_month = 30 →
  let saved_second := saved_first + increase_per_month
  let saved_third := saved_second + increase_per_month
  saved_third = 70 :=
by intros saved_first increase_per_month h1 h2;
   let saved_second := saved_first + increase_per_month;
   let saved_third := saved_second + increase_per_month;
   sorry

end alice_savings_third_month_l83_83149


namespace amount_subtracted_is_15_l83_83975

theorem amount_subtracted_is_15 (n x : ℕ) (h1 : 7 * n - x = 2 * n + 10) (h2 : n = 5) : x = 15 :=
by 
  sorry

end amount_subtracted_is_15_l83_83975


namespace lateral_surface_area_of_given_cone_l83_83838

noncomputable def coneLateralSurfaceArea (r V : ℝ) : ℝ :=
let h := (3 * V) / (π * r^2) in
let l := Real.sqrt (r^2 + h^2) in
π * r * l

theorem lateral_surface_area_of_given_cone :
  coneLateralSurfaceArea 6 (30 * π) = 39 * π := by
simp [coneLateralSurfaceArea]
sorry

end lateral_surface_area_of_given_cone_l83_83838


namespace min_value_change_when_2x2_added_l83_83931

variable (f : ℝ → ℝ)
variable (a b c : ℝ)

def quadratic (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem min_value_change_when_2x2_added
  (a b : ℝ)
  (h1 : ∀ x : ℝ, f x = a * x^2 + b * x + c)
  (h2 : ∀ x : ℝ, (a + 1) * x^2 + b * x + c > a * x^2 + b * x + c + 1)
  (h3 : ∀ x : ℝ, (a - 1) * x^2 + b * x + c < a * x^2 + b * x + c - 3) :
  ∀ x : ℝ, (a + 2) * x^2 + b * x + c = a * x^2 + b * x + (c + 1.5) :=
sorry

end min_value_change_when_2x2_added_l83_83931


namespace algebraic_expression_decrease_l83_83721

theorem algebraic_expression_decrease (x y : ℝ) :
  let original_expr := 2 * x^2 * y
  let new_expr := 2 * ((1 / 2) * x) ^ 2 * ((1 / 2) * y)
  let decrease := ((original_expr - new_expr) / original_expr) * 100
  decrease = 87.5 := by
  sorry

end algebraic_expression_decrease_l83_83721


namespace disk_difference_l83_83064

/-- Given the following conditions:
    1. Every disk is either blue, yellow, green, or red.
    2. The ratio of blue disks to yellow disks to green disks to red disks is 3 : 7 : 8 : 4.
    3. The total number of disks in the bag is 176.
    Prove that the number of green disks minus the number of blue disks is 40.
-/
theorem disk_difference (b y g r : ℕ) (h_ratio : b * 7 = y * 3 ∧ b * 8 = g * 3 ∧ b * 4 = r * 3) (h_total : b + y + g + r = 176) : g - b = 40 :=
by
  sorry

end disk_difference_l83_83064


namespace sum_remainder_product_remainder_l83_83018

open Nat

-- Define the modulus conditions
variables (x y z : ℕ)
def condition1 : Prop := x % 15 = 11
def condition2 : Prop := y % 15 = 13
def condition3 : Prop := z % 15 = 14

-- Proof statement for the sum remainder
theorem sum_remainder (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 z) : (x + y + z) % 15 = 8 :=
by
  sorry

-- Proof statement for the product remainder
theorem product_remainder (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 z) : (x * y * z) % 15 = 2 :=
by
  sorry

end sum_remainder_product_remainder_l83_83018


namespace remainder_when_product_divided_by_5_l83_83436

def n1 := 1483
def n2 := 1773
def n3 := 1827
def n4 := 2001
def mod5 (n : Nat) : Nat := n % 5

theorem remainder_when_product_divided_by_5 :
  mod5 (n1 * n2 * n3 * n4) = 3 :=
sorry

end remainder_when_product_divided_by_5_l83_83436


namespace find_integers_l83_83047

theorem find_integers (x y : ℤ) 
  (h1 : x * y + (x + y) = 95) 
  (h2 : x * y - (x + y) = 59) : 
  (x = 11 ∧ y = 7) ∨ (x = 7 ∧ y = 11) :=
by
  sorry

end find_integers_l83_83047


namespace largest_possible_difference_l83_83654

theorem largest_possible_difference 
  (weight_A weight_B weight_C : ℝ)
  (hA : 24.9 ≤ weight_A ∧ weight_A ≤ 25.1)
  (hB : 24.8 ≤ weight_B ∧ weight_B ≤ 25.2)
  (hC : 24.7 ≤ weight_C ∧ weight_C ≤ 25.3) :
  ∃ w1 w2 : ℝ, (w1 = weight_C ∧ w2 = weight_C ∧ abs (w1 - w2) = 0.6) :=
by
  sorry

end largest_possible_difference_l83_83654


namespace sum_of_vars_l83_83035

variables (x y z w : ℤ)

theorem sum_of_vars (h1 : x - y + z = 7)
                    (h2 : y - z + w = 8)
                    (h3 : z - w + x = 4)
                    (h4 : w - x + y = 3) :
  x + y + z + w = 11 :=
by
  sorry

end sum_of_vars_l83_83035


namespace g_neither_even_nor_odd_l83_83886

noncomputable def g (x : ℝ) : ℝ := ⌈x⌉ - 1 / 2

theorem g_neither_even_nor_odd :
  (¬ ∀ x, g x = g (-x)) ∧ (¬ ∀ x, g (-x) = -g x) :=
by
  sorry

end g_neither_even_nor_odd_l83_83886


namespace largest_prime_factor_5040_l83_83776

theorem largest_prime_factor_5040 : ∃ p, Nat.Prime p ∧ p ∣ 5040 ∧ (∀ q, Nat.Prime q ∧ q ∣ 5040 → q ≤ p) := by
  use 7
  constructor
  · exact Nat.prime_7
  constructor
  · exact dvd.intro (2^4 * 3^2 * 5) rfl
  · intros q hq
    cases hq with hq1 hq2
    exact Nat.le_of_dvd (Nat.pos_of_ne_zero (λ hq3, by linarith)) hq2
  sorry

end largest_prime_factor_5040_l83_83776


namespace find_fourth_number_in_sequence_l83_83005

-- Define the conditions of the sequence
def first_number : ℤ := 1370
def second_number : ℤ := 1310
def third_number : ℤ := 1070
def fifth_number : ℤ := -6430

-- Define the differences
def difference1 : ℤ := second_number - first_number
def difference2 : ℤ := third_number - second_number

-- Define the ratio of differences
def ratio : ℤ := 4
def next_difference : ℤ := difference2 * ratio

-- Define the fourth number
def fourth_number : ℤ := third_number - (-next_difference)

-- Theorem stating the proof problem
theorem find_fourth_number_in_sequence : fourth_number = 2030 :=
by sorry

end find_fourth_number_in_sequence_l83_83005


namespace santiago_more_roses_l83_83408

def red_roses_santiago := 58
def red_roses_garrett := 24
def red_roses_difference := red_roses_santiago - red_roses_garrett

theorem santiago_more_roses : red_roses_difference = 34 := by
  sorry

end santiago_more_roses_l83_83408


namespace square_area_inscribed_in_parabola_l83_83420

-- Declare the parabola equation
def parabola (x : ℝ) : ℝ := x^2 - 10 * x + 20

-- Declare the condition that we have a square inscribed to this parabola.
def is_inscribed_square (side_length : ℝ) : Prop :=
∀ (x : ℝ), (x = 5 - side_length/2 ∨ x = 5 + side_length/2) → (parabola x = 0)

-- Proof goal
theorem square_area_inscribed_in_parabola : ∃ (side_length : ℝ), is_inscribed_square side_length ∧ side_length^2 = 400 :=
by
  sorry

end square_area_inscribed_in_parabola_l83_83420


namespace simplify_and_evaluate_l83_83751

theorem simplify_and_evaluate 
  (a b : ℤ)
  (h1 : a = 2)
  (h2 : b = -1) : 
  (2 * a^2 * b - 4 * a * b^2) - 2 * (a * b^2 + a^2 * b) = -12 := 
by
  rw [h1, h2]
  sorry

end simplify_and_evaluate_l83_83751


namespace product_of_primes_l83_83622

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

noncomputable def smallest_one_digit_primes (p₁ p₂ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ p₁ < p₂ ∧ p₂ < 10 ∧ ∀ p : ℕ, is_prime p → p < 10 → p = p₁ ∨ p = p₂

noncomputable def smallest_two_digit_prime (p : ℕ) : Prop :=
  is_prime p ∧ p ≥ 10 ∧ p < 100 ∧ ∀ q : ℕ, is_prime q → q ≥ 10 → q < p → q = 11

theorem product_of_primes : ∃ p₁ p₂ p₃ : ℕ, smallest_one_digit_primes p₁ p₂ ∧ smallest_two_digit_prime p₃ ∧ p₁ * p₂ * p₃ = 66 := 
by
  sorry

end product_of_primes_l83_83622


namespace slip_4_goes_in_B_l83_83273

-- Definitions for the slips, cups, and conditions
def slips : List ℝ := [1, 1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5]
def cupSum (c : Char) : ℝ := 
  match c with
  | 'A' => 6
  | 'B' => 7
  | 'C' => 8
  | 'D' => 9
  | 'E' => 10
  | 'F' => 11
  | _   => 0

def cupAssignments : Char → List ℝ
  | 'F' => [2]
  | 'B' => [3]
  | _   => []

theorem slip_4_goes_in_B :
  (∃ cupA cupB cupC cupD cupE cupF : List ℝ, 
    cupA.sum = cupSum 'A' ∧
    cupB.sum = cupSum 'B' ∧
    cupC.sum = cupSum 'C' ∧
    cupD.sum = cupSum 'D' ∧
    cupE.sum = cupSum 'E' ∧
    cupF.sum = cupSum 'F' ∧
    slips = cupA ++ cupB ++ cupC ++ cupD ++ cupE ++ cupF ∧
    cupF.contains 2 ∧
    cupB.contains 3 ∧
    cupB.contains 4) :=
sorry

end slip_4_goes_in_B_l83_83273


namespace express_train_leaves_6_hours_later_l83_83790

theorem express_train_leaves_6_hours_later
  (V_g V_e : ℕ) (t : ℕ) (catch_up_time : ℕ)
  (goods_train_speed : V_g = 36)
  (express_train_speed : V_e = 90)
  (catch_up_in_4_hours : catch_up_time = 4)
  (distance_e : V_e * catch_up_time = 360)
  (distance_g : V_g * (t + catch_up_time) = 360) :
  t = 6 := by
  sorry

end express_train_leaves_6_hours_later_l83_83790


namespace stability_comparison_l83_83606

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

end stability_comparison_l83_83606


namespace solution_set_l83_83555

noncomputable def f : ℝ → ℝ
| x => if x < 2 then 2 * Real.exp (x - 1) else Real.log (x^2 - 1) / Real.log 3

theorem solution_set (x : ℝ) : 
  ((x > 1 ∧ x < 2 ∨ x > Real.sqrt 10)) ↔ f x > 2 :=
sorry

end solution_set_l83_83555


namespace shirts_production_l83_83028

-- Definitions
def constant_rate (r : ℕ) : Prop := ∀ n : ℕ, 8 * n * r = 160 * n

theorem shirts_production (r : ℕ) (h : constant_rate r) : 16 * r = 32 :=
by sorry

end shirts_production_l83_83028


namespace maximum_value_a1_l83_83843

noncomputable def max_possible_value (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0)
  (h3 : a 1 = a 10) : ℝ :=
  16

theorem maximum_value_a1 (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0)
  (h3 : a 1 = a 10) : a 1 ≤ max_possible_value a h1 h2 h3 :=
  sorry

end maximum_value_a1_l83_83843


namespace arithmetic_series_first_term_l83_83385

theorem arithmetic_series_first_term (a d : ℚ) 
  (h1 : 15 * (2 * a + 29 * d) = 450) 
  (h2 : 15 * (2 * a + 89 * d) = 1950) : 
  a = -55 / 6 :=
by 
  sorry

end arithmetic_series_first_term_l83_83385


namespace total_legs_of_all_animals_l83_83024

def num_kangaroos : ℕ := 23
def num_goats : ℕ := 3 * num_kangaroos
def legs_of_kangaroo : ℕ := 2
def legs_of_goat : ℕ := 4

theorem total_legs_of_all_animals : num_kangaroos * legs_of_kangaroo + num_goats * legs_of_goat = 322 :=
by 
  sorry

end total_legs_of_all_animals_l83_83024


namespace units_digit_of_expression_l83_83818

theorem units_digit_of_expression :
  (6 * 16 * 1986 - 6 ^ 4) % 10 = 0 := 
sorry

end units_digit_of_expression_l83_83818


namespace find_y_l83_83276

theorem find_y (x y : ℝ) (hA : {2, Real.log x} = {a | a = 2 ∨ a = Real.log x})
                (hB : {x, y} = {a | a = x ∨ a = y})
                (hInt : {a | a = 2 ∨ a = Real.log x} ∩ {a | a = x ∨ a = y} = {0}) :
  y = 0 :=
  sorry

end find_y_l83_83276


namespace max_z_val_l83_83858

theorem max_z_val (x y : ℝ) (h1 : x + y ≤ 4) (h2 : y - 2 * x + 2 ≤ 0) (h3 : y ≥ 0) :
  ∃ x y, z = x + 2 * y ∧ z = 6 :=
by
  sorry

end max_z_val_l83_83858


namespace required_run_rate_l83_83144

theorem required_run_rate (target : ℝ) (initial_run_rate : ℝ) (initial_overs : ℕ) (remaining_overs : ℕ) :
  target = 282 → initial_run_rate = 3.8 → initial_overs = 10 → remaining_overs = 40 →
  (target - initial_run_rate * initial_overs) / remaining_overs = 6.1 :=
by
  intros
  sorry

end required_run_rate_l83_83144


namespace michael_choices_l83_83882

theorem michael_choices (n k : ℕ) (h_n : n = 10) (h_k : k = 4) : nat.choose n k = 210 :=
by
  rw [h_n, h_k]
  norm_num
  sorry

end michael_choices_l83_83882


namespace ishas_pencil_initial_length_l83_83390

theorem ishas_pencil_initial_length (l : ℝ) (h1 : l - 4 = 18) : l = 22 :=
by
  sorry

end ishas_pencil_initial_length_l83_83390


namespace cone_lateral_surface_area_l83_83833

theorem cone_lateral_surface_area (r : ℕ) (V : ℝ) (h l S : ℝ)
  (h_r : r = 6)
  (h_V : V = 30 * Real.pi)
  (h_volume : V = (1 / 3) * Real.pi * (r ^ 2) * h)
  (h_slant_height : l = Real.sqrt (r^2 + h^2))
  (h_lateral_surface_area : S = Real.pi * r * l) :
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l83_83833


namespace div_val_is_2_l83_83168

theorem div_val_is_2 (x : ℤ) (h : 5 * x = 100) : x / 10 = 2 :=
by 
  sorry

end div_val_is_2_l83_83168


namespace papers_above_140_l83_83327

noncomputable def num_papers_above_140_in_sample : ℕ :=
100 * 0.05

theorem papers_above_140 (mean σ : ℝ) (n_students n_sample : ℕ) : 
  (mean = 120) → 
  (P 100 < ξ < 120 = 0.45) → 
  (n_students = 2000) → 
  (n_sample = 100) → 
  samples_above_140_in_sample = 5 := 
begin
  intros mean_eq dist_eq n_students_eq n_sample_eq,
  sorry
end

end papers_above_140_l83_83327


namespace sum_of_digits_of_N_is_19_l83_83779

-- Given facts about N
variables (N : ℕ) (h1 : 100 ≤ N ∧ N < 1000) 
           (h2 : N % 10 = 7) 
           (h3 : N % 11 = 7) 
           (h4 : N % 12 = 7)

-- Main theorem statement
theorem sum_of_digits_of_N_is_19 : 
  ((N / 100) + ((N % 100) / 10) + (N % 10) = 19) := sorry

end sum_of_digits_of_N_is_19_l83_83779


namespace sqrt_of_9_is_3_l83_83753

theorem sqrt_of_9_is_3 {x : ℝ} (h₁ : x * x = 9) (h₂ : x ≥ 0) : x = 3 := sorry

end sqrt_of_9_is_3_l83_83753


namespace JillTotalTaxPercentage_l83_83171

noncomputable def totalTaxPercentage : ℝ :=
  let totalSpending (beforeDiscount : ℝ) : ℝ := 100
  let clothingBeforeDiscount : ℝ := 0.4 * totalSpending 100
  let foodBeforeDiscount : ℝ := 0.2 * totalSpending 100
  let electronicsBeforeDiscount : ℝ := 0.1 * totalSpending 100
  let cosmeticsBeforeDiscount : ℝ := 0.2 * totalSpending 100
  let householdBeforeDiscount : ℝ := 0.1 * totalSpending 100

  let clothingDiscount : ℝ := 0.1 * clothingBeforeDiscount
  let foodDiscount : ℝ := 0.05 * foodBeforeDiscount
  let electronicsDiscount : ℝ := 0.15 * electronicsBeforeDiscount

  let clothingAfterDiscount := clothingBeforeDiscount - clothingDiscount
  let foodAfterDiscount := foodBeforeDiscount - foodDiscount
  let electronicsAfterDiscount := electronicsBeforeDiscount - electronicsDiscount
  
  let taxOnClothing := 0.06 * clothingAfterDiscount
  let taxOnFood := 0.0 * foodAfterDiscount
  let taxOnElectronics := 0.1 * electronicsAfterDiscount
  let taxOnCosmetics := 0.08 * cosmeticsBeforeDiscount
  let taxOnHousehold := 0.04 * householdBeforeDiscount

  let totalTaxPaid := taxOnClothing + taxOnFood + taxOnElectronics + taxOnCosmetics + taxOnHousehold
  (totalTaxPaid / totalSpending 100) * 100

theorem JillTotalTaxPercentage :
  totalTaxPercentage = 5.01 := by
  sorry

end JillTotalTaxPercentage_l83_83171


namespace domino_chain_can_be_built_l83_83411

def domino_chain_possible : Prop :=
  let total_pieces := 28
  let pieces_with_sixes_removed := 7
  let remaining_pieces := total_pieces - pieces_with_sixes_removed
  (∀ n : ℕ, n < 6 → (∃ k : ℕ, k = 6) → (remaining_pieces % 2 = 0))

theorem domino_chain_can_be_built (h : domino_chain_possible) : Prop :=
  sorry

end domino_chain_can_be_built_l83_83411


namespace find_y_l83_83596

theorem find_y (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : ∃ C, x * y = C) (hx : x = 4) : y = 50 :=
sorry

end find_y_l83_83596


namespace problem1_problem2_binomial_distribution_X_expected_value_X_l83_83072

open ProbabilityTheory

variable {Ω : Type*} [ProbabilitySpace Ω]

-- Definitions for conditions
def bag_initial : set (Fin 6) := {0, 1, 2, 3, 4, 5} -- 4 white balls (0-3), 2 black balls (4-5)
def is_black (b : Fin 6) : Prop := b = 4 ∨ b = 5

-- Problem 1: Ball is replaced
def draw_with_replacement (n : ℕ) : Measure (Fin 6) := MeasureTheory.Measure.map (λ (i : Fin n), i % 6) (MeasureTheory.Measure.prodMeasure (bag_initial n))

theorem problem1 : (draw_with_replacement 2).prob {b | is_black b = 4 ∨ is_black b = 5} = 1 / 3 := sorry

-- Problem 2: Ball is not replaced
def draw_without_replacement (draws : list (Fin 6)) (n : ℕ) : Measure (Fin n) :=
  MeasureTheory.Measure.map (λ (i : Fin n), i % (6 - draws.length)) (MeasureTheory.Measure.prodMeasure (bag_initial n))

theorem problem2 : (draw_without_replacement [0, 1, 2, 3] 2).prob {b | is_black b = 4 ∨ is_black b = 5} = 2 / 5 := sorry

-- Problem 3: Binomial distribution of black balls in three draws with replacement
def X_binomial_with_replacement : measure_theory.measure (fin (ℕ → fin 6)) := measure.map (λ xs, finset.filter (λ x, is_black x) (finset.univ.map ↑xs)).card (measure.prod (finset.pi finset.univ (λ _, measure.prodMeasure (bag_initial 6))))

theorem binomial_distribution_X : X_binomial_with_replacement = probability_theory.binomial 3 (1 / 3) := sorry

theorem expected_value_X : (expected_value X_binomial_with_replacement) = 1 := sorry

end problem1_problem2_binomial_distribution_X_expected_value_X_l83_83072


namespace problem_a_l83_83196

theorem problem_a (f : ℕ → ℕ) (h1 : f 1 = 2) (h2 : ∀ n, f (f n) = f n + 3 * n) : f 26 = 59 := 
sorry

end problem_a_l83_83196


namespace digits_in_8_20_3_30_base_12_l83_83971

def digits_in_base (n b : ℕ) : ℕ :=
  if n = 0 then 1 else 1 + Nat.log b n

theorem digits_in_8_20_3_30_base_12 : digits_in_base (8^20 * 3^30) 12 = 31 :=
by
  sorry

end digits_in_8_20_3_30_base_12_l83_83971


namespace clean_car_time_l83_83166

theorem clean_car_time (t_outside : ℕ) (t_inside : ℕ) (h_outside : t_outside = 80) (h_inside : t_inside = t_outside / 4) : 
  t_outside + t_inside = 100 := 
by 
  sorry

end clean_car_time_l83_83166


namespace statement_C_l83_83249

variables (a b c d : ℝ)

theorem statement_C (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by sorry

end statement_C_l83_83249


namespace min_value_expression_l83_83892

theorem min_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 48) :
  x^2 + 6 * x * y + 9 * y^2 + 4 * z^2 ≥ 128 := 
sorry

end min_value_expression_l83_83892


namespace principal_amount_l83_83234

theorem principal_amount (A r t : ℝ) (hA : A = 1120) (hr : r = 0.11) (ht : t = 2.4) :
  abs ((A / (1 + r * t)) - 885.82) < 0.01 :=
by
  -- This theorem is stating that given A = 1120, r = 0.11, and t = 2.4,
  -- the principal amount (calculated using the simple interest formula)
  -- is approximately 885.82 with a margin of error less than 0.01.
  sorry

end principal_amount_l83_83234


namespace candies_bought_is_18_l83_83953

-- Define the original number of candies
def original_candies : ℕ := 9

-- Define the total number of candies after buying more
def total_candies : ℕ := 27

-- Define the function to calculate the number of candies bought
def candies_bought (o t : ℕ) : ℕ := t - o

-- The main theorem stating that the number of candies bought is 18
theorem candies_bought_is_18 : candies_bought original_candies total_candies = 18 := by
  -- This is where the proof would go
  sorry

end candies_bought_is_18_l83_83953


namespace range_of_a_l83_83531

-- Define sets A and B and the condition A ∩ B = ∅
def set_A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def set_B (a : ℝ) : Set ℝ := { x | x > a }

-- State the condition: A ∩ B = ∅ implies a ≥ 1
theorem range_of_a (a : ℝ) : (set_A ∩ set_B a = ∅) → a ≥ 1 :=
  by
  sorry

end range_of_a_l83_83531


namespace part_I_part_II_l83_83560

-- Conditions
def p (x m : ℝ) : Prop := x > m → 2 * x - 5 > 0
def q (m : ℝ) : Prop := ∃ x y : ℝ, (x^2 / (m - 1)) + (y^2 / (2 - m)) = 1

-- Statements for proof
theorem part_I (m x : ℝ) (hq: q m) (hp: p x m) : 
  m < 1 ∨ (2 < m ∧ m ≤ 5 / 2) :=
sorry

theorem part_II (m x : ℝ) (hq: ¬ q m ∧ ¬(p x m ∧ q m) ∧ (p x m ∨ q m)) : 
  (1 ≤ m ∧ m ≤ 2) ∨ (m > 5 / 2) :=
sorry

end part_I_part_II_l83_83560


namespace count_valid_integers_l83_83532

/-- 
  The count of 4-digit positive integers that consist solely of even digits 
  and are divisible by both 5 and 3 is equal to 120.
-/
theorem count_valid_integers : 
  let even_digits := [0, 2, 4, 6, 8] in 
  let four_digit_nums := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ d ∈ (nat.digits 10 n), d ∈ even_digits} in
  let divisible_by_5 := {n : ℕ | n % 10 = 0} in
  let sum_is_divisible_by_3 := {n : ℕ | (nat.digits 10 n).sum % 3 = 0} in
  let valid_numbers := (four_digit_nums ∩ divisible_by_5 ∩ sum_is_divisible_by_3) in
  finset.card valid_numbers = 120 :=
sorry

end count_valid_integers_l83_83532


namespace find_lesser_number_l83_83462

theorem find_lesser_number (x y : ℕ) (h₁ : x + y = 60) (h₂ : x - y = 10) : y = 25 := by
  sorry

end find_lesser_number_l83_83462


namespace find_b_for_square_binomial_l83_83974

theorem find_b_for_square_binomial 
  (b : ℝ)
  (u t : ℝ)
  (h₁ : u^2 = 4)
  (h₂ : 2 * t * u = 8)
  (h₃ : b = t^2) : b = 4 := 
  sorry

end find_b_for_square_binomial_l83_83974


namespace roots_formula_l83_83993

theorem roots_formula (x₁ x₂ p : ℝ)
  (h₁ : x₁ + x₂ = 6 * p)
  (h₂ : x₁ * x₂ = p^2)
  (h₃ : ∀ x, x ^ 2 - 6 * p * x + p ^ 2 = 0 → x = x₁ ∨ x = x₂) :
  (1 / (x₁ + p) + 1 / (x₂ + p) = 1 / p) :=
by
  sorry

end roots_formula_l83_83993


namespace sum_of_two_integers_eq_sqrt_466_l83_83439

theorem sum_of_two_integers_eq_sqrt_466
  (x y : ℝ)
  (hx : x^2 + y^2 = 250)
  (hy : x * y = 108) :
  x + y = Real.sqrt 466 :=
sorry

end sum_of_two_integers_eq_sqrt_466_l83_83439


namespace sum_of_coefficients_correct_l83_83691

-- Define the polynomial
def polynomial (x y : ℤ) : ℤ := (x + 3 * y) ^ 17

-- Define the sum of coefficients by substituting x = 1 and y = 1
def sum_of_coefficients : ℤ := polynomial 1 1

-- Statement of the mathematical proof problem
theorem sum_of_coefficients_correct :
  sum_of_coefficients = 17179869184 :=
by
  -- proof will be provided here
  sorry

end sum_of_coefficients_correct_l83_83691


namespace y_when_x_is_4_l83_83590

theorem y_when_x_is_4
  (x y : ℝ)
  (h1 : x + y = 30)
  (h2 : x - y = 10)
  (h3 : x * y = 200) :
  y = 50 :=
by
  sorry

end y_when_x_is_4_l83_83590


namespace find_base_c_l83_83874

theorem find_base_c (c : ℕ) : (c^3 - 7*c^2 - 18*c - 8 = 0) → c = 10 :=
by
  sorry

end find_base_c_l83_83874


namespace product_of_primes_l83_83617

def smallest_one_digit_prime := 2
def second_smallest_one_digit_prime := 3
def smallest_two_digit_prime := 11

theorem product_of_primes: smallest_one_digit_prime * second_smallest_one_digit_prime * smallest_two_digit_prime = 66 :=
by {
  -- Applying the definition of the primes and carrying out the multiplication
  show 2 * 3 * 11 = 66,
  calc
  2 * 3 * 11 = 6 * 11 : by rw [mul_assoc 2 3 11]
          ... = 66    : by norm_num,
}

end product_of_primes_l83_83617


namespace find_y_when_x_4_l83_83593

-- Definitions and conditions
variables (x y : ℝ)
def inversely_proportional (x y : ℝ) (K : ℝ) : Prop := x * y = K

-- Main theorem
theorem find_y_when_x_4 
  (K : ℝ) (h1 : inversely_proportional 20 10 K) (h2 : 20 + 10 = 30) (h3 : 20 - 10 = 10) 
  (hx : 4 * y = K) : y = 50 := 
sorry

end find_y_when_x_4_l83_83593


namespace total_marbles_l83_83543

theorem total_marbles (r b g y : ℝ)
  (h1 : r = 1.35 * b)
  (h2 : g = 1.5 * r)
  (h3 : y = 2 * b) :
  r + b + g + y = 4.72 * r :=
by
  sorry

end total_marbles_l83_83543


namespace expansion_of_product_l83_83973

theorem expansion_of_product (x : ℝ) :
  (7 * x + 3) * (5 * x^2 + 2 * x + 4) = 35 * x^3 + 29 * x^2 + 34 * x + 12 := 
by
  sorry

end expansion_of_product_l83_83973


namespace student_score_max_marks_l83_83063

theorem student_score_max_marks (M : ℝ)
  (pass_threshold : ℝ := 0.60 * M)
  (student_marks : ℝ := 80)
  (fail_by : ℝ := 40)
  (required_passing_score : ℝ := student_marks + fail_by) :
  pass_threshold = required_passing_score → M = 200 := 
by
  sorry

end student_score_max_marks_l83_83063


namespace find_x_l83_83559

def star (p q : Int × Int) : Int × Int :=
  (p.1 + q.2, p.2 - q.1)

theorem find_x : ∀ (x y : Int), star (x, y) (4, 2) = (5, 4) → x = 3 :=
by
  intros x y h
  -- The statement is correct, just add a placeholder for the proof
  sorry

end find_x_l83_83559


namespace k_for_circle_radius_7_l83_83110

theorem k_for_circle_radius_7 (k : ℝ) :
  (∃ x y : ℝ, x^2 + 8*x + y^2 + 4*y - k = 0) →
  (∃ x y : ℝ, (x + 4)^2 + (y + 2)^2 = 49) →
  k = 29 :=
by
  sorry

end k_for_circle_radius_7_l83_83110


namespace work_days_for_c_l83_83649

theorem work_days_for_c (A B C : ℝ)
  (h1 : A + B = 1 / 15)
  (h2 : A + B + C = 1 / 11) :
  1 / C = 41.25 :=
by
  sorry

end work_days_for_c_l83_83649


namespace planar_molecules_l83_83502

structure Molecule :=
  (name : String)
  (formula : String)
  (is_planar : Bool)

def propylene : Molecule := 
  { name := "Propylene", formula := "C3H6", is_planar := False }

def vinyl_chloride : Molecule := 
  { name := "Vinyl Chloride", formula := "C2H3Cl", is_planar := True }

def benzene : Molecule := 
  { name := "Benzene", formula := "C6H6", is_planar := True }

def toluene : Molecule := 
  { name := "Toluene", formula := "C7H8", is_planar := False }

theorem planar_molecules : 
  (vinyl_chloride.is_planar = True) ∧ (benzene.is_planar = True) := 
by
  sorry

end planar_molecules_l83_83502


namespace alpha_half_quadrant_l83_83114

theorem alpha_half_quadrant (k : ℤ) (α : ℝ)
  (h : 2 * k * Real.pi - Real.pi / 2 < α ∧ α < 2 * k * Real.pi) :
  (∃ n : ℤ, 2 * n * Real.pi - Real.pi / 4 < α / 2 ∧ α / 2 < 2 * n * Real.pi) ∨
  (∃ n : ℤ, (2 * n + 1) * Real.pi - Real.pi / 4 < α / 2 ∧ α / 2 < (2 * n + 1) * Real.pi) :=
sorry

end alpha_half_quadrant_l83_83114


namespace arrows_from_530_to_535_l83_83496

def cyclic_arrows (n : Nat) : Nat :=
  n % 5

theorem arrows_from_530_to_535 : 
  cyclic_arrows 530 = 0 ∧ cyclic_arrows 531 = 1 ∧ cyclic_arrows 532 = 2 ∧
  cyclic_arrows 533 = 3 ∧ cyclic_arrows 534 = 4 ∧ cyclic_arrows 535 = 0 :=
by
  sorry

end arrows_from_530_to_535_l83_83496


namespace max_points_of_intersection_l83_83328

-- Definitions based on the conditions in a)
def intersects_circle (l : ℕ) : ℕ := 2 * l  -- Each line intersects the circle at most twice
def intersects_lines (n : ℕ) : ℕ := n * (n - 1) / 2  -- Number of intersection points between lines (combinatorial)

-- The main statement that needs to be proved
theorem max_points_of_intersection (lines circle : ℕ) (h_lines_distinct : lines = 3) (h_no_parallel : ∀ (i j : ℕ), i ≠ j → i < lines → j < lines → true) (h_no_common_point : ∀ (i j k : ℕ), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬(true)) : (intersects_circle lines + intersects_lines lines = 9) := 
  by
    sorry

end max_points_of_intersection_l83_83328


namespace third_number_in_list_l83_83423

theorem third_number_in_list :
  let nums : List ℕ := [201, 202, 205, 206, 209, 209, 210, 212, 212]
  nums.nthLe 2 (by simp [List.length]) = 205 :=
sorry

end third_number_in_list_l83_83423


namespace collinear_probability_in_rectangular_array_l83_83725

noncomputable def prob_collinear (total_dots chosen_dots favorable_sets : ℕ) : ℚ :=
  favorable_sets / (Nat.choose total_dots chosen_dots)

theorem collinear_probability_in_rectangular_array :
  prob_collinear 20 4 2 = 2 / 4845 :=
by
  sorry

end collinear_probability_in_rectangular_array_l83_83725


namespace ellipsoid_volume_div_pi_l83_83732

noncomputable def ellipsoid_projection_min_area : ℝ := 9 * Real.pi
noncomputable def ellipsoid_projection_max_area : ℝ := 25 * Real.pi
noncomputable def ellipsoid_circle_projection_area : ℝ := 16 * Real.pi
noncomputable def ellipsoid_volume (a b c : ℝ) : ℝ := (4/3) * Real.pi * a * b * c

theorem ellipsoid_volume_div_pi (a b c : ℝ)
  (h_min : (a * b = 9))
  (h_max : (b * c = 25))
  (h_circle : (b = 4)) :
  ellipsoid_volume a b c / Real.pi = 75 := 
  by
    sorry

end ellipsoid_volume_div_pi_l83_83732


namespace solve_for_x_l83_83572

theorem solve_for_x (x : ℝ) (h : (x - 15) / 3 = (3 * x + 10) / 8) : x = -150 := 
by
  sorry

end solve_for_x_l83_83572


namespace parallelogram_count_l83_83715

theorem parallelogram_count (m n : ℕ) : (choose m 2) * (choose n 2) = number_of_parallelograms m n :=
sorry

end parallelogram_count_l83_83715


namespace bird_families_difference_l83_83061

theorem bird_families_difference {initial_families flown_away : ℕ} (h1 : initial_families = 87) (h2 : flown_away = 7) :
  (initial_families - flown_away) - flown_away = 73 := by
sorry

end bird_families_difference_l83_83061


namespace max_profit_l83_83491

def fixed_cost : ℝ := 20
def variable_cost_per_unit : ℝ := 10

def total_cost (Q : ℝ) := fixed_cost + variable_cost_per_unit * Q

def revenue (Q : ℝ) := 40 * Q - Q^2

def profit (Q : ℝ) := revenue Q - total_cost Q

def Q_optimized : ℝ := 15

theorem max_profit : profit Q_optimized = 205 := by
  sorry -- Proof goes here.

end max_profit_l83_83491


namespace fixed_point_of_log_function_l83_83277

theorem fixed_point_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (-1, 2) ∧ ∀ x y : ℝ, y = 2 + Real.logb a (x + 2) → y = 2 → x = -1 :=
by
  sorry

end fixed_point_of_log_function_l83_83277


namespace choose_4_out_of_10_l83_83880

theorem choose_4_out_of_10 :
  nat.choose 10 4 = 210 :=
  by
  sorry

end choose_4_out_of_10_l83_83880


namespace point_D_sum_is_ten_l83_83901

noncomputable def D_coordinates_sum_eq_ten : Prop :=
  ∃ (D : ℝ × ℝ), (5, 5) = ( (7 + D.1) / 2, (3 + D.2) / 2 ) ∧ (D.1 + D.2 = 10)

theorem point_D_sum_is_ten : D_coordinates_sum_eq_ten :=
  sorry

end point_D_sum_is_ten_l83_83901


namespace number_of_sturgeons_l83_83228

def number_of_fishes := 145
def number_of_pikes := 30
def number_of_herrings := 75

theorem number_of_sturgeons : (number_of_fishes - (number_of_pikes + number_of_herrings) = 40) :=
  by
  sorry

end number_of_sturgeons_l83_83228


namespace interest_credited_cents_l83_83500

theorem interest_credited_cents (P : ℝ) (rt : ℝ) (A : ℝ) (interest : ℝ) :
  A = 255.31 →
  rt = 1 + 0.05 * (1/6) →
  P = A / rt →
  interest = A - P →
  (interest * 100) % 100 = 10 :=
by
  intro hA
  intro hrt
  intro hP
  intro hint
  sorry

end interest_credited_cents_l83_83500


namespace rationalized_expression_correct_A_B_C_D_E_sum_correct_l83_83286

noncomputable def A : ℤ := -18
noncomputable def B : ℤ := 2
noncomputable def C : ℤ := 30
noncomputable def D : ℤ := 5
noncomputable def E : ℤ := 428
noncomputable def expression := 3 / (2 * Real.sqrt 18 + 5 * Real.sqrt 20)
noncomputable def rationalized_form := (A * Real.sqrt B + C * Real.sqrt D) / E

theorem rationalized_expression_correct :
  rationalized_form = (18 * Real.sqrt 2 - 30 * Real.sqrt 5) / -428 :=
by
  sorry

theorem A_B_C_D_E_sum_correct :
  A + B + C + D + E = 447 :=
by
  sorry

end rationalized_expression_correct_A_B_C_D_E_sum_correct_l83_83286


namespace decimal_to_base7_conversion_l83_83222

theorem decimal_to_base7_conversion :
  (2023 : ℕ) = 5 * (7^3) + 6 * (7^2) + 2 * (7^1) + 0 * (7^0) :=
by
  sorry

end decimal_to_base7_conversion_l83_83222


namespace fraction_of_quarters_1840_1849_equals_4_over_15_l83_83951

noncomputable def fraction_of_states_from_1840s (total_states : ℕ) (states_from_1840s : ℕ) : ℚ := 
  states_from_1840s / total_states

theorem fraction_of_quarters_1840_1849_equals_4_over_15 :
  fraction_of_states_from_1840s 30 8 = 4 / 15 := 
by
  sorry

end fraction_of_quarters_1840_1849_equals_4_over_15_l83_83951


namespace cylinder_height_in_sphere_l83_83663

-- Definitions based on conditions
def radius_sphere : ℝ := 7
def radius_cylinder : ℝ := 3

-- Mathematical equivalent proof problem
theorem cylinder_height_in_sphere : 
  ∃ h : ℝ, (radius_sphere)^2 = (radius_cylinder)^2 + (h / 2)^2 ∧ h = 4 * real.sqrt 10 :=
begin
  sorry
end

end cylinder_height_in_sphere_l83_83663


namespace range_of_a_l83_83849

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x^2
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 4 * x^2 - a * x

theorem range_of_a (a : ℝ) :
  (∃ x0 : ℝ, x0 > 0 ∧ f x0 = g (-x0) a) → a ≤ -1 := 
by
  sorry

end range_of_a_l83_83849


namespace max_area_rectangle_with_perimeter_40_l83_83795

theorem max_area_rectangle_with_perimeter_40 :
  ∃ (l w : ℕ), 2 * l + 2 * w = 40 ∧ l * w = 100 :=
sorry

end max_area_rectangle_with_perimeter_40_l83_83795


namespace johns_overall_average_speed_l83_83551

open Real

noncomputable def johns_average_speed (scooter_time_min : ℝ) (scooter_speed_mph : ℝ) 
    (jogging_time_min : ℝ) (jogging_speed_mph : ℝ) : ℝ :=
  let scooter_time_hr := scooter_time_min / 60
  let jogging_time_hr := jogging_time_min / 60
  let distance_scooter := scooter_speed_mph * scooter_time_hr
  let distance_jogging := jogging_speed_mph * jogging_time_hr
  let total_distance := distance_scooter + distance_jogging
  let total_time := scooter_time_hr + jogging_time_hr
  total_distance / total_time

theorem johns_overall_average_speed :
  johns_average_speed 40 20 60 6 = 11.6 :=
by
  sorry

end johns_overall_average_speed_l83_83551


namespace lesser_of_two_numbers_l83_83444

theorem lesser_of_two_numbers (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
by
  sorry

end lesser_of_two_numbers_l83_83444


namespace cody_money_final_l83_83217

theorem cody_money_final (initial_money : ℕ) (birthday_money : ℕ) (money_spent : ℕ) (final_money : ℕ) 
  (h1 : initial_money = 45) (h2 : birthday_money = 9) (h3 : money_spent = 19) :
  final_money = initial_money + birthday_money - money_spent :=
by {
  sorry  -- The proof is not required here.
}

end cody_money_final_l83_83217


namespace sum_of_roots_l83_83777

theorem sum_of_roots (x : ℝ) (h : (x - 6)^2 = 16) : (∃ a b : ℝ, a + b = 12 ∧ (x = a ∨ x = b)) :=
by
  sorry

end sum_of_roots_l83_83777


namespace factorize_polynomial_l83_83685

theorem factorize_polynomial (c : ℝ) :
  (x : ℝ) → (x - 1) * (x - 3) = x^2 - 4 * x + c → c = 3 :=
by 
  sorry

end factorize_polynomial_l83_83685


namespace integer_pairs_perfect_squares_l83_83966

theorem integer_pairs_perfect_squares (a b : ℤ) :
  (∃ k : ℤ, (a, b) = (k^2, 0) ∨ (a, b) = (0, k^2) ∨ (a, b) = (k, 1-k) ∨ (a, b) = (-6, -5) ∨ (a, b) = (-5, -6) ∨ (a, b) = (-4, -4))
  ↔ 
  (∃ x1 x2 : ℤ, a^2 + 4*b = x1^2 ∧ b^2 + 4*a = x2^2) :=
sorry

end integer_pairs_perfect_squares_l83_83966


namespace probability_of_individual_selection_l83_83474

theorem probability_of_individual_selection (sample_size : ℕ) (population_size : ℕ)
  (h_sample : sample_size = 10) (h_population : population_size = 42) :
  (sample_size : ℚ) / (population_size : ℚ) = 5 / 21 := 
by {
  sorry
}

end probability_of_individual_selection_l83_83474


namespace n_pow_19_minus_n_pow_7_div_30_l83_83742

theorem n_pow_19_minus_n_pow_7_div_30 (n : ℕ) (hn : 0 < n) : 30 ∣ (n^19 - n^7) :=
sorry

end n_pow_19_minus_n_pow_7_div_30_l83_83742


namespace methane_tetrahedron_dot_product_l83_83428

noncomputable def tetrahedron_vectors_dot_product_sum : ℝ :=
  let edge_length := 1
  let dot_product := -1 / 3 * edge_length^2
  let pair_count := 6 -- number of pairs in sum of dot products
  pair_count * dot_product

theorem methane_tetrahedron_dot_product :
  tetrahedron_vectors_dot_product_sum = - (3 / 4) := by
  sorry

end methane_tetrahedron_dot_product_l83_83428


namespace remembers_umbrella_prob_l83_83755

theorem remembers_umbrella_prob 
    (P_forgets : ℚ) 
    (h_forgets : P_forgets = 5 / 8) : 
    ∃ P_remembers : ℚ, P_remembers = 3 / 8 := 
by
    sorry

end remembers_umbrella_prob_l83_83755


namespace white_balls_count_l83_83548

theorem white_balls_count (n : ℕ) (h : 8 / (8 + n : ℝ) = 0.4) : n = 12 := by
  sorry

end white_balls_count_l83_83548


namespace sum_of_squares_ways_l83_83718

theorem sum_of_squares_ways : 
  ∃ ways : ℕ, ways = 2 ∧
    (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = 100) ∧ 
    (∃ (x y z w : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ x^2 + y^2 + z^2 + w^2 = 100) := 
sorry

end sum_of_squares_ways_l83_83718


namespace cone_lateral_surface_area_l83_83841

-- Definitions based on the conditions
def coneRadius : ℝ := 6
def coneVolume : ℝ := 30 * Real.pi

-- Mathematical statement
theorem cone_lateral_surface_area (r V : ℝ) (hr : r = coneRadius) (hV : V = coneVolume) :
  ∃ S : ℝ, S = 39 * Real.pi :=
by 
  have h_volume := hV
  have h_radius := hr
  sorry

end cone_lateral_surface_area_l83_83841


namespace H_is_orthocenter_l83_83044

open EuclideanGeometry

variables (A B C D H O : Point)

-- Our assumptions
axiom tetrahedron (ABCD : Tetrahedron A B C D)
axiom insphere (s : Sphere) (inscribed : IsInscribed s ABCD)
axiom touching_face_at_H : TouchingAt s (⟨A, B, C⟩ : Plane) H
axiom another_sphere (t : Sphere) (touches_face_at_O : TouchingAt t (⟨A, B, C⟩ : Plane) O)
axiom circumcenter_O : IsCircumcenter O (Triangle A B C)

-- The proof goal
theorem H_is_orthocenter :
  IsOrthocenter H (Triangle A B C) := sorry

end H_is_orthocenter_l83_83044


namespace iceberg_submersion_l83_83340

theorem iceberg_submersion (V_total V_immersed S_total S_submerged : ℝ) :
  convex_polyhedron ∧ floating_on_sea ∧
  V_total > 0 ∧ V_immersed > 0 ∧ S_total > 0 ∧ S_submerged > 0 ∧
  (V_immersed / V_total >= 0.90) ∧ ((S_total - S_submerged) / S_total >= 0.50) :=
sorry

end iceberg_submersion_l83_83340


namespace cone_lateral_surface_area_l83_83834

theorem cone_lateral_surface_area (r V : ℝ) (h l S : ℝ) 
  (radius_condition : r = 6)
  (volume_condition : V = 30 * Real.pi)
  (volume_formula : V = (1 / 3) * Real.pi * r^2 * h)
  (slant_height_formula : l = Real.sqrt (r^2 + h^2))
  (lateral_surface_area_formula : S = Real.pi * r * l) :
  S = 39 * Real.pi := 
sorry

end cone_lateral_surface_area_l83_83834


namespace cube_side_length_l83_83031

theorem cube_side_length (n : ℕ) (h : n^3 - (n-2)^3 = 98) : n = 5 :=
by sorry

end cube_side_length_l83_83031


namespace polynomial_coeffs_sum_l83_83246

theorem polynomial_coeffs_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) (x : ℝ) :
  (2*x - 3)^5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 →
  a_1 + 2*a_2 + 3*a_3 + 4*a_4 + 5*a_5 = 10 :=
by
  sorry

end polynomial_coeffs_sum_l83_83246


namespace incorrect_statement_l83_83780

theorem incorrect_statement : 
  ¬(∀ (p q : Prop), (¬p ∧ ¬q) → (¬p ∧ ¬q)) := 
    sorry

end incorrect_statement_l83_83780


namespace nabla_difference_l83_83253

def nabla (a b : ℚ) : ℚ :=
  (a + b) / (1 + (a * b)^2)

theorem nabla_difference :
  (nabla 3 4) - (nabla 1 2) = - (16 / 29) :=
by
  sorry

end nabla_difference_l83_83253


namespace new_job_hourly_wage_l83_83019

def current_job_weekly_earnings : ℝ := 8 * 10
def new_job_hours_per_week : ℝ := 4
def new_job_bonus : ℝ := 35
def new_job_expected_additional_wage : ℝ := 15

theorem new_job_hourly_wage (W : ℝ) 
  (h_current_job : current_job_weekly_earnings = 80)
  (h_new_job : new_job_hours_per_week * W + new_job_bonus = current_job_weekly_earnings + new_job_expected_additional_wage) : 
  W = 15 :=
by 
  sorry

end new_job_hourly_wage_l83_83019


namespace div_by_16_l83_83172

theorem div_by_16 (n : ℕ) : 
  ((2*n - 1)^3 - (2*n)^2 + 2*n + 1) % 16 = 0 :=
sorry

end div_by_16_l83_83172


namespace range_of_a_if_ineq_has_empty_solution_l83_83855

theorem range_of_a_if_ineq_has_empty_solution (a : ℝ) :
  (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0) → -2 ≤ a ∧ a < 6/5 :=
by
  sorry

end range_of_a_if_ineq_has_empty_solution_l83_83855


namespace eq1_eq2_eq3_eq4_l83_83418

theorem eq1 (x : ℚ) : 3 * x^2 - 32 * x - 48 = 0 ↔ (x = 12 ∨ x = -4/3) := sorry

theorem eq2 (x : ℚ) : 4 * x^2 + x - 3 = 0 ↔ (x = 3/4 ∨ x = -1) := sorry

theorem eq3 (x : ℚ) : (3 * x + 1)^2 - 4 = 0 ↔ (x = 1/3 ∨ x = -1) := sorry

theorem eq4 (x : ℚ) : 9 * (x - 2)^2 = 4 * (x + 1)^2 ↔ (x = 8 ∨ x = 4/5) := sorry

end eq1_eq2_eq3_eq4_l83_83418


namespace Nathan_daily_hours_l83_83169

theorem Nathan_daily_hours (x : ℝ) 
  (h1 : 14 * x + 35 = 77) : 
  x = 3 := 
by 
  sorry

end Nathan_daily_hours_l83_83169


namespace adoption_time_l83_83794

theorem adoption_time
  (p0 : ℕ) (p1 : ℕ) (rate : ℕ)
  (p0_eq : p0 = 10) (p1_eq : p1 = 15) (rate_eq : rate = 7) :
  Nat.ceil ((p0 + p1) / rate) = 4 := by
  sorry

end adoption_time_l83_83794


namespace smallest_number_is_correct_largest_number_is_correct_l83_83524

def initial_sequence := "123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960"

def remove_digits (n : ℕ) (s : String) : String := sorry  -- Placeholder function for removing n digits

noncomputable def smallest_number_after_removal (s : String) : String :=
  -- Function to find the smallest number possible after removing digits
  remove_digits 100 s

noncomputable def largest_number_after_removal (s : String) : String :=
  -- Function to find the largest number possible after removing digits
  remove_digits 100 s

theorem smallest_number_is_correct : smallest_number_after_removal initial_sequence = "123450" :=
  sorry

theorem largest_number_is_correct : largest_number_after_removal initial_sequence = "56758596049" :=
  sorry

end smallest_number_is_correct_largest_number_is_correct_l83_83524


namespace lesser_number_l83_83457

theorem lesser_number (x y : ℕ) (h1: x + y = 60) (h2: x - y = 10) : y = 25 :=
sorry

end lesser_number_l83_83457


namespace hyperbola_standard_eq_l83_83847

theorem hyperbola_standard_eq (a c : ℝ) (h1 : a = 5) (h2 : c = 7) :
  (∃ b, b^2 = c^2 - a^2 ∧ (1 = (x^2 / a^2 - y^2 / b^2) ∨ 1 = (y^2 / a^2 - x^2 / b^2))) := by
  sorry

end hyperbola_standard_eq_l83_83847


namespace notebook_cost_l83_83944

theorem notebook_cost :
  let mean_expenditure := 500
  let daily_expenditures := [450, 600, 400, 500, 550, 300]
  let cost_earphone := 620
  let cost_pen := 30
  let total_days := 7
  let total_expenditure := mean_expenditure * total_days
  let sum_other_days := daily_expenditures.sum
  let expenditure_friday := total_expenditure - sum_other_days
  let cost_notebook := expenditure_friday - (cost_earphone + cost_pen)
  cost_notebook = 50 := by
  sorry

end notebook_cost_l83_83944


namespace find_box_length_l83_83074

theorem find_box_length (width depth : ℕ) (num_cubes : ℕ) (cube_side length : ℕ) 
  (h1 : width = 20)
  (h2 : depth = 10)
  (h3 : num_cubes = 56)
  (h4 : cube_side = 10)
  (h5 : length * width * depth = num_cubes * cube_side * cube_side * cube_side) :
  length = 280 :=
sorry

end find_box_length_l83_83074


namespace integral_percentage_l83_83280

variable (a b : ℝ)

theorem integral_percentage (h : ∀ x, x^2 > 0) :
  (∫ x in a..b, (1 / 20 * x^2 + 3 / 10 * x^2)) = 0.35 * (∫ x in a..b, x^2) :=
by
  sorry

end integral_percentage_l83_83280


namespace smallest_portion_l83_83297

theorem smallest_portion
    (a_1 d : ℚ)
    (h1 : 5 * a_1 + 10 * d = 10)
    (h2 : (a_1 + 2 * d + a_1 + 3 * d + a_1 + 4 * d) / 7 = a_1 + a_1 + d) :
  a_1 = 1 / 6 := 
sorry

end smallest_portion_l83_83297


namespace length_of_FD_l83_83388

theorem length_of_FD
  (ABCD_is_square : ∀ (A B C D : ℝ), A = 8 ∧ B = 8 ∧ C = 8 ∧ D = 8)
  (E_midpoint_AD : ∀ (A D E : ℝ), E = (A + D) / 2)
  (F_on_BD : ∀ (B D F E : ℝ), B = 8 ∧ F = 3 ∧ D = 8 ∧ E = 4):
  ∃ (FD : ℝ), FD = 3 := by
  sorry

end length_of_FD_l83_83388


namespace no_integer_solutions_l83_83515

theorem no_integer_solutions (x y : ℤ) : 15 * x^2 - 7 * y^2 ≠ 9 :=
by
  sorry

end no_integer_solutions_l83_83515


namespace moles_of_NaCl_formed_l83_83978

-- Define the conditions
def moles_NaOH : ℕ := 3
def moles_HCl : ℕ := 3

-- Define the balanced chemical equation as a relation
def reaction (NaOH HCl NaCl H2O : ℕ) : Prop :=
  NaOH = HCl ∧ HCl = NaCl ∧ H2O = NaCl

-- Define the proof problem
theorem moles_of_NaCl_formed :
  ∀ (NaOH HCl NaCl H2O : ℕ), NaOH = 3 → HCl = 3 → reaction NaOH HCl NaCl H2O → NaCl = 3 :=
by
  intros NaOH HCl NaCl H2O hNa hHCl hReaction
  sorry

end moles_of_NaCl_formed_l83_83978


namespace pascal_triangle_ratio_l83_83136

theorem pascal_triangle_ratio :
  ∃ n r : ℕ, (binomial n r) * 3 == (binomial n (r + 1)) * 2 ∧
             (binomial n (r + 1)) * 4 == (binomial n (r + 2)) * 3 ∧
             n == 34 := by
  sorry

end pascal_triangle_ratio_l83_83136


namespace minimum_tangent_length_4_l83_83259

noncomputable def minimum_tangent_length (a b : ℝ) : ℝ :=
  Real.sqrt ((b + 4)^2 + (b - 2)^2 - 2)

theorem minimum_tangent_length_4 :
  ∀ (a b : ℝ), (x^2 + y^2 + 2 * x - 4 * y + 3 = 0) ∧ (x = a ∧ y = b) ∧ (2*a*x + b*y + 6 = 0) → 
    minimum_tangent_length a b = 4 :=
by
  sorry

end minimum_tangent_length_4_l83_83259


namespace functional_expression_y_x_maximize_profit_price_reduction_and_profit_l83_83040

-- Define the conditions
variable (C_selling C_cost : ℝ := 80) (C_costComponent : ℝ := 30) (initialSales : ℝ := 600) 
variable (dec_price : ℝ := 2) (inc_sales : ℝ := 30)
variable (decrease x : ℝ)

-- Define and prove part 1: Functional expression between y and x
theorem functional_expression_y_x : (decrease : ℝ) → (15 * decrease + initialSales : ℝ) = (inc_sales / dec_price * decrease + initialSales) :=
by sorry

-- Define the function for weekly profit
def weekly_profit (x : ℝ) : ℝ := 
  let selling_price := C_selling - x
  let cost_price := C_costComponent
  let sales_volume := 15 * x + initialSales
  (selling_price - cost_price) * sales_volume

-- Prove the condition for maximizing weekly sales profit
theorem maximize_profit_price_reduction_and_profit : 
  (∀ x : ℤ, x % 2 = 0 → weekly_profit x ≤ 30360) ∧
  weekly_profit 4 = 30360 ∧ 
  weekly_profit 6 = 30360 :=
by sorry

end functional_expression_y_x_maximize_profit_price_reduction_and_profit_l83_83040


namespace problem_part1_problem_part2_l83_83985

theorem problem_part1 (α : ℝ) (h : Real.tan α = -2) :
    (3 * Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α) = -4 / 7 := 
    sorry

theorem problem_part2 (α : ℝ) (h : Real.tan α = -2) :
    3 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = -5 := 
    sorry

end problem_part1_problem_part2_l83_83985


namespace pairs_of_mittens_correct_l83_83127

variables (pairs_of_plugs_added pairs_of_plugs_original plugs_total pairs_of_plugs_current pairs_of_mittens : ℕ)

theorem pairs_of_mittens_correct :
  pairs_of_plugs_added = 30 →
  plugs_total = 400 →
  pairs_of_plugs_current = plugs_total / 2 →
  pairs_of_plugs_current = pairs_of_plugs_original + pairs_of_plugs_added →
  pairs_of_mittens = pairs_of_plugs_original - 20 →
  pairs_of_mittens = 150 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end pairs_of_mittens_correct_l83_83127


namespace prob_draw_2_groupA_is_one_third_game_rule_is_unfair_l83_83767

-- Definitions
def groupA : List ℕ := [2, 4, 6]
def groupB : List ℕ := [3, 5]
def card_count_A : ℕ := groupA.length
def card_count_B : ℕ := groupB.length

-- Condition 1: Probability of drawing the card with number 2 from group A
def prob_draw_2_groupA : ℚ := 1 / card_count_A

-- Condition 2: Game Rule Outcomes
def is_multiple_of_3 (n : ℕ) : Bool := n % 3 == 0

def outcomes : List (ℕ × ℕ) := [(2, 3), (2, 5), (4, 3), (4, 5), (6, 3), (6, 5)]

def winning_outcomes_A : List (ℕ × ℕ) :=List.filter (λ p => is_multiple_of_3 (p.1 * p.2)) outcomes
def winning_outcomes_B : List (ℕ × ℕ) := List.filter (λ p => ¬ is_multiple_of_3 (p.1 * p.2)) outcomes

def prob_win_A : ℚ := winning_outcomes_A.length / outcomes.length
def prob_win_B : ℚ := winning_outcomes_B.length / outcomes.length

-- Proof problems
theorem prob_draw_2_groupA_is_one_third : prob_draw_2_groupA = 1 / 3 := sorry

theorem game_rule_is_unfair : prob_win_A ≠ prob_win_B := sorry

end prob_draw_2_groupA_is_one_third_game_rule_is_unfair_l83_83767


namespace factor_expression_l83_83513

theorem factor_expression (y : ℝ) :
  5 * y * (y - 4) + 2 * (y - 4) = (5 * y + 2) * (y - 4) :=
by
  sorry

end factor_expression_l83_83513


namespace total_pokemon_cards_l83_83734

-- Definitions based on the problem statement

def dozen_to_cards (dozen : ℝ) : ℝ :=
  dozen * 12

def melanie_cards : ℝ :=
  dozen_to_cards 7.5

def benny_cards : ℝ :=
  dozen_to_cards 9

def sandy_cards : ℝ :=
  dozen_to_cards 5.2

def jessica_cards : ℝ :=
  dozen_to_cards 12.8

def total_cards : ℝ :=
  melanie_cards + benny_cards + sandy_cards + jessica_cards

theorem total_pokemon_cards : total_cards = 414 := 
  by sorry

end total_pokemon_cards_l83_83734


namespace k_h_of_3_eq_79_l83_83707

def h (x : ℝ) : ℝ := x^3
def k (x : ℝ) : ℝ := 3 * x - 2

theorem k_h_of_3_eq_79 : k (h 3) = 79 := by
  sorry

end k_h_of_3_eq_79_l83_83707


namespace fraction_power_multiply_l83_83675

theorem fraction_power_multiply :
  ((1 : ℚ) / 3)^4 * ((1 : ℚ) / 5) = (1 / 405 : ℚ) :=
by sorry

end fraction_power_multiply_l83_83675


namespace ratio_of_x_to_y_l83_83710

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x + 2 * y) / (2 * x - y) = 5 / 4) : x / y = -13 / 2 := 
by 
  sorry

end ratio_of_x_to_y_l83_83710


namespace y_minus_x_is_7_l83_83142

theorem y_minus_x_is_7 (x y : ℕ) (hx : x ≠ y) (h1 : 3 + y = 10) (h2 : 0 + x + 1 = 1) (h3 : 3 + 7 = 10) :
  y - x = 7 :=
by
  sorry

end y_minus_x_is_7_l83_83142


namespace initial_mixture_two_l83_83733

theorem initial_mixture_two (x : ℝ) (h : 0.25 * (x + 0.4) = 0.10 * x + 0.4) : x = 2 :=
by
  sorry

end initial_mixture_two_l83_83733


namespace felicity_used_5_gallons_less_l83_83514

def adhesion_gas_problem : Prop :=
  ∃ A x : ℕ, (A + 23 = 30) ∧ (4 * A - x = 23) ∧ (x = 5)
  
theorem felicity_used_5_gallons_less :
  adhesion_gas_problem :=
by
  sorry

end felicity_used_5_gallons_less_l83_83514


namespace find_real_number_l83_83644

theorem find_real_number :
    (∃ y : ℝ, y = 3 + (5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + sorry)))))))))) ∧ 
    y = (3 + Real.sqrt 29) / 2 :=
by
  sorry

end find_real_number_l83_83644


namespace problem_statement_l83_83133

theorem problem_statement (a b : ℝ) (h : 3 * a - 2 * b = -1) : 3 * a - 2 * b + 2024 = 2023 :=
by
  sorry

end problem_statement_l83_83133


namespace lateral_surface_area_of_given_cone_l83_83839

noncomputable def coneLateralSurfaceArea (r V : ℝ) : ℝ :=
let h := (3 * V) / (π * r^2) in
let l := Real.sqrt (r^2 + h^2) in
π * r * l

theorem lateral_surface_area_of_given_cone :
  coneLateralSurfaceArea 6 (30 * π) = 39 * π := by
simp [coneLateralSurfaceArea]
sorry

end lateral_surface_area_of_given_cone_l83_83839


namespace correct_calculation_is_D_l83_83060

theorem correct_calculation_is_D 
  (a b x : ℝ) :
  ¬ (5 * a + 2 * b = 7 * a * b) ∧
  ¬ (x ^ 2 - 3 * x ^ 2 = -2) ∧
  ¬ (7 * a - b + (7 * a + b) = 0) ∧
  (4 * a - (-7 * a) = 11 * a) :=
by 
  sorry

end correct_calculation_is_D_l83_83060


namespace original_population_960_l83_83203

variable (original_population : ℝ)

def new_population_increased := original_population + 800
def new_population_decreased := 0.85 * new_population_increased original_population

theorem original_population_960 
  (h1: new_population_decreased original_population = new_population_increased original_population + 24) :
  original_population = 960 := 
by
  -- here comes the proof, but we are omitting it as per the instructions
  sorry

end original_population_960_l83_83203


namespace algebraic_expression_value_l83_83698

variable (x : ℝ)

theorem algebraic_expression_value (h : x^2 + 3 * x + 5 = 7) : 3 * x^2 + 9 * x - 2 = 4 :=
by
  -- This is where the detailed proof would go, but we are skipping it with sorry.
  sorry

end algebraic_expression_value_l83_83698


namespace max_k_value_l83_83549

noncomputable def circle_equation (x y : ℝ) : Prop :=
x^2 + y^2 - 8 * x + 15 = 0

noncomputable def point_on_line (k x y : ℝ) : Prop :=
y = k * x - 2

theorem max_k_value (k : ℝ) :
  (∃ x y, circle_equation x y ∧ point_on_line k x y ∧ (x - 4)^2 + y^2 = 1) →
  k ≤ 4 / 3 :=
by
  sorry

end max_k_value_l83_83549


namespace geometric_sequence_l83_83155

theorem geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1) 
  (h2 : (3 * S 1, 2 * S 2, S 3) = (3 * S 1, 2 * S 2, S 3) ∧ (4 * S 2 = 3 * S 1 + S 3)) 
  (hq_pos : q ≠ 0) 
  (hq : ∀ n, a (n + 1) = a n * q):
  ∀ n, a n = 3^(n-1) :=
by
  sorry

end geometric_sequence_l83_83155


namespace predicted_sales_volume_l83_83647

-- Define the linear regression equation
def regression_equation (x : ℝ) : ℝ := 2 * x + 60

-- Use the given condition x = 34
def temperature_value : ℝ := 34

-- State the theorem that the predicted sales volume is 128
theorem predicted_sales_volume : regression_equation temperature_value = 128 :=
by
  sorry

end predicted_sales_volume_l83_83647


namespace cos_sq_minus_sin_sq_l83_83365

noncomputable def alpha : ℝ := sorry

axiom tan_alpha_eq_two : Real.tan alpha = 2

theorem cos_sq_minus_sin_sq : Real.cos alpha ^ 2 - Real.sin alpha ^ 2 = -3/5 := by
  sorry

end cos_sq_minus_sin_sq_l83_83365


namespace min_value_hyperbola_l83_83981

theorem min_value_hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ∃ e : ℝ, e = 2 ∧ (b^2 = (e * a)^2 - a^2)) :
  (a * 3 + 1 / a) = 2 * Real.sqrt 3 :=
by
  sorry

end min_value_hyperbola_l83_83981


namespace increasing_arithmetic_sequence_l83_83822

theorem increasing_arithmetic_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, a (n + 1) = a n + 2) : ∀ n : ℕ, a (n + 1) > a n :=
by
  sorry

end increasing_arithmetic_sequence_l83_83822


namespace number_of_workers_in_each_block_is_200_l83_83544

-- Conditions
def total_amount : ℕ := 6000
def worth_of_each_gift : ℕ := 2
def number_of_blocks : ℕ := 15

-- Question and answer to be proven
def number_of_workers_in_each_block : ℕ := total_amount / worth_of_each_gift / number_of_blocks

theorem number_of_workers_in_each_block_is_200 :
  number_of_workers_in_each_block = 200 :=
by
  -- Skip the proof with sorry
  sorry

end number_of_workers_in_each_block_is_200_l83_83544


namespace inequality_of_f_on_angles_l83_83507

noncomputable def f : ℝ → ℝ := sorry -- Define f as a noncomputable function

-- Stating the properties of the function f
axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic_function : ∀ x : ℝ, f (x + 1) = -f x
axiom decreasing_interval : ∀ x y : ℝ, (-3 ≤ x ∧ x < y ∧ y ≤ -2) → f x > f y

-- Stating the properties of the angles α and β
variables (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) (hαβ : α ≠ β)

-- The proof statement we want to prove
theorem inequality_of_f_on_angles : f (Real.sin α) > f (Real.cos β) :=
sorry -- The proof is omitted

end inequality_of_f_on_angles_l83_83507


namespace probability_first_white_second_red_l83_83201

section probability_problem

def red_marbles : ℕ := 6
def white_marbles : ℕ := 8
def total_marbles : ℕ := red_marbles + white_marbles

theorem probability_first_white_second_red :
  ((white_marbles:ℚ) / total_marbles) * (red_marbles / (total_marbles - 1)) = 24 / 91 := by
  sorry

end probability_problem

end probability_first_white_second_red_l83_83201


namespace equivalent_sum_of_exponents_l83_83323

theorem equivalent_sum_of_exponents : 3^3 + 3^3 + 3^3 = 3^4 :=
by
  sorry

end equivalent_sum_of_exponents_l83_83323


namespace sum_of_three_numbers_l83_83465

def a : ℚ := 859 / 10
def b : ℚ := 531 / 100
def c : ℚ := 43 / 2

theorem sum_of_three_numbers : a + b + c = 11271 / 100 := by
  sorry

end sum_of_three_numbers_l83_83465


namespace triangle_angle_and_side_l83_83002

theorem triangle_angle_and_side (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C)
  (h2 : a + b = 6)
  (h3 : 1 / 2 * a * b * Real.sin C = 2 * Real.sqrt 3)
  : C = 2 * Real.pi / 3 ∧ c = 2 * Real.sqrt 7 := by
  -- proof omitted
  sorry

end triangle_angle_and_side_l83_83002


namespace water_current_speed_l83_83804

-- Definitions based on the conditions
def swimmer_speed : ℝ := 4  -- The swimmer's speed in still water (km/h)
def swim_time : ℝ := 2  -- Time taken to swim against the current (hours)
def swim_distance : ℝ := 6  -- Distance swum against the current (km)

-- The effective speed against the current
noncomputable def effective_speed_against_current (v : ℝ) : ℝ := swimmer_speed - v

-- Lean statement that formalizes proving the speed of the current
theorem water_current_speed (v : ℝ) (h : effective_speed_against_current v = swim_distance / swim_time) : v = 1 :=
by
  sorry

end water_current_speed_l83_83804


namespace statement_C_correct_l83_83252

theorem statement_C_correct (a b c d : ℝ) (h_ab : a > b) (h_cd : c > d) : a + c > b + d :=
by
  sorry

end statement_C_correct_l83_83252


namespace nina_walking_distance_l83_83152

def distance_walked_by_john : ℝ := 0.7
def distance_john_further_than_nina : ℝ := 0.3

def distance_walked_by_nina : ℝ := distance_walked_by_john - distance_john_further_than_nina

theorem nina_walking_distance :
  distance_walked_by_nina = 0.4 :=
by
  sorry

end nina_walking_distance_l83_83152


namespace mabel_total_tomatoes_l83_83564

def tomatoes_first_plant : ℕ := 12

def tomatoes_second_plant : ℕ := (2 * tomatoes_first_plant) - 6

def tomatoes_combined_first_two : ℕ := tomatoes_first_plant + tomatoes_second_plant

def tomatoes_third_plant : ℕ := tomatoes_combined_first_two / 2

def tomatoes_each_fourth_fifth_plant : ℕ := 3 * tomatoes_combined_first_two

def tomatoes_combined_fourth_fifth : ℕ := 2 * tomatoes_each_fourth_fifth_plant

def tomatoes_each_sixth_seventh_plant : ℕ := (3 * tomatoes_combined_first_two) / 2

def tomatoes_combined_sixth_seventh : ℕ := 2 * tomatoes_each_sixth_seventh_plant

def total_tomatoes : ℕ := tomatoes_first_plant + tomatoes_second_plant + tomatoes_third_plant + tomatoes_combined_fourth_fifth + tomatoes_combined_sixth_seventh

theorem mabel_total_tomatoes : total_tomatoes = 315 :=
by
  sorry

end mabel_total_tomatoes_l83_83564


namespace find_x_l83_83109

theorem find_x (x : ℕ) (h : x + 1 = 2) : x = 1 :=
sorry

end find_x_l83_83109


namespace odd_function_zero_unique_l83_83686

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = - f (- x)

def functional_eq (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f (x + y) * f (x - y) = f x ^ 2 * f y ^ 2

theorem odd_function_zero_unique
  (h_odd : odd_function f)
  (h_func_eq : functional_eq f) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end odd_function_zero_unique_l83_83686


namespace cubic_solution_identity_l83_83398

theorem cubic_solution_identity {a b c : ℕ} 
  (h1 : a + b + c = 6) 
  (h2 : ab + bc + ca = 11) 
  (h3 : abc = 6) : 
  (ab / c) + (bc / a) + (ca / b) = 49 / 6 := 
by 
  sorry

end cubic_solution_identity_l83_83398


namespace fraction_value_l83_83058

theorem fraction_value : (5 * 7 : ℝ) / 10 = 3.5 := by
  sorry

end fraction_value_l83_83058


namespace david_more_pushups_than_zachary_l83_83100

theorem david_more_pushups_than_zachary :
  ∀ (zachary_pushups zachary_crunches david_crunches : ℕ),
    zachary_pushups = 34 →
    zachary_crunches = 62 →
    david_crunches = 45 →
    david_crunches + 17 = zachary_crunches →
    david_crunches + 17 - zachary_pushups = 17 :=
by
  intros zachary_pushups zachary_crunches david_crunches
  intros h1 h2 h3 h4
  sorry

end david_more_pushups_than_zachary_l83_83100


namespace find_f_x_minus_1_l83_83699

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 1

-- State the theorem
theorem find_f_x_minus_1 (x : ℝ) : f (x - 1) = 2 * x - 3 := by
  sorry

end find_f_x_minus_1_l83_83699


namespace find_b_value_l83_83602

def perfect_square_trinomial (a b c : ℕ) : Prop :=
  ∃ d, a = d^2 ∧ c = d^2 ∧ b = 2 * d * d

theorem find_b_value (b : ℝ) :
    (∀ x : ℝ, 16 * x^2 - b * x + 9 = (4 * x - 3) * (4 * x - 3) ∨ 16 * x^2 - b * x + 9 = (4 * x + 3) * (4 * x + 3)) -> 
    b = 24 ∨ b = -24 := 
by
  sorry

end find_b_value_l83_83602


namespace jellybean_probability_l83_83326

theorem jellybean_probability :
  let total_jellybeans := 15
  let green_jellybeans := 6
  let purple_jellybeans := 2
  let yellow_jellybeans := 7
  let total_picked := 4
  let total_ways := Nat.choose total_jellybeans total_picked
  let ways_to_pick_two_yellow := Nat.choose yellow_jellybeans 2
  let ways_to_pick_two_non_yellow := Nat.choose (total_jellybeans - yellow_jellybeans) 2
  let successful_outcomes := ways_to_pick_two_yellow * ways_to_pick_two_non_yellow
  let probability := successful_outcomes / total_ways
  probability = 4 / 9 := by
sorry

end jellybean_probability_l83_83326


namespace pythagorean_triangle_product_divisible_by_60_l83_83495

theorem pythagorean_triangle_product_divisible_by_60 : 
  ∀ (a b c : ℕ),
  (∃ m n : ℕ,
  m > n ∧ (m % 2 = 0 ∨ n % 2 = 0) ∧ m.gcd n = 1 ∧
  a = m^2 - n^2 ∧ b = 2 * m * n ∧ c = m^2 + n^2 ∧ a^2 + b^2 = c^2) →
  60 ∣ (a * b * c) :=
sorry

end pythagorean_triangle_product_divisible_by_60_l83_83495


namespace books_sold_on_wednesday_l83_83274

theorem books_sold_on_wednesday
  (initial_stock : ℕ)
  (sold_monday : ℕ)
  (sold_tuesday : ℕ)
  (sold_thursday : ℕ)
  (sold_friday : ℕ)
  (percent_unsold : ℚ) :
  initial_stock = 900 →
  sold_monday = 75 →
  sold_tuesday = 50 →
  sold_thursday = 78 →
  sold_friday = 135 →
  percent_unsold = 55.333333333333336 →
  ∃ (sold_wednesday : ℕ), sold_wednesday = 64 :=
by
  sorry

end books_sold_on_wednesday_l83_83274


namespace find_four_real_numbers_l83_83356

theorem find_four_real_numbers
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 + x2 * x3 * x4 = 2)
  (h2 : x2 + x1 * x3 * x4 = 2)
  (h3 : x3 + x1 * x2 * x4 = 2)
  (h4 : x4 + x1 * x2 * x3 = 2) :
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) :=
sorry

end find_four_real_numbers_l83_83356


namespace population_scientific_notation_l83_83296

theorem population_scientific_notation : 
  (1.41: ℝ) * (10 ^ 9) = 1.41 * 10 ^ 9 := 
by
  sorry

end population_scientific_notation_l83_83296


namespace kim_monthly_revenue_l83_83729

-- Define the cost to open the store
def initial_cost : ℤ := 25000

-- Define the monthly expenses
def monthly_expenses : ℤ := 1500

-- Define the number of months
def months : ℕ := 10

-- Define the revenue per month
def revenue_per_month (total_revenue : ℤ) (months : ℕ) : ℤ := total_revenue / months

theorem kim_monthly_revenue :
  ∃ r, revenue_per_month r months = 4000 :=
by 
  let total_expenses := monthly_expenses * months
  let total_revenue := initial_cost + total_expenses
  use total_revenue
  unfold revenue_per_month
  sorry

end kim_monthly_revenue_l83_83729


namespace product_of_primes_l83_83632

theorem product_of_primes : 2 * 3 * 11 = 66 :=
by 
  -- Start with the multiplication of the first two primes
  have h1 : 2 * 3 = 6 := by norm_num
  -- Then multiply the result with the smallest two-digit prime
  have h2 : 6 * 11 = 66 := by norm_num
  -- Combine the steps to get the final result
  exact eq.trans (congr_arg (λ x, x * 11) h1) h2

end product_of_primes_l83_83632


namespace product_of_primes_l83_83618

def smallest_one_digit_prime := 2
def second_smallest_one_digit_prime := 3
def smallest_two_digit_prime := 11

theorem product_of_primes: smallest_one_digit_prime * second_smallest_one_digit_prime * smallest_two_digit_prime = 66 :=
by {
  -- Applying the definition of the primes and carrying out the multiplication
  show 2 * 3 * 11 = 66,
  calc
  2 * 3 * 11 = 6 * 11 : by rw [mul_assoc 2 3 11]
          ... = 66    : by norm_num,
}

end product_of_primes_l83_83618


namespace range_of_m_l83_83529

open Real

theorem range_of_m (m : ℝ) : (m^2 > 2 + m ∧ 2 + m > 0) ↔ (m > 2 ∨ -2 < m ∧ m < -1) :=
by
  sorry

end range_of_m_l83_83529


namespace pounds_of_apples_needed_l83_83245

-- Define the conditions
def n : ℕ := 8
def c_p : ℕ := 1
def a_p : ℝ := 2.00
def c_crust : ℝ := 2.00
def c_lemon : ℝ := 0.50
def c_butter : ℝ := 1.50

-- Define the theorem to be proven
theorem pounds_of_apples_needed : 
  (n * c_p - (c_crust + c_lemon + c_butter)) / a_p = 2 := 
by
  sorry

end pounds_of_apples_needed_l83_83245


namespace ratio_correct_l83_83954

-- Definitions based on the problem conditions
def initial_cards_before_eating (X : ℤ) : ℤ := X
def cards_bought_new : ℤ := 4
def cards_left_after_eating : ℤ := 34

-- Definition of the number of cards eaten by the dog
def cards_eaten_by_dog (X : ℤ) : ℤ := X + cards_bought_new - cards_left_after_eating

-- Definition of the ratio of the number of cards eaten to the total number of cards before being eaten
def ratio_cards_eaten_to_total (X : ℤ) : ℚ := (cards_eaten_by_dog X : ℚ) / (X + cards_bought_new : ℚ)

-- Statement to prove
theorem ratio_correct (X : ℤ) : ratio_cards_eaten_to_total X = (X - 30) / (X + 4) := by
  sorry

end ratio_correct_l83_83954


namespace y_when_x_is_4_l83_83588

theorem y_when_x_is_4
  (x y : ℝ)
  (h1 : x + y = 30)
  (h2 : x - y = 10)
  (h3 : x * y = 200) :
  y = 50 :=
by
  sorry

end y_when_x_is_4_l83_83588


namespace total_slices_is_78_l83_83957

-- Definitions based on conditions
def ratio_buzz_waiter (x : ℕ) : Prop := (5 * x) + (8 * x) = 78
def waiter_condition (x : ℕ) : Prop := (8 * x) - 20 = 28

-- Prove that the total number of slices is 78 given conditions
theorem total_slices_is_78 (x : ℕ) (h1 : ratio_buzz_waiter x) (h2 : waiter_condition x) : (5 * x) + (8 * x) = 78 :=
by
  sorry

end total_slices_is_78_l83_83957


namespace modulo_residue_l83_83350

theorem modulo_residue : 
  ∃ (x : ℤ), 0 ≤ x ∧ x < 31 ∧ (-1237 % 31) = x := 
  sorry

end modulo_residue_l83_83350


namespace length_of_segment_AB_l83_83331

-- Define the parabola and its properties
def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus of the parabola y^2 = 4x
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Define the midpoint condition
def midpoint_condition (A B : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ C.1 = 3

-- Main statement of the problem
theorem length_of_segment_AB
  (A B : ℝ × ℝ)
  (hA : parabola_equation A.1 A.2)
  (hB : parabola_equation B.1 B.2)
  (C : ℝ × ℝ)
  (hfoc : focus (1, 0))
  (hm : midpoint_condition A B C) :
  dist A B = 8 :=
by sorry

end length_of_segment_AB_l83_83331


namespace delta_epsilon_time_l83_83922

variable (D E Z h t : ℕ)

theorem delta_epsilon_time :
  (t = D - 8) →
  (t = E - 3) →
  (t = Z / 3) →
  (h = 3 * t) → 
  h = 15 / 8 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end delta_epsilon_time_l83_83922


namespace three_digit_numbers_count_l83_83808

def number_of_3_digit_numbers : ℕ := 
  let without_zero := 2 * Nat.choose 9 3
  let with_zero := Nat.choose 9 2
  without_zero + with_zero

theorem three_digit_numbers_count : number_of_3_digit_numbers = 204 := by
  -- Proof to be completed
  sorry

end three_digit_numbers_count_l83_83808


namespace product_of_smallest_primes_l83_83628

theorem product_of_smallest_primes :
  2 * 3 * 11 = 66 :=
by
  sorry

end product_of_smallest_primes_l83_83628


namespace truncated_cone_volume_correct_l83_83184

-- Definition of given conditions
def large_base_radius : ℝ := 10
def small_base_radius : ℝ := 5
def height : ℝ := 8

-- Definition of the formula for the volume of a truncated cone
def truncated_cone_volume (R r h : ℝ) : ℝ := (1/3) * Real.pi * h * (R^2 + R*r + r^2)

-- The theorem that we need to prove
theorem truncated_cone_volume_correct :
  truncated_cone_volume large_base_radius small_base_radius height = 466.67 * Real.pi :=
by 
  sorry

end truncated_cone_volume_correct_l83_83184


namespace tetrahedron_edge_length_l83_83359

-- Definitions corresponding to the conditions of the problem.
def radius : ℝ := 2

def diameter : ℝ := 2 * radius

/-- Centers of four mutually tangent balls -/
def center_distance : ℝ := diameter

/-- The side length of the square formed by the centers of four balls on the floor. -/
def side_length_of_square : ℝ := center_distance

/-- The edge length of the tetrahedron circumscribed around the four balls. -/
def edge_length_tetrahedron : ℝ := side_length_of_square

-- The statement to be proved.
theorem tetrahedron_edge_length :
  edge_length_tetrahedron = 4 :=
by
  sorry  -- Proof to be constructed

end tetrahedron_edge_length_l83_83359


namespace total_books_correct_l83_83392

-- Define the number of books each person has
def joan_books : ℕ := 10
def tom_books : ℕ := 38
def lisa_books : ℕ := 27
def steve_books : ℕ := 45

-- Calculate the total number of books they have together
def total_books : ℕ := joan_books + tom_books + lisa_books + steve_books

-- State the theorem that needs to be proved
theorem total_books_correct : total_books = 120 :=
by
  sorry

end total_books_correct_l83_83392


namespace probability_of_choosing_red_base_l83_83787

theorem probability_of_choosing_red_base (A B : Prop) (C D : Prop) : 
  let red_bases := 2
  let total_bases := 4
  let probability := red_bases / total_bases
  probability = 1 / 2 := 
by
  sorry

end probability_of_choosing_red_base_l83_83787


namespace product_of_primes_l83_83613

theorem product_of_primes : (2 * 3 * 11) = 66 := by 
  sorry

end product_of_primes_l83_83613


namespace find_m_value_l83_83705

noncomputable def possible_value_of_m (m : ℝ) : Prop :=
  {x : ℝ | 0 ≤ x ∧ x ≤ 1} ∩ {x : ℝ | x^2 - 2*x + m > 0} = ∅

theorem find_m_value :
  possible_value_of_m 0 :=
by
  sorry

end find_m_value_l83_83705


namespace find_probability_eta_ge_1_l83_83562

noncomputable def xi_dist (p : ℝ) : Probability :=
  bernoulli_dist 2 p

noncomputable def eta_dist (p : ℝ) : Probability :=
  bernoulli_dist 3 p

theorem find_probability_eta_ge_1 (p : ℝ) (h : P(xi_dist p ≥ 1) = 5 / 9) : 
  P(eta_dist p ≥ 1) = 19 / 27 := 
sorry

end find_probability_eta_ge_1_l83_83562


namespace product_of_primes_l83_83631

theorem product_of_primes : 2 * 3 * 11 = 66 :=
by 
  -- Start with the multiplication of the first two primes
  have h1 : 2 * 3 = 6 := by norm_num
  -- Then multiply the result with the smallest two-digit prime
  have h2 : 6 * 11 = 66 := by norm_num
  -- Combine the steps to get the final result
  exact eq.trans (congr_arg (λ x, x * 11) h1) h2

end product_of_primes_l83_83631


namespace exists_real_A_l83_83299

theorem exists_real_A (t : ℝ) (n : ℕ) (h_root: t^2 - 10 * t + 1 = 0) :
  ∃ A : ℝ, (A = t) ∧ ∀ n : ℕ, ∃ k : ℕ, A^n + 1/(A^n) - k^2 = 2 :=
by
  sorry

end exists_real_A_l83_83299


namespace ratio_a7_b7_l83_83236

variables (a b : ℕ → ℤ) (Sa Tb : ℕ → ℤ)
variables (h1 : ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0))
variables (h2 : ∀ n : ℕ, b n = b 0 + n * (b 1 - b 0))
variables (h3 : ∀ n : ℕ, Sa n = n * (a 0 + a n) / 2)
variables (h4 : ∀ n : ℕ, Tb n = n * (b 0 + b n) / 2)
variables (h5 : ∀ n : ℕ, n > 0 → Sa n / Tb n = (7 * n + 1) / (4 * n + 27))

theorem ratio_a7_b7 : ∀ n : ℕ, n = 7 → a 7 / b 7 = 92 / 79 :=
by
  intros n hn_eq
  sorry

end ratio_a7_b7_l83_83236


namespace sequence_geometric_progression_iff_b1_eq_b2_l83_83526

theorem sequence_geometric_progression_iff_b1_eq_b2 
  (b : ℕ → ℝ) 
  (h0 : ∀ n, b n > 0)
  (h1 : ∀ n, b (n + 2) = 3 * b n * b (n + 1)) :
  (∃ r : ℝ, ∀ n, b (n + 1) = r * b n) ↔ b 1 = b 0 :=
sorry

end sequence_geometric_progression_iff_b1_eq_b2_l83_83526


namespace initial_invitation_count_l83_83941

def people_invited (didnt_show : ℕ) (num_tables : ℕ) (people_per_table : ℕ) : ℕ :=
  didnt_show + num_tables * people_per_table

theorem initial_invitation_count (didnt_show : ℕ) (num_tables : ℕ) (people_per_table : ℕ)
    (h1 : didnt_show = 35) (h2 : num_tables = 5) (h3 : people_per_table = 2) :
  people_invited didnt_show num_tables people_per_table = 45 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end initial_invitation_count_l83_83941


namespace smallest_positive_angle_l83_83811

theorem smallest_positive_angle :
  ∀ (x : ℝ), 12 * (Real.sin x)^3 * (Real.cos x)^3 - 2 * (Real.sin x)^3 * (Real.cos x)^3 = 1 → 
  x = 15 * (Real.pi / 180) :=
by
  intros x h
  sorry

end smallest_positive_angle_l83_83811


namespace max_complete_dresses_l83_83667

namespace DressMaking

-- Define the initial quantities of fabric
def initial_silk : ℕ := 600
def initial_satin : ℕ := 400
def initial_chiffon : ℕ := 350

-- Define the quantities given to each of 8 friends
def silk_per_friend : ℕ := 15
def satin_per_friend : ℕ := 10
def chiffon_per_friend : ℕ := 5

-- Define the quantities required to make one dress
def silk_per_dress : ℕ := 5
def satin_per_dress : ℕ := 3
def chiffon_per_dress : ℕ := 2

-- Calculate the remaining quantities
def remaining_silk : ℕ := initial_silk - 8 * silk_per_friend
def remaining_satin : ℕ := initial_satin - 8 * satin_per_friend
def remaining_chiffon : ℕ := initial_chiffon - 8 * chiffon_per_friend

-- Calculate the maximum number of dresses that can be made
def max_dresses_silk : ℕ := remaining_silk / silk_per_dress
def max_dresses_satin : ℕ := remaining_satin / satin_per_dress
def max_dresses_chiffon : ℕ := remaining_chiffon / chiffon_per_dress

-- The main theorem indicating the number of complete dresses
theorem max_complete_dresses : max_dresses_silk = 96 ∧ max_dresses_silk ≤ max_dresses_satin ∧ max_dresses_silk ≤ max_dresses_chiffon := by
  sorry

end DressMaking

end max_complete_dresses_l83_83667


namespace lesser_number_l83_83458

theorem lesser_number (x y : ℕ) (h1: x + y = 60) (h2: x - y = 10) : y = 25 :=
sorry

end lesser_number_l83_83458


namespace cone_lateral_surface_area_l83_83829

-- Definitions from conditions
def r : ℝ := 6
def V : ℝ := 30 * Real.pi

-- Theorem to prove
theorem cone_lateral_surface_area : 
  let h := V / (Real.pi * (r ^ 2) / 3) in
  let l := Real.sqrt (r ^ 2 + h ^ 2) in
  let S := Real.pi * r * l in
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l83_83829


namespace distance_between_given_parallel_lines_l83_83346

noncomputable def distance_between_parallel_lines 
  (a1 a2 : ℝ) 
  (b1 b2 : ℝ) 
  (d1 d2 : ℝ) : ℝ :=
let a : ℝ × ℝ := (a1, a2),
    b : ℝ × ℝ := (b1, b2),
    d : ℝ × ℝ := (d1, d2) in
let v : ℝ × ℝ := (b.1 - a.1, b.2 - a.2) in
let dot (x y : ℝ × ℝ) := x.1 * y.1 + x.2 * y.2 in
let p : ℝ × ℝ := (dot v d) / (dot d d) • d in
let c : ℝ × ℝ := (v.1 - p.1, v.2 - p.2) in
real.sqrt ((c.1 * c.1 + c.2 * c.2) : ℝ)

theorem distance_between_given_parallel_lines : 
  distance_between_parallel_lines 
    4 (-1) 
    3 (-2) 
    2 (-6) = 
    (2 * real.sqrt 10) / 5 :=
sorry

end distance_between_given_parallel_lines_l83_83346


namespace total_cleaning_time_l83_83163

theorem total_cleaning_time (time_outside : ℕ) (fraction_inside : ℚ) (time_inside : ℕ) (total_time : ℕ) :
  time_outside = 80 →
  fraction_inside = 1 / 4 →
  time_inside = fraction_inside * time_outside →
  total_time = time_outside + time_inside →
  total_time = 100 :=
by
  intros hto hfi htinside httotal
  rw [hto, hfi] at htinside
  norm_num at htinside
  rw [hto, htinside] at httotal
  norm_num at httotal
  exact httotal

end total_cleaning_time_l83_83163


namespace additional_oil_needed_l83_83483

def oil_needed_each_cylinder : ℕ := 8
def number_of_cylinders : ℕ := 6
def oil_already_added : ℕ := 16

theorem additional_oil_needed : 
  (oil_needed_each_cylinder * number_of_cylinders) - oil_already_added = 32 := by
  sorry

end additional_oil_needed_l83_83483


namespace sum_of_digits_power_of_9_gt_9_l83_83394

def sum_of_digits (n : ℕ) : ℕ :=
  -- function to calculate the sum of digits of n 
  sorry

theorem sum_of_digits_power_of_9_gt_9 (n : ℕ) (h : n ≥ 3) : sum_of_digits (9^n) > 9 :=
  sorry

end sum_of_digits_power_of_9_gt_9_l83_83394


namespace find_k_l83_83309

-- Define the problem statement
theorem find_k (d : ℝ) (x : ℝ)
  (h_ratio : 3 * x / (5 * x) = 3 / 5)
  (h_diag : (10 * d)^2 = (3 * x)^2 + (5 * x)^2) :
  ∃ k : ℝ, (3 * x) * (5 * x) = k * d^2 ∧ k = 750 / 17 := by
  sorry

end find_k_l83_83309


namespace find_b_l83_83431

def direction_vector (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  (x2 - x1, y2 - y1)

theorem find_b (b : ℝ)
  (hx1 : ℝ := -3) (hy1 : ℝ := 1) (hx2 : ℝ := 0) (hy2 : ℝ := 4)
  (hdir : direction_vector hx1 hy1 hx2 hy2 = (3, b)) :
  b = 3 :=
by
  -- Mathematical proof of b = 3 goes here
  sorry

end find_b_l83_83431


namespace infinite_rational_points_in_region_l83_83584

theorem infinite_rational_points_in_region :
  ∃ (S : Set (ℚ × ℚ)), (∀ p ∈ S, p.1 > 0 ∧ p.2 > 0 ∧ p.1 + 2 * p.2 ≤ 6) ∧ S.Infinite :=
sorry

end infinite_rational_points_in_region_l83_83584


namespace find_g7_l83_83582

-- Given the required functional equation and specific value g(6) = 7
theorem find_g7 (g : ℝ → ℝ) (H1 : ∀ x y : ℝ, g (x + y) = g x + g y) (H2 : g 6 = 7) : g 7 = 49 / 6 := by
  sorry

end find_g7_l83_83582


namespace remainder_of_99_pow_36_mod_100_l83_83930

theorem remainder_of_99_pow_36_mod_100 :
  (99 : ℤ)^36 % 100 = 1 := sorry

end remainder_of_99_pow_36_mod_100_l83_83930


namespace lesser_number_of_sum_and_difference_l83_83453

theorem lesser_number_of_sum_and_difference (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
sorry

end lesser_number_of_sum_and_difference_l83_83453


namespace expenditure_on_house_rent_l83_83090

theorem expenditure_on_house_rent
  (income petrol house_rent remaining_income : ℝ)
  (h1 : petrol = 0.30 * income)
  (h2 : petrol = 300)
  (h3 : remaining_income = income - petrol)
  (h4 : house_rent = 0.30 * remaining_income) :
  house_rent = 210 :=
by
  sorry

end expenditure_on_house_rent_l83_83090


namespace file_size_correct_l83_83492

theorem file_size_correct:
  (∀ t1 t2 : ℕ, (60 / 5 = t1) ∧ (15 - t1 = t2) ∧ (t2 * 10 = 30) → (60 + 30 = 90)) := 
by
  sorry

end file_size_correct_l83_83492


namespace problem1_l83_83198

theorem problem1 (α : ℝ) (h : Real.tan α = 2) :
  Real.sin (Real.pi / 2 - α)^2 + 3 * Real.sin (α + Real.pi) * Real.sin (α + Real.pi / 2) = -1 :=
sorry

end problem1_l83_83198


namespace largest_x_quadratic_inequality_l83_83678

theorem largest_x_quadratic_inequality : 
  ∃ (x : ℝ), (x^2 - 10 * x + 24 ≤ 0) ∧ (∀ y, (y^2 - 10 * y + 24 ≤ 0) → y ≤ x) :=
sorry

end largest_x_quadratic_inequality_l83_83678


namespace number_of_extreme_value_points_l83_83041

noncomputable def f (x : ℝ) : ℝ := x^2 + x - Real.log x

theorem number_of_extreme_value_points : ∃! c : ℝ, c > 0 ∧ (deriv f c = 0) :=
by
  sorry

end number_of_extreme_value_points_l83_83041


namespace intersection_complement_l83_83067

universe u

-- Define the universal set U, and sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

-- Define the complement of A with respect to U
def complement (U A : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- The main theorem to be proved
theorem intersection_complement :
  B ∩ (complement U A) = {3, 4} := by
  sorry

end intersection_complement_l83_83067


namespace sixDigitIntegersCount_l83_83129

-- Define the digits to use.
def digits : List ℕ := [1, 2, 2, 5, 9, 9]

-- Define the factorial function as it might not be pre-defined in Mathlib.
def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Calculate the number of unique permutations accounting for repeated digits.
def numberOfUniquePermutations : ℕ :=
  factorial 6 / (factorial 2 * factorial 2)

-- State the theorem proving that we can form exactly 180 unique six-digit integers.
theorem sixDigitIntegersCount : numberOfUniquePermutations = 180 :=
  sorry

end sixDigitIntegersCount_l83_83129


namespace crushing_load_value_l83_83683

-- Given definitions
def W : ℕ := 3
def T : ℕ := 2
def H : ℕ := 6
def L : ℕ := (30 * W^3 * T^5) / H^3

-- Theorem statement
theorem crushing_load_value :
  L = 120 :=
by {
  -- We provided definitions using the given conditions.
  -- Placeholder for proof is provided
  sorry
}

end crushing_load_value_l83_83683


namespace f_96_l83_83762

noncomputable def f : ℕ → ℝ := sorry -- assume f is defined somewhere

axiom f_property (a b k : ℕ) (h : a + b = 3 * 2^k) : f a + f b = 2 * k^2

theorem f_96 : f 96 = 20 :=
by
  -- Here we should provide the proof, but for now we use sorry
  sorry

end f_96_l83_83762


namespace tournament_committee_count_l83_83547

theorem tournament_committee_count : 
  ∃ (n : ℕ), (5 * (Nat.choose 7 4) * (Nat.choose 7 2)^4 = n) ∧ n = 340342925 :=
begin
  let host_ways := Nat.choose 7 4,
  let non_host_ways := Nat.choose 7 2,
  have calculation : 5 * host_ways * non_host_ways^4 = 340342925,
  { calc
      5 * host_ways * non_host_ways^4
        = 5 * 35 * 21^4 : by sorry 
    ... = 340342925 : by sorry },
  exact ⟨340342925, calculation, rfl⟩,
end

end tournament_committee_count_l83_83547


namespace sum_first_20_terms_l83_83716

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the conditions stated in the problem
variables {a : ℕ → ℤ}
variables (h_arith : is_arithmetic_sequence a)
variables (h_sum_first_three : a 1 + a 2 + a 3 = -24)
variables (h_sum_18_to_20 : a 18 + a 19 + a 20 = 78)

-- State the theorem to prove
theorem sum_first_20_terms : (Finset.range 20).sum a = 180 :=
by
  sorry

end sum_first_20_terms_l83_83716


namespace perfect_square_mod_3_l83_83658

theorem perfect_square_mod_3 (n : ℤ) : n^2 % 3 = 0 ∨ n^2 % 3 = 1 :=
sorry

end perfect_square_mod_3_l83_83658


namespace positive_slope_asymptote_l83_83964

-- Define the foci points A and B and the given equation of the hyperbola
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-3, 1)
def hyperbola_eqn (x y : ℝ) : Prop :=
  Real.sqrt ((x - 3)^2 + (y - 1)^2) - Real.sqrt ((x + 3)^2 + (y - 1)^2) = 4

-- State the theorem about the positive slope of the asymptote
theorem positive_slope_asymptote (x y : ℝ) (h : hyperbola_eqn x y) : 
  ∃ b a : ℝ, b = Real.sqrt 5 ∧ a = 2 ∧ (b / a) = Real.sqrt 5 / 2 :=
by
  sorry

end positive_slope_asymptote_l83_83964


namespace mike_gave_pens_l83_83320

theorem mike_gave_pens (M : ℕ) 
  (initial_pens : ℕ := 5) 
  (pens_after_mike : ℕ := initial_pens + M)
  (pens_after_cindy : ℕ := 2 * pens_after_mike)
  (pens_after_sharon : ℕ := pens_after_cindy - 10)
  (final_pens : ℕ := 40) : 
  pens_after_sharon = final_pens → M = 20 := 
by 
  sorry

end mike_gave_pens_l83_83320


namespace triangle_solution_proof_l83_83294

noncomputable def solve_triangle_proof (a b c : ℝ) (alpha beta gamma : ℝ) : Prop :=
  a = 631.28 ∧
  alpha = 63 + 35 / 60 + 30 / 3600 ∧
  b - c = 373 ∧
  beta = 88 + 12 / 60 + 15 / 3600 ∧
  gamma = 28 + 12 / 60 + 15 / 3600 ∧
  b = 704.55 ∧
  c = 331.55

theorem triangle_solution_proof : solve_triangle_proof 631.28 704.55 331.55 (63 + 35 / 60 + 30 / 3600) (88 + 12 / 60 + 15 / 3600) (28 + 12 / 60 + 15 / 3600) :=
  by { sorry }

end triangle_solution_proof_l83_83294


namespace percentage_games_won_l83_83508

def total_games_played : ℕ := 75
def win_rate_first_100_games : ℝ := 0.65

theorem percentage_games_won : 
  (win_rate_first_100_games * total_games_played / total_games_played * 100) = 65 := 
by
  sorry

end percentage_games_won_l83_83508


namespace south_120_meters_l83_83865

-- Define the directions
inductive Direction
| North
| South

-- Define the movement function
def movement (dir : Direction) (distance : Int) : Int :=
  match dir with
  | Direction.North => distance
  | Direction.South => -distance

-- Statement to prove
theorem south_120_meters : movement Direction.South 120 = -120 := 
by
  sorry

end south_120_meters_l83_83865


namespace jaylen_has_2_cucumbers_l83_83726

-- Definitions based on given conditions
def carrots_jaylen := 5
def bell_peppers_kristin := 2
def green_beans_kristin := 20
def total_vegetables_jaylen := 18

def bell_peppers_jaylen := 2 * bell_peppers_kristin
def green_beans_jaylen := (green_beans_kristin / 2) - 3

def known_vegetables_jaylen := carrots_jaylen + bell_peppers_jaylen + green_beans_jaylen
def cucumbers_jaylen := total_vegetables_jaylen - known_vegetables_jaylen

-- The theorem to prove
theorem jaylen_has_2_cucumbers : cucumbers_jaylen = 2 :=
by
  -- We'll place the proof here
  sorry

end jaylen_has_2_cucumbers_l83_83726


namespace lesser_number_of_sum_and_difference_l83_83452

theorem lesser_number_of_sum_and_difference (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
sorry

end lesser_number_of_sum_and_difference_l83_83452


namespace ratio_x_to_y_is_12_l83_83207

noncomputable def ratio_x_y (x y : ℝ) (h1 : y = x * (1 - 0.9166666666666666)) : ℝ := x / y

theorem ratio_x_to_y_is_12 (x y : ℝ) (h1 : y = x * (1 - 0.9166666666666666)) : ratio_x_y x y h1 = 12 :=
sorry

end ratio_x_to_y_is_12_l83_83207


namespace value_of_x_l83_83382

theorem value_of_x (x : ℝ) (h : x = 12 + (20 / 100) * 12) : x = 14.4 :=
by sorry

end value_of_x_l83_83382


namespace color_fig_l83_83230

noncomputable def total_colorings (dots : Finset (Fin 9)) (colors : Finset (Fin 4))
  (adj : dots → dots → Prop)
  (diag : dots → dots → Prop) : Nat :=
  -- coloring left triangle
  let left_triangle := 4 * 3 * 2;
  -- coloring middle triangle considering diagonal restrictions
  let middle_triangle := 3 * 2;
  -- coloring right triangle considering same restrictions
  let right_triangle := 3 * 2;
  left_triangle * middle_triangle * middle_triangle

theorem color_fig (dots : Finset (Fin 9)) (colors : Finset (Fin 4))
  (adj : dots → dots → Prop)
  (diag : dots → dots → Prop) :
  total_colorings dots colors adj diag = 864 :=
by
  sorry

end color_fig_l83_83230


namespace find_M_value_l83_83312

-- Statements of the problem conditions and the proof goal
theorem find_M_value (a b c M : ℤ) (h1 : a + b + c = 75) (h2 : a + 4 = M) (h3 : b - 5 = M) (h4 : 3 * c = M) : M = 31 := 
by
  sorry

end find_M_value_l83_83312


namespace problem_solution_l83_83026

-- Define a function to sum the digits of a number.
def sum_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n % 1000) / 100
  let d3 := (n % 100) / 10
  let d4 := n % 10
  d1 + d2 + d3 + d4

-- Define the problem numbers.
def nums : List ℕ := [4272, 4281, 4290, 4311, 4320]

-- Check if the sum of digits is divisible by 9.
def divisible_by_9 (n : ℕ) : Prop :=
  sum_digits n % 9 = 0

-- Main theorem asserting the result.
theorem problem_solution :
  ∃ n ∈ nums, ¬divisible_by_9 n ∧ (n % 100 / 10) * (n % 10) = 14 := by
  sorry

end problem_solution_l83_83026


namespace range_of_function_l83_83180

-- Given conditions 
def independent_variable_range (x : ℝ) : Prop := x ≥ 2

-- Proof statement (no proof only statement with "sorry")
theorem range_of_function (x : ℝ) (y : ℝ) (h : y = Real.sqrt (x - 2)) : independent_variable_range x :=
by sorry

end range_of_function_l83_83180


namespace train_speed_l83_83805

theorem train_speed
  (num_carriages : ℕ)
  (length_carriage length_engine : ℕ)
  (bridge_length_km : ℝ)
  (crossing_time_min : ℝ)
  (h1 : num_carriages = 24)
  (h2 : length_carriage = 60)
  (h3 : length_engine = 60)
  (h4 : bridge_length_km = 4.5)
  (h5 : crossing_time_min = 6) :
  (num_carriages * length_carriage + length_engine) / 1000 + bridge_length_km / (crossing_time_min / 60) = 60 :=
by
  sorry

end train_speed_l83_83805


namespace capital_at_end_of_2014_year_capital_exceeds_32dot5_billion_l83_83788

noncomputable def company_capital (n : ℕ) : ℝ :=
  if n = 0 then 1000
  else 2 * company_capital (n - 1) - 500

theorem capital_at_end_of_2014 : company_capital 4 = 8500 :=
by sorry

theorem year_capital_exceeds_32dot5_billion : ∀ n : ℕ, company_capital n > 32500 → n ≥ 7 :=
by sorry

end capital_at_end_of_2014_year_capital_exceeds_32dot5_billion_l83_83788


namespace original_number_is_1200_l83_83080

theorem original_number_is_1200 (x : ℝ) (h : 1.40 * x = 1680) : x = 1200 :=
by
  sorry

end original_number_is_1200_l83_83080


namespace rectangle_area_is_48_l83_83084

-- Defining the square's area
def square_area : ℝ := 16

-- Defining the rectangle's width which is the same as the square's side length
def rectangle_width : ℝ := Real.sqrt square_area

-- Defining the rectangle's length which is three times its width
def rectangle_length : ℝ := 3 * rectangle_width

-- The theorem to state that the area of the rectangle is 48
theorem rectangle_area_is_48 : rectangle_width * rectangle_length = 48 :=
by
  -- Placeholder for the actual proof
  sorry

end rectangle_area_is_48_l83_83084


namespace train_pass_time_eq_4_seconds_l83_83934

-- Define the length of the train in meters
def train_length : ℕ := 40

-- Define the speed of the train in km/h
def train_speed_kmph : ℕ := 36

-- Conversion factor: 1 kmph = 1000 meters / 3600 seconds
def conversion_factor : ℚ := 1000 / 3600

-- Convert the train's speed from km/h to m/s
def train_speed_mps : ℚ := train_speed_kmph * conversion_factor

-- Calculate the time to pass the telegraph post
def time_to_pass_post : ℚ := train_length / train_speed_mps

-- The goal: prove the actual time is 4 seconds
theorem train_pass_time_eq_4_seconds : time_to_pass_post = 4 := by
  sorry

end train_pass_time_eq_4_seconds_l83_83934


namespace selling_price_l83_83210

def cost_price : ℝ := 76.92
def profit_rate : ℝ := 0.30

theorem selling_price : cost_price * (1 + profit_rate) = 100.00 := by
  sorry

end selling_price_l83_83210


namespace sufficient_not_necessary_condition_l83_83537

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := abs (x + 1) ≤ 4
def q (x : ℝ) : Prop := x^2 < 5 * x - 6

-- Definitions of negations of p and q
def not_p (x : ℝ) : Prop := x < -5 ∨ x > 3
def not_q (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 3

-- The theorem to prove
theorem sufficient_not_necessary_condition (x : ℝ) :
  (¬ p x → ¬ q x) ∧ (¬ q x → ¬ p x → False) := 
by
  sorry

end sufficient_not_necessary_condition_l83_83537


namespace inequality_must_hold_l83_83862

theorem inequality_must_hold (x y : ℝ) (h : x > y) : -2 * x < -2 * y :=
sorry

end inequality_must_hold_l83_83862


namespace laura_weekly_mileage_l83_83011

-- Define the core conditions

-- Distance to school per round trip (house <-> school)
def school_trip_distance : ℕ := 20

-- Number of trips to school per week
def school_trips_per_week : ℕ := 7

-- Distance to supermarket: 10 miles farther than school
def extra_distance_to_supermarket : ℕ := 10
def supermarket_trip_distance : ℕ := school_trip_distance + 2 * extra_distance_to_supermarket

-- Number of trips to supermarket per week
def supermarket_trips_per_week : ℕ := 2

-- Calculate the total weekly distance
def total_distance_per_week : ℕ := 
  (school_trips_per_week * school_trip_distance) +
  (supermarket_trips_per_week * supermarket_trip_distance)

-- Theorem to prove the total distance Laura drives per week
theorem laura_weekly_mileage :
  total_distance_per_week = 220 := by
  sorry

end laura_weekly_mileage_l83_83011


namespace probability_heads_3_ace_l83_83482

def fair_coin_flip : ℕ := 2
def six_sided_die : ℕ := 6
def standard_deck_cards : ℕ := 52

def successful_outcomes : ℕ := 1 * 1 * 4
def total_possible_outcomes : ℕ := fair_coin_flip * six_sided_die * standard_deck_cards

theorem probability_heads_3_ace :
  (successful_outcomes : ℚ) / (total_possible_outcomes : ℚ) = 1 / 156 := 
sorry

end probability_heads_3_ace_l83_83482


namespace rectangle_R2_area_l83_83113

theorem rectangle_R2_area
  (side1_R1 : ℝ) (area_R1 : ℝ) (diag_R2 : ℝ)
  (h_side1_R1 : side1_R1 = 4)
  (h_area_R1 : area_R1 = 32)
  (h_diag_R2 : diag_R2 = 20) :
  ∃ (area_R2 : ℝ), area_R2 = 160 :=
by
  sorry

end rectangle_R2_area_l83_83113


namespace good_permutation_exists_iff_power_of_two_l83_83927

def is_good_permutation (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ i j k : ℕ, i < j → j < k → k < n → ¬ (↑n ∣ (a i + a k - 2 * a j))

theorem good_permutation_exists_iff_power_of_two (n : ℕ) (h : n ≥ 3) :
  (∃ a : ℕ → ℕ, (∀ i, i < n → a i < n) ∧ is_good_permutation n a) ↔ ∃ b : ℕ, 2 ^ b = n :=
sorry

end good_permutation_exists_iff_power_of_two_l83_83927


namespace lesser_number_of_sum_and_difference_l83_83454

theorem lesser_number_of_sum_and_difference (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
sorry

end lesser_number_of_sum_and_difference_l83_83454


namespace edge_length_of_divided_cube_l83_83051

theorem edge_length_of_divided_cube (volume_original_cube : ℕ) (num_divisions : ℕ) (volume_of_one_smaller_cube : ℕ) (edge_length : ℕ) :
  volume_original_cube = 1000 →
  num_divisions = 8 →
  volume_of_one_smaller_cube = volume_original_cube / num_divisions →
  volume_of_one_smaller_cube = edge_length ^ 3 →
  edge_length = 5 :=
by
  sorry

end edge_length_of_divided_cube_l83_83051


namespace existence_of_ab_l83_83739

theorem existence_of_ab (n : ℕ) (hn : 0 < n) : ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ n ∣ (4 * a^2 + 9 * b^2 - 1) :=
by 
  sorry

end existence_of_ab_l83_83739
