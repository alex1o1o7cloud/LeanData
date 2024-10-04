import Mathlib

namespace length_of_DP_l538_538119

theorem length_of_DP (ABC' : Cube) (P : Point) (h1 : segment B' D intersects plane ACD' at P) :
  length (D P) = (sqrt 3)/3 := 
sorry

end length_of_DP_l538_538119


namespace main_problem_proof_l538_538682

def main_problem : Prop :=
  (1 : ℤ)^10 + (-1 : ℤ)^8 + (-1 : ℤ)^7 + (1 : ℤ)^5 = 2

theorem main_problem_proof : main_problem :=
by {
  sorry
}

end main_problem_proof_l538_538682


namespace lines_in_4x4_grid_l538_538082

theorem lines_in_4x4_grid : 
  let points := finset.range 16 in
  let combinations := finset.pair_combinations points in
  let horizontal_lines := 16 in
  let vertical_lines := 16 in
  let main_diagonal_lines := 2 in
  let minor_diagonal_lines := 2 in
  (horizontal_lines + vertical_lines + main_diagonal_lines + minor_diagonal_lines) = 36 :=
by sorry

end lines_in_4x4_grid_l538_538082


namespace sin_690_eq_neg_half_l538_538405

theorem sin_690_eq_neg_half :
  let rad := Real.pi / 180 in -- Convert degrees to radians
  Real.sin (690 * rad) = -1 / 2 :=
by
  sorry

end sin_690_eq_neg_half_l538_538405


namespace travis_has_hot_potato_l538_538464

noncomputable def hot_potato_probability : ℚ :=
  let p1 := (1 / 3) * (1 / 3) * (1 / 3) -- Case 1
  let p2 := (1 / 3) * (1 / 3)            -- Case 2
  let p3 := (1 / 3) * (1 / 3) * (1 / 3) -- Case 3
  p1 + p2 + p3
  
-- The statement of the theorem.
theorem travis_has_hot_potato :
  hot_potato_probability = 5 / 27 :=
begin
  sorry
end

end travis_has_hot_potato_l538_538464


namespace find_ratio_l538_538580

theorem find_ratio (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 → 
  P / (x + 6) + Q / (x * (x - 5)) = (x^2 - x + 15) / (x^3 + x^2 - 30 * x)) :
  Q / P = 5 / 6 := sorry

end find_ratio_l538_538580


namespace find_tricksters_l538_538368

def inhab_group : Type := { n : ℕ // n < 65 }
def is_knight (i : inhab_group) : Prop := ∀ g : inhab_group, true -- Placeholder for the actual property

theorem find_tricksters (inhabitants : inhab_group → Prop)
  (is_knight : inhab_group → Prop)
  (knight_always_tells_truth : ∀ i : inhab_group, is_knight i → inhabitants i = true)
  (tricksters_2_and_rest_knights : ∃ t1 t2 : inhab_group, t1 ≠ t2 ∧ ¬is_knight t1 ∧ ¬is_knight t2 ∧
    (∀ i : inhab_group, i ≠ t1 → i ≠ t2 → is_knight i)) :
  ∃ find_them : inhab_group → inhab_group → Prop, (∀ q_count : ℕ, q_count ≤ 16) → 
  ∃ t1 t2 : inhab_group, find_them t1 t2 :=
by 
  -- The proof goes here
  sorry

end find_tricksters_l538_538368


namespace common_divisors_of_150_and_m_l538_538510

theorem common_divisors_of_150_and_m (m : ℕ) (h : ∃ (divs : Finset ℕ), divs = {1, q, q^2} ∧ 150 ∣ m ∧ (∀ d ∈ divs, d ∣ 150) ∧ (∀ d ∈ divs, d ∣ m)) : 
  ∃ q : ℕ, (nat.prime q) ∧ (q = 5) ∧ (150 ∣ m) ∧ (∀ d ∈ {1, q, q^2 : ℕ}, d ∣ 150) ∧ (∀ d ∈ {1, q, q^2 : ℕ}, d ∣ m) ∧ q^2 = 25 :=
by
  -- Proof construction skipped
  sorry

end common_divisors_of_150_and_m_l538_538510


namespace arithmetic_sequence_max_sum_l538_538964

theorem arithmetic_sequence_max_sum (a : ℕ → ℝ) (d : ℝ) (m : ℕ) (S : ℕ → ℝ):
  (∀ n, a n = a 1 + (n - 1) * d) → 
  3 * a 8 = 5 * a m → 
  a 1 > 0 →
  (∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * d)) →
  (∀ n, S n ≤ S 20) →
  m = 13 := 
by {
  -- State the corresponding solution steps leading to the proof.
  sorry
}

end arithmetic_sequence_max_sum_l538_538964


namespace max_area_of_triangle_ABC_l538_538531

noncomputable def max_area_triangle (a b c A B C : ℝ) : ℝ :=
  if a = 2 ∧ (tan A / tan B) = 4 / 3 then
    max_area := 1 / 2
  else
    0

theorem max_area_of_triangle_ABC (a b c A B C : ℝ)
  (hA : a = 2)
  (hB : (tan A) / (tan B) = 4 / 3) :
  max_area_triangle a b c A B C  = 1 / 2 := 
begin
  cases max_area_triangle a b c A B C,
  condition => sorry
end

end max_area_of_triangle_ABC_l538_538531


namespace proof_problem_l538_538532

variable (a b c : ℝ)
variable (B : ℝ) -- Angle B

noncomputable def sinB : ℝ :=
sorry -- sin B is defined here

theorem proof_problem
  (h1 : b^2 = a * c)
  (h2 : a^2 + b * c = c^2 + a * c)
  : c / (b * sinB B) = 2 * real.sqrt 3 / 3 := 
sorry

end proof_problem_l538_538532


namespace period_of_tan_cot_eq_pi_l538_538237

noncomputable def period_of_tan_cot_expression (k : ℝ) (h : k ≠ 0) : ℝ :=
let y := (λ x : ℝ, k * (Real.tan x + Real.cot x)) in
π

theorem period_of_tan_cot_eq_pi (k : ℝ) (h : k ≠ 0) :
  ∃ T > 0, ∀ x, (y x = y (x + T)) :=
begin
  let y := (λ x : ℝ, k * (Real.tan x + Real.cot x)),
  use π,
  split,
  { apply Real.pi_pos, },
  { intro x,
    sorry, -- Omit the proof
  }
end

end period_of_tan_cot_eq_pi_l538_538237


namespace device_identification_l538_538906

def sum_of_device_numbers (numbers : List ℕ) : ℕ :=
  numbers.foldr (· + ·) 0

def is_standard_device (d : List ℕ) : Prop :=
  (d = [1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧ (sum_of_device_numbers d = 45)

theorem device_identification (d : List ℕ) : 
  (sum_of_device_numbers d = 45) → is_standard_device d :=
by
  sorry

end device_identification_l538_538906


namespace find_tricksters_l538_538347

theorem find_tricksters (inhabitants : Fin 65 → Prop) (is_knight : Fin 65 → Prop)
    (total_inhabitants : ∀ n, inhabitants n)
    (knights : ∀ n, is_knight n → inhabitants n)
    (tricksters_count : (∑ n, if ¬ is_knight n then 1 else 0) = 2)
    (knights_count : (∑ n, if is_knight n then 1 else 0) = 63)
    (knight_truth : ∀ n, is_knight n → ∀ l : list (Fin 65), (∀ m ∈ l, is_knight m) ↔ true)
    (ask_question : ∀ n, inhabitants n → ∀ l : list (Fin 65), bool) :
  ∃ (find_tricksters_function : (Fin 65 → Prop) → (Fin 65 → bool) → (list (Fin 65))) ,
    (length (find_tricksters_function inhabitants ask_question) ≤ 2) →
    (length (find_tricksters_function inhabitants ask_question) = 2) ∧
    ∀ t ∈ (find_tricksters_function inhabitants ask_question), ¬ is_knight t :=
by sorry

end find_tricksters_l538_538347


namespace volcanoes_remain_73_l538_538731

noncomputable def percentage_explode (initial: ℕ) (percent: ℚ) : ℕ :=
  int.to_nat (initial * percent / 100)

-- define conditions
def initial_volcanoes : ℕ := 250
def first_month_explode : ℕ := percentage_explode initial_volcanoes 15
def after_first_month : ℕ := initial_volcanoes - first_month_explode

def next_two_months_explode : ℕ := percentage_explode after_first_month 25
def after_two_months : ℕ := after_first_month - next_two_months_explode

def fourth_to_sixth_month_explode : ℕ := percentage_explode after_two_months 35
def after_six_months : ℕ := after_two_months - fourth_to_sixth_month_explode

def last_quarter_explode : ℕ := percentage_explode after_six_months 30
def end_of_year : ℕ := after_six_months - last_quarter_explode

-- The final statement to prove
theorem volcanoes_remain_73 : end_of_year = 73 := by
  -- Proof steps will come here when needed
  sorry

end volcanoes_remain_73_l538_538731


namespace find_x_l538_538023

open Nat

theorem find_x (n x : ℕ) (h1 : x = 2^n - 32) (h2 : x.prime_divisors.length = 3) (h3 : 2 ∈ x.prime_divisors) :
  x = 2016 ∨ x = 16352 := 
by
  -- The proof is omitted
  sorry

end find_x_l538_538023


namespace product_last_digit_l538_538012

theorem product_last_digit (x : ℕ → ℕ) (h : ∀ n > 1, x n = 2^n / x (n - 1)) : 
  (x 1 * x 2 * ... * x 200) % 10 = 6 :=
sorry

end product_last_digit_l538_538012


namespace find_n_l538_538447

theorem find_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 14) : n ≡ 14567 [MOD 15] → n = 2 := 
by
  sorry

end find_n_l538_538447


namespace find_tricksters_within_30_questions_l538_538319

/-- 
Given 65 inhabitants in a village where:
- Two inhabitants are tricksters and the rest are knights.
- Knights always tell the truth.
- Tricksters can either tell the truth or lie.
- One can show any inhabitant a list of some group of inhabitants (which can consist of one person)
  and ask if all of them are knights.

Prove that it is possible to find both tricksters with no more than 30 questions.
-/
theorem find_tricksters_within_30_questions :
  ∃ (ask_knights : (inhabitants : fin 65) → list (fin 65) → Prop),
  (∀ (i j : fin 65), i ≠ j → ask_knights i [j] = true → (inhabitants[j] = knight) ∨ (inhabitants[j] = trickster))
  ∧ ∀ (inhabitants : fin 65),
  (∃ S : finset (fin 65), S.card = 2 ∧ 
  (∀ i, i ∈ S → inhabitants[i] = trickster) ∧ 
  by asking no more than 30 questions,
  you can identify both tricksters.

end find_tricksters_within_30_questions_l538_538319


namespace selling_price_to_equal_percentage_profit_and_loss_l538_538673

-- Definition of the variables and conditions
def cost_price : ℝ := 1500
def sp_profit_25 : ℝ := 1875
def sp_loss : ℝ := 1280

theorem selling_price_to_equal_percentage_profit_and_loss :
  ∃ SP : ℝ, SP = 1720.05 ∧
  (sp_profit_25 = cost_price * 1.25) ∧
  (sp_loss < cost_price) ∧
  (14.67 = ((SP - cost_price) / cost_price) * 100) ∧
  (14.67 = ((cost_price - sp_loss) / cost_price) * 100) :=
by
  sorry

end selling_price_to_equal_percentage_profit_and_loss_l538_538673


namespace T_perimeter_correct_l538_538232

def rect1_width : ℝ := 3
def rect1_height : ℝ := 6
def rect2_width : ℝ := 2
def rect2_height : ℝ := 5

-- Place second rectangle's longer edge at the center of first rectangle’s longer edge.
def T_width : ℝ := rect2_height -- 5 inches
def T_height : ℝ := (rect1_width / 2) + rect2_width + (rect1_width / 2) -- 3/2 + 2 + 3/2 inches

def initial_perimeter : ℝ := 2 * (T_width + T_height) -- 2 * (5 + 8) = 26 inches
def correction : ℝ := 2 * (rect1_width / 2) -- 2 * 1.5 = 3 inches

def T_perimeter : ℝ := initial_perimeter - correction -- 26 - 3 = 23 inches

theorem T_perimeter_correct : T_perimeter = 23 := by
  sorry

end T_perimeter_correct_l538_538232


namespace limit_sum_d_l538_538066

/-- The limit of the sum of the lengths of the segments intercepted 
    by the parabola on the x-axis for given quadratic functions, 
    as the number of functions approaches infinity, is equal to 1. -/
theorem limit_sum_d (d : ℕ → ℝ) (h : ∀ n, d n = ∑ i in finset.range n, 
  let a := i + 1 in |((2 * a + 1) / (a * (a + 1))) - ((2 * a + 1) / (a * (a + 1)))|) :
  tendsto (λ n, ∑ i in finset.range n, d i) at_top (nhds 1) := 
begin
  sorry
end

end limit_sum_d_l538_538066


namespace two_n_minus_one_lt_n_plus_one_squared_l538_538809

theorem two_n_minus_one_lt_n_plus_one_squared (n : ℕ) (h : n > 0) : 2 * n - 1 < (n + 1) ^ 2 := 
by
  sorry

end two_n_minus_one_lt_n_plus_one_squared_l538_538809


namespace price_of_child_ticket_l538_538306

theorem price_of_child_ticket (total_seats : ℕ) (adult_ticket_price : ℕ) (total_revenue : ℕ)
  (child_tickets_sold : ℕ) (child_ticket_price : ℕ) :
  total_seats = 80 →
  adult_ticket_price = 12 →
  total_revenue = 519 →
  child_tickets_sold = 63 →
  (17 * adult_ticket_price) + (child_tickets_sold * child_ticket_price) = total_revenue →
  child_ticket_price = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end price_of_child_ticket_l538_538306


namespace tan_585_eq_1_l538_538777

theorem tan_585_eq_1 : Real.tan (585 * Real.pi / 180) = 1 := 
by
  sorry

end tan_585_eq_1_l538_538777


namespace new_player_weight_l538_538685

theorem new_player_weight 
  (original_players : ℕ)
  (original_avg_weight : ℝ)
  (new_players : ℕ)
  (new_avg_weight : ℝ)
  (new_total_weight : ℝ) :
  original_players = 20 →
  original_avg_weight = 180 →
  new_players = 21 →
  new_avg_weight = 181.42857142857142 →
  new_total_weight = 3810 →
  (new_total_weight - original_players * original_avg_weight) = 210 :=
by
  intros
  sorry

end new_player_weight_l538_538685


namespace tan_585_eq_one_l538_538779

theorem tan_585_eq_one : Real.tan (585 * Real.pi / 180) = 1 :=
by
  sorry

end tan_585_eq_one_l538_538779


namespace train_speed_l538_538745

noncomputable def speed_in_kmh (distance : ℕ) (time : ℕ) : ℚ :=
  (distance : ℚ) / (time : ℚ) * 3600 / 1000

theorem train_speed
  (distance : ℕ) (time : ℕ)
  (h_dist : distance = 150)
  (h_time : time = 9) :
  speed_in_kmh distance time = 60 :=
by
  rw [h_dist, h_time]
  sorry

end train_speed_l538_538745


namespace part1_part2_part3_l538_538856

-- Part 1
theorem part1 (b c : ℝ) (h1 : 1 + b + c = 4) (h2 : 2 + (1 / 2) * b + c = 4) : b = 2 ∧ c = 1 :=
sorry

-- Part 2
theorem part2 (b : ℝ) (h : ∀ x₁ x₂ ∈ set.Icc (-1 : ℝ) 1, abs (x₁^2 + b * x₁ - x₂^2 - b * x₂) ≤ 4) : -2 ≤ b ∧ b ≤ 2 :=
sorry

-- Part 3
theorem part3 (a : ℝ) (h : 0 < a)
  (hx : ∀ m n p ∈ set.Icc (-((2 : ℝ) * real.sqrt 5 / 5)) (2 * real.sqrt 5 / 5), let t := λ x, real.sqrt ((1 - x^4) / (1 + x^2)) in
    let y := λ x, t x + a / t x in
    by let f1 (x : ℝ) := x + b; exact (f1 (y m) + f1 (y n) > f1 (y p)) ∧ (f1 (y n) + f1 (y p) > f1 (y m)) ∧ (f1 (y p) + f1 (y m) > f1 (y n)))
  : (1 / 15 < a ∧ a < 5 / 3) :=
sorry

end part1_part2_part3_l538_538856


namespace sin_690_eq_neg_half_l538_538402

theorem sin_690_eq_neg_half :
  let rad := Real.pi / 180 in -- Convert degrees to radians
  Real.sin (690 * rad) = -1 / 2 :=
by
  sorry

end sin_690_eq_neg_half_l538_538402


namespace intersection_A_B_l538_538070

open Set

-- Define the universal set U
def U : Set ℤ := { x | abs (x - 1) < 3 }

-- Define the sets A and C_U B
def A : Set ℤ := {1, 2, 3}
def CU_B : Set ℤ := {-1, 3}

-- Define the set B as U \ CU_B
def B : Set ℤ := U \ CU_B

-- Lean 4 statement to prove A ∩ B = {1, 2}
theorem intersection_A_B :
  A ∩ B = {1, 2} :=
sorry

end intersection_A_B_l538_538070


namespace math_score_prob_l538_538551

noncomputable def normal_dist_prob (μ σ: ℝ) (a b: ℝ) : ℝ :=
  ∫ x in a..b, PDF (Normal μ σ) x dx

theorem math_score_prob 
  (μ σ : ℝ) (hμ : μ = 90) (ha : normal_dist_prob μ σ 70 90 = 0.4) :
  normal_dist_prob μ σ (∅) 110 = 0.9 :=
by 
  sorry

end math_score_prob_l538_538551


namespace which_is_monotonically_decreasing_l538_538249

noncomputable def fA (x : ℝ) : ℝ := (x - 1) ^ 3
noncomputable def fB (x : ℝ) : ℝ := 2 ^ (| - x |)
noncomputable def fC (x : ℝ) : ℝ := -real.log x / real.log 2
noncomputable def fD (x : ℝ) : ℝ := | real.log x / real.log (1 / 2) |

theorem which_is_monotonically_decreasing (x : ℝ) (h1 : 0 < x) :
  ∃ f, (f = fC) ∧ (∀ x y, 0 < x → x < y → y ∈ (0, +∞) → f y < f x) ∧ 
  (∀ f', (f' = fA ∨ f' = fB ∨ f' = fD) → ¬(∀ x y, 0 < x → x < y → y ∈ (0, +∞) → f' y < f' x)) :=
by sorry

end which_is_monotonically_decreasing_l538_538249


namespace three_digit_palindromes_count_l538_538875

def is_palindrome (n : Nat) : Prop :=
  (100 ≤ n) ∧ (n < 1000) ∧ (n / 100 = n % 10)

def middle_digit_even (n : Nat) : Prop :=
  (n / 10 % 10 % 2 = 0)

def in_range_200_600 (n : Nat) : Prop :=
  (200 ≤ n) ∧ (n < 600)

theorem three_digit_palindromes_count :
  ((Finset.filter (λ n, is_palindrome n ∧ middle_digit_even n) (Finset.range 1000)).filter in_range_200_600).card = 20 :=
by
  sorry

end three_digit_palindromes_count_l538_538875


namespace question1_question2_l538_538823

noncomputable def P : set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

noncomputable def S (m : ℝ) : set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem question1 :
  ¬∃ (m : ℝ), S m = P :=
by
  assume h : ∃ (m : ℝ), S m = P
  rcases h with ⟨m, (hs : S m = P)⟩
  sorry -- Proof required here

theorem question2 (m : ℝ) :
  m ≤ 0 → (S m ⊆ P ∧ S m ≠ P) :=
by
  assume h : m ≤ 0
  have h_subset : S m ⊆ P := sorry -- Proof required here
  have h_neq : S m ≠ P := sorry -- Proof required here
  exact ⟨h_subset, h_neq⟩

end question1_question2_l538_538823


namespace length_segment_AB_l538_538587

variable (Ω ω : Type) [MetricSpace Ω] [MetricSpace ω]
variable (O P C D A B : Ω)
variable (radius_Ω radius_ω : ℝ)

noncomputable def circles_intersect_at (c₁ c₂ : Type) [MetricSpace c₁] [MetricSpace c₂] (p1 p2 : ℕ) (intersection1 intersection2 : ℕ) : Prop :=
  sorry -- Formal definition of circle intersection

axiom h1 : O ≠ P
axiom h2 : radius_ω < radius_Ω
axiom h3 : circles_intersect_at Ω ω C D
axiom h4 : chord_circle Ω O B intersects ω at A
axiom h5 : BD BC = 5

theorem length_segment_AB (h : radius_ω < radius_Ω) (h' : BD * BC = 5) : AB = Real.sqrt 5 := sorry

end length_segment_AB_l538_538587


namespace minimum_distinct_midpoints_l538_538217

-- Define the problem conditions
variables (n : ℕ) 
variables (points : Fin n → ℤ) 

-- Ensure points are distinct and ordered
def distinct_points (points : Fin n → ℤ) : Prop :=
  function.injective points

def ordered_points (points : Fin n → ℤ) : Prop :=
  ∀ (i j : Fin n), i < j → points i < points j

-- Define the midpoints of all pairs
def midpoints (points : Fin n → ℤ) : Finset ℚ :=
  {m | ∃ i j, i ≠ j ∧ m = (points i + points j) / 2}

-- The theorem to be proved
theorem minimum_distinct_midpoints (h100 : n = 100) 
  (h_distinct : distinct_points points) 
  (h_ordered : ordered_points points) :
  (midpoints points).card = 197 :=
sorry

end minimum_distinct_midpoints_l538_538217


namespace find_minimum_n_l538_538678

noncomputable def a_seq (n : ℕ) : ℕ := 3 ^ (n - 1)

noncomputable def S_n (n : ℕ) : ℕ := 1 / 2 * (3 ^ n - 1)

theorem find_minimum_n (S_n : ℕ → ℕ) (n : ℕ) :
  (3^n - 1) / 2 > 1000 → n = 7 := 
sorry

end find_minimum_n_l538_538678


namespace steel_ingot_weight_l538_538227

theorem steel_ingot_weight 
  (initial_weight : ℕ)
  (percent_increase : ℚ)
  (ingot_cost : ℚ)
  (discount_threshold : ℕ)
  (discount_percent : ℚ)
  (total_cost : ℚ)
  (added_weight : ℚ)
  (number_of_ingots : ℕ)
  (ingot_weight : ℚ)
  (h1 : initial_weight = 60)
  (h2 : percent_increase = 0.6)
  (h3 : ingot_cost = 5)
  (h4 : discount_threshold = 10)
  (h5 : discount_percent = 0.2)
  (h6 : total_cost = 72)
  (h7 : added_weight = initial_weight * percent_increase)
  (h8 : added_weight = ingot_weight * number_of_ingots)
  (h9 : total_cost = (ingot_cost * number_of_ingots) * (1 - discount_percent)) :
  ingot_weight = 2 := 
by
  sorry

end steel_ingot_weight_l538_538227


namespace circular_segment_l538_538144

-- Defining the circle X and its properties
variables {Q D E Z : ℝ^2} {r : ℝ}

-- Condition: D is a point on the circle with center Q and radius r
def on_circle (D Q : ℝ^2) (r : ℝ) : Prop :=
  dist D Q = r

-- Condition: E is a point inside the circle X
def inside_circle (E Q : ℝ^2) (r : ℝ) : Prop :=
  dist E Q < r

-- Claim: Determine the set of all points Z satisfying the given condition
def required_set (D E Q Z : ℝ^2) (r : ℝ) : set ℝ^2 :=
  { Z | dist Z D ≤ dist Z E }

theorem circular_segment (Q D E : ℝ^2) (r : ℝ) 
  (h1 : on_circle D Q r) 
  (h2 : inside_circle E Q r) : 
  ∃ S : set ℝ^2, S = { Z | dist Z D ≤ dist Z E } ∧ 
  ∀ Z ∈ S, Z lies_in_circular_segment_including_D_bounded_by_perpendicular_bisector D E :=
sorry

end circular_segment_l538_538144


namespace graph_f_abs_shifted_l538_538224

def f (x : ℝ) : ℝ :=
  if x ∈ Icc (-3 : ℝ) 0 then -2 - x
  else if x ∈ Icc 0 2 then real.sqrt (4 - (x - 2)^2) - 2
  else if x ∈ Icc 2 3 then 2 * (x - 2)
  else 0 -- f is defined for given interval, assume 0 otherwise

noncomputable def f_abs_shifted (x : ℝ) : ℝ :=
  f (abs (x + 1))

theorem graph_f_abs_shifted :
  ∃ (g : Set (ℝ × ℝ)), -- Suppose g represents the graph set of f(|x+1|) 
  (∀ x, (x, f_abs_shifted x) ∈ g) ∧
  (/* defining properties of Graph E that we can check manually if needed */) :=
sorry

end graph_f_abs_shifted_l538_538224


namespace abs_neg_one_half_eq_one_half_l538_538191

theorem abs_neg_one_half_eq_one_half : abs (-1/2) = 1/2 := 
by sorry

end abs_neg_one_half_eq_one_half_l538_538191


namespace hh3_eq_2050_l538_538886

def h(x : ℝ) : ℝ := 3 * x^2 + x - 4

theorem hh3_eq_2050 : h(h(3)) = 2050 :=
by
    sorry

end hh3_eq_2050_l538_538886


namespace solve_for_x_l538_538498

noncomputable def f : ℝ → ℝ := λ x, 2 * x - (1 / 2^(abs x))

theorem solve_for_x (x : ℝ) (h : f x = 3/2) : x = 1 :=
sorry

end solve_for_x_l538_538498


namespace external_angle_ABD_is_136_l538_538252

-- Conditions
variables (A B C D : Type) [Triangle A B C]
variables (isosceles_ABC : Isosceles A B C)
variables (angle_A : Angle A = 92)
variables (extends_C_B_to_D : Extends B C D)

-- Proof that the exterior angle ∠ABD = 136°
theorem external_angle_ABD_is_136 :
  ∠(A, B, D) = 136 := 
sorry

end external_angle_ABD_is_136_l538_538252


namespace break_even_price_l538_538188

noncomputable def initial_investment : ℝ := 1500
noncomputable def cost_per_tshirt : ℝ := 3
noncomputable def num_tshirts_break_even : ℝ := 83
noncomputable def total_cost_equipment_tshirts : ℝ := initial_investment + (cost_per_tshirt * num_tshirts_break_even)
noncomputable def price_per_tshirt := total_cost_equipment_tshirts / num_tshirts_break_even

theorem break_even_price : price_per_tshirt = 21.07 := by
  sorry

end break_even_price_l538_538188


namespace sum_inequality_l538_538154

theorem sum_inequality (n : ℕ) (a b : Fin n → ℝ) (h_pos_a : ∀ i, 0 < a i) (h_pos_b : ∀ i, 0 < b i) (h_sum_eq : (∑ i, a i) = (∑ i, b i)) :
  (∑ i, (a i)^2 / (a i + b i)) ≥ (1 / 2) * (∑ i, a i) := 
by
  sorry

end sum_inequality_l538_538154


namespace mod_remainder_of_expression_l538_538806

theorem mod_remainder_of_expression : (7 * 10^20 + 2^20) % 9 = 2 := by
  sorry

end mod_remainder_of_expression_l538_538806


namespace cross_columns_then_row_has_single_asterisk_l538_538115

noncomputable theory

-- Definitions capturing conditions
def grid (n : ℕ) := fin n → fin n → bool  -- asterisk presence represented by boolean
def row_crossed_out (n : ℕ) (f : grid n) (rows : fin n → bool) : grid n := 
  λ i j, if rows i then false else f i j

def column_with_single_asterisk (n : ℕ) (f : grid n) (crossed_out_rows : fin n → bool) :=
  ∃ c : fin n, (∑ i, if crossed_out_rows i then 0 else if f i c then 1 else 0) = 1

def column_crossed_out (n : ℕ) (f : grid n) (columns : fin n → bool) : grid n := 
  λ i j, if columns j then false else f i j

def row_with_single_asterisk (n : ℕ) (f : grid n) (crossed_out_columns : fin n → bool) :=
  ∃ r : fin n, (∑ j, if crossed_out_columns j then 0 else if f r j then 1 else 0) = 1

-- Theorem statement
theorem cross_columns_then_row_has_single_asterisk
  (n : ℕ) (f : grid n)
  (H : ∀ rows : fin n → bool, (∑ i, if rows i then 0 else 1) ≠ 0 →
    column_with_single_asterisk n f rows) :
  ∀ columns : fin n → bool, (∑ j, if columns j then 0 else 1) ≠ 0 →
    row_with_single_asterisk n f columns :=
by
  sorry

end cross_columns_then_row_has_single_asterisk_l538_538115


namespace investment_return_formula_l538_538676

noncomputable def investment_return (x : ℕ) (x_pos : x > 0) : ℝ :=
  if x = 1 then 0.5
  else 2 ^ (x - 2)

theorem investment_return_formula (x : ℕ) (x_pos : x > 0) : investment_return x x_pos = 2 ^ (x - 2) := 
by
  sorry

end investment_return_formula_l538_538676


namespace Mark_spends_more_time_l538_538163

-- Definition of constants associated with the problem
def Chris_speed : ℝ := 3 -- miles per hour
def School_distance : ℝ := 9 -- miles
def Mark_turnaround_distance : ℝ := 3 -- miles

-- Additional calculated distances
def Mark_total_walk : ℝ := 2 * Mark_turnaround_distance + School_distance

-- Calculation of times taken
def Chris_time : ℝ := School_distance / Chris_speed
def Mark_time : ℝ := 2 * (Mark_turnaround_distance / Chris_speed) + (School_distance / Chris_speed)

-- Theorem statement
theorem Mark_spends_more_time : Mark_time - Chris_time = 2 :=
by
  sorry

end Mark_spends_more_time_l538_538163


namespace identify_tricksters_in_30_or_less_questions_l538_538354

-- Define the problem parameters
def inhabitants : Type := Fin 65

def is_knight (inhabitant : inhabitants) : Prop := sorry
def is_trickster (inhabitant : inhabitants) : Prop := sorry

-- Define the properties
axiom knight_truthful : ∀ (x : inhabitants), is_knight x → (forall y : inhabitants, True ↔ (is_knight y = x is_knight y))
axiom trickster_mixed : ∀ (x : inhabitants), is_trickster x → ((∀ y : inhabitants, True) ∨ (∃ y : inhabitants, y ∉ (is_knight y)))

-- Problem statement
theorem identify_tricksters_in_30_or_less_questions
  (inhabitants : Type)
  (n_tricksters : ℕ := 2) -- 2 tricksters
  (total_inhabitants : ℕ := 65) -- 65 total inhabitants
  (questions_limit : ℕ := 30) -- limit of 30 questions
  (knights : inhabitants → Prop)
  (tricksters : inhabitants → Prop) :
    ∃ (solution_exists : ∀ (is_trickster : inhabitants → Prop), ∃ k : inhabitants, (knights k) ∧ (is_trickster k)) 
    (possible_to_find_tricksters : ∀ (is_knight : inhabitants → Prop) (is_trickster : inhabitants → Prop), 
    ∃ (questions_used ≤ questions_limit), ∀ (xs : set inhabitants), questions_used ≤ 30 ∧ 
    (∃ trickster1 trickster2 : inhabitants, (tricksters trickster1 ∧ tricksters trickster2 ∧ trickster1 ≠ trickster2))) :=
sorry

end identify_tricksters_in_30_or_less_questions_l538_538354


namespace tan_585_eq_1_l538_538776

theorem tan_585_eq_1 : Real.tan (585 * Real.pi / 180) = 1 := 
by
  sorry

end tan_585_eq_1_l538_538776


namespace find_tricksters_l538_538323

structure Inhabitant :=
  (isKnight : Prop)

constants (inhabitants : Fin 65 → Inhabitant)
          (tricksters : Fin 2 → Fin 65)

axiom two_tricksters_unique :
  ∃! (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65))

axiom valid_question :
  ∀ (a : Fin 65) (group : Set (Fin 65)), (inhabitants a).isKnight → 
  (∀ i ∈ group, (inhabitants i).isKnight) ↔ 
  (knight a).isKnight

theorem find_tricksters :
  ∃ (q : ℕ), q ≤ 16 ∧
  ∃ (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65)) :=
sorry

end find_tricksters_l538_538323


namespace range_of_a_l538_538885

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x ≤ a → (x + y + 1 ≤ 2 * (x + 1) - 3 * (y + 1))) → a ≤ -2 :=
by 
  intros h
  sorry

end range_of_a_l538_538885


namespace number_of_lines_passing_through_intersection_and_1_unit_from_origin_l538_538754

-- Define the two lines as functions
def line1 (x y : ℝ) := x + 3 * y - 10
def line2 (x y : ℝ) := 3 * x - y

-- Define the distance formula from a point to a line
def point_to_line_distance (a b c x0 y0 : ℝ) := abs (a * x0 + b * y0 + c) / sqrt (a^2 + b^2)

-- Theorem statement: Number of lines passing through the intersection point and at a distance of 1 from origin
theorem number_of_lines_passing_through_intersection_and_1_unit_from_origin :
  let A := (1, 3) in  -- Intersection point solving x + 3y - 10 = 0 and 3x - y = 0
  let lines := Set.filter (λ l, point_to_line_distance l.1.1 l.1.2 l.2 0 0 = 1) 
                (Set.of_list [(4, -3, 5), (1, 0, -1)]) in
  Set.card lines = 2 :=
by
  sorry

end number_of_lines_passing_through_intersection_and_1_unit_from_origin_l538_538754


namespace calculate_box_2_3_neg2_l538_538814

-- Define the box function
def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

-- State the problem as a theorem
theorem calculate_box_2_3_neg2 : box 2 3 (-2) = 107 / 9 :=
by
  sorry

end calculate_box_2_3_neg2_l538_538814


namespace polar_intersection_point_l538_538553

/-- In the polar coordinate system (ρ, θ) where 0 ≤ θ < 2π,
    find the polar coordinates of the intersection points of the curves
    ρ = 2 * sin θ and ρ * cos θ = -1. -/
theorem polar_intersection_point :
  ∃ ρ θ : ℝ, ρ = Real.sqrt(8 + 4 * Real.sqrt 3) ∧ θ = 3 * Real.pi / 4 ∧
            0 ≤ θ ∧ θ < 2 * Real.pi ∧
            ρ = 2 * Real.sin θ ∧ ρ * Real.cos θ = -1 :=
sorry

end polar_intersection_point_l538_538553


namespace find_m_l538_538865

-- We start by defining the vectors and the condition that 2a - b is perpendicular to c.
variable (m : ℝ)
def a : ℝ × ℝ := (m, 2)
def b : ℝ × ℝ := (1, 1)
def c : ℝ × ℝ := (1, 3)
def two_a_minus_b : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)
def perp : Prop := two_a_minus_b.1 * c.1 + two_a_minus_b.2 * c.2 = 0

-- The theorem we want to prove.
theorem find_m : perp m → m = -4 := by
  sorry

end find_m_l538_538865


namespace circle_equation_l538_538097

-- Defining the conditions
def circle_center_on_x_axis (a : ℝ) : Prop := ∃ (a : ℝ), true
def circle_radius (r : ℝ) : Prop := r = sqrt 5
def center_left_of_y_axis (a : ℝ) : Prop := a < 0
def tangent_to_line (a : ℝ) (r : ℝ) : Prop := (abs (a + 0)) / sqrt (1^2 + 2^2) = r

-- The main theorem statement
theorem circle_equation (a : ℝ) (r : ℝ) (x y : ℝ) 
    (h1 : circle_center_on_x_axis a)
    (h2 : circle_radius r)
    (h3 : center_left_of_y_axis a)
    (h4 : tangent_to_line a r) : 
    (x + 5)^2 + y^2 = 5 :=
sorry

end circle_equation_l538_538097


namespace poster_distance_from_wall_end_l538_538735

theorem poster_distance_from_wall_end (w_wall w_poster : ℝ) (h1 : w_wall = 25) (h2 : w_poster = 4) (h3 : 2 * x + w_poster = w_wall) : x = 10.5 :=
by
  sorry

end poster_distance_from_wall_end_l538_538735


namespace abs_diff_of_two_numbers_l538_538681

variable {x y : ℝ}

theorem abs_diff_of_two_numbers (h1 : x + y = 40) (h2 : x * y = 396) : abs (x - y) = 4 := by
  sorry

end abs_diff_of_two_numbers_l538_538681


namespace find_tricksters_l538_538353

theorem find_tricksters (inhabitants : Fin 65 → Prop) (is_knight : Fin 65 → Prop)
    (total_inhabitants : ∀ n, inhabitants n)
    (knights : ∀ n, is_knight n → inhabitants n)
    (tricksters_count : (∑ n, if ¬ is_knight n then 1 else 0) = 2)
    (knights_count : (∑ n, if is_knight n then 1 else 0) = 63)
    (knight_truth : ∀ n, is_knight n → ∀ l : list (Fin 65), (∀ m ∈ l, is_knight m) ↔ true)
    (ask_question : ∀ n, inhabitants n → ∀ l : list (Fin 65), bool) :
  ∃ (find_tricksters_function : (Fin 65 → Prop) → (Fin 65 → bool) → (list (Fin 65))) ,
    (length (find_tricksters_function inhabitants ask_question) ≤ 2) →
    (length (find_tricksters_function inhabitants ask_question) = 2) ∧
    ∀ t ∈ (find_tricksters_function inhabitants ask_question), ¬ is_knight t :=
by sorry

end find_tricksters_l538_538353


namespace arithmetic_sequence_sum_l538_538473

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (b : ℕ → ℚ) (T : ℕ → ℚ) :
  (a 3 = 7) ∧ (a 5 + a 7 = 26) →
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, S n = n^2 + 2 * n) ∧
  (∀ n, b n = 1 / ((a n)^2 - 1)) →
  (∀ n, T n = n / (4 * (n + 1))) := sorry

end arithmetic_sequence_sum_l538_538473


namespace dot_product_eq_neg_two_l538_538044

variable {α : Type*} [InnerProductSpace ℝ α]

def cos_theta (a b : α) : ℝ := 1/4
def norm_a : ℝ := 2
def norm_b : ℝ := 4

theorem dot_product_eq_neg_two (a b : α) 
  (h_cos : cos_theta a b = 1/4)
  (h_norm_a : ‖a‖ = norm_a)
  (h_norm_b : ‖b‖ = norm_b) :
  ⟪a, b - a⟫ = -2 :=
by
  sorry

end dot_product_eq_neg_two_l538_538044


namespace odds_against_Z_l538_538512

theorem odds_against_Z (h₁ : odds_against_X = (3, 1))
                       (h₂ : odds_against_Y = (2, 3)) :
                       odds_against_Z = (17, 3) := by
sorry

end odds_against_Z_l538_538512


namespace proof_problem_l538_538588

noncomputable def g : ℝ → ℝ := sorry
axiom g_condition1 : g 2 = 2
axiom g_condition2 : ∀ (x y : ℝ), g (xy + g x) = x * g y + g x

/-- Let g : ℝ → ℝ be a function such that g(2) = 2 and
    g(xy + g(x)) = xg(y) + g(x) for all x, y ∈ ℝ. 
    Determine the number of possible values of g(1/3) and 
    the sum of all possible values of g(1/3). 
    Prove that m * t = 1/3, where m is the number of possible values, and t is their sum. -/
theorem proof_problem : ∃ (m t : ℝ), m * t = 1/3 :=
sorry

end proof_problem_l538_538588


namespace max_correct_answers_l538_538539

theorem max_correct_answers (a b c : ℕ) (h1 : a + b + c = 80) (h2 : 5 * a - 2 * c = 150) : a ≤ 44 :=
by
  sorry

end max_correct_answers_l538_538539


namespace sin_690_eq_negative_one_half_l538_538416

theorem sin_690_eq_negative_one_half : Real.sin (690 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_690_eq_negative_one_half_l538_538416


namespace NaCl_production_l538_538804

-- Define the number of initial moles for the reactants.
constant moles_NH4Cl : ℕ := 3
constant moles_NaOH : ℕ := 5
constant moles_NaHCO3 : ℕ := 1

-- Define the reaction steps as conditions.
axiom step1 : moles_NH4Cl ≤ moles_NaOH

-- Define the production of NaCl in step 1.
def NaCl_produced_in_step1 := moles_NH4Cl

-- Define a lemma stating the production in step 2.
lemma no_NaCl_produced_in_step2 : 0 = 0 := rfl

-- Define a lemma stating the production in step 3.
lemma no_NaCl_produced_in_step3 : 0 = 0 := rfl

-- The theorem that encapsulates our conditions and the final answer.
theorem NaCl_production : NaCl_produced_in_step1 = 3 := by
  -- Generating the NaCl total production considering steps 2 and 3 do not produce NaCl.
  have h : 0 + 0 + NaCl_produced_in_step1 = moles_NH4Cl := by
    rw [no_NaCl_produced_in_step2, no_NaCl_produced_in_step3]
    exact rfl
  exact Nat.eq_refl moles_NH4Cl


end NaCl_production_l538_538804


namespace total_payment_combined_l538_538299

-- Definitions as per conditions
def no_discount (amt : ℝ) : ℝ :=
  if amt ≤ 200 then amt else 0

def discount_10 (amt : ℝ) : ℝ :=
  if (amt > 200 ∧ amt ≤ 500) then amt * 0.1 else 0

def discount_portion_500 (amt : ℝ) : ℝ :=
  if amt > 500 then 500 * 0.1 else 0

def discount_above_500 (amt : ℝ) : ℝ :=
  if amt > 500 then (amt - 500) * 0.3 else 0

def calculate_total_payment (amt : ℝ) : ℝ :=
  let discount1 := no_discount amt
  let discount2 := discount_10 amt
  let discount3 := discount_portion_500 amt
  let discount4 := discount_above_500 amt
  amt - discount2 - discount3 - discount4

-- Prices of products A and B separately
def price_A : ℝ := 168
def price_B : ℝ := 423

-- Total price
def total_price : ℝ := price_A + price_B

-- Lean 4 statement to prove total payment for combined transaction
theorem total_payment_combined : calculate_total_payment total_price = 546.6 := by
  sorry

end total_payment_combined_l538_538299


namespace cos_A_of_given_conditions_l538_538544

-- Define the right triangle with given conditions
structure RightTriangle (A B C : Type) where
  angle_C : ∠ C = 90
  AB : ℝ
  AC : ℝ
  BC : ℝ
  h_hypotenuse : AB = sqrt (AC^2 + BC^2)

noncomputable def cos_angle_A {A B C : Type} (triangle : RightTriangle A B C) : ℝ :=
  triangle.AC / triangle.AB

theorem cos_A_of_given_conditions {A B C : Type} (triangle : RightTriangle A B C)
  (hAB : triangle.AB = 6) (hAC : triangle.AC = 2) : cos_angle_A triangle = 1/3 :=
by
  rw [cos_angle_A]
  rw [hAC]
  rw [hAB]
  norm_num
  sorry

end cos_A_of_given_conditions_l538_538544


namespace point_above_line_l538_538049

theorem point_above_line (a : ℝ) : 3 * (-3) - 2 * (-1) - a < 0 ↔ a > -7 :=
by sorry

end point_above_line_l538_538049


namespace dot_product_result_l538_538046

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Conditions
def cos_angle_a_b : ℝ := 1 / 4
def magnitude_a : ℝ := 2
def magnitude_b : ℝ := 4

-- Correct Answer
theorem dot_product_result :
  (∥a∥ = magnitude_a) →
  (∥b∥ = magnitude_b) →
  (real_inner a b = ∥a∥ * ∥b∥ * cos_angle_a_b) →
  (real_inner a (b - a) = -2) :=
by
  intros h_magnitude_a h_magnitude_b h_cos_inner
  sorry

end dot_product_result_l538_538046


namespace seating_possible_l538_538169

-- Define the conditions and questions of the problem
structure SeatingGrid where
  students : Fin 169
  vertex : Fin 13 × Fin 13
  
def distance (p1 p2 : Fin 13 × Fin 13) : ℝ :=
  real.sqrt (((p2.1 - p1.1).val)^2 + ((p2.2 - p1.2).val)^2)

structure HistoryClass where
  n : ℕ := 169
  grid : Fin 169 → (Fin 13 × Fin 13)
  best_friends : Fin 169 → Finset (Fin 169)
  mutual : ∀ s t, t ∈ best_friends s ↔ s ∈ best_friends t
  max_friends : ∀ s, (best_friends s).card ≤ 3
  disruptive : ∀ s t, t ∈ best_friends s → distance (grid s) (grid t) ≤ 3 → False

theorem seating_possible (h : HistoryClass) : 
  ∃ seating : Fin 169 → (Fin 13 × Fin 13),
  (∀ s t, t ∈ (h.best_friends s) → distance (seating s) (seating t) > 3) :=
sorry

end seating_possible_l538_538169


namespace schedule_courses_l538_538086

/-- Definition of valid schedule count where at most one pair of courses is consecutive. -/
def count_valid_schedules : ℕ := 180

/-- Given 7 periods and 3 courses, determine the number of valid schedules 
    where at most one pair of these courses is consecutive. -/
theorem schedule_courses (periods : ℕ) (courses : ℕ) (valid_schedules : ℕ) :
  periods = 7 → courses = 3 → valid_schedules = count_valid_schedules →
  valid_schedules = 180 :=
by
  intros h1 h2 h3
  sorry

end schedule_courses_l538_538086


namespace dot_product_result_l538_538045

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Conditions
def cos_angle_a_b : ℝ := 1 / 4
def magnitude_a : ℝ := 2
def magnitude_b : ℝ := 4

-- Correct Answer
theorem dot_product_result :
  (∥a∥ = magnitude_a) →
  (∥b∥ = magnitude_b) →
  (real_inner a b = ∥a∥ * ∥b∥ * cos_angle_a_b) →
  (real_inner a (b - a) = -2) :=
by
  intros h_magnitude_a h_magnitude_b h_cos_inner
  sorry

end dot_product_result_l538_538045


namespace sin_690_deg_l538_538427

noncomputable def sin_690_eq_neg_one_half : Prop :=
  sin (690 * real.pi / 180) = -(1 / 2)

theorem sin_690_deg : sin_690_eq_neg_one_half :=
  by sorry

end sin_690_deg_l538_538427


namespace voldemort_caloric_intake_l538_538234

theorem voldemort_caloric_intake :
  ∀ (cake calories_chips calories_coke calories_breakfast calories_lunch daily_limit : ℕ),
  cake = 110 →
  calories_chips = 310 →
  calories_coke = 215 →
  calories_breakfast = 560 →
  calories_lunch = 780 →
  daily_limit = 2500 →
  (daily_limit - (calories_breakfast + calories_lunch + cake + calories_chips + calories_coke) = 525) :=
begin
  intros,
  sorry
end

end voldemort_caloric_intake_l538_538234


namespace fraction_unspent_is_correct_l538_538762

noncomputable def fraction_unspent (S : ℝ) : ℝ :=
  let after_tax := S - 0.15 * S
  let after_first_week := after_tax - 0.25 * after_tax
  let after_second_week := after_first_week - 0.3 * after_first_week
  let after_third_week := after_second_week - 0.2 * S
  let after_fourth_week := after_third_week - 0.1 * after_third_week
  after_fourth_week / S

theorem fraction_unspent_is_correct (S : ℝ) (hS : S > 0) : 
  fraction_unspent S = 0.221625 :=
by
  sorry

end fraction_unspent_is_correct_l538_538762


namespace greatest_whole_number_satisfying_inequalities_l538_538446

theorem greatest_whole_number_satisfying_inequalities :
  ∃ x : ℕ, 3 * (x : ℤ) - 5 < 1 - x ∧ 2 * (x : ℤ) + 4 ≤ 8 ∧ ∀ y : ℕ, y > x → ¬ (3 * (y : ℤ) - 5 < 1 - y ∧ 2 * (y : ℤ) + 4 ≤ 8) :=
sorry

end greatest_whole_number_satisfying_inequalities_l538_538446


namespace exists_angle_X_l538_538576

theorem exists_angle_X (A B C X : ℝ) (hB : 0 < B ∧ B < π/2) (hC : 0 < C ∧ C < π/2) :
  ∃ X, sin X = (sin B * sin C) / (1 - cos A * cos B * cos C) := by
  sorry

end exists_angle_X_l538_538576


namespace sin_690_eq_neg_half_l538_538403

theorem sin_690_eq_neg_half :
  let rad := Real.pi / 180 in -- Convert degrees to radians
  Real.sin (690 * rad) = -1 / 2 :=
by
  sorry

end sin_690_eq_neg_half_l538_538403


namespace probability_of_picking_peach_l538_538110

-- Define the counts of each type of fruit
def apples : ℕ := 5
def pears : ℕ := 3
def peaches : ℕ := 2

-- Define the total number of fruits
def total_fruits : ℕ := apples + pears + peaches

-- Define the probability of picking a peach
def probability_of_peach : ℚ := peaches / total_fruits

-- State the theorem
theorem probability_of_picking_peach : probability_of_peach = 1/5 := by
  -- proof goes here
  sorry

end probability_of_picking_peach_l538_538110


namespace rule_for_sequence_natural_number_self_map_power_of_2_to_single_digit_l538_538922

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def transition_rule (n : ℕ) : ℕ :=
  2 * (sum_of_digits n)

theorem rule_for_sequence :
  transition_rule 3 = 6 ∧ transition_rule 6 = 12 :=
by
  sorry

theorem natural_number_self_map :
  ∀ n : ℕ, transition_rule n = n ↔ n = 18 :=
by
  sorry

theorem power_of_2_to_single_digit :
  ∃ x : ℕ, transition_rule (2^1991) = x ∧ x < 10 :=
by
  sorry

end rule_for_sequence_natural_number_self_map_power_of_2_to_single_digit_l538_538922


namespace jerry_added_7_figures_l538_538132

theorem jerry_added_7_figures :
  ∀ (x : ℕ), let initial_figures := 5,
                 initial_books := 9,
                 total_figures := initial_figures + x,
                 total_books := initial_books in
              total_figures = total_books + 3 → x = 7 :=
by
  intro x
  dsimp only
  intro h
  sorry

end jerry_added_7_figures_l538_538132


namespace find_profit_pct_l538_538308

variables (total_sugar sold_at_12 overall_profit : ℝ)
variables (quantity_sold_remaining profit_pct_remaining : ℝ)

-- Given conditions
def total_sugar := 1600
def sold_at_12 := 1200
def overall_profit := 0.11

-- Calculate remaining quantity and the profit percentage on the remaining quantity
def quantity_sold_remaining := total_sugar - sold_at_12
def profit_total := total_sugar * overall_profit
def profit_from_12 := sold_at_12 * 0.12
def profit_needed_remaining := profit_total - profit_from_12

def profit_pct_remaining := profit_needed_remaining / quantity_sold_remaining

theorem find_profit_pct : profit_pct_remaining = 0.08 := by
  sorry

end find_profit_pct_l538_538308


namespace ball_bounce_time_formula_l538_538080

variable (h0 k g : ℝ)
#check h0
#check k
#check g
def ball_bounce_time (h0 k g : ℝ) : ℝ :=
  (1 + Real.sqrt k) / (1 - Real.sqrt k) * Real.sqrt (2 * h0 / g)

theorem ball_bounce_time_formula (h0 k g : ℝ) : 
  ball_bounce_time h0 k g = (1 + Real.sqrt k) / (1 - Real.sqrt k) * Real.sqrt (2 * h0 / g) := 
  sorry

end ball_bounce_time_formula_l538_538080


namespace find_x1_l538_538835

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 3) : 
  x1 = 4 / 5 := 
  sorry

end find_x1_l538_538835


namespace inequality_solution_set_l538_538489

theorem inequality_solution_set (a b x : ℝ) (h1 : a > 0) (h2 : b = a) :
  (ax - b > 0 → x ∈ (1, +∞)) →
  ∀x, (ax + b) / (x - 2) ≤ 3a - b ↔ x ∈ (-∞, 2) ∪ [5, +∞) := by
  intros h3 x
  have h4 : a = b := h2
  sorry

end inequality_solution_set_l538_538489


namespace find_a_l538_538595

-- Define the problem conditions
variables (a : ℝ) (i : ℂ)
def z : ℂ := (a + i) / (1 + i)

-- State that i is the imaginary unit
axiom imag_unit : i * i = -1

-- State that z is a pure imaginary number
def is_pure_imag (z : ℂ) : Prop := z.re = 0

-- The final theorem statement
theorem find_a (h : is_pure_imag ((a + i) / (1 + i))) : a = -1 := 
sorry

end find_a_l538_538595


namespace roses_in_centerpiece_l538_538167

variable (r : ℕ)

theorem roses_in_centerpiece (h : 6 * 15 * (3 * r + 6) = 2700) : r = 8 := 
  sorry

end roses_in_centerpiece_l538_538167


namespace train_distance_l538_538753

theorem train_distance (t : ℝ) : 
  let s := 3 + 120 * t in 
  s = 3 + 120 * t :=
by {
  sorry
}

end train_distance_l538_538753


namespace num_positive_integers_l538_538790

noncomputable def countPositiveIntegers (a b : ℝ) (c : ℕ) : ℕ :=
  (finset.filter (λ n, (a * n) ^ c > n ^ (3 * c)) (Icc 1 (int.to_nat (real.sqrt a))))
  .to_finset.card

theorem num_positive_integers : countPositiveIntegers 105 3 30 = 1 :=
  sorry

end num_positive_integers_l538_538790


namespace initial_budget_is_60_l538_538251

-- Define the initial budget and conditions
variable (B : ℝ)

-- Conditions: new price is 1.20B, smaller frame price is 0.90B, and remaining $6 after purchase.
def new_price := 1.20 * B
def smaller_frame_price := 0.90 * B
def remaining_amount := B - 0.90 * B

-- The theorem to prove the initial budget B is $60
theorem initial_budget_is_60 (h1 : new_price = 1.20 * B)
                            (h2 : smaller_frame_price = 0.90 * B)
                            (h3 : remaining_amount = 6) : B = 60 := by
  sorry

end initial_budget_is_60_l538_538251


namespace count_256_nums_with_5_or_6_in_base_8_l538_538874

def uses_digit_5_or_6 (n : ℕ) : Prop :=
  ∃ k : ℕ, 5 = (n / (8 ^ k)) % 8 ∨ 6 = (n / (8 ^ k)) % 8

def count_uses_digit_5_or_6 : ℕ :=
  (Finset.range 256).filter uses_digit_5_or_6 |>.card

theorem count_256_nums_with_5_or_6_in_base_8 :
  count_uses_digit_5_or_6 = 220 :=
  sorry

end count_256_nums_with_5_or_6_in_base_8_l538_538874


namespace bug_visits_tiles_l538_538737

open Int

theorem bug_visits_tiles (width_tiles length_tiles : ℕ) (h_width : width_tiles = 6) (h_length : length_tiles = 13) :
  width_tiles + length_tiles - gcd width_tiles length_tiles = 18 := by
  sorry

end bug_visits_tiles_l538_538737


namespace num_divisible_by_10_in_range_correct_l538_538554

noncomputable def num_divisible_by_10_in_range : ℕ :=
  let a1 := 100
  let d := 10
  let an := 500
  (an - a1) / d + 1

theorem num_divisible_by_10_in_range_correct :
  num_divisible_by_10_in_range = 41 := by
  sorry

end num_divisible_by_10_in_range_correct_l538_538554


namespace part_one_part_two_l538_538059

def f (x : ℝ) : ℝ := |x - 1|

theorem part_one (x : ℝ) :
  f(x) + f(x + 4) ≥ 8 ↔ x ≤ -5 ∨ x ≥ 3 := 
by
  sorry

theorem part_two (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) (h : a ≠ 0) :
  f(a * b) > |a| * f(b / a) := 
by
  sorry

end part_one_part_two_l538_538059


namespace quadratic_root_c_l538_538107

theorem quadratic_root_c : ∃ c : ℝ, (∀ x : ℝ, 2 * x^2 + 16 * x + c = 0 ↔ x = (-16 + real.sqrt 24) / 4 ∨ x = (-16 - real.sqrt 24) / 4) → c = 29 :=
by {
    sorry
}

end quadratic_root_c_l538_538107


namespace tangent_lines_to_curve_l538_538438

-- Define the curve
def curve (x : ℝ) : ℝ := x^3

-- Define the general form of a tangent line
def tangent_line (x : ℝ) (y : ℝ) (m : ℝ) (x0 : ℝ) (y0 : ℝ) : Prop :=
  y - y0 = m * (x - x0)

-- Define the conditions
def condition1 : Prop :=
  tangent_line 1 1 3 1 1

def condition2 : Prop :=
  tangent_line 1 1 (3/4) (-1/2) ((-1/2)^3)

-- Define the equations of the tangent lines
def line1 : Prop :=
  ∀ x y : ℝ, 3 * x - y - 2 = 0

def line2 : Prop :=
  ∀ x y : ℝ, 3 * x - 4 * y + 1 = 0

-- The final theorem statement
theorem tangent_lines_to_curve :
  (condition1 → line1) ∧ (condition2 → line2) :=
  by
    sorry -- Placeholder for proof

end tangent_lines_to_curve_l538_538438


namespace area_of_triangle_ABF_l538_538032

theorem area_of_triangle_ABF (A B F : ℝ × ℝ) (hF : F = (1, 0)) (hA_parabola : A.2^2 = 4 * A.1) (hB_parabola : B.2^2 = 4 * B.1) (h_midpoint_AB : (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 2) : 
  ∃ area : ℝ, area = 2 :=
sorry

end area_of_triangle_ABF_l538_538032


namespace smallest_positive_period_of_f_l538_538494

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * cos (ω * x - π / 4) + 3

theorem smallest_positive_period_of_f (ω : ℝ) (h : 1 < ω ∧ ω < 2) 
    (symmetry : ∀ x : ℝ, f ω (x + π) = f ω (π - x)) :
    ∃ T > 0, ∀ x, f ω (x + T) = f ω x ∧ T = 8 * π / 5 := 
sorry

end smallest_positive_period_of_f_l538_538494


namespace find_tricksters_l538_538362

def inhab_group : Type := { n : ℕ // n < 65 }
def is_knight (i : inhab_group) : Prop := ∀ g : inhab_group, true -- Placeholder for the actual property

theorem find_tricksters (inhabitants : inhab_group → Prop)
  (is_knight : inhab_group → Prop)
  (knight_always_tells_truth : ∀ i : inhab_group, is_knight i → inhabitants i = true)
  (tricksters_2_and_rest_knights : ∃ t1 t2 : inhab_group, t1 ≠ t2 ∧ ¬is_knight t1 ∧ ¬is_knight t2 ∧
    (∀ i : inhab_group, i ≠ t1 → i ≠ t2 → is_knight i)) :
  ∃ find_them : inhab_group → inhab_group → Prop, (∀ q_count : ℕ, q_count ≤ 16) → 
  ∃ t1 t2 : inhab_group, find_them t1 t2 :=
by 
  -- The proof goes here
  sorry

end find_tricksters_l538_538362


namespace locus_midpoint_l538_538055

/-- Given a fixed point A (4, -2) and a moving point B on the curve x^2 + y^2 = 4,
    prove that the locus of the midpoint P of the line segment AB satisfies the equation 
    (x - 2)^2 + (y + 1)^2 = 1. -/
theorem locus_midpoint (A B P : ℝ × ℝ)
  (hA : A = (4, -2))
  (hB : ∃ (x y : ℝ), B = (x, y) ∧ x^2 + y^2 = 4)
  (hP : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (P.1 - 2)^2 + (P.2 + 1)^2 = 1 :=
sorry

end locus_midpoint_l538_538055


namespace complex_number_problem_l538_538036

noncomputable def i : ℂ := complex.I

theorem complex_number_problem (m : ℝ) (hm : (1 - m * i) / (i^3) = 1 + i) : m = 1 :=
by
  sorry

end complex_number_problem_l538_538036


namespace part1_part2_l538_538122

variable {a b c m t y1 y2 : ℝ}

-- Condition: point (2, m) lies on the parabola y = ax^2 + bx + c where axis of symmetry is x = t
def point_lies_on_parabola (a b c m : ℝ) := m = a * 2^2 + b * 2 + c

-- Condition: axis of symmetry x = t
def axis_of_symmetry (a b t : ℝ) := t = -b / (2 * a)

-- Condition: m = c
theorem part1 (a c : ℝ) (h : m = c) (h₀ : point_lies_on_parabola a (-2 * a) c m) :
  axis_of_symmetry a (-2 * a) 1 :=
by sorry

-- Additional Condition: c < m
def c_lt_m (c m : ℝ) := c < m

-- Points (-1, y1) and (3, y2) lie on the parabola y = ax^2 + bx + c
def points_on_parabola (a b c y1 y2 : ℝ) :=
  y1 = a * (-1)^2 + b * (-1) + c ∧ y2 = a * 3^2 + b * 3 + c

-- Comparison result
theorem part2 (a : ℝ) (h₁ : c_lt_m c m) (h₂ : 2 * a + (-2 * a) > 0) (h₂' : points_on_parabola a (-2 * a) c y1 y2) :
  y2 > y1 :=
by sorry

end part1_part2_l538_538122


namespace decreasing_sequence_exists_l538_538374

-- Definition of the sequence and conditions for n ≥ 2018
variable (a : ℕ → ℝ)
variable (N : ℕ)
variable (P : ℕ → Polynomial ℝ)
variable (h2018 : 2018 ≤ N)
variable (h_seq : ∀ n ≥ 2018, ∀ x, P n (a (n+1)) = x^(2*n) + a 1 * x^(2*n-2) + a 2 * x^(2*n-4) + ... + a n)

-- statement of the proof problem
theorem decreasing_sequence_exists :
  ∃ (N : ℕ), (N ≥ 2018) ∧ ∀ (m : ℕ), (m ≥ N) → a (m+1) < a m :=
sorry

end decreasing_sequence_exists_l538_538374


namespace stockholm_to_uppsala_distance_l538_538999

-- Definitions based on conditions
def map_distance_cm : ℝ := 3
def scale_cm_to_km : ℝ := 80

-- Theorem statement based on the question and correct answer
theorem stockholm_to_uppsala_distance : 
  (map_distance_cm * scale_cm_to_km = 240) :=
by 
  -- This is where the proof would go
  sorry

end stockholm_to_uppsala_distance_l538_538999


namespace area_bounded_by_arccos_sin_x_is_2_pi_squared_l538_538800

noncomputable def area_bounded_by_arccos_sin_x : ℝ :=
  ∫ x in (Real.Icc (π/2) (7 * π / 2)), Real.arccos (Real.sin x)

theorem area_bounded_by_arccos_sin_x_is_2_pi_squared :
  area_bounded_by_arccos_sin_x = 2 * π^2 :=
  sorry

end area_bounded_by_arccos_sin_x_is_2_pi_squared_l538_538800


namespace axis_of_symmetry_parabola_l538_538052

theorem axis_of_symmetry_parabola (x y : ℝ) :
  x^2 + 2*x*y + y^2 + 3*x + y = 0 → x + y + 1 = 0 :=
by {
  sorry
}

end axis_of_symmetry_parabola_l538_538052


namespace range_for_positive_f_l538_538596

noncomputable def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

noncomputable def derivative_condition (f f' : ℝ → ℝ) := ∀ x : ℝ, x > 0 → x * f' x - f x < 0

theorem range_for_positive_f (f f' : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_f_m1 : f (-1) = 0)
  (h_deriv_cond : derivative_condition f f') :
  ∀ x : ℝ, f x > 0 ↔ x ∈ (Iio (-1) ∪ Ioo 0 1) :=
sorry

end range_for_positive_f_l538_538596


namespace prime_multiple_l538_538575

theorem prime_multiple (p m n : ℕ) (hp_prime : Nat.Prime p) (hp_gt_two : p > 2)
  (h_frac_sum : 1 + ∑ k in Finset.range (p - 1), (1 : ℚ) / (k + 1 : ℚ) ^ 3 = m / n)
  (h_coprime : Nat.coprime m n): p ∣ m :=
sorry

end prime_multiple_l538_538575


namespace bisector_triangle_relation_l538_538664

noncomputable def triangle_relation (a b c : ℝ) : Prop :=
(b^2 - a^2)^2 = c^2 * (a^2 + b^2)

theorem bisector_triangle_relation (a b c : ℝ) 
  (h : internal_and_external_bisectors_equal a b c) : 
  triangle_relation a b c :=
sorry


end bisector_triangle_relation_l538_538664


namespace fraction_of_square_shaded_l538_538626

theorem fraction_of_square_shaded (s : ℝ) :
  let area_square := s^2,
      area_shaded := s^2 / 2
  in (area_shaded / area_square) = 1 / 2 :=
by
  sorry

end fraction_of_square_shaded_l538_538626


namespace sin_690_degree_l538_538423

theorem sin_690_degree : sin (690 : ℝ) * (Real.pi / 180) = -(1 / 2) := by
  sorry

end sin_690_degree_l538_538423


namespace point_on_transformed_plane_l538_538149

def A := (2, 5, 1)
def plane (x y z : ℝ) := 5 * x - 2 * y + z - 3 = 0
def k := 1 / 3
def transformed_plane (x y z : ℝ) := 5 * x - 2 * y + z - 1 = 0

theorem point_on_transformed_plane :
  transformed_plane A.1 A.2 A.3 := by
  sorry

end point_on_transformed_plane_l538_538149


namespace final_coordinates_l538_538207

noncomputable def initial_point : ℝ × ℝ × ℝ := (2, 2, 2)

def rotate_y_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(- p.1, p.2, - p.3)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(- p.1, p.2, p.3)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(p.1, - p.2, p.3)

def rotate_x_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(p.1, - p.2, - p.3)

def final_transformation (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
reflect_yz (rotate_x_180 (reflect_xz (reflect_yz (rotate_y_180 p))))

theorem final_coordinates : final_transformation initial_point = (-2, 2, 2) :=
by
  simp [initial_point, rotate_y_180, reflect_yz, reflect_xz, rotate_x_180, final_transformation]
  -- This will simplify through all the steps and get to the final tuple
  sorry

end final_coordinates_l538_538207


namespace min_distance_from_ellipse_to_line_l538_538830

-- Define the ellipse and the line
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 9 = 1
def line (x y : ℝ) : Prop := 2 * x + y - 10 = 0

-- The minimum distance we need to prove
def min_distance := sqrt 5

-- The theorem stating that the minimum distance from any point on the ellipse to the line is sqrt(5)
theorem min_distance_from_ellipse_to_line :
  ∀ (x y : ℝ), ellipse x y → ∃ (d : ℝ), d = min_distance ∧ (∀ x' y', line x' y' → dist (x, y) (x', y') ≥ d) :=
by {
  sorry
}

end min_distance_from_ellipse_to_line_l538_538830


namespace find_pairs_l538_538441

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  (3 * a + 1 ∣ 4 * b - 1) ∧ (2 * b + 1 ∣ 3 * a - 1) ↔ (a = 2 ∧ b = 2) := 
by 
  sorry

end find_pairs_l538_538441


namespace Sophie_donuts_problem_l538_538643

noncomputable def total_cost_before_discount (cost_per_box : ℝ) (num_boxes : ℕ) : ℝ :=
  cost_per_box * num_boxes

noncomputable def discount_amount (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  total_cost * discount_rate

noncomputable def total_cost_after_discount (total_cost : ℝ) (discount : ℝ) : ℝ :=
  total_cost - discount

noncomputable def total_donuts (donuts_per_box : ℕ) (num_boxes : ℕ) : ℕ :=
  donuts_per_box * num_boxes

noncomputable def donuts_left (total_donuts : ℕ) (donuts_given_away : ℕ) : ℕ :=
  total_donuts - donuts_given_away

theorem Sophie_donuts_problem
  (budget : ℝ)
  (cost_per_box : ℝ)
  (discount_rate : ℝ)
  (num_boxes : ℕ)
  (donuts_per_box : ℕ)
  (donuts_given_to_mom : ℕ)
  (donuts_given_to_sister : ℕ)
  (half_dozen : ℕ) :
  budget = 50 →
  cost_per_box = 12 →
  discount_rate = 0.10 →
  num_boxes = 4 →
  donuts_per_box = 12 →
  donuts_given_to_mom = 12 →
  donuts_given_to_sister = 6 →
  half_dozen = 6 →
  total_cost_after_discount (total_cost_before_discount cost_per_box num_boxes) (discount_amount (total_cost_before_discount cost_per_box num_boxes) discount_rate) = 43.2 ∧
  donuts_left (total_donuts donuts_per_box num_boxes) (donuts_given_to_mom + donuts_given_to_sister) = 30 :=
by
  sorry

end Sophie_donuts_problem_l538_538643


namespace cade_gave_dylan_eight_marbles_l538_538768

-- Declaring the conditions
axiom original_marbles : ℕ
axiom remaining_marbles : ℕ
  
-- Given conditions
axiom h1 : original_marbles = 87
axiom h2 : remaining_marbles = 79

-- Define the number of marbles given
def marbles_given : ℕ := original_marbles - remaining_marbles

-- Theorem to prove
theorem cade_gave_dylan_eight_marbles : marbles_given = 8 :=
by {
  -- Prove the theorem using the given conditions
  rw [h1, h2],
  sorry -- proof goes here
}

end cade_gave_dylan_eight_marbles_l538_538768


namespace find_m_l538_538862

-- Conditions
def A : Set ℝ := {1, 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | m * x - 3 = 0}

-- Lean Statement
theorem find_m (m : ℝ) : (A ∪ B m = A) → (m = 0 ∨ m = 1 ∨ m = 3) :=
begin
  sorry
end

end find_m_l538_538862


namespace product_is_minus_45i_l538_538946

noncomputable def Q : ℂ := 6 + 3 * complex.I
noncomputable def E : ℂ := -complex.I
noncomputable def D : ℂ := 6 - 3 * complex.I

theorem product_is_minus_45i : Q * E * D = -45 * complex.I := 
by {
  sorry
}

end product_is_minus_45i_l538_538946


namespace find_tricksters_l538_538326

structure Inhabitant :=
  (isKnight : Prop)

constants (inhabitants : Fin 65 → Inhabitant)
          (tricksters : Fin 2 → Fin 65)

axiom two_tricksters_unique :
  ∃! (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65))

axiom valid_question :
  ∀ (a : Fin 65) (group : Set (Fin 65)), (inhabitants a).isKnight → 
  (∀ i ∈ group, (inhabitants i).isKnight) ↔ 
  (knight a).isKnight

theorem find_tricksters :
  ∃ (q : ℕ), q ≤ 16 ∧
  ∃ (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65)) :=
sorry

end find_tricksters_l538_538326


namespace shape_of_intersections_is_rectangle_l538_538449

open Real

theorem shape_of_intersections_is_rectangle :
  (∀ (x y : ℝ), (xy = 18 ∧ x^2 + y^2 = 45) →
    set_of_points_formed_is_rectangle [{(6, 3), (-6, -3), (3, 6), (-3, -6)}]) :=
by
  sorry -- Proof to be filled in later

end shape_of_intersections_is_rectangle_l538_538449


namespace sum_arithmetic_series_angle_identity_triangle_angles_probability_multiple_of_5_l538_538515

-- 1. Prove A = 380 given A = 11 + 12 + 13 + ... + 29
theorem sum_arithmetic_series : (11 + 12 + 13 + ... + 29) = 380 :=
  sorry

-- 2. Prove B = 70 given \sin 20^\circ = \cos B^\circ, 0 < B < 90
theorem angle_identity : ∀ (B : ℝ), sin 20 = cos B ∧ 0 < B ∧ B < 90 → B = 70 :=
  sorry

-- 3. Prove n = 60 given ∠PQR = 70, ∠PRQ = 50, and ∠QSR = n 
theorem triangle_angles : ∀ (n : ℝ), (70 + 50 + n = 180) ∧ (n = ∠QSR) -> n = 60 :=
  sorry

-- 4. Prove m = 5 given n cards marked from 1 to n and chance of drawing a multiple of 5 = 1/m
theorem probability_multiple_of_5 : ∀ (n m : ℕ), (m * ⌊n / 5⌋ = n) ∧ (1 / m = ⌊n / 5⌋ / n) → m = 5 :=
  sorry

end sum_arithmetic_series_angle_identity_triangle_angles_probability_multiple_of_5_l538_538515


namespace Mark_spends_more_time_l538_538164

-- Definition of constants associated with the problem
def Chris_speed : ℝ := 3 -- miles per hour
def School_distance : ℝ := 9 -- miles
def Mark_turnaround_distance : ℝ := 3 -- miles

-- Additional calculated distances
def Mark_total_walk : ℝ := 2 * Mark_turnaround_distance + School_distance

-- Calculation of times taken
def Chris_time : ℝ := School_distance / Chris_speed
def Mark_time : ℝ := 2 * (Mark_turnaround_distance / Chris_speed) + (School_distance / Chris_speed)

-- Theorem statement
theorem Mark_spends_more_time : Mark_time - Chris_time = 2 :=
by
  sorry

end Mark_spends_more_time_l538_538164


namespace infinite_primes_divide_conditions_l538_538574

theorem infinite_primes_divide_conditions (a : ℤ) :
  ∃^∞ p : ℕ, (∃ n m : ℤ, p ∣ n^2 + 3 ∧ p ∣ m^3 - a) :=
sorry

end infinite_primes_divide_conditions_l538_538574


namespace trains_crossing_time_l538_538260

noncomputable def time_to_cross (l1 l2 v1 v2 : ℝ) : ℝ :=
  let relative_speed := (v1 + v2) * (5/18) in
  let total_distance := l1 + l2 in
  total_distance / relative_speed

theorem trains_crossing_time :
  time_to_cross 180 160 60 40 ≈ 12.23 :=
by
  unfold time_to_cross
  -- skip the detailed computation steps
  sorry  -- This will be the placeholder for the proof

end trains_crossing_time_l538_538260


namespace sin_690_eq_negative_one_half_l538_538415

theorem sin_690_eq_negative_one_half : Real.sin (690 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_690_eq_negative_one_half_l538_538415


namespace debt_payment_installments_l538_538273

theorem debt_payment_installments (n : ℕ) (h1 : ∀ i, i < 20 → payment i = 410)
  (h2 : ∀ i, 20 ≤ i ∧ i < n → payment i = 475)
  (h_avg : (∑ i in range n, payment i) / n = 455) : n = 65 := by
  sorry

def payment (i : ℕ) : ℤ := sorry -- Define payment as per the conditions

lemma payment_eq_410 (i : ℕ) (h : i < 20) : payment i = 410 := sorry

lemma payment_eq_475 (i : ℕ) (h : 20 ≤ i ∧ i < n) : payment i = 475 := sorry

lemma avg_eq_455 : (∑ i in range n, payment i) / n = 455 := sorry

end debt_payment_installments_l538_538273


namespace sin_690_eq_neg_half_l538_538409

theorem sin_690_eq_neg_half :
  Real.sin (690 * Real.pi / 180) = -1 / 2 :=
by {
  sorry
}

end sin_690_eq_neg_half_l538_538409


namespace no_perfect_square_with_300_ones_l538_538793

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℤ, n = k * k

theorem no_perfect_square_with_300_ones : ¬ ∃ n : ℕ, 
  (nat.digits 10 n).count 1 = 300 ∧
  (nat.digits 10 n).count' 0 = (nat.digits 10 n).length - 300 ∧
  is_perfect_square n := 
sorry

end no_perfect_square_with_300_ones_l538_538793


namespace cafeteria_total_cost_l538_538607

-- Definitions based on conditions
def cost_per_coffee := 4
def cost_per_cake := 7
def cost_per_ice_cream := 3
def mell_coffee := 2 
def mell_cake := 1 
def friends_coffee := 2 
def friends_cake := 1 
def friends_ice_cream := 1 
def num_friends := 2
def total_coffee := mell_coffee + num_friends * friends_coffee
def total_cake := mell_cake + num_friends * friends_cake
def total_ice_cream := num_friends * friends_ice_cream

-- Total cost
def total_cost := total_coffee * cost_per_coffee + total_cake * cost_per_cake + total_ice_cream * cost_per_ice_cream

-- Theorem statement
theorem cafeteria_total_cost : total_cost = 51 := by
  sorry

end cafeteria_total_cost_l538_538607


namespace circle_distance_range_l538_538791

theorem circle_distance_range (m : ℝ) :
  (x^2 + y^2 + 2 * x + 4 * y + m = 0) → 
  (∀ (x y : ℝ), dist ⟨-1, -2⟩ ⟨x, y⟩ = sqrt 2) →
  -3 < m ∧ m < 5 :=
by {
  sorry
}

end circle_distance_range_l538_538791


namespace complex_modulus_z1_plus_z2_l538_538843

variable {z1 z2 : ℂ}

theorem complex_modulus_z1_plus_z2 
  (h1 : |z1| = 1) 
  (h2 : |z2| = 1) 
  (h3 : |z1 - z2| = 1) : 
  |z1 + z2| = sqrt 3 :=
sorry

end complex_modulus_z1_plus_z2_l538_538843


namespace angle_B_l538_538966

open Set

variables {Point Line : Type}

variable (l m n p : Line)
variable (A B C D : Point)
variable (angle : Point → Point → Point → ℝ)

-- Definitions of the conditions
def parallel (x y : Line) : Prop := sorry
def intersects (x y : Line) (P : Point) : Prop := sorry
def measure_angle (P Q R : Point) : ℝ := sorry

-- Assumptions based on conditions
axiom parallel_lm : parallel l m
axiom intersection_n_l : intersects n l A
axiom angle_A : measure_angle B A D = 140
axiom intersection_p_m : intersects p m C
axiom angle_C : measure_angle A C B = 70
axiom intersection_p_l : intersects p l D
axiom not_parallel_np : ¬ parallel n p

-- Proof goal
theorem angle_B : measure_angle C B D = 140 := sorry

end angle_B_l538_538966


namespace digit_count_of_sum_l538_538877

theorem digit_count_of_sum (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) :
  let sum := 9876 + 4 * 100 + A * 10 + 2 + B * 100 + 51 in
  sum >= 10000 ∧ sum < 100000 :=
by
  sorry

end digit_count_of_sum_l538_538877


namespace no_incorrect_statements_l538_538654

-- Conditions
variables {n : ℕ} {x y : ℝ} {λ μ : ℝ}
variables (samples_x : fin n → ℝ) (samples_y : fin n → ℝ)

-- Condition statements
def average (s : fin n → ℝ) : ℝ := (∑ i, s i) / n

def combined_average := λ * x + μ * y

def line_l (λ μ x y : ℝ) := (λ + 2) * x - (1 + 2 * μ) * y + 1 - 3 * λ = 0

def line_l' (λ μ x y : ℝ) := (2*λ - 3) * x - (3 - μ) * y = 0

def circle (x y : ℝ) := (x - 1)^2 + (y - 1)^2 = 4

-- Problem statement as a Lean theorem
theorem no_incorrect_statements
  (h₁ : average samples_x = x)
  (h₂ : average samples_y = y)
  (h₃ : y ≠ x)
  (h₄ : combined_average = λ * x + μ * y = (λ = 1/2 ∧ μ = 1/2)) :
  (line_l λ μ x y) ∧ 
  (line_l x y 1 1) ∧ 
  (∃ (x y : ℝ), circle x y ∧ line_l x y) ∧ 
  (abs ((-3/4) - (λ * x)) * (max_dist_to_origin = √2) ∧
  ((-3/4) = λ * x)) ∧
  (line_l' λ μ x y) ∧ 
  ((2*λ - 3) * x - (3 - μ) * y = 0 ∧ (slope (line_l) = -1 / slope (line_l'))) ↔ 0 := 
sorry

end no_incorrect_statements_l538_538654


namespace perpendicular_condition_l538_538071

variables {α : Type*} [plane α] (a b : line α) (l : line ℝ^3)

-- Assume the conditions of the problem
def lines_in_plane (a b : line α) : Prop := true -- lines a and b in plane α
def line_in_space (l : line ℝ^3) : Prop := true  -- line l in space
def line_perp (l : line ℝ^3) (x : line ℝ^3) : Prop := true -- l ⊥ x

theorem perpendicular_condition (h1 : lines_in_plane a b) (h2 : line_in_space l)
  (h3 : line_perp l a) (h4 : line_perp l b) : 
  (∃ ha hb, line_perp l a ∧ line_perp l b) ∧ ¬(line_perp l α) :=
sorry

end perpendicular_condition_l538_538071


namespace triangle_area_is_correct_l538_538472

noncomputable def triangle_area (a b c B : ℝ) : ℝ := 
  0.5 * a * c * Real.sin B

theorem triangle_area_is_correct :
  let a := Real.sqrt 2
  let c := Real.sqrt 2
  let b := Real.sqrt 6
  let B := 2 * Real.pi / 3 -- 120 degrees in radians
  triangle_area a b c B = Real.sqrt 3 / 2 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end triangle_area_is_correct_l538_538472


namespace solution_set_of_xf_x_gt_0_l538_538953

noncomputable def f (x : ℝ) : ℝ := sorry

axiom h1 : ∀ x : ℝ, f (-x) = - f x
axiom h2 : f 2 = 0
axiom h3 : ∀ x : ℝ, 0 < x → x * (deriv f x) + f x < 0

theorem solution_set_of_xf_x_gt_0 :
  {x : ℝ | x * f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by {
  sorry
}

end solution_set_of_xf_x_gt_0_l538_538953


namespace bananas_to_apples_l538_538990

-- Definitions based on conditions
def bananas := ℕ
def oranges := ℕ
def apples := ℕ

-- Condition 1: 3/4 of 16 bananas are worth 12 oranges
def condition1 : Prop := 3 / 4 * 16 = 12

-- Condition 2: price of one banana equals the price of two apples
def price_equiv_banana_apple : Prop := 1 = 2

-- Proof: 1/3 of 9 bananas are worth 6 apples
theorem bananas_to_apples 
  (c1: condition1)
  (c2: price_equiv_banana_apple) : 1 / 3 * 9 * 2 = 6 :=
by sorry

end bananas_to_apples_l538_538990


namespace karl_drive_distance_l538_538941

theorem karl_drive_distance :
  (let miles_per_gallon := 40 in
  let tank_capacity := 16 in
  let initial_drive := 400 in
  let gas_bought := 10 in
  let half_tank := tank_capacity / 2 in
  let initial_gas_used := initial_drive / miles_per_gallon in
  let remaining_gas_after_initial_drive := tank_capacity - initial_gas_used in
  let gas_after_refuel := remaining_gas_after_initial_drive + gas_bought in
  let gas_left_upon_arrival := half_tank in
  let gas_used_second_leg := gas_after_refuel - gas_left_upon_arrival in
  let second_leg_distance := gas_used_second_leg * miles_per_gallon in
  let total_distance := initial_drive + second_leg_distance in
  total_distance = 720) :=
by {
  sorry
}

end karl_drive_distance_l538_538941


namespace seunghwa_express_bus_distance_per_min_l538_538178

noncomputable def distance_per_min_on_express_bus (total_distance : ℝ) (total_time : ℝ) (time_on_general : ℝ) (gasoline_general : ℝ) (distance_per_gallon : ℝ) (gasoline_used : ℝ) : ℝ :=
  let distance_general := (gasoline_used * distance_per_gallon) / gasoline_general
  let distance_express := total_distance - distance_general
  let time_express := total_time - time_on_general
  (distance_express / time_express)

theorem seunghwa_express_bus_distance_per_min :
  distance_per_min_on_express_bus 120 110 (70) 6 (40.8) 14 = 0.62 :=
by
  sorry

end seunghwa_express_bus_distance_per_min_l538_538178


namespace extreme_points_of_cos_2x_l538_538526

noncomputable def y : ℝ → ℝ := λ x, Real.cos (2 * x)

theorem extreme_points_of_cos_2x (m : ℝ) (h : m = Real.pi) :
  ∃ n ∈ Ioo (-(Real.pi / 4)) m, ∃ k ∈ Ioo (-(Real.pi / 4)) m, n ≠ k ∧
  (∀ x ∈ Icc (-(Real.pi / 4)) m, x = n ∨ x = k ∧ derivative (λ x : ℝ, Real.cos (2 * x)) x = 0) :=
by
  sorry

end extreme_points_of_cos_2x_l538_538526


namespace algebraic_expression_value_l538_538074

theorem algebraic_expression_value (a : ℝ) (h : a^2 + 2023 * a - 1 = 0) : 
  a * (a + 1) * (a - 1) + 2023 * a^2 + 1 = 1 :=
by
  sorry

end algebraic_expression_value_l538_538074


namespace DC_correct_l538_538547

noncomputable def DC_length : ℝ :=
  let AB := 30
  let A_angle_sin := (4 / 5 : ℝ)
  let C_angle_sin := (1 / 2 : ℝ)
  let BD := A_angle_sin * AB
  let BC := BD / C_angle_sin
  let CD := Real.sqrt (BC^2 - BD^2)
  CD

theorem DC_correct :
  let AB := 30
  let θA := 90 -- not used but should indicate the right triangle
  let sinA := (4 / 5 : ℝ)
  let sinC := (1 / 2 : ℝ)
  let BD := sinA * AB
  let BC := BD / sinC
  let CD := Real.sqrt(BC^2 - BD^2)
  CD = 24 * Real.sqrt(3) :=
by 
  sorry

end DC_correct_l538_538547


namespace eccentricity_of_conic_l538_538092

theorem eccentricity_of_conic (m : ℝ) (h : m = real.sqrt (2 * 8)) (h1 : m = 4 ∨ m = -4) :
  e = real.sqrt 3 / 2 ∨ e = real.sqrt 5 :=
  sorry

end eccentricity_of_conic_l538_538092


namespace mod_eq_six_l538_538001

theorem mod_eq_six
  (x y : ℤ) 
  (h7 : 7 * x ≡ 1 [MOD 72])
  (h13 : 13 * y ≡ 1 [MOD 72]) :
  3 * x + 9 * y ≡ 6 [MOD 72] :=
sorry

end mod_eq_six_l538_538001


namespace k_value_l538_538073

def a : ℝ × ℝ × ℝ := (1, 5, -1)
def b : ℝ × ℝ × ℝ := (-2, 3, 5)

def k := (106 / 3 : ℝ)

def vec_add (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2, v1.3 + v2.3)
def vec_scale (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (c * v.1, c * v.2, c * v.3)
def vec_sub (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)
def vec_dot (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem k_value (k : ℝ) (h : vec_dot (vec_add (vec_scale k a) b) (vec_sub a (vec_scale 3 b)) = 0) :
  k = 106 / 3 :=
sorry

end k_value_l538_538073


namespace area_of_f_is_75_l538_538578

noncomputable def f : ℝ → ℝ
| x := if x < 5 then x else 3 * x - 10

def area_under_curve (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  let integral := ∫ x in a..b, f x in integral

theorem area_of_f_is_75 : area_under_curve f 0 10 = 75 := sorry

end area_of_f_is_75_l538_538578


namespace increase_a1_intervals_of_increase_l538_538062

noncomputable def f (x a : ℝ) : ℝ := x - (a + 1) * Real.log x - a / x

-- Prove that when a = 1, f(x) has no extreme points (i.e., it is monotonically increasing in (0, +∞))
theorem increase_a1 : ∀ x : ℝ, 0 < x → f x 1 = x - 2 * Real.log x - 1 / x :=
sorry

-- Find the intervals of increase for f(x) = x - (a+1) ln x - a/x
theorem intervals_of_increase (a : ℝ) : 
  (a ≤ 0 → ∀ x : ℝ, 1 < x → 0 ≤ (f x a - f 1 a)) ∧ 
  (0 < a ∧ a < 1 → (∀ x : ℝ, 0 < x ∧ x < a → 0 ≤ f x a) ∧ ∀ x : ℝ, 1 < x → 0 ≤ f x a ) ∧ 
  (a = 1 → ∀ x : ℝ, 0 < x → 0 ≤ f x a) ∧ 
  (a > 1 → (∀ x : ℝ, 0 < x ∧ x < 1 → 0 ≤ f x a) ∧ ∀ x : ℝ, a < x → 0 ≤ f x a ) :=
sorry

end increase_a1_intervals_of_increase_l538_538062


namespace calculation_1500_increased_by_45_percent_l538_538719

theorem calculation_1500_increased_by_45_percent :
  1500 * (1 + 45 / 100) = 2175 := 
by
  sorry

end calculation_1500_increased_by_45_percent_l538_538719


namespace number_of_arrangements_l538_538688

def A_left_of_B (s : list ℕ) := s.index_of 1 < s.index_of 2
def B_left_of_C (s : list ℕ) := s.index_of 2 < s.index_of 3
def A_adjacent_B (s : list ℕ) := abs (s.index_of 1 - s.index_of 2) = 1
def B_not_adjacent_C (s : list ℕ) := abs (s.index_of 2 - s.index_of 3) ≠ 1

theorem number_of_arrangements : 
  ∃ s : list ℕ, s.length = 7 ∧ 
                (∀ x, x ∈ s → x ∈ (list.range 7)) ∧ -- restrict to values from 0 to 6, assuming each number stands for a unique person
                A_left_of_B s ∧ B_left_of_C s ∧ 
                A_adjacent_B s ∧ B_not_adjacent_C s ∧
                s.permutations.count = 240 := 
  sorry

end number_of_arrangements_l538_538688


namespace constant_sum_powers_l538_538442

theorem constant_sum_powers (n : ℕ) (x y z : ℝ) (h_sum : x + y + z = 0) (h_prod : x * y * z = 1) :
  (∀ x y z : ℝ, x + y + z = 0 → x * y * z = 1 → x^n + y^n + z^n = x^n + y^n + z^n ↔ (n = 1 ∨ n = 3)) :=
by
  sorry

end constant_sum_powers_l538_538442


namespace alex_kate_jenna_probability_l538_538540

noncomputable def alex_prob : ℚ := 3 / 5
noncomputable def kate_prob : ℚ := 4 / 15
noncomputable def jenna_prob : ℚ := 2 / 15

theorem alex_kate_jenna_probability :
  ∑ p in [alex_prob, kate_prob, jenna_prob], p = 1 →
  (8.choose 4) * (4 * 3 * 1) * (alex_prob^4) * (kate_prob^3) * (jenna_prob^1) = 576 / 1500 :=
by
  sorry

end alex_kate_jenna_probability_l538_538540


namespace find_tricksters_within_30_questions_l538_538314

/-- 
Given 65 inhabitants in a village where:
- Two inhabitants are tricksters and the rest are knights.
- Knights always tell the truth.
- Tricksters can either tell the truth or lie.
- One can show any inhabitant a list of some group of inhabitants (which can consist of one person)
  and ask if all of them are knights.

Prove that it is possible to find both tricksters with no more than 30 questions.
-/
theorem find_tricksters_within_30_questions :
  ∃ (ask_knights : (inhabitants : fin 65) → list (fin 65) → Prop),
  (∀ (i j : fin 65), i ≠ j → ask_knights i [j] = true → (inhabitants[j] = knight) ∨ (inhabitants[j] = trickster))
  ∧ ∀ (inhabitants : fin 65),
  (∃ S : finset (fin 65), S.card = 2 ∧ 
  (∀ i, i ∈ S → inhabitants[i] = trickster) ∧ 
  by asking no more than 30 questions,
  you can identify both tricksters.

end find_tricksters_within_30_questions_l538_538314


namespace inequality_arith_geo_mean_l538_538146

variable (a k : ℝ)
variable (h1 : 1 ≤ k)
variable (h2 : k ≤ 3)
variable (h3 : 0 < k)

theorem inequality_arith_geo_mean (h1 : 1 ≤ k) (h2 : k ≤ 3) (h3 : 0 < k):
    ( (a + k * a) / 2 ) ^ 2 ≥ ( (a * (k * a)) ^ (1/2) ) ^ 2 :=
by
  sorry

end inequality_arith_geo_mean_l538_538146


namespace tan_theta_l538_538824

theorem tan_theta (θ : ℝ) (h₀ : sin θ + cos θ = 7 / 13) (h₁ : 0 < θ ∧ θ < π) : tan θ = -(12 / 5) :=
  sorry

end tan_theta_l538_538824


namespace simplified_expression_form_l538_538986

noncomputable def simplify_expression (x : ℚ) : ℚ := 
  3 * x - 7 * x^2 + 5 - (6 - 5 * x + 7 * x^2)

theorem simplified_expression_form (x : ℚ) : 
  simplify_expression x = -14 * x^2 + 8 * x - 1 :=
by
  sorry

end simplified_expression_form_l538_538986


namespace sin_theta_zero_suff_but_not_necess_l538_538825

theorem sin_theta_zero_suff_but_not_necess (θ : ℝ) : 
  (sin θ = 0 → sin (2 * θ) = 0) ∧ (∃ θ, cos θ = 0 ∧ sin θ ≠ 0 ∧ sin (2 * θ) = 0) := 
sorry

end sin_theta_zero_suff_but_not_necess_l538_538825


namespace Annika_hike_time_l538_538376

-- Define the conditions
def hike_rate : ℝ := 12 -- in minutes per kilometer
def initial_distance_east : ℝ := 2.75 -- in kilometers
def total_distance_east : ℝ := 3.041666666666667 -- in kilometers
def total_time_needed : ℝ := 40 -- in minutes

-- The theorem to prove
theorem Annika_hike_time : 
  (initial_distance_east + (total_distance_east - initial_distance_east)) * hike_rate + total_distance_east * hike_rate = total_time_needed := 
by
  sorry

end Annika_hike_time_l538_538376


namespace amy_race_time_l538_538708

theorem amy_race_time (patrick_time : ℕ) (manu_time : ℕ) (amy_time : ℕ)
  (h1 : patrick_time = 60)
  (h2 : manu_time = patrick_time + 12)
  (h3 : amy_time = manu_time / 2) : 
  amy_time = 36 := 
sorry

end amy_race_time_l538_538708


namespace triangle_pentagon_side_ratio_l538_538373

theorem triangle_pentagon_side_ratio :
  let triangle_perimeter := 60
  let pentagon_perimeter := 60
  let triangle_side := triangle_perimeter / 3
  let pentagon_side := pentagon_perimeter / 5
  (triangle_side : ℕ) / (pentagon_side : ℕ) = 5 / 3 :=
by
  sorry

end triangle_pentagon_side_ratio_l538_538373


namespace total_people_wearing_hats_l538_538571

theorem total_people_wearing_hats (total_adults : ℕ) (percent_women percent_women_hats percent_men percent_men_hats : ℚ)
  (H1 : total_adults = 2000)
  (H2 : percent_women = 0.6) (H3 : percent_men = 0.4)
  (H4 : percent_women_hats = 0.15) (H5 : percent_men_hats = 0.12) :
  let number_women := (percent_women * total_adults : ℚ)
  let number_men := (percent_men * total_adults : ℚ)
  let number_women_hats := (percent_women_hats * number_women : ℚ)
  let number_men_hats := (percent_men_hats * number_men : ℚ)
  number_women + number_men = 2000 ∧ number_women_hats + number_men_hats = 276 :=
by
  have number_women := percent_women * total_adults
  have number_men := percent_men * total_adults
  have number_women_hats := percent_women_hats * number_women
  have number_men_hats := percent_men_hats * number_men
  split
  · -- Prove number_women + number_men = 2000
    have h1 : number_women + number_men = total_adults := sorry
    exact h1
  · -- Show number_women_hats + number_men_hats = 276
    let hats_sum := number_women_hats + number_men_hats
    have number_women_hats_eq := (percent_women_hats * number_women : ℚ)
    have number_men_hats_eq := (percent_men_hats * number_men : ℚ)
    have hats_sum_eq : hats_sum = 276 := sorry
    exact hats_sum_eq

end total_people_wearing_hats_l538_538571


namespace find_tricksters_l538_538365

def inhab_group : Type := { n : ℕ // n < 65 }
def is_knight (i : inhab_group) : Prop := ∀ g : inhab_group, true -- Placeholder for the actual property

theorem find_tricksters (inhabitants : inhab_group → Prop)
  (is_knight : inhab_group → Prop)
  (knight_always_tells_truth : ∀ i : inhab_group, is_knight i → inhabitants i = true)
  (tricksters_2_and_rest_knights : ∃ t1 t2 : inhab_group, t1 ≠ t2 ∧ ¬is_knight t1 ∧ ¬is_knight t2 ∧
    (∀ i : inhab_group, i ≠ t1 → i ≠ t2 → is_knight i)) :
  ∃ find_them : inhab_group → inhab_group → Prop, (∀ q_count : ℕ, q_count ≤ 16) → 
  ∃ t1 t2 : inhab_group, find_them t1 t2 :=
by 
  -- The proof goes here
  sorry

end find_tricksters_l538_538365


namespace triangle_inequality_difference_l538_538926

theorem triangle_inequality_difference :
  ∀ (x : ℕ), (x + 8 > 10) → (x + 10 > 8) → (8 + 10 > x) →
    (17 - 3 = 14) :=
by
  intros x hx1 hx2 hx3
  sorry

end triangle_inequality_difference_l538_538926


namespace increasing_function_positive_slope_l538_538100

/-- 
If the derivative f'(x) of the function f(x) is always greater than 0 on ℝ,
then for any x₁, x₂ (x₁ ≠ x₂) on ℝ, the sign of (f(x₁)-f(x₂))/(x₁-x₂) is positive.
-/
theorem increasing_function_positive_slope
  (f : ℝ → ℝ) 
  (h_derivative : ∀ x : ℝ, (f' x) > 0) :
  ∀ x1 x2 : ℝ, x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0 := by
  sorry

end increasing_function_positive_slope_l538_538100


namespace roots_of_polynomial_l538_538183

def P (x : ℝ) : ℝ := x^4 - 6 * x^3 + 7 * x^2 + 6 * x - 2

theorem roots_of_polynomial :
  (∃ x1 x2 : ℝ, P x1 = 0 ∧ P x2 = 0 ∧ x1 - x2 = 1) →
  ∃ x1 x2 x3 x4, 
    P x1 = 0 ∧ P x2 = 0 ∧ P x3 = 0 ∧ P x4 = 0 ∧
    (x1 = 1 + Real.sqrt 3 ∧ x2 = 2 + Real.sqrt 3 ∧
     x3 = 1 - Real.sqrt 3 ∧ x4 = 2 - Real.sqrt 3) :=
begin
  sorry
end

end roots_of_polynomial_l538_538183


namespace find_x_for_binary_operation_l538_538521

def binary_operation (n : ℤ) (x : ℤ) : ℤ := n - (n * x)

theorem find_x_for_binary_operation :
  ∃ x : ℤ, binary_operation 5 x < 21 ∧ x = -3 :=
by
  use -3
  simp [binary_operation]
  norm_num
  sorry

end find_x_for_binary_operation_l538_538521


namespace false_statements_count_l538_538093

-- Definitions of the propositions
variables (p q : Prop)
axiom p_true : p
axiom q_false : ¬q

def prop1 := p ∧ q
def prop2 := p ∨ q
def prop3 := ¬p
def prop4 := ¬q

theorem false_statements_count : 
  (¬prop1 → true) ∧ (¬prop2 → false) ∧ (prop3 → true) ∧ (¬prop4 → false) → 2 = 2 :=
by
  sorry

end false_statements_count_l538_538093


namespace amy_race_time_l538_538707

theorem amy_race_time (patrick_time : ℕ) (manu_time : ℕ) (amy_time : ℕ)
  (h1 : patrick_time = 60)
  (h2 : manu_time = patrick_time + 12)
  (h3 : amy_time = manu_time / 2) : 
  amy_time = 36 := 
sorry

end amy_race_time_l538_538707


namespace units_digit_of_n_l538_538436

-- Definitions
def units_digit (x : ℕ) : ℕ := x % 10

-- Conditions
variables (m n : ℕ)
axiom condition1 : m * n = 23^5
axiom condition2 : units_digit m = 4

-- Theorem statement
theorem units_digit_of_n : units_digit n = 8 :=
sorry

end units_digit_of_n_l538_538436


namespace find_b_l538_538530

-- Definitions of given constants
def a : ℝ := 2
def B : ℝ := (60 : ℝ) * Real.pi / 180  -- Converting degrees to radians
def c : ℝ := 3

-- Using cosine rule to determine b
def b : ℝ := Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B)

-- The target statement to prove
theorem find_b : b = Real.sqrt 7 := by sorry

end find_b_l538_538530


namespace crew_of_four_exists_l538_538911

structure Astronaut :=
  (id : Nat)

structure Friendship (A : Type) :=
  (friends : A → A → Prop)
  (symmetric : ∀ x y, friends x y ↔ friends y x)
  (at_least_14_friends : ∀ x, ∃ S : Finset A, S.card ≥ 14 ∧ ∀ y ∈ S, friends x y)

theorem crew_of_four_exists (astronauts : Finset Astronaut) (F : Friendship Astronaut) :
  astronauts.card = 20 →
  (∀ a ∈ astronauts, ∃ friends : Finset Astronaut, friends.card ≥ 14 ∧ ∀ f ∈ friends, F.friends a f) →
  ∃ crew : Finset Astronaut, crew.card = 4 ∧ ∀ a b ∈ crew, F.friends a b :=
by
  intros h_card h_friends
  sorry

end crew_of_four_exists_l538_538911


namespace difference_of_set_has_10_integers_l538_538081

theorem difference_of_set_has_10_integers :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  in (∃ (f : ℤ → ℤ → ℤ), (∀ x y, (x ∈ S ∧ y ∈ S) → x - y ∈ [0..9]) ∧ 
                        (∃ (differences : Finset ℤ), differences = S.bUnion (λ x, S.image (λ y, x - y)) ∧
                        differences.card = 10)) :=
by
  sorry

end difference_of_set_has_10_integers_l538_538081


namespace total_weight_correct_l538_538684

-- Define the weights given in the problem
def dog_weight_kg := 2 -- weight in kilograms
def dog_weight_g := 600 -- additional grams
def cat_weight_g := 3700 -- weight in grams

-- Convert dog's weight to grams
def dog_weight_total_g : ℕ := dog_weight_kg * 1000 + dog_weight_g

-- Define the total weight of the animals (dog + cat)
def total_weight_animals_g : ℕ := dog_weight_total_g + cat_weight_g

-- Theorem stating that the total weight of the animals is 6300 grams
theorem total_weight_correct : total_weight_animals_g = 6300 := by
  sorry

end total_weight_correct_l538_538684


namespace gen_formula_is_arith_seq_l538_538075

-- Given: The sum of the first n terms of the sequence {a_n} is S_n = n^2 + 2n
def sum_seq (S : ℕ → ℕ) := ∀ n : ℕ, S n = n^2 + 2 * n

-- The general formula for {a_n} is a_n = 2n + 1
theorem gen_formula (S : ℕ → ℕ) (h : sum_seq S) : ∀ n : ℕ,  n > 0 → (∃ a : ℕ → ℕ, a n = 2 * n + 1 ∧ ∀ m : ℕ, m < n → a m = S (m + 1) - S m) :=
by sorry

-- The sequence {a_n} defined by a_n = 2n + 1 is an arithmetic sequence
theorem is_arith_seq : ∀ n : ℕ, n > 0 → (∀ a : ℕ → ℕ, (∀ k, k > 0 → a k = 2 * k + 1) → ∃ d : ℕ, d = 2 ∧ ∀ j > 0, a j - a (j - 1) = d) :=
by sorry

end gen_formula_is_arith_seq_l538_538075


namespace mode_department_A_is_32_median_department_B_is_26_department_A_younger_on_average_l538_538726

def ages_dept_A : list ℕ := [32, 30, 28, 32, 33, 32, 34, 31, 29, 32]
def ages_dept_B : list ℕ := [24, 26, 28, 22, 30, 26, 25, 27, 28, 26]

theorem mode_department_A_is_32 :
  mode ages_dept_A = 32 :=
sorry

theorem median_department_B_is_26 :
  median ages_dept_B = 26 :=
sorry

theorem department_A_younger_on_average :
  average ages_dept_A < average ages_dept_B :=
sorry

end mode_department_A_is_32_median_department_B_is_26_department_A_younger_on_average_l538_538726


namespace Sn_divisible_by_7_l538_538627

open Real

-- Definitions for x1, x2, x3
def x1 := (2 * sin (π / 7))^2
def x2 := (2 * sin (2 * π / 7))^2
def x3 := (2 * sin (3 * π / 7))^2

-- Defining Sn
noncomputable def S (n : ℕ) : ℝ := x1^n + x2^n + x3^n

-- Main theorem statement
theorem Sn_divisible_by_7 (n : ℕ) : 7 ^ (n / 3) ∣ S n :=
  sorry

end Sn_divisible_by_7_l538_538627


namespace sin_690_degree_l538_538419

theorem sin_690_degree : sin (690 : ℝ) * (Real.pi / 180) = -(1 / 2) := by
  sorry

end sin_690_degree_l538_538419


namespace axis_of_symmetry_l538_538201

-- Define the initial function y = cos(x - π/3)
def initial_function (x : ℝ) : ℝ := Real.cos (x - Real.pi / 3)

-- Define the function after the abscissa is stretched to twice the original length
def stretched_function (x : ℝ) : ℝ := Real.cos ((1 / 2) * x - Real.pi / 3)

-- Define the function after it is shifted to the left by π/6 units
def transformed_function (x : ℝ) : ℝ := Real.cos ((1 / 2) * (x + Real.pi / 6) - Real.pi / 3)

-- Simplified form of the transformed function
def simplified_function (x : ℝ) : ℝ := Real.cos ((1 / 2) * x - Real.pi / 4)

-- The axis of symmetry for the transformed function
theorem axis_of_symmetry : ∀ k : ℤ, ∃ x : ℝ, x = 2 * k * Real.pi + Real.pi / 2 :=
by
  sorry

end axis_of_symmetry_l538_538201


namespace average_age_of_instructors_l538_538193

theorem average_age_of_instructors
  (total_members : ℕ) (avg_age_members : ℕ)
  (num_boys : ℕ) (avg_age_boys : ℕ)
  (num_girls : ℕ) (avg_age_girls : ℕ)
  (num_instructors : ℕ)
  (H1 : total_members = 50) (H2 : avg_age_members = 22)
  (H3 : num_boys = 30) (H4 : avg_age_boys = 20)
  (H5 : num_girls = 15) (H6 : avg_age_girls = 24)
  (H7 : num_instructors = 5) :
  let total_age := total_members * avg_age_members in
  let total_age_boys := num_boys * avg_age_boys in
  let total_age_girls := num_girls * avg_age_girls in
  let total_age_instructors := total_age - total_age_boys - total_age_girls in
  let avg_age_instructors := total_age_instructors / num_instructors in
  avg_age_instructors = 28 :=
by {
  sorry
}

end average_age_of_instructors_l538_538193


namespace exists_graph_no_triangle_chromatic_gt_n_l538_538514

variable (n : ℕ)

theorem exists_graph_no_triangle_chromatic_gt_n (n : ℕ) : ∃ G : SimpleGraph ℕ, ¬∃ (t : Finset (G.vertex)), t.card = 3 ∧ G.Subgraph t ∧ G.IsComplete t ∧ G.IsChromatic n :=
by 
  sorry

end exists_graph_no_triangle_chromatic_gt_n_l538_538514


namespace l_shaped_area_l538_538000

theorem l_shaped_area (A B C D : Type) (side_abcd: ℝ) (side_small_1: ℝ) (side_small_2: ℝ)
  (area_abcd : side_abcd = 6)
  (area_small_1 : side_small_1 = 2)
  (area_small_2 : side_small_2 = 4)
  (no_overlap : true) :
  side_abcd * side_abcd - (side_small_1 * side_small_1 + side_small_2 * side_small_2) = 16 := by
  sorry

end l_shaped_area_l538_538000


namespace find_x_l538_538022

-- Define that x is a natural number expressed as 2^n - 32
def x (n : ℕ) : ℕ := 2^n - 32

-- We assume x has exactly three distinct prime divisors
def has_three_distinct_prime_divisors (x : ℕ) : Prop :=
  ∃ (p q r : ℕ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r ∧ prime p ∧ prime q ∧ prime r

-- One of the prime divisors is 2
def prime_divisor_2 (x : ℕ) : Prop :=
  2 ∣ x 

-- Correct answer is either 2016 or 16352
theorem find_x (n : ℕ) : has_three_distinct_prime_divisors (x n) ∧ prime_divisor_2 (x n) → 
  (x n = 2016 ∨ x n = 16352) :=
sorry

end find_x_l538_538022


namespace cafeteria_total_cost_l538_538609

-- Definitions based on conditions
def cost_per_coffee := 4
def cost_per_cake := 7
def cost_per_ice_cream := 3
def mell_coffee := 2 
def mell_cake := 1 
def friends_coffee := 2 
def friends_cake := 1 
def friends_ice_cream := 1 
def num_friends := 2
def total_coffee := mell_coffee + num_friends * friends_coffee
def total_cake := mell_cake + num_friends * friends_cake
def total_ice_cream := num_friends * friends_ice_cream

-- Total cost
def total_cost := total_coffee * cost_per_coffee + total_cake * cost_per_cake + total_ice_cream * cost_per_ice_cream

-- Theorem statement
theorem cafeteria_total_cost : total_cost = 51 := by
  sorry

end cafeteria_total_cost_l538_538609


namespace sequence_general_term_and_probability_l538_538469

theorem sequence_general_term_and_probability :
  (∀ n : ℕ, n > 0 → ∃ a : ℕ → ℤ, a 1 = 2 ∧ (∀ n > 0, a (n + 1) = -2 * a n) ∧ a n = 2 * (-2)^(n-1)) ∧
  (let first_10 := (λ n, {n ∈ (1:ℕ) .. 10 | a n ≥ 8}) in first_10.card / 10 = 2 / 5) :=
  sorry

end sequence_general_term_and_probability_l538_538469


namespace cos_S15_arithmetic_sequence_l538_538488

theorem cos_S15_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : ∀ m n, a n = a m + (n - m) * (a 2 - a 1))
  (h3 : a 7 + a 8 + a 9 = π / 6) :
  cos (S 15) = -√3/2 :=
sorry

end cos_S15_arithmetic_sequence_l538_538488


namespace tamara_diff_3kim_height_l538_538992

variables (K T X : ℕ) -- Kim's height, Tamara's height, and the difference inches respectively

-- Conditions
axiom ht_Tamara : T = 68
axiom combined_ht : T + K = 92
axiom diff_eqn : T = 3 * K - X

theorem tamara_diff_3kim_height (h₁ : T = 68) (h₂ : T + K = 92) (h₃ : T = 3 * K - X) : X = 4 :=
by
  sorry

end tamara_diff_3kim_height_l538_538992


namespace fifth_valid_student_number_is_12_l538_538211

-- The conditions as given:
def random_number_table : list (list ℕ) :=
  [[0627, 4313, 2432, 5327, 0941, 2512, 6317, 6323, 2616, 8045, 6011],
   [1410, 9577, 7424, 6762, 4281, 1457, 2042, 5332, 3732, 2707, 3607],
   [5124, 5179, 3014, 2310, 2118, 2191, 3726, 3890, 0140, 0523, 2617]]

-- Helper function to fetch the valid student numbers based on the rules
def valid_student_numbers : list ℕ :=
  let nums := random_number_table.flatten.map (λ x => (x / 100, x % 100)) in
  let valid_nums := nums.filter (λ n => n.2 ≤ 50) in
  valid_nums.map (λ n => n.2)

-- The statement to prove:
theorem fifth_valid_student_number_is_12 : valid_student_numbers.nth 4 = some 12 :=
begin
  -- We leave the proof part empty as instructed
  sorry
end

end fifth_valid_student_number_is_12_l538_538211


namespace find_symmetry_l538_538502

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * sin x - b * cos x

theorem find_symmetry {a b : ℝ} (h : a ≠ 0)
  (sym : ∀ x, f a b (x - π / 4) = f a b (3π / 4 - x)) :
  f a b (3π / 4 - x) = -f a b x ∧ (∃ x0 : ℝ, f a b x0 = 0) :=
sorry

end find_symmetry_l538_538502


namespace find_lambda_l538_538783

noncomputable def parabola_line_intersection (x1 x2 : ℝ) : Prop :=
  y_squared_eq_4x: ∀ y x : ℝ, y^2 = 4 * x →
  line_through_focus: y = x - 1 →
  x1 = 3 + 2 * Real.sqrt 2 ∧ x2 = 3 - 2 * Real.sqrt 2

theorem find_lambda (λ : ℝ) : parabola_line_intersection (3 + 2 * Real.sqrt 2) (3 - 2 * Real.sqrt 2) →
  λ = (3 + 2 * Real.sqrt 2) := by
  sorry

end find_lambda_l538_538783


namespace prob_wearing_hat_given_sunglasses_l538_538971

variables (A B : Type) [DecidableEq A] [DecidableEq B]
variable (people : Set A)
variable (wears_sunglasses : Set A)
variable (wears_hat : Set A)
variable (number_people : ℕ)
variable (number_wearing_sunglasses : ℕ)
variable (number_wearing_hat : ℕ)
variable (number_wearing_both : ℕ)
variable (prob_hat_given_sunglasses : ℚ)
variable (prob_sunglasses_given_hat : ℚ)
variable (number_hat_given_sunglasses : ℕ)

axiom condition1 : number_wearing_sunglasses = 75
axiom condition2 : number_wearing_hat = 60
axiom condition3 : prob_sunglasses_given_hat = 1/3
axiom condition4 : number_wearing_both = number_wearing_hat * prob_sunglasses_given_hat

theorem prob_wearing_hat_given_sunglasses : prob_hat_given_sunglasses = (number_wearing_both : ℚ) / (number_wearing_sunglasses : ℚ) :=
by
  rw [condition1, condition2, condition3, condition4]
  sorry -- this step assumes the computation and simplification to 4/15

end prob_wearing_hat_given_sunglasses_l538_538971


namespace inequality_lambda_mu_l538_538042

theorem inequality_lambda_mu (λ μ : ℝ) (P : ℝ × ℝ) (C : ℝ × ℝ → Prop) (O A B : ℝ × ℝ)
  (hC : C (2 * λ + 2 * μ, λ - μ))
  (hA : A = (2, 1))
  (hB : B = (2, -1))
  (hOP : (P.1, P.2) = λ • (A.1, A.2) + μ • (B.1, B.2))
  (h_eq : 4 * λ * μ = 1) :
  λ^2 + μ^2 ≥ 1 / 2 :=
sorry

end inequality_lambda_mu_l538_538042


namespace sample_size_correct_l538_538229

-- Define the total number of students in a certain grade.
def total_students : ℕ := 500

-- Define the number of students selected for statistical analysis.
def selected_students : ℕ := 30

-- State the theorem to prove the selected students represent the sample size.
theorem sample_size_correct : selected_students = 30 := by
  -- The proof would go here, but we use sorry to indicate it is skipped.
  sorry

end sample_size_correct_l538_538229


namespace tricksters_identification_l538_538338

variable (Inhabitant : Type)
variable [inhab : Fintype Inhabitant]
variable [decEqInhab : DecidableEq Inhabitant]
variable (knight : Inhabitant → Prop)
variable (trickster : Inhabitant → Prop)

variables (n : ℕ) (q : ℕ)
variable (is_truthful : Inhabitant → Prop)

constant inhabitants_count : 65
constant tricksters_count : 2

-- Define the property that a knight always tells the truth.
axiom knight_truth (x : Inhabitant) : knight x → is_truthful x

-- Define the property that a trickster can tell the truth or lie.
axiom trickster_behavior (x : Inhabitant) : trickster x → (is_truthful x ∨ ¬ is_truthful x)

 -- Define the type of the question which can be asked to an inhabitant.
inductive Question (Inhabitant : Type) : Type
| is_knight : Inhabitant → Question

-- Define the type of the answer to the question.
inductive Answer (Inhabitant : Type) : Type
| yes : Answer
| no :  Answer

-- Define a function that simulates asking a question to an inhabitant.
constant ask : Inhabitant → Question Inhabitant → Answer Inhabitant

noncomputable def find_tricksters (inhabitants : fin inhabitants_count → Inhabitant) : (fin tricksters_count → Inhabitant) :=
sorry

theorem tricksters_identification : 
  ∃ (f : (fin inhabitants_count → Inhabitant) → (fin tricksters_count → Inhabitant)), 
    ∀ inhabitants : fin inhabitants_count → Inhabitant, 
      (∀ (q_list : list (Inhabitant × Question Inhabitant)),
        q_list.length ≤ 30 → 
        let a_list := q_list.map (λ pq, ask (pq.fst) (pq.snd)) in 
        true) ∧ 
      (∃ t1 t2, trickster (f inhabitants) t1 ∧ trickster (f inhabitants) t2) :=
sorry

end tricksters_identification_l538_538338


namespace find_circle_center_l538_538026

variables {A B C : Point}
variables {ruler_with_parallel_edges : Prop}

def is_circle_center (P : Point) : Prop :=
  -- Definition that P is the center of the circle where arc passes through A, B, and C
  is_perpendicular_bisector (line_through AB) (line_through P) ∧ 
  is_perpendicular_bisector (line_through AC) (line_through P)

theorem find_circle_center 
  (arc : Arc)
  (P : Point)
  (h1 : arc_contains_points A B C arc)
  (h2 : ruler_with_parallel_edges) :
  is_circle_center P :=
sorry

end find_circle_center_l538_538026


namespace necessary_but_not_sufficient_l538_538661

variable {f : ℝ → ℝ}
variable {x₀ : ℝ}

-- Definitions based on conditions
def p : Prop := deriv f x₀ = 0
def q : Prop := ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → f x ≤ f x₀ ∨ f x ≥ f x₀

-- Statement that p is a necessary but not sufficient condition for q
theorem necessary_but_not_sufficient (h : q) : p := by
  sorry

#check necessary_but_not_sufficient

end necessary_but_not_sufficient_l538_538661


namespace kylie_coins_left_l538_538943

-- Definitions for each condition
def coins_from_piggy_bank : ℕ := 15
def coins_from_brother : ℕ := 13
def coins_from_father : ℕ := 8
def coins_given_to_friend : ℕ := 21

-- The total coins Kylie has initially
def initial_coins : ℕ := coins_from_piggy_bank + coins_from_brother
def total_coins_after_father : ℕ := initial_coins + coins_from_father
def coins_left : ℕ := total_coins_after_father - coins_given_to_friend

-- The theorem to prove the final number of coins left is 15
theorem kylie_coins_left : coins_left = 15 :=
by
  sorry -- Proof goes here

end kylie_coins_left_l538_538943


namespace expanded_polynomial_correct_l538_538439

noncomputable def polynomial_product (x : ℚ) : ℚ :=
  (2 * x^3 - 3 * x + 1) * (x^2 + 4 * x + 3)

theorem expanded_polynomial_correct (x : ℚ) : 
  polynomial_product x = 2 * x^5 + 8 * x^4 + 3 * x^3 - 11 * x^2 - 5 * x + 3 := 
by
  sorry

end expanded_polynomial_correct_l538_538439


namespace proof_eq4_proof_eq19_proof_general_eq_l538_538010

-- Definitions based on given conditions
def eq1 : Prop := sqrt (1 + 1 / 1^2 + 1 / 2^2) = 1 + 1 / 1 - 1 / (1 + 1)
def eq2 : Prop := sqrt (1 + 1 / 2^2 + 1 / 3^2) = 1 + 1 / 2 - 1 / (2 + 1)
def eq3 : Prop := sqrt (1 + 1 / 3^2 + 1 / 4^2) = 1 + 1 / 3 - 1 / (3 + 1)

def eq4 : Prop := sqrt (1 + 1 / 4^2 + 1 / 5^2) = 1 + 1 / 4 - 1 / (4 + 1)
def eq19 : Prop := sqrt (1 + 1 / 19^2 + 1 / 20^2) = 1 + 1 / 19 - 1 / (19 + 1)

-- Generalized formula
def general_eq (n: Nat) : Prop := sqrt (1 + 1 / (n ^ 2) + 1 / ((n + 1) ^ 2)) = 1 + 1 / n - 1 / (n + 1)

theorem proof_eq4 : eq4 := by sorry
theorem proof_eq19 : eq19 := by sorry
theorem proof_general_eq (n : Nat) (hn : n > 0) : general_eq n := by sorry

end proof_eq4_proof_eq19_proof_general_eq_l538_538010


namespace tricksters_identification_l538_538342

variable (Inhabitant : Type)
variable [inhab : Fintype Inhabitant]
variable [decEqInhab : DecidableEq Inhabitant]
variable (knight : Inhabitant → Prop)
variable (trickster : Inhabitant → Prop)

variables (n : ℕ) (q : ℕ)
variable (is_truthful : Inhabitant → Prop)

constant inhabitants_count : 65
constant tricksters_count : 2

-- Define the property that a knight always tells the truth.
axiom knight_truth (x : Inhabitant) : knight x → is_truthful x

-- Define the property that a trickster can tell the truth or lie.
axiom trickster_behavior (x : Inhabitant) : trickster x → (is_truthful x ∨ ¬ is_truthful x)

 -- Define the type of the question which can be asked to an inhabitant.
inductive Question (Inhabitant : Type) : Type
| is_knight : Inhabitant → Question

-- Define the type of the answer to the question.
inductive Answer (Inhabitant : Type) : Type
| yes : Answer
| no :  Answer

-- Define a function that simulates asking a question to an inhabitant.
constant ask : Inhabitant → Question Inhabitant → Answer Inhabitant

noncomputable def find_tricksters (inhabitants : fin inhabitants_count → Inhabitant) : (fin tricksters_count → Inhabitant) :=
sorry

theorem tricksters_identification : 
  ∃ (f : (fin inhabitants_count → Inhabitant) → (fin tricksters_count → Inhabitant)), 
    ∀ inhabitants : fin inhabitants_count → Inhabitant, 
      (∀ (q_list : list (Inhabitant × Question Inhabitant)),
        q_list.length ≤ 30 → 
        let a_list := q_list.map (λ pq, ask (pq.fst) (pq.snd)) in 
        true) ∧ 
      (∃ t1 t2, trickster (f inhabitants) t1 ∧ trickster (f inhabitants) t2) :=
sorry

end tricksters_identification_l538_538342


namespace count_increasing_sequences_bi_odd_l538_538430

open nat

noncomputable def count_increasing_sequences (n m : ℕ) : ℕ :=
  nat.choose (n + m - 1) m

theorem count_increasing_sequences_bi_odd :
  let b_seq_count := count_increasing_sequences 1005 15 in
  b_seq_count = 1019 :=
by
  let b_seq_count := count_increasing_sequences 1005 15
  exact (nat.choose_eq_succ _ _).symm

end count_increasing_sequences_bi_odd_l538_538430


namespace average_speed_round_trip_l538_538713

theorem average_speed_round_trip:
  (distance : ℝ) (h_distance_pos : 0 < distance) :
  let speed_pq := 80
  let speed_qp := 88
  let time_pq := distance / speed_pq
  let time_qp := distance / speed_qp
  let total_distance := 2 * distance
  let total_time := time_pq + time_qp
  let average_speed := total_distance / total_time
  average_speed = 83.80952380952381 :=
by 
  sorry

end average_speed_round_trip_l538_538713


namespace ratio_sum_proof_l538_538228

-- Define the points and their properties
variables (E F O G H C Q Z W : Type)
variables (EF EO FO OH HC EG FO_OH OC : ℝ)
variables [field E] [field F] [field O] [field G] [field H] [field C]
variables (IsCongruent : Prop)
variables (Isosceles : Prop)
variables (Perpendicular : Prop)
variables (MidpointEH : Prop)
variables (MidpointGC : Prop)

-- Specific distances between points
def distance_property : Prop :=
  EF = 20 ∧ EO = 20 ∧ FO = 20 ∧ OH = 20 ∧ HC = 20 ∧ EG = 25 ∧ OC = 25

-- Trapezoid and its area
def trapezoid_EGHC_area : ℝ := 562.5

-- Smaller trapezoids division and their areas
def smaller_trapezoid_areas : Prop :=
  let half_height := 7.5 in
  let area_EGWZ := 0.5 * half_height * (25 + 25) in
  let area_ZWCH := 0.5 * half_height * (25 + 50) in
  area_EGWZ = 187.5 ∧ area_ZWCH = 281.25

-- Ratio and sum computation
def ratio_and_sum : ℝ := 15 + 23

-- The final statement to be proven in Lean 4
theorem ratio_sum_proof
  (congr_triangles : IsCongruent)
  (isosceles_triangles : Isosceles)
  (perp_OQ : Perpendicular)
  (mid_Z_of_EH : MidpointEH)
  (mid_W_of_GC : MidpointGC)
  (dist_prop : distance_property)
  (trapezoid_area : trapezoid_EGHC_area)
  (small_trap_areas : smaller_trapezoid_areas) :
  ratio_and_sum = 38 := 
sorry

end ratio_sum_proof_l538_538228


namespace least_homeowners_l538_538257

theorem least_homeowners (M W : ℕ) (total_members : M + W = 150)
  (men_homeowners : ∃ n : ℕ, n = 10 * M / 100) 
  (women_homeowners : ∃ n : ℕ, n = 20 * W / 100) : 
  ∃ homeowners : ℕ, homeowners = 16 := 
sorry

end least_homeowners_l538_538257


namespace area_of_closed_figure_l538_538196

-- Define the polynomial expansion condition
def constant_term_in_expansion : ℤ :=
  let general_term (r : ℕ) := binom ⟨6, by norm_num⟩ ⟨r, by norm_num⟩ * (-1) ^ r * x ^ (6 - 2 * r) in
  general_term 4 + general_term 3

-- Main statement
theorem area_of_closed_figure :
  (constant_term_in_expansion = -5) →
  (∫ x in 0..5, (-x^2 + 5 * x)) = 125 / 6 :=
by sorry

end area_of_closed_figure_l538_538196


namespace solve_arithmetic_sequence_l538_538181

theorem solve_arithmetic_sequence :
  ∀ (x : ℝ), x > 0 ∧ x^2 = (2^2 + 5^2) / 2 → x = Real.sqrt (29 / 2) :=
by
  intro x
  intro hx
  sorry

end solve_arithmetic_sequence_l538_538181


namespace largest_prime_factor_of_set_l538_538250

def largest_prime_factor (n : ℕ) : ℕ :=
  -- pseudo-code for determining the largest prime factor of n
  sorry

lemma largest_prime_factor_45 : largest_prime_factor 45 = 5 := sorry
lemma largest_prime_factor_65 : largest_prime_factor 65 = 13 := sorry
lemma largest_prime_factor_85 : largest_prime_factor 85 = 17 := sorry
lemma largest_prime_factor_119 : largest_prime_factor 119 = 17 := sorry
lemma largest_prime_factor_143 : largest_prime_factor 143 = 13 := sorry

theorem largest_prime_factor_of_set :
  max (largest_prime_factor 45)
    (max (largest_prime_factor 65)
      (max (largest_prime_factor 85)
        (max (largest_prime_factor 119)
          (largest_prime_factor 143)))) = 17 :=
by
  rw [largest_prime_factor_45,
      largest_prime_factor_65,
      largest_prime_factor_85,
      largest_prime_factor_119,
      largest_prime_factor_143]
  sorry

end largest_prime_factor_of_set_l538_538250


namespace ratio_of_third_median_to_third_altitude_in_scalene_triangle_l538_538114

theorem ratio_of_third_median_to_third_altitude_in_scalene_triangle 
  {A B C A' B' C' : Point}
  (H1 : scalene_triangle A B C)
  (H2 : is_median A A' (triangle A B C))
  (H3 : is_median B B' (triangle A B C))
  (H4 : is_median C C' (triangle A B C))
  (H5 : is_altitude A A'' (triangle A B C))
  (H6 : is_altitude B B'' (triangle A B C))
  (H7 : is_altitude C C'' (triangle A B C))
  (H8 : dist (B B'') = dist (A A'))
  (H9 : dist (C C'') = dist (B B')) :
  dist (C C') / dist (C C'') = 3.5 := 
sorry

end ratio_of_third_median_to_third_altitude_in_scalene_triangle_l538_538114


namespace tangent_line_curve_l538_538669

theorem tangent_line_curve (a b k : ℝ) (A : ℝ × ℝ) (hA : A = (1, 2))
  (tangent_condition : ∀ x, (k * x + 1 = x ^ 3 + a * x + b) → (k = 3 * x ^ 2 + a)) : b - a = 5 := by
  have hA := congr_arg (λ p : ℝ × ℝ, p.2) hA
  rw hA at tangent_condition
  sorry

end tangent_line_curve_l538_538669


namespace max_min_fractions_max_min_expression_l538_538031

variable (x y : ℝ)
def circle : Prop := (x + 2)^2 + y^2 = 1

theorem max_min_fractions (h : circle x y) : 
  let k := (y - 2) / (x - 1) in 
  ((3 - Real.sqrt 3) / 4 ≤ k ∧ k ≤ (3 + Real.sqrt 3) / 4) :=
sorry

theorem max_min_expression (h : circle x y) :
  let b := x - 2 * y in 
  (-Real.sqrt 5 - 2 ≤ b ∧ b ≤ Real.sqrt 5 - 2) :=
sorry

end max_min_fractions_max_min_expression_l538_538031


namespace notebook_pre_tax_cost_eq_l538_538732

theorem notebook_pre_tax_cost_eq :
  (∃ (n c X : ℝ), n + c = 3 ∧ n = 2 + c ∧ 1.1 * X = 3.3 ∧ X = n + c → n = 2.5) :=
by
  sorry

end notebook_pre_tax_cost_eq_l538_538732


namespace min_value_expression_l538_538803

theorem min_value_expression : 
  ∃ (x y : ℝ), x^2 + 2 * x * y + 2 * y^2 + 3 * x - 5 * y = -8.5 := by
  sorry

end min_value_expression_l538_538803


namespace pie_slices_l538_538078

theorem pie_slices (total_pies : ℕ) (sold_pies : ℕ) (gifted_pies : ℕ) (left_pieces : ℕ) (eaten_fraction : ℚ) :
  total_pies = 4 →
  sold_pies = 1 →
  gifted_pies = 1 →
  eaten_fraction = 2/3 →
  left_pieces = 4 →
  (total_pies - sold_pies - gifted_pies) * (left_pieces * 3 / (1 - eaten_fraction)) / (total_pies - sold_pies - gifted_pies) = 6 :=
by
  sorry

end pie_slices_l538_538078


namespace count_specific_integers_less_than_100_l538_538084

-- Define the concept of a number having an even number of positive divisors, all of which are even
def has_even_number_of_even_divisors (n : ℕ) : Prop :=
  (∃ m : ℕ, m * m = n) ∧ ∀ d : ℕ, d ∣ n → d % 2 = 0

-- Define the numbers less than 100 that meet the condition
def integers_less_than_100_with_even_even_divisors : ℕ :=
  ∑ n in finset.Ico 1 100, if has_even_number_of_even_divisors n then 1 else 0

theorem count_specific_integers_less_than_100 : integers_less_than_100_with_even_even_divisors = 94 := 
begin
  sorry -- Proof is omitted
end

end count_specific_integers_less_than_100_l538_538084


namespace constant_term_binomial_expansion_l538_538962

noncomputable def a : ℝ := ∫ x in 0..π, sin x

theorem constant_term_binomial_expansion : (a = 2) → (binom_coeff 6 3) * 2^3 * (-1)^3 = -160 :=
by
  sorry

end constant_term_binomial_expansion_l538_538962


namespace ways_to_assign_friends_to_rooms_l538_538269

-- Definitions based on the conditions
def rooms : ℕ := 5
def friends : ℕ := 6

-- Statement to prove
theorem ways_to_assign_friends_to_rooms : 
  ∃ (ways : ℕ), ways = 7200 ∧ (∀ f : Fin friends → Fin rooms, ∑ i, if hf : f i < friends then 1 else 0 ≤ 2) := 
sorry

end ways_to_assign_friends_to_rooms_l538_538269


namespace green_beads_in_each_necklace_l538_538723

theorem green_beads_in_each_necklace (G : ℕ) :
  (∀ n, (n = 5) → (6 * n ≤ 45) ∧ (3 * n ≤ 45) ∧ (G * n = 45)) → G = 9 :=
by
  intros h
  have hn : 5 = 5 := rfl
  cases h 5 hn
  sorry

end green_beads_in_each_necklace_l538_538723


namespace xyz_relation_l538_538177

theorem xyz_relation (x y z : ℝ) (h1 : x ^ 2022 = Real.exp 1) (h2 : 2022 ^ y = 2023) (h3 : 2022 * z = 2023) : x > z ∧ z > y :=
sorry

end xyz_relation_l538_538177


namespace solve_complex_eq_l538_538098

open Complex

theorem solve_complex_eq (z : ℂ) (h : conj z - abs z = -1 - 3 * Complex.I) : z = 4 + 3 * Complex.I :=
sorry

end solve_complex_eq_l538_538098


namespace proof_triangle_problem_l538_538129

open Real -- Real numbers
open EuclideanGeometry -- Euclidean geometry functions and theorems

noncomputable def triangle_problem (BC : ℝ) (angleACB : ℝ) (radius : ℝ) (CK : ℝ) (MK : ℝ) (AB : ℝ) (areaCMN : ℝ) : Prop :=
  BC = 4 ∧
  angleACB = 2 * π / 3 ∧
  radius = 2 * √3 ∧
  CK = 2 ∧
  MK = 6 ∧
  AB = 4 * √13 ∧
  areaCMN = (72 * √3) / 13

-- Now, we define a theorem for these conditions and expected results.
theorem proof_triangle_problem : 
  triangle_problem 4 (2 * π / 3) (2 * √3) 2 6 (4 * √13) ((72 * √3) / 13) :=
by {
  -- Proof should be provided here
  sorry
}

end proof_triangle_problem_l538_538129


namespace sum_of_reciprocals_in_A_l538_538944

noncomputable def sum_reciprocals (A : Set ℕ) : ℚ :=
  ∑' x in A, (1 : ℚ) / x

def A : Set ℕ :=
  {n : ℕ | ∀ p ∣ n, p = 1 ∨ p = 2 ∨ p = 7}

theorem sum_of_reciprocals_in_A (h_rel_prime : Nat.coprime 7 3) : 
  sum_reciprocals A = 7 / 3 ∧ (7 + 3 = 10) :=
by 
  sorry

end sum_of_reciprocals_in_A_l538_538944


namespace sum_of_first_fifteen_multiples_of_eight_l538_538242

theorem sum_of_first_fifteen_multiples_of_eight :
  (∑ k in finset.range 15, 8 * (k + 1)) = 960 :=
by
  sorry

end sum_of_first_fifteen_multiples_of_eight_l538_538242


namespace find_m_plus_n_l538_538278

noncomputable def overlapping_points (A B: ℝ × ℝ) (C D: ℝ × ℝ) : Prop :=
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  let M_AB := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let axis_slope := - 1 / k_AB
  let k_CD := (D.2 - C.2) / (D.1 - C.1)
  let M_CD := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  k_CD = axis_slope ∧ (M_CD.2 - M_AB.2) = axis_slope * (M_CD.1 - M_AB.1)

theorem find_m_plus_n : 
  ∃ (m n: ℝ), overlapping_points (0, 2) (4, 0) (7, 3) (m, n) ∧ m + n = 34 / 5 :=
sorry

end find_m_plus_n_l538_538278


namespace student_sums_attempted_l538_538304

theorem student_sums_attempted (sums_right sums_wrong : ℕ) (h1 : sums_wrong = 2 * sums_right) (h2 : sums_right = 16) :
  sums_right + sums_wrong = 48 :=
by
  sorry

end student_sums_attempted_l538_538304


namespace find_tricksters_within_30_questions_l538_538321

/-- 
Given 65 inhabitants in a village where:
- Two inhabitants are tricksters and the rest are knights.
- Knights always tell the truth.
- Tricksters can either tell the truth or lie.
- One can show any inhabitant a list of some group of inhabitants (which can consist of one person)
  and ask if all of them are knights.

Prove that it is possible to find both tricksters with no more than 30 questions.
-/
theorem find_tricksters_within_30_questions :
  ∃ (ask_knights : (inhabitants : fin 65) → list (fin 65) → Prop),
  (∀ (i j : fin 65), i ≠ j → ask_knights i [j] = true → (inhabitants[j] = knight) ∨ (inhabitants[j] = trickster))
  ∧ ∀ (inhabitants : fin 65),
  (∃ S : finset (fin 65), S.card = 2 ∧ 
  (∀ i, i ∈ S → inhabitants[i] = trickster) ∧ 
  by asking no more than 30 questions,
  you can identify both tricksters.

end find_tricksters_within_30_questions_l538_538321


namespace identify_tricksters_in_30_or_less_questions_l538_538356

-- Define the problem parameters
def inhabitants : Type := Fin 65

def is_knight (inhabitant : inhabitants) : Prop := sorry
def is_trickster (inhabitant : inhabitants) : Prop := sorry

-- Define the properties
axiom knight_truthful : ∀ (x : inhabitants), is_knight x → (forall y : inhabitants, True ↔ (is_knight y = x is_knight y))
axiom trickster_mixed : ∀ (x : inhabitants), is_trickster x → ((∀ y : inhabitants, True) ∨ (∃ y : inhabitants, y ∉ (is_knight y)))

-- Problem statement
theorem identify_tricksters_in_30_or_less_questions
  (inhabitants : Type)
  (n_tricksters : ℕ := 2) -- 2 tricksters
  (total_inhabitants : ℕ := 65) -- 65 total inhabitants
  (questions_limit : ℕ := 30) -- limit of 30 questions
  (knights : inhabitants → Prop)
  (tricksters : inhabitants → Prop) :
    ∃ (solution_exists : ∀ (is_trickster : inhabitants → Prop), ∃ k : inhabitants, (knights k) ∧ (is_trickster k)) 
    (possible_to_find_tricksters : ∀ (is_knight : inhabitants → Prop) (is_trickster : inhabitants → Prop), 
    ∃ (questions_used ≤ questions_limit), ∀ (xs : set inhabitants), questions_used ≤ 30 ∧ 
    (∃ trickster1 trickster2 : inhabitants, (tricksters trickster1 ∧ tricksters trickster2 ∧ trickster1 ≠ trickster2))) :=
sorry

end identify_tricksters_in_30_or_less_questions_l538_538356


namespace max_sum_eq_4022_l538_538645

theorem max_sum_eq_4022 
  (x : Fin 2011 → ℕ) (h_pos : ∀ i, x i > 0) (h_eq : (Finset.univ.sum x) = (Finset.univ.prod x)) : 
  Finset.univ.sum x ≤ 4022 := 
sorry

end max_sum_eq_4022_l538_538645


namespace angle_A_possible_values_l538_538579

-- Let O and H denote the circumcenter and orthocenter of triangle ABC, respectively.
-- Given: AO = AH
-- Prove: ∠A = 60° or ∠A = 120°
theorem angle_A_possible_values (A B C O H : Type*)
    [triangle ABC]
    [circumcenter O ABC]
    [orthocenter H ABC]
    (h1 : dist A O = dist A H) : 
    (angle A = 60) ∨ (angle A = 120) := 
sorry

end angle_A_possible_values_l538_538579


namespace locus_equation_l538_538056

-- Defining the fixed point A
def A : ℝ × ℝ := (4, -2)

-- Defining the predicate that B lies on the circle
def on_circle (B : ℝ × ℝ) : Prop := B.1^2 + B.2^2 = 4

-- Defining the locus of the midpoint P
def locus_of_midpoint (P : ℝ × ℝ) : Prop := (P.1 - 2)^2 + (P.2 + 1)^2 = 1

-- Main theorem statement
theorem locus_equation (P : ℝ × ℝ) (B : ℝ × ℝ) :
  on_circle(B) →
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  locus_of_midpoint(P) :=
by
  sorry

end locus_equation_l538_538056


namespace triangle_angle_A_l538_538559

theorem triangle_angle_A (A B C : ℝ) (h1 : C = 3 * B) (h2 : B = 30) (h3 : A + B + C = 180) : A = 60 := by
  sorry

end triangle_angle_A_l538_538559


namespace num_solutions_l538_538513

-- Define the conditions for the complex number z
def is_solution (z : ℂ) : Prop :=
  (complex.abs z < 30) ∧ (complex.exp z = (z - 1) / (z + 1))

-- State the theorem we want to prove
theorem num_solutions : (finset.univ.filter is_solution).card = 10 :=
sorry

end num_solutions_l538_538513


namespace sin_690_deg_l538_538428

noncomputable def sin_690_eq_neg_one_half : Prop :=
  sin (690 * real.pi / 180) = -(1 / 2)

theorem sin_690_deg : sin_690_eq_neg_one_half :=
  by sorry

end sin_690_deg_l538_538428


namespace triangle_inequality_difference_l538_538925

theorem triangle_inequality_difference :
  ∀ (x : ℕ), (x + 8 > 10) → (x + 10 > 8) → (8 + 10 > x) →
    (17 - 3 = 14) :=
by
  intros x hx1 hx2 hx3
  sorry

end triangle_inequality_difference_l538_538925


namespace value_of_otimes_difference_l538_538597

def otimes (a b : ℚ) : ℚ := (a^3) / (b^2)

theorem value_of_otimes_difference :
  otimes (otimes 2 3) 4 - otimes 2 (otimes 3 4) = - 1184 / 243 := 
by
  sorry

end value_of_otimes_difference_l538_538597


namespace common_sum_7by7_square_l538_538189

theorem common_sum_7by7_square : 
  ∃ (matrix : Fin 7 → Fin 7 → ℕ), 
  (∀ (i : Fin 7), (∑ j, matrix i j = 175)) ∧
  (∀ (j : Fin 7), (∑ i, matrix i j = 175)) ∧
  (∑ i, matrix i i = 175) ∧
  (∑ i, matrix i (Fin.reverse i) = 175) :=
begin
  sorry
end

end common_sum_7by7_square_l538_538189


namespace monotonicity_intervals_range_k_sum_ineq_l538_538853

def f (x k : ℝ) := ln (x - 1) - k * (x - 1) + 1

theorem monotonicity_intervals (x k : ℝ) (h₁ : x > 1) : 
  (k ≤ 0 → ∀ x, x > 1 → deriv (λ x, f x k) x > 0) ∧ 
  (k > 0 → (∀ x, 1 < x ∧ x < 1 + 1/k → deriv (λ x, f x k) x > 0) ∧ 
           (∀ x, x > 1 + 1/k → deriv (λ x, f x k) x < 0)) := 
sorry

theorem range_k (k : ℝ) : 
  (∀ x, x > 1 → f x k ≤ 0) → k ≥ 1 := 
sorry

theorem sum_ineq (n : ℕ) (h₁ : 0 < n) : 
  (∑ i in finset.range n, ln (i + 1) / (i + 2)) < (n * (n - 1) / 4) := 
sorry

end monotonicity_intervals_range_k_sum_ineq_l538_538853


namespace exists_ten_distinct_integers_satisfying_condition_l538_538437

theorem exists_ten_distinct_integers_satisfying_condition :
  ∃ (x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 : ℤ),
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ x1 ≠ x6 ∧ x1 ≠ x7 ∧ x1 ≠ x8 ∧ x1 ≠ x9 ∧ x1 ≠ x10 ∧
  x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧ x2 ≠ x6 ∧ x2 ≠ x7 ∧ x2 ≠ x8 ∧ x2 ≠ x9 ∧ x2 ≠ x10 ∧
  x3 ≠ x4 ∧ x3 ≠ x5 ∧ x3 ≠ x6 ∧ x3 ≠ x7 ∧ x3 ≠ x8 ∧ x3 ≠ x9 ∧ x3 ≠ x10 ∧
  x4 ≠ x5 ∧ x4 ≠ x6 ∧ x4 ≠ x7 ∧ x4 ≠ x8 ∧ x4 ≠ x9 ∧ x4 ≠ x10 ∧
  x5 ≠ x6 ∧ x5 ≠ x7 ∧ x5 ≠ x8 ∧ x5 ≠ x9 ∧ x5 ≠ x10 ∧
  x6 ≠ x7 ∧ x6 ≠ x8 ∧ x6 ≠ x9 ∧ x6 ≠ x10 ∧
  x7 ≠ x8 ∧ x7 ≠ x9 ∧ x7 ≠ x10 ∧
  x8 ≠ x9 ∧ x8 ≠ x10 ∧
  x9 ≠ x10 ∧
  ∃ (n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 : ℤ),
  (let S := x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 in
  S - x1 = n1^2 ∧
  S - x2 = n2^2 ∧
  S - x3 = n3^2 ∧
  S - x4 = n4^2 ∧
  S - x5 = n5^2 ∧
  S - x6 = n6^2 ∧
  S - x7 = n7^2 ∧
  S - x8 = n8^2 ∧
  S - x9 = n9^2 ∧
  S - x10 = n10^2) :=
sorry

end exists_ten_distinct_integers_satisfying_condition_l538_538437


namespace bones_in_graveyard_l538_538112

theorem bones_in_graveyard : 
  ∃ (total_skeletons : ℕ) 
    (num_adult_women : ℕ) 
    (num_adult_men : ℕ) 
    (num_children : ℕ) 
    (bones_adult_woman : ℕ) 
    (bones_adult_man : ℕ) 
    (bones_child : ℕ),
  total_skeletons = 20 ∧
  num_adult_women = total_skeletons / 2 ∧
  num_adult_men = (total_skeletons - num_adult_women) / 2 ∧
  num_children = num_adult_men ∧
  bones_adult_woman = 20 ∧
  bones_adult_man = bones_adult_woman + 5 ∧
  bones_child = bones_adult_woman / 2 ∧
  num_adult_women * bones_adult_woman + num_adult_men * bones_adult_man + num_children * bones_child = 375 :=
begin
  use 20,
  use 10,
  use 5,
  use 5,
  use 20,
  use 25,
  use 10,
  split,
  exact rfl,
  split,
  exact (nat.div_eq_of_eq_mul (eq.refl 20)),
  split,
  exact nat.div_eq_of_eq_mul (eq.refl 10),
  split,
  exact rfl,
  split,
  exact rfl,
  split,
  refl,
  split,
  exact (nat.div_eq_of_eq_mul (eq.refl 20)),
  exact sorry

end bones_in_graveyard_l538_538112


namespace number_of_integers_satisfying_conditions_l538_538899

-- Define the problem constraints and proof statement
open Real

noncomputable def satisfies_equation (a : ℝ) : Prop :=
  ∃ x : ℕ, (x ≠ 2) ∧ ((x + a) / (x - 2) + (2 * x) / (2 - x) = 1)

noncomputable def satisfies_inequalities (a : ℝ) : Prop :=
  let solutions := { y : ℤ | (y ≥ 2 + 10 * y - 3) ∧ (y ≤ 14) } in
  (solutions.size = 4)

theorem number_of_integers_satisfying_conditions :
  (finset.filter satisfies_equation (finset.range 11)).size = 4 :=
begin
  sorry -- Proof goes here
end

end number_of_integers_satisfying_conditions_l538_538899


namespace strict_increasing_function_identity_l538_538808

section
open Nat

/-- Let f be a strictly increasing function from ℕ to ℕ such that:
   - f(2) = 2
   - ∀ n m, gcd(n, m) = 1 → f(n * m) = f(n) * f(m)
   We aim to prove that f(n) = n for all n. -/
theorem strict_increasing_function_identity
  (f : ℕ → ℕ) (h_inc : ∀ {a b}, a < b → f(a) < f(b))
  (h_prop : ∀ n m, gcd n m = 1 → f (n * m) = f n * f m)
  (h_two : f 2 = 2)
  : ∀ n : ℕ, f n = n := sorry

end

end strict_increasing_function_identity_l538_538808


namespace problem_fraction_of_complex_numbers_l538_538034

/--
Given \(i\) is the imaginary unit, prove that \(\frac {1-i}{1+i} = -i\).
-/
theorem problem_fraction_of_complex_numbers (i : ℂ) (h_i : i^2 = -1) : 
  ((1 - i) / (1 + i)) = -i := 
sorry

end problem_fraction_of_complex_numbers_l538_538034


namespace B_is_werewolf_l538_538557

def is_werewolf (x : Type) : Prop := sorry
def is_knight (x : Type) : Prop := sorry
def is_liar (x : Type) : Prop := sorry

variables (A B : Type)

-- Conditions
axiom one_is_werewolf : is_werewolf A ∨ is_werewolf B
axiom only_one_werewolf : ¬ (is_werewolf A ∧ is_werewolf B)
axiom A_statement : is_werewolf A → is_knight A
axiom B_statement : is_werewolf B → is_liar B

theorem B_is_werewolf : is_werewolf B := 
by
  sorry

end B_is_werewolf_l538_538557


namespace range_of_f_l538_538006

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arccos (x / 3))^2 + π * (Real.arcsin (x / 3)) - (Real.arcsin (x / 3))^2 - (π^2 / 12) * (x^2 - 3 * x + 9)

theorem range_of_f :
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → ∃ y ∈ Set.Icc (-3 * π^2 / 4) (π^2 / 2), f x = y :=
by
  sorry

end range_of_f_l538_538006


namespace max_elements_set_l538_538039

theorem max_elements_set {n : ℕ} (h : n ≥ 2) :
  ∃ S ⊆ {i | 1 ≤ i ∧ i ≤ n} , 
    (∀ (a b ∈ S), a ≠ b → ¬ (a ∣ b ∨ b ∣ a)) ∧ 
    (∀ (a b ∈ S), a ≠ b → ¬ (nat.coprime a b)) ∧ 
    S.card = (n + 2) / 4 :=
sorry

end max_elements_set_l538_538039


namespace mark_vs_chris_walking_time_difference_l538_538166

theorem mark_vs_chris_walking_time_difference :
  ∀ (walking_speed : ℝ) (distance_to_lunch : ℝ) (distance_to_school : ℝ),
    0 < walking_speed →
    0 < distance_to_lunch →
    0 < distance_to_school →
    distance_to_lunch = 3 →
    distance_to_school = 9 →
    walking_speed = 3 →
    (2 = ((2 * distance_to_lunch + distance_to_school) / walking_speed) - (distance_to_school / walking_speed)) :=
by
  intros walking_speed distance_to_lunch distance_to_school hs hd1 hd2 hlunch hschool hspeed
  have htmark : (2 * distance_to_lunch + distance_to_school) / walking_speed = (2 * 3 + 9) / 3, from
    by simp [hlunch, hschool, hspeed]
  have htchris : distance_to_school / walking_speed = 9 / 3, from
    by simp [hschool, hspeed]
  have result : 2 = (5 - 3), from
    by simp
  simp [htmark, htchris, result]
  exact sorry

end mark_vs_chris_walking_time_difference_l538_538166


namespace g_of_x_l538_538091

theorem g_of_x (f g : ℕ → ℕ) (h1 : ∀ x, f x = 2 * x + 3)
  (h2 : ∀ x, g (x + 2) = f x) : ∀ x, g x = 2 * x - 1 :=
by
  sorry

end g_of_x_l538_538091


namespace sum_of_circle_center_coordinates_l538_538008

theorem sum_of_circle_center_coordinates 
  (h : ∀ x y, x^2 + y^2 = -4 * x + 6 * y - 12) : 
  let center_x := -2,
      center_y := 3 in
  (center_x + center_y = 1) :=
by
  sorry

end sum_of_circle_center_coordinates_l538_538008


namespace find_min_value_l538_538370

theorem find_min_value :
  (∃ x : ℝ, (2^x + 4 / 2^x = 4)) ∧ (∃ x : ℝ, (x^2 + 4 / x^2 = 4)) ∧ 
  ¬ (∃ x : ℝ, (log x + 4 / log x = 4)) ∧ ¬ (∃ x : ℝ, (|sin x| + 4 / |sin x| = 4)) :=
by
  sorry

end find_min_value_l538_538370


namespace triangle_area_and_sum_of_fraction_numerator_denominator_l538_538560

open Real

theorem triangle_area_and_sum_of_fraction_numerator_denominator (A B C D : Point) (angleA_angleB : angle A > real.pi / 2 ∧ angle B = 20)
    (AB_eq1 : AB = 1) (CD_eq4 : CD = 4) : 
    let S := triangle_area_of_ABC A B C in
    let S2 := S^2 in
    let simplest_form_fraction := simplify_fraction S2 in
    let num := numerator simplest_form_fraction in
    let den := denominator simplest_form_fraction in
    num + den = 7 :=
sorry

end triangle_area_and_sum_of_fraction_numerator_denominator_l538_538560


namespace tan_A_l538_538543

-- Given conditions
variables (A B C : Type) [euclidean_space A B C] (triangle_ABC : right_triangle A B C)
variables (angle_BAC_right : angle triangle_ABC A B C = 90) (AB_length : length A B = 8)
variables (BC_length : length B C = 17)

-- Proof goal
theorem tan_A (triangle_ABC A B C : right_triangle) (angle_BAC_right : angle triangle_ABC A B C = 90)
    (AB_length : length A B = 8) (BC_length : length B C = 17) :
    tan (angle A B C) = 15 / 8 :=
sorry

end tan_A_l538_538543


namespace num_paths_from_top_to_bottom_l538_538432

-- Define a cube with six faces
inductive Face
| Top
| Side (n : Fin 4)  -- Four side faces labeled 1 to 4
| Bottom

open Face

-- Transition between faces is allowed only via adjacent side faces and cannot move directly from Side 1 to Side 4
def validTransition (f1 f2 : Face) : Prop :=
  match (f1, f2) with
  | (Top, Side _), (Side _, Bottom), (Side n1, Side n2) =>
    n1 ≠ n2 ∧ (n1, n2) ∈ {(0, 1), (1, 2), (2, 3), (3, 0), (1, 0), (2, 1), (3, 2), (0, 3)}
  | _ => False

-- Each face is visited at most once
def uniqueVisit (path : List Face) : Prop :=
  path.nodup

-- A path is valid if it starts at Top, ends at Bottom, and respects valid transitions and unique visits
def validPath (path : List Face) : Prop :=
  path.head = some Top ∧   -- starts at Top 
  path.ilast = some Bottom ∧  -- ends at Bottom
  (∀ p ∈ path.zip path.tail, validTransition p.1 p.2) ∧  -- valid transitions between faces
  uniqueVisit path  -- unique visit for each face
   
-- Prove there are exactly 5 valid paths from Top to Bottom
theorem num_paths_from_top_to_bottom : 
  {path | validPath path}.card = 5 :=
sorry

end num_paths_from_top_to_bottom_l538_538432


namespace find_pqr_l538_538593

-- Define the problem with the given conditions.
variables {a b c p q r : ℂ}

-- State the conditions as hypotheses.
hypothesis h1 : a ≠ 0
hypothesis h2 : b ≠ 0
hypothesis h3 : c ≠ 0
hypothesis h4 : p ≠ 0
hypothesis h5 : q ≠ 0
hypothesis h6 : r ≠ 0
hypothesis h7 : a = (b + c) / (p - 3)
hypothesis h8 : b = (a + c) / (q - 3)
hypothesis h9 : c = (a + b) / (r - 3)
hypothesis h10 : p * q + p * r + q * r = 10
hypothesis h11 : p + q + r = 6

-- State the theorem that we need to prove.
theorem find_pqr : p * q * r = 14 := 
by 
  sorry

end find_pqr_l538_538593


namespace samantha_sandwiches_l538_538983

-- Define the total number of different kinds of sandwiches
def total_sandwiches (toppings : ℕ) (slice_choices : ℕ) : ℕ :=
  (2 ^ toppings) * slice_choices

-- State the problem using definitions and proving the equality
theorem samantha_sandwiches :
  total_sandwiches 10 4 = 4096 :=
by
  unfold total_sandwiches
  have h1 : 2 ^ 10 = 1024 := by norm_num
  have h2 : 1024 * 4 = 4096 := by norm_num
  rw [h1, h2]
  sorry

end samantha_sandwiches_l538_538983


namespace buns_per_student_l538_538171

/-- Problem statement -/
theorem buns_per_student :
  ∀ (students : ℕ) (packages_per_class : ℕ) (buns_per_package : ℕ)
    (packages_first_class packages_second_class packages_third_class packages_fourth_class : ℕ),
  students = 30 →
  buns_per_package = 8 →
  packages_first_class = 20 →
  packages_second_class = 25 →
  packages_third_class = 30 →
  packages_fourth_class = 35 →
  (packages_fourth_class * buns_per_package) / students = 9 :=
by
  intros students packages_per_class buns_per_package
      packages_first_class packages_second_class packages_third_class packages_fourth_class
      h_students h_buns_per_package h_package_first h_package_second h_package_third h_package_fourth
  simp [h_students, h_buns_per_package, h_package_first, h_package_second, h_package_third, h_package_fourth]
  sorry

end buns_per_student_l538_538171


namespace number_of_valid_n_l538_538457

theorem number_of_valid_n : 
  let valid_n (n : ℕ) : Prop := ∃ p : ℕ, Nat.Prime p ∧ (p ∣ n) ∧ (∀ q : ℕ, Nat.Prime q ∧ q ∣ n → p ≤ q) ∧ (p^2 + p + 1 ∣ n)
  in Nat.card { n : ℕ | n < 2013 ∧ valid_n n } = 212 :=
by 
  let valid_n (n : ℕ) : Prop := ∃ p : ℕ, Nat.Prime p ∧ (p ∣ n) ∧ (∀ q : ℕ, Nat.Prime q ∧ q ∣ n → p ≤ q) ∧ (p^2 + p + 1 ∣ n)
  have H : Nat.card { n : ℕ | n < 2013 ∧ valid_n n } = 212 := sorry
  exact H

end number_of_valid_n_l538_538457


namespace susie_rooms_l538_538223

-- Define the conditions
def vacuum_time_per_room : ℕ := 20  -- in minutes
def total_vacuum_time : ℕ := 2 * 60  -- 2 hours in minutes

-- Define the number of rooms in Susie's house
def number_of_rooms (total_time room_time : ℕ) : ℕ := total_time / room_time

-- Prove that Susie has 6 rooms in her house
theorem susie_rooms : number_of_rooms total_vacuum_time vacuum_time_per_room = 6 :=
by
  sorry -- proof goes here

end susie_rooms_l538_538223


namespace find_x_l538_538021

-- Define that x is a natural number expressed as 2^n - 32
def x (n : ℕ) : ℕ := 2^n - 32

-- We assume x has exactly three distinct prime divisors
def has_three_distinct_prime_divisors (x : ℕ) : Prop :=
  ∃ (p q r : ℕ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r ∧ prime p ∧ prime q ∧ prime r

-- One of the prime divisors is 2
def prime_divisor_2 (x : ℕ) : Prop :=
  2 ∣ x 

-- Correct answer is either 2016 or 16352
theorem find_x (n : ℕ) : has_three_distinct_prime_divisors (x n) ∧ prime_divisor_2 (x n) → 
  (x n = 2016 ∨ x n = 16352) :=
sorry

end find_x_l538_538021


namespace explicit_formula_inequality_solution_set_l538_538870

section proof_problem

variables (a b c : ℝ) (f : ℝ → ℝ)
hypothesis ha : a = -3
hypothesis hb : b = 5
hypothesis hf : f = λ x, -3 * x ^ 2 + 5 * x + 18

theorem explicit_formula :
  (∀ x ∈ Ioo (-3:ℝ) 2, f x > 0) ∧ (∀ x ∈ (Iio (-3) ∪ Ioi 2), f x < 0) →
  ∃ y : ℝ → ℝ, y = f := sorry

theorem inequality_solution_set :
  a = -3 → b = 5 → (∀ x ∈ Ioo (-3:ℝ) 2, f x > 0) ∧ 
  (∀ x ∈ (Iio (-3) ∪ Ioi 2), f x < 0) →
  ∀ c : ℝ, (∀ x : ℝ, -3 * x ^ 2 + 5 * x + c ≤ 0) ↔ c ≤ -25 / 12 := sorry

end proof_problem

end explicit_formula_inequality_solution_set_l538_538870


namespace tetrahedron_volume_l538_538918

variable (a b d θ : ℝ)

theorem tetrahedron_volume (a b d : ℝ) (θ : ℝ) : 
  (1/6) * a * b * d * Real.sin θ = volume_of_tetrahedron AB CD θ d :=
sorry

end tetrahedron_volume_l538_538918


namespace min_value_of_exponential_l538_538890

theorem min_value_of_exponential (a b : ℝ) (h : a + b = 2) : 2^a + 2^b = 4 :=
by
-- Proof goes here
sorry

end min_value_of_exponential_l538_538890


namespace ara_height_l538_538638

variable {originalHeight sheaCurrentHeight sheaGrowth araCurrentHeight : ℝ}

-- Conditions
def sheaOriginalHeight : Prop := sheaCurrentHeight = originalHeight * 1.25
def araGrowth : Prop := sheaGrowth = sheaCurrentHeight - originalHeight ∧ araCurrentHeight = originalHeight + sheaGrowth / 3

-- Given final values
def finalValues : Prop := sheaCurrentHeight = 70 ∧ originalHeight = 56

-- The theorem we aim to prove
theorem ara_height : sheaOriginalHeight ∧ araGrowth ∧ finalValues → araCurrentHeight = 60.67 := 
by 
  sorry

end ara_height_l538_538638


namespace remainders_equality_l538_538247

open Nat

theorem remainders_equality (P P' D R R' r r': ℕ) 
  (hP : P > P')
  (hP_R : P % D = R)
  (hP'_R' : P' % D = R')
  (hPP' : (P * P') % D = r)
  (hRR' : (R * R') % D = r') : r = r' := 
sorry

end remainders_equality_l538_538247


namespace sin_cos_interval_l538_538140

theorem sin_cos_interval (a b c d : ℝ)
    (h1 : a ∈ Icc (-π/2) (π/2))
    (h2 : b ∈ Icc (-π/2) (π/2))
    (h3 : c ∈ Icc (-π/2) (π/2))
    (h4 : d ∈ Icc (-π/2) (π/2))
    (h5 : sin a + sin b + sin c + sin d = 1)
    (h6 : cos (2 * a) + cos (2 * b) + cos (2 * c) + cos (2 * d) ≥ 10 / 3) :
  a ∈ Icc 0 (π / 6) ∧ b ∈ Icc 0 (π / 6) ∧ c ∈ Icc 0 (π / 6) ∧ d ∈ Icc 0 (π / 6) :=
sorry

end sin_cos_interval_l538_538140


namespace num_solutions_to_congruence_l538_538839

def is_solution (x : ℕ) : Prop :=
  x < 150 ∧ ((x + 17) % 46 = 75 % 46)

theorem num_solutions_to_congruence : (finset.univ.filter is_solution).card = 3 :=
by
  sorry

end num_solutions_to_congruence_l538_538839


namespace sum_of_cubes_l538_538950

noncomputable def roots_are (a b c : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (Polynomial.eval a (Polynomial.C 1 * X ^ 3 - Polynomial.C 5 * X ^ 2 + Polynomial.C 13 * X - Polynomial.C 7) = 0) ∧
  (Polynomial.eval b (Polynomial.C 1 * X ^ 3 - Polynomial.C 5 * X ^ 2 + Polynomial.C 13 * X - Polynomial.C 7) = 0) ∧
  (Polynomial.eval c (Polynomial.C 1 * X ^ 3 - Polynomial.C 5 * X ^ 2 + Polynomial.C 13 * X - Polynomial.C 7) = 0)

theorem sum_of_cubes (a b c : ℝ) (h : roots_are a b c) : 
  (a + b + 2)^3 + (b + c + 2)^3 + (c + a + 2)^3 = 490 := by
  sorry

end sum_of_cubes_l538_538950


namespace probability_black_given_not_white_l538_538721

theorem probability_black_given_not_white
  (total_balls : ℕ)
  (white_balls : ℕ)
  (yellow_balls : ℕ)
  (black_balls : ℕ)
  (H1 : total_balls = 25)
  (H2 : white_balls = 10)
  (H3 : yellow_balls = 5)
  (H4 : black_balls = 10)
  (H5 : total_balls = white_balls + yellow_balls + black_balls)
  (H6 : ¬white_balls = total_balls) :
  (10 / (25 - 10) : ℚ) = 2 / 3 :=
by
  sorry

end probability_black_given_not_white_l538_538721


namespace optimal_savings_l538_538302

open Real

def initial_order := 15000
def discount_set_1 := [0.25, 0.15, 0.10]
def discount_set_2 := [0.30, 0.10, 0.05]

def apply_discounts (initial : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ acc d, acc * (1 - d)) initial

def final_price_1 := apply_discounts initial_order discount_set_1
def final_price_2 := apply_discounts initial_order discount_set_2

def final_savings := final_price_2 - final_price_1

theorem optimal_savings :
  final_savings = 371.25 := by
  sorry

end optimal_savings_l538_538302


namespace not_multiplicative_l538_538834

def r_s (s n : ℕ) : ℕ :=
  (Finset.univ.filter (λ v : Fin s, (v.val)^2).sum = n).card

def f_s (s n : ℕ) : ℚ := r_s s n / (2 * s)

theorem not_multiplicative (s : ℕ) (h : s ≠ 1 ∧ s ≠ 2 ∧ s ≠ 4 ∧ s ≠ 8) :
  ¬ ∀ m n : ℕ, m.coprime n → f_s s (m * n) = f_s s m * f_s s n :=
begin
  sorry
end

end not_multiplicative_l538_538834


namespace males_band_not_orchestra_l538_538190

/-- Define conditions as constants -/
def total_females_band := 150
def total_males_band := 130
def total_females_orchestra := 140
def total_males_orchestra := 160
def females_both := 90
def males_both := 80
def total_students_either := 310

/-- The number of males in the band who are NOT in the orchestra -/
theorem males_band_not_orchestra : total_males_band - males_both = 50 := by
  sorry

end males_band_not_orchestra_l538_538190


namespace maximum_set_size_l538_538985

/-- A set S satisfies the given conditions. -/
def satisfies_conditions (S : Finset ℕ) : Prop :=
  (∀ a ∈ S, a ≤ 100) ∧
  (∀ a b ∈ S, a ≠ b → ∃ c ∈ S, c ≠ a ∧ c ≠ b ∧ Nat.gcd (a + b) c = 1) ∧
  (∀ a b ∈ S, a ≠ b → ∃ c ∈ S, c ≠ a ∧ c ≠ b ∧ Nat.gcd (a + b) c > 1)

/-- The proof problem: Find the maximum size of set S satisfying the conditions. -/
theorem maximum_set_size : ∃ S : Finset ℕ, satisfies_conditions S ∧ S.card = 50 :=
sorry

end maximum_set_size_l538_538985


namespace regular_polygon_angle_not_divisible_by_five_l538_538961

theorem regular_polygon_angle_not_divisible_by_five :
  ∃ (n_values : Finset ℕ), n_values.card = 5 ∧
    ∀ n ∈ n_values, 3 ≤ n ∧ n ≤ 15 ∧
      ¬ (∃ k : ℕ, (180 * (n - 2)) / n = 5 * k) := 
by
  sorry

end regular_polygon_angle_not_divisible_by_five_l538_538961


namespace set_D_forms_triangle_l538_538758

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem set_D_forms_triangle : triangle_inequality 3 4 6 :=
by {
  unfold triangle_inequality,
  split,
  { norm_num, },
  { split,
    { norm_num, },
    { norm_num, }
  }
}

end set_D_forms_triangle_l538_538758


namespace triangle_area_sum_l538_538846

variables {p : ℝ} (hp : p > 0)
variables {A B C : ℝ × ℝ} -- Vertices of triangle ABC
variables {F : ℝ × ℝ} -- Focus of the parabola
variables {m n : ℝ} (hm : m ≠ 0)

-- Given conditions
def parabola_eq (x y : ℝ) : Prop := y^2 = 2 * p * x
def focus_eq (pt : ℝ × ℝ) : Prop := pt = (1, 0)

-- Positions of points on the parabola
def on_parabola (pt : ℝ × ℝ) : Prop := parabola_eq pt.1 pt.2

-- Focus condition
def focus_condition (F A B C : ℝ × ℝ) : Prop := 
  prod.fst F + prod.fst A + prod.fst B + prod.fst C = 3 ∧ 
  parabola_eq A.1 A.2 ∧ parabola_eq B.1 B.2 ∧ parabola_eq C.1 C.2

-- Line containing median from BC
def line_eq (m n x y : ℝ) : Prop := m * x + n * y - m = 0

-- Areas of triangles OFA, OFB, OFC
def area (O F P : ℝ × ℝ) : ℝ := 
  0.5 * ((prod.fst O) * (prod.snd F - prod.snd P) + (prod.fst F) * (prod.snd P - prod.snd O) + (prod.fst P) * (prod.snd O - prod.snd F)) 

-- Problem statement in Lean
theorem triangle_area_sum (hp : p > 0) (hm : m ≠ 0)
  (A B C F : ℝ × ℝ)
  (hF_on_parabola : focus_eq F)
  (hFA_sum_ABC : focus_condition F A B C)
  (h_line : ∀ {x y : ℝ}, line_eq m n x y ↔ (x, y) ∈ (set_of (λ P : ℝ × ℝ, line_eq m n P.1 P.2)))
  (h_median_on_line : line_eq m n (A.1 / 2) (A.2 / 2) = true)
  : (area (0, 0) F A)^2 + (area (0, 0) F B)^2 + (area (0, 0) F C)^2 = 3 := 
sorry

end triangle_area_sum_l538_538846


namespace extremum_at_1_implies_a_1_intervals_of_monotonicity_l538_538851

noncomputable section

-- First problem
variable (a : ℝ) (f : ℝ → ℝ)
hypothesis h_def_f : ∀ x, f x = (1/2) * x^2 - a * log x
hypothesis h_extremum_at_1 : ∀ f', ((∀ x, f' x = deriv f x) ∧ f' 1 = 0)

theorem extremum_at_1_implies_a_1 
  (f' : ℝ → ℝ) 
  (h_deriv : ∀ x, f' x = deriv f x) 
  (h_extremum_at_1 : f' 1 = 0) : 
  a = 1 := 
by
  sorry

-- Second problem
variable (f2 : ℝ → ℝ)
hypothesis h_def_f2 : ∀ x, f2 x = (1/2) * x^2 - 2 * log x

theorem intervals_of_monotonicity 
  (f2' : ℝ → ℝ) 
  (h_deriv_f2 : ∀ x, f2' x = deriv f2 x) :
  (∀ x, (0 < x ∧ x < real.sqrt 2) → (f2' x < 0)) ∧ 
  (∀ x, (real.sqrt 2 < x) → (f2' x > 0)) := 
by
  sorry

end extremum_at_1_implies_a_1_intervals_of_monotonicity_l538_538851


namespace maximize_median_money_distribution_l538_538642

def initial_money_A := 28
def initial_money_B := 72
def initial_money_C := 98
def total_money := initial_money_A + initial_money_B + initial_money_C
def median_value := 99

theorem maximize_median_money_distribution 
  (h_total: total_money = 198)
  (h_median: median_value = 99)
  : True :=
begin
  -- Proof that if the money is redistributed to achieve a maximum median of $99,
  -- the third person will hold $99 given the conditions.
  sorry
end

end maximize_median_money_distribution_l538_538642


namespace find_A_l538_538991

theorem find_A
  (A B C : ℝ)
  (h : (λ x : ℝ, 1 / (x^3 - 2*x^2 - 13*x + 10)) = (λ x, A / (x + 2) + B / (x - 1) + C / (x - 1)^2))
  (fac : ∀ x : ℝ, x^3 - 2*x^2 - 13*x + 10 = (x + 2) * (x - 1)^2) :
  A = 1 / 9 :=
by
  sorry

end find_A_l538_538991


namespace tan_585_eq_1_l538_538772

noncomputable def tan_deg (θ : ℝ) : ℝ := Real.tan (θ * Real.pi / 180)

theorem tan_585_eq_1 :
  tan_deg 585 = 1 := 
by
  have h1 : 585 - 360 = 225 := by norm_num
  have h2 : tan_deg 225 = tan_deg 45 :=
    by have h3 : 225 = 180 + 45 := by norm_num
       rw [h3, tan_deg]
       exact Real.tan_add_pi_div_two_simp_left (Real.pi * 45 / 180)
  rw [← tan_deg]
  rw [h1, h2]
  exact Real.tan_pi_div_four

end tan_585_eq_1_l538_538772


namespace max_s_inv_value_l538_538631

-- Definitions and conditions
def s (x y : ℝ) := x^2 + y^2

noncomputable def max_s_inv : ℝ :=
  1 / (⨆ (x y : ℝ) (h : 4*x^2 - 5*x*y + 4*y^2 = 5), s x y)

-- The main statement
theorem max_s_inv_value :
  max_s_inv = 3 / 10 :=
sorry

end max_s_inv_value_l538_538631


namespace slices_leftover_l538_538389

def total_initial_slices : ℕ := 12 * 2
def bob_slices : ℕ := 12 / 2
def tom_slices : ℕ := 12 / 3
def sally_slices : ℕ := 12 / 6
def jerry_slices : ℕ := 12 / 4
def total_slices_eaten : ℕ := bob_slices + tom_slices + sally_slices + jerry_slices

theorem slices_leftover : total_initial_slices - total_slices_eaten = 9 := by
  sorry

end slices_leftover_l538_538389


namespace susie_rooms_l538_538220

theorem susie_rooms
  (house_vacuum_time_hours : ℕ)
  (room_vacuum_time_minutes : ℕ)
  (total_vacuum_time_minutes : ℕ)
  (total_vacuum_time_computed : house_vacuum_time_hours * 60 = total_vacuum_time_minutes)
  (rooms_count : ℕ)
  (rooms_count_computed : total_vacuum_time_minutes / room_vacuum_time_minutes = rooms_count) :
  house_vacuum_time_hours = 2 →
  room_vacuum_time_minutes = 20 →
  rooms_count = 6 :=
by
  intros h1 h2
  sorry

end susie_rooms_l538_538220


namespace scale_model_dome_height_is_correct_l538_538796

-- Define the conditions
def original_dome_height : ℝ := 55
def original_dome_volume : ℝ := 250000
def scale_model_volume : ℝ := 0.2
def volume_ratio : ℝ := original_dome_volume / scale_model_volume
def scale_factor : ℝ := volume_ratio^(1 / 3)

-- State the target height for the model
def target_height : ℝ := original_dome_height / scale_factor

-- Define the main proof goal
theorem scale_model_dome_height_is_correct :
  target_height ≈ 0.5 :=
by
  sorry

end scale_model_dome_height_is_correct_l538_538796


namespace value_of_f_4_plus_f_neg4_l538_538058

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2 else Real.sqrt (-x)

theorem value_of_f_4_plus_f_neg4 :
  f 4 + f (-4) = 4 := by
  sorry

end value_of_f_4_plus_f_neg4_l538_538058


namespace Amy_finish_time_l538_538706

-- Definitions and assumptions based on conditions
def Patrick_time : ℕ := 60
def Manu_time : ℕ := Patrick_time + 12
def Amy_time : ℕ := Manu_time / 2

-- Theorem statement to be proved
theorem Amy_finish_time : Amy_time = 36 :=
by
  sorry

end Amy_finish_time_l538_538706


namespace find_d_l538_538275

-- Define the coordinates for the centers of the circles
def c1_center := (-12 : ℝ, -3 : ℝ)
def c2_center := (4 : ℝ, 10 : ℝ)

-- Define the radii of the circles
def r1 := 15 : ℝ
def r2 := Real.sqrt 85

-- Equation of the first circle
def circle1 (x y : ℝ) := (x + 12) ^ 2 + (y + 3) ^ 2 = r1 ^ 2

-- Equation of the second circle
def circle2 (x y : ℝ) := (x - 4) ^ 2 + (y - 10) ^ 2 = r2 ^ 2

-- The theorem stating the value of d
theorem find_d :
    (∃ x y : ℝ, circle1 x y ∧ circle2 x y ∧ x + y = -37 / 58) :=
sorry

end find_d_l538_538275


namespace total_number_of_digits_l538_538194

theorem total_number_of_digits (n S S₅ S₃ : ℕ) (h1 : S = 20 * n) (h2 : S₅ = 5 * 12) (h3 : S₃ = 3 * 33) : n = 8 :=
by
  sorry

end total_number_of_digits_l538_538194


namespace max_f_on_M_l538_538017

variables (p q : ℝ)

def f (x : ℝ) : ℝ := 2 * x^2 + 3 * p * x + 2 * q
def φ (x : ℝ) : ℝ := x + 4 / x

theorem max_f_on_M : 
  (∀ x ∈ set.Icc 1 (9 / 4), ∃ x₀ ∈ set.Icc 1 (9 / 4), f p q x ≥ f p q x₀ ∧ φ x ≥ φ x₀ ∧ f p q x₀ = φ x₀) →
  (∃ x ∈ set.Icc 1 (9 / 4), f p q x = 6) :=
begin
  sorry
end

end max_f_on_M_l538_538017


namespace probability_three_tails_l538_538624

noncomputable def binomial (n k : ℕ) : ℚ :=
  nat.desc_factorial n k / nat.factorial k

theorem probability_three_tails :
  let p_heads := (1 : ℚ) / 3
  let p_tails := (2 : ℚ) / 3
  let n := 8
  let k := 3
  let P : ℚ := binomial n k * (p_tails ^ k) * (p_heads ^ (n - k))
  P = 448 / 6561 := by
  sorry

end probability_three_tails_l538_538624


namespace compare_polynomials_l538_538718

theorem compare_polynomials (x : ℝ) (h : x ≥ 0) : 
  (x > 2 → 5*x^2 - 1 > 3*x^2 + 3*x + 1) ∧ 
  (x = 2 → 5*x^2 - 1 = 3*x^2 + 3*x + 1) ∧ 
  (0 ≤ x → x < 2 → 5*x^2 - 1 < 3*x^2 + 3*x + 1) :=
sorry

end compare_polynomials_l538_538718


namespace positive_difference_between_jo_and_anne_l538_538133

def jo_sum : ℕ := (100 * (100 + 1)) / 2

def round_to_nearest_5 (n : ℕ) : ℕ :=
  if n % 5 = 0 then n
  else if n % 5 < 3 then n - (n % 5)
  else n - (n % 5) + 5

def anne_sum : ℕ := (Finset.range 101).sum (λ n, round_to_nearest_5 n)

theorem positive_difference_between_jo_and_anne : abs (jo_sum - anne_sum) = 0 :=
by
  sorry

end positive_difference_between_jo_and_anne_l538_538133


namespace identify_tricksters_in_30_or_less_questions_l538_538360

-- Define the problem parameters
def inhabitants : Type := Fin 65

def is_knight (inhabitant : inhabitants) : Prop := sorry
def is_trickster (inhabitant : inhabitants) : Prop := sorry

-- Define the properties
axiom knight_truthful : ∀ (x : inhabitants), is_knight x → (forall y : inhabitants, True ↔ (is_knight y = x is_knight y))
axiom trickster_mixed : ∀ (x : inhabitants), is_trickster x → ((∀ y : inhabitants, True) ∨ (∃ y : inhabitants, y ∉ (is_knight y)))

-- Problem statement
theorem identify_tricksters_in_30_or_less_questions
  (inhabitants : Type)
  (n_tricksters : ℕ := 2) -- 2 tricksters
  (total_inhabitants : ℕ := 65) -- 65 total inhabitants
  (questions_limit : ℕ := 30) -- limit of 30 questions
  (knights : inhabitants → Prop)
  (tricksters : inhabitants → Prop) :
    ∃ (solution_exists : ∀ (is_trickster : inhabitants → Prop), ∃ k : inhabitants, (knights k) ∧ (is_trickster k)) 
    (possible_to_find_tricksters : ∀ (is_knight : inhabitants → Prop) (is_trickster : inhabitants → Prop), 
    ∃ (questions_used ≤ questions_limit), ∀ (xs : set inhabitants), questions_used ≤ 30 ∧ 
    (∃ trickster1 trickster2 : inhabitants, (tricksters trickster1 ∧ tricksters trickster2 ∧ trickster1 ≠ trickster2))) :=
sorry

end identify_tricksters_in_30_or_less_questions_l538_538360


namespace angle_E_of_convex_pentagon_l538_538755

theorem angle_E_of_convex_pentagon : 
  ∀ {ABCDE : Type} {A B C D E : ABCDE} (h1 : ∀ (x y : ABCDE), x = y)
  (h2 : ∃ (A B : ABCDE), ∠A = 90 ∧ ∠B = 90), 
  ∠E = 150 :=
by 
  sorry

end angle_E_of_convex_pentagon_l538_538755


namespace divergence_r_divergence_p_divergence_q_l538_538002

noncomputable def vector_r (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

noncomputable def vector_p (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let denom := (x + y + z)^(2 / 3)
  (1 / denom, 1 / denom, 1 / denom)

noncomputable def vector_q (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let exp_xy := Real.exp (x * y)
  (-x * exp_xy, y * exp_xy, x * y * exp_xy)

theorem divergence_r (x y z : ℝ) : Real.divergence (vector_r x y z) = 3 := sorry

theorem divergence_p (x y z : ℝ) :
  Real.divergence (vector_p x y z) = -2 * (x + y + z)^(-5 / 3) := sorry

theorem divergence_q (x y z : ℝ) : Real.divergence (vector_q x y z) = 0 := sorry

end divergence_r_divergence_p_divergence_q_l538_538002


namespace always_find_2x2_square_l538_538463

-- Define the chessboard and conditions
def chessboard := matrix (fin 8) (fin 8) bool

-- Define the nature of the $2 \times 1$ rectangles
def rectangle2x1 (cb : chessboard) := 
  ∃ (r : fin 8) (c : fin 7), cb r c = ff ∧ cb r (c + 1) = ff

-- Define the initial condition
def eightRectanglesRemoved (cb : chessboard) :=
  ∃ (r1 r2 r3 r4 r5 r6 r7 r8 : fin 8) (c1 c2 c3 c4 c5 c6 c7 c8 : fin 7), 
    rectangle2x1 cb ∧ rectangle2x1 cb ∧ rectangle2x1 cb ∧ rectangle2x1 cb ∧ 
    rectangle2x1 cb ∧ rectangle2x1 cb ∧ rectangle2x1 cb ∧ rectangle2x1 cb

-- Statement of the problem in Lean
theorem always_find_2x2_square (cb : chessboard) (removed : eightRectanglesRemoved cb) :
  ∃ (r : fin 7) (c : fin 7), 
    cb r c = ff ∧ cb r (c + 1) = ff ∧ cb (r + 1) c = ff ∧ cb (r + 1) (c + 1) = ff :=
sorry

end always_find_2x2_square_l538_538463


namespace cage_cost_correct_l538_538938

noncomputable def total_amount_paid : ℝ := 20
noncomputable def change_received : ℝ := 0.26
noncomputable def cat_toy_cost : ℝ := 8.77
noncomputable def cage_cost := total_amount_paid - change_received

theorem cage_cost_correct : cage_cost = 19.74 := by
  sorry

end cage_cost_correct_l538_538938


namespace part1_part2_l538_538159

variable (a : ℝ)

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}

def B : Set ℝ := {x | -1 - 2 * a ≤ x ∧ x ≤ a - 2}

theorem part1 (h : A ∩ B = A) : 7 ≤ a := sorry

theorem part2 (h : ∀ x, x ∈ B → x ∈ A) : a < 1/3 := sorry

end part1_part2_l538_538159


namespace log_base_change_l538_538884

theorem log_base_change:
  (log₂ : ℝ → ℝ) (y : ℝ) :
  (log₂ (Real.log 4 (Real.log 16 4) ^ Real.log 4 16)) = -2 := by
  sorry

end log_base_change_l538_538884


namespace cos_2theta_correct_l538_538465

-- Given condition
axiom tan_theta_eq_3 : ∀ θ : ℝ, tan θ = 3

-- Define what we need to prove, given the condition
theorem cos_2theta_correct (θ : ℝ) (h : tan θ = 3) : cos (2 * θ) = -4 / 5 :=
by {
  sorry
}

end cos_2theta_correct_l538_538465


namespace sin_690_eq_neg_half_l538_538408

theorem sin_690_eq_neg_half :
  Real.sin (690 * Real.pi / 180) = -1 / 2 :=
by {
  sorry
}

end sin_690_eq_neg_half_l538_538408


namespace problem_124th_rising_number_l538_538391

def is_rising_number (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ), n = 10000*a + 1000*b + 100*c + 10*d + e ∧
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ 9

def nth_rising_number (n : ℕ) : ℕ :=
  sorry

theorem problem_124th_rising_number :
  nth_rising_number 124 = 23489 ∧
  ∀ d ∈ {5, 6, 7}, ¬ (d ∈ {2, 3, 4, 8, 9}) :=
by
  sorry

end problem_124th_rising_number_l538_538391


namespace train_speed_l538_538748

def length_of_train : ℝ := 150
def time_to_cross_pole : ℝ := 9

def speed_in_m_per_s := length_of_train / time_to_cross_pole
def speed_in_km_per_hr := speed_in_m_per_s * (3600 / 1000)

theorem train_speed : speed_in_km_per_hr = 60 := by
  -- Length of train is 150 meters
  -- Time to cross pole is 9 seconds
  -- Speed in m/s = 150 meters / 9 seconds = 16.67 m/s
  -- Speed in km/hr = 16.67 m/s * 3.6 = 60 km/hr
  sorry

end train_speed_l538_538748


namespace centers_of_parallelograms_l538_538120

def is_skew_lines (l1 l2 l3 l4 : Line) : Prop :=
  -- A function that checks if 4 lines are pairwise skew and no three of them are parallel to the same plane.
  sorry

def count_centers_of_parallelograms (l1 l2 l3 l4 : Line) : ℕ :=
  -- A function that counts the number of lines through which the centers of parallelograms formed by the intersections of the lines pass.
  sorry

theorem centers_of_parallelograms (l1 l2 l3 l4 : Line) (h_skew: is_skew_lines l1 l2 l3 l4) : count_centers_of_parallelograms l1 l2 l3 l4 = 3 :=
  sorry

end centers_of_parallelograms_l538_538120


namespace find_tricksters_in_16_questions_l538_538334

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

end find_tricksters_in_16_questions_l538_538334


namespace find_a_l538_538184

theorem find_a (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
  (h1 : a^2 / b = 1) (h2 : b^2 / c = 2) (h3 : c^2 / a = 3) : 
  a = 12^(1/7 : ℝ) :=
by
  sorry

end find_a_l538_538184


namespace find_function_l538_538490

-- Conditions and definitions
def P := (-1 : ℝ, 2 : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * x^2
def tangent_line_parallel_slope : ℝ := -3

-- The equivalent proof problem
theorem find_function (a b : ℝ) (h1 : f P.1 = P.2)
  (h2 : (3 * a * P.1^2 + 2 * b * P.1) = tangent_line_parallel_slope) :
  f = λ x, x^3 + 3 * x^2 :=
by
  sorry

end find_function_l538_538490


namespace triangle_side_b_l538_538471

theorem triangle_side_b (a b : ℝ) (B C : ℝ) (h1 : a = sqrt 3) (h2 : sin B = 1 / 2) (h3 : C = π / 6) : b = 1 := 
by 
  sorry

end triangle_side_b_l538_538471


namespace tetrahedron_equal_edges_equal_perimeters_l538_538630

theorem tetrahedron_equal_edges_equal_perimeters
  (A B C D : Point)
  (T : Tetrahedron)
  (hAB : T.edge_length A B = T.edge_length C D) :
  ∀ (π : Plane), (is_parallel_to_edges π A B C D T) → 
    (perimeter_of_intersection π T = 2 * T.edge_length A B) :=
by
  sorry

end tetrahedron_equal_edges_equal_perimeters_l538_538630


namespace fraction_base_4_conversion_l538_538583

theorem fraction_base_4_conversion : 
  let m := 16 ^ 1500 
  in m / 8 = 4 ^ 2998.5 :=
by 
  let m := 16 ^ 1500
  show m / 8 = 4 ^ 2998.5 
  sorry

end fraction_base_4_conversion_l538_538583


namespace binary_computation_l538_538769

theorem binary_computation :
  (0b101101 * 0b10101 + 0b1010 / 0b10) = 0b110111100000 := by
  sorry

end binary_computation_l538_538769


namespace find_m_l538_538868

def vec2 := ℝ × ℝ

def orthogonal (u v : vec2) : Prop :=
u.1 * v.1 + u.2 * v.2 = 0

variables (a b c : vec2) (m : ℝ)

theorem find_m (h1 : a = (m, 2))
                (h2 : b = (1, 1))
                (h3 : c = (1, 3))
                (h4 : orthogonal (2 a - b) c) : m = -4 :=
sorry

end find_m_l538_538868


namespace integer_point_functions_count_l538_538546

def is_integer_point (p : ℝ × ℝ) : Prop :=
  p.1.floor = p.1 ∧ p.2.floor = p.2

def is_first_order_integer_point_function (f : ℝ → ℝ) : Prop :=
  ∃! (p : ℝ × ℝ), is_integer_point p ∧ p.2 = f p.1

-- Define the given functions
def f (x : ℝ) : ℝ := Real.sin (2 * x)
def g (x : ℝ) : ℝ := x^3
def h (x : ℝ) : ℝ := (1/3) ^ x
def φ (x : ℝ) : ℝ := Real.log x

-- Prove the final result
theorem integer_point_functions_count : 
  (is_first_order_integer_point_function f) + 
  (is_first_order_integer_point_function φ) + 
  (is_first_order_integer_point_function g) + 
  (is_first_order_integer_point_function h) = 2 :=
by sorry

end integer_point_functions_count_l538_538546


namespace inequality_solution_range_l538_538527

theorem inequality_solution_range (a : ℝ) :
  (∃ x ∈ Ioo (1 : ℝ) 4, x^2 - 4*x - 2 - a > 0) → a < -2 :=
by
  sorry

end inequality_solution_range_l538_538527


namespace min_expenditure_l538_538399

structure OliveJar where
  olives : ℕ
  cost : ℕ   -- cost in cents

def jar10 : OliveJar := ⟨10, 100⟩
def jar20 : OliveJar := ⟨20, 150⟩
def jar30 : OliveJar := ⟨30, 250⟩
def jar40 : OliveJar := ⟨40, 400⟩

def discountThreshold : ℕ := 3
def discountRate : ℚ := 0.1
def CliveMoney : ℕ := 1000  -- money in cents

def totalOlives (jars : List OliveJar) : ℕ :=
  jars.foldr (λ jar acc => jar.olives + acc) 0

def totalCost (jars : List OliveJar) : ℚ :=
  let grouped := jars.groupBy (·.olives)
  grouped.foldr (λ grp acc => 
    let cnt := grp.length
    if cnt >= discountThreshold then
      acc + (grp.length * grp.head!.cost * (1 - discountRate))
    else
      acc + (grp.length * grp.head!.cost)
  ) 0

theorem min_expenditure :
  ∃ jars : List OliveJar,
    totalOlives jars = 80 ∧ totalCost jars = 555 ∧ (CliveMoney - 555) = 445 := by
  sorry

end min_expenditure_l538_538399


namespace magnitude_of_z_l538_538847

theorem magnitude_of_z (z : ℂ) (h : z * (complex.I ^ 2018) = 3 + 4 * complex.I) : |z| = 5 := 
sorry

end magnitude_of_z_l538_538847


namespace number_of_valid_a_l538_538903

theorem number_of_valid_a :
  (∃ a : ℤ, 
    (∃ x : ℕ, (x + a) / (x - 2) + (2 * x) / (2 - x) = 1) ∧
    (∃ s : Finset ℤ, s.card = 4 ∧ ∀ y ∈ s, 
      (y + 1) / 5 ≥ y / 2 - 1 ∧ 
      y + a < 11 * y - 3)
  ) = 4 := sorry

end number_of_valid_a_l538_538903


namespace cylinder_volume_expansion_l538_538893

def base_area (r : ℝ) : ℝ := Real.pi * r^2
def cylinder_volume (r h : ℝ) : ℝ := base_area r * h

theorem cylinder_volume_expansion
  (r h : ℝ) : cylinder_volume (3 * r) h = 9 * (cylinder_volume r h) := by
  sorry

end cylinder_volume_expansion_l538_538893


namespace range_of_m_hidden_symmetric_points_l538_538459

def hidden_symmetric_points (f : ℝ → ℝ) (x0 : ℝ) : Prop :=
  f x0 = -f (-x0)

noncomputable def f (m : ℝ) : ℝ → ℝ :=
  λ x, if x < 0 then x^2 + 2 * x else m * x + 4

theorem range_of_m_hidden_symmetric_points :
  (∃ x0 : ℝ, hidden_symmetric_points (f m) x0) → m ≤ -2 :=
sorry

end range_of_m_hidden_symmetric_points_l538_538459


namespace s_inverse_sum_l538_538981

noncomputable def s (x y : ℝ) := x^2 + y^2

theorem s_inverse_sum
  (x y : ℝ)
  (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5)
  : let s_min := min (s x y)
        s_max := max (s x y)
    in (1 / s_min + 1 / s_max) = 8 / 5 :=
begin
  sorry
end

end s_inverse_sum_l538_538981


namespace train_cross_time_l538_538751

-- Definitions of the conditions provided
def speed_kmph : ℝ := 48        -- Speed in kilometers per hour
def length_train : ℝ := 120     -- Length of the train in meters

def km_to_m : ℝ := 1000         -- Conversion factor from kilometers to meters
def hr_to_s : ℝ := 3600         -- Conversion factor from hours to seconds

-- Derived definitions based on the conditions
def speed_mps : ℝ := (speed_kmph * km_to_m) / hr_to_s  -- Speed in meters per second
def time_to_cross : ℝ := length_train / speed_mps      -- Time to cross the pole in seconds

-- The theorem to prove the equivalence
theorem train_cross_time : time_to_cross = 9 := by
  -- Proof is omitted as specified
  sorry

end train_cross_time_l538_538751


namespace age_of_B_l538_538256

-- Define the ages of A and B
variables (A B : ℕ)

-- The conditions given in the problem
def condition1 (a b : ℕ) : Prop := a + 10 = 2 * (b - 10)
def condition2 (a b : ℕ) : Prop := a = b + 9

theorem age_of_B (A B : ℕ) (h1 : condition1 A B) (h2 : condition2 A B) : B = 39 :=
by
  sorry

end age_of_B_l538_538256


namespace find_tricksters_within_30_questions_l538_538320

/-- 
Given 65 inhabitants in a village where:
- Two inhabitants are tricksters and the rest are knights.
- Knights always tell the truth.
- Tricksters can either tell the truth or lie.
- One can show any inhabitant a list of some group of inhabitants (which can consist of one person)
  and ask if all of them are knights.

Prove that it is possible to find both tricksters with no more than 30 questions.
-/
theorem find_tricksters_within_30_questions :
  ∃ (ask_knights : (inhabitants : fin 65) → list (fin 65) → Prop),
  (∀ (i j : fin 65), i ≠ j → ask_knights i [j] = true → (inhabitants[j] = knight) ∨ (inhabitants[j] = trickster))
  ∧ ∀ (inhabitants : fin 65),
  (∃ S : finset (fin 65), S.card = 2 ∧ 
  (∀ i, i ∈ S → inhabitants[i] = trickster) ∧ 
  by asking no more than 30 questions,
  you can identify both tricksters.

end find_tricksters_within_30_questions_l538_538320


namespace min_value_problem_l538_538864

variable (x y : ℝ)

-- Define the vectors and the orthogonality condition
def a := (x-1, 2: ℝ)
def b := (4, y: ℝ)

-- Define the orthogonality condition
def orthogonal (a b : ℝ × ℝ) := a.1 * b.1 + a.2 * b.2 = 0

-- Define the minimum value problem
def min_value_expr : ℝ := 9^x + 3^y

-- Problem statement: Finding the minimum value given the orthogonality condition
theorem min_value_problem 
  (h : orthogonal a b) : min_value_expr x y = 6 :=
  sorry

end min_value_problem_l538_538864


namespace problem_solution_l538_538644

noncomputable def x : ℂ := (1 + complex.I * real.sqrt 3) / 2

theorem problem_solution : (1 / (x^2 + x)) = (-complex.I * real.sqrt 3 / 3) := by
  sorry

end problem_solution_l538_538644


namespace Petya_wins_l538_538977

-- Define the game configuration
structure GameBoard :=
  (board : Fin 7 → Fin 7 → Option (Fin 7))
  (valid : ∀ r c, ∀ n : Fin 7, board r c = some n → 
    (∀ r', board r' c ≠ some n) ∧ (∀ c', board r c' ≠ some n))

-- Define valid moves
structure Move (b : GameBoard) :=
  (r : Fin 7)
  (c : Fin 7)
  (n : Fin 7)
  (valid : b.board r c = none)

-- Function to execute a move
def makeMove (b : GameBoard) (m : Move b) : GameBoard :=
{ board := λ r' c', if r' = m.r ∧ c' = m.c then some m.n else b.board r' c',
  valid := sorry } -- proof of validity condition is omitted here

-- Define the strategy
def antiSymmetricStrategy (b : GameBoard) : Prop :=
  (∀ r c, r ≠ 3 → c ≠ 3 → (∃ n1 n2, b.board r c = some n1 ∧ b.board (6 - r) (6 - c) = some n2 ∧ n1 + n2 = 7))

-- Define the main theorem
theorem Petya_wins : ∃ f : (GameBoard → Move), ∀ b : GameBoard, antiSymmetricStrategy b → ∃ m : Move b, f b = m :=
begin
  sorry
end

end Petya_wins_l538_538977


namespace seq1_general_formula_seq2_general_formula_l538_538710

-- Sequence (1): Initial condition and recurrence relation
def seq1 (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) = a n + (2 * n - 1)

-- Proving the general formula for sequence (1)
theorem seq1_general_formula (a : ℕ → ℕ) (n : ℕ) (h : seq1 a) :
  a n = (n - 1) ^ 2 :=
sorry

-- Sequence (2): Initial condition and recurrence relation
def seq2 (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = 3 * a n

-- Proving the general formula for sequence (2)
theorem seq2_general_formula (a : ℕ → ℕ) (n : ℕ) (h : seq2 a) :
  a n = 3 ^ n :=
sorry

end seq1_general_formula_seq2_general_formula_l538_538710


namespace problem_proof_l538_538829

-- Definitions for conditions
variable (l m : Line)
variable (α β : Plane)

-- Given conditions
variable (perpendicular_l_α : l ⊥ α)
variable (contain_m_β : m ∈ β)

-- Statements formalized in Lean
def statement1 : Prop := (α ∥ β) → (l ⊥ m)
def statement2 : Prop := (α ⊥ β) → (l ∥ m)
def statement3 : Prop := (l ∥ m) → (α ⊥ β)
def statement4 : Prop := (l ⊥ m) → (α ∥ β)

-- Correct answer formalized
def correct_answer : Prop := statement1 ∧ statement3

-- Theorems to prove
theorem problem_proof : correct_answer := 
by
  sorry

end problem_proof_l538_538829


namespace distance_A_focus_l538_538683

-- Definitions from the problem conditions
def parabola_eq (x y : ℝ) : Prop := x^2 = 4 * y
def point_A (x : ℝ) : Prop := parabola_eq x 4
def focus_y_coord : ℝ := 1 -- Derived from the standard form of the parabola x^2 = 4py where p=1

-- State the theorem in Lean 4
theorem distance_A_focus (x : ℝ) (hA : point_A x) : |4 - focus_y_coord| = 3 :=
by
  -- Proof would go here
  sorry

end distance_A_focus_l538_538683


namespace value_of_a_100_l538_538555

def arithmetic_sequence_an (n : ℕ) : ℕ :=
1 + 4 * (n - 1)

theorem value_of_a_100 : arithmetic_sequence_an 100 = 397 :=
by {
  unfold arithmetic_sequence_an,
  simp,
  sorry
}

end value_of_a_100_l538_538555


namespace proof_1_proof_2_l538_538394

noncomputable def expr_1 (a b : ℝ) : ℝ :=
  2 * real.sqrt a - real.sqrt a

noncomputable def expr_2 (a b c : ℝ) : ℝ :=
  real.sqrt a * real.sqrt b / (1 / real.sqrt c)

theorem proof_1 : expr_1 2 2 = real.sqrt 2 :=
by sorry

theorem proof_2 : expr_2 2 10 5 = 10 :=
by sorry

end proof_1_proof_2_l538_538394


namespace find_m_l538_538952

theorem find_m 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (m : ℝ)
  (h_f : ∀ x, f x = x^2 - 4*x + m)
  (h_g : ∀ x, g x = x^2 - 2*x + 2*m)
  (h_cond : 3 * f 3 = g 3)
  : m = 12 := 
sorry

end find_m_l538_538952


namespace g_properties_l538_538061

def f (x : ℝ) : ℝ := x

def g (x : ℝ) : ℝ := -f x

theorem g_properties :
  (∀ x : ℝ, g (-x) = -g x) ∧ (∀ x y : ℝ, x < y → g x > g y) :=
by
  sorry

end g_properties_l538_538061


namespace avg_age_when_youngest_born_l538_538912

theorem avg_age_when_youngest_born
  (num_people : ℕ) (avg_age_now : ℝ) (youngest_age_now : ℝ) (sum_ages_others_then : ℝ) 
  (h1 : num_people = 7) 
  (h2 : avg_age_now = 30) 
  (h3 : youngest_age_now = 6) 
  (h4 : sum_ages_others_then = 150) :
  (sum_ages_others_then / num_people) = 21.43 :=
by
  sorry

end avg_age_when_youngest_born_l538_538912


namespace sin_690_eq_negative_one_half_l538_538417

theorem sin_690_eq_negative_one_half : Real.sin (690 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_690_eq_negative_one_half_l538_538417


namespace tan_of_cos_l538_538836

theorem tan_of_cos (α : ℝ) (h1 : cos α = -3 / 5) (h2 : 0 < α ∧ α < π) : tan α = -4 / 3 :=
by
  sorry

end tan_of_cos_l538_538836


namespace min_abs_z_w_l538_538591

-- Definitions of constants given in the problem
def z : ℂ := sorry
def w : ℂ := sorry
def a : ℂ := 2 + 4 * complex.I
def b : ℂ := 5 + 6 * complex.I

-- Conditions given in the problem
axiom hz : abs (z - a) = 2
axiom hw : abs (w - b) = 4

-- The theorem we need to prove
theorem min_abs_z_w : ∃ z w : ℂ, abs (z - a) = 2 ∧ abs (w - b) = 4 ∧ abs (z - w) = real.sqrt 13 - 6 :=
sorry

end min_abs_z_w_l538_538591


namespace sin_690_deg_l538_538429

noncomputable def sin_690_eq_neg_one_half : Prop :=
  sin (690 * real.pi / 180) = -(1 / 2)

theorem sin_690_deg : sin_690_eq_neg_one_half :=
  by sorry

end sin_690_deg_l538_538429


namespace trig_inequalities_l538_538204

theorem trig_inequalities :
  let sin_168 := Real.sin (168 * Real.pi / 180)
  let cos_10 := Real.cos (10 * Real.pi / 180)
  let tan_58 := Real.tan (58 * Real.pi / 180)
  let tan_45 := Real.tan (45 * Real.pi / 180)
  sin_168 < cos_10 ∧ cos_10 < tan_58 :=
  sorry

end trig_inequalities_l538_538204


namespace problem_solving_days_l538_538636

theorem problem_solving_days 
  (a b c : ℕ) 
  (h1 : 5 * (11 * a + 7 * b + 9 * c) = 16 * (4 * a + 2 * b + 3 * c))
  (h2 : b = 3 * a + c) :
  let x := 40 in
  x = 40 :=
by sorry

end problem_solving_days_l538_538636


namespace perpendicular_lines_a_value_l538_538072

theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, (x + a * y - a = 0) ∧ (a * x - (2 * a - 3) * y - 1 = 0) → x ≠ y) →
  a = 0 ∨ a = 2 :=
sorry

end perpendicular_lines_a_value_l538_538072


namespace sum_of_solutions_l538_538007

theorem sum_of_solutions (x y : ℝ) (h1 : y = 10) (h2 : x^2 + y^2 = 200) : x = 10 ∨ x = -10 ∧ (10 + (-10) = 0) :=
by 
    have h3: x^2 = 100, 
        from (by rw [h1] at h2; linarith)
    cases em (x = 10),
    case inl h4 { 
        -- x = 10
        exact Or.inl h4
    },
    case inr h4 { 
        -- x = -10
        exact Or.inr (And.intro h4 (by sorry)) 
    }


end sum_of_solutions_l538_538007


namespace remaining_box_mass_l538_538692

theorem remaining_box_mass (masses : List ℕ)
  (h1 : masses = [15, 16, 18, 19, 20, 31])
  (h2 : ∃ x, ∃ boxes1 boxes2 : List ℕ,
                  boxes1 ++ boxes2 = masses \ [20] ∧
                  sum boxes1 = x ∧ sum boxes2 = 2 * x ∧ 
                  sum masses = sum boxes1 + sum boxes2 + 20) : 
  20 ∈ masses ∧ (20 % 3 = 2) :=
by
  rw masses at h1
  cases h2 with x hx
  cases hx with boxes1 hx'
  cases hx' with boxes2 hx''
  cases hx'' with h_union h_sums
  cases h_sums with h_box1 h_box2
  apply And.intro
  . rw h1
    exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_self _)))))
  . have h_total_sum : 15 + 16 + 18 + 19 + 20 + 31 = 119 := by norm_num
    rw masses at h_total_sum
    cases h_sums with  h_box_sum
    rw ← h_total_sum at h_box_sum
    have h_mod : 119 % 3 = 2 := by norm_num
    exact h_mod
  sorry

end remaining_box_mass_l538_538692


namespace volume_of_T_l538_538213

def T (x y z : ℝ) : Prop :=
  abs x + abs y ≤ 2 ∧ abs x + abs z ≤ 2 ∧ abs y + abs z ≤ 2

theorem volume_of_T : volume {p : ℝ × ℝ × ℝ | T p.1 p.2 p.3} = 8 / 3 := 
sorry

end volume_of_T_l538_538213


namespace number_of_integers_satisfying_conditions_l538_538898

-- Define the problem constraints and proof statement
open Real

noncomputable def satisfies_equation (a : ℝ) : Prop :=
  ∃ x : ℕ, (x ≠ 2) ∧ ((x + a) / (x - 2) + (2 * x) / (2 - x) = 1)

noncomputable def satisfies_inequalities (a : ℝ) : Prop :=
  let solutions := { y : ℤ | (y ≥ 2 + 10 * y - 3) ∧ (y ≤ 14) } in
  (solutions.size = 4)

theorem number_of_integers_satisfying_conditions :
  (finset.filter satisfies_equation (finset.range 11)).size = 4 :=
begin
  sorry -- Proof goes here
end

end number_of_integers_satisfying_conditions_l538_538898


namespace original_price_proof_l538_538292

noncomputable def original_price (profit selling_price : ℝ) : ℝ :=
  (profit / 0.20)

theorem original_price_proof (P : ℝ) : 
  original_price 600 (P + 600) = 3000 :=
by
  sorry

end original_price_proof_l538_538292


namespace monotonic_f_range_l538_538525

-- Define the function f(x) = 2 * sin(2 * x + π/6)
def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

-- Define the ranges where the function is monotonically increasing
def interval_1 (x₀ : ℝ) : Prop := 0 < x₀ ∧ x₀ / 3 ≤ Real.pi / 6
def interval_2 (x₀ : ℝ) : Prop := 2 * x₀ ≥ 2 * Real.pi / 3 ∧ 2 * x₀ ≤ 7 * Real.pi / 6

-- Main theorem statement
theorem monotonic_f_range (x₀ : ℝ) (h₁ : interval_1 x₀) (h₂ : interval_2 x₀) : x₀ ∈ Set.Icc (Real.pi / 3) (Real.pi / 2) :=
sorry

end monotonic_f_range_l538_538525


namespace arithmetic_sequence_sum_ratio_l538_538691

theorem arithmetic_sequence_sum_ratio
  (S : ℕ → ℝ) (T : ℕ → ℝ) (a b : ℕ → ℝ)
  (k : ℝ)
  (h1 : ∀ n, S n = 3 * k * n^2)
  (h2 : ∀ n, T n = k * n * (2 * n + 1))
  (h3 : ∀ n, a n = S n - S (n - 1))
  (h4 : ∀ n, b n = T n - T (n - 1))
  (h5 : ∀ n, S n / T n = (3 * n) / (2 * n + 1)) :
  (a 1 + a 2 + a 14 + a 19) / (b 1 + b 3 + b 17 + b 19) = 17 / 13 :=
sorry

end arithmetic_sequence_sum_ratio_l538_538691


namespace jesus_squares_l538_538622

theorem jesus_squares (J : ℕ) (linden_squares : ℕ) (pedro_squares : ℕ)
  (h1 : linden_squares = 75)
  (h2 : pedro_squares = 200)
  (h3 : pedro_squares = J + linden_squares + 65) : 
  J = 60 := 
by
  sorry

end jesus_squares_l538_538622


namespace isosceles_triangle_area_l538_538375

noncomputable def area_of_isosceles_triangle (b s : ℝ) (h1 : 10 * 10 + b * b = s * s) (h2 : s + b = 20) : ℝ :=
  1 / 2 * (2 * b) * 10

theorem isosceles_triangle_area (b s : ℝ) (h1 : 10 * 10 + b * b = s * s) (h2 : s + b = 20)
  (h3 : 2 * s + 2 * b = 40) : area_of_isosceles_triangle b s h1 h2 = 75 :=
sorry

end isosceles_triangle_area_l538_538375


namespace sin_identity_l538_538090

theorem sin_identity (α : ℝ) (h : Real.cos (75 * Real.pi / 180 + α) = 1 / 3) :
  Real.sin (60 * Real.pi / 180 + 2 * α) = 7 / 9 :=
by
  sorry

end sin_identity_l538_538090


namespace ball_bounces_l538_538615

theorem ball_bounces (width height : ℕ) (start_angle : ℕ) 
  (P Q R S : string) (bottom_left top_right : string) : 
  width = 5 ∧ height = 2 ∧ start_angle = 45 ∧ 
  P = "bottom left corner" ∧ S = "top right corner" → 
  ∃ bounces : ℕ, bounces = 5 :=
begin
  sorry
end

end ball_bounces_l538_538615


namespace evaluate_g_at_neg_four_l538_538880

def g (x : ℤ) : ℤ := 5 * x + 2

theorem evaluate_g_at_neg_four : g (-4) = -18 := 
by 
  sorry

end evaluate_g_at_neg_four_l538_538880


namespace train_platform_proof_l538_538309

-- Defining the given conditions
def train_length : ℝ := 100
def platform_2_length : ℝ := 300
def time_to_cross_platform_1 : ℝ := 15
def time_to_cross_platform_2 : ℝ := 20
def distance_crossed_platform_1 (L : ℝ) : ℝ := train_length + L
def distance_crossed_platform_2 : ℝ := train_length + platform_2_length
def speed_platform_1 (L : ℝ) : ℝ := distance_crossed_platform_1 L / time_to_cross_platform_1
def speed_platform_2 : ℝ := distance_crossed_platform_2 / time_to_cross_platform_2

-- Problem statement to prove
theorem train_platform_proof (L : ℝ) : speed_platform_1 L = speed_platform_2 → L = 200 :=
by
  sorry

end train_platform_proof_l538_538309


namespace locus_equation_l538_538057

-- Defining the fixed point A
def A : ℝ × ℝ := (4, -2)

-- Defining the predicate that B lies on the circle
def on_circle (B : ℝ × ℝ) : Prop := B.1^2 + B.2^2 = 4

-- Defining the locus of the midpoint P
def locus_of_midpoint (P : ℝ × ℝ) : Prop := (P.1 - 2)^2 + (P.2 + 1)^2 = 1

-- Main theorem statement
theorem locus_equation (P : ℝ × ℝ) (B : ℝ × ℝ) :
  on_circle(B) →
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  locus_of_midpoint(P) :=
by
  sorry

end locus_equation_l538_538057


namespace sum_of_segments_eq_radius_l538_538296

open_locale big_operators

-- Definitions and assumptions
variables {k : ℕ} (n : ℕ := 4 * k + 2) (R : ℝ) (O : ℝ)

def regular_polygon (n : ℕ) : Prop := true -- Assume a regular polygon inscribed in a circle

-- Main theorem statement
theorem sum_of_segments_eq_radius
  (h_polygon : regular_polygon n)
  (h_inscribed : true) -- The polygon is inscribed in a circle with radius R
  (h_center : true) -- The center of the circle is O
  (segments : fin k → ℝ) :
  segments.sum = R :=
sorry

end sum_of_segments_eq_radius_l538_538296


namespace num_distinct_x_intercepts_l538_538872

def f (x : ℝ) : ℝ := (x - 5) * (x^3 + 5*x^2 + 9*x + 9)

theorem num_distinct_x_intercepts : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x1 ∨ x = x2) :=
sorry

end num_distinct_x_intercepts_l538_538872


namespace area_of_reflected_arcs_l538_538297

noncomputable def area_of_bounded_region (s : ℝ) : ℝ :=
  2 + 2 * Real.sqrt 2 - (Real.pi / (2 - Real.sqrt 2)) + 2 * Real.sqrt (2 - Real.sqrt 2)

theorem area_of_reflected_arcs :
  ∀ (s : ℝ), s = 1 → 
  let area := area_of_bounded_region s in 
  area = 2 + 2 * Real.sqrt 2 - (Real.pi / (2 - Real.sqrt 2)) + 2 * Real.sqrt (2 - Real.sqrt 2) :=
begin
  intros s h1,
  simp [h1, area_of_bounded_region],
  sorry
end

end area_of_reflected_arcs_l538_538297


namespace total_spent_two_years_l538_538564

def home_game_price : ℕ := 60
def away_game_price : ℕ := 75
def home_playoff_price : ℕ := 120
def away_playoff_price : ℕ := 100

def this_year_home_games : ℕ := 2
def this_year_away_games : ℕ := 2
def this_year_home_playoff_games : ℕ := 1
def this_year_away_playoff_games : ℕ := 0

def last_year_home_games : ℕ := 6
def last_year_away_games : ℕ := 3
def last_year_home_playoff_games : ℕ := 1
def last_year_away_playoff_games : ℕ := 1

def calculate_total_cost : ℕ :=
  let this_year_cost := this_year_home_games * home_game_price + this_year_away_games * away_game_price + this_year_home_playoff_games * home_playoff_price + this_year_away_playoff_games * away_playoff_price
  let last_year_cost := last_year_home_games * home_game_price + last_year_away_games * away_game_price + last_year_home_playoff_games * home_playoff_price + last_year_away_playoff_games * away_playoff_price
  this_year_cost + last_year_cost

theorem total_spent_two_years : calculate_total_cost = 1195 :=
by
  sorry

end total_spent_two_years_l538_538564


namespace area_ratio_of_triangles_l538_538127

-- Define the conditions
variables (a : ℝ)
          (A B C D E : ℝ)
          (AB CD : ℝ)
          (beta : ℝ)

-- Assume given conditions
axiom AB_diameter (AB = 2 * a): Prop
axiom CD_chord_parallel_AB (CD = a): Prop
axiom AC_BD_intersect_E : Prop
axiom angle_AED (beta = 30): Prop

-- Theorem to prove
theorem area_ratio_of_triangles 
  (h1 : AB_diameter (AB := 2 * a)) 
  (h2 : CD_chord_parallel_AB (CD := a)) 
  (h3 : AC_BD_intersect_E)
  (h4 : angle_AED (beta := 30)) : 
  (area (CDE) / area (ABE)) = 3 / 4 :=
  sorry

end area_ratio_of_triangles_l538_538127


namespace imo_power_of_i_l538_538838

noncomputable def i : ℂ := complex.I

theorem imo_power_of_i :
  i ^ 2015 = -i :=
by
  sorry

end imo_power_of_i_l538_538838


namespace determine_x_for_which_h3x_equals_3hx_l538_538185

noncomputable def h (x : ℝ) : ℝ := real.root 4 ((x + 5) / 5)

theorem determine_x_for_which_h3x_equals_3hx :
  ∃ x : ℝ, h (3 * x) = 3 * h x ∧ x = -200 / 39 :=
by
  sorry

end determine_x_for_which_h3x_equals_3hx_l538_538185


namespace inverse_proposition_true_l538_538860

theorem inverse_proposition_true (x : ℝ) (h : x > 1 → x^2 > 1) : x^2 ≤ 1 → x ≤ 1 :=
by
  intros h₂
  sorry

end inverse_proposition_true_l538_538860


namespace triangle_construction_condition_l538_538396

variable (varrho_a varrho_b m_c : ℝ)

theorem triangle_construction_condition :
  (∃ (triangle : Type) (ABC : triangle)
    (r_a : triangle → ℝ)
    (r_b : triangle → ℝ)
    (h_from_C : triangle → ℝ),
      r_a ABC = varrho_a ∧
      r_b ABC = varrho_b ∧
      h_from_C ABC = m_c)
  ↔ 
  (1 / m_c = 1 / 2 * (1 / varrho_a + 1 / varrho_b)) :=
sorry

end triangle_construction_condition_l538_538396


namespace annulus_area_l538_538782

theorem annulus_area (r R x : ℝ) (hR_gt_r : R > r) (h_tangent : r^2 + x^2 = R^2) : 
  π * x^2 = π * (R^2 - r^2) :=
by
  sorry

end annulus_area_l538_538782


namespace consumer_installment_credit_l538_538763

theorem consumer_installment_credit (A C : ℝ) 
  (h1 : A = 0.36 * C) 
  (h2 : 57 = 1 / 3 * A) : 
  C = 475 := 
by 
  sorry

end consumer_installment_credit_l538_538763


namespace average_speed_of_entire_trip_l538_538270

/-- Conditions -/
def distance_local : ℝ := 40  -- miles
def speed_local : ℝ := 20  -- mph
def distance_highway : ℝ := 180  -- miles
def speed_highway : ℝ := 60  -- mph

/-- Average speed proof statement -/
theorem average_speed_of_entire_trip :
  let total_distance := distance_local + distance_highway
  let total_time := distance_local / speed_local + distance_highway / speed_highway
  total_distance / total_time = 44 :=
by
  sorry

end average_speed_of_entire_trip_l538_538270


namespace tricksters_identification_l538_538340

variable (Inhabitant : Type)
variable [inhab : Fintype Inhabitant]
variable [decEqInhab : DecidableEq Inhabitant]
variable (knight : Inhabitant → Prop)
variable (trickster : Inhabitant → Prop)

variables (n : ℕ) (q : ℕ)
variable (is_truthful : Inhabitant → Prop)

constant inhabitants_count : 65
constant tricksters_count : 2

-- Define the property that a knight always tells the truth.
axiom knight_truth (x : Inhabitant) : knight x → is_truthful x

-- Define the property that a trickster can tell the truth or lie.
axiom trickster_behavior (x : Inhabitant) : trickster x → (is_truthful x ∨ ¬ is_truthful x)

 -- Define the type of the question which can be asked to an inhabitant.
inductive Question (Inhabitant : Type) : Type
| is_knight : Inhabitant → Question

-- Define the type of the answer to the question.
inductive Answer (Inhabitant : Type) : Type
| yes : Answer
| no :  Answer

-- Define a function that simulates asking a question to an inhabitant.
constant ask : Inhabitant → Question Inhabitant → Answer Inhabitant

noncomputable def find_tricksters (inhabitants : fin inhabitants_count → Inhabitant) : (fin tricksters_count → Inhabitant) :=
sorry

theorem tricksters_identification : 
  ∃ (f : (fin inhabitants_count → Inhabitant) → (fin tricksters_count → Inhabitant)), 
    ∀ inhabitants : fin inhabitants_count → Inhabitant, 
      (∀ (q_list : list (Inhabitant × Question Inhabitant)),
        q_list.length ≤ 30 → 
        let a_list := q_list.map (λ pq, ask (pq.fst) (pq.snd)) in 
        true) ∧ 
      (∃ t1 t2, trickster (f inhabitants) t1 ∧ trickster (f inhabitants) t2) :=
sorry

end tricksters_identification_l538_538340


namespace number_of_valid_a_l538_538895

theorem number_of_valid_a : 
  (∃(a : ℤ), ∃(x : ℕ), (frac_eq : (x + a) / (x - 2) + 2x / (2 - x) = 1) ∧ 
  (ineq_sys : exactIntegerSolutions (λ y, (y + 1) / 5 ≥ y / 2 - 1 ∧ y + a < 11y - 3) 4)) ↔ 
  ((a = -2) ∨ (a = 0) ∨ (a = 4) ∨ (a = 6)) := 
begin
  sorry
end

end number_of_valid_a_l538_538895


namespace multiple_in_subset_l538_538822

theorem multiple_in_subset (n : ℕ) (h : 0 < n) (s : Finset ℕ) (hs : s.card = n + 1) (hsub : s ⊆ Finset.range (2 * n + 1)) :
  ∃ a b ∈ s, (a ≠ b) ∧ (a % b = 0 ∨ b % a = 0) :=
begin
  sorry
end

end multiple_in_subset_l538_538822


namespace math_problem_l538_538495

noncomputable def f (x : ℝ) : ℝ := if h : x > 0 then x * Real.log x else 0

theorem math_problem :
  (∀ x, x > 0 → f x = x * Real.log x) ∧
  (∀ x, x ≤ 0 → f x = 0) ∧ 
  (∃ x > 0, Real.log x + 1 = 2 → x = Real.exp 1) ∧
  (∀ x, x > 1 / Real.exp 1 → Real.log x + 1 > 0) ∧
  (∀ ε, 0 < ε → ∃ x, x = 1 / Real.exp 1 ∧ f x = -1 / Real.exp 1) :=
by sorry

end math_problem_l538_538495


namespace james_january_income_l538_538937

variable (January February March : ℝ)
variable (h1 : February = 2 * January)
variable (h2 : March = February - 2000)
variable (h3 : January + February + March = 18000)

theorem james_january_income : January = 4000 := by
  sorry

end james_january_income_l538_538937


namespace find_tricksters_l538_538324

structure Inhabitant :=
  (isKnight : Prop)

constants (inhabitants : Fin 65 → Inhabitant)
          (tricksters : Fin 2 → Fin 65)

axiom two_tricksters_unique :
  ∃! (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65))

axiom valid_question :
  ∀ (a : Fin 65) (group : Set (Fin 65)), (inhabitants a).isKnight → 
  (∀ i ∈ group, (inhabitants i).isKnight) ↔ 
  (knight a).isKnight

theorem find_tricksters :
  ∃ (q : ℕ), q ≤ 16 ∧
  ∃ (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65)) :=
sorry

end find_tricksters_l538_538324


namespace necklace_length_l538_538289

-- Given conditions as definitions in Lean
def num_pieces : ℕ := 16
def piece_length : ℝ := 10.4
def overlap_length : ℝ := 3.5
def effective_length : ℝ := piece_length - overlap_length
def total_length : ℝ := effective_length * num_pieces

-- The theorem to prove
theorem necklace_length :
  total_length = 110.4 :=
by
  -- Proof omitted
  sorry

end necklace_length_l538_538289


namespace isosceles_triangle_AE_equals_BD_l538_538117

theorem isosceles_triangle_AE_equals_BD
  (A B C D E : Type) [triangle ABC] (isosceles : is_isosceles ABC)
  (angle_B: measure_angle B = 108)
  (D_on_angle_bisector : is_angle_bisector C D (∠ A C B))
  (E_perpendicular : is_perpendicular D E A C ) :
  AE = BD := 
sorry

end isosceles_triangle_AE_equals_BD_l538_538117


namespace loom_weave_rate_l538_538274

-- Defining the given conditions as constants
def time_taken : ℝ := 118.11
def cloth_woven : ℝ := 15

-- Defining the expected rate
def expected_rate : ℝ := 0.127

-- Stating the proof problem
theorem loom_weave_rate :
  (cloth_woven / time_taken ≈ expected_rate) :=
sorry

end loom_weave_rate_l538_538274


namespace hyperbola_equation_common_asymptote_l538_538658

theorem hyperbola_equation_common_asymptote (A : ℝ × ℝ) (AsymptoticHyperbola : Set (ℝ × ℝ))
    (hA : A = (3 * Real.sqrt 3, -3))
    (hHyp : AsymptoticHyperbola = { P : ℝ × ℝ | (P.1^2 / 16) - (P.2^2 / 9) = 1 }) :
    ∃ (λ : ℝ), λ ≠ 0 ∧
      ∀ P : ℝ × ℝ, P ∈ AsymptoticHyperbola → (P.1^2 / 16) - (P.2^2 / 9) = λ ∧
      ∀ P : ℝ × ℝ, P = A → P.1^2 / 11 - P.2^2 * (16 / 99) = 1 := 
sorry

end hyperbola_equation_common_asymptote_l538_538658


namespace close_one_station_l538_538125

-- Definitions based on conditions
variable {Station : Type}
variable [Fintype Station]
variable (connected : Station → Station → Prop)
variable (interconnected : ∀ A B : Station, ∃ path: List Station, path.head = A ∧ path.last = B ∧ ∀ (i j : ℕ), i < j ∧ j < path.length → connected (path.nth_le i sorry) (path.nth_le j sorry))

-- Theorem statement based on the question
theorem close_one_station (S : Station) :
  ∃ T : Station, ∀ U V : Station, U ≠ T ∧ V ≠ T → ∃ path : List Station, path.head = U ∧ path.last = V ∧ ∀ (i j : ℕ), i < j ∧ j < path.length → (path.nth_le i sorry) ≠ T ∧ (path.nth_le j sorry) ≠ T ∧ connected (path.nth_le i sorry) (path.nth_le j sorry) :=
by
  sorry

end close_one_station_l538_538125


namespace pyramid_volume_l538_538736

theorem pyramid_volume (a b l : ℝ) (h_base : a = 4) (w_base : b = 10) (edge_a_b : l = 15) : 
  let volume := (1 / 3) * (a * b) * (√(l^2 - (a^2 + b^2) / 4)) 
  in volume = 560 / 3 :=
by
  sorry

end pyramid_volume_l538_538736


namespace x_interval_l538_538887

theorem x_interval (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -4) (h3 : 2 * x - 1 > 0) : x > 1 / 2 := 
sorry

end x_interval_l538_538887


namespace slices_leftover_is_9_l538_538387

-- Conditions and definitions
def total_pizzas : ℕ := 2
def slices_per_pizza : ℕ := 12
def bob_ate : ℕ := slices_per_pizza / 2
def tom_ate : ℕ := slices_per_pizza / 3
def sally_ate : ℕ := slices_per_pizza / 6
def jerry_ate : ℕ := slices_per_pizza / 4

-- Calculate total slices eaten and left over
def total_slices_eaten : ℕ := bob_ate + tom_ate + sally_ate + jerry_ate
def total_slices_available : ℕ := total_pizzas * slices_per_pizza
def slices_leftover : ℕ := total_slices_available - total_slices_eaten

-- Theorem to prove the number of slices left over
theorem slices_leftover_is_9 : slices_leftover = 9 := by
  -- Proof: omitted, add relevant steps here
  sorry

end slices_leftover_is_9_l538_538387


namespace tricksters_identification_l538_538341

variable (Inhabitant : Type)
variable [inhab : Fintype Inhabitant]
variable [decEqInhab : DecidableEq Inhabitant]
variable (knight : Inhabitant → Prop)
variable (trickster : Inhabitant → Prop)

variables (n : ℕ) (q : ℕ)
variable (is_truthful : Inhabitant → Prop)

constant inhabitants_count : 65
constant tricksters_count : 2

-- Define the property that a knight always tells the truth.
axiom knight_truth (x : Inhabitant) : knight x → is_truthful x

-- Define the property that a trickster can tell the truth or lie.
axiom trickster_behavior (x : Inhabitant) : trickster x → (is_truthful x ∨ ¬ is_truthful x)

 -- Define the type of the question which can be asked to an inhabitant.
inductive Question (Inhabitant : Type) : Type
| is_knight : Inhabitant → Question

-- Define the type of the answer to the question.
inductive Answer (Inhabitant : Type) : Type
| yes : Answer
| no :  Answer

-- Define a function that simulates asking a question to an inhabitant.
constant ask : Inhabitant → Question Inhabitant → Answer Inhabitant

noncomputable def find_tricksters (inhabitants : fin inhabitants_count → Inhabitant) : (fin tricksters_count → Inhabitant) :=
sorry

theorem tricksters_identification : 
  ∃ (f : (fin inhabitants_count → Inhabitant) → (fin tricksters_count → Inhabitant)), 
    ∀ inhabitants : fin inhabitants_count → Inhabitant, 
      (∀ (q_list : list (Inhabitant × Question Inhabitant)),
        q_list.length ≤ 30 → 
        let a_list := q_list.map (λ pq, ask (pq.fst) (pq.snd)) in 
        true) ∧ 
      (∃ t1 t2, trickster (f inhabitants) t1 ∧ trickster (f inhabitants) t2) :=
sorry

end tricksters_identification_l538_538341


namespace rectangle_width_l538_538192

theorem rectangle_width (side_length_square : ℕ) (length_rectangle : ℕ) (area_equal : side_length_square * side_length_square = length_rectangle * w) : w = 4 := by
  sorry

end rectangle_width_l538_538192


namespace part1_1_has_property_P_part1_2_has_property_P_part1_3_does_not_have_property_P_part2_count_not_have_property_P_l538_538959

-- Part (1)
def has_property_P (n x y z : ℤ) : Prop :=
  n = x^3 + y^3 + z^3 - 3 * x * y * z

theorem part1_1_has_property_P : ∃ (x y z : ℤ), has_property_P 1 x y z :=
sorry

theorem part1_2_has_property_P : ∃ (x y z : ℤ), has_property_P 2 x y z :=
sorry

theorem part1_3_does_not_have_property_P : ¬∃ (x y z : ℤ), has_property_P 3 x y z :=
sorry

-- Part (2)
theorem part2_count_not_have_property_P : (1..2014).count (λ n, ¬∃ (x y z : ℤ), has_property_P n x y z) = 223 :=
sorry

end part1_1_has_property_P_part1_2_has_property_P_part1_3_does_not_have_property_P_part2_count_not_have_property_P_l538_538959


namespace top3_criteria_l538_538552

-- Define the setup: there are 7 students with different scores
variables (scores : Fin 7 → ℝ)

-- Define Xiaohong's score
variable (xiaohong_score : ℝ)

-- Define the condition that all scores are different
axiom distinct_scores : function.injective scores

-- Define the median of a list of 7 unique sorted scores
def median (s : Fin 7 → ℝ) : ℝ :=
  Multiset.median (Multiset.map (λ i => s i) Finset.univ.val)

-- Define the problem statement
theorem top3_criteria :
  ∃ (median_score : ℝ), (median scores = median_score) :=
  sorry

end top3_criteria_l538_538552


namespace find_sum_of_pqrs_l538_538152

variables (p q r s : ℝ)

-- Defining conditions
def distinct (a b c d : ℝ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem find_sum_of_pqrs
    (h_distinct : distinct p q r s)
    (h_roots1 : r + s = 8 * p ∧ r * s = -12 * q)
    (h_roots2 : p + q = 8 * r ∧ p * q = -12 * s) :
    p + q + r + s = 864 := 
sorry

end find_sum_of_pqrs_l538_538152


namespace cyclic_quad_side_product_eq_diagonal_product_l538_538978

variable (A B C D : Point)
variable (AB BC CD DA AC BD : ℝ)

-- A predicate that checks if a quadrilateral is cyclic (i.e., inscribed in a circle)
def cyclic_quad (A B C D : Point) : Prop := 
  -- some definition of a cyclic quadrilateral involving angles or other properties

theorem cyclic_quad_side_product_eq_diagonal_product
  (h_cyclic : cyclic_quad A B C D)
  (h1 : side_length A B = AB)
  (h2 : side_length B C = BC)
  (h3 : side_length C D = CD)
  (h4 : side_length D A = DA)
  (ha : diagonal_length A C = AC)
  (hb : diagonal_length B D = BD) :
  AC * BD = AD * BC + AB * DC :=
by
  -- proof
  sorry

end cyclic_quad_side_product_eq_diagonal_product_l538_538978


namespace susie_rooms_l538_538222

-- Define the conditions
def vacuum_time_per_room : ℕ := 20  -- in minutes
def total_vacuum_time : ℕ := 2 * 60  -- 2 hours in minutes

-- Define the number of rooms in Susie's house
def number_of_rooms (total_time room_time : ℕ) : ℕ := total_time / room_time

-- Prove that Susie has 6 rooms in her house
theorem susie_rooms : number_of_rooms total_vacuum_time vacuum_time_per_room = 6 :=
by
  sorry -- proof goes here

end susie_rooms_l538_538222


namespace max_value_of_quadratic_l538_538522

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := -2 * x^2 + 8

-- State the problem formally in Lean
theorem max_value_of_quadratic : ∀ x : ℝ, quadratic x ≤ quadratic 0 :=
by
  -- Skipping the proof
  sorry

end max_value_of_quadratic_l538_538522


namespace h_odd_l538_538105

variable (f g : ℝ → ℝ)

-- f is odd and g is even
axiom f_odd : ∀ x, -2 ≤ x ∧ x ≤ 2 → f (-x) = -f x
axiom g_even : ∀ x, -2 ≤ x ∧ x ≤ 2 → g (-x) = g x

-- Prove that h(x) = f(x) * g(x) is odd
theorem h_odd : ∀ x, -2 ≤ x ∧ x ≤ 2 → (f x) * (g x) = (f (-x)) * (g (-x)) := by
  sorry

end h_odd_l538_538105


namespace price_of_turban_l538_538077

theorem price_of_turban : 
  ∃ T : ℝ, (9 / 12) * (90 + T) = 40 + T ↔ T = 110 :=
by
  sorry

end price_of_turban_l538_538077


namespace find_coefficients_g_l538_538208

-- Define the polynomial f(x)
def f : Polynomial ℝ := Polynomial.C 3 + Polynomial.C 2 * Polynomial.X + Polynomial.C 1 * Polynomial.X ^ 2 + Polynomial.X ^ 3

-- Define the polynomial g(x)
def g : Polynomial ℝ := Polynomial.C (-9) + Polynomial.C (-2) * Polynomial.X + Polynomial.C 3 * Polynomial.X ^ 2 + Polynomial.X ^ 3

-- Problem statement
theorem find_coefficients_g :
  (∃ b c d, g = Polynomial.X ^ 3 + Polynomial.C b * Polynomial.X ^ 2 + Polynomial.C c * Polynomial.X + Polynomial.C d
            ∧ (∀ r, r ∈ (f.roots : Multiset ℝ) → r^2 ∈ (g.roots : Multiset ℝ))
            ∧ b = 3 ∧ c = -2 ∧ d = -9) :=
sorry

end find_coefficients_g_l538_538208


namespace total_trophies_correct_l538_538936

-- Define the current number of Michael's trophies
def michael_current_trophies : ℕ := 30

-- Define the number of trophies Michael will have in three years
def michael_trophies_in_three_years : ℕ := michael_current_trophies + 100

-- Define the number of trophies Jack will have in three years
def jack_trophies_in_three_years : ℕ := 10 * michael_current_trophies

-- Define the total number of trophies Jack and Michael will have after three years
def total_trophies_in_three_years : ℕ := michael_trophies_in_three_years + jack_trophies_in_three_years

-- Prove that the total number of trophies after three years is 430
theorem total_trophies_correct : total_trophies_in_three_years = 430 :=
by
  sorry -- proof is omitted

end total_trophies_correct_l538_538936


namespace green_peaches_count_l538_538219

def red_peaches : ℕ := 17
def green_peaches (x : ℕ) : Prop := red_peaches = x + 1

theorem green_peaches_count (x : ℕ) (h : green_peaches x) : x = 16 :=
by
  sorry

end green_peaches_count_l538_538219


namespace sum_divisible_by_15_l538_538979

theorem sum_divisible_by_15 (a : ℤ) : 15 ∣ (9 * a^5 - 5 * a^3 - 4 * a) :=
sorry

end sum_divisible_by_15_l538_538979


namespace greatest_divisors_l538_538186

theorem greatest_divisors (b n : ℕ) (hb : b ≤ 20) (hn : n ≤ 20) (hb_pos : 0 < b) (hn_pos : 0 < n) :
  ∃ b n, b ≤ 20 ∧ n ≤ 20 ∧ (λ b n, ∃ d, ∀ m, (m = (1 + 20) * (1 + 40) = 861)) :=
sorry

end greatest_divisors_l538_538186


namespace find_tricksters_in_16_questions_l538_538336

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

end find_tricksters_in_16_questions_l538_538336


namespace tricksters_identification_l538_538345

variable (Inhabitant : Type)
variable [inhab : Fintype Inhabitant]
variable [decEqInhab : DecidableEq Inhabitant]
variable (knight : Inhabitant → Prop)
variable (trickster : Inhabitant → Prop)

variables (n : ℕ) (q : ℕ)
variable (is_truthful : Inhabitant → Prop)

constant inhabitants_count : 65
constant tricksters_count : 2

-- Define the property that a knight always tells the truth.
axiom knight_truth (x : Inhabitant) : knight x → is_truthful x

-- Define the property that a trickster can tell the truth or lie.
axiom trickster_behavior (x : Inhabitant) : trickster x → (is_truthful x ∨ ¬ is_truthful x)

 -- Define the type of the question which can be asked to an inhabitant.
inductive Question (Inhabitant : Type) : Type
| is_knight : Inhabitant → Question

-- Define the type of the answer to the question.
inductive Answer (Inhabitant : Type) : Type
| yes : Answer
| no :  Answer

-- Define a function that simulates asking a question to an inhabitant.
constant ask : Inhabitant → Question Inhabitant → Answer Inhabitant

noncomputable def find_tricksters (inhabitants : fin inhabitants_count → Inhabitant) : (fin tricksters_count → Inhabitant) :=
sorry

theorem tricksters_identification : 
  ∃ (f : (fin inhabitants_count → Inhabitant) → (fin tricksters_count → Inhabitant)), 
    ∀ inhabitants : fin inhabitants_count → Inhabitant, 
      (∀ (q_list : list (Inhabitant × Question Inhabitant)),
        q_list.length ≤ 30 → 
        let a_list := q_list.map (λ pq, ask (pq.fst) (pq.snd)) in 
        true) ∧ 
      (∃ t1 t2, trickster (f inhabitants) t1 ∧ trickster (f inhabitants) t2) :=
sorry

end tricksters_identification_l538_538345


namespace math_problem_l538_538089

theorem math_problem (m n : ℝ) (h1 : 2 ^ m = 5) (h2 : 5 ^ n = 2) : 
  (1 / (m + 1) + 1 / (n + 1) = 1) :=
sorry

end math_problem_l538_538089


namespace cost_price_percentage_l538_538997

theorem cost_price_percentage (CP SP : ℝ) (h1 : SP = 4 * CP) : (CP / SP) * 100 = 25 :=
by
  sorry

end cost_price_percentage_l538_538997


namespace smallest_percent_increase_l538_538913

def question_values : List ℕ := [100, 300, 600, 900, 1500, 2400]

def percent_increase (v1 v2 : ℕ) : ℚ := 100 * (v2 - v1) / v1

def percent_increases (vals : List ℕ) : List ℚ :=
  vals.zipWith vals.tail percent_increase

theorem smallest_percent_increase : 
  percent_increases question_values = [200, 100, 50, 66.67, 60] →
  argmin (percent_increases question_values) = 2 :=
begin
  sorry
end

end smallest_percent_increase_l538_538913


namespace find_x_l538_538020

-- Define that x is a natural number expressed as 2^n - 32
def x (n : ℕ) : ℕ := 2^n - 32

-- We assume x has exactly three distinct prime divisors
def has_three_distinct_prime_divisors (x : ℕ) : Prop :=
  ∃ (p q r : ℕ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r ∧ prime p ∧ prime q ∧ prime r

-- One of the prime divisors is 2
def prime_divisor_2 (x : ℕ) : Prop :=
  2 ∣ x 

-- Correct answer is either 2016 or 16352
theorem find_x (n : ℕ) : has_three_distinct_prime_divisors (x n) ∧ prime_divisor_2 (x n) → 
  (x n = 2016 ∨ x n = 16352) :=
sorry

end find_x_l538_538020


namespace zero_points_in_intervals_l538_538499

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) * x - Real.log x

theorem zero_points_in_intervals :
  (∀ x : ℝ, x ∈ Set.Ioo (1 / Real.exp 1) 1 → f x ≠ 0) ∧
  (∃ x : ℝ, x ∈ Set.Ioo 1 (Real.exp 1) ∧ f x = 0) :=
by
  sorry

end zero_points_in_intervals_l538_538499


namespace candy_last_days_l538_538456

def pieces_from_neighbors : ℝ := 11.0
def pieces_from_sister : ℝ := 5.0
def pieces_per_day : ℝ := 8.0
def total_pieces : ℝ := pieces_from_neighbors + pieces_from_sister

theorem candy_last_days : total_pieces / pieces_per_day = 2 := by
    sorry

end candy_last_days_l538_538456


namespace find_tricksters_l538_538348

theorem find_tricksters (inhabitants : Fin 65 → Prop) (is_knight : Fin 65 → Prop)
    (total_inhabitants : ∀ n, inhabitants n)
    (knights : ∀ n, is_knight n → inhabitants n)
    (tricksters_count : (∑ n, if ¬ is_knight n then 1 else 0) = 2)
    (knights_count : (∑ n, if is_knight n then 1 else 0) = 63)
    (knight_truth : ∀ n, is_knight n → ∀ l : list (Fin 65), (∀ m ∈ l, is_knight m) ↔ true)
    (ask_question : ∀ n, inhabitants n → ∀ l : list (Fin 65), bool) :
  ∃ (find_tricksters_function : (Fin 65 → Prop) → (Fin 65 → bool) → (list (Fin 65))) ,
    (length (find_tricksters_function inhabitants ask_question) ≤ 2) →
    (length (find_tricksters_function inhabitants ask_question) = 2) ∧
    ∀ t ∈ (find_tricksters_function inhabitants ask_question), ¬ is_knight t :=
by sorry

end find_tricksters_l538_538348


namespace find_tricksters_l538_538363

def inhab_group : Type := { n : ℕ // n < 65 }
def is_knight (i : inhab_group) : Prop := ∀ g : inhab_group, true -- Placeholder for the actual property

theorem find_tricksters (inhabitants : inhab_group → Prop)
  (is_knight : inhab_group → Prop)
  (knight_always_tells_truth : ∀ i : inhab_group, is_knight i → inhabitants i = true)
  (tricksters_2_and_rest_knights : ∃ t1 t2 : inhab_group, t1 ≠ t2 ∧ ¬is_knight t1 ∧ ¬is_knight t2 ∧
    (∀ i : inhab_group, i ≠ t1 → i ≠ t2 → is_knight i)) :
  ∃ find_them : inhab_group → inhab_group → Prop, (∀ q_count : ℕ, q_count ≤ 16) → 
  ∃ t1 t2 : inhab_group, find_them t1 t2 :=
by 
  -- The proof goes here
  sorry

end find_tricksters_l538_538363


namespace number_of_team_members_l538_538635

-- Let's define the conditions.
def packs : ℕ := 3
def pouches_per_pack : ℕ := 6
def total_pouches : ℕ := packs * pouches_per_pack
def coaches : ℕ := 3
def helpers : ℕ := 2
def total_people (members : ℕ) : ℕ := members + coaches + helpers

-- Prove the number of members on the baseball team.
theorem number_of_team_members (members : ℕ) (h : total_people members = total_pouches) : members = 13 :=
by
  sorry

end number_of_team_members_l538_538635


namespace find_tricksters_in_16_questions_l538_538337

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

end find_tricksters_in_16_questions_l538_538337


namespace part1_part2_part3_l538_538594

-- Define the complex number z
def z (m : ℝ) : ℂ :=
  ⟨m^2 - 3*m + 2, m^2 - 1⟩  -- Note: This forms a complex number with real and imaginary parts

-- (1) Proof for z = 0 if and only if m = 1
theorem part1 (m : ℝ) : z m = 0 ↔ m = 1 :=
by sorry

-- (2) Proof for z being a pure imaginary number if and only if m = 2
theorem part2 (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2 :=
by sorry

-- (3) Proof for the point corresponding to z being in the second quadrant if and only if 1 < m < 2
theorem part3 (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 ↔ 1 < m ∧ m < 2 :=
by sorry

end part1_part2_part3_l538_538594


namespace sin_690_eq_neg_half_l538_538400

theorem sin_690_eq_neg_half :
  let rad := Real.pi / 180 in -- Convert degrees to radians
  Real.sin (690 * rad) = -1 / 2 :=
by
  sorry

end sin_690_eq_neg_half_l538_538400


namespace line_intersects_circle_l538_538892

theorem line_intersects_circle (m : ℝ) : 
    let circle_eq := λ (x y : ℝ), x^2 + y^2 + 2 * x - 24 = 0 in
    let line_eq := λ (x y : ℝ), x - m * y + 9 = 0 in
    (∃ (x1 y1 x2 y2 : ℝ), circle_eq x1 y1 ∧ circle_eq x2 y2 ∧ 
                          line_eq x1 y1 ∧ line_eq x2 y2 ∧ 
                          (x1 - x2)^2 + (y1 - y2)^2 = 36) → (m = Real.sqrt 3 ∨ m = -Real.sqrt 3) :=
by sorry

end line_intersects_circle_l538_538892


namespace sum_of_quotient_and_reciprocal_l538_538214

theorem sum_of_quotient_and_reciprocal (x y : ℝ) (h1 : x + y = 45) (h2 : x * y = 500) : 
    (x / y + y / x) = 41 / 20 := 
sorry

end sum_of_quotient_and_reciprocal_l538_538214


namespace eat_five_pounds_in_46_875_min_l538_538170

theorem eat_five_pounds_in_46_875_min
  (fat_rate : ℝ) (thin_rate : ℝ) (combined_rate : ℝ) (total_fruit : ℝ)
  (hf1 : fat_rate = 1 / 15)
  (hf2 : thin_rate = 1 / 25)
  (h_comb : combined_rate = fat_rate + thin_rate)
  (h_fruit : total_fruit = 5) :
  total_fruit / combined_rate = 46.875 :=
by
  sorry

end eat_five_pounds_in_46_875_min_l538_538170


namespace product_of_mixed_numbers_l538_538238

theorem product_of_mixed_numbers :
  let fraction1 := (13 : ℚ) / 6
  let fraction2 := (29 : ℚ) / 9
  (fraction1 * fraction2) = 377 / 54 := 
by
  sorry

end product_of_mixed_numbers_l538_538238


namespace alternating_sum_sequence_l538_538244

theorem alternating_sum_sequence :
  let a := 3
  let d := 3
  let terms := 20
  (Σ i in (Finset.range (terms/2)).map (λ n, -(a * (2 * n + 2 - 1))))
    == -30 := 
by 
  let a := 3
  let d := 3
  let terms := 20
  have h : terms = 20 := rfl
  sorry

end alternating_sum_sequence_l538_538244


namespace total_divisors_7350_l538_538698

def primeFactorization (n : ℕ) : List (ℕ × ℕ) :=
  if n = 7350 then [(2, 1), (3, 1), (5, 2), (7, 2)] else []

def totalDivisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc (p : ℕ × ℕ) => acc * (p.snd + 1)) 1

theorem total_divisors_7350 : totalDivisors (primeFactorization 7350) = 36 :=
by
  sorry

end total_divisors_7350_l538_538698


namespace kite_altitude_l538_538818

-- Definitions of the points and distances
variables (A B C D O K : Type) 
variables (distance : A → B → ℝ)
variables (north_of : A → O → Prop)
variables (west_of : B → O → Prop)
variables (south_of : C → O → Prop)
variables (east_of : D → O → Prop)
variables (above : K → O → Prop)

-- Given distances and relationships
axiom dist_AB : distance A B = 160
axiom len_KA : distance O K = 170
axiom len_KB : distance O K = 140

-- Pythagorean theorem applications and altitude calculation
theorem kite_altitude {a b k: ℝ} (hA : north_of A O) (hB : west_of B O)
  (hC : south_of C O) (hD : east_of D O) (hK : above K O)
  (hOA : distance O A = a) (hOB : distance O B = b) : 
  distance O K = 107 * real.sqrt 1.5 := 
  sorry

end kite_altitude_l538_538818


namespace correct_option_l538_538248

-- Define the conditions
def c1 (a : ℝ) : Prop := (2 * a^2)^3 ≠ 6 * a^6
def c2 (a : ℝ) : Prop := (a^8) / (a^2) ≠ a^4
def c3 (x y : ℝ) : Prop := (4 * x^2 * y) / (-2 * x * y) ≠ -2
def c4 : Prop := Real.sqrt ((-2)^2) = 2

-- The main statement to be proved
theorem correct_option (a x y : ℝ) (h1 : c1 a) (h2 : c2 a) (h3 : c3 x y) (h4 : c4) : c4 :=
by
  apply h4

end correct_option_l538_538248


namespace find_tricksters_l538_538327

structure Inhabitant :=
  (isKnight : Prop)

constants (inhabitants : Fin 65 → Inhabitant)
          (tricksters : Fin 2 → Fin 65)

axiom two_tricksters_unique :
  ∃! (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65))

axiom valid_question :
  ∀ (a : Fin 65) (group : Set (Fin 65)), (inhabitants a).isKnight → 
  (∀ i ∈ group, (inhabitants i).isKnight) ↔ 
  (knight a).isKnight

theorem find_tricksters :
  ∃ (q : ℕ), q ≤ 16 ∧
  ∃ (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65)) :=
sorry

end find_tricksters_l538_538327


namespace total_seeds_eaten_l538_538382

theorem total_seeds_eaten :
  ∃ (first second third : ℕ), 
  first = 78 ∧ 
  second = 53 ∧ 
  third = second + 30 ∧ 
  first + second + third = 214 :=
by
  use 78, 53, 83
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end total_seeds_eaten_l538_538382


namespace min_sqrt_diff_l538_538139

theorem min_sqrt_diff (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ x y : ℕ, x = (p - 1) / 2 ∧ y = (p + 1) / 2 ∧ x ≤ y ∧
    ∀ a b : ℕ, (a ≤ b) → (Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b ≥ 0) → 
      (Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y) ≤ (Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b) := 
by 
  -- Proof to be filled in
  sorry

end min_sqrt_diff_l538_538139


namespace area_of_EFGH_l538_538461

def short_side_length : ℕ := 4
def long_side_length : ℕ := short_side_length * 2
def number_of_rectangles : ℕ := 4
def larger_rectangle_length : ℕ := short_side_length
def larger_rectangle_width : ℕ := number_of_rectangles * long_side_length

theorem area_of_EFGH :
  (larger_rectangle_length * larger_rectangle_width) = 128 := 
  by
    sorry

end area_of_EFGH_l538_538461


namespace find_x2_l538_538694

theorem find_x2 (x1 x2 x3 : ℝ) (h1 : x1 + x2 = 14) (h2 : x1 + x3 = 17) (h3 : x2 + x3 = 33) : x2 = 15 :=
by
  sorry

end find_x2_l538_538694


namespace find_tricksters_l538_538350

theorem find_tricksters (inhabitants : Fin 65 → Prop) (is_knight : Fin 65 → Prop)
    (total_inhabitants : ∀ n, inhabitants n)
    (knights : ∀ n, is_knight n → inhabitants n)
    (tricksters_count : (∑ n, if ¬ is_knight n then 1 else 0) = 2)
    (knights_count : (∑ n, if is_knight n then 1 else 0) = 63)
    (knight_truth : ∀ n, is_knight n → ∀ l : list (Fin 65), (∀ m ∈ l, is_knight m) ↔ true)
    (ask_question : ∀ n, inhabitants n → ∀ l : list (Fin 65), bool) :
  ∃ (find_tricksters_function : (Fin 65 → Prop) → (Fin 65 → bool) → (list (Fin 65))) ,
    (length (find_tricksters_function inhabitants ask_question) ≤ 2) →
    (length (find_tricksters_function inhabitants ask_question) = 2) ∧
    ∀ t ∈ (find_tricksters_function inhabitants ask_question), ¬ is_knight t :=
by sorry

end find_tricksters_l538_538350


namespace correct_statements_l538_538048

-- Define the conditions given in the problem
def f (x : ℝ) : ℝ := if x < 0 then exp x * (x + 1) else -exp (-x) * (x - 1)

theorem correct_statements :
  (∀ x > 0, f x = exp (-x) * (x - 1))
  ∧ (∃ x, f x = 0 ∧ ∃ y ≠ x, f y = 0 ∧ ∃ z ≠ y ∧ z ≠ x, f z = 0)
  ∧ (∀ x, f x < 0 ↔ x ∈ set.Ioo (-∞) (-1) ∨ x ∈ set.Ioo 0 1)
  ∧ (∀ x1 x2 : ℝ, abs (f x1 - f x2) < 2) := sorry

end correct_statements_l538_538048


namespace merchant_marked_price_l538_538730

theorem merchant_marked_price (L P x S : ℝ)
  (h1 : L = 100)
  (h2 : P = 70)
  (h3 : S = 0.8 * x)
  (h4 : 0.8 * x - 70 = 0.3 * (0.8 * x)) :
  x = 125 :=
by
  sorry

end merchant_marked_price_l538_538730


namespace sequence_sum_expression_l538_538128

theorem sequence_sum_expression (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) 
    (h1 : a_n 2 = 4) (h2 : a_n 3 = 15)
    (h3 : ∃ (r : ℝ), ∀ n, a_n n + n = (a_n 2 + 2) * r^(n-2)) :

    S_n = λ n, 3^n - (n^2 + n) / 2 - 1 := 
by 
  sorry

end sequence_sum_expression_l538_538128


namespace mark_vs_chris_walking_time_difference_l538_538165

theorem mark_vs_chris_walking_time_difference :
  ∀ (walking_speed : ℝ) (distance_to_lunch : ℝ) (distance_to_school : ℝ),
    0 < walking_speed →
    0 < distance_to_lunch →
    0 < distance_to_school →
    distance_to_lunch = 3 →
    distance_to_school = 9 →
    walking_speed = 3 →
    (2 = ((2 * distance_to_lunch + distance_to_school) / walking_speed) - (distance_to_school / walking_speed)) :=
by
  intros walking_speed distance_to_lunch distance_to_school hs hd1 hd2 hlunch hschool hspeed
  have htmark : (2 * distance_to_lunch + distance_to_school) / walking_speed = (2 * 3 + 9) / 3, from
    by simp [hlunch, hschool, hspeed]
  have htchris : distance_to_school / walking_speed = 9 / 3, from
    by simp [hschool, hspeed]
  have result : 2 = (5 - 3), from
    by simp
  simp [htmark, htchris, result]
  exact sorry

end mark_vs_chris_walking_time_difference_l538_538165


namespace prop_1_not_prop_2_prop_3_not_prop_4_l538_538633

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) - 2 * sqrt 3 * sin x * cos x

theorem prop_1 (x1 x2 : ℝ) (h : x1 - x2 = π) : f x1 = f x2 :=
sorry

theorem not_prop_2 : ¬(∀ x ∈ Icc (-π/6) (π/3), ∀ y ∈ Icc (-π/6) (π/3), x < y → f x < f y) :=
sorry

theorem prop_3 : ∀ x : ℝ, f (x + π/12) = f (-x + π/12) :=
sorry

theorem not_prop_4 : ¬(∀ x : ℝ, f (x - 5*π/12) = 2 * sin (2 * x)) :=
sorry

end prop_1_not_prop_2_prop_3_not_prop_4_l538_538633


namespace number_of_satisfying_conditions_eq_420_l538_538005

noncomputable def number_satisfying_condition (n : ℕ) : ℕ :=
  if h1 : 1 ≤ n ∧ n < 3^8 then
    ∑ k in finset.range (n / 3 + 1), if (nat.factorial n) % ((nat.factorial (n - 3 * k)) * (nat.factorial k) * 3^(k + 1)) = 0 then 0 else 1
  else 0

theorem number_of_satisfying_conditions_eq_420 :
  (finset.range (3^8)).filter (λ n, number_satisfying_condition n > 0).card = 420 := 
  sorry

end number_of_satisfying_conditions_eq_420_l538_538005


namespace laptop_weight_difference_is_3_67_l538_538137

noncomputable def karen_tote_weight : ℝ := 8
noncomputable def kevin_empty_briefcase_weight : ℝ := karen_tote_weight / 2
noncomputable def umbrella_weight : ℝ := kevin_empty_briefcase_weight / 2
noncomputable def briefcase_full_weight_rainy_day : ℝ := 2 * karen_tote_weight
noncomputable def work_papers_weight : ℝ := (briefcase_full_weight_rainy_day - umbrella_weight) / 6
noncomputable def laptop_weight : ℝ := briefcase_full_weight_rainy_day - umbrella_weight - work_papers_weight
noncomputable def weight_difference : ℝ := laptop_weight - karen_tote_weight

theorem laptop_weight_difference_is_3_67 : weight_difference = 3.67 := by
  sorry

end laptop_weight_difference_is_3_67_l538_538137


namespace total_trophies_correct_l538_538935

-- Define the current number of Michael's trophies
def michael_current_trophies : ℕ := 30

-- Define the number of trophies Michael will have in three years
def michael_trophies_in_three_years : ℕ := michael_current_trophies + 100

-- Define the number of trophies Jack will have in three years
def jack_trophies_in_three_years : ℕ := 10 * michael_current_trophies

-- Define the total number of trophies Jack and Michael will have after three years
def total_trophies_in_three_years : ℕ := michael_trophies_in_three_years + jack_trophies_in_three_years

-- Prove that the total number of trophies after three years is 430
theorem total_trophies_correct : total_trophies_in_three_years = 430 :=
by
  sorry -- proof is omitted

end total_trophies_correct_l538_538935


namespace area_outside_triangle_l538_538945

-- Define the problem conditions
variables {ABC : Triangle}
variables (angleBAC : ABC.angle A B C = π / 2) (lengthAB : ABC.side A B = 8)
variables (P Q : Point) (circle : Circle)
variables (tangentP : circle.isTangent P ABC.side A B)
variables (tangentQ : circle.isTangent Q ABC.side A C)
variables (P' Q' : Point)
variables (diametricP' : P'.isDiametricOpposite P)
variables (diametricQ' : Q'.isDiametricOpposite Q)
variables (lieBC : P'.liesOn ABC.side B C ∧ Q'.liesOn ABC.side B C)

-- Define the goal
theorem area_outside_triangle : 
  ∃ (circle : Circle), ∃ (r : ℝ), 
  circle.radius = r ∧ r = 2 ∧ 
  (circle.area_outside_triangle ABC = π - 2) := 
by
  sorry

end area_outside_triangle_l538_538945


namespace sin_690_eq_neg_half_l538_538404

theorem sin_690_eq_neg_half :
  let rad := Real.pi / 180 in -- Convert degrees to radians
  Real.sin (690 * rad) = -1 / 2 :=
by
  sorry

end sin_690_eq_neg_half_l538_538404


namespace f_at_10_l538_538019

-- Define the functional equation condition
def f (x : ℝ) : ℝ := 2 * f (1 / x) * Real.log x + 1

-- State the theorem to be proved
theorem f_at_10 : f 10 = 3 / 5 := sorry

end f_at_10_l538_538019


namespace complex_multiplication_l538_538954

theorem complex_multiplication :
  let i : ℂ := complex.I
  (2 + 3 * i) * (3 - 2 * i) = 12 + 5 * i :=
by
  sorry

end complex_multiplication_l538_538954


namespace cut_out_square_possible_l538_538462

/-- 
Formalization of cutting out eight \(2 \times 1\) rectangles from an \(8 \times 8\) 
checkered board, and checking if it is always possible to cut out a \(2 \times 2\) square
from the remaining part of the board.
-/
theorem cut_out_square_possible :
  ∀ (board : ℕ) (rectangles : ℕ), (board = 64) ∧ (rectangles = 8) → (4 ∣ board) →
  ∃ (remaining_squares : ℕ), (remaining_squares = 48) ∧ 
  (∃ (square_size : ℕ), (square_size = 4) ∧ (remaining_squares ≥ square_size)) :=
by {
  sorry
}

end cut_out_square_possible_l538_538462


namespace tian_ji_win_prob_normal_tian_ji_win_prob_strategy_l538_538680

-- Definitions based on conditions
variables (A a B b C c : Horse)
variable (races : List (Horse × Horse))

-- Condition: Relative strength of the six horses
axiom horse_strength : A > a ∧ a > B ∧ B > b ∧ b > C ∧ C > c

-- Condition: The one who wins two or more races is the winner
def wins_2_or_more (r : List (Horse × Horse)) : Bool :=
  let tians_horses := [a, b, c]
  let king_horses := [A, B, C]

  let tians_wins := r.countp (λ matchup, matchup.1 ∈ tians_horses ∧ matchup.2 ∈ king_horses ∧ 
                                      horse_strength ∧ (matchup.1 > matchup.2))

  tians_wins ≥ 2

-- Problem 1: Probability of Tian Ji winning under normal circumstances
theorem tian_ji_win_prob_normal : calculate_probability (wins_2_or_more (A,a) (B,b) (C,c) ∨ 
                                                        wins_2_or_more (A,a) (B,c) (C,b) ∨ 
                                                        wins_2_or_more (A,b) (B,c) (C,a) ∨ 
                                                        wins_2_or_more (A,b) (B,a) (C,c) ∨ 
                                                        wins_2_or_more (A,c) (B,a) (C,b) ∨ 
                                                        wins_2_or_more (A,c) (B,b) (C,a)) = 1 / 6 := 
by
  sorry

-- Problem 2: Probability of Tian Ji winning when using the strategy after obtaining intelligence
theorem tian_ji_win_prob_strategy : calculate_probability (wins_2_or_more (c,A) (a,B) (b,C) ∨ 
                                                           wins_2_or_more (c,A) (b,B) (a,C)) = 1 / 2 := 
by
  sorry

end tian_ji_win_prob_normal_tian_ji_win_prob_strategy_l538_538680


namespace count_irrational_numbers_l538_538659

noncomputable def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def number_list := [22 / 7, 3.1415, Real.sqrt 5, Real.pi / 2, -0.3, 
  Real.of_real (.11.sum (List.repeat 0 0 ∩ List.repeat 3 0)).natAbs / 
  .1.sum (List.repeat 0 0 ∩ List.repeat 3 1)).nat 

theorem count_irrational_numbers : (number_list.filter is_irrational).length = 3 := 
by 
  sorry

end count_irrational_numbers_l538_538659


namespace necessary_and_sufficient_condition_l538_538558

-- Definitions for sides opposite angles A, B, and C in a triangle.
variables {A B C : Real} {a b c : Real}

-- Condition p: sides a, b related to angles A, B via cosine
def condition_p (a b : Real) (A B : Real) : Prop := a / Real.cos A = b / Real.cos B

-- Condition q: sides a and b are equal
def condition_q (a b : Real) : Prop := a = b

theorem necessary_and_sufficient_condition (h1 : condition_p a b A B) : condition_q a b ↔ condition_p a b A B :=
by
  sorry

end necessary_and_sufficient_condition_l538_538558


namespace train_cross_time_l538_538752

-- Definitions of the conditions provided
def speed_kmph : ℝ := 48        -- Speed in kilometers per hour
def length_train : ℝ := 120     -- Length of the train in meters

def km_to_m : ℝ := 1000         -- Conversion factor from kilometers to meters
def hr_to_s : ℝ := 3600         -- Conversion factor from hours to seconds

-- Derived definitions based on the conditions
def speed_mps : ℝ := (speed_kmph * km_to_m) / hr_to_s  -- Speed in meters per second
def time_to_cross : ℝ := length_train / speed_mps      -- Time to cross the pole in seconds

-- The theorem to prove the equivalence
theorem train_cross_time : time_to_cross = 9 := by
  -- Proof is omitted as specified
  sorry

end train_cross_time_l538_538752


namespace number_of_valid_a_l538_538897

theorem number_of_valid_a : 
  (∃(a : ℤ), ∃(x : ℕ), (frac_eq : (x + a) / (x - 2) + 2x / (2 - x) = 1) ∧ 
  (ineq_sys : exactIntegerSolutions (λ y, (y + 1) / 5 ≥ y / 2 - 1 ∧ y + a < 11y - 3) 4)) ↔ 
  ((a = -2) ∨ (a = 0) ∨ (a = 4) ∨ (a = 6)) := 
begin
  sorry
end

end number_of_valid_a_l538_538897


namespace identify_tricksters_in_30_or_less_questions_l538_538359

-- Define the problem parameters
def inhabitants : Type := Fin 65

def is_knight (inhabitant : inhabitants) : Prop := sorry
def is_trickster (inhabitant : inhabitants) : Prop := sorry

-- Define the properties
axiom knight_truthful : ∀ (x : inhabitants), is_knight x → (forall y : inhabitants, True ↔ (is_knight y = x is_knight y))
axiom trickster_mixed : ∀ (x : inhabitants), is_trickster x → ((∀ y : inhabitants, True) ∨ (∃ y : inhabitants, y ∉ (is_knight y)))

-- Problem statement
theorem identify_tricksters_in_30_or_less_questions
  (inhabitants : Type)
  (n_tricksters : ℕ := 2) -- 2 tricksters
  (total_inhabitants : ℕ := 65) -- 65 total inhabitants
  (questions_limit : ℕ := 30) -- limit of 30 questions
  (knights : inhabitants → Prop)
  (tricksters : inhabitants → Prop) :
    ∃ (solution_exists : ∀ (is_trickster : inhabitants → Prop), ∃ k : inhabitants, (knights k) ∧ (is_trickster k)) 
    (possible_to_find_tricksters : ∀ (is_knight : inhabitants → Prop) (is_trickster : inhabitants → Prop), 
    ∃ (questions_used ≤ questions_limit), ∀ (xs : set inhabitants), questions_used ≤ 30 ∧ 
    (∃ trickster1 trickster2 : inhabitants, (tricksters trickster1 ∧ tricksters trickster2 ∧ trickster1 ≠ trickster2))) :=
sorry

end identify_tricksters_in_30_or_less_questions_l538_538359


namespace principal_amount_correct_l538_538378

noncomputable def initial_amount (A : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (A * 100) / (R * T + 100)

theorem principal_amount_correct : initial_amount 950 9.230769230769232 5 = 650 := by
  sorry

end principal_amount_correct_l538_538378


namespace domain_tan_function_l538_538657

theorem domain_tan_function :
  ∀ x : ℝ, (∃ k : ℤ, x ≠ k * real.pi + 3 * real.pi / 4) ↔
           (∃ k : ℤ, x ≠ k * real.pi + 3 * real.pi / 4) :=
by
  sorry

end domain_tan_function_l538_538657


namespace number_of_integers_satisfying_conditions_l538_538900

-- Define the problem constraints and proof statement
open Real

noncomputable def satisfies_equation (a : ℝ) : Prop :=
  ∃ x : ℕ, (x ≠ 2) ∧ ((x + a) / (x - 2) + (2 * x) / (2 - x) = 1)

noncomputable def satisfies_inequalities (a : ℝ) : Prop :=
  let solutions := { y : ℤ | (y ≥ 2 + 10 * y - 3) ∧ (y ≤ 14) } in
  (solutions.size = 4)

theorem number_of_integers_satisfying_conditions :
  (finset.filter satisfies_equation (finset.range 11)).size = 4 :=
begin
  sorry -- Proof goes here
end

end number_of_integers_satisfying_conditions_l538_538900


namespace perimeter_of_triangle_XYZ_l538_538300

noncomputable theory

structure Point (ℝ : Type*) :=
  (x : ℝ) (y : ℝ) (z : ℝ)

structure Prism (ℝ : Type*) :=
  (base : list (Point ℝ))
  (height : ℝ)
  (regular_hex_side_length : ℝ)

def midpoint (A B : Point ℝ) : Point ℝ :=
  { x := (A.x + B.x) / 2,
    y := (A.y + B.y) / 2,
    z := (A.z + B.z) / 2 }

def calculate_distance (A B : Point ℝ) : ℝ :=
  real.sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2 + (A.z - B.z) ^ 2)

def perimeter_of_triangle (A B C : Point ℝ) : ℝ :=
  calculate_distance A B + calculate_distance B C + calculate_distance C A

variables (PQ QR RS : Point ℝ)
variables (PX QY RZ X Y Z : Point ℝ)

definition midpoint_property : Prop :=
  X = midpoint PQ QR ∧ Y = midpoint QR RS ∧ Z = midpoint RS QR

definition solid_prism_property (PQR : list (Point ℝ)) (height : ℝ) (side_length : ℝ) : Prop :=
  ∃ (prism : Prism ℝ), prism.base = PQR ∧ prism.height = height ∧ prism.regular_hex_side_length = side_length

theorem perimeter_of_triangle_XYZ 
  (PQR : list (Point ℝ)) (height : ℝ) (side_length : ℝ) 
  (h_prism : solid_prism_property PQR height side_length) 
  (h_midpoint : midpoint_property PQ QR RS PX QY RZ X Y Z) :
  perimeter_of_triangle X Y Z = 15 := by sorry

end perimeter_of_triangle_XYZ_l538_538300


namespace find_tricksters_l538_538352

theorem find_tricksters (inhabitants : Fin 65 → Prop) (is_knight : Fin 65 → Prop)
    (total_inhabitants : ∀ n, inhabitants n)
    (knights : ∀ n, is_knight n → inhabitants n)
    (tricksters_count : (∑ n, if ¬ is_knight n then 1 else 0) = 2)
    (knights_count : (∑ n, if is_knight n then 1 else 0) = 63)
    (knight_truth : ∀ n, is_knight n → ∀ l : list (Fin 65), (∀ m ∈ l, is_knight m) ↔ true)
    (ask_question : ∀ n, inhabitants n → ∀ l : list (Fin 65), bool) :
  ∃ (find_tricksters_function : (Fin 65 → Prop) → (Fin 65 → bool) → (list (Fin 65))) ,
    (length (find_tricksters_function inhabitants ask_question) ≤ 2) →
    (length (find_tricksters_function inhabitants ask_question) = 2) ∧
    ∀ t ∈ (find_tricksters_function inhabitants ask_question), ¬ is_knight t :=
by sorry

end find_tricksters_l538_538352


namespace height_difference_l538_538689

variables (S K T H : ℕ)  -- S stands for Soohyun, K for Kiyoon, T for Taehun, H for Hwajun

def kiyoon_height (S : ℕ) : ℕ := S + 207

def hwajun_height (S : ℕ) : ℕ := S + 207 + 839

theorem height_difference (S : ℕ) : hwajun_height S = S + 1046 :=
by {
  unfold hwajun_height kiyoon_height,
  sorry
}

end height_difference_l538_538689


namespace other_student_questions_l538_538972

theorem other_student_questions (m k o : ℕ) (h1 : m = k - 3) (h2 : k = o + 8) (h3 : m = 40) : o = 35 :=
by
  -- proof goes here
  sorry

end other_student_questions_l538_538972


namespace graph_transformation_l538_538850

def f (x : ℝ) (ϕ : ℝ) : ℝ := 2 * sin (π + x) * sin (x + π / 3 + ϕ)
def g (x : ℝ) (ϕ : ℝ) : ℝ := cos (2 * x - ϕ)

theorem graph_transformation (ϕ : ℝ) (hϕ : 0 < ϕ ∧ ϕ < π) :
  (∀ x : ℝ, f (x - π / 3) ϕ = g x ϕ) :=
by
  sorry

end graph_transformation_l538_538850


namespace distance_from_origin_l538_538197

theorem distance_from_origin (A : ℝ) (h : |A - 0| = 4) : A = 4 ∨ A = -4 :=
by {
  sorry
}

end distance_from_origin_l538_538197


namespace find_radius_l538_538509

-- Define the conditions
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y = 25
def line2 (x y : ℝ) : Prop := 117 * x - 44 * y = 175
def point (x y : ℝ) : Prop := True  -- Just to define point A (x, y)

def projection_b (p q : ℝ) : Prop := ∃ x y : ℝ, line1 x y ∧ (x - p)^2 + (y - q)^2 < eps
def projection_c (p q : ℝ) : Prop := ∃ x y : ℝ, line2 x y ∧ (x - p)^2 + (y - q)^2 < eps

def distance_to_line1 (p q : ℝ) : ℝ := abs (3 * p + 4 * q - 25) / 5
def distance_to_line2 (p q : ℝ) : ℝ := abs (117 * p - 44 * q - 175) / 125

def area (p q : ℝ) : ℝ := (1 / 2) * distance_to_line1 p q * distance_to_line2 p q * (1 / 5)
def locus (x y : ℝ) : Prop := ∃ p q : ℝ, point p q ∧ abs (351 * (p - 3)^2 - 176 * (q - 4)^2 + 336 * (p - 3) * (q - 4)) = 3600 ∧ (x = p ∧ y = q)

def circle (x y r : ℝ) : Prop := (x - 39/5)^2 + (y - 27/5)^2 = r^2

-- Main proof problem
theorem find_radius (r : ℝ) : (∃ x y : ℝ, locus x y ∧ circle x y r) ∧ (set.finite {p : ℝ × ℝ | locus p.1 p.2 ∧ circle p.1 p.2 r} ∧ set.card {p : ℝ × ℝ | locus p.1 p.2 ∧ circle p.1 p.2 r} = 7) -> r = 8 :=
by
  sorry

end find_radius_l538_538509


namespace slices_leftover_l538_538390

def total_initial_slices : ℕ := 12 * 2
def bob_slices : ℕ := 12 / 2
def tom_slices : ℕ := 12 / 3
def sally_slices : ℕ := 12 / 6
def jerry_slices : ℕ := 12 / 4
def total_slices_eaten : ℕ := bob_slices + tom_slices + sally_slices + jerry_slices

theorem slices_leftover : total_initial_slices - total_slices_eaten = 9 := by
  sorry

end slices_leftover_l538_538390


namespace sin_690_degree_l538_538421

theorem sin_690_degree : sin (690 : ℝ) * (Real.pi / 180) = -(1 / 2) := by
  sorry

end sin_690_degree_l538_538421


namespace solve_for_n_l538_538254

theorem solve_for_n : ∃ n : ℕ, 4 * 2^(2 * n) = 2^36 ∧ n = 17 :=
by {
  -- Construct the proof content to match the conditions and questions
  use 17,
  split,
  case_left {
    -- Verification of condition 4 * 2^(2 * n) = 2^36
    calc 4 * 2^(2 * 17) = (2^2) * 2^(2 * 17) : by rw [pow_two]
                   ... = 2^(2 + 2 * 17) : by rw [pow_add]
                   ... = 2^36 : by norm_num,
  },
  case_right {
    -- n = 17 given as conclusion
    refl,
  }
}

end solve_for_n_l538_538254


namespace benny_spending_l538_538766

theorem benny_spending (initial_amount : ℕ) (leftover_amount : ℕ) (spent_amount : ℕ) 
  (h_initial : initial_amount = 67)
  (h_leftover : leftover_amount = 33) : 
  spent_amount = initial_amount - leftover_amount := 
begin 
  rw [h_initial, h_leftover],
  exact rfl,
end

#eval benny_spending 67 33 34 (rfl) (rfl) -- This will help us check if the statement holds true.

end benny_spending_l538_538766


namespace simplify_expression_l538_538640

theorem simplify_expression (a b c : ℝ) (ha : a = 7.4) (hb : b = 5 / 37) :
  1.6 * ((1 / a + 1 / b - 2 * c / (a * b)) * (a + b + 2 * c)) / 
  ((1 / a^2 + 1 / b^2 + 2 / (a * b) - 4 * c^2 / (a^2 * b^2))) = 1.6 :=
by 
  rw [ha, hb] 
  sorry

end simplify_expression_l538_538640


namespace clock_angle_8_20_l538_538236

-- Define the time of interest
def time_of_interest : ℕ × ℕ := (8, 20)

-- Define the degrees per hour and per minute
def degrees_per_hour : ℝ := 30 -- 360 / 12
def degrees_per_minute : ℝ := 6 -- 360 / 60

-- Define the position of the minute hand
def minute_hand_position (m : ℕ) : ℝ := (m / 60) * 360

-- Define the position of the hour hand
def hour_hand_position (h m : ℕ) : ℝ := ((h + (m / 60.0)) / 12) * 360

-- Define the absolute difference between the hour and minute hands positions
def angle_difference (h m : ℕ) : ℝ := 
  let hour_pos := hour_hand_position h m in
  let minute_pos := minute_hand_position m in
  abs (hour_pos - minute_pos)

-- The final theorem to be proven
theorem clock_angle_8_20 :
  angle_difference 8 20 = 130 :=
  by
  -- skip the proof for now
  sorry

end clock_angle_8_20_l538_538236


namespace find_f_neg_m_l538_538958

def f (x : ℝ) : ℝ := 
  Real.log (Real.sqrt (1 + Real.pi^2 * x^2) - Real.pi * x) + Real.pi

def m : ℝ := sorry

theorem find_f_neg_m (h : f m = 3) : f (-m) = 2 * Real.pi - 3 :=
by
  sorry

end find_f_neg_m_l538_538958


namespace ratio_division_of_triangle_l538_538282

theorem ratio_division_of_triangle (A B C M N : Point)
  (h1 : is_triangle A B C)
  (h2 : parallel MN BC)
  (h3 : area_ratio (triangle M A N ) (triangle M N B) = 2 / 1) :
  length_ratio (segment M A) (segment M B) = (Real.sqrt 6 + 2) / 1 :=
by 
  sorry

end ratio_division_of_triangle_l538_538282


namespace solution_part1_solution_part2_l538_538155

open Complex Real

noncomputable def complex_value_problem (z : ℂ) (ω : ℝ) :=
  (-1 < ω ∧ ω < 2) → ∃ x y : ℝ, (z = x + y * I ∧ y ≠ 0 ∧
  (ω = x + (1 / (x^2 + y^2)) + (y - y / (x^2 + y^2)) * I) ∧
  (abs z = 1) ∧ (-1 < 2 * x < 2))

theorem solution_part1 (z : ℂ) (ω : ℝ) (hω : -1 < ω ∧ ω < 2) :
  ∃ x : ℝ, abs z = 1 ∧ -1 / 2 < x ∧ x < 1 :=
sorry

theorem solution_part2 (z : ℂ) :
  ∀ x y : ℝ, z = x + y * I ∧ y ≠ 0 ∧ abs z = 1 → 
  ∃ (μ : ℂ), μ = (1 - z) / (1 + z) ∧ (μ.im ≠ 0 ∧ μ.re = 0) :=
sorry

end solution_part1_solution_part2_l538_538155


namespace four_digit_number_count_l538_538759

theorem four_digit_number_count (A : ℕ → ℕ → ℕ)
  (odd_digits even_digits : Finset ℕ)
  (odds : ∀ x ∈ odd_digits, x % 2 = 1)
  (evens : ∀ x ∈ even_digits, x % 2 = 0) :
  odd_digits = {1, 3, 5, 7, 9} ∧ 
  even_digits = {2, 4, 6, 8} →
  A 5 2 * A 7 2 = 840 :=
by
  intros h1
  sorry

end four_digit_number_count_l538_538759


namespace trig_identity_tangent_l538_538518

variable {θ : ℝ}

theorem trig_identity_tangent (h : Real.tan θ = 2) : 
  (Real.sin θ * (Real.cos θ * Real.cos θ - Real.sin θ * Real.sin θ)) / (Real.cos θ - Real.sin θ) = 6 / 5 := 
sorry

end trig_identity_tangent_l538_538518


namespace triangle_area_l538_538819

/-- Definition of the vertices of the triangle -/
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (2, 0)

/-- Calculation of the area of triangle ABC using the coordinates method -/
theorem triangle_area : 
  let area := (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))
  in area = 2 := 
by {
  let area := (1 / 2) * abs (0 * (2 - 0) + 2 * (0 - 0) + 2 * (0 - 2)),
  have h : area = (1 / 2) * abs (-4) := rfl,
  rw [abs_neg],
  norm_num at h,
  exact h
}

end triangle_area_l538_538819


namespace trees_left_after_typhoon_and_growth_l538_538079

-- Conditions
def initial_trees : ℕ := 9
def trees_died_in_typhoon : ℕ := 4
def new_trees : ℕ := 5

-- Question (Proof Problem)
theorem trees_left_after_typhoon_and_growth : 
  initial_trees - trees_died_in_typhoon + new_trees = 10 := 
by
  sorry

end trees_left_after_typhoon_and_growth_l538_538079


namespace train_cross_pole_l538_538750

noncomputable def time_to_cross_pole (speed_kmh : ℕ) (length_m : ℕ) : ℕ :=
  lengths_m / (speed_kmh * 1000 / 3600)

theorem train_cross_pole (speed_kmh length_m : ℕ) :
  speed_kmh = 48 →
  length_m = 120 →
  time_to_cross_pole speed_kmh length_m = 9 :=
by
  intros h_speed h_len
  rw [h_speed, h_len]
  have h_speed_ms : (48 : ℚ) = 48 * 1000 / 3600 by norm_num
  have h_time_calc : 120 / (48 * 1000 / 3600) = 9 by norm_num
  exact h_time_calc

end train_cross_pole_l538_538750


namespace range_of_a_l538_538496

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a + 6) * x + 1

theorem range_of_a (a : ℝ) :
  (∃ x_max x_min, f a x_max = (some value) ∧ f a x_min = (some value)) → (a < -3 ∨ a > 6) :=
by
  sorry

end range_of_a_l538_538496


namespace auspicious_dragon_cards_count_l538_538907

open Finset

theorem auspicious_dragon_cards_count : 
  let digits := Finset.range 10 in
  let combinations := digits.powerset.filter (λ s, s.card = 4) in
  let ascending_orders := combinations.count (λ s, s.to_list = s.to_list.sorted) in
  ascending_orders = 210 :=
by sorry

end auspicious_dragon_cards_count_l538_538907


namespace hexagon_triangle_area_percentage_l538_538280

theorem hexagon_triangle_area_percentage (s : ℝ) (hs : s > 0) : 
  let hex_area := s^2 + (sqrt 3 / 4) * s^2 in
  let tri_area := (sqrt 3 / 4) * s^2 in
  let fraction := tri_area / hex_area in
  fraction * 100 ≈ 20.77 := 
by
  sorry

end hexagon_triangle_area_percentage_l538_538280


namespace bug_visibility_proof_l538_538377

noncomputable theory

def distance (a b : ℝ) : ℝ := abs (a - b)

def pentagon_side_length : ℝ := 20
def speed_A : ℝ := 5
def speed_B : ℝ := 4
def initial_position_A : ℝ := 0
def initial_position_B : ℝ := 20

def visible_time : ℝ := 6

def can_bug_B_see_bug_A (side_length speed_A speed_B init_pos_A init_pos_B : ℝ) : ℝ :=
  let duration_A := side_length / speed_A
  let distance_covered_B := duration_A * speed_B
  if distance_covered_B <= side_length then
    duration_A
  else
    sorry -- complete calculation would go here

theorem bug_visibility_proof :
  can_bug_B_see_bug_A pentagon_side_length speed_A speed_B initial_position_A initial_position_B = visible_time := sorry

end bug_visibility_proof_l538_538377


namespace distinct_values_of_expr_l538_538797

theorem distinct_values_of_expr : 
  let a := 3^(3^(3^3));
  let b := 3^((3^3)^3);
  let c := ((3^3)^3)^3;
  let d := (3^(3^3))^3;
  let e := (3^3)^(3^3);
  (a ≠ b) ∧ (c ≠ b) ∧ (d ≠ b) ∧ (d ≠ a) ∧ (e ≠ a) ∧ (e ≠ b) ∧ (e ≠ d) := sorry

end distinct_values_of_expr_l538_538797


namespace soda_selected_individuals_proof_l538_538533

-- Definitions based on the conditions in the problem
variables (total_people : ℕ) (soda_angle : ℚ) (total_degrees : ℕ)
#check (195: ℚ)

-- Definition of the number of individuals who selected "Soda"
def individuals_selected_soda (total_people : ℕ) (soda_angle : ℚ) (total_degrees : ℕ) : ℕ :=
  let fraction := soda_angle / total_degrees
  let raw_count := total_people * fraction
  raw_count.toNat

-- Lean 4 statement for the proof problem
theorem soda_selected_individuals_proof : 
  individuals_selected_soda 780 195 360 = 423 :=
by
  sorry

end soda_selected_individuals_proof_l538_538533


namespace range_increases_after_adding_new_score_l538_538562

def initial_scores : List ℕ := [38, 42, 45, 45, 49, 49, 52, 55]
def new_score : ℕ := 35

theorem range_increases_after_adding_new_score :
  let updated_scores := new_score :: initial_scores
  let initial_range := initial_scores.maximum! - initial_scores.minimum!
  let updated_range := updated_scores.maximum! - updated_scores.minimum!
  updated_range > initial_range := by
  sorry

end range_increases_after_adding_new_score_l538_538562


namespace arithmetic_sequence_sum_l538_538581

variable {a_n : ℕ → ℕ} -- the arithmetic sequence

-- Define condition
def condition (a : ℕ → ℕ) : Prop :=
  a 1 + a 5 + a 9 = 18

-- The sum of the first n terms of arithmetic sequence is S_n
def S (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  (n * (a 1 + a n)) / 2

-- The goal is to prove that S 9 = 54
theorem arithmetic_sequence_sum (h : condition a_n) : S 9 a_n = 54 :=
sorry

end arithmetic_sequence_sum_l538_538581


namespace melissa_bananas_now_l538_538968

-- Define the conditions
def initial_bananas : ℕ := 88
def bananas_shared : ℕ := 4
def bananas_left_after_share (initial_bananas bananas_shared : ℕ) : ℕ := initial_bananas - bananas_shared
def bananas_bought (bananas_left : ℕ) : ℕ := 3 * bananas_left

-- Prove that the total number of bananas Melissa has now is 336
theorem melissa_bananas_now :
  let initial := initial_bananas,
      shared := bananas_shared,
      left := bananas_left_after_share initial shared,
      bought := bananas_bought left in
  left + bought = 336 :=
by
  sorry

end melissa_bananas_now_l538_538968


namespace proper_subsets_of_set_card_seven_l538_538205

theorem proper_subsets_of_set_card_seven : 
  ∃ (M : Set ℕ), M = {2, 4, 6} ∧ (M.card - 1)^2 = 7 := 
by {
  sorry
}

end proper_subsets_of_set_card_seven_l538_538205


namespace slices_leftover_is_9_l538_538388

-- Conditions and definitions
def total_pizzas : ℕ := 2
def slices_per_pizza : ℕ := 12
def bob_ate : ℕ := slices_per_pizza / 2
def tom_ate : ℕ := slices_per_pizza / 3
def sally_ate : ℕ := slices_per_pizza / 6
def jerry_ate : ℕ := slices_per_pizza / 4

-- Calculate total slices eaten and left over
def total_slices_eaten : ℕ := bob_ate + tom_ate + sally_ate + jerry_ate
def total_slices_available : ℕ := total_pizzas * slices_per_pizza
def slices_leftover : ℕ := total_slices_available - total_slices_eaten

-- Theorem to prove the number of slices left over
theorem slices_leftover_is_9 : slices_leftover = 9 := by
  -- Proof: omitted, add relevant steps here
  sorry

end slices_leftover_is_9_l538_538388


namespace exists_i_with_a_i_eq_one_l538_538697

def sequence (d : ℕ) : ℕ → ℕ 
| 0       := 1
| (n + 1) := if sequence d n % 2 = 0 then (sequence d n) / 2 else (sequence d n) + d

theorem exists_i_with_a_i_eq_one (d : ℕ) (h : d % 2 = 1) : ∃ i : ℕ, 0 < i ∧ sequence d i = 1 :=
by
  -- Proof goes here
  sorry

end exists_i_with_a_i_eq_one_l538_538697


namespace frac_nonneg_iff_pos_l538_538671

theorem frac_nonneg_iff_pos (x : ℝ) : (2 / x ≥ 0) ↔ (x > 0) :=
by sorry

end frac_nonneg_iff_pos_l538_538671


namespace find_tricksters_within_30_questions_l538_538315

/-- 
Given 65 inhabitants in a village where:
- Two inhabitants are tricksters and the rest are knights.
- Knights always tell the truth.
- Tricksters can either tell the truth or lie.
- One can show any inhabitant a list of some group of inhabitants (which can consist of one person)
  and ask if all of them are knights.

Prove that it is possible to find both tricksters with no more than 30 questions.
-/
theorem find_tricksters_within_30_questions :
  ∃ (ask_knights : (inhabitants : fin 65) → list (fin 65) → Prop),
  (∀ (i j : fin 65), i ≠ j → ask_knights i [j] = true → (inhabitants[j] = knight) ∨ (inhabitants[j] = trickster))
  ∧ ∀ (inhabitants : fin 65),
  (∃ S : finset (fin 65), S.card = 2 ∧ 
  (∀ i, i ∈ S → inhabitants[i] = trickster) ∧ 
  by asking no more than 30 questions,
  you can identify both tricksters.

end find_tricksters_within_30_questions_l538_538315


namespace binomial_distribution_prob_l538_538837

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem binomial_distribution_prob :
  binomial_probability 3 2 (1 / 3) = 2 / 9 :=
by
  sorry

end binomial_distribution_prob_l538_538837


namespace tan_585_eq_one_l538_538780

theorem tan_585_eq_one : Real.tan (585 * Real.pi / 180) = 1 :=
by
  sorry

end tan_585_eq_one_l538_538780


namespace AE_eq_CD_l538_538919

open EuclideanGeometry

-- Definitions of points and key conditions
variables (A B C D E X : Point)
variable (ABCDE : ConvexPentagon A B C D E)
variable (Intersection_AD_CE_X : Intersection (Line_through A D) (Line_through C E) X)
variable (ABCD_parallelogram : Parallelogram A B C D)
variable (BD_CX : Segment_length (Segment.mk B D) = Segment_length (Segment.mk C X))
variable (BE_AX : Segment_length (Segment.mk B E) = Segment_length (Segment.mk A X))

-- Theorem statement
theorem AE_eq_CD : Segment_length (Segment.mk A E) = Segment_length (Segment.mk C D) :=
by
  sorry

end AE_eq_CD_l538_538919


namespace water_in_tank_at_end_of_fourth_hour_l538_538511

-- Define the initial amount of water in the tank
def initial_water : ℕ := 80

-- Define the loss rate and add rate for each hour
def loss_rate : ℕ → ℕ
| 1 := 1
| 2 := 2
| 3 := 4
| 4 := 8
| _ := 0

def add_rate : ℕ → ℕ
| 1 := 0
| 2 := 2
| 3 := 4
| 4 := 8
| _ := 0

-- Define the function to calculate the water left at the end of each hour
def water_left (n : ℕ) : ℕ :=
  ((initial_water - loss_rate 1) + add_rate 1 - loss_rate 2 + add_rate 2 - loss_rate 3 + add_rate 3 - loss_rate 4 + add_rate 4)

theorem water_in_tank_at_end_of_fourth_hour : water_left 4 = 79 :=
by
  -- Starting with 80 gallons
  let initial := 80
  -- After hour 1: 80 - 1 = 79 gallons
  let hour1 := initial - 1
  -- After hour 2: 79 + 2 - 2 = 79 gallons
  let hour2 := hour1 + 2 - 2
  -- After hour 3: 79 + 4 - 4 = 79 gallons
  let hour3 := hour2 + 4 - 4
  -- After hour 4: 79 + 8 - 8 = 79 gallons
  let hour4 := hour3 + 8 - 8
  -- Prove the final amount
  show hour4 = 79, from sorry

end water_in_tank_at_end_of_fourth_hour_l538_538511


namespace chess_tournament_boys_l538_538111

noncomputable def num_boys_in_tournament (n k : ℕ) : Prop :=
  (6 + k * n = (n + 2) * (n + 1) / 2) ∧ (n > 2)

theorem chess_tournament_boys :
  ∃ (n : ℕ), num_boys_in_tournament n (if n = 5 then 3 else if n = 10 then 6 else 0) ∧ (n = 5 ∨ n = 10) :=
by
  sorry

end chess_tournament_boys_l538_538111


namespace log_ratio_l538_538888

theorem log_ratio (a b : ℝ) (h₁ : a = log 4 80) (h₂ : b = log 2 10) : a = b / 2 :=
by
  sorry

end log_ratio_l538_538888


namespace abs_gt_not_implies_gt_l538_538675

noncomputable def abs_gt_implies_gt (a b : ℝ) : Prop :=
  |a| > |b| → a > b

theorem abs_gt_not_implies_gt (a b : ℝ) :
  ¬ abs_gt_implies_gt a b :=
sorry

end abs_gt_not_implies_gt_l538_538675


namespace least_possible_mean_YZ_l538_538267

noncomputable def mean_combined_weight (X_n Y_n Z_n X_mean Y_mean XY_mean XZ_mean : ℕ) (m_total : ℕ) : ℕ := 
  (Y_mean * Y_n + m_total) / (Y_n + Z_n)

theorem least_possible_mean_YZ (X_n Y_n Z_n : ℕ) (X_mean Y_mean XY_mean XZ_mean : ℕ) 
  (hX_mean : X_mean = 30) (hY_mean : Y_mean = 60) 
  (hXY_mean : XY_mean = 50) (hXZ_mean : XZ_mean = 45) : 
  let X_n := 2 * Y_n in
  let m_total := 30 * Y_n + 45 * Z_n in 
  mean_combined_weight X_n Y_n Z_n X_mean Y_mean XY_mean XZ_mean m_total = 45 :=
by
  intros
  sorry

end least_possible_mean_YZ_l538_538267


namespace upstream_speed_is_10_l538_538283

noncomputable def downstream_speed : ℝ := 20
noncomputable def still_water_speed : ℝ := 15

theorem upstream_speed_is_10 :
  let C := downstream_speed - still_water_speed in
  let U := still_water_speed - C in
  U = 10 :=
by
  sorry

end upstream_speed_is_10_l538_538283


namespace spadesuit_nested_evaluation_l538_538815

def spadesuit (a b : ℝ) : ℝ := a - (1 / b)

theorem spadesuit_nested_evaluation :
  spadesuit 3 (spadesuit 3 (spadesuit 3 6)) = 118 / 45 := 
by
  sorry

end spadesuit_nested_evaluation_l538_538815


namespace find_tricksters_within_30_questions_l538_538318

/-- 
Given 65 inhabitants in a village where:
- Two inhabitants are tricksters and the rest are knights.
- Knights always tell the truth.
- Tricksters can either tell the truth or lie.
- One can show any inhabitant a list of some group of inhabitants (which can consist of one person)
  and ask if all of them are knights.

Prove that it is possible to find both tricksters with no more than 30 questions.
-/
theorem find_tricksters_within_30_questions :
  ∃ (ask_knights : (inhabitants : fin 65) → list (fin 65) → Prop),
  (∀ (i j : fin 65), i ≠ j → ask_knights i [j] = true → (inhabitants[j] = knight) ∨ (inhabitants[j] = trickster))
  ∧ ∀ (inhabitants : fin 65),
  (∃ S : finset (fin 65), S.card = 2 ∧ 
  (∀ i, i ∈ S → inhabitants[i] = trickster) ∧ 
  by asking no more than 30 questions,
  you can identify both tricksters.

end find_tricksters_within_30_questions_l538_538318


namespace find_tricksters_l538_538367

def inhab_group : Type := { n : ℕ // n < 65 }
def is_knight (i : inhab_group) : Prop := ∀ g : inhab_group, true -- Placeholder for the actual property

theorem find_tricksters (inhabitants : inhab_group → Prop)
  (is_knight : inhab_group → Prop)
  (knight_always_tells_truth : ∀ i : inhab_group, is_knight i → inhabitants i = true)
  (tricksters_2_and_rest_knights : ∃ t1 t2 : inhab_group, t1 ≠ t2 ∧ ¬is_knight t1 ∧ ¬is_knight t2 ∧
    (∀ i : inhab_group, i ≠ t1 → i ≠ t2 → is_knight i)) :
  ∃ find_them : inhab_group → inhab_group → Prop, (∀ q_count : ℕ, q_count ≤ 16) → 
  ∃ t1 t2 : inhab_group, find_them t1 t2 :=
by 
  -- The proof goes here
  sorry

end find_tricksters_l538_538367


namespace train_cross_pole_l538_538749

noncomputable def time_to_cross_pole (speed_kmh : ℕ) (length_m : ℕ) : ℕ :=
  lengths_m / (speed_kmh * 1000 / 3600)

theorem train_cross_pole (speed_kmh length_m : ℕ) :
  speed_kmh = 48 →
  length_m = 120 →
  time_to_cross_pole speed_kmh length_m = 9 :=
by
  intros h_speed h_len
  rw [h_speed, h_len]
  have h_speed_ms : (48 : ℚ) = 48 * 1000 / 3600 by norm_num
  have h_time_calc : 120 / (48 * 1000 / 3600) = 9 by norm_num
  exact h_time_calc

end train_cross_pole_l538_538749


namespace oa_dot_ob_eq_neg2_l538_538016

/-!
# Problem Statement
Given AB as the diameter of the smallest radius circle centered at C(0,1) that intersects 
the graph of y = 1 / (|x| - 1), where O is the origin. Prove that the dot product 
\overrightarrow{OA} · \overrightarrow{OB} equals -2.
-/

noncomputable def smallest_radius_circle_eqn (x : ℝ) : ℝ :=
  x^2 + ((1 / (|x| - 1)) - 1)^2

noncomputable def radius_of_circle (x : ℝ) : ℝ :=
  Real.sqrt (smallest_radius_circle_eqn x)

noncomputable def OA (x : ℝ) : ℝ × ℝ :=
  (x, (1 / (|x| - 1)) + 1)

noncomputable def OB (x : ℝ) : ℝ × ℝ :=
  (-x, 1 - (1 / (|x| - 1)))

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem oa_dot_ob_eq_neg2 (x : ℝ) (hx : |x| > 1) :
  let a := OA x
  let b := OB x
  dot_product a b = -2 :=
by
  sorry

end oa_dot_ob_eq_neg2_l538_538016


namespace smallest_n_exists_l538_538450

theorem smallest_n_exists (n : ℕ) :
  (∀ (a : Fin n → ℝ), (∀ i, 1 < a i ∧ a i < 1000) →
    ∃ (i j : Fin n), i ≠ j ∧ 0 < a i - a j ∧ a i - a j < 1 + 3 * Real.cbrt (a i * a j)) ↔ n ≥ 11 :=
begin
  sorry
end

end smallest_n_exists_l538_538450


namespace decimal_digits_of_expression_l538_538392

noncomputable def problem_statement : ℝ :=
  (2^10 + 1)^(4 / 3)

theorem decimal_digits_of_expression:
  let digits := (problem_statement - problem_statement.to_int) * 1000
  (digits.to_int % 1000) = 320 :=
by
  sorry

end decimal_digits_of_expression_l538_538392


namespace value_of_m_l538_538484

theorem value_of_m 
  (m : ℝ)
  (h1 : |m - 1| = 1)
  (h2 : m - 2 ≠ 0) : 
  m = 0 :=
  sorry

end value_of_m_l538_538484


namespace slope_CD_is_one_l538_538124

open Real

noncomputable def point := ℝ × ℝ

def line_through_origin (k : ℝ) (x : ℝ) : ℝ × ℝ := (x, k * x)

def intersects_curve (k x : ℝ) : Prop := k * x = exp (x - 1)

def curve_exp (x : ℝ) : ℝ := exp (x - 1)

def curve_ln (y : ℝ) : ℝ := log y

def line_slope (p1 p2 : point) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

theorem slope_CD_is_one (k x1 x2 : ℝ) (hx1 : intersects_curve k x1) (hx2 : intersects_curve k x2) :
  let A := line_through_origin k x1,
      B := line_through_origin k x2,
      C := (A.2, curve_ln A.2),
      D := (B.2, curve_ln B.2) in
  line_slope C D = 1 :=
begin
  sorry
end

end slope_CD_is_one_l538_538124


namespace total_widgets_sold_after_15_days_l538_538973

def widgets_sold_day_n (n : ℕ) : ℕ :=
  2 + (n - 1) * 3

def sum_of_widgets (n : ℕ) : ℕ :=
  n * (2 + widgets_sold_day_n n) / 2

theorem total_widgets_sold_after_15_days : 
  sum_of_widgets 15 = 345 :=
by
  -- Prove the arithmetic sequence properties and sum.
  sorry

end total_widgets_sold_after_15_days_l538_538973


namespace find_tricksters_l538_538364

def inhab_group : Type := { n : ℕ // n < 65 }
def is_knight (i : inhab_group) : Prop := ∀ g : inhab_group, true -- Placeholder for the actual property

theorem find_tricksters (inhabitants : inhab_group → Prop)
  (is_knight : inhab_group → Prop)
  (knight_always_tells_truth : ∀ i : inhab_group, is_knight i → inhabitants i = true)
  (tricksters_2_and_rest_knights : ∃ t1 t2 : inhab_group, t1 ≠ t2 ∧ ¬is_knight t1 ∧ ¬is_knight t2 ∧
    (∀ i : inhab_group, i ≠ t1 → i ≠ t2 → is_knight i)) :
  ∃ find_them : inhab_group → inhab_group → Prop, (∀ q_count : ℕ, q_count ≤ 16) → 
  ∃ t1 t2 : inhab_group, find_them t1 t2 :=
by 
  -- The proof goes here
  sorry

end find_tricksters_l538_538364


namespace find_t_l538_538869

open Real

namespace VectorSpace

def vector (α : Type*) := α × α

noncomputable def dot_product (u v : vector ℝ) :=
  u.1 * v.1 + u.2 * v.2

variables {t : ℝ} (a b : vector ℝ) (ta_b : vector ℝ)

def given_vectors : Prop :=
  a = (1, -1) ∧ b = (6, -4)

def perpendicular_condition : Prop :=
  dot_product a (t • a + b) = 0

theorem find_t (h1 : given_vectors a b) (h2 : perpendicular_condition a b t) : t = -5 :=
sorry

end VectorSpace

end find_t_l538_538869


namespace runner_speed_difference_l538_538616

noncomputable def speed_difference : ℝ :=
  let u := (1 : ℝ) / 2 -- time in hours
  let s₁ := 8 * u -- distance covered by the first runner
  let s₂ := λ v : ℝ, v * u -- distance covered by the second runner as a function of its speed
  let distance := 5 -- distance apart after 1/2 hour
  let v := 6 -- speed of the second runner calculated
  8 - v

theorem runner_speed_difference:
  let u := (1 : ℝ) / 2 in
  let s₁ := 8 * u in
  let distance := 5 in
  8 - (v = 6) = 2 :=
begin
  sorry
end

end runner_speed_difference_l538_538616


namespace area_comparison_parallelogram_condition_l538_538014

-- Definitions of the geometric entities
variable (A B C H I A1 A2 B1 B2 C1 C2 A' B' C' J R S T K : Point)

-- Conditions for the points and lines
variable (orthocenter : is_orthocenter H (triangle ABC))
variable (incenter : is_incenter I (triangle ABC))
variable (on_rays_A : on_ray A B A1 ∧ on_ray A C A2)
variable (on_rays_B : on_ray B C B1 ∧ on_ray B A B2)
variable (on_rays_C : on_ray C A C1 ∧ on_ray C B C2)
variable (equal_segments_A : segment_length A A1 = segment_length A A2 = segment_length B C)
variable (equal_segments_B : segment_length B B1 = segment_length B B2 = segment_length C A)
variable (equal_segments_C : segment_length C C1 = segment_length C C2 = segment_length A B)
variable (intersect_points : intersects (line_through B1 B2) (line_through C1 C2) A' ∧
                            intersects (line_through C1 C2) (line_through A1 A2) B' ∧
                            intersects (line_through A1 A2) (line_through B1 B2) C')
variable (circ_center : is_circumcenter J (triangle A' B' C'))
variable (intersect_lines_A : intersects (line_through A J) (line_through B C) R)
variable (intersect_lines_B : intersects (line_through B J) (line_through C A) S)
variable (intersect_lines_C : intersects (line_through C J) (line_through A B) T)
variable (common_point_K : common_point_of_circumcircles K (triangle A S T) (triangle B T R) (triangle C R S))

-- Statements to prove in Lean
theorem area_comparison :
  triangle_area (triangle A' B' C') ≤ triangle_area (triangle ABC) :=
sorry

theorem parallelogram_condition (h_not_isosceles : ¬ is_isosceles (triangle ABC)) :
  is_parallelogram (quadrilateral I H J K) :=
sorry

end area_comparison_parallelogram_condition_l538_538014


namespace initial_ratio_of_plants_fruiting_to_flowering_ratio_l538_538982

variable (F f total_remaining : ℕ) 

def initial_flowering_plants : ℕ := 7
def plants_on_saturday (initial_F : ℕ) : ℕ * ℕ := (initial_flowering_plants + 3, initial_F + 2)
def plants_on_sunday (initial_F : ℕ) : ℕ * ℕ := let (flowering, fruiting) := plants_on_saturday initial_F in (flowering - 1, fruiting - 4)
def total_plants (initial_F : ℕ) : ℕ := let (flowering, fruiting) := plants_on_sunday initial_F in flowering + fruiting

theorem initial_ratio_of_plants : ℕ :=
  let initial_F := 21 - 7 in   -- derived from equation solving in solution
  (initial_F, initial_flowering_plants)

theorem fruiting_to_flowering_ratio : (ℕ × ℕ) :=
  let (initial_F, f) := initial_ratio_of_plants in (initial_F / Nat.gcd initial_F f, f / Nat.gcd initial_F f)

#eval fruiting_to_flowering_ratio -- This should output (2, 1)

end initial_ratio_of_plants_fruiting_to_flowering_ratio_l538_538982


namespace symmetric_point_origin_l538_538801

def Point := (ℤ × ℤ)

def symmetric_with_respect_to_origin (P: Point) : Point := 
  (-P.1, -P.2)

theorem symmetric_point_origin :
  ∀ P : Point,
  P = (3, -4) →
  symmetric_with_respect_to_origin P = (-3, 4) :=
by
  intros P h
  simp [symmetric_with_respect_to_origin]
  cases h
  refl

end symmetric_point_origin_l538_538801


namespace sequence_a2017_l538_538108

theorem sequence_a2017 (a : ℕ → ℚ) (h₁ : a 1 = 1 / 2)
  (h₂ : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n / (3 * a n + 2)) :
  a 2017 = 1 / 3026 :=
sorry

end sequence_a2017_l538_538108


namespace max_value_of_expression_l538_538956

theorem max_value_of_expression (x y z : ℝ) (h₀ : x ≥ 0) (h₁ : y ≥ 0) (h₂ : z ≥ 0) (h₃ : x^2 + y^2 + z^2 = 1) : 
  3 * x * z * Real.sqrt 2 + 9 * y * z ≤ Real.sqrt 27 :=
sorry

end max_value_of_expression_l538_538956


namespace count_terminating_decimals_l538_538813

theorem count_terminating_decimals :
  (∃ m : ℕ, m = (finset.filter (λ n : ℕ, (1 ≤ n ∧ n ≤ 449 ∧ (450 % n == 0)) ) (finset.range 450)).card ∧ m = 24) :=
begin
  sorry,
end

end count_terminating_decimals_l538_538813


namespace sin_690_eq_negative_one_half_l538_538414

theorem sin_690_eq_negative_one_half : Real.sin (690 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_690_eq_negative_one_half_l538_538414


namespace find_tricksters_l538_538366

def inhab_group : Type := { n : ℕ // n < 65 }
def is_knight (i : inhab_group) : Prop := ∀ g : inhab_group, true -- Placeholder for the actual property

theorem find_tricksters (inhabitants : inhab_group → Prop)
  (is_knight : inhab_group → Prop)
  (knight_always_tells_truth : ∀ i : inhab_group, is_knight i → inhabitants i = true)
  (tricksters_2_and_rest_knights : ∃ t1 t2 : inhab_group, t1 ≠ t2 ∧ ¬is_knight t1 ∧ ¬is_knight t2 ∧
    (∀ i : inhab_group, i ≠ t1 → i ≠ t2 → is_knight i)) :
  ∃ find_them : inhab_group → inhab_group → Prop, (∀ q_count : ℕ, q_count ≤ 16) → 
  ∃ t1 t2 : inhab_group, find_them t1 t2 :=
by 
  -- The proof goes here
  sorry

end find_tricksters_l538_538366


namespace find_a_l538_538041

theorem find_a (a b : ℝ) (h1 : a * (a - 4) = 5) (h2 : b * (b - 4) = 5) (h3 : a ≠ b) (h4 : a + b = 4) : a = -1 :=
by 
sorry

end find_a_l538_538041


namespace cookies_last_days_l538_538131

theorem cookies_last_days (cookies_per_day_oldest : ℕ) (cookies_per_day_youngest : ℕ) (total_cookies : ℕ) :
  cookies_per_day_oldest = 4 →
  cookies_per_day_youngest = 2 →
  total_cookies = 54 →
  total_cookies / (cookies_per_day_oldest + cookies_per_day_youngest) = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end cookies_last_days_l538_538131


namespace printer_cost_l538_538288

theorem printer_cost (num_keyboards : ℕ) (num_printers : ℕ) (total_cost : ℕ) (keyboard_cost : ℕ) (printer_cost : ℕ) :
  num_keyboards = 15 →
  num_printers = 25 →
  total_cost = 2050 →
  keyboard_cost = 20 →
  (total_cost - (num_keyboards * keyboard_cost)) / num_printers = printer_cost →
  printer_cost = 70 :=
by
  sorry

end printer_cost_l538_538288


namespace parabola_hyperbola_focus_l538_538102

theorem parabola_hyperbola_focus (p : ℝ) :
  let parabolaFocus := (p / 2, 0)
  let hyperbolaRightFocus := (2, 0)
  (parabolaFocus = hyperbolaRightFocus) → p = 4 := 
by
  intro h
  sorry

end parabola_hyperbola_focus_l538_538102


namespace exists_spheres_l538_538716

-- Defining a structure for a regular polyhedron with certain properties
structure RegularPolyhedron where
  vertices : Set ℝ -- Assuming some geometric space
  faces : Set (Set ℝ)
  distance_face_to_vertex : ℝ
  dihedral_angle : ℝ
  is_regular : True -- Simplified property assuming it's true

-- The theorem statement
theorem exists_spheres (P : RegularPolyhedron) 
  (a : ℝ) (φ : ℝ) (h1 : P.distance_face_to_vertex = a)
  (h2 : P.dihedral_angle = 2 * φ) : 
  ∃ circumscribed_sphere inscribed_sphere, 
  (∀ v ∈ P.vertices, v ∈ circumscribed_sphere) ∧ 
  (∀ f ∈ P.faces, ∃ t, t ∈ inscribed_sphere ∧ t ∈ f) := 
by
  sorry

end exists_spheres_l538_538716


namespace price_of_72_cans_is_34_56_l538_538258

noncomputable def regular_price_per_can : ℝ := 0.60
noncomputable def discount : ℝ := 0.20
noncomputable def case_size : ℕ := 24
noncomputable def total_cans : ℕ := 72

def discounted_price_per_can : ℝ := regular_price_per_can * (1 - discount)
def price_for_72_cans : ℝ := (total_cans / case_size) * (case_size * discounted_price_per_can)

theorem price_of_72_cans_is_34_56 :
  price_for_72_cans = 34.56 :=
sorry

end price_of_72_cans_is_34_56_l538_538258


namespace valid_p_values_l538_538882

theorem valid_p_values (p : ℕ) (h : p = 3 ∨ p = 4 ∨ p = 5 ∨ p = 12) :
  0 < (4 * p + 34) / (3 * p - 8) ∧ (4 * p + 34) % (3 * p - 8) = 0 :=
by
  sorry

end valid_p_values_l538_538882


namespace time_to_complete_puzzles_l538_538563

-- Definitions based on conditions
def num_puzzles : ℕ := 2
def pieces_per_puzzle : ℕ := 2000
def pieces_per_minute : ℕ := 10 := 100 / 10

-- Proposition stating that James will take 400 minutes to finish both puzzles
theorem time_to_complete_puzzles : 
  let total_pieces := num_puzzles * pieces_per_puzzle in
  let time := total_pieces / pieces_per_minute in
  time = 400 :=
by
  let total_pieces := num_puzzles * pieces_per_puzzle
  have h1 : total_pieces = 4000 := by simp [num_puzzles, pieces_per_puzzle, Nat.mul_eq_mul]
  let time := total_pieces / pieces_per_minute
  have h2 : time = 400 := by simp [h1, pieces_per_minute, Nat.div_eq_div, Nat.mul_div, Nat.div_self]
  exact h2

end time_to_complete_puzzles_l538_538563


namespace side_length_of_equilateral_triangle_l538_538646

-- Define the problem context
variables (A B C P : Type) 
           [triangle_abc : equilateral_triangle A B C]  -- Equilateral triangle ABC
           (s : ℝ)  -- Side length of the triangle

-- Define points and distances as given
variables (AP : A → ℝ) (BP : B → ℝ) (CP : C → ℝ) 
           (unique_point_P : ∃! P, AP P = 1 ∧ BP P = sqrt 3 ∧ CP P = 2)

-- The main goal to prove
theorem side_length_of_equilateral_triangle (A B C P : Type) 
    [equilateral_triangle A B C] 
    (s : ℝ)
    (AP BP CP : Type → ℝ)
    (unique_point_P : ∃! P, AP P = 1 ∧ BP P = sqrt 3 ∧ CP P = 2) : s = sqrt 7 := 
  sorry  -- Proof skipped

end side_length_of_equilateral_triangle_l538_538646


namespace total_cost_calculation_l538_538603

-- Definitions
def coffee_price : ℕ := 4
def cake_price : ℕ := 7
def ice_cream_price : ℕ := 3

def mell_coffee_qty : ℕ := 2
def mell_cake_qty : ℕ := 1
def friends_coffee_qty : ℕ := 2
def friends_cake_qty : ℕ := 1
def friends_ice_cream_qty : ℕ := 1

def total_coffee_qty : ℕ := 3 * mell_coffee_qty
def total_cake_qty : ℕ := 3 * mell_cake_qty
def total_ice_cream_qty : ℕ := 2 * friends_ice_cream_qty

def total_cost : ℕ := total_coffee_qty * coffee_price + total_cake_qty * cake_price + total_ice_cream_qty * ice_cream_price

-- Theorem Statement
theorem total_cost_calculation : total_cost = 51 := by
  sorry

end total_cost_calculation_l538_538603


namespace prime_bound_l538_538150

open Nat

/-- The nth prime number. -/
def nth_prime : ℕ → ℕ
| 0 => 2
| 1 => 3
| n + 2 => (unbounded_sieve n).nthLe n (unbounded_sieve_bound n)

/-- Statement of the problem -/
theorem prime_bound (n : ℕ) : nth_prime n ≤ 2 ^ (2 ^ (n - 1)) :=
begin
  sorry
end

end prime_bound_l538_538150


namespace total_cost_price_l538_538742

-- Definitions for each item's selling price and profit percentage
def SP_A : ℝ := 150
def SP_B : ℝ := 200
def SP_C : ℝ := 250
def Profit_A : ℝ := 0.25
def Profit_B : ℝ := 0.20
def Profit_C : ℝ := 0.15

-- Compute the cost price for each item
def CP_A : ℝ := SP_A / (1 + Profit_A)
def CP_B : ℝ := SP_B / (1 + Profit_B)
def CP_C : ℝ := SP_C / (1 + Profit_C)

-- Prove that the total cost price equals the calculated value
theorem total_cost_price : CP_A + CP_B + CP_C = 504.06 := by
  sorry

end total_cost_price_l538_538742


namespace exists_center_for_homothety_l538_538142

-- Define the problem statement in Lean 4
theorem exists_center_for_homothety (P : set (ℝ²)) (hP : convex P) :
  ∃ O : ℝ², ∀ p ∈ P, (1/2 : ℝ) • (p - O) + O ∈ P :=
begin
  sorry
end

end exists_center_for_homothety_l538_538142


namespace expected_value_sum_two_random_marbles_l538_538087

theorem expected_value_sum_two_random_marbles :
  let marbles := [1, 2, 3, 4, 5, 6, 7] in
  let pairs := (marbles.pairs.filter (λ p, p.fst ≠ p.snd)).map (λ p, p.fst + p.snd) in
  (pairs.sum : ℚ) / pairs.length = 52 / 7 :=
by
  sorry

end expected_value_sum_two_random_marbles_l538_538087


namespace solution_set_inequality_l538_538104

noncomputable def decreasing_interval (a : ℝ) (x : ℝ) : Prop :=
f (x : ℝ) = a ^ (x^2 - 2 * x + 1) → ∀ x1 x2, (1 < x1 ∧ x1 < 3 ∧ 1 < x2 ∧ x2 < 3 ∧ x1 < x2) → f x1 > f x2

theorem solution_set_inequality (a : ℝ) (x : ℝ) (h1 : decreasing_interval a x) :
  (a^x > 1) ↔ (x < 0) := 
sorry

end solution_set_inequality_l538_538104


namespace sin_690_deg_l538_538426

noncomputable def sin_690_eq_neg_one_half : Prop :=
  sin (690 * real.pi / 180) = -(1 / 2)

theorem sin_690_deg : sin_690_eq_neg_one_half :=
  by sorry

end sin_690_deg_l538_538426


namespace magnitude_of_z_l538_538491

noncomputable def z : ℂ := 2 / (1 + (I : ℂ))

theorem magnitude_of_z (h : z * (1 + I) = 2) : complex.abs z = real.sqrt 2 := 
by sorry

end magnitude_of_z_l538_538491


namespace carlsson_max_candies_l538_538686

theorem carlsson_max_candies : 
  let n := 32
  in let initial_value := 1
     in let candies (x y : ℕ) := x * y
        in ∑ i in finset.range (n * (n - 1) / 2), candies initial_value initial_value = 496 :=
by
  sorry

end carlsson_max_candies_l538_538686


namespace simplest_square_root_l538_538371

theorem simplest_square_root :
  let A := Real.sqrt 0.1
  let B := 1 / Real.sqrt 2
  let C := Real.sqrt 2
  let D := Real.sqrt 8
  simplest_form C := C := by sorry

end simplest_square_root_l538_538371


namespace point_Q_coordinates_l538_538625

noncomputable def coordinates_of_point_Q (start : ℝ × ℝ) (arc_length : ℝ) : ℝ × ℝ :=
  let θ := arc_length in
  (Real.cos θ, Real.sin θ)

theorem point_Q_coordinates :
  coordinates_of_point_Q (1, 0) (4 * Real.pi / 3) = (-1 / 2, -Real.sqrt 3 / 2) :=
by
  sorry

end point_Q_coordinates_l538_538625


namespace smallest_number_collected_l538_538136

-- Define the numbers collected by each person according to the conditions
def jungkook : ℕ := 6 * 3
def yoongi : ℕ := 4
def yuna : ℕ := 5

-- The statement to prove
theorem smallest_number_collected : yoongi = min (min jungkook yoongi) yuna :=
by sorry

end smallest_number_collected_l538_538136


namespace point_C_on_line_l538_538757

-- Define the points A, B, C, D
def point_A := (0 : ℝ, -1 / 2 : ℝ)
def point_B := (-3 : ℝ, -2 : ℝ)
def point_C := (-2 : ℝ, -3 : ℝ)
def point_D := (1 / 2 : ℝ, 0 : ℝ)

-- Define the equation of the line y = 2x + 1
def line_equation (x y : ℝ) : Prop := y = 2 * x + 1

-- The theorem to be proved
theorem point_C_on_line : line_equation (fst point_C) (snd point_C) :=
by
  sorry

end point_C_on_line_l538_538757


namespace area_triangle_proof_l538_538848

variables {t1 t2 : ℂ}

def area_triangle (t1 t2 : ℂ) : ℂ :=
  (1 / (4 * complex.I)) * (complex.conj t1 * t2 - t1 * complex.conj t2)

def sin_angle_AOB (t1 t2 : ℂ) : ℂ :=
  (complex.conj (t1 * t2) - t1 * complex.conj t2) / (2 * complex.I * complex.abs (t1 * t2))

-- The proof statement
theorem area_triangle_proof (t1 t2 : ℂ) : 
  (1 / 2) * complex.Im (complex.conj t1 * t2) = (1 / (4 * complex.I)) * (complex.conj t1 * t2 - t1 * complex.conj t2) :=
sorry

end area_triangle_proof_l538_538848


namespace p_sufficient_not_necessary_for_q_l538_538963

open Real

def p (x : ℝ) : Prop := abs x < 1
def q (x : ℝ) : Prop := x^2 + x - 6 < 0

theorem p_sufficient_not_necessary_for_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l538_538963


namespace continuous_stripe_probability_l538_538795

-- Definitions based on the conditions
def face_stripe_orientation := { orientation : Bool // orientation = true ∨ orientation = false }

def total_possible_combinations : ℕ := 2^6

def continuous_encircling_stripe_combinations : ℕ := 12

-- Mathematical statement to be proven in Lean 4
theorem continuous_stripe_probability :
  (continuous_encircling_stripe_combinations / total_possible_combinations : ℚ) = 3 / 16 :=
begin
  sorry
end

end continuous_stripe_probability_l538_538795


namespace abs_diff_squares_104_98_l538_538700

theorem abs_diff_squares_104_98 : abs ((104 : ℤ)^2 - (98 : ℤ)^2) = 1212 := by
  sorry

end abs_diff_squares_104_98_l538_538700


namespace divisor_of_51234_plus_3_l538_538704

theorem divisor_of_51234_plus_3 : ∃ d : ℕ, d > 1 ∧ (51234 + 3) % d = 0 ∧ d = 3 :=
by {
  sorry
}

end divisor_of_51234_plus_3_l538_538704


namespace equation_of_parallel_line_through_point_l538_538003

theorem equation_of_parallel_line_through_point (c : ℝ) : 
  (∃ l : ℝ, ∃ m : ℝ, l x + m y + c = 0 ∧ ∃ x0 y0, (x0 = 1 ∧ y0 = 0) ∧ parallel_to x - 2*y - 2 = (x - 2*y - 1)) :=
sorry

end equation_of_parallel_line_through_point_l538_538003


namespace equidistant_trajectory_l538_538199

theorem equidistant_trajectory {x y z : ℝ} :
  (x + 1)^2 + (y - 1)^2 + z^2 = (x - 2)^2 + (y + 1)^2 + (z + 1)^2 →
  3 * x - 2 * y - z = 2 :=
sorry

end equidistant_trajectory_l538_538199


namespace printer_cost_l538_538285

theorem printer_cost (total_cost : ℕ) (num_keyboards : ℕ) (keyboard_cost : ℕ) (num_printers : ℕ) (printer_cost : ℕ) :
  total_cost = 2050 → num_keyboards = 15 → keyboard_cost = 20 → num_printers = 25 →
  (total_cost - num_keyboards * keyboard_cost) / num_printers = 70 := 
by
  intros h_tc h_nk h_kc h_np
  rw [h_tc, h_nk, h_kc, h_np]
  norm_num
  sorry

end printer_cost_l538_538285


namespace distance_between_tents_l538_538623

variables (p m s : ℕ)

-- Conditions
def cond1 := s + m = p + 20
def cond2 := s + p = m + 16

-- Theorem
theorem distance_between_tents : cond1 ∧ cond2 → s = 18 ∧ m = p + 2 :=
by
  sorry

end distance_between_tents_l538_538623


namespace remainder_of_division_987543_12_l538_538239

theorem remainder_of_division_987543_12 : 987543 % 12 = 7 := by
  sorry

end remainder_of_division_987543_12_l538_538239


namespace isosceles_triangle_congruent_side_length_l538_538541

theorem isosceles_triangle_congruent_side_length
  (PQ : ℝ) (A : ℝ)
  (hPQ : PQ = 30)
  (hA : A = 72) :
  let h := 4.8 in -- height calculated from area
  let PR := real.sqrt (15^2 + h^2) in
  PR ≈ 15.75 :=
by sorry

end isosceles_triangle_congruent_side_length_l538_538541


namespace series_imaginary_unit_l538_538216

noncomputable def sum_series : ℂ → ℕ → ℂ
| i, 0     := 0
| i, (n+1) := i^(n+1) + sum_series i n

theorem series_imaginary_unit :
  sum_series_complex i 2011 = -1 :=
by sorry

end series_imaginary_unit_l538_538216


namespace am_hm_inequality_l538_538145

theorem am_hm_inequality (a1 a2 a3 : ℝ) (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) (h_sum : a1 + a2 + a3 = 1) : 
  (1 / a1) + (1 / a2) + (1 / a3) ≥ 9 :=
by
  sorry

end am_hm_inequality_l538_538145


namespace sum_logarithms_l538_538781

open Real

theorem sum_logarithms :
  ∑ k in Finset.range (20 - 3 + 1) + 3, (log 3 (1 + 2 / k) * log k 3 * log (k + 2) 3) =
  1 - 1 / log 3 21 :=
by
  sorry

end sum_logarithms_l538_538781


namespace pauly_cannot_make_more_omelets_l538_538976

-- Pauly's omelet data
def total_eggs : ℕ := 36
def plain_omelet_eggs : ℕ := 3
def cheese_omelet_eggs : ℕ := 4
def vegetable_omelet_eggs : ℕ := 5

-- Requested omelets
def requested_plain_omelets : ℕ := 4
def requested_cheese_omelets : ℕ := 2
def requested_vegetable_omelets : ℕ := 3

-- Number of eggs used for each type of requested omelet
def total_requested_eggs : ℕ :=
  (requested_plain_omelets * plain_omelet_eggs) +
  (requested_cheese_omelets * cheese_omelet_eggs) +
  (requested_vegetable_omelets * vegetable_omelet_eggs)

-- The remaining number of eggs
def remaining_eggs : ℕ := total_eggs - total_requested_eggs

theorem pauly_cannot_make_more_omelets :
  remaining_eggs < min plain_omelet_eggs (min cheese_omelet_eggs vegetable_omelet_eggs) :=
by
  sorry

end pauly_cannot_make_more_omelets_l538_538976


namespace total_items_18_l538_538172

-- Define the number of dogs, biscuits per dog, and boots per set
def num_dogs : ℕ := 2
def biscuits_per_dog : ℕ := 5
def boots_per_set : ℕ := 4

-- Calculate the total number of items
def total_items (num_dogs biscuits_per_dog boots_per_set : ℕ) : ℕ :=
  (num_dogs * biscuits_per_dog) + (num_dogs * boots_per_set)

-- Prove that the total number of items is 18
theorem total_items_18 : total_items num_dogs biscuits_per_dog boots_per_set = 18 := by
  -- Proof is not provided
  sorry

end total_items_18_l538_538172


namespace area_of_right_triangle_with_30_60_degrees_l538_538647

noncomputable def area_of_triangle_with_altitude (h : ℝ) := sorry

theorem area_of_right_triangle_with_30_60_degrees
  (h : ℝ) (h_eq_4 : h = 4) : 
  area_of_triangle_with_altitude h = 16 * real.sqrt 3 :=
sorry

end area_of_right_triangle_with_30_60_degrees_l538_538647


namespace install_time_per_window_l538_538290

/-- A new building needed 14 windows. The builder had already installed 8 windows.
    It will take the builder 48 hours to install the rest of the windows. -/
theorem install_time_per_window (total_windows installed_windows remaining_install_time : ℕ)
  (h_total : total_windows = 14)
  (h_installed : installed_windows = 8)
  (h_remaining_time : remaining_install_time = 48) :
  (remaining_install_time / (total_windows - installed_windows)) = 8 :=
by
  -- Insert usual proof steps here
  sorry

end install_time_per_window_l538_538290


namespace find_angle_B_find_triangle_area_l538_538930

theorem find_angle_B (a b c : ℝ) (S : ℝ)
  (h_area : S = \frac{\sqrt{3}}{4} * (a^2 + c^2 - b^2)) :
  ∃ (B : ℝ), B = \frac{\π}{3} :=
by sorry

theorem find_triangle_area (a b c : ℝ) (S : ℝ)
  (h_B : B = \frac{\π}{3})
  (h_b : b = 2 * \sqrt{3})
  (h_sum : a + c = 6)
  (h_area : S = \frac{\sqrt{3}}{4} * (a^2 + c^2 - b^2)) :
  S = 2 * \sqrt{3} :=
by sorry

end find_angle_B_find_triangle_area_l538_538930


namespace problem_statement_l538_538470

def seq_a {n : ℕ} (a : ℕ → ℕ) : Prop :=
  a 2 = 2 ∧ ∀ n : ℕ, (n + 1) * a (n + 1) = (n + 2) * a n + 1

def seq_b (a b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = a n * 2^(a n)

def sum_S (b : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in Finset.range n, b (i + 1)

theorem problem_statement (a b S : ℕ → ℕ) (h_seq_a : seq_a a)
  (h_seq_b : seq_b a b) (h_sum_S : sum_S b S) :
  a 1 = 1 ∧ a 3 = 3 ∧ (∀ n : ℕ, a n = n) ∧ ∀ n : ℕ, S n = 2 * b n - 2^(n + 1) + 2 :=
sorry

end problem_statement_l538_538470


namespace evaluate_g_at_neg_four_l538_538878

def g (x : Int) : Int := 5 * x + 2

theorem evaluate_g_at_neg_four : g (-4) = -18 := by
  sorry

end evaluate_g_at_neg_four_l538_538878


namespace time_to_fill_box_correct_l538_538168

def total_toys := 50
def mom_rate := 5
def mia_rate := 3

def time_to_fill_box (total_toys mom_rate mia_rate : ℕ) : ℚ :=
  let net_rate_per_cycle := mom_rate - mia_rate
  let cycles := ((total_toys - 1) / net_rate_per_cycle) + 1
  let total_seconds := cycles * 30
  total_seconds / 60

theorem time_to_fill_box_correct : time_to_fill_box total_toys mom_rate mia_rate = 12.5 :=
by
  sorry

end time_to_fill_box_correct_l538_538168


namespace xiao_wang_scores_problem_l538_538711

-- Defining the problem conditions and solution as a proof problem
theorem xiao_wang_scores_problem (x y : ℕ) (h1 : (x * y + 98) / (x + 1) = y + 1) 
                                 (h2 : (x * y + 98 + 70) / (x + 2) = y - 1) :
  (x + 2 = 10) ∧ (y - 1 = 88) :=
by 
  sorry

end xiao_wang_scores_problem_l538_538711


namespace find_tricksters_l538_538322

structure Inhabitant :=
  (isKnight : Prop)

constants (inhabitants : Fin 65 → Inhabitant)
          (tricksters : Fin 2 → Fin 65)

axiom two_tricksters_unique :
  ∃! (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65))

axiom valid_question :
  ∀ (a : Fin 65) (group : Set (Fin 65)), (inhabitants a).isKnight → 
  (∀ i ∈ group, (inhabitants i).isKnight) ↔ 
  (knight a).isKnight

theorem find_tricksters :
  ∃ (q : ℕ), q ≤ 16 ∧
  ∃ (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65)) :=
sorry

end find_tricksters_l538_538322


namespace connectivity_after_n_minus_1_flights_cancelled_l538_538226

theorem connectivity_after_n_minus_1_flights_cancelled 
  (n : ℕ) (h : n > 0)
  (cities : Finset ℕ)
  (H1 : cities.card = 2 * n)
  (connections : ℕ → Finset ℕ)
  (H2 : ∀ c ∈ cities, (connections c).card ≥ n)
  (cancelled_flights : Finset (ℕ × ℕ))
  (H3 : cancelled_flights.card = n - 1) :
  ∀ c₁ c₂ ∈ cities, ∃ path : List ℕ, path.length ≥ 2 ∧ path.head = c₁ ∧ path.last = c₂ ∧ 
                    ∀ i ∈ List.zip path (path.tail), (cancelled_flights ∪ Finset.singleton i).disjoint (bi_relation connections) :=
sorry

end connectivity_after_n_minus_1_flights_cancelled_l538_538226


namespace total_trophies_l538_538933

theorem total_trophies (michael_now : ℕ) (increase : ℕ) (jack_multiplication : ℕ) :
  michael_now = 30 →
  increase = 100 →
  jack_multiplication = 10 →
  let michael_future := michael_now + increase in
  let jack_future := jack_multiplication * michael_now in
  michael_future + jack_future = 430 :=
by
  intros h_michael h_increase h_jack
  simp [h_michael, h_increase, h_jack]
  sorry

end total_trophies_l538_538933


namespace train_speed_invariant_l538_538310

-- Problem domain definitions
def initial_length : ℝ := 160 -- Initial length of the train in meters
def initial_speed : ℝ := 30 -- Initial speed of the train in meters per second
def rate_of_length_increase : ℝ := 2 -- Rate of increase of the length of the train in meters per second
def final_speed : ℝ := 30 -- The speed of the train after crossing the man, as given in condition

-- Theorem statement
theorem train_speed_invariant (length_increase_effect : |rate_of_length_increase -> |fin_speed) (no_wind_resistance : True) : 
  (initial_speed = final_speed) :=
by
  sorry

end train_speed_invariant_l538_538310


namespace arc_length_equality_l538_538264

variables {R : Type*} [MetricSpace R] [NormedAddCommGroup R] [NormedSpace ℝ R]
variables (O A B C D E F G : R)

def projection (p : R) (L : R) : R := sorry -- Placeholder for projection function
def midpoint (p₁ p₂ : R) : R := (p₁ + p₂) / 2

variables (h₁ : dist O A = dist O B) (h₂ : abs (angle O A B - π / 2) < ε) (h₃ : dist B D < dist A D)
variables (hD : projection C O = D) (hE : projection C O = E) (hF : midpoint D E = F) 
variables (hG : angle O F G = π / 2)

theorem arc_length_equality 
  (h_dist_DE : dist D E = dist D C + dist C E)
  (h_dist_CF : dist C F = dist C D + dist D E / 2)
  (h_mid_oc : midpoint O C = G)
  (h_dist_GC : dist G C = dist G F):
  arc O G = arc O C := sorry

end arc_length_equality_l538_538264


namespace sum_of_smallest_odd_primes_math_problem_part_b_math_problem_l538_538453

theorem sum_of_smallest_odd_primes (a b : ℤ) (p : ℤ) 
    (hp : Nat.Prime p) 
    (hodd : p % 2 = 1) 
    (hnot_div_b : ¬ p ∣ b) 
    (hev : b % 2 = 0) 
    (heq : p ^ 2 = a ^ 3 + b ^ 2) :
  p = 13 ∨ p = 109 := 
begin 
  sorry 
end

theorem math_problem_part_b :
  13 + 109 = 122 := 
by norm_num

theorem math_problem : 
  (∃ p1 p2 : ℤ, (∀ (a b : ℤ), Nat.Prime p1 ∧ p1 % 2 = 1 ∧ ¬ p1 ∣ b ∧ b % 2 = 0 ∧ p1 ^ 2 = a ^ 3 + b ^ 2) ∧ 
                    (∀ (a b : ℤ), Nat.Prime p2 ∧ p2 % 2 = 1 ∧ ¬ p2 ∣ b ∧ b % 2 = 0 ∧ p2 ^ 2 = a ^ 3 + b ^ 2) ∧ 
                    p1 ≠ p2) → 
    (13 + 109 = 122) := 
by exact math_problem_part_b

end sum_of_smallest_odd_primes_math_problem_part_b_math_problem_l538_538453


namespace find_tricksters_within_30_questions_l538_538317

/-- 
Given 65 inhabitants in a village where:
- Two inhabitants are tricksters and the rest are knights.
- Knights always tell the truth.
- Tricksters can either tell the truth or lie.
- One can show any inhabitant a list of some group of inhabitants (which can consist of one person)
  and ask if all of them are knights.

Prove that it is possible to find both tricksters with no more than 30 questions.
-/
theorem find_tricksters_within_30_questions :
  ∃ (ask_knights : (inhabitants : fin 65) → list (fin 65) → Prop),
  (∀ (i j : fin 65), i ≠ j → ask_knights i [j] = true → (inhabitants[j] = knight) ∨ (inhabitants[j] = trickster))
  ∧ ∀ (inhabitants : fin 65),
  (∃ S : finset (fin 65), S.card = 2 ∧ 
  (∀ i, i ∈ S → inhabitants[i] = trickster) ∧ 
  by asking no more than 30 questions,
  you can identify both tricksters.

end find_tricksters_within_30_questions_l538_538317


namespace sin_min_period_set_l538_538200

-- Function definition
def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sin (ω * x - Real.pi / 6)

-- Lean statement
theorem sin_min_period_set (ω : ℝ) (hω : ω > 0) (T : ℝ) (hT : T = 4 * Real.pi) :
  (∀ x : ℝ, f ω x = 2 * Real.sin (ω * x - Real.pi / 6)) →
  T = (2 * Real.pi / ω) →
  {x | ∃ k : ℤ, x = 4 * (k : ℝ) * Real.pi - 2 * Real.pi / 3} =
  {x | f ω x = -2} :=
by
  -- You can add the actual proof here.
  sorry

end sin_min_period_set_l538_538200


namespace monotonic_when_a_is_neg1_find_extreme_points_l538_538504

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/3) * x ^ 3 - (1/2) * (a^2 + a + 2) * x ^ 2 + a^2 * (a + 2) * x

theorem monotonic_when_a_is_neg1 :
  ∀ x : ℝ, f x (-1) ≤ f x (-1) :=
sorry

theorem find_extreme_points (a : ℝ) :
  if h : a = -1 ∨ a = 2 then
    True  -- The function is monotonically increasing, no extreme points
  else if h : a < -1 ∨ a > 2 then
    ∃ x_max x_min : ℝ, x_max = a + 2 ∧ x_min = a^2 ∧ (f x_max a ≥ f x a ∧ f x_min a ≤ f x a) 
  else
    ∃ x_max x_min : ℝ, x_max = a^2 ∧ x_min = a + 2 ∧ (f x_max a ≥ f x a ∧ f x_min a ≤ f x a) :=
sorry

end monotonic_when_a_is_neg1_find_extreme_points_l538_538504


namespace sara_steps_l538_538984

theorem sara_steps (n : ℕ) (h : n^2 ≤ 210) : n = 14 :=
sorry

end sara_steps_l538_538984


namespace identify_tricksters_in_30_or_less_questions_l538_538358

-- Define the problem parameters
def inhabitants : Type := Fin 65

def is_knight (inhabitant : inhabitants) : Prop := sorry
def is_trickster (inhabitant : inhabitants) : Prop := sorry

-- Define the properties
axiom knight_truthful : ∀ (x : inhabitants), is_knight x → (forall y : inhabitants, True ↔ (is_knight y = x is_knight y))
axiom trickster_mixed : ∀ (x : inhabitants), is_trickster x → ((∀ y : inhabitants, True) ∨ (∃ y : inhabitants, y ∉ (is_knight y)))

-- Problem statement
theorem identify_tricksters_in_30_or_less_questions
  (inhabitants : Type)
  (n_tricksters : ℕ := 2) -- 2 tricksters
  (total_inhabitants : ℕ := 65) -- 65 total inhabitants
  (questions_limit : ℕ := 30) -- limit of 30 questions
  (knights : inhabitants → Prop)
  (tricksters : inhabitants → Prop) :
    ∃ (solution_exists : ∀ (is_trickster : inhabitants → Prop), ∃ k : inhabitants, (knights k) ∧ (is_trickster k)) 
    (possible_to_find_tricksters : ∀ (is_knight : inhabitants → Prop) (is_trickster : inhabitants → Prop), 
    ∃ (questions_used ≤ questions_limit), ∀ (xs : set inhabitants), questions_used ≤ 30 ∧ 
    (∃ trickster1 trickster2 : inhabitants, (tricksters trickster1 ∧ tricksters trickster2 ∧ trickster1 ≠ trickster2))) :=
sorry

end identify_tricksters_in_30_or_less_questions_l538_538358


namespace solve_for_y_l538_538506

theorem solve_for_y (x y : ℝ) (h : 4 * x - y = 3) : y = 4 * x - 3 :=
by sorry

end solve_for_y_l538_538506


namespace subletter_payment_correct_l538_538940

noncomputable def johns_monthly_rent : ℕ := 900
noncomputable def johns_yearly_rent : ℕ := johns_monthly_rent * 12
noncomputable def johns_profit_per_year : ℕ := 3600
noncomputable def total_rent_collected : ℕ := johns_yearly_rent + johns_profit_per_year
noncomputable def number_of_subletters : ℕ := 3
noncomputable def subletter_annual_payment : ℕ := total_rent_collected / number_of_subletters
noncomputable def subletter_monthly_payment : ℕ := subletter_annual_payment / 12

theorem subletter_payment_correct :
  subletter_monthly_payment = 400 :=
by
  sorry

end subletter_payment_correct_l538_538940


namespace largest_divisor_of_expression_for_odd_l538_538520

theorem largest_divisor_of_expression_for_odd (x : ℕ) (h_odd : x % 2 = 1) : 
  ∃ k : ℕ, (∀ y : ℕ, y % 2 = 1 → k ∣ (10 * y + 2) * (10 * y + 6) * (5 * y + 5)) ∧ ∀ n, (∃ y : ℕ, y % 2 = 1 ∧ n ∣ (10 * y + 2) * (10 * y + 6) * (5 * y + 5)) → n ≤ k :=
begin
  use 960,
  split,
  { intro y, intro hy_odd,
    -- Proof that 960 divides (10 * y + 2) * (10 * y + 6) * (5 * y + 5) for odd y.
    sorry }, 
  { intros n hn,
    -- Proof that no larger number than 960 can always divide the expression.
    sorry 
  }
end

end largest_divisor_of_expression_for_odd_l538_538520


namespace john_days_to_lose_weight_l538_538568

noncomputable def john_calories_intake : ℕ := 1800
noncomputable def john_calories_burned : ℕ := 2300
noncomputable def calories_to_lose_1_pound : ℕ := 4000
noncomputable def pounds_to_lose : ℕ := 10

theorem john_days_to_lose_weight :
  (john_calories_burned - john_calories_intake) * (pounds_to_lose * calories_to_lose_1_pound / (john_calories_burned - john_calories_intake)) = 80 :=
by
  sorry

end john_days_to_lose_weight_l538_538568


namespace cafeteria_total_cost_l538_538608

-- Definitions based on conditions
def cost_per_coffee := 4
def cost_per_cake := 7
def cost_per_ice_cream := 3
def mell_coffee := 2 
def mell_cake := 1 
def friends_coffee := 2 
def friends_cake := 1 
def friends_ice_cream := 1 
def num_friends := 2
def total_coffee := mell_coffee + num_friends * friends_coffee
def total_cake := mell_cake + num_friends * friends_cake
def total_ice_cream := num_friends * friends_ice_cream

-- Total cost
def total_cost := total_coffee * cost_per_coffee + total_cake * cost_per_cake + total_ice_cream * cost_per_ice_cream

-- Theorem statement
theorem cafeteria_total_cost : total_cost = 51 := by
  sorry

end cafeteria_total_cost_l538_538608


namespace inequality_proof_l538_538064

theorem inequality_proof
  (x y : ℝ)
  (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 :=
sorry

end inequality_proof_l538_538064


namespace lines_in_4x4_grid_l538_538083

theorem lines_in_4x4_grid : 
  let points := finset.range 16 in
  let combinations := finset.pair_combinations points in
  let horizontal_lines := 16 in
  let vertical_lines := 16 in
  let main_diagonal_lines := 2 in
  let minor_diagonal_lines := 2 in
  (horizontal_lines + vertical_lines + main_diagonal_lines + minor_diagonal_lines) = 36 :=
by sorry

end lines_in_4x4_grid_l538_538083


namespace cricket_bat_profit_percentage_l538_538712

-- Definitions for the problem conditions
def selling_price : ℝ := 850
def profit : ℝ := 255
def cost_price : ℝ := selling_price - profit
def expected_profit_percentage : ℝ := 42.86

-- The theorem to be proven
theorem cricket_bat_profit_percentage : 
  (profit / cost_price) * 100 = expected_profit_percentage :=
by 
  sorry

end cricket_bat_profit_percentage_l538_538712


namespace total_cost_calculation_l538_538601

-- Definitions
def coffee_price : ℕ := 4
def cake_price : ℕ := 7
def ice_cream_price : ℕ := 3

def mell_coffee_qty : ℕ := 2
def mell_cake_qty : ℕ := 1
def friends_coffee_qty : ℕ := 2
def friends_cake_qty : ℕ := 1
def friends_ice_cream_qty : ℕ := 1

def total_coffee_qty : ℕ := 3 * mell_coffee_qty
def total_cake_qty : ℕ := 3 * mell_cake_qty
def total_ice_cream_qty : ℕ := 2 * friends_ice_cream_qty

def total_cost : ℕ := total_coffee_qty * coffee_price + total_cake_qty * cake_price + total_ice_cream_qty * ice_cream_price

-- Theorem Statement
theorem total_cost_calculation : total_cost = 51 := by
  sorry

end total_cost_calculation_l538_538601


namespace sequence_100th_term_eq_l538_538970

-- Definitions for conditions
def numerator (n : ℕ) : ℕ := 1 + (n - 1) * 2
def denominator (n : ℕ) : ℕ := 2 + (n - 1) * 3

-- The statement of the problem as a Lean 4 theorem
theorem sequence_100th_term_eq :
  (numerator 100) / (denominator 100) = 199 / 299 :=
by
  sorry

end sequence_100th_term_eq_l538_538970


namespace largest_m_dividing_pow_product_l538_538784

def pow (n : ℕ) : ℕ := λ fact, let max_prime_factor := (max (factorization n).to_list.keys.filter (Prime)) in fact * max_prime_factor

def is_divisible_by (a b : ℕ) := ∃ (k : ℕ), b * k = a

noncomputable def max_m_dividing_product_pow : ℕ :=
    let product := ∏ i in (range 2023).filter (λ i, i ≥ 2), pow i in
    let m := 386 in
    if (630 ^ m) ∣ product then m else 0

theorem largest_m_dividing_pow_product :
  is_divisible_by (∏ n in (range 2023).filter (λ (n : ℕ), n ≥ 2), pow n) (630 ^ 386) := sorry

end largest_m_dividing_pow_product_l538_538784


namespace count_terminating_decimals_l538_538812

theorem count_terminating_decimals :
  (∃ m : ℕ, m = (finset.filter (λ n : ℕ, (1 ≤ n ∧ n ≤ 449 ∧ (450 % n == 0)) ) (finset.range 450)).card ∧ m = 24) :=
begin
  sorry,
end

end count_terminating_decimals_l538_538812


namespace time_for_pipe_a_to_fill_l538_538305

noncomputable def pipe_filling_time (a_rate b_rate c_rate : ℝ) (fill_time_together : ℝ) : ℝ := 
  (1 / a_rate)

theorem time_for_pipe_a_to_fill (a_rate b_rate c_rate : ℝ) (fill_time_together : ℝ) 
  (h1 : b_rate = 2 * a_rate) 
  (h2 : c_rate = 2 * b_rate) 
  (h3 : (a_rate + b_rate + c_rate) * fill_time_together = 1) : 
  pipe_filling_time a_rate b_rate c_rate fill_time_together = 42 :=
sorry

end time_for_pipe_a_to_fill_l538_538305


namespace find_a_from_inequality_system_l538_538858

theorem find_a_from_inequality_system (a x : ℝ) 
    (h1 : 2 * x - 1 ≥ 1) 
    (h2 : x ≥ a) 
    (h_solution : x ≥ 2) : 
    a = 2 :=
begin
  sorry
end

end find_a_from_inequality_system_l538_538858


namespace box_selection_min_cost_mass_l538_538756

theorem box_selection_min_cost_mass (n k : ℕ) (a : ℕ → ℕ) (h : ℕ) 
  (mass_cond : ∀ i, i ≤ n → a i ≤ a (i + 1))
  (price : ℕ → ℕ) (price_cond : ∀ i, i ≤ n → price i = i) 
  (n_eq : n = 10) (k_eq : k = 3) : 
  let selected_boxes := [4, 7, 10] in
  (∀ j : ℕ, j < k → selected_boxes.nth j > (j * n / k)) ∧ 
  (∑ i in selected_boxes, a i ≥ k * ∑ i in finset.range n, a i / h) := 
  sorry

end box_selection_min_cost_mass_l538_538756


namespace center_of_circle_is_correct_minimum_length_of_tangent_segment_is_correct_l538_538921

noncomputable theory

open Real

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

def circle_in_polar : (θ : ℝ) → ℝ := 
  λ θ, (sqrt 2) * cos (θ + π / 4)

def line_l (t : ℝ) : ℝ × ℝ := 
  (t, t + 2)

theorem center_of_circle_is_correct :
  ∃ (x y : ℝ), 
    (sqrt 2 * cos (atan2 y x + π / 4))^2 = x^2 + y^2 ∧ 
    x = 1 / 2 ∧ y = -1 / 2 :=
by sorry

theorem minimum_length_of_tangent_segment_is_correct :
  ∀ (t : ℝ), 
    let (x, y) := line_l t in
    sqrt ((x - 1 / 2)^2 + (y + 1 / 2)^2 - (sqrt 2 / 2)^2) ≥ 2 :=
by sorry

end center_of_circle_is_correct_minimum_length_of_tangent_segment_is_correct_l538_538921


namespace cricket_average_l538_538727

theorem cricket_average (A : ℝ) (h : 20 * A + 120 = 21 * (A + 4)) : A = 36 :=
by sorry

end cricket_average_l538_538727


namespace inequality_proof_l538_538262

variables {n : ℕ} {a : Fin n → ℝ}

-- Conditions:
axiom distinct_positive (h : ∀ i j : Fin n, i ≠ j → a i ≠ a j) (h' : ∀ k : Fin n, a k > 0)
axiom n_ge_two (h_n : 2 ≤ n)
axiom sum_inverse_2n (h_sum : ∑ k in Finset.univ, (a k) ^ (-2 * n) = 1)

-- Problem statement:
theorem inequality_proof :
  ∑ k in Finset.univ, (a k) ^ (2 * n)
  - n^2 * ∑ i in Finset.univ, ∑ j in Finset.filter (λ j, i < j) Finset.univ,
            ((a i / a j) - (a j / a i))^2 > n^2 :=
sorry

end inequality_proof_l538_538262


namespace find_constants_exist_l538_538443

theorem find_constants_exist :
  ∃ A B C, (∀ x, 4 * x / ((x - 5) * (x - 3)^2) = A / (x - 5) + B / (x - 3) + C / (x - 3)^2)
  ∧ (A = 5) ∧ (B = -5) ∧ (C = -6) := 
sorry

end find_constants_exist_l538_538443


namespace num_two_digit_numbers_l538_538085

def set := {3, 4, 5, 6}

theorem num_two_digit_numbers : 
  ∃ n : ℕ, n = 12 ∧ ∀ a b : ℕ, a ∈ set ∧ b ∈ set ∧ a ≠ b → 
    (a * 10 + b) ∈ {x : ℕ | 10 ≤ x ∧ x < 100} ∧
    {x : ℕ | x = a * 10 + b} ⊆ {34, 35, 36, 43, 45, 46, 53, 54, 56, 63, 64, 65} := by
  -- Proof
  sorry

end num_two_digit_numbers_l538_538085


namespace sum_of_first_fifteen_multiples_of_eight_l538_538241

theorem sum_of_first_fifteen_multiples_of_eight :
  (∑ k in finset.range 15, 8 * (k + 1)) = 960 :=
by
  sorry

end sum_of_first_fifteen_multiples_of_eight_l538_538241


namespace locus_of_circumcenter_l538_538121

theorem locus_of_circumcenter (θ : ℝ) :
  let M := (3, 3 * Real.tan (θ - Real.pi / 3))
  let N := (3, 3 * Real.tan θ)
  let C := (3 / 2, 3 / 2 * Real.tan θ)
  ∃ (x y : ℝ), (x - 4) ^ 2 / 4 - y ^ 2 / 12 = 1 :=
by
  sorry

end locus_of_circumcenter_l538_538121


namespace train_speed_l538_538746

noncomputable def speed_in_kmh (distance : ℕ) (time : ℕ) : ℚ :=
  (distance : ℚ) / (time : ℚ) * 3600 / 1000

theorem train_speed
  (distance : ℕ) (time : ℕ)
  (h_dist : distance = 150)
  (h_time : time = 9) :
  speed_in_kmh distance time = 60 :=
by
  rw [h_dist, h_time]
  sorry

end train_speed_l538_538746


namespace a_n_form_T_n_form_l538_538027

section SeqProperties

variable {a : ℕ → ℕ} {b : ℕ → ℕ} {S : ℕ → ℕ}

-- Conditions:
-- The sequence a_n is such that a_1 = 1
def a1 : Prop := a 1 = 1

-- And a_(n+1) = 3 * (sum of first n terms of a) + 1
def recurrence_relation (n : ℕ) : Prop := a (n+1) = 3 * (finset.sum (finset.range n) a) + 1

-- The sequence b_n is defined as b_n = n + a_n
def b_def (n : ℕ) : Prop := b n = n + a n

-- Questions to prove:

-- 1. Prove that a_n = 4^{n-1}
theorem a_n_form (n : ℕ) (h1: a1) (hr: ∀ i, recurrence_relation i) : a n = 4^(n-1) :=
sorry

-- 2. Prove that T_n = b₁ + b₂ + ... + bₙ = (n(n+1))/2 + (4^n - 1)/3
theorem T_n_form (n : ℕ) (h1: a1) (hr: ∀ i, recurrence_relation i) (hb: ∀ i, b_def i) : 
  (finset.sum (finset.range n) b) = (n * (n + 1))/2 + (4^n - 1)/3 :=
sorry

end SeqProperties

end a_n_form_T_n_form_l538_538027


namespace number_of_solutions_l538_538841

theorem number_of_solutions (x : ℤ) (h1 : 0 < x) (h2 : x < 150) (h3 : (x + 17) % 46 = 75 % 46) : 
  ∃ n : ℕ, n = 3 :=
sorry

end number_of_solutions_l538_538841


namespace clock_hands_alignment_l538_538175

theorem clock_hands_alignment :
  ∃ (t : ℝ), 0 < t ∧
  let minute_hand_speed := 12 in
  let hour_hand_speed := 1 in
  let initial_minute_position := 9 in
  let initial_hour_position := 4 + 3 / 4 in
  let initial_separation := initial_minute_position -
                            (initial_hour_position * minute_hand_speed / hour_hand_speed) in
  let adjusted_initial_separation := initial_separation % minute_hand_speed in
  let seventh_alignment_position := 7 * (minute_hand_speed - hour_hand_speed) in
    t = (adjusted_initial_separation + seventh_alignment_position) / minute_hand_speed ∧
    t * 60 = 435 :=
begin
  sorry,
end

end clock_hands_alignment_l538_538175


namespace area_ratio_of_triangles_l538_538174

theorem area_ratio_of_triangles
  {A B C C1 A1 B1 : Type}
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
  [AddCommGroup C1] [AddCommGroup A1] [AddCommGroup B1]
  (tri_ABC : Triangle A B C)
  (tri_A1B1C1 : Triangle A1 B1 C1)
  (h1 : length AB = length BC1)
  (h2 : length BC = length CA1)
  (h3 : length CA = length AB1)
  (h4 : OnExtension A B C1)
  (h5 : OnExtension B C A1)
  (h6 : OnExtension C A B1)
  : area_triangle A1 B1 C1 = 7 * area_triangle A B C :=
sorry

end area_ratio_of_triangles_l538_538174


namespace range_of_k_l538_538492

noncomputable def equation (k x : ℝ) : ℝ :=
  sqrt 2 * abs (x - k) - k * sqrt x

theorem range_of_k :
  (∀ k : ℝ, (0 < k ∧ k ≤ 1) ↔
    ∃ x1 x2 ∈ set.Icc (k - 1) (k + 1), x1 ≠ x2 ∧ equation k x1 = 0 ∧ equation k x2 = 0) :=
sorry

end range_of_k_l538_538492


namespace triangle_angle_construction_l538_538629

-- Step d): Lean 4 Statement
theorem triangle_angle_construction (a b c : ℝ) (α β : ℝ) (γ : ℝ) (h1 : γ = 120)
  (h2 : a < c) (h3 : c < a + b) (h4 : b < c)  (h5 : c < a + b) :
    (∃ α' β' γ', α' = 60 ∧ β' = α ∧ γ' = 60 + β) ∧ 
    (∃ α'' β'' γ'', α'' = 60 ∧ β'' = β ∧ γ'' = 60 + α) :=
  sorry

end triangle_angle_construction_l538_538629


namespace find_tricksters_l538_538351

theorem find_tricksters (inhabitants : Fin 65 → Prop) (is_knight : Fin 65 → Prop)
    (total_inhabitants : ∀ n, inhabitants n)
    (knights : ∀ n, is_knight n → inhabitants n)
    (tricksters_count : (∑ n, if ¬ is_knight n then 1 else 0) = 2)
    (knights_count : (∑ n, if is_knight n then 1 else 0) = 63)
    (knight_truth : ∀ n, is_knight n → ∀ l : list (Fin 65), (∀ m ∈ l, is_knight m) ↔ true)
    (ask_question : ∀ n, inhabitants n → ∀ l : list (Fin 65), bool) :
  ∃ (find_tricksters_function : (Fin 65 → Prop) → (Fin 65 → bool) → (list (Fin 65))) ,
    (length (find_tricksters_function inhabitants ask_question) ≤ 2) →
    (length (find_tricksters_function inhabitants ask_question) = 2) ∧
    ∀ t ∈ (find_tricksters_function inhabitants ask_question), ¬ is_knight t :=
by sorry

end find_tricksters_l538_538351


namespace polynomial_multiple_root_iff_common_root_l538_538176

theorem polynomial_multiple_root_iff_common_root (f : Polynomial ℝ) (n : ℕ) 
  (h_n : n ≥ 2) (h_deg : f.degree = n) : 
  (∃ x, IsMultipleRoot f x) ↔ (∃ x, f.eval x = 0 ∧ (f.derivative).eval x = 0) :=
by
  sorry

end polynomial_multiple_root_iff_common_root_l538_538176


namespace neg_p_necessary_not_sufficient_neg_q_l538_538883

def p (x : ℝ) : Prop := x^2 - 1 > 0
def q (x : ℝ) : Prop := (x + 1) * (x - 2) > 0
def not_p (x : ℝ) : Prop := ¬ (p x)
def not_q (x : ℝ) : Prop := ¬ (q x)

theorem neg_p_necessary_not_sufficient_neg_q : ∀ (x : ℝ), (not_q x → not_p x) ∧ ¬ (not_p x → not_q x) :=
by
  sorry

end neg_p_necessary_not_sufficient_neg_q_l538_538883


namespace student_conclusions_l538_538832

noncomputable def f : ℝ → ℝ := λ x, if x ∈ [0,2] then Real.log ((x + 1) / Real.log 2) else 0

theorem student_conclusions (f_odd : ∀ x, f (-x) = -f x)
                            (f_shift : ∀ x, f (x - 4) = -f x)
                            (f_def : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x = Real.log ((x + 1) / Real.log 2)) :
  (f 3 = 1) ∧ (∀ x, -6 ≤ x ∧ x ≤ -2 → ∃ ε, ε > 0 ∧ ∀ y, x ≤ y ∧ y ≤ x + ε → f y ≤ f x) ∧
  (∃ m, 0 < m ∧ m < 1 ∧ (∑ x in {x | f x = m}, x = -8)) ∧ ¬(∀ x, f (4 - x) = -f x) 
  :=
by
  sorry

end student_conclusions_l538_538832


namespace lengths_equality_l538_538230

variable {A B C D E : Type} [EuclideanGeometry A B C] [TriangleABC : triangle ℝ A B C]
variable {ω : circle A B C} -- Incircle
variable {ω_A : circle A B C} -- A-excircle
variable {γ_B : circle A B} -- Circle passing through B, tangent to ω and ω_A
variable {γ_C : circle A C} -- Circle passing through C, tangent to ω and ω_A

-- Given γ_B passing through B and tangent to ω, ω_A
axiom γ_B_properties : passes_through γ_B B ∧ tangent_to γ_B ω ∧ tangent_to γ_B ω_A

-- Given γ_C passing through C and tangent to ω, ω_A
axiom γ_C_properties : passes_through γ_C C ∧ tangent_to γ_C ω ∧ tangent_to γ_C ω_A

-- Intersection points D and E
axiom D_point : intersection_point γ_B (line B C) = D
axiom E_point : intersection_point γ_C (line B C) = E

-- Goal: Prove BD = EC
theorem lengths_equality : length (segment B D) = length (segment E C) := sorry

end lengths_equality_l538_538230


namespace common_difference_of_arithmetic_sequence_l538_538096

noncomputable def smallest_angle : ℝ := 25
noncomputable def largest_angle : ℝ := 105
noncomputable def num_angles : ℕ := 5

theorem common_difference_of_arithmetic_sequence :
  ∃ d : ℝ, (smallest_angle + (num_angles - 1) * d = largest_angle) ∧ d = 20 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l538_538096


namespace problem_condition_l538_538503

noncomputable def f (x b : ℝ) := Real.exp x * (x - b)
noncomputable def f_prime (x b : ℝ) := Real.exp x * (x - b + 1)
noncomputable def g (x : ℝ) := (x^2 + 2*x) / (x + 1)

theorem problem_condition (b : ℝ) :
  (∃ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x b + x * f_prime x b > 0) → b < 8 / 3 :=
by
  sorry

end problem_condition_l538_538503


namespace gamma_minus_alpha_l538_538040

theorem gamma_minus_alpha (α β γ : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < γ) (h4 : γ < 2 * Real.pi)
    (h5 : ∀ x : ℝ, Real.cos (x + α) + Real.cos (x + β) + Real.cos (x + γ) = 0) : 
    γ - α = (4 * Real.pi) / 3 :=
sorry

end gamma_minus_alpha_l538_538040


namespace projection_on_angle_bisector_l538_538130

-- Definitions for the conditions stated.
variables {M : Type*} [Point M]
variables {m n : Line M}
variables {plane_of_mn : Plane M}

-- Assumptions and Conditions
axiom equidistant_lines (M : Point M) (m n : Line M) : point_eq_distance M m n
axiom projection_plane (M : Point M) (plane_of_mn : Plane M) : orthogonal_projection M plane_of_mn

-- Proof Statement: M_1 lies on the bisector of one of the angles formed by lines m and n.
theorem projection_on_angle_bisector (M : Point M) (m n : Line M) (plane_of_mn : Plane M) :
  (equidistant_lines M m n) ∧ (projection_plane M plane_of_mn) ->
  orthogonal_projection M plane_of_mn ∈ angle_bisector m n :=
sorry

end projection_on_angle_bisector_l538_538130


namespace find_cost_price_of_bicycle_for_A_l538_538253

-- Define variables and constants
variables (CP_A SP_B SP_C: ℝ)

-- Conditions
def condition1 := SP_B = 1.35 * CP_A
def condition2 := SP_C = 1.45 * SP_B
def final_sp := SP_C = 225

-- Theorem statement
theorem find_cost_price_of_bicycle_for_A :
  condition1 ∧ condition2 ∧ final_sp → CP_A = 114.94 :=
begin
  intro h, -- introduce the hypotheses
  -- various steps go here
  sorry -- placeholder for the actual proof
end

end find_cost_price_of_bicycle_for_A_l538_538253


namespace increasing_functions_l538_538786

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2, x1 < x2 → f x1 < f x2

def f1 (x : ℝ) : ℝ := 2 * x
def f2 (x : ℝ) : ℝ := -x + 1
def f3 (x : ℝ) : ℝ := x^2
def f4 (x : ℝ) : ℝ := -1 / x

theorem increasing_functions :
  is_increasing f1 ∧ is_increasing (λ x : ℝ, f3 x) (x > 0) ∧ ¬ is_increasing f2 ∧ ¬ is_increasing f4 :=
by
  sorry

end increasing_functions_l538_538786


namespace parallelogram_perimeter_area_sum_l538_538431

-- Define the vertices
def A := (1 : ℝ, 3 : ℝ)
def B := (5 : ℝ, 3 : ℝ)
def C := (7 : ℝ, 8 : ℝ)
def D := (3 : ℝ, 8 : ℝ)

-- Define the lengths of sides calculated from the given vertices
def side1 := (B.1 - A.1)
def side2 := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)

-- Calculate perimeter of the parallelogram
def p := side1 + side2 + side1 + side2

-- Calculate area of the parallelogram
def a := side1 * (D.2 - A.2)

-- Define the sum of perimeter and area
def sum_p_a := p + a

-- Theorem to prove
theorem parallelogram_perimeter_area_sum : 
  sum_p_a = 28 + 2 * (Real.sqrt 29) :=
by
  -- Proof omitted as per instructions.
  sorry

end parallelogram_perimeter_area_sum_l538_538431


namespace determine_function_l538_538573

open Nat

theorem determine_function (f : ℕ × ℕ → ℕ)
  (h₁ : ∀ (a b c : ℕ), f(gcd a b, c) = gcd a (f(c, b)))
  (h₂ : ∀ a : ℕ, f(a, a) ≥ a) :
  ∀ (a b : ℕ), f(a, b) = gcd a b := 
by
  sorry

end determine_function_l538_538573


namespace probability_male_monday_female_tuesday_l538_538013

structure Volunteers where
  men : ℕ
  women : ℕ
  total : ℕ

def group : Volunteers := {men := 2, women := 2, total := 4}

def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)
def combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_male_monday_female_tuesday :
  let n := permutations group.total 2
  let m := combinations group.men 1 * combinations group.women 1
  (m / n : ℚ) = 1 / 3 :=
by
  sorry

end probability_male_monday_female_tuesday_l538_538013


namespace geometric_sequence_sum_l538_538598

-- We state the main problem in Lean as a theorem.
theorem geometric_sequence_sum (S : ℕ → ℕ) (S_4_eq : S 4 = 8) (S_8_eq : S 8 = 24) : S 12 = 88 :=
  sorry

end geometric_sequence_sum_l538_538598


namespace gcd_lcm_relationship_l538_538157

-- Definitions of GCD and LCM for positive integers
def gcd (a b : ℕ) : ℕ := nat.gcd a b
def lcm (a b : ℕ) : ℕ := nat.lcm a b
def gcd3 (a b c : ℕ) : ℕ := gcd (gcd a b) c
def lcm3 (a b c : ℕ) : ℕ := lcm (lcm a b) c

-- Theorem statement
theorem gcd_lcm_relationship (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((lcm3 a b c) ^ 2 / (lcm a b * lcm b c * lcm c a) : ℚ) = 
  ((gcd3 a b c) ^ 2 / (gcd a b * gcd b c * gcd c a) : ℚ) := by
  sorry

end gcd_lcm_relationship_l538_538157


namespace april_earnings_l538_538761

namespace AprilFlowers

-- Define the prices of the flowers in the morning and afternoon
def morning_price (flower : String) : Int :=
  if flower = "roses" then 4 else
  if flower = "tulips" then 3 else
  if flower = "daisies" then 2 else 0

def afternoon_price (flower : String) : Int :=
  if flower = "roses" then 5 else
  if flower = "tulips" then 4 else
  if flower = "daisies" then 3 else 0

-- Define the initial and remaining counts of flowers
def initial_count (flower : String) : Int :=
  if flower = "roses" then 13 else
  if flower = "tulips" then 10 else
  if flower = "daisies" then 8 else 0

def morning_remaining (flower : String) : Int :=
  if flower = "roses" then 7 else
  if flower = "tulips" then 6 else
  if flower = "daisies" then 4 else 0

def end_of_day_remaining (flower : String) : Int :=
  if flower = "roses" then 4 else
  if flower = "tulips" then 3 else
  if flower = "daisies" then 1 else 0

-- Define the bulk discount and sales tax
def bulk_discount := 0.15
def sales_tax := 0.10

-- Calculate the number of flowers sold
def sold_in_morning (flower : String) : Int :=
  initial_count flower - morning_remaining flower

def sold_in_afternoon (flower : String) : Int :=
  morning_remaining flower - end_of_day_remaining flower

-- Define the proof problem statement
theorem april_earnings : 
  let morning_earnings := 
      (sold_in_morning "roses") * (morning_price "roses") + 
      (sold_in_morning "tulips") * (morning_price "tulips") + 
      (sold_in_morning "daisies") * (morning_price "daisies")
  let afternoon_earnings := 
      (sold_in_afternoon "roses") * (afternoon_price "roses") + 
      (sold_in_afternoon "tulips") * (afternoon_price "tulips") + 
      (sold_in_afternoon "daisies") * (afternoon_price "daisies")
  let total_earnings_before_discount := morning_earnings + afternoon_earnings
  let total_earnings := 
      if (sold_in_afternoon "roses" >= 5) 
      then total_earnings_before_discount - (bulk_discount * (sold_in_afternoon "roses" * afternoon_price "roses"))
      else total_earnings_before_discount
  let earnings_after_tax := total_earnings / (1 + sales_tax) 
  earnings_after_tax ≈ 72.73 := by sorry

end AprilFlowers

end april_earnings_l538_538761


namespace grasshopper_avoids_set_l538_538572

theorem grasshopper_avoids_set (n : ℕ) (a : Fin n → ℕ) (M : Finset ℕ) (h_distinct : Function.Injective a)
  (h_card_M : M.card = n - 1) (h_pos_a : ∀ i, 0 < a i)
  (h_not_in_M : (Finset.univ.sum a) ∉ M) :
  ∃ σ : Fin n → Fin n, ∀ k ≤ n, (Finset.range k).sum (a ∘ σ) ∉ M :=
sorry

end grasshopper_avoids_set_l538_538572


namespace triangle_side_difference_l538_538928

theorem triangle_side_difference :
  ∀ x : ℝ, (x > 2 ∧ x < 18) → ∃ a b : ℤ, (∀ y : ℤ, y ∈ set.Icc a b → (y : ℝ) = x) ∧ (b - a = 14) :=
by
  assume x hx,
  sorry

end triangle_side_difference_l538_538928


namespace area_of_wrapping_paper_l538_538295

-- Define variables and constants
variables (w h x : ℝ)

-- Define the main theorem
theorem area_of_wrapping_paper (w h x : ℝ) : 
  let s := real.sqrt ((h + x)^2 + (w / 2)^2) in
  (h + x)^2 + (w^2 / 4) = s^2 :=
by 
  sorry

end area_of_wrapping_paper_l538_538295


namespace find_initial_coins_l538_538744

def has_initial_coins (x y : ℕ) : Prop :=
  20 * x + 15 * y = 150 ∧ x > y

theorem find_initial_coins :
  ∃ x y : ℕ, has_initial_coins x y ∧ x = 6 ∧ y = 2 :=
by {
  use [6, 2],
  simp [has_initial_coins],
  split, { norm_num },
  exact six_gt_two,
}

end find_initial_coins_l538_538744


namespace find_m_l538_538866

-- We start by defining the vectors and the condition that 2a - b is perpendicular to c.
variable (m : ℝ)
def a : ℝ × ℝ := (m, 2)
def b : ℝ × ℝ := (1, 1)
def c : ℝ × ℝ := (1, 3)
def two_a_minus_b : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)
def perp : Prop := two_a_minus_b.1 * c.1 + two_a_minus_b.2 * c.2 = 0

-- The theorem we want to prove.
theorem find_m : perp m → m = -4 := by
  sorry

end find_m_l538_538866


namespace Walter_age_in_2010_l538_538383

-- Define Walter's age in 2005 as y
def Walter_age_2005 (y : ℕ) : Prop :=
  (2005 - y) + (2005 - 3 * y) = 3858

-- Define Walter's age in 2010
theorem Walter_age_in_2010 (y : ℕ) (hy : Walter_age_2005 y) : y + 5 = 43 :=
by
  sorry

end Walter_age_in_2010_l538_538383


namespace evaluate_g_at_neg_four_l538_538881

def g (x : ℤ) : ℤ := 5 * x + 2

theorem evaluate_g_at_neg_four : g (-4) = -18 := 
by 
  sorry

end evaluate_g_at_neg_four_l538_538881


namespace problem1_problem2_problem3_problem4_l538_538047

-- Omega function definition
def isOmega (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T ≠ 0 ∧ ∀ x : ℝ, f(x) = T * f(x + T)

-- Even function definition
def isEven (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = f(x)

-- Problem 1: F(x) = x is not an Omega function; h(x) = sin(πx) is an Omega function
theorem problem1 :
  ¬ (∃ T, isOmega (fun x => x) T) ∧ (∃ T, isOmega (fun x => Real.sin (Real.pi * x)) T) :=
sorry

-- Problem 2: if f(x) is Omega function and even, then f(x) is periodic with period 2T
theorem problem2 (f : ℝ → ℝ) (T : ℝ)
  (h_Omega : isOmega f T) (h_even : isEven f) : ∀ x, f(x) = f(x + 2 * T) :=
sorry

-- Problem 3: For f(x) = sin(kx) to be an Omega function, k = tπ for t in Z.
theorem problem3 (k T : ℝ) : isOmega (fun x => Real.sin (k * x)) T ↔ ∃ t : ℤ, k = (t * Real.pi) :=
sorry

-- Problem 4: Define a function g(x) that meets all the given properties
def g (x : ℝ) : ℝ := 3 ^ x * Real.sin (2 * Real.pi * x)

theorem problem4 :
  (∃ x0 : ℝ, g x0 ≠ 0) ∧ (∀ x : ℝ, g(x + 2) = 9 * g(x)) ∧ (∃ x y : ℝ, x < y ∧ g x > g y ∧ continuous g) :=
sorry

end problem1_problem2_problem3_problem4_l538_538047


namespace printer_cost_l538_538287

theorem printer_cost (num_keyboards : ℕ) (num_printers : ℕ) (total_cost : ℕ) (keyboard_cost : ℕ) (printer_cost : ℕ) :
  num_keyboards = 15 →
  num_printers = 25 →
  total_cost = 2050 →
  keyboard_cost = 20 →
  (total_cost - (num_keyboards * keyboard_cost)) / num_printers = printer_cost →
  printer_cost = 70 :=
by
  sorry

end printer_cost_l538_538287


namespace mark_birth_year_proof_l538_538932

-- Conditions
def current_year := 2021
def janice_age := 21
def graham_age := 2 * janice_age
def mark_age := graham_age + 3
def mark_birth_year (current_year : ℕ) (mark_age : ℕ) := current_year - mark_age

-- Statement to prove
theorem mark_birth_year_proof : 
  mark_birth_year current_year mark_age = 1976 := by
  sorry

end mark_birth_year_proof_l538_538932


namespace poly_roots_arith_progression_l538_538699

theorem poly_roots_arith_progression (a b c : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, -- There exist roots x₁, x₂, x₃
    (x₁ + x₃ = 2 * x₂) ∧ -- Roots form an arithmetic progression
    (x₁ * x₂ * x₃ = -c) ∧ -- Roots satisfy polynomial's product condition
    (x₁ + x₂ + x₃ = -a) ∧ -- Roots satisfy polynomial's sum condition
    ((x₁ * x₂) + (x₂ * x₃) + (x₃ * x₁) = b)) -- Roots satisfy polynomial's sum of products condition
  → (2 * a^3 / 27 - a * b / 3 + c = 0) := 
sorry -- proof is not required

end poly_roots_arith_progression_l538_538699


namespace minimum_value_expression_l538_538582

-- Define the conditions
variables (α β : ℝ)

-- Define the function to minimize
def expression (α β : ℝ) : ℝ := (3 * real.cos α + 4 * real.sin β - 10)^2 + (3 * real.sin α + 4 * real.cos β - 18)^2

-- Define the theorem stating the minimum value
theorem minimum_value_expression : ∃ (m : ℝ), m = 169 ∧ ∀ (α β : ℝ), expression α β ≥ m :=
by
  use 169
  intros α β
  sorry

end minimum_value_expression_l538_538582


namespace remainder_of_8_pow_2048_mod_50_l538_538702

theorem remainder_of_8_pow_2048_mod_50 :
  8^2048 % 50 = 38 :=
by {
  have h1 : 8^2 % 50 = 14, by norm_num,
  have h2 : 8^3 % 50 = 12, by norm_num,
  have h3 : 8^4 % 50 = 46, by norm_num,
  have h4 : 8^5 % 50 = 18, by norm_num,
  have h5 : 8^6 % 50 = 44, by norm_num,
  have h6 : 8^7 % 50 = 2, by norm_num,
  have h7 : 8^8 % 50 = 16, by norm_num,
  have h10 : 8^10 % 50 = 24, by norm_num,
  sorry
}

end remainder_of_8_pow_2048_mod_50_l538_538702


namespace Alec_goal_ratio_l538_538311

theorem Alec_goal_ratio (total_students half_votes thinking_votes more_needed fifth_votes : ℕ)
  (h_class : total_students = 60)
  (h_half : half_votes = total_students / 2)
  (remaining_students : ℕ := total_students - half_votes)
  (h_thinking : thinking_votes = 5)
  (h_fifth : fifth_votes = (remaining_students - thinking_votes) / 5)
  (h_current_votes : half_votes + thinking_votes + fifth_votes = 40)
  (h_needed : more_needed = 5)
  :
  (half_votes + thinking_votes + fifth_votes + more_needed) / total_students = 3 / 4 :=
by sorry

end Alec_goal_ratio_l538_538311


namespace tiling_impossible_with_2_and_3_unit_squares_l538_538931

theorem tiling_impossible_with_2_and_3_unit_squares 
  (size: ℕ) 
  (smallsquare1: ℕ) 
  (smallsquare2: ℕ)
  (checkerboard: ∀ (i j : ℕ), i < size ∧ j < size → ℕ)
  : (size = 101) ∧ (smallsquare1 = 2) ∧ (smallsquare2 = 3) → 
    ¬ (∀ (x y : ℕ), x ≤ 101 ∧ y ≤ 101 → 
      (∃ (a b : ℕ), a*smallsquare1 + b*smallsquare2 = size))
:=
begin
  intros h,
  sorry
end

end tiling_impossible_with_2_and_3_unit_squares_l538_538931


namespace arithmetic_sequence_m_value_l538_538905

theorem arithmetic_sequence_m_value (m : ℝ) (h : 2 + 6 = 2 * m) : m = 4 :=
by sorry

end arithmetic_sequence_m_value_l538_538905


namespace more_stable_shooting_performance_l538_538764

theorem more_stable_shooting_performance :
  ∀ (SA2 SB2 : ℝ), SA2 = 1.9 → SB2 = 3 → (SA2 < SB2) → "A" = "Athlete with more stable shooting performance" :=
by
  intros SA2 SB2 h1 h2 h3
  sorry

end more_stable_shooting_performance_l538_538764


namespace letters_into_mailboxes_l538_538690

theorem letters_into_mailboxes : 
  (number_of_letters : ℕ) (number_of_mailboxes : ℕ) 
  (number_of_letters = 4) (number_of_mailboxes = 3) :
  (number_of_mailboxes^number_of_letters = 81) :=
sorry

end letters_into_mailboxes_l538_538690


namespace john_days_to_lose_weight_l538_538567

noncomputable def john_calories_intake : ℕ := 1800
noncomputable def john_calories_burned : ℕ := 2300
noncomputable def calories_to_lose_1_pound : ℕ := 4000
noncomputable def pounds_to_lose : ℕ := 10

theorem john_days_to_lose_weight :
  (john_calories_burned - john_calories_intake) * (pounds_to_lose * calories_to_lose_1_pound / (john_calories_burned - john_calories_intake)) = 80 :=
by
  sorry

end john_days_to_lose_weight_l538_538567


namespace expression_value_l538_538703

theorem expression_value : (4 - 2) ^ 3 = 8 :=
by sorry

end expression_value_l538_538703


namespace electric_field_at_center_of_quarter_circle_uniformly_charged_rod_l538_538307

-- Define the conditions of the problem
variables (σ R : ℝ) (ε₀ : ℝ) [fact (ε₀ > 0)] [fact (R > 0)]

-- The proposition to be proved
theorem electric_field_at_center_of_quarter_circle_uniformly_charged_rod :
  (σ / (2 * π * ε₀ * R)) = E := sorry

end electric_field_at_center_of_quarter_circle_uniformly_charged_rod_l538_538307


namespace sqrt_domain_condition_l538_538517

theorem sqrt_domain_condition (x : ℝ) : (2 * x - 6 ≥ 0) ↔ (x ≥ 3) :=
by
  sorry

end sqrt_domain_condition_l538_538517


namespace fourth_quadrant_point_l538_538845

theorem fourth_quadrant_point (a : ℤ) (h1 : 2 * a + 6 > 0) (h2 : 3 * a + 3 < 0) :
  (2 * a + 6, 3 * a + 3) = (2, -3) :=
sorry

end fourth_quadrant_point_l538_538845


namespace five_digit_odd_and_multiples_of_5_sum_l538_538577

theorem five_digit_odd_and_multiples_of_5_sum :
  let A := 9 * 10^3 * 5
  let B := 9 * 10^3 * 1
  A + B = 45000 := by
sorry

end five_digit_odd_and_multiples_of_5_sum_l538_538577


namespace triangle_ratio_l538_538929

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) (hA : A = 60) (hb : b = 1) (h_area : sqrt 3 = (1/2 * b * c * sin A)) :
  (a + b + c) / (sin A + sin B + sin C) = 2 * sqrt 39 / 3 :=
by sorry

end triangle_ratio_l538_538929


namespace find_m_l538_538867

def vec2 := ℝ × ℝ

def orthogonal (u v : vec2) : Prop :=
u.1 * v.1 + u.2 * v.2 = 0

variables (a b c : vec2) (m : ℝ)

theorem find_m (h1 : a = (m, 2))
                (h2 : b = (1, 1))
                (h3 : c = (1, 3))
                (h4 : orthogonal (2 a - b) c) : m = -4 :=
sorry

end find_m_l538_538867


namespace circle_tangent_geom_mean_l538_538231

open Real

variables {O : Point} {R : ℝ}
variables {A B M : Point}
variables (AM BM : ℝ)
variables [Circle O R]
variables [Tangent A M O] [Tangent M B O]
variables [Parallel_Tangents A B]

-- Definition of the Power of a Point theorem might be necessary for the full proof.

theorem circle_tangent_geom_mean
  (h1 : dist O A = R) -- Tangent condition at A
  (h2 : dist O B = R) -- Tangent condition at B
  (h3 : dist O M = R) -- Tangent condition at M
  (h4 : AM * BM = R^2) -- Application of Power of a Point theorem or geometric mean directly

  : R = sqrt (AM * BM) :=
by sorry

end circle_tangent_geom_mean_l538_538231


namespace probability_of_3_correct_answers_is_31_over_135_expected_value_of_total_score_is_50_l538_538113

noncomputable def probability_correct_answers : ℚ :=
  let pA := (1/5 : ℚ)
  let pB := (3/5 : ℚ)
  let pC := (1/5 : ℚ)
  ((pA * (3/9 : ℚ) * (2/3)^2 * (1/3)) + (pB * (6/9 : ℚ) * (2/3) * (1/3)^2) + (pC * (1/9 : ℚ) * (1/3)^3))

theorem probability_of_3_correct_answers_is_31_over_135 :
  probability_correct_answers = 31 / 135 := by
  sorry

noncomputable def expected_score : ℚ :=
  let E_m := (1/5 * 1 + 3/5 * 2 + 1/5 * 3 : ℚ)
  let E_n := (3 * (2/3 : ℚ))
  (15 * E_m + 10 * E_n)

theorem expected_value_of_total_score_is_50 :
  expected_score = 50 := by
  sorry

end probability_of_3_correct_answers_is_31_over_135_expected_value_of_total_score_is_50_l538_538113


namespace probability_fav_song_not_heard_within_first_5_minutes_l538_538771

-- Define the lengths of the 12 songs on Celeste's MP3 player
def song_lengths : List ℕ := [30, 70, 110, 150, 190, 230, 270, 310, 350, 390, 430, 470]

-- The index of the favorite song, in 0-based indexing, would be 10 since it's 430s
def fav_song_index : ℕ := 10

-- The first 5 minutes in seconds
def five_minutes : ℕ := 300

-- Define the problem statement
theorem probability_fav_song_not_heard_within_first_5_minutes : 
  ∀ (arrangement : List ℕ), 
    arrangement.perm song_lengths →
    (∑ i in (arrangement.takeWhile (λ len, len ≤ 300)), id i) < 430 →
    1 = 1 :=
by
  intros 
  sorry


end probability_fav_song_not_heard_within_first_5_minutes_l538_538771


namespace total_cost_calculation_l538_538602

-- Definitions
def coffee_price : ℕ := 4
def cake_price : ℕ := 7
def ice_cream_price : ℕ := 3

def mell_coffee_qty : ℕ := 2
def mell_cake_qty : ℕ := 1
def friends_coffee_qty : ℕ := 2
def friends_cake_qty : ℕ := 1
def friends_ice_cream_qty : ℕ := 1

def total_coffee_qty : ℕ := 3 * mell_coffee_qty
def total_cake_qty : ℕ := 3 * mell_cake_qty
def total_ice_cream_qty : ℕ := 2 * friends_ice_cream_qty

def total_cost : ℕ := total_coffee_qty * coffee_price + total_cake_qty * cake_price + total_ice_cream_qty * ice_cream_price

-- Theorem Statement
theorem total_cost_calculation : total_cost = 51 := by
  sorry

end total_cost_calculation_l538_538602


namespace sin_690_eq_neg_half_l538_538407

theorem sin_690_eq_neg_half :
  Real.sin (690 * Real.pi / 180) = -1 / 2 :=
by {
  sorry
}

end sin_690_eq_neg_half_l538_538407


namespace amoeba_doubling_time_l538_538202

theorem amoeba_doubling_time (H1 : ∀ t : ℕ, t = 60 → 2^(t / 3) = 2^20) :
  ∀ t : ℕ, 2 * 2^(t / 3) = 2^20 → t = 57 :=
by
  intro t
  intro H2
  sorry

end amoeba_doubling_time_l538_538202


namespace ava_planted_more_trees_l538_538384

theorem ava_planted_more_trees (L : ℕ) (h1 : 9 + L = 15) : 9 - L = 3 := 
by
  sorry

end ava_planted_more_trees_l538_538384


namespace degree_of_monomial_l538_538998

def degree (m : String) : Nat :=  -- Placeholder type, replace with appropriate type that represents a monomial
  sorry  -- Logic to compute the degree would go here, if required for full implementation

theorem degree_of_monomial : degree "-(3/5) * a * b^2" = 3 := by
  sorry

end degree_of_monomial_l538_538998


namespace jack_typing_time_l538_538569

def typing_problem : Prop :=
  ∃ (J : ℝ), 40 / J = (40 / 40) + (40 / 30) + (40 / 10) ∧ J = 24

theorem jack_typing_time : typing_problem :=
by
  use 24
  split
  calc
    40 / 24 = 40 * (1 / 24) : by rw div_eq_mul_inv
    ... = 40 * (1 / 24) : rfl
    ... = (40 / 40) + (40 / 30) + (40 / 10) : by calc
        1 + 4 / 3 + 40 / 24 = _
        sorry
  rfl

end jack_typing_time_l538_538569


namespace rectangle_diagonals_equal_l538_538160

theorem rectangle_diagonals_equal {A B C D : Type} [is_rectangle A B C D] (h : diagonal_length A C = 8) : diagonal_length B D = 8 :=
sorry

end rectangle_diagonals_equal_l538_538160


namespace min_distance_squared_to_origin_on_line_l538_538476

theorem min_distance_squared_to_origin_on_line :
  (∃ x y : ℝ, x + y = 4 ∧ ∀ x' y' : ℝ, (x' + y' = 4 → x^2 + y^2 ≤ x'^2 + y'^2)) → (x^2 + y^2 = 8) :=
begin
  sorry
end

end min_distance_squared_to_origin_on_line_l538_538476


namespace sin_690_degree_l538_538422

theorem sin_690_degree : sin (690 : ℝ) * (Real.pi / 180) = -(1 / 2) := by
  sorry

end sin_690_degree_l538_538422


namespace normal_force_ratio_l538_538523

theorem normal_force_ratio
  (m R ω g : ℝ)
  (hR : R > 0)
  (hω : ω > 0)
  (hg : g > 0) :
  let F_N1 := m * g
  let F_N2 := m * g - m * ω^2 * R
  (F_N2 / F_N1) = 1 - (R * ω^2 / g) :=
by
  unfold F_N1 F_N2
  sorry

end normal_force_ratio_l538_538523


namespace euler_disproven_conjecture_solution_l538_538620

theorem euler_disproven_conjecture_solution : 
  ∃ (n : ℕ), n^5 = 133^5 + 110^5 + 84^5 + 27^5 ∧ n = 144 :=
by
  use 144
  have h : 144^5 = 133^5 + 110^5 + 84^5 + 27^5 := sorry
  exact ⟨h, rfl⟩

end euler_disproven_conjecture_solution_l538_538620


namespace dot_product_eq_neg_two_l538_538043

variable {α : Type*} [InnerProductSpace ℝ α]

def cos_theta (a b : α) : ℝ := 1/4
def norm_a : ℝ := 2
def norm_b : ℝ := 4

theorem dot_product_eq_neg_two (a b : α) 
  (h_cos : cos_theta a b = 1/4)
  (h_norm_a : ‖a‖ = norm_a)
  (h_norm_b : ‖b‖ = norm_b) :
  ⟪a, b - a⟫ = -2 :=
by
  sorry

end dot_product_eq_neg_two_l538_538043


namespace identify_tricksters_in_30_or_less_questions_l538_538357

-- Define the problem parameters
def inhabitants : Type := Fin 65

def is_knight (inhabitant : inhabitants) : Prop := sorry
def is_trickster (inhabitant : inhabitants) : Prop := sorry

-- Define the properties
axiom knight_truthful : ∀ (x : inhabitants), is_knight x → (forall y : inhabitants, True ↔ (is_knight y = x is_knight y))
axiom trickster_mixed : ∀ (x : inhabitants), is_trickster x → ((∀ y : inhabitants, True) ∨ (∃ y : inhabitants, y ∉ (is_knight y)))

-- Problem statement
theorem identify_tricksters_in_30_or_less_questions
  (inhabitants : Type)
  (n_tricksters : ℕ := 2) -- 2 tricksters
  (total_inhabitants : ℕ := 65) -- 65 total inhabitants
  (questions_limit : ℕ := 30) -- limit of 30 questions
  (knights : inhabitants → Prop)
  (tricksters : inhabitants → Prop) :
    ∃ (solution_exists : ∀ (is_trickster : inhabitants → Prop), ∃ k : inhabitants, (knights k) ∧ (is_trickster k)) 
    (possible_to_find_tricksters : ∀ (is_knight : inhabitants → Prop) (is_trickster : inhabitants → Prop), 
    ∃ (questions_used ≤ questions_limit), ∀ (xs : set inhabitants), questions_used ≤ 30 ∧ 
    (∃ trickster1 trickster2 : inhabitants, (tricksters trickster1 ∧ tricksters trickster2 ∧ trickster1 ≠ trickster2))) :=
sorry

end identify_tricksters_in_30_or_less_questions_l538_538357


namespace unique_n_divisors_satisfies_condition_l538_538151

theorem unique_n_divisors_satisfies_condition:
  ∃ (n : ℕ), (∃ d1 d2 d3 : ℕ, d1 = 1 ∧ d2 > d1 ∧ d3 > d2 ∧ n = d3 ∧
  n = d2^2 + d3^3) ∧ n = 68 := by
  sorry

end unique_n_divisors_satisfies_condition_l538_538151


namespace divide_green_area_eq_parts_l538_538209

theorem divide_green_area_eq_parts (ABCD MNPQ : Type) [rectangle ABCD] [rectangle MNPQ] (inside : is_inside MNPQ ABCD) :
    ∃ l : line, line_through_centers ABCD MNPQ l ∧ divides_green_area_eq_parts ABCD MNPQ l :=
sorry

end divide_green_area_eq_parts_l538_538209


namespace area_of_square_l538_538915

theorem area_of_square
    {A B C D P Q : Type} [square : parallelogram (A B C D)]:
    ∃ (R : Type),
    R ∈ (segment (B P) ∩ segment (C Q)) ∧
    angle (B R C) = π / 2 ∧
    dist B R = 5 ∧
    dist P R = 9 →
    area (square) = 196 := sorry

end area_of_square_l538_538915


namespace susie_rooms_l538_538221

theorem susie_rooms
  (house_vacuum_time_hours : ℕ)
  (room_vacuum_time_minutes : ℕ)
  (total_vacuum_time_minutes : ℕ)
  (total_vacuum_time_computed : house_vacuum_time_hours * 60 = total_vacuum_time_minutes)
  (rooms_count : ℕ)
  (rooms_count_computed : total_vacuum_time_minutes / room_vacuum_time_minutes = rooms_count) :
  house_vacuum_time_hours = 2 →
  room_vacuum_time_minutes = 20 →
  rooms_count = 6 :=
by
  intros h1 h2
  sorry

end susie_rooms_l538_538221


namespace common_factor_is_n_plus_1_l538_538995

def polynomial1 (n : ℕ) : ℕ := n^2 - 1
def polynomial2 (n : ℕ) : ℕ := n^2 + n

theorem common_factor_is_n_plus_1 (n : ℕ) : 
  ∃ (d : ℕ), d ∣ polynomial1 n ∧ d ∣ polynomial2 n ∧ d = n + 1 := by
  sorry

end common_factor_is_n_plus_1_l538_538995


namespace units_digit_product_l538_538455

theorem units_digit_product : 
  ( ∃ (n1 n2 n3 : ℕ), 
    n1 = 17 ∧ 
    n2 = 59 ∧ 
    n3 = 23 ∧ 
    ((n1 * n2) * n3 % 10) = 9 ) := 
begin
  use [17, 59, 23],
  split; try {refl},
  split; try {refl},
  split; try {refl},
  change ( (17 * 59) * 23 % 10 = 9),
  sorry
end

end units_digit_product_l538_538455


namespace number_of_valid_a_l538_538901

theorem number_of_valid_a :
  (∃ a : ℤ, 
    (∃ x : ℕ, (x + a) / (x - 2) + (2 * x) / (2 - x) = 1) ∧
    (∃ s : Finset ℤ, s.card = 4 ∧ ∀ y ∈ s, 
      (y + 1) / 5 ≥ y / 2 - 1 ∧ 
      y + a < 11 * y - 3)
  ) = 4 := sorry

end number_of_valid_a_l538_538901


namespace part1_part2_l538_538500

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2
noncomputable def g (x a : ℝ) : ℝ := x * Real.exp x - (a - 1) * x^2 - x - 2 * Real.log x

theorem part1 : ∃ xₘ : ℝ, (∀ x > 0, f x ≤ f xₘ) ∧ f xₘ = -1 :=
by sorry

theorem part2 (a : ℝ) : (∀ x > 0, f x + g x a ≥ 0) ↔ a ≤ 1 :=
by sorry

end part1_part2_l538_538500


namespace no_arithmetic_filling_possible_l538_538263

-- Define the initial grid as a 4x4 matrix with given numbers.
def initial_grid : Matrix (Fin 4) (Fin 4) (Option ℝ) :=
  ![![none, some 9, none, none],
    ![some 1, none, none, none],
    ![none, none, none, some 5],
    ![none, none, some 8, none]]

-- Define the problem statement as a Lean theorem.
theorem no_arithmetic_filling_possible :
  ¬(∃ (grid : Matrix (Fin 4) (Fin 4) ℝ),
      ∀ i j : Fin 4,
      (initial_grid i j).isSome → grid i j = (initial_grid i j).getOrElse 0 ∧
      (∀ k : Fin 4, 
        (grid i (i + k)) = grid i i + k * ((grid i (i + 1)) - grid i i) ∧
        (grid (i + k) j) = grid i j + k * ((grid (i + 1) j) - grid i j))) :=
begin
  sorry
end

end no_arithmetic_filling_possible_l538_538263


namespace area_ratio_l538_538695

/-- 
Prove that the ratio of the area of region R to the area of the square ABCD is 1/2.
-/
theorem area_ratio (A B C D M : Point) (speed : ℝ) (P1 P2 : ℕ → Point)
  (h_A : A = (0, 0)) (h_B : B = (2, 0)) (h_C : C = (2, 2)) (h_D : D = (0, 2))
  (h_M : M = (1, 2)) (h_speed : speed > 0)
  (start_P1 : P1 0 = A) (start_P2 : P2 0 = M)
  (path_P1 : ∀ t : ℕ, if t % 8 = 0 then P1(t+1) = P1(t) + (2, 0)  
                      else if t % 8 = 2 then P1(t+1) = P1(t) + (0, 2)
                      else if t % 8 = 4 then P1(t+1) = P1(t) + (-2, 0) 
                      else if t % 8 = 6 then P1(t+1) = P1(t) + (0, -2)
                      else sorry)
  (path_P2 : ∀ t : ℕ, if t % 8 = 0 then P2(t+1) = P2(t) + (-2, 0)  
                      else if t % 8 = 2 then P2(t+1) = P2(t) + (0, -2)
                      else if t % 8 = 4 then P2(t+1) = P2(t) + (2, 0) 
                      else if t % 8 = 6 then P2(t+1) = P2(t) + (0, 2)
                      else sorry) :
  let R := setPoint (midpoint (P1 t) (P2 t)) ∈ real_line 
  in (area R) / (area ABCD) = 1/2 :=
sorry

end area_ratio_l538_538695


namespace translate_non_intersecting_sets_l538_538468

noncomputable def exists_non_intersecting_translations (S : Finset ℕ) (A : Finset ℕ): Prop :=
∃ (t: Fin ₙ → ℕ), 
(∀ i j: Fin ₙ, i ≠ j → (A.image (λ x, x + t i)) ∩ (A.image (λ x, x + t j)) = ∅)

theorem translate_non_intersecting_sets (A : Finset ℕ) (hA : A.card = 101) : 
  exists_non_intersecting_translations (Finset.range 1000000) A :=
begin
  sorry,
end

end translate_non_intersecting_sets_l538_538468


namespace total_seeds_eaten_l538_538381

theorem total_seeds_eaten :
  ∃ (first second third : ℕ), 
  first = 78 ∧ 
  second = 53 ∧ 
  third = second + 30 ∧ 
  first + second + third = 214 :=
by
  use 78, 53, 83
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end total_seeds_eaten_l538_538381


namespace smallest_z_value_l538_538550

theorem smallest_z_value :
  ∃ (w x y z : ℕ), w < x ∧ x < y ∧ y < z ∧
  w + 1 = x ∧ x + 1 = y ∧ y + 1 = z ∧
  w^3 + x^3 + y^3 = z^3 ∧ z = 6 := by
  sorry

end smallest_z_value_l538_538550


namespace total_trophies_l538_538934

theorem total_trophies (michael_now : ℕ) (increase : ℕ) (jack_multiplication : ℕ) :
  michael_now = 30 →
  increase = 100 →
  jack_multiplication = 10 →
  let michael_future := michael_now + increase in
  let jack_future := jack_multiplication * michael_now in
  michael_future + jack_future = 430 :=
by
  intros h_michael h_increase h_jack
  simp [h_michael, h_increase, h_jack]
  sorry

end total_trophies_l538_538934


namespace outer_wheel_circumference_l538_538271
noncomputable theory

def wheel_circumference (dist_between_wheels : ℝ) (speed_ratio : ℝ) : ℝ :=
  let r_inner := dist_between_wheels / (speed_ratio - 1)
  let r_outer := r_inner + dist_between_wheels
  let d_outer := 2 * r_outer
  π * d_outer

theorem outer_wheel_circumference
  (dist_between_wheels : ℝ) (speed_ratio : ℝ)
  (h_dist : dist_between_wheels = 1.5) 
  (h_speed : speed_ratio = 2) :
  wheel_circumference dist_between_wheels speed_ratio ≈ 18.85 :=
by
  sorry

end outer_wheel_circumference_l538_538271


namespace find_a_value_l538_538904

noncomputable def a_value (a : ℝ) : Prop :=
  let C1 := λ x : ℝ, a * x^3 - 6 * x^2 + 12 * x
  let C2 := λ x : ℝ, Real.exp x
  let slope_C1 := (3 * a * 1^2) - (12 * 1) + 12
  let slope_C2 := Real.exp 1
  slope_C1 = slope_C2 → a = (Real.exp 1) / 3

theorem find_a_value : ∃ a : ℝ, a_value a := by
  sorry

end find_a_value_l538_538904


namespace slope_angle_of_line_l538_538215

theorem slope_angle_of_line (a b c : ℝ) (f : ℝ → ℝ) 
  (h₁ : ∀ x, f x = a * sin x - b * cos x)
  (h₂ : ∀ x, f (π / 4 - x) = f (π / 4 + x)) :
  ∃ θ : ℝ, θ = 3 * π / 4 :=
by
  -- Let us ignore the proof for now.
  sorry

end slope_angle_of_line_l538_538215


namespace general_term_arithmetic_sequence_sum_of_reciprocal_sequence_b_l538_538487

-- Define the arithmetic sequence {a_n}
def arithmetic_sequence (a1 d : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d

-- Problem 1: Prove the general term formula a_n = 2n + 1 of the arithmetic sequence
theorem general_term_arithmetic_sequence {a1 d : ℕ} (h1 : a1 + (a1 + 4 * d) = (2 / 7) * (a1 + 2 * d) ^ 2) (h2 : 7 * (a1 + 3 * d) = 63) :
  ∃ (n : ℕ), arithmetic_sequence a1 d n = 2 * n + 1 :=
sorry

-- Define the sequence {b_n}
def sequence_b (a : ℕ → ℕ) (b1 : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then b1 else b1 + Sum (Finset.range n) (λ k, a (k + 1))

-- Problem 2: Prove the sum of the first n terms T_n of the sequence {1 / b_n}
theorem sum_of_reciprocal_sequence_b (a : ℕ → ℕ) (b : ℕ → ℕ) (h1 : ∀ n, a n = 2 * n + 1)
  (h2 : ∀ n, b (n + 1) - b n = a (n + 1)) (h3 : b 1 = a 1) :
  ∃ (Tn : ℕ → ℝ), Tn = λ n, (3 / 4) - (1 / 2) * (1 / (n + 1) + 1 / (n + 2)) :=
sorry

end general_term_arithmetic_sequence_sum_of_reciprocal_sequence_b_l538_538487


namespace intersection_set_l538_538507

-- Definition of the sets A and B
def setA : Set ℝ := { x | -2 < x ∧ x < 2 }
def setB : Set ℝ := { x | x < 0.5 }

-- The main theorem: Finding the intersection A ∩ B
theorem intersection_set : { x : ℝ | -2 < x ∧ x < 0.5 } = setA ∩ setB := by
  sorry

end intersection_set_l538_538507


namespace tangent_line_condition_l538_538667

theorem tangent_line_condition (a b k : ℝ) (h1 : (1 : ℝ) + a + b = 2) (h2 : 3 + a = k) (h3 : k = 1) :
    b - a = 5 := 
by 
    sorry

end tangent_line_condition_l538_538667


namespace circumradii_sum_geq_half_perimeter_l538_538586

theorem circumradii_sum_geq_half_perimeter
  (A B C D E F : Type)
  [Add A] [Add B] [Add C] [Add D] [Add E] [Add F]
  (R_A R_C R_E p : ℝ)
  (hexagon_convex: convex_hull ℝ (set.of_list [A, B, C, D, E, F]))
  (parallel_AB_ED : parallel AB ED)
  (parallel_BC_FE : parallel BC FE)
  (parallel_CD_AF : parallel CD AF)
  (R_A_def : R_A = circumradius (triangle FAB))
  (R_C_def : R_C = circumradius (triangle BCD))
  (R_E_def : R_E = circumradius (triangle DEF))
  (p_def : p = perimeter [A, B, C, D, E, F])
  : R_A + R_C + R_E ≥ p / 2 := sorry

end circumradii_sum_geq_half_perimeter_l538_538586


namespace sin_690_eq_neg_half_l538_538411

theorem sin_690_eq_neg_half :
  Real.sin (690 * Real.pi / 180) = -1 / 2 :=
by {
  sorry
}

end sin_690_eq_neg_half_l538_538411


namespace terminating_decimal_count_l538_538811

/-- 
There are 49 integers n between 1 and 449 inclusive such that the decimal representation
of the fraction n/450 terminates.
-/
theorem terminating_decimal_count:
  ∑ n in range(1, 450), (n % 9 = 0) = 49 := 
sorry

end terminating_decimal_count_l538_538811


namespace proof_f_f_pi_div_12_l538_538849

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then 4 * x^2 - 1 else (Real.sin x)^2 - (Real.cos x)^2

theorem proof_f_f_pi_div_12 : f (f (Real.pi / 12)) = 2 := by
  sorry

end proof_f_f_pi_div_12_l538_538849


namespace smaller_square_area_proof_l538_538739

-- Definitions based on conditions
structure Square :=
  (side_length : ℝ)
  (area : ℝ := side_length ^ 2)

def circle_radius (sq : Square) : ℝ :=
  sq.side_length * (2^.5) / 2

-- Problem conditions
def larger_square : Square := { side_length := 4 }
def smaller_square_side_length (larger_sq : Square) : ℝ :=
  2 * (2^.5)

-- The required property
def smaller_square_area (larger_sq : Square) : Square :=
  { side_length := smaller_square_side_length larger_sq, area := (smaller_square_side_length larger_sq)^2 }

-- Lean theorem statement
theorem smaller_square_area_proof :
  (smaller_square_area larger_square).area = (larger_square.area) / 2 :=
by
  -- Proof can be filled in here later
  sorry

end smaller_square_area_proof_l538_538739


namespace locus_midpoint_l538_538054

/-- Given a fixed point A (4, -2) and a moving point B on the curve x^2 + y^2 = 4,
    prove that the locus of the midpoint P of the line segment AB satisfies the equation 
    (x - 2)^2 + (y + 1)^2 = 1. -/
theorem locus_midpoint (A B P : ℝ × ℝ)
  (hA : A = (4, -2))
  (hB : ∃ (x y : ℝ), B = (x, y) ∧ x^2 + y^2 = 4)
  (hP : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (P.1 - 2)^2 + (P.2 + 1)^2 = 1 :=
sorry

end locus_midpoint_l538_538054


namespace find_tricksters_in_16_questions_l538_538335

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

end find_tricksters_in_16_questions_l538_538335


namespace _l538_538770

lemma sum_binomial_theorem (n : ℕ) :
  ∑ k in finset.range n, (3^(k) * (nat.choose n (k+1))) = (1 / 3 : ℚ) * (4^n - 1 - 2 * 3^n) :=
by
  sorry

end _l538_538770


namespace minimum_value_PR_plus_QR_l538_538481

def circle1 (P : ℝ × ℝ) : Prop :=
  (P.1 - 1)^2 + P.2^2 = 1

def circle2 (Q : ℝ × ℝ) : Prop :=
  (Q.1 - 4)^2 + (Q.2 - 1)^2 = 4

def line (R : ℝ × ℝ) : Prop :=
  R.1 - R.2 + 1 = 0

def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem minimum_value_PR_plus_QR
    (P Q R : ℝ × ℝ)
    (hP : circle1 P)
    (hQ : circle2 Q)
    (hR : line R) :
  distance P R + distance Q R ≥ sqrt 26 - 3 :=
  sorry

end minimum_value_PR_plus_QR_l538_538481


namespace matrix_not_invertible_iff_y_eq_one_seventh_l538_538009

open Matrix

def A (y : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![2 + y, 5], ![4 - y, 9]]

theorem matrix_not_invertible_iff_y_eq_one_seventh (y : ℚ) : 
  ¬ invertible (A y) ↔ y = 1 / 7 := by
sorry

end matrix_not_invertible_iff_y_eq_one_seventh_l538_538009


namespace ratio_of_liquid_rises_l538_538233

theorem ratio_of_liquid_rises
  (r1 : ℝ) (r2 : ℝ) (h1 h2 : ℝ) (V_m1 V_m2 V1 V2 : ℝ)
  (h1_eq : h1 = 4 * h2)
  (r1_eq : r1 = 4)
  (r2_eq : r2 = 8)
  (V_m1_eq : V_m1 = (4 / 3) * π * 2^3)
  (V_m2_eq : V_m2 = (4 / 3) * π * 1^3)
  (V1_eq : V1 = (1 / 3) * π * r1^2 * h1)
  (V2_eq : V2 = (1 / 3) * π * r2^2 * h2)
  :
  let r_narrow := (V1 + V_m1) / ((1 / 3) * π * r1^2)
  let r_wide := (V2 + V_m2) / ((1 / 3) * π * r2^2)
  in (r_narrow - h1) / (r_wide - h2) = 8 :=
by
  sorry

end ratio_of_liquid_rises_l538_538233


namespace ben_final_amount_l538_538765

-- Definition of the conditions
def daily_start := 50
def daily_spent := 15
def daily_saving := daily_start - daily_spent
def days := 7
def mom_double (s : ℕ) := 2 * s
def dad_addition := 10

-- Total amount calculation based on the conditions
noncomputable def total_savings := daily_saving * days
noncomputable def after_mom := mom_double total_savings
noncomputable def total_amount := after_mom + dad_addition

-- The final theorem to prove Ben's final amount is $500 after the given conditions
theorem ben_final_amount : total_amount = 500 :=
by sorry

end ben_final_amount_l538_538765


namespace power_function_value_l538_538891

theorem power_function_value (α : ℝ) (h₁ : (2 : ℝ) ^ α = (Real.sqrt 2) / 2) : (9 : ℝ) ^ α = 1 / 3 := 
by
  sorry

end power_function_value_l538_538891


namespace compute_G_784_G_786_l538_538561

-- Definitions based on conditions
def fib_modified : ℕ → ℤ
| 0       := 0
| 1       := 1
| (n + 2) := 3 * fib_modified (n + 1) + 2 * fib_modified n

-- Theorem statement
theorem compute_G_784_G_786 :
  fib_modified 784 * fib_modified 786 - 4 * (fib_modified 785)^2 = -3 ^ 785 :=
by
  -- Placeholder for the proof
  sorry

end compute_G_784_G_786_l538_538561


namespace maximal_area_of_cyclic_quadrilateral_l538_538628

noncomputable def p (a b c d : ℝ) := (a + b + c + d) / 2

noncomputable def brahmagupta_area (a b c d : ℝ) : ℝ :=
  Real.sqrt ((p a b c d - a) * (p a b c d - b) * (p a b c d - c) * (p a b c d - d))

theorem maximal_area_of_cyclic_quadrilateral {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ∀ (Q : Type) [Quadrilateral Q], (convex Q) → (sidelengths Q = (a, b, c, d)) → 
  area (inscribe_in_circle Q) = brahmagupta_area a b c d → 
  (∀ (Q' : Type) [Quadrilateral Q'], (convex Q') → (sidelengths Q' = (a, b, c, d)) → 
  area Q' ≤ brahmagupta_area a b c d) := sorry

end maximal_area_of_cyclic_quadrilateral_l538_538628


namespace equation_solutions_35_implies_n_26_l538_538955

theorem equation_solutions_35_implies_n_26 (n : ℕ) (h3x3y2z_eq_n : ∃ (s : Finset (ℕ × ℕ × ℕ)), (∀ t ∈ s, ∃ (x y z : ℕ), 
  t = (x, y, z) ∧ 3 * x + 3 * y + 2 * z = n ∧ x > 0 ∧ y > 0 ∧ z > 0) ∧ s.card = 35) : n = 26 := 
sorry

end equation_solutions_35_implies_n_26_l538_538955


namespace rate_per_sq_meter_l538_538665

theorem rate_per_sq_meter
  (L : ℝ) (W : ℝ) (total_cost : ℝ)
  (hL : L = 6) (hW : W = 4.75) (h_total_cost : total_cost = 25650) :
  total_cost / (L * W) = 900 :=
by
  sorry

end rate_per_sq_meter_l538_538665


namespace book_club_snacks_fee_l538_538612

theorem book_club_snacks_fee :
  let num_members := 6
  let cost_hardcover_per_book := 30
  let num_hardcover_books := 6
  let cost_paperback_per_book := 12
  let num_paperback_books := 6
  let total_amount_collected := 2412
  let total_books_fee_per_member := (cost_hardcover_per_book * num_hardcover_books) + (cost_paperback_per_book * num_paperback_books)
  let total_books_fee := num_members * total_books_fee_per_member
  let total_snacks_fee := total_amount_collected - total_books_fee
  let S := total_snacks_fee / num_members
  in S = 150 := by
  sorry

end book_club_snacks_fee_l538_538612


namespace total_seeds_eaten_correct_l538_538380

-- Define the number of seeds each player ate
def seeds_first_player : ℕ := 78
def seeds_second_player : ℕ := 53
def seeds_third_player (seeds_second_player : ℕ) : ℕ := seeds_second_player + 30

-- Define the total seeds eaten
def total_seeds_eaten (seeds_first_player seeds_second_player seeds_third_player : ℕ) : ℕ :=
  seeds_first_player + seeds_second_player + seeds_third_player

-- Statement of the theorem
theorem total_seeds_eaten_correct : total_seeds_eaten seeds_first_player seeds_second_player (seeds_third_player seeds_second_player) = 214 :=
by
  sorry

end total_seeds_eaten_correct_l538_538380


namespace find_x_l538_538024

open Nat

theorem find_x (n x : ℕ) (h1 : x = 2^n - 32) (h2 : x.prime_divisors.length = 3) (h3 : 2 ∈ x.prime_divisors) :
  x = 2016 ∨ x = 16352 := 
by
  -- The proof is omitted
  sorry

end find_x_l538_538024


namespace BinbinFatherHeight_BinbinMotherShorter_l538_538386

-- Definition of the heights
def BinbinHeight : ℝ := 1.46
def HeightDifferenceFather : ℝ := 0.32
def MotherHeight : ℝ := 1.5

-- Proof problem statements translated to Lean
theorem BinbinFatherHeight : BinbinHeight + HeightDifferenceFather = 1.78 :=
by
  have h : BinbinHeight + HeightDifferenceFather = 1.46 + 0.32 := rfl
  rw [h]
  norm_num

theorem BinbinMotherShorter : (BinbinHeight + HeightDifferenceFather) - MotherHeight = 0.28 :=
by
  have h1 : BinbinHeight + HeightDifferenceFather = 1.78 := by
    have h2 : BinbinHeight + HeightDifferenceFather = 1.46 + 0.32 := rfl
    rw [h2]
    norm_num
  rw [h1]
  norm_num

end BinbinFatherHeight_BinbinMotherShorter_l538_538386


namespace concyclic_points_l538_538479

noncomputable def circumcircle (A B C : Point) : Circle :=
  sorry -- Definition for the circumcircle

noncomputable def incenter (A B C : Point) : Point :=
  sorry -- Definition for the incenter

noncomputable def tangent_circle (O : Circle) (A : Point) : Circle :=
  sorry -- Definition for the tangent circle internally tangent at a point

noncomputable def midarc_point (O : Circle) (B C : Point) : Point :=
  sorry -- Definition for the midpoint of arc BC not containing a specific point

noncomputable def intersects (c1 c2 : Circle) : Set Point :=
  sorry -- Definition for intersection points of two circles

theorem concyclic_points 
  (A B C D I E F : Point) (O O1 O2 : Circle)
  (h1 : O = circumcircle A B C)
  (h2 : O1 = tangent_circle O A)
  (h3 : D ∈ B C ∩ O1)
  (h4 : I = incenter A B C)
  (h5 : O2 = circumcircle I B C)
  (h6 : E F ∈ intersects O1 O2) :
  concyclic O1 E O2 F :=
  sorry -- Proof goes here, omitted.

end concyclic_points_l538_538479


namespace total_cost_proof_l538_538606

-- Define the prices of items
def price_coffee : ℕ := 4
def price_cake : ℕ := 7
def price_ice_cream : ℕ := 3

-- Define the number of items ordered by Mell and her friends
def mell_coffee : ℕ := 2
def mell_cake : ℕ := 1
def friend_coffee : ℕ := 2
def friend_cake : ℕ := 1
def friend_ice_cream : ℕ := 1
def number_of_friends : ℕ := 2

-- Calculate total cost for Mell
def total_mell : ℕ := (mell_coffee * price_coffee) + (mell_cake * price_cake)

-- Calculate total cost per friend
def total_friend : ℕ := (friend_coffee * price_coffee) + (friend_cake * price_cake) + (friend_ice_cream * price_ice_cream)

-- Calculate total cost for all friends
def total_friends : ℕ := number_of_friends * total_friend

-- Calculate total cost for Mell and her friends
def total_cost : ℕ := total_mell + total_friends

-- The theorem to prove
theorem total_cost_proof : total_cost = 51 := by
  sorry

end total_cost_proof_l538_538606


namespace probability_at_A_after_9_meters_l538_538141

def P : ℕ → ℚ
| 0     := 1
| 1     := 0
| (n+1) := (1 - P n) / 3

theorem probability_at_A_after_9_meters : 
  (P 9 = 1640 / 6561) ∧ (1640 * 3 = 4920) :=
by
  sorry

end probability_at_A_after_9_meters_l538_538141


namespace minimum_value_of_func_l538_538063

noncomputable def func (x : ℝ) : ℝ := (sin x)^2 + sin (2 * x) + 3 * (cos x)^2

theorem minimum_value_of_func : 
  ∃ (x : ℝ), (∀ (k : ℤ), x = k * π - (3 * π / 8)) ∧ func x = 2 - real.sqrt 2 := 
sorry

end minimum_value_of_func_l538_538063


namespace arithmetic_mean_sequence_l538_538650

-- Define the arithmetic sequence starting from 3 with a common difference of 1
def sequence (n : ℕ) : ℕ := n + 2

-- Prove that the arithmetic mean of the first 60 terms is 32.5
theorem arithmetic_mean_sequence :
  let terms := (list.range 60).map sequence in
  let sum := terms.sum in
  let mean := (sum : ℚ) / 60 in
  mean = 32.5 :=
by {
  let terms : list ℕ := (list.range 60).map sequence,
  let sum : ℕ := terms.sum,
  have h_sum : sum = 1950 := sorry,
  have h_mean : (sum : ℚ) / 60 = 32.5 := sorry,
  exact h_mean
}

end arithmetic_mean_sequence_l538_538650


namespace normal_vector_to_line_l538_538291

theorem normal_vector_to_line : 
  ∀ (x y : ℝ), x - 3 * y + 6 = 0 → (1, -3) = (1, -3) :=
by
  intros x y h_line
  sorry

end normal_vector_to_line_l538_538291


namespace coefficient_x5_expansion_l538_538656

theorem coefficient_x5_expansion :
  let expansion := λ (x : ℝ), (x^2 - 2 / sqrt x)^10 in
  let general_term := λ (r : ℕ), (-(2 : ℝ))^r * (Nat.choose 10 r) * x^(20 - (5 * r)/2) in
  ∃ (r : ℕ) (coeff : ℝ), r = 6 ∧ coeff = (-(2 : ℝ))^r * (Nat.choose 10 r) ∧ coeff = 13440 :=
by 
  sorry

end coefficient_x5_expansion_l538_538656


namespace greatest_integer_not_exceeding_x2_div_50_l538_538619

noncomputable def trapezoid_condition (b h : ℝ) :=
  let longer_base := b + 50 in
  let midline := (b + longer_base) / 2 in
  midline = b + 25 ∧
  (2 * b + 25) = 3 * (b + 37.5) ∧
  b = 37.5

noncomputable def segment_condition (x : ℝ) (b : ℝ := 37.5) :=
  (2 * (b + x) = 125) ∧ (x^2 - 37.5 * x + 2812.5 = 0)

theorem greatest_integer_not_exceeding_x2_div_50 (x : ℝ) (h : ℝ) :
  trapezoid_condition 37.5 h → segment_condition x 37.5 → 
  (⌊x^2 / 50⌋ = 112) :=
by
  intros h1 h2
  sorry

end greatest_integer_not_exceeding_x2_div_50_l538_538619


namespace bracelet_arrangements_l538_538393

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem bracelet_arrangements : 
  (factorial 8) / (8 * 2) = 2520 := by
    sorry

end bracelet_arrangements_l538_538393


namespace collinear_A_R_P_l538_538617

-- Definitions and conditions
variables {A B C D P Q R : Type} [inhabited A] [inhabited B] [inhabited C]
variables (IsSquare : Square A B C D) (P_on_BC : P ∈ segment B C) 
variables (circle1 : Circle A B P Q) (intersect_Q : Q ∈ diagonal B D ∩ Circle A B P Q)
variables (circle2 : Circle C P Q R) (intersect_R : R ∈ diagonal B D ∩ Circle C P Q R)

-- Theorem statement
theorem collinear_A_R_P : Collinear A R P :=
sorry

end collinear_A_R_P_l538_538617


namespace find_tricksters_l538_538369

def inhab_group : Type := { n : ℕ // n < 65 }
def is_knight (i : inhab_group) : Prop := ∀ g : inhab_group, true -- Placeholder for the actual property

theorem find_tricksters (inhabitants : inhab_group → Prop)
  (is_knight : inhab_group → Prop)
  (knight_always_tells_truth : ∀ i : inhab_group, is_knight i → inhabitants i = true)
  (tricksters_2_and_rest_knights : ∃ t1 t2 : inhab_group, t1 ≠ t2 ∧ ¬is_knight t1 ∧ ¬is_knight t2 ∧
    (∀ i : inhab_group, i ≠ t1 → i ≠ t2 → is_knight i)) :
  ∃ find_them : inhab_group → inhab_group → Prop, (∀ q_count : ℕ, q_count ≤ 16) → 
  ∃ t1 t2 : inhab_group, find_them t1 t2 :=
by 
  -- The proof goes here
  sorry

end find_tricksters_l538_538369


namespace lights_out_l538_538303

theorem lights_out (n : ℕ) (h : n = 20) : 
  (∑ k in Finset.range (n+1), if k > 0 then Nat.choose n k else 0) = 2^20 - 1 :=
by 
  sorry

end lights_out_l538_538303


namespace num_solutions_to_congruence_l538_538840

def is_solution (x : ℕ) : Prop :=
  x < 150 ∧ ((x + 17) % 46 = 75 % 46)

theorem num_solutions_to_congruence : (finset.univ.filter is_solution).card = 3 :=
by
  sorry

end num_solutions_to_congruence_l538_538840


namespace total_cost_price_l538_538741

-- Definitions for each item's selling price and profit percentage
def SP_A : ℝ := 150
def SP_B : ℝ := 200
def SP_C : ℝ := 250
def Profit_A : ℝ := 0.25
def Profit_B : ℝ := 0.20
def Profit_C : ℝ := 0.15

-- Compute the cost price for each item
def CP_A : ℝ := SP_A / (1 + Profit_A)
def CP_B : ℝ := SP_B / (1 + Profit_B)
def CP_C : ℝ := SP_C / (1 + Profit_C)

-- Prove that the total cost price equals the calculated value
theorem total_cost_price : CP_A + CP_B + CP_C = 504.06 := by
  sorry

end total_cost_price_l538_538741


namespace four_lines_r_condition_l538_538203

noncomputable def range_of_r (parabola : ℝ → ℝ, circle : ℝ × ℝ → ℝ, midpoint : ℝ × ℝ) :=
  ∃ (l : Line), 
    (∀ A B : Point, parabola A.2 = 6 * A.1 ∧ parabola B.2 = 6 * B.1 ∧ (l.intersect parabola) = {A, B} ∧ 
      (midpoint (A, B) = M) ∧ circle M = r^2 ∧ l.tangent_at M) ∧
    range (circle) = set.Ioo 3 (3 * ℝ.sqrt 3)

theorem four_lines_r_condition :
  ∀ parabola circle midpoint M, 
    (∃ l, (line_tangent_to_circle parabola circle midpoint M l) ∧ num_lines parabola circle midpoint M l = 4) ↔ 
    range_of_r parabola circle midpoint = set.Ioo 3 (3 * ℝ.sqrt 3) := 
sorry

end four_lines_r_condition_l538_538203


namespace simplify_expression_l538_538179

theorem simplify_expression :
  (Real.sqrt (Real.sqrt (81)) - Real.sqrt (8 + 1 / 2)) ^ 2 = (35 / 2) - 3 * Real.sqrt 34 :=
by
  sorry

end simplify_expression_l538_538179


namespace maria_mushrooms_l538_538162

theorem maria_mushrooms (potatoes carrots onions green_beans bell_peppers mushrooms : ℕ) 
  (h1 : carrots = 6 * potatoes)
  (h2 : onions = 2 * carrots)
  (h3 : green_beans = onions / 3)
  (h4 : bell_peppers = 4 * green_beans)
  (h5 : mushrooms = 3 * bell_peppers)
  (h0 : potatoes = 3) : 
  mushrooms = 144 :=
by
  sorry

end maria_mushrooms_l538_538162


namespace sequence_a_n_sequence_b_n_sum_c_n_l538_538158

-- Definition of the sequence {a_n}
def sequenceSum (S : ℕ → ℝ) (a : ℕ → ℝ) := ∀ n : ℕ, S n = 2 - a n

-- General formula for the sequence {a_n}
theorem sequence_a_n (S : ℕ → ℝ) (a : ℕ → ℝ) (h : sequenceSum S a) :
  ∀ n : ℕ, a (n + 1) = (1 / 2) * a n ∧ a 1 = 1 → a n = (1 / 2)^(n - 1) := 
sorry

-- General formula for the sequence {b_n}
theorem sequence_b_n (b : ℕ → ℝ) (a : ℕ → ℝ) :
  b 1 = 1 → (∀ n : ℕ, b (n + 1) = b n + a n) → (∀ n : ℕ, a (n + 1) = (1 / 2) * a n) → a 1 = 1 →
  ∀ n : ℕ, b n = 3 - 1 / (2 ^ (n - 2)) :=
sorry

-- Sum of the first n terms of {c_n}
theorem sum_c_n (b : ℕ → ℝ) (c : ℕ → ℝ) :
  (∀ n : ℕ, c n = (n * (3 - b n) / 2)) → (b 1 = 1) → (∀ n : ℕ, b (n + 1) = b n + a n) → (∀ n : ℕ, a (n + 1) = (1 / 2) * a n) → 
  a 1 = 1 → ∀ n : ℕ, T n = 4 - (2 + n) / (2 ^ (n - 1)) :=
sorry

end sequence_a_n_sequence_b_n_sum_c_n_l538_538158


namespace total_cost_l538_538600

-- Given conditions
def pen_cost : ℕ := 4
def briefcase_cost : ℕ := 5 * pen_cost

-- Theorem stating the total cost Marcel paid for both items
theorem total_cost (pen_cost briefcase_cost : ℕ) (h_pen: pen_cost = 4) (h_briefcase: briefcase_cost = 5 * pen_cost) :
  pen_cost + briefcase_cost = 24 := by
  sorry

end total_cost_l538_538600


namespace mindmaster_codes_count_l538_538549

theorem mindmaster_codes_count : 
  let num_colors := 7 in
  let num_slots := 5 in
  num_colors ^ num_slots = 16807 :=
by
  sorry

end mindmaster_codes_count_l538_538549


namespace Gwendolyn_reading_time_l538_538871

theorem Gwendolyn_reading_time
  (sentences_per_hour : ℕ)
  (paragraphs_per_page : ℕ)
  (sentences_per_paragraph : ℕ)
  (pages : ℕ)
  (H1 : sentences_per_hour = 300)
  (H2 : paragraphs_per_page = 40)
  (H3 : sentences_per_paragraph = 20)
  (H4 : pages = 150) :
  (pages * paragraphs_per_page * sentences_per_paragraph) / sentences_per_hour = 400 :=
by 
  -- conditions
  have Hsentences := H1,
  have Hparagraphs := H2,
  have HsentencesInParagraph := H3,
  have Hpages := H4,
  sorry -- Proof will be provided here

end Gwendolyn_reading_time_l538_538871


namespace centroid_lies_on_line_l538_538863

-- Defining the problem conditions and variables
variables (L1 L2 L3 : set ℝ^3) -- The three pairwise skew lines
variables (π : set ℝ^3) -- The given plane
variables (triangle : set ℝ^3) -- A triangle with vertices on the three lines and parallel to the plane
variables (G : ℝ^3) -- The centroid of the triangle

-- Defining the skew-line condition
def pairwise_skew (L1 L2 L3 : set ℝ^3) := 
  ∀ (l1 ∈ L1) (l2 ∈ L2) (l3 ∈ L3), (l1 ∥ l2) ∧ (l2 ∥ l3) ∧ (l1 ∥ l3) = false

-- Defining the plane parallel condition
def parallel_to_plane (triangle : set ℝ^3) (π : set ℝ^3) := 
  ∃ u v w ∈ π, ∀ (p ∈ triangle), p = u + v + w

-- Defining the centroid G as the intersection of medians
def is_centroid (G : ℝ^3) (triangle : set ℝ^3) :=
  ∃ (A B C : ℝ^3), A ∈ triangle ∧ B ∈ triangle ∧ C ∈ triangle ∧ G = (A + B + C) / 3

-- The statement of the proof problem
theorem centroid_lies_on_line
  (pairwise_skew L1 L2 L3)
  (parallel_to_plane triangle π)
  (is_centroid G triangle) :
  ∃ (l : set ℝ^3), ∀ (G : ℝ^3), is_centroid G triangle → G ∈ l :=
sorry

end centroid_lies_on_line_l538_538863


namespace number_of_real_solutions_is_one_l538_538143

noncomputable def num_real_solutions_eq : Bool :=
  let eq (x : ℝ) : Bool := (3 * x ^ 3 - 30 * (⌊x⌋:ℝ) + 34 = 0)
  let sols := { x : ℝ | eq x }.toSet
  sols.finite.length = 1

theorem number_of_real_solutions_is_one : num_real_solutions_eq = true := by {
  sorry
}

end number_of_real_solutions_is_one_l538_538143


namespace area_of_win_sector_l538_538725

theorem area_of_win_sector (r : ℝ) (p : ℝ) (A : ℝ) (h_1 : r = 10) (h_2 : p = 1 / 4) (h_3 : A = π * r^2) : 
  (p * A) = 25 * π := 
by
  sorry

end area_of_win_sector_l538_538725


namespace number_of_valid_a_l538_538896

theorem number_of_valid_a : 
  (∃(a : ℤ), ∃(x : ℕ), (frac_eq : (x + a) / (x - 2) + 2x / (2 - x) = 1) ∧ 
  (ineq_sys : exactIntegerSolutions (λ y, (y + 1) / 5 ≥ y / 2 - 1 ∧ y + a < 11y - 3) 4)) ↔ 
  ((a = -2) ∨ (a = 0) ∨ (a = 4) ∨ (a = 6)) := 
begin
  sorry
end

end number_of_valid_a_l538_538896


namespace function_satisfies_condition_l538_538787

def f (x : ℝ) : ℝ :=
  if x = 1 then arbitrary else x + 1

theorem function_satisfies_condition (f : ℝ → ℝ) :
  (∀ x : ℝ, if x = 1 then f(x) = arbitrary else f(x) = x + 1) → 
  (∀ x : ℝ, (x+1) * f(x+2) - 2 * (x+2) * f(-x-1) = 3 * x^2 + 8 * x + 3) := 
sorry

end function_satisfies_condition_l538_538787


namespace least_n_rays_acute_angle_in_R3_l538_538448

theorem least_n_rays_acute_angle_in_R3 : ∃ n : ℕ, (∀ (rays : Fin n → ℝ^3), 
  (rays = 7) → ∃ i j : Fin n, i ≠ j ∧ (rays i, rays j).dot > 0) := by
  sorry

end least_n_rays_acute_angle_in_R3_l538_538448


namespace cube_of_number_l538_538733

theorem cube_of_number (n : ℕ) (h1 : 40000 < n^3) (h2 : n^3 < 50000) (h3 : (n^3 % 10) = 6) : n = 36 := by
  sorry

end cube_of_number_l538_538733


namespace initial_bottles_proof_l538_538161

-- Define the conditions as variables and statements
def initial_bottles (X : ℕ) : Prop :=
X - 8 + 45 = 51

-- Theorem stating the proof problem
theorem initial_bottles_proof : initial_bottles 14 :=
by
  -- We need to prove the following:
  -- 14 - 8 + 45 = 51
  sorry

end initial_bottles_proof_l538_538161


namespace four_digit_count_special_condition_count_eighty_fifth_item_l538_538817

-- Define the set of digits used
def digits := {0, 1, 2, 3, 4, 5}

-- (Ⅰ) Theorem stating the number of different four-digit numbers
theorem four_digit_count : 
  (finset.bind (finset.image (λp : fin 4 → ℕ, digits.val ∈ digits) 
  (finset.powersetLen 4 (finset.range 6))).filter (λ x, x.nth 0 ≠ 0)).card = 300 :=
sorry

-- (Ⅱ) Theorem stating the number of four-digit numbers with specific ten's digit condition
theorem special_condition_count :
  (finset.bind (finset.image (λp : fin 4 → ℕ, digits.val ∈ digits) 
  (finset.powersetLen 4 (finset.range 6))).filter (λ x, x.nth 0 ≠ 0 ∧ x.nth 2 < x.nth 1 ∧ x.nth 3 < x.nth 1)).card = 100 :=
sorry

-- (Ⅲ) Theorem stating the 85th item in the sequence
theorem eighty_fifth_item :
  (list.sort (λ x y, x < y) 
  (finset.bind (finset.image (λp : fin 4 → ℕ, digits.val ∈ digits)
  (finset.powersetLen 4 (finset.range 6))).filter (λ x, x.nth 0 ≠ 0))).nth 84 = 2301 :=
sorry

end four_digit_count_special_condition_count_eighty_fifth_item_l538_538817


namespace printer_cost_l538_538286

theorem printer_cost (total_cost : ℕ) (num_keyboards : ℕ) (keyboard_cost : ℕ) (num_printers : ℕ) (printer_cost : ℕ) :
  total_cost = 2050 → num_keyboards = 15 → keyboard_cost = 20 → num_printers = 25 →
  (total_cost - num_keyboards * keyboard_cost) / num_printers = 70 := 
by
  intros h_tc h_nk h_kc h_np
  rw [h_tc, h_nk, h_kc, h_np]
  norm_num
  sorry

end printer_cost_l538_538286


namespace rational_terms_binomial_expansion_l538_538050

theorem rational_terms_binomial_expansion :
  let binom_expansion_term (n r : ℕ) (x : ℝ) : ℝ :=
    (nat.choose n r) * (sqrt x)^(n - r) * (1/(24*x))^r in
  (∀ (x : ℝ), 
    (sum (λ r, if r % 2 = 1 then nat.choose 10 r else 0) (range (10 + 1)) = 512) →
    (rational_terms : List ℝ :=
      [binom_expansion_term 10 0 x, binom_expansion_term 10 4 x, binom_expansion_term 10 8 x]) ∧
    rational_terms = [x^5, (105/8)*x^2, 45/(256*x)]) :=
sorry

end rational_terms_binomial_expansion_l538_538050


namespace smallest_positive_integer_divisible_12_15_16_l538_538451

theorem smallest_positive_integer_divisible_12_15_16 : 
  ∃ (n : ℕ), 0 < n ∧ n % 12 = 0 ∧ n % 15 = 0 ∧ n % 16 = 0 ∧ ∀ m : ℕ, 0 < m ∧ m % 12 = 0 ∧ m % 15 = 0 ∧ m % 16 = 0 → n ≤ m :=
begin
  let n := 240,
  use n,
  split,
  { exact nat.zero_lt_succ 239 },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3 hm4,
    sorry },
end

end smallest_positive_integer_divisible_12_15_16_l538_538451


namespace find_tricksters_l538_538328

structure Inhabitant :=
  (isKnight : Prop)

constants (inhabitants : Fin 65 → Inhabitant)
          (tricksters : Fin 2 → Fin 65)

axiom two_tricksters_unique :
  ∃! (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65))

axiom valid_question :
  ∀ (a : Fin 65) (group : Set (Fin 65)), (inhabitants a).isKnight → 
  (∀ i ∈ group, (inhabitants i).isKnight) ↔ 
  (knight a).isKnight

theorem find_tricksters :
  ∃ (q : ℕ), q ≤ 16 ∧
  ∃ (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65)) :=
sorry

end find_tricksters_l538_538328


namespace initial_savings_correct_l538_538632

noncomputable def initial_savings := 10450.73

def balance_february (S : ℝ) := 0.80 * S
def balance_march (S : ℝ) := 0.60 * balance_february(S) * 1.05
def balance_april (S : ℝ) := balance_march(S) - 2500
def balance_may (S : ℝ) := balance_april(S) + 1100
def balance_june (S : ℝ) := balance_may(S) * 0.75

theorem initial_savings_correct :
  balance_june(initial_savings) = 2900 :=
by
  unfold initial_savings balance_february balance_march balance_april balance_may balance_june
  sorry

end initial_savings_correct_l538_538632


namespace proof_problem_l538_538951

theorem proof_problem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 :=
sorry

end proof_problem_l538_538951


namespace complex_solution_l538_538035

noncomputable def i : ℂ := complex.I

theorem complex_solution (z : ℂ) (h : (2 - i) * z = i ^ 3) : 
  z = (1 / 5) - (2 / 5) * i :=
sorry

end complex_solution_l538_538035


namespace distinct_blue_numbers_impossible_l538_538614

theorem distinct_blue_numbers_impossible (S : Finset ℕ) (hS : ∀ n ∈ S, n < 100) (card_S : S.card = 49) :
  ¬ (∀ p q ∈ S, p ≠ q → (∀ i j ∈ (S.image (λ x y, Nat.gcd x y)), i ≠ j)) := sorry

end distinct_blue_numbers_impossible_l538_538614


namespace identify_tricksters_in_30_or_less_questions_l538_538355

-- Define the problem parameters
def inhabitants : Type := Fin 65

def is_knight (inhabitant : inhabitants) : Prop := sorry
def is_trickster (inhabitant : inhabitants) : Prop := sorry

-- Define the properties
axiom knight_truthful : ∀ (x : inhabitants), is_knight x → (forall y : inhabitants, True ↔ (is_knight y = x is_knight y))
axiom trickster_mixed : ∀ (x : inhabitants), is_trickster x → ((∀ y : inhabitants, True) ∨ (∃ y : inhabitants, y ∉ (is_knight y)))

-- Problem statement
theorem identify_tricksters_in_30_or_less_questions
  (inhabitants : Type)
  (n_tricksters : ℕ := 2) -- 2 tricksters
  (total_inhabitants : ℕ := 65) -- 65 total inhabitants
  (questions_limit : ℕ := 30) -- limit of 30 questions
  (knights : inhabitants → Prop)
  (tricksters : inhabitants → Prop) :
    ∃ (solution_exists : ∀ (is_trickster : inhabitants → Prop), ∃ k : inhabitants, (knights k) ∧ (is_trickster k)) 
    (possible_to_find_tricksters : ∀ (is_knight : inhabitants → Prop) (is_trickster : inhabitants → Prop), 
    ∃ (questions_used ≤ questions_limit), ∀ (xs : set inhabitants), questions_used ≤ 30 ∧ 
    (∃ trickster1 trickster2 : inhabitants, (tricksters trickster1 ∧ tricksters trickster2 ∧ trickster1 ≠ trickster2))) :=
sorry

end identify_tricksters_in_30_or_less_questions_l538_538355


namespace tan_585_eq_1_l538_538773

noncomputable def tan_deg (θ : ℝ) : ℝ := Real.tan (θ * Real.pi / 180)

theorem tan_585_eq_1 :
  tan_deg 585 = 1 := 
by
  have h1 : 585 - 360 = 225 := by norm_num
  have h2 : tan_deg 225 = tan_deg 45 :=
    by have h3 : 225 = 180 + 45 := by norm_num
       rw [h3, tan_deg]
       exact Real.tan_add_pi_div_two_simp_left (Real.pi * 45 / 180)
  rw [← tan_deg]
  rw [h1, h2]
  exact Real.tan_pi_div_four

end tan_585_eq_1_l538_538773


namespace vehicles_traveled_l538_538138

theorem vehicles_traveled (V : ℕ)
  (h1 : 40 * V = 800 * 100000000) : 
  V = 2000000000 := 
sorry

end vehicles_traveled_l538_538138


namespace sin_690_eq_neg_half_l538_538401

theorem sin_690_eq_neg_half :
  let rad := Real.pi / 180 in -- Convert degrees to radians
  Real.sin (690 * rad) = -1 / 2 :=
by
  sorry

end sin_690_eq_neg_half_l538_538401


namespace minimum_value_l538_538435

-- Define the conditions
variables (A B C M N : Type) [inner_product_space ℝ A] -- Define points in inner product space
variables (AB AC : A)
variables (x y : ℝ)

-- Given conditions
def conditions (AB AC : A) (x y : ℝ) : Prop :=
  inner_product AB AC = 2 * real.sqrt 3 ∧
  ∠ BAC = π / 6 ∧ -- 30 degrees in radians
  f N = (1 / 2, x, y)

-- The main theorem to be proved
theorem minimum_value (AB AC : A) (x y : ℝ) (h : conditions AB AC x y) : 
  1 / x + 4 / y = 18 :=
begin
  sorry,
end

end minimum_value_l538_538435


namespace chocolate_syrup_per_glass_l538_538397

-- Definitions from the conditions
def each_glass_volume : ℝ := 8
def milk_per_glass : ℝ := 6.5
def total_milk : ℝ := 130
def total_chocolate_syrup : ℝ := 60
def total_chocolate_milk : ℝ := 160

-- Proposition and statement to prove
theorem chocolate_syrup_per_glass : 
  (total_chocolate_milk / each_glass_volume) * milk_per_glass = total_milk → 
  (each_glass_volume - milk_per_glass = 1.5) := 
by 
  sorry

end chocolate_syrup_per_glass_l538_538397


namespace find_x_l538_538025

open Nat

theorem find_x (n x : ℕ) (h1 : x = 2^n - 32) (h2 : x.prime_divisors.length = 3) (h3 : 2 ∈ x.prime_divisors) :
  x = 2016 ∨ x = 16352 := 
by
  -- The proof is omitted
  sorry

end find_x_l538_538025


namespace point_X_on_BC_segment_l538_538585

noncomputable def Point := ℝ × ℝ -- Assuming points are defined in a Euclidean plane
noncomputable def Circle := { center : Point // radius : ℝ }

variables {A B C O P Q X : Point} 
variables {Γ₁ Γ₂ : Circle}

-- Given conditions
def is_center_of_circle (O : Point) (Γ : Circle) : Prop := Γ.center = O

def on_angle_bisector (O A B C : Point) : Prop := 
  ∃ line₁ line₂, 
    line₁ ∈ angle A O B ∧ 
    line₂ ∈ angle A O C ∧ 
    is_mid_line O line₁ line₂ -- Custom definition indicating the angle bisector

def passes_through (Γ : Circle) (p : Point) : Prop :=
  dist Γ.center p = Γ.radius

def intersection_of_lines (PQ AO : Line) (X : Point) : Prop :=
  X ∈ PQ ∧ X ∈ AO

def belongs_to_segment (X B C : Point) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ X = t • B + (1 - t) • C

-- Problem statement in Lean
theorem point_X_on_BC_segment 
  (h1 : is_center_of_circle O Γ₁)
  (h2 : on_angle_bisector O A B C)
  (h3 : passes_through Γ₁ B)
  (h4 : passes_through Γ₁ C)
  (h5 : passes_through Γ₂ O)
  (h6 : passes_through Γ₂ A)
  (h7 : Γ₁ ∩ Γ₂ = {P, Q})
  (h8 : intersection_of_lines PQ AO X) :
  belongs_to_segment X B C := 
sorry

end point_X_on_BC_segment_l538_538585


namespace normal_prob_l538_538894

/-- Given random variable ξ which is normally distributed with mean 2 and variance 1,
    and given that P(ξ > 3) = 0.1587, prove that P(ξ > 1) = 0.8413. -/
theorem normal_prob (ξ : ℝ → ℝ) (h1 : ∀ x, ξ x ∼ Normal 2 1) (h2 : Prob (ξ > 3) = 0.1587) : 
  Prob (ξ > 1) = 0.8413 :=
sorry

end normal_prob_l538_538894


namespace sin_690_eq_negative_one_half_l538_538412

theorem sin_690_eq_negative_one_half : Real.sin (690 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_690_eq_negative_one_half_l538_538412


namespace min_value_f_l538_538015

noncomputable def f (x : ℝ) : ℝ :=
∫ t in -∞..x, (2 * t - 4)

theorem min_value_f : ∃ x ∈ Icc (-1 : ℝ) 3, f x = -4 := by
  sorry

end min_value_f_l538_538015


namespace profit_percentage_correct_l538_538372

def cost_price : ℝ := 47.5
def selling_price : ℝ := 67.47
def deduction_rate : ℝ := 0.12 

theorem profit_percentage_correct : 
  let marked_price := selling_price / (1 - deduction_rate) in
  let profit := selling_price - cost_price in
  let profit_percentage := (profit / cost_price) * 100 in
  profit_percentage = 42.04 :=
by 
  sorry

end profit_percentage_correct_l538_538372


namespace probability_of_selection_l538_538910

def total_group : ℝ := 100
def percent_women : ℝ := 60
def percent_men : ℝ := 40

def percent_women_lawyers : ℝ := 35
def percent_women_doctors : ℝ := 25
def percent_women_engineers : ℝ := 20
def percent_women_architects : ℝ := 10
def percent_women_finance : ℝ := 10

def percent_men_lawyers : ℝ := 30
def percent_men_doctors : ℝ := 25
def percent_men_engineers : ℝ := 20
def percent_men_architects : ℝ := 15
def percent_men_finance : ℝ := 10

def number_women : ℝ := total_group * (percent_women / 100)
def number_men : ℝ := total_group * (percent_men / 100)

def number_female_engineers : ℝ := number_women * (percent_women_engineers / 100)
def number_male_doctors : ℝ := number_men * (percent_men_doctors / 100)
def number_male_lawyers : ℝ := number_men * (percent_men_lawyers / 100)

def probability_female_engineer : ℝ := (number_female_engineers / total_group) * 100
def probability_male_doctor : ℝ := (number_male_doctors / total_group) * 100
def probability_male_lawyer : ℝ := (number_male_lawyers / total_group) * 100

def total_probability : ℝ := 
  probability_female_engineer + probability_male_doctor + probability_male_lawyer

theorem probability_of_selection :
  total_probability = 34 := by
  sorry

end probability_of_selection_l538_538910


namespace marble_arrangements_l538_538570

theorem marble_arrangements : 
  let marbles := { "R", "S", "C", "M", "E" } in
  (∀ arrangement: List String, arrangement ∈ marbles.permutations → 
  (arrangement.head ≠ "M" ∧ arrangement.head ≠ "E") ∧ 
  (arrangement.last ≠ "M" ∧ arrangement.last ≠ "E")) → 
  marbles.permutations.count (λ arrangement =>
   (arrangement.head ≠ "M" ∧ arrangement.head ≠ "E") ∧ 
   (arrangement.last ≠ "M" ∧ arrangement.last ≠ "E")) = 36 := by
  sorry

end marble_arrangements_l538_538570


namespace focal_distance_of_ellipse_l538_538816

section Ellipse

variables x y : ℝ

def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem focal_distance_of_ellipse :
  is_ellipse 2 (sqrt 2) x y → (2 * sqrt (2)) = 2 * sqrt 2 := 
by
  intro h,
  sorry

end Ellipse

end focal_distance_of_ellipse_l538_538816


namespace two_by_two_squares_in_divided_grid_l538_538535

theorem two_by_two_squares_in_divided_grid :
  let n := 100
  let num_cuts := 10000
  let num_figures := 2500
  let total_perimeter := 4 * n + 2 * num_cuts := 20400
  let number_of_2x2_squares := 2300
  8 * number_of_2x2_squares + 10 * (num_figures - number_of_2x2_squares) = total_perimeter :=
by
  let n := 100
  let num_cuts := 10000
  let num_figures := 2500
  let total_perimeter := 4 * n + 2 * num_cuts := 20400
  let number_of_2x2_squares := 2300
  show 8 * number_of_2x2_squares + 10 * (num_figures - number_of_2x2_squares) = total_perimeter
     from sorry

end two_by_two_squares_in_divided_grid_l538_538535


namespace function_pairs_are_inverses_one_div_x_involution_neg_x_involution_inverse_of_increasing_convex_is_concave_inverse_of_decreasing_convex_is_convex_l538_538225

-- Step 1: Prove the given function pairs are inverses
theorem function_pairs_are_inverses (a x n : ℝ) : 
  (∀ y : ℝ, y = a ^ x ↔ x = log a y) ∧ 
  (∀ y : ℝ, y = x ^ n ↔ x = y ^ (1 / n)) ∧ 
  (∀ y : ℝ, y = (1 / x ^ n) ↔ x = (1 / y ^ (1 / n))) :=
by sorry

-- Step 2: Prove that 1/x is an involutory function
theorem one_div_x_involution (x : ℝ) (h : x ≠ 0) : 
  (1 / (1 / x)) = x :=
by sorry

-- Step 2: Find another involutory function example
theorem neg_x_involution (x : ℝ) : 
  (-(-x)) = x :=
by sorry

-- Step 3: Prove that the inverse of a monotonically increasing convex function is concave
theorem inverse_of_increasing_convex_is_concave
  (f : ℝ → ℝ) (φ : ℝ → ℝ)
  (hf : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2)
  (convex_f : ∀ x1 x2 : ℝ, ∀ λ : ℝ, λ ∈ Icc (0 : ℝ) (1 : ℝ) → f (λ * x1 + (1 - λ) * x2) ≤ λ * f x1 + (1 - λ) * f x2)
  (hf_inv : ∀ y : ℝ, φ (f y) = y) :
  (∀ x1 x2 : ℝ, ∀ λ : ℝ, λ ∈ Icc (0 : ℝ) (1 : ℝ) → φ ((1 - λ) * f x1 + λ * f x2) ≥ (1 - λ) * φ (f x1) + λ * φ (f x2)) :=
by sorry

-- Step 4: Prove that the inverse of a monotonically decreasing convex function is also convex
theorem inverse_of_decreasing_convex_is_convex
  (f : ℝ → ℝ) (φ : ℝ → ℝ)
  (hf : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2)
  (convex_f : ∀ x1 x2 : ℝ, ∀ λ : ℝ, λ ∈ Icc (0 : ℝ) (1 : ℝ) → f (λ * x1 + (1 - λ) * x2) ≤ λ * f x1 + (1 - λ) * f x2)
  (hf_inv : ∀ y : ℝ, φ (f y) = y) :
  (∀ x1 x2 : ℝ, ∀ λ : ℝ, λ ∈ Icc (0 : ℝ) (1 : ℝ) → φ ((1 - λ) * f x1 + λ * f x2) ≤ (1 - λ) * φ (f x1) + λ * φ (f x2)) :=
by sorry

end function_pairs_are_inverses_one_div_x_involution_neg_x_involution_inverse_of_increasing_convex_is_concave_inverse_of_decreasing_convex_is_convex_l538_538225


namespace multiplicity_of_root_one_l538_538701

noncomputable def P (n : ℕ) (X : ℝ) : ℝ := X^(2 * n) - n * X^(n + 1) + n * X^n -  X^2

theorem multiplicity_of_root_one (n : ℕ) (hn : 2 ≤ n) :
  (P n).rootMultiplicity 1 = if n = 2 then 2 else 1 :=
by
  sorry

end multiplicity_of_root_one_l538_538701


namespace round_trip_tickets_l538_538173

theorem round_trip_tickets (P : ℕ) (h1 : 0.20 * P ∈ ℚ) (h2 : 0.80 * ((0.20 * P) / 0.80) ∈ ℕ) :
    (∃ R : ℕ, R = 0.25 * P) :=
by
  sorry

end round_trip_tickets_l538_538173


namespace largest_prime_divisor_is_397_l538_538802

def largest_prime_divisor_base7 : Prop :=
  let n := (2 * 7^7 + 1 * 7^6 + 0 * 7^5 + 1 * 7^4 + 2 * 7^3 + 0 * 7^2 + 2 * 7^1 + 1 * 7^0)
  n = 1769837 ∧
  ∀ p : ℕ, prime p → p ∣ n → p ≤ 397

theorem largest_prime_divisor_is_397 : largest_prime_divisor_base7 :=
sorry

end largest_prime_divisor_is_397_l538_538802


namespace tan_585_eq_1_l538_538774

noncomputable def tan_deg (θ : ℝ) : ℝ := Real.tan (θ * Real.pi / 180)

theorem tan_585_eq_1 :
  tan_deg 585 = 1 := 
by
  have h1 : 585 - 360 = 225 := by norm_num
  have h2 : tan_deg 225 = tan_deg 45 :=
    by have h3 : 225 = 180 + 45 := by norm_num
       rw [h3, tan_deg]
       exact Real.tan_add_pi_div_two_simp_left (Real.pi * 45 / 180)
  rw [← tan_deg]
  rw [h1, h2]
  exact Real.tan_pi_div_four

end tan_585_eq_1_l538_538774


namespace trig_identity_l538_538844

theorem trig_identity (α m : ℝ) (h : Real.tan α = m) :
  (Real.sin (π / 4 + α))^2 - (Real.sin (π / 6 - α))^2 - Real.cos (5 * π / 12) * Real.sin (5 * π / 12 - 2 * α) = 2 * m / (1 + m^2) :=
by
  sorry

end trig_identity_l538_538844


namespace find_tricksters_in_16_questions_l538_538332

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

end find_tricksters_in_16_questions_l538_538332


namespace sum_of_roots_of_quadratic_l538_538519

theorem sum_of_roots_of_quadratic:
  (m n : ℝ) (h : ∀ x, (x = m ∨ x = n) ↔ (x^2 - 3*x - 2 = 0)) :
  m + n = 3 :=
by
  sorry

end sum_of_roots_of_quadratic_l538_538519


namespace angle_between_diagonals_l538_538198

open Real

theorem angle_between_diagonals
  (a b c : ℝ) :
  ∃ θ : ℝ, θ = arccos (a^2 / sqrt ((a^2 + b^2) * (a^2 + c^2))) :=
by
  -- Placeholder for the proof
  sorry

end angle_between_diagonals_l538_538198


namespace minimum_distance_l538_538123

noncomputable def distance (M Q : ℝ × ℝ) : ℝ :=
  ( (M.1 - Q.1) ^ 2 + (M.2 - Q.2) ^ 2 ) ^ (1 / 2)

theorem minimum_distance (M : ℝ × ℝ) :
  ∃ Q : ℝ × ℝ, ( (Q.1 - 1) ^ 2 + Q.2 ^ 2 = 1 ) ∧ distance M Q = 1 :=
sorry

end minimum_distance_l538_538123


namespace combined_cost_of_one_item_l538_538634

-- Definitions representing the given conditions
def initial_amount : ℝ := 50
def final_amount : ℝ := 14
def mangoes_purchased : ℕ := 6
def apple_juice_purchased : ℕ := 6

-- Hypothesis: The cost of mangoes and apple juice are the same
variables (M A : ℝ)

-- Total amount spent
def amount_spent : ℝ := initial_amount - final_amount

-- Combined number of items
def total_items : ℕ := mangoes_purchased + apple_juice_purchased

-- Lean statement to prove the combined cost of one mango and one carton of apple juice is $3
theorem combined_cost_of_one_item (h : mangoes_purchased * M + apple_juice_purchased * A = amount_spent) :
    (amount_spent / total_items) = (3 : ℝ) :=
by
  sorry

end combined_cost_of_one_item_l538_538634


namespace find_tricksters_l538_538349

theorem find_tricksters (inhabitants : Fin 65 → Prop) (is_knight : Fin 65 → Prop)
    (total_inhabitants : ∀ n, inhabitants n)
    (knights : ∀ n, is_knight n → inhabitants n)
    (tricksters_count : (∑ n, if ¬ is_knight n then 1 else 0) = 2)
    (knights_count : (∑ n, if is_knight n then 1 else 0) = 63)
    (knight_truth : ∀ n, is_knight n → ∀ l : list (Fin 65), (∀ m ∈ l, is_knight m) ↔ true)
    (ask_question : ∀ n, inhabitants n → ∀ l : list (Fin 65), bool) :
  ∃ (find_tricksters_function : (Fin 65 → Prop) → (Fin 65 → bool) → (list (Fin 65))) ,
    (length (find_tricksters_function inhabitants ask_question) ≤ 2) →
    (length (find_tricksters_function inhabitants ask_question) = 2) ∧
    ∀ t ∈ (find_tricksters_function inhabitants ask_question), ¬ is_knight t :=
by sorry

end find_tricksters_l538_538349


namespace students_with_average_age_of_16_l538_538652

theorem students_with_average_age_of_16
  (N : ℕ) (A : ℕ) (N14 : ℕ) (A15 : ℕ) (N16 : ℕ)
  (h1 : N = 15) (h2 : A = 15) (h3 : N14 = 5) (h4 : A15 = 11) :
  N16 = 9 :=
sorry

end students_with_average_age_of_16_l538_538652


namespace negation_example_l538_538672

open_locale classical

theorem negation_example:
  (∃ x : ℝ, x^2 + x + 1 ≤ 0) ↔ ¬ (∀ x : ℝ, x^2 + x + 1 > 0) :=
by sorry

end negation_example_l538_538672


namespace length_MN_eq_four_length_MF_eq_val_l538_538065

-- Define the parabola with parameter p > 0
variables {p : ℝ} (hp : p > 0)

-- Define the points O, F, M, and N
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (p / 2, 0)
def M : ℝ × ℝ := (p / 4, (real.sqrt 2 / 2) * p)
def N : ℝ × ℝ := (p / 4, -(real.sqrt 2 / 2) * p)

-- Define the length of the segment MN
def length_MN : ℝ := real.sqrt ((M.1 - N.1) ^ 2 + (M.2 - N.2) ^ 2)

-- Define the length of the segment MF
def length_MF : ℝ := real.sqrt ((M.1 - F.1) ^ 2 + (M.2 - F.2) ^ 2)

-- Theorems related to distances and conditions
theorem length_MN_eq_four (hMN : length_MN hp = 4) : p = 2 * real.sqrt 2 :=
by {
  sorry
}

theorem length_MF_eq_val (hp_pos : hp) (hMF : length_MF (hp := length_MN_eq_four hp_pos)) : length_MF hp_pos = 3 * real.sqrt 2 / 2 :=
by {
  sorry
}

end length_MN_eq_four_length_MF_eq_val_l538_538065


namespace minimum_Sqrt3_l538_538153

theorem minimum_Sqrt3 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^2 + y^2 + z^2 = 1) :
  ∃ s : ℝ, s = (xy / z) + (yz / x) + (zx / y) ∧ s ≥ sqrt3 ∧ (∀ t : ℝ, t < sqrt3 → ¬ (t = (xy / z) + (yz / x) + (zx / y))) :=
begin
  sorry
end

end minimum_Sqrt3_l538_538153


namespace num_combinations_L_shape_l538_538298

theorem num_combinations_L_shape (n : ℕ) (k : ℕ) (grid_size : ℕ) (L_shape_blocks : ℕ) 
  (h1 : n = 6) (h2 : k = 4) (h3 : grid_size = 36) (h4 : L_shape_blocks = 4) : 
  ∃ (total_combinations : ℕ), total_combinations = 1800 := by
  sorry

end num_combinations_L_shape_l538_538298


namespace faster_speed_l538_538293

theorem faster_speed (x : ℝ) (h1 : 40 = 8 * 5) (h2 : 60 = x * 5) : x = 12 :=
sorry

end faster_speed_l538_538293


namespace correct_conclusion_l538_538281

def num_students := 270
def first_grade_students := 108
def second_grade_students := 81
def third_grade_students := 81
def sample_size := 10

def sample1 := [7, 34, 61, 88, 115, 142, 169, 196, 223, 250]
def sample2 := [5, 9, 100, 107, 111, 121, 180, 195, 200, 265]
def sample3 := [11, 38, 65, 92, 119, 146, 173, 200, 227, 254]
def sample4 := [30, 57, 84, 111, 138, 165, 192, 219, 246, 270]

-- Define conditions for stratified sampling
def is_stratified (sample : list ℕ) : Prop := -- Insert appropriate condition for stratified sampling
  sorry

-- Define conditions for systematic sampling
def is_systematic (sample : list ℕ) : Prop := -- Insert appropriate condition for systematic sampling
  sorry

theorem correct_conclusion :
  (is_stratified sample1 ∧ is_stratified sample3) ∧ 
  ¬(is_systematic sample2 ∧ is_systematic sample3) ∧
  ¬(is_stratified sample2 ∧ is_stratified sample4) ∧
  ¬(is_systematic sample1 ∧ is_systematic sample4) :=
 sorry

end correct_conclusion_l538_538281


namespace tricksters_identification_l538_538344

variable (Inhabitant : Type)
variable [inhab : Fintype Inhabitant]
variable [decEqInhab : DecidableEq Inhabitant]
variable (knight : Inhabitant → Prop)
variable (trickster : Inhabitant → Prop)

variables (n : ℕ) (q : ℕ)
variable (is_truthful : Inhabitant → Prop)

constant inhabitants_count : 65
constant tricksters_count : 2

-- Define the property that a knight always tells the truth.
axiom knight_truth (x : Inhabitant) : knight x → is_truthful x

-- Define the property that a trickster can tell the truth or lie.
axiom trickster_behavior (x : Inhabitant) : trickster x → (is_truthful x ∨ ¬ is_truthful x)

 -- Define the type of the question which can be asked to an inhabitant.
inductive Question (Inhabitant : Type) : Type
| is_knight : Inhabitant → Question

-- Define the type of the answer to the question.
inductive Answer (Inhabitant : Type) : Type
| yes : Answer
| no :  Answer

-- Define a function that simulates asking a question to an inhabitant.
constant ask : Inhabitant → Question Inhabitant → Answer Inhabitant

noncomputable def find_tricksters (inhabitants : fin inhabitants_count → Inhabitant) : (fin tricksters_count → Inhabitant) :=
sorry

theorem tricksters_identification : 
  ∃ (f : (fin inhabitants_count → Inhabitant) → (fin tricksters_count → Inhabitant)), 
    ∀ inhabitants : fin inhabitants_count → Inhabitant, 
      (∀ (q_list : list (Inhabitant × Question Inhabitant)),
        q_list.length ≤ 30 → 
        let a_list := q_list.map (λ pq, ask (pq.fst) (pq.snd)) in 
        true) ∧ 
      (∃ t1 t2, trickster (f inhabitants) t1 ∧ trickster (f inhabitants) t2) :=
sorry

end tricksters_identification_l538_538344


namespace john_will_lose_weight_in_80_days_l538_538566

-- Assumptions based on the problem conditions
def calories_eaten : ℕ := 1800
def calories_burned : ℕ := 2300
def calories_to_lose_one_pound : ℕ := 4000
def pounds_to_lose : ℕ := 10

-- Definition of the net calories burned per day
def net_calories_burned_per_day : ℕ := calories_burned - calories_eaten

-- Definition of total calories to lose the target weight
def total_calories_to_lose_target_weight (pounds_to_lose : ℕ) : ℕ :=
  calories_to_lose_one_pound * pounds_to_lose

-- Definition of days to lose the target weight
def days_to_lose_weight (target_calories : ℕ) (daily_net_calories : ℕ) : ℕ :=
  target_calories / daily_net_calories

-- Prove that John will lose 10 pounds in 80 days
theorem john_will_lose_weight_in_80_days :
  days_to_lose_weight (total_calories_to_lose_target_weight pounds_to_lose) net_calories_burned_per_day = 80 := by
  sorry

end john_will_lose_weight_in_80_days_l538_538566


namespace find_tricksters_in_16_questions_l538_538330

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

end find_tricksters_in_16_questions_l538_538330


namespace find_m_when_f_is_odd_l538_538103

theorem find_m_when_f_is_odd (m : ℝ) :
  (∀ x : ℝ, f x = 1 / (x - 2 * m + 1) ∧ 
            (∀ x : ℝ, f (-x) = -f (x))) → 
  m = 1 / 2 :=
by
  sorry

end find_m_when_f_is_odd_l538_538103


namespace sin_690_deg_l538_538425

noncomputable def sin_690_eq_neg_one_half : Prop :=
  sin (690 * real.pi / 180) = -(1 / 2)

theorem sin_690_deg : sin_690_eq_neg_one_half :=
  by sorry

end sin_690_deg_l538_538425


namespace distance_between_A_and_B_l538_538924

def distance (A B : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.3 - B.3)^2)

def point_A : ℝ × ℝ × ℝ := (1, -2, 3)
def point_B : ℝ × ℝ × ℝ := (0, 1, -1)

theorem distance_between_A_and_B : distance point_A point_B = real.sqrt 26 :=
by
  sorry

end distance_between_A_and_B_l538_538924


namespace tan_ratio_l538_538590

open Real

variables (p q : ℝ)

-- Conditions
def cond1 := (sin p / cos q + sin q / cos p = 2)
def cond2 := (cos p / sin q + cos q / sin p = 3)

-- Proof statement
theorem tan_ratio (hpq : cond1 p q) (hq : cond2 p q) :
  (tan p / tan q + tan q / tan p = 8 / 5) :=
sorry

end tan_ratio_l538_538590


namespace ursula_change_correct_l538_538696

noncomputable def hot_dogs_cost : ℕ → ℝ → ℝ
| n, price_per_item :=
  let paid_for = n - (n / 5) in
  paid_for * price_per_item

noncomputable def salads_cost : ℕ → ℝ → ℝ → ℝ
| n, price_per_item, discount :=
  let raw_cost = n * price_per_item in
  raw_cost - (discount * raw_cost)

noncomputable def soft_drinks_cost : ℕ → ℝ → ℝ
| n, price_per_item := n * price_per_item

noncomputable def pretzel_cost : ℝ → ℝ → ℝ
| price, discount := price - (price * discount)

noncomputable def ice_creams_cost : ℕ → ℝ → ℝ → ℝ
| n, price_per_item, tax :=
  let paid_for = n - (n / 3) in
  let raw_cost = paid_for * price_per_item in
  raw_cost + (tax * raw_cost)

noncomputable def total_cost (hd_cost sal_cost sd_cost pre_cost ic_cost : ℝ) : ℝ :=
  hd_cost + sal_cost + sd_cost + pre_cost + ic_cost

def ursula_change : ℝ → ℝ → ℝ := λ total_money total_cost => total_money - total_cost

noncomputable def problem_solution : ℝ :=
  let hot_dogs = hot_dogs_cost 5 1.50,
      salads = salads_cost 3 2.50 0.10,
      soft_drinks = soft_drinks_cost 2 1.25,
      pretzel = pretzel_cost 2.00 0.20,
      ice_creams = ice_creams_cost 4 1.75 0.05,
      total = total_cost hot_dogs salads soft_drinks pretzel ice_creams,
      change = ursula_change 40.00 total in
  change

theorem ursula_change_correct : problem_solution = 17.64 := by
  sorry

end ursula_change_correct_l538_538696


namespace range_of_t_l538_538109

theorem range_of_t (α β t : ℝ)
  (h₁ : t = cos β ^ 3 + (α / 2) * cos β)
  (h₂ : α - 5 * cos β ≤ t)
  (h₃ : t ≤ α - 5 * cos β) :
  -2/3 ≤ t ∧ t ≤ 1 :=
sorry

end range_of_t_l538_538109


namespace pyramid_surface_area_l538_538949

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
let s := (a + b + c) / 2 in
Real.sqrt (s * (s - a) * (s - b) * (s - c))

def isosceles_triangle_area (a b c : ℝ) : ℝ :=
if a = b then
  let h := Real.sqrt (c^2 - (a / 2)^2) in
  (1 / 2) * a * h
else if a = c then
  let h := Real.sqrt (b^2 - (a / 2)^2) in
  (1 / 2) * a * h
else
  let h := Real.sqrt (a^2 - (b / 2)^2) in
  (1 / 2) * b * h

theorem pyramid_surface_area 
  (a b c : ℝ) 
  (h_a : a = 10) 
  (h_b : b = 20) 
  (h_c : c = 10) :
  4 * (isosceles_triangle_area a b c) = 100 * Real.sqrt 15 :=
by
  sorry

end pyramid_surface_area_l538_538949


namespace ramu_profit_percent_l538_538980

noncomputable def carCost : ℝ := 42000
noncomputable def repairCost : ℝ := 13000
noncomputable def sellingPrice : ℝ := 60900
noncomputable def totalCost : ℝ := carCost + repairCost
noncomputable def profit : ℝ := sellingPrice - totalCost
noncomputable def profitPercent : ℝ := (profit / totalCost) * 100

theorem ramu_profit_percent : profitPercent = 10.73 := 
by
  sorry

end ramu_profit_percent_l538_538980


namespace p_necessary_not_sufficient_for_q_l538_538474

-- Define the necessary conditions
axiom Line (a: Type) -- a represents a line
axiom Plane (α: Type) -- α represents a plane
axiom PerpendicularLineToLinesInPlane : Line → Plane → Prop -- a is perpendicular to Countless lines within plane α
axiom PerpendicularLineToPlane : Line → Plane → Prop -- Line a is perpendicular to plane α

-- Define the given conditions p and q
def p (a : Line) (α : Plane) : Prop := PerpendicularLineToLinesInPlane a α -- p is a predicate meaning Line a is perpendicular to countless lines within plane α.
def q (a : Line) (α : Plane) : Prop := PerpendicularLineToPlane a α -- q means Line a is perpendicular to plane α.

theorem p_necessary_not_sufficient_for_q (a : Line) (α : Plane) : (p a α) → ¬(q a α) :=
by
  intros hpa
  -- Proof omitted
  sorry

end p_necessary_not_sufficient_for_q_l538_538474


namespace part_a_part_b_l538_538740

/-- Pedrinho's game definition -/
def can_end_with_all_three_stone_stacks (initial_stones : ℕ) : Prop :=
∀ (s : finset ℕ), -- some set of stacks
  (∀ x ∈ s, x ≥ 3) →  
  (∀ x ∈ s, ∃ n : ℕ, x = 3 * n + 1) →  
  ∃ t : finset ℕ, (t.card = 3) ∧ (∀ x ∈ t, x = 3)

/-- Part (a): Prove that starting with 19 stones, Pedrinho can end up with stacks of 3 stones each -/
theorem part_a : can_end_with_all_three_stone_stacks 19 :=
sorry

/-- Part (b): Prove that starting with 1001 stones, Pedrinho cannot end up with stacks of 3 stones each -/
theorem part_b : ¬ can_end_with_all_three_stone_stacks 1001 :=
sorry

end part_a_part_b_l538_538740


namespace a_varies_inversely_with_b_squared_l538_538187

theorem a_varies_inversely_with_b_squared {a b : ℝ} (h1 : a * (2 : ℝ)^2 = 16) (h2 : 8 = (8^2 : ℝ) = 64) :
  let k := 16 in a = 1/4 := 
sorry

end a_varies_inversely_with_b_squared_l538_538187


namespace cos_theta_max_value_l538_538246

theorem cos_theta_max_value (θ : ℝ) (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = Real.sin x - 3 * Real.cos x)
  (h₁ : ∃ θ, f θ = f (θ + 2 * Int.pi * (n : ℕ)) + f (θ + Int.pi / 2) - f θ) :
  Real.cos θ = -3 * Real.sqrt 10 / 10 :=
sorry

end cos_theta_max_value_l538_538246


namespace specifically_intersecting_remainder_1000_l538_538785

open Finset

def is_specifically_intersecting (A B C : Finset ℕ) : Prop :=
  |A ∩ B| = 2 ∧ |B ∩ C| = 1 ∧ |C ∩ A| = 1 ∧ A ∩ B ∩ C = ∅

def specifically_intersecting_count (U : Finset ℕ) : ℕ :=
  (U.powerset.filter (λ A => 
     U.powerset.filter (λ B =>
        U.powerset.filter (λ C => is_specifically_intersecting A B C))).card).card

theorem specifically_intersecting_remainder_1000 :
  specifically_intersecting_count (range 1 10) % 1000 = 288 :=
by
  sorry

end specifically_intersecting_remainder_1000_l538_538785


namespace alice_correct_tuple_l538_538313

noncomputable def alice_tuple : Prop :=
  ∃ (a b c d : ℕ), a ≤ b ∧ b ≤ c ∧ c ≤ d ∧
    ({a * b + c * d, a * c + b * d, a * d + b * c} = {40, 70, 100}) ∧
    (a, b, c, d) = (1, 4, 6, 16)

-- The theorem statement asserting the only possible tuple that satisfies the conditions
theorem alice_correct_tuple : alice_tuple := 
by
  sorry

end alice_correct_tuple_l538_538313


namespace exactly_one_vs_two_white_l538_538821

-- Definitions based on conditions in the problem
def balls := ["red", "red", "white", "white"]

def event_one_white (draw: list string) : Prop :=
  draw.filter (· == "white") |>.length = 1

def event_two_white (draw: list string) : Prop :=
  draw.filter (· == "white") |>.length = 2

def mutually_exclusive (e1 e2 : list string → Prop) : Prop :=
  ∀ draw, ¬ (e1 draw ∧ e2 draw)

def complementary (e1 e2 : list string → Prop) : Prop :=
  ∀ draw, e1 draw ∨ e2 draw

 -- The Lean statement
theorem exactly_one_vs_two_white :
  mutually_exclusive event_one_white event_two_white ∧
  ¬ complementary event_one_white event_two_white := by
  sorry

end exactly_one_vs_two_white_l538_538821


namespace general_term_sum_first_n_terms_range_of_angle_l538_538067

noncomputable theory

variables {a : ℕ → ℕ} {b : ℕ → ℕ}
variables {sin cos : ℝ → ℝ} {T : ℕ → ℝ} {A : ℝ}

def a_seq (n : ℕ) : ℕ :=
  if n = 1 then 3 else 3 * a_seq (n - 1) + 3^n

def b_seq (n : ℕ) : ℕ :=
  (log 3 (a_seq n / n))

def T_seq (n : ℕ) : ℝ :=
  ∑ i in finset.range n, (1 / (b_seq i * b_seq (i + 1)))

theorem general_term (n : ℕ) (h_pos : n ≥ 2) : a n = n * 3^n :=
sorry

theorem sum_first_n_terms (n : ℕ) : 
  let S_n := ∑ i in finset.range n, a_seq i
  in S_n = (3 / 4:ℝ) + ((2 * n - 1) / 4:ℝ) * 3^(n + 1) :=
sorry

theorem range_of_angle (h : (1 / 2) * sin (2 * A) > (sqrt 3 / 4) * T_seq n):
  A ∈ (set.Icc (π / 6) (π / 3)) :=
sorry

end general_term_sum_first_n_terms_range_of_angle_l538_538067


namespace part1_part2_l538_538831

noncomputable def triangleABC (a : ℝ) (cosB : ℝ) (b : ℝ) (SinA : ℝ) : Prop :=
  cosB = 3 / 5 ∧ b = 4 → SinA = 2 / 5

noncomputable def triangleABC2 (a : ℝ) (cosB : ℝ) (S : ℝ) (b c : ℝ) : Prop :=
  cosB = 3 / 5 ∧ S = 4 → b = Real.sqrt 17 ∧ c = 5

theorem part1 :
  triangleABC 2 (3 / 5) 4 (2 / 5) :=
by {
  sorry
}

theorem part2 :
  triangleABC2 2 (3 / 5) 4 (Real.sqrt 17) 5 :=
by {
  sorry
}

end part1_part2_l538_538831


namespace walking_speed_l538_538889

theorem walking_speed (x : ℝ) (h1 : 20 / x = 40 / (x + 5)) : x + 5 = 10 :=
  by
  sorry

end walking_speed_l538_538889


namespace min_pencils_to_ensure_18_l538_538655

theorem min_pencils_to_ensure_18 :
  ∀ (total red green yellow blue brown black : ℕ),
  total = 120 → red = 35 → green = 23 → yellow = 14 → blue = 26 → brown = 11 → black = 11 →
  ∃ (n : ℕ), n = 88 ∧
  (∀ (picked_pencils : ℕ → ℕ), (
    (picked_pencils 0 + picked_pencils 1 + picked_pencils 2 + picked_pencils 3 + picked_pencils 4 + picked_pencils 5 = n) →
    (picked_pencils 0 ≤ red) → (picked_pencils 1 ≤ green) → (picked_pencils 2 ≤ yellow) →
    (picked_pencils 3 ≤ blue) → (picked_pencils 4 ≤ brown) → (picked_pencils 5 ≤ black) →
    (picked_pencils 0 ≥ 18 ∨ picked_pencils 1 ≥ 18 ∨ picked_pencils 2 ≥ 18 ∨ picked_pencils 3 ≥ 18 ∨ picked_pencils 4 ≥ 18 ∨ picked_pencils 5 ≥ 18)
  )) := 
sorry

end min_pencils_to_ensure_18_l538_538655


namespace area_ratio_greater_two_ninths_l538_538556

variable (A B C P Q R : Type) [LinearOrder P] [LinearOrder Q] [LinearOrder R]

constant triangle_ABC : Triangle A B C
constant points_on_AB : PointsOnAB P Q triangle_ABC
constant perimeter_divided_three_parts : PerimeterDividedThreeParts P Q R triangle_ABC

theorem area_ratio_greater_two_ninths :
  (area (Triangle P Q R)) / (area (Triangle A B C)) > 2/9 :=
sorry

end area_ratio_greater_two_ninths_l538_538556


namespace cos_angle_BAC_l538_538038

theorem cos_angle_BAC (O : Type) [circumcenter O] (A B C : Type) [acute_triangle A B C]
  (AB AC : ℝ) (x y : ℝ) (AO : vector_representation A B C)
  (h1 : AB = 6) (h2 : AC = 10) (h3 : AO = x * vector_AB + y * vector_AC)
  (h4 : 2 * x + 10 * y = 5) : 
  cos (angle BAC) = 1 / 3 := 
by 
  sorry

end cos_angle_BAC_l538_538038


namespace power_of_fraction_to_decimal_l538_538245

theorem power_of_fraction_to_decimal : ∃ x : ℕ, (1 / 9 : ℚ) ^ x = 1 / 81 ∧ x = 2 :=
by
  use 2
  simp
  sorry

end power_of_fraction_to_decimal_l538_538245


namespace swimming_speed_in_still_water_l538_538284

theorem swimming_speed_in_still_water 
  (v : ℝ) -- The man's swimming speed in still water
  (s : ℝ) -- The speed of the stream
  (H1 : s = 1.5) -- The speed of the stream is 1.5 km/h
  (H2 : (d : ℝ) (h : d > 0) → (d / (v - s) = 2 * (d / (v + s)))) -- It takes twice as long to swim upstream than downstream
  : v = 4.5 := 
sorry

end swimming_speed_in_still_water_l538_538284


namespace number_of_valid_a_l538_538902

theorem number_of_valid_a :
  (∃ a : ℤ, 
    (∃ x : ℕ, (x + a) / (x - 2) + (2 * x) / (2 - x) = 1) ∧
    (∃ s : Finset ℤ, s.card = 4 ∧ ∀ y ∈ s, 
      (y + 1) / 5 ≥ y / 2 - 1 ∧ 
      y + a < 11 * y - 3)
  ) = 4 := sorry

end number_of_valid_a_l538_538902


namespace combination_identity_l538_538792

theorem combination_identity (C : ℕ → ℕ → ℕ)
  (comb_formula : ∀ n r, C r n = Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r)))
  (identity_1 : ∀ n r, C r n = C (n-r) n)
  (identity_2 : ∀ n r, C r (n+1) = C r n + C (r-1) n) :
  C 2 100 + C 97 100 = C 3 101 :=
by sorry

end combination_identity_l538_538792


namespace calvin_gym_duration_l538_538395

theorem calvin_gym_duration (initial_weight loss_per_month final_weight : ℕ) (h1 : initial_weight = 250)
    (h2 : loss_per_month = 8) (h3 : final_weight = 154) : 
    (initial_weight - final_weight) / loss_per_month = 12 :=
by 
  sorry

end calvin_gym_duration_l538_538395


namespace number_of_correct_answers_l538_538116

theorem number_of_correct_answers (C W : ℕ) (h1 : C + W = 100) (h2 : 5 * C - 2 * W = 210) : C = 58 :=
sorry

end number_of_correct_answers_l538_538116


namespace sum_fractions_bounds_l538_538589

variable {α : Type*} [LinearOrder α] [OrderedAddCommGroup α] [OrderedCommRing α] [FloorRing α] -- to handle real numbers and positivity constraints

theorem sum_fractions_bounds 
    (n : ℕ) 
    (h_pos : 3 ≤ n) 
    (x : Fin n → α) 
    (h_x : ∀ i, 0 < x i) : 
    1 < ∑ i : Fin n, x i / (x i + x (i + 1) % n) ∧ ∑ i : Fin n, x i / (x i + x (i + 1) % n) < n - 1 :=
by 
    sorry

end sum_fractions_bounds_l538_538589


namespace sin_690_eq_neg_half_l538_538406

theorem sin_690_eq_neg_half :
  Real.sin (690 * Real.pi / 180) = -1 / 2 :=
by {
  sorry
}

end sin_690_eq_neg_half_l538_538406


namespace sports_minutes_in_newscast_l538_538663

-- Definitions based on the conditions
def total_newscast_minutes : ℕ := 30
def national_news_minutes : ℕ := 12
def international_news_minutes : ℕ := 5
def weather_forecasts_minutes : ℕ := 2
def advertising_minutes : ℕ := 6

-- The problem statement
theorem sports_minutes_in_newscast (t : ℕ) (n : ℕ) (i : ℕ) (w : ℕ) (a : ℕ) :
  t = 30 → n = 12 → i = 5 → w = 2 → a = 6 → t - n - i - w - a = 5 := 
by sorry

end sports_minutes_in_newscast_l538_538663


namespace triangle_right_angle_l538_538030

theorem triangle_right_angle (α β γ : ℝ) (h : cos α + cos β = sin α + sin β) 
  (h_sum : α + β + γ = π) : γ = π / 2 :=
sorry

end triangle_right_angle_l538_538030


namespace triangle_AC_len_l538_538480

theorem triangle_AC_len 
  (A B C D E F H : Type) 
  [h1 : ∃ (P : Type), is_triangle_side P A B 3] 
  [h2 : ∃ (Q : Type), is_triangle_side Q B C 4] 
  (midpoints_collinear : ∃ (R : Type), are_midpoints_of_altitudes_collinear R D E F) 
  : length_side A C = 5 :=
sorry

end triangle_AC_len_l538_538480


namespace combinedAverageAge_l538_538653

-- Definitions
def numFifthGraders : ℕ := 50
def avgAgeFifthGraders : ℕ := 10
def numParents : ℕ := 75
def avgAgeParents : ℕ := 40

-- Calculation of total ages
def totalAgeFifthGraders := numFifthGraders * avgAgeFifthGraders
def totalAgeParents := numParents * avgAgeParents
def combinedTotalAge := totalAgeFifthGraders + totalAgeParents

-- Calculation of total number of individuals
def totalIndividuals := numFifthGraders + numParents

-- The claim to prove
theorem combinedAverageAge : 
  combinedTotalAge / totalIndividuals = 28 := by
  -- Skipping the proof details.
  sorry

end combinedAverageAge_l538_538653


namespace solve_absolute_value_equation_l538_538182

theorem solve_absolute_value_equation (y : ℝ) : (|y - 4| + 3 * y = 15) ↔ (y = 19 / 4) := by
  sorry

end solve_absolute_value_equation_l538_538182


namespace sufficient_condition_for_monotonicity_l538_538156

theorem sufficient_condition_for_monotonicity (a : ℝ) (h : a < 3) : 
  ∀ x : ℝ, 1 ≤ x → deriv (λ x, x^2 + x - a * real.log x) x ≥ 0 :=
by
  intro x hx
  have h_deriv : deriv (λ x, x^2 + x - a * real.log x) x = 2 * x + 1 - a / x :=
    by simp [deriv_pow, deriv_id', deriv_log, deriv_sub, deriv_const, deriv_add]
  rw h_deriv
  calc
    2 * x + 1 - a / x ≥ 2 * 1 + 1 - a / 1 :  by linarith [hx, h]
                ... = 3 - a : by simp
                ... ≥ 0 : by linarith [le_of_lt h]

end sufficient_condition_for_monotonicity_l538_538156


namespace A_in_terms_of_B_l538_538957

variables (A B x : ℝ)
variables (f g : ℝ → ℝ)

def f_def := ∀ x : ℝ, f x = A * x - 2 * B^2
def g_def := ∀ x : ℝ, g x = B * x^2
def B_nonzero := B ≠ 0
def fg1_zero := f (g 1) = 0

theorem A_in_terms_of_B (h1 : f_def f) (h2 : g_def g) (h3 : B_nonzero) (h4 : fg1_zero) : A = 2 * B := 
sorry

end A_in_terms_of_B_l538_538957


namespace slope_angle_tangent_curve_l538_538679

theorem slope_angle_tangent_curve (x : ℝ) (h : x = 1) :
  let y := (1/3) * x^3 - 2 in
  let y' := x^2 in
  Real.atan y' = π / 4 :=
by
  let y := (1/3) * x^3 - 2
  let y' := x^2
  have hx : h : x = 1, from h
  sorry

end slope_angle_tangent_curve_l538_538679


namespace total_surface_area_of_solid_l538_538440

theorem total_surface_area_of_solid
  (base_layer_cubes : fin 12 → unit)
  (second_layer_cubes : fin 3 → unit)
  (base_layer_arranged : (∃ (base : fin 12 → Prop), ∀ i, base_layer_cubes i -> base i))
  (second_layer_centered : (∃ (second : fin 3 → Prop), ∀ i, second_layer_cubes i -> second i))
  : (surface_area base_layer_cubes second_layer_cubes base_layer_arranged second_layer_centered) = 36 := 
sorry

def surface_area (base_layer_cubes : fin 12 → unit) 
                  (second_layer_cubes : fin 3 → unit) 
                  (base_layer_arranged : ∃ (base : fin 12 → Prop), ∀ i, base_layer_cubes i -> base i) 
                  (second_layer_centered: ∃ (second : fin 3 → Prop), ∀ i, second_layer_cubes i -> second i) 
                  : ℕ :=
-- compute the surface area of the solid formed by the base and second layer cubes in the given arrangement
-- This part is skipped and needs to be defined
sorry

end total_surface_area_of_solid_l538_538440


namespace john_will_lose_weight_in_80_days_l538_538565

-- Assumptions based on the problem conditions
def calories_eaten : ℕ := 1800
def calories_burned : ℕ := 2300
def calories_to_lose_one_pound : ℕ := 4000
def pounds_to_lose : ℕ := 10

-- Definition of the net calories burned per day
def net_calories_burned_per_day : ℕ := calories_burned - calories_eaten

-- Definition of total calories to lose the target weight
def total_calories_to_lose_target_weight (pounds_to_lose : ℕ) : ℕ :=
  calories_to_lose_one_pound * pounds_to_lose

-- Definition of days to lose the target weight
def days_to_lose_weight (target_calories : ℕ) (daily_net_calories : ℕ) : ℕ :=
  target_calories / daily_net_calories

-- Prove that John will lose 10 pounds in 80 days
theorem john_will_lose_weight_in_80_days :
  days_to_lose_weight (total_calories_to_lose_target_weight pounds_to_lose) net_calories_burned_per_day = 80 := by
  sorry

end john_will_lose_weight_in_80_days_l538_538565


namespace total_cost_l538_538599

-- Given conditions
def pen_cost : ℕ := 4
def briefcase_cost : ℕ := 5 * pen_cost

-- Theorem stating the total cost Marcel paid for both items
theorem total_cost (pen_cost briefcase_cost : ℕ) (h_pen: pen_cost = 4) (h_briefcase: briefcase_cost = 5 * pen_cost) :
  pen_cost + briefcase_cost = 24 := by
  sorry

end total_cost_l538_538599


namespace sin_690_eq_negative_one_half_l538_538413

theorem sin_690_eq_negative_one_half : Real.sin (690 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_690_eq_negative_one_half_l538_538413


namespace total_cost_proof_l538_538604

-- Define the prices of items
def price_coffee : ℕ := 4
def price_cake : ℕ := 7
def price_ice_cream : ℕ := 3

-- Define the number of items ordered by Mell and her friends
def mell_coffee : ℕ := 2
def mell_cake : ℕ := 1
def friend_coffee : ℕ := 2
def friend_cake : ℕ := 1
def friend_ice_cream : ℕ := 1
def number_of_friends : ℕ := 2

-- Calculate total cost for Mell
def total_mell : ℕ := (mell_coffee * price_coffee) + (mell_cake * price_cake)

-- Calculate total cost per friend
def total_friend : ℕ := (friend_coffee * price_coffee) + (friend_cake * price_cake) + (friend_ice_cream * price_ice_cream)

-- Calculate total cost for all friends
def total_friends : ℕ := number_of_friends * total_friend

-- Calculate total cost for Mell and her friends
def total_cost : ℕ := total_mell + total_friends

-- The theorem to prove
theorem total_cost_proof : total_cost = 51 := by
  sorry

end total_cost_proof_l538_538604


namespace extreme_value_f_sum_of_roots_g_l538_538466

def f (x m : ℝ) : ℝ := Real.log x - x + m
def g (x m : ℝ) : ℝ := f (x + m) m
def h (x : ℝ) : ℝ := Real.exp x - x

theorem extreme_value_f (m : ℝ) : ∀ x, (f x m).maximum = m - 1 := 
sorry

theorem sum_of_roots_g (m x1 x2 : ℝ) (h_m : m > 1) (h_g1 : g x1 m = 0) (h_g2 : g x2 m = 0) : x1 + x2 < 0 :=
sorry

end extreme_value_f_sum_of_roots_g_l538_538466


namespace train_speed_l538_538747

def length_of_train : ℝ := 150
def time_to_cross_pole : ℝ := 9

def speed_in_m_per_s := length_of_train / time_to_cross_pole
def speed_in_km_per_hr := speed_in_m_per_s * (3600 / 1000)

theorem train_speed : speed_in_km_per_hr = 60 := by
  -- Length of train is 150 meters
  -- Time to cross pole is 9 seconds
  -- Speed in m/s = 150 meters / 9 seconds = 16.67 m/s
  -- Speed in km/hr = 16.67 m/s * 3.6 = 60 km/hr
  sorry

end train_speed_l538_538747


namespace situps_problem_l538_538385

theorem situps_problem (B : ℝ)
  (h1 : Carrie_situps_per_min := 2 * B)
  (h2 : Jerrie_situps_per_min := 2 * B + 5)
  (total_situps : B + 2 * Carrie_situps_per_min * 2 + Jerrie_situps_per_min * 3 = 510) :
  B = 45 := by
  sorry

end situps_problem_l538_538385


namespace rhombus_area_l538_538445

noncomputable def area_of_rhombus (ABCD : Type) [rhombus ABCD] (R_ABD R_ACD : ℝ) (h₁ : R_ABD = 12.5) (h₂ : R_ACD = 25) : ℝ :=
  let a := 10
  let b := 20
  let area := (20 * 40) / 2
  area

theorem rhombus_area (ABCD : Type) [rhombus ABCD] (R_ABD R_ACD : ℝ) (h₁ : R_ABD = 12.5) (h₂ : R_ACD = 25) :
  area_of_rhombus ABCD R_ABD R_ACD h₁ h₂ = 400 :=
sorry

end rhombus_area_l538_538445


namespace correct_propositions_l538_538493

-- Proposition 1
def prop1 (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ↔ (a < x ∧ x < 3*a)

-- Proposition 2
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f(x + 1) = f(-(x + 1))
def prop2 (f : ℝ → ℝ) : Prop := even_function f → ∀ x, f(x) = f(2 - x)

-- Proposition 3
def prop3 (a : ℝ) : Prop := (∀ x, ¬ (|x - 4| + |x - 3| < a)) → a ≤ 1

-- Proposition 4
def prop4 (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x1 x2, f(x1) = a → f(x2) = a → x1 = x2

theorem correct_propositions (f : ℝ → ℝ) :
  ¬ (∀ a x, prop1 a x) ∧
  (prop2 f) ∧
  (∀ a, prop3 a) ∧
  (∀ a, prop4 f a) :=
by
  sorry

end correct_propositions_l538_538493


namespace tan_585_eq_1_l538_538775

theorem tan_585_eq_1 : Real.tan (585 * Real.pi / 180) = 1 := 
by
  sorry

end tan_585_eq_1_l538_538775


namespace counting_numbers_without_one_in_digit_l538_538714

theorem counting_numbers_without_one_in_digit
    (S := set_of (λ i : ℕ, i < 1000 ∧ ∀ j ∈ to_digits 10 i, j ≠ 1)) :
  S.card = 810 := sorry

end counting_numbers_without_one_in_digit_l538_538714


namespace expression_simplification_l538_538717

theorem expression_simplification : 
  sqrt 3 * real.cos (real.pi / 6) + (3 - real.pi) ^ 0 - 2 * real.tan (real.pi / 4) = 1 / 2 :=
begin
  sorry
end

end expression_simplification_l538_538717


namespace original_expression_equals_l538_538639

noncomputable def evaluate_expression (a : ℝ) : ℝ :=
  ( (a / (a + 2) + 1 / (a^2 - 4)) / ( (a - 1) / (a + 2) + 1 / (a - 2) ))

theorem original_expression_equals (a : ℝ) (h : a = 2 + Real.sqrt 2) :
  evaluate_expression a = (Real.sqrt 2 + 1) :=
sorry

end original_expression_equals_l538_538639


namespace range_of_m_l538_538528

theorem range_of_m (m : ℝ) :
  (∃ P : ℝ × ℝ, P = (1, 1) ∧ ∃ C : ℝ × ℝ → ℝ , C = (λ ⟨x, y⟩, x^2 + y^2 + m * x + m * y + 2) ∧
  ∀ P C, (λ x y, ∃ d r : ℝ, d = ∥P - (-m / 2, -m / 2)∥ ∧ r = real.sqrt (m ^ 2 / 2 - 2) ∧ d > r)) ↔ m > 2 :=
sorry

end range_of_m_l538_538528


namespace triangle_angles_l538_538265

theorem triangle_angles (α β γ : ℝ) (A B C P Q : Type)
  [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq P] [decidable_eq Q]
  (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) (h3 : γ = π/2)
  (h4 : α + β + γ = π)
  (AP BP CP : ℝ)
  (hAP : AP = 4) (hBP : BP = 2) (hCP : CP = 1)
  (P_Q_symmetric_ac : symmetric P Q AC)
  (Q_on_circumcircle : on_circumcircle Q A B C)
  (angles_correct : (α = π/6 ∧ β = π/3 ∧ γ = π/2) ∨ (α = π/3 ∧ β = π/6 ∧ γ = π/2)) :
  α = π/6 ∧ β = π/3 ∧ γ = π/2 :=
sorry

end triangle_angles_l538_538265


namespace compensate_fairly_l538_538118

theorem compensate_fairly (p q y : ℕ) (h_p2 : p^2 = 10 * (2 * q + 1) + y) (h_y : 0 < y ∧ y < 10) (h_odd : (2 * q + 1).odd) :
  (10 - y) / 2 = 2 := by
  sorry

end compensate_fairly_l538_538118


namespace balanced_marking_on_triominoes_l538_538720

theorem balanced_marking_on_triominoes (m n : ℕ) (h1 : m = 2010) (h2 : n = 2010) (triominoes : set (set (ℕ × ℕ))) : 
(set.card (triominoes) = (m * n / 3)) ∧
∀ tr ∈ triominoes, (set.card tr = 3) ∧ (∃ i j, tr = { (i, j), (i+1, j), (i, j+1) } ∨ tr = { (i, j), (i, j+1), (i+1, j+1) } ∨ tr = { (i, j), (i+1, j), (i+1, j+1) }) →
∃ (mark : ℕ × ℕ → bool), (∀ tr ∈ triominoes, ∃! (i j : ℕ), (i, j) ∈ tr ∧ mark (i, j)) ∧ 
  ∀ k, (∑ i in (fin_range m), if mark (i, k) then 1 else 0 = ∑ j in (fin_range n), if mark (k, j) then 1 else 0) :=
by
  sorry

end balanced_marking_on_triominoes_l538_538720


namespace range_of_m_length_of_chord_MN_l538_538053

-- Proof for the range of m
theorem range_of_m (m : ℝ): (x y : ℝ) → (x^2 + y^2 - 2 * x - 4 * y + m = 0) → m < 5 :=
by sorry

-- Proof for the length of chord MN when m = 4
theorem length_of_chord_MN (x y : ℝ) (m : ℝ) (h : m = 4) (line_eq : x + 2 * y - 4 = 0):
  (length_MN (x, y) (1,2) = 4 * real.sqrt 5 / 5) :=
by sorry

end range_of_m_length_of_chord_MN_l538_538053


namespace minimum_apples_l538_538088

theorem minimum_apples (n : ℕ) (A : ℕ) (h1 : A = 25 * n + 24) (h2 : A > 300) : A = 324 :=
sorry

end minimum_apples_l538_538088


namespace AB_ge_nine_l538_538266

-- Define the sides of a triangle
variables (a b c : ℝ)
-- Conditions to ensure they form a triangle
variables (triangle_ineq1 : a + b > c) 
          (triangle_ineq2 : b + c > a) 
          (triangle_ineq3 : c + a > b)

-- Definition of A
def A := (a^2 + b*c) / (b + c) + 
         (b^2 + c*a) / (c + a) + 
         (c^2 + a*b) / (a + b)

-- Definition of B
def B := 1 / real.sqrt((a + b - c) * (b + c - a)) + 
         1 / real.sqrt((b + c - a) * (c + a - b)) + 
         1 / real.sqrt((c + a - b) * (a + b - c))

-- The theorem to be proved
theorem AB_ge_nine (hA : A ≥ a + b + c) (hB : B ≥ 1 / a + 1 / b + 1 / c) : A * B ≥ 9 := sorry

end AB_ge_nine_l538_538266


namespace coordinate_transformations_l538_538536

theorem coordinate_transformations
  (t α ρ θ: ℝ)
  (x y: ℝ → ℝ → ℝ)
  (C1: ∀ t α, Prop) 
  (C2: ∀ ρ θ, Prop)
  (MidPoint_AB: ∀ t1 t2, Prop)
  (proof1: C2 ρ θ → (x - 1)^2 + y^2 = 9)
  (proof2: MidPoint_AB t1 t2 → (x - 1)^2 + (y - (1/2))^2 = 1/4)
: (∀ t α, C1 t α) → (∀ ρ θ, C2 ρ θ) → 
  (C2 ρ θ → (x - 1)^2 + y^2 = 9) → 
  (MidPoint_AB t1 t2 → (x - 1)^2 + (y - (1/2))^2 = 1/4) → Prop :=
begin
  sorry
end

end coordinate_transformations_l538_538536


namespace typing_orders_count_l538_538914

theorem typing_orders_count :
  let S := {1, 2, 3, 4, 5, 6, 7, 8}
  let k := ∀ k ∈ powerset S
  ∑ k in S.powerset, (fintype.card k + 2) * (fintype.card {x // x ∈ S} k) = 1232 :=
by
  sorry

end typing_orders_count_l538_538914


namespace max_factors_of_bn_l538_538478

theorem max_factors_of_bn (b n : ℕ) (hb : b = 8) (hb_le : b ≤ 15) (hn_le : n ≤ 15) :
  ∃ k, k = 46 ∧ (∃ b n, b = 8 ∧ n ≤ 15 ∧ (∀ k, k = (if n = 15 then 46 else (45 + 1))) = b^n) :=
by
  sorry

end max_factors_of_bn_l538_538478


namespace triangle_angles_l538_538974

open Real

-- Definitions based on conditions
def is_right_triangle (A B C : Point) : Prop :=
  angle B A C = π / 2

def on_hypotenuse (A B C K : Point) : Prop :=
  is_right_triangle A B C ∧ dist C K = dist B C

def intersect_midpoint_at_angle_bisector (A B C K : Point) : Prop :=
  ∃ L, is_right_triangle A B C ∧ angle_bisector A B C = L ∧ midpoint K L ∈ line (C, K)

-- The theorem to be proven
theorem triangle_angles (A B C K : Point) (hK : on_hypotenuse A B C K)
  (hMid : intersect_midpoint_at_angle_bisector A B C K) :
  angle B A C = 36 ∧ angle A B C = 54 ∧ angle A C B = 90 :=
sorry

end triangle_angles_l538_538974


namespace infinite_triples_exist_l538_538760

variable {a : ℕ → ℕ}

-- Given conditions
axiom (h1 : ∀ n, n ≥ 1 → a (a n) ≤ a n + a (n + 3))
axiom (h2 : ∀ n m, n < m → a n < a m)

-- The theorem to prove
theorem infinite_triples_exist
  : ∃∞ k l m : ℕ, k < l ∧ l < m ∧ a k + a m = 2 * a l := sorry

end infinite_triples_exist_l538_538760


namespace probability_intervals_l538_538861

open ProbabilityTheory

noncomputable def standard_normal := measure_theory.measure_space.measure

theorem probability_intervals {p : ℝ} (h : p = P(λ x : ℝ, x > 1)) :
  P(λ x: ℝ, -1 < x ∧ x < 0) = 0.5 - p :=
by
  sorry

end probability_intervals_l538_538861


namespace apples_count_l538_538279

variable (A : ℕ)

axiom h1 : 134 = 80 + 54
axiom h2 : A + 98 = 134

theorem apples_count : A = 36 :=
by
  sorry

end apples_count_l538_538279


namespace coordinates_of_fixed_point_A_l538_538545

theorem coordinates_of_fixed_point_A :
  ∀ (B : ℝ × ℝ), B.2 < 0 → 
  ∀ (l : ℝ → ℝ), (∀ x, ∃ y, (y = l x) ∧ ((x^2) / 2 + y^2 = 1)) → 
  ∃ A : ℝ × ℝ, A = (0, 1) :=
by
  intros B B_y_neg l line_intersects_ellipse
  use (0, 1)
  sorry

end coordinates_of_fixed_point_A_l538_538545


namespace arithmetic_mean_multiplied_correct_l538_538235

-- Define the fractions involved
def frac1 : ℚ := 3 / 4
def frac2 : ℚ := 5 / 8

-- Define the arithmetic mean and the final multiplication result
def mean_and_multiply_result : ℚ := ( (frac1 + frac2) / 2 ) * 3

-- Statement to prove that the calculated result is equal to 33/16
theorem arithmetic_mean_multiplied_correct : mean_and_multiply_result = 33 / 16 := 
by 
  -- Skipping the proof with sorry for the statement only requirement
  sorry

end arithmetic_mean_multiplied_correct_l538_538235


namespace correct_answer_sum_valid_primes_l538_538240

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_prime (n : ℕ) : Prop := Nat.Prime n

def swap_digits (n : ℕ) (i j : ℕ) : ℕ :=
  let digits := repr n |>.toList |>.reverse
  if i < digits.length ∧ j < digits.length then
    let mut new_digits := digits
    let temp := new_digits.get i
    new_digits := new_digits.set i (new_digits.get j)
    new_digits := new_digits.set j temp
    repr (String.mk (new_digits.reverse)).toNat!
  else 0

def remains_prime_when_swap (n : ℕ) : Prop :=
  ∀ (i j : Fin 3), is_prime (swap_digits n i j)

def valid_prime_number (n : ℕ) : Prop :=
  is_three_digit n ∧ is_prime n ∧ remains_prime_when_swap n

def sum_valid_primes : ℕ :=
  Finset.universe.filter (λ n, valid_prime_number n) |>.sum

theorem correct_answer_sum_valid_primes :
  sum_valid_primes = 3424 :=
sorry

end correct_answer_sum_valid_primes_l538_538240


namespace hyperbola_condition_l538_538099

theorem hyperbola_condition (k : ℝ) : 
  (∃ (x y : ℝ), (x^2 / (4 + k) + y^2 / (1 - k) = 1)) ↔ (k < -4 ∨ k > 1) :=
by 
  sorry

end hyperbola_condition_l538_538099


namespace ratio_abcd_efgh_l538_538255

variable (a b c d e f g h : ℚ)

theorem ratio_abcd_efgh :
  (a / b = 1 / 3) ->
  (b / c = 2) ->
  (c / d = 1 / 2) ->
  (d / e = 3) ->
  (e / f = 1 / 2) ->
  (f / g = 5 / 3) ->
  (g / h = 4 / 9) ->
  (a * b * c * d) / (e * f * g * h) = 1 / 97 :=
by
  sorry

end ratio_abcd_efgh_l538_538255


namespace smallest_value_of_3a_plus_2_l538_538094

variable (a : ℝ)

theorem smallest_value_of_3a_plus_2 (h : 5 * a^2 + 7 * a + 2 = 1) : 3 * a + 2 = -1 :=
sorry

end smallest_value_of_3a_plus_2_l538_538094


namespace general_term_sequence_series_sum_bound_l538_538029

variable (a : ℕ → ℝ)

-- Given conditions
def sequence_condition (n : ℕ) : Prop :=
  (n ≥ 2) → (a n = (2^(n+1) + 2*a (n-1)) / 2)

theorem general_term_sequence (h1 : a 1 = 3) (h2 : ∀ n, sequence_condition a n) :
  ∀ n, a n = 2^(n+1) - 1 := sorry

theorem series_sum_bound (h1 : a 1 = 3) (h2 : ∀ n, sequence_condition a n) :
  ∀ n, (∑ i in Finset.range n, 1 / (a (i + 1) + 1)) < 1/2 := sorry

end general_term_sequence_series_sum_bound_l538_538029


namespace missing_digit_in_mean_l538_538993

theorem missing_digit_in_mean : 
  let N := [7, 77, 777, 999, 9999, 99999, 777777, 7777777, 99999999]
  let mean := (N.sum : Real) / N.length
  let rounded_mean := Int.ofNat (mean.floor)
  let digits := Set.univ.filter (fun x => x ∈ rounded_mean.digits)
  
  (digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} \ {2, 5, 8, 9}) := sorry

end missing_digit_in_mean_l538_538993


namespace distance_left_to_drive_l538_538610

theorem distance_left_to_drive (total_distance : ℕ) (distance_driven : ℕ) 
  (h1 : total_distance = 78) (h2 : distance_driven = 32) : 
  total_distance - distance_driven = 46 := by
  sorry

end distance_left_to_drive_l538_538610


namespace max_gcd_a_is_25_l538_538206

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 100 + n^2 + 2 * n

-- Define the gcd function
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

-- Define the theorem to prove the maximum value of d_n as 25
theorem max_gcd_a_is_25 : ∃ n : ℕ, d n = 25 := 
sorry

end max_gcd_a_is_25_l538_538206


namespace arithmetic_sequence_conditions_l538_538947

open Nat

theorem arithmetic_sequence_conditions (S : ℕ → ℤ) (d : ℤ) (a1 : ℤ) 
  (h1 : S 6 > S 7) (h2 : S 7 > S 5) :
  d < 0 ∧ S 11 > 0 := 
sorry

end arithmetic_sequence_conditions_l538_538947


namespace sin_690_degree_l538_538420

theorem sin_690_degree : sin (690 : ℝ) * (Real.pi / 180) = -(1 / 2) := by
  sorry

end sin_690_degree_l538_538420


namespace imaginary_part_of_z_l538_538018

def z : ℂ := (i : ℂ) / (1 - i)

theorem imaginary_part_of_z : z.im = 1/2 := by
  sorry

end imaginary_part_of_z_l538_538018


namespace arithmetic_mean_of_sixty_integers_beginning_at_3_l538_538648

theorem arithmetic_mean_of_sixty_integers_beginning_at_3 : 
  let seq (n : ℕ) := 3 + (n - 1)
  ∑ i in (finset.range 60).map (finset.ite 0), seq i / 60 = 32.5 := by
  sorry

end arithmetic_mean_of_sixty_integers_beginning_at_3_l538_538648


namespace factorization_l538_538798

def factor_polynomial : Prop :=
  ∃ (a b c d e f : ℤ), 
    a < d ∧
    (a * x^2 + b * x + c) * (d * x^2 + e * x + f) = x^2 - 6 * x + 9 - 64 * x^4 ∧
    (a * x^2 + b * x + c) = -8 * x^2 + x - 3 ∧
    (d * x^2 + e * x + f) = 8 * x^2 + x - 3

theorem factorization :
  factor_polynomial :=
begin
  use [-8, 1, -3, 8, 1, -3],
  split,
  { linarith },
  split,
  { ring },
  split;
  { ring },
end

end factorization_l538_538798


namespace minimum_PA_PF_value_l538_538734

noncomputable def minimum_distance (P A F : ℝ × ℝ) : ℝ :=
  let PA := real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PF := real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)
  PA + PF

theorem minimum_PA_PF_value : 
  (∀ (P : ℝ × ℝ), P.1 ^ 2 = 4 * P.2) ∧ 
  (A = (2, 3)) ∧ 
  (F = (0, 1)) → 
  (∃ (P : ℝ × ℝ), minimum_distance P A F = 4) := 
by sorry

end minimum_PA_PF_value_l538_538734


namespace sin_690_eq_neg_half_l538_538410

theorem sin_690_eq_neg_half :
  Real.sin (690 * Real.pi / 180) = -1 / 2 :=
by {
  sorry
}

end sin_690_eq_neg_half_l538_538410


namespace sufficient_but_not_necessary_l538_538477

theorem sufficient_but_not_necessary {a b : ℝ} (h1 : a > 1) (h2 : b > 2) :
  (a + b > 3 ∧ a * b > 2) ∧ ¬((a + b > 3 ∧ a * b > 2) → (a > 1 ∧ b > 2)) :=
by
  sorry

end sufficient_but_not_necessary_l538_538477


namespace arrangement_count_l538_538542

def natSet := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def evens := {2, 4, 6, 8, 10}

def odds := {1, 3, 5, 7, 9}

def is_odd_sum (col : Finset ℕ) : Prop :=
  col.sum id % 2 = 1

def valid_arrangement (table : Fin 2 → Fin 5 → ℕ) : Prop :=
  (∀ j, is_odd_sum (finset.univ.image (λ i, table i j))) ∧
  ∀ i j, table i j ∈ natSet

theorem arrangement_count :
  ∃ (n : ℕ), n = 2^5 * (5!)^2 ∧
  ∃ (tables : Finset (Fin 2 → Fin 5 → ℕ)), 
  tables.card = n ∧
  ∀ table ∈ tables, valid_arrangement table :=
sorry

end arrangement_count_l538_538542


namespace downstream_speed_l538_538729

noncomputable def upstream_speed : ℝ := 5
noncomputable def still_water_speed : ℝ := 15

theorem downstream_speed:
  ∃ (Vd : ℝ), Vd = 25 ∧ (still_water_speed = (upstream_speed + Vd) / 2) := 
sorry

end downstream_speed_l538_538729


namespace pyramid_volume_l538_538538

-- Define the volume of the pyramid given conditions
theorem pyramid_volume
  (area_SAB : ℝ = 9)
  (area_SBC : ℝ = 9)
  (area_SCD : ℝ = 27)
  (area_SDA : ℝ = 27)
  (equal_dihedral_angles : Prop) -- placeholder for the condition of equal dihedral angles
  (inscribed_quadrilateral : Prop) -- placeholder for the condition that ABCD is inscribed in a circle
  (area_ABCD : ℝ = 36) :
  -- Prove the volume of the pyramid is 54
  ∃ (V : ℝ), V = 54 :=
sorry

end pyramid_volume_l538_538538


namespace maximum_fleas_l538_538618

-- Define the board dimensions
def board_size : ℕ := 10

-- Condition: Fleas can jump to strictly one of the four adjacent squares and maintain or reverse direction at the board edge
def flea_movement_rule := sorry -- Placeholder for the functional definition

-- Condition: Fleas starting on cells of either same color (white or black in a checkerboard pattern)
def checkerboard_pattern :=
by: 
  let black_cells := [insert actual checkerboard pattern config]
  let white_cells := [insert actual checkerboard pattern config]
  -- Define fleas jump only within their respective color zones
  sorry

-- Prove the maximum number of fleas jumping on the board without landing on the same square
theorem maximum_fleas : 
  let possible_fleas := 2 * board_size + 2 * board_size in
  possible_fleas = 40 := 
begin 
  sorry 
end

end maximum_fleas_l538_538618


namespace part1_intersection_part1_union_part2_complement_l538_538475

-- Definitions for the sets A, B, and U
def A : Set ℝ := {x | -4 ≤ x ∧ x ≤ -2}
def B : Set ℝ := {x | x ≥ -3}
def U : Set ℝ := {x | x ≤ -1}

-- Part (1): Prove the identities for intersection and union
theorem part1_intersection : A ∩ B = {x | -3 ≤ x ∧ x ≤ -2} :=
  sorry

theorem part1_union : A ∪ B = {x | x ≥ -4} :=
  sorry

-- Part (2): Prove the complement in U
theorem part2_complement : U \ (A ∩ B) = {x | x < -3 ∨ (-2 < x ∧ x ≤ -1)} :=
  sorry

end part1_intersection_part1_union_part2_complement_l538_538475


namespace bisect_line_through_point_l538_538294

structure Angle (V A B : Type) :=
(vertex : V)
(side1 : A)
(side2 : B)

theorem bisect_line_through_point (V A B : Type) [add_group V] [module ℝ V]
  (angle : Angle V A B) (M : V) (hM_inside : M ∈ interior angle) :
  ∃ line : Set V, M ∈ line ∧ bisects M line angle :=
sorry

end bisect_line_through_point_l538_538294


namespace part1_part2_range_of_a_l538_538501

noncomputable def f1 (x : ℝ) : ℝ := Real.sin x - Real.log (x + 1)

theorem part1 (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) : f1 x ≥ 0 := sorry

noncomputable def f2 (x a : ℝ) : ℝ := Real.sin x - a * Real.log (x + 1)

theorem part2 {a : ℝ} (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ Real.pi) : f2 x a ≤ 2 * Real.exp x - 2 := sorry

theorem range_of_a : {a : ℝ | ∀ x : ℝ, 0 ≤ x → x ≤ Real.pi → f2 x a ≤ 2 * Real.exp x - 2} = {a : ℝ | a ≥ -1} := sorry

end part1_part2_range_of_a_l538_538501


namespace find_a_plus_b_l538_538857

theorem find_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, x^2 + (a+1)*x + ab = 0 → (x = -1 ∨ x = 4)) → a + b = -3 :=
by
  sorry

end find_a_plus_b_l538_538857


namespace infinitely_many_zeros_f_l538_538434

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.log x + (Real.pi / 4))

theorem infinitely_many_zeros_f :
  ∃∞ x ∈ (Set.Ioo 0 1), f x = 0 := by
  sorry

end infinitely_many_zeros_f_l538_538434


namespace interior_angle_ratio_l538_538051

variables (α β γ : ℝ)

theorem interior_angle_ratio
  (h1 : 2 * α + 3 * β = 4 * γ)
  (h2 : α = 4 * β - γ) :
  ∃ k : ℝ, k ≠ 0 ∧ 
  (α = 2 * k ∧ β = 9 * k ∧ γ = 4 * k) :=
sorry

end interior_angle_ratio_l538_538051


namespace quadrant_of_alpha_l538_538876

theorem quadrant_of_alpha 
  (α : ℝ)
  (h1 : (sin α) / (tan α) > 0)
  (h2 : (tan α) / (cos α) < 0) :
  3*π/2 < α ∧ α < 2*π :=
begin
  sorry
end

end quadrant_of_alpha_l538_538876


namespace find_tricksters_within_30_questions_l538_538316

/-- 
Given 65 inhabitants in a village where:
- Two inhabitants are tricksters and the rest are knights.
- Knights always tell the truth.
- Tricksters can either tell the truth or lie.
- One can show any inhabitant a list of some group of inhabitants (which can consist of one person)
  and ask if all of them are knights.

Prove that it is possible to find both tricksters with no more than 30 questions.
-/
theorem find_tricksters_within_30_questions :
  ∃ (ask_knights : (inhabitants : fin 65) → list (fin 65) → Prop),
  (∀ (i j : fin 65), i ≠ j → ask_knights i [j] = true → (inhabitants[j] = knight) ∨ (inhabitants[j] = trickster))
  ∧ ∀ (inhabitants : fin 65),
  (∃ S : finset (fin 65), S.card = 2 ∧ 
  (∀ i, i ∈ S → inhabitants[i] = trickster) ∧ 
  by asking no more than 30 questions,
  you can identify both tricksters.

end find_tricksters_within_30_questions_l538_538316


namespace two_colored_plane_has_points_distance_one_l538_538674

theorem two_colored_plane_has_points_distance_one (color : Point → Color) (p : Point → Prop) :
  ∃ x y : Point, p x ∧ p y ∧ color x = color y ∧ dist x y = 1 :=
by
  sorry

end two_colored_plane_has_points_distance_one_l538_538674


namespace min_expression_value_l538_538789

noncomputable def expression (x y : ℝ) : ℝ := 2*x^2 + 2*y^2 - 8*x + 6*y + 25

theorem min_expression_value : ∃ (x y : ℝ), expression x y = 12.5 :=
by
  sorry

end min_expression_value_l538_538789


namespace math_problem_l538_538960

theorem math_problem
  (n : ℝ)
  (h : ∃ (a b c : ℕ), n = a + real.sqrt (b + real.sqrt c) ∧
    n = (real.sqrt (52 + real.sqrt 216) + 12))
  (hn_eq : ∀ x, (x = n) ↔ (∃ x, (4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20)) = x ^ 2 - 12 * x - 5)) :
  ∃ (a b c : ℕ), a + b + c = 280 := sorry

end math_problem_l538_538960


namespace quadrilateral_inscribed_circle_problem_l538_538724

variables A B C D P Q : Point
variable (r x : ℝ)
variables (AP PB CQ QD : ℝ)

-- Conditions
def AP_eq : AP = x := by sorry
def PB_eq : PB = x + 7 := by sorry
def CQ_eq : CQ = 3x - 2 := by sorry
def QD_eq : QD = 2x + 3 := by sorry
def tangency_condition : (∃ circle, is_tangent quadrant circle P Q) := by sorry
def tangency_lengths : AP + PB = CQ + QD := by sorry

-- We are to prove that x = 2 and r^2 = 78
theorem quadrilateral_inscribed_circle_problem :
  (AP_eq ∧ PB_eq ∧ CQ_eq ∧ QD_eq ∧ tangency_condition ∧ tangency_lengths) →
  x = 2 ∧ r^2 = 78 := by
  sorry

end quadrilateral_inscribed_circle_problem_l538_538724


namespace tricksters_identification_l538_538339

variable (Inhabitant : Type)
variable [inhab : Fintype Inhabitant]
variable [decEqInhab : DecidableEq Inhabitant]
variable (knight : Inhabitant → Prop)
variable (trickster : Inhabitant → Prop)

variables (n : ℕ) (q : ℕ)
variable (is_truthful : Inhabitant → Prop)

constant inhabitants_count : 65
constant tricksters_count : 2

-- Define the property that a knight always tells the truth.
axiom knight_truth (x : Inhabitant) : knight x → is_truthful x

-- Define the property that a trickster can tell the truth or lie.
axiom trickster_behavior (x : Inhabitant) : trickster x → (is_truthful x ∨ ¬ is_truthful x)

 -- Define the type of the question which can be asked to an inhabitant.
inductive Question (Inhabitant : Type) : Type
| is_knight : Inhabitant → Question

-- Define the type of the answer to the question.
inductive Answer (Inhabitant : Type) : Type
| yes : Answer
| no :  Answer

-- Define a function that simulates asking a question to an inhabitant.
constant ask : Inhabitant → Question Inhabitant → Answer Inhabitant

noncomputable def find_tricksters (inhabitants : fin inhabitants_count → Inhabitant) : (fin tricksters_count → Inhabitant) :=
sorry

theorem tricksters_identification : 
  ∃ (f : (fin inhabitants_count → Inhabitant) → (fin tricksters_count → Inhabitant)), 
    ∀ inhabitants : fin inhabitants_count → Inhabitant, 
      (∀ (q_list : list (Inhabitant × Question Inhabitant)),
        q_list.length ≤ 30 → 
        let a_list := q_list.map (λ pq, ask (pq.fst) (pq.snd)) in 
        true) ∧ 
      (∃ t1 t2, trickster (f inhabitants) t1 ∧ trickster (f inhabitants) t2) :=
sorry

end tricksters_identification_l538_538339


namespace total_gold_coins_l538_538989

theorem total_gold_coins (n c : ℕ) 
  (h1 : n = 11 * (c - 3))
  (h2 : n = 7 * c + 5) : 
  n = 75 := 
by 
  sorry

end total_gold_coins_l538_538989


namespace t_shaped_cannot_tile_10x10_board_l538_538715

theorem t_shaped_cannot_tile_10x10_board :
  ¬ ∃ (t_shapes : fin 10 × fin 10 → Prop), 
    (∀ r c, t_shapes (r, c) → r < 10 ∧ c < 10) ∧
    (∃ n : ℕ, n = 25 ∧ 
      (∀ x : fin 10 × fin 10, t_shapes x → a_piece x)) :=
sorry

end t_shaped_cannot_tile_10x10_board_l538_538715


namespace Cube_diagonal_area_l538_538728

-- Define the cube and its properties
structure Cube :=
  (side_length : ℝ)

-- Define the vertices and midpoints (not the specific coordinates but the properties)
def Cube.vertices (c : Cube) : set (ℝ × ℝ × ℝ) :=
  { (0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2), (2, 2, 0), (2, 0, 2), (0, 2, 2), (2, 2, 2) }

def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

-- Define the specific points A, B, C, D
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def C : ℝ × ℝ × ℝ := (2, 2, 2)
def B : ℝ × ℝ × ℝ := midpoint (0, 2, 0) (0, 0, 2)
def D : ℝ × ℝ × ℝ := midpoint (2, 2, 0) (2, 0, 2)

-- Area of quadrilateral ABCD calculation
noncomputable def area_ABCD : ℝ :=
  let d1 := (Math.sqrt (A.1 - C.1)^2 + (A.2 - C.2)^2 + (A.3 - C.3)^2)
  let d2 := (Math.sqrt (B.1 - D.1)^2 + (B.2 - D.2)^2 + (B.3 - D.3)^2)
  in 1/2 * d1 * d2

-- Final theorem statement
theorem Cube_diagonal_area : Cube → area_ABCD = 2 * Math.sqrt 6 :=
by
  intro c
  -- Proof steps to be inserted here
  sorry

end Cube_diagonal_area_l538_538728


namespace soccer_substitutions_remainder_l538_538738

/-- A soccer team has 23 players in total, 12 players start the game, while the remaining 11 are available as substitutes. 
    The coach is allowed to make up to 5 substitutions during the game. Any one of the players in the game can be replaced 
    by one of the substitutes, but substituting players cannot return to the game after being removed. The order of substitutions 
    and players involved is considered in calculating the number of possible substitutions. Prove the remainder when 
    the number of ways the coach can make substitutions during the game (including the possibility of making no substitutions) 
    is divided by 100. -/
theorem soccer_substitutions_remainder : 
  let b : ℕ → ℕ
    | 0     := 1
    | (n+1) := 12 * (12 - n) * b n
  in ((b 0 + b 1 + b 2 + b 3 + b 4 + b 5) % 100) = 93 :=
  by
  let b : ℕ → ℕ
    | 0     := 1
    | (n+1) := 12 * (12 - n) * b n
  have b_0 : b 0 = 1 := rfl
  have b_1 : b 1 = 132 := by simp [b, b_0]
  have b_2 : b 2 = 15840 := by simp [b, b_1]
  have b_3 : b 3 = 1710720 := by simp [b, b_2]
  have b_4 : b 4 = 164147200 := by simp [b, b_3]
  have b_5 : b 5 = 13788422400 := by simp [b, b_4]
  have m : ℕ := (b 0 + b 1 + b 2 + b 3 + b 4 + b 5) % 100
  have : m = 93 := by
    unfold m b_0 b_1 b_2 b_3 b_4 b_5
    simp
  exact this
  sorry

end soccer_substitutions_remainder_l538_538738


namespace find_tricksters_l538_538329

structure Inhabitant :=
  (isKnight : Prop)

constants (inhabitants : Fin 65 → Inhabitant)
          (tricksters : Fin 2 → Fin 65)

axiom two_tricksters_unique :
  ∃! (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65))

axiom valid_question :
  ∀ (a : Fin 65) (group : Set (Fin 65)), (inhabitants a).isKnight → 
  (∀ i ∈ group, (inhabitants i).isKnight) ↔ 
  (knight a).isKnight

theorem find_tricksters :
  ∃ (q : ℕ), q ≤ 16 ∧
  ∃ (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65)) :=
sorry

end find_tricksters_l538_538329


namespace marble_count_l538_538994

theorem marble_count (pos_left : ℕ) (pos_right : ℕ) (total : ℕ) : 
  pos_left = 5 ∧ pos_right = 3 → total = 7 :=
by 
  intros h,
  cases h with pos_left_eq pos_right_eq,
  sorry

end marble_count_l538_538994


namespace coffee_cost_l538_538709

def ounces_per_donut : ℕ := 2
def ounces_per_pot : ℕ := 12
def cost_per_pot : ℕ := 3
def dozens_of_donuts : ℕ := 3
def donuts_per_dozen : ℕ := 12

theorem coffee_cost :
  let total_donuts := dozens_of_donuts * donuts_per_dozen
  let total_ounces := ounces_per_donut * total_donuts
  let total_pots := total_ounces / ounces_per_pot
  let total_cost := total_pots * cost_per_pot
  total_cost = 18 := by
  sorry

end coffee_cost_l538_538709


namespace tricksters_identification_l538_538343

variable (Inhabitant : Type)
variable [inhab : Fintype Inhabitant]
variable [decEqInhab : DecidableEq Inhabitant]
variable (knight : Inhabitant → Prop)
variable (trickster : Inhabitant → Prop)

variables (n : ℕ) (q : ℕ)
variable (is_truthful : Inhabitant → Prop)

constant inhabitants_count : 65
constant tricksters_count : 2

-- Define the property that a knight always tells the truth.
axiom knight_truth (x : Inhabitant) : knight x → is_truthful x

-- Define the property that a trickster can tell the truth or lie.
axiom trickster_behavior (x : Inhabitant) : trickster x → (is_truthful x ∨ ¬ is_truthful x)

 -- Define the type of the question which can be asked to an inhabitant.
inductive Question (Inhabitant : Type) : Type
| is_knight : Inhabitant → Question

-- Define the type of the answer to the question.
inductive Answer (Inhabitant : Type) : Type
| yes : Answer
| no :  Answer

-- Define a function that simulates asking a question to an inhabitant.
constant ask : Inhabitant → Question Inhabitant → Answer Inhabitant

noncomputable def find_tricksters (inhabitants : fin inhabitants_count → Inhabitant) : (fin tricksters_count → Inhabitant) :=
sorry

theorem tricksters_identification : 
  ∃ (f : (fin inhabitants_count → Inhabitant) → (fin tricksters_count → Inhabitant)), 
    ∀ inhabitants : fin inhabitants_count → Inhabitant, 
      (∀ (q_list : list (Inhabitant × Question Inhabitant)),
        q_list.length ≤ 30 → 
        let a_list := q_list.map (λ pq, ask (pq.fst) (pq.snd)) in 
        true) ∧ 
      (∃ t1 t2, trickster (f inhabitants) t1 ∧ trickster (f inhabitants) t2) :=
sorry

end tricksters_identification_l538_538343


namespace johns_speed_is_45_l538_538134

-- Define the conditions
def johns_total_distance (S : ℝ) : Prop := (2 * S + 3 * 50 = 240)

-- State the proof problem
theorem johns_speed_is_45 : 
  ∃ S : ℝ, johns_total_distance S ∧ S = 45 :=
by {
  use 45,
  unfold johns_total_distance,
  simp,
  sorry
}

end johns_speed_is_45_l538_538134


namespace age_difference_l538_538637

variable (S R : ℝ)

theorem age_difference (h1 : S = 38.5) (h2 : S / R = 11 / 9) : S - R = 7 :=
by
  sorry

end age_difference_l538_538637


namespace evaluate_g_at_neg_four_l538_538879

def g (x : Int) : Int := 5 * x + 2

theorem evaluate_g_at_neg_four : g (-4) = -18 := by
  sorry

end evaluate_g_at_neg_four_l538_538879


namespace stacked_lego_volume_l538_538454

theorem stacked_lego_volume 
  (lego_volume : ℝ)
  (rows columns layers : ℕ)
  (h1 : lego_volume = 1)
  (h2 : rows = 7)
  (h3 : columns = 5)
  (h4 : layers = 3) :
  rows * columns * layers * lego_volume = 105 :=
by
  sorry

end stacked_lego_volume_l538_538454


namespace ratio_of_speeds_l538_538195

theorem ratio_of_speeds
  (speed_of_tractor : ℝ)
  (speed_of_bike : ℝ)
  (speed_of_car : ℝ)
  (h1 : speed_of_tractor = 575 / 25)
  (h2 : speed_of_car = 331.2 / 4)
  (h3 : speed_of_bike = 2 * speed_of_tractor) :
  speed_of_car / speed_of_bike = 1.8 :=
by
  sorry

end ratio_of_speeds_l538_538195


namespace find_range_of_m_l538_538826

open Classical

variable (m : ℝ)

-- Define the conditions
def p : Prop := ∃ x y : ℝ, x ≠ y ∧ x ^ 2 - 2 * m * x + 1 = 0 ∧ y ^ 2 - 2 * m * y + 1 = 0
def q : Prop := ∃ x : ℝ, 0 < x ∧ x ^ 2 - 2 * Real.exp(1) * Real.log x ≤ m

-- Lean statement for the problem
theorem find_range_of_m (hpq : p ∨ q) (hnpq : ¬(p ∧ q)) : m ∈ Icc 0 1 ∨ m < -1 := sorry

end find_range_of_m_l538_538826


namespace find_correct_t_l538_538485

theorem find_correct_t (t : ℝ) :
  (∃! x1 x2 x3 : ℝ, x1^2 - 4*|x1| + 3 = t ∧
                     x2^2 - 4*|x2| + 3 = t ∧
                     x3^2 - 4*|x3| + 3 = t) → t = 3 :=
by
  sorry

end find_correct_t_l538_538485


namespace trajectory_C_min_area_circumcircle_l538_538828

-- Define the given fixed point and line.
def F : Point := ⟨0, 1⟩
def l : Line := {slope := 0, intercept := -1}

-- Define the condition of moving circle M passing through point F and tangent to line l.
def moving_circle (M : Point) : Prop :=
  let d := distance M l in
  distance M F = d

-- Prove the trajectory C of the center of the moving circle M is x^2 = 4y.
theorem trajectory_C (M : Point) (h : moving_circle M) : 
  M.x^2 = 4 * M.y := 
sorry

-- Prove the minimum area of the circumcircle of triangle PAB is 4π.
theorem min_area_circumcircle (F A B P: Point) (hF1 : lies_on_line F (line_through_points F B))
 (hF2 : lies_on_curve A (x^2 = 4 * y)) (hF3 : lies_on_curve B (x^2 = 4 * y))
 (tangent1 : is_tangent P A (x^2 = 4 * y)) (tangent2 : is_tangent P B (x^2 = 4 * y)) 
 (intersection : P = line_intersection tangent1 tangent2): 
  circumcircle_area P A B = 4 * pi := 
sorry

end trajectory_C_min_area_circumcircle_l538_538828


namespace tangent_line_condition_l538_538668

theorem tangent_line_condition (a b k : ℝ) (h1 : (1 : ℝ) + a + b = 2) (h2 : 3 + a = k) (h3 : k = 1) :
    b - a = 5 := 
by 
    sorry

end tangent_line_condition_l538_538668


namespace param_correct_B_param_correct_D_valid_parameterizations_l538_538666

-- Define the line in terms of the slope and intercept
def on_line (x y : ℝ) : Prop := y = 3 * x + 4

-- Define the direction vector criteria
def direction_valid (dx dy : ℝ) : Prop := dy / dx = 3

-- Define the parameterizations
def param_A (t : ℝ) : ℝ × ℝ :=
  (0 + t * 3, 4 + t * 1)

def param_B (t : ℝ) : ℝ × ℝ :=
  (-4/3 + t * -1, 0 + t * -3)

def param_C (t : ℝ) : ℝ × ℝ :=
  (1 + t * 9, 7 + t * 3)

def param_D (t : ℝ) : ℝ × ℝ :=
  (2 + t * (1/3), 10 + t * 1)

def param_E (t : ℝ) : ℝ × ℝ :=
  (-4 + t * (1/10), -8 + t * (1/3))

-- Define the proof for each parameterization
theorem param_correct_B : 
  ∃ t, on_line (-4/3 + t * -1) (0 + t * -3) ∧ 
  direction_valid (-1) (-3) := sorry

theorem param_correct_D :
  ∃ t, on_line (2 + t / 3) (10 + t) ∧ 
  direction_valid (1/3) (1) := sorry

-- Define the final theorem to check the valid parameterizations
theorem valid_parameterizations : 
  (param_B, param_D) = boxed( B, D ) := sorry

end param_correct_B_param_correct_D_valid_parameterizations_l538_538666


namespace cups_per_serving_l538_538722

-- Define the conditions
def total_cups : ℕ := 18
def servings : ℕ := 9

-- State the theorem to prove the answer
theorem cups_per_serving : total_cups / servings = 2 := by
  sorry

end cups_per_serving_l538_538722


namespace distance_A_to_plane_BCD_l538_538923

variables {P Q : Type} [InnerProductSpace ℝ P]

def A : P := ⟨1, 1, 1⟩
def B : P := ⟨2, 3, 4⟩
def n : P := ⟨-1, 2, 1⟩

def distance_point_to_plane (A B n : P) : ℝ :=
  (|inner (B - A) n|) / ∥n∥

theorem distance_A_to_plane_BCD :
  distance_point_to_plane A B n = real.sqrt 6 :=
sorry

end distance_A_to_plane_BCD_l538_538923


namespace employee_selection_l538_538276

theorem employee_selection
  (total_employees : ℕ)
  (under_35 : ℕ)
  (between_35_and_49 : ℕ)
  (over_50 : ℕ)
  (selected_employees : ℕ) :
  total_employees = 500 →
  under_35 = 125 →
  between_35_and_49 = 280 →
  over_50 = 95 →
  selected_employees = 100 →
  (under_35 * selected_employees / total_employees = 25) ∧
  (between_35_and_49 * selected_employees / total_employees = 56) ∧
  (over_50 * selected_employees / total_employees = 19) := by
  intros h1 h2 h3 h4 h5
  sorry

end employee_selection_l538_538276


namespace max_ratio_of_two_digit_numbers_with_mean_55_l538_538584

theorem max_ratio_of_two_digit_numbers_with_mean_55 (x y : ℕ) (h1 : 10 ≤ x) (h2 : x ≤ 99) (h3 : 10 ≤ y) (h4 : y ≤ 99) (h5 : (x + y) / 2 = 55) : x / y ≤ 9 :=
sorry

end max_ratio_of_two_digit_numbers_with_mean_55_l538_538584


namespace total_seeds_eaten_correct_l538_538379

-- Define the number of seeds each player ate
def seeds_first_player : ℕ := 78
def seeds_second_player : ℕ := 53
def seeds_third_player (seeds_second_player : ℕ) : ℕ := seeds_second_player + 30

-- Define the total seeds eaten
def total_seeds_eaten (seeds_first_player seeds_second_player seeds_third_player : ℕ) : ℕ :=
  seeds_first_player + seeds_second_player + seeds_third_player

-- Statement of the theorem
theorem total_seeds_eaten_correct : total_seeds_eaten seeds_first_player seeds_second_player (seeds_third_player seeds_second_player) = 214 :=
by
  sorry

end total_seeds_eaten_correct_l538_538379


namespace find_AC_length_l538_538548

variable (AB AD DC : ℝ)
variable (H_AB : AB = 14)
variable (H_AD : AD = 6)
variable (H_DC : DC = 21)

theorem find_AC_length : AC ≈ 26.2 :=
  sorry

end find_AC_length_l538_538548


namespace sequence_properties_l538_538028

-- Definitions from conditions
def S (n : ℕ) := n^2 - n
def a (n : ℕ) := if n = 1 then 0 else 2 * (n - 1)
def b (n : ℕ) := 2^(n - 1)
def c (n : ℕ) := a n * b n
def T (n : ℕ) := (n - 2) * 2^(n + 1) + 4

-- Theorem statement proving the required identities
theorem sequence_properties {n : ℕ} (hn : n ≠ 0) :
  (a n = (if n = 1 then 0 else 2 * (n - 1))) ∧ 
  (b 2 = a 2) ∧ 
  (b 4 = a 5) ∧ 
  (T n = (n - 2) * 2^(n + 1) + 4) := by
  sorry

end sequence_properties_l538_538028


namespace ratio_q_p_l538_538799

variables {p q : ℚ}

/-- Conditions: 
1. Fifty slips are placed into a hat, each bearing a number 1 through 10, with each number entered on five slips.
2. Five slips are drawn from the hat at random and without replacement.
3. p is the probability that all five slips bear the same number.
4. q is the probability that three of the slips bear a number a and the other two bear a number b ≠ a.
-/

def total_ways_to_choose_slips : ℕ := Nat.binomial 50 5

def ways_all_same_number : ℕ := 10

def ways_three_a_two_b : ℕ := 4500

def p : ℚ := 10 / total_ways_to_choose_slips

def q : ℚ := 4500 / total_ways_to_choose_slips

theorem ratio_q_p : q / p = 450 :=
by
  sorry

end ratio_q_p_l538_538799


namespace model_x_completion_time_l538_538277

theorem model_x_completion_time (T_x : ℝ) : 
  (24 : ℕ) * (1 / T_x + 1 / 36) = 1 → T_x = 72 := 
by 
  sorry

end model_x_completion_time_l538_538277


namespace intersection_of_sets_l538_538068

-- Define the sets M and N
def M : Set ℝ := { x | 2 < x ∧ x < 3 }
def N : Set ℝ := { x | 2 < x ∧ x ≤ 5 / 2 }

-- State the theorem to prove
theorem intersection_of_sets : M ∩ N = { x | 2 < x ∧ x ≤ 5 / 2 } :=
by 
  sorry

end intersection_of_sets_l538_538068


namespace sum_even_and_odd_numbers_up_to_50_l538_538452

def sum_even_numbers (n : ℕ) : ℕ :=
  (2 + 50) * n / 2

def sum_odd_numbers (n : ℕ) : ℕ :=
  (1 + 49) * n / 2

theorem sum_even_and_odd_numbers_up_to_50 : 
  sum_even_numbers 25 + sum_odd_numbers 25 = 1275 :=
by
  sorry

end sum_even_and_odd_numbers_up_to_50_l538_538452


namespace area_of_triangle_APB_l538_538301

/-- Square with side length 8 -/
def square_side : ℝ := 8

/-- Coordinates of the vertices of the square -/
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 0)
def F : ℝ × ℝ := (0, 8)
def D : ℝ × ℝ := (8, 8)

/-- Midpoint P of the diagonal AF -/
def P : ℝ × ℝ := ((A.1 + F.1) / 2, (A.2 + F.2) / 2)

/-- Prove the area of triangle APB is 16 square inches -/
theorem area_of_triangle_APB :
  let PA := real.sqrt ((P.1 - A.1) ^ 2 + (P.2 - A.2) ^ 2),
      PB := real.sqrt ((P.1 - B.1) ^ 2 + (P.2 - B.2) ^ 2),
      base := real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2),
      height := abs (P.2 - A.2)
  in PA = PB ∧ base = 8 ∧ height = 4 → 
  let area := (1 / 2) * base * height
  in area = 16 :=
by
  sorry

end area_of_triangle_APB_l538_538301


namespace average_temperature_in_october_l538_538210

-- Definitions of the conditions
def temperature_function (a A x : ℝ) : ℝ := a + A * Real.cos (Real.pi / 6 * (x - 6))

def is_maximum_in_june (a A : ℝ) : Prop := 28 = a + A
def is_minimum_in_december (a A : ℝ) : Prop := 18 = a - A

-- The statement to prove
theorem average_temperature_in_october : 
  ∀ (a A : ℝ), is_maximum_in_june a A → is_minimum_in_december a A → temperature_function a A 10 = 20.5 := 
by intros; sorry

end average_temperature_in_october_l538_538210


namespace right_triangle_inradius_l538_538433

theorem right_triangle_inradius (a b c : ℕ) (h : a = 6) (h2 : b = 8) (h3 : c = 10) :
  ((a^2 + b^2 = c^2) ∧ (1/2 * ↑a * ↑b = 24) ∧ ((a + b + c) / 2 = 12) ∧ (24 = 12 * 2)) :=
by 
  sorry

end right_triangle_inradius_l538_538433


namespace total_cost_proof_l538_538605

-- Define the prices of items
def price_coffee : ℕ := 4
def price_cake : ℕ := 7
def price_ice_cream : ℕ := 3

-- Define the number of items ordered by Mell and her friends
def mell_coffee : ℕ := 2
def mell_cake : ℕ := 1
def friend_coffee : ℕ := 2
def friend_cake : ℕ := 1
def friend_ice_cream : ℕ := 1
def number_of_friends : ℕ := 2

-- Calculate total cost for Mell
def total_mell : ℕ := (mell_coffee * price_coffee) + (mell_cake * price_cake)

-- Calculate total cost per friend
def total_friend : ℕ := (friend_coffee * price_coffee) + (friend_cake * price_cake) + (friend_ice_cream * price_ice_cream)

-- Calculate total cost for all friends
def total_friends : ℕ := number_of_friends * total_friend

-- Calculate total cost for Mell and her friends
def total_cost : ℕ := total_mell + total_friends

-- The theorem to prove
theorem total_cost_proof : total_cost = 51 := by
  sorry

end total_cost_proof_l538_538605


namespace largest_value_expression_l538_538268

theorem largest_value_expression :
  ∀ (a b c : ℕ),
    a ∈ {1, 2, 4} ∧ b ∈ {1, 2, 4} ∧ c ∈ {1, 2, 4} ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
    ((a / 2) / (b / c) : ℚ) ≤ 8 :=
begin
  sorry
end

end largest_value_expression_l538_538268


namespace min_rooms_proposition_l538_538687

def min_rooms_required (students : List ℕ) : ℕ :=
  let gcd := Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd 120 240) 360) 480) 600
  students.map (λ n => n / gcd).sum

theorem min_rooms_proposition : min_rooms_required [120, 240, 360, 480, 600] = 15 :=
  by sorry

end min_rooms_proposition_l538_538687


namespace prob_sum_24_four_dice_l538_538460

-- The probability of each die landing on six
def prob_die_six : ℚ := 1 / 6

-- The probability of all four dice showing six
theorem prob_sum_24_four_dice : 
  prob_die_six ^ 4 = 1 / 1296 :=
by
  -- Equivalent Lean statement asserting the probability problem
  sorry

end prob_sum_24_four_dice_l538_538460


namespace coordinates_of_point_in_fourth_quadrant_l538_538482

-- Define the conditions as separate hypotheses
def point_in_fourth_quadrant (x y : ℝ) : Prop := (x > 0) ∧ (y < 0)

-- State the main theorem
theorem coordinates_of_point_in_fourth_quadrant
  (x y : ℝ) (h1 : point_in_fourth_quadrant x y) (h2 : |x| = 3) (h3 : |y| = 5) :
  (x = 3) ∧ (y = -5) :=
by
  sorry

end coordinates_of_point_in_fourth_quadrant_l538_538482


namespace num_factors_8_3_5_5_7_2_l538_538873

theorem num_factors_8_3_5_5_7_2 :
  let n := 8^3 * 5^5 * 7^2
  ∃ (f : ℕ), f = 180 ∧ 
              (∑ a in (finset.range 10), ∑ b in (finset.range 6), ∑ c in (finset.range 3), 1) = 180 :=
by
  let n := 8^3 * 5^5 * 7^2
  use 180
  split
  sorry
  sorry

end num_factors_8_3_5_5_7_2_l538_538873


namespace area_of_triangle_ABC_l538_538820

/- Define the points A, B, and C in Cartesian coordinates -/
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (0, 1)
def C : ℝ × ℝ := (1, 0)

/- The function to calculate the area of a triangle given three points -/
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

/- The theorem statement -/
theorem area_of_triangle_ABC : triangle_area A B C = 1 / 2 := 
  sorry

end area_of_triangle_ABC_l538_538820


namespace total_oranges_proof_l538_538613

def jeremyMonday : ℕ := 100
def jeremyTuesdayPlusBrother : ℕ := 3 * jeremyMonday
def jeremyWednesdayPlusBrotherPlusCousin : ℕ := 2 * jeremyTuesdayPlusBrother
def jeremyThursday : ℕ := (70 * jeremyMonday) / 100
def cousinWednesday : ℕ := jeremyTuesdayPlusBrother - (20 * jeremyTuesdayPlusBrother) / 100
def cousinThursday : ℕ := cousinWednesday + (30 * cousinWednesday) / 100

def total_oranges : ℕ :=
  jeremyMonday + jeremyTuesdayPlusBrother + jeremyWednesdayPlusBrotherPlusCousin + (jeremyThursday + (jeremyWednesdayPlusBrotherPlusCousin - cousinWednesday) + cousinThursday)

theorem total_oranges_proof : total_oranges = 1642 :=
by
  sorry

end total_oranges_proof_l538_538613


namespace part1_part2_l538_538508

open Set

noncomputable def U := @univ ℝ
def A := {x : ℝ | -3 ≤ x ∧ x ≤ 5}
def B (m : ℝ) := {x : ℝ | x < 2 * m - 3}

theorem part1 (m : ℝ) (h : m = 5) :
  (A ∩ B m = A) ∧ (compl A ∪ B m = U) :=
by
  rw [h]
  unfold A B U
  sorry

theorem part2 (m : ℝ) :
  (A ⊆ B m) ↔ m > 4 :=
by
  unfold A B
  sorry

end part1_part2_l538_538508


namespace simplify_trig_expr_l538_538180

theorem simplify_trig_expr (α : ℝ) :
  (sin (2 * π - α) * cos (π + α) * cos (π / 2 + α) * cos (11 * π / 2 - α)) /
  (cos (π - α) * sin (3 * π - α) * sin (-π - α) * sin (9 * π / 2 + α)) = 
  -tan α := 
  sorry

end simplify_trig_expr_l538_538180


namespace identify_tricksters_in_30_or_less_questions_l538_538361

-- Define the problem parameters
def inhabitants : Type := Fin 65

def is_knight (inhabitant : inhabitants) : Prop := sorry
def is_trickster (inhabitant : inhabitants) : Prop := sorry

-- Define the properties
axiom knight_truthful : ∀ (x : inhabitants), is_knight x → (forall y : inhabitants, True ↔ (is_knight y = x is_knight y))
axiom trickster_mixed : ∀ (x : inhabitants), is_trickster x → ((∀ y : inhabitants, True) ∨ (∃ y : inhabitants, y ∉ (is_knight y)))

-- Problem statement
theorem identify_tricksters_in_30_or_less_questions
  (inhabitants : Type)
  (n_tricksters : ℕ := 2) -- 2 tricksters
  (total_inhabitants : ℕ := 65) -- 65 total inhabitants
  (questions_limit : ℕ := 30) -- limit of 30 questions
  (knights : inhabitants → Prop)
  (tricksters : inhabitants → Prop) :
    ∃ (solution_exists : ∀ (is_trickster : inhabitants → Prop), ∃ k : inhabitants, (knights k) ∧ (is_trickster k)) 
    (possible_to_find_tricksters : ∀ (is_knight : inhabitants → Prop) (is_trickster : inhabitants → Prop), 
    ∃ (questions_used ≤ questions_limit), ∀ (xs : set inhabitants), questions_used ≤ 30 ∧ 
    (∃ trickster1 trickster2 : inhabitants, (tricksters trickster1 ∧ tricksters trickster2 ∧ trickster1 ≠ trickster2))) :=
sorry

end identify_tricksters_in_30_or_less_questions_l538_538361


namespace bobs_income_after_changes_l538_538767

variable (initial_salary : ℝ) (february_increase_rate : ℝ) (march_reduction_rate : ℝ)

def february_salary (initial_salary : ℝ) (increase_rate : ℝ) : ℝ :=
  initial_salary * (1 + increase_rate)

def march_salary (february_salary : ℝ) (reduction_rate : ℝ) : ℝ :=
  february_salary * (1 - reduction_rate)

theorem bobs_income_after_changes (h1 : initial_salary = 2750)
  (h2 : february_increase_rate = 0.15)
  (h3 : march_reduction_rate = 0.10) :
  march_salary (february_salary initial_salary february_increase_rate) march_reduction_rate = 2846.25 := 
sorry

end bobs_income_after_changes_l538_538767


namespace floor_area_l538_538312

theorem floor_area (length_feet : ℝ) (width_feet : ℝ) (feet_to_meters : ℝ) 
  (h_length : length_feet = 15) (h_width : width_feet = 10) (h_conversion : feet_to_meters = 0.3048) :
  let length_meters := length_feet * feet_to_meters
  let width_meters := width_feet * feet_to_meters
  let area_meters := length_meters * width_meters
  area_meters = 13.93 := 
by
  sorry

end floor_area_l538_538312


namespace tangent_line_min_slope_equation_l538_538004

def curve (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 1

theorem tangent_line_min_slope_equation :
  ∃ (k : ℝ) (b : ℝ), (∀ x y, y = curve x → y = k * x + b)
  ∧ (k = 3)
  ∧ (b = -2)
  ∧ (3 * x - y - 2 = 0) :=
by
  sorry

end tangent_line_min_slope_equation_l538_538004


namespace increasing_interval_maximum_value_l538_538060

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x - π / 3)

theorem increasing_interval (k : ℤ) :
  ∀ x, k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12 → 
  differentiable ℝ (f x) ∧ 
  ∀ x₁ x₂, k * π - π / 12 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ k * π + 5 * π / 12 → f x₁ ≤ f x₂ :=
sorry

theorem maximum_value (k : ℤ) : 
  ∃ x, (∃ k : ℤ, x = k * π + 5 * π / 12) ∧ f x = 2 :=
sorry

end increasing_interval_maximum_value_l538_538060


namespace max_robot_weight_l538_538621

-- Definitions of the given conditions
def standard_robot_weight : ℕ := 100
def battery_weight : ℕ := 20
def min_payload : ℕ := 10
def max_payload : ℕ := 25
def min_robot_weight_extra : ℕ := 5
def min_robot_weight : ℕ := standard_robot_weight + min_robot_weight_extra

-- Definition for total minimum weight of the robot
def min_total_weight : ℕ := min_robot_weight + battery_weight + min_payload

-- Proposition for the maximum weight condition
theorem max_robot_weight :
  2 * min_total_weight = 270 :=
by
  -- Insert proof here
  sorry

end max_robot_weight_l538_538621


namespace tangent_line_curve_l538_538670

theorem tangent_line_curve (a b k : ℝ) (A : ℝ × ℝ) (hA : A = (1, 2))
  (tangent_condition : ∀ x, (k * x + 1 = x ^ 3 + a * x + b) → (k = 3 * x ^ 2 + a)) : b - a = 5 := by
  have hA := congr_arg (λ p : ℝ × ℝ, p.2) hA
  rw hA at tangent_condition
  sorry

end tangent_line_curve_l538_538670


namespace odd_function_monotonic_decreasing_max_min_values_l538_538147

def f (x : ℝ) : ℝ := sorry

axiom functional_eq : ∀ x y : ℝ, f(x + y) = f(x) + f(y)
axiom f_neg : ∀ x : ℝ, x > 0 → f(x) < 0
axiom f_one : f(1) = -1

theorem odd_function : ∀ x : ℝ, f(-x) = -f(x) :=
sorry

theorem monotonic_decreasing : ∀ x₁ x₂ : ℝ, x₁ > x₂ → f(x₁) < f(x₂) :=
sorry

theorem max_min_values : 
  ∃ (max_val min_val : ℝ), max_val = f(-3) ∧ min_val = f(3) ∧ (∀ x, -3 ≤ x ∧ x ≤ 3 → f(x) ≤ max_val ∧ f(x) ≥ min_val) :=
sorry

end odd_function_monotonic_decreasing_max_min_values_l538_538147


namespace bottles_of_cola_sold_l538_538909

theorem bottles_of_cola_sold
    (price_cola : ℝ := 3)
    (price_water : ℝ := 1)
    (price_juice : ℝ := 1.5)
    (water_bottles_sold : ℤ := 25)
    (juice_bottles_sold : ℤ := 12)
    (total_earnings : ℝ := 88) :
    ∃ (C : ℤ), (3 * C + 25 + 18 = 88) ∧ C = 15 := 
by
  use 15
  split
  solve_by_elim
  sorry

end bottles_of_cola_sold_l538_538909


namespace max_dot_product_l538_538529

noncomputable def vector_a : Type := ℝ × ℝ

axiom is_unit_vector (a : vector_a) : Prop :=
  let (x, y) := a in x^2 + y^2 = 1

def vector_b : vector_a := (Real.sqrt 3, -1)

def dot_product (a b : vector_a) : ℝ :=
  let (x1, y1) := a
  let (x2, y2) := b
  x1 * x2 + y1 * y2

theorem max_dot_product (a : vector_a) (h : is_unit_vector a) : dot_product a vector_b ≤ 2 := 
  sorry

end max_dot_product_l538_538529


namespace harmonic_sum_inequality_l538_538095

theorem harmonic_sum_inequality (a_n : ℕ → ℝ) (n : ℕ) 
  (h1 : ∀ n, a_n = ∑ k in finset.range (n+1), 1 / (k + 1)) 
  (h2 : n ≥ 2) : 
  a_n^2 ≥ 2 * ((∑ k in finset.Icc 2 n, a_k / k)) := 
sorry

end harmonic_sum_inequality_l538_538095


namespace complex_conjugate_sum_l538_538037

theorem complex_conjugate_sum (x y : ℝ) (i : ℂ) (h_i : i * i = -1) 
    (h_conjugate : x + y * i = complex.conj (↑(3 : ℝ) + i) / (↑(1 : ℝ) + i)) : 
    x + y = 3 := sorry

end complex_conjugate_sum_l538_538037


namespace lice_checks_l538_538677

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

end lice_checks_l538_538677


namespace coin_flip_expectation_l538_538939

/-- Given 64 fair coins, each coin flipped up to 4 times if it lands on tails,
    the expected number of coins that show heads after these operations is 60. -/
theorem coin_flip_expectation :
  let num_coins := 64 in
  let p_head_first := 1 / 2 in
  let p_head_second := (1 / 2) * (1 / 2) in
  let p_head_third := (1 / 2) * (1 / 2) * (1 / 2) in
  let p_head_fourth := (1 / 2) * (1 / 2) * (1 / 2) * (1 / 2) in
  let p_head := p_head_first + p_head_second + p_head_third + p_head_fourth in
  let expected_heads := num_coins * p_head in
  expected_heads = 60 :=
by 
  let num_coins := 64
  let p_head_first := 1 / 2
  let p_head_second := (1 / 2) * (1 / 2)
  let p_head_third := (1 / 2) * (1 / 2) * (1 / 2)
  let p_head_fourth := (1 / 2) * (1 / 2) * (1 / 2) * (1 / 2)
  let p_head := p_head_first + p_head_second + p_head_third + p_head_fourth
  let expected_heads := num_coins * p_head
  sorry

end coin_flip_expectation_l538_538939


namespace store_sells_2_kg_per_week_l538_538743

def packets_per_week := 20
def grams_per_packet := 100
def grams_per_kg := 1000
def kg_per_week (p : Nat) (gr_per_pkt : Nat) (gr_per_kg : Nat) : Nat :=
  (p * gr_per_pkt) / gr_per_kg

theorem store_sells_2_kg_per_week :
  kg_per_week packets_per_week grams_per_packet grams_per_kg = 2 :=
  sorry

end store_sells_2_kg_per_week_l538_538743


namespace linear_if_abs_k_eq_1_l538_538101

theorem linear_if_abs_k_eq_1 (k : ℤ) : |k| = 1 ↔ (k = 1 ∨ k = -1) := by
  sorry

end linear_if_abs_k_eq_1_l538_538101


namespace pencils_after_exchange_l538_538076

-- Define initial pencil counts
def initial_pencils_Gloria := 2500
def initial_pencils_Lisa := 75800
def initial_pencils_Tim := 1950

-- Define the half of Lisa's pencils
def half_pencils_Lisa := initial_pencils_Lisa / 2

-- Define final pencil counts after the exchange
def final_pencils_Gloria := initial_pencils_Gloria + half_pencils_Lisa
def final_pencils_Lisa := initial_pencils_Lisa - initial_pencils_Lisa
def final_pencils_Tim := initial_pencils_Tim + half_pencils_Lisa

-- Theorem to prove the final state
theorem pencils_after_exchange : 
  final_pencils_Gloria = 40400 ∧ final_pencils_Lisa = 0 ∧ final_pencils_Tim = 39850 := 
begin
  sorry
end

end pencils_after_exchange_l538_538076


namespace f_log2_9_eq_neg_16_div_9_l538_538833

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x - 2) = f x
axiom f_range_0_1 : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 2 ^ x

theorem f_log2_9_eq_neg_16_div_9 : f (Real.log 9 / Real.log 2) = -16 / 9 := 
by 
  sorry

end f_log2_9_eq_neg_16_div_9_l538_538833


namespace focus_of_parabola_l538_538996

theorem focus_of_parabola :
  let h := 0
  let k := 0
  let a := - (1 / 8 : ℝ)
  let p := - (1 / (4 * a))
  (h, k + p) = (0, -2) := 
by
  let h := 0
  let k := 0
  let a := - (1 / 8 : ℝ)
  let p := - (1 / (4 * a))
  have focus_coordinates : (h, k + p) = (0, -2) := by
    -- Proof goes here
    sorry
  exact focus_coordinates

end focus_of_parabola_l538_538996


namespace beth_wins_if_arjun_plays_first_l538_538917

/-- 
In the game where players take turns removing one, two adjacent, or two non-adjacent bricks from 
walls, given certain configurations, the configuration where Beth has a guaranteed winning 
strategy if Arjun plays first is (7, 3, 1).
-/
theorem beth_wins_if_arjun_plays_first :
  let nim_value_1 := 1
  let nim_value_2 := 2
  let nim_value_3 := 3
  let nim_value_7 := 2 -- computed as explained in the solution
  ∀ config : List ℕ,
    config = [7, 1, 1] ∨ config = [7, 2, 1] ∨ config = [7, 2, 2] ∨ config = [7, 3, 1] ∨ config = [7, 3, 2] →
    match config with
    | [7, 3, 1] => true
    | _ => false :=
by
  sorry

end beth_wins_if_arjun_plays_first_l538_538917


namespace find_tricksters_l538_538325

structure Inhabitant :=
  (isKnight : Prop)

constants (inhabitants : Fin 65 → Inhabitant)
          (tricksters : Fin 2 → Fin 65)

axiom two_tricksters_unique :
  ∃! (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65))

axiom valid_question :
  ∀ (a : Fin 65) (group : Set (Fin 65)), (inhabitants a).isKnight → 
  (∀ i ∈ group, (inhabitants i).isKnight) ↔ 
  (knight a).isKnight

theorem find_tricksters :
  ∃ (q : ℕ), q ≤ 16 ∧
  ∃ (t₁ t₂ : Fin 65), t₁ ≠ t₂ ∧ t₁ ∈ (tricksters : Set (Fin 65)) ∧ t₂ ∈ (tricksters : Set (Fin 65)) :=
sorry

end find_tricksters_l538_538325


namespace largest_inscribed_triangle_area_l538_538398

theorem largest_inscribed_triangle_area
  (D : Type) 
  (radius : ℝ) 
  (r_eq : radius = 8) 
  (triangle_area : ℝ)
  (max_area : triangle_area = 64) :
  ∃ (base height : ℝ), (base = 2 * radius) ∧ (height = radius) ∧ (triangle_area = (1 / 2) * base * height) := 
by
  sorry

end largest_inscribed_triangle_area_l538_538398


namespace wheat_bread_served_l538_538975

theorem wheat_bread_served (total_loaves : ℝ) (white_loaf : ℝ) (wheat_loaf : ℝ) 
    (h1 : total_loaves = 0.9) (h2 : white_loaf = 0.4) : 
    wheat_loaf = total_loaves - white_loaf → wheat_loaf = 0.5 := 
by 
    intro h3
    rw [h1, h2] at h3
    exact h3


end wheat_bread_served_l538_538975


namespace triangle_side_difference_l538_538927

theorem triangle_side_difference :
  ∀ x : ℝ, (x > 2 ∧ x < 18) → ∃ a b : ℤ, (∀ y : ℤ, y ∈ set.Icc a b → (y : ℝ) = x) ∧ (b - a = 14) :=
by
  assume x hx,
  sorry

end triangle_side_difference_l538_538927


namespace tangent_line_at_neg_one_l538_538852

noncomputable def f (x : ℝ) : ℝ := 1 / Real.exp x - 1

theorem tangent_line_at_neg_one :
  (∀ x, derivative (λ x, 1 / Real.exp x - 1) x = - real.exp (-x)) →
  let y := f (-1) in
  let m := -Real.exp(1) in
  ∀ (y: ℝ), y = m * (-1) + y - m ->
  let eq_tangent_line := fun (x y : ℝ) => Real.exp x + y + 1 = 0 in 
  eq_tangent_line (-1) y :=
begin
  sorry
end

end tangent_line_at_neg_one_l538_538852


namespace solve_for_x_l538_538641

theorem solve_for_x (x : ℝ) (h : 2^x + 10 = 3 * 2^x - 14) : x = Real.log 12 / Real.log 2 :=
by
  sorry

end solve_for_x_l538_538641


namespace terminating_decimal_count_l538_538810

/-- 
There are 49 integers n between 1 and 449 inclusive such that the decimal representation
of the fraction n/450 terminates.
-/
theorem terminating_decimal_count:
  ∑ n in range(1, 450), (n % 9 = 0) = 49 := 
sorry

end terminating_decimal_count_l538_538810


namespace smallest_sum_primes_l538_538807

noncomputable def smallestPossibleSum : ℕ :=
  findMinSum (S : finset ℕ) (digits : finset ℕ) (hPrimes : ∀ p ∈ S, nat.prime p)
    (hDigitsUsed : (digits.to_finset ∪ɸ S.to_finset) = {1, 2, 3, 4, 5, 6, 7, 8}) :=
  206

theorem smallest_sum_primes : 
  ∃ (S : finset ℕ) (digits : finset ℕ), 
    (∀ p ∈ S, nat.prime p) 
    ∧ ((digits ∪ S) = ({1, 2, 3, 4, 5, 6, 7, 8})) 
    ∧ S.sum = smallestPossibleSum :=
  sorry

end smallest_sum_primes_l538_538807


namespace part1_part2_l538_538965

open Real

/-- Definitions of M, the interval constraints for a and b -/
def M : Set ℝ := {x | -1/2 < x ∧ x < 1/2}

variables (a b : ℝ)
hypothesis (ha : a ∈ M) (hb : b ∈ M)

/-- 1. | a + (1/2)b | < 3/4 -/
theorem part1 {a b : ℝ} (ha : a ∈ M) (hb : b ∈ M) : |a + 1/2 * b| < 3/4 := sorry

/-- 2. |4ab - 1| > 2|b - a| -/
theorem part2 {a b : ℝ} (ha : a ∈ M) (hb : b ∈ M) : |4 * a * b - 1| > 2 * |b - a| := sorry


end part1_part2_l538_538965


namespace smallest_number_is_1013_l538_538259

def smallest_number_divisible (n : ℕ) : Prop :=
  n - 5 % Nat.lcm 12 (Nat.lcm 16 (Nat.lcm 18 (Nat.lcm 21 28))) = 0

theorem smallest_number_is_1013 : smallest_number_divisible 1013 :=
by
  sorry

end smallest_number_is_1013_l538_538259


namespace gathering_handshakes_l538_538987

theorem gathering_handshakes :
  let N := 12       -- twelve people, six couples
  let shakes_per_person := 9   -- each person shakes hands with 9 others
  let total_shakes := (N * shakes_per_person) / 2
  total_shakes = 54 := 
by
  sorry

end gathering_handshakes_l538_538987


namespace right_triangle_even_or_odd_l538_538524

theorem right_triangle_even_or_odd (a b c : ℕ) (ha : Even a ∨ Odd a) (hb : Even b ∨ Odd b) (h : a^2 + b^2 = c^2) : 
  Even c ∨ (Even a ∧ Odd b) ∨ (Odd a ∧ Even b) :=
by
  sorry

end right_triangle_even_or_odd_l538_538524


namespace gdp_scientific_notation_l538_538537

theorem gdp_scientific_notation (trillion : ℕ) (five_year_growth : ℝ) (gdp : ℝ) :
  trillion = 10^12 ∧ 1 ≤ gdp / 10^14 ∧ gdp / 10^14 < 10 ∧ gdp = 121 * 10^12 → gdp = 1.21 * 10^14
:= by
  sorry

end gdp_scientific_notation_l538_538537


namespace dutch_exam_problem_l538_538805

theorem dutch_exam_problem (a b c d : ℝ) : 
  (a * b + c + d = 3) ∧ 
  (b * c + d + a = 5) ∧ 
  (c * d + a + b = 2) ∧ 
  (d * a + b + c = 6) → 
  (a = 2 ∧ b = 0 ∧ c = 0 ∧ d = 3) := 
by
  sorry

end dutch_exam_problem_l538_538805


namespace percentage_caught_sampling_l538_538908

theorem percentage_caught_sampling (S : ℝ) (hS : S = 23.157894736842106) : 
  let C := 0.95 * S in 
  C = 22 :=
by
  sorry

end percentage_caught_sampling_l538_538908


namespace problem_1_problem_2_l538_538497

def f (x a : ℝ) := |x + a| + |x + 3|
def g (x : ℝ) := |x - 1| + 2

theorem problem_1 : ∀ x : ℝ, |g x| < 3 ↔ 0 < x ∧ x < 2 := 
by
  sorry

theorem problem_2 : (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 a = g x2) ↔ a ≥ 5 ∨ a ≤ 1 := 
by
  sorry

end problem_1_problem_2_l538_538497


namespace complement_union_l538_538069

def U := {1, 2, 3}
def A := {1}
def B := {2}

theorem complement_union (U A B : Set ℕ) (hU : U = {1, 2, 3}) (hA : A = {1}) (hB : B = {2}) : 
  U \ (A ∪ B) = {3} :=
by
  rw [hU, hA, hB]
  dsimp
  rw [Set.union_comm]
  sorry

end complement_union_l538_538069


namespace find_tricksters_in_16_questions_l538_538331

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

end find_tricksters_in_16_questions_l538_538331


namespace fixed_point_l538_538662

-- Defining the function
def f (a x : ℝ) : ℝ := 2 * a^(x-1) + 1

-- Statement to prove existence of the fixed point (1, 3)
theorem fixed_point (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) : ∃ x y : ℝ, f a x = y ∧ (x, y) = (1, 3) :=
by
  sorry

end fixed_point_l538_538662


namespace chord_length_k_eq_2_intersection_one_point_iff_l538_538859

-- Define the given parabola
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the line passing through (0, -1) with slope k
def line (k x y : ℝ) : Prop := y = k * x - 1

-- Problem (1): 
-- Prove that if the slope k = 2, the length of the chord cut by the line from the parabola is 5√3
theorem chord_length_k_eq_2 : 
  let k := 2 in 
  ∀ (x1 x2 y1 y2 : ℝ), 
  parabola x1 y1 → parabola x2 y2 → 
  line k x1 y1 → line k x2 y2 → 
  (x1 ≠ x2) → 
  real.sqrt (1 + k^2) * real.sqrt ((x1 + x2)^2 - 4 * x1 * x2) = 5 * real.sqrt 3 := 
  sorry

-- Problem (2):
-- Prove that the line has only one intersection point with the parabola iff k = 0 or k = -3
theorem intersection_one_point_iff :
  ∀ (k : ℝ), 
  (∃ (x : ℝ), parabola x (k * x - 1)) ∧ ¬(∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 
  parabola x1 (k * x1 - 1) ∧ parabola x2 (k * x2 - 1)) ↔ k = 0 ∨ k = -3 :=
  sorry

end chord_length_k_eq_2_intersection_one_point_iff_l538_538859


namespace ms_warren_walking_speed_correct_l538_538611

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

end ms_warren_walking_speed_correct_l538_538611


namespace number_of_subsets_without_2_l538_538516

theorem number_of_subsets_without_2 :
  let M := { S : Set ℕ | S ⊆ {1, 2, 3} ∧ 2 ∉ S }
  in M.toFinset.card = 4 :=
by
  sorry

end number_of_subsets_without_2_l538_538516


namespace range_of_m_and_sum_of_roots_l538_538458

def operation (a b : ℝ) : ℝ :=
if a ≤ b then a^2 - a*b else b^2 - a*b

def f (x : ℝ) : ℝ := operation (2*x - 1) (x - 1)

theorem range_of_m_and_sum_of_roots :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ f x1 = m ∧ f x2 = m ∧ f x3 = m) →
  ∃ (m_range xsum_range : Set ℝ),
    m_range = {m : ℝ | 0 < m ∧ m < 1/4} ∧
    xsum_range = {s : ℝ | (5 - Real.sqrt 3) / 4 < s ∧ s < 1} :=
sorry

end range_of_m_and_sum_of_roots_l538_538458


namespace biking_time_to_patrick_l538_538969

-- Definitions based on the conditions provided
def distance_nicole_nathan : ℝ := 2  -- distance in miles
def time_nicole_nathan : ℝ := 8      -- time in minutes
def distance_nicole_patrick : ℝ := 5 -- distance in miles

-- The rate Nicole rides her bike
def nicole_biking_rate := distance_nicole_nathan / time_nicole_nathan

-- The time it would take for Nicole to ride to Patrick's house
def time_nicole_patrick : ℝ := distance_nicole_patrick / nicole_biking_rate

-- Prove that the calculated time is 20 minutes
theorem biking_time_to_patrick : time_nicole_patrick = 20 := by
  sorry

end biking_time_to_patrick_l538_538969


namespace tan_585_eq_one_l538_538778

theorem tan_585_eq_one : Real.tan (585 * Real.pi / 180) = 1 :=
by
  sorry

end tan_585_eq_one_l538_538778


namespace ants_cannot_reach_final_positions_l538_538693

def Point := (ℝ × ℝ)

def area_of_triangle (A B C : Point) : ℝ :=
  (1 / 2) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

def initial_positions : list Point :=
  [(0, 0), (0, 1), (1, 0)]

def final_positions : list Point :=
  [(-1, 0), (0, 1), (1, 0)]

theorem ants_cannot_reach_final_positions :
  let A := (0, 0)
      B := (0, 1)
      C := (1, 0)
      A' := (-1, 0)
      B' := (0, 1)
      C' := (1, 0)
  in
  area_of_triangle A B C = (1 / 2) ∧
  (∀ n : ℕ, ∀ (An Bn Cn : Point), 
    area_of_triangle An Bn Cn = (1 / 2) →
    (∃ (i : ℕ), (An, Bn, Cn) = (initial_positions.nth i, initial_positions.nth i, initial_positions.nth i))) →
  ¬ (area_of_triangle A' B' C' = (1 / 2)) ∧
  (∀ (An Bn Cn : Point),
    ∀ (m : ℕ),
    area_of_triangle An Bn Cn = (1 / 2) →
    (∃ (i : ℕ), (initial_positions.nth i, initial_positions.nth i, initial_positions.nth i) ≠ (A', B', C')) :=
by
  sorry

end ants_cannot_reach_final_positions_l538_538693


namespace log_value_l538_538505

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^a

theorem log_value : (∃ a : ℝ, f (1 / 2) a = (sqrt 2) / 2) → Real.logb 4 (f 2 (1 / 2)) = 1 / 4 :=
by
  intro h
  cases h with a ha
  sorry

end log_value_l538_538505


namespace percent_less_than_m_plus_d_l538_538272

-- Define the given conditions
variables (m d : ℝ) (distribution : ℝ → ℝ)

-- Assume the distribution is symmetric about the mean m
axiom symmetric_distribution :
  ∀ x, distribution (m + x) = distribution (m - x)

-- 84 percent of the distribution lies within one standard deviation d of the mean
axiom within_one_sd :
  ∫ x in -d..d, distribution (m + x) = 0.84

-- The goal is to prove that 42 percent of the distribution is less than m + d
theorem percent_less_than_m_plus_d : 
  ( ∫ x in -d..0, distribution (m + x) ) = 0.42 :=
by 
  sorry

end percent_less_than_m_plus_d_l538_538272


namespace possible_n_values_l538_538794

theorem possible_n_values (x y n : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : n > 0)
  (top_box_eq : x * y * n^2 = 720) :
  ∃ k : ℕ,  k = 6 :=
by 
  sorry

end possible_n_values_l538_538794


namespace angles_with_axes_of_vector_l538_538444

noncomputable def angle_with_x_axis (a : ℝ × ℝ × ℝ) : ℝ :=
real.arccos (a.1 / real.sqrt (a.1^2 + a.2^2 + a.3^2))

noncomputable def angle_with_y_axis (a : ℝ × ℝ × ℝ) : ℝ :=
real.arccos (a.2 / real.sqrt (a.1^2 + a.2^2 + a.3^2))

noncomputable def angle_with_z_axis (a : ℝ × ℝ × ℝ) : ℝ :=
real.arccos (a.3 / real.sqrt (a.1^2 + a.2^2 + a.3^2))

theorem angles_with_axes_of_vector :
  let a := (-2, 2, 1)
  in angle_with_x_axis a ≈ real.pi * 131 / 180 ∧
     angle_with_y_axis a ≈ real.pi * 48 / 180 ∧
     angle_with_z_axis a ≈ real.pi * 70 / 180 :=
by {
  let a := (-2, 2, 1),
  sorry
}

end angles_with_axes_of_vector_l538_538444


namespace toothpicks_needed_for_8_step_staircase_l538_538967

theorem toothpicks_needed_for_8_step_staircase:
  ∀ n toothpicks : ℕ, n = 4 → toothpicks = 30 → 
  (∃ additional_toothpicks : ℕ, additional_toothpicks = 88) :=
by
  sorry

end toothpicks_needed_for_8_step_staircase_l538_538967


namespace weight_of_box_of_goodies_l538_538942

def initial_weight (jelly_beans brownies gummy_worms : ℝ) : ℝ :=
  jelly_beans + brownies + gummy_worms

def after_chocolate_bars (initial : ℝ) : ℝ :=
  initial + 0.40 * initial

def after_pretzels_popcorn (weight : ℝ) : ℝ :=
  weight + 0.6 - 0.35 + 0.85

def after_cookies (current_weight : ℝ) : ℝ :=
  current_weight + 0.60 * current_weight

def after_brownies_removal (current_weight : ℝ) : ℝ :=
  current_weight - 0.45

def final_weight (initial : ℝ) : ℝ :=
  5 * initial

theorem weight_of_box_of_goodies
  (jelly_beans brownies gummy_worms : ℝ)
  (initial := initial_weight jelly_beans brownies gummy_worms)
  (after_chocolate := after_chocolate_bars initial)
  (after_pretzels := after_pretzels_popcorn after_chocolate)
  (after_cookies_added := after_cookies after_pretzels)
  (after_brownies_removed := after_brownies_removal after_cookies_added)
  (final := final_weight initial) :
  jelly_beans = 1.25 → brownies = 0.75 → gummy_worms = 1.5 →
  after_brownies_removed = final :=
by
  intros hjb hbr hgw
  rw [hjb, hbr, hgw, initial_weight, after_chocolate_bars, after_pretzels_popcorn, after_cookies, after_brownies_removal, final_weight]
  norm_num
  sorry

end weight_of_box_of_goodies_l538_538942


namespace triangle_area_is_27_l538_538534

def point := (ℝ, ℝ)

def vertices : list point := [(3, -3), (9, 6), (3, 6)]

def triangle_area (vertices : list point) : ℝ :=
  match vertices with
  | [(x1, y1), (x2, y2), (x3, y3)] =>
    0.5 * abs ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)))
  | _ => 0
  end

theorem triangle_area_is_27 : triangle_area [(3, -3), (9, 6), (3, 6)] = 27 := 
  by
    sorry

end triangle_area_is_27_l538_538534


namespace matrix_vector_computation_l538_538948

-- Define the problem context
variable (N : Matrix (Fin 2) (Fin 2) ℝ) (x y : Vector (Fin 2) ℝ)

-- Given conditions
axiom h1 : N.mulVec x = ![5, -1]
axiom h2 : N.mulVec y = ![1, 4]

-- The theorem statement
theorem matrix_vector_computation : N.mulVec (2 • x - y) = ![9, -6] :=
by
  sorry

end matrix_vector_computation_l538_538948


namespace number_of_solutions_l538_538842

theorem number_of_solutions (x : ℤ) (h1 : 0 < x) (h2 : x < 150) (h3 : (x + 17) % 46 = 75 % 46) : 
  ∃ n : ℕ, n = 3 :=
sorry

end number_of_solutions_l538_538842


namespace arithmetic_mean_of_sixty_integers_beginning_at_3_l538_538649

theorem arithmetic_mean_of_sixty_integers_beginning_at_3 : 
  let seq (n : ℕ) := 3 + (n - 1)
  ∑ i in (finset.range 60).map (finset.ite 0), seq i / 60 = 32.5 := by
  sorry

end arithmetic_mean_of_sixty_integers_beginning_at_3_l538_538649


namespace cone_lateral_area_l538_538106

theorem cone_lateral_area (r l : ℝ) (h_r : r = 3) (h_l : l = 5) : 
  π * r * l = 15 * π := by
  sorry

end cone_lateral_area_l538_538106


namespace incorrect_statement_among_props_l538_538126

theorem incorrect_statement_among_props 
    (A: Prop := True)  -- Axioms in mathematics are accepted truths that do not require proof.
    (B: Prop := True)  -- A mathematical proof can proceed in different valid sequences depending on the approach and insights.
    (C: Prop := True)  -- All concepts utilized in a proof must be clearly defined before their use in arguments.
    (D: Prop := False) -- Logical deductions based on false premises can lead to valid conclusions.
    (E: Prop := True): -- Proof by contradiction only needs one assumption to be negated and shown to lead to a contradiction to be valid.
  ¬D := 
by sorry

end incorrect_statement_among_props_l538_538126


namespace math_problem_l538_538827

variable {n : ℕ} {x : Fin n → ℝ} {m : ℝ} {a s : ℝ}

theorem math_problem 
  (hx0 : ∀ i, x i > 0)
  (hm : 0 < m)
  (ha : 0 ≤ a)
  (hs : (∑ i, x i) = s ∧ s ≤ n) :
  ∏ i, (x i ^ m + 1 / x i ^ m + a) ≥ ((s / n) ^ m + (n / s) ^ m + a) ^ n :=
by
  sorry

end math_problem_l538_538827


namespace partition_count_l538_538212

def M := {x : ℕ | 1 ≤ x ∧ x ≤ 12}

def valid_partition (s : finset (finset ℕ)) : Prop :=
  s.card = 4 ∧
  ∀ t ∈ s, t.card = 3 ∧
    ∃ a b c, t = {a, b, c} ∧ a = b + c ∧ b < a ∧ c < a

theorem partition_count : finset.card 
  (finset.filter valid_partition 
  (finset.powerset (finset.fin_range 13))) = 8 := 
sorry

end partition_count_l538_538212


namespace initial_investment_l538_538011

open Real

-- Definitions and conditions
def annual_interest_rate : ℝ := 0.12
def interest_period : ℝ := 5
def future_value : ℝ := 705.03

-- Statement of the proof problem
theorem initial_investment :
    ∃ (x : ℝ), x * (1 + annual_interest_rate)^interest_period = future_value ∧ x = 400 :=
by
  let x := 400
  existsi x
  constructor
  { -- Part where we prove x * (1 + r)^t = future_value
    have : (1.12:ℝ)^5 = 1.7623 := sorry, -- Approximately 1.7623, needs exact proof or using more precision
    have eq1 : x * (1 + annual_interest_rate)^interest_period = 705.03,
      calc 400 * (1.12)^5 = 400 * 1.7623 : by rw this
            ...          = 705.03       : sorry,
    exact eq1 },
  { -- Part where we prove x = 400
    refl }

end initial_investment_l538_538011


namespace find_tricksters_in_16_questions_l538_538333

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

end find_tricksters_in_16_questions_l538_538333


namespace sum_A_B_C_l538_538916

open BigOperators

-- Given conditions
def side_length_grid : ℝ := 8
def length_small_square : ℝ := 1
def num_circles : ℕ := 5
def diameter_circle : ℝ := 2
def side_length_square : ℝ := 2

-- Definitions derived from conditions
def area_grid : ℝ := side_length_grid * side_length_grid
def radius_circle : ℝ := diameter_circle / 2
def area_one_circle : ℝ := Real.pi * (radius_circle ^ 2)
def area_circles : ℝ := num_circles * area_one_circle
def area_square : ℝ := side_length_square * side_length_square

-- The visible shaded area in the grid
def visible_shaded_area : ℝ := area_grid - area_circles - area_square

-- The values for A, B, and C
def A : ℝ := 64
def B : ℝ := 5
def C : ℝ := 4

-- The question to prove
theorem sum_A_B_C : A + B + C = 73 := by
  -- Proof omitted
  sorry

end sum_A_B_C_l538_538916


namespace find_tricksters_l538_538346

theorem find_tricksters (inhabitants : Fin 65 → Prop) (is_knight : Fin 65 → Prop)
    (total_inhabitants : ∀ n, inhabitants n)
    (knights : ∀ n, is_knight n → inhabitants n)
    (tricksters_count : (∑ n, if ¬ is_knight n then 1 else 0) = 2)
    (knights_count : (∑ n, if is_knight n then 1 else 0) = 63)
    (knight_truth : ∀ n, is_knight n → ∀ l : list (Fin 65), (∀ m ∈ l, is_knight m) ↔ true)
    (ask_question : ∀ n, inhabitants n → ∀ l : list (Fin 65), bool) :
  ∃ (find_tricksters_function : (Fin 65 → Prop) → (Fin 65 → bool) → (list (Fin 65))) ,
    (length (find_tricksters_function inhabitants ask_question) ≤ 2) →
    (length (find_tricksters_function inhabitants ask_question) = 2) ∧
    ∀ t ∈ (find_tricksters_function inhabitants ask_question), ¬ is_knight t :=
by sorry

end find_tricksters_l538_538346


namespace find_area_ΔABE_l538_538592

-- Definitions of the problem
variables {A B C D E F G H : Type}
variables [convex_quadrilateral ABCD] (pointE_on_CD : lies_on_line_segment E C D)
variables [circumcircle_of_ABE_tangent_to_CD : tangent (circumcircle (triangle A B E)) (line_segment C D)]
variables {line_segment_AC} {line_segment_BE} {line_segment_BD} {line_segment_AE} {line_segment_AC_meets_BE_at_F : meets line_segment_AC line_segment_BE F}
variables {line_segment_BD_meets_AE_at_G : meets line_segment_BD line_segment_AE G}
variables {line_segment_AC_meets_BD_at_H : meets line_segment_AC line_segment_BD H}
variables [FG_parallel_CD : parallel F G C D]
variables [area_ΔABH : area (triangle A B H) = 2] 
variables [area_ΔBCE : area (triangle B C E) = 3] 
variables [area_ΔADE : area (triangle A D E) = 4]

-- The statement we want to prove
theorem find_area_ΔABE :
  area (triangle A B E) = 1 + sqrt 15 :=
sorry

end find_area_ΔABE_l538_538592


namespace domain_transform_l538_538483

theorem domain_transform (f : ℝ → ℝ) :
  (∀ x : ℝ, 1 ≤ 2*x-3 ∧ 2*x-3 < 3 → f (2*x-3)) →
  (∃ x : ℝ, -2/3 < x ∧ x ≤ 2/3) :=
sorry

end domain_transform_l538_538483


namespace sin_690_deg_l538_538424

noncomputable def sin_690_eq_neg_one_half : Prop :=
  sin (690 * real.pi / 180) = -(1 / 2)

theorem sin_690_deg : sin_690_eq_neg_one_half :=
  by sorry

end sin_690_deg_l538_538424


namespace arithmetic_mean_sequence_l538_538651

-- Define the arithmetic sequence starting from 3 with a common difference of 1
def sequence (n : ℕ) : ℕ := n + 2

-- Prove that the arithmetic mean of the first 60 terms is 32.5
theorem arithmetic_mean_sequence :
  let terms := (list.range 60).map sequence in
  let sum := terms.sum in
  let mean := (sum : ℚ) / 60 in
  mean = 32.5 :=
by {
  let terms : list ℕ := (list.range 60).map sequence,
  let sum : ℕ := terms.sum,
  have h_sum : sum = 1950 := sorry,
  have h_mean : (sum : ℚ) / 60 = 32.5 := sorry,
  exact h_mean
}

end arithmetic_mean_sequence_l538_538651


namespace triangles_coloring_ways_l538_538920

theorem triangles_coloring_ways : 
  ∃ (S T U V : Type) (coloring : S → T → U → V → Prop), 
  (∀ b : S → bool, ∃ c : T → bool, 
   ∃ d : U → bool, ∃ e : V → bool, 
   ∃ shared_side : (bool → bool → bool), 
   list.count (list.append [b c d e] false) = 2 ∧ 
   list.count (list.append [b c d e] true) = 2 ∧ 
   shared_side = (λ x y z w, (w = true ∧ (x = true ∨ y = true ∨ z = true)))) ↔ 
   list.length (list.filter (λ t, shared_side t) [S, T, U, V]) = 3 :=
sorry

end triangles_coloring_ways_l538_538920


namespace f_2009_value_l538_538033

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom even_function (f : ℝ → ℝ) : ∀ x, f x = f (-x)
axiom odd_function (g : ℝ → ℝ) : ∀ x, g x = -g (-x)
axiom f_value : f 1 = 0
axiom g_def : ∀ x, g x = f (x - 1)

theorem f_2009_value : f 2009 = 0 :=
by
  sorry

end f_2009_value_l538_538033


namespace angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees_l538_538261

-- Define the concept of a cube and the diagonals of its faces.
structure Cube :=
  (faces : Fin 6 → (Fin 4 → ℝ × ℝ × ℝ))    -- Representing each face as a set of four vertices in 3D space

def is_square_face (face : Fin 4 → ℝ × ℝ × ℝ) : Prop :=
  -- A function that checks if a given set of four vertices forms a square face.
  sorry

def are_adjacent_faces_perpendicular_diagonals 
  (face1 face2 : Fin 4 → ℝ × ℝ × ℝ) (c : Cube) : Prop :=
  -- A function that checks if the diagonals of two given adjacent square faces of a cube are perpendicular.
  sorry

-- The theorem stating the required proof:
theorem angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees
  (c : Cube)
  (h1 : is_square_face (c.faces 0))
  (h2 : is_square_face (c.faces 1))
  (h_adj: are_adjacent_faces_perpendicular_diagonals (c.faces 0) (c.faces 1) c) :
  ∃ q : ℝ, q = 90 :=
by
  sorry

end angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees_l538_538261


namespace area_of_plot_area_in_terms_of_P_l538_538135

-- Conditions and definitions.
variables (P : ℝ) (l w : ℝ)
noncomputable def perimeter := 2 * (l + w)
axiom h_perimeter : perimeter l w = 120
axiom h_equality : l = 2 * w

-- Proofs statements
theorem area_of_plot : l + w = 60 → l = 2 * w → (4 * w)^2 = 6400 := by
  sorry

theorem area_in_terms_of_P : (4 * (P / 6))^2 = (2 * P / 3)^2 → (2 * P / 3)^2 = 4 * P^2 / 9 := by
  sorry

end area_of_plot_area_in_terms_of_P_l538_538135


namespace problem_solution_l538_538486

variable {f : ℝ → ℝ}

-- Define the conditions given in the problem
def tangent_condition (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ y, (f(x) = y) ∧ (2*x + y - 1 = 0) ∧ (∀ x', f x' = y → x' = x)
 
theorem problem_solution (h1 : tangent_condition f 1) 
                         (h2 : deriv f 1 = -2) : 
   f 1 + deriv f 1 = -3 :=
by 
  -- Will contain the proof
  sorry

end problem_solution_l538_538486


namespace unit_digit_product_is_zero_l538_538243

-- Definitions based on conditions in (a)
def a_1 := 6245
def a_2 := 7083
def a_3 := 9137
def a_4 := 4631
def a_5 := 5278
def a_6 := 3974

-- Helper function to get the unit digit of a number
def unit_digit (n : Nat) : Nat := n % 10

-- Main theorem to prove
theorem unit_digit_product_is_zero :
  unit_digit (a_1 * a_2 * a_3 * a_4 * a_5 * a_6) = 0 := by
  sorry

end unit_digit_product_is_zero_l538_538243


namespace determine_a_l538_538788

open Real

theorem determine_a :
  (∃ a : ℝ, |x^2 + a*x + 4*a| ≤ 3 → x^2 + a*x + 4*a = 3) ↔ (a = 8 + 2*sqrt 13 ∨ a = 8 - 2*sqrt 13) :=
by
  sorry

end determine_a_l538_538788


namespace num_six_year_olds_l538_538218

theorem num_six_year_olds (x : ℕ) 
  (h3 : 13 = 13) 
  (h4 : 20 = 20) 
  (h5 : 15 = 15) 
  (h_sum1 : 13 + 20 = 33) 
  (h_sum2 : 15 + x = 15 + x) 
  (h_avg : 2 * 35 = 70) 
  (h_total : 33 + (15 + x) = 70) : 
  x = 22 :=
by
  sorry

end num_six_year_olds_l538_538218


namespace Amy_finish_time_l538_538705

-- Definitions and assumptions based on conditions
def Patrick_time : ℕ := 60
def Manu_time : ℕ := Patrick_time + 12
def Amy_time : ℕ := Manu_time / 2

-- Theorem statement to be proved
theorem Amy_finish_time : Amy_time = 36 :=
by
  sorry

end Amy_finish_time_l538_538705


namespace part1_part2_l538_538148

noncomputable def f (x : ℝ) : ℝ := sorry -- Function definition placeholder

variables (a : ℝ) (h_pos_a : a > 0) 
          (h_even : ∀ x : ℝ, f x = f (-x)) 
          (h_symm_1 : ∀ x : ℝ, f (2 - x) = f x) 
          (h_multiplicative : ∀ x1 x2 : ℝ, x1 ∈ Icc (0 : ℝ) (1 / 2) → x2 ∈ Icc (0 : ℝ) (1 / 2) → f (x1 + x2) = f x1 * f x2)
          (h_f1 : f 1 = a)

open Real

theorem part1 : f (1 / 2) = sqrt a ∧ f (1 / 4) = a^(1 / 4) := 
by
  sorry -- Proof for part 1

theorem part2 : ∃ T, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x :=
by
  use 2
  split
  · linarith
  · intro x
    apply eq.trans (h_even (-x)) (h_even x)
    sorry -- Proof for periodicity

end part1_part2_l538_538148


namespace max_omega_l538_538854

noncomputable def func := λ (x : ℝ) (ω : ℝ) (φ : ℝ), sin (ω * x + φ)

theorem max_omega (ω > 0) (|φ| ≤ π/2) (func(-π/4, ω, φ) = 0) 
(func(π/4, ω, φ) = 0) (monotonic_on func (π/18) (5*π/36))
: ω ≤ 9 :=
by
  sorry

end max_omega_l538_538854


namespace sin_690_degree_l538_538418

theorem sin_690_degree : sin (690 : ℝ) * (Real.pi / 180) = -(1 / 2) := by
  sorry

end sin_690_degree_l538_538418


namespace median_production_l538_538660

def production_data : List ℕ := [5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10]

def median (l : List ℕ) : ℕ :=
  if l.length % 2 = 1 then
    l.nthLe (l.length / 2) sorry
  else
    let m := l.length / 2
    (l.nthLe (m - 1) sorry + l.nthLe m sorry) / 2

theorem median_production :
  median (production_data) = 8 :=
by
  sorry

end median_production_l538_538660


namespace derivative_at_pi_l538_538855

def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem derivative_at_pi : (deriv f π) = -1 :=
by
  sorry

end derivative_at_pi_l538_538855


namespace cos_neg_x_pi_half_l538_538467

theorem cos_neg_x_pi_half (x : ℝ) (h1 : x ∈ Ioo (π / 2) π) (h2 : tan x = -4 / 3) : cos (-x - π / 2) = -4 / 5 :=
by sorry

end cos_neg_x_pi_half_l538_538467


namespace somu_age_to_father_age_ratio_l538_538988

theorem somu_age_to_father_age_ratio
  (S : ℕ) (F : ℕ)
  (h1 : S = 10)
  (h2 : S - 5 = (1/5) * (F - 5)) :
  S / F = 1 / 3 :=
by
  sorry

end somu_age_to_father_age_ratio_l538_538988
