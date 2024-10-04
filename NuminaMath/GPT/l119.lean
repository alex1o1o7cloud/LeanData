import Mathlib

namespace new_average_with_grace_marks_l119_119184

theorem new_average_with_grace_marks :
  ∀ (n : ℕ) (original_avg grace : ℕ), n = 35 → original_avg = 37 → grace = 3 →
  let original_total := original_avg * n,
      total_grace := grace * n,
      new_total := original_total + total_grace,
      new_avg := new_total / n
  in new_avg = 40 :=
by
  intros n original_avg grace hn_avg h_avg h_grace
  rw [hn_avg, h_avg, h_grace]
  let original_total := (37 * 35 : ℕ)
  let total_grace := (3 * 35 : ℕ)
  let new_total := original_total + total_grace
  let new_avg := new_total / 35
  have : new_total = 1400 := by norm_num [original_total, total_grace]
  have : new_avg = 1400 / 35 := by rw [this] <;> norm_num
  rw this
  norm_num
  sorry

end new_average_with_grace_marks_l119_119184


namespace ice_cost_l119_119624

def people : Nat := 15
def ice_needed_per_person : Nat := 2
def pack_size : Nat := 10
def cost_per_pack : Nat := 3

theorem ice_cost : 
  let total_ice_needed := people * ice_needed_per_person
  let number_of_packs := total_ice_needed / pack_size
  total_ice_needed = 30 ∧ number_of_packs = 3 ∧ number_of_packs * cost_per_pack = 9 :=
by
  let total_ice_needed := people * ice_needed_per_person
  let number_of_packs := total_ice_needed / pack_size
  have h1 : total_ice_needed = 30 := by sorry
  have h2 : number_of_packs = 3 := by sorry
  have h3 : number_of_packs * cost_per_pack = 9 := by sorry
  exact And.intro h1 (And.intro h2 h3)

end ice_cost_l119_119624


namespace max_divided_parts_by_three_planes_l119_119327

theorem max_divided_parts_by_three_planes (P1 P2 P3 : set (ℝ × ℝ × ℝ)) 
(plane1 : P1 ≠ ∅) (plane2 : P2 ≠ ∅) (plane3 : P3 ≠ ∅): 
  ∃ (max_parts : ℕ), max_parts = 8 := 
by
  sorry

end max_divided_parts_by_three_planes_l119_119327


namespace euler_rectangular_form_l119_119231

theorem euler_rectangular_form :
  2 * exp (13 * Real.pi * Complex.I / 6) = Complex.ofReal (Real.sqrt 3) + Complex.I := sorry

end euler_rectangular_form_l119_119231


namespace proof_viggo_age_condition_l119_119538

variable (current_age_younger : ℕ) (years_ago : ℕ)

def was_viggo_age_more_than_twice_plus_ten (younger_age_current : ℕ) (b : ℕ) : Prop :=
  let viggo_age := 2 * b + 10
  viggo_age = 2 * (younger_age_current - years_ago) + 10

theorem proof_viggo_age_condition :
  ∀ (younger_age_current younger_age_then : ℕ) (x:ℕ), 
  younger_age_current = 10 →
  younger_age_then = younger_age_current - x →
  was_viggo_age_more_than_twice_plus_ten younger_age_current younger_age_then :=
by 
  intros younger_age_current younger_age_then x current_age younger_age_eq
  rw younger_age_eq
  dsimp [was_viggo_age_more_than_twice_plus_ten]
  sorry

end proof_viggo_age_condition_l119_119538


namespace even_function_monotone_increasing_l119_119440

def f (x : ℝ) : ℝ := x^2

theorem even_function : ∀ x : ℝ, f x = f (-x) :=
by
  intro x
  show f x = f (-x)
  rewrite [f, f]
  ring

theorem monotone_increasing : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 ≤ f x2 :=
by
  intros x1 x2 h0 hx
  show f x1 ≤ f x2
  rewrite [f, f]
  apply sq_le_sq
  apply le_of_lt h0
  apply le_trans (le_of_lt h0) (le_of_lt hx)

end even_function_monotone_increasing_l119_119440


namespace cyclic_AB1PC1_l119_119796

noncomputable def isosceles_triangle (A B C : Point) :=
  ∃ AB_eq_BC : dist A B = dist B C, True

def cyclic_quadrilateral (A B C D : Point) : Prop :=
  ∃ O, is_circle O ∧ A ∈ O ∧ B ∈ O ∧ C ∈ O ∧ D ∈ O

theorem cyclic_AB1PC1 
  (A B C A1 B1 C1 P : Point) 
  (h_iso : isosceles_triangle A B C) 
  (h_A1 : is_on_line C B A1) 
  (h_B1 : is_on_line A C B1) 
  (h_C1 : is_on_line B A C1)
  (h_angle_eq : ∀ α, angle B C1 A1 = α ∧ angle C A1 B1 = α ∧ angle B A C = α)
  (h_P : is_intersection (join B B1) (join C C1) P) :
  cyclic_quadrilateral A B1 P C1 :=
begin
  sorry
end

end cyclic_AB1PC1_l119_119796


namespace number_of_umbrella_numbers_l119_119593

def is_umbrella_number (a b c : ℕ) : Prop :=
  b > a ∧ b > c

def from_set (a b c : ℕ) : Prop :=
  a ∈ ({1, 2, 3, 4, 5, 6} : set ℕ) ∧
  b ∈ ({1, 2, 3, 4, 5, 6} : set ℕ) ∧
  c ∈ ({1, 2, 3, 4, 5, 6} : set ℕ)

def distinct_digits (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem number_of_umbrella_numbers :
  ∃ n : ℕ,
    n = 40 ∧
    (∀ a b c : ℕ, from_set a b c → distinct_digits a b c → is_umbrella_number a b c → (10 * b + 100 * a + c) ∈ (insert (100 * a + 10 * b + c) ∅)) :=
by
  sorry

end number_of_umbrella_numbers_l119_119593


namespace roots_in_interval_l119_119716

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := (1 / 3) * x^3 - a * x^2 + 1

theorem roots_in_interval (a : ℝ) (h : a > 2) :
  ∃! x ∈ Ioo 0 2, f x a = 0 :=
sorry

end roots_in_interval_l119_119716


namespace andrew_donuts_l119_119604

/--
Andrew originally asked for 3 donuts for each of his 2 friends, Brian and Samuel. 
Then invited 2 more friends and asked for the same amount of donuts for them. 
Andrew’s mother wants to buy one more donut for each of Andrew’s friends. 
Andrew's mother is also going to buy the same amount of donuts for Andrew as everybody else.
Given these conditions, the total number of donuts Andrew’s mother needs to buy is 20.
-/
theorem andrew_donuts : (3 * 2) + (3 * 2) + 4 + 4 = 20 :=
by
  -- Given:
  -- 1. Andrew asked for 3 donuts for each of his two friends, Brian and Samuel.
  -- 2. He later invited 2 more friends and asked for the same amount of donuts for them.
  -- 3. Andrew’s mother wants to buy one more donut for each of Andrew’s friends.
  -- 4. Andrew’s mother is going to buy the same amount of donuts for Andrew as everybody else.
  -- Prove: The total number of donuts Andrew’s mother needs to buy is 20.
  sorry

end andrew_donuts_l119_119604


namespace adjacent_complementary_is_complementary_l119_119890

/-- Two angles are complementary if their sum is 90 degrees. -/
def complementary (α β : ℝ) : Prop :=
  α + β = 90

/-- Two angles are adjacent complementary if they are complementary and adjacent. -/
def adjacent_complementary (α β : ℝ) : Prop :=
  complementary α β ∧ α > 0 ∧ β > 0

/-- Prove that adjacent complementary angles are complementary. -/
theorem adjacent_complementary_is_complementary (α β : ℝ) : adjacent_complementary α β → complementary α β :=
by
  sorry

end adjacent_complementary_is_complementary_l119_119890


namespace exists_arithmetic_sequence_l119_119452

theorem exists_arithmetic_sequence (k : ℕ) (hk : k > 0) :
  ∃ (a b : ℕ → ℕ), 
    (∀ i, a i > 0 ∧ b i > 0 ∧ Nat.coprime (a i) (b i)) ∧
    (∀ i j, i ≠ j → a i ≠ a j ∧ b i ≠ b j) ∧
    (∀ i j, i ≠ j → a i ≠ a j) :=
sorry

end exists_arithmetic_sequence_l119_119452


namespace same_point_bisector_l119_119714

theorem same_point_bisector (a b : ℝ)
  (h : (a, b) = (b, a)) :
  (a = b ∨ a = -b) ∧ ((a, b) lies on the bisector of the angle between the two coordinate axes in the first and third quadrants) :=
by
  sorry

end same_point_bisector_l119_119714


namespace limit_sum_sqrt_inv_l119_119967

theorem limit_sum_sqrt_inv : 
  (∀ n : ℕ, Sum (λ k, 1 / Real.sqrt (n^2 - k^2)) (Finset.range n)) = 
  (fun a, a / n) 
  (λ x, 1 / Real.sqrt (1 - x^2)) →
  filter.tendsto (Sum (λ (k : ℕ), 1 / Real.sqrt (n^2 - k^2)) (Finset.range n)) filter.atTop (𝓝 (Real.pi / 2)) :=
begin
  sorry
end

end limit_sum_sqrt_inv_l119_119967


namespace wealthiest_individuals_income_l119_119469

/-- The formula N = 2 * 10^9 * x^(-4/3) gives the number of individuals whose income exceeds x euros.
Given the exchange rate is 1.2 dollars per euro, prove that the lowest income, in dollars, of the 
wealthiest 200 individuals is at least 213394. -/
theorem wealthiest_individuals_income (x : ℝ) (euro_to_dollar : ℝ) 
  (N : ℝ := 2 * 10^9 * x^(-4/3)) (exchange_rate : euro_to_dollar = 1.2) (H : N = 200) : 
  x * euro_to_dollar ≥ 213394 := 
  sorry

end wealthiest_individuals_income_l119_119469


namespace select_intersection_subsets_exists_l119_119809

theorem select_intersection_subsets_exists :
  ∃ (H : Fin₁₆ → Set (Fin 10000)), 
  ∀ (n : Fin 10000), 
  ∃ (subset_indices : Finset (Fin₁₆)),
    subset_indices.card = 8 ∧ 
    n = (⋂ i ∈ subset_indices, H i) :=
sorry

end select_intersection_subsets_exists_l119_119809


namespace fraction_traditionalists_l119_119197

theorem fraction_traditionalists {P T : ℕ} (h1 : ∀ (i : ℕ), i < 5 → T = P / 15) (h2 : T = P / 15) :
  (5 * T : ℚ) / (P + 5 * T : ℚ) = 1 / 4 :=
by
  sorry

end fraction_traditionalists_l119_119197


namespace infinite_rationals_sqrt_rational_l119_119817

theorem infinite_rationals_sqrt_rational : ∃ᶠ x : ℚ in Filter.atTop, ∃ y : ℚ, y = Real.sqrt (x^2 + x + 1) :=
sorry

end infinite_rationals_sqrt_rational_l119_119817


namespace cucumbers_new_weight_l119_119244

/-- 
Given 100 pounds of cucumbers where 99% of the weight is water, 
and after some water evaporates they are 95% water by weight,
prove that the new weight of the cucumbers is 20 pounds.
-/
theorem cucumbers_new_weight (initial_weight : ℝ) (initial_water_percentage : ℝ) (final_water_percentage : ℝ) :
  initial_weight = 100 →
  initial_water_percentage = 0.99 →
  final_water_percentage = 0.95 →
  let solid_weight := initial_weight * (1 - initial_water_percentage) in
  let new_weight := solid_weight / (1 - final_water_percentage) in
  new_weight = 20 :=
by
  intros h1 h2 h3
  let solid_weight := initial_weight * (1 - initial_water_percentage)
  have h_solid_weight : solid_weight = 1 := by
    calc
      solid_weight = 100 * (1 - 0.99) : by rw [h1, h2]
      ... = 100 * 0.01        : by rfl
      ... = 1                 : by norm_num
  let new_weight := solid_weight / (1 - final_water_percentage)
  have h_new_weight : new_weight = 20 := by
    calc
      new_weight = 1 / (1 - 0.95) : by rw [h_solid_weight, h3]
      ... = 1 / 0.05              : by rfl
      ... = 20                    : by norm_num
  exact h_new_weight

end cucumbers_new_weight_l119_119244


namespace cos_over_sin_eq_3_l119_119295

noncomputable def α : ℝ := sorry
def π : ℝ := Real.pi
def tan (θ : ℝ) : ℝ := Real.tan θ
def cos (θ : ℝ) : ℝ := Real.cos θ
def sin (θ : ℝ) : ℝ := Real.sin θ

axiom tan_alpha : tan α = 2 * tan (π / 5)

theorem cos_over_sin_eq_3 : (cos (α - 3 * π / 10)) / (sin (α - π / 5)) = 3 :=
by
  rw [tan_alpha]
  sorry

end cos_over_sin_eq_3_l119_119295


namespace smaller_angle_clock_3_to_7_l119_119908

theorem smaller_angle_clock_3_to_7 :
  ∀ (n : ℕ) (angle : ℕ),
  n = 12 →
  angle = 30 →
  4 * angle = 120 :=
by
  intros n angle h1 h2
  rw [h1, h2]
  exact rfl

end smaller_angle_clock_3_to_7_l119_119908


namespace problem_statement_l119_119334

theorem problem_statement (x : ℝ) (h : (x / 5) / 3 = 5 / (x / 3)) : x = 15 ∨ x = -15 := 
by
  sorry

end problem_statement_l119_119334


namespace fraction_torn_off_l119_119922

theorem fraction_torn_off (P: ℝ) (A_remaining: ℝ) (fraction: ℝ):
  P = 32 → 
  A_remaining = 48 → 
  fraction = 1 / 4 :=
by 
  sorry

end fraction_torn_off_l119_119922


namespace steel_plate_conversion_purchasing_plans_count_max_profit_l119_119879

-- Part 1
theorem steel_plate_conversion (x y : ℕ) (h1 : 2 * x + y = 14) (h2 : x + 3 * y = 12) :
  x = 6 ∧ y = 2 := sorry

-- Part 2
theorem purchasing_plans_count :
  ∃ n : ℕ, n = 7 ∧ ∀ a : ℕ, 30 ≤ a ∧ a ≤ 36 ↔ a ∈ finset.range 37 ∧ 30 ≤ a :=
begin
  use 7,
  split,
  { refl },
  { intros a,
    exact ⟨λ h, ⟨finset.mem_range.2 (lt_of_lt_of_le h.2 (nat.le_refl 36)), h.1⟩,
          λ h2, ⟨h2.2, finset.mem_range.1 h2.1⟩⟩ }
end

-- Part 3
theorem max_profit (pC pD : ℕ) (a : ℕ) (h1 : pC = 100) (h2 : pD = 120) (h3 : 30 ≤ a ∧ a ≤ 36) :
  ∃ maxP, maxP = 18800 ∧ ∀ a' : ℕ, 30 ≤ a' ∧ a' ≤ 36 → 
           (let rC := 2 * a' + 50 - a',
                rD := a' + 3 * (50 - a')
            in rC * pC + rD * pD ≤ 18800) :=
begin
  use 18800,
  split,
  { refl },
  { intros a' ha',
    sorry } -- Proof detailing the calculation
end

end steel_plate_conversion_purchasing_plans_count_max_profit_l119_119879


namespace matrix_sixth_power_l119_119405

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 3;
    2, 2]

theorem matrix_sixth_power :
  let I := 1 : Matrix (Fin 2) (Fin 2) ℝ in
  B ^ 6 = 673 • B + 820 • I :=
by
  let I := 1 : Matrix (Fin 2) (Fin 2) ℝ
  sorry

end matrix_sixth_power_l119_119405


namespace find_train_length_l119_119594

noncomputable def speed_in_m_per_s (speed_kmh : ℕ) : ℝ :=
  speed_kmh * 1000 / 3600

theorem find_train_length :
  ∀ (speed_kmh : ℕ) (bridge_crossing_time_s : ℕ) 
  (total_length_m : ℕ), 
  speed_kmh = 45 → 
  bridge_crossing_time_s = 30 → 
  total_length_m = 245 → 
  let speed_m_per_s := speed_in_m_per_s speed_kmh in
  let distance_covered := speed_m_per_s * bridge_crossing_time_s in
  distance_covered = 375 →
  ∃ (train_length : ℝ), train_length = distance_covered - total_length_m ∧ train_length = 130 :=
begin
  intros,
  sorry
end

end find_train_length_l119_119594


namespace geometric_seq_formula_sum_of_b_n_l119_119282

section geometric_sequence

-- Conditions: a positive geometric sequence {a_n} with a_1 = 1/2
def a (n : ℕ) : ℚ := 1 / 2 ^ n

-- Condition: the geometric mean of a_2 and a_4 is 1/8
lemma geometric_mean_condition : (a 2 * a 4).sqrt ⁻¹ = 1 / 8 :=
by sorry

-- Question I: Prove the general formula for the sequence {a_n} is a_n = 1 / 2^n
theorem geometric_seq_formula (n : ℕ) : a n = 1 / 2 ^ n :=
by sorry

-- Define {b_n} = n * a_n
def b (n : ℕ) : ℚ := n / 2 ^ n

-- Question II: Prove the sum of the first n terms of the sequence {b_n} is S_n = 2 - (n + 2) / 2^n
def S (n : ℕ) : ℚ := ∑ i in range (n+1), b i

theorem sum_of_b_n (n : ℕ) : S n = 2 - (n + 2) / 2 ^ n :=
by sorry

end geometric_sequence

end geometric_seq_formula_sum_of_b_n_l119_119282


namespace probability_of_9_heads_in_12_l119_119124

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l119_119124


namespace people_not_in_pool_l119_119394

-- Define families and their members
def karen_donald_family : ℕ := 2 + 6
def tom_eva_family : ℕ := 2 + 4
def luna_aidan_family : ℕ := 2 + 5
def isabel_jake_family : ℕ := 2 + 3

-- Total number of people
def total_people : ℕ := karen_donald_family + tom_eva_family + luna_aidan_family + isabel_jake_family

-- Number of legs in the pool and people in the pool
def legs_in_pool : ℕ := 34
def people_in_pool : ℕ := legs_in_pool / 2

-- People not in the pool: people who went to store and went to bed
def store_people : ℕ := 2
def bed_people : ℕ := 3
def not_available_people : ℕ := store_people + bed_people

-- Prove (given conditions) number of people not in the pool
theorem people_not_in_pool : total_people - people_in_pool - not_available_people = 4 :=
by
  -- ...proof steps or "sorry"
  sorry

end people_not_in_pool_l119_119394


namespace probability_heads_9_of_12_is_correct_l119_119031

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l119_119031


namespace probability_exactly_9_heads_l119_119117

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l119_119117


namespace sequence_eq_3n_l119_119285

-- Define the sequence with its initial condition and recursive relation
def sequence : ℕ → ℕ
| 1     := 3
| (n+1) := sequence n + 3

-- Theorem stating that the sequence follows the pattern 3n
theorem sequence_eq_3n (n : ℕ) : sequence n = 3 * n :=
by
  induction n with n ih
  -- Base case
  case zero =>
    -- sequence 1 should be equal to 3 * 1
    show sequence 1 = 3 * 1
    simp [sequence]
  -- Inductive step
  case succ =>
    -- Assume sequence n = 3 * n and prove sequence (n+1) = 3 * (n+1)
    simp [sequence, ih]
  sorry

end sequence_eq_3n_l119_119285


namespace adjacent_cells_sum_divisible_by_4_l119_119987

theorem adjacent_cells_sum_divisible_by_4 :
  ∀ (board: array (fin 2006) (array (fin 2006) ℕ)), 
  (∀ i j, 1 ≤ board[i]![j] ∧ board[i]![j] ≤ 2006^2) →
  ∃ i j, (i < 2005 ∨ j < 2005) ∧
         ((board[i]![j] + board[i + 1]![j]) % 4 = 0 ∨ 
          (board[i]![j] + board[i]![j + 1]) % 4 = 0 ∨ 
          (board[i]![j] + board[i + 1]![j + 1]) % 4 = 0 ∨ 
          (board[i]![j] + board[i + 1]![j - 1]) % 4 = 0) :=
by
  intros board h
  sorry

end adjacent_cells_sum_divisible_by_4_l119_119987


namespace heather_distance_to_meet_l119_119819

noncomputable def distance_walked_by_heather : ℝ :=
  let h_speed : ℝ := 6
  let s_speed : ℝ := 8
  let initial_distance : ℝ := 50
  let heather_start_delay : ℝ := 0.75
  let time_to_meet : ℝ := (50 - 6 * 0.75) / (6 + 8) in
  time_to_meet * h_speed

theorem heather_distance_to_meet (h_speed : ℝ) (s_speed : ℝ) (initial_distance : ℝ) 
(heather_start_delay : ℝ) :
  h_speed = 6 →
  s_speed = h_speed + 2 →
  initial_distance = 50 →
  heather_start_delay = 0.75 →
  distance_walked_by_heather = 18.84 :=
by
  intros
  sorry

end heather_distance_to_meet_l119_119819


namespace son_l119_119948

def woman's_age (W S : ℕ) : Prop := W = 2 * S + 3
def sum_of_ages (W S : ℕ) : Prop := W + S = 84

theorem son's_age_is_27 (W S : ℕ) (h1: woman's_age W S) (h2: sum_of_ages W S) : S = 27 :=
by
  sorry

end son_l119_119948


namespace smallest_product_l119_119439

def is_valid_digit (d : ℕ) : Prop := d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8

def valid_combination (a b c d : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)

def to_two_digit (t u : ℕ) : ℕ := 10 * t + u

theorem smallest_product :
  ∃ (a b c d : ℕ), valid_combination a b c d ∧
  let n1 := to_two_digit a b in
  let n2 := to_two_digit c d in
  n1 * n2 = 3876 :=
begin
  sorry
end

end smallest_product_l119_119439


namespace collection_count_l119_119797

-- Variables and setups for all elements in the problem
def vowels := multiset.mk ['O', 'U', 'A', 'I', 'O', 'A']
def consonants := multiset.mk ['C', 'M', 'P', 'T', 'T', 'N']

-- Combinatoric functions for choosing elements, considering combinations with repetition
noncomputable def count_combinations {A : Type} (s : multiset A) (n : ℕ) : ℕ :=
  (multiset.powerset_len n s).card

/-- Theorem stating the number of distinct collections -/
theorem collection_count :
  let distinct_vowel_ways := count_combinations vowels 3,
      distinct_consonant_ways := count_combinations consonants 3
  in distinct_vowel_ways * distinct_consonant_ways = 196 :=
begin
  -- Production of actual counts for verification
  sorry
end

end collection_count_l119_119797


namespace sum_symmetric_prob_43_l119_119830

def prob_symmetric_sum_43_with_20 : Prop :=
  let n_dice := 9
  let min_sum := n_dice * 1
  let max_sum := n_dice * 6
  let midpoint := (min_sum + max_sum) / 2
  let symmetric_sum := 2 * midpoint - 20
  symmetric_sum = 43

theorem sum_symmetric_prob_43 (n_dice : ℕ) (h₁ : n_dice = 9) (h₂ : ∀ i : ℕ, i ≥ 1 ∧ i ≤ 6) :
  prob_symmetric_sum_43_with_20 :=
by
  sorry

end sum_symmetric_prob_43_l119_119830


namespace find_m_n_l119_119339

theorem find_m_n (m n : ℤ) :
  (∀ x : ℤ, (x + 4) * (x - 2) = x^2 + m * x + n) → (m = 2 ∧ n = -8) :=
by
  intro h
  sorry

end find_m_n_l119_119339


namespace probability_9_heads_12_flips_l119_119164

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l119_119164


namespace polar_eq_C1_area_C1PQ_l119_119741

-- Define the parametric equations of the circle C1
def param_eq_C1 (α : ℝ) : ℝ × ℝ := 
  (2 + 2 * Real.cos α, 1 + 2 * Real.sin α)

-- Define the Cartesian and polar conversion
def cart_to_polar (x y : ℝ) : ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2)
  let θ := Real.atan2 y x
  (ρ, θ)

-- Define the condition: polar equation of C2
def polar_eq_C2 (ρ : ℝ) : ℝ :=
  Real.pi / 4

-- Prove the polar equation of C1
theorem polar_eq_C1 (ρ θ : ℝ) : 
  let (x, y) := (ρ * Real.cos θ, ρ * Real.sin θ)
  (x - 2)^2 + (y - 1)^2 = 4 → ρ^2 - 4 * ρ * Real.cos θ - 2 * ρ * Real.sin θ + 1 = 0 :=
by
  intros h
  sorry

-- Prove the area of triangle C1PQ
theorem area_C1PQ :
  let θ := Real.pi / 4
  let a := (param_eq_C1 0).fst -- x coordinate of one solution
  let b := (param_eq_C1 Real.pi).fst -- x coordinate of the other solution
  let area := (Real.sqrt 14 / 2) * (Real.sqrt 2 / 2)
  area = Real.sqrt 7 / 2 :=
  by
  sorry

end polar_eq_C1_area_C1PQ_l119_119741


namespace carrie_money_left_l119_119225

/-- Carrie was given $91. She bought a sweater for $24, 
    a T-shirt for $6, a pair of shoes for $11,
    and a pair of jeans originally costing $30 with a 25% discount. 
    Prove that she has $27.50 left. -/
theorem carrie_money_left :
  let init_money := 91
  let sweater := 24
  let t_shirt := 6
  let shoes := 11
  let jeans := 30
  let discount := 25 / 100
  let jeans_discounted_price := jeans * (1 - discount)
  let total_cost := sweater + t_shirt + shoes + jeans_discounted_price
  let money_left := init_money - total_cost
  money_left = 27.50 :=
by
  intros
  sorry

end carrie_money_left_l119_119225


namespace compare_f_minus1_f_1_l119_119304

variable (f : ℝ → ℝ)

-- Given conditions
variable (h_diff : Differentiable ℝ f)
variable (h_eq : ∀ x : ℝ, f x = x^2 + 2 * x * (f 2 - 2 * x))

-- Goal statement
theorem compare_f_minus1_f_1 : f (-1) > f 1 :=
by sorry

end compare_f_minus1_f_1_l119_119304


namespace more_elements_in_set_N_l119_119863

theorem more_elements_in_set_N 
  (M N : Finset ℕ) 
  (h_partition : ∀ x, x ∈ M ∨ x ∈ N) 
  (h_disjoint : ∀ x, x ∈ M → x ∉ N) 
  (h_total_2000 : M.card + N.card = 10^2000 - 10^1999) 
  (h_total_1000 : (10^1000 - 10^999) * (10^1000 - 10^999) < 10^2000 - 10^1999) : 
  N.card > M.card :=
by { sorry }

end more_elements_in_set_N_l119_119863


namespace integer_roots_if_q_positive_no_integer_roots_if_q_negative_l119_119970

theorem integer_roots_if_q_positive (p q : ℤ) (hq : q > 0) :
  (∃ x1 x2 : ℤ, x1 * x2 = q ∧ x1 + x2 = p) ∧
  (∃ y1 y2 : ℤ, y1 * y2 = q ∧ y1 + y2 = p + 1) :=
sorry

theorem no_integer_roots_if_q_negative (p q : ℤ) (hq : q < 0) :
  ¬ ((∃ x1 x2 : ℤ, x1 * x2 = q ∧ x1 + x2 = p) ∧
  (∃ y1 y2 : ℤ, y1 * y2 = q ∧ y1 + y2 = p + 1)) :=
sorry

end integer_roots_if_q_positive_no_integer_roots_if_q_negative_l119_119970


namespace bigger_part_l119_119192

theorem bigger_part (x y : ℕ) (h1 : x + y = 54) (h2 : 10 * x + 22 * y = 780) : y = 34 :=
sorry

end bigger_part_l119_119192


namespace part1_part2_l119_119193

-- Probability that A successfully hits the target
def P_A_hit : ℚ := 2/3

-- Probability that B successfully hits the target
def P_B_hit : ℚ := 3/4

-- Prove that probability of A missing at least once in 3 shots is 19/27
theorem part1 : 
  let P_A_miss := 1 - P_A_hit in
  let P_A1 := 1 - P_A_hit ^ 3 in
  P_A1 = 19 / 27 :=
  by sorry

-- Prove that probability of A hitting twice and B hitting once in two shots is 1/6
theorem part2 : 
  let P_A2 := (2.choose 2) * P_A_hit ^ 2 in
  let P_B_miss := 1 - P_B_hit in
  let P_B2 := (2.choose 1) * P_B_hit * P_B_miss in
  P_A2 * P_B2 = 1 / 6 :=
  by sorry

end part1_part2_l119_119193


namespace remaining_bottles_l119_119176

variable (s : ℕ) (b : ℕ) (ps : ℚ) (pb : ℚ)

theorem remaining_bottles (h1 : s = 6000) (h2 : b = 14000) (h3 : ps = 0.20) (h4 : pb = 0.23) : 
  s - Nat.floor (ps * s) + b - Nat.floor (pb * b) = 15580 :=
by
  sorry

end remaining_bottles_l119_119176


namespace jiwoo_hexagon_knob_turn_l119_119759

theorem jiwoo_hexagon_knob_turn :
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 = 720 :=
by
  intros n h
  rw h
  norm_num
  sorry

end jiwoo_hexagon_knob_turn_l119_119759


namespace probability_of_9_heads_in_12_l119_119134

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l119_119134


namespace prob_X_geq_six_minus_m_l119_119422

open MeasureTheory ProbabilityTheory

-- Define the random variable X and its properties
noncomputable def X : MeasureTheory.ProbMeasure ℝ := 
MeasureTheory.ℙ (MeasureTheory.Normal 3 (σ^2))

-- Conditions of the problem
variable (σ : ℝ) (m : ℝ)
hypothesis (h : ∀ m : ℝ, MeasureTheory.ℙ (X > m) = 0.3)

-- Lean statement asserting the final proof goal
theorem prob_X_geq_six_minus_m : MeasureTheory.ℙ (X ≥ 6 - m) = 0.7 := by
  sorry

end prob_X_geq_six_minus_m_l119_119422


namespace function_defined_for_all_reals_l119_119636

theorem function_defined_for_all_reals (m : ℝ) :
  (∀ x : ℝ, 7 * x ^ 2 + m - 6 ≠ 0) → m > 6 :=
by
  sorry

end function_defined_for_all_reals_l119_119636


namespace vanya_speed_problem_l119_119528

theorem vanya_speed_problem (v : ℝ) (hv : (v + 2) / v = 2.5) : ((v + 4) / v) = 4 :=
by
  have v_pos : v ≠ 0 := sorry  -- Prove that v is not zero
  have v_eq : v = 4 / 3 := sorry  -- Use hv to solve for v
  rw [v_eq] at v_eq,
  rw [v_eq]
  sorry

end vanya_speed_problem_l119_119528


namespace find_three_digit_number_l119_119822

noncomputable def three_digit_number := ∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ 100 * x + 10 * y + z = 345 ∧
  (100 * z + 10 * y + x = 100 * x + 10 * y + z + 198) ∧
  (100 * x + 10 * z + y = 100 * x + 10 * y + z + 9) ∧
  (x^2 + y^2 + z^2 - 2 = 4 * (x + y + z))

theorem find_three_digit_number : three_digit_number :=
sorry

end find_three_digit_number_l119_119822


namespace phillip_initial_marbles_l119_119240

theorem phillip_initial_marbles
  (dilan_marbles : ℕ) (martha_marbles : ℕ) (veronica_marbles : ℕ) 
  (total_after_redistribution : ℕ) 
  (individual_marbles_after : ℕ) :
  dilan_marbles = 14 →
  martha_marbles = 20 →
  veronica_marbles = 7 →
  total_after_redistribution = 4 * individual_marbles_after →
  individual_marbles_after = 15 →
  ∃phillip_marbles : ℕ, phillip_marbles = 19 :=
by
  intro h_dilan h_martha h_veronica h_total_after h_individual
  have total_initial := 60 - (14 + 20 + 7)
  existsi total_initial
  sorry

end phillip_initial_marbles_l119_119240


namespace probability_heads_exactly_9_of_12_l119_119012

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l119_119012


namespace probability_of_9_heads_in_12_flips_l119_119062

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l119_119062


namespace kevin_marbles_l119_119812

theorem kevin_marbles (M : ℕ) (h1 : 40 * 3 = 120) (h2 : 4 * M = 320 - 120) :
  M = 50 :=
by {
  sorry
}

end kevin_marbles_l119_119812


namespace unique_prime_factorization_l119_119814

theorem unique_prime_factorization (n : ℕ) (hn : n ≥ 2) :
  ∃ ps : List ℕ, (∀ p ∈ ps, Nat.Prime p) ∧ (ps.prod = n) ∧ (∀ qs : List ℕ, (∀ q ∈ qs, Nat.Prime q) ∧ (qs.prod = n) → Multiset.coalesce ps = Multiset.coalesce qs) :=
sorry

end unique_prime_factorization_l119_119814


namespace hexagon_circle_ratio_correct_l119_119479

noncomputable def hexagon_circle_area_ratio (s r : ℝ) (h : 6 * s = 2 * π * r) : ℝ :=
  let A_hex := (3 * Real.sqrt 3 / 2) * s^2
  let A_circ := π * r^2
  (A_hex / A_circ)

theorem hexagon_circle_ratio_correct (s r : ℝ) (h : 6 * s = 2 * π * r) :
    hexagon_circle_area_ratio s r h = (π * Real.sqrt 3 / 6) :=
sorry

end hexagon_circle_ratio_correct_l119_119479


namespace min_sum_of_factors_of_2310_l119_119857

theorem min_sum_of_factors_of_2310 : ∃ a b c : ℕ, a * b * c = 2310 ∧ a + b + c = 52 :=
by
  sorry

end min_sum_of_factors_of_2310_l119_119857


namespace travel_time_difference_l119_119203

theorem travel_time_difference 
  (speed : ℕ) (distance1 distance2 : ℕ) (time_per_hour : ℕ) 
  (h_speed : speed = 40) (h_distance1 : distance1 = 360) (h_distance2 : distance2 = 400)
  (h_time_per_hour : time_per_hour = 60) : 
  (distance2 / speed - distance1 / speed) * time_per_hour = 60 := 
by
  rw [h_speed, h_distance1, h_distance2, h_time_per_hour]
  norm_num
  sorry

end travel_time_difference_l119_119203


namespace equilateral_triangle_six_equal_areas_l119_119602

/-- Given an equilateral triangle ABC with centroid P inside it,
    the six smaller triangles formed by drawing lines from P to each
    side of the triangle have equal areas. -/
theorem equilateral_triangle_six_equal_areas
  (A B C P : Point)
  (equilateral_triangle : triangle A B C)
  (centroid : centroid_in_triangle P A B C) :
  six_smaller_triangles_have_equal_areas A B C P :=
sorry

end equilateral_triangle_six_equal_areas_l119_119602


namespace integer_1000_column_l119_119952

-- Definitions for the arrangement pattern and columns
def column_pattern : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'E', 'D', 'C', 'B']

-- Definition to find the column of the nth integer
def find_column (n : Nat) : Option Char :=
  column_pattern[(n - 1) % 10]

-- The theorem to determine the column of the integer 1000
theorem integer_1000_column : find_column 1000 = some 'C' := by
  sorry

end integer_1000_column_l119_119952


namespace min_sum_of_factors_l119_119846

theorem min_sum_of_factors (a b c : ℕ) (h1 : a * b * c = 2310) : a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l119_119846


namespace interior_triangle_area_l119_119688

theorem interior_triangle_area (a b c : ℝ)
  (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100)
  (hpythagorean : a^2 + b^2 = c^2) :
  1/2 * a * b = 24 :=
by
  sorry

end interior_triangle_area_l119_119688


namespace probability_exactly_9_heads_in_12_flips_l119_119103

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l119_119103


namespace chord_eq_line_l119_119443

def point := (ℝ × ℝ)
def hyperbola (x y : ℝ) := x^2 - 4*y^2 = 4
def midpoint (A B : point) (P : point) := P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2
def chord_line (A B : point) := 2 * A.1 - A.2 - 15 = 0 ∧ 2 * B.1 - B.2 - 15 = 0

theorem chord_eq_line (P A B : point) 
  (hP : P = (8, 1))
  (h_mid : midpoint A B P)
  (hA_hyper : hyperbola A.1 A.2)
  (hB_hyper : hyperbola B.1 B.2) :
  chord_line A B :=
by {
  sorry
}

end chord_eq_line_l119_119443


namespace find_m_and_f_max_l119_119352

noncomputable def f (x m : ℝ) : ℝ :=
  (Real.sqrt 3) * Real.sin (2 * x) + 2 * (Real.cos x)^2 + m

theorem find_m_and_f_max (m a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x m ≥ 3) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), ∃ y, f y m = 3) →
  (∀ x ∈ Set.Icc a (a + Real.pi), ∃ y, f y m = 6) →
  m = 3 ∧ ∀ x ∈ Set.Icc a (a + Real.pi), f x 3 ≤ 6 :=
sorry

end find_m_and_f_max_l119_119352


namespace charges_are_equal_l119_119646

variable (a : ℝ)  -- original price for both travel agencies

def charge_A (a : ℝ) : ℝ := a + 2 * 0.7 * a
def charge_B (a : ℝ) : ℝ := 3 * 0.8 * a

theorem charges_are_equal : charge_A a = charge_B a :=
by
  sorry

end charges_are_equal_l119_119646


namespace probability_of_9_heads_in_12_flips_l119_119049

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l119_119049


namespace binomial_sum_mod_p_squared_l119_119417

theorem binomial_sum_mod_p_squared (p : ℕ) (hp : Nat.Prime p) (odd_p : p % 2 = 1) : 
  ∑ j in Finset.range (p + 1), Nat.choose p j * Nat.choose (p + j) j ≡ 2 ^ p + 1 [MOD p ^ 2] :=
by sorry

end binomial_sum_mod_p_squared_l119_119417


namespace find_a_l119_119399

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (x + 1)

-- Define the hypothesis that f(f(x)) = x for all x ≠ -1 
def condition (a : ℝ) (x : ℝ) (h : x ≠ -1) : Prop :=
  f a (f a x) = x

-- The main statement: find a such that the condition holds for all x ≠ -1
theorem find_a : ∃ a : ℝ, ∀ x : ℝ, x ≠ -1 → condition a x (by assumption) := 
sorry

end find_a_l119_119399


namespace normal_vector_of_line_l119_119549

theorem normal_vector_of_line :
  ∀ (x y : ℝ), |determinant (2, 1) (x, y)| = 0 → (1, -2) ∈ normal_vector_set x y := 
begin
  intros x y h,
  sorry
end

end normal_vector_of_line_l119_119549


namespace smallest_n_real_numbers_l119_119411

theorem smallest_n_real_numbers (n : ℕ) (y : Fin n → ℝ)
  (h1 : ∀ i, |y i| < 1)
  (h2 : (∑ i, |y i|) = 21 + |∑ i, y i|) :
  n = 22 :=
by {
  sorry
}

end smallest_n_real_numbers_l119_119411


namespace cookies_last_days_l119_119386

variable (c1 c2 t : ℕ)

/-- Jackson's oldest son gets 4 cookies after school each day, and his youngest son gets 2 cookies. 
There are 54 cookies in the box, so the number of days the box will last is 9. -/
theorem cookies_last_days (h1 : c1 = 4) (h2 : c2 = 2) (h3 : t = 54) : 
  t / (c1 + c2) = 9 := by
  sorry

end cookies_last_days_l119_119386


namespace diana_hourly_wage_l119_119641

-- Define the variables and conditions
def hours_per_day (day : String) : ℕ :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" then 10
  else if day = "Tuesday" ∨ day = "Thursday" then 15
  else 0

def total_weekly_hours : ℕ :=
  hours_per_day "Monday" +
  hours_per_day "Tuesday" +
  hours_per_day "Wednesday" +
  hours_per_day "Thursday" +
  hours_per_day "Friday"

def weekly_earnings : ℕ := 1800

def hourly_wage := weekly_earnings / total_weekly_hours

-- The theorem statement
theorem diana_hourly_wage : hourly_wage = 30 :=
by
  unfold hourly_wage
  unfold weekly_earnings total_weekly_hours hours_per_day
  split_ifs
  simp [add_assoc, nat.div_eq_of_eq_mul_right (by decide) rfl]
  sorry

end diana_hourly_wage_l119_119641


namespace sector_area_l119_119685

theorem sector_area (theta : ℝ) (L : ℝ) (h_theta : theta = π / 3) (h_L : L = 4) :
  ∃ r : ℝ, (L = r * theta ∧ ∃ A : ℝ, A = 1/2 * r^2 * theta ∧ A = 24 / π) := by
  sorry

end sector_area_l119_119685


namespace probability_heads_9_of_12_flips_l119_119007

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l119_119007


namespace left_handed_jazz_lovers_l119_119488

variables (total_people left_handed ambidextrous right_handed jazz_lovers right_handed_non_jazz : ℕ)

-- Given conditions
axiom total_people_eq : total_people = 30
axiom left_handed_eq : left_handed = 12
axiom ambidextrous_eq : ambidextrous = 3
axiom right_handed_eq : right_handed = total_people - left_handed - ambidextrous
axiom jazz_lovers_eq : jazz_lovers = 20
axiom right_handed_non_jazz_eq : right_handed_non_jazz = 4
axiom right_handed_jazz_eq : right_handed - right_handed_non_jazz = 11

-- Derived condition
axiom equation : ∀ x y : ℕ, x + y + 11 = jazz_lovers

-- The proof
theorem left_handed_jazz_lovers : ∃ x : ℕ, x + 3 = 9 ∧ x = 6 :=
by
  use 6
  split
  . refl
  . refl

end left_handed_jazz_lovers_l119_119488


namespace sum_of_three_numbers_l119_119555

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : ab + bc + ca = 72) : 
  a + b + c = 14 := 
by 
  sorry

end sum_of_three_numbers_l119_119555


namespace probability_of_9_heads_in_12_l119_119136

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l119_119136


namespace probability_heads_9_of_12_is_correct_l119_119035

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l119_119035


namespace sequence_geometric_T_n_formula_l119_119786

-- Definitions directly from the conditions

def seq_sum_condition (n : ℕ) (a S : ℕ → ℝ) : Prop :=
  ∀ m : ℕ, 0 < m → S m = 2 * a m + m

-- Questions to prove
theorem sequence_geometric (a : ℕ → ℝ) (S : ℕ → ℝ) (h : seq_sum_condition S a) :
  ∃ r, ∀ n, n > 0 → (a n - 1) = r * (a (n-1) - 1) := sorry

theorem T_n_formula (a b S : ℕ → ℝ) (T : ℕ → ℝ) (h1 : seq_sum_condition S a)
  (h2 : ∀ n, b n = (1 - a n) / (a n * a (n + 1))) :
  ∀ n, T n = 1 - 1 / (2^(n + 1) - 1) := sorry

end sequence_geometric_T_n_formula_l119_119786


namespace average_stickers_per_pack_l119_119813

-- Define the conditions given in the problem
def pack1 := 5
def pack2 := 7
def pack3 := 7
def pack4 := 10
def pack5 := 11
def num_packs := 5
def total_stickers := pack1 + pack2 + pack3 + pack4 + pack5

-- Statement to prove the average number of stickers per pack
theorem average_stickers_per_pack :
  (total_stickers / num_packs) = 8 := by
  sorry

end average_stickers_per_pack_l119_119813


namespace probability_heads_exactly_9_of_12_l119_119013

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l119_119013


namespace probability_heads_9_of_12_l119_119138

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l119_119138


namespace probability_exactly_9_heads_l119_119116

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l119_119116


namespace probability_heads_9_of_12_is_correct_l119_119027

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l119_119027


namespace sum_of_odd_primes_11_to_41_l119_119263

noncomputable def sumOfOddPrimesBetween (a b : ℕ) : ℕ :=
  (Finset.filter (fun n => (Prime n) ∧ (Odd n)) (Finset.range (b + 1))).sum id

theorem sum_of_odd_primes_11_to_41 :
  sumOfOddPrimesBetween 11 41 = 221 :=
  sorry

end sum_of_odd_primes_11_to_41_l119_119263


namespace least_value_of_T_l119_119268

noncomputable def f (t : ℝ) : ℝ := sorry -- Define f(t) as described in conditions

theorem least_value_of_T
    (n : ℕ)
    (t : ℕ → ℝ)
    (h_inc : ∀ k : ℕ, 1 ≤ k → t k ≥ t (k - 1) + 1)
    (h_f0 : f (t 0) = 1 / 2)
    (h_lim : ∀ k : ℕ, k ≤ n → ∀ ε > 0, ∃ δ > 0, ∀ t', 0 < t' - t k ∧ t' - t k < δ → abs (f' t' ) < ε)
    (h_f''_interval : ∀ k : ℕ, k < n → ∀ t : ℝ, t k < t ∧ t < t (k + 1) → f'' t = k + 1)
    (h_f''_after : ∀ t > t n, f'' t = n + 1) :
  ∃ T : ℝ, f (t 0 + T) = 2023 ∧ T = 89 :=
begin
  -- The proof will be added here
  sorry
end

end least_value_of_T_l119_119268


namespace weighted_average_profit_margin_l119_119930

def P_A := 0.20
def P_B := 0.25
def P_C := 0.30

def Q_A := 10
def Q_B := 10
def Q_C := 10

def Q_Total := Q_A + Q_B + Q_C

def P_WA := (P_A * Q_A + P_B * Q_B + P_C * Q_C) / Q_Total

theorem weighted_average_profit_margin : P_WA = 0.25 := by
  unfold P_WA Q_A Q_B Q_C P_A P_B P_C Q_Total
  have h1 : Q_Total = 30 := rfl
  have h2 : P_A * Q_A + P_B * Q_B + P_C * Q_C = 7.5 := by norm_num
  rw [h1, h2]
  norm_num

end weighted_average_profit_margin_l119_119930


namespace vanya_faster_speed_l119_119511

def vanya_speed (v : ℝ) : Prop :=
  (v + 2) / v = 2.5

theorem vanya_faster_speed (v : ℝ) (h : vanya_speed v) : (v + 4) / v = 4 :=
by
  sorry

end vanya_faster_speed_l119_119511


namespace min_sum_of_factors_l119_119845

theorem min_sum_of_factors (a b c : ℕ) (h1 : a * b * c = 2310) : a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l119_119845


namespace book_arrangement_problem_l119_119450

/-- 
  Define the number of ways to arrange the books given the conditions.
  - There are 10 books: 2 Arabic, 4 German, and 4 Spanish
  - Arabic books must be kept together
  - Spanish books must be kept together
  Problem:
  Prove in Lean 4 that the number of arrangements is equal to 34,560.
--/
def ways_to_arrange_books : Nat :=
  6! * 2! * 4!

theorem book_arrangement_problem :
  ways_to_arrange_books = 34560 := by
  sorry

end book_arrangement_problem_l119_119450


namespace divides_number_of_ones_l119_119357

theorem divides_number_of_ones (n : ℕ) (h1 : ¬(2 ∣ n)) (h2 : ¬(5 ∣ n)) : ∃ k : ℕ, n ∣ ((10^k - 1) / 9) :=
by
  sorry

end divides_number_of_ones_l119_119357


namespace vanya_faster_by_4_l119_119535

variable (v : ℝ) -- initial speed in m/s
variable (school_distance : ℝ) -- distance to school in meters
variable (time_to_school : ℝ) -- time to school in seconds
variable (increased_speed : ℝ → ℝ) -- increased speed function

-- Conditions
def initial_speed : (v > 0) := by sorry
def increased_speed_by_2 {v : ℝ} : (v + 2 > 0) := by sorry
def faster_by_2_5 {v : ℝ} : (time_to_school = (school_distance / v)) → (time_to_school / 2.5 = (school_distance / (v + 2))) := by sorry

-- Main statement to prove
theorem vanya_faster_by_4 (h1 : initial_speed v) (h2 : increased_speed_by_2 v) (h3 : faster_by_2_5 v time_to_school school_distance) :
    (time_to_school / 4 = (school_distance / (v + 4))) := by
  sorry

end vanya_faster_by_4_l119_119535


namespace total_area_stage4_l119_119349

theorem total_area_stage4 :
  let area (r : ℕ) := Real.pi * (r : ℝ) ^ 2
  in ∑ r in (Finset.range 4), area (r + 1) = 30 * Real.pi :=
by
  sorry

end total_area_stage4_l119_119349


namespace find_some_ounce_size_l119_119625

variable (x : ℕ)
variable (h_total : 122 = 6 * 5 + 4 * x + 15 * 4)

theorem find_some_ounce_size : x = 8 := by
  sorry

end find_some_ounce_size_l119_119625


namespace prove_values_of_a_l119_119627

-- Definitions of the conditions
def condition_1 (a x y : ℝ) : Prop := (x * y)^(1/3) = a^(a^2)
def condition_2 (a x y : ℝ) : Prop := (Real.log x / Real.log a * Real.log y / Real.log a) + (Real.log y / Real.log a * Real.log x / Real.log a) = 3 * a^3

-- The proof problem
theorem prove_values_of_a (a x y : ℝ) (h1 : condition_1 a x y) (h2 : condition_2 a x y) : a > 0 ∧ a ≤ 2/3 :=
sorry

end prove_values_of_a_l119_119627


namespace pyramid_sphere_area_l119_119289

theorem pyramid_sphere_area (a : ℝ) (PA PB PC : ℝ) 
  (h1 : PA = PB) (h2 : PA = 2 * PC) 
  (h3 : PA = 2 * a) (h4 : PB = 2 * a) 
  (h5 : 4 * π * (PA^2 + PB^2 + PC^2) / 9 = 9 * π) :
  a = 1 :=
by
  sorry

end pyramid_sphere_area_l119_119289


namespace fraction_of_males_l119_119218

theorem fraction_of_males (M F : ℝ) 
  (h1 : M + F = 1)
  (h2 : (7 / 8) * M + (4 / 5) * F = 0.845) :
  M = 0.6 :=
by
  sorry

end fraction_of_males_l119_119218


namespace tire_mileage_l119_119947

theorem tire_mileage (total_miles vehicle_tires used_tires : ℕ)
  (h1 : total_miles = 48000)
  (h2 : vehicle_tires = 6)
  (h3 : used_tires = 4) :
  (total_miles * used_tires) / vehicle_tires = 32000 :=
by
  rw [h1, h2, h3]
  calc
    48000 * 4 / 6 = 192000 / 6 : by norm_num
    ...           = 32000 : by norm_num

end tire_mileage_l119_119947


namespace OQ_perpendicular_PQ_l119_119419

-- Definitions
def is_convex_quadrilateral_inscribed_in_circle 
  (A B C D O : Point) : Prop :=
  convex_quadrilateral A B C D ∧
  ∃ (circ : Circle), circ.center = O ∧
  PointOnCircle A circ ∧ PointOnCircle B circ ∧ 
  PointOnCircle C circ ∧ PointOnCircle D circ

def intersection_point 
  (A C B D P : Point) : Prop :=
  LineIntersect (LineThrough A C) (LineThrough B D) P

def circumcircles_intersect 
  (A B P C D Q : Point) : Prop :=
  ∃ (circ₁ circ₂ : Circle),
  PointOnCircle A circ₁ ∧ PointOnCircle B circ₁ ∧ 
  PointOnCircle P circ₁ ∧ PointOnCircle C circ₂ ∧
  PointOnCircle D circ₂ ∧ PointOnCircle P circ₂ ∧
  PointOnCircle Q circ₁ ∧ PointOnCircle Q circ₂

-- Theorem statement
theorem OQ_perpendicular_PQ
  (A B C D O P Q : Point)
  (h1 : is_convex_quadrilateral_inscribed_in_circle A B C D O)
  (h2 : intersection_point A C B D P)
  (h3 : circumcircles_intersect A B P C D Q)
  (h4 : O ≠ P ∧ P ≠ Q ∧ O ≠ Q) : 
  LinePerpendicular (LineThrough O Q) (LineThrough P Q) :=
sorry

end OQ_perpendicular_PQ_l119_119419


namespace CarlsonOriginalLandSize_l119_119973

/-- Carlson bought land that cost $8000 and additional land that cost $4000.
The land he bought costs $20 per square meter.
His land is 900 square meters after buying the new land.
Prove that the size of Carlson's original land before buying the new land is 300 square meters. -/
theorem CarlsonOriginalLandSize
  (cost_new_land1 : ℕ)
  (cost_new_land2 : ℕ)
  (cost_per_sqm : ℕ)
  (total_land_size : ℕ)
  (total_cost_new_land : ℕ := cost_new_land1 + cost_new_land2)
  (new_land_size : ℕ := total_cost_new_land / cost_per_sqm)
  (original_land_size : ℕ := total_land_size - new_land_size) :
  cost_new_land1 = 8000 → cost_new_land2 = 4000 → cost_per_sqm = 20 → total_land_size = 900 →
  original_land_size = 300 :=
begin
  intros h1 h2 h3 h4,
  have h5 : total_cost_new_land = 12000, 
  { rw [h1, h2], norm_num },
  have h6 : new_land_size = 600,
  { rw [h3, h5], norm_num },
  have h7 : original_land_size = 300,
  { rw [h4, h6], norm_num },
  exact h7,
end

end CarlsonOriginalLandSize_l119_119973


namespace solid_rotation_volume_l119_119573

theorem solid_rotation_volume 
  (part1_height part1_width part2_height part2_width : ℝ)
  (h_part1 : part1_height = 5 ∧ part1_width = 1)
  (h_part2 : part2_height = 2 ∧ part2_width = 3) :
  let volume_cylinder1 := π * (part1_height ^ 2) * part1_width in
  let volume_cylinder2 := π * (part2_height ^ 2) * part2_width in
  volume_cylinder1 + volume_cylinder2 = 37 * π :=
by
  sorry

end solid_rotation_volume_l119_119573


namespace sqrt_meaningful_l119_119727

theorem sqrt_meaningful (x : ℝ) (h : 2 - x ≥ 0) : x ≤ 2 :=
sorry

end sqrt_meaningful_l119_119727


namespace digit_sum_equality_l119_119808

-- Definitions for the conditions
def is_permutation_of_digits (a b : ℕ) : Prop :=
  -- Assume implementation that checks if b is a permutation of the digits of a
  sorry

def sum_of_digits (n : ℕ) : ℕ :=
  -- Assume implementation that computes the sum of digits of n
  sorry

-- The theorem statement
theorem digit_sum_equality (a b : ℕ)
  (h : is_permutation_of_digits a b) :
  sum_of_digits (5 * a) = sum_of_digits (5 * b) :=
sorry

end digit_sum_equality_l119_119808


namespace parallel_lines_perpendicular_lines_l119_119702

theorem parallel_lines (a : ℝ) :
  (∃ b c : ℝ, (ax - y + b = 0) ∧ ((a + 2) * x - ay - c = 0)) →
  (∀ s1 s2 : ℝ, s1 = a → s2 = (a + 2) / a → s1 = s2) →
  a = 2 :=
by
  intros
  -- Proof goes here
  sorry

theorem perpendicular_lines (a : ℝ) :
  (∃ b c : ℝ, (ax - y + b = 0) ∧ ((a + 2) * x - ay - c = 0)) →
  (∀ s1 s2 : ℝ, s1 = a → s2 = (a + 2) / a → s1 * s2 = -1) →
  a = 0 ∨ a = -3 :=
by
  intros
  -- Proof goes here
  sorry

end parallel_lines_perpendicular_lines_l119_119702


namespace determine_valid_n_l119_119984

def are_pairwise_coprime (s : Set ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ s → b ∈ s → a ≠ b → Nat.gcd a b = 1

def relatively_prime_numbers (n : ℕ) : Set ℕ :=
  {m | 1 ≤ m ∧ m ≤ n ∧ Nat.gcd m n = 1}

theorem determine_valid_n :
  {n | are_pairwise_coprime (relatively_prime_numbers n)} =
  {1, 2, 3, 4, 6, 8, 12, 18, 24, 30} :=
begin
  sorry
end

end determine_valid_n_l119_119984


namespace infinite_composite_numbers_exist_l119_119454

theorem infinite_composite_numbers_exist (t : ℕ) : ∃ n, n = 3 ^ (2 ^ t) - 2 ^ (2 ^ t) ∧ 
  ¬ nat.prime n ∧ n ∣ (3 ^ (n - 1) - 2 ^ (n - 1)) := 
sorry

end infinite_composite_numbers_exist_l119_119454


namespace marly_pureed_vegetables_l119_119426

theorem marly_pureed_vegetables:
  let milk_quarts := 2 in
  let chicken_stock_quarts := 3 * milk_quarts in
  let total_soup_quarts := 3 * 3 in
  let pureed_vegetable_quarts := total_soup_quarts - (milk_quarts + chicken_stock_quarts) in
  pureed_vegetable_quarts = 1 := 
by sorry

end marly_pureed_vegetables_l119_119426


namespace arrangement_count_l119_119607

theorem arrangement_count:
  ∃! (n : Nat), 
    (n = 24 ∧ 
     ∀ (arrangement : List (Fin 5)), 
      (∃ (l: List (Fin 5)), arrangement.contains (l.get ⟨0, sorry⟩) = some 0 ∧ 
                            arrangement.contains (l.get ⟨1, sorry⟩) = some 1) ∧ 
      ¬((∃ (m: List (Fin 5)), arrangement.contains (m.get ⟨2, sorry⟩) = some 2 ∧ 
                             arrangement.contains (m.get ⟨3, sorry⟩) = some 3)) ∧ 
      ∀ p q r s t, 
        (arrangement = [p, q, r, s, t]))) := sorry

end arrangement_count_l119_119607


namespace trapezoid_perimeter_calculation_l119_119670

noncomputable def trapezoid_perimeter (AB BC CD DA : ℝ) : ℝ :=
  AB + BC + CD + DA

theorem trapezoid_perimeter_calculation :
  let AB := 40
  let BC := 50
  let CD := 60
  let DA := 70
  trapezoid_perimeter AB BC CD DA = 220 :=
by
  -- Definitions of sides for clarity
  let AB := 40
  let BC := 50
  let CD := 60
  let DA := 70
  -- Calculation of the perimeter
  exact trapezoid_perimeter AB BC CD DA = 220
  sorry

end trapezoid_perimeter_calculation_l119_119670


namespace tina_quarters_count_l119_119872

-- Definitions 
def initial_avg_value := 15
def added_coins := 2
def new_avg_value := 17
def initial_coins (n : ℕ) := 15 * n
def new_total_value (n : ℕ) := initial_coins n + 20
def new_number_of_coins (n : ℕ) := n + 2

-- To be proven
theorem tina_quarters_count : 
  ∀ n : ℕ, new_avg_value * new_number_of_coins n = new_total_value n → n = 7 → ∃ q : ℕ, q = 4 := 
by 
  intro n h1 h2 
  use 4 
  sorry

end tina_quarters_count_l119_119872


namespace total_files_on_flash_drive_l119_119601

theorem total_files_on_flash_drive :
  ∀ (music_files video_files picture_files : ℝ),
    music_files = 4.0 ∧ video_files = 21.0 ∧ picture_files = 23.0 →
    music_files + video_files + picture_files = 48.0 :=
by
  sorry

end total_files_on_flash_drive_l119_119601


namespace max_subset_size_with_condition_l119_119935

theorem max_subset_size_with_condition :
  ∃ (S : set ℕ), (∀ a b ∈ S, a ≠ 4 * b ∧ b ≠ 4 * a) ∧ S ⊆ {1..150} ∧ ∀ T : set ℕ, (∀ a b ∈ T, a ≠ 4 * b ∧ b ≠ 4 * a) ∧ T ⊆ {1..150} → T.card ≤ 143 :=
by
  sorry

end max_subset_size_with_condition_l119_119935


namespace num_integers_satisfying_inequality_l119_119712

theorem num_integers_satisfying_inequality :
  ∃ (x : ℕ), ∀ (y: ℤ), (-3 ≤ 3 * y + 2 → 3 * y + 2 ≤ 8) ↔ 4 = x :=
by
  sorry

end num_integers_satisfying_inequality_l119_119712


namespace simplest_quadratic_radical_l119_119600

/-- Definition of the given expressions -/
def exprA := Real.sqrt (1 / 3)
def exprB := Real.sqrt 7
def exprC := Real.sqrt 9
def exprD := Real.sqrt 20

/-- The main statement we want to prove -/
theorem simplest_quadratic_radical : exprB = Real.sqrt 7 :=
by
  sorry

end simplest_quadratic_radical_l119_119600


namespace smallest_composite_no_prime_factors_lt_20_l119_119640

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n : ℕ, (n > 1 ∧ ¬ Prime n ∧ (∀ p : ℕ, Prime p → p < 20 → p ∣ n → False)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_lt_20_l119_119640


namespace find_y_l119_119378

variables {A B C D : Type} [nonempty A] [nonempty B] [nonempty C] [nonempty D]

constants (triangle_ABC : Type) 
          (angle : triangle_ABC → triangle_ABC → triangle_ABC → ℝ)
          (BAC : triangle_ABC → ℝ)
          (ABD : triangle_ABC → ℝ)
          (BAD : triangle_ABC → ℝ)
          (D_on_BC : Prop)
          (y : ℝ)
          (BDA : triangle_ABC → ℝ)

#check C

axiom angle_sum_triangle : ∀ (x y z : triangle_ABC), angle x y z + angle y z x + angle z x y = 180
axiom angle_BAC_eq_90 : BAC = 90
axiom angle_ABD_eq_2y : ABD = 2 * y
axiom angle_BAD_eq_y : BAD = y
axiom BDA_eq_90 : BDA = 90

theorem find_y (h1 : BDA = 90) : y = 30 := by 
  -- proof here
  sorry

end find_y_l119_119378


namespace angle_is_right_l119_119185

structure Triangle :=
(A B C : Type)
(line_AB : A → B → Prop)
(line_BC : B → C → Prop)
(line_CA : C → A → Prop)

def symmetric_point (P Q R : Type) (line_QR : Q → R → Prop) : Type :=
sorry  -- Define how to get the symmetric point of P w.r.t line QR

axiom collinear (P Q R : Type) : Prop

axiom angle (A B C : Type) : ℝ

variables {A B C: Type}
variables [Triangle A B C]
variables {A1 C1 : Type}
variables (h_sym_A1 : A1 = symmetric_point A C B)
variables (h_sym_C1 : C1 = symmetric_point C A B)
variables (h_collinear : collinear A1 B C1)
variables (h_length : dist C1 B = 2 * dist A1 B)

theorem angle_is_right : angle C A1 B = 90 :=
by sorry

end angle_is_right_l119_119185


namespace triangle_area_l119_119994

-- Define the points A, B, and C
def A := (3, -5 : ℝ × ℝ)
def B := (-2, 0 : ℝ × ℝ)
def C := (1, -6 : ℝ × ℝ)

-- Define the vectors v and w
def v := (A.1 - C.1, A.2 - C.2 : ℝ × ℝ)
def w := (B.1 - C.1, B.2 - C.2 : ℝ × ℝ)

-- Define the determinant function to calculate the area of the parallelogram
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Calculate the area of the parallelogram and the area of the triangle
def area_parallelogram := |det v.1 w.1 v.2 w.2|
def area_triangle := (1 / 2) * area_parallelogram

-- The theorem we want to prove
theorem triangle_area : area_triangle = 15 / 2 :=
by
  sorry

end triangle_area_l119_119994


namespace distribute_spots_l119_119918

theorem distribute_spots (total_spots classes : ℕ) (class_spots : ℕ → ℕ) 
  (h_total : total_spots = 12) (h_classes : classes = 6) 
  (h_class_spots_each : ∀ i, 1 ≤ class_spots i) (h_sum : (∑ i in Finset.range classes, class_spots i) = total_spots) :
  (Finset.choose 11 5) = 462 := 
by sorry

end distribute_spots_l119_119918


namespace horners_rule_correct_l119_119495

open Classical

variables (x : ℤ) (poly_val : ℤ)

def original_polynomial (x : ℤ) : ℤ := 7 * x^3 + 3 * x^2 - 5 * x + 11

def horner_evaluation (x : ℤ) : ℤ := ((7 * x + 3) * x - 5) * x + 11

theorem horners_rule_correct : (poly_val = horner_evaluation 23) ↔ (poly_val = original_polynomial 23) :=
by {
  sorry
}

end horners_rule_correct_l119_119495


namespace problem1_problem2_problem3_problem4_l119_119900

-- Problem 1
theorem problem1 (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (h_pos : ∀ n, a_n n > 0) (h_Sn : ∀ n, S_n n = 1/6 * a_n n * (a_n n + 3)) : 
  ∀ n, a_n n = 3 * n :=
sorry

-- Problem 2
theorem problem2 (a : ℕ → ℝ) (h_a1 : a 1 = 0) (h_rec : ∀ n, a (n + 1) = (a n - sqrt(3)) / (sqrt(3) * a n + 1)) : 
  a 2015 = -sqrt(3) :=
sorry

-- Problem 3
theorem problem3 (a b : ℕ → ℝ) (S_n : ℕ → ℝ) 
  (h_an : ∀ n, a n = (Finset.range n).sum (λ k, (k + 1 : ℕ) / (n + 1))) 
  (h_bn : ∀ n, b n = 2 / (a n * a (n + 1))) 
  (h_Sn : ∀ n, S_n n = (Finset.range n).sum b) : 
  ∀ n, S_n n = 8 * n / (n + 1) :=
sorry

-- Problem 4
theorem problem4 (a : ℕ → ℚ) (f : ℚ → ℚ) (h_f : ∃! x, f x = x) 
  (h_f_def : ∀ x, f x = (2:ℤ)*x / (x + 2)) (h_a1 : a 1 = 1 / 2) 
  (h_a_rec : ∀ n, 1 / a (n + 1) = f (1 / a n)) : 
  a 2018 = 1009 := 
sorry

end problem1_problem2_problem3_problem4_l119_119900


namespace b_arithmetic_sum_of_b_and_general_term_of_a_compare_magnitude_l119_119746

-- Define the sequences and conditions
variable {a b : ℕ → ℝ}
variable (q : ℝ) (a1 b1 b3 b5 d : ℤ)

-- Given conditions
axiom h1 : a1 > 1
axiom h2 : q > 0
axiom h3 : ∀ n, b n = log (a n) / log 2
axiom h4 : b1 + b3 + b5 = 6
axiom h5 : b1 * b3 * b5 = 0

-- Prove b_n is arithmetic sequence
theorem b_arithmetic : d = -1 → ∀ n, b (n + 1) - b n = d := sorry

-- Find the sum of first n terms of b_n and general term of a_n
theorem sum_of_b_and_general_term_of_a (n : ℕ) :
  S (n) = (9 * n - n * n) / 2 ∧ a n = 2 ^ (5 - n) := sorry

-- Compare magnitudes of a_n and S_n
theorem compare_magnitude (n : ℕ) :
  (a n > S (n) ∧ (n = 1 ∨ n = 2 ∨ n ≥ 9)) ∨ 
  (a n < S (n) ∧ (n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8)) := sorry

end b_arithmetic_sum_of_b_and_general_term_of_a_compare_magnitude_l119_119746


namespace maria_trip_distance_l119_119897

def total_distance_to_destination (D : ℝ) : Prop :=
  let distance_after_first_stop := D / 2
  let remaining_distance_after_second_stop := distance_after_first_stop - distance_after_first_stop / 4
  remaining_distance_after_second_stop = 135

theorem maria_trip_distance :
  ∃ D : ℝ, total_distance_to_destination D ∧ D = 360 :=
by
  exists 360
  unfold total_distance_to_destination
  norm_num
  sorry

end maria_trip_distance_l119_119897


namespace probability_heads_9_of_12_is_correct_l119_119030

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l119_119030


namespace football_cost_correct_l119_119429

def cost_marble : ℝ := 9.05
def cost_baseball : ℝ := 6.52
def total_cost : ℝ := 20.52
def cost_football : ℝ := total_cost - cost_marble - cost_baseball

theorem football_cost_correct : cost_football = 4.95 := 
by
  -- The proof is omitted, as per instructions.
  sorry

end football_cost_correct_l119_119429


namespace distance_between_A_and_G_l119_119471

-- Define the distances in meters
def d : String → String → ℕ

-- Given conditions
axiom dAB : d "A" "B" = 600
axiom dVG : d "V" "G" = 600

-- Define the possible distance between A and G
def possible_distances (dist : ℕ) : Prop :=
  (dist = 900 ∨ dist = 1800)

-- The main theorem to be proven
theorem distance_between_A_and_G (dBV : ℕ) :
  d "A" "G" = 3 * dBV → possible_distances (d "A" "G") :=
by
  sorry

end distance_between_A_and_G_l119_119471


namespace min_value_of_a_l119_119300

theorem min_value_of_a : 
  ∃ (a : ℤ), ∃ x y : ℤ, x ≠ y ∧ |x| ≤ 10 ∧ (x - y^2 = a) ∧ (y - x^2 = a) ∧ a = -111 :=
by
  sorry

end min_value_of_a_l119_119300


namespace probability_heads_9_of_12_l119_119151

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l119_119151


namespace cone_height_radius_9_circle_sector_l119_119909
open Real

theorem cone_height_radius_9_circle_sector : 
  ∀ r : ℝ, ∀ l : ℝ, ∀ n : ℕ, r = 9 → l = 9 → n = 4 → 
  ∃ h : ℝ, h = sqrt (81 - (2.25)^2) :=
begin
  intro r,
  intro l,
  intro n,
  intro hr,
  intro hl,
  intro hn,
  use sqrt (81 - 5.0625),
  have hr2 : r = 9 := hr,
  have hl2 : l = 9 := hl,
  have hn4 : n = 4 := hn,
  sorry
end

end cone_height_radius_9_circle_sector_l119_119909


namespace prod_div_sum_le_square_l119_119842

theorem prod_div_sum_le_square (m n : ℕ) (h : (m * n) ∣ (m + n)) : m + n ≤ n^2 := sorry

end prod_div_sum_le_square_l119_119842


namespace total_hours_proof_l119_119438

-- Definitions and conditions
def kate_hours : ℕ := 22
def pat_hours : ℕ := 2 * kate_hours
def mark_hours : ℕ := kate_hours + 110

-- Statement of the proof problem
theorem total_hours_proof : pat_hours + kate_hours + mark_hours = 198 := by
  sorry

end total_hours_proof_l119_119438


namespace log_comparison_l119_119277

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 14 / Real.log 7

theorem log_comparison : a > b ∧ b > c := by
  sorry

end log_comparison_l119_119277


namespace top_square_is_fourteen_l119_119894

def initial_grid := [1, 2, 3, 4,
                     5, 6, 7, 8,
                     9, 10, 11, 12,
                     13, 14, 15, 16]

-- Function to perform the 1st fold: right half over the left half.
def fold_right_half (grid : list ℕ) : list ℕ :=
  [grid[2], grid[1], grid[0], grid[3],
   grid[6], grid[5], grid[4], grid[7],
   grid[10], grid[9], grid[8], grid[11],
   grid[14], grid[13], grid[12], grid[15]]

-- Function to perform the 2nd fold: bottom half over the top half.
def fold_bottom_half (grid : list ℕ) : list ℕ :=
  [grid[12], grid[13], grid[14], grid[15],
   grid[8], grid[9], grid[10], grid[11],
   grid[4], grid[5], grid[6], grid[7],
   grid[0], grid[1], grid[2], grid[3]]

-- Function to perform the 3rd fold: right half over the left half again.
def fold_right_half_again (grid : list ℕ) : list ℕ :=
  [grid[1], grid[0], grid[2], grid[3],
   grid[5], grid[4], grid[6], grid[7],
   grid[9], grid[8], grid[10], grid[11],
   grid[13], grid[12], grid[14], grid[15]]

def final_top_square (grid : list ℕ) : ℕ :=
  let after_first_fold := fold_right_half grid
  let after_second_fold := fold_bottom_half after_first_fold
  let final_grid := fold_right_half_again after_second_fold
  final_grid.head -- The top square after all folds

theorem top_square_is_fourteen :
  final_top_square initial_grid = 14 :=
by
  sorry

end top_square_is_fourteen_l119_119894


namespace shadow_building_length_l119_119201

-- Define the basic parameters
def height_flagpole : ℕ := 18
def shadow_flagpole : ℕ := 45
def height_building : ℕ := 20

-- Define the condition on similar conditions
def similar_conditions (h₁ s₁ h₂ s₂ : ℕ) : Prop :=
  h₁ * s₂ = h₂ * s₁

-- Theorem statement
theorem shadow_building_length :
  similar_conditions height_flagpole shadow_flagpole height_building 50 := 
sorry

end shadow_building_length_l119_119201


namespace probability_9_heads_12_flips_l119_119155

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l119_119155


namespace emily_max_servings_l119_119925

variable (chocolate sugar milk : ℕ)

-- Define the recipe's required amounts for 4 servings
def chocolate_for_4_servings : ℕ := 3
def sugar_for_4_servings : ℚ := 1/2
def milk_for_4_servings : ℕ := 3

-- Calculate the maximum number of servings based on each ingredient constraint
def max_servings_based_on_chocolate : ℕ :=
  (9 / chocolate_for_4_servings) * 4
def max_servings_based_on_sugar : ℚ :=
  (3 / sugar_for_4_servings) * 4
def max_servings_based_on_milk : ℚ :=
  (10 / milk_for_4_servings) * 4

-- Prove that the greatest number of servings is 12
theorem emily_max_servings :
  min (max_servings_based_on_chocolate)
      (min (max_servings_based_on_sugar) max_servings_based_on_milk) = 12 := by
  sorry

end emily_max_servings_l119_119925


namespace area_enclosed_by_line_and_curve_l119_119754

theorem area_enclosed_by_line_and_curve :
  ∃ area, ∀ (x : ℝ), x^2 = 4 * (x - 4/2) → 
    area = ∫ (t : ℝ) in Set.Icc (-1 : ℝ) 2, (1/4 * t + 1/2 - 1/4 * t^2) :=
sorry

end area_enclosed_by_line_and_curve_l119_119754


namespace proof_a1_or_a2_l119_119293

-- Define the sets and the conditions.
def A (a : ℕ) : Set ℕ := {0, a}
def B : Set ℕ := {b | b^2 - 3 * b < 0 ∧ b ∈ Int.toNat}

-- Define the proof problem.
theorem proof_a1_or_a2 (a : ℕ) (h1 : (A a) ∩ B ≠ ∅) : a = 1 ∨ a = 2 := 
by
  sorry

end proof_a1_or_a2_l119_119293


namespace omega_range_l119_119296

noncomputable def f (ω x : ℝ) : ℝ :=
  sin (ω * x) + sqrt 3 * cos (ω * x)

theorem omega_range (ω : ℝ) :
  (∀ x ∈ Icc (π / 6) (π / 4), 0 < ω ∧ f ω x = sin (ω * x) + sqrt 3 * cos (ω * x) ∧ monotone_on (f ω) (Icc (π / 6) (π / 4))) ↔ ((0 < ω ∧ ω ≤ 2 / 3) ∨ (7 ≤ ω ∧ ω ≤ 26 / 3)) :=
by
  sorry

end omega_range_l119_119296


namespace sum_imaginary_parts_roots_eq_zero_l119_119774

noncomputable def sum_imaginary_parts_roots (a b c : ℂ) : ℂ :=
  let Δ := b^2 - 4 * a * c in
  let sqrtΔ := complex.sqrt Δ in
  let z1 := (-b + sqrtΔ) / (2 * a) in
  let z2 := (-b - sqrtΔ) / (2 * a) in
  (z1.im + z2.im)

theorem sum_imaginary_parts_roots_eq_zero :
  let i := complex.I in
  sum_imaginary_parts_roots 1 (-3) (-(8 - 6 * i)) = 0 :=
by
  sorry

end sum_imaginary_parts_roots_eq_zero_l119_119774


namespace find_PN_distance_l119_119292

open Real

-- Define M and N as given points
def M : Point := (0, -2)
def N : Point := (0, 2)

-- Define P(x, y) and the equation it satisfies
variable (x y : ℝ)
def P_satisfies : Prop := sqrt (x^2 + y^2 + 4*y + 4) + sqrt (x^2 + y^2 - 4*y + 4) = 10

-- Condition: |PM| = 7
def PM_distance : ℝ := 7
def PM_eq : Prop := dist (x, y) M = PM_distance

-- Goal: |PN| = 3
theorem find_PN_distance (h1 : P_satisfies) (h2 : PM_eq) : dist (x, y) N = 3 := by
  sorry

end find_PN_distance_l119_119292


namespace monic_quartic_polynomial_with_given_roots_l119_119252

theorem monic_quartic_polynomial_with_given_roots :
  ∃ (p : Polynomial ℚ), 
    p = Polynomial.monic * (Polynomial.X^4 - 10 * Polynomial.X^3 + 25 * Polynomial.X^2 + 2 * Polynomial.X - 12) ∧
    p.eval (3 + Real.sqrt 5) = 0 ∧ 
    p.eval (2 - Real.sqrt 7) = 0 ∧
    p.eval (3 - Real.sqrt 5) = 0 ∧
    p.eval (2 + Real.sqrt 7) = 0 :=
by {
  -- Proof will be provided to satisfy the conditions
  sorry
}

end monic_quartic_polynomial_with_given_roots_l119_119252


namespace find_m_l119_119677

variable {α : Type} [DecidableEq α]

theorem find_m 
  (m : α) 
  (A : Set α) (B : Set α)
  (hA : A = {0, m}) 
  (hB : B = {0, 2 : α}) 
  (hUnion : A ∪ B = {0, 1, 2}) : 
  m = 1 := 
  sorry

end find_m_l119_119677


namespace cement_weight_percentage_l119_119711

/-- 
Given original weights of type-A, type-B, and type-C cement as 215 lbs, 137 lbs, and 150 lbs 
respectively, and their weight losses as 3%, 2%, and 5% respectively, prove that the percentage 
of the total original cement they now have in the form of type-D cement is approximately 96.67%.
 -/
theorem cement_weight_percentage :
  let original_weight_A := 215
      original_weight_B := 137
      original_weight_C := 150
      loss_A := 0.03
      loss_B := 0.02
      loss_C := 0.05
      remaining_weight_A := original_weight_A * (1 - loss_A)
      remaining_weight_B := original_weight_B * (1 - loss_B)
      remaining_weight_C := original_weight_C * (1 - loss_C)
      total_remaining_weight := remaining_weight_A + remaining_weight_B + remaining_weight_C
      total_original_weight := original_weight_A + original_weight_B + original_weight_C
  in (total_remaining_weight / total_original_weight) * 100 ≈ 96.67 := sorry

end cement_weight_percentage_l119_119711


namespace probability_heads_9_of_12_is_correct_l119_119036

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l119_119036


namespace rectangle_area_l119_119735

-- Define the vertices of the rectangle
def V1 : ℝ × ℝ := (-7, 1)
def V2 : ℝ × ℝ := (1, 1)
def V3 : ℝ × ℝ := (1, -6)
def V4 : ℝ × ℝ := (-7, -6)

-- Define the function to compute the area of the rectangle given the vertices
noncomputable def area_of_rectangle (A B C D : ℝ × ℝ) : ℝ :=
  let length := abs (B.1 - A.1)
  let width := abs (A.2 - D.2)
  length * width

-- The statement to prove
theorem rectangle_area : area_of_rectangle V1 V2 V3 V4 = 56 := by
  sorry

end rectangle_area_l119_119735


namespace profit_in_december_l119_119731

variable (a : ℝ)

theorem profit_in_december (h_a: a > 0):
  (1 - 0.06) * (1 + 0.10) * a = (1 - 0.06) * (1 + 0.10) * a :=
by
  sorry

end profit_in_december_l119_119731


namespace triangle_area_vertices_l119_119595

theorem triangle_area_vertices :
  let (x1, y1) := (-2 : ℝ, 3 : ℝ)
  let (x2, y2) := (7 : ℝ, -3 : ℝ)
  let (x3, y3) := (4 : ℝ, 6 : ℝ)
  let area := (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
  area = 31.5 :=
by
  sorry

end triangle_area_vertices_l119_119595


namespace problem_one_problem_two_l119_119315

noncomputable def f (x m : ℝ) : ℝ := x^2 + m * x + 4

-- Problem (I)
theorem problem_one (m : ℝ) :
  (∀ x, 1 < x ∧ x < 2 → f x m < 0) ↔ m ≤ -5 :=
sorry

-- Problem (II)
theorem problem_two (m : ℝ) :
  (∀ x, (x = 1 ∨ x = 2) → abs ((f x m - x^2) / m) < 1) ↔ (-4 < m ∧ m ≤ -2) :=
sorry

end problem_one_problem_two_l119_119315


namespace committee_form_count_l119_119608

def numWaysToFormCommittee (departments : Fin 4 → (ℕ × ℕ)) : ℕ :=
  let waysCase1 := 6 * 81 * 81
  let waysCase2 := 6 * 9 * 9 * 2 * 9 * 9
  waysCase1 + waysCase2

theorem committee_form_count (departments : Fin 4 → (ℕ × ℕ)) 
  (h : ∀ i, departments i = (3, 3)) :
  numWaysToFormCommittee departments = 48114 := 
by
  sorry

end committee_form_count_l119_119608


namespace arrangement_reads_XXOXXOXO_l119_119267

open Finset

noncomputable def probability_specific_arrangement : Real :=
1 / (choose 8 5)

theorem arrangement_reads_XXOXXOXO :
  probability_specific_arrangement = 1 / 56 :=
by
  sorry

end arrangement_reads_XXOXXOXO_l119_119267


namespace number_of_good_permutations_l119_119585

def is_bad_permutation (σ : Perm (Fin 10)) : Prop :=
  ∃ i j k : Fin 10, i < j ∧ j < k ∧ σ j < σ k ∧ σ k < σ i

def is_good_permutation (σ : Perm (Fin 10)) : Prop :=
  ¬ is_bad_permutation σ

theorem number_of_good_permutations : 
  (Finset.filter is_good_permutation (Finset.univ : Finset (Perm (Fin 10)))).card = 16796 :=
sorry

end number_of_good_permutations_l119_119585


namespace quotient_ker_is_iso_im_l119_119771

open GroupTheory

-- Define the groups G and H
variables (G : Type*) [Group G] (H : Type*) [Group H]

-- Define the group homomorphism ϕ: G → H
variable (ϕ : G →* H)

-- Prove that G / Ker(ϕ) is isomorphic to Im(ϕ)
theorem quotient_ker_is_iso_im : (G ⧸ ϕ.ker) ≃* (ϕ.range) :=
  sorry

end quotient_ker_is_iso_im_l119_119771


namespace greatest_possible_edges_l119_119281

theorem greatest_possible_edges (n : ℕ) (h : n ≥ 4) 
  (cond : ∀ (G : Type) [fintype G], ∀ (u v : G), ∃ (w : G), ¬(w = u) ∧ ¬(w = v)) : 
  ∃ E : ℕ, E = (n - 1) * (n - 3) / 2 :=
by 
  sorry

end greatest_possible_edges_l119_119281


namespace solve_for_x_l119_119722

theorem solve_for_x (x : ℤ) (h : (2^x) - (2^(x-2)) = 3 * (2^12)) : x = 15 :=
by
  sorry

end solve_for_x_l119_119722


namespace max_sum_length_l119_119376

-- Define the setting: a regular 2n-gon with vertices represented by vectors from the origin.
def vertices (n : ℕ) := { k : ℕ // k < 2 * n }

-- The property we want to prove: maximum sum length is achieved with n vectors.
theorem max_sum_length (n : ℕ) : ∃ v : Finset (vertices n), v.card = n ∧ (∀ w : Finset (vertices n), w.card = n → sum_length v ≥ sum_length w) :=
begin
  sorry
end

end max_sum_length_l119_119376


namespace sum_of_coordinates_F_l119_119372

-- Define the points A, B, C, D, and E
def A : ℝ × ℝ := (0, 6)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (8, 0)
def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the lines AE and CD
def line_AE (x : ℝ) : ℝ := -(3/2) * x + 6
def line_CD (x : ℝ) : ℝ := -(3/8) * x + 3

-- Define the intersection point F
def F : ℝ × ℝ := 
  let x := 8 / 3 in
  (x, line_AE x)

-- The theorem statement
theorem sum_of_coordinates_F : F.1 + F.2 = 14 / 3 :=
by { -- Proof would go here
  sorry 
}

end sum_of_coordinates_F_l119_119372


namespace min_value_expression_l119_119446

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_condition : a^2 * b + b^2 * c + c^2 * a = 3) : 
  (sqrt (a^6 + b^4 * c^6) / b + 
   sqrt (b^6 + c^4 * a^6) / c + 
   sqrt (c^6 + a^4 * b^6) / a) ≥ 3 * sqrt 2 :=
by
  sorry

end min_value_expression_l119_119446


namespace polygon_sides_arithmetic_progression_l119_119833

theorem polygon_sides_arithmetic_progression
  (n : ℕ)
  (h1 : ∀ i, 1 ≤ i → i ≤ n → 172 - (i - 1) * 8 > 0) -- Each angle in the sequence is positive
  (h2 : (∀ i, 1 ≤ i → i ≤ n → (172 - (i - 1) * 8) < 180)) -- Each angle < 180 degrees
  (h3 : n * (172 - (n-1) * 4) = 180 * (n - 2)) -- Sum of interior angles formula
  : n = 10 :=
sorry

end polygon_sides_arithmetic_progression_l119_119833


namespace probability_9_heads_12_flips_l119_119161

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l119_119161


namespace amount_spent_first_time_l119_119423

variables (M : ℝ)

def condition1 := ∃ M, M > 0
def condition2 := ∃ i, i = (2 / 5) * M
def condition3 := ∃ r, r = (3 / 5) * M + 240
def condition4 := ∃ s, s = (2 / 3) * ((3 / 5) * M + 240) ∧ s = 720

theorem amount_spent_first_time (M : ℝ) (c1 : condition1) (c2 : condition2) (c3 : condition3) (c4 : condition4) :
  c2 → ∃ a, a = (2 / 5) * M := 
by 
  sorry

end amount_spent_first_time_l119_119423


namespace interval_of_monotonic_increase_l119_119834

-- Define the function y
def y (x : ℝ) : ℝ := Real.log (-x^2 + 4 * x)

-- Define the monotonic increasing interval
def monotonic_increasing_interval : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem interval_of_monotonic_increase : 
  ∀ x, (0 < x ∧ x ≤ 2) ↔ ∃ δ > 0, ∀ ε, 0 < ε < δ → y (x + ε) > y x :=
  sorry

end interval_of_monotonic_increase_l119_119834


namespace probability_heads_in_nine_of_twelve_flips_l119_119090

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l119_119090


namespace x_plus_p_l119_119345

theorem x_plus_p (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) : x + p = 2 * p + 3 :=
by
  sorry

end x_plus_p_l119_119345


namespace AdjacentComplementaryAnglesAreComplementary_l119_119892

-- Definitions of angles related to the propositions

def acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2
def complementary (θ φ : ℝ) : Prop := θ + φ = π / 2
def adjacent_complementary (θ φ : ℝ) : Prop := complementary θ φ ∧ θ ≠ φ  -- Simplified for demonstration
def corresponding (θ φ : ℝ) : Prop := sorry  -- Definition placeholder
def interior_alternate (θ φ : ℝ) : Prop := sorry  -- Definition placeholder

theorem AdjacentComplementaryAnglesAreComplementary :
  ∀ θ φ : ℝ, adjacent_complementary θ φ → complementary θ φ :=
by
  intros θ φ h
  cases h with h_complementary h_adjacent
  exact h_complementary

end AdjacentComplementaryAnglesAreComplementary_l119_119892


namespace arrangement_schemes_count_l119_119642

open Finset

-- Definitions based on the given conditions
def teachers : Finset ℕ := {1, 2}
def students : Finset ℕ := {1, 2, 3, 4}
def choose_two (s : Finset ℕ) := s.choose 2

-- The theorem to prove the total number of different arrangement schemes is 12
theorem arrangement_schemes_count : 
  (teachers.card.choose 1) * ((students.card.choose 2)) = 12 :=
by
  -- Number of ways to select teachers
  have teachers_ways : teachers.card.choose 1 = 2 := by sorry
  -- Number of ways to select students
  have students_ways : students.card.choose 2 = 6 := by sorry
  -- Multiplying the ways
  rw [teachers_ways, students_ways]
  norm_num

end arrangement_schemes_count_l119_119642


namespace probability_exactly_9_heads_in_12_flips_l119_119098

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l119_119098


namespace songs_today_l119_119757

-- Define the conditions
variable (S_yesterday : ℕ)
variable (S_total : ℕ)
variable (S_today : ℕ)

-- Assign values to the conditions
axiom S_yesterday_9 : S_yesterday = 9
axiom S_total_23 : S_total = 23

-- Define the goal
theorem songs_today (h₁ : S_yesterday = 9) (h₂ : S_total = 23) : S_today = 23 - 9 :=
by
  rw [h₁, h₂]
  exact rfl

end songs_today_l119_119757


namespace triangle_b_c_sum_l119_119360

variables (A : Type) [linear_ordered_field A]

noncomputable def find_b_plus_c (cos_A a bc : A) (h1 : cos_A = (1 : A) / 3)
                                (h2 : a = real.sqrt 3)
                                (h3 : bc = (3 : A) / 2) : A :=
by sorry

theorem triangle_b_c_sum (cos_A a b c : A) (h1 : cos_A = (1 : A) / 3)
                         (h2 : a = real.sqrt 3)
                         (h3 : bc = (3 : A) / 2)
                         (h4 : a^2 = b^2 + c^2 - 2 * bc * cos_A) :
  b + c = real.sqrt 7 :=
by sorry

end triangle_b_c_sum_l119_119360


namespace part1_part2_l119_119725

noncomputable def star (a b : ℚ) := 4 * a * b

theorem part1 : star 3 (-4) = -48 :=
by
  have h : 4 * 3 * (-4) = -48 := by norm_num
  exact h

theorem part2 : star (-2) (star 6 3) = -576 :=
by
  have h1 : star 6 3 = 72 := by norm_num
  have h2 : star (-2) 72 = -576 := by norm_num
  exact h2

end part1_part2_l119_119725


namespace min_value_l119_119773

theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  ∃ (min_val : ℝ), min_val = 9 ∧ (∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 2 * b = 1 → (1 / a + 2 / b) ≥ min_val) :=
begin
  use 9,
  split,
  { refl }, 
  { intros x y hx hy hxy,
    sorry
  }
end

end min_value_l119_119773


namespace min_sum_of_factors_l119_119856

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 2310) : 
  a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l119_119856


namespace probability_exactly_9_heads_in_12_flips_l119_119100

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l119_119100


namespace candy_store_price_per_pound_fudge_l119_119567

theorem candy_store_price_per_pound_fudge 
  (fudge_pounds : ℕ)
  (truffles_dozen : ℕ)
  (truffles_price_each : ℝ)
  (pretzels_dozen : ℕ)
  (pretzels_price_each : ℝ)
  (total_revenue : ℝ) 
  (truffles_total : ℕ := truffles_dozen * 12)
  (pretzels_total : ℕ := pretzels_dozen * 12)
  (truffles_revenue : ℝ := truffles_total * truffles_price_each)
  (pretzels_revenue : ℝ := pretzels_total * pretzels_price_each)
  (fudge_revenue : ℝ := total_revenue - (truffles_revenue + pretzels_revenue))
  (fudge_price_per_pound : ℝ := fudge_revenue / fudge_pounds) :
  fudge_pounds = 20 →
  truffles_dozen = 5 →
  truffles_price_each = 1.50 →
  pretzels_dozen = 3 →
  pretzels_price_each = 2.00 →
  total_revenue = 212 →
  fudge_price_per_pound = 2.5 :=
by 
  sorry

end candy_store_price_per_pound_fudge_l119_119567


namespace bob_expected_rolls_in_leap_year_l119_119611

-- Definitions derived from the problem conditions
def die_sides := {n : ℕ | 1 ≤ n ∧ n ≤ 8}
def composite (n : ℕ) : Prop := n = 4 ∨ n = 6 ∨ n = 8
def prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7
def reroll (n : ℕ) : Prop := n = 1 ∨ n = 8
def leap_year_days : ℕ := 366

-- Theorem statement based on the problem's question and answer
theorem bob_expected_rolls_in_leap_year : 
  (∑ n in die_sides, 
    if composite n ∨ prime n then 1 
    else if reroll n then (1 + (E : ℝ)) 
    else 0 ) * leap_year_days = 488 := 
sorry

end bob_expected_rolls_in_leap_year_l119_119611


namespace line_segment_represents_feet_l119_119928

theorem line_segment_represents_feet : 
  ∀ (length_in_inches : ℝ) (scale : ℝ), length_in_inches = 7.5 → scale = 500 → (length_in_inches * scale = 3750) :=
by
  intros length_in_inches scale h_length h_scale
  rw [h_length, h_scale]
  norm_num
  sorry

end line_segment_represents_feet_l119_119928


namespace maximize_two_presses_number_after_2023_presses_l119_119490

-- Given conditions:
def initial_number := -1

def operation_A (n : ℤ) := n + 3
def operation_B (n : ℤ) := n * -2
def operation_C (n : ℤ) := n / 4

-- Prove that the key sequence B -> A maximizes the number after two presses
theorem maximize_two_presses :
  operation_A (operation_B initial_number) = 5 :=
by
  sorry

-- Prove that the number after 2023 key presses in the sequence A -> B -> C -> A -> B -> C -> ... is 2
theorem number_after_2023_presses :
  let sequence := [operation_A, operation_B, operation_C]
  let final_number := list.foldl (fun n f => f n) initial_number (list.take 2023 (sequence.cycle))
  final_number = 2 :=
by
  sorry

end maximize_two_presses_number_after_2023_presses_l119_119490


namespace ratio_sheila_purity_l119_119459

theorem ratio_sheila_purity (rose_share : ℕ) (total_rent : ℕ) (purity_share : ℕ) (sheila_share : ℕ) 
  (h1 : rose_share = 1800) 
  (h2 : total_rent = 5400) 
  (h3 : rose_share = 3 * purity_share)
  (h4 : total_rent = purity_share + rose_share + sheila_share) : 
  sheila_share / purity_share = 5 :=
by
  -- Proof will be here
  sorry

end ratio_sheila_purity_l119_119459


namespace P_lt_Q_l119_119679

noncomputable def P (a : ℝ) : ℝ := (Real.sqrt (a + 41)) - (Real.sqrt (a + 40))
noncomputable def Q (a : ℝ) : ℝ := (Real.sqrt (a + 39)) - (Real.sqrt (a + 38))

theorem P_lt_Q (a : ℝ) (h : a > -38) : P a < Q a := by sorry

end P_lt_Q_l119_119679


namespace probability_heads_9_of_12_flips_l119_119002

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l119_119002


namespace probability_heads_in_nine_of_twelve_flips_l119_119095

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l119_119095


namespace false_statement_c_l119_119551

-- Definitions based on given conditions:
def vertical_angles_are_equal : Prop :=
  ∀ (a b c d : ℝ), a + b = 180 ∧ c + d = 180 ∧ a = c ∧ b = d → a = d ∧ b = c

def corresponding_angles_and_parallel_lines : Prop :=
  ∀ (L₁ L₂ : ℝ), (∃ t : ℝ, t ∠ L₁ = t ∠ L₂) → L₁ ∥ L₂

def alternate_interior_angles_and_parallel_lines : Prop :=
  ∀ (L₁ L₂ : ℝ), L₁ ∥ L₂ → (∃ t₁ t₂ : ℝ, t₁ ∠ L₁ = t₂ ∠ L₂ ∧ t₁ ≠ t₂)

def parallel_postulate : Prop :=
  ∀ (L : ℝ) (P : ℝ), (∀ t : ℝ, t ∠ L = P) → (∃! L' : ℝ, P ∠ L' = P ∠ L ∧ L ∥ L')

-- The proof problem:
theorem false_statement_c :
  (¬ alternate_interior_angles_and_parallel_lines) :=
sorry

end false_statement_c_l119_119551


namespace ramus_profit_percent_l119_119810

theorem ramus_profit_percent
  (purchase_cost : ℕ) (repair_cost : ℕ) (insurance_cost : ℕ) (registration_fee : ℕ)
  (depreciation_rate : ℕ) (part_exchange_value : ℕ) (selling_price : ℕ) :
  purchase_cost = 42000 →
  repair_cost = 13000 →
  insurance_cost = 5000 →
  registration_fee = 3000 →
  depreciation_rate = 10 →
  part_exchange_value = 8000 →
  selling_price = 76000 →
  let total_cost := purchase_cost + repair_cost + insurance_cost + registration_fee in
  let depreciation := (depreciation_rate * purchase_cost) / 100 in
  let value_after_depreciation := purchase_cost - depreciation in
  let effective_cost := total_cost - part_exchange_value in
  let profit := selling_price - effective_cost in
  let profit_percent := (profit * 100) / effective_cost in
  profit_percent = 38.18 := by
  intros _pc _rc _ic _rf _dr _pev _sp;
  simp [total_cost, depreciation, value_after_depreciation, effective_cost, profit, profit_percent];
  sorry

end ramus_profit_percent_l119_119810


namespace marked_price_proof_l119_119587

def purchased_price (original_price discount : ℝ) : ℝ :=
  original_price * (1 - discount)

def selling_price (cost gain : ℝ) : ℝ :=
  cost * (1 + gain)

def marked_price (selling applied_discount : ℝ) : ℝ :=
  selling / (1 - applied_discount)

theorem marked_price_proof : marked_price (selling_price (purchased_price 30 0.15) 0.25) 0.10 = 35.42 := by
  sorry

end marked_price_proof_l119_119587


namespace probability_heads_in_9_of_12_flips_l119_119068

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l119_119068


namespace problem_statement_l119_119333

theorem problem_statement (x : ℝ) (h : (x / 5) / 3 = 5 / (x / 3)) : x = 15 ∨ x = -15 := 
by
  sorry

end problem_statement_l119_119333


namespace daily_evaporation_rate_l119_119904

theorem daily_evaporation_rate (initial_amount : ℝ) (period : ℕ) (percentage_evaporated : ℝ) (h_initial : initial_amount = 10) (h_period : period = 50) (h_percentage : percentage_evaporated = 4) : 
  (percentage_evaporated / 100 * initial_amount) / period = 0.008 :=
by
  -- Ensures that the conditions translate directly into the Lean theorem statement
  rw [h_initial, h_period, h_percentage]
  -- Insert the required logical proof here
  sorry

end daily_evaporation_rate_l119_119904


namespace lattice_points_distance_5_l119_119380

def is_lattice_point (x y z : ℤ) : Prop :=
  x^2 + y^2 + z^2 = 25

theorem lattice_points_distance_5 : 
  ∃ S : Finset (ℤ × ℤ × ℤ), 
    (∀ p ∈ S, is_lattice_point p.1 p.2.1 p.2.2) ∧
    S.card = 78 :=
by
  sorry

end lattice_points_distance_5_l119_119380


namespace compare_y_coords_l119_119331

theorem compare_y_coords (y₁ y₂ : ℝ) :
  (∃ (x1 : ℝ), x1 = 2 * Real.sqrt 3 ∧ y₁ = 2 * x1 - 1) ∧ 
  (∃ (x2 : ℝ), x2 = 3 * Real.sqrt 2 ∧ y₂ = 2 * x2 - 1) → 
  y₁ < y₂ :=
begin
  sorry
end

end compare_y_coords_l119_119331


namespace chip_position_bulbs_lit_l119_119487
open Nat

theorem chip_position_bulbs_lit :
  ∃ m : ℕ, (∀ x : ℕ, 2 ≤ x ∧ x ≤ 64 → (m * (x - 1)) % 64 = 0) ∧ (m % 64 = 0) :=
by
  use 64
  split
  {
    intro x
    intro hx
    cases hx with hx1 hx2
    have h_div: 64 ∣ m * (x - 1), from dvd_mul_of_dvd_right (dvd_refl 64) (x - 1)
    exact Nat.dvd_iff_mod_eq_zero.mp h_div
  }
  {
    exact Nat.dvd_iff_mod_eq_zero.mpr (dvd_refl 64)
  }

end chip_position_bulbs_lit_l119_119487


namespace younger_brother_age_l119_119867

variable (x y : ℕ)

theorem younger_brother_age :
  x + y = 46 →
  y = x / 3 + 10 →
  y = 19 :=
by
  intros h1 h2
  sorry

end younger_brother_age_l119_119867


namespace angle_equality_proof_l119_119768

noncomputable def acute_triangle (ABC : Type*) [nonempty ABC] :=
  ∃ (A B C : ABC), ∠ A B C < 90 ∧ ∠ B A C < 90 ∧ ∠ C B A < 90 ∧ ∠ B A C > 45

open_locale euclidean_geometry

theorem angle_equality_proof
  {ABC : Type*}
  [metric_space ABC] [normed_group ABC] [normed_space ℝ ABC] [inner_product_space ℝ ABC]
  [nonempty ABC]
  (A B C : ABC)
  (h_ABC : acute_triangle ABC)
  (O P Q : ABC)
  (hO : is_circumcenter O A B C)
  (hP : is_concyclic A P O B)
  (h_perpendicular : ∠(P - B) = ∠(C - B) + π / 2)
  (hQ : lies_on_segment Q B P)
  (h_parallel : is_parallel (A - Q) (P - O)) : 
  ∠ QCB = ∠ PCO :=
by sorry

end angle_equality_proof_l119_119768


namespace work_together_days_l119_119905

theorem work_together_days (A_rate B_rate x total_work B_days_worked : ℚ)
  (hA : A_rate = 1/4)
  (hB : B_rate = 1/8)
  (hCombined : (A_rate + B_rate) * x + B_rate * B_days_worked = total_work)
  (hTotalWork : total_work = 1)
  (hBDays : B_days_worked = 2) : x = 2 :=
by
  sorry

end work_together_days_l119_119905


namespace common_plane_exists_l119_119924

theorem common_plane_exists :
  ∀ (n : ℕ), n ∈ {7, 15, 23, 26, 31, 39, 47, 53, 55, 63, 71, 79, 80, 87, 95} →
  ∃ (i j : ℕ) (m m' : ℕ), (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ n) ∧ (m = i ∧ m' = j) ∧ i ≠ j :=
sorry

end common_plane_exists_l119_119924


namespace andrew_donuts_l119_119603

/--
Andrew originally asked for 3 donuts for each of his 2 friends, Brian and Samuel. 
Then invited 2 more friends and asked for the same amount of donuts for them. 
Andrew’s mother wants to buy one more donut for each of Andrew’s friends. 
Andrew's mother is also going to buy the same amount of donuts for Andrew as everybody else.
Given these conditions, the total number of donuts Andrew’s mother needs to buy is 20.
-/
theorem andrew_donuts : (3 * 2) + (3 * 2) + 4 + 4 = 20 :=
by
  -- Given:
  -- 1. Andrew asked for 3 donuts for each of his two friends, Brian and Samuel.
  -- 2. He later invited 2 more friends and asked for the same amount of donuts for them.
  -- 3. Andrew’s mother wants to buy one more donut for each of Andrew’s friends.
  -- 4. Andrew’s mother is going to buy the same amount of donuts for Andrew as everybody else.
  -- Prove: The total number of donuts Andrew’s mother needs to buy is 20.
  sorry

end andrew_donuts_l119_119603


namespace calculate_expression_l119_119620

theorem calculate_expression :
  (4 - Real.sqrt 3) ^ 0
  - 3 * Real.tan (Float.pi / 3)
  - (-1 / 2)⁻¹
  + Real.sqrt 12
  = 3 - Real.sqrt 3 :=
by 
  have h1 : (4 - Real.sqrt 3) ^ 0 = 1 := by norm_num
  have h2 : Real.tan (Float.pi / 3) = Real.sqrt 3 := by apply Real.tan_pi_div_three
  have h3 : (-1 / 2 : ℝ)⁻¹ = -2 := by norm_num
  have h4 : Real.sqrt 12 = 2 * Real.sqrt 3 := by norm_num
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end calculate_expression_l119_119620


namespace n_gon_diagonal_regions_l119_119186

theorem n_gon_diagonal_regions (n : ℕ) (h : n ≥ 3) :
  let D := n * (n - 3) / 2,
      I := n * (n - 1) * (n - 2) * (n - 3) / 24 in
  D + I + 1 = (n * (n - 1) * (n - 2) * (n - 3)) / 24 + (n * (n - 3)) / 2 + 1 := 
begin
  sorry
end

end n_gon_diagonal_regions_l119_119186


namespace color_points_l119_119740

theorem color_points {points : Set Point} (h_card : points.card = 2005)
  (h_no_four_coplanar : ∀ (p1 p2 p3 p4 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points → ¬ coplanar p1 p2 p3 p4) :
  ∃ (f : Point → Prop), (∀ (p1 p2 : Point), p1 ∈ points → p2 ∈ points → (f p1 = f p2 ↔ odd (num_planes_separating p1 p2))) :=
begin
  sorry
end

end color_points_l119_119740


namespace find_m_n_l119_119340

theorem find_m_n (m n : ℤ) :
  (∀ x : ℤ, (x + 4) * (x - 2) = x^2 + m * x + n) → (m = 2 ∧ n = -8) :=
by
  intro h
  sorry

end find_m_n_l119_119340


namespace positive_difference_proof_l119_119868

noncomputable def solve_system : Prop :=
  ∃ (x y : ℝ), 
  (x + y = 40) ∧ 
  (3 * y - 4 * x = 10) ∧ 
  abs (y - x) = 8.58

theorem positive_difference_proof : solve_system := 
  sorry

end positive_difference_proof_l119_119868


namespace vanya_faster_speed_l119_119522

theorem vanya_faster_speed (v : ℝ) (h : v + 2 = 2.5 * v) : (v + 4) / v = 4 := by
  sorry

end vanya_faster_speed_l119_119522


namespace adam_finances_l119_119949

-- Define the initial balances
def initial_primary_balance : ℝ := 3179.37
def initial_secondary_balance : ℝ := 1254.12

-- Define the transactions
def chores_money : ℝ := 21.85
def birthday_gift : ℝ := 150.00
def gadget_cost : ℝ := 87.41
def save_percentage : ℝ := 0.15

-- Define the final amounts to prove
def available_primary_balance : ℝ := 2646.74
def final_secondary_balance : ℝ := 1404.12

theorem adam_finances :
  let primary_balance_after_chores := initial_primary_balance + chores_money in
  let primary_balance_after_gadget := primary_balance_after_chores - gadget_cost in
  let saved_amount := primary_balance_after_gadget * save_percentage in
  let final_primary_balance := primary_balance_after_gadget - saved_amount in
  let final_secondary_balance := initial_secondary_balance + birthday_gift in
  final_primary_balance = available_primary_balance ∧ final_secondary_balance = final_secondary_balance :=
by
  have h1 : primary_balance_after_chores = initial_primary_balance + chores_money, by sorry
  have h2 : primary_balance_after_gadget = primary_balance_after_chores - gadget_cost, by sorry
  have h3 : saved_amount = primary_balance_after_gadget * save_percentage, by sorry
  have h4 : final_primary_balance = primary_balance_after_gadget - saved_amount, by sorry
  have h5 : final_secondary_balance = initial_secondary_balance + birthday_gift, by sorry
  have h6 : final_primary_balance = available_primary_balance, by sorry
  have h7 : final_secondary_balance = final_secondary_balance, by sorry
  exact ⟨h6, h7⟩

end adam_finances_l119_119949


namespace common_ratio_of_infinite_geometric_series_l119_119653

theorem common_ratio_of_infinite_geometric_series 
  (a b : ℚ) 
  (h1 : a = 8 / 10) 
  (h2 : b = -6 / 15) 
  (h3 : b = a * r) : 
  r = -1 / 2 :=
by
  -- The proof goes here
  sorry

end common_ratio_of_infinite_geometric_series_l119_119653


namespace prism_surface_area_volume_l119_119206

noncomputable def side_length (r : ℝ) : ℝ :=
  r * Real.tan (Real.pi / 8)

noncomputable def height (r : ℝ) : ℝ :=
  2 * r

noncomputable def surface_area_prism (r : ℝ) (a : ℝ) (h : ℝ) : ℝ :=
  let A_base := 2 * (1 + Real.sqrt 2) * a^2
  let A_lateral := 8 * a * h
  2 * A_base + A_lateral

noncomputable def volume_prism (A_base : ℝ) (h : ℝ) : ℝ :=
  A_base * h

theorem prism_surface_area_volume (r : ℝ) (h : ℝ) :
  let a := side_length r
  let A_base := 2 * (1 + Real.sqrt 2) * a^2
  surface_area_prism r a h = 109.007 ∧ volume_prism A_base h = 121.334 :=
by
  sorry

end prism_surface_area_volume_l119_119206


namespace probability_of_9_heads_in_12_flips_l119_119067

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l119_119067


namespace vanya_faster_speed_l119_119518

theorem vanya_faster_speed (v : ℝ) (h : v + 2 = 2.5 * v) : (v + 4) / v = 4 := by
  sorry

end vanya_faster_speed_l119_119518


namespace range_of_x_satisfying_inequality_l119_119234

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def derivative_condition (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 2 * f x + x * f' x < 2

def satisfies_inequality (f : ℝ → ℝ) (x : ℝ) : Prop :=
  x^2 * f x - 4 * f 2 < x^2 - 4

theorem range_of_x_satisfying_inequality (f f' : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_derivative : derivative_condition f f') :
  ∀ x : ℝ, satisfies_inequality f x ↔ x ∈ set.union set.Iio (-2) (set.Ioi 2) :=
sorry

end range_of_x_satisfying_inequality_l119_119234


namespace experiment_b_is_classical_probability_model_l119_119888

/-
Given a classical probability model definition and the experiment B,
prove that experiment B is a classical probability model.
-/

def classical_probability_model (experiment : Type) [Fintype experiment] (p : ProbabilityMeasure experiment) : Prop :=
  ∀ e : experiment, p e = 1 / Fintype.card experiment

def white_ball := Unit
def black_ball := Unit

def experiment_b : Type := Σ (b : Bool), if b then white_ball else black_ball

def prob_experiment_b : ProbabilityMeasure experiment_b :=
  ⟨λ e, 1 / 4, sorry⟩

theorem experiment_b_is_classical_probability_model :
  classical_probability_model experiment_b prob_experiment_b :=
sorry

end experiment_b_is_classical_probability_model_l119_119888


namespace probability_of_9_heads_in_12_flips_l119_119051

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l119_119051


namespace probability_heads_exactly_9_of_12_l119_119014

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l119_119014


namespace vanya_faster_by_4_l119_119537

variable (v : ℝ) -- initial speed in m/s
variable (school_distance : ℝ) -- distance to school in meters
variable (time_to_school : ℝ) -- time to school in seconds
variable (increased_speed : ℝ → ℝ) -- increased speed function

-- Conditions
def initial_speed : (v > 0) := by sorry
def increased_speed_by_2 {v : ℝ} : (v + 2 > 0) := by sorry
def faster_by_2_5 {v : ℝ} : (time_to_school = (school_distance / v)) → (time_to_school / 2.5 = (school_distance / (v + 2))) := by sorry

-- Main statement to prove
theorem vanya_faster_by_4 (h1 : initial_speed v) (h2 : increased_speed_by_2 v) (h3 : faster_by_2_5 v time_to_school school_distance) :
    (time_to_school / 4 = (school_distance / (v + 4))) := by
  sorry

end vanya_faster_by_4_l119_119537


namespace inverse_matrices_l119_119321

variable (a b : ℚ)

def mat1 : Matrix (Fin 2) (Fin 2) ℚ := ![![4, b], ![-7, 10]]
def mat2 : Matrix (Fin 2) (Fin 2) ℚ := ![![10, a], ![7, 4]]
def I : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![0, 1]]

theorem inverse_matrices (h : mat1 ⬝ mat2 = I) : (a, b) = (39 / 7, -39 / 7) :=
  sorry

end inverse_matrices_l119_119321


namespace necessary_but_not_sufficient_l119_119451

def p (x : ℝ) : Prop := x ^ 2 = 3 * x + 4
def q (x : ℝ) : Prop := x = Real.sqrt (3 * x + 4)

theorem necessary_but_not_sufficient (x : ℝ) : (p x → q x) ∧ ¬ (q x → p x) := by
  sorry

end necessary_but_not_sufficient_l119_119451


namespace number_of_ways_l119_119665

theorem number_of_ways : 
  ∃ (a_1 a_2 a_3 : ℕ), 
    a_1 ∈ {n | 1 ≤ n ∧ n ≤ 14} ∧ 
    a_2 ∈ {n | 1 ≤ n ∧ n ≤ 14} ∧ 
    a_3 ∈ {n | 1 ≤ n ∧ n ≤ 14} ∧ 
    a_1 < a_2 ∧ 
    a_2 < a_3 ∧ 
    a_2 - a_1 ≥ 4 ∧ 
    a_3 - a_2 ≥ 4 ∧ 
    (finset.univ.filter (λ (a_1 a_2 a_3 : ℕ), 
      a_1 ∈ finset.range 15 ∧
      a_2 ∈ finset.range 15 ∧
      a_3 ∈ finset.range 15 ∧
      a_1 < a_2 ∧ 
      a_2 < a_3 ∧ 
      a_2 - a_1 ≥ 4 ∧ 
      a_3 - a_2 ≥ 4)).card = 56 := 
sorry

end number_of_ways_l119_119665


namespace annual_interest_payment_l119_119972

def principal : ℝ := 10000
def quarterly_rate : ℝ := 0.05

theorem annual_interest_payment :
  (principal * quarterly_rate * 4) = 2000 :=
by sorry

end annual_interest_payment_l119_119972


namespace igor_number_is_five_l119_119950

-- Define the initial lineup
def initial_lineup := [2, 9, 11, 10, 6, 8, 3, 5, 4, 1, 7]

-- Define the condition for running away
def can_run_away (lineup : List Int) (index : Nat) : Prop :=
  (index > 0 ∧ lineup.get! index < lineup.get! (index - 1)) ∨
  (index < lineup.length - 1 ∧ lineup.get! index < lineup.get! (index + 1))

-- Define the function that simulates running away after one command
def run_away (lineup : List Int) : List Int :=
  lineup.enum.filter (λ (idx_val : Nat × Int), ¬ can_run_away lineup idx_val.fst).map Prod.snd

-- Define the function that simulates the commands until only n players are left
def simulate_until (lineup : List Int) (n : Nat) : List Int :=
  if lineup.length ≤ n then lineup else simulate_until (run_away lineup) n

-- Define the main theorem to prove
theorem igor_number_is_five : ∃ lineup : List Int, 
  initial_lineup = lineup ∧
  (∃ idx, lineup.get! idx = 5 ∧ simulate_until lineup 3 = [9, 11, 10]) :=
sorry

end igor_number_is_five_l119_119950


namespace find_r_l119_119719

noncomputable def value_of_r : ℝ := 
  let r := 16 in
  if h : r > 0 ∧ (∀ x y : ℝ, 2*x + 2*y = r → x^2 + y^2 = 2*r) then 
    r 
  else 
    0

theorem find_r (r : ℝ) 
  (hr : r > 0) 
  (ht : ∀ x y : ℝ, 2*x + 2*y = r → x^2 + y^2 = 2*r) : 
  r = 16 :=
by {
  sorry
}

end find_r_l119_119719


namespace rationalize_sqrt_l119_119811

theorem rationalize_sqrt (h : Real.sqrt 35 ≠ 0) : 35 / Real.sqrt 35 = Real.sqrt 35 := 
by 
sorry

end rationalize_sqrt_l119_119811


namespace alt_fib_factorial_seq_last_two_digits_eq_85_l119_119979

noncomputable def alt_fib_factorial_seq_last_two_digits : ℕ :=
  let f0 := 1   -- 0!
  let f1 := 1   -- 1!
  let f2 := 2   -- 2!
  let f3 := 6   -- 3!
  let f5 := 120 -- 5! (last two digits 20)
  (f0 - f1 + f1 - f2 + f3 - (f5 % 100)) % 100

theorem alt_fib_factorial_seq_last_two_digits_eq_85 :
  alt_fib_factorial_seq_last_two_digits = 85 :=
by 
  sorry

end alt_fib_factorial_seq_last_two_digits_eq_85_l119_119979


namespace min_sum_of_factors_l119_119848

theorem min_sum_of_factors (a b c : ℕ) (h1 : a * b * c = 2310) : a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l119_119848


namespace leaves_blew_away_correct_l119_119789

-- Definitions based on conditions
def original_leaves : ℕ := 356
def leaves_left : ℕ := 112
def leaves_blew_away : ℕ := original_leaves - leaves_left

-- Theorem statement based on the question and correct answer
theorem leaves_blew_away_correct : leaves_blew_away = 244 := by {
  -- Proof goes here (omitted for now)
  sorry
}

end leaves_blew_away_correct_l119_119789


namespace proof_problem_l119_119724

noncomputable def x : ℂ := (1 + complex.i * sqrt(3)) / 2
noncomputable def y : ℂ := (1 - complex.i * sqrt(3)) / 2

theorem proof_problem : x^6 + y^6 = 2 := by
  sorry

end proof_problem_l119_119724


namespace goal_amount_is_correct_l119_119482

def earnings_three_families : ℕ := 3 * 10
def earnings_fifteen_families : ℕ := 15 * 5
def total_earned : ℕ := earnings_three_families + earnings_fifteen_families
def goal_amount : ℕ := total_earned + 45

theorem goal_amount_is_correct : goal_amount = 150 :=
by
  -- We are aware of the proof steps but they are not required here
  sorry

end goal_amount_is_correct_l119_119482


namespace perimeter_complex_figure_l119_119875

theorem perimeter_complex_figure (A B C D E F G H J : Point)
    (AB AC AD AE BC CD DE EF FG GH HJ JG GA : ℝ) 
    (is_equilateral_ABC : Equilateral A B C)
    (is_equilateral_ADE : Equilateral A D E)
    (is_equilateral_EFG : Equilateral E F G)
    (is_equilateral_GHJ : Equilateral G H J)
    (midpoint_D : Midpoint A D C)
    (midpoint_G : Midpoint A G E)
    (midpoint_H : Midpoint E H G)
    (AB_length : AB = 6) :
    perimeter [A, B, C, D, E, F, G, H, J] = 51 := 
by
  sorry

end perimeter_complex_figure_l119_119875


namespace last_s_replacement_is_g_l119_119571

noncomputable def triangular_number (n : ℕ) : ℕ :=
n * (n + 1) / 2

def shift_letter_mod (ch : Char) (shift : ℕ) : Char :=
let pos := ch.to_nat - 'a'.to_nat + 1
pos' := (pos + shift - 1) % 26 + 1
in Char.of_nat (pos' + 'a'.to_nat - 1)

def resulting_character (message : String) : Char :=
let s_occurrence := 11
shift := (triangular_number s_occurrence) % 26
initial_char := 's'
shift_letter_mod initial_char shift

theorem last_s_replacement_is_g :
  resulting_character "Lee's sis is a Mississippi fish, Chriss!" = 'g' :=
by
  sorry

end last_s_replacement_is_g_l119_119571


namespace intersection_A_B_l119_119324

def A : Set ℤ := {-2, -1, 1, 2}

def B : Set ℤ := {x | x^2 - x - 2 ≥ 0}

theorem intersection_A_B : (A ∩ B) = {-2, -1, 2} := by
  sorry

end intersection_A_B_l119_119324


namespace log_exp_identity_l119_119294

theorem log_exp_identity (a m n : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) (h3 : Real.log a 2 = m) (h4 : Real.log a 3 = n) : 
  a ^ (2 * m + n) = 12 := by
  sorry

end log_exp_identity_l119_119294


namespace mean_of_data_set_l119_119929

theorem mean_of_data_set : 
  let data_set := [8, 12, 10, 11, 9] in
  let n := data_set.length in
  let sum := data_set.foldl (· + ·) 0 in
  let mean := sum / n in
  mean = 10 :=
by
  let data_set := [8, 12, 10, 11, 9]
  let n := data_set.length
  let sum := data_set.foldl (· + ·) 0
  let mean := sum / n
  have h_mean : mean = 10 := sorry
  exact h_mean

end mean_of_data_set_l119_119929


namespace range_of_p_l119_119284

theorem range_of_p (a_n : ℕ → ℝ) (p : ℝ) (h1 : ∀ n : ℕ, a_n n = 2 * (n:ℝ)^2 + p * n)
  (h2 : ∀ n : ℕ, a_n n < a_n (n + 1)) : -6 < p :=
begin
  sorry,
end

end range_of_p_l119_119284


namespace probability_of_9_heads_in_12_flips_l119_119041

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l119_119041


namespace proof_problem_l119_119675

def p : Prop := ∀ x : ℝ, 2 * x^2 + 2 * x + 1 / 2 < 0
def q : Prop := ∃ x : ℝ, sin x - cos x = sqrt 2

theorem proof_problem : ¬q := 
by
  unfold q
  sorry

end proof_problem_l119_119675


namespace magazines_in_fifth_pile_l119_119552

theorem magazines_in_fifth_pile :
  ∀ (magazines total_piles : ℕ), 
  total_piles ≤ 10 → 
  magazines = 100 →
  ∃ (a b c d e : ℕ), 
  (a = 3) ∧ 
  (b = a + 1) ∧ 
  (c = b + 2) ∧ 
  (d = c + 3) ∧ 
  (e = d + 4) → 
  e = 13 :=
by
  intros magazines total_piles h_total_piles h_magazines
  use 3, 4, 6, 9, 13
  simp
  split
  { exact rfl }
  split
  { exact rfl }
  split
  { exact rfl }
  split
  { exact rfl }
  exact rfl

end magazines_in_fifth_pile_l119_119552


namespace area_of_triangle_BQW_l119_119375

theorem area_of_triangle_BQW (AZ WC AB : ℝ) (h_trap_area : ℝ) (h_eq : AZ = WC) (AZ_val : AZ = 8) (AB_val : AB = 16) (trap_area_val : h_trap_area = 160) : 
  ∃ (BQW_area: ℝ), BQW_area = 48 :=
by
  let h_2 := 2 * h_trap_area / (AZ + AB)
  let h := AZ + h_2
  let BZW_area := h_trap_area - (1 / 2) * AZ * AB
  let BQW_area := 1 / 2 * BZW_area
  have AZ_eq : AZ = 8 := AZ_val
  have AB_eq : AB = 16 := AB_val
  have trap_area_eq : h_trap_area = 160 := trap_area_val
  let h_2_val : ℝ := 10 -- Calculated from h_2 = 2 * 160 / 32
  let h_val : ℝ := AZ + h_2_val -- full height
  let BZW_area_val : ℝ := 96 -- BZW area from 160 - 64
  let BQW_area_val : ℝ := 48 -- Half of BZW
  exact ⟨48, by sorry⟩ -- To complete the theorem

end area_of_triangle_BQW_l119_119375


namespace ratio_of_DE_to_CE_l119_119363

-- Constants
variables {A B C D E : Type} [InnerProductSpace ℝ E]

-- Conditions given in the problem
def parallelogram (A B C D : E) : Prop :=
  (B - A) + (D - C) = (D - A) + (C - B) ∧
  ∃ r : ℝ, r = 1 / 2

def height (A D : E) (E : E) : Prop :=
  ∃ h : ℝ, E = A + h • (D - A)

-- Main proof goal
theorem ratio_of_DE_to_CE 
  (A B C D E : E)
  (h_parallelogram : parallelogram A B C D)
  (h_height : height A D E) :
  let DE := dist D E
  let CE := dist C E
  DE / CE = 19 / 21 :=
sorry

end ratio_of_DE_to_CE_l119_119363


namespace net_income_in_June_l119_119792

theorem net_income_in_June 
  (daily_milk_production : ℕ) 
  (price_per_gallon : ℝ) 
  (daily_expense : ℝ) 
  (days_in_month : ℕ)
  (monthly_expense : ℝ) : 
  daily_milk_production = 200 →
  price_per_gallon = 3.55 →
  daily_expense = daily_milk_production * price_per_gallon →
  days_in_month = 30 →
  monthly_expense = 3000 →
  (daily_expense * days_in_month - monthly_expense) = 18300 :=
begin
  intros h_prod h_price h_daily_inc h_days h_monthly_exp,
  sorry
end

end net_income_in_June_l119_119792


namespace pears_to_apples_l119_119217

variable (pears grapes apples : Type) 
variable (cost : pears → ℝ) (cost2 : grapes → ℝ) (cost3 : apples → ℝ)

-- Conditions from Step a)
variable (p1 p2 p3 p4 : pears)
variable (g1 g2 g3 : grapes)
variable (a1 a2 : apples)

-- Assume the cost relationships
axiom h₁ : cost p1 + cost p2 + cost p3 + cost p4 = cost2 g1 + cost2 g2 + cost2 g3
axiom h₂ : cost2 g1 + cost2 g2 + cost2 g3 + cost2 g4 = cost3 a1 + cost3 a2

-- Theorem to prove from Step c)
theorem pears_to_apples : 
  24 * (cost p1) = 12 * (cost3 a1) :=
by {
  sorry
}

end pears_to_apples_l119_119217


namespace number_of_five_digit_numbers_greater_than_20000_number_of_five_digit_numbers_with_exactly_two_adjacent_even_digits_l119_119574

-- Define the set of digits we are using
def digits : List ℕ := [0, 1, 2, 3, 4]

-- First problem: number of five-digit numbers greater than 20000
theorem number_of_five_digit_numbers_greater_than_20000 :
  (List.permutations digits).count (λ l, l.headI ≥ 2) = 72 := by
sorry

-- Second problem: number of five-digit numbers with exactly two adjacent even digits
theorem number_of_five_digit_numbers_with_exactly_two_adjacent_even_digits :
  count_spec_digits digits = 56 := by
sorry

-- Helper function for counting permutations with exactly two adjacent even digits
noncomputable def count_spec_digits (l : List ℕ) : ℕ :=
  (List.permutations l).count (λ l, (l.headI % 2 = 0 ∧ (l.tailI).headI % 2 = 0) ∨
                                  (l.init.last % 2 = 0 ∧ l.last % 2 = 0))


end number_of_five_digit_numbers_greater_than_20000_number_of_five_digit_numbers_with_exactly_two_adjacent_even_digits_l119_119574


namespace abs_equation_solution_l119_119993

theorem abs_equation_solution (x : ℝ) (h : |x - 3| = 2 * x + 4) : x = -1 / 3 :=
by
  sorry

end abs_equation_solution_l119_119993


namespace probability_exactly_9_heads_l119_119121

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l119_119121


namespace counties_no_rain_l119_119734

theorem counties_no_rain 
  (P_A : ℝ) (P_B : ℝ) (P_A_and_B : ℝ) :
  P_A = 0.7 → P_B = 0.5 → P_A_and_B = 0.4 →
  (1 - (P_A + P_B - P_A_and_B) = 0.2) :=
by intros h1 h2 h3; sorry

end counties_no_rain_l119_119734


namespace min_value_expression_l119_119449

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 * b + b^2 * c + c^2 * a = 3) :
  ∃ A : ℝ, A = 3 * Real.sqrt 2 ∧ 
  (A = (Real.sqrt (a^6 + b^4 * c^6) / b) + 
       (Real.sqrt (b^6 + c^4 * a^6) / c) + 
       (Real.sqrt (c^6 + a^4 * b^6) / a)) :=
sorry

end min_value_expression_l119_119449


namespace minimum_omega_l119_119318

theorem minimum_omega (ω : ℝ) (φ : ℝ) (hω : ω > 0) :
  (∃ n : ℤ, ω = 6 * n) → ω = 6 :=
by
  intros h
  cases h with n hn
  have h_pos : 6 * n > 0,
  { rw [hn],
    exact hω },
  have h_min : n = 1,
  { linarith },
  rw [h_min, mul_one] at hn,
  exact hn

end minimum_omega_l119_119318


namespace sasha_wins_l119_119582

theorem sasha_wins : 
  let initial_number := 2018 in
  let move_sasha (n : ℕ) := n * 10 + (some digit between 0 and 9) in
  let move_andrei (n : ℕ) := n * 100 + (some digits between 0 and 99) in
  -- conditions
  -- Sasha starts the game
  -- Andrei wins if the number is divisible by 111
  -- Sasha wins if the number reaches 2018 digits and is not divisible by 111
  ∀ (n : ℕ), 
    (n = initial_number) ∧ -- start condition
    (∀ moves : list (ℕ → ℕ), (moves.map (λ m, m n)).length = 2018) ∧ -- reach 2018 digits
    (∀ moves : list (ℕ → ℕ), (moves.enumerate.map (λ ⟨i,m⟩, if i % 2 = 0 then move_sasha else move_andrei)).length < 2018 ∧ 
      ∀ k, moves.enumerate.map (λ ⟨i,m⟩, if i % 2 = 0 then move_sasha else move_andrei) k % 111 ≠ 0) 
    → ∃ strategy, strategy_sasha_wins
  sorry

end sasha_wins_l119_119582


namespace probability_heads_in_9_of_12_flips_l119_119078

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l119_119078


namespace smallest_positive_period_of_f_max_min_f_in_interval_l119_119693

noncomputable def f (x : ℝ) : ℝ :=
  2 * sqrt 3 * sin x * cos x + 2 * cos x ^ 2 - 1

theorem smallest_positive_period_of_f : 
  ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x : ℝ, (f (x + T') = f x) → T' >= T)) ∧ T = π := 
sorry

theorem max_min_f_in_interval : 
  ∃ max_val min_val, 
    (∀ x ∈ set.Icc (-π / 6) (π / 4), f x ≤ max_val) 
    ∧ (∀ x ∈ set.Icc (-π / 6) (π / 4), min_val ≤ f x) 
    ∧ max_val = 2 
    ∧ min_val = 0 := 
sorry

end smallest_positive_period_of_f_max_min_f_in_interval_l119_119693


namespace probability_of_9_heads_in_12_l119_119133

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l119_119133


namespace ctg_prod_eq_three_l119_119299

noncomputable theory
open Real Trigonometric

variables {A B C : ℝ}

theorem ctg_prod_eq_three (h1 : A + B + C = π) (h2 : sin A - sin B = sin B - sin C) :
  (1 / tan (A / 2)) * (1 / tan (C / 2)) = 3 :=
sorry

end ctg_prod_eq_three_l119_119299


namespace vertical_asymptote_condition_l119_119664

theorem vertical_asymptote_condition (k : ℝ) :
  (g(x:ℝ) = (x^2 - 2*x + k) / (x^2 - 3*x - 10)) ∧
  (x = 5 ∨ x = -2 ∧ ¬(x = 5 ∧ x = -2)) →
  k = -15 ∨ k = -8 :=
by sorry

end vertical_asymptote_condition_l119_119664


namespace count_subsets_with_even_elements_l119_119400

theorem count_subsets_with_even_elements :
  let A := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let B := {B ⊆ A | B.card = 3 ∧ (B.filter even).card ≥ 2}
  B.card = 60 := by
sorry

end count_subsets_with_even_elements_l119_119400


namespace sqrt_2_minus_x_domain_l119_119730

theorem sqrt_2_minus_x_domain (x : ℝ) : (∃ y, y = sqrt (2 - x)) → x ≤ 2 := by
  sorry

end sqrt_2_minus_x_domain_l119_119730


namespace average_of_hidden_primes_l119_119961

theorem average_of_hidden_primes (a b c : ℕ) (ha : nat.prime a) (hb : nat.prime b) (hc : nat.prime c) 
  (h_sum : 44 + a = 59 + b ∧ 44 + a = 38 + c) : (a + b + c) / 3 = 14 := by
  sorry

end average_of_hidden_primes_l119_119961


namespace sqrt_meaningful_l119_119728

theorem sqrt_meaningful (x : ℝ) (h : 2 - x ≥ 0) : x ≤ 2 :=
sorry

end sqrt_meaningful_l119_119728


namespace sqrt_45_f_eval_l119_119784

def f (x : ℝ) : ℝ :=
  if x.val.floor⌉ then
    x.val.floor + 7
  else
    7*x + 6

theorem sqrt_45_f_eval :
  f (real.sqrt 45) = 13 :=
by
  sorry

end sqrt_45_f_eval_l119_119784


namespace vanya_speed_l119_119508

variable (v : ℝ)

theorem vanya_speed (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end vanya_speed_l119_119508


namespace log_expression_l119_119615

open Real

theorem log_expression : 
  let lg := Real.log10 in
  lg 5 * lg 20 + (lg 2)^2 = 1 :=
  sorry

end log_expression_l119_119615


namespace trig_inequality_l119_119770

theorem trig_inequality 
  (a b α β : ℝ)
  (ha : 0 ≤ a ∧ a ≤ 1)
  (hb : 0 ≤ b ∧ b ≤ 1)
  (hα : 0 ≤ α ∧ α ≤ π / 2)
  (hβ : 0 ≤ β ∧ β ≤ π / 2)
  (h : a * b * real.cos (α - β) ≤ real.sqrt ((1 - a ^ 2) * (1 - b ^ 2))) :
  a * real.cos α + b * real.sin β ≤ 1 + a * b * real.sin (β - α) :=
sorry

end trig_inequality_l119_119770


namespace surface_area_of_inscribed_cube_l119_119308

theorem surface_area_of_inscribed_cube (V : ℝ) (h1 : V = 256 * π / 3) : 
  let R := (3 * V / (4 * π))^(1 / 3) in  -- Using the volume formula to get R
  let a := (2 * R) / sqrt 3 in         -- Deriving edge length a from R
  6 * a^2 = 128 :=                     -- Proving this is equal to 128
by {
  sorry
}

end surface_area_of_inscribed_cube_l119_119308


namespace probability_exactly_9_heads_l119_119119

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l119_119119


namespace probability_of_9_heads_in_12_flips_l119_119066

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l119_119066


namespace comb_15_6_eq_5005_perm_6_eq_720_l119_119884

open Nat

-- Prove that \frac{15!}{6!(15-6)!} = 5005
theorem comb_15_6_eq_5005 : (factorial 15) / (factorial 6 * factorial (15 - 6)) = 5005 := by
  sorry

-- Prove that the number of ways to arrange 6 items in a row is 720
theorem perm_6_eq_720 : factorial 6 = 720 := by
  sorry

end comb_15_6_eq_5005_perm_6_eq_720_l119_119884


namespace leila_spending_l119_119397

theorem leila_spending (sweater jewelry total money_left : ℕ) (h1 : sweater = 40) (h2 : sweater * 4 = total) (h3 : money_left = 20) (h4 : total - sweater - jewelry = money_left) : jewelry - sweater = 60 :=
by
  sorry

end leila_spending_l119_119397


namespace integral_evaluation_l119_119248

noncomputable def definite_integral (a b : ℝ) (f : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, f x

theorem integral_evaluation : 
  definite_integral 1 2 (fun x => 1 / x + x) = Real.log 2 + 3 / 2 :=
  sorry

end integral_evaluation_l119_119248


namespace largest_subset_count_l119_119941

def is_subset_valid (S : set ℕ) : Prop :=
  ∀ (x y : ℕ), x ∈ S → y ∈ S → (x = 4 * y ∨ y = 4 * x) → false

def largest_valid_subset (n : ℕ) :=
  {S : set ℕ | S ⊆ {1, ..., n} ∧ is_subset_valid S}

theorem largest_subset_count : ∃ S ∈ largest_valid_subset 150, S.card = 142 := sorry

end largest_subset_count_l119_119941


namespace count_double_single_numbers_l119_119902

def is_double_single_number (n : ℕ) : Prop :=
  let d1 := n / 100 in
  let d2 := (n / 10) % 10 in
  let d3 := n % 10 in
  100 ≤ n ∧ n < 1000 ∧ d1 = d2 ∧ d1 ≠ d3

theorem count_double_single_numbers : 
  ({ n | is_double_single_number n }.to_finset.card = 81) :=
by
  sorry

end count_double_single_numbers_l119_119902


namespace vanya_speed_problem_l119_119529

theorem vanya_speed_problem (v : ℝ) (hv : (v + 2) / v = 2.5) : ((v + 4) / v) = 4 :=
by
  have v_pos : v ≠ 0 := sorry  -- Prove that v is not zero
  have v_eq : v = 4 / 3 := sorry  -- Use hv to solve for v
  rw [v_eq] at v_eq,
  rw [v_eq]
  sorry

end vanya_speed_problem_l119_119529


namespace product_decimal_places_l119_119794

noncomputable def decimal_places (x : ℚ) : ℕ :=
  let as_string := x.abs.toString
  if '.' ∈ as_string then
    as_string.length - (as_string.indexOf '.' + 1)
  else 
    0

theorem product_decimal_places (A B : ℚ) (hA : decimal_places A = 2) (hB : decimal_places B = 2) : 
  decimal_places (A * B) = 4 :=
by
  sorry

end product_decimal_places_l119_119794


namespace f_equals_cos_over_3_l119_119821

variable {ℝ : Type} [Real ℝ]

noncomputable def f : ℝ → ℝ := sorry
noncomputable def T : ℝ := sorry

axiom periodic_f (x : ℝ) : f(x) = f(x - T)
axiom cos_related_f (x : ℝ) : cos x = f x - 2 * f (x - π)
axiom T_form (n : ℕ) : T = 2 * π * n

theorem f_equals_cos_over_3 (x : ℝ) (n : ℕ) : f(x) = cos x / 3 := 
by
  have hT : T = 2 * π * n := T_form n
  skip

end f_equals_cos_over_3_l119_119821


namespace adjacent_complementary_is_complementary_l119_119889

/-- Two angles are complementary if their sum is 90 degrees. -/
def complementary (α β : ℝ) : Prop :=
  α + β = 90

/-- Two angles are adjacent complementary if they are complementary and adjacent. -/
def adjacent_complementary (α β : ℝ) : Prop :=
  complementary α β ∧ α > 0 ∧ β > 0

/-- Prove that adjacent complementary angles are complementary. -/
theorem adjacent_complementary_is_complementary (α β : ℝ) : adjacent_complementary α β → complementary α β :=
by
  sorry

end adjacent_complementary_is_complementary_l119_119889


namespace trapezoid_area_l119_119978

noncomputable def greatest_integer_not_exceeding (x : ℝ) : ℝ :=
  if x < 0 then 0 else floor (x / 150)

theorem trapezoid_area (b h x : ℝ) (h_nonneg : 0 ≤ h) (h_conditions : x = 300 ∧ b = 150) :
  greatest_integer_not_exceeding (x^2) = 600 := by
  sorry

end trapezoid_area_l119_119978


namespace probability_of_9_heads_in_12_l119_119135

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l119_119135


namespace frac_pow_eq_l119_119966

theorem frac_pow_eq : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by 
  sorry

end frac_pow_eq_l119_119966


namespace min_a_value_l119_119683

theorem min_a_value 
  (a x y : ℤ) 
  (h1 : x - y^2 = a) 
  (h2 : y - x^2 = a) 
  (h3 : x ≠ y) 
  (h4 : |x| ≤ 10) : 
  a = -111 :=
sorry

end min_a_value_l119_119683


namespace increase_in_value_l119_119761

-- Define the conditions
def starting_weight : ℝ := 400
def weight_multiplier : ℝ := 1.5
def price_per_pound : ℝ := 3

-- Define new weight and values
def new_weight : ℝ := starting_weight * weight_multiplier
def value_at_starting_weight : ℝ := starting_weight * price_per_pound
def value_at_new_weight : ℝ := new_weight * price_per_pound

-- Theorem to prove
theorem increase_in_value : value_at_new_weight - value_at_starting_weight = 600 := by
  sorry

end increase_in_value_l119_119761


namespace distance_midpoint_to_point_l119_119616

noncomputable section -- Use noncomputable since we are working with real numbers

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_midpoint_to_point :
  let M := midpoint (2, -3) (8, 5)
  in distance M (3, 7) = 2 * Real.sqrt 10 :=
by
  sorry

end distance_midpoint_to_point_l119_119616


namespace probability_9_heads_12_flips_l119_119154

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l119_119154


namespace digit_57_of_one_over_seventeen_is_2_l119_119167

def decimal_rep_of_one_over_seventeen : ℕ → ℕ :=
λ n, ([0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7].cycle.take n).get n

theorem digit_57_of_one_over_seventeen_is_2 : decimal_rep_of_one_over_seventeen 57 = 2 :=
sorry

end digit_57_of_one_over_seventeen_is_2_l119_119167


namespace vanya_speed_increased_by_4_l119_119502

variable (v : ℝ)
variable (h1 : (v + 2) / v = 2.5)

theorem vanya_speed_increased_by_4 (h1 : (v + 2) / v = 2.5) : (v + 4) / v = 4 := 
sorry

end vanya_speed_increased_by_4_l119_119502


namespace frank_total_pages_read_l119_119273

-- Definitions of given conditions
def first_book_pages (pages_per_day : ℕ) (days : ℕ) := pages_per_day * days
def second_book_pages (pages_per_day : ℕ) (days : ℕ) := pages_per_day * days
def third_book_pages (pages_per_day : ℕ) (days : ℕ) := pages_per_day * days

-- Given values
def pages_first_book := first_book_pages 22 569
def pages_second_book := second_book_pages 35 315
def pages_third_book := third_book_pages 18 450

-- Total number of pages read by Frank
def total_pages := pages_first_book + pages_second_book + pages_third_book

-- Statement to prove
theorem frank_total_pages_read : total_pages = 31643 := by
  sorry

end frank_total_pages_read_l119_119273


namespace oldest_daughter_ages_l119_119581

theorem oldest_daughter_ages (a b c : ℕ) (h1 : a * b * c = 168) (h2 : ∃ n : ℕ, a + b + c = n ∧ (∀ m : ℕ, a + b + c = m → (a, b, c) = (m, b, c) ∨ (a, b, c) = (a, m, c) ∨ (a, b, m) = (a, b, c))) :
    a ≤ b ∧ b ≤ c → c ∈ {12, 14, 21} := 
by 
  sorry

end oldest_daughter_ages_l119_119581


namespace trees_survived_more_than_died_l119_119951

theorem trees_survived_more_than_died (initial_trees : ℕ) (died_trees : ℕ) (survived_trees : ℕ) :
  initial_trees = 11 → died_trees = 2 → survived_trees = initial_trees - died_trees → survived_trees - died_trees = 7 :=
by
  intros h_initial h_died h_survived
  rw [h_initial, h_died] at *
  rw h_survived
  calc
    11 - 2 - 2 = 9 - 2 : by rw Nat.sub_right_comm
           ... = 7     : by norm_num

end trees_survived_more_than_died_l119_119951


namespace PS_correct_and_int_part_pf_l119_119493

-- Define the lengths of the sides
def PQ : ℝ := 47
def QR : ℝ := 14
def RP : ℝ := 50

-- Define the term m
def m : ℝ := 8

-- Define the term n which is not divisible by the square of any prime
def n : ℝ := 47

-- Define the desired expression for PS
def PS : ℝ := m * Real.sqrt n

-- The proof problem statement
theorem PS_correct_and_int_part_pf (PS : ℝ) (hPS : PS = 8 * Real.sqrt 47) : ⌊m + Real.sqrt n⌋ = 14 :=
by
  -- sorry is used to skip the proof
  sorry

end PS_correct_and_int_part_pf_l119_119493


namespace sqrt_2_minus_x_domain_l119_119729

theorem sqrt_2_minus_x_domain (x : ℝ) : (∃ y, y = sqrt (2 - x)) → x ≤ 2 := by
  sorry

end sqrt_2_minus_x_domain_l119_119729


namespace certain_number_proof_l119_119347

theorem certain_number_proof (h1 : 213 * 16 = 3408) : 
  ∃ x : ℝ, 0.016 * x = 0.03408 ∧ x = 2.13 := 
by
  use 2.13
  split
  · have : 213 * 16 = 3408 := h1
    norm_num
  · norm_num
  
-- Proof is skipped with sorry
sorry

end certain_number_proof_l119_119347


namespace pythagorean_consecutive_numbers_unique_l119_119325

theorem pythagorean_consecutive_numbers_unique :
  ∀ (x : ℕ), (x + 2) * (x + 2) = (x + 1) * (x + 1) + x * x → x = 3 :=
by
  sorry 

end pythagorean_consecutive_numbers_unique_l119_119325


namespace smaller_square_perimeter_l119_119931

theorem smaller_square_perimeter (s : ℕ) (h1 : 4 * s = 144) : 
  let smaller_s := s / 3 
  let smaller_perimeter := 4 * smaller_s 
  smaller_perimeter = 48 :=
by
  let smaller_s := s / 3
  let smaller_perimeter := 4 * smaller_s 
  sorry

end smaller_square_perimeter_l119_119931


namespace number_of_bad_carrots_l119_119561

-- Definitions for conditions
def olivia_picked : ℕ := 20
def mother_picked : ℕ := 14
def good_carrots : ℕ := 19

-- Sum of total carrots picked
def total_carrots : ℕ := olivia_picked + mother_picked

-- Theorem stating the number of bad carrots
theorem number_of_bad_carrots : total_carrots - good_carrots = 15 :=
by
  sorry

end number_of_bad_carrots_l119_119561


namespace weeks_in_semester_l119_119959

-- Define the conditions and the question as a hypothesis
def annie_club_hours : Nat := 13

theorem weeks_in_semester (w : Nat) (h : 13 * (w - 2) = 52) : w = 6 := by
  sorry

end weeks_in_semester_l119_119959


namespace probability_heads_in_nine_of_twelve_flips_l119_119083

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l119_119083


namespace odd_integers_count_between_fractions_l119_119326

theorem odd_integers_count_between_fractions :
  ∃ (count : ℕ), count = 14 ∧
  ∀ (n : ℤ), (25:ℚ)/3 < (n : ℚ) ∧ (n : ℚ) < (73 : ℚ)/2 ∧ (n % 2 = 1) :=
sorry

end odd_integers_count_between_fractions_l119_119326


namespace product_of_roots_eq_zero_l119_119489

theorem product_of_roots_eq_zero :
  let a b c : ℝ := 0, -2, 2 in
  let roots := {a, b, c} in
  ∃ a b c, (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧
            a * b * c = 0 ∧ 
            ∀ x : ℝ, x ∈ roots ↔ (x = 0 ∨ x = -2 ∨ x = 2) ∧
            (x^3 - 4 * x = 0) :=
by 
  sorry

end product_of_roots_eq_zero_l119_119489


namespace vanya_speed_l119_119505

variable (v : ℝ)

theorem vanya_speed (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end vanya_speed_l119_119505


namespace probability_heads_in_9_of_12_flips_l119_119072

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l119_119072


namespace mixed_grains_approximation_l119_119743

/--
Given:
- Total grains: 1534 stones.
- Sample size: 254 grains.
- Mixed grains in the sample: 28 grains.
Prove that the approximate amount of mixed grains in the entire batch is 169 stones.
-/
theorem mixed_grains_approximation :
  let total_grains := 1534
  let sample_size := 254
  let mixed_in_sample := 28
  total_grains * (mixed_in_sample / sample_size) ≈ 169 :=
by
  sorry

end mixed_grains_approximation_l119_119743


namespace probability_of_9_heads_in_12_flips_l119_119043

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l119_119043


namespace problem1_problem2_l119_119211

-- Definitions of the three conditions given
def condition1 (x y : Nat) : Prop := x > y
def condition2 (y z : Nat) : Prop := y > z
def condition3 (x z : Nat) : Prop := 2 * z > x

-- Problem 1: If the number of teachers is 4, prove the maximum number of female students is 6.
theorem problem1 (z : Nat) (hz : z = 4) : ∃ y : Nat, (∀ x : Nat, condition1 x y → condition2 y z → condition3 x z) ∧ y = 6 :=
by
  sorry

-- Problem 2: Prove the minimum number of people in the group is 12.
theorem problem2 : ∃ z x y : Nat, (condition1 x y ∧ condition2 y z ∧ condition3 x z ∧ z < y ∧ y < x ∧ x < 2 * z) ∧ z = 3 ∧ x = 5 ∧ y = 4 ∧ x + y + z = 12 :=
by
  sorry

end problem1_problem2_l119_119211


namespace solid_circles_2010_l119_119209

noncomputable def P (n : ℕ) : ℕ :=
-- Definition of P can be inferred based on the conditions provided

theorem solid_circles_2010 : P 2010 = 61 :=
sorry

end solid_circles_2010_l119_119209


namespace mady_total_balls_l119_119424

def convert_to_base_8 (n : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  (n / 512, (n % 512) / 64, (n % 64) / 8, n % 8)

theorem mady_total_balls (n : ℕ) : 
  n = 2023 →
  let (d3, d2, d1, d0) := convert_to_base_8 n in
  (d3 + d2 + d1 + d0) = 21 :=
by 
  intro h
  rw [h]
  let (d3, d2, d1, d0) := convert_to_base_8 2023
  have : d3 = 3 := by sorry
  have : d2 = 7 := by sorry
  have : d1 = 4 := by sorry
  have : d0 = 7 := by sorry
  calc
    d3 + d2 + d1 + d0 = 3 + 7 + 4 + 7 := by rw [‹d3 = 3›, ‹d2 = 7›, ‹d1 = 4›, ‹d0 = 7›]
    ... = 21 := by norm_num

end mady_total_balls_l119_119424


namespace min_sum_factors_l119_119852

theorem min_sum_factors (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prod : a * b * c = 2310) : a + b + c = 40 :=
sorry

end min_sum_factors_l119_119852


namespace triangle_equi_div_concurrent_l119_119245

theorem triangle_equi_div_concurrent (p : ℕ) (hp : Prime p)
    (divs : ∀ (ABC : Triangle),
    ∃ (a b c : ℕ),
    a < p ∧ b < p ∧ c < p ∧
    cevas_theorem a b c p) :
  p = 2 :=
by
  -- sorry is placed here as the actual proof is skipped
  sorry

end triangle_equi_div_concurrent_l119_119245


namespace find_angle_MAN_l119_119436

noncomputable def triangle_abc (A B C X Y M N : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited X] [Inhabited Y] [Inhabited M] [Inhabited N] :=
  ∃ (triangle_abc : Type)
  (α : triangle_abc → Prop)
  (angle_AYB : 134)
  (angle_AXC : 134)
  (point_YB : triangle_abc)
  (point_XC : triangle_abc)
  (condition_MB : Prop)
  (condition_CN : Prop),
  α triangle_abc ∧ 
  angle_AYB ∧
  angle_AXC ∧
  α point_YB ∧
  α point_XC ∧
  α condition_MB ∧
  α condition_CN

theorem find_angle_MAN (A B C X Y M N : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited X] [Inhabited Y] [Inhabited M] [Inhabited N]:
  @triangle_abc A B C X Y M N → ∃ (angle_MAN : Real), angle_MAN = 46 :=
by
  intro h
  sorry

end find_angle_MAN_l119_119436


namespace min_value_of_expr_l119_119882

-- Define the expression
def expr (x y : ℝ) : ℝ := (x * y + 1)^2 + (x - y)^2

-- Statement to prove that the minimum value of the expression is 1
theorem min_value_of_expr : ∃ x y : ℝ, expr x y = 1 ∧ ∀ a b : ℝ, expr a b ≥ 1 :=
by
  -- Here the proof would be provided, but we leave it as sorry as per instructions.
  sorry

end min_value_of_expr_l119_119882


namespace angle_AZX_constant_l119_119485

variables (A B C D : Point) (Γ : Circle) (AB CD: Line)
variable (X : Point)
variables {hc : Trapezoid ABCD}
variables {h_inscribed : Inscribed ABCD Γ}
variables {h_bases : Bases ABCD AB CD}
variables {X : Point} {hX_arc : OnArc X Γ AB (λ c, c ≠ C ∧ c ≠ D)}
variables {DX : Line} {Y : Point} {hY : Intersect DX AB Y}
variables {Z : Point} {hZ : OnSegment CX Z (λ XZ XZ, XZ / XC = AY / AB)}

theorem angle_AZX_constant (h_trapezoid : Trapezoid ABCD)
                          (h_inscribed : Inscribed ABCD Γ)
                          (h_bases : Bases ABCD AB CD)
                          (hX_arc : OnArc X Γ AB (λ c, c ≠ C ∧ c ≠ D))
                          (hY : Intersect DX AB Y)
                          (hZ : OnSegment CX Z (λ XZ XC, XZ / XC = AY / AB)) :
  ∃ α : Angle, ∀ X, ∡AZX = α :=
sorry

end angle_AZX_constant_l119_119485


namespace find_a_l119_119408

noncomputable def g (a x : ℝ) : ℝ := a * x ^ 2 - real.sqrt 3

theorem find_a (a : ℝ) (h : g a (g a (real.sqrt 3)) = -real.sqrt 3) (ha : 0 < a) : 
  a = real.sqrt 3 / 3 :=
sorry

end find_a_l119_119408


namespace find_y_l119_119720

theorem find_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 18) : y = 5 :=
sorry

end find_y_l119_119720


namespace rebecca_haircuts_l119_119455

-- Definitions based on the conditions
def charge_per_haircut : ℕ := 30
def charge_per_perm : ℕ := 40
def charge_per_dye_job : ℕ := 60
def dye_cost_per_job : ℕ := 10
def num_perms : ℕ := 1
def num_dye_jobs : ℕ := 2
def tips : ℕ := 50
def total_amount : ℕ := 310

-- Define the unknown number of haircuts scheduled
variable (H : ℕ)

-- Statement of the proof problem
theorem rebecca_haircuts :
  charge_per_haircut * H + charge_per_perm * num_perms + charge_per_dye_job * num_dye_jobs
  - dye_cost_per_job * num_dye_jobs + tips = total_amount → H = 4 :=
by
  sorry

end rebecca_haircuts_l119_119455


namespace unique_two_scoop_sundaes_l119_119957

theorem unique_two_scoop_sundaes (n : ℕ) (h : n = 8) : 
  ∃ k : ℕ, k = 2 ∧ 
  (nat.choose n k = 28) :=
by
  sorry

end unique_two_scoop_sundaes_l119_119957


namespace max_subset_cardinality_146_l119_119939

noncomputable def max_subset_cardinality : ℕ :=
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 150 ∧ (∀ m ∈ (1:150), (n = 4 * m) → false) } in
  S.card = 146

theorem max_subset_cardinality_146 :
  ∃ S : set ℕ, (∀ x ∈ S, 1 ≤ x ∧ x ≤ 150) ∧
               (∀ x y ∈ S, x ≠ y → x ≠ 4 * y ∧ y ≠ 4 * x) ∧
               S.card = 146 := 
sorry

end max_subset_cardinality_146_l119_119939


namespace vanya_speed_increased_by_4_l119_119498

variable (v : ℝ)
variable (h1 : (v + 2) / v = 2.5)

theorem vanya_speed_increased_by_4 (h1 : (v + 2) / v = 2.5) : (v + 4) / v = 4 := 
sorry

end vanya_speed_increased_by_4_l119_119498


namespace shorts_different_from_jersey_probability_l119_119395

noncomputable def probability_shorts_different_from_jersey : ℚ := 2 / 3

theorem shorts_different_from_jersey_probability :
  let S := { "black", "gold" }     -- Possible colors for shorts
  let J := { "black", "white", "gold" } -- Possible colors for jerseys
  let total_configurations := 2 * 3 -- Total number of configurations
  let non_matching_configurations := 2 + 2 -- Non-matching configurations
  let probability := non_matching_configurations / total_configurations -- Probability calculation
  probability = probability_shorts_different_from_jersey :=
by 
  -- Proof goes here
  sorry

end shorts_different_from_jersey_probability_l119_119395


namespace maximum_value_of_expression_l119_119779

noncomputable def problem_statement (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2)

theorem maximum_value_of_expression (x y z : ℝ ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 3) :
  problem_statement x y z ≤ 81 / 4 :=
sorry

end maximum_value_of_expression_l119_119779


namespace combined_tax_rate_l119_119221

theorem combined_tax_rate (M : ℝ) (h_pos : M > 0) : 
  let Mork_income := M,
      Mindy_income := 4 * M,
      Mork_tax_rate := 0.45,
      Mindy_tax_rate := 0.15,
      combined_tax := (Mork_tax_rate * Mork_income) + (Mindy_tax_rate * Mindy_income),
      combined_income := Mork_income + Mindy_income,
      combined_tax_rate := combined_tax / combined_income
  in combined_tax_rate * 100 = 21 := 
by 
  -- here goes the proof, which is currently skipped
  sorry

end combined_tax_rate_l119_119221


namespace max_value_of_f_l119_119353

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_f :
  ∀ a b : ℝ, (f (-1) a b = 0) → (f (-3) a b = 0) → (f 1 a b = 0) → (f (-5) a b = 0) →
  is_symmetric_about_line (f x a b) (-2) → ∀ x : ℝ, f x a b ≤ 16 := 
by
  intros a b h1 h2 h3 h4 h_symm x
  -- Proof steps go here
  sorry

end max_value_of_f_l119_119353


namespace socks_pairs_guarantee_l119_119912

/--
In a drawer containing the following socks:
- 120 red socks
- 100 green socks
- 80 blue socks
- 60 yellow socks
- 40 black socks

Prove that the smallest number of socks that must be selected to guarantee that the selection contains at least 15 pairs is 33.
-/
theorem socks_pairs_guarantee :
  ∀ (red green blue yellow black : ℕ), red = 120 → green = 100 → blue = 80 → yellow = 60 → black = 40 →
  (∃ n, n ≥ 33 ∧ (∀ k, k < n → (k / 2) < 15)) :=
  by intros red green blue yellow black h_red h_green h_blue h_yellow h_black
     have h_pairs := sorry -- this will contain the actual proof
     exact h_pairs

end socks_pairs_guarantee_l119_119912


namespace probability_of_9_heads_in_12_flips_l119_119050

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l119_119050


namespace periodic_sequence_a2019_l119_119483

theorem periodic_sequence_a2019 :
  (∃ (a : ℕ → ℤ),
    a 1 = 1 ∧ a 2 = 1 ∧ a 3 = -1 ∧ 
    (∀ n : ℕ, n ≥ 4 → a n = a (n-1) * a (n-3)) ∧
    a 2019 = -1) :=
sorry

end periodic_sequence_a2019_l119_119483


namespace inequality1_inequality2_l119_119780

noncomputable def convex_function (f : ℝ → ℝ) : Prop :=
∀ x y z : ℝ, x < y → y < z → (f y - f x) / (y - x) ≥ (f z - f y) / (z - y)

variables {f : ℝ → ℝ} (conv_f : convex_function f)
variables {u v : ℕ → ℝ}
variables (H1 : ∀ i : ℕ, u i ≥ u (i + 1))
variables (H2 : ∀ i : ℕ, v i ≥ v (i + 1))

def D (i : ℕ) : ℝ := (f (v i) - f (u i)) / (v i - u i)

def U (i : ℕ) : ℝ := (finset.range (i + 1)).sum u
def V (i : ℕ) : ℝ := (finset.range (i + 1)).sum v

variables (H3 : ∀ i < k, U u i < V v i) (H4 : U u k = V v k)

theorem inequality1 : 
  U u 0 * (D f 0 - D f 1) + U u 1 * (D f 1 - D f 2) + U u 2 * (D f 2 - D f 3) + ... + U u (k-1) * (D f (k-1) - D f k) + U u k * D f k
  < 
  V v 0 * (D f 0 - D f 1) + V v 1 * (D f 1 - D f 2) + V v 2 * (D f 2 - D f 3) + ... + V v (k-1) * (D f (k-1) - D f k) + V v k * D f k := 
sorry

theorem inequality2 : 
  u 0 * D f 0 + u 1 * D f 1 + ... + u k * D f k
  < 
  v 0 * D f 0 + v 1 * D f 1 + ... + v k * D f k := 
sorry

end inequality1_inequality2_l119_119780


namespace class_gpa_l119_119362

theorem class_gpa (n : ℕ) (h_n : n = 60)
  (n1 : ℕ) (h_n1 : n1 = 20) (gpa1 : ℕ) (h_gpa1 : gpa1 = 15)
  (n2 : ℕ) (h_n2 : n2 = 15) (gpa2 : ℕ) (h_gpa2 : gpa2 = 17)
  (n3 : ℕ) (h_n3 : n3 = 25) (gpa3 : ℕ) (h_gpa3 : gpa3 = 19) :
  (20 * 15 + 15 * 17 + 25 * 19 : ℕ) / 60 = 1717 / 100 := 
sorry

end class_gpa_l119_119362


namespace largest_M_bound_l119_119676

noncomputable def largest_constant_M : ℝ :=
  2019 ^ (-(1 / 2019))

theorem largest_M_bound (b : ℕ → ℝ) :
  (∀ k, 0 ≤ k ∧ k ≤ 2019 → 1 ≤ b k) ∧
  (∀ k j, 0 ≤ k ∧ k < j ∧ j ≤ 2019 → b k < b j) →
  let z : ℕ → ℂ := λ k, (roots (∑ k in finset.range 2020, (b k) * X ^ k)).nth k in 
  (1 / 2019) * ∑ k in finset.range 2019, ∥z k∥ ≥ largest_constant_M :=
sorry

end largest_M_bound_l119_119676


namespace max_area_ABCD_l119_119402

-- Define the convex quadrilateral ABCD with the given conditions
variables {A B C D : Point}
variables [metric_space Point] [normed_group Point] [normed_space ℝ Point]

-- conditions 
def is_convex_quadrilateral (A B C D : Point) : Prop := 
  ∀ P Q R, P ∈ {A, B, C, D} → Q ∈ {A, B, C, D} → R ∈ {A, B, C, D} → 
  P ≠ Q → Q ≠ R → P ≠ R → 
  (P - Q) ∉ ℝ≥0 → (Q - R) ∉ ℝ≥0 → (R - P) ∉ ℝ≥0

variables (BC : ℝ := 3) (CD : ℝ := 4)
variable (AB_eq_AD : ∥A - B∥ = ∥A - D∥)

-- Define centroid of triangles
def centroid (P Q R : Point) : Point := (P + Q + R) / 3

-- Define the condition for centroids forming an equilateral
def centroids_form_equilateral (A B C D : Point) : Prop :=
  centroid A B C ≠ centroid B C D ∧ centroid B C D ≠ centroid A C D ∧ centroid A C D ≠ centroid A B C ∧
  ∥centroid A B C - centroid B C D∥ = ∥centroid B C D - centroid A C D∥ ∧
  ∥centroid B C D - centroid A C D∥ = ∥centroid A C D - centroid A B C∥

-- The main theorem statement, saying the maximum possible area is achieved under these conditions
theorem max_area_ABCD 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : ∥B - C∥ = BC)
  (h3 : ∥C - D∥ = CD)
  (h4 : AB_eq_AD)
  (h5 : centroids_form_equilateral A B C D) :
  ∃ area : ℝ, area = 12 + 10 * real.sqrt 3 := sorry

end max_area_ABCD_l119_119402


namespace ellipse_equation_dot_product_constant_l119_119302

noncomputable def foci_of_hyperbola := (sqrt 3, 0)

def equation_of_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 2 = 1

def ellipse_formed_by_equilateral_triangle (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

theorem ellipse_equation :
  ellipse_formed_by_equilateral_triangle x y ↔
  (x^2 / 4 + y^2 = 1) := sorry

theorem dot_product_constant (m : ℝ) :
  let l := λ x k : ℝ, k * (x - 1)
  in let P := (λ k, (1 + 4 * k^2) * x^2 - 8 * k^2 * x + 4 * k^2 - 4 = 0)
  in let Q := λ k, (1 + 4 * k^2) * x^2 - 8 * k^2 * x + 4 * k^2 - 4 = 0
  in let MP := (m - x_P, -y_P)
  in let MQ := (m - x_Q, -y_Q)
  in (MP.1 * MQ.1 + MP.2 * MQ.2 = (4 * m^2 - 8 * m + 1) * k^2 + (m^2 - 4) / (1 + 4 * k^2)) ↔
  m = 17 / 8 := sorry

end ellipse_equation_dot_product_constant_l119_119302


namespace paul_initial_books_l119_119803

theorem paul_initial_books (sold_books : ℕ) (left_books : ℕ) (initial_books : ℕ) 
  (h_sold_books : sold_books = 109)
  (h_left_books : left_books = 27)
  (h_initial_books_formula : initial_books = sold_books + left_books) : 
  initial_books = 136 :=
by
  rw [h_sold_books, h_left_books] at h_initial_books_formula
  exact h_initial_books_formula

end paul_initial_books_l119_119803


namespace concurrency_of_perpendiculars_l119_119682

open EuclideanGeometry

-- Given data
variables {A B C A' B' C' : Point}
variable {l₁ l₂ l₃ : Line}
hypothesis h_triangle : acute_triangle A B C
hypothesis h_altitudes : is_altitude A' B' C' A B C
hypothesis h_l1_perp_BC' : ∃ K, K ∈ l₁ ∧ K ∈ perp_line_through A B'C' 
hypothesis h_l2_perp_CA' : ∃ L, L ∈ l₂ ∧ L ∈ perp_line_through B C'A' 
hypothesis h_l3_perp_AB' : ∃ M, M ∈ l₃ ∧ M ∈ perp_line_through C A'B' 

-- The theorem to be proven
theorem concurrency_of_perpendiculars : ∃ P, P ∈ l₁ ∧ P ∈ l₂ ∧ P ∈ l₃ :=
begin
  sorry -- The proof details will be filled in here.
end

end concurrency_of_perpendiculars_l119_119682


namespace inequality_proof_l119_119453

def omega (n : ℕ) : ℕ :=
  if n = 1 then 0 else Multiset.card (n.factors.toFinset)

noncomputable def primes_sum (N : ℕ) : ℝ :=
  ∑ q in (Finset.range (N+1)).filter (λ q, q = 1 ∨ q.prime), (1 / q).toReal

theorem inequality_proof (k N : ℕ) (hk : 0 < k) (hN : 0 < N) :
  ( (1 / N.toReal) * (∑ n in Finset.range (N+1), ((omega n)^k).toReal) )^(1 / k.toReal) <=
  k + primes_sum N :=
by
  sorry

end inequality_proof_l119_119453


namespace probability_heads_in_nine_of_twelve_flips_l119_119087

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l119_119087


namespace find_angle_C_l119_119383

theorem find_angle_C (A B C : ℝ) (h1 : |Real.cos A - (Real.sqrt 3 / 2)| + (1 - Real.tan B)^2 = 0) :
  C = 105 :=
by
  sorry

end find_angle_C_l119_119383


namespace probability_of_9_heads_in_12_flips_l119_119063

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l119_119063


namespace measure_angle_C_l119_119382

noncomputable def triangle_angles_sum (a b c : ℝ) : Prop :=
  a + b + c = 180

noncomputable def angle_B_eq_twice_angle_C (b c : ℝ) : Prop :=
  b = 2 * c

noncomputable def angle_A_eq_40 : ℝ := 40

theorem measure_angle_C :
  ∀ (B C : ℝ), triangle_angles_sum angle_A_eq_40 B C → angle_B_eq_twice_angle_C B C → C = 140 / 3 :=
by
  intros B C h1 h2
  sorry

end measure_angle_C_l119_119382


namespace max_subset_cardinality_146_l119_119937

noncomputable def max_subset_cardinality : ℕ :=
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 150 ∧ (∀ m ∈ (1:150), (n = 4 * m) → false) } in
  S.card = 146

theorem max_subset_cardinality_146 :
  ∃ S : set ℕ, (∀ x ∈ S, 1 ≤ x ∧ x ≤ 150) ∧
               (∀ x y ∈ S, x ≠ y → x ≠ 4 * y ∧ y ≠ 4 * x) ∧
               S.card = 146 := 
sorry

end max_subset_cardinality_146_l119_119937


namespace time_to_coffee_shop_is_18_l119_119330

variable (cycle_constant_pace : Prop)
variable (time_cycle_library : ℕ)
variable (distance_cycle_library : ℕ)
variable (distance_to_coffee_shop : ℕ)

theorem time_to_coffee_shop_is_18
  (h_const_pace : cycle_constant_pace)
  (h_time_library : time_cycle_library = 30)
  (h_distance_library : distance_cycle_library = 5)
  (h_distance_coffee : distance_to_coffee_shop = 3)
  : (30 / 5) * 3 = 18 :=
by
  sorry

end time_to_coffee_shop_is_18_l119_119330


namespace tan_beta_plus_pi_div_four_l119_119342

noncomputable def tan (x : ℝ) : ℝ :=
  Real.sin x / Real.cos x

theorem tan_beta_plus_pi_div_four {α β : ℝ}
  (h1 : tan (α + β) = 3 / 4)
  (h2 : tan (α - Real.pi / 4) = 1 / 2) :
  tan (β + Real.pi / 4) = 2 / 11 :=
begin
  sorry
end

end tan_beta_plus_pi_div_four_l119_119342


namespace final_number_is_1458_l119_119472

noncomputable def lastRemainingNumber (n : ℕ) : ℕ :=
  if n = 1 then 1 else
  let rec eraseStep (l : List ℕ) : List ℕ :=
    (l.enum.filter (λ p, (p.1 + 1) % 3 ≠ 1)).map Prod.snd in
  match List.iterate eraseStep ⟨List.range n, n⟩ with
  | [] => 0
  | [x] => x
  | _ => 0

theorem final_number_is_1458 : lastRemainingNumber 2002 = 1458 := 
  sorry

end final_number_is_1458_l119_119472


namespace total_volume_of_removed_tetrahedra_l119_119232

noncomputable def volume_removed (x : ℝ) (edge : ℝ) :=
  (8 * (1/3) * (1/2) * (1 - x/√2)^2 * (edge - x / √2)) / 24

theorem total_volume_of_removed_tetrahedra :
  ∀ (x : ℝ),  
  let y := (1 / (√2 + 1)) * (√2 - 1) in
  (x = y) →
  volume_removed x 1 = 10 - 7 * √2 / 3 :=
begin
  intros x y,
  assume hyp : x = y,
  sorry
end

end total_volume_of_removed_tetrahedra_l119_119232


namespace wade_tips_l119_119881

def tips_per_customer := 2.00
def customers_friday := 28
def customers_saturday := 3 * customers_friday
def customers_sunday := 36

noncomputable def total_tips := 
  (customers_friday * tips_per_customer) + 
  (customers_saturday * tips_per_customer) + 
  (customers_sunday * tips_per_customer)

theorem wade_tips : total_tips = 296.00 := 
  by sorry

end wade_tips_l119_119881


namespace num_shoes_sold_sab_dane_sold_6_pairs_of_shoes_l119_119458

theorem num_shoes_sold (price_shoes : ℕ) (num_shirts : ℕ) (price_shirts : ℕ) (total_earn_per_person : ℕ) : ℕ :=
  let total_earnings_shirts := num_shirts * price_shirts
  let total_earnings := total_earn_per_person * 2
  let earnings_from_shoes := total_earnings - total_earnings_shirts
  let num_shoes_sold := earnings_from_shoes / price_shoes
  num_shoes_sold

theorem sab_dane_sold_6_pairs_of_shoes :
  num_shoes_sold 3 18 2 27 = 6 :=
by
  sorry

end num_shoes_sold_sab_dane_sold_6_pairs_of_shoes_l119_119458


namespace sum_of_first_50_digits_of_fraction_l119_119886

theorem sum_of_first_50_digits_of_fraction (h : mod_periodic 2222 5 (0, 0, 0, 4, 5)) : 
    (List.sum $ List.take 50 (List.cycle [0,0,0,4,5])) = 90 := by
  sorry

end sum_of_first_50_digits_of_fraction_l119_119886


namespace team_X_finishes_with_more_points_l119_119649

open Probability

theorem team_X_finishes_with_more_points (X Y : ℕ) 
  (h_conditions : 
    8 = 7 + 1 ∧
    ∀ (x y : ℕ), x ≠ y → (P(team_X_wins (x, y)) = 0.5)) : 
  Probability.event (team_X_points > team_Y_points) = 610 / 1024 :=
sorry

end team_X_finishes_with_more_points_l119_119649


namespace sequence_of_constants_exists_l119_119816

open MeasureTheory

theorem sequence_of_constants_exists 
  (seq : ℕ → ℝ) 
  (ξ : ℕ → ℝ → ℝ)
  [is_probability_space (ℝ → ℝ)] :
  ∃ (ε : ℕ → ℝ), (∀ n, ε n > 0) ∧ Filter.Tendsto (λ n, ε n * ξ n) Filter.at_top (𝓝 0) := by
  sorry

end sequence_of_constants_exists_l119_119816


namespace imo_shl_1995_p36_l119_119671

theorem imo_shl_1995_p36 
  {A B C A1 A2 B1 B2 C1 C2 : Type*} 
  [acute_triangle : ∀ (X : triangle), acute X]
  (exists_A1_A2_on_BC : ∃ A1 A2 : point, lies_on_segment A1 A2 B C ∧ between A2 A1 C)
  (exists_B1_B2_on_CA : ∃ B1 B2 : point, lies_on_segment B1 B2 C A ∧ between B2 B1 A)
  (exists_C1_C2_on_AB : ∃ C1 C2 : point, lies_on_segment C1 C2 A B ∧ between C2 C1 B)
  (angle_conditions : ∀ (X Y Z W : point), 
    ∠X Y Z = ∠X Z Y = ∠W X Y = ∠W Y X = ∠Y Z W = ∠Y W Z) :

  ∃ (O : circle), 
    (lies_on_circle A1 O) ∧ 
    (lies_on_circle A2 O) ∧ 
    (lies_on_circle B1 O) ∧ 
    (lies_on_circle B2 O) ∧ 
    (lies_on_circle C1 O) ∧ 
    (lies_on_circle C2 O) := 
sorry

end imo_shl_1995_p36_l119_119671


namespace no_such_b_exists_l119_119992

theorem no_such_b_exists (b : ℝ) (hb : 0 < b) :
  ¬(∃ k : ℝ, 0 < k ∧ ∀ n : ℕ, 0 < n → (n - k ≤ (⌊b * n⌋ : ℤ) ∧ (⌊b * n⌋ : ℤ) < n)) :=
by
  sorry

end no_such_b_exists_l119_119992


namespace probability_exactly_9_heads_l119_119123

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l119_119123


namespace sufficient_but_not_necessary_l119_119188

theorem sufficient_but_not_necessary (a b : ℝ) : (a > abs b) → (a^2 > b^2) ∧ ¬ (a^2 > b^2 → a > abs b) :=
by
  intro h
  split
  -- Sufficient part
  {
    exact lt_trans (abs_nonneg b) h,
  }
  -- Not necessary part
  {
    -- Let's assume (a^2 > b^2) holds and derive a contradiction on (a > abs b)
    {
      sorry
    }
  }

end sufficient_but_not_necessary_l119_119188


namespace student_enrollment_l119_119361

variable (T TG B E G : ℕ)

theorem student_enrollment (hT : T = 32) (hB : B = 12) (hTG : TG = 22) (h_all : T = E + G + B) (hG : G = TG - B) :
  E = 10 :=
by
  have G_val : G = 10 := by
    rw [hB, hTG]
    exact Nat.sub_self 12
  have E_val : E = 10 := by
    rw [hT, h_all, G_val, hB]
    exact Nat.sub_self 22
  exact E_val

end student_enrollment_l119_119361


namespace probability_heads_9_of_12_flips_l119_119005

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l119_119005


namespace total_donuts_needed_l119_119605

theorem total_donuts_needed :
  (initial_friends : ℕ) (additional_friends : ℕ) (donuts_per_friend : ℕ) (extra_donuts_per_friend : ℕ) 
  (donuts_for_Andrew : ℕ) (total_friends : ℕ) 
  (h1 : initial_friends = 2)
  (h2 : additional_friends = 2)
  (h3 : total_friends = initial_friends + additional_friends)
  (h4 : donuts_per_friend = 3)
  (h5 : extra_donuts_per_friend = 1)
  (h6 : donuts_for_Andrew = donuts_per_friend + extra_donuts_per_friend)
  :
  let initial_donuts := total_friends * donuts_per_friend,
      extra_donuts := total_friends * extra_donuts_per_friend,
      total_donuts_for_friends := initial_donuts + extra_donuts,
      total_donuts := total_donuts_for_friends + donuts_for_Andrew 
  in total_donuts = 20 := by
  sorry

end total_donuts_needed_l119_119605


namespace incorrect_equation_l119_119953

theorem incorrect_equation (a b c : ℝ) : 
  (log 3 = 2 * a - b) → 
  (log 5 = a + c) → 
  (log 8 = 3 - 3 * a - 3 * c) → 
  (log 9 = 4 * a - 2 * b) → 
  (log 15 ≠ 3 * a - b + c + 1) :=
by
  intros h1 h2 h3 h4
  sorry

end incorrect_equation_l119_119953


namespace number_is_20164_l119_119200

theorem number_is_20164 :
  ∃ n : ℕ, (nat.sqrt n) ^ 2 = n ∧ n.digits = [2, 2, 2, 1, 0] ∧ 10000 <= n ∧ n <= 99999 := 
sorry

end number_is_20164_l119_119200


namespace surface_area_of_large_cube_l119_119911

theorem surface_area_of_large_cube (l w h : ℕ) (cube_side : ℕ) 
  (volume_cuboid : ℕ := l * w * h) 
  (n_cubes := volume_cuboid / (cube_side ^ 3))
  (side_length_large_cube : ℕ := cube_side * (n_cubes^(1/3 : ℕ))) 
  (surface_area_large_cube : ℕ := 6 * (side_length_large_cube ^ 2)) :
  l = 25 → w = 10 → h = 4 → cube_side = 1 → surface_area_large_cube = 600 :=
by
  intros hl hw hh hcs
  subst hl
  subst hw
  subst hh
  subst hcs
  sorry

end surface_area_of_large_cube_l119_119911


namespace area_of_circumcircle_l119_119666

-- Given triangle ABC with side lengths a, b, and c:
variables {A B C : ℝ} {a b c : ℝ}

-- Define the conditions a = 1 and 2 * cos C + c = 2 * b
def condition_1 : Prop := a = 1
def condition_2 : Prop := 2 * Real.cos C + c = 2 * b

-- The theorem to be proved
theorem area_of_circumcircle (h1 : condition_1) (h2 : condition_2) :
  let R := a / (2 * Real.sin A) in
  let S := Real.pi * R ^ 2 in
  S = Real.pi / 3 :=
by
  sorry

end area_of_circumcircle_l119_119666


namespace prove_b_value_l119_119718

theorem prove_b_value (b : ℚ) (h : b + b / 4 = 10 / 4) : b = 2 :=
sorry

end prove_b_value_l119_119718


namespace last_letter_of_100th_permutation_is_B_l119_119835

theorem last_letter_of_100th_permutation_is_B :
  (permutations_of ["B", "H", "M", "S", "Z"]).nth 99).last = "B" :=
by
  sorry

end last_letter_of_100th_permutation_is_B_l119_119835


namespace fraction_eq_four_l119_119717

theorem fraction_eq_four (a b : ℝ) (h1 : a * b ≠ 0) (h2 : 3 * b = 2 * a) : 
  (2 * a + b) / b = 4 := 
by 
  sorry

end fraction_eq_four_l119_119717


namespace functional_eq_proof_l119_119255

noncomputable def f : ℝ → ℝ := λ x, 1/2 - x

theorem functional_eq_proof : ∀ x y : ℝ, f (x - f y) = 1 - x - y :=
by
  intro x y
  simp [f]
  sorry

end functional_eq_proof_l119_119255


namespace inequality_solution_l119_119237

theorem inequality_solution (x : ℝ) :
  x ∉ {0, 1} ∧ (3 * x - 8) * (x - 4) / (x * (x - 1)) ≥ 0 ↔ 
  x ∈ (Set.Iio 0) ∪ (Set.Icc 1 (8 / 3)) ∪ (Set.Ici 4) :=
sorry

end inequality_solution_l119_119237


namespace solve_system_l119_119818

theorem solve_system : ∃ (x y : ℚ), 4 * x - 3 * y = -2 ∧ 8 * x + 5 * y = 7 ∧ x = 1 / 4 ∧ y = 1 :=
by
  sorry

end solve_system_l119_119818


namespace number_of_lines_l119_119320

-- Definition of Line
structure Line where
  k : ℝ
  b : ℝ
  eq_line : ∀ x y : ℝ, y = k * (x - 2) + 3 → Line

-- Definition of area of triangle
def area_triangle (A B : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.1 * B.2 - A.2 * B.1)

-- Points A and B
def point_A (k : ℝ) : ℝ × ℝ := (2 - 3 / k, 0)
def point_B (k : ℝ) : ℝ × ℝ := (0, 3 - 2 * k)

-- Area condition
def area_condition (k : ℝ) : Prop :=
  area_triangle (point_A k) (point_B k) = 12

-- Main theorem
theorem number_of_lines : 
  (∃ k1 k2 k3 : ℝ,
  area_condition k1 ∧ area_condition k2 ∧ area_condition k3 ∧
  k1 ≠ k2 ∧ k1 ≠ k3 ∧ k2 ≠ k3) :=
sorry

end number_of_lines_l119_119320


namespace shelf_life_at_33_degrees_l119_119864

noncomputable def e : ℝ := Real.exp 1 -- Define the base of natural logarithm 'e'.

variables (k b : ℝ) -- Constants in the exponential function.
variables (shelf_life : ℝ → ℝ) -- Define the shelf life as a function.

-- Define conditions.
def condition1 : Prop := ∀ x, shelf_life x = e^(k * x + b)
def condition2 : Prop := shelf_life 0 = 192
def condition3 : Prop := shelf_life 22 = 48

-- Prove that the shelf life at 33°C is 24 hours.
theorem shelf_life_at_33_degrees (h1 : condition1) (h2 : condition2) (h3 : condition3) : shelf_life 33 = 24 := by
  sorry

end shelf_life_at_33_degrees_l119_119864


namespace number_of_diagonals_in_nine_sided_polygon_l119_119584

theorem number_of_diagonals_in_nine_sided_polygon:
  let n := 9 in 
  (n * (n - 3)) / 2 = 27 := 
by
  dsimp
  norm_num

end number_of_diagonals_in_nine_sided_polygon_l119_119584


namespace extreme_values_pos_extreme_values_neg_range_of_m_l119_119314

-- Given conditions in the problem
def f (x : ℝ) (a : ℝ) : ℝ := a * (log x) - 2 * a * x + 3
def f' (x : ℝ) (a : ℝ) : ℝ := a * (1 - 2 * x) / x
def g (x : ℝ) (a m : ℝ) : ℝ := (1 / 3) * x^3 + x^2 * (f' x a + m)

-- Given constraints
axiom a_ne_zero (a : ℝ) : a ≠ 0
axiom tangent_slope (a : ℝ) : f' 2 a = 3 / 2
axiom g_not_monotonic (m : ℝ) (a : ℝ) : ∃ x : ℝ, 1 < x ∧ x < 3 ∧ (g x a m ≠ g (x+1) a m)
axiom f_not_less (m a : ℝ) : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f x a ≥ (2 / 3) * x^3 - 2 * m

-- Proofs to show ranges and extreme values
theorem extreme_values_pos (a : ℝ) (h : a > 0) : ∀ x : ℝ, f x a ≤ -a * log 2 - a + 3 := sorry
theorem extreme_values_neg (a : ℝ) (h : a < 0) : ∀ x : ℝ, f x a ≥ -a * log 2 - a + 3 := sorry
theorem range_of_m (m : ℝ) (a : ℝ) : m ∈ set.Ioo (-10 / 3) (-2) ∧ m ≥ -13 / 6 := sorry

end extreme_values_pos_extreme_values_neg_range_of_m_l119_119314


namespace proof_problem_l119_119441

structure Rectangle :=
  (A B C D : EuclideanSpace ℝ (fin 2))
  (rect_cond : A.y = D.y ∧ A.x = B.x ∧ B.y = C.y ∧ D.x = C.x)

variable (R : Rectangle)

noncomputable def K_point (R : Rectangle) : EuclideanSpace ℝ (fin 2) :=
  let p := R.B - R.A
  let q := R.C - R.A
  let mag_q := (∑ i, q i * q i).sqrt
  R.A + p * (mag_q / p.x)

noncomputable def M_point (R : Rectangle) (K : EuclideanSpace ℝ (fin 2)) : EuclideanSpace ℝ (fin 2) :=
  let q := R.C - K
  let mag_q := (∑ i, q i * q i).sqrt
  K + q.unitVector

theorem proof_problem (R : Rectangle) (K : EuclideanSpace ℝ (fin 2))
  (M : EuclideanSpace ℝ (fin 2)) :
  let AK :=  ((K - R.A).x ^ 2 + (K - R.A).y ^ 2).sqrt
  let BM := ((M - R.B).x ^ 2 + (M - R.B).y ^ 2).sqrt
  let CM := ((R.C - M).x ^ 2 + (R.C - M).y ^ 2).sqrt
  CK = (R.C - K).magnitude
  by {
    have AK_BM_CM : AK + BM = CM := sorry,
    exact AK_BM_CM
  }

end proof_problem_l119_119441


namespace jills_basket_full_weight_l119_119758

theorem jills_basket_full_weight :
  ∀ (apples_in_jacks_basket : ℕ) (jacks_basket_capacity : ℕ)
  (jills_basket_capacity : ℕ) (apple_weight : ℕ),
  apples_in_jacks_basket = 12 →
  jills_basket_capacity = 2 * apples_in_jacks_basket →
  apple_weight = 150 →
  jills_basket_capacity * apple_weight = 3600 := by
  intros apples_in_jacks_basket jacks_basket_capacity jills_basket_capacity apple_weight
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end jills_basket_full_weight_l119_119758


namespace digit_57_of_one_over_seventeen_is_2_l119_119166

def decimal_rep_of_one_over_seventeen : ℕ → ℕ :=
λ n, ([0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7].cycle.take n).get n

theorem digit_57_of_one_over_seventeen_is_2 : decimal_rep_of_one_over_seventeen 57 = 2 :=
sorry

end digit_57_of_one_over_seventeen_is_2_l119_119166


namespace monomial_coefficient_and_degree_l119_119827

noncomputable def monomial_with_coefficient_and_degree (a b : ℝ) : ℝ × ℕ :=
  let monomial := - (2 * a^3 * b^4) / 7
  (-(2 / 7), 3 + 4)

/-- The coefficient and degree of the monomial - 2 * a^3 * b^4 / 7 are - 2 / 7 and 7 respectively -/
theorem monomial_coefficient_and_degree (a b : ℝ) :
  monomial_with_coefficient_and_degree a b = (-(2 / 7), 7) :=
begin
  sorry
end

end monomial_coefficient_and_degree_l119_119827


namespace probability_heads_in_nine_of_twelve_flips_l119_119082

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l119_119082


namespace probability_heads_in_9_of_12_flips_l119_119069

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l119_119069


namespace vanya_faster_speed_l119_119517

theorem vanya_faster_speed (v : ℝ) (h : v + 2 = 2.5 * v) : (v + 4) / v = 4 := by
  sorry

end vanya_faster_speed_l119_119517


namespace nelly_payment_is_correct_l119_119795

-- Given definitions and conditions
def joes_bid : ℕ := 160000
def additional_amount : ℕ := 2000

-- Nelly's total payment
def nellys_payment : ℕ := (3 * joes_bid) + additional_amount

-- The proof statement we need to prove that Nelly's payment equals 482000 dollars
theorem nelly_payment_is_correct : nellys_payment = 482000 :=
by
  -- This is a placeholder for the actual proof.
  -- You can fill in the formal proof here.
  sorry

end nelly_payment_is_correct_l119_119795


namespace correct_option_l119_119802

-- Define the operations as functions to be used in the Lean statement.
def optA : ℕ := 3 + 5 * 7 + 9
def optB : ℕ := 3 + 5 + 7 * 9
def optC : ℕ := 3 * 5 * 7 - 9
def optD : ℕ := 3 * 5 * 7 + 9
def optE : ℕ := 3 * 5 + 7 * 9

-- The theorem to prove that the correct option is (E).
theorem correct_option : optE = 78 ∧ optA ≠ 78 ∧ optB ≠ 78 ∧ optC ≠ 78 ∧ optD ≠ 78 := by {
  sorry
}

end correct_option_l119_119802


namespace probability_9_heads_12_flips_l119_119157

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l119_119157


namespace probability_exactly_9_heads_in_12_flips_l119_119097

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l119_119097


namespace prob_even_sum_l119_119388

noncomputable theory

def spinnerA := {1, 5, 6}
def spinnerB := {2, 2, 5}
def spinnerC := {1, 3, 4}

def is_even (n : Int) : Bool := n % 2 = 0

def count_odd (s : Set Int) : Int :=
  s.toList.count (λ x => ¬is_even x)

def prod_prob (spinners : List (Set Int)) (f : List Int → Prop) : ℚ :=
  let n := spinners.map (λ s => s.size).prod
  let match_count :=
    spinners.toList.prod ([0, 1, 2]).count (λ ls => f ls)
  match_count / n

theorem prob_even_sum :
  prod_prob [spinnerA, spinnerB, spinnerC] (λ s => is_even (s.sum)) = 2 / 9 :=
  sorry

end prob_even_sum_l119_119388


namespace modulo_remainder_one_l119_119775

-- Let p be a prime number and let c and d be integers such that c ≡ d⁻¹ (mod p) and d ≠ 0
variables (p : ℕ) [hp : Fact (Nat.Prime p)] (c d : ℤ)
hypotheses (h1 : c ≡ d⁻¹ [ZMOD p]) (h2 : d ≠ 0)

-- Prove that the remainder when cd is divided by p is 1
theorem modulo_remainder_one : (c * d) % p = 1 :=
by sorry

end modulo_remainder_one_l119_119775


namespace parabola_focus_directrix_distance_l119_119839

theorem parabola_focus_directrix_distance (p : ℝ) (x y : ℝ) (M : ℝ × ℝ) (hM : M = (1, 2)) 
  (h_eq : y = 2 * p * x^2) :
  p = 1 → x = 1 → y = 2 →  1 / (4 * (y / (2 * x^2))) = (1 / 4) :=
by {
  intros hp hx hy,
  rw [hx, hy, ←h_eq, hM] at *,
  sorry,
}

end parabola_focus_directrix_distance_l119_119839


namespace factorize_quadratic_l119_119990

theorem factorize_quadratic : ∀ x : ℝ, x^2 - 7*x + 10 = (x - 2)*(x - 5) :=
by
  sorry

end factorize_quadratic_l119_119990


namespace probability_of_9_heads_in_12_flips_l119_119059

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l119_119059


namespace value_of_x_l119_119336

theorem value_of_x (x : ℝ) (h : (x / 5 / 3) = (5 / (x / 3))) : x = 15 ∨ x = -15 := 
by sorry

end value_of_x_l119_119336


namespace time_to_empty_cistern_with_leaks_l119_119597

noncomputable def cistern_filling_time := 10
noncomputable def additional_time_first_leak := 2
noncomputable def additional_time_second_leak := 4
noncomputable def additional_time_third_leak := 6

theorem time_to_empty_cistern_with_leaks : 
  let R := 1 / cistern_filling_time,
      R1 := 1 / (cistern_filling_time + additional_time_first_leak),
      R2 := 1 / (cistern_filling_time + additional_time_second_leak),
      R3 := 1 / (cistern_filling_time + additional_time_third_leak),
      leak_rate := (R - R1) + (R - R2) + (R - R3),
      time_to_empty := 1 / leak_rate in
  abs (time_to_empty - 12.09) < 0.01 :=
by 
  sorry

end time_to_empty_cistern_with_leaks_l119_119597


namespace total_outfits_l119_119456

theorem total_outfits 
  (Trousers : ℕ)
  (Shirts : ℕ)
  (Jackets : ℕ)
  (Shoes : ℕ)
  (hTrousers : Trousers = 5)
  (hShirts : Shirts = 8)
  (hJackets : Jackets = 4)
  (hShoes : Shoes = 2) 
  : Trousers * Shirts * Jackets * Shoes = 320 := 
by
  -- Use the given hypotheses to conclude the total number of outfits.
  rw [hTrousers, hShirts, hJackets, hShoes]
  -- The calculation 5 * 8 * 4 * 2 = 320 can now be explicitly stated.
  norm_num
  -- Proof completed
  sorry  -- skipping the proof as per instructions

end total_outfits_l119_119456


namespace probability_9_heads_12_flips_l119_119152

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l119_119152


namespace ice_cost_l119_119623

def people : Nat := 15
def ice_needed_per_person : Nat := 2
def pack_size : Nat := 10
def cost_per_pack : Nat := 3

theorem ice_cost : 
  let total_ice_needed := people * ice_needed_per_person
  let number_of_packs := total_ice_needed / pack_size
  total_ice_needed = 30 ∧ number_of_packs = 3 ∧ number_of_packs * cost_per_pack = 9 :=
by
  let total_ice_needed := people * ice_needed_per_person
  let number_of_packs := total_ice_needed / pack_size
  have h1 : total_ice_needed = 30 := by sorry
  have h2 : number_of_packs = 3 := by sorry
  have h3 : number_of_packs * cost_per_pack = 9 := by sorry
  exact And.intro h1 (And.intro h2 h3)

end ice_cost_l119_119623


namespace triangle_BC_length_l119_119749

open_locale real

/-- In a triangle ABC with sides AB = 2 and AC = 3, where the median from A to BC has the same length
    as BC and the perimeter of the triangle is 10, the length of BC is 5. --/
theorem triangle_BC_length (BC : ℝ) (AB : ℝ) (AC : ℝ) (median_condition : ℝ) 
  (perimeter_condition : ℝ) (hAB : AB = 2) (hAC : AC = 3) 
  (hmedian : median_condition = BC) (hperimeter : AB + AC + BC = 10) : 
  BC = 5 := 
by
  sorry

end triangle_BC_length_l119_119749


namespace smallest_period_pi_sin_double_alpha_l119_119695

-- Definitions used directly in Lean 4 statement
def f (x : Real) : Real := (sqrt 3) * sin x * cos x + (cos x) ^ 2

-- Prove that the smallest positive period of the function f(x) is π
theorem smallest_period_pi : 
  ∃ (T : Real), T > 0 ∧ ∀ x : Real, f(x + T) = f(x) ∧ T = π :=
sorry

-- Given -π/2 < α < 0 and f(α) = 5/6, prove that sin 2α = (√3 - 2√2) / 6
theorem sin_double_alpha 
  (α : Real) (h : -π/2 < α ∧ α < 0) (h1 : f(α) = 5/6) :
  sin (2 * α) = (sqrt 3 - 2 * sqrt 2) / 6 :=
sorry

end smallest_period_pi_sin_double_alpha_l119_119695


namespace cookies_last_days_l119_119385

variable (c1 c2 t : ℕ)

/-- Jackson's oldest son gets 4 cookies after school each day, and his youngest son gets 2 cookies. 
There are 54 cookies in the box, so the number of days the box will last is 9. -/
theorem cookies_last_days (h1 : c1 = 4) (h2 : c2 = 2) (h3 : t = 54) : 
  t / (c1 + c2) = 9 := by
  sorry

end cookies_last_days_l119_119385


namespace distinct_values_of_z_l119_119310

theorem distinct_values_of_z : ∀ (x y z : ℤ), 
  (10 ≤ x ∧ x ≤ 99) ∧
  (10 ≤ y ∧ y ≤ 99) ∧
  (∃ (a b : ℤ), (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ x = 10 * a + b ∧ y = 10 * b + a) ∧
  (z = |x - y|) →
  (∃ (n : ℕ), n = 9).

end distinct_values_of_z_l119_119310


namespace probability_heads_exactly_9_of_12_l119_119023

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l119_119023


namespace sum_x_y_m_l119_119778

theorem sum_x_y_m (a b x y m : ℕ) (ha : a - b = 3) (hx : x = 10 * a + b) (hy : y = 10 * b + a) (hxy : x^2 - y^2 = m^2) : x + y + m = 178 := sorry

end sum_x_y_m_l119_119778


namespace log_c_eq_pi_l119_119907

noncomputable def log_c (d c : ℝ) := Real.logBase c d

theorem log_c_eq_pi (c d : ℝ) (hc : c > 0) (hd : d > 0) :
    ∃ (r C : ℝ), r = Real.logBase 5 (c^3) ∧ C = Real.logBase 5 (d^6) ∧ C = 2 * Real.pi * r → log_c d c = Real.pi :=
by
  sorry

end log_c_eq_pi_l119_119907


namespace avg_speed_back_home_l119_119228

-- Convert the given minutes to hours
def minutes_to_hours (m : ℕ) : ℝ := m / 60

-- Given conditions
def avg_speed_to_work : ℝ := 75  -- km/h
def time_to_work : ℝ := minutes_to_hours 210
def total_round_trip_time : ℝ := 6  -- hours

-- Distance travelled to work
def distance_to_work : ℝ := avg_speed_to_work * time_to_work

-- Time taken for the return trip
def time_back_home : ℝ := total_round_trip_time - time_to_work

-- The statement to prove
theorem avg_speed_back_home : (distance_to_work / time_back_home) = 105 := by
  sorry

end avg_speed_back_home_l119_119228


namespace initial_distance_between_A_and_B_l119_119194

theorem initial_distance_between_A_and_B
    (speed_A : ℕ)
    (speed_B : ℕ)
    (time : ℕ)
    (distance_A : ℕ := speed_A * time)
    (distance_B : ℕ := speed_B * time) :
    speed_A = 12 →
    speed_B = 13 →
    time = 1 →
    distance_A + distance_B = 25 :=
by
    intros hA hB ht
    rw [hA, hB, ht]
    simp
    sorry

end initial_distance_between_A_and_B_l119_119194


namespace time_after_1550_minutes_l119_119885

/-- Given that the start time is midnight on January 1, 2011,
    and 1550 minutes have elapsed since the start time,
    along with the condition that there are 60 minutes in an hour
    and 24 hours in a day, prove that the time is January 2 at 1:50 AM. -/
theorem time_after_1550_minutes :
  (start_time : Nat) (start_date : Nat) (minutes_elapsed : Nat) (hours_in_day : Nat) (minutes_in_hour : Nat) 
  (h1 : start_time = 0)
  (h2 : start_date = 1)
  (h3 : minutes_elapsed = 1550)
  (h4 : minutes_in_hour = 60)
  (h5 : hours_in_day = 24) :
  (final_date : Nat) (final_hour : Nat) (final_minute : Nat) →
  final_date = 2 ∧ final_hour = 1 ∧ final_minute = 50 := by
  sorry

end time_after_1550_minutes_l119_119885


namespace calculate_expression_l119_119619

theorem calculate_expression :
  (4 - Real.sqrt 3) ^ 0
  - 3 * Real.tan (Float.pi / 3)
  - (-1 / 2)⁻¹
  + Real.sqrt 12
  = 3 - Real.sqrt 3 :=
by 
  have h1 : (4 - Real.sqrt 3) ^ 0 = 1 := by norm_num
  have h2 : Real.tan (Float.pi / 3) = Real.sqrt 3 := by apply Real.tan_pi_div_three
  have h3 : (-1 / 2 : ℝ)⁻¹ = -2 := by norm_num
  have h4 : Real.sqrt 12 = 2 * Real.sqrt 3 := by norm_num
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end calculate_expression_l119_119619


namespace fourth_root_of_num_l119_119634

-- Define the number
def num : ℕ := 52200625

-- Define the proposition
theorem fourth_root_of_num : Real.sqrt.num.root 4 = 51 :=
by
  sorry -- Proof is not required

end fourth_root_of_num_l119_119634


namespace probability_heads_in_nine_of_twelve_flips_l119_119093

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l119_119093


namespace line_equation_through_point_with_slope_fraction_l119_119996

theorem line_equation_through_point_with_slope_fraction 
    (A : ℝ × ℝ) (k : ℝ) (line_eq : ℝ → ℝ → Prop)
    (hA : A = (1, 3))
    (line_slope : ℝ → ℝ) 
    (hl : ∀ x, line_eq x (line_slope x)) :
    k = -4 * (1 / 3) →
    (line_eq = (λ x y, 4 * x + 3 * y - 13 = 0)) :=
by
  sorry

end line_equation_through_point_with_slope_fraction_l119_119996


namespace probability_of_9_heads_in_12_flips_l119_119053

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l119_119053


namespace solve_for_cubic_l119_119721

theorem solve_for_cubic (x y : ℝ) (h₁ : x * (x + y) = 49) (h₂: y * (x + y) = 63) : (x + y)^3 = 448 * Real.sqrt 7 := 
sorry

end solve_for_cubic_l119_119721


namespace fraction_of_pizza_covered_by_pepperoni_l119_119564

theorem fraction_of_pizza_covered_by_pepperoni :
  ∀ (d_pizza d_pepperoni : ℝ) (n_pepperoni : ℕ) (overlap_fraction : ℝ),
  d_pizza = 16 ∧ d_pepperoni = d_pizza / 8 ∧ n_pepperoni = 32 ∧ overlap_fraction = 0.25 →
  (π * d_pepperoni^2 / 4 * (1 - overlap_fraction) * n_pepperoni) / (π * (d_pizza / 2)^2) = 3 / 8 :=
by
  intro d_pizza d_pepperoni n_pepperoni overlap_fraction
  intro h
  sorry

end fraction_of_pizza_covered_by_pepperoni_l119_119564


namespace max_sequence_terms_sum_l119_119915

theorem max_sequence_terms_sum (s : ℕ → ℝ) :
  (∀ n, 2017 ≤ n -> ∑ k in finset.range 2017, s (n + k) < 0) →
  (∀ n, 2018 ≤ n -> ∑ k in finset.range 2018, s (n + k) > 0) →
  ∃ M m, (M = 4033) ∧ (m = 4032) ∧ (M + m = 8065) :=
sorry

end max_sequence_terms_sum_l119_119915


namespace f_analytical_expression_f_monotonic_f_extreme_values_l119_119696

noncomputable def f (x : ℝ) (n : ℝ) : ℝ := x - (1 / x^n)

axiom f_spec : f 2 n = 3/2

theorem f_analytical_expression :
  f = λ x, x - (1 / x) :=
by
  sorry

theorem f_monotonic (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (h : x1 < x2) :
  f x1 1 < f x2 1 :=
by
  sorry

theorem f_extreme_values :
  f 2 1 = 3/2 ∧ f 5 1 = 24/5 :=
by
  sorry

end f_analytical_expression_f_monotonic_f_extreme_values_l119_119696


namespace vanya_faster_speed_l119_119514

def vanya_speed (v : ℝ) : Prop :=
  (v + 2) / v = 2.5

theorem vanya_faster_speed (v : ℝ) (h : vanya_speed v) : (v + 4) / v = 4 :=
by
  sorry

end vanya_faster_speed_l119_119514


namespace min_value_expression_l119_119448

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 * b + b^2 * c + c^2 * a = 3) :
  ∃ A : ℝ, A = 3 * Real.sqrt 2 ∧ 
  (A = (Real.sqrt (a^6 + b^4 * c^6) / b) + 
       (Real.sqrt (b^6 + c^4 * a^6) / c) + 
       (Real.sqrt (c^6 + a^4 * b^6) / a)) :=
sorry

end min_value_expression_l119_119448


namespace goldfish_initial_count_l119_119437

theorem goldfish_initial_count (catsfish : ℕ) (fish_left : ℕ) (fish_disappeared : ℕ) (goldfish_initial : ℕ) :
  catsfish = 12 →
  fish_left = 15 →
  fish_disappeared = 4 →
  goldfish_initial = (fish_left + fish_disappeared) - catsfish →
  goldfish_initial = 7 :=
by
  intros h1 h2 h3 h4
  rw [h2, h3, h1] at h4
  exact h4

end goldfish_initial_count_l119_119437


namespace max_value_of_f_l119_119354

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_f :
  ∀ a b : ℝ, (f (-1) a b = 0) → (f (-3) a b = 0) → (f 1 a b = 0) → (f (-5) a b = 0) →
  is_symmetric_about_line (f x a b) (-2) → ∀ x : ℝ, f x a b ≤ 16 := 
by
  intros a b h1 h2 h3 h4 h_symm x
  -- Proof steps go here
  sorry

end max_value_of_f_l119_119354


namespace other_root_of_quadratic_l119_119351

theorem other_root_of_quadratic (a : ℝ) (h : (-1)^2 + 3*(-1) + a = 0) : 
  ∃ b : ℝ, b = -2 ∧ b^2 + 3*b + a = 0 :=
by
  use -2
  split
  . rfl
  . sorry

end other_root_of_quadratic_l119_119351


namespace admissible_set_gcd_l119_119969

def is_admissible (A : Set ℤ) : Prop :=
  ∀ (x y ∈ A) (k : ℤ), x^2 + k * x * y + y^2 ∈ A

theorem admissible_set_gcd {m n : ℤ} (hm : m ≠ 0) (hn : n ≠ 0) :
  (∀ A : Set ℤ, is_admissible A → m ∈ A → n ∈ A → A = Set.univ) ↔ Int.gcd m n = 1 :=
by sorry

end admissible_set_gcd_l119_119969


namespace zero_in_interval_x5_y8_l119_119799

theorem zero_in_interval_x5_y8
    (x y : ℝ)
    (h1 : x^5 < y^8)
    (h2 : y^8 < y^3)
    (h3 : y^3 < x^6)
    (h4 : x < 0)
    (h5 : 0 < y)
    (h6 : y < 1) :
    0 ∈ set.Ioo (x^5) (y^8) :=
by
  sorry

end zero_in_interval_x5_y8_l119_119799


namespace solve_x4_minus_81_eq_0_l119_119462

theorem solve_x4_minus_81_eq_0 (x : ℂ) : (x^4 - 81 = 0) ↔ (x = 3 ∨ x = -3 ∨ x = 3 * complex.I ∨ x = -3 * complex.I) :=
by
  sorry

end solve_x4_minus_81_eq_0_l119_119462


namespace cannot_be_six_l119_119373

theorem cannot_be_six (n r : ℕ) (h_n : n = 6) : 3 * n ≠ 4 * r :=
by
  sorry

end cannot_be_six_l119_119373


namespace distance_A_B_l119_119674

noncomputable def distance_3d (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

theorem distance_A_B :
  distance_3d 4 1 9 10 (-1) 6 = 7 :=
by
  sorry

end distance_A_B_l119_119674


namespace youseff_blocks_l119_119557

theorem youseff_blocks : 
  ∀ (x : ℕ), 
    (∀ (walk_per_block_min : ℕ), walk_per_block_min = 1) →
    (∀ (bike_per_block_sec : ℕ), bike_per_block_sec = 20) →
    (60 * (x * walk_per_block_min - x * (bike_per_block_sec / 60)) = 12 * 60) →
    x = 18 :=
by
  intro x
  intro walk_per_block_min hw
  intro bike_per_block_sec hb
  intro h_time
  have h_walk_min: walk_per_block_min = 1 := hw
  have h_bike_min: (bike_per_block_sec : ℕ) = 20 := hb
  sorry

end youseff_blocks_l119_119557


namespace distance_MK_l119_119365

noncomputable theory

open Real

variables (A B C D M N K : ℝ)
variables (c : ℝ) (y z : ℝ)

-- Hypotenuse of the right-angled triangle ABC
def hypotenuse (A B C : ℝ) : ℝ := c

-- Altitude from C to AB equals the diameter of the circle
def altitude (C D : ℝ) : ℝ := sqrt (y * z)

-- Distance from A to D and from D to B
def AD_DB_sum (y z : ℝ) : ℝ := y + z = c

-- Distance from A to K and B to K
def tangents_distance (K M N : ℝ) : ℝ := abs K = c / 3

theorem distance_MK (A B C D M N K : ℝ) (c y z : ℝ) :
  right_triangle A B C ∧ hypotenuse A B C = c ∧ altitude C D = sqrt (y * z) ∧ AD_DB_sum y z = c →
  tangents_distance K M N := by
  sorry

end distance_MK_l119_119365


namespace probability_neither_orange_nor_white_l119_119181

/-- Define the problem conditions. -/
def num_orange_balls : ℕ := 8
def num_black_balls : ℕ := 7
def num_white_balls : ℕ := 6

/-- Define the total number of balls. -/
def total_balls : ℕ := num_orange_balls + num_black_balls + num_white_balls

/-- Define the probability of picking a black ball (neither orange nor white). -/
noncomputable def probability_black_ball : ℚ := num_black_balls / total_balls

/-- The main statement to be proved: The probability is 1/3. -/
theorem probability_neither_orange_nor_white : probability_black_ball = 1 / 3 :=
sorry

end probability_neither_orange_nor_white_l119_119181


namespace tan_eq_of_angle_l119_119998

theorem tan_eq_of_angle (m : ℤ) (h₁ : -90 < m ∧ m < 90) (h₂ : tan (m * real.pi / 180) = tan (1230 * real.pi / 180)) : 
  m = -30 := 
sorry

end tan_eq_of_angle_l119_119998


namespace convex_pentagon_arithmetic_angles_l119_119477

theorem convex_pentagon_arithmetic_angles :
  ∃ (sequences_count : ℕ), sequences_count = 2 ∧
  ( ∀ (x d : ℕ), 5 * x + 10 * d = 540 → x + 2 * d = 108 → (x + 4 * d < 120) ∧ (x > 0) ∧ (d % 2 = 0) ∧ (d < 6) →
  ∃ (angles : list ℕ), angles = [x, x + d, x + 2 * d, x + 3 * d, x + 4 * d] ∧ (∀ (angle : ℕ), angle ∈ angles → angle < 120) )

end convex_pentagon_arithmetic_angles_l119_119477


namespace probability_of_9_heads_in_12_l119_119137

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l119_119137


namespace infinite_inequality_l119_119413

theorem infinite_inequality {a : ℕ → ℝ}
  (h_pos : ∀ n, 0 < a n) :
  ∃ᶠ n in at_top, 1 + a n > a (n-1) * (2 : ℝ)^(1 / (5 : ℝ)) :=
begin
  sorry
end

end infinite_inequality_l119_119413


namespace average_ratio_l119_119943

theorem average_ratio (x : Fin 50 → ℝ) (A : ℝ) (hA : A = (∑ i, x i) / 50) :
    (let A' := (∑ i, x i + 100) / 51 in A' / A = (50 + 100 / A) / 51) := 
by
  sorry

end average_ratio_l119_119943


namespace probability_of_9_heads_in_12_flips_l119_119054

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l119_119054


namespace emily_final_score_l119_119560

theorem emily_final_score :
  16 + 33 - 48 = 1 :=
by
  -- proof skipped
  sorry

end emily_final_score_l119_119560


namespace probability_exactly_9_heads_in_12_flips_l119_119107

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l119_119107


namespace ellipse_dot_product_range_l119_119311

noncomputable def ellipse_foci_dot_product_range : Set ℝ :=
  let F1 := (Real.sqrt 3, 0)
  let F2 := (-Real.sqrt 3, 0)
  let P θ := (2 * Real.cos θ, Real.sin θ)
  Set.Icc (-2.0) 1.0

theorem ellipse_dot_product_range :
  ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi →
  (let P := (2 * Real.cos θ, Real.sin θ)
   let PF1 := (F1.1 - P.1, F1.2 - P.2)
   let PF2 := (F2.1 - P.1, F2.2 - P.2)
   (PF1.1 * PF2.1 + PF1.2 * PF2.2) ∈ ellipse_foci_dot_product_range) :=
by
  intros θ hθ
  let F1 := (Real.sqrt 3, 0)
  let F2 := (-Real.sqrt 3, 0)
  let P := (2 * Real.cos θ, Real.sin θ)
  let PF1 := (F1.1 - P.1, F1.2 - P.2)
  let PF2 := (F2.1 - P.1, F2.2 - P.2)
  sorry

end ellipse_dot_product_range_l119_119311


namespace intersection_points_count_l119_119478

def polynomial1 (x y : ℝ) := (x - y + 3) * (4 * x + y - 5) = 0
def polynomial2 (x y : ℝ) := (x + y - 3) * (3 * x - 4 * y + 8) = 0

theorem intersection_points_count : 
  (∃ p1 p2 p3 p4 : ℝ × ℝ, (polynomial1 p1.1 p1.2 ∧ polynomial2 p1.1 p1.2) ∧
                      (polynomial1 p2.1 p2.2 ∧ polynomial2 p2.1 p2.2) ∧
                      (polynomial1 p3.1 p3.2 ∧ polynomial2 p3.1 p3.2) ∧
                      (polynomial1 p4.1 p4.2 ∧ polynomial2 p4.1 p4.2) ∧
                      p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧
                      p2 ≠ p3 ∧ p2 ≠ p4 ∧
                      p3 ≠ p4) ∧
  ∀ p : ℝ × ℝ , (polynomial1 p.1 p.2 ∧ polynomial2 p.1 p.2) →
       p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4 :=
begin
  sorry
end

end intersection_points_count_l119_119478


namespace range_of_function_l119_119238

theorem range_of_function :
  ∀ (y : ℝ),
    (∃ (x : ℝ), y = x + 9 / (x + 1) ∧ x ≠ -1) ↔
    (y ∈ set.Iic (-7) ∨ y ∈ set.Ici 5) :=
by
  sorry

end range_of_function_l119_119238


namespace area_under_curve_eq_ln2_l119_119466

noncomputable def enclosed_area : ℝ :=
  ∫ x in 1..2, 1 / x

theorem area_under_curve_eq_ln2 :
  enclosed_area = Real.log 2 :=
by
  sorry

end area_under_curve_eq_ln2_l119_119466


namespace sale_price_monday_to_wednesday_sale_price_thursday_to_saturday_sale_price_super_saver_sunday_sale_price_festive_friday_selected_sale_price_festive_friday_non_selected_l119_119861

def original_price : ℝ := 150
def discount_monday_to_wednesday : ℝ := 0.20
def tax_monday_to_wednesday : ℝ := 0.05
def discount_thursday_to_saturday : ℝ := 0.15
def tax_thursday_to_saturday : ℝ := 0.04
def discount_super_saver_sunday1 : ℝ := 0.25
def discount_super_saver_sunday2 : ℝ := 0.10
def tax_super_saver_sunday : ℝ := 0.03
def discount_festive_friday : ℝ := 0.20
def tax_festive_friday : ℝ := 0.04
def additional_discount_festive_friday : ℝ := 0.05

theorem sale_price_monday_to_wednesday : (original_price * (1 - discount_monday_to_wednesday)) * (1 + tax_monday_to_wednesday) = 126 :=
by sorry

theorem sale_price_thursday_to_saturday : (original_price * (1 - discount_thursday_to_saturday)) * (1 + tax_thursday_to_saturday) = 132.60 :=
by sorry

theorem sale_price_super_saver_sunday : ((original_price * (1 - discount_super_saver_sunday1)) * (1 - discount_super_saver_sunday2)) * (1 + tax_super_saver_sunday) = 104.29 :=
by sorry

theorem sale_price_festive_friday_selected : ((original_price * (1 - discount_festive_friday)) * (1 + tax_festive_friday)) * (1 - additional_discount_festive_friday) = 118.56 :=
by sorry

theorem sale_price_festive_friday_non_selected : (original_price * (1 - discount_festive_friday)) * (1 + tax_festive_friday) = 124.80 :=
by sorry

end sale_price_monday_to_wednesday_sale_price_thursday_to_saturday_sale_price_super_saver_sunday_sale_price_festive_friday_selected_sale_price_festive_friday_non_selected_l119_119861


namespace find_n_correct_l119_119782

theorem find_n_correct (x y : ℤ) (h1 : x = 3) (h2 : y = -3) : 
  let n := x - y^(x - (y + 1)) 
  in n = 246 :=
by
  sorry

end find_n_correct_l119_119782


namespace min_sum_of_factors_of_2310_l119_119859

theorem min_sum_of_factors_of_2310 : ∃ a b c : ℕ, a * b * c = 2310 ∧ a + b + c = 52 :=
by
  sorry

end min_sum_of_factors_of_2310_l119_119859


namespace probability_exactly_9_heads_l119_119110

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l119_119110


namespace vanya_speed_problem_l119_119527

theorem vanya_speed_problem (v : ℝ) (hv : (v + 2) / v = 2.5) : ((v + 4) / v) = 4 :=
by
  have v_pos : v ≠ 0 := sorry  -- Prove that v is not zero
  have v_eq : v = 4 / 3 := sorry  -- Use hv to solve for v
  rw [v_eq] at v_eq,
  rw [v_eq]
  sorry

end vanya_speed_problem_l119_119527


namespace machine_A_produces_50_parts_in_10_minutes_l119_119788

def machine_A_produces_double_rate (rate_B rate_A : ℕ → ℚ) : Prop :=
  ∀ t, rate_A t = 2 * rate_B t

def machine_B_produces_in_time (rate_B : ℕ → ℚ) (parts : ℕ) (time : ℚ) : Prop :=
  ∀ (t : ℚ), t = time → rate_B (nat.floor t) = parts

def machine_A_constant_rate (rate_A : ℕ → ℚ) (parts_10 : ℕ) : Prop :=
  ∀ t, t = 10 → rate_A (nat.floor t) = parts_10

theorem machine_A_produces_50_parts_in_10_minutes
  (rate_B rate_A : ℕ → ℚ)
  (H1 : machine_A_produces_double_rate rate_B rate_A)
  (H2 : machine_B_produces_in_time rate_B 100 40)
  (H3 : machine_A_constant_rate rate_A 50) :
  machine_A_constant_rate rate_A 50 :=
by
  sorry

end machine_A_produces_50_parts_in_10_minutes_l119_119788


namespace probability_heads_9_of_12_flips_l119_119000

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l119_119000


namespace gardener_cabbages_increased_by_197_l119_119578

theorem gardener_cabbages_increased_by_197 (x : ℕ) (last_year_cabbages : ℕ := x^2) (increase : ℕ := 197) :
  (x + 1)^2 = x^2 + increase → (x + 1)^2 = 9801 :=
by
  intros h
  sorry

end gardener_cabbages_increased_by_197_l119_119578


namespace probability_of_9_heads_in_12_l119_119131

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l119_119131


namespace sum_le_square_l119_119844

theorem sum_le_square (m n : ℕ) (h: (m * n) % (m + n) = 0) : m + n ≤ n^2 :=
by sorry

end sum_le_square_l119_119844


namespace factorial_mod_13_l119_119662

theorem factorial_mod_13 :
  ∃ (r : ℤ), r = 10! % 13 ∧ r = 6 :=
by
  sorry

end factorial_mod_13_l119_119662


namespace tangent_line_at_origin_range_of_m_l119_119699

-- Define the function f(x) when m = 0
def f (x : ℝ) : ℝ := Math.sin x - x^3

-- Part (1): Tangent line at (0, f(0))
theorem tangent_line_at_origin (x : ℝ) (hx : x = 0) : 
    let f_x := Math.sin x - x^3 
    let slope := (deriv f) 0 
    let tangent := fun x => slope * x
    tangent x = x :=
by 
    sorry

-- Part (2): Finding range of m for nonpositivity of f(x)
def g (x : ℝ) (m : ℝ) : ℝ := Math.sin x - x^3 - m * x

theorem range_of_m (m : ℝ): (∀ x ∈ Icc (0 : ℝ) (Real.pi), g x m ≤ 0) ↔ (1 ≤ m) :=
by
    sorry

end tangent_line_at_origin_range_of_m_l119_119699


namespace proof_problem_l119_119297

variable (m n : Line)
variable (α β : Plane)

def different_lines : Prop := m ≠ n
def different_planes : Prop := α ≠ β
def perpendicular_planes : Prop := α ⊥ β
def line_perpendicular_to_plane : Prop := m ⊥ β
def line_not_in_plane : Prop := ¬ m ⊂ α
def line_parallel_to_plane : Prop := m ∥ α

theorem proof_problem
  (h1 : different_lines m n)
  (h2 : different_planes α β)
  (h3 : perpendicular_planes α β)
  (h4 : line_perpendicular_to_plane m β)
  (h5 : line_not_in_plane m α) :
  line_parallel_to_plane m α :=
sorry

end proof_problem_l119_119297


namespace probability_participation_on_both_days_l119_119266

theorem probability_participation_on_both_days :
  let students := fin 5 -> bool in
  let total_outcomes := 2^5 in
  let same_day_outcomes := 2 in
  let both_days_outcomes := total_outcomes - same_day_outcomes in
  both_days_outcomes / total_outcomes = (15 / 16 : ℚ) :=
by
  let students := fin 5 -> bool
  let total_outcomes := 2^5
  let same_day_outcomes := 2
  let both_days_outcomes := total_outcomes - same_day_outcomes
  have h1 : both_days_outcomes = 30 := by norm_num
  have h2 : total_outcomes = 32 := by norm_num
  show both_days_outcomes / total_outcomes = 15 / 16
  calc 
    30 / 32 = 15 / 16 : by norm_num
        ... = (15 / 16 : ℚ) : by norm_cast

end probability_participation_on_both_days_l119_119266


namespace strongest_erosive_power_l119_119239

-- Definition of the options
inductive Period where
  | MayToJune : Period
  | JuneToJuly : Period
  | JulyToAugust : Period
  | AugustToSeptember : Period

-- Definition of the eroding power function (stub)
def erosivePower : Period → ℕ
| Period.MayToJune => 1
| Period.JuneToJuly => 2
| Period.JulyToAugust => 3
| Period.AugustToSeptember => 1

-- Statement that July to August has the maximum erosive power
theorem strongest_erosive_power : erosivePower Period.JulyToAugust = 3 := 
by 
  sorry

end strongest_erosive_power_l119_119239


namespace probability_third_ball_white_l119_119673

-- Definitions and conditions
variable (n : ℕ) (h : n > 2)
definition bag (k : ℕ) : Type := 
  { balls // balls ∈ ({i : ℕ // i < n} → bool) ∧ 
           (∃ (whites reds : Fin n → ℕ), 
             whites = λ i => ite (i < n-k) 1 0 ∧
             reds = λ i => ite (i < k) 1 0 ∧ 
             ∀ i, balls i = carr whites reds)}

-- Theorem stating the probability
theorem probability_third_ball_white 
  (h : n > 2) :
  let bags := Fin n → bag n;
  let random_bag := bags;
  let draw_three_without_replacement (bag : bag n) : list ball := sorry
  in ∑ b in bags, (∑ e in (list.nth (draw_three_without_replacement b) 2) = white, 1) = (n-1) / (2*n)
:=
  sorry

end probability_third_ball_white_l119_119673


namespace vanya_faster_speed_l119_119520

theorem vanya_faster_speed (v : ℝ) (h : v + 2 = 2.5 * v) : (v + 4) / v = 4 := by
  sorry

end vanya_faster_speed_l119_119520


namespace soccer_team_lineups_l119_119432

noncomputable def num_starting_lineups (n k t g : ℕ) : ℕ :=
  n * (n - 1) * (Nat.choose (n - 2) k)

theorem soccer_team_lineups :
  num_starting_lineups 18 9 1 1 = 3501120 := by
    sorry

end soccer_team_lineups_l119_119432


namespace lateral_area_of_cylinder_maximize_lateral_area_cylinder_l119_119548

variable (R H x : ℝ)

def radiusOfInscribedCylinder (R H x : ℝ) : ℝ := (H - x) * R / H

def lateralAreaCylinder (R H x : ℝ) : ℝ := (2 * Real.pi * R / H) * (-x^2 + H * x)

theorem lateral_area_of_cylinder (h : 0 < x ∧ x < H) :
  lateralAreaCylinder R H x = (2 * Real.pi * R / H) * (-x^2 + H * x) :=
by sorry

theorem maximize_lateral_area_cylinder (h : 0 < x ∧ x < H) :
  (lateralAreaCylinder R H (H / 2)) = (1 / 2) * Real.pi * R * H :=
by sorry

end lateral_area_of_cylinder_maximize_lateral_area_cylinder_l119_119548


namespace sequence_general_term_sequence_sum_l119_119306

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h : ∀ n, 2 * S n = 3 * a n - 1) :
  ∀ n, a n = 3 ^ (n - 1) :=
begin
  sorry
end

theorem sequence_sum (S : ℕ → ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℝ)
  (h1 : ∀ n, 2 * S n = 3 * a n - 1)
  (h2 : ∀ n, a n = 3 ^ (n - 1))
  (h3 : ∀ n, b n = n * a n) :
  ∀ n, T n = ((2 * n - 1) * 3 ^ (n - 1) + 1) / 4 :=
begin
  sorry
end

end sequence_general_term_sequence_sum_l119_119306


namespace vanya_faster_by_4_l119_119531

variable (v : ℝ) -- initial speed in m/s
variable (school_distance : ℝ) -- distance to school in meters
variable (time_to_school : ℝ) -- time to school in seconds
variable (increased_speed : ℝ → ℝ) -- increased speed function

-- Conditions
def initial_speed : (v > 0) := by sorry
def increased_speed_by_2 {v : ℝ} : (v + 2 > 0) := by sorry
def faster_by_2_5 {v : ℝ} : (time_to_school = (school_distance / v)) → (time_to_school / 2.5 = (school_distance / (v + 2))) := by sorry

-- Main statement to prove
theorem vanya_faster_by_4 (h1 : initial_speed v) (h2 : increased_speed_by_2 v) (h3 : faster_by_2_5 v time_to_school school_distance) :
    (time_to_school / 4 = (school_distance / (v + 4))) := by
  sorry

end vanya_faster_by_4_l119_119531


namespace find_constant_l119_119652

theorem find_constant (N : ℝ) (C : ℝ) (h1 : N = 12.0) (h2 : C + 0.6667 * N = 0.75 * N) : C = 0.9996 :=
by
  sorry

end find_constant_l119_119652


namespace total_washer_dryer_cost_l119_119212

def washer_cost : ℕ := 710
def dryer_cost : ℕ := washer_cost - 220

theorem total_washer_dryer_cost :
  washer_cost + dryer_cost = 1200 :=
  by sorry

end total_washer_dryer_cost_l119_119212


namespace vanya_speed_increased_by_4_l119_119499

variable (v : ℝ)
variable (h1 : (v + 2) / v = 2.5)

theorem vanya_speed_increased_by_4 (h1 : (v + 2) / v = 2.5) : (v + 4) / v = 4 := 
sorry

end vanya_speed_increased_by_4_l119_119499


namespace smallest_possible_value_correct_l119_119445

noncomputable def smallest_possible_value (n : ℕ) : ℕ :=
if n = 1 then 2 else if n = 3 then 11 else 4 * n + 1

theorem smallest_possible_value_correct (n : ℕ) (a : ℕ → ℕ) :
  (∀ i j, 0 ≤ i ∧ i < j ∧ j ≤ n → ¬(prime (a j - a i))) ∧
  (∀ m, (∀ i j, 0 ≤ i ∧ i < j ∧ j ≤ m → ¬(prime (a j - a i))) → a m ≥ smallest_possible_value m) := 
sorry

end smallest_possible_value_correct_l119_119445


namespace quartic_polynomial_has_roots_l119_119253

theorem quartic_polynomial_has_roots :
  ∃ (p : Polynomial ℚ),
    Polynomial.monic p ∧
    (3 + Real.sqrt 5 ∈ Polynomial.roots p.map(Polynomial.C) ∧
     3 - Real.sqrt 5 ∈ Polynomial.roots p.map(Polynomial.C) ∧
     2 + Real.sqrt 7 ∈ Polynomial.roots p.map(Polynomial.C) ∧
     2 - Real.sqrt 7 ∈ Polynomial.roots p.map(Polynomial.C) ∧
      p = Polynomial.mk [(-12 : ℚ), 2, 25, -10, 1]) :=
  sorry

end quartic_polynomial_has_roots_l119_119253


namespace max_value_of_f_l119_119356

noncomputable def f (x a b : ℝ) := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_f (a b : ℝ) (h_symmetric : ∀ x : ℝ, f (-2 - x) a b = f (-2 + x) a b) :
  ∃ x : ℝ, f x a b = 16 := by
  sorry

end max_value_of_f_l119_119356


namespace find_z_l119_119654

noncomputable def solution (z : ℂ) : Prop :=
  let a := z.re
  let b := z.im in
  (a - 2)^2 + b^2 = (a + 1)^2 + (b - 1)^2 ∧
  (a + 1)^2 + (b - 1)^2 = (a - 1)^2 + (b - 2)^2 ∧
  z = complex.mk 0 1

theorem find_z : ∃ z : ℂ, solution z :=
by {
  let z := complex.mk 0 1
  use z
  -- Proof skipped with 'sorry'
  sorry
}

end find_z_l119_119654


namespace arithmetic_sequence_product_l119_119484

theorem arithmetic_sequence_product
  (a d : ℤ)
  (h1 : a + 5 * d = 17)
  (h2 : d = 2) :
  (a + 2 * d) * (a + 3 * d) = 143 :=
by
  sorry

end arithmetic_sequence_product_l119_119484


namespace number_of_ordered_pairs_l119_119862

noncomputable def geometric_seq_log_sum (b s : ℕ) (hb : 0 < b) (hs : 0 < s) : Prop :=
  let seq_log_sum := (list.range 10).sum (λ i, Real.logb 4 (b * s ^ i))
  seq_log_sum = 1003

theorem number_of_ordered_pairs :
  {p : ℕ × ℕ | let b := p.1, s := p.2 in geometric_seq_log_sum b s (nat.succ_pos b) (nat.succ_pos s)}.to_finset.card = 40 :=
sorry

end number_of_ordered_pairs_l119_119862


namespace exactly_two_visits_in_300_days_l119_119765

def visits_periodically (n : ℕ) (d : ℕ) : Prop :=
  ∀ (i : ℕ), i < n → (i + 1) % d = 0

def exactly_two_visits_in_day (day : ℕ) : Prop :=
  (day % 4 = 0 ∧ day % 6 = 0 ∧ day % 8 ≠ 0) ∨
  (day % 4 = 0 ∧ day % 8 = 0 ∧ day % 6 ≠ 0) ∨
  (day % 6 = 0 ∧ day % 8 = 0 ∧ day % 4 ≠ 0)

theorem exactly_two_visits_in_300_days :
  (∑ d in Finset.range 300, if exactly_two_visits_in_day d then 1 else 0) = 36 := sorry

end exactly_two_visits_in_300_days_l119_119765


namespace probability_heads_exactly_9_of_12_l119_119017

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l119_119017


namespace log_base_change_log_base_8_of_1_over_64_l119_119246

theorem log_base_change (a b c : ℝ) (hc : 0 < c) (hb : b ≠ 1) (ha : a > 0):
  log c a = log b a / log b c := sorry

theorem log_base_8_of_1_over_64 :
  ∃ x : ℝ, x = log 8 (1 / 64) ∧ x = -2 := 
by
  -- Define conditions
  have h1 : 8 = 2^3 := by norm_num
  have h2 : 1 / 64 = 2^(-6) := by norm_num

  -- Evaluate logs
  have h_log8_64 := log_base_change (1 / 64) 8 2 (by norm_num) (by norm_num) (by norm_num)
  rw [h1, h2] at h_log8_64

  use -2
  split
  -- Showing x = log 8 (1 / 64)
  {
    exact h_log8_64.symm
  }
  -- Showing x = -2
  {
    norm_num
  }

end log_base_change_log_base_8_of_1_over_64_l119_119246


namespace count_powers_of_3_not_powers_of_9_l119_119328

theorem count_powers_of_3_not_powers_of_9 :
  let N := 1000000
  let pow3_count := (List.range 13).filter (fun n => 3^n < N).length
  let pow9_count := (List.range 7).filter (fun k => 9^k < N).length
  pow3_count - pow9_count = 6 :=
by
  sorry

end count_powers_of_3_not_powers_of_9_l119_119328


namespace complement_of_angle_l119_119332

theorem complement_of_angle (A : ℝ) (hA : A = 42) : (90 - A) = 48 :=
by
  -- Condition: A = 42
  have h1 : A = 42 := hA
  -- Complement of A: 90 - A
  have h2 : 90 - A = 48
  -- Sorry to skip the proof
  sorry

end complement_of_angle_l119_119332


namespace find_mnc_sum_l119_119829

theorem find_mnc_sum (m n c : ℝ)
    (h1 : ∃ (x : ℝ), y = x^3 + m * x + c )
    (P_tangent : (y = 2 * x + 1) (P : x = 1, y = n)) :
    m + n + c = 5 := sorry

end find_mnc_sum_l119_119829


namespace smallest_tasty_integer_l119_119986

theorem smallest_tasty_integer : 
  ∃ k : ℤ, (∀ n : ℤ, (k ≤ n) → (∑ i in finset.Icc k n, i) = 2023 → k = -1011) :=
sorry

end smallest_tasty_integer_l119_119986


namespace sum_st_sn_eq_l119_119412

noncomputable def S (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i, 1 / (i + 1 : ℚ))

theorem sum_st_sn_eq (n : ℕ) :
    (Finset.range n).sum (λ i, (i + 1 : ℚ) * S (i + 1)) = (n * (n + 1) / 2) * (S (n + 1) - 1 / 2) := 
  sorry

end sum_st_sn_eq_l119_119412


namespace probability_heads_9_of_12_l119_119146

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l119_119146


namespace probability_heads_in_9_of_12_flips_l119_119076

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l119_119076


namespace dan_initial_money_l119_119982

def initial_amount (spent_candy : ℕ) (spent_chocolate : ℕ) (remaining : ℕ) : ℕ :=
  spent_candy + spent_chocolate + remaining

theorem dan_initial_money 
  (spent_candy : ℕ) (spent_chocolate : ℕ) (remaining : ℕ)
  (h_candy : spent_candy = 2)
  (h_chocolate : spent_chocolate = 3)
  (h_remaining : remaining = 2) :
  initial_amount spent_candy spent_chocolate remaining = 7 :=
by
  rw [h_candy, h_chocolate, h_remaining]
  unfold initial_amount
  rfl

end dan_initial_money_l119_119982


namespace largest_square_area_l119_119750

theorem largest_square_area (XY XZ YZ : ℝ)
  (h1 : XZ^2 = 2 * XY^2)
  (h2 : XY^2 + YZ^2 = XZ^2)
  (h3 : XY^2 + YZ^2 + XZ^2 = 450) :
  XZ^2 = 225 :=
by
  -- Proof skipped
  sorry

end largest_square_area_l119_119750


namespace find_min_n_l119_119283

def sequence (a : ℕ → ℝ) : Prop :=
∀ n, n ≥ 1 → (3 * a (n + 1) + a n = 4)

def initial_condition (a : ℕ → ℝ) : Prop :=
a 1 = 9

def sum_of_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n, S n = ∑ i in Finset.range n, a (i + 1)

def target_inequality (S : ℕ → ℝ) (n : ℕ) : Prop :=
|S n - n - 6| < 1 / 125

theorem find_min_n (a S : ℕ → ℝ) (h_seq : sequence a) (h_init : initial_condition a) (h_sum : sum_of_sequence S a) :
  ∃ n : ℕ, target_inequality S n ∧ (∀ m < n, ¬ target_inequality S m) :=
begin
  sorry,
end

end find_min_n_l119_119283


namespace bounded_sequence_l119_119322

def sequence_property (p : ℕ → ℕ) : Prop :=
  ∀ n ≥ 2, nat.prime (p 0) ∧ nat.prime (p 1) ∧
  ∃ k, p n = nat.factors (p (n-1) + p (n-2) + 2016) k ∧ ∀ j ≠ k, nat.factors (p (n-1) + p (n-2) + 2016) j ≤ nat.factors (p (n-1) + p (n-2) + 2016) k

theorem bounded_sequence (p : ℕ → ℕ) (h : sequence_property p) : 
  ∃ M : ℝ, ∀ n, p n ≤ M :=
sorry

end bounded_sequence_l119_119322


namespace random_events_l119_119599

-- Definitions for conditions
def Condition1 := ∃ coins: List ℕ, List.mem 10 coins ∧ List.mem 50 coins ∧ List.mem 100 coins
def Condition2 := ¬ (∃ P: ℝ, P = 90 ∧ ¬ boils_at_p P)
def Condition3 := ∃ shooter : Type, shoots_10_ring shooter
def Condition4 := ∀ dice1 dice2: ℕ, dice1 + dice2 ≤ 12

-- Assumptions
axiom coins_in_pocket : Condition1
axiom water_at_90_impossible : Condition2
axiom shooter_hits : Condition3
axiom dice_sum_not_exceed_12 : Condition4

-- Theorem stating the given conditions and proving the random events are exactly condition1 and condition3
theorem random_events : (Condition1 ∧ Condition3) ∧ ¬(Condition2 ∧ Condition4) :=
by
  split
  -- The given random events
  { split,
    apply coins_in_pocket,
    apply shooter_hits }
  -- The non-random events
  { split,
    intro h,
    exact water_at_90_impossible h,
    intro h,
    exact dice_sum_not_exceed_12 h }

sorry -- proof omitted

end random_events_l119_119599


namespace expand_polynomial_l119_119650

theorem expand_polynomial :
  (7 * x^2 + 5 * x - 3) * (3 * x^3 + 2 * x^2 - x + 4) = 21 * x^5 + 29 * x^4 - 6 * x^3 + 17 * x^2 + 23 * x - 12 :=
by
  sorry

end expand_polynomial_l119_119650


namespace probability_of_9_heads_in_12_flips_l119_119065

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l119_119065


namespace distribution_ways_5_items_3_bags_l119_119219

theorem distribution_ways_5_items_3_bags : 
  let items := 5
      bags := 3
  in distribute_identical_bags items bags = 36 :=
  sorry

end distribution_ways_5_items_3_bags_l119_119219


namespace probability_heads_in_9_of_12_flips_l119_119073

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l119_119073


namespace closest_multiple_of_15_to_500_l119_119170

/-- An integer m is a multiple of 15 if there exists an integer n such that m = 15 * n. -/
def is_multiple_of_15 (m : ℤ) : Prop := ∃ n : ℤ, m = 15 * n

theorem closest_multiple_of_15_to_500 : ∃ m : ℤ, is_multiple_of_15(m) ∧ (500 - m).abs ≤ (500 - (m + 15)).abs ∧ (500 - m).abs < (500 - (m - 15)).abs :=
sorry

end closest_multiple_of_15_to_500_l119_119170


namespace country_albums_count_l119_119174

-- Definitions based on conditions
def pop_albums : Nat := 8
def songs_per_album : Nat := 7
def total_songs : Nat := 70

-- Theorem to prove the number of country albums
theorem country_albums_count : (total_songs - pop_albums * songs_per_album) / songs_per_album = 2 := by
  sorry

end country_albums_count_l119_119174


namespace total_amount_spent_on_cookies_l119_119269

def days_in_april : ℕ := 30
def cookies_per_day : ℕ := 3
def cost_per_cookie : ℕ := 18

theorem total_amount_spent_on_cookies : days_in_april * cookies_per_day * cost_per_cookie = 1620 := by
  sorry

end total_amount_spent_on_cookies_l119_119269


namespace probability_of_9_heads_in_12_flips_l119_119046

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l119_119046


namespace vanya_faster_by_4_l119_119536

variable (v : ℝ) -- initial speed in m/s
variable (school_distance : ℝ) -- distance to school in meters
variable (time_to_school : ℝ) -- time to school in seconds
variable (increased_speed : ℝ → ℝ) -- increased speed function

-- Conditions
def initial_speed : (v > 0) := by sorry
def increased_speed_by_2 {v : ℝ} : (v + 2 > 0) := by sorry
def faster_by_2_5 {v : ℝ} : (time_to_school = (school_distance / v)) → (time_to_school / 2.5 = (school_distance / (v + 2))) := by sorry

-- Main statement to prove
theorem vanya_faster_by_4 (h1 : initial_speed v) (h2 : increased_speed_by_2 v) (h3 : faster_by_2_5 v time_to_school school_distance) :
    (time_to_school / 4 = (school_distance / (v + 4))) := by
  sorry

end vanya_faster_by_4_l119_119536


namespace max_puns_exists_max_puns_l119_119214

theorem max_puns (x y z : ℕ) (hx : x ≥ 1) (hy : y ≥ 1)
  (h : 3 * x + 4 * y + 9 * z = 108) : z ≤ 10 :=
by
  sorry

theorem exists_max_puns : ∃ x y z, x ≥ 1 ∧ y ≥ 1 ∧ 3 * x + 4 * y + 9 * z = 108 ∧ z = 10 :=
by
  use [2, 2, 10]
  split
  { exact nat.zero_lt_succ 1 }
  split
  { exact nat.zero_lt_succ 1 }
  split
  { refl }
  refl

end max_puns_exists_max_puns_l119_119214


namespace proof_problem_l119_119367

-- Given Conditions
variables (O A B C D : Type) [MetricSpace O] [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (p q : ℝ) (unit_circle : MetricSpace) (dist : O → A → ℝ) (dist_center : MetricSpace)
variables (unit_radius : dist_center.dist O A = 1)
variables (parallel_eq_dist : dist_center.dist (line_segment A B) (line_segment C D) = dist_center.dist O A)
variables (chord_length_AC : dist_center.dist A C = p)
variables (chord_length_BD : dist_center.dist B D = p)
variables (chord_length_CD : dist_center.dist C D = p)
variables (chord_length_AB : dist_center.dist A B = q)

-- Proof Statement
theorem proof_problem : q - p = 2 :=
  sorry

end proof_problem_l119_119367


namespace sin_A_in_right_triangle_l119_119369

theorem sin_A_in_right_triangle (B C A : Real) (hBC: B + C = π / 2) 
(h_sinB: Real.sin B = 3 / 5) (h_sinC: Real.sin C = 4 / 5) : 
Real.sin A = 1 := 
by 
  sorry

end sin_A_in_right_triangle_l119_119369


namespace capacity_percent_l119_119183

-- Definitions for the given conditions
def heightA := 10
def circA := 7
def heightB := 7
def circB := 10
def pi := Real.pi

def radiusA := circA / (2 * pi)
def radiusB := circB / (2 * pi)

def volumeA := pi * (radiusA ^ 2) * heightA
def volumeB := pi * (radiusB ^ 2) * heightB

-- Theorem statement
theorem capacity_percent :
  (volumeA / volumeB) * 100 = 70 := by
  -- Proof to be provided later
  sorry

end capacity_percent_l119_119183


namespace tropical_polynomial_concave_l119_119614

-- Define the tropical polynomial form
def tropical_polynomial (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  (List.range (n+1)).map (λ k, a k + k * x) |>.minimum' id

-- Prove the concavity property of tropical polynomials
theorem tropical_polynomial_concave (a : ℕ → ℝ) (n : ℕ) (x y : ℝ) :
  tropical_polynomial a n ((x + y) / 2) ≥ (tropical_polynomial a n x + tropical_polynomial a n y) / 2 :=
by
  sorry

end tropical_polynomial_concave_l119_119614


namespace find_side_lines_l119_119707

noncomputable def point := (ℝ × ℝ)
noncomputable def line := { l : ℝ × ℝ × ℝ // l.1 ≠ 0 ∨ l.2 ≠ 0 }

def point_on_line (p : point) (l : line) : Prop :=
  l.1.1 * p.1 + l.1.2 * p.2 + l.1.3 = 0

structure triangle :=
(A B C : point)

def median (t : triangle) (A_mid : point) (mid : point) (median_line : line) : Prop :=
  A_mid = t.A ∧ point_on_line A_mid median_line ∧ point_on_line mid median_line

def side_line (p1 p2 : point) : line :=
  (p1.2 - p2.2, p2.1 - p1.1, p1.1 * p2.2 - p2.1 * p1.2).toSubtype sorry

theorem find_side_lines (A B C : point)
  (A_med_AB_eq : line)
  (A_med_AC_eq : line)
  (hA : A = (1, 3))
  (hA_med_AB : A_med_AB_eq = (1, -2, 1).toSubtype sorry)
  (hA_med_AC : A_med_AC_eq = (0, 1, -1).toSubtype sorry)
  (hB : B = (5, 1))
  (hC : C = (-3, -1)) :
  side_line A B = (1, 2, -7).toSubtype sorry ∧
  side_line B C = (1, -4, -1).toSubtype sorry ∧
  side_line A C = (1, -1, 2).toSubtype sorry :=
sorry

end find_side_lines_l119_119707


namespace compute_abs_expression_l119_119977

theorem compute_abs_expression : abs(3 * real.pi - abs(real.pi - 4)) = 4 * real.pi - 4 := by
  have pi_lt_4 : real.pi < 4 := sorry
  sorry

end compute_abs_expression_l119_119977


namespace smallest_partitioned_n_l119_119261

def can_be_partitioned (n : ℕ) : Prop := 
  ∃ (triples : Fin n → (ℕ × ℕ × ℕ)), 
    (∀ i, 1 ≤ triples i).1 ∧ (triples i).1 ≤ 3 * n ∧
    (∀ i, 1 ≤ (triples i).2 ∧ 
    (triples i).2 ≤ 3 * n ∧ 
    (∀ i, 1 ≤ (triples i).2 ∧ 
    (triples i).2 ≤ 3 * n ∧ 
    (triples i).1 + (triples i).2 = 3 * (triples i).2) ∧
    (∀ i j, i ≠ j → triples i ≠ triples j) ∧
    (Finset.univ : Finset (Fin n)) = 
      Finset.univ.bUnion (λ i, 
        Finset.single (triples i).1 ∪ 
        Finset.single (triples i).2 ∪ 
        Finset.single (triples i).2)

theorem smallest_partitioned_n : ∃ n, can_be_partitioned n ∧ n = 5 :=
begin
  use 5,
  -- Adding the proof of this theorem
  sorry -- Replace with actual proof later
end

end smallest_partitioned_n_l119_119261


namespace cost_savings_difference_l119_119435

/-
Original Order Amount: $20000
Option 1: Three successive discounts of 25%, 15%, and 5%.
Option 2: Three successive discounts of 20%, 10%, and 5%, followed by a lump-sum rebate of $300.
-/

theorem cost_savings_difference : 
  let original_amount := 20000
  let option1 := original_amount * 0.75 * 0.85 * 0.95
  let option2 := (original_amount * 0.80 * 0.90 * 0.95) - 300
  option2 - option1 = 1267.5 := 
by
  let original_amount := 20000
  let option1 := original_amount * 0.75 * 0.85 * 0.95
  let option2 := (original_amount * 0.80 * 0.90 * 0.95) - 300
  have h : option2 - option1 = 1267.5 := sorry
  exact h

end cost_savings_difference_l119_119435


namespace vanya_speed_problem_l119_119530

theorem vanya_speed_problem (v : ℝ) (hv : (v + 2) / v = 2.5) : ((v + 4) / v) = 4 :=
by
  have v_pos : v ≠ 0 := sorry  -- Prove that v is not zero
  have v_eq : v = 4 / 3 := sorry  -- Use hv to solve for v
  rw [v_eq] at v_eq,
  rw [v_eq]
  sorry

end vanya_speed_problem_l119_119530


namespace correct_product_is_504_l119_119457

-- Definitions for the problem conditions
def digits_reversed (a b : ℕ) : Prop :=
  let a_rev := (a % 10) * 10 + (a / 10) in
  a_rev * b = 378

def correct_product (a b : ℕ) : ℕ :=
  a * b

-- Main theorem statement, proving the correct product
theorem correct_product_is_504 (a b : ℕ) (h : digits_reversed a b) : correct_product a b = 504 :=
sorry

end correct_product_is_504_l119_119457


namespace probability_exactly_9_heads_in_12_flips_l119_119101

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l119_119101


namespace distance_AK_l119_119486

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (0, -1)
noncomputable def C : ℝ × ℝ := (1, 0)
noncomputable def D : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)

-- Define the line equations
noncomputable def line_AB (x : ℝ) : Prop := x = 0
noncomputable def line_CD (x y : ℝ) : Prop := y = (Real.sqrt 2) / (2 - Real.sqrt 2) * (x - 1)

-- Define the intersection point K
noncomputable def K : ℝ × ℝ := (0, -(Real.sqrt 2 + 1))

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Prove the desired distance
theorem distance_AK : distance A K = Real.sqrt 2 + 1 :=
by
  -- Proof details are omitted
  sorry

end distance_AK_l119_119486


namespace pairs_of_parallel_edges_of_rectangular_prism_l119_119205

theorem pairs_of_parallel_edges_of_rectangular_prism
  (l w h : ℕ) 
  (hl : l = 2)
  (hw : w = 3)
  (hh : h = 4) :
  (number_of_pairs_of_parallel_edges l w h) = 6 := 
sorry

end pairs_of_parallel_edges_of_rectangular_prism_l119_119205


namespace probability_heads_in_nine_of_twelve_flips_l119_119091

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l119_119091


namespace tom_dollars_more_than_jerry_l119_119390

theorem tom_dollars_more_than_jerry (total_slices : ℕ)
  (jerry_slices : ℕ)
  (tom_slices : ℕ)
  (plain_cost : ℕ)
  (pineapple_additional_cost : ℕ)
  (total_cost : ℕ)
  (cost_per_slice : ℚ)
  (cost_jerry : ℚ)
  (cost_tom : ℚ)
  (jerry_ate_plain : jerry_slices = 5)
  (tom_ate_pineapple : tom_slices = 5)
  (total_slices_10 : total_slices = 10)
  (plain_cost_10 : plain_cost = 10)
  (pineapple_additional_cost_3 : pineapple_additional_cost = 3)
  (total_cost_13 : total_cost = plain_cost + pineapple_additional_cost)
  (cost_per_slice_calc : cost_per_slice = total_cost / total_slices)
  (cost_jerry_calc : cost_jerry = cost_per_slice * jerry_slices)
  (cost_tom_calc : cost_tom = cost_per_slice * tom_slices) :
  cost_tom - cost_jerry = 0 := by
  sorry

end tom_dollars_more_than_jerry_l119_119390


namespace range_of_x_l119_119681

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_monotone : ∀ x y, (0 < x → 0 < y → x ≤ y → f(x) ≤ f(y))
axiom f_functional : ∀ x y, (0 < x → 0 < y → f(x * y) = f(x) + f(y))
axiom f_value : f 3 = 1

theorem range_of_x (x : ℝ) (h1 : 0 < x) (h2 : 0 < x - 8) (h3 : f(x) + f(x - 8) ≤ 2) : 
  8 < x ∧ x ≤ 9 := 
begin
  sorry
end

end range_of_x_l119_119681


namespace max_PM_minus_PN_l119_119403

noncomputable def hyperbola : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ x^2 - y^2 / 15 = 1 }
noncomputable def circle1 : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x + 4)^2 + y^2 = 4 }
noncomputable def circle2 : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x - 4)^2 + y^2 = 1 }

theorem max_PM_minus_PN (P M N : ℝ × ℝ)
  (hP : P ∈ hyperbola) 
  (hM : M ∈ circle1) 
  (hN : N ∈ circle2) : 
  ∃ Q : ℝ, Q = |(P.fst - M.fst, P.snd - M.snd)| - |(P.fst - N.fst, P.snd - N.snd)| ∧ Q ≤ 5 :=
sorry

end max_PM_minus_PN_l119_119403


namespace total_sum_value_l119_119323

open Finset

def M : Finset ℕ := filter (λ x, 1 ≤ x ∧ x ≤ 10) (range 11)

def sum_transformed_elements (A : Finset ℕ) : ℤ :=
  A.sum (λ k, (-1) ^ k * k)

def total_sum : ℤ :=
  M.powerset.erase ∅
    .sum sum_transformed_elements

theorem total_sum_value :
  total_sum = 2560 := by {
  -- proof here
  sorry
}

end total_sum_value_l119_119323


namespace probability_9_heads_12_flips_l119_119158

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l119_119158


namespace total_T_equals_2_l119_119404

noncomputable def T : ℝ := ∑ x in {x : ℝ | 0 < x ∧ x^4 = 2^(x^2)}, x

theorem total_T_equals_2 : T = 2 := by
  sorry

end total_T_equals_2_l119_119404


namespace max_distinct_sums_l119_119410

theorem max_distinct_sums (n k : ℕ) (h1 : n > k) (h2 : k ≥ 2) (P : Finset ℕ) (hP : P.card = n) :
  let C_Q := (P.subsets k).card in
  C_Q = n.choose k :=
sorry

end max_distinct_sums_l119_119410


namespace smallest_n_for_book_price_l119_119566

theorem smallest_n_for_book_price (n x : ℕ) (h_pos : n > 0) (h_eq : 1.05 * x = 100 * n) : n = 21 :=
sorry

end smallest_n_for_book_price_l119_119566


namespace min_value_reciprocal_sum_l119_119463

theorem min_value_reciprocal_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ c d : ℝ, 0 < c ∧ 0 < d ∧ c + d = 2 → (1/c + 1/d) ≥ m := 
sorry

end min_value_reciprocal_sum_l119_119463


namespace exists_point_Q_on_circumcircle_l119_119767

open EuclideanGeometry

variable {A B C P D E F Q : Point}

theorem exists_point_Q_on_circumcircle
  (h_triangle : Triangle A B C)
  (hP : ¬Collinear A B P ∧ ¬Collinear B C P ∧ ¬Collinear C A P)
  (hD : D ∈ Line B C)
  (hE : E ∈ Line A C)
  (hF : F ∈ Line A B)
  (h_parallel_DE_CP : Parallel (Line D E) (Line P C))
  (h_parallel_DF_BP : Parallel (Line D F) (Line P B)) :
  ∃ Q, Q ∈ Circumcircle A E F ∧ Similar (Triangle B A Q) (Triangle P A C) := sorry

end exists_point_Q_on_circumcircle_l119_119767


namespace three_digit_numbers_last_three_digits_of_square_l119_119256

theorem three_digit_numbers_last_three_digits_of_square (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  (n^2 % 1000) = n ↔ n = 376 ∨ n = 625 := 
sorry

end three_digit_numbers_last_three_digits_of_square_l119_119256


namespace zero_in_interval_l119_119800

theorem zero_in_interval (x y : ℝ) (hx_lt_0 : x < 0) (hy_gt_0 : 0 < y) (hy_lt_1 : y < 1) (h : x^5 < y^8 ∧ y^8 < y^3 ∧ y^3 < x^6) : x^5 < 0 ∧ 0 < y^8 :=
by
  sorry

end zero_in_interval_l119_119800


namespace remainder_calc_l119_119169

theorem remainder_calc : 
  ∃ (remainder : ℕ), let dividend := 166 in
                      let divisor := 18 in
                      let quotient := 9 in
                      dividend = (divisor * quotient) + remainder ∧ remainder = 4 :=
by
  sorry

end remainder_calc_l119_119169


namespace part1_part2_part3_l119_119692

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 1 - x - a * x^2

theorem part1 (x : ℝ) : f x 0 ≥ 0 :=
sorry

theorem part2 {a : ℝ} (h : ∀ x ≥ 0, f x a ≥ 0) : a ≤ 1 / 2 :=
sorry

theorem part3 (x : ℝ) (hx : x > 0) : (Real.exp x - 1) * Real.log (x + 1) > x^2 :=
sorry

end part1_part2_part3_l119_119692


namespace angle_after_2_hours_40_minutes_l119_119547

-- Define the conditions of the problem
def time_in_minutes (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * 60 + minutes

def proportion_of_hour (minutes : ℕ) : ℚ :=
  minutes / 60

-- Define the provided specific values for hours and minutes
def minutes_after_2_hours_40_minutes : ℕ := time_in_minutes 2 40

-- Define the angle turned by the minute hand for given minutes
def angle_turned_by_minute_hand (minutes : ℕ) : ℤ :=
  - (minutes / 60) * 360 - (360 * proportion_of_hour (minutes % 60))

-- Lean 4 statement to prove
theorem angle_after_2_hours_40_minutes :
  angle_turned_by_minute_hand minutes_after_2_hours_40_minutes = -960 :=
by
  -- proof goes here
  sorry

end angle_after_2_hours_40_minutes_l119_119547


namespace weight_of_sparrow_l119_119745

variable (a b : ℝ)

-- Define the conditions as Lean statements
-- 1. Six sparrows and seven swallows are balanced
def balanced_initial : Prop :=
  6 * b = 7 * a

-- 2. Sparrows are heavier than swallows
def sparrows_heavier : Prop :=
  b > a

-- 3. If one sparrow and one swallow are exchanged, the balance is maintained
def balanced_after_exchange : Prop :=
  5 * b + a = 6 * a + b

-- The theorem to prove the weight of one sparrow in terms of the weight of one swallow
theorem weight_of_sparrow (h1 : balanced_initial a b) (h2 : sparrows_heavier a b) (h3 : balanced_after_exchange a b) : 
  b = (5 / 4) * a :=
sorry

end weight_of_sparrow_l119_119745


namespace probability_of_9_heads_in_12_flips_l119_119052

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l119_119052


namespace factory_profit_l119_119913

def cost_per_unit : ℝ := 2.00
def fixed_cost : ℝ := 500.00
def selling_price_per_unit : ℝ := 2.50

theorem factory_profit (x : ℕ) (hx : x > 1000) :
  selling_price_per_unit * x > fixed_cost + cost_per_unit * x :=
by
  sorry

end factory_profit_l119_119913


namespace part_I_part_II_part_III_l119_119317

def f (x a : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 1

theorem part_I (h : ∃ a : ℝ, (deriv (λ x, f x a) 1 = 0)) : (∃ a, a = -1) :=
by {
  have : deriv (λ x, f x a) x = 6 * x^2 + 6 * a * x,
  have h' := h (deriv (λ x, f x a) 1),
  sorry
}

theorem part_II (a : ℝ) :
  (if a = 0 then ∀ x, Monotonic (f x 0)
  else if a > 0 then ∀ x, Monotonic (f x a)
  else ∀ x, Monotonic (f x a)
  :=
by {
  intro h,
  sorry,
}

theorem part_III (a : ℝ) :
  (if a >= 0 then ∃ m, m = 1
  else if -2 < a < 0 then ∃ m, m = a^3 + 1
  else ∃ m, m = 17 + 12 * a)
  :=
by {
  sorry,
}

end part_I_part_II_part_III_l119_119317


namespace probability_heads_9_of_12_is_correct_l119_119028

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l119_119028


namespace probable_deviation_l119_119785

noncomputable def density (σ a x : ℝ) : ℝ :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - a) ^ 2) / (2 * σ ^ 2))

theorem probable_deviation (a σ : ℝ) (hσ : σ > 0) :
  ∃ E : ℝ, P (|X - a| < E) = 0.5 ∧ E = 0.675 * σ :=
  sorry

end probable_deviation_l119_119785


namespace probability_heads_9_of_12_flips_l119_119006

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l119_119006


namespace range_of_3t_plus_s_l119_119983

noncomputable def f : ℝ → ℝ := sorry

def is_increasing (f : ℝ → ℝ) := ∀ x y, x ≤ y → f x ≤ f y

def symmetric_about (f : ℝ → ℝ) (a b : ℝ) := 
  ∀ x, f (x - a) = b - f (a - x)

def satisfies_inequality (s t : ℝ) (f : ℝ → ℝ) := 
  f (s^2 - 2*s) ≥ -f (2*t - t^2)

def in_interval (s : ℝ) := 1 ≤ s ∧ s ≤ 4

theorem range_of_3t_plus_s (f : ℝ → ℝ) :
  is_increasing f ∧ symmetric_about f 3 0 →
  (∀ s t, satisfies_inequality s t f → in_interval s → -2 ≤ 3 * t + s ∧ 3 * t + s ≤ 16) :=
sorry

end range_of_3t_plus_s_l119_119983


namespace probability_of_9_heads_in_12_flips_l119_119056

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l119_119056


namespace probability_heads_9_of_12_is_correct_l119_119026

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l119_119026


namespace chad_total_spend_on_ice_l119_119621

-- Define the given conditions
def num_people : ℕ := 15
def pounds_per_person : ℕ := 2
def pounds_per_bag : ℕ := 1
def price_per_pack : ℕ := 300 -- Price in cents to avoid floating-point issues
def bags_per_pack : ℕ := 10

-- The main statement to prove
theorem chad_total_spend_on_ice : 
  (num_people * pounds_per_person * 100 / (pounds_per_bag * bags_per_pack) * price_per_pack / 100 = 9) :=
by sorry

end chad_total_spend_on_ice_l119_119621


namespace repeating_decimal_exceeds_decimal_l119_119224

noncomputable def repeating_decimal_to_fraction : ℚ := 9 / 11
noncomputable def decimal_to_fraction : ℚ := 3 / 4

theorem repeating_decimal_exceeds_decimal :
  repeating_decimal_to_fraction - decimal_to_fraction = 3 / 44 :=
by
  sorry

end repeating_decimal_exceeds_decimal_l119_119224


namespace vanya_speed_problem_l119_119526

theorem vanya_speed_problem (v : ℝ) (hv : (v + 2) / v = 2.5) : ((v + 4) / v) = 4 :=
by
  have v_pos : v ≠ 0 := sorry  -- Prove that v is not zero
  have v_eq : v = 4 / 3 := sorry  -- Use hv to solve for v
  rw [v_eq] at v_eq,
  rw [v_eq]
  sorry

end vanya_speed_problem_l119_119526


namespace probability_9_heads_12_flips_l119_119165

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l119_119165


namespace probability_of_9_heads_in_12_l119_119130

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l119_119130


namespace david_deductions_l119_119233

/-- David earns 25 dollars per hour. 
2.5% of his earnings are deducted for local taxes, 
and 3% are contributed to a retirement fund. 
Prove that David pays 137.5 cents per hour 
for local taxes and retirement contributions combined. -/
theorem david_deductions (hourly_wage : ℝ) (tax_rate : ℝ) (retirement_rate : ℝ) :
  hourly_wage = 25 → 
  tax_rate = 0.025 → 
  retirement_rate = 0.03 →
  tax_rate * (hourly_wage * 100) + retirement_rate * (hourly_wage * 100) = 137.5 :=
by
  intros hWage hTaxRate hRetireRate
  rw [hWage, hTaxRate, hRetireRate]
  norm_num
  sorry

end david_deductions_l119_119233


namespace probability_of_9_heads_in_12_l119_119128

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l119_119128


namespace integral_x_pow_l119_119689

theorem integral_x_pow (n : ℝ) (h : 0 < n) : ∫ x in 0..1, x^n = 1 / (n + 1) :=
sorry

end integral_x_pow_l119_119689


namespace math_problem_proof_l119_119617

noncomputable def math_problem : Prop := 
  (4 - Real.sqrt 3)^0 - 3 * Real.tan (Real.pi / 3) - (-1 / 2)^(-1) + Real.sqrt 12 = 3 - Real.sqrt 3

theorem math_problem_proof : math_problem := by
  sorry

end math_problem_proof_l119_119617


namespace parity_F_l119_119706

noncomputable def f (x : ℝ) (m : ℤ) : ℝ := (1 / x) ^ (3 + 2 * m - m ^ 2)
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := a * x ^ (-4) + (a - 2) * x ^ 6

axiom monotonically_decreasing_even (m : ℤ) : (∀ x > 0, (x * x > 0) → f x m = x ^ -(4)) ∧ f 1 m = f (-1) m

theorem parity_F (a : ℝ) : 
  if a = 0 then ∀ x, F(-x) = -F(x)
  else if a = 2 then ∀ x, F(x) = F(-x)
  else ¬∀ x, (F(1) = F(-1)) ∨ ¬∀ x, (F(1) = -F(-1)) :=
by sorry

end parity_F_l119_119706


namespace fraction_zero_l119_119171

theorem fraction_zero (x : ℝ) (h₁ : 2 * x = 0) (h₂ : x + 2 ≠ 0) : (2 * x) / (x + 2) = 0 :=
by {
  sorry
}

end fraction_zero_l119_119171


namespace probability_exactly_9_heads_l119_119114

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l119_119114


namespace football_cost_l119_119430

-- Definitions derived from conditions
def marbles_cost : ℝ := 9.05
def baseball_cost : ℝ := 6.52
def total_spent : ℝ := 20.52

-- The statement to prove the cost of the football
theorem football_cost :
  ∃ (football_cost : ℝ), football_cost = total_spent - marbles_cost - baseball_cost :=
sorry

end football_cost_l119_119430


namespace range_of_a_l119_119316

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x + 4 else 2^x

theorem range_of_a (a : ℝ) (h : f(a) ≥ 2) : a ∈ Set.Icc (-2 : ℝ) 0 ∪ Set.Ici (1 : ℝ) :=
sorry

end range_of_a_l119_119316


namespace isosceles_trapezoid_area_l119_119958

theorem isosceles_trapezoid_area (x y : ℝ)
  (h1 : 0.8 = real.sin (real.arcsin (0.8)))
  (h2 : 16 = y + 1.2 * x)
  (h3 : 2 * y + 1.2 * x = 2 * x) 
  (h4 : x = 10)
  (h5 : y = 4)
  : 1 / 2 * (4 + 16) * (0.8 * 10) = 80 :=
by
  sorry


end isosceles_trapezoid_area_l119_119958


namespace find_omega_l119_119694

theorem find_omega
    (ω : ℝ)
    (hω : 0 < ω)
    (h_eq : sin (ω * (π / 6) + (π / 3)) = sin (ω * (π / 3) + (π / 3)))
    (h_min : ∀ x, π / 6 < x ∧ x < π / 3 → sin (ω * x + π / 3) ≠ 1) :
    ω = 14 / 3 :=
by
  sorry

end find_omega_l119_119694


namespace jason_less_than_jenny_l119_119389

-- Definition of conditions

def grade_Jenny : ℕ := 95
def grade_Bob : ℕ := 35
def grade_Jason : ℕ := 2 * grade_Bob -- Bob's grade is half of Jason's grade

-- The theorem we need to prove
theorem jason_less_than_jenny : grade_Jenny - grade_Jason = 25 :=
by
  sorry

end jason_less_than_jenny_l119_119389


namespace largest_n_distinct_numbers_in_circle_l119_119540

theorem largest_n_distinct_numbers_in_circle (n : ℕ) :
  (∀ (a : ℕ → ℕ), ∃ (n ≤ 6), (∀ (i : ℕ), a i * a ((i + 1) % n) = a ((i + 2) % n)) → (n ≤ 6)) :=
sorry

end largest_n_distinct_numbers_in_circle_l119_119540


namespace problem_solution_l119_119661

def F (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 * n + 1 else 2 * n + 1

theorem problem_solution : (∑ n in Finset.range 99 \ {0, 1}, F (n + 2)) = 10197 :=
by
  sorry

end problem_solution_l119_119661


namespace point_C_on_circle_l119_119381

-- Given conditions
variable {A B C O : Type*}
variable [MetricSpace A]
variable [MetricSpace B]
variable [MetricSpace C]
variable [MetricSpace O]
variable [dist : HasDist O]

-- Definitions based on the given problem's conditions
variable {AB AC BC : ℝ}
variable (is_midpoint : dist O A = dist O B)
variable (radius : dist O A = dist O O)
variable (AB_eq : AB = 13)
variable (AC_eq : AC = 12)
variable (BC_eq : BC = 5)
variable (is_right_triangle : AB ^ 2 = AC ^ 2 + BC ^ 2)

-- Conclusion to prove
theorem point_C_on_circle :
  dist O C = dist O A :=
by
  sorry

end point_C_on_circle_l119_119381


namespace angle_CBD_is_115_l119_119739

theorem angle_CBD_is_115 
{A B C D : Type} 
(triangle_ABC_isosceles : ∀ (T : Type), (A T) = (C T) → (B T) = (C T)) 
(angle_C_50 : ∀ {α : Type}, ∃ (T : α), (∠ T) = 50) 
(D_on_AB : ∃ (d : D), d ∈ [A, B]) : 
  ∃ (m : ℝ), ∠ CBD = 115 :=
by 
  sorry

end angle_CBD_is_115_l119_119739


namespace max_value_m_l119_119291

variables {A B : Point}
def P (m : ℝ) : Point := ⟨-m, 0⟩
def P (m : ℝ) : Point := ⟨m, 0⟩
def C : Circle := ⟨⟨3, 4⟩, 1⟩
def PA_perp_PB (P : Point) (A B : Point) : Prop := ∃ P, P.on_circle C ∧ P.AB_perpendicular

theorem max_value_m 
  (h1: ∀ P, P.on_circle ⟨3, 4⟩ 1 → (PA_perp_PB P A B)) 
  (h2: A = ⟨-m, 0⟩) 
  (h3: B = ⟨m, 0⟩) : 
  m = 6 := 
sorry

end max_value_m_l119_119291


namespace fly_minimum_distance_l119_119927

noncomputable def minimum_distance : ℝ := 707.107

variables (r h d₁ d₂ : ℝ)

axiom cone_radius : r = 300
axiom cone_height : h = 150 * Real.sqrt 7
axiom start_distance : d₁ = 100
axiom end_distance : d₂ = 350 * Real.sqrt 2

theorem fly_minimum_distance :
  let R := Real.sqrt (r^2 + h^2),
      C := 2 * Real.pi * r,
      θ := C / R
    in Real.sqrt ((d₂ * Real.cos (θ / 2) - d₁)^2 + (d₂ * Real.sin (θ / 2))^2) = minimum_distance :=
by
  sorry

end fly_minimum_distance_l119_119927


namespace train_crosses_platform_time_l119_119180

theorem train_crosses_platform_time :
  let speed_kmh := 132
  let length_train := 110
  let length_platform := 165
  let speed_mps := (132 * 1000) / 3600
  let total_distance := length_train + length_platform
  let time := total_distance / speed_mps
  abs (time - 7.49) < 0.01 := 
by {
  -- Definitions are assumed to be given as conditions above.
  let speed_kmh := 132
  let length_train := 110
  let length_platform := 165
  let speed_mps := (132 * 1000) / 3600
  let total_distance := length_train + length_platform
  let time := total_distance / speed_mps
  have time_approx : abs (time - 7.49) < 0.01, sorry,
  exact time_approx
}

end train_crosses_platform_time_l119_119180


namespace vanya_speed_l119_119509

variable (v : ℝ)

theorem vanya_speed (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end vanya_speed_l119_119509


namespace probability_exactly_9_heads_l119_119112

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l119_119112


namespace find_smallest_N_l119_119655

theorem find_smallest_N :
  ∃ N: ℕ, N > 0 ∧ (∃ i ∈ {0, 1, 2}, (∃ m ∈ {2^2, 3^2, 5^2}, (N + i) % m = 0) ∧
                              (∃ i ∈ {0, 1, 2} - {i}, (∃ m ∈ ({2^2, 3^2, 5^2} - {m}), (N + i) % m = 0)) ∧
                              (∃ i ∈ {0, 1, 2} - {i,i}, (∃ m ∈ ({2^2, 3^2, 5^2} - {m,m}), (N + i) % m = 0))) ∧
    N = 475 :=
sorry

end find_smallest_N_l119_119655


namespace probability_of_9_heads_in_12_flips_l119_119057

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l119_119057


namespace probability_exactly_9_heads_in_12_flips_l119_119102

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l119_119102


namespace bicycle_inventory_diff_l119_119434

-- Definitions of conditions based on the problem
def initial_inventory : ℕ := 200
def decrease_rate_feb : ℕ := 4
def decrease_rate_mar : ℕ := 6
def decrease_rate_apr : ℕ := 8
def extra_sales_may : ℕ := 5
def extra_sales_jul : ℕ := 10
def extra_sales_sep : ℕ := 15
def shipment_jun : ℕ := 50
def shipment_aug : ℕ := 60

-- The sequence of decrease rates from May onwards
def decrease_rate (n : ℕ) : ℕ :=
  if n = 1 then decrease_rate_mar + decrease_rate_apr
  else if n = 2 then decrease_rate_apr + (decrease_rate_mar + decrease_rate_apr)
  else (decrease_rate n-1) + (decrease_rate n-2)

-- The main theorem stating the final proof
theorem bicycle_inventory_diff :
  initial_inventory - 
  (
    (initial_inventory - decrease_rate_feb - decrease_rate_mar - decrease_rate_apr - 
     (decrease_rate 1) - extra_sales_may - (decrease_rate 2) + shipment_jun -
     (decrease_rate 3) - extra_sales_jul - (decrease_rate 4) + shipment_aug -
     (decrease_rate 5) - extra_sales_sep)
  ) = 162 :=
by sorry

end bicycle_inventory_diff_l119_119434


namespace leila_spending_l119_119396

theorem leila_spending (sweater jewelry total money_left : ℕ) (h1 : sweater = 40) (h2 : sweater * 4 = total) (h3 : money_left = 20) (h4 : total - sweater - jewelry = money_left) : jewelry - sweater = 60 :=
by
  sorry

end leila_spending_l119_119396


namespace corrected_mean_l119_119837

theorem corrected_mean (mean : ℝ) (n : ℕ) (incorrect : ℝ) (correct : ℝ) (incorrect_mean : ℝ)  
  (H1 : mean = 36)
  (H2 : n = 50)
  (H3 : incorrect = 23)
  (H4 : correct = 48)
  (H5 : incorrect_mean = 36) : 
  (mean + (correct - incorrect) / n = 36.5) :=
by 
  have S_incorrect := mean * n
  have difference := correct - incorrect
  have S_corrected := S_incorrect + difference
  have mean_corrected := S_corrected / n
  calc mean + (correct - incorrect) / n = mean + (25) / 50 : by sorry
  ... = 36.5 : by sorry

end corrected_mean_l119_119837


namespace cube_surface_area_l119_119747

/-- A cube with an edge length of 10 cm has smaller cubes with edge length 2 cm 
    dug out from the middle of each face. The surface area of the new shape is 696 cm². -/
theorem cube_surface_area (original_edge : ℝ) (small_cube_edge : ℝ)
  (original_edge_eq : original_edge = 10) (small_cube_edge_eq : small_cube_edge = 2) :
  let original_surface := 6 * original_edge ^ 2
  let removed_area := 6 * small_cube_edge ^ 2
  let added_area := 6 * 5 * small_cube_edge ^ 2
  let new_surface := original_surface - removed_area + added_area
  new_surface = 696 := by
  sorry

end cube_surface_area_l119_119747


namespace true_proposition_among_ABCD_l119_119955

theorem true_proposition_among_ABCD : 
  (∀ x : ℝ, x^2 < x + 1) = false ∧
  (∀ x : ℝ, x^2 ≥ x + 1) = false ∧
  (∃ x : ℝ, ∀ y : ℝ, x * y^2 ≠ y^2) = true ∧
  (∀ x : ℝ, ∃ y : ℝ, x > y^2) = false :=
by 
  sorry

end true_proposition_among_ABCD_l119_119955


namespace common_root_quad_eqns_l119_119558

theorem common_root_quad_eqns (a1 a2 a3 b1 b2 b3 : ℝ) (h1 : a1^2 - 4 * b1 ≥ 0) (h2 : a2^2 - 4 * b2 ≥ 0) (h3 : a3^2 - 4 * b3 ≥ 0) 
  (h_common_root12 : ∃ x  : ℝ, x^2 - a1 * x + b1 = 0 ∧ x^2 - a2 * x + b2 = 0) 
  (h_common_root13 : ∃ x  : ℝ, x^2 - a1 * x + b1 = 0 ∧ x^2 - a3 * x + b3 = 0) 
  (h_common_root23 : ∃ x  : ℝ, x^2 - a2 * x + b2 = 0 ∧ x^2 - a3 * x + b3 = 0) :
  let A := 1 / 2 * (a1 + a2 + a3)
  in b1 = (A - a2) * (A - a3) ∧ b2 = (A - a3) * (A - a1) ∧ b3 = (A - a1) * (A - a2) :=
by
  sorry

end common_root_quad_eqns_l119_119558


namespace BECD_is_rhombus_l119_119748

-- Definitions of points and trapezoid
variables {α : Type*} [linear_ordered_field α]

noncomputable def is_trapezoid (A B C D : α → α → Type*) : Prop :=
∃ (AB BC CD DA : line α), (_on_line A B AB) ∧ (on_line B C BC) ∧ (on_line C D CD) ∧ (on_line D A DA) ∧ parallel AB CD

noncomputable def is_equal_length (P Q R S : α → α → Type*) : Prop :=
distance P Q = distance Q R ∧ distance Q R = distance R S ∧ distance R S = distance S P

-- Given conditions
variables {A B C D E O : α → α → Type*}

-- Definitions for trapezoid and conditions
def trapezoid_ABC_exists_side_equal := is_trapezoid A B C D ∧ is_equal_length A B C D
def diagonals_intersect (O : α → α → Type*) := ∃ (diag_1 diag_2 : line α), (on_line A C diag_1) ∧ (on_line B D diag_2) ∧ intersect diag_1 diag_2 O

def circumcircle_intersects_base (A B O E : α → α → Type*) : Prop := 
∃ (circ : circle α), on_circle A circ ∧ on_circle B circ ∧ on_circle O circ ∧ on_circle E circ

-- Theorem statement
theorem BECD_is_rhombus
  (h1 : trapezoid_ABC_exists_side_equal)
  (h2 : diagonals_intersect O)
  (h3 : circumcircle_intersects_base A B O E) :
  is_rhombus B E C D :=
sorry

end BECD_is_rhombus_l119_119748


namespace vanya_speed_increased_by_4_l119_119501

variable (v : ℝ)
variable (h1 : (v + 2) / v = 2.5)

theorem vanya_speed_increased_by_4 (h1 : (v + 2) / v = 2.5) : (v + 4) / v = 4 := 
sorry

end vanya_speed_increased_by_4_l119_119501


namespace tan_alpha_eq_neg_one_third_l119_119274

theorem tan_alpha_eq_neg_one_third (α : ℝ) (h : (sin α - 2 * cos α) / (2 * sin α + cos α) = -1) :
  tan α = -1 / 3 :=
by
  sorry

end tan_alpha_eq_neg_one_third_l119_119274


namespace vanya_faster_speed_l119_119515

def vanya_speed (v : ℝ) : Prop :=
  (v + 2) / v = 2.5

theorem vanya_faster_speed (v : ℝ) (h : vanya_speed v) : (v + 4) / v = 4 :=
by
  sorry

end vanya_faster_speed_l119_119515


namespace smallest_possible_N_l119_119776

theorem smallest_possible_N (p q r s t : ℕ) (h_pos : p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧ t > 0)
(h_sum : p + q + r + s + t = 2022) :
    ∃ N : ℕ, N = 506 ∧ N = max (p + q) (max (q + r) (max (r + s) (s + t))) :=
by
    sorry

end smallest_possible_N_l119_119776


namespace edward_washers_remaining_l119_119647

theorem edward_washers_remaining (pipe_length : ℕ) (feet_per_bolt : ℕ) (washers_per_bolt : ℕ) (initial_washers : ℕ) (bolts_needed : ℕ) (washers_used : ℕ) :
  pipe_length = 40 →
  feet_per_bolt = 5 →
  washers_per_bolt = 2 →
  initial_washers = 20 →
  bolts_needed = pipe_length / feet_per_bolt →
  washers_used = bolts_needed * washers_per_bolt →
  initial_washers - washers_used = 4 :=
by
  intro pipe_length_eq
  intro feet_per_bolt_eq
  intro washers_per_bolt_eq
  intro initial_washers_eq
  intro bolts_needed_eq
  intro washers_used_eq
  rw [pipe_length_eq, feet_per_bolt_eq, washers_per_bolt_eq, initial_washers_eq, bolts_needed_eq, washers_used_eq]
  sorry

end edward_washers_remaining_l119_119647


namespace fraction_simplification_l119_119344

theorem fraction_simplification
  (a b c x : ℝ)
  (hb : b ≠ 0)
  (hxc : c ≠ 0)
  (h : x = a / b)
  (ha : a ≠ c * b) :
  (a + c * b) / (a - c * b) = (x + c) / (x - c) :=
by
  sorry

end fraction_simplification_l119_119344


namespace prob_at_least_one_acceptance_l119_119877

theorem prob_at_least_one_acceptance :
  let P_A : ℝ := 0.6,
      P_B : ℝ := 0.7 in
  let P_A_not : ℝ := 1 - P_A,
      P_B_not : ℝ := 1 - P_B in
  let P_neither : ℝ := P_A_not * P_B_not in
  let P_at_least_one : ℝ := 1 - P_neither in
  P_at_least_one = 0.88 :=
by
  sorry

end prob_at_least_one_acceptance_l119_119877


namespace probability_heads_9_of_12_l119_119143

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l119_119143


namespace find_radius_l119_119208

noncomputable def sphere_intersections : Prop :=
  let center_xz := (3:ℝ, 0, 5)
  let radius_xz := (2:ℝ)
  let center_xy := (3:ℝ, 7, 0)
  let radius_xy := (sqrt 28:ℝ)
  let sphere_center := (3:ℝ, 7, 5)
  
  ∃ r : ℝ, 
  (sphere_center = (3, 7, 5)) ∧ 
  (dist (3, 0, 5) sphere_center = sqrt(4 + 49)) ∧ 
  (dist (3, 7, 0) sphere_center = radius_xy)

theorem find_radius : sphere_intersections :=
sorry

end find_radius_l119_119208


namespace angle_between_PA_BC_l119_119945

-- Define the key elements in the given problem.
variables {Point : Type} [metric_space Point]

noncomputable def angle_between (u v : Point) : ℝ := sorry

-- Define points and conditions based on the problem description.
variables (P A B C P1 P2 P3 : Point)
variables (hp1 : collinear [P1, C, P2])
variables (hp2 : collinear [P2, B, P3])
variables (hp3 : dist P1 P2 = dist P2 P3)

-- The proof statement to show the angle between PA and BC.
theorem angle_between_PA_BC (P A B C P1 P2 P3 : Point)
  (hp1 : collinear [P1, C, P2])
  (hp2 : collinear [P2, B, P3])
  (hp3 : dist P1 P2 = dist P2 P3) : 
  angle_between (dist PA BC) = 90 :=
sorry

end angle_between_PA_BC_l119_119945


namespace cookies_cost_l119_119492

def cost_of_cookies (total_spent candy_bar_cost : ℕ) : ℕ :=
  total_spent - candy_bar_cost

theorem cookies_cost (total_spent candy_bar_cost : ℕ) (h_total_spent : total_spent = 53) (h_candy_bar_cost : candy_bar_cost = 14) :
  cost_of_cookies total_spent candy_bar_cost = 39 :=
by
  rw [h_total_spent, h_candy_bar_cost]
  have : 53 - 14 = 39 := sorry
  simp [cost_of_cookies, this]
  sorry

end cookies_cost_l119_119492


namespace john_runs_with_dog_for_half_hour_l119_119391

noncomputable def time_with_dog_in_hours (t : ℝ) : Prop := 
  let d1 := 6 * t          -- Distance run with the dog
  let d2 := 4 * (1 / 2)    -- Distance run alone
  (d1 + d2 = 5) ∧ (t = 1 / 2)

theorem john_runs_with_dog_for_half_hour : ∃ t : ℝ, time_with_dog_in_hours t := 
by
  use (1 / 2)
  sorry

end john_runs_with_dog_for_half_hour_l119_119391


namespace probability_heads_in_nine_of_twelve_flips_l119_119088

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l119_119088


namespace product_of_k_plus_1_numbers_l119_119494

open Nat

theorem product_of_k_plus_1_numbers (k : ℕ) (hk : k > 0) : 
  ∏ i in finset.range (k + 1) (λ i, (k + 1) + i) = ∏ i in finset.range (k + 1) (λ i, i + 1) := 
sorry

end product_of_k_plus_1_numbers_l119_119494


namespace fraction_given_to_classmates_l119_119398

theorem fraction_given_to_classmates
  (total_boxes : ℕ) (pens_per_box : ℕ)
  (percentage_to_friends : ℝ) (pens_left_after_classmates : ℕ) :
  total_boxes = 20 →
  pens_per_box = 5 →
  percentage_to_friends = 0.40 →
  pens_left_after_classmates = 45 →
  (15 / (total_boxes * pens_per_box - percentage_to_friends * total_boxes * pens_per_box)) = 1 / 4 :=
by
  intros h1 h2 h3 h4
  sorry

end fraction_given_to_classmates_l119_119398


namespace vanya_faster_speed_l119_119510

def vanya_speed (v : ℝ) : Prop :=
  (v + 2) / v = 2.5

theorem vanya_faster_speed (v : ℝ) (h : vanya_speed v) : (v + 4) / v = 4 :=
by
  sorry

end vanya_faster_speed_l119_119510


namespace perpendicular_lines_a_eq_neg6_l119_119358

theorem perpendicular_lines_a_eq_neg6 
  (a : ℝ) 
  (h1 : ∀ x y : ℝ, ax + 2*y + 1 = 0) 
  (h2 : ∀ x y : ℝ, x + 3*y - 2 = 0) 
  (h_perpendicular : ∀ m1 m2 : ℝ, m1 * m2 = -1) : 
  a = -6 := 
by 
  sorry

end perpendicular_lines_a_eq_neg6_l119_119358


namespace probability_red_black_red_l119_119932

def suits := {'hearts', 'diamonds', 'spades', 'clubs'} -- 4 suits
def colors := {'red', 'black'} -- 2 colors
-- hearts and diamonds are red, spades and clubs are black
def color_of_suit (s: string) : string :=
  if s = 'hearts' ∨ s = 'diamonds' then 'red' else 'black'

-- 52 cards is a triviality in this context, hence we can assume we've 52 containers equally distributed.
theorem probability_red_black_red :
  let deck := (suits.product (List.range' 1 14)).product (List.replicate 4 13) in -- 52 cards generation 
  let red_cards := filter (λ (c : (string × ℕ) × ℕ), color_of_suit (c.fst.fst) = 'red') deck in
  let black_cards := filter (λ (c : (string × ℕ) × ℕ), color_of_suit (c.fst.fst) = 'black') deck in
  (deck.length = 52) →
  (red_cards.length = 26) →
  (black_cards.length = 26) →
  let prob_first_red := (red_cards.length.to_rat / deck.length.to_rat) in
  let prob_second_black_given_first_red := (black_cards.length.to_rat / (deck.length - 1).to_rat) in
  let prob_third_red_given_first_red_and_second_black := (red_cards.length - 1).to_rat / (deck.length - 2).to_rat in
  prob_first_red * prob_second_black_given_first_red * prob_third_red_given_first_red_and_second_black = 13 / 102 :=
by
  intros deck red_cards black_cards h_deck_length h_red_cards_length h_black_cards_length
  unfold prob_first_red prob_second_black_given_first_red prob_third_red_given_first_red_and_second_black
  sorry

end probability_red_black_red_l119_119932


namespace sums_not_equal_l119_119763

-- First we define the function f.
def f (x : ℕ) : ℚ := 1 / (x^2 + 1)

-- Then we express the sum of f(x) over a set.
def sum_f (s : Finset ℕ) : ℚ := s.sum f

-- Kostya and Andrey's selections are modeled as subsets of {1, ..., 30}.
variable {Kostya Andrey : Finset ℕ}

-- We formalize the condition that their chosen subsets are disjoint and each has 15 numbers.
def valid_selection (Kostya Andrey : Finset ℕ) : Prop :=
  Kostya ∪ Andrey = (Finset.range 30).filter (λ x, x + 1 ≠ 0)
  ∧ Kostya.card = 15
  ∧ Andrey.card = 15
  ∧ ∀ (x ∈ Kostya), x ∉ Andrey

-- Finally, we state the theorem to be proved.
theorem sums_not_equal (Kostya Andrey : Finset ℕ) (h : valid_selection Kostya Andrey) : 
  sum_f Kostya ≠ sum_f Andrey := 
sorry

end sums_not_equal_l119_119763


namespace FG_parallel_DE_l119_119401

open EuclideanGeometry

variables {A B C D E F G : Point} (TriangleABC : Triangle A B C)
variables (H1 : acuteTriangle TriangleABC)
variables (H2 : onSegment D A B)
variables (H3 : onSegment E A C)
variables (H4 : dist A D = dist A E)
variables (H5 : F = intersection (perpendicularBisector (segment C E)) (minorArc A B))
variables (H6 : G = intersection (perpendicularBisector (segment B D)) (minorArc A C))

theorem FG_parallel_DE : parallel (line F G) (line D E) :=
by
  sorry

end FG_parallel_DE_l119_119401


namespace strawberries_weight_l119_119425

theorem strawberries_weight (marco_weight dad_increase : ℕ) (h_marco: marco_weight = 30) (h_diff: marco_weight = dad_increase + 13) : marco_weight + (marco_weight - 13) = 47 :=
by
  sorry

end strawberries_weight_l119_119425


namespace probability_exactly_9_heads_in_12_flips_l119_119096

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l119_119096


namespace probability_exactly_9_heads_in_12_flips_l119_119099

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l119_119099


namespace frog_jump_within_2_meters_l119_119576

noncomputable def frog_jump_probability : ℝ :=
  let u₁ u₂ u₃ : ℝ × ℝ := (1, 0)
  let v₁ v₂ : ℝ × ℝ := (1/2, 0)
  let r : ℝ × ℝ := (2 * u₁.1 + 2 * u₂.1 + 2 * u₃.1 + v₁.1 + v₂.1,
                     2 * u₁.2 + 2 * u₂.2 + 2 * u₃.2 + v₁.2 + v₂.2)
  if ∥r∥ ≤ 2 then 1 else 0

theorem frog_jump_within_2_meters :
  frog_jump_probability = 1 / 10 :=
sorry

end frog_jump_within_2_meters_l119_119576


namespace vanya_faster_speed_l119_119521

theorem vanya_faster_speed (v : ℝ) (h : v + 2 = 2.5 * v) : (v + 4) / v = 4 := by
  sorry

end vanya_faster_speed_l119_119521


namespace bob_distance_from_start_l119_119917

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Distance formula between two points
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Define the conditions from the problem
def movement_distance := 12
def side_length := 3
def start_point := Point.mk 0 0

-- Assume Bob walks along the perimeter of the regular octagon
def final_point := Point.mk 3 3

-- Proof that Bob's distance from starting point is 3√2 km
theorem bob_distance_from_start :
  distance start_point final_point = 3 * real.sqrt 2 :=
sorry

end bob_distance_from_start_l119_119917


namespace actual_ranking_correct_l119_119659

-- Defining students as an enumerated type
inductive Student : Type
| A | B | C | D | E
deriving DecidableEq, Repr

open Student

-- Defining the prediction lists for X and Y
def X_prediction : List Student := [A, B, C, D, E]
def Y_prediction : List Student := [D, A, E, C, B]

-- Defining the conditions for X
def wrong_positions_X (ranking : List Student) : Prop :=
  ∀ i, i < X_prediction.length → (ranking.get! i ≠ X_prediction.get! i)

def wrong_pairs_X (ranking : List Student) : Prop :=
  ∀ i, i + 1 < X_prediction.length → 
  (ranking.get! i ≠ X_prediction.get! i ∧ ranking.get! (i + 1) ≠ X_prediction.get! (i + 1))

-- Defining the conditions for Y
def correct_positions_Y (ranking : List Student) : Prop :=
  (ranking.get! 0 = D ∧ ranking.get! 4 = B)

def correct_pairs_Y (ranking : List Student) : Prop :=
  (ranking.get! 3 = C → ranking.get! 4 = B)

-- Combining all conditions
def valid_ranking (ranking : List Student) : Prop :=
  ranking.length = 5 ∧ 
  (List.all (fun x => List.contains ranking x) [A, B, C, D, E]) ∧
  wrong_positions_X ranking ∧ wrong_pairs_X ranking ∧ 
  correct_positions_Y ranking ∧ correct_pairs_Y ranking

-- The actual ranking order
def actual_ranking : List Student := [E, D, A, C, B]

theorem actual_ranking_correct : valid_ranking actual_ranking :=
by
  -- Proof here is omitted by 'sorry'
  sorry

end actual_ranking_correct_l119_119659


namespace football_cost_l119_119431

-- Definitions derived from conditions
def marbles_cost : ℝ := 9.05
def baseball_cost : ℝ := 6.52
def total_spent : ℝ := 20.52

-- The statement to prove the cost of the football
theorem football_cost :
  ∃ (football_cost : ℝ), football_cost = total_spent - marbles_cost - baseball_cost :=
sorry

end football_cost_l119_119431


namespace probability_of_9_heads_in_12_l119_119126

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l119_119126


namespace find_hypotenuse_square_l119_119626

axiom p q r s t u k : ℂ 
axiom h_poly : (p^3 + q^3 + r^3 + s*p^2 + s*q^2 + s*r^2 + t*p + t*q + t*r + u = 0) 
axiom h_modulus_sum : (|p|^2 + |q|^2 + |r|^2 = 360)
axiom h_right_triangle : (right_triangle (p, q, r)) 

theorem find_hypotenuse_square
  (h_poly : (p^3 + s*p^2 + t*p + u = 0))
  (h_modulus_sum : |p|^2 + |q|^2 + |r|^2 = 360)
  (h_right_triangle : (right_triangle (p, q, r))) :
  k^2 = 540 :=
sorry

end find_hypotenuse_square_l119_119626


namespace min_elements_in_valid_subset_l119_119820

def two_digit_numbers : set (ℕ × ℕ) :=
  { p | p.1 ∈ { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 } ∧ p.2 ∈ { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 } }

def valid_subset (X : set (ℕ × ℕ)) : Prop :=
  ∀ (s : ℕ → ℕ), (∀ n, s n ∈ { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }) →
  ∃ n, (s n, s (n+1)) ∈ X

theorem min_elements_in_valid_subset :
  ∃ (X : set (ℕ × ℕ)), valid_subset X ∧ X.card = 55 :=
sorry

end min_elements_in_valid_subset_l119_119820


namespace geometry_problem_l119_119418

variable {ABC : Triangle}
variable (p R r q : ℝ)
variable (h₁ : semiperimeter ABC = p)
variable (h₂ : circumscribed_circle_radius ABC = R)
variable (h₃ : inscribed_circle_radius ABC = r)
variable (h₄ : semiperimeter (feet_of_altitudes ABC) = q)

theorem geometry_problem 
  (h₁ : semiperimeter ABC = p)
  (h₂ : circumscribed_circle_radius ABC = R)
  (h₃ : inscribed_circle_radius ABC = r)
  (h₄ : semiperimeter (feet_of_altitudes ABC) = q) :
  R / r = p / q := by
sorry

end geometry_problem_l119_119418


namespace max_value_f_l119_119836

theorem max_value_f : ∃ x : ℝ, (sin (2 * x) - cos (x + 3 * Real.pi / 4) = 2) :=
by
  sorry

end max_value_f_l119_119836


namespace bisection_second_value_l119_119933

-- Define the function f(x) = log10 x + x - 2
def f (x : ℝ) : ℝ := log10 x + x - 2

-- Define the proof problem
theorem bisection_second_value :
  ∃ x : ℝ, 1 <  x ∧ x < 2 ∧ f 1 < 0 ∧ f 2 > 0 ∧ abs(x - 1.75) < 0.1 :=
by
  sorry

end bisection_second_value_l119_119933


namespace smallest_positive_m_l119_119545

theorem smallest_positive_m (m : ℕ) : 
  (∃ n : ℤ, (10 * n * (n + 1) = 600) ∧ (m = 10 * (n + (n + 1)))) → (m = 170) :=
by 
  sorry

end smallest_positive_m_l119_119545


namespace probability_heads_9_of_12_flips_l119_119004

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l119_119004


namespace john_final_push_time_l119_119760

-- Define the initial conditions
theorem john_final_push_time (t : ℝ) :
  (t : ℝ) = 36 :=
begin
  -- Given conditions
  let dJohn := 4.2 * t,
  let dSteve := 3.7 * t,
  let aheadBy := 2,
  -- Given distances relation where John is ahead by 2 meters with initial 16 meter disadvantage
  have h : dJohn = dSteve + 16 + aheadBy,
  -- Substitute the given distances
  calc
    dJohn = 4.2 * t : by rfl
        ... = 3.7 * t + 16 + 2 : by sorry,
end

end john_final_push_time_l119_119760


namespace probability_heads_9_of_12_l119_119140

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l119_119140


namespace probability_heads_in_nine_of_twelve_flips_l119_119084

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l119_119084


namespace perimeter_of_unshaded_rectangle_l119_119926

theorem perimeter_of_unshaded_rectangle (length width height base area shaded_area perimeter : ℝ)
  (h1 : length = 12)
  (h2 : width = 9)
  (h3 : height = 3)
  (h4 : base = (2 * shaded_area) / height)
  (h5 : shaded_area = 18)
  (h6 : perimeter = 2 * ((length - base) + width))
  : perimeter = 24 := by
  sorry

end perimeter_of_unshaded_rectangle_l119_119926


namespace correct_statements_count_l119_119954

theorem correct_statements_count : 
  let s1 := (0 ∈ ({0} : Set ℕ))
  let s2 := (∅ ⊆ ({0} : Set ℕ))
  let s3 := ({0, 1} ⊆ ({(0, 1)} : Set (ℕ × ℕ)))
  let s4 := ({(a, b)} = ({(b, a)} : Set (ℕ × ℕ)))
  (Nat (if s1 then 1 else 0) + (if s2 then 1 else 0) + (if s3 then 1 else 0) + (if s4 then 1 else 0)) = 2 :=
by
  sorry

end correct_statements_count_l119_119954


namespace num_edges_upper_bound_l119_119359

-- Definitions and conditions
variable (G : Type) [SimpleGraph G] (K3 : SimpleGraph G) [¬(completeGraph 3 ⊆ G)]
variable (n : Nat) (h_even : Even n)

-- Statement of the theorem
theorem num_edges_upper_bound (hK3 : ∀ K3: SimpleGraph G, ¬(completeGraph 3 ⊆ G)) 
  (hn_even : Even n) : 
  numEdges(G) ≤ n^2 / 4 := 
sorry

end num_edges_upper_bound_l119_119359


namespace cost_per_box_l119_119613

theorem cost_per_box (trays : ℕ) (cookies_per_tray : ℕ) (cookies_per_box : ℕ) (total_cost : ℕ) (box_cost : ℝ) 
  (h1 : trays = 3) 
  (h2 : cookies_per_tray = 80) 
  (h3 : cookies_per_box = 60)
  (h4 : total_cost = 14) 
  (h5 : (trays * cookies_per_tray) = 240)
  (h6 : (240 / cookies_per_box : ℕ) = 4) 
  (h7 : (total_cost / 4 : ℝ) = box_cost) : 
  box_cost = 3.5 := 
by sorry

end cost_per_box_l119_119613


namespace trapezoid_area_l119_119168

noncomputable def area_trapezoid : ℝ :=
  let x1 := 10
  let x2 := -10
  let y1 := 10
  let h := 10
  let a := 20  -- length of top side at y = 10
  let b := 10  -- length of lower side
  (a + b) * h / 2

theorem trapezoid_area : area_trapezoid = 150 := by
  sorry

end trapezoid_area_l119_119168


namespace sum_primes_between_1_and_100_l119_119656

theorem sum_primes_between_1_and_100 :
  ∑ p in {n : ℕ | 1 < n ∧ n < 100 ∧ Prime n ∧ n % 6 = 1 ∧ n % 7 = 6}, p = 116 := by
  sorry

end sum_primes_between_1_and_100_l119_119656


namespace probability_heads_9_of_12_is_correct_l119_119039

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l119_119039


namespace perpendiculars_intersect_at_reflection_l119_119198

noncomputable theory
open EuclideanGeometry

/-- Given a triangle ABC with a circumcircle, and points A1, B1, C1 defined as the intersections of the
circumcircle with lines parallel to BC, CA, AB passing through A, B, C respectively, prove that
the perpendiculars dropped from A1 to BC, B1 to CA, and C1 to AB intersect at a single point
which is the reflection of the orthocenter H over the circumcenter O. -/
theorem perpendiculars_intersect_at_reflection
  (A B C : Point)
  (circumcircle : Circle)
  (H O : Point)
  (A1 B1 C1 : Point)
  (hTriangle : is_triangle A B C)
  (hCircum : is_circumcircle_of_triangle circumcircle A B C)
  (hA1 : is_parallel (line A A1) (line B C) ∧ lies_on_circle A1 circumcircle)
  (hB1 : is_parallel (line B B1) (line C A) ∧ lies_on_circle B1 circumcircle)
  (hC1 : is_parallel (line C C1) (line A B) ∧ lies_on_circle C1 circumcircle)
  (hPerpendiculars : ∀ P Q R, is_orthocenter H (triangle P Q R) → intersects_at P Q R (reflected_point H O))
  :
  exists P : Point, is_common_intersection [perpendicular_to_line A1 B C, perpendicular_to_line B1 C A, perpendicular_to_line C1 A B] P :=
begin
  sorry,
end

end perpendiculars_intersect_at_reflection_l119_119198


namespace min_sum_of_factors_l119_119855

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 2310) : 
  a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l119_119855


namespace students_passed_both_tests_l119_119191

theorem students_passed_both_tests
  (n : ℕ) (A : ℕ) (B : ℕ) (C : ℕ)
  (h1 : n = 100) 
  (h2 : A = 60) 
  (h3 : B = 40) 
  (h4 : C = 20) :
  A + B - ((n - C) - (A + B - n)) = 20 :=
by
  sorry

end students_passed_both_tests_l119_119191


namespace helicopter_final_height_helicopter_fuel_consumption_l119_119242

-- Part 1: Height after five performance actions
theorem helicopter_final_height :
  let h1 := 4.5
  let h2 := -2.1
  let h3 := 1.4
  let h4 := -1.6
  let h5 := 0.9
  h1 + h2 + h3 + h4 + h5 = 3.1 :=
by
  sorry

-- Part 2: Total fuel consumption during five performance actions
theorem helicopter_fuel_consumption :
  let h1 := 4.5
  let h2 := -2.1
  let h3 := 1.4
  let h4 := -1.6
  let h5 := 0.9
  let fuel_ascent := 5
  let fuel_descent := 3
  let total_fuel := (|h1| + |h3| + |h5|) * fuel_ascent + (|h2| + |h4|) * fuel_descent
  total_fuel = 45.1 :=
by
  sorry

end helicopter_final_height_helicopter_fuel_consumption_l119_119242


namespace length_of_segment_AB_l119_119751

variable (A B C M E F D : Type)
variables [Triangle ABC] [Median AE B F CD] [Concyclic E C F M]
variables [segment_length CD = n]

theorem length_of_segment_AB (A B C M E F D : Type) 
  [Triangle ABC] [Median AE B F CD] [Concyclic E C F M] 
  [segment_length CD = n] : 
  segment_length AB = (2 * n * Real.sqrt 3) / 3 := 
  sorry

end length_of_segment_AB_l119_119751


namespace probability_m_gt_0_l119_119278

def f (x : ℝ) : ℝ :=
  if h : 0 ≤ x ∧ x ≤ 2 then 2^x else
  if h : -2 ≤ x ∧ x < 0 then x + 1 else 0

theorem probability_m_gt_0 :
  let M := {y | ∃ x, f x = y} in
  (∃ A B : set ℝ, (∀ x ∈ A, f x ∈ B) ∧ B = set.Icc (-1 : ℝ) 4 ∧
   let positive_subset := set.Ioo (0 : ℝ) 4 in
   (positive_subset.card : ℚ) / (B.card : ℚ) = (4/5 : ℚ)) :=
sorry

end probability_m_gt_0_l119_119278


namespace cost_of_one_dozen_pens_l119_119828

noncomputable def cost_of_one_pen_and_one_pencil_ratio := 5

theorem cost_of_one_dozen_pens
  (cost_pencil : ℝ)
  (cost_3_pens_5_pencils : 3 * (cost_of_one_pen_and_one_pencil_ratio * cost_pencil) + 5 * cost_pencil = 200) :
  12 * (cost_of_one_pen_and_one_pencil_ratio * cost_pencil) = 600 :=
by
  sorry

end cost_of_one_dozen_pens_l119_119828


namespace water_pouring_problem_l119_119577

theorem water_pouring_problem : ∃ n : ℕ, n = 3 ∧
  (1 / (2 * n - 1) = 1 / 5) :=
by
  sorry

end water_pouring_problem_l119_119577


namespace chinese_new_year_workers_l119_119960

theorem chinese_new_year_workers 
  (n : ℕ) (x : ℕ) (w_remaining : ℕ) (total_worker_days : ℕ) (weekend_days : ℕ) :
  n = 15 →
  w_remaining = 121 →
  total_worker_days = 2011 →
  weekend_days = 4 →
  15 * x = 120 :=
by
  intros h1 h2 h3 h4
  have h := calc
    11 * 121 + (3 + 4 + 5 + 6 + 7 + 10 + 11 + 12 + 13 + 14) * x = 2011 : sorry
    1331 + 85 * x = 2011 : sorry
    85 * x = 680 : sorry
    x = 8 : sorry
  show 15 * x = 120 by 
    rw [h]
    sorry

end chinese_new_year_workers_l119_119960


namespace quartic_polynomial_has_roots_l119_119254

theorem quartic_polynomial_has_roots :
  ∃ (p : Polynomial ℚ),
    Polynomial.monic p ∧
    (3 + Real.sqrt 5 ∈ Polynomial.roots p.map(Polynomial.C) ∧
     3 - Real.sqrt 5 ∈ Polynomial.roots p.map(Polynomial.C) ∧
     2 + Real.sqrt 7 ∈ Polynomial.roots p.map(Polynomial.C) ∧
     2 - Real.sqrt 7 ∈ Polynomial.roots p.map(Polynomial.C) ∧
      p = Polynomial.mk [(-12 : ℚ), 2, 25, -10, 1]) :=
  sorry

end quartic_polynomial_has_roots_l119_119254


namespace monic_quartic_polynomial_with_given_roots_l119_119251

theorem monic_quartic_polynomial_with_given_roots :
  ∃ (p : Polynomial ℚ), 
    p = Polynomial.monic * (Polynomial.X^4 - 10 * Polynomial.X^3 + 25 * Polynomial.X^2 + 2 * Polynomial.X - 12) ∧
    p.eval (3 + Real.sqrt 5) = 0 ∧ 
    p.eval (2 - Real.sqrt 7) = 0 ∧
    p.eval (3 - Real.sqrt 5) = 0 ∧
    p.eval (2 + Real.sqrt 7) = 0 :=
by {
  -- Proof will be provided to satisfy the conditions
  sorry
}

end monic_quartic_polynomial_with_given_roots_l119_119251


namespace volume_ratio_l119_119753

noncomputable def tetrahedron_volume_ratio (a b d : ℝ) (ϕ : ℝ) : ℝ :=
a * b * d * sin ϕ

theorem volume_ratio (a b d ϕ : ℝ) : 
  let v := tetrahedron_volume_ratio a b d ϕ in
  let V1 := 10 * v / 81 in
  let V2 := 7 * v / 162 in
  (V1 / V2) = (20 / 7) :=
by
  let v := tetrahedron_volume_ratio a b d ϕ
  let V1 := 10 * v / 81
  let V2 := 7 * v / 162
  have h : V1 / V2 = (10 * v / 81) / (7 * v / 162)
  calc
    (10 * v / 81) / (7 * v / 162) = 10 * 162 / (81 * 7) : by field_simp [V1, V2]
                               ... = 1620 / 567 : by ring
                               ... = 20 / 7 : by norm_num
  exact h

end volume_ratio_l119_119753


namespace length_of_bridge_l119_119590

theorem length_of_bridge (ship_length : ℝ) (ship_speed_kmh : ℝ) (time : ℝ) (bridge_length : ℝ) :
  ship_length = 450 → ship_speed_kmh = 24 → time = 202.48 → bridge_length = (6.67 * 202.48 - 450) → bridge_length = 900.54 :=
by
  intros h1 h2 h3 h4
  sorry

end length_of_bridge_l119_119590


namespace vanya_speed_increased_by_4_l119_119497

variable (v : ℝ)
variable (h1 : (v + 2) / v = 2.5)

theorem vanya_speed_increased_by_4 (h1 : (v + 2) / v = 2.5) : (v + 4) / v = 4 := 
sorry

end vanya_speed_increased_by_4_l119_119497


namespace percentage_basketball_l119_119737

theorem percentage_basketball (total_students : ℕ) (chess_percentage : ℝ) (students_like_chess_basketball : ℕ) 
  (percentage_conversion : ∀ p : ℝ, 0 ≤ p → p / 100 = p) 
  (h_total : total_students = 250) 
  (h_chess : chess_percentage = 10) 
  (h_chess_basketball : students_like_chess_basketball = 125) :
  ∃ (basketball_percentage : ℝ), basketball_percentage = 40 := by
  sorry

end percentage_basketball_l119_119737


namespace articles_sold_l119_119572

theorem articles_sold (C : ℝ) :
    let listed_price := 1.5 * C,
    let selling_price_after_discount := 1.35 * C,
    let profit_selling_price := 1.2 * C,
    let total_cost_price := 40 * C in
    ∃ (n : ℕ), n * profit_selling_price = total_cost_price ∧ n = 33 :=
begin
  sorry
end

end articles_sold_l119_119572


namespace area_of_triangle_eq_sqrt_3_div_2_l119_119286

namespace TriangleArea

-- Definitions of sides and angle given in the problem
def a : Real := Real.sqrt 2
def c : Real := Real.sqrt 6
def C : Real := 2 * Real.pi / 3

-- Statement of the theorem
theorem area_of_triangle_eq_sqrt_3_div_2 : 
  (1/2 * a * c * Real.sin C) = Real.sqrt 3 / 2 :=
by 
  sorry

end TriangleArea

end area_of_triangle_eq_sqrt_3_div_2_l119_119286


namespace min_sum_of_factors_l119_119854

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 2310) : 
  a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l119_119854


namespace probability_exactly_9_heads_l119_119120

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l119_119120


namespace min_sum_factors_l119_119851

theorem min_sum_factors (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prod : a * b * c = 2310) : a + b + c = 40 :=
sorry

end min_sum_factors_l119_119851


namespace probability_heads_9_of_12_flips_l119_119009

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l119_119009


namespace probability_heads_in_9_of_12_flips_l119_119074

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l119_119074


namespace find_x_average_l119_119265

theorem find_x_average :
  ∃ x : ℝ, (x + 8 + (7 * x - 3) + (3 * x + 10) + (-x + 6)) / 4 = 5 * x - 4 ∧ x = 3.7 :=
  by
  use 3.7
  sorry

end find_x_average_l119_119265


namespace seven_pow_fifty_one_mod_103_l119_119806

theorem seven_pow_fifty_one_mod_103 : (7^51 - 1) % 103 = 0 := 
by
  -- Fermat's Little Theorem: If p is a prime number and a is an integer not divisible by p,
  -- then a^(p-1) ≡ 1 ⧸ p.
  -- 103 is prime, so for 7 which is not divisible by 103, we have 7^102 ≡ 1 ⧸ 103.
sorry

end seven_pow_fifty_one_mod_103_l119_119806


namespace probability_heads_exactly_9_of_12_l119_119018

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l119_119018


namespace vanya_faster_speed_l119_119512

def vanya_speed (v : ℝ) : Prop :=
  (v + 2) / v = 2.5

theorem vanya_faster_speed (v : ℝ) (h : vanya_speed v) : (v + 4) / v = 4 :=
by
  sorry

end vanya_faster_speed_l119_119512


namespace palindrome_probability_divisible_by_11_l119_119583

namespace PalindromeProbability

-- Define the concept of a five-digit palindrome and valid digits
def is_five_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ n = 10001 * a + 1010 * b + 100 * c

-- Define the condition for a number being divisible by 11
def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- Count all five-digit palindromes
def count_five_digit_palindromes : ℕ :=
  9 * 10 * 10  -- There are 9 choices for a (1-9), and 10 choices for b and c (0-9)

-- Count five-digit palindromes that are divisible by 11
def count_divisible_by_11_five_digit_palindromes : ℕ :=
  9 * 10  -- There are 9 choices for a, and 10 valid (b, c) pairs for divisibility by 11

-- Calculate the probability
theorem palindrome_probability_divisible_by_11 :
  (count_divisible_by_11_five_digit_palindromes : ℚ) / count_five_digit_palindromes = 1 / 10 :=
  by sorry -- Proof goes here

end PalindromeProbability

end palindrome_probability_divisible_by_11_l119_119583


namespace ideal_gas_amount_l119_119199

-- Define the constants
def a : ℝ := 1.2 -- meters
def delta_h : ℝ := 0.8 * 10^(-3) -- meters
def T : ℝ := 300 -- Kelvin (27°C converted to Kelvin)
def R : ℝ := 8.31 -- J/(K·mol)
def g : ℝ := 10 -- m/s²
def rho : ℝ := 810 -- kg/m³
def epsilon : ℝ := 8 * 10^(-10) -- Pa⁻¹

-- Define the initial pressure p' at the bottom of the tank
def p' : ℝ := rho * g * a

-- Define the final pressure p after injecting the ideal gas
def p : ℝ := p' + delta_h / (a * epsilon)

-- Define the volume V' of the space below the piston
def V' : ℝ := a^2 * delta_h

-- Calculate the amount of ideal gas needed
def n : ℝ := (p * V') / (R * T)

-- Prove the amount of ideal gas required
theorem ideal_gas_amount : n ≈ 3.854 := by
  sorry

end ideal_gas_amount_l119_119199


namespace allison_upload_rate_l119_119598

theorem allison_upload_rate (x : ℕ) (h1 : 15 * x + 30 * x = 450) : x = 10 :=
by
  sorry

end allison_upload_rate_l119_119598


namespace min_tan4_cot4_l119_119259

theorem min_tan4_cot4 (x : ℝ) (h1 : tan x ≠ 0) (h2 : cot x ≠ 0) (h3 : tan^2 x + cot^2 x = 2) (h4 : tan x * cot x = 1) :
  tan^4 x + cot^4 x = 2 := by
sorry

end min_tan4_cot4_l119_119259


namespace vanya_speed_l119_119507

variable (v : ℝ)

theorem vanya_speed (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end vanya_speed_l119_119507


namespace part1_part2_i_part2_ii_l119_119742

theorem part1 :
  ¬ ∃ x : ℝ, - (4 / x) = x := 
sorry

theorem part2_i (a c : ℝ) (ha : a ≠ 0) :
  (∃! x : ℝ, x = a * (x^2) + 6 * x + c ∧ x = 5 / 2) ↔ (a = -1 ∧ c = -25 / 4) :=
sorry

theorem part2_ii (m : ℝ) :
  (∃ (a c : ℝ), a = -1 ∧ c = - 25 / 4 ∧
    ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → - (x^2) + 6 * x - 25 / 4 + 1/4 ≥ -1 ∧ - (x^2) + 6 * x - 25 / 4 + 1/4 ≤ 3) ↔
    (3 ≤ m ∧ m ≤ 5) :=
sorry

end part1_part2_i_part2_ii_l119_119742


namespace fraction_to_decimal_l119_119989

theorem fraction_to_decimal :
  (45 : ℚ) / (5 ^ 3) = 0.360 :=
by
  sorry

end fraction_to_decimal_l119_119989


namespace problem_l119_119680

theorem problem (a : ℝ) (h : a^2 - 2 * a - 2 = 0) :
  (1 - 1 / (a + 1)) / (a^3 / (a^2 + 2 * a + 1)) = 1 / 2 :=
by
  sorry

end problem_l119_119680


namespace extreme_values_of_f_l119_119697

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2 * x^2 + x

-- Stating the theorem: Given the function touches the x-axis at (1, 0)
-- Prove that the extreme values are as stated
theorem extreme_values_of_f :
  ∃ x₁ x₂ ∈ (set.Icc 0 1), f x₁ = 0 ∧ f' 0 ≤ f x₁ ∧ f x₂ = 4/27 ∧ f' x₁ ≥ f x₂ := 
sorry

end extreme_values_of_f_l119_119697


namespace probability_heads_exactly_9_of_12_l119_119015

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l119_119015


namespace percentage_apples_sold_l119_119916

theorem percentage_apples_sold (A P : ℕ) (h1 : A = 600) (h2 : A * (100 - P) / 100 = 420) : P = 30 := 
by {
  sorry
}

end percentage_apples_sold_l119_119916


namespace probability_heads_exactly_9_of_12_l119_119016

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l119_119016


namespace probability_of_9_heads_in_12_flips_l119_119044

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l119_119044


namespace modulus_of_2_plus_i_over_1_plus_2i_l119_119669

open Complex

noncomputable def modulus_of_complex_fraction : ℂ := 
  let z : ℂ := (2 + I) / (1 + 2 * I)
  abs z

theorem modulus_of_2_plus_i_over_1_plus_2i :
  modulus_of_complex_fraction = 1 := by
  sorry

end modulus_of_2_plus_i_over_1_plus_2i_l119_119669


namespace probability_heads_9_of_12_l119_119149

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l119_119149


namespace points_coplanar_iff_neg_one_l119_119257

def vectors := list (list ℝ)

noncomputable def coplanar (v1 v2 v3 : vectors) : Prop :=
  let det := matrix.det (matrix.of_lists ![v1, v2, v3]) in
  det = 0

theorem points_coplanar_iff_neg_one (b : ℝ) :
  coplanar 
    [1, 0, b] 
    [0, b, 1] 
    [b, 1, 0] → 
  b = -1 :=
begin
  sorry
end

end points_coplanar_iff_neg_one_l119_119257


namespace polar_tangent_equation_l119_119705

-- Definitions based on conditions
def circle_parametric_equation (theta : ℝ) : ℝ × ℝ := sorry  -- Define parametric circle equation
def intersection_y_axis (c : ℝ × ℝ → ℝ × ℝ) : ℝ × ℝ := sorry -- Intersection with y-axis

noncomputable def polar_equation_of_tangent_line (P : ℝ × ℝ) : (ℝ × ℝ) → ℝ := sorry  -- Polar equation of tangent

-- Problem statement
theorem polar_tangent_equation (C : ℝ × ℝ → ℝ × ℝ)
                                (P : ℝ × ℝ)
                                (hP: P = intersection_y_axis C) :
  polar_equation_of_tangent_line P = (λ (r θ : ℝ), r * cos θ) :=
sorry

end polar_tangent_equation_l119_119705


namespace probability_of_9_heads_in_12_l119_119129

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l119_119129


namespace proof_problem_l119_119672

variables {α : Type*} [add_comm_group α] [module ℚ α]

variable {a : ℕ → α}
variable {S : ℕ → α}
variable {d : α}
variable {a₁ : α}

-- Conditions
def arithmetic_sequence (a : ℕ → α) : Prop :=
∀ n, a (n + 1) - a n = d

def sum_of_terms (S : ℕ → α) (a : ℕ → α) : Prop :=
∀ n, S n = (n * (a 1 + a n) / 2 : ℚ)

axiom condition1 : S 8 - S 2 = 30

-- Question to prove
theorem proof_problem : S 10 = 50 := by
  sorry

end proof_problem_l119_119672


namespace probability_heads_9_of_12_l119_119145

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l119_119145


namespace ratio_water_duck_to_pig_l119_119974

theorem ratio_water_duck_to_pig :
  let gallons_per_minute := 3
  let pumping_minutes := 25
  let total_gallons := gallons_per_minute * pumping_minutes
  let corn_rows := 4
  let plants_per_row := 15
  let gallons_per_corn_plant := 0.5
  let total_corn_plants := corn_rows * plants_per_row
  let total_corn_water := total_corn_plants * gallons_per_corn_plant
  let pig_count := 10
  let gallons_per_pig := 4
  let total_pig_water := pig_count * gallons_per_pig
  let duck_count := 20
  let total_duck_water := total_gallons - total_corn_water - total_pig_water
  let gallons_per_duck := total_duck_water / duck_count
  let ratio := gallons_per_duck / gallons_per_pig
  ratio = 1 / 16 := 
by
  sorry

end ratio_water_duck_to_pig_l119_119974


namespace min_sum_of_factors_l119_119853

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 2310) : 
  a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l119_119853


namespace positive_integers_with_finite_multiples_l119_119658

noncomputable def has_finite_multiples_with_n_divisors (n : ℕ) : Prop :=
  ∃ (N : ℕ), ∀ (m : ℕ), m > N → ∃ k, (n * k).succ_dvd_count = n

theorem positive_integers_with_finite_multiples (n : ℕ) (hn : n > 0) :
  has_finite_multiples_with_n_divisors n ↔ (∀ p : ℕ, p.prime → p^2 ∤ n) ∨ n = 4 :=
sorry

end positive_integers_with_finite_multiples_l119_119658


namespace segment_QZ_length_l119_119374

-- Define the variables and their values from the problem conditions
variables (A Z : Point) (AZ BQ QY QZ : ℝ)
hyp : AZ = 42 
hyp2 : BQ = 12
hyp3 : QY = 24

-- Define the parallel condition
def segments_parallel (AB YZ : Segment) : Prop := AB.parallel_to YZ

-- Define the triangles similarity condition due to parallelism
def triangles_similar 
  {α β γ : Triangle}
  (h1 : α.a = β.a) (h2 : α.b = β.b) : Prop :=
α.similar_to γ

-- Formalize the Lean statement for the given problem
theorem segment_QZ_length : 
  (segments_parallel AB YZ) → 
  (AZ = 42) →
  (BQ = 12) →
  (QY = 24) →
  (QZ = 28) :=
by
  -- Proof steps would go here 
  sorry

end segment_QZ_length_l119_119374


namespace difference_of_digits_l119_119177

theorem difference_of_digits (p q : ℕ) (h1 : ∀ n, n < 100 → n ≥ 10 → ∀ m, m < 100 → m ≥ 10 → 9 * (p - q) = 9) : 
  p - q = 1 :=
sorry

end difference_of_digits_l119_119177


namespace problem_statement_l119_119279

theorem problem_statement (x y : ℝ) (h1 : x + y = 2) (h2 : xy = -2) : (1 - x) * (1 - y) = -3 := by
  sorry

end problem_statement_l119_119279


namespace ant_probability_no_collision_l119_119648
noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def derangement (n : ℕ) : ℕ :=
  factorial n * (List.range (n+1)).sum (λ k, (-1 : ℤ) ^ k / factorial k)

theorem ant_probability_no_collision : 
  let num_ants := 8
  let total_moves := 3 ^ num_ants
  let favorable_outcomes := derangement num_ants
  let probability := favorable_outcomes / total_moves
  probability = 14833 / 264241152 :=
by
  sorry

end ant_probability_no_collision_l119_119648


namespace probability_two_red_one_blue_l119_119565

theorem probability_two_red_one_blue
  (total_reds : ℕ) (total_blues : ℕ) (total_draws : ℕ)
  (h_reds : total_reds = 12) (h_blues : total_blues = 8) (h_draws : total_draws = 3) :
  let total_marbles := total_reds + total_blues in
  (total_reds * (total_reds - 1) * total_blues) / (total_marbles * (total_marbles - 1) * (total_marbles - 2)) = 44 / 95 := sorry

end probability_two_red_one_blue_l119_119565


namespace truck_speed_is_105_point_01_mph_l119_119556

-- Definitions from the conditions
def truck_length : ℝ := 88 -- in feet
def tunnel_length : ℝ := 528 -- in feet
def total_time : ℝ := 4 -- in seconds
def feet_per_mile : ℝ := 5280 -- 1 mile in feet
def seconds_per_hour : ℝ := 3600 -- 1 hour in seconds

-- Derived values
def total_distance : ℝ := tunnel_length + truck_length
def total_distance_miles : ℝ := total_distance / feet_per_mile
def total_time_hours : ℝ := total_time / seconds_per_hour

-- The theorem to prove
theorem truck_speed_is_105_point_01_mph : (total_distance_miles / total_time_hours) = 105.01 :=
by
  sorry

end truck_speed_is_105_point_01_mph_l119_119556


namespace probability_9_heads_12_flips_l119_119160

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l119_119160


namespace find_k_l119_119869

-- Given vectors
variables (k : ℕ)
def OA := (k, 12)
def OB := (4, 5)
def OC := (10, 8)

-- Collinearity condition for points A, B, and C
def collinear (A B C : ℕ × ℕ) : Prop :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  let (x3, y3) := C in
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

-- Statement to be proven
theorem find_k : collinear OA OB OC → k = 18 := 
begin
  sorry
end

end find_k_l119_119869


namespace largest_subset_count_l119_119940

def is_subset_valid (S : set ℕ) : Prop :=
  ∀ (x y : ℕ), x ∈ S → y ∈ S → (x = 4 * y ∨ y = 4 * x) → false

def largest_valid_subset (n : ℕ) :=
  {S : set ℕ | S ⊆ {1, ..., n} ∧ is_subset_valid S}

theorem largest_subset_count : ∃ S ∈ largest_valid_subset 150, S.card = 142 := sorry

end largest_subset_count_l119_119940


namespace probability_heads_in_nine_of_twelve_flips_l119_119092

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l119_119092


namespace max_distinct_license_plates_l119_119569

theorem max_distinct_license_plates
  (plates : List (Vector ℕ 6))
  (cond : ∀ p q ∈ plates, p ≠ q → ∃ i₁ i₂, i₁ ≠ i₂ ∧ p.nth i₁ ≠ q.nth i₁ ∧ p.nth i₂ ≠ q.nth i₂) :
  plates.length ≤ 100000 :=
sorry

end max_distinct_license_plates_l119_119569


namespace base12_remainder_l119_119172

def base12_to_base10 (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) : ℕ :=
  a * 12^3 + b * 12^2 + c * 12^1 + d * 12^0

theorem base12_remainder (a b c d : ℕ) 
  (h1531 : base12_to_base10 a b c d = 1 * 12^3 + 5 * 12^2 + 3 * 12^1 + 1 * 12^0):
  (base12_to_base10 a b c d) % 8 = 5 :=
by
  unfold base12_to_base10 at h1531
  sorry

end base12_remainder_l119_119172


namespace vanya_faster_speed_l119_119519

theorem vanya_faster_speed (v : ℝ) (h : v + 2 = 2.5 * v) : (v + 4) / v = 4 := by
  sorry

end vanya_faster_speed_l119_119519


namespace distance_between_lines_l119_119637

def line1 : ℝ → ℝ → Prop := λ x y, x + y - 1 = 0
def line2 : ℝ → ℝ → Prop := λ x y, 2 * x + 2 * y + 1 = 0

noncomputable def distance_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  (abs (C1 - C2)) / (real.sqrt (A * A + B * B))

theorem distance_between_lines : 
  distance_parallel_lines 2 2 (-2) 1 = (3 * real.sqrt 2) / 4 := 
by 
  sorry

end distance_between_lines_l119_119637


namespace tallest_is_first_l119_119433

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

end tallest_is_first_l119_119433


namespace joey_average_speed_l119_119755

-- Defining the parameters given in the problem
def route_distance : ℝ := 5
def time_out : ℝ := 1
def return_speed : ℝ := 20

-- Define the key term we want to prove
def round_trip_average_speed : ℝ :=
  let total_distance := 2 * route_distance
  let time_return := route_distance / return_speed
  let total_time := time_out + time_return
  total_distance / total_time

-- The main statement to be proven
theorem joey_average_speed : round_trip_average_speed = 8 := by
  sorry

end joey_average_speed_l119_119755


namespace calc_result_l119_119690

theorem calc_result: 
  let f1 : ℝ := 1/5
  let f2 : ℝ := 1/2
  (16 * f1 * 5 * f2 / 2 = 4) :=
by 
  let c1 := 16 * (1/5) * 5 * (1/2) / 2
  have : c1 = 4,
  sorry

end calc_result_l119_119690


namespace vanya_faster_speed_l119_119523

theorem vanya_faster_speed (v : ℝ) (h : v + 2 = 2.5 * v) : (v + 4) / v = 4 := by
  sorry

end vanya_faster_speed_l119_119523


namespace logarithmic_inequality_l119_119276

theorem logarithmic_inequality :
  let a := log (sqrt 2)
  let b := log (sqrt 5)
  let c := log (sqrt 3)
  a < c ∧ c < b :=
by
  let a := Real.log (Real.sqrt 2)
  let b := Real.log (Real.sqrt 5)
  let c := Real.log (Real.sqrt 3)
  sorry

end logarithmic_inequality_l119_119276


namespace probability_exactly_9_heads_l119_119115

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l119_119115


namespace percentage_cities_in_range_l119_119204

-- Definitions of percentages as given conditions
def percentage_cities_between_50k_200k : ℕ := 40
def percentage_cities_below_50k : ℕ := 35
def percentage_cities_above_200k : ℕ := 25

-- Statement of the problem
theorem percentage_cities_in_range :
  percentage_cities_between_50k_200k = 40 := 
by
  sorry

end percentage_cities_in_range_l119_119204


namespace probability_of_9_heads_in_12_l119_119127

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l119_119127


namespace resulting_figure_perimeter_l119_119591

def original_square_side : ℕ := 100

def original_square_area : ℕ := original_square_side * original_square_side

def rect1_side1 : ℕ := original_square_side
def rect1_side2 : ℕ := original_square_side / 2

def rect2_side1 : ℕ := original_square_side
def rect2_side2 : ℕ := original_square_side / 2

def new_figure_perimeter : ℕ :=
  3 * original_square_side + 4 * (original_square_side / 2)

theorem resulting_figure_perimeter :
  new_figure_perimeter = 500 :=
by {
    sorry
}

end resulting_figure_perimeter_l119_119591


namespace sector_area_correct_l119_119686

noncomputable def sector_area (r α : ℝ) : ℝ :=
  (1 / 2) * r^2 * α

theorem sector_area_correct :
  sector_area 3 2 = 9 :=
by
  sorry

end sector_area_correct_l119_119686


namespace probability_heads_9_of_12_l119_119142

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l119_119142


namespace sherman_total_weekly_driving_time_l119_119460

def daily_commute_time : Nat := 1  -- 1 hour for daily round trip commute time
def work_days : Nat := 5  -- Sherman works 5 days a week
def weekend_day_driving_time : Nat := 2  -- 2 hours of driving each weekend day
def weekend_days : Nat := 2  -- There are 2 weekend days

theorem sherman_total_weekly_driving_time :
  daily_commute_time * work_days + weekend_day_driving_time * weekend_days = 9 := 
by
  sorry

end sherman_total_weekly_driving_time_l119_119460


namespace exists_m_divisible_by_1988_l119_119415

def f (x : ℤ) : ℤ := 3 * x + 2

def f_comp (k : ℕ) (x : ℤ) : ℤ :=
  nat.iterate f k x

theorem exists_m_divisible_by_1988 :
  ∃ (m : ℕ), f_comp 100 m % 1988 = 0 :=
sorry

end exists_m_divisible_by_1988_l119_119415


namespace probability_heads_in_nine_of_twelve_flips_l119_119085

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l119_119085


namespace min_additional_coins_l119_119631

-- Given: Dan has 15 friends and 70 coins
-- Define the conditions
def num_friends : ℕ := 15
def initial_coins : ℕ := 70

-- Calculation to find the minimum number of additional coins needed
theorem min_additional_coins : 
  let total_coins_needed := (num_friends * (num_friends + 1)) / 2 in
  total_coins_needed - initial_coins = 50 :=
by
  have total_coins_needed : ℕ := (num_friends * (num_friends + 1)) / 2
  have h1 : total_coins_needed = 120 := by sorry
  have h2 : 120 - initial_coins = 50 := by sorry
  exact h2

end min_additional_coins_l119_119631


namespace smallest_positive_multiple_of_17_with_condition_l119_119543

theorem smallest_positive_multiple_of_17_with_condition :
  ∃ k : ℕ, k > 0 ∧ (k % 17 = 0) ∧ (k - 3) % 101 = 0 ∧ k = 306 :=
by
  sorry

end smallest_positive_multiple_of_17_with_condition_l119_119543


namespace cookie_problem_l119_119630

theorem cookie_problem : 
  ∃ (B : ℕ), B = 130 ∧ B - 80 = 50 ∧ B/2 + 20 = 85 :=
by
  sorry

end cookie_problem_l119_119630


namespace find_costs_functional_relationship_minimize_cost_l119_119893

-- Conditions
variables (x y w : ℝ)
def condition1 : Prop := x + 3 * y = 23
def condition2 : Prop := 2 * x = y + 4
def condition3 : Prop := (w = 5 * x + 6 * (12 - x))
def condition4 : Prop := ∀ x, x = 7 → w = -x + 72

-- Facts
theorem find_costs : condition1 → condition2 → x = 5 ∧ y = 6 :=
by
  -- This would be the proof part
  sorry

theorem functional_relationship : x + y = 12 → w = 5*x + 6*(12 - x) := 
by
  -- This would be the proof part
  sorry

theorem minimize_cost : x + y = 12 ∧ y ≥ 5 ∧ ∀ z, z = 7 → w = 65 :=
by
  -- This would be the proof part
  sorry

end find_costs_functional_relationship_minimize_cost_l119_119893


namespace painting_time_l119_119710

theorem painting_time (t₁₂ : ℕ) (h : t₁₂ = 6) (r : ℝ) (hr : r = t₁₂ / 12) (n : ℕ) (hn : n = 20) : 
  t₁₂ + n * r = 16 := by
  sorry

end painting_time_l119_119710


namespace train_crosses_platform_l119_119178

noncomputable def length_of_platform (train_length : ℝ) (train_speed_kmph : ℝ) (time_seconds : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_speed_mps * time_seconds
  total_distance - train_length

theorem train_crosses_platform :
  length_of_platform 140 84 16 ≈ 233.33 := by sorry

end train_crosses_platform_l119_119178


namespace range_func_l119_119985

noncomputable def func (x : ℝ) : ℝ := x + 4 / x

theorem range_func (x : ℝ) (hx : x ≠ 0) : func x ≤ -4 ∨ func x ≥ 4 := by
  sorry

end range_func_l119_119985


namespace linear_system_solution_correct_l119_119264

theorem linear_system_solution_correct {x y : ℝ} 
  (h1 : x = 1) 
  (h2 : y = -2) 
  (h3 : x + y = -1) 
  (h4 : x - y = 3) : 
  true :=
by {
  have : 1 + -2 = -1, by norm_num,
  have : 1 - -2 = 3, by norm_num,
  exact trivial
}

end linear_system_solution_correct_l119_119264


namespace locus_of_centroid_is_circle_with_radius_one_third_l119_119444

-- Define the problem conditions:
variables {A B C O M : Type} -- Points A, B, C, O, and M are variables of some type
variables {R : ℝ} -- R is the radius of the original circle

-- Assume points A and B are fixed on a circle and O is the midpoint of AB:
axiom A_on_circle : ∃ (circle : Circle) (radius : ℝ), radius = R ∧ circle.center O ∧ circle.contains A ∧ circle.contains B
axiom B_on_circle : ∃ (circle : Circle) (radius : ℝ), radius = R ∧ circle.contains A ∧ circle.contains B

-- Assume C is a moving point on the circle:
axiom C_moves_on_circle : ∀ (circle : Circle) (radius : ℝ), (radius = R ∧ circle.contains A ∧ circle.contains B) → ∃ (C : point), circle.contains C 

-- O is the midpoint of segment AB:
axiom O_midpoint_AB : midpoint O A B

-- M is the centroid of triangle ABC:
axiom M_centroid_ABC : centroid M A B C

-- Proof goal:
theorem locus_of_centroid_is_circle_with_radius_one_third :
  locus (λ O C, centroid A B C) = circle O (1/3 * R) :=
sorry

end locus_of_centroid_is_circle_with_radius_one_third_l119_119444


namespace inequality_proof_l119_119414

theorem inequality_proof
    (n : ℕ)
    (a : ℕ → ℝ)
    (h_pos : ∀ k, 1 ≤ k ∧ k ≤ n → 0 < a k) :
    (∑ k in finset.range n, (a (k+1)) ^ (k+1) / ((k+2).to_nat)) 
    ≤ (∑ k in finset.range n, a (k+1)) 
    + real.sqrt ((2 * (real.pi ^ 2 - 3)) / 9 * ∑ k in finset.range n, a (k+1)) :=
sorry

end inequality_proof_l119_119414


namespace probability_exactly_9_heads_in_12_flips_l119_119104

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l119_119104


namespace innkeeper_assignment_count_l119_119196

theorem innkeeper_assignment_count :
  ∃ (f : Fin 7 → Fin 5), (∀ x : Fin 5, 1 ≤ (Finset.univ.filter (λ y, f y = x)).card ∧ (Finset.univ.filter (λ y, f y = x)).card ≤ 2) ∧ Finite.card {f | true} = 12600 :=
by
  sorry

end innkeeper_assignment_count_l119_119196


namespace equation_solutions_eq_number_of_divisors_l119_119416

theorem equation_solutions_eq_number_of_divisors (n : ℕ) :
  (∃ solutions : ℕ × ℕ → Prop, (∀ (x y : ℕ), solutions (x, y) ↔ (1/x + 1/y = 1/n)) ∧ 
    (Finset.card {d : ℕ | d ∣ n^2}) = (Finset.card {p : ℕ × ℕ | solutions p})) :=
sorry

end equation_solutions_eq_number_of_divisors_l119_119416


namespace find_m_n_l119_119337

theorem find_m_n (x m n : ℝ) : (x + 4) * (x - 2) = x^2 + m * x + n → m = 2 ∧ n = -8 := 
by
  intro h
  -- Steps to prove the theorem would be here
  sorry

end find_m_n_l119_119337


namespace gnomes_in_westerville_l119_119465

variable (R W : ℕ) -- Let R be the number of gnomes in Ravenswood forest and W in Westerville woods

-- Definitions based on conditions
def condition1 : Prop := R = 4 * W
def condition2 : Prop := 0.60 * R = 48

-- The theorem to be proven given the conditions
theorem gnomes_in_westerville : condition1 ∧ condition2 → W = 20 := 
by sorry

end gnomes_in_westerville_l119_119465


namespace vanya_faster_by_4_l119_119532

variable (v : ℝ) -- initial speed in m/s
variable (school_distance : ℝ) -- distance to school in meters
variable (time_to_school : ℝ) -- time to school in seconds
variable (increased_speed : ℝ → ℝ) -- increased speed function

-- Conditions
def initial_speed : (v > 0) := by sorry
def increased_speed_by_2 {v : ℝ} : (v + 2 > 0) := by sorry
def faster_by_2_5 {v : ℝ} : (time_to_school = (school_distance / v)) → (time_to_school / 2.5 = (school_distance / (v + 2))) := by sorry

-- Main statement to prove
theorem vanya_faster_by_4 (h1 : initial_speed v) (h2 : increased_speed_by_2 v) (h3 : faster_by_2_5 v time_to_school school_distance) :
    (time_to_school / 4 = (school_distance / (v + 4))) := by
  sorry

end vanya_faster_by_4_l119_119532


namespace triangle_find_b_l119_119368

theorem triangle_find_b 
  (A B C : ℝ)
  (a b c : ℝ)
  (hABC : ∀ {A B C : ℝ}, acute_triangle A B C)
  (h_side_a : a = 2)
  (h_sin_A : sin A = (2 * sqrt 2) / 3)
  (h_area_ABC : 1/2 * b * c * sin A = sqrt 2) :
  b = sqrt 3 :=
by
  sorry

end triangle_find_b_l119_119368


namespace casper_original_candies_l119_119226

noncomputable def casper_candies : ℕ := 
  let x := 336 in
  let remaining_after_first_day := (3 / 4 : ℚ) * x - 3 in
  let remaining_after_second_day := (3 / 4 : ℚ) * remaining_after_first_day - 5 in
  let remaining_after_second_day := (1 / 4 : ℚ) * (remaining_after_second_day) in
  let remaining_after_last_day := 10 in
  have h : remaining_after_second_day = remaining_after_last_day := 10,
  x

theorem casper_original_candies :
  casper_candies = 336 :=
by sorry

end casper_original_candies_l119_119226


namespace min_sum_of_factors_of_2310_l119_119858

theorem min_sum_of_factors_of_2310 : ∃ a b c : ℕ, a * b * c = 2310 ∧ a + b + c = 52 :=
by
  sorry

end min_sum_of_factors_of_2310_l119_119858


namespace bert_money_left_l119_119610

theorem bert_money_left (initial_money : ℕ) (spent_hardware : ℕ) (spent_cleaners : ℕ) (spent_grocery : ℕ) :
  initial_money = 52 →
  spent_hardware = initial_money * 1 / 4 →
  spent_cleaners = 9 →
  spent_grocery = (initial_money - spent_hardware - spent_cleaners) / 2 →
  initial_money - spent_hardware - spent_cleaners - spent_grocery = 15 := 
by
  intros h_initial h_hardware h_cleaners h_grocery
  rw [h_initial, h_hardware, h_cleaners, h_grocery]
  sorry

end bert_money_left_l119_119610


namespace number_of_men_l119_119346

theorem number_of_men (acres1 acres2: ℝ) (days1 days2: ℝ) (men1: ℕ) (rate1 rate2: ℝ) :
  acres1 / days1 = rate1 →
  acres2 / days2 = rate2 →
  men1 / rate1 = M / rate2 →
  M = 27 :=
begin
  assume h1 h2 h3,
  rw ← h1 at h2,
  rw ← h2 at h3,
  have h4: men1 * (acres2 / days2) = M * (acres1 / days1),
  { calc
      men1 * (acres2 / days2) = M * (acres1 / days1) : by { exact h3 },
    },
  sorry
end

end number_of_men_l119_119346


namespace arithmetic_sequence_general_term_l119_119738

theorem arithmetic_sequence_general_term (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S n = 3 * n^2 + 2 * n) →
  a 1 = S 1 ∧ (∀ n ≥ 2, a n = S n - S (n - 1)) →
  ∀ n, a n = 6 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l119_119738


namespace find_B_angle_l119_119732

theorem find_B_angle
  {A B C a b c : ℝ}
  (h1 : a = c * (cos B) / (sqrt 2 * cos C))
  (h2 : B > 0)
  (h3 : B < π) :
  B = π / 4 :=
sorry

end find_B_angle_l119_119732


namespace range_f_l119_119638

def closest_multiple (k n : ℤ) : ℤ :=
  let m := 2*n + 1
  k + (n - 1) / 2 - ((k + (n - 1) / 2) % m)

def f (k : ℤ) : ℤ :=
  (closest_multiple k 1) + (closest_multiple (2*k) 2) + (closest_multiple (3*k) 3) - 6*k

theorem range_f : ∀ y ∈ set.range f, -6 ≤ y ∧ y ≤ 6 :=
sorry

end range_f_l119_119638


namespace vanya_faster_speed_l119_119516

def vanya_speed (v : ℝ) : Prop :=
  (v + 2) / v = 2.5

theorem vanya_faster_speed (v : ℝ) (h : vanya_speed v) : (v + 4) / v = 4 :=
by
  sorry

end vanya_faster_speed_l119_119516


namespace vanya_speed_problem_l119_119524

theorem vanya_speed_problem (v : ℝ) (hv : (v + 2) / v = 2.5) : ((v + 4) / v) = 4 :=
by
  have v_pos : v ≠ 0 := sorry  -- Prove that v is not zero
  have v_eq : v = 4 / 3 := sorry  -- Use hv to solve for v
  rw [v_eq] at v_eq,
  rw [v_eq]
  sorry

end vanya_speed_problem_l119_119524


namespace probability_9_heads_12_flips_l119_119153

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l119_119153


namespace quadratic_has_two_real_roots_l119_119271

-- Define the condition that the discriminant must be non-negative
def discriminant_nonneg (a b c : ℝ) : Prop := b * b - 4 * a * c ≥ 0

-- Define our specific quadratic equation conditions: x^2 - 2x + m = 0
theorem quadratic_has_two_real_roots (m : ℝ) :
  discriminant_nonneg 1 (-2) m → m ≤ 1 :=
by
  sorry

end quadratic_has_two_real_roots_l119_119271


namespace machines_work_together_l119_119870

theorem machines_work_together (x : ℝ) (h₁ : 1/(x+4) + 1/(x+2) + 1/(x+3) = 1/x) : x = 1 :=
sorry

end machines_work_together_l119_119870


namespace ellipse_foci_distance_l119_119715

theorem ellipse_foci_distance
  (P : ℝ × ℝ)
  (PF1 PF2 : ℝ)
  (x y : ℝ)
  (hP : (x, y) = P)
  (hE : x^2 / 16 + y^2 / 11 = 1)
  (hF1 : PF1 = 3)
  (h2a : 2 * real.sqrt 16 = 8) :
  PF2 = 8 - PF1 := by
    sorry

end ellipse_foci_distance_l119_119715


namespace basketballs_per_player_l119_119824

theorem basketballs_per_player (total_basketballs : ℕ) (total_players : ℕ) (h1 : total_basketballs = 242) (h2 : total_players = 22) : total_basketballs / total_players = 11 :=
by {
    rw [h1, h2],
    exact Nat.div_eq_of_eq_mul Nat.mul_comm,
}

end basketballs_per_player_l119_119824


namespace find_tanθ_l119_119831

def f (x : ℝ) : ℝ := (1 / 3) * (Real.sin x)^4 + (1 / 4) * (Real.cos x)^4

theorem find_tanθ (θ : ℝ) (h : f θ = 1 / 7) : Real.tan θ = Real.sqrt (3 / 4) ∨ Real.tan θ = -Real.sqrt (3 / 4) :=
by
  sorry

end find_tanθ_l119_119831


namespace center_of_symmetry_on_interval_l119_119874

def f (x : ℝ) : ℝ := -Real.cos (x / 2)
def f' (x : ℝ) : ℝ := (1 / 2) * Real.sin (x / 2)
def g (x : ℝ) : ℝ := f(x) / f'(x)

theorem center_of_symmetry_on_interval :
  ∃ x, (0 ≤ x ∧ x ≤ 2 * Real.pi) ∧ (g' x = 0) ∧ (g x = 0) :=
sorry

end center_of_symmetry_on_interval_l119_119874


namespace probability_heads_9_of_12_is_correct_l119_119029

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l119_119029


namespace standard_equation_of_ellipse_l119_119562

-- Define the conditions of the ellipse
def ellipse_condition_A (m n : ℝ) : Prop := n * (5 / 3) ^ 2 = 1
def ellipse_condition_B (m n : ℝ) : Prop := m + n = 1

-- The theorem to prove the standard equation of the ellipse
theorem standard_equation_of_ellipse (m n : ℝ) (hA : ellipse_condition_A m n) (hB : ellipse_condition_B m n) :
  m = 16 / 25 ∧ n = 9 / 25 :=
sorry

end standard_equation_of_ellipse_l119_119562


namespace f_nplus1_expr_F_nplus1_expr_l119_119896

-- Define the function f recursively
def f : ℕ → ℕ → ℕ
| 1 := λ x, 1
| 2 := λ x, x
| (n+3) := λ x, x * f (n+2) x + f (n+1) x

-- Define the binomial coefficient (combinations)
def C : ℕ → ℕ → ℕ
| n 0 := 1
| 0 k := 0
| (n+1) (k+1) := C n (k+1) + C n k

-- Prove the first statement for f
theorem f_nplus1_expr (n : ℕ) (x : ℕ) :
  f (n+1) x = x^n + (C (n-1) 1 * x^(n-2)) + (C (n-2) 2 * x^(n-4)) + 
             (C (n-3) 3 * x^(n-6)) + ... :=
sorry

-- Define the function F
def F : ℕ → ℕ
| n := f n 1

-- Prove the second statement for F
theorem F_nplus1_expr (n : ℕ) :
  F (n+1) = 1 + (C (n-1) 1) + (C (n-2) 2) + (C (n-3) 3) + ... :=
sorry

end f_nplus1_expr_F_nplus1_expr_l119_119896


namespace probability_exactly_9_heads_in_12_flips_l119_119108

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l119_119108


namespace sum_of_interior_angles_l119_119815

theorem sum_of_interior_angles (n : ℕ) (h : n ≥ 3) (simple_polygon : SimplePolygon n) :
  sum_of_interior_angles simple_polygon = (n - 2) * 180 :=
sorry

end sum_of_interior_angles_l119_119815


namespace max_subset_cardinality_146_l119_119938

noncomputable def max_subset_cardinality : ℕ :=
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 150 ∧ (∀ m ∈ (1:150), (n = 4 * m) → false) } in
  S.card = 146

theorem max_subset_cardinality_146 :
  ∃ S : set ℕ, (∀ x ∈ S, 1 ≤ x ∧ x ≤ 150) ∧
               (∀ x y ∈ S, x ≠ y → x ≠ 4 * y ∧ y ≠ 4 * x) ∧
               S.card = 146 := 
sorry

end max_subset_cardinality_146_l119_119938


namespace zero_in_interval_x5_y8_l119_119798

theorem zero_in_interval_x5_y8
    (x y : ℝ)
    (h1 : x^5 < y^8)
    (h2 : y^8 < y^3)
    (h3 : y^3 < x^6)
    (h4 : x < 0)
    (h5 : 0 < y)
    (h6 : y < 1) :
    0 ∈ set.Ioo (x^5) (y^8) :=
by
  sorry

end zero_in_interval_x5_y8_l119_119798


namespace fair_die_probability_at_least_five_5_or_more_in_six_rolls_l119_119914

noncomputable def probability_at_least_five (n : ℕ) (k : ℕ) : ℚ :=
(binomial n k) * (1/3)^k * (2/3)^(n-k)

theorem fair_die_probability_at_least_five_5_or_more_in_six_rolls :
  (probability_at_least_five 6 5 + probability_at_least_five 6 6) = 13/729 :=
by 
  -- This is the statement proof step.
  sorry

end fair_die_probability_at_least_five_5_or_more_in_six_rolls_l119_119914


namespace probability_heads_exactly_9_of_12_l119_119019

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l119_119019


namespace blocks_on_chessboard_l119_119901

theorem blocks_on_chessboard (n : ℕ) (h_size : n = 8) (h_cut : 2 = 2) :
  let total_squares := n * n,
      remaining_squares := total_squares - 2,
      block_size := 2,
      blocks_horizontally := (6 * 4) + (2 * 3),
      blocks_vertically := (6 * 4) + (2 * 3),
      total_blocks := blocks_horizontally + blocks_vertically
  in total_blocks / 2 = 30 := 
by
  sorry

end blocks_on_chessboard_l119_119901


namespace probability_of_9_heads_in_12_flips_l119_119040

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l119_119040


namespace vanya_speed_problem_l119_119525

theorem vanya_speed_problem (v : ℝ) (hv : (v + 2) / v = 2.5) : ((v + 4) / v) = 4 :=
by
  have v_pos : v ≠ 0 := sorry  -- Prove that v is not zero
  have v_eq : v = 4 / 3 := sorry  -- Use hv to solve for v
  rw [v_eq] at v_eq,
  rw [v_eq]
  sorry

end vanya_speed_problem_l119_119525


namespace sharks_at_other_beach_is_12_l119_119971

-- Define the conditions
def cape_may_sharks := 32
def sharks_other_beach (S : ℕ) := 2 * S + 8

-- Statement to prove
theorem sharks_at_other_beach_is_12 (S : ℕ) (h : cape_may_sharks = sharks_other_beach S) : S = 12 :=
by
  -- Sorry statement to skip the proof part
  sorry

end sharks_at_other_beach_is_12_l119_119971


namespace probability_two_green_after_red_l119_119387

-- Define the conditions as hypotheses and state the theorem
theorem probability_two_green_after_red :
  let total_apples := 10
  let red_apples := 5
  let green_apples := 5
  let special_red_picked := red_apples - 1 
  let apples_remaining := total_apples - 1
  let remaining_red_apples := red_apples - 1
  let remaining_green_apples := green_apples
  let total_ways_to_pick_two := Nat.choose (apples_remaining) 2
  let ways_to_pick_two_green := Nat.choose (remaining_green_apples) 2
  let probability := (ways_to_pick_two_green : ℚ) / total_ways_to_pick_two
  in probability = 1 / 6 :=
by
  sorry

end probability_two_green_after_red_l119_119387


namespace probability_exactly_9_heads_l119_119113

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l119_119113


namespace smallest_number_divisible_by_55_with_117_divisors_l119_119838

theorem smallest_number_divisible_by_55_with_117_divisors :
  ∃ (a : ℕ), (a = 2^12 * 5^2 * 11^2) ∧ Nat.gcd a 55 = 55 ∧ (Nat.divisors a).length = 117 ∧ a = 12390400 :=
by {
  sorry
}

end smallest_number_divisible_by_55_with_117_divisors_l119_119838


namespace chad_total_spend_on_ice_l119_119622

-- Define the given conditions
def num_people : ℕ := 15
def pounds_per_person : ℕ := 2
def pounds_per_bag : ℕ := 1
def price_per_pack : ℕ := 300 -- Price in cents to avoid floating-point issues
def bags_per_pack : ℕ := 10

-- The main statement to prove
theorem chad_total_spend_on_ice : 
  (num_people * pounds_per_person * 100 / (pounds_per_bag * bags_per_pack) * price_per_pack / 100 = 9) :=
by sorry

end chad_total_spend_on_ice_l119_119622


namespace probability_heads_exactly_9_of_12_l119_119020

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l119_119020


namespace probability_heads_9_of_12_l119_119147

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l119_119147


namespace student_mistake_difference_l119_119210

def number : ℝ := 100.00000000000003

def correct_answer : ℝ := (4 / 5) * number

def incorrect_answer : ℝ := (5 / 4) * number

theorem student_mistake_difference :
  incorrect_answer - correct_answer = 45.00000000000002 := 
sorry

end student_mistake_difference_l119_119210


namespace river_length_l119_119473

theorem river_length (S C : ℝ) (h1 : S = C / 3) (h2 : S + C = 80) : S = 20 :=
by 
  sorry

end river_length_l119_119473


namespace probability_heads_in_9_of_12_flips_l119_119081

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l119_119081


namespace probability_heads_9_of_12_l119_119141

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l119_119141


namespace probability_heads_9_of_12_flips_l119_119001

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l119_119001


namespace findCostPrices_l119_119589

def costPriceOfApple (sp_a : ℝ) (cp_a : ℝ) : Prop :=
  sp_a = (5 / 6) * cp_a

def costPriceOfOrange (sp_o : ℝ) (cp_o : ℝ) : Prop :=
  sp_o = (3 / 4) * cp_o

def costPriceOfBanana (sp_b : ℝ) (cp_b : ℝ) : Prop :=
  sp_b = (9 / 8) * cp_b

theorem findCostPrices (sp_a sp_o sp_b : ℝ) (cp_a cp_o cp_b : ℝ) :
  costPriceOfApple sp_a cp_a → 
  costPriceOfOrange sp_o cp_o → 
  costPriceOfBanana sp_b cp_b → 
  sp_a = 20 → sp_o = 15 → sp_b = 6 → 
  cp_a = 24 ∧ cp_o = 20 ∧ cp_b = 16 / 3 :=
by 
  intro h1 h2 h3 sp_a_eq sp_o_eq sp_b_eq
  -- proof goes here
  sorry

end findCostPrices_l119_119589


namespace arthur_spending_l119_119216

noncomputable def appetizer_price : ℝ := 8
noncomputable def entree_price : ℝ := 30
noncomputable def wine_price : ℝ := 4
noncomputable def dessert_price : ℝ := 7
noncomputable def entree_discount : ℝ := 0.4
noncomputable def appetizer_discount_limit : ℝ := 10
noncomputable def complimentary_dessert_wine_count : ℕ := 2
noncomputable def overall_discount : ℝ := 0.1
noncomputable def tax_rate : ℝ := 0.08
noncomputable def waiter_tip_rate : ℝ := 0.2
noncomputable def busser_tip_rate : ℝ := 0.05

noncomputable def total_spent (appetizer_price : ℝ) (entree_price : ℝ) (wine_price : ℝ)
  (dessert_price : ℝ) (entree_discount : ℝ) (complimentary_dessert_wine_count : ℕ)
  (overall_discount : ℝ) (tax_rate : ℝ) (waiter_tip_rate : ℝ) (busser_tip_rate : ℝ) : ℝ :=
let appetizer_cost := 0 in
let entree_cost := entree_price * (1 - entree_discount) in
let wine_cost := wine_price * complimentary_dessert_wine_count in
let dessert_cost := 0 in
let subtotal := appetizer_cost + entree_cost + wine_cost in
let discounted_total := subtotal * (1 - overall_discount) in
let tax := discounted_total * tax_rate in
let total_with_tax := discounted_total + tax in
let original_total := appetizer_price + entree_price + wine_cost + dessert_price in
let original_tax := original_total * tax_rate in
let original_total_with_tax := original_total + original_tax in
let waiter_tip := original_total_with_tax * waiter_tip_rate in
let total_with_waiter_tip := total_with_tax + waiter_tip in
let busser_tip := total_with_waiter_tip * busser_tip_rate in
total_with_tax + waiter_tip + busser_tip

theorem arthur_spending : total_spent appetizer_price entree_price wine_price 
  dessert_price entree_discount complimentary_dessert_wine_count overall_discount 
  tax_rate waiter_tip_rate busser_tip_rate = 38.556 := 
sorry

end arthur_spending_l119_119216


namespace vanya_speed_l119_119503

variable (v : ℝ)

theorem vanya_speed (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end vanya_speed_l119_119503


namespace concyclic_E_C_M_D_l119_119298

open_locale classical

variables {O A B C D E F M : Type}
variable [metric_space O]

-- Definition of what it means to be on the circumference and configuration
variables {circ : O}
variables (on_circ : ∀ {X : O}, X = A → X = B → X ∈ circle (A + B) )
variables (diameter : A - B = 0)

variables (C_on_circ : C ∈ circle (A + B))
variables (D_on_circ : D ∈ circle (A + B))

-- Definition of tangents intersecting at E
variables {tangent_C : O → O}
variables {tangent_D : O → O}
variables (tangent_intersects : tangent_C C = tangent_D D → tangent_C = E)

-- Definition of intersection points
variables (AD_BC_intersect : ∃ F, intersection (AD) (BC) = F)
variables (EF_AB_intersect : ∃ M, intersection (EF) (AB) = M)

-- Proof goal: Points \( E \), \( C \), \( M \), and \( D \) are concyclic.
theorem concyclic_E_C_M_D :
  is_cyclic E C M D :=
begin
  sorry
end

end concyclic_E_C_M_D_l119_119298


namespace solution_set_inequality_l119_119698

noncomputable def f (x : ℝ) : ℝ := abs x * (10^x - 10^(-x))

theorem solution_set_inequality : {x : ℝ | f(1 - 2 * x) + f(3) > 0} = set.Iio 2 :=
by
  sorry

end solution_set_inequality_l119_119698


namespace bob_calories_consumed_l119_119222

theorem bob_calories_consumed 
  (total_slices : ℕ)
  (half_slices : ℕ)
  (calories_per_slice : ℕ) 
  (H1 : total_slices = 8) 
  (H2 : half_slices = total_slices / 2) 
  (H3 : calories_per_slice = 300) : 
  half_slices * calories_per_slice = 1200 := 
by 
  sorry

end bob_calories_consumed_l119_119222


namespace fewer_games_attended_l119_119243

-- Define the conditions
def games_played_each_year : ℕ := 20
def attendance_first_year_fraction : ℚ := 0.90
def games_attended_second_year : ℕ := 14
-- Calculate number of games attended in the first year
def games_attended_first_year : ℕ := (attendance_first_year_fraction * games_played_each_year).toNat

-- Define the correct answer using the conditions
theorem fewer_games_attended : games_attended_first_year - games_attended_second_year = 4 := by
  -- Sorry placeholder as no proof is needed
  sorry

end fewer_games_attended_l119_119243


namespace quadratic_function_m_value_l119_119700

theorem quadratic_function_m_value :
  ∃ m : ℝ, (m - 3 ≠ 0) ∧ (m^2 - 7 = 2) ∧ m = -3 :=
by
  sorry

end quadratic_function_m_value_l119_119700


namespace Oslo_Bank_Coin_Problem_l119_119823

theorem Oslo_Bank_Coin_Problem (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ 2 * n) (h3 : n ≤ k) (h4 : k ≤ Nat.ceil (3 * n / 2)) :
  ∃ (initial_ordering : list ℕ) (A B : list ℕ), 
  (initial_ordering.length = 2 * n) ∧ 
  (A.length = n) ∧ 
  (B.length = n) ∧ 
  (initial_ordering = A ++ B) :=
sorry

end Oslo_Bank_Coin_Problem_l119_119823


namespace shortest_chord_length_through_D_l119_119442

noncomputable def circle_radius : ℝ := 5
noncomputable def distance_OD : ℝ := 3

theorem shortest_chord_length_through_D (O D : Type) [MetricSpace O] [MetricSpace D] 
  (r : ℝ := circle_radius) (d : ℝ := distance_OD) :
  ∃ AB : ℝ, ∀ chord : ℝ, chord = 2 * sqrt(r^2 - d^2) → AB = 8cm :=
sorry

end shortest_chord_length_through_D_l119_119442


namespace isosceles_trapezoid_AB_over_CD_eq_13_over_3_l119_119384

theorem isosceles_trapezoid_AB_over_CD_eq_13_over_3
  (AB CD : ℝ) (AB_parallel_CD : AB = CD)
  (exists_tangent_circle : ∃ Γ, circle_Γ_tangent_to_sides_of_trapezoid_ABCDA)
  (T : point) (T_on_BC : T ∈ (circle_Γ ∩ BC))
  (P : point) (P_on_AT : P ∈ (circle_Γ ∩ AT)) (P_ne_T : P ≠ T)
  (AP AT : ℝ) (ratio_AP_over_AT : AP / AT = 2 / 5)
  : AB / CD = 13 / 3 := 
by 
  sorry

end isosceles_trapezoid_AB_over_CD_eq_13_over_3_l119_119384


namespace sum_of_slope_and_intercept_l119_119805

theorem sum_of_slope_and_intercept (x1 x2 : ℝ) (h : x1 ≠ x2) : 
  let m := (15 - 13) / (x2 - x1)
  let b := 13
  m + b = 13 + 2 / (x2 - x1) :=
by
  let m := (15 - 13) / (x2 - x1)
  let b := 13
  have h1 : m = 2 / (x2 - x1) := rfl
  have h2 : b = 13 := rfl
  rw [h1, h2]
  sorry

end sum_of_slope_and_intercept_l119_119805


namespace volume_of_prism_l119_119588

theorem volume_of_prism (l w h : ℝ) (hlw : l * w = 15) (hwh : w * h = 20) (hlh : l * h = 24) : l * w * h = 60 := 
sorry

end volume_of_prism_l119_119588


namespace equation_of_tangent_circle_l119_119301

/-- Lean Statement for the circle problem -/
theorem equation_of_tangent_circle (center_C : ℝ × ℝ)
    (h1 : ∃ x, center_C = (x, 0) ∧ x - 0 + 1 = 0)
    (circle_tangent : ∃ r, ((2 - (center_C.1))^2 + (3 - (center_C.2))^2 = (2 * Real.sqrt 2) + r)) :
    ∃ r, (x + 1)^2 + y^2 = r^2 := 
sorry

end equation_of_tangent_circle_l119_119301


namespace probability_heads_9_of_12_is_correct_l119_119032

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l119_119032


namespace river_length_l119_119474

theorem river_length (S C : ℝ) (h1 : S = C / 3) (h2 : S + C = 80) : S = 20 :=
by 
  sorry

end river_length_l119_119474


namespace sum_of_integers_with_abs_less_than_2005_l119_119866

theorem sum_of_integers_with_abs_less_than_2005 :
  (Finset.sum (Finset.filter (λ x, |x| < 2005) (Finset.Icc (-2004) 2004)) id) = 0 :=
by
  sorry

end sum_of_integers_with_abs_less_than_2005_l119_119866


namespace original_ratio_l119_119878

theorem original_ratio (x y : ℤ) (h₁ : y = 72) (h₂ : (x + 6) / y = 1 / 3) : y / x = 4 := 
by
  sorry

end original_ratio_l119_119878


namespace lim_M_n_l119_119629

-- Define the region Ω_n enclosed by the ellipse
def region_omega_n (n : ℕ) : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in x^2 / 4 + n * y^2 / (4 * n + 1) = 1 }

-- Define the maximum value of x + y for a point (x, y) on Ω_n
def M_n (n : ℕ) : ℝ :=
  Real.sup {x + y | (x, y) ∈ region_omega_n n }

-- Statement of the problem
theorem lim_M_n (M : ℕ → ℝ) : 
  (∀ n, M n = M_n n) →
  (∃ L : ℝ, tendsto M at_top (𝓝 L) ∧ L = 2 * Real.sqrt 2) :=
by
  sorry

end lim_M_n_l119_119629


namespace cube_planes_divide_space_l119_119480

theorem cube_planes_divide_space (cube : ℝ^3) (faces : list (ℝ^3 → ℝ)) :
  (faces.length = 6) → 
  (∀ face ∈ faces, ∃ a b c d : ℝ, face = λ x, a * x.1 + b * x.2 + c * x.3 + d) →
  (∀ face1 face2 ∈ faces, face1 ≠ face2 → ∀ x y : ℝ^3, face1 x = 0 → face2 y = 0 → x ≠ y) →
  (number_of_divided_parts_by_planes faces = 27) :=
by
  intros h_faces_length h_planes_presence h_planes_intersection
  sorry

end cube_planes_divide_space_l119_119480


namespace possible_values_of_S_l119_119762

noncomputable section

-- Define a point on the grid as a pair of integers
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define a snake as a list of points
def is_snake (points : List Point) : Prop :=
  points.length % 2 = 0 ∧
  ∀ i, i < points.length - 1 → 
    ((points.get i).x = (points.get (i+1)).x ∧ (points.get i).y ≠ (points.get (i+1)).y) ∨ 
    ((points.get i).y = (points.get (i+1)).y ∧ (points.get i).x ≠ (points.get (i+1)).x)

-- Sum coordinates of points in a list
def sum_coords (points : List Point) : ℤ :=
  points.foldr (λ p acc => p.x + p.y + acc) 0

-- Calculate the value S given a list of points forming a snake
def calculate_S (points : List Point) : ℤ :=
  let odds := points.filterWithIndex (λ i _ => i % 2 = 0)
  let evens := points.filterWithIndex (λ i _ => i % 2 = 1)
  sum_coords odds - sum_coords evens

-- Define possible values S can take for given n
def possible_S (n : ℕ) : set ℤ :=
  {S | ∃ k, k ∈ (List.range (2 * n + 1)).filter (λ x => x % 2 = n % 2) ∧ S = k - int.of_nat n}

-- Main theorem to be proved
theorem possible_values_of_S (n : ℕ) (points : List Point) 
  (h_snake : is_snake points) (h_len : points.length = 2 * n) : 
  calculate_S points ∈ possible_S n := 
sorry

end possible_values_of_S_l119_119762


namespace infinite_primes_dividing_polynomial_l119_119632

theorem infinite_primes_dividing_polynomial (P : ℤ[X]) (hdeg : P.degree ≥ 1) :
  ∃ᶠ p in Nat.primes, ∃ n : ℕ+, p ∣ P.eval n :=
sorry

end infinite_primes_dividing_polynomial_l119_119632


namespace min_sum_factors_l119_119850

theorem min_sum_factors (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prod : a * b * c = 2310) : a + b + c = 40 :=
sorry

end min_sum_factors_l119_119850


namespace gcd_is_3_l119_119997

def gcd_6273_14593 : ℕ := Nat.gcd 6273 14593

theorem gcd_is_3 : gcd_6273_14593 = 3 :=
by
  sorry

end gcd_is_3_l119_119997


namespace shortest_path_length_l119_119379

-- Define the points A, D and the center of the circle O
def A : ℝ × ℝ := (0, 0)
def D : ℝ × ℝ := (10, 20)
def O : ℝ × ℝ := (5, 10)

-- Circle defined by (x - 5)^2 + (y - 10)^2 = 16
def circle : ℝ × ℝ → Prop := 
  λ p, (p.1 - 5)^2 + (p.2 - 10)^2 = 16

-- Distance calculation function
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Calculate the cosine inverse length
noncomputable def cos_inv_len : ℝ :=
  4 * real.arccos(4 / (5 * real.sqrt 5))

-- Shortest path length
noncomputable def shortest_path_len : ℝ :=
  2 * real.sqrt 109 + cos_inv_len

-- The theorem statement
theorem shortest_path_length :
  (dist A O = 5 * real.sqrt 5) →
  (dist (5 - 4) A = 4) →
  (dist (5 - real.sqrt 109), D = sqrt 109) →
  shortest_path_len = 2 * real.sqrt 109 + 4 * real.arccos ( 4 / (5 * real.sqrt 5)) :=
sorry

end shortest_path_length_l119_119379


namespace julia_dimes_l119_119393

theorem julia_dimes : ∃ (d : ℕ), 20 < d ∧ d < 200 ∧ (d % 6 = 1) ∧ (d % 7 = 1) ∧ (d % 8 = 1) ∧ d = 169 :=
by {
    use 169,
    split,
    norm_num,
    split,
    norm_num,
    split,
    norm_num,
    split,
    norm_num,
    norm_num,
    sorry
}

end julia_dimes_l119_119393


namespace A_worked_alone_after_B_left_l119_119903

/-- A and B can together finish a work in 40 days. They worked together for 10 days and then B left.
    A alone can finish the job in 80 days. We need to find out how many days did A work alone after B left. -/
theorem A_worked_alone_after_B_left
  (W : ℝ)
  (A_work_rate : ℝ := W / 80)
  (B_work_rate : ℝ := W / 80)
  (AB_work_rate : ℝ := W / 40)
  (work_done_together_in_10_days : ℝ := 10 * (W / 40))
  (remaining_work : ℝ := W - work_done_together_in_10_days)
  (A_rate_alone : ℝ := W / 80) :
  ∃ D : ℝ, D * (W / 80) = remaining_work → D = 60 :=
by
  sorry

end A_worked_alone_after_B_left_l119_119903


namespace arith_seq_formula_l119_119744

noncomputable def arith_seq (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n + a (n + 2) = 4 * n + 6

theorem arith_seq_formula (a : ℕ → ℤ) (h : arith_seq a) : ∀ n : ℕ, a n = 2 * n + 1 :=
by
  intros
  sorry

end arith_seq_formula_l119_119744


namespace max_a_value_l119_119377

theorem max_a_value (a : ℝ)
  (H : ∀ x : ℝ, (x - 1) * x - (a - 2) * (a + 1) ≥ 1) :
  a ≤ 3 / 2 := by
  sorry

end max_a_value_l119_119377


namespace cone_volume_l119_119307

theorem cone_volume {r h : ℝ} (h₁ : ∃ r, 2 * ℝ.pi * r = ℝ.pi * 2 ∧ r = 1)
  (h₂ : h = Real.sqrt (2^2 - 1^2))
  (h₃ : r = 1) :
  (1/3) * ℝ.pi * r^2 * h = (Real.sqrt 3 / 3) * ℝ.pi :=
by sorry

end cone_volume_l119_119307


namespace max_roses_purchase_l119_119182

/--
Given three purchasing options for roses:
1. Individual roses cost $5.30 each.
2. One dozen (12) roses cost $36.
3. Two dozen (24) roses cost $50.
Given a total budget of $680, prove that the maximum number of roses that can be purchased is 317.
-/
noncomputable def max_roses : ℝ := 317

/--
Prove that given the purchasing options and the budget, the maximum number of roses that can be purchased is 317.
-/
theorem max_roses_purchase (individual_cost dozen_cost two_dozen_cost budget : ℝ) 
  (h1 : individual_cost = 5.30) 
  (h2 : dozen_cost = 36) 
  (h3 : two_dozen_cost = 50) 
  (h4 : budget = 680) : 
  max_roses = 317 := 
sorry

end max_roses_purchase_l119_119182


namespace calculate_B_plus_D_l119_119981

def polynomial_p (z : ℂ) : ℂ := z^4 - 6*z^3 + 11*z^2 - 6*z + 1

def f (z : ℂ) : ℂ := -2 * complex.I * conj z

-- Conditions from Vieta's formulas:
def sum_roots (z1 z2 z3 z4 : ℂ) : Prop := z1 + z2 + z3 + z4 = 6
def sum_products_root_pairs (z1 z2 z3 z4 : ℂ) : Prop := 
  z1*z2 + z1*z3 + z1*z4 + z2*z3 + z2*z4 + z3*z4 = 11
def product_roots (z1 z2 z3 z4 : ℂ) : Prop := z1*z2*z3*z4 = 1

-- New polynomial Q(z) roots:
def new_roots (z1 z2 z3 z4 : ℂ) : list ℂ := 
  [f z1, f z2, f z3, f z4]

-- Vieta's formulas for Q(z):
def B (z1 z2 z3 z4 : ℂ) : ℂ :=
  (-2 * complex.I)^2 * conj (z1*z2 + z1*z3 + z1*z4 + z2*z3 + z2*z4 + z3*z4)
def D (z1 z2 z3 z4 : ℂ) : ℂ := 
  (-2 * complex.I)^4 * conj (z1*z2*z3*z4)

theorem calculate_B_plus_D (z1 z2 z3 z4 : ℂ) 
  (h_sum_roots : sum_roots z1 z2 z3 z4)
  (h_sum_products_root_pairs : sum_products_root_pairs z1 z2 z3 z4)
  (h_product_roots : product_roots z1 z2 z3 z4) : 
  B z1 z2 z3 z4 + D z1 z2 z3 z4 = 60 :=
by 
  sorry

end calculate_B_plus_D_l119_119981


namespace min_power_sum_two_l119_119563

theorem min_power_sum_two (x y a : ℝ) (m : ℕ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y = a) (hm : 1 ≤ m) :
  (∀ x y, (x ≥ 0) ∧ (y ≥ 0) ∧ (x + y = a) → ∃ c, c = x^m + y^m ∧ c ≥ (a^m / 2^(m-1))) :=
begin
  sorry
end

end min_power_sum_two_l119_119563


namespace shares_owned_l119_119906

theorem shares_owned (expected_earnings dividend_ratio additional_per_10c actual_earnings total_dividend : ℝ)
  ( h1 : expected_earnings = 0.80 )
  ( h2 : dividend_ratio = 0.50 )
  ( h3 : additional_per_10c = 0.04 )
  ( h4 : actual_earnings = 1.10 )
  ( h5 : total_dividend = 156.0 ) :
  ∃ shares : ℝ, shares = total_dividend / (expected_earnings * dividend_ratio + (max ((actual_earnings - expected_earnings) / 0.10) 0) * additional_per_10c) ∧ shares = 300 := 
sorry

end shares_owned_l119_119906


namespace exists_seq_nat_lcm_decreasing_l119_119241

-- Natural number sequence and conditions
def seq_nat_lcm_decreasing : Prop :=
  ∃ (a : Fin 100 → ℕ), 
  ((∀ i j : Fin 100, i < j → a i < a j) ∧
  (∀ (i : Fin 99), Nat.lcm (a i) (a (i + 1)) > Nat.lcm (a (i + 1)) (a (i + 2))))

theorem exists_seq_nat_lcm_decreasing : seq_nat_lcm_decreasing :=
  sorry

end exists_seq_nat_lcm_decreasing_l119_119241


namespace probability_heads_in_9_of_12_flips_l119_119075

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l119_119075


namespace prove_a_eq_b_l119_119769

theorem prove_a_eq_b 
    (a b : ℕ) 
    (h_pos : a > 0 ∧ b > 0) 
    (h_multiple : ∃ k : ℤ, a^2 + a * b + 1 = k * (b^2 + b * a + 1)) : 
    a = b := 
sorry

end prove_a_eq_b_l119_119769


namespace probability_of_9_heads_in_12_flips_l119_119060

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l119_119060


namespace total_origami_stars_l119_119764

def typeA_initial : ℕ := 3
def typeA_later : ℕ := 5
def typeA_capacity : ℕ := 30
def typeB : ℕ := 4
def typeB_capacity : ℕ := 50
def typeC : ℕ := 2
def typeC_capacity : ℕ := 70

theorem total_origami_stars (typeA_initial typeA_later typeA_capacity typeB typeB_capacity typeC typeC_capacity : ℕ) :
  typeA_initial * typeA_capacity + typeA_later * typeA_capacity + typeB * typeB_capacity + typeC * typeC_capacity = 580 := by
  -- The following values are substituted into the formula from the conditions
  let total_stars_A := (typeA_initial + typeA_later) * typeA_capacity
  let total_stars_B := typeB * typeB_capacity
  let total_stars_C := typeC * typeC_capacity
  let total_stars := total_stars_A + total_stars_B + total_stars_C
  have h : total_stars = 580 := by rfl -- Given solution result
  exact h

#eval total_origami_stars 3 5 30 4 50 2 70

end total_origami_stars_l119_119764


namespace combination_binomial_proof_l119_119678

theorem combination_binomial_proof
  (h1 : nat.choose 18 11 = 31824)
  (h2 : nat.choose 18 10 = 18564)
  (h3 : nat.choose 20 13 = 77520) :
  nat.choose 19 13 = 27132 :=
begin
  sorry
end

end combination_binomial_proof_l119_119678


namespace mart_income_percentage_j_l119_119427

variables (J T M : ℝ)

-- condition: Tim's income is 40 percent less than Juan's income
def tims_income := T = 0.60 * J

-- condition: Mart's income is 40 percent more than Tim's income
def marts_income := M = 1.40 * T

-- goal: Prove that Mart's income is 84 percent of Juan's income
theorem mart_income_percentage_j (J : ℝ) (T : ℝ) (M : ℝ)
  (h1 : T = 0.60 * J) 
  (h2 : M = 1.40 * T) : 
  M = 0.84 * J := 
sorry

end mart_income_percentage_j_l119_119427


namespace jane_winning_probability_l119_119756

def spinner_numbers := {n : Nat // 1 ≤ n ∧ n ≤ 6}

def jane_wins (j b : spinner_numbers) : Prop :=
  abs (j.val - b.val) ≤ 3

def total_outcomes := 6 * 6

noncomputable def probability_jane_wins : ℚ :=
  5 / 6

theorem jane_winning_probability : 
  (∑ j b, jane_wins j b) / total_outcomes = probability_jane_wins := 
by
  sorry

end jane_winning_probability_l119_119756


namespace sqrt_square_eq_self_l119_119975

theorem sqrt_square_eq_self (n : ℝ) (h : n ≥ 0) : (Real.sqrt n)^2 = n := by
  sorry

example : (Real.sqrt 978121)^2 = 978121 := by
  apply sqrt_square_eq_self
  norm_num

end sqrt_square_eq_self_l119_119975


namespace exists_pythagorean_triple_rational_k_l119_119781

theorem exists_pythagorean_triple_rational_k (k : ℚ) (hk : k > 1) :
  ∃ (a b c : ℕ), (a^2 + b^2 = c^2) ∧ ((a + c : ℚ) / b = k) := by
  sorry

end exists_pythagorean_triple_rational_k_l119_119781


namespace cosine_between_A_B_and_A_C_l119_119995

open Real

def point := (ℝ × ℝ × ℝ) -- define a point in 3D space

-- Given points A, B, and C
def A : point := (-4, 0, 4)
def B : point := (-1, 6, 7)
def C : point := (1, 10, 9)

-- Define a function to compute the vector between two points
def vec (P Q : point) : point := (Q.1 - P.1, Q.2 - P.2, Q.3 - P.3)

-- Define dot product of two vectors
def dot_product (v w : point) : ℝ := v.1 * w.1 + v.2 * w.2 + v.3 * w.3

-- Define the magnitude of a vector
def magnitude (v : point) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- Prove the cosine of the angle between vectors AB and AC is 1
theorem cosine_between_A_B_and_A_C :
  let AB := vec A B;
      AC := vec A C in
  cos_angle AB AC = 1 :=
by
  let AB := vec A B;
  let AC := vec A C;
  let cos_angle (v w : point) := dot_product v w / (magnitude v * magnitude w);
  sorry

end cosine_between_A_B_and_A_C_l119_119995


namespace probability_heads_9_of_12_flips_l119_119003

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l119_119003


namespace probability_of_9_heads_in_12_flips_l119_119048

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l119_119048


namespace midpoint_product_l119_119772

theorem midpoint_product (x y z : ℤ) 
  (h1 : (2 + x) / 2 = 4) 
  (h2 : (10 + y) / 2 = 6) 
  (h3 : (5 + z) / 2 = 3) : 
  x * y * z = 12 := 
by
  sorry

end midpoint_product_l119_119772


namespace probability_heads_exactly_9_of_12_l119_119022

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l119_119022


namespace limit_of_s_l119_119312

noncomputable def M (m : ℝ) : ℝ :=
  -1 - Real.sqrt (m + 4)

noncomputable def s (m : ℝ) : ℝ :=
  (M (-2 * m) - M m) / (2 * m)

theorem limit_of_s as m approaches 0:
  filter.tendsto s (nhds 0) (nhds (3 / 8)) := by
  sorry

end limit_of_s_l119_119312


namespace max_subset_size_with_condition_l119_119936

theorem max_subset_size_with_condition :
  ∃ (S : set ℕ), (∀ a b ∈ S, a ≠ 4 * b ∧ b ≠ 4 * a) ∧ S ⊆ {1..150} ∧ ∀ T : set ℕ, (∀ a b ∈ T, a ≠ 4 * b ∧ b ≠ 4 * a) ∧ T ⊆ {1..150} → T.card ≤ 143 :=
by
  sorry

end max_subset_size_with_condition_l119_119936


namespace price_per_small_bottle_l119_119392

theorem price_per_small_bottle 
  (total_large_bottles : ℕ)
  (total_small_bottles : ℕ)
  (price_large_bottle : ℝ)
  (approx_avg_price : ℝ)
  (total_cost_large_bottles : ℝ)
  (total_bottles : ℕ)
  (total_cost_all_bottles : ℝ)
  (price_small_bottle : ℝ): 
  total_large_bottles = 1375 →
  total_small_bottles = 690 →
  price_large_bottle = 1.75 →
  approx_avg_price = 1.6163438256658595 →
  total_cost_large_bottles = total_large_bottles * price_large_bottle →
  total_bottles = total_large_bottles + total_small_bottles →
  total_cost_all_bottles = total_cost_large_bottles + total_small_bottles * price_small_bottle →
  price_small_bottle = 
    (approx_avg_price * total_bottles - total_cost_large_bottles) / total_small_bottles →
  price_small_bottle = 1.34988436247191 :=
by {
  intros,
  sorry
}

end price_per_small_bottle_l119_119392


namespace discount_percentage_is_25_l119_119364

-- Define the conditions
def cost_of_coffee : ℕ := 6
def cost_of_cheesecake : ℕ := 10
def final_price_with_discount : ℕ := 12

-- Define the total cost without discount
def total_cost_without_discount : ℕ := cost_of_coffee + cost_of_cheesecake

-- Define the discount amount
def discount_amount : ℕ := total_cost_without_discount - final_price_with_discount

-- Define the percentage discount
def percentage_discount : ℕ := (discount_amount * 100) / total_cost_without_discount

-- Proof Statement
theorem discount_percentage_is_25 : percentage_discount = 25 := by
  sorry

end discount_percentage_is_25_l119_119364


namespace probability_heads_exactly_9_of_12_l119_119024

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l119_119024


namespace perimeter_difference_is_20_l119_119612

def perimeter_difference {r1 r2 rect_width rect_length : ℝ} 
  (h_radius1: r1 = 5) (h_radius2: r2 = 2.5) 
  (h_rect_width: rect_width = 5) (h_rect_length: rect_length = 10) : 
  ℝ :=
  let large_quarter_circle_perimeter := 2 * (1/4) * (2 * Real.pi * r1)
  let small_quarter_circle_perimeter := 2 * (1/4) * (2 * Real.pi * r2)
  let rectangle_perimeter_shape1 := 2 * rect_length + 2 * rect_width
  let rectangle_perimeter_shape2 := rect_length
  let perimeter_shape1 := large_quarter_circle_perimeter + small_quarter_circle_perimeter + rectangle_perimeter_shape1
  let perimeter_shape2 := large_quarter_circle_perimeter + small_quarter_circle_perimeter + rectangle_perimeter_shape2
  perimeter_shape1 - perimeter_shape2

theorem perimeter_difference_is_20 :
  perimeter_difference (h_radius1 := rfl) (h_radius2 := rfl) (h_rect_width := rfl) (h_rect_length := rfl) = 20 :=
sorry

end perimeter_difference_is_20_l119_119612


namespace remaining_plants_after_bugs_l119_119962

theorem remaining_plants_after_bugs (initial_plants first_day_eaten second_day_fraction third_day_eaten remaining_plants : ℕ) : 
  initial_plants = 30 →
  first_day_eaten = 20 →
  second_day_fraction = 2 →
  third_day_eaten = 1 →
  remaining_plants = initial_plants - first_day_eaten - (initial_plants - first_day_eaten) / second_day_fraction - third_day_eaten →
  remaining_plants = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end remaining_plants_after_bugs_l119_119962


namespace min_sum_of_factors_of_2310_l119_119860

theorem min_sum_of_factors_of_2310 : ∃ a b c : ℕ, a * b * c = 2310 ∧ a + b + c = 52 :=
by
  sorry

end min_sum_of_factors_of_2310_l119_119860


namespace probability_of_9_heads_in_12_l119_119132

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l119_119132


namespace probability_heads_9_of_12_is_correct_l119_119033

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l119_119033


namespace proof_problem_l119_119657

def base_7_to_10 (n : ℕ) : ℕ := 8 + 6 * 7 + 4 * 7^2 + 2 * 7^3
def base_5_to_10 (n : ℕ) : ℕ := 3 + 2 * 5 + 1 * 5^2
def base_6_to_10 (n : ℕ) : ℕ := 4 + 1 * 6 + 2 * 6^2 + 3 * 6^3
def base_8_to_10 (n : ℕ) : ℕ := 1 + 9 * 8 + 8 * 8^2 + 7 * 8^3

noncomputable def expression : ℝ :=
  (base_7_to_10 2468 : ℝ) / (base_5_to_10 123 : ℝ) - (base_6_to_10 3214 : ℝ) + (base_8_to_10 7891 : ℝ)

theorem proof_problem : expression ≈ 3464 :=
by
  sorry

end proof_problem_l119_119657


namespace equation_of_parabola_max_slope_OQ_l119_119704

section parabola

variable (p : ℝ)
variable (y : ℝ) (x : ℝ)
variable (n : ℝ) (m : ℝ)

-- Condition: p > 0 and distance from focus F to directrix being 2
axiom positive_p : p > 0
axiom distance_focus_directrix : ∀ {F : ℝ}, F = 2 * p → 2 * p = 2

-- Prove these two statements
theorem equation_of_parabola : (y^2 = 4 * x) :=
  sorry

theorem max_slope_OQ : (∃ K : ℝ, K = 1 / 3) :=
  sorry

end parabola

end equation_of_parabola_max_slope_OQ_l119_119704


namespace AdjacentComplementaryAnglesAreComplementary_l119_119891

-- Definitions of angles related to the propositions

def acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2
def complementary (θ φ : ℝ) : Prop := θ + φ = π / 2
def adjacent_complementary (θ φ : ℝ) : Prop := complementary θ φ ∧ θ ≠ φ  -- Simplified for demonstration
def corresponding (θ φ : ℝ) : Prop := sorry  -- Definition placeholder
def interior_alternate (θ φ : ℝ) : Prop := sorry  -- Definition placeholder

theorem AdjacentComplementaryAnglesAreComplementary :
  ∀ θ φ : ℝ, adjacent_complementary θ φ → complementary θ φ :=
by
  intros θ φ h
  cases h with h_complementary h_adjacent
  exact h_complementary

end AdjacentComplementaryAnglesAreComplementary_l119_119891


namespace probability_heads_in_nine_of_twelve_flips_l119_119094

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l119_119094


namespace initial_average_mark_l119_119825

-- Define the conditions
def total_students := 13
def average_mark := 72
def excluded_students := 5
def excluded_students_average := 40
def remaining_students := total_students - excluded_students
def remaining_students_average := 92

-- Define the total marks calculations
def initial_total_marks (A : ℕ) : ℕ := total_students * A
def excluded_total_marks : ℕ := excluded_students * excluded_students_average
def remaining_total_marks : ℕ := remaining_students * remaining_students_average

-- Prove the initial average mark
theorem initial_average_mark : 
  initial_total_marks average_mark = excluded_total_marks + remaining_total_marks →
  average_mark = 72 :=
by
  sorry

end initial_average_mark_l119_119825


namespace graph_EQ_a_l119_119804

theorem graph_EQ_a (x y : ℝ) : (x - 2) * (y + 3) = 0 ↔ x = 2 ∨ y = -3 :=
by sorry

end graph_EQ_a_l119_119804


namespace faculty_reduction_l119_119175

theorem faculty_reduction (x : ℝ) (h1 : 0.75 * x = 195) : x = 260 :=
by sorry

end faculty_reduction_l119_119175


namespace smallest_positive_angle_l119_119628

theorem smallest_positive_angle (x : ℝ) (hx : 0 < x ∧ x < 90) :
  tan (6 * x) = (cos x - sin x) / (cos x + sin x) → x = 45 / 7 :=
by
  intro H
  sorry

end smallest_positive_angle_l119_119628


namespace degree_g_greater_than_5_l119_119766

-- Definitions according to the given conditions
variables {f g : Polynomial ℤ}
variables (h : Polynomial ℤ)
variables (r : Fin 81 → ℤ)

-- Condition 1: g(x) divides f(x), meaning there exists an h(x) such that f(x) = g(x) * h(x)
def divides (g f : Polynomial ℤ) := ∃ (h : Polynomial ℤ), f = g * h

-- Condition 2: f(x) - 2008 has at least 81 distinct integer roots
def has_81_distinct_roots (f : Polynomial ℤ) (roots : Fin 81 → ℤ) : Prop :=
  ∀ i : Fin 81, f.eval (roots i) = 2008 ∧ Function.Injective roots

-- The theorem to prove
theorem degree_g_greater_than_5 (nonconst_f : f.degree > 0) (nonconst_g : g.degree > 0) 
  (g_div_f : divides g f) (f_has_roots : has_81_distinct_roots (f - Polynomial.C 2008) r) :
  g.degree > 5 :=
sorry

end degree_g_greater_than_5_l119_119766


namespace tim_meets_elan_distance_traveled_l119_119871

theorem tim_meets_elan_distance_traveled
  (initial_distance : ℝ)
  (tim_speed : ℝ)
  (elan_speed : ℝ)
  (double_speed : ∀ t : ℝ, 0 < t → (tim_speed * 2 ^ t) ∧ (elan_speed * 2 ^ t))
  (move_towards_each_other : Bool)
  (tim_initial_time : ℝ)
  (elan_initial_time : ℝ) :
  initial_distance = 30 →
  tim_speed = 10 →
  elan_speed = 5 →
  move_towards_each_other = true →
  tim_initial_time = 0 →
  elan_initial_time = 0 →
  double_speed 1 (by norm_num) →
  double_speed 2 (by norm_num) →
  ∃ t, t > 0 ∧ (tim_speed * (2 ^ (t - 1)) * t + elan_speed * (2 ^ (t - 1)) * t = 30) →
  let tim_total_distance := tim_speed * (2 ^ (t - 1)) * t satisfies (tim_total_distance = 20).

end tim_meets_elan_distance_traveled_l119_119871


namespace sum_of_extrema_of_g_l119_119235

noncomputable def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2*x - 8| + |x - 7|

theorem sum_of_extrema_of_g :
  let g (x : ℝ) := |x - 3| + |x - 5| - |2*x - 8| + |x - 7| in
  3 ≤ x ∧ x ≤ 9 →
  (∀ x : ℝ, 3 ≤ x ∧ x ≤ 9 → g(x) ≥ -8) ∧ (∀ x : ℝ, 3 ≤ x ∧ x ≤ 9 → g(x) ≤ 22) →
  (22 + (-8) = 14) :=
by
  have g_3 : g(3) = 0 := by sorry
  have g_4 : g(4) = 4 * 4 - 23 := by sorry
  have g_5 : g(5) = 3 * 5 - 23 := by sorry
  have g_7 : g(7) = 3 * 7 - 23 := by sorry
  have g_9 : g(9) = 5 * 9 - 23 := by sorry
  have smallest_value := -8
  have largest_value := 22
  exact (largest_value + smallest_value = 14)

end sum_of_extrema_of_g_l119_119235


namespace vanya_speed_l119_119504

variable (v : ℝ)

theorem vanya_speed (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end vanya_speed_l119_119504


namespace probability_heads_in_nine_of_twelve_flips_l119_119089

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l119_119089


namespace correct_option_D_l119_119956

noncomputable theory

-- Definitions of the statements
def stmtA : Prop := ∀ (L1 L2 : ℝ×ℝ → ℝ), (∀ t, L1 t = L2 t) → L1 = L2
def stmtB : Prop := ∀ (P1 P2 : ℝ → ℝ → ℝ), (∃ l : ℝ → ℝ, ∀ x y, P1 x y = l x ∧ P2 x y = l y) → P1 = P2
def stmtC : Prop := ∀ (P1 P2 : ℝ → ℝ → ℝ), (∃ p : ℝ → ℝ → ℝ, ∀ x y, P1 x y = p x y ∧ P2 x y = p x y) → P1 = P2
def stmtD : Prop := ∀ (L1 L2 : ℝ → ℝ → ℝ), (∃ P : ℝ → ℝ → ℝ, ∀ x y, (L1 x = P x y) ∧ (L2 x = P x y)) → L1 = L2

-- Stating that stmtD is true
theorem correct_option_D : stmtD :=
by sorry

end correct_option_D_l119_119956


namespace ages_are_correct_l119_119421

variable (A B C : ℕ)

-- Conditions
def condition1 := A + B = 2 * C
def condition2 := B = A / 2 + 4
def condition3 := A - C = (B - C) + 16

-- Goal (The theorem we want to prove)
theorem ages_are_correct (h1 : condition1 A B C) (h2 : condition2 A B C) (h3 : condition3 A B C) :
  A = 40 ∧ B = 24 ∧ C = 32 :=
by 
  sorry

end ages_are_correct_l119_119421


namespace correct_statement_l119_119668

noncomputable def f (x : ℝ) := Real.exp x - x
noncomputable def g (x : ℝ) := Real.log x + x + 1

def proposition_p := ∀ x : ℝ, f x > 0
def proposition_q := ∃ x0 : ℝ, 0 < x0 ∧ g x0 = 0

theorem correct_statement : (proposition_p ∧ proposition_q) :=
by
  sorry

end correct_statement_l119_119668


namespace prod_div_sum_le_square_l119_119841

theorem prod_div_sum_le_square (m n : ℕ) (h : (m * n) ∣ (m + n)) : m + n ≤ n^2 := sorry

end prod_div_sum_le_square_l119_119841


namespace coupon_savings_difference_l119_119207

theorem coupon_savings_difference :
  ∃ (x y : ℝ),
    (∀ P : ℝ, (P > 120) →
      (let Sx := 0.20 * P,
           Sy := 40,
           Sz := 0.30 * (P - 120)
      in Sx ≥ Sy ∧ Sx ≥ Sz) → P = x ∨ P = y) ∧
    (y - x = 160) :=
by {
  sorry
}

end coupon_savings_difference_l119_119207


namespace probability_exactly_9_heads_l119_119111

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l119_119111


namespace find_m_n_l119_119338

theorem find_m_n (x m n : ℝ) : (x + 4) * (x - 2) = x^2 + m * x + n → m = 2 ∧ n = -8 := 
by
  intro h
  -- Steps to prove the theorem would be here
  sorry

end find_m_n_l119_119338


namespace cos_squared_sum_l119_119807

theorem cos_squared_sum (α : ℝ) : 
  (cos α)^2 + (cos (2/3 * Real.pi + α))^2 + (cos (2/3 * Real.pi - α))^2 = 3/2 := 
sorry

end cos_squared_sum_l119_119807


namespace more_type_A_than_type_B_coverings_l119_119980

-- Definitions for the problem
def is_covering (covering : set (pair (pair nat nat) (pair nat nat))) : Prop :=
  ∀ square ∈ covering, <<condition to cover square with domino>>

def type_A_covering (covering : set (pair (pair nat nat) (pair nat nat))) : Prop :=
  (pair (pair 1 1) (pair 1 2)) ∈ covering ∧ is_covering covering

def type_B_covering (covering : set (pair (pair nat nat) (pair nat nat))) : Prop :=
  (pair (pair 2 2) (pair 2 3)) ∈ covering ∧ is_covering covering

-- Theorem statement
theorem more_type_A_than_type_B_coverings :
  ∃ (A_coverings B_coverings : set (set (pair (pair nat nat) (pair nat nat)))), 
    (∀ (covering : set (pair (pair nat nat) (pair nat nat))), 
      covering ∈ A_coverings ↔ type_A_covering covering) ∧
    (∀ (covering : set (pair (pair nat nat) (pair nat nat))), 
      covering ∈ B_coverings ↔ type_B_covering covering) ∧
    ∀ (p ∈ B_coverings), ∃ (q ∈ A_coverings), q ≠ p :=
sorry

end more_type_A_than_type_B_coverings_l119_119980


namespace customer_buys_5_bulbs_l119_119736

theorem customer_buys_5_bulbs 
    (total_bulbs : ℕ)
    (defective_bulbs : ℕ)
    (prob_non_defective : ℝ)
    (total_bulbs = 10)
    (defective_bulbs = 4)
    (prob_non_defective = 1 / 15) :
    ∃ (n : ℕ), n = 5 :=
sorry

end customer_buys_5_bulbs_l119_119736


namespace probability_heads_9_of_12_l119_119148

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l119_119148


namespace value_of_a_squared_plus_b_squared_l119_119348

variable (a b : ℝ)

theorem value_of_a_squared_plus_b_squared (h1 : a - b = 10) (h2 : a * b = 55) : a^2 + b^2 = 210 := 
by 
sorry

end value_of_a_squared_plus_b_squared_l119_119348


namespace total_income_in_june_l119_119791

-- Establishing the conditions
def daily_production : ℕ := 200
def days_in_june : ℕ := 30
def price_per_gallon : ℝ := 3.55

-- Defining total milk production in June as a function of daily production and days in June
def total_milk_production_in_june : ℕ :=
  daily_production * days_in_june

-- Defining total income as a function of milk production and price per gallon
def total_income (milk_production : ℕ) (price : ℝ) : ℝ :=
  milk_production * price

-- Stating the theorem that we need to prove
theorem total_income_in_june :
  total_income total_milk_production_in_june price_per_gallon = 21300 := 
sorry

end total_income_in_june_l119_119791


namespace max_imaginary_part_of_polynomial_l119_119230

theorem max_imaginary_part_of_polynomial :
  let z_roots : Finset ℂ := {z | z^6 - z^4 + z^2 - 1 = 0} in
  let max_imag_part := z_roots.maxBy (λ z, z.im) in
  ∃ θ : ℝ, max_imag_part.im = Real.sin θ ∧ -Real.pi / 2 ≤ θ ∧ θ ≤ Real.pi / 2 ∧ θ = Real.pi / 4 :=
sorry

end max_imaginary_part_of_polynomial_l119_119230


namespace probability_9_heads_12_flips_l119_119163

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l119_119163


namespace average_absolute_sum_written_as_fraction_l119_119270

def average_absolute_sum_of_permutations : ℚ :=
  let s : ℚ := 
    (2 * (1 + 3 + 6 + 10 + 15 + 21 + 28 + 36 + 45 + 55 + 66)) 
    in (s * 6 / ((12:ℚ).choose 2))

theorem average_absolute_sum_written_as_fraction(p q : ℕ) (hpq : nat.coprime p q) :
  average_absolute_sum_of_permutations = p / q → p + q = 297 :=
  sorry

end average_absolute_sum_written_as_fraction_l119_119270


namespace probability_9_heads_12_flips_l119_119159

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l119_119159


namespace smallest_k_remainder_2_l119_119542

theorem smallest_k_remainder_2 (k : ℕ) :
  k > 1 ∧
  k % 13 = 2 ∧
  k % 7 = 2 ∧
  k % 3 = 2 →
  k = 275 :=
by sorry

end smallest_k_remainder_2_l119_119542


namespace determine_point_T_l119_119991

noncomputable def point : Type := ℝ × ℝ

def is_square (O P Q R : point) : Prop :=
  O.1 = 0 ∧ O.2 = 0 ∧
  Q.1 = 3 ∧ Q.2 = 3 ∧
  P.1 = 3 ∧ P.2 = 0 ∧
  R.1 = 0 ∧ R.2 = 3

def twice_area_square_eq_area_triangle (O P Q T : point) : Prop :=
  2 * (3 * 3) = abs ((P.1 * Q.2 + Q.1 * T.2 + T.1 * P.2 - P.2 * Q.1 - Q.2 * T.1 - T.2 * P.1) / 2)

theorem determine_point_T (O P Q R T : point) (h1 : is_square O P Q R) : 
  twice_area_square_eq_area_triangle O P Q T ↔ T = (3, 12) :=
sorry

end determine_point_T_l119_119991


namespace probability_heads_in_9_of_12_flips_l119_119070

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l119_119070


namespace geometric_sequence_l119_119787

variable (S : ℕ → ℝ) (a : ℕ → ℝ)

-- Condition: Sum formula relation
axiom sum_relation (n : ℕ) : 2 * (S (n + 1)) = S n - 1

-- Initial condition
axiom initial_sum : S 0 + 1 = 3 / 2

-- Proof that {S_{n+1}} is a geometric sequence
theorem geometric_sequence : ∀ n : ℕ, S (n + 1) + 1 = (3 / 2) * (1 / 2) ^ n :=
  sorry

-- General formula for the sequence a_n
noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 1 / 2 else -3 * (1 / 2) ^ n

example : ∀ n : ℕ, a 1 = 1 / 2 ∧ (n ≥ 2 → a n = -3 * (1 / 2) ^ n) :=
  sorry

end geometric_sequence_l119_119787


namespace football_cost_correct_l119_119428

def cost_marble : ℝ := 9.05
def cost_baseball : ℝ := 6.52
def total_cost : ℝ := 20.52
def cost_football : ℝ := total_cost - cost_marble - cost_baseball

theorem football_cost_correct : cost_football = 4.95 := 
by
  -- The proof is omitted, as per instructions.
  sorry

end football_cost_correct_l119_119428


namespace angle_BDC_l119_119899

theorem angle_BDC (A B C D : Type) (h : triangle A B C)
  (right_C : is_right_angle A C B)
  (angle_A_30 : angle A = 30)
  (is_bisector : is_angle_bisector B D (C A))
  (D_on_AC : D ∈ line_segment A C) :
  angle B D C = 60 :=
sorry

end angle_BDC_l119_119899


namespace total_distance_after_fourth_bounce_l119_119195

noncomputable def total_distance_traveled (initial_height : ℝ) (bounce_ratio : ℝ) (num_bounces : ℕ) : ℝ :=
  let fall_distances := (List.range (num_bounces + 1)).map (λ n => initial_height * bounce_ratio^n)
  let rise_distances := (List.range num_bounces).map (λ n => initial_height * bounce_ratio^(n+1))
  fall_distances.sum + rise_distances.sum

theorem total_distance_after_fourth_bounce :
  total_distance_traveled 25 (5/6 : ℝ) 4 = 154.42 :=
by
  sorry

end total_distance_after_fourth_bounce_l119_119195


namespace probability_heads_in_9_of_12_flips_l119_119071

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l119_119071


namespace probability_exactly_9_heads_l119_119118

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l119_119118


namespace find_f7_l119_119319

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x + 7

theorem find_f7 (a b : ℝ) (h : f (-7) a b = -17) : f (7) a b = 31 := 
by
  sorry

end find_f7_l119_119319


namespace min_value_reciprocal_sum_l119_119343

theorem min_value_reciprocal_sum 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = 2) : 
  ∃ x, x = 2 ∧ (∀ y, y = (1 / a) + (1 / b) → x ≤ y) := 
sorry

end min_value_reciprocal_sum_l119_119343


namespace vanya_speed_increased_by_4_l119_119496

variable (v : ℝ)
variable (h1 : (v + 2) / v = 2.5)

theorem vanya_speed_increased_by_4 (h1 : (v + 2) / v = 2.5) : (v + 4) / v = 4 := 
sorry

end vanya_speed_increased_by_4_l119_119496


namespace count_powers_of_3_not_powers_of_9_l119_119329

theorem count_powers_of_3_not_powers_of_9 :
  let N := 1000000
  let pow3_count := (List.range 13).filter (fun n => 3^n < N).length
  let pow9_count := (List.range 7).filter (fun k => 9^k < N).length
  pow3_count - pow9_count = 6 :=
by
  sorry

end count_powers_of_3_not_powers_of_9_l119_119329


namespace odd_function_condition_l119_119667

variable {x : ℝ} {k : ℤ} {ϕ : ℝ}

theorem odd_function_condition : 
  (ϕ = 0 → (∀ x, f x = sin (2 * x + ϕ) → ∀ y, f (-y) = -f y)) ∧
  (∀ x, f x = sin (2 * x + ϕ) → ∀ y, f (-y) = -f y → ϕ = k * π + π) → 
  ϕ ≠ 0 :=
sorry

end odd_function_condition_l119_119667


namespace impossible_values_l119_119420

theorem impossible_values :
  ∃ (a b c x : ℝ), 
    (n = 2 ^ (x * 0.15)) ∧ 
    (n^b = 32) ∧ 
    (n = 5 ^ (a * sin c)) → 
  false :=
by {
  sorry
}

end impossible_values_l119_119420


namespace min_questionnaires_needed_l119_119726

-- Define the condition
def response_rate := 0.62
def required_responses := 300

-- Define the minimum number of questionnaires required as per the given problem
def min_questionnaires := (required_responses / response_rate).ceil

-- State the theorem we need to prove
theorem min_questionnaires_needed : min_questionnaires = 484 := sorry

end min_questionnaires_needed_l119_119726


namespace eval_expression_l119_119247

theorem eval_expression : -20 + 12 * ((5 + 15) / 4) = 40 :=
by
  sorry

end eval_expression_l119_119247


namespace no_integer_sided_triangle_with_odd_perimeter_1995_l119_119643

theorem no_integer_sided_triangle_with_odd_perimeter_1995 :
  ¬ ∃ (a b c : ℕ), (a + b + c = 1995) ∧ (∃ (h1 h2 h3 : ℕ), true) :=
by
  sorry

end no_integer_sided_triangle_with_odd_perimeter_1995_l119_119643


namespace expected_surnames_not_repositioned_l119_119592

theorem expected_surnames_not_repositioned (n : ℕ) :
  (∑ k in Finset.range n, (1 / (k + 1) : ℝ)) = 
  (1 + ∑ i in Finset.range (n-1), (1 / (i + 2) : ℝ)) :=
sorry

end expected_surnames_not_repositioned_l119_119592


namespace find_b_15_l119_119288

variable {a : ℕ → ℤ} (b : ℕ → ℤ) (S : ℕ → ℤ)

/-- An arithmetic sequence where S_n is the sum of the first n terms, with S_9 = -18 and S_13 = -52
   and a geometric sequence where b_5 = a_5 and b_7 = a_7. -/
theorem find_b_15 
  (h1 : S 9 = -18) 
  (h2 : S 13 = -52) 
  (h3 : b 5 = a 5) 
  (h4 : b 7 = a 7) 
  : b 15 = -64 := 
sorry

end find_b_15_l119_119288


namespace volume_of_sphere_l119_119280

noncomputable def cone_surface_area (r l : ℝ) : ℝ :=
  π * r * l

noncomputable def sphere_surface_area (R : ℝ) : ℝ :=
  4 * π * R^2

noncomputable def sphere_volume (R : ℝ) : ℝ :=
  (4 / 3) * π * R^3

theorem volume_of_sphere
  (h : ℝ) (r : ℝ) (R : ℝ)
  (h_eq : h = 3)
  (r_eq : r = 4)
  (cone_lateral_surface_area_eq : π * r * (sqrt (h^2 + r^2)) = 20 * π)
  (sphere_surface_area_eq : 4 * π * R^2 = 20 * π) :
  sphere_volume (sqrt 5) = (20 * sqrt 5 * π) / 3 :=
by
  sorry

end volume_of_sphere_l119_119280


namespace min_sum_of_factors_l119_119847

theorem min_sum_of_factors (a b c : ℕ) (h1 : a * b * c = 2310) : a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l119_119847


namespace math_problem_l119_119236

def valid_interval (x : ℝ) : Prop :=
  (x < (1 - math.sqrt 13) / 6) ∨ (x > (1 - math.sqrt 13) / 6 ∧ x < (1 + math.sqrt 13) / 6) ∨ (x > (1 + math.sqrt 13) / 6 ∧ x < 0) ∨ (x > 0 ∧ x ≤ 2)

theorem math_problem (x : ℝ) (h2 : x ≤ 2) (h1 : valid_interval x) :
  (x^3 - x^4 - 3 * x^5) / (x^2 - x^3 - 3 * x^4) ≤ 2 :=
sorry

end math_problem_l119_119236


namespace height_relationship_l119_119609

theorem height_relationship (B V G : ℝ) (h1 : B = 2 * V) (h2 : V = (2 / 3) * G) : B = (4 / 3) * G :=
sorry

end height_relationship_l119_119609


namespace line_is_tangent_to_circle_l119_119639

def line_eq (k x : ℝ) := k * (x - 1) + 1

def circle_eq (x y : ℝ) := x^2 + y^2 - 2 * y = 0

theorem line_is_tangent_to_circle (k : ℝ) :
  (∀ x y : ℝ, circle_eq x y → y = line_eq k x) ↔ Δ (line_eq k) = 0 :=
by
  sorry

end line_is_tangent_to_circle_l119_119639


namespace number_of_volumes_l119_119366
-- Import the Mathlib for mathematical definitions and theorems.

-- Define the problem conditions.
def isosceles_right_triangle (a b c : ℝ) : Prop := 
  a = b ∧ c = a * Real.sqrt 2

def equilateral_triangle (a b c : ℝ) : Prop := 
  a = b ∧ b = c 

-- Define the tetrahedron with specific face properties.
def tetrahedron (vertices : Fin 4 → ℝ × ℝ × ℝ) : Prop := 
  let ⟨S, A, B, C⟩ := vertices in
  isosceles_right_triangle (dist S A) (dist S B) (dist A B) ∧
  isosceles_right_triangle (dist S A) (dist S C) (dist A C) ∧
  equilateral_triangle (dist A B) (dist B C) (dist C A) ∧
  dist A B = 1

-- Define the volume of a tetrahedron.
def volume_tetrahedron (vertices : Fin 4 → ℝ × ℝ × ℝ) : ℝ :=
  let ⟨S, A, B, C⟩ := vertices in
  (1 / 6) * Real.abs (Real.sqrt ((dist S A) * (dist S B) * (dist S C)))

-- Define the main theorem: proving the number of distinct volumes.

theorem number_of_volumes (vertices : Fin 4 → ℝ × ℝ × ℝ) : 
  tetrahedron vertices → 
  ∃! (volumes : Finset ℝ), volumes.card = 3 := 
sorry

end number_of_volumes_l119_119366


namespace charlies_age_22_l119_119227

variable (A : ℕ) (C : ℕ)

theorem charlies_age_22 (h1 : C = 2 * A + 8) (h2 : C = 22) : A = 7 := by
  sorry

end charlies_age_22_l119_119227


namespace length_of_train_75_l119_119179

variable (L : ℝ) -- Length of the train in meters

-- Condition 1: The train crosses a bridge of length 150 m in 7.5 seconds
def crosses_bridge (L: ℝ) : Prop := (L + 150) / 7.5 = L / 2.5

-- Condition 2: The train crosses a lamp post in 2.5 seconds
def crosses_lamp (L: ℝ) : Prop := L / 2.5 = L / 2.5

theorem length_of_train_75 (L : ℝ) (h1 : crosses_bridge L) (h2 : crosses_lamp L) : L = 75 := 
by 
  sorry

end length_of_train_75_l119_119179


namespace matrix_inverse_linear_combination_l119_119406

theorem matrix_inverse_linear_combination (c d : ℝ) (N : Matrix (Fin 2) (Fin 2) ℝ) (I : Matrix (Fin 2) (Fin 2) ℝ) (hN : N = ![[3, 0], [2, -4]]) :
    N⁻¹ = c • N + d • I :=
by
  have hNi: N⁻¹ = ![[1 / 3, 0], [1 / 6, -1 / 4]] := by sorry
  have hCo: c = 1 / 12 := by sorry
  have hDo: d = 1 / 12 := by sorry
  simp [hNi, hCo, hDo]
  sorry

end matrix_inverse_linear_combination_l119_119406


namespace abe_family_total_yen_l119_119213

theorem abe_family_total_yen (yen_checking : ℕ) (yen_savings : ℕ) (h₁ : yen_checking = 6359) (h₂ : yen_savings = 3485) : yen_checking + yen_savings = 9844 :=
by
  sorry

end abe_family_total_yen_l119_119213


namespace sum_le_square_l119_119843

theorem sum_le_square (m n : ℕ) (h: (m * n) % (m + n) = 0) : m + n ≤ n^2 :=
by sorry

end sum_le_square_l119_119843


namespace sequence_average_ge_neg_half_l119_119272

noncomputable def sequence (n : ℕ) : ℕ → ℝ
| 0 => 0
| (k + 1) => sign (1 - 2 * (k % 2)) * (abs (sequence k) + 1)

theorem sequence_average_ge_neg_half (n : ℕ) :
  (∑ i in range n, sequence i) / n ≥ -1/2 :=
sorry

end sequence_average_ge_neg_half_l119_119272


namespace smallest_sum_arithmetic_sequence_l119_119371

noncomputable def a_n (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

theorem smallest_sum_arithmetic_sequence {a1 : ℝ} (h1 : a1 < 0) (d : ℝ) 
  (h2 : 3 * (a1 + 4 * d) = 7 * (a1 + 9 * d)) : 
  let d := -4 * a1 / 51 in 
  let a_n := λ (n : ℕ), a1 + (n - 1) * d in 
  let S_n := λ (n : ℕ), ∑ i in finset.range n, a_n i in 
  S_13 < S_n  :=
begin
  sorry -- proof not required
end

end smallest_sum_arithmetic_sequence_l119_119371


namespace river_length_l119_119476

theorem river_length (x : ℝ) (h1 : 3 * x + x = 80) : x = 20 :=
sorry

end river_length_l119_119476


namespace Brianna_reading_time_l119_119215

def reading_time_Brianna (pages : ℕ) (time_Anna : ℕ) (speed_factor : ℕ) : ℕ :=
(pages / (time_Anna * speed_factor))

theorem Brianna_reading_time :
  reading_time_Brianna 100 1 2 = 50 :=
by
  simp [reading_time_Brianna]
  norm_num

end Brianna_reading_time_l119_119215


namespace probability_heads_9_of_12_flips_l119_119010

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l119_119010


namespace vanya_faster_by_4_l119_119534

variable (v : ℝ) -- initial speed in m/s
variable (school_distance : ℝ) -- distance to school in meters
variable (time_to_school : ℝ) -- time to school in seconds
variable (increased_speed : ℝ → ℝ) -- increased speed function

-- Conditions
def initial_speed : (v > 0) := by sorry
def increased_speed_by_2 {v : ℝ} : (v + 2 > 0) := by sorry
def faster_by_2_5 {v : ℝ} : (time_to_school = (school_distance / v)) → (time_to_school / 2.5 = (school_distance / (v + 2))) := by sorry

-- Main statement to prove
theorem vanya_faster_by_4 (h1 : initial_speed v) (h2 : increased_speed_by_2 v) (h3 : faster_by_2_5 v time_to_school school_distance) :
    (time_to_school / 4 = (school_distance / (v + 4))) := by
  sorry

end vanya_faster_by_4_l119_119534


namespace crow_fetches_worms_15_times_l119_119570

-- Define the conditions
def distance_nest_to_ditch : ℝ := 200 -- meters
def crow_speed_kph : ℝ := 4 -- kilometers per hour
def gathering_time : ℝ := 1.5 -- hours

-- Define the necessary conversions
def meters_per_kilometer : ℝ := 1000
def minutes_per_hour : ℝ := 60

-- Conversion of crow's speed to meters per minute
def crow_speed_mpm : ℝ :=
  (crow_speed_kph * meters_per_kilometer) / minutes_per_hour

-- Define the round trip distance
def round_trip_distance : ℝ := 2 * distance_nest_to_ditch -- 400 meters

-- Find the number of times the crow fetches worms
def number_of_round_trips : ℝ :=
  (gathering_time * minutes_per_hour) / (round_trip_distance / crow_speed_mpm)

-- State the theorem
theorem crow_fetches_worms_15_times :
  number_of_round_trips = 15 :=
  by
    sorry

end crow_fetches_worms_15_times_l119_119570


namespace coordinates_of_P_l119_119687

theorem coordinates_of_P (a : ℝ) (x y : ℝ) (h₁ : x = a + 2) (h₂ : y = 3a - 6) (h₃ : abs x = abs y) :
  (x, y) = (6, 6) ∨ (x, y) = (3, -3) :=
by sorry

end coordinates_of_P_l119_119687


namespace lemonade_second_intermission_l119_119644

theorem lemonade_second_intermission (first_intermission third_intermission total_lemonade second_intermission : ℝ) 
  (h1 : first_intermission = 0.25) 
  (h2 : third_intermission = 0.25) 
  (h3 : total_lemonade = 0.92) 
  (h4 : second_intermission = total_lemonade - (first_intermission + third_intermission)) : 
  second_intermission = 0.42 := 
by 
  sorry

end lemonade_second_intermission_l119_119644


namespace find_a_range_l119_119313

noncomputable def f (x : ℝ) : ℝ := Real.exp x - (1 / Real.exp x) - 2 * x

theorem find_a_range (a : ℝ) (h : f (a - 3) + f (2 * a ^ 2) ≤ 0) : -3 / 2 ≤ a ∧ a ≤ 1 :=
begin
  sorry
end

end find_a_range_l119_119313


namespace probability_of_9_heads_in_12_flips_l119_119061

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l119_119061


namespace sum_powers_of_i_l119_119968

theorem sum_powers_of_i : ∑ k in Finset.range 2012, complex.I ^ (k + 1) = 0 := by
  -- The rest of the proof is omitted
  sorry

end sum_powers_of_i_l119_119968


namespace find_multiple_of_diff_l119_119920

theorem find_multiple_of_diff (n sum diff remainder k : ℕ) 
  (hn : n = 220070) 
  (hs : sum = 555 + 445) 
  (hd : diff = 555 - 445)
  (hr : remainder = 70)
  (hmod : n % sum = remainder) 
  (hquot : n / sum = k) :
  ∃ k, k = 2 ∧ k * diff = n / sum := 
by 
  sorry

end find_multiple_of_diff_l119_119920


namespace sum_of_squares_of_rates_l119_119988

variable (b j s : ℤ) -- rates in km/h
-- conditions
def ed_condition : Prop := 3 * b + 4 * j + 2 * s = 86
def sue_condition : Prop := 5 * b + 2 * j + 4 * s = 110

theorem sum_of_squares_of_rates (b j s : ℤ) (hEd : ed_condition b j s) (hSue : sue_condition b j s) : 
  b^2 + j^2 + s^2 = 3349 := 
sorry

end sum_of_squares_of_rates_l119_119988


namespace probability_exactly_9_heads_in_12_flips_l119_119105

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l119_119105


namespace particle_speed_l119_119921

theorem particle_speed (t : ℝ) : 
  let position : ℝ × ℝ := (3 * t + 1, 5 * t - 2) in
  let Δx := (3 * (t + 1) + 1 - (3 * t + 1)) in
  let Δy := (5 * (t + 1) - 2 - (5 * t - 2)) in 
  sqrt (Δx^2 + Δy^2) = sqrt 34 :=
by
  let position := (3 * t + 1, 5 * t - 2)
  let Δx := (3 * (t + 1) + 1 - (3 * t + 1))
  let Δy := (5 * (t + 1) - 2 - (5 * t - 2))
  have hΔx : Δx = 3 := by sorry
  have hΔy : Δy = 5 := by sorry
  rw [hΔx, hΔy]
  have : 3^2 + 5^2 = 34 := by norm_num
  rw [this, sqrt_eq_rfl]
  norm_num
  sorry

end particle_speed_l119_119921


namespace vanya_speed_l119_119506

variable (v : ℝ)

theorem vanya_speed (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end vanya_speed_l119_119506


namespace probability_heads_9_of_12_l119_119144

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l119_119144


namespace man_l119_119919

noncomputable def straight_wind_favor_speed_with_current := 15
noncomputable def straight_wind_against_speed_with_current := 12
noncomputable def current_speed_straight := 2.5
noncomputable def wind_impact := 1

noncomputable def bend_wind_favor_speed_with_current := 10
noncomputable def bend_wind_against_speed_with_current := 7
noncomputable def current_speed_bend := 3.5

-- Man's rowing speed against the current and wind in different scenarios
def straight_against_current_wind_favor :=
  straight_wind_favor_speed_with_current - current_speed_straight - wind_impact

def straight_against_current_wind_against :=
  straight_wind_against_speed_with_current - current_speed_straight - wind_impact

def bend_against_current_wind_favor :=
  bend_wind_favor_speed_with_current - current_speed_bend - wind_impact

def bend_against_current_wind_against :=
  bend_wind_against_speed_with_current - current_speed_bend - wind_impact

-- Average speed calculations
def straight_avg_against_current :=
  (straight_against_current_wind_favor + straight_against_current_wind_against) / 2

def bend_avg_against_current :=
  (bend_against_current_wind_favor + bend_against_current_wind_against) / 2

-- Overall average speed
def overall_avg_against_current :=
  (straight_avg_against_current + bend_avg_against_current) / 2

theorem man's_avg_speed_against_current :
  overall_avg_against_current = 7 := by
  sorry

end man_l119_119919


namespace min_sum_factors_l119_119849

theorem min_sum_factors (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prod : a * b * c = 2310) : a + b + c = 40 :=
sorry

end min_sum_factors_l119_119849


namespace list_size_is_2017_l119_119580

def has_sum (L : List ℤ) (n : ℤ) : Prop :=
  List.sum L = n

def has_product (L : List ℤ) (n : ℤ) : Prop :=
  List.prod L = n

def includes (L : List ℤ) (n : ℤ) : Prop :=
  n ∈ L

theorem list_size_is_2017 
(L : List ℤ) :
  has_sum L 2018 ∧ 
  has_product L 2018 ∧ 
  includes L 2018 
  → L.length = 2017 :=
by 
  sorry

end list_size_is_2017_l119_119580


namespace order_of_ab_l119_119407

theorem order_of_ab (a b n ω_a ω_b : ℕ)
  (h_coprime_n_a : Nat.coprime a n)
  (h_coprime_n_b : Nat.coprime b n)
  (h_order_a : order_of a n = ω_a)
  (h_order_b : order_of b n = ω_b)
  (h_coprime : Nat.coprime ω_a ω_b) :
  order_of (a * b) n = ω_a * ω_b :=
sorry

end order_of_ab_l119_119407


namespace point_in_third_quadrant_l119_119350

theorem point_in_third_quadrant (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : 
  (-b < 0) ∧ (a < 0) ∧ (-b > a) :=
by
  sorry

end point_in_third_quadrant_l119_119350


namespace vector_parallel_l119_119708

theorem vector_parallel (a : ℝ) :
  let m := (a, -2)
  let n := (1, 2 - a)
  (∃ k : ℝ, m.1 = k * n.1 ∧ m.2 = k * n.2) → (a = 1 + Real.sqrt 3 ∨ a = 1 - Real.sqrt 3) := 
by
  intro h
  sorry

end vector_parallel_l119_119708


namespace probability_heads_9_of_12_l119_119139

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l119_119139


namespace probability_heads_in_9_of_12_flips_l119_119077

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l119_119077


namespace has_real_roots_l119_119887

def f (x : ℝ) : ℝ := x^2 / (abs x + 1)

theorem has_real_roots : ∃ x : ℝ, f x = 2 :=
sorry

end has_real_roots_l119_119887


namespace g_g_has_two_distinct_real_roots_l119_119409

-- Defining the function g
def g (x c : ℝ) : ℝ := x^2 + 2*x + c^2

-- Stating the main theorem
theorem g_g_has_two_distinct_real_roots (c : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g (g x₁ c) c = 0 ∧ g (g x₂ c) c = 0) ↔ c = 1 ∨ c = -1 := 
sorry

end g_g_has_two_distinct_real_roots_l119_119409


namespace base6_addition_problem_l119_119250

theorem base6_addition_problem (X Y : ℕ) (h1 : Y + 3 = X) (h2 : X + 2 = 7) : X + Y = 7 := 
by
  sorry

end base6_addition_problem_l119_119250


namespace plants_remaining_l119_119965

theorem plants_remaining (plants_initial plants_first_day plants_second_day_eaten plants_third_day_eaten : ℕ)
  (h1 : plants_initial = 30)
  (h2 : plants_first_day = 20)
  (h3 : plants_second_day_eaten = (plants_initial - plants_first_day) / 2)
  (h4 : plants_third_day_eaten = 1)
  : (plants_initial - plants_first_day - plants_second_day_eaten - plants_third_day_eaten) = 4 := 
by
  sorry

end plants_remaining_l119_119965


namespace line_equation_direction_point_l119_119684

theorem line_equation_direction_point 
  (d : ℝ × ℝ) (A : ℝ × ℝ) :
  d = (2, -1) →
  A = (1, 0) →
  ∃ (a b c : ℝ), a = 1 ∧ b = 2 ∧ c = -1 ∧ ∀ x y : ℝ, a * x + b * y + c = 0 ↔ x + 2 * y - 1 = 0 :=
by
  sorry

end line_equation_direction_point_l119_119684


namespace paths_below_or_touch_at_most_once_paths_not_intersect_AB_l119_119701

theorem paths_below_or_touch_at_most_once (n m : ℕ) (h_pos : 0 < n ∧ 0 < m) (h_lt : n < m) :
  -- number of paths from A(0,0) to C(m,n) that lie entirely below AB and touch at most once
  (number_of_paths_below_or_touch := (m - n) / m * (binom (m + n - 1) n)) := sorry

theorem paths_not_intersect_AB (n m : ℕ) (h_pos : 0 < n ∧ 0 < m) (h_lt : n < m) :
  -- number of paths from A(0,0) to C(m,n) that do not intersect AB
  (number_of_paths_not_intersect := (m - n) / m * (binom (m + n - 1) n)) := sorry

end paths_below_or_touch_at_most_once_paths_not_intersect_AB_l119_119701


namespace num_divisors_ge_l119_119777

theorem num_divisors_ge {n : ℕ} {p : fin n → ℕ} 
  (hp : ∀ i : fin n, prime (p i)) 
  (hp_gt : ∀ i : fin n, p i > 3) : 
  nat.num_divisors (2 ^ (p 0) * p 1 * ... * p (n - 1) + 1) ≥ 4 ^ n := 
sorry

end num_divisors_ge_l119_119777


namespace find_integer_n_l119_119999

theorem find_integer_n {n : ℤ} (hn_range : 0 ≤ n ∧ n ≤ 15) : n ≡ 14525 [MOD 16] → n = 13 := by
  sorry

end find_integer_n_l119_119999


namespace cot_square_difference_l119_119341

theorem cot_square_difference (x : ℝ) 
  (h : is_geometric_sequence (sin x) (sin (2 * x)) (tan x)) : 
  cot 2x ^ 2 - (cot x) ^ 2 = 
  sorry :=
by
  sorry

end cot_square_difference_l119_119341


namespace max_value_of_f_l119_119355

noncomputable def f (x a b : ℝ) := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_f (a b : ℝ) (h_symmetric : ∀ x : ℝ, f (-2 - x) a b = f (-2 + x) a b) :
  ∃ x : ℝ, f x a b = 16 := by
  sorry

end max_value_of_f_l119_119355


namespace angle_C_is_60_degrees_length_c_is_6_l119_119709

-- Problem 1: Proving that angle C is 60 degrees
theorem angle_C_is_60_degrees (A B C : ℝ) (sin_A sin_B sin_C cos_A cos_B : ℝ) :
  ((sin_A * cos_B + cos_A * sin_B) = sin (A + B)) →
  (sin (A + B) = sin (2 * C)) →
  C = 60 :=
by 
  sorry

-- Problem 2: Proving that side length c is 6
theorem length_c_is_6 (A B C a b c : ℝ) (sin_A sin_B sin_C : ℝ)  :
  (sin_A + sin_B = 2 * sin_C) →
  (a * b * cos 60 = 18) →
  (cos C = 1/2) →
  c = 6 :=
by 
  sorry

end angle_C_is_60_degrees_length_c_is_6_l119_119709


namespace right_triangle_and_area_l119_119865

-- Define the side lengths of the triangle
def a : ℝ := 15
def b : ℝ := 36
def c : ℝ := 39

-- Pythagorean theorem check for right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Area calculation for a right triangle
def area (a b : ℝ) : ℝ :=
  (1 / 2) * a * b

theorem right_triangle_and_area :
  is_right_triangle a b c ∧ area a b = 270 := by
  sorry

end right_triangle_and_area_l119_119865


namespace probability_heads_in_9_of_12_flips_l119_119080

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l119_119080


namespace vectors_perpendicular_l119_119290

open Real

variables (a b : ℝ^3)

-- Given conditions: a and b are non-zero vectors, and |a + b| = |a - b|
def non_zero_vectors (a b : ℝ^3) : Prop := a ≠ 0 ∧ b ≠ 0
def equal_norm_condition (a b : ℝ^3) : Prop := ∥a + b∥ = ∥a - b∥

-- Prove that a is perpendicular to b
theorem vectors_perpendicular (a b : ℝ^3) (h1 : non_zero_vectors a b) (h2 : equal_norm_condition a b) : a ⬝ b = 0 :=
sorry

end vectors_perpendicular_l119_119290


namespace triangles_satisfying_equation_l119_119173

theorem triangles_satisfying_equation (a b c : ℝ) (h₂ : a + b > c) (h₃ : a + c > b) (h₄ : b + c > a) :
  (c ^ 2 - a ^ 2) / b + (b ^ 2 - c ^ 2) / a = b - a →
  (a = b ∨ c ^ 2 = a ^ 2 + b ^ 2) := 
sorry

end triangles_satisfying_equation_l119_119173


namespace probability_heads_9_of_12_is_correct_l119_119037

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l119_119037


namespace value_of_x_l119_119335

theorem value_of_x (x : ℝ) (h : (x / 5 / 3) = (5 / (x / 3))) : x = 15 ∨ x = -15 := 
by sorry

end value_of_x_l119_119335


namespace total_income_in_june_l119_119790

-- Establishing the conditions
def daily_production : ℕ := 200
def days_in_june : ℕ := 30
def price_per_gallon : ℝ := 3.55

-- Defining total milk production in June as a function of daily production and days in June
def total_milk_production_in_june : ℕ :=
  daily_production * days_in_june

-- Defining total income as a function of milk production and price per gallon
def total_income (milk_production : ℕ) (price : ℝ) : ℝ :=
  milk_production * price

-- Stating the theorem that we need to prove
theorem total_income_in_june :
  total_income total_milk_production_in_june price_per_gallon = 21300 := 
sorry

end total_income_in_june_l119_119790


namespace duty_arrangements_l119_119579

theorem duty_arrangements (science_teachers : Finset ℕ) (liberal_arts_teachers : Finset ℕ) :
  science_teachers.card = 6 →
  liberal_arts_teachers.card = 2 →
  (∃ arrangements : Finset (Finset ℕ × Finset ℕ × Finset ℕ),
    arrangements.card = 540) :=
by
  intros h_science h_liberal
  sorry

end duty_arrangements_l119_119579


namespace coplanar_vectors_lambda_value_l119_119309

theorem coplanar_vectors_lambda_value :
  ∀ (λ m n : ℝ),
    (a b c : ℝ × ℝ × ℝ)
    (hₐ : a = (2, -1, 3))
    (hᵦ : b = (-1, 4, -2))
    (h𝚌 : c = (7, 5, λ))
    (h₁ : 7 = 2 * m - n)
    (h₂ : 5 = -m + 4 * n),
    λ = 3 * m - 2 * n → λ = 65 / 7 :=
begin
  intros λ m n a b c hₐ hᵦ h𝚌 h₁ h₂ hλ,
  sorry
end

end coplanar_vectors_lambda_value_l119_119309


namespace zero_in_interval_l119_119801

theorem zero_in_interval (x y : ℝ) (hx_lt_0 : x < 0) (hy_gt_0 : 0 < y) (hy_lt_1 : y < 1) (h : x^5 < y^8 ∧ y^8 < y^3 ∧ y^3 < x^6) : x^5 < 0 ∧ 0 < y^8 :=
by
  sorry

end zero_in_interval_l119_119801


namespace smallest_n_integral_solutions_l119_119544

theorem smallest_n_integral_solutions :
  ∃ n : ℕ, (∀ x : ℚ, 15 * x^2 - n * x + 315 = 0 → x ∈ ℤ) ∧
    ∀ m : ℕ, (m < n → (∀ x : ℚ, 15 * x^2 - m * x + 315 ≠ 0 ∨ x ∉ ℤ)) :=
  sorry

end smallest_n_integral_solutions_l119_119544


namespace min_value_expression_l119_119541

theorem min_value_expression (a b c : ℝ) (h : 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 4) :
  (a - 1)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (4 / c - 1)^2 = 12 - 8 * Real.sqrt 2 :=
sorry

end min_value_expression_l119_119541


namespace soldiers_movement_l119_119539

theorem soldiers_movement (n : ℕ) 
  (initial_positions : Fin (n+3) × Fin (n+1) → Prop) 
  (moves_to_adjacent : ∀ p : Fin (n+3) × Fin (n+1), initial_positions p → initial_positions (p.1 + 1, p.2) ∨ initial_positions (p.1 - 1, p.2) ∨ initial_positions (p.1, p.2 + 1) ∨ initial_positions (p.1, p.2 - 1))
  (final_positions : Fin (n+1) × Fin (n+3) → Prop) : Even n := 
sorry

end soldiers_movement_l119_119539


namespace bricks_in_wall_l119_119223

-- Definitions for individual working times and breaks
def Bea_build_time := 8  -- hours
def Bea_break_time := 10 / 60  -- hours per hour
def Ben_build_time := 12  -- hours
def Ben_break_time := 15 / 60  -- hours per hour

-- Total effective rates
def Bea_effective_rate (h : ℕ) := h / (Bea_build_time * (1 - Bea_break_time))
def Ben_effective_rate (h : ℕ) := h / (Ben_build_time * (1 - Ben_break_time))

-- Decreased rate due to talking
def total_effective_rate (h : ℕ) := Bea_effective_rate h + Ben_effective_rate h - 12

-- Define the Lean proof statement
theorem bricks_in_wall (h : ℕ) :
  (6 * total_effective_rate h = h) → h = 127 :=
by sorry

end bricks_in_wall_l119_119223


namespace base6_sum_l119_119262

theorem base6_sum (D C : ℕ) (h₁ : D + 2 = C) (h₂ : C + 3 = 7) : C + D = 6 :=
by
  sorry

end base6_sum_l119_119262


namespace no_primes_divisible_by_35_l119_119713

theorem no_primes_divisible_by_35 : ∀ p : ℕ, Prime p → ¬ (35 ∣ p) :=
by
  intro p hp hdiv
  have h35 := Nat.prime_of_dvd_prime hp hdiv
  have : 35 = 5 * 7 := by norm_num
  have h5 : Prime 5 := by exact Nat.prime_def_lt.mp (by norm_num : Prime 5)
  have h7 : Prime 7 := by exact Nat.prime_def_lt.mp (by norm_num : Prime 7)
  rw [this, Nat.prime_def_lt] at h35
  contradiction
  sorry

end no_primes_divisible_by_35_l119_119713


namespace probability_perfect_square_divisors_of_15_factorial_l119_119923

theorem probability_perfect_square_divisors_of_15_factorial :
  let number_of_divisors := (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1),
      number_of_perfect_square_divisors := 
        (6) * (4) * (2) * (2) * (1) * (1),
      probability := number_of_perfect_square_divisors / number_of_divisors
  in probability = 1 / 42 := 
by 
-- Definitions of intermediate steps to make the theorem precise
let number_of_divisors := (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1) in 
let number_of_perfect_square_divisors := (6) * (4) * (2) * (2) * (1) * (1) in 
let probability := number_of_perfect_square_divisors / number_of_divisors in
have h0 : number_of_divisors = 4032 := by sorry, -- Calculation of number of divisors
have h1 : number_of_perfect_square_divisors = 96 := by sorry, -- Calculation of perfect square divisors
have h2 : probability = 96 / 4032 := by rw [h0, h1] at *, -- Using calculated values to determine probability
show probability = 1 / 42, from by sorry

end probability_perfect_square_divisors_of_15_factorial_l119_119923


namespace sqrt_simplification_l119_119976

theorem sqrt_simplification (x : ℕ) : (∃ (a b : ℕ) (ha : a = 3) (hb : b = 4), x = 3^2 * 4^4) → (sqrt (x) = 48) :=
by
  rintro ⟨a, b, ha, hb, rfl⟩
  rw [ha, hb]
  have : sqrt (3^2 * 4^4) = sqrt ((3 * 4^2)^2) := by sorry
  rw [this]
  have : sqrt ((3 * 16)^2) = 3 * 16 := by sorry
  rw [this]
  exact rfl

end sqrt_simplification_l119_119976


namespace train_speed_proof_l119_119876

def identical_trains_speed : Real :=
  11.11

theorem train_speed_proof :
  ∀ (v : ℝ),
  (∀ (t t' : ℝ), 
  (t = 150 / v) ∧ 
  (t' = 300 / v) ∧ 
  ((t' + 100 / v) = 36)) → v = identical_trains_speed :=
by
  sorry

end train_speed_proof_l119_119876


namespace net_income_in_June_l119_119793

theorem net_income_in_June 
  (daily_milk_production : ℕ) 
  (price_per_gallon : ℝ) 
  (daily_expense : ℝ) 
  (days_in_month : ℕ)
  (monthly_expense : ℝ) : 
  daily_milk_production = 200 →
  price_per_gallon = 3.55 →
  daily_expense = daily_milk_production * price_per_gallon →
  days_in_month = 30 →
  monthly_expense = 3000 →
  (daily_expense * days_in_month - monthly_expense) = 18300 :=
begin
  intros h_prod h_price h_daily_inc h_days h_monthly_exp,
  sorry
end

end net_income_in_June_l119_119793


namespace probability_heads_9_of_12_flips_l119_119011

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l119_119011


namespace intersection_points_on_circle_l119_119229

theorem intersection_points_on_circle
  (x y : ℝ)
  (h1 : y = (x + 2)^2)
  (h2 : x + 2 = (y - 1)^2) :
  (x + 2)^2 + (y - 1)^2 = 2 :=
sorry

end intersection_points_on_circle_l119_119229


namespace max_subset_size_with_condition_l119_119934

theorem max_subset_size_with_condition :
  ∃ (S : set ℕ), (∀ a b ∈ S, a ≠ 4 * b ∧ b ≠ 4 * a) ∧ S ⊆ {1..150} ∧ ∀ T : set ℕ, (∀ a b ∈ T, a ≠ 4 * b ∧ b ≠ 4 * a) ∧ T ⊆ {1..150} → T.card ≤ 143 :=
by
  sorry

end max_subset_size_with_condition_l119_119934


namespace train_pole_time_l119_119944

theorem train_pole_time
  (L : ℕ) (T_platform : ℕ) (L_platform : ℕ)
  (hL : L = 240)
  (hT_platform : T_platform = 89)
  (hL_platform : L_platform = 650) :
  let v := (L + L_platform) / T_platform,
      t := L / v in
  t = 24 := by
  sorry

end train_pole_time_l119_119944


namespace fraction_meaningful_condition_l119_119873

theorem fraction_meaningful_condition (x : ℝ) : (1 / (x - 5)) ≠ ∞ ↔ x ≠ 5 :=
by
  sorry

end fraction_meaningful_condition_l119_119873


namespace domain_of_f1_x2_l119_119190

theorem domain_of_f1_x2 (f : ℝ → ℝ) : 
  (∀ x, -1 ≤ x ∧ x ≤ 2 → ∃ y, y = f x) → 
  (∀ x, -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2 → ∃ y, y = f (1 - x^2)) :=
by
  sorry

end domain_of_f1_x2_l119_119190


namespace f_is_odd_f_is_decreasing_range_of_f_l119_119303

noncomputable def f : ℝ → ℝ := sorry

axiom additivity : ∀ (x y : ℝ), f (x + y) = f x + f y
axiom negativity : ∀ (x : ℝ), x > 0 → f x < 0
axiom value_at_neg_one : f (-1) = 2

-- 1. Prove f is an odd function
theorem f_is_odd : ∀ (x : ℝ), f (-x) = -f x :=
by sorry

-- 2. Prove f is a decreasing function on ℝ
theorem f_is_decreasing : ∀ (x1 x2 : ℝ), x2 > x1 → f x2 < f x1 :=
by sorry

-- 3. Find the range of f on the interval [-2, 4]
theorem range_of_f : set.range (f ∘ (λ (x : ℤ), if x = -2 then -2 else if x = 4 then 4 else (x : ℝ))) = set.interval (-8 : ℝ) 4 :=
by sorry

end f_is_odd_f_is_decreasing_range_of_f_l119_119303


namespace prime_iff_factorial_mod_l119_119461

theorem prime_iff_factorial_mod (p : ℕ) : 
  Nat.Prime p ↔ (Nat.factorial (p - 1)) % p = p - 1 := by
  sorry

end prime_iff_factorial_mod_l119_119461


namespace vanya_speed_increased_by_4_l119_119500

variable (v : ℝ)
variable (h1 : (v + 2) / v = 2.5)

theorem vanya_speed_increased_by_4 (h1 : (v + 2) / v = 2.5) : (v + 4) / v = 4 := 
sorry

end vanya_speed_increased_by_4_l119_119500


namespace age_of_b_l119_119895

variables (a b c : ℕ)

-- Conditions
def condition1 := a = b + 2
def condition2 := b = 2 * c
def condition3 := a + b + c = 27

-- Theorem to prove
theorem age_of_b (h1 : condition1 a b c) (h2 : condition2 b c) (h3 : condition3 a b c) : b = 10 :=
sorry

end age_of_b_l119_119895


namespace student_age_is_24_l119_119553

-- Defining the conditions
variables (S M : ℕ)
axiom h1 : M = S + 26
axiom h2 : M + 2 = 2 * (S + 2)

-- The proof statement
theorem student_age_is_24 : S = 24 :=
by
  sorry

end student_age_is_24_l119_119553


namespace probability_of_9_heads_in_12_flips_l119_119055

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l119_119055


namespace sum_even_integers_below_82_l119_119546

theorem sum_even_integers_below_82 :
  let a : ℕ := 2
      d : ℕ := 2
      l : ℕ := 80
      n : ℕ := (l - a) / d + 1
      S : ℕ := n * (a + l) / 2
  in S = 1640 := by
  let a := 2
  let d := 2
  let l := 80
  let n := (l - a) / d + 1
  let S := n * (a + l) / 2
  show S = 1640
  sorry

end sum_even_integers_below_82_l119_119546


namespace probability_exactly_9_heads_l119_119122

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l119_119122


namespace probability_heads_9_of_12_l119_119150

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l119_119150


namespace An_odd_iff_even_perfect_square_l119_119660

/-- For any integer n ≥ 2, we define A_n as the number of positive integers m such that the distance 
from n to the nearest non-negative multiple of m is equal to the distance from n^3 to the nearest 
non-negative multiple of m. This statement proves that A_n is odd if and only if n is an even perfect 
square. -/
theorem An_odd_iff_even_perfect_square (n: ℕ) (h: n ≥ 2) : 
    let A_n := ∑ m in Finset.range (n^3 - n + 1), 
                  if ((n % m = n^3 % m) ∨ (n % m = (m - n^3 % m) % m)) then 1 else 0
    in A_n % 2 = 1 ↔ ∃ k, n = 4 * k^2 :=
by sorry

end An_odd_iff_even_perfect_square_l119_119660


namespace total_donation_l119_119220

-- Define the conditions in the problem
def Barbara_stuffed_animals : ℕ := 9
def Trish_stuffed_animals : ℕ := 2 * Barbara_stuffed_animals
def Barbara_sale_price : ℝ := 2
def Trish_sale_price : ℝ := 1.5

-- Define the goal as a theorem to be proven
theorem total_donation : Barbara_sale_price * Barbara_stuffed_animals + Trish_sale_price * Trish_stuffed_animals = 45 := by
  sorry

end total_donation_l119_119220


namespace fraction_dark_tiles_l119_119575

theorem fraction_dark_tiles (n : ℕ) (h1 : n = 8) :
  let total_tiles := n * n in
  let dark_tiles := n / 2 in
  dark_tiles / total_tiles = 1 / 16 :=
by
  -- The proof is omitted
  sorry

end fraction_dark_tiles_l119_119575


namespace statements_c_and_d_true_l119_119550

theorem statements_c_and_d_true :
  (∃ n : ℕ, n > 0 ∧ 2^n < n^2) ∧
  (∃ (a b : ℝ), a > b ∧ (∃ c : ℝ, c = 0 ∧ a * c^2 = b * c^2)) ∧
  (∀ (a b : ℝ), a > b ∧ b > 0 → a^2 > a * b ∧ a * b > b^2) ∧
  (∀ (a b c d : ℝ), a > b ∧ b > 0 ∧ c < d ∧ d < 0 → 1 / (a - c) < 1 / (b - d)) :=
by {
  split,
  { -- ∃ n : ℕ, n > 0 ∧ 2^n < n^2
    use 3,
    split,
    { norm_num },
    { norm_num },
  },
  split,
  { -- ∃ (a b : ℝ), a > b ∧ (∃ c : ℝ, c = 0 ∧ a * c^2 = b * c^2)
    use [1, 0],
    split,
    { norm_num },
    { use 0, norm_num },
  },
  split,
  { -- ∀ (a b : ℝ), a > b ∧ b > 0 → a^2 > a * b ∧ a * b > b^2
    intros a b hab,
    have h1 : a^2 > a * b,
    { exact mul_lt_mul hab.left hab.left (le_of_lt hab.right) (le_of_lt hab.right), },
    have h2 : a * b > b^2,
    { exact mul_lt_mul hab.left hab.right (le_of_lt hab.right) hab.right, },
    exact ⟨h1, h2⟩,
  },
  { -- ∀ (a b c d : ℝ), a > b ∧ b > 0 ∧ c < d ∧ d < 0 → 1 / (a - c) < 1 / (b - d)
    intros a b c d hab hcd,
    rw (div_lt_div_right _).mpr,
    { exact sub_lt_sub_right hab.left c, },
    { exact sub_pos_of_lt (sub_lt_sub_right hab.left c), },
  },
}

end statements_c_and_d_true_l119_119550


namespace part_a_l119_119559

theorem part_a (p q : ℕ) (hp : p.prime) (hq : q.prime) (h_dist : p ≠ q) (h_div : (p + q^2) ∣ (p^2 + q)) :
  (p + q^2) ∣ (p * q - 1) :=
sorry

end part_a_l119_119559


namespace vanya_faster_by_4_l119_119533

variable (v : ℝ) -- initial speed in m/s
variable (school_distance : ℝ) -- distance to school in meters
variable (time_to_school : ℝ) -- time to school in seconds
variable (increased_speed : ℝ → ℝ) -- increased speed function

-- Conditions
def initial_speed : (v > 0) := by sorry
def increased_speed_by_2 {v : ℝ} : (v + 2 > 0) := by sorry
def faster_by_2_5 {v : ℝ} : (time_to_school = (school_distance / v)) → (time_to_school / 2.5 = (school_distance / (v + 2))) := by sorry

-- Main statement to prove
theorem vanya_faster_by_4 (h1 : initial_speed v) (h2 : increased_speed_by_2 v) (h3 : faster_by_2_5 v time_to_school school_distance) :
    (time_to_school / 4 = (school_distance / (v + 4))) := by
  sorry

end vanya_faster_by_4_l119_119533


namespace total_donuts_needed_l119_119606

theorem total_donuts_needed :
  (initial_friends : ℕ) (additional_friends : ℕ) (donuts_per_friend : ℕ) (extra_donuts_per_friend : ℕ) 
  (donuts_for_Andrew : ℕ) (total_friends : ℕ) 
  (h1 : initial_friends = 2)
  (h2 : additional_friends = 2)
  (h3 : total_friends = initial_friends + additional_friends)
  (h4 : donuts_per_friend = 3)
  (h5 : extra_donuts_per_friend = 1)
  (h6 : donuts_for_Andrew = donuts_per_friend + extra_donuts_per_friend)
  :
  let initial_donuts := total_friends * donuts_per_friend,
      extra_donuts := total_friends * extra_donuts_per_friend,
      total_donuts_for_friends := initial_donuts + extra_donuts,
      total_donuts := total_donuts_for_friends + donuts_for_Andrew 
  in total_donuts = 20 := by
  sorry

end total_donuts_needed_l119_119606


namespace find_radius_q_l119_119826

-- Given conditions
variables {K1 K2 K : Type} [circle K1] [circle K2] [circle K]
variables {O1 O2 O : point} {r R : ℝ}

-- Their centers and radii
axiom K1_center : center K1 = O1
axiom K2_center : center K2 = O2
axiom K_center : center K = O
axiom K1_radius : radius K1 = r
axiom K2_radius : radius K2 = r
axiom K_radius : radius K = R

-- Given touches conditions
axiom touches_K1_K : touches K1 K
axiom touches_K2_K : touches K2 K

-- Given angle condition
axiom angle_O1_O_O2 : angle O1 O O2 = 120°

-- Prove the radius q of another circle that touches K1, K2 externally, and K internally
theorem find_radius_q (q : ℝ) (Q : circle) : 
  touches Q K1 ∧ touches Q K2 ∧ touches Q K ∧ radius Q = q →
  q = R * (R - r) / (R + 3 * r) ∨ q = 3 * R * (R - r) / (3 * R + r) := 
by
  sorry

end find_radius_q_l119_119826


namespace min_value_sin_cos_interval_l119_119260

open Real

noncomputable def min_value_sin_cos : ℝ :=
  -1

theorem min_value_sin_cos_interval :
  ∀ x ∈ Icc (-π / 12) (π / 3), (sin x)^4 - (cos x)^4 ≥ -1 :=
by
  sorry

end min_value_sin_cos_interval_l119_119260


namespace probability_9_heads_12_flips_l119_119162

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l119_119162


namespace polynomial_form_l119_119635

variable {R : Type*} [CommRing R]

theorem polynomial_form (P : R[X]) :
  (∀ x : R, (x^2 - 6*x + 8) * P.eval x = (x^2 + 2*x) * P.eval (x - 2)) →
  ∃ c : R, P = c • (X^2 * (X^2 - 4)) :=
begin
  intro h,
  sorry,
end

end polynomial_form_l119_119635


namespace probability_of_9_heads_in_12_l119_119125

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l119_119125


namespace remaining_plants_after_bugs_l119_119963

theorem remaining_plants_after_bugs (initial_plants first_day_eaten second_day_fraction third_day_eaten remaining_plants : ℕ) : 
  initial_plants = 30 →
  first_day_eaten = 20 →
  second_day_fraction = 2 →
  third_day_eaten = 1 →
  remaining_plants = initial_plants - first_day_eaten - (initial_plants - first_day_eaten) / second_day_fraction - third_day_eaten →
  remaining_plants = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end remaining_plants_after_bugs_l119_119963


namespace probability_heads_exactly_9_of_12_l119_119021

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l119_119021


namespace aluminium_atoms_l119_119910

theorem aluminium_atoms (n : ℕ) 
  (atomic_weight_Al : ℝ := 26.98) 
  (atomic_weight_O : ℝ := 16.00) 
  (atomic_weight_H : ℝ := 1.01) 
  (total_molecular_weight : ℝ := 78) 
  (oxygen_atoms : ℕ := 3) 
  (hydrogen_atoms : ℕ := 3) 
  (combined_weight_O_H : ℝ) ( := 51.03) 
  : n = 1 :=
by
  sorry

end aluminium_atoms_l119_119910


namespace probability_of_9_heads_in_12_flips_l119_119064

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l119_119064


namespace probability_heads_exactly_9_of_12_l119_119025

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l119_119025


namespace probability_exactly_9_heads_in_12_flips_l119_119106

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l119_119106


namespace vanya_faster_speed_l119_119513

def vanya_speed (v : ℝ) : Prop :=
  (v + 2) / v = 2.5

theorem vanya_faster_speed (v : ℝ) (h : vanya_speed v) : (v + 4) / v = 4 :=
by
  sorry

end vanya_faster_speed_l119_119513


namespace construct_triangle_l119_119880

theorem construct_triangle (A B C D : Point) (r1 r2 : Real) 
  (h1 : r1 = distance A B) 
  (h2 : r2 = distance A C) 
  (h3 : ∀ (D : Point), distance B D / distance C D = r1 / r2) : 
  ∃ (P : Point), distance A P = r1 ∧ distance A C = r2 ∧ distance B D / distance C D = r1 / r2 := 
sorry

end construct_triangle_l119_119880


namespace five_fridays_in_august_l119_119464

theorem five_fridays_in_august (N : ℕ) (H : ∃ k1 k2 k3 k4 k5, 
  k1 ∈ {1, 2, 3} ∧ k1 + 7 ∈ {8, 9, 10} ∧ k1 + 14 ∈ {15, 16, 17} ∧ 
  k1 + 21 ∈ {22, 23, 24} ∧ k1 + 28 ∈ {29, 30, 31} ∧ k2 = k1 + 7 ∧
  k3 = k1 + 14 ∧ k4 = k1 + 21 ∧ k5 = k1 + 28) : 
  ∃ d, d = "Friday" ∧ 
  ((∀ i ∈ (1 : ℕ)..31, (day_of_week (N, 8, i) = d) → (i ∈ finset.range 5)) :=
  sorry

end five_fridays_in_august_l119_119464


namespace largest_subset_count_l119_119942

def is_subset_valid (S : set ℕ) : Prop :=
  ∀ (x y : ℕ), x ∈ S → y ∈ S → (x = 4 * y ∨ y = 4 * x) → false

def largest_valid_subset (n : ℕ) :=
  {S : set ℕ | S ⊆ {1, ..., n} ∧ is_subset_valid S}

theorem largest_subset_count : ∃ S ∈ largest_valid_subset 150, S.card = 142 := sorry

end largest_subset_count_l119_119942


namespace prove_sufficient_and_necessary_l119_119187

-- The definition of the focus of the parabola y^2 = 4x.
def focus_parabola : (ℝ × ℝ) := (1, 0)

-- The condition that the line passes through a given point.
def line_passes_through (m b : ℝ) (p : ℝ × ℝ) : Prop := 
  p.2 = m * p.1 + b

-- Let y = x + b and the equation of the parabola be y^2 = 4x.
def sufficient_and_necessary (b : ℝ) : Prop :=
  line_passes_through 1 b focus_parabola ↔ b = -1

theorem prove_sufficient_and_necessary : sufficient_and_necessary (-1) :=
by
  sorry

end prove_sufficient_and_necessary_l119_119187


namespace geometric_sequence_sum_series_l119_119305

noncomputable def a_sequence (n : ℕ) : ℕ := 2^n + 1

theorem geometric_sequence (a : ℕ → ℕ) (a_1_eq : a 1 = 3) (a_2_eq : a 2 = 5)
  (log_seq_arith : ∃ d, ∀ n, log 2 (a (n + 1) - 1) = log 2 (a n - 1) + d) :
  ∀ n, a (n + 1) - 1 = 2 * (a n - 1) :=
sorry

theorem sum_series (a : ℕ → ℕ) (a_1_eq : a 1 = 3) (a_2_eq : a 2 = 5)
  (log_seq_arith : ∃ d, ∀ n, log 2 (a (n + 1) - 1) = log 2 (a n - 1) + d) :
  ∀ n, (∑ i in finset.range n, 1 / (a (i + 2) - a (i + 1))) = 1 - 1 / 2^n :=
sorry

end geometric_sequence_sum_series_l119_119305


namespace probability_exactly_9_heads_in_12_flips_l119_119109

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l119_119109


namespace probability_heads_in_9_of_12_flips_l119_119079

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l119_119079


namespace product_of_tangents_l119_119633

open Real

-- Define the set S
def S : Set (ℝ × ℝ) :=
  { (x, y) | x ∈ {0, 1, 2, 3, 4, 5} ∧ y ∈ {0, 1, 2, 3, 4} } \ { (0, 0) }

-- Define right triangles in S
def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ x y, 
    ((A = (x, y) ∧ B ∈ S ∧ C ∈ S ∧ 
      ((B.1 = C.1 ∨ B.2 = C.2) ∧ 
       (B ≠ C) ∧ 
       (A ≠ B) ∧ 
       (A ≠ C))))

/-- T is the set of all right triangles with vertices in S -/
def T : Set (ℝ × ℝ) :=
  {t | ∃ A B C, is_right_triangle A B C ∧ t = (A, B, C)}

-- Define the function f(t)
def f (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ :=
  let (A, B, C) = t in
  if h : is_right_triangle A B C then
    let θ := atan2 (B.2 - C.2) (B.1 - C.1)
    atan2 (A.2 - C.2) (A.1 - C.1) / θ
  else
    0

theorem product_of_tangents : (∏ t ∈ T, f t) = 1 := by
  sorry

end product_of_tangents_l119_119633


namespace max_binomial_term_l119_119651

noncomputable def binomial_term (n k : ℕ) : ℝ :=
  (n.choose k : ℝ) * (real.sqrt 11) ^ k

theorem max_binomial_term :
  (argmax (binomial_term 208) = 160) :=
by
  -- Proof required here
  sorry

end max_binomial_term_l119_119651


namespace mass_percentage_O_is_26_2_l119_119258

noncomputable def mass_percentage_O_in_Benzoic_acid : ℝ :=
  let molar_mass_C := 12.01
  let molar_mass_H := 1.01
  let molar_mass_O := 16.00
  let molar_mass_Benzoic_acid := (7 * molar_mass_C) + (6 * molar_mass_H) + (2 * molar_mass_O)
  let mass_O_in_Benzoic_acid := 2 * molar_mass_O
  (mass_O_in_Benzoic_acid / molar_mass_Benzoic_acid) * 100

theorem mass_percentage_O_is_26_2 :
  mass_percentage_O_in_Benzoic_acid = 26.2 := by
  sorry

end mass_percentage_O_is_26_2_l119_119258


namespace probability_heads_9_of_12_flips_l119_119008

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l119_119008


namespace probability_of_9_heads_in_12_flips_l119_119058

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l119_119058


namespace probability_heads_9_of_12_is_correct_l119_119034

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l119_119034


namespace probability_of_9_heads_in_12_flips_l119_119045

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l119_119045


namespace sum_of_fraction_parts_of_max_sin_cos_expression_l119_119287

theorem sum_of_fraction_parts_of_max_sin_cos_expression (A B C : ℝ) (h : A + B + C = 180) :
  let expr := (sin A * cos B + sin B * cos C + sin C * cos A) ^ 2
  let fract := 27 / 16
  let max_expr := fract
  let numerator := 27
  let denominator := 16
  numerator + denominator = 43 := 
sorry

end sum_of_fraction_parts_of_max_sin_cos_expression_l119_119287


namespace probability_9_heads_12_flips_l119_119156

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l119_119156


namespace quadrilateral_diagonals_perpendicular_concurrence_l119_119467

theorem quadrilateral_diagonals_perpendicular_concurrence {A B C D P A' B' C' D' : ℝ^2} 
  (h_convex: convex_hull ({A, B, C, D} : set (ℝ^2)))
  (h_perpendicular_diagonals: ∀ P (h1: segment A C ∩ segment B D = {P}), AC ⊥ BD)
  (h_circumcenter_A': is_circumcenter_of A' A B D)
  (h_circumcenter_B': is_circumcenter_of B' B C A)
  (h_circumcenter_C': is_circumcenter_of C' C D B)
  (h_circumcenter_D': is_circumcenter_of D' D A C) 
  : concurrent ({AA', BB', CC', DD'} : set (line ℝ^2)) := 
by sorry

end quadrilateral_diagonals_perpendicular_concurrence_l119_119467


namespace log_equation_solution_l119_119703

theorem log_equation_solution (a : ℕ) : 10 - 2 * a > 0 ∧ a - 2 > 0 ∧ a - 2 ≠ 1 → a = 4 :=
by
  intro h
  have h1 : 10 - 2 * a > 0 := h.1
  have h2 : a - 2 > 0 := h.2.1
  have h3 : a - 2 ≠ 1 := h.2.2
  have ha := Nat.lt_of_add_lt_add_right h1
  have hb := Nat.lt_of_add_lt_add_left (lt_of_not_ge h3)
  have hc := Nat.ne_of_lt ha (ne_of_eq_of_ne (sub_eq_zero_of_eq (lt_of_not_ge h3))
  exact sorry

end log_equation_solution_l119_119703


namespace part1_eval_integral_result_l119_119249

noncomputable def complex_eval : ℂ :=
  let i : ℂ := Complex.i in
  let part1 := (-1 + i) * i^100 + ((1 - i) / (1 + i))^5 in
  let part2 := ((1 + i) / Real.sqrt 2)^20 in
  part1 ^ 2017 - part2

theorem part1_eval : complex_eval = -2 * Complex.i := sorry

noncomputable def integral_eval : ℝ :=
  ∫ (x : ℝ) in -1..1, 3 * Real.tan x + Real.sin x - 2 * x^3 + Real.sqrt(16 - (x - 1)^2)

theorem integral_result : integral_eval = (4 * Real.pi / 3 + 2 * Real.sqrt 3) := sorry

end part1_eval_integral_result_l119_119249


namespace min_value_expression_l119_119447

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_condition : a^2 * b + b^2 * c + c^2 * a = 3) : 
  (sqrt (a^6 + b^4 * c^6) / b + 
   sqrt (b^6 + c^4 * a^6) / c + 
   sqrt (c^6 + a^4 * b^6) / a) ≥ 3 * sqrt 2 :=
by
  sorry

end min_value_expression_l119_119447


namespace ratio_of_areas_of_triangles_l119_119752

theorem ratio_of_areas_of_triangles
  (A B C M N P : Type)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)
  (angle_A_eq : angle_A = 60)
  (angle_B_eq : angle_B = 45)
  (angle_sum_eq : angle_A + angle_B + angle_C = 180)
  (h_ABC : triangle A B C)
  (h_MNP : triangle M N P)
  (h_perpendicular_A : is_perpendicular (height A) B C)
  (h_perpendicular_B : is_perpendicular (height B) A C)
  (h_perpendicular_C : is_perpendicular (height C) A B)
  (h_circumcircle : circumcircle A B C M N P) :
  area_ratio (triangle A B C) (triangle M N P) = real.sqrt 2 * real.sin (75 * π / 180) :=
sorry

end ratio_of_areas_of_triangles_l119_119752


namespace guilty_D_l119_119491

def isGuilty (A B C D : Prop) : Prop :=
  ¬A ∧ (B → ∃! x, x ≠ A ∧ (x = C ∨ x = D)) ∧ (C → ∃! x₁ x₂, x₁ ≠ x₂ ∧ x₁ ≠ A ∧ x₂ ≠ A ∧ ((x₁ = B ∨ x₁ = D) ∧ (x₂ = B ∨ x₂ = D))) ∧ (¬A ∨ B ∨ C ∨ D)

theorem guilty_D (A B C D : Prop) (h : isGuilty A B C D) : D :=
by
  sorry

end guilty_D_l119_119491


namespace truck_speed_kmph_l119_119946

theorem truck_speed_kmph (d : ℕ) (t : ℕ) (km_m : ℕ) (hr_s : ℕ) 
  (h1 : d = 600) (h2 : t = 20) (h3 : km_m = 1000) (h4 : hr_s = 3600) : 
  (d / t) * (hr_s / km_m) = 108 := by
  sorry

end truck_speed_kmph_l119_119946


namespace sqrt_50_value_l119_119783

noncomputable def f (x : ℝ) : ℝ :=
  if x % 1 = 0 then 7 * x + 3 else floor x + 6

theorem sqrt_50_value : f (Real.sqrt 50) = 13 := by
  sorry

end sqrt_50_value_l119_119783


namespace trihedral_angle_properties_l119_119596

-- Define what it means to be an orthogonal trihedral angle
def orthogonal_trihedral (O A B C : Point) : Prop :=
  ∠ AOB = 90 ∧ ∠ BOC = 90 ∧ ∠ COA = 90

-- The main statement to be proved
theorem trihedral_angle_properties {O A B C: Point} (h: orthogonal_trihedral O A B C) (ha : A ≠ O) (hb : B ≠ O) (hc : C ≠ O) :
  (triangle_acute A B C) ∧ (projection O (plane A B C) = orthocenter A B C) :=
begin
  sorry,
end

end trihedral_angle_properties_l119_119596


namespace distance_calculation_l119_119568

-- Define the problem with the given conditions
def distance_covered(car_time: ℕ, factor: ℚ, required_speed: ℕ) : ℕ :=
  let initial_time := car_time
  let new_time := (factor * initial_time.toRat).toNat -- convert to ℕ after multiplication by rational factor
  new_time * required_speed

-- Example parameters based on the problem statement
def car_time : ℕ := 6 
def factor : ℚ := 3 / 2
def required_speed : ℕ := 60

-- Define the theorem we want to prove
theorem distance_calculation : distance_covered car_time factor required_speed = 540 :=
by 
  -- We provide the proof given the conditions
  -- Convert 6 hours to a rational
  have h1 : (6 : ℚ) * (3 / 2) = 9, by norm_num,
  -- Calculate the distance covered
  have h2 : 9 * required_speed = 540, by norm_num,
  -- Conclude the proof by using these facts
  rw [distance_covered, h1, h2],
  sorry

end distance_calculation_l119_119568


namespace plants_remaining_l119_119964

theorem plants_remaining (plants_initial plants_first_day plants_second_day_eaten plants_third_day_eaten : ℕ)
  (h1 : plants_initial = 30)
  (h2 : plants_first_day = 20)
  (h3 : plants_second_day_eaten = (plants_initial - plants_first_day) / 2)
  (h4 : plants_third_day_eaten = 1)
  : (plants_initial - plants_first_day - plants_second_day_eaten - plants_third_day_eaten) = 4 := 
by
  sorry

end plants_remaining_l119_119964


namespace probability_heads_in_nine_of_twelve_flips_l119_119086

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l119_119086


namespace percentage_of_men_l119_119733

theorem percentage_of_men (M : ℝ) 
  (h1 : 0 < M ∧ M < 1) 
  (h2 : 0.2 * M + 0.4 * (1 - M) = 0.3) : M = 0.5 :=
by
  sorry

end percentage_of_men_l119_119733


namespace total_bears_and_foxes_l119_119645

variable {A_black A_white A_brown A_foxes : ℕ}
variable {B_black B_white B_brown B_foxes : ℕ}
variable {C_black C_white C_brown C_foxes : ℕ}

-- Park A conditions
axiom A_black_value : A_black = 60
axiom A_white_value : A_white = A_black / 2
axiom A_brown_value : A_brown = A_black + 40
axiom A_foxes_value : A_foxes = 0.2 * (A_black + A_white + A_brown)

-- Park B conditions
axiom B_black_value : B_black = 3 * A_black
axiom B_white_value : B_white = B_black / 3
axiom B_brown_value : B_brown = B_black + 70
axiom B_foxes_value : B_foxes = 0.2 * (B_black + B_white + B_brown)

-- Park C conditions
axiom C_black_value : C_black = 2 * B_black
axiom C_white_value : C_white = C_black / 4
axiom C_brown_value : C_brown = C_black + 100
axiom C_foxes_value : C_foxes = 0.2 * (C_black + C_white + C_brown)

theorem total_bears_and_foxes :
  A_black + A_white + A_brown + A_foxes +
  B_black + B_white + B_brown + B_foxes +
  C_black + C_white + C_brown + C_foxes = 1908 := 
sorry

end total_bears_and_foxes_l119_119645


namespace find_max_a_l119_119691

noncomputable def f (a x : ℝ) : ℝ := - (1 / 3) * x^3 + (1 / 2) * x^2 + 2 * a * x

theorem find_max_a {a : ℝ} (h : -1/12 ≤ a ∧ a < 0) :
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 → |f a x1 - f a x2| ≤ (sqrt (8 * a + 1)) / 12) ↔ a = -1/16 :=
sorry

end find_max_a_l119_119691


namespace sum_lent_out_l119_119468

noncomputable def compound_interest (P R : ℝ) (n : ℕ) : ℝ :=
  P * ((1 + R/100)^n) - P

noncomputable def simple_interest (P R T : ℝ) : ℝ :=
  P * R * T / 100

theorem sum_lent_out (R T : ℝ) (d : ℝ) (hR : R = 10) (hT : T = 1) (hd : d = 25) :
  ∃ P : ℝ,
    let CI := compound_interest P (R/2) (2 * T).to_nat
    let SI := simple_interest P R T
    CI - SI = d ∧ P = 10000 :=
by
  sorry

end sum_lent_out_l119_119468


namespace problem_statement_l119_119470

noncomputable def y_function (A ω φ : ℝ) (x : ℝ) : ℝ :=
  A * sin (ω * x + φ)

theorem problem_statement
  (A ω φ : ℝ)
  (hA : A = 2)
  (hω : ω = 3)
  (hφ : φ = -π / 4)
  (h1 : A > 0)
  (h2 : ω > 0)
  (h3 : abs φ < π / 2)
  (h4 : y_function A ω φ (π / 4) = 2)
  (h5 : y_function A ω φ (7 * π / 12) = -2) :
  ∃ f : ℝ → ℝ, 
    (∀ x, f x = y_function A ω φ x) ∧
    (f (π / 4) = 2) ∧ 
    (f (7 * π / 12) = -2) ∧
    (∀ x, f x = sqrt 3 → ∃ k : ℤ, x ∈ {(2 / 3 : ℝ) * ↑k * π + 7 * π / 36, (2 / 3 : ℝ) * ↑k * π + 11 * π / 36,
                                         (2 / 3 : ℝ) * ↑k * π + 31 * π / 36, (2 / 3 : ℝ) * ↑k * π + 35 * π / 36,
                                         (2 / 3 : ℝ) * ↑k * π + 55 * π / 36, (2 / 3 : ℝ) * ↑k * π + 59 * π / 36}) ∧
    (1 < A → A < 2 → finset.sum (finset.filter (λ x, 0 ≤ x ∧ x ≤ 2 * π)
      (finset.map finset.univ (λ k : ℕ, (π / 2) + (4 / 3) * ↑k * π))) = 11 * π / 2)
:=
sorry

end problem_statement_l119_119470


namespace roots_of_quadratic_eq_l119_119723

theorem roots_of_quadratic_eq (c : ℝ)
  (h : ∃ r : ℝ, r^2 - 3*r + c = 0 ∧ (-r)^2 + 3*(-r) - c = 0) : 
  (0^2 - 3*0 + c = 0 ∧ 3^2 - 3*3 + c = 0) :=
begin
  sorry
end

end roots_of_quadratic_eq_l119_119723


namespace trajectory_of_moving_point_line_passing_fixed_point_l119_119370

-- Define the basic setup and conditions
def point (x y : ℝ) : ℝ × ℝ := (x, y)

def circle_passes_through_origin (P M O : ℝ × ℝ) : Prop :=
  let ⟨px, py⟩ := P
  let ⟨mx, my⟩ := M
  let ⟨ox, oy⟩ := O
  (px - ox) * (mx - ox) + (py - oy) * (my - oy) = 0

-- Conditions: Points P and M with the circle passing through origin
def conditions (P M : ℝ × ℝ) : Prop :=
  let O := (0, 0) in
  circle_passes_through_origin P M O

-- The theorem to prove the equation of the trajectory
theorem trajectory_of_moving_point (P : ℝ × ℝ) (h : conditions P (P.1, -4)) :
  P.2 = (1/4) * (P.1 ^ 2) :=
sorry

-- Define symmetry with respect to the y-axis
def symmetric_point (A : ℝ × ℝ) : ℝ × ℝ :=
  (-A.1, A.2)

-- The theorem to determine if line A'B always passes through a fixed point
theorem line_passing_fixed_point (A B : ℝ × ℝ) (E : ℝ × ℝ) (l : ℝ → ℝ) (W : ℝ → ℝ)
  (h_traj : ∀ x, W x = (1/4) * (x ^ 2)) (h_E : E = (0, -4))
  (h_A_on_W : W A.1 = A.2) (h_B_on_W : W B.1 = B.2)
  (h_line : ∀ x, l x = x - 4) (A' : ℝ × ℝ := symmetric_point A) :
  ∃ F : ℝ × ℝ, F = (0, 4) ∧ ∀ x, (A'.2 - B.2) * x = (A'.2 - B.2) * (x - A'.1) + B.2 → x = F.1 ∧ (A'.2 - B.2) * x + 4 = F.2 :=
sorry

end trajectory_of_moving_point_line_passing_fixed_point_l119_119370


namespace kerosene_cost_is_48_cents_l119_119554

-- Define the conditions
def dozen_eggs_cost_pound_rice (pound_rice_cost : ℝ) (dozen_eggs_cost : ℝ) : Prop := dozen_eggs_cost = pound_rice_cost
def half_liter_kerosene_cost_8_eggs (egg_cost : ℝ) (half_liter_kerosene_cost : ℝ) : Prop := half_liter_kerosene_cost = 8 * egg_cost
def pound_rice_cost : ℝ := 0.36
def dollar_to_cents : ℝ := 100

-- Calculate egg cost and kerosene cost based on given conditions
def egg_cost : ℝ := pound_rice_cost / 12
def half_liter_kerosene_cost : ℝ := 8 * egg_cost
def liter_kerosene_cost : ℝ := 2 * half_liter_kerosene_cost
def liter_kerosene_cost_cents : ℝ := liter_kerosene_cost * dollar_to_cents

-- Theorem to prove the conclusion
theorem kerosene_cost_is_48_cents (h1 : dozen_eggs_cost_pound_rice pound_rice_cost (12 * egg_cost))
                                   (h2 : half_liter_kerosene_cost_8_eggs egg_cost half_liter_kerosene_cost) :
  liter_kerosene_cost_cents = 48 :=
  sorry

end kerosene_cost_is_48_cents_l119_119554


namespace work_completion_time_l119_119898

-- Given conditions in Lean Definitions
def efficiency_q := 1 -- efficiency of q in units of work per day
def efficiency_p := 1.2 * efficiency_q -- p is 20% more efficient than q
def time_p := 22 -- p can complete the work in 22 days
def total_work := efficiency_p * time_p -- calculation of the total work p does
def efficiency_r := 0.7 * efficiency_q -- r is 30% less efficient than q
def combined_efficiency := efficiency_p + efficiency_q + efficiency_r -- combined efficiency of p, q, and r

-- Proving that the combined days taken to complete the work equals 9.1 days
def total_time := total_work / combined_efficiency -- time to complete work by p, q, and r together

theorem work_completion_time :
  total_time = 26.4 / 2.9 :=
by
  simp [total_work, combined_efficiency, total_time]
  norm_num
  sorry

end work_completion_time_l119_119898


namespace dollar_symmetric_l119_119663

def dollar (a b : ℝ) : ℝ := (a - b)^2

theorem dollar_symmetric {x y : ℝ} : dollar (x + y) (y + x) = 0 :=
by
  sorry

end dollar_symmetric_l119_119663


namespace butter_and_flour_calculation_l119_119481

theorem butter_and_flour_calculation
  (butter_to_sugar_ratio: ℕ) (flour_to_sugar_ratio: ℕ) (sugar_ratio: ℕ)
  (sugar_in_cups: ℕ) :
  butter_to_sugar_ratio = 1 →
  flour_to_sugar_ratio = 6 →
  sugar_ratio = 4 →
  sugar_in_cups = 10 →
  (sugar_in_cups / sugar_ratio * butter_to_sugar_ratio = 2.5) ∧ 
  (sugar_in_cups / sugar_ratio * flour_to_sugar_ratio = 15) :=
by
  intros h1 h6 h4 h10
  sorry

end butter_and_flour_calculation_l119_119481


namespace parking_lot_perimeter_l119_119586

theorem parking_lot_perimeter (a b : ℝ) 
  (h_diag : a^2 + b^2 = 784) 
  (h_area : a * b = 180) : 
  2 * (a + b) = 68 := 
by 
  sorry

end parking_lot_perimeter_l119_119586


namespace math_problem_proof_l119_119618

noncomputable def math_problem : Prop := 
  (4 - Real.sqrt 3)^0 - 3 * Real.tan (Real.pi / 3) - (-1 / 2)^(-1) + Real.sqrt 12 = 3 - Real.sqrt 3

theorem math_problem_proof : math_problem := by
  sorry

end math_problem_proof_l119_119618


namespace number_of_prime_sums_is_6_l119_119840

def is_prime (n : ℕ) : Prop := Nat.Prime n

-- List of the first 15 primes
def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

-- Function to compute cumulative sums of a list
def cumulative_sums : List ℕ → List ℕ
| [] => []
| (h::t) => h :: List.map (λ x, h + x) (cumulative_sums t)

def first_15_sums := cumulative_sums primes |>.take 15

-- Problem statement
theorem number_of_prime_sums_is_6 :
  (first_15_sums.filter is_prime).length = 6 :=
sorry

end number_of_prime_sums_is_6_l119_119840


namespace probability_heads_9_of_12_is_correct_l119_119038

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l119_119038


namespace cos_of_pi_over_3_minus_alpha_l119_119275

theorem cos_of_pi_over_3_minus_alpha (α : Real) (h : Real.sin (Real.pi / 6 + α) = 2 / 3) :
  Real.cos (Real.pi / 3 - α) = 2 / 3 :=
by
  sorry

end cos_of_pi_over_3_minus_alpha_l119_119275


namespace probability_of_9_heads_in_12_flips_l119_119047

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l119_119047


namespace river_length_l119_119475

theorem river_length (x : ℝ) (h1 : 3 * x + x = 80) : x = 20 :=
sorry

end river_length_l119_119475


namespace probability_of_9_heads_in_12_flips_l119_119042

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l119_119042


namespace intersection_at_7_m_l119_119832

def f (x : Int) (d : Int) : Int := 4 * x + d

theorem intersection_at_7_m (d m : Int) (h₁ : f 7 d = m) (h₂ : 7 = f m d) : m = 7 := by
  sorry

end intersection_at_7_m_l119_119832


namespace no_non_negative_solutions_l119_119189

theorem no_non_negative_solutions (a b : ℕ) (h_diff : a ≠ b) (d := Nat.gcd a b) 
                                 (a' := a / d) (b' := b / d) (n := d * (a' * b' - a' - b')) :
  ¬ ∃ x y : ℕ, a * x + b * y = n := 
by
  sorry

end no_non_negative_solutions_l119_119189


namespace find_z_coordinate_of_point_on_line_l119_119202

theorem find_z_coordinate_of_point_on_line (x1 y1 z1 x2 y2 z2 x_target : ℝ) 
(h1 : x1 = 1) (h2 : y1 = 3) (h3 : z1 = 2) 
(h4 : x2 = 4) (h5 : y2 = 4) (h6 : z2 = -1)
(h_target : x_target = 7) : 
∃ z_target : ℝ, z_target = -4 := 
by {
  sorry
}

end find_z_coordinate_of_point_on_line_l119_119202


namespace power_of_7_mod_8_l119_119883

theorem power_of_7_mod_8 : 7^123 % 8 = 7 :=
by sorry

end power_of_7_mod_8_l119_119883
