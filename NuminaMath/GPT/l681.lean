import Mathlib

namespace sequence_a_sum_b_l681_681634

noncomputable def a (n : ℕ) : ℝ := 2^((2 * n - 1) / 2)
noncomputable def b (n : ℕ) : ℝ := 1 / (Real.log2 (a n) * Real.log2 (a (n + 1)))

theorem sequence_a (n : ℕ) (hn : n > 0) : 
  a n * a (n + 1) = 4^n := sorry

theorem sum_b (n : ℕ) : 
  ∑ i in Finset.range n, b (i + 1) = 4 * n / (2 * n + 1) := sorry

end sequence_a_sum_b_l681_681634


namespace ab_value_l681_681218

theorem ab_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 30) (h4 : 3 * a * b + 5 * a = 4 * b + 180) : a * b = 29 :=
sorry

end ab_value_l681_681218


namespace length_OP_is_2_l681_681161

-- Define the given conditions
def unit_circle_center (O : Point) : Prop :=
  ∀ (P : Point) (A B : Point),
  is_unit_circle O ∧ P ∉ circle O 1 ∧
  tangent_line P A ∧ tangent_line P B ∧
  ∠AOP = 60 ∧ ∠BOP = 60
  
-- Define the proof problem
theorem length_OP_is_2 (O P : Point) (h : unit_circle_center O) : distance O P = 2 :=
sorry

end length_OP_is_2_l681_681161


namespace largest_n_rational_sqrt_l681_681191

theorem largest_n_rational_sqrt : ∃ n : ℕ, 
  (∀ k l : ℤ, k = Int.natAbs (Int.sqrt (n - 100)) ∧ l = Int.natAbs (Int.sqrt (n + 100)) → 
  k + l = 100) ∧ 
  (n = 2501) :=
by
  sorry

end largest_n_rational_sqrt_l681_681191


namespace integral_inequality_of_derivative_l681_681300

variable {f : ℝ → ℝ} {x₀ : ℝ}

theorem integral_inequality_of_derivative (h₀ : 0 ≤ x₀) (h₁ : x₀ ≤ 1)
  (hf :  ∀ x, 0 ≤ x ∧ x ≤ 1 → Differentiable ℝ f) 
  (hf₀ : f x₀ = 0) :
  ∫ (x : ℝ) in 0..1, (f x)^2 ≤ 4 * ∫ (x : ℝ) in 0..1, (deriv f x)^2 :=
by
  sorry

end integral_inequality_of_derivative_l681_681300


namespace arthur_walked_total_distance_l681_681101

-- Define the total number of blocks walked.
def total_blocks_walked (blocks_west : ℕ) (blocks_south : ℕ) : ℕ :=
  blocks_west + blocks_south

-- Define the distance per block.
def distance_per_block_miles : ℝ := 1 / 4

-- Calculate the total distance walked.
noncomputable def total_distance_walked (blocks_west blocks_south : ℕ) : ℝ :=
  total_blocks_walked blocks_west blocks_south * distance_per_block_miles

-- Prove the total distance walked by Arthur.
theorem arthur_walked_total_distance (blocks_west blocks_south : ℕ) (h_blocks_west : blocks_west = 8)
(h_blocks_south : blocks_south = 10) : total_distance_walked blocks_west blocks_south = 4.5 :=
by
  -- Given specific values for blocks_west and blocks_south, plug in these values.
  rw [h_blocks_west, h_blocks_south]
  -- Evaluate the total distance walked.
  show total_distance_walked 8 10 = 4.5
  -- Using definition of total_distance_walked and total_blocks_walked
  sorry

end arthur_walked_total_distance_l681_681101


namespace num_subsets_of_S_l681_681240

-- Definitions:
variable (a b : ℝ)
def M : Set ℝ := {a, b, -(a + b)}
def P : Set ℝ := {1, 0, -1}

-- Mapping condition:
axiom fG : ∀ x ∈ M, x ∈ P

-- Possible (a, b) pairs:
def valid_pairs : Set (ℝ × ℝ) := {
  (1, 0), (-1, 0), (1, -1), (0, 1), (0, -1), (-1, 1)}

-- The set S of valid pairs
def S : Set (ℝ × ℝ) := valid_pairs

-- Statement to prove the number of subsets of S is 64:
theorem num_subsets_of_S : 2 ^ (S.to_finset.card) = 64 := sorry

end num_subsets_of_S_l681_681240


namespace exponentiation_example_l681_681722

theorem exponentiation_example (n b y : ℝ) (h1 : n = 2 ^ 0.15) (h2 : b = 33.333333333333314) (h3 : n^b = y) : y = 32 :=
by
  sorry

end exponentiation_example_l681_681722


namespace arithmetic_mean_is_39_l681_681849

noncomputable def arithmetic_sequence_mean (n : ℕ) : ℕ :=
  let a₁ := 5
  let d := 2
  let aₙ := a₁ + (n - 1) * d
  (nat.sum (range n) (λ i, a₁ + i * d)) / n

theorem arithmetic_mean_is_39 :
  arithmetic_sequence_mean 35 = 39 := 
sorry

end arithmetic_mean_is_39_l681_681849


namespace remainder_a_mod_4_l681_681310

def a : ℕ := (finset.range 1001).sum (λ n, 3^n)

theorem remainder_a_mod_4: a % 4 = 1 := 
by {
  sorry
}

end remainder_a_mod_4_l681_681310


namespace problem_1_problem_2_problem_3_l681_681216

variable (α : ℝ)
variable (tan_alpha_two : Real.tan α = 2)

theorem problem_1 : (2 * Real.cos α + 3 * Real.sin α) / (3 * Real.cos α + Real.sin α) = 8 / 5 :=
by
  sorry

theorem problem_2 : (Real.cos α ^ 2 + Real.sin α * Real.cos α) / (2 * Real.sin α * Real.cos α + Real.sin α ^ 2) = 3 / 8 :=
by
  sorry

theorem problem_3 : (Real.sin α ^ 2 - Real.sin α * Real.cos α + 2) = 12 / 5 :=
by
  sorry

end problem_1_problem_2_problem_3_l681_681216


namespace expressions_equality_l681_681475

-- Defining the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Defining the permutation (arrangement) function P(n, k) = n! / (n - k)!
def perm (n k : ℕ) : ℕ :=
  fact n / fact (n - k)

-- The math problem in Lean statement
theorem expressions_equality (n : ℕ) :
  perm n (n-1) = fact n ∧ (perm (n+1) (n+1) / (n+1)) = fact n :=
by sorry

end expressions_equality_l681_681475


namespace sum_arithmetic_sequence_l681_681120

theorem sum_arithmetic_sequence : ∀ (a d l : ℕ), 
  (d = 2) → (a = 2) → (l = 20) → 
  ∃ (n : ℕ), (l = a + (n - 1) * d) ∧ 
  (∑ k in Finset.range n, (a + k * d)) = 110 :=
by
  intros a d l h_d h_a h_l
  use 10
  split
  · sorry
  · sorry

end sum_arithmetic_sequence_l681_681120


namespace find_line_equation_l681_681669

-- Definitions for the conditions
def point := (ℝ × ℝ)

def A : point := (3, 3)
def B : point := (5, 2)

def l1 : ℝ × ℝ → Prop := λ (x y), 3 * x - y - 1 = 0
def l2 : ℝ × ℝ → Prop := λ (x y), x + y - 3 = 0

-- Proposition to be proved
theorem find_line_equation (P : point) (P_intersection_l1_l2 : l1 P.1 P.2 ∧ l2 P.1 P.2) 
    (dist_A_l : ∀ l : ℝ × ℝ → Prop, l A.1 A.2 → l A.1 A.2) (dist_B_l : ∀ l : ℝ × ℝ → Prop, l B.1 B.2 → l B.1 B.2) :
    (∃ l : ℝ × ℝ → Prop, (l P.1 P.2 ∧ (l (3,3) ↔ l (5,2))) ∧ 
        (∀ x y, l x y ↔ (x + 2 * y - 5 = 0 ∨ x - 6 * y + 11 = 0))) :=
by
  -- Placeholder for the proof
  sorry

end find_line_equation_l681_681669


namespace toothpicks_used_l681_681400

theorem toothpicks_used :
  ∀ (length width gap_length gap_width : ℕ), 
    length = 70 → width = 40 → gap_length = 10 → gap_width = 5 →
    let total_toothpicks := (length + 1) * width + (width + 1) * length in
    let gap_toothpicks := (gap_length + 1) * gap_width + (gap_width + 1) * gap_length in
    total_toothpicks - gap_toothpicks = 5595 :=
begin
  sorry
end

end toothpicks_used_l681_681400


namespace arithmetic_sequence_sum_l681_681144

-- Define the arithmetic sequence properties
def is_arithmetic_sequence (seq : ℕ → ℕ) :=
  ∀ n : ℕ, seq (n + 1) = seq n + 2

-- Define the arithmetic sequence in question
def sequence : ℕ → ℕ
| 0       := 2
| (n + 1) := sequence n + 2

-- Check that our sequence matches the properties of an arithmetic sequence
lemma sequence_is_arithmetic : is_arithmetic_sequence sequence :=
by intros n; simp [sequence]

-- Define the sum of the first n terms of the sequence
def sum_n_terms (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, sequence i

-- State the main theorem to be proven: the sum of the first 10 terms is 110
theorem arithmetic_sequence_sum : sum_n_terms 10 = 110 :=
sorry

end arithmetic_sequence_sum_l681_681144


namespace arithmetic_sequence_sum_l681_681141

-- Define the arithmetic sequence properties
def is_arithmetic_sequence (seq : ℕ → ℕ) :=
  ∀ n : ℕ, seq (n + 1) = seq n + 2

-- Define the arithmetic sequence in question
def sequence : ℕ → ℕ
| 0       := 2
| (n + 1) := sequence n + 2

-- Check that our sequence matches the properties of an arithmetic sequence
lemma sequence_is_arithmetic : is_arithmetic_sequence sequence :=
by intros n; simp [sequence]

-- Define the sum of the first n terms of the sequence
def sum_n_terms (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, sequence i

-- State the main theorem to be proven: the sum of the first 10 terms is 110
theorem arithmetic_sequence_sum : sum_n_terms 10 = 110 :=
sorry

end arithmetic_sequence_sum_l681_681141


namespace binom_18_6_eq_13260_l681_681518

/-- The binomial coefficient formula. -/
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The specific proof problem: compute binom(18, 6) and show that it equals 13260. -/
theorem binom_18_6_eq_13260 : binom 18 6 = 13260 :=
by
  sorry

end binom_18_6_eq_13260_l681_681518


namespace only_fB_is_function_from_A_to_B_l681_681362

noncomputable def A : Set ℝ := { x | -1 < x ∧ x < 1 }
noncomputable def B : Set ℝ := { x | -1 < x ∧ x < 1 }

def fA (x : ℝ) : ℝ := 2 * x
def fB (x : ℝ) : ℝ := abs x
noncomputable def fC (x : ℝ) : ℝ := x^(1 / 2)
noncomputable def fD (x : ℝ) : ℝ := tan x

theorem only_fB_is_function_from_A_to_B :
  (∀ (x : ℝ), x ∈ A → fA x ∈ B) ∧
  (∀ (x : ℝ), x ∈ A → fB x ∈ B ∧
    ∀ (y z : ℝ), y, z ∈ A → fB y = fB z → y = z) ∧
  (∀ (x : ℝ), x ∈ A → fC x ∈ B) ∧
  (∀ (x : ℝ), x ∈ A → fD x ∈ B) := sorry

end only_fB_is_function_from_A_to_B_l681_681362


namespace area_of_cross_section_of_parallelepiped_l681_681574

noncomputable def area_cross_section (d : ℝ) : ℝ :=
  (2 * Real.sqrt 5 * d^2) / 12

theorem area_of_cross_section_of_parallelepiped (d : ℝ) :
  ∀ (AC1 BD A1C : ℝ)
  (P Q R : Point)
  (inclination_angle base_plane_angle : ℝ)
  (h1 : inclination_angle = π / 6)
  (h2 : base_plane_angle = π / 4)
  (h3 : AC1 = d)
  (h4 : BD = AC1)
  (h5 : A1C = AC1) :
  area_cross_section d = (2 * Real.sqrt 5 * d^2) / 12 :=
by
  sorry

end area_of_cross_section_of_parallelepiped_l681_681574


namespace blocks_per_box_l681_681324

theorem blocks_per_box (total_blocks : ℕ) (boxes : ℕ) (h1 : total_blocks = 16) (h2 : boxes = 8) : total_blocks / boxes = 2 :=
by
  sorry

end blocks_per_box_l681_681324


namespace friends_meeting_time_l681_681010

noncomputable def speed_B (t : ℕ) : ℝ := 4 + 0.75 * (t - 1)

noncomputable def distance_B (t : ℕ) : ℝ :=
  t * 4 + (0.375 * t * (t - 1))

noncomputable def distance_A (t : ℕ) : ℝ := 5 * t

theorem friends_meeting_time :
  ∃ t : ℝ, 5 * t + (t / 2) * (7.25 + 0.75 * t) = 120 ∧ t = 8 :=
by
  sorry

end friends_meeting_time_l681_681010


namespace number_of_triangles_with_perimeter_11_l681_681696

theorem number_of_triangles_with_perimeter_11 :
  {t : (ℕ × ℕ × ℕ) // let (a, b, c) := t in a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ c + a > b}.card = 4 :=
by sorry

end number_of_triangles_with_perimeter_11_l681_681696


namespace hexagon_trapezoid_area_l681_681387

theorem hexagon_trapezoid_area :
  let s := 12 in
  let h := s * sqrt 3 in
  let area := (1 / 2) * (s / 2 + s / 2) * h in
  area = 36 * sqrt 3 :=
by
  let s := 12
  let h := s * sqrt 3
  let area := (1 / 2) * (s / 2 + s / 2) * h
  sorry

end hexagon_trapezoid_area_l681_681387


namespace equal_divided_value_l681_681792

def n : ℕ := 8^2022

theorem equal_divided_value : n / 4 = 4^3032 := 
by {
  -- We state the equivalence and details used in the proof.
  sorry
}

end equal_divided_value_l681_681792


namespace number_of_factors_of_72_l681_681700

/-- 
Given that 72 can be factorized as 2^3 * 3^2, 
prove that the number of distinct positive factors of 72 is 12 
--/
theorem number_of_factors_of_72 : 
  let n := 72 in 
  let a := 3 in 
  let b := 2 in 
  n = 2 ^ a * 3 ^ b → (a + 1) * (b + 1) = 12 := 
by 
  intros n a b h,
  rw h,
  simp,
  sorry

end number_of_factors_of_72_l681_681700


namespace find_seimcircle_perimeter_approx_l681_681083

noncomputable def semicircle_perimeter (r : ℝ) : ℝ :=
  let pi := Real.pi
  12 * pi + 24

theorem find_seimcircle_perimeter_approx (r : ℝ) (h : r = 12) : semicircle_perimeter 12 ≈ 61.7 :=
by
  sorry

end find_seimcircle_perimeter_approx_l681_681083


namespace find_a2023_l681_681211

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∃ a1 : ℤ, ∀ n : ℕ, a n = a1 + n * d

theorem find_a2023 (a : ℕ → ℤ) (h_arith : arithmetic_sequence a)
  (h_cond1 : a 2 + a 7 = a 8 + 1)
  (h_cond2 : (a 4)^2 = a 2 * a 8) :
  a 2023 = 2023 := 
sorry

end find_a2023_l681_681211


namespace dodecagon_distance_squares_equal_l681_681774

theorem dodecagon_distance_squares_equal {R : ℝ} :
  let P := fin 12
  (dist_squared : P → P → ℝ) := (λ i j, (2 * R * Real.sin ((i.val - j.val : nat) * Real.pi / 12)) ^ 2) in
  dist_squared 0 1 + dist_squared 0 3 + dist_squared 0 5 + dist_squared 0 7 + dist_squared 0 9 + dist_squared 0 11 =
  dist_squared 0 2 + dist_squared 0 4 + dist_squared 0 6 + dist_squared 0 8 + dist_squared 0 10 :=
by
  sorry

end dodecagon_distance_squares_equal_l681_681774


namespace average_age_union_l681_681417

variables {A B C : Type} [Fintype A] [Fintype B] [Fintype C]
variables {age : A → ℝ} {age : B → ℝ} {age : C → ℝ}
variables (disjA : Disjoint A B) (disjB : Disjoint B C) (disjC : Disjoint A C)
variables (avgA : (∑ x in A, age x) / Fintype.card A = 30)
variables (avgB : (∑ x in B, age x) / Fintype.card B = 20)
variables (avgC : (∑ x in C, age x) / Fintype.card C = 50)
variables (avgAB : (∑ x in (A ∪ B : set (A ⊕ B)), age x) / Fintype.card (A ∪ B) = 25)
variables (avgAC : (∑ x in (A ∪ C : set (A ⊕ C)), age x) / Fintype.card (A ∪ C) = 40)
variables (avgBC : (∑ x in (B ∪ C : set (B ⊕ C)), age x) / Fintype.card (B ∪ C) = 35)

theorem average_age_union :
  (∑ x in (A ∪ B ∪ C : set (A ⊕ B ⊕ C)), age x) / Fintype.card (A ∪ B ∪ C) = 33.33 := sorry

end average_age_union_l681_681417


namespace calculate_dividend_l681_681273

theorem calculate_dividend (d q r : ℕ) (hd : d = 36) (hq : q = 19) (hr : r = 5) : 
  let n := (d * q) + r in 
  n = 689 := by
  -- The conditions are defined here.
  have hd : d = 36 := hd
  have hq : q = 19 := hq
  have hr : r = 5 := hr
  sorry  -- Proof will be filled here.

end calculate_dividend_l681_681273


namespace kim_trip_time_l681_681328

-- Definitions
def distance_freeway : ℝ := 120
def distance_mountain : ℝ := 25
def speed_ratio : ℝ := 4
def time_mountain : ℝ := 75

-- The problem statement
theorem kim_trip_time : ∃ t_freeway t_total : ℝ,
  t_freeway = distance_freeway / (speed_ratio * (distance_mountain / time_mountain)) ∧
  t_total = time_mountain + t_freeway ∧
  t_total = 165 := by
  sorry

end kim_trip_time_l681_681328


namespace volume_equality_l681_681228

theorem volume_equality 
  (r1 : ℝ → ℝ) 
  (r2 : ℝ → ℝ) 
  (R : ℝ) 
  (r : ℝ) 
  (outer_r : ℝ → ℝ) 
  (inner_r : ℝ → ℝ) :
  (∀ y, r1 y = 2 * sqrt y) →
  (∀ y, r2 y = 2 * sqrt y) →
  (R = 4) →
  (r = 2) →
  (∀ t, outer_r t = 16 - t^2) →
  (∀ t, inner_r t = 4 * t - t^2) →
  ∫ (y : ℝ) in 0..R, (π * (R^2 - (r1 y)^2)) = V₁ →
  ∫ (t : ℝ) in 0..2, (π * (outer_r t - inner_r t)) = V₂ →
  V₁ = V₂ :=
sorry

end volume_equality_l681_681228


namespace cary_net_calorie_deficit_is_250_l681_681149

-- Define the conditions
def miles_walked : ℕ := 3
def candy_bar_calories : ℕ := 200
def calories_per_mile : ℕ := 150

-- Define the function to calculate total calories burned
def total_calories_burned (miles : ℕ) (calories_per_mile : ℕ) : ℕ :=
  miles * calories_per_mile

-- Define the function to calculate net calorie deficit
def net_calorie_deficit (total_calories : ℕ) (candy_calories : ℕ) : ℕ :=
  total_calories - candy_calories

-- The statement to be proven
theorem cary_net_calorie_deficit_is_250 :
  net_calorie_deficit (total_calories_burned miles_walked calories_per_mile) candy_bar_calories = 250 :=
  by sorry

end cary_net_calorie_deficit_is_250_l681_681149


namespace ratio_of_efficiencies_l681_681936

def WorkEfficiency (days : ℕ) : ℚ := 1 / days

variable (a b : ℚ)

/-- Condition 1: b alone can do the work in 30 days -/
axiom b_eff : WorkEfficiency b = 1 / 30

/-- Condition 2: a and b together can do it in 10 days -/
axiom together_eff : WorkEfficiency a + WorkEfficiency b = 1 / 10

theorem ratio_of_efficiencies (ha : a = 15) (hb : b = 30) : (WorkEfficiency a) / (WorkEfficiency b) = 2 := 
by 
    /- Let's calculate WorkEfficiency a and b -/
    let a_eff := WorkEfficiency a
    let b_eff := WorkEfficiency b
    
    have b_eff_def : b_eff = 1 / 30 := by { exact b_eff }
    have together : a_eff + b_eff = 1 / 10 := by { exact together_eff }
    
    /- With the given values of a and b -/
    have a_eff_def : a_eff = 1 / 15 := by sorry
    have ratio : (1 / 15) / (1 / 30) = 2 := by sorry
    
    exact ratio

end ratio_of_efficiencies_l681_681936


namespace intersection_impossible_l681_681076

theorem intersection_impossible :
  ∀ (A : Matrix (Fin 7) (Fin 7) ℕ), (∀ i j, i ≠ j → A i j = 1 ∨ A i j = 0) → 
  (∀ i, A i i = 0) → 
  (∀ i, (∑ j, A i j) = 3) → 
  (∀ i j, A i j = A j i) → 
  False :=
begin
  sorry
end

end intersection_impossible_l681_681076


namespace sequence_converges_to_one_l681_681673

noncomputable def sequence (a : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then a
  else let x := sequence a (n - 1)
       in (4 / π^2) * (arccos x + (π / 2)) * (arcsin x)

theorem sequence_converges_to_one (a : ℝ) (h : 0 < a ∧ a < 1) : 
  ∃ L : ℝ, L = 1 ∧ ∃ N : ℕ, ∀ n ≥ N, |sequence a n - L| < 0.0001 :=
begin
  sorry -- Proof required here.
end

end sequence_converges_to_one_l681_681673


namespace area_enclosed_by_curve_l681_681356

theorem area_enclosed_by_curve : 
  let curve_eq (x y : ℝ) := abs (x - 1) + abs (y - 1) = 1 in
  (area of the region enclosed by curve_eq equals 2) :=
begin
  sorry
end

end area_enclosed_by_curve_l681_681356


namespace probability_all_three_defective_l681_681462

/-- A shipment contains 500 smartphones, 85 of which are defective. 
 If a customer buys three smartphones at random, the probability 
 that all three smartphones are defective is approximately 0.0047. -/
theorem probability_all_three_defective :
  let total_smartphones := 500
  let defective_smartphones := 85
  let first_prob := (defective_smartphones : ℝ) / total_smartphones
  let second_prob := (defective_smartphones - 1 : ℝ) / (total_smartphones - 1)
  let third_prob := (defective_smartphones - 2 : ℝ) / (total_smartphones - 2)
  (first_prob * second_prob * third_prob) ≈ 0.0047 :=
by
  let total_smartphones := 500
  let defective_smartphones := 85
  let first_prob := (defective_smartphones : ℝ) / total_smartphones
  let second_prob := (defective_smartphones - 1 : ℝ) / (total_smartphones - 1)
  let third_prob := (defective_smartphones - 2 : ℝ) / (total_smartphones - 2)
  let prob_all_three := first_prob * second_prob * third_prob
  have h : prob_all_three ≈ 0.0047 := sorry
  exact h

end probability_all_three_defective_l681_681462


namespace pure_imaginary_z1z2_l681_681794

def z1 : ℂ := 3 + 2 * Complex.i
def z2 (m : ℝ) : ℂ := 1 + m * Complex.i

theorem pure_imaginary_z1z2 (m : ℝ) : (z1 * z2 m).re = 0 → m = 3 / 2 :=
by
  sorry

end pure_imaginary_z1z2_l681_681794


namespace number_of_triangles_with_perimeter_11_l681_681685

theorem number_of_triangles_with_perimeter_11 : (∃ triangles : List (ℕ × ℕ × ℕ), 
  (∀ t ∈ triangles, let (a, b, c) := t in 
    a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ a + c > b) 
  ∧ triangles.length = 10) := 
sorry

end number_of_triangles_with_perimeter_11_l681_681685


namespace area_bounded_by_curves_eq_l681_681942

open Real

noncomputable def area_bounded_by_curves : ℝ :=
  1 / 2 * (∫ (φ : ℝ) in (π/4)..(π/2), (sqrt 2 * cos (φ - π / 4))^2) +
  1 / 2 * (∫ (φ : ℝ) in (π/2)..(3 * π / 4), (sqrt 2 * sin (φ - π / 4))^2)

theorem area_bounded_by_curves_eq : area_bounded_by_curves = (π + 2) / 4 :=
  sorry

end area_bounded_by_curves_eq_l681_681942


namespace cone_height_relationship_l681_681402

theorem cone_height_relationship
  (r₁ h₁ r₂ h₂ : ℝ)
  (volume_eq : (1 / 3) * ℝ.pi * r₁^2 * h₁ = (1 / 3) * ℝ.pi * r₂^2 * h₂)
  (radius_rel : r₂ = (6 / 5) * r₁) :
  h₁ = (36 / 25) * h₂ := by
  sorry

end cone_height_relationship_l681_681402


namespace net_calorie_deficit_l681_681150

-- Define the conditions as constants.
def total_distance : ℕ := 3
def calories_burned_per_mile : ℕ := 150
def calories_in_candy_bar : ℕ := 200

-- Prove the net calorie deficit.
theorem net_calorie_deficit : total_distance * calories_burned_per_mile - calories_in_candy_bar = 250 := by
  sorry

end net_calorie_deficit_l681_681150


namespace doughnut_completion_time_l681_681948

noncomputable def time_completion : Prop :=
  let start_time : ℕ := 7 * 60 -- 7:00 AM in minutes
  let quarter_complete_time : ℕ := 10 * 60 + 20 -- 10:20 AM in minutes
  let efficiency_decrease_time : ℕ := 12 * 60 -- 12:00 PM in minutes
  let one_quarter_duration : ℕ := quarter_complete_time - start_time
  let total_time_before_efficiency_decrease : ℕ := 5 * 60 -- from 7:00 AM to 12:00 PM is 5 hours
  let remaining_time_without_efficiency : ℕ := 4 * one_quarter_duration - total_time_before_efficiency_decrease
  let adjusted_remaining_time : ℕ := remaining_time_without_efficiency * 10 / 9 -- decrease by 10% efficiency
  let total_job_duration : ℕ := total_time_before_efficiency_decrease + adjusted_remaining_time
  let completion_time := efficiency_decrease_time + adjusted_remaining_time
  completion_time = 21 * 60 + 15 -- 9:15 PM in minutes

theorem doughnut_completion_time : time_completion :=
  by 
    sorry

end doughnut_completion_time_l681_681948


namespace perimeter_of_second_square_correct_l681_681868

noncomputable def perimeter_of_first_square := 40 -- perimeter of the first square in cm
noncomputable def area_of_first_square := (perimeter_of_first_square / 4)^2 -- area of the first square

noncomputable def perimeter_of_third_square := 24 -- perimeter of the third square in cm
noncomputable def area_of_third_square := (perimeter_of_third_square / 4)^2 -- area of the third square

noncomputable def area_of_second_square := area_of_third_square + area_of_first_square -- area of the second square

noncomputable def side_length_of_second_square := real.sqrt area_of_second_square -- side length of the second square

noncomputable def perimeter_of_second_square := 4 * side_length_of_second_square -- perimeter of the second square

theorem perimeter_of_second_square_correct : perimeter_of_second_square ≈ 46.64 := sorry

end perimeter_of_second_square_correct_l681_681868


namespace tetrahedron_intersection_iff_products_equal_l681_681335

theorem tetrahedron_intersection_iff_products_equal
  (A B C D : Point)
  (AB AC AD BC BD CD : Real)
  (I1 I2 I3 I4 : Point) :
  (lines_meet_at_same_point [A, I1] [B, I2] [C, I3] [D, I4])
  ↔ (AB * CD = AC * BD ∧ AC * BD = AD * BC) := 
sorry

end tetrahedron_intersection_iff_products_equal_l681_681335


namespace arithmetic_progression_sum_l681_681727

theorem arithmetic_progression_sum (a d : ℝ)
  (h1 : 10 * (2 * a + 19 * d) = 200)
  (h2 : 25 * (2 * a + 49 * d) = 0) :
  35 * (2 * a + 69 * d) = -466.67 :=
by
  sorry

end arithmetic_progression_sum_l681_681727


namespace factorization_problem_l681_681187

theorem factorization_problem (x : ℝ) :
  (x^4 + x^2 - 4) * (x^4 + x^2 + 3) + 10 =
  (x^2 + x + 1) * (x^2 - x + 1) * (x^2 + 2) * (x + 1) * (x - 1) :=
sorry

end factorization_problem_l681_681187


namespace magnitude_complex_number_l681_681788

theorem magnitude_complex_number (i : ℂ) (z : ℂ) (h_i : i = complex.I)
  (h_z : z = (1 + i) / 2) : complex.abs z = (real.sqrt 2) / 2 :=
by sorry

end magnitude_complex_number_l681_681788


namespace travel_time_calculation_l681_681046

theorem travel_time_calculation (speed distance : ℝ) (h1 : speed = 50) (h2 : distance = 790) :
  distance / speed = 15.8 :=
by
  rw [h1, h2]
  norm_num
  sorry

end travel_time_calculation_l681_681046


namespace problem1_problem2_l681_681053

-- Proof for Problem 1
theorem problem1 : sqrt 8 - (1/2)^(-1:ℝ) + 4 * real.sin (real.pi / 6) = 2 * sqrt 2 :=
by sorry

-- Proof for Problem 2
theorem problem2 (m : ℝ) (h : m = 2) : 
  (m^2 - 9) / (m^2 + 6 * m + 9) / (1 - 2 / (m + 3)) = -1 / 3 :=
by sorry

end problem1_problem2_l681_681053


namespace binom_18_6_eq_13260_l681_681517

/-- The binomial coefficient formula. -/
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The specific proof problem: compute binom(18, 6) and show that it equals 13260. -/
theorem binom_18_6_eq_13260 : binom 18 6 = 13260 :=
by
  sorry

end binom_18_6_eq_13260_l681_681517


namespace chairs_needed_l681_681973

def seminar_participants : Nat := 3 * 6^2 + 1 * 6^1 + 5 * 6^0
def participants_per_chair : Nat := 3

theorem chairs_needed : Nat :=
  let total_participants := seminar_participants
  let chairs := (total_participants + participants_per_chair - 1) / participants_per_chair -- rounding up division
  chairs = 40

end chairs_needed_l681_681973


namespace gcd_294_84_l681_681014

theorem gcd_294_84 : gcd 294 84 = 42 :=
by
  sorry

end gcd_294_84_l681_681014


namespace inequality_solution_l681_681196

theorem inequality_solution (x : ℝ) : 3 * x^2 - 4 * x + 7 > 0 → (1 - 2 * x) / (3 * x^2 - 4 * x + 7) ≥ 0 ↔ x ≤ 1 / 2 :=
by
  intro h
  sorry

end inequality_solution_l681_681196


namespace smallest_sum_of_three_integers_l681_681384

theorem smallest_sum_of_three_integers (a b c : ℕ) (h1: a ≠ b) (h2: b ≠ c) (h3: a ≠ c) (h4: a * b * c = 72) :
  a + b + c = 13 :=
sorry

end smallest_sum_of_three_integers_l681_681384


namespace Jack_age_l681_681484

-- Definitions based on conditions
variables (j a : ℕ)
def condition1 : Prop := j = 2 * a - 20
def condition2 : Prop := j + a = 60

-- The theorem stating that Jack's age is 33
theorem Jack_age : condition1 j a ∧ condition2 j a → j = 33 :=
by
  intros,
  sorry

end Jack_age_l681_681484


namespace james_needs_to_work_50_hours_l681_681763

def wasted_meat := 20
def cost_meat_per_pound := 5
def wasted_vegetables := 15
def cost_vegetables_per_pound := 4
def wasted_bread := 60
def cost_bread_per_pound := 1.5
def janitorial_hours := 10
def janitor_rate := 10
def time_and_half_multiplier := 1.5
def min_wage := 8

theorem james_needs_to_work_50_hours :
  let cost_meat := wasted_meat * cost_meat_per_pound in
  let cost_vegetables := wasted_vegetables * cost_vegetables_per_pound in
  let cost_bread := wasted_bread * cost_bread_per_pound in
  let time_and_half_rate := janitor_rate * time_and_half_multiplier in
  let cost_janitorial := janitorial_hours * time_and_half_rate in
  let total_cost := cost_meat + cost_vegetables + cost_bread + cost_janitorial in
  let hours_to_work := total_cost / min_wage in
  hours_to_work = 50 := by
  sorry

end james_needs_to_work_50_hours_l681_681763


namespace length_PP1P2_l681_681955

noncomputable def length_segment_P1P2 (x : ℝ) : ℝ :=
  let P1 := 0
  let P2 := Real.cos (Real.arccos (2 / 3))
  Real.abs (P2 - P1)

theorem length_PP1P2 : ∀ x ∈ set.Ioo 0 (Real.pi / 2), 
  (4 * Real.tan x = 6 * Real.sin x) → length_segment_P1P2 x = 2 / 3 :=
by
  intros x hx h
  -- skipped proof
  sorry

end length_PP1P2_l681_681955


namespace polynomial_not_factored_l681_681313

variables {R : Type*} [CommRing R]

theorem polynomial_not_factored (n : ℕ) (a : Fin n → ℤ) (h_unique : ∀ i j, a i = a j → i = j) :
  ¬ ∃ (g h : Polynomial ℤ), g.degree > 0 ∧ h.degree > 0 ∧
    (Polynomial.ofRatRingHom (RatRingHomExt R)) =
    g * h :=
  sorry

end polynomial_not_factored_l681_681313


namespace combination_18_6_l681_681502

theorem combination_18_6 : (nat.choose 18 6) = 18564 := 
by 
  sorry

end combination_18_6_l681_681502


namespace find_abc_exist_mn_l681_681610

-- Define the quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define conditions
axiom a_nonzero (a : ℝ) : a ≠ 0
axiom symmetric_condition (a b c : ℝ) : ∀ x : ℝ, quadratic_function a b c (-x + 1) = quadratic_function a b c (x + 1)
axiom f_at_2_zero (a b c : ℝ) : quadratic_function a b c 2 = 0
axiom double_root_condition (a b c : ℝ) : ∃ x : ℝ, quadratic_function a b c x = x ∧ ∀ y : ℝ, quadratic_function a b c y = x → y = x

theorem find_abc : ∃ a b c : ℝ, a = -1/2 ∧ b = 1 ∧ c = 0 :=
by
  sorry

theorem exist_mn : ∃ m n : ℝ, m < n ∧  
                                (∀ x : ℝ, m ≤ x ∧ x ≤ n → quadratic_function (-1/2) 1 0 x ∈ set.Icc (3 * m) (3 * n)) :=
by
  exact ⟨-4, 0, by norm_num, sorry⟩

end find_abc_exist_mn_l681_681610


namespace geometric_sequences_product_and_quotient_l681_681635

variable {R : Type*} [Field R]

-- Defining sequences {a_n} and {b_n} as geometric sequences
def is_geometric_sequence (a : ℕ → R) (q : R) : Prop :=
  ∀ n, a (n + 1) = q * a n

variable (a b : ℕ → R)
variable (q1 q2 : R)

-- Defining the conditions that {a_n} and {b_n} are geometric sequences with common ratios q1 and q2 respectively
def a_is_geometric : Prop := is_geometric_sequence a q1
def b_is_geometric : Prop := is_geometric_sequence b q2

-- The statement that {a_n * b_n} is a geometric sequence
def product_geometric_sequence : Prop :=
  ∃ q, is_geometric_sequence (λ n, a n * b n) q

-- The statement that {a_n / b_n} is a geometric sequence
def quotient_geometric_sequence : Prop :=
  ∃ q, is_geometric_sequence (λ n, a n / b n) q

-- The main theorem combining the conditions and the proof statement
theorem geometric_sequences_product_and_quotient
  (ha : a_is_geometric a q1)
  (hb : b_is_geometric b q2) :
  product_geometric_sequence a b ∧ quotient_geometric_sequence a b :=
sorry

end geometric_sequences_product_and_quotient_l681_681635


namespace midpoint_distance_half_OB_l681_681680

open EuclideanGeometry

variable {O A P B S T : point}

/-- The given conditions: -/
variable [concentric_circles : ConcentricCircles O A B]
variable [on_small_circle : OnCircle A P]
variable [on_large_circle : OnCircle B]
variable [perpendicular : Perpendicular AP BP]
variable [midpoint_S : Midpoint S A B]
variable [midpoint_T : Midpoint T O P]

/-- The theorem to be proven: -/
theorem midpoint_distance_half_OB :
  distance S T = (1 / 2) * distance O B := by
  sorry

end midpoint_distance_half_OB_l681_681680


namespace floor_47_l681_681552

theorem floor_47 : Int.floor 4.7 = 4 :=
by
  sorry

end floor_47_l681_681552


namespace bears_on_each_shelf_l681_681985

theorem bears_on_each_shelf 
    (initial_bears : ℕ) (shipment_bears : ℕ) (shelves : ℕ)
    (h1 : initial_bears = 4) (h2 : shipment_bears = 10) (h3 : shelves = 2) :
    (initial_bears + shipment_bears) / shelves = 7 := by
  sorry

end bears_on_each_shelf_l681_681985


namespace secant_tangent_property_l681_681082

theorem secant_tangent_property (A B C D : Point)
  (ABC : Triangle A B C) (ABD : Triangle A B D) (ACD : Triangle A C D)
  (r : ℝ) (s s1 s2 : ℝ) (t a : ℝ)
  (hr1 : inradius ABD = r) (hr2 : inradius ACD = r)
  (hs : semi_perimeter ABC = s)
  (ha : side BC = a) :
  segment_secat A t BC = s - a :=
by sorry

end secant_tangent_property_l681_681082


namespace pyramid_height_eq_3_75_l681_681445

-- Define the edge length of the cube
def cube_edge_length : ℝ := 5

-- Define the base edge length of the pyramid
def pyramid_base_edge_length : ℝ := 10

-- Define the volume of the cube
def V_cube : ℝ := cube_edge_length ^ 3

-- Define the volume of the pyramid
def V_pyramid (h : ℝ) : ℝ := (1 / 3) * (pyramid_base_edge_length ^ 2) * h

-- Proof that the height of the pyramid is 3.75 units
theorem pyramid_height_eq_3_75 :
  ∃ h : ℝ, V_cube = V_pyramid h ∧ h = 3.75 :=
by
  use 3.75
  split
  . sorry
  . rfl

end pyramid_height_eq_3_75_l681_681445


namespace compare_abc_l681_681784

noncomputable def a : ℝ := 2 ^ 1.5
noncomputable def b : ℝ := Real.log 1.5 / Real.log (1/2)
noncomputable def c : ℝ := (1/2) ^ 1.5

theorem compare_abc : a > c ∧ c > b := 
by
  -- Using the definitions and the properties of exponential and logarithmic functions
  -- we skip the proof for now with 'sorry'
  sorry

end compare_abc_l681_681784


namespace equal_divided_value_l681_681791

def n : ℕ := 8^2022

theorem equal_divided_value : n / 4 = 4^3032 := 
by {
  -- We state the equivalence and details used in the proof.
  sorry
}

end equal_divided_value_l681_681791


namespace greatest_base8_three_digit_divisible_by_7_l681_681907

theorem greatest_base8_three_digit_divisible_by_7 :
  ∃ n : ℕ, n < 8^3 ∧ n ≥ 8^2 ∧ (n % 7 = 0) ∧ (to_base 8 n = 777) :=
sorry

end greatest_base8_three_digit_divisible_by_7_l681_681907


namespace product_evaluation_l681_681184

theorem product_evaluation :
  (∏ n in Finset.range(99) + 2, (1 - (1 / n))) = (1 / 100) :=
by
  sorry

end product_evaluation_l681_681184


namespace f_6_is_4_l681_681937

def f : ℤ → ℤ
| n := if h : n ≥ 4 then (f (n - 1)) - n else sorry -- Note: Placeholder for non-recursive base cases

theorem f_6_is_4 (f : ℤ → ℤ) (h_recur : ∀ n : ℤ, n ≥ 4 → f n = (f (n - 1)) - n) (h_init : f 4 = 15) : 
  f 6 = 4 :=
by
  have f5_eq : f 5 = 10 := by sorry
  have f6_eq : f 6 = 4 := by sorry
  exact f6_eq

end f_6_is_4_l681_681937


namespace angle_relation_in_rectangle_l681_681102

theorem angle_relation_in_rectangle
  (A B C D E F G H : Point)
  (EF_l AD_l AB_l l : Line)
  (h1 : Quadrilateral A B C D)
  (h2 : Midpoint E A D)
  (h3 : Midpoint F B C)
  (h4 : On G E F)
  (h5 : Symmetric D H (PerpendicularBisector A G))
  (h6 : Perpendicular EF_l A D)
  (h7 : Parallel EF_l A B) :
  ∠ H A B = 3 * ∠ G A B := 
sorry

end angle_relation_in_rectangle_l681_681102


namespace max_hours_is_70_l681_681323

-- Define the conditions
def regular_hourly_rate : ℕ := 8
def first_20_hours : ℕ := 20
def max_weekly_earnings : ℕ := 660
def overtime_rate_multiplier : ℕ := 25

-- Define the overtime hourly rate
def overtime_hourly_rate : ℕ := regular_hourly_rate + (regular_hourly_rate * overtime_rate_multiplier / 100)

-- Define the earnings for the first 20 hours
def earnings_first_20_hours : ℕ := regular_hourly_rate * first_20_hours

-- Define the maximum overtime earnings
def max_overtime_earnings : ℕ := max_weekly_earnings - earnings_first_20_hours

-- Define the maximum overtime hours
def max_overtime_hours : ℕ := max_overtime_earnings / overtime_hourly_rate

-- Define the maximum total hours
def max_total_hours : ℕ := first_20_hours + max_overtime_hours

-- Theorem to prove that the maximum number of hours is 70
theorem max_hours_is_70 : max_total_hours = 70 :=
by
  sorry

end max_hours_is_70_l681_681323


namespace probability_product_multiple_of_4_l681_681711

theorem probability_product_multiple_of_4 :
  let cards := {1, 2, 3, 4, 5, 6}
  let pairs := { (a, b) | a ∈ cards ∧ b ∈ cards ∧ a < b }
  let total_pairs := 15
  let valid_pairs := { (1, 4), (2, 4), (3, 4), (4, 5), (4, 6) }
  let num_valid_pairs := 5
  num_valid_pairs / total_pairs = 1 / 3 := by
  sorry

end probability_product_multiple_of_4_l681_681711


namespace similar_triangles_in_cyclic_quadrilateral_l681_681740

variable {A B C D E : Type}
variable [circle : Circle A B C D] -- Assumption: A, B, C, and D are points on a circle
variable (AC BD : LineSegment A C) (BD : LineSegment B D)
variable (E : Point) -- The intersection point of diagonals

theorem similar_triangles_in_cyclic_quadrilateral :
  ∃ (P : E = intersection_point AC BD), 
    similarity (△ A B E) (△ D C E) ∧
    similarity (△ A D E) (△ B C E) :=
by
  sorry

end similar_triangles_in_cyclic_quadrilateral_l681_681740


namespace minimum_length_of_segment_l681_681471

noncomputable def minimum_segment_length {a : ℝ} (h_pos : 0 < a) : ℝ :=
  let f := λ x : ℝ, 5 * x ^ 2 - 4 * a * x + a ^ 2
  let x_min := 2 * a / 5
  real.sqrt (f x_min)

theorem minimum_length_of_segment (a : ℝ) (h_pos : 0 < a) :
  minimum_segment_length h_pos = a / real.sqrt 5 :=
sorry

end minimum_length_of_segment_l681_681471


namespace complement_U_M_l681_681242

noncomputable def U : Set ℝ := {x : ℝ | x > 0}

noncomputable def M : Set ℝ := {x : ℝ | 2 * x - x^2 > 0}

theorem complement_U_M : (U \ M) = {x : ℝ | x ≥ 2} := 
by
  sorry

end complement_U_M_l681_681242


namespace side_face_area_l681_681168

noncomputable def box_lengths (l w h : ℕ) : Prop :=
  (w * h = (1 / 2) * l * w ∧
   l * w = (3 / 2) * l * h ∧
   l * w * h = 5184 ∧
   2 * (l + h) = (6 / 5) * 2 * (l + w))

theorem side_face_area :
  ∃ (l w h : ℕ), box_lengths l w h ∧ l * h = 384 := by
  sorry

end side_face_area_l681_681168


namespace expansion_constant_term_l681_681539

theorem expansion_constant_term :
  (let k := 3 in
   let binom := Nat.choose 8 k in
   let term := binom * (5 ^ (8 - k)) * (2 ^ k) in
   term)
  = 1400000 :=
by
  sorry

end expansion_constant_term_l681_681539


namespace hundred_chickens_solution_l681_681747

theorem hundred_chickens_solution (x y : ℕ) (z : ℕ := 81) 
  (h1 : x + y + z = 100) 
  (h2 : 5 * x + 3 * y + z / 3 = 100) : 
  x = 8 ∧ y = 11 :=
by
  have h3 : z = 81 := rfl
  have h4 : 81 / 3 = 27 := by norm_num
  rw [h3, h4] at h2
  sorry

end hundred_chickens_solution_l681_681747


namespace arithmetic_sequence_sum_l681_681131

theorem arithmetic_sequence_sum :
  let sequence := list.range (20 / 2) in
  let sum := sequence.map (λ n, 2 * (n + 1)).sum in
  sum = 110 :=
by
  -- Define the sequence as the arithmetic series
  let sequence := list.range (20 / 2)
  -- Calculate the sum of the arithmetic sequence
  let sum := sequence.map (λ n, 2 * (n + 1)).sum
  -- Check the sum
  have : sum = 110 := sorry
  exact this

end arithmetic_sequence_sum_l681_681131


namespace instantaneous_velocity_at_3_l681_681968

noncomputable def particle_motion : ℝ → ℝ := 
  λ t, 1 / t^2

theorem instantaneous_velocity_at_3 :
  (deriv particle_motion 3) = -2 / 27 := 
by 
  sorry

end instantaneous_velocity_at_3_l681_681968


namespace floor_47_l681_681553

theorem floor_47 : Int.floor 4.7 = 4 :=
by
  sorry

end floor_47_l681_681553


namespace angle_MXN_is_32_l681_681329

open EuclideanGeometry

theorem angle_MXN_is_32 :
  ∃ (A B C M N X: Point) (AB BC AC: Real),
    is_isosceles_triangle ABC B C A 44 ∧
    distance A M = distance B N ∧
    distance A C = distance M A ∧
    distance B C = distance N B ∧
    is_on_ray A C N ∧
    distance M X = distance A B 
    → angle M X N = 32 := 
begin
  sorry
end

end angle_MXN_is_32_l681_681329


namespace angle_in_third_quadrant_l681_681872

theorem angle_in_third_quadrant (θ : ℝ) (k : ℤ) :
  θ = 2016 → θ % 360 = 216 →
  180 ≤ θ % 360 ∧ θ % 360 < 270 :=
by sorry

end angle_in_third_quadrant_l681_681872


namespace diamonds_in_F20_l681_681372

def F (n : ℕ) : ℕ :=
  -- Define recursively the number of diamonds in figure F_n
  match n with
  | 1 => 1
  | 2 => 9
  | n + 1 => F n + 4 * (n + 1)

theorem diamonds_in_F20 : F 20 = 761 :=
by sorry

end diamonds_in_F20_l681_681372


namespace greatest_3_digit_base8_num_div_by_7_eq_511_l681_681918

noncomputable def greatest_base8_number_divisible_by_7 : ℕ := 7 * 73

theorem greatest_3_digit_base8_num_div_by_7_eq_511 : 
  greatest_base8_number_divisible_by_7 = 511 :=
by 
  sorry

end greatest_3_digit_base8_num_div_by_7_eq_511_l681_681918


namespace min_value_in_interval_l681_681862

noncomputable def f (x : ℝ) : ℝ := x^4 - 4 * x + 3

theorem min_value_in_interval :
  ∃ x ∈ Icc (-2 : ℝ) 3, f x = 0 ∧ ∀ y ∈ Icc (-2 : ℝ) 3, f y ≥ 0 :=
by sorry

end min_value_in_interval_l681_681862


namespace retailer_profit_percent_l681_681459

theorem retailer_profit_percent (purchase_price overhead_expenses selling_price : ℝ) 
  (h_purchase : purchase_price = 225) 
  (h_overhead : overhead_expenses = 20) 
  (h_selling : selling_price = 300) :
  (selling_price - (purchase_price + overhead_expenses)) / (purchase_price + overhead_expenses) * 100 = 22.45 :=
by 
  -- Definitions and conditions
  have h_cost_price : purchase_price + overhead_expenses = 245, by rw [h_purchase, h_overhead]
  have h_profit : selling_price - (purchase_price + overhead_expenses) = 55, by rw [←h_cost_price, h_selling]
  -- Calculating the profit percent
  have h_profit_percent : ((selling_price - (purchase_price + overhead_expenses)) / (purchase_price + overhead_expenses)) * 100 = 22.44897959183673,
  by norm_num [h_profit, h_cost_price]
  -- Rounding to 2 decimal places
  have h_rounded_profit_percent : ((selling_price - (purchase_price + overhead_expenses)) / (purchase_price + overhead_expenses)) * 100 ≈ 22.45,
  by linarith [h_profit_percent]
  -- Asserting the approximate equality
  exact h_rounded_profit_percent

end retailer_profit_percent_l681_681459


namespace find_factor_l681_681964

theorem find_factor {n f : ℝ} (h1 : n = 10) (h2 : f * (2 * n + 8) = 84) : f = 3 :=
by
  sorry

end find_factor_l681_681964


namespace cake_serving_increase_l681_681427

theorem cake_serving_increase
    (initial_radius : ℝ) (num_people : ℝ) (radius_increase_percentage : ℝ) (height_same : Prop) :
    initial_radius = 20 →
    num_people = 4 →
    radius_increase_percentage = 1.5 →
    height_same →
    let new_radius := initial_radius * (1 + radius_increase_percentage) in
    let original_area := π * initial_radius^2 in
    let new_area := π * new_radius^2 in
    new_area / original_area * num_people = 25 := 
by
    intros h_radius h_people h_percentage h_height
    let new_radius := initial_radius * (1 + radius_increase_percentage)
    have h_new_radius : new_radius = 50 := by
        rw [h_radius, h_percentage]
        norm_num
    let original_area := π * initial_radius^2
    let new_area := π * new_radius^2
    have h_original_area : original_area = 400 * π := by
        rw [h_radius]
        norm_num
    have h_new_area : new_area = 2500 * π := by
        rw [h_new_radius]
        norm_num
    have h_area_ratio : new_area / original_area = 6.25 := by
        rw [h_original_area, h_new_area]
        field_simp
        norm_num
    have volume_increase : new_area / original_area * num_people = 25 := by
        rw [h_area_ratio, h_people]
        norm_num
    exact volume_increase

end cake_serving_increase_l681_681427


namespace emmalyn_earnings_l681_681181

theorem emmalyn_earnings :
  let rate_per_meter := 0.20
  let number_of_fences := 50
  let length_per_fence := 500
  let total_length := number_of_fences * length_per_fence
  let total_earnings := total_length * rate_per_meter
  total_earnings = 5000 := 
by
  let rate_per_meter := 0.20
  let number_of_fences := 50
  let length_per_fence := 500
  let total_length := number_of_fences * length_per_fence
  let total_earnings := total_length * rate_per_meter
  sorry

end emmalyn_earnings_l681_681181


namespace min_value_x_plus_y_l681_681205

theorem min_value_x_plus_y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) 
  (h : 2 * x + 8 * y = x * y) : x + y ≥ 18 :=
sorry

end min_value_x_plus_y_l681_681205


namespace combination_18_6_l681_681526

theorem combination_18_6 : nat.choose 18 6 = 18564 :=
by {
  sorry
}

end combination_18_6_l681_681526


namespace james_needs_to_work_50_hours_l681_681761

def wasted_meat := 20
def cost_meat_per_pound := 5
def wasted_vegetables := 15
def cost_vegetables_per_pound := 4
def wasted_bread := 60
def cost_bread_per_pound := 1.5
def janitorial_hours := 10
def janitor_rate := 10
def time_and_half_multiplier := 1.5
def min_wage := 8

theorem james_needs_to_work_50_hours :
  let cost_meat := wasted_meat * cost_meat_per_pound in
  let cost_vegetables := wasted_vegetables * cost_vegetables_per_pound in
  let cost_bread := wasted_bread * cost_bread_per_pound in
  let time_and_half_rate := janitor_rate * time_and_half_multiplier in
  let cost_janitorial := janitorial_hours * time_and_half_rate in
  let total_cost := cost_meat + cost_vegetables + cost_bread + cost_janitorial in
  let hours_to_work := total_cost / min_wage in
  hours_to_work = 50 := by
  sorry

end james_needs_to_work_50_hours_l681_681761


namespace rikki_poetry_sales_l681_681827

theorem rikki_poetry_sales :
  let words_per_5min := 25
  let total_minutes := 2 * 60
  let intervals := total_minutes / 5
  let total_words := words_per_5min * intervals
  let total_earnings := 6
  let price_per_word := total_earnings / total_words
  price_per_word = 0.01 :=
by
  sorry

end rikki_poetry_sales_l681_681827


namespace number_of_multiples_of_6_less_than_96_l681_681248

theorem number_of_multiples_of_6_less_than_96 :
  let lcm_2_3 := Nat.lcm 2 3 in
  let count_multiples_6 := Nat.floor (95 / lcm_2_3) in
  lcm_2_3 = 6 ∧ count_multiples_6 = 15 :=
by
  let lcm_2_3 := Nat.lcm 2 3
  let count_multiples_6 := Nat.floor (95 / lcm_2_3)
  have h1 : lcm_2_3 = 6 := sorry 
  have h2 : count_multiples_6 = 15 := sorry 
  exact ⟨h1, h2⟩

end number_of_multiples_of_6_less_than_96_l681_681248


namespace prob_one_solution_prob_point_in_fourth_quadrant_l681_681070

-- Define the given conditions
def events_space : Finset (ℕ × ℕ) := 
  Finset.product (Finset.range 6) (Finset.range 6)

def system_has_only_one_solution (a b : ℕ) : Prop := 
  ¬ (a = 2 * b)

def point_in_fourth_quadrant (a b : ℕ) : Prop := 
  (2 * b - a > 0) ∧ 
  ((3 * b - 2) / (2 * b - a) > 0) ∧
  ((4 - 3 * a) / (2 * b - a) < 0)

-- Prove that the probability that the system of equations has only one solution is 11/12
theorem prob_one_solution : 
  (Finset.filter (λ ab : ℕ × ℕ, system_has_only_one_solution ab.1 ab.2) events_space).card = 33 / 36 * events_space.card := 
sorry

-- Prove that the probability that point P lies in the fourth quadrant is 7/12
theorem prob_point_in_fourth_quadrant :
  (Finset.filter (λ ab : ℕ × ℕ, point_in_fourth_quadrant ab.1 ab.2) events_space).card = 21 / 36 * events_space.card :=
sorry

end prob_one_solution_prob_point_in_fourth_quadrant_l681_681070


namespace sin_difference_identity_l681_681217

variable {α β : ℝ}

def trig_conditions (α β : ℝ) : Prop :=
  (tan β = 2 * tan α) ∧ (cos α * sin β = 2 / 3)

theorem sin_difference_identity (h : trig_conditions α β) : sin (α - β) = -1 / 3 :=
by
  -- Import the conditions from the hypothesis
  cases h with h1 h2
  sorry

end sin_difference_identity_l681_681217


namespace line_equation_l681_681374

variable (x y : ℝ)

theorem line_equation (x1 y1 m : ℝ) (h : x1 = -2 ∧ y1 = 3 ∧ m = 2) :
    -2 * x + y = 1 := by
  sorry

end line_equation_l681_681374


namespace divide_F_nk_l681_681199

noncomputable def F (n k : ℕ) : ℕ :=
  ∑ r in Finset.range (n + 1), r ^ (2 * k - 1)

theorem divide_F_nk (n k : ℕ) (h₁ : 0 < n) (h₂ : 0 < k) : F n 1 ∣ F n k := 
  sorry

end divide_F_nk_l681_681199


namespace area_of_triangle_ABC_l681_681331

theorem area_of_triangle_ABC :
  ∀ (A B C M N : ℝ) 
    (h1 : M = 10 / 3)
    (h2 : N = 5)
    (h3 : ∀ x, A ^ 2 + x ^ 2 = B ^ 2 )
    (h4 : ∀ y, N ^ 2 + y ^ 2 = C ^ 2 ) 
    (h5 : AC = 15)
    (h6 : B = (5 * sqrt(3)) / 2) :
    area ∆ABC = (75 * sqrt(3)) / 4 :=
sorry

end area_of_triangle_ABC_l681_681331


namespace rosie_pies_l681_681341

-- Definition of known conditions
def apples_per_pie (apples_pies_ratio : ℕ × ℕ) : ℕ :=
  apples_pies_ratio.1 / apples_pies_ratio.2

def pies_from_apples (total_apples : ℕ) (apples_per_pie : ℕ) : ℕ :=
  total_apples / apples_per_pie

-- Theorem statement
theorem rosie_pies (apples_pies_ratio : ℕ × ℕ) (total_apples : ℕ) :
  apples_pies_ratio = (12, 3) →
  total_apples = 36 →
  pies_from_apples total_apples (apples_per_pie apples_pies_ratio) = 9 :=
by
  intros h_ratio h_apples
  rw [h_ratio, h_apples]
  sorry

end rosie_pies_l681_681341


namespace fourth_pentagon_has_31_dots_l681_681483

def numDots (n: ℕ) : ℕ :=
  if n = 1 then 1 else numDots (n - 1) + 5 * (n - 1)

theorem fourth_pentagon_has_31_dots :
  numDots 4 = 31 :=
by
  sorry

end fourth_pentagon_has_31_dots_l681_681483


namespace parabola_ellipse_intersection_distance_l681_681453

theorem parabola_ellipse_intersection_distance
  (f1 : ℝ × ℝ)
  (h_focus_shared : f1 = (3, 0) ∨ f1 = (-3, 0))
  (intersection_pts : set (ℝ × ℝ))
  (h_directrix : (∀ p ∈ intersection_pts, ∃ q ∈ intersection_pts, (p.1 = -(q.1) ∧ p.2 = q.2)))
  (h_ellipse : ∀ pt ∈ intersection_pts, (pt.1^2 / 25 + pt.2^2 / 16 = 1))
  (h_parabola : ∀ pt ∈ intersection_pts, pt.1 = pt.2^2 / 6 + 1.5) :
  ∃ b : ℝ, (∃ y1 y2 : ℝ, y1 = b ∧ y2 = -b ∧ ∀ pt ∈ intersection_pts, pt = (1.5 + (2 * b) / 4, b) ∨ pt = (1.5 + (2 * b) / 4, -b)) ∧ 
  ∀ p1 p2 ∈ intersection_pts, (p1 = (1.5 + (2 * b) / 4, b) ∧ p2 = (1.5 + (2 * b) / 4, -b)) → dist p1 p2 = 2 * b := sorry

end parabola_ellipse_intersection_distance_l681_681453


namespace evaluate_expression_l681_681837

theorem evaluate_expression (x y : ℤ) (hx : x = -2) (hy : y = 1) :
  ([(x + 2 * y) * (x - 2 * y) + 4 * (x - y)^2] / (-x) = 18) :=
by
  sorry

end evaluate_expression_l681_681837


namespace factorization_problem_l681_681186

theorem factorization_problem (x : ℝ) :
  (x^4 + x^2 - 4) * (x^4 + x^2 + 3) + 10 =
  (x^2 + x + 1) * (x^2 - x + 1) * (x^2 + 2) * (x + 1) * (x - 1) :=
sorry

end factorization_problem_l681_681186


namespace product_greater_than_sum_probability_l681_681889

theorem product_greater_than_sum_probability : 
  let s := {n | n ∈ Finset.range' 1 6} in 
  let count_pairs := (s.product s).filter (λ p : ℕ × ℕ, (p.1 - 1) * (p.2 - 1) > 2) in
  (count_pairs.card : ℚ) / (s.card * s.card) = 7 / 18 := 
by 
  sorry

end product_greater_than_sum_probability_l681_681889


namespace y_intercept_of_l_l681_681224

-- Define the point A
def A : ℝ × ℝ := (2, -2)

-- Define the circle C
def C (p : ℝ × ℝ) : Prop := (p.1)^2 + (p.2)^2 - 4 * p.2 = 0

-- Define the condition for the distance from points on C to line l being maximum
def max_distance_condition (l : ℝ → ℝ) : Prop :=
  ∀ (p : ℝ × ℝ), C p → 
   -- this condition to be replaced with proper math expression
   sorry 

-- state the theorem to prove the y-intercept is -3
theorem y_intercept_of_l (l : ℝ → ℝ) (h : l = λ x, (1/2) * x - 3) 
  (hA : l 2 = -2) (h_max : max_distance_condition l) : 
  l 0 = -3 :=
sorry

end y_intercept_of_l_l681_681224


namespace coloring_methods_count_l681_681016

theorem coloring_methods_count :
  (∑ n in finset.range 2, ((nat.choose 6 (n+2)) * 2 * (2^(n+1) - nat.choose (n+2) 2 * 2))) = 390 :=
by simp [finset.range, finset.sum]; sorry

end coloring_methods_count_l681_681016


namespace solve_trig_system_l681_681844

theorem solve_trig_system
  (k n : ℤ) :
  (∃ x y : ℝ,
    (2 * Real.sin x ^ 2 + 2 * Real.sqrt 2 * Real.sin x * Real.sin (2 * x) ^ 2 + Real.sin (2 * x) ^ 2 = 0 ∧
     Real.cos x = Real.cos y) ∧
    ((x = 2 * Real.pi * k ∧ y = 2 * Real.pi * n) ∨
     (x = Real.pi + 2 * Real.pi * k ∧ y = Real.pi + 2 * Real.pi * n) ∨
     (x = -Real.pi / 4 + 2 * Real.pi * k ∧ (y = Real.pi / 4 + 2 * Real.pi * n ∨ y = -Real.pi / 4 + 2 * Real.pi * n)) ∨
     (x = -3 * Real.pi / 4 + 2 * Real.pi * k ∧ (y = 3 * Real.pi / 4 + 2 * Real.pi * n ∨ y = -3 * Real.pi / 4 + 2 * Real.pi * n)))) :=
sorry

end solve_trig_system_l681_681844


namespace simplify_sqrt_expression_l681_681222

noncomputable def simplify_expression (x y : ℝ) (h : x * y < 0) : ℝ :=
  x * real.sqrt (-y / x^2)

theorem simplify_sqrt_expression (x y : ℝ) (h : x * y < 0) (hy : y < 0) :
  simplify_expression x y h = real.sqrt (- y) := by
  sorry

end simplify_sqrt_expression_l681_681222


namespace statement_C_is_incorrect_l681_681259

noncomputable def g (x : ℝ) : ℝ := (2 * x + 3) / (x - 2)

theorem statement_C_is_incorrect : g (-2) ≠ 0 :=
by
  sorry

end statement_C_is_incorrect_l681_681259


namespace negation_example_l681_681864

theorem negation_example :
  (¬ (∀ x : ℝ, abs (x - 2) + abs (x - 4) > 3)) ↔ (∃ x : ℝ, abs (x - 2) + abs (x - 4) ≤ 3) :=
by
  sorry

end negation_example_l681_681864


namespace medians_sum_geq_4_circumradius_l681_681334

theorem medians_sum_geq_4_circumradius (ABC : Triangle) (not_obtuse_triangle : ¬ABC.∠A > π / 2 ∧ ¬ABC.∠B > π / 2 ∧ ¬ABC.∠C > π / 2) :
  ABC.median_to_side BC + ABC.median_to_side CA + ABC.median_to_side AB ≥ 4 * ABC.circumradius :=
sorry

end medians_sum_geq_4_circumradius_l681_681334


namespace parallel_PR_BC_l681_681729

namespace Geometry

open EuclideanGeometry

variables {A B C P R Q M : Point} 

def is_midpoint (M B C : Point) : Prop := dist B M = dist C M ∧ M ≠ B ∧ M ≠ C
def is_parallel (l1 l2 : Line) : Prop := ∃ (v1 v2 : Vector), v1 ≠ 0 ∧ v1 ∥ l1 ∧ v2 ≠ 0 ∧ v2 ∥ l2 ∧ v1 ∥ v2

theorem parallel_PR_BC
  (h1 : triangle A B C)
  (h2 : is_midpoint M B C)
  (h3 : P ∈ line_through A B)
  (h4 : R ∈ line_through A C)
  (h5 : intersects (line_through P R) (line_through A M) Q)
  (h6 : is_midpoint Q P R) : is_parallel (line_through P R) (line_through B C) :=
by {
  sorry,  -- proof not required as per instructions
}

end Geometry

end parallel_PR_BC_l681_681729


namespace least_number_to_add_l681_681027

theorem least_number_to_add (a b n x : ℕ) (h_a : a = 27) (h_b : b = 31) (h_n : n = 1056) (h_x : x = 618) : 
  ∃ k : ℕ, n + x = k * (Nat.lcm a b) :=
by
  rw [h_a, h_b, h_n, h_x]
  use Nat.lcm
  sorry

end least_number_to_add_l681_681027


namespace solve_for_m_l681_681797

def z1 := Complex.mk 3 2
def z2 (m : ℝ) := Complex.mk 1 m

theorem solve_for_m (m : ℝ) (h : (z1 * z2 m).re = 0) : m = 3 / 2 :=
by
  sorry

end solve_for_m_l681_681797


namespace pyramid_height_l681_681439

-- Define the edge length of the cube
def cube_edge_length : ℝ := 5

-- Define the base edge length of the pyramid
def pyramid_base_edge_length : ℝ := 10

-- Define the volume of a cube with edge length 5 units
def cube_volume : ℝ := cube_edge_length ^ 3

-- Define the volume of a pyramid with a square base
def pyramid_volume (h : ℝ) : ℝ := (1 / 3) * (pyramid_base_edge_length ^ 2) * h

-- Add a theorem to prove the height of the pyramid
theorem pyramid_height : ∃ h : ℝ, cube_volume = pyramid_volume h ∧ h = 3.75 :=
by
  -- Given conditions and correct answer lead to the proof of the height being 3.75
  sorry

end pyramid_height_l681_681439


namespace discount_percentage_is_20_l681_681001

def original_price : ℝ := 975
def paid_price : ℝ := 780

def discount_percentage (original_price paid_price : ℝ) : ℝ :=
  ((original_price - paid_price) / original_price) * 100

theorem discount_percentage_is_20 :
  discount_percentage original_price paid_price = 20 := 
by
  sorry

end discount_percentage_is_20_l681_681001


namespace distance_from_point_to_line_P1_l681_681366

theorem distance_from_point_to_line_P1 (x y : ℝ) (h1 : x = 1) (h2 : y = 0) : 
  let P := (x, y)
  let A := 1
  let B := -1
  let C := -3
  let d := abs (A * x + B * y + C) / sqrt (A ^ 2 + B ^ 2)
  in d = sqrt 2 := 
by 
  sorry

end distance_from_point_to_line_P1_l681_681366


namespace charley_pencils_lost_l681_681154

theorem charley_pencils_lost :
  ∃ x : ℕ, (30 - x - (1/3 : ℝ) * (30 - x) = 16) ∧ x = 6 :=
by
  -- Since x must be an integer and the equations naturally produce whole numbers,
  -- we work within the context of natural numbers, then cast to real as needed.
  use 6
  -- Express the main condition in terms of x
  have h: (30 - 6 - (1/3 : ℝ) * (30 - 6) = 16) := by sorry
  exact ⟨h, rfl⟩

end charley_pencils_lost_l681_681154


namespace james_age_when_john_turned_35_l681_681770

theorem james_age_when_john_turned_35 :
  ∀ (J : ℕ) (Tim : ℕ) (John : ℕ),
  (Tim = 5) →
  (Tim + 5 = 2 * John) →
  (Tim = 79) →
  (John = 35) →
  (J = John) →
  J = 35 :=
by
  intros J Tim John h1 h2 h3 h4 h5
  rw [h4] at h5
  exact h5

end james_age_when_john_turned_35_l681_681770


namespace greatest_div_by_seven_base_eight_l681_681895

theorem greatest_div_by_seven_base_eight : ∃ n : ℕ, 
  (n < 512) ∧ (Divisibility.divides 7 n) ∧ 
  (∀ m : ℕ, (m < 512) → (Divisibility.divides 7 m) → m ≤ n) ∧ 
  nat.to_digits 8 n = [7, 7, 4] := 
sorry

end greatest_div_by_seven_base_eight_l681_681895


namespace michael_truck_meeting_l681_681325

noncomputable def michael_speed := 4 -- Michael's speed in feet per second
noncomputable def truck_speed := 12 -- Truck's speed in feet per second
noncomputable def stop_time := 40 -- Truck's stop time at each pail in seconds
noncomputable def pail_distance := 250 -- Distance between successive pails

def michael_pos (t : ℝ) : ℝ := michael_speed * t -- Position function for Michael
def truck_cycle := (pail_distance / truck_speed) + stop_time -- Time for one full truck cycle (motion + stop)
def truck_pos (t : ℝ) : ℝ :=
  let c := t / truck_cycle in
  let k := t % truck_cycle in
  if k < pail_distance / truck_speed then
    truck_speed * k + c * pail_distance
  else 
    c * pail_distance

theorem michael_truck_meeting :
  ∃ n : ℕ, ∃ t : ℝ, t = n * truck_cycle ∧ (michael_pos t) = (truck_pos t) :=
sorry

end michael_truck_meeting_l681_681325


namespace bernardo_larger_than_silvia_l681_681489

def bernardo_set : finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def silvia_set : finset ℕ := {1, 2, 3, 4, 5, 6, 7}

def number_of_ways_to_choose_3_from_bernardo : ℚ :=
  ((bernardo_set.card).choose 3)

def number_of_ways_to_choose_3_from_silvia : ℚ :=
  ((silvia_set.card).choose 3)

def probability_of_bernardo_choosing_9 : ℚ :=
  ((bernardo_set.erase 9).card.choose 2) / number_of_ways_to_choose_3_from_bernardo

def probability_of_large_number_without_9 : ℚ :=
  17 / 35

theorem bernardo_larger_than_silvia :
  (probability_of_bernardo_choosing_9
    + (1 - probability_of_bernardo_choosing_9) * probability_of_large_number_without_9) 
  = 112 / 175 :=
begin
  sorry
end

end bernardo_larger_than_silvia_l681_681489


namespace find_a_range_l681_681655

noncomputable def f (x a : ℝ) := 2 ^ (x * (x - a))

theorem find_a_range (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → deriv (λ x, f x a) x < 0) → 2 ≤ a :=
by sorry

end find_a_range_l681_681655


namespace tan_alpha_plus_pi_over_4_sin_2alpha_fraction_l681_681603

-- Question 1 (Proving tan(alpha + pi/4) = -3 given tan(alpha) = 2)
theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 :=
sorry

-- Question 2 (Proving the given fraction equals 1 given tan(alpha) = 2)
theorem sin_2alpha_fraction (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (2 * α) / 
   (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2 * α) - 1)) = 1 :=
sorry

end tan_alpha_plus_pi_over_4_sin_2alpha_fraction_l681_681603


namespace common_ratio_is_neg2_l681_681617

-- Definitions and conditions
def sum_of_first_n_terms (a_n : ℕ → ℤ) (n : ℕ) : ℤ := ∑ i in finset.range n, a_n i
def is_arithmetic_sequence {α : Type*} [add_comm_group α] [module ℤ α] (S : ℕ → α) : Prop :=
  2 * S 4 = S 5 + S 6

-- Definition of the arithmetic sequence and its common ratio
variable {a_n : ℕ → ℤ}
variable [arith_seq : ∀ n : ℕ, a_n n = n * (-2)]

-- Statement to prove
theorem common_ratio_is_neg2 (h_seq : is_arithmetic_sequence (sum_of_first_n_terms a_n)) :
  ∃ q : ℤ, q = -2 :=
by {
  use -2,
  sorry
}

end common_ratio_is_neg2_l681_681617


namespace number_of_goats_l681_681059

theorem number_of_goats (C G : ℕ) 
  (h1 : C = 2) 
  (h2 : ∀ G : ℕ, 460 * C + 60 * G = 1400) 
  (h3 : 460 = 460) 
  (h4 : 60 = 60) : 
  G = 8 :=
by
  sorry

end number_of_goats_l681_681059


namespace intersection_point_count_l681_681376

noncomputable def log4 (x : ℝ) : ℝ := real.log x / real.log 4
noncomputable def log_x_4 (x : ℝ) : ℝ := real.log 4 / real.log x
noncomputable def exp4 (x : ℝ) : ℝ := 4^x
noncomputable def log_x_inv4 (x : ℝ) : ℝ := real.log (1 / 4) / real.log x

theorem intersection_point_count : 
  (finset.image (λ x : ℝ, (x, log4 x)) {4, 1/4} ∪ 
   finset.image (λ x : ℝ, (x, log_x_4 x)) {4, 1/4} ∪ 
   finset.image (λ x : ℝ, (x, exp4 x)) {4, 1/4} ∪
   finset.image (λ x : ℝ, (x, log_x_inv4 x)) {4, 1/4}).card = 2 :=
by sorry

end intersection_point_count_l681_681376


namespace max_real_roots_of_polynomial_l681_681583

theorem max_real_roots_of_polynomial (n : ℕ) (hn : 0 < n) :
  ∃ k, k = 1 ∧ ∀ x : ℝ, (x^n - x^(n-1) + x^(n-2) - ... + (-1)^(n-1) * x + (-1)^n = 0) → x = 1 ∨ (n % 2 = 0 ∧ x = -1) :=
sorry

end max_real_roots_of_polynomial_l681_681583


namespace number_of_customers_l681_681980

theorem number_of_customers 
    (boxes_opened : ℕ) 
    (samples_per_box : ℕ) 
    (samples_left_over : ℕ) 
    (samples_limit_per_person : ℕ)
    (h1 : boxes_opened = 12)
    (h2 : samples_per_box = 20)
    (h3 : samples_left_over = 5)
    (h4 : samples_limit_per_person = 1) : 
    ∃ customers : ℕ, customers = (boxes_opened * samples_per_box) - samples_left_over ∧ customers = 235 :=
by {
  sorry
}

end number_of_customers_l681_681980


namespace algebraic_expression_value_l681_681604

theorem algebraic_expression_value (x y : ℝ) (h : x = 2 * y + 3) : 4 * x - 8 * y + 9 = 21 := by
  sorry

end algebraic_expression_value_l681_681604


namespace fourth_term_is_integer_l681_681320

def starts_with_seven (a : ℕ) : Prop := a = 7

def next_term (a : ℕ) (coin : bool) : ℕ :=
  if coin then 3 * a + 2
  else if even a then a / 2 - 2
  else a + 3

def sequence (a : ℕ) (coin : bool) (n : ℕ) : ℕ :=
  match n with
  | 0 => a
  | n + 1 => next_term (sequence a coin n) coin

theorem fourth_term_is_integer (seq : ℕ → ℕ) (coin1 coin2 coin3 : bool):
  starts_with_seven 7 →
  ∀ n m, sequence 7 coin1 4 = n → (sequence 7 coin2 4 = m) →
  ∃ n : ℕ, n ∈ [seq n] :=
by sorry

end fourth_term_is_integer_l681_681320


namespace dot_product_AB_AC_dot_product_AB_BC_l681_681277

-- The definition of equilateral triangle with side length 6
structure EquilateralTriangle (A B C : Type*) :=
  (side_len : ℝ)
  (angle_ABC : ℝ)
  (angle_BCA : ℝ)
  (angle_CAB : ℝ)
  (AB_len : ℝ)
  (AC_len : ℝ)
  (BC_len : ℝ)
  (AB_eq_AC : AB_len = AC_len)
  (AB_eq_BC : AB_len = BC_len)
  (cos_ABC : ℝ)
  (cos_BCA : ℝ)
  (cos_CAB : ℝ)

-- Given an equilateral triangle with side length 6 where the angles are defined,
-- we can define the specific triangle
noncomputable def triangleABC (A B C : Type*) : EquilateralTriangle A B C :=
{ side_len := 6,
  angle_ABC := 120,
  angle_BCA := 60,
  angle_CAB := 60,
  AB_len := 6,
  AC_len := 6,
  BC_len := 6,
  AB_eq_AC := rfl,
  AB_eq_BC := rfl,
  cos_ABC := -0.5,
  cos_BCA := 0.5,
  cos_CAB := 0.5 }

-- Prove the dot product of vectors AB and AC
theorem dot_product_AB_AC (A B C : Type*) 
  (T : EquilateralTriangle A B C) : 
  (T.AB_len * T.AC_len * T.cos_BCA) = 18 :=
by sorry

-- Prove the dot product of vectors AB and BC
theorem dot_product_AB_BC (A B C : Type*) 
  (T : EquilateralTriangle A B C) : 
  (T.AB_len * T.BC_len * T.cos_ABC) = -18 :=
by sorry

end dot_product_AB_AC_dot_product_AB_BC_l681_681277


namespace angle_between_unit_vectors_l681_681678

variables {m n : EuclideanSpace ℝ (Fin 3)}
variable (unit_m : ∥m∥ = 1)
variable (unit_n : ∥n∥ = 1)
variable (cond : ∥m - (2 : ℝ) • n∥ = Real.sqrt 7)

theorem angle_between_unit_vectors (unit_m : ∥m∥ = 1) (unit_n : ∥n∥ = 1) (cond : ∥m - (2 : ℝ) • n∥ = Real.sqrt 7) :
  ∃ θ : ℝ, θ = (2 * Real.pi / 3) ∧
  ∀ u : EuclideanSpace ℝ (Fin 3), ∀ v : EuclideanSpace ℝ (Fin 3), (∥u∥ = 1 ∧ ∥v∥ = 1 ∧ ∥u - (2 : ℝ) • v∥ = Real.sqrt 7) → 
  (u ⬝ v = -1 / 2) :=
sorry

end angle_between_unit_vectors_l681_681678


namespace sqrt_eq_implies_ge_two_l681_681718

theorem sqrt_eq_implies_ge_two (a : ℝ) : sqrt((a - 2)^2) = a - 2 → a ≥ 2 :=
by
  intros h
  sorry

end sqrt_eq_implies_ge_two_l681_681718


namespace cricket_average_increase_l681_681436

theorem cricket_average_increase :
  ∀ (x : ℝ), (11 * (33 + x) = 407) → (x = 4) :=
  by 
  intros x hx
  sorry

end cricket_average_increase_l681_681436


namespace segment_AB_length_l681_681751

-- Defining the conditions
def area_ratio (AB CD : ℝ) : Prop := AB / CD = 5 / 2
def length_sum (AB CD : ℝ) : Prop := AB + CD = 280

-- The theorem stating the problem
theorem segment_AB_length (AB CD : ℝ) (h₁ : area_ratio AB CD) (h₂ : length_sum AB CD) : AB = 200 :=
by {
  -- Proof step would be inserted here, but it is omitted as per instructions
  sorry
}

end segment_AB_length_l681_681751


namespace length_of_AB_l681_681753

noncomputable def height (t : Type) [LinearOrderedField t] := sorry
def area (a b h : ℝ) : ℝ := 0.5 * a * h

theorem length_of_AB (AB CD : ℝ) (h : ℝ) (ratio : area AB h / area CD h = 5 / 2) (sum : AB + CD = 280) :
  AB = 200 :=
begin
  sorry
end

end length_of_AB_l681_681753


namespace quadrilateral_ABCD_area_l681_681281

def quadrilateral_area (A B C D : Point) :=
  let α : ℝ := 135 -- ∠A
  let β : ℝ := 90  -- ∠B
  let δ : ℝ := 90  -- ∠D
  let BC : ℝ := 2
  let AD : ℝ := 2
  area_of_quadrilateral A B C D = 4

theorem quadrilateral_ABCD_area :
  ∀ (A B C D : EuclideanGeometry.Point),
  (∠ A + B + ∠ D + ∠ C = 360) ∧ (∠ A = 135) ∧ (∠ B = 90) ∧ (∠ D = 90) ∧
  (distance B C = 2) ∧ (distance A D = 2) →
  area_of_quadrilateral A B C D = 4 :=
begin
  sorry
end

end quadrilateral_ABCD_area_l681_681281


namespace decreasing_interval_range_l681_681649

theorem decreasing_interval_range (a : ℝ) :
  (∀ x y ∈ Ioo 0 1, x < y → 2^(x * (x-a)) > 2^(y * (y-a))) ↔ a ≥ 2 :=
by
  sorry

end decreasing_interval_range_l681_681649


namespace monotonic_decreasing_condition_l681_681650

theorem monotonic_decreasing_condition {f : ℝ → ℝ} (a : ℝ) :
  (∀ x ∈ Ioo (0:ℝ) 1, (2:ℝ) ^ (x * (x - a)) < (2:ℝ) ^ ((1:ℝ) * (1 - a))) → a ≥ 2 :=
begin
  sorry
end

end monotonic_decreasing_condition_l681_681650


namespace complement_A_eq_l681_681239

def R : Type := ℝ

def A : set R := {x | x ≥ 3} ∪ {x | x < -1}

def complement (A : set R) : set R := {x | x ∉ A}

theorem complement_A_eq : complement A = {x | -1 ≤ x ∧ x < 3} :=
  sorry

end complement_A_eq_l681_681239


namespace total_weight_correct_l681_681319

def Marco_strawberry_weight : ℕ := 15
def Dad_strawberry_weight : ℕ := 22
def total_strawberry_weight : ℕ := Marco_strawberry_weight + Dad_strawberry_weight

theorem total_weight_correct :
  total_strawberry_weight = 37 :=
by
  sorry

end total_weight_correct_l681_681319


namespace sum_G_values_l681_681198

def G (n : ℕ) : ℕ := 2 * n

theorem sum_G_values : (Finset.sum (Finset.range 999) (λ n, G (n + 2))) = 999000 :=
by {
  -- Here we should provide the proof that sum_G_values is indeed 999000 given the conditions
  sorry
}

end sum_G_values_l681_681198


namespace grill_run_time_l681_681432

-- Define the conditions 1, 2, and 3
def burns_rate : ℕ := 15 -- coals burned every 20 minutes
def burns_time : ℕ := 20 -- minutes to burn some coals
def coals_per_bag : ℕ := 60 -- coals per bag
def num_bags : ℕ := 3 -- number of bags

-- The main theorem to prove the time taken to burn all the coals
theorem grill_run_time : 
  let total_coals := num_bags * coals_per_bag in
  let burn_time_per_coals := total_coals * burns_time / burns_rate in
  burn_time_per_coals / 60 = 4 := 
by
  sorry

end grill_run_time_l681_681432


namespace segment_AC_length_l681_681291

-- Define segments AB and BC
def AB : ℝ := 4
def BC : ℝ := 3

-- Define segment AC in terms of the conditions given
def AC_case1 : ℝ := AB - BC
def AC_case2 : ℝ := AB + BC

-- The proof problem statement
theorem segment_AC_length : AC_case1 = 1 ∨ AC_case2 = 7 := by
  sorry

end segment_AC_length_l681_681291


namespace other_acute_angle_l681_681971

theorem other_acute_angle (acute_angle : ℝ) (h1 : acute_angle = 25) : 
  ∃ other_acute_angle : ℝ, other_acute_angle = 65 :=
by
  use 90 - acute_angle
  rw h1
  norm_num
  sorry

end other_acute_angle_l681_681971


namespace distance_walked_together_l681_681338

-- Definitions based on conditions
def distance_raj_walked (H : ℕ) : ℕ := 4 * H - 10

axiom condition_1 (H : ℕ) : ∀ (R : ℕ), R = distance_raj_walked H
axiom condition_2 : 18 = distance_raj_walked 7

-- Statement of the proof problem
theorem distance_walked_together (H : ℕ) (R : ℕ) 
 (h1: condition_1 H) (h2: condition_2) : H = 7 :=
sorry

end distance_walked_together_l681_681338


namespace binom_18_6_eq_4765_l681_681530

def binom (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_18_6_eq_4765 : binom 18 6 = 4765 := by
  sorry

end binom_18_6_eq_4765_l681_681530


namespace shortest_distance_between_circles_l681_681028

-- Define the first circle equation
def circle1 := ∀ (x y : ℝ), x^2 - 6*x + y^2 + 2*y - 11 = 0

-- Define the second circle equation
def circle2 := ∀ (x y : ℝ), x^2 + 10*x + y^2 - 8*y + 25 = 0

-- Prove the shortest distance between the given circles is as stated
theorem shortest_distance_between_circles 
  (hx1 : ∃ x y : ℝ, circle1 x y) 
  (hx2 : ∃ x y : ℝ, circle2 x y) : 
  shortest_distance circle1 circle2 = (sqrt 89) - (sqrt 21) - 4 :=
sorry

end shortest_distance_between_circles_l681_681028


namespace sum_arithmetic_seq_l681_681135

theorem sum_arithmetic_seq (a d n : ℕ) :
  a = 2 → d = 2 → a + (n - 1) * d = 20 → (n / 2) * (a + (a + (n - 1) * d)) = 110 :=
by sorry

end sum_arithmetic_seq_l681_681135


namespace binom_18_6_eq_13260_l681_681519

/-- The binomial coefficient formula. -/
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The specific proof problem: compute binom(18, 6) and show that it equals 13260. -/
theorem binom_18_6_eq_13260 : binom 18 6 = 13260 :=
by
  sorry

end binom_18_6_eq_13260_l681_681519


namespace cylinder_radius_maximized_volume_l681_681880

noncomputable def cylinder_radius_max_vol (S : ℝ) : ℝ :=
sorry

theorem cylinder_radius_maximized_volume (S : ℝ) :
  (∃ (r h : ℝ), 2 * Real.pi * r * (r + h) = S ∧ h = S / (2 * Real.pi * r) - r) →
  cylinder_radius_max_vol(S) = r :=
sorry

end cylinder_radius_maximized_volume_l681_681880


namespace num_customers_who_tried_sample_l681_681977

theorem num_customers_who_tried_sample :
  ∀ (samples_per_box boxes_opened samples_left : ℕ), 
  samples_per_box = 20 →
  boxes_opened = 12 →
  samples_left = 5 →
  let total_samples := samples_per_box * boxes_opened in
  let samples_used := total_samples - samples_left in
  samples_used = 235 :=
by 
  intros samples_per_box boxes_opened samples_left h_samples_per_box h_boxes_opened h_samples_left total_samples samples_used
  simp [h_samples_per_box, h_boxes_opened, h_samples_left]
  sorry

end num_customers_who_tried_sample_l681_681977


namespace arthur_walks_distance_l681_681099

theorem arthur_walks_distance :
  ∀ (blocks_west blocks_south : ℕ) (block_length : ℝ), 
  blocks_west = 8 → 
  blocks_south = 10 → 
  block_length = 0.25 → 
  (blocks_west + blocks_south) * block_length = 4.5 := 
by {
  intros blocks_west blocks_south block_length,
  assume h_west h_south h_length,
  rw [h_west, h_south, h_length],
  norm_num,
}

end arthur_walks_distance_l681_681099


namespace find_x_l681_681619

def f (x : ℝ) : ℝ := 2 * x - 3 -- Definition of the function f

def c : ℝ := 11 -- Definition of the constant c

theorem find_x : 
  ∃ x : ℝ, 2 * f x - c = f (x - 2) ↔ x = 5 :=
by 
  sorry

end find_x_l681_681619


namespace find_value_of_p_l681_681668

-- Definition of the parabola and ellipse
def parabola (p : ℝ) : Set (ℝ × ℝ) := {xy | xy.1 ^ 2 = 2 * p * xy.2}
def ellipse : Set (ℝ × ℝ) := {xy | xy.1 ^ 2 / 6 + xy.2 ^ 2 / 4 = 1}

-- Hypotheses
variables (p : ℝ) (h_pos : p > 0)

-- Latus rectum tangent to the ellipse
theorem find_value_of_p (h_tangent : ∃ (x y : ℝ),
  (parabola p (x, y) ∧ ellipse (x, y) ∧ y = -p / 2)) : p = 4 := sorry

end find_value_of_p_l681_681668


namespace chocolate_bars_gigantic_box_l681_681073

def large_boxes : ℕ := 50
def medium_boxes : ℕ := 25
def small_boxes : ℕ := 10
def chocolate_bars_per_small_box : ℕ := 45

theorem chocolate_bars_gigantic_box : 
  large_boxes * medium_boxes * small_boxes * chocolate_bars_per_small_box = 562500 :=
by
  sorry

end chocolate_bars_gigantic_box_l681_681073


namespace area_triangle_ABC_l681_681750

theorem area_triangle_ABC (x y : ℝ) (h : x * y ≠ 0) (hAOB : 1 / 2 * |x * y| = 4) : 
  1 / 2 * |(x * (-2 * y) + x * (2 * y) + (-x) * (2 * y))| = 8 :=
by
  sorry

end area_triangle_ABC_l681_681750


namespace smallest_positive_period_of_f_is_one_l681_681373

def f (x : ℝ) : ℝ := x - floor x

theorem smallest_positive_period_of_f_is_one : 
  ∀ r > 0, (∀ x : ℝ, f(x + r) = f x) → r ≥ 1 := by 
  sorry

end smallest_positive_period_of_f_is_one_l681_681373


namespace probability_different_colors_l681_681593

theorem probability_different_colors :
  let hat_colors := ["red", "blue"];
      cloak_colors := ["red", "blue", "green"];
      total_combinations := 6;
      non_matching_combinations := 4;
  (non_matching_combinations / total_combinations = 2 / 3) :=
by {
  let hat_colors := ["red", "blue"];
  let cloak_colors := ["red", "blue", "green"];
  let total_combinations := 6;
  let non_matching_combinations := 4;
  have h : (non_matching_combinations : ℚ) / total_combinations = 2 / 3 := 
    by norm_num,
  exact h,
  sorry
}

end probability_different_colors_l681_681593


namespace cobbler_hourly_rate_l681_681996

theorem cobbler_hourly_rate (x : ℝ)
  (h_mold : 250)
  (h_hours : 8)
  (h_discount : 0.80)
  (h_total_paid : 730) :
  250 + 0.80 * 8 * x = 730 → x = 75 :=
by
  sorry

end cobbler_hourly_rate_l681_681996


namespace greatest_3_digit_base_8_divisible_by_7_l681_681913

open Nat

def is_3_digit_base_8 (n : ℕ) : Prop := n < 8^3

def is_divisible_by_7 (n : ℕ) : Prop := 7 ∣ n

theorem greatest_3_digit_base_8_divisible_by_7 :
  ∃ x : ℕ, is_3_digit_base_8 x ∧ is_divisible_by_7 x ∧ x = 7 * (8 * (8 * 7 + 7) + 7) :=
by
  sorry

end greatest_3_digit_base_8_divisible_by_7_l681_681913


namespace james_needs_50_hours_l681_681766

-- Define the given conditions
def meat_cost_per_pound : ℝ := 5
def meat_pounds_wasted : ℝ := 20
def fruits_veg_cost_per_pound : ℝ := 4
def fruits_veg_pounds_wasted : ℝ := 15
def bread_cost_per_pound : ℝ := 1.5
def bread_pounds_wasted : ℝ := 60
def janitorial_hourly_rate : ℝ := 10
def janitorial_hours_worked : ℝ := 10
def james_hourly_rate : ℝ := 8

-- Calculate the total costs separately
def total_meat_cost : ℝ := meat_cost_per_pound * meat_pounds_wasted
def total_fruits_veg_cost : ℝ := fruits_veg_cost_per_pound * fruits_veg_pounds_wasted
def total_bread_cost : ℝ := bread_cost_per_pound * bread_pounds_wasted
def janitorial_time_and_a_half_rate : ℝ := janitorial_hourly_rate * 1.5
def total_janitorial_cost : ℝ := janitorial_time_and_a_half_rate * janitorial_hours_worked

-- Calculate the total cost James has to pay
def total_cost : ℝ := total_meat_cost + total_fruits_veg_cost + total_bread_cost + total_janitorial_cost

-- Calculate the number of hours James needs to work
def james_work_hours : ℝ := total_cost / james_hourly_rate

-- The theorem to be proved
theorem james_needs_50_hours : james_work_hours = 50 :=
by
  sorry

end james_needs_50_hours_l681_681766


namespace johanns_path_probability_l681_681451

-- Define the necessary structures and conditions
def interior_lattice_points (n : ℕ) : list (ℕ × ℕ) := 
  (list.range n).bind (λ i, (list.range n).map (prod.mk i))

def is_even (n : ℕ) : Prop := n % 2 = 0

def contains_even_number_of_lattice_points (a b : ℕ) : Prop := 
  is_even (Nat.gcd a (2004 - b))

def probability_even_lattice_points (total : ℕ) (even : ℕ) : ℚ :=
  even / total

noncomputable def probability_johanns_path_even : ℚ :=
  let total_points := 9801
  let even_points := 2401
  probability_even_lattice_points total_points even_points

-- The main theorem to prove
theorem johanns_path_probability :
  probability_johanns_path_even = 3 / 4 :=
by sorry

end johanns_path_probability_l681_681451


namespace decimal_to_binary_41_l681_681166

theorem decimal_to_binary_41 : nat.binary_repr 41 = "101001" :=
by sorry

end decimal_to_binary_41_l681_681166


namespace james_hours_to_work_l681_681768

theorem james_hours_to_work :
  let meat_cost := 20 * 5
  let fruits_vegetables_cost := 15 * 4
  let bread_cost := 60 * 1.5
  let janitorial_cost := 10 * (10 * 1.5)
  let total_cost := meat_cost + fruits_vegetables_cost + bread_cost + janitorial_cost
  let hourly_wage := 8
  let hours_to_work := total_cost / hourly_wage
  hours_to_work = 50 :=
by 
  sorry

end james_hours_to_work_l681_681768


namespace teammates_scored_36_points_l681_681734

noncomputable def calculate_teammates_points : ℕ :=
  let lizzie_score := 4
  let nathalie_score := 2 * lizzie_score + 3
  let aimee_score := 2 * (lizzie_score + nathalie_score) + 1
  let julia_score := (nathalie_score / 2) - 2
  let ellen_score := (Real.sqrt aimee_score).toNat * 3
  let total_individual_scores := lizzie_score + nathalie_score + aimee_score + julia_score + ellen_score
  let team_total_score := 100
  team_total_score - total_individual_scores

-- Theorem stating the points scored by teammates are 36
theorem teammates_scored_36_points : calculate_teammates_points = 36 := by
  sorry

end teammates_scored_36_points_l681_681734


namespace length_of_each_side_is_25_nails_l681_681975

-- Definitions based on the conditions
def nails_per_side := 25
def total_nails := 96

-- The theorem stating the equivalent mathematical problem
theorem length_of_each_side_is_25_nails
  (n : ℕ) (h1 : n = nails_per_side * 4 - 4)
  (h2 : total_nails = 96):
  n = nails_per_side :=
by
  sorry

end length_of_each_side_is_25_nails_l681_681975


namespace min_abs_sum_is_12_l681_681029

theorem min_abs_sum_is_12 (x : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, y = min (lvert x + 3\rvert + lvert x + 4\rvert + lvert x + 6\rvert + lvert x + 8\rvert) y :=  12
:= 
sorry

end min_abs_sum_is_12_l681_681029


namespace count_two_digit_integers_sum_seven_l681_681596

theorem count_two_digit_integers_sum_seven : 
  ∃ n : ℕ, (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a + b = 7 → n = 7) := 
by
  sorry

end count_two_digit_integers_sum_seven_l681_681596


namespace frustum_lateral_surface_area_l681_681275

theorem frustum_lateral_surface_area (a SH HH1 : ℝ) (a_pos : a > 0) (SH_pos : SH > 0) (HH1_pos : HH1 > 0) (H1_base: a = 6) (H2_height: SH = 4) (H3_plane: HH1 = 1) :
  let SH1 := SH - HH1 in
  let H1E1 := (a / 2 * (SH1 / SH)) in
  let A1D1 := 2 * H1E1 in
  let SE1 := real.sqrt ((SH1)^2 + H1E1^2) in
  let SE := real.sqrt (SH^2 + (a / 2)^2) in
  let p2 := 2 * a in
  let p1 := A1D1 in
  let S_lateral1 := p1 * SE1 in
  let S_lateral2 := p2 * SE in
  S_lateral2 - S_lateral1 = 26.25 :=
  sorry

end frustum_lateral_surface_area_l681_681275


namespace percentage_is_60_l681_681066

-- Definitions based on the conditions
def fraction_value (x : ℕ) : ℕ := x / 3
def percentage_less_value (x p : ℕ) : ℕ := x - (p * x) / 100

-- Lean statement based on the mathematically equivalent proof problem
theorem percentage_is_60 : ∀ (x p : ℕ), x = 180 → fraction_value x = 60 → percentage_less_value 60 p = 24 → p = 60 :=
by
  intros x p H1 H2 H3
  -- Proof is not required, so we use sorry
  sorry

end percentage_is_60_l681_681066


namespace moles_of_Silver_Hydroxide_l681_681193

def Silver_Nitrate := Type
def Sodium_Hydroxide := Type
def Silver_Hydroxide := Type
def Sodium_Nitrate := Type

def n_AgNO3 (n : ℕ) := n
def n_NaOH (n : ℕ) := n
def n_AgOH (n : ℕ) := n
def n_NaNO3 (n : ℕ) := n

axiom balanced_reaction :
  ∀ (n_AgNO3 n_NaOH : ℕ), n_AgNO3 = n_NaOH → (∃ (n_AgOH n_NaNO3 : ℕ), n_AgOH = n_AgNO3 ∧ n_NaNO3 = n_NaOH)

theorem moles_of_Silver_Hydroxide:
  n_AgNO3 2 = n_NaOH 2 → n_AgOH 2 = 2 :=
by
  intro h
  have h_reaction := balanced_reaction 2 2 h
  cases h_reaction with n_AgOH' h1
  cases h1 with h_AgOH h_NaNO3
  rw h_AgOH
  exact h_AgOH
  sorry

end moles_of_Silver_Hydroxide_l681_681193


namespace scale_drawing_to_feet_l681_681461

theorem scale_drawing_to_feet (len_cm : ℝ) (cm_to_meters_ratio : ℝ) (meters_to_feet_ratio : ℝ) :
  len_cm = 6.5 →
  cm_to_meters_ratio = 250 →
  meters_to_feet_ratio = 3.281 →
  len_cm * cm_to_meters_ratio * meters_to_feet_ratio = 5332.125 :=
by
  intros h_len h_cm_to_meters_ratio h_meters_to_feet_ratio
  rw [h_len, h_cm_to_meters_ratio, h_meters_to_feet_ratio]
  linarith

end scale_drawing_to_feet_l681_681461


namespace greatest_3digit_base8_divisible_by_7_l681_681900

def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : ℕ), n = 0b777 ∧ (base8_to_base10 0b777) % 7 = 0 ∧ ∀ m < 0o777, m % 7 = 0 → base8_to_base10 m < base8_to_base10 0b777 :=
by
  sorry

end greatest_3digit_base8_divisible_by_7_l681_681900


namespace gcd_conditions_equivalent_l681_681846

theorem gcd_conditions_equivalent (a b c d : ℤ)
  (h_gcd : Int.gcd (Int.gcd a b) (Int.gcd c d) = 1) :
  (∀ p : ℕ, Prime p → p ∣ (a * d - b * c) → p ∣ a ∧ p ∣ c) ↔ (∀ n : ℤ, Int.gcd (a * n + b) (c * n + d) = 1) :=
sorry

end gcd_conditions_equivalent_l681_681846


namespace binom_18_6_l681_681511

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_18_6 : binomial 18 6 = 18564 := 
by
  sorry

end binom_18_6_l681_681511


namespace polygon_sides_l681_681077

theorem polygon_sides (n : ℕ) : 
  (∃ D, D = 104) ∧ (D = (n - 1) * (n - 4) / 2)  → n = 17 :=
by
  sorry

end polygon_sides_l681_681077


namespace length_of_platform_l681_681411

theorem length_of_platform (train_length : ℕ) (train_speed_kmph : ℕ) (cross_time_sec : ℕ)
  (h_train_length : train_length = 110)
  (h_train_speed : train_speed_kmph = 52)
  (h_cross_time : cross_time_sec = 30) :
  (↑train_speed_kmph * 1000 / 3600 * ↑cross_time_sec - ↑train_length) = 323.2 := by
  sorry

end length_of_platform_l681_681411


namespace pure_imaginary_z1z2_l681_681795

def z1 : ℂ := 3 + 2 * Complex.i
def z2 (m : ℝ) : ℂ := 1 + m * Complex.i

theorem pure_imaginary_z1z2 (m : ℝ) : (z1 * z2 m).re = 0 → m = 3 / 2 :=
by
  sorry

end pure_imaginary_z1z2_l681_681795


namespace simplest_form_fraction_C_l681_681406

def fraction_A (x : ℤ) (y : ℤ) : ℚ := (2 * x + 4) / (6 * x + 8)
def fraction_B (x : ℤ) (y : ℤ) : ℚ := (x + y) / (x^2 - y^2)
def fraction_C (x : ℤ) (y : ℤ) : ℚ := (x^2 + y^2) / (x + y)
def fraction_D (x : ℤ) (y : ℤ) : ℚ := (x^2 - y^2) / (x^2 - 2 * x * y + y^2)

theorem simplest_form_fraction_C (x y : ℤ) :
  ¬ (∃ (A : ℚ), A ≠ fraction_C x y ∧ (A = fraction_C x y)) :=
by
  intros
  sorry

end simplest_form_fraction_C_l681_681406


namespace sum_arithmetic_sequence_l681_681117

theorem sum_arithmetic_sequence : ∀ (a d l : ℕ), 
  (d = 2) → (a = 2) → (l = 20) → 
  ∃ (n : ℕ), (l = a + (n - 1) * d) ∧ 
  (∑ k in Finset.range n, (a + k * d)) = 110 :=
by
  intros a d l h_d h_a h_l
  use 10
  split
  · sorry
  · sorry

end sum_arithmetic_sequence_l681_681117


namespace triangle_obtuse_l681_681731

-- Given conditions and target
theorem triangle_obtuse (A B C : ℝ) (h : sin A * sin B < cos C) : 
  ∃ (ABC : Triangle), 
    (ABC.angle_A = A ∧ ABC.angle_B = B ∧ ABC.angle_C = C) ∧ ABC.is_obtuse :=
by
  sorry

end triangle_obtuse_l681_681731


namespace probability_multiple_of_4_is_2_over_5_l681_681714

-- Definitions from the conditions
def cards := {1, 2, 3, 4, 5, 6}

-- A function to determine if the product of two numbers is a multiple of 4
def is_multiple_of_4 (a b : ℕ) : Prop :=
  (a * b) % 4 = 0

-- Combinations of drawing 2 cards without replacement from a set of 6 cards
def pairs := { (a, b) : ℕ × ℕ | a ∈ cards ∧ b ∈ cards ∧ a < b }

-- Counting the total number of possible pairs
def total_pairs := pairs.size

-- Counting the favorable pairs where the product is a multiple of 4
def favorable_pairs := { p ∈ pairs | is_multiple_of_4 p.1 p.2 }.size

-- Probability calculation: favorable pairs divided by total pairs
noncomputable def probability := (favorable_pairs.toRat) / (total_pairs.toRat)

-- The main theorem stating the desired probability
theorem probability_multiple_of_4_is_2_over_5 : probability = 2 / 5 :=
  sorry

end probability_multiple_of_4_is_2_over_5_l681_681714


namespace greatest_div_by_seven_base_eight_l681_681896

theorem greatest_div_by_seven_base_eight : ∃ n : ℕ, 
  (n < 512) ∧ (Divisibility.divides 7 n) ∧ 
  (∀ m : ℕ, (m < 512) → (Divisibility.divides 7 m) → m ≤ n) ∧ 
  nat.to_digits 8 n = [7, 7, 4] := 
sorry

end greatest_div_by_seven_base_eight_l681_681896


namespace find_a5_geometric_sequence_l681_681535

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

end find_a5_geometric_sequence_l681_681535


namespace recurring_decimal_exceeds_by_fraction_l681_681112

theorem recurring_decimal_exceeds_by_fraction : 
  let y := (36 : ℚ) / 99
  let x := (36 : ℚ) / 100
  ((4 : ℚ) / 11) - x = (4 : ℚ) / 1100 :=
by
  sorry

end recurring_decimal_exceeds_by_fraction_l681_681112


namespace geometric_shape_l681_681208

noncomputable def zProperty (z : ℂ) : Prop := (abs (z - complex.i) = 1 ∧ z ≠ 0 ∧ z ≠ 2 * complex.i)
noncomputable def isReal (ω z : ℂ) : Prop := 
  (ω / (ω - 2 * complex.i) * (z - 2 * complex.i) / z).im = 0

theorem geometric_shape (z ω : ℂ) (hz : zProperty z) (hr : isReal ω z) : 
  abs (ω - complex.i) = 1 ∧ ω ≠ 0 ∧ ω ≠ 2 * complex.i :=
  sorry

end geometric_shape_l681_681208


namespace area_of_black_parts_l681_681351

theorem area_of_black_parts (x y : ℕ) (h₁ : x + y = 106) (h₂ : x + 2 * y = 170) : y = 64 :=
sorry

end area_of_black_parts_l681_681351


namespace common_difference_is_3_l681_681200

variables {a : ℕ → ℝ} {d a1 : ℝ}

-- Define the arithmetic sequence
def arithmetic_sequence (a_n : ℕ → ℝ) (a1 d : ℝ) : Prop := 
  ∀ n, a_n n = a1 + (n - 1) * d

-- Conditions
def a2_eq : a 2 = 3 := sorry
def a5_eq : a 5 = 12 := sorry

-- Theorem to prove the common difference is 3
theorem common_difference_is_3 :
  ∀ {a : ℕ → ℝ} {a1 d : ℝ},
  (arithmetic_sequence a a1 d)
  → a 2 = 3 
  → a 5 = 12 
  → d = 3 :=
  by
  intros a a1 d h_seq h_a2 h_a5
  sorry

end common_difference_is_3_l681_681200


namespace picture_distance_from_wall_l681_681972

theorem picture_distance_from_wall 
  (wall_width : ℝ) (border_width : ℝ) (picture_width : ℝ) (effective_width : ℝ) (x : ℝ) 
  (hw : wall_width = 28) (hb : border_width = 2) (hp : picture_width = 4) 
  (hw_border : effective_width = wall_width - 2 * border_width)
  (hx : 2 * x + picture_width = effective_width) :
  x + border_width = 12 :=
by
  rw [hw, hb, hp] at hw_border hx
  sorry  

end picture_distance_from_wall_l681_681972


namespace five_digit_numbers_divisible_by_11_l681_681306

theorem five_digit_numbers_divisible_by_11 :
  let n_lower := 10000
  let n_upper := 99999
  let count_valid_n := (λ (n : ℕ), ∃ (q r : ℕ), n = 50 * q + r ∧ (1000 ≤ q) ∧ (q ≤ 1999) ∧ (0 ≤ r) ∧ (r < 50) ∧ (q + r) % 11 = 0)
  ∃ valid_n_count : ℕ, valid_n_count = 5000 ∧ (valid_n_count = (Finset.card (Finset.filter count_valid_n (Finset.range (n_upper + 1) \ Finset.range n_lower)))) :=
by
  sorry

end five_digit_numbers_divisible_by_11_l681_681306


namespace cone_to_sphere_ratio_l681_681080

-- Prove the ratio of the cone's altitude to its base radius
theorem cone_to_sphere_ratio (r h : ℝ) (h_r_pos : 0 < r) 
  (vol_cone : ℝ) (vol_sphere : ℝ) 
  (hyp_vol_relation : vol_cone = (1 / 3) * vol_sphere)
  (vol_sphere_def : vol_sphere = (4 / 3) * π * r^3)
  (vol_cone_def : vol_cone = (1 / 3) * π * r^2 * h) :
  h / r = 4 / 3 :=
by
  sorry

end cone_to_sphere_ratio_l681_681080


namespace inequality_solution_l681_681842

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (3 * x + 4) < 5 ↔ x < -4 / 3 ∨ -3 / 2 < x := 
sorry

end inequality_solution_l681_681842


namespace probability_multiple_of_4_is_2_over_5_l681_681716

-- Definitions from the conditions
def cards := {1, 2, 3, 4, 5, 6}

-- A function to determine if the product of two numbers is a multiple of 4
def is_multiple_of_4 (a b : ℕ) : Prop :=
  (a * b) % 4 = 0

-- Combinations of drawing 2 cards without replacement from a set of 6 cards
def pairs := { (a, b) : ℕ × ℕ | a ∈ cards ∧ b ∈ cards ∧ a < b }

-- Counting the total number of possible pairs
def total_pairs := pairs.size

-- Counting the favorable pairs where the product is a multiple of 4
def favorable_pairs := { p ∈ pairs | is_multiple_of_4 p.1 p.2 }.size

-- Probability calculation: favorable pairs divided by total pairs
noncomputable def probability := (favorable_pairs.toRat) / (total_pairs.toRat)

-- The main theorem stating the desired probability
theorem probability_multiple_of_4_is_2_over_5 : probability = 2 / 5 :=
  sorry

end probability_multiple_of_4_is_2_over_5_l681_681716


namespace num_sequences_with_an_zero_l681_681778

open Nat

theorem num_sequences_with_an_zero :
  let S := { (a1, a2, a3, a4) : ℕ × ℕ × ℕ × ℕ | 1 ≤ a1 ∧ a1 ≤ 15 ∧ 1 ≤ a2 ∧ a2 ≤ 15 ∧ 1 ≤ a3 ∧ a3 ≤ 15 ∧ 1 ≤ a4 ∧ a4 ≤ 15 }
  in ∃ (seq_in_S : (ℕ × ℕ × ℕ × ℕ) → ℕ → ℕ), 
  (∀ a ∈ S, ∀ n ≥ 5, seq_in_S a n = (seq_in_S a (n - 1)) * abs (seq_in_S a (n - 2) - seq_in_S a (n - 3))) ∧
  (num_zeros : S → ℕ) ∧
  (∀ a ∈ S, num_zeros a = if (∃ n, n ≥ 5 ∧ seq_in_S a n = 0) then 1 else 0) ∧
  (∑ a in S, num_zeros a) = 6750 :=
by
  sorry

end num_sequences_with_an_zero_l681_681778


namespace original_number_is_16_l681_681963

theorem original_number_is_16 (x : ℤ) (h1 : 3 * (2 * x + 5) = 111) : x = 16 :=
by
  sorry

end original_number_is_16_l681_681963


namespace customers_tried_sample_l681_681984

theorem customers_tried_sample
  (samples_per_box : ℕ)
  (boxes_opened : ℕ)
  (samples_left_over : ℕ)
  (samples_per_customer : ℕ := 1)
  (h_samples_per_box : samples_per_box = 20)
  (h_boxes_opened : boxes_opened = 12)
  (h_samples_left_over : samples_left_over = 5) :
  (samples_per_box * boxes_opened - samples_left_over) / samples_per_customer = 235 :=
by
  sorry

end customers_tried_sample_l681_681984


namespace chest_value_in_base10_l681_681456

def base5_to_base10 (n: Nat) : Nat :=
  nat.rec_on n 0 (λ d rec, (d % 10) * 5 ^ (n / 10) + rec (n / 10))

def convert_and_sum : Nat :=
  let silverware := 3214
  let gemstones := 3022
  let silk := 202
  base5_to_base10 silverware + base5_to_base10 gemstones + base5_to_base10 silk

theorem chest_value_in_base10 :
  convert_and_sum = 873 :=
  by
    sorry

end chest_value_in_base10_l681_681456


namespace number_of_functions_l681_681533

theorem number_of_functions :
  ∃ (f : Fin 9 → Fin 9), (∀ x : Fin 9, (f ∘ f ∘ f ∘ f ∘ f) x = x) → 3025 := 
sorry

end number_of_functions_l681_681533


namespace area_enclosed_by_curve_l681_681357

theorem area_enclosed_by_curve : 
  let curve_eq (x y : ℝ) := abs (x - 1) + abs (y - 1) = 1 in
  (area of the region enclosed by curve_eq equals 2) :=
begin
  sorry
end

end area_enclosed_by_curve_l681_681357


namespace find_CM_l681_681953

noncomputable theory

-- Define necessary geometric entities and given conditions
def AM : ℝ := 1
def BM : ℝ := 4
def angle_BAC : ℝ := 120

-- Translate the problem statement to Lean
theorem find_CM (hAM : AM = 1) (hBM : BM = 4) (hAngle : angle_BAC = 120) : 
  ∃ CM : ℝ, CM = Real.sqrt 273 := 
sorry

end find_CM_l681_681953


namespace bisect_by_median_l681_681481

open EuclideanGeometry

theorem bisect_by_median
  (A B C D E F K O : Point)
  (h1 : Excircle O (triangle A B C) BC)
  (h2 : TangentPoint D O BC)
  (h3 : TangentPoint E O CA)
  (h4 : TangentPoint F O AB)
  (h5 : IntersectAt K (LineThrough O D) (LineThrough E F)) :
  Bisects (LineThrough A K) BC :=
sorry

end bisect_by_median_l681_681481


namespace P_le_0_l681_681225

noncomputable def Ξ := MeasureTheory.ProbabilityTheory.Normal 2 (σ^2)

axiom P_le_4 : MeasureTheory.ProbabilityTheory.Probability (Ξ ≤ 4) = 0.68

theorem P_le_0 : MeasureTheory.ProbabilityTheory.Probability (Ξ ≤ 0) = 0.32 := 
by 
  sorry

end P_le_0_l681_681225


namespace numberOfRectangularFormations_l681_681962

-- Defining the conditions
def isRectangularFormation (s t : ℕ) : Prop := s * t = 420

def validMusiciansPerRow (t : ℕ) : Prop := 12 ≤ t ∧ t ≤ 50

-- Defining the statement x is the number of valid formations
theorem numberOfRectangularFormations : ∃ z : ℕ, z = 8 ∧ 
  (z = (List.length ((List.filter 
    (λ (t : ℕ × ℕ), isRectangularFormation t.1 t.2 ∧ validMusiciansPerRow t.2) 
    (List.product (List.range 421) (List.range 51))) : List (ℕ × ℕ)))) :=
sorry

end numberOfRectangularFormations_l681_681962


namespace mary_score_unique_l681_681813

theorem mary_score_unique (c w : ℕ) (s : ℕ) (h_score_formula : s = 35 + 4 * c - w)
  (h_limit : c + w ≤ 35) (h_greater_90 : s > 90) :
  (∀ s' > 90, s' ≠ s → ¬ ∃ c' w', s' = 35 + 4 * c' - w' ∧ c' + w' ≤ 35) → s = 91 :=
by
  sorry

end mary_score_unique_l681_681813


namespace number_of_valid_three_digit_numbers_l681_681253

-- Define the conditions for a three-digit number with each digit being 2 or 5.
def is_valid_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 5

def is_valid_three_digit_number (n : ℕ) : Prop :=
  let digits := [n / 100, (n % 100) / 10, n % 10] in
  ∀ d ∈ digits, is_valid_digit d

-- The proof statement: Prove that there are exactly 8 three-digit numbers satisfying the conditions.
theorem number_of_valid_three_digit_numbers : 
  (∃! (n : ℕ), 
      n >= 100 ∧ 
      n < 1000 ∧ 
      is_valid_three_digit_number n) = 8 :=
by sorry

end number_of_valid_three_digit_numbers_l681_681253


namespace parabola_equation_l681_681966

theorem parabola_equation (h1: ∃ k, ∀ x y : ℝ, (x, y) = (4, -2) → y^2 = k * x) 
                          (h2: ∃ m, ∀ x y : ℝ, (x, y) = (4, -2) → x^2 = -2 * m * y) :
                          (y : ℝ)^2 = x ∨ (x : ℝ)^2 = -8 * y :=
by 
  sorry

end parabola_equation_l681_681966


namespace floor_of_4point7_l681_681549

theorem floor_of_4point7 : (Real.floor 4.7) = 4 :=
by
  sorry

end floor_of_4point7_l681_681549


namespace solve_exp_l681_681048

theorem solve_exp (x : ℕ) : 8^x = 2^9 → x = 3 :=
by
  sorry

end solve_exp_l681_681048


namespace perimeter_quarter_circle_square_l681_681458

theorem perimeter_quarter_circle_square (s : ℝ) (h : s = 4 / real.pi) : 
  let r := s in
  let C := 2 * real.pi * r in
  let quarter_arc_length := C / 4 in
  4 * quarter_arc_length = 8 :=
by
  sorry

end perimeter_quarter_circle_square_l681_681458


namespace chickens_cheaper_than_eggs_l681_681828

-- Define the initial costs of the chickens
def initial_cost_chicken1 : ℝ := 25
def initial_cost_chicken2 : ℝ := 30
def initial_cost_chicken3 : ℝ := 22
def initial_cost_chicken4 : ℝ := 35

-- Define the weekly feed costs for the chickens
def weekly_feed_cost_chicken1 : ℝ := 1.50
def weekly_feed_cost_chicken2 : ℝ := 1.30
def weekly_feed_cost_chicken3 : ℝ := 1.10
def weekly_feed_cost_chicken4 : ℝ := 0.90

-- Define the weekly egg production for the chickens
def weekly_egg_prod_chicken1 : ℝ := 4
def weekly_egg_prod_chicken2 : ℝ := 3
def weekly_egg_prod_chicken3 : ℝ := 5
def weekly_egg_prod_chicken4 : ℝ := 2

-- Define the cost of a dozen eggs at the store
def cost_per_dozen_eggs : ℝ := 2

-- Define total initial costs, total weekly feed cost, and weekly savings
def total_initial_cost : ℝ := initial_cost_chicken1 + initial_cost_chicken2 + initial_cost_chicken3 + initial_cost_chicken4
def total_weekly_feed_cost : ℝ := weekly_feed_cost_chicken1 + weekly_feed_cost_chicken2 + weekly_feed_cost_chicken3 + weekly_feed_cost_chicken4
def weekly_savings : ℝ := cost_per_dozen_eggs

-- Define the condition for the number of weeks (W) when the chickens become cheaper
def breakeven_weeks : ℝ := 40

theorem chickens_cheaper_than_eggs (W : ℕ) :
  total_initial_cost + W * total_weekly_feed_cost = W * weekly_savings :=
sorry

end chickens_cheaper_than_eggs_l681_681828


namespace solve_x_l681_681267

theorem solve_x (x : ℝ) (h : 2 - 2 / (1 - x) = 2 / (1 - x)) : x = -2 := 
by
  sorry

end solve_x_l681_681267


namespace trapezium_area_correct_l681_681047

-- Define the given lengths and distance
def side1 : ℝ := 20
def side2 : ℝ := 18
def distance : ℝ := 20

-- Define the area formula for a trapezium
def trapezium_area (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

-- State the theorem
theorem trapezium_area_correct : trapezium_area side1 side2 distance = 380 := by
  sorry

end trapezium_area_correct_l681_681047


namespace find_f_neg_5_l681_681626

noncomputable def f (x : ℝ) : ℝ := sorry

-- Conditions
axiom is_even : ∀ x : ℝ, f(-x) = f(x)
axiom transform_condition : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f(2 + x) = f(2 - x)
axiom function_condition : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f(x) = x^2 - 2 * x

-- Theorem stating the problem
theorem find_f_neg_5 : f(-5) = -1 :=
by sorry

end find_f_neg_5_l681_681626


namespace greatest_base8_three_digit_divisible_by_7_l681_681906

theorem greatest_base8_three_digit_divisible_by_7 :
  ∃ n : ℕ, n < 8^3 ∧ n ≥ 8^2 ∧ (n % 7 = 0) ∧ (to_base 8 n = 777) :=
sorry

end greatest_base8_three_digit_divisible_by_7_l681_681906


namespace probability_of_area_condition_l681_681970

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  1 / 2 * a * b

def probability (triangle_area half_triangle_area: ℝ) : ℝ :=
  (triangle_area - half_triangle_area) / triangle_area

theorem probability_of_area_condition
  : let X := (0, 6) in
    let Y := (0, 0) in
    let Z := (9, 0) in
    let area_XYZ := area_of_triangle 9 6 0 in
    let half_area_XYZ := 1 / 2 * area_XYZ in
    probability area_XYZ half_area_XYZ = 3 / 4 :=
by
  sorry

end probability_of_area_condition_l681_681970


namespace domain_of_log3_l681_681853

-- Definitions
def f (x : ℝ) : ℝ := log x

-- Problem Statement: Prove that the domain of f(x) = log_3(x-1) is (1, +∞)
theorem domain_of_log3 (x : ℝ) : (∃ (y : ℝ), f (x - 1) = y) ↔ x ∈ set.Ioo 1 (⊤) :=
by
  sorry

end domain_of_log3_l681_681853


namespace mutually_exclusive_A_C_l681_681202

-- Definitions of events:
def A (items : List Bool) := ∀ (i : Bool) in items, i = true
def B (items : List Bool) := ∀ (i : Bool) in items, i = false
def C (items : List Bool) := ∃ (i : Bool) in items, i = false

-- Statement to prove
theorem mutually_exclusive_A_C (items : List Bool) :
  (A items ∧ C items) -> False := by
  sorry

end mutually_exclusive_A_C_l681_681202


namespace distinct_triangles_count_l681_681691

theorem distinct_triangles_count (a b c : ℕ) (h1 : a + b + c = 11) (h2 : a + b > c) 
  (h3 : a + c > b) (h4 : b + c > a) : 
  10 := sorry

end distinct_triangles_count_l681_681691


namespace monotonic_decreasing_condition_l681_681652

theorem monotonic_decreasing_condition {f : ℝ → ℝ} (a : ℝ) :
  (∀ x ∈ Ioo (0:ℝ) 1, (2:ℝ) ^ (x * (x - a)) < (2:ℝ) ^ ((1:ℝ) * (1 - a))) → a ≥ 2 :=
begin
  sorry
end

end monotonic_decreasing_condition_l681_681652


namespace distinct_triangles_count_l681_681689

theorem distinct_triangles_count (a b c : ℕ) (h1 : a + b + c = 11) (h2 : a + b > c) 
  (h3 : a + c > b) (h4 : b + c > a) : 
  10 := sorry

end distinct_triangles_count_l681_681689


namespace line_equation_l681_681210

theorem line_equation (P : ℝ × ℝ) (l : ℝ → ℝ → Prop) : 
  P = (2, 3) → 
  (∃ a b: ℝ, a > 0 ∧ b > 0 ∧ (l a b ↔ (x/a + y/b = 1)) 
  ∧ a + b = 0 
  ∧ (1/2 * a * b = 16)) 
  → (∀ x y, l x y ↔ (x - y + 1 = 0) ∨ (9 * x + 2 * y - 24 = 0)) := 
begin
  intro hP,
  intros hexists,
  sorry -- This is a complex proof and is left as an exercise
end

end line_equation_l681_681210


namespace problem_l681_681789

theorem problem (n : ℕ) (h : n = 8 ^ 2022) : n / 4 = 4 ^ 3032 := 
sorry

end problem_l681_681789


namespace commute_days_l681_681454

theorem commute_days (train_count car_morning_count car_afternoon_count : ℕ) (h_train : train_count = 9) (h_car_morning : car_morning_count = 8) (h_car_afternoon : car_afternoon_count = 15) :
  let x := (train_count + car_morning_count + car_afternoon_count) / 2 in x = 16 :=
by
  sorry

end commute_days_l681_681454


namespace range_a_ineq_am_gm_ineq_l681_681055

-- Problem 1
theorem range_a_ineq (a : ℝ) : (∀ x ∈ (set.Icc (1/2:ℝ) 1), |x - a| + |2 * x - 1| ≤ |2 * x + 1|) ↔ -1 ≤ a ∧ a ≤ 5 / 2 :=
sorry

-- Problem 2
theorem am_gm_ineq (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
  a^2 * b^2 + a^2 + b^2 ≥ a * b * (a + b + 1) :=
sorry

end range_a_ineq_am_gm_ineq_l681_681055


namespace find_standard_parabola_equation_find_tangent_lines_to_parabola_l681_681214

noncomputable def parabola_focus (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p / 2, 0)

theorem find_standard_parabola_equation (p : ℝ) (hp : p > 0) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ (A.2^2 = 2 * p * A.1) ∧ (B.2^2 = 2 * p * B.1) ∧
    let F := parabola_focus p hp in
    let l := line_through F (F.1, F.2, real.sqrt 3) in
    A ∈ l ∧ B ∈ l ∧ dist A B = 16 / 3) →
  (p = 2) ∧ (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x) :=
sorry

noncomputable def left_focus_of_ellipse (a : ℝ) (b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ × ℝ := ( -a, 0)

theorem find_tangent_lines_to_parabola (p : ℝ) (hp : p > 0) :
  let C := λ x y : ℝ, y^2 = 2 * p * x in
  let ellipse := λ x y : ℝ, x^2 / 2 + y^2 = 1 in
  let F := left_focus_of_ellipse 1 1 (by linarith) (by linarith) in
  (∃ n m : ℝ, is_tangent_line n m F (2, C)) →
  ((∀ y : ℝ, ∃ x : ℝ, x = y - 1 ∨ x = -y - 1) ↔ is_tangent_parabola 1 2 F (C y)) :=
sorry

end find_standard_parabola_equation_find_tangent_lines_to_parabola_l681_681214


namespace james_needs_50_hours_l681_681764

-- Define the given conditions
def meat_cost_per_pound : ℝ := 5
def meat_pounds_wasted : ℝ := 20
def fruits_veg_cost_per_pound : ℝ := 4
def fruits_veg_pounds_wasted : ℝ := 15
def bread_cost_per_pound : ℝ := 1.5
def bread_pounds_wasted : ℝ := 60
def janitorial_hourly_rate : ℝ := 10
def janitorial_hours_worked : ℝ := 10
def james_hourly_rate : ℝ := 8

-- Calculate the total costs separately
def total_meat_cost : ℝ := meat_cost_per_pound * meat_pounds_wasted
def total_fruits_veg_cost : ℝ := fruits_veg_cost_per_pound * fruits_veg_pounds_wasted
def total_bread_cost : ℝ := bread_cost_per_pound * bread_pounds_wasted
def janitorial_time_and_a_half_rate : ℝ := janitorial_hourly_rate * 1.5
def total_janitorial_cost : ℝ := janitorial_time_and_a_half_rate * janitorial_hours_worked

-- Calculate the total cost James has to pay
def total_cost : ℝ := total_meat_cost + total_fruits_veg_cost + total_bread_cost + total_janitorial_cost

-- Calculate the number of hours James needs to work
def james_work_hours : ℝ := total_cost / james_hourly_rate

-- The theorem to be proved
theorem james_needs_50_hours : james_work_hours = 50 :=
by
  sorry

end james_needs_50_hours_l681_681764


namespace students_not_invited_count_l681_681271

-- Define the total number of students
def total_students : ℕ := 30

-- Define the number of students not invited to the event
def not_invited_students : ℕ := 14

-- Define the sets representing different levels of friends of Anna
-- This demonstrates that the total invited students can be derived from given conditions

def anna_immediate_friends : ℕ := 4
def anna_second_level_friends : ℕ := (12 - anna_immediate_friends)
def anna_third_level_friends : ℕ := (16 - 12)

-- Define total invited students
def invited_students : ℕ := 
  anna_immediate_friends + 
  anna_second_level_friends +
  anna_third_level_friends

-- Prove that the number of not invited students is 14
theorem students_not_invited_count : (total_students - invited_students) = not_invited_students :=
by
  sorry

end students_not_invited_count_l681_681271


namespace probability_multiple_of_4_is_2_over_5_l681_681715

-- Definitions from the conditions
def cards := {1, 2, 3, 4, 5, 6}

-- A function to determine if the product of two numbers is a multiple of 4
def is_multiple_of_4 (a b : ℕ) : Prop :=
  (a * b) % 4 = 0

-- Combinations of drawing 2 cards without replacement from a set of 6 cards
def pairs := { (a, b) : ℕ × ℕ | a ∈ cards ∧ b ∈ cards ∧ a < b }

-- Counting the total number of possible pairs
def total_pairs := pairs.size

-- Counting the favorable pairs where the product is a multiple of 4
def favorable_pairs := { p ∈ pairs | is_multiple_of_4 p.1 p.2 }.size

-- Probability calculation: favorable pairs divided by total pairs
noncomputable def probability := (favorable_pairs.toRat) / (total_pairs.toRat)

-- The main theorem stating the desired probability
theorem probability_multiple_of_4_is_2_over_5 : probability = 2 / 5 :=
  sorry

end probability_multiple_of_4_is_2_over_5_l681_681715


namespace bee_distance_Q0_to_Q1024_l681_681062

/-
A bee starts flying from point Q0. She flies 2 inches due east to point Q1.
For k ≥ 1, once the bee reaches point Qk, she turns 45 degrees counterclockwise and then flies 2k inches 
straight to point Q(k+1). When the bee reaches Q1024, how far from Q0 is she, in inches?
-/

def distance_from_origin_to_Q1024 : ℝ :=
  let ξ := complex.exp (real.pi * complex.I / 4)
  -- Calculate z using the provided conditions
  let z := (∑ k in finset.range 1024, (2 + 2 * k : ℂ) * ξ ^ k)
  -- The modulus of z
  complex.abs z

theorem bee_distance_Q0_to_Q1024 :
  distance_from_origin_to_Q1024 = 2049 * real.sqrt 2 :=
sorry

end bee_distance_Q0_to_Q1024_l681_681062


namespace bobby_has_candy_left_l681_681995

def initial_candy := 36
def candy_eaten_first := 17
def candy_eaten_second := 15

theorem bobby_has_candy_left : 
  initial_candy - (candy_eaten_first + candy_eaten_second) = 4 := 
by
  sorry


end bobby_has_candy_left_l681_681995


namespace smallest_three_digit_divisible_by_3_and_6_l681_681031

theorem smallest_three_digit_divisible_by_3_and_6 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999 ∧ n % 3 = 0 ∧ n % 6 = 0) ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 3 = 0 ∧ m % 6 = 0 → n ≤ m) ∧ n = 102 := 
by {sorry}

end smallest_three_digit_divisible_by_3_and_6_l681_681031


namespace monotonic_decreasing_condition_l681_681653

theorem monotonic_decreasing_condition {f : ℝ → ℝ} (a : ℝ) :
  (∀ x ∈ Ioo (0:ℝ) 1, (2:ℝ) ^ (x * (x - a)) < (2:ℝ) ^ ((1:ℝ) * (1 - a))) → a ≥ 2 :=
begin
  sorry
end

end monotonic_decreasing_condition_l681_681653


namespace lucy_sales_is_43_l681_681829

def total_packs : Nat := 98
def robyn_packs : Nat := 55
def lucy_packs : Nat := total_packs - robyn_packs

theorem lucy_sales_is_43 : lucy_packs = 43 :=
by
  sorry

end lucy_sales_is_43_l681_681829


namespace arcsin_neg_one_half_l681_681496

theorem arcsin_neg_one_half : Real.arcsin (-1 / 2) = -Real.pi / 6 :=
by
  sorry

end arcsin_neg_one_half_l681_681496


namespace sum_of_elements_less_than_0_3_l681_681883

-- Defining the list of numbers
def list : List ℝ := [0.8, 1/2, 0.9, 0.2, 1/3]

-- Theorem statement
theorem sum_of_elements_less_than_0_3 :
  (list.filter (λ x : ℝ, x < 0.3)).sum = 0.2 :=
by
  -- Skip the actual proof for now
  sorry

end sum_of_elements_less_than_0_3_l681_681883


namespace find_initial_sum_l681_681086

-- Define the conditions as per the problem statement
def initial_sum (P : ℝ) : Prop :=
  let SI := P * 0.09 * 3
  let CI := P * (1.08 ^ 2) - P
  let TI := SI + CI
  TI = 4016.25

-- Define the statement that needs to be proven
theorem find_initial_sum : 
  ∃ P : ℝ, initial_sum P ∧ P ≈ 97.52 := 
by 
  sorry


end find_initial_sum_l681_681086


namespace greatest_3_digit_base8_num_div_by_7_eq_511_l681_681921

noncomputable def greatest_base8_number_divisible_by_7 : ℕ := 7 * 73

theorem greatest_3_digit_base8_num_div_by_7_eq_511 : 
  greatest_base8_number_divisible_by_7 = 511 :=
by 
  sorry

end greatest_3_digit_base8_num_div_by_7_eq_511_l681_681921


namespace total_balls_l681_681735

theorem total_balls (r b g : ℕ) (ratio : r = 2 * k ∧ b = 4 * k ∧ g = 6 * k) (green_balls : g = 36) : r + b + g = 72 :=
by
  sorry

end total_balls_l681_681735


namespace polynomial_root_value_l681_681803

def roots_condition (a b c : ℝ) : Prop :=
  a + b + c = 15 ∧ ab + bc + ca = 22 ∧ abc = 8

theorem polynomial_root_value {a b c : ℝ} (h : roots_condition a b c) :
  (2 + a) * (2 + b) * (2 + c) = 120 :=
by
  -- The proof is omitted as instructed
  sorry

end polynomial_root_value_l681_681803


namespace population_at_end_of_three_years_l681_681383

def InitialPopulation : ℕ := 60000
def GrowthRate_FirstYear : ℚ := 0.10
def GrowthRate_SecondYear : ℚ := 0.07
def GrowthRate_ThirdYear : ℚ := 0.15

def PopulationEndOfYear (initial_pop : ℕ) (growth_rate : ℚ) : ℕ :=
  initial_pop + (initial_pop * growth_rate).toNat

def PopulationEndOfFirstYear : ℕ :=
  PopulationEndOfYear InitialPopulation GrowthRate_FirstYear

def PopulationEndOfSecondYear : ℕ :=
  PopulationEndOfYear PopulationEndOfFirstYear GrowthRate_SecondYear

def PopulationEndOfThirdYear : ℕ :=
  PopulationEndOfYear PopulationEndOfSecondYear GrowthRate_ThirdYear

theorem population_at_end_of_three_years : PopulationEndOfThirdYear = 81213 := by
  sorry

end population_at_end_of_three_years_l681_681383


namespace even_function_periodic_property_value_of_f_at_8pi_over_3_l681_681856

def f (x : Real) : Real :=
if h : 0 ≤ x ∧ x < Real.pi / 2 then
  sqrt 3 * Real.tan x - 1
else if 0 ≤ -x ∧ -x < Real.pi / 2 then
  sqrt 3 * Real.tan (-x) - 1
else
  sorry  -- This is a placeholder to handle other cases by periodicity.

theorem even_function_periodic_property : 
  (∀ x : Real, f (-x) = f x) ∧ 
  (∀ (x : Real) (n : Int), f (x + n * Real.pi) = f x) ∧
  f (Real.pi / 3) = sqrt 3 * sqrt 3 - 1 := 
by sorry

theorem value_of_f_at_8pi_over_3 : f (8 * Real.pi / 3) = 2 :=
by
  have even_function := even_function_periodic_property.1
  have periodic_function := even_function_periodic_property.2
  have value_at_pi_over_3 := even_function_periodic_property.3
  calc
    f (8 * Real.pi / 3) = f (- Real.pi / 3) := by rw [periodic_function, even_function]
    ... = f (Real.pi / 3) := by rw [even_function]
    ... = sqrt 3 * Real.sqrt 3 - 1 := value_at_pi_over_3
    ... = 3 - 1 := by sorry
    ... = 2 := by sorry

end even_function_periodic_property_value_of_f_at_8pi_over_3_l681_681856


namespace probability_of_product_multiple_of_4_is_2_5_l681_681708

open Finset BigOperators

def all_pairs (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s \ s.diag

def num_favorable_pairs (s : Finset ℕ) : ℕ :=
  (all_pairs s).filter (λ p => (p.1 * p.2) % 4 = 0).card

def probability_multiple_of_4 : ℚ :=
  let s := (finset.range 7).filter (λ n => n ≠ 0)
  let total_pairs := (all_pairs s).card
  let favorable_pairs := num_favorable_pairs s
  favorable_pairs / total_pairs

theorem probability_of_product_multiple_of_4_is_2_5 :
  probability_multiple_of_4 = 2 / 5 := by
  -- skipping the proof
  sorry

end probability_of_product_multiple_of_4_is_2_5_l681_681708


namespace find_third_grade_students_l681_681884

variable (third_grade_students fourth_grade_students total_students : ℕ)

-- Given conditions
def condition1 : fourth_grade_students = 237 := rfl
def condition2 : total_students = 391 := rfl

-- Proof statement
def third_grade_students_agreed : Prop :=
  third_grade_students = total_students - fourth_grade_students

-- The actual number of third grade students who agreed
theorem find_third_grade_students : third_grade_students = 154 := by
  have h1 : fourth_grade_students = 237 := condition1
  have h2 : total_students = 391 := condition2
  calc
    third_grade_students = total_students - fourth_grade_students := by 
      exact third_grade_students_agreed
    ... = 391 - 237 := by rw [condition2, condition1]
    ... = 154 := by norm_num

end find_third_grade_students_l681_681884


namespace percentage_decrease_in_consumption_l681_681726

variable (P C : ℝ) (h : P > 0) (hc : C > 0)

theorem percentage_decrease_in_consumption :
  let C_new := P * C / (P + 40) in
  let percentage_decrease := (40 / (P + 40)) * 100 in
  (C - C_new) / C * 100 = percentage_decrease :=
by
  sorry

end percentage_decrease_in_consumption_l681_681726


namespace probability_inside_triangle_l681_681215

theorem probability_inside_triangle (A B C P : Point) (h : (PB + PC + 4 * PA = 0)) :
  probability_in_triangle PBC ABC = 1/3 :=
sorry

end probability_inside_triangle_l681_681215


namespace bread_slices_per_loaf_l681_681452

theorem bread_slices_per_loaf (friends: ℕ) (total_loaves : ℕ) (slices_per_friend: ℕ) (total_slices: ℕ)
  (h1 : friends = 10) (h2 : total_loaves = 4) (h3 : slices_per_friend = 6) (h4 : total_slices = friends * slices_per_friend):
  total_slices / total_loaves = 15 :=
by
  sorry

end bread_slices_per_loaf_l681_681452


namespace milk_production_l681_681105

theorem milk_production (a b c d e : ℕ) (f g : ℝ) (hf : f = 0.8) (hg : g = 1.1) :
  ((d : ℝ) * e * g * (b : ℝ) / (a * c)) = 1.1 * b * d * e / (a * c) := by
  sorry

end milk_production_l681_681105


namespace manager_profit_goal_l681_681378

noncomputable def percentage_profit (cost_price sell_price : ℝ) (wastage_percentage : ℝ) : ℝ :=
  let remaining_percentage := 1 - wastage_percentage
  let revenue := sell_price * remaining_percentage
  let profit := (revenue - cost_price) * 100 / cost_price
  profit

theorem manager_profit_goal : percentage_profit 0.80 0.968888888888889 0.10 ≈ 9 := 
by
  sorry

end manager_profit_goal_l681_681378


namespace polynomial_condition_matches_l681_681586

noncomputable def q (x : ℝ) : ℝ := 16 * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e

theorem polynomial_condition_matches (q : ℝ → ℝ)
  (h : ∀ x : ℝ, q (x ^ 4) - q (x ^ 4 - 4) = (q x) ^ 2 + 16) :
  ∃ b c d e : ℝ, q = λ x, 16 * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e :=
sorry

end polynomial_condition_matches_l681_681586


namespace percent_decrease_long_distance_call_l681_681360

theorem percent_decrease_long_distance_call :
  let c1990 := 35
  let c2010 := 5
  let percent_decrease := ((c1990 - c2010).toFloat / c1990.toFloat) * 100
  abs (percent_decrease - 86) < 1 := 
by 
  sorry

end percent_decrease_long_distance_call_l681_681360


namespace number_of_triangles_with_perimeter_11_l681_681699

theorem number_of_triangles_with_perimeter_11 :
  {t : (ℕ × ℕ × ℕ) // let (a, b, c) := t in a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ c + a > b}.card = 4 :=
by sorry

end number_of_triangles_with_perimeter_11_l681_681699


namespace weight_of_new_man_l681_681413

-- Definitions based on the given conditions
def original_average_weight := δ  -- let δ be the original average weight of the 10 men, this variable is introduced for conceptual clarity

-- Given conditions translated into Lean
def original_weight_sum := 10 * δ -- The total original weight of the 10 men
def increased_average_weight := original_average_weight + 2.5 -- The new average weight after the replacement
def increased_weight_sum := 10 * increased_average_weight -- The new total weight of the 10 men
def weight_of_replaced_man := 58 -- The weight of the man that is replaced

-- Expression for the new total weight after replacement
def additional_weight := 10 * 2.5 -- Total increase in weight for the 10 men

-- Prove that the weight of the new man is 83 kg
theorem weight_of_new_man :
  let weight_of_new_man := weight_of_replaced_man + additional_weight in
  weight_of_new_man = 83 :=
by
  sorry

end weight_of_new_man_l681_681413


namespace rectangle_BP_PQ_QD_l681_681742

theorem rectangle_BP_PQ_QD
  (ABCD : ℝ → ℝ → Prop)
  (H1 : ∀ A B C D : ℝ, rectangle AB CD)
  (H2 : ∀ A B : ℝ, AB = 8)
  (H3 : ∀ B C : ℝ, BC = 4)
  (E F : ℝ)
  (H4 : BE = 2 * EF)
  (H5 : EF = FC)
  (H6 : E ∈ BC)
  (P Q : ℝ)
  (H7 : AE ∩ BD = P)
  (H8 : AF ∩ BD = Q) :
  let BP : ℝ := |B - P|
  let PQ : ℝ := |P - Q|
  let QD : ℝ := |Q - D|
  in BP / PQ = 1 / 2 ∧ PQ / QD = 2 / 1 := sorry

end rectangle_BP_PQ_QD_l681_681742


namespace length_of_train_l681_681986

-- Definitions of the given conditions
def train_speed_kmph : ℝ := 72 -- speed in kmph
def platform_length : ℝ := 50.024 -- length of the platform in meters
def time_to_cross_platform : ℝ := 15 -- time in seconds

-- Definition for the conversion factor
def kmph_to_mps : ℝ := 1 / 3.6

-- Convert train speed from kmph to mps
def train_speed_mps := train_speed_kmph * kmph_to_mps

-- The total distance covered by the train while crossing the platform
def total_distance_covered := train_speed_mps * time_to_cross_platform

-- Definition for the length of the train
def train_length := total_distance_covered - platform_length

-- Statement to be proved
theorem length_of_train : 
  train_length = 249.976 := 
by 
  sorry

end length_of_train_l681_681986


namespace minimal_pairs_in_chessboard_l681_681892

theorem minimal_pairs_in_chessboard (R : Fin 8 → Fin 8 → Prop) (hR : ∑ (i : Fin 8) (j : Fin 8), if R i j then 1 else 0 = 16) :
  ∑ i : Fin 8, (∑ j : Fin 8, if R i j then 1 else 0) * (∑ j : Fin 8, if R i j then 1 else 0 - 1) / 2 +
  ∑ j : Fin 8, (∑ i : Fin 8, if R i j then 1 else 0) * (∑ i : Fin 8, if R i j then 1 else 0 - 1) / 2 ≥ 16 := 
  sorry

end minimal_pairs_in_chessboard_l681_681892


namespace binom_18_6_eq_4765_l681_681529

def binom (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_18_6_eq_4765 : binom 18 6 = 4765 := by
  sorry

end binom_18_6_eq_4765_l681_681529


namespace general_term_an_sum_bn_l681_681748

noncomputable def arithmetic_sequence_sum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

noncomputable def geometric_sequence_sum (p : ℝ) (n : ℕ) : ℝ :=
  p * (1 - p^(n : ℝ)) / (1 - p)

noncomputable def Tn_p_eq_1 (n : ℕ) : ℝ :=
  n * (n + 1) / 2

noncomputable def Tn_p_neq_1 (p : ℝ) (n : ℕ) : ℝ :=
  geometric_sequence_sum p n - n * p^(n + 1) / (1 - p)

theorem general_term_an (a : ℕ → ℕ) (Sn : ℤ → ℚ) (n : ℕ) :
  (a 1 = 1) →
  ∀ n : ℕ, Sn (2 * n) / Sn n = ((4 * n + 2) : ℚ) / (n + 1) →
  a n = n :=
sorry

theorem sum_bn (a : ℕ → ℕ) (b : ℕ → ℝ) (Tn : ℕ → ℝ) (p : ℝ) (n : ℕ) :
  (a 1 = 1) →
  ∀ n : ℕ, Tn n = if p = 1 then Tn_p_eq_1 n else Tn_p_neq_1 p n →
  b n = a n * p^(a n) →
  p > 0 →
  Tn n = sum_list (λ i, b i) n :=
sorry

end general_term_an_sum_bn_l681_681748


namespace max_value_expr_l681_681781

variable (a b : ℝ)
variable (a_pos : 0 < a) (b_pos : 0 < b)

theorem max_value_expr (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) :
  ∃ x : ℝ, (3(a - x)*(2*x + real.sqrt(x^2 + 4*b^2)) ≤ 3*a^2 + 12*b^2) :=
sorry

end max_value_expr_l681_681781


namespace sum_of_cubes_of_even_numbers_l681_681419

theorem sum_of_cubes_of_even_numbers :
  (∑ k in finset.range 20, (2 * (k + 1)) ^ 3) = 352800 :=
by
  sorry

end sum_of_cubes_of_even_numbers_l681_681419


namespace hyperbola_eccentricity_l681_681666

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (h_a : a > 0) (h_b : b > 0) : ℝ :=
let c := real.sqrt (a^2 + b^2) in c / a

theorem hyperbola_eccentricity
  (a b : ℝ)
  (h_a : a > 0)
  (h_b : b > 0)
  (h_hyperbola : ∀ (x y : ℝ), (x, y) = (4, 2) ∨ (x, y) = (2, 0) ∨ (x, y) = (-4, 3) ∨ (x, y) = (4, 3) → ((x^2 / a^2) - (y^2 / b^2) = 1))
  (h_point_count : (if (4^2 / a^2) - (2^2 / b^2) = 1 then 1 else 0) + (if (2^2 / a^2) - (0^2 / b^2) = 1 then 1 else 0) + (if ((-4)^2 / a^2) - (3^2 / b^2) = 1 then 1 else 0) + (if (4^2 / a^2) - (3^2 / b^2) = 1 then 1 else 0) = 3)
  : eccentricity_of_hyperbola a b h_a h_b = real.sqrt 7 / 2 :=
sorry

end hyperbola_eccentricity_l681_681666


namespace right_triangle_counterexample_l681_681933

def is_acute_angle (α : ℝ) : Prop := 0 < α ∧ α < 90

def is_right_angle (α : ℝ) : Prop := α = 90

def is_triangle (α β γ : ℝ) : Prop := α + β + γ = 180

def is_acute_triangle (α β γ : ℝ) : Prop := is_acute_angle α ∧ is_acute_angle β ∧ is_acute_angle γ

def is_right_triangle (α β γ : ℝ) : Prop := 
  (is_right_angle α ∧ is_acute_angle β ∧ is_acute_angle γ) ∨ 
  (is_acute_angle α ∧ is_right_angle β ∧ is_acute_angle γ) ∨ 
  (is_acute_angle α ∧ is_acute_angle β ∧ is_right_angle γ)

theorem right_triangle_counterexample (α β γ : ℝ) : 
  is_triangle α β γ → is_right_triangle α β γ → ¬ is_acute_triangle α β γ :=
by
  intro htri hrt hacute
  sorry

end right_triangle_counterexample_l681_681933


namespace range_of_x_satisfying_inequality_l681_681873

theorem range_of_x_satisfying_inequality (x : ℝ) : 
  (|x+1| + |x| < 2) ↔ (-3/2 < x ∧ x < 1/2) :=
by sorry

end range_of_x_satisfying_inequality_l681_681873


namespace number_of_triangles_with_perimeter_11_l681_681687

theorem number_of_triangles_with_perimeter_11 : (∃ triangles : List (ℕ × ℕ × ℕ), 
  (∀ t ∈ triangles, let (a, b, c) := t in 
    a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ a + c > b) 
  ∧ triangles.length = 10) := 
sorry

end number_of_triangles_with_perimeter_11_l681_681687


namespace geometric_product_geometric_quotient_l681_681637

def is_geometric_sequence (s : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, s (n + 1) = q * s n

variables {a b : ℕ → ℝ} {q1 q2 : ℝ}

-- Assume {a_n} and {b_n} are geometric sequences with ratios q1 and q2, respectively
axiom geom_a : is_geometric_sequence a q1
axiom geom_b : is_geometric_sequence b q2
axiom nonzero_a : ∀ n, a n ≠ 0
axiom nonzero_b : ∀ n, b n ≠ 0

-- Prove that {a_n * b_n} is a geometric sequence
theorem geometric_product : is_geometric_sequence (λ n, a n * b n) (q1 * q2) := sorry

-- Prove that {a_n / b_n} is a geometric sequence
theorem geometric_quotient : is_geometric_sequence (λ n, a n / b n) (q1 / q2) := sorry

end geometric_product_geometric_quotient_l681_681637


namespace arithmetic_sequence_sum_l681_681139

-- Define the arithmetic sequence properties
def is_arithmetic_sequence (seq : ℕ → ℕ) :=
  ∀ n : ℕ, seq (n + 1) = seq n + 2

-- Define the arithmetic sequence in question
def sequence : ℕ → ℕ
| 0       := 2
| (n + 1) := sequence n + 2

-- Check that our sequence matches the properties of an arithmetic sequence
lemma sequence_is_arithmetic : is_arithmetic_sequence sequence :=
by intros n; simp [sequence]

-- Define the sum of the first n terms of the sequence
def sum_n_terms (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, sequence i

-- State the main theorem to be proven: the sum of the first 10 terms is 110
theorem arithmetic_sequence_sum : sum_n_terms 10 = 110 :=
sorry

end arithmetic_sequence_sum_l681_681139


namespace complement_of_A_within_U_l681_681676

open set

variable {U : set ℝ} {A : set ℝ}

def U : set ℝ := { x | -3 < x ∧ x < 3 }
def A : set ℝ := { x | 0 < x ∧ x < 2 }

theorem complement_of_A_within_U :
  (U \ A) = ({ x | -3 < x ∧ x ≤ 0 } ∪ { x | 2 ≤ x ∧ x < 3 }) :=
by sorry

end complement_of_A_within_U_l681_681676


namespace percentage_runs_by_running_l681_681949

theorem percentage_runs_by_running 
  (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) 
  (runs_per_boundary : ℕ) (runs_per_six : ℕ)
  (H_total_runs : total_runs = 120)
  (H_boundaries : boundaries = 3)
  (H_sixes : sixes = 8)
  (H_runs_per_boundary : runs_per_boundary = 4)
  (H_runs_per_six : runs_per_six = 6) :
  ((total_runs - (boundaries * runs_per_boundary + sixes * runs_per_six)) / total_runs : ℚ) * 100 = 50 := 
by
  sorry

end percentage_runs_by_running_l681_681949


namespace probability_of_winning_l681_681448

noncomputable def probability_winning_prize (n_types n_bags : ℕ) : ℝ :=
let total_combinations := n_types ^ n_bags in
let not_winning_combinations := nat.choose n_types 2 * n_types ^ (n_bags - 1) - n_types in
let probability_not_winning := not_winning_combinations / total_combinations in
1 - probability_not_winning

theorem probability_of_winning :
  probability_winning_prize 3 4 = 4 / 9 := 
sorry

end probability_of_winning_l681_681448


namespace arithmetic_sequence_sum_l681_681130

theorem arithmetic_sequence_sum :
  let sequence := list.range (20 / 2) in
  let sum := sequence.map (λ n, 2 * (n + 1)).sum in
  sum = 110 :=
by
  -- Define the sequence as the arithmetic series
  let sequence := list.range (20 / 2)
  -- Calculate the sum of the arithmetic sequence
  let sum := sequence.map (λ n, 2 * (n + 1)).sum
  -- Check the sum
  have : sum = 110 := sorry
  exact this

end arithmetic_sequence_sum_l681_681130


namespace football_team_birthday_collision_moscow_birthday_collision_l681_681045

theorem football_team_birthday_collision (n : ℕ) (k : ℕ) (h1 : n ≥ 11) (h2 : k = 7) : 
  ∃ (d : ℕ) (p1 p2 : ℕ), p1 ≠ p2 ∧ p1 ≤ n ∧ p2 ≤ n ∧ d ≤ k :=
by sorry

theorem moscow_birthday_collision (population : ℕ) (days : ℕ) (h1 : population > 10000000) (h2 : days = 366) :
  ∃ (day : ℕ) (count : ℕ), count ≥ 10000 ∧ count ≤ population / days :=
by sorry

end football_team_birthday_collision_moscow_birthday_collision_l681_681045


namespace floor_of_47_l681_681559

theorem floor_of_47 : int.floor 4.7 = 4 :=
sorry

end floor_of_47_l681_681559


namespace beginning_games_approx_12_l681_681089

variable (G B : ℝ)
variable (hb1 : 0.40 * B + 0.80 * (G - B) = 0.50 * G)
variable (hg : G = 40)

theorem beginning_games_approx_12 (hb1 : hb1) (hg : hg) : B = 12 := by
  sorry

end beginning_games_approx_12_l681_681089


namespace tangent_simplification_l681_681835

-- Definitions from conditions
def tangent_addition (A B : ℝ) : ℝ := (Real.tan A + Real.tan B) / (1 - Real.tan A * Real.tan B)
def tan_45 : Real := 1
def A : ℝ := Real.pi / 9
def B : ℝ := 5 * Real.pi / 36

-- The main theorem to be proven
theorem tangent_simplification : (1 + Real.tan A) * (1 + Real.tan B) = 2 :=
by
  -- Use the given conditions to derive the required result.
  sorry

end tangent_simplification_l681_681835


namespace distance_between_intersections_l681_681577

open Real

theorem distance_between_intersections :
  (∀ x y : ℝ, x^2 + y = 12 ∧ x + y = 8 → 
    sqrt ((x - (8 - x)) ^ 2 + (y - (12 - x^2)) ^ 2) = sqrt 34) :=
begin
  -- Proof of the theorem would go here
  sorry
end

end distance_between_intersections_l681_681577


namespace fractional_inequality_solution_l681_681390

theorem fractional_inequality_solution (x : ℝ) :
  (x - 2) / (x + 1) < 0 ↔ -1 < x ∧ x < 2 :=
sorry

end fractional_inequality_solution_l681_681390


namespace g_nonneg_diff_l681_681363

open Real

-- Define the conditions: f and g are differentiable on [0, 1]
variables {f g : ℝ → ℝ}
hypothesis (hf : ∀ x ∈ Icc 0 1, DifferentiableAt ℝ f x)
hypothesis (hg : ∀ x ∈ Icc 0 1, DifferentiableAt ℝ g x)

-- Define the boundary conditions: f(0) = 1 and f(1) = 1
hypothesis (h_f_0 : f 0 = 1)
hypothesis (h_f_1 : f 1 = 1)

-- Define the nonnegative condition: 19 f' g + 93 f g' ≥ 0
hypothesis (h_nonneg : ∀ x ∈ Icc 0 1, 19 * (derivative f x) * (g x) + 93 * (f x) * (derivative g x) ≥ 0)

-- Prove that: g(1) ≥ g(0)
theorem g_nonneg_diff : g 1 ≥ g 0 :=
by
  sorry

end g_nonneg_diff_l681_681363


namespace Amy_earnings_l681_681478

theorem Amy_earnings :
  (1.5 + 40 / 60 + (11.5 - 9.25)) * 4 = 18 :=
by
  -- Substitute 11.5 - 9.25 with 2.25
  have h1 : 11.5 - 9.25 = 2.25 := by norm_num
  -- Sum the hours worked each day
  have h2 : 1.5 + 40 / 60 + 2.25 = 4.4166667 := by norm_num
  -- Calculate the earnings
  have h3 : 4.4166667 * 4 = 17.6666668 := by norm_num
  -- Round the earnings to the nearest dollar
  have h4 : round 17.6666668 = 18 := by norm_num
  -- Complete the proof
  exact h4

end Amy_earnings_l681_681478


namespace total_time_including_break_is_correct_l681_681473

def ally_rate : ℝ := 1 / 3
def bob_rate : ℝ := 1 / 4
def total_task : ℝ := 1
def initial_work_time : ℝ := 2
def break_time : ℝ := 0.5
def combined_rate : ℝ := ally_rate + bob_rate

theorem total_time_including_break_is_correct (t : ℝ) :
  (t - break_time) * combined_rate = total_task ↔ t = 2.5 :=
by 
  sorry

end total_time_including_break_is_correct_l681_681473


namespace train_length_is_120_meters_l681_681959

-- Define the given conditions
def jogger_speed_kmph : ℝ := 9
def jogger_head_start_m : ℝ := 240
def train_speed_kmph : ℝ := 45
def passing_time_sec : ℝ := 36

-- Speed conversion from km/h to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := (speed_kmph * 1000) / 3600

def jogger_speed_mps : ℝ := kmph_to_mps jogger_speed_kmph
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- Calculate the relative speed of the train with respect to the jogger
def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps

-- Calculate the distance the train travels in the given time (in meters)
def distance_traveled_m : ℝ := relative_speed_mps * passing_time_sec

-- Calculate the length of the train
def train_length : ℝ := distance_traveled_m - jogger_head_start_m

-- The proof statement
theorem train_length_is_120_meters : train_length = 120 := by
  sorry

end train_length_is_120_meters_l681_681959


namespace kubik_family_arrangements_l681_681486

theorem kubik_family_arrangements (n : ℕ) (h_n : n = 7) :
  let total_arrangements := (n - 1)!
  let invalid_arrangements := 2 * (n - 2)!
  let valid_arrangements := total_arrangements - invalid_arrangements
  valid_arrangements = 480 :=
by
  sorry

end kubik_family_arrangements_l681_681486


namespace grandparents_to_parents_ratio_l681_681109

-- Definitions corresponding to the conditions
def wallet_cost : ℕ := 100
def betty_half_money : ℕ := wallet_cost / 2
def parents_contribution : ℕ := 15
def betty_needs_more : ℕ := 5
def grandparents_contribution : ℕ := 95 - (betty_half_money + parents_contribution)

-- The mathematical statement for the proof
theorem grandparents_to_parents_ratio :
  grandparents_contribution / parents_contribution = 2 := by
  sorry

end grandparents_to_parents_ratio_l681_681109


namespace union_complement_eq_set_l681_681315

-- Define the universal set I
def I : Set ℤ := {x | -3 < x ∧ x < 3}

-- Define set A and set B
def A := {1, 2}
def B := {-2, -1, 2}

-- Define the complement of B in I
def complement_I_B := I \ B

-- Prove that A ∪ (complement_I_B) equals {0,1,2}
theorem union_complement_eq_set :
  A ∪ (complement_I_B) = {0, 1, 2} := by
  sorry

end union_complement_eq_set_l681_681315


namespace binom_18_6_eq_13260_l681_681516

/-- The binomial coefficient formula. -/
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The specific proof problem: compute binom(18, 6) and show that it equals 13260. -/
theorem binom_18_6_eq_13260 : binom 18 6 = 13260 :=
by
  sorry

end binom_18_6_eq_13260_l681_681516


namespace meet_at_eleven_pm_l681_681043

noncomputable def meet_time (start_time hours : ℝ) : ℝ := start_time + hours

theorem meet_at_eleven_pm
  (start_time : ℝ) (distance speedA speedB : ℝ)
  (h_start : start_time = 18)  -- 6 PM in 24-hour format
  (h_distance : distance = 50)
  (h_speedA : speedA = 6)
  (h_speedB : speedB = 4) :
  meet_time start_time (distance / (speedA + speedB)) = 23 :=
by
  have h_relative_speed : speedA + speedB = 10 := by
    rw [h_speedA, h_speedB]
  have h_time_to_meet : distance / (speedA + speedB) = 5 := by
    rw [h_distance, h_relative_speed]
    norm_num
  rw [h_start, h_time_to_meet]
  norm_num
  sorry

end meet_at_eleven_pm_l681_681043


namespace proof_range_of_a_l681_681780

noncomputable def range_of_a (a : ℝ) := a ≥ 2

theorem proof_range_of_a (a : ℝ) (h : a > 1)
  (hxy : ∀ x ∈ set.Icc a (2 * a), ∃ y ∈ set.Icc a (a^2), real.logb a x + real.logb a y = 3) :
  range_of_a a := 
sorry

end proof_range_of_a_l681_681780


namespace greatest_3digit_base8_divisible_by_7_l681_681899

def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : ℕ), n = 0b777 ∧ (base8_to_base10 0b777) % 7 = 0 ∧ ∀ m < 0o777, m % 7 = 0 → base8_to_base10 m < base8_to_base10 0b777 :=
by
  sorry

end greatest_3digit_base8_divisible_by_7_l681_681899


namespace complex_magnitude_l681_681606

-- Define the complex number 'z' and the condition it satisfies
variables (z : ℂ)
hypothesis hz : z / (2 + I) = I^2015 + I^2016

-- The proof statement
theorem complex_magnitude (hz : z / (2 + I) = I^2015 + I^2016) :
  |z| = Real.sqrt 10 :=
sorry

end complex_magnitude_l681_681606


namespace sum_arithmetic_seq_l681_681136

theorem sum_arithmetic_seq (a d n : ℕ) :
  a = 2 → d = 2 → a + (n - 1) * d = 20 → (n / 2) * (a + (a + (n - 1) * d)) = 110 :=
by sorry

end sum_arithmetic_seq_l681_681136


namespace degrees_to_radians_l681_681537

theorem degrees_to_radians (degrees : ℝ) (pi : ℝ) : 
  degrees * (pi / 180) = pi / 15 ↔ degrees = 12 :=
by 
  sorry

end degrees_to_radians_l681_681537


namespace combination_18_6_l681_681521

theorem combination_18_6 : nat.choose 18 6 = 18564 :=
by {
  sorry
}

end combination_18_6_l681_681521


namespace positive_integers_sum_reciprocal_l681_681393

theorem positive_integers_sum_reciprocal (a b c : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) 
  (h_sum : a + b + c = 2010) (h_recip : (1 : ℚ)/a + (1 : ℚ)/b + (1 : ℚ)/c = 1/58) :
  (a = 1740 ∧ b = 180 ∧ c = 90) ∨ 
  (a = 1740 ∧ b = 90 ∧ c = 180) ∨ 
  (a = 180 ∧ b = 90 ∧ c = 1740) ∨ 
  (a = 180 ∧ b = 1740 ∧ c = 90) ∨ 
  (a = 90 ∧ b = 1740 ∧ c = 180) ∨ 
  (a = 90 ∧ b = 180 ∧ c = 1740) := 
sorry

end positive_integers_sum_reciprocal_l681_681393


namespace trigonometric_identity_l681_681346

variable (B : Real)

def cot (x : Real) := (cos x) / (sin x)
def csc (x : Real) := 1 / (sin x)
def tan (x : Real) := (sin x) / (cos x)
def sec (x : Real) := 1 / (cos x)

theorem trigonometric_identity (B : Real) :
  (1 + cot B ^ 2 - csc B) * (1 + tan B ^ 2 + sec B) = 1 :=
  sorry

end trigonometric_identity_l681_681346


namespace gasoline_price_increase_l681_681860

theorem gasoline_price_increase (highest_price lowest_price : ℝ) (h1 : highest_price = 24) (h2 : lowest_price = 15) : 
  ((highest_price - lowest_price) / lowest_price) * 100 = 60 :=
by
  sorry

end gasoline_price_increase_l681_681860


namespace triangle_area_l681_681848

-- Define the line equation as a condition.
def line_equation (x : ℝ) : ℝ :=
  4 * x + 8

-- Define the y-intercept (condition 1).
def y_intercept := line_equation 0

-- Define the x-intercept (condition 2).
def x_intercept := (-8) / 4

-- Define the area of the triangle given the intercepts and prove it equals 8 (question and correct answer).
theorem triangle_area :
  (1 / 2) * abs x_intercept * y_intercept = 8 :=
by
  sorry

end triangle_area_l681_681848


namespace cannot_reach_right_grid_l681_681741

structure Grid5x5 :=
  (cell : Fin 5 × Fin 5 → Bool)

def num_shaded_cells_per_column (g : Grid5x5) (j : Fin 5) : Nat :=
  Finset.card (Finset.filter (fun i => g.cell (i, j)) (Finset.univ : Finset (Fin 5)))

def num_shaded_cells_constraints (g : Grid5x5) : Prop :=
  ∃ j1 j2 j3 j4 j5 : Fin 5,
    j1 ≠ j2 ∧ j1 ≠ j3 ∧ j1 ≠ j4 ∧ j1 ≠ j5 ∧
    j2 ≠ j3 ∧ j2 ≠ j4 ∧ j2 ≠ j5 ∧
    j3 ≠ j4 ∧ j3 ≠ j5 ∧
    j4 ≠ j5 ∧
    num_shaded_cells_per_column g j1 = 4 ∧
    num_shaded_cells_per_column g j2 = 3 ∧
    num_shaded_cells_per_column g j3 = 3 ∧
    num_shaded_cells_per_column g j4 = 3 ∧
    num_shaded_cells_per_column g j5 = 2

theorem cannot_reach_right_grid (left_grid right_grid : Grid5x5) :
  num_shaded_cells_constraints left_grid →
  (¬ num_shaded_cells_constraints right_grid) →
  ¬ ∃ (swap_sequence : List (Fin 5 × Fin 5))
    (g : Grid5x5) (H : g = left_grid),
    ∀ swap_item ∈ swap_sequence,
      (swap_item.fst < 5) ∧ (swap_item.snd < 5) /\
      -- Assuming swap logic is defined
      ∃ g', swap g swap_item.fst swap_item.snd = g' ∧ g' = right_grid := sorry

end cannot_reach_right_grid_l681_681741


namespace half_abs_diff_squares_l681_681022

theorem half_abs_diff_squares (a b : ℕ) (ha : a = 15) (hb : b = 13) : (abs (a^2 - b^2)) / 2 = 28 := by
  sorry

end half_abs_diff_squares_l681_681022


namespace sandro_children_ratio_l681_681831

theorem sandro_children_ratio (d : ℕ) (h1 : d + 3 = 21) : d / 3 = 6 :=
by
  sorry

end sandro_children_ratio_l681_681831


namespace min_b1_b2_sum_l681_681005

theorem min_b1_b2_sum {b : ℕ → ℕ}
  (h_seq : ∀ n ≥ 1, b (n + 2) = (b n + 1024) / (1 + 2 * b (n + 1)))
  (h_pos : ∀ n, b n > 0) :
  ∃ b1 b2 : ℕ, b1 = b 1 ∧ b2 = b 2 ∧ b1 + b2 = 48 :=
begin
  -- remaining is proof which is not required
  sorry
end

end min_b1_b2_sum_l681_681005


namespace speed_of_second_train_l681_681468

/-- 
Given:
1. A train leaves Mumbai at 9 am at a speed of 40 kmph.
2. After one hour, another train leaves Mumbai in the same direction at an unknown speed.
3. The two trains meet at a distance of 80 km from Mumbai.

Prove that the speed of the second train is 80 kmph.
-/
theorem speed_of_second_train (v : ℝ) :
  (∃ (distance_first : ℝ) (distance_meet : ℝ) (initial_speed_first : ℝ) (hours_later : ℤ),
    distance_first = 40 ∧ distance_meet = 80 ∧ initial_speed_first = 40 ∧ hours_later = 1 ∧
    v = distance_meet / (distance_meet / initial_speed_first - hours_later)) → v = 80 := by
  sorry

end speed_of_second_train_l681_681468


namespace binom_18_6_eq_18564_l681_681505

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binom_18_6_eq_18564 : binomial 18 6 = 18564 := by
  sorry

end binom_18_6_eq_18564_l681_681505


namespace max_value_of_expression_l681_681783

theorem max_value_of_expression (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (h : a^2 + b^2 + c^2 = 1) :
  3 * a * b * real.sqrt 3 + 9 * b * c ≤ 3 :=
sorry

end max_value_of_expression_l681_681783


namespace complex_z_solution_l681_681207

noncomputable def complex_number_z : ℂ :=
  let is_purely_imaginary (w : ℂ) : Prop := w.re = 0
  let meets_conditions (z : ℂ) : Prop := 
    abs z = sqrt 13 ∧ is_purely_imaginary ((2 + 3 * complex.I) * z)
  if meets_conditions (3 + 2 * complex.I) then
    (3 + 2 * complex.I)
  else if meets_conditions (-3 - 2 * complex.I) then
    (-3 - 2 * complex.I)
  else
    (0 : ℂ)

theorem complex_z_solution : complex_number_z = 3 + 2 * complex.I ∨ complex_number_z = -3 - 2 * complex.I :=
sorry

end complex_z_solution_l681_681207


namespace find_a_range_l681_681654

noncomputable def f (x a : ℝ) := 2 ^ (x * (x - a))

theorem find_a_range (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → deriv (λ x, f x a) x < 0) → 2 ≤ a :=
by sorry

end find_a_range_l681_681654


namespace inequality_x_y_z_l681_681333

theorem inequality_x_y_z (x y z : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) :
  x * (1 - y) + y * (1 - z) + z * (1 - x) < 1 := 
by sorry

end inequality_x_y_z_l681_681333


namespace multiply_3_6_and_0_3_l681_681188

theorem multiply_3_6_and_0_3 : 3.6 * 0.3 = 1.08 :=
by
  sorry

end multiply_3_6_and_0_3_l681_681188


namespace binom_18_6_eq_13260_l681_681520

/-- The binomial coefficient formula. -/
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The specific proof problem: compute binom(18, 6) and show that it equals 13260. -/
theorem binom_18_6_eq_13260 : binom 18 6 = 13260 :=
by
  sorry

end binom_18_6_eq_13260_l681_681520


namespace car_travel_distance_l681_681065

theorem car_travel_distance:
  (∃ r, r = 3 / 4 ∧ ∀ t, t = 2 → ((r * 60) * t = 90)) :=
by
  sorry

end car_travel_distance_l681_681065


namespace measure_angle_C_sides_a_b_l681_681614

namespace TriangleProblem

noncomputable def angle_C : ℝ := π / 3
noncomputable def side_a : ℝ := 3 * Real.sqrt(5) / 5
noncomputable def side_b : ℝ := 6 * Real.sqrt(5) / 5

variables {A B C a b c : ℝ}

-- First part: Prove angle C
theorem measure_angle_C (h1 : c = 3) 
  (h2 : sin (C - π / 6) * cos C = 1 / 4) :
  C = π / 3 :=
by
  sorry

-- Second part: Prove sides a and b
theorem sides_a_b (h1 : c = 3)
  (h2 : sin (C - π / 6) * cos C = 1 / 4)
  (h3 : (1, sin A) = (2, sin B)) :
  a = 3 * Real.sqrt(5) / 5 ∧ b = 6 * Real.sqrt(5) / 5 :=
by
  sorry

end TriangleProblem

end measure_angle_C_sides_a_b_l681_681614


namespace binom_18_6_eq_4765_l681_681527

def binom (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_18_6_eq_4765 : binom 18 6 = 4765 := by
  sorry

end binom_18_6_eq_4765_l681_681527


namespace greatest_3_digit_base_8_divisible_by_7_l681_681912

open Nat

def is_3_digit_base_8 (n : ℕ) : Prop := n < 8^3

def is_divisible_by_7 (n : ℕ) : Prop := 7 ∣ n

theorem greatest_3_digit_base_8_divisible_by_7 :
  ∃ x : ℕ, is_3_digit_base_8 x ∧ is_divisible_by_7 x ∧ x = 7 * (8 * (8 * 7 + 7) + 7) :=
by
  sorry

end greatest_3_digit_base_8_divisible_by_7_l681_681912


namespace floor_of_47_l681_681562

theorem floor_of_47 : int.floor 4.7 = 4 :=
sorry

end floor_of_47_l681_681562


namespace distance_between_trees_l681_681543

theorem distance_between_trees (d : ℝ) (h : d = 80) : 
  let tree_intervals := 7 in 
  let interval_distance := d / 3 in 
  let total_distance := tree_intervals * interval_distance in
  total_distance = 560 / 3 := 
by
  -- Explanation and calculations would be here
  sorry

end distance_between_trees_l681_681543


namespace floor_47_l681_681556

theorem floor_47 : Int.floor 4.7 = 4 :=
by
  sorry

end floor_47_l681_681556


namespace alpha_solution_l681_681779

noncomputable def α_values : set ℂ :=
  {α | α ≠ Complex.I ∧ α ≠ -Complex.I ∧
       Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1) ∧
       Complex.abs (α^4 - 1)^2 = 9 * Complex.abs (α - 1)^2}

theorem alpha_solution :
  ∀ α : ℂ, α ∈ α_values → α = (1 / 2) + Complex.I * (Real.sqrt 35 / 2) ∨ α = (1 / 2) - Complex.I * (Real.sqrt 35 / 2) :=
by
  intros α h
  sorry

end alpha_solution_l681_681779


namespace f_2011_l681_681643

-- Define f(x)
def f (x : ℝ) (a α b β : ℝ) : ℝ := a * Real.sin(π * x + α) + b * Real.cos(π * x + β)

-- Given f(2009) = 3
axiom f_2009_is_3 (a α b β : ℝ) : f 2009 a α b β = 3

-- Prove that f(2011) = 3
theorem f_2011 (a α b β : ℝ) : f 2011 a α b β = 3 :=
by
  -- We need to show that f(2011) = 3 using the given conditions
  have h₁ : Real.sin (π * (2009:ℝ) + α) = -Real.sin α := by sorry
  have h₂ : Real.cos (π * (2009:ℝ) + β) = -Real.cos β := by sorry
  have h₃ : Real.sin (π * (2011:ℝ) + α) = -Real.sin α := by sorry
  have h₄ : Real.cos (π * (2011:ℝ) + β) = -Real.cos β := by sorry
  calc
    f 2009 a α b β = -a * Real.sin α - b * Real.cos β := by rw [f, h₁, h₂]
    ... = 3 := f_2009_is_3 a α b β
    f 2011 a α b β = -a * Real.sin α - b * Real.cos β := by rw [f, h₃, h₄]
    ... = 3 := by assumption

end f_2011_l681_681643


namespace calculate_gas_cost_l681_681682

theorem calculate_gas_cost :
  ∀ (odometer_start odometer_end : ℕ) (fuel_efficiency : ℝ) (price_per_gallon : ℝ),
    odometer_start = 32150 →
    odometer_end = 32178 →
    fuel_efficiency = 25 →
    price_per_gallon = 3.75 →
    (let distance := odometer_end - odometer_start in
     let gallons_used := distance / fuel_efficiency in
     let cost := gallons_used * price_per_gallon in
     Float.round (cost * 100) / 100 = 4.20) :=
begin
  intros,
  sorry
end

end calculate_gas_cost_l681_681682


namespace bob_correct_answers_l681_681736

-- Define the variables, c for correct answers, w for incorrect answers, total problems 15, score 54
variables (c w : ℕ)

-- Define the conditions
axiom total_problems : c + w = 15
axiom total_score : 6 * c - 3 * w = 54

-- Prove that the number of correct answers is 11
theorem bob_correct_answers : c = 11 :=
by
  -- Here, you would provide the proof, but for the sake of the statement, we'll use sorry.
  sorry

end bob_correct_answers_l681_681736


namespace product_closest_to_127_l681_681871

-- Define the expression as a condition
def expr := 2.4 * (53.2 + 0.25)

-- Possible options
def options := {120, 127, 130, 135, 140}

theorem product_closest_to_127 : (abs (expr - 127) < abs (expr - 120)) ∧
                                (abs (expr - 127) < abs (expr - 130)) ∧
                                (abs (expr - 127) < abs (expr - 135)) ∧
                                (abs (expr - 127) < abs (expr - 140)) :=
by
  sorry

end product_closest_to_127_l681_681871


namespace possible_multisets_count_l681_681173

theorem possible_multisets_count {b : Fin 9 → ℤ} (b_8_nonzero : b 0 ≠ 0) (b_0_nonzero : b 8 ≠ 0) :
  ∃ S : Multiset ℤ, (∀ r ∈ S, (r = 1 ∨ r = -1)) ∧ (S.card = 8) ∧ (∃ k : ℕ, 0 ≤ k ∧ k ≤ 8 ∧ k = S.count 1) ∧ (∃ n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ (8 - k = S.count (-1))) :=
sorry

end possible_multisets_count_l681_681173


namespace log_sum_equals_619_l681_681352

theorem log_sum_equals_619 (b : ℝ) (h1 : b > 1) (h2 : log 5 (log 5 b + log b 125) = 2) :
  log 5 (b ^ log 5 b) + log b (125 ^ log b 125) = 619 := 
sorry

end log_sum_equals_619_l681_681352


namespace inequality_solution_l681_681843

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (3 * x + 4) < 5 ↔ x < -4 / 3 ∨ -3 / 2 < x := 
sorry

end inequality_solution_l681_681843


namespace A_takes_200_seconds_l681_681044

/-- 
  A can give B a start of 50 meters or 10 seconds in a kilometer race.
  How long does A take to complete the race?
-/
theorem A_takes_200_seconds (v_A : ℝ) (distance : ℝ) (start_meters : ℝ) (start_seconds : ℝ) :
  (start_meters = 50) ∧ (start_seconds = 10) ∧ (distance = 1000) ∧ 
  (v_A = start_meters / start_seconds) → distance / v_A = 200 :=
by
  sorry

end A_takes_200_seconds_l681_681044


namespace initial_liquid_X_percentage_is_30_l681_681974

variable (initial_liquid_X_percentage : ℝ)

theorem initial_liquid_X_percentage_is_30
  (solution_total_weight : ℝ := 8)
  (initial_water_percentage : ℝ := 70)
  (evaporated_water_weight : ℝ := 3)
  (added_solution_weight : ℝ := 3)
  (new_liquid_X_percentage : ℝ := 41.25)
  (total_new_solution_weight : ℝ := 8)
  :
  initial_liquid_X_percentage = 30 :=
sorry

end initial_liquid_X_percentage_is_30_l681_681974


namespace solve_inequality_l681_681841

theorem solve_inequality (x : ℝ) : 
  3 - (1 / (3 * x + 4)) < 5 ↔ x ∈ set.Ioo (-∞ : ℝ) (-7 / 6) ∪ set.Ioo (-4 / 3) ∞ := 
sorry

end solve_inequality_l681_681841


namespace percent_increase_is_30_l681_681410

-- Defining the conditions
def sales_this_year : ℝ := 416
def sales_last_year : ℝ := 320

-- Define the percent increase calculation
def percent_increase (sales_this_year sales_last_year : ℝ) : ℝ :=
  ((sales_this_year - sales_last_year) / sales_last_year) * 100

-- Theorem stating the percent increase in sales is 30%
theorem percent_increase_is_30 :
  percent_increase sales_this_year sales_last_year = 30 :=
by
  sorry

end percent_increase_is_30_l681_681410


namespace f_properties_l681_681449

noncomputable def f : ℝ → ℝ := sorry

axiom f_eq (x y : ℝ) : f(x + y) + f(x - y) = 2 * f(x) * f(y)
axiom f_nonzero : f(0) ≠ 0

theorem f_properties :
    f(0) = 1 ∧
    (∀ x, f(-x) = f(x)) ∧ 
    (∃ c, ∀ x, f(x + 2 * c) = f(x)) :=
sorry

end f_properties_l681_681449


namespace number_of_trousers_given_l681_681815

-- Define the conditions
def shirts_given : Nat := 589
def total_clothing_given : Nat := 934

-- Define the expected answer
def expected_trousers_given : Nat := 345

-- The theorem statement to prove the number of trousers given
theorem number_of_trousers_given : total_clothing_given - shirts_given = expected_trousers_given :=
by
  sorry

end number_of_trousers_given_l681_681815


namespace sum_arithmetic_seq_l681_681138

theorem sum_arithmetic_seq (a d n : ℕ) :
  a = 2 → d = 2 → a + (n - 1) * d = 20 → (n / 2) * (a + (a + (n - 1) * d)) = 110 :=
by sorry

end sum_arithmetic_seq_l681_681138


namespace trapezoid_incircle_center_on_MN_l681_681802

noncomputable def center_of_incircle_lies_on_MN 
  (ABCD : Type) 
  [trapezoid ABCD] 
  (AB CD : Line) 
  [is_parallel AB CD] 
  [AB > CD] 
  (M N : Point) 
  [M_tangent_to_incircle_on AB] 
  [N_tangent_to_incircle_on AC] 
  (O : Point) 
  [O_center_of_incircle ABCD]
  : Prop :=
  collinear {M, N, O}

open trapezoid

theorem trapezoid_incircle_center_on_MN
  (ABCD : Type)
  [t : trapezoid ABCD AB CD hAB_CD hAB_gt_CD]
  (M N : Point)
  (A B C D AB CD) :
  ∃ O, O_center_of_incircle ABCD O ∧ collinear { M, N, O } := sorry

end trapezoid_incircle_center_on_MN_l681_681802


namespace equation_solutions_l681_681408

theorem equation_solutions :
  ∀ x y : ℤ, x^2 + x * y + y^2 + x + y - 5 = 0 → (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -3) ∨ (x = -3 ∧ y = 1) :=
by
  intro x y h
  sorry

end equation_solutions_l681_681408


namespace find_a_range_l681_681657

noncomputable def f (x a : ℝ) := 2 ^ (x * (x - a))

theorem find_a_range (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → deriv (λ x, f x a) x < 0) → 2 ≤ a :=
by sorry

end find_a_range_l681_681657


namespace pyramid_height_l681_681442

theorem pyramid_height (h : ℝ) :
  let V_cube := 5^3
  let V_pyramid := (1/3) * 10^2 * h
  V_cube = V_pyramid → h = 3.75 :=
by
  let V_cube := 5^3
  let V_pyramid := (1/3) * 10^2 * h
  intros h_eq
  sorry

end pyramid_height_l681_681442


namespace average_of_eight_twelve_and_N_is_12_l681_681865

theorem average_of_eight_twelve_and_N_is_12 (N : ℝ) (hN : 11 < N ∧ N < 19) : (8 + 12 + N) / 3 = 12 :=
by
  -- Place the complete proof step here
  sorry

end average_of_eight_twelve_and_N_is_12_l681_681865


namespace alice_password_prob_l681_681990

/-- 
Alice's password conditions:
1. The password consists of a non-negative single-digit number, a letter, and another non-negative single-digit number, which is different from the first.
2. Even numbers in the set of non-negative single-digit numbers are {0, 2, 4, 6, 8}.
3. Positive numbers in the set of non-negative single-digit numbers are {1, 2, 3, 4, 5, 6, 7, 8, 9}.
-/
def alice_password_prob_correct : Prop :=
  let even_numbers := {0, 2, 4, 6, 8}
  let positive_numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let prob_first_even := 5 / 10
  let prob_last_positive_different := (1 / 5 * 1) + (4 / 5 * 8 / 9) 
  prob_first_even * 1 * prob_last_positive_different = 41 / 90

theorem alice_password_prob : alice_password_prob_correct :=
  sorry

end alice_password_prob_l681_681990


namespace minimize_sum_of_digits_l681_681777

def f (n : ℕ) : ℕ := n^2 - 69 * n + 2250

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def find_minimizing_prime : ℕ :=
  3

theorem minimize_sum_of_digits :
  ∀ (p : ℕ) (H : Nat.Prime p),
    sum_of_digits (f (p^2 + 32)) ≥ sum_of_digits (f (3^2 + 32)) :=
by
  sorry

end minimize_sum_of_digits_l681_681777


namespace ram_krish_together_time_l681_681824

theorem ram_krish_together_time : 
  let t_R := 36
  let t_K := t_R / 2
  let task_per_day_R := 1 / t_R
  let task_per_day_K := 1 / t_K
  let task_per_day_together := task_per_day_R + task_per_day_K
  let T := 1 / task_per_day_together
  T = 12 := 
by
  sorry

end ram_krish_together_time_l681_681824


namespace uncovered_side_length_l681_681457

theorem uncovered_side_length :
  ∃ (L : ℝ) (W : ℝ), L * W = 680 ∧ 2 * W + L = 146 ∧ L = 136 := by
  sorry

end uncovered_side_length_l681_681457


namespace probability_of_two_pairs_and_different_fifth_die_l681_681178

theorem probability_of_two_pairs_and_different_fifth_die : 
  let total_outcomes := 10^5
      choose_pairs := Nat.choose 10 2
      choose_third_number := 8
      arrange_digits := Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)
      successful_outcomes := choose_pairs * choose_third_number * arrange_digits
  in (successful_outcomes / total_outcomes : ℚ) = 0.108 := by
  sorry

end probability_of_two_pairs_and_different_fifth_die_l681_681178


namespace calculate_expression_l681_681495

theorem calculate_expression :
  ( (128^2 - 5^2) / (72^2 - 13^2) * ((72 - 13) * (72 + 13)) / ((128 - 5) * (128 + 5)) * (128 + 5) / (72 + 13) )
  = (133 / 85) :=
by
  -- placeholder for the proof
  sorry

end calculate_expression_l681_681495


namespace coefficients_identity_l681_681706

theorem coefficients_identity :
  let a_0 := 64
  let a_1 := 240
  let a_2 := 300
  let a_3 := 125
  (a_0 + a_2) - (a_1 + a_3) = -1 :=
by
  -- Letting the coefficients be as stated in the problem
  let a_0 := 64
  let a_1 := 240
  let a_2 := 300
  let a_3 := 125
  -- The goal is to prove (a_0 + a_2) - (a_1 + a_3) = -1
  show (a_0 + a_2) - (a_1 + a_3) = -1
  sorry

end coefficients_identity_l681_681706


namespace problem1_problem2_problem3_problem4_l681_681423

-- Problem 1
theorem problem1 (a b : ℝ) (h1 : ∀ x, f(x) = x^3 + a * x^2 + b * x + a^2)
  (h2 : f(1) = 10) (h3 : f' (1) = 0) : (a + b = -7) :=
sorry

-- Problem 2
theorem problem2 : 
  (∑i in range 5, A 8 (i + 1)) / (∑i in range 6, A 9 (i + 1) - 9 * A 9 5) = 5/27 :=
sorry

-- Problem 3
theorem problem3 (perimeter : ℝ) (h1 : perimeter = 20) : 
  let r := (perimeter - 2 * h) / 2 in
  r^2 * h * π = 4000/27 * π :=
sorry

-- Problem 4
theorem problem4 : 
  is_H (λ x, 3 * x - 2 * (sin x - cos x)) ∧ is_H (λ x, e^x + 1) :=
sorry

-- Assuming is_H definition as follows:
def is_H (f : ℝ → ℝ) : Prop :=
∀ x1 x2 : ℝ, x1 ≠ x2 → x1 * f x1 + x2 * f x2 > x1 * f x2 + x2 * f x1

end problem1_problem2_problem3_problem4_l681_681423


namespace james_hours_to_work_l681_681769

theorem james_hours_to_work :
  let meat_cost := 20 * 5
  let fruits_vegetables_cost := 15 * 4
  let bread_cost := 60 * 1.5
  let janitorial_cost := 10 * (10 * 1.5)
  let total_cost := meat_cost + fruits_vegetables_cost + bread_cost + janitorial_cost
  let hourly_wage := 8
  let hours_to_work := total_cost / hourly_wage
  hours_to_work = 50 :=
by 
  sorry

end james_hours_to_work_l681_681769


namespace length_of_generatrix_l681_681631

/-- Given that the base radius of a cone is sqrt(2), and its lateral surface is unfolded into a semicircle,
prove that the length of the generatrix of the cone is 2 sqrt(2). -/
theorem length_of_generatrix (r l : ℝ) (h1 : r = Real.sqrt 2)
    (h2 : 2 * Real.pi * r = Real.pi * l) : l = 2 * Real.sqrt 2 :=
by
  sorry

end length_of_generatrix_l681_681631


namespace ratio_smaller_base_to_altitude_l681_681278

theorem ratio_smaller_base_to_altitude (x : ℝ) (h1 : x > 0) 
  (h2 : ∀ (B L D A : ℝ), B = x → L = 2 * x → D = 3 * x → A = x → L = 2 * B ∧ D = 1.5 * L ∧ A = B) : 
  (∀ (B A : ℝ), B = x → A = x → B / A = 1) := 
by
  intros B A hB hA
  rw [hB, hA]
  apply div_self
  exact ne_of_gt h1
  -- Proof ends here, adding sorry to indicate it.
  sorry

end ratio_smaller_base_to_altitude_l681_681278


namespace cube_distance_sum_l681_681068

theorem cube_distance_sum (A X Y Z : Point) (h_A : closest_to_plane A) 
                          (h_X : adjacent_to A X ∧ height_above_plane X = 10) 
                          (h_Y : adjacent_to A Y ∧ height_above_plane Y = 11) 
                          (h_Z : adjacent_to A Z ∧ height_above_plane Z = 12)
                          (r s t : ℕ) (h_expr : distance_from A to_plane = (r - real.sqrt s) / t) 
                          (h_bound : r + s + t < 1000) : 
                          r + s + t = 330 := 
sorry

end cube_distance_sum_l681_681068


namespace length_of_AB_l681_681754

noncomputable def height (t : Type) [LinearOrderedField t] := sorry
def area (a b h : ℝ) : ℝ := 0.5 * a * h

theorem length_of_AB (AB CD : ℝ) (h : ℝ) (ratio : area AB h / area CD h = 5 / 2) (sum : AB + CD = 280) :
  AB = 200 :=
begin
  sorry
end

end length_of_AB_l681_681754


namespace count_triangles_with_perimeter_11_l681_681695

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem count_triangles_with_perimeter_11 :
  {t : (ℕ × ℕ × ℕ) // let ⟨a, b, c⟩ := t in a + b + c = 11 ∧ is_triangle a b c}.to_finset.card = 9 :=
by sorry

end count_triangles_with_perimeter_11_l681_681695


namespace greatest_base8_three_digit_divisible_by_7_l681_681909

theorem greatest_base8_three_digit_divisible_by_7 :
  ∃ n : ℕ, n < 8^3 ∧ n ≥ 8^2 ∧ (n % 7 = 0) ∧ (to_base 8 n = 777) :=
sorry

end greatest_base8_three_digit_divisible_by_7_l681_681909


namespace proof_part_1_proof_part_2_l681_681611

-- Definitions for given conditions
def seq_a (a : ℕ → ℕ) :=
  ∀ n : ℕ, n ≥ 2 → a n + 2^(n-1) = a (n-1)

def initial_value_a : (a : ℕ → ℕ) → Prop :=
  λ a, a 1 = 2

def c_n (a : ℕ → ℕ) (n : ℕ) :=
  (a n) / 2^n + 1

def seq_b (b : ℕ → ℕ) (a : ℕ → ℕ) :=
  ∀ n : ℕ, ∑ i in finset.range (n+1), b i / (4 - a i) = n

-- Proof of sequence c_n being geometric and formula for a_n
theorem proof_part_1 (a : ℕ → ℕ) (seq_a_cond : seq_a a) (init_a_cond : initial_value_a a)
    (cn_expr : ∀ n, n ≥ 1 → c_n a n = 2 * (1 / 2)^(n-1)) :
  (∀ n : ℕ, n ≥ 1 → c_n a n = (1 / 2)^(n-2)) ∧ (∀ n : ℕ, a n = 4 - 2^n) :=
sorry

-- Proof for sum of first n terms of {1 / b_n} being less than 1
theorem proof_part_2 (a : ℕ → ℕ) (b : ℕ → ℕ) (seq_b_cond : seq_b b a) (init_a_cond : initial_value_a a):
  ∀ n : ℕ, ∑ i in finset.range (n+1), 1 / (b i) < 1 :=
sorry

end proof_part_1_proof_part_2_l681_681611


namespace sequence_remainder_mod_6_l681_681588

/-- Prove that the remainder when 3 * 8 * 13 * 18 * ... * 98 * 103 is divided by 6 is 3. -/
theorem sequence_remainder_mod_6 :
  let seq := λ n : ℕ, 5 * n + 3
  let product := ∏ n in Finset.range 21, seq n
  product % 6 = 3 :=
by
  let seq := λ n : ℕ, 5 * n + 3
  let product := ∏ n in Finset.range 21, seq n
  have hmod : ∀ n, seq n % 6 = 3 := by
    intro n
    simp [seq]
    norm_num [Nat.mod_eq_of_lt]
  let product_mod := ∏ n in Finset.range 21, 3
  simp [hmod]
  exact Nat.mod_eq_of_lt sorry sorry

end sequence_remainder_mod_6_l681_681588


namespace bonferroni_inequalities_l681_681943

theorem bonferroni_inequalities (m r n : ℕ) (Pm Bgtm Sm S : ℕ → ℝ)
  (hEven : ∃ k, r = 2 * k)
  (hSm1 : ∀ m, S m = ∑ r in finset.range (n + 1), if r ≥ m then nat.choose r m * Pm r else 0)
  (hSm2 : ∀ m, S m = ∑ r in finset.range (n + 1), if r ≥ m then nat.choose (r-1) (m-1) * Bgtm r else 0)
  (hPm : ∀ m, Pm m = S m + ∑ k in finset.range (n - m + 1), (-1) ^ k * nat.choose (k + m) k * S (k + m))
  (xi : ℕ → ℝ := λ r, ∑ k in finset.range (n - m + 1), (-1) ^ k * nat.choose (k + m) k * S (k + m)) (hEven_r : ∃ k, r = 2 * k) :
  S m + ∑ k in finset.range (r + 2), (-1) ^ k * nat.choose (m + k) k * S (m + k) ≤ Pm m ∧
  Pm m ≤ S m + ∑ k in finset.range (r + 1), (-1) ^ k * nat.choose (m + k) k * S (m + k) ∧
  S m + ∑ k in finset.range (r + 2), (-1) ^ k * nat.choose (m + k - 1) k * S (m + k) ≤ Bgtm m ∧
  Bgtm m ≤ S m + ∑ k in finset.range (r + 1), (-1) ^ k * nat.choose (m + k - 1) k * S (m + k) := 
by
  -- Lean 4 proof omitted
  sorry

end bonferroni_inequalities_l681_681943


namespace cyclic_quad_midpoint_proportion_l681_681103

variables {A B C D P E : Type} 
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space P] [metric_space E]
variables [is_metric_space_cyclic_quadrilateral A B C D] (h1 : ∃ P, cyclic_quadrilateral_diagonals_intersect A B C D P)
variables (h2 : (∃ x y z w : ℝ, line_ratio A B D x y ∧ line_ratio C B D x w ∧ x = w))
variables (h3 : E = midpoint A C)

theorem cyclic_quad_midpoint_proportion :
  ∃ BE ED BP PD : ℝ, segment_ratio BE ED = segment_ratio BP PD :=
sorry

end cyclic_quad_midpoint_proportion_l681_681103


namespace divisor_of_44404_l681_681388

theorem divisor_of_44404: ∃ k : ℕ, 2 * 11101 = k ∧ k ∣ (44402 + 2) :=
by
  sorry

end divisor_of_44404_l681_681388


namespace taxi_fare_distance_l681_681854

-- Define the fare calculation and distance function
def fare (x : ℕ) : ℝ :=
  if x ≤ 4 then 10
  else 10 + (x - 4) * 1.5

-- Proof statement
theorem taxi_fare_distance (x : ℕ) : fare x = 16 → x = 8 :=
by
  -- Proof skipped
  sorry

end taxi_fare_distance_l681_681854


namespace arithmetic_sequence_sum_l681_681143

-- Define the arithmetic sequence properties
def is_arithmetic_sequence (seq : ℕ → ℕ) :=
  ∀ n : ℕ, seq (n + 1) = seq n + 2

-- Define the arithmetic sequence in question
def sequence : ℕ → ℕ
| 0       := 2
| (n + 1) := sequence n + 2

-- Check that our sequence matches the properties of an arithmetic sequence
lemma sequence_is_arithmetic : is_arithmetic_sequence sequence :=
by intros n; simp [sequence]

-- Define the sum of the first n terms of the sequence
def sum_n_terms (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, sequence i

-- State the main theorem to be proven: the sum of the first 10 terms is 110
theorem arithmetic_sequence_sum : sum_n_terms 10 = 110 :=
sorry

end arithmetic_sequence_sum_l681_681143


namespace greatest_3digit_base8_divisible_by_7_l681_681904

def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : ℕ), n = 0b777 ∧ (base8_to_base10 0b777) % 7 = 0 ∧ ∀ m < 0o777, m % 7 = 0 → base8_to_base10 m < base8_to_base10 0b777 :=
by
  sorry

end greatest_3digit_base8_divisible_by_7_l681_681904


namespace num_customers_who_tried_sample_l681_681978

theorem num_customers_who_tried_sample :
  ∀ (samples_per_box boxes_opened samples_left : ℕ), 
  samples_per_box = 20 →
  boxes_opened = 12 →
  samples_left = 5 →
  let total_samples := samples_per_box * boxes_opened in
  let samples_used := total_samples - samples_left in
  samples_used = 235 :=
by 
  intros samples_per_box boxes_opened samples_left h_samples_per_box h_boxes_opened h_samples_left total_samples samples_used
  simp [h_samples_per_box, h_boxes_opened, h_samples_left]
  sorry

end num_customers_who_tried_sample_l681_681978


namespace matrix_inverse_problem_l681_681266

noncomputable theory
open_locale classical

universe u
variables {m n : Type u} [fintype n] [decidable_eq n] [fintype m] [decidable_eq m]
variables {α : Type u} [field α]

def inv_matrix {n : Type u} [decidable_eq n] [fintype n] (B : matrix n n α) : Prop :=
  invertible B

theorem matrix_inverse_problem
  {n : Type u} [decidable_eq n] [fintype n]
  {α : Type u} [field α]
  (B : matrix n n α) (hB_inv : inv_matrix B)
  (h_eq : (B - (3 : α) • (1 : matrix n n α)) * (B - (5 : α) • (1 : matrix n n α)) = 0) :
  B + (12 : α) • (B⁻¹) = (8 : α) • (1 : matrix n n α) :=
sorry

end matrix_inverse_problem_l681_681266


namespace maximum_pizzas_baked_on_Friday_l681_681203

def george_bakes := 
  let total_pizzas : ℕ := 1000
  let monday_pizzas := total_pizzas * 7 / 10
  let tuesday_pizzas := if monday_pizzas * 4 / 5 < monday_pizzas * 9 / 10 
                        then monday_pizzas * 4 / 5 
                        else monday_pizzas * 9 / 10
  let wednesday_pizzas := if tuesday_pizzas * 4 / 5 < tuesday_pizzas * 9 / 10 
                          then tuesday_pizzas * 4 / 5 
                          else tuesday_pizzas * 9 / 10
  let thursday_pizzas := if wednesday_pizzas * 4 / 5 < wednesday_pizzas * 9 / 10 
                         then wednesday_pizzas * 4 / 5 
                         else wednesday_pizzas * 9 / 10
  let friday_pizzas := if thursday_pizzas * 4 / 5 < thursday_pizzas * 9 / 10 
                       then thursday_pizzas * 4 / 5 
                       else thursday_pizzas * 9 / 10
  friday_pizzas

theorem maximum_pizzas_baked_on_Friday : george_bakes = 2 := by
  sorry

end maximum_pizzas_baked_on_Friday_l681_681203


namespace smallest_x_l681_681707

theorem smallest_x (y : ℤ) (h1 : 0.9 = (y : ℚ) / (151 + x)) (h2 : 0 < x) (h3 : 0 < y) : x = 9 :=
sorry

end smallest_x_l681_681707


namespace max_f_on_interval_l681_681206

def f (θ : ℝ) : ℝ := 
  sqrt (1 - cos θ + sin θ) + sqrt (cos θ + 2) + sqrt (3 - sin θ)

theorem max_f_on_interval :
  ∃ θ ∈ set.Icc (0 : ℝ) (π : ℝ), 
  ∀ θ' ∈ set.Icc (0 : ℝ) (π : ℝ), f θ' ≤ 3 * sqrt 2 ∧ f θ = 3 * sqrt 2 :=
sorry

end max_f_on_interval_l681_681206


namespace customers_tried_sample_l681_681982

theorem customers_tried_sample
  (samples_per_box : ℕ)
  (boxes_opened : ℕ)
  (samples_left_over : ℕ)
  (samples_per_customer : ℕ := 1)
  (h_samples_per_box : samples_per_box = 20)
  (h_boxes_opened : boxes_opened = 12)
  (h_samples_left_over : samples_left_over = 5) :
  (samples_per_box * boxes_opened - samples_left_over) / samples_per_customer = 235 :=
by
  sorry

end customers_tried_sample_l681_681982


namespace num_factors_of_72_l681_681702

def num_factors (n : ℕ) : ℕ :=
  (n.factorization.to_multiset.map (λ x, x.2 + 1)).prod

theorem num_factors_of_72 :
  num_factors 72 = 12 :=
by
  -- Lean specific details to calculate the number of factors based on prime factorization
  have prime_factors : 72.factorization = [(2, 3), (3, 2)].to_finmap,
  by sorry,
  rw [num_factors, prime_factors],
  norm_num

end num_factors_of_72_l681_681702


namespace pyramid_height_l681_681437

-- Define the edge length of the cube
def cube_edge_length : ℝ := 5

-- Define the base edge length of the pyramid
def pyramid_base_edge_length : ℝ := 10

-- Define the volume of a cube with edge length 5 units
def cube_volume : ℝ := cube_edge_length ^ 3

-- Define the volume of a pyramid with a square base
def pyramid_volume (h : ℝ) : ℝ := (1 / 3) * (pyramid_base_edge_length ^ 2) * h

-- Add a theorem to prove the height of the pyramid
theorem pyramid_height : ∃ h : ℝ, cube_volume = pyramid_volume h ∧ h = 3.75 :=
by
  -- Given conditions and correct answer lead to the proof of the height being 3.75
  sorry

end pyramid_height_l681_681437


namespace floor_of_47_l681_681557

theorem floor_of_47 : int.floor 4.7 = 4 :=
sorry

end floor_of_47_l681_681557


namespace ellipse_problem_part1_ellipse_problem_part2_l681_681301

-- Statement of the problem
theorem ellipse_problem_part1 :
  ∃ k : ℝ, (∀ x y : ℝ, (x^2 / 2) + y^2 = 1 → (
    (∃ t > 0, x = t * y + 1) → k = (Real.sqrt 2) / 2)) :=
sorry

theorem ellipse_problem_part2 :
  ∃ S_max : ℝ, ∀ (t : ℝ), (t > 0 → (S_max = (4 * (t^2 + 1)^2) / ((t^2 + 2) * (2 * t^2 + 1)))) → t^2 = 1 → S_max = 16 / 9 :=
sorry

end ellipse_problem_part1_ellipse_problem_part2_l681_681301


namespace sum_C2_eq_165_l681_681998

noncomputable def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def sum_binom_ctwo : ℕ :=
  (∑ n in Finset.range 9, binom (n + 2) 2)

theorem sum_C2_eq_165 : sum_binom_ctwo = 165 := by
  sorry

end sum_C2_eq_165_l681_681998


namespace sum_of_intersection_points_l681_681162

theorem sum_of_intersection_points : 
  ∃ x_vals, 
  (∀ c d : ℕ+, (2 : ℕ) * c * x_vals + 12 = 0 ∧ 6 * x_vals + d = 0) → 
    Σ (x : ℝ), x ∈ x_vals = -14.5 :=
by
  sorry


end sum_of_intersection_points_l681_681162


namespace common_sale_days_count_l681_681064

def bookstore_sale_days : list ℕ := [4, 8, 12, 16, 20, 24, 28]
def shoe_store_sale_days : list ℕ := [5, 12, 19, 26]

theorem common_sale_days_count : (list.filter (λ day, day ∈ shoe_store_sale_days) bookstore_sale_days).length = 2 :=
by
  sorry

end common_sale_days_count_l681_681064


namespace problem_condition_l681_681855

noncomputable def m := 2
def f (x : ℝ) : ℝ := x ^ 2015

theorem problem_condition
  (a b : ℝ)
  (h1 : a + b > 0)
  (h2 : a * b < 0) :
  f(a) + f(b) > 0 :=
by
  sorry

end problem_condition_l681_681855


namespace greatest_div_by_seven_base_eight_l681_681894

theorem greatest_div_by_seven_base_eight : ∃ n : ℕ, 
  (n < 512) ∧ (Divisibility.divides 7 n) ∧ 
  (∀ m : ℕ, (m < 512) → (Divisibility.divides 7 m) → m ≤ n) ∧ 
  nat.to_digits 8 n = [7, 7, 4] := 
sorry

end greatest_div_by_seven_base_eight_l681_681894


namespace sandbox_width_l681_681081

theorem sandbox_width :
  ∀ (length area width : ℕ), length = 312 → area = 45552 →
  area = length * width → width = 146 :=
by
  intros length area width h_length h_area h_eq
  sorry

end sandbox_width_l681_681081


namespace find_sum_u_v_l681_681104

theorem find_sum_u_v (u v : ℤ) (huv : 0 < v ∧ v < u) (pentagon_area : u^2 + 3 * u * v = 451) : u + v = 21 :=
by 
  sorry

end find_sum_u_v_l681_681104


namespace probability_odd_and_greater_than_5000_l681_681364

theorem probability_odd_and_greater_than_5000 :
  (∃ (a b c d : ℕ), {a, b, c, d} = {3, 5, 7, 11} 
     ∧ 5000 < 1000 * a + 100 * b + 10 * c + d 
     ∧ (d % 2 = 1)) →
  (18 / 24 = 3 / 4) := 
  by sorry

end probability_odd_and_greater_than_5000_l681_681364


namespace largest_integer_solution_l681_681403

theorem largest_integer_solution (x : ℤ) (h : (x : ℚ) / 3 + 4 / 5 < 5 / 3) : x ≤ 2 :=
sorry

end largest_integer_solution_l681_681403


namespace sandy_total_sums_attempted_l681_681832

theorem sandy_total_sums_attempted (C I : ℕ) 
  (marks_per_correct_sum : ℕ := 3) 
  (marks_lost_per_incorrect_sum : ℕ := 2) 
  (total_marks : ℕ := 45) 
  (correct_sums : ℕ := 21) 
  (H : 3 * correct_sums - 2 * I = total_marks) 
  : C + I = 30 := 
by 
  sorry

end sandy_total_sums_attempted_l681_681832


namespace testing_schemes_count_l681_681395

theorem testing_schemes_count :
  let genuine_products := 5
  let defective_products := 4
  let total_tests := 10
  let required_tests := 6
  /*
    The number of ways to choose and arrange the products given the conditions.
    1. Choosing 1 defective product for the 6th test
    2. Choosing 2 out of 5 genuine products
    3. Arranging remaining 3 defective and 2 genuine in first 5 tests
  */
  (choose defective_products 1) * (choose genuine_products 2) * (factorial 5) = 4800 := sorry

end testing_schemes_count_l681_681395


namespace first_player_wins_l681_681006

-- Definitions for the conditions:
def numLamps : ℕ := 2012
def initialPlayer : ℕ := 1 -- 1 for the first player, 2 for the second player

/- 
The main assertion: The first player has a winning strategy given the rules of the game.
-/
theorem first_player_wins (numLamps : ℕ) (initialPlayer : ℕ) : initialPlayer = 1 → ∀ states : finset (fin numLamps → bool), ∃ strategy : (fin numLamps → bool) → (fin numLamps → bool), 
    (∀ state, state ∈ states → strategy state ∉ states) → ∀ state ∈ states, strategy state = state :=
begin
  intros,
  sorry
end

#eval first_player_wins numLamps initialPlayer

end first_player_wins_l681_681006


namespace cosine_of_angle_between_diagonals_l681_681967

noncomputable def vector_a : ℝ × ℝ × ℝ := (3, 0, 4)
noncomputable def vector_b : ℝ × ℝ × ℝ := (1, 2, 3)

noncomputable def cosine_theta : ℝ := 
  let a₊b := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2, vector_a.3 + vector_b.3)
  let b₋a := (vector_b.1 - vector_a.1, vector_b.2 - vector_a.2, vector_b.3 - vector_a.3)
  let dot_product := (a₊b.1 * b₋a.1) + (a₊b.2 * b₋a.2) + (a₊b.3 * b₋a.3)
  let norm_a₊b := real.sqrt (a₊b.1 ^ 2 + a₊b.2 ^ 2 + a₊b.3 ^ 2)
  let norm_b₋a := real.sqrt (b₋a.1 ^ 2 + b₋a.2 ^ 2 + b₋a.3 ^ 2)
  dot_product / (norm_a₊b * norm_b₋a)

theorem cosine_of_angle_between_diagonals :
  cosine_theta = -11 / (3 * real.sqrt 69) :=
sorry

end cosine_of_angle_between_diagonals_l681_681967


namespace correct_propositions_l681_681093

/-- Definitions of each proposition as conditions -/
def prop1 := ∀ (P1 P2 : Plane) (L : Line), (P1 ∥ L ∧ P2 ∥ L) → P1 ∥ P2
def prop2 := ∀ (P1 P2 P3 : Plane), (P1 ∥ P3 ∧ P2 ∥ P3) → P1 ∥ P2
def prop3 := ∀ (L1 L2 L3 : Line), (L1 ⊥ L3 ∧ L2 ⊥ L3) → L1 ∥ L2
def prop4 := ∀ (L1 L2 : Line) (P : Plane), (L1 ⊥ P ∧ L2 ⊥ P) → L1 ∥ L2

/-- Math proof problem stating that propositions 2 and 4 are correct -/
theorem correct_propositions (P1 P2 P3 : Plane) (L1 L2 L3 : Line) (P : Plane) :
  (prop2 P1 P2 P3) ∧ (prop4 L1 L2 P) :=
begin
  sorry
end

end correct_propositions_l681_681093


namespace inequality_l681_681632

def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_derivative_inequality : ∀ x : ℝ, deriv f x < f x

theorem inequality : e ^ 2 * f 2 > f 0 ∧ f 0 > e ^ (-1) * f 1 :=
sorry

end inequality_l681_681632


namespace probability_product_multiple_of_4_l681_681713

theorem probability_product_multiple_of_4 :
  let cards := {1, 2, 3, 4, 5, 6}
  let pairs := { (a, b) | a ∈ cards ∧ b ∈ cards ∧ a < b }
  let total_pairs := 15
  let valid_pairs := { (1, 4), (2, 4), (3, 4), (4, 5), (4, 6) }
  let num_valid_pairs := 5
  num_valid_pairs / total_pairs = 1 / 3 := by
  sorry

end probability_product_multiple_of_4_l681_681713


namespace area_of_triangle_AOB_l681_681620

open Real
open EuclideanGeometry

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 6 * x - 8 * y = 0

-- Define the line l1
def line_l1 (x y : ℝ) : Prop :=
  x - sqrt 3 * y = 0

-- Define the line l2
def line_l2 (x y : ℝ) : Prop :=
  sqrt 3 * x - y = 0

-- Area calculation proof statement
theorem area_of_triangle_AOB :
  ∃ A B : ℝ × ℝ,
    (curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧ line_l1 A.1 A.2 ∧ line_l2 B.1 B.2) ∧
    let α := (A.1 - 3)/5, β := (A.2 - 4)/5 in 
    let ρ1 := 6 * cos (π / 6) + 8 * sin (π / 6),
        ρ2 := 6 * cos (π / 3) + 8 * sin (π / 3) in
    1 / 2 * ρ1 * ρ2 * sin (π / 3 - π / 6) = 12 + 25 * sqrt 3 / 4 :=
sorry

end area_of_triangle_AOB_l681_681620


namespace AreaOfConvexPentagon_l681_681286

variables (A B C D E : Type) [InnerProductSpace ℝ (A × B × C × D × E)]
variables (a b : ℝ)
variables (angle_A angle_C : ℝ)
variables (AB AE BC CD AC : ℝ)

def isConvexPentagon (ABCDE : Set (A × B × C × D × E)) :=
  angle_A = 90 ∧ angle_C = 90 ∧ AB = a ∧ AE = a ∧ BC = b ∧ CD = b ∧ AC = 1

theorem AreaOfConvexPentagon (ABCDE : Set (A × B × C × D × E)) (h : isConvexPentagon ABCDE) : 
  (Area ABCDE = 0.5) :=
by
  sorry

end AreaOfConvexPentagon_l681_681286


namespace jack_marathon_time_l681_681759

-- Define the conditions
def marathon_distance : ℝ := 42 -- in kilometers
def jill_time : ℝ := 4.2 -- in hours
def speed_ratio : ℝ := 0.84

-- Define Jill's average speed
def jill_speed : ℝ := marathon_distance / jill_time

-- Define Jack's average speed based on the ratio
def jack_speed : ℝ := speed_ratio * jill_speed

-- Prove Jack's time to complete the marathon
theorem jack_marathon_time : marathon_distance / jack_speed = 5 :=
by
  -- skip the proof for now
  sorry

end jack_marathon_time_l681_681759


namespace max_length_small_stick_l681_681396

theorem max_length_small_stick (a b c : ℕ) 
  (ha : a = 24) (hb : b = 32) (hc : c = 44) :
  Nat.gcd (Nat.gcd a b) c = 4 :=
by
  rw [ha, hb, hc]
  -- At this point, the gcd calculus will be omitted, filing it with sorry
  sorry

end max_length_small_stick_l681_681396


namespace sin_A_value_l681_681625

theorem sin_A_value (A B C : ℝ) (a b c : ℝ) 
  (cosB : ℝ) (h1 : cosB = -1 / 4) 
  (h2 : a = 6) 
  (area : ℝ) (h3 : area = 3 * real.sqrt 15)
  (h4 : 0 < b) : 
  sin A = 3 * real.sqrt 15 / 16 :=
by
  sorry

end sin_A_value_l681_681625


namespace time_to_cross_platform_l681_681987

-- Definitions from conditions
def train_speed_kmph : ℕ := 72
def speed_conversion_factor : ℕ := 1000 / 3600
def train_speed_mps : ℤ := train_speed_kmph * speed_conversion_factor
def time_cross_man_sec : ℕ := 16
def platform_length_meters : ℕ := 280

-- Proving the total time to cross platform
theorem time_to_cross_platform : ∃ t : ℕ, t = (platform_length_meters + (train_speed_mps * time_cross_man_sec)) / train_speed_mps ∧ t = 30 := 
by
  -- Since the proof isn't required, we add "sorry" to act as a placeholder.
  sorry

end time_to_cross_platform_l681_681987


namespace flags_required_for_track_l681_681004

theorem flags_required_for_track (track_length flag_interval : ℕ) (start_flag_distance : ℕ) :
  track_length = 400 → flag_interval = 90 → start_flag_distance = 0 → 
  ∃ (n : ℕ), n = 5 ∧ (∀ m, m ∈ (finset.range n) →
  ((m * flag_interval + start_flag_distance) % track_length) ≠ 0) :=
by
  sorry

end flags_required_for_track_l681_681004


namespace horizontal_distance_is_1_l681_681961

-- Define the relevant points
def dave_location : ℝ × ℝ := (8, -15)
def emily_location : ℝ × ℝ := (4, 18)
def fiona_location : ℝ × ℝ := (5, (18 + (-15)) / 2)

-- Define the midpoint calculation
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Calculate the midpoint of Dave and Emily's locations
def dave_emily_midpoint : ℝ × ℝ := midpoint dave_location emily_location

-- Prove the horizontal distance from the midpoint to Fiona's location
theorem horizontal_distance_is_1 :
  abs (dave_emily_midpoint.1 - fiona_location.1) = 1 :=
sorry

end horizontal_distance_is_1_l681_681961


namespace smallest_k_for_square_l681_681598

theorem smallest_k_for_square : ∃ k : ℕ, (2016 * 2017 * 2018 * 2019 + k) = n^2 ∧ k = 1 :=
by
  use 1
  sorry

end smallest_k_for_square_l681_681598


namespace f_neg1_gt_f_1_l681_681220

-- Definition of the function f and its properties.
variable {f : ℝ → ℝ}
variable (df : Differentiable ℝ f)
variable (eq_f : ∀ x : ℝ, f x = x^2 + 2 * x * f' 2)

-- The problem statement to prove f(-1) > f(1).
theorem f_neg1_gt_f_1 (h_deriv : ∀ x : ℝ, deriv f x = 2 * x - 8):
  f (-1) > f 1 :=
by
  sorry

end f_neg1_gt_f_1_l681_681220


namespace sum_binom_eq_neg_two_pow_49_l681_681052

theorem sum_binom_eq_neg_two_pow_49:
  (\sum k in Finset.range 50, (-1:ℤ)^k * Nat.choose 99 (2 * k)) = -2^49 := 
  sorry

end sum_binom_eq_neg_two_pow_49_l681_681052


namespace area_of_triangle_GCD_l681_681349

-- Definitions for conditions
def area_of_square := 144
def ratio_BE_EC := 3 / 1
def area_of_quad_BE_GF := 25

-- Derived conditions based on conditions given initially are provided here but marked as noncomputable as an assumption.
noncomputable def side_length_of_square := Real.sqrt area_of_square
noncomputable def length_BE := (ratio_BE_EC / (ratio_BE_EC + 1)) * side_length_of_square
noncomputable def length_EC := side_length_of_square - length_BE
noncomputable def area_of_triangle_ABE := 0.5 * length_BE * side_length_of_square
noncomputable def area_of_triangle_ECD := 0.5 * length_EC * side_length_of_square
noncomputable def area_of_triangle_AED := area_of_square - area_of_triangle_ABE - area_of_triangle_ECD
noncomputable def area_of_triangle_FE_G := 0.25 * area_of_triangle_AED

-- Hypotheses and final statement
theorem area_of_triangle_GCD : 
  Π (area_of_square = 144) 
    (BE_ratio_EC : BE / EC = 3 / 1)
    (area_of_quadrilateral_BEGF = 25), 
    area_of_GCD = 9 :=
by 
  -- Provided all derivations are accurate, we will assert that the area of GCD is 9
  sorry

end area_of_triangle_GCD_l681_681349


namespace equation_of_circle_center_tangent_l681_681369

noncomputable def circle_radius_to_line (A B C x0 y0 : ℝ) : ℝ :=
  |A * x0 + B * y0 + C| / Real.sqrt (A^2 + B^2)

theorem equation_of_circle_center_tangent 
  (radius : ℝ)
  (center : ℝ × ℝ)
  (line_coeffs : ℝ × ℝ × ℝ)
  (tangent_distance_eq_radius : ∀ x0 y0 : ℝ, x0 = center.1 → y0 = center.2 → circle_radius_to_line line_coeffs.1 line_coeffs.2 line_coeffs.3 x0 y0 = radius) :
  ∃ r, x^2 + y^2 = r^2 ∧ r = radius := 
by
  let center := (0, 0)
  let line_coeffs := (1, 1, -2)
  have tangent_at_origin : circle_radius_to_line 1 1 (-2) 0 0 = Real.sqrt 2 := 
    by sorry -- omitted computation
  show ∃ r, x^2 + y^2 = r^2 ∧ r = Real.sqrt 2 := 
    by sorry -- omitted formal proof
  exact sorry -- overall proof

end equation_of_circle_center_tangent_l681_681369


namespace angle_DGO_is_50_degrees_l681_681294

theorem angle_DGO_is_50_degrees
  (triangle_DOG : Type)
  (D G O : triangle_DOG)
  (angle_DOG : ℝ)
  (angle_DGO : ℝ)
  (angle_OGD : ℝ)
  (bisect : Prop) :

  angle_DGO = 50 := 
by
  -- Conditions
  have h1 : angle_DGO = angle_DOG := sorry
  have h2 : angle_DOG = 40 := sorry
  have h3 : bisect := sorry
  -- Goal
  sorry

end angle_DGO_is_50_degrees_l681_681294


namespace count_valid_arrangements_l681_681280

-- Definitions based on conditions
def male_students : ℕ := 2
def female_students : ℕ := 2
def valid_arrangements : ℕ := 8

-- Theorem statement
theorem count_valid_arrangements (m f : ℕ) (h₁ : m = male_students) (h₂ : f = female_students) :
  (∃ (arrangements : ℕ), arrangements = valid_arrangements) :=
by
  use valid_arrangements
  have case1 : (2! * 2!) = 4, by simp [factorial]
  have case2 : (2! * 2!) = 4, by simp [factorial]
  have total : 4 + 4 = 8, by norm_num
  exact total

end count_valid_arrangements_l681_681280


namespace correct_meiosis_fertilization_l681_681094

-- Define the individual statements about meiosis and fertilization
def statement_A : Prop := "Half of the genetic material in the zygote comes from the father, and half from the mother"
def statement_B : Prop := "Meiosis and fertilization maintain a constant number of chromosomes in the somatic cells of parents and offspring"
def statement_C : Prop := "Human secondary spermatocytes contain 0 or 1 Y chromosome"
def statement_D : Prop := "Fertilization achieves gene recombination, leading to the diversity of offspring through sexual reproduction"

-- Define the correctness predicates for the statements
def is_correct_A : Prop := False
def is_correct_B : Prop := True
def is_correct_C : Prop := False
def is_correct_D : Prop := False

-- The proof problem: Prove that statement B is the correct one
theorem correct_meiosis_fertilization : 
  (¬is_correct_A) ∧ (is_correct_B) ∧ (¬is_correct_C) ∧ (¬is_correct_D) :=
by
  { sorry }

end correct_meiosis_fertilization_l681_681094


namespace decreasing_on_interval_l681_681660

noncomputable def f (a x : ℝ) : ℝ := 2^(x * (x - a))

theorem decreasing_on_interval (a : ℝ) : (a ≥ 2) ↔ ∀ x ∈ Set.Ioo 0 1, (deriv (λ x, 2^(x * (x - a)))) x ≤ 0 :=
sorry

end decreasing_on_interval_l681_681660


namespace find_alpha_l681_681670

-- Definitions
def polar_equation (θ : ℝ) : ℝ := 4 * cos θ

def line_parametric_eq (t α : ℝ) : ℝ × ℝ :=
  (1 + t * cos α, t * sin α)

-- Conditions
def curve_cartesian_eq (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4

def distance_AB (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

axiom AB_distance : ∀ (α : ℝ), α ∈ set.Icc 0 real.pi → (distance_AB (line_parametric_eq (-1 - root 3 α).fst (line_parametric_eq (-1 - root 3 α).snd (line_parametric_eq (1 - root 3 α).fst (line_parametric_eq (1 - root 3 α).snd) = real.sqrt 14

-- Proof problem statement
theorem find_alpha (α : ℝ) (hα : α ∈ set.Icc 0 real.pi) : cos α = real.sqrt 2 / 2 ∨ cos α = - real.sqrt 2 / 2 :=
sorry

end find_alpha_l681_681670


namespace volume_of_open_box_l681_681469

-- Declare the dimensions of the trapezoidal sheet and the squares cut from the corners
structure Sheet where
  top_side : ℕ
  bottom_side : ℕ
  height : ℕ
  cut_squares : List ℕ

def trapezoidal_sheet := Sheet.mk 60 90 40 [6, 8, 10, 12]

-- Function to calculate new dimensions of the trapezoidal sheet
def new_dim (lengths : List ℕ) (s : Sheet) : ℕ :=
  s.top_side - (lengths.head! + lengths.tail.head!)

def calculate_height (cut_squares : List ℕ) : ℕ :=
  Nat.minimum cut_squares

def calculate_volume (new_top : ℕ) (new_bottom : ℕ) (original_height : ℕ) (box_height : ℕ) : ℕ :=
  (1/2:ℚ) * (new_top + new_bottom) * original_height * box_height |>.toNat

theorem volume_of_open_box :
  let top_new := new_dim [6, 8] trapezoidal_sheet
  let bottom_new := new_dim [10, 12] trapezoidal_sheet
  let box_height := calculate_height trapezoidal_sheet.cut_squares
  calculate_volume top_new bottom_new trapezoidal_sheet.height box_height = 13680 :=
by
  -- Lean code to handle computations and assertions
  sorry

end volume_of_open_box_l681_681469


namespace arthur_walks_distance_l681_681098

theorem arthur_walks_distance :
  ∀ (blocks_west blocks_south : ℕ) (block_length : ℝ), 
  blocks_west = 8 → 
  blocks_south = 10 → 
  block_length = 0.25 → 
  (blocks_west + blocks_south) * block_length = 4.5 := 
by {
  intros blocks_west blocks_south block_length,
  assume h_west h_south h_length,
  rw [h_west, h_south, h_length],
  norm_num,
}

end arthur_walks_distance_l681_681098


namespace sum_of_possible_n_values_l681_681289

theorem sum_of_possible_n_values (m n : ℕ) 
  (h : 0 < m ∧ 0 < n)
  (eq1 : 1/m + 1/n = 1/5) : 
  n = 6 ∨ n = 10 ∨ n = 30 → 
  m = 30 ∨ m = 10 ∨ m = 6 ∨ m = 5 ∨ m = 25 ∨ m = 1 →
  (6 + 10 + 30 = 46) := 
by 
  sorry

end sum_of_possible_n_values_l681_681289


namespace greatest_3_digit_base_8_divisible_by_7_l681_681911

open Nat

def is_3_digit_base_8 (n : ℕ) : Prop := n < 8^3

def is_divisible_by_7 (n : ℕ) : Prop := 7 ∣ n

theorem greatest_3_digit_base_8_divisible_by_7 :
  ∃ x : ℕ, is_3_digit_base_8 x ∧ is_divisible_by_7 x ∧ x = 7 * (8 * (8 * 7 + 7) + 7) :=
by
  sorry

end greatest_3_digit_base_8_divisible_by_7_l681_681911


namespace angle_between_vectors_l681_681246
open Real

variables {a b : ℝ^3} -- assuming ℝ^3 for vector space

def unit_vector (v : ℝ^3) : Prop := ∥v∥ = 1

theorem angle_between_vectors (ha : unit_vector a)
                              (hb : unit_vector b)
                              (h_dot : dot_product a b = -1/4)
                              (c := a + 2 • b) :
                              ⟪a, c⟫ = arccos (1/4) :=
sorry

end angle_between_vectors_l681_681246


namespace mean_is_minus_one_l681_681192

noncomputable def mean_of_solutions : ℝ :=
  let roots : Multiset ℝ := Polynomial.roots (Polynomial.mk [(-10 : ℝ), (-13 : ℝ), 2, 1])
  in (roots.sum / roots.card)

theorem mean_is_minus_one :
  mean_of_solutions = -1 := by
  sorry

end mean_is_minus_one_l681_681192


namespace inlet_rate_in_litres_per_minute_l681_681479

open Real

theorem inlet_rate_in_litres_per_minute 
  (capacity : ℝ) (outlet_time : ℝ) (extra_time : ℝ) (total_time : ℝ) (correct_answer : ℝ) :
  capacity = 6400 →
  outlet_time = 5 →
  extra_time = 3 →
  total_time = outlet_time + extra_time →
  correct_answer = 8 →
  let R_o := capacity / outlet_time in
  let R_effective := capacity / total_time in
  let R_i := R_o - R_effective in
  R_i / 60 = correct_answer :=
by
  sorry

end inlet_rate_in_litres_per_minute_l681_681479


namespace funnel_height_l681_681954

-- Definitions for the problem conditions
def volume_of_cone (r h : ℝ) : ℝ := (1 / 3) * (Real.pi) * r^2 * h

-- Prove the height of the funnel with given conditions
theorem funnel_height :
  ∃ h : ℝ, round h = 9 ∧ volume_of_cone 4 h = 150 :=
by
  sorry

end funnel_height_l681_681954


namespace greatest_number_of_matching_pairs_l681_681811

theorem greatest_number_of_matching_pairs 
  (original_pairs : ℕ := 27)
  (lost_shoes : ℕ := 9) 
  (remaining_pairs : ℕ := original_pairs - (lost_shoes / 1))
  : remaining_pairs = 18 := by
  sorry

end greatest_number_of_matching_pairs_l681_681811


namespace smallest_value_of_x_l681_681195

theorem smallest_value_of_x : ∃ x : ℝ, |4 * x + 7| = 15 ∧ (∀ y : ℝ, |4 * y + 7| = 15 → y ≥ x) :=
by
  use -5.5
  split
  · norm_num
    apply abs_eq_iff
    split
    · norm_num
    · norm_num
  · intro y hy
    cases abs_eq hy
    · linarith
    · linarith

end smallest_value_of_x_l681_681195


namespace csc_product_identity_l681_681930

theorem csc_product_identity :
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ (m ^ n = ∏ k in finset.range 30, (real.csc (real.pi / 60 * (3 * (k + 1))))^2) ∧ (m + n = 61) :=
by
  sorry

end csc_product_identity_l681_681930


namespace number_of_zeros_l681_681381

noncomputable def f (x : Real) : Real :=
if x > 0 then -1 + Real.log x
else 3 * x + 4

theorem number_of_zeros : (∃ a b : Real, f a = 0 ∧ f b = 0 ∧ a ≠ b) := 
sorry

end number_of_zeros_l681_681381


namespace find_ab_bc_value_l681_681256

theorem find_ab_bc_value
  (a b c : ℝ)
  (h : a / 3 = b / 4 ∧ b / 4 = c / 5) :
  (a + b) / (b - c) = -7 := by
sorry

end find_ab_bc_value_l681_681256


namespace min_omega_l681_681645

noncomputable def f (ω φ : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + φ)

theorem min_omega (ω φ : ℝ) (hω : ω > 0)
  (h_sym : ∀ x : ℝ, f ω φ (2 * (π / 3) - x) = f ω φ x)
  (h_val : f ω φ (π / 12) = 0) :
  ω = 2 :=
sorry

end min_omega_l681_681645


namespace branch_fraction_remaining_l681_681772

theorem branch_fraction_remaining :
  let original_length := (3 : ℚ) in
  let third := original_length / 3 in
  let fifth := original_length / 5 in
  let removed_length := third + fifth in
  let remaining_length := original_length - removed_length in
  let remaining_fraction := remaining_length / original_length in
  remaining_fraction = (7 / 15) :=
by
  sorry

end branch_fraction_remaining_l681_681772


namespace sum_infinite_series_l681_681197

def G (n : ℕ) : ℝ := ∑ i in finset.range (n+1), 1 / (i + 1)^2

theorem sum_infinite_series :
  (∑' n : ℕ, 1 / ((n + 1) * G n * G (n + 1))) = 1 - 6 / (Real.pi^2) :=
sorry

end sum_infinite_series_l681_681197


namespace num_valid_triangle_triples_l681_681869

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem num_valid_triangle_triples : 
  (finset.univ.filter (λ (abc : ℕ × ℕ × ℕ), 
    let (a, b, c) := abc in 
      a ≤ b ∧ b ≤ c ∧ a + b + c = 31 ∧ is_triangle a b c)).card = 24 := 
by
  sorry

end num_valid_triangle_triples_l681_681869


namespace sum_digits_product_of_sevens_and_threes_l681_681145

-- Definitions for conditions
def string_of_sevens (n : ℕ) : ℕ := nat.iterate (λ s, 10 * s + 7) n 0
def string_of_threes (n : ℕ) : ℕ := nat.iterate (λ t, 10 * t + 3) n 0

-- Theorem statement
theorem sum_digits_product_of_sevens_and_threes :
  (nat.digits 10 (string_of_sevens 80 * string_of_threes 80)).sum = 240 :=
by
  sorry

end sum_digits_product_of_sevens_and_threes_l681_681145


namespace evaluate_expression_l681_681304

-- Define the operation "clubsuit"
def clubsuit (a b : ℝ) : ℝ :=
  (3 * a / b) * (b / a)

-- Prove the main statement
theorem evaluate_expression :
  (clubsuit 7 (clubsuit 4 8)) = 3 → (clubsuit (clubsuit 7 (clubsuit 4 8)) 2 = 3) :=
by
  sorry

end evaluate_expression_l681_681304


namespace book_arrangement_l681_681255

-- Define the books as distinct elements
def math_books := fin 3
def chinese_books := fin 3

/-- Prove that the number of ways to arrange 3 different math books and 3 different Chinese books 
such that no two books of the same subject are adjacent is 72. -/
theorem book_arrangement :
  let ways_to_arrange := 2 * nat.factorial 3 * nat.factorial 3  in
  ways_to_arrange = 72 :=
by
  sorry

end book_arrangement_l681_681255


namespace intersection_tangents_median_l681_681407

open Triangle

def triangle (A B C : Point) : Type := ...

def altitude (B K : Point) : Prop := ...

def diameter_circle_intersects (B K : Point) (circle_diameter : Circle) (A B C : Line) (E F : Point) : Prop :=
  circle_intersects AB E ∧ circle_intersects BC F

def tangents_intersection_on_median (A B C E F I : Point) (tangent_E tangent_F : Line) : Prop :=
  tangents E tangent_E ∧ tangents F tangent_F ∧
  intersection tangent_E tangent_F I ∧
  lies_on_median A B C I

theorem intersection_tangents_median :
  ∀ (A B C E F K I : Point) (BK_circle : Circle),
    acute_triangle A B C →
    altitude B K →
    diameter_circle_intersects B K BK_circle A B C E F →
    tangents_intersection_on_median A B C E F I →
    lies_on_median A B C I :=
sorry

end intersection_tangents_median_l681_681407


namespace combination_18_6_l681_681522

theorem combination_18_6 : nat.choose 18 6 = 18564 :=
by {
  sorry
}

end combination_18_6_l681_681522


namespace sum_complex_powers_l681_681997

theorem sum_complex_powers (i : ℂ) (h : i^4 = 1) : 
  (∑ k in finset.range 2016, i ^ k) = -1 :=
by {
  sorry
}

end sum_complex_powers_l681_681997


namespace find_a_l681_681723

-- Definitions as per the conditions
def f (a x : ℝ) : ℝ := x / ((x + 1) * (x + a))

-- Function symmetry condition: f is symmetrical about the origin, hence odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

-- Theorem statement
theorem find_a (a : ℝ) (h : is_odd (f a)) : a = -1 :=
sorry

end find_a_l681_681723


namespace garden_area_excluding_pond_l681_681464

-- Define necessary mathematical structures and the problem statement.
noncomputable def garden_area_not_taken_by_pond : ℝ :=
  let side_length := 48 / 4 in
  let garden_area := side_length * side_length in
  let pond_rectangle_area := 3 * 2 in
  let pond_semi_circle_radius := (4 : ℝ) / 2 in
  let pond_semi_circle_area := (Real.pi * pond_semi_circle_radius^2) / 2 in
  let total_pond_area := pond_rectangle_area + pond_semi_circle_area in
  let area_not_taken_by_pond := garden_area - total_pond_area in
  area_not_taken_by_pond

theorem garden_area_excluding_pond: 
  |garden_area_not_taken_by_pond - 131.72| < 0.01 :=
by
  -- Exact proof would be provided here, so we put sorry as a placeholder.
  sorry

end garden_area_excluding_pond_l681_681464


namespace Joan_finds_money_l681_681928

theorem Joan_finds_money (dimes_in_jacket dimes_in_shorts : ℕ)
  (value_of_dime cents_in_dollar : ℕ)
  (hj : dimes_in_jacket = 15)
  (hs : dimes_in_shorts = 4)
  (hv : value_of_dime = 10)
  (hc : cents_in_dollar = 100) :
  (dimes_in_jacket + dimes_in_shorts) * value_of_dime / cents_in_dollar = 1.90 :=
by
  sorry

end Joan_finds_money_l681_681928


namespace pyramid_height_l681_681438

-- Define the edge length of the cube
def cube_edge_length : ℝ := 5

-- Define the base edge length of the pyramid
def pyramid_base_edge_length : ℝ := 10

-- Define the volume of a cube with edge length 5 units
def cube_volume : ℝ := cube_edge_length ^ 3

-- Define the volume of a pyramid with a square base
def pyramid_volume (h : ℝ) : ℝ := (1 / 3) * (pyramid_base_edge_length ^ 2) * h

-- Add a theorem to prove the height of the pyramid
theorem pyramid_height : ∃ h : ℝ, cube_volume = pyramid_volume h ∧ h = 3.75 :=
by
  -- Given conditions and correct answer lead to the proof of the height being 3.75
  sorry

end pyramid_height_l681_681438


namespace initial_volume_of_mixture_l681_681075

theorem initial_volume_of_mixture (M W : ℕ) (h1 : 2 * M = 3 * W) (h2 : 4 * M = 3 * (W + 46)) : M + W = 115 := 
sorry

end initial_volume_of_mixture_l681_681075


namespace max_area_of_triangle_ABC_l681_681194

noncomputable def max_area_triangle_ABC (a b c A C : ℝ) : ℝ := 
  if hA : (cos A - sqrt 3 * a * cos C = 0) ∧ (0 < C ∧ C < π) ∧ (a > 0) ∧ (b > 0) 
  then sqrt 3 else 0

theorem max_area_of_triangle_ABC (a b : ℝ) (h_cosA : ∃ (A C : ℝ), cos A - sqrt 3 * a * cos C = 0) 
  (h_range_C : ∃ C, 0 < C ∧ C < π) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  max_area_triangle_ABC a b 2 h_cosA.some h_range_C.some = sqrt 3 :=
sorry

end max_area_of_triangle_ABC_l681_681194


namespace diamonds_in_F20_l681_681371

def F (n : ℕ) : ℕ :=
  -- Define recursively the number of diamonds in figure F_n
  match n with
  | 1 => 1
  | 2 => 9
  | n + 1 => F n + 4 * (n + 1)

theorem diamonds_in_F20 : F 20 = 761 :=
by sorry

end diamonds_in_F20_l681_681371


namespace coeff_a3_expansion_l681_681204

theorem coeff_a3_expansion :
  let a0 := 1,
      a1 := -5,
      a2 := 10,
      a3 := -10,
      a4 := 5,
      a5 := -1 in
  (x - 1) ^ 5 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5 → a3 = 10 :=
by
  sorry

end coeff_a3_expansion_l681_681204


namespace translated_curve_correct_l681_681886

variable {R : Type} [Field R]

def translate_curve (f : R → R) (x : R) : R := f (x - 1) + 2

theorem translated_curve_correct (f : R → R) (h_point : f 1 = 1) : 
  (translate_curve f 2) = 3 :=
by 
  have h : translate_curve f 2 = f (2 - 1) + 2 := rfl
  rw h_point at h
  rw [<-h]
  rfl

end translated_curve_correct_l681_681886


namespace proof_problem_l681_681536

variable (p q r : Prop)

def statement1 := p ∧ ¬q ∧ r
def statement2 := ¬p ∧ ¬q ∧ r
def statement3 := p ∧ ¬q ∧ ¬r
def statement4 := ¬p ∧ q ∧ r

def implies_formula (s : Prop) : Prop := s → ((q → p) → ¬r)

theorem proof_problem :
  (if implies_formula p q r statement1 then 1 else 0) +
  (if implies_formula p q r statement2 then 1 else 0) +
  (if implies_formula p q r statement3 then 1 else 0) +
  (if implies_formula p q r statement4 then 1 else 0) = 2 :=
by
  sorry

end proof_problem_l681_681536


namespace cylinder_increase_l681_681009

-- Definitions based on conditions
def initial_radius : ℝ := 10
def initial_height : ℝ := 4
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

-- Statement to prove
theorem cylinder_increase (x : ℝ) (hx : x ≠ 0) :
  volume_cylinder (initial_radius + x) initial_height = volume_cylinder initial_radius (initial_height + x) →
  x = 5 :=
by
  intros h_eq
  sorry

end cylinder_increase_l681_681009


namespace pawns_impossible_to_cover_all_positions_l681_681950

-- Definitions
def is_even_position (black_square white_square : ℕ) : Prop :=
  (black_square % 2 = white_square % 2)

def is_odd_position (black_square white_square : ℕ) : Prop :=
  ¬ is_even_position black_square white_square

-- Main theorem
theorem pawns_impossible_to_cover_all_positions :
  ¬ ∃ (move_sequence : list (ℕ × ℕ)),
    (∀ p ∈ move_sequence, (p.1 ≠ p.2 ∧ 0 ≤ p.1 ∧ p.1 < 64 ∧ 0 ≤ p.2 ∧ p.2 < 64)) ∧ 
    (∀ (i j : ℕ), 0 ≤ i ∧ i < 64 ∧ 0 ≤ j ∧ j < 64 → (i ≠ j → ((i, j) ∈ move_sequence ∨ (j, i) ∈ move_sequence))) :=
begin
  sorry
end

end pawns_impossible_to_cover_all_positions_l681_681950


namespace admittedApplicants_l681_681435

-- Definitions for the conditions in the problem
def totalApplicants : ℕ := 70
def task1Applicants : ℕ := 35
def task2Applicants : ℕ := 48
def task3Applicants : ℕ := 64
def task4Applicants : ℕ := 63

-- The proof statement
theorem admittedApplicants : 
  ∀ (totalApplicants task3Applicants task4Applicants : ℕ),
  totalApplicants = 70 →
  task3Applicants = 64 →
  task4Applicants = 63 →
  ∃ (interApplicants : ℕ), interApplicants = 57 :=
by
  intros totalApplicants task3Applicants task4Applicants
  intros h_totalApps h_task3Apps h_task4Apps
  sorry

end admittedApplicants_l681_681435


namespace alicia_tax_cents_per_hour_l681_681470

-- Define Alicia's hourly wage in dollars.
def alicia_hourly_wage_dollars : ℝ := 25
-- Define the conversion rate from dollars to cents.
def cents_per_dollar : ℝ := 100
-- Define the local tax rate as a percentage.
def tax_rate_percent : ℝ := 2

-- Convert Alicia's hourly wage to cents.
def alicia_hourly_wage_cents : ℝ := alicia_hourly_wage_dollars * cents_per_dollar

-- Define the theorem that needs to be proved.
theorem alicia_tax_cents_per_hour : alicia_hourly_wage_cents * (tax_rate_percent / 100) = 50 := by
  sorry

end alicia_tax_cents_per_hour_l681_681470


namespace find_m_of_perpendicular_vectors_l681_681640

variable (m : ℝ)
variable (AB BC : ℝ × ℝ)

def vector_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_m_of_perpendicular_vectors
  (hAB : AB = (2, 5))
  (hBC : BC = (m, -2))
  (h_perp : vector_perpendicular AB BC) :
  m = 5 :=
by
  subst hAB
  subst hBC
  simp [vector_perpendicular] at h_perp
  solve_by_elim

end find_m_of_perpendicular_vectors_l681_681640


namespace haley_initial_trees_l681_681988

theorem haley_initial_trees (dead_trees trees_left initial_trees : ℕ) 
    (h_dead: dead_trees = 2)
    (h_left: trees_left = 10)
    (h_initial: initial_trees = trees_left + dead_trees) : 
    initial_trees = 12 := 
by sorry

end haley_initial_trees_l681_681988


namespace floor_of_4point7_l681_681546

theorem floor_of_4point7 : (Real.floor 4.7) = 4 :=
by
  sorry

end floor_of_4point7_l681_681546


namespace prime_factors_converse_l681_681799

theorem prime_factors_converse (P : ℤ[X]) (hP_nonconst : ¬is_constant P) :
  (∀ n : ℤ, ∀ p : ℕ, p.prime → (p ∣ P.eval n → p ∣ n)) → 
  (∀ n : ℤ, ∀ p : ℕ, p.prime → (p ∣ n → p ∣ P.eval n)) :=
by
  sorry

end prime_factors_converse_l681_681799


namespace math_problem_proof_l681_681932

noncomputable def solve_problem : Prop :=
  let product := ∏ k in finset.range 30, csc (3 * (k + 1) : ℝ) * π / 180
  ∃ (m n : ℕ), 1 < m ∧ 1 < n ∧ product = m ^ n ∧ m + n = 31

theorem math_problem_proof : solve_problem :=
sorry

end math_problem_proof_l681_681932


namespace volume_of_wall_is_16128_l681_681861

def wall_width : ℝ := 4
def wall_height : ℝ := 6 * wall_width
def wall_length : ℝ := 7 * wall_height

def wall_volume : ℝ := wall_length * wall_width * wall_height

theorem volume_of_wall_is_16128 :
  wall_volume = 16128 := by
  sorry

end volume_of_wall_is_16128_l681_681861


namespace girl_travel_distance_l681_681958

def speed : ℝ := 6 -- meters per second
def time : ℕ := 16 -- seconds

def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem girl_travel_distance : distance speed time = 96 :=
by 
  unfold distance
  sorry

end girl_travel_distance_l681_681958


namespace min_expression_value_l681_681584

theorem min_expression_value :
  ∃ x y : ℝ, (9 - x^2 - 8 * x * y - 16 * y^2 > 0) ∧ 
  (∀ x y : ℝ, 9 - x^2 - 8 * x * y - 16 * y^2 > 0 →
  (13 * x^2 + 24 * x * y + 13 * y^2 + 16 * x + 14 * y + 68) / 
  (9 - x^2 - 8 * x * y - 16 * y^2)^(5/2) = (7 / 27)) :=
sorry

end min_expression_value_l681_681584


namespace triangle_area_tangent_log2_l681_681358

open Real

noncomputable def log_base_2 (x : ℝ) : ℝ := log x / log 2

theorem triangle_area_tangent_log2 :
  let y := log_base_2
  let f := fun x : ℝ => y x
  let deriv := (deriv f 1)
  let tangent_line := fun x : ℝ => deriv * (x - 1) + f 1
  let x_intercept := 1
  let y_intercept := tangent_line 0
  
  (1 : ℝ) * (abs y_intercept) / 2 = 1 / (2 * log 2) := by
  sorry

end triangle_area_tangent_log2_l681_681358


namespace arithmetic_series_sum_l681_681123

theorem arithmetic_series_sum : 
  let a := 2 in 
  let d := 2 in 
  let n := 10 in 
  let l := 20 in 
  (a + l) * n / 2 = 110 := 
by
  sorry

end arithmetic_series_sum_l681_681123


namespace monotonicity_f_a_eq_1_domain_condition_inequality_condition_l681_681644

noncomputable def f (x a : ℝ) := (Real.log (x^2 - 2 * x + a)) / (x - 1)

theorem monotonicity_f_a_eq_1 :
  ∀ x : ℝ, 1 < x → 
  (f x 1 < f (e + 1) 1 → 
   ∀ y, 1 < y ∧ y < e + 1 → f y 1 < f (e + 1) 1) ∧ 
  (f (e + 1) 1 < f x 1 → 
   ∀ z, e + 1 < z → f (e + 1) 1 < f z 1) :=
sorry

theorem domain_condition (a : ℝ) :
  (∀ x : ℝ, (x < 1 ∨ x > 1) → x^2 - 2 * x + a > 0) ↔ a ≥ 1 :=
sorry

theorem inequality_condition (a : ℝ) :
  (∀ x : ℝ, 1 < x → (f x a < (x - 1) * Real.exp x)) ↔ (1 + 1 / Real.exp 1 ≤ a ∧ a ≤ 2) :=
sorry

end monotonicity_f_a_eq_1_domain_condition_inequality_condition_l681_681644


namespace no_prime_arithmetic_progression_l681_681176
open Nat

theorem no_prime_arithmetic_progression :
  ¬∃ (a₁ : Nat) (d : Nat), d ≠ 0 ∧ 
  (∀ n : Nat, Prime (a₁ + n * d)) :=
by
  sorry

end no_prime_arithmetic_progression_l681_681176


namespace area_of_triangle_MDA_l681_681270

theorem area_of_triangle_MDA (O A B M D : Type) (r : ℝ)
  (circle_center : O = O)
  (radius_r : ∀ P : Type, (dist O P = r) → (P = A ∨ P = B))
  (chord_AB : dist A B = r * real.sqrt 2)
  (OM_perpendicular_AB : (dist O M)^(2) = (dist O A)^(2) - (dist A M)^(2) ∧ dist O M = dist O B)
  (M_midpoint_AB : dist A M = dist B M ∧ dist A B = 2 * dist A M)
  (MD_perpendicular_OA : (dist M A)^2 + (dist A D)^2 = (dist M D)^2 ∧ ∠ M D A = 90) :
  ∃ (area : ℝ), area = (1 / 2) * (dist A D) * (dist D M) ∧ area = r^2 / 4 := 
sorry

end area_of_triangle_MDA_l681_681270


namespace length_BC_ratio_CD_BP_ratio_AC_BP_l681_681295

/-- Given triangle ABC with given angles and side length AB, prove that BC = sqrt(3) + 1 --/
theorem length_BC 
  (A B C : Type) [OrderedAddCommGroup A] [VectorCMSpace ℝ] 
  (AB : ℝ) (angle_B : ℝ) (angle_C : ℝ) 
  (hB : angle_B = 30) (hC : angle_C = 45) (hAB : AB = 2) :
  ∃ BC : ℝ, BC = sqrt(3) + 1 := 
sorry

/-- Given point P on AB, prove that CD/BP = 1/2 --/
theorem ratio_CD_BP 
  (A B C D P : Type) [OrderedAddCommGroup A] [VectorCMSpace ℝ] 
  (PA_perp_AD : Prop) (P_on_AB : Prop) :
  ∀ (CD BP : ℝ), CD / BP = 1 / 2 :=
sorry

/-- Given point P on perpendicular bisector of AB, prove that AC/BP = (1 + sqrt(3)) / 2 --/
theorem ratio_AC_BP 
  (A B C D P : Type) [OrderedAddCommGroup A] [VectorCMSpace ℝ] 
  (PA_perp_AD : Prop) (P_on_perpendicular_bisector : Prop) :
  ∀ (AC BP : ℝ), AC / BP = (1 + sqrt(3)) / 2 :=
sorry

end length_BC_ratio_CD_BP_ratio_AC_BP_l681_681295


namespace skew_lines_angle_range_l681_681386

theorem skew_lines_angle_range (θ : ℝ) (h_skew : θ > 0 ∧ θ ≤ 90) :
  0 < θ ∧ θ ≤ 90 :=
sorry

end skew_lines_angle_range_l681_681386


namespace part1_part2_l681_681292

noncomputable def seq (a : ℕ → ℚ) : Prop :=
  a 7 = 16 / 3 ∧ ∀ n >= 2, a n = (3 * a (n - 1) + 4) / (7 - a (n - 1))

theorem part1 (a : ℕ → ℚ) (h : seq a) : ∃ m : ℕ, (∀ n > m, a n < 2) ∧ (∀ n ≤ m, a n > 2) :=
sorry

theorem part2 (a : ℕ → ℚ) (h : seq a) : ∀ n ≥ 10, (a (n - 1) + a (n + 1)) / 2 < a n :=
sorry

end part1_part2_l681_681292


namespace number_of_triangles_with_perimeter_11_l681_681698

theorem number_of_triangles_with_perimeter_11 :
  {t : (ℕ × ℕ × ℕ) // let (a, b, c) := t in a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ c + a > b}.card = 4 :=
by sorry

end number_of_triangles_with_perimeter_11_l681_681698


namespace initial_price_after_markup_l681_681079

theorem initial_price_after_markup (P : ℝ) (h : 2 * P - 1.8 * P = 5) : 1.8 * P = 45 :=
by
  have h₁ : 0.2 * P = 5 := by linarith
  have h₂ : P = 25 := by linarith
  rw [h₂]
  norm_num
  sorry

end initial_price_after_markup_l681_681079


namespace geom_seq_sum_log_10_l681_681627

-- Define the geometric sequence and the conditions
variables {a : ℕ → ℝ} -- the geometric sequence {a_n}
variable (h_pos : ∀ n, 0 < a n) -- all terms are positive
variable (h_condition : a 5 * a 6 = 4) -- the given product condition

-- Define the sum of the first 10 terms of log2 {a_n}
def sum_log_sequence (a : ℕ → ℝ) : ℝ :=
  ∑ i in Finset.range 10, Real.log2 (a i)

-- The theorem stating the desired result
theorem geom_seq_sum_log_10 : sum_log_sequence a = 10 :=
by
  sorry

end geom_seq_sum_log_10_l681_681627


namespace specialIntegers_count_l681_681595

noncomputable def countSpecialIntegers : ℕ :=
  (finset.range 1000).filter (λ n, n % 3 = 0 ∧ nat.odd n) .card

theorem specialIntegers_count :
  countSpecialIntegers = 167 := 
sorry

end specialIntegers_count_l681_681595


namespace value_ranges_l681_681219

theorem value_ranges 
  (a b c : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_eq1 : 3 * a + 2 * b + c = 5)
  (h_eq2 : 2 * a + b - 3 * c = 1) :
  (3 / 7 ≤ c ∧ c ≤ 7 / 11) ∧ 
  (-5 / 7 ≤ (3 * a + b - 7 * c) ∧ (3 * a + b - 7 * c) ≤ -1 / 11) :=
by 
  sorry

end value_ranges_l681_681219


namespace matrix_multiplication_identity_l681_681582

theorem matrix_multiplication_identity :
  let M := ⟨λ i j, [[-7, -4, 0], [-5, -3, 0], [0, 0, 1]].nth i >>= list.nth j⟩ : Matrix (Fin 3) (Fin 3) ℝ
  let A := ⟨λ i j, [[-3, 4, 0], [5, -7, 0], [0, 0, 1]].nth i >>= list.nth j⟩ : Matrix (Fin 3) (Fin 3) ℝ
  let I := ⟨λ i j, [[1, 0, 0], [0, 1, 0], [0, 0, 1]].nth i >>= list.nth j⟩ : Matrix (Fin 3) (Fin 3) ℝ
  M ⬝ A = I :=
by {
  rw [Matrix.mul],
  -- further verification not required as per instruction
  sorry
}

end matrix_multiplication_identity_l681_681582


namespace circumcenter_concurrency_lines_circumcenter_of_medians_l681_681480

-- Definitions for the conditions
def triangle (A B C : Point) := true -- this is a placeholder; actual definition would be a set of three non-collinear points

variables (A B C O P A' B' C' A_1 B_1 C_1 : Point)
  (circumcenter : ∀ (A B C : Point), Point) -- takes three points of the triangle and returns the circumcenter
  (midpoint : ∀ (P Q : Point), Point) -- takes two points of a segment and returns the midpoint
  (symmetric : ∀ (P : Point) (AB : Point × Point), Point) -- takes a point and a line segment, returns the symmetric point

-- Assumptions
variables (H1 : O = circumcenter A B C)
variables (H2 : A' = symmetric O (B, C))
variables (H3 : B' = symmetric O (C, A))
variables (H4 : C' = symmetric O (A, B))
variables (H5 : A_1 = midpoint B C)
variables (H6 : B_1 = midpoint C A)
variables (H7 : C_1 = midpoint A B)

-- Prove the following:
theorem circumcenter_concurrency_lines :
  concurrent (line A A') (line B B') (line C C') := sorry

theorem circumcenter_of_medians :
  circumcenter A_1 B_1 C_1 = P := sorry

end circumcenter_concurrency_lines_circumcenter_of_medians_l681_681480


namespace distance_travelled_l681_681428

-- Definitions based on conditions
def first_distance : ℕ := 5
def acceleration : ℕ := 9
def time_interval : ℕ := 40

-- Theorem statement
theorem distance_travelled : 
  let sequence := λ (n : ℕ), first_distance + (n - 1) * acceleration in
  let total_distance := (time_interval * (sequence 1 + sequence time_interval)) / 2 in
  total_distance = 7220 := 
by {
  let sequence := λ (n : ℕ), first_distance + (n - 1) * acceleration;
  let total_distance := (time_interval * (sequence 1 + sequence time_interval)) / 2;
  show total_distance = 7220;
  :: sorry
}

end distance_travelled_l681_681428


namespace hotel_profit_calculation_l681_681450

theorem hotel_profit_calculation
  (operations_expenses : ℝ)
  (meetings_fraction : ℝ) (events_fraction : ℝ) (rooms_fraction : ℝ)
  (meetings_tax_rate : ℝ) (meetings_commission_rate : ℝ)
  (events_tax_rate : ℝ) (events_commission_rate : ℝ)
  (rooms_tax_rate : ℝ) (rooms_commission_rate : ℝ)
  (total_profit : ℝ) :
  operations_expenses = 5000 →
  meetings_fraction = 5/8 →
  events_fraction = 3/10 →
  rooms_fraction = 11/20 →
  meetings_tax_rate = 0.10 →
  meetings_commission_rate = 0.05 →
  events_tax_rate = 0.08 →
  events_commission_rate = 0.06 →
  rooms_tax_rate = 0.12 →
  rooms_commission_rate = 0.03 →
  total_profit = (operations_expenses * (meetings_fraction + events_fraction + rooms_fraction)
                - (operations_expenses
                  + operations_expenses * (meetings_fraction * (meetings_tax_rate + meetings_commission_rate)
                  + events_fraction * (events_tax_rate + events_commission_rate)
                  + rooms_fraction * (rooms_tax_rate + rooms_commission_rate)))) ->
  total_profit = 1283.75 :=
by sorry

end hotel_profit_calculation_l681_681450


namespace paths_spell_contests_l681_681534
-- Import the necessary library

open_locale classical

-- Define the conditions as given in the problem statement
def triangular_grid := [
  ['C', 'O', 'N', 'T', 'E', 'S', 'T', 'S', 'C'],
  ['T', 'E', 'T', 'N', 'T', 'E', 'T'],
  ['S', 'E', 'S', 'E', 'S'],
  ['T', 'E', 'T'],
  ['S']
]

-- Define the main theorem statement
theorem paths_spell_contests : 
  let count_paths (start: char) (end: char) : ℕ := 
    if start = end then 1 else 2 * count_paths start end
  in 
  count_paths 'C' 'S' = 256 := 
begin
  sorry,
end

end paths_spell_contests_l681_681534


namespace intersection_eq_T_l681_681262

noncomputable def S : Set ℝ := { y : ℝ | ∃ x : ℝ, y = 3 * x + 2 }
noncomputable def T : Set ℝ := { y : ℝ | ∃ x : ℝ, y = x ^ 2 - 1 }

theorem intersection_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_eq_T_l681_681262


namespace math_problem_proof_l681_681931

noncomputable def solve_problem : Prop :=
  let product := ∏ k in finset.range 30, csc (3 * (k + 1) : ℝ) * π / 180
  ∃ (m n : ℕ), 1 < m ∧ 1 < n ∧ product = m ^ n ∧ m + n = 31

theorem math_problem_proof : solve_problem :=
sorry

end math_problem_proof_l681_681931


namespace ratio_of_square_areas_l681_681350

-- Define the side lengths of the squares
variable (y : ℝ) (h_y : y > 0)

-- Define the area of square C
def area_square_C : ℝ := y * y

-- Define the area of square D
def area_square_D : ℝ := (3 * y) * (3 * y)

-- Define the ratio of areas
def ratio_of_areas : ℝ := area_square_C y / area_square_D y

-- State the theorem
theorem ratio_of_square_areas (y : ℝ) (h_y : y > 0) : ratio_of_areas y = 1/9 :=
by
  -- Apply sorry to denote the proof is omitted for now
  sorry

end ratio_of_square_areas_l681_681350


namespace arithmetic_series_sum_l681_681121

theorem arithmetic_series_sum : 
  let a := 2 in 
  let d := 2 in 
  let n := 10 in 
  let l := 20 in 
  (a + l) * n / 2 = 110 := 
by
  sorry

end arithmetic_series_sum_l681_681121


namespace base_9_perfect_square_b_l681_681263

theorem base_9_perfect_square_b (b : ℕ) (a : ℕ) 
  (h0 : 0 < b) (h1 : b < 9) (h2 : a < 9) : 
  ∃ n, n^2 ≡ 729 * b + 81 * a + 54 [MOD 81] :=
sorry

end base_9_perfect_square_b_l681_681263


namespace sum_increase_even_positions_l681_681609

noncomputable def geometric_progression (b : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, b (n + 1) = b n * q

theorem sum_increase_even_positions
  (b : ℕ → ℝ)
  (q : ℝ)
  (S : ℝ)
  (hb : ∀ n, b n > 0)
  (hg : geometric_progression b q)
  (hS : S = b 0 * (1 - q^3000) / (1 - q))
  (hmultiple3 : S + 49 * (b 0) * q^2 * ((1 - q^3000) / (1 - q^3)) = 10 * S) :
  let new_S := S + (2 * b 0) * q * ((1 - q^3000) / (1 - q^2))
  in new_S = (11 / 8) * S :=
begin
  sorry -- Proof goes here
end

end sum_increase_even_positions_l681_681609


namespace grill_run_time_l681_681431

-- Define the conditions 1, 2, and 3
def burns_rate : ℕ := 15 -- coals burned every 20 minutes
def burns_time : ℕ := 20 -- minutes to burn some coals
def coals_per_bag : ℕ := 60 -- coals per bag
def num_bags : ℕ := 3 -- number of bags

-- The main theorem to prove the time taken to burn all the coals
theorem grill_run_time : 
  let total_coals := num_bags * coals_per_bag in
  let burn_time_per_coals := total_coals * burns_time / burns_rate in
  burn_time_per_coals / 60 = 4 := 
by
  sorry

end grill_run_time_l681_681431


namespace tangent_sum_equals_external_tangent_property_internal_tangent_property_l681_681601

noncomputable def is_tangent (P A B : Point) (C : Circle) : Prop := 
 -- define the tangency property: PA and PB are tangents to C

noncomputable def is_equilateral (A B C : Point) : Prop :=
 -- define equilateral triangle property

theorem tangent_sum_equals
  (a b c : Circle) (ABC : Triangle)
  (ha : is_equilateral ABC) (r : ℝ)
  (d : Circle) (hd : d.tangent_internally a b c)
  (P : Point) (P_on_d : d.contains P) :
  ∀ (PA PB PC : Segment), 
  is_tangent P PA a ∧ is_tangent P PB b ∧ is_tangent P PC c →
  (PA.length + PB.length = PC.length) := 
sorry

theorem external_tangent_property
  (a b c : Circle) (ABC : Triangle)
  (ha : is_equilateral ABC) (r : ℝ)
  (d_star : Circle) (hd_star : d_star.tangent_externally a b c)
  (P_star : Point) (P_star_on_d : d_star.contains P_star) :
  ∀ (PA_star PB_star PC_star : Segment), 
  is_tangent P_star PA_star a ∧ is_tangent P_star PB_star b ∧ is_tangent P_star PC_star c →
  (PA_star.length + PB_star.length = PC_star.length) := 
sorry

theorem internal_tangent_property
  (a b c : Circle) (ABC : Triangle)
  (ha : is_equilateral ABC) (r : ℝ)
  (d_internal : Circle) (hd_internal : d_internal.tangent_internally a b c)
  (P_internal : Point) (P_internal_on_d : d_internal.contains P_internal) :
  ∀ (PA_internal PB_internal PC_internal : Segment), 
  is_tangent P_internal PA_internal a ∧ is_tangent P_internal PB_internal b ∧ is_tangent P_internal PC_internal c →
  (PA_internal.length + PB_internal.length = PC_internal.length) := 
sorry

end tangent_sum_equals_external_tangent_property_internal_tangent_property_l681_681601


namespace profit_percentage_is_25_percent_l681_681063

noncomputable def costPrice : ℝ := 47.50
noncomputable def markedPrice : ℝ := 64.54
noncomputable def discountRate : ℝ := 0.08

noncomputable def discountAmount : ℝ := discountRate * markedPrice
noncomputable def sellingPrice : ℝ := markedPrice - discountAmount
noncomputable def profit : ℝ := sellingPrice - costPrice
noncomputable def profitPercentage : ℝ := (profit / costPrice) * 100

theorem profit_percentage_is_25_percent :
  profitPercentage = 25 := by
  sorry

end profit_percentage_is_25_percent_l681_681063


namespace calculate_sum_of_squares_l681_681312

variables {p q r : ℝ}

-- Define the polynomial and specify the roots
def polynomial := (x : ℝ) ^ 3 - 15 * x ^ 2 + 25 * x - 10

-- Assume p, q, and r are roots of the polynomial
def is_root (x : ℝ) : Prop := polynomial x = 0

-- Given conditions as hypothesis
axiom root_p : is_root p
axiom root_q : is_root q
axiom root_r : is_root r

theorem calculate_sum_of_squares (h1 : p + q + r = 15) (h2 : p * q + q * r + r * p = 25) : (p + q)^2 + (q + r)^2 + (r + p)^2 = 400 := 
sorry


end calculate_sum_of_squares_l681_681312


namespace sum_cubes_gt_40_l681_681878

theorem sum_cubes_gt_40 {n : ℕ} (a : fin n → ℝ) (h_pos : ∀ i, 0 < a i) 
  (h_sum : ∑ i, a i = 10) (h_sum_sq : ∑ i, (a i)^2 > 20) : 
  ∑ i, (a i)^3 > 40 :=
by
  sorry

end sum_cubes_gt_40_l681_681878


namespace floor_of_47_l681_681561

theorem floor_of_47 : int.floor 4.7 = 4 :=
sorry

end floor_of_47_l681_681561


namespace convert_to_scientific_notation_l681_681818

theorem convert_to_scientific_notation :
  (26.62 * 10^9) = 2.662 * 10^9 :=
by
  sorry

end convert_to_scientific_notation_l681_681818


namespace example_problem_l681_681308

theorem example_problem 
  (x : ℝ) (y : ℝ)
  (hx : x = (2001 : ℝ) ^ 1002 - (2001 : ℝ) ^ (-1002))
  (hy : y = (2001 : ℝ) ^ 1002 + (2001 : ℝ) ^ (-1002)) :
  x^2 - y^2 = -4 := by
  sorry

end example_problem_l681_681308


namespace combination_18_6_l681_681523

theorem combination_18_6 : nat.choose 18 6 = 18564 :=
by {
  sorry
}

end combination_18_6_l681_681523


namespace race_time_difference_l681_681318

theorem race_time_difference : 
  ∀ (s_m s_j d : ℕ),
  s_m = 7 ∧ s_j = 8 ∧ d = 12 →
  (s_j * d) - (s_m * d) = 12 :=
by
  intros s_m s_j d h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h3, h4, h1]
  calc
    8 * 12 - 7 * 12 = (8 - 7) * 12 : by ring
                ... = 1 * 12        : by norm_num
                ... = 12            : by norm_num

end race_time_difference_l681_681318


namespace g_diff_l681_681773

noncomputable section

-- Definition of g(n) as given in the problem statement
def g (n : ℕ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((1 + Real.sqrt 3) / 2)^n +
  (3 - 2 * Real.sqrt 3) / 6 * ((1 - Real.sqrt 3) / 2)^n

-- The statement to prove g(n+2) - g(n) = -1/4 * g(n)
theorem g_diff (n : ℕ) : g (n + 2) - g n = -1 / 4 * g n :=
by
  sorry

end g_diff_l681_681773


namespace value_of_f_1_l681_681858

noncomputable def f : ℝ → ℝ := sorry

axiom increasing_on {a b : ℝ} (h : 0 < a → b ≤ +∞) : (∀ x y, f x < f y → x < y)

axiom func_nat_star (n : ℕ) (h : 0 < n) : (f n : ℕ) = f n

axiom func_property (n : ℕ) (h : 0 < n) : f (f n) = 3 * n

theorem value_of_f_1 : f 1 = 2 :=
by sorry

end value_of_f_1_l681_681858


namespace batsman_average_after_11th_inning_l681_681409

variable (x : ℝ) -- The average before the 11th inning
variable (new_average : ℝ) -- The average after the 11th inning
variable (total_runs : ℝ) -- Total runs scored after 11 innings

-- Given conditions
def condition1 := total_runs = 11 * (x + 5)
def condition2 := total_runs = 10 * x + 110

theorem batsman_average_after_11th_inning : 
  ∀ (x : ℝ), 
    (x = 55) → (x + 5 = 60) :=
by
  intros
  sorry

end batsman_average_after_11th_inning_l681_681409


namespace james_hours_to_work_l681_681767

theorem james_hours_to_work :
  let meat_cost := 20 * 5
  let fruits_vegetables_cost := 15 * 4
  let bread_cost := 60 * 1.5
  let janitorial_cost := 10 * (10 * 1.5)
  let total_cost := meat_cost + fruits_vegetables_cost + bread_cost + janitorial_cost
  let hourly_wage := 8
  let hours_to_work := total_cost / hourly_wage
  hours_to_work = 50 :=
by 
  sorry

end james_hours_to_work_l681_681767


namespace cube_partition_has_31_cubes_l681_681067

theorem cube_partition_has_31_cubes :
  ∃ (N : ℕ), (∀ (cubes : list ℕ), -- cube edge lengths in cm
    (∀ x ∈ cubes, x ∈ [1, 2, 3]) ∧ -- edge lengths are 1, 2, or 3
    (list.sum (cubes.map (λ a, a ^ 3)) = 64) ∧ -- total volume sums to 64 cm³
    (¬ list.all (λ a, a = 1) cubes) ∧ -- not all cubes are of size 1
    list.length cubes = N) ∧ 
    N = 31 :=
begin
  sorry
end

end cube_partition_has_31_cubes_l681_681067


namespace cary_net_calorie_deficit_is_250_l681_681147

-- Define the conditions
def miles_walked : ℕ := 3
def candy_bar_calories : ℕ := 200
def calories_per_mile : ℕ := 150

-- Define the function to calculate total calories burned
def total_calories_burned (miles : ℕ) (calories_per_mile : ℕ) : ℕ :=
  miles * calories_per_mile

-- Define the function to calculate net calorie deficit
def net_calorie_deficit (total_calories : ℕ) (candy_calories : ℕ) : ℕ :=
  total_calories - candy_calories

-- The statement to be proven
theorem cary_net_calorie_deficit_is_250 :
  net_calorie_deficit (total_calories_burned miles_walked calories_per_mile) candy_bar_calories = 250 :=
  by sorry

end cary_net_calorie_deficit_is_250_l681_681147


namespace max_intersections_circle_quadrilateral_max_intersections_correct_l681_681924

-- Define the intersection property of a circle and a line segment
def max_intersections_per_side (circle : Type) (line_segment : Type) : ℕ := 2

-- Define a quadrilateral as a shape having four sides
def sides_of_quadrilateral : ℕ := 4

-- The theorem stating the maximum number of intersection points between a circle and a quadrilateral
theorem max_intersections_circle_quadrilateral (circle : Type) (quadrilateral : Type) : Prop :=
  max_intersections_per_side circle quadrilateral * sides_of_quadrilateral = 8

-- Proof is skipped with 'sorry'
theorem max_intersections_correct (circle : Type) (quadrilateral : Type) :
  max_intersections_circle_quadrilateral circle quadrilateral :=
by
  sorry

end max_intersections_circle_quadrilateral_max_intersections_correct_l681_681924


namespace max_length_OC_is_correct_l681_681744

noncomputable def max_length_OC : ℝ :=
  let A := (2, 0)
  let B (θ : ℝ) (hθ1 : 0 ≤ θ) (hθ2 : θ ≤ π) := (Real.cos θ, Real.sin θ)
  let C (θ : ℝ) (hθ1 : 0 ≤ θ) (hθ2 : θ ≤ π) := (2 + Real.sin θ, 2 - Real.cos θ)
  let OC_length (θ : ℝ) (hθ1 : 0 ≤ θ) (hθ2 : θ ≤ π) :=
    Real.sqrt ((2 + Real.sin θ)^2 + (2 - Real.cos θ)^2)
  Real.max (OC_length 0 ⟨le_refl 0, le_of_lt Real.pi_pos⟩) (Real.max (OC_length π ⟨zero_le_pi, le_refl π⟩) (OC_length (3 * π / 4) ⟨le_of_lt (div_pos (mul_pos (by norm_num1 [pi_pos]) (by norm_num1)) pi_pos), le_of_lt (div_pos (mul_pos three_pos pi_pos) (mul_pos two_pos (mul_pos one_div_two_pos pi_pos)))⟩))

theorem max_length_OC_is_correct : max_length_OC = 1 + 2 * Real.sqrt 2 :=
  sorry

end max_length_OC_is_correct_l681_681744


namespace marlene_total_payment_l681_681321

def price_shirt : ℝ := 50
def price_pant : ℝ := 40
def price_shoe : ℝ := 60
def discount_shirt : ℝ := 0.20
def pants_to_pay : ℕ := 2
def full_price_shoes_count : ℕ := 2
def discount_shoe : ℝ := 0.50
def sales_tax : ℝ := 0.08
def number_of_shirts : ℕ := 6
def number_of_pants : ℕ := 4
def number_of_shoes : ℕ := 3

def cost_shirts : ℝ := (number_of_shirts * price_shirt) * (1 - discount_shirt)
def cost_pants : ℝ := (number_of_pants / 2 * price_pant)
def cost_shoes : ℝ := (full_price_shoes_count * price_shoe) + (discount_shoe * price_shoe)
def total_cost_before_tax : ℝ := cost_shirts + cost_pants + cost_shoes
def total_cost_including_tax : ℝ := total_cost_before_tax * (1 + sales_tax)

theorem marlene_total_payment : total_cost_including_tax = 507.60 :=
  by
  sorry

end marlene_total_payment_l681_681321


namespace association_test_dist_table_correctness_l681_681008

-- Definitions for the conditions given in the problem
def total_students : ℕ := 2000
def myopia_percentage : ℚ := 0.4
def phone_usage_percentage : ℚ := 0.2
def myopia_rate_usage : ℚ := 0.5

-- Calculation based on conditions
def students_with_myopia : ℕ := (myopia_percentage * total_students).toNat
def students_exceeding_1_hour : ℕ := (phone_usage_percentage * total_students).toNat
def myopic_students_exceeding_1_hour : ℕ := (myopia_rate_usage * students_exceeding_1_hour).toNat
def students_less_than_1_hour : ℕ := total_students - students_exceeding_1_hour
def myopic_students_less_than_1_hour : ℕ := students_with_myopia - myopic_students_exceeding_1_hour

-- Corresponding Chi-square statistic definition
def chi_square_statistic (a b c d n : ℕ) : ℚ :=
  (n * ((a * d - b * c)^2) : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d) : ℚ)

-- Values to plug into the chi-square formula
def χ2 : ℚ := chi_square_statistic myopic_students_exceeding_1_hour
                                             (students_exceeding_1_hour - myopic_students_exceeding_1_hour)
                                             myopic_students_less_than_1_hour
                                             (students_less_than_1_hour - myopic_students_less_than_1_hour)
                                             total_students

-- Assertion for chi-square test result and independence determination
theorem association_test : χ2 > 10.828 := by
  -- Calculated χ2 value: 20.833
  have χ2_calculated : χ2 = 20.833 := by sorry
  exact by linarith

-- Distribution calculation for random variable X (step 2 of problem)
def binomial_coefficient (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.choose n k

def P_X_eq_0 : ℚ := (binomial_coefficient 6 3 : ℚ) / (binomial_coefficient 8 3)
def P_X_eq_1 : ℚ := (binomial_coefficient 6 2 * binomial_coefficient 2 1 : ℚ) / (binomial_coefficient 8 3)
def P_X_eq_2 : ℚ := (binomial_coefficient 6 1 * binomial_coefficient 2 2 : ℚ) / (binomial_coefficient 8 3)

-- Assertion for the distribution table correctness
theorem dist_table_correctness : P_X_eq_0 = 5/14 ∧ 
                                  P_X_eq_1 = 15/28 ∧
                                  P_X_eq_2 = 3/28 := by
  -- Individual probabilities calculation (Use sorry to skip proof)
  have h0 : P_X_eq_0 = 5/14 := by sorry
  have h1 : P_X_eq_1 = 15/28 := by sorry
  have h2 : P_X_eq_2 = 3/28 := by sorry
  exact ⟨h0, h1, h2⟩

end association_test_dist_table_correctness_l681_681008


namespace vector_parallel_has_value_x_l681_681679

-- Define the vectors a and b
def a : ℝ × ℝ := (3, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the parallel condition
def parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

-- The theorem statement
theorem vector_parallel_has_value_x :
  ∀ x : ℝ, parallel a (b x) → x = 6 :=
by
  intros x h
  sorry

end vector_parallel_has_value_x_l681_681679


namespace floor_of_4point7_l681_681545

theorem floor_of_4point7 : (Real.floor 4.7) = 4 :=
by
  sorry

end floor_of_4point7_l681_681545


namespace arithmetic_sequence_sum_l681_681132

theorem arithmetic_sequence_sum :
  let sequence := list.range (20 / 2) in
  let sum := sequence.map (λ n, 2 * (n + 1)).sum in
  sum = 110 :=
by
  -- Define the sequence as the arithmetic series
  let sequence := list.range (20 / 2)
  -- Calculate the sum of the arithmetic sequence
  let sum := sequence.map (λ n, 2 * (n + 1)).sum
  -- Check the sum
  have : sum = 110 := sorry
  exact this

end arithmetic_sequence_sum_l681_681132


namespace min_gx1_gx2_l681_681232

noncomputable def f (x a : ℝ) : ℝ := x - (1 / x) - a * Real.log x
noncomputable def g (x a : ℝ) : ℝ := x - (a / 2) * Real.log x

theorem min_gx1_gx2 (x1 x2 a : ℝ) (h1 : 0 < x1 ∧ x1 < Real.exp 1) (h2 : 0 < x2) (hx1x2: x1 * x2 = 1) (ha : a > 0) :
  f x1 a = 0 ∧ f x2 a = 0 →
  g x1 a - g x2 a = -2 / Real.exp 1 :=
by sorry

end min_gx1_gx2_l681_681232


namespace integer_solution_count_l681_681866

theorem integer_solution_count :
  {x : ℤ // (x^2 - x - 1)^(x + 2) = 1}.card = 4 :=
sorry

end integer_solution_count_l681_681866


namespace transform_correct_l681_681477

variable {α : Type} [Mul α] [DecidableEq α]

theorem transform_correct (a b c : α) (h : a = b) : a * c = b * c :=
by sorry

end transform_correct_l681_681477


namespace find_multiplier_l681_681037

theorem find_multiplier (x : ℝ) (h : (9 / 6) * x = 18) : x = 12 := sorry

end find_multiplier_l681_681037


namespace distinct_real_nums_condition_l681_681307

theorem distinct_real_nums_condition 
  (p q r : ℝ) (h1 : p ≠ q) (h2 : q ≠ r) (h3 : r ≠ p)
  (h4 : p / (q - r) + q / (r - p) + r / (p - q) = 1) :
  p^2 / (q - r)^2 + q^2 / (r - p)^2 + r^2 / (p - q)^2 = 0 :=
by
  sorry

end distinct_real_nums_condition_l681_681307


namespace sum_of_integers_l681_681392

-- Defining the two integers
variables (L S : ℤ)

-- Given conditions
def condition1 : Prop := S = 10
def condition2 : Prop := 2 * L = 5 * S - 10

-- The final statement to prove
theorem sum_of_integers : condition1 → condition2 → L + S = 30 := by
  intros h1 h2
  sorry

end sum_of_integers_l681_681392


namespace binom_18_6_eq_4765_l681_681531

def binom (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_18_6_eq_4765 : binom 18 6 = 4765 := by
  sorry

end binom_18_6_eq_4765_l681_681531


namespace sum_arithmetic_sequence_l681_681116

theorem sum_arithmetic_sequence : ∀ (a d l : ℕ), 
  (d = 2) → (a = 2) → (l = 20) → 
  ∃ (n : ℕ), (l = a + (n - 1) * d) ∧ 
  (∑ k in Finset.range n, (a + k * d)) = 110 :=
by
  intros a d l h_d h_a h_l
  use 10
  split
  · sorry
  · sorry

end sum_arithmetic_sequence_l681_681116


namespace time_per_piece_of_furniture_l681_681180

theorem time_per_piece_of_furniture:
  (4 + 2 > 0) → 48 / (4 + 2) = 8 :=
by
  intros h
  have h1 : 4 + 2 = 6 := rfl
  have h2 : 48 / 6 = 8 := rfl
  rw [←h1] at h2
  exact h2

end time_per_piece_of_furniture_l681_681180


namespace boxes_needed_l681_681316

def num_red_pencils := 45
def num_yellow_pencils := 80
def num_pencils_per_red_box := 15
def num_pencils_per_blue_box := 25
def num_pencils_per_yellow_box := 10
def num_pencils_per_green_box := 30

def num_blue_pencils (x : Nat) := 3 * x + 6
def num_green_pencils (red : Nat) (blue : Nat) := 2 * (red + blue)

def total_boxes_needed : Nat :=
  let red_boxes := num_red_pencils / num_pencils_per_red_box
  let blue_boxes := (num_blue_pencils num_red_pencils) / num_pencils_per_blue_box + 
                    if ((num_blue_pencils num_red_pencils) % num_pencils_per_blue_box) = 0 then 0 else 1
  let yellow_boxes := num_yellow_pencils / num_pencils_per_yellow_box
  let green_boxes := (num_green_pencils num_red_pencils (num_blue_pencils num_red_pencils)) / num_pencils_per_green_box + 
                     if ((num_green_pencils num_red_pencils (num_blue_pencils num_red_pencils)) % num_pencils_per_green_box) = 0 then 0 else 1
  red_boxes + blue_boxes + yellow_boxes + green_boxes

theorem boxes_needed : total_boxes_needed = 30 := sorry

end boxes_needed_l681_681316


namespace sqrt_t6_plus_t4_eq_t2_sqrt_t2_plus_1_l681_681563

theorem sqrt_t6_plus_t4_eq_t2_sqrt_t2_plus_1 (t : ℝ) : 
  Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) :=
sorry

end sqrt_t6_plus_t4_eq_t2_sqrt_t2_plus_1_l681_681563


namespace half_abs_difference_squares_l681_681020

theorem half_abs_difference_squares :
  (1 / 2) * |(15^2 - 13^2)| = 28 :=
by
  sorry

end half_abs_difference_squares_l681_681020


namespace area_of_larger_triangle_l681_681422

theorem area_of_larger_triangle {S : ℝ} 
  (h_similar : true) -- indicating the triangles are similar
  (h_ratio : (2 / 1)) 
  (h_sum_of_areas : S + (S / 4) = 25) :
  S = 20 := 
by 
  -- include essential steps taken from the solution
  have h_ratio_S : S / (S / 4) = 4, from by calc
    S / (S / 4) = 4   : by sorry,
  calc
    S + (S / 4) = 25  : by sorry
    ... = 20          : by sorry 

  sorry

end area_of_larger_triangle_l681_681422


namespace parabola_decreasing_m_geq_neg2_l681_681236

theorem parabola_decreasing_m_geq_neg2 (m : ℝ) :
  (∀ x ≥ 2, ∃ y, y = -5 * (x + m)^2 - 3 ∧ (∀ x1 y1, x1 ≥ 2 → y1 = -5 * (x1 + m)^2 - 3 → y1 ≤ y)) →
  m ≥ -2 := 
by
  intro h
  sorry

end parabola_decreasing_m_geq_neg2_l681_681236


namespace greatest_base8_three_digit_divisible_by_7_l681_681905

theorem greatest_base8_three_digit_divisible_by_7 :
  ∃ n : ℕ, n < 8^3 ∧ n ≥ 8^2 ∧ (n % 7 = 0) ∧ (to_base 8 n = 777) :=
sorry

end greatest_base8_three_digit_divisible_by_7_l681_681905


namespace broken_line_perimeter_eq_triangle_perimeter_l681_681814

-- Define the triangle and its properties
variables {A B C A1 B1 C1 A2 B2 C2 : Point}
variables {a b c : ℝ}

-- Define the conditions
def midpoint (A B P : Point) : Prop := dist P A = dist P B
def foot_of_altitude (A B C P : Point) : Prop := ∠ BPA = 90

-- The problem statement in Lean 4
theorem broken_line_perimeter_eq_triangle_perimeter (hA1 : midpoint B C A1) (hB1 : midpoint C A B1) (hC1 : midpoint A B C1)
  (hA2 : foot_of_altitude A B C A2) (hB2 : foot_of_altitude B C A B2) (hC2 : foot_of_altitude C A B C2)
  (hp : dist A B = a ∧ dist B C = b ∧ dist C A = c) :
  dist A1 B2 + dist B2 C1 + dist C1 A2 + dist A2 B1 + dist B1 C2 + dist C2 A1 = a + b + c :=
sorry

end broken_line_perimeter_eq_triangle_perimeter_l681_681814


namespace cos_theta_solution_of_convex_quadrilateral_l681_681279

theorem cos_theta_solution_of_convex_quadrilateral (AB CD : ℝ) (AD BC : ℝ) (angleA angleC perimeter : ℝ) (cos_theta : ℝ) :
  convex_quadrilateral ABCD ∧
  angleA = 2 * angleC ∧
  AB = CD ∧
  AB = 200 ∧
  CD = 200 ∧
  AD ≠ BC ∧
  AB + BC + CD + AD = perimeter ∧
  perimeter = 720 ∧
  cos 2 * angleC = 2 * cos(angleC)^2 - 1 → 
  cos (angleC) = cos_theta := 
sorry

end cos_theta_solution_of_convex_quadrilateral_l681_681279


namespace greatest_3digit_base8_divisible_by_7_l681_681903

def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : ℕ), n = 0b777 ∧ (base8_to_base10 0b777) % 7 = 0 ∧ ∀ m < 0o777, m % 7 = 0 → base8_to_base10 m < base8_to_base10 0b777 :=
by
  sorry

end greatest_3digit_base8_divisible_by_7_l681_681903


namespace max_street_lamps_l681_681460
-- Importing the necessary library

-- Lean 4 statement for the math proof problem
theorem max_street_lamps (road_length lamp_illumination : ℕ) (H1 : road_length = 1000) (H2 : lamp_illumination = 1) (H3 : ∀ n : ℕ, n ≤ 1998 → (turn_off_lamp n → ¬ fully_illuminated road_length lamp_illumination 1998)) :
  max_lamps road_length lamp_illumination = 1998 :=
sorry

-- Define the predicates used
def fully_illuminated (road_length lamp_illumination total_lamps : ℕ) : Prop := 
  road_length <= total_lamps * lamp_illumination

def turn_off_lamp (n : ℕ) : Prop :=
  -- Placeholder for turning off the lamp number n
  sorry

def max_lamps (road_length lamp_illumination : ℕ) : ℕ :=
  find_max_lamps road_length lamp_illumination 0 0  -- Placeholder for the actual implementation

-- Finding the maximum number of lamps that can be placed
def find_max_lamps (road_length lamp_illumination curr_lamps max_lamps : ℕ) : ℕ :=
  if curr_lamps <= road_length then
    find_max_lamps road_length lamp_illumination (curr_lamps + 1) (max curr_lamps max_lamps)
  else max_lamps

end max_street_lamps_l681_681460


namespace pyramid_height_l681_681441

theorem pyramid_height (h : ℝ) :
  let V_cube := 5^3
  let V_pyramid := (1/3) * 10^2 * h
  V_cube = V_pyramid → h = 3.75 :=
by
  let V_cube := 5^3
  let V_pyramid := (1/3) * 10^2 * h
  intros h_eq
  sorry

end pyramid_height_l681_681441


namespace enumerate_A_l681_681674

-- Define the set A according to the given conditions
def A : Set ℕ := {X : ℕ | 8 % (6 - X) = 0}

-- The equivalent proof problem
theorem enumerate_A : A = {2, 4, 5} :=
by sorry

end enumerate_A_l681_681674


namespace cone_volume_ratio_l681_681405

theorem cone_volume_ratio (r_C h_C r_D h_D : ℝ) (h_rC : r_C = 20) (h_hC : h_C = 40) 
  (h_rD : r_D = 40) (h_hD : h_D = 20) : 
  (1 / 3 * pi * r_C^2 * h_C) / (1 / 3 * pi * r_D^2 * h_D) = 1 / 2 :=
by
  rw [h_rC, h_hC, h_rD, h_hD]
  sorry

end cone_volume_ratio_l681_681405


namespace mike_spent_on_car_parts_l681_681594

-- Define the costs as constants
def cost_speakers : ℝ := 118.54
def cost_tires : ℝ := 106.33
def cost_cds : ℝ := 4.58

-- Define the total cost of car parts excluding the CDs
def total_cost_car_parts : ℝ := cost_speakers + cost_tires

-- The theorem we want to prove
theorem mike_spent_on_car_parts :
  total_cost_car_parts = 224.87 := 
by 
  -- Proof omitted
  sorry

end mike_spent_on_car_parts_l681_681594


namespace corresponding_lines_count_l681_681170

open Set Function

noncomputable def f : ℝ × ℝ → ℝ × ℝ := λ P, (3 * P.2, 2 * P.1)

-- Define "corresponding line" for the mapping f.
def is_corresponding_line (l : ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), (l = λ x, k * x) ∧ (k = sqrt (2 / 3) ∨ k = - sqrt (2 / 3))

theorem corresponding_lines_count :
  (∃ (l1 l2 : ℝ → ℝ), l1 ≠ l2 ∧ is_corresponding_line l1 ∧ is_corresponding_line l2) ∧
  (∀ l, is_corresponding_line l → (l = λ x, sqrt (2 / 3) * x ∨ l = λ x, - sqrt (2 / 3) * x)) :=
by
  sorry

end corresponding_lines_count_l681_681170


namespace segment_AB_length_l681_681752

-- Defining the conditions
def area_ratio (AB CD : ℝ) : Prop := AB / CD = 5 / 2
def length_sum (AB CD : ℝ) : Prop := AB + CD = 280

-- The theorem stating the problem
theorem segment_AB_length (AB CD : ℝ) (h₁ : area_ratio AB CD) (h₂ : length_sum AB CD) : AB = 200 :=
by {
  -- Proof step would be inserted here, but it is omitted as per instructions
  sorry
}

end segment_AB_length_l681_681752


namespace floor_of_47_l681_681560

theorem floor_of_47 : int.floor 4.7 = 4 :=
sorry

end floor_of_47_l681_681560


namespace fraction_of_time_spent_covering_initial_distance_l681_681110

variables (D T : ℝ) (h1 : T = ((2 / 3) * D) / 80 + ((1 / 3) * D) / 40)

theorem fraction_of_time_spent_covering_initial_distance (h1 : T = ((2 / 3) * D) / 80 + ((1 / 3) * D) / 40) :
  ((2 / 3) * D / 80) / T = 1 / 2 :=
by
  sorry

end fraction_of_time_spent_covering_initial_distance_l681_681110


namespace find_k_l681_681235

-- Define the line l: kx - y - 3 = 0
def line_eq (k : ℝ) (x y : ℝ) : Prop := k * x - y - 3 = 0

-- Define the circle O: x^2 + y^2 = 4
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Assume the intersection points A and B for line and circle
def intersects (k : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, line_eq k A.1 A.2 ∧ line_eq k B.1 B.2 ∧ circle_eq A.1 A.2 ∧ circle_eq B.1 B.2

-- Assume the dot product condition
def dot_product_condition (A B : ℝ × ℝ) (OA OB : ℝ × ℝ) : Prop :=
  (OA.1 * OB.1 + OA.2 * OB.2 = 2)

-- The main theorem we need to prove
theorem find_k : ∀ k : ℝ,
  (∃ A B : ℝ × ℝ, intersects k ∧ dot_product_condition A B ⟨0, 0⟩ ⟨0, 0⟩) → 
  (k = sqrt 2 ∨ k = -sqrt 2) :=
by
  sorry

end find_k_l681_681235


namespace smallest_number_among_given_l681_681095

theorem smallest_number_among_given :
  ∀ (a b c d : ℚ), a = -2 → b = -5/2 → c = 0 → d = 1/5 →
  (min (min (min a b) c) d) = b :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  sorry

end smallest_number_among_given_l681_681095


namespace triangle_area_l681_681233

noncomputable def hyperbola : Set (ℝ × ℝ) := {p | (p.1^2 / 9) - (p.2^2 / 27) = 1}
noncomputable def parabola (p : ℝ) : Set (ℝ × ℝ) := {pt | pt.2^2 = 2 * p * pt.1}

theorem triangle_area
  (p : ℝ)
  (hp : p = 12)
  (F1 F2 P : ℝ × ℝ)
  (hF1 : ∀ x y, (x, y) ∈ hyperbola → (x, y) = F1 ∨ ∃ a b, F1 = (a, b) ∧ (x, y) = (-a, -b))
  (hF2 : F2.1^2 / 9 - F2.2^2 / 27 = 1)
  (hP_hyperbola : P ∈ hyperbola)
  (hP_parabola : P ∈ parabola p) :
  abs (P.1 * (F1.2 - F2.2) + F1.1 * (F2.2 - P.2) + F2.1 * (P.2 - F1.2)) / 2 = 36 * sqrt 6 := by
  sorry

end triangle_area_l681_681233


namespace esther_commute_distance_l681_681183

theorem esther_commute_distance (D : ℕ) :
  (D / 45 + D / 30 = 1) → D = 18 :=
by
  sorry

end esther_commute_distance_l681_681183


namespace vector_magnitude_case1_l681_681243

variables {R : Type*} [RealField R]
variables (a b c : EuclideanSpace R (Fin 2)) 

-- Assume magnitudes
axiom magnitude_a : ‖a‖ = 1
axiom magnitude_b : ‖b‖ = 2
axiom magnitude_c : ‖c‖ = 3

-- Assume angles
axiom equal_angle_120 : (a ⬝ b = -1 ∧ a ⬝ c = -3/2 ∧ b ⬝ c = -3)
axiom equal_angle_0 : (a ⬝ b = 2 ∧ a ⬝ c = 3 ∧ b ⬝ c = 6)

theorem vector_magnitude_case1 : 
  (‖a + b + c‖ = √3) ∨ (‖a + b + c‖ = 5) :=
sorry

end vector_magnitude_case1_l681_681243


namespace greatest_3_digit_base8_num_div_by_7_eq_511_l681_681917

noncomputable def greatest_base8_number_divisible_by_7 : ℕ := 7 * 73

theorem greatest_3_digit_base8_num_div_by_7_eq_511 : 
  greatest_base8_number_divisible_by_7 = 511 :=
by 
  sorry

end greatest_3_digit_base8_num_div_by_7_eq_511_l681_681917


namespace slope_of_line_l681_681630

theorem slope_of_line (θ : ℝ) (h_cosθ : (Real.cos θ) = 4/5) : (Real.sin θ) / (Real.cos θ) = 3/4 :=
by
  sorry

end slope_of_line_l681_681630


namespace intersection_sets_l681_681241

noncomputable def set1 (x : ℝ) : Prop := (x - 2) / (x + 1) ≤ 0
noncomputable def set2 (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0

theorem intersection_sets :
  { x : ℝ | set1 x } ∩ { x : ℝ | set2 x } = { x | (-1 : ℝ) < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_sets_l681_681241


namespace solve_for_m_l681_681796

def z1 := Complex.mk 3 2
def z2 (m : ℝ) := Complex.mk 1 m

theorem solve_for_m (m : ℝ) (h : (z1 * z2 m).re = 0) : m = 3 / 2 :=
by
  sorry

end solve_for_m_l681_681796


namespace increase_s_for_min_product_l681_681599

-- Given four positive numbers p, q, r, and s in increasing order
variables (p q r s : ℝ)
-- Condition: p < q < r < s and all are positive
variables (hpq : p < q) (hqr : q < r) (hrs : r < s)
variables (hp_pos : 0 < p) (hq_pos : 0 < q) (hr_pos : 0 < r) (hs_pos : 0 < s)

-- Proof statement that s should be increased
theorem increase_s_for_min_product : 
  let ΔP_s : ℝ := p * q * r in
  let ΔP_p : ℝ := q * r * s in
  let ΔP_q : ℝ := p * r * s in
  let ΔP_r : ℝ := p * q * s in
  ΔP_s < ΔP_p ∧ ΔP_s < ΔP_q ∧ ΔP_s < ΔP_r := 
by {
  sorry
}

end increase_s_for_min_product_l681_681599


namespace floor_of_abs_add_l681_681544

-- Definitions for absolute value, addition, and floor
def abs_val := abs (-47.3)
def add_val := abs_val + 0.7
def floor_val := floor add_val

-- Final proof statement
theorem floor_of_abs_add : floor_val = 48 := by
  sorry

end floor_of_abs_add_l681_681544


namespace expected_red_lights_correct_l681_681455

variable (events : Fin 3 → ℝ)
variable (prob_red : ℝ)
variable (independent : ∀ (i j : Fin 3), i ≠ j → ProbEvent i = ProbEvent j = 0.3)

def expected_red_lights (n : ℕ) : ℝ :=
  if h : n = 3 then 3 * prob_red else 0

theorem expected_red_lights_correct :
  expected_red_lights 3 = 0.9 := by
  sorry

end expected_red_lights_correct_l681_681455


namespace periodic_sequence_of_f_l681_681776

open Function

-- Define the set of positive integers
def pos_ints := { n : ℕ // n > 0 }

-- Define the function f with given properties
theorem periodic_sequence_of_f
  (f : pos_ints → pos_ints)
  (H1 : ∀ (m n : pos_ints), (f^[n] m - m) / n ∈ pos_ints)
  (H2 : (univ \ set.range f).finite) :
  ∃ (d : ℕ), d > 0 ∧ ∀ (n : pos_ints), f (↑n + d) = f ↑n + d :=
  sorry

end periodic_sequence_of_f_l681_681776


namespace tim_balloons_count_l681_681167

-- Defining the conditions
def dan_balloons : ℝ := 29.0
def ratio : ℝ := 7.0
def tim_balloons : ℕ := Int.ofNat (round (dan_balloons / ratio))

-- Theorem stating the required proof
theorem tim_balloons_count : tim_balloons = 4 :=
by
  -- Proof would go here
  sorry

end tim_balloons_count_l681_681167


namespace sum_of_row_equals_square_l681_681823

theorem sum_of_row_equals_square (k : ℕ) (hk : k > 0) :
  ∑ i in finset.range (2 * k - 1), (k + i) = (2 * k - 1) ^ 2 := 
by
  sorry

end sum_of_row_equals_square_l681_681823


namespace mark_jump_height_l681_681812

theorem mark_jump_height (M : ℝ) :
  (let Lisa_height := 2 * M in
   let Jacob_height := 2 * Lisa_height in
   16 = (2 / 3) * Jacob_height) → M = 6 := 
by
  intros h
  sorry

end mark_jump_height_l681_681812


namespace distinct_triangles_count_l681_681690

theorem distinct_triangles_count (a b c : ℕ) (h1 : a + b + c = 11) (h2 : a + b > c) 
  (h3 : a + c > b) (h4 : b + c > a) : 
  10 := sorry

end distinct_triangles_count_l681_681690


namespace exists_100_disjoint_chords_l681_681605

theorem exists_100_disjoint_chords : 
  ∃ (S : set (fin (2^500 × fin 2^500))) (h : S.card = 100), 
  ∀ (x y ∈ S), x ≠ y → x.fst + x.snd = y.fst + y.snd :=
sorry

end exists_100_disjoint_chords_l681_681605


namespace ratio_EG_GF_l681_681293

-- Definitions of the conditions
variable {A B C E F G M : Type} [point : Point]
variable {between_AC_AE : Between point A C E}
variable {between_AB_AF : Between point A B F}
variable {between_EF_AG : Between point E F G}
variable {midpoint_BC_M : Midpoint point B C M}
variable {length_AB : SegmentLength point A B = 12}
variable {length_AC : SegmentLength point A C = 16}
variable {length_AE_AF : 2 * SegmentLength point A F = SegmentLength point A E}

-- The mathematically equivalent proof problem to prove
theorem ratio_EG_GF : SegmentRatio point E G F = 3 / 2 :=
by
  sorry

end ratio_EG_GF_l681_681293


namespace first_coloring_book_pictures_l681_681337

-- Given conditions
def total_pictures_colored : ℕ := 44
def pictures_left_to_color : ℕ := 11
def second_coloring_book_pictures : ℕ := 32

-- Derived condition
def total_pictures := total_pictures_colored + pictures_left_to_color

-- To prove
theorem first_coloring_book_pictures :
  total_pictures - second_coloring_book_pictures = 23 :=
by
  have total_pictures_def : total_pictures = 55 := by
    unfold total_pictures total_pictures_colored pictures_left_to_color
    simp
  rw total_pictures_def
  norm_num
  exact rfl

end first_coloring_book_pictures_l681_681337


namespace area_square_given_diagonal_l681_681024

theorem area_square_given_diagonal (d : ℝ) (h : d = 16) : (∃ A : ℝ, A = 128) :=
by 
  sorry

end area_square_given_diagonal_l681_681024


namespace hens_count_l681_681071

theorem hens_count (H R : ℕ) (h₁ : H = 9 * R - 5) (h₂ : H + R = 75) : H = 67 :=
by {
  sorry
}

end hens_count_l681_681071


namespace certain_event_l681_681221

-- Definitions for a line and plane
inductive Line
| mk : Line

inductive Plane
| mk : Plane

-- Definitions for parallel and perpendicular relations
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def plane_parallel (p₁ p₂ : Plane) : Prop := sorry

-- Given conditions and the proof statement
theorem certain_event (l : Line) (α β : Plane) (h1 : perpendicular l α) (h2 : perpendicular l β) : plane_parallel α β :=
sorry

end certain_event_l681_681221


namespace solve_X_l681_681169

def diamond (X Y : ℝ) : ℝ := 4 * X - 3 * Y + 2

theorem solve_X :
  (∃ X : ℝ, diamond X 6 = 35) ↔ (X = 51 / 4) := by
  sorry

end solve_X_l681_681169


namespace binom_18_6_eq_4765_l681_681532

def binom (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_18_6_eq_4765 : binom 18 6 = 4765 := by
  sorry

end binom_18_6_eq_4765_l681_681532


namespace remainder_division_l681_681965

theorem remainder_division :
  ∃ N R1 Q2, N = 44 * 432 + R1 ∧ N = 30 * Q2 + 18 ∧ R1 < 44 ∧ 18 = R1 :=
by
  sorry

end remainder_division_l681_681965


namespace quadratic_solution_l681_681348

theorem quadratic_solution (p q x : ℝ) :
  (∃ y z : ℝ, x = y + z ∧ 2 * z + p = 0 ∧ y^2 = z^2 + p * z + q ∧
  y = - z ± sqrt (z^2 + p * z + q)) → 
  x = - p / 2 ± sqrt (p^2 / 4 - q) :=
by
  sorry

end quadratic_solution_l681_681348


namespace prob_甲_third_try_success_prob_at_least_one_success_first_try_l681_681940

noncomputable def P (event : Prop) (prob : ℝ) : ℝ := if event then prob else 1 - prob

def P_A : ℝ := 0.8
def P_B : ℝ := 0.6

def P_not_A : ℝ := 1 - P_A
def P_not_B : ℝ := 1 - P_B

theorem prob_甲_third_try_success :
  P (P_not_A ∧ P_not_A ∧ P_A) 0.2 * 0.2 * 0.8 = 0.032 := 
  sorry

theorem prob_at_least_one_success_first_try :
  1 - (P_not_A * P_not_B) = 0.92 :=
  sorry

end prob_甲_third_try_success_prob_at_least_one_success_first_try_l681_681940


namespace problem_statement_l681_681303

def quadratic_minimum (f : ℝ → ℝ) (x_min : ℝ) (y_min : ℝ) :=
  ∀ x, f x ≥ y_min ∧ f x_min = y_min

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then 2
  else (sequence (n - 1)) + 0.5

def sum_sequence (n : ℕ) : ℝ :=
  (n / 2) * (2 * 2 + (n - 1) * 0.5)

theorem problem_statement : 
  quadratic_minimum (λ x, x^2 - 2 * x + 3) 1 2 →
  sum_sequence 9 = 36 :=
by
  sorry

end problem_statement_l681_681303


namespace intersection_planes_l681_681628

def plane (α β : Type) := α → β → Prop
def line (l : Type) := l → Prop
def contains (α : Type) (m : line Type): Prop := ∀ (x : Type), m x → α x
def intersects (α β : plane Type) (l : line Type) : Prop := ∀ P, α P ∧ β P
def point (P : Type) := P → Prop

variable {α β : Type} [plane α β] {m l : Type}
variable {P : Type} [point P]

theorem intersection_planes {m_line : contains α m} {l_line : intersects α β l} {m_inter_l : P ∈ l ∧ P ∈ m} : 
  ∃ (p : line Type), contains β p ∧ (∃ q : line Type, contains β q ∧ (¬parallel β m q ∧ perpendicular β m p)) :=
begin
  sorry -- Proof goes here
end

end intersection_planes_l681_681628


namespace angle_ABD_eq_90_l681_681887

/-- 
  In a triangle ABC, where ∠ABC = 135°, the triangle is inscribed in
  a circle ω. The lines tangent to ω at points A and C intersect at point D.
  Given that line AB bisects the segment CD, prove that ∠ABD = 90°.
--/
theorem angle_ABD_eq_90
  (ABC : Triangle) 
  (ω : Circle)
  (A B C D : Point)
  (hABC : ∠ABC = 135°)
  (hInscribed : IsInscribedTriangle ABC ω)
  (hTangents : TangentLinesIntersectAt ω A C D)
  (hBisect : SegmentBisects AB CD) :
  ∠ABD = 90° :=
sorry

end angle_ABD_eq_90_l681_681887


namespace count_triangles_with_perimeter_11_l681_681693

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem count_triangles_with_perimeter_11 :
  {t : (ℕ × ℕ × ℕ) // let ⟨a, b, c⟩ := t in a + b + c = 11 ∧ is_triangle a b c}.to_finset.card = 9 :=
by sorry

end count_triangles_with_perimeter_11_l681_681693


namespace greatest_base8_three_digit_divisible_by_7_l681_681910

theorem greatest_base8_three_digit_divisible_by_7 :
  ∃ n : ℕ, n < 8^3 ∧ n ≥ 8^2 ∧ (n % 7 = 0) ∧ (to_base 8 n = 777) :=
sorry

end greatest_base8_three_digit_divisible_by_7_l681_681910


namespace min_expression_value_l681_681585

theorem min_expression_value :
  ∃ x y : ℝ, (9 - x^2 - 8 * x * y - 16 * y^2 > 0) ∧ 
  (∀ x y : ℝ, 9 - x^2 - 8 * x * y - 16 * y^2 > 0 →
  (13 * x^2 + 24 * x * y + 13 * y^2 + 16 * x + 14 * y + 68) / 
  (9 - x^2 - 8 * x * y - 16 * y^2)^(5/2) = (7 / 27)) :=
sorry

end min_expression_value_l681_681585


namespace angle_BPM_right_iff_B1C_eq_3AB1_l681_681285

variables {α : Type} [EuclideanGeometry α]
variables (A B C A1 B1 M P : α)
variables (altitude_A : is_altitude A C B A1)
variables (altitude_B : is_altitude B A C B1)
variables (orthocenter_M : is_orthocenter A B C M)
variables (median_BP : is_median B A C P)
variables (intersect : lies_on_line P A1 B1)

theorem angle_BPM_right_iff_B1C_eq_3AB1 :
  ∠ B P M = 90 ↔ dist B1 C = 3 * dist A B1 :=
by
  sorry

end angle_BPM_right_iff_B1C_eq_3AB1_l681_681285


namespace f_2_eq_23_f_4_eq_23_max_f_x_f_3_eq_25_and_max_l681_681491

def bus_speed := (12 : ℝ) / (20 : ℝ)  -- Speed in miles per minute

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 10 + (5 * x) / 3
  else if 11 ≤ x ∧ x ≤ 12 then 20 - (5 * x) / 3
  else if 1 < x ∧ x ≤ 3 then 20 + (5 * x) / 3
  else if 3 < x ∧ x ≤ 6 then 30 - (5 * x) / 3
  else if 6 < x ∧ x ≤ 9 then 10 + (5 * x) / 3
  else if 9 < x ∧ x < 11 then 40 - (5 * x) / 3
  else 0  -- This should never be reached given the domain of x

theorem f_2_eq_23 : f 2 = 23 :=
by
  unfold f
  simp

theorem f_4_eq_23 : f 4 = 23 :=
by
  unfold f
  simp

theorem max_f_x : ∀ x : ℝ, 0 < x ∧ x < 12 → f x ≤ 25 :=
by
  intro x hx
  unfold f
  split_ifs;
  -- Each case analysis...
  sorry

theorem f_3_eq_25_and_max : f 3 = 25 :=
by
  unfold f
  simp

end f_2_eq_23_f_4_eq_23_max_f_x_f_3_eq_25_and_max_l681_681491


namespace central_angle_of_sector_l681_681607

/-- The central angle of the sector obtained by unfolding the lateral surface of a cone with
    base radius 1 and slant height 2 is \(\pi\). -/
theorem central_angle_of_sector (r_base : ℝ) (r_slant : ℝ) (α : ℝ)
  (h1 : r_base = 1) (h2 : r_slant = 2) (h3 : 2 * π = α * r_slant) : α = π :=
by
  sorry

end central_angle_of_sector_l681_681607


namespace grill_runtime_l681_681429

theorem grill_runtime
    (burn_rate : ℕ)
    (burn_time : ℕ)
    (bags : ℕ)
    (coals_per_bag : ℕ)
    (total_burnt_coals : ℕ)
    (total_time : ℕ)
    (h1 : burn_rate = 15)
    (h2 : burn_time = 20)
    (h3 : bags = 3)
    (h4 : coals_per_bag = 60)
    (h5 : total_burnt_coals = bags * coals_per_bag)
    (h6 : total_time = (total_burnt_coals / burn_rate) * burn_time) :
    total_time = 240 :=
by sorry

end grill_runtime_l681_681429


namespace incenter_identity_l681_681775

variable {A B C S X Y Z : Point} -- Defining the points A, B, C, S, X, Y, Z
variable {R r : ℝ} -- Defining the circumradius R and inradius r

-- Define the geometrical relationships
variable [InTriangle A B C S] -- S is inside the triangle ABC
variable (AS : LineSegment A S) (BS : LineSegment B S) (CS : LineSegment C S)
variable (AX : LineSegment A X) (BX : LineSegment B X) (CX : LineSegment C X)
variable (AY : LineSegment A Y) (BY : LineSegment B Y) (CY : LineSegment C Y)
variable (AZ : LineSegment A Z) (BZ : LineSegment B Z) (CZ : LineSegment C Z)

-- intersections at points X, Y, Z
axiom AS_intersects_BC_at_X : IntersectAt (LineThrough A S) (LineThrough B C) X
axiom BS_intersects_CA_at_Y : IntersectAt (LineThrough B S) (LineThrough C A) Y
axiom CS_intersects_AB_at_Z : IntersectAt (LineThrough C S) (LineThrough A B) Z

-- Define the theorem to prove
theorem incenter_identity (h : IsIncenter S A B C) :
  (BX.length * CX.length) / (AX.length ^ 2) +
  (CY.length * AY.length) / (BY.length ^ 2) +
  (AZ.length * BZ.length) / (CZ.length ^ 2) = R / r - 1 :=
sorry

end incenter_identity_l681_681775


namespace decreasing_on_interval_l681_681658

noncomputable def f (a x : ℝ) : ℝ := 2^(x * (x - a))

theorem decreasing_on_interval (a : ℝ) : (a ≥ 2) ↔ ∀ x ∈ Set.Ioo 0 1, (deriv (λ x, 2^(x * (x - a)))) x ≤ 0 :=
sorry

end decreasing_on_interval_l681_681658


namespace fraction_irreducible_l681_681336

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_irreducible_l681_681336


namespace circle_angle_bisector_l681_681155

-- Given points and circles
variables (A B C C₁ : Point) 
variables (S_A S_B S : Circle)
variables (LineAB : Line)

-- Conditions
variables (hA_B : A ≠ B) (hA_C : A ≠ C) (hB_C : B ≠ C) (hInc₁ : S.contains_point C₁) 

noncomputable def centers_on_LineAB := 
  S_A.center ∈ LineAB ∧ S_B.center ∈ LineAB

noncomputable def S_A_passes_through_AC := 
  S_A.contains_point A ∧ S_A.contains_point C

noncomputable def S_B_passes_through_BC := 
  S_B.contains_point B ∧ S_B.contains_point C

noncomputable def S_touches_S_A_S_B_and_C₁ := 
  S.tangent_to S_A ∧ S.tangent_to S_B ∧ S.contains_point C₁

theorem circle_angle_bisector
  (hCond1 : centers_on_LineAB)
  (hCond2 : S_A_passes_through_AC)
  (hCond3 : S_B_passes_through_BC)
  (hCond4 : S_touches_S_A_S_B_and_C₁)
  : is_angle_bisector C C₁ A B :=
sorry

end circle_angle_bisector_l681_681155


namespace combination_18_6_l681_681499

theorem combination_18_6 : (nat.choose 18 6) = 18564 := 
by 
  sorry

end combination_18_6_l681_681499


namespace tetrakaidecagon_in_square_area_l681_681424

noncomputable def tetrakaidecagon_area (perimeter : ℝ) (side_segments : ℕ) : ℝ :=
let side_length := perimeter / 4 in
let segment_length := side_length / side_segments in
let square_area := side_length ^ 2 in
let triangular_segment_area := (1 / 2) * segment_length * segment_length in
let total_triangular_area := 16 * triangular_segment_area in
square_area - total_triangular_area

theorem tetrakaidecagon_in_square_area :
  tetrakaidecagon_area 56 7 = 21.92 :=
sorry

end tetrakaidecagon_in_square_area_l681_681424


namespace value_of_phi_l681_681724

theorem value_of_phi (φ : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = sin (x + φ) + cos x) (h_max : ∀ x, f x ≤ 2) : 
    φ = π / 2 ∨ (∃ k : ℤ, φ = π / 2 + 2 * k * π) :=
by
  sorry

end value_of_phi_l681_681724


namespace joe_two_different_fruits_in_a_day_l681_681485

def joe_meal_event : Type := {meal : ℕ // meal = 4}
def joe_fruit_choice : Type := {fruit : ℕ // fruit ≤ 4}

noncomputable def prob_all_same_fruit : ℚ := (1 / 4) ^ 4 * 4
noncomputable def prob_at_least_two_diff_fruits : ℚ := 1 - prob_all_same_fruit

theorem joe_two_different_fruits_in_a_day :
  prob_at_least_two_diff_fruits = 63 / 64 :=
by
  sorry

end joe_two_different_fruits_in_a_day_l681_681485


namespace find_abc_monotonicity_find_m_l681_681806

def f (x : ℝ) (a b c : ℝ) := (a * x^2 + 1) / (b * x + c)

theorem find_abc (a b c : ℤ) (h1 : f (-1) a b c = -2) (h2 : f 2 a b c < 3) (h3 : ∀ x : ℝ, f (-x) a b c = - f x a b c) : 
  a = 1 ∧ b = 1 ∧ c = 0 :=
sorry

theorem monotonicity (a b c : ℤ) (h1 : a = 1) (h2 : b = 1) (h3 : c = 0) : 
  (∀ x1 x2 : ℝ, x1 < x2 ∧ x2 ≤ -1 ∧ x1 < 0 ∧ x2 < 0 → f x1 a b c < f x2 a b c) ∧ 
  (∀ x1 x2 : ℝ, x1 < x2 ∧ -1 ≤ x1 ∧ x1 < 0 ∧ x2 < 0 → f x1 a b c > f x2 a b c) :=
sorry

theorem find_m (a b c : ℤ) (h1 : a = 1) (h2 : b = 1) (h3 : c = 0) (h4 : ∀ x : ℝ, x < 0 → 2 * m - 1 > f x a b c) : 
  m > -1/2 :=
sorry

end find_abc_monotonicity_find_m_l681_681806


namespace greatest_3_digit_base_8_divisible_by_7_l681_681914

open Nat

def is_3_digit_base_8 (n : ℕ) : Prop := n < 8^3

def is_divisible_by_7 (n : ℕ) : Prop := 7 ∣ n

theorem greatest_3_digit_base_8_divisible_by_7 :
  ∃ x : ℕ, is_3_digit_base_8 x ∧ is_divisible_by_7 x ∧ x = 7 * (8 * (8 * 7 + 7) + 7) :=
by
  sorry

end greatest_3_digit_base_8_divisible_by_7_l681_681914


namespace decreasing_interval_range_l681_681648

theorem decreasing_interval_range (a : ℝ) :
  (∀ x y ∈ Ioo 0 1, x < y → 2^(x * (x-a)) > 2^(y * (y-a))) ↔ a ≥ 2 :=
by
  sorry

end decreasing_interval_range_l681_681648


namespace max_distanceMN_l681_681749

-- Definition of Curve C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Transformation equations
def transform_x (x : ℝ) : ℝ := x / 2
def transform_y (y : ℝ) : ℝ := y

-- Definition of Curve C2 after transformation
def C2_parametric (φ : ℝ) : ℝ × ℝ := (2 * Real.cos φ, Real.sin φ)
def C2 (x y : ℝ) : Prop := ∃ φ, (x, y) = C2_parametric φ

-- Define the polar equation of C3 and conversion to rectangular form
def C3_polar (ρ θ : ℝ) : Prop := ρ * (ρ - 6 * Real.sin θ) = -8

def C3_rectangular (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 1

-- Definition of the distance |MN| and its maximum value
def distance (M N : ℝ × ℝ) : ℝ :=
  Real.sqrt (((M.1 - N.1)^2 + (M.2 - N.2)^2))

-- Main theorem stating the maximum distance |MN|
theorem max_distanceMN :
  ∀ φ ∈ Icc (-Real.pi/2) (Real.pi/2),
  let M := C2_parametric φ in
  let N := (0, 3) in
  distance M N <= Real.sqrt 15 :=
sorry

end max_distanceMN_l681_681749


namespace decimal_to_binary_41_l681_681165

theorem decimal_to_binary_41 : nat.binary_repr 41 = "101001" :=
by sorry

end decimal_to_binary_41_l681_681165


namespace tank_fraction_after_adding_water_l681_681683

def tank_full_fraction (t : ℚ) (f : ℚ) (a : ℚ) : ℚ :=
  (f * t + a) / t

theorem tank_fraction_after_adding_water :
  ∀ (t : ℚ) (f : ℚ) (a : ℚ),
  t = 72 → f = 3 / 4 → a = 9 →
  tank_full_fraction t f a = 7 / 8 :=
by
  intros t f a ht hf ha
  rw [ht, hf, ha]
  calc
    tank_full_fraction 72 (3 / 4) 9
      = (3 / 4 * 72 + 9) / 72 : rfl
  ... = (54 + 9) / 72        : by norm_num
  ... = 63 / 72              : rfl
  ... = 7 / 8                : by norm_num

end tank_fraction_after_adding_water_l681_681683


namespace arithmetic_sequence_sum_l681_681128

theorem arithmetic_sequence_sum :
  let sequence := list.range (20 / 2) in
  let sum := sequence.map (λ n, 2 * (n + 1)).sum in
  sum = 110 :=
by
  -- Define the sequence as the arithmetic series
  let sequence := list.range (20 / 2)
  -- Calculate the sum of the arithmetic sequence
  let sum := sequence.map (λ n, 2 * (n + 1)).sum
  -- Check the sum
  have : sum = 110 := sorry
  exact this

end arithmetic_sequence_sum_l681_681128


namespace ground_beef_cost_l681_681681

theorem ground_beef_cost (unit_price quantity : ℕ) (h_unit_price : unit_price = 5) (h_quantity : quantity = 12) : unit_price * quantity = 60 :=
by
  rw [h_unit_price, h_quantity]
  norm_num

end ground_beef_cost_l681_681681


namespace customers_tried_sample_l681_681983

theorem customers_tried_sample
  (samples_per_box : ℕ)
  (boxes_opened : ℕ)
  (samples_left_over : ℕ)
  (samples_per_customer : ℕ := 1)
  (h_samples_per_box : samples_per_box = 20)
  (h_boxes_opened : boxes_opened = 12)
  (h_samples_left_over : samples_left_over = 5) :
  (samples_per_box * boxes_opened - samples_left_over) / samples_per_customer = 235 :=
by
  sorry

end customers_tried_sample_l681_681983


namespace find_first_term_l681_681879

theorem find_first_term
  (S : ℝ) (a r : ℝ)
  (h1 : S = 10)
  (h2 : a + a * r = 6)
  (h3 : a = 2 * r) :
  a = -1 + Real.sqrt 13 ∨ a = -1 - Real.sqrt 13 := by
  sorry

end find_first_term_l681_681879


namespace find_centroid_of_triangle_l681_681758

-- Definitions for points and geometrical constructs
variable {A B C O L M N : Type}
variable [affine_space ℝ Triangle]

-- Conditions from the problem
def on_line_segment (P : Type) (A B : Type) : Prop := sorry
def parallel_lines (L M : Type) : Prop := sorry
def area_equal (T1 T2 : Type) : Prop := sorry
def centroid (P Q R S : Type) : Prop := sorry

axiom L_condition : on_line_segment L A B
axiom M_condition : on_line_segment M B C
axiom N_condition : on_line_segment N C A
axiom OL_parallel_BC : parallel_lines (line A B O) (line B C)
axiom OM_parallel_AC : parallel_lines (line A O M) (line A C)
axiom ON_parallel_AB : parallel_lines (line O N A) (line A B)

theorem find_centroid_of_triangle :
  (area_equal (triangle B O L) (triangle C O M) ∧
   area_equal (triangle C O M) (triangle A O N)) →
   centroid O A B C :=
by
  sorry

end find_centroid_of_triangle_l681_681758


namespace necessary_but_not_sufficient_condition_l681_681622

-- Let p be the proposition |x| < 2
def p (x : ℝ) : Prop := abs x < 2

-- Let q be the proposition x^2 - x - 2 < 0
def q (x : ℝ) : Prop := x^2 - x - 2 < 0

-- The proof statement
theorem necessary_but_not_sufficient_condition (x : ℝ) : q x → p x ∧ ¬ (p x → q x) := 
sorry

end necessary_but_not_sufficient_condition_l681_681622


namespace rearrangement_inequality_square_sums_l681_681050

theorem rearrangement_inequality_square_sums {n : ℕ} 
    (x y : Fin n → ℝ) 
    (h1 : ∀ i j : Fin n, i ≤ j → x i ≥ x j)
    (h2 : ∀ i j : Fin n, i ≤ j → y i ≥ y j) 
    (z : Fin n → ℝ) 
    (hz : ∃ σ : Equiv.Perm (Fin n), ∀ i, z (σ i) = y i) : 
    ∑ i in Finset.finRange n, (x i - y i)^2 ≤ ∑ i in Finset.finRange n, (x i - z i)^2 := by
  sorry

end rearrangement_inequality_square_sums_l681_681050


namespace arith_seq_ratio_l681_681677

theorem arith_seq_ratio {a b : ℕ → ℕ} {S T : ℕ → ℕ}
  (h₁ : ∀ n, S n = (n * (2 * a n - a 1)) / 2)
  (h₂ : ∀ n, T n = (n * (2 * b n - b 1)) / 2)
  (h₃ : ∀ n, S n / T n = (5 * n + 3) / (2 * n + 7)) :
  (a 9 / b 9 = 88 / 41) :=
sorry

end arith_seq_ratio_l681_681677


namespace combination_18_6_l681_681525

theorem combination_18_6 : nat.choose 18 6 = 18564 :=
by {
  sorry
}

end combination_18_6_l681_681525


namespace sum_5n_is_630_l681_681728

variable (n : ℕ)

def sum_first_k (k : ℕ) : ℕ :=
  k * (k + 1) / 2

theorem sum_5n_is_630 (h : sum_first_k (3 * n) = sum_first_k n + 210) : sum_first_k (5 * n) = 630 := sorry

end sum_5n_is_630_l681_681728


namespace x_100_gt_0_99_l681_681867

variable (x : ℕ → ℝ)
variable (h₁ : x 1 = 1/2)
variable (h₂ : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 99 → x (n + 1) = 1 - (x 1 * x 2 * x 3 * ... * x n))

theorem x_100_gt_0_99 (x : ℕ → ℝ) 
  (h₁ : x 1 = 1/2) 
  (h₂ : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 99 → x (n + 1) = 1 - (x 1 * x 2 * x 3 * ... * x n)) :
  x 100 > 0.99 :=
sorry

end x_100_gt_0_99_l681_681867


namespace midpoints_of_AC_CD_l681_681944

/-- Points A, B, C, D lie on the same plane, areas of triangles ABD, BCD, ABC are in the
ratio 3:4:1, points M and N lie on AC and CD respectively, and B, M, N are collinear. -/
variables {A B C D M N : Point}
variables {area_ABD area_BCD area_ABC : ℝ}
variables (h_ratio : area_ABD / area_ABC = 3 ∧ area_BCD / area_ABC = 4)
variables (h_AM_AC : ∃ μ : ℝ, AM / AC = μ ∧ CN / CD = μ)
variables (h_collinear : Collinear B M N)

/-- M and N are the midpoints of AC and CD respectively. -/
theorem midpoints_of_AC_CD (h_ratio : area_ABD / area_ABC = 3 ∧ area_BCD / area_ABC = 4)
  (h_AM_AC : ∃ μ : ℝ, AM / AC = μ ∧ CN / CD = μ) (h_collinear : Collinear B M N):
  AM / AC = 1 / 2 ∧ CN / CD = 1 / 2 :=
sorry

end midpoints_of_AC_CD_l681_681944


namespace simplify_expression_l681_681825

theorem simplify_expression (a b : ℝ) (h : a ≠ b) : 
  ((a^3 - b^3) / (a * b)) - ((a * b^2 - b^3) / (a * b - a^3)) = (2 * a * (a - b)) / b :=
by
  sorry

end simplify_expression_l681_681825


namespace algebraic_expression_value_l681_681234

theorem algebraic_expression_value (a : ℝ) (h1 : 1 ≤ a) (h2 : a < 2) :
  sqrt (a + 2 * sqrt (a - 1)) + sqrt (a - 2 * sqrt (a - 1)) = 2 :=
sorry

end algebraic_expression_value_l681_681234


namespace todd_spending_l681_681885

theorem todd_spending :
  let candy_bar : ℝ := 1.14
  let discount : ℝ := 0.10
  let cookies : ℝ := 2.39
  let soda : ℝ := 1.75
  let tax : ℝ := 0.07
  let discounted_candy_bar : ℝ := (candy_bar * (1 - discount)).round(2)
  let total_before_tax : ℝ := (discounted_candy_bar + cookies + soda).round(2)
  let total_tax : ℝ := (total_before_tax * tax).round(2)
  let total_cost : ℝ := total_before_tax + total_tax
  in total_cost.round(2) = 5.53 := by
sorry

end todd_spending_l681_681885


namespace boat_and_current_speed_boat_and_current_speed_general_log_drift_time_l681_681057

-- Problem 1: Specific case
theorem boat_and_current_speed (x y : ℝ) 
  (h1 : 3 * (x + y) = 75) 
  (h2 : 5 * (x - y) = 75) : 
  x = 20 ∧ y = 5 := 
sorry

-- Problem 2: General case
theorem boat_and_current_speed_general (x y : ℝ) (a b S : ℝ) 
  (h1 : a * (x + y) = S) 
  (h2 : b * (x - y) = S) : 
  x = (a + b) * S / (2 * a * b) ∧ 
  y = (b - a) * S / (2 * a * b) := 
sorry

theorem log_drift_time (y S a b : ℝ)
  (h_y : y = (b - a) * S / (2 * a * b)) : 
  S / y = 2 * a * b / (b - a) := 
sorry

end boat_and_current_speed_boat_and_current_speed_general_log_drift_time_l681_681057


namespace find_f2_l681_681805

-- Define the function f(x) = ax + b
def f (a b x : ℝ) : ℝ := a * x + b

-- Condition: f'(x) = a
def f_derivative (a b x : ℝ) : ℝ := a

-- Given conditions
variables (a b : ℝ)
axiom h1 : f a b 1 = 2
axiom h2 : f_derivative a b 1 = 2

theorem find_f2 : f a b 2 = 4 :=
by
  sorry

end find_f2_l681_681805


namespace intersection_reciprocal_sum_l681_681745

noncomputable def polar_equation_C1 (ρ θ : ℝ) : Prop := 
   ρ^2 - 4 * ρ * cos θ - 4 * ρ * sin θ + 7 = 0

def polar_equation_C2 (θ : ℝ) : Prop := 
  θ = π / 3

theorem intersection_reciprocal_sum :
  ∃ (ρ1 ρ2 : ℝ), 
    polar_equation_C1 ρ1 (π/3) ∧ 
    polar_equation_C1 ρ2 (π/3) ∧ 
    ρ1 ≠ 0 ∧ 
    ρ2 ≠ 0 ∧ 
    (1 / ρ1 + 1 / ρ2 = (2 * real.sqrt 3 + 2) / 7) :=
  sorry

end intersection_reciprocal_sum_l681_681745


namespace expected_divisor_count_l681_681466

noncomputable def expected_divisors (S : set ℕ) : ℚ :=
if S = ∅ then 1 else ∑ p in {2, 3, 5, 7}, (1 + 1/2 * ∑ k in S, k)

def expected_number_of_divisors_of_S : ℚ :=
(1/4) * (ℚ) (45/2 + 315/8 + 375/8 + 315/4)

theorem expected_divisor_count : 
  expected_number_of_divisors_of_S = 375 / 8 := by
  sorry

end expected_divisor_count_l681_681466


namespace find_inverse_l681_681580

noncomputable def inverse_matrix_2x2 (a b c d : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  if ad_bc : (a * d - b * c) = 0 then (0, 0, 0, 0)
  else (d / (a * d - b * c), -b / (a * d - b * c), -c / (a * d - b * c), a / (a * d - b * c))

theorem find_inverse :
  inverse_matrix_2x2 5 7 2 3 = (3, -7, -2, 5) :=
by 
  sorry

end find_inverse_l681_681580


namespace kids_afternoon_soccer_camp_l681_681394

def total_kids : ℕ := 2000
def fraction_to_soccer : ℚ := 1/2
def fraction_morning : ℚ := 1/4

theorem kids_afternoon_soccer_camp : 
  let kids_soccer := total_kids * fraction_to_soccer in
  let kids_morning := kids_soccer * fraction_morning in
  kids_soccer - kids_morning = 750 :=
by
  sorry

end kids_afternoon_soccer_camp_l681_681394


namespace count_triangles_with_perimeter_11_l681_681694

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem count_triangles_with_perimeter_11 :
  {t : (ℕ × ℕ × ℕ) // let ⟨a, b, c⟩ := t in a + b + c = 11 ∧ is_triangle a b c}.to_finset.card = 9 :=
by sorry

end count_triangles_with_perimeter_11_l681_681694


namespace find_s_l681_681570

theorem find_s (s : ℝ) (m : ℤ) (d : ℝ) (h_floor : ⌊s⌋ = m) (h_decompose : s = m + d) (h_fractional : 0 ≤ d ∧ d < 1) (h_equation : ⌊s⌋ - s = -10.3) : s = -9.7 :=
by
  sorry

end find_s_l681_681570


namespace find_a_of_parallel_lines_l681_681244

theorem find_a_of_parallel_lines :
  ∀ (a : ℝ), (∀ (x y : ℝ), (x + 2 * y - 3 = 0) → (2 * x - a * y + 3 = 0)) → a = -4 :=
by
  intros a h
  have slope_l1 := (-1 : ℝ) / 2
  have slope_l2 := (-2 : ℝ) / a
  have slopes_equal := slope_l1 = slope_l2
  calc a = -4 : sorry

end find_a_of_parallel_lines_l681_681244


namespace right_triangle_median_len_XY_l681_681159

theorem right_triangle_median_len_XY
  (X Y Z : Type) [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (XY YZ XZ : ℝ)
  (h : ∃ (XYZ : Triangle X Y Z), XYZ.right_angle Y)
  (median_X_to_YZ : Real) (median_Y_to_XZ : Real)
  (h_median_X : median_X_to_YZ = 8)
  (h_median_Y : median_Y_to_XZ = 3 * Real.sqrt 5) :
  XY = 3 * Real.sqrt 174 := sorry

end right_triangle_median_len_XY_l681_681159


namespace apples_left_l681_681049

theorem apples_left (baskets : Fin 11 → ℕ) (h1 : (∑ i : Fin 11, baskets i) = 1000)
    (h2 : ∀ i : Fin 11, baskets i > 0) :
    (1000 - ∑ i : Fin 11, 10 * (i + 1)) = 340 :=
by 
  sorry

end apples_left_l681_681049


namespace binom_18_6_eq_18564_l681_681506

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binom_18_6_eq_18564 : binomial 18 6 = 18564 := by
  sorry

end binom_18_6_eq_18564_l681_681506


namespace cosine_identity_l681_681257

variable (α : ℝ)

theorem cosine_identity (h : Real.sin (Real.pi / 6 - α) = 1 / 3) : 
  Real.cos (2 * Real.pi / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cosine_identity_l681_681257


namespace line_parabola_midpoint_l681_681353

theorem line_parabola_midpoint (a b : ℝ) 
  (r s : ℝ) 
  (intersects_parabola : ∀ x, x = r ∨ x = s → ax + b = x^2)
  (midpoint_cond : (r + s) / 2 = 5 ∧ (r^2 + s^2) / 2 = 101) :
  a + b = -41 :=
sorry

end line_parabola_midpoint_l681_681353


namespace terminal_tangent_330_l681_681261

theorem terminal_tangent_330 {x y : ℝ} (h : x ≠ 0) :
  (∃ (P : ℝ × ℝ), P = (x, y) ∧ (∃ θ : ℝ, θ = 330 ∧ cos θ = x/√(x^2 + y^2) ∧ sin θ = y/√(x^2 + y^2))) 
  → y / x = -√3 / 3 :=
by
  intros h1
  sorry

end terminal_tangent_330_l681_681261


namespace count_valid_arrangements_l681_681201

-- Define the grid with fixed placements.
def grid := Array (Array char)

-- Define the conditions.
def valid_grid (g : grid) : Prop :=
  (∀ i : ℕ, i < 4 → g[i][0] == 'Y' ∨ g[i][0] == 'Z') ∧
  (∀ j : ℕ, j < 3 → g[0][j] == 'Y' ∨ g[0][j] == 'Z') ∧
  (∀ i j : ℕ, i < 4 ∧ j < 3 → g[i][j] ∈ {'X', 'Y', 'Z'}) ∧
  (∀ i : ℕ, i < 4 → ∑ j : ℕ in finRange 3, if g[i][j] == 'X' then 1 else 0 = 1) ∧
  (∀ j : ℕ, j < 3 → ∑ i : ℕ in finRange 4, if g[i][j] == 'X' then 1 else 0 = 1) ∧
  (∀ i : ℕ, i < 4 → ∑ j : ℕ in finRange 3, if g[i][j] == 'Y' then 1 else 0 = 1) ∧
  (∀ j : ℕ, j < 3 → ∑ i : ℕ in finRange 4, if g[i][j] == 'Y' then 1 else 0 = 1) ∧
  (∀ i : ℕ, i < 4 → ∑ j : ℕ in finRange 3, if g[i][j] == 'Z' then 1 else 0 = 1) ∧
  (∀ j : ℕ, j < 3 → ∑ i : ℕ in finRange 4, if g[i][j] == 'Z' then 1 else 0 = 1) ∧
  g[0][0] == 'X'

-- The problem is to show that the number of valid grids that satisfy all conditions is 2.
def counting_valid_grids : ℕ :=
  let valid_grids := {g : grid // valid_grid g}
  valid_grids.toList.length

-- Statement to be proven.
theorem count_valid_arrangements : counting_valid_grids = 2 :=
  sorry

end count_valid_arrangements_l681_681201


namespace P_eq_Q_l681_681810

def P (m : ℝ) : Prop := -1 < m ∧ m < 0

def quadratic_inequality (m : ℝ) (x : ℝ) : Prop := m * x^2 + 4 * m * x - 4 < 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, quadratic_inequality m x

theorem P_eq_Q : ∀ m : ℝ, P m ↔ Q m := 
by 
  sorry

end P_eq_Q_l681_681810


namespace solve_for_a_l681_681314

noncomputable def problem_statement (a : ℤ) : Prop := 
  {1, 3, a} ⊇ {1, a^2 - a + 1}

theorem solve_for_a (a : ℤ) : problem_statement a ↔ a = -1 ∨ a = 2 := by 
  sorry

end solve_for_a_l681_681314


namespace kitchen_chairs_count_l681_681069

-- Define the conditions
def total_chairs : ℕ := 9
def living_room_chairs : ℕ := 3

-- Prove the number of kitchen chairs
theorem kitchen_chairs_count : total_chairs - living_room_chairs = 6 := by
  -- Proof goes here
  sorry

end kitchen_chairs_count_l681_681069


namespace greatest_3_digit_base8_num_div_by_7_eq_511_l681_681919

noncomputable def greatest_base8_number_divisible_by_7 : ℕ := 7 * 73

theorem greatest_3_digit_base8_num_div_by_7_eq_511 : 
  greatest_base8_number_divisible_by_7 = 511 :=
by 
  sorry

end greatest_3_digit_base8_num_div_by_7_eq_511_l681_681919


namespace cube_surface_area_l681_681881

variable (V : ℝ) (s : ℝ)

theorem cube_surface_area (hV : V = 343) (hs : s = Real.cbrt V) : 6 * s^2 = 294 := by
  sorry

end cube_surface_area_l681_681881


namespace max_b_plus_c_triangle_l681_681296

theorem max_b_plus_c_triangle (a b c : ℝ) (A : ℝ) 
  (h₁ : a = 4) (h₂ : A = Real.pi / 3) (h₃ : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) :
  b + c ≤ 8 :=
by
  -- sorry is added to skip the proof for now.
  sorry

end max_b_plus_c_triangle_l681_681296


namespace matrix_transformation_l681_681572

theorem matrix_transformation (P : Matrix (Fin 3) (Fin 3) ℝ) :
  (P ⬝ (Matrix.vecCons (Matrix.vecCons (a, b, c) (Matrix.vecCons (d, e, f) (Matrix.vecCons (g, h, i) Matrix.vecEmpty))) = 
    Matrix.vecCons (Matrix.vecCons (3*a, 3*b, 3*c) (Matrix.vecCons (g, h, i) (Matrix.vecCons (d, e, f) Matrix.vecEmpty))) :=
    P = ![![3, 0, 0], ![0, 0, 1], ![0, 1, 0]] :=
sorry

end matrix_transformation_l681_681572


namespace exists_book_name_l681_681404

theorem exists_book_name : ∃ (name : String), True :=
by { existsi "The name you will recall", triv}

end exists_book_name_l681_681404


namespace find_a_range_l681_681656

noncomputable def f (x a : ℝ) := 2 ^ (x * (x - a))

theorem find_a_range (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → deriv (λ x, f x a) x < 0) → 2 ≤ a :=
by sorry

end find_a_range_l681_681656


namespace find_m_l681_681787

noncomputable def g (d e f x : ℤ) : ℤ := d * x * x + e * x + f

theorem find_m (d e f m : ℤ) (h₁ : g d e f 2 = 0)
    (h₂ : 60 < g d e f 6 ∧ g d e f 6 < 70) 
    (h₃ : 80 < g d e f 9 ∧ g d e f 9 < 90)
    (h₄ : 10000 * m < g d e f 100 ∧ g d e f 100 < 10000 * (m + 1)) :
  m = -1 :=
sorry

end find_m_l681_681787


namespace expected_adjacent_black_pairs_60_cards_l681_681850

noncomputable def expected_adjacent_black_pairs 
(deck_size : ℕ) (black_cards : ℕ) (red_cards : ℕ) : ℚ :=
  if h : deck_size = black_cards + red_cards 
  then (black_cards:ℚ) * (black_cards - 1) / (deck_size - 1) 
  else 0

theorem expected_adjacent_black_pairs_60_cards :
  expected_adjacent_black_pairs 60 36 24 = 1260 / 59 := by
  sorry

end expected_adjacent_black_pairs_60_cards_l681_681850


namespace determine_c_l681_681927

noncomputable def value_of_c (y : ℝ) (c k x : ℝ) (z : ℝ) : ℝ :=
  if h1 : y = c * exp (k * x) then
  if h2 : z = log y then
  if h3 : z = 0.4 * x + 2 then
  exp 2
  else 0
  else 0
  else 0

theorem determine_c :
  ∀ (c k x y z : ℝ),
  (y = c * exp (k * x)) →
  (z = log y) →
  (z = 0.4 * x + 2) →
  c = exp 2 :=
by
  intros c k x y z h1 h2 h3
  sorry

end determine_c_l681_681927


namespace num_factors_of_72_l681_681703

def num_factors (n : ℕ) : ℕ :=
  (n.factorization.to_multiset.map (λ x, x.2 + 1)).prod

theorem num_factors_of_72 :
  num_factors 72 = 12 :=
by
  -- Lean specific details to calculate the number of factors based on prime factorization
  have prime_factors : 72.factorization = [(2, 3), (3, 2)].to_finmap,
  by sorry,
  rw [num_factors, prime_factors],
  norm_num

end num_factors_of_72_l681_681703


namespace minimum_guests_at_banquet_l681_681414

theorem minimum_guests_at_banquet (total_food : ℝ) (max_food_per_guest : ℝ) (min_guests : ℕ) 
  (h1 : total_food = 411) (h2 : max_food_per_guest = 2.5) : min_guests = 165 :=
by
  -- Proof omitted
  sorry

end minimum_guests_at_banquet_l681_681414


namespace solve_inequality_l681_681840

theorem solve_inequality (x : ℝ) : 
  3 - (1 / (3 * x + 4)) < 5 ↔ x ∈ set.Ioo (-∞ : ℝ) (-7 / 6) ∪ set.Ioo (-4 / 3) ∞ := 
sorry

end solve_inequality_l681_681840


namespace num_valid_paths_from_A_to_B_l681_681247

-- Define the points as an enumeration
inductive Point
| A | B | C | D | E | F | G
deriving DecidableEq

open Point

-- Define the edges as a set of pairs of points
def edges : Finset (Point × Point) :=
  {(A, C), (A, D), (A, 1.5), (C, B), (D, C), (D, F), (D, E), (E, F), (F, G), (G, B) : (D,1.5)}

-- Define a path as a list of points
def path := List Point

-- Function to check if a path is valid (i.e., uses edges and doesn't revisit points)
def is_valid_path (p : path) : Prop :=
  p.head = A ∧ p.last = B ∧
  (∀ (i : ℕ), i < p.length - 1 → (p.nth i, p.nth (i + 1)) ∈ edges) ∧
  (p.nodup)

-- Define the set of all valid paths from A to B
def valid_paths : Finset path := {
  -- Placeholder: the actual construction of valid_paths would be a function
  -- generating all possible paths from A to B and filtering out invalid ones.
}

-- The theorem stating the number of valid paths
theorem num_valid_paths_from_A_to_B : valid_paths.card = 12 := 
by sorry

end num_valid_paths_from_A_to_B_l681_681247


namespace range_of_slope_angle_l681_681821

open Real

theorem range_of_slope_angle (x α : ℝ) (h₁ : 0 ≤ α) (h₂ : α < π) (h₃ : α = atan (- sqrt 3 * sin x)) :
  α ∈ Icc 0 (π / 3) ∪ Ico (2 * π / 3) π :=
by
  sorry

end range_of_slope_angle_l681_681821


namespace tangent_line_eq_number_of_zeros_l681_681662

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ :=
  2 * a^2 * Real.log x - x^2

-- Define the derivative of f(x)
def f' (x : ℝ) (a : ℝ) : ℝ :=
  (2 - 2 * x^2) / x

-- Prove the equation of the tangent line at x = 1 when a = 1
theorem tangent_line_eq (x : ℝ) (hx : x = 1) (a : ℝ) (ha : a = 1) :
  f' 1 1 = 0 ∧ f 1 1 = -1 ∧ (∃ C : ℝ, ∀ y : ℝ, y = f 1 1 + f' 1 1 * (x - 1) -> y + 1 = C) :=
  sorry

-- Define the number of zeros of f(x) in the interval (1, e^2) based on the value of a
theorem number_of_zeros (a : ℝ) (h₀ : 0 < a) :
  (0 < a ∧ a < Real.sqrt Real.exp 1 -> ∀ x : ℝ, x > 1 ∧ x < Real.exp 2 -> f x a ≠ 0) ∧
  (a = Real.sqrt Real.exp 1 ∨ a ≥ Real.exp 2 / 2 -> ∃ x : ℝ, x > 1 ∧ x < Real.exp 2 ∧ f x a = 0 ∧ ∀ y : ℝ, y ≠ x -> y > 1 ∧ y < Real.exp 2 ∧ f y a ≠ 0) ∧
  (Real.sqrt Real.exp 1 < a ∧ a < Real.exp 2 / 2 -> ∃ x1 x2 : ℝ, x1 > 1 ∧ x1 < a ∧ x2 > a ∧ x2 < Real.exp 2 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧ ∀ y : ℝ, y ≠ x1 ∧ y ≠ x2 -> y > 1 ∧ y < Real.exp 2 ∧ f y a ≠ 0) :=
  sorry

end tangent_line_eq_number_of_zeros_l681_681662


namespace combination_18_6_l681_681501

theorem combination_18_6 : (nat.choose 18 6) = 18564 := 
by 
  sorry

end combination_18_6_l681_681501


namespace area_under_parabola_l681_681575

theorem area_under_parabola (a : ℝ) (a_pos : a > 0) : 
  let f := λ x : ℝ, x^2
  ∃ S : ℝ, (S = (∫ x in 0..a, f x)) ∧ S = a^3 / 3 := 
sorry

end area_under_parabola_l681_681575


namespace unique_solution_p_l681_681158

theorem unique_solution_p (p : ℚ) :
  (∀ x : ℝ, (2 * x + 3) / (p * x - 2) = x) ↔ p = -4 / 3 := sorry

end unique_solution_p_l681_681158


namespace ratio_of_marbles_given_l681_681487

-- Define the initial number of marbles
def initial_marbles : ℕ := 25

-- Condition: Baez loses 20% of her marbles
def lost_marbles : ℕ := (20 * initial_marbles) / 100

-- Condition: Number of marbles Baez has after losing 20%
def marbles_after_loss : ℕ := initial_marbles - lost_marbles

-- Condition: Total number of marbles Baez has after receiving some from a friend
def final_marbles : ℕ := 60

-- Define the number of marbles given by the friend
def marbles_given_by_friend : ℕ := final_marbles - marbles_after_loss

-- Condition (Ratio calculation)
def ratio_given_to_remaining (given remaining: ℕ) : ℕ × ℕ :=
  let g := Nat.gcd given remaining in
  (given / g, remaining / g)

theorem ratio_of_marbles_given : ratio_given_to_remaining marbles_given_by_friend marbles_after_loss = (2, 1) :=
by
  rw [marbles_given_by_friend, marbles_after_loss, lost_marbles]
  have h1: marbles_after_loss = 20 := by
    unfold marbles_after_loss lost_marbles
    norm_num
  have h2 : marbles_given_by_friend = 40 := by
    rw [marbles_after_loss, <- Nat.sub_add_lost_marbles h1]
    norm_num
  rw [h1, h2]
  have gcd_40_20 : Nat.gcd 40 20 = 20 := by
    norm_num
  rw [ratio_given_to_remaining, gcd_40_20]
  norm_num
  rfl

end ratio_of_marbles_given_l681_681487


namespace mame_on_top_probability_l681_681018

noncomputable def probability_mame_on_top : ℝ := 
  let total_parts : ℕ := 8
  in 1 / total_parts

theorem mame_on_top_probability : 
  probability_mame_on_top = 1 / 8 := 
by 
  sorry

end mame_on_top_probability_l681_681018


namespace floor_of_47_l681_681558

theorem floor_of_47 : int.floor 4.7 = 4 :=
sorry

end floor_of_47_l681_681558


namespace binom_18_6_eq_18564_l681_681508

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binom_18_6_eq_18564 : binomial 18 6 = 18564 := by
  sorry

end binom_18_6_eq_18564_l681_681508


namespace area_enclosed_by_curve_l681_681354

theorem area_enclosed_by_curve :
  ∃ (area : ℝ), (∀ (x y : ℝ), |x - 1| + |y - 1| = 1 → area = 2) :=
sorry

end area_enclosed_by_curve_l681_681354


namespace infinitely_many_heinersch_numbers_l681_681019

def is_heinersch (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^3

theorem infinitely_many_heinersch_numbers :
  ∃ (H : ℕ → ℕ), (∀ (n : ℕ), is_heinersch (H n) ∧ is_heinersch (H n - 1) ∧ is_heinersch (H n + 1)) :=
begin
  sorry,
end

end infinitely_many_heinersch_numbers_l681_681019


namespace avoid_mistakes_l681_681034

-- Definitions for conditions
def minions_shared_info (m : string) (website : string) := m = "Gru's phone number" ∧ website = "untrusted website"

def minions_downloaded_exe (file : string) := file = "banana_cocktail.pdf.exe"

-- Assumption representing the mistakes
axiom minions_mistake (m : string) (w : string) (f : string) :
  minions_shared_info m w ∧ minions_downloaded_exe f

-- Theorem to prove the correct answer
theorem avoid_mistakes (share_info : ∀ (m : string) (w : string), ¬ (minions_shared_info m w))
  (download_file : ∀ (f : string), ¬ (minions_downloaded_exe f)) :
  ∀ (m : string) (w : string) (f : string), ¬ (minions_shared_info m w ∧ minions_downloaded_exe f) :=
by
  intros
  sorry

end avoid_mistakes_l681_681034


namespace greatest_3_digit_base8_num_div_by_7_eq_511_l681_681922

noncomputable def greatest_base8_number_divisible_by_7 : ℕ := 7 * 73

theorem greatest_3_digit_base8_num_div_by_7_eq_511 : 
  greatest_base8_number_divisible_by_7 = 511 :=
by 
  sorry

end greatest_3_digit_base8_num_div_by_7_eq_511_l681_681922


namespace proof_q_values_proof_q_comparison_l681_681108

-- Definitions of the conditions given.
def q : ℝ → ℝ := 
  sorry -- The definition is not required to be constructed, as we are only focusing on the conditions given.

-- Conditions
axiom cond1 : q 2 = 5
axiom cond2 : q 1.5 = 3

-- Statements to prove
theorem proof_q_values : (q 2 = 5) ∧ (q 1.5 = 3) := 
  by sorry

theorem proof_q_comparison : q 2 > q 1.5 :=
  by sorry

end proof_q_values_proof_q_comparison_l681_681108


namespace solution_set_of_inequality_l681_681720

noncomputable def f : ℝ → ℝ := sorry 

axiom f_cond : ∀ x : ℝ, f x + deriv f x > 1
axiom f_at_zero : f 0 = 4

theorem solution_set_of_inequality : {x : ℝ | f x > 3 / Real.exp x + 1} = { x : ℝ | x > 0 } :=
by
  sorry

end solution_set_of_inequality_l681_681720


namespace number_of_triangles_with_perimeter_11_l681_681684

theorem number_of_triangles_with_perimeter_11 : (∃ triangles : List (ℕ × ℕ × ℕ), 
  (∀ t ∈ triangles, let (a, b, c) := t in 
    a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ a + c > b) 
  ∧ triangles.length = 10) := 
sorry

end number_of_triangles_with_perimeter_11_l681_681684


namespace net_calorie_deficit_l681_681152

-- Define the conditions as constants.
def total_distance : ℕ := 3
def calories_burned_per_mile : ℕ := 150
def calories_in_candy_bar : ℕ := 200

-- Prove the net calorie deficit.
theorem net_calorie_deficit : total_distance * calories_burned_per_mile - calories_in_candy_bar = 250 := by
  sorry

end net_calorie_deficit_l681_681152


namespace exists_infinite_addition_l681_681757

-- Define the function f(n) that adds to n the product of all its nonzero digits
def f (n : Nat) : Nat :=
  n + (n.digits.filter (≠ 0)).prod

-- State the formal theorem
theorem exists_infinite_addition (n : Nat) (h : n > 0) : 
  ∃ x > 0, ∀ k : Nat, ∃ m : Nat, n + k * x = m + f(m) :=
sorry

end exists_infinite_addition_l681_681757


namespace minimal_M_exists_l681_681157

theorem minimal_M_exists :
  ∃ (M n : ℕ), M > 0 ∧ n > 0 ∧
  (M = (∑ i in Finset.range (n + 1), (i + x)^2) for some x in  ℕ) ∧
  (2 * M = (∑ i in Finset.range (2 * n + 1), (i + y)^2) for some y in ℕ) ∧
  ∀ (M' n' : ℕ), M' > 0 ∧ n' > 0 ∧
  (M' = (∑ i in Finset.range (n' + 1), (i + x')^2) for some x' in ℕ) ∧
  (2 * M' = (∑ i in Finset.range (2 * n' + 1), (i + y')^2) for some y' in ℕ) →
  M ≤ M' :=
begin
    sorry
end

end minimal_M_exists_l681_681157


namespace part_I_part_II_l681_681268

-- Conditions
variables {A B C : ℝ} {a b c S : ℝ}

-- Given conditions
axiom cos_C_cond : cos C = 4 / 5
axiom c_2b_cos_A_cond : c = 2 * b * cos A
axiom area_cond : S = 15 / 2

-- Questions to be proved
theorem part_I (h1 : 0 < A) (h2 : A < real.pi) (h3 : 0 < B) (h4 : B < real.pi) 
  (h5 : sin (A - B) = 0) : A = B := sorry

theorem part_II (h1 : S = (1 / 2) * a * b * sqrt (1 - (4 / 5)^2))
  (h2 : a = b) : c = sqrt 10 := sorry

end part_I_part_II_l681_681268


namespace probability_fourth_roll_six_l681_681153

noncomputable def fair_die_prob : ℚ := 1 / 6
noncomputable def biased_die_prob : ℚ := 3 / 4
noncomputable def biased_die_other_face_prob : ℚ := 1 / 20
noncomputable def prior_prob : ℚ := 1 / 2

def p := 41
def q := 67

theorem probability_fourth_roll_six (p q : ℕ) (h1 : fair_die_prob = 1 / 6) (h2 : biased_die_prob = 3 / 4) (h3 : prior_prob = 1 / 2) :
  p + q = 108 :=
sorry

end probability_fourth_roll_six_l681_681153


namespace solution_set_l681_681786

variable {ℝ : Type*} [OrderedField ℝ] [Real ℝ]

noncomputable def f : ℝ → ℝ := sorry -- placeholder for the actual function definition

axiom even_f : ∀ x, f x = f (-x)
axiom f_one_zero : f 1 = 0
axiom condition_1 : ∀ x > 0, x * deriv (deriv f) x - f x > 0

theorem solution_set :
  {x : ℝ | x * f x > 0} = {x : ℝ | -1 < x ∧ x < 0 ∨ 1 < x} :=
by {
  -- proof goes here
  sorry
}

end solution_set_l681_681786


namespace angle_DFE_90_degrees_l681_681888

-- Definitions
variables (A B C D E F : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]

/-- Circles centered at A and B with given radii -/
def circleA := {P : Type | dist P A = 3}
def circleB := {P : Type | dist P B = 2}

/-- Point C is the internal touching point of the two circles -/
def touching_internally_C := C ∈ circleA ∧ C ∈ circleB ∧ dist A B = abs (3 - 2)

/-- Line AB is extended to meet the larger circle at D and the smaller circle at E -/
def line_AB_extended_AD := D ∈ line AB ∧ D ∈ circleA
def line_AB_extended_BE := E ∈ line AB ∧ E ∈ circleB

/-- The circles intersect at another point F -/
def circles_intersect_F := F ∈ circleA ∧ F ∈ circleB ∧ F ≠ C 

/-- Prove the measure of angle DFE is 90 degrees -/
theorem angle_DFE_90_degrees : touching_internally_C A B C → line_AB_extended_AD A B D → line_AB_extended_BE A B E → circles_intersect_F A B F → 
  ∠ D F E = 90 :=
sorry

end angle_DFE_90_degrees_l681_681888


namespace greatest_possible_x_for_equation_l681_681025

theorem greatest_possible_x_for_equation :
  ∃ x, (x = (9 : ℝ) / 5) ∧ 
  ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20 := by
  sorry

end greatest_possible_x_for_equation_l681_681025


namespace value_of_a2014_l681_681672

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = 1 / (a n - 1) + 1

theorem value_of_a2014 (a : ℕ → ℚ) (h : sequence a) : a 2014 = 3 / 2 :=
by
  sorry

end value_of_a2014_l681_681672


namespace csc_product_identity_l681_681929

theorem csc_product_identity :
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ (m ^ n = ∏ k in finset.range 30, (real.csc (real.pi / 60 * (3 * (k + 1))))^2) ∧ (m + n = 61) :=
by
  sorry

end csc_product_identity_l681_681929


namespace range_set_A_l681_681587

def is_prime (n : ℕ) : Prop := (∀ m : ℕ, 1 < m → m < n → n % m ≠ 0)

def set_A : set ℕ := {n | n > 15 ∧ n < 100 ∧ is_prime n}

noncomputable def range_of_set (s : set ℕ) : ℕ :=
  if h : s.nonempty then
    let min := s.to_finset.min' h
    let max := s.to_finset.max' h
    max - min
  else 0

theorem range_set_A : range_of_set set_A = 80 := by
  sorry

end range_set_A_l681_681587


namespace min_triples_in_colored_complete_graph_l681_681591

theorem min_triples_in_colored_complete_graph (k: ℕ) (n: ℕ) (r: ℕ)
  (h_n: n = 36)
  (h_r: r = 5)
  (h_k: k = 3780) :
  ∃ G : simple_graph (fin n), 
  G = complete_graph (fin n) ∧
  ∀ c : edge_coloring G r,
  ∃ t : set (fin n × fin n × fin n),
  (∀ A B C : fin n, (A, B, C) ∈ t → 
    (G.edge_set (A, B) ∧ G.edge_set (B, C) ∧ 
    c (A, B) = c (B, C))) ∧
  t.card ≥ k :=
by sorry

end min_triples_in_colored_complete_graph_l681_681591


namespace express_x_in_terms_of_y_l681_681564

theorem express_x_in_terms_of_y (x y : ℝ) (h : 2 * x - 3 * y = 7) : x = 7 / 2 + 3 / 2 * y :=
by
  sorry

end express_x_in_terms_of_y_l681_681564


namespace sum_arithmetic_seq_l681_681133

theorem sum_arithmetic_seq (a d n : ℕ) :
  a = 2 → d = 2 → a + (n - 1) * d = 20 → (n / 2) * (a + (a + (n - 1) * d)) = 110 :=
by sorry

end sum_arithmetic_seq_l681_681133


namespace number_of_triangles_with_perimeter_11_l681_681697

theorem number_of_triangles_with_perimeter_11 :
  {t : (ℕ × ℕ × ℕ) // let (a, b, c) := t in a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ c + a > b}.card = 4 :=
by sorry

end number_of_triangles_with_perimeter_11_l681_681697


namespace arithmetic_sequence_sum_l681_681127

theorem arithmetic_sequence_sum :
  let sequence := list.range (20 / 2) in
  let sum := sequence.map (λ n, 2 * (n + 1)).sum in
  sum = 110 :=
by
  -- Define the sequence as the arithmetic series
  let sequence := list.range (20 / 2)
  -- Calculate the sum of the arithmetic sequence
  let sum := sequence.map (λ n, 2 * (n + 1)).sum
  -- Check the sum
  have : sum = 110 := sorry
  exact this

end arithmetic_sequence_sum_l681_681127


namespace arithmetic_sequence_sum_l681_681129

theorem arithmetic_sequence_sum :
  let sequence := list.range (20 / 2) in
  let sum := sequence.map (λ n, 2 * (n + 1)).sum in
  sum = 110 :=
by
  -- Define the sequence as the arithmetic series
  let sequence := list.range (20 / 2)
  -- Calculate the sum of the arithmetic sequence
  let sum := sequence.map (λ n, 2 * (n + 1)).sum
  -- Check the sum
  have : sum = 110 := sorry
  exact this

end arithmetic_sequence_sum_l681_681129


namespace distance_P_to_origin_l681_681365

-- Define the point P with coordinates (1, 2, 2)
def P : ℝ × ℝ × ℝ := (1, 2, 2)

-- Define the distance formula in 3D space
def distance_to_origin (P : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (P.1 ^ 2 + P.2 ^ 2 + P.3 ^ 2)

-- The problem statement
theorem distance_P_to_origin : distance_to_origin P = 3 := sorry

end distance_P_to_origin_l681_681365


namespace can_write_13121_not_12131_l681_681245

theorem can_write_13121_not_12131 :
  ∀ (a b : ℕ), (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨ ∃ (m n : ℕ), (a = nat.succ 0 ∧ b = nat.succ 1) ∨ (a = nat.succ 1 ∧ b = nat.succ 0) →
  (∃ k l : ℕ, a*b + a + b = 13120 → (k = 2 ∧ l = 3^8)) ∧
  ¬ (∃ k l : ℕ, a*b + a + b = 12130 → (k = 2^2 ∧ l = 3^2 * nat.prime_pos.337)) :=
by
  sorry

end can_write_13121_not_12131_l681_681245


namespace range_of_function_l681_681368

noncomputable def f (x : ℕ) := x^2 - 2*x

def domain := {0, 1, 2, 3}

def range (f : ℕ → ℤ) (domain : set ℕ) : set ℤ :=
  {y | ∃ x ∈ domain, y = f x}

theorem range_of_function : range f domain = {-1, 0, 3} :=
by
  sorry

end range_of_function_l681_681368


namespace find_a_l681_681223

variable (ξ : ℕ → ℝ)

def distribution (a : ℝ) : Prop :=
  ∀ i, i ∈ {1, 2, 3} → ξ i = a * (1 / 3) ^ i

theorem find_a (a : ℝ) (h : distribution ξ a) : a = 27 / 13 := 
by 
  sorry

end find_a_l681_681223


namespace area_of_triangle_AEB_l681_681282

theorem area_of_triangle_AEB :
  ∀ (A B C D F G E : Type) (AB BC DF GC : ℝ),
  is_rectangle A B C D →
  distance A B = 7 →
  distance B C = 4 →
  point_on_line_segment D F CD →
  point_on_line_segment C G CD →
  distance D F = 2 →
  distance G C = 1 →
  lines_intersect_at AF BG E →
  area_of_triangle A E B = 22.4 :=
by
  sorry

end area_of_triangle_AEB_l681_681282


namespace avoid_mistakes_l681_681033

-- Definitions for conditions
def minions_shared_info (m : string) (website : string) := m = "Gru's phone number" ∧ website = "untrusted website"

def minions_downloaded_exe (file : string) := file = "banana_cocktail.pdf.exe"

-- Assumption representing the mistakes
axiom minions_mistake (m : string) (w : string) (f : string) :
  minions_shared_info m w ∧ minions_downloaded_exe f

-- Theorem to prove the correct answer
theorem avoid_mistakes (share_info : ∀ (m : string) (w : string), ¬ (minions_shared_info m w))
  (download_file : ∀ (f : string), ¬ (minions_downloaded_exe f)) :
  ∀ (m : string) (w : string) (f : string), ¬ (minions_shared_info m w ∧ minions_downloaded_exe f) :=
by
  intros
  sorry

end avoid_mistakes_l681_681033


namespace find_BP_find_QT_l681_681283

-- Definitions for conditions
variables (A B C D P Q T S R : Point)
variables (PA AQ QP AB : ℝ)
variable [EuclideanGeometry]
(noncomputable def PA_value : PA := 24)
(noncomputable def AQ_value : AQ := 7)
(noncomputable def QP_value : QP := 25)

-- Given conditions
axiom rectangle_ABCD : is_rectangle A B C D
axiom P_on_BC : P ∈ BC
axiom angle_APD_90 : ∠ APD = 90
axiom perpendicular_TS_BC : perpendicular TS BC
axiom BP_eq_PT : BP = PT
axiom PD_intersects_TS_at_Q : PD ∩ TS = Q
axiom R_on_CD : R ∈ CD
axiom RA_passes_through_Q : R ∈ line_through A Q
axiom PA_def : PA = PA_value
axiom AQ_def : AQ = AQ_value
axiom QP_def : QP = QP_value

-- Required proofs
theorem find_BP : BP = sqrt 351 := sorry
theorem find_QT : QT = sqrt 274 := sorry

end find_BP_find_QT_l681_681283


namespace sum_arithmetic_sequence_l681_681118

theorem sum_arithmetic_sequence : ∀ (a d l : ℕ), 
  (d = 2) → (a = 2) → (l = 20) → 
  ∃ (n : ℕ), (l = a + (n - 1) * d) ∧ 
  (∑ k in Finset.range n, (a + k * d)) = 110 :=
by
  intros a d l h_d h_a h_l
  use 10
  split
  · sorry
  · sorry

end sum_arithmetic_sequence_l681_681118


namespace simplify_complex_fraction_l681_681836

theorem simplify_complex_fraction :
  (⟨3, 5⟩ : ℂ) / (⟨-2, 7⟩ : ℂ) = (29 / 53) - (31 / 53) * I :=
by sorry

end simplify_complex_fraction_l681_681836


namespace binom_18_6_l681_681510

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_18_6 : binomial 18 6 = 18564 := 
by
  sorry

end binom_18_6_l681_681510


namespace distance_between_chords_l681_681935

theorem distance_between_chords (R AB CD : ℝ) (hR : R = 15) (hAB : AB = 18) (hCD : CD = 24) : 
  ∃ d : ℝ, d = 21 :=
by 
  sorry

end distance_between_chords_l681_681935


namespace internet_service_charge_l681_681179

theorem internet_service_charge (C1 C2: ℝ) (x: ℝ) :
  (C2 = 2.5 * C1) →
  (40 = C1 + 5 * x + 0.05 * (C1 + 5 * x)) →
  (76 = C2 + 8 * x + 0.08 * (C2 + 8 * x)) →
  x ≈ 5.525 :=
by
  sorry

end internet_service_charge_l681_681179


namespace number_of_whole_numbers_diagonal_AC_l681_681597

theorem number_of_whole_numbers_diagonal_AC 
  (x : ℝ) (h1: x > 4) (h2: x < 18) (h3: x > 4) (h4: x < 30) : 
  13 :=
by
  sorry

end number_of_whole_numbers_diagonal_AC_l681_681597


namespace median_mode_of_shots_l681_681737

def number_of_shots : List ℕ := [6, 6, 7, 7, 8, 8, 8, 9]

theorem median_mode_of_shots :
  List.median number_of_shots = 7.5 ∧ List.mode number_of_shots = 8 :=
by
  sorry

end median_mode_of_shots_l681_681737


namespace simplify_radicals_l681_681344

theorem simplify_radicals : 
  let a := Real.sqrt (4 - 2 * Real.sqrt 3)
  let b := Real.sqrt (4 + 2 * Real.sqrt 3)
  a - b = -2 := 
by
  let a := Real.sqrt (4 - 2 * Real.sqrt 3)
  let b := Real.sqrt (4 + 2 * Real.sqrt 3)
  have h1 : a = Real.sqrt (Real.sqrt 3 - 1)^2 := sorry
  have h2 : b = Real.sqrt (Real.sqrt 3 + 1)^2 := sorry
  have h3 : Real.sqrt (Real.sqrt 3 - 1)^2 = Real.sqrt 3 - 1 := sorry
  have h4 : Real.sqrt (Real.sqrt 3 + 1)^2 = Real.sqrt 3 + 1 := sorry
  calc a - b
         = (Real.sqrt 3 - 1) - (Real.sqrt 3 + 1) : by rw [←h1, ←h2, h3, h4]
     ... = -2 : by ring

end simplify_radicals_l681_681344


namespace number_of_n_tronder_walks_l681_681947

def tronder_walk (n : ℕ) : ℕ :=
  -- Calculate the double factorial (2n - 1)!!
  if n == 0 then 1
  else (2 * n - 1) * tronder_walk(n - 1)

theorem number_of_n_tronder_walks (n : ℕ) :
  (n : ℕ) ≥ 0 →
  let count_tronder_walks := tronder_walk n in
  count_tronder_walks = (2 * n - 1)!! := 
sorry

end number_of_n_tronder_walks_l681_681947


namespace coeff_x3_in_expansion_l681_681852

theorem coeff_x3_in_expansion :
  (∑ i in Finset.range 8, 
    (∑ j in Finset.range (i + 1), 
      (binomial 7 i * binomial i j * (-2)^j * (x^(1/2))^(i - j) * x^(-j))) = x^3) * 7 = 7 
:= 
sorry

end coeff_x3_in_expansion_l681_681852


namespace points_A_B_D_collinear_l681_681602

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables {P : Type*} [affine_space V P]

-- Given conditions
variables (a b : V) (A B C D : P)
variables (h1 : P)

-- Equivalent vector conditions
def AB := a + 5 • b
def BC := -2 • a + 8 • b
def CD := 3 • a - 3 • b
def BD := BC + CD

-- Proof statement
theorem points_A_B_D_collinear :
  AB = BD →
  collinear ℝ ({A, B, D} : set P) :=
by
  sorry

end points_A_B_D_collinear_l681_681602


namespace largest_n_satisfying_inequality_l681_681581

theorem largest_n_satisfying_inequality : 
  ∃ (n : ℕ), (∀ k : ℕ, (8 : ℚ) / 15 < n / (n + k) ∧ n / (n + k) < (7 : ℚ) / 13) ∧ 
  ∀ n' : ℕ, (∀ k : ℕ, (8 : ℚ) / 15 < n' / (n' + k) ∧ n' / (n' + k) < (7 : ℚ) / 13) → n' ≤ n :=
sorry

end largest_n_satisfying_inequality_l681_681581


namespace sum_arithmetic_sequence_l681_681119

theorem sum_arithmetic_sequence : ∀ (a d l : ℕ), 
  (d = 2) → (a = 2) → (l = 20) → 
  ∃ (n : ℕ), (l = a + (n - 1) * d) ∧ 
  (∑ k in Finset.range n, (a + k * d)) = 110 :=
by
  intros a d l h_d h_a h_l
  use 10
  split
  · sorry
  · sorry

end sum_arithmetic_sequence_l681_681119


namespace gcd_294_84_l681_681012

-- Define the numbers for the GCD calculation
def a : ℕ := 294
def b : ℕ := 84

-- Define the greatest common divisor function using Euclidean algorithm
def gcd_euclidean : ℕ → ℕ → ℕ
| x, 0 => x
| x, y => gcd_euclidean y (x % y)

-- Theorem stating that the GCD of 294 and 84 is 42
theorem gcd_294_84 : gcd_euclidean a b = 42 :=
by
  -- Proof is omitted
  sorry

end gcd_294_84_l681_681012


namespace distance_from_C_to_B_is_80_l681_681890

theorem distance_from_C_to_B_is_80
  (x : ℕ)
  (h1 : x = 60)
  (h2 : ∀ (ab cb : ℕ), ab = x → cb = x + 20  → (cb = 80))
  : x + 20 = 80 := by
  sorry

end distance_from_C_to_B_is_80_l681_681890


namespace arithmetic_series_sum_l681_681126

theorem arithmetic_series_sum : 
  let a := 2 in 
  let d := 2 in 
  let n := 10 in 
  let l := 20 in 
  (a + l) * n / 2 = 110 := 
by
  sorry

end arithmetic_series_sum_l681_681126


namespace three_lines_determining_plane_l681_681284

theorem three_lines_determining_plane (L1 L2 L3 : set ℝ^3) 
  (h1 : ∃ p1, p1 ∈ L1 ∩ L2)
  (h2 : ∃ p2, p2 ∈ L2 ∩ L3)
  (h3 : ∃ p3, p3 ∈ L3 ∩ L1)
  (h4 : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) : 
  ∃ P : set (set ℝ^3), ∀ (p : ℝ^3), p ∈ P ↔ (p ∈ p1 ∨ p ∈ p2 ∨ p ∈ p3) := 
sorry

end three_lines_determining_plane_l681_681284


namespace monotonicity_on_interval_max_value_on_interval_min_value_on_interval_l681_681231

noncomputable def f (x : ℝ) : ℝ := Real.log(2 * x + 3) + x^2

theorem monotonicity_on_interval (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : 
    ∀ x, 0 ≤ x ∧ x ≤ 1 → f' x > 0 := sorry

theorem max_value_on_interval : f 1 = Real.log 5 + 1 := sorry

theorem min_value_on_interval : f 0 = Real.log 3 := sorry

end monotonicity_on_interval_max_value_on_interval_min_value_on_interval_l681_681231


namespace geometric_sequences_product_and_quotient_l681_681636

variable {R : Type*} [Field R]

-- Defining sequences {a_n} and {b_n} as geometric sequences
def is_geometric_sequence (a : ℕ → R) (q : R) : Prop :=
  ∀ n, a (n + 1) = q * a n

variable (a b : ℕ → R)
variable (q1 q2 : R)

-- Defining the conditions that {a_n} and {b_n} are geometric sequences with common ratios q1 and q2 respectively
def a_is_geometric : Prop := is_geometric_sequence a q1
def b_is_geometric : Prop := is_geometric_sequence b q2

-- The statement that {a_n * b_n} is a geometric sequence
def product_geometric_sequence : Prop :=
  ∃ q, is_geometric_sequence (λ n, a n * b n) q

-- The statement that {a_n / b_n} is a geometric sequence
def quotient_geometric_sequence : Prop :=
  ∃ q, is_geometric_sequence (λ n, a n / b n) q

-- The main theorem combining the conditions and the proof statement
theorem geometric_sequences_product_and_quotient
  (ha : a_is_geometric a q1)
  (hb : b_is_geometric b q2) :
  product_geometric_sequence a b ∧ quotient_geometric_sequence a b :=
sorry

end geometric_sequences_product_and_quotient_l681_681636


namespace count_triangles_with_perimeter_11_l681_681692

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem count_triangles_with_perimeter_11 :
  {t : (ℕ × ℕ × ℕ) // let ⟨a, b, c⟩ := t in a + b + c = 11 ∧ is_triangle a b c}.to_finset.card = 9 :=
by sorry

end count_triangles_with_perimeter_11_l681_681692


namespace maximum_reflections_l681_681074

theorem maximum_reflections (θ : ℕ) (h : θ = 10) (max_angle : ℕ) (h_max : max_angle = 180) : 
∃ n : ℕ, n ≤ max_angle / θ ∧ n = 18 := by
  sorry

end maximum_reflections_l681_681074


namespace range_of_finequality_l681_681608

variable {f : ℝ → ℝ}

def differentiability (f : ℝ → ℝ) :=
  ∃ f', ∀ x, f' x = deriv f x

theorem range_of_finequality (h_diff : differentiability f)
  (h_deriv : ∀ x, deriv f x < 1)
  (h_f3 : f 3 = 4) :
  ∀ x, f (x + 1) < x + 2 ↔ x > 2 :=
by
  sorry

end range_of_finequality_l681_681608


namespace smallest_possible_positive_value_of_w_l681_681845

noncomputable def smallest_positive_w (y w: ℝ) : Prop :=
  sin y = 0 ∧ sin (y + w) = real.sqrt 3 / 2 → w = real.pi / 3

-- The actual theorem statement
theorem smallest_possible_positive_value_of_w (y w: ℝ) :
  smallest_positive_w y w :=
by sorry

end smallest_possible_positive_value_of_w_l681_681845


namespace best_fit_model_l681_681274

theorem best_fit_model 
  (R2_model1 : ℝ)
  (R2_model2 : ℝ)
  (R2_model3 : ℝ)
  (R2_model4 : ℝ)
  (h1 : R2_model1 = 0.87)
  (h2 : R2_model2 = 0.97)
  (h3 : R2_model3 = 0.50)
  (h4 : R2_model4 = 0.25) : 
  R2_model2 > R2_model1 ∧ R2_model2 > R2_model3 ∧ R2_model2 > R2_model4 :=
by 
  -- According to given conditions
  rw [h1, h2, h3, h4],
  -- Now it suffices to prove 0.97 > 0.87 ∧ 0.97 > 0.50 ∧ 0.97 > 0.25
  exact ⟨by linarith, by linarith, by linarith⟩

end best_fit_model_l681_681274


namespace find_k_l681_681667

-- Define the conditions of the problem as hypotheses
variables (k : ℝ) 

-- The statement to be proven
theorem find_k (h1 : ∀ (x y : ℝ), (x ^ 2 + y ^ 2 = 9) → (k * x + y = 9))
                (h2 : ∀ (A B O : ℝ), (O = 0) → (A = B) → (A ≠ O) → (AB.1 ^ 2 + AB.2 ^ 2 = 27)) :
  k = sqrt 11 ∨ k = -sqrt 11 :=
sorry

end find_k_l681_681667


namespace find_angle_y_l681_681287

-- Definitions of the angles in the triangle
def angle_ACD : ℝ := 90
def angle_DEB : ℝ := 58

-- Theorem proving the value of angle DCE (denoted as y)
theorem find_angle_y (angle_sum_property : angle_ACD + y + angle_DEB = 180) : y = 32 :=
by sorry

end find_angle_y_l681_681287


namespace bill_took_six_naps_l681_681490

def total_hours (days : Nat) : Nat := days * 24

def hours_left (total : Nat) (worked : Nat) : Nat := total - worked

def naps_taken (remaining : Nat) (duration : Nat) : Nat := remaining / duration

theorem bill_took_six_naps :
  let days := 4
  let hours_worked := 54
  let nap_duration := 7
  naps_taken (hours_left (total_hours days) hours_worked) nap_duration = 6 := 
by {
  sorry
}

end bill_took_six_naps_l681_681490


namespace arithmetic_sequence_problem_l681_681616

theorem arithmetic_sequence_problem 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ) 
  (b : ℕ → ℝ) 
  (T : ℕ → ℝ) 
  (h1 : a 3 = 7) 
  (h2 : a 5 + a 7 = 26) 
  (h3 : ∀ n, S n = (n * (2 * n + 4)) / 2) 
  (h4 : ∀ n, b n = - (1 / (a n ^ 2 - 1))) 
  (h5 : ∀ n, T n = ∑ i in Finset.range n, b (i + 1)) :
  (∀ n, a n = 2 * n + 1) ∧ 
  (∀ n, S n = n^2 + 2 * n) ∧ 
  (∀ n, T n = - (n : ℝ) / (4 * (n + 1))) :=
by
  sorry

end arithmetic_sequence_problem_l681_681616


namespace binom_18_6_l681_681513

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_18_6 : binomial 18 6 = 18564 := 
by
  sorry

end binom_18_6_l681_681513


namespace num_elements_in_M_l681_681809

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {5, 7}
def M : Set ℕ := {x | ∃ a ∈ A, ∃ b ∈ B, x = a * b}

theorem num_elements_in_M : M.toFinset.card = 6 :=
  sorry

end num_elements_in_M_l681_681809


namespace probability_same_stock_probability_two_or_less_same_stock_l681_681177

noncomputable def probability_all_four_same (stocks : Finset ℕ) (individuals : Finset ℕ) : ℚ :=
if individuals.card = 4 ∧ stocks.card = 6 then 6 * (1 / 6)^(4 - 1) else 0

noncomputable def probability_at_most_two_same (stocks : Finset ℕ) (individuals : Finset ℕ) : ℚ :=
if individuals.card = 4 ∧ stocks.card = 6 then
  let term1 := (nat.choose 4 2 * nat.choose 2 2 / nat.perm 2 2 * nat.perm 6 2) / 6^4 in
  let term2 := (nat.choose 4 2 * nat.perm 6 3) / 6^4 in
  let term3 := (nat.perm 6 4) / 6^4 in
  term1 + term2 + term3
else 0

theorem probability_same_stock :
  ∃ (stocks individuals : Finset ℕ), probability_all_four_same stocks individuals = 1 / 216 :=
sorry

theorem probability_two_or_less_same_stock :
  ∃ (stocks individuals : Finset ℕ), probability_at_most_two_same stocks individuals = 65 / 72 :=
sorry

end probability_same_stock_probability_two_or_less_same_stock_l681_681177


namespace incircle_radius_l681_681401

theorem incircle_radius {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (right_angle_at_C : ∠C = 90)
  (angle_A_45 : ∠A = 45)
  (AC_length : dist A C = 12) 
  : ∃ r : ℝ, r = 6 * (2 - Real.sqrt 2) :=
by 
  sorry

end incircle_radius_l681_681401


namespace problem1_problem2_l681_681416

-- Problem 1
theorem problem1 : (2 * Real.sqrt 12 - 3 * Real.sqrt (1 / 3)) * Real.sqrt 6 = 9 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) (h1 : x / (2 * x - 1) = 2 - 3 / (1 - 2 * x)) : x = -1 / 3 := by
  sorry

end problem1_problem2_l681_681416


namespace quadratic_roots_correctness_l681_681671

theorem quadratic_roots_correctness :
  let a := 2 * Real.sqrt 3 + Real.sqrt 2,
      b := 2 * (Real.sqrt 3 + Real.sqrt 2),
      c := Real.sqrt 2 - 2 * Real.sqrt 3,
      x1 := -b / a,
      x2 := c / a in
  x1 ≠ ((4 + Real.sqrt 6) / 5) ∧ x1 ≠ -2 * (Real.sqrt 3 + Real.sqrt 2) ∧
  x2 ≠ ((7 - 2 * Real.sqrt 6) / 5) ∧ x2 ≠ 2 * Real.sqrt 3 - Real.sqrt 2 →
  ((1 : ℕ) = (4 : ℕ)) ∨ ((1 : ℕ) = (1 : ℕ)) := by
  intros
  sorry

end quadratic_roots_correctness_l681_681671


namespace sum_of_coefficients_l681_681493

theorem sum_of_coefficients (x y : ℝ) :
  (x - 3 * y)^20 | x = 1, y = 1 = 1048576 := by
  sorry

end sum_of_coefficients_l681_681493


namespace entire_grid_black_probability_l681_681060

noncomputable def probability_entire_grid_black (p : ℕ → ℕ → ℚ) : ℚ :=
  let single_square := (1 / 2 : ℚ)
  let central_squares := single_square ^ 4
  let other_squares := (1 / 2 : ℚ)
  central_squares * other_squares ^ 12

theorem entire_grid_black_probability
  (p : ℕ → ℕ → ℚ) :
  probability_entire_grid_black p = 1 / 32 :=
by {
  unfold probability_entire_grid_black,
  sorry
}

end entire_grid_black_probability_l681_681060


namespace no_integers_satisfy_eq_l681_681567

theorem no_integers_satisfy_eq (m n : ℤ) : ¬ (m^3 = 4 * n + 2) := 
  sorry

end no_integers_satisfy_eq_l681_681567


namespace shaded_quadrilateral_area_maximum_shaded_area_l681_681391

-- Given definitions from conditions
def side_length := (3 : ℝ)
def quadratic_area_fn (x : ℝ) : ℝ :=
  (-x^2 + 3 * x + 9 / 2)

-- Part (a): Function that calculates the area of the shaded quadrilateral
theorem shaded_quadrilateral_area (x : ℝ) (hx : 0 ≤ x ∧ x ≤ side_length / 2) :
  (∃ f : ℝ → ℝ, f = quadratic_area_fn) :=
begin
  use quadratic_area_fn,
  sorry -- Proof of equivalence to problem conditions
end

-- Part (b): Maximum area
theorem maximum_shaded_area :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ side_length / 2 → quadratic_area_fn x ≤ 27 / 4) :=
begin
  sorry -- Proof that the maximum is achieved at the stated value
end

end shaded_quadrilateral_area_maximum_shaded_area_l681_681391


namespace binom_18_6_l681_681509

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_18_6 : binomial 18 6 = 18564 := 
by
  sorry

end binom_18_6_l681_681509


namespace sequence_term_sequence_sum_l681_681876

def a_seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else 3^(n-1)

def S_n (n : ℕ) : ℕ :=
  (3^n - 1) / 2

theorem sequence_term (n : ℕ) (h : n ≥ 1) :
  a_seq n = 3^(n-1) :=
sorry

theorem sequence_sum (n : ℕ) :
  S_n n = (3^n - 1) / 2 :=
sorry

end sequence_term_sequence_sum_l681_681876


namespace area_enclosed_by_sqrt_and_cube_l681_681492

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := x^3

theorem area_enclosed_by_sqrt_and_cube :
  ∫ x in 0..1, (f x - g x) = (5 : ℝ) / 12 := by
  sorry

end area_enclosed_by_sqrt_and_cube_l681_681492


namespace imaginary_part_of_complex_l681_681579

theorem imaginary_part_of_complex (z : ℂ) (i : ℂ) (h : i^2 = -1) : 
  z = (2 + i) / i → z.im = -2 :=
by
  intro hz
  rw hz
  have h1 : z = (2 + i) / i := hz
  sorry

end imaginary_part_of_complex_l681_681579


namespace shorter_trisector_length_l681_681623

noncomputable def triangle_shorter_trisector (DE EF : ℝ) (h : DE = 5 ∧ EF = 12) : Prop :=
  let DF := Real.sqrt (DE^2 + EF^2)
  let x := (5 * (60 / (5 + 12 * Real.sqrt 3))) / 5
  let y := (60 * (5 - 12 * Real.sqrt 3)) / -407
  let FQ := 2 * y
  FQ = (600 - 1440 * Real.sqrt 3) / 407

theorem shorter_trisector_length :
  triangle_shorter_trisector 5 12 by decide :=
sorry

end shorter_trisector_length_l681_681623


namespace pencils_in_fifth_box_l681_681830

theorem pencils_in_fifth_box (a b c d : ℕ) (p1 p2 p3 p4 : ℕ) 
  (h1 : p1 = 78) (h2 : p2 = 87) (h3 : p3 = 96) (h4 : p4 = 105) 
  (step : ∀ n, p2 - p1 = 9 ∧ p3 - p2 = 9 ∧ p4 - p3 = 9) :
  ∃ p5, p5 = p4 + 9 ∧ p5 = 114 := 
by
  -- The proof goes here
  sorry

end pencils_in_fifth_box_l681_681830


namespace part1_profit_in_april_part2_price_reduction_l681_681042

-- Given conditions
def cost_per_bag : ℕ := 16
def original_price_per_bag : ℕ := 30
def reduction_amount : ℕ := 5
def increase_in_sales_rate : ℕ := 20
def original_sales_volume : ℕ := 200
def target_profit : ℕ := 2860

-- Part 1: When the price per bag of noodles is reduced by 5 yuan
def profit_in_april_when_reduced_by_5 (cost_per_bag original_price_per_bag reduction_amount increase_in_sales_rate original_sales_volume : ℕ) : ℕ := 
  let new_price := original_price_per_bag - reduction_amount
  let new_sales_volume := original_sales_volume + (increase_in_sales_rate * reduction_amount)
  let profit_per_bag := new_price - cost_per_bag
  profit_per_bag * new_sales_volume

theorem part1_profit_in_april :
  profit_in_april_when_reduced_by_5 16 30 5 20 200 = 2700 :=
sorry

-- Part 2: Determine the price reduction for a specific target profit
def price_reduction_for_profit (cost_per_bag original_price_per_bag increase_in_sales_rate original_sales_volume target_profit : ℕ) : ℕ :=
  let x := (target_profit - (original_sales_volume * (original_price_per_bag - cost_per_bag))) / (increase_in_sales_rate * (original_price_per_bag - cost_per_bag) - increase_in_sales_rate - original_price_per_bag)
  x

theorem part2_price_reduction :
  price_reduction_for_profit 16 30 20 200 2860 = 3 :=
sorry

end part1_profit_in_april_part2_price_reduction_l681_681042


namespace sum_a_b_product_l681_681624

noncomputable theory

-- Definition of an arithmetic sequence function
def arith_seq (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

-- Definition of sequence {a_n} with given constraints
def a_seq (n : ℕ) : ℝ :=
  let a1 := 3 / 2
  let d := 1 / 2
  arith_seq a1 d n

-- Definition of sequence {b_n} with given sum conditions
def b_seq (n : ℕ) : ℝ :=
  2^n

-- Definition of the sum of the products of sequences {a_n} and {b_n}
def sum_a_times_b (n : ℕ) : ℝ :=
  (n + 1) * 2^n - 1

-- Theorem statement combining findings
theorem sum_a_b_product (n : ℕ) : 
  let a := arith_seq (3/2) (1/2) n
  let b := 2^n
  (a * b = (n / 2 + 1) * 2^n) ∧ (sum_a_times_b n = (n + 1) * 2^n - 1) := by
  { sorry }

end sum_a_b_product_l681_681624


namespace arithmetic_series_sum_l681_681122

theorem arithmetic_series_sum : 
  let a := 2 in 
  let d := 2 in 
  let n := 10 in 
  let l := 20 in 
  (a + l) * n / 2 = 110 := 
by
  sorry

end arithmetic_series_sum_l681_681122


namespace system_inconsistent_l681_681945

theorem system_inconsistent :
  ¬(∃ (x1 x2 x3 x4 : ℝ), 
    (5 * x1 + 12 * x2 + 19 * x3 + 25 * x4 = 25) ∧
    (10 * x1 + 22 * x2 + 16 * x3 + 39 * x4 = 25) ∧
    (5 * x1 + 12 * x2 + 9 * x3 + 25 * x4 = 30) ∧
    (20 * x1 + 46 * x2 + 34 * x3 + 89 * x4 = 70)) := 
by
  sorry

end system_inconsistent_l681_681945


namespace difference_in_area_l681_681939

-- Define the areas and the side lengths based on the given conditions
def area1 : ℝ := 10000  -- Area of the first field in square meters
def side1 : ℝ := real.sqrt area1  -- Side length of the first field in meters
def side2 : ℝ := 1.01 * side1  -- Side length of the second field (1% larger)
def area2 : ℝ := side2^2  -- Area of the second field

-- Define the theorem to prove the difference in areas is 201 square meters
theorem difference_in_area : (area2 - area1) = 201 := by
  sorry

end difference_in_area_l681_681939


namespace average_salary_of_technicians_l681_681276

theorem average_salary_of_technicians
  (total_workers : ℕ)
  (avg_salary_all_workers : ℕ)
  (total_technicians : ℕ)
  (avg_salary_non_technicians : ℕ)
  (h1 : total_workers = 18)
  (h2 : avg_salary_all_workers = 8000)
  (h3 : total_technicians = 6)
  (h4 : avg_salary_non_technicians = 6000) :
  (72000 / total_technicians) = 12000 := 
  sorry

end average_salary_of_technicians_l681_681276


namespace hyperbola_asymptotes_and_eccentricity_l681_681665

theorem hyperbola_asymptotes_and_eccentricity 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : asin (1 / 2) = π / 6) 
  (hypeq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) : 
  (∀ x, y = (± (sqrt 3) / 3) * x) ∧ 
  (sqrt (1 + (b^2 / a^2)) = 2 * sqrt 3 / 3) :=
sorry

end hyperbola_asymptotes_and_eccentricity_l681_681665


namespace rational_solutions_equation_l681_681874

theorem rational_solutions_equation :
  ∃ x : ℚ, (|x - 19| + |x - 93| = 74 ∧ x ∈ {y : ℚ | 19 ≤ y ∨ 19 < y ∧ y < 93 ∨ y ≥ 93}) :=
sorry

end rational_solutions_equation_l681_681874


namespace smallest_possible_students_group_l681_681817

theorem smallest_possible_students_group 
  (students : ℕ) :
  (∀ n, 2 ≤ n ∧ n ≤ 15 → ∃ k, students = k * n) ∧
  ¬∃ k, students = k * 10 ∧ ¬∃ k, students = k * 25 ∧ ¬∃ k, students = k * 50 ∧
  ∀ m n, 1 ≤ m ∧ m ≤ 15 ∧ 1 ≤ n ∧ n ≤ 15 ∧ (students ≠ m * n) → (m = n ∨ m ≠ n)
  → students = 120 := sorry

end smallest_possible_students_group_l681_681817


namespace log_function_domain_l681_681578

noncomputable def domain_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : Set ℝ :=
  { x : ℝ | x < a }

theorem log_function_domain (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x, x ∈ domain_of_log_function a h1 h2 ↔ x < a :=
by
  sorry

end log_function_domain_l681_681578


namespace set_intersection_l681_681213

def A (x : ℝ) : Prop := -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 3
def B (x : ℝ) : Prop := (x + 1) / x ≤ 0
def C_x_B (x : ℝ) : Prop := x < -1 ∨ x ≥ 0

theorem set_intersection :
  {x : ℝ | A x} ∩ {x : ℝ | C_x_B x} = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
sorry

end set_intersection_l681_681213


namespace binom_18_6_l681_681512

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_18_6 : binomial 18 6 = 18564 := 
by
  sorry

end binom_18_6_l681_681512


namespace domain_of_f_smallest_positive_period_inc_dec_intervals_extremum_values_l681_681663

noncomputable def f (x : ℝ) : ℝ := 4 * tan x * sin (π / 2 - x) * cos (x - π / 3) - sqrt 3

theorem domain_of_f : 
  ∀ x, x ∉ {x | ∃ k : ℤ, x = k * π + π / 2} → ∃y : ℝ, f y = f x :=
sorry

theorem smallest_positive_period :
  ∀ x, f (x + π) = f x :=
sorry

theorem inc_dec_intervals :
  (∀ x ∈ Icc (-π / 12) (π / 4), ∀ y ∈ Icc x (π / 4), f x ≤ f y) ∧
  (∀ x ∈ Icc (-π / 4) (-π / 12), ∀ y ∈ Icc x (-π / 12), f x ≥ f y) :=
sorry

theorem extremum_values :
  (∀ x ∈ Icc (-π / 4) (π / 4), f x ≥ -2) ∧ 
  f (-π / 12) = -2 ∧ 
  (∀ x ∈ Icc (-π / 4) (π / 4), f x ≤ 1) ∧ 
  f (π / 4) = 1 :=
sorry

end domain_of_f_smallest_positive_period_inc_dec_intervals_extremum_values_l681_681663


namespace average_rainfall_l681_681733

theorem average_rainfall (r d h : ℕ) (rainfall_eq : r = 450) (days_eq : d = 30) (hours_eq : h = 24) :
  r / (d * h) = 25 / 16 := 
  by 
    -- Insert appropriate proof here
    sorry

end average_rainfall_l681_681733


namespace f_2018_eq_l681_681808

open Finset

def P (n : ℕ) : Finset ℕ := range (n + 1) \ {0}

def valid_subset (A Pn : Finset ℕ) : Prop :=
  ∀ x ∈ A, (2 * x) ∉ A ∧ ∀ y ∈ (Pn \ A), (2 * y) ∉ (Pn \ A)

noncomputable def f (n : ℕ) : ℕ :=
  (P n).powerset.count (λ A, valid_subset A (P n))

theorem f_2018_eq : f 2018 = 2 ^ 1009 := 
by
  sorry

end f_2018_eq_l681_681808


namespace line_circle_intersects_l681_681870

def line (x : ℝ) : ℝ := 2 * x - 1

def circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * y = 0

theorem line_circle_intersects : 
  (∃ x y : ℝ, circle x y ∧ y = line x) :=
sorry

end line_circle_intersects_l681_681870


namespace math_problem_l681_681305

variable (a b c d : ℝ)

-- The initial condition provided in the problem
def given_condition : Prop := (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7

-- The statement that needs to be proven
theorem math_problem 
  (h : given_condition a b c d) : 
  (a - c) * (b - d) / ((a - b) * (c - d)) = -1 := 
by 
  sorry

end math_problem_l681_681305


namespace algae_cells_count_10_days_l681_681991

-- Define the initial condition where the pond starts with one algae cell.
def initial_algae_cells : ℕ := 1

-- Define the daily splitting of each cell into 3 new cells.
def daily_split (cells : ℕ) : ℕ := cells * 3

-- Define the function to compute the number of algae cells after n days.
def algae_cells_after_days (n : ℕ) : ℕ :=
  initial_algae_cells * (3 ^ n)

-- State the theorem to be proved.
theorem algae_cells_count_10_days : algae_cells_after_days 10 = 59049 :=
by {
  sorry
}

end algae_cells_count_10_days_l681_681991


namespace basis_of_R3_l681_681800

def e1 : ℝ × ℝ × ℝ := (1, 0, 0)
def e2 : ℝ × ℝ × ℝ := (0, 1, 0)
def e3 : ℝ × ℝ × ℝ := (0, 0, 1)

theorem basis_of_R3 :
  ∀ (u : ℝ × ℝ × ℝ), ∃ (α β γ : ℝ), u = α • e1 + β • e2 + γ • e3 ∧ 
  (∀ (a b c : ℝ), a • e1 + b • e2 + c • e3 = (0, 0, 0) → a = 0 ∧ b = 0 ∧ c = 0) :=
by
  sorry

end basis_of_R3_l681_681800


namespace positive_factors_23232_l681_681250

theorem positive_factors_23232:
    let n := 23232 in
    ∃ a b c : ℕ, a = 6 ∧ b = 1 ∧ c = 2 ∧ (n = 2^a * 3^b * 11^c) →
    ∏ (i : ℕ) in {a, b, c}.map(+1), i = 42 :=
by 
  sorry

end positive_factors_23232_l681_681250


namespace quadratic_has_one_solution_l681_681590

theorem quadratic_has_one_solution (n : ℤ) : 
  (n ^ 2 - 64 = 0) ↔ (n = 8 ∨ n = -8) := 
by
  sorry

end quadratic_has_one_solution_l681_681590


namespace greatest_3digit_base8_divisible_by_7_l681_681902

def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : ℕ), n = 0b777 ∧ (base8_to_base10 0b777) % 7 = 0 ∧ ∀ m < 0o777, m % 7 = 0 → base8_to_base10 m < base8_to_base10 0b777 :=
by
  sorry

end greatest_3digit_base8_divisible_by_7_l681_681902


namespace shorter_piece_length_l681_681425

theorem shorter_piece_length (L : ℝ) (k : ℝ) (shorter_piece : ℝ) : 
  L = 28 ∧ k = 2.00001 / 5 ∧ L = shorter_piece + k * shorter_piece → 
  shorter_piece = 20 :=
by
  sorry

end shorter_piece_length_l681_681425


namespace part1_option1_payment_part1_option2_payment_part2_cost_effective_part3_more_cost_effective_l681_681952

def cost_option1 (x : ℕ) : ℕ :=
  20 * x + 1200

def cost_option2 (x : ℕ) : ℕ :=
  18 * x + 1440

theorem part1_option1_payment (x : ℕ) (h : x > 20) : cost_option1 x = 20 * x + 1200 :=
  by sorry

theorem part1_option2_payment (x : ℕ) (h : x > 20) : cost_option2 x = 18 * x + 1440 :=
  by sorry

theorem part2_cost_effective (x : ℕ) (h : x = 30) : cost_option1 x < cost_option2 x :=
  by sorry

theorem part3_more_cost_effective (x : ℕ) (h : x = 30) : 20 * 80 + 20 * 10 * 9 / 10 = 1780 :=
  by sorry

end part1_option1_payment_part1_option2_payment_part2_cost_effective_part3_more_cost_effective_l681_681952


namespace MISSISSIPPI_arrangements_l681_681172

def count_letters (s : String) : List (Char × Nat) :=
  s.toList.foldl (λ acc c, match acc.find? (λ p, p.1 = c) with
  | some (c, n) => acc.filter (λ p, p.1 ≠ c) ++ [(c, n + 1)]
  | none => acc ++ [(c, 1)]
  end) []

theorem MISSISSIPPI_arrangements :
  let letters := "MISSISSIPPI"
  let counts := count_letters letters
  let total := counts.foldl (λ acc (_, n), acc + n) 0
  let fact := Nat.factorial
  total = 11 ∧
  counts = [('M', 1), ('I', 4), ('S', 4), ('P', 2)] →
  fact 11 / (fact 4 * fact 4 * fact 2 * fact 1) = 34650 :=
by
  intros letters counts total fact hTotal hCounts
  sorry

end MISSISSIPPI_arrangements_l681_681172


namespace Pk_to_Pk_plus_one_l681_681238

theorem Pk_to_Pk_plus_one 
  (a : ℝ) (k : ℕ) 
  (h_a_pos : a > 0)
  (h_a_ne_one : a ≠ 1) 
  (h_k_pos : k ≥ 1) :
  (1 + ∑ i in finset.range (k+1), a^(2*(i+1))) / (∑ i in finset.range (k+1), a^(2*i+1)) 
  > (k + 2) / (k + 1) :=
sorry

end Pk_to_Pk_plus_one_l681_681238


namespace bookstore_earnings_difference_l681_681993

def base_price_TOP := 8.0
def base_price_ABC := 23.0
def discount_TOP := 0.10
def discount_ABC := 0.05
def sales_tax := 0.07
def num_TOP_sold := 13
def num_ABC_sold := 4

def discounted_price (base_price discount : Float) : Float :=
  base_price * (1.0 - discount)

def final_price (discounted_price tax : Float) : Float :=
  discounted_price * (1.0 + tax)

def total_earnings (final_price : Float) (quantity : Nat) : Float :=
  final_price * (quantity.toFloat)

theorem bookstore_earnings_difference :
  let discounted_price_TOP := discounted_price base_price_TOP discount_TOP
  let discounted_price_ABC := discounted_price base_price_ABC discount_ABC
  let final_price_TOP := final_price discounted_price_TOP sales_tax
  let final_price_ABC := final_price discounted_price_ABC sales_tax
  let total_earnings_TOP := total_earnings final_price_TOP num_TOP_sold
  let total_earnings_ABC := total_earnings final_price_ABC num_ABC_sold
  total_earnings_TOP - total_earnings_ABC = 6.634 :=
by
  sorry

end bookstore_earnings_difference_l681_681993


namespace find_B_l681_681096

variable {A B C D : ℕ}

-- Condition 1: The first dig site (A) was dated 352 years more recent than the second dig site (B)
axiom h1 : A = B + 352

-- Condition 2: The third dig site (C) was dated 3700 years older than the first dig site (A)
axiom h2 : C = A - 3700

-- Condition 3: The fourth dig site (D) was twice as old as the third dig site (C)
axiom h3 : D = 2 * C

-- Condition 4: The age difference between the second dig site (B) and the third dig site (C) was four times the difference between the fourth dig site (D) and the first dig site (A)
axiom h4 : B - C = 4 * (D - A)

-- Condition 5: The fourth dig site is dated 8400 BC.
axiom h5 : D = 8400

-- Prove the question
theorem find_B : B = 7548 :=
by
  sorry

end find_B_l681_681096


namespace price_increase_decrease_l681_681085

theorem price_increase_decrease (P : ℝ) (x : ℝ) (h : P > 0) :
  (P * (1 + x / 100) * (1 - x / 100) = 0.64 * P) → (x = 60) :=
by
  sorry

end price_increase_decrease_l681_681085


namespace minions_mistake_score_l681_681036

theorem minions_mistake_score :
  (minions_left_phone_on_untrusted_website ∧
   downloaded_file_from_untrusted_source ∧
   guidelines_by_cellular_operators ∧
   avoid_sharing_personal_info ∧
   unverified_files_may_be_harmful ∧
   double_extensions_signify_malicious_software) →
  score = 21 :=
by
  -- Here we would provide the proof steps which we skip with sorry
  sorry

end minions_mistake_score_l681_681036


namespace john_speed_is_30_kmph_l681_681771

/-- Problem Statement: Given the distance from John's house to his office is 24 km, 
and that at a speed of 40 kmph he reaches 8 minutes earlier, and at a certain speed he reaches 
4 minutes late, prove that the speed at which he reaches 4 minutes late is 30 kmph. -/
theorem john_speed_is_30_kmph (distance : ℝ) (speed_early : ℝ) (time_diff_early : ℝ) (time_diff_late : ℝ) :
  distance = 24 → speed_early = 40 → time_diff_early = 8/60 → time_diff_late = 4/60 →
  let usual_time := distance / speed_early + time_diff_early in
  let late_time := usual_time + time_diff_late in
  ∃ (v : ℝ), distance / v = late_time ∧ v = 30 :=
begin
  intros,
  simp only,
  sorry
end

end john_speed_is_30_kmph_l681_681771


namespace period_cos_3x_over_4_l681_681174

theorem period_cos_3x_over_4 :
  ∃ (T : ℝ), (∀ x : ℝ, cos ((3 / 4 : ℚ) * (x + T : ℚ)) = cos ((3 / 4 : ℚ) * x)) ∧ T = 8 * π / 3 :=
sorry

end period_cos_3x_over_4_l681_681174


namespace find_a_100_l681_681000

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  (a 1 = 3) ∧ (a 2 = 7) ∧ (∀ n, a (n + 1) = a n - a (n - 1))

theorem find_a_100 (a : ℕ → ℤ) (h : sequence a) : a 100 = -3 :=
by
  sorry

end find_a_100_l681_681000


namespace mixture_percent_chemical_a_l681_681838

-- Defining the conditions
def solution_x : ℝ := 0.4
def solution_y : ℝ := 0.5
def percent_x_in_mixture : ℝ := 0.3
def percent_y_in_mixture : ℝ := 1.0 - percent_x_in_mixture

-- The goal is to prove that the mixture is 47% chemical a
theorem mixture_percent_chemical_a : (solution_x * percent_x_in_mixture + solution_y * percent_y_in_mixture) * 100 = 47 :=
by
  -- Calculation here
  sorry

end mixture_percent_chemical_a_l681_681838


namespace find_m_l681_681254

-- Given assumptions
variables {a b m : ℝ}
hypothesis h1 : 2^a = m
hypothesis h2 : 5^b = m
hypothesis h3 : (1 / a) + (1 / b) = 2

-- The main theorem
theorem find_m : m = 10 :=
by
  -- proof will go here
  sorry

end find_m_l681_681254


namespace slope_and_intercept_of_given_function_l681_681877

-- Defining the form of a linear function
def linear_function (m b : ℝ) (x : ℝ) : ℝ := m * x + b

-- The given linear function
def given_function (x : ℝ) : ℝ := 3 * x + 2

-- Stating the problem as a theorem
theorem slope_and_intercept_of_given_function :
  (∀ x : ℝ, given_function x = linear_function 3 2 x) :=
by
  intro x
  sorry

end slope_and_intercept_of_given_function_l681_681877


namespace train_length_l681_681467

noncomputable section

-- Define the variables involved in the problem.
def train_length_cross_signal (V : ℝ) : ℝ := V * 18
def train_speed_cross_platform (L : ℝ) (platform_length : ℝ) : ℝ := (L + platform_length) / 40

-- Define the main theorem to prove the length of the train.
theorem train_length (V L : ℝ) (platform_length : ℝ) (h1 : L = V * 18)
(h2 : L + platform_length = V * 40) (h3 : platform_length = 366.67) :
L = 300 := 
by
  sorry

end train_length_l681_681467


namespace range_of_m_l681_681264

theorem range_of_m (m : ℝ) :
  (∃ ρ θ : ℝ, m * ρ * (Real.cos θ)^2 + 3 * ρ * (Real.sin θ)^2 - 6 * (Real.cos θ) = 0 ∧
    (∃ ρ₀ θ₀ : ℝ, ∀ ρ θ, m * ρ * (Real.cos θ)^2 + 3 * ρ * (Real.sin θ)^2 - 6 * (Real.cos θ) = 
      m * ρ₀ * (Real.cos θ₀)^2 + 3 * ρ₀ * (Real.sin θ₀)^2 - 6 * (Real.cos θ₀))) →
  m > 0 ∧ m ≠ 3 := sorry

end range_of_m_l681_681264


namespace number_of_customers_l681_681981

theorem number_of_customers 
    (boxes_opened : ℕ) 
    (samples_per_box : ℕ) 
    (samples_left_over : ℕ) 
    (samples_limit_per_person : ℕ)
    (h1 : boxes_opened = 12)
    (h2 : samples_per_box = 20)
    (h3 : samples_left_over = 5)
    (h4 : samples_limit_per_person = 1) : 
    ∃ customers : ℕ, customers = (boxes_opened * samples_per_box) - samples_left_over ∧ customers = 235 :=
by {
  sorry
}

end number_of_customers_l681_681981


namespace problem_f_neg_10_l681_681664

def f : ℝ → ℝ
| x => if x > 0 then real.log x / real.log 2 else f (x + 3)

theorem problem_f_neg_10 : f (-10) = 1 :=
by
  sorry

end problem_f_neg_10_l681_681664


namespace binom_18_6_eq_18564_l681_681503

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binom_18_6_eq_18564 : binomial 18 6 = 18564 := by
  sorry

end binom_18_6_eq_18564_l681_681503


namespace max_mx_plus_ny_l681_681629

theorem max_mx_plus_ny 
  (m n x y : ℝ) 
  (h1 : m^2 + n^2 = 6) 
  (h2 : x^2 + y^2 = 24) : 
  mx + ny ≤ 12 :=
sorry

end max_mx_plus_ny_l681_681629


namespace binom_18_6_l681_681514

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_18_6 : binomial 18 6 = 18564 := 
by
  sorry

end binom_18_6_l681_681514


namespace minimum_black_edges_l681_681541

theorem minimum_black_edges (cube : ℕ) (edges_colored : ℕ → ℕ) 
  (colored : ∀ e, edges_colored e ∈ {0, 1})  -- 0 denotes red, 1 denotes black edge
  (faces_opposite_with_black_edges : ∀ f1 f2, f1 ≠ f2 ∧ opposite_faces f1 f2 → (edges_colored f1 ≥ 2 ∧ edges_colored f2 ≥ 2)) : 
  ∃ n, n = 4 ∧ (∀ m, m < 4 → ¬ (faces_opposite_with_black_edges cube edges_colored m)) :=
sorry

end minimum_black_edges_l681_681541


namespace coefficient_x3_term_in_expansion_l681_681576

noncomputable def p (x : ℕ) := 3 * x^3 + 2 * x^2 + x + 1
noncomputable def q (x : ℕ) := 4 * x^3 + 3 * x^2 + 2 * x + 5

theorem coefficient_x3_term_in_expansion :
  let r := p * q in
  r.coeff 3 = 22 :=
by
  sorry

end coefficient_x3_term_in_expansion_l681_681576


namespace cost_of_5_pound_bag_l681_681956

theorem cost_of_5_pound_bag
  (p : ℚ) -- price per 5-pound bag
  (p10 : ℚ) (p10 = 20.43) -- price per 10-pound bag
  (p25 : ℚ) (p25 = 32.20) -- price per 25-pound bag
  (total_weight : ℕ) (65 ≤ total_weight) (total_weight ≤ 80) -- total weight range
  (total_cost : ℚ) (total_cost = 98.68) -- minimum total cost
  (h : ∃ (n5 n10 n25 : ℕ), n5 * 5 + n10 * 10 + n25 * 25 = total_weight ∧ n5 * p + n10 * p10 + n25 * p25 = total_cost) :
  p = 2.08 :=
sorry

end cost_of_5_pound_bag_l681_681956


namespace min_sum_intercepts_of_line_l681_681260

theorem min_sum_intercepts_of_line (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a + 2 / b = 1) : a + b = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_sum_intercepts_of_line_l681_681260


namespace number_of_triangles_with_perimeter_11_l681_681686

theorem number_of_triangles_with_perimeter_11 : (∃ triangles : List (ℕ × ℕ × ℕ), 
  (∀ t ∈ triangles, let (a, b, c) := t in 
    a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ a + c > b) 
  ∧ triangles.length = 10) := 
sorry

end number_of_triangles_with_perimeter_11_l681_681686


namespace intersect_circle_line_constructible_l681_681891

-- Point and Circle definitions
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Definitions for the problem
variable (A B : Point)
variable (S : Circle)

-- The theorem statement
theorem intersect_circle_line_constructible (A B : Point) (S : Circle) :
  ∃ P : Point, (∃ Q₁ : Point, Q₁ ∈ S ∧ line_through A B = line_through A Q₁ ∧ 
                 ∃ Q₂ : Point, Q₂ ∈ S ∧ line_through A B = line_through A Q₂) :=
sorry

end intersect_circle_line_constructible_l681_681891


namespace count_divisible_by_seven_between_100_and_400_l681_681704

theorem count_divisible_by_seven_between_100_and_400 : 
    (∃ n, ∀ x, 100 ≤ x ∧ x ≤ 400 ∧ x % 7 = 0 → x ∈ List.range (105) ((399 - 105) // 7 + 1) ∧ n = ((399 - 105) // 7 + 1)) :=
by
  sorry

end count_divisible_by_seven_between_100_and_400_l681_681704


namespace sum_divisors_mod_2_even_l681_681571

open Nat

-- Definition: The number of positive divisors of \( n \)
def d (n : Nat) : Nat := (List.filter (λ m, n % m = 0) (List.range n.succ)).length

-- Claim: The sum of d(n) from 1 to 1989 is even
theorem sum_divisors_mod_2_even : (finset.range 1990).sum d % 2 = 0 := by
  sorry

end sum_divisors_mod_2_even_l681_681571


namespace find_f_f_2_l681_681857

def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2 ^ x - 2 else 2 * Real.sin (Real.pi / 12 * x) - 1

theorem find_f_f_2 : f (f 2) = -1 := by
  sorry

end find_f_f_2_l681_681857


namespace decreasing_interval_range_l681_681647

theorem decreasing_interval_range (a : ℝ) :
  (∀ x y ∈ Ioo 0 1, x < y → 2^(x * (x-a)) > 2^(y * (y-a))) ↔ a ≥ 2 :=
by
  sorry

end decreasing_interval_range_l681_681647


namespace nell_more_ace_cards_than_baseball_l681_681326

-- Definitions based on conditions
def original_baseball_cards : ℕ := 239
def original_ace_cards : ℕ := 38
def current_ace_cards : ℕ := 376
def current_baseball_cards : ℕ := 111

-- The statement we need to prove
theorem nell_more_ace_cards_than_baseball :
  current_ace_cards - current_baseball_cards = 265 :=
by
  -- Add the proof here
  sorry

end nell_more_ace_cards_than_baseball_l681_681326


namespace problem_statement_l681_681801

theorem problem_statement (x y z : ℝ) (hx : x + y + z = 2) (hxy : xy + xz + yz = -9) (hxyz : xyz = 1) :
  (yz / x) + (xz / y) + (xy / z) = 77 := sorry

end problem_statement_l681_681801


namespace greatest_3_digit_base_8_divisible_by_7_l681_681916

open Nat

def is_3_digit_base_8 (n : ℕ) : Prop := n < 8^3

def is_divisible_by_7 (n : ℕ) : Prop := 7 ∣ n

theorem greatest_3_digit_base_8_divisible_by_7 :
  ∃ x : ℕ, is_3_digit_base_8 x ∧ is_divisible_by_7 x ∧ x = 7 * (8 * (8 * 7 + 7) + 7) :=
by
  sorry

end greatest_3_digit_base_8_divisible_by_7_l681_681916


namespace cyclic_quadrilateral_l681_681819

theorem cyclic_quadrilateral 
  (A B C D K N L M : ℝ) -- Representing the points' coordinates on a single real number line for simplicity
  (h_square : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) -- Non-degenerate square with distinct points 
  (h_on_sides : K ∈ [A, B] ∧ N ∈ [A, D]) -- Points K and N on sides AB and AD respectively
  (h_product : (K - A) * (N - A) = 2 * ((B - K) * (D - N))) -- The given product condition
  (h_intersections : L ∈ [C, K] ∧ M ∈ [C, N]) -- Points L and M are intersections of CK and CN with BD
  : ∃ O : ℝ, ∀ P ∈ {K, L, M, N, A}, -- There exists a circle with center O containing all points
    dist O P = dist O A := -- Distances to the center O are equal
sorry

end cyclic_quadrilateral_l681_681819


namespace inequality_proof_equality_case_l681_681332

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 / (b^3 * c) - a / (b^2) ≥ c / b - (c^2) / a) :=
sorry

theorem equality_case (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 / (b^3 * c) - a / (b^2) = c / b - c^2 / a) ↔ (a = b * c) :=
sorry

end inequality_proof_equality_case_l681_681332


namespace floor_47_l681_681555

theorem floor_47 : Int.floor 4.7 = 4 :=
by
  sorry

end floor_47_l681_681555


namespace length_of_MN_l681_681822

-- Define the lengths and trapezoid properties
variables (a b: ℝ)

-- Define the problem statement
theorem length_of_MN (a b: ℝ) :
  ∃ (MN: ℝ), ∀ (M N: ℝ) (is_trapezoid : True),
  (MN = 3 * a * b / (a + 2 * b)) :=
sorry

end length_of_MN_l681_681822


namespace expression_value_l681_681032

theorem expression_value (x y z : ℤ) (h1: x = 2) (h2: y = -3) (h3: z = 1) :
  x^2 + y^2 - 2*z^2 + 3*x*y = -7 := 
by
  sorry

end expression_value_l681_681032


namespace z_conjugate_in_first_quadrant_l681_681227

-- Define the given complex number z
def z : ℂ := 1 / (1 + complex.I)

-- Define the conjugate of z
def z_conjugate : ℂ := complex.conj z

-- Define the point corresponding to the conjugate of z
def point_conjugate : ℝ × ℝ := (z_conjugate.re, z_conjugate.im)

-- Prove that the point lies in the first quadrant
theorem z_conjugate_in_first_quadrant : 
  (point_conjugate.1 > 0) ∧ (point_conjugate.2 > 0) :=
sorry

end z_conjugate_in_first_quadrant_l681_681227


namespace quad_with_midpoint_rhombus_eq_diagonals_l681_681421

theorem quad_with_midpoint_rhombus_eq_diagonals (Q : Type) [quadrilateral Q]
  (h1 : connects_midpoints_rhombus Q) : equal_diagonals Q :=
sorry

end quad_with_midpoint_rhombus_eq_diagonals_l681_681421


namespace half_abs_difference_squares_l681_681021

theorem half_abs_difference_squares :
  (1 / 2) * |(15^2 - 13^2)| = 28 :=
by
  sorry

end half_abs_difference_squares_l681_681021


namespace time_after_12345_seconds_l681_681297

theorem time_after_12345_seconds 
  (h0 : (10, 0, 0) = (10 * 3600 + 0 * 60 + 0))
  (duration : 12345)
  : let final_time := 10 * 3600 + 0 * 60 + 0 + duration in 
    (final_time / 3600, (final_time % 3600) / 60, (final_time % 3600) % 60) = (13, 25, 45) :=
by
  sorry

end time_after_12345_seconds_l681_681297


namespace arithmetic_series_sum_l681_681125

theorem arithmetic_series_sum : 
  let a := 2 in 
  let d := 2 in 
  let n := 10 in 
  let l := 20 in 
  (a + l) * n / 2 = 110 := 
by
  sorry

end arithmetic_series_sum_l681_681125


namespace area_ratio_l681_681756

-- Given the triangle with named sides and altitudes.
variables (A B C D E : Type)
variables [IsTriangle A B C] (AD: Altitude A D) (CE: Altitude C E)

-- Given the side lengths.
variables (AB AC BC : ℝ)
variable (hAB : AB = 6)
variable (hAC : AC = 5)
variable (hBC : BC = 7)

theorem area_ratio (h : ∀ {A B C AD CE}, IsTriangle A B C → Altitude A D → Altitude C E → AB = 6 → AC = 5 → BC = 7 → 
  area_ratio (triangle_area A B C) (triangle_area A E D) = 49 / 5) :
  ∀ {A B C AD CE}, Altitude A D → Altitude C E → IsTriangle A B C →
  (area (triangle A B C) / area (triangle A E D)) = 49 / 5 :=
begin
  intro h,
  assumption,
  sorry
end

end area_ratio_l681_681756


namespace gcd_294_84_l681_681013

-- Define the numbers for the GCD calculation
def a : ℕ := 294
def b : ℕ := 84

-- Define the greatest common divisor function using Euclidean algorithm
def gcd_euclidean : ℕ → ℕ → ℕ
| x, 0 => x
| x, y => gcd_euclidean y (x % y)

-- Theorem stating that the GCD of 294 and 84 is 42
theorem gcd_294_84 : gcd_euclidean a b = 42 :=
by
  -- Proof is omitted
  sorry

end gcd_294_84_l681_681013


namespace john_has_22_quarters_l681_681298

-- Definitions based on conditions
def number_of_quarters (Q : ℕ) : ℕ := Q
def number_of_dimes (Q : ℕ) : ℕ := Q + 3
def number_of_nickels (Q : ℕ) : ℕ := Q - 6

-- Total number of coins condition
def total_number_of_coins (Q : ℕ) : Prop := 
  (number_of_quarters Q) + (number_of_dimes Q) + (number_of_nickels Q) = 63

-- Goal: Proving the number of quarters is 22
theorem john_has_22_quarters : ∃ Q : ℕ, total_number_of_coins Q ∧ Q = 22 :=
by
  -- Proof skipped 
  sorry

end john_has_22_quarters_l681_681298


namespace sum_arithmetic_seq_l681_681134

theorem sum_arithmetic_seq (a d n : ℕ) :
  a = 2 → d = 2 → a + (n - 1) * d = 20 → (n / 2) * (a + (a + (n - 1) * d)) = 110 :=
by sorry

end sum_arithmetic_seq_l681_681134


namespace decreasing_on_interval_l681_681661

noncomputable def f (a x : ℝ) : ℝ := 2^(x * (x - a))

theorem decreasing_on_interval (a : ℝ) : (a ≥ 2) ↔ ∀ x ∈ Set.Ioo 0 1, (deriv (λ x, 2^(x * (x - a)))) x ≤ 0 :=
sorry

end decreasing_on_interval_l681_681661


namespace guard_team_size_l681_681833

theorem guard_team_size (b n s : ℕ) (h_total : b * s * n = 1001) (h_condition : s < n ∧ n < b) : s = 7 := 
by
  sorry

end guard_team_size_l681_681833


namespace simon_change_l681_681834

noncomputable def calculate_change 
  (pansies_count : ℕ) (pansies_price : ℚ) (pansies_discount : ℚ) 
  (hydrangea_count : ℕ) (hydrangea_price : ℚ) (hydrangea_discount : ℚ) 
  (petunias_count : ℕ) (petunias_price : ℚ) (petunias_discount : ℚ) 
  (lilies_count : ℕ) (lilies_price : ℚ) (lilies_discount : ℚ) 
  (orchids_count : ℕ) (orchids_price : ℚ) (orchids_discount : ℚ) 
  (sales_tax : ℚ) (payment : ℚ) : ℚ :=
  let pansies_total := (pansies_count * pansies_price) * (1 - pansies_discount)
  let hydrangea_total := (hydrangea_count * hydrangea_price) * (1 - hydrangea_discount)
  let petunias_total := (petunias_count * petunias_price) * (1 - petunias_discount)
  let lilies_total := (lilies_count * lilies_price) * (1 - lilies_discount)
  let orchids_total := (orchids_count * orchids_price) * (1 - orchids_discount)
  let total_price := pansies_total + hydrangea_total + petunias_total + lilies_total + orchids_total
  let final_price := total_price * (1 + sales_tax)
  payment - final_price

theorem simon_change : calculate_change
  5 2.50 0.10
  1 12.50 0.15
  5 1.00 0.20
  3 5.00 0.12
  2 7.50 0.08
  0.06 100 = 43.95 := by sorry

end simon_change_l681_681834


namespace functions_equiv_l681_681038

noncomputable def f_D (x : ℝ) : ℝ := Real.log (Real.sqrt x)
noncomputable def g_D (x : ℝ) : ℝ := (1/2) * Real.log x

theorem functions_equiv : ∀ x : ℝ, x > 0 → f_D x = g_D x := by
  intro x h
  sorry

end functions_equiv_l681_681038


namespace count_of_n_l681_681251

theorem count_of_n :
  let valid_n (n : ℕ) := ∃ (r : ℕ), n = 2 * r + 1 ∧ r * (r + 1) % 5 = 0
  in (Finset.filter valid_n (Finset.range 150)).card = 14 :=
by
  let valid_n (n : ℕ) := ∃ (r : ℕ), n = 2 * r + 1 ∧ r * (r + 1) % 5 = 0
  exact sorry

end count_of_n_l681_681251


namespace total_bill_is_66_l681_681041

def cost_of_orange := 0.50
def cost_of_apple := 1.00
def cost_of_watermelon := 4 * cost_of_apple
def number_of_each_fruit := 36 / 3

def total_orange_cost := number_of_each_fruit * cost_of_orange
def total_apple_cost := number_of_each_fruit * cost_of_apple
def total_watermelon_cost := number_of_each_fruit * cost_of_watermelon

def total_cost := total_orange_cost + total_apple_cost + total_watermelon_cost

theorem total_bill_is_66 : total_cost = 66 := 
by 
  sorry

end total_bill_is_66_l681_681041


namespace solve_for_a_l681_681347

/-- Solve for a in the equation: log10(2 * a^2 - 20 * a) = 2 -/
theorem solve_for_a : 
  {a : ℝ} (h : log 10 (2 * a^2 - 20 * a) = 2) →
  a = 5 + 5 * real.sqrt 3 ∨ a = 5 - 5 * real.sqrt 3 :=
sorry

end solve_for_a_l681_681347


namespace inequality_solution_l681_681568

theorem inequality_solution (x : ℝ) : 
  (1 / (x^2 + 1) > 4 / x + 23 / 10) ↔ x ∈ Ioo (-2 : ℝ) 0 :=
sorry

end inequality_solution_l681_681568


namespace exclusivity_and_not_opposite_l681_681600

-- Definitions based on the conditions
def draw_two_from_bag (bag : list ℕ) : list (ℕ × ℕ) :=
  [(bag.nth 0, bag.nth 1), (bag.nth 0, bag.nth 2), (bag.nth 0, bag.nth 3), 
   (bag.nth 1, bag.nth 2), (bag.nth 1, bag.nth 3), (bag.nth 2, bag.nth 3)]
   
def exactly_one_white (draw: ℕ × ℕ) : Prop :=
  (draw.fst = 1 ∧ draw.snd = 2) ∨ (draw.fst = 2 ∧ draw.snd = 1)

def exactly_two_white (draw: ℕ × ℕ) : Prop :=
  (draw.fst = 1 ∧ draw.snd = 1)

-- Prove the given statement
theorem exclusivity_and_not_opposite :
  ∀ (bag : list ℕ),
  bag.length = 4 →
  (∀ draw ∈ draw_two_from_bag bag, exactly_one_white draw → ¬ exactly_two_white draw) ∧
  ¬ (∀ draw ∈ draw_two_from_bag bag, exactly_one_white draw ↔ ¬ exactly_two_white draw) :=
by
  intros bag h
  sorry

end exclusivity_and_not_opposite_l681_681600


namespace hyperbola_property_l681_681992

noncomputable def hyperbola (a : ℝ) : set (ℝ × ℝ) :=
  {p | p.1^2 - p.2^2 = a^2}

noncomputable def directrix_intersection (a : ℝ) : ℝ × ℝ :=
  (-a / Real.sqrt 2, 0)

noncomputable def focus (a : ℝ) : ℝ × ℝ :=
  (Real.sqrt 2 * a, 0)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

theorem hyperbola_property (a : ℝ) (D M N F : ℝ × ℝ)
  (MN_perpendicular : ∀ x, let p := (x, ((M.2 - N.2) / (M.1 - N.1)) * (x - M.1) + M.2) in (p ∈ hyperbola a) → x = M.1 ∨ x = N.1)
  (F_perpendicular_MN : ∀ x, let p := (x, ((-N.2 + M.2) / (N.1 - M.1)) * (x - F.1) + F.2) in (p ∈ hyperbola a) → x = F.1) :
  let P := (sign F.1 * Real.sqrt (2 * F.1 - D.1), F.2),
      Q := (sign F.1 * Real.sqrt (2 * F.1 - M.1), F.2) in
  distance F P * distance F Q = 2 * distance D M * distance D N :=
by
  sorry

end hyperbola_property_l681_681992


namespace slope_of_line_through_points_l681_681721

-- Define the points A and B
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (2, 2 + Real.sqrt 3)

-- Define the slope function between two points
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- State the theorem
theorem slope_of_line_through_points :
  slope A B = Real.sqrt 3 / 3 :=
by
  sorry

end slope_of_line_through_points_l681_681721


namespace no_counterexample_exists_for_T_l681_681798

def sum_of_digits (n : Nat) : Nat :=
  -- Function to compute sum of digits of a number
  (n / 100) + (n % 100 / 10) + (n % 10)

theorem no_counterexample_exists_for_T :
  ∀ (n : ℕ), n = 117 ∨ n = 126 ∨ n = 144 ∨ n = 171 →
    (sum_of_digits n) % 9 = 0 →
    n % 9 = 0 :=
by
  intros n h1 h2
  cases h1 with
  | inl inl heq1 => rw [heq1] at h2; sorry -- 117
  | inl inr heq2 => rw [heq2] at h2; sorry -- 126
  | inr inl heq3 => rw [heq3] at h2; sorry -- 144
  | inr inr heq4 => rw [heq4] at h2; sorry -- 171
  admit -- Prove the divisibility directly using arithmetic all satisfied cases

end no_counterexample_exists_for_T_l681_681798


namespace polygonal_number_8_8_l681_681379

-- Definitions based on conditions
def triangular_number (n : ℕ) : ℕ := (n^2 + n) / 2
def square_number (n : ℕ) : ℕ := n^2
def pentagonal_number (n : ℕ) : ℕ := (3 * n^2 - n) / 2
def hexagonal_number (n : ℕ) : ℕ := (4 * n^2 - 2 * n) / 2

-- General formula for k-sided polygonal number
def polygonal_number (n k : ℕ) : ℕ := ((k - 2) * n^2 + (4 - k) * n) / 2

-- The proposition to be proved
theorem polygonal_number_8_8 : polygonal_number 8 8 = 176 := by
  sorry

end polygonal_number_8_8_l681_681379


namespace problem_statement_l681_681725

-- Definitions of the operations △ and ⊗
def triangle (a b : ℤ) : ℤ := a + b + a * b - 1
def otimes (a b : ℤ) : ℤ := a * a - a * b + b * b

-- The theorem statement
theorem problem_statement : triangle 3 (otimes 2 4) = 50 := by
  sorry

end problem_statement_l681_681725


namespace sumsquare_properties_l681_681087

theorem sumsquare_properties {a b c d e f g h i : ℕ} (hc1 : a + b + c = d + e + f) 
(hc2 : d + e + f = g + h + i) 
(hc3 : a + e + i = d + e + f) 
(hc4 : c + e + g = d + e + f) : 
∃ m : ℕ, m % 3 = 0 ∧ (a ≤ (2 * m / 3 - 1)) ∧ (b ≤ (2 * m / 3 - 1)) ∧ (c ≤ (2 * m / 3 - 1)) ∧ (d ≤ (2 * m / 3 - 1)) ∧ (e ≤ (2 * m / 3 - 1)) ∧ (f ≤ (2 * m / 3 - 1)) ∧ (g ≤ (2 * m / 3 - 1)) ∧ (h ≤ (2 * m / 3 - 1)) ∧ (i ≤ (2 * m / 3 - 1)) := 
by {
  sorry
}

end sumsquare_properties_l681_681087


namespace gcd_294_84_l681_681015

theorem gcd_294_84 : gcd 294 84 = 42 :=
by
  sorry

end gcd_294_84_l681_681015


namespace flower_problem_solution_l681_681007

/-
Given the problem conditions:
1. There are 88 flowers.
2. Each flower was visited by at least one bee.
3. Each bee visited exactly 54 flowers.

Prove that bitter flowers exceed sweet flowers by 14.
-/

noncomputable def flower_problem : Prop :=
  ∃ (s g : ℕ), 
    -- Condition: The total number of flowers
    s + g + (88 - s - g) = 88 ∧ 
    -- Condition: Total number of visits by bees
    3 * 54 = 162 ∧ 
    -- Proof goal: Bitter flowers exceed sweet flowers by 14
    g - s = 14

theorem flower_problem_solution : flower_problem :=
by
  sorry

end flower_problem_solution_l681_681007


namespace sum_of_all_numbers_after_n_steps_l681_681994

def initial_sum : ℕ := 2

def sum_after_step (n : ℕ) : ℕ :=
  2 * 3^n

theorem sum_of_all_numbers_after_n_steps (n : ℕ) : 
  sum_after_step n = 2 * 3^n :=
by sorry

end sum_of_all_numbers_after_n_steps_l681_681994


namespace sum_abc_l681_681339

variable {a b c : ℝ}

-- Defining the conditions as hypotheses
def condition1 : Prop := a^2 + 6 * b = -17
def condition2 : Prop := b^2 + 8 * c = -23
def condition3 : Prop := c^2 + 2 * a = 14

-- The final statement we need to prove
theorem sum_abc : condition1 → condition2 → condition3 → a + b + c = -8 := by
  sorry

end sum_abc_l681_681339


namespace find_fx_for_neg_x_l681_681618

-- Let f be an odd function defined on ℝ 
variable {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = - f x)

-- Given condition for x > 0
variable (h_pos : ∀ x, 0 < x → f x = x^2 + x - 1)

-- Problem: Prove that f(x) = -x^2 + x + 1 for x < 0
theorem find_fx_for_neg_x (x : ℝ) (h_neg : x < 0) : f x = -x^2 + x + 1 :=
sorry

end find_fx_for_neg_x_l681_681618


namespace trigonometric_identity_proof_l681_681418

theorem trigonometric_identity_proof :
  sin (π / 4) * sin (5 * π / 12) + sin (π / 4) * sin (π / 12) = sqrt 3 / 2 := 
by
  sorry

end trigonometric_identity_proof_l681_681418


namespace equilateral_triangle_DEF_l681_681182

variables {Point : Type} [AffineGeometry Point]

structure Equilateral (A B C : Point) : Prop :=
  (AB_eq_BC : dist A B = dist B C)
  (BC_eq_CA : dist B C = dist C A)

noncomputable def parallelogram (A B C D : Point) : Prop :=
  (affine_segment A B).parallel (affine_segment C D) ∧
  (affine_segment B C).parallel (affine_segment D A)

theorem equilateral_triangle_DEF
  (A B C D E F : Point)
  (parallelogram_ABCD : parallelogram A B C D)
  (equilateral_ABE : Equilateral A B E)
  (equilateral_BCF : Equilateral B C F)
  : Equilateral D E F :=
sorry

end equilateral_triangle_DEF_l681_681182


namespace recurring_fraction_difference_l681_681113

theorem recurring_fraction_difference :
  let x := (36 / 99 : ℚ)
  let y := (36 / 100 : ℚ)
  x - y = (1 / 275 : ℚ) :=
by
  sorry

end recurring_fraction_difference_l681_681113


namespace num_customers_who_tried_sample_l681_681976

theorem num_customers_who_tried_sample :
  ∀ (samples_per_box boxes_opened samples_left : ℕ), 
  samples_per_box = 20 →
  boxes_opened = 12 →
  samples_left = 5 →
  let total_samples := samples_per_box * boxes_opened in
  let samples_used := total_samples - samples_left in
  samples_used = 235 :=
by 
  intros samples_per_box boxes_opened samples_left h_samples_per_box h_boxes_opened h_samples_left total_samples samples_used
  simp [h_samples_per_box, h_boxes_opened, h_samples_left]
  sorry

end num_customers_who_tried_sample_l681_681976


namespace product_of_three_numbers_l681_681003

theorem product_of_three_numbers (x y z n : ℝ)
  (h_sum : x + y + z = 180)
  (h_n_eq_8x : n = 8 * x)
  (h_n_eq_y_minus_10 : n = y - 10)
  (h_n_eq_z_plus_10 : n = z + 10) :
  x * y * z = (180 / 17) * ((1440 / 17) ^ 2 - 100) := by
  sorry

end product_of_three_numbers_l681_681003


namespace find_point_B_l681_681746

structure Point where
  x : Int
  y : Int

def translation (p : Point) (dx dy : Int) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem find_point_B :
  let A := Point.mk (-2) 3
  let A' := Point.mk 3 2
  let B' := Point.mk 4 0
  let dx := 5
  let dy := -1
  (translation A dx dy = A') →
  ∃ B : Point, translation B dx dy = B' ∧ B = Point.mk (-1) (-1) :=
by
  intros
  use Point.mk (-1) (-1)
  constructor
  sorry
  rfl

end find_point_B_l681_681746


namespace true_propositions_l681_681641

-- Define the propositions
def proposition1 : Prop :=
  ∀ a b c : ℝ, (¬ (ax + b) * (ax + b) - 4 * a * c > 0 → (ax + b) * (ax + b) = 0)

def proposition2 : Prop :=
  ∀ a b c d : ℝ, (a = b ∧ b = c ∧ c = d ∧ d = a → a * a = b * c)

def proposition3 : Prop :=
  ∀ x : ℝ, (x*x ≠ 9 ∨ x = 3)

def proposition4 : Prop :=
  ∀ x y : ℝ, (x = y → True)  -- This should actually represent the converse of vertically opposite angles, but simplified for the purpose of this task.

-- Proving which propositions are true
theorem true_propositions :
  proposition1 ∧ proposition2 ∧ proposition3 ∧ ¬ proposition4 := by
  sorry

end true_propositions_l681_681641


namespace prob_second_red_correct_l681_681488

noncomputable def prob_second_red (bagA_white : ℕ) (bagA_black : ℕ)
  (bagB_red : ℕ) (bagB_green : ℕ)
  (bagC_red : ℕ) (bagC_green : ℕ) : ℕ × ℕ :=
let prob_white_A := bagA_white / (bagA_white + bagA_black : ℚ),
    prob_black_A := bagA_black / (bagA_white + bagA_black : ℚ),
    prob_red_B := bagB_red / (bagB_red + bagB_green : ℚ),
    prob_red_C := bagC_red / (bagC_red + bagC_green : ℚ),
    prob_path1 := prob_white_A * prob_red_B,
    prob_path2 := prob_black_A * prob_red_C,
    total_prob := prob_path1 + prob_path2
in (total_prob.num, total_prob.denom)

theorem prob_second_red_correct :
  prob_second_red 4 5 3 7 5 3 = (12, 25) :=
by sorry

end prob_second_red_correct_l681_681488


namespace calculate_annual_interest_rate_l681_681426

-- Given conditions as definitions
def principal := 150
def total_repayment := 162
def repayment_period_years := 1.5

-- Define interest as the difference between total repayment and principal
def interest := total_repayment - principal

-- Define the annual interest rate formula
def annual_interest_rate : ℚ := (interest / (principal * repayment_period_years)) * 100

-- The statement to prove
theorem calculate_annual_interest_rate : (annual_interest_rate : ℚ) = 5 := by
  -- All necessary calculations are handled in the above definitions
  sorry

end calculate_annual_interest_rate_l681_681426


namespace largest_number_with_digit_sum_18_l681_681923

def digits := {d : ℕ | d > 0 ∧ d < 10}

noncomputable def digit_sum_18 (n : ℕ) : Prop :=
  let ds := n.digits 10 in
  (∀ d ∈ ds, d ∈ digits) ∧
  ds.sum = 18 ∧
  ∀ i j ∈ ds, i ≠ j

theorem largest_number_with_digit_sum_18 : 
  ∃ (n : ℕ), digit_sum_18 n ∧ (∀ m, digit_sum_18 m → m ≤ n) :=
begin
  use 6543,
  split,
  {
    ds := 6543.digits 10,
    have h1 : (∀ d ∈ ds, d ∈ digits), from sorry,
    have h2 : ds.sum = 18, from sorry,
    have h3 : ∀ i j ∈ ds, i ≠ j, from sorry,
    exact ⟨h1, h2, h3⟩,
  },
  {
    intros m hm,
    -- Add argument here to show 6543 is the largest
    sorry
  }
end

end largest_number_with_digit_sum_18_l681_681923


namespace cube_fly_triangulation_l681_681397

theorem cube_fly_triangulation :
  ∃ (v1 v2 v3 : ℝ × ℝ × ℝ) (u1 u2 u3 : ℝ × ℝ × ℝ),
  (is_vertex_of_cube v1)
  ∧ (is_vertex_of_cube v2)
  ∧ (is_vertex_of_cube v3)
  ∧ (permuted_position v1 u1)
  ∧ (permuted_position v2 u2)
  ∧ (permuted_position v3 u3)
  ∧ congruent_triangle v1 v2 v3 u1 u2 u3 :=
sorry

def is_vertex_of_cube (v : ℝ × ℝ × ℝ) : Prop :=
  v ∈ {(0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1)}

def permuted_position (v u : ℝ × ℝ × ℝ) : Prop :=
  is_vertex_of_cube u ∧ v ≠ u

def congruent_triangle (v1 v2 v3 u1 u2 u3 : ℝ × ℝ × ℝ) : Prop :=
  dist v1 v2 = dist u1 u2
  ∧ dist v2 v3 = dist u2 u3
  ∧ dist v3 v1 = dist u3 u1

noncomputable def dist (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := v1;
  let (x2, y2, z2) := v2;
  sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

end cube_fly_triangulation_l681_681397


namespace largest_area_E_l681_681237

def Polygon : Type := string
def unit_square_area := 1
def right_triangle_area := 0.5

def Area : Polygon → ℕ
| "A" => 6 * unit_square_area
| "B" => 3 * unit_square_area + 2 * right_triangle_area
| "C" => 4 * unit_square_area + 4 * right_triangle_area
| "D" => 5 * unit_square_area
| "E" => 6 * unit_square_area + 2 * right_triangle_area
| _   => 0

theorem largest_area_E :
  Area "E" = 7 ∧ (∀ p : Polygon, Area p ≤ Area "E") := 
by 
  sorry

end largest_area_E_l681_681237


namespace ratio_of_sides_l681_681078

-- Given
variables (sh ss : ℝ) (hex_perimeter square_perimeter : ℝ)
hypothesis (h1 : hex_perimeter = 6 * sh)
hypothesis (h2 : square_perimeter = 4 * ss)
hypothesis (h3 : hex_perimeter = square_perimeter)

-- To Prove
theorem ratio_of_sides (sh ss : ℝ) (hex_perimeter square_perimeter : ℝ)
  (h1 : hex_perimeter = 6 * sh)
  (h2 : square_perimeter = 4 * ss)
  (h3 : hex_perimeter = square_perimeter) :
  ss / sh = 3 / 2 :=
by
  sorry

end ratio_of_sides_l681_681078


namespace james_buys_78_toys_l681_681760

def buys_toys :=
  ∃ (cars soldiers : ℕ), 
    (cars > 25 ∨ cars ≤ 25) ∧  -- condition for discount applicability
    (∃ initial_cars, initial_cars = 20) ∧  -- initial cars James buys
    (cars > 25 → cars = 26) ∧  -- to maximize discount, he buys 26 cars if more than 25 initial cars
    (soldiers = 2 * cars) ∧ -- he buys twice as many toy soldiers as toy cars
    (initial_cars > 25 → cars + soldiers == 78 ∧ initial_cars ≤ 25 → cars + soldiers == 78)  -- total number of toys should be 78

theorem james_buys_78_toys : buys_toys :=
sorry

end james_buys_78_toys_l681_681760


namespace james_needs_to_work_50_hours_l681_681762

def wasted_meat := 20
def cost_meat_per_pound := 5
def wasted_vegetables := 15
def cost_vegetables_per_pound := 4
def wasted_bread := 60
def cost_bread_per_pound := 1.5
def janitorial_hours := 10
def janitor_rate := 10
def time_and_half_multiplier := 1.5
def min_wage := 8

theorem james_needs_to_work_50_hours :
  let cost_meat := wasted_meat * cost_meat_per_pound in
  let cost_vegetables := wasted_vegetables * cost_vegetables_per_pound in
  let cost_bread := wasted_bread * cost_bread_per_pound in
  let time_and_half_rate := janitor_rate * time_and_half_multiplier in
  let cost_janitorial := janitorial_hours * time_and_half_rate in
  let total_cost := cost_meat + cost_vegetables + cost_bread + cost_janitorial in
  let hours_to_work := total_cost / min_wage in
  hours_to_work = 50 := by
  sorry

end james_needs_to_work_50_hours_l681_681762


namespace cary_net_calorie_deficit_is_250_l681_681148

-- Define the conditions
def miles_walked : ℕ := 3
def candy_bar_calories : ℕ := 200
def calories_per_mile : ℕ := 150

-- Define the function to calculate total calories burned
def total_calories_burned (miles : ℕ) (calories_per_mile : ℕ) : ℕ :=
  miles * calories_per_mile

-- Define the function to calculate net calorie deficit
def net_calorie_deficit (total_calories : ℕ) (candy_calories : ℕ) : ℕ :=
  total_calories - candy_calories

-- The statement to be proven
theorem cary_net_calorie_deficit_is_250 :
  net_calorie_deficit (total_calories_burned miles_walked calories_per_mile) candy_bar_calories = 250 :=
  by sorry

end cary_net_calorie_deficit_is_250_l681_681148


namespace numbers_sum_prod_l681_681639

theorem numbers_sum_prod (x y : ℝ) (h_sum : x + y = 10) (h_prod : x * y = 24) : (x = 4 ∧ y = 6) ∨ (x = 6 ∧ y = 4) :=
by
  sorry

end numbers_sum_prod_l681_681639


namespace angle_B_determined_y_as_function_of_x_and_range_of_AC_l681_681730

variables {A B C D : Type} [metric_space D] {a b c x y : ℝ}
variables {angle_A angle_B angle_C : ℝ}
variables {m n : ℝ × ℝ}

-- Conditions
def vector_m := (2 * a + c, b)
def vector_n := (real.cos angle_B, real.cos angle_C)
def perpendicular (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 = 0
def sides := (a * real.sin angle_A, b * real.sin angle_B, c * real.sin angle_C)
def law_of_sines := a / (real.sin angle_A) = b / (real.sin angle_B) = c / (real.sin angle_C)

-- Given condition: bisector of ∠ABC intersects AC at point D
def bisector_condition (BD : ℝ) := BD = 1
def triangle_sides := BC = x ∧ BA = y

-- Prove segment (1): angle B
theorem angle_B_determined (h : perpendicular vector_m vector_n) : cos angle_B = -1/2 → angle_B = 2 * π / 3 :=
sorry 

-- Prove segment (2): y as a function of x and the range of AC
theorem y_as_function_of_x_and_range_of_AC (h_sides : sides x y c)
(h_bisector: bisector_condition 1) : y = x / (x - 1) ∧ (2 * sqrt 3 ≤ AC ∧ AC < 4) :=
sorry

end angle_B_determined_y_as_function_of_x_and_range_of_AC_l681_681730


namespace binom_18_6_eq_13260_l681_681515

/-- The binomial coefficient formula. -/
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The specific proof problem: compute binom(18, 6) and show that it equals 13260. -/
theorem binom_18_6_eq_13260 : binom 18 6 = 13260 :=
by
  sorry

end binom_18_6_eq_13260_l681_681515


namespace combination_18_6_l681_681498

theorem combination_18_6 : (nat.choose 18 6) = 18564 := 
by 
  sorry

end combination_18_6_l681_681498


namespace greatest_div_by_seven_base_eight_l681_681893

theorem greatest_div_by_seven_base_eight : ∃ n : ℕ, 
  (n < 512) ∧ (Divisibility.divides 7 n) ∧ 
  (∀ m : ℕ, (m < 512) → (Divisibility.divides 7 m) → m ≤ n) ∧ 
  nat.to_digits 8 n = [7, 7, 4] := 
sorry

end greatest_div_by_seven_base_eight_l681_681893


namespace part1_part2_l681_681793

variables {a b x t : ℝ}

noncomputable def y := λ a x b => a * x^2 + x - b
noncomputable def inequality := λ a x b => y a x b < (a-1) * x^2 + (b+2) * x - 2 * b

theorem part1 (a b : ℝ) : 
  if b < 1 then { x : ℝ | b < x ∧ x < 1 } 
  else if b = 1 then { x : ℝ | false } 
  else if b > 1 then { x : ℝ | 1 < x ∧ x < b } :=
sorry

noncomputable def P := { x : ℝ | y a x b > 0 }
noncomputable def Q := λ t, { x : ℝ | -2 - t < x ∧ x < -2 + t }

theorem part2 (a b : ℝ) (h1 : a > 0) (h2 : b > 1) (h3 : ∀ t > 0, (P a x b) ∩ (Q t) ≠ ∅) : 
  (1/a - 1/b = 1/2) :=
sorry

end part1_part2_l681_681793


namespace student_solves_exactly_20_problems_l681_681465

theorem student_solves_exactly_20_problems :
  (∀ n, 1 ≤ (a : ℕ → ℕ) n) ∧ (∀ k, a (k + 7) ≤ a k + 12) ∧ a 77 ≤ 132 →
  ∃ i j, i < j ∧ a j - a i = 20 := sorry

end student_solves_exactly_20_problems_l681_681465


namespace root_ratio_equiv_l681_681999

theorem root_ratio_equiv :
  (81 ^ (1 / 3)) / (81 ^ (1 / 4)) = 81 ^ (1 / 12) :=
by
  sorry

end root_ratio_equiv_l681_681999


namespace new_sales_tax_rate_l681_681875
open Real

constant original_sales_tax_rate : ℝ := 0.035
constant market_price : ℝ := 6600
constant sales_tax_difference : ℝ := 10.999999999999991

theorem new_sales_tax_rate :
  let original_sales_tax_amount := market_price * original_sales_tax_rate in
  let new_sales_tax_amount := original_sales_tax_amount + sales_tax_difference in
  let new_sales_tax_rate := (new_sales_tax_amount / market_price) * 100 in
  abs (new_sales_tax_rate - 3.67) < 1e-2 :=
by
  sorry

end new_sales_tax_rate_l681_681875


namespace func_value_at_l681_681642

def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2 else 5 ^ x

theorem func_value_at 
: f(f(1/4)) = 1/25 := sorry

end func_value_at_l681_681642


namespace quadratic_equation_equivalence_l681_681258

theorem quadratic_equation_equivalence
  (a_0 a_1 a_2 : ℝ)
  (r s : ℝ)
  (h_roots : a_0 + a_1 * r + a_2 * r^2 = 0 ∧ a_0 + a_1 * s + a_2 * s^2 = 0)
  (h_a2_nonzero : a_2 ≠ 0) :
  (∀ x, a_0 ≠ 0 ↔ a_0 + a_1 * x + a_2 * x^2 = a_0 * (1 - x / r) * (1 - x / s)) :=
sorry

end quadratic_equation_equivalence_l681_681258


namespace percentage_cost_for_overhead_l681_681385

theorem percentage_cost_for_overhead
  (P M N : ℝ)
  (hP : P = 48)
  (hM : M = 50)
  (hN : N = 12) :
  (P + M - P - N) / P * 100 = 79.17 := by
  sorry

end percentage_cost_for_overhead_l681_681385


namespace cost_price_of_article_is_correct_l681_681463

-- Define the given conditions as variables
variables (SP : ℝ) (profit_percent : ℝ)

-- The cost price calculated from given conditions
def CP := SP / (1 + profit_percent)

-- Statement of the problem in Lean 4
theorem cost_price_of_article_is_correct (h1 : SP = 100) (h2 : profit_percent = 0.40) : CP SP profit_percent = 71.43 :=
by
  -- Expand definitions and use provided hypotheses
  unfold CP 
  rw [h1, h2]
  norm_num
  rw [div_eq_mul_inv]
  norm_num
  have h: (100 : ℝ) / 1.40 = 100 * (1 / 1.40), from div_eq_mul_inv 100 1.40
  rw [h]
  norm_num
  sorry

end cost_price_of_article_is_correct_l681_681463


namespace problem_l681_681785

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (α : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (π * x + α) + b * Real.cos (π * x + β)

theorem problem 
  (a b α β : ℝ) 
  (h₁ : ab ≠ 0) 
  (h₂ : ∀ k : ℤ, α ≠ k * π) 
  (h₃ : f 2009 a b α β = 5) : 
  f 2010 a b α β = -5 :=
by
  sorry

end problem_l681_681785


namespace modified_determinant_l681_681717

def determinant_2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem modified_determinant (x y z w : ℝ)
  (h : determinant_2x2 x y z w = 6) :
  determinant_2x2 x (5 * x + 4 * y) z (5 * z + 4 * w) = 24 := by
  sorry

end modified_determinant_l681_681717


namespace leading_coeff_of_polynomial_l681_681382

theorem leading_coeff_of_polynomial (f : ℕ → ℝ) (h : ∀ x : ℕ, f (x + 1) - f x = 2 * (x:ℝ) ^ 2 + 6 * x + 4) :
  polynomial.leading_coeff (polynomial.of_real (f : ℕ → ℝ)) = 1 / 3 := 
sorry

end leading_coeff_of_polynomial_l681_681382


namespace english_marks_are_67_l681_681343

noncomputable def marks_in_english
                  (math_marks : ℕ) (science_marks : ℕ) (social_studies_marks : ℕ) (biology_marks : ℕ)
                  (average_marks : ℕ) (num_subjects : ℕ) : ℕ :=
  average_marks * num_subjects - (math_marks + science_marks + social_studies_marks + biology_marks)

theorem english_marks_are_67 :
  marks_in_english 76 65 82 85 75 5 = 67 :=
by
  simp [marks_in_english]
  norm_num

end english_marks_are_67_l681_681343


namespace james_needs_50_hours_l681_681765

-- Define the given conditions
def meat_cost_per_pound : ℝ := 5
def meat_pounds_wasted : ℝ := 20
def fruits_veg_cost_per_pound : ℝ := 4
def fruits_veg_pounds_wasted : ℝ := 15
def bread_cost_per_pound : ℝ := 1.5
def bread_pounds_wasted : ℝ := 60
def janitorial_hourly_rate : ℝ := 10
def janitorial_hours_worked : ℝ := 10
def james_hourly_rate : ℝ := 8

-- Calculate the total costs separately
def total_meat_cost : ℝ := meat_cost_per_pound * meat_pounds_wasted
def total_fruits_veg_cost : ℝ := fruits_veg_cost_per_pound * fruits_veg_pounds_wasted
def total_bread_cost : ℝ := bread_cost_per_pound * bread_pounds_wasted
def janitorial_time_and_a_half_rate : ℝ := janitorial_hourly_rate * 1.5
def total_janitorial_cost : ℝ := janitorial_time_and_a_half_rate * janitorial_hours_worked

-- Calculate the total cost James has to pay
def total_cost : ℝ := total_meat_cost + total_fruits_veg_cost + total_bread_cost + total_janitorial_cost

-- Calculate the number of hours James needs to work
def james_work_hours : ℝ := total_cost / james_hourly_rate

-- The theorem to be proved
theorem james_needs_50_hours : james_work_hours = 50 :=
by
  sorry

end james_needs_50_hours_l681_681765


namespace number_of_factors_of_72_l681_681701

/-- 
Given that 72 can be factorized as 2^3 * 3^2, 
prove that the number of distinct positive factors of 72 is 12 
--/
theorem number_of_factors_of_72 : 
  let n := 72 in 
  let a := 3 in 
  let b := 2 in 
  n = 2 ^ a * 3 ^ b → (a + 1) * (b + 1) = 12 := 
by 
  intros n a b h,
  rw h,
  simp,
  sorry

end number_of_factors_of_72_l681_681701


namespace binary_representation_of_41_l681_681164

def decimal_to_binary (n : ℕ) : ℕ :=
match n with
| 0     => 0
| _     => decimal_to_binary (n / 2) * 10 + (n % 2)

theorem binary_representation_of_41 :
  decimal_to_binary 41 = 101001 :=
by
  sorry

end binary_representation_of_41_l681_681164


namespace greatest_3digit_base8_divisible_by_7_l681_681901

def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : ℕ), n = 0b777 ∧ (base8_to_base10 0b777) % 7 = 0 ∧ ∀ m < 0o777, m % 7 = 0 → base8_to_base10 m < base8_to_base10 0b777 :=
by
  sorry

end greatest_3digit_base8_divisible_by_7_l681_681901


namespace determine_scalar_k_l681_681302

theorem determine_scalar_k (A B C D E : Type*) [AddCommGroup E] [Module ℝ E]
  (OA OB OC OD OE : E)
  (h : 2 • OA - 3 • OB + 4 • OC + k • OD + 2 • OE = (0 : E)) :
  ∃ k : ℝ, k = -5 :=
by
  sorry

end determine_scalar_k_l681_681302


namespace combination_18_6_l681_681500

theorem combination_18_6 : (nat.choose 18 6) = 18564 := 
by 
  sorry

end combination_18_6_l681_681500


namespace count_negative_numbers_l681_681092

def is_negative (n : ℚ) : Prop := n < 0

theorem count_negative_numbers :
  let nums := [-8/3, 9/14, -3, 25/10, 0, -48/10, 5, -1]
  list.count is_negative nums = 4 :=
by
  let nums: list ℚ := [-8/3, 9/14, -3, 25/10, 0, -48/10, 5, -1]
  sorry

end count_negative_numbers_l681_681092


namespace arithmetic_sequence_sum_l681_681142

-- Define the arithmetic sequence properties
def is_arithmetic_sequence (seq : ℕ → ℕ) :=
  ∀ n : ℕ, seq (n + 1) = seq n + 2

-- Define the arithmetic sequence in question
def sequence : ℕ → ℕ
| 0       := 2
| (n + 1) := sequence n + 2

-- Check that our sequence matches the properties of an arithmetic sequence
lemma sequence_is_arithmetic : is_arithmetic_sequence sequence :=
by intros n; simp [sequence]

-- Define the sum of the first n terms of the sequence
def sum_n_terms (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, sequence i

-- State the main theorem to be proven: the sum of the first 10 terms is 110
theorem arithmetic_sequence_sum : sum_n_terms 10 = 110 :=
sorry

end arithmetic_sequence_sum_l681_681142


namespace probability_product_multiple_of_4_l681_681712

theorem probability_product_multiple_of_4 :
  let cards := {1, 2, 3, 4, 5, 6}
  let pairs := { (a, b) | a ∈ cards ∧ b ∈ cards ∧ a < b }
  let total_pairs := 15
  let valid_pairs := { (1, 4), (2, 4), (3, 4), (4, 5), (4, 6) }
  let num_valid_pairs := 5
  num_valid_pairs / total_pairs = 1 / 3 := by
  sorry

end probability_product_multiple_of_4_l681_681712


namespace number_of_parallel_planes_l681_681380

variable {Point : Type} [EuclideanSpace Point]

-- Definitions for points and planes
def is_parallel (π₁ π₂ : Set Point) : Prop := sorry -- definition of parallel planes
def on_the_same_side (π : Set Point) (P Q : Point) : Prop := sorry -- definition of points on the same side

-- Theorem statement
theorem number_of_parallel_planes (P Q : Point) (π : Set Point) :
  (∃ π' : Set Point, is_parallel π π' ∧ (P ∈ π') ∧ (Q ∈ π')) →
  (∃ d : ℕ, d = 0 ∨ d = 1) := sorry

end number_of_parallel_planes_l681_681380


namespace sequence1_general_term_sequence2_general_term_sequence3_general_term_sequence4_general_term_l681_681039

-- Sequence 1: 4, 6, 8, 10, ...
theorem sequence1_general_term (n : ℕ) (h : n > 0) : (∃ a : ℕ, a = 2 * (n + 1)) :=
  sorry

-- Sequence 2: -1/(1×2), 1/(2×3), -1/(3×4), 1/(4×5), ...
theorem sequence2_general_term (n : ℕ) (h : n > 0) :
  (∃ a : ℚ, a = (-1)^n * (1 / (n * (n + 1)))) :=
  sorry

-- Sequence 3: a, b, a, b, a, b, ... (where a, b are real numbers)
theorem sequence3_general_term (n : ℕ) (h : n > 0) (a b : ℝ) :
  (∃ c : ℝ, (c = a ∧ odd n) ∨ (c = b ∧ even n)) :=
  sorry

-- Sequence 4: 9, 99, 999, 9999, ...
theorem sequence4_general_term (n : ℕ) (h : n > 0) : 
  (∃ a : ℕ, a = 10^n - 1) :=
  sorry

end sequence1_general_term_sequence2_general_term_sequence3_general_term_sequence4_general_term_l681_681039


namespace sum_of_digits_of_palindrome_x_l681_681061

-- Define a three-digit number as a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

/-- The sum of digits of x, when x is a three-digit palindrome and x + 50 is also a palindrome 
    (either three or four digits), is 22. -/
theorem sum_of_digits_of_palindrome_x (x : ℕ) (hx1 : 100 ≤ x ∧ x ≤ 999) (hx2 : is_palindrome x) (hx3 : is_palindrome (x + 50)) : 
  x.digits 10.sum = 22 := 
sorry

end sum_of_digits_of_palindrome_x_l681_681061


namespace moon_speed_in_kilometers_per_second_l681_681863

def kilometers_per_hour_to_per_second (speed_kmh : ℕ) : ℚ :=
  speed_kmh / 3600

theorem moon_speed_in_kilometers_per_second (h : kilometers_per_hour_to_per_second 3744 = 1.04) : True :=
by {
  sorry,
}

end moon_speed_in_kilometers_per_second_l681_681863


namespace sum_three_distinct_zero_l681_681299

variable {R : Type} [Field R]

theorem sum_three_distinct_zero
  (a b c x y : R)
  (h1 : a ^ 3 + a * x + y = 0)
  (h2 : b ^ 3 + b * x + y = 0)
  (h3 : c ^ 3 + c * x + y = 0)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  a + b + c = 0 := by
  sorry

end sum_three_distinct_zero_l681_681299


namespace acute_triangle_side_length_range_l681_681615

theorem acute_triangle_side_length_range (a : ℝ) (h : 1 < 2 ∧ 2 < a): 
  (\sqrt{3} < a ∧ a < \sqrt{5}) :=
begin
  sorry
end

end acute_triangle_side_length_range_l681_681615


namespace monkey_seat_probability_l681_681592

theorem monkey_seat_probability :
  let seats := ["P", "Q", "R", "S", "T"];
  let monkeys := [1, 2, 3, 4, 5];
  let Q := seats.indexOf "Q";
  let R := seats.indexOf "R";
  let result :=
    -- Probabilities in each scenario
    (1/5) * (1/4) * 3 in
  result = 3 / 20 :=
by
  -- Sorry to skip the proof
  sorry

end monkey_seat_probability_l681_681592


namespace max_distance_from_curve_to_polar_point_l681_681290

-- Define the polar curve
def curve (θ : ℝ) : ℝ := 2 * Real.cos θ

-- Define the polar point of interest
def polar_point : ℝ × ℝ := (1, Real.pi)

-- Define the rectangular coordinates of the point of interest
def rectangular_polar_point : ℝ × ℝ := (-1, 0)

-- The center of the circle obtained from converting the polar curve to rectangular coordinates
def center_of_circle : ℝ × ℝ := (1, 0)

-- The radius of the circle obtained in the conversion
def radius_of_circle : ℝ := 1

-- Calculate the distance between the center of the circle and the rectangular_polar_point
def distance_to_polar_point : ℝ := Real.dist center_of_circle rectangular_polar_point -- which is 2

-- State the theorem
theorem max_distance_from_curve_to_polar_point : 
  ∀ θ : ℝ, ∃ ρ : ℝ, ρ = curve θ →
  Real.dist (ρ * Real.cos θ, ρ * Real.sin θ) (polar_point.1 * Real.cos polar_point.2, polar_point.1 * Real.sin polar_point.2) = 3 :=
sorry

end max_distance_from_curve_to_polar_point_l681_681290


namespace train_crossing_time_l681_681090

-- Defining basic conditions
def train_length : ℕ := 150
def platform_length : ℕ := 100
def time_to_cross_post : ℕ := 15

-- The time it takes for the train to cross the platform
theorem train_crossing_time :
  (train_length + platform_length) / (train_length / time_to_cross_post) = 25 := 
sorry

end train_crossing_time_l681_681090


namespace combination_18_6_l681_681524

theorem combination_18_6 : nat.choose 18 6 = 18564 :=
by {
  sorry
}

end combination_18_6_l681_681524


namespace distinct_triangles_count_l681_681688

theorem distinct_triangles_count (a b c : ℕ) (h1 : a + b + c = 11) (h2 : a + b > c) 
  (h3 : a + c > b) (h4 : b + c > a) : 
  10 := sorry

end distinct_triangles_count_l681_681688


namespace find_solutions_l681_681189

theorem find_solutions :
  ∀ x y z : ℝ,
  (x^2 + y + z = 1) ∧
  (x + y^2 + z = 1) ∧
  (x + y + z^2 = 1) →
  (x, y, z) = (1, 0, 0) ∨
  (x, y, z) = (0, 1, 0) ∨
  (x, y, z) = (0, 0, 1) ∨
  (x, y, z) = (-1 - real.sqrt 2, -1 - real.sqrt 2, -1 - real.sqrt 2) ∨
  (x, y, z) = (-1 + real.sqrt 2, -1 + real.sqrt 2, -1 + real.sqrt 2) :=
by
  sorry

end find_solutions_l681_681189


namespace cake_fraction_after_six_trips_l681_681934

-- Definition of the fraction of cake eaten on each trip.
def fractionEatenAfterNTips (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, 1 / (3 ^ (i + 1))

-- The main theorem to prove
theorem cake_fraction_after_six_trips :
  fractionEatenAfterNTips 6 = 364 / 729 := 
by
  sorry

end cake_fraction_after_six_trips_l681_681934


namespace expression_value_is_k_times_30_power_l681_681494

theorem expression_value_is_k_times_30_power (k : ℕ) (a b : ℕ) :
  (a^1003 + b^1004)^2 - (a^1003 - b^1004)^2 = k * (a * b) ^ 1003 :=
begin
  sorry
end

example : expression_value_is_k_times_30_power 24 5 6 := by
  sorry

end expression_value_is_k_times_30_power_l681_681494


namespace bus_initial_count_l681_681106

theorem bus_initial_count (x : ℕ) (got_off : ℕ) (remained : ℕ) (h1 : got_off = 47) (h2 : remained = 43) (h3 : x - got_off = remained) : x = 90 :=
by
  rw [h1, h2] at h3
  sorry

end bus_initial_count_l681_681106


namespace meet_probability_objects_l681_681816

theorem meet_probability_objects (C D : Type) : 
    (prob_meet C D (0,0) (4,6) (5:ℕ) (2^5: ℕ) (2^5: ℕ)) = (55 / 1024 : ℚ) :=
sorry

-- Definitions: 

-- Object C and D each takes 1 step per move
def prob_meet (C D : Type) (startC startD : ℤ × ℤ) (steps : ℕ) (total_moves_C total_moves_D : ℕ) : ℚ :=
    ∑ i in finset.range (steps + 1), 
      (nat.choose steps i) * (nat.choose steps (i+1)) / (total_moves_C * total_moves_D : ℚ)

end meet_probability_objects_l681_681816


namespace choose_two_items_proof_l681_681705

   def number_of_ways_to_choose_two_items (n : ℕ) : ℕ :=
     n * (n - 1) / 2

   theorem choose_two_items_proof (n : ℕ) : number_of_ways_to_choose_two_items n = (n * (n - 1)) / 2 :=
   by
     sorry
   
end choose_two_items_proof_l681_681705


namespace function_for_C2_is_f_1_sub_x_l681_681859

variable (f : ℝ → ℝ)

def C := λ x, f x

def C1 := λ x, f (2 - x)

def C2 := λ x, f (2 - (x + 2))

theorem function_for_C2_is_f_1_sub_x : C2 f = λ x, f (1 - x) :=
by
  funext x
  simp [C2]
  sorry

end function_for_C2_is_f_1_sub_x_l681_681859


namespace solve_problem_l681_681566

noncomputable def find_functions (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f(x + y) = f(x) + f(y) + 2 * x * y) ∧
  (∀ n : ℕ, differentiable ℝ^[n] f) ∧
  ∃ a : ℝ, ∀ x : ℝ, f(x) = x^2 + a * x

-- The goal is to prove this statement:
theorem solve_problem {f : ℝ → ℝ} : find_functions f := sorry

end solve_problem_l681_681566


namespace pyramid_height_l681_681440

theorem pyramid_height (h : ℝ) :
  let V_cube := 5^3
  let V_pyramid := (1/3) * 10^2 * h
  V_cube = V_pyramid → h = 3.75 :=
by
  let V_cube := 5^3
  let V_pyramid := (1/3) * 10^2 * h
  intros h_eq
  sorry

end pyramid_height_l681_681440


namespace floor_of_4point7_l681_681548

theorem floor_of_4point7 : (Real.floor 4.7) = 4 :=
by
  sorry

end floor_of_4point7_l681_681548


namespace region_S_area_correct_l681_681826

noncomputable def area_of_region_S_in_rhombus_G : ℝ :=
  let side_length := 3
  let angle_G := 150
  -- Definition of the region S in terms of perpendicular bisectors
  let area_S := (9 * (2 - Math.sqrt 3)) / 4
  area_S

theorem region_S_area_correct :
  area_of_region_S_in_rhombus_G = (9 * (2 - Math.sqrt 3)) / 4 :=
begin
  sorry
end

end region_S_area_correct_l681_681826


namespace greatest_3_digit_base_8_divisible_by_7_l681_681915

open Nat

def is_3_digit_base_8 (n : ℕ) : Prop := n < 8^3

def is_divisible_by_7 (n : ℕ) : Prop := 7 ∣ n

theorem greatest_3_digit_base_8_divisible_by_7 :
  ∃ x : ℕ, is_3_digit_base_8 x ∧ is_divisible_by_7 x ∧ x = 7 * (8 * (8 * 7 + 7) + 7) :=
by
  sorry

end greatest_3_digit_base_8_divisible_by_7_l681_681915


namespace final_l681_681804

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ∈ [-3, -2] then 4 * x
  else sorry

lemma f_periodic (h : ∀ x : ℝ, f (x + 3) = - (1 / f x)) :
 ∀ x : ℝ, f (x + 6) = f x :=
sorry

lemma f_even (h : ∀ x : ℝ, f x = f (-x)) : ℕ := sorry

theorem final (h1 : ∀ x : ℝ, f (x + 3) = - (1 / f x))
  (h2 : ∀ x : ℝ, f x = f (-x))
  (h3 : ∀ x : ℝ, x ∈ [-3, -2] → f x = 4 * x) :
  f 107.5 = 1 / 10 :=
sorry

end final_l681_681804


namespace minimum_value_of_abs_phi_l681_681265

theorem minimum_value_of_abs_phi (φ : ℝ) :
  (∃ k : ℤ, φ = k * π - (13 * π) / 6) → 
  ∃ φ_min : ℝ, 0 ≤ φ_min ∧ φ_min = abs φ ∧ φ_min = π / 6 :=
by
  sorry

end minimum_value_of_abs_phi_l681_681265


namespace find_sum_lent_l681_681951

-- Define the given conditions
def P : ℝ -- Sum lent
def rate : ℝ := 0.07 -- Interest rate per annum at simple interest
def time : ℝ := 7 -- Time in years
def interest : ℝ := P * rate * time -- Interest formula

-- The problem condition that interest is $2500 less than the sum lent
def condition := interest = P - 2500

-- Lean statement to prove that P equals 4901.96 under the given conditions
theorem find_sum_lent (h : condition) : P = 4901.96 :=
by
  sorry

end find_sum_lent_l681_681951


namespace max_non_threatening_cells_l681_681088

-- Define basic elements for the grid
def Cell := (ℕ × ℕ)
def distance (a b : Cell) : ℕ := 
  max (abs (a.fst - b.fst)) (abs (a.snd - b.snd))

def max_marked_cells : ℕ :=
  6050

-- Properties and conditions of the problem
variable (n : ℕ) (cells : Finset Cell)
variable (H_cells_size : cells.card = n)
variable (H_cells_bound : ∀ (x y : Cell), x ∈ cells → y ∈ cells → x ≠ y → distance x y ≠ 15)
variable (H_grid_size : ∀ (x : Cell), x ∈ cells → x.fst < 110 ∧ x.snd < 110)

theorem max_non_threatening_cells :
  max_marked_cells = 6050 :=
sorry

end max_non_threatening_cells_l681_681088


namespace find_2a_minus_3b_l681_681675

theorem find_2a_minus_3b
  (a b : ℝ)
  (h1 : a * 2 - b * 1 = 4)
  (h2 : a * 2 + b * 1 = 2) :
  2 * a - 3 * b = 6 :=
by
  sorry

end find_2a_minus_3b_l681_681675


namespace fibonacci_sum_identity_find_fibonacci_indices_sum_fibonacci_indices_sum_correct_l681_681847

def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

theorem fibonacci_sum_identity :
  (∑ i1 in finRange (101), ∑ i2 in finRange (101), ∑ i3 in finRange (101), ∑ i4 in finRange (101), ∑ i5 in finRange (101), fibonacci (i1 + i2 + i3 + i4 + i5))
  = fibonacci 510 - 5 * fibonacci 409 + 10 * fibonacci 308 - 10 * fibonacci 207 + 5 * fibonacci 106 - fibonacci 5 :=
sorry

theorem find_fibonacci_indices_sum :
  let n1 := 510
  let n2 := 409
  let n3 := 308
  let n4 := 207
  let n5 := 106
  let n6 := 5
  n1 + n2 + n3 + n4 + n5 + n6 = 1545 :=
by
  simp only [add_comm, add_assoc]
  norm_num
  exact Nat.add_comm 5 1540

theorem fibonacci_indices_sum_correct : ∃ n1 n2 n3 n4 n5 n6 : ℕ,
  (∑ i1 in finRange (101), ∑ i2 in finRange (101), ∑ i3 in finRange (101), ∑ i4 in finRange (101), ∑ i5 in finRange (101), fibonacci (i1 + i2 + i3 + i4 + i5))
  = fibonacci n1 - 5 * fibonacci n2 + 10 * fibonacci n3 - 10 * fibonacci n4 + 5 * fibonacci n5 - fibonacci n6
  ∧ n1 + n2 + n3 + n4 + n5 + n6 = 1545 :=
by
  use 510, 409, 308, 207, 106, 5
  refine ⟨_, find_fibonacci_indices_sum⟩
  exact fibonacci_sum_identity

end fibonacci_sum_identity_find_fibonacci_indices_sum_fibonacci_indices_sum_correct_l681_681847


namespace max_students_seated_l681_681097

/-- 
An auditorium has 20 rows; the first row has 12 seats, with each subsequent row having 2 more seats than the previous one. 
Given that students cannot sit next to each other in the same row, prove that the maximum number of students that can be seated is 310.
-/
theorem max_students_seated : 
  let rows := 20
  let seats (i : ℕ) := 10 + 2 * i
  let max_students_per_row (n : ℕ) := (n + 1) / 2
  310 = ∑ i in finset.range rows, max_students_per_row (seats i) :=
sorry

end max_students_seated_l681_681097


namespace problem_l681_681790

theorem problem (n : ℕ) (h : n = 8 ^ 2022) : n / 4 = 4 ^ 3032 := 
sorry

end problem_l681_681790


namespace reflect_point_across_x_axis_l681_681361

theorem reflect_point_across_x_axis 
  (P : ℝ × ℝ) (hP : P = (2, 5)) 
  : reflects_across_x_axis P = (2, -5) :=
sorry

def reflects_across_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

end reflect_point_across_x_axis_l681_681361


namespace guards_catch_the_monkey_l681_681739

/- Define the problem conditions -/
inductive Position where
  | A | B | C | D | E | F 

structure State where
  guard1 : Position
  guard2 : Position
  monkey : Position

/- Define the move function that ensures the monkey runs 3 times faster -/
/- This is a simplified conceptual description because actual path and timing intricacies require a detailed simulation model -/
constant faster_monkey : State → State → Prop

/- Initial state where guards are at one vertex and monkey at another -/
def initialState : State :=
  { guard1 := Position.A, guard2 := Position.A, monkey := Position.B }

theorem guards_catch_the_monkey (s : State) :
  ∃ s', faster_monkey s s' →
  (s.guard1 ≠ s'.monkey ∧ s.guard2 ≠ s'.monkey) →
  eventual_catch s' :=
sorry

end

end guards_catch_the_monkey_l681_681739


namespace amara_threw_away_15_clothes_l681_681091

variable (initial_clothes donated1 donated2 remaining_clothes : ℕ)

def total_donated := donated1 + donated2
def total_used := initial_clothes - remaining_clothes
def total_thrown_away := total_used - total_donated

theorem amara_threw_away_15_clothes 
  (h1 : initial_clothes = 100)
  (h2 : donated1 = 5)
  (h3 : donated2 = 3 * donated1)
  (h4 : remaining_clothes = 65) :
  total_thrown_away initial_clothes donated1 donated2 remaining_clothes = 15 := by
  sorry

end amara_threw_away_15_clothes_l681_681091


namespace mean_of_sets_l681_681938

theorem mean_of_sets (x : ℝ) (h : (28 + x + 42 + 78 + 104) / 5 = 62) : 
  (48 + 62 + 98 + 124 + x) / 5 = 78 :=
by
  sorry

end mean_of_sets_l681_681938


namespace tan_value_l681_681056

theorem tan_value (α : ℝ) (h₁ : sin α + cos α = 7 / 13) (h₂ : α > 0) (h₃ : α < π) : tan α = -12 / 5 :=
sorry

end tan_value_l681_681056


namespace downstream_speed_is_40_l681_681960

variable (Vu : ℝ) (Vs : ℝ) (Vd : ℝ)

theorem downstream_speed_is_40 (h1 : Vu = 26) (h2 : Vs = 33) :
  Vd = 40 :=
by
  sorry

end downstream_speed_is_40_l681_681960


namespace area_of_three_presentable_set_l681_681342

def is_three_presentable (z : ℂ) : Prop :=
  ∃ w : ℂ, abs w = 3 ∧ z = w - (1 / w)

def three_presentable_set : set ℂ := {z | is_three_presentable z}

theorem area_of_three_presentable_set :
  area (three_presentable_set) = (80 / 9) * real.pi :=
sorry

end area_of_three_presentable_set_l681_681342


namespace largest_possible_sum_l681_681054

-- Define whole numbers
def whole_numbers : Set ℕ := Set.univ

-- Define the given conditions
variables (a b : ℕ)
axiom h1 : a ∈ whole_numbers
axiom h2 : b ∈ whole_numbers
axiom h3 : a * b = 48

-- Prove the largest sum condition
theorem largest_possible_sum : a + b ≤ 49 :=
sorry

end largest_possible_sum_l681_681054


namespace triangle_angles_l681_681755

-- Define the given angles and equality constraints for points on the triangle.
variables {A B C D E F : Point}
variables (AB AC BC CE CD BF BD : ℝ)

-- Define conditions given in the problem.
def is_isosceles_triangle (a b c : Point) (ab ac : ℝ) : Prop :=
ab = ac ∧ ∃ α : ℝ, ∠(b, a, c) = α

def side_points_conditions (d e f b c a : Point) (ce cd bf bd : ℝ) : Prop :=
(ce = cd) ∧ (bf = bd)

-- Define the question and assert the expected answer.
theorem triangle_angles (A B C D E F : Point)
  (h1 : is_isosceles_triangle A B C AB AC)
  (h2 : ∠(B, A, C) = 70)
  (h3 : side_points_conditions D E F B C A CE CD BF BD)
  (h4 : E ∈ line_segment A B)
  : ∠(E, D, F) = 55 := 
sorry

end triangle_angles_l681_681755


namespace cooper_needs_1043_bricks_l681_681538

def wall1_length := 15
def wall1_height := 6
def wall1_depth := 3

def wall2_length := 20
def wall2_height := 4
def wall2_depth := 2

def wall3_length := 25
def wall3_height := 5
def wall3_depth := 3

def wall4_length := 17
def wall4_height := 7
def wall4_depth := 2

def bricks_needed_for_wall (length height depth: Nat) : Nat :=
  length * height * depth

def total_bricks_needed : Nat :=
  bricks_needed_for_wall wall1_length wall1_height wall1_depth +
  bricks_needed_for_wall wall2_length wall2_height wall2_depth +
  bricks_needed_for_wall wall3_length wall3_height wall3_depth +
  bricks_needed_for_wall wall4_length wall4_height wall4_depth

theorem cooper_needs_1043_bricks : total_bricks_needed = 1043 := by
  sorry

end cooper_needs_1043_bricks_l681_681538


namespace binary_representation_of_41_l681_681163

def decimal_to_binary (n : ℕ) : ℕ :=
match n with
| 0     => 0
| _     => decimal_to_binary (n / 2) * 10 + (n % 2)

theorem binary_representation_of_41 :
  decimal_to_binary 41 = 101001 :=
by
  sorry

end binary_representation_of_41_l681_681163


namespace possible_arrangements_l681_681398

-- Define the three girls: Anya, Sanya, and Tanya
inductive Girl | A | S | T deriving DecidableEq

open Girl

-- Define the proof goal
theorem possible_arrangements : 
  {list.permutations [A, S, T]} = 
  { [[A, S, T], [A, T, S], [S, A, T], [S, T, A], [T, A, S], [T, S, A]] } :=
by
  sorry

end possible_arrangements_l681_681398


namespace angle_bisector_DC_l681_681482

-- Definitions of the circles and points
variables (α : Type*) [euclidean_space α] 
variables (O C D A B : α)
variables (S1 S2 : set α)

-- Conditions translation
variable (OC_chord : is_chord OC S1)
variable (S2_intersects_OC_at_D : S2 ∩ (line_segment O C) = {O, D})
variable (D_neq_C : D ≠ C)
variable (S2_intersects_S1_at_AB : S2 ∩ S1 = {A, B})

-- Theorem statement
theorem angle_bisector_DC (OC_chord : is_chord OC S1)
  (S2_intersects_OC_at_D : S2 ∩ (line_segment O C) = {O, D})
  (D_neq_C : D ≠ C)
  (S2_intersects_S1_at_AB : S2 ∩ S1 = {A, B}) :
  is_angle_bisector (line_segment D C) (angle_at A C B) :=
by sorry

end angle_bisector_DC_l681_681482


namespace parallelogram_area_l681_681311

-- Defining the vectors u and z
def u : ℝ × ℝ := (4, -1)
def z : ℝ × ℝ := (9, -3)

-- Computing the area of parallelogram formed by vectors u and z
def area_parallelogram (u z : ℝ × ℝ) : ℝ :=
  abs (u.1 * (z.2 + u.2) - u.2 * (z.1 + u.1))

-- Lean statement asserting that the area of the parallelogram is 3
theorem parallelogram_area : area_parallelogram u z = 3 := by
  sorry

end parallelogram_area_l681_681311


namespace min_questions_to_determine_product_50_numbers_l681_681851

/-- Prove that to uniquely determine the product of 50 numbers each either +1 or -1 
arranged on the circumference of a circle by asking for the product of three 
consecutive numbers, one must ask a minimum of 50 questions. -/
theorem min_questions_to_determine_product_50_numbers : 
  ∀ (a : ℕ → ℤ), (∀ i, a i = 1 ∨ a i = -1) → 
  (∀ i, ∃ b : ℤ, b = a i * a (i+1) * a (i+2)) → 
  ∃ n, n = 50 :=
by
  sorry

end min_questions_to_determine_product_50_numbers_l681_681851


namespace triangle_min_perimeter_l681_681613

theorem triangle_min_perimeter {A B C D : Type*} 
  (h_int : ∀ u : ℝ, u ∈ {A, B, C} → ∃ k : ℤ, u = k)
  (h_angle_bisect : BD ∥= \(angle A B C\))
  (h_AD : ∥AD∥ = 4)
  (h_DC : ∥DC∥ = 6)
  (h_D_on_AC : D ∈ segment(A, C)) :
  ∃ (P : ℝ), minimum_possible_perimeter(ABC) P := 
begin
  use 25,
  sorry
end

end triangle_min_perimeter_l681_681613


namespace greatest_3_digit_base8_num_div_by_7_eq_511_l681_681920

noncomputable def greatest_base8_number_divisible_by_7 : ℕ := 7 * 73

theorem greatest_3_digit_base8_num_div_by_7_eq_511 : 
  greatest_base8_number_divisible_by_7 = 511 :=
by 
  sorry

end greatest_3_digit_base8_num_div_by_7_eq_511_l681_681920


namespace decreasing_on_interval_l681_681659

noncomputable def f (a x : ℝ) : ℝ := 2^(x * (x - a))

theorem decreasing_on_interval (a : ℝ) : (a ≥ 2) ↔ ∀ x ∈ Set.Ioo 0 1, (deriv (λ x, 2^(x * (x - a)))) x ≤ 0 :=
sorry

end decreasing_on_interval_l681_681659


namespace half_abs_diff_squares_l681_681023

theorem half_abs_diff_squares (a b : ℕ) (ha : a = 15) (hb : b = 13) : (abs (a^2 - b^2)) / 2 = 28 := by
  sorry

end half_abs_diff_squares_l681_681023


namespace packets_of_tomato_seeds_l681_681957

theorem packets_of_tomato_seeds(
  (c_p : ℝ) (c_t : ℝ) (c_c : ℝ) (n_p : ℕ) (n_c : ℕ) (S : ℝ) (n_t: ℕ)) 
  (h1 : c_p = 2.50) 
  (h2 : c_t = 1.50) 
  (h3 : c_c = 0.90) 
  (h4 : n_p = 3)
  (h5 : n_c = 5)
  (h6 : S = 18) 
  : n_t = 4 := 
by 
  sorry

end packets_of_tomato_seeds_l681_681957


namespace radius_of_sphere_l681_681252

theorem radius_of_sphere (R : ℝ) (shots_count : ℕ) (shot_radius : ℝ) :
  shots_count = 125 →
  shot_radius = 1 →
  (shots_count : ℝ) * (4 / 3 * Real.pi * shot_radius^3) = 4 / 3 * Real.pi * R^3 →
  R = 5 :=
by
  intros h1 h2 h3
  sorry

end radius_of_sphere_l681_681252


namespace sequence_condition_l681_681051

noncomputable def sequence (m : ℕ) (a1 : ℕ) : ℕ → ℕ :=
  λ n, if n = 0 then a1
       else if (sequence m a1 (n-1)) < 2^m then (sequence m a1 (n-1))^2 + 2^m
       else (sequence m a1 (n-1)) / 2

theorem sequence_condition (m : ℕ) (a1 : ℕ) (h1 : a1 > 0)
  (h2 : ∀ n, (sequence m a1 n) > 0 ∧ (sequence m a1 (n+1) = if (sequence m a1 n) < 2^m then (sequence m a1 n)^2 + 2^m else (sequence m a1 n) / 2))
  : 
  (m = 2) ∧ ∃ (ℓ : ℕ), ℓ ≥ 1 ∧ ∀ k n: ℕ, n > 0 → (sequence 2 (2^ℓ) k) = 2^{n-1} ∨ (sequence 2 (2^ℓ) k) = 4 * 2^{n-2} := 
sorry

end sequence_condition_l681_681051


namespace decreasing_interval_range_l681_681646

theorem decreasing_interval_range (a : ℝ) :
  (∀ x y ∈ Ioo 0 1, x < y → 2^(x * (x-a)) > 2^(y * (y-a))) ↔ a ≥ 2 :=
by
  sorry

end decreasing_interval_range_l681_681646


namespace recurring_fraction_difference_l681_681114

theorem recurring_fraction_difference :
  let x := (36 / 99 : ℚ)
  let y := (36 / 100 : ℚ)
  x - y = (1 / 275 : ℚ) :=
by
  sorry

end recurring_fraction_difference_l681_681114


namespace sum_arithmetic_sequence_l681_681115

theorem sum_arithmetic_sequence : ∀ (a d l : ℕ), 
  (d = 2) → (a = 2) → (l = 20) → 
  ∃ (n : ℕ), (l = a + (n - 1) * d) ∧ 
  (∑ k in Finset.range n, (a + k * d)) = 110 :=
by
  intros a d l h_d h_a h_l
  use 10
  split
  · sorry
  · sorry

end sum_arithmetic_sequence_l681_681115


namespace grill_runtime_l681_681430

theorem grill_runtime
    (burn_rate : ℕ)
    (burn_time : ℕ)
    (bags : ℕ)
    (coals_per_bag : ℕ)
    (total_burnt_coals : ℕ)
    (total_time : ℕ)
    (h1 : burn_rate = 15)
    (h2 : burn_time = 20)
    (h3 : bags = 3)
    (h4 : coals_per_bag = 60)
    (h5 : total_burnt_coals = bags * coals_per_bag)
    (h6 : total_time = (total_burnt_coals / burn_rate) * burn_time) :
    total_time = 240 :=
by sorry

end grill_runtime_l681_681430


namespace baker_cakes_final_count_l681_681107

theorem baker_cakes_final_count :
  let original_cakes := 121
  let sold_percent := 0.75
  let bought_percent := 1.5
  let cakes_sold := Int.floor (sold_percent * original_cakes)
  let cakes_left := original_cakes - cakes_sold
  let cakes_bought := Int.floor (bought_percent * original_cakes)
  let total_cakes := cakes_left + cakes_bought
  in total_cakes = 212 :=
by
  let original_cakes := 121
  let sold_percent := 0.75
  let bought_percent := 1.5
  let cakes_sold := Int.floor (sold_percent * original_cakes)
  let cakes_left := original_cakes - cakes_sold
  let cakes_bought := Int.floor (bought_percent * original_cakes)
  let total_cakes := cakes_left + cakes_bought
  show total_cakes = 212
  sorry

end baker_cakes_final_count_l681_681107


namespace total_fruit_count_l681_681330

theorem total_fruit_count :
  let gerald_apple_bags := 5
  let gerald_orange_bags := 4
  let apples_per_gerald_bag := 30
  let oranges_per_gerald_bag := 25
  let pam_apple_bags := 6
  let pam_orange_bags := 4
  let sue_apple_bags := 2 * gerald_apple_bags
  let sue_orange_bags := gerald_orange_bags / 2
  let apples_per_sue_bag := apples_per_gerald_bag - 10
  let oranges_per_sue_bag := oranges_per_gerald_bag + 5
  
  let gerald_apples := gerald_apple_bags * apples_per_gerald_bag
  let gerald_oranges := gerald_orange_bags * oranges_per_gerald_bag
  
  let pam_apples := pam_apple_bags * (3 * apples_per_gerald_bag)
  let pam_oranges := pam_orange_bags * (2 * oranges_per_gerald_bag)
  
  let sue_apples := sue_apple_bags * apples_per_sue_bag
  let sue_oranges := sue_orange_bags * oranges_per_sue_bag

  let total_apples := gerald_apples + pam_apples + sue_apples
  let total_oranges := gerald_oranges + pam_oranges + sue_oranges
  total_apples + total_oranges = 1250 :=

by
  sorry

end total_fruit_count_l681_681330


namespace percentage_selected_B_l681_681272

-- Definitions for the problem conditions
def selected_A : ℕ := (0.06 * 8000).to_nat
def appeared_A : ℕ := 8000
def appeared_B : ℕ := 8000
def selected_B : ℕ := selected_A + 80

-- The proof statement
theorem percentage_selected_B (h : appeared_A = appeared_B) : 
  ((selected_B : ℝ) / (appeared_B : ℝ)) * 100 = 7 :=
by
  sorry

end percentage_selected_B_l681_681272


namespace arthur_walked_total_distance_l681_681100

-- Define the total number of blocks walked.
def total_blocks_walked (blocks_west : ℕ) (blocks_south : ℕ) : ℕ :=
  blocks_west + blocks_south

-- Define the distance per block.
def distance_per_block_miles : ℝ := 1 / 4

-- Calculate the total distance walked.
noncomputable def total_distance_walked (blocks_west blocks_south : ℕ) : ℝ :=
  total_blocks_walked blocks_west blocks_south * distance_per_block_miles

-- Prove the total distance walked by Arthur.
theorem arthur_walked_total_distance (blocks_west blocks_south : ℕ) (h_blocks_west : blocks_west = 8)
(h_blocks_south : blocks_south = 10) : total_distance_walked blocks_west blocks_south = 4.5 :=
by
  -- Given specific values for blocks_west and blocks_south, plug in these values.
  rw [h_blocks_west, h_blocks_south]
  -- Evaluate the total distance walked.
  show total_distance_walked 8 10 = 4.5
  -- Using definition of total_distance_walked and total_blocks_walked
  sorry

end arthur_walked_total_distance_l681_681100


namespace find_x_intervals_l681_681565

theorem find_x_intervals :
  {x : ℝ | x^3 - x^2 + 11*x - 42 < 0} = { x | -2 < x ∧ x < 3 ∨ 3 < x ∧ x < 7 } :=
by sorry

end find_x_intervals_l681_681565


namespace calculate_fraction_6_5_general_form_calculate_sum_l681_681989

-- Question (1): Calculate \(\frac{1}{{\sqrt{6}+\sqrt{5}}} = \sqrt{6} - \sqrt{5}\)
theorem calculate_fraction_6_5 : 
  1 / (Real.sqrt 6 + Real.sqrt 5) = Real.sqrt 6 - Real.sqrt 5 :=
sorry

-- Question (2): Find the general form \(\frac{1}{{\sqrt{n+1}+\sqrt{n}}} = \sqrt{n+1} - \sqrt{n}\)
theorem general_form (n : ℕ) (hn : n ≥ 1) : 
  1 / (Real.sqrt (n + 1) + Real.sqrt n) = Real.sqrt (n + 1) - Real.sqrt n :=
sorry

-- Question (3): Calculate 
-- \(\left(\sum_{{k=2}}^{{2024}} \frac{1}{{\sqrt{k}+\sqrt{k-1}}}\right) (\sqrt{2024}+1) = 2023\)
theorem calculate_sum : 
  (\sum k in Finset.range 2023 + 1 | k ≥ 2, 1 / (Real.sqrt k + Real.sqrt (k - 1))) * (Real.sqrt 2024 + 1) = 2023 :=
sorry

end calculate_fraction_6_5_general_form_calculate_sum_l681_681989


namespace pyramid_height_eq_3_75_l681_681444

-- Define the edge length of the cube
def cube_edge_length : ℝ := 5

-- Define the base edge length of the pyramid
def pyramid_base_edge_length : ℝ := 10

-- Define the volume of the cube
def V_cube : ℝ := cube_edge_length ^ 3

-- Define the volume of the pyramid
def V_pyramid (h : ℝ) : ℝ := (1 / 3) * (pyramid_base_edge_length ^ 2) * h

-- Proof that the height of the pyramid is 3.75 units
theorem pyramid_height_eq_3_75 :
  ∃ h : ℝ, V_cube = V_pyramid h ∧ h = 3.75 :=
by
  use 3.75
  split
  . sorry
  . rfl

end pyramid_height_eq_3_75_l681_681444


namespace factory_needs_at_least_10_workers_for_profit_l681_681447

noncomputable def min_workers_to_profit
  (fixed_cost : ℕ) (wage_per_hour : ℕ) (work_hours : ℕ)
  (production_per_hour : ℕ) (sale_price : ℕ → ℝ) : ℕ :=
inf { n : ℕ | let daily_wage := wage_per_hour * work_hours,
                  total_daily_cost := fixed_cost + daily_wage * n,
                  daily_output := production_per_hour * work_hours * n,
                  daily_revenue := (sale_price (production_per_hour * work_hours)) * n
              in total_daily_cost < daily_revenue }

theorem factory_needs_at_least_10_workers_for_profit :
  min_workers_to_profit 600 20 9 6 (λ x, 4.5) = 10 :=
by
  sorry

end factory_needs_at_least_10_workers_for_profit_l681_681447


namespace min_f_over_f_prime_at_1_l681_681807

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def quadratic_derivative (a b x : ℝ) : ℝ := 2 * a * x + b

theorem min_f_over_f_prime_at_1 (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b > 0) (h₂ : ∀ x, quadratic_function a b c x ≥ 0) :
  (∃ k, (∀ x, quadratic_function a b c x ≥ 0 → quadratic_function a b c ((-b)/(2*a)) ≤ x) ∧ k = 2) :=
by
  sorry

end min_f_over_f_prime_at_1_l681_681807


namespace find_a45_l681_681612

theorem find_a45 :
  ∃ (a : ℕ → ℝ), 
    a 0 = 11 ∧ a 1 = 11 ∧ 
    (∀ m n : ℕ, a (m + n) = (1/2) * (a (2 * m) + a (2 * n)) - (m - n)^2) ∧ 
    a 45 = 1991 := by
  sorry

end find_a45_l681_681612


namespace total_passengers_l681_681719

theorem total_passengers (P : ℕ) 
  (h1 : P = (1/12 : ℚ) * P + (1/4 : ℚ) * P + (1/9 : ℚ) * P + (1/6 : ℚ) * P + 42) :
  P = 108 :=
sorry

end total_passengers_l681_681719


namespace find_triples_l681_681569

theorem find_triples (x y n : ℕ) (hx : x > 0) (hy : y > 0) (hn : n > 0) :
  (x! + y!) / n! = (3:ℕ)^n ↔ (x = 2 ∧ y = 1 ∧ n = 1) ∨ (x = 1 ∧ y = 2 ∧ n = 1) :=
by
  sorry

end find_triples_l681_681569


namespace net_calorie_deficit_l681_681151

-- Define the conditions as constants.
def total_distance : ℕ := 3
def calories_burned_per_mile : ℕ := 150
def calories_in_candy_bar : ℕ := 200

-- Prove the net calorie deficit.
theorem net_calorie_deficit : total_distance * calories_burned_per_mile - calories_in_candy_bar = 250 := by
  sorry

end net_calorie_deficit_l681_681151


namespace quadratic_inequality_solution_l681_681002

theorem quadratic_inequality_solution :
  {x : ℝ | 2*x^2 - 3*x - 2 ≥ 0} = {x : ℝ | x ≤ -1/2 ∨ x ≥ 2} :=
sorry

end quadratic_inequality_solution_l681_681002


namespace find_edge_BD_l681_681472

-- Given conditions
def AB : ℝ := 9
def BC : ℝ := 13
def angle_ADC : ℝ := Real.pi / 3  -- 60 degrees in radians

-- Since AD = BC and CD = AB due to symmetry of the pyramid with acute-angled faces
def AD : ℝ := BC
def CD : ℝ := AB

-- Proof to find BD
theorem find_edge_BD : BD = Real.sqrt (AD^2 + CD^2 - 2 * AD * CD * Real.cos angle_ADC) :=
by
  have hBD_squared : BD^2 = AD^2 + CD^2 - 2 * AD * CD * Real.cos angle_ADC := sorry
  have hBD_sqrt : BD = Real.sqrt (AD^2 + CD^2 - 2 * AD * CD * Real.cos angle_ADC) := sorry
  exact hBD_sqrt

end find_edge_BD_l681_681472


namespace value_of_a_4_l681_681160

theorem value_of_a_4 :
  ∃ a : ℕ → ℕ, 
  a 1 = 2 ∧ 
  (∀ n : ℕ, n > 0 → a (n + 1) = (∑ i in finset.range n, a (i + 1)) + 3 ^ n) ∧ 
  a 4 = 50 :=
begin
  sorry
end

end value_of_a_4_l681_681160


namespace distance_between_A_and_B_l681_681969

theorem distance_between_A_and_B 
  (v1 v2: ℝ) (s: ℝ)
  (h1 : (s - 8) / v1 = s / v2)
  (h2 : s / (2 * v1) = (s - 15) / v2)
  (h3: s = 40) : 
  s = 40 := 
sorry

end distance_between_A_and_B_l681_681969


namespace volume_of_the_solid_l681_681389

-- The given conditions
variables (t : ℝ) (ht : t = 4 * sqrt 3)

-- Definition for the length of the upper edge
def upper_edge_length : ℝ := 3 * t

-- Definition for the volume of a regular tetrahedron
def tetrahedron_volume (S : ℝ) : ℝ := (sqrt 2 * S^3) / 12

-- Definition for the larger tetrahedron's side length
def larger_tetrahedron_side_length : ℝ := 3 * t

-- The original problem asks for the volume of the solid
theorem volume_of_the_solid : tetrahedron_volume (larger_tetrahedron_side_length t) / 2 = 216 * sqrt 6 :=
by
  sorry

end volume_of_the_solid_l681_681389


namespace floor_of_4point7_l681_681547

theorem floor_of_4point7 : (Real.floor 4.7) = 4 :=
by
  sorry

end floor_of_4point7_l681_681547


namespace greatest_div_by_seven_base_eight_l681_681897

theorem greatest_div_by_seven_base_eight : ∃ n : ℕ, 
  (n < 512) ∧ (Divisibility.divides 7 n) ∧ 
  (∀ m : ℕ, (m < 512) → (Divisibility.divides 7 m) → m ≤ n) ∧ 
  nat.to_digits 8 n = [7, 7, 4] := 
sorry

end greatest_div_by_seven_base_eight_l681_681897


namespace perpendicular_slope_l681_681540

theorem perpendicular_slope (x y : ℝ) (h : 5 * x - 4 * y = 20) : 
  ∃ m : ℝ, m = -4 / 5 :=
sorry

end perpendicular_slope_l681_681540


namespace probability_of_product_multiple_of_4_is_2_5_l681_681709

open Finset BigOperators

def all_pairs (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s \ s.diag

def num_favorable_pairs (s : Finset ℕ) : ℕ :=
  (all_pairs s).filter (λ p => (p.1 * p.2) % 4 = 0).card

def probability_multiple_of_4 : ℚ :=
  let s := (finset.range 7).filter (λ n => n ≠ 0)
  let total_pairs := (all_pairs s).card
  let favorable_pairs := num_favorable_pairs s
  favorable_pairs / total_pairs

theorem probability_of_product_multiple_of_4_is_2_5 :
  probability_multiple_of_4 = 2 / 5 := by
  -- skipping the proof
  sorry

end probability_of_product_multiple_of_4_is_2_5_l681_681709


namespace floor_of_4point7_l681_681550

theorem floor_of_4point7 : (Real.floor 4.7) = 4 :=
by
  sorry

end floor_of_4point7_l681_681550


namespace product_probability_correct_l681_681820

/-- Define probabilities for spins of Paco and Dani --/
def prob_paco := 1 / 5
def prob_dani := 1 / 15

/-- Define the probability that the product of spins is less than 30 --/
def prob_product_less_than_30 : ℚ :=
  (2 / 5) + (1 / 5) * (9 / 15) + (1 / 5) * (7 / 15) + (1 / 5) * (5 / 15)

theorem product_probability_correct : prob_product_less_than_30 = 17 / 25 :=
by sorry

end product_probability_correct_l681_681820


namespace driver_net_pay_rate_is_25_l681_681446

noncomputable def distance (time: ℕ) (speed: ℕ) : ℕ := 
  time * speed

noncomputable def gasoline_used (distance: ℕ) (efficiency: ℕ) : ℕ := 
  distance / efficiency

noncomputable def earnings (distance: ℕ) (rate: ℕ) : ℕ := 
  distance * rate

noncomputable def gas_expense (gasoline: ℕ) (price: ℕ) : ℕ := 
  gasoline * price

noncomputable def net_earnings (earnings: ℕ) (expense: ℕ) : ℕ :=
  earnings - expense

noncomputable def net_pay_rate_per_hour (net_earnings: ℕ) (time: ℕ) : ℕ :=
  net_earnings / time

theorem driver_net_pay_rate_is_25 
    (time : ℕ) 
    (speed : ℕ) 
    (efficiency : ℕ) 
    (pay_rate : ℕ) 
    (gas_price : ℕ) : net_pay_rate_per_hour 
        (net_earnings 
          (earnings (distance time speed) pay_rate) 
          (gas_expense (gasoline_used (distance time speed) efficiency) gas_price)) 
        time = 25 := 
by 
  -- Given
  let time := 3
  let speed := 50
  let efficiency := 25
  let pay_rate := 60  -- representing $0.60 in cents for easier computation
  let gas_price := 250 -- representing $2.50 in cents for easier computation
  -- Calculate the values
  let d := distance time speed
  let g := gasoline_used d efficiency
  let e := earnings d pay_rate
  let c := gas_expense g gas_price
  let n := net_earnings e c
  let p := net_pay_rate_per_hour n time
  -- Show the result
  show p = 25 
  sorry

end driver_net_pay_rate_is_25_l681_681446


namespace distance_of_point_P_to_origin_l681_681621

noncomputable def dist_to_origin (P : ℝ × ℝ) : ℝ :=
  Real.sqrt (P.1 ^ 2 + P.2 ^ 2)

theorem distance_of_point_P_to_origin :
  let F1 := (-Real.sqrt 2, 0)
  let F2 := (Real.sqrt 2, 0)
  let y_P := 1 / 2
  ∃ x_P : ℝ, (x_P, y_P) = P ∧
    (dist_to_origin P = Real.sqrt 6 / 2) :=
by
  sorry

end distance_of_point_P_to_origin_l681_681621


namespace convex_quadrilateral_inequality_l681_681209

variable (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]

theorem convex_quadrilateral_inequality
    (AB CD BC AD AC BD : ℝ)
    (h : AB * CD + BC * AD >= AC * BD)
    (convex_quadrilateral : Prop) :
  AB * CD + BC * AD >= AC * BD :=
by
  sorry

end convex_quadrilateral_inequality_l681_681209


namespace probability_real_part_greater_imag_part_l681_681011

noncomputable def probability_real_gt_imag (x y : Fin 6) : ℚ :=
  if (x + 1 : ℕ) > (y + 1 : ℕ) then 1 else 0

theorem probability_real_part_greater_imag_part :
  let outcomes := Fin 6 × Fin 6;
  let favorable_outcomes := (outcomes.filter $ λ (x y : Fin 6), (x.val + 1) > (y.val + 1));
  let probability := (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ)
  probability = 5 / 12 :=
by
  sorry

end probability_real_part_greater_imag_part_l681_681011


namespace pyramid_height_eq_3_75_l681_681443

-- Define the edge length of the cube
def cube_edge_length : ℝ := 5

-- Define the base edge length of the pyramid
def pyramid_base_edge_length : ℝ := 10

-- Define the volume of the cube
def V_cube : ℝ := cube_edge_length ^ 3

-- Define the volume of the pyramid
def V_pyramid (h : ℝ) : ℝ := (1 / 3) * (pyramid_base_edge_length ^ 2) * h

-- Proof that the height of the pyramid is 3.75 units
theorem pyramid_height_eq_3_75 :
  ∃ h : ℝ, V_cube = V_pyramid h ∧ h = 3.75 :=
by
  use 3.75
  split
  . sorry
  . rfl

end pyramid_height_eq_3_75_l681_681443


namespace principal_arg_conjugate_l681_681226

noncomputable def z (θ : ℝ) : ℂ := 1 - Real.sin θ + Complex.i * Real.cos θ

theorem principal_arg_conjugate (θ : ℝ) (hθ : 0 < θ ∧ θ < π) :
  Complex.arg (Complex.conj (z θ)) = -θ / 2 :=
sorry

end principal_arg_conjugate_l681_681226


namespace fiona_shirt_number_l681_681399

def is_two_digit_prime (n : ℕ) : Prop := 
  (n ≥ 10 ∧ n < 100 ∧ Nat.Prime n)

theorem fiona_shirt_number (d e f : ℕ) 
  (h1 : is_two_digit_prime d)
  (h2 : is_two_digit_prime e)
  (h3 : is_two_digit_prime f)
  (h4 : e + f = 36)
  (h5 : d + e = 30)
  (h6 : d + f = 32) : 
  f = 19 := 
sorry

end fiona_shirt_number_l681_681399


namespace floor_47_l681_681554

theorem floor_47 : Int.floor 4.7 = 4 :=
by
  sorry

end floor_47_l681_681554


namespace number_of_customers_l681_681979

theorem number_of_customers 
    (boxes_opened : ℕ) 
    (samples_per_box : ℕ) 
    (samples_left_over : ℕ) 
    (samples_limit_per_person : ℕ)
    (h1 : boxes_opened = 12)
    (h2 : samples_per_box = 20)
    (h3 : samples_left_over = 5)
    (h4 : samples_limit_per_person = 1) : 
    ∃ customers : ℕ, customers = (boxes_opened * samples_per_box) - samples_left_over ∧ customers = 235 :=
by {
  sorry
}

end number_of_customers_l681_681979


namespace height_of_conical_cup_l681_681434

def volume_of_cone (r : ℝ) (h : ℝ) : ℝ :=
  (1/3) * Real.pi * r^2 * h

theorem height_of_conical_cup (h : ℝ) (r : ℝ = 4) (V : ℝ = 150) : h = 9 :=
  by
  have V : ℝ := 150
  have r : ℝ := 4
  sorry

end height_of_conical_cup_l681_681434


namespace arithmetic_sequence_sum_l681_681140

-- Define the arithmetic sequence properties
def is_arithmetic_sequence (seq : ℕ → ℕ) :=
  ∀ n : ℕ, seq (n + 1) = seq n + 2

-- Define the arithmetic sequence in question
def sequence : ℕ → ℕ
| 0       := 2
| (n + 1) := sequence n + 2

-- Check that our sequence matches the properties of an arithmetic sequence
lemma sequence_is_arithmetic : is_arithmetic_sequence sequence :=
by intros n; simp [sequence]

-- Define the sum of the first n terms of the sequence
def sum_n_terms (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, sequence i

-- State the main theorem to be proven: the sum of the first 10 terms is 110
theorem arithmetic_sequence_sum : sum_n_terms 10 = 110 :=
sorry

end arithmetic_sequence_sum_l681_681140


namespace shaded_area_square_l681_681288

-- Define the square with side length 4 units
def square_ABCD : Prop :=
  ∃ A B C D : ℝ × ℝ,
  ((B.1 - A.1) * (B.1 - A.1) + (B.2 - A.2) * (B.2 - A.2) = 16) ∧
  ((C.1 - B.1) * (C.1 - B.1) + (C.2 - B.2) * (C.2 - B.2) = 16) ∧
  ((D.1 - C.1) * (D.1 - C.1) + (D.2 - C.2) * (D.2 - C.2) = 16) ∧
  ((A.1 - D.1) * (A.1 - D.1) + (A.2 - D.2) * (A.2 - D.2) = 16) ∧
  ((B.1 - C.1) * (A.1 - D.1) + (B.2 - C.2) * (A.2 - D.2) = 0) ∧      -- Perpendicular sides

-- Define the midpoints P, Q, R, S
def midpoints_PQRS (A B C D P Q R S : ℝ × ℝ) : Prop :=
  (P = (A + B) / 2) ∧
  (Q = (B + C) / 2) ∧
  (R = (C + D) / 2) ∧
  (S = (D + A) / 2)

-- Prove the area of the shaded region is 12 units
theorem shaded_area_square : square_ABCD ∧
  ∀ (P Q R S : ℝ × ℝ), midpoints_PQRS A B C D P Q R S →
  area_shaded_region A B C D P Q R S = 12 :=
sorry


end shaded_area_square_l681_681288


namespace binom_18_6_eq_4765_l681_681528

def binom (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_18_6_eq_4765 : binom 18 6 = 4765 := by
  sorry

end binom_18_6_eq_4765_l681_681528


namespace smallest_x_y_sum_l681_681212

theorem smallest_x_y_sum (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ y)
                        (h4 : (1 / (x : ℝ)) + (1 / (y : ℝ)) = (1 / 20)) :
    x + y = 81 :=
sorry

end smallest_x_y_sum_l681_681212


namespace exist_infinite_quadratic_permutation_no_cubic_permutation_l681_681946

def is_perfect_square (x : ℕ) : Prop :=
  ∃ y : ℕ, y^2 = x

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ y : ℕ, y^3 = x

def quadratic_permutation (n : ℕ) (a : Fin n → Fin n.succ) : Prop :=
  ∀ i : Fin n.pred, is_perfect_square (a i * a ⟨i + 1, Nat.lt_of_lt_pred i.is_lt⟩ + 1)

def cubic_permutation (n : ℕ) (a : Fin n → Fin n.succ) : Prop :=
  ∀ i : Fin n.pred, is_perfect_cube (a i * a ⟨i + 1, Nat.lt_of_lt_pred i.is_lt⟩ + 1)

theorem exist_infinite_quadratic_permutation :
  ∃∞ n : ℕ, ∃ a : Fin n → Fin n.succ, quadratic_permutation n a := sorry

theorem no_cubic_permutation (n : ℕ) :
  ∀ a : Fin n → Fin n.succ, ¬ cubic_permutation n a := sorry

end exist_infinite_quadratic_permutation_no_cubic_permutation_l681_681946


namespace sum_series_evaluation_l681_681185

theorem sum_series_evaluation :
  (∑ n in Finset.range 99 \ Finset.range 2, 1 / ((3 * n.succ - 2) * (3 * n.succ + 1))) = 99 / 1204 :=
by
  sorry

end sum_series_evaluation_l681_681185


namespace sum_of_all_possible_radii_l681_681433

noncomputable def sum_of_radii : ℝ :=
let r₁ := 6 + 2 * Real.sqrt 6 in
let r₂ := 6 - 2 * Real.sqrt 6 in
r₁ + r₂

theorem sum_of_all_possible_radii :
  ∃ r₁ r₂ : ℝ, 
    (r₁ = 6 + 2 * Real.sqrt 6 ∧ r₂ = 6 - 2 * Real.sqrt 6) ∧ 
    r₁ + r₂ = 12 := 
by {
  use [6 + 2 * Real.sqrt 6, 6 - 2 * Real.sqrt 6],
  split,
  {
    split; refl
  },
  {
    rw [←add_assoc, show 2 * Real.sqrt 6 + -2 * Real.sqrt 6 = 0, from add_neg_self _],
    refl
  }
}

end sum_of_all_possible_radii_l681_681433


namespace sqrt_sum_l681_681925

theorem sqrt_sum (a b : ℝ) (ha : a = 24 - 10 * Real.sqrt 5) (hb : b = 24 + 10 * Real.sqrt 5) :
  Real.sqrt a + Real.sqrt b = 2 * Real.sqrt 19 :=
by
  have hx : Real.sqrt (24 - 10 * Real.sqrt 5) + Real.sqrt (24 + 10 * Real.sqrt 5) = 2 * Real.sqrt 19 := by sorry
  exact hx

end sqrt_sum_l681_681925


namespace original_savings_l681_681412

theorem original_savings (tv_cost : ℚ) (fraction_on_tv : ℚ) (original_savings : ℚ) :
  (tv_cost = 220) → (fraction_on_tv = 1 / 4) → (original_savings * fraction_on_tv = tv_cost) →
  original_savings = 880 :=
by
  intros h_tv_cost h_fraction_on_tv h_equal
  sorry

end original_savings_l681_681412


namespace floor_47_l681_681551

theorem floor_47 : Int.floor 4.7 = 4 :=
by
  sorry

end floor_47_l681_681551


namespace find_integer_b_l681_681370

theorem find_integer_b (z : ℝ) : ∃ b : ℝ, (z^2 - 6*z + 17 = (z - 3)^2 + b) ∧ b = 8 :=
by
  -- The proof would go here
  sorry

end find_integer_b_l681_681370


namespace find_diagonal_length_l681_681738

noncomputable def length_of_diagonal (m n p : ℝ) : ℝ :=
  120 * Real.sqrt ((4 / (m ^ 2 + p ^ 2 - n ^ 2)) +
                   (4 / (m ^ 2 - p ^ 2 + n ^ 2)) +
                   (4 / (-m ^ 2 + p ^ 2 + n ^ 2)))

theorem find_diagonal_length (m n p : ℝ) (h₁ : m ≠ 0) (h₂ : n ≠ 0) (h₃ : p ≠ 0) :
  (length_of_diagonal m n p) = 120 * Real.sqrt ((4 / (m ^ 2 + p ^ 2 - n ^ 2)) +
                                                (4 / (m ^ 2 - p ^ 2 + n ^ 2)) +
                                                (4 / (-m ^ 2 + p ^ 2 + n ^ 2))) :=
begin
  sorry
end

end find_diagonal_length_l681_681738


namespace sum_arithmetic_seq_l681_681137

theorem sum_arithmetic_seq (a d n : ℕ) :
  a = 2 → d = 2 → a + (n - 1) * d = 20 → (n / 2) * (a + (a + (n - 1) * d)) = 110 :=
by sorry

end sum_arithmetic_seq_l681_681137


namespace percent_students_scoring_in_range_l681_681072

theorem percent_students_scoring_in_range
  (f_90_to_100 : Nat)
  (f_80_to_89 : Nat)
  (f_70_to_79 : Nat)
  (f_60_to_69 : Nat)
  (f_below_60 : Nat)
  (h_90_to_100 : f_90_to_100 = 5)
  (h_80_to_89 : f_80_to_89 = 7)
  (h_70_to_79 : f_70_to_79 = 9)
  (h_60_to_69 : f_60_to_69 = 4)
  (h_below_60 : f_below_60 = 6) :
  let total_students := f_90_to_100 + f_80_to_89 + f_70_to_79 + f_60_to_69 + f_below_60
  let percent_70_to_79 := (f_70_to_79 * 100) / total_students
  percent_70_to_79 = 29.03 := 
by
  sorry

end percent_students_scoring_in_range_l681_681072


namespace max_x_for_lcm_l681_681377

open Nat

-- Define the condition for the least common multiple function for three numbers
def lcm3 (a b c : ℕ) : ℕ := lcm (lcm a b) c

theorem max_x_for_lcm (x : ℕ) : lcm3 x 12 15 = 180 -> x = 180 := by
  sorry

end max_x_for_lcm_l681_681377


namespace domain_of_log2_x2_minus_1_l681_681367

variables {x : ℝ}

def f (x : ℝ) : ℝ := log 2 (x^2 - 1)

def domain_of_f : set ℝ := {x | x < -1 ∨ x > 1}

theorem domain_of_log2_x2_minus_1 : ∀ x : ℝ, (∃ y : ℝ, y = f x) ↔ x ∈ domain_of_f :=
by
  sorry

end domain_of_log2_x2_minus_1_l681_681367


namespace function_fixed_point_l681_681375

theorem function_fixed_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : (a^(-2+2) + 1 = 2) :=
by
  sorry

end function_fixed_point_l681_681375


namespace mean_value_of_quadrilateral_angles_l681_681171

-- Definition of the problem conditions
def number_of_sides (quad : Type) : ℕ := 4

-- Statement: Prove that the mean value of the measures of the four interior angles of any quadrilateral is 90 degrees
theorem mean_value_of_quadrilateral_angles (quad : Type) :
  (number_of_sides quad) = 4 → mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l681_681171


namespace paul_catches_up_with_mary_25_minutes_later_l681_681322

theorem paul_catches_up_with_mary_25_minutes_later :
  ∀ (mary_speed paul_speed : ℝ) (time_interval_in_hours : ℝ),
  mary_speed = 50 →
  paul_speed = 80 →
  time_interval_in_hours = 1 / 4 →
  let mary_distance_ahead := mary_speed * time_interval_in_hours in
  let catch_up_speed := paul_speed - mary_speed in
  let time_to_catch_up_in_hours := mary_distance_ahead / catch_up_speed in
  let time_to_catch_up_in_minutes := time_to_catch_up_in_hours * 60 in
  time_to_catch_up_in_minutes = 25 :=
by intros; sorry

end paul_catches_up_with_mary_25_minutes_later_l681_681322


namespace f_odd_range_k_l681_681633

def f (x : ℝ) : ℝ :=
if x > 0 then x / 3 - 2^x
else if x < 0 then x / 3 + 2^(-x)
else 0

theorem f_odd (x : ℝ) : f (-x) = -f x :=
by
  cases lt_or_gt_of_ne (ne_of_lt_or_gt (lt_trichotomy (-x) (0 : ℝ))) <:
  case inl => simp only [f, if_neg (mt (by intro; linarith) h)]
  case inr => simp only [f, if_neg (by cases h with h h; contradiction)] <:
  sorry

theorem range_k (k : ℝ) : (∀ t : ℝ, f (t^2 - 2 * t) + f (2 * t^2 - k) < 0) → k < -1 / 3 :=
by
  intro h
  have key := ∀ t, f (t^2 - 2 * t) < -f (2 * t^2 - k)
  sorry

end f_odd_range_k_l681_681633


namespace probability_of_product_multiple_of_4_is_2_5_l681_681710

open Finset BigOperators

def all_pairs (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s \ s.diag

def num_favorable_pairs (s : Finset ℕ) : ℕ :=
  (all_pairs s).filter (λ p => (p.1 * p.2) % 4 = 0).card

def probability_multiple_of_4 : ℚ :=
  let s := (finset.range 7).filter (λ n => n ≠ 0)
  let total_pairs := (all_pairs s).card
  let favorable_pairs := num_favorable_pairs s
  favorable_pairs / total_pairs

theorem probability_of_product_multiple_of_4_is_2_5 :
  probability_multiple_of_4 = 2 / 5 := by
  -- skipping the proof
  sorry

end probability_of_product_multiple_of_4_is_2_5_l681_681710


namespace Nora_paid_dimes_l681_681327

theorem Nora_paid_dimes (c : ℕ) (h1 : c = 9) (h2 : 1 * 10 = 10) : 10 * c = 90 := by
  rw [h1]
  norm_num

end Nora_paid_dimes_l681_681327


namespace math_club_probability_l681_681882

open BigOperators

theorem math_club_probability :
  let p := (1/4 : ℚ)
  let prob_club_1 := 1 / (nat.choose 6 3 : ℚ)
  let prob_club_2 := 1 / (nat.choose 9 3 : ℚ)
  let prob_club_3 := 1 / (nat.choose 11 3 : ℚ)
  let prob_club_4 := 1 / (nat.choose 13 3 : ℚ)
  p * (prob_club_1 + prob_club_2 + prob_club_3 + prob_club_4) = (905 / 55440 : ℚ) :=
by
  sorry

end math_club_probability_l681_681882


namespace monotonic_decreasing_condition_l681_681651

theorem monotonic_decreasing_condition {f : ℝ → ℝ} (a : ℝ) :
  (∀ x ∈ Ioo (0:ℝ) 1, (2:ℝ) ^ (x * (x - a)) < (2:ℝ) ^ ((1:ℝ) * (1 - a))) → a ≥ 2 :=
begin
  sorry
end

end monotonic_decreasing_condition_l681_681651


namespace madeline_needs_work_hours_l681_681317

def rent : ℝ := 1200
def groceries : ℝ := 400
def medical_expenses : ℝ := 200
def utilities : ℝ := 60
def emergency_savings : ℝ := 200
def hourly_wage : ℝ := 15

def total_expenses : ℝ := rent + groceries + medical_expenses + utilities + emergency_savings

noncomputable def total_hours_needed : ℝ := total_expenses / hourly_wage

theorem madeline_needs_work_hours :
  ⌈total_hours_needed⌉ = 138 := by
  sorry

end madeline_needs_work_hours_l681_681317


namespace triangle_proof_l681_681732

variable (A B C a b c : ℝ)

-- Define conditions
variable (h1 : 2 * sin (2 * A) + sin (A - B) = sin C)
variable (h2 : A ≠ π / 2)
variable (h3 : c = 2)
variable (h4 : C = π / 3)

-- Proof statements
theorem triangle_proof : (2 * sin (2 * A) + sin (A - B) = sin C) ∧ 
                         (A ≠ π / 2) ∧ 
                         (2 * a = b) ∧ 
                         (c = 2) ∧ 
                         (C = π / 3) → 
                         (a / b = 1 / 2) ∧ 
                         (1 / 2 * a * b * sin C = 2 * sqrt 3 / 3) := by
  intros
  sorry

end triangle_proof_l681_681732


namespace binomial_identity_vandermonde_identity_norlund_identity_l681_681941

-- Definition of falling factorial (x)_n
def falling_factorial (x : ℕ) (n : ℕ) : ℕ :=
  if n == 0 then 1 else x * falling_factorial (x - 1) (n - 1)

-- Definition of rising factorial [x]_n
def rising_factorial (x : ℕ) (n : ℕ) : ℕ :=
  if n == 0 then 1 else x * rising_factorial (x + 1) (n - 1)

-- Binomial identity statement
theorem binomial_identity (x y n : ℕ) : 
  (x + y)^n = ∑ k in finset.range (n + 1), nat.choose n k * x^k * y^(n - k) := 
begin
  sorry
end

-- Vandermonde's identity statement
theorem vandermonde_identity (x y n : ℕ) : 
  falling_factorial (x + y) n = ∑ k in finset.range (n + 1), nat.choose n k * falling_factorial x k * falling_factorial y (n - k) := 
begin
  sorry
end

-- Norlund's identity statement
theorem norlund_identity (x y n : ℕ) : 
  rising_factorial (x + y) n = ∑ k in finset.range (n + 1), nat.choose n k * rising_factorial x k * rising_factorial y (n - k) := 
begin
  sorry
end

end binomial_identity_vandermonde_identity_norlund_identity_l681_681941


namespace combination_18_6_l681_681497

theorem combination_18_6 : (nat.choose 18 6) = 18564 := 
by 
  sorry

end combination_18_6_l681_681497


namespace count_ordered_pairs_l681_681249

theorem count_ordered_pairs : 
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ m ≥ n ∧ m^2 - n^2 = 120 ∧
  (count (λ (p : ℕ × ℕ), p.1 > 0 ∧ p.2 > 0 ∧ p.1 ≥ p.2 ∧ p.1^2 - p.2^2 = 120) 
       (finset.univ.product finset.univ)) = 4 :=
sorry

end count_ordered_pairs_l681_681249


namespace minions_mistake_score_l681_681035

theorem minions_mistake_score :
  (minions_left_phone_on_untrusted_website ∧
   downloaded_file_from_untrusted_source ∧
   guidelines_by_cellular_operators ∧
   avoid_sharing_personal_info ∧
   unverified_files_may_be_harmful ∧
   double_extensions_signify_malicious_software) →
  score = 21 :=
by
  -- Here we would provide the proof steps which we skip with sorry
  sorry

end minions_mistake_score_l681_681035


namespace max_a_value_l681_681782

theorem max_a_value (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 50) :
  a ≤ 2924 :=
by sorry

end max_a_value_l681_681782


namespace equal_lengths_tangent_circles_l681_681415

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Definitions of points and properties
variables {A B C O Q P I J : V}

-- Point A tangent to circle (O) at points B and C
def tangents (A B O : V) : Prop := (B - A) ⬝ (O - B) = 0

-- Q is a point inside angle ∠BAC
def inside_angle (A B C Q : V) : Prop := 
  inner ((B - A) ⬝ (Q - A)) > 0 ∧ inner ((C - A) ⬝ (Q - A)) > 0

-- P is on the ray AQ such that OP ⊥ AQ
def perpendicular (O P A Q : V) : Prop := inner (O - P) (A - Q) = 0

-- OP intersects the circumcircles of triangles BPQ and CPQ at points I and J respectively
def intersects_circumcircles (O P I J B C Q : V) : Prop := 
  dist O I = dist O J

theorem equal_lengths_tangent_circles 
  (tangent_AB : tangents A B O)
  (tangent_AC : tangents A C O)
  (inside_ang : inside_angle A B C Q)
  (perp_OPAQ : perpendicular O P A Q)
  (intersect_circles : intersects_circumcircles O P I J B C Q) : 
  dist O I = dist O J := 
sorry

end equal_lengths_tangent_circles_l681_681415


namespace cannot_determine_parallel_l681_681474

theorem cannot_determine_parallel 
  (h1 : ∀ (l1 l2 : Line), (∃ p1 p2 p3 p4 : Point, corresponding_angles_equal l1 l2 p1 p2 p3 p4) → parallel l1 l2)
  (h2 : ∀ (l1 l2 : Line), (∃ p1 p2 p3 p4 : Point, alternate_angles_equal l1 l2 p1 p2 p3 p4) → parallel l1 l2)
  (h3 : ∀ (l1 l2 : Line), (∃ p1 p2 : Point, consecutive_interior_angles_supplementary l1 l2 p1 p2) → parallel l1 l2)
  : ∀ (l1 l2 : Line), (∃ p1 p2 : Point, vertically_opposite_angles_equal l1 l2 p1 p2) → ¬ parallel l1 l2 :=
by sorry

end cannot_determine_parallel_l681_681474


namespace kho_kho_only_l681_681058

theorem kho_kho_only (K H B total : ℕ) (h1 : K + B = 10) (h2 : B = 5) (h3 : K + H + B = 25) : H = 15 :=
by {
  sorry
}

end kho_kho_only_l681_681058


namespace arcsin_one_half_eq_pi_over_six_arccos_one_half_eq_pi_over_three_l681_681156

theorem arcsin_one_half_eq_pi_over_six : Real.arcsin (1/2) = Real.pi/6 :=
by 
  sorry

theorem arccos_one_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi/3 :=
by 
  sorry

end arcsin_one_half_eq_pi_over_six_arccos_one_half_eq_pi_over_three_l681_681156


namespace tagged_fish_count_l681_681269

theorem tagged_fish_count (N x : ℕ) (hN : N = 250) (h1 : 50 * x = 50 * 50 / N) : x = 10 :=
by {
  rw hN at h1,
  calc
    50 * x = 50 * 50 / 250 : h1
    ...  = 10 : by norm_num
}

end tagged_fish_count_l681_681269


namespace binom_18_6_eq_18564_l681_681504

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binom_18_6_eq_18564 : binomial 18 6 = 18564 := by
  sorry

end binom_18_6_eq_18564_l681_681504


namespace correct_choice_is_C_l681_681476

def is_opposite_number (a b : ℤ) : Prop := a + b = 0

def option_A : Prop := ¬is_opposite_number (2^3) (3^2)
def option_B : Prop := ¬is_opposite_number (-2) (-|-2|)
def option_C : Prop := is_opposite_number ((-3)^2) (-3^2)
def option_D : Prop := ¬is_opposite_number 2 (-(-2))

theorem correct_choice_is_C : option_C ∧ option_A ∧ option_B ∧ option_D :=
by
  sorry

end correct_choice_is_C_l681_681476


namespace simplify_fraction_l681_681926

theorem simplify_fraction : (3 ^ 2016 - 3 ^ 2014) / (3 ^ 2016 + 3 ^ 2014) = 4 / 5 :=
by
  sorry

end simplify_fraction_l681_681926


namespace binom_18_6_eq_18564_l681_681507

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binom_18_6_eq_18564 : binomial 18 6 = 18564 := by
  sorry

end binom_18_6_eq_18564_l681_681507


namespace part1_minimum_value_part2_max_k_l681_681230

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x
noncomputable def g (x : ℝ) : ℝ := (x + x * Real.log x) / (x - 1)

theorem part1_minimum_value : ∃ x₀ : ℝ, x₀ = Real.exp (-2) ∧ f x₀ = -Real.exp (-2) := 
by
  use Real.exp (-2)
  sorry

theorem part2_max_k (k : ℤ) : (∀ x > 1, f x > k * (x - 1)) → k ≤ 3 := 
by
  sorry

end part1_minimum_value_part2_max_k_l681_681230


namespace find_denomination_of_coins_l681_681175

-- Define the problem conditions
variables (total_amount_paid total_bills_and_coins bills_count : ℕ)
variable (coins_denomination : ℕ)

-- Given Conditions
def condition1 : total_amount_paid = 285 := rfl
def condition2 : total_bills_and_coins = 24 := rfl
def condition3 : bills_count = 11 := rfl
def condition4 : (bills_count + bills_count) = total_bills_and_coins := by rw [condition3]; exact rfl

-- Formulate the equation
def equation : 11 * 20 + (bills_count * coins_denomination) = 285 := by {
  iterate 2 { rw condition3 },
  exact rfl
}
    
-- Define the proof statement
theorem find_denomination_of_coins :
  (11 * coins_denomination = 65) → coins_denomination = 5 := by
    intro h,
    rw ←h,
    norm_num

end find_denomination_of_coins_l681_681175


namespace find_k_values_l681_681190

theorem find_k_values :
  ∀ x k : ℝ, (x^2 - (k - 2) * x - k + 8 > 0) ↔ (k ∈ set.Ioo (-2 * real.sqrt 7) (2 * real.sqrt 7)) := sorry

end find_k_values_l681_681190


namespace simplify_and_evaluate_l681_681345

theorem simplify_and_evaluate (x : ℝ) (h : x = -3) :
  (1 - (1 / (x - 1))) / ((x ^ 2 - 4 * x + 4) / (x ^ 2 - 1)) = (2 / 5) :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l681_681345


namespace pyramid_volume_l681_681340

def volume_of_pyramid {α : Type*} [LinearOrderedRing α] (AB BC PA : α) (h1 : AB = 10) (h2 : BC = 5) (h3 : PA = 9)
    (h4 : ∀ A B D, ⟪PA, AB⟫ = 0 ∧ ⟪PA, AD⟫ = 0) : α :=
    1 / 3 * (AB * BC) * PA

theorem pyramid_volume : volume_of_pyramid 10 5 9 (rfl) (rfl) (rfl) (sorry) = 150 := 
sorry

end pyramid_volume_l681_681340


namespace ratio_of_speeds_l681_681542

def eddy_time := 3
def eddy_distance := 480
def freddy_time := 4
def freddy_distance := 300

def eddy_speed := eddy_distance / eddy_time
def freddy_speed := freddy_distance / freddy_time

theorem ratio_of_speeds : (eddy_speed / freddy_speed) = 32 / 15 :=
by
  sorry

end ratio_of_speeds_l681_681542


namespace parallelogram_area_l681_681573

def base : ℕ := 34
def height : ℕ := 18
def area_of_parallelogram (b h : ℕ) : ℕ := b * h

theorem parallelogram_area : area_of_parallelogram base height = 612 := by
  sorry

end parallelogram_area_l681_681573


namespace cloth_selling_price_gain_l681_681084

/-- Prove that the number of meters of cloth's selling price gained is 10
    given that the shop owner sells 25 meters of cloth and the gain percentage is 66.67%. -/
theorem cloth_selling_price_gain
  (C S : ℝ)
  (hSP : S = (5/3) * C)
  (hGainPercentage : 2/3)
  (CP := 25 * C)
  (SP := 25 * S)
  (G := SP - CP)
  (hGain : G = (2/3) * CP) :
  ∃ M : ℝ, M * S = G ∧ M = 10 :=
by
  sorry

end cloth_selling_price_gain_l681_681084


namespace find_ellipse_equation_find_line_equation_l681_681229

-- Theorem 1: Finding the equation of the ellipse
theorem find_ellipse_equation (a b : ℝ) (a_pos : a > b) (b_pos : b > 0) (f1 f2 : ℝ × ℝ) 
(P : ℝ × ℝ) (pf1 pf2 : (ℝ × ℝ) → (ℝ × ℝ))
(h_f1 : f1 = (-2, 0)) (h_f2 : f2 = (2, 0))
(h_max : ∀ P, P ∈ (λ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1) → (pf1 P) • (pf2 P) <= 2):
  (a = sqrt 6) ∧ (b = sqrt 2) ∧ (∀ x y : ℝ, x^2 / 6 + y^2 / 2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

-- Theorem 2: Finding the equation of the line l
theorem find_line_equation (k : ℝ) (θ : ℝ) (S : ℝ → ℝ → ℝ)
(M N : (ℝ × ℝ)) (O : ℝ × ℝ) (h_O : O = (0, 0))
(h_MN_sinθ : ∀ M N, ∃ k, Sin.cos (M-θ) = (4 * sqrt 6 / 3) Cos.θ)
(θ_ne_pi_div_2 : θ ≠ π / 2)
(h_ellipse_eq : ∀ x y : ℝ, x^2 / 6 + y^2 / 2 = 1):
  (∀ x y : ℝ, (x = -2) ∨ (y = (sqrt 3 / 3) * (x + 2)) ∨ (y = -(sqrt 3 / 3) * (x + 2))) :=
sorry

end find_ellipse_equation_find_line_equation_l681_681229


namespace last_digits_6811_21000_3999_l681_681026

theorem last_digits_6811_21000_3999 :
  (6^811 % 10 = 6) ∧
  (2^1000 % 10 = 6) ∧
  (3^999 % 10 = 7) :=
by
  have h1 : ∀ n : ℕ, 6^n % 10 = 6, from sorry,
  have h2 : 2^1000 % 10 = 2^(1000 % 4 + 4) % 10, by sorry,
  have h3 : 3^999 % 10 = 3^(999 % 4 + 4) % 10, by sorry,
  split,
  { exact h1 811 },
  split,
  { rw [←nat.modeq.modeq_zero_iff, nat.mod_eq_of_lt] at h2,
    exact nat.gcd_eq_right (nat.div_dvd 1000 4) h2 },
  { rw [←nat.modeq.modeq_zero_iff, nat.mod_eq_of_lt] at h3,
    exact nat.gcd_eq_right (nat.div_dvd 999 4) h3 }

end last_digits_6811_21000_3999_l681_681026


namespace fish_caught_l681_681040

theorem fish_caught (x y : ℕ) 
  (h1 : y - 2 = 4 * (x + 2))
  (h2 : y - 6 = 2 * (x + 6)) :
  x = 4 ∧ y = 26 :=
by
  sorry

end fish_caught_l681_681040


namespace average_age_proof_l681_681359

noncomputable def average_age_of_remaining_students
  (average_age_class : ℕ → ℝ → ℝ)
  (remove_student : ℝ → ℝ → ℝ)
  (initial_people : ℕ)
  (initial_average_age : ℝ)
  (leaving_age : ℝ)
  (remaining_people : ℕ) : Prop :=
  average_age_class remaining_people (remove_student (initial_average_age * initial_people) leaving_age) = 28.857

def conditions : Prop :=
  let initial_people := 8 in
  let initial_average_age := 28 in
  let leaving_age := 22 in
  let remaining_people := 7 in
  average_age_of_remaining_students
    (λ people total_age, total_age / people)
    (λ total_age age, total_age - age)
    initial_people
    initial_average_age
    leaving_age
    remaining_people

theorem average_age_proof : conditions := 
  sorry

end average_age_proof_l681_681359


namespace area_enclosed_by_curve_l681_681355

theorem area_enclosed_by_curve :
  ∃ (area : ℝ), (∀ (x y : ℝ), |x - 1| + |y - 1| = 1 → area = 2) :=
sorry

end area_enclosed_by_curve_l681_681355


namespace length_AD_l681_681743

open Real
open_locale classical

variables {A B C D : Type} [NormedAddCommGroup A] [NormedSpace ℝ A]
variable (m : ℝ)
variable (α : ℝ)
-- Conditions
variables (ABC : A) (BAC : angle ABC = π / 2) 
variables (AD : A) {BC : ℝ} (hBC : BC = m)
variable (B : ℝ) (hB : B = α)
variables (hAD_Perpendicular_BC : is_perpendicular AD BC)

-- Proof problem to be proved
theorem length_AD (h : AD.perp BC) : AD.length = m * sin α * cos α := sorry

end length_AD_l681_681743


namespace no_real_solutions_l681_681839

noncomputable def original_eq (x : ℝ) : Prop := (x^2 + x + 1) / (x + 1) = x^2 + 5 * x + 6

theorem no_real_solutions (x : ℝ) : ¬ original_eq x :=
by
  sorry

end no_real_solutions_l681_681839


namespace sum_of_integers_in_range_l681_681589

noncomputable def f (x : ℝ) : ℝ :=
  Real.logBase 3 (10 * Real.cos (2 * x) + 17)

def interval_x : Set ℝ :=
  Set.Icc (
    1.25 * Real.arctan 0.25 * Real.cos (Real.pi - Real.arcsin (-0.6))
  ) (Real.arctan 3)

theorem sum_of_integers_in_range :
  (∑ (n : ℤ) in (Set.Icc 2 3), n) = 5 :=
by
  sorry

end sum_of_integers_in_range_l681_681589


namespace maximum_x1_x2_x3_l681_681309

theorem maximum_x1_x2_x3 :
  ∀ (x1 x2 x3 x4 x5 x6 x7 : ℕ),
  x1 < x2 → x2 < x3 → x3 < x4 → x4 < x5 → x5 < x6 → x6 < x7 →
  x1 + x2 + x3 + x4 + x5 + x6 + x7 = 159 →
  x1 + x2 + x3 ≤ 61 := 
by sorry

end maximum_x1_x2_x3_l681_681309


namespace recurring_decimal_exceeds_by_fraction_l681_681111

theorem recurring_decimal_exceeds_by_fraction : 
  let y := (36 : ℚ) / 99
  let x := (36 : ℚ) / 100
  ((4 : ℚ) / 11) - x = (4 : ℚ) / 1100 :=
by
  sorry

end recurring_decimal_exceeds_by_fraction_l681_681111


namespace geometric_product_geometric_quotient_l681_681638

def is_geometric_sequence (s : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, s (n + 1) = q * s n

variables {a b : ℕ → ℝ} {q1 q2 : ℝ}

-- Assume {a_n} and {b_n} are geometric sequences with ratios q1 and q2, respectively
axiom geom_a : is_geometric_sequence a q1
axiom geom_b : is_geometric_sequence b q2
axiom nonzero_a : ∀ n, a n ≠ 0
axiom nonzero_b : ∀ n, b n ≠ 0

-- Prove that {a_n * b_n} is a geometric sequence
theorem geometric_product : is_geometric_sequence (λ n, a n * b n) (q1 * q2) := sorry

-- Prove that {a_n / b_n} is a geometric sequence
theorem geometric_quotient : is_geometric_sequence (λ n, a n / b n) (q1 / q2) := sorry

end geometric_product_geometric_quotient_l681_681638


namespace greatest_base8_three_digit_divisible_by_7_l681_681908

theorem greatest_base8_three_digit_divisible_by_7 :
  ∃ n : ℕ, n < 8^3 ∧ n ≥ 8^2 ∧ (n % 7 = 0) ∧ (to_base 8 n = 777) :=
sorry

end greatest_base8_three_digit_divisible_by_7_l681_681908


namespace round_nearest_hundredth_l681_681017

theorem round_nearest_hundredth (x : ℝ) (hx : x = 0.9247) : round (x * 100) / 100 = 0.92 :=
by
  have ha : round (0.9247 * 100) = 92 := sorry
  rw [hx, ha]

end round_nearest_hundredth_l681_681017


namespace smallest_prime_factor_3465_l681_681030

theorem smallest_prime_factor_3465 : ∃ p : ℕ, prime p ∧ p ∣ 3465 ∧ (∀ q : ℕ, prime q ∧ q ∣ 3465 → p ≤ q) :=
sorry

end smallest_prime_factor_3465_l681_681030


namespace greatest_div_by_seven_base_eight_l681_681898

theorem greatest_div_by_seven_base_eight : ∃ n : ℕ, 
  (n < 512) ∧ (Divisibility.divides 7 n) ∧ 
  (∀ m : ℕ, (m < 512) → (Divisibility.divides 7 m) → m ≤ n) ∧ 
  nat.to_digits 8 n = [7, 7, 4] := 
sorry

end greatest_div_by_seven_base_eight_l681_681898


namespace arithmetic_series_sum_l681_681124

theorem arithmetic_series_sum : 
  let a := 2 in 
  let d := 2 in 
  let n := 10 in 
  let l := 20 in 
  (a + l) * n / 2 = 110 := 
by
  sorry

end arithmetic_series_sum_l681_681124


namespace find_m_l681_681420

def pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem find_m :
  ∀ (m : ℝ), pure_imaginary (m ^ 2 - 4 + (m + 2 : ℂ).i) → m = -2 :=
by {
  sorry,
}

end find_m_l681_681420


namespace units_digit_4659_pow_157_l681_681146

theorem units_digit_4659_pow_157 : 
  (4659^157) % 10 = 9 := 
by 
  sorry

end units_digit_4659_pow_157_l681_681146
