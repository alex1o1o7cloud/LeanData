import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.ConicSections
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Combinatorics
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.CombinatorialGame
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Tactic
import data.real.basic
import order.floor

namespace count_3_digit_multiples_of_13_l683_683785

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683785


namespace num_three_digit_div_by_13_l683_683856

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683856


namespace count_3_digit_numbers_divisible_by_13_l683_683703

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683703


namespace count_3_digit_numbers_divisible_by_13_l683_683663

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683663


namespace count_three_digit_numbers_divisible_by_13_l683_683936

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683936


namespace prime_numbers_in_list_l683_683577

noncomputable def list_numbers : ℕ → ℕ
| 0       => 43
| (n + 1) => 43 * ((10 ^ (2 * n + 2) - 1) / 99) 

theorem prime_numbers_in_list : ∃ n:ℕ, (∀ m, (m > n) → ¬ Prime (list_numbers m)) ∧ Prime (list_numbers 0) := 
by
  sorry

end prime_numbers_in_list_l683_683577


namespace num_three_digit_div_by_13_l683_683845

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683845


namespace tangent_line_at_1_f_increasing_decreasing_intervals_l683_683565

noncomputable def f : ℝ → ℝ := λ x, x^2 - Real.log x

theorem tangent_line_at_1 (x : ℝ) :
  let p := (1, f 1)
  in x = 1 → (f '(1) * (x - 1) + f 1 = x) := 
sorry

theorem f_increasing_decreasing_intervals :
  (∀ x, x > Real.sqrt 2 / 2 → 2 * x - 1 / x > 0) ∧ 
  (∀ x, x > 0 ∧ x < Real.sqrt 2 / 2 → 2 * x - 1 / x < 0) :=
sorry

end tangent_line_at_1_f_increasing_decreasing_intervals_l683_683565


namespace three_digit_numbers_divisible_by_13_count_l683_683630

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683630


namespace three_digit_numbers_divisible_by_13_count_l683_683629

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683629


namespace system_solutions_range_b_l683_683163

theorem system_solutions_range_b (b : ℝ) :
  (∀ x y : ℝ, x^2 - y^2 = 0 → x^2 + (y - b)^2 = 2 → x = 0 ∧ y = 0 ∨ y = b) →
  b ≥ 2 ∨ b ≤ -2 :=
sorry

end system_solutions_range_b_l683_683163


namespace num_valid_permutations_l683_683457

open List

def is_valid_permutation (l : List ℕ) : Prop :=
  l.length = 9 ∧
  (∀ i, i < 7 → (l.nth_le i sorry + l.nth_le (i + 1) sorry + l.nth_le (i + 2) sorry) % 3 = 0)

noncomputable def valid_permutations : List (List ℕ) := 
  (Finset.univ : Finset (List ℕ)).filter is_valid_permutation

theorem num_valid_permutations : valid_permutations.length = 1296 := 
sorry

end num_valid_permutations_l683_683457


namespace lost_card_number_l683_683379

theorem lost_card_number (n x : ℕ) (sum_n : ℕ) (h_sum : sum_n = n * (n + 1) / 2) (h_remaining_sum : sum_n - x = 101) : x = 4 :=
sorry

end lost_card_number_l683_683379


namespace change_in_mean_and_median_l683_683312

-- Original attendance data
def original_data : List ℕ := [15, 23, 17, 19, 17, 20]

-- Corrected attendance data
def corrected_data : List ℕ := [15, 23, 17, 19, 17, 25]

-- Function to compute mean
def mean (data: List ℕ) : ℚ := (data.sum : ℚ) / data.length

-- Function to compute median
def median (data: List ℕ) : ℚ :=
  let sorted := data.toArray.qsort (· ≤ ·) |>.toList
  if sorted.length % 2 == 0 then
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2
  else
    sorted.get! (sorted.length / 2)

-- Lean statement verifying the expected change in mean and median
theorem change_in_mean_and_median :
  mean corrected_data - mean original_data = 1 ∧ median corrected_data = median original_data :=
by -- Note the use of 'by' to structure the proof
  sorry -- Proof omitted

end change_in_mean_and_median_l683_683312


namespace count_3digit_numbers_div_by_13_l683_683880

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683880


namespace num_three_digit_div_by_13_l683_683843

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683843


namespace three_digit_numbers_divisible_by_13_l683_683740

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683740


namespace total_books_read_l683_683003

-- Definitions based on conditions
def books_per_month : Nat := 3
def months_in_year : Nat := 12
def books_per_year : Nat := books_per_month * months_in_year

variable (c s : Nat) -- c: number of classes, s: number of students per class

def total_students := s * c

-- Statement to prove
theorem total_books_read (c s : Nat) : 
  total_books_read c s = books_per_year * total_students s c := by
  sorry

end total_books_read_l683_683003


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683587

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683587


namespace evaluation_of_expression_l683_683417

theorem evaluation_of_expression :
  10 * (1 / 8) - 6.4 / 8 + 1.2 * 0.125 = 0.6 :=
by sorry

end evaluation_of_expression_l683_683417


namespace count_3_digit_numbers_divisible_by_13_l683_683709

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683709


namespace lost_card_l683_683368

noncomputable def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

theorem lost_card (n : ℕ) (h1 : sum_natural n - 101 < n) (h2 : sum_natural (n - 1) ≤ 101) :
  ∃ x : ℕ, x = sum_natural n - 101 ∧ x = 4 :=
begin
  have h_sum : sum_natural 14 = 105,
  {
    unfold sum_natural,
    norm_num,
  },
  use 4,
  split,
  {
    unfold sum_natural at *,
    norm_num,
    rw h_sum,
  },
  {
    norm_num,
  },
  sorry,
end

end lost_card_l683_683368


namespace two_pair_probability_l683_683340

noncomputable def prob_two_pair : ℚ :=
  let total_outcomes := (Nat.choose 52 5 : ℚ) in
  let successful_outcomes := (13 * 6 * 12 * 6 * 11 * 4 : ℚ) in
  successful_outcomes / total_outcomes

theorem two_pair_probability :
  prob_two_pair = 108 / 1005 := by sorry

end two_pair_probability_l683_683340


namespace total_games_played_l683_683197

-- Variables representing the number of teams and games played between each team.
variable (n : ℕ) (g : ℕ)

-- Condition: There are 12 teams in the league.
def num_teams := 12

-- Condition: Each team plays exactly 4 games with each of the other teams.
def games_per_pair := 4

-- Question: Prove that the total number of games played during the season is 264.
theorem total_games_played : 
  (n = num_teams) → (g = games_per_pair) → (4 * (n * (n - 1) / 2) = 264) :=
  by
    intros,
    sorry

#check total_games_played

end total_games_played_l683_683197


namespace count_3_digit_numbers_divisible_by_13_l683_683869

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683869


namespace new_energy_vehicle_quality_control_l683_683266

noncomputable def sample_mean := 70
noncomputable def probability_interval := 0.8186
noncomputable def probability_defective := 0.016
noncomputable def probability_defective_from_first_line := 5 / 8

/-- 
Assuming the quality deviation follows a normal distribution with given parameters,
prove the sample mean and calculated probabilities. 
-/
theorem new_energy_vehicle_quality_control
  (X : ℝ → ℝ) -- Quality deviation follows N(μ, σ²)
  (μ : ℝ := 70) (σ_sq : ℝ := 36)
  (deviations : List ℝ := [56, 67, 70, 78, 86])
  (counts : List ℕ := [10, 20, 48, 19, 3])
  (total_count : ℕ := 100)
  (prob_first_line : ℝ := 0.015)
  (prob_second_line : ℝ := 0.018)
  (efficiency_ratio : ℝ := 2 / 1) :
  -- 1. Sample mean calculation
  ((List.zipWith (· * ·) deviations (counts.map (· : ℝ))).sum / total_count = sample_mean) ∧
  -- 2. Probability interval calculation
  (probability_interval = 0.8186) ∧
  -- 3. Probability of selecting a defective part
  (prob_first_line * efficiency_ratio / (efficiency_ratio + 1) + 
   prob_second_line / (efficiency_ratio + 1) = probability_defective) ∧
  -- 4. Probability that a defective part came from the first production line
  ((efficiency_ratio / (efficiency_ratio + 1) * prob_first_line) / 
   probability_defective = probability_defective_from_first_line) :=
begin
  sorry
end

end new_energy_vehicle_quality_control_l683_683266


namespace percentage_of_children_allowed_to_draw_l683_683328

def total_jelly_beans := 100
def total_children := 40
def remaining_jelly_beans := 36
def jelly_beans_per_child := 2

theorem percentage_of_children_allowed_to_draw :
  ((total_jelly_beans - remaining_jelly_beans) / jelly_beans_per_child : ℕ) * 100 / total_children = 80 := by
  sorry

end percentage_of_children_allowed_to_draw_l683_683328


namespace three_digit_numbers_div_by_13_l683_683730

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683730


namespace number_of_three_digit_numbers_divisible_by_13_l683_683938

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683938


namespace count_three_digit_numbers_divisible_by_13_l683_683610

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683610


namespace min_value_sqrt_x2_y2_l683_683081

theorem min_value_sqrt_x2_y2 {x y : ℝ} (h1 : 3 * x + 4 * y = 24) (h2 : x ≥ 0) :
  ∃ (m : ℝ), m = sqrt (x^2 + y^2) ∧ m = 24 / 5 :=
sorry

end min_value_sqrt_x2_y2_l683_683081


namespace increasing_log_condition_range_of_a_l683_683559

noncomputable def t (x a : ℝ) := x^2 - a*x + 3*a

theorem increasing_log_condition :
  (∀ x ≥ 2, 2 * x - a ≥ 0) ∧ a > -4 ∧ a ≤ 4 →
  ∀ x ≥ 2, x^2 - a*x + 3*a > 0 :=
by
  sorry

theorem range_of_a
  (h1 : ∀ x ≥ 2, 2 * x - a ≥ 0)
  (h2 : 4 - 2 * a + 3 * a > 0)
  (h3 : ∀ x ≥ 2, t x a > 0)
  : a > -4 ∧ a ≤ 4 :=
by
  sorry

end increasing_log_condition_range_of_a_l683_683559


namespace smallest_positive_period_l683_683149

def f (x : ℝ) : ℝ := Real.sin x ^ 2

theorem smallest_positive_period : ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ T = Real.pi :=
by
  sorry

end smallest_positive_period_l683_683149


namespace sum_first_n_terms_div_an_an1_l683_683553

theorem sum_first_n_terms_div_an_an1 {n : ℕ} (h : ∀ n : ℕ, ∑ i in finset.range (n + 1), a i = n^2 + 2 * n) :
  ∑ i in finset.range n, 1 / (a i * a (i + 1)) = n / (3 * (2 * n + 3)) :=
by
  sorry

end sum_first_n_terms_div_an_an1_l683_683553


namespace determine_n_l683_683057

theorem determine_n : ∃ n : ℤ, 0 ≤ n ∧ n < 8 ∧ -2222 % 8 = n := by
  use 2
  sorry

end determine_n_l683_683057


namespace card_probability_l683_683113

theorem card_probability (cards : Finset ℕ) (first_even second_odd : ℕ → Prop) (card_draw : ℕ → ℕ → Prop)
  (h1 : cards = {1, 2, 3, 4, 5})
  (h2 : ∀ x ∈ cards, first_even x ↔ x % 2 = 0)
  (h3 : ∀ x ∈ cards, ¬ first_even x ↔ x % 2 ≠ 0)
  (h4 : ∀ x ∈ cards, second_odd x ↔ x % 2 = 1)
  (h5 : ∃ c ∈ cards, first_even c)
  (h6 : ∃ c1 c2 ∈ cards, card_draw c1 c2 ∧ first_even c1 ∧ second_odd c2) :
  (∃ c1, first_even c1 ∧ ∀ c2, c2 ≠ c1 → second_odd c2) →
  ∃ c1 c2, card_draw c1 c2 ∧ first_even c1 ∧ second_odd c2 ∧ (classic.prob (λ x, second_odd x) (λ x, card_draw c1 x) = 3 / 4) :=
by
  -- proof here
  sorry

end card_probability_l683_683113


namespace hyperbola_real_axis_length_l683_683965

theorem hyperbola_real_axis_length :
  ∀ (a : ℝ), (a > 0) → 
  (∃ k : ℝ, k = -3 ∧ 
    ∀ (x y : ℝ), y = (a / 3) * x ∨ y = (-a / 3) * x → 
    y = k * x → 
    2 * a = 18) := by
  intros a ha
  exists (-3)
  constructor
  { refl }
  { intros x y h_asym h_perp
    cases h_asym
    all_goals sorry }

end hyperbola_real_axis_length_l683_683965


namespace kids_french_fries_cost_l683_683461

noncomputable def cost_burger : ℝ := 5
noncomputable def cost_fries : ℝ := 3
noncomputable def cost_soft_drink : ℝ := 3
noncomputable def cost_special_burger_meal : ℝ := 9.50
noncomputable def cost_kids_burger : ℝ := 3
noncomputable def cost_kids_juice_box : ℝ := 2
noncomputable def cost_kids_meal : ℝ := 5
noncomputable def savings : ℝ := 10

noncomputable def total_adult_meal_individual : ℝ := 2 * cost_burger + 2 * cost_fries + 2 * cost_soft_drink
noncomputable def total_adult_meal_deal : ℝ := 2 * cost_special_burger_meal

noncomputable def total_kids_meal_individual (F : ℝ) : ℝ := 2 * cost_kids_burger + 2 * F + 2 * cost_kids_juice_box
noncomputable def total_kids_meal_deal : ℝ := 2 * cost_kids_meal

noncomputable def total_cost_individual (F : ℝ) : ℝ := total_adult_meal_individual + total_kids_meal_individual F
noncomputable def total_cost_deal : ℝ := total_adult_meal_deal + total_kids_meal_deal

theorem kids_french_fries_cost : ∃ F : ℝ, total_cost_individual F - total_cost_deal = savings ∧ F = 3.50 := 
by
  use 3.50
  sorry

end kids_french_fries_cost_l683_683461


namespace three_digit_numbers_divisible_by_13_l683_683743

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683743


namespace doctors_visit_cost_l683_683334

-- Conditions
def cost_of_vaccines := 10 * 45
def trip_cost := 1200
def total_payment := 1340
def insurance_coverage_rate := 0.80
def out_of_pocket_payment_rate := 0.20

-- Question: How much does the doctor's visit cost?
noncomputable def cost_of_doctors_visit := sorry

-- Proof Statement
theorem doctors_visit_cost : 
  let D := cost_of_doctors_visit in
  0.20 * (cost_of_vaccines + D) = total_payment - trip_cost → D = 250 := 
by 
  intros h
  sorry

end doctors_visit_cost_l683_683334


namespace count_3_digit_numbers_divisible_by_13_l683_683699

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683699


namespace bug_distance_l683_683424

theorem bug_distance : 
  let start := 3
  let point1 := -4
  let point2 := 8
  let point3 := -1
  abs(point1 - start) + abs(point2 - point1) + abs(point3 - point2) = 28 :=
by
  sorry

end bug_distance_l683_683424


namespace gcd_monomials_l683_683291

variable {α : Type}
variables (a b c : α) [CommRing α] [Nonzero α]

def monomial1 : α := 4 * a^2 * b^2 * c
def monomial2 : α := 6 * a * b^3

theorem gcd_monomials (a b c : α) [CommRing α] [Nonzero α] :
  (gcd monomial1 monomial2) = 2 * a * b^2 :=
sorry

end gcd_monomials_l683_683291


namespace problem_statement_l683_683116

namespace ProofProblems

open Set

def U : Set ℕ := {2, 3, 4, 5, 6}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {2, 4, 5, 6}

theorem problem_statement : M ∪ N = U := sorry

end ProofProblems

end problem_statement_l683_683116


namespace find_m_range_l683_683086

noncomputable def quadratic_inequality_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x + 2 > 0

theorem find_m_range :
  { m : ℝ | quadratic_inequality_condition m } = { m : ℝ | 1 ≤ m ∧ m < 9 } :=
sorry

end find_m_range_l683_683086


namespace original_hourly_wage_l683_683025

theorem original_hourly_wage 
  (daily_wage_increase : ∀ W : ℝ, 1.60 * W + 10 = 45)
  (work_hours : ℝ := 8) : 
  ∃ W_hourly : ℝ, W_hourly = 2.73 :=
by 
  have W : ℝ := (45 - 10) / 1.60 
  have W_hourly : ℝ := W / work_hours
  use W_hourly 
  sorry

end original_hourly_wage_l683_683025


namespace progress_regress_ratio_l683_683416

theorem progress_regress_ratio :
  let progress_rate := 1.2
  let regress_rate := 0.8
  let log2 := 0.3010
  let log3 := 0.4771
  let target_ratio := 10000
  (progress_rate / regress_rate) ^ 23 = target_ratio :=
by
  sorry

end progress_regress_ratio_l683_683416


namespace mean_median_difference_l683_683189

theorem mean_median_difference 
  (perc65 perc75 perc85 perc95 : ℝ)
  (h_perc65 : perc65 = 0.20) (h_perc75 : perc75 = 0.25) 
  (h_perc85 : perc85 = 0.25) (h_perc95 : perc95 = 0.30)
  (score65 score75 score85 score95 : ℝ)
  (h_score65 : score65 = 65) (h_score75 : score75 = 75)
  (h_score85 : score85 = 85) (h_score95 : score95 = 95) :
  let mean := perc65 * score65 + perc75 * score75 + perc85 * score85 + perc95 * score95 in
  let median := 85 in
  median - mean = 3.5 :=
sorry

end mean_median_difference_l683_683189


namespace isosceles_right_triangle_perimeter_l683_683084

noncomputable def perimeter_isosceles_right_triangle (a b c : ℝ) : ℝ :=
  a + b + c

theorem isosceles_right_triangle_perimeter :
  ∃ (a b c : ℝ), 
    (3 / 5) = a / b ∧
    50 = (1 / 2) * a * b ∧
    c^2 = a^2 + b^2 ∧
    perimeter_isosceles_right_triangle a b c ≈ 29.78 :=
by
  sorry

end isosceles_right_triangle_perimeter_l683_683084


namespace count_3_digit_numbers_divisible_by_13_l683_683714

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683714


namespace evaluate_series_l683_683097

def closest_integer (n : ℕ) : ℕ :=
  if sqrt n * sqrt n = n then sqrt n
  else if (sqrt n + 1) * (sqrt n + 1) - n < n - sqrt n * sqrt n then sqrt n + 1
  else sqrt n

theorem evaluate_series : 
  (∑' n : ℕ, (3^(closest_integer n) + 3^(-closest_integer n)) / 3^n) = 3 := 
sorry

end evaluate_series_l683_683097


namespace smallest_integer_relative_prime_to_2310_l683_683352

theorem smallest_integer_relative_prime_to_2310 (n : ℕ) : (2 < n → n ≤ 13 → ¬ (n ∣ 2310)) → n = 13 := by
  sorry

end smallest_integer_relative_prime_to_2310_l683_683352


namespace contest_paths_l683_683470

def grid := ["C O N T E S T S",
             "O N T E S T S E",
             "N T E S T S E T",
             "T E S T S E T N",
             "E S T S E T N O",
             "S T S E T N O C",
             "T S E T N O C N",
             "S E T N O C N O",
             "E T N O C N O T"]

def word := "CONTESTS"

def is_valid_path (path: List (Nat × Nat)) : Prop :=
  -- Ensure the path follows the word "CONTESTS" horizontally or vertically
  path.length = 8 ∧
  (path.map (fun p => grid.getD p.fst "").getD p.snd ' ') = list.to_string word ∧
  ∀ i, 0 ≤ i < 7 → (path.getD i.snd.1 - path.getD i.fst.1).abs + (path.getD i.snd.2 - path.getD i.fst.2).abs = 1

def count_paths (grid : List String) (word : String) : Nat := sorry

theorem contest_paths : count_paths grid word = 384 := sorry

end contest_paths_l683_683470


namespace ratio_areas_BC_Y_and_BA_X_proof_l683_683212

open Real

noncomputable def ratio_areas_BC_Y_and_BA_X (A B C X Y : Point) 
    (hB_on_AB : IsOnSegment B A X)
    (hY_on_AC : IsOnSegment Y A C)
    (hBX_bis_ABC : AngleBisector B X A C)
    (hCY_bis_ACB : AngleBisector C Y A B)
    (AB_eq : dist A B = 35)
    (AC_eq : dist A C = 42)
    (BC_eq : dist B C = 48) : ℚ := 
  have area_BC_Y : Real := area B C Y
  have area_BA_X : Real := area B A X
  area_BC_Y / area_BA_X

theorem ratio_areas_BC_Y_and_BA_X_proof (A B C X Y : Point)
    (hB_on_AB : IsOnSegment B A X)
    (hY_on_AC : IsOnSegment Y A C)
    (hBX_bis_ABC : AngleBisector B X A C)
    (hCY_bis_ACB : AngleBisector C Y A B)
    (AB_eq : dist A B = 35)
    (AC_eq : dist A C = 42)
    (BC_eq : dist B C = 48)
    : ratio_areas_BC_Y_and_BA_X A B C X Y hB_on_AB hY_on_AC hBX_bis_ABC hCY_bis_ACB AB_eq AC_eq BC_eq = (16/5 : ℚ) :=
by 
  sorry

end ratio_areas_BC_Y_and_BA_X_proof_l683_683212


namespace number_of_three_digit_numbers_divisible_by_13_l683_683949

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683949


namespace lost_card_number_l683_683381

theorem lost_card_number (n x : ℕ) (sum_n : ℕ) (h_sum : sum_n = n * (n + 1) / 2) (h_remaining_sum : sum_n - x = 101) : x = 4 :=
sorry

end lost_card_number_l683_683381


namespace three_digit_numbers_div_by_13_l683_683721

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683721


namespace count_3_digit_numbers_divisible_by_13_l683_683859

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683859


namespace number_of_three_digit_numbers_divisible_by_13_l683_683947

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683947


namespace count_3_digit_numbers_divisible_by_13_l683_683677

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683677


namespace find_alpha_beta_l683_683075

theorem find_alpha_beta :
  ∃ (α β : ℝ), (∀ (x : ℝ), x ≠ -β ∧ x^2 - 144*x + 1050 ≠ 0 →
    (x - α) / (x + β) = (x^2 + 120*x + 1575) / (x^2 - 144*x + 1050)) ∧ α + β = 5 :=
begin
  sorry
end

end find_alpha_beta_l683_683075


namespace board_coloring_condition_l683_683488

def neighbors (m n : ℕ) (board : Fin m × Fin n → bool) (cell : Fin m × Fin n) : List (Fin m × Fin n) :=
  let (i, j) := cell
  [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1), (i + 1, j + 1), (i - 1, j - 1), (i + 1, j - 1), (i - 1, j + 1)]
  |>.filter (λ (x, y), x < m ∧ y < n)

def neighbor_same_colored_cells_odd (m n : ℕ) (board : Fin m × Fin n → bool) : Prop :=
  ∀ cell, (neighbors m n board cell)
    .filter (λ neighbor, board cell = board neighbor)
    .length % 2 = 1

theorem board_coloring_condition (m n : ℕ) (board : Fin m × Fin n → bool) :
  neighbor_same_colored_cells_odd m n board ↔ (m % 2 = 0 ∨ n % 2 = 0) :=
sorry

end board_coloring_condition_l683_683488


namespace distance_from_A_to_origin_l683_683158

open Real

theorem distance_from_A_to_origin 
  (x1 y1 : ℝ)
  (hx1 : y1^2 = 4 * x1)
  (hratio : (x1 + 1) / abs y1 = 5 / 4)
  (hAF_gt_2 : dist (x1, y1) (1, 0) > 2) : 
  dist (x1, y1) (0, 0) = 4 * sqrt 2 :=
sorry

end distance_from_A_to_origin_l683_683158


namespace solution_for_system_of_inequalities_l683_683512

theorem solution_for_system_of_inequalities (p : ℝ) : 
  (19 * p < 10) ∧ (p > 0.5) ↔ (0.5 < p ∧ p < 10/19) := 
by {
  sorry
}

end solution_for_system_of_inequalities_l683_683512


namespace count_three_digit_numbers_divisible_by_13_l683_683613

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683613


namespace ratio_a_to_c_l683_683405

theorem ratio_a_to_c (a b c : ℕ) (h1 : a / b = 5 / 3) (h2 : b / c = 1 / 5) : a / c = 1 / 3 :=
sorry

end ratio_a_to_c_l683_683405


namespace num_pos_3_digit_div_by_13_l683_683774

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683774


namespace first_installment_amount_l683_683214

-- Define the conditions stated in the problem
def original_price : ℝ := 480
def discount_rate : ℝ := 0.05
def monthly_installment : ℝ := 102
def number_of_installments : ℕ := 3

-- The final price after discount
def final_price : ℝ := original_price * (1 - discount_rate)

-- The total amount of the 3 monthly installments
def total_of_3_installments : ℝ := monthly_installment * number_of_installments

-- The first installment paid
def first_installment : ℝ := final_price - total_of_3_installments

-- The main theorem to prove the first installment amount
theorem first_installment_amount : first_installment = 150 := by
  unfold first_installment
  unfold final_price
  unfold total_of_3_installments
  unfold original_price
  unfold discount_rate
  unfold monthly_installment
  unfold number_of_installments
  sorry

end first_installment_amount_l683_683214


namespace total_sequins_correct_l683_683219

def blue_rows : ℕ := 6
def blue_columns : ℕ := 8
def purple_rows : ℕ := 5
def purple_columns : ℕ := 12
def green_rows : ℕ := 9
def green_columns : ℕ := 6

def total_sequins : ℕ :=
  (blue_rows * blue_columns) + (purple_rows * purple_columns) + (green_rows * green_columns)

theorem total_sequins_correct : total_sequins = 162 := by
  sorry

end total_sequins_correct_l683_683219


namespace count_3_digit_numbers_divisible_by_13_l683_683664

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683664


namespace point_A_in_region_l683_683477

theorem point_A_in_region : (0 : ℝ) + (0 : ℝ) - 1 < 0 := 
by {
  calc
    (0 : ℝ) + (0 : ℝ) - 1 = -1 : by norm_num
    ... < 0 : by norm_num
}

end point_A_in_region_l683_683477


namespace num_pos_3_digit_div_by_13_l683_683770

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683770


namespace angles_of_triangle_CDE_l683_683270

-- Define the equilateral triangle and the points in 3D space or affine space
variable {V : Type*} [inner_product_space ℝ V] [finite_dimensional ℝ V]

structure EquilateralTriangle (A B C : V) : Prop :=
(is_ABC_equilateral : dist A B = dist B C ∧ dist B C = dist C A)

structure Parallel (u v : V) : Prop :=
(is_parallel : ∃ c : ℝ, u = c • v)

structure Midpoint (A B E : V) : Prop :=
(midpoint_property : 2 • E = A + B)

structure Centroid (A B C D : V) : Prop :=
(centroid_property : D = (A + B + C) / 3)

def find_angles_of_triangle (C D E : V) : Prop :=
  let θ1 := 30
  let θ2 := 60
  let θ3 := 90
  ∠ CDE = θ1 ∧ ∠ DCE = θ2 ∧ ∠ ECD = θ3

theorem angles_of_triangle_CDE {A B C M N E D : V}
  (h1 : EquilateralTriangle A B C)
  (h2 : Parallel (M - A) (C - A))
  (h3 : Midpoint A N E)
  (h4 : Centroid B M N D) :
  find_angles_of_triangle C D E :=
begin
  sorry
end

end angles_of_triangle_CDE_l683_683270


namespace three_digit_numbers_divisible_by_13_l683_683744

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683744


namespace expression_evaluation_l683_683490

theorem expression_evaluation : 
  ( ((2 + 2)^2 / 2^2) * ((3 + 3 + 3 + 3)^3 / (3 + 3 + 3)^3) * ((6 + 6 + 6 + 6 + 6 + 6)^6 / (6 + 6 + 6 + 6)^6) = 108 ) := 
by 
  sorry

end expression_evaluation_l683_683490


namespace length_NB_l683_683159

-- Definitions based on the conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def F : ℝ × ℝ := (1, 0)
def point_A (x y : ℝ) : Prop := parabola x y ∧ x > 0 ∧ y > 0
def center_of_circle : Prop := F
def radius_of_circle : ℝ := 1
def projection_on_y_axis (x y : ℝ) : ℝ × ℝ := (0, y)
def N (y : ℝ) : ℝ × ℝ := (0, y)
def o : ℝ × ℝ := (0, 0)
def distance_o_n (n : ℝ × ℝ) : ℝ := real.sqrt ((o.1 - n.1)^2 + (o.2 - n.2)^2)

-- Given conditions
axiom A_coords : ∃ x y, point_A x y ∧ (x, y) = (3, 2 * real.sqrt 3)
axiom N_coords : N (2 * real.sqrt 3) = (0, 2 * real.sqrt 3)
axiom intersect_line_AF_circle : ∃ B, (B : ℝ × ℝ) = ((3 / 2, real.sqrt 3 / 2)) ∧ (distance_o_n ((3 / 2, real.sqrt 3 / 2)) = 3)

-- The proof problem
theorem length_NB : ∀ B : ℝ × ℝ, (B = (3 / 2, real.sqrt 3 / 2)) → ∀ N : ℝ × ℝ, (N =  (0, 2 * real.sqrt 3)) → (real.sqrt ((N.1 - B.1)^2 + (N.2 - B.2)^2) = 3):=
by
  intros B hB N hN
  rw [hB, hN]
  simp
  sorry

end length_NB_l683_683159


namespace count_three_digit_div_by_13_l683_683680

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683680


namespace number_of_three_digit_numbers_divisible_by_13_l683_683956

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683956


namespace count_3_digit_numbers_divisible_by_13_l683_683702

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683702


namespace points_for_win_l683_683190

variable (W T : ℕ)

theorem points_for_win (W T : ℕ) (h1 : W * (T + 12) + T = 60) : W = 2 :=
by {
  sorry
}

end points_for_win_l683_683190


namespace distance_between_stations_l683_683338

/-
Train A travels at 20 km/hr for the first 2 hours,
then increases speed to 30 km/hr.
Train B travels at 25 km/hr for the first 3 hours,
then stops for 1 hour, and continues at 25 km/hr.
Train A has traveled 180 km more than Train B when they meet.
Prove that the distance between Station A and Station B is 2180 km.
-/

theorem distance_between_stations :
  let t := 38 in
  let distanceA := 20 * 2 + 30 * t in
  let distanceB := 25 * 3 + 25 * (t - 1) in
  distanceA - distanceB = 180 →
  distanceA + distanceB = 2180 :=
by
  sorry

end distance_between_stations_l683_683338


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683588

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683588


namespace paths_from_A_to_B_l683_683048

theorem paths_from_A_to_B :
  let rows := 5
  let columns := 8
  let start := (0, 0)
  let end := (columns, rows)
  let forbidden_segments := [((4, 3), (4, 4)), ((6, 1), (6, 2))]
  number_of_paths rows columns start end forbidden_segments = 692 :=
by
  sorry

end paths_from_A_to_B_l683_683048


namespace three_digit_numbers_divisible_by_13_l683_683750

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683750


namespace find_OD_expression_l683_683261

variables {O A B D : Type} [metric_space O] 
variables (theta s c : ℝ) (circle_radius : ℝ := 2)

-- Define the geometric constraints
variables (rAD : dist O A = circle_radius)   -- Point A on the circle
variables (rBO : dist O B = sorry)           -- assume some distance for point B
variables (ao_extension : O -[A] -> ℰ B)      -- B is on the extension of AO
variables (tangent : angular deflection (angle A O B) = 2 * theta)
variables (tangent_property : tangent_to_circle A B AO)

-- Define sin and cos
variables (sin_2theta : s = real.sin (2 * theta))
variables (cos_2theta : c = real.cos (2 * theta))

-- D is on segment OA, closer to O 
variables (closer_D_O : dist O D < dist O A)

-- BD bisects angle ABO
variables (bisect_angle : is_bisector B D (angle A B O) (angle D B O))

-- The goal is to express OD in terms of s and c
def OD_in_terms_of_s_c (theta s c : ℝ) : ℝ := 
  OD = 2 / (1 + s)

theorem find_OD_expression (rAD rBO ao_extension tangent tangent_property sin_2theta cos_2theta closer_D_O bisect_angle) : 
  OD_in_terms_of_s_c theta s c := by
  sorry

end find_OD_expression_l683_683261


namespace sum_series_value_l683_683100

noncomputable def closest_int_sqrt (n : ℕ) : ℤ := 
  int.of_nat (nat.floor (real.sqrt n))

theorem sum_series_value : 
  (∑' n : ℕ, (3 ^ closest_int_sqrt n + 3 ^ -closest_int_sqrt n) / 3 ^ n) = 4.03703 := 
by 
  simp only 
  sorry

end sum_series_value_l683_683100


namespace number_of_workers_l683_683468

theorem number_of_workers (N C : ℕ) 
  (h1 : N * C = 300000) 
  (h2 : N * (C + 50) = 325000) : 
  N = 500 :=
sorry

end number_of_workers_l683_683468


namespace find_a_smallest_positive_period_maximum_value_l683_683145

variables {α : Type*} [linear_ordered_field α] [topological_space α] [algebra ℝ α]

noncomputable def f (a x : α) := (a / 2) * sin (2 * x) - cos (2 * x)

theorem find_a (a : α) : (f a (π / 8) = 0) → (a = 2) :=
by { sorry }

theorem smallest_positive_period : (sin (2 * (π / 8) - π / 4) = 0) -> ∀ x, f 2 (x + π) = f 2 x  :=
by { sorry }

theorem maximum_value : (sin (2 * (π / 8) - π / 4) = 0) ->  ∀ (x : α), ∃ s, ∀ y, s = sqrt 2 ∧ f 2 y ≤ s :=
by { sorry }

end find_a_smallest_positive_period_maximum_value_l683_683145


namespace minimum_value_f_l683_683144

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.log x

theorem minimum_value_f :
  ∃ x > 0, (∀ y > 0, f x ≤ f y) ∧ f x = 1 :=
sorry

end minimum_value_f_l683_683144


namespace similar_triangles_perimeter_l683_683176

theorem similar_triangles_perimeter (a b c d e f : ℝ) 
  (h_sim : triangle_similar_three a b c d e f) 
  (h_sides_ABC : (a, b, c) = (3, 5, 6).perm)
  (h_shortest_DEF : min_triple d e f = 9)
  : d + e + f = 42 := 
sorry

end similar_triangles_perimeter_l683_683176


namespace infinite_geometric_series_sum_l683_683067

theorem infinite_geometric_series_sum :
  let a := (4 : ℚ) / 3
  let r := -(9 : ℚ) / 16
  (a / (1 - r)) = (64 : ℚ) / 75 :=
by
  sorry

end infinite_geometric_series_sum_l683_683067


namespace cyclic_quadrilateral_circumcircle_l683_683976

theorem cyclic_quadrilateral_circumcircle (A B C D O X Y Z : Type) 
  [Incidence A B C D] [Circumcenter O A B C D] [Intersection X A C B D]
  [Circumcircle A X D] [Circumcircle B X C] [Intersection Y AXD BXC]
  [Circumcircle A X B] [Circumcircle C X D] [Intersection Z AXB CXD]
  (O_inside : O ∈ interior A B C D)
  (O_X_Y_Z_distinct : ∀ P Q R S : Type, P ≠ Q ≠ R ≠ S ≠ P )
  : Cyclic O X Y Z :=
sorry

end cyclic_quadrilateral_circumcircle_l683_683976


namespace quadratic_has_two_distinct_real_roots_l683_683568

theorem quadratic_has_two_distinct_real_roots (k : ℝ) : 
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ r1^2 + 2 * k * r1 + (k - 1) = 0 ∧ r2^2 + 2 * k * r2 + (k - 1) = 0 := 
by 
  sorry

end quadratic_has_two_distinct_real_roots_l683_683568


namespace three_digit_numbers_divisible_by_13_l683_683756

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683756


namespace constant_term_expansion_l683_683205

theorem constant_term_expansion (n : ℕ) (x : ℝ) (h1 : n = 6) (h2 : ∀ r : ℕ, 0 ≤ r ∧ r ≤ n → (nat.choose n r) ≤ (nat.choose n 4)) :
  let T := (nat.choose 6 4) * (x^2)^2 * (1/x)^4
  in T = 15 :=
by
  sorry

end constant_term_expansion_l683_683205


namespace correct_grammatical_phrase_l683_683452

-- Define the conditions as lean definitions 
def number_of_cars_produced_previous_year : ℕ := sorry  -- number of cars produced in previous year
def number_of_cars_produced_2004 : ℕ := 3 * number_of_cars_produced_previous_year  -- number of cars produced in 2004

-- Define the theorem stating the correct phrase to describe the production numbers
theorem correct_grammatical_phrase : 
  (3 * number_of_cars_produced_previous_year = number_of_cars_produced_2004) → 
  ("three times as many cars" = "three times as many cars") := 
by
  sorry

end correct_grammatical_phrase_l683_683452


namespace three_digit_numbers_div_by_13_l683_683726

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683726


namespace number_of_3_digit_divisible_by_13_l683_683805

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683805


namespace unique_involution_l683_683491

noncomputable def f (x : ℤ) : ℤ := sorry

theorem unique_involution (f : ℤ → ℤ) :
  (∀ x : ℤ, f (f x) = x) →
  (∀ x y : ℤ, (x + y) % 2 = 1 → f x + f y ≥ x + y) →
  (∀ x : ℤ, f x = x) :=
sorry

end unique_involution_l683_683491


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683595

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683595


namespace count_three_digit_div_by_13_l683_683692

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683692


namespace barrels_count_l683_683216

noncomputable def cask_capacity : ℕ := 20
noncomputable def barrel_capacity : ℕ := 2 * cask_capacity + 3
noncomputable def total_capacity : ℕ := 172

theorem barrels_count : ∃ B : ℕ, B * barrel_capacity + cask_capacity = total_capacity ∧ B = 3 := by
  use 3
  split
  . show 3 * (2 * cask_capacity + 3) + cask_capacity = total_capacity
    calc 
      3 * (2 * cask_capacity + 3) + cask_capacity
        = 3 * 43 + 20 : by rw [cask_capacity, barrel_capacity]
        = 129 + 20 : by norm_num
        = 172 : by norm_num
  . show 3 = 3
    rfl

end barrels_count_l683_683216


namespace largest_of_three_consecutive_integers_l683_683323

theorem largest_of_three_consecutive_integers (N : ℤ) (h : N + (N + 1) + (N + 2) = 18) : N + 2 = 7 :=
sorry

end largest_of_three_consecutive_integers_l683_683323


namespace kyle_paper_delivery_l683_683238

-- Define the number of houses Kyle delivers to from Monday through Saturday
def housesDaily : ℕ := 100

-- Define the number of days from Monday to Saturday
def daysWeek : ℕ := 6

-- Define the adjustment for Sunday (10 fewer, 30 additional)
def sundayAdjust : ℕ := 30 - 10 + 100

-- Define the total number of papers delivered from Monday to Saturday
def papersMonToSat : ℕ := daysWeek * housesDaily

-- Define the total number of papers delivered on Sunday
def papersSunday : ℕ := sundayAdjust

-- Define the total number of papers delivered each week
def totalPapers : ℕ := papersMonToSat + papersSunday

-- The theorem we want to prove
theorem kyle_paper_delivery : totalPapers = 720 := by
  -- We are focusing only on the statement here.
  sorry

end kyle_paper_delivery_l683_683238


namespace lost_card_number_l683_683378

theorem lost_card_number (n x : ℕ) (sum_n : ℕ) (h_sum : sum_n = n * (n + 1) / 2) (h_remaining_sum : sum_n - x = 101) : x = 4 :=
sorry

end lost_card_number_l683_683378


namespace count_three_digit_numbers_divisible_by_13_l683_683612

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683612


namespace count_3_digit_numbers_divisible_by_13_l683_683716

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683716


namespace rise_in_height_of_field_l683_683408

theorem rise_in_height_of_field
  (field_length : ℝ)
  (field_width : ℝ)
  (pit_length : ℝ)
  (pit_width : ℝ)
  (pit_depth : ℝ)
  (field_area : ℝ := field_length * field_width)
  (pit_area : ℝ := pit_length * pit_width)
  (remaining_area : ℝ := field_area - pit_area)
  (pit_volume : ℝ := pit_length * pit_width * pit_depth)
  (rise_in_height : ℝ := pit_volume / remaining_area) :
  field_length = 20 →
  field_width = 10 →
  pit_length = 8 →
  pit_width = 5 →
  pit_depth = 2 →
  rise_in_height = 0.5 :=
by
  intros
  sorry

end rise_in_height_of_field_l683_683408


namespace lost_card_number_l683_683372

theorem lost_card_number (n : ℕ) (x : ℕ) (h₁ : (n * (n + 1)) / 2 - x = 101) : x = 4 :=
sorry

end lost_card_number_l683_683372


namespace are_concyclic_l683_683259

open Segment

-- Given (Conditions as definitions)
variables {A B C D P Q R S M N : Point}
variable {ABCD_convex : ConvexQuadrilateral A B C D}
variable {angle_DAB : ∠DAB = 90}
variable {angle_BCD : ∠BCD = 90}
variable {angle_ABC_gt_angle_CDA : ∠ABC > ∠CDA}
variable {Q_on_BC : OnSegment Q (Segment.mk B C)}
variable {R_on_CD : OnSegment R (Segment.mk C D)}
variable {P_on_AB : OnLine P (Line.mk A B)}
variable {S_on_AD : OnLine S (Line.mk A D)}
variable {PQ_eq_RS : Distance P Q = Distance R S}
variable {M_midpoint_BD : Midpoint M (Segment.mk B D)}
variable {N_midpoint_QR : Midpoint N (Segment.mk Q R)}

-- Prove (Question == Answer)
theorem are_concyclic : Concyclic A N M C := sorry

end are_concyclic_l683_683259


namespace number_of_puppies_l683_683278

def total_portions : Nat := 105
def feeding_days : Nat := 5
def feedings_per_day : Nat := 3

theorem number_of_puppies (total_portions feeding_days feedings_per_day : Nat) : 
  (total_portions / feeding_days / feedings_per_day = 7) := 
by 
  sorry

end number_of_puppies_l683_683278


namespace set_union_example_l683_683574

theorem set_union_example (a b : ℝ) (h₁ : ∀ x ∈ ({5, real.log 2 a} : set ℝ), x = 1) (h₂ : {5, real.log 2 a} ∩ ({a, b} : set ℝ) = {1}) :
  {5, 1} ∪ {2, 1} = {1, 2, 5} :=
by {
  sorry,
}

end set_union_example_l683_683574


namespace solution_of_system_of_inequalities_l683_683518

theorem solution_of_system_of_inequalities (p : ℝ) :
  19 * p < 10 ∧ p > 0.5 ↔ (1 / 2) < p ∧ p < (10 / 19) :=
by
  sorry

end solution_of_system_of_inequalities_l683_683518


namespace six_power_six_div_two_l683_683406

theorem six_power_six_div_two : 6 ^ (6 / 2) = 216 := by
  sorry

end six_power_six_div_two_l683_683406


namespace count_3_digit_numbers_divisible_by_13_l683_683666

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683666


namespace system_of_inequalities_solution_l683_683515

theorem system_of_inequalities_solution (p : ℝ) (h1 : 19 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 10 / 19 := by
  sorry

end system_of_inequalities_solution_l683_683515


namespace count_3digit_numbers_div_by_13_l683_683895

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683895


namespace num_pos_3_digit_div_by_13_l683_683763

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683763


namespace largest_of_three_consecutive_integers_l683_683321

theorem largest_of_three_consecutive_integers (N : ℤ) (h : N + (N + 1) + (N + 2) = 18) : N + 2 = 7 :=
sorry

end largest_of_three_consecutive_integers_l683_683321


namespace count_3_digit_numbers_divisible_by_13_l683_683700

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683700


namespace probability_gt_9_probability_lt_6_three_times_l683_683350

/-
  Given:
  1. We have two dice, each with 6 faces.
  2. All outcomes for rolling two dice together are independent and uniformly distributed.

  Prove:
  1. The probability of rolling a sum greater than 9 in a single roll of two dice is 1/6.
  2. The probability of rolling a sum less than 6 three times in a row with two dice is 125/5832.
-/

def two_dice_six_faces : finset (ℕ × ℕ) := finset.product (finset.range 1 7) (finset.range 1 7)

def favorable_outcomes_gt_9 (s : ℕ × ℕ) : bool := s.1 + s.2 > 9

def favorable_outcomes_lt_6 (s : ℕ × ℕ) : bool := s.1 + s.2 < 6

theorem probability_gt_9 :
  (finset.filter favorable_outcomes_gt_9 two_dice_six_faces).card.to_rat / two_dice_six_faces.card.to_rat = 1 / 6 :=
sorry

theorem probability_lt_6_three_times :
  ((finset.filter favorable_outcomes_lt_6 two_dice_six_faces).card.to_rat / two_dice_six_faces.card.to_rat) ^ 3 = 125 / 5832 :=
sorry

end probability_gt_9_probability_lt_6_three_times_l683_683350


namespace number_of_3_digit_divisible_by_13_l683_683814

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683814


namespace angle_sum_l683_683423

variables (A B C D E F : Type) 
variables [geometry A] [geometry B] [geometry C] [geometry D] [geometry E] [geometry F]

-- Given conditions:
variable (isosceles_ABC : is_isosceles_triangle ABC AB AC)
variable (isosceles_DEF : is_isosceles_triangle DEF DE DF)
variable (angle_BAC : angle BAC = 25)
variable (angle_EDF : angle EDF = 35)

-- The problem statement to be proved:
theorem angle_sum :
  let angle_DAC := ∠ DAC
  let angle_ADE := ∠ ADE
  angle_DAC + angle_ADE = 150 :=
begin
  sorry
end

end angle_sum_l683_683423


namespace no_polyhedron_with_surface_2015_l683_683997

/--
It is impossible to glue together 1 × 1 × 1 cubes to form a polyhedron whose surface area is 2015.
-/
theorem no_polyhedron_with_surface_2015 (n k : ℕ) : 6 * n - 2 * k ≠ 2015 :=
by
  sorry

end no_polyhedron_with_surface_2015_l683_683997


namespace compare_doubling_l683_683961

theorem compare_doubling (a b : ℝ) (h : a > b) : 2 * a > 2 * b :=
  sorry

end compare_doubling_l683_683961


namespace num_3_digit_div_by_13_l683_683830

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683830


namespace count_three_digit_div_by_13_l683_683697

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683697


namespace lost_card_l683_683366

noncomputable def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

theorem lost_card (n : ℕ) (h1 : sum_natural n - 101 < n) (h2 : sum_natural (n - 1) ≤ 101) :
  ∃ x : ℕ, x = sum_natural n - 101 ∧ x = 4 :=
begin
  have h_sum : sum_natural 14 = 105,
  {
    unfold sum_natural,
    norm_num,
  },
  use 4,
  split,
  {
    unfold sum_natural at *,
    norm_num,
    rw h_sum,
  },
  {
    norm_num,
  },
  sorry,
end

end lost_card_l683_683366


namespace find_lost_card_number_l683_683389

theorem find_lost_card_number (n : ℕ) (S : ℕ) (x : ℕ) 
  (h1 : S = n * (n + 1) / 2) 
  (h2 : S - x = 101) 
  (h3 : n = 14) : 
  x = 4 := 
by sorry

end find_lost_card_number_l683_683389


namespace count_3_digit_numbers_divisible_by_13_l683_683872

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683872


namespace greatest_integer_a_exists_l683_683079

theorem greatest_integer_a_exists (a x : ℤ) (h : (x - a) * (x - 7) + 3 = 0) : a ≤ 11 := by
  sorry

end greatest_integer_a_exists_l683_683079


namespace Makenna_vegetable_garden_larger_l683_683230

theorem Makenna_vegetable_garden_larger :
  let A_Karl := 30 * 50 in
  let A_Karl_veg := A_Karl - 300 in
  let A_Makenna := 35 * 45 in
  A_Makenna - A_Karl_veg = 375 :=
by
  sorry

end Makenna_vegetable_garden_larger_l683_683230


namespace count_three_digit_numbers_divisible_by_13_l683_683932

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683932


namespace number_of_3_digit_divisible_by_13_l683_683806

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683806


namespace coeff_of_x_in_expr_l683_683989

noncomputable def expr := (x^2 + 3*x + 2)^5

theorem coeff_of_x_in_expr : coeff expr 1 = 240 :=
sorry

end coeff_of_x_in_expr_l683_683989


namespace minimal_volley_hits_anyas_triangle_l683_683031

-- Definition of conditions
def points_on_circle (n : ℕ) : Prop := (n > 0)

def anya_triangle (vertices : Finset ℕ) : Prop := 
  vertices.card = 3 ∧ ∀ v ∈ vertices, v ∈ Finset.range 29

def distinct_shots (shots : Finset (ℕ × ℕ)) : Prop :=
  shots.card = 100 ∧ ∀ (k m : ℕ), (k, m) ∈ shots → k ≠ m ∧ k < 29 ∧ m < 29

-- The statement we want to prove
theorem minimal_volley_hits_anyas_triangle :
  ∀ (vertices : Finset ℕ) (shots : Finset (ℕ × ℕ)),
  points_on_circle 29 →
  anya_triangle vertices →
  distinct_shots shots →
  ∃ (k m : ℕ), (k, m) ∈ shots ∧
  ∀ (a b c : ℕ), a ∈ vertices → b ∈ vertices → c ∈ vertices → 
  segment_intersects_triangle (a, b, c) (k, m) := 
sorry


end minimal_volley_hits_anyas_triangle_l683_683031


namespace total_people_l683_683199

-- Given definitions
def students : ℕ := 37500
def ratio_students_professors : ℕ := 15
def professors : ℕ := students / ratio_students_professors

-- The statement to prove
theorem total_people : students + professors = 40000 := by
  sorry

end total_people_l683_683199


namespace p1_suff_not_necess_q1_p2_necess_not_suff_q2_p3_necess_suff_q3_l683_683173

-- Definitions for conditions
variable (x : ℝ) (c a b : ℝ)
def p1 := 0 < x ∧ x < 3
def q1 := abs (x - 1) < 2

def p2 := (x - 2) * (x - 3) = 0
def q2 := x = 2

def p3 := c = 0
def q3 := ∃ a b, y = a * x ^ 2 + b * x + c ∧ y = 0 ∧ x = 0

-- Proof statements
theorem p1_suff_not_necess_q1 : p1 x → q1 x ∧ ¬(q1 x → p1 x) := by
  sorry

theorem p2_necess_not_suff_q2 : q2 x → p2 x ∧ ¬(p2 x → q2 x) := by
  sorry

theorem p3_necess_suff_q3 : (p3 c ↔ q3 c) := by
  sorry

end p1_suff_not_necess_q1_p2_necess_not_suff_q2_p3_necess_suff_q3_l683_683173


namespace four_consecutive_integers_product_2520_l683_683527

theorem four_consecutive_integers_product_2520 {a b c d : ℕ}
  (h1 : a + 1 = b) 
  (h2 : b + 1 = c) 
  (h3 : c + 1 = d) 
  (h4 : a * b * c * d = 2520) : 
  a = 6 := 
sorry

end four_consecutive_integers_product_2520_l683_683527


namespace six_digit_palindromes_count_l683_683339

theorem six_digit_palindromes_count : 
  ∃ n : ℕ, n = 27 ∧ 
  (∀ (A B C : ℕ), 
       (A = 6 ∨ A = 7 ∨ A = 8) ∧ 
       (B = 6 ∨ B = 7 ∨ B = 8) ∧ 
       (C = 6 ∨ C = 7 ∨ C = 8) → 
       ∃ p : ℕ, 
         p = (A * 10^5 + B * 10^4 + C * 10^3 + C * 10^2 + B * 10 + A) ∧ 
         (6 ≤ p / 10^5 ∧ p / 10^5 ≤ 8) ∧ 
         (6 ≤ (p / 10^4) % 10 ∧ (p / 10^4) % 10 ≤ 8) ∧ 
         (6 ≤ (p / 10^3) % 10 ∧ (p / 10^3) % 10 ≤ 8)) :=
  by sorry

end six_digit_palindromes_count_l683_683339


namespace num_pos_3_digit_div_by_13_l683_683759

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683759


namespace relay_go_match_outcomes_l683_683286

theorem relay_go_match_outcomes : (Nat.choose 14 7) = 3432 := by
  sorry

end relay_go_match_outcomes_l683_683286


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683594

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683594


namespace solution_of_system_of_inequalities_l683_683519

theorem solution_of_system_of_inequalities (p : ℝ) :
  19 * p < 10 ∧ p > 0.5 ↔ (1 / 2) < p ∧ p < (10 / 19) :=
by
  sorry

end solution_of_system_of_inequalities_l683_683519


namespace problem_statement_l683_683140

variable {α : Type*} [OrderedCommRing α]

def even_function (f : α → α) : Prop := ∀ x, f (-x) = f x
def monotonically_decreasing_on_nonneg (f : α → α) : Prop := ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem problem_statement (f : α → α) (h_even : even_function f) (h_monotone : monotonically_decreasing_on_nonneg f) :
  f 1 > f (-6) :=
by
  have h1 := h_even 6
  have h2 := h_monotone 1 6 zero_le_one (by norm_num) (by norm_num)
  simp only [h1] at h2
  exact h2

end problem_statement_l683_683140


namespace count_3_digit_numbers_divisible_by_13_l683_683914

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683914


namespace number_of_three_digit_numbers_divisible_by_13_l683_683940

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683940


namespace sum_of_cubes_divisible_by_3_l683_683245

theorem sum_of_cubes_divisible_by_3 :
  ∀ (r_1 r_2 r_3 : ℝ), (∃ (h : r_1 ≠ r_2 ∧ r_2 ≠ r_3 ∧ r_1 ≠ r_3),
  polynomial.eval r_1 (polynomial.C 1 * (polynomial.X ^ 3) - polynomial.C 2019 * (polynomial.X ^ 2) - polynomial.C 2020 * polynomial.X + polynomial.C 2021) = 0 ∧
  polynomial.eval r_2 (polynomial.C 1 * (polynomial.X ^ 3) - polynomial.C 2019 * (polynomial.X ^ 2) - polynomial.C 2020 * polynomial.X + polynomial.C 2021) = 0 ∧
  polynomial.eval r_3 (polynomial.C 1 * (polynomial.X ^ 3) - polynomial.C 2019 * (polynomial.X ^ 2) - polynomial.C 2020 * polynomial.X + polynomial.C 2021) = 0) →
  (r_1 ^ 3 + r_2 ^ 3 + r_3 ^ 3) % 3 = 0 := sorry

end sum_of_cubes_divisible_by_3_l683_683245


namespace evaluate_series_l683_683094

noncomputable def closest_int_sqrt (n : ℕ) : ℤ :=
  round (real.sqrt n)

theorem evaluate_series : 
  (∑' (n : ℕ) in set.Icc 1 (nat.mul (set.Ioi 1)) (3 ^ (closest_int_sqrt n) + 3 ^ -(closest_int_sqrt n)) / 3 ^ n) = 3 :=
begin
  sorry
end

end evaluate_series_l683_683094


namespace count_3_digit_multiples_of_13_l683_683787

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683787


namespace num_pos_3_digit_div_by_13_l683_683765

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683765


namespace find_f_ln_one_third_l683_683121

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (2^x) / ((2^x) + 1) + a * x

-- Main theorem statement
theorem find_f_ln_one_third (a : ℝ) (h1 : f a (Real.log 3) = 2) : f a (Real.log (1/3)) = -1 := 
by 
  -- The proof is omitted
  sorry

end find_f_ln_one_third_l683_683121


namespace xunzi_statement_l683_683363

/-- 
Given the conditions:
  "If not accumulating small steps, then not reaching a thousand miles."
  Which can be represented as: ¬P → ¬q.
Prove that accumulating small steps (P) is a necessary but not sufficient condition for
reaching a thousand miles (q).
-/
theorem xunzi_statement (P q : Prop) (h : ¬P → ¬q) : (q → P) ∧ ¬(P → q) :=
by sorry

end xunzi_statement_l683_683363


namespace intersection_lies_on_CD_l683_683201

open Set
open Function

variables {A B C D U V P Q S R: Type} [Pointed A] [Pointed B] [Pointed C] [Pointed D]
          [Pointed U] [Pointed V] [Pointed P] [Pointed Q] [Pointed S] [Pointed R]

-- Definitions for conditions
def diagonals_perpendicular (AC BD: Set Type) : Prop := 
  ∃ (U: Type), ∃ (V: Type), (U ∈ AC) ∧ (V ∈ BD) ∧ (perpendicular AC BD)

def lies_on (P: Type) (line: Set Type) : Prop := P ∈ line
def intersection (line1 line2: Set Type) (X: Type) : Prop := X ∈ line1 ∧ X ∈ line2

-- Assert the conditions
axiom h1 : quadrilateral A B C D
axiom h2 : diagonals_perpendicular (AC) (BD)
axiom h3 : lies_on P (line_through A B)
axiom h4 : intersection (line_through P U) (line_through B C) Q
axiom h5 : intersection (line_through P V) (line_through A D) S

-- Theorem statement
theorem intersection_lies_on_CD : 
  ∃ (R: Type), intersection (line_through Q V) (line_through S U) R → lies_on R (line_through C D) :=
sorry

end intersection_lies_on_CD_l683_683201


namespace Garys_hourly_wage_l683_683434

-- Define the conditions as constants
constant total_hours : ℕ := 52
constant overtime_threshold : ℕ := 40
constant overtime_rate : ℚ := 1.5
constant total_paycheck : ℚ := 696

-- Define Gary's normal hourly wage as x
variable (x : ℚ)

-- Define the equations based on the conditions
def normal_hours_pay := 40 * x
def overtime_hours_pay := (total_hours - overtime_threshold) * (overtime_rate * x)
def total_earnings := normal_hours_pay + overtime_hours_pay

-- The main statement to prove
theorem Garys_hourly_wage : total_earnings = total_paycheck → x = 12 :=
by
  sorry

end Garys_hourly_wage_l683_683434


namespace complement_union_A_B_in_U_l683_683264

open Set Nat

def U : Set ℕ := { x | x < 6 ∧ x > 0 }
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_union_A_B_in_U : (U \ (A ∪ B)) = {2, 4} := by
  sorry

end complement_union_A_B_in_U_l683_683264


namespace count_3digit_numbers_div_by_13_l683_683882

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683882


namespace three_digit_numbers_div_by_13_l683_683731

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683731


namespace number_of_three_digit_numbers_divisible_by_13_l683_683939

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683939


namespace three_digit_numbers_divisible_by_13_l683_683749

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683749


namespace translated_line_equation_l683_683335

def linear_function (x : ℝ) : ℝ := x - 2

def point := (2 : ℝ, 3 : ℝ)

theorem translated_line_equation : 
    ∃ (b : ℝ), (∀ x : ℝ, let y := x + b in y = 3 → x = 2) 
               ∧ (∀ x : ℝ, let y := x + b in y = f x → x = 2)
               ∧ b = 1 := 
sorry

end translated_line_equation_l683_683335


namespace probability_of_face_and_number_sum_14_l683_683465

theorem probability_of_face_and_number_sum_14 :
  let deck := (fin 52) in
  let face_cards := {card ∈ deck | card = 11 ∨ card = 12 ∨ card = 13 ∨ card = 24 ∨ card = 25 ∨ card = 26 ∨ card = 37 ∨ card = 38 ∨ card = 39 ∨ card = 50 ∨ card = 51 ∨ card = 52} in
  let number_cards := {card ∈ deck | card = 2 ∨ card = 3 ∨ card = 4 ∨ card = 5 ∨ card = 6 ∨ card = 7 ∨ card = 8 ∨ card = 9 ∨ card = 10} in
  let total_value card := if card < 10 then card else if card <= 13 then 10 else if card <= 26 then card - 13 else if card <= 39 then card - 26 else card - 39 in
  (Σ (c1 c2 ∈ deck) (h1: c1 ≠ c2) (h2: total_value c1 + total_value c2 = 14 ) (h3: (c1 ∈ face_cards ∧ c2 ∈ number_cards) ∨ (c1 ∈ number_cards ∧ c2 ∈ face_cards)), 1) / (52 * 51) = 16/1326 :=
sorry

end probability_of_face_and_number_sum_14_l683_683465


namespace monotonically_increasing_interval_l683_683300

theorem monotonically_increasing_interval :
  ∀ x, (0 ≤ x ∧ x ≤ 2 * Real.pi) → 0 ≤ x ∧ x ≤ Real.pi → 
    MonotonicallyIncreasing (fun x => 2019 * Real.sin (1 / 3 * x + Real.pi / 6)) :=
by
  intros x h1 h2
  -- To be proved
  sorry

end monotonically_increasing_interval_l683_683300


namespace count_3_digit_numbers_divisible_by_13_l683_683675

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683675


namespace shaded_area_exclusion_l683_683437

noncomputable def total_shaded_area (width length : ℝ) (circle_radius : ℝ) : ℝ :=
  let rect_area := width * length
  let circle_area := 4 * (Math.pi * circle_radius^2)
  rect_area - circle_area

theorem shaded_area_exclusion (width : ℝ) (h_w : width = 30) 
  (length : ℝ) (h_l : length = 2 * width) 
  (radius : ℝ) (h_r : radius = width / 4) : 
  total_shaded_area width length radius = 1800 - 225 * Math.pi :=
by
  sorry

end shaded_area_exclusion_l683_683437


namespace count_3_digit_numbers_divisible_by_13_l683_683867

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683867


namespace father_age_triple_weiwei_l683_683341

theorem father_age_triple_weiwei (x : ℕ) (weiwei_age_current : ℕ) (father_age_current : ℕ)
    (h1 : weiwei_age_current = 8) (h2 : father_age_current = 34) :
    father_age_current + x = 3 * (weiwei_age_current + x) → x = 5 :=
by
  intros h
  rw [h1, h2] at h
  norm_num at h
  exact h

end father_age_triple_weiwei_l683_683341


namespace count_3_digit_numbers_divisible_by_13_l683_683673

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683673


namespace length_CF_l683_683255

-- Define the problem: properties of triangle, midpoints, perpendicular medians, lengths

variables (A B C D E F G : Type) [MetricSpace A B C D E F G]

-- Conditions of the problem
axiom is_triangle : triangle A B C
axiom midpoint_BC : midpoint B C D
axiom midpoint_AC : midpoint A C E
axiom midpoint_AB : midpoint A B F
axiom medians_perpendicular : perpendicular (median A D) (median B E)
axiom length_AD : distance A D = 18
axiom length_BE : distance B E = 13.5

-- Question: Prove that the length of the third median CF is 22.5
theorem length_CF : distance C F = 22.5 :=
sorry

end length_CF_l683_683255


namespace count_integer_coordinate_points_l683_683020

/-- Define point C and point D in the coordinate plane. --/
structure Point where
  x : Int
  y : Int

def C : Point := {x := -4, y := 3}
def D : Point := {x := 4, y := -3}

def ManhattanDistance (p1 p2 : Point) : Int :=
  Int.abs (p1.x - p2.x) + Int.abs (p1.y - p2.y)

def validPathLength : Int := 22

/-- The main theorem to determine the count of integer-coordinate points 
    that the robot can visit on at least one of its valid paths. --/
theorem count_integer_coordinate_points : ∃ (n : Int), n = 225 ∧ ∀ (P : Point),
  ManhattanDistance C P + ManhattanDistance P D ≤ validPathLength →
  isIntegerCoordinatePoint P := by
  sorry

end count_integer_coordinate_points_l683_683020


namespace distance_from_P_to_other_focus_on_ellipse_l683_683541

noncomputable def distance_to_other_focus (P : ℝ × ℝ) : ℝ :=
  let a := 5 in   -- derived from a^2 = 25
  2 * a - 4

theorem distance_from_P_to_other_focus_on_ellipse {x y : ℝ}
  (h_ellipse : x^2 / 25 + y^2 / 16 = 1)
  (h_distance : ∃ P1, (x = 5 * real.cos P1) ∧ (y = 4 * real.sin P1) ∧ dist (x, y) (5, 0) = 4) :
  distance_to_other_focus (x, y) = 6 :=
sorry

end distance_from_P_to_other_focus_on_ellipse_l683_683541


namespace even_function_b_eq_one_monotone_f_range_m_l683_683129

variable (a b : ℝ)

-- Condition: Even function f(x) = a^x + b * a^(-x)
def f (x : ℝ) : ℝ := a^x + b * a^(-x)

-- Questions
-- Question 1: Prove that b = 1 given f(x) is even
theorem even_function_b_eq_one (h_even : ∀ x : ℝ, f x = f (-x)) (h_pos : a > 0) (h_ne : a ≠ 1) : b = 1 :=
sorry

-- Question 2: Prove the monotonicity of f(x), given b = 1
theorem monotone_f (h_pos : a > 0) (h_ne : a ≠ 1) : ∀ {x1 x2 : ℝ}, 0 ≤ x1 → x1 < x2 → f x1 < f x2 :=
sorry

-- Question 3: Find the range of m such that 
-- ∀ x ∈ [2, 4], f((log 2 x) ^ 2 - log 2 x + 1) ≥ f(m + log (1/2) x ^ 2)
theorem range_m (h_pos : a > 0) (h_ne : a ≠ 1) (h_b : b = 1) :
  ∀ m : ℝ, (∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → 
    f ((Real.log 2 x) ^ 2 - Real.log 2 x + 1) ≥ f (m + Real.log (1/2 : ℝ) x ^ 2)) ↔ 
    (m ∈ Set.Icc (5 / 4 : ℝ) 3) :=
sorry

end even_function_b_eq_one_monotone_f_range_m_l683_683129


namespace count_3_digit_numbers_divisible_by_13_l683_683906

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683906


namespace count_3_digit_numbers_divisible_by_13_l683_683660

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683660


namespace min_value_of_f_is_neg_one_l683_683558

-- Define the function
def f (x : ℝ) : ℝ := 2^x + 1 / 2^(x + 2)

-- Statement for the proof problem
theorem min_value_of_f_is_neg_one : ∀ x : ℝ, f(-1) ≤ f(x) :=
by
  sorry

end min_value_of_f_is_neg_one_l683_683558


namespace train_and_wagon_number_l683_683415

def cipher_numbers (С О Е К Р Т : ℕ) : Prop :=
  С = О + 2 ∧ Е - Т = 0 ∧ (С * 100000 + Е * 10000 + К * 1000 + Р * 100 + Е * 10 + Т) -
  (О * 100000 + Т * 10000 + К * 1000 + Р * 100 + О * 10 + Т) = 20010
  ∧ Т = 9 ∧ О = 2 ∧ Е = 0

theorem train_and_wagon_number (С О Е К Р Т : ℕ) :
  cipher_numbers С О Е К Р Т → С * 100 + К * 10 + Р = 392 ∧ К = 3 ∧ Р = 2 :=
by
  intros h
  sorry

end train_and_wagon_number_l683_683415


namespace cyclist_motorcyclist_intersection_l683_683413

theorem cyclist_motorcyclist_intersection : 
  ∃ t : ℝ, (4 * t^2 + (t - 1)^2 - 2 * |t| * |t - 1| = 49) ∧ (t = 4 ∨ t = -4) := 
by 
  sorry

end cyclist_motorcyclist_intersection_l683_683413


namespace sin_alpha_on_ray_l683_683187

theorem sin_alpha_on_ray (x y : ℝ) (r : ℝ) (h_ray : 3 * x + 4 * y = 0)
  (h_x_pos : 0 < x) (h_r : r = Real.sqrt (x^2 + y^2)) : 
  (y / r = -3/5) :=
begin
  -- We state but won't prove the theorem
  sorry
end

end sin_alpha_on_ray_l683_683187


namespace number_of_3_digit_divisible_by_13_l683_683804

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683804


namespace count_3_digit_numbers_divisible_by_13_l683_683669

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683669


namespace functional_relation_proof_profit_eq_proof_max_profit_proof_l683_683002

variables (x y : ℕ)

-- Question 1: Functional relationship between sales volume y and selling price x
def sales_volume_relation : Prop := y = 1000 - 10 * x

-- Conditions
def unit_cost : ℕ := 30
def initial_selling_price : ℕ := 40
def initial_sales_volume : ℕ := 600
def price_increase : ℕ := 1
def sales_volume_decrease : ℕ := 10

-- Question 2: Selling price for a profit of 10,000 yuan
def profit_eq_10000 (x : ℕ) : Prop := (x - unit_cost) * (1000 - 10 * x) = 10000

-- Question 3: Maximum profit with constraints
def profit_function (x : ℕ) : ℕ := -10 * x^2 + 1300 * x - 30000
def max_profit_with_constraints : Prop :=
  44 ≤ x ∧ x ≤ 46 → profit_function x = 8640

-- Lean 4 theorem statements without proofs
theorem functional_relation_proof : sales_volume_relation y x := sorry

theorem profit_eq_proof (x : ℕ) : profit_eq_10000 x → x = 50 ∨ x = 80 := sorry

theorem max_profit_proof (x : ℕ) : max_profit_with_constraints x := sorry

end functional_relation_proof_profit_eq_proof_max_profit_proof_l683_683002


namespace three_digit_numbers_div_by_13_l683_683732

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683732


namespace probability_A_selected_is_three_fourths_l683_683528

-- Definition and the theorem based on the given conditions and question

noncomputable def total_events : ℕ := (nat.choose 4 3)
noncomputable def favorable_events : ℕ := (nat.choose 1 1) * (nat.choose 3 2)
noncomputable def probability_A_selected : ℚ := favorable_events / total_events

theorem probability_A_selected_is_three_fourths : probability_A_selected = 3 / 4 := 
by
  sorry

end probability_A_selected_is_three_fourths_l683_683528


namespace count_three_digit_numbers_divisible_by_13_l683_683609

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683609


namespace translate_parabola_up_four_units_l683_683336

theorem translate_parabola_up_four_units :
  ∀ (x : ℝ), (x^2 + 4) = (translate_up y (4)) := 
by
  sorry

end translate_parabola_up_four_units_l683_683336


namespace solve_for_x_l683_683530

-- Lean 4 statement for the problem
theorem solve_for_x (x : ℝ) (h : (x + 1)^3 = -27) : x = -4 := by
  sorry

end solve_for_x_l683_683530


namespace count_three_digit_numbers_divisible_by_13_l683_683923

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683923


namespace count_three_digit_numbers_divisible_by_13_l683_683615

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683615


namespace sum_series_value_l683_683101

noncomputable def closest_int_sqrt (n : ℕ) : ℤ := 
  int.of_nat (nat.floor (real.sqrt n))

theorem sum_series_value : 
  (∑' n : ℕ, (3 ^ closest_int_sqrt n + 3 ^ -closest_int_sqrt n) / 3 ^ n) = 4.03703 := 
by 
  simp only 
  sorry

end sum_series_value_l683_683101


namespace train_trip_distance_l683_683449

theorem train_trip_distance (x D : ℝ) :
  -- Conditions of the problem
  (forall t1 t2 t3 t4 : ℝ,
    (t1 = 1) ->  -- The train runs for 1 hour before the accident.
    (t2 = 0.5) ->  -- The accident detains the train for half an hour.
    (t3 = 1.5 + (4 * (D - x)) / (3 * x)) ->  -- Actual total travel time of the first scenario.
    (t3 = 4 + 3.5) ->  -- Total travel time including delay is 4 hours and train is 3.5 hours late.
    (t4 = x + 90) ->  -- If the accident had happened 90 miles farther.
    -- Solving for first scenario:
    let delay1 := t3 = 1.5 + (4 * (D - x)) / (3 * x), let total_time1 := t3 - 3.5,
    -- Solving for second scenario:
    let delay2 := t4 = (x + 90) / x + (4 * (D - (x + 90))) / (3 * x) + 0.5, let total_time2 := t4 - 3,
    -- Set up the equation:
    ((90 / x) - (90 / (3 * x / 4)) + (0.5) = 0) -> 
    (x = 60) -> -- Solving the equation yields the speed x
    (D = 600)) -- Using this speed yelds the distance D
  → (D = 600) := sorry

end train_trip_distance_l683_683449


namespace minimum_number_of_blocs_l683_683269

theorem minimum_number_of_blocs (states : ℕ) (max_states_per_bloc : ℕ) (min_blocs_cover : ℕ) 
  (h1 : states = 100) (h2 : max_states_per_bloc = 50) (h3 : min_blocs_cover = 3) :
  ∃ m, (∀ (B : finset (fin 100)), B.card ≤ 50) ∧ (∀ (s₁ s₂ : fin 100), ∃ B, s₁ ∈ B ∧ s₂ ∈ B) ∧ m = 6 :=
sorry

end minimum_number_of_blocs_l683_683269


namespace initial_volume_proof_l683_683192

-- Definitions for initial mixture and ratios
variables (x : ℕ)

def initial_milk := 4 * x
def initial_water := x
def initial_volume := initial_milk x + initial_water x

def add_water (water_added : ℕ) := initial_water x + water_added

def resulting_ratio := initial_milk x / add_water x 9 = 2

theorem initial_volume_proof (h : resulting_ratio x) : initial_volume x = 45 :=
by sorry

end initial_volume_proof_l683_683192


namespace count_3_digit_numbers_divisible_by_13_l683_683674

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683674


namespace folded_paper_holes_l683_683017

theorem folded_paper_holes :
  let paper := "rectangular piece of paper" in
  let fold_1 := "bottom to top" in
  let fold_2 := "left to right" in
  let fold_3 := "top to bottom" in
  let punch := "punch at top left corner" in
  holes_in_paper_after_unfolding paper [fold_1, fold_2, fold_3] punch = 4 :=
begin
  sorry
end

end folded_paper_holes_l683_683017


namespace count_three_digit_numbers_divisible_by_13_l683_683607

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683607


namespace remainder_consec_even_div12_l683_683351

theorem remainder_consec_even_div12 (n : ℕ) (h: n % 2 = 0)
  (h1: 11234 ≤ n ∧ n + 12 ≥ 11246) : 
  (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) + (n + 12)) % 12 = 6 :=
by 
  sorry

end remainder_consec_even_div12_l683_683351


namespace num_3_digit_div_by_13_l683_683823

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683823


namespace three_digit_numbers_divisible_by_13_l683_683755

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683755


namespace unique_solution_eq_l683_683526

noncomputable def equation := ∀ (x a : ℝ), (x^2 - a)^2 + 2*(x^2 - a) + (x - a) + 2 = 0

theorem unique_solution_eq (x a : ℝ) : (equation x a) ↔ a = 3 / 4 := by
  sorry

end unique_solution_eq_l683_683526


namespace g_at_10_l683_683262

noncomputable def g : ℕ → ℝ :=
sorry

axiom g_1 : g 1 = 2
axiom g_prop (m n : ℕ) (hmn : m ≥ n) : g (m + n) + g (m - n) = 2 * (g m + g n)

theorem g_at_10 : g 10 = 200 := 
sorry

end g_at_10_l683_683262


namespace range_of_a_l683_683148

noncomputable def f (a x : ℝ) := Real.logb (1 / 2) (x^2 - a * x - a)

theorem range_of_a :
  (∀ a : ℝ, (∀ x : ℝ, f a x ∈ Set.univ) ∧ 
            (∀ x1 x2 : ℝ, -3 < x1 ∧ x1 < x2 ∧ x2 < 1 - Real.sqrt 3 → f a x1 < f a x2)) → 
  (0 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_a_l683_683148


namespace count_3_digit_numbers_divisible_by_13_l683_683860

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683860


namespace flowers_in_each_basket_l683_683053

-- Definitions based on the conditions
def initial_flowers (d1 d2 : Nat) : Nat := d1 + d2
def grown_flowers (initial growth : Nat) : Nat := initial + growth
def remaining_flowers (grown dead : Nat) : Nat := grown - dead
def flowers_per_basket (remaining baskets : Nat) : Nat := remaining / baskets

-- Given conditions in Lean 4
theorem flowers_in_each_basket 
    (daughters_flowers : Nat) 
    (growth : Nat) 
    (dead : Nat) 
    (baskets : Nat) 
    (h_daughters : daughters_flowers = 5 + 5) 
    (h_growth : growth = 20) 
    (h_dead : dead = 10) 
    (h_baskets : baskets = 5) : 
    flowers_per_basket (remaining_flowers (grown_flowers (initial_flowers 5 5) growth) dead) baskets = 4 := 
sorry

end flowers_in_each_basket_l683_683053


namespace num_pos_3_digit_div_by_13_l683_683769

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683769


namespace find_a9_l683_683207

-- Define the geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Given conditions of the problem
variables {a : ℕ → ℝ}
axiom h_geom_seq : is_geometric_sequence a
axiom h_root1 : a 3 * a 15 = 1
axiom h_root2 : a 3 + a 15 = -4

-- The proof statement
theorem find_a9 : a 9 = 1 := 
by sorry

end find_a9_l683_683207


namespace figure_Z_has_largest_shaded_area_l683_683332

noncomputable def shaded_area_X :=
  let rectangle_area := 4 * 2
  let circle_area := Real.pi * (1)^2
  rectangle_area - circle_area

noncomputable def shaded_area_Y :=
  let rectangle_area := 4 * 2
  let semicircle_area := (1 / 2) * Real.pi * (1)^2
  rectangle_area - semicircle_area

noncomputable def shaded_area_Z :=
  let outer_square_area := 4^2
  let inner_square_area := 2^2
  outer_square_area - inner_square_area

theorem figure_Z_has_largest_shaded_area :
  shaded_area_Z > shaded_area_X ∧ shaded_area_Z > shaded_area_Y :=
by
  sorry

end figure_Z_has_largest_shaded_area_l683_683332


namespace count_three_digit_numbers_divisible_by_13_l683_683927

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683927


namespace least_five_digit_congruent_to_7_mod_18_l683_683346

theorem least_five_digit_congruent_to_7_mod_18 : 
  ∃ n, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 18 = 7 ∧ ∀ m, 10000 ≤ m ∧ m ≤ 99999 ∧ m % 18 = 7 → n ≤ m :=
  ∃ n, 10015 ≤ n ∧ n ≤ 99999 ∧ n % 18 = 7 ∧ ∀ m, 10000 ≤ m ∧ m ≤ 99999 ∧ m % 18 = 7 → n ≤ m :=
sorry

end least_five_digit_congruent_to_7_mod_18_l683_683346


namespace distance_between_centers_of_tangent_circles_l683_683165

theorem distance_between_centers_of_tangent_circles (r1 r2 : ℝ) (h1 : r1 = 7) (h2 : r2 = 1) (h_tangent : ∀ x y : ℝ, x = r1 → y = r2 → (x + y = 8)) : distance_centers := 
by
  have distance := r1 + r2
  rw [h1, h2] at distance
  exact distance

end distance_between_centers_of_tangent_circles_l683_683165


namespace min_value_a2_plus_b2_l683_683137

theorem min_value_a2_plus_b2 (a b : ℝ) (h : ∀ x : ℝ, x^2 + a * x + 2 * b = 0 -> x = -2) : (∃ a b, a = 1 ∧ b = -1 ∧ ∀ a' b', a^2 + b^2 ≥ a'^2 + b'^2) := 
by {
  sorry
}

end min_value_a2_plus_b2_l683_683137


namespace count_3_digit_numbers_divisible_by_13_l683_683902

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683902


namespace max_boxes_in_wooden_box_l683_683024

def wooden_box_dims := (4 : ℤ, 2 : ℤ, 4 : ℤ) -- in meters
def small_box_dims := (4 : ℤ, 2 : ℤ, 2 : ℤ)  -- in centimeters

theorem max_boxes_in_wooden_box : 
  let wooden_box_vol := (wooden_box_dims.1 * 100) * (wooden_box_dims.2 * 100) * (wooden_box_dims.3 * 100)
  let small_box_vol := small_box_dims.1 * small_box_dims.2 * small_box_dims.3
  wooden_box_vol / small_box_vol = 2_000_000 := by
  sorry

end max_boxes_in_wooden_box_l683_683024


namespace eggs_distributed_equally_l683_683066

-- Define the total number of eggs
def total_eggs : ℕ := 8484

-- Define the number of baskets
def baskets : ℕ := 303

-- Define the expected number of eggs per basket
def eggs_per_basket : ℕ := 28

-- State the theorem
theorem eggs_distributed_equally :
  total_eggs / baskets = eggs_per_basket := sorry

end eggs_distributed_equally_l683_683066


namespace notes_count_l683_683362

theorem notes_count (x : ℕ) (num_2_yuan num_5_yuan num_10_yuan total_notes total_amount : ℕ) 
    (h1 : total_amount = 160)
    (h2 : total_notes = 25)
    (h3 : num_5_yuan = x)
    (h4 : num_10_yuan = x)
    (h5 : num_2_yuan = total_notes - 2 * x)
    (h6 : 2 * num_2_yuan + 5 * num_5_yuan + 10 * num_10_yuan = total_amount) :
    num_5_yuan = 10 ∧ num_10_yuan = 10 ∧ num_2_yuan = 5 :=
by
  sorry

end notes_count_l683_683362


namespace count_3digit_numbers_div_by_13_l683_683896

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683896


namespace three_digit_numbers_div_by_13_l683_683722

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683722


namespace polar_coordinates_of_point_l683_683052

open Real

theorem polar_coordinates_of_point : 
  ∀ (x y : ℝ) (r θ : ℝ), 
    (x = -3) → 
    (y = 3) → 
    (r = sqrt (x^2 + y^2)) → 
    (tan θ = y / x) → 
    (0 ≤ θ ∧ θ < 2 * π) → 
    (r > 0) → 
    (r = 3 * sqrt 2 ∧ θ = 3 * π / 4) :=
begin
  sorry
end

end polar_coordinates_of_point_l683_683052


namespace prob_one_homeowner_second_expected_value_renovation_l683_683425

noncomputable def boxA_red : ℕ := 2
noncomputable def boxA_white : ℕ := 2
noncomputable def boxB_red : ℕ := 3
noncomputable def boxB_white : ℕ := 2

def second_prize_probability : ℝ :=
  (↑(boxA_red.choose 2) * ↑(boxB_red.choose 1) * ↑(boxB_white.choose 1) + 
   ↑(boxA_red.choose 1) * ↑(boxA_white.choose 1) * ↑(boxB_red.choose 2)) / 
   (↑(4.choose 2) * ↑(5.choose 2))

def one_homeowner_second_prize : ℝ := 
  ↑(3.choose 1) * second_prize_probability * (1 - second_prize_probability)^2

theorem prob_one_homeowner_second : 
  one_homeowner_second_prize = 441 / 1000 := 
by 
  sorry

def prob_X (n: ℕ) : ℝ :=
  match n with
  | 10000 => ↑(boxA_red.choose 2) * ↑(boxB_red.choose 2) / (↑(4.choose 2) * ↑(5.choose 2))
  | 5000 => second_prize_probability
  | 3000 => (↑(boxA_red.choose 2) * ↑(boxB_white.choose 2) + 
            ↑(boxA_white.choose 2) * ↑(boxB_red.choose 2) + 
            ↑(boxA_red.choose 1) * ↑(boxA_white.choose 1) * ↑(boxB_red.choose 1) * ↑(boxB_white.choose 1)) / 
            (↑(4.choose 2) * ↑(5.choose 2))
  | 1500 => 1 - prob_X 10000 - prob_X 5000 - prob_X 3000
  | _ => 0

def expected_value_X : ℝ :=
  10000 * prob_X 10000 + 5000 * prob_X 5000 + 3000 * prob_X 3000 + 1500 * prob_X 1500

theorem expected_value_renovation : 
  expected_value_X = 3675 := 
by 
  sorry

end prob_one_homeowner_second_expected_value_renovation_l683_683425


namespace three_digit_numbers_divisible_by_13_l683_683752

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683752


namespace three_digit_numbers_divisible_by_13_count_l683_683624

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683624


namespace number_of_three_digit_numbers_divisible_by_13_l683_683951

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683951


namespace largest_possible_sum_l683_683304

theorem largest_possible_sum :
  ∃ (A B : Finset ℕ), 
    A ∪ B = {2, 3, 5, 7, 11, 13, 17, 19} ∧
    A ∩ B = ∅ ∧
    A.card = 4 ∧
    B.card = 4 ∧
    A.sum = 38 ∧
    B.sum = 39 ∧
    (A.sum * B.sum = 1482) :=
by
  sorry

end largest_possible_sum_l683_683304


namespace count_3_digit_numbers_divisible_by_13_l683_683861

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683861


namespace solution_for_system_of_inequalities_l683_683511

theorem solution_for_system_of_inequalities (p : ℝ) : 
  (19 * p < 10) ∧ (p > 0.5) ↔ (0.5 < p ∧ p < 10/19) := 
by {
  sorry
}

end solution_for_system_of_inequalities_l683_683511


namespace original_pencils_l683_683327

-- Definition of the conditions
def pencils_initial := 115
def pencils_added := 100
def pencils_total := 215

-- Theorem stating the problem to be proved
theorem original_pencils :
  pencils_initial + pencils_added = pencils_total :=
by
  sorry

end original_pencils_l683_683327


namespace b_seq_is_arithmetic_minimal_m_value_l683_683538

noncomputable def a_seq (n : ℕ) : ℝ :=
if n = 1 then 2 else 2 - (1 / a_seq (n-1))

noncomputable def b_seq (n : ℕ) := 1 / (a_seq n - 1)

def is_arithmetic_seq (seq : ℕ → ℝ) : Prop :=
∃ d : ℝ, ∀ n : ℕ, seq (n + 1) = seq n + d

theorem b_seq_is_arithmetic :
  is_arithmetic_seq b_seq :=
sorry

noncomputable def c_seq (b_seq : ℕ → ℝ) (n : ℕ) : ℝ := 1 / (b_seq n * b_seq (n + 2))

noncomputable def T_seq (b_seq : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ k in finset.range n, c_seq b_seq k

theorem minimal_m_value (T_seq : ℕ → ℝ) (m : ℕ) :
  ∀ n : ℕ, T_seq n ≤ m / 12 → 9 ≤ m :=
sorry

end b_seq_is_arithmetic_minimal_m_value_l683_683538


namespace count_three_digit_numbers_divisible_by_13_l683_683599

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683599


namespace largest_integer_a_l683_683077

theorem largest_integer_a (x a : ℤ) :
  ∃ x : ℤ, (x - a) * (x - 7) + 3 = 0 → a ≤ 11 :=
sorry

end largest_integer_a_l683_683077


namespace kyle_paper_delivery_l683_683239

-- Define the number of houses Kyle delivers to from Monday through Saturday
def housesDaily : ℕ := 100

-- Define the number of days from Monday to Saturday
def daysWeek : ℕ := 6

-- Define the adjustment for Sunday (10 fewer, 30 additional)
def sundayAdjust : ℕ := 30 - 10 + 100

-- Define the total number of papers delivered from Monday to Saturday
def papersMonToSat : ℕ := daysWeek * housesDaily

-- Define the total number of papers delivered on Sunday
def papersSunday : ℕ := sundayAdjust

-- Define the total number of papers delivered each week
def totalPapers : ℕ := papersMonToSat + papersSunday

-- The theorem we want to prove
theorem kyle_paper_delivery : totalPapers = 720 := by
  -- We are focusing only on the statement here.
  sorry

end kyle_paper_delivery_l683_683239


namespace count_3_digit_numbers_divisible_by_13_l683_683711

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683711


namespace count_3_digit_multiples_of_13_l683_683779

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683779


namespace problem_statement_l683_683134

theorem problem_statement (a b : ℕ) (m n : ℕ)
  (h1 : 32 + (2 / 7 : ℝ) = 3 * (2 / 7 : ℝ))
  (h2 : 33 + (3 / 26 : ℝ) = 3 * (3 / 26 : ℝ))
  (h3 : 34 + (4 / 63 : ℝ) = 3 * (4 / 63 : ℝ))
  (h4 : 32014 + (m / n : ℝ) = 2014 * 3 * (m / n : ℝ))
  (h5 : 32016 + (a / b : ℝ) = 2016 * 3 * (a / b : ℝ)) :
  (b + 1) / (a * a) = 2016 :=
sorry

end problem_statement_l683_683134


namespace triangle_perimeter_arithmetic_seq_sin_l683_683539

theorem triangle_perimeter_arithmetic_seq_sin (a b c : ℝ) (A : ℝ) :
  a > b ∧ b > c ∧ c > 0 ∧ (a - b = 2 ∧ b - c = 2) ∧ (sin A = sqrt(3) / 2) ∧ A = Real.arcsin (sqrt(3) / 2) + π := 
  a + b + c = 15 :=
by
  sorry

end triangle_perimeter_arithmetic_seq_sin_l683_683539


namespace range_of_t_l683_683529

def gauss_function (x : ℝ) : ℤ := Int.floor x

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1 then x - x^2
  else if 1 ≤ x ∧ x < 2 then x - gauss_function x
  else if 2 ≤ x ∧ x < 3 then (f (x - 2)) * 2
  else if 3 ≤ x ∧ x < 4 then (f (x - 2)) * 2
  else if 4 ≤ x ∧ x < 6 then (f (x - 4)) * 4
  else 0 -- assuming f is 0 outside the given intervals
  
theorem range_of_t (t : ℝ) :
  (∀ x ∈ Set.Ico 4 6, f x < t - 4 / t + 1) ↔ (t ∈ Set.Ico (-1) 0 ∪ Set.Ici 4) :=
by
  sorry

end range_of_t_l683_683529


namespace number_of_elements_in_P_l683_683570

open Set

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {1, 3, 5}
def P : Set ℕ := M ∪ N

theorem number_of_elements_in_P : ∀ P, P = (M ∪ N) → (card P = 6) :=
by
  intros P hP
  rw [hP]
  exact sorry

end number_of_elements_in_P_l683_683570


namespace lost_card_l683_683369

noncomputable def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

theorem lost_card (n : ℕ) (h1 : sum_natural n - 101 < n) (h2 : sum_natural (n - 1) ≤ 101) :
  ∃ x : ℕ, x = sum_natural n - 101 ∧ x = 4 :=
begin
  have h_sum : sum_natural 14 = 105,
  {
    unfold sum_natural,
    norm_num,
  },
  use 4,
  split,
  {
    unfold sum_natural at *,
    norm_num,
    rw h_sum,
  },
  {
    norm_num,
  },
  sorry,
end

end lost_card_l683_683369


namespace system_of_inequalities_solution_l683_683516

theorem system_of_inequalities_solution (p : ℝ) (h1 : 19 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 10 / 19 := by
  sorry

end system_of_inequalities_solution_l683_683516


namespace length_of_BC_prime_l683_683330

theorem length_of_BC_prime (A B C B' C': Point) 
  (r : ℝ) (hr : 1 < r ∧ r < 2) (d : ℝ) (hd : d = 2)
  (hAB : dist A B = d) (hBC : dist B C = d) (hCA : dist C A = d)
  (hB' : dist A B' = r ∧ dist C B' = r ∧ dist B B' ≠ r)
  (hC' : dist A C' = r ∧ dist B C' = r ∧ dist C C' ≠ r) :
  dist B' C' = 1 + sqrt (3 * (r^2 - 1)) :=
sorry

end length_of_BC_prime_l683_683330


namespace triangle_side_length_l683_683970

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) 
  (h1 : a * real.sin B = real.sqrt 2 * real.sin C)
  (h2 : real.cos C = 1 / 3)
  (h3 : (1 / 2) * a * b * real.sin C = 4) : c = 6 :=
sorry

end triangle_side_length_l683_683970


namespace students_in_grade6_l683_683068

noncomputable def num_students_total : ℕ := 100
noncomputable def num_students_grade4 : ℕ := 30
noncomputable def num_students_grade5 : ℕ := 35
noncomputable def num_students_grade6 : ℕ := num_students_total - (num_students_grade4 + num_students_grade5)

theorem students_in_grade6 : num_students_grade6 = 35 := by
  sorry

end students_in_grade6_l683_683068


namespace count_3_digit_multiples_of_13_l683_683797

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683797


namespace maximum_area_of_rectangular_farm_l683_683467

theorem maximum_area_of_rectangular_farm :
  ∃ l w : ℕ, 2 * (l + w) = 160 ∧ l * w = 1600 :=
by
  sorry

end maximum_area_of_rectangular_farm_l683_683467


namespace transformed_roots_l683_683263
  
variable (p q r : ℝ)

-- Definitions
def original_poly := p * X^2 + q * X + r
def roots_of_original_polynomial (u v : ℝ) := (original_poly p q r).roots = {u, v}

-- Theorem Statement
theorem transformed_roots :
  ∀ (u v : ℝ), roots_of_original_polynomial p q r u v → 
  ∃ (qu_r qv_r : ℝ), 
    is_root ((p * Y^2) - (2 * p * r - q) * Y + (p * r - q^2 + q * r)) qu_r ∧
    is_root ((p * Y^2) - (2 * p * r - q) * Y + (p * r - q^2 + q * r)) qv_r
  sorry

end transformed_roots_l683_683263


namespace count_3_digit_numbers_divisible_by_13_l683_683661

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683661


namespace max_value_of_f_on_interval_l683_683501

def f (x : ℝ) : ℝ := -x + 1

def interval : Set ℝ := {x | x ≥ 1/2 ∧ x ≤ 2}

theorem max_value_of_f_on_interval : ∃ max, max ∈ f '' interval ∧ ∀ (y ∈ f '' interval), y ≤ max :=
by
  sorry
 
end max_value_of_f_on_interval_l683_683501


namespace least_five_digit_congruent_to_7_mod_18_l683_683348

theorem least_five_digit_congruent_to_7_mod_18 :
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ n % 18 = 7 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m < n → m % 18 ≠ 7 :=
by
  sorry

end least_five_digit_congruent_to_7_mod_18_l683_683348


namespace range_of_a_l683_683141

open Function

theorem range_of_a (f : ℝ → ℝ) (hf : Monotone (flip f)) (a : ℝ) (h : f (2 - a^2) > f a) : a > 1 ∨ a < -2 := 
sorry

end range_of_a_l683_683141


namespace students_who_chose_water_l683_683485

-- Defining the conditions
def percent_juice : ℚ := 75 / 100
def percent_water : ℚ := 25 / 100
def students_who_chose_juice : ℚ := 90
def ratio_water_to_juice : ℚ := percent_water / percent_juice  -- This should equal 1/3

-- The theorem we need to prove
theorem students_who_chose_water : students_who_chose_juice * ratio_water_to_juice = 30 := 
by
  sorry

end students_who_chose_water_l683_683485


namespace marks_in_english_l683_683473

theorem marks_in_english (math_marks phys_marks chem_marks bio_marks : ℝ)
  (avg_marks num_subjects : ℝ)
  (H_math : math_marks = 60)
  (H_phys : phys_marks = 78)
  (H_chem : chem_marks = 60)
  (H_bio : bio_marks = 65)
  (H_avg : avg_marks = 66.6)
  (H_num : num_subjects = 5) :
  (avg_marks * num_subjects - (math_marks + phys_marks + chem_marks + bio_marks) = 70) :=
by
  have total_marks := avg_marks * num_subjects
  have calculated_marks := total_marks - (math_marks + phys_marks + chem_marks + bio_marks)
  have H_total_marks : total_marks = 333 := by sorry
  have H_calculated_marks : calculated_marks = 70 := by sorry
  exact H_calculated_marks

end marks_in_english_l683_683473


namespace count_3_digit_numbers_divisible_by_13_l683_683874

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683874


namespace max_value_of_f_l683_683564

def f (x : ℝ) : ℝ := x + real.sqrt(1 - x)

theorem max_value_of_f :
  ∃ x : ℝ, x ≤ 1 ∧ ∀ y : ℝ, y ≤ 1 → f(y) ≤ (5/4) := by {
  sorry
}

end max_value_of_f_l683_683564


namespace count_three_digit_numbers_divisible_by_13_l683_683652

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683652


namespace num_3_digit_div_by_13_l683_683818

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683818


namespace largest_of_three_consecutive_integers_sum_18_l683_683318

theorem largest_of_three_consecutive_integers_sum_18 (n : ℤ) (h : n + (n + 1) + (n + 2) = 18) : n + 2 = 7 :=
by
  sorry

end largest_of_three_consecutive_integers_sum_18_l683_683318


namespace total_sequins_correct_l683_683220

def blue_rows : ℕ := 6
def blue_columns : ℕ := 8
def purple_rows : ℕ := 5
def purple_columns : ℕ := 12
def green_rows : ℕ := 9
def green_columns : ℕ := 6

def total_sequins : ℕ :=
  (blue_rows * blue_columns) + (purple_rows * purple_columns) + (green_rows * green_columns)

theorem total_sequins_correct : total_sequins = 162 := by
  sorry

end total_sequins_correct_l683_683220


namespace three_digit_numbers_divisible_by_13_count_l683_683634

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683634


namespace number_of_three_digit_numbers_divisible_by_13_l683_683941

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683941


namespace jill_initial_investment_l683_683225

noncomputable def initial_investment (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  A / (1 + r / n)^(n * t)

theorem jill_initial_investment :
  initial_investment 10815.83 0.0396 2 2 ≈ 10000 :=
by
  sorry

end jill_initial_investment_l683_683225


namespace infinite_sum_equals_fraction_l683_683104

def closest_integer_to_sqrt (n : ℕ) : ℤ :=
  if h : 0 < n then (Nat.floor (Real.sqrt n.toReal)).toInt 
  else 0

theorem infinite_sum_equals_fraction :
  (∑ n in (Finset.range (n+1)).erase 0, ((3 ^ closest_integer_to_sqrt n) + (3 ^ (- closest_integer_to_sqrt n))) / (3 ^ n))
  = 39 / 27 := by
  sorry

end infinite_sum_equals_fraction_l683_683104


namespace average_burgers_per_day_l683_683462

variable (total_spent : ℝ) (total_burgers : ℝ) (days_in_june : ℝ)
variable (average_daily_burgers : ℝ)

-- Given conditions
def conditions := total_spent = 372 ∧ total_burgers = 12 ∧ days_in_june = 30

-- The mathematical statement to prove
theorem average_burgers_per_day (h : conditions) : average_daily_burgers = total_burgers / days_in_june → average_daily_burgers = 0.4 :=
by
  intro h_avg
  rw [←h_avg]
  cases h with h_spent h_rest
  cases h_rest with h_burgers h_days
  rw [h_burgers, h_days]
  simp
  -- Provide proof steps here
  sorry

end average_burgers_per_day_l683_683462


namespace EC_dot_ED_eq_three_l683_683986

-- Define the setup for the problem
variables {A B C D E : Type} [metric_space A] [metric_space B]
variables (A B C D : point) (E : point)
variable (side_length : length)
variable (midpoint : point)

-- Define the conditions
def square (A B C D : point) (side_length : ℝ) : Prop := 
  is_square A B C D side_length

def is_midpoint (E : point) (A B : point) : Prop :=
  midpoint E A B

-- State the problem
theorem EC_dot_ED_eq_three
  (h1 : square A B C D 2) 
  (h2 : is_midpoint E A B) : 
  (vector_between_points E C) • (vector_between_points E D) = 3 := 
sorry

end EC_dot_ED_eq_three_l683_683986


namespace correct_incorrect_difference_l683_683365

variable (x : ℝ)

theorem correct_incorrect_difference : (x - 2152) - (x - 1264) = 888 := by
  sorry

end correct_incorrect_difference_l683_683365


namespace count_three_digit_numbers_divisible_by_13_l683_683639

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683639


namespace greatest_n_l683_683243

noncomputable section

-- Definitions
variables {a d : ℕ}
variables (h_a : a > 1) (h_d : d > 1) (coprime_ad : nat.coprime a d)

-- Sequence definition
def sequence (x : ℕ → ℕ) : Prop :=
  x 1 = 1 ∧
  ∀ k, x (k + 1) = 
    if a ∣ x k then x k / a else x k + d

-- Goal
theorem greatest_n (x : ℕ → ℕ) (hx : sequence x) :
  ∃ k, a ^ (⌈real.log a d⌉₊) ∣ x k :=
sorry

end greatest_n_l683_683243


namespace solve_system_of_inequalities_l683_683507

theorem solve_system_of_inequalities (p : ℚ) : (19 * p < 10) ∧ (1 / 2 < p) → (1 / 2 < p ∧ p < 10 / 19) :=
by
  sorry

end solve_system_of_inequalities_l683_683507


namespace count_three_digit_div_by_13_l683_683678

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683678


namespace find_lost_card_number_l683_683386

theorem find_lost_card_number (n : ℕ) (S : ℕ) (x : ℕ) 
  (h1 : S = n * (n + 1) / 2) 
  (h2 : S - x = 101) 
  (h3 : n = 14) : 
  x = 4 := 
by sorry

end find_lost_card_number_l683_683386


namespace train_cross_first_platform_l683_683022

noncomputable def time_to_cross_first_platform (L_t L_p1 L_p2 t2 : ℕ) : ℕ :=
  (L_t + L_p1) / ((L_t + L_p2) / t2)

theorem train_cross_first_platform :
  time_to_cross_first_platform 100 200 300 20 = 15 :=
by
  sorry

end train_cross_first_platform_l683_683022


namespace find_lost_card_number_l683_683385

theorem find_lost_card_number (n : ℕ) (S : ℕ) (x : ℕ) 
  (h1 : S = n * (n + 1) / 2) 
  (h2 : S - x = 101) 
  (h3 : n = 14) : 
  x = 4 := 
by sorry

end find_lost_card_number_l683_683385


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683583

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683583


namespace count_3_digit_numbers_divisible_by_13_l683_683670

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683670


namespace solution_for_system_of_inequalities_l683_683510

theorem solution_for_system_of_inequalities (p : ℝ) : 
  (19 * p < 10) ∧ (p > 0.5) ↔ (0.5 < p ∧ p < 10/19) := 
by {
  sorry
}

end solution_for_system_of_inequalities_l683_683510


namespace num_three_digit_div_by_13_l683_683841

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683841


namespace num_3_digit_div_by_13_l683_683820

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683820


namespace board_coloring_condition_l683_683489

def neighbors (m n : ℕ) (board : Fin m × Fin n → bool) (cell : Fin m × Fin n) : List (Fin m × Fin n) :=
  let (i, j) := cell
  [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1), (i + 1, j + 1), (i - 1, j - 1), (i + 1, j - 1), (i - 1, j + 1)]
  |>.filter (λ (x, y), x < m ∧ y < n)

def neighbor_same_colored_cells_odd (m n : ℕ) (board : Fin m × Fin n → bool) : Prop :=
  ∀ cell, (neighbors m n board cell)
    .filter (λ neighbor, board cell = board neighbor)
    .length % 2 = 1

theorem board_coloring_condition (m n : ℕ) (board : Fin m × Fin n → bool) :
  neighbor_same_colored_cells_odd m n board ↔ (m % 2 = 0 ∨ n % 2 = 0) :=
sorry

end board_coloring_condition_l683_683489


namespace three_digit_sequence_exists_l683_683996

-- We define the concept of a three-digit number and associated conditions.
def three_digit_number (n : ℕ) : Prop := 
  n >= 100 ∧ n < 1000 ∧ n % 10 ≠ 0

-- Define the problem statement in Lean.
theorem three_digit_sequence_exists :
  (∃ seq : List ℕ, 
    (∀ n, n ∈ seq → three_digit_number n) ∧
    (∀ i, i < seq.length - 1 → (seq[i] % 10 = seq[i + 1] / 100))) :=
sorry

end three_digit_sequence_exists_l683_683996


namespace num_3_digit_div_by_13_l683_683825

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683825


namespace least_five_digit_congruent_to_7_mod_18_l683_683349

theorem least_five_digit_congruent_to_7_mod_18 :
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ n % 18 = 7 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m < n → m % 18 ≠ 7 :=
by
  sorry

end least_five_digit_congruent_to_7_mod_18_l683_683349


namespace solution_for_system_of_inequalities_l683_683513

theorem solution_for_system_of_inequalities (p : ℝ) : 
  (19 * p < 10) ∧ (p > 0.5) ↔ (0.5 < p ∧ p < 10/19) := 
by {
  sorry
}

end solution_for_system_of_inequalities_l683_683513


namespace lost_card_l683_683371

noncomputable def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

theorem lost_card (n : ℕ) (h1 : sum_natural n - 101 < n) (h2 : sum_natural (n - 1) ≤ 101) :
  ∃ x : ℕ, x = sum_natural n - 101 ∧ x = 4 :=
begin
  have h_sum : sum_natural 14 = 105,
  {
    unfold sum_natural,
    norm_num,
  },
  use 4,
  split,
  {
    unfold sum_natural at *,
    norm_num,
    rw h_sum,
  },
  {
    norm_num,
  },
  sorry,
end

end lost_card_l683_683371


namespace num_pos_3_digit_div_by_13_l683_683767

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683767


namespace count_three_digit_numbers_divisible_by_13_l683_683603

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683603


namespace poly_divisible_by_x2_x_1_l683_683276

noncomputable def ω : ℂ := exp (2 * π * I / 3)

theorem poly_divisible_by_x2_x_1 (n : ℕ) (hn : 0 < n)
  (ω_prop : ω ^ 3 = 1 ∧ ω ≠ 1 ∧ (ω ^ 2 + ω + 1) = 0):
  (λ x : ℂ, (x + 1) ^ (2 * n + 1) + x ^ (n + 2)) x % (x ^ 2 + x + 1) = 0 :=
by {
  sorry
}

end poly_divisible_by_x2_x_1_l683_683276


namespace sum_b_formula_l683_683127

def S (n : ℕ) : ℤ :=
  2 - (2 - 2 * n) * 2^(n + 1)

def b (n : ℕ) : ℤ :=
  n^2 * 2^n

def sum_b (n : ℕ) : ℤ :=
  ∑ i in Finset.range( n + 1), b i

theorem sum_b_formula (n : ℕ) : sum_b n = (n^2 - 2*n + 4) * 2^(n + 1) - 2 := 
by
  -- Proof will be supplied here.
  sorry

end sum_b_formula_l683_683127


namespace count_3_digit_multiples_of_13_l683_683778

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683778


namespace num_pos_3_digit_div_by_13_l683_683764

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683764


namespace num_three_digit_div_by_13_l683_683853

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683853


namespace fuel_tank_capacity_l683_683456

variable (C : ℝ)
variable (EthanolContent_A EthanolContent_B TotalEthanol : ℝ)
variable (Volume_A : ℝ)

-- Conditions:
def Condition1 := EthanolContent_A = 0.12
def Condition2 := EthanolContent_B = 0.16
def Condition3 := TotalEthanol = 30
def Condition4 := Volume_A = 82

-- Proof statement
theorem fuel_tank_capacity : Condition1 ∧ Condition2 ∧ Condition3 ∧ Condition4 → C = 208 :=
by 
  intros h
  sorry

end fuel_tank_capacity_l683_683456


namespace system_of_inequalities_solution_l683_683517

theorem system_of_inequalities_solution (p : ℝ) (h1 : 19 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 10 / 19 := by
  sorry

end system_of_inequalities_solution_l683_683517


namespace three_digit_numbers_div_by_13_l683_683718

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683718


namespace count_3digit_numbers_div_by_13_l683_683878

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683878


namespace count_3_digit_numbers_divisible_by_13_l683_683911

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683911


namespace Kyle_papers_delivered_each_week_l683_683236

-- Definitions representing the conditions
def papers_per_day := 100
def days_Mon_to_Sat := 6
def regular_Sunday_customers := 100
def non_regular_Sunday_customers := 30
def no_delivery_customers_on_Sunday := 10

-- The total number of papers delivered each week
def total_papers_per_week : ℕ :=
  days_Mon_to_Sat * papers_per_day +
  regular_Sunday_customers - no_delivery_customers_on_Sunday + non_regular_Sunday_customers

-- Prove that Kyle delivers 720 papers each week
theorem Kyle_papers_delivered_each_week : total_papers_per_week = 720 :=
sorry

end Kyle_papers_delivered_each_week_l683_683236


namespace find_number_for_f_eq_zero_l683_683033

noncomputable def f : ℕ → ℤ := sorry

theorem find_number_for_f_eq_zero :
  (∃ a : ℕ, f(a) = 0) ∧
  (∀ m n : ℕ, f(m + n) = f(m) + f(n) + 4 * (9 * m * n - 1)) ∧
  (f 17 = 4832) →
  (∃ a : ℕ, f(a) = 0 ∧ a = 1) :=
  sorry

end find_number_for_f_eq_zero_l683_683033


namespace find_lost_card_number_l683_683388

theorem find_lost_card_number (n : ℕ) (S : ℕ) (x : ℕ) 
  (h1 : S = n * (n + 1) / 2) 
  (h2 : S - x = 101) 
  (h3 : n = 14) : 
  x = 4 := 
by sorry

end find_lost_card_number_l683_683388


namespace caterpillar_to_scorpion_ratio_l683_683466

theorem caterpillar_to_scorpion_ratio 
  (roach_count : ℕ) (scorpion_count : ℕ) (total_insects : ℕ) 
  (h_roach : roach_count = 12) 
  (h_scorpion : scorpion_count = 3) 
  (h_cricket : cricket_count = roach_count / 2) 
  (h_total : total_insects = 27) 
  (h_non_cricket_count : non_cricket_count = roach_count + scorpion_count + cricket_count) 
  (h_caterpillar_count : caterpillar_count = total_insects - non_cricket_count) : 
  (caterpillar_count / scorpion_count) = 2 := 
by 
  sorry

end caterpillar_to_scorpion_ratio_l683_683466


namespace total_stars_l683_683325

theorem total_stars (total_students : ℕ) (perc_8_stars : ℕ) (stars_8 : ℕ) (stars_12 : ℕ) 
    (h_total_students : total_students = 500)
    (h_perc_8_stars : perc_8_stars = 70)
    (h_stars_8 : stars_8 = 8)
    (h_stars_12 : stars_12 = 12) : 
    (0.01 * perc_8_stars * total_students * stars_8) + ((1 - 0.01 * perc_8_stars) * total_students * stars_12) = 4600 := 
by
  sorry

end total_stars_l683_683325


namespace find_r_in_binomial_expansion_l683_683480

theorem find_r_in_binomial_expansion :
  ∃ r : ℕ, (binomial 20 (4 * r - 1) = binomial 20 (r + 1)) ∧ r = 4 := by
  -- a mathematical proof goes here
  sorry

end find_r_in_binomial_expansion_l683_683480


namespace chord_length_l683_683427

theorem chord_length (O P Q R : Point) :
  (dist O P = 5) ∧ (dist O R = 4) ∧ (midpoint R P Q) → dist P Q = 6 := by
 sorry

end chord_length_l683_683427


namespace count_3_digit_numbers_divisible_by_13_l683_683668

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683668


namespace trapezoid_inscribed_midpoint_circle_l683_683023

/-- A statement to prove that the circumcircle of triangle APQ passes through the midpoint of AB.
    Given:
    - A trapezoid ABCD is inscribed in a circle centered at O.
    - AB and CD are the bases of the trapezoid.
    - AP and AQ are tangents from A to the circumcircle of triangle CDO.
    Prove: The circumcircle of triangle APQ passes through the midpoint of AB. -/
theorem trapezoid_inscribed_midpoint_circle
  {A B C D O P Q : Point}
  (h_circle : IsCyclic A B C D)
  (h_O_center : Center O A B C D)
  (base1 : IsBase A B)
  (base2 : IsBase C D)
  (tangent1 : IsTangent A P (circumcircle C D O))
  (tangent2 : IsTangent A Q (circumcircle C D O)) :
  PassesThrough (midpoint A B) (circumcircle A P Q) :=
sorry

end trapezoid_inscribed_midpoint_circle_l683_683023


namespace babylonian_area_formula_rectangle_l683_683288

theorem babylonian_area_formula_rectangle (a b c d : ℝ) (h_ab : a = b) (h_cd : c = d) :
  (a + b) / 2 * (c + d) / 2 = a * c :=
by
  have h1 : (a + b) / 2 = a := by rw [h_ab, add_self_div_two]
  have h2 : (c + d) / 2 = c := by rw [h_cd, add_self_div_two]
  rw [h1, h2]
  -- complete the proof
  sorry

end babylonian_area_formula_rectangle_l683_683288


namespace solution_of_system_of_inequalities_l683_683521

theorem solution_of_system_of_inequalities (p : ℝ) :
  19 * p < 10 ∧ p > 0.5 ↔ (1 / 2) < p ∧ p < (10 / 19) :=
by
  sorry

end solution_of_system_of_inequalities_l683_683521


namespace count_3_digit_numbers_divisible_by_13_l683_683708

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683708


namespace proof_problem_l683_683108

theorem proof_problem (a b A B : ℝ) (f : ℝ → ℝ)
  (h_f : ∀ θ : ℝ, f θ ≥ 0)
  (h_f_def : ∀ θ : ℝ, f θ = 1 + a * Real.cos θ + b * Real.sin θ + A * Real.sin (2 * θ) + B * Real.cos (2 * θ)) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end proof_problem_l683_683108


namespace count_3_digit_numbers_divisible_by_13_l683_683658

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683658


namespace donuts_Niraek_covers_l683_683459

/- Define the radii of the donut holes -/
def radius_Niraek : ℕ := 5
def radius_Theo : ℕ := 9
def radius_Akshaj : ℕ := 10
def radius_Lily : ℕ := 7

/- Define the surface areas of the donut holes -/
def surface_area (r : ℕ) : ℕ := 4 * r * r

/- Compute the surface areas -/
def sa_Niraek := surface_area radius_Niraek
def sa_Theo := surface_area radius_Theo
def sa_Akshaj := surface_area radius_Akshaj
def sa_Lily := surface_area radius_Lily

/- Define a function to compute the LCM of a list of natural numbers -/
def lcm_of_list (l : List ℕ) : ℕ := l.foldr Nat.lcm 1

/- Compute the lcm of the surface areas -/
def lcm_surface_areas := lcm_of_list [sa_Niraek, sa_Theo, sa_Akshaj, sa_Lily]

/- Compute the answer -/
def num_donuts_Niraek_covers := lcm_surface_areas / sa_Niraek

/- Prove the statement -/
theorem donuts_Niraek_covers : num_donuts_Niraek_covers = 63504 :=
by
  /- Skipping the proof for now -/
  sorry

end donuts_Niraek_covers_l683_683459


namespace max_value_of_trig_fn_l683_683299

theorem max_value_of_trig_fn : 
  ∀ (x : ℝ), (f x = (1/5) * real.sin (x + π/3) + real.cos (x - π/6)) → 
  (∀ y : ℝ, f y ≤ 6/5) ∧ (∃ z : ℝ, f z = 6/5) :=
begin
  sorry
end

def f (x : ℝ) : ℝ :=
  (1/5) * real.sin (x + π/3) + real.cos (x - π/6)

end max_value_of_trig_fn_l683_683299


namespace number_of_increasing_six_digit_integers_mod_1000_l683_683247

theorem number_of_increasing_six_digit_integers_mod_1000 :
  (finset.card {l | l.length = 6 ∧ l.sorted ∧ l.all (λ x, x ≤ 8) ∧ multiset.card l = 6} % 1000) = 3 := sorry

end number_of_increasing_six_digit_integers_mod_1000_l683_683247


namespace count_3_digit_numbers_divisible_by_13_l683_683898

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683898


namespace count_three_digit_numbers_divisible_by_13_l683_683654

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683654


namespace count_3_digit_numbers_divisible_by_13_l683_683868

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683868


namespace number_of_3_digit_divisible_by_13_l683_683803

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683803


namespace second_month_sale_l683_683009

theorem second_month_sale (S : ℝ) :
  (S + 5420 + 6200 + 6350 + 6500 = 30000) → S = 5530 :=
by
  sorry

end second_month_sale_l683_683009


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683582

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683582


namespace n_is_even_l683_683179

def sum_of_digits (m : ℕ) : ℕ :=
  m.digits.foldr (· + ·) 0

theorem n_is_even (n : ℕ) (h1 : n > 0) (h2 : sum_of_digits n = 2014) (h3 : sum_of_digits (5 * n) = 1007) : even n :=
  sorry

end n_is_even_l683_683179


namespace area_ratio_l683_683203

noncomputable def initial_areas (a b c : ℝ) :=
  a > 0 ∧ b > 0 ∧ c > 0

noncomputable def misallocated_areas (a b : ℝ) :=
  let b' := b + 0.1 * a - 0.5 * b
  b' = 0.4 * (a + b)

noncomputable def final_ratios (a b c : ℝ) :=
  let a' := 0.9 * a + 0.5 * b
  let b' := b + 0.1 * a - 0.5 * b
  let c' := 0.5 * c
  a' + b' + c' = a + b + c ∧ a' / b' = 2 ∧ b' / c' = 1 

theorem area_ratio (a b c m : ℝ) (h1 : initial_areas a b c) 
  (h2 : misallocated_areas a b)
  (h3 : final_ratios a b c) : 
  (m = 0.4 * a) → (m / (a + b + c) = 1 / 20) :=
sorry

end area_ratio_l683_683203


namespace sum_valid_pairs_l683_683004

-- Define the dimensions of the board
def rows := 15
def columns := 19

-- Define the row-wise numbering function
def row_num (i j : Nat) : Nat :=
  19 * (i - 1) + j

-- Define the column-wise numbering function
def col_num (i j : Nat) : Nat :=
  15 * (j - 1) + i

-- Define the properties of valid (i, j) pairs
def valid_pairs (i j : Nat) : Prop :=
  row_num i j = col_num i j

-- Define all pairs that satisfy the valid_pair property within the board constraints
def valid_pairs_sum : Nat :=
  ∑ (i : Fin rows) (j : Fin columns), 
    if valid_pairs (i + 1) (j + 1) then row_num (i + 1) (j + 1) else 0

-- State the theorem
theorem sum_valid_pairs : valid_pairs_sum = 286 :=
  sorry

end sum_valid_pairs_l683_683004


namespace positive_integers_not_in_S_l683_683242

def smallestSetSatisfyingConditions : Set ℕ :=
  {n | 
    2 ∈ S ∧ 
    (∀ m : ℕ, m^2 ∈ S → m ∈ S) ∧ 
    (forall m : ℕ, m ∈ S → (m + 5)^2 ∈ S)}

theorem positive_integers_not_in_S : ∀ n : ℕ, 
  (n ∉ smallestSetSatisfyingConditions ↔ (n = 1 ∨ ∃ k : ℕ, n = 5 * k)) :=
by
  sorry

end positive_integers_not_in_S_l683_683242


namespace largest_sum_l683_683040

theorem largest_sum : max (1/4 + 1/9) (max (1/4 + 1/10) (max (1/4 + 1/2) (max (1/4 + 1/12) (1/4 + 1/11)))) = 3/4 :=
by {
  -- The proof goes here
  sorry
}

end largest_sum_l683_683040


namespace lost_card_number_l683_683377

theorem lost_card_number (n : ℕ) (x : ℕ) (h₁ : (n * (n + 1)) / 2 - x = 101) : x = 4 :=
sorry

end lost_card_number_l683_683377


namespace range_of_m_not_monotonic_value_of_a_compare_sizes_l683_683566

variable {a : ℝ} (f g : ℝ → ℝ)

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + a
def g (x : ℝ) : ℝ := log x / log a

-- Condition: a > 0 and a ≠ 1
axiom (a_pos : a > 0)
axiom (a_ne_one : a ≠ 1)

-- Prove: Range of m such that f(x) is not monotonic on [-1, 2m]
theorem range_of_m_not_monotonic (m : ℝ) : 
  (∃ (m : ℝ), (m > 1/2) ↔ ¬(∀ x ∈ closed_interval (-1:ℝ) (2*m), 
  deriv f x ≤ 0 ∨ deriv f x ≥ 0)) := sorry

-- Prove: Value of a given f(1) = g(1)
theorem value_of_a : f 1 = g 1 → a = 2 := sorry

-- Prove: Comparison of t1, t2, t3 for x ∈ (0, 1)
axiom t1 (x : ℝ) : ℝ := 1/2 * f x
axiom t2 (x : ℝ) : ℝ := g x
axiom t3 (x : ℝ) : ℝ := 2^x

theorem compare_sizes (x : ℝ) 
  (hx : 0 < x ∧ x < 1) : t2 x < t1 x ∧ t1 x < t3 x := sorry

end range_of_m_not_monotonic_value_of_a_compare_sizes_l683_683566


namespace machines_production_difference_l683_683977

theorem machines_production_difference (A B: ℕ) (prod_rate_a prod_rate_b: ℕ) (total_prod_b: ℕ) (start_time same_start: Prop) (prod_per_min_a eq_prod_per_min_a: A = prod_per_min_a) (prod_per_min_b eq_prod_per_min_b: B = prod_per_min_b)
  (rate_a: prod_rate_a = 5) (rate_b: prod_rate_b = 8) (total_b: total_prod_b = 40) : 
  (total_prod_b - (prod_rate_a * (total_prod_b / prod_rate_b))) = 15 :=
by
  sorry

end machines_production_difference_l683_683977


namespace total_marbles_eq_3_4_r_l683_683975

theorem total_marbles_eq_3_4_r (r : ℕ) (b g : ℕ) 
    (h1 : r = b + 0.25 * b) 
    (h2 : g = r + 0.6 * r) : r + b + g = 3.4 * r := 
by 
  sorry

end total_marbles_eq_3_4_r_l683_683975


namespace num_3_digit_div_by_13_l683_683829

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683829


namespace A_ge_B_l683_683131

-- Definitions of the given conditions
variables {k n : ℕ} (h_k : k ≥ 1) (h_n : n ≥ 1)
variables {a : ℕ → ℝ} (h_a : ∀ i j, i ≠ j → a i ≠ a j)
variables {ℓ : ℝ} -- the real number ell
def S : set ℝ := {x | ∃ i, x = a i ∨ x = -a i}

-- Definitions of A and B
noncomputable def G (α : ℝ) : ℝ :=
  2 * ∑ i in finset.range k, real.cos (2 * π * (a i) * α)

noncomputable def A : ℝ :=
  ∫ α in 0..1, (G α) ^ (2 * n)

noncomputable def B : ℝ :=
  ∫ α in 0..1, (G α) ^ (2 * n) * real.exp (-(2 * π * ℓ * α * complex.I)).re

-- Theorem statement to prove A ≥ B
theorem A_ge_B : A h_k h_n h_a ≥ B h_k h_n h_a :=
begin
  -- Placeholder, actual proof omitted
  sorry
end

end A_ge_B_l683_683131


namespace dodecahedron_decagon_area_sum_l683_683440

theorem dodecahedron_decagon_area_sum {a b c : ℕ} (h1 : Nat.Coprime a c) (h2 : b ≠ 0) (h3 : ¬ ∃ p : ℕ, p.Prime ∧ p * p ∣ b) 
  (area_eq : (5 + 5 * Real.sqrt 5) / 4 = (a * Real.sqrt b) / c) : a + b + c = 14 :=
sorry

end dodecahedron_decagon_area_sum_l683_683440


namespace find_b_value_l683_683076

theorem find_b_value :
  ∃ b : ℕ, 70 = (2 * (b + 1)^2 + 3 * (b + 1) + 4) - (2 * (b - 1)^2 + 3 * (b - 1) + 4) ∧ b > 0 ∧ b < 1000 :=
by
  sorry

end find_b_value_l683_683076


namespace tan_pi_add_alpha_eq_two_l683_683117

theorem tan_pi_add_alpha_eq_two
  (α : ℝ)
  (h : Real.tan (Real.pi + α) = 2) :
  (2 * Real.sin α - Real.cos α) / (3 * Real.sin α + 2 * Real.cos α) = 3 / 8 :=
sorry

end tan_pi_add_alpha_eq_two_l683_683117


namespace number_of_3_digit_divisible_by_13_l683_683812

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683812


namespace count_3_digit_numbers_divisible_by_13_l683_683873

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683873


namespace train_time_l683_683448

theorem train_time (T : ℕ) (D : ℝ) (h1 : D = 48 * (T / 60)) (h2 : D = 60 * (40 / 60)) : T = 50 :=
by
  sorry

end train_time_l683_683448


namespace count_3digit_numbers_div_by_13_l683_683889

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683889


namespace repeating_decimal_to_fraction_l683_683500

theorem repeating_decimal_to_fraction : 
  let n := "0.414141..." -- Informal handle for infinite repeating decimal
  n = (41 : ℚ) / 99 := 
sorry

end repeating_decimal_to_fraction_l683_683500


namespace distance_from_focus_to_asymptote_l683_683153

theorem distance_from_focus_to_asymptote (a b e : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) 
    (h_ecc : ∀ e > 0, a^2 + b^2 = (e*a)^2) 
    (h_point1 : e^2 / a^2 - 1 / b^2 = 1) 
    (h_point2 : 3 / a^2 - 2 / b^2 = 1) 
    (points_on_hyperbola : (e, 1) ∈ setOf (λ (p : ℝ × ℝ), (p.1^2 / a^2 - p.2^2 / b^2 = 1)) ∧ 
                              (-sqrt 3, sqrt 2) ∈ setOf (λ (p : ℝ × ℝ), (p.1^2 / a^2 - p.2^2 / b^2 = 1))) :
    let c := sqrt (a^2 + b^2),
    let f := c / a in
    let d := c / sqrt (a^2 + b^2) in
    d = 1 := sorry

end distance_from_focus_to_asymptote_l683_683153


namespace largest_of_three_consecutive_integers_l683_683315

theorem largest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 18) : x + 2 = 7 := 
sorry

end largest_of_three_consecutive_integers_l683_683315


namespace maximum_value_f_l683_683120

noncomputable def f (a : ℝ) : ℝ :=
  ∫ x in 0..1, (2 * a * x^2 - a^2 * x)

theorem maximum_value_f :
  ∃ (a : ℝ), f a = 2 / 9 :=
begin
  use 2 / 3,
  have h : ∀ a, f a = -1 / 2 * (a - 2 / 3)^2 + 2 / 9, from sorry,
  simp [h],
  linarith,
end

end maximum_value_f_l683_683120


namespace sulfuric_acid_needed_for_zinc_sulfate_l683_683082

def zinc_to_zinc_sulfate (moles_of_zinc : ℕ) (moles_of_zinc_sulfate : ℕ) (moles_of_sulfuric_acid : ℕ): Prop :=
  -- balanced equation: Zn + H2SO4 → ZnSO4 + H2
  -- 1 mole of Zn reacts with 1 mole of H2SO4 to produce 1 mole of ZnSO4
  moles_of_zinc_sulfate = moles_of_zinc ∧ moles_of_sulfuric_acid = moles_of_zinc

theorem sulfuric_acid_needed_for_zinc_sulfate (moles_of_zinc : ℕ) (moles_of_zinc_sulfate : ℕ):
  zinc_to_zinc_sulfate moles_of_zinc moles_of_zinc_sulfate 2 :=
by
  -- Given:
  assume moles_of_zinc = 2
  assume moles_of_zinc_sulfate = 2

  -- Hence,
  -- number of moles of Sulfuric acid combined = 2 (based on the given balanced equation's stoichiometry)
  show zinc_to_zinc_sulfate 2 2 2 from sorry

end sulfuric_acid_needed_for_zinc_sulfate_l683_683082


namespace count_3_digit_multiples_of_13_l683_683789

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683789


namespace initial_number_of_people_l683_683215

theorem initial_number_of_people (P : ℕ) : P * 10 = (P + 1) * 5 → P = 1 :=
by sorry

end initial_number_of_people_l683_683215


namespace count_3_digit_numbers_divisible_by_13_l683_683900

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683900


namespace count_three_digit_numbers_divisible_by_13_l683_683653

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683653


namespace total_sequins_is_162_l683_683217

/-- Jane sews 6 rows of 8 blue sequins each. -/
def rows_of_blue_sequins : Nat := 6
def sequins_per_blue_row : Nat := 8
def total_blue_sequins : Nat := rows_of_blue_sequins * sequins_per_blue_row

/-- Jane sews 5 rows of 12 purple sequins each. -/
def rows_of_purple_sequins : Nat := 5
def sequins_per_purple_row : Nat := 12
def total_purple_sequins : Nat := rows_of_purple_sequins * sequins_per_purple_row

/-- Jane sews 9 rows of 6 green sequins each. -/
def rows_of_green_sequins : Nat := 9
def sequins_per_green_row : Nat := 6
def total_green_sequins : Nat := rows_of_green_sequins * sequins_per_green_row

/-- The total number of sequins Jane adds to her costume. -/
def total_sequins : Nat := total_blue_sequins + total_purple_sequins + total_green_sequins

theorem total_sequins_is_162 : total_sequins = 162 := 
by
  sorry

end total_sequins_is_162_l683_683217


namespace three_digit_numbers_divisible_by_13_l683_683757

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683757


namespace num_three_digit_div_by_13_l683_683847

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683847


namespace count_three_digit_div_by_13_l683_683679

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683679


namespace solve_for_x_l683_683087

theorem solve_for_x : ∃ x : ℝ, (16^(x-1) / 8^(x-1) = 64^(x+2)) ∧ x = -13/5 := by
  sorry

end solve_for_x_l683_683087


namespace circle_parabola_intersect_exactly_two_points_l683_683481

-- Definitions based on the conditions
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4
def parabola (x y b : ℝ) : Prop := y = x^2 - b * x

-- The statement translating the problem into Lean 4
theorem circle_parabola_intersect_exactly_two_points (b : ℝ) :
  (∃ x y : ℝ, circle x y ∧ parabola x y b) → 
  b = 2 :=
sorry

end circle_parabola_intersect_exactly_two_points_l683_683481


namespace number_of_three_digit_numbers_divisible_by_13_l683_683953

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683953


namespace trailing_zeros_15_factorial_in_base_18_l683_683037

theorem trailing_zeros_15_factorial_in_base_18 : 
  let f := factorial 15
  let n := 18 -- equivalent to 2 * 3^2
  count_trailing_zeros_in_base f n = 3 := 
by sorry

end trailing_zeros_15_factorial_in_base_18_l683_683037


namespace green_to_yellow_ratio_of_concentric_circles_l683_683042

theorem green_to_yellow_ratio_of_concentric_circles :
  ∀ (r_small r_large : ℝ)
  (h1 : r_small = 1)
  (h2 : r_large = 3),
  let A_yellow := π * r_small^2,
      A_large := π * r_large^2,
      A_green := A_large - A_yellow
  in A_green / A_yellow = 8 :=
by
  intros r_small r_large h1 h2,
  rw [h1, h2],
  let A_yellow := π * (1:ℝ)^2,
  let A_large := π * (3:ℝ)^2,
  let A_green := A_large - A_yellow,
  have hA_yellow : A_yellow = π,
  { simp [A_yellow] },
  have hA_large : A_large = 9 * π,
  { simp [A_large] },
  have hA_green : A_green = 8 * π,
  { simp [A_green, hA_yellow, hA_large] },
  rw [hA_yellow, hA_green],
  simp,
  exact sorry

end green_to_yellow_ratio_of_concentric_circles_l683_683042


namespace count_3digit_numbers_div_by_13_l683_683893

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683893


namespace sum_of_valid_z_for_14z48_div_by_9_l683_683479

theorem sum_of_valid_z_for_14z48_div_by_9 : 
  (∑ z in finset.range 10, if (17 + z) % 9 = 0 then z else 0) = 1 := 
by
  sorry

end sum_of_valid_z_for_14z48_div_by_9_l683_683479


namespace AnalyticExpressionOfFunction_RangeOfFunction_RangeOfTheta_l683_683146

-- Given definitions from conditions
def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)
def g (θ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (2 * (x - θ) + Real.pi / 6)

-- The first proof statement
theorem AnalyticExpressionOfFunction :
    f = λ x, 2 * Real.sin (2 * x + Real.pi / 6) :=
  sorry

-- The second proof statement
theorem RangeOfFunction (x : ℝ) (h : -Real.pi / 2 ≤ x ∧ x ≤ 0) :
    -2 ≤ f x ∧ f x ≤ 1 :=
  sorry

-- The third proof statement
theorem RangeOfTheta (θ : ℝ) (hθ : 0 < θ ∧ θ < Real.pi / 2)
    (h : ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 4 → g θ x ≤ g θ (x + 1)) :
    Real.pi / 12 ≤ θ ∧ θ ≤ Real.pi / 3 :=
  sorry

end AnalyticExpressionOfFunction_RangeOfFunction_RangeOfTheta_l683_683146


namespace largest_value_of_n_l683_683475

noncomputable def largest_n_under_200000 : ℕ :=
  if h : 199999 < 200000 ∧ (8 * (199999 - 3)^5 - 2 * 199999^2 + 18 * 199999 - 36) % 7 = 0 then 199999 else 0

theorem largest_value_of_n (n : ℕ) :
  n < 200000 → (8 * (n - 3)^5 - 2 * n^2 + 18 * n - 36) % 7 = 0 → n = 199999 :=
by sorry

end largest_value_of_n_l683_683475


namespace count_3_digit_numbers_divisible_by_13_l683_683875

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683875


namespace probability_no_adjacent_same_color_l683_683112

-- Define the problem space
def total_beads : ℕ := 9
def red_beads : ℕ := 4
def white_beads : ℕ := 3
def blue_beads : ℕ := 2

-- Define the total number of arrangements
def total_arrangements := Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial blue_beads)

-- State the probability computation theorem
theorem probability_no_adjacent_same_color :
  (∃ valid_arrangements : ℕ,
     valid_arrangements / total_arrangements = 1 / 63) := sorry

end probability_no_adjacent_same_color_l683_683112


namespace yuri_lost_card_l683_683396

theorem yuri_lost_card (n : ℕ) (remaining_sum : ℕ) : 
  n = 14 ∧ remaining_sum = 101 → 
  ∃ x : ℕ, x = (n * (n + 1)) / 2 - remaining_sum ∧ x = 4 :=
by 
  intros h,
  cases h with hn hr,
  use (n * (n + 1)) / 2 - remaining_sum,
  split,
  {
    simp only [hn, hr],
    norm_num,
  },
  {
    exact eq.refl 4,
  }

end yuri_lost_card_l683_683396


namespace inequalities_always_true_l683_683267

variables {x y a b : Real}

/-- All given conditions -/
def conditions (x y a b : Real) :=
  x < a ∧ y < b ∧ x < 0 ∧ y < 0 ∧ a > 0 ∧ b > 0

theorem inequalities_always_true {x y a b : Real} (h : conditions x y a b) :
  (x + y < a + b) ∧ 
  (x - y < a - b) ∧ 
  (x * y < a * b) ∧ 
  ((x + y) / (x - y) < (a + b) / (a - b)) :=
sorry

end inequalities_always_true_l683_683267


namespace fraction_to_decimal_l683_683418

theorem fraction_to_decimal : (3 : ℚ) / 80 = 0.0375 :=
by
  sorry

end fraction_to_decimal_l683_683418


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683589

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683589


namespace set_union_complement_eq_universal_l683_683249

open Set

variable (U M : Set ℕ)

theorem set_union_complement_eq_universal :
  U = {1, 2, 3, 4, 5, 6} →
  M = {1, 2, 4} →
  let C := U \ M in
  C ∪ M = U :=
by
  intros hU hM C_def
  rw [hU, hM, union_diff_self]
  sorry

end set_union_complement_eq_universal_l683_683249


namespace exists_sequences_l683_683258

theorem exists_sequences (n : ℕ) (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) :
  ∃ sequences : Finset (ℕ × ℕ × ℕ), sequences.card ≥ 2 * (n + 1) ∧
    ∀ ⟨x, y, z⟩ ∈ sequences, x * y * z = p^n * (x + y + z) ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
      ∀ ⟨x1, y1, z1⟩ ⟨x2, y2, z2⟩ ∈ sequences, (x1, y1, z1) ≠ (x2, y2, z2) ∧
        (x1, y1, z1) ≠ (y1, x1, z1) ∧ (x1, y1, z1) ≠ (z1, x1, y1) ∧ 
        (x1, y1, z1) ≠ (x1, z1, y1) ∧ (x1, y1, z1) ≠ (y1, z1, x1) ∧ (x1, y1, z1) ≠ (z1, y1, x1) :=
sorry

end exists_sequences_l683_683258


namespace class_students_l683_683438

theorem class_students :
  ∃ n : ℕ,
    (∃ m : ℕ, 2 * m = n) ∧
    (∃ q : ℕ, 4 * q = n) ∧
    (∃ l : ℕ, 7 * l = n) ∧
    (∀ f : ℕ, f < 6 → n - (n / 2) - (n / 4) - (n / 7) = f) ∧
    n = 28 :=
by
  sorry

end class_students_l683_683438


namespace cost_per_quart_l683_683180

theorem cost_per_quart (paint_cost : ℝ) (coverage : ℝ) (cost_to_paint_cube : ℝ) (cube_edge : ℝ) 
    (h_coverage : coverage = 1200) (h_cost_to_paint_cube : cost_to_paint_cube = 1.60) 
    (h_cube_edge : cube_edge = 10) : paint_cost = 3.20 := by 
  sorry

end cost_per_quart_l683_683180


namespace count_3digit_numbers_div_by_13_l683_683885

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683885


namespace preimage_of_point_l683_683157

-- Define the function f
def f (x y : ℝ) : ℝ × ℝ := (2 * x + y, x * y)

-- Define the specific point as per the problem statement
def point : ℝ × ℝ := (1 / 6, -1 / 6)

-- State the main theorem
theorem preimage_of_point :
  ∃ x y : ℝ, f x y = point ∧ ((x = 1 / 3 ∧ y = -1 / 2) ∨ (x = -1 / 4 ∧ y = 2 / 3)) :=
sorry

end preimage_of_point_l683_683157


namespace num_three_digit_div_by_13_l683_683838

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683838


namespace count_3_digit_numbers_divisible_by_13_l683_683917

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683917


namespace num_three_digit_div_by_13_l683_683846

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683846


namespace count_three_digit_numbers_divisible_by_13_l683_683600

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683600


namespace lost_card_number_l683_683382

theorem lost_card_number (n x : ℕ) (sum_n : ℕ) (h_sum : sum_n = n * (n + 1) / 2) (h_remaining_sum : sum_n - x = 101) : x = 4 :=
sorry

end lost_card_number_l683_683382


namespace num_pos_3_digit_div_by_13_l683_683758

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683758


namespace count_3digit_numbers_div_by_13_l683_683894

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683894


namespace counterexample_to_divisibility_by_4_l683_683482

noncomputable def generate_sequence (Q : ℕ × ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ × ℕ :=
  match Q with
  | (a, b, c, d) => (|a - b|, |b - c|, |c - d|, |d - a|)

theorem counterexample_to_divisibility_by_4 :
  ∃ (Q₀ : ℕ × ℕ × ℕ × ℕ),  
  let Q₁ := generate_sequence Q₀,
      Q₂ := generate_sequence Q₁,
      Q₃ := generate_sequence Q₂,
      Q₄ := generate_sequence Q₃ in
  (Q₄.1 % 2 = 0 ∧ Q₄.2 % 2 = 0 ∧ Q₄.3 % 2 = 0 ∧ Q₄.4 % 2 = 0) ∧
  (Q₄.1 % 4 ≠ 0 ∧ Q₄.2 % 4 ≠ 0 ∧ Q₄.3 % 4 ≠ 0 ∧ Q₄.4 % 4 ≠ 0) :=
sorry

end counterexample_to_divisibility_by_4_l683_683482


namespace three_digit_numbers_div_by_13_l683_683729

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683729


namespace Eight_teammates_points_l683_683210

noncomputable theory

variables (x y : ℕ) 

-- Define the conditions
def Linda_points := (1 / 5 : ℚ) * x
def Maria_points := (3 / 8 : ℚ) * x
def Kelly_points := 18
def Total_team_points := x
def Other_teammates_points := y
def Other_teammates_limit :=  (∀ i : ℤ, (i ≥ 1 ∧ i ≤ 8) → (Other_teammates_points ≤ 2 * 8))

-- Define the equation for total points scored
def Total_points_equation := Linda_points + Maria_points + Kelly_points + Other_teammates_points = Total_team_points

-- Theorem stating the proof problem
theorem Eight_teammates_points :
  Total_points_equation ∧ Other_teammates_limit → Other_teammates_points = 16 :=
sorry

end Eight_teammates_points_l683_683210


namespace infinite_sum_equals_fraction_l683_683105

def closest_integer_to_sqrt (n : ℕ) : ℤ :=
  if h : 0 < n then (Nat.floor (Real.sqrt n.toReal)).toInt 
  else 0

theorem infinite_sum_equals_fraction :
  (∑ n in (Finset.range (n+1)).erase 0, ((3 ^ closest_integer_to_sqrt n) + (3 ^ (- closest_integer_to_sqrt n))) / (3 ^ n))
  = 39 / 27 := by
  sorry

end infinite_sum_equals_fraction_l683_683105


namespace find_a_b_find_solution_set_l683_683160

-- Conditions
variable {a b c x : ℝ}

-- Given inequality condition
def given_inequality (x : ℝ) (a b : ℝ) : Prop := a * x^2 + x + b > 0

-- Define the solution set
def solution_set (x : ℝ) (a b : ℝ) : Prop :=
  (x < -2 ∨ x > 1) ↔ given_inequality x a b

-- Part I: Prove values of a and b
theorem find_a_b
  (H : ∀ x, solution_set x a b) :
  a = 1 ∧ b = -2 := by sorry

-- Define the second inequality
def second_inequality (x : ℝ) (c : ℝ) : Prop := x^2 - (c - 2) * x - 2 * c < 0

-- Solution set for the second inequality
def second_solution_set (x : ℝ) (c : ℝ) : Prop :=
  (c = -2 → False) ∧
  (c > -2 → -2 < x ∧ x < c) ∧
  (c < -2 → c < x ∧ x < -2)

-- Part II: Prove the solution set
theorem find_solution_set
  (H : a = 1)
  (H1 : b = -2) :
  ∀ x, second_solution_set x c ↔ second_inequality x c := by sorry

end find_a_b_find_solution_set_l683_683160


namespace three_digit_numbers_div_by_13_l683_683724

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683724


namespace jill_initial_investment_l683_683223

noncomputable def initial_investment (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  A / (1 + r / n)^(n * t)

theorem jill_initial_investment :
  initial_investment 10815.83 0.0396 2 2 ≈ 10000 :=
by
  sorry

end jill_initial_investment_l683_683223


namespace odd_function_increasing_function_l683_683563

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem odd_function (x : ℝ) : 
  (f (1 / 2) (-x)) = -(f (1 / 2) x) := 
by
  sorry

theorem increasing_function : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f (1 / 2) x₁ < f (1 / 2) x₂ := 
by
  sorry

end odd_function_increasing_function_l683_683563


namespace vector_AB_magnitude_l683_683168

/-- Define vectors OA and OB -/
def OA := (1 : ℝ, 2 : ℝ)
def OB := (3 : ℝ, 1 : ℝ)

/-- Calculate the vector AB -/
def AB := (OB.1 - OA.1, OB.2 - OA.2)

/-- Calculate the magnitude of vector AB -/
def magnitude_AB := real.sqrt (AB.1 ^ 2 + AB.2 ^ 2)

/-- Prove that vector AB is (2, -1) and its magnitude is √5 -/
theorem vector_AB_magnitude :
  AB = (2, -1) ∧ magnitude_AB = real.sqrt 5 :=
by
  sorry

end vector_AB_magnitude_l683_683168


namespace min_value_a2_plus_b2_l683_683130

-- Define the problem conditions in Lean 4.

def unit_circle_radius : ℝ := 1

-- A, B, and C are points on the circle with center O (origin) and radius 1.
-- Let vectors OA and OB form a 60 degree angle.
def O := (0 : ℝ)
def A := (1 : ℝ)  -- arbitrary point on the unit circle
def B := (exp (complex.I * π / 3))  -- another point on the unit circle at 60 degrees from A
def angle_AOB := real.angle (complex.arg B - complex.arg A) = (π / 3)

-- Let vector OC be a linear combination of OA and OB
def OC (a b : ℝ) := a * A + b * B

-- The main statement to prove.
theorem min_value_a2_plus_b2 (a b : ℝ) : ∀ a b ∈ ℝ, ((∀ a b : ℝ, 
  OC a b = a * A + b * B) 
  ∧ angle_AOB) → 
  a^2 + b^2 ≥ (2/3) := 
sorry

end min_value_a2_plus_b2_l683_683130


namespace explicit_form_of_f_monotonicity_of_f_odd_function_l683_683550

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x ^ 2 + 1)

theorem explicit_form_of_f :
  ∀ (x : ℝ), f(x) = (2 * x) / (x ^ 2 + 1) := 
sorry

theorem monotonicity_of_f :
  ∀ (x : ℝ), -1 < x → x < 1 → f' x ≥ 0 := 
  let f' (x : ℝ) : ℝ := (2 - 2 * x ^ 2) / (x ^ 2 + 1) ^ 2 in
sorry

example : f(1) = 1 := by
  simp [f]
  rw [pow_two, add_one, div_self]
  norm_num

example : f (-x) = -f x := by
  simp [f]
  rw [neg_mul, pow_two, neg_add, add_comm]
  rw [neg_div, neg_eq_iff_add_eq_zero]

theorem odd_function :
  ∀ (x : ℝ), f(-x) = -f(x) := 
  sorry

end explicit_form_of_f_monotonicity_of_f_odd_function_l683_683550


namespace count_three_digit_numbers_divisible_by_13_l683_683920

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683920


namespace count_three_digit_div_by_13_l683_683684

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683684


namespace yuri_lost_card_l683_683399

theorem yuri_lost_card (n : ℕ) (remaining_sum : ℕ) : 
  n = 14 ∧ remaining_sum = 101 → 
  ∃ x : ℕ, x = (n * (n + 1)) / 2 - remaining_sum ∧ x = 4 :=
by 
  intros h,
  cases h with hn hr,
  use (n * (n + 1)) / 2 - remaining_sum,
  split,
  {
    simp only [hn, hr],
    norm_num,
  },
  {
    exact eq.refl 4,
  }

end yuri_lost_card_l683_683399


namespace count_3_digit_numbers_divisible_by_13_l683_683910

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683910


namespace baker_sales_difference_l683_683036

theorem baker_sales_difference :
  let cakes_sold := 97 in
  let pastries_sold := 8 in
  cakes_sold - pastries_sold = 89 :=
by {
  let cakes_sold := 97
  let pastries_sold := 8
  show cakes_sold - pastries_sold = 89
  sorry
}

end baker_sales_difference_l683_683036


namespace minimum_teams_l683_683442

theorem minimum_teams (S : ℕ) (play_match : ℕ → ℕ → Prop)
      (points : ℕ → ℕ) (total_points : ℕ → ℕ → ℕ) :
  (∀ i j, i ≠ j → play_match i j) →
  (∀ x y, play_match x y → if x = y then points x = 1 else points x = 2) →
  (∃ T, (∀ t, t ∈ T → points t = maximum (points '' T)) ∧ 
        (∀ t1 t2, t1 ∈ T ∧ t2 ∈ T ∧ t1 ≠ t2 → total_points t1 ≤ total_points t2)) →
  S ≥ 6 :=
by
  sorry

end minimum_teams_l683_683442


namespace probability_point_closer_to_0_2_l683_683014

noncomputable def probability_closer_point
  (Q : ℝ × ℝ)
  (in_rect : ∀ x y, Q = (x, y) → 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 2) : Prop :=
∃ p : ℝ,
  p = 1/2 ∧
  ∀ (x y : ℝ), Q = (x, y) →
    (dist (x, y) (0, 2) < dist (x, y) (4, 2) → p = 1/2)

theorem probability_point_closer_to_0_2 :
  probability_closer_point (λ (x y : ℝ), ∀ x y, 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 2)
    sorry

end probability_point_closer_to_0_2_l683_683014


namespace arithmetic_sequence_fifth_term_l683_683204

theorem arithmetic_sequence_fifth_term (a_1 d : ℕ) (h_a_1 : a_1 = 3) (h_d : d = 4) : 
  a_1 + 4 * d = 19 :=
by 
  rw [h_a_1, h_d]
  norm_num
  sorry

end arithmetic_sequence_fifth_term_l683_683204


namespace circle_ratio_l683_683430

theorem circle_ratio (R r a c : ℝ) (hR : 0 < R) (hr : 0 < r) (h_c_lt_a : 0 < c ∧ c < a) 
  (condition : π * R^2 = (a - c) * (π * R^2 - π * r^2)) :
  R / r = Real.sqrt ((a - c) / (c + 1 - a)) :=
by
  sorry

end circle_ratio_l683_683430


namespace angle_of_inclination_l683_683074

theorem angle_of_inclination (θ : ℝ) : 
  (∃ (m : ℝ), (m = sqrt 3) ∧ (θ = real.arctan m)) → θ = π / 3 :=
by
  sorry

end angle_of_inclination_l683_683074


namespace ellen_paint_time_l683_683064

variables (time_lily time_rose time_vine time_orchid total_time : ℕ)
variable (time_to_paint_orchid : ℕ)

noncomputable def ellen_paint_time_proof : Prop :=
  let k := 213 - ((17 * 5) + (10 * 7) + (20 * 2)) in
  (time_to_paint_orchid = k / 6)

theorem ellen_paint_time :
  time_to_paint_orchid = 3 := 
by sorry

end ellen_paint_time_l683_683064


namespace count_three_digit_numbers_divisible_by_13_l683_683651

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683651


namespace board_coloring_condition_l683_683487

theorem board_coloring_condition (m n : ℕ) 
  (paint : Fin m → Fin n → Bool)
  (h : ∀ (i : Fin m) (j : Fin n), odd (neighbors_with_same_color paint i j)) : 
  even m ∨ even n := sorry

end board_coloring_condition_l683_683487


namespace yuri_lost_card_l683_683400

theorem yuri_lost_card (n : ℕ) (remaining_sum : ℕ) : 
  n = 14 ∧ remaining_sum = 101 → 
  ∃ x : ℕ, x = (n * (n + 1)) / 2 - remaining_sum ∧ x = 4 :=
by 
  intros h,
  cases h with hn hr,
  use (n * (n + 1)) / 2 - remaining_sum,
  split,
  {
    simp only [hn, hr],
    norm_num,
  },
  {
    exact eq.refl 4,
  }

end yuri_lost_card_l683_683400


namespace ratio_of_efficacy_l683_683030

-- Define original conditions
def original_sprigs_of_mint := 3
def green_tea_leaves_per_sprig := 2

-- Define new condition
def new_green_tea_leaves := 12

-- Calculate the number of sprigs of mint corresponding to the new green tea leaves in the new mud
def new_sprigs_of_mint := new_green_tea_leaves / green_tea_leaves_per_sprig

-- Statement of the theorem: ratio of the efficacy of new mud to original mud is 1:2
theorem ratio_of_efficacy : new_sprigs_of_mint = 2 * original_sprigs_of_mint :=
by
    sorry

end ratio_of_efficacy_l683_683030


namespace number_of_three_digit_numbers_divisible_by_13_l683_683950

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683950


namespace three_digit_numbers_divisible_by_13_l683_683741

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683741


namespace three_digit_numbers_divisible_by_13_count_l683_683632

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683632


namespace phi_equals_2pi_over_3_l683_683297

noncomputable def transformed_graph_phi (phi : ℝ) : ℝ :=
  sin (2 * (x - π/4) + φ)

theorem phi_equals_2pi_over_3 (phi : ℝ) (h : 0 < φ ∧ φ < π)
  (shift_eq_overlap : ∀ x, sin (2 * (x - π / 2) + φ) = sin (2 * x - π / 3)) :
  φ = 2 * π / 3 :=
by
  sorry

end phi_equals_2pi_over_3_l683_683297


namespace jills_initial_investment_l683_683228

theorem jills_initial_investment
  (A : ℝ) (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ)
  (hA : A = 10815.83)
  (hr : r = 0.0396)
  (hn : n = 2)
  (ht : t = 2)
  (compound_interest : A = P * (1 + r / n) ^ (n * t)) :
  P ≈ 10000 :=
by
  sorry

end jills_initial_investment_l683_683228


namespace percentage_saved_is_25_l683_683012

def monthly_salary : ℝ := 1000

def increase_percentage : ℝ := 0.10

def saved_amount_after_increase : ℝ := 175

def calculate_percentage_saved (x : ℝ) : Prop := 
  1000 - (1000 - (x / 100) * monthly_salary) * (1 + increase_percentage) = saved_amount_after_increase

theorem percentage_saved_is_25 :
  ∃ x : ℝ, x = 25 ∧ calculate_percentage_saved x :=
sorry

end percentage_saved_is_25_l683_683012


namespace triangle_tanA_sinB_l683_683213

theorem triangle_tanA_sinB (A B C : Type) [IsTriangle A B C] (AB BC : float)
  (hAB : AB = 25) (hBC : BC = 20) :
  ∃ (AC : float),
  (AC = Real.sqrt (AB^2 - BC^2)) ∧
  (Real.tan A = BC / AC) ∧
  (Real.sin B = AC / AB) :=
by {
  -- Assume AC is known by Pythagorean theorem
  let AC := Real.sqrt (AB^2 - BC^2),
  have hAC : AC = 15 := by {
    calc
      AC = Real.sqrt (25^2 - 20^2) : by { rw [hAB, hBC] }
      ... = Real.sqrt 625 - 400    : by { norm_num }
      ... = Real.sqrt 225          : by { norm_num }
      ... = 15                     : by { norm_num },
  },
  use AC,
  split,
  exact hAC,
  split,
  calc
    Real.tan A = 20 / 15 : by { rw [hBC, ←hAC], norm_num }
    ... = 4 / 3         : by { norm_num },
  calc
    Real.sin B = 15 / 25 : by { rw [hAB, ←hAC], norm_num }
    ... = 3 / 5         : by { norm_num },
}

end triangle_tanA_sinB_l683_683213


namespace num_three_digit_div_by_13_l683_683850

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683850


namespace sum_of_roots_l683_683178

theorem sum_of_roots (m n : ℝ) (h1 : ∀ x, x^2 - 3 * x - 1 = 0 → x = m ∨ x = n) : m + n = 3 :=
sorry

end sum_of_roots_l683_683178


namespace smaller_angle_at_3_pm_l683_683034

-- Define the condition for minute hand position at 3:00 p.m.
def minute_hand_position_at_3_pm_deg : ℝ := 0

-- Define the condition for hour hand position at 3:00 p.m.
def hour_hand_position_at_3_pm_deg : ℝ := 90

-- Define the angle between the minute hand and hour hand
def angle_between_hands (minute_deg hour_deg : ℝ) : ℝ :=
  abs (hour_deg - minute_deg)

-- The main theorem we need to prove
theorem smaller_angle_at_3_pm :
  angle_between_hands minute_hand_position_at_3_pm_deg hour_hand_position_at_3_pm_deg = 90 :=
by
  sorry

end smaller_angle_at_3_pm_l683_683034


namespace min_value_expression_l683_683109

theorem min_value_expression (x : ℝ) (h : x > 3) : 
  ∃ y : ℝ, (y = 2 * Real.sqrt 21) ∧ 
           (∀ z : ℝ, (z = (x + 18) / Real.sqrt (x - 3)) → y ≤ z) := 
sorry

end min_value_expression_l683_683109


namespace Z_cardinality_l683_683133

open Set

def A := {-1, 1}
def B := {0, 2}

def Z : Set ℤ := {z | ∃ x ∈ A, ∃ y ∈ B, z = x + y}

theorem Z_cardinality : Set.card Z = 3 := by
  sorry

end Z_cardinality_l683_683133


namespace count_3_digit_multiples_of_13_l683_683782

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683782


namespace count_three_digit_numbers_divisible_by_13_l683_683649

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683649


namespace number_of_three_digit_numbers_divisible_by_13_l683_683943

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683943


namespace three_digit_numbers_divisible_by_13_l683_683753

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683753


namespace sum_series_value_l683_683103

noncomputable def closest_int_sqrt (n : ℕ) : ℤ := 
  int.of_nat (nat.floor (real.sqrt n))

theorem sum_series_value : 
  (∑' n : ℕ, (3 ^ closest_int_sqrt n + 3 ^ -closest_int_sqrt n) / 3 ^ n) = 4.03703 := 
by 
  simp only 
  sorry

end sum_series_value_l683_683103


namespace distance_to_pinedale_mall_l683_683409

-- Define the conditions given in the problem
def average_speed : ℕ := 60  -- km/h
def stops_interval : ℕ := 5   -- minutes
def number_of_stops : ℕ := 8

-- The distance from Yahya's house to Pinedale Mall
theorem distance_to_pinedale_mall : 
  (average_speed * (number_of_stops * stops_interval / 60) = 40) :=
by
  sorry

end distance_to_pinedale_mall_l683_683409


namespace infinite_integer_and_noninteger_terms_l683_683257

theorem infinite_integer_and_noninteger_terms (m : Nat) (h_m : m > 0) :
  ∃ (infinite_int_terms : Nat → Prop) (infinite_nonint_terms : Nat → Prop),
  (∀ n, ∃ k, infinite_int_terms k ∧ ∀ k, infinite_int_terms k → ∃ N, k = n + N + 1) ∧
  (∀ n, ∃ k, infinite_nonint_terms k ∧ ∀ k, infinite_nonint_terms k → ∃ N, k = n + N + 1) :=
sorry

end infinite_integer_and_noninteger_terms_l683_683257


namespace Adam_bought_9_cat_food_packages_l683_683451

def num_cat_food_packages (c : ℕ) : Prop :=
  let cat_cans := 10 * c
  let dog_cans := 7 * 5
  cat_cans = dog_cans + 55

theorem Adam_bought_9_cat_food_packages : num_cat_food_packages 9 :=
by
  unfold num_cat_food_packages
  sorry

end Adam_bought_9_cat_food_packages_l683_683451


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683586

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683586


namespace distance_hyperbola_ellipse_at_least_9_over_7_l683_683279

theorem distance_hyperbola_ellipse_at_least_9_over_7 :
  ∀ (a : ℝ) (α : ℝ),
    0 < a →
    0 ≤ α → α ≤ π / 2 →
    (sqrt ((sqrt 6 * cos α - a)^2 + (5 / a - sin α)^2)) ≥ 9 / 7 :=
by
  intros a α ha hα1 hα2
  sorry

end distance_hyperbola_ellipse_at_least_9_over_7_l683_683279


namespace num_three_digit_div_by_13_l683_683842

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683842


namespace polar_eq_of_midpoint_line_l683_683534

-- Define the curve C and line l
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1
def transformed_curve (x y : ℝ) : Prop := x^2 + (y / 2)^2 = 1
def line_l (x y : ℝ) : Prop := 2 * x + y - 2 = 0

-- Define the intersection points and the required mid-point
def P1 : ℝ × ℝ := (1, 0)
def P2 : ℝ × ℝ := (0, 2)
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ := ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)

-- Define the proof statement to show the polar coordinate equation of the required line
theorem polar_eq_of_midpoint_line :
  let midpoint := midpoint P1 P2 in
  let perp_line (x y : ℝ) : Prop := x - 2 * y + 3 / 2 = 0 in
  ∀ ρ α, (perp_line (ρ * cos α) (ρ * sin α)) → ρ = 3 / (4 * sin α - 2 * cos α) :=
begin
  sorry
end

end polar_eq_of_midpoint_line_l683_683534


namespace angle_between_twice_a_neg_b_l683_683185

variables (a b : ℝ) (angle_between : ℝ → ℝ → ℂ → ℂ → ℝ)

-- Given: angle between a and b is 60 degrees
axiom angle_ab : angle_between a b ≔ 60

-- Prove: angle between 2a and -b is 120 degrees
theorem angle_between_twice_a_neg_b : angle_between (2 * a) (-b) = 120 :=
by sorry

end angle_between_twice_a_neg_b_l683_683185


namespace cube_no_90_degree_rotation_l683_683431

theorem cube_no_90_degree_rotation (initial_position : ℝ^3) (faces : fin 6 → set ℝ^3)
  (face_up_initial : fin 6) (rolled_times : ℕ)
  (same_position : ℝ^3) (same_face_up : fin 6) :
  rolled_times > 0 ∧ same_position = initial_position ∧ same_face_up = face_up_initial →
  ∀ orientation Δ, Δ ∉ {0, 180} → ¬ (face_up_initial = face_up_initial.rotate4(orientation)) := sorry

end cube_no_90_degree_rotation_l683_683431


namespace lost_card_l683_683367

noncomputable def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

theorem lost_card (n : ℕ) (h1 : sum_natural n - 101 < n) (h2 : sum_natural (n - 1) ≤ 101) :
  ∃ x : ℕ, x = sum_natural n - 101 ∧ x = 4 :=
begin
  have h_sum : sum_natural 14 = 105,
  {
    unfold sum_natural,
    norm_num,
  },
  use 4,
  split,
  {
    unfold sum_natural at *,
    norm_num,
    rw h_sum,
  },
  {
    norm_num,
  },
  sorry,
end

end lost_card_l683_683367


namespace count_3_digit_multiples_of_13_l683_683792

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683792


namespace solve_system_of_inequalities_l683_683509

theorem solve_system_of_inequalities (p : ℚ) : (19 * p < 10) ∧ (1 / 2 < p) → (1 / 2 < p ∧ p < 10 / 19) :=
by
  sorry

end solve_system_of_inequalities_l683_683509


namespace quadratic_vertex_l683_683569

theorem quadratic_vertex (x : ℝ) : 
  let y := -2 * (x - 1)^2 - 3 in 
  ∃ v : ℝ × ℝ, v = (1, -3) ∧ ∀ x : ℝ, y = -2 * (x - v.1)^2 + v.2 :=
by
  sorry

end quadratic_vertex_l683_683569


namespace count_three_digit_numbers_divisible_by_13_l683_683601

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683601


namespace progression_has_zero_term_l683_683998

variable (a : ℕ → ℤ) (d a1 : ℤ)
variable (n m : ℕ)

-- Condition that the sequence is arithmetic progression
def arithmetic_progression (a : ℕ → ℤ) : Prop :=
∀ k, a k = a1 + (k-1) * d

-- Condition that a_{2n}/a_{2m} = -1
def condition (a : ℕ → ℤ) (n m : ℕ) : Prop :=
a (2 * n) / a (2 * m) = -1

theorem progression_has_zero_term (a : ℕ → ℤ) (d a1 : ℤ) (n m : ℕ) 
  (h1 : arithmetic_progression a) (h2 : condition a n m) :
  ∃ k, a k = 0 ∧ k = n + m :=
sorry

end progression_has_zero_term_l683_683998


namespace simplify_complex_fraction_l683_683280

theorem simplify_complex_fraction : (3 - 2 * complex.I) / (4 + 5 * complex.I) = (2 / 41) - (23 / 41) * complex.I :=
by sorry

end simplify_complex_fraction_l683_683280


namespace three_digit_numbers_div_by_13_l683_683720

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683720


namespace find_lost_card_number_l683_683384

theorem find_lost_card_number (n : ℕ) (S : ℕ) (x : ℕ) 
  (h1 : S = n * (n + 1) / 2) 
  (h2 : S - x = 101) 
  (h3 : n = 14) : 
  x = 4 := 
by sorry

end find_lost_card_number_l683_683384


namespace power_of_n_l683_683964

theorem power_of_n (n : ℝ) (b : ℝ) (h₁ : n = 2^0.15) (h₂ : b = 19.99999999999999) : n^b = 8 := by
  sorry

end power_of_n_l683_683964


namespace number_of_3_digit_divisible_by_13_l683_683811

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683811


namespace overall_percentage_l683_683447

theorem overall_percentage (a b c : ℝ) (h1 : a = 50) (h2 : b = 60) (h3 : c = 70) :
  (a + b + c) / 3 = 60 := 
by 
  -- Use the given conditions
  have ha : a = 50 := h1,
  have hb : b = 60 := h2,
  have hc : c = 70 := h3,
  
  -- Substitute and show the calculation step-by-step
  calc
    (a + b + c) / 3
        = (50 + 60 + 70) / 3 : by rw [ha, hb, hc]
    ... = 180 / 3 : by norm_num
    ... = 60 : by norm_num

end overall_percentage_l683_683447


namespace tangent_line_equation_l683_683555

-- Define the given curve and the point
def curve (x : ℝ) : ℝ := (1 / 3) * x^3 + (4 / 3)
def P : ℝ × ℝ := (2, 4)

-- Define the proof statement
theorem tangent_line_equation :
  ∀ (x y: ℝ), (curve x = y) → 
    (x = 2) → 
    (y = 4) → 
    (4 * x - y - 4 = 0) :=
sorry

end tangent_line_equation_l683_683555


namespace number_of_3_digit_divisible_by_13_l683_683799

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683799


namespace test_scores_ordering_l683_683265

variable (M Q S Z K : ℕ)
variable (M_thinks_lowest : M > K)
variable (Q_thinks_same : Q = K)
variable (S_thinks_not_highest : S < K)
variable (Z_thinks_not_middle : (Z < S ∨ Z > M))

theorem test_scores_ordering : (Z < S) ∧ (S < Q) ∧ (Q < M) := by
  -- proof
  sorry

end test_scores_ordering_l683_683265


namespace total_number_of_coins_l683_683460

-- Define conditions
def pennies : Nat := 38
def nickels : Nat := 27
def dimes : Nat := 19
def quarters : Nat := 24
def half_dollars : Nat := 13
def one_dollar_coins : Nat := 17
def two_dollar_coins : Nat := 5
def australian_fifty_cent_coins : Nat := 4
def mexican_one_peso_coins : Nat := 12

-- Define the problem as a theorem
theorem total_number_of_coins : 
  pennies + nickels + dimes + quarters + half_dollars + one_dollar_coins + two_dollar_coins + australian_fifty_cent_coins + mexican_one_peso_coins = 159 := by
  sorry

end total_number_of_coins_l683_683460


namespace yuri_lost_card_l683_683401

theorem yuri_lost_card (n : ℕ) (remaining_sum : ℕ) : 
  n = 14 ∧ remaining_sum = 101 → 
  ∃ x : ℕ, x = (n * (n + 1)) / 2 - remaining_sum ∧ x = 4 :=
by 
  intros h,
  cases h with hn hr,
  use (n * (n + 1)) / 2 - remaining_sum,
  split,
  {
    simp only [hn, hr],
    norm_num,
  },
  {
    exact eq.refl 4,
  }

end yuri_lost_card_l683_683401


namespace count_three_digit_numbers_divisible_by_13_l683_683648

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683648


namespace count_3_digit_numbers_divisible_by_13_l683_683870

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683870


namespace strip_of_width_2_l683_683446

open Set

-- Define the concept of a strip covering
def strip_covers (w : ℝ) (S : Set (ℝ × ℝ)) : Prop :=
  ∃ l : ℝ × ℝ → ℝ, ∀ p ∈ S, abs (l p) ≤ w / 2

-- Define the problem and the corresponding proof statement
theorem strip_of_width_2 {S : Set (ℝ × ℝ)} (n : ℤ) (h1 : n ≥ 3) (h2 : S.card = n) : 
  (∀ T ⊆ S, T.card = 3 → strip_covers 1 T) → strip_covers 2 S :=
sorry

end strip_of_width_2_l683_683446


namespace range_of_b_l683_683132

theorem range_of_b (a b c : ℝ) (h1 : a + b + c = 9) (h2 : ab + bc + ca = 24) : 
  1 ≤ b ∧ b ≤ 5 := 
sorry

end range_of_b_l683_683132


namespace number_of_three_digit_numbers_divisible_by_13_l683_683944

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683944


namespace sum_of_arithmetic_progression_l683_683402

theorem sum_of_arithmetic_progression (a d : ℝ) (n : ℕ) (h_a : a = 0) (h_d : d = 1 / 3) (h_n : n = 15) :
  let S_n := n * (a + (n - 1) * d / 2) in
  S_n = 35 :=
by
  sorry

end sum_of_arithmetic_progression_l683_683402


namespace three_digit_numbers_div_by_13_l683_683734

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683734


namespace count_3_digit_numbers_divisible_by_13_l683_683871

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683871


namespace right_triangle_shorter_leg_length_l683_683979

theorem right_triangle_shorter_leg_length {a b c : ℝ} 
  (h_median : c / 2 = 15)
  (h_leg_relation : b = a + 9) 
  (h_pythagorean : a ^ 2 + b ^ 2 = c ^ 2) : 
  a = (-9 + real.sqrt 1719) / 2 :=
by 
  -- Skipping the proof
  sorry

end right_triangle_shorter_leg_length_l683_683979


namespace three_digit_numbers_divisible_by_13_l683_683748

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683748


namespace num_3_digit_div_by_13_l683_683822

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683822


namespace residue_calculation_l683_683088

theorem residue_calculation 
  (h1 : 182 ≡ 0 [MOD 14])
  (h2 : 182 * 12 ≡ 0 [MOD 14])
  (h3 : 15 * 7 ≡ 7 [MOD 14])
  (h4 : 3 ≡ 3 [MOD 14]) :
  (182 * 12 - 15 * 7 + 3) ≡ 10 [MOD 14] :=
sorry

end residue_calculation_l683_683088


namespace lost_card_number_l683_683376

theorem lost_card_number (n : ℕ) (x : ℕ) (h₁ : (n * (n + 1)) / 2 - x = 101) : x = 4 :=
sorry

end lost_card_number_l683_683376


namespace three_digit_numbers_divisible_by_13_count_l683_683620

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683620


namespace finite_composites_condition_l683_683071

def sequence (b c : ℕ) : ℕ → ℕ
| 0     := b
| 1     := c
| (n+2) := nat.abs (3 * (sequence c (n+1)) - 2 * (sequence c n))

def is_composite (n : ℕ) : Prop := 
  ∃ (d : ℕ), (d > 1) ∧ (d < n) ∧ (n % d = 0)

def num_composites (f : ℕ → ℕ) (limit : ℕ) : ℕ :=
  finset.card { n ∈ finset.range limit | is_composite (f n) }

theorem finite_composites_condition (b c : ℕ) (h1 : b > 0) (h2 : c > 0) :
  (∀ n : ℕ, num_composites (sequence b c) n ≤ some_finite_bound) ↔ (b = 1 ∧ c = 1) :=
sorry

end finite_composites_condition_l683_683071


namespace slope_y_intercept_product_l683_683011

theorem slope_y_intercept_product (x1 y1 x2 y2 : ℝ) (h1 : x1 = -2) (h2 : y1 = -3) (h3 : x2 = 3) (h4 : y2 = 4) :
  let m := (y2 - y1) / (x2 - x1),
      b := (m * -x1 + y1)
  in m * b = -7/25 :=
by
  sorry

end slope_y_intercept_product_l683_683011


namespace count_3_digit_numbers_divisible_by_13_l683_683659

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683659


namespace count_3digit_numbers_div_by_13_l683_683886

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683886


namespace count_3_digit_numbers_divisible_by_13_l683_683865

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683865


namespace power_function_monotonic_decreasing_l683_683361

theorem power_function_monotonic_decreasing (α : ℝ) (h : ∀ x y : ℝ, 0 < x → x < y → x^α > y^α) : α < 0 :=
sorry

end power_function_monotonic_decreasing_l683_683361


namespace water_consumption_comparison_l683_683492

-- Define the given conditions
def waterConsumptionWest : ℝ := 21428
def waterConsumptionNonWest : ℝ := 26848.55
def waterConsumptionRussia : ℝ := 302790.13

-- Theorem statement to prove that the water consumption per person matches the given values
theorem water_consumption_comparison :
  waterConsumptionWest = 21428 ∧
  waterConsumptionNonWest = 26848.55 ∧
  waterConsumptionRussia = 302790.13 :=
by
  -- Sorry to skip the proof
  sorry

end water_consumption_comparison_l683_683492


namespace chord_length_l683_683428

theorem chord_length (radius : ℝ) (distance_to_chord : ℝ) (chord_length : ℝ) :
  radius = 5 → distance_to_chord = 4 → chord_length = 6 :=
by
  intro hradius,
  intro hdistance_to_chord,
  simp [hradius, hdistance_to_chord],
  sorry

end chord_length_l683_683428


namespace count_three_digit_div_by_13_l683_683686

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683686


namespace log_sum_geometric_sequence_l683_683978

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, r > 0 ∧ a (n + 1) = a n * r

theorem log_sum_geometric_sequence
  (h_geom : is_geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_mean : a 5 * a 15 = (2 * Real.sqrt 2) ^ 2) :
  Real.log 2 (a 4) + Real.log 2 (a 16) = 3 :=
sorry

end log_sum_geometric_sequence_l683_683978


namespace product_remainder_mod3_l683_683504

theorem product_remainder_mod3 :
  let seq := list.range' 7 10 in
  (list.product seq) % 3 = 0 := by
    sorry

end product_remainder_mod3_l683_683504


namespace count_3_digit_multiples_of_13_l683_683790

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683790


namespace count_three_digit_div_by_13_l683_683683

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683683


namespace num_3_digit_div_by_13_l683_683832

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683832


namespace problem_solution_l683_683575

theorem problem_solution :
  ∃ (b₂ b₃ b₄ b₅ b₆ b₇ : ℤ),
    (0 ≤ b₂ ∧ b₂ < 2) ∧
    (0 ≤ b₃ ∧ b₃ < 3) ∧
    (0 ≤ b₄ ∧ b₄ < 4) ∧
    (0 ≤ b₅ ∧ b₅ < 5) ∧
    (0 ≤ b₆ ∧ b₆ < 6) ∧
    (0 ≤ b₇ ∧ b₇ < 8) ∧
    (6 / 7 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040) ∧
    (b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 11) :=
sorry

end problem_solution_l683_683575


namespace three_digit_numbers_divisible_by_13_count_l683_683628

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683628


namespace find_w_l683_683971

/-- The configuration of points and angles given in the problem -/
variables (P Q R S : Type) 
variables [collinear P R S]
variables (PQ PR QS : ℝ) (wg : ℝ)

/-- Given conditions -/
variables (hPQPR : PQ = PR) (hPRQS : PR = QS) (hangleQPR : ∠ Q P R = 30)

/-- The value of w where ∠ R Q S = 45 degrees -/
theorem find_w (h1 : PQ = PR) (h2 : PR = QS) (h3 : ∠ Q P R = 30) : ∠ R Q S = 45 :=
sorry

end find_w_l683_683971


namespace total_score_is_938_l683_683973

-- Define the average score condition
def average_score (S : ℤ) : Prop := 85.25 ≤ (S : ℚ) / 11 ∧ (S : ℚ) / 11 < 85.35

-- Define the condition that each student's score is an integer
def total_score (S : ℤ) : Prop := average_score S ∧ ∃ n : ℕ, S = n

-- Lean 4 statement for the proof problem
theorem total_score_is_938 : ∃ S : ℤ, total_score S ∧ S = 938 :=
by
  sorry

end total_score_is_938_l683_683973


namespace parabola_symmetry_l683_683551

theorem parabola_symmetry (b c : ℝ) 
  (h : ∀ x : ℝ, (1 + x)^2 + b * (1 + x) + c = (1 - x)^2 + b * (1 - x) + c) :
  let f := λ x, x^2 + b * x + c in
  f 4 > f 2 ∧ f 2 > f 1 :=
by
  -- We need to prove this theorem, but the steps and proof are not provided.
  sorry

end parabola_symmetry_l683_683551


namespace charlie_dana_ratio_l683_683039

variable (b : ℕ → ℕ → ℕ)
variable (i : Fin 50)
variable (j : Fin 60)

-- definition of row sum S_i
def S (i : Fin 50) : ℕ := ∑ j : Fin 60, b i j

-- definition of column sum T_j
def T (j : Fin 60) : ℕ := ∑ i : Fin 50, b i j

-- definition of Charlie's average C
def C : ℕ := (∑ i : Fin 50, S b i) / 50

-- definition of Dana's average D
def D : ℕ := (∑ j : Fin 60, T b j) / 60

-- the final theorem
theorem charlie_dana_ratio : C b / D b = 6 / 5 := by
  sorry

end charlie_dana_ratio_l683_683039


namespace intersection_A_B_l683_683544

-- Define the set A based on the condition x^2 <= 1
def A : Set ℝ := {x | x^2 ≤ 1}

-- Define the set B based on the condition x^2 - 2x - 3 < 0 and x in natural numbers
def B : Set ℕ := {x | x^2 - 2 * x - 3 < 0}

-- Define the expected result of the intersection of A and B
def expected_result : Set ℕ := {0, 1}

-- The theorem to prove that A ∩ B is equal to the expected result
theorem intersection_A_B : (A ∩ B : Set ℝ) = expected_result := 
by
  sorry

end intersection_A_B_l683_683544


namespace binary_to_octal_l683_683472

theorem binary_to_octal (bin : ℕ) (h : bin = 0b101101) : nat.toDigits 8 bin = [5, 5] :=
by
  sorry

end binary_to_octal_l683_683472


namespace parallelogram_proof_l683_683309

noncomputable def sin_angle_degrees (θ : ℝ) : ℝ := Real.sin (θ * Real.pi / 180)

theorem parallelogram_proof (x : ℝ) (A : ℝ) (r : ℝ) (side1 side2 : ℝ) (P : ℝ):
  (A = 972) → (r = 4 / 3) → (sin_angle_degrees 45 = Real.sqrt 2 / 2) →
  (side1 = 4 * x) → (side2 = 3 * x) →
  (A = side1 * (side2 * (Real.sqrt 2 / 2 / 3))) →
  x = 9 * 2^(3/4) →
  side1 = 36 * 2^(3/4) →
  side2 = 27 * 2^(3/4) →
  (P = 2 * (side1 + side2)) →
  (P = 126 * 2^(3/4)) :=
by
  intros
  sorry

end parallelogram_proof_l683_683309


namespace num_3_digit_div_by_13_l683_683826

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683826


namespace count_3_digit_numbers_divisible_by_13_l683_683706

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683706


namespace count_3_digit_numbers_divisible_by_13_l683_683713

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683713


namespace imaginary_part_conjugate_z_is_negative_one_l683_683554

def z : ℂ := (3 - 2 * complex.I) * (1 + complex.I)
def z_conjugate : ℂ := complex.conj z
def imaginary_part_conjugate_z : ℂ := z_conjugate.im

theorem imaginary_part_conjugate_z_is_negative_one :
  imaginary_part_conjugate_z = -1 :=
sorry

end imaginary_part_conjugate_z_is_negative_one_l683_683554


namespace tan_angle_F1PF2_l683_683549

noncomputable def ellipse (x y : ℝ) := 
  (x^2 / 4) + (y^2 / 3) = 1

def inradius (P F1 F2 : ℝ × ℝ) (r : ℝ) :=
  let d := dist P F1 + dist P F2 + dist F1 F2
  r = 1 / 2

theorem tan_angle_F1PF2
  (P F1 F2 : ℝ × ℝ)
  (hP : ellipse P.1 P.2)
  (hr : inradius P F1 F2 (1/2)) :
  tan (angle P F1 F2) = 4 / 3 := 
sorry

end tan_angle_F1PF2_l683_683549


namespace greatest_x_value_l683_683344

theorem greatest_x_value : 
  ∃ x : ℝ, (∀ y : ℝ, (y = (4 * x - 16) / (3 * x - 4)) → (y^2 + y = 12)) ∧ (x = 2) := by
  sorry

end greatest_x_value_l683_683344


namespace find_xy_l683_683073

theorem find_xy (x y : ℝ) :
  (x - 8) ^ 2 + (y - 9) ^ 2 + (x - y) ^ 2 = 1 / 3 ↔ 
  (x = 25 / 3 ∧ y = 26 / 3) :=
by
  sorry

end find_xy_l683_683073


namespace particle_position_after_12_moves_l683_683013

def cis (θ : ℝ) : ℂ := complex.exp (θ * complex.I)

noncomputable def ω : ℂ := cis (real.pi / 3)

def initial_position : ℂ := 10

def move (z : ℂ) : ℂ := ω * z + 6

def nth_move (n : ℕ) : ℂ :=
  nat.rec_on n initial_position (λ n zn, move zn)

theorem particle_position_after_12_moves : nth_move 12 = 10 :=
by
  sorry

end particle_position_after_12_moves_l683_683013


namespace num_3_digit_div_by_13_l683_683819

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683819


namespace distinct_values_S_l683_683115

def i : ℂ := Complex.I -- the imaginary unit, i

def S (n : ℤ) : ℂ := 2 * (i ^ n) + (i ^ -n)

theorem distinct_values_S : set.range S = {3, i, -3, -i} := by
  sorry

end distinct_values_S_l683_683115


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683597

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683597


namespace num_3_digit_div_by_13_l683_683837

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683837


namespace arcsin_one_half_eq_pi_six_l683_683045

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = π / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l683_683045


namespace yuri_lost_card_l683_683397

theorem yuri_lost_card (n : ℕ) (remaining_sum : ℕ) : 
  n = 14 ∧ remaining_sum = 101 → 
  ∃ x : ℕ, x = (n * (n + 1)) / 2 - remaining_sum ∧ x = 4 :=
by 
  intros h,
  cases h with hn hr,
  use (n * (n + 1)) / 2 - remaining_sum,
  split,
  {
    simp only [hn, hr],
    norm_num,
  },
  {
    exact eq.refl 4,
  }

end yuri_lost_card_l683_683397


namespace angles_count_geometric_seq_l683_683058

theorem angles_count_geometric_seq :
  let S := {θ : ℝ | 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ ∀ n : ℤ, θ ≠ n * Real.pi / 2} in
  {θ ∈ S | ∃ (f g h : ℝ), {f, g, h} = {Real.sin θ, Real.cos θ, Real.tan θ} ∧
    (g = f * (Real.sqrt ((h/g) * h) : ℝ) ∨ g = f / (Real.sqrt ((h/g) * h) : ℝ))}.toFinset.card = 4 :=
sorry

end angles_count_geometric_seq_l683_683058


namespace water_consumption_comparison_l683_683493

-- Define the given conditions
def waterConsumptionWest : ℝ := 21428
def waterConsumptionNonWest : ℝ := 26848.55
def waterConsumptionRussia : ℝ := 302790.13

-- Theorem statement to prove that the water consumption per person matches the given values
theorem water_consumption_comparison :
  waterConsumptionWest = 21428 ∧
  waterConsumptionNonWest = 26848.55 ∧
  waterConsumptionRussia = 302790.13 :=
by
  -- Sorry to skip the proof
  sorry

end water_consumption_comparison_l683_683493


namespace find_x_l683_683174

theorem find_x (x : ℝ) (h : 3^(x-4) = 27^(3/2)) : x = 8.5 :=
sorry

end find_x_l683_683174


namespace probability_red_buttons_l683_683221

theorem probability_red_buttons
  (initial_red_buttons_A initial_blue_buttons_A : ℕ)
  (remaining_ratio_A : ℚ)
  (removed_red_buttons_A removed_blue_buttons_A : ℕ) :
  initial_red_buttons_A = 6 →
  initial_blue_buttons_A = 9 →
  remaining_ratio_A = 2/3 →
  removed_red_buttons_A = 3 →
  removed_blue_buttons_A = 2 →
  let final_red_buttons_A := initial_red_buttons_A - removed_red_buttons_A in
  let final_blue_buttons_A := initial_blue_buttons_A - removed_blue_buttons_A in
  let total_buttons_A := final_red_buttons_A + final_blue_buttons_A in
  let final_red_buttons_B := removed_red_buttons_A in
  let final_blue_buttons_B := removed_blue_buttons_A in
  let total_buttons_B := final_red_buttons_B + final_blue_buttons_B in
  (total_buttons_A = 15 * remaining_ratio_A) →
  (final_red_buttons_A / total_buttons_A) * (final_red_buttons_B / total_buttons_B) = 9/50 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  sorry

end probability_red_buttons_l683_683221


namespace arithmetic_sequence_of_triangle_sides_l683_683188

theorem arithmetic_sequence_of_triangle_sides 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (angleA : 0 < A ∧ A < π)
  (angleB : 0 < B ∧ B < π)
  (angleC : 0 < C ∧ C < π)
  (h : a * (Real.cos (C / 2)) ^ 2 + c * (Real.cos (A / 2)) ^ 2 = (3 / 2) * b)
  (angle_sum : A + B + C = π) :
  a + c = 2 * b := 
begin
  sorry
end

end arithmetic_sequence_of_triangle_sides_l683_683188


namespace range_of_2a_sub_b_l683_683114

theorem range_of_2a_sub_b (a b : ℝ) (h : -1 < a ∧ a < b ∧ b < 2) : -4 < 2 * a - b ∧ 2 * a - b < 2 :=
by
  sorry

end range_of_2a_sub_b_l683_683114


namespace problem_8_9_grade_problem_10_11_grade_l683_683542

-- Define the sequences and conditions
def sequences (a b : ℕ → ℕ) : Prop :=
∀ n, a (n + 2) = a (n + 1) + a n ∧ b (n + 2) = b (n + 1) + b n

def coprime_and_bounds (a1 a2 b1 b2 : ℕ) : Prop :=
Nat.coprime a1 a2 ∧ Nat.coprime b1 b2 ∧ a1 < 1000 ∧ a2 < 1000 ∧ b1 < 1000 ∧ b2 < 1000

-- Problem statements
theorem problem_8_9_grade
  (a b : ℕ → ℕ)
  (h1 : sequences a b)
  (a1 a2 b1 b2 : ℕ)
  (h2 : coprime_and_bounds a1 a2 b1 b2)
  (h3 : a 0 = a1) (h4 : a 1 = a2)
  (h5 : b 0 = b1) (h6 : b 1 = b2) :
  (∀ n, b n ∣ a n → n < 50) :=
sorry

theorem problem_10_11_grade
  (a b : ℕ → ℕ)
  (h1 : sequences a b)
  (a1 a2 b1 b2 : ℕ)
  (h2 : coprime_and_bounds a1 a2 b1 b2)
  (h3 : a 0 = a1) (h4 : a 1 = a2)
  (h5 : b 0 = b1) (h6 : b 1 = b2) :
  (∀ n, b n ∣ (a n ^ 100) → n < 5000) :=
sorry

end problem_8_9_grade_problem_10_11_grade_l683_683542


namespace count_3_digit_numbers_divisible_by_13_l683_683916

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683916


namespace num_three_digit_div_by_13_l683_683851

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683851


namespace compute_expression_l683_683049

def diamond (a b : ℝ) : ℝ := a - (1 / b)

theorem compute_expression : ((diamond (diamond 4 5) 6) - (diamond 4 (diamond 5 6))) = (-139 / 870) :=
by
  -- Here, we would provide a detailed proof.
  sorry

end compute_expression_l683_683049


namespace gas_volume_correct_l683_683414

def mass_Na2SO3 := 14.175 -- in grams
def molar_mass_Na2SO3 := 126 -- in grams per mole
def molar_volume := 22.4 -- in liters per mole

def n_Na2SO3 (m : ℝ) (M : ℝ) : ℝ :=
  m / M

def n_SO2 (n_na2so3: ℝ) : ℝ :=
  n_na2so3

def volume_SO2 (n_so2 : ℝ) (V_m : ℝ) : ℝ :=
  n_so2 * V_m

theorem gas_volume_correct :
  volume_SO2 (n_SO2 (n_Na2SO3 mass_Na2SO3 molar_mass_Na2SO3)) molar_volume = 2.52 :=
by
  sorry

end gas_volume_correct_l683_683414


namespace angle_ACB_90_l683_683987

def A : ℝ × ℝ := (4, 2)
def B : ℝ × ℝ := (1, -2)

def on_x_axis (x : ℝ) : ℝ × ℝ := (x, 0)

theorem angle_ACB_90 (x : ℝ) :
  (∃ C : ℝ × ℝ, C = on_x_axis x ∧ ∠(A, C, B) = 90) →
  (x = 0 ∨ x = 5) := 
sorry

end angle_ACB_90_l683_683987


namespace number_of_valid_house_numbers_l683_683484

def is_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ Prime n

def digit_sum_odd (n : ℕ) : Prop :=
  (n / 10 + n % 10) % 2 = 1

def valid_house_number (W X Y Z : ℕ) : Prop :=
  W ≠ 0 ∧ X ≠ 0 ∧ Y ≠ 0 ∧ Z ≠ 0 ∧
  is_two_digit_prime (10 * W + X) ∧ is_two_digit_prime (10 * Y + Z) ∧
  10 * W + X ≠ 10 * Y + Z ∧
  10 * W + X < 60 ∧ 10 * Y + Z < 60 ∧
  digit_sum_odd (10 * W + X)

theorem number_of_valid_house_numbers : ∃ n, n = 108 ∧
  (∀ W X Y Z, valid_house_number W X Y Z → valid_house_number_count = 108) :=
sorry

end number_of_valid_house_numbers_l683_683484


namespace inequality_proof_l683_683119

-- Define the given conditions
def a : ℝ := Real.log 0.99
def b : ℝ := Real.exp 0.1
def c : ℝ := Real.exp (Real.log 0.99) ^ Real.exp 1

-- State the goal to be proved
theorem inequality_proof : a < c ∧ c < b := 
by
  sorry

end inequality_proof_l683_683119


namespace equivalent_expression_l683_683359

theorem equivalent_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 :=
sorry

end equivalent_expression_l683_683359


namespace count_3_digit_multiples_of_13_l683_683786

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683786


namespace solve_equation_l683_683283

theorem solve_equation (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^(2 * y - 1) + (x + 1)^(2 * y - 1) = (x + 2)^(2 * y - 1) ↔ (x = 1 ∧ y = 1) := by
  sorry

end solve_equation_l683_683283


namespace correct_propositions_l683_683027

-- Definitions based on conditions
def diameter_perpendicular_bisects_chord (d : ℝ) (c : ℝ) : Prop :=
  ∃ (r : ℝ), d = 2 * r ∧ c = r

def triangle_vertices_determine_circle (a b c : ℝ) : Prop :=
  ∃ (O : ℝ), O = (a + b + c) / 3

def cyclic_quadrilateral_diagonals_supplementary (a b c d : ℕ) : Prop :=
  a + b + c + d = 360 -- incorrect statement

def tangent_perpendicular_to_radius (r t : ℝ) : Prop :=
  r * t = 1 -- assuming point of tangency

-- Theorem based on the problem conditions
theorem correct_propositions :
  diameter_perpendicular_bisects_chord 2 1 ∧
  triangle_vertices_determine_circle 1 2 3 ∧
  ¬ cyclic_quadrilateral_diagonals_supplementary 90 90 90 90 ∧
  tangent_perpendicular_to_radius 1 1 :=
by
  sorry

end correct_propositions_l683_683027


namespace three_digit_numbers_divisible_by_13_l683_683746

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683746


namespace num_suitable_n_eq_336_l683_683958

theorem num_suitable_n_eq_336 :
  (∃ n : ℕ, n > 0 ∧ n % 8 = 0 ∧ Nat.lcm 40320 n = 8 * Nat.gcd 479001600 n) = 336 := 
sorry

end num_suitable_n_eq_336_l683_683958


namespace problem_l683_683525

theorem problem (f : ℝ → ℝ) (h : ∀ x, (x - 3) * (deriv f x) ≤ 0) : 
  f 0 + f 6 ≤ 2 * f 3 := 
sorry

end problem_l683_683525


namespace number_of_three_digit_numbers_divisible_by_13_l683_683957

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683957


namespace count_3_digit_multiples_of_13_l683_683793

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683793


namespace count_three_digit_numbers_divisible_by_13_l683_683928

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683928


namespace yuri_lost_card_l683_683398

theorem yuri_lost_card (n : ℕ) (remaining_sum : ℕ) : 
  n = 14 ∧ remaining_sum = 101 → 
  ∃ x : ℕ, x = (n * (n + 1)) / 2 - remaining_sum ∧ x = 4 :=
by 
  intros h,
  cases h with hn hr,
  use (n * (n + 1)) / 2 - remaining_sum,
  split,
  {
    simp only [hn, hr],
    norm_num,
  },
  {
    exact eq.refl 4,
  }

end yuri_lost_card_l683_683398


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683579

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683579


namespace count_3_digit_numbers_divisible_by_13_l683_683863

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683863


namespace count_3digit_numbers_div_by_13_l683_683883

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683883


namespace correct_calculation_option_B_l683_683356

theorem correct_calculation_option_B :
  (∀ x y : ℝ, √x * √y = √(x * y)) →
  (√3 - √2 ≠ 1) ∧
  (∀ x y : ℝ, √x - √y ≠ 1) ∧
  (√8 ≠ 4 * √2) ∧
  (√((-5 : ℝ) ^ 2) ≠ -5) ∧
  (√3 * √2 = √6) :=
by 
  -- Root multiplication property
  intro h_sqrt_mul 
  split 
  -- Option A: √3 - √2 ≠ 1
  { norm_num, linarith, exact 1 }
  split
  -- General case: ∀ x y : ℝ, √x - √y ≠ 1
  { intros x y, norm_num, linarith, exact 1 }
  split
  -- Option C: √8 ≠ 4 * √2
  { norm_num, linarith, exact 1 }
  split
  -- Option D: √((-5) ^ 2) ≠ -5
  { norm_num, linarith, exact 1 }
  -- Option B: √3 * √2 = √6 using the root multiplication property
  { apply h_sqrt_mul, exact 3, exact 2 }
sorry

end correct_calculation_option_B_l683_683356


namespace problem_1_problem_2_problem_3_l683_683571

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def P : Set ℝ := {x | x ≤ 0 ∨ x ≥ 5 / 2}

theorem problem_1 : A ∩ B = {x | -1 < x ∧ x < 2} := sorry

theorem problem_2 : compl B ∪ P = {x | x ≤ 0 ∨ x ≥ 5 / 2} := sorry

theorem problem_3 : (A ∩ B) ∩ compl P = {x | 0 < x ∧ x < 2} := sorry

end problem_1_problem_2_problem_3_l683_683571


namespace kyle_paper_delivery_l683_683240

-- Define the number of houses Kyle delivers to from Monday through Saturday
def housesDaily : ℕ := 100

-- Define the number of days from Monday to Saturday
def daysWeek : ℕ := 6

-- Define the adjustment for Sunday (10 fewer, 30 additional)
def sundayAdjust : ℕ := 30 - 10 + 100

-- Define the total number of papers delivered from Monday to Saturday
def papersMonToSat : ℕ := daysWeek * housesDaily

-- Define the total number of papers delivered on Sunday
def papersSunday : ℕ := sundayAdjust

-- Define the total number of papers delivered each week
def totalPapers : ℕ := papersMonToSat + papersSunday

-- The theorem we want to prove
theorem kyle_paper_delivery : totalPapers = 720 := by
  -- We are focusing only on the statement here.
  sorry

end kyle_paper_delivery_l683_683240


namespace count_three_digit_div_by_13_l683_683688

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683688


namespace lost_card_number_l683_683395

variable (n : ℕ)
variable (s : ℕ)

-- Axioms based on the given conditions.
axiom sum_of_first_n : s = n * (n + 1) / 2
axiom remaining_sum : s - 101 ∈ finset.singleton 101

-- The theorem we need to prove:
theorem lost_card_number (n : ℕ) (s : ℕ) (h₁ : s = n * (n + 1) / 2) (h₂ : s - 101 = 101) : n = 14 → (n * (n + 1) / 2 - 101 = 4) :=
by
  sorry

end lost_card_number_l683_683395


namespace chickens_in_zoo_l683_683061

theorem chickens_in_zoo (c e : ℕ) (h_legs : 2 * c + 4 * e = 66) (h_heads : c + e = 24) : c = 15 :=
by
  sorry

end chickens_in_zoo_l683_683061


namespace num_three_digit_div_by_13_l683_683855

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683855


namespace find_a2023_l683_683543

noncomputable def seq (n : ℕ) : ℤ :=
  nat.rec_on n
    0 -- base case
    (λ n a_n, -|a_n + (n + 1)|) -- recursive definition

theorem find_a2023 : seq 2023 = -1011 :=
  sorry

end find_a2023_l683_683543


namespace expression_has_linear_factor_l683_683474

theorem expression_has_linear_factor (x y z : ℤ) :
  ∃ (a b c : ℤ), x^2 - (y + z)^2 + 2x + y - z = (x - y - z) * (a + b + c) :=
sorry

end expression_has_linear_factor_l683_683474


namespace sequence_periodic_l683_683524

def last_digit (n : ℕ) : ℕ := n % 10

noncomputable def a_n (n : ℕ) : ℕ := last_digit (n^(n^n))

theorem sequence_periodic :
  ∃ period : ℕ, period = 20 ∧ ∀ n m : ℕ, n ≡ m [MOD period] → a_n n = a_n m :=
sorry

end sequence_periodic_l683_683524


namespace count_3_digit_numbers_divisible_by_13_l683_683899

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683899


namespace isosceles_trapezoid_area_l683_683343

theorem isosceles_trapezoid_area (a b c : ℝ) (h : ℝ) :
  a = 5 ∧ b = 7 ∧ c = 13 ∧ h = 4 → 
  (1 / 2) * (b + c) * h = 40 :=
by
  intros
  apply And.intro
  { exact rfl }
  apply And.intro
  { exact rfl }
  apply And.intro
  { exact rfl }
  sorry

end isosceles_trapezoid_area_l683_683343


namespace num_three_digit_div_by_13_l683_683852

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683852


namespace three_digit_numbers_divisible_by_13_l683_683742

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683742


namespace count_3_digit_numbers_divisible_by_13_l683_683715

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683715


namespace dot_product_EC_ED_l683_683984

-- Using definitions derived from the conditions
def is_midpoint (A B E : Point) : Prop := 
  dist A E = dist E B ∧ dist E B = dist A B / 2

def square (A B C D : Point) (s : ℝ) : Prop := 
  dist A B = s ∧ dist B C = s ∧ dist C D = s ∧ dist D A = s ∧
  dist A C = sqrt (s^2 + s^2) ∧ dist B D = sqrt (s^2 + s^2)

-- Using variables and the actual theorem statement
variables (A B C D E : Point) (s : ℝ)
variables (hABCD : square A B C D s) (hE_midpoint : is_midpoint A B E)

theorem dot_product_EC_ED : 
  s = 2 → \(\overrightarrow{EC}\cdot \overrightarrow{ED} = 3\) := 
sorry

end dot_product_EC_ED_l683_683984


namespace tv_purchase_price_l683_683445

theorem tv_purchase_price 
  (x : ℝ) 
  (marked_price : ℝ) 
  (final_selling_price : ℝ) 
  (profit : ℝ) 
  (markup: marked_price = 1.35 * x)
  (discount: final_selling_price = 0.9 * marked_price - 50)
  (profit_eq: profit = final_selling_price - x)
  (given_profit: profit = 208) : 
  x = 1200 := 
by {
  have priced_eq : final_selling_price = 0.9 * 1.35 * x - 50,
  { rw [discount, markup] },
  have simplified_profit_eq : 0.215 * x = 258,
  { linarith [profit_eq, given_profit, priced_eq] },
  exact eq_of_mult_eq x 1200 0.215 258 simplified_profit_eq,
}

end tv_purchase_price_l683_683445


namespace simplify_f_alpha_value_f_alpha_l683_683532

open Real

-- Function definition
def f (α : ℝ) : ℝ := 
  (sin (7 * π - α) * cos (α + 3 * π / 2) * cos (3 * π + α)) / 
  (sin (α - 3 * π / 2) * cos (α + 5 * π / 2) * tan (α - 5 * π))

-- Simplification proof
theorem simplify_f_alpha (α : ℝ) : f α = cos α := by
  sorry

-- Condition for the second part
variable (α : ℝ)
axiom cos_condition : cos (3 * π / 2 + α) = 1 / 7
axiom alpha_in_second_quadrant : π / 2 < α ∧ α < π

-- Proof that under these conditions, the function evaluates to the specified value
theorem value_f_alpha (h1 : cos (3 * π / 2 + α) = 1 / 7) (h2 : π / 2 < α ∧ α < π) : f α = - (4 * sqrt 3) / 7 := by
  sorry

end simplify_f_alpha_value_f_alpha_l683_683532


namespace arithmetic_sequence_sum_l683_683128

theorem arithmetic_sequence_sum {a : ℕ → ℤ} (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0) (h2 : a 2 = 1) (h3 : a 3 = 3) : 
  let S_n := λ n, (n * (a 1 + a n)) / 2 
  in S_n 4 = 8 :=
sorry

end arithmetic_sequence_sum_l683_683128


namespace count_3_digit_numbers_divisible_by_13_l683_683904

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683904


namespace three_digit_numbers_divisible_by_13_count_l683_683637

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683637


namespace no_hiphop_or_contemporary_not_in_slow_l683_683329

namespace DanceProblem

-- Defining the conditions
def total_kids : ℕ := 140
def fraction_dancers : ℚ := 1 / 4
def total_dancers : ℕ := (total_kids * fraction_dancers).natAbs
def ratio_slow_hiphop_contemporary : ℚ := 5 / 10 + 3 / 10 + 2 / 10
def slow_dance_ratio : ℚ := 5 / ratio_slow_hiphop_contemporary
def slow_dance_kids : ℕ := 25
def ratio_part_kids : ℕ := (slow_dance_kids / 5).natAbs
def hiphop_kids : ℕ := (3 * ratio_part_kids).natAbs
def contemporary_kids : ℕ := (2 * ratio_part_kids).natAbs
def total_dancers_minus_slow : ℕ := total_dancers - slow_dance_kids

-- Theorem statement
theorem no_hiphop_or_contemporary_not_in_slow :
  (hiphop_kids + contemporary_kids) - (total_dancers_minus_slow) = 0 :=
by
  sorry

end DanceProblem

end no_hiphop_or_contemporary_not_in_slow_l683_683329


namespace tangent_line_equation_g_extrema_l683_683250

noncomputable theory

open Real

variables (a b : ℝ)
def f (x : ℝ) : ℝ := x^3 + a * x^2 + b * x + 1
def f' (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b
def g (x : ℝ) : ℝ := f' x * exp (-x)

-- Given conditions
axiom f'_1 : 2 * a = f' 1 
axiom f'_2 : -b = f' 2

theorem tangent_line_equation : 
  tangent_line f 1 = (6 : ℝ) * (x : ℝ) + 2 * (y : ℝ) - 1 :=
sorry

theorem g_extrema :
  ∃ (min max : ℝ), min = g 0 ∧ max = g 3 ∧ min = -3 ∧ max = 15 * exp(-3) :=
sorry

end tangent_line_equation_g_extrema_l683_683250


namespace Kyle_papers_delivered_each_week_proof_l683_683233

-- Definitions based on identified conditions
def k_m := 100        -- Number of papers delivered from Monday to Saturday
def d_m := 6          -- Number of days from Monday to Saturday
def k_s1 := 90        -- Number of regular customers on Sunday
def k_s2 := 30        -- Number of Sunday-only customers

-- Total number of papers delivered in a week
def total_papers_week := (k_m * d_m) + (k_s1 + k_s2)

theorem Kyle_papers_delivered_each_week_proof :
  total_papers_week = 720 :=
by
  sorry

end Kyle_papers_delivered_each_week_proof_l683_683233


namespace freight_train_departures_l683_683209

-- Definitions conforming with the identified conditions
def train := { 'A', 'B', 'C', 'D', 'E', 'F' }
def group1 : Finset (Finset char) := { { 'A', 'B', 'C' }, { 'A', 'B', 'D' }, { 'A', 'B', 'E' }, { 'A', 'B', 'F' } }

-- Theorem statement to prove the total number of different possible departure sequences is 144
theorem freight_train_departures :
  let group_permutations (g : Finset char) : ℕ := (Finset.toList g).perm.count
  ∑ g in group1, group_permutations g *
  group_permutations (train \ g) = 144 :=
by
  sorry

end freight_train_departures_l683_683209


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683593

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683593


namespace infinite_sum_equals_fraction_l683_683106

def closest_integer_to_sqrt (n : ℕ) : ℤ :=
  if h : 0 < n then (Nat.floor (Real.sqrt n.toReal)).toInt 
  else 0

theorem infinite_sum_equals_fraction :
  (∑ n in (Finset.range (n+1)).erase 0, ((3 ^ closest_integer_to_sqrt n) + (3 ^ (- closest_integer_to_sqrt n))) / (3 ^ n))
  = 39 / 27 := by
  sorry

end infinite_sum_equals_fraction_l683_683106


namespace EC_dot_ED_eq_three_l683_683985

-- Define the setup for the problem
variables {A B C D E : Type} [metric_space A] [metric_space B]
variables (A B C D : point) (E : point)
variable (side_length : length)
variable (midpoint : point)

-- Define the conditions
def square (A B C D : point) (side_length : ℝ) : Prop := 
  is_square A B C D side_length

def is_midpoint (E : point) (A B : point) : Prop :=
  midpoint E A B

-- State the problem
theorem EC_dot_ED_eq_three
  (h1 : square A B C D 2) 
  (h2 : is_midpoint E A B) : 
  (vector_between_points E C) • (vector_between_points E D) = 3 := 
sorry

end EC_dot_ED_eq_three_l683_683985


namespace number_of_three_digit_numbers_divisible_by_13_l683_683954

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683954


namespace evaluate_series_l683_683093

noncomputable def closest_int_sqrt (n : ℕ) : ℤ :=
  round (real.sqrt n)

theorem evaluate_series : 
  (∑' (n : ℕ) in set.Icc 1 (nat.mul (set.Ioi 1)) (3 ^ (closest_int_sqrt n) + 3 ^ -(closest_int_sqrt n)) / 3 ^ n) = 3 :=
begin
  sorry
end

end evaluate_series_l683_683093


namespace num_pos_3_digit_div_by_13_l683_683760

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683760


namespace count_three_digit_div_by_13_l683_683682

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683682


namespace num_pos_3_digit_div_by_13_l683_683762

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683762


namespace Materik_position_correct_l683_683208

-- Define the alphabet and the known positions
variable alph : List Char := ['A', 'E', 'I', 'K', 'M', 'R', 'T']
variable pos_Metrika : Nat := 3634
variable pos_Materik : Nat := 3745

-- State the problem as an equivalent theorem in Lean 4
theorem Materik_position_correct :
  (alphabetical_position "Metrika" alph pos_Metrika) →
  (alphabetical_position "Materik" alph = pos_Materik) :=
by sorry

end Materik_position_correct_l683_683208


namespace jills_initial_investment_l683_683226

theorem jills_initial_investment
  (A : ℝ) (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ)
  (hA : A = 10815.83)
  (hr : r = 0.0396)
  (hn : n = 2)
  (ht : t = 2)
  (compound_interest : A = P * (1 + r / n) ^ (n * t)) :
  P ≈ 10000 :=
by
  sorry

end jills_initial_investment_l683_683226


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683580

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683580


namespace smallest_positive_period_strictly_increasing_intervals_max_min_values_l683_683557

def f (x : ℝ) : ℝ := 4 * cos x * sin (x + π / 6) - 1

-- Statement 1: Smallest positive period
theorem smallest_positive_period : ∀ x : ℝ, f (x + π) = f x := sorry

-- Statement 2: Intervals where f(x) is strictly increasing
theorem strictly_increasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, (k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6) → strict_mono_on f (set.Icc (k * π - π / 3) (k * π + π / 6)) := sorry

-- Statement 3: Maximum and Minimum values on specific interval
theorem max_min_values :
  ∀ x : ℝ, (x ∈ set.Icc (-π / 6) (π / 4)) → (f x ≥ -1 ∧ f x ≤ 2) := sorry

end smallest_positive_period_strictly_increasing_intervals_max_min_values_l683_683557


namespace angle_between_2a_neg_b_l683_683182

variables (a b : ℝ → ℝ) -- treating 'a' and 'b' as vectors in a real-valued space
variables (angle : ℝ → ℝ → ℝ) -- function to denote the angle between two vectors

-- defining the angle condition
def angle_between_a_b : Prop := angle a b = 60

-- defining the goal: angle between 2a and -b
theorem angle_between_2a_neg_b (ha : angle a b = 60) : angle (λ x, 2 * a x) (λ x, - b x) = 120 := 
by {
  -- provided condition
  sorry
}

end angle_between_2a_neg_b_l683_683182


namespace simple_and_compound_interest_difference_l683_683354

theorem simple_and_compound_interest_difference 
  (R : ℝ) (T : ℝ) (diff : ℝ) (P : ℝ) :
  R = 8 →
  T = 4 →
  diff = 40.48896000000036 →
  (diff = (0.36048896 * P) - (0.32 * P)) →
  P = 1000 :=
  by 
    intros hR hT hdiff heq_diff 
    rw [hR, hT, hdiff] at heq_diff
    sorry

end simple_and_compound_interest_difference_l683_683354


namespace evaluate_series_l683_683092

noncomputable def closest_int_sqrt (n : ℕ) : ℤ :=
  round (real.sqrt n)

theorem evaluate_series : 
  (∑' (n : ℕ) in set.Icc 1 (nat.mul (set.Ioi 1)) (3 ^ (closest_int_sqrt n) + 3 ^ -(closest_int_sqrt n)) / 3 ^ n) = 3 :=
begin
  sorry
end

end evaluate_series_l683_683092


namespace count_3_digit_numbers_divisible_by_13_l683_683662

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683662


namespace complex_div_real_imag_opposites_l683_683166

theorem complex_div_real_imag_opposites (b : ℝ) :
  let z1 := complex.of_real 1 + complex.I * b,
      z2 := complex.of_real (-2) + complex.I in
    (z1 / z2).re = - (z1 / z2).im → b = -1/3 := by
  let z1 := ⟨1, b⟩
  let z2 := ⟨-2, 1⟩
  have hz1 : z1 = complex.of_real 1 + complex.I * b, from rfl
  have hz2 : z2 = complex.of_real (-2) + complex.I, from rfl
  calc
    (z1 / z2).re = - (z1 / z2).im : sorry
    b = -1/3 : sorry

end complex_div_real_imag_opposites_l683_683166


namespace simplify_fraction_rationalize_denominator_l683_683282

theorem simplify_fraction_rationalize_denominator :
  let a := sqrt 75
  let b := 3 * sqrt 48
  let c := sqrt 27
  a = 5 * sqrt 3 → b = 12 * sqrt 3 → c = 3 * sqrt 3 →
  (5 / (a + b + c)) = (sqrt 3 / 12) := by
  intros ha hb hc
  sorry

end simplify_fraction_rationalize_denominator_l683_683282


namespace largest_of_three_consecutive_integers_sum_18_l683_683319

theorem largest_of_three_consecutive_integers_sum_18 (n : ℤ) (h : n + (n + 1) + (n + 2) = 18) : n + 2 = 7 :=
by
  sorry

end largest_of_three_consecutive_integers_sum_18_l683_683319


namespace coterminal_angle_l683_683993

theorem coterminal_angle (theta : ℝ) (lower : ℝ) (upper : ℝ) (k : ℤ) : 
  -950 = k * 360 + theta ∧ (lower ≤ theta ∧ theta ≤ upper) → theta = 130 :=
by
  -- Given conditions
  sorry

end coterminal_angle_l683_683993


namespace num_pos_3_digit_div_by_13_l683_683772

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683772


namespace number_of_real_solutions_l683_683476

theorem number_of_real_solutions :
  (∀ x : ℝ, 2 ^ (x^2 - 8 * x + 15) = 1 → (x = 3 ∨ x = 5)) ∧ 
  (∃! x1 x2 : ℝ, (x1 = 3 ∨ x1 = 5) ∧ (x2 = 3 ∨ x2 = 5) ∧ x1 ≠ x2) := 
sorry

end number_of_real_solutions_l683_683476


namespace vector_magnitude_l683_683567

variable {a b : ℝ}
def condition1 (a b : ℝ) : Prop := b * (a + b) = 3
def condition2 : a = 1 := rfl
def condition3 : b = 2 := rfl

theorem vector_magnitude : condition1 a b → condition2 → condition3 → |a + b| = Real.sqrt 3 :=
by
  intros
  unfold condition1 at *
  rw [condition2, condition3] at *
  -- more rewrite steps if needed
  sorry

end vector_magnitude_l683_683567


namespace count_3digit_numbers_div_by_13_l683_683879

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683879


namespace number_of_3_digit_divisible_by_13_l683_683800

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683800


namespace grid_path_count_l683_683171

theorem grid_path_count (n : ℕ) : 
  (finset.univ.filter (λ (p : list (ℕ×ℕ)), p.length = 2*n ∧ (p.filter (λ x, x = (1, 0))).length = n ∧ (p.filter (λ x, x = (0, 1))).length = n)).card = nat.choose (2 * n) n := by
  sorry

end grid_path_count_l683_683171


namespace sandbag_weight_l683_683333

theorem sandbag_weight (s : ℝ) (f : ℝ) (h : ℝ) : 
  f = 0.75 ∧ s = 450 ∧ h = 0.65 → f * s + h * (f * s) = 556.875 :=
by
  intro hfs
  sorry

end sandbag_weight_l683_683333


namespace children_l683_683043

def children's_ticket_cost : ℝ :=
  let adult_ticket_cost := 12 in
  let total_bill := 138 in
  let total_tickets := 12 in
  let additional_children_tickets := 8 in
  let number_of_adult_tickets := (total_tickets - additional_children_tickets) / 2 in
  let number_of_children_tickets := number_of_adult_tickets + additional_children_tickets in
  let total_adult_cost := number_of_adult_tickets * adult_ticket_cost in
  let total_children_cost := total_bill - total_adult_cost in
  total_children_cost / number_of_children_tickets

theorem children's_ticket_cost_correct : children's_ticket_cost = 11.40 :=
  by sorry

end children_l683_683043


namespace total_dots_not_visible_l683_683331

theorem total_dots_not_visible :
  let dice_faces := [1, 2, 3, 4, 5, 6]
  let total_dots_per_die := (List.sum dice_faces)
  let total_dots := 3 * total_dots_per_die
  let visible_dots := [1, 3, 4, 5, 6]
  let sum_visible_dots := (List.sum visible_dots)
  total_dots - sum_visible_dots = 44 :=
by
  let dice_faces := [1, 2, 3, 4, 5, 6]
  let total_dots_per_die := (List.sum dice_faces)
  let total_dots := 3 * total_dots_per_die
  let visible_dots := [1, 3, 4, 5, 6]
  let sum_visible_dots := (List.sum visible_dots)
  show total_dots - sum_visible_dots = 44 from sorry

end total_dots_not_visible_l683_683331


namespace value_of_a4_l683_683992

variables {a : ℕ → ℝ} -- Define the sequence as a function from natural numbers to real numbers.

-- Conditions: The sequence is geometric, positive and satisfies the given product condition.
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n k, a (n + k) = (a n) * (a k)

-- Condition: All terms are positive.
def all_terms_positive (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

-- Given product condition:
axiom a1_a7_product : a 1 * a 7 = 36

-- The theorem to prove:
theorem value_of_a4 (h_geo : is_geometric_sequence a) (h_pos : all_terms_positive a) : 
  a 4 = 6 :=
sorry

end value_of_a4_l683_683992


namespace count_3_digit_numbers_divisible_by_13_l683_683717

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683717


namespace find_width_of_new_section_l683_683441

-- Definitions and conditions
def area (length width : ℝ) : ℝ :=
  length * width

-- The problem statement as a Lean theorem
theorem find_width_of_new_section (area_new_section : ℝ) (length_new_section : ℝ) 
  (h1 : area_new_section = 35) (h2 : length_new_section = 5) : ∃ width_new_section : ℝ, width_new_section = 7 :=
begin
  sorry
end

end find_width_of_new_section_l683_683441


namespace count_3_digit_multiples_of_13_l683_683781

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683781


namespace count_3_digit_numbers_divisible_by_13_l683_683908

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683908


namespace Kyle_papers_delivered_each_week_l683_683237

-- Definitions representing the conditions
def papers_per_day := 100
def days_Mon_to_Sat := 6
def regular_Sunday_customers := 100
def non_regular_Sunday_customers := 30
def no_delivery_customers_on_Sunday := 10

-- The total number of papers delivered each week
def total_papers_per_week : ℕ :=
  days_Mon_to_Sat * papers_per_day +
  regular_Sunday_customers - no_delivery_customers_on_Sunday + non_regular_Sunday_customers

-- Prove that Kyle delivers 720 papers each week
theorem Kyle_papers_delivered_each_week : total_papers_per_week = 720 :=
sorry

end Kyle_papers_delivered_each_week_l683_683237


namespace tan_squared_sum_prism_l683_683536

theorem tan_squared_sum_prism {α β γ : ℝ} 
  (all_edges_equal_length : ∀ {x y : ℝ}, x = y) 
  (intersect_height : ∃ p : ℝ, p > 0) 
  (angles_defined : ∀ {x : ℝ}, x = α ∨ x = β ∨ x = γ) : 
  tan α ^ 2 + tan β ^ 2 + tan γ ^ 2 = 12 :=
sorry

end tan_squared_sum_prism_l683_683536


namespace students_and_swimmers_l683_683439

theorem students_and_swimmers (N : ℕ) (x : ℕ) 
  (h1 : x = N / 4) 
  (h2 : x / 2 = 4) : 
  N = 32 ∧ N - x = 24 := 
by 
  sorry

end students_and_swimmers_l683_683439


namespace father_age_l683_683090

theorem father_age : 
  let S := 40 in
  let Si := S - 10 in
  let B := Si - 7 in
  let S_5_years_ago := S - 5 in
  let Si_5_years_ago := Si - 5 in
  let B_5_years_ago := B - 5 in
  S_5_years_ago + Si_5_years_ago + B_5_years_ago = (3 / 4 : ℝ) * (F_5_years_ago : ℝ) →
  (4 / 3 : ℝ) * (S_5_years_ago + Si_5_years_ago + B_5_years_ago) + 5 = 109 :=
by
  intros S Si B S_5_years_ago Si_5_years_ago B_5_years_ago h
  sorry

end father_age_l683_683090


namespace isosceles_triangle_largest_angle_l683_683198

theorem isosceles_triangle_largest_angle (A B C : Type) (α β γ : ℝ)
  (h_iso : α = β) (h_angles : α = 50) (triangle: α + β + γ = 180) : γ = 80 :=
sorry

end isosceles_triangle_largest_angle_l683_683198


namespace problem_a_problem_b_l683_683403

-- Problem (a) Lean statement
theorem problem_a (n m : ℕ) :
  (∃ (x : Fin m → ℕ), (∑ i, x i = n) ∧ (∀ i, 0 < x i)) ↔
  (∃ (y : Fin (n - (∑ i : Fin m, 1)) → ℕ), (∑ i, y i = n - m) ∧ (∀ i, y i ∈ Fin m)) :=
sorry

-- Problem (b) Lean statement
theorem problem_b (n m : ℕ) :
  (∃ (x : Fin m → ℕ), (∑ i, x i = n) ∧ (∀ i j, i < j → x i < x j)) ↔
  (∃ (y : Fin (n - (m * (m + 1) / 2 - (∑ i : Fin m, 1))) → ℕ), (∑ i, y i = n - m * (m + 1) / 2) ∧ (∀ i, y i ∈ Fin n)) :=
sorry

end problem_a_problem_b_l683_683403


namespace num_three_digit_div_by_13_l683_683849

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683849


namespace non_negative_integer_solutions_l683_683302

theorem non_negative_integer_solutions : {p : ℕ × ℕ // 2 * p.1 + p.2 = 5} = {(0, 5), (1, 3), (2, 1)} :=
sorry

end non_negative_integer_solutions_l683_683302


namespace count_3_digit_numbers_divisible_by_13_l683_683672

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683672


namespace tangent_line_ellipse_l683_683533

variable (a b x0 y0 : ℝ)
variable (x y : ℝ)

def ellipse (x y a b : ℝ) := (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1

theorem tangent_line_ellipse :
  ellipse x y a b ∧ a > b ∧ (x0 ≠ 0 ∨ y0 ≠ 0) ∧ (x0 ^ 2) / (a ^ 2) + (y0 ^ 2) / (b ^ 2) > 1 →
  (x0 * x) / (a ^ 2) + (y0 * y) / (b ^ 2) = 1 :=
  sorry

end tangent_line_ellipse_l683_683533


namespace num_pos_3_digit_div_by_13_l683_683771

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683771


namespace joke_spread_after_one_minute_l683_683200

theorem joke_spread_after_one_minute :
  let n := 1
  let friends := 6
  let intervals := 6
  let total_people := ∑ i in Finset.range (intervals + 1), friends^i
  total_people = 1 + 6 + 6^2 + 6^3 + 6^4 + 6^5 + 6^6 :=
by
  sorry
-- This statement expresses that the calculations considering the series result in the sum 55987.
-- Note: Finset.range (intervals + 1) gives the range 0 to 6.

end joke_spread_after_one_minute_l683_683200


namespace question_1_question_2_l683_683314

-- Define the transformation matrices
def M1 : matrix (fin 2) (fin 2) ℝ := ![![0, -1], [1, 0]]
def M2 : matrix (fin 2) (fin 2) ℝ := ![![1, 1], [0, 1]]

-- Define the point P(2, 1)
def P : vector ℝ 2 := ![2, 1]

-- Resulting point P'
def P' : vector ℝ 2 := ![-1, 2]

-- Definition of the function to be transformed
def f (x : ℝ) : ℝ := x ^ 2

-- Combined transformation matrix
def M : matrix (fin 2) (fin 2) ℝ := M2.mul M1

-- Define the transformed equation
axiom transformed_equation : ∀ x y : ℝ, (x, y) ∈ graph (M * f) ↔ y - x = y ^ 2

-- Proof statements
theorem question_1 : (M1.mul_vec P) = P' := by sorry

theorem question_2 : ∀ x y : ℝ, (x, y) ∈ transformed_graph ↔ y - x = y^2 := by sorry

end question_1_question_2_l683_683314


namespace find_P_coordinates_l683_683548

noncomputable def point_on_circle_and_right_triangle (P : ℝ × ℝ) : Prop :=
  (P.1^2 + P.2^2 = 8) ∧ 
  (∃ (A B : ℝ × ℝ), 
    A = (4, 0) ∧ 
    B = (0, 4) ∧ 
    (P.1 = -2 ∧ P.2 = 2 ∨
     P.1 = 1 - real.sqrt 3 ∧ P.2 = 1 + real.sqrt 3 ∨
     P.1 = 1 + real.sqrt 3 ∧ P.2 = 1 - real.sqrt 3)
  ∧ 
  (∃ angle : string, 
   angle = "PAB" ∨ angle = "ABP" ∨ angle = "APB"))

theorem find_P_coordinates (P: ℝ × ℝ) (A B : ℝ × ℝ) (angle : string) 
  (hP : point_on_circle_and_right_triangle P) (hA : A = (4, 0)) (hB : B = (0, 4)) :
  angle = "PAB" ∨ angle = "ABP" ∨ angle = "APB" :=
sorry

end find_P_coordinates_l683_683548


namespace lost_card_number_l683_683393

variable (n : ℕ)
variable (s : ℕ)

-- Axioms based on the given conditions.
axiom sum_of_first_n : s = n * (n + 1) / 2
axiom remaining_sum : s - 101 ∈ finset.singleton 101

-- The theorem we need to prove:
theorem lost_card_number (n : ℕ) (s : ℕ) (h₁ : s = n * (n + 1) / 2) (h₂ : s - 101 = 101) : n = 14 → (n * (n + 1) / 2 - 101 = 4) :=
by
  sorry

end lost_card_number_l683_683393


namespace problem_part1_problem_part2_problem_part3_l683_683562

noncomputable theory

open Real

def f (x : ℝ) (a : ℝ) := sin x + a * cos x

def zero_condition (a : ℝ) := f (π / 4) a = 0

def monotonic_intervals (x : ℝ) (k : ℤ) :=
  2 * k * π - π / 4 ≤ x ∧ x ≤ 2 * k * π + 3 * π / 4

def alpha_condition (a α : ℝ) := f α a = sqrt(10) / 5

def beta_condition (a β : ℝ) := f β a = 3 * sqrt(5) / 5

theorem problem_part1 (a : ℝ) : zero_condition a → a = -1 :=
sorry

theorem problem_part2 (a : ℝ) (x : ℝ) (k : ℤ) : 
  (∀ x, f x a = sin (x - π / 4)) → monotonic_intervals x k →
  monotone (f x a) :=
sorry

theorem problem_part3 (a α β : ℝ) :
   alpha_condition a α → beta_condition a β →
   sin (α + β) = - sqrt(2) / 10 :=
sorry

end problem_part1_problem_part2_problem_part3_l683_683562


namespace distance_from_point_to_line_l683_683155

theorem distance_from_point_to_line 
  (a : ℝ) 
  (A : ℝ × ℝ) 
  (hA : A = (2, 5))
  (l₁ : ℝ → ℝ) 
  (hl₁ : ∀ x: ℝ, l₁ x = a * x - 2 * a + 5)
  : (dist_to_line A 1 (-2) 3) = sqrt 5 :=
by
  sorry

def dist_to_line (A : ℝ × ℝ) (A1 A2 A3 : ℝ) : ℝ :=
  abs (A1 * A.1 + A2 * A.2 + A3) / sqrt (A1^2 + A2^2)

end distance_from_point_to_line_l683_683155


namespace cyclist_motorcyclist_intersection_l683_683412

theorem cyclist_motorcyclist_intersection : 
  ∃ t : ℝ, (4 * t^2 + (t - 1)^2 - 2 * |t| * |t - 1| = 49) ∧ (t = 4 ∨ t = -4) := 
by 
  sorry

end cyclist_motorcyclist_intersection_l683_683412


namespace jill_initial_investment_l683_683224

noncomputable def initial_investment (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  A / (1 + r / n)^(n * t)

theorem jill_initial_investment :
  initial_investment 10815.83 0.0396 2 2 ≈ 10000 :=
by
  sorry

end jill_initial_investment_l683_683224


namespace find_k_l683_683994

def triangle_sides (a b c : ℕ) : Prop :=
a < b + c ∧ b < a + c ∧ c < a + b

def is_right_triangle (a b c : ℕ) : Prop :=
a * a + b * b = c * c

def angle_bisector_length (a b c l : ℕ) : Prop :=
∃ k : ℚ, l = k * Real.sqrt 2 ∧ k = 5 / 2

theorem find_k :
  ∀ (AB BC AC BD : ℕ),
  triangle_sides AB BC AC ∧ is_right_triangle AB BC AC ∧
  AB = 5 ∧ BC = 12 ∧ AC = 13 ∧ angle_bisector_length 5 12 13 BD →
  ∃ k : ℚ, BD = k * Real.sqrt 2 ∧ k = 5 / 2 := by
  sorry

end find_k_l683_683994


namespace P_k_rotation_l683_683126

theorem P_k_rotation
  (P : ℕ → complex)
  (A : ℕ → complex)
  (h₁ : ∀ k, P (k + 1) = (1 + complex.abs (exp (complex.I * real.pi / 3))) * A (k + 1) - exp (complex.I * real.pi / 3) * P k)
  (h₂ : P 1986 = P 0)
  : (A 3 - A 1 = exp (complex.I * real.pi / 3) * (A 2 - A 1)) :=
sorry

end P_k_rotation_l683_683126


namespace passengers_final_count_l683_683324

structure BusStop :=
  (initial_passengers : ℕ)
  (first_stop_increase : ℕ)
  (other_stops_decrease : ℕ)
  (other_stops_increase : ℕ)

def passengers_at_last_stop (b : BusStop) : ℕ :=
  b.initial_passengers + b.first_stop_increase - b.other_stops_decrease + b.other_stops_increase

theorem passengers_final_count :
  passengers_at_last_stop ⟨50, 16, 22, 5⟩ = 49 := by
  rfl

end passengers_final_count_l683_683324


namespace largest_x_not_defined_l683_683345

theorem largest_x_not_defined : 
  (∀ x, (6 * x ^ 2 - 17 * x + 5 = 0) → x ≤ 2.5) ∧
  (∃ x, (6 * x ^ 2 - 17 * x + 5 = 0) ∧ x = 2.5) :=
by
  sorry

end largest_x_not_defined_l683_683345


namespace smallest_number_in_set_l683_683028

noncomputable def my_numbers : set ℝ := {0, 5, -0.3, -1/3}

theorem smallest_number_in_set : Inf my_numbers = -1/3 :=
by
  sorry

end smallest_number_in_set_l683_683028


namespace count_three_digit_div_by_13_l683_683691

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683691


namespace find_g_l683_683251

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g (g : ℝ → ℝ)
  (H : ∀ x y : ℝ, (g x * g y - g (x * y)) / 5 = x + y + 4) :
  g = fun x => x + 5 :=
by
  sorry

end find_g_l683_683251


namespace minor_premise_correct_l683_683054

-- Definitions corresponding to conditions
def is_square (x : Type) : Prop := sorry
def is_parallelogram (x : Type) : Prop := sorry
def is_trapezoid (x : Type) : Prop := sorry

-- Given conditions
axiom square_is_parallelogram : ∀ (x : Type), is_square x → is_parallelogram x
axiom trapezoid_is_not_parallelogram : ∀ (x : Type), is_trapezoid x → ¬ is_parallelogram x
axiom trapezoid_is_not_square : ∀ (x : Type), is_trapezoid x → ¬ is_square x

-- Proof problem statement
theorem minor_premise_correct (x : Type) : trapezoid_is_not_parallelogram x = (∀ (x : Type), is_trapezoid x → ¬ is_parallelogram x) :=
sorry

end minor_premise_correct_l683_683054


namespace symmetric_circle_l683_683535

def circle_equation_symmetric (C l : (ℝ → ℝ → Prop)) : Prop :=
  ∃ a b r, C (λ x y => (x-1)^2 + (y-2)^2 = r) ∧ l (λ x y => x - y = 0)
  ∧ C' (λ x y => (x-a)^2 + (y-b)^2 = r)
  ∧ a = 2 ∧ b = 1 ∧ r = 5

theorem symmetric_circle :
  circle_equation_symmetric
    (λ x y => (x-1)^2 + (y-2)^2 = 5)
    (λ x y => x - y = 0) :=
begin
  sorry
end

end symmetric_circle_l683_683535


namespace point_outside_circle_l683_683177

noncomputable def a : ℝ := 1 / 2
noncomputable def b : ℝ := -3 / 2

theorem point_outside_circle : 
  (a^2 + b^2 > 2) → a + b * complex.I = (2 + complex.I) / (1 - complex.I) :=
sorry

end point_outside_circle_l683_683177


namespace number_of_3_digit_divisible_by_13_l683_683817

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683817


namespace rectangles_containing_both_circles_l683_683988

theorem rectangles_containing_both_circles :
  let horizontal_lines := ['A', 'B', 'C', 'D', 'E'],
      vertical_lines := ['a', 'b', 'c', 'd', 'e'],
      left_side := 'a',
      right_sides := ['d', 'e'],
      top_sides := ['A', 'B'],
      bottom_sides := ['D', 'E']
  in (fintype.card
     (set_of (λ (l r t b : char),
       l ∈ { 'a'} ∧ r ∈ { 'd', 'e'} ∧ t ∈ { 'A', 'B'} ∧ b ∈ { 'D', 'E'})) = 8) := sorry

end rectangles_containing_both_circles_l683_683988


namespace count_three_digit_numbers_divisible_by_13_l683_683602

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683602


namespace water_consumption_correct_l683_683495

theorem water_consumption_correct (w n r : ℝ) 
  (hw : w = 21428) 
  (hn : n = 26848.55) 
  (hr : r = 302790.13) :
  w = 21428 ∧ n = 26848.55 ∧ r = 302790.13 :=
by 
  sorry

end water_consumption_correct_l683_683495


namespace billiard_minimum_balls_l683_683422

theorem billiard_minimum_balls :
  ∃ (balls : ℕ), balls = 4 ∧ 
    ∀ (pockets : set (ℝ × ℝ)),
      (pockets = { (0, 0), (2, 0), (0, 1), (2, 1), (0.5, 0), (1.5, 0) } →
        ∀ (pocket ∈ pockets), ∃ (ball_positions : set (ℝ × ℝ)), 
          (balls = card ball_positions ∧ 
            ∀ (p1 p2 ∈ ball_positions), has_collinear_line p1 p2 pocket)) :=
begin
  sorry
end

end billiard_minimum_balls_l683_683422


namespace balance_difference_l683_683222

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

theorem balance_difference :
  let P_J := 12000
  let r_J := 0.025
  let n_J := 50
  let P_M := 15000
  let r_M := 0.06
  let t_M := 25
  let A_J := compound_interest P_J r_J n_J
  let A_M := simple_interest P_M r_M t_M
  abs (A_J - A_M) ≈ 3136 :=
by
  sorry

end balance_difference_l683_683222


namespace count_3_digit_numbers_divisible_by_13_l683_683913

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683913


namespace margaret_age_in_12_years_l683_683041

theorem margaret_age_in_12_years
  (brian_age : ℝ)
  (christian_age : ℝ)
  (margaret_age : ℝ)
  (h1 : christian_age = 3.5 * brian_age)
  (h2 : brian_age + 12 = 45)
  (h3 : margaret_age = christian_age - 10) :
  margaret_age + 12 = 117.5 :=
by
  sorry

end margaret_age_in_12_years_l683_683041


namespace customOp_eval_l683_683968

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := a * (b + 1) + a * b

-- State the theorem we need to prove
theorem customOp_eval : customOp 4 (-1) = -4 :=
  by
    sorry

end customOp_eval_l683_683968


namespace three_digit_numbers_divisible_by_13_l683_683754

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683754


namespace count_3_digit_numbers_divisible_by_13_l683_683901

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683901


namespace extremum_a_monotonicity_range_m_l683_683147

def f (x a : ℝ) : ℝ := log x + x^2 - a * x
def f' (x a : ℝ) : ℝ := 1 / x + 2 * x - a

theorem extremum_a (h1 : f' 1 a = 0) : a = 3 := sorry

theorem monotonicity (h2 : 0 < a ∧ a ≤ 2) : ∀ x > 0, f' x a > 0 := sorry

theorem range_m (h3 : ∀ (a ∈ (1:ℝ, 2)), ∀ x0 ∈ [1,2], f x0 a > m * log a) : m ≤ -log 2 := sorry

end extremum_a_monotonicity_range_m_l683_683147


namespace num_three_digit_div_by_13_l683_683848

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683848


namespace number_of_solutions_l683_683502

open Complex

noncomputable def num_solutions : ℕ :=
  let S := { z : ℂ | abs z = 1 ∧ abs ((z / conj z) - (conj z / z)) = 2 }
  fintype.card (set.finite S).to_finset

theorem number_of_solutions :
  num_solutions = 4 :=
sorry

end number_of_solutions_l683_683502


namespace count_three_digit_numbers_divisible_by_13_l683_683647

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683647


namespace area_of_shaded_region_l683_683018

-- Define the side length of the regular hexagon
def side_length : ℝ := 1.5

-- Define the radius of the semicircle
def radius : ℝ := side_length / 2

-- Define the area of the regular hexagon
def hexagon_area : ℝ := (3 * Real.sqrt 3 / 2) * side_length^2

-- Define the area of one semicircle
def semicircle_area : ℝ := (1 / 2) * Real.pi * radius^2

-- Define the total area of six semicircles
def total_semicircles_area : ℝ := 6 * semicircle_area

-- Define the area of the shaded region
def shaded_region_area : ℝ := hexagon_area - total_semicircles_area

-- The Lean statement to be proven
theorem area_of_shaded_region : shaded_region_area = 3.375 * Real.sqrt 3 - 1.6875 * Real.pi :=
by
  sorry

end area_of_shaded_region_l683_683018


namespace three_digit_numbers_divisible_by_13_l683_683751

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683751


namespace count_3_digit_multiples_of_13_l683_683783

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683783


namespace average_speed_first_part_l683_683001

noncomputable def speed_of_first_part (v : ℝ) : Prop :=
  let distance_first_part := 124
  let speed_second_part := 60
  let distance_second_part := 250 - distance_first_part
  let total_time := 5.2
  (distance_first_part / v) + (distance_second_part / speed_second_part) = total_time

theorem average_speed_first_part : speed_of_first_part 40 :=
  sorry

end average_speed_first_part_l683_683001


namespace smallest_even_integer_l683_683311

theorem smallest_even_integer (n : ℕ) (h_even : n % 2 = 0)
  (h_2digit : 10 ≤ n ∧ n ≤ 98)
  (h_property : (n - 2) * n * (n + 2) = 5 * ((n - 2) + n + (n + 2))) :
  n = 86 :=
by
  sorry

end smallest_even_integer_l683_683311


namespace question_l683_683252

variables {m n β : Type} [LinearSpace m] [LinearSpace n] [Plane β]

def parallel (x y : Type) [LinearSpace x] [LinearSpace y] := ∀ p : Point, p ∈ x → p ∈ y
def perpendicular (x : Type) [LinearSpace x] (y : Type) [Plane y] := ∀ p : Point, p ∈ x → p ∈ y → ⊥ 

axioms (m_parallel_n : parallel m n)
       (n_perpendicular_beta : perpendicular n β)

theorem question (m_parallel_n : parallel m n) (n_perpendicular_beta : perpendicular n β) : perpendicular m β := 
by sorry

end question_l683_683252


namespace three_digit_numbers_div_by_13_l683_683725

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683725


namespace number_of_three_digit_numbers_divisible_by_13_l683_683945

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683945


namespace circle_equation_l683_683499

theorem circle_equation (x y : ℝ) (h k : ℝ) (r : ℝ) :
    h = 2 → k = -3 → r = Real.sqrt (2^2 + (-3)^2) → (x - h)^2 + (y - k)^2 = r^2 → (x - 2)^2 + (y + 3)^2 = 13 := 
by
  assume h_eq k_eq r_eq circle_eq
  rw [h_eq, k_eq, r_eq] at circle_eq
  simp at r_eq
  sorry

end circle_equation_l683_683499


namespace count_three_digit_numbers_divisible_by_13_l683_683642

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683642


namespace count_3_digit_multiples_of_13_l683_683788

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683788


namespace count_three_digit_div_by_13_l683_683681

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683681


namespace rearrange_square_to_rectangle_perimeter_l683_683444

def square_perimeter := 4
def square_side := square_perimeter / 4
def rearranged_rectangle_perimeter (x : ℝ) :=
  let x := (3 - Real.sqrt 5) / 2 in
  8 - 4 * x

theorem rearrange_square_to_rectangle_perimeter :
  rearranged_rectangle_perimeter square_side = 2 * Real.sqrt 5 :=
by
  sorry

end rearrange_square_to_rectangle_perimeter_l683_683444


namespace constant_term_exists_l683_683531

theorem constant_term_exists (n : ℕ) (hn : 5 ≤ n ∧ n ≤ 16) (hpos : 0 < n) :
  (∃ r : ℕ, (x - 1 / x^3) ^ n = C(n, r) * (x ^ (n - 4 * r))) ↔ (n = 8 ∨ n = 12 ∨ n = 16) :=
sorry

end constant_term_exists_l683_683531


namespace prime_number_between_50_and_60_with_remainder_3_l683_683503

open Nat

theorem prime_number_between_50_and_60_with_remainder_3 : 
  ∃ n : ℕ, 50 ≤ n ∧ n ≤ 60 ∧ Prime n ∧ n % 10 = 3 :=
sorry

end prime_number_between_50_and_60_with_remainder_3_l683_683503


namespace number_of_3_digit_divisible_by_13_l683_683798

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683798


namespace rectangle_area_l683_683496

theorem rectangle_area (P : ℝ) (twice : ℝ → ℝ) (L W A : ℝ) 
  (h1 : P = 40) 
  (h2 : ∀ W, L = twice W) 
  (h3 : ∀ L W, P = 2 * L + 2 * W) 
  (h4 : ∀ L W, A = L * W) 
  (h5 : twice = (λ W, 2 * W)) :
  A = 800 / 9 := 
sorry

end rectangle_area_l683_683496


namespace sarah_hardback_books_l683_683035

theorem sarah_hardback_books :
  ∀ (H: ℕ), 
  (2 + 2 * H = 10) →
  H = 4 :=
begin
  intros H h,
  sorry,
end

end sarah_hardback_books_l683_683035


namespace count_three_digit_numbers_divisible_by_13_l683_683611

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683611


namespace count_3_digit_multiples_of_13_l683_683791

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683791


namespace num_pos_3_digit_div_by_13_l683_683766

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683766


namespace Parabola_vertex_form_l683_683990

theorem Parabola_vertex_form (x : ℝ) (y : ℝ) : 
  (∃ h k : ℝ, (h = -2) ∧ (k = 1) ∧ (y = (x + h)^2 + k) ) ↔ (y = (x + 2)^2 + 1) :=
by
  sorry

end Parabola_vertex_form_l683_683990


namespace area_of_quadrilateral_abcd_l683_683194

-- Define the quadrilateral ABCD with diagonals intersecting at E
variables (A B C D E : Type) [AffineSpace A E] [AffineSpace B E] [AffineSpace C E] [AffineSpace D E]
variables {α β : ℝ}

-- Define the condition for areas of triangles
def area_triangle_abe : ℝ := 40
def area_triangle_ade : ℝ := 25

-- Define a function representing the total area of the quadrilateral ABCD
noncomputable def total_area_of_quadrilateral (aabe aade abce acde : ℝ) : ℝ :=
  aabe + aade + abce + acde

-- Define the main theorem to prove that area of quadrilateral ABCD is 130 square units
theorem area_of_quadrilateral_abcd :
  total_area_of_quadrilateral 40 25 25 40 = 130 :=
by
  -- Assume the given areas
  let aabe := area_triangle_abe
  let aade := area_triangle_ade

  -- Assume the derived areas from the solution steps
  let abce := aade
  let acde := aabe

  -- Calculate total area
  let total_area := total_area_of_quadrilateral aabe aade abce acde

  -- Show that the total area is indeed 130
  have h_total_area : total_area = 130 := by
    sorry -- proof here

  exact h_total_area

end area_of_quadrilateral_abcd_l683_683194


namespace number_of_3_digit_divisible_by_13_l683_683802

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683802


namespace area_of_quadrilateral_l683_683202

-- Defining curves C1 and C2 as parametric equations
def C1 (φ : Real) : Real × Real := (Real.cos φ, Real.sin φ)
def C2 (φ : Real) (a b : Real) : Real × Real := (a * Real.cos φ, b * Real.sin φ)

-- Definitions of a and b
def a := 3
def b := 1

-- Define intersection calculations at various angles
def intersection_alpha_0 (α : Real) := C1 α = (1, 0) ∧ C2 α a b = (a, 0)
def intersection_alpha_pi2 (α : Real) := C1 α = (0, 1) ∧ C2 α a b = (0, b)
def intersection_alpha (α : Real) := C1 α = (Real.cos α, Real.sin α) ∧ C2 α a b = (a * Real.cos α, b * Real.sin α)

-- Define A1, A2, B1, B2 based on intersections
def A1 := C1 (π / 4)
def B1 := C2 (π / 4) a b
def A2 := C1 (-π / 4)
def B2 := C2 (-π / 4) a b

-- Statement to prove the area of quadrilateral A1A2B2B1 is 4/25
theorem area_of_quadrilateral : 
  let x1 := Real.cos (π / 4) 
  let x2 := a * Real.cos (π / 4)
  let area := (|2 * x2 + 2 * x1| * |x2 - x1|) / 2 
  area = 4 / 25 :=
sorry

end area_of_quadrilateral_l683_683202


namespace log_properties_l683_683419

theorem log_properties :
  (Real.log 5) ^ 2 + (Real.log 2) * (Real.log 50) = 1 :=
by sorry

end log_properties_l683_683419


namespace savings_account_total_l683_683284

-- Definition of the problem's conditions
def initial_deposit := 5000
def additional_deposit := 350
def interest_rate := 1.04
def total_years := 20
def start_year_addition := 3

-- Function to compute the future value of regular additions
def future_value_additions : ℕ → ℕ → ℕ → ℝ → ℝ
| 0, _, _, _ => 0
| (n+1), base_year, addition, rate =>
  let years := total_years - base_year
  let value := addition * rate ^ years
  value + future_value_additions n (base_year + 2) addition rate

-- Function to compute the total amount in the account at the beginning of the 20th year
def total_amount := initial_deposit * interest_rate ^ (total_years - 1) +
                    future_value_additions (total_years // 2 - 1) start_year_addition additional_deposit interest_rate

-- The statement to prove
theorem savings_account_total : total_amount = 15107 := by
  sorry

end savings_account_total_l683_683284


namespace determine_functions_l683_683056

def f (n : ℕ) : ℕ := n
def g (n : ℕ) : ℕ := 1

theorem determine_functions :
  ∀ n : ℕ, (f (f (n) + 1) (n) + (g (g (n)) (n)) = (f (n + 1) - g (n + 1) + 1)) :=
begin
  assume n,
  sorry -- proof to be completed
end

-- Ensure that definitions match the conditions in the problem
example : (f (n) = n) ∧ (g (n) = 1) :=
begin
  split,
  { assume n, refl },
  { assume n, refl }
end

end determine_functions_l683_683056


namespace three_digit_numbers_divisible_by_13_count_l683_683631

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683631


namespace functional_equation_identity_l683_683070

theorem functional_equation_identity (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f (x + 1) + y - 1) = f x + y) →
  (∀ x : ℝ, f x = x) :=
by
  assume h : ∀ x y : ℝ, f (f (x + 1) + y - 1) = f x + y
  sorry

end functional_equation_identity_l683_683070


namespace count_three_digit_numbers_divisible_by_13_l683_683919

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683919


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683585

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683585


namespace chord_length_l683_683426

theorem chord_length (O P Q R : Point) :
  (dist O P = 5) ∧ (dist O R = 4) ∧ (midpoint R P Q) → dist P Q = 6 := by
 sorry

end chord_length_l683_683426


namespace count_three_digit_numbers_divisible_by_13_l683_683925

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683925


namespace count_3_digit_numbers_divisible_by_13_l683_683877

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683877


namespace count_three_digit_numbers_divisible_by_13_l683_683935

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683935


namespace tangent_line_at_0_1_is_correct_l683_683295

theorem tangent_line_at_0_1_is_correct :
  let f := λ x : ℝ, x * Real.exp x + 2 * x + 1
  let f' := λ x : ℝ, (1 + x) * Real.exp x + 2
  ∀ x : ℝ, f 0 = 1 → f' 0 = 3 → (∀ x : ℝ, f' 0 * x + 1 = 3 * x + 1) :=
by
  intro f f'
  assume h₁ h₂
  intro x
  rw [← h₂, mul_comm]
  rfl

end tangent_line_at_0_1_is_correct_l683_683295


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683581

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683581


namespace count_three_digit_numbers_divisible_by_13_l683_683646

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683646


namespace max_sum_of_integer_pairs_on_circle_l683_683326

theorem max_sum_of_integer_pairs_on_circle : 
  ∃ (x y : ℤ), x^2 + y^2 = 169 ∧ ∀ (a b : ℤ), a^2 + b^2 = 169 → x + y ≥ a + b :=
sorry

end max_sum_of_integer_pairs_on_circle_l683_683326


namespace three_digit_numbers_divisible_by_13_count_l683_683619

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683619


namespace three_digit_numbers_divisible_by_13_l683_683745

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683745


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683578

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683578


namespace count_three_digit_numbers_divisible_by_13_l683_683645

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683645


namespace three_digit_numbers_div_by_13_l683_683735

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683735


namespace cannot_form_basis_l683_683545

variables (e1 e2 : Type*) [AddCommGroup e1] [AddCommGroup e2] [Module ℝ e1] [Module ℝ e2]

-- Define the given vectors as Lean variables
def v1 := 2 • e1 - e2
def v2 := (1 / 2) • e2 - e1

-- The goal is to show that v1 and v2 are collinear, which implies they cannot form a basis
theorem cannot_form_basis : ∃ (k : ℝ), v1 = k • v2 := by
  -- Proof is omitted for this statement
  sorry

end cannot_form_basis_l683_683545


namespace number_of_3_digit_divisible_by_13_l683_683810

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683810


namespace sum_of_squares_of_coefficients_l683_683355

theorem sum_of_squares_of_coefficients :
  let expr := (3 * (x^3 - 2 * x^2 + 3)) - (5 * (x^4 - 4 * x^2 + 2))
  let terms := -5 * x^4 + 3 * x^3 + 14 * x^2 - 1
  (let coef := [-5, 3, 14, -1] in
    let sum_squares := coef.map (λ a => a ^ 2) |>.sum
    sum_squares = 231) :=
by
  sorry

end sum_of_squares_of_coefficients_l683_683355


namespace range_of_m_l683_683560

def f (x : ℝ) : ℝ :=
if x ≤ 1 then 1 - |x| else x^2 - 4*x + 3

theorem range_of_m (m : ℝ) : 
  (f (f m)) ≥ 0 ↔ (m ∈ (Set.Icc (-2) (2 + Real.sqrt 2)) ∪ Set.Ioi 4) :=
sorry

end range_of_m_l683_683560


namespace dawn_income_per_hour_l683_683999

theorem dawn_income_per_hour (painting_time hours_per_painting commission paintings total_earnings : ℕ) :
  (hours_per_painting = 2) →
  (paintings = 12) →
  (total_earnings = 3600) →
  (painting_time = hours_per_painting * paintings) →
  total_earnings / painting_time = 150 :=
by {
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  exact (by norm_num : 3600 / 24 = 150)
}

end dawn_income_per_hour_l683_683999


namespace find_AM_l683_683974

theorem find_AM 
  (R a AM : ℝ)
  (H_AB : 0 ≤ a)
  (H_R : 0 < R)
  (H_ratio : ∃ y : ℝ, y > 0 ∧ PM = 3 * y ∧ MQ = y)
  (H_power_of_point : ∃ x : ℝ, x = AM ∧ x * (a - x) = 3 * ((x * sqrt (4 * R^2 - a^2)) / (2 * R))^2) :
  AM = (4 * R^2 * a) / (16 * R^2 - 3 * a^2) := 
by 
  sorry

end find_AM_l683_683974


namespace count_3digit_numbers_div_by_13_l683_683892

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683892


namespace lost_card_l683_683370

noncomputable def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

theorem lost_card (n : ℕ) (h1 : sum_natural n - 101 < n) (h2 : sum_natural (n - 1) ≤ 101) :
  ∃ x : ℕ, x = sum_natural n - 101 ∧ x = 4 :=
begin
  have h_sum : sum_natural 14 = 105,
  {
    unfold sum_natural,
    norm_num,
  },
  use 4,
  split,
  {
    unfold sum_natural at *,
    norm_num,
    rw h_sum,
  },
  {
    norm_num,
  },
  sorry,
end

end lost_card_l683_683370


namespace sum_series_value_l683_683102

noncomputable def closest_int_sqrt (n : ℕ) : ℤ := 
  int.of_nat (nat.floor (real.sqrt n))

theorem sum_series_value : 
  (∑' n : ℕ, (3 ^ closest_int_sqrt n + 3 ^ -closest_int_sqrt n) / 3 ^ n) = 4.03703 := 
by 
  simp only 
  sorry

end sum_series_value_l683_683102


namespace eq_expression_l683_683358

theorem eq_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 :=
by
  sorry

end eq_expression_l683_683358


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683590

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683590


namespace Kyle_papers_delivered_each_week_l683_683235

-- Definitions representing the conditions
def papers_per_day := 100
def days_Mon_to_Sat := 6
def regular_Sunday_customers := 100
def non_regular_Sunday_customers := 30
def no_delivery_customers_on_Sunday := 10

-- The total number of papers delivered each week
def total_papers_per_week : ℕ :=
  days_Mon_to_Sat * papers_per_day +
  regular_Sunday_customers - no_delivery_customers_on_Sunday + non_regular_Sunday_customers

-- Prove that Kyle delivers 720 papers each week
theorem Kyle_papers_delivered_each_week : total_papers_per_week = 720 :=
sorry

end Kyle_papers_delivered_each_week_l683_683235


namespace number_of_3_digit_divisible_by_13_l683_683801

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683801


namespace count_3_digit_numbers_divisible_by_13_l683_683698

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683698


namespace valid_lineups_count_l683_683271

-- Definitions of the problem conditions
def num_players : ℕ := 18
def quadruplets : Finset ℕ := {0, 1, 2, 3} -- Indices of Benjamin, Brenda, Brittany, Bryan
def total_starters : ℕ := 8

-- Function to count lineups based on given constraints
noncomputable def count_valid_lineups : ℕ :=
  let others := num_players - quadruplets.card
  Nat.choose others total_starters + quadruplets.card * Nat.choose others (total_starters - 1)

-- The theorem to prove the count of valid lineups
theorem valid_lineups_count : count_valid_lineups = 16731 := by
  -- Placeholder for the actual proof
  sorry

end valid_lineups_count_l683_683271


namespace common_rational_root_is_negative_non_integer_l683_683050

theorem common_rational_root_is_negative_non_integer 
    (a b c d e f g : ℤ)
    (p : ℚ)
    (h1 : 90 * p^4 + a * p^3 + b * p^2 + c * p + 15 = 0)
    (h2 : 15 * p^5 + d * p^4 + e * p^3 + f * p^2 + g * p + 90 = 0)
    (h3 : ¬ (∃ k : ℤ, p = k))
    (h4 : p < 0) : 
  p = -1 / 3 := 
sorry

end common_rational_root_is_negative_non_integer_l683_683050


namespace lost_card_number_l683_683392

variable (n : ℕ)
variable (s : ℕ)

-- Axioms based on the given conditions.
axiom sum_of_first_n : s = n * (n + 1) / 2
axiom remaining_sum : s - 101 ∈ finset.singleton 101

-- The theorem we need to prove:
theorem lost_card_number (n : ℕ) (s : ℕ) (h₁ : s = n * (n + 1) / 2) (h₂ : s - 101 = 101) : n = 14 → (n * (n + 1) / 2 - 101 = 4) :=
by
  sorry

end lost_card_number_l683_683392


namespace simplified_expression_eq_ab_l683_683281

open Real

noncomputable def simplify_expression (a b : ℝ) : ℝ :=
  a^(2 / (log b a) + 1) * b - 2 * a^(log a b + 1) * b^(log b a + 1) + a * b^(2 / (log a b) + 1)

theorem simplified_expression_eq_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) :
  simplify_expression a b = a * b * (a - b)^2 :=
by
  sorry

end simplified_expression_eq_ab_l683_683281


namespace trig_identity_proof_l683_683522

theorem trig_identity_proof :
  (sin (70 * Real.pi / 180) * sin (20 * Real.pi / 180)) / 
  (cos (155 * Real.pi / 180) ^ 2 - sin (155 * Real.pi / 180) ^ 2) = 1 / 2 := 
sorry

end trig_identity_proof_l683_683522


namespace num_3_digit_div_by_13_l683_683828

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683828


namespace solution_of_system_of_inequalities_l683_683520

theorem solution_of_system_of_inequalities (p : ℝ) :
  19 * p < 10 ∧ p > 0.5 ↔ (1 / 2) < p ∧ p < (10 / 19) :=
by
  sorry

end solution_of_system_of_inequalities_l683_683520


namespace num_three_digit_div_by_13_l683_683839

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683839


namespace solve_system_of_inequalities_l683_683508

theorem solve_system_of_inequalities (p : ℚ) : (19 * p < 10) ∧ (1 / 2 < p) → (1 / 2 < p ∧ p < 10 / 19) :=
by
  sorry

end solve_system_of_inequalities_l683_683508


namespace EF_eq_AO_l683_683260

variables {A B C O M E F : Type} [euclidean_geometry A B C O M E F]
variables (distance : A → A → ℝ) (triangle_circumcenter : Type → Type) (triangle_orthocenter : Type → Type)
variables (on_ray : A → A → A → Prop) (equals_distance : ℝ)

-- Assume O is the circumcenter of triangle ABC
axiom circumcenter_def : triangle_circumcenter A O ↔ (distance A O = distance B O ∧ distance B O = distance C O)

-- Assume M is the orthocenter of triangle ABC
axiom orthocenter_def : triangle_orthocenter A M ↔ (∃ H_a H_b H_c, altitude A B C H_a ∧ altitude B A C H_b ∧ altitude C A B H_c)

-- Assume E is on the ray AC such that AE = AO
axiom E_on_AC : on_ray A C E
axiom AE_eq_AO : distance A E = distance A O

-- Assume F is on the ray AB such that AF = AM
axiom F_on_AB : on_ray A B F
axiom AF_eq_AM : distance A F = distance A M

theorem EF_eq_AO : distance E F = distance A O :=
sorry

end EF_eq_AO_l683_683260


namespace num_three_digit_div_by_13_l683_683844

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683844


namespace solution_l683_683038

noncomputable def problem (x : ℝ) (h : x ≠ 3) : ℝ :=
  (3 * x / (x - 3)) + ((x + 6) / (3 - x))

theorem solution (x : ℝ) (h : x ≠ 3) : problem x h = 2 :=
by
  sorry

end solution_l683_683038


namespace max_interval_length_eq_four_pi_thirds_l683_683293

noncomputable def max_interval_length (a b : Real) : Real :=
  b - a

def is_sin_domain {a b : ℝ} (f : ℝ → ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f x = Real.sin x

def is_sin_range {a b : ℝ} (f : ℝ → ℝ) : Prop :=
  ∀ y, y = f x → -1 ≤ y ∧ y ≤ 1 / 2

theorem max_interval_length_eq_four_pi_thirds :
  ∃ a b : ℝ, is_sin_domain (Real.sin) ∧ is_sin_range (Real.sin) ∧ max_interval_length a b = 4 * Real.pi / 3 :=
sorry

end max_interval_length_eq_four_pi_thirds_l683_683293


namespace line_parallel_to_hyperbola_asymptote_l683_683966

theorem line_parallel_to_hyperbola_asymptote {k : ℝ} (h : k > 0) :
  (∃ {x y : ℝ}, y = k * x - 1) ∧ ∃ {x y : ℝ}, (x^2 / 16) - (y^2 / 9) = 1 → k = 3 / 4 :=
by
  sorry

end line_parallel_to_hyperbola_asymptote_l683_683966


namespace zero_in_interval_2_3_l683_683289

-- Define the function f
def f (x : ℝ) : ℝ := Real.log x - 2 / x

-- The main statement to prove
theorem zero_in_interval_2_3 : 
  (0 < 2) ∧ (0 < 3) ∧ ∀ x, 0 < x → continuous_at f x ∧ 
  monotone f ∧ f 2 < 0 ∧ f 3 > 0 → ∃ c : ℝ, c ∈ set.Ioo 2 3 ∧ f c = 0 :=
by
  sorry

end zero_in_interval_2_3_l683_683289


namespace log_eqn_solve_l683_683059

theorem log_eqn_solve :
  ∃ n : ℕ, n > 0 ∧ ((log 3 (log 27 n)) = (log 9 (log 9 n))) ∧ n = 27 :=
begin
  sorry
end

end log_eqn_solve_l683_683059


namespace num_pos_3_digit_div_by_13_l683_683777

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683777


namespace count_three_digit_numbers_divisible_by_13_l683_683930

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683930


namespace range_of_2x_plus_y_l683_683552

-- Given that positive numbers x and y satisfy this equation:
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x + y + 4 * x * y = 15 / 2

-- Define the range for 2x + y
def range_2x_plus_y (x y : ℝ) : ℝ :=
  2 * x + y

-- State the theorem.
theorem range_of_2x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : satisfies_equation x y) :
  3 ≤ range_2x_plus_y x y :=
by
  sorry

end range_of_2x_plus_y_l683_683552


namespace count_three_digit_numbers_divisible_by_13_l683_683918

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683918


namespace ariane_winning_strategy_l683_683253

theorem ariane_winning_strategy (n : ℕ) (hn : n ≥ 2) :
  (n = 2) ∨ (n = 4) ∨ (n = 8) ↔
    (∃ winning_strategy_for_ariane : strategy, winning_strategy_for_ariane n) ∧
    ∀ m, m ≠ 2 ∧ m ≠ 4 ∧ m ≠ 8 → (∃ winning_strategy_for_berénice : strategy, winning_strategy_for_berénice m)
:= sorry

end ariane_winning_strategy_l683_683253


namespace count_three_digit_numbers_divisible_by_13_l683_683933

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683933


namespace count_3_digit_numbers_divisible_by_13_l683_683912

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683912


namespace count_3_digit_numbers_divisible_by_13_l683_683710

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683710


namespace number_of_three_digit_numbers_divisible_by_13_l683_683946

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683946


namespace orange_is_faster_by_l683_683032

def forest_run_time (distance speed : ℕ) : ℕ := distance / speed
def beach_run_time (distance speed : ℕ) : ℕ := distance / speed
def mountain_run_time (distance speed : ℕ) : ℕ := distance / speed

def total_time_in_minutes (forest_distance forest_speed beach_distance beach_speed mountain_distance mountain_speed : ℕ) : ℕ :=
  (forest_run_time forest_distance forest_speed + beach_run_time beach_distance beach_speed + mountain_run_time mountain_distance mountain_speed) * 60

def apple_total_time := total_time_in_minutes 18 3 6 2 3 1
def mac_total_time := total_time_in_minutes 20 4 8 3 3 1
def orange_total_time := total_time_in_minutes 22 5 10 4 3 2

def combined_time := apple_total_time + mac_total_time
def orange_time_difference := combined_time - orange_total_time

theorem orange_is_faster_by :
  orange_time_difference = 856 := sorry

end orange_is_faster_by_l683_683032


namespace count_three_digit_numbers_divisible_by_13_l683_683641

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683641


namespace lost_card_number_l683_683383

theorem lost_card_number (n x : ℕ) (sum_n : ℕ) (h_sum : sum_n = n * (n + 1) / 2) (h_remaining_sum : sum_n - x = 101) : x = 4 :=
sorry

end lost_card_number_l683_683383


namespace number_of_3_digit_divisible_by_13_l683_683807

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683807


namespace elderly_people_arrangement_l683_683019

theorem elderly_people_arrangement :
  let V := 4 -- number of volunteers
  let E := 2 -- number of elderly people
  let TotalEntities := V + E - 1 -- treating the 2 elderly people as a single entity
  let TotalArrangements := factorial TotalEntities -- 5!
  let InvalidEndArrangements := 2 * factorial V -- 2 * 4!
  let ValidArrangements := TotalArrangements - InvalidEndArrangements -- 120 - 48
  let InternalArrangements := factorial E -- 2!
  let TotalValidArrangements := ValidArrangements * InternalArrangements -- 72 * 2
  True → TotalValidArrangements = 144 :=
by
  intros
  let V := 4
  let E := 2
  let TotalEntities := V + E - 1
  let TotalArrangements := factorial TotalEntities
  let InvalidEndArrangements := 2 * factorial V
  let ValidArrangements := TotalArrangements - InvalidEndArrangements
  let InternalArrangements := factorial E
  let TotalValidArrangements := ValidArrangements * InternalArrangements
  have fact_5 : factorial 5 = 120 := by library_search
  have fact_4 : factorial 4 = 24 := by library_search
  have fact_2 : factorial 2 = 2 := by library_search
  calc
    TotalValidArrangements = (TotalArrangements - InvalidEndArrangements) * InternalArrangements : rfl
    ... = (120 - 48) * 2 : by rw [fact_5, fact_4, fact_2]
    ... = 72 * 2 : rfl
    ... = 144 : rfl
  sorry

end elderly_people_arrangement_l683_683019


namespace three_digit_numbers_div_by_13_l683_683736

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683736


namespace count_three_digit_div_by_13_l683_683685

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683685


namespace lost_card_number_l683_683373

theorem lost_card_number (n : ℕ) (x : ℕ) (h₁ : (n * (n + 1)) / 2 - x = 101) : x = 4 :=
sorry

end lost_card_number_l683_683373


namespace multiply_103_97_l683_683044

theorem multiply_103_97 : (103 * 97 = 9991) := 
by 
  have h1: 103 = 100 + 3 := rfl
  have h2: 97 = 100 - 3 := rfl
  have identity: (100 + 3) * (100 - 3) = (100^2 - 3^2) := by ring
  rw [←h1, ←h2, identity]
  norm_num
  sorry

end multiply_103_97_l683_683044


namespace no_real_roots_iff_k_lt_neg_one_l683_683186

theorem no_real_roots_iff_k_lt_neg_one (k : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - k ≠ 0) ↔ k < -1 :=
by sorry

end no_real_roots_iff_k_lt_neg_one_l683_683186


namespace total_sequins_is_162_l683_683218

/-- Jane sews 6 rows of 8 blue sequins each. -/
def rows_of_blue_sequins : Nat := 6
def sequins_per_blue_row : Nat := 8
def total_blue_sequins : Nat := rows_of_blue_sequins * sequins_per_blue_row

/-- Jane sews 5 rows of 12 purple sequins each. -/
def rows_of_purple_sequins : Nat := 5
def sequins_per_purple_row : Nat := 12
def total_purple_sequins : Nat := rows_of_purple_sequins * sequins_per_purple_row

/-- Jane sews 9 rows of 6 green sequins each. -/
def rows_of_green_sequins : Nat := 9
def sequins_per_green_row : Nat := 6
def total_green_sequins : Nat := rows_of_green_sequins * sequins_per_green_row

/-- The total number of sequins Jane adds to her costume. -/
def total_sequins : Nat := total_blue_sequins + total_purple_sequins + total_green_sequins

theorem total_sequins_is_162 : total_sequins = 162 := 
by
  sorry

end total_sequins_is_162_l683_683218


namespace infinitely_many_primes_not_in_S_l683_683244

def polynomial_with_int_coeff (n : ℕ) : Type := fin n → polynomial ℤ

noncomputable def set_S (P : ∀ (i : fin n), polynomial ℤ) : set ℕ :=
  {N | ∃ a : ℕ, ∃ i : fin n, polynomial.eval (a : ℤ) (P i) = N}

theorem infinitely_many_primes_not_in_S (n : ℕ) (P : polynomial_with_int_coeff n) :
  ∃ (primes : set ℕ), (∀ p ∈ primes, nat.prime p) ∧ (primes.infinite ∧ primes ⊆ - (set_S P)) :=
by
  sorry

end infinitely_many_primes_not_in_S_l683_683244


namespace num_pos_3_digit_div_by_13_l683_683776

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683776


namespace min_value_of_mn_l683_683122

theorem min_value_of_mn (m n : ℝ) (h : ∀ x : ℝ, x > -m → x + m ≤ exp (2 * x / m + n)) : m * n = -2 / exp 2 :=
sorry

end min_value_of_mn_l683_683122


namespace number_of_three_digit_numbers_divisible_by_13_l683_683955

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683955


namespace perimeter_quadrilateral_220_l683_683006

variable {EFGH : Type} [quadrilateral E F G H]
variable (Q : EFGH → Point)
variable (area_EFGH : ℝ)
variable (distance_QE : ℝ)
variable (distance_QF : ℝ)
variable (distance_QG : ℝ)
variable (distance_QH : ℝ)

noncomputable def perimeter_of_quadrilateral (EFGH : Type) : ℝ := 
  sorry

theorem perimeter_quadrilateral_220 : 
    convex_quadrilateral E F G H →
    area_EFGH = 3000 →
    Q = some_point_inside_quadrilateral →
    distance_QE = 30 →
    distance_QF = 40 →
    distance_QG = 35 →
    distance_QH = 50 →
    perimeter_of_quadrilateral E F G H Q distance_QE distance_QF distance_QG distance_QH = 220 := 
  sorry

end perimeter_quadrilateral_220_l683_683006


namespace three_digit_numbers_divisible_by_13_count_l683_683621

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683621


namespace count_three_digit_numbers_divisible_by_13_l683_683604

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683604


namespace proposition_3_true_proposition_4_true_l683_683172

def exp_pos (x : ℝ) : Prop := Real.exp x > 0

def two_power_gt_xsq (x : ℝ) : Prop := 2^x > x^2

def prod_gt_one (a b : ℝ) (ha : a > 1) (hb : b > 1) : Prop := a * b > 1

def geom_seq_nec_suff (a b c : ℝ) : Prop := ¬(b = Real.sqrt (a * c) ∨ (a * b = c * b ∧ b^2 = a * c))

theorem proposition_3_true (a b : ℝ) (ha : a > 1) (hb : b > 1) : prod_gt_one a b ha hb :=
sorry

theorem proposition_4_true (a b c : ℝ) : geom_seq_nec_suff a b c :=
sorry

end proposition_3_true_proposition_4_true_l683_683172


namespace robotics_club_students_l683_683268

theorem robotics_club_students (total cs e both neither : ℕ) 
  (h1 : total = 80)
  (h2 : cs = 52)
  (h3 : e = 38)
  (h4 : both = 25)
  (h5 : neither = total - (cs - both + e - both + both)) :
  neither = 15 :=
by
  sorry

end robotics_club_students_l683_683268


namespace largest_of_three_consecutive_integers_sum_18_l683_683320

theorem largest_of_three_consecutive_integers_sum_18 (n : ℤ) (h : n + (n + 1) + (n + 2) = 18) : n + 2 = 7 :=
by
  sorry

end largest_of_three_consecutive_integers_sum_18_l683_683320


namespace number_of_incorrect_props_is_four_l683_683556

-- Proposition 1: vectors equality implies coinciding points and equal line segment
def prop1 (A B C D : Type*) (AB CD : A → B → Type*) (vAB vCD : Type*) : Prop :=
  vAB = vCD → (A = C ∧ B = D ∧ AB = CD)

-- Proposition 2: dot product less than zero implies obtuse angle
def prop2 (a b : Type*) (dot : a → b → ℝ) : Prop :=
  dot a b < 0 → ∃ θ : ℝ, π / 2 < θ ∧ θ ≤ π

-- Proposition 3: direction vector and scalar multiples
def prop3 (a : Type*) (l : Type*) (λ : ℝ) : Prop :=
  (a = direction_vector l) → (λ * a = direction_vector l)

-- Proposition 4: coplanar non-zero vectors imply overall coplanarity
def prop4 (a b c : Type*) : Prop :=
  (is_nonzero a ∧ is_nonzero b ∧ is_nonzero c) →
  (coplanar a b ∧ coplanar b c ∧ coplanar c a) → coplanar a b c

-- The number of incorrect propositions is 4
theorem number_of_incorrect_props_is_four : nat :=
  let incorrect_count := 4 in
  incorrect_count

end number_of_incorrect_props_is_four_l683_683556


namespace find_b_collinear_l683_683313

theorem find_b_collinear (b : ℚ) : 
  Let p1 : prod ℚ ℚ := (5, -3)
  Let p2 : prod ℚ ℚ := (-b + 3, 5)
  Let p3 : prod ℚ ℚ := (3b + 1, 4)
  collinear p1 p2 p3 → b = 18 / 31 :=
by
  sorry

end find_b_collinear_l683_683313


namespace hair_cut_amount_l683_683454

theorem hair_cut_amount (initial_length final_length cut_length : ℕ) (h1 : initial_length = 11) (h2 : final_length = 7) : cut_length = 4 :=
by 
  sorry

end hair_cut_amount_l683_683454


namespace count_three_digit_numbers_divisible_by_13_l683_683598

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683598


namespace sequence_general_formula_sum_of_bn_sequence_l683_683248

variables {a : ℕ → ℕ} {S : ℕ → ℕ} {b : ℕ → ℕ} {T : ℕ → ℕ}

-- Conditions
axiom a1 : a 1 = 3
axiom a_recursive : ∀ n : ℕ, S n = (∑ i in finset.range (n + 1), a i) 
axiom a_recursive_next : ∀ n : ℕ, a (n + 1) = 2 * S n + 3

-- Definitions for \( S \), \( b \) and \( T \)
def S_def (n : ℕ) := ∑ i in finset.range (n + 1), a i
def b_def (n : ℕ) := (2 * n - 1) * a n
def T_def (n : ℕ) := ∑ i in finset.range (n + 1), b i

-- Theorem to be proven
theorem sequence_general_formula : ∀ n : ℕ, a n = 3^n :=
sorry

theorem sum_of_bn_sequence : ∀ n : ℕ, T n = (n - 1) * 3^(n+1) + 3 :=
sorry

end sequence_general_formula_sum_of_bn_sequence_l683_683248


namespace distinct_values_modulo_999_l683_683170

theorem distinct_values_modulo_999 :
  ∃ n : ℕ, n = 15 ∧
  ∀ (x : ℤ), x^9 % 999 =
    match x % 37 with
    | 0  => x^9 % 27 in {0, 1, 26}
    | 1  => x^9 % 27 in {0, 1, 26}
    | 36 => x^9 % 27 in {0, 1, 26}
    | 6  => x^9 % 27 in {0, 1, 26}
    | 31 => x^9 % 27 in {0, 1, 26}
    | _  => false

end distinct_values_modulo_999_l683_683170


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683591

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683591


namespace slow_population_growth_before_ir_l683_683463

-- Define the conditions
def low_level_social_productivity_before_ir : Prop := sorry
def high_birth_rate_before_ir : Prop := sorry
def high_mortality_rate_before_ir : Prop := sorry

-- The correct answer
def low_natural_population_growth_rate_before_ir : Prop := sorry

-- The theorem to prove
theorem slow_population_growth_before_ir 
  (h1 : low_level_social_productivity_before_ir) 
  (h2 : high_birth_rate_before_ir) 
  (h3 : high_mortality_rate_before_ir) : low_natural_population_growth_rate_before_ir := 
sorry

end slow_population_growth_before_ir_l683_683463


namespace largest_of_three_consecutive_integers_l683_683322

theorem largest_of_three_consecutive_integers (N : ℤ) (h : N + (N + 1) + (N + 2) = 18) : N + 2 = 7 :=
sorry

end largest_of_three_consecutive_integers_l683_683322


namespace find_angle_B_l683_683540

variable (ABC : Triangle)
variable (P : Point)
variable (R_ABC R_APC : ℝ)
variable (α : ℝ)

-- Condition: P is the incenter of triangle ABC
axiom P_is_incenter : P = incenter ABC

-- Condition: Circumradius of triangles ABC and APC are equal
axiom equal_circumradii : R_ABC = R_APC

-- Required to prove: α = angle B in triangle ABC is 60 degrees.
theorem find_angle_B (h1 : P_is_incenter) (h2 : equal_circumradii) : α = 60 := 
by 
  -- Skip the proof part
  sorry

end find_angle_B_l683_683540


namespace greatest_integer_a_exists_l683_683080

theorem greatest_integer_a_exists (a x : ℤ) (h : (x - a) * (x - 7) + 3 = 0) : a ≤ 11 := by
  sorry

end greatest_integer_a_exists_l683_683080


namespace find_a_if_x_is_1_root_l683_683138

theorem find_a_if_x_is_1_root {a : ℝ} (h : (1 : ℝ)^2 + a * 1 - 2 = 0) : a = 1 :=
by sorry

end find_a_if_x_is_1_root_l683_683138


namespace polynomial_root_existence_l683_683139

noncomputable def exists_root_with_bounds (n : ℕ) (C : Fin (n + 1) → ℂ) : Prop :=
  ∃ z_0 : ℂ, ∥z_0∥ ≤ 1 ∧ ∥(Finset.range (n + 1)).sum (λ k, C k * z_0^(n - k))∥ ≥ ∥C 0∥ + ∥C n∥

theorem polynomial_root_existence (n : ℕ) (C : Fin (n + 1) → ℂ) (hC0 : 0 < n) :
  exists_root_with_bounds n C :=
sorry

end polynomial_root_existence_l683_683139


namespace triangle_PAB_area_probability_l683_683995

-- Definitions
def AB : ℝ := 2
def AC : ℝ := 5
def cosA : ℝ := 4 / 5
noncomputable def sinA : ℝ := real.sqrt(1 - cosA ^ 2)
noncomputable def area_ABC : ℝ := (1 / 2) * AB * AC * sinA

-- Probability for area PAB to be in specific range
def probability_area_PAB_in_range : ℝ :=
  if 1 < area_ABC ∧ area_ABC ≤ 2 then 1 / 3 else 0

-- Statement to be proved
theorem triangle_PAB_area_probability (h : AB = 2) (h2 : AC = 5) (h3 : cosA = 4 / 5) :
  probability_area_PAB_in_range = 1 / 3 :=
sorry

end triangle_PAB_area_probability_l683_683995


namespace three_digit_numbers_divisible_by_13_count_l683_683626

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683626


namespace fraction_above_line_l683_683298

-- Define the coordinates of the vertices of the square
def vertex1 : ℝ × ℝ := (2, 0)
def vertex2 : ℝ × ℝ := (5, 0)
def vertex3 : ℝ × ℝ := (5, 3)
def vertex4 : ℝ × ℝ := (2, 3)

-- Define the points through which the line passes
def pointA : ℝ × ℝ := (2, 3)
def pointB : ℝ × ℝ := (5, 0)

theorem fraction_above_line (pA pB : ℝ × ℝ) (v1 v2 v3 v4 : ℝ × ℝ) 
  (hA : pA = (2, 3)) (hB : pB = (5, 0))
  (hv1 : v1 = (2, 0)) (hv2 : v2 = (5, 0)) (hv3 : v3 = (5, 3)) (hv4 : v4 = (2, 3)) :
  let area_triangle := 1 / 2 * (v2.1 - v1.1) * (v3.2 - v1.2),
      area_square := (v2.1 - v1.1) * (v3.2 - v1.2) in
  (area_square - area_triangle) / area_square = 1 / 2 := 
by
  sorry

end fraction_above_line_l683_683298


namespace triangle_is_isosceles_l683_683969

theorem triangle_is_isosceles {A B C a b c : ℝ} 
  (h1 : ∀ (A B C : ℝ), A + B + C = π)
  (h2 : ∀ (a b c A B C : ℝ), a = 2 * real.sin A * real.cos B)
  (h3 : ∀ (a b c A B C : ℝ), b = 2 * real.sin B * real.cos A)
  (h4 : a * real.cos B = b * real.cos A) : 
  A = B :=
by 
  sorry

end triangle_is_isosceles_l683_683969


namespace paint_room_together_l683_683483

variable (t : ℚ)
variable (Doug_rate : ℚ := 1/5)
variable (Dave_rate : ℚ := 1/7)
variable (Diana_rate : ℚ := 1/6)
variable (Combined_rate : ℚ := Doug_rate + Dave_rate + Diana_rate)
variable (break_time : ℚ := 2)

theorem paint_room_together:
  Combined_rate * (t - break_time) = 1 :=
sorry

end paint_room_together_l683_683483


namespace count_3digit_numbers_div_by_13_l683_683890

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683890


namespace solve_system_of_inequalities_l683_683506

theorem solve_system_of_inequalities (p : ℚ) : (19 * p < 10) ∧ (1 / 2 < p) → (1 / 2 < p ∧ p < 10 / 19) :=
by
  sorry

end solve_system_of_inequalities_l683_683506


namespace largest_of_three_consecutive_integers_l683_683316

theorem largest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 18) : x + 2 = 7 := 
sorry

end largest_of_three_consecutive_integers_l683_683316


namespace find_number_of_valid_fractions_l683_683469

open Nat

theorem find_number_of_valid_fractions (
    α : ℤ,
    hα : α = 6
) : 
    let count_valid_fractions := λ n : ℤ, n > 1 ∧ n < 7 
    in (Finset.range 10).filter (λ n, count_valid_fractions n).card = 5 := 
by
    sorry

end find_number_of_valid_fractions_l683_683469


namespace sum_of_integers_from_80_to_100_l683_683353

theorem sum_of_integers_from_80_to_100 :
  (∑ i in Finset.range (100 + 1), if 80 ≤ i then i else 0) = 1890 :=
by
  sorry

end sum_of_integers_from_80_to_100_l683_683353


namespace count_3_digit_numbers_divisible_by_13_l683_683701

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683701


namespace number_of_3_digit_divisible_by_13_l683_683815

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683815


namespace tan_alpha_eq_cot_sin_product_eq_l683_683135

variable (α : ℝ)

theorem tan_alpha_eq : 3 * sin α + 4 * cos α = 5 → tan α = 3 / 4 := 
by 
  intro h
  sorry

theorem cot_sin_product_eq : 3 * sin α + 4 * cos α = 5 →
  (cot (3 * Real.pi / 2 - α)) * (sin (3 * Real.pi / 2 + α))^2 = 12 / 25 := 
by 
  intro h
  sorry

end tan_alpha_eq_cot_sin_product_eq_l683_683135


namespace infinite_sum_equals_fraction_l683_683107

def closest_integer_to_sqrt (n : ℕ) : ℤ :=
  if h : 0 < n then (Nat.floor (Real.sqrt n.toReal)).toInt 
  else 0

theorem infinite_sum_equals_fraction :
  (∑ n in (Finset.range (n+1)).erase 0, ((3 ^ closest_integer_to_sqrt n) + (3 ^ (- closest_integer_to_sqrt n))) / (3 ^ n))
  = 39 / 27 := by
  sorry

end infinite_sum_equals_fraction_l683_683107


namespace lost_card_number_l683_683391

variable (n : ℕ)
variable (s : ℕ)

-- Axioms based on the given conditions.
axiom sum_of_first_n : s = n * (n + 1) / 2
axiom remaining_sum : s - 101 ∈ finset.singleton 101

-- The theorem we need to prove:
theorem lost_card_number (n : ℕ) (s : ℕ) (h₁ : s = n * (n + 1) / 2) (h₂ : s - 101 = 101) : n = 14 → (n * (n + 1) / 2 - 101 = 4) :=
by
  sorry

end lost_card_number_l683_683391


namespace cube_surface_area_including_inside_l683_683450

theorem cube_surface_area_including_inside (edge_length large_hole_length : ℝ) 
  (h_edge : edge_length = 4)
  (h_hole : large_hole_length = 2) : 
  let original_surface_area := 6 * (edge_length^2)
  let removed_area := 6 * (large_hole_length^2)
  let added_internal_area := 6 * (large_hole_length^2)
  original_surface_area - removed_area + added_internal_area = 168 :=
by
  -- Definitions of the variables.
  have h1 : original_surface_area = 6 * (edge_length^2), from rfl,
  have h2 : removed_area = 6 * (large_hole_length^2), from rfl,
  have h3 : added_internal_area = 6 * (large_hole_length^2), from rfl,
  
  -- Substitute the given values.
  rw [h_edge, h_hole] at *,
  
  -- Substitute in the surface area equation.
  have h_surface : (6 * (4^2)) - (6 * (2^2)) + (6 * (2^2)) = 168, by
    calculate

  exact h_surface,

  -- Here we add 
  sorry

end cube_surface_area_including_inside_l683_683450


namespace polynomial_sum_explicit_form_l683_683055

-- Define the problem statement using given conditions
variable {n : ℕ}
variable (f_i : ℕ → Polynomial ℝ)

-- Assume that each f_i is from problem 61050 (this would be expanded based on specific properties in problem 61050)
axiom problem_61050_condition : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → Polynomial ℝ

-- Define the polynomial f(x) as the sum of given polynomials
def f (x : ℝ) : Polynomial ℝ := ∑ i in Finset.range (nat.succ n), f_i i

-- The goal is to prove that f(x) = 1
theorem polynomial_sum_explicit_form (h : n > 0) : f = 1 := by
  -- Given axioms and definitions
  sorry

end polynomial_sum_explicit_form_l683_683055


namespace count_3digit_numbers_div_by_13_l683_683888

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683888


namespace squirrel_rise_per_circuit_l683_683021

noncomputable def rise_per_circuit
    (height : ℕ)
    (circumference : ℕ)
    (distance : ℕ) :=
    height / (distance / circumference)

theorem squirrel_rise_per_circuit : rise_per_circuit 25 3 15 = 5 :=
by
  sorry

end squirrel_rise_per_circuit_l683_683021


namespace count_three_digit_numbers_divisible_by_13_l683_683606

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683606


namespace chord_length_squared_of_tangent_circles_l683_683337

noncomputable def circle (r : ℝ) := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r ^ 2 }

theorem chord_length_squared_of_tangent_circles :
  ∀ (O5 O10 O15 : ℝ × ℝ) (P Q : ℝ × ℝ),
    let R5 := 5
    let R10 := 10
    let R15 := 15
    circle R5 ≠ ∅ ∧ circle R10 ≠ ∅ ∧ circle R15 ≠ ∅
    ∧ dist O5 O10 = R5 + R10
    ∧ dist O5 O15 = R15 - R5
    ∧ dist O10 O15 = R15 - R10
    ∧ (∃ A5 A10 : ℝ × ℝ, dist P A5 = dist Q A5 ∧ dist P A10 = dist Q A10) →
  dist P Q ^ 2 = 500 :=
by
  sorry

end chord_length_squared_of_tangent_circles_l683_683337


namespace count_3_digit_numbers_divisible_by_13_l683_683907

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683907


namespace equation_of_line_l683_683142

theorem equation_of_line :
  ∃ l : ℝ → ℝ → Prop,
    (∀ (x y : ℝ), (x^2 + (y-3)^2 = 4 → l x y)) ∧ 
    (∀ (x y : ℝ), (x + y + 1 = 0 ↔ l x y)) ↔ l = (λ x y : ℝ, x - y + 3 = 0) :=
sorry

end equation_of_line_l683_683142


namespace water_consumption_correct_l683_683494

theorem water_consumption_correct (w n r : ℝ) 
  (hw : w = 21428) 
  (hn : n = 26848.55) 
  (hr : r = 302790.13) :
  w = 21428 ∧ n = 26848.55 ∧ r = 302790.13 :=
by 
  sorry

end water_consumption_correct_l683_683494


namespace dot_product_EC_ED_l683_683983

-- Using definitions derived from the conditions
def is_midpoint (A B E : Point) : Prop := 
  dist A E = dist E B ∧ dist E B = dist A B / 2

def square (A B C D : Point) (s : ℝ) : Prop := 
  dist A B = s ∧ dist B C = s ∧ dist C D = s ∧ dist D A = s ∧
  dist A C = sqrt (s^2 + s^2) ∧ dist B D = sqrt (s^2 + s^2)

-- Using variables and the actual theorem statement
variables (A B C D E : Point) (s : ℝ)
variables (hABCD : square A B C D s) (hE_midpoint : is_midpoint A B E)

theorem dot_product_EC_ED : 
  s = 2 → \(\overrightarrow{EC}\cdot \overrightarrow{ED} = 3\) := 
sorry

end dot_product_EC_ED_l683_683983


namespace best_estimate_volume_l683_683433

/- 
  Conditions:
  1. A large cylinder can hold 50 L of chocolate milk when full.
  2. The tick marks show the division of the cylinder into four parts of equal volume.
  3. The milk level is slightly below three-fourths of the total volume.
  
  Question:
  Prove that the best estimate for the volume of chocolate milk in the cylinder is 36 L.
-/

noncomputable def volume_full : ℝ := 50
noncomputable def parts : ℝ := 4
noncomputable def volume_per_part : ℝ := volume_full / parts
noncomputable def milk_level_fraction : ℝ := 3 / 4

theorem best_estimate_volume : volume_per_part * milk_level_fraction * parts - 1 = 36 :=
by
synth sorry  -- Leaving the proof as a placeholder

end best_estimate_volume_l683_683433


namespace three_digit_numbers_divisible_by_13_l683_683747

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683747


namespace auto_parts_company_profit_l683_683464

theorem auto_parts_company_profit:
  (2017_profit: ℝ) (2019_profit: ℝ)
  (profit_2017_eq: 2017_profit = 2) (profit_2019_eq: 2019_profit = 3.38)
  (growth_rate: ℝ)
  (average_annual_growth_rate_eq: (1 + growth_rate)^2 = 1.69) :
  (average_annual_growth_rate_eq: growth_rate = 0.3) ∧ 
  (2020_profit_eq: 2017_profit * ((1 + growth_rate) ^ 3) > 4) :=
by { sorry }

end auto_parts_company_profit_l683_683464


namespace drunken_walk_tree_integer_steps_l683_683007

-- Defining necessary structures and notations
noncomputable theory
open_locale classical

def expected_steps {V : Type} [DecidableEq V] (graph : set (V × V)) (a b : V) : ℝ := sorry

theorem drunken_walk_tree_integer_steps {V : Type} [DecidableEq V] [Fintype V] 
(graph : set (V × V)) (tree : ∀ (u v : V), (u, v) ∈ graph → (v, u) ∈ graph)
(a b : V) : 
    (∀ v : V, ∃! u : V, ∀ w : V, (v, w) ∈ graph → u = w ∨ (w, u) ∉ graph) → 
    ∃ n : ℤ, expected_steps graph a b = n :=
by
    -- all proofs deferred
    sorry


end drunken_walk_tree_integer_steps_l683_683007


namespace tangent_lines_l683_683164

theorem tangent_lines (Γ1 Γ2 Γ3 : Circle) (l : Line) (A B C E F X Y Z : Point)
  (h_tangent1 : Γ1.tangent l A) 
  (h_tangent2 : Γ2.tangent l B)
  (h_tangent3 : Γ3.tangent l C)
  (h_between : between B A C)
  (h_tangent_ext1 : external_tangent Γ2 Γ1 E)
  (h_tangent_ext2 : external_tangent Γ2 Γ3 F)
  (h_common_tangent1 : common_external_tangent Γ1 Γ3)
  (h_common_tangent2 : intersects_at Γ2 X Y)
  (h_perpendicular : perpendicular (line_through B) l)
  (h_intersection : intersects_at Γ2 B Z)
  (h_not_coincident : Z ≠ B) :
  tangent (line_through Z X)
    (circle_with_diameter A C)
    ∧ tangent (line_through Z Y)
    (circle_with_diameter A C) := sorry

end tangent_lines_l683_683164


namespace count_three_digit_numbers_divisible_by_13_l683_683650

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683650


namespace count_three_digit_numbers_divisible_by_13_l683_683655

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683655


namespace count_three_digit_numbers_divisible_by_13_l683_683617

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683617


namespace find_positive_integers_l683_683085

noncomputable def positive_integer_solutions_ineq (x : ℕ) : Prop :=
  x > 0 ∧ (x : ℝ) < 4

theorem find_positive_integers (x : ℕ) : 
  (x > 0 ∧ (↑x - 3)/3 < 7 - 5*(↑x)/3) ↔ positive_integer_solutions_ineq x :=
by
  sorry

end find_positive_integers_l683_683085


namespace area_transformation_l683_683307

def g : ℝ → ℝ := sorry

theorem area_transformation (h : ∫ x in (a : ℝ), g x = 15 ) : 
  (∫ x in (a : ℝ), 4 * g (x + 3)) = 60 :=
by
  sorry

end area_transformation_l683_683307


namespace system_of_inequalities_solution_l683_683514

theorem system_of_inequalities_solution (p : ℝ) (h1 : 19 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 10 / 19 := by
  sorry

end system_of_inequalities_solution_l683_683514


namespace find_lost_card_number_l683_683387

theorem find_lost_card_number (n : ℕ) (S : ℕ) (x : ℕ) 
  (h1 : S = n * (n + 1) / 2) 
  (h2 : S - x = 101) 
  (h3 : n = 14) : 
  x = 4 := 
by sorry

end find_lost_card_number_l683_683387


namespace euclidean_algorithm_steps_l683_683275

theorem euclidean_algorithm_steps (a b : ℕ) (n k : ℕ)
  (h1 : a > b)
  (h2 : b ≥ nat.fib (n + 1))
  (h3 : 10^(k - 1) ≤ b ∧ b < 10^k) :
  n ≤ 5 * k := 
sorry

end euclidean_algorithm_steps_l683_683275


namespace count_3_digit_numbers_divisible_by_13_l683_683705

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683705


namespace number_of_ordered_quadruples_nonnegative_reals_l683_683083

open Real

theorem number_of_ordered_quadruples_nonnegative_reals :
  (Finset.card {abcd : ℝ × ℝ × ℝ × ℝ |
    let a := abcd.1, b := abcd.2.1, c := abcd.2.2.1, d := abcd.2.2.2 in
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
    a^2 + b^2 + c^2 + d^2 = 9 ∧
    (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81 }) = 15 := by
  sorry

end number_of_ordered_quadruples_nonnegative_reals_l683_683083


namespace count_three_digit_numbers_divisible_by_13_l683_683656

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683656


namespace num_three_digit_div_by_13_l683_683857

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683857


namespace count_three_digit_numbers_divisible_by_13_l683_683924

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683924


namespace three_digit_numbers_div_by_13_l683_683719

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683719


namespace neg_prop_l683_683301

theorem neg_prop : (¬∃ x : ℝ, sin x > 1) ↔ (∀ x : ℝ, sin x ≤ 1) :=
sorry

end neg_prop_l683_683301


namespace pairs_1_and_3_exhibit_strong_correlation_l683_683167

variable {n1 n2 n3 n4 : ℕ}
variable {r1 r2 r3 r4 : ℝ}

def strong_linear_correlation (n : ℕ) (r : ℝ) : Prop :=
  r.abs > 0.8 ∧ n > 5

theorem pairs_1_and_3_exhibit_strong_correlation
  (h1 : n1 = 7) (h2 : r1 = 0.9533)
  (h3 : n2 = 15) (h4 : r2 = 0.301)
  (h5 : n3 = 17) (h6 : r3 = 0.9991)
  (h7 : n4 = 3) (h8 : r4 = 0.9950) :
  strong_linear_correlation n1 r1 ∧ strong_linear_correlation n3 r3 :=
by
  sorry

end pairs_1_and_3_exhibit_strong_correlation_l683_683167


namespace firework_burst_time_l683_683576

theorem firework_burst_time : 
  ∃ t : ℝ, (h t = -3.6 * t^2 + 28.8 * t) ∧ t = 4 :=
begin
  -- Definition of h as given h(t) = -3.6 * t^2 + 28.8 * t
  let h := λ t : ℝ, -3.6 * t^2 + 28.8 * t,
  -- Show that the maximum height occurs at t = 4
  use 4,
  -- Sorry to skip the proof
  sorry
end

end firework_burst_time_l683_683576


namespace problem_statement_l683_683175

theorem problem_statement (m : ℤ) (h : (m + 2)^2 = 64) : (m + 1) * (m + 3) = 63 :=
sorry

end problem_statement_l683_683175


namespace count_3_digit_numbers_divisible_by_13_l683_683665

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683665


namespace evaluate_series_l683_683095

noncomputable def closest_int_sqrt (n : ℕ) : ℤ :=
  round (real.sqrt n)

theorem evaluate_series : 
  (∑' (n : ℕ) in set.Icc 1 (nat.mul (set.Ioi 1)) (3 ^ (closest_int_sqrt n) + 3 ^ -(closest_int_sqrt n)) / 3 ^ n) = 3 :=
begin
  sorry
end

end evaluate_series_l683_683095


namespace count_3_digit_multiples_of_13_l683_683784

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683784


namespace num_of_sets_eq_four_l683_683303

open Finset

theorem num_of_sets_eq_four : ∀ B : Finset ℕ, (insert 1 (insert 2 B) = {1, 2, 3, 4, 5}) → (B = {3, 4, 5} ∨ B = {1, 3, 4, 5} ∨ B = {2, 3, 4, 5} ∨ B = {1, 2, 3, 4, 5}) := 
by
  sorry

end num_of_sets_eq_four_l683_683303


namespace three_digit_numbers_divisible_by_13_count_l683_683627

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683627


namespace minimize_total_cost_l683_683008

theorem minimize_total_cost (
  (daily_flour_need : ℕ),
  (flour_price : ℕ),
  (storage_cost_per_ton_per_day : ℕ),
  (shipping_fee : ℕ)
)
  (daily_flour_need = 6)
  (flour_price = 1800)
  (storage_cost_per_ton_per_day = 3)
  (shipping_fee = 900) :
  (min_days : ℕ) (min_days = 10) :=
sorry

end minimize_total_cost_l683_683008


namespace ellens_initial_legos_l683_683063

-- Define the initial number of Legos as a proof goal
theorem ellens_initial_legos : ∀ (x y : ℕ), (y = x - 17) → (x = 2080) :=
by
  intros x y h
  sorry

end ellens_initial_legos_l683_683063


namespace three_digit_numbers_div_by_13_l683_683733

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683733


namespace find_y_common_solution_l683_683060

theorem find_y_common_solution (y : ℝ) :
  (∃ x : ℝ, x^2 + y^2 = 11 ∧ x^2 = 4*y - 7) ↔ (7/4 ≤ y ∧ y ≤ Real.sqrt 11) :=
by
  sorry

end find_y_common_solution_l683_683060


namespace stratified_sampling_elderly_employees_l683_683005

-- Definitions for the conditions
def total_employees : ℕ := 430
def young_employees : ℕ := 160
def middle_aged_employees : ℕ := 180
def elderly_employees : ℕ := 90
def sample_young_employees : ℕ := 32

-- The property we want to prove
theorem stratified_sampling_elderly_employees :
  (sample_young_employees / young_employees) * elderly_employees = 18 :=
by
  sorry

end stratified_sampling_elderly_employees_l683_683005


namespace count_3_digit_numbers_divisible_by_13_l683_683862

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683862


namespace find_radius_of_circle_l683_683154

theorem find_radius_of_circle (r : ℝ) (h1 : r > 0)
    (line_eq : ∀ (x y : ℝ), 3 * x - 4 * y + 5 = 0)
    (circle_eq : ∀ (x y : ℝ), x^2 + y^2 = r^2)
    (A B : ℝ × ℝ)
    (ha : line_eq A.1 A.2)
    (hb : line_eq B.1 B.2)
    (ha_circle : circle_eq A.1 A.2)
    (hb_circle : circle_eq B.1 B.2)
    (angle_AOB : ∠(A - (0, 0)) (B - (0, 0)) = 120) :
  r = 2 :=
by sorry

end find_radius_of_circle_l683_683154


namespace digit_100_of_27_by_250_l683_683342

noncomputable def decimal_representation (n d : ℕ) : ℕ × ℝ :=
  (n, n / d)

theorem digit_100_of_27_by_250 :
  let decimal : ℝ := 27 / 250 in
  (decimal * 10^100).floor % 10 = 0 :=
by
  sorry

end digit_100_of_27_by_250_l683_683342


namespace sum_nonnegative_reals_l683_683967

variable {x y z : ℝ}

theorem sum_nonnegative_reals (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 24) : 
  x + y + z = 10 := 
by sorry

end sum_nonnegative_reals_l683_683967


namespace count_three_digit_numbers_divisible_by_13_l683_683605

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683605


namespace switches_in_position_A_l683_683285

/-- 
  There are 500 switches, each with four positions A, B, C, and D. Each switch's label is of 
  the form \( 2^{x_i} 3^{y_i} 5^{z_i} 7^{w_i} \) where \( x_i, y_i, z_i, w_i \) range from 0 to 9. 
  Initially, all switches are in position A. At each step i, switch i and all switches whose labels 
  divide the label of switch i advance one position. 
  The goal is to prove that after 500 steps, the number of switches in position A is 469. 
-/
theorem switches_in_position_A :
  let switch_labels := λ i, ∃ (x_i y_i z_i w_i : ℕ), 0 ≤ x_i ∧ x_i ≤ 9 ∧ 
                                          0 ≤ y_i ∧ y_i ≤ 9 ∧ 
                                          0 ≤ z_i ∧ z_i ≤ 9 ∧ 
                                          0 ≤ w_i ∧ w_i ≤ 9 ∧ 
                                          i = 2 ^ x_i * 3 ^ y_i * 5 ^ z_i * 7 ^ w_i 
  in 
  let divisor_count := λ d_i, ∃ (x_i y_i z_i w_i : ℕ), d_i = (x_i + 1) * (y_i + 1) * (z_i + 1) * (w_i + 1)
  in
  let counts := ∑ i in range(1, 501), if (divisor_count i) % 4 = 0 then 1 else 0
  in counts = 469 :=
by sorry

end switches_in_position_A_l683_683285


namespace number_of_3_digit_divisible_by_13_l683_683809

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683809


namespace count_3_digit_numbers_divisible_by_13_l683_683876

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683876


namespace concurrency_of_tangent_lines_l683_683241

-- Definitions for the given conditions
variables {Γ Γ1 Γ2 Γ3 : Circle} -- Circles Γ, Γ1, Γ2, Γ3
variables {A1 B1 C1 A B C : Point} -- Points of tangency and intersection

-- The conditions translated to Lean definitions
def tangent_to (Γ1 Γ2 : Circle) (P : Point) : Prop :=
  Γ1.is_tangent_at P ∧ Γ2.is_tangent_at P

def common_tangent_intersect_at (Γ1 Γ2 : Circle) (P : Point) : Prop :=
  ∃ l : Line, l.is_tangent_to Γ1 ∧ l.is_tangent_to Γ2 ∧ P ∈ l

-- The statement we need to prove, given the conditions
theorem concurrency_of_tangent_lines 
  (H1 : tangent_to Γ1 Γ A1)
  (H2 : tangent_to Γ2 Γ B1)
  (H3 : tangent_to Γ3 Γ C1)
  (H4 : common_tangent_intersect_at Γ2 Γ3 A)
  (H5 : common_tangent_intersect_at Γ1 Γ3 B)
  (H6 : common_tangent_intersect_at Γ2 Γ1 C) :
  concurrent (line_through_points A A1) (line_through_points B B1) (line_through_points C C1) :=
sorry

end concurrency_of_tangent_lines_l683_683241


namespace num_3_digit_div_by_13_l683_683835

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683835


namespace num_three_digit_div_by_13_l683_683840

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683840


namespace num_3_digit_div_by_13_l683_683831

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683831


namespace Kyle_papers_delivered_each_week_proof_l683_683234

-- Definitions based on identified conditions
def k_m := 100        -- Number of papers delivered from Monday to Saturday
def d_m := 6          -- Number of days from Monday to Saturday
def k_s1 := 90        -- Number of regular customers on Sunday
def k_s2 := 30        -- Number of Sunday-only customers

-- Total number of papers delivered in a week
def total_papers_week := (k_m * d_m) + (k_s1 + k_s2)

theorem Kyle_papers_delivered_each_week_proof :
  total_papers_week = 720 :=
by
  sorry

end Kyle_papers_delivered_each_week_proof_l683_683234


namespace transformed_sum_of_coeffs_l683_683471

theorem transformed_sum_of_coeffs : 
  let P := polynomial.C 1 * polynomial.X ^ 4 + 
            polynomial.C 2 * polynomial.X ^ 3 + 
            polynomial.C 6 * polynomial.X ^ 2 + 
            polynomial.C 8 * polynomial.X + 
            polynomial.C 16 in
  let roots := (P.roots : multiset ℂ) in
  ∀ (w₁ w₂ w₃ w₄ : ℂ),
    (w₁ ∈ roots) →
    (w₂ ∈ roots) →
    (w₃ ∈ roots) →
    (w₄ ∈ roots) →
    roots.card = 4 → 
    let g := λ (z : ℂ), -2 * complex.conj(z) in
    let new_roots := {g w₁, g w₂, g w₃, g w₄} in
    let Q := polynomial.C 1 * polynomial.X ^ 4 + 
             polynomial.C 0 * polynomial.X ^ 3 + 
             polynomial.C 24 * polynomial.X ^ 2 + 
             polynomial.C 0 * polynomial.X + 
             polynomial.C 256 in
    Q.coeff 2 + Q.coeff 0 = 280 :=
begin
  sorry
end

end transformed_sum_of_coeffs_l683_683471


namespace alice_souvenir_cost_l683_683453

def yen_to_usd (yen : ℝ) (rate : ℝ) : ℝ := yen / rate
def discounted_price (price : ℝ) (discount : ℝ) : ℝ := price - (price * discount / 100)

theorem alice_souvenir_cost :
  let original_price := 300
  let discount_rate := 10
  let conversion_rate := 120
  let final_price := discounted_price original_price discount_rate
  (yen_to_usd final_price conversion_rate).round = 2.25 :=
by
  let original_price := 300
  let discount_rate := 10
  let conversion_rate := 120
  let final_price := discounted_price original_price discount_rate
  sorry

end alice_souvenir_cost_l683_683453


namespace problem_statement_l683_683046

def compute_floor (n : ℕ) : ℤ :=
  Int.floor (((n + 1)^3 / ((n - 1) * n)) - ((n - 1)^3 / (n * (n + 1))))

theorem problem_statement : compute_floor 2009 = 8 := by
  sorry

end problem_statement_l683_683046


namespace num_3_digit_div_by_13_l683_683834

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683834


namespace smallest_solution_l683_683505

theorem smallest_solution :
  ∃ x y : ℕ, 5 * x ^ 2 = 3 * y ^ 5 ∧ (∀ z w : ℕ, 5 * z ^ 2 = 3 * w ^ 5 → (x <= z ∧ y <= w)) :=
begin
  use [675, 15],
  split,
  { norm_num },
  { intros z w h,
    split;
    sorry
  }
end

end smallest_solution_l683_683505


namespace product_gcf_lcm_l683_683254

def gcf (a b c : Nat) : Nat := Nat.gcd (Nat.gcd a b) c
def lcm (a b c : Nat) : Nat := Nat.lcm (Nat.lcm a b) c

theorem product_gcf_lcm :
  let A := gcf 6 18 24
  let B := lcm 6 18 24
  A * B = 432 :=
by
  let A := gcf 6 18 24
  let B := lcm 6 18 24
  have hA : A = Nat.gcd (Nat.gcd 6 18) 24 := rfl
  have hB : B = Nat.lcm (Nat.lcm 6 18) 24 := rfl
  sorry

end product_gcf_lcm_l683_683254


namespace chocolate_chips_per_cookie_l683_683231

theorem chocolate_chips_per_cookie
  (num_batches : ℕ)
  (cookies_per_batch : ℕ)
  (num_people : ℕ)
  (chocolate_chips_per_person : ℕ) :
  (num_batches = 3) →
  (cookies_per_batch = 12) →
  (num_people = 4) →
  (chocolate_chips_per_person = 18) →
  (chocolate_chips_per_person / (num_batches * cookies_per_batch / num_people) = 2) :=
by
  sorry

end chocolate_chips_per_cookie_l683_683231


namespace count_3_digit_numbers_divisible_by_13_l683_683915

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683915


namespace three_digit_numbers_divisible_by_13_count_l683_683635

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683635


namespace min_value_x_plus_inv_x_l683_683546

theorem min_value_x_plus_inv_x (x : ℝ) (hx : x > 0) : ∃ y, (y = x + 1/x) ∧ (∀ z, z = x + 1/x → z ≥ 2) :=
by
  sorry

end min_value_x_plus_inv_x_l683_683546


namespace number_of_3_digit_divisible_by_13_l683_683808

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683808


namespace find_radius_circle_l683_683537

variables (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (AC BC : ℝ)
variables (θ : ℝ) (h_AB : ℝ) (h_CH : ℝ) (R : ℝ)

noncomputable def triangle_abc := 
  right_triangle AC BC θ

noncomputable def line_through_C := 
  through_line_C C θ

axiom (AC_eq_3 : AC = 3)
axiom (BC_eq_4 : BC = 4)
axiom (theta_eq_45 : θ = 45)
axiom (hypotenuse_AB : h_AB = 5)
axiom (height_CH : h_CH = 12 / 5)
axiom (radius : R = (35 * (5 * Real.sqrt(2) ± 4 * Real.sqrt(3))) / 2)

theorem find_radius_circle : 
  ∃ (R : ℝ), ∀ AC BC θ, AC = 3 → BC = 4 → θ = 45 → 
  right_triangle AC BC θ → through_line_C C θ → R = (35 * (5 * Real.sqrt(2) ± 4 * Real.sqrt(3))) / 2 := 
begin
  use (35 * (5 * Real.sqrt(2) ± 4 * Real.sqrt(3))) / 2,
  assume AC_eq_3 BC_eq_4 theta_eq_45 h_triangle h_line_through_C,
  sorry
end

end find_radius_circle_l683_683537


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683592

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683592


namespace count_3_digit_numbers_divisible_by_13_l683_683671

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683671


namespace angle_between_twice_a_neg_b_l683_683184

variables (a b : ℝ) (angle_between : ℝ → ℝ → ℂ → ℂ → ℝ)

-- Given: angle between a and b is 60 degrees
axiom angle_ab : angle_between a b ≔ 60

-- Prove: angle between 2a and -b is 120 degrees
theorem angle_between_twice_a_neg_b : angle_between (2 * a) (-b) = 120 :=
by sorry

end angle_between_twice_a_neg_b_l683_683184


namespace sum_of_k_values_l683_683294

theorem sum_of_k_values 
  (h : ∀ (k : ℤ), (∀ x y : ℤ, x * y = 15 → x + y = k) → k > 0 → false) : 
  ∃ k_values : List ℤ, 
  (∀ (k : ℤ), k ∈ k_values → (∀ x y : ℤ, x * y = 15 → x + y = k) ∧ k > 0) ∧ 
  k_values.sum = 24 := sorry

end sum_of_k_values_l683_683294


namespace projection_equals_l683_683015

noncomputable def projection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  let u := (1/√26, -5/√26)
  let uT := ![u.1, u.2]
  let uuT := uT ⬝ uT.transpose
  let I := 1 - uuT
  uuT

def proj_vec (v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  projection_matrix ⬝ v

def vec_2 : Fin 2 → ℝ := ![2, -5]
def vec_1_2 : Fin 2 → ℝ := ![1/2, -5/2]
def vec_3_6 : Fin 2 → ℝ := ![3, 6]
def vec_correct_proj : Fin 2 → ℝ := ![-27/26, 135/26]

axiom projection_condition : proj_vec vec_2 = vec_1_2

theorem projection_equals {-27/26, 135/26} :
  proj_vec(vec_3_6) = vec_correct_proj :=
by
  -- Proof placeholder
  sorry

end projection_equals_l683_683015


namespace arithmetic_seq_max_n_l683_683980

def arithmetic_seq_max_sum (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) : Prop :=
  (a 1 > 0) ∧ (3 * (a 1 + 4 * d) = 5 * (a 1 + 7 * d)) ∧
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d) ∧
  (S 12 = -72 * d)

theorem arithmetic_seq_max_n
  (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) : 
  arithmetic_seq_max_sum a d S → n = 12 :=
by
  sorry

end arithmetic_seq_max_n_l683_683980


namespace num_3_digit_div_by_13_l683_683827

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683827


namespace necessary_but_not_sufficient_l683_683169

def vector := ℝ × ℝ

def a : vector := (1, 1)
def b (x : ℝ) : vector := (x, -1)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def orthogonal (v1 v2 : vector) : Prop :=
  dot_product v1 v2 = 0

theorem necessary_but_not_sufficient (x : ℝ) :
  orthogonal (a.1 + b(x).1, a.2 + b(x).2) (b x) ↔ x = 0 :=
sorry

end necessary_but_not_sufficient_l683_683169


namespace smallest_m_for_integral_solutions_l683_683478

theorem smallest_m_for_integral_solutions :
  ∃ (m : ℕ), (∀ (x : ℤ), (12 * x^2 - m * x + 504 = 0 → ∃ (p q : ℤ), p + q = m / 12 ∧ p * q = 42)) ∧
  m = 156 := by
sorry

end smallest_m_for_integral_solutions_l683_683478


namespace number_of_three_digit_numbers_divisible_by_13_l683_683952

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683952


namespace Paula_bumper_cars_rides_l683_683272

axiom Paula_rides : ℕ -- Paula rides the bumper cars this many times
axiom go_kart_rides : ℕ := 1
axiom go_kart_cost : ℕ := 4
axiom bumper_car_cost : ℕ := 5
axiom total_tickets : ℕ := 24

theorem Paula_bumper_cars_rides (h1 : go_kart_rides * go_kart_cost + Paula_rides * bumper_car_cost = total_tickets) :
  Paula_rides = 4 :=
by
  -- problem statement here
  sorry

end Paula_bumper_cars_rides_l683_683272


namespace largest_of_three_consecutive_integers_l683_683317

theorem largest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 18) : x + 2 = 7 := 
sorry

end largest_of_three_consecutive_integers_l683_683317


namespace count_3_digit_multiples_of_13_l683_683795

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683795


namespace g_g_g_g_of_one_plus_i_l683_683091

def g (z : ℂ) : ℂ :=
  if z.im = 0 then -(z ^ 3)
  else z ^ 3

theorem g_g_g_g_of_one_plus_i :
  g (g (g (g (1 + (1:ℂ) * Complex.I)))) = -8192 - 45056 * Complex.I :=
by
  sorry

end g_g_g_g_of_one_plus_i_l683_683091


namespace count_3digit_numbers_div_by_13_l683_683884

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683884


namespace count_three_digit_numbers_divisible_by_13_l683_683929

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683929


namespace find_the_number_l683_683089

theorem find_the_number (x : ℕ) (h : 18396 * x = 183868020) : x = 9990 :=
by
  sorry

end find_the_number_l683_683089


namespace num_pos_3_digit_div_by_13_l683_683775

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683775


namespace count_3_digit_multiples_of_13_l683_683794

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683794


namespace tangents_at_common_point_same_l683_683152

-- Definitions of f and g on (0, +∞)
def f (x : ℝ) : ℝ := x^2 - 5
def g (x : ℝ) : ℝ := 6 * Real.log x - 4 * x

-- Derivatives
def f_prime (x : ℝ) : ℝ := 2 * x
def g_prime (x : ℝ) : ℝ := 6 / x - 4

-- Assuming the common point x_0 and function values and derivatives are equal
theorem tangents_at_common_point_same (x₀ : ℝ) (h₀ : x₀ > 0) (hf : f x₀ = g x₀) (hf_prime : f_prime x₀ = g_prime x₀) :
  5 = 5 :=
  sorry

end tangents_at_common_point_same_l683_683152


namespace initial_ripe_peaches_l683_683000

theorem initial_ripe_peaches (P U R: ℕ) (H1: P = 18) (H2: 2 * 5 = 10) (H3: (U + 7) + U = 15 - 3) (H4: R + 10 = U + 7) : 
  R = 1 :=
by
  sorry

end initial_ripe_peaches_l683_683000


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683596

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683596


namespace volume_relation_tetrahedron_l683_683181

theorem volume_relation_tetrahedron (A B C D O : Point)
  (hO_inside : inside_tetrahedron O A B C D) :
  volume(O, B, C, D) • vec(A, O) + 
  volume(O, A, C, D) • vec(B, O) + 
  volume(O, A, B, D) • vec(C, O) + 
  volume(O, A, B, C) • vec(D, O) = 0 := 
sorry

end volume_relation_tetrahedron_l683_683181


namespace coefficient_of_y_in_second_equation_l683_683110

theorem coefficient_of_y_in_second_equation :
  ∃ (a x y : ℝ), 
  (5 * x + y = 19) ∧ 
  (x + a * y = 1) ∧ 
  (3 * x + 2 * y = 10) ∧ 
  (a = 3) :=
begin
  sorry
end

end coefficient_of_y_in_second_equation_l683_683110


namespace find_a_l683_683150

theorem find_a (a : ℝ) (extreme_at_neg_2 : ∀ x : ℝ, (3 * a * x^2 + 2 * x) = 0 → x = -2) :
    a = 1 / 3 :=
sorry

end find_a_l683_683150


namespace count_3digit_numbers_div_by_13_l683_683897

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683897


namespace count_3digit_numbers_div_by_13_l683_683881

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683881


namespace count_three_digit_div_by_13_l683_683687

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683687


namespace range_of_a_min_value_f_l683_683561

def f (a x : ℝ) : ℝ := (a / x) - x + a * log x

def extreme_points (f : ℝ → ℝ → ℝ) (a : ℝ) : Prop :=
 ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (f a x1 = 0) ∧ (f a x2 = 0)

theorem range_of_a (a : ℝ) (h : extreme_points f a) : 4 < a :=
sorry

theorem min_value_f (a : ℝ) (h : 4 < a) : (λ (x1 x2 : ℝ), f a x1 + f a x2 - 3 * a) (x1 x2) = -real.exp 2 :=
sorry

end range_of_a_min_value_f_l683_683561


namespace chord_length_l683_683429

theorem chord_length (radius : ℝ) (distance_to_chord : ℝ) (chord_length : ℝ) :
  radius = 5 → distance_to_chord = 4 → chord_length = 6 :=
by
  intro hradius,
  intro hdistance_to_chord,
  simp [hradius, hdistance_to_chord],
  sorry

end chord_length_l683_683429


namespace alexio_card_probability_l683_683026

noncomputable def card_probability_multiple : ℚ :=
  let n := 200
  let multiples_4 := finset.filter (λ x, x % 4 = 0) (finset.range (n + 1)).card
  let multiples_5 := finset.filter (λ x, x % 5 = 0) (finset.range (n + 1)).card
  let multiples_6 := finset.filter (λ x, x % 6 = 0) (finset.range (n + 1)).card
  let multiples_4_5 := finset.filter (λ x, x % (nat.lcm 4 5) = 0) (finset.range (n + 1)).card
  let multiples_4_6 := finset.filter (λ x, x % (nat.lcm 4 6) = 0) (finset.range (n + 1)).card
  let multiples_5_6 := finset.filter (λ x, x % (nat.lcm 5 6) = 0) (finset.range (n + 1)).card
  let multiples_4_5_6 := finset.filter (λ x, x % (nat.lcm (nat.lcm 4 5) 6) = 0) (finset.range (n + 1)).card
  let total_multiples := multiples_4 + multiples_5 + multiples_6 
                        - multiples_4_5 - multiples_4_6 - multiples_5_6 
                        + multiples_4_5_6
  (total_multiples : ℚ) / n

theorem alexio_card_probability : card_probability_multiple = 47 / 100 := by 
  sorry

end alexio_card_probability_l683_683026


namespace small_bottles_initial_l683_683443

theorem small_bottles_initial
  (S : ℤ)
  (big_bottles_initial : ℤ := 15000)
  (sold_small_bottles_percentage : ℚ := 0.11)
  (sold_big_bottles_percentage : ℚ := 0.12)
  (remaining_bottles_in_storage : ℤ := 18540)
  (remaining_small_bottles : ℚ := 0.89 * S)
  (remaining_big_bottles : ℚ := 0.88 * big_bottles_initial)
  (h : remaining_small_bottles + remaining_big_bottles = remaining_bottles_in_storage)
  : S = 6000 :=
by
  sorry

end small_bottles_initial_l683_683443


namespace three_digit_numbers_divisible_by_13_count_l683_683636

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683636


namespace four_digit_number_sum_l683_683069

theorem four_digit_number_sum (x y z w : ℕ) (h1 : 1001 * x + 101 * y + 11 * z + 2 * w = 2003)
  (h2 : x = 1) : (x = 1 ∧ y = 9 ∧ z = 7 ∧ w = 8) ↔ (1000 * x + 100 * y + 10 * z + w = 1978) :=
by sorry

end four_digit_number_sum_l683_683069


namespace A_symmetric_diff_B_count_min_X_l683_683111

-- Define the function f_M(x)
def f_M (M : Set ℕ) (x : ℕ) : Int :=
if x ∈ M then -1 else 1

-- Define the symmetric difference based on f_M function.
def symmetric_difference (M N : Set ℕ) : Set ℕ :=
{x | f_M M x * f_M N x = -1}

-- Define sets A and B
def A : Set ℕ := {2, 4, 6, 8, 10}
def B : Set ℕ := {1, 2, 4, 8, 16}

-- Prove that A △ B is as specified.
theorem A_symmetric_diff_B :
  symmetric_difference A B = {1, 6, 10, 16} :=
sorry

-- Count the number of sets X that minimize Card(X △ A) + Card(X △ B)
theorem count_min_X :
  (∃ X : Set ℕ,
   (symmetric_difference X A).card + (symmetric_difference X B).card = 4) ∧
   (∀ X : Set ℕ, (symmetric_difference X A).card + (symmetric_difference X B).card = 4 →
    X ⊆ (A ∪ B) ∧ X ⊇ (A ∩ B)) ∧
   finset.card (finset.powerset {1, 6, 10, 16}.to_finset) = 16 :=
sorry

end A_symmetric_diff_B_count_min_X_l683_683111


namespace three_digit_numbers_divisible_by_13_count_l683_683622

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683622


namespace temperature_ordering_l683_683191

def frequency_data : list ℕ := 
  [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

def freq_counts : list ℕ := 
  [15, 10, 8, 8, 8, 7, 7, 2, 2, 2]

def temperature_values : list ℕ :=
  (frequency_data.zip freq_counts).bind (λ pair, list.repeat (pair.fst) pair.snd)

noncomputable def mean_temperature : ℝ :=
  let sum_temperatures := (temperature_values.map (λ x, (x : ℝ))).sum
  let n := (temperature_values.length : ℝ)
  sum_temperatures / n

noncomputable def median_temperature : ℝ :=
  let sorted_values := temperature_values.sort
  sorted_values.nth (sorted_values.length / 2)

noncomputable def mode_temperature : ℕ :=
  frequency_data.zip freq_counts |>.max_by (λ pair, pair.snd) |>.fst

theorem temperature_ordering :
  let μ := mean_temperature
  let M := median_temperature
  let d := mode_temperature
  d < M ∧ M < μ :=
by
  sorry

end temperature_ordering_l683_683191


namespace total_bricks_l683_683287

theorem total_bricks (n1 n2 r1 r2 : ℕ) (w1 w2 : ℕ)
  (h1 : n1 = 60) (h2 : r1 = 100) (h3 : n2 = 80) (h4 : r2 = 120)
  (h5 : w1 = 5) (h6 : w2 = 5) :
  (w1 * (n1 * r1) + w2 * (n2 * r2)) = 78000 :=
by sorry

end total_bricks_l683_683287


namespace positive_3_digit_numbers_divisible_by_13_count_l683_683584

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l683_683584


namespace quadratic_roots_l683_683072

theorem quadratic_roots (a b k : ℝ) (h₁ : a + b = -2) (h₂ : a * b = k / 3)
    (h₃ : |a - b| = 1/2 * (a^2 + b^2)) : k = 0 ∨ k = 6 :=
sorry

end quadratic_roots_l683_683072


namespace count_three_digit_div_by_13_l683_683693

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683693


namespace intersection_times_l683_683411

def distance_squared_motorcyclist (t: ℝ) : ℝ := (72 * t)^2
def distance_squared_bicyclist (t: ℝ) : ℝ := (36 * (t - 1))^2
def law_of_cosines (t: ℝ) : ℝ := distance_squared_motorcyclist t +
                                      distance_squared_bicyclist t -
                                      2 * 72 * 36 * |t| * |t - 1| * (1/2)

def equation_simplified (t: ℝ) : ℝ := 4 * t^2 + t^2 - 2 * |t| * |t - 1|

theorem intersection_times :
  ∀ t: ℝ, (0 < t ∨ t < 1) → equation_simplified t = 49 → (t = 4 ∨ t = -4) := 
by
  intros t ht_eq
  intro h
  sorry

end intersection_times_l683_683411


namespace num_3_digit_div_by_13_l683_683824

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683824


namespace problem1_problem2_problem3_l683_683143

-- Problem 1: Angle AOB = pi/2 implies k = ±√7
theorem problem1 (k : ℝ) (h1 : ∀ (A B : ℝ × ℝ), A ≠ B
                  → (A = (-2, k * -2 - 4) ∨ A = (2, k * 2 - 4))
                  → (B = (-2, k * -2 - 4) ∨ B = (2, k * 2 - 4))
                  → ∃ O, O = (0, 0)
                  ∧ ∃∠ AOB, (∠AOB = real.pi / 2)) :
  k = sqrt 7 ∨ k = -sqrt 7 := sorry

-- Problem 2: Determine the fixed point for line CD when k=1
theorem problem2 (h2 : ∃ (P : ℝ × ℝ), (P.2 = 1 * P.1 - 4)
                  ∧ ∀ (C D : ℝ × ℝ), (C ≠ D)
                  ∧ (circle (0, 0) 2)
                  ∧ tangents_from P C circle
                  ∧ tangents_from P D circle
                  ∧ passes_through_fixed_point line_CD) :
  ∃ (fixed_point : ℝ × ℝ), fixed_point = (1, -1) := sorry

-- Problem 3: Maximum area of quadrilateral EGFH
theorem problem3 (h3 : circle (0, 0) 2)
                  (EF GH : line)
                  (perpendicular_chords EF GH)
                  (intersection_point_M : (1, sqrt 2)) :
  max_area_quadrilateral EF GH = 5 := sorry

end problem1_problem2_problem3_l683_683143


namespace rational_terms_count_l683_683310

-- Assuming n is chosen such that 2 * binom(n, 2) = 9900, we compute the number of rational terms in the expansion
theorem rational_terms_count (n : ℕ) (h : 2 * (n.choose 2) = 9900) : 
  ∃ k_set : Finset ℕ, k_set.card = 9 ∧ ∀ k ∈ k_set, (k % 3 = 0 ∧ (100 - k) % 4 = 0) :=
begin
  sorry
end

end rational_terms_count_l683_683310


namespace player_wins_if_bottom_right_wins_player_wins_if_bottom_right_loses_l683_683420

-- Definitions based on conditions from part a)
def chessboard : Type :=
  array 8 (array 8 ℕ)

def init_position : (ℕ × ℕ) := (1, 1)

def moves (pos : ℕ × ℕ) : list (ℕ × ℕ) :=
  [(pos.1, pos.2 + 1), (pos.1, pos.2 + 2), (pos.1, pos.2 + 3), (pos.1, pos.2 + 4),
   (pos.1 + 1, pos.2), (pos.1 + 2, pos.2), (pos.1 + 3, pos.2)]

-- The two parts of the problem as theorem statements in Lean
theorem player_wins_if_bottom_right_wins :
  ∀ board: chessboard, (1, 1) → (8, 8) ∃ strategy, winning strategy
:= sorry

theorem player_wins_if_bottom_right_loses :
  ∀ board: chessboard, (1, 1) → (8, 8) ∃ strategy, losing strategy
:= sorry

end player_wins_if_bottom_right_wins_player_wins_if_bottom_right_loses_l683_683420


namespace lost_card_number_l683_683380

theorem lost_card_number (n x : ℕ) (sum_n : ℕ) (h_sum : sum_n = n * (n + 1) / 2) (h_remaining_sum : sum_n - x = 101) : x = 4 :=
sorry

end lost_card_number_l683_683380


namespace trajectory_of_M_eqn_l683_683573

-- Define the points F1 and F2
def F1 : ℝ × ℝ := (-4, 0)
def F2 : ℝ × ℝ := (4, 0)

-- Define the distance function between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
    real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the sum of distances condition for point M
def sum_of_distances (M : ℝ × ℝ) : ℝ :=
    distance M F1 + distance M F2

-- Statement to prove the equation of the trajectory of point M
theorem trajectory_of_M_eqn (M : ℝ × ℝ) : 
  sum_of_distances M = 10 ↔ ∃ (a b : ℝ) (h1 : a = 5) (h2 : b = 3), (M.1 / a)^2 + (M.2 / b)^2 = 1 :=
by {
  sorry
}

end trajectory_of_M_eqn_l683_683573


namespace Parabola_vertex_form_l683_683991

theorem Parabola_vertex_form (x : ℝ) (y : ℝ) : 
  (∃ h k : ℝ, (h = -2) ∧ (k = 1) ∧ (y = (x + h)^2 + k) ) ↔ (y = (x + 2)^2 + 1) :=
by
  sorry

end Parabola_vertex_form_l683_683991


namespace weight_of_replaced_person_l683_683290

theorem weight_of_replaced_person
  (increase_in_avg_weight : ℝ)
  (num_persons : ℕ)
  (new_person_weight : ℝ)
  (total_increase_in_weight : ℝ := num_persons * increase_in_avg_weight)
  (weight_of_replaced_person : ℝ := new_person_weight - total_increase_in_weight) :
  increase_in_avg_weight = 1.5 ∧
  num_persons = 9 ∧
  new_person_weight = 78.5 →
  weight_of_replaced_person = 65 := 
by
  intros h
  obtain ⟨h_avg, h_num, h_new⟩ := h
  unfold weight_of_replaced_person total_increase_in_weight
  rw [h_avg, h_num, h_new]
  norm_num
  sorry

end weight_of_replaced_person_l683_683290


namespace points_per_round_l683_683972

-- Definitions based on conditions
def final_points (jane_points : ℕ) : Prop := jane_points = 60
def lost_points (jane_lost : ℕ) : Prop := jane_lost = 20
def rounds_played (jane_rounds : ℕ) : Prop := jane_rounds = 8

-- The theorem we want to prove
theorem points_per_round (jane_points jane_lost jane_rounds points_per_round : ℕ) 
  (h1 : final_points jane_points) 
  (h2 : lost_points jane_lost) 
  (h3 : rounds_played jane_rounds) : 
  points_per_round = ((jane_points + jane_lost) / jane_rounds) := 
sorry

end points_per_round_l683_683972


namespace inclusion_exclusion_l683_683196

-- Defining the conditions as variables in Lean
variable (total_students : ℕ) -- Total number of students
variable (B : ℕ) -- Number of students who like basketball
variable (C : ℕ) -- Number of students who like cricket
variable (S : ℕ) -- Number of students who like soccer
variable (BC : ℕ) -- Number of students who like both basketball and cricket
variable (BS : ℕ) -- Number of students who like both basketball and soccer
variable (CS : ℕ) -- Number of students who like both cricket and soccer
variable (BCS : ℕ) -- Number of students who like all three sports

theorem inclusion_exclusion : 
  total_students = 50 ->
  B = 16 -> 
  C = 11 -> 
  S = 10 -> 
  BC = 5 -> 
  BS = 4 -> 
  CS = 3 -> 
  BCS = 2 ->
  B + C + S - BC - BS - CS + BCS = 27 :=
by
  intros h_total h_B h_C h_S h_BC h_BS h_CS h_BCS
  rw [h_B, h_C, h_S, h_BC, h_BS, h_CS, h_BCS]
  sorry

end inclusion_exclusion_l683_683196


namespace min_phi_l683_683296

example : ∃ (φ > 0), (∀ x : ℝ, (∀ k : ℤ, (φ = (π / 4) - k * π)) ∧ f(x - φ) = f(φ - x)) ∧ φ = π / 4 := 
sorry

noncomputable def f (x : ℝ) : ℝ := (real.sin x) + real.cos x

theorem min_phi : ∃ (φ > 0), (∀ x : ℝ, ( ∀ k : ℤ, (φ = (π / 4) - k * π)) ∧ ((λ x, f(x - φ)) = (λ x, f(φ - x)))) ∧ φ = π / 4 := 
sorry

end min_phi_l683_683296


namespace count_3_digit_numbers_divisible_by_13_l683_683864

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683864


namespace min_segment_length_l683_683124

theorem min_segment_length (a : ℝ) (h : a > 0) :
  ∃ (M N : ℝ × ℝ × ℝ),
  (line_through A A1 M) ∧ (line_through B C N) ∧
  (intersects_C1D1 M N) ∧
  (distance M N ≥ 3 * a)  :=
sorry

end min_segment_length_l683_683124


namespace count_three_digit_numbers_divisible_by_13_l683_683640

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683640


namespace placement_of_pawns_l683_683962

noncomputable def num_ways_to_place_pawns : ℕ :=
  5! * 5!

theorem placement_of_pawns :
  num_ways_to_place_pawns = 14400 :=
by sorry

end placement_of_pawns_l683_683962


namespace evaluate_series_l683_683099

def closest_integer (n : ℕ) : ℕ :=
  if sqrt n * sqrt n = n then sqrt n
  else if (sqrt n + 1) * (sqrt n + 1) - n < n - sqrt n * sqrt n then sqrt n + 1
  else sqrt n

theorem evaluate_series : 
  (∑' n : ℕ, (3^(closest_integer n) + 3^(-closest_integer n)) / 3^n) = 3 := 
sorry

end evaluate_series_l683_683099


namespace three_digit_numbers_div_by_13_l683_683737

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683737


namespace area_of_rectangle_l683_683306

theorem area_of_rectangle (a b : ℝ) (h1 : 2 * (a + b) = 16) (h2 : 2 * a^2 + 2 * b^2 = 68) :
  a * b = 15 :=
by
  have h3 : a + b = 8 := by sorry
  have h4 : a^2 + b^2 = 34 := by sorry
  have h5 : (a + b) ^ 2 = a^2 + b^2 + 2 * a * b := by sorry
  have h6 : 64 = 34 + 2 * a * b := by sorry
  have h7 : 2 * a * b = 30 := by sorry
  exact sorry

end area_of_rectangle_l683_683306


namespace count_three_digit_numbers_divisible_by_13_l683_683614

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683614


namespace number_of_good_circles_l683_683123

-- Definition of the 5 points on a plane
variables {P : Point ℝ} [fintype P] (points : finset P)
(h_points : points.card = 5)
-- Conditions: No three points are collinear
(h_collinear : ∀ (A B C : P), A ∈ points → B ∈ points → C ∈ points → 
  ¬ collinear ℝ {A, B, C})
-- No four points are concyclic
(h_concyclic : ∀ (A B C D : P), A ∈ points → B ∈ points → C ∈ points → D ∈ points → 
  ¬ concyclic ℝ {A, B, C, D})

-- Definition of a "good circle"
def good_circle (C : circle ℝ) (pts : finset P) : Prop :=
  ∃ (A B D inside outside : P), A ∈ pts ∧ B ∈ pts ∧ C ∈ pts ∧ D ∈ pts ∧ (inside ≠ outside) ∧ 
  (inside = A ∨ inside = B ∨ inside = C ∨ inside = D) ∧
  (outside = A ∨ outside = B ∨ outside = C ∨ outside = D) ∧
  (inside ∈ C.interior ∧ outside ∈ C.exterior)

-- Theorem: The number of "good circles" is exactly 4
theorem number_of_good_circles :
  ∀ points : finset P, points.card = 5 ∧ -- there are 5 points
  (∀ {A B C : P}, A ∈ points → B ∈ points → C ∈ points → ¬ collinear ℝ {A, B, C}) ∧
  (∀ {A B C D : P}, A ∈ points → B ∈ points → C ∈ points → D ∈ points → ¬ concyclic ℝ {A, B, C, D}) →
  (finset.filter (λ C, good_circle C points) (finset.powerset points)).card = 4 :=
sorry

end number_of_good_circles_l683_683123


namespace count_three_digit_numbers_divisible_by_13_l683_683644

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683644


namespace count_3_digit_numbers_divisible_by_13_l683_683909

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683909


namespace selling_price_correct_l683_683436

noncomputable def discount1 (price : ℝ) : ℝ := price * 0.85
noncomputable def discount2 (price : ℝ) : ℝ := price * 0.90
noncomputable def discount3 (price : ℝ) : ℝ := price * 0.95

noncomputable def final_price (initial_price : ℝ) : ℝ :=
  discount3 (discount2 (discount1 initial_price))

theorem selling_price_correct : final_price 3600 = 2616.30 := by
  sorry

end selling_price_correct_l683_683436


namespace count_3_digit_numbers_divisible_by_13_l683_683704

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683704


namespace count_three_digit_numbers_divisible_by_13_l683_683922

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683922


namespace num_3_digit_div_by_13_l683_683821

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683821


namespace total_water_needed_l683_683065

def adults : ℕ := 7
def children : ℕ := 3
def hours : ℕ := 24
def replenish_bottles : ℚ := 14
def water_per_hour_adult : ℚ := 1/2
def water_per_hour_child : ℚ := 1/3

theorem total_water_needed : 
  let total_water_per_hour := (adults * water_per_hour_adult) + (children * water_per_hour_child)
  let total_water := total_water_per_hour * hours 
  let initial_water_needed := total_water - replenish_bottles
  initial_water_needed = 94 := by 
  sorry

end total_water_needed_l683_683065


namespace num_pos_3_digit_div_by_13_l683_683761

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683761


namespace three_digit_numbers_divisible_by_13_count_l683_683633

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683633


namespace count_3_digit_numbers_divisible_by_13_l683_683712

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683712


namespace loci_tangent_centers_circle_or_ellipse_l683_683125

-- Define the problem conditions
variables {A O : Point} {R : ℝ} (hA_inside : dist A O < R)

-- Define the centers of circles tangent to the given circle
noncomputable def loci_of_tangent_centers : Set Point :=
  {C : Point | ∃ r : ℝ, dist A C = r ∧ dist O C = R - r}

-- The statement to prove
theorem loci_tangent_centers_circle_or_ellipse :
  loci_of_tangent_centers hA_inside = 
    if A = O then {C : Point | dist O C = R}
    else {C : Point | dist A C + dist O C = R ∧ dist O A < R} :=
sorry

end loci_tangent_centers_circle_or_ellipse_l683_683125


namespace count_three_digit_numbers_divisible_by_13_l683_683616

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683616


namespace AT_eq_TM_l683_683458

-- Definitions of points O, A, B, C, E, F, T, and M
variables (O A B C E F T M : Type) [IsCircumcenter ABC O]
variables [Intersects BO AC E]
variables [Intersects CO AB F]
variables [Intersects (extension AO) EF T]
variables [IntersectsCircle BOF COE M]

theorem AT_eq_TM : distance A T = distance T M := by
  sorry

end AT_eq_TM_l683_683458


namespace num_three_digit_div_by_13_l683_683854

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l683_683854


namespace three_digit_numbers_divisible_by_13_count_l683_683623

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683623


namespace max_height_reachable_l683_683364

/-- Define the conditions of the marble placements and movements. -/
structure SolitaireGame where
  initial_marbles : set (ℤ × ℤ)
  initial_marbles_condition : ∀ x ∈ ℤ, ∀ y ∈ ℤ, y ≤ 0 → (x, y) ∈ initial_marbles

/-- Define the maximum reachable height in the Solitaire game -/
theorem max_height_reachable (game : SolitaireGame) : ∃ h : ℕ, h = 4 := 
sorry

end max_height_reachable_l683_683364


namespace count_3_digit_numbers_divisible_by_13_l683_683905

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683905


namespace rectangle_diagonal_length_l683_683305

theorem rectangle_diagonal_length (P L W k d : ℝ) 
  (h1 : P = 72) 
  (h2 : L / W = 3 / 2) 
  (h3 : L = 3 * k) 
  (h4 : W = 2 * k) 
  (h5 : P = 2 * (L + W))
  (h6 : d = Real.sqrt ((L^2) + (W^2))) :
  d = 25.96 :=
by
  sorry

end rectangle_diagonal_length_l683_683305


namespace greatest_common_divisor_of_B_l683_683246

def is_in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2)

theorem greatest_common_divisor_of_B : gcd_forall (is_in_B) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l683_683246


namespace count_3_digit_numbers_divisible_by_13_l683_683707

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l683_683707


namespace forty_percent_of_number_l683_683407

theorem forty_percent_of_number (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 16) : (40/100) * N = 192 :=
by
  sorry

end forty_percent_of_number_l683_683407


namespace count_three_digit_numbers_divisible_by_13_l683_683926

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683926


namespace carrie_total_spend_l683_683404

def cost_per_tshirt : ℝ := 9.15
def number_of_tshirts : ℝ := 22

theorem carrie_total_spend : (cost_per_tshirt * number_of_tshirts) = 201.30 := by 
  sorry

end carrie_total_spend_l683_683404


namespace count_three_digit_numbers_divisible_by_13_l683_683608

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l683_683608


namespace power_of_two_expression_l683_683959

theorem power_of_two_expression :
  2^2010 - 2^2009 - 2^2008 + 2^2007 - 2^2006 = 5 * 2^2006 :=
by
  sorry

end power_of_two_expression_l683_683959


namespace count_3digit_numbers_div_by_13_l683_683887

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683887


namespace value_of_a_l683_683162

theorem value_of_a (a : ℝ) (A : set ℝ) (B : set ℝ) 
  (hA : A = {-1, 0, 2}) 
  (hB : B = {2^a}) 
  (hSubset : B ⊆ A) : a = 1 :=
sorry

end value_of_a_l683_683162


namespace constant_term_in_binomial_expansion_l683_683206

theorem constant_term_in_binomial_expansion (n : ℕ) :
  (∀ k, k < 3 → (k = 0 → (x + 1/(2 * (x^(1/3))))^{n} = c_{n, k} * x^{a_k}) ∧
            (k = 1 → c_{n, k} * x^{a_k} = (n / 2) * x^{a_{k-1}}) ∧
            (k = 2 → c_{n, k} * x^{a_k} = (n * (n - 1) / 8) * x^{a_{k-1}})) →
        (c_{8, 6} * (1 / 2)^6 = 7 / 16) := 
begin
  assume h,
  sorry
end

end constant_term_in_binomial_expansion_l683_683206


namespace profit_percent_l683_683435

theorem profit_percent (cost_price : ℝ) (selling_price : ℝ) (marked_price : ℝ) (n_pens : ℕ) 
  (h1 : n_pens = 60) (h2 : marked_price = 1) (h3 : cost_price = (46 : ℝ) / (60 : ℝ)) 
  (h4 : selling_price = 0.99 * marked_price) : 
  (selling_price - cost_price) / cost_price * 100 = 29.11 :=
by
  sorry

end profit_percent_l683_683435


namespace count_three_digit_numbers_divisible_by_13_l683_683931

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683931


namespace count_three_digit_numbers_divisible_by_13_l683_683638

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683638


namespace jills_initial_investment_l683_683227

theorem jills_initial_investment
  (A : ℝ) (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ)
  (hA : A = 10815.83)
  (hr : r = 0.0396)
  (hn : n = 2)
  (ht : t = 2)
  (compound_interest : A = P * (1 + r / n) ^ (n * t)) :
  P ≈ 10000 :=
by
  sorry

end jills_initial_investment_l683_683227


namespace lost_card_number_l683_683394

variable (n : ℕ)
variable (s : ℕ)

-- Axioms based on the given conditions.
axiom sum_of_first_n : s = n * (n + 1) / 2
axiom remaining_sum : s - 101 ∈ finset.singleton 101

-- The theorem we need to prove:
theorem lost_card_number (n : ℕ) (s : ℕ) (h₁ : s = n * (n + 1) / 2) (h₂ : s - 101 = 101) : n = 14 → (n * (n + 1) / 2 - 101 = 4) :=
by
  sorry

end lost_card_number_l683_683394


namespace three_digit_numbers_divisible_by_13_count_l683_683618

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683618


namespace three_digit_numbers_divisible_by_13_count_l683_683625

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l683_683625


namespace equivalent_expression_l683_683360

theorem equivalent_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 :=
sorry

end equivalent_expression_l683_683360


namespace count_three_digit_div_by_13_l683_683689

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683689


namespace find_a_l683_683156

def line_circle_intersection (a : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, 
    A ≠ B ∧
    (A.1 - A.2 - a = 0) ∧ 
    (B.1 - B.2 - a = 0) ∧ 
    (A.1^2 + A.2^2 = 4) ∧ 
    (B.1^2 + B.2^2 = 4) ∧ 
    (A ≠ (0, 0)) ∧ 
    (B ≠ (0, 0))

def is_equilateral_triangle (A B : ℝ × ℝ) : Prop :=
  let O : ℝ × ℝ := (0, 0) in
  dist O A = dist O B ∧
  dist O A = dist A B

theorem find_a (a : ℝ) :
  line_circle_intersection a ∧ ∃ A B : ℝ × ℝ, is_equilateral_triangle A B → a = √6 ∨ a = -√6 :=
begin
  sorry
end

end find_a_l683_683156


namespace count_three_digit_div_by_13_l683_683690

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683690


namespace count_three_digit_numbers_divisible_by_13_l683_683657

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683657


namespace three_digit_numbers_divisible_by_13_l683_683738

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683738


namespace max_m_value_max_a_value_l683_683151

theorem max_m_value (f g : ℝ → ℝ) (m : ℝ) (h0 : ∀ x, f x = |x+3|) (h1 : ∀ x, g x = m - 2|x-11|) (h2 : ∀ x, 2*f x ≥ g (x+4)) :
  m ≤ 20 :=
sorry

theorem max_a_value (x y z a t : ℝ) (h3 : 2 * x^2 + 3 * y^2 + 6 * z^2 = a) (h4 : a > 0) (h5 : t = 20) (h6 : ∀ x y z, x + y + z ≤ (t / 20)) :
  a = 1 :=
sorry

end max_m_value_max_a_value_l683_683151


namespace count_3_digit_numbers_divisible_by_13_l683_683676

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683676


namespace lost_card_number_l683_683390

variable (n : ℕ)
variable (s : ℕ)

-- Axioms based on the given conditions.
axiom sum_of_first_n : s = n * (n + 1) / 2
axiom remaining_sum : s - 101 ∈ finset.singleton 101

-- The theorem we need to prove:
theorem lost_card_number (n : ℕ) (s : ℕ) (h₁ : s = n * (n + 1) / 2) (h₂ : s - 101 = 101) : n = 14 → (n * (n + 1) / 2 - 101 = 4) :=
by
  sorry

end lost_card_number_l683_683390


namespace rational_root_is_integer_and_divisible_l683_683256

def is_integer (P : ℚ) : Prop := ∃ k : ℤ, P = k

theorem rational_root_is_integer_and_divisible (n : ℕ) (a : Fin n.succ → ℤ) (P : ℚ)
  (hP : (P.num : ℤ) ≠ 0 ∧ (nat.gcd P.num.nat_abs P.denom = 1)) 
  (h : (P^(n : ℤ) + ∑ i in Finset.range n, (a i : ℚ) * (P ^ (i : ℤ))) = 0) :
  is_integer P ∧ ∀ m : ℤ, (P - m : ℚ) ∣ (∑ i in Finset.range n, a i * (m ^ (i : ℤ)) - m^n) := 
by
  sorry

end rational_root_is_integer_and_divisible_l683_683256


namespace second_certificate_interest_rate_l683_683029

noncomputable def find_ann_rate_after_second_period (initial_investment : ℝ) (first_rate : ℝ) (first_duration_months : ℝ)
  (first_value : ℝ) (second_value : ℝ) : ℝ :=
  let first_interest := first_rate / 12 * first_duration_months
  let intermediate_value := initial_investment * (1 + first_interest / 100)
  let second_duration_months := 3
  let s := ((second_value / intermediate_value) - 1) * 400 / (second_duration_months / 12)
  s

theorem second_certificate_interest_rate
  (initial_investment : ℝ) 
  (first_rate : ℝ) 
  (first_duration_months : ℝ) 
  (first_value : ℝ) 
  (second_value : ℝ) 
  (h_initial : initial_investment = 12000) 
  (h_first_rate : first_rate = 8) 
  (h_first_duration : first_duration_months = 3) 
  (h_first_value : first_value = 12240)
  (h_second_value : second_value = 12435) :
  find_ann_rate_after_second_period initial_investment first_rate first_duration_months first_value second_value ≈ 6.38 :=
by
  -- Using the given conditions to show the equality
  sorry

end second_certificate_interest_rate_l683_683029


namespace river_depth_mid_July_l683_683982

theorem river_depth_mid_July :
  let d_May := 5
  let d_June := d_May + 10
  let d_July := 3 * d_June
  d_July = 45 :=
by
  sorry

end river_depth_mid_July_l683_683982


namespace eq_expression_l683_683357

theorem eq_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 :=
by
  sorry

end eq_expression_l683_683357


namespace infinitely_many_divisible_by_sum_of_digits_l683_683963

theorem infinitely_many_divisible_by_sum_of_digits :
    ∀ (n : ℕ), let number := (10^3^n - 1) / 9 in
    (number % (3^n) = 0) ∧ (∀ d ∈ digits 10 number, d ≠ 0) := 
sorry

end infinitely_many_divisible_by_sum_of_digits_l683_683963


namespace three_digit_numbers_div_by_13_l683_683723

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683723


namespace angle_between_2a_neg_b_l683_683183

variables (a b : ℝ → ℝ) -- treating 'a' and 'b' as vectors in a real-valued space
variables (angle : ℝ → ℝ → ℝ) -- function to denote the angle between two vectors

-- defining the angle condition
def angle_between_a_b : Prop := angle a b = 60

-- defining the goal: angle between 2a and -b
theorem angle_between_2a_neg_b (ha : angle a b = 60) : angle (λ x, 2 * a x) (λ x, - b x) = 120 := 
by {
  -- provided condition
  sorry
}

end angle_between_2a_neg_b_l683_683183


namespace num_pos_3_digit_div_by_13_l683_683773

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683773


namespace count_three_digit_div_by_13_l683_683696

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683696


namespace alcohol_percentage_in_new_mixture_l683_683421

theorem alcohol_percentage_in_new_mixture :
  let afterShaveLotionVolume := 200
  let afterShaveLotionConcentration := 0.35
  let solutionVolume := 75
  let solutionConcentration := 0.15
  let waterVolume := 50
  let totalVolume := afterShaveLotionVolume + solutionVolume + waterVolume
  let alcoholVolume := (afterShaveLotionVolume * afterShaveLotionConcentration) + (solutionVolume * solutionConcentration)
  let alcoholPercentage := (alcoholVolume / totalVolume) * 100
  alcoholPercentage = 25 := 
  sorry

end alcohol_percentage_in_new_mixture_l683_683421


namespace american_summits_more_water_l683_683523

-- Definitions based on the conditions
def FosterFarmsChickens := 45
def AmericanSummitsWater := 2 * FosterFarmsChickens
def HormelChickens := 3 * FosterFarmsChickens
def BoudinButchersChickens := HormelChickens / 3
def TotalItems := 375
def ItemsByFourCompanies := FosterFarmsChickens + AmericanSummitsWater + HormelChickens + BoudinButchersChickens
def DelMonteWater := TotalItems - ItemsByFourCompanies
def WaterDifference := AmericanSummitsWater - DelMonteWater

theorem american_summits_more_water : WaterDifference = 30 := by
  sorry

end american_summits_more_water_l683_683523


namespace count_3_digit_numbers_divisible_by_13_l683_683858

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683858


namespace expected_attempts_for_10_suitcases_l683_683432

noncomputable def expected_attempts (n : ℕ) : ℝ :=
  (1 / 2) * (n * (n + 1) / 2) + (n / 2) - (Real.log n + 0.577)

theorem expected_attempts_for_10_suitcases :
  abs (expected_attempts 10 - 29.62) < 1 :=
by
  sorry

end expected_attempts_for_10_suitcases_l683_683432


namespace count_3_digit_numbers_divisible_by_13_l683_683667

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683667


namespace parallelogram_D_value_l683_683277

theorem parallelogram_D_value :
  ∃ D : ℂ, let A : ℂ := 1 + 3 * complex.i,
                B : ℂ := 2 - complex.i,
                C : ℂ := -3 + complex.i,
                AB := B - A,
                DC := C - D 
            in AB = DC ∧ D = -4 + 5 * complex.i :=
by
  sorry

end parallelogram_D_value_l683_683277


namespace find_C_l683_683547

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def A : ℕ := sum_of_digits (4568 ^ 7777)
noncomputable def B : ℕ := sum_of_digits A
noncomputable def C : ℕ := sum_of_digits B

theorem find_C : C = 5 :=
by
  sorry

end find_C_l683_683547


namespace intersection_point_of_AB_CD_l683_683981

noncomputable def point : Type := {x y z : ℝ}

def A : point := {x := 3, y := -5, z := 4}
def B : point := {x := 13, y := -15, z := 9}
def C : point := {x := -6, y := 6, z := -12}
def D : point := {x := -4, y := -2, z := 8}

def intersection_point : point := {x := -4 / 3, y := 35, z := 3 / 2}

theorem intersection_point_of_AB_CD :
  ∃ t s : ℝ, 
  { x := A.x + t * (B.x - A.x), y := A.y + t * (B.y - A.y), z := A.z + t * (B.z - A.z) } =
  { x := C.x + s * (D.x - C.x), y := C.y + s * (D.y - C.y), z := C.z + s * (D.z - C.z) } ∧
  { x := A.x + t * (B.x - A.x), y := A.y + t * (B.y - A.y), z := A.z + t * (B.z - A.z) } = intersection_point := 
by
  sorry

end intersection_point_of_AB_CD_l683_683981


namespace total_spent_in_may_l683_683062

-- Conditions as definitions
def cost_per_weekday : ℕ := (2 * 15) + (2 * 18)
def cost_per_weekend_day : ℕ := (3 * 12) + (2 * 20)
def weekdays_in_may : ℕ := 22
def weekend_days_in_may : ℕ := 9

-- The statement to prove
theorem total_spent_in_may :
  cost_per_weekday * weekdays_in_may + cost_per_weekend_day * weekend_days_in_may = 2136 :=
by
  sorry

end total_spent_in_may_l683_683062


namespace num_3_digit_div_by_13_l683_683833

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683833


namespace lost_card_number_l683_683375

theorem lost_card_number (n : ℕ) (x : ℕ) (h₁ : (n * (n + 1)) / 2 - x = 101) : x = 4 :=
sorry

end lost_card_number_l683_683375


namespace max_m_for_inequality_l683_683161

theorem max_m_for_inequality (m : ℝ) : 
  (∀ x ∈ set.Ioc 0 1, x^2 - 4 * x - m ≥ 0) ↔ m ≤ -3 :=
by sorry

end max_m_for_inequality_l683_683161


namespace count_three_digit_numbers_divisible_by_13_l683_683934

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683934


namespace count_three_digit_numbers_divisible_by_13_l683_683937

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683937


namespace count_3_digit_multiples_of_13_l683_683796

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683796


namespace sergey_can_determine_contents_l683_683211

def box_contents (label : String) : String :=
  match label with
  | "red"   => "white or mixed"
  | "white" => "red or mixed"
  | "mixed" => "red or white"
  | _       => "invalid"

def sergey_strategy (label : String) (ball : String) : String :=
  match label, ball with
  | "mixed", "red"   => "all_red"
  | "mixed", "white" => "all_white"
  | _                => "invalid"

-- Sergey can determine the contents of the boxes by opening one box.
theorem sergey_can_determine_contents :
  ∀ (labels : List String) (ball_from_mixed : String),
    (labels.length = 3) ∧
    (labels.contains "red") ∧
    (labels.contains "white") ∧
    (labels.contains "mixed") ∧
    (labels.nodup) ∧
    (ball_from_mixed = "red" ∨ ball_from_mixed = "white") →
    (sergey_strategy "mixed" ball_from_mixed = "all_red" ∨
     sergey_strategy "mixed" ball_from_mixed = "all_white") :=
by
  intros
  sorry

end sergey_can_determine_contents_l683_683211


namespace min_n_Sn_greater_1020_l683_683308

theorem min_n_Sn_greater_1020 : ∃ n : ℕ, (n ≥ 0) ∧ (2^(n+1) - 2 - n > 1020) ∧ ∀ m : ℕ, (m ≥ 0) ∧ (m < n) → (2^(m+1) - 2 - m ≤ 1020) :=
by
  sorry

end min_n_Sn_greater_1020_l683_683308


namespace sin_double_angle_second_quadrant_l683_683136

open Real

theorem sin_double_angle_second_quadrant (α : ℝ) (h₁ : π < α ∧ α < 2 * π) (h₂ : sin (π - α) = 3 / 5) : sin (2 * α) = -24 / 25 :=
by sorry

end sin_double_angle_second_quadrant_l683_683136


namespace three_digit_numbers_divisible_by_13_l683_683739

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l683_683739


namespace largest_integer_a_l683_683078

theorem largest_integer_a (x a : ℤ) :
  ∃ x : ℤ, (x - a) * (x - 7) + 3 = 0 → a ≤ 11 :=
sorry

end largest_integer_a_l683_683078


namespace distance_between_foci_of_hyperbola_l683_683498

theorem distance_between_foci_of_hyperbola :
  (∀ (x y : ℝ), (y^2 / 25) - (x^2 / 16) = 1 → true) →
  (2 * real.sqrt 41 = 2 * real.sqrt 41) := 
by
  intro h
  trivial

end distance_between_foci_of_hyperbola_l683_683498


namespace evaluate_series_l683_683098

def closest_integer (n : ℕ) : ℕ :=
  if sqrt n * sqrt n = n then sqrt n
  else if (sqrt n + 1) * (sqrt n + 1) - n < n - sqrt n * sqrt n then sqrt n + 1
  else sqrt n

theorem evaluate_series : 
  (∑' n : ℕ, (3^(closest_integer n) + 3^(-closest_integer n)) / 3^n) = 3 := 
sorry

end evaluate_series_l683_683098


namespace board_coloring_condition_l683_683486

theorem board_coloring_condition (m n : ℕ) 
  (paint : Fin m → Fin n → Bool)
  (h : ∀ (i : Fin m) (j : Fin n), odd (neighbors_with_same_color paint i j)) : 
  even m ∨ even n := sorry

end board_coloring_condition_l683_683486


namespace michelle_scored_30_l683_683195

-- Define the total team points
def team_points : ℕ := 72

-- Define the number of other players
def num_other_players : ℕ := 7

-- Define the average points scored by the other players
def avg_points_other_players : ℕ := 6

-- Calculate the total points scored by the other players
def total_points_other_players : ℕ := num_other_players * avg_points_other_players

-- Define the points scored by Michelle
def michelle_points : ℕ := team_points - total_points_other_players

-- Prove that the points scored by Michelle is 30
theorem michelle_scored_30 : michelle_points = 30 :=
by
  -- Here would be the proof, but we skip it with sorry.
  sorry

end michelle_scored_30_l683_683195


namespace count_3_digit_numbers_divisible_by_13_l683_683903

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683903


namespace additional_houses_built_by_october_l683_683455

def total_houses : ℕ := 2000
def fraction_built_first_half : ℚ := 3 / 5
def houses_needed_by_october : ℕ := 500

def houses_built_first_half : ℚ := fraction_built_first_half * total_houses
def houses_built_by_october : ℕ := total_houses - houses_needed_by_october

theorem additional_houses_built_by_october :
  (houses_built_by_october - houses_built_first_half) = 300 := by
  sorry

end additional_houses_built_by_october_l683_683455


namespace lost_card_number_l683_683374

theorem lost_card_number (n : ℕ) (x : ℕ) (h₁ : (n * (n + 1)) / 2 - x = 101) : x = 4 :=
sorry

end lost_card_number_l683_683374


namespace number_of_three_digit_numbers_divisible_by_13_l683_683942

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683942


namespace count_three_digit_div_by_13_l683_683694

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683694


namespace Kyle_papers_delivered_each_week_proof_l683_683232

-- Definitions based on identified conditions
def k_m := 100        -- Number of papers delivered from Monday to Saturday
def d_m := 6          -- Number of days from Monday to Saturday
def k_s1 := 90        -- Number of regular customers on Sunday
def k_s2 := 30        -- Number of Sunday-only customers

-- Total number of papers delivered in a week
def total_papers_week := (k_m * d_m) + (k_s1 + k_s2)

theorem Kyle_papers_delivered_each_week_proof :
  total_papers_week = 720 :=
by
  sorry

end Kyle_papers_delivered_each_week_proof_l683_683232


namespace convert_1101011_to_base5_l683_683051

-- Define the conversion from binary to decimal
def binary_to_decimal (b : list ℕ) : ℕ :=
  b.reverse.zip_with (λ a i, a * 2^i) (list.range b.length) |>.sum

-- Define the conversion from decimal to base 5
def decimal_to_base5 (n : ℕ) : list ℕ :=
  if n = 0 then [0]
  else list.unfoldr (λ x, if x = 0 then none else some (x % 5, x / 5)) n |>.reverse

-- The specific problem statement
theorem convert_1101011_to_base5 :
  decimal_to_base5 (binary_to_decimal [1, 1, 0, 1, 0, 1, 1]) = [4, 1, 2] :=
by
  sorry

end convert_1101011_to_base5_l683_683051


namespace domain_of_lg_sin_2x_plus_sqrt_9_minus_x2_is_correct_l683_683292

theorem domain_of_lg_sin_2x_plus_sqrt_9_minus_x2_is_correct :
  (∀ x, (sin (2 * x) > 0 ∧ 9 - x^2 ≥ 0) → (x ∈ (-∞, - 3] ∪ [- (π / 2), 0] ∪ [0, (π / 2)] ∪ [(π / 2), 3])) :=
by
  sorry

end domain_of_lg_sin_2x_plus_sqrt_9_minus_x2_is_correct_l683_683292


namespace number_of_3_digit_divisible_by_13_l683_683816

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683816


namespace pipe_Q_drain_portion_l683_683273

noncomputable def portion_liquid_drain_by_Q (T_Q T_P T_R : ℝ) (h1 : T_P = 3 / 4 * T_Q) (h2 : T_R = T_P) : ℝ :=
  let rate_P := 1 / T_P
  let rate_Q := 1 / T_Q
  let rate_R := 1 / T_R
  let combined_rate := rate_P + rate_Q + rate_R
  (rate_Q / combined_rate)

theorem pipe_Q_drain_portion (T_Q T_P T_R : ℝ) (h1 : T_P = 3 / 4 * T_Q) (h2 : T_R = T_P) :
  portion_liquid_drain_by_Q T_Q T_P T_R h1 h2 = 3 / 11 :=
by
  sorry

end pipe_Q_drain_portion_l683_683273


namespace three_digit_numbers_div_by_13_l683_683728

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683728


namespace number_of_3_digit_divisible_by_13_l683_683813

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l683_683813


namespace angles_of_triangle_MBC_l683_683010

-- Geometric definitions and conditions
variables {A B C D M : Type} [EuclideanGeometry A B C D M]
hypothesis ABC_angle_condition : angle B A C = 110
hypothesis parallel_condition : parallel_through_point C (angle_bisector B D C) (line A B).extended.point M

-- Angles of triangle MBC
theorem angles_of_triangle_MBC : 
  angle B M C = 55 ∧ angle B C M = 55 ∧ angle M B C = 70 :=
sorry

end angles_of_triangle_MBC_l683_683010


namespace count_three_digit_numbers_divisible_by_13_l683_683643

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l683_683643


namespace three_digit_numbers_div_by_13_l683_683727

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l683_683727


namespace count_3_digit_multiples_of_13_l683_683780

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l683_683780


namespace count_3digit_numbers_div_by_13_l683_683891

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l683_683891


namespace evaluate_series_l683_683096

def closest_integer (n : ℕ) : ℕ :=
  if sqrt n * sqrt n = n then sqrt n
  else if (sqrt n + 1) * (sqrt n + 1) - n < n - sqrt n * sqrt n then sqrt n + 1
  else sqrt n

theorem evaluate_series : 
  (∑' n : ℕ, (3^(closest_integer n) + 3^(-closest_integer n)) / 3^n) = 3 := 
sorry

end evaluate_series_l683_683096


namespace dist_AB_l683_683572

def point3d := ℝ × ℝ × ℝ

def dist (A B : point3d) : ℝ :=
  match A, B with
  | (x1, y1, z1), (x2, y2, z2) =>
    Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

def A : point3d := (2, -1, 3)
def B : point3d := (-1, 4, -2)

theorem dist_AB : dist A B = Real.sqrt 59 := by
  sorry

end dist_AB_l683_683572


namespace no_solution_for_conditions_l683_683274

theorem no_solution_for_conditions :
  ∀ (x y : ℝ), 0 < x → 0 < y → x * y = 2^15 → (Real.log x / Real.log 2) * (Real.log y / Real.log 2) = 60 → False :=
by
  intro x y x_pos y_pos h1 h2
  sorry

end no_solution_for_conditions_l683_683274


namespace count_3_digit_numbers_divisible_by_13_l683_683866

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l683_683866


namespace number_of_three_digit_numbers_divisible_by_13_l683_683948

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l683_683948


namespace count_three_digit_numbers_divisible_by_13_l683_683921

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l683_683921


namespace rectangle_area_l683_683497

theorem rectangle_area (P : ℝ) (twice : ℝ → ℝ) (L W A : ℝ) 
  (h1 : P = 40) 
  (h2 : ∀ W, L = twice W) 
  (h3 : ∀ L W, P = 2 * L + 2 * W) 
  (h4 : ∀ L W, A = L * W) 
  (h5 : twice = (λ W, 2 * W)) :
  A = 800 / 9 := 
sorry

end rectangle_area_l683_683497


namespace num_3_digit_div_by_13_l683_683836

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l683_683836


namespace count_three_digit_div_by_13_l683_683695

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l683_683695


namespace intersection_times_l683_683410

def distance_squared_motorcyclist (t: ℝ) : ℝ := (72 * t)^2
def distance_squared_bicyclist (t: ℝ) : ℝ := (36 * (t - 1))^2
def law_of_cosines (t: ℝ) : ℝ := distance_squared_motorcyclist t +
                                      distance_squared_bicyclist t -
                                      2 * 72 * 36 * |t| * |t - 1| * (1/2)

def equation_simplified (t: ℝ) : ℝ := 4 * t^2 + t^2 - 2 * |t| * |t - 1|

theorem intersection_times :
  ∀ t: ℝ, (0 < t ∨ t < 1) → equation_simplified t = 49 → (t = 4 ∨ t = -4) := 
by
  intros t ht_eq
  intro h
  sorry

end intersection_times_l683_683410


namespace num_pos_3_digit_div_by_13_l683_683768

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l683_683768


namespace min_perimeter_rectangle_l683_683016

theorem min_perimeter_rectangle : 
    ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ 2 * (11 + 15) = 52 :=
by {
  use 1,
  use 3,
  simp,
  exact all_goals_nat.le_refl,
  sorry
}

end min_perimeter_rectangle_l683_683016


namespace inverse_of_square_l683_683960

theorem inverse_of_square (A : Matrix (Fin 2) (Fin 2) ℝ) (hA_inv : A⁻¹ = ![![3, 4], ![-2, -2]]) :
  (A^2)⁻¹ = ![![1, 4], ![-2, -4]] :=
by
  sorry

end inverse_of_square_l683_683960


namespace quadruples_satisfy_conditions_l683_683047

noncomputable def number_of_ordered_quadruples : ℂ := 24

theorem quadruples_satisfy_conditions :
  ∃ (w x y z : ℂ), 
  (w * x * y * z = 1) ∧
  (w * x * y^2 + w * x^2 * z + w^2 * y * z + x * y * z^2 = 2) ∧
  (w * x^2 * y + w^2 * y^2 + w^2 * x * z + x * y^2 * z + x^2 * z^2 + y * w * z^2 = -3) ∧
  (w^2 * x * y + x^2 * y * z + w * y^2 * z + w * x * z^2 = -1) :=
begin
  sorry
end

end quadruples_satisfy_conditions_l683_683047


namespace inequality_proof_l683_683118

-- Define the given conditions
def a : ℝ := Real.log 0.99
def b : ℝ := Real.exp 0.1
def c : ℝ := Real.exp (Real.log 0.99) ^ Real.exp 1

-- State the goal to be proved
theorem inequality_proof : a < c ∧ c < b := 
by
  sorry

end inequality_proof_l683_683118


namespace square_perimeter_l683_683193

theorem square_perimeter (total_area : ℝ) (overlapping_area : ℝ) (circle_area : ℝ) (square_perimeter : ℝ) 
  (h_total : total_area = 2018) 
  (h_overlap : overlapping_area = 137) 
  (h_circle : circle_area = 1371) 
  (h_square_perimeter : square_perimeter = 112) : 
  square_perimeter = 4 * real.sqrt (2018 - (1371 - 137)) := 
  by
    sorry

end square_perimeter_l683_683193


namespace least_five_digit_congruent_to_7_mod_18_l683_683347

theorem least_five_digit_congruent_to_7_mod_18 : 
  ∃ n, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 18 = 7 ∧ ∀ m, 10000 ≤ m ∧ m ≤ 99999 ∧ m % 18 = 7 → n ≤ m :=
  ∃ n, 10015 ≤ n ∧ n ≤ 99999 ∧ n % 18 = 7 ∧ ∀ m, 10000 ≤ m ∧ m ≤ 99999 ∧ m % 18 = 7 → n ≤ m :=
sorry

end least_five_digit_congruent_to_7_mod_18_l683_683347


namespace smallest_next_divisor_of_m_l683_683229

theorem smallest_next_divisor_of_m (m : ℕ) (h1 : m % 2 = 0) (h2 : 10000 ≤ m ∧ m < 100000) (h3 : 523 ∣ m) : 
  ∃ d : ℕ, 523 < d ∧ d ∣ m ∧ ∀ e : ℕ, 523 < e ∧ e ∣ m → d ≤ e :=
by
  sorry

end smallest_next_divisor_of_m_l683_683229
