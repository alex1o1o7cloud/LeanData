import Mathlib

namespace minimum_m_value_l1149_114974

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem minimum_m_value :
  (∃ m, ∀ (x1 x2 x3 : ℝ), 0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ Real.pi → |f x1 - f x2| + |f x2 - f x3| ≤ m) ∧
  ∀ m', (∀ (x1 x2 x3 : ℝ), 0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ Real.pi → |f x1 - f x2| + |f x2 - f x3| ≤ m') → 3 + Real.sqrt 3 / 2 ≤ m' :=
by
  sorry

end minimum_m_value_l1149_114974


namespace value_of_expression_l1149_114970

def g (x : ℝ) (p q r s t : ℝ) : ℝ :=
  p * x^4 + q * x^3 + r * x^2 + s * x + t

theorem value_of_expression (p q r s t : ℝ) (h : g (-1) p q r s t = 4) :
  12 * p - 6 * q + 3 * r - 2 * s + t = 13 :=
sorry

end value_of_expression_l1149_114970


namespace smallest_number_of_cubes_l1149_114932

def box_length : ℕ := 49
def box_width : ℕ := 42
def box_depth : ℕ := 14
def gcd_box_dimensions : ℕ := Nat.gcd (Nat.gcd box_length box_width) box_depth

theorem smallest_number_of_cubes :
  (box_length / gcd_box_dimensions) *
  (box_width / gcd_box_dimensions) *
  (box_depth / gcd_box_dimensions) = 84 := by
  sorry

end smallest_number_of_cubes_l1149_114932


namespace crayons_total_correct_l1149_114914

-- Definitions from the conditions
def initial_crayons : ℕ := 9
def added_crayons : ℕ := 3

-- Expected total crayons as per the conditions and the correct answer
def total_crayons_expected : ℕ := 12

-- The proof statement
theorem crayons_total_correct :
  initial_crayons + added_crayons = total_crayons_expected :=
by
  -- Proof details here
  sorry

end crayons_total_correct_l1149_114914


namespace find_m_plus_n_l1149_114940

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * x^2 + x

theorem find_m_plus_n (m n : ℝ) (h1 : m < n ∧ n ≤ 1) (h2 : ∀ (x : ℝ), m ≤ x ∧ x ≤ n → 3 * m ≤ f x ∧ f x ≤ 3 * n) : m + n = -4 :=
by
  have H1 : - (1 / 2) * m^2 + m = 3 * m := sorry
  have H2 : - (1 / 2) * n^2 + n = 3 * n := sorry
  sorry

end find_m_plus_n_l1149_114940


namespace factor_expression_l1149_114917

theorem factor_expression (y : ℝ) : 
  5 * y * (y + 2) + 8 * (y + 2) + 15 = (5 * y + 8) * (y + 2) + 15 := 
by
  sorry

end factor_expression_l1149_114917


namespace magical_stack_card_count_l1149_114913

theorem magical_stack_card_count :
  ∃ n, n = 157 + 78 ∧ 2 * n = 470 :=
by
  let n := 235
  use n
  have h1: n = 157 + 78 := by sorry
  have h2: 2 * n = 470 := by sorry
  exact ⟨h1, h2⟩

end magical_stack_card_count_l1149_114913


namespace all_girls_select_same_color_probability_l1149_114927

theorem all_girls_select_same_color_probability :
  let white_marbles := 10
  let black_marbles := 10
  let red_marbles := 10
  let girls := 15
  ∀ (total_marbles : ℕ), total_marbles = white_marbles + black_marbles + red_marbles →
  (white_marbles < girls ∧ black_marbles < girls ∧ red_marbles < girls) →
  0 = 0 :=
by
  intros
  sorry

end all_girls_select_same_color_probability_l1149_114927


namespace simplify_and_evaluate_l1149_114973

variable (a : ℚ)
variable (a_val : a = -1/2)

theorem simplify_and_evaluate : (4 - 3 * a) * (1 + 2 * a) - 3 * a * (1 - 2 * a) = 3 := by
  sorry

end simplify_and_evaluate_l1149_114973


namespace cylinder_surface_area_l1149_114946

noncomputable def total_surface_area_cylinder (r h : ℝ) : ℝ :=
  let base_area := 64 * Real.pi
  let lateral_surface_area := 2 * Real.pi * r * h
  let total_surface_area := 2 * base_area + lateral_surface_area
  total_surface_area

theorem cylinder_surface_area (r h : ℝ) (hr : Real.pi * r^2 = 64 * Real.pi) (hh : h = 2 * r) : 
  total_surface_area_cylinder r h = 384 * Real.pi := by
  sorry

end cylinder_surface_area_l1149_114946


namespace landscape_length_l1149_114909

-- Define the conditions from the problem
def breadth (b : ℝ) := b > 0
def length_of_landscape (l b : ℝ) := l = 8 * b
def area_of_playground (pg_area : ℝ) := pg_area = 1200
def playground_fraction (A b : ℝ) := A = 8 * b^2
def fraction_of_landscape (pg_area A : ℝ) := pg_area = (1/6) * A

-- Main theorem statement
theorem landscape_length (b l A pg_area : ℝ) 
  (H_b : breadth b) 
  (H_length : length_of_landscape l b)
  (H_pg_area : area_of_playground pg_area)
  (H_pg_fraction : playground_fraction A b)
  (H_pg_landscape_fraction : fraction_of_landscape pg_area A) :
  l = 240 :=
by
  sorry

end landscape_length_l1149_114909


namespace total_difference_in_cents_l1149_114998

variable (q : ℕ)

def charles_quarters := 6 * q + 2
def charles_dimes := 3 * q - 2

def richard_quarters := 2 * q + 10
def richard_dimes := 4 * q + 3

def cents_from_quarters (n : ℕ) : ℕ := 25 * n
def cents_from_dimes (n : ℕ) : ℕ := 10 * n

theorem total_difference_in_cents : 
  (cents_from_quarters (charles_quarters q) + cents_from_dimes (charles_dimes q)) - 
  (cents_from_quarters (richard_quarters q) + cents_from_dimes (richard_dimes q)) = 
  90 * q - 250 :=
by
  sorry

end total_difference_in_cents_l1149_114998


namespace Cody_book_series_total_count_l1149_114979

theorem Cody_book_series_total_count :
  ∀ (weeks: ℕ) (books_first_week: ℕ) (books_second_week: ℕ) (books_per_week_after: ℕ),
    weeks = 7 ∧ books_first_week = 6 ∧ books_second_week = 3 ∧ books_per_week_after = 9 →
    (books_first_week + books_second_week + (weeks - 2) * books_per_week_after) = 54 :=
by
  sorry

end Cody_book_series_total_count_l1149_114979


namespace find_n_l1149_114937

theorem find_n (x : ℝ) (hx : x > 0) (h : x / n + x / 25 = 0.24000000000000004 * x) : n = 5 :=
sorry

end find_n_l1149_114937


namespace rectangle_length_l1149_114903

theorem rectangle_length (s w : ℝ) (A : ℝ) (L : ℝ) (h1 : s = 9) (h2 : w = 3) (h3 : A = s * s) (h4 : A = w * L) : L = 27 :=
by
  sorry

end rectangle_length_l1149_114903


namespace solve_quadratic_equation_l1149_114948

theorem solve_quadratic_equation : ∀ x : ℝ, x * (x - 14) = 0 ↔ x = 0 ∨ x = 14 :=
by
  sorry

end solve_quadratic_equation_l1149_114948


namespace contrapositive_example_l1149_114955

theorem contrapositive_example (x : ℝ) : (x > 1 → x^2 > 1) → (x^2 ≤ 1 → x ≤ 1) :=
sorry

end contrapositive_example_l1149_114955


namespace laura_pants_count_l1149_114944

def cost_of_pants : ℕ := 54
def cost_of_shirt : ℕ := 33
def number_of_shirts : ℕ := 4
def total_money_given : ℕ := 250
def change_received : ℕ := 10

def laura_spent : ℕ := total_money_given - change_received
def total_cost_shirts : ℕ := cost_of_shirt * number_of_shirts
def spent_on_pants : ℕ := laura_spent - total_cost_shirts
def pairs_of_pants_bought : ℕ := spent_on_pants / cost_of_pants

theorem laura_pants_count : pairs_of_pants_bought = 2 :=
by
  sorry

end laura_pants_count_l1149_114944


namespace find_Sn_find_Tn_l1149_114994

def Sn (n : ℕ) : ℕ := n^2 + n

def Tn (n : ℕ) : ℚ := (n : ℚ) / (n + 1)

section
variables {a₁ d : ℕ}

-- Given conditions
axiom S5 : 5 * a₁ + 10 * d = 30
axiom S10 : 10 * a₁ + 45 * d = 110

-- Problem statement 1
theorem find_Sn (n : ℕ) : Sn n = n^2 + n :=
sorry

-- Problem statement 2
theorem find_Tn (n : ℕ) : Tn n = (n : ℚ) / (n + 1) :=
sorry

end

end find_Sn_find_Tn_l1149_114994


namespace gear_ratios_l1149_114918

variable (x y z w : ℝ)
variable (ω_A ω_B ω_C ω_D : ℝ)
variable (k : ℝ)

theorem gear_ratios (h : x * ω_A = y * ω_B ∧ y * ω_B = z * ω_C ∧ z * ω_C = w * ω_D) : 
    ω_A/ω_B = yzw/xzw ∧ ω_B/ω_C = xzw/xyw ∧ ω_C/ω_D = xyw/xyz ∧ ω_A/ω_C = yzw/xyw := 
sorry

end gear_ratios_l1149_114918


namespace fill_buckets_lcm_l1149_114920

theorem fill_buckets_lcm :
  (∀ (A B C : ℕ), (2 / 3 : ℚ) * A = 90 ∧ (1 / 2 : ℚ) * B = 120 ∧ (3 / 4 : ℚ) * C = 150 → lcm A (lcm B C) = 1200) :=
by
  sorry

end fill_buckets_lcm_l1149_114920


namespace total_cartons_packed_l1149_114965

-- Define the given conditions
def cans_per_carton : ℕ := 20
def cartons_loaded : ℕ := 40
def cans_left : ℕ := 200

-- Formalize the proof problem
theorem total_cartons_packed : cartons_loaded + (cans_left / cans_per_carton) = 50 := by
  sorry

end total_cartons_packed_l1149_114965


namespace total_homework_pages_l1149_114966

theorem total_homework_pages (R : ℕ) (H1 : R + 3 = 8) : R + (R + 3) = 13 :=
by sorry

end total_homework_pages_l1149_114966


namespace geometric_sequence_value_l1149_114919

theorem geometric_sequence_value (a : ℝ) (h₁ : 280 ≠ 0) (h₂ : 35 ≠ 0) : 
  (∃ r : ℝ, 280 * r = a ∧ a * r = 35 / 8 ∧ a > 0) → a = 35 :=
by {
  sorry
}

end geometric_sequence_value_l1149_114919


namespace value_of_a_plus_d_l1149_114993

variable (a b c d : ℝ)

theorem value_of_a_plus_d
  (h1 : a + b = 4)
  (h2 : b + c = 5)
  (h3 : c + d = 3) :
  a + d = 1 :=
by
sorry

end value_of_a_plus_d_l1149_114993


namespace initial_juggling_objects_l1149_114991

theorem initial_juggling_objects (x : ℕ) : (∀ i : ℕ, i = 5 → x + 2*i = 13) → x = 3 :=
by 
  intro h
  sorry

end initial_juggling_objects_l1149_114991


namespace probability_of_at_least_one_black_ball_l1149_114962

noncomputable def probability_at_least_one_black_ball := 
  let total_outcomes := Nat.choose 4 2
  let favorable_outcomes := (Nat.choose 2 1) * (Nat.choose 2 1) + (Nat.choose 2 2)
  favorable_outcomes / total_outcomes

theorem probability_of_at_least_one_black_ball :
  probability_at_least_one_black_ball = 5 / 6 :=
by
  sorry

end probability_of_at_least_one_black_ball_l1149_114962


namespace seokjin_rank_l1149_114912

-- Define the ranks and the people between them as given conditions in the problem
def jimin_rank : Nat := 4
def people_between : Nat := 19

-- The goal is to prove that Seokjin's rank is 24
theorem seokjin_rank : jimin_rank + people_between + 1 = 24 := 
by
  sorry

end seokjin_rank_l1149_114912


namespace determine_digits_l1149_114945

def product_eq_digits (A B C D x : ℕ) : Prop :=
  x * (x + 1) = 1000 * A + 100 * B + 10 * C + D

def product_minus_3_eq_digits (A B C D x : ℕ) : Prop :=
  (x - 3) * (x - 2) = 1000 * C + 100 * A + 10 * B + D

def product_minus_30_eq_digits (A B C D x : ℕ) : Prop :=
  (x - 30) * (x - 29) = 1000 * B + 100 * C + 10 * A + D

theorem determine_digits :
  ∃ (A B C D x : ℕ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  product_eq_digits A B C D x ∧
  product_minus_3_eq_digits A B C D x ∧
  product_minus_30_eq_digits A B C D x ∧
  A = 8 ∧ B = 3 ∧ C = 7 ∧ D = 2 :=
by
  sorry

end determine_digits_l1149_114945


namespace ratio_area_perimeter_eq_sqrt3_l1149_114907

theorem ratio_area_perimeter_eq_sqrt3 :
  let side_length := 12
  let altitude := side_length * (Real.sqrt 3) / 2
  let area := (1 / 2) * side_length * altitude
  let perimeter := 3 * side_length
  let ratio := area / perimeter
  ratio = Real.sqrt 3 := 
by
  sorry

end ratio_area_perimeter_eq_sqrt3_l1149_114907


namespace options_equal_results_l1149_114990

theorem options_equal_results :
  (4^3 ≠ 3^4) ∧
  ((-5)^3 = (-5^3)) ∧
  ((-6)^2 ≠ -6^2) ∧
  ((- (5/2))^2 ≠ (- (2/5))^2) :=
by {
  sorry
}

end options_equal_results_l1149_114990


namespace garden_area_l1149_114924

theorem garden_area (posts : Nat) (distance : Nat) (n_corners : Nat) (a b : Nat)
  (h_posts : posts = 20)
  (h_distance : distance = 4)
  (h_corners : n_corners = 4)
  (h_total_posts : 2 * (a + b) = posts)
  (h_side_relation : b + 1 = 2 * (a + 1)) :
  (distance * (a + 1 - 1)) * (distance * (b + 1 - 1)) = 336 := 
by 
  sorry

end garden_area_l1149_114924


namespace direct_proportion_function_l1149_114901

-- Definitions of the functions
def fA (x : ℝ) : ℝ := 3 * x - 4
def fB (x : ℝ) : ℝ := -2 * x + 1
def fC (x : ℝ) : ℝ := 3 * x
def fD (x : ℝ) : ℝ := 3 * x^2 + 2

-- Definition of a direct proportion function
def is_direct_proportion (f : ℝ → ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ (∀ x : ℝ, f x = k * x)

-- Theorem statement
theorem direct_proportion_function : is_direct_proportion fC ∧ ¬ is_direct_proportion fA ∧ ¬ is_direct_proportion fB ∧ ¬ is_direct_proportion fD :=
by
  sorry

end direct_proportion_function_l1149_114901


namespace ratio_of_sequences_is_5_over_4_l1149_114947

-- Definitions of arithmetic sequences
def arithmetic_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Hypotheses
def sequence_1_sum : ℕ :=
  arithmetic_sum 5 5 16

def sequence_2_sum : ℕ :=
  arithmetic_sum 4 4 16

-- Main statement to be proven
theorem ratio_of_sequences_is_5_over_4 : sequence_1_sum / sequence_2_sum = 5 / 4 := sorry

end ratio_of_sequences_is_5_over_4_l1149_114947


namespace intersect_sets_l1149_114985

def M : Set ℝ := { x | x ≥ -1 }
def N : Set ℝ := { x | -2 < x ∧ x < 2 }

theorem intersect_sets :
  M ∩ N = { x | -1 ≤ x ∧ x < 2 } := by
  sorry

end intersect_sets_l1149_114985


namespace positive_integers_of_m_n_l1149_114936

theorem positive_integers_of_m_n (m n : ℕ) (p : ℕ) (a : ℕ) (k : ℕ) (h_m_ge_2 : m ≥ 2) (h_n_ge_2 : n ≥ 2) 
  (h_prime_q : Prime (m + 1)) (h_4k_1 : m + 1 = 4 * k - 1) 
  (h_eq : (m ^ (2 ^ n - 1) - 1) / (m - 1) = m ^ n + p ^ a) : 
  (m, n) = (p - 1, 2) ∧ Prime p ∧ ∃k, p = 4 * k - 1 := 
by {
  sorry
}

end positive_integers_of_m_n_l1149_114936


namespace no_generating_combination_l1149_114921

-- Representing Rubik's Cube state as a type (assume a type exists)
axiom CubeState : Type

-- A combination of turns represented as a function on states
axiom A : CubeState → CubeState

-- Simple rotations
axiom P : CubeState → CubeState
axiom Q : CubeState → CubeState

-- Rubik's Cube property of generating combination (assuming generating implies all states achievable)
def is_generating (A : CubeState → CubeState) :=
  ∀ X : CubeState, ∃ m n : ℕ, P X = A^[m] X ∧ Q X = A^[n] X

-- Non-commutativity condition
axiom non_commutativity : ∀ X : CubeState, P (Q X) ≠ Q (P X)

-- Formal statement of the problem
theorem no_generating_combination : ¬ ∃ A : CubeState → CubeState, is_generating A :=
by sorry

end no_generating_combination_l1149_114921


namespace problem_proof_l1149_114981

noncomputable def f : ℝ → ℝ := sorry

theorem problem_proof (h1 : ∀ x : ℝ, f (-x) = f x)
    (h2 : ∀ x y : ℝ, x < y ∧ y ≤ -1 → f x < f y) : 
    f 2 < f (-3 / 2) ∧ f (-3 / 2) < f (-1) :=
by
  sorry

end problem_proof_l1149_114981


namespace find_multiple_of_diff_l1149_114963

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

end find_multiple_of_diff_l1149_114963


namespace first_present_cost_is_18_l1149_114975

-- Conditions as definitions
variables (x : ℕ)

-- Given conditions
def first_present_cost := x
def second_present_cost := x + 7
def third_present_cost := x - 11
def total_cost := first_present_cost x + second_present_cost x + third_present_cost x

-- Statement of the problem
theorem first_present_cost_is_18 (h : total_cost x = 50) : x = 18 :=
by {
  sorry  -- Proof omitted
}

end first_present_cost_is_18_l1149_114975


namespace cost_to_paint_cube_l1149_114929

theorem cost_to_paint_cube :
  let cost_per_kg := 50
  let coverage_per_kg := 20
  let side_length := 20
  let surface_area := 6 * (side_length * side_length)
  let amount_of_paint := surface_area / coverage_per_kg
  let total_cost := amount_of_paint * cost_per_kg
  total_cost = 6000 :=
by
  sorry

end cost_to_paint_cube_l1149_114929


namespace inequality_holds_l1149_114969

theorem inequality_holds (k : ℝ) : (∀ x : ℝ, x^2 + k * x + 1 > 0) ↔ (k > -2 ∧ k < 2) :=
by
  sorry

end inequality_holds_l1149_114969


namespace smallest_positive_multiple_of_45_divisible_by_3_l1149_114950

theorem smallest_positive_multiple_of_45_divisible_by_3 
  (x : ℕ) (hx: x > 0) : ∃ y : ℕ, y = 45 ∧ 45 ∣ y ∧ 3 ∣ y ∧ ∀ z : ℕ, (45 ∣ z ∧ 3 ∣ z ∧ z > 0) → z ≥ y :=
by
  sorry

end smallest_positive_multiple_of_45_divisible_by_3_l1149_114950


namespace amazing_rectangle_area_unique_l1149_114954

def isAmazingRectangle (a b : ℕ) : Prop :=
  a = 2 * b ∧ a * b = 3 * (2 * (a + b))

theorem amazing_rectangle_area_unique :
  ∃ (a b : ℕ), isAmazingRectangle a b ∧ a * b = 162 :=
by
  sorry

end amazing_rectangle_area_unique_l1149_114954


namespace female_democrats_l1149_114959

theorem female_democrats (F M : ℕ) (h1 : F + M = 840) (h2 : F / 2 + M / 4 = 280) : F / 2 = 140 :=
by 
  sorry

end female_democrats_l1149_114959


namespace set_nonempty_iff_nonneg_l1149_114987

theorem set_nonempty_iff_nonneg (a : ℝ) :
  (∃ x : ℝ, x^2 ≤ a) ↔ a ≥ 0 :=
sorry

end set_nonempty_iff_nonneg_l1149_114987


namespace trees_planted_l1149_114956

theorem trees_planted (interval trail_length : ℕ) (h1 : interval = 30) (h2 : trail_length = 1200) : 
  trail_length / interval = 40 :=
by
  sorry

end trees_planted_l1149_114956


namespace union_of_sets_l1149_114910

def M := {x : ℝ | -1 < x ∧ x < 1}
def N := {x : ℝ | x^2 - 3 * x ≤ 0}

theorem union_of_sets : M ∪ N = {x : ℝ | -1 < x ∧ x ≤ 3} :=
by sorry

end union_of_sets_l1149_114910


namespace integer_a_satisfies_equation_l1149_114984

theorem integer_a_satisfies_equation (a b c : ℤ) :
  (∃ b c : ℤ, (x - a) * (x - 5) + 2 = (x + b) * (x + c)) → 
    a = 2 :=
by
  intro h_eq
  -- Proof goes here
  sorry

end integer_a_satisfies_equation_l1149_114984


namespace bobby_initial_candy_l1149_114961

theorem bobby_initial_candy (candy_ate_start candy_ate_more candy_left : ℕ)
  (h1 : candy_ate_start = 9) (h2 : candy_ate_more = 5) (h3 : candy_left = 8) :
  candy_ate_start + candy_ate_more + candy_left = 22 :=
by
  rw [h1, h2, h3]
  -- sorry


end bobby_initial_candy_l1149_114961


namespace certain_number_divisibility_l1149_114992

-- Define the conditions and the main problem statement
theorem certain_number_divisibility (n : ℕ) (h1 : 0 < n) (h2 : n < 11) (h3 : (18888 - n) % k = 0) (h4 : n = 1) : k = 11 :=
by
  sorry

end certain_number_divisibility_l1149_114992


namespace inscribed_sphere_radius_l1149_114902

theorem inscribed_sphere_radius {V S1 S2 S3 S4 R : ℝ} :
  (1/3) * R * (S1 + S2 + S3 + S4) = V → 
  R = 3 * V / (S1 + S2 + S3 + S4) :=
by
  intro h
  sorry

end inscribed_sphere_radius_l1149_114902


namespace find_digits_l1149_114977

-- Definitions, conditions and statement of the problem
def satisfies_condition (z : ℕ) (k : ℕ) (n : ℕ) : Prop :=
  n ≥ 1 ∧ (n^9 % 10^k) / 10^(k - 1) = z

theorem find_digits (z : ℕ) (k : ℕ) :
  k ≥ 1 →
  (z = 0 ∨ z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) →
  ∃ n, satisfies_condition z k n := 
sorry

end find_digits_l1149_114977


namespace units_digit_of_150_factorial_is_zero_l1149_114931

-- Define the conditions for the problem
def is_units_digit_zero_of_factorial (n : ℕ) : Prop :=
  n = 150 → (Nat.factorial n % 10 = 0)

-- The statement of the proof problem
theorem units_digit_of_150_factorial_is_zero : is_units_digit_zero_of_factorial 150 :=
  sorry

end units_digit_of_150_factorial_is_zero_l1149_114931


namespace projectile_height_time_l1149_114939

-- Define constants and the height function
def a : ℝ := -4.9
def b : ℝ := 29.75
def c : ℝ := -35
def y (t : ℝ) : ℝ := a * t^2 + b * t

-- Problem statement
theorem projectile_height_time (h : y t = 35) : ∃ t : ℝ, 0 < t ∧ abs (t - 1.598) < 0.001 := by
  -- Placeholder for actual proof
  sorry

end projectile_height_time_l1149_114939


namespace polar_line_equation_l1149_114942

theorem polar_line_equation (r θ: ℝ) (p : r = 3 ∧ θ = 0) : r = 3 := 
by 
  sorry

end polar_line_equation_l1149_114942


namespace ellipse_range_l1149_114968

theorem ellipse_range (t : ℝ) (x y : ℝ) :
  (10 - t > 0) → (t - 4 > 0) → (10 - t ≠ t - 4) →
  (t ∈ (Set.Ioo 4 7 ∪ Set.Ioo 7 10)) :=
by
  intros h1 h2 h3
  sorry

end ellipse_range_l1149_114968


namespace scarlet_savings_l1149_114997

noncomputable def remaining_savings (initial_savings earrings_cost necklace_cost bracelet_cost jewelry_set_cost jewelry_set_discount sales_tax_percentage : ℝ) : ℝ :=
  let total_item_cost := earrings_cost + necklace_cost + bracelet_cost
  let discounted_jewelry_set_cost := jewelry_set_cost * (1 - jewelry_set_discount / 100)
  let total_cost_before_tax := total_item_cost + discounted_jewelry_set_cost
  let total_sales_tax := total_cost_before_tax * (sales_tax_percentage / 100)
  let final_total_cost := total_cost_before_tax + total_sales_tax
  initial_savings - final_total_cost

theorem scarlet_savings : remaining_savings 200 23 48 35 80 25 5 = 25.70 :=
by
  sorry

end scarlet_savings_l1149_114997


namespace solution_l1149_114988

variable (f g : ℝ → ℝ)

open Real

-- Define f(x) and g(x) as given in the problem
def isSolution (x : ℝ) : Prop :=
  f x + g x = sqrt ((1 + cos (2 * x)) / (1 - sin x)) ∧
  (∀ x, f (-x) = -f x) ∧
  (∀ x, g (-x) = g x)

-- The theorem we want to prove
theorem solution (x : ℝ) (hx : -π / 2 < x ∧ x < π / 2)
  (h : isSolution f g x) : (f x)^2 - (g x)^2 = -2 * cos x := 
sorry

end solution_l1149_114988


namespace number_of_real_solutions_l1149_114905

noncomputable def f (x : ℝ) : ℝ := 2^(-x) + x^2 - 3

theorem number_of_real_solutions :
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧
  (∀ x : ℝ, f x = 0 → (x = x₁ ∨ x = x₂)) :=
by
  sorry

end number_of_real_solutions_l1149_114905


namespace first_week_tickets_calc_l1149_114941

def total_tickets : ℕ := 90
def second_week_tickets : ℕ := 17
def tickets_left : ℕ := 35

theorem first_week_tickets_calc : total_tickets - (second_week_tickets + tickets_left) = 38 := by
  sorry

end first_week_tickets_calc_l1149_114941


namespace inequality_proof_l1149_114976

theorem inequality_proof (a b c d : ℝ) : 
  (a^2 + b^2 + 1) * (c^2 + d^2 + 1) ≥ 2 * (a + c) * (b + d) :=
by sorry

end inequality_proof_l1149_114976


namespace total_blocks_l1149_114938

-- Conditions
def original_blocks : ℝ := 35.0
def added_blocks : ℝ := 65.0

-- Question and proof goal
theorem total_blocks : original_blocks + added_blocks = 100.0 := 
by
  -- The proof would be provided here
  sorry

end total_blocks_l1149_114938


namespace smallest_x_solution_l1149_114953

theorem smallest_x_solution :
  (∃ x : ℝ, (3 * x^2 + 36 * x - 90 = 2 * x * (x + 16)) ∧ ∀ y : ℝ, (3 * y^2 + 36 * y - 90 = 2 * y * (y + 16)) → x ≤ y) ↔ x = -10 :=
by
  sorry

end smallest_x_solution_l1149_114953


namespace min_value_expr_l1149_114934

theorem min_value_expr (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : x + 2 * y = 5) :
  (1 / (x - 1) + 1 / (y - 1)) = (3 / 2 + Real.sqrt 2) :=
sorry

end min_value_expr_l1149_114934


namespace fraction_problem_l1149_114982

theorem fraction_problem (a b c d e: ℚ) (val: ℚ) (h_a: a = 1/4) (h_b: b = 1/3) 
  (h_c: c = 1/6) (h_d: d = 1/8) (h_val: val = 72) :
  (a * b * c * val + d) = 9 / 8 :=
by {
  sorry
}

end fraction_problem_l1149_114982


namespace number_thought_of_eq_95_l1149_114908

theorem number_thought_of_eq_95 (x : ℝ) (h : (x / 5) + 23 = 42) : x = 95 := 
by
  sorry

end number_thought_of_eq_95_l1149_114908


namespace right_triangle_width_l1149_114915

theorem right_triangle_width (height : ℝ) (side_square : ℝ) (width : ℝ) (n_triangles : ℕ) 
  (triangle_right : height = 2)
  (fit_inside_square : side_square = 2)
  (number_triangles : n_triangles = 2) :
  width = 2 :=
sorry

end right_triangle_width_l1149_114915


namespace total_games_won_l1149_114971

theorem total_games_won 
  (bulls_games : ℕ) (heat_games : ℕ) (knicks_games : ℕ)
  (bulls_condition : bulls_games = 70)
  (heat_condition : heat_games = bulls_games + 5)
  (knicks_condition : knicks_games = 2 * heat_games) :
  bulls_games + heat_games + knicks_games = 295 :=
by
  sorry

end total_games_won_l1149_114971


namespace at_least_one_inequality_false_l1149_114958

open Classical

theorem at_least_one_inequality_false (a b c d : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) :
  ¬ (a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) :=
by
  sorry

end at_least_one_inequality_false_l1149_114958


namespace points_satisfy_equation_l1149_114989

theorem points_satisfy_equation :
  ∀ (x y : ℝ), x^2 - y^4 = Real.sqrt (18 * x - x^2 - 81) ↔ 
               (x = 9 ∧ y = Real.sqrt 3) ∨ (x = 9 ∧ y = -Real.sqrt 3) := 
by 
  intros x y 
  sorry

end points_satisfy_equation_l1149_114989


namespace determine_value_of_x_l1149_114933

theorem determine_value_of_x {b x : ℝ} (hb : 1 < b) (hx : 0 < x) 
  (h_eq : (4 * x)^(Real.logb b 2) = (5 * x)^(Real.logb b 3)) : 
  x = (4 / 5)^(Real.logb (3 / 2) b) :=
by
  sorry

end determine_value_of_x_l1149_114933


namespace fraction_power_rule_example_l1149_114904

theorem fraction_power_rule_example : (5 / 6)^4 = 625 / 1296 :=
by
  sorry

end fraction_power_rule_example_l1149_114904


namespace find_x_l1149_114972

variables (a b c d x : ℤ)

theorem find_x (h1 : a - b = c + d + 9) (h2 : a - c = 3) (h3 : a + b = c - d - x) : x = 3 :=
sorry

end find_x_l1149_114972


namespace find_some_number_l1149_114999

theorem find_some_number (x some_number : ℝ) (h1 : (27 / 4) * x - some_number = 3 * x + 27) (h2 : x = 12) :
  some_number = 18 :=
by
  sorry

end find_some_number_l1149_114999


namespace stage_order_permutations_l1149_114930

-- Define the problem in Lean terms
def permutations (n : ℕ) : ℕ := Nat.factorial n

theorem stage_order_permutations :
  let total_students := 6
  let predetermined_students := 3
  (permutations total_students) / (permutations predetermined_students) = 120 := by
  sorry

end stage_order_permutations_l1149_114930


namespace last_four_digits_5_to_2019_l1149_114952

theorem last_four_digits_5_to_2019 :
  ∃ (x : ℕ), (5^2019) % 10000 = x ∧ x = 8125 :=
by
  sorry

end last_four_digits_5_to_2019_l1149_114952


namespace zacharys_bus_ride_length_l1149_114943

theorem zacharys_bus_ride_length (Vince Zachary : ℝ) (hV : Vince = 0.62) (hDiff : Vince = Zachary + 0.13) : Zachary = 0.49 :=
by
  sorry

end zacharys_bus_ride_length_l1149_114943


namespace quadratic_roots_l1149_114980

theorem quadratic_roots (m x1 x2 : ℝ) (h1 : x1 + x2 = 1) (h2 : x1*x1 + m*x1 + 2*m = 0) (h3 : x2*x2 + m*x2 + 2*m = 0) : x1 * x2 = -2 := 
by sorry

end quadratic_roots_l1149_114980


namespace factor_is_three_l1149_114925

theorem factor_is_three (x f : ℝ) (h1 : 2 * x + 5 = y) (h2 : f * y = 111) (h3 : x = 16):
  f = 3 :=
by
  sorry

end factor_is_three_l1149_114925


namespace conclusion_A_conclusion_B_conclusion_C1_conclusion_C2_l1149_114995

variable {r a b x1 y1 x2 y2 : ℝ} -- variables used in the problem

-- conditions
def circle1 : x1^2 + y1^2 = r^2 := sorry -- Circle C1 equation
def circle2 : (x1 + a)^2 + (y1 + b)^2 = r^2 := sorry -- Circle C2 equation
def r_positive : r > 0 := sorry -- r > 0
def not_both_zero : ¬ (a = 0 ∧ b = 0) := sorry -- a, b are not both zero
def distinct_points : x1 ≠ x2 ∧ y1 ≠ y2 := sorry -- A(x1, y1) and B(x2, y2) are distinct

-- Proofs to be provided for each of the conclusions
theorem conclusion_A : 2 * a * x1 + 2 * b * y1 + a^2 + b^2 = 0 := sorry
theorem conclusion_B : a * (x1 - x2) + b * (y1 - y2) = 0 := sorry
theorem conclusion_C1 : x1 + x2 = -a := sorry
theorem conclusion_C2 : y1 + y2 = -b := sorry

end conclusion_A_conclusion_B_conclusion_C1_conclusion_C2_l1149_114995


namespace tangent_slope_at_one_one_l1149_114949

noncomputable def curve (x : ℝ) : ℝ := x * Real.exp (x - 1)

theorem tangent_slope_at_one_one : (deriv curve 1) = 2 := 
sorry

end tangent_slope_at_one_one_l1149_114949


namespace sum_of_consecutive_integers_l1149_114900

theorem sum_of_consecutive_integers (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c = 14) : a + b + c = 39 := 
by 
  sorry

end sum_of_consecutive_integers_l1149_114900


namespace minimum_red_vertices_l1149_114967

theorem minimum_red_vertices (n : ℕ) (h : 0 < n) :
  ∃ R : ℕ, (∀ i j : ℕ, i < n ∧ j < n →
    (i + j) % 2 = 0 → true) ∧
    R = Int.ceil (n^2 / 2 : ℝ) :=
sorry

end minimum_red_vertices_l1149_114967


namespace jerry_pool_depth_l1149_114922

theorem jerry_pool_depth :
  ∀ (total_gallons : ℝ) (gallons_drinking_cooking : ℝ) (gallons_per_shower : ℝ)
    (number_of_showers : ℝ) (pool_length : ℝ) (pool_width : ℝ)
    (gallons_per_cubic_foot : ℝ),
    total_gallons = 1000 →
    gallons_drinking_cooking = 100 →
    gallons_per_shower = 20 →
    number_of_showers = 15 →
    pool_length = 10 →
    pool_width = 10 →
    gallons_per_cubic_foot = 1 →
    (total_gallons - (gallons_drinking_cooking + gallons_per_shower * number_of_showers)) / 
    (pool_length * pool_width) = 6 := 
by
  intros total_gallons gallons_drinking_cooking gallons_per_shower number_of_showers pool_length pool_width gallons_per_cubic_foot
  intros total_gallons_eq drinking_cooking_eq shower_eq showers_eq length_eq width_eq cubic_foot_eq
  sorry

end jerry_pool_depth_l1149_114922


namespace part1_part2_l1149_114983

variable (a : ℝ)

-- Defining the set A
def setA (a : ℝ) : Set ℝ := { x : ℝ | (x - 2) * (x - 3 * a - 1) < 0 }

-- Part 1: For a = 2, setB should be {x | 2 < x < 7}
theorem part1 : setA 2 = { x : ℝ | 2 < x ∧ x < 7 } :=
by
  sorry

-- Part 2: If setA a = setB, then a = -1
theorem part2 (B : Set ℝ) (h : setA a = B) : a = -1 :=
by
  sorry

end part1_part2_l1149_114983


namespace candy_store_revenue_l1149_114923

/-- A candy store sold 20 pounds of fudge for $2.50 per pound,
    5 dozen chocolate truffles for $1.50 each, 
    and 3 dozen chocolate-covered pretzels at $2.00 each.
    Prove that the total money made by the candy store is $212.00. --/
theorem candy_store_revenue :
  let fudge_pounds := 20
  let fudge_price_per_pound := 2.50
  let truffle_dozen := 5
  let truffle_price_each := 1.50
  let pretzel_dozen := 3
  let pretzel_price_each := 2.00
  (fudge_pounds * fudge_price_per_pound) + 
  (truffle_dozen * 12 * truffle_price_each) + 
  (pretzel_dozen * 12 * pretzel_price_each) = 212 :=
by
  sorry

end candy_store_revenue_l1149_114923


namespace probability_scrapped_l1149_114928

variable (P_A P_B_given_not_A : ℝ)
variable (prob_scrapped : ℝ)

def fail_first_inspection (P_A : ℝ) := 1 - P_A
def fail_second_inspection_given_fails_first (P_B_given_not_A : ℝ) := 1 - P_B_given_not_A

theorem probability_scrapped (h1 : P_A = 0.8) (h2 : P_B_given_not_A = 0.9) (h3 : prob_scrapped = fail_first_inspection P_A * fail_second_inspection_given_fails_first P_B_given_not_A) :
  prob_scrapped = 0.02 := by
  sorry

end probability_scrapped_l1149_114928


namespace complement_U_P_l1149_114935

def U (y : ℝ) : Prop := y > 0
def P (y : ℝ) : Prop := 0 < y ∧ y < 1/3

theorem complement_U_P :
  {y : ℝ | U y} \ {y : ℝ | P y} = {y : ℝ | y ≥ 1/3} :=
by
  sorry

end complement_U_P_l1149_114935


namespace pokemon_cards_per_friend_l1149_114960

theorem pokemon_cards_per_friend (total_cards : ℕ) (num_friends : ℕ) 
  (hc : total_cards = 56) (hf : num_friends = 4) : (total_cards / num_friends) = 14 := 
by
  sorry

end pokemon_cards_per_friend_l1149_114960


namespace shooter_variance_l1149_114916

def scores : List ℝ := [9.7, 9.9, 10.1, 10.2, 10.1] -- Defining the scores

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length -- Calculating the mean

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length -- Defining the variance

theorem shooter_variance :
  variance scores = 0.032 :=
by
  sorry -- Proof to be provided later

end shooter_variance_l1149_114916


namespace mixed_number_evaluation_l1149_114957

theorem mixed_number_evaluation :
  let a := (4 + 1 / 3 : ℚ)
  let b := (3 + 2 / 7 : ℚ)
  let c := (2 + 5 / 6 : ℚ)
  let d := (1 + 1 / 2 : ℚ)
  let e := (5 + 1 / 4 : ℚ)
  let f := (3 + 2 / 5 : ℚ)
  (a + b - c) * (d + e) / f = 9 + 198 / 317 :=
by {
  let a : ℚ := 4 + 1 / 3
  let b : ℚ := 3 + 2 / 7
  let c : ℚ := 2 + 5 / 6
  let d : ℚ := 1 + 1 / 2
  let e : ℚ := 5 + 1 / 4
  let f : ℚ := 3 + 2 / 5
  sorry
}

end mixed_number_evaluation_l1149_114957


namespace nutty_professor_mixture_weight_l1149_114951

/-- The Nutty Professor's problem translated to Lean 4 -/
theorem nutty_professor_mixture_weight :
  let cashews_weight := 20
  let cashews_cost_per_pound := 6.75
  let brazil_nuts_cost_per_pound := 5.00
  let mixture_cost_per_pound := 5.70
  ∃ (brazil_nuts_weight : ℝ), cashews_weight * cashews_cost_per_pound + brazil_nuts_weight * brazil_nuts_cost_per_pound =
                             (cashews_weight + brazil_nuts_weight) * mixture_cost_per_pound ∧
                             (cashews_weight + brazil_nuts_weight = 50) := 
sorry

end nutty_professor_mixture_weight_l1149_114951


namespace modular_inverse_calculation_l1149_114978

theorem modular_inverse_calculation : 
  (3 * (49 : ℤ) + 12 * (40 : ℤ)) % 65 = 42 := 
by
  sorry

end modular_inverse_calculation_l1149_114978


namespace constant_term_l1149_114911

theorem constant_term (n : ℕ) (h : (Nat.choose n 4 * 2^4) / (Nat.choose n 2 * 2^2) = (56 / 3)) :
  (∃ k : ℕ, k = 2 ∧ n = 10 ∧ Nat.choose 10 k * 2^k = 180) := by
  sorry

end constant_term_l1149_114911


namespace checkerboard_no_identical_numbers_l1149_114964

theorem checkerboard_no_identical_numbers :
  ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 11 ∧ 1 ≤ j ∧ j ≤ 19 → 19 * (i - 1) + j = 11 * (j - 1) + i → false :=
by
  sorry

end checkerboard_no_identical_numbers_l1149_114964


namespace point_in_plane_region_l1149_114906

-- Defining the condition that the inequality represents a region on the plane
def plane_region (x y : ℝ) : Prop := x + 2 * y - 1 > 0

-- Stating that the point (0, 1) lies within the plane region represented by the inequality
theorem point_in_plane_region : plane_region 0 1 :=
by {
    sorry
}

end point_in_plane_region_l1149_114906


namespace Ellen_strawberries_used_l1149_114996

theorem Ellen_strawberries_used :
  let yogurt := 0.1
  let orange_juice := 0.2
  let total_ingredients := 0.5
  let strawberries := total_ingredients - (yogurt + orange_juice)
  strawberries = 0.2 :=
by
  sorry

end Ellen_strawberries_used_l1149_114996


namespace upper_bound_neg_expr_l1149_114926

theorem upper_bound_neg_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  - (1 / (2 * a) + 2 / b) ≤ - (9 / 2) := 
sorry

end upper_bound_neg_expr_l1149_114926


namespace area_of_quadrilateral_is_195_l1149_114986

-- Definitions and conditions
def diagonal_length : ℝ := 26
def offset1 : ℝ := 9
def offset2 : ℝ := 6

-- Prove the area of the quadrilateral is 195 cm²
theorem area_of_quadrilateral_is_195 :
  1 / 2 * diagonal_length * offset1 + 1 / 2 * diagonal_length * offset2 = 195 := 
by
  -- The proof steps would go here
  sorry

end area_of_quadrilateral_is_195_l1149_114986
