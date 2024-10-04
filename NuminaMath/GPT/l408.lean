import Mathlib

namespace four_digit_numbers_with_5_or_8_l408_408152

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  n.digits 10 ∋ d

def contains_5_or_8 (n : ℕ) : Prop :=
  contains_digit n 5 ∨ contains_digit n 8

theorem four_digit_numbers_with_5_or_8 : 
  { n : ℕ | is_four_digit n ∧ contains_5_or_8 n }.to_finset.card = 5416 :=
by
  sorry

end four_digit_numbers_with_5_or_8_l408_408152


namespace Lou_receives_lollipops_l408_408225

theorem Lou_receives_lollipops
  (initial_lollipops : ℕ)
  (fraction_to_Emily : ℚ)
  (lollipops_kept : ℕ)
  (lollipops_given_to_Lou : ℕ) :
  initial_lollipops = 42 →
  fraction_to_Emily = 2 / 3 →
  lollipops_kept = 4 →
  lollipops_given_to_Lou = initial_lollipops - (initial_lollipops * fraction_to_Emily).natAbs - lollipops_kept →
  lollipops_given_to_Lou = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end Lou_receives_lollipops_l408_408225


namespace f_of_integral_ratio_l408_408745

variable {f : ℝ → ℝ} (h_cont : ∀ x > 0, continuous_at f x)
variable (h_int : ∀ a b : ℝ, a > 0 → b > 0 → ∃ g : ℝ → ℝ, (∫ x in a..b, f x) = g (b / a))

theorem f_of_integral_ratio :
  (∃ c : ℝ, ∀ x > 0, f x = c / x) :=
sorry

end f_of_integral_ratio_l408_408745


namespace trajectory_of_P_l408_408107

-- Definitions for points and distance
structure Point where
  x : ℝ
  y : ℝ

noncomputable def dist (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- Fixed points F1 and F2
variable (F1 F2 : Point)
-- Distance condition
axiom dist_F1F2 : dist F1 F2 = 8

-- Moving point P satisfying the condition
variable (P : Point)
axiom dist_PF1_PF2 : dist P F1 + dist P F2 = 8

-- Proof goal: P lies on the line segment F1F2
theorem trajectory_of_P : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = ⟨(1 - t) * F1.x + t * F2.x, (1 - t) * F1.y + t * F2.y⟩ :=
  sorry

end trajectory_of_P_l408_408107


namespace point_P1557_is_20_17_l408_408382

/-- Define the function P representing the sequence of points on the spiral -/
def P : ℕ → ℤ × ℤ
| 0     := (0, 0)
| 1     := (1, 0)
| n + 1 :=
  let (x, y) := P n in
  if x = y then -- vertical direction change
    (x + 1, y)
  else if x + y = 0 then -- horizontal direction change
    if x > 0 then (x, y - 1) else (x, y + 1)
  else if x >  0 && x > -y then (x - 1, y)    -- top horizontal row
  else if y > 0 && y > -x then (x, y - 1)    -- right vertical row
  else (-x, -y)

/-- Theorem stating the coordinates of P1557 -/
theorem point_P1557_is_20_17 : P 1557 = (20, 17) := sorry

end point_P1557_is_20_17_l408_408382


namespace inequality_am_gm_l408_408119

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a + b + c) / 3 ≥ Real.cbrt (((a + b) * (b + c) * (c + a)) / 8) ∧
    Real.cbrt (((a + b) * (b + c) * (c + a)) / 8) ≥ (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a)) / 3 :=
by
  sorry

end inequality_am_gm_l408_408119


namespace min_value_expression_l408_408596

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ 64 :=
by
  sorry

end min_value_expression_l408_408596


namespace discount_percentage_l408_408301

theorem discount_percentage
  (number_of_fandoms : ℕ)
  (tshirts_per_fandom : ℕ)
  (price_per_shirt : ℝ)
  (tax_rate : ℝ)
  (total_paid : ℝ)
  (total_expected_price_with_discount_without_tax : ℝ)
  (total_expected_price_without_discount : ℝ)
  (discount_amount : ℝ)
  (discount_percentage : ℝ) :

  number_of_fandoms = 4 ∧
  tshirts_per_fandom = 5 ∧
  price_per_shirt = 15 ∧
  tax_rate = 10 / 100 ∧
  total_paid = 264 ∧
  total_expected_price_with_discount_without_tax = total_paid / (1 + tax_rate) ∧
  total_expected_price_without_discount = number_of_fandoms * tshirts_per_fandom * price_per_shirt ∧
  discount_amount = total_expected_price_without_discount - total_expected_price_with_discount_without_tax ∧
  discount_percentage = (discount_amount / total_expected_price_without_discount) * 100 ->

  discount_percentage = 20 :=
sorry

end discount_percentage_l408_408301


namespace sufficient_but_not_necessary_condition_l408_408555

theorem sufficient_but_not_necessary_condition (b : ℝ) :
  (∀ x : ℝ, b * x^2 - b * x + 1 > 0) ↔ (b = 0 ∨ (0 < b ∧ b < 4)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l408_408555


namespace find_length_of_field_l408_408198

-- Definition of the conditions
def tape_length : ℕ := 250
def tape_remaining : ℕ := 90
def field_width : ℕ := 20
def tape_used : ℕ := tape_length - tape_remaining
def perimeter_field : ℕ := 2 * (field_width + field_length)

-- The theorem we need to prove:
theorem find_length_of_field (h1 : tape_used = 160) : ∃ (field_length : ℕ), 2 * (field_width + field_length) = tape_used ∧ field_length = 60 :=
by 
  use 60
  sorry

end find_length_of_field_l408_408198


namespace probability_odd_card_is_two_fifths_l408_408295

theorem probability_odd_card_is_two_fifths :
  let cards := [2, 3, 4, 5, 6]
  let odd_numbers := [3, 5]
  let total_cards := cards.length
  let favorable_outcomes := odd_numbers.length
  probability (x ∈ odd_numbers | x ∈ cards) = (favorable_outcomes : ℚ) / (total_cards : ℚ) := 
by 
  let cards : List ℕ := [2, 3, 4, 5, 6]
  let odd_numbers : List ℕ := [3, 5] 
  let total_cards : ℕ := cards.length
  let favorable_outcomes : ℕ := odd_numbers.length
  have h : probability (x ∈ odd_numbers | x ∈ cards) = (favorable_outcomes : ℚ) / (total_cards : ℚ) := 
    by
      sorry
  exact h

end probability_odd_card_is_two_fifths_l408_408295


namespace intersection_M_N_l408_408139

-- Define sets M and N
def M := {x : ℝ | x^2 - 2*x ≤ 0}
def N := {x : ℝ | -2 < x ∧ x < 1}

-- The theorem stating the intersection of M and N equals [0, 1)
theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := 
by
  sorry

end intersection_M_N_l408_408139


namespace flavors_needed_this_year_l408_408146

def num_flavors_total : ℕ := 100

def num_flavors_two_years_ago : ℕ := num_flavors_total / 4

def num_flavors_last_year : ℕ := 2 * num_flavors_two_years_ago

def num_flavors_tried_so_far : ℕ := num_flavors_two_years_ago + num_flavors_last_year

theorem flavors_needed_this_year : 
  (num_flavors_total - num_flavors_tried_so_far) = 25 := by {
  sorry
}

end flavors_needed_this_year_l408_408146


namespace quadrilateral_inequality_l408_408096

-- Definitions for our quadrilateral convexity and distances.
variables {A B C D : Type} [ConvexQuadrilateral A B C D]
variables {AB CD BC AD AC BD : ℝ}

-- The main statement to be proved.
theorem quadrilateral_inequality 
  (AB CD BC AD AC BD : ℝ)
  (convex : ConvexQuadrilateral A B C D) :
  AB * CD + BC * AD ≥ AC * BD := 
sorry

end quadrilateral_inequality_l408_408096


namespace problem_statement_l408_408649

noncomputable def sgn (x : ℝ) : ℝ :=
if x < 0 then -1 else if x = 0 then 0 else 1

noncomputable def a : ℝ :=
1 / Real.logb (1 / 4) (1 / 2015) + 1 / Real.logb (1 / 504) (1 / 2015)

def b : ℝ := 2017

theorem problem_statement :
  (a + b + (a - b) * sgn (a - b)) / 2 = 2017 :=
sorry

end problem_statement_l408_408649


namespace arithmetic_sequence_properties_sequences_general_formulas_sum_first_n_terms_l408_408977

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b_n (n : ℕ) : ℕ := 3 ^ n

noncomputable def c_n (n : ℕ) : ℕ := a_n n + b_n n

noncomputable def S_n (n : ℕ) : ℕ := n^2 + (3 * (3^n - 1)) / 2

theorem arithmetic_sequence_properties (n : ℕ) :
  a_n 3 = 5 ∧ a_n 5 - 2 * a_n 2 = 3 ∧ b_n 1 = 3 :=
by {
  split,
  {
    simp [a_n],
    norm_num,
  },
  split,
  {
    simp [a_n],
    norm_num,
  },
  {
    simp [b_n],
    norm_num,
  }
}

theorem sequences_general_formulas (n : ℕ) :
  (a_n n = 2 * n - 1) ∧ (b_n n = 3 ^ n) :=
by {
  split,
  {
    simp [a_n],
  },
  {
    simp [b_n],
  }
}

theorem sum_first_n_terms (n : ℕ) :
  S_n n = n^2 + (3 * (3^n - 1)) / 2 :=
by {
  simp [S_n],
}

end arithmetic_sequence_properties_sequences_general_formulas_sum_first_n_terms_l408_408977


namespace count_whole_numbers_between_cube_roots_l408_408531

theorem count_whole_numbers_between_cube_roots :
  (2 < (20 : ℝ)^(1/3) ∧ (20 : ℝ)^(1/3) < 3) →
  (7 < (500 : ℝ)^(1/3) ∧ (500 : ℝ)^(1/3) < 8) →
  ∀ n : ℕ, nat.card { x : ℕ | (20 : ℝ)^(1/3) < x ∧ x < (500 : ℝ)^(1/3) } = 5 :=
by
  sorry

end count_whole_numbers_between_cube_roots_l408_408531


namespace sum_of_two_numbers_l408_408333

theorem sum_of_two_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 23 :=
by
  sorry

end sum_of_two_numbers_l408_408333


namespace next_number_property_l408_408672

theorem next_number_property (a b c d : ℕ) (h : a = 1 ∧ b = 8):
  ((10 * a + b) * (10 * c + d)) = (n * n) → 
  (1818 < (1000 * a + 100 * b + 10 * c + d)) →
  (∃ abcd, (1000 * a + 100 * b + 10 * c + d) = 1832) :=
by
  sorry

end next_number_property_l408_408672


namespace values_of_n_l408_408536

def has_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def f (n : ℤ) (x : ℝ) : ℝ :=
  Real.cos (n * x) * Real.sin ((4 / n) * x)

theorem values_of_n (n : ℤ) :
  has_period (f n) (3 * Real.pi) ↔ n ∈ ({2, -2, 6, -6} : Set ℤ) :=
sorry

end values_of_n_l408_408536


namespace find_z_and_modulus_l408_408897

noncomputable def z (b : ℝ) : ℂ := 3 + b * complex.I

theorem find_z_and_modulus (b : ℝ) (h : (1 + 3 * complex.I) * (z b) = (1 + 3 * complex.I) * (3 + 1 * complex.I)) :
  z b = 3 + complex.I ∧ 
  let ω := (z b) / (2 + complex.I) in complex.abs ω = real.sqrt 2 :=
by
  sorry

end find_z_and_modulus_l408_408897


namespace equation1_solutions_equation2_solutions_equation3_solutions_equation4_no_real_solutions_l408_408259

theorem equation1_solutions (x : ℝ) : x^2 - 4 * x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) :=
by sorry

theorem equation2_solutions (x : ℝ) : x * (3 * x + 1) = 2 * (3 * x + 1) ↔ (x = -1 / 3 ∨ x = 2) :=
by sorry

theorem equation3_solutions (x : ℝ) : 2 * x^2 + x - 4 = 0 ↔ (x = (-1 + Real.sqrt 33) / 4 ∨ x = (-1 - Real.sqrt 33) / 4) :=
by sorry

theorem equation4_no_real_solutions (x : ℝ) : ¬ ∃ x, 4 * x^2 - 3 * x + 1 = 0 :=
by sorry

end equation1_solutions_equation2_solutions_equation3_solutions_equation4_no_real_solutions_l408_408259


namespace min_value_l408_408488

theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b) : a + 4 * b ≥ 9 :=
sorry

end min_value_l408_408488


namespace bacteria_fill_table_l408_408431

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib n + fib (n+1)

-- We state the theorem that needs to be proved
theorem bacteria_fill_table (m n : ℕ) : 
  let f := fib (2*n + 1) in 
  (2^(n-1)) * (f^(m-1)) = 2^(n-1) * (fib (2*n + 1))^(m-1) := 
sorry

end bacteria_fill_table_l408_408431


namespace next_four_digit_number_l408_408668

def isPerfectSquare (n : ℕ) : Prop :=
∃ m : ℕ, m * m = n

def satisfiesProperty (n : ℕ) : Prop :=
  ∃ a b : ℕ, 10 ≤ a ∧ a < 100 ∧ 0 ≤ b ∧ b < 10 ∧ n = 1800 + a * 10 + b ∧ isPerfectSquare (18 * a)

theorem next_four_digit_number (n : ℕ) (h₀ : n = 1818) (h₁ : satisfiesProperty n) :
  ∃ m : ℕ, m > n ∧ m < 2000 ∧ satisfiesProperty m ∧ (∀ k : ℕ, n < k ∧ k < m → ¬ satisfiesProperty k) :=
begin
  use 1832,
  split,
  { exact 1832 > 1818, },
  split,
  { exact 1832 < 2000, },
  split,
  { unfold satisfiesProperty,
    use 32,
    use 0,
    split, { exact nat.le_of_lt 32 10 100, },
    split, { exact nat.lt_of_le_of_lt 0 32 40, },
    split, { exact nat.le_of_lt 0 0 10, },
    split, { exact nat.lt_of_le_of_lt 0 0 10, },
    split, { refl, },
    { unfold isPerfectSquare,
      use 24,
      exact nat.mul_self_eq 18 32 32, }, },
  { intros k hk,
    unfold satisfiesProperty at hk,
    cases hk with a ha,
    cases ha with b hb,
    cases hb with hb1 hb2,
    cases hb2 with hb3 hb4,
    cases hb4 with hb5 hb6,
    cases hb6 with hb7 hb8,
    cases hb8 with hb9 hb10,
    cases hb10 with hb11 hb12,
    exact nat.ne_of_lt_decides hk hb2 hb8 24, },
end

end next_four_digit_number_l408_408668


namespace equivalent_expression_ratio_l408_408242

theorem equivalent_expression_ratio :
  let c : ℚ := 8
  let p : ℚ := -3 / 8
  let q : ℚ := 604 / 32
  (c * ((j + p)^2) + q = 8 * j^2 - 6 * j + 20) →
  (q / p) = -151 / 3 :=
by
  intros c p q h
  rw [← h]
  sorry

end equivalent_expression_ratio_l408_408242


namespace max_value_of_symmetric_function_l408_408951

def f (x a b : ℝ) := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function 
  (a b : ℝ)
  (symmetric : ∀ t : ℝ, f (-2 + t) a b = f (-2 - t) a b) :
  ∃ M : ℝ, M = 16 ∧ ∀ x : ℝ, f x a b ≤ M :=
by
  use 16
  sorry

end max_value_of_symmetric_function_l408_408951


namespace next_number_property_l408_408675

theorem next_number_property (a b c d : ℕ) (h : a = 1 ∧ b = 8):
  ((10 * a + b) * (10 * c + d)) = (n * n) → 
  (1818 < (1000 * a + 100 * b + 10 * c + d)) →
  (∃ abcd, (1000 * a + 100 * b + 10 * c + d) = 1832) :=
by
  sorry

end next_number_property_l408_408675


namespace centroid_distance_relation_l408_408203

def squared_distance (u v : ℝ^3) : ℝ :=
  ∥u - v∥^2

noncomputable def centroid (a b c : ℝ^3) : ℝ^3 :=
  (a + b + c) / 3

theorem centroid_distance_relation (A B C P : ℝ^3) :
  let G := centroid A B C in
  squared_distance P A + squared_distance P B + squared_distance P C =
  3 * squared_distance P G + squared_distance G A + squared_distance G B + squared_distance G C :=
sorry

end centroid_distance_relation_l408_408203


namespace determine_a_l408_408646

def f (x a : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem determine_a (a : ℝ) (h : ∃ x, f' x a = 0 ∧ x = -3) : a = 5 := by
  sorry

noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

end determine_a_l408_408646


namespace minimum_value_l408_408598

noncomputable theory

open_locale big_operators

theorem minimum_value (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 8) : 
  ∃ m : ℝ, m = 64 ∧ (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ m :=
by
  sorry

end minimum_value_l408_408598


namespace circle_equation_through_ABC_circle_equation_with_center_and_points_l408_408348

-- Define points
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 0⟩
def B : Point := ⟨4, 0⟩
def C : Point := ⟨6, -2⟩

-- First problem: proof of the circle equation given points A, B, and C
theorem circle_equation_through_ABC :
  ∃ (D E F : ℝ), 
  (∀ (P : Point), (P = A ∨ P = B ∨ P = C) → P.x^2 + P.y^2 + D * P.x + E * P.y + F = 0) 
  ↔ (D = -5 ∧ E = 7 ∧ F = 4) := sorry

-- Second problem: proof of the circle equation given the y-coordinate of the center and points A and B
theorem circle_equation_with_center_and_points :
  ∃ (h k r : ℝ), 
  (h = (A.x + B.x) / 2 ∧ k = 2) ∧
  ∀ (P : Point), (P = A ∨ P = B) → (P.x - h)^2 + (P.y - k)^2 = r^2
  ↔ (h = 5 / 2 ∧ k = 2 ∧ r = 5 / 2) := sorry

end circle_equation_through_ABC_circle_equation_with_center_and_points_l408_408348


namespace area_of_square_with_perimeter_32_l408_408343

theorem area_of_square_with_perimeter_32 :
  ∀ (s : ℝ), 4 * s = 32 → s * s = 64 :=
by
  intros s h
  sorry

end area_of_square_with_perimeter_32_l408_408343


namespace gcd_of_repeated_three_digit_l408_408364

theorem gcd_of_repeated_three_digit : 
  ∀ (n : ℕ), 100 ≤ n ∧ n < 1000 → ∀ m ∈ {k : ℕ | ∃ n, 100 ≤ n ∧ n < 1000 ∧ k = 1001 * n}, Nat.gcd 1001 m = 1001 :=
by
  sorry

end gcd_of_repeated_three_digit_l408_408364


namespace area_ratio_XPQ_PQYZ_l408_408559

variables (X Y Z P Q : Type)
variables [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
variables (XY YZ XZ XP XQ : ℝ)
variables (h1 : XY = 30) (h2 : YZ = 45) (h3 : XZ = 54)
variables (h4 : XP = 18) (h5 : XQ = 36)

def area (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : ℝ := sorry

theorem area_ratio_XPQ_PQYZ (X Y Z P Q : Type) [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (XY YZ XZ XP XQ : ℝ)
  (h1 : XY = 30) (h2 : YZ = 45) (h3 : XZ = 54) (h4 : XP = 18) (h5 : XQ = 36):
  area X P Q / (area P Q Y + area P Q Z) = 27 / 50 :=
sorry   -- Proof will be filled by using the conditions as per the given problem.

end area_ratio_XPQ_PQYZ_l408_408559


namespace next_valid_number_after_1818_l408_408679

open Nat

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem next_valid_number_after_1818 : 
  ∀ (a b : ℕ), 10 * a + b > 18 → 18 * (10 * a + b) = 1832 → isPerfectSquare (18 * (10 * a + b)) → 
  10 * 10 * 8 + 10 * a + b = 1832 :=
by {
  intros,
  sorry
}

end next_valid_number_after_1818_l408_408679


namespace sum_of_coeffs_l408_408153

theorem sum_of_coeffs (a_5 a_4 a_3 a_2 a_1 a : ℤ) (h_eq : (x - 2)^5 = a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a) (h_a : a = -32) :
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 :=
by
  sorry

end sum_of_coeffs_l408_408153


namespace probability_BC_proof_l408_408694

-- Define the conditions as part of the structures 
structure Path (start end : Type) (steps : Nat) :=
  (intermediate : List start)
  (movesEast : Nat)
  (movesSouth : Nat)

def probability_passes_through_B_and_C (totalPaths throughBAndCPaths : Nat) : Real :=
  throughBAndCPaths / totalPaths

-- Given conditions
axiom city_map : Path ℕ ℕ 8
axiom paths_A_to_B : Path ℕ ℕ 4
axiom paths_B_to_C : Path ℕ ℕ 2
axiom paths_C_to_D : Path ℕ ℕ 3

-- Given values based on the problem statement
constant paths_A_to_D_via_B_and_C : Nat := 4 * 2 * 3
constant paths_A_to_D : Nat := 56

-- Problem statement: Prove the probability is 3/7
theorem probability_BC_proof : probability_passes_through_B_and_C paths_A_to_D paths_A_to_D_via_B_and_C = 3 / 7 :=
by sorry

end probability_BC_proof_l408_408694


namespace sequence_sum_l408_408030

theorem sequence_sum :
  (∀ r: ℝ, 
    (∀ x y: ℝ,
      r = 1 / 4 → 
      x = 256 * r → 
      y = x * r → 
      x + y = 80)) :=
by sorry

end sequence_sum_l408_408030


namespace find_m_n_sum_l408_408411

open Fintype

theorem find_m_n_sum : 
  ∀ (m n : ℕ), 
  (m + n).coprime -> 
  let p := (λ c : (Fin 2)^(Fin 4 × Fin 4), ¬ ∃ (i j : Fin 2), ∀ (di dj : Fin 3), c ⟨⟨i + di, di_sle⟩, ⟨j + dj, dj_sle⟩⟩ = 0) in
  p (m :ℝ) / (n :ℝ) = (65275 / 65536) -> 
  m + n = 130811 :=
sorry

end find_m_n_sum_l408_408411


namespace rick_sisters_count_l408_408626

theorem rick_sisters_count :
  ∃ n : ℕ,
  let initial_cards := 130,
      cards_kept := 15,
      cards_to_miguel := 13,
      friends := 8,
      cards_per_friend := 12,
      cards_per_sister := 3,
      remaining_cards := initial_cards - cards_kept - cards_to_miguel - (friends * cards_per_friend)
  in remaining_cards / cards_per_sister = 2 :=
by
  sorry

end rick_sisters_count_l408_408626


namespace proof_problem_l408_408516

open Set Real

noncomputable def U : Set ℝ := univ

noncomputable def A : Set ℝ := { x | x^2 > 2*x + 3 }

noncomputable def B : Set ℝ := { x | real.log x / real.log 3 > 1 }

noncomputable def C_U (B : Set ℝ) : Set ℝ := { x | x ∉ B }

theorem proof_problem :
  A ∪ C_U B = U := by
  sorry

end proof_problem_l408_408516


namespace largest_among_5_8_9_7_l408_408785

theorem largest_among_5_8_9_7 : ∀ n ∈ ({5, 8, 9, 7} : set ℕ), n ≤ 9 ∧ (n = 9 → ∃ m ∈ ({5, 8, 9, 7} : set ℕ), m = n) :=
by
  sorry

end largest_among_5_8_9_7_l408_408785


namespace system_solutions_l408_408514

theorem system_solutions (x y b : ℝ) (h1 : 4 * x + 2 * y = b) (h2 : 3 * x + 7 * y = 3 * b) (hx : x = -1) : 
  b = -22 :=
by 
  sorry

end system_solutions_l408_408514


namespace geom_seq_sum_2016_2017_l408_408187

noncomputable def geom_seq (n : ℕ) (a1 q : ℝ) : ℝ := a1 * q ^ (n - 1)

noncomputable def sum_geometric_seq (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then
  a1 * n
else
  a1 * (1 - q ^ n) / (1 - q)

theorem geom_seq_sum_2016_2017 :
  (a1 = 2) →
  (geom_seq 2 a1 q + geom_seq 5 a1 q = 0) →
  sum_geometric_seq a1 q 2016 + sum_geometric_seq a1 q 2017 = 2 :=
by
  sorry

end geom_seq_sum_2016_2017_l408_408187


namespace oak_trees_initial_count_l408_408297

theorem oak_trees_initial_count (x : ℕ) (cut_down : ℕ) (remaining : ℕ) (h_cut : cut_down = 2) (h_remaining : remaining = 7)
  (h_equation : (x - cut_down) = remaining) : x = 9 := by
  -- We are given that cut_down = 2
  -- and remaining = 7
  -- and we need to show that the initial count x = 9
  sorry

end oak_trees_initial_count_l408_408297


namespace combined_average_yield_l408_408817

theorem combined_average_yield (yield_A : ℝ) (price_A : ℝ) (yield_B : ℝ) (price_B : ℝ) (yield_C : ℝ) (price_C : ℝ) :
  yield_A = 0.20 → price_A = 100 → yield_B = 0.12 → price_B = 200 → yield_C = 0.25 → price_C = 300 →
  (yield_A * price_A + yield_B * price_B + yield_C * price_C) / (price_A + price_B + price_C) = 0.1983 :=
by
  intros hYA hPA hYB hPB hYC hPC
  sorry

end combined_average_yield_l408_408817


namespace sequence_sum_l408_408049

noncomputable def r : ℝ := 1 / 4
def t₀ : ℝ := 4096
def t₁ : ℝ := t₀ * r
def t₂ : ℝ := t₁ * r
def x : ℝ := t₂ * r
def y : ℝ := x * r

theorem sequence_sum : x + y = 80 := by
  -- The proof will go here
  sorry

end sequence_sum_l408_408049


namespace range_of_a_l408_408110

open Real

def prop_p (a : ℝ) : Prop :=
  ∀ m, m ∈ Icc (-1 : ℝ) 1 → a^2 - 5 * a - 3 ≥ sqrt (m^2 + 8)

def prop_q (a : ℝ) : Prop :=
  ∃ x, x^2 + a * x + 2 < 0

def pq_formulation (a : ℝ) : Prop :=
  (prop_p a ∨ prop_q a) ∧ ¬ (prop_p a ∧ prop_q a)

theorem range_of_a (a : ℝ) : pq_formulation a → a ∈ Icc (-real.sqrt 8) (-1) ∪ Ioo (real.sqrt 8) 6 :=
sorry

end range_of_a_l408_408110


namespace claire_gift_card_balance_l408_408813

/--
Claire has a $100 gift card to her favorite coffee shop.
A latte costs $3.75.
A croissant costs $3.50.
Claire buys one latte and one croissant every day for a week.
Claire buys 5 cookies, each costing $1.25.

Prove that the amount of money left on Claire's gift card after a week is $43.00.
-/
theorem claire_gift_card_balance :
  let initial_balance : ℝ := 100
  let latte_cost : ℝ := 3.75
  let croissant_cost : ℝ := 3.50
  let daily_expense : ℝ := latte_cost + croissant_cost
  let weekly_expense : ℝ := daily_expense * 7
  let cookie_cost : ℝ := 1.25
  let total_cookie_expense : ℝ := cookie_cost * 5
  let total_expense : ℝ := weekly_expense + total_cookie_expense
  let remaining_balance : ℝ := initial_balance - total_expense
  remaining_balance = 43 :=
by
  sorry

end claire_gift_card_balance_l408_408813


namespace cos_double_angle_of_parallel_vectors_l408_408520

theorem cos_double_angle_of_parallel_vectors (α : ℝ) 
  (h_parallel : (1 / 3, Real.tan α) = (Real.cos α, 1)) : 
  Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end cos_double_angle_of_parallel_vectors_l408_408520


namespace max_f_value_when_a_b_half_range_of_a_for_slope_unique_m_value_for_equation_l408_408601

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log x - (1 / 2) * a * x^2 - b * x

theorem max_f_value_when_a_b_half : 
  ∀ x : ℝ, 0 < x → (f x (1 / 2) (1 / 2)) ≤ (-3 / 4):=
  sorry

noncomputable def F (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  f x a b + (1 / 2) * a * x^2 + b * x + a / x

theorem range_of_a_for_slope:
  ∀ x : ℝ, 0 < x ∧ x ≤ 3 → (∂ (F x a b) / ∂ x) ≤ (1 / 2) → a ≥ (1 / 2):=
  sorry

theorem unique_m_value_for_equation :
  ∀ m : ℝ, m > 0 → (∃! x : ℝ, 2 * m * f x 0 (-1) = x^2) → m = (1 / 2):=
  sorry

end max_f_value_when_a_b_half_range_of_a_for_slope_unique_m_value_for_equation_l408_408601


namespace f_sum_l408_408600

def f (x : ℝ) : ℝ :=
  if x > 3 then x^2 - 1 else
  if 0 ≤ x ∧ x ≤ 3 then 3*x + 2 else
  5

theorem f_sum : f (-1) + f 1 + f 4 = 25 := by
  sorry

end f_sum_l408_408600


namespace degree_of_expr_l408_408128

-- Define the first polynomial factor
noncomputable def p1 : Polynomial ℤ := Polynomial.of_finsupp (finsupp.of_multiset x {5, -2*4, 3*3, 1, -14})

-- Define the second polynomial factor
noncomputable def p2 : Polynomial ℤ := Polynomial.of_finsupp (finsupp.of_multiset x {11*4, -8*8, 7*5, 40})

-- Define the polynomial to be subtracted
noncomputable def q : Polynomial ℤ := (2*X^3 + 3)^7

-- Define the complete expression
noncomputable def expr : Polynomial ℤ := (p1 * p2) - q

-- Statement to prove that degree equals to 21
theorem degree_of_expr : expr.degree = 21 :=
by { sorry }

end degree_of_expr_l408_408128


namespace evaluate_products_l408_408310

theorem evaluate_products : 
  (Real.cbrt 125) * (Real.root 256 4) * (Real.sqrt 16) = 80 := 
sorry

end evaluate_products_l408_408310


namespace tangent_line_zero_l408_408132

variable {ℝ : Type*}

def f (x a : ℝ) : ℝ := Real.exp x + a * x

theorem tangent_line_zero (a : ℝ) :
  (∃ x : ℝ, f x a = 0 ∧ Deriv f x a = 0) → a = -Real.exp 1 :=
by
  sorry

end tangent_line_zero_l408_408132


namespace s_point_condition_l408_408588

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x)

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 2 * a * x
noncomputable def g_prime (a : ℝ) (x : ℝ) : ℝ := 1 / x

theorem s_point_condition (a : ℝ) (x₀ : ℝ) (h_f_g : f a x₀ = g a x₀) (h_f'g' : f_prime a x₀ = g_prime a x₀) :
  a = 2 / Real.exp 1 :=
by
  sorry

end s_point_condition_l408_408588


namespace linemen_count_l408_408351

-- Define the initial conditions
def linemen_drink := 8
def skill_position_players_drink := 6
def total_skill_position_players := 10
def cooler_capacity := 126
def skill_position_players_drink_first := 5

-- Define the number of ounces drunk by skill position players during the first break
def skill_position_players_first_break := skill_position_players_drink_first * skill_position_players_drink

-- Define the theorem stating that the number of linemen (L) is 12 given the conditions
theorem linemen_count :
  ∃ L : ℕ, linemen_drink * L + skill_position_players_first_break = cooler_capacity ∧ L = 12 :=
by {
  sorry -- Proof to be provided.
}

end linemen_count_l408_408351


namespace factorize_P_l408_408437

noncomputable def P (y : ℝ) : ℝ :=
  (16 * y ^ 7 - 36 * y ^ 5 + 8 * y) - (4 * y ^ 7 - 12 * y ^ 5 - 8 * y)

theorem factorize_P (y : ℝ) : P y = 8 * y * (3 * y ^ 6 - 6 * y ^ 4 + 4) :=
  sorry

end factorize_P_l408_408437


namespace f_irreducible_l408_408592

noncomputable def f (n : ℕ) (x : ℤ) : ℤ := x^n + 5 * x^(n-1) + 3

theorem f_irreducible (n : ℕ) (hn : n > 1) : Irreducible (f n) :=
sorry

end f_irreducible_l408_408592


namespace rectangle_perimeter_l408_408625

open Real

theorem rectangle_perimeter (x y a b : ℝ) (A_ellipse : ℝ) (A_rectangle : ℝ) (major_axis : 2 * a)
  (minor_axis : 2 * b) (diagonal_eq : x^2 + y^2 = 4 * (a^2 - b^2)) (area_ellipse : π * a * b = 2016 * π)
  (area_rectangle : x * y = 2016) : 2 * (x + y) = 8 * sqrt (1008) :=
by
  have h_b : b = sqrt (1008),
  { sorry }, -- proof that b = sqrt(1008)
  have h_a : a = 2 * sqrt (1008),
  { sorry }, -- proof that a = 2 * sqrt(1008)
  calc
    2 * (x + y) = 2 * (2 * a) : by rw [hyp_sum_eq_2a]
           ... = 4 * a     : by ring
           ... = 8 * sqrt (1008) : by rw [h_a, mul_assoc]


end rectangle_perimeter_l408_408625


namespace dogs_adopted_l408_408300

theorem dogs_adopted {D : ℕ} :
  let initial_dogs := 36 in
  let initial_cats := 29 in
  let additional_cats := 12 in
  let final_pets := 57 in
  final_pets = (initial_dogs - D) + (initial_cats + additional_cats) → D = 20 :=
by
  intro h
  sorry

end dogs_adopted_l408_408300


namespace part_a_part_b_l408_408835

noncomputable def boys_on_field := 15

def no_two_distances_equal (boys : Finset ℕ) : Prop :=
  ∀ (x y z w : ℕ) (hx : x ∈ boys) (hy : y ∈ boys) (hz : z ∈ boys) (hw : w ∈ boys), (x ≠ y ∧ z ≠ w) → dist x y ≠ dist z w

def each_boy_throws_to_closest (boys : Finset ℕ) (throws : ℕ → ℕ) : Prop :=
  ∀ (x : ℕ), x ∈ boys → throws x = closest_boy x boys

theorem part_a (boys : Finset ℕ) (throws : ℕ → ℕ) (h_length : boys.card = boys_on_field)
  (h_distinct : no_two_distances_equal boys)
  (h_closest : each_boy_throws_to_closest boys throws) :
  ∃ x : ℕ, x ∈ boys ∧ ∀ y : ℕ, throws y ≠ x := sorry

theorem part_b (boys : Finset ℕ) (throws : ℕ → ℕ) (h_length : boys.card = boys_on_field)
  (h_distinct : no_two_distances_equal boys)
  (h_closest : each_boy_throws_to_closest boys throws) :
  ∀ x : ℕ, x ∈ boys → (Finset.univ.filter (λ y, throws y = x)).card ≤ 5 := sorry

end part_a_part_b_l408_408835


namespace chord_bisected_by_point_l408_408167

theorem chord_bisected_by_point (x y : ℝ) (h : (x - 2)^2 / 16 + (y - 1)^2 / 8 = 1) :
  ∃ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = -3 ∧ (∀ x y : ℝ, (a * x + b * y + c = 0 ↔ (x - 2)^2 / 16 + (y - 1)^2 / 8 = 1)) := by
  sorry

end chord_bisected_by_point_l408_408167


namespace dolly_should_buy_more_tickets_l408_408407

-- Define the conditions
def ferris_wheel_rides := 2
def roller_coaster_rides := 3
def log_ride_rides := 7

def ferris_wheel_cost_per_ride := 2
def roller_coaster_cost_per_ride := 5
def log_ride_cost_per_ride := 1

def initial_tickets := 20

-- Declare the theorem to prove
theorem dolly_should_buy_more_tickets :
  let total_needed_tickets := 
      ferris_wheel_rides * ferris_wheel_cost_per_ride + 
      roller_coaster_rides * roller_coaster_cost_per_ride + 
      log_ride_rides * log_ride_cost_per_ride in
  let additional_tickets := total_needed_tickets - initial_tickets in
  additional_tickets = 6 :=
by
  sorry

end dolly_should_buy_more_tickets_l408_408407


namespace next_four_digit_number_l408_408671

def isPerfectSquare (n : ℕ) : Prop :=
∃ m : ℕ, m * m = n

def satisfiesProperty (n : ℕ) : Prop :=
  ∃ a b : ℕ, 10 ≤ a ∧ a < 100 ∧ 0 ≤ b ∧ b < 10 ∧ n = 1800 + a * 10 + b ∧ isPerfectSquare (18 * a)

theorem next_four_digit_number (n : ℕ) (h₀ : n = 1818) (h₁ : satisfiesProperty n) :
  ∃ m : ℕ, m > n ∧ m < 2000 ∧ satisfiesProperty m ∧ (∀ k : ℕ, n < k ∧ k < m → ¬ satisfiesProperty k) :=
begin
  use 1832,
  split,
  { exact 1832 > 1818, },
  split,
  { exact 1832 < 2000, },
  split,
  { unfold satisfiesProperty,
    use 32,
    use 0,
    split, { exact nat.le_of_lt 32 10 100, },
    split, { exact nat.lt_of_le_of_lt 0 32 40, },
    split, { exact nat.le_of_lt 0 0 10, },
    split, { exact nat.lt_of_le_of_lt 0 0 10, },
    split, { refl, },
    { unfold isPerfectSquare,
      use 24,
      exact nat.mul_self_eq 18 32 32, }, },
  { intros k hk,
    unfold satisfiesProperty at hk,
    cases hk with a ha,
    cases ha with b hb,
    cases hb with hb1 hb2,
    cases hb2 with hb3 hb4,
    cases hb4 with hb5 hb6,
    cases hb6 with hb7 hb8,
    cases hb8 with hb9 hb10,
    cases hb10 with hb11 hb12,
    exact nat.ne_of_lt_decides hk hb2 hb8 24, },
end

end next_four_digit_number_l408_408671


namespace minimum_length_proof_l408_408097

noncomputable def minimum_segment_length (a : ℝ) : ℝ :=
  2 * a * (Real.sqrt 3 - Real.sqrt 2)

theorem minimum_length_proof (a : ℝ) :
  ∃ (MN : ℝ), 
  let angle_condition := 60, 
      edge_length := a in
  cube_conditions a AB_1 BC_1 ABCD MN angle_condition ->
  MN = minimum_segment_length a :=
sorry

end minimum_length_proof_l408_408097


namespace total_tickets_sold_l408_408355

/-
Problem: Prove that the total number of tickets sold is 65 given the conditions.
Conditions:
1. Senior citizen tickets cost 10 dollars each.
2. Regular tickets cost 15 dollars each.
3. Total sales were 855 dollars.
4. 24 senior citizen tickets were sold.
-/

def senior_tickets_sold : ℕ := 24
def senior_ticket_cost : ℕ := 10
def regular_ticket_cost : ℕ := 15
def total_sales : ℕ := 855

theorem total_tickets_sold (R : ℕ) (H : total_sales = senior_tickets_sold * senior_ticket_cost + R * regular_ticket_cost) :
  senior_tickets_sold + R = 65 :=
by
  sorry

end total_tickets_sold_l408_408355


namespace impossible_to_reach_final_configuration_l408_408545

-- Definitions for the conditions
inductive SquareType
| EE | EO | OE | OO

inductive MoveDirection
| horizontal | vertical | diagonal

structure Position (n : ℕ) :=
(row : Fin n)
(col : Fin n)

structure Move :=
(from : Position 8)
(to : Position 8)
(direction : MoveDirection)

def is_valid_move (move : Move) (board : Fin 8 → Fin 8 → Option SquareType) : Bool :=
  match move.direction with
  | MoveDirection.horizontal => (move.from.row = move.to.row ∧ move.from.col ≠ move.to.col)
  | MoveDirection.vertical => (move.from.col = move.to.col ∧ move.from.row ≠ move.to.row)
  | MoveDirection.diagonal => ((move.from.row.val + move.from.col.val + 2 * Fin n) % 2 == move.to.row.val + move.to.col.val) % 2

-- The proof statement
theorem impossible_to_reach_final_configuration :
  ∀ (initial_board final_board : Fin 8 → Fin 8 → Option SquareType), 
    (∀ i j, (i < 3) → initial_board ⟨i, sorry⟩ ⟨j, sorry⟩ ≠ none) → 
    (∀ i j, (i > 4) → final_board ⟨i, sorry⟩ ⟨j, sorry⟩ ≠ none) → 
    (∀ move, is_valid_move move initial_board → (initial_board move.from.row move.from.col = final_board move.to.row move.to.col)) → 
    false := 
  sorry

end impossible_to_reach_final_configuration_l408_408545


namespace find_a_l408_408892

theorem find_a (a x : ℝ) (h : x = -1) (heq : -2 * (x - a) = 4) : a = 1 :=
by
  sorry

end find_a_l408_408892


namespace derivative_f_l408_408912

def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6)

theorem derivative_f (x : ℝ) : deriv f x = -2 * Real.sin (2 * x - Real.pi / 6) := 
by 
  sorry

end derivative_f_l408_408912


namespace solve_for_x_l408_408454

theorem solve_for_x (x : ℝ) (h : sqrt (x - 5) = 10) : x = 105 :=
by
  sorry

end solve_for_x_l408_408454


namespace polynomial_characterization_l408_408072

theorem polynomial_characterization (P : polynomial ℝ) :
  (∀ x : ℂ, P.eval x * P.eval (x - 1) = P.eval (x^2)) →
  ∃ n : ℕ, P = (polynomial.C (1 : ℝ) * polynomial.X^2 + polynomial.C (1 : ℝ) * polynomial.X + polynomial.C (1 : ℝ))^n :=
by
  sorry

end polynomial_characterization_l408_408072


namespace submodular_iff_mono_decreasing_g_l408_408211

variables {S : Type*} [fintype S] {f : set S → ℝ}

-- Define monotonically decreasing function on the powerset of S
def mono_decreasing (f : set S → ℝ) : Prop :=
  ∀ {X Y : set S}, X ⊆ Y → f X ≥ f Y

-- Define the submodularity property
def submodular (f : set S → ℝ) : Prop :=
  ∀ (X Y : set S), f (X ∪ Y) + f (X ∩ Y) ≤ f X + f Y

-- Define the function g
def g (f : set S → ℝ) (a : S) (X : set S) : ℝ :=
  f (X ∪ {a}) - f X

-- Define the monotonically decreasing property for g
def g_mono_decreasing (f : set S → ℝ) (a : S) : Prop :=
  ∀ {X Y : set S}, X ⊆ Y → g f a X ≥ g f a Y

theorem submodular_iff_mono_decreasing_g (f : set S → ℝ) :
  submodular f ↔ ∀ a ∈ S, g_mono_decreasing f a :=
sorry

end submodular_iff_mono_decreasing_g_l408_408211


namespace complex_expression_identity_l408_408218

open Complex

theorem complex_expression_identity
  (x y : ℂ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hxy : x^2 + x * y + y^2 = 0) :
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 :=
by
  sorry

end complex_expression_identity_l408_408218


namespace evaluate_magnitude_product_l408_408419

-- Definitions of complex numbers
def z1 := Complex.mk 7 (-4)
def z2 := Complex.mk 3 11

-- The magnitude of z1
def magnitude_z1 := Complex.abs z1

-- The magnitude of z2
def magnitude_z2 := Complex.abs z2

-- Lean 4 statement expressing the problem and its final answer
theorem evaluate_magnitude_product : Complex.abs (z1 * z2) = Real.sqrt 8450 := by
  sorry

end evaluate_magnitude_product_l408_408419


namespace tuples_satisfy_equation_l408_408017

theorem tuples_satisfy_equation (a b c : ℤ) :
  (a - b)^3 * (a + b)^2 = c^2 + 2 * (a - b) + 1 ↔ (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = -1 ∧ b = 0 ∧ c = 0) :=
sorry

end tuples_satisfy_equation_l408_408017


namespace Cody_money_final_l408_408010

-- Define the initial amount of money Cody had
def Cody_initial : ℝ := 45.0

-- Define the birthday gift amount
def birthday_gift : ℝ := 9.0

-- Define the amount spent on the game
def game_expense : ℝ := 19.0

-- Define the percentage of remaining money spent on clothes as a fraction
def clothes_spending_fraction : ℝ := 0.40

-- Define the late birthday gift received
def late_birthday_gift : ℝ := 4.5

-- Define the final amount of money Cody has
def Cody_final : ℝ :=
  let after_birthday := Cody_initial + birthday_gift
  let after_game := after_birthday - game_expense
  let spent_on_clothes := clothes_spending_fraction * after_game
  let after_clothes := after_game - spent_on_clothes
  after_clothes + late_birthday_gift

theorem Cody_money_final : Cody_final = 25.5 := by
  sorry

end Cody_money_final_l408_408010


namespace min_angle_of_inclination_l408_408121

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 3 * x

noncomputable def df (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) - 3

theorem min_angle_of_inclination :
  let α := fun x => Real.arctan (df x) in
  (∀ (x : ℝ), -1 / 2 ≤ x ∧ x ≤ 1 / 2 → α x = Real.atan (df x)) →
  ∃ (x : ℝ), -1 / 2 ≤ x ∧ x ≤ 1 / 2 ∧ α x = 3 * Real.pi / 4 :=
by
  sorry

end min_angle_of_inclination_l408_408121


namespace jill_total_watch_time_l408_408563

theorem jill_total_watch_time :
  ∀ (length_first_show length_second_show total_watch_time : ℕ),
    length_first_show = 30 →
    length_second_show = 4 * length_first_show →
    total_watch_time = length_first_show + length_second_show →
    total_watch_time = 150 :=
by
  sorry

end jill_total_watch_time_l408_408563


namespace steve_total_time_on_roads_each_day_l408_408643

-- Definitions of conditions
def distance_to_work : ℝ := 35
def speed_back : ℝ := 17.5
def speed_to := speed_back / 2

-- Time calculations
def time_to_work := distance_to_work / speed_to
def time_back := distance_to_work / speed_back

-- Calculating total time spent on roads each day
def total_time_on_roads := time_to_work + time_back

-- Theorem stating the result
theorem steve_total_time_on_roads_each_day : total_time_on_roads = 6 := by
  -- Omitted proof steps go here
  sorry

end steve_total_time_on_roads_each_day_l408_408643


namespace seven_digit_palindromes_count_l408_408525

def is_palindrome (n : ℕ) : Prop :=
  let digits := to_digits n in
  digits = digits.reverse

def count_palindromes : ℕ :=
  let digits := [1, 1, 2, 2, 2, 4, 4] in
  let palindromes := {n | is_palindrome n ∧ to_digits n ⊆ digits} in
  card palindromes

theorem seven_digit_palindromes_count :
  count_palindromes = 6 :=
sorry

end seven_digit_palindromes_count_l408_408525


namespace lunch_combinations_l408_408795

/-- The number of different lunch combinations Sam can choose from,
    given there are 4 main courses, 3 beverages, and 2 snacks, is 24. -/
theorem lunch_combinations (main_courses : ℕ) (beverages : ℕ) (snacks : ℕ) 
  (h_main_courses : main_courses = 4) (h_beverages : beverages = 3) (h_snacks : snacks = 2) :
  (main_courses * beverages * snacks) = 24 :=
by
  rw [h_main_courses, h_beverages, h_snacks]
  exact dec_trivial

end lunch_combinations_l408_408795


namespace total_buildings_proof_l408_408195

-- Given conditions
variables (stores_pittsburgh hospitals_pittsburgh schools_pittsburgh police_stations_pittsburgh : ℕ)
variables (stores_new hospitals_new schools_new police_stations_new buildings_new : ℕ)

-- Given values for Pittsburgh
def stores_pittsburgh := 2000
def hospitals_pittsburgh := 500
def schools_pittsburgh := 200
def police_stations_pittsburgh := 20

-- Definitions for the new city
def stores_new := stores_pittsburgh / 2
def hospitals_new := 2 * hospitals_pittsburgh
def schools_new := schools_pittsburgh - 50
def police_stations_new := police_stations_pittsburgh + 5
def buildings_new := stores_new + hospitals_new + schools_new + police_stations_new

-- Statement to prove
theorem total_buildings_proof : buildings_new = 2175 := by
  dsimp [buildings_new, stores_new, hospitals_new, schools_new, police_stations_new] 
  dsimp [stores_pittsburgh, hospitals_pittsburgh, schools_pittsburgh, police_stations_pittsburgh]
  rfl

end total_buildings_proof_l408_408195


namespace largest_in_set_l408_408157

theorem largest_in_set (a : ℤ) (h : a = -4) : 
  ∃ x ∈ ({-2 * a^2, 5 * a, 40 / a, 3 * a^2, 2} : set ℚ), 
  (∀ y ∈ ({-2 * a^2, 5 * a, 40 / a, 3 * a^2, 2} : set ℚ), y ≤ x) ∧ x = 3 * a^2 := by
  sorry

end largest_in_set_l408_408157


namespace units_digit_first_2023_odd_squares_l408_408728

noncomputable def units_digit_of_squares_sum (n : ℕ) : ℕ :=
  let odd_squares_units := λ k, (2 * k + 1) ^ 2 % 10 in
  (List.sum (List.map odd_squares_units (List.range n))) % 10

theorem units_digit_first_2023_odd_squares :
  units_digit_of_squares_sum 2023 = 5 :=
sorry

end units_digit_first_2023_odd_squares_l408_408728


namespace geometric_series_sum_l408_408008

-- Conditions
def is_geometric_series (a r : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a * r ^ n

-- The problem statement translated into Lean: Proving the sum of the series
theorem geometric_series_sum : ∃ S : ℕ → ℝ, is_geometric_series 1 (1/4) S ∧ ∑' n, S n = 4/3 :=
by
  sorry

end geometric_series_sum_l408_408008


namespace cover_rectangles_exactly_l408_408713

-- Definition of rectangle covering problem
def smallest_number_of_covering_rectangles (w h: ℕ) (rw rh: ℕ) : ℕ :=
  (w * h) / (rw * rh)

-- Proof statement
theorem cover_rectangles_exactly (w h rw rh: ℕ) (h_rw: rw = 3) (h_rh: rh = 4) 
                                (h_w: w = 6) (h_h: h = 8)
                                (h_area: (w * h) % (rw * rh) = 0) : 
  smallest_number_of_covering_rectangles w h rw rh = 4 :=
  by {
    -- Using conditions and definitions provided
    rw [h_rw, h_rh, h_w, h_h],
    -- Calculate the number of rectangles
    trivial,
  }

end cover_rectangles_exactly_l408_408713


namespace dolly_needs_more_tickets_l408_408409

-- Definitions of the conditions
def rides_ferris_wheel := 2
def cost_ferris_wheel := 2
def rides_roller_coaster := 3
def cost_roller_coaster := 5
def rides_log_ride := 7
def cost_log_ride := 1
def tickets_dolly_has := 20

-- The theorem statement
theorem dolly_needs_more_tickets :
  let total_tickets_needed := rides_ferris_wheel * cost_ferris_wheel + 
                              rides_roller_coaster * cost_roller_coaster +
                              rides_log_ride * cost_log_ride
  in total_tickets_needed - tickets_dolly_has = 6 :=
by
  sorry

end dolly_needs_more_tickets_l408_408409


namespace units_digit_first_2023_odd_squares_l408_408730

noncomputable def units_digit_of_squares_sum (n : ℕ) : ℕ :=
  let odd_squares_units := λ k, (2 * k + 1) ^ 2 % 10 in
  (List.sum (List.map odd_squares_units (List.range n))) % 10

theorem units_digit_first_2023_odd_squares :
  units_digit_of_squares_sum 2023 = 5 :=
sorry

end units_digit_first_2023_odd_squares_l408_408730


namespace remainder_N_div_5_is_1_l408_408754

-- The statement proving the remainder of N when divided by 5 is 1
theorem remainder_N_div_5_is_1 (N : ℕ) (h1 : N % 2 = 1) (h2 : N % 35 = 1) : N % 5 = 1 :=
sorry

end remainder_N_div_5_is_1_l408_408754


namespace optimal_path_exists_l408_408630

-- Defining the conditions:
def total_distance : ℕ := 76
def num_segments : ℕ := 16
def segment_length : ℕ := 1
def total_settlements : ℕ := total_distance + 1 -- settlements are 1 km apart
def settlements_to_avoid : ℕ := 3

-- Given the settlements are placed on a grid 1 km apart.
-- Prove that there exists a path of the specified length and segments that avoids exactly 3 settlements.
theorem optimal_path_exists 
  (d : ℕ := total_distance) 
  (s : ℕ := num_segments) 
  (l : ℕ := segment_length) 
  (t_s : ℕ := total_settlements) 
  (avoid : ℕ := settlements_to_avoid) :
  ∃ (path : List (ℕ × ℕ)), -- Representing the path as a list of coordinates
    (path.length = s + 1) ∧  -- Path consists of s+1 points (16 segments)
    ((list.distinct path) = true) ∧ -- No revisiting points except missing 3
    (path.last val = (d, ?m)) ∧ -- Path finishes at the distance covered
    (list.count ((λ p, p ∉ path) path) = avoid) := 
  sorry

end optimal_path_exists_l408_408630


namespace sign_of_fx0_l408_408115

noncomputable def f (x : ℝ) : ℝ := (1/3)^x + Real.logb (1/3) x

variable {a : ℝ} (ha : f a = 0)
variable {x_0 : ℝ} (hx_0 : 0 < x_0) (hxa : x_0 < a)

theorem sign_of_fx0 : f(x_0) > 0 :=
sorry

end sign_of_fx0_l408_408115


namespace exists_good_n_l408_408805

def is_good (m : ℕ) : Prop :=
  ∃ a b c : ℤ, m = a^3 + 2 * b^3 + 4 * c^3 - 6 * a * b * c

theorem exists_good_n :
  ∃ n : ℕ, n < 2024 ∧ ∃ᶠ p in filter.at_top, prime p ∧ is_good (n * p) :=
by
  sorry

end exists_good_n_l408_408805


namespace monotonic_intervals_max_k_for_g_lt_f_l408_408133

-- Definitions and conditions
def f (x : ℝ) := (1 / 2) * x^2 + -3 * x + 2 * Real.log x
def g (x k : ℝ) := (1 / 2) * x^2 + k * x + (2 - x) * Real.log x - k

-- Statement 1
theorem monotonic_intervals :
  ∀ x > 0, (∃ a, a = -3 → 
  ((∀ x ∈ Set.Ioo 0 1, HasDerivAt f' x (x - 3 + 2 / x) → Deriv f x > 0) ∧ 
   (∀ x ∈ Set.Ioo 1 2, HasDerivAt f' x (x - 3 + 2 / x) → Deriv f x < 0) ∧ 
   (∀ x ∈ Set.Ioi 2, HasDerivAt f' x (x - 3 + 2 / x) → Deriv f x > 0))) :=
by
sorrry

-- Statement 2
theorem max_k_for_g_lt_f :
  ∀ x > 1, (a = 1 → g x k < f x) → (k ∈ ℤ → max_k = 3) :=
by
sorrry

end monotonic_intervals_max_k_for_g_lt_f_l408_408133


namespace notable_phone_numbers_count_l408_408804

open Nat

def is_palindrome (n : ℕ) : Prop :=
  (n.toDigits 10).reverse == n.toDigits 10

def valid_triple (d f b : ℕ) : Prop :=
  d + f + b = 15

theorem notable_phone_numbers_count :
  (∑ a in (finset.range 10),
    ∑ b in (finset.range 10),
    ∑ c in (finset.range 10),
    ∑ d in (finset.range 10),
    ∑ f in (finset.range 10),
    ∑ g in (finset.range 10),
    ite (a = g ∧ c = f ∧ valid_triple d f b) 1 0) = 2700 :=
by {
  -- this would contain the detailed proof steps
  sorry
}

end notable_phone_numbers_count_l408_408804


namespace greatest_four_digit_number_l408_408315

theorem greatest_four_digit_number (x : ℕ) :
  x ≡ 1 [MOD 7] ∧ x ≡ 5 [MOD 8] ∧ 1000 ≤ x ∧ x < 10000 → x = 9997 :=
by
  sorry

end greatest_four_digit_number_l408_408315


namespace units_digit_sum_squares_first_2023_odd_integers_l408_408719

theorem units_digit_sum_squares_first_2023_odd_integers :
  let pattern := [1, 9, 5, 9, 1] in
  let full_cycles := 2023 / 5 in
  let remaining := 2023 % 5 in
  (full_cycles * (pattern.foldl (λ acc x => acc + x) 0) + (remaining * pattern.foldl (λ acc x => acc + if x = (remaining - 1).natAbs then x else 0) 0)) % 10 = 5 :=
by
  sorry

end units_digit_sum_squares_first_2023_odd_integers_l408_408719


namespace min_value_inequality_l408_408586

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  (Real.sqrt ((x^2 + y^2 + z^2) * (4 * x^2 + y^2 + z^2))) / (x * y * z)

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  min_value_expression x y z ≥ 3 / 2 := by
  sorry

end min_value_inequality_l408_408586


namespace problem_1_problem_2_l408_408878

noncomputable def S (n : ℕ) : ℕ := n * (n - 1) / 2
def a (n : ℕ) : ℕ := n - 1
noncomputable def b (n : ℕ) : ℝ := (4 / 15) * (-2) ^ (a n)

theorem problem_1 (k : ℕ) : let d_k := b (2 * k + 1) - b (2 * k - 1) in
  d_k / (b ((2 * k + 2) + 1) - b ((2 * k + 2) - 1)) = 1/4 := sorry

theorem problem_2 (k : ℕ) : 
  let d_k := b (2 * k + 1) - b (2 * k - 1) 
  let d_k1 := b (2 * (k + 1) + 1) - b (2 * (k + 1) - 1) in
  if odd k then 
    card {x : ℤ | d_k < x ∧ x < d_k1} = 3 * (4^k + 1) / 5
  else 
    card {x : ℤ | d_k < x ∧ x < d_k1} = 3 * (4^k - 1) / 5 := sorry

end problem_1_problem_2_l408_408878


namespace sequence_sum_l408_408031

theorem sequence_sum :
  ∃ (r : ℝ) (x y : ℝ),
    (r = 1 / 4) ∧
    (x = 256 * r) ∧
    (y = x * r) ∧
    (x + y = 80) :=
by
  use 1 / 4, 64, 16
  split
  . exact rfl
  split
  . exact rfl
  split
  . exact rfl
  . norm_num


end sequence_sum_l408_408031


namespace largest_lambda_ineq_l408_408844

theorem largest_lambda_ineq :
  ∀ (a b c d : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a^2 + b^2 + c^2 + d^2 ≥ ab + 0 * bc + 2 * cd + ad :=
by
  intros a b c d ha hb hc hd
  have h1 : a^2 + b^2 ≥ 2 * ab := by nlinarith
  have h2 : c^2 + d^2 ≥ 2 * cd := by nlinarith
  have h3 : a^2 + d^2 ≥ 2 * ad := by nlinarith
  nlinarith

end largest_lambda_ineq_l408_408844


namespace p_q_work_l408_408330

theorem p_q_work (p_rate q_rate : ℝ) (h1: 1 / p_rate + 1 / q_rate = 1 / 6) (h2: p_rate = 15) : q_rate = 10 :=
by
  sorry

end p_q_work_l408_408330


namespace angle_OQP_right_angle_l408_408289

variables (A B C D P Q O : Type)
           [metric_space O]
           [circumscribed_quad A B C D O]
           [circumcircle_intersections AC BD P]
           [circumcircles_intersecting_twice ABP DCP Q]

-- Given that the points A, B, C, and D lie on a circle with center O, 
-- the diagonals AC and BD intersect at P, and the circumcircles of 
-- triangles ABP and DCP intersect again at Q, prove that ∠OQP = 90°.
theorem angle_OQP_right_angle
  (h1 : quadri_is_circumscribed A B C D O)
  (h2 : intersecting_diagonals AC BD P)
  (h3 : circumcircles_intersect_twice ABP DCP Q) :
  ∠ O Q P = 90 := 
sorry

end angle_OQP_right_angle_l408_408289


namespace find_x_l408_408936

theorem find_x (y : ℝ) (x : ℝ) (h : x / (x - 1) = (y^2 + 2 * y + 3) / (y^2 + 2 * y - 2)) :
  x = (y^2 + 2 * y + 3) / 5 := by
  sorry

end find_x_l408_408936


namespace josh_total_money_left_l408_408569

-- Definitions of the conditions
def profit_per_bracelet : ℝ := 1.5 - 1
def total_bracelets : ℕ := 12
def cost_of_cookies : ℝ := 3

-- The proof problem: 
theorem josh_total_money_left : total_bracelets * profit_per_bracelet - cost_of_cookies = 3 :=
by
  sorry

end josh_total_money_left_l408_408569


namespace matching_function_l408_408168

open Real

def table_data : List (ℝ × ℝ) := [(1, 4), (2, 2), (4, 1)]

theorem matching_function :
  ∃ a b c : ℝ, a > 0 ∧ 
               (∀ x y, (x, y) ∈ table_data → y = a * x^2 + b * x + c) := 
sorry

end matching_function_l408_408168


namespace distinct_intersection_points_l408_408457

theorem distinct_intersection_points (lines : Finset (Set Point)) (h_distinct : lines.card = 5) (h_no_three_intersect : ∀ (l1 l2 l3 : Set Point), l1 ∈ lines → l2 ∈ lines → l3 ∈ lines → l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 → ∃ p, p ∈ l1 ∩ l2 ∧ p ∉ l3) :
  ∑ (p : Point), (∃ l1 l2 ∈ lines, l1 ≠ l2 ∧ p ∈ l1 ∧ p ∈ l2) = 55 :=
sorry

end distinct_intersection_points_l408_408457


namespace initial_machines_count_l408_408940

noncomputable def initialMachines := Nat

noncomputable def daysForInitialMachinesWork (M : initialMachines) : Nat := 36

noncomputable def additionalMachines := 5

noncomputable def daysWithAdditionalMachinesWork (M : initialMachines) : Nat := 27 

theorem initial_machines_count (M : initialMachines) :
    M / 36 = (M + 5) / 27 → M = 20 := 
by
  sorry

end initial_machines_count_l408_408940


namespace sum_arithmetic_sequence_min_value_l408_408481

theorem sum_arithmetic_sequence_min_value (a d : ℤ) 
  (S : ℕ → ℤ) 
  (H1 : S 8 ≤ 6) 
  (H2 : S 11 ≥ 27)
  (H_Sn : ∀ n, S n = n * a + (n * (n - 1) / 2) * d) : 
  S 19 ≥ 133 :=
by
  sorry

end sum_arithmetic_sequence_min_value_l408_408481


namespace leah_probability_of_seeing_change_l408_408779

open Set

-- Define the length of each color interval
def green_duration := 45
def yellow_duration := 5
def red_duration := 35

-- Total cycle duration
def total_cycle_duration := green_duration + yellow_duration + red_duration

-- Leah's viewing intervals
def change_intervals : Set (ℕ × ℕ) :=
  {(40, 45), (45, 50), (80, 85)}

-- Probability calculation
def favorable_time := 15
def probability_of_change := (favorable_time : ℚ) / (total_cycle_duration : ℚ)

theorem leah_probability_of_seeing_change : probability_of_change = 3 / 17 :=
by
  -- We use sorry here as we are only required to state the theorem without proof.
  sorry

end leah_probability_of_seeing_change_l408_408779


namespace projection_of_b_onto_a_is_a_l408_408958

noncomputable def vector_a : (Real × Real) := (1, 0)
noncomputable def vector_b : (Real × Real) := (1, Real.sqrt 3)

-- Function to calculate the projection of one vector onto another.
def proj (a b : Real × Real) : (Real × Real) :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_squared := a.1 ^ 2 + a.2 ^ 2
  let scalar := dot_product / magnitude_squared
  (scalar * a.1, scalar * a.2)

theorem projection_of_b_onto_a_is_a : proj vector_a vector_b = vector_a := by
  sorry

end projection_of_b_onto_a_is_a_l408_408958


namespace reporters_not_covering_politics_l408_408737

noncomputable def percent_not_covering_politics (total_reporters covering_local_politics total_politics : ℝ) : ℝ :=
  100 - (covering_local_politics / (total_politics / 100))

theorem reporters_not_covering_politics
  (total_reporters : ℝ)
  (percent_covering_local_politics : ℝ) 
  (percent_total_politics_not_covering_local : ℝ) 
  (approx_answer : ℝ) :
  percent_covering_local_politics = 5 →
  percent_total_politics_not_covering_local = 30 →
  percent_not_covering_politics total_reporters percent_covering_local_politics (percent_covering_local_politics / 0.70) ≈ approx_answer :=
by
  sorry

end reporters_not_covering_politics_l408_408737


namespace intersection_volume_fraction_of_tetrahedra_in_cube_l408_408703

theorem intersection_volume_fraction_of_tetrahedra_in_cube (a : ℝ) (h_a_pos : a > 0) :
  let V_cube := a^3 in
  let V_tetrahedron := (1 / 6) * V_cube in
  let V_octahedron := (1 / 6) * V_cube in
  V_octahedron / V_cube = 1 / 6 :=
by
  sorry

end intersection_volume_fraction_of_tetrahedra_in_cube_l408_408703


namespace units_digit_sum_of_squares_odd_integers_l408_408723

theorem units_digit_sum_of_squares_odd_integers :
  let units_digit (n : ℤ) : ℤ := n % 10
  let sum_units_digits (n : ℕ) : ℤ :=
    (List.range (2 * n + 1)).filter (λ x => x % 2 = 1).map (λ k => units_digit (k * k)).sum in
  units_digit (sum_units_digits 2023) = 5 :=
begin
  sorry
end

end units_digit_sum_of_squares_odd_integers_l408_408723


namespace claire_balance_after_week_l408_408810

theorem claire_balance_after_week :
  ∀ (gift_card : ℝ) (latte_cost croissant_cost : ℝ) (days : ℕ) (cookie_cost : ℝ) (cookies : ℕ),
  gift_card = 100 ∧
  latte_cost = 3.75 ∧
  croissant_cost = 3.50 ∧
  days = 7 ∧
  cookie_cost = 1.25 ∧
  cookies = 5 →
  (gift_card - (days * (latte_cost + croissant_cost) + cookie_cost * cookies) = 43) :=
by
  -- Skipping proof details with sorry
  sorry

end claire_balance_after_week_l408_408810


namespace angle_CDM_l408_408359

-- Definitions of the points and conditions in the problem
variables {A B C D M : Type} [AddCommGroup A] [AffineSpace B A]
          [AddCommGroup C] [AffineSpace D C] [AddCommGroup M] [AffineSpace M B]

-- Given conditions as hypotheses
hypothesis (h1 : ∠ACB = 90°)
hypothesis (h2 : inscribed_in_circle A B C)
hypothesis (h3 : BC > AC)
hypothesis (h4 : D ∈ line BC ∧ BD = AC)
hypothesis (h5 : M = midpoint_arc_non_including B)

-- The statement of the theorem
theorem angle_CDM :
  measure_angle C D M = 45° :=
by
  sorry

end angle_CDM_l408_408359


namespace train_cross_time_l408_408705

/-- Given the conditions:
1. Two trains run in opposite directions and cross a man in 17 seconds and some unknown time respectively.
2. They cross each other in 22 seconds.
3. The ratio of their speeds is 1 to 1.
Prove the time it takes for the first train to cross the man. -/
theorem train_cross_time (v_1 v_2 L_1 L_2 : ℝ) (t_2 : ℝ) (h1 : t_2 = 17) (h2 : v_1 = v_2)
  (h3 : (L_1 + L_2) / (v_1 + v_2) = 22) : (L_1 / v_1) = 27 := 
by 
  -- The actual proof will go here
  sorry

end train_cross_time_l408_408705


namespace next_number_property_l408_408674

theorem next_number_property (a b c d : ℕ) (h : a = 1 ∧ b = 8):
  ((10 * a + b) * (10 * c + d)) = (n * n) → 
  (1818 < (1000 * a + 100 * b + 10 * c + d)) →
  (∃ abcd, (1000 * a + 100 * b + 10 * c + d) = 1832) :=
by
  sorry

end next_number_property_l408_408674


namespace find_standard_equation_of_ellipse_find_range_of_k_l408_408104

noncomputable def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem find_standard_equation_of_ellipse 
(eccentricity : ℝ) (focal_distance : ℝ)
(a b : ℝ) (h1 : focal_distance = 4) (h2 : a > b) 
(h3 : b > 0) (h4 : a * eccentricity = focal_distance)
(h5 : eccentricity = (Real.sqrt 2) / 2) :
ellipse_eq a b = ellipse_eq (2 * Real.sqrt 2) 2 := by
  sorry

theorem find_range_of_k 
(a b : ℝ) (k : ℝ) (M : ℝ × ℝ) (h1 : M = (0, 1)) 
(h2 : ellipse_eq (2 * Real.sqrt 2) 2) :
(k < 1 / 8) ↔ 
  let x1 x2 : ℝ, y1 y2 : ℝ,
  A B : ℝ × ℝ
      := let line_eq : ℝ × ℝ → Prop := λ p, p.snd = k * p.fst + 1,
      pair_A : y1,
      pair_B : y2,
      M * Real.sqrt ((x1 - 2)*(x2 - 2)+ y1*y2) := sorry 
in
  ∀ F : ℝ × ℝ, F = (2,0) (let focus_eq : ℝ → ℝ × ℝ → Prop := λ slope p, slope * M + F
  sorry
  in true) := by
  sorry

end find_standard_equation_of_ellipse_find_range_of_k_l408_408104


namespace incorrect_statement_l408_408964

noncomputable def data : List ℝ := [6, 8, 8, 9, 8, 9, 8, 8, 7, 9]

def mode (l : List ℝ) : ℝ := l.mode.get
def mean (l : List ℝ) : ℝ := l.sum / l.length
def median (l : List ℝ) : ℝ := l.median.get
def variance (l : List ℝ) : ℝ :=
  let μ := mean l in
  (l.map (λ x => (x - μ)^2)).sum / l.length

theorem incorrect_statement :
  mode data = 8 ∧
  mean data = 8 ∧
  median data = 8 ∧
  variance data ≠ 8 :=
by
  sorry

end incorrect_statement_l408_408964


namespace mike_age_proof_l408_408384

theorem mike_age_proof (a m : ℝ) (h1 : m = 3 * a - 20) (h2 : m + a = 70) : m = 47.5 := 
by {
  sorry
}

end mike_age_proof_l408_408384


namespace volume_phi_l408_408704

noncomputable def volume_of_midpoints_figure (tetrahedron_edge_length : ℝ) (symmetric : ∀ x y, symmetric_relation x y) : ℝ :=
5 / 6

theorem volume_phi (tetrahedron_edge_length : ℝ) (symmetric : ∀ x y, symmetric_relation x y) (Hedge_length : tetrahedron_edge_length = real.sqrt 2) :
  volume_of_midpoints_figure tetrahedron_edge_length symmetric = 5 / 6 :=
sorry

end volume_phi_l408_408704


namespace num_white_balls_l408_408345

theorem num_white_balls (W : ℕ) (h : (W : ℝ) / (6 + W) = 0.45454545454545453) : W = 5 :=
by
  sorry

end num_white_balls_l408_408345


namespace middle_card_is_five_l408_408697

noncomputable theory

def distinct_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def increasing_order (a b c : ℕ) : Prop :=
  a < b ∧ b < c

def sum_to_sixteen (a b c : ℕ) : Prop :=
  a + b + c = 16

def andy_unknown (a : ℕ) (sets : list (ℕ × ℕ × ℕ)) : Prop :=
  ∀ s ∈ sets, s.1 = a → 
    (∃ t ∈ sets, t.1 = a ∧ t.2 ≠ s.2 ∨ t.1 = a ∧ t.3 ≠ s.3)

def cindy_unknown (c : ℕ) (sets : list (ℕ × ℕ × ℕ)) : Prop :=
  ∀ s ∈ sets, s.3 = c → 
    (∃ t ∈ sets, t.3 = c ∧ t.1 ≠ s.1 ∨ t.3 = c ∧ t.2 ≠ s.2)

def beth_unknown (b : ℕ) (sets : list (ℕ × ℕ × ℕ)) : Prop :=
  ∀ s ∈ sets, s.2 = b → 
    (∃ t ∈ sets, t.2 = b ∧ t.1 ≠ s.1 ∧ t.3 ≠ s.3)

theorem middle_card_is_five : 
  ∀ (a b c : ℕ), 
    distinct_integers a b c →
    increasing_order a b c →
    sum_to_sixteen a b c →
    let sets := [(1, 2, 13), (1, 3, 12), (1, 4, 11), (1, 5, 10), (1, 6, 9), (1, 7, 8), (2, 3, 11), (2, 4, 10), (2, 5, 9), (2, 6, 8), (3, 4, 9), (3, 5, 8), (3, 6, 7), (4, 5, 7)] in
    andy_unknown a sets →
    cindy_unknown c sets →
    beth_unknown b sets →
    b = 5 :=
by
  intros
  sorry

end middle_card_is_five_l408_408697


namespace flavors_needed_this_year_l408_408147

def num_flavors_total : ℕ := 100

def num_flavors_two_years_ago : ℕ := num_flavors_total / 4

def num_flavors_last_year : ℕ := 2 * num_flavors_two_years_ago

def num_flavors_tried_so_far : ℕ := num_flavors_two_years_ago + num_flavors_last_year

theorem flavors_needed_this_year : 
  (num_flavors_total - num_flavors_tried_so_far) = 25 := by {
  sorry
}

end flavors_needed_this_year_l408_408147


namespace triangle_bisector_right_angle_l408_408238

variables {A B C D E M : Type} [EuclideanPlane A B C]

theorem triangle_bisector_right_angle (ABC: Triangle A B C) (hABC: ∠ABC = 120°) : 
  let D := angle_bisector_intersection B ABC,
      E := angle_bisector_intersection A ABC,
      M := angle_bisector_intersection C ABC,
      bisector_triangle := Triangle D E M in
  is_right_angled bisector_triangle :=
sorry

end triangle_bisector_right_angle_l408_408238


namespace max_average_speed_l408_408000

theorem max_average_speed :
  ∃ avg_speed : ℝ,
    let initial_odometer := 12321,
        palindromes := [12421, 12521, 12621, 12721],
        time := 4,
        speed_limit := 80,
        min_speed := 60 in
    ∀ odometer ∈ palindromes,
      ∃ (distance := odometer - initial_odometer),
        distance ≤ speed_limit * time ∧
        avg_speed = distance / time ∧
        avg_speed > min_speed ∧
        avg_speed = 75 := sorry

end max_average_speed_l408_408000


namespace relationship_among_a_b_and_c_l408_408580

theorem relationship_among_a_b_and_c :
  let a := log 4 (0.3)
  let b := log 3 (4)
  let c := 0.3^(-2)
  a < b ∧ b < c :=
by
  sorry

end relationship_among_a_b_and_c_l408_408580


namespace constant_term_binomial_expansion_l408_408207

theorem constant_term_binomial_expansion :
  let a := ∫ x in (0 : ℝ)..(Real.exp 2 - 1), 1 / (x + 1)
  (x^2 - a / x)^9.expand.coeff 0 = 5376 := by
  sorry

end constant_term_binomial_expansion_l408_408207


namespace least_possible_integer_l408_408352

/-- 
A group of 20 friends were discussing a large positive integer. 
The integer can be divided by 1, 2, 3, ..., 18. 
Exactly two friends were incorrect, and those two friends said consecutive numbers 19 and 20. 
The least possible integer they were discussing is 12252240.
-/
theorem least_possible_integer :
  ∃ (N : ℕ), (∀ i : ℕ, 1 ≤ i ∧ i ≤ 18 → i ∣ N) ∧ ¬19 ∣ N ∧ ¬20 ∣ N ∧ N = 12252240 :=
by {
  use 12252240,
  split,
  { intros i hi,
    have : i ∣ 12252240, sorry, },
  split,
  { have : ¬19 ∣ 12252240, sorry, },
  split,
  { have : ¬20 ∣ 12252240, sorry, },
  refl,
}

end least_possible_integer_l408_408352


namespace prosecutor_cases_knight_or_liar_l408_408304

-- Define the conditions as premises
variable (X : Prop)
variable (Y : Prop)
variable (prosecutor : Prop) -- Truthfulness of the prosecutor (true for knight, false for liar)

-- Define the statements made by the prosecutor
axiom statement1 : X  -- "X is guilty."
axiom statement2 : ¬ (X ∧ Y)  -- "Both X and Y cannot both be guilty."

-- Lean 4 statement for the proof problem
theorem prosecutor_cases_knight_or_liar (h1 : prosecutor) (h2 : ¬prosecutor) : 
  (prosecutor ∧ X ∧ ¬Y) :=
by sorry

end prosecutor_cases_knight_or_liar_l408_408304


namespace express_x_using_a_l408_408436

theorem express_x_using_a (a x : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : 0 < x) (h₄ : x ≠ 1)
  (h : sqrt (log a (a * x) + log x (a * x)) + sqrt (log a (x / a) + log x (a / x)) = 2) :
  x = a :=
sorry

end express_x_using_a_l408_408436


namespace no_sol_log_inequality_l408_408404

theorem no_sol_log_inequality :
  ∀ x : ℤ, 50 < x ∧ x < 70 → ¬ (log 5 (x - 50) + log 5 (70 - x) < 3) :=
by
  intro x hx
  sorry

end no_sol_log_inequality_l408_408404


namespace factor_x4_plus_12_l408_408233

theorem factor_x4_plus_12 : ∃ p : ℤ[X], (X^2 - 3 * X + 3) * p = X^4 + 12 := sorry

end factor_x4_plus_12_l408_408233


namespace right_triangle_BC_length_l408_408172

theorem right_triangle_BC_length
  (A B C : Type)
  (angle_A_eq_90 : angle A = 90)
  (cos_B_eq_4_div_5 : cos B = 4 / 5)
  (AB_eq_40 : AB = 40) :
  BC = 24 :=
by
  sorry

end right_triangle_BC_length_l408_408172


namespace simplify_expression_l408_408935

variable (x y z : ℝ)

theorem simplify_expression (hxz : x > z) (hzy : z > y) (hy0 : y > 0) :
  (x^z * z^y * y^x) / (z^z * y^y * x^x) = x^(z-x) * z^(y-z) * y^(x-y) :=
sorry

end simplify_expression_l408_408935


namespace factorize_expr_l408_408067

theorem factorize_expr (a x y : ℝ) : a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) :=
by
  sorry

end factorize_expr_l408_408067


namespace geometric_series_sum_l408_408009

-- Conditions
def is_geometric_series (a r : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a * r ^ n

-- The problem statement translated into Lean: Proving the sum of the series
theorem geometric_series_sum : ∃ S : ℕ → ℝ, is_geometric_series 1 (1/4) S ∧ ∑' n, S n = 4/3 :=
by
  sorry

end geometric_series_sum_l408_408009


namespace find_principal_l408_408778

noncomputable def principal_amount (A : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  A / Real.exp (r * t)

theorem find_principal : 
  principal_amount 5673981 0.1125 7.5 ≈ 2438971.57 :=
by 
  sorry

end find_principal_l408_408778


namespace EF_bisects_AD_l408_408549

theorem EF_bisects_AD
    (A B C : Point)
    (hA : Angle (angle A B C) < π / 2)
    (E F : Point)
    (hE : IsAltitude B E A C)
    (hF : IsAltitude C F A B)
    (O H : Point)
    (hO : IsCircumcenter O A B C)
    (hH : IsOrthocenter H A B C)
    (D : Point)
    (hD : OnLine D B C)
    (P : Point)
    (hP : Perpendicular (A, D) (O, H) P) :
    Bisects (E, F) (A, D) :=
sorry

end EF_bisects_AD_l408_408549


namespace milton_city_accelerated_growth_l408_408175

noncomputable def percentage_growth (start end : ℕ) : ℕ :=
  end - start

theorem milton_city_accelerated_growth :
  let milton_2000 := 3
  let milton_2010 := 9
  let milton_2020 := 18
  let milton_2030 := 35

  let rivertown_2000 := 4
  let rivertown_2010 := 7
  let rivertown_2020 := 13
  let rivertown_2030 := 20

  percentage_growth milton_2000 milton_2010 = 6 ∧
  percentage_growth milton_2010 milton_2020 = 9 ∧
  percentage_growth milton_2020 milton_2030 = 17 ∧
  percentage_growth rivertown_2000 rivertown_2010 = 3 ∧
  percentage_growth rivertown_2010 rivertown_2020 = 6 ∧
  percentage_growth rivertown_2020 rivertown_2030 = 7 →
  
  (percentage_growth milton_2000 milton_2010 < percentage_growth milton_2010 milton_2020 ∧
   percentage_growth milton_2010 milton_2020 < percentage_growth milton_2020 milton_2030) → 
  (percentage_growth rivertown_2000 rivertown_2010 ≥ percentage_growth rivertown_2010 rivertown_2020 ∧
   percentage_growth rivertown_2010 rivertown_2020 ≥ percentage_growth rivertown_2020 rivertown_2030) →
  true := 
by {
  sorry
}

end milton_city_accelerated_growth_l408_408175


namespace num_squares_in_rectangle_l408_408548

def rectangle_length : ℕ := 6
def rectangle_width : ℕ := 4
def square_side_length : ℕ := 1

theorem num_squares_in_rectangle : (rectangle_length * rectangle_width) / (square_side_length * square_side_length) = 24 :=
by
  have length_squares := rectangle_length / square_side_length
  have width_squares := rectangle_width / square_side_length
  calc
    (rectangle_length * rectangle_width) / (square_side_length * square_side_length)
      = (length_squares * width_squares) : by sorry
      = 24 : by sorry

end num_squares_in_rectangle_l408_408548


namespace units_digit_sum_of_squares_odd_integers_l408_408725

theorem units_digit_sum_of_squares_odd_integers :
  let units_digit (n : ℤ) : ℤ := n % 10
  let sum_units_digits (n : ℕ) : ℤ :=
    (List.range (2 * n + 1)).filter (λ x => x % 2 = 1).map (λ k => units_digit (k * k)).sum in
  units_digit (sum_units_digits 2023) = 5 :=
begin
  sorry
end

end units_digit_sum_of_squares_odd_integers_l408_408725


namespace total_buildings_proof_l408_408196

-- Given conditions
variables (stores_pittsburgh hospitals_pittsburgh schools_pittsburgh police_stations_pittsburgh : ℕ)
variables (stores_new hospitals_new schools_new police_stations_new buildings_new : ℕ)

-- Given values for Pittsburgh
def stores_pittsburgh := 2000
def hospitals_pittsburgh := 500
def schools_pittsburgh := 200
def police_stations_pittsburgh := 20

-- Definitions for the new city
def stores_new := stores_pittsburgh / 2
def hospitals_new := 2 * hospitals_pittsburgh
def schools_new := schools_pittsburgh - 50
def police_stations_new := police_stations_pittsburgh + 5
def buildings_new := stores_new + hospitals_new + schools_new + police_stations_new

-- Statement to prove
theorem total_buildings_proof : buildings_new = 2175 := by
  dsimp [buildings_new, stores_new, hospitals_new, schools_new, police_stations_new] 
  dsimp [stores_pittsburgh, hospitals_pittsburgh, schools_pittsburgh, police_stations_pittsburgh]
  rfl

end total_buildings_proof_l408_408196


namespace concurrence_FJ_IG_AC_l408_408653

noncomputable def incircle_touches (A B C D E F G H X I J : Point) (ω : Circle) :=
  tangent (ω) A B E ∧ 
  tangent (ω) B C F ∧ 
  tangent (ω) C D G ∧ 
  tangent (ω) D A H ∧
  lies_on_segment X A C ∧
  lies_inside X ω ∧
  intersects_on (line_through X B) ω I ∧
  intersects_on (line_through X D) ω J

theorem concurrence_FJ_IG_AC
  (A B C D E F G H X I J : Point) (ω : Circle)
  (h: incircle_touches A B C D E F G H X I J ω) :
  concurrent (line_through F J) (line_through I G) (line_through A C) :=
begin
  sorry
end

end concurrence_FJ_IG_AC_l408_408653


namespace unique_time_displays_l408_408380

def is_valid_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def is_valid_minute_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 5

def distinct_digits (digits : List ℕ) : Prop :=
  digits.nodup

def valid_time (a b c d : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_minute_digit c ∧ is_valid_digit d ∧ 
  distinct_digits [1, 0, a, b, c, d] ∧ a ≠ 1 ∧ a ≠ 0 ∧ b ≠ 1 ∧ b ≠ 0 ∧ c ≠ 1 ∧ c ≠ 0 ∧ d ≠ 1 ∧ d ≠ 0

def count_valid_times : ℕ :=
  90  -- The already-calculated number from the solution

theorem unique_time_displays : 
  ∃ n : ℕ, n = count_valid_times ∧ n = 90 := 
by
  -- We claim that there are exactly 90 such times with distinct digits within the given range.
  sorry

end unique_time_displays_l408_408380


namespace regular_polygon_min_area_regular_polygon_min_perimeter_l408_408236

noncomputable def circle_area (R : ℝ) : ℝ :=
  π * R^2

noncomputable def regular_polygon_area (n : ℕ) (R : ℝ) : ℝ :=
  (1 / 2) * n * R^2 * Real.sin (360 / n)

noncomputable def polygon_perimeter (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  ∑ i in Finset.range n, a i

theorem regular_polygon_min_area (n : ℕ) (R : ℝ) (M : ℕ → ℝ) :
  regular_polygon_area n R ≤ circle_area R - ∑ i in Finset.range n, (some_segment_area R (M i)) :=
sorry

theorem regular_polygon_min_perimeter (n : ℕ) (R : ℝ) (a : ℕ → ℝ) :
  regular_polygon_area n R ≤ polygon_perimeter n a :=
sorry

end regular_polygon_min_area_regular_polygon_min_perimeter_l408_408236


namespace find_pairs_l408_408071

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_pairs (a b : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : 
  (digit_sum (a^(b+1)) = a^b) ↔ 
  ((a = 1) ∨ (a = 3 ∧ b = 2) ∨ (a = 9 ∧ b = 1)) :=
by
  sorry

end find_pairs_l408_408071


namespace sum_arithmetic_series_mod_l408_408389

theorem sum_arithmetic_series_mod (a d n m : ℤ) (f : ℕ → ℤ) :
  a = 1 →
  d = 6 →
  n = 23 →
  m = 17 →
  (∀ k, 0 ≤ k → k < n → f k = a + k * d) →
  (∑ k in Finset.range n, f k) % m = 13 :=
by
  intros ha hd hn hm hf
  sorry

end sum_arithmetic_series_mod_l408_408389


namespace cos_product_identity_l408_408118

theorem cos_product_identity (alpha : ℝ) (h : alpha = 2 * Real.pi / 1999) :
    (∏ k in Finset.range 1000, Real.cos (k * alpha)) = 1 / (2:ℝ) ^ 999 :=
by
  sorry

end cos_product_identity_l408_408118


namespace max_points_without_right_triangle_l408_408615

theorem max_points_without_right_triangle (n : ℕ) (C : fin n → fin n → Prop) : 
  (∀ i j k l m n, ¬(C i j ∧ C k l ∧ C m n ∧ ((i = k ∧ j = l) ∨ (i = m ∧ j = n) ∨ (k = m ∧ l = n)))) 
  → (n ≤ 14) :=
sorry

end max_points_without_right_triangle_l408_408615


namespace triangle_circumcircles_l408_408985

theorem triangle_circumcircles (A B C D E F : Point)
  (h1 : AB < AC) (h2 : AC < BC)
  (hD : D ∈ line BC) (hE : E ∈ line_extension BA)
  (hBD : BD = BE) (hBE : BE = AC) (hBD' : BD = AC)
  (hF : F ∈ circumcircle BDE ∧ F ∈ circumcircle ABC) :
  BF = AF + CF :=
sorry

end triangle_circumcircles_l408_408985


namespace ratio_area_of_S_l408_408305

-- Define the vertices of equilateral triangle
def A : ℝ × ℝ := (1 / 2, Real.sqrt 3 / 2)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (1, 0)

-- Define the midpoint of AC
def E : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

-- Define the midpoint of the segment joining B and E
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- The main theorem statement
theorem ratio_area_of_S (P₁ P₂ : ℝ × ℝ) (t : ℝ)
  (startP₁ : P₁ = B)
  (startP₂ : P₂ = E)
  (same_speed : ∀ t₁ t₂, t₁ = t₂ → midpoint P₁ P₂ = midpoint (P₁ t₁) (P₂ t₂))
  : ∃ S : ℝ × ℝ → Prop,
    S.trace_midpoint P₁ P₂ t → 
    ratio_area_S_ABC S (triangle_area A B C) = 1 / 16 :=
by
  sorry

end ratio_area_of_S_l408_408305


namespace largest_number_among_three_l408_408924

theorem largest_number_among_three (h1 : Real.exp (-1 * Real.log 2 * π) < 1)
  (h2 : 1 < Real.log 2 3)
  (h3 : Real.log 2 3 < Real.log 2 π) :
  max (max (Real.exp (-1 * Real.log 2 * π)) (Real.log 2 3)) (Real.log 2 π) = Real.log 2 π :=
by
  sorry

end largest_number_among_three_l408_408924


namespace total_dreams_correct_l408_408858

def dreams_per_day : Nat := 4
def days_in_year : Nat := 365
def current_year_dreams : Nat := dreams_per_day * days_in_year
def last_year_dreams : Nat := 2 * current_year_dreams
def total_dreams : Nat := current_year_dreams + last_year_dreams

theorem total_dreams_correct : total_dreams = 4380 :=
by
  -- prime verification needed here
  sorry

end total_dreams_correct_l408_408858


namespace decreasing_interval_f_l408_408284

open Real

noncomputable def f (x : ℝ) := ln (3 - x)

theorem decreasing_interval_f :
  ∀ x, x < 3 → ∃ y, f y < f x :=
by {
  sorry
}

end decreasing_interval_f_l408_408284


namespace median_of_data_set_l408_408081

def data_set := [2, 3, 3, 4, 6, 6, 8, 8]

def calculate_50th_percentile (l : List ℕ) : ℕ :=
  if H : l.length % 2 = 0 then
    (l.get ⟨l.length / 2 - 1, sorry⟩ + l.get ⟨l.length / 2, sorry⟩) / 2
  else
    l.get ⟨l.length / 2, sorry⟩

theorem median_of_data_set : calculate_50th_percentile data_set = 5 :=
by
  -- Insert the proof here
  sorry

end median_of_data_set_l408_408081


namespace find_larger_part_l408_408342

theorem find_larger_part (x y : ℕ) (h1 : x + y = 52) (h2 : 10 * x + 22 * y = 780) : x = max x y := by
  let x_val := 30
  have h : x = x_val := sorry  -- Placeholder for the actual proof
  rw [h]
  exact rfl

end find_larger_part_l408_408342


namespace sum_of_first_9_terms_l408_408137

open Nat

theorem sum_of_first_9_terms 
  (a : ℕ → ℝ) 
  (arithmetic_seq : ∃ d, ∀ n, a (n + 1) = a n + d)
  (a5_eq_3 : a 5 = 3) : 
  (Finset.range 9).sum (λ n, a (n + 1)) = 27 := 
  sorry

end sum_of_first_9_terms_l408_408137


namespace particle_position_90_l408_408357

def initial_position : ℂ := 8

def omega : ℂ := (1/2 : ℂ) + (complex.I * (real.sqrt 3 / 2))

def move (z : ℂ) : ℂ := omega * z + 8

def particle_position (n : ℕ) : ℂ :=
  nat.rec_on n initial_position (λ n z, move z)

theorem particle_position_90 : particle_position 90 = 8 :=
by sorry

end particle_position_90_l408_408357


namespace sequence_formula_l408_408100

noncomputable def a (n : ℕ) : ℝ := 2 - 1 / (2^n)

def S (n : ℕ) : ℝ := (finset.range n).sum (λ k, a (k + 1))

theorem sequence_formula (n : ℕ) (hn : 0 < n) :
  a n = 2 - (1 / (2^n)) := sorry

end sequence_formula_l408_408100


namespace projection_a_onto_b_is_sqrt5_l408_408927

/- Define the vectors a and b -/
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (-2, 4)

/- Define the dot product of two vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/- Define the magnitude of a vector -/
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

/- Define the projection of a onto b -/
def projection (a b : ℝ × ℝ) : ℝ :=
  dot_product a b / magnitude b

/- The theorem to prove -/
theorem projection_a_onto_b_is_sqrt5 : projection a b = Real.sqrt 5 :=
by sorry

end projection_a_onto_b_is_sqrt5_l408_408927


namespace irreducible_f_l408_408595

def f (n : ℕ) (x : ℤ) : ℤ := x^n + 5 * x^(n - 1) + 3

theorem irreducible_f (n : ℕ) (hn : n > 1) : Irreducible (f n : ℤ[X]) :=
  sorry

end irreducible_f_l408_408595


namespace sufficient_not_necessary_l408_408746

namespace ProofExample

variable {x : ℝ}

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B : Set ℝ := {x | x < 2}

-- Theorem: "1 < x < 2" is a sufficient but not necessary condition for "x < 2" to hold.
theorem sufficient_not_necessary : 
  (∀ x, 1 < x ∧ x < 2 → x < 2) ∧ ¬(∀ x, x < 2 → 1 < x ∧ x < 2) := 
by
  sorry

end ProofExample

end sufficient_not_necessary_l408_408746


namespace sequence_x_y_sum_l408_408043

theorem sequence_x_y_sum (r : ℝ) (x y : ℝ)
  (h₁ : r = 1 / 4)
  (h₂ : x = 256 * r)
  (h₃ : y = x * r) :
  x + y = 80 :=
by
  sorry

end sequence_x_y_sum_l408_408043


namespace find_angle_A_find_bc_l408_408490

-- Definition of the conditions
variable (a b c A B C : ℝ)

-- Given conditions as hypotheses
def cond1 : Prop := a * Real.cos C + Real.sqrt 3 * a * Real.sin C - b - c = 0
def cond2 (a : ℝ) : Prop := a = 2
def cond3 (b c : ℝ) : Prop := (1/2) * b * c * Real.sin A = Real.sqrt 3

-- Lean 4 statement for part (1)
theorem find_angle_A 
  (h : cond1 a b c) : A = 60 := 
sorry

-- Lean 4 statement for part (2)
theorem find_bc 
  (h1 : cond2 a) 
  (h2 : cond3 b c) 
  (h3 : A = 60) :
  b = 2 ∧ c = 2 := 
sorry

end find_angle_A_find_bc_l408_408490


namespace brilliant_numbers_count_l408_408736

def double_or_subtract (n : ℕ) : ℕ :=
  if n <= 20 then 2 * n else n - 10

def is_brilliant_sequence (G : ℕ) : Prop :=
  ∀ n, (iterate double_or_subtract n G) ≠ 20

def count_brilliant_numbers : ℕ :=
  (60 - (Nat.div 60 5)) -- Subtracting the 12 multiples of 5 from the range 1 to 60.

theorem brilliant_numbers_count : count_brilliant_numbers = 48  :=
begin
  sorry, -- proof steps
end

end brilliant_numbers_count_l408_408736


namespace sequence_value_l408_408039

theorem sequence_value (r x y : ℝ) 
  (h1 : r = 1/4) 
  (h2 : x = 256 * r)
  (h3 : y = x * r) : 
  x + y = 80 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end sequence_value_l408_408039


namespace find_number_l408_408848

theorem find_number (x : ℝ) (h : x - (3/5) * x = 60) : x = 150 :=
by
  sorry

end find_number_l408_408848


namespace membership_codes_div_10_eq_312_l408_408757

open Finset
open Multiset

-- Definitions based on problem conditions
def chars := {'B', 'E', 'T', 'A', '2', '0', '3'}
def chars_available : Multiset Char := ['B', 'E', 'T', 'A', '2', '2', '0', '3']

def valid_codes (code : Multiset Char) : Prop :=
  code.card = 5 ∧ code ≤ chars_available

/-- Prove that the number of valid membership codes divided by 10 equals 312 -/
theorem membership_codes_div_10_eq_312 :
  (Multiset.card {code | valid_codes (↑code : Multiset Char)}.to_finset) / 10 = 312 :=
sorry

end membership_codes_div_10_eq_312_l408_408757


namespace necessary_condition_for_acute_angle_necessary_but_not_sufficient_condition_l408_408521

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (2, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 2)

-- Define the dot product calculation
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Conditionally state that x > -3 is necessary for an acute angle
theorem necessary_condition_for_acute_angle (x : ℝ) :
  dot_product vector_a (vector_b x) > 0 → x > -3 := by
  sorry

-- Define the theorem for necessary but not sufficient condition
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x > -3) → (dot_product vector_a (vector_b x) > 0 ∧ x ≠ 4 / 3) := by
  sorry

end necessary_condition_for_acute_angle_necessary_but_not_sufficient_condition_l408_408521


namespace air_quality_probabilities_average_exercisers_in_park_relationship_exercise_air_quality_l408_408774

-- Condition definitions
def exercise_data : ℕ → ℕ → ℕ
| 1 1 := 2
| 1 2 := 16
| 1 3 := 25
| 2 1 := 5
| 2 2 := 10
| 2 3 := 12
| 3 1 := 6
| 3 2 := 7
| 3 3 := 8
| 4 1 := 7
| 4 2 := 2
| 4 3 := 0
| _ _ := 0

-- Part 1: Probabilities of air quality levels
theorem air_quality_probabilities :
  (∑ x in [1, 2, 3], exercise_data 1 x) / 100 = 43 / 100 ∧
  (∑ x in [1, 2, 3], exercise_data 2 x) / 100 = 27 / 100 ∧
  (∑ x in [1, 2, 3], exercise_data 3 x) / 100 = 21 / 100 ∧
  (∑ x in [1, 2, 3], exercise_data 4 x) / 100 = 9 / 100 :=
sorry

-- Part 2: Estimated average number of people exercising
def average_exercise : ℝ :=
(1 / 100 : ℝ) * (∑ x in [1, 2, 3], exercise_data 1 x * 100 + exercise_data 2 x * 300 + exercise_data 3 x * 500)

theorem average_exercisers_in_park :
  average_exercise = 350 :=
sorry

-- Part 3: Relationship between exercise times and air quality
def contingency_good_air_quality :=
  exercise_data 1 1 + exercise_data 2 1 + exercise_data 1 2 + exercise_data 2 2

def contingency_poor_air_quality :=
  exercise_data 3 1 + exercise_data 4 1 + exercise_data 3 2 + exercise_data 4 2

def contingency_exercise_leq_400 :=
  contingency_good_air_quality + contingency_poor_air_quality

def contingency_exercise_gt_400 :=
  (∑ x in [3], exercise_data 1 x + exercise_data 2 x + exercise_data 3 x + exercise_data 4 x)

theorem relationship_exercise_air_quality :
  let K2 := 100 * ((33 * 8 - 37 * 22)^2 : ℝ) / (70 * 30 * 55 * 45) in K2 > 3.841 :=
sorry

end air_quality_probabilities_average_exercisers_in_park_relationship_exercise_air_quality_l408_408774


namespace find_metal_molecular_weight_l408_408846

noncomputable def molecular_weight_of_metal (compound_mw: ℝ) (oh_mw: ℝ) : ℝ :=
  compound_mw - oh_mw

theorem find_metal_molecular_weight :
  let compound_mw := 171.00
  let oxygen_mw := 16.00
  let hydrogen_mw := 1.01
  let oh_ions := 2
  let oh_mw := oh_ions * (oxygen_mw + hydrogen_mw)
  molecular_weight_of_metal compound_mw oh_mw = 136.98 :=
by
  sorry

end find_metal_molecular_weight_l408_408846


namespace tensor_identity_l408_408399

def tensor (a b : ℝ) : ℝ := a^3 - b

theorem tensor_identity (a : ℝ) : tensor a (tensor a (tensor a a)) = a^3 - a :=
by
  sorry

end tensor_identity_l408_408399


namespace next_valid_number_after_1818_l408_408678

open Nat

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem next_valid_number_after_1818 : 
  ∀ (a b : ℕ), 10 * a + b > 18 → 18 * (10 * a + b) = 1832 → isPerfectSquare (18 * (10 * a + b)) → 
  10 * 10 * 8 + 10 * a + b = 1832 :=
by {
  intros,
  sorry
}

end next_valid_number_after_1818_l408_408678


namespace parallel_lines_a_value_l408_408654

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, ((a + 1) * x + 3 * y + 3 = 0) → (x + (a - 1) * y + 1 = 0)) → a = -2 :=
by
  sorry

end parallel_lines_a_value_l408_408654


namespace Number_of_Divisors_of_a_cube_l408_408346

noncomputable def a (p1 p2 : ℕ) (α1 α2 : ℕ) : ℕ := p1^α1 * p2^α2

theorem Number_of_Divisors_of_a_cube (p1 p2 α1 α2 : ℕ) (h1 : p1.prime) (h2 : p2.prime) (h_distinct : p1 ≠ p2)
    (h_α1_pos : 0 < α1) (h_α2_pos : 0 < α2) (h : (2 * α1 + 1) * (2 * α2 + 1) = 35) :
    let a := p1^α1 * p2^α2 in
    (3 * α1 + 1) * (3 * α2 + 1) = 70 := by
  sorry

end Number_of_Divisors_of_a_cube_l408_408346


namespace find_z_l408_408540

theorem find_z 
  {x y z : ℕ}
  (hx : x = 4)
  (hy : y = 7)
  (h_least : x - y - z = 17) : 
  z = 14 :=
by
  sorry

end find_z_l408_408540


namespace cos_double_angle_identity_l408_408093

theorem cos_double_angle_identity (α : ℝ) (h : sin(α + π/3) = 1/3) : 
  cos (2 * α - π / 3) = -7 / 9 := 
sorry

end cos_double_angle_identity_l408_408093


namespace distinct_real_roots_equal_real_roots_no_real_roots_l408_408462

-- Define the discriminant function based on m
def discriminant (m : ℝ) : ℝ := 16 * m + 12

-- Prove the conditions for m based on the discriminant
theorem distinct_real_roots (m : ℝ) : discriminant m > 0 ↔ m ∈ set.Ioo (-3 / 4) (-1 / 2) ∪ set.Ioi (-1 / 2) :=
by sorry

theorem equal_real_roots (m : ℝ) : discriminant m = 0 ↔ m = -3 / 4 :=
by sorry

theorem no_real_roots (m : ℝ) : discriminant m < 0 ↔ m ∈ set.Iio (-3 / 4) :=
by sorry

end distinct_real_roots_equal_real_roots_no_real_roots_l408_408462


namespace hexagon_area_problem_l408_408201

noncomputable def area_of_hexagon (A B C D E F : ℝ × ℝ) : ℝ := 
by
  sorry

theorem hexagon_area_problem :
  let A := (0, 0)
  let B := (27, 3)
  let C := (x1, 6)
  let D := (x2, 9)
  let E := (x3, 12)
  let F := (12, 15)
  in
  ∠FAB = 120 ∧
  (A, B, C, D, E, F).coord_y.distinct ∧
  AB.parallel CD ∧
  BC.parallel EF ∧
  DE.parallel FA ∧
  area_of_hexagon A B C D E F = 729 :=
begin
  sorry
end

end hexagon_area_problem_l408_408201


namespace modulus_product_l408_408426

open Complex

theorem modulus_product :
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  (Complex.abs (z1 * z2)) = Real.sqrt 8450 :=
by
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  sorry

end modulus_product_l408_408426


namespace bin_to_oct_l408_408822

theorem bin_to_oct (n : ℕ) (hn : n = 0b11010) : n = 0o32 := by
  sorry

end bin_to_oct_l408_408822


namespace compute_limit_l408_408013

noncomputable def limit := (λ (f : ℝ → ℝ) (a L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x, |x - a| < δ → |f x - L| < ε)

theorem compute_limit :
  limit (λ Δx, (Real.sin (Real.pi / 6 + Δx) - Real.sin (Real.pi / 6)) / Δx) 0 (Real.sqrt 3 / 2) :=
by
  sorry

end compute_limit_l408_408013


namespace chickens_after_years_l408_408610

theorem chickens_after_years : 
  ∀ (initial_chickens annual_increase years : ℕ),
  initial_chickens = 550 →
  annual_increase = 150 →
  years = 9 →
  initial_chickens + (annual_increase * years) = 1900 :=
by
  intros initial_chickens annual_increase years h1 h2 h3
  rw [h1, h2, h3]
  rfl

end chickens_after_years_l408_408610


namespace digits_arrangement_count_l408_408974

theorem digits_arrangement_count : 
  let digits := [4, 5, 5, 2, 0, 1] in
  (∑ p in (multisets.permutations digits), if p.head ≠ 0 then 1 else 0) = 300 :=
by
  -- Skipping the actual proof
  sorry

end digits_arrangement_count_l408_408974


namespace minimal_number_with_2023_divisors_has_m_plus_k_equal_9222_l408_408280

theorem minimal_number_with_2023_divisors_has_m_plus_k_equal_9222 :
  ∃ (m k : ℕ), (∃ n : ℕ, (∀ d : ℕ, d ∣ n → d ≤ n) ∧
                n.factors.prod.card = 2023 ∧
                n = m * 6^k ∧ ¬ (6 ∣ m) ∧
                m + k = 9222) :=
begin
  sorry
end

end minimal_number_with_2023_divisors_has_m_plus_k_equal_9222_l408_408280


namespace carla_benjamin_ratio_l408_408385

def Benjamin_eggs : ℕ := 6
def Trisha_eggs (B_eggs : ℕ) : ℕ := B_eggs - 4
def Carla_eggs (mult : ℕ) (B_eggs : ℕ) : ℕ := mult * B_eggs

theorem carla_benjamin_ratio 
  (B_eggs : ℕ) 
  (T_eggs : ℕ)
  (C_eggs : ℕ) 
  (total_eggs : ℕ) 
  (x : ℕ) 
  (hb : B_eggs = 6) 
  (ht : T_eggs = Trisha_eggs B_eggs) 
  (hc : C_eggs = Carla_eggs x B_eggs) 
  (htotal: B_eggs + T_eggs + C_eggs = total_eggs)
  (total_is_26 : total_eggs = 26) : 
  x = 3 := 
by {
  rw [hb, ht, hc] at *,
  sorry
}

end carla_benjamin_ratio_l408_408385


namespace find_angle_A_find_area_triangle_l408_408961

variable {a b c A B C : ℝ}
variable {area : ℝ}
variable (triangleABC : Type) [Triangle triangleABC]

-- Define the given conditions
def condition1 (a b : ℝ) (sin B cos A : ℝ) : Prop :=
  a * sin B = sqrt 3 * b * cos A

def condition2 (a b c : ℝ) : (a = sqrt 7) ∧ (b = 2) ∧ (A = π / 3) ∧ (c = 3) ∧ ∃ area, area = 3 * sqrt 3 / 2

-- Question 1: Proving the measure of angle A
theorem find_angle_A (h : condition1 a b (sin B) (cos A)) : A = π / 3 :=
sorry

-- Question 2: Proving the area of the triangle given specific values for a, b, and A
theorem find_area_triangle (h1 : a = sqrt 7) (h2 : b = 2) (h3 : A = π / 3) (h4 : c = 3) : 
  ∃ area, area = 3 * sqrt 3 / 2 :=
sorry

end find_angle_A_find_area_triangle_l408_408961


namespace evaluate_magnitude_product_l408_408418

-- Definitions of complex numbers
def z1 := Complex.mk 7 (-4)
def z2 := Complex.mk 3 11

-- The magnitude of z1
def magnitude_z1 := Complex.abs z1

-- The magnitude of z2
def magnitude_z2 := Complex.abs z2

-- Lean 4 statement expressing the problem and its final answer
theorem evaluate_magnitude_product : Complex.abs (z1 * z2) = Real.sqrt 8450 := by
  sorry

end evaluate_magnitude_product_l408_408418


namespace sequence_value_l408_408042

theorem sequence_value (r x y : ℝ) 
  (h1 : r = 1/4) 
  (h2 : x = 256 * r)
  (h3 : y = x * r) : 
  x + y = 80 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end sequence_value_l408_408042


namespace largest_power_of_2_dividing_product_l408_408576

-- Define the product of the first 50 positive even integers
def product_first_50_even : ℕ := List.prod (List.map (λ n, 2 * n) (List.range 50).map (λ i, i + 1))

-- We need to prove that the largest integer j such that 
-- product_first_50_even is divisible by 2^j is 97
theorem largest_power_of_2_dividing_product :
  ∃ j : ℕ, (2^j ∣ product_first_50_even ∧ (∀ k, 2^k ∣ product_first_50_even → k ≤ j)) ∧ j = 97 :=
sorry

end largest_power_of_2_dividing_product_l408_408576


namespace triangle_area_l408_408707

theorem triangle_area : 
  let l1 := λ x : ℝ, 3 * x - 6,
      l2 := λ x : ℝ, -4 * x + 24,
      y_axis_intersect := (0 : ℝ),
      x_intersect := 30 / 7,
      y_intersect := 48 / 7,
      base := 30,
      height := 30 / 7 in
  (1 / 2) * base * height = 450 / 7 :=
by
  sorry

end triangle_area_l408_408707


namespace find_slope_of_dividing_line_l408_408975

/-- The vertices of the rectangle and the triangle forming the region in the xy-plane. --/
def rectangle : set (ℝ × ℝ) := {p | (p = (0,0)) ∨ (p = (0,2)) ∨ (p = (4,2)) ∨ (p = (4,0))}
def triangle : set (ℝ × ℝ) := {p | (p = (1,2)) ∨ (p = (3,2)) ∨ (p = (2,4))}

/-- The slope of the line through the origin (0,0) and the vertex (2,4) of the triangle. --/
def slope (p₁ p₂ : ℝ × ℝ) : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)

theorem find_slope_of_dividing_line :
  slope (0, 0) (2, 4) = 2 :=
by
  -- We know the slope is calculated as (y2 - y1) / (x2 - x1)
  calc slope (0, 0) (2, 4) = (4 - 0) / (2 - 0) : rfl
                          ... = 4 / 2        : rfl
                          ... = 2            : rfl
sorry

end find_slope_of_dividing_line_l408_408975


namespace binary_to_octal_conversion_l408_408824

-- Define the binary number 11010 in binary
def bin_value : ℕ := 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0

-- Define the octal value 32 in octal as decimal
def oct_value : ℕ := 3 * 8^1 + 2 * 8^0

-- The theorem to prove the binary equivalent of 11010 is the octal 32
theorem binary_to_octal_conversion : bin_value = oct_value :=
by
  -- Skip actual proof
  sorry

end binary_to_octal_conversion_l408_408824


namespace s_point_value_l408_408591

def has_s_point (f g : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = g x₀ ∧ deriv f x₀ = deriv g x₀

noncomputable def f (a : ℝ) : ℝ → ℝ := λ x, a * x^2 - 1
noncomputable def g (a : ℝ) : ℝ → ℝ := λ x, log (a * x)

theorem s_point_value (a : ℝ) (x₀ : ℝ) (h : has_s_point (f a) (g a) x₀) : a = 2 / real.exp(1) :=
sorry

end s_point_value_l408_408591


namespace harkamal_purchased_9_kg_mangoes_l408_408523

/-- Harkamal purchased 8 kg of grapes at 70 per kg, 
and some kg of mangoes at 55 per kg. He paid 1055 to the shopkeeper.
Prove that he purchased 9 kg of mangoes. --/
theorem harkamal_purchased_9_kg_mangoes :
  ∃ (m : ℕ), (8 * 70 + m * 55 = 1055) ∧ (m = 9) :=
by
  use 9
  simp
  rw [Nat.mul_comm, Nat.mul_comm 8 70]
  norm_num
  sorry

end harkamal_purchased_9_kg_mangoes_l408_408523


namespace mass_percentage_of_N_is_9_66_l408_408078

-- Define the mass percentage condition of nitrogen in the compound
def mass_percentage_N (compound_mass : ℕ → ℝ) (percentage_N : ℝ) : Prop :=
  percentage_N = 9.66

-- Theorem stating the mass percentage of nitrogen in the compound as given
theorem mass_percentage_of_N_is_9_66 :
  ∀ (compound_mass : ℕ → ℝ), mass_percentage_N compound_mass 9.66 :=
begin
  intros compound_mass,
  unfold mass_percentage_N,
  exact rfl,
end

end mass_percentage_of_N_is_9_66_l408_408078


namespace consecutive_differences_equal_l408_408362

-- Define the set and the condition
def S : Set ℕ := {n : ℕ | n > 0}

-- Condition that for any two numbers a and b in S with a > b, at least one of a + b or a - b is also in S
axiom h_condition : ∀ a b : ℕ, a ∈ S → b ∈ S → a > b → (a + b ∈ S ∨ a - b ∈ S)

-- The main theorem that we want to prove
theorem consecutive_differences_equal (a : ℕ) (s : Fin 2003 → ℕ) 
  (hS : ∀ i, s i ∈ S)
  (h_ordered : ∀ i j, i < j → s i < s j) :
  ∃ (d : ℕ), ∀ i, i < 2002 → (s (i + 1)) - (s i) = d :=
sorry

end consecutive_differences_equal_l408_408362


namespace triangleABC_proof_l408_408634

noncomputable def triangle_proof (ABC : Triangle ℝ) (H : Point ℝ) (M N : Point ℝ)
  (CF AE : Line ℝ) (FM EN : Real) (angle_ABC : Real) (area_ABC : Real) (circumradius : Real) : Prop :=
  ∃ (AC BC AB : ℝ) (A B C F E : Point ℝ), 
    Altitude CF ABC C ∧ Altitude AE ABC A ∧
    Intersection CF AE H ∧ 
    Midpoint M A H ∧ Midpoint N C H ∧ 
    FM = 1 ∧ EN = 7 ∧ 
    Parallel FM EN ∧ 
    Acute ABC ∧ 
    angle_ABC = 60 ∧ 
    area_ABC = 45 * √3 ∧ 
    circumradius = 2 * √19

theorem triangleABC_proof : ∀ (ABC : Triangle ℝ) (H : Point ℝ) (M N : Point ℝ)
  (CF AE : Line ℝ), ∃ (FM EN : Real) (angle_ABC area_ABC circumradius : Real), 
  triangle_proof ABC H M N CF AE FM EN angle_ABC area_ABC circumradius := 
begin
  /- proof construction with the given conditions will be done here -/
  sorry
end

end triangleABC_proof_l408_408634


namespace max_possible_k_is_813_l408_408857

noncomputable def max_possible_k : ℕ :=
  let S := finset.range 2024
  ∃ (k : ℕ) (pairs : finset (ℕ × ℕ)), 
    (∀ p ∈ pairs, p.1 < p.2) ∧
    (pairs.card = k) ∧
    (pairs.val.nodup) ∧
    (∀ p₁ p₂ ∈ pairs, p₁ ≠ p₂ → (p₁.1 + p₁.2 ≠ p₂.1 + p₂.2)) ∧
    (∀ p ∈ pairs, p.1 + p.2 ≤ 2033) ∧
    k ≤ 813

theorem max_possible_k_is_813 : max_possible_k = 813 :=
sorry

end max_possible_k_is_813_l408_408857


namespace value_of_c_infinite_solutions_l408_408463

theorem value_of_c_infinite_solutions (c : ℝ) :
  (∀ y : ℝ, 3 * (5 + 2 * c * y) = 18 * y + 15) ↔ (c = 3) :=
by
  sorry

end value_of_c_infinite_solutions_l408_408463


namespace highest_degree_term_correct_linear_term_coefficient_correct_l408_408277

noncomputable def P (x y : ℝ) : ℝ :=
  - (4 / 5) * x^2 * y + (2 / 3) * x^4 * y^2 - x + 1

def highestDegreeTerm (x y : ℝ) : ℝ :=
  (2 / 3) * x^4 * y^2

def linearCoefficient : ℝ :=
  -1

theorem highest_degree_term_correct (x y : ℝ) :
  exists (term : ℝ), ∀ (p q : ℝ), 
    (term = highestDegreeTerm p q) →
    (P p q = - (4 / 5) * p^2 * q + term - p + 1) :=
by
  intros
  sorry

theorem linear_term_coefficient_correct :
  ∀ (x : ℝ), 
    coefficient_of_linear_term (P x 0) = linearCoefficient :=
by
  intros
  sorry

end highest_degree_term_correct_linear_term_coefficient_correct_l408_408277


namespace relationship_among_abc_l408_408487

noncomputable def a := Real.logBase 4 9
noncomputable def b := Real.logBase (1 / 3) 2
noncomputable def c := (1 / 2) ^ (-4)

theorem relationship_among_abc : b < a ∧ a < c :=
by
  -- Translate the conditions into Lean proofs
  have h1 : 1 < a := by sorry
  have h2 : a < 2 := by sorry
  have h3 : b < 0 := by sorry
  have h4 : c = 16 := by sorry
  -- Combining all the inequalities to prove the required relationship.
  exact ⟨h3.trans h1, h2.trans_le (le_of_eq h4.symm)⟩

end relationship_among_abc_l408_408487


namespace area_complex_set_l408_408094

noncomputable def complex_set_M (n : ℕ) : Set ℂ :=
  {z : ℂ | ∑ k in Finset.range(n), 1 / complex.abs (z - k) ≥ 1 }

theorem area_complex_set (n : ℕ) (h : n > 0) : 
  ∃ (A : ℝ), ∀ z ∈ complex_set_M n, ∃ r > 0, complex.abs z < r ∧ 
  A ≥ (π / 12) * (11 * n^2 + 1) :=
sorry

end area_complex_set_l408_408094


namespace probability_target_hit_l408_408750

theorem probability_target_hit (pA pB : ℝ) (hA : pA = 1 / 2) (hB : pB = 1 / 3) :
  (1 - (1 - pA) * (1 - pB)) = 2 / 3 :=
by {
  rw [hA, hB],
  sorry
}

end probability_target_hit_l408_408750


namespace distinguishable_triangles_count_l408_408299

-- We define the problem based on the conditions and the expected outcome.

def num_distinguishable_triangles (num_colors : ℕ) : ℕ :=
  let all_same := num_colors,
      two_same_one_diff := num_colors * (num_colors - 1),
      all_diff := Nat.choose num_colors 3
  in num_colors * (all_same + two_same_one_diff + all_diff)

theorem distinguishable_triangles_count :
  num_distinguishable_triangles 8 = 960 := by
    -- Proof omitted
    sorry

end distinguishable_triangles_count_l408_408299


namespace minimum_n_l408_408370

theorem minimum_n (n : ℕ) (problems : fin 15 → fin n → ℕ) (h1 : n > 12)
  (h2 : ∀ s : finset (fin n), s.card = 12 → 36 ≤ s.sum (λ i, finset.univ.sum (λ j, problems j i))) :
  n ≥ 15 :=
by
  sorry

end minimum_n_l408_408370


namespace units_digit_of_sum_of_squares_of_first_2023_odd_integers_is_5_l408_408718

theorem units_digit_of_sum_of_squares_of_first_2023_odd_integers_is_5 :
  (sum (fun n : ℕ => ((2 * n + 1) ^ 2) % 10) (range 2023)) % 10 = 5 :=
sorry

end units_digit_of_sum_of_squares_of_first_2023_odd_integers_is_5_l408_408718


namespace center_of_symmetry_exists_no_common_points_cos_value_l408_408906

noncomputable def f (x : ℝ) := 2 * cos (x - π / 3) + 2 * sin (3 * π / 2 - x) + 1

-- (1) Prove the center of symmetry
theorem center_of_symmetry_exists (k : ℤ) : 
    ∃ c, c = (π / 6 + k * π, 1) :=
by sorry

-- (2) Prove the range of a when the line y = a has no common points with the graph of f(x)
theorem no_common_points (a : ℝ) : 
    (∀ x : ℝ, f x ≠ a) ↔ (a < -1 ∨ a > 3) :=
by sorry

-- (3) Given f(x) = 6/5, show the value of cos(2x - π / 3)
theorem cos_value (x : ℝ) (hx : f x = 6 / 5) : 
    cos (2 * x - π / 3) = 49 / 50 :=
by sorry

end center_of_symmetry_exists_no_common_points_cos_value_l408_408906


namespace reconstruct_pentagon_l408_408884

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

noncomputable def p_seq : V → V → V → V → V → V :=
  λ A' B' C' D' E', (1/32 : ℝ) • A' + (1/16 : ℝ) • B' + (1/8 : ℝ) • C' + (1/4 : ℝ) • D' + (1/2 : ℝ) • E'

theorem reconstruct_pentagon (A A' B' C' D' E' : V) :
  A = p_seq A' B' C' D' E' :=
  sorry

end reconstruct_pentagon_l408_408884


namespace proj_neg2v_w_l408_408206

-- Definitions of vector v, w and their projection
variable (v w : ℝ^3)

-- Given condition
def condition_proj_v_w : (proj w v = ⟨1, 0, -3⟩) := by sorry

-- Statement to prove
theorem proj_neg2v_w :
  proj w (-2 • v) = ⟨-2, 0, 6⟩ :=
by sorry

end proj_neg2v_w_l408_408206


namespace Adam_Smith_inheritance_amount_l408_408230

theorem Adam_Smith_inheritance_amount (x : ℝ) (h1 : 0.25 * x) (h2 : 0.15 * (x - 0.25 * x)) (h3 : 0.25 * x + 0.15 * (0.75 * x) = 15000) : x = 41379 :=
by 
  have h4 : 0.15 * (0.75 * x) = 0.1125 * x := by ring
  rw h4 at h3
  have h5 : 0.25 * x + 0.1125 * x = 0.3625 * x := by ring
  rw h5 at h3
  sorry

end Adam_Smith_inheritance_amount_l408_408230


namespace max_moving_segment_length_l408_408477

theorem max_moving_segment_length (A B C D : Type) [MetricSpace A] (AB AC : ℝ) (hBAC : ∠ BAC = 90) (hAB : AB = 3) (hAC : AC = 4) :
    (max_length_of_moving_segment A B C D hBAC hAB hAC) = 12 / 5 :=
by
  sorry

end max_moving_segment_length_l408_408477


namespace sqrt13_int_part_and_decimal_part_l408_408210

theorem sqrt13_int_part_and_decimal_part (m n : ℝ) (h_m : m = ⌊sqrt 13⌋) (h_n : n = sqrt 13 - ⌊sqrt 13⌋) :
  (m - n)^2 = 49 - 12 * sqrt 13 :=
by 
  have h1 : sqrt 13 > (3 : ℝ),
  { sorry },
  have h2 : sqrt 13 < (4 : ℝ),
  { sorry },
  have h3 : m = 3,
  { sorry },
  have h4 : n = sqrt 13 - 3,
  { sorry },
  have h5 : m - n = 6 - sqrt 13,
  { sorry },
  have h6 : (6 - sqrt 13)^2 = 49 - 12 * sqrt 13,
  { sorry },
  sorry

end sqrt13_int_part_and_decimal_part_l408_408210


namespace find_a_for_perpendicular_lines_l408_408499

theorem find_a_for_perpendicular_lines :
  ∀ a : ℝ,
  (∀ x y : ℝ, x + a * y + 2 = 0 → 2 * x + 3 * y + 1 = 0 → 
  ((-1 / a) = ⊥ → (a = -2 / 3))) :=
by
  sorry

end find_a_for_perpendicular_lines_l408_408499


namespace max_three_primes_l408_408325

def is_prime (n : ℕ) : Prop := nat.prime n

theorem max_three_primes (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m > n) : 
  (is_prime (m + n) ∨ is_prime (m - n) ∨ is_prime (m * n) ∨ is_prime (m / n)) → 
  (is_prime (m + n) ∧ is_prime (m - n) ∧ is_prime (m * n) ∧ is_prime (m / n) → false) ∧
  (3 ≤ finset.card {x | x ∈ [{m+n, m-n, m*n, m / n}] ∧ is_prime x}) := sorry

end max_three_primes_l408_408325


namespace x_intercept_l408_408840

theorem x_intercept (x y : ℝ) (h : 4 * x - 3 * y = 24) (hy : y = 0) : (x, y) = (6, 0) :=
by {
  sorry
}

end x_intercept_l408_408840


namespace factor_difference_of_squares_l408_408066

theorem factor_difference_of_squares (x : ℝ) : 36 - 9 * x^2 = 9 * (2 - x) * (2 + x) :=
by
  sorry

end factor_difference_of_squares_l408_408066


namespace count_even_three_digit_numbers_l408_408248

-- Defining the sets of digits
def even_digits : Finset ℕ := {0, 2, 4}
def odd_digits : Finset ℕ := {1, 3}

-- Defining the main theorem to show the count of valid three-digit even numbers
theorem count_even_three_digit_numbers : 
  (Finset.card (even_digits ×ₙ even_digits ×ₙ odd_digits) +
   Finset.card (even_digits ×ₙ odd_digits ×ₙ even_digits) +
   Finset.card (odd_digits ×ₙ even_digits ×ₙ even_digits)) = 20 :=
sorry

end count_even_three_digit_numbers_l408_408248


namespace geometric_arithmetic_sequence_sum_l408_408098

theorem geometric_arithmetic_sequence_sum {a b : ℕ → ℝ} (q : ℝ) (n : ℕ) 
(h1 : a 2 = 2)
(h2 : a 2 = 2)
(h3 : 2 * (a 3 + 1) = a 2 + a 4)
(h4 : ∀ (n : ℕ), (a (n + 1)) = a 0 * q ^ (n + 1))
(h5 : b n = n * (n + 1)) :
a 8 + (b 8 - b 7) = 144 :=
by { sorry }

end geometric_arithmetic_sequence_sum_l408_408098


namespace irreducible_f_l408_408594

def f (n : ℕ) (x : ℤ) : ℤ := x^n + 5 * x^(n - 1) + 3

theorem irreducible_f (n : ℕ) (hn : n > 1) : Irreducible (f n : ℤ[X]) :=
  sorry

end irreducible_f_l408_408594


namespace problem_statement_l408_408886

variable {θ : Real}

def p : Prop := ∀ x : Real, x^2 - 2 * x * Real.sin θ + 1 ≥ 0

def q : Prop := ∀ α β : Real, Real.sin (α + β) ≤ Real.sin α + Real.sin β

theorem problem_statement : p ∧ ¬q :=
by
  have hp : p := 
    sorry -- p's proof
  have hq : ¬q := 
    sorry -- q's proof
  exact ⟨hp, hq⟩

end problem_statement_l408_408886


namespace problem_part_2_l408_408903

def f (x : ℝ) : ℝ := | x - 1 | + | x + 1 |

noncomputable def P : set ℝ := { x | x > 2 ∨ x < -2 }

theorem problem_part_2 (m n : ℝ) (hm : m ∈ P) (hn : n ∈ P) : 
  | m * n + 4 | > 2 * | m + n | :=
sorry

end problem_part_2_l408_408903


namespace cost_of_tax_free_items_l408_408990

theorem cost_of_tax_free_items (total_cost : ℝ) (tax_40_percent : ℝ) 
  (tax_30_percent : ℝ) (discount : ℝ) : 
  (total_cost = 120) →
  (tax_40_percent = 0.4 * total_cost) →
  (tax_30_percent = 0.3 * total_cost) →
  (discount = 0.05 * tax_30_percent) →
  (tax-free_items = total_cost - (tax_40_percent + (tax_30_percent - discount))) → 
  tax_free_items = 36 :=
by sorry

end cost_of_tax_free_items_l408_408990


namespace probability_set_A_on_Saturday_l408_408699

-- Define the probability function P for each day
def P : ℕ → ℚ
| 1 := 1             -- P_1 = 1 since set A is used on Monday
| 2 := 0             -- P_2 = 0 since set A was just used on Monday
| 3 := 1 / 3         -- P_3 = 1 / 3 since there are three equally likely sets for Wednesday
| n := (1 - P (n-1)) * 1 / 3   -- Recurrence relation for P_n for n > 3

-- Prove the probability of using set A on Saturday
theorem probability_set_A_on_Saturday : P 6 = 20 / 81 :=
by { sorry }

end probability_set_A_on_Saturday_l408_408699


namespace sequence_within_rides_l408_408755

-- Condition: There are 12 cars numbered 1 through 12
def num_cars : ℕ := 12

-- Condition: A passenger equally likely to ride in any car each time.
def car_prob (n : ℕ) : ℚ := if 1 ≤ n ∧ n ≤ num_cars then 1 / num_cars else 0

-- Condition: The passenger rides the roller coaster 20 times.
def num_rides : ℕ := 20

-- Condition: The passenger must ride in the sequence 1 through 6 in increasing order and 12 through 7 in decreasing order at least once each within 20 rides.
def correct_sequence (rides : List ℕ) : Prop :=
  List.isInfix [1, 2, 3, 4, 5, 6] rides ∧ List.isInfix [12, 11, 10, 9, 8, 7] rides

-- Theorem: Given above conditions, we are investigating the probability that the passenger will successfully complete the sequence within the 20 rides.
theorem sequence_within_rides (rides : List ℕ) (h_len : rides.length = num_rides) :
  correct_sequence rides → sorry :=
sorry

end sequence_within_rides_l408_408755


namespace total_soaking_time_l408_408001

def stain_times (n_grass n_marinara n_coffee n_ink : Nat) (t_grass t_marinara t_coffee t_ink : Nat) : Nat :=
  n_grass * t_grass + n_marinara * t_marinara + n_coffee * t_coffee + n_ink * t_ink

theorem total_soaking_time :
  let shirt_grass_stains := 2
  let shirt_grass_time := 3
  let shirt_marinara_stains := 1
  let shirt_marinara_time := 7
  let pants_coffee_stains := 1
  let pants_coffee_time := 10
  let pants_ink_stains := 1
  let pants_ink_time := 5
  let socks_grass_stains := 1
  let socks_grass_time := 3
  let socks_marinara_stains := 2
  let socks_marinara_time := 7
  let socks_ink_stains := 1
  let socks_ink_time := 5
  let additional_ink_time := 2

  let shirt_time := stain_times shirt_grass_stains shirt_marinara_stains 0 0 shirt_grass_time shirt_marinara_time 0 0
  let pants_time := stain_times 0 0 pants_coffee_stains pants_ink_stains 0 0 pants_coffee_time pants_ink_time
  let socks_time := stain_times socks_grass_stains socks_marinara_stains 0 socks_ink_stains socks_grass_time socks_marinara_time 0 socks_ink_time
  let total_time := shirt_time + pants_time + socks_time
  let total_ink_stains := pants_ink_stains + socks_ink_stains
  let additional_ink_total_time := total_ink_stains * additional_ink_time
  let final_total_time := total_time + additional_ink_total_time

  final_total_time = 54 :=
by
  sorry

end total_soaking_time_l408_408001


namespace tangent_line_through_origin_unique_real_root_when_a_neg_l408_408911

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem tangent_line_through_origin (a : ℝ) : 
  let y := (a * Real.exp 1 - 1) / Real.exp 1 * (x : ℝ) 
  in y = (a - 1 / Real.exp 1) * x :=
begin
  sorry
end

theorem unique_real_root_when_a_neg (a : ℝ) (h : a < 0) : 
  ∃ (x : ℝ), (f a x) + a * x^2 = 0 ∧ ∀ x' : ℝ, (f a x') + a * x'^2 = 0 → x = x' :=
begin
  sorry
end

end tangent_line_through_origin_unique_real_root_when_a_neg_l408_408911


namespace living_room_side_length_l408_408607

-- Definition of the problem's conditions
def bedroom_width : ℝ := 10
def bedroom_length : ℝ := 12
def wall_height : ℝ := 10
def total_painted_area : ℝ := 1640

-- Calculate the area of the bedroom walls
def bedroom_wall_area : ℝ :=
  2 * bedroom_width * wall_height + 2 * bedroom_length * wall_height

-- Calculate the area required for the living room walls
def living_room_wall_area : ℝ :=
  total_painted_area - bedroom_wall_area

-- The theorem statement
theorem living_room_side_length : ∃ x : ℝ, 3 * x * wall_height = living_room_wall_area ∧ x = 40 :=
begin
  sorry
end

end living_room_side_length_l408_408607


namespace card_pairs_divisible_by_seven_l408_408617

-- Definitions and conditions given in the problem
def cards : Finset ℕ := Finset.range' 3 140 |>.map (λ n => 3 * (n + 1))

noncomputable theory

def is_divisible_by_seven (n : ℕ) : Prop := n % 7 = 0

-- The proof statement. We need to show that the number of ways to choose 
-- 2 cards such that their sum is divisible by 7 equals 1390
theorem card_pairs_divisible_by_seven : 
  ∃ n, n = 1390 ∧ (n = cards.filter (λ x : ℕ, ∀ y ∈ cards, x < y → is_divisible_by_seven (x + y))).card :=
sorry

end card_pairs_divisible_by_seven_l408_408617


namespace count_elements_in_S_l408_408510

noncomputable def a_n_seq : ℕ → ℤ
| 0       := 1
| (n + 1) := 2 * a_n_seq n - n

def S_n (n : ℕ) : ℤ := (Finset.range n).sum (λ i, a_n_seq i)

def Δ_a_n (n : ℕ) : ℤ := a_n_seq (n + 1) - a_n_seq n

def Δ_Δ_a_n (n : ℕ) : ℤ := Δ_a_n (n + 1) - Δ_a_n n

def S : Finset ℕ := 
  (Finset.range (11 + 1)).filter (λ n, Δ_Δ_a_n n ≥ -2015)

theorem count_elements_in_S : S.card = 11 := 
by 
  sorry

end count_elements_in_S_l408_408510


namespace convert_to_polar_l408_408015

noncomputable def r (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)

noncomputable def θ (x y : ℝ) : ℝ := 
if y ≥ 0 then real.atan (y / x)
else 2 * real.pi - real.atan (-y / x)

theorem convert_to_polar :
  let x := 3
  let y := -4
  r x y = 5 ∧ θ x y = 2 * real.pi - real.atan (4 / 3) := by
  have hx : r 3 (-4) = 5 := by sorry
  have hy : θ 3 (-4) = 2 * real.pi - real.atan (4 / 3) := by sorry
  exact ⟨hx, hy⟩

end convert_to_polar_l408_408015


namespace parallel_lines_slope_l408_408538

theorem parallel_lines_slope (m : ℚ) (h : (x - y = 1) → (m + 3) * x + m * y - 8 = 0) :
  m = -3 / 2 :=
sorry

end parallel_lines_slope_l408_408538


namespace max_distance_P_to_AB_l408_408484

variables {P : ℝ × ℝ} {A B : ℝ × ℝ}

-- Define the circle C as (x - 3)^2 + (y - 1)^2 = 9
def on_circle (x y : ℝ) : Prop := (x - 3) ^ 2 + (y - 1) ^ 2 = 9

-- Define the points A and B on the circle
variables (A_on_circle : on_circle (A.1) (A.2)) (B_on_circle : on_circle (B.1) (B.2))

-- Define the distance AB as |AB| = 2√5
def distance_AB : Prop := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * real.sqrt 5 

-- Define point P
def point_P := (0, -3) = P

-- Prove the maximum distance from P to the line AB is 7
theorem max_distance_P_to_AB : distance_AB → (P.1 = 0 ∧ P.2 = -3) → 7 = sorry :=
sorry

end max_distance_P_to_AB_l408_408484


namespace eight_digit_numbers_with_product_700_l408_408847

def digit_product_is_700 (n : ℕ) : Prop :=
  ∃ (d : Fin 8 → ℕ), (∀ i, d i ≤ 9) ∧ (n = list.prod (list.of_fn d))

theorem eight_digit_numbers_with_product_700 :
  (finset.range 100000000).filter (λ n, digit_product_is_700 n ∧ 10000000 ≤ n ∧ n < 100000000).card = 2520 := by
  sorry

end eight_digit_numbers_with_product_700_l408_408847


namespace value_of_f_f_neg_two_l408_408125

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 2 then x^2 + 1 else
  if x ≥ 2 then 2 - x else
  if -2 < x ∧ x ≤ 0 then x^2 + 1 else
  2 + x

theorem value_of_f_f_neg_two : f (f (-2)) = 1 := by 
  sorry

end value_of_f_f_neg_two_l408_408125


namespace sequence_sum_l408_408025

theorem sequence_sum :
  (∀ r: ℝ, 
    (∀ x y: ℝ,
      r = 1 / 4 → 
      x = 256 * r → 
      y = x * r → 
      x + y = 80)) :=
by sorry

end sequence_sum_l408_408025


namespace volleyball_tournament_l408_408972

theorem volleyball_tournament (
  p m : ℕ,
  x: ℕ 
) 
(h1 : 1 ≤ x) 
(h2 : x < 73)
(h3 : p ≠ m) 
(h4 : x * p + (73 - x) * m = 2628) : 
  false := 
by 
  sorry

end volleyball_tournament_l408_408972


namespace measure_of_angle_α_length_of_CD_l408_408323

-- Angles and lengths as real numbers
variables (AB CD α : ℝ)

-- Given conditions
axiom parallel_to_mirror_I_hits_mirror_II_at_A :
  ∀ (ray : Line) (A : Point), parallel_to(rays.mirror_I) ∧ hits_mirror_II(ray, A)

axiom angles_of_reflection_and_incidence_equal :
  ∀ (ray : Line) (m : Mirror) (A : Point), angle_of_reflection(ray, m, A) = angle_of_incidence(ray, m, A)

axiom triangle_AB_perpendicular_to_mirror_I :
  ∀ (A B : Point), perpendicular(AB, mirror_I)

constant line_ray_parallel_to_I : Line
constant point_A : Point
constant mirror_I : Mirror
constant mirror_II : Mirror

-- Question a: Prove that the measure of α is 22.5 degrees given the conditions.
theorem measure_of_angle_α (h1 : parallel_to_mirror_I_hits_mirror_II_at_A line_ray_parallel_to_I point_A)
                           (h2 : angles_of_reflection_and_incidence_equal line_ray_parallel_to_I mirror_II point_A)
                           (h3 : triangle_AB_perpendicular_to_mirror_I point_A B) :
  α = 22.5 := 
sorry

-- Question b: Prove that length CD is 10 cm given AB = 10 cm and the conditions.
theorem length_of_CD (h1 : AB = 10) (h2 : parallel_to_mirror_I_hits_mirror_II_at_A line_ray_parallel_to_I point_A)
                     (h3 : angles_of_reflection_and_incidence_equal line_ray_parallel_to_I mirror_II point_A)
                     (h4 : triangle_AB_perpendicular_to_mirror_I point_A B) :
  CD = 10 := 
sorry

end measure_of_angle_α_length_of_CD_l408_408323


namespace number_of_ways_to_choose_roles_l408_408181

theorem number_of_ways_to_choose_roles (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 3) :
  (Finset.univ.card : ℕ) ^ k - Finset.card (Finset.univ.filter (λ x, x ≠ x)) = 336 :=
by
  simp [h1, h2]
  sorry

end number_of_ways_to_choose_roles_l408_408181


namespace oranges_to_apples_equivalence_l408_408197

theorem oranges_to_apples_equivalence :
  (forall (o l a : ℝ), 4 * o = 3 * l ∧ 5 * l = 7 * a -> 20 * o = 21 * a) :=
by
  intro o l a
  intro h
  sorry

end oranges_to_apples_equivalence_l408_408197


namespace sum_of_remainders_l408_408577

-- Define Q as the set of all possible remainders when 3^n is divided by 500
def Q : Set ℕ := {m | ∃ n : ℕ, m = (3 ^ n) % 500}

-- Define T as the sum of the elements in Q
noncomputable def T : ℕ := Q.sum

-- State the proof problem, showing T % 500 is the given sum
theorem sum_of_remainders :
  T % 500 = (Finset.range 101).sum (λ i => (3 ^ i) % 500) % 500 := sorry

end sum_of_remainders_l408_408577


namespace number_of_valid_four_digit_numbers_l408_408527

-- Define the conditions
def is_valid_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9
def is_valid_product (a b : ℕ) : Prop := (a * b) > 10
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999
def first_digit_greater_than_4999 (d : ℕ) : Prop := 5 ≤ d ∧ d ≤ 9

-- Define the main statement
theorem number_of_valid_four_digit_numbers : 
  ∃ (count : ℕ), count = 3050 ∧ 
  ∀ (n : ℕ), is_four_digit_number(n) → 
             first_digit_greater_than_4999(n / 1000) →
             is_valid_product((n / 10) % 10) ((n / 100) % 10) → 
             n ∈ { m : ℕ | is_valid_digit(m % 10) } :=
sorry

end number_of_valid_four_digit_numbers_l408_408527


namespace count_factors_of_3_l408_408996

-- Define the sequence of the first 100 positive odd integers
def first_100_odd_integers : List ℕ := List.range 200 |>.filter (λ n => ∀ d, d | 2 → even d → d * 2 = n → odd n)

-- Define the product of numbers in the sequence
noncomputable def product_first_100_odd_integers : ℕ := first_100_odd_integers.foldl (λ acc x => acc * x) 1

-- Define the function to count the multiples of a given number up to 100 in the positive odd integers
def count_multiples (m : ℕ) : ℕ := (List.range 100).filter (λ k => 2 * k + 1 % m = 0).length

-- The actual problem statement
theorem count_factors_of_3 : ∀ k, 
  (3 ^ k ∣ product_first_100_odd_integers) ↔ k ≤ (count_multiples 3 + count_multiples 9 + count_multiples 27 + count_multiples 81) :=
  sorry

end count_factors_of_3_l408_408996


namespace probability_units_digit_is_1_is_zero_l408_408766

def m_values := {2, 7, 8}
def n_values := finset.range (2025 - 2005 + 1) + 2005 -- This constructs the natural numbers from 2005 to 2025
def p_values := {1, 3, 4}

def units_digit (x : ℕ) : ℕ := x % 10

noncomputable def m_n_p_combination_is_1 : Prop :=
  ∀ m ∈ m_values, ∀ n ∈ n_values, ∀ p ∈ p_values, units_digit (m ^ n + p) = 1 → false

theorem probability_units_digit_is_1_is_zero : m_n_p_combination_is_1 :=
by {
  sorry
}

end probability_units_digit_is_1_is_zero_l408_408766


namespace problem_l408_408331

theorem problem (m n : ℕ) (h_hcf : nat.gcd m n = 6) (h_lcm : nat.lcm m n = 210) (h_sum : m + n = 80) :
  1 / (m : ℚ) + 1 / (n : ℚ) = 2 / 31.5 := 
sorry

end problem_l408_408331


namespace unique_integer_solution_l408_408460

def is_point_in_circle (x y cx cy radius : ℝ) : Prop :=
  (x - cx)^2 + (y - cy)^2 ≤ radius^2

theorem unique_integer_solution : ∃! (x : ℤ), is_point_in_circle (2 * x) (-x) 4 6 8 := by
  sorry

end unique_integer_solution_l408_408460


namespace deposit_percentage_l408_408988

noncomputable def last_year_cost : ℝ := 250
noncomputable def increase_percentage : ℝ := 0.40
noncomputable def amount_paid_at_pickup : ℝ := 315
noncomputable def total_cost := last_year_cost * (1 + increase_percentage)
noncomputable def deposit := total_cost - amount_paid_at_pickup
noncomputable def percentage_deposit := deposit / total_cost * 100

theorem deposit_percentage :
  percentage_deposit = 10 := 
  by
    sorry

end deposit_percentage_l408_408988


namespace same_set_representation_l408_408324

-- Define the problem conditions
def set1a : Set (ℕ × ℕ) := {(3, 2)}
def set1b : Set (ℕ × ℕ) := {(2, 3)}

def set2a : Set ℕ := {3, 2}
def set2b : Set ℕ := {2, 3}

def set3a : Set (ℕ × ℕ) := {(x, y) | x + y = 1}
def set3b : Set ℕ := {y | ∃ x, x + y = 1}

def set4a : Set ℕ := {2, 3}
def set4b : Set (ℕ × ℕ) := {(2, 3)}

-- Proof statement
theorem same_set_representation : (set1a ≠ set1b) ∧ (set2a = set2b) ∧ (set3a ≠ set3b) ∧ (set4a ≠ set4b) := 
sorry

end same_set_representation_l408_408324


namespace x_plus_y_l408_408059

noncomputable def sequence_constant : ℝ := 1 / 4

def x : ℝ := 256 * sequence_constant

def y : ℝ := x * sequence_constant

theorem x_plus_y : x + y = 80 := 
by
  -- Placeholder for the proof
  sorry

end x_plus_y_l408_408059


namespace maximum_queens_attack_condition_l408_408317

noncomputable def max_queens_on_board : ℕ := 36

theorem maximum_queens_attack_condition (n : ℕ) (hboard : n = 8) (hattack : ∀ (x y : ℕ), (x ≠ y) → (n > 0) → (n ≤ 8) → x > 0 → y > 0 → x ≤ 8 → y ≤ 8) :
  ∃ (queens : ℕ), (queens = max_queens_on_board) ∧ (∀ (r c : ℕ), (r ≠ c) → (queens > 0) → (queens ≤ (hboard * hboard))) :=
begin
  sorry
end

end maximum_queens_attack_condition_l408_408317


namespace fraction_sum_le_41_over_42_l408_408335

theorem fraction_sum_le_41_over_42 (a b c : ℕ) (h : 1/a + 1/b + 1/c < 1) : 1/a + 1/b + 1/c ≤ 41/42 :=
sorry

end fraction_sum_le_41_over_42_l408_408335


namespace chickens_after_9_years_l408_408608

-- Definitions from the conditions
def annual_increase : ℕ := 150
def current_chickens : ℕ := 550
def years : ℕ := 9

-- Lean statement for the proof
theorem chickens_after_9_years : current_chickens + annual_increase * years = 1900 :=
by
  sorry

end chickens_after_9_years_l408_408608


namespace probability_pairs_problem_l408_408061

-- Define the problem in Lean
theorem probability_pairs_problem :
  ∃ (p q : ℕ), Nat.gcd p q = 1 ∧ (p + q = 9) ∧ 
  (∀ (k : ℕ), (k < 4) → (¬ ∃ pairs : set (fin 8 × fin 8), 
     (set.card pairs = k) ∧ 
     (∀ (pair ∈ pairs, exists (i : fin 8), (pair.1 = i) ∧ (pair.2 = i)))) → 
  (p = 1 ∧ q = 8) := sorry

end probability_pairs_problem_l408_408061


namespace pi_irrational_l408_408734

def rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem pi_irrational : ¬ rational π :=
sorry

end pi_irrational_l408_408734


namespace hyperbola_eccentricity_range_l408_408485

open Real

noncomputable def eccentricity_range (a b: ℝ) (e: ℝ) : Prop :=
∀ (F₁ F₂ G : ℝ × ℝ), 
(F₁.1 = -c) ∧ (F₂.1 = c) ∧ 
(c^2 = a^2 + b^2) ∧ 
(G.1^2 / a^2 - G.2^2 / b^2 = 1) ∧ 
( (abs (dist G F₁) / abs (dist G F₂)) = 9 ) → 
1 < e ∧ e ≤ 5 / 4

theorem hyperbola_eccentricity_range {a b e: ℝ} (h: a > 0 ∧ b > 0) :
eccentricity_range a b e :=
sorry

end hyperbola_eccentricity_range_l408_408485


namespace cost_per_pound_peanuts_l408_408855

-- Defining the conditions as needed for our problem
def one_dollar_bills := 7
def five_dollar_bills := 4
def ten_dollar_bills := 2
def twenty_dollar_bills := 1
def change := 4
def pounds_per_day := 3
def days_in_week := 7

-- Calculating the total initial amount of money Frank has
def total_initial_money := (one_dollar_bills * 1) + (five_dollar_bills * 5) + (ten_dollar_bills * 10) + (twenty_dollar_bills * 20)

-- Calculating the total amount spent on peanuts
def total_spent := total_initial_money - change

-- Calculating the total pounds of peanuts
def total_pounds := pounds_per_day * days_in_week

-- The proof statement
theorem cost_per_pound_peanuts : total_spent / total_pounds = 3 := sorry

end cost_per_pound_peanuts_l408_408855


namespace units_digit_sum_squares_first_2023_odd_integers_l408_408722

theorem units_digit_sum_squares_first_2023_odd_integers :
  let pattern := [1, 9, 5, 9, 1] in
  let full_cycles := 2023 / 5 in
  let remaining := 2023 % 5 in
  (full_cycles * (pattern.foldl (λ acc x => acc + x) 0) + (remaining * pattern.foldl (λ acc x => acc + if x = (remaining - 1).natAbs then x else 0) 0)) % 10 = 5 :=
by
  sorry

end units_digit_sum_squares_first_2023_odd_integers_l408_408722


namespace next_four_digit_number_after_1818_l408_408682

-- Formalization of the given conditions as Lean definitions
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def next_valid_number (n : ℕ) : ℕ :=
  let d1 := n / 1000 in
  let d2 := (n / 100) % 10 in
  let d3 := (n / 10) % 10 in
  let d4 := n % 10 in
  let next_n := (d1 * 10 + d2) * 100 + (d3 * 10 + d4) + 14*10 in -- moving to the next valid 4 digits in base 10 series
  next_n

def condition (n : ℕ) : Prop :=
  let d3d4 := n % 100 in
  is_perfect_square (18 * d3d4)

theorem next_four_digit_number_after_1818 :
  ∀ n : ℕ, 
    condition 1818 → 
    next_valid_number 1818 = 1832 :=
begin
  intro n,
  intro cond,
  sorry
end

end next_four_digit_number_after_1818_l408_408682


namespace number_of_such_four_digit_numbers_is_14_l408_408377

-- Define the problem conditions as a proposition
def four_digit_numbers_odd_adjacent_5_6 : Prop :=
  ∃ (l : List ℕ), l.length = 4 ∧
    (∀ n ∈ l, n ∈ [2, 3, 4, 5, 6]) ∧
    l.nodup ∧
    (List.getLast l (by sorry) % 2 = 1) ∧
    (∃ i, (l.nth i = some 5 ∧ l.nth (i + 1) = some 6) ∨ (l.nth i = some 6 ∧ l.nth (i + 1) = some 5))

-- State the theorem to prove the number of such numbers is 14
theorem number_of_such_four_digit_numbers_is_14 : 
  ∃ l : List (List ℕ), (∀ x ∈ l, four_digit_numbers_odd_adjacent_5_6) ∧ l.length = 14 :=
sorry

end number_of_such_four_digit_numbers_is_14_l408_408377


namespace sequence_bound_l408_408743

theorem sequence_bound (a : ℕ → ℕ) (a_1 : ℕ) :
  (∀ n : ℕ, ∃ d ∈ (finset.range 10 \ {0}), a(2*n) = a(2*n-1) + d) ∧
  (∀ n : ℕ, ∃ d ∈ (finset.range 10 \ {0}), a(2*n+1) = a(2*n) - d) →
  (∀ n : ℕ, a n ≤ 4 * a 1 + 44) :=
sorry

end sequence_bound_l408_408743


namespace limit_of_ratio_f_f_l408_408574

noncomputable theory
open Real

variables {f : ℝ → ℝ}

-- twice continuously differentiable function on (0, ∞)
def twice_cont_diff_on_pos : Prop :=
  differentiable_on ℝ f (Ioi 0) ∧ 
  (∀ x ∈ Ioi 0, differentiable_at ℝ (deriv f) x)

-- lim(x → 0⁺) f'(x) = -∞
def lim_deriv_at_0_pos_eq_neg_inf : Prop :=
  tendsto (deriv f) (𝓝[>] 0) at_bot

-- lim(x → 0⁺) f''(x) = ∞
def lim_2nd_deriv_at_0_pos_eq_inf : Prop :=
  tendsto (deriv (deriv f)) (𝓝[>] 0) at_top

theorem limit_of_ratio_f_f'_at_0_pos_eq_zero
  (h1 : twice_cont_diff_on_pos f)
  (h2 : lim_deriv_at_0_pos_eq_neg_inf f)
  (h3 : lim_2nd_deriv_at_0_pos_eq_inf f) :
  tendsto (λ x, (f x) / (deriv f x)) (𝓝[>] 0) (𝓝 0) :=
sorry

end limit_of_ratio_f_f_l408_408574


namespace find_angle_A_find_area_l408_408120

noncomputable def triangle_area (a b c : ℝ) (A : ℝ) : ℝ :=
  0.5 * b * c * Real.sin A

theorem find_angle_A (a b c A : ℝ)
  (h1: ∀ x, 4 * Real.cos x * Real.sin (x - π/6) ≤ 4 * Real.cos A * Real.sin (A - π/6))
  (h2: a = b^2 + c^2 - 2 * b * c * Real.cos A) : 
  A = π / 3 := by
  sorry

theorem find_area (a b c : ℝ)
  (A : ℝ) (hA : A = π / 3)
  (ha : a = Real.sqrt 7) (hb : b = 2) 
  : triangle_area a b c A = (3 * Real.sqrt 3) / 2 := by
  sorry

end find_angle_A_find_area_l408_408120


namespace number_of_arrangements_l408_408256

theorem number_of_arrangements (total_people : ℕ) (A B : ℕ) : 
  total_people = 6 → 
  let P := finset.univ.filter (λ s, s.card = 2) in
  let Q := finset.univ.filter (λ s, s.card = 4) in
  let A4 := finset.card P in
  let A6 := finset.card finset.univ in
  (A6 - (A4 ^ 2 * A4 ^ 4)) = 432 :=
sorry

end number_of_arrangements_l408_408256


namespace slope_of_line_l_l408_408914

noncomputable def circle_polar_equation (θ : ℝ) : ℝ := 4 * real.cos θ - 6 * real.sin θ

noncomputable def line_parametric_equation (t θ : ℝ) : ℝ × ℝ := (4 + t * real.cos θ, t * real.sin θ)

theorem slope_of_line_l (θ : ℝ) (t : ℝ) (P Q : ℝ × ℝ)
  (hp : P = line_parametric_equation 0 θ)
  (hq : Q = line_parametric_equation 4 θ)
  (hintersect : ∃ P Q : ℝ × ℝ, (P ≠ Q) ∧ (P, Q ∈ line_parametric_equation t θ))
  (hchord : real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2) = 4) :
  ∃ k : ℝ, k = 0 ∨ k = -12 / 5 := by
sorry

end slope_of_line_l_l408_408914


namespace cricket_initial_overs_l408_408177

theorem cricket_initial_overs
  (target_runs : ℚ) (initial_run_rate : ℚ) (remaining_run_rate : ℚ) (remaining_overs : ℕ)
  (total_runs_needed : target_runs = 282)
  (run_rate_initial : initial_run_rate = 3.4)
  (run_rate_remaining : remaining_run_rate = 6.2)
  (overs_remaining : remaining_overs = 40) :
  ∃ (initial_overs : ℕ), initial_overs = 10 :=
by
  sorry

end cricket_initial_overs_l408_408177


namespace magnitude_product_complex_l408_408413

theorem magnitude_product_complex :
  let z1 := Complex.mk 7 (-4)
  let z2 := Complex.mk 3 11
  Complex.abs (z1 * z2) = Real.sqrt 8450 :=
by
  sorry

end magnitude_product_complex_l408_408413


namespace sum_of_ages_in_20_years_l408_408381

theorem sum_of_ages_in_20_years :
  let ann := 6
  let tom := 2 * ann
  let bill := tom - 3
  let cathy := 2 * tom
  let emily := cathy / 2
  let in_20_years (age : ℕ) : ℕ := age + 20
  in 
  in_20_years ann + in_20_years tom + in_20_years bill + in_20_years cathy + in_20_years emily = 163 :=
by
  sorry

end sum_of_ages_in_20_years_l408_408381


namespace problem_statement_l408_408731

-- Define the positive integers a and b representing the simplified components
def a : ℕ := 12
def b : ℕ := 6

-- Define the given expression
def expr := Real.root 4 (2^9 * 3^5)

-- Define the simplified version
def simplified_expr := a * Real.root 4 b

-- Define the question: prove the equivalence and sum
theorem problem_statement : expr = simplified_expr ∧ a + b = 18 :=
by
  sorry

end problem_statement_l408_408731


namespace num_representable_numbers_l408_408787

theorem num_representable_numbers : 
  (∃ count : ℕ, (count = (100 - (∑ x in (finset.range 11) \ {0}, ((101 - x) / x).nat_succ ) - 1) ∧ count = 74)) :=
sorry

end num_representable_numbers_l408_408787


namespace base6_subtraction_addition_l408_408801

def base6_to_decimal (n : Nat) : Nat :=
  n.digits 6.reverse.foldl (λ acc x, 6 * acc + x) 0

def decimal_to_base6 (n : Nat) : Nat :=
  let rec to_base6_rec (n : Nat) (acc : List Nat) : List Nat :=
    if n < 6 then n :: acc
    else to_base6_rec (n / 6) ((n % 6) :: acc)
  Nat.ofDigits 6 (to_base6_rec n [])

theorem base6_subtraction_addition : 
  let n1 := 655
  let n2 := 222
  let n3 := 111
  let result_decimal := base6_to_decimal n1 - base6_to_decimal n2 + base6_to_decimal n3
  decimal_to_base6 result_decimal = 544
:= by
  sorry

end base6_subtraction_addition_l408_408801


namespace range_of_omega_for_three_zeros_l408_408902

theorem range_of_omega_for_three_zeros (ω : ℝ) (hω : ω > 0) :
  (∀ x ∈ set.Icc 0 (2 * Real.pi), (sin (ω * x) - 1) = 0 → ∃! x, True) →
  ω ∈ set.Icc (9 / 4 : ℝ) (13 / 4 : ℝ) :=
by sorry

end range_of_omega_for_three_zeros_l408_408902


namespace Lou_receives_lollipops_l408_408226

theorem Lou_receives_lollipops
  (initial_lollipops : ℕ)
  (fraction_to_Emily : ℚ)
  (lollipops_kept : ℕ)
  (lollipops_given_to_Lou : ℕ) :
  initial_lollipops = 42 →
  fraction_to_Emily = 2 / 3 →
  lollipops_kept = 4 →
  lollipops_given_to_Lou = initial_lollipops - (initial_lollipops * fraction_to_Emily).natAbs - lollipops_kept →
  lollipops_given_to_Lou = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end Lou_receives_lollipops_l408_408226


namespace least_number_of_stamps_l408_408002

theorem least_number_of_stamps (p q : ℕ) (h : 5 * p + 4 * q = 50) : p + q = 11 :=
sorry

end least_number_of_stamps_l408_408002


namespace relation_x_y_l408_408546

open Set

variables {O A B C D E : Point} {a x y : ℝ}

-- Define the conditions from the problem
noncomputable def circle (O : Point) (a : ℝ) := { P : Point | dist P O = a }

-- Assume the existence of points with given properties
variables (h1 : dist O A = a) 
          (h2 : dist O B = a) 
          (h3 : ∀ P ∈ circle O a, P ≠ A → P ≠ B → collinear O P A ∧ collinear O P B )
          (AB_diameter : dist A B = 2 * a)
          (h4 : ∃ C D, extends_line_segment A D C ∧ is_tangent (circle O a) B C ∧ dist A E = dist D C)
          (h5 : y = dist E (line_through O A))
          (h6 : x = dist E (line_through A B))

-- Define the Lean statement
theorem relation_x_y : y^2 = x^3 / (2 * a - x) :=
sorry

end relation_x_y_l408_408546


namespace binomial_expansion_terms_l408_408268

theorem binomial_expansion_terms (x n : ℝ) (hn : n = 8) : 
  ∃ t, t = 3 :=
  sorry

end binomial_expansion_terms_l408_408268


namespace math_proof_l408_408204

noncomputable def P : ℚ := 8^(0.25 : ℝ) * (2 : ℝ)^(1/4) + ((27 : ℝ) / 64)^(-1/3) - (-2018)^0
noncomputable def Q : ℝ := 2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3
noncomputable def m (Q : ℝ) : ℝ := Real.exp ((Real.log 2 + Real.log 5) / (2 * Real.log 10))

theorem math_proof
  (hP : P = 7/3)
  (hQ : Q = 2)
  (hcond1 : ∀ a b m : ℝ, 2^a = m → 5^b = m → (1/a + 1/b = Q) → m = Real.exp (1/2 * Real.log 10)) :
  ∃ m : ℝ, (2^a = m ∧ 5^b = m ∧ (1/a + 1/b = Q) → m = Real.exp (1/2 * Real.log 10)) :=
begin
  use Real.exp (1/2 * Real.log 10),
  intros a b _ h1 h2 h3,
  specialize hcond1 a b (Real.exp (1/2 * Real.log 10)),
  exact hcond1 h1 h2 h3,
end

end math_proof_l408_408204


namespace employees_percentage_l408_408405

theorem employees_percentage (X : ℕ) :
  let T := 7 * X + 4 * X + 3 * X + 3 * X + 2 * X + 2 * X + X + X + X,
      E := 2 * X + X + X + X in
  (E : ℚ) / (T : ℚ) * 100 = 21.74 :=
by
  sorry

end employees_percentage_l408_408405


namespace ratio_first_number_l408_408159

theorem ratio_first_number (x : ℝ) (y : ℝ) (h1 : x = 0.8571428571428571) (h2 : y / x = 7 / 8) : y ≈ 0.75 :=
by
  sorry

end ratio_first_number_l408_408159


namespace problem1987_divisibility_l408_408127

theorem problem1987_divisibility (n : ℕ) (h1 : 0 < n)
  (h2 : 1987 ∣ (list.repeat 1 n).foldl (λ sum d, sum * 10 + d) 0) :
  let p := (list.repeat 1 n).foldl (λ sum d, sum * 10 + d) 0 * (10^(3*n) + 9 * 10^(2*n) + 8 * 10^n + 7)
  ∧
  let q := (list.repeat 1 (n+1)).foldl (λ sum d, sum * 10 + d) 0 * (10^(3*(n+1)) + 9 * 10^(2*(n+1)) + 8 * 10^(n+1) + 7)
in 1987 ∣ p ∧ 1987 ∣ q := 
 sorry

end problem1987_divisibility_l408_408127


namespace sec_330_eq_l408_408443

-- Definitions of conditions
def sec (θ : ℝ) : ℝ := 1 / real.cos θ
noncomputable def angle_330 := 330 * real.pi / 180
noncomputable def angle_30 := 30 * real.pi / 180
noncomputable def cos_330 := real.cos (2 * real.pi - angle_30)
noncomputable def cos_30 := real.cos angle_30

-- Condition equality
axiom angle_relation : cos_330 = cos_30
axiom cos_30_value : cos_30 = real.sqrt 3 / 2

-- Theorem to be proved
theorem sec_330_eq : sec angle_330 = 2 * real.sqrt 3 / 3 :=
by
  sorry

end sec_330_eq_l408_408443


namespace solution_l408_408275

def f (ω x : ℝ) : ℝ := cos (ω * x + (π / 6))
axiom ω_pos : ω > 0

theorem solution (ω : ℝ) (hω : ω > 0) :
  (f ω (π / 2) = - (√3 / 2)) ∧
  (∀ x, f ω (x + π) = f ω x) ∧
  (∀ x, 0 < x ∧ x < (π / (3 * ω)) → f ω x < f ω (x + δ) ∧ 0 < δ) :=
begin
  sorry
end

end solution_l408_408275


namespace find_f6_l408_408217

def f : ℕ → ℤ
| 1     := 24
| 2     := f 1 - 2
| 3     := f 2 - 3
| 4     := f 3 - 4
| n@(5) := g (n-1) + 2*n
| n@(6) := g (n-1) + 2*n
-- in general case we need to continue the piecewise definition
| n     := if n >= 9 then (g (n-1) * n : ℤ) else f (n - 1) - n

def g : ℕ → ℤ
| 0     := 0    -- Assumed base value not given in problem explicitly
| 1     := 2 * g 0 - 1
| 2     := 2 * g 1 - 2
| 3     := 2 * g 2 - 3
| 4     := 2 * g 3 - 4
| n@(5) := f (n-1) + (n ! : ℤ)
-- in general case we need to continue the piecewise definition
| n     := if n >= 9 then (f (n-1) ^ n : ℤ) else 2 * g (n-1) - n

theorem find_f6 : f 6 = 147 := 
by sorry

end find_f6_l408_408217


namespace curve_C1_polar_equation_chord_length_AB_l408_408507

noncomputable def curve_C1_param (t : ℝ) : ℝ × ℝ :=
  (t, t^2)

noncomputable def curve_C2_polar (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ - Real.pi / 3) = 1

theorem curve_C1_polar_equation {ρ θ : ℝ} (t : ℝ) :
  let (x, y) := curve_C1_param t in
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ →
  Real.sin θ = ρ * (Real.cos θ)^2 :=
  sorry

theorem chord_length_AB {x₁ x₂ : ℝ} (h₁ : x₁ + x₂ = Real.sqrt 3) (h₂ : x₁ * x₂ = -2) :
  Real.sqrt ((1 + 3) * ((x₁ + x₂)^2 - 4 * x₁ * x₂)) = 2 * Real.sqrt 11 :=
  sorry

end curve_C1_polar_equation_chord_length_AB_l408_408507


namespace parabola_properties_l408_408122

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)
def parabola_equation (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
def inclination (θ : ℝ) (x y : ℝ) : Prop := y = Real.tan θ * (x - 1)

theorem parabola_properties (m : ℝ) :
  (∃ p : ℝ, parabola_equation p 2 m ∧ distance 2 m (parabola_focus p).1 (parabola_focus p).2 = 3) ∧
  (let θ := Real.pi / 3 in inclination θ 2 m) →
  (∃ (p : ℝ), p = 2 ∧ ∀ x y : ℝ, y^2 = 4 * x) ∧ 
  (∃ (length : ℝ), length = 16 / 3) :=
by
  sorry

end parabola_properties_l408_408122


namespace sec_330_eq_l408_408442

-- Definitions of conditions
def sec (θ : ℝ) : ℝ := 1 / real.cos θ
noncomputable def angle_330 := 330 * real.pi / 180
noncomputable def angle_30 := 30 * real.pi / 180
noncomputable def cos_330 := real.cos (2 * real.pi - angle_30)
noncomputable def cos_30 := real.cos angle_30

-- Condition equality
axiom angle_relation : cos_330 = cos_30
axiom cos_30_value : cos_30 = real.sqrt 3 / 2

-- Theorem to be proved
theorem sec_330_eq : sec angle_330 = 2 * real.sqrt 3 / 3 :=
by
  sorry

end sec_330_eq_l408_408442


namespace equivalent_expression_ratio_l408_408243

theorem equivalent_expression_ratio :
  let c : ℚ := 8
  let p : ℚ := -3 / 8
  let q : ℚ := 604 / 32
  (c * ((j + p)^2) + q = 8 * j^2 - 6 * j + 20) →
  (q / p) = -151 / 3 :=
by
  intros c p q h
  rw [← h]
  sorry

end equivalent_expression_ratio_l408_408243


namespace find_LP_l408_408558

-- In triangle ABC, side lengths AC and BC are given, and various geometric points and line segments are defined.
variables {A B C K L M P : Type}
variables {AC BC AM LP : ℝ}
variables (AK CK CL : ℝ)
variables {AB : ℝ} -- Length of side AB.
variables {x : ℝ}

-- Conditions
variables (AC_eq : AC = 900) (BC_eq : BC = 600)
variables (AK_eq_CK : AK = CK)
variables (AM_eq : AM = 360)
variables (AB_eq : AB = 5 * x)
variables (AL_eq : AL = 3 * x) (LB_eq : LB = 2 * x)
variables (angle_bisector : AK = CK) (parallelogram : AC = CP)

-- The proof statement for the problem
theorem find_LP :
  LP = 144 :=
begin
  -- initiate the proof environment
  sorry  -- The actual proof steps will go here.
end

end find_LP_l408_408558


namespace selling_price_correct_l408_408379

theorem selling_price_correct (cost_price : ℝ) (loss_percent : ℝ) (selling_price : ℝ) 
  (h_cost : cost_price = 600) 
  (h_loss : loss_percent = 25)
  (h_selling_price : selling_price = cost_price - (loss_percent / 100) * cost_price) : 
  selling_price = 450 := 
by 
  rw [h_cost, h_loss] at h_selling_price
  norm_num at h_selling_price
  exact h_selling_price

#check selling_price_correct

end selling_price_correct_l408_408379


namespace ratio_of_distances_l408_408856

-- Define the speeds and times for ferries P and Q
def speed_P : ℝ := 8
def time_P : ℝ := 3
def speed_Q : ℝ := speed_P + 1
def time_Q : ℝ := time_P + 5

-- Define the distances covered by ferries P and Q
def distance_P : ℝ := speed_P * time_P
def distance_Q : ℝ := speed_Q * time_Q

-- The statement to prove: the ratio of the distances
theorem ratio_of_distances : distance_Q / distance_P = 3 :=
sorry

end ratio_of_distances_l408_408856


namespace pool_cannot_be_filled_l408_408765

noncomputable def pool := 48000 -- Pool capacity in gallons
noncomputable def hose_rate := 3 -- Rate of each hose in gallons per minute
noncomputable def number_of_hoses := 6 -- Number of hoses
noncomputable def leakage_rate := 18 -- Leakage rate in gallons per minute

theorem pool_cannot_be_filled : 
  (number_of_hoses * hose_rate - leakage_rate <= 0) -> False :=
by
  -- Skipping the proof with 'sorry' as per instructions
  sorry

end pool_cannot_be_filled_l408_408765


namespace sequence_value_l408_408041

theorem sequence_value (r x y : ℝ) 
  (h1 : r = 1/4) 
  (h2 : x = 256 * r)
  (h3 : y = x * r) : 
  x + y = 80 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end sequence_value_l408_408041


namespace modulus_of_z_l408_408473

-- Definition: condition that z = 1 - i
def z : ℂ := 1 - Complex.i

-- Statement of the problem (in Lean 4): Prove that |z| = sqrt(2)
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := sorry

end modulus_of_z_l408_408473


namespace sequence_a_l408_408916

theorem sequence_a (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 5)
  (h_rec : ∀ n ≥ 3, a n = 2 * a (n - 1) + 3 * a (n - 2)) :
  a 20 - 3 * a 19 = -1 :=
by
  have := sorry
  assumption

end sequence_a_l408_408916


namespace domain_f_2x_plus_1_l408_408124

def f (x : ℝ) : ℝ := sorry

noncomputable def domain_f : set ℝ := {x | -1 < x ∧ x < 0}

theorem domain_f_2x_plus_1 :
  {x : ℝ | f(2*x + 1) = f x} ∈ {x : ℝ | -1 < x ∧ x < -1/2} :=
by
  sorry

end domain_f_2x_plus_1_l408_408124


namespace math_problem_l408_408532

theorem math_problem
  (x : ℝ)
  (h : x - sqrt (x^2 - 4) + 1 / (x + sqrt (x^2 - 4)) = 10) :
  x^2 - sqrt (x^4 - 16) + 1 / (x^2 - sqrt (x^4 - 16)) = 237 / 2 :=
by
  sorry

end math_problem_l408_408532


namespace square_area_with_tangent_circles_l408_408084

theorem square_area_with_tangent_circles :
  let r := 3 -- radius of each circle in inches
  let d := 2 * r -- diameter of each circle in inches
  let side_length := 2 * d -- side length of the square in inches
  let area := side_length * side_length -- area of the square in square inches
  side_length = 12 ∧ area = 144 :=
by
  let r := 3
  let d := 2 * r
  let side_length := 2 * d
  let area := side_length * side_length
  sorry

end square_area_with_tangent_circles_l408_408084


namespace solve_y_minus_x_l408_408665

theorem solve_y_minus_x (x y : ℝ) (h1 : x + y = 399) (h2 : x / y = 0.9) : y - x = 21 :=
sorry

end solve_y_minus_x_l408_408665


namespace rainfall_calculation_l408_408635

theorem rainfall_calculation :
  let chi_to_cun := (10 : ℝ),
  let top_diameter := (2 * chi_to_cun + 8 : ℝ),
  let bottom_diameter := (1 * chi_to_cun + 2 : ℝ),
  let basin_depth := (1 * chi_to_cun + 18 : ℝ),
  let water_depth := (9 : ℝ),
  let volume_filled := (1 / 3) * real.pi * water_depth * 
                       ((top_diameter / 2) ^ 2 + (top_diameter / 2) * (bottom_diameter / 2) + 
                       (bottom_diameter / 2) ^ 2),
  let mouth_area := real.pi * (top_diameter / 2) ^ 2
  in volume_filled / mouth_area = 3 :=
by sorry

end rainfall_calculation_l408_408635


namespace parallel_vectors_l408_408926

theorem parallel_vectors (λ : ℝ) : 
  let a := (2, λ)
  let b := (λ - 1, 1)
  (a.1 * b.2 - a.2 * b.1 = 0) → (λ = -1 ∨ λ = 2) :=
begin
  intros,
  have h : 2 * 1 - λ * (λ - 1) = 0,
  {
    sorry,  -- This will be where the simplification and proof steps will go.
  },
  -- Since we are working with the conditions given, this would involve resolving the quadratic.
  sorry
end

end parallel_vectors_l408_408926


namespace correct_option_l408_408397

noncomputable def proposition_C (m n : Line) (α β : Plane) :=
  (Perpendicular m α) ∧ (Parallel n β) ∧ (Parallel α β) → (Perpendicular m n)

theorem correct_option (m n : Line) (α β : Plane) :
  (Perpendicular m α) ∧ (Parallel n β) ∧ (Parallel α β) → (Perpendicular m n) :=
by
  intros h_conditions
  have ⟨h_m_perp_α, h_n_parallel_β, h_α_parallel_β⟩ := h_conditions
  sorry

end correct_option_l408_408397


namespace triangular_array_multiples_of_71_l408_408372

theorem triangular_array_multiples_of_71 :
  let a(n k : ℕ) := 2^(n-1) * (n + 2*k - 2)
  ∃! t, t = 65 ∧ ∀ n k, (k ≤ 101 - n) → (71 ∣ a n k) → t = t :=
begin
  let a := λ (n k : ℕ), 2^(n-1) * (n + 2*k - 2),
  sorry
end

end triangular_array_multiples_of_71_l408_408372


namespace tan_double_alpha_l408_408890

variable (α : ℝ)

-- Conditions
def sin_pi_minus_alpha : Prop := sin (Real.pi - α) = 3 * Real.sqrt 10 / 10
def alpha_acute : Prop := 0 < α ∧ α < Real.pi / 2

-- Theorem to prove
theorem tan_double_alpha (h1 : sin_pi_minus_alpha α) (h2 : alpha_acute α) : 
  Real.tan (2 * α) = -3 / 4 :=
sorry

end tan_double_alpha_l408_408890


namespace problem_solution_l408_408114

theorem problem_solution
  (a b : ℝ)
  (h₁ : a > b)
  (h₂ : b > 0)
  (h3 : (sqrt a) ^ 2 - 5 * (sqrt a) + 2 = 0)
  (h4 : (sqrt b) ^ 2 - 5 * (sqrt b) + 2 = 0) :
  ( (a * sqrt a + b * sqrt b) / (a - b) * 
    (2 / sqrt a - 2 / sqrt b) / 
    (sqrt a - (a + b) / sqrt b) + 
    5 * (5 * sqrt a - a) / (b + 2)
  ) = 5 :=
by
  sorry

end problem_solution_l408_408114


namespace spherical_segment_equals_circle_area_l408_408666

noncomputable def spherical_segment_surface_area (R H : ℝ) : ℝ := 2 * Real.pi * R * H
noncomputable def circle_area (b : ℝ) : ℝ := Real.pi * (b * b)

theorem spherical_segment_equals_circle_area
  (R H b : ℝ) 
  (hb : b^2 = 2 * R * H) 
  : spherical_segment_surface_area R H = circle_area b :=
by
  sorry

end spherical_segment_equals_circle_area_l408_408666


namespace exponent_multiplication_l408_408004

theorem exponent_multiplication (a : ℝ) : (a^3) * (a^2) = a^5 := 
by
  -- Using the property of exponents: a^m * a^n = a^(m + n)
  sorry

end exponent_multiplication_l408_408004


namespace length_AF_is_25_l408_408215

open Classical

noncomputable def length_AF : ℕ :=
  let AB := 5
  let AC := 11
  let DE := 8
  let EF := 4
  let BC := AC - AB
  let CD := BC / 3
  let AF := AB + BC + CD + DE + EF
  AF

theorem length_AF_is_25 :
  length_AF = 25 := by
  sorry

end length_AF_is_25_l408_408215


namespace find_r_l408_408074

theorem find_r (r: ℝ) : log 8 (r + 8) = (7 / 3) → r = 120 := 
by 
  sorry

end find_r_l408_408074


namespace product_of_two_integers_l408_408539

theorem product_of_two_integers (x y : ℕ) (h1 : x + y = 26) (h2 : x^2 - y^2 = 52) (h3 : x > y) : x * y = 168 := by
  sorry

end product_of_two_integers_l408_408539


namespace sequence_sum_l408_408052

noncomputable def r : ℝ := 1 / 4
def t₀ : ℝ := 4096
def t₁ : ℝ := t₀ * r
def t₂ : ℝ := t₁ * r
def x : ℝ := t₂ * r
def y : ℝ := x * r

theorem sequence_sum : x + y = 80 := by
  -- The proof will go here
  sorry

end sequence_sum_l408_408052


namespace probability_diff_by_three_is_one_eighth_l408_408374

theorem probability_diff_by_three_is_one_eighth :
  let die_rolls := (1:ℕ) .. 8 in
  let valid_pairs := [(1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3), (7, 4), (8, 5)] in
  (valid_pairs.length : ℚ) / (die_rolls.length * die_rolls.length) = 1 / 8 := by
  sorry

end probability_diff_by_three_is_one_eighth_l408_408374


namespace right_triangle_hypotenuse_squared_l408_408390

theorem right_triangle_hypotenuse_squared 
  (p q r : ℂ)
  (s t u : ℂ)
  (h₀ : (∀ z : ℂ, (z - p) * (z - q) * (z - r) = z^3 + s * z^2 + t * z + u))
  (h₁ : |p|^2 + |q|^2 + |r|^2 = 300)
  (h₂ : (∃ (a b : ℂ), (p, q, r) = (a + b * complex.I, a - b * complex.I, p))) :
  (∃ k : ℝ, k^2 = 450) := sorry

end right_triangle_hypotenuse_squared_l408_408390


namespace satellite_modular_units_l408_408361

variable (U N S T : ℕ)

def condition1 : Prop := N = (1/8 : ℝ) * S
def condition2 : Prop := T = 4 * S
def condition3 : Prop := U * N = 3 * S

theorem satellite_modular_units
  (h1 : condition1 N S)
  (h2 : condition2 T S)
  (h3 : condition3 U N S) :
  U = 24 :=
sorry

end satellite_modular_units_l408_408361


namespace T_30_is_13515_l408_408394

def sequence_first_element (n : ℕ) : ℕ := 1 + n * (n - 1) / 2

def sequence_last_element (n : ℕ) : ℕ := sequence_first_element n + n - 1

def sum_sequence_set (n : ℕ) : ℕ :=
  n * (sequence_first_element n + sequence_last_element n) / 2

theorem T_30_is_13515 : sum_sequence_set 30 = 13515 := by
  sorry

end T_30_is_13515_l408_408394


namespace number_of_students_scoring_above_130_l408_408956

noncomputable def probability_above_130 (μ σ : ℝ) (n : ℕ) : ℝ :=
  let σ_sq := σ^2
  let R := NormalDist.mk μ σ
  let p := 1 / 2 * (1 - 0.6826)
  n * p

theorem number_of_students_scoring_above_130
  (n : ℕ)
  (μ : ℝ)
  (σ_sq : ℝ)
  (h_n : n = 40)
  (h_μ : μ = 120)
  (h_σ_sq : σ_sq = 100)
  (P_normal : ∀ k, (|X - μ| < σ * k) → P (|X - μ| < σ * k) = [0.6826, 0.9544, 0.9974].nth (k-1).get_or_else 0 := 
begin
  sorry
end
/- The number of students scoring above 130 is approximately 6 -/

end number_of_students_scoring_above_130_l408_408956


namespace total_birds_correct_l408_408859

def num_female_doves := 80
def num_male_pigeons := 50
def eggs_per_dove := 6
def eggs_per_pigeon := 4
def dove_hatch_rate := 8 / 10
def pigeon_hatch_rate := 2 / 3

noncomputable def total_birds : ℕ :=
  num_female_doves + (num_female_doves * eggs_per_dove * dove_hatch_rate).toNat +
  num_male_pigeons + (num_male_pigeons * eggs_per_pigeon * pigeon_hatch_rate).toNat

theorem total_birds_correct :
  total_birds = 647 := by
  sorry

end total_birds_correct_l408_408859


namespace temperature_negation_l408_408807

theorem temperature_negation (h : 1 = +1) : -1 = -1 :=
by
  sorry

end temperature_negation_l408_408807


namespace sequence_sum_l408_408054

noncomputable def r : ℝ := 1 / 4
def t₀ : ℝ := 4096
def t₁ : ℝ := t₀ * r
def t₂ : ℝ := t₁ * r
def x : ℝ := t₂ * r
def y : ℝ := x * r

theorem sequence_sum : x + y = 80 := by
  -- The proof will go here
  sorry

end sequence_sum_l408_408054


namespace polygon_to_rectangle_polygons_equiv_decomposition_l408_408338

-- Define what it means for a polygon to be decomposable into a rectangle of a specific side length
def polygon (P : Type) : Prop := 
sorry -- A rigorous definition of polygon can be given here, for the sake of this problem we skip this.

def decomposable_to_rectangle (P : Type) (l : ℝ) : Prop := 
sorry -- This should define a process of transformation from polygon P to a rectangle with one side of length l.

-- Part (a): Any polygon can be decomposed and reassembled into a rectangle with one side being 1.
theorem polygon_to_rectangle : ∀ (P : Type), polygon P → decomposable_to_rectangle P 1 :=
by 
  sorry

-- Define what it means for two polygons to have equal area.
def equal_area (P1 P2 : Type) [polygon P1] [polygon P2] : Prop :=
sorry -- This definition should formalize the concept of two polygons having the same area.

-- Part (b): Given two polygons of equal area, show the first can be reassembled into the second.
theorem polygons_equiv_decomposition : 
  ∀ (P1 P2 : Type) 
  (h1 : polygon P1) 
  (h2 : polygon P2),
  equal_area P1 P2 → decomposable_to_rectangle P1 1 → 
  decomposable_to_rectangle P2 1 → 
  ∃ Q : Type, decomposable_to_rectangle Q 1 ∧ decomposable_to_rectangle P2 1 :=
by 
  sorry

end polygon_to_rectangle_polygons_equiv_decomposition_l408_408338


namespace s_point_value_l408_408590

def has_s_point (f g : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = g x₀ ∧ deriv f x₀ = deriv g x₀

noncomputable def f (a : ℝ) : ℝ → ℝ := λ x, a * x^2 - 1
noncomputable def g (a : ℝ) : ℝ → ℝ := λ x, log (a * x)

theorem s_point_value (a : ℝ) (x₀ : ℝ) (h : has_s_point (f a) (g a) x₀) : a = 2 / real.exp(1) :=
sorry

end s_point_value_l408_408590


namespace at_least_half_boys_probability_l408_408568

noncomputable def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

noncomputable def probability_at_least_half_boys : ℝ :=
  let n := 7 in
  let p := 0.6 in
  let probs := [binomial_prob n 4 p, binomial_prob n 5 p, binomial_prob n 6 p, binomial_prob n 7 p] in
  list.sum probs

theorem at_least_half_boys_probability :
  probability_at_least_half_boys ≈ 0.6506 :=
by
  sorry

end at_least_half_boys_probability_l408_408568


namespace f_zero_f_analytic_A_cap_complement_B_l408_408904

variable {f : ℝ → ℝ} {a : ℝ}

-- Conditions
axiom h1 : ∀ x y : ℝ, f(x + y) - f(y) = x * (x + 2 * y + 1)
axiom h2 : f(1) = 0

-- Proof goal 1: f(0) = -2
theorem f_zero : f 0 = -2 :=
sorry

-- Proof goal 2: ∀ x, f(x) = x^2 + x - 2
theorem f_analytic : ∀ x : ℝ, f x = x^2 + x - 2 :=
sorry

-- Definition of P and Q conditions
def P (a : ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 3 / 4 → f x + 3 < 2 * x + a
def Q (a : ℝ) : Prop := ∀ x, -2 ≤ x ∧ x ≤ 2 → ∀ y, f(x) - a * x < f(y) - a * y → x < y

-- Define sets A and B
def A : set ℝ := { a | P a }
def B : set ℝ := { a | Q a }

-- Proof goal 3: A ∩ (ℝ \ B) = (1, 5)
theorem A_cap_complement_B : A ∩ (λ x, x ∉ B) = (λ x, (1 < x) ∧ (x < 5)) :=
sorry

end f_zero_f_analytic_A_cap_complement_B_l408_408904


namespace correct_statements_l408_408241

-- Definitions for the problem statements
def is_direct_proportion (A B : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, A * x = k * B * y

def is_inverse_proportion (A B : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, A * x * B * y = k

def is_not_proportional (A B : ℝ) : Prop :=
  ¬(is_direct_proportion A B ∨ is_inverse_proportion A B)

-- Conditions given in the problem for each statement
def statement1 : Prop :=
  is_direct_proportion (area_covered / number_of_bricks) 1

def statement2 : Prop :=
  is_inverse_proportion (average_distance_per_minute * time_taken) 1

def statement3 : Prop :=
  is_not_proportional (perimeter_of_square / side_length) 1

def statement4 : Prop :=
  is_not_proportional (area_of_circle / radius) 1

-- Mathematically equivalent proof problem
theorem correct_statements : {1, 2, 4} = {i | i = 1 ∨ i = 2 ∨ i = 4} :=
by {
  sorry -- Proof will go here
}

end correct_statements_l408_408241


namespace next_four_digit_number_after_1818_l408_408685

-- Formalization of the given conditions as Lean definitions
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def next_valid_number (n : ℕ) : ℕ :=
  let d1 := n / 1000 in
  let d2 := (n / 100) % 10 in
  let d3 := (n / 10) % 10 in
  let d4 := n % 10 in
  let next_n := (d1 * 10 + d2) * 100 + (d3 * 10 + d4) + 14*10 in -- moving to the next valid 4 digits in base 10 series
  next_n

def condition (n : ℕ) : Prop :=
  let d3d4 := n % 100 in
  is_perfect_square (18 * d3d4)

theorem next_four_digit_number_after_1818 :
  ∀ n : ℕ, 
    condition 1818 → 
    next_valid_number 1818 = 1832 :=
begin
  intro n,
  intro cond,
  sorry
end

end next_four_digit_number_after_1818_l408_408685


namespace sin_theta_value_l408_408887

theorem sin_theta_value (θ : ℝ) (h1 : 10 * (Real.tan θ) = 4 * (Real.cos θ)) (h2 : 0 < θ ∧ θ < π) : Real.sin θ = 1/2 :=
by
  sorry

end sin_theta_value_l408_408887


namespace range_of_m_l408_408866

theorem range_of_m {m : ℝ} (p : ∃ x : ℝ, m * x^2 + 2 ≤ 0) (q : ∀ x : ℝ, x^2 - 2 * m * x + 1 > 0) (h : ¬ (p ∨ q)) : m ≥ 1 := 
sorry

end range_of_m_l408_408866


namespace player_b_can_determine_a_l408_408702

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def f : ℕ → ℕ
| n := if is_prime n then 1 else 0

theorem player_b_can_determine_a (a : ℕ) (h : 1 ≤ a ∧ a ≤ 2006) :
  ∃ S : fin 2005 → ℕ, ∀ i j : ℕ, i ≠ j →
  (0 ≤ i ∧ i ≤ 2005) ∧ (0 ≤ j ∧ j ≤ 2005) ∧ f (S i + a) ≠ f (S j + a) :=
sorry

end player_b_can_determine_a_l408_408702


namespace six_pow_k_minus_k_pow_six_l408_408939

theorem six_pow_k_minus_k_pow_six (k : ℕ) (h1 : 18^k ∣ 624938) : 6^k - k^6 = 1 := by
  have h2 : k = 0 := by sorry
  rw [h2]
  simp
  exact one_sub_zero

end six_pow_k_minus_k_pow_six_l408_408939


namespace arithmetic_sequence_a1_geometric_sequence_sum_l408_408662

-- Definition of the arithmetic sequence problem
theorem arithmetic_sequence_a1 (a_n s_n : ℕ) (d : ℕ) (h1 : a_n = 32) (h2 : s_n = 63) (h3 : d = 11) :
  ∃ a_1 : ℕ, a_1 = 10 :=
by
  sorry

-- Definition of the geometric sequence problem
theorem geometric_sequence_sum (a_1 q : ℕ) (h1 : a_1 = 1) (h2 : q = 2) (m : ℕ) :
  let a_m := a_1 * (q ^ (m - 1))
  let a_m_sq := a_m * a_m
  let sm'_sum := (1 - 4^m) / (1 - 4)
  sm'_sum = (4^m - 1) / 3 :=
by
  sorry

end arithmetic_sequence_a1_geometric_sequence_sum_l408_408662


namespace matrix_power_50_eq_l408_408571

noncomputable def A : Matrix (Fin 2) (Fin 2) ℤ := ![![2, 1], [-4, -1]]

theorem matrix_power_50_eq :
  matrix.pow A 50 = (50 * 8^49 : ℤ) • ![![2, 1], [-4, -1]] - (399 * 8^49 : ℤ) • (1 : Matrix (Fin 2) (Fin 2) ℤ) :=
by
  sorry

end matrix_power_50_eq_l408_408571


namespace gretchen_flavors_l408_408148

/-- 
Gretchen's local ice cream shop offers 100 different flavors. She tried a quarter of the flavors 2 years ago and double that amount last year. Prove how many more flavors she needs to try this year to have tried all 100 flavors.
-/
theorem gretchen_flavors (F T2 T1 T R : ℕ) (h1 : F = 100)
  (h2 : T2 = F / 4)
  (h3 : T1 = 2 * T2)
  (h4 : T = T2 + T1)
  (h5 : R = F - T) : R = 25 :=
sorry

end gretchen_flavors_l408_408148


namespace polynomial_nonnegative_iff_eq_l408_408819

variable {R : Type} [LinearOrderedField R]

def polynomial_p (x a b c : R) : R :=
  (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem polynomial_nonnegative_iff_eq (a b c : R) :
  (∀ x : R, polynomial_p x a b c ≥ 0) ↔ a = b ∧ b = c :=
by
  sorry

end polynomial_nonnegative_iff_eq_l408_408819


namespace correct_statement_is_F4_l408_408209

variable (f : ℝ → ℝ)

def F1 (x : ℝ) : ℝ := f(x) * f(-x)
def F2 (x : ℝ) : ℝ := f(x) * abs(f(-x))
def F3 (x : ℝ) : ℝ := f(x) - f(-x)
def F4 (x : ℝ) : ℝ := f(x) + f(-x)

-- Prove that F1 is even, F2 is not necessarily odd, F3 is odd, and F4 is even.
theorem correct_statement_is_F4 :
  (∀ x, F1 f x = F1 f (-x)) ∧ 
  ¬ (∀ x, F2 f x = -F2 f (-x)) ∧ 
  (∀ x, F3 f x = -F3 f (-x)) ∧ 
  (∀ x, F4 f x = F4 f (-x)) :=
sorry

end correct_statement_is_F4_l408_408209


namespace probability_of_winning_pair_l408_408303

-- Conditions: Define the deck composition and the winning pair.
inductive Color
| Red
| Green
| Blue

inductive Label
| A
| B
| C

structure Card :=
(color : Color)
(label : Label)

def deck : List Card :=
  [ {color := Color.Red, label := Label.A},
    {color := Color.Red, label := Label.B},
    {color := Color.Red, label := Label.C},
    {color := Color.Green, label := Label.A},
    {color := Color.Green, label := Label.B},
    {color := Color.Green, label := Label.C},
    {color := Color.Blue, label := Label.A},
    {color := Color.Blue, label := Label.B},
    {color := Color.Blue, label := Label.C} ]

def is_winning_pair (c1 c2 : Card) : Prop :=
  c1.color = c2.color ∨ c1.label = c2.label

-- Question: Prove the probability of drawing a winning pair.
theorem probability_of_winning_pair :
  (∃ (c1 c2 : Card), c1 ∈ deck ∧ c2 ∈ deck ∧ c1 ≠ c2 ∧ is_winning_pair c1 c2) →
  (∃ (c1 c2 : Card), c1 ∈ deck ∧ c2 ∈ deck ∧ c1 ≠ c2) →
  (9 + 9) / 36 = 1 / 2 :=
sorry

end probability_of_winning_pair_l408_408303


namespace fraction_august_tips_l408_408739

variable (A : ℝ) -- Define the average monthly tips A for March, April, May, June, July, and September
variable (august_tips : ℝ) -- Define the tips for August
variable (total_tips : ℝ) -- Define the total tips for all months

-- Define the conditions
def condition_average_tips : Prop := total_tips = 12 * A
def condition_august_tips : Prop := august_tips = 6 * A

-- The theorem we need to prove
theorem fraction_august_tips :
  condition_average_tips A total_tips →
  condition_august_tips A august_tips →
  (august_tips / total_tips) = (1 / 2) :=
by
  intros h_avg h_aug
  rw [condition_average_tips] at h_avg
  rw [condition_august_tips] at h_aug
  rw [h_avg, h_aug]
  simp
  sorry

end fraction_august_tips_l408_408739


namespace seq_arithmetic_seq_a_general_formula_l408_408917

noncomputable def a : ℕ → ℚ
| 1     := 1
| n + 2 := (2 * (S (n + 1))^2) / (2 * (S (n+1)) - 1)

def S : ℕ → ℚ
| 1     := 1
| n + 2 := 1 / (2 * n + 1)

theorem seq_arithmetic_seq (n : ℕ) (h : n ≥ 2) : 
    (1 / S n) - (1 / S (n-1)) = 2 :=
sorry

theorem a_general_formula (n : ℕ) : 
    a n = if n = 1 then 1 else -2 / ((2 * n - 1) * (2 * n - 3)) :=
sorry

end seq_arithmetic_seq_a_general_formula_l408_408917


namespace arcsin_zero_l408_408012

theorem arcsin_zero : Real.arcsin 0 = 0 := by
  sorry

end arcsin_zero_l408_408012


namespace ellipse_slope_k_l408_408883

theorem ellipse_slope_k (a b k : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (sqrt 3) / 2 = sqrt (a^2 - b^2) / a) 
  (h4 : k > 0) 
  (h5 : ∀ F A B : ℝ × ℝ, 
    (a^2 * (F.1 - A.1) + b^2 * (F.2 - A.2) = 3 * b^2 * (F.2 - B.2)) 
    → (F.1 = sqrt (a^2 - b^2)) 
    ∧ (F.2 = 0)) : 
  k = sqrt 2 := 
sorry

end ellipse_slope_k_l408_408883


namespace max_value_of_symmetric_function_l408_408947

def f (x a b : ℝ) : ℝ := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function (a b : ℝ) (h_sym : ∀ x : ℝ, f x a b = f (-4 - x) a b) : 
  ∃ x : ℝ, (∀ y : ℝ, f y a b ≤ f x a b) ∧ f x a b = 16 :=
by
  sorry

end max_value_of_symmetric_function_l408_408947


namespace sequence_sum_l408_408026

theorem sequence_sum :
  (∀ r: ℝ, 
    (∀ x y: ℝ,
      r = 1 / 4 → 
      x = 256 * r → 
      y = x * r → 
      x + y = 80)) :=
by sorry

end sequence_sum_l408_408026


namespace unique_zero_range_l408_408913

theorem unique_zero_range (a : ℝ) (x0 : ℝ) (h1 : ∀ x, ax^3 - 3x^2 + 1 = 0 ↔ x = x0) (h2 : x0 < 0) :
  a > 2 :=
sorry

end unique_zero_range_l408_408913


namespace billy_hiking_distance_l408_408798

def billy_displacement : ℝ :=
  let v1 := (0, 3)  -- Movement 1: 3 miles north
  let v2 := (5 * real.cos (real.pi / 4), 5 * real.sin (real.pi / 4))  -- Movement 2: 45 degrees eastward, 5 miles
  let v3 := (4 * real.cos (3 * real.pi / 4), 4 * real.sin (3 * real.pi / 4))  -- Movement 3: 45 degrees southward, 4 miles
  let total_x := v1.1 + v2.1 + v3.1
  let total_y := v1.2 + v2.2 + v3.2
  (total_x^2 + total_y^2).sqrt

theorem billy_hiking_distance : billy_displacement = real.sqrt (45 + 3 * real.sqrt 2) := 
  by
  sorry

end billy_hiking_distance_l408_408798


namespace sector_properties_l408_408877

theorem sector_properties
  (R : ℝ) (L : ℝ)
  (hR : R = 8)
  (hL : L = 12) :
  let α := L / R in
  α = 3 / 2 ∧ (1 / 2) * L * R = 48 :=
by {
  let α := L / R,
  have h1 : α = 3 / 2, by sorry,
  have h2 : (1 / 2) * L * R = 48, by sorry,
  exact ⟨h1, h2⟩
}

end sector_properties_l408_408877


namespace evaluate_log_expression_l408_408430

theorem evaluate_log_expression (h : 1000 = 10^3) :
  (log 10 (8 * log 10 1000)) ^ 2 = (log 10 24) ^ 2 :=
by
  sorry

end evaluate_log_expression_l408_408430


namespace find_projection_l408_408922

noncomputable def projection_vector (u v : ℝ × ℝ) : ℝ × ℝ :=
  let d := (u.1 - v.1, u.2 - v.2)
  let t := -(u.1 * d.1 + u.2 * d.2) / (d.1 * d.1 + d.2 * d.2)
  (u.1 + t * d.1, u.2 + t * d.2)

theorem find_projection :
  let u := (3 : ℝ, 2)
  let v := (1 : ℝ, 5)
  projection_vector u v = (3, 2) :=
by
  let u := (3 : ℝ, 2)
  let v := (1 : ℝ, 5)
  let d := (u.1 - v.1, u.2 - v.2)
  let t := -(u.1 * d.1 + u.2 * d.2) / (d.1 * d.1 + d.2 * d.2)
  have h : projection_vector u v = (u.1 + t * d.1, u.2 + t * d.2) := rfl
  have ht : t = 0 := sorry -- This part needs proof steps, but is skipped for statement only
  rw [ht]
  simp [projection_vector, d, t]
  sorry

end find_projection_l408_408922


namespace triangle_perimeter_l408_408191

noncomputable theory

variables {A B C : ℝ}
variables {a b c : ℝ}

-- Define the perimeter of the triangle
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Define the conditions
axiom cos_eq_half : cos A = 1/2
axiom given_equation : c * cos B + b * cos C = 2 * a * cos A
axiom side_a : a = 2
axiom area_eq : (1 / 2) * b * c * sin A = sqrt 3

-- The theorem to prove
theorem triangle_perimeter : perimeter a b c = 6 :=
sorry

end triangle_perimeter_l408_408191


namespace next_four_digit_number_l408_408670

def isPerfectSquare (n : ℕ) : Prop :=
∃ m : ℕ, m * m = n

def satisfiesProperty (n : ℕ) : Prop :=
  ∃ a b : ℕ, 10 ≤ a ∧ a < 100 ∧ 0 ≤ b ∧ b < 10 ∧ n = 1800 + a * 10 + b ∧ isPerfectSquare (18 * a)

theorem next_four_digit_number (n : ℕ) (h₀ : n = 1818) (h₁ : satisfiesProperty n) :
  ∃ m : ℕ, m > n ∧ m < 2000 ∧ satisfiesProperty m ∧ (∀ k : ℕ, n < k ∧ k < m → ¬ satisfiesProperty k) :=
begin
  use 1832,
  split,
  { exact 1832 > 1818, },
  split,
  { exact 1832 < 2000, },
  split,
  { unfold satisfiesProperty,
    use 32,
    use 0,
    split, { exact nat.le_of_lt 32 10 100, },
    split, { exact nat.lt_of_le_of_lt 0 32 40, },
    split, { exact nat.le_of_lt 0 0 10, },
    split, { exact nat.lt_of_le_of_lt 0 0 10, },
    split, { refl, },
    { unfold isPerfectSquare,
      use 24,
      exact nat.mul_self_eq 18 32 32, }, },
  { intros k hk,
    unfold satisfiesProperty at hk,
    cases hk with a ha,
    cases ha with b hb,
    cases hb with hb1 hb2,
    cases hb2 with hb3 hb4,
    cases hb4 with hb5 hb6,
    cases hb6 with hb7 hb8,
    cases hb8 with hb9 hb10,
    cases hb10 with hb11 hb12,
    exact nat.ne_of_lt_decides hk hb2 hb8 24, },
end

end next_four_digit_number_l408_408670


namespace sequence_not_periodic_l408_408751

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def sequence (k : ℕ) : ℕ :=
  if sum_of_digits k % 2 = 0 then 0 else 1

theorem sequence_not_periodic : ¬ ∃ (ℓ : ℕ), ℓ > 0 ∧ ∀ (k : ℕ), sequence k = sequence (k + ℓ) :=
by
  sorry

end sequence_not_periodic_l408_408751


namespace geometric_locus_is_sphere_l408_408881

noncomputable def geometric_locus (A B C : ℝ × ℝ × ℝ) : set (ℝ × ℝ × ℝ) :=
  { X | (X.1 - A.1)^2 + (X.2 - A.2)^2 + (X.3 - A.3)^2 + (X.1 - B.1)^2 + (X.2 - B.2)^2 + (X.3 - B.3)^2 = (X.1 - C.1)^2 + (X.2 - C.2)^2 + (X.3 - C.3)^2 }

theorem geometric_locus_is_sphere (A B C : ℝ × ℝ × ℝ) :
  ∃ M : ℝ × ℝ × ℝ, ∃ R : ℝ, ∀ X : ℝ × ℝ × ℝ, X ∈ geometric_locus A B C ↔ (X.1 - M.1)^2 + (X.2 - M.2)^2 + (X.3 - M.3)^2 = R^2 :=
sorry

end geometric_locus_is_sphere_l408_408881


namespace complex_modulus_l408_408421

theorem complex_modulus :
  abs ((7 - 4*complex.I) * (3 + 11*complex.I)) = Real.sqrt 8450 :=
by
  sorry

end complex_modulus_l408_408421


namespace find_f_300_l408_408208

noncomputable def f : ℝ → ℝ := sorry

axiom f_property (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f(x * y) = f(x) / y
axiom f_at_250 : f 250 = 4

theorem find_f_300 : f 300 = 10 / 3 := sorry

end find_f_300_l408_408208


namespace vector_expression_l408_408144

-- Define the vectors a, b, and c.
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (0, 3) - a
def c : ℝ × ℝ := (1, 5)

-- State the theorem to be proven.
theorem vector_expression :
  c = 2 • a + b :=
by
-- Include specific conditions as sorries to express the proof obligations.
calc   c = (1, 5)           : by rfl
     ... = 2 • (1, 2) + b    : by { sorry }
     ... = 2 • a + b        : by rfl

-- Sorry to skip the detailed proof steps.

end vector_expression_l408_408144


namespace range_of_f_l408_408451

def f (x : ℝ) : ℝ := x + (real.sqrt (2 * x - 1))

theorem range_of_f :
  set.Ici (1 / 2) = set.range f := 
sorry

end range_of_f_l408_408451


namespace largest_in_arithmetic_progression_l408_408251

theorem largest_in_arithmetic_progression 
  (a d : ℝ)
  (h1: a^3 + (a + d)^3 + (a + 2*d)^3 + (a + 3*d)^3 + (a + 4*d)^3 + (a + 5*d)^3 + (a + 6*d)^3 = 0)
  (h2: a^2 + (a + d)^2 + (a + 2*d)^2 + (a + 3*d)^2 + (a + 4*d)^2 + (a + 5*d)^2 + (a + 6*d)^2 = -224) :
  max a (max (a + d) (max (a + 2*d) (max (a + 3*d) (max (a + 4*d) (max (a + 5*d) (a + 6*d)))))) = 6 * sqrt 2 :=
sorry

end largest_in_arithmetic_progression_l408_408251


namespace competition_duration_correct_l408_408963

-- Definitions of the conditions in the problem
def jackson_catches_per_day : ℕ := 6
def jonah_catches_per_day : ℕ := 4
def george_catches_per_day : ℕ := 8
def total_fishes_caught : ℕ := 90

-- Definition of the duration of the competition
def competition_duration : ℕ := 5

-- Lean 4 statement containing the proof problem
theorem competition_duration_correct 
  (jackson_catches_per_day jonah_catches_per_day george_catches_per_day total_fishes_caught : ℕ)
  (h1 : jackson_catches_per_day = 6)
  (h2 : jonah_catches_per_day = 4)
  (h3 : george_catches_per_day = 8)
  (h4 : total_fishes_caught = 90) :
  let d := 5 in
  6*d + 4*d + 8*d = total_fishes_caught := 
by
  sorry

end competition_duration_correct_l408_408963


namespace problem1a_problem1b_problem2_l408_408140

-- Define the set M
def M := { x : ℝ | x^2 - 3 * x - 18 ≤ 0 }

-- Define the set N with parameter a
def N (a : ℝ) := { x : ℝ | 1 - a ≤ x ∧ x ≤ 2 * a + 1 }

-- Problem 1a: Find M ∩ N when a = 3
theorem problem1a : 
  let a := 3 in M ∩ N a = { x : ℝ | -2 ≤ x ∧ x ≤ 6 } :=
by sorry

-- Problem 1b: Find the complement of N when a = 3
theorem problem1b : 
  let a := 3 in { x : ℝ | ¬ (N a x) } = { x : ℝ | x < -2 ∨ x > 7 } :=
by sorry

-- Problem 2: Find the range of a given M ⊆ N
theorem problem2 : 
  (∀ a : ℝ, M ⊆ N a → a ≥ 4) :=
by sorry

end problem1a_problem1b_problem2_l408_408140


namespace similar_prism_triples_l408_408770

open Int Nat

theorem similar_prism_triples :
  let b := 1995 in
  let count_divisors n := (finset.Icc 1 n).filter (λ d, n % d = 0).card in
  let num_triples := (count_divisors (1995^2) + 1) / 2 - 1 in
  num_triples = 40 :=
by
  let b := 1995
  let n := b ^ 2
  let count_divisors n := (finset.Icc 1 n).filter (λ d, n % d = 0).card
  let num_triples := (count_divisors n + 1) / 2 - 1
  sorry

end similar_prism_triples_l408_408770


namespace vector_magnitude_problem_l408_408925

open Real

variables {a b : ℝ^3}

theorem vector_magnitude_problem (h1 : ‖a‖ = 1) 
                                (h2 : ‖b‖ = 2)
                                (h3 : ‖a + b‖ = ‖a - b‖) : 
  ‖2 • a + b‖ = 2 * sqrt 2 :=
sorry

end vector_magnitude_problem_l408_408925


namespace simplify_trig_expression_l408_408628

-- Definitions of trigonometric identities as assumptions
lemma trig_identities (x : ℝ) :
  sin(x + 2 * π) = sin x ∧
  cos(x + π) = -cos x ∧
  sin(x + π) = -sin x ∧
  cos(x + (π / 2)) = -sin x ∧
  sin(x + (π / 2)) = cos x := 
begin
  split;
  { try {apply sin_add_two_pi},
    try {apply cos_add_pi},
    try {apply sin_add_pi},
    try {apply cos_add_half_pi},
    apply sin_add_half_pi },
end

-- Main theorem to be proved
theorem simplify_trig_expression (α : ℝ) :
  (sin(2*π - α) * cos(π + α) * cos(π/2 + α) * cos(11*π/2 - α)) / 
  (cos(π - α) * sin(3*π - α) * sin(-π - α) * sin(9*π/2 + α)) = -tan(α)^2 :=
by sorry

end simplify_trig_expression_l408_408628


namespace value_of_a3_plus_a4_l408_408938

theorem value_of_a3_plus_a4 (a : ℕ → ℝ) (h : ∀ x ∈ Ioo (-0.5 : ℝ) 1, 
  ∃ f : ℕ → ℝ, (Σ' i, f i * x ^ i) =  x / (1 + x - 2 * x^2)):
  a 3 + a 4 = -4 := sorry

end value_of_a3_plus_a4_l408_408938


namespace units_digit_of_sum_of_squares_of_first_2023_odd_integers_is_5_l408_408716

theorem units_digit_of_sum_of_squares_of_first_2023_odd_integers_is_5 :
  (sum (fun n : ℕ => ((2 * n + 1) ^ 2) % 10) (range 2023)) % 10 = 5 :=
sorry

end units_digit_of_sum_of_squares_of_first_2023_odd_integers_is_5_l408_408716


namespace sequence_x_y_sum_l408_408047

theorem sequence_x_y_sum (r : ℝ) (x y : ℝ)
  (h₁ : r = 1 / 4)
  (h₂ : x = 256 * r)
  (h₃ : y = x * r) :
  x + y = 80 :=
by
  sorry

end sequence_x_y_sum_l408_408047


namespace df_fe_ratio_l408_408657

noncomputable def triangle_data (A B C : Type) [metric_space A] (a b c : dist) : Prop :=
AB < AC ∧ a + b + c = 7 * a

noncomputable def inscribed_circle_conditions (A B C E D F : Type) [metric_space A] : Prop :=
(inscribed_circle A B C E) ∧ (diameter_of_inscribed_circle D E) ∧ (median_intersect F (median A B C) D E)

theorem df_fe_ratio {A B C E D F : Type} [metric_space A]
    (h_triangle : triangle_data A B C (dist A B) (dist A C) (dist B C))
    (h_inscribed_conditions : inscribed_circle_conditions A B C E D F) : 
    DF / FE = 5 / 7 := sorry

end df_fe_ratio_l408_408657


namespace monotonicity_f_range_of_m_l408_408907

noncomputable def f (x a : ℝ) : ℝ := Real.log x - a / x
noncomputable def g (x : ℝ) : ℝ := f x 2 + 2 * x - 6 * Real.log x
noncomputable def h (x m : ℝ) : ℝ := x^2 - m * x + 4
noncomputable def g' (x : ℝ) : ℝ := (2 * x^2 - 5 * x + 2) / x^2

theorem monotonicity_f (a : ℝ) : (∀ x > 0, f' x a > 0) ↔ (∀ x > 0, a ≥ 0) ∨ (∀ x > 0, a < 0 ∧ x ≠ -a ∧ (f' x a < 0 ↔ x < -a)) :=
sorry

theorem range_of_m (m : ℝ) : 
  (∃ x1 ∈ Ioo 0 1, ∀ x2 ∈ Icc 1 2, g x1 ≥ h x2 m) → 
  m ≥ 8 - 5 * Real.log 2 :=
sorry

end monotonicity_f_range_of_m_l408_408907


namespace number_of_real_roots_l408_408579

noncomputable def det := λ (x a b c k : ℝ), 
  x * (x^2 + k^2 * a^2 + b^2 + c^2) = 0

theorem number_of_real_roots
  (a b c k : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hk : k ≠ 0) :
  ∃ x : ℝ, det x a b c k = 0 ∧ ∀ y : ℝ, det y a b c k = 0 → y = x :=
by
  sorry

end number_of_real_roots_l408_408579


namespace units_digit_of_sum_of_squares_of_first_2023_odd_integers_is_5_l408_408715

theorem units_digit_of_sum_of_squares_of_first_2023_odd_integers_is_5 :
  (sum (fun n : ℕ => ((2 * n + 1) ^ 2) % 10) (range 2023)) % 10 = 5 :=
sorry

end units_digit_of_sum_of_squares_of_first_2023_odd_integers_is_5_l408_408715


namespace no_positive_integer_solutions_l408_408744

theorem no_positive_integer_solutions (x y : ℕ) (hx : x > 0) (hy : y > 0) : x^5 ≠ y^2 + 4 := 
by sorry

end no_positive_integer_solutions_l408_408744


namespace increasing_interval_of_f_l408_408278

-- Define the function y = x - e^x
def f (x : ℝ) : ℝ := x - Real.exp x

-- State the main theorem
theorem increasing_interval_of_f : ∀ x : ℝ, x < 0 → f' x > 0 :=
by
  sorry

end increasing_interval_of_f_l408_408278


namespace verify_correct_propositions_l408_408909

noncomputable def f (x α : ℝ) : ℝ := sin (x - α) + 2 * cos x

def proposition2 (α : ℝ) : Prop :=
  ∃ x θ : ℝ, f(x, α) = sqrt (5 - 4 * sin α) * sin (x + θ) ∧
             -sqrt (5 - 4 * sin α) ≤ 0 -- minimum value proof related assumption
             
def proposition3 (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * π + π / 2 ∨ α = k * π - π / 2

def proposition5 (α x : ℝ) : Prop :=
  (α = π / 6) → (x = -π / 3) → sin (x - π / 3) + 2 * cos x = 0

def correct_propositions : set ℕ := {2, 3, 5}

theorem verify_correct_propositions (α : ℝ) : 
  (proposition2 α) ∧ 
  (proposition3 α) ∧ 
  (proposition5 α (-π / 3)) → correct_propositions = {2, 3, 5} :=
sorry

end verify_correct_propositions_l408_408909


namespace find_value_at_one_l408_408647

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 - m * x + 3

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f(x) ≤ f(y)

def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f(x) ≥ f(y)

theorem find_value_at_one :
  (∀ x, -∞ < x ∧ x < -2 → deriv (f x) (2 * x^2 - m * x + 3) ≤ 0) ∧ 
  (∀ x, -2 ≤ x ∧ x < +∞ → deriv (f x) (2 * x^2 - m * x + 3) ≥ 0) →
  f 1 (-8) = 13 :=
by
  sorry

end find_value_at_one_l408_408647


namespace minimum_value_is_4_l408_408584

noncomputable def minimum_value (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z) : ℝ :=
  real.sqrt ((x^2 + y^2 + z^2) * (4 * x^2 + y^2 + z^2)) / (x * y * z)

theorem minimum_value_is_4 (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z) : (minimum_value x y z h) = 4 :=
sorry

end minimum_value_is_4_l408_408584


namespace evaluate_magnitude_product_l408_408420

-- Definitions of complex numbers
def z1 := Complex.mk 7 (-4)
def z2 := Complex.mk 3 11

-- The magnitude of z1
def magnitude_z1 := Complex.abs z1

-- The magnitude of z2
def magnitude_z2 := Complex.abs z2

-- Lean 4 statement expressing the problem and its final answer
theorem evaluate_magnitude_product : Complex.abs (z1 * z2) = Real.sqrt 8450 := by
  sorry

end evaluate_magnitude_product_l408_408420


namespace common_difference_of_arithmetic_sequence_l408_408480

/--
Given an arithmetic sequence {a_n}, the sum of the first n terms is S_n,
a_3 and a_7 are the two roots of the equation 2x^2 - 12x + c = 0,
and S_{13} = c.
Prove that the common difference of the sequence {a_n} satisfies d = -3/2 or d = -7/4.
-/
theorem common_difference_of_arithmetic_sequence 
  (S : ℕ → ℚ)
  (a : ℕ → ℚ)
  (c : ℚ)
  (h1 : ∃ a_3 a_7, (2 * a_3^2 - 12 * a_3 + c = 0) ∧ (2 * a_7^2 - 12 * a_7 + c = 0))
  (h2 : S 13 = c) :
  ∃ d : ℚ, d = -3/2 ∨ d = -7/4 :=
sorry

end common_difference_of_arithmetic_sequence_l408_408480


namespace tan_half_odd_periodic_l408_408082

noncomputable def tan_half (x : ℝ) : ℝ := Real.tan (x / 2)

theorem tan_half_odd_periodic :
  (∀ x : ℝ, tan_half (-x) = -tan_half x) ∧ (∀ x : ℝ, tan_half (x + 2 * Real.pi) = tan_half x) :=
by
  split
  · intro x
    sorry
  · intro x
    sorry

end tan_half_odd_periodic_l408_408082


namespace solve_equation_l408_408631

theorem solve_equation : ∀ (x : ℝ), -2 * x + 3 - 2 * x + 3 = 3 * x - 6 → x = 12 / 7 :=
by 
  intro x
  intro h
  sorry

end solve_equation_l408_408631


namespace product_of_300_and_even_is_perfect_square_product_1200_is_perfect_square_l408_408458

theorem product_of_300_and_even_is_perfect_square :
  ∃ (p : ℤ), even p ∧ p > 3 ∧ (300 * p = 1200) :=
by
  use 4
  split
  · exact even_iff_two_dvd.mpr ⟨2, (by ring)⟩
  split
  · linarith
  · ring

theorem product_1200_is_perfect_square :
  ∃ (n : ℤ), n^2 = 1200 :=
by
  use 40
  ring
  sorry  -- Proof that 40^2 = 1200

end product_of_300_and_even_is_perfect_square_product_1200_is_perfect_square_l408_408458


namespace range_of_m_l408_408512

def setA (m : ℝ) : set (ℝ × ℝ) :=
  { p | let (x, y) := p in (m / 2) ≤ (x - 2) ^ 2 + y ^ 2 ∧ (x - 2) ^ 2 + y ^ 2 ≤ m ^ 2 }

def setB (m : ℝ) : set (ℝ × ℝ) :=
  { p | let (x, y) := p in 2 * m ≤ x + y ∧ x + y ≤ 2 * m + 1 }

def range_m := {m : ℝ | 1 / 2 ≤ m ∧ m ≤ 2 + Real.sqrt 2}

theorem range_of_m (m : ℝ) (nonempty : ∃ (p : ℝ × ℝ), p ∈ setA m ∧ p ∈ setB m) :
  m ∈ range_m :=
  sorry

end range_of_m_l408_408512


namespace range_exponential_function_l408_408660

theorem range_exponential_function : ∀ x : ℝ, 0 < 3 ^ x :=
by
  sorry

end range_exponential_function_l408_408660


namespace reflected_ray_is_correct_l408_408354

def point := ℝ × ℝ

variable (emitted_point : point) (slope : ℝ)

noncomputable def incident_ray_equation (emitted_point : point) (slope : ℝ) : ℝ → ℝ → Prop := 
  λ x y, y - emitted_point.2 = slope * (x - emitted_point.1)

noncomputable def symmetric_point (p : point) : point :=
  (-p.1, p.2)

noncomputable def reflected_ray_equation (emitted_point : point) (slope : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x + 2 * y - 4 = 0

theorem reflected_ray_is_correct :
  ∀ (p1 p2 : point),
    emitted_point = (2, 3) →
    slope = 1 / 2 →
    incident_ray_equation emitted_point slope 0 2 →
    p2 = symmetric_point emitted_point →
    reflected_ray_equation emitted_point slope (-2) 3 :=
begin
  intros,
  -- Proof goes here
  sorry
end

end reflected_ray_is_correct_l408_408354


namespace probability_y_le_sin_x_l408_408491

theorem probability_y_le_sin_x : ∀ {x y : ℝ}, (0 ≤ x ∧ x ≤ π / 2) ∧ (0 ≤ y ∧ y ≤ π / 2) → 
  (@measure_theory.measure_space.volume ℝ _ (set.Icc 0 (π / 2)) (set.Icc 0 (π / 2)) ∩ set_of (λ xy : ℝ × ℝ, xy.snd ≤ sin xy.fst)).measure / 
  (@measure_theory.measure_space.volume ℝ _ (set.Icc 0 (π / 2) × set.Icc 0 (π / 2)).measure) = (4 / π^2) :=
begin
  -- Proof goes here
  sorry
end

end probability_y_le_sin_x_l408_408491


namespace log_base_inequality_l408_408155

theorem log_base_inequality (a b : ℝ) (h1 : log a 2 < log b 2) (h2 : log b 2 < 0) : 0 < b ∧ b < a ∧ a < 1 :=
by
  sorry

end log_base_inequality_l408_408155


namespace real_solutions_of_equation_l408_408828

theorem real_solutions_of_equation : 
  ∃! x₁ x₂ : ℝ, (3 * x₁^2 - 10 * x₁ + 7 = 0) ∧ (3 * x₂^2 - 10 * x₂ + 7 = 0) ∧ x₁ ≠ x₂ :=
sorry

end real_solutions_of_equation_l408_408828


namespace one_zero_implies_a_eq_pm2_zero_in_interval_implies_a_in_open_interval_l408_408944

def f (a x : ℝ) : ℝ := x^2 - a * x + 1

theorem one_zero_implies_a_eq_pm2 (a : ℝ) : (∃! x, f a x = 0) → (a = 2 ∨ a = -2) := by
  sorry

theorem zero_in_interval_implies_a_in_open_interval (a : ℝ) : (∃ x, f a x = 0 ∧ 0 < x ∧ x < 1) → 2 < a := by
  sorry

end one_zero_implies_a_eq_pm2_zero_in_interval_implies_a_in_open_interval_l408_408944


namespace circleC_eq_ellipseD_eq_ellipse_not_inside_circle_OM_dot_OL_const_l408_408976

-- Definitions from the problem conditions.
def circle := {p : ℝ × ℝ // p.1^2 + p.2^2 = 1}
def lineAB (m : ℝ) := {p : ℝ × ℝ // p.1 = m}
noncomputable def circleC (m : ℝ) := {p : ℝ × ℝ // (p.1 - m)^2 + p.2^2 = 1 - m^2}
noncomputable def ellipseD (b : ℝ) (h : b > 0) := {p : ℝ × ℝ // p.1^2 / (b^2 + 1) + p.2^2 / b^2 = 1}
def F : ℝ × ℝ := (1, 0)

-- Part 1: Prove the equations of circleC and ellipseD.
theorem circleC_eq (m : ℝ) : (circleC m) = {p : ℝ × ℝ // (p.1 - m)^2 + p.2^2 = 1 - m^2} :=
sorry

theorem ellipseD_eq (b : ℝ) (h : b > 0) : (ellipseD b h) = {p : ℝ × ℝ // p.1^2 / (b^2 + 1) + p.2^2 / b^2 = 1} :=
sorry

-- Part 2: Prove that when b = 1, any point on ellipse D does not lie inside circle C.
theorem ellipse_not_inside_circle (m : ℝ) (h : -1 < m ∧ m < 1) :
  ∀ (p : ellipseD 1 (by norm_num)), (p.1 - m)^2 + p.2^2 ≥ 1 - m^2 :=
sorry

-- Part 3: Prove that \(\overrightarrow{OM} \cdot \overrightarrow{OL} = b^2 + 1\) is a constant value.
theorem OM_dot_OL_const (b : ℝ) (h : b > 0) (M : ℝ × ℝ) (H : M ∈ {p : ellipseD b h // p.2 = 0}) :
  ∀ (PQ : ℝ × ℝ × ℝ × ℝ), 
    let P := (PQ.1, PQ.2), Q := (PQ.3, PQ.4),
        N := (PQ.1, -PQ.2), L := (1, 0) in
    (P ∈ ellipseD b h ∧ Q ∈ ellipseD b h ∧ P.1 ≠ Q.1 ∧ P.2 ≠ Q.2 ∧ 
        (∀ R : ℝ × ℝ, R = Q ∨ R = N → (L.1 - R.1) * (L.2 - R.2) = 0)) →
    (M.1 * 1) = b^2 + 1 :=
sorry

end circleC_eq_ellipseD_eq_ellipse_not_inside_circle_OM_dot_OL_const_l408_408976


namespace valid_speaker_schedules_l408_408780

-- Define the problem parameters
def speakers := {A, B, C, D, E}

def valid_schedules : list (list char) := sorry -- placeholder for valid schedules filtering

def total_valid_schedules : nat := (valid_schedules.length)

-- State the theorem
theorem valid_speaker_schedules : total_valid_schedules = 60 := 
sorry

end valid_speaker_schedules_l408_408780


namespace point_outside_circle_l408_408165

theorem point_outside_circle (a b : ℝ)
  (h_line_intersects_circle : ∃ (x1 y1 x2 y2 : ℝ), 
     x1^2 + y1^2 = 1 ∧ 
     x2^2 + y2^2 = 1 ∧ 
     a * x1 + b * y1 = 1 ∧ 
     a * x2 + b * y2 = 1 ∧ 
     (x1, y1) ≠ (x2, y2)) : 
  a^2 + b^2 > 1 :=
sorry

end point_outside_circle_l408_408165


namespace range_of_a_l408_408885

theorem range_of_a (a : ℝ) (p : ∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) (q : 0 < 2 * a - 1 ∧ 2 * a - 1 < 1) : 
  (1 / 2) < a ∧ a ≤ (2 / 3) :=
sorry

end range_of_a_l408_408885


namespace common_difference_is_7_l408_408185

variable (a : ℕ → ℤ) (d : ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n m : ℕ, n < m → a m = a n + (m - n) * d

axiom a3_plus_a6 (h_arith : is_arithmetic_sequence a d) : a 3 + a 6 = 11
axiom a5_plus_a8 (h_arith : is_arithmetic_sequence a d) : a 5 + a 8 = 39

theorem common_difference_is_7
  (h_arith : is_arithmetic_sequence a d)
  (h1 : a3_plus_a6 h_arith)
  (h2 : a5_plus_a8 h_arith) :
  d = 7 := by
  sorry

end common_difference_is_7_l408_408185


namespace smallest_four_digit_minus_three_digit_eq_903_l408_408582

theorem smallest_four_digit_minus_three_digit_eq_903 :
  let m := Nat.find (λ k, 100 ≤ 7 * k + 3 ∧ 7 * k + 3 < 1000) in
  let n := Nat.find (λ l, 1000 ≤ 7 * l + 3 ∧ 7 * l + 3 < 10000) in
  n - m = 903 :=
by {
  let m := Nat.find (λ k, 100 ≤ 7 * k + 3 ∧ 7 * k + 3 < 1000),
  let n := Nat.find (λ l, 1000 ≤ 7 * l + 3 ∧ 7 * l + 3 < 10000),
  have h1 : m = 14 := sorry,
  have h2 : n = 143 := sorry,
  rw [h1, h2],
  calc 143 * 7 + 3 - (14 * 7 + 3) = 1004 - 101 : by sorry
                          ... = 903         : by refl
}

end smallest_four_digit_minus_three_digit_eq_903_l408_408582


namespace find_principal_l408_408777

noncomputable def principal_amount (A : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  A / Real.exp (r * t)

theorem find_principal : 
  principal_amount 5673981 0.1125 7.5 ≈ 2438971.57 :=
by 
  sorry

end find_principal_l408_408777


namespace gary_sold_his_pet_snake_for_55_l408_408466

theorem gary_sold_his_pet_snake_for_55 (a b c : ℝ) (h1 : a = 73.0) (h2 : b = 128.0) (h3 : c = b - a) : c = 55 :=
by
  rw [h1, h2] at h3
  rw h3
  norm_num

end gary_sold_his_pet_snake_for_55_l408_408466


namespace find_m_l408_408501

-- Define the condition that the equation has a positive root
def hasPositiveRoot (m : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (2 / (x - 2) = 1 - (m / (x - 2)))

-- State the theorem
theorem find_m : ∀ m : ℝ, hasPositiveRoot m → m = -2 :=
by
  sorry

end find_m_l408_408501


namespace sequence_sum_l408_408027

theorem sequence_sum :
  (∀ r: ℝ, 
    (∀ x y: ℝ,
      r = 1 / 4 → 
      x = 256 * r → 
      y = x * r → 
      x + y = 80)) :=
by sorry

end sequence_sum_l408_408027


namespace probability_closer_to_5_than_1_l408_408767

theorem probability_closer_to_5_than_1 :
  let interval := set.Icc (0 : ℝ) 8
  in ∫ x in interval, indicator (set.Ioc 3 8) x ∂volume / ∫ x in interval, (1 : ℝ) ∂volume = 0.6 :=
by
  let interval := set.Icc (0 : ℝ) 8
  have h_volume_interval : volume interval = 8 := sorry
  have h_volume_closer_region : volume (set.Ioc 3 8) = 5 := sorry
  calc
    ∫ x in interval, indicator (set.Ioc 3 8) x ∂volume / ∫ x in interval, (1 : ℝ) ∂volume
        = (∫ x in interval, indicator (set.Ioc 3 8) x ∂volume) / volume interval : sorry
    ... = volume (set.Ioc 3 8) / volume interval : sorry
    ... = 5 / 8 : by rw [h_volume_closer_region, h_volume_interval]
    ... = 0.625 : by norm_num
    ... = 0.6 : sorry

end probability_closer_to_5_than_1_l408_408767


namespace probability_neither_snow_nor_rain_in_5_days_l408_408288

def probability_no_snow (p_snow : ℚ) : ℚ := 1 - p_snow
def probability_no_rain (p_rain : ℚ) : ℚ := 1 - p_rain
def probability_no_snow_and_no_rain (p_no_snow p_no_rain : ℚ) : ℚ := p_no_snow * p_no_rain
def probability_no_snow_and_no_rain_5_days (p : ℚ) : ℚ := p ^ 5

theorem probability_neither_snow_nor_rain_in_5_days
    (p_snow : ℚ) (p_rain : ℚ)
    (h1 : p_snow = 2/3) (h2 : p_rain = 1/2) :
    probability_no_snow_and_no_rain_5_days (probability_no_snow_and_no_rain (probability_no_snow p_snow) (probability_no_rain p_rain)) = 1/7776 := by
  sorry

end probability_neither_snow_nor_rain_in_5_days_l408_408288


namespace MashaGiftsMax_l408_408605

def MashaCandies :=
  { Lastochka Grams Truffle kg PtichyeMoloko kg Citron kg: ℕ
  Lastochka := 2000,
  Truffle := 3000,
  PtichyeMoloko := 4000,
  Citron := 5000
  }

def CandiesConstraint :=
  ∀ (A B C : ℕ), (A ≠ B ∧ B ≠ C ∧ A ≠ C) → (3000 : ℕ)

theorem MashaGiftsMax :
  (2000, 3000, 4000, 5000) → 3000))

end MashaGiftsMax_l408_408605


namespace combined_average_yield_l408_408815

variable (YieldA : ℝ) (PriceA : ℝ)
variable (YieldB : ℝ) (PriceB : ℝ)
variable (YieldC : ℝ) (PriceC : ℝ)

def AnnualIncome (Yield : ℝ) (Price : ℝ) : ℝ :=
  Yield * Price

def TotalAnnualIncome (IncomeA IncomeB IncomeC : ℝ) : ℝ :=
  IncomeA + IncomeB + IncomeC

def TotalInvestment (PriceA PriceB PriceC : ℝ) : ℝ :=
  PriceA + PriceB + PriceC

def CombinedAverageYield (TotalIncome TotalInvestment : ℝ) : ℝ :=
  TotalIncome / TotalInvestment

theorem combined_average_yield :
  YieldA = 0.20 → PriceA = 100 → YieldB = 0.12 → PriceB = 200 → YieldC = 0.25 → PriceC = 300 →
  CombinedAverageYield
    (TotalAnnualIncome
      (AnnualIncome YieldA PriceA)
      (AnnualIncome YieldB PriceB)
      (AnnualIncome YieldC PriceC))
    (TotalInvestment PriceA PriceB PriceC) = 0.1983 :=
by
  intro hYA hPA hYB hPB hYC hPC
  rw [hYA, hPA, hYB, hPB, hYC, hPC]
  sorry

end combined_average_yield_l408_408815


namespace find_a_range_f_when_a_is_1_f_is_increasing_l408_408502

-- Given the conditions
def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- Problem 1: Prove that a = 1 if f is an odd function
theorem find_a (hodd: ∀ x, f a x = -f a (-x)) : a = 1 := by
  sorry

-- Problem 2: Prove that the range of f(x) is (-1, 1) when a = 1
theorem range_f_when_a_is_1 : ∀ y, y = f 1 x → -1 < y ∧ y < 1 := by
  sorry

-- Problem 3: Prove f(x) is increasing for all x, when a = 1
theorem f_is_increasing : ∀ x1 x2, x1 < x2 → f 1 x1 < f 1 x2 := by
  sorry

end find_a_range_f_when_a_is_1_f_is_increasing_l408_408502


namespace sum_of_inscribed_radii_eq_r_l408_408659

theorem sum_of_inscribed_radii_eq_r (a b c r : ℝ) (T : ℝ) 
    (hT : T = (a + b + c) / 2 * r) : 
  let r_a := (T - a * r) / T * r,
      r_b := (T - b * r) / T * r,
      r_c := (T - c * r) / T * r in
  r_a + r_b + r_c = r :=
by
  have h_steps_7_8:
    r_a + r_b + r_c = r * (1 / T) * (3 * T - 2 * Σ'ᵥ (t ∈ {a,b,c}, t)) := sorry

  have h_step_9: r * (1 / T) * (3 * T - 2 * Σ'ᵥ (t ∈ {a,b,c}, t)) = r :=
    by sorry

  exact h_step_9

end sum_of_inscribed_radii_eq_r_l408_408659


namespace sec_330_eq_l408_408444

-- Definitions of conditions
def sec (θ : ℝ) : ℝ := 1 / real.cos θ
noncomputable def angle_330 := 330 * real.pi / 180
noncomputable def angle_30 := 30 * real.pi / 180
noncomputable def cos_330 := real.cos (2 * real.pi - angle_30)
noncomputable def cos_30 := real.cos angle_30

-- Condition equality
axiom angle_relation : cos_330 = cos_30
axiom cos_30_value : cos_30 = real.sqrt 3 / 2

-- Theorem to be proved
theorem sec_330_eq : sec angle_330 = 2 * real.sqrt 3 / 3 :=
by
  sorry

end sec_330_eq_l408_408444


namespace bus_students_remain_l408_408163

theorem bus_students_remain (init_students : ℕ) 
  (third_got_off : ℕ → ℕ) 
  (first_stop_second_third_fourth : ℕ ≠ 0 ∧ init_students = 64 ∧ 
   ∀ s, third_got_off s = (s * 2) / 3 ∧ 
   third_got_off (init_students * 2 / 3) = ((init_students * 2 / 3) * 2) / 3 ∧ 
   third_got_off ((init_students * 2 / 3) * 2 / 3) = (((init_students * 2 / 3) * 2 / 3) * 2) / 3 ∧ 
   third_got_off ((((init_students * 2 / 3) * 2 / 3) * 2 / 3) * 2) / 3) : 
  (((((init_students * 2) / 3) * 2) / 3) * 2 / 3 * 2) / 3 * 2 = 1024 / 81 :=
by sorry

end bus_students_remain_l408_408163


namespace count_squares_with_side_length_at_least_9_l408_408663

-- Define the set H
def H : set (ℤ × ℤ) := { p | ∃ x y, p = (x, y) ∧ -8 ≤ x ∧ x ≤ 8 ∧ -8 ≤ y ∧ y ≤ 8 }

-- Define the predicate for a square with a given side length having its vertices in H
def is_square_in_H (side : ℕ) : Prop :=
  ∀ x y : ℤ, 
    -8 ≤ x ∧ x ≤ 8 - side + 1 ∧
    -8 ≤ y ∧ y ≤ 8 - side + 1 →
    ((x, y) ∈ H ∧
     (x + side, y) ∈ H ∧
     (x, y + side) ∈ H ∧
     (x + side, y + side) ∈ H)

-- Number of squares with side length 9 whose vertices are in H
def number_of_squares_with_side_length_9 : ℕ :=
  if h : is_square_in_H 9 then 81 else 0

theorem count_squares_with_side_length_at_least_9 :
  number_of_squares_with_side_length_9 = 81 :=
by {
  sorry
}

end count_squares_with_side_length_at_least_9_l408_408663


namespace next_tutoring_day_lcm_l408_408825

theorem next_tutoring_day_lcm :
  Nat.lcm (Nat.lcm (Nat.lcm 4 5) 6) 8 = 120 := by
  sorry

end next_tutoring_day_lcm_l408_408825


namespace ratio_miles_traveled_l408_408753

theorem ratio_miles_traveled (D_AB D_BC : ℝ) (r : ℝ) 
  (h1 : D_AB = r * D_BC) 
  (h2 : (D_AB + D_BC) / ((D_AB / 25) + (D_BC / 30)) = 26.47) : 
  r ≈ 2.256 := 
by 
  sorry

end ratio_miles_traveled_l408_408753


namespace total_amount_spent_l408_408570

theorem total_amount_spent (T : ℝ) (h1 : 5000 + 200 + 0.30 * T = T) : 
  T = 7428.57 :=
by
  sorry

end total_amount_spent_l408_408570


namespace cone_lateral_surface_area_l408_408941

theorem cone_lateral_surface_area (a : ℝ) (π : ℝ) (sqrt_3 : ℝ) 
  (h₁ : 0 < a)
  (h_area : (1 / 2) * a^2 * (sqrt_3 / 2) = sqrt_3) :
  π * 1 * 2 = 2 * π :=
by
  sorry

end cone_lateral_surface_area_l408_408941


namespace tangent_line_at_point_l408_408865

noncomputable def f (x : ℝ) := x^3
def point : ℝ × ℝ := (1, 1)
def tangent_lines : ℝ → ℝ → ℝ := λ m b, ∀ (x : ℝ) (y : ℝ), y = m * x + b → 
  (∃ (x₀ : ℝ), y = f x₀ ∧ f 1 = 1)

theorem tangent_line_at_point (m b : ℝ) :
  tangent_lines 3 (-2) ∨ tangent_lines (3/4) (-1/4) :=
by
  sorry

end tangent_line_at_point_l408_408865


namespace sec_330_eq_2_sqrt_3_over_3_l408_408439

theorem sec_330_eq_2_sqrt_3_over_3 :
  (sec (330 * real.pi / 180) = 2 * real.sqrt 3 / 3) :=
by
  have h1: cos (330 * real.pi / 180) = cos (30 * real.pi / 180),
  {
    have h2: cos (330 * real.pi / 180) = cos ((360 - 30) * real.pi / 180), by norm_num,
    have h3: cos ((360 - 30) * real.pi / 180) = cos (-30 * real.pi / 180), by norm_num,
    have h4: cos (-30 * real.pi / 180) = cos (30 * real.pi / 180), by norm_num,
    exact eq.trans (eq.trans h2 h3) h4,
  },
  rw [sec, h1],
  have h_cos_30: cos (30 * real.pi / 180) = real.sqrt 3 / 2, by norm_num,
  rw h_cos_30,
  norm_num

end sec_330_eq_2_sqrt_3_over_3_l408_408439


namespace proof_problem_l408_408489

noncomputable def f (a b : ℝ) : ℝ := (1 / a) + (2 / b)

theorem proof_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : f a b = 2) : 
  ∀ (c : ℝ), (a + b ≥ c) → c ≤ (3 / 2 + Real.sqrt 2) :=
sory

end proof_problem_l408_408489


namespace remaining_count_after_removals_l408_408293

open Finset

def S : Finset ℕ := (range 51).filter (λ n => n ≥ 1)

def multiples (n : ℕ) (s : Finset ℕ) : Finset ℕ :=
  s.filter (λ x => n ∣ x)

def remaining (s : Finset ℕ) : Finset ℕ :=
  s \ (multiples 2 s) \ (multiples 3 s)

theorem remaining_count_after_removals : (remaining S).card = 17 := by
  sorry

end remaining_count_after_removals_l408_408293


namespace relationship_abc_l408_408087

theorem relationship_abc (a b c : ℕ) (ha : a = 2^555) (hb : b = 3^444) (hc : c = 6^222) : a < c ∧ c < b := by
  sorry

end relationship_abc_l408_408087


namespace ellipse_intersects_line_l408_408943

theorem ellipse_intersects_line 
  (m n : ℝ) 
  (h_intersection : ∃ A B : ℝ × ℝ, 
    (m * A.1^2 + n * A.2^2 = 1) ∧ (m * B.1^2 + n * B.2^2 = 1) ∧ (A.1 + A.2 = 1) ∧ (B.1 + B.2 = 1))
  (h_slope : ∃ M : ℝ × ℝ, (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ (M.2 = (sqrt 2 / 2) * M.1))) : 
  n / m = sqrt 2 :=
sorry

end ellipse_intersects_line_l408_408943


namespace part1_part2_l408_408506

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 2) - abs (x + 1)

theorem part1 (x : ℝ) : f x ≤ 3 ↔ -2/3 ≤ x ∧ x ≤ 6 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x, f x ≤ abs (x + 1) + a^2) ↔ a ≤ -2 ∨ 2 ≤ a :=
by
  sorry

end part1_part2_l408_408506


namespace yule_log_surface_area_increase_l408_408749

noncomputable def yuleLogIncreaseSurfaceArea : ℝ := 
  let h := 10
  let d := 5
  let r := d / 2
  let n := 9
  let initialSurfaceArea := 2 * Real.pi * r * h + 2 * Real.pi * r^2
  let sliceHeight := h / n
  let sliceSurfaceArea := 2 * Real.pi * r * sliceHeight + 2 * Real.pi * r^2
  let totalSlicesSurfaceArea := n * sliceSurfaceArea
  let increaseSurfaceArea := totalSlicesSurfaceArea - initialSurfaceArea
  increaseSurfaceArea

theorem yule_log_surface_area_increase : yuleLogIncreaseSurfaceArea = 100 * Real.pi := by
  sorry

end yule_log_surface_area_increase_l408_408749


namespace units_digit_sum_of_squares_odd_integers_l408_408726

theorem units_digit_sum_of_squares_odd_integers :
  let units_digit (n : ℤ) : ℤ := n % 10
  let sum_units_digits (n : ℕ) : ℤ :=
    (List.range (2 * n + 1)).filter (λ x => x % 2 = 1).map (λ k => units_digit (k * k)).sum in
  units_digit (sum_units_digits 2023) = 5 :=
begin
  sorry
end

end units_digit_sum_of_squares_odd_integers_l408_408726


namespace solution_l408_408276

def f (ω x : ℝ) : ℝ := cos (ω * x + (π / 6))
axiom ω_pos : ω > 0

theorem solution (ω : ℝ) (hω : ω > 0) :
  (f ω (π / 2) = - (√3 / 2)) ∧
  (∀ x, f ω (x + π) = f ω x) ∧
  (∀ x, 0 < x ∧ x < (π / (3 * ω)) → f ω x < f ω (x + δ) ∧ 0 < δ) :=
begin
  sorry
end

end solution_l408_408276


namespace largest_number_in_ap_l408_408249

theorem largest_number_in_ap (a d : ℝ) :
  let s := [a, a + d, a + 2*d, a + 3*d, a + 4*d, a + 5*d, a + 6*d] in
  (∑ x in s, x^3 = 0) →
  (∑ x in s, x^2 = -224) →
  max a (a + 6*d) = 6 * Real.sqrt 2 := 
sorry

end largest_number_in_ap_l408_408249


namespace triangle_area_l408_408706

theorem triangle_area : 
  let l1 := λ x : ℝ, 3 * x - 6,
      l2 := λ x : ℝ, -4 * x + 24,
      y_axis_intersect := (0 : ℝ),
      x_intersect := 30 / 7,
      y_intersect := 48 / 7,
      base := 30,
      height := 30 / 7 in
  (1 / 2) * base * height = 450 / 7 :=
by
  sorry

end triangle_area_l408_408706


namespace trig_expression_tangent_l408_408467

theorem trig_expression_tangent (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / 
  (Real.cos (3 * π / 2 + α) + Real.cos (π - α)) = 1 :=
sorry

end trig_expression_tangent_l408_408467


namespace problem_statement_l408_408513

open Set

def M : Set ℝ := {x | x^2 - 2008 * x - 2009 > 0}
def N (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}

theorem problem_statement (a b : ℝ) :
  (M ∪ N a b = univ) →
  (M ∩ N a b = {x | 2009 < x ∧ x ≤ 2010}) →
  (a = 2009 ∧ b = 2010) :=
by
  sorry

end problem_statement_l408_408513


namespace integral_sqrt_1_minus_x_squared_l408_408830

theorem integral_sqrt_1_minus_x_squared :
  ∫ x in 0..1, Real.sqrt (1 - x^2) = (π / 4) := by
sorry

end integral_sqrt_1_minus_x_squared_l408_408830


namespace next_number_property_l408_408673

theorem next_number_property (a b c d : ℕ) (h : a = 1 ∧ b = 8):
  ((10 * a + b) * (10 * c + d)) = (n * n) → 
  (1818 < (1000 * a + 100 * b + 10 * c + d)) →
  (∃ abcd, (1000 * a + 100 * b + 10 * c + d) = 1832) :=
by
  sorry

end next_number_property_l408_408673


namespace proof_angle_CKM_ninety_degrees_l408_408995

variables (A B C P S D M K : Type*)
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables [AddCommGroup P] [AddCommGroup S] [AddCommGroup D]
variables [AddCommGroup M] [AddCommGroup K]
variables [CommRing (A → B)] [CommRing (B → C)] [CommRing (C → A)]
variables [CommRing (P → S)] [CommRing (S → D)] [CommRing (D → P)]
variables [CommRing (M → K)] [CommRing (K → M)]
variables [AffineSpace A] [AffineSpace B] [AffineSpace C]
variables [AffineSpace P] [AffineSpace S] [AffineSpace D] 
variables [AffineSpace M] [AffineSpace K]
variables (omega Omega : Triangle A B C)

noncomputable def midpoint (P Q : A) : A := sorry
noncomputable def altitude (A B C : A) : A := sorry
noncomputable def circumcircle (A B C : A) : Circle := sorry
noncomputable def tangent (C : Circle) (P : A) : AffineSpace := sorry
noncomputable def intersection (P Q : AffineSpace) : A := sorry
noncomputable def intersect_at_point (C D : Circle) : Prop := sorry
noncomputable def angle (P Q R : A) : ℝ := sorry

def problem_conditions : Prop :=
  M = midpoint A C ∧
  Omega = circumcircle A B C ∧
  (∃ (tangent_to_A : AffineSpace) (tangent_to_C: AffineSpace), 
    tangent Omega A = tangent_to_A ∧ 
    tangent Omega C = tangent_to_C ∧ 
    intersection tangent_to_A tangent_to_C = P) ∧
  intersection (line_through B P) (line_through A C) = S ∧
  D = altitude A B P ∧
  omega = circumcircle C S D ∧
  intersect_at_point Omega omega ∧ 
  K ≠ C

theorem proof_angle_CKM_ninety_degrees (h : problem_conditions A B C P S D M K omega Omega) : 
  angle C K M = 90 :=
sorry

end proof_angle_CKM_ninety_degrees_l408_408995


namespace three_fifths_difference_products_l408_408802

theorem three_fifths_difference_products :
  (3 / 5) * ((7 * 9) - (4 * 3)) = 153 / 5 :=
by
  sorry

end three_fifths_difference_products_l408_408802


namespace magnitude_of_combined_vector_l408_408145

-- Define the vectors
def a : ℝ × ℝ := (2, 3)
def b (m : ℝ) : ℝ × ℝ := (m, -6)

-- Perpendicular condition
def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Scalar multiplication
def scalar_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)

-- Vector addition
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)

-- Magnitude calculation
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Main theorem statement
theorem magnitude_of_combined_vector (m : ℝ) (h : is_perpendicular a (b m)) : magnitude (vector_add (scalar_mul 2 a) (b m)) = 13 := sorry

end magnitude_of_combined_vector_l408_408145


namespace original_average_of_sequence_l408_408771

theorem original_average_of_sequence : 
  ∃ (x : ℝ), (x + 4.5 = 20) ∧ 
  ((x - 9) + (x - 7) + (x - 5) + (x - 3) + (x - 1) + (x + 1) + (x + 3) + (x + 5) + (x + 7) + (x + 9)) / 10 = 15.5 :=
begin
  use 15.5,
  split,
  { refl },
  { sorry }
end

end original_average_of_sequence_l408_408771


namespace tiles_needed_to_cover_floor_l408_408759

theorem tiles_needed_to_cover_floor :
  ∀ (floor_length floor_width tile_length_inch tile_width_inch : ℝ)
    (inch_to_foot : ℝ),
    floor_length = 10 →
    floor_width = 15 →
    tile_length_inch = 5 →
    tile_width_inch = 8 →
    inch_to_foot = 12 →
    let tile_length := tile_length_inch / inch_to_foot in
    let tile_width := tile_width_inch / inch_to_foot in
    let tile_area := tile_length * tile_width in
    let floor_area := floor_length * floor_width in
    floor_area / tile_area = 540 :=
begin
  intros floor_length floor_width tile_length_inch tile_width_inch inch_to_foot,
  intros h1 h2 h3 h4 h5,
  let tile_length := tile_length_inch / inch_to_foot,
  let tile_width := tile_width_inch / inch_to_foot,
  let tile_area := tile_length * tile_width,
  let floor_area := floor_length * floor_width,
  have : floor_length * floor_width / (tile_length * tile_width) = 540,
  sorry
end

end tiles_needed_to_cover_floor_l408_408759


namespace sec_330_eq_2sqrt3_div_3_l408_408447

theorem sec_330_eq_2sqrt3_div_3 : sec 330 = 2 * sqrt 3 / 3 := by
  sorry

end sec_330_eq_2sqrt3_div_3_l408_408447


namespace PA_PB_product_l408_408188
noncomputable def circle_eq (theta : ℝ) : ℝ × ℝ := (4 * Real.cos theta, 4 * Real.sin theta)

noncomputable def line_eq (t : ℝ) : ℝ × ℝ :=
  (1 + (Real.sqrt 3 / 2) * t, 2 + (1 / 2) * t)

theorem PA_PB_product (t1 t2 : ℝ) (pointA pointB : ℝ × ℝ) (P : ℝ × ℝ) :
  (circle_eq (Real.arccos (fst pointA / 4)) = pointA ∧ circle_eq (Real.arcsin (snd pointA / 4)) = pointA) ∧
  (circle_eq (Real.arccos (fst pointB / 4)) = pointB ∧ circle_eq (Real.arcsin (snd pointB / 4)) = pointB) ∧
  pointA = line_eq t1 ∧ pointB = line_eq t2 ∧ P = (1, 2) ∧ t1 * t2 = -11 →
  dist P pointA * dist P pointB = 11 :=
  sorry

end PA_PB_product_l408_408188


namespace chickens_after_9_years_l408_408609

-- Definitions from the conditions
def annual_increase : ℕ := 150
def current_chickens : ℕ := 550
def years : ℕ := 9

-- Lean statement for the proof
theorem chickens_after_9_years : current_chickens + annual_increase * years = 1900 :=
by
  sorry

end chickens_after_9_years_l408_408609


namespace find_m_and_r_l408_408522

-- Define the problem conditions
def vector_a (m : ℝ) : ℝ × ℝ × ℝ := (m, 5, -1)
def vector_b (r : ℝ) : ℝ × ℝ × ℝ := (3, 1, r)

-- Define parallel vectors condition
def are_parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2, k * b.3)

-- Define the main theorem
theorem find_m_and_r (m r : ℝ) :
  are_parallel (vector_a m) (vector_b r) → m = 15 ∧ r = -1/5 :=
by
  intro h_parallel
  sorry

end find_m_and_r_l408_408522


namespace circumcircles_touch_each_other_l408_408603

open EuclideanGeometry

variables {k1 k2 : Circle} {A P Q R S : Point}

axiom common_tangent1 : tangent k1 P ∧ tangent k2 Q
axiom common_tangent2 : tangent k1 R ∧ tangent k2 S
axiom intersection_A : A ∈ k1 ∧ A ∈ k2
axiom different_points: P ≠ Q ∧ R ≠ S ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S

theorem circumcircles_touch_each_other :
  (∃ γ1 γ2 : Circle, circumcircle (Triangle.mk P A Q) γ1 ∧ circumcircle (Triangle.mk R A S) γ2 ∧
    tangent γ1 A ∧ tangent γ2 A) ∧ (γ1 = γ2) :=
by
  sorry

end circumcircles_touch_each_other_l408_408603


namespace polynomial_value_l408_408893

noncomputable def f : ℚ → ℚ := sorry  -- Definition of the polynomial function

theorem polynomial_value :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2013 → f k = 2 / k) → polynomial.degree f = 2012 →
  (2014 * f 2014 = 4) :=
begin
  intros h_keyvals h_degree,
  -- Construct the polynomial 
  let g := (λ x : ℚ, x * f x - 2),
  -- Prove that g(k) = 0 for k = 1, 2, ..., 2013
  have zeros_g : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 2013 → g k = 0,
  {
    intros k hk_range,
    specialize h_keyvals k hk_range,
    unfold g,
    rw h_keyvals,
    linarith,
  },
  -- essential properties of polynomial and degrees (to be shown in proof)
  sorry
end

end polynomial_value_l408_408893


namespace max_remaining_chips_is_1009_squared_l408_408799

-- Define the problem conditions
def board_size : ℕ := 2018
def chip_colors := ["black", "white"]

-- Define the operations as described
def remove_black_chips (cols_with_white : Finset ℕ) (black_chip_positions : Finset (ℕ × ℕ)) : Finset (ℕ × ℕ) :=
  black_chip_positions.filter (λ ⟨r, c⟩, c ∉ cols_with_white)

def remove_white_chips (rows_with_black : Finset ℕ) (white_chip_positions : Finset (ℕ × ℕ)) : Finset (ℕ × ℕ) :=
  white_chip_positions.filter (λ ⟨r, c⟩, r ∉ rows_with_black)

-- Define a function to compute the remaining chips W and B after the operations
def remaining_chips (white_chip_positions black_chip_positions : Finset (ℕ × ℕ)) : ℕ :=
  let cols_with_white := white_chip_positions.image Prod.snd in
  let rows_with_black := black_chip_positions.image Prod.fst in
  let remaining_white := remove_white_chips rows_with_black white_chip_positions in
  let remaining_black := remove_black_chips cols_with_white black_chip_positions in
  min remaining_white.card remaining_black.card

theorem max_remaining_chips_is_1009_squared :
  ∀ (white_chip_positions black_chip_positions : Finset (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ white_chip_positions ∨ p ∈ black_chip_positions → p.1 < board_size ∧ p.2 < board_size) →
    remaining_chips white_chip_positions black_chip_positions ≤ 1009 * 1009 :=
sorry

end max_remaining_chips_is_1009_squared_l408_408799


namespace a3_value_l408_408891

theorem a3_value : 
  let a (n : ℕ) := (Finset.range (n^2 - n + 1)).sum (λ k, 1 / (n + k)) in 
  a 3 = 1/3 + 1/4 + 1/5 + 1/6 + 1/7 + 1/8 + 1/9 := 
by {
  sorry
}

end a3_value_l408_408891


namespace trigonometric_identity_l408_408235

theorem trigonometric_identity 
  (α : ℝ)
  (h_tan : Real.tan α = Real.sin α / Real.cos α)
  (h_cot : Real.cot α = Real.cos α / Real.sin α) :
  (4.12 * ((Real.sin α ^ 2 + Real.tan α ^ 2 + 1) * (Real.cos α ^ 2 - Real.cot α ^ 2 + 1)) / ((Real.cos α ^ 2 + Real.cot α ^ 2 + 1) * (Real.sin α ^ 2 + Real.tan α ^ 2 - 1)) = 1) :=
by
  -- Insert proof steps here
  sorry

end trigonometric_identity_l408_408235


namespace satellite_orbit_time_approx_l408_408003

noncomputable def earth_radius_km : ℝ := 6371
noncomputable def satellite_speed_kmph : ℝ := 7000

theorem satellite_orbit_time_approx :
  let circumference := 2 * Real.pi * earth_radius_km 
  let time := circumference / satellite_speed_kmph 
  5.6 < time ∧ time < 5.8 :=
by
  sorry

end satellite_orbit_time_approx_l408_408003


namespace triangle_side_length_l408_408984

theorem triangle_side_length (a b c x : ℕ) (A C : ℝ) (h1 : b = x) (h2 : a = x - 2) (h3 : c = x + 2)
  (h4 : C = 2 * A) (h5 : x + 2 = 10) : a = 8 :=
by
  sorry

end triangle_side_length_l408_408984


namespace monotone_intervals_triangle_area_l408_408518

noncomputable def a (x : ℝ) : ℝ × ℝ :=
  (2 * Real.sin x, Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 * Real.cos x, 2 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ :=
  (a x).fst * (b x).fst + (a x).snd * (b x).snd - 1

theorem monotone_intervals :
  ∀ k : ℤ, ∀ x : ℝ, k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 →
  0 ≤ Real.sqrt 3 * 2 * (Real.sin (2 * x + Real.pi / 6)) :=
sorry

variables (B : ℝ) (b c : ℝ) (hb : b = Real.sqrt 3) (hc : c = 2) (hfB : f B = 1)

theorem triangle_area :
  geometry.area_triangle b c (⊥_ RF B) = Real.sqrt 3 / 2 :=
sorry

end monotone_intervals_triangle_area_l408_408518


namespace sum_exponents_sqrt_l408_408322

theorem sum_exponents_sqrt (n : ℕ) (fact : n = 15) : 
    let primes := [2, 3, 5, 7, 11, 13]
    let exp (p : ℕ) := ∑ k in (list.range (n + 1)).tail, n / p^k
    let largest_square := prod (primes.map (λ p, p ^ (exp p).div 2 * 2))
    let sqrt_largest_square := prod (primes.map (λ p, p ^ (exp p).div 2))
    let sum_exponents := (primes.map (λ p, (exp p).div 2)).sum
    sum_exponents = 10 :=
by
  sorry

end sum_exponents_sqrt_l408_408322


namespace part1_part2_l408_408929

noncomputable def vec_m (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -1)
noncomputable def vec_n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 2), Real.cos (x / 2) ^ 2)
noncomputable def f (x : ℝ) : ℝ := (vec_m x).1 * (vec_n x).1 + (vec_m x).2 * (vec_n x).2 + 1

-- Part 1
theorem part1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) (hf : f x = 11 / 10) : 
  x = (Real.pi / 6) + Real.arcsin (3 / 5) :=
sorry

-- Part 2
theorem part2 {A B C a b c : ℝ} 
  (hABC : A + B + C = Real.pi) 
  (habc : 2 * b * Real.cos A ≤ 2 * c - Real.sqrt 3 * a) : 
  (0 < B ∧ B ≤ Real.pi / 6) → 
  ∃ y, (0 < y ∧ y ≤ 1 / 2 ∧ f B = y) :=
sorry

end part1_part2_l408_408929


namespace triangle_base_length_equals_5_l408_408554

theorem triangle_base_length_equals_5
  (perimeter_square : Real)
  (height_triangle : Real)
  (common_side : Real)
  (square_area_equals_triangle_area : Real → Real) :
  perimeter_square = 40 →
  height_triangle = 40 →
  common_side = 10 → -- Due to the side length from the perimeter calculation
  square_area_equals_triangle_area common_side = 100 →
  ∃ x : Real, (x * 20 = 100) ∧ (x = 5) :=
by
  intros h1 h2 h3 h4
  use 5
  constructor
  · rw [h1, h2, h3] at h4
    sorry
  · reflexivity

end triangle_base_length_equals_5_l408_408554


namespace find_triplets_l408_408449

theorem find_triplets (a b p : ℕ) (ha : 0 < a) (hb : 0 < b) (hp : Nat.Prime p) (h_eq : (a + b)^p = p^a + p^b) : (a = 1 ∧ b = 1 ∧ p = 2) :=
by
  sorry

end find_triplets_l408_408449


namespace person_A_speed_l408_408832

-- Let v be the speed of Person B
variable (v : ℕ)

-- Define the conditions
def distance_between_towns : ℕ := 45
def person_A_faster : ℕ := v + 1
def travel_time : ℕ := 5
def meeting_distance : ℕ := travel_time * v + travel_time * person_A_faster

-- The final statement to prove
theorem person_A_speed (h : meeting_distance = distance_between_towns) : person_A_faster = 5 := by
  sorry

end person_A_speed_l408_408832


namespace avg_temp_Brookdale_l408_408373

noncomputable def avg_temp (temps : List ℚ) : ℚ :=
  temps.sum / temps.length

theorem avg_temp_Brookdale : avg_temp [51, 67, 64, 61, 50, 65, 47] = 57.9 :=
by
  sorry

end avg_temp_Brookdale_l408_408373


namespace power_function_through_point_l408_408650

theorem power_function_through_point :
  ∃ (k α : ℝ), (f : ℝ → ℝ) (h₁ : f = λ x, k * x ^ α) (h₂ : f (1/3) = 9), k + α = -1 :=
sorry

end power_function_through_point_l408_408650


namespace evaluate_products_l408_408311

theorem evaluate_products : 
  (Real.cbrt 125) * (Real.root 256 4) * (Real.sqrt 16) = 80 := 
sorry

end evaluate_products_l408_408311


namespace cupcakes_packages_l408_408378

theorem cupcakes_packages :
  (baked : ℕ) → (eaten : ℕ) → (per_package : ℕ) →
  baked = 50 →
  eaten = 5 →
  per_package = 5 →
  (baked - eaten) / per_package = 9 :=
by
  intros baked eaten per_package h_baked h_eaten h_per_package
  rw [h_baked, h_eaten, h_per_package]
  exact Nat.div_eq_of_eq_mul 9 (by norm_num)
  sorry

end cupcakes_packages_l408_408378


namespace hyperbola_equation_l408_408160

-- Given Problem Constants and Conditions
constant a : ℝ := 8
constant b : ℝ := 4
constant hyperbola_foci : ℝ := 4 * Real.sqrt 3
constant hyperbola_asymptote_m : ℝ := Real.sqrt 3

-- Definition of ellipse
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Definition of hyperbola with provided foci and asymptote slope
def hyperbola (x y : ℝ) : Prop := x^2 / ((2 * hyperbola_foci / 2_Pow 2 2)^2 / 3) - y^2 / ((2 * hyperbola_foci / 2_Pow 2 2)^2 / 12) = 1

-- Prove that the hyperbola fits the given conditions
theorem hyperbola_equation :
  hyperbola (x : ℝ) (y : ℝ) ↔ (x^2 / 36 - y^2 / 12 = 1) :=
sorry

end hyperbola_equation_l408_408160


namespace parallel_lines_sufficient_not_necessary_condition_l408_408469

theorem parallel_lines_sufficient_not_necessary_condition {a : ℝ} :
  (a = 4) → (∀ x y : ℝ, (a * x + 8 * y - 3 = 0) ↔ (2 * x + a * y - a = 0)) :=
by sorry

end parallel_lines_sufficient_not_necessary_condition_l408_408469


namespace no_such_m_exists_l408_408219

def f (x m : ℝ) : ℝ := x - (1 / x) + 2 * m * Real.log x

def f_prime (x m : ℝ) : ℝ := (x^2 - 2 * m * x + 1) / x^2

def roots (m : ℝ) : ℝ × ℝ :=
  let Δ := 4 * (m^2 - 1)
  if Δ ≤ 0 then (m, m)
  else (m - Real.sqrt (m^2 - 1), m + Real.sqrt (m^2 - 1))

noncomputable def k (x1 x2 m : ℝ) : ℝ :=
  (f x2 m - f x1 m) / (x2 - x1)

theorem no_such_m_exists (m : ℝ) : ¬ ∃ m₀ : ℝ,
  let (x1, x2) := roots m₀ in
  x1 < x2 ∧ k x1 x2 m₀ = 2 - 2 * m₀ :=
begin
  -- Proof would go here
  sorry,
end

end no_such_m_exists_l408_408219


namespace bin_to_oct_l408_408821

theorem bin_to_oct (n : ℕ) (hn : n = 0b11010) : n = 0o32 := by
  sorry

end bin_to_oct_l408_408821


namespace transform_polynomial_l408_408212

theorem transform_polynomial (a b c : ℝ) (h : Polynomial.root (Polynomial.mk [1, -3, 4, -1]) a)
  (h1 : Polynomial.root (Polynomial.mk [1, -3, 4, -1]) b)
  (h2 : Polynomial.root (Polynomial.mk [1, -3, 4, -1]) c) :
  Polynomial.mk [1, -12, 49, -67].isRoot (a + 3) ∧
  Polynomial.mk [1, -12, 49, -67].isRoot (b + 3) ∧
  Polynomial.mk [1, -12, 49, -67].isRoot (c + 3) :=
by
  sorry

end transform_polynomial_l408_408212


namespace simplify_f_range_g_l408_408910

-- Define the initial function f(x) with given conditions
def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) : ℝ := 
  √3 * real.sin (ω * x + ϕ) + 2 * real.sin ((ω * x + ϕ) / 2) ^ 2 - 1

-- First theorem: Simplification of f(x) given the period and f(0) = 0
theorem simplify_f (ω > 0) (0 < ϕ < π) :
  f ω ϕ x = 2 * real.sin (2 * x) := by
  sorry

-- Define the transformed function g(x)
def g (x : ℝ) : ℝ := 2 * real.sin (4 * x - π / 3)

-- Second theorem: Range of g(x) for x in [-π/12, π/6]
theorem range_g (x : ℝ) (h : x ∈ Icc (-π / 12) (π / 6)) :
  g x ∈ Icc (-2 : ℝ) real.sqrt 3 := by
  sorry

end simplify_f_range_g_l408_408910


namespace sin_66_eq_1_minus_2a_squared_l408_408862

theorem sin_66_eq_1_minus_2a_squared (a : Real) (h : Real.sin (Real.toRadians 12) = a) : 
  Real.sin (Real.toRadians 66) = 1 - 2 * a ^ 2 :=
sorry

end sin_66_eq_1_minus_2a_squared_l408_408862


namespace units_digit_first_2023_odd_squares_l408_408727

noncomputable def units_digit_of_squares_sum (n : ℕ) : ℕ :=
  let odd_squares_units := λ k, (2 * k + 1) ^ 2 % 10 in
  (List.sum (List.map odd_squares_units (List.range n))) % 10

theorem units_digit_first_2023_odd_squares :
  units_digit_of_squares_sum 2023 = 5 :=
sorry

end units_digit_first_2023_odd_squares_l408_408727


namespace number_of_valid_four_digit_numbers_l408_408526

-- Define the conditions
def is_valid_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9
def is_valid_product (a b : ℕ) : Prop := (a * b) > 10
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999
def first_digit_greater_than_4999 (d : ℕ) : Prop := 5 ≤ d ∧ d ≤ 9

-- Define the main statement
theorem number_of_valid_four_digit_numbers : 
  ∃ (count : ℕ), count = 3050 ∧ 
  ∀ (n : ℕ), is_four_digit_number(n) → 
             first_digit_greater_than_4999(n / 1000) →
             is_valid_product((n / 10) % 10) ((n / 100) % 10) → 
             n ∈ { m : ℕ | is_valid_digit(m % 10) } :=
sorry

end number_of_valid_four_digit_numbers_l408_408526


namespace arithmetic_S1_S2_S3_l408_408550

variable {a : ℕ → ℤ}
variable {n : ℕ}
variable {d : ℤ}
variable (arithmetic_seq : ∀ k : ℕ, a (k + 1) = a k + d)
variable (S1 S2 S3 : ℤ)
variable (hS1 : S1 = ∑ i in finset.range n, a (i + 1))
variable (hS2 : S2 = ∑ i in finset.range n, a (n + i + 1))
variable (hS3 : S3 = ∑ i in finset.range n, a (2 * n + i + 1))

theorem arithmetic_S1_S2_S3 :
  S2 = (S1 + S3) / 2 :=
sorry

end arithmetic_S1_S2_S3_l408_408550


namespace exists_n_in_range_multiple_of_11_l408_408402

def is_multiple_of_11 (n : ℕ) : Prop :=
  (3 * n^5 + 4 * n^4 + 5 * n^3 + 7 * n^2 + 6 * n + 2) % 11 = 0

theorem exists_n_in_range_multiple_of_11 : ∃ n : ℕ, (2 ≤ n ∧ n ≤ 101) ∧ is_multiple_of_11 n :=
sorry

end exists_n_in_range_multiple_of_11_l408_408402


namespace chord_length_of_intersection_is_sqrt3_l408_408134

def limacon_curve (ρ θ : ℝ) : Prop := 2 * ρ * cos θ = 1
def circle_curve (ρ θ : ℝ) : Prop := ρ = 2 * cos θ

def cartesian_l (x y : ℝ) : Prop := 2 * x = 1
def cartesian_c (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

theorem chord_length_of_intersection_is_sqrt3 :
  ∃ x y : ℝ, cartesian_l x y ∧ cartesian_c x y ∧
  (sqrt((x - x)^2 + ((y - -y)^2)) = sqrt 3) :=
begin
  sorry
end

end chord_length_of_intersection_is_sqrt3_l408_408134


namespace tan_sum_inverse_eq_neg8_l408_408889

theorem tan_sum_inverse_eq_neg8 
  (α : ℝ)
  (h : (cos (2 * α) / (sqrt 2 * sin (α + π / 4))) = sqrt 5 / 2) : 
  tan α + (1 / tan α) = -8 :=
by
  sorry

end tan_sum_inverse_eq_neg8_l408_408889


namespace david_trip_distance_l408_408782

theorem david_trip_distance (t : ℝ) (d : ℝ) : 
  (40 * (t + 1) = d) →
  (d - 40 = 60 * (t - 0.75)) →
  d = 130 := 
by
  intro h1 h2
  sorry

end david_trip_distance_l408_408782


namespace probability_greater_than_mean_l408_408987

noncomputable def size_distribution : MeasureTheory.MeasurableSpace ℝ := 
  MeasureTheory.Measure.Normal 22.5 0.1  

theorem probability_greater_than_mean :
  MeasureTheory.Measure.prob (size_distribution) {x | x > 22.5} = 0.5 :=
sorry

end probability_greater_than_mean_l408_408987


namespace xy_identity_l408_408932

theorem xy_identity (x y : ℝ) (h1 : (x + y)^2 = 16) (h2 : x * y = 6) : x^2 + y^2 = 4 := 
by 
  sorry

end xy_identity_l408_408932


namespace mean_temperature_l408_408286

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem mean_temperature (temps : List ℝ) (length_temps_10 : temps.length = 10)
    (temps_vals : temps = [78, 80, 82, 85, 88, 90, 92, 95, 97, 95]) : 
    mean temps = 88.2 := by
  sorry

end mean_temperature_l408_408286


namespace geometric_series_sum_l408_408007

theorem geometric_series_sum :
  let a := 1
  let r := (1 / 4 : ℚ)
  (a / (1 - r)) = 4 / 3 :=
by
  sorry

end geometric_series_sum_l408_408007


namespace sum_squared_distances_eq_l408_408768

variables {R : ℝ} {n : ℕ} {θ : ℝ}

def regular_ngon_vertices (R : ℝ) (n : ℕ) : ℕ → ℝ × ℝ :=
  λ k, (R * real.cos (2 * k * real.pi / n), R * real.sin (2 * k * real.pi / n))

def point_on_circle (R : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (R * real.cos θ, R * real.sin θ)

noncomputable def squared_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem sum_squared_distances_eq {R : ℝ} {n : ℕ} {θ : ℝ} :
  let S := point_on_circle R θ in
  let A := regular_ngon_vertices R n in
  ∑ k in finset.range n, squared_distance S (A k) = 2 * n * R^2 :=
by sorry

end sum_squared_distances_eq_l408_408768


namespace fourth_term_expansion_l408_408263

noncomputable def sum_of_coefficients (x : ℝ) (n : ℕ) : ℝ :=
  (2 * real.cbrt(x) - 1)^n

def sum_of_binomial_coefficients (n : ℕ) : ℝ :=
  2^n

def geometric_sequence_condition (M N : ℝ) : Prop :=
  M * N = 64

theorem fourth_term_expansion {x : ℝ} {n : ℕ} (M N : ℝ)
  (h1 : sum_of_coefficients 1 n = 1)
  (h2 : sum_of_binomial_coefficients n = N)
  (h3 : geometric_sequence_condition M N)
  (h4 : x = 1) : coeff((2 * real.cbrt(x) - 1)^n) 3 = -160 * x :=
sorry

end fourth_term_expansion_l408_408263


namespace evaluate_magnitude_product_l408_408417

-- Definitions of complex numbers
def z1 := Complex.mk 7 (-4)
def z2 := Complex.mk 3 11

-- The magnitude of z1
def magnitude_z1 := Complex.abs z1

-- The magnitude of z2
def magnitude_z2 := Complex.abs z2

-- Lean 4 statement expressing the problem and its final answer
theorem evaluate_magnitude_product : Complex.abs (z1 * z2) = Real.sqrt 8450 := by
  sorry

end evaluate_magnitude_product_l408_408417


namespace sum_odd_terms_a_18_l408_408879

noncomputable def a : ℕ → ℤ
| 0     := -7
| 1     := 5
| (n+2) := a n + 2

theorem sum_odd_terms_a_18 : 
  (∑ i in finset.range (9), a (2 * i)) = 114 := by
sorry

end sum_odd_terms_a_18_l408_408879


namespace probability_three_friends_meet_l408_408698

open Real

theorem probability_three_friends_meet :
  let X (x : ℝ) := 0 ≤ x ∧ x ≤ 7
  let Y (y : ℝ) := 0 ≤ y ∧ y ≤ 7
  let Z (z : ℝ) := 0 ≤ z ∧ z ≤ 7
  ∃ x y z : ℝ, X x ∧ Y y ∧ Z z ∧ |x - y| ≤ 1 ∧ |y - z| ≤ 1 ∧ |z - x| ≤ 1 →
  (finset.card (finset.filter (λ (xyz : ℝ × ℝ × ℝ), (|xyz.1 - xyz.2| ≤ 1) ∧ (|xyz.2 - xyz.3| ≤ 1) ∧ (|xyz.3 - xyz.1| ≤ 1)) (finset.product (finset.product finset.Icc (0:ℝ) (7:ℝ) finset.Icc (0:ℝ) (7:ℝ)) finset.Icc (0:ℝ) (7:ℝ)))) / (7 * 7 * 7) = (19 / 343) :=
sorry

end probability_three_friends_meet_l408_408698


namespace B_3_1_eq_13_l408_408400

def B : ℕ → ℕ → ℕ
| 0, n      := n + 1
| (m+1), 0 := B m 1
| (m+1), (n+1) := B m (B (m+1) n)

theorem B_3_1_eq_13 : B 3 1 = 13 := 
by 
sorry

end B_3_1_eq_13_l408_408400


namespace modulus_product_l408_408428

open Complex

theorem modulus_product :
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  (Complex.abs (z1 * z2)) = Real.sqrt 8450 :=
by
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  sorry

end modulus_product_l408_408428


namespace next_valid_number_after_1818_l408_408681

open Nat

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem next_valid_number_after_1818 : 
  ∀ (a b : ℕ), 10 * a + b > 18 → 18 * (10 * a + b) = 1832 → isPerfectSquare (18 * (10 * a + b)) → 
  10 * 10 * 8 + 10 * a + b = 1832 :=
by {
  intros,
  sorry
}

end next_valid_number_after_1818_l408_408681


namespace rectangle_if_and_only_if_cyclic_l408_408350

theorem rectangle_if_and_only_if_cyclic
    (A B C D A1 B1 C1 D1 E F G H : Point)
    (convex_quadrilateral : ConvexQuadrilateral A B C D)
    (incircle_touch : IncircleTouches A B C D A1 B1 C1 D1)
    (E_midpoint : E = midpoint A1 B1)
    (F_midpoint : F = midpoint B1 C1)
    (G_midpoint : G = midpoint C1 D1)
    (H_midpoint : H = midpoint D1 A1) :
    IsRectangle E F G H ↔ Concyclic A B C D := sorry

end rectangle_if_and_only_if_cyclic_l408_408350


namespace x_plus_y_l408_408058

noncomputable def sequence_constant : ℝ := 1 / 4

def x : ℝ := 256 * sequence_constant

def y : ℝ := x * sequence_constant

theorem x_plus_y : x + y = 80 := 
by
  -- Placeholder for the proof
  sorry

end x_plus_y_l408_408058


namespace sequence_value_l408_408040

theorem sequence_value (r x y : ℝ) 
  (h1 : r = 1/4) 
  (h2 : x = 256 * r)
  (h3 : y = x * r) : 
  x + y = 80 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end sequence_value_l408_408040


namespace functional_eq_zero_l408_408448

theorem functional_eq_zero (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = x * f x + y * f y) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end functional_eq_zero_l408_408448


namespace average_divisible_by_4_l408_408327

theorem average_divisible_by_4 (l : List ℕ) (H : l = [8, 12, 16, 20, 24, 28]) : 
  (l.sum) / l.length = 18 :=
by
  have H_sum : l.sum = 108 := by sorry
  have H_length : l.length = 6 := by sorry
  rw [H_sum, H_length]
  norm_num

end average_divisible_by_4_l408_408327


namespace jill_show_duration_l408_408565

theorem jill_show_duration :
  let first_show_duration := 30
  let second_show_duration := 4 * first_show_duration
  first_show_duration + second_show_duration = 150 :=
by
  let first_show_duration := 30
  let second_show_duration := 4 * first_show_duration
  have h1 : second_show_duration = 120 := by rfl
  have h2 : first_show_duration + second_show_duration = 150 := by rfl
  show first_show_duration + second_show_duration = 150 from h2

end jill_show_duration_l408_408565


namespace sequence_sum_l408_408034

theorem sequence_sum :
  ∃ (r : ℝ) (x y : ℝ),
    (r = 1 / 4) ∧
    (x = 256 * r) ∧
    (y = x * r) ∧
    (x + y = 80) :=
by
  use 1 / 4, 64, 16
  split
  . exact rfl
  split
  . exact rfl
  split
  . exact rfl
  . norm_num


end sequence_sum_l408_408034


namespace sequence_x_y_sum_l408_408046

theorem sequence_x_y_sum (r : ℝ) (x y : ℝ)
  (h₁ : r = 1 / 4)
  (h₂ : x = 256 * r)
  (h₃ : y = x * r) :
  x + y = 80 :=
by
  sorry

end sequence_x_y_sum_l408_408046


namespace chickens_after_years_l408_408611

theorem chickens_after_years : 
  ∀ (initial_chickens annual_increase years : ℕ),
  initial_chickens = 550 →
  annual_increase = 150 →
  years = 9 →
  initial_chickens + (annual_increase * years) = 1900 :=
by
  intros initial_chickens annual_increase years h1 h2 h3
  rw [h1, h2, h3]
  rfl

end chickens_after_years_l408_408611


namespace minimum_value_l408_408599

noncomputable theory

open_locale big_operators

theorem minimum_value (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 8) : 
  ∃ m : ℝ, m = 64 ∧ (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ m :=
by
  sorry

end minimum_value_l408_408599


namespace bridge_length_l408_408332

theorem bridge_length :
  ∀ (length_train : ℝ) (speed_kmph : ℝ) (time_seconds : ℝ), 
  length_train = 100 ∧ speed_kmph = 45 ∧ time_seconds = 30 →
  let speed_mps := speed_kmph * (1000 / 3600) in
  let total_distance := speed_mps * time_seconds in
  let bridge_length := total_distance - length_train in
  bridge_length = 275 :=
by
  intros length_train speed_kmph time_seconds h,
  rcases h with ⟨h_len_train, h_speed_kmph, h_time_seconds⟩,
  rw [h_len_train, h_speed_kmph, h_time_seconds],
  let speed_mps := 45 * (1000 / 3600),
  let total_distance := speed_mps * 30,
  let bridge_length := total_distance - 100,
  linarith

end bridge_length_l408_408332


namespace maximum_area_l408_408575

-- Define points A, B, and C(p, q)
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 2, y := 1 }
def B : Point := { x := 5, y := 3 }

-- Define the parabola equation y = -x^2 + 7x - 10, where C is on this parabola with 2 ≤ p ≤ 5
def parabola (x : ℝ) : ℝ := -x^2 + 7 * x - 10

-- Define point C with coordinate constraints 2 ≤ p ≤ 5
def C (p : ℝ) (hp : 2 ≤ p ∧ p ≤ 5) : Point := { x := p, y := parabola p }

-- Define the function to calculate the area of triangle ABC using the Shoelace theorem
def shoelace_area (A B C : Point) : ℝ :=
  1/2 * | A.x * B.y + B.x * C.y + C.x * A.y - A.y * B.x - B.y * C.x - C.y * A.x |

-- Define the maximum area of the triangle ABC
def max_area_of_triangle : ℝ := 13 / 8

-- The theorem to prove the maximum area
theorem maximum_area (p : ℝ) (hp : 2 ≤ p ∧ p ≤ 5) : 
  ∃ p, 2 ≤ p ∧ p ≤ 5 ∧ (shoelace_area A B (C p hp) = max_area_of_triangle) :=
sorry

end maximum_area_l408_408575


namespace not_divisor_60_l408_408262

variable (k : ℤ)
def n : ℤ := k * (k + 1) * (k + 2)

theorem not_divisor_60 
  (h₁ : ∃ k, n = k * (k + 1) * (k + 2) ∧ 5 ∣ n) : ¬(60 ∣ n) := 
sorry

end not_divisor_60_l408_408262


namespace compare_powers_l408_408089

theorem compare_powers (a b c : ℝ) (h1 : a = 2^555) (h2 : b = 3^444) (h3 : c = 6^222) : a < c ∧ c < b :=
by
  sorry

end compare_powers_l408_408089


namespace sum_odd_is_13_over_27_l408_408732

-- Define the probability for rolling an odd and an even number
def prob_odd := 1 / 3
def prob_even := 2 / 3

-- Define the probability that the sum of three die rolls is odd
def prob_sum_odd : ℚ :=
  3 * prob_odd * prob_even^2 + prob_odd^3

-- Statement asserting the goal to be proved
theorem sum_odd_is_13_over_27 :
  prob_sum_odd = 13 / 27 :=
by
  sorry

end sum_odd_is_13_over_27_l408_408732


namespace capital_of_z_l408_408334

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

end capital_of_z_l408_408334


namespace valid_expressions_l408_408470

variable {α : Type*} [LinearOrderedField α] {a b : α}
-- Variables for sequences
variables {n : ℕ} (x : ℕ → α) (y : ℕ → α)

-- Conditions
-- Condition: a and b are distinct positive numbers
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom distinct_a_b : a ≠ b

-- Condition: n is a natural number greater than 1
axiom n_ge_2 : n ≥ 2

-- Arithmetic sequence condition between a and b involving xi
axiom arith_seq_condition (i : ℕ) (h : 1 ≤ i ∧ i ≤ n) : x (i - 1) = a + i * ((b - a) / (n + 1))

-- Geometric sequence condition between a and b involving yi
axiom geom_seq_condition (i : ℕ) (h : 1 ≤ i ∧ i ≤ n) : y (i - 1) = a * (b / a) ^ i

theorem valid_expressions :
  (∑ i in finset.range n, x i = n * (a + b) / 2) ∧
  (1 / n * ∑ i in finset.range n, x i = (a + b) / 2 ∧ (a + b) / 2 > sqrt (a * b) + (sqrt a - sqrt b)^2 / 4) :=
sorry

end valid_expressions_l408_408470


namespace price_per_pot_l408_408806

-- Definitions based on conditions
def total_pots : ℕ := 80
def proportion_not_cracked : ℚ := 3 / 5
def total_revenue : ℚ := 1920

-- The Lean statement to prove she sold each clay pot for $40
theorem price_per_pot : (total_revenue / (total_pots * proportion_not_cracked)) = 40 := 
by sorry

end price_per_pot_l408_408806


namespace find_third_number_l408_408534

theorem find_third_number (x : ℝ) (third_number : ℝ) : 
  0.6 / 0.96 = third_number / 8 → x = 0.96 → third_number = 5 :=
by
  intro h1 h2
  sorry

end find_third_number_l408_408534


namespace intersection_A_B_l408_408511

open Set

def set_A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def set_B : Set ℤ := {x | 0 < x ∧ x < 5}

theorem intersection_A_B : set_A ∩ set_B = {1, 3} := 
by 
  sorry

end intersection_A_B_l408_408511


namespace sequence_sum_l408_408053

noncomputable def r : ℝ := 1 / 4
def t₀ : ℝ := 4096
def t₁ : ℝ := t₀ * r
def t₂ : ℝ := t₁ * r
def x : ℝ := t₂ * r
def y : ℝ := x * r

theorem sequence_sum : x + y = 80 := by
  -- The proof will go here
  sorry

end sequence_sum_l408_408053


namespace jeans_cost_l408_408179

-- Definitions based on conditions
def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def total_cost : ℕ := 51
def n_shirts : ℕ := 3
def n_hats : ℕ := 4
def n_jeans : ℕ := 2

-- The goal is to prove that the cost of one pair of jeans (J) is 10
theorem jeans_cost (J : ℕ) (h : n_shirts * shirt_cost + n_jeans * J + n_hats * hat_cost = total_cost) : J = 10 :=
  sorry

end jeans_cost_l408_408179


namespace sum_of_x_satisfies_sqrt_eq_9_l408_408321

open Real

theorem sum_of_x_satisfies_sqrt_eq_9 : 
  (∀ x : ℝ, abs (x - 2) = 9 → x = 11 ∨ x = -7) →
  (∑ x in {x : ℝ | abs (x - 2) = 9}, x) = 4 :=
by
  intros h
  have hx : {x : ℝ | abs (x - 2) = 9} = {11, -7} := by
    ext x
    simp [Set.mem_setOf_eq, abs_eq_iff, abs_eq_iff'.mp]
  rw [hx, Finset.sum_insert, Finset.sum_singleton]
  norm_num
  all_goals { simp }
sorry

end sum_of_x_satisfies_sqrt_eq_9_l408_408321


namespace joey_speed_on_way_back_eq_six_l408_408738

theorem joey_speed_on_way_back_eq_six :
  ∃ (v : ℝ), 
    (∀ (d t : ℝ), 
      d = 2 ∧ t = 1 →  -- Joey runs a 2-mile distance in 1 hour
      (∀ (d_total t_avg : ℝ),
        d_total = 4 ∧ t_avg = 3 →  -- Round trip distance is 4 miles with average speed 3 mph
        (3 = 4 / (1 + 2 / v) → -- Given average speed equation
         v = 6))) := sorry

end joey_speed_on_way_back_eq_six_l408_408738


namespace total_earnings_l408_408735

variable (h_1 h_2 : ℕ) -- hours worked in first and second weeks
variable (w : ℝ) -- hourly wage
variable (E_diff : ℝ) -- additional earnings in second week

-- Given conditions
def hours_first_week := h_1 = 15
def hours_second_week := h_2 = 22
def additional_earnings := E_diff = 47.60
def constant_hourly_wage := ∀ t₁ t₂ : ℝ, t₁ * w = t₂ * w → t₁ = t₂

-- Prove the total earnings for the two weeks
theorem total_earnings (hw1 : hours_first_week)
                        (hw2 : hours_second_week)
                        (he : additional_earnings)
                        (chw : constant_hourly_wage w) :
  37 * w = 251.60 := by
  sorry

end total_earnings_l408_408735


namespace find_constant_a_l408_408497

theorem find_constant_a :
  (∃ (a : ℝ), a > 0 ∧ (a + 2 * a + 3 * a + 4 * a = 1)) →
  ∃ (a : ℝ), a = 1 / 10 :=
sorry

end find_constant_a_l408_408497


namespace emily_points_l408_408062

theorem emily_points (total_points : ℕ) (points_per_other_player : ℕ) (total_players : ℕ) (emily : ℕ) : 
  total_points = 39 → 
  points_per_other_player = 2 → 
  total_players = 8 → 
  emily = total_points - (points_per_other_player * (total_players - 1)) → 
  emily = 25 :=
by
  -- We state the assumptions
  intros h1 h2 h3 h4
  -- Plug in the given values and simplify
  rw [h1, h2, h3] at h4
  -- We need to show that 39 - (2 * (8 - 1)) equals 25
  have : 39 - (2 * (8 - 1)) = 25 := sorry,
  exact this

end emily_points_l408_408062


namespace sec_330_eq_2_sqrt_3_over_3_l408_408441

theorem sec_330_eq_2_sqrt_3_over_3 :
  (sec (330 * real.pi / 180) = 2 * real.sqrt 3 / 3) :=
by
  have h1: cos (330 * real.pi / 180) = cos (30 * real.pi / 180),
  {
    have h2: cos (330 * real.pi / 180) = cos ((360 - 30) * real.pi / 180), by norm_num,
    have h3: cos ((360 - 30) * real.pi / 180) = cos (-30 * real.pi / 180), by norm_num,
    have h4: cos (-30 * real.pi / 180) = cos (30 * real.pi / 180), by norm_num,
    exact eq.trans (eq.trans h2 h3) h4,
  },
  rw [sec, h1],
  have h_cos_30: cos (30 * real.pi / 180) = real.sqrt 3 / 2, by norm_num,
  rw h_cos_30,
  norm_num

end sec_330_eq_2_sqrt_3_over_3_l408_408441


namespace rectangle_area_l408_408756

theorem rectangle_area (r length width : ℝ) (h_ratio : length = 3 * width) (h_incircle : width = 2 * r) (h_r : r = 7) : length * width = 588 :=
by
  sorry

end rectangle_area_l408_408756


namespace new_edition_pages_less_l408_408764

theorem new_edition_pages_less :
  let new_edition_pages := 450
  let old_edition_pages := 340
  (2 * old_edition_pages - new_edition_pages) = 230 :=
by
  let new_edition_pages := 450
  let old_edition_pages := 340
  sorry

end new_edition_pages_less_l408_408764


namespace probability_of_male_selected_l408_408309

-- Define the total number of students
def num_students : ℕ := 100

-- Define the number of male students
def num_male_students : ℕ := 25

-- Define the number of students selected
def num_students_selected : ℕ := 20

theorem probability_of_male_selected :
  (num_students_selected : ℚ) / num_students = 1 / 5 :=
by
  sorry

end probability_of_male_selected_l408_408309


namespace max_glows_in_time_range_l408_408353

theorem max_glows_in_time_range (start_time end_time : ℤ) (interval : ℤ) (h1 : start_time = 3600 + 3420 + 58) (h2 : end_time = 10800 + 1200 + 47) (h3 : interval = 21) :
  (end_time - start_time) / interval = 236 := 
  sorry

end max_glows_in_time_range_l408_408353


namespace largest_possible_students_l408_408772

theorem largest_possible_students (n m S : ℕ) (h_n : n = 8) (h_m : m = 3)
  (condition : ∀ (x : ℕ), x ≤ ⌊ (n - 1) / (m - 1) ⌋) :
  S = ⌊ (n * ⌊ (n - 1) / (m - 1) ⌋) / m ⌋ := by
  sorry

end largest_possible_students_l408_408772


namespace arctan_sum_eq_half_pi_l408_408836

theorem arctan_sum_eq_half_pi (y : ℚ) :
  2 * Real.arctan (1 / 3) + Real.arctan (1 / 10) + Real.arctan (1 / 30) + Real.arctan (1 / y) = Real.pi / 2 →
  y = 547 / 620 := by
  sorry

end arctan_sum_eq_half_pi_l408_408836


namespace diligent_number_problems_l408_408535

def is_diligent_number (a b c : ℕ) : Prop :=
  b = (a + c) / 2

noncomputable def has_real_roots (a : ℕ) : Prop :=
  let Δ := (2 * a) ^ 2 - 4 * (a - 5) * (a - 8)
  Δ ≥ 0

def digit_sum_in_range (a b c : ℕ) : Prop :=
  7 < (a + b + c) ∧ (a + b + c) < 10

noncomputable def sum_diligent_numbers (L : List (ℕ × ℕ × ℕ)) : ℕ :=
  L.filter (λ ⟨a, b, c⟩, is_diligent_number a b c ∧ has_real_roots a ∧ digit_sum_in_range a b c)
   |>.map (λ ⟨a, b, c⟩, a * 100 + b * 10 + c)
   |>.sum

theorem diligent_number_problems :
  let largest_diligent_number := (999 : ℕ)
  let all_diligent_numbers := [(4, 3, 2), (6, 3, 0)]
  largest_diligent_number = 999 ∧ sum_diligent_numbers all_diligent_numbers = 1062 :=
by
  sorry

end diligent_number_problems_l408_408535


namespace largest_root_eq_l408_408077

theorem largest_root_eq : ∃ x, (∀ y, (abs (Real.cos (Real.pi * y) + y^3 - 3 * y^2 + 3 * y) = 3 - y^2 - 2 * y^3) → y ≤ x) ∧ x = 1 := sorry

end largest_root_eq_l408_408077


namespace farmer_cunninghams_lambs_l408_408024

theorem farmer_cunninghams_lambs :
  ∀ (white_lambs black_lambs : ℕ), (white_lambs = 193) → (black_lambs = 5855) → (white_lambs + black_lambs = 6048) :=
by
  intros white_lambs black_lambs h_white h_black
  rw [h_white, h_black]
  exact rfl

end farmer_cunninghams_lambs_l408_408024


namespace circle_equation_l408_408349

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 16 * x
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def vertex (x y : ℝ) : Prop := x = 0 ∧ y = 0
def focus (x y : ℝ) : Prop := x = 4 ∧ y = 0
def center (cx cy : ℝ) : Prop := parabola cx cy ∧ first_quadrant cx cy
def passes_through (x₀ y₀ cx cy r : ℝ) : Prop :=
  (x₀ - cx)^2 + (y₀ - cy)^2 = r^2

theorem circle_equation :
  ∃ cx cy r,
    center cx cy ∧
    passes_through 0 0 cx cy r ∧
    passes_through 4 0 cx cy r ∧
    (∀ x y, (x - cx)^2 + (y - cy)^2 = r^2 ↔ (x - 2)^2 + (y - 4 * real.sqrt 2)^2 = 36) :=
by
  -- Variables for center and radius
  let x_center := 2
  let y_center := 4 * real.sqrt 2
  let radius := 6
  
  -- Proving the statements
  existsi [x_center, y_center, radius]
  simp [center, passes_through, parabola, first_quadrant, vertex, focus]
  split
  { 
    -- Proving the center lies on the parabola and is in the first quadrant
    use [x_center, y_center]
    split
    { show y_center^2 = 16 * x_center },
    { show x_center > 0 },
    { show y_center > 0 }
  },
  sorry,
  sorry

end circle_equation_l408_408349


namespace math_problem_l408_408898

-- Define the ellipse C with given eccentricity and other properties
def ellipse_eqn (a b : ℝ) (h1 : a > b) (h2 : b > 0) (e : ℝ) (h3 : e = sqrt 3 / 2) 
  : Prop :=
  let c := sqrt (a^2 - b^2) in
  a * e = c ∧
  (1 / 2) * (2 * a) * (2 * b) = 4 ∧
  c^2 = a^2 - b^2 ∧
  ((a = 2) ∧ (b = 1))

-- Define the circle M with given radius r
def circle_eqn (r : ℝ) (h4 : 0 < r) (h5 : r < 1) : Prop :=
  ∀ x y, ((x + 1)^2 + y^2 = r^2)

-- Conditions for slopes of lines AB and AD to be k1 and k2
def slopes_AB_AD (r : ℝ) (k1 k2 : ℝ) : Prop :=
  let eq := (1 - r^2) * k1^2 - 2 * k1 + 1 - r^2 in
  eq = 0 ∧
  (1 - r^2) * k2^2 - 2 * k2 + 1 - r^2 = 0

theorem math_problem (a b r k1 k2 : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : sqrt 3 / 2 = sqrt (a^2 - b^2) / a)
                     (h4 : 0 < r) (h5 : r < 1) (h6 : ellipse_eqn a b h1 h2 (sqrt 3 / 2) h3)
                     (h7 : circle_eqn r h4 h5) (h8 : slopes_AB_AD r k1 k2) : 
  (a = 2 ∧ b = 1) ∧
  (k1 * k2 = 1) ∧
  ∃ y, ( ∀ x, ((4 * k1^2 + 1) * x^2 + 8 * k1 * x + (1 - 4 * k1^2)) = 0 
    → ((B = (-(8*k1)/(4*k1^2 + 1), -(4*k1^2 - 1)/(4*k1^2 + 1)) ∧ 
    D = (-(8*k2)/(4*k2^2 + 1), -(4*k2^2 - 1)/(4*k2^2 + 1)) 
    → (line BD : y = (-k1 - k2)/3 * (x + 8*k1/(4*k1^2 + 1)) + (-20*k1^2 - 5)/(3 * (4*k1^2 + 1))
    → BD passes through fixed point (0, y) ∧ y = -5/3) )) :=
begin
  sorry
end

end math_problem_l408_408898


namespace prove_inequality_l408_408117

variable (x y z : ℝ)
variable (h₁ : x > 0)
variable (h₂ : y > 0)
variable (h₃ : z > 0)
variable (h₄ : x + y + z = 1)

theorem prove_inequality :
  (3 * x^2 - x) / (1 + x^2) +
  (3 * y^2 - y) / (1 + y^2) +
  (3 * z^2 - z) / (1 + z^2) ≥ 0 :=
by
  sorry

end prove_inequality_l408_408117


namespace product_equals_fraction_l408_408998

/-- Define the sequence G using a fixed-point operator
defined as: G_1 = 1, G_2 = 2, and G_{n+1} = G_n + G_{n-1} -/
def G : ℕ → ℕ
  | 0     => 1     -- Lean indices start at 0, so G_1 is G 0
  | 1     => 2     -- G_2 is G 1
  | (n+2) => G n + G (n+1) -- the recursive definition

/-- Prove that the given product equals G_{50} / G_{51} -/
theorem product_equals_fraction :
  (∏ k in Finset.range 49, (G (k + 2) / G (k + 1)) - (G (k + 2) / G (k + 3))) = 
  (G 50 / G 51) := by
  sorry

end product_equals_fraction_l408_408998


namespace ivan_obs_correct_l408_408434

def bus_schedule (time_intervals : set ℝ) : Prop := sorry

def ivan_arrival (random_time : ℝ) (time_intervals : set ℝ) : Prop := sorry

def bus_first_comes 
  (time_intervals_A time_intervals_B : set ℝ) 
  (random_time : ℝ) 
  (arrival_freq_A arrival_freq_B : ℝ) : Prop := 
arrival_freq_A * time_intervals_A = arrival_freq_B * time_intervals_B

theorem ivan_obs_correct 
  (arrival_freq_A arrival_freq_B : ℝ) 
  (time_intervals_A time_intervals_B : set ℝ)
  (random_time : ℝ) : 
  ivan_arrival random_time (time_intervals_A ∪ time_intervals_B) →
  bus_first_comes time_intervals_A time_intervals_B random_time arrival_freq_A arrival_freq_B →
  (arrival_freq_B = 2 * arrival_freq_A) ∧ 
  ((size time_intervals_B) / (size (time_intervals_A ∪ time_intervals_B))) = 2 / 3 :=
sorry

end ivan_obs_correct_l408_408434


namespace range_of_independent_variable_l408_408980

theorem range_of_independent_variable {x : ℝ} (hx : 1 - x ≥ 0) : x ≤ 1 :=
by {
  have h : -x ≥ -1 := by linarith,
  exact (ge_iff_le.mp h),
}

end range_of_independent_variable_l408_408980


namespace max_value_of_k_l408_408339

def is_arithmetic_sequence (s : Set ℕ) : Prop :=
  ∃ d a, ∀ x ∈ s, ∃ n : ℕ, x = a + n * d

def arithmetic_intersection_max_value (n : ℕ) : ℕ :=
  ∑ k in Finset.range 4, Nat.choose n k

theorem max_value_of_k (A : ℕ → Set ℕ) (k n : ℕ) (h_subsets : ∀ i < k, A i ⊆ Finset.range n)
  (h_distinct : Function.Injective A)
  (h_arith_inter : ∀ i j, i < j → is_arithmetic_sequence (A i ∩ A j)) :
  k ≤ arithmetic_intersection_max_value n :=
sorry

end max_value_of_k_l408_408339


namespace max_a6_l408_408111

theorem max_a6 (a : Fin 6 → ℝ) (h_sorted : ∀ i j, i ≤ j → a i ≤ a j)
  (h_sum : ∑ i, a i = 10)
  (h_dev_sq : ∑ i, (a i - 1) ^ 2 = 6) :
  a 5 ≤ 10 / 3 :=
sorry

end max_a6_l408_408111


namespace chocolate_distribution_l408_408991

theorem chocolate_distribution :
  let total_chocolate := 60 / 7
  let piles := 5
  let eaten_piles := 1
  let friends := 2
  let one_pile := total_chocolate / piles
  let remaining_chocolate := total_chocolate - eaten_piles * one_pile
  let chocolate_per_friend := remaining_chocolate / friends
  chocolate_per_friend = 24 / 7 :=
by
  sorry

end chocolate_distribution_l408_408991


namespace maximum_m_proof_l408_408954

noncomputable def maximum_m (f : ℝ → ℝ) : ℤ :=
  if h : (∀ x : ℝ, x < 0 → f x < f (x - 1))
     ∧ (∀ x : ℝ, x > 0 → f x > f (x + 1))
  then (-1 : ℤ)
  else 0

theorem maximum_m_proof :
  (∀ (m : ℤ) (x : ℝ), f x = x ^ ((1 - 3 * m) / 5)
    → (∀ x : ℝ, x < 0 → f x < f (x - 1))
    ∧ (∀ x : ℝ, x > 0 → f x > f (x + 1)))
  → maximum_m (λ x, x ^ ((1 - 3 * m) / 5)) = -1 :=
begin
  intros m h,
  dsimp [maximum_m],
  split_ifs,
  exact dec_trivial,
  contradiction,
  sorry,
end


end maximum_m_proof_l408_408954


namespace MashaGiftsMax_l408_408606

def MashaCandies :=
  { Lastochka Grams Truffle kg PtichyeMoloko kg Citron kg: ℕ
  Lastochka := 2000,
  Truffle := 3000,
  PtichyeMoloko := 4000,
  Citron := 5000
  }

def CandiesConstraint :=
  ∀ (A B C : ℕ), (A ≠ B ∧ B ≠ C ∧ A ≠ C) → (3000 : ℕ)

theorem MashaGiftsMax :
  (2000, 3000, 4000, 5000) → 3000))

end MashaGiftsMax_l408_408606


namespace smallest_odd_number_with_specified_prime_factors_l408_408714

theorem smallest_odd_number_with_specified_prime_factors : 
  ∃ n : ℕ, (n % 2 = 1) ∧ (∃ (p q r s : ℕ), prime p ∧ prime q ∧ prime r ∧ prime s ∧ p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ n = p * q * r * s ∧ (13 = p ∨ 13 = q ∨ 13 = r ∨ 13 = s)) ∧ 
  (∀ m : ℕ, (m % 2 = 1) ∧ (∃ (p q r s : ℕ), prime p ∧ prime q ∧ prime r ∧ prime s ∧ p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ m = p * q * r * s ∧ (13 = p ∨ 13 = q ∨ 13 = r ∨ 13 = s)) → n ≤ m) ∧ 
  n = 1365 :=
sorry

end smallest_odd_number_with_specified_prime_factors_l408_408714


namespace compare_powers_l408_408090

theorem compare_powers (a b c : ℝ) (h1 : a = 2^555) (h2 : b = 3^444) (h3 : c = 6^222) : a < c ∧ c < b :=
by
  sorry

end compare_powers_l408_408090


namespace mitchell_pizzas_l408_408612

def pizzas_bought (slices_per_goal goals_per_game games slices_per_pizza : ℕ) : ℕ :=
  (slices_per_goal * goals_per_game * games) / slices_per_pizza

theorem mitchell_pizzas : pizzas_bought 1 9 8 12 = 6 := by
  sorry

end mitchell_pizzas_l408_408612


namespace artist_paints_37_sq_meters_l408_408788

-- Define the structure of the sculpture
def top_layer : ℕ := 1
def middle_layer : ℕ := 5
def bottom_layer : ℕ := 11
def edge_length : ℕ := 1

-- Define the exposed surface areas
def exposed_surface_top_layer := 5 * top_layer
def exposed_surface_middle_layer := 1 * 5 + 4 * 4
def exposed_surface_bottom_layer := bottom_layer

-- Calculate the total exposed surface area
def total_exposed_surface_area := exposed_surface_top_layer + exposed_surface_middle_layer + exposed_surface_bottom_layer

-- The final theorem statement
theorem artist_paints_37_sq_meters (hyp1 : top_layer = 1)
  (hyp2 : middle_layer = 5)
  (hyp3 : bottom_layer = 11)
  (hyp4 : edge_length = 1)
  : total_exposed_surface_area = 37 := 
by
  sorry

end artist_paints_37_sq_meters_l408_408788


namespace modulus_product_l408_408427

open Complex

theorem modulus_product :
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  (Complex.abs (z1 * z2)) = Real.sqrt 8450 :=
by
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  sorry

end modulus_product_l408_408427


namespace find_common_difference_l408_408221

def is_arithmetic_sequence (a : (ℕ → ℝ)) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def is_arithmetic_sequence_with_sum (a : (ℕ → ℝ)) (S : (ℕ → ℝ)) (d : ℝ) : Prop :=
  S 0 = a 0 ∧
  ∀ n, S (n + 1) = S n + a (n + 1) ∧
        ∀ n, (S (n + 1) / a (n + 1) - S n / a n) = d

theorem find_common_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a →
  is_arithmetic_sequence_with_sum a S d →
  (d = 1 ∨ d = 1 / 2) :=
sorry

end find_common_difference_l408_408221


namespace certain_number_is_four_l408_408851

theorem certain_number_is_four (k : ℕ) (h₁ : k = 16) : 64 / k = 4 :=
by
  sorry

end certain_number_is_four_l408_408851


namespace other_x_intercept_is_56_over_11_l408_408789

     -- Define the basic structures for a point and the ellipse's properties
     structure Point where
       x : ℝ
       y : ℝ

     def focus1 : Point := { x := 0, y := 3 }
     def focus2 : Point := { x := 4, y := 0 }
     def intercept1 : Point := { x := 0, y := 0 }
     def intercept2 : Point := { x := 56 / 11, y := 0 }

     -- Define the function to compute the distance between two points
     def distance (p1 p2 : Point) : ℝ := ( ( p1.x - p2.x )^2 + ( p1.y - p2.y )^2 ).sqrt

     -- Define the property holding for all points on the ellipse
     def ellipse_property (P : Point) : Prop :=
       distance P focus1 + distance P focus2 = 7

     -- The proof problem statement
     theorem other_x_intercept_is_56_over_11 : ellipse_property intercept1 → intercept2.x = 56 / 11 :=
     by
       sorry
     
end other_x_intercept_is_56_over_11_l408_408789


namespace inequality_solution_l408_408537

theorem inequality_solution (a b : ℝ)
  (h : ∀ x : ℝ, 0 ≤ x → (x^2 + 1 ≥ a * x + b ∧ a * x + b ≥ (3 / 2) * x^(2 / 3) )) :
  (2 - Real.sqrt 2) / 4 ≤ b ∧ b ≤ (2 + Real.sqrt 2) / 4 ∧
  (1 / Real.sqrt (2 * b)) ≤ a ∧ a ≤ 2 * Real.sqrt (1 - b) :=
  sorry

end inequality_solution_l408_408537


namespace tara_books_sold_l408_408264

def savings_before_loss : ℕ := 10
def clarinet_cost : ℕ := 90
def book_sale_price : ℕ := 5
def accessory_cost : ℕ := 20
def half_goal : ℕ := (clarinet_cost - savings_before_loss) / 2
def books_needed_to_half_goal : ℕ := half_goal / book_sale_price
def full_savings_needed : ℕ := clarinet_cost - savings_before_loss
def total_savings_needed : ℕ := full_savings_needed + accessory_cost
def books_needed_post_loss : ℕ := total_savings_needed / book_sale_price

theorem tara_books_sold : 
  let books_before_loss := books_needed_to_half_goal,
      books_after_loss := books_needed_post_loss
  in books_before_loss + books_after_loss = 28 :=
by sorry

end tara_books_sold_l408_408264


namespace max_value_of_symmetric_function_l408_408945

def f (x a b : ℝ) : ℝ := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function (a b : ℝ) (h_sym : ∀ x : ℝ, f x a b = f (-4 - x) a b) : 
  ∃ x : ℝ, (∀ y : ℝ, f y a b ≤ f x a b) ∧ f x a b = 16 :=
by
  sorry

end max_value_of_symmetric_function_l408_408945


namespace wall_height_is_4_l408_408781

-- Definitions based on the conditions
def wall_width : ℝ := 4
def wall_area : ℝ := 16

-- Goal: Prove the height of the wall
theorem wall_height_is_4 :
  ∃ height : ℝ, height = wall_area / wall_width ∧ height = 4 :=
by
  exists 4
  split
  sorry
  rfl

end wall_height_is_4_l408_408781


namespace line_curve_disjoint_range_x_plus_y_l408_408108

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  ( (sqrt 2 / 2) * t, (sqrt 2 / 2) * t + 4 * sqrt 2 )

noncomputable def polar_curve (θ : ℝ) : ℝ :=
  2 * cos (θ + π / 4)

-- Goal 1: The line l and the curve C are disjoint.
theorem line_curve_disjoint :
  ∀ (l_point : ℝ × ℝ) (θ : ℝ),
  let l_point := parametric_line t in
  let curve_point := 
    ((sqrt 2 / 2) + cos θ, -(sqrt 2 / 2) + sin θ) in
  distance l_point curve_point > 1 := 
  sorry

-- Goal 2: The range of values for x + y for any point on curve C.
theorem range_x_plus_y :
  ∀ (θ : ℝ),
  let x := (sqrt 2 / 2) + cos θ in
  let y := -(sqrt 2 / 2) + sin θ in
  x + y ∈ set.interval (-sqrt 2) (sqrt 2) := 
  sorry

end line_curve_disjoint_range_x_plus_y_l408_408108


namespace incorrect_statement_D_l408_408973

def total_balls : ℕ := 5
def red_balls : ℕ := 3
def white_balls : ℕ := 2

theorem incorrect_statement_D :
  ¬((3 / 5) * (2 / 4) = 1 / 5) :=
by
  -- Compute the correct probability of drawing two red balls
  have correct_prob_of_two_reds : (3 / 5) * (2 / 4) = 3 / 10 := sorry,
  -- Use this fact to establish the incorrectness of statement D
  intro h,
  rw ← correct_prob_of_two_reds at h,
  -- Reach a contradiction showing that statement D is indeed incorrect
  sorry

end incorrect_statement_D_l408_408973


namespace next_valid_number_after_1818_l408_408677

open Nat

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem next_valid_number_after_1818 : 
  ∀ (a b : ℕ), 10 * a + b > 18 → 18 * (10 * a + b) = 1832 → isPerfectSquare (18 * (10 * a + b)) → 
  10 * 10 * 8 + 10 * a + b = 1832 :=
by {
  intros,
  sorry
}

end next_valid_number_after_1818_l408_408677


namespace sqrt_expression_is_80_l408_408312

theorem sqrt_expression_is_80 :
  real.cbrt 125 * real.root 256 4 * real.sqrt 16 = 80 := by
  sorry

end sqrt_expression_is_80_l408_408312


namespace quadrilateral_angle_C_l408_408547

/-
Problem: Given a quadrilateral ABCD with the following conditions,
prove that angle C in quadrilateral can be 60 degrees.

Conditions:
1. angle ABE = 40 degrees
2. angle AEB = 70 degrees
3. sides AB = BE = BC
-/

theorem quadrilateral_angle_C (ABCD : Quadrilateral) 
  (H1 : ∠ ABE = 70)
  (H2 : ∠ ABE = 40)
  (H3 : AB = BE ∧ BE = BC) : ∠C = 60 := 
begin
  sorry,
end

end quadrilateral_angle_C_l408_408547


namespace ellipse_equation_solution_line_intersection_solution_l408_408882

def ellipse_equation(a b : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) 

def focal_sum(a b : ℝ) : Prop :=
  ∀ (x y : ℝ), (∃ (f1 f2 : ℝ × ℝ), 
    (f1 = (sqrt (a^2 - b^2), 0) ∧ f2 = (-sqrt (a^2 - b^2), 0)) ∧ 
    (sqrt ((x - f1.1)^2 + y^2) + sqrt ((x - f2.1)^2 + y^2) = 6))

def line_intersection(k : ℝ) : Prop :=
  ∀ (x y : ℝ), (y = k * x - 2) ∧ (∃ (y_1 y_2 : ℝ), |(0,1) - (x,y_1)| = |(0,1) - (x,y_2)|)

theorem ellipse_equation_solution :
  (∃ (a b : ℝ), a = 3 ∧ b = sqrt (9 - b^2) ∧ 
    ellipse_equation 3 (sqrt 3) ∧ focal_sum 3 (sqrt 3)) :=
begin
  sorry
end

theorem line_intersection_solution :
  ∀ (k : ℝ), (line_intersection k → (k = 1 ∨ k = -1) 
  ∧ ((∀ x, y = x-2) ∨ (∀ x, y = -x-2))) :=
begin
  sorry
end

end ellipse_equation_solution_line_intersection_solution_l408_408882


namespace Foster_Farms_donated_45_chickens_l408_408080

def number_of_dressed_chickens_donated_by_Foster_Farms (C AS H BB D : ℕ) : Prop :=
  C + AS + H + BB + D = 375 ∧
  AS = 2 * C ∧
  H = 3 * C ∧
  BB = C ∧
  D = 2 * C - 30

theorem Foster_Farms_donated_45_chickens:
  ∃ C, number_of_dressed_chickens_donated_by_Foster_Farms C (2*C) (3*C) C (2*C - 30) ∧ C = 45 :=
by 
  sorry

end Foster_Farms_donated_45_chickens_l408_408080


namespace equivalent_expression_ratio_l408_408244

theorem equivalent_expression_ratio :
  let c : ℚ := 8
  let p : ℚ := -3 / 8
  let q : ℚ := 604 / 32
  (c * ((j + p)^2) + q = 8 * j^2 - 6 * j + 20) →
  (q / p) = -151 / 3 :=
by
  intros c p q h
  rw [← h]
  sorry

end equivalent_expression_ratio_l408_408244


namespace mass_percentage_Al_in_AlBr₃_l408_408842

theorem mass_percentage_Al_in_AlBr₃ :
  let Al_mass := 26.98
  let Br_mass := 79.90
  let M_AlBr₃ := Al_mass + 3 * Br_mass
  (Al_mass / M_AlBr₃ * 100) = 10.11 :=
by 
  let Al_mass := 26.98
  let Br_mass := 79.90
  let M_AlBr₃ := Al_mass + 3 * Br_mass
  have : (Al_mass / M_AlBr₃ * 100) = 10.11 := sorry
  assumption

end mass_percentage_Al_in_AlBr₃_l408_408842


namespace x_8_sufficient_condition_l408_408957

noncomputable def magnitude (a : ℝ × ℝ) : ℝ :=
  real.sqrt (a.1 * a.1 + a.2 * a.2)

theorem x_8_sufficient_condition (x : ℝ) (h : x = 8) : magnitude (6, x) = 10 :=
by
  -- Proof to be provided
  sorry

end x_8_sufficient_condition_l408_408957


namespace combined_average_yield_l408_408818

theorem combined_average_yield (yield_A : ℝ) (price_A : ℝ) (yield_B : ℝ) (price_B : ℝ) (yield_C : ℝ) (price_C : ℝ) :
  yield_A = 0.20 → price_A = 100 → yield_B = 0.12 → price_B = 200 → yield_C = 0.25 → price_C = 300 →
  (yield_A * price_A + yield_B * price_B + yield_C * price_C) / (price_A + price_B + price_C) = 0.1983 :=
by
  intros hYA hPA hYB hPB hYC hPC
  sorry

end combined_average_yield_l408_408818


namespace dolly_needs_more_tickets_l408_408408

-- Definitions of the conditions
def rides_ferris_wheel := 2
def cost_ferris_wheel := 2
def rides_roller_coaster := 3
def cost_roller_coaster := 5
def rides_log_ride := 7
def cost_log_ride := 1
def tickets_dolly_has := 20

-- The theorem statement
theorem dolly_needs_more_tickets :
  let total_tickets_needed := rides_ferris_wheel * cost_ferris_wheel + 
                              rides_roller_coaster * cost_roller_coaster +
                              rides_log_ride * cost_log_ride
  in total_tickets_needed - tickets_dolly_has = 6 :=
by
  sorry

end dolly_needs_more_tickets_l408_408408


namespace max_value_of_g_l408_408648

def g : ℕ → ℕ
| n := if n < 15 then 2 * n + 3 else g (n - 7)

theorem max_value_of_g : ∃ n, ∀ m, g m ≤ 31 ∧ g n = 31 :=
by {
  sorry -- proof would go here
}

end max_value_of_g_l408_408648


namespace benches_needed_l408_408638

def base6_to_decimal : ℕ → ℕ :=
  λ n, let d2 := n / 100, d1 := (n / 10) % 10, d0 := n % 10 in
       d2 * 6^2 + d1 * 6^1 + d0 * 6^0

theorem benches_needed (n : ℕ) : n = 320 → (base6_to_decimal n) / 3 = 40 :=
by 
  intros h
  rw [h, base6_to_decimal]
  norm_num
  sorry

end benches_needed_l408_408638


namespace longest_chord_intersecting_point_P_l408_408375

def circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0

def point_on_circle (x y : ℝ) : Prop := x = 0 ∧ y = 1

theorem longest_chord_intersecting_point_P : 
  ∀ (L : ℝ → ℝ → Prop),
  (∀ x y, circle x y → point_on_circle x y → L x y = 0) →
  (L 1 0 = 0 ∧ L 0 1 = 0) →
  L = (λ x y, x + y - 1) :=
by
  sorry

end longest_chord_intersecting_point_P_l408_408375


namespace cos_angle_bac_cone_l408_408216

theorem cos_angle_bac_cone {O A B C : Point}
  (hOAB : angle O A B = 90)
  (hOBC : angle O B C = 90)
  (hOCA : angle O C A = 90)
  (a : ℝ) (b : ℝ)
  (ha : sin (angle A O C) = a)
  (hb : sin (angle B O C) = b) :
  cos (angle B A C) = 1 / (sqrt (1 + a^2) * sqrt (1 + b^2)) :=
sorry

end cos_angle_bac_cone_l408_408216


namespace cos2α_eq_β_eq_l408_408928

variables (α β : ℝ)
local notation "π" => Real.pi

-- Condition 1: α ∈ (0, π/2)
def α_in_range := 0 < α ∧ α < π / 2

-- Condition 2: Vectors m and n are orthogonal
def m := (Real.cos α, -1)
def n := (2, Real.sin α)
def m_perp_n := m.1 * n.1 + m.2 * n.2 = 0

-- Condition 3: sin(α - β) = √10 / 10
def sin_alpha_minus_beta := Real.sin (α - β) = Real.sqrt 10 / 10

-- Condition 4: β ∈ (0, π/2)
def β_in_range := 0 < β ∧ β < π / 2

-- Theorem 1: To prove cos(2α) = -3/5 given the conditions
theorem cos2α_eq :
  α_in_range α →
  m_perp_n α →
  Real.cos (2 * α) = -3 / 5 :=
sorry

-- Theorem 2: To prove β = π/4 given the conditions
theorem β_eq :
  α_in_range α →
  β_in_range β →
  sin_alpha_minus_beta α β →
  β = π / 4 :=
sorry

end cos2α_eq_β_eq_l408_408928


namespace diagonals_bisect_l408_408621

def midpoint (x1 y1 x2 y2 : ℤ) : ℤ × ℤ :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

theorem diagonals_bisect (A C : ℤ × ℤ) : A = (1, -5) → C = (11, 7) → midpoint 1 (-5) 11 7 = (6, 1) :=
by
  intros h1 h2
  rw [h1, h2]
  simp
  sorry

end diagonals_bisect_l408_408621


namespace circle_has_greatest_symmetry_l408_408733

theorem circle_has_greatest_symmetry (
  eq_triangle_symm : ℕ := 3
  iso_trapezoid_symm : ℕ := 1
  circle_symm : ℕ := 0 -- we will use ℕ for simplicity but consider it as "infinite"
  ns_rectangle_symm : ℕ := 2
  reg_pentagon_symm : ℕ := 5
) : ∀ x : ℕ, x ≠ 0 → x ≤ eq_triangle_symm ∨ x ≤ iso_trapezoid_symm ∨ x ≤ ns_rectangle_symm ∨ x ≤ reg_pentagon_symm :=
begin
  intros x hx,
  sorry,  -- Proof to be filled in.
end

end circle_has_greatest_symmetry_l408_408733


namespace arctan_gt_arccot_l408_408073

theorem arctan_gt_arccot {x : ℝ} (h : 1 < x) : real.arctan x > real.arccot x :=
sorry

end arctan_gt_arccot_l408_408073


namespace sin_300_l408_408829

def sin_300_eq : Real := - Real.sqrt 3 / 2

theorem sin_300 : Real.sin (300 * Real.pi / 180) = sin_300_eq := 
  sorry

end sin_300_l408_408829


namespace f_prime_even_g_prime_symmetric_about_3_2_f_period_6_l408_408220

variable {ℝ : Type} [Field ℝ] [Differentiable ℝ]

-- Conditions
variables (f g : ℝ → ℝ) 
hypothesis H1 : ∀ x, f (x + 2) - g (1 - x) = 2
hypothesis H2 : ∀ x, deriv f x = deriv g x
hypothesis H3 : ∀ x, g (-x) = -g (x)

-- Prove B: f'(x) is an even function
theorem f_prime_even : ∀ x, deriv f (-x) = deriv f x := 
by {
  sorry 
}

-- Prove C: The graph of g'(x) is symmetric about the point (3/2, 0)
theorem g_prime_symmetric_about_3_2 : ∀ x, deriv g x = -deriv g (3 - x) := 
by {
  sorry 
}

-- Prove D: One period of f(x) is 6
theorem f_period_6 : ∀ x, f (x + 6) = f x := 
by {
  sorry 
}

end f_prime_even_g_prime_symmetric_about_3_2_f_period_6_l408_408220


namespace length_XY_eq_l408_408553

theorem length_XY_eq (O A B X Y : Type) [MetricSpace O] [MetricSpace A] [MetricSpace B] [MetricSpace X] [MetricSpace Y]
  (r : ℝ) (h_r : r = 12)
  (angle_AOB : Real.Angle) (h_angle_AOB : angle_AOB = Real.Angle.ofDeg 120)
  (OX_perpendicular_AB : IsPerpendicular (line O Y) (line A B))
  (distance_OA_OB : dist O A = r ∧ dist O B = r)
  (length_OY : dist O Y = r) :
  dist X Y = 12 - 6 * Real.sqrt 3 := by
  sorry 

end length_XY_eq_l408_408553


namespace nature_of_F_l408_408166

noncomputable def F (x : ℝ) : ℝ := 
  let base_area := √3 / 4
  let height := sorry -- The height as a function of x, which depends on geometric calculations
  (base_area * height) / 3

theorem nature_of_F :
  ∃ x_max ∈ (0, √3), (∀ x ∈ (0, x_max], F x ≤ F x_max) ∧ (∀ x ∈ [x_max, √3), F x_max ≥ F x) := sorry

end nature_of_F_l408_408166


namespace wrong_observation_value_l408_408282

theorem wrong_observation_value :
  ∃ v : ℝ, let old_total := 50 * 30 in
           let new_total := 50 * 30.5 in
           let diff := new_total - old_total in
           v = 48 + diff :=
by
  let old_total := 50 * 30
  let new_total := 50 * 30.5
  let diff := new_total - old_total
  use (48 + diff)
  sorry

end wrong_observation_value_l408_408282


namespace large_pizza_slices_l408_408620

-- Definitions and conditions based on the given problem
def slicesEatenByPhilAndre : ℕ := 9 * 2
def slicesLeft : ℕ := 2 * 2
def slicesOnSmallCheesePizza : ℕ := 8
def totalSlices : ℕ := slicesEatenByPhilAndre + slicesLeft

-- The theorem to be proven
theorem large_pizza_slices (slicesEatenByPhilAndre slicesLeft slicesOnSmallCheesePizza : ℕ) :
  slicesEatenByPhilAndre = 18 ∧ slicesLeft = 4 ∧ slicesOnSmallCheesePizza = 8 →
  totalSlices - slicesOnSmallCheesePizza = 14 :=
by
  intros h
  sorry

end large_pizza_slices_l408_408620


namespace find_p_l408_408129

noncomputable def f (p : ℝ) : ℝ := 2 * p^2 + 20 * Real.sin p

theorem find_p : ∃ p : ℝ, f (f (f (f p))) = -4 :=
by
  sorry

end find_p_l408_408129


namespace f_monotonically_increasing_l408_408827

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x + Real.pi / 6)

theorem f_monotonically_increasing (x : ℝ) :
  x ∈ Set.Icc (-Real.pi / 6) 0 → Monotone (λ x, f x) :=
sorry

end f_monotonically_increasing_l408_408827


namespace find_a_for_exactly_two_solutions_l408_408839

theorem find_a_for_exactly_two_solutions :
  ∃ a : ℝ, (∀ x : ℝ, (|x + a| = 1/x) ↔ (a = -2) ∧ (x ≠ 0)) ∧ ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (|x1 + a| = 1/x1) ∧ (|x2 + a| = 1/x2) :=
sorry

end find_a_for_exactly_two_solutions_l408_408839


namespace area_triangle_ABC_l408_408982

noncomputable theory

open Real EuclideanGeometry

-- Definitions reflecting the conditions in the problem
def triangle_ABC (A B C : Point) : Prop :=
  true

def median_CD (A B C D : Point) : Prop :=
  (dist D A = dist D B) ∧ (segment C D ∈ line_segment A B)

def side_AB (A B : Point) : ℝ :=
  dist A B = 6

def sum_BC_AC (B C A : Point) : ℝ :=
  dist B C + dist A C = 8

-- The statement reflecting the question and the correct answer
theorem area_triangle_ABC {A B C D : Point}
  (hABC : triangle_ABC A B C)
  (hCD : median_CD A B C D)
  (hAB : side_AB A B)
  (hBCAC : sum_BC_AC B C A) :
  area A B C = 7 :=
sorry

end area_triangle_ABC_l408_408982


namespace quadratic_inequality_range_l408_408854

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2 : ℝ) 2 :=
sorry

end quadratic_inequality_range_l408_408854


namespace jill_show_duration_l408_408566

theorem jill_show_duration :
  let first_show_duration := 30
  let second_show_duration := 4 * first_show_duration
  first_show_duration + second_show_duration = 150 :=
by
  let first_show_duration := 30
  let second_show_duration := 4 * first_show_duration
  have h1 : second_show_duration = 120 := by rfl
  have h2 : first_show_duration + second_show_duration = 150 := by rfl
  show first_show_duration + second_show_duration = 150 from h2

end jill_show_duration_l408_408566


namespace problem_rewrite_expression_l408_408245

theorem problem_rewrite_expression (j : ℝ) : 
  ∃ (c p q : ℝ), (8 * j^2 - 6 * j + 20 = c * (j + p)^2 + q) ∧ (q / p = -77) :=
sorry

end problem_rewrite_expression_l408_408245


namespace roots_shifted_by_one_l408_408459

theorem roots_shifted_by_one (P : Polynomial ℝ) (h_deg : P.degree = 45)
  (h_coeffs : ∃ σ : Fin 46 → Fin 46, ∀ i, P.coeff i = (σ i).val + 1)
  (h_distinct_roots : P.roots.Nodup) : 
  (∃ nPos nNeg : ℕ, nPos = nNeg ∧
    ∀ r ∈ P.roots, (r + 1 ∈ Set.Ioo (-1 : ℝ) 0 → nPos += 1) ∧ 
                   (r + 1 < 0 → nNeg += 1)) :=
sorry

end roots_shifted_by_one_l408_408459


namespace point_equidistant_from_B_and_C_l408_408841

noncomputable def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2 + (Q.3 - P.3)^2)

theorem point_equidistant_from_B_and_C :
  ∃ (y : ℝ), let A := (0, y, 0) in
  let B := (2, 2, 4) in
  let C := (0, 4, 2) in
  distance A B = distance A C ∧ A = (0, -1, 0) :=
by
  use -1
  let A := (0, -1, 0)
  let B := (2, 2, 4)
  let C := (0, 4, 2)
  simp [distance]
  sorry

end point_equidistant_from_B_and_C_l408_408841


namespace max_projection_CD_on_AB_l408_408471

open Real

theorem max_projection_CD_on_AB (x : ℝ) : 
  (let AB := (-1, x + 2) in 
   let CD := (x, 1) in 
   let numerator := -1 * x + (x + 2) * 1 in 
   let denominator := sqrt (1 + (x + 2)^2) in 
   numerator / denominator <= 2) :=
by
  let AB := (-1, x + 2)
  let CD := (x, 1)
  let numerator := -1 * x + (x + 2) * 1
  let denominator := sqrt (1 + (x + 2)^2)
  have h : numerator / denominator = 2 / (sqrt (1 + (x + 2)^2)) := rfl
  have h2 : 2 / (sqrt (1 + (x + 2)^2)) <= 2 := sorry
  exact h2

end max_projection_CD_on_AB_l408_408471


namespace range_of_m_l408_408109

theorem range_of_m (m : ℝ) (p : Prop := (∃ a b : ℝ, a + b = -m ∧ a * b = 1 ∧ a < 0 ∧ b < 0))
    (q : Prop := ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 > 0) 
    (p_or_q : p ∨ q)
    (p_and_not_q : p ∧ ¬q ∨ ¬p ∧ q) : 
    m ∈ [3, +∞) ∪ (1, 2] :=
sorry

end range_of_m_l408_408109


namespace correct_conclusions_l408_408274

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := cos (ω * x + π/6)

theorem correct_conclusions (ω : ℝ) (hω : ω > 0) :
  (f ω (π/2) = -√3/2) ∧
  ((∃ p : ℝ, ∀ x, f ω (x + p) = f ω x) ∧ (p = π)) ∧
  (¬even_function (λ x, f ω (x - π/12))) ∧
  (∀ x, 0 < x ∧ x < π/(3 * ω) → f ω x < f ω (x-1)) :=
sorry

end correct_conclusions_l408_408274


namespace abs_az_bz_le_one_iff_l408_408573

open Complex

theorem abs_az_bz_le_one_iff (a b : ℂ) : 
  (∀ z : ℂ, abs z = 1 → abs (a * z + b * conj z) ≤ 1) ↔ (abs a + abs b ≤ 1) :=
sorry

end abs_az_bz_le_one_iff_l408_408573


namespace ratio_milk_water_resulting_mixture_l408_408329

/-- In a mixture of 45 litres with an initial ratio of milk to water as 4:1, 
    after adding 18 litres of water, the new ratio of milk to water is 4:3. -/
theorem ratio_milk_water_resulting_mixture :
  let initial_volume := 45
  let milk_ratio := 4
  let water_ratio := 1
  let additional_water := 18
  let total_parts := milk_ratio + water_ratio
  let initial_milk := (milk_ratio * initial_volume) / total_parts
  let initial_water := (water_ratio * initial_volume) / total_parts
  let new_water := initial_water + additional_water
  let new_ratio := initial_milk / new_water
  4 : 3 := new_ratio :=
sorry

end ratio_milk_water_resulting_mixture_l408_408329


namespace next_perfect_square_after_1818_l408_408691

def is_next_perfect_square_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (∃ m : ℕ, n = 1800 + m ∧ m < 100 ∧ (∃ k : ℕ, 18 * m = k^2))

theorem next_perfect_square_after_1818 : ∃ n : ℕ, is_next_perfect_square_number n ∧ n > 1818 ∧ n = 1832 :=
by
  existsi 1832
  unfold is_next_perfect_square_number
  split
  . apply and.intro
    { linarith }
    { split
      . existsi 32
        split
        . linarith
        . existsi 18
          linarith
      . sorry }

end next_perfect_square_after_1818_l408_408691


namespace min_value_inequality_l408_408587

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  (Real.sqrt ((x^2 + y^2 + z^2) * (4 * x^2 + y^2 + z^2))) / (x * y * z)

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  min_value_expression x y z ≥ 3 / 2 := by
  sorry

end min_value_inequality_l408_408587


namespace square_diagonal_length_l408_408880

theorem square_diagonal_length (R : ℝ) : 
  ∃ (x : ℝ), 
  (2 * R * x - x^2 = x^2 / 4) ∧
  x = 8 * R / 5 ∧
  ∀ (d : ℝ), d = x * sqrt 2 -> d = 8 * sqrt 2 * R / 5 := 
begin
  sorry
end

end square_diagonal_length_l408_408880


namespace extremum_point_of_f_l408_408403

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem extremum_point_of_f :
  ∃ x : ℝ, x = -1 ∧ ∀ y : ℝ, f' x = 0 ∧ f'' x < 0 :=
by 
  sorry

end extremum_point_of_f_l408_408403


namespace find_bc_l408_408500

theorem find_bc 
  (b c : ℝ)
  (h1 : ∀ x y, y = x^3 + b * x + c → (1, 2) → (1, 1))
  (h2 : ∀ x y, y' = 3 * x^2 + b → (1, 2) → (y' = 1)) : 
  b * c = -6 :=
by
  sorry

end find_bc_l408_408500


namespace at_least_one_first_grade_product_l408_408701

noncomputable def probability_first_intern := 2 / 3
noncomputable def probability_second_intern := 1 / 2

theorem at_least_one_first_grade_product :
  let pA := probability_first_intern,
      pB := probability_second_intern,
      p_not_A := 1 - pA,
      p_not_B := 1 - pB,
      p_neither := p_not_A * p_not_B,
      p_at_least_one := 1 - p_neither in
  p_at_least_one = 5 / 6 :=
by
  let pA := probability_first_intern
  let pB := probability_second_intern
  let p_not_A := 1 - pA
  let p_not_B := 1 - pB
  let p_neither := p_not_A * p_not_B
  let p_at_least_one := 1 - p_neither
  show p_at_least_one = 5 / 6
  sorry

end at_least_one_first_grade_product_l408_408701


namespace algorithm_sequential_structure_l408_408064

theorem algorithm_sequential_structure (no_conditional_needed: ¬ (∃ c: Type, ∃ a: c → bool))
                                        (no_looping_needed: ¬ (∃ l: ℕ → bool))
                                        (must_have_sequence: ∀ a: Type, (list a → list a)):
  ∃ seq: Type, must_have_sequence list :=
by
  sorry

end algorithm_sequential_structure_l408_408064


namespace sum_of_c_and_e_l408_408495

-- Define the coordinates of point P
def P : ℝ × ℝ × ℝ := (-4, -2, 3)

-- Define the reflection conditions
def symmetric_xoy (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (P.1, P.2, -P.3)
def symmetric_y_axis (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (-P.1, P.2, P.3)

-- Assign the reflected points to the given variable names
def Q : ℝ × ℝ × ℝ := symmetric_xoy P  -- Reflected point about xoy plane
def R : ℝ × ℝ × ℝ := symmetric_y_axis P  -- Reflected point about y-axis

-- Extract the coordinates from the reflected points
def a : ℝ := Q.1
def b : ℝ := Q.2
def c : ℝ := Q.3

def e : ℝ := R.1
def f : ℝ := R.2
def d : ℝ := R.3

-- Theorem to prove that the sum of c and e is equal to 1
theorem sum_of_c_and_e : c + e = 1 := by
  -- Provide proof here
  sorry

end sum_of_c_and_e_l408_408495


namespace triangle_side_c_sqrt_7_l408_408983

noncomputable def side_c_in_triangle (a b : ℝ) (angle_C : ℝ) : ℝ :=
  real.sqrt $ a^2 + b^2 - 2 * a * b * real.cos angle_C

theorem triangle_side_c_sqrt_7 (a b : ℝ) (C : ℝ) (ha : a = 1) (hb : b = 2) (hC : C = (2 * real.pi) / 3) :
  side_c_in_triangle a b C = real.sqrt 7 := 
  by sorry

end triangle_side_c_sqrt_7_l408_408983


namespace equation_of_curve_C_area_of_triangle_AOB_is_constant_l408_408870

-- Definitions for the conditions
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

def point_N : ℝ × ℝ := (1, 0)

def point_moves_on_circle (P : ℝ × ℝ) : Prop := circle_equation P.1 P.2

def line_segment_NP (N P Q : ℝ × ℝ) : Prop := (N.1 + P.1) / 2 = Q.1 ∧ (N.2 + P.2) / 2 = Q.2

def point_GQ_perpendicular_to_NP (G Q N P : ℝ × ℝ) : Prop :=
  let np := (P.1 - N.1, P.2 - N.2)
  let gq := (G.1 - Q.1, G.2 - Q.2)
  np.1 * gq.1 + np.2 * gq.2 = 0

def point_G (G Q N : ℝ × ℝ) : Prop := G.1 + Q.1 = N.1 ∧ G.2 + Q.2 = N.2

-- Definition for the problem
theorem equation_of_curve_C :
  ∀ (M N P Q G : ℝ × ℝ),
    N = point_N →
    circle_equation M.1 M.2 →
    point_moves_on_circle P →
    line_segment_NP N P Q →
    point_GQ_perpendicular_to_NP G Q N P →
    point_G G Q N →
    (∃ x y : ℝ, 4 * x^2 + 3 * y^2 = 12) :=
sorry

theorem area_of_triangle_AOB_is_constant :
  ∀ (A B O : ℝ × ℝ) (k m : ℝ),
    (∀ k m, ∀ x y : ℝ, 4 * x^2 + 3 * y^2 = 12 ∧ y = k * x + m) →
    let k_OA := (A.2 - O.2) / (A.1 - O.1)
    let k_OB := (B.2 - O.2) / (B.1 - O.1)
    k_OA * k_OB = -3/4 →
    let area := real.sqrt 3 in
    area = real.sqrt 3 :=
sorry

end equation_of_curve_C_area_of_triangle_AOB_is_constant_l408_408870


namespace monotonic_decreasing_interval_l408_408942

noncomputable def f (x : ℝ) : ℝ := (1/3)*x^3 - 2*x^2 + 3*x + c

def f_prime (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem monotonic_decreasing_interval :
  ∀ (c : ℝ), ∃ (a b : ℝ), 0 < a ∧ b < 2 ∧ ∀ x ∈ set.Ioo a b, (f' (x+1) < 0) :=
by 
  sorry

end monotonic_decreasing_interval_l408_408942


namespace inclination_angle_range_l408_408290

theorem inclination_angle_range (m : ℝ) :
  let k := (2 * m) / (m^2 + 1) in
  (0 <= m → 0 <= k ∧ k <= 1) →
  (m < 0 → -1 <= k ∧ k < 0) →
  (∃ θ : ℝ, θ ∈ [0, Real.pi / 4] ∪ [3 * Real.pi / 4, Real.pi]) :=
by
  sorry

end inclination_angle_range_l408_408290


namespace range_of_a_l408_408905

variable {f : ℝ → ℝ}
variable {a : ℝ}

theorem range_of_a (h_deriv : ∀ x, deriv f x = a * (x + 1) * (x - a))
    (h_local_max : IsLocalMax f a) : a ∈ Ioo (-1 : ℝ) 0 := 
by
  sorry

end range_of_a_l408_408905


namespace max_value_of_symmetric_f_l408_408948

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_f :
  ∀ (a b : ℝ),
    (f 1 a b = 0) →
    (f (-1) a b = 0) →
    (f (-5) a b = 0) →
    (f (-3) a b = 0) →
    (∃ x : ℝ, f x 8 15 = 16) :=
by
  sorry

end max_value_of_symmetric_f_l408_408948


namespace trapezoid_construction_l408_408014

theorem trapezoid_construction (r s : ℝ) (h : s > 8 * r) : 
  ∃ (AB BC CD DA : ℝ), 
    (AB = BC) ∧ 
    (DA = AB + 2 * r) ∧ 
    (CD = 2 * r) ∧ 
    (AB + BC + DA + CD = s) ∧ 
    (circle_in_tangential_trapezoid r s AB BC CD DA) :=
  sorry

def circle_in_tangential_trapezoid (r s AB BC CD DA : ℝ) : Prop := 
  -- Here we would define what it means for the circle to be inscribed 
  -- in terms of the given sides of the trapezoid. In our case, this
  -- would be implicit in the construction.
  sorry

end trapezoid_construction_l408_408014


namespace sequence_sum_l408_408029

theorem sequence_sum :
  (∀ r: ℝ, 
    (∀ x y: ℝ,
      r = 1 / 4 → 
      x = 256 * r → 
      y = x * r → 
      x + y = 80)) :=
by sorry

end sequence_sum_l408_408029


namespace excluded_number_is_35_l408_408637

theorem excluded_number_is_35 (numbers : List ℝ) 
  (h_len : numbers.length = 5)
  (h_avg1 : (numbers.sum / 5) = 27)
  (h_len_excl : (numbers.length - 1) = 4)
  (avg_remaining : ℝ)
  (remaining_numbers : List ℝ)
  (remaining_condition : remaining_numbers.length = 4)
  (h_avg2 : (remaining_numbers.sum / 4) = 25) :
  numbers.sum - remaining_numbers.sum = 35 :=
by sorry

end excluded_number_is_35_l408_408637


namespace max_value_of_symmetric_function_l408_408946

def f (x a b : ℝ) : ℝ := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function (a b : ℝ) (h_sym : ∀ x : ℝ, f x a b = f (-4 - x) a b) : 
  ∃ x : ℝ, (∀ y : ℝ, f y a b ≤ f x a b) ∧ f x a b = 16 :=
by
  sorry

end max_value_of_symmetric_function_l408_408946


namespace x_plus_y_l408_408060

noncomputable def sequence_constant : ℝ := 1 / 4

def x : ℝ := 256 * sequence_constant

def y : ℝ := x * sequence_constant

theorem x_plus_y : x + y = 80 := 
by
  -- Placeholder for the proof
  sorry

end x_plus_y_l408_408060


namespace problem_rewrite_expression_l408_408247

theorem problem_rewrite_expression (j : ℝ) : 
  ∃ (c p q : ℝ), (8 * j^2 - 6 * j + 20 = c * (j + p)^2 + q) ∧ (q / p = -77) :=
sorry

end problem_rewrite_expression_l408_408247


namespace family_members_l408_408383

variable (p : ℝ) (i : ℝ) (c : ℝ)

theorem family_members (h1 : p = 1.6) (h2 : i = 0.25) (h3 : c = 16) :
  (c / (2 * (p * (1 + i)))) = 4 := by
  sorry

end family_members_l408_408383


namespace sum_of_two_digit_divisors_of_154_with_remainder_10_is_180_l408_408581

theorem sum_of_two_digit_divisors_of_154_with_remainder_10_is_180:
  (∑ d in ({d : ℕ | d > 0 ∧ 154 % d = 10 ∧ 10 ≤ d ∧ d < 100} : Finset ℕ), d) = 180 :=
by
  sorry

end sum_of_two_digit_divisors_of_154_with_remainder_10_is_180_l408_408581


namespace y_coordinate_of_second_point_l408_408556

theorem y_coordinate_of_second_point
  (m n : ℝ)
  (h₁ : m = 2 * n + 3)
  (h₂ : m + 2 = 2 * (n + 1) + 3) :
  (n + 1) = n + 1 :=
by
  -- proof to be provided
  sorry

end y_coordinate_of_second_point_l408_408556


namespace sum_of_cubes_l408_408231

theorem sum_of_cubes (n : ℕ) (hn : n > 0) : 
  (∑ k in Finset.range (n + 1), k^3) = ((n * (n + 1) / 2) ^ 2) :=
sorry

end sum_of_cubes_l408_408231


namespace gcf_270_108_150_l408_408711

theorem gcf_270_108_150 : Nat.gcd (Nat.gcd 270 108) 150 = 30 := 
  sorry

end gcf_270_108_150_l408_408711


namespace general_formula_l408_408918

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n / (a n + 2)

theorem general_formula (a : ℕ → ℝ) (n : ℕ) (h : sequence a) : 
  n > 0 → a n = 2 / (n + 1) :=
begin
  sorry
end

end general_formula_l408_408918


namespace sequence_x_y_sum_l408_408048

theorem sequence_x_y_sum (r : ℝ) (x y : ℝ)
  (h₁ : r = 1 / 4)
  (h₂ : x = 256 * r)
  (h₃ : y = x * r) :
  x + y = 80 :=
by
  sorry

end sequence_x_y_sum_l408_408048


namespace integral_evaluation_l408_408011

open Real

def integrand (θ : ℝ) : ℝ := (2 * sin θ + 3 * cos θ - 3) / (13 * cos θ - 5)

theorem integral_evaluation : 
  ∫ (θ : ℝ) in 0..π, integrand θ = (3 * π / 13) - (4 / 13) * log (3 / 2) := 
by {
  sorry
}

end integral_evaluation_l408_408011


namespace abs_bx2_min_y2_plus_2axy_le_sqrt2_l408_408095

theorem abs_bx2_min_y2_plus_2axy_le_sqrt2 (x y a b : ℝ) (h1 : x^2 + y^2 ≤ 1) (h2 : a^2 + b^2 ≤ 2) :
  |b * (x^2 - y^2) + 2 * a * x * y| ≤ real.sqrt 2 := 
sorry

end abs_bx2_min_y2_plus_2axy_le_sqrt2_l408_408095


namespace molecular_weight_of_compound_l408_408318

theorem molecular_weight_of_compound (total_weight : ℕ) (number_of_moles : ℕ) 
  (H : total_weight = 525) (H1 : number_of_moles = 3) :
  total_weight / number_of_moles = 175 := 
by
  rw [H, H1]
  norm_num

end molecular_weight_of_compound_l408_408318


namespace magnitude_product_complex_l408_408415

theorem magnitude_product_complex :
  let z1 := Complex.mk 7 (-4)
  let z2 := Complex.mk 3 11
  Complex.abs (z1 * z2) = Real.sqrt 8450 :=
by
  sorry

end magnitude_product_complex_l408_408415


namespace find_m_l408_408135

theorem find_m :
  ∀ (m : ℝ),
  (∀ (A B : ℝ × ℝ), 
    (A.1 - 9 * A.2 - 8 = 0) ∧ (A.2 = A.1^3 - m * A.1^2 + 3 * A.1) ∧ 
    (B.1 - 9 * B.2 - 8 = 0) ∧ (B.2 = B.1^3 - m * B.1^2 + 3 * B.1) ∧ 
    ((3 * A.1^2 - 2 * m * A.1 + 3) = (3 * B.1^2 - 2 * m * B.1 + 3))) 
  → m = 4 ∨ m = -3 :=
begin
  sorry
end

end find_m_l408_408135


namespace number_of_solutions_given_angle_l408_408479

structure Point where
  x : ℝ
  y : ℝ

def isCentroid (A B S : Point) (ratio : ℝ) : Prop :=
  ∃ (m₁ m₂ : ℝ), m₁ / m₂ = ratio ∧ 
  ∀ (C : Point), (2 * distance A S = m₁ * 3) ∧ (2 * distance B S = m₂ * 3)

axiom known_angle : ℝ

theorem number_of_solutions_given_angle (A B : Point) (α : ℝ) (h : isCentroid A B (1/2)) :
  (1 : ℝ) ≤ α ∧ α < 90 →
  true := sorry

end number_of_solutions_given_angle_l408_408479


namespace obtain_2015_in_4_operations_obtain_2015_in_3_operations_l408_408695

-- Define what an operation is
def operation (cards : List ℕ) : List ℕ :=
  sorry  -- Implementation of this is unnecessary for the statement

-- Check if 2015 can be obtained in 4 operations
def can_obtain_2015_in_4_operations (initial_cards : List ℕ) : Prop :=
  ∃ cards, (operation^[4] initial_cards) = cards ∧ 2015 ∈ cards

-- Check if 2015 can be obtained in 3 operations
def can_obtain_2015_in_3_operations (initial_cards : List ℕ) : Prop :=
  ∃ cards, (operation^[3] initial_cards) = cards ∧ 2015 ∈ cards

theorem obtain_2015_in_4_operations :
  can_obtain_2015_in_4_operations [1, 2] :=
sorry

theorem obtain_2015_in_3_operations :
  can_obtain_2015_in_3_operations [1, 2] :=
sorry

end obtain_2015_in_4_operations_obtain_2015_in_3_operations_l408_408695


namespace quadrilateral_division_l408_408237

def has_circumcircle (Q : Quadrilateral) : Prop :=
  ∃ C, is_circumcircle C Q

def can_be_divided_into_n_with_circumcircle (Q : Quadrilateral) (n : ℕ) :=
  ∃ (Q₁ Q₂ ... Qn : list Quadrilateral), Q₁ ++ Q₂ ++ ... ++ Qn = Q ∧
  (∀ (Qi : Quadrilateral), Qi ∈ Q₁ ++ Q₂ ++ ... ++ Qn → has_circumcircle Qi)

theorem quadrilateral_division (Q : Quadrilateral) (n : ℕ) 
  (h_circumcircle : has_circumcircle Q) (h_n_ge_4 : n ≥ 4) :
  can_be_divided_into_n_with_circumcircle Q n :=
sorry

end quadrilateral_division_l408_408237


namespace max_value_of_symmetric_function_l408_408953

def f (x a b : ℝ) := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function 
  (a b : ℝ)
  (symmetric : ∀ t : ℝ, f (-2 + t) a b = f (-2 - t) a b) :
  ∃ M : ℝ, M = 16 ∧ ∀ x : ℝ, f x a b ≤ M :=
by
  use 16
  sorry

end max_value_of_symmetric_function_l408_408953


namespace x_plus_y_l408_408055

noncomputable def sequence_constant : ℝ := 1 / 4

def x : ℝ := 256 * sequence_constant

def y : ℝ := x * sequence_constant

theorem x_plus_y : x + y = 80 := 
by
  -- Placeholder for the proof
  sorry

end x_plus_y_l408_408055


namespace principal_amount_l408_408775

noncomputable def exponential (r t : ℝ) :=
  Real.exp (r * t)

theorem principal_amount (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) :
  A = 5673981 ∧ r = 0.1125 ∧ t = 7.5 ∧ P = 2438978.57 →
  P = A / exponential r t := 
by
  intros h
  sorry

end principal_amount_l408_408775


namespace problem_rewrite_expression_l408_408246

theorem problem_rewrite_expression (j : ℝ) : 
  ∃ (c p q : ℝ), (8 * j^2 - 6 * j + 20 = c * (j + p)^2 + q) ∧ (q / p = -77) :=
sorry

end problem_rewrite_expression_l408_408246


namespace translation_from_Lori_to_Alex_l408_408639

theorem translation_from_Lori_to_Alex :
  ∃ t : ℝ × ℝ, t = (-8, -7) ∧
    (6, 3) + t = (-2, -4) := 
by
  use (-8, -7)
  split
  { refl }
  { sorry }

end translation_from_Lori_to_Alex_l408_408639


namespace length_PT_30_l408_408182

noncomputable def length_PT (PQ QR : ℝ) (angle_QRT : ℝ) (T_on_RS : Prop) : ℝ := 
  if h : PQ = 30 ∧ QR = 15 ∧ angle_QRT = 75 then 30 else 0

theorem length_PT_30 (PQ QR : ℝ) (angle_QRT : ℝ) (T_on_RS : Prop) :
  PQ = 30 → QR = 15 → angle_QRT = 75 → length_PT PQ QR angle_QRT T_on_RS = 30 :=
sorry

end length_PT_30_l408_408182


namespace units_digit_of_sum_of_squares_of_first_2023_odd_integers_is_5_l408_408717

theorem units_digit_of_sum_of_squares_of_first_2023_odd_integers_is_5 :
  (sum (fun n : ℕ => ((2 * n + 1) ^ 2) % 10) (range 2023)) % 10 = 5 :=
sorry

end units_digit_of_sum_of_squares_of_first_2023_odd_integers_is_5_l408_408717


namespace probability_convex_quadrilateral_from_chords_l408_408627

theorem probability_convex_quadrilateral_from_chords :
  let total_chords := Nat.choose 7 2,
      total_ways := Nat.choose total_chords 4,
      favorable_outcomes := Nat.choose 7 4 in
  (favorable_outcomes : ℚ) / total_ways = 1 / 171 :=
by
  sorry

end probability_convex_quadrilateral_from_chords_l408_408627


namespace z_solutions_correct_l408_408452

noncomputable def z_solutions := 
  {z : ℂ | z ^ 6 = -8}

theorem z_solutions_correct : z_solutions = 
  {1 + complex.I * complex.sqrt (complex.cube 2), 
   1 - complex.I * complex.sqrt (complex.cube 2), 
   -1 + complex.I * complex.sqrt (complex.cube 2), 
   -1 - complex.I * complex.sqrt (complex.cube 2), 
   complex.I * complex.sqrt (complex.cube 2), 
   -complex.I * complex.sqrt (complex.cube 2)} :=
sorry

end z_solutions_correct_l408_408452


namespace students_speak_both_l408_408965

theorem students_speak_both (total E T N : ℕ) (h1 : total = 150) (h2 : E = 55) (h3 : T = 85) (h4 : N = 30) :
  E + T - (total - N) = 20 := by
  -- Main proof logic
  sorry

end students_speak_both_l408_408965


namespace sum_first_11_terms_l408_408486

variable {a_n : ℕ → ℝ} -- Arithmetic sequence
variable {S : ℕ → ℝ} -- Sum of the first n terms

-- Defining the conditions as Lean 4 statements
-- S_n is the sum of the first n terms of an arithmetic sequence {a_n}
axiom h1 : ∀ (n : ℕ), S n = ∑ i in finset.range n, a_n (i + 1)

-- Given condition: 2(a_1 + a_3 + a_5) + 3(a_8 + a_{10}) = 36
axiom h2 : 2 * (a_n 1 + a_n 3 + a_n 5) + 3 * (a_n 8 + a_n 10) = 36

-- Goal: To prove S_{11} = 33
theorem sum_first_11_terms : S 11 = 33 :=
by
  sorry

end sum_first_11_terms_l408_408486


namespace correct_conclusions_l408_408273

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := cos (ω * x + π/6)

theorem correct_conclusions (ω : ℝ) (hω : ω > 0) :
  (f ω (π/2) = -√3/2) ∧
  ((∃ p : ℝ, ∀ x, f ω (x + p) = f ω x) ∧ (p = π)) ∧
  (¬even_function (λ x, f ω (x - π/12))) ∧
  (∀ x, 0 < x ∧ x < π/(3 * ω) → f ω x < f ω (x-1)) :=
sorry

end correct_conclusions_l408_408273


namespace find_a_range_l408_408868

open Real

def f (a x : ℝ) : ℝ := x + a * log x

def holds_for_any_distinct_x1_x2_in_interval (f : ℝ → ℝ → ℝ) (a : ℝ) (P : ℝ → ℝ → Prop) :=
  ∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 ≤ 3 ∧ 1 ≤ x2 ∧ x2 ≤ 3 ∧ x1 ≠ x2 → P (f a x1) (f a x2)

theorem find_a_range :
  ∀ a : ℝ,
  (0 < a →
   holds_for_any_distinct_x1_x2_in_interval f a (λ f_x1 f_x2, |f_x1 - f_x2| < |(1 / x1) - (1 / x2)|)) →
  (0 < a ∧ a < 8 / 3) :=
begin
  intros a ha,
  sorry
end

end find_a_range_l408_408868


namespace triangle_BC_l408_408173

theorem triangle_BC (A B C: Point)
  (h_triangle : ∠ A B C = 90°)
  (h_cosB : cos (∠ B C A) = 4/5)
  (h_AB : dist A B = 40) :
  dist B C = 24 := 
sorry

end triangle_BC_l408_408173


namespace arithmetic_sequence_general_formula_geometric_sequence_sum_l408_408978

theorem arithmetic_sequence_general_formula (a : ℕ → ℕ) (h₁ : a 2 = 3) (h₂ : a 4 = 7) :
  ∀ n, a n = 2 * n - 1 :=
sorry

theorem geometric_sequence_sum (a b : ℕ → ℕ) (h₁ : ∀ n, b n - a n = 2^n)
  (h₂ : b 1 = 3) (h₃ : ∀ n, a n = 2 * n - 1) :
  ∀ n, (∑ k in Finset.range n, b (k + 1)) = n ^ 2 - 2 + 2 ^ (n + 1) :=
sorry

end arithmetic_sequence_general_formula_geometric_sequence_sum_l408_408978


namespace smallest_angle_proof_l408_408178

noncomputable def smallest_angle_in_triangle : ℝ :=
  let angle1 := 135
  let angle_supplementary := 180 - angle1
  let triangle_angles := (60, angle_supplementary)
  180 - (triangle_angles.1 + triangle_angles.2)

theorem smallest_angle_proof : smallest_angle_in_triangle = 45 := 
by 
  let angle1 := 135
  let angle_supplementary := 180 - angle1
  let triangle_angles := (60, angle_supplementary)
  let x := 180 - (triangle_angles.1 + triangle_angles.2)
  have conclusion : x = 45 := by
    simp [triangle_angles]
    calc
            180 - (60 + 45) = 180 - 105 : by simp
            ... = 75 : by simp
  sorry

end smallest_angle_proof_l408_408178


namespace relationship_abc_l408_408088

theorem relationship_abc (a b c : ℕ) (ha : a = 2^555) (hb : b = 3^444) (hc : c = 6^222) : a < c ∧ c < b := by
  sorry

end relationship_abc_l408_408088


namespace classmates_problem_l408_408786

theorem classmates_problem (students : Fin 60 → Fin 60) 
  (h : ∀ (t : Finset (Fin 60)), t.card = 10 → ∃ x ∈ t, ∃ y ∈ t, ∃ z ∈ t, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ students x = students y ∧ students y = students z) : 
  ∃ t : Finset (Fin 60), t.card ≥ 15 ∧ ∀ x y ∈ t, students x = students y :=
sorry

end classmates_problem_l408_408786


namespace max_value_of_symmetric_f_l408_408950

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_f :
  ∀ (a b : ℝ),
    (f 1 a b = 0) →
    (f (-1) a b = 0) →
    (f (-5) a b = 0) →
    (f (-3) a b = 0) →
    (∃ x : ℝ, f x 8 15 = 16) :=
by
  sorry

end max_value_of_symmetric_f_l408_408950


namespace sqrt_expression_is_80_l408_408313

theorem sqrt_expression_is_80 :
  real.cbrt 125 * real.root 256 4 * real.sqrt 16 = 80 := by
  sorry

end sqrt_expression_is_80_l408_408313


namespace total_buildings_proof_l408_408194

-- Given conditions
variables (stores_pittsburgh hospitals_pittsburgh schools_pittsburgh police_stations_pittsburgh : ℕ)
variables (stores_new hospitals_new schools_new police_stations_new buildings_new : ℕ)

-- Given values for Pittsburgh
def stores_pittsburgh := 2000
def hospitals_pittsburgh := 500
def schools_pittsburgh := 200
def police_stations_pittsburgh := 20

-- Definitions for the new city
def stores_new := stores_pittsburgh / 2
def hospitals_new := 2 * hospitals_pittsburgh
def schools_new := schools_pittsburgh - 50
def police_stations_new := police_stations_pittsburgh + 5
def buildings_new := stores_new + hospitals_new + schools_new + police_stations_new

-- Statement to prove
theorem total_buildings_proof : buildings_new = 2175 := by
  dsimp [buildings_new, stores_new, hospitals_new, schools_new, police_stations_new] 
  dsimp [stores_pittsburgh, hospitals_pittsburgh, schools_pittsburgh, police_stations_pittsburgh]
  rfl

end total_buildings_proof_l408_408194


namespace even_number_difference_l408_408642

theorem even_number_difference :
  let digits := [1, 2, 5, 6, 9] in
  let largest_even_num := 96512 in
  let smallest_even_num := 12596 in
  (largest_even_num - smallest_even_num) = 83916 :=
by
  -- Definitions from conditions
  let digits := [1, 2, 5, 6, 9]
  let largest_even_num := 96512
  let smallest_even_num := 12596
  -- Inject the answer directly:
  have diff := 83916
  exact Nat.sub_eq_iff_eq_add.mpr (by simp [largest_even_num, smallest_even_num, diff])
  sorry

end even_number_difference_l408_408642


namespace probability_difference_one_l408_408793

theorem probability_difference_one :
  let n := 6
  let pairs := { (a, b) | a ∈ (Finset.range n) ∧ b ∈ (Finset.range n) ∧ a < b }
  let favorable_pairs := { (a, b) | a ∈ (Finset.range n) ∧ b ∈ (Finset.range n) ∧ a < b ∧ abs (a - b) = 1 }
  let total_pairs := Finset.card pairs
  let favorable := Finset.card favorable_pairs
  let probability := favorable.to_rat / total_pairs.to_rat
  probability = 1 / 3 :=
by 
  let n := 6
  let pairs := { (a, b) | a ∈ (Finset.range n) ∧ b ∈ (Finset.range n) ∧ a < b }
  let favorable_pairs := { (a, b) | a ∈ (Finset.range n) ∧ b ∈ (Finset.range n) ∧ a < b ∧ abs (a - b) = 1 }
  let total_pairs := Finset.card pairs
  let favorable := Finset.card favorable_pairs
  let probability := favorable.to_rat / total_pairs.to_rat
  have h_total : total_pairs = 15, from by sorry,
  have h_favorable : favorable = 5, from by sorry,
  have h_prob : probability = 1 / 3, from by sorry,
  exact h_prob

end probability_difference_one_l408_408793


namespace smallest_part_proportion_l408_408154

theorem smallest_part_proportion (x : ℕ) (h1 : 3 * x + 5 * x + 7 * x = 90) :
  3 * x = 18 :=
begin
  sorry
end

end smallest_part_proportion_l408_408154


namespace range_of_positive_integers_l408_408604

theorem range_of_positive_integers 
  (n : ℤ) 
  (h1 : List D = List.range' (-10) 20) 
  (h2 : ∀ x ∈ D, x >= -10 ∧ x < 10) : 
  (List.maximum (List.filter (λ x, x > 0) D) - List.minimum (List.filter (λ x, x > 0) D)) = 8 :=
sorry

end range_of_positive_integers_l408_408604


namespace equation_of_circle_equation_of_line_through_B_l408_408472

-- Step 1: Define the basic conditions
def pointA : (ℝ × ℝ) := (-1, 2)
def tangentLine (x y : ℝ) : Prop := x + 2*y + 7 = 0
def pointB : (ℝ × ℝ) := (-2, 0)

-- Step 2: Define the circle theorem and required radius calculation
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 20

-- Theorem: The equation of circle A is (x+1)^2 + (y-2)^2 = 20
theorem equation_of_circle : 
  ∃ r : ℝ, r = abs ((-1) + 2 * 2 + 7) / real.sqrt 5 ∧ 
  (∀ x y, tangentLine x y → dist (x, y) pointA = r) ∧ 
  (circle_eq = λ x y, dist pointA (x, y) = r) := 
sorry

-- Step 3: Define the line theorem with conditions given midpoint Q and chord length |MN|
def line_eq1 (x y : ℝ) : Prop := 3*x - 4*y + 6 = 0
def line_eq2 (x : ℝ) : Prop := x = -2

-- Theorem: The equation of line l is 3x - 4y + 6 = 0 or x = -2, given the properties
theorem equation_of_line_through_B : 
  ∃ (|MN| : ℝ), |MN| = 2 * real.sqrt 19 ∧ 
  (∀ l, (line_eq1 = l ∨ line_eq2 = l) → 
  (∀ M N, dist M N = |MN| → midpoint M N = pointB)) := 
sorry

end equation_of_circle_equation_of_line_through_B_l408_408472


namespace probability_two_cards_sum_to_twelve_l408_408302

theorem probability_two_cards_sum_to_twelve :
  let total_cards := 52
  let num_specific_cards := 32
  let total_combinations := num_specific_cards * 4 / (total_cards * (total_cards - 1))
  let num_six := 4
  let six_combinations := num_six * (num_six - 1) / (total_cards * (total_cards - 1))
  prob := total_combinations / (total_combinations + six_combinations)
  prob = 35/663 := by sorry

end probability_two_cards_sum_to_twelve_l408_408302


namespace problem_l408_408131

def f (x a b : ℝ) : ℝ := a * x ^ 3 - b * x + 1

theorem problem (a b : ℝ) (h : f 2 a b = -1) : f (-2) a b = 3 :=
by {
  sorry
}

end problem_l408_408131


namespace product_of_p_r_s_l408_408933

-- Definition of conditions
def eq1 (p : ℕ) : Prop := 4^p + 4^3 = 320
def eq2 (r : ℕ) : Prop := 3^r + 27 = 108
def eq3 (s : ℕ) : Prop := 2^s + 7^4 = 2617

-- Main statement
theorem product_of_p_r_s (p r s : ℕ) (h1 : eq1 p) (h2 : eq2 r) (h3 : eq3 s) : p * r * s = 112 :=
by sorry

end product_of_p_r_s_l408_408933


namespace x_plus_y_l408_408057

noncomputable def sequence_constant : ℝ := 1 / 4

def x : ℝ := 256 * sequence_constant

def y : ℝ := x * sequence_constant

theorem x_plus_y : x + y = 80 := 
by
  -- Placeholder for the proof
  sorry

end x_plus_y_l408_408057


namespace sequence_sum_l408_408036

theorem sequence_sum :
  ∃ (r : ℝ) (x y : ℝ),
    (r = 1 / 4) ∧
    (x = 256 * r) ∧
    (y = x * r) ∧
    (x + y = 80) :=
by
  use 1 / 4, 64, 16
  split
  . exact rfl
  split
  . exact rfl
  split
  . exact rfl
  . norm_num


end sequence_sum_l408_408036


namespace monotonic_increasing_interval_l408_408285

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem monotonic_increasing_interval : ∀ x, -1 < x → 0 < f' x := 
by
  sorry

end monotonic_increasing_interval_l408_408285


namespace largest_in_arithmetic_progression_l408_408252

theorem largest_in_arithmetic_progression 
  (a d : ℝ)
  (h1: a^3 + (a + d)^3 + (a + 2*d)^3 + (a + 3*d)^3 + (a + 4*d)^3 + (a + 5*d)^3 + (a + 6*d)^3 = 0)
  (h2: a^2 + (a + d)^2 + (a + 2*d)^2 + (a + 3*d)^2 + (a + 4*d)^2 + (a + 5*d)^2 + (a + 6*d)^2 = -224) :
  max a (max (a + d) (max (a + 2*d) (max (a + 3*d) (max (a + 4*d) (max (a + 5*d) (a + 6*d)))))) = 6 * sqrt 2 :=
sorry

end largest_in_arithmetic_progression_l408_408252


namespace remainder_expr_div_by_5_l408_408934

theorem remainder_expr_div_by_5 (n : ℤ) : 
  (7 - 2 * n + (n + 5)) % 5 = (-n + 2) % 5 := 
sorry

end remainder_expr_div_by_5_l408_408934


namespace cubic_eq_three_natural_roots_l408_408837

noncomputable def three_natural_roots (x1 x2 x3 p : ℝ) : Prop :=
  ((5 * x1 ^ 3 - 5 * (p + 1) * x1 ^ 2 + (71 * p - 1) * x1 + 1 = 66 * p) ∧
   (5 * x2 ^ 3 - 5 * (p + 1) * x2 ^ 2 + (71 * p - 1) * x2 + 1 = 66 * p) ∧
   (5 * x3 ^ 3 - 5 * (p + 1) * x3 ^ 2 + (71 * p - 1) * x3 + 1 = 66 * p)) ∧
  (x1 ∈ ℕ) ∧ (x2 ∈ ℕ) ∧ (x3 ∈ ℕ)

theorem cubic_eq_three_natural_roots :
  ∃ (x1 x2 x3 : ℝ), three_natural_roots x1 x2 x3 76 :=
begin
  -- execute the proof here
  sorry
end

end cubic_eq_three_natural_roots_l408_408837


namespace sequence_sum_l408_408035

theorem sequence_sum :
  ∃ (r : ℝ) (x y : ℝ),
    (r = 1 / 4) ∧
    (x = 256 * r) ∧
    (y = x * r) ∧
    (x + y = 80) :=
by
  use 1 / 4, 64, 16
  split
  . exact rfl
  split
  . exact rfl
  split
  . exact rfl
  . norm_num


end sequence_sum_l408_408035


namespace general_equation_of_C2_range_of_values_on_C2_l408_408184

-- Parametric equations for curve C1
def C1 (α : ℝ) : ℝ × ℝ := 
  (3 / 2 * Real.cos α, Real.sin α)

-- Definition of point P given point M on curve C1
def M_to_P (M : ℝ × ℝ) : ℝ × ℝ := 
  (2 * M.1, 2 * M.2)

-- General equation of the curve C2
def C2_equation (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 4 = 1

-- Prove the general equation of C2
theorem general_equation_of_C2 : ∀ α : ℝ, 
  let P := M_to_P (C1 (α / 2)) in 
  C2_equation P.1 P.2 := 
sorry

-- Range of values for x + 2y on curve C2
def range_of_values (x y : ℝ) : Prop :=
  -5 ≤ x + 2 * y ∧ x + 2 * y ≤ 5

-- Prove the range of values for x+2y
theorem range_of_values_on_C2 (x y : ℝ) (h : C2_equation x y) :
  range_of_values x y :=
sorry

end general_equation_of_C2_range_of_values_on_C2_l408_408184


namespace lines_parallel_l408_408494

noncomputable def f (x : ℝ) (k : ℝ) (b : ℝ) : ℝ := k * x + b

theorem lines_parallel (k b x₀ y₀ : ℝ) :
  ∀ x : ℝ,
  (f x k b = f x₀ k b → False) →
  (∀ y : ℝ, ∃ y₁ : ℝ, y₁ = f x k b ∧ y = f x₀ k b → False) →
  ∀ x₁ : ℝ,
  (∃ y₁ : ℝ, y₁ = f x₁ k b → False) →
  ∀ x₁ : ℝ,
  (y₁ = y₀ + k * (x₁ - x₀) → f x₁ k b = k * x₁ + b) →
  ∀ y : ℝ, 
  ∀ y₁ : ℝ, 
  (y₁ = y₀ + k * (x - x₀) → lines_are_parallel y y₁)
  sorry

end lines_parallel_l408_408494


namespace total_games_played_l408_408199

-- Define the conditions as parameters
def ratio_games_won_lost (W L : ℕ) : Prop := W / 2 = L / 3

-- Let's state the problem formally in Lean
theorem total_games_played (W L : ℕ) (h1 : ratio_games_won_lost W L) (h2 : W = 18) : W + L = 30 :=
by 
  sorry  -- The proof will be filled in


end total_games_played_l408_408199


namespace market_size_scientific_notation_l408_408265

-- Define the conversion from billion to numerical value
def billion_to_num := 10^9

-- Define the given market size in billions
def market_size_billion := 677.1

-- Calculate the market size in numerical value
def market_size_num := market_size_billion * billion_to_num

-- Define the scientific notation result for the market size
def market_size_sci := 6.771 * 10^11

-- Prove that converting the market size to scientific notation yields the correct result
theorem market_size_scientific_notation : market_size_num = market_size_sci :=
by
  sorry

end market_size_scientific_notation_l408_408265


namespace fill_space_with_tetrahedra_and_octahedra_l408_408254

-- Definitions for the crystalline structure
def crystalline_structure (space : Set ℝ^3) (unit_cells : Set (Set ℝ^3)) :=
  (∀ unit ∈ unit_cells, ∃ transformations : ℕ → Set ℝ^3, unit = transformations 0) ∧
  (∀ point ∈ space, ∃ unit ∈ unit_cells, point ∈ unit)

-- A periodic unit cell that can be divided into two tetrahedra and one octahedron
def periodic_unit_cell (unit : Set ℝ^3) :=
  ∃ tetrahedra octahedron : Set (Set ℝ^3), (unit = tetrahedra ∪ octahedron) ∧
  ∀ t ∈ tetrahedra, ∃ regular_tetrahedron : Set ℝ^3, t = regular_tetrahedron ∧ is_regular_tetrahedron regular_tetrahedron ∧
  ∀ o ∈ octahedron, is_regular_octahedron o

-- To prove that any space filled by a crystalline structure periodically can be filled with regular tetrahedra and octahedra
theorem fill_space_with_tetrahedra_and_octahedra (space : Set ℝ^3) :
  ∃ unit_cells : Set (Set ℝ^3), crystalline_structure space unit_cells →
  ∀ unit ∈ unit_cells, periodic_unit_cell unit := sorry

end fill_space_with_tetrahedra_and_octahedra_l408_408254


namespace total_bees_in_hive_l408_408542

theorem total_bees_in_hive (worker_bees drones_ratio additional_drones : ℕ)
    (h_worker_bees : worker_bees = 128)
    (h_drones_ratio : drones_ratio = 16)
    (h_additional_drones : additional_drones = 8) : 
    worker_bees + (worker_bees / drones_ratio + additional_drones) = 144 := 
by 
  have initial_drones : ℕ := worker_bees / drones_ratio
  have total_drones : ℕ := initial_drones + additional_drones
  calc 
    worker_bees + total_drones = worker_bees + (worker_bees / drones_ratio + additional_drones) : by rfl
    ... = 128 + (128 / 16 + 8) : by rw [h_worker_bees, h_drones_ratio, h_additional_drones]
    ... = 128 + (8 + 8) : by norm_num
    ... = 128 + 16 : by norm_num
    ... = 144 : by norm_num

end total_bees_in_hive_l408_408542


namespace total_legs_in_christophers_room_l408_408808

def total_legs (num_spiders num_legs_per_spider num_ants num_butterflies num_beetles num_legs_per_insect : ℕ) : ℕ :=
  let spider_legs := num_spiders * num_legs_per_spider
  let ant_legs := num_ants * num_legs_per_insect
  let butterfly_legs := num_butterflies * num_legs_per_insect
  let beetle_legs := num_beetles * num_legs_per_insect
  spider_legs + ant_legs + butterfly_legs + beetle_legs

theorem total_legs_in_christophers_room : total_legs 12 8 10 5 5 6 = 216 := by
  -- Calculation and reasoning omitted
  sorry

end total_legs_in_christophers_room_l408_408808


namespace sequence_sum_S2018_l408_408478

noncomputable def a : ℕ → ℝ 
| 0       := 0  -- unused term a_0 for natural numbers index
| 1       := 4 / 5
| (n + 1) := if h : a n ≤ 1 / 2 then 2 * a n else 2 * a n - 1

def S (n : ℕ) : ℝ :=
(nat.sum_range n (λ k, a (k + 1))).to_real

theorem sequence_sum_S2018 :
  S 2018 = 5047 / 5 := 
sorry

end sequence_sum_S2018_l408_408478


namespace next_perfect_square_after_1818_l408_408689

def is_next_perfect_square_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (∃ m : ℕ, n = 1800 + m ∧ m < 100 ∧ (∃ k : ℕ, 18 * m = k^2))

theorem next_perfect_square_after_1818 : ∃ n : ℕ, is_next_perfect_square_number n ∧ n > 1818 ∧ n = 1832 :=
by
  existsi 1832
  unfold is_next_perfect_square_number
  split
  . apply and.intro
    { linarith }
    { split
      . existsi 32
        split
        . linarith
        . existsi 18
          linarith
      . sorry }

end next_perfect_square_after_1818_l408_408689


namespace vector_magnitude_5_l408_408508

noncomputable def vector_magnitude {ι} [InnerProductSpace ℝ (EuclideanSpace _ _)] 
(a b : EuclideanSpace ℝ (Fin 2)) : ℝ :=
(norm (a + b))

theorem vector_magnitude_5 
  (a b : EuclideanSpace ℝ (Fin 2))
  (h₁ : a = ⟨ λ i, if i = 0 then 1/2 else if i = 1 then (√3)/2 else 0 ⟩)
  (h₂ : ⟦(Fin 2 × ℝ)⟧ (b : EuclideanSpace ℝ (Fin 2)) = 2 * √5)
  (h₃ : dot_product a (a + b) = 3) :
  vector_magnitude a b = 5 :=
sorry

end vector_magnitude_5_l408_408508


namespace triangle_inequality_third_side_l408_408971

theorem triangle_inequality_third_side (a b x : ℝ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : 0 < x) (h₄ : x < a + b) (h₅ : a < b + x) (h₆ : b < a + x) :
  ¬(x = 9) := by
  sorry

end triangle_inequality_third_side_l408_408971


namespace value_of_n_l408_408896

theorem value_of_n (n : ℝ) : (∀ (x y : ℝ), x^2 + y^2 - 2 * n * x + 2 * n * y + 2 * n^2 - 8 = 0 → (x + 1)^2 + (y - 1)^2 = 2) → n = 1 :=
by
  sorry

end value_of_n_l408_408896


namespace expected_points_l408_408797

theorem expected_points (r : ℝ) (diameter : ℝ) (ceil : ℝ → ℝ → ℝ) :
  diameter = 10 →
  r = 5 →
  (∀ x, 0 ≤ x → x ≤ r → 
        ∃ k : ℤ, ceil k x = 5 - x ∧ 
        (0 ≤ x ∧ x < 1 → k = 5) ∧ 
        (1 ≤ x ∧ x < 2 → k = 4) ∧ 
        (2 ≤ x ∧ x < 3 → k = 3) ∧ 
        (3 ≤ x ∧ x < 4 → k = 2) ∧ 
        (4 ≤ x ∧ x ≤ 5 → k = 1)
  ) →
  ∀ (E : ℝ), E = ∑ k in finset.range 5, (k + 1) * (2 * k + 1) / 25 → 
    E = 11 / 5 :=
by
  intros
  sorry

end expected_points_l408_408797


namespace next_four_digit_number_l408_408669

def isPerfectSquare (n : ℕ) : Prop :=
∃ m : ℕ, m * m = n

def satisfiesProperty (n : ℕ) : Prop :=
  ∃ a b : ℕ, 10 ≤ a ∧ a < 100 ∧ 0 ≤ b ∧ b < 10 ∧ n = 1800 + a * 10 + b ∧ isPerfectSquare (18 * a)

theorem next_four_digit_number (n : ℕ) (h₀ : n = 1818) (h₁ : satisfiesProperty n) :
  ∃ m : ℕ, m > n ∧ m < 2000 ∧ satisfiesProperty m ∧ (∀ k : ℕ, n < k ∧ k < m → ¬ satisfiesProperty k) :=
begin
  use 1832,
  split,
  { exact 1832 > 1818, },
  split,
  { exact 1832 < 2000, },
  split,
  { unfold satisfiesProperty,
    use 32,
    use 0,
    split, { exact nat.le_of_lt 32 10 100, },
    split, { exact nat.lt_of_le_of_lt 0 32 40, },
    split, { exact nat.le_of_lt 0 0 10, },
    split, { exact nat.lt_of_le_of_lt 0 0 10, },
    split, { refl, },
    { unfold isPerfectSquare,
      use 24,
      exact nat.mul_self_eq 18 32 32, }, },
  { intros k hk,
    unfold satisfiesProperty at hk,
    cases hk with a ha,
    cases ha with b hb,
    cases hb with hb1 hb2,
    cases hb2 with hb3 hb4,
    cases hb4 with hb5 hb6,
    cases hb6 with hb7 hb8,
    cases hb8 with hb9 hb10,
    cases hb10 with hb11 hb12,
    exact nat.ne_of_lt_decides hk hb2 hb8 24, },
end

end next_four_digit_number_l408_408669


namespace smallest_n_condition_l408_408396

def pow_mod (a b m : ℕ) : ℕ := a^(b % m)

def n (r s : ℕ) : ℕ := 2^r - 16^s

def r_condition (r : ℕ) : Prop := ∃ k : ℕ, r = 3 * k + 1

def s_condition (s : ℕ) : Prop := ∃ h : ℕ, s = 3 * h + 2

theorem smallest_n_condition (r s : ℕ) (hr : r_condition r) (hs : s_condition s) :
  (n r s) % 7 = 5 → (n r s) = 768 := sorry

end smallest_n_condition_l408_408396


namespace rectangle_ABCD_BP_CP_tan_angle_APD_l408_408552

theorem rectangle_ABCD_BP_CP_tan_angle_APD (ABCD : Type) (A B C D P : ABCD)
  (h1 : BP = 12) (h2 : CP = 6) (h3 : tan (∠ APD) = 2) : AB = 15 := by 
  sorry

end rectangle_ABCD_BP_CP_tan_angle_APD_l408_408552


namespace three_digit_numbers_count_l408_408296

theorem three_digit_numbers_count :
  let cards := [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
  ∃ (k : ℕ), k = 432 ∧ (#(set_of (λ (s : multiset ℕ), s.card = 3 ∧
    ∃ (c1 c2 c3 : (ℕ × ℕ)), 
    c1 ∈ cards ∧ c2 ∈ cards ∧ c3 ∈ cards ∧
    c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
    s = {fst c1, snd c1, fst c2, snd c2, fst c3, snd c3}
    )) : ℕ) = k :=
by
  let cards := [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
  sorry

end three_digit_numbers_count_l408_408296


namespace num_solutions_l408_408530

theorem num_solutions : 
  {n : ℤ | -9 ≤ n ∧ n ≤ 13 ∧ (n - 1) * (n + 5) * (n + 11) < 0}.toFinset.card = 5 := 
  sorry

end num_solutions_l408_408530


namespace area_of_triangle_formed_by_lines_l408_408709

theorem area_of_triangle_formed_by_lines :
  let L1 := λ x : ℝ, 3*x - 6
  let L2 := λ x : ℝ, -4*x + 24
  let intersection_xy : ℝ × ℝ := (30/7, 48/7)
  let intercept_y1 : ℝ × ℝ := (0, -6)
  let intercept_y2 : ℝ × ℝ := (0, 24)
  let area_triangle : ℝ := (1 / 2) * 30 * (30 / 7)
  L1 (30/7) = 48/7 ∧ L2 (30/7) = 48/7 →
  area_triangle = 450 / 7 := 
  by
    intros
    sorry

end area_of_triangle_formed_by_lines_l408_408709


namespace x_plus_y_l408_408056

noncomputable def sequence_constant : ℝ := 1 / 4

def x : ℝ := 256 * sequence_constant

def y : ℝ := x * sequence_constant

theorem x_plus_y : x + y = 80 := 
by
  -- Placeholder for the proof
  sorry

end x_plus_y_l408_408056


namespace ratio_semi_to_triangle_area_l408_408923

-- Definitions corresponding to the conditions
variables (L M N : Point)
variable (h_line : ℝ → ℝ → Prop)
variable (h_LN_neq_MN : ¬ (dist L N = dist M N))
variables (A B C : Point)
variables (center_A : CenterCircle L M A)
variables (center_B : CenterCircle M N B)
variables (center_C : CenterCircle L N C)

-- State the theorem
theorem ratio_semi_to_triangle_area 
  (hM_between_LN : M ≠ L ∧ M ≠ N ∧ N ≠ L)
  (h_ratio : ∀ r R : ℝ, 
    let S := (π / 2) * r^2 + (π / 2) * R^2 + (π / 2) * (r + R)^2 in
    let S_triangle := (r + R) * (2 * R + r) - (1 / 2) * ((R + r) * (R - r) + R * (2 * r + R) + r * (2 * R + r)) in
    (S / S_triangle) = π) :
  true :=
by skip

end ratio_semi_to_triangle_area_l408_408923


namespace sequence_sum_l408_408392

theorem sequence_sum :
  ((∑ i in finset.Ico 2001 2094, i) - (∑ i in finset.Ico 201 294, i) - (∑ i in finset.Ico 1 94, i)) = 165044 :=
by
  -- sorry for now
  sorry

end sequence_sum_l408_408392


namespace proof_problem_l408_408803

-- Define the conditions
def a : ℤ := -3
def b : ℤ := -4
def cond1 := a^4 = 81
def cond2 := b^3 = -64

-- Define the goal in terms of the conditions
theorem proof_problem : a^4 + b^3 = 17 :=
by
  have h1 : a^4 = 81 := sorry
  have h2 : b^3 = -64 := sorry
  rw [h1, h2]
  norm_num

end proof_problem_l408_408803


namespace ratio_BD_BC1_l408_408981

theorem ratio_BD_BC1 {A B C A1 B1 C1 D : ℝ} 
  (h1 : ∀ {x y z : ℝ}, AA1C1C_square (4 : ℝ))
  (h2 : ∀ {u v : ℝ}, plane_ABC_perp_plane_AA1C1C)
  (h3 : dist A B = 3)
  (h4 : dist B C = 5)
  (h5 : collinear_points_on_BC1 D)
  (h6 : perp AD A1B) :
  BD / BC1 = 9/25 := 
sorry

end ratio_BD_BC1_l408_408981


namespace magnitude_of_vector_sum_l408_408517

open Real

noncomputable def vector_a := (2 : ℝ, 0 : ℝ)
noncomputable def vector_b : ℝ × ℝ := (cos (π / 3), sin (π / 3))  -- since |b| = 1 and angle is 60°

theorem magnitude_of_vector_sum : ‖vector_a + 2 • vector_b‖ = 2 * sqrt 3 := by
  sorry

end magnitude_of_vector_sum_l408_408517


namespace range_of_independent_variable_l408_408291

theorem range_of_independent_variable (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 2) / 2) → (x ≥ 2) := by
  sorry

end range_of_independent_variable_l408_408291


namespace volume_of_hexagonal_pyramid_l408_408358

noncomputable def hexagonal_pyramid_surface_area : ℝ := 720

noncomputable def triangular_face_area (A_hex : ℝ) : ℝ := (1 / 3) * A_hex

noncomputable def total_surface_area (A_hex : ℝ) : ℝ := A_hex + 6 * triangular_face_area A_hex

axiom hexagonal_base_area : ∃ A_hex : ℝ, total_surface_area A_hex = hexagonal_pyramid_surface_area

noncomputable def volume_pyramid (A_hex : ℝ) (H : ℝ) : ℝ := (1 / 3) * A_hex * H

theorem volume_of_hexagonal_pyramid : ∃ (A_hex H V : ℝ), 
  hexagonal_base_area ∧ 
  V = volume_pyramid A_hex H :=
  by
    sorry

end volume_of_hexagonal_pyramid_l408_408358


namespace range_of_mn_l408_408959

/-- In ΔABC, ∠C = 45°, and O is the circumcenter of ΔABC.
If OC = m*OA + n*OB (m, n ∈ ℝ), then the range of values for m+n is [-√2, 1]. -/
theorem range_of_mn (O A B C : ℝ) (m n : ℝ) (hC : ∠C = 45) (hO : is_circumcenter O A B C)
  (hOC: OC = m * OA + n * OB) :
  -Real.sqrt 2 ≤ m + n ∧ m + n ≤ 1 :=
sorry

end range_of_mn_l408_408959


namespace vectors_perpendicular_to_unit_l408_408142

def vector_perpendicular_unit_vectors (a b : ℝ × ℝ × ℝ) : List (ℝ × ℝ × ℝ) :=
  let n1 := ((-Real.sqrt 3 / 3), (Real.sqrt 3 / 3), (-Real.sqrt 3 / 3))
  let n2 := (Real.sqrt 3 / 3, (-Real.sqrt 3 / 3), Real.sqrt 3 / 3)
  [n1, n2]

theorem vectors_perpendicular_to_unit {a b : ℝ × ℝ × ℝ} :
  a = (2, 1, -1) →
  b = (-2, 1, 3) →
  (h : vector_perpendicular_unit_vectors a b = [(((-Real.sqrt 3 / 3)), ((Real.sqrt 3 / 3)), ((-Real.sqrt 3 / 3))),
                ((Real.sqrt 3 / 3), ((-Real.sqrt 3 / 3)), ((Real.sqrt 3 / 3)))]) := sorry

end vectors_perpendicular_to_unit_l408_408142


namespace find_y_l408_408455

theorem find_y (y : ℝ) (h : (y + 10 + (5 * y) + 4 + (3 * y) + 12) / 3 = 6 * y - 8) :
  y = 50 / 9 := by
  sorry

end find_y_l408_408455


namespace smallest_positive_period_l408_408871

-- Define the function f on Real numbers
variable (f : ℝ → ℝ)

-- Define the conditions
def condition1 : Prop := ∀ x, f (2 + x) = -f (2 - x)
def condition2 : Prop := ∀ x, f (4 + x) = f (4 - x)

-- State the theorem
theorem smallest_positive_period (h₁ : condition1 f) (h₂ : condition2 f) : ∃ p > 0, (∀ x, f (x + p) = f x) ∧ ∀ q, (q > 0 → q < p → ¬ ∀ x, f (x + q) = f x) :=
by
  use 8
  split
  { exact zero_lt_eight }
  split
  { sorry }
  { sorry }

end smallest_positive_period_l408_408871


namespace collinear_K_T_P_l408_408794

variables {Γ Γ₁ : Type*} [circle Γ] [circle Γ₁]
variables {A B C D K P N M E F E1 E2 F1 F2 T : Type*}
variables {AB CD : line} {AD BC : line}
variables (tangent_to_chords : Γ₁.tangent_to AB N)
variables (tangent_to_chords_2 : Γ₁.tangent_to CD M)
variables (tangent_to_circle : Γ₁.tangent_to Γ P)
variables (ext_common_tangent_1 : Γ.ext_common_tangent_to Γ₁ AD E)
variables (ext_common_tangent_2 : Γ.ext_common_tangent_to Γ₁ BC F)
variables (KE_intersect : line_through K E)
variables (KF_intersect: line_through K F)
variables (E1E2_intersect: circle_intersect KE_intersect Γ₁ E1 E2)
variables (F1F2_intersect: circle_intersect KF_intersect Γ₁ F1 F2)
variables (T_intersect : intersection E2 F1 F2 E1 T)

theorem collinear_K_T_P :
  collinear K T P :=
by
  sorry

end collinear_K_T_P_l408_408794


namespace find_p_l408_408533

theorem find_p (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : p = 52 / 11 :=
by {
  -- Proof steps would go here
  sorry
}

end find_p_l408_408533


namespace equilateral_triangle_of_complex_vertices_l408_408578

theorem equilateral_triangle_of_complex_vertices
  (ω : ℂ) (ω_cubed_eq_one : ω^3 = 1)
  (z1 z2 z3 : ℂ) (h : z1 + ω * z2 + (ω^2) * z3 = 0) :
  (abs (z1 - z2) = abs (z2 - z3)) ∧ (abs (z2 - z3) = abs (z3 - z1)) :=
sorry

end equilateral_triangle_of_complex_vertices_l408_408578


namespace beakers_with_copper_l408_408760

theorem beakers_with_copper :
  ∀ (total_beakers no_copper_beakers beakers_with_copper drops_per_beaker total_drops_used : ℕ),
    total_beakers = 22 →
    no_copper_beakers = 7 →
    drops_per_beaker = 3 →
    total_drops_used = 45 →
    total_drops_used = drops_per_beaker * beakers_with_copper →
    total_beakers = beakers_with_copper + no_copper_beakers →
    beakers_with_copper = 15 := 
-- inserting the placeholder proof 'sorry'
sorry

end beakers_with_copper_l408_408760


namespace simple_sampling_methods_l408_408655

theorem simple_sampling_methods :
  methods_of_implementing_simple_sampling = ["lottery method", "random number table method"] :=
sorry

end simple_sampling_methods_l408_408655


namespace john_bonus_last_year_l408_408567

theorem john_bonus_last_year :
  ∃ B : ℝ, (0 < B) ∧ (200_000 * B = 20_000) ∧ (100_000 * B = 10_000) :=
by
  use 0.1
  split
  { norm_num }
  split
  { norm_num }
  { norm_num }

end john_bonus_last_year_l408_408567


namespace gcd_of_repeated_three_digit_l408_408363

theorem gcd_of_repeated_three_digit : 
  ∀ (n : ℕ), 100 ≤ n ∧ n < 1000 → ∀ m ∈ {k : ℕ | ∃ n, 100 ≤ n ∧ n < 1000 ∧ k = 1001 * n}, Nat.gcd 1001 m = 1001 :=
by
  sorry

end gcd_of_repeated_three_digit_l408_408363


namespace sum_inverse_less_half_l408_408869

noncomputable def sequence (a : ℝ) : ℕ → ℝ
| 0     := 1
| 1     := a
| (n+2) := (sequence a (n + 1))^2 / (sequence a n)^2 - 2 * (sequence a (n + 1))

theorem sum_inverse_less_half (a : ℝ) (ha : a > 2) (k : ℕ) :
  (∑ i in Finset.range (k+1), (1 / sequence a i)) < (1 / 2 * (2 + a - Real.sqrt (a^2 - 4))) :=
sorry

end sum_inverse_less_half_l408_408869


namespace miles_per_gallon_city_l408_408752

theorem miles_per_gallon_city
  (T : ℝ) -- tank size
  (h c : ℝ) -- miles per gallon on highway 'h' and in the city 'c'
  (h_eq : h = (462 / T))
  (c_eq : c = (336 / T))
  (relation : c = h - 9)
  (solution : c = 24) : c = 24 := 
sorry

end miles_per_gallon_city_l408_408752


namespace four_digit_numbers_count_l408_408528

theorem four_digit_numbers_count :
  let digit_set := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let first_digit_choices := {5, 6, 7, 8, 9}
  let middle_digit_pairs := { (a, b) | a ∈ digit_set ∧ b ∈ digit_set ∧ a * b > 10 }
  let last_digit_choices := digit_set
  first_digit_choices.card * middle_digit_pairs.card * last_digit_choices.card = 2700 :=
by
  sorry

end four_digit_numbers_count_l408_408528


namespace correct_tangent_line_b_value_l408_408937

noncomputable
def tangent_line_b_value (x : ℝ) (ln_x_nonzero : x ≠ 0) (hx : x = Real.exp (-3)) : ℝ :=
  let y := x * Real.log x
  let dydx := Real.log x + 1
  have h_slope : dydx = -2 := by 
    rw [hx, Real.log_exp, add_eq_neg_iff_eq_neg_add, neg_add_eq_sub, eq_self_iff_true]
  have h_tangent := Real.exp (-3) * (Real.log (Real.exp (-3))) = -3 * Real.exp (-3) :=
    by rw [Real.log_exp, add_eq_self, eq_self_iff_true]
  -Real.exp (-3)
  
theorem correct_tangent_line_b_value : 
  ∀ x : ℝ, x ≠ 0 → x = Real.exp (-3) → tangent_line_b_value x sorry sorry = -Real.exp (-3) :=
by 
    intros x hx h_exp
    rw [tangent_line_b_value]
    sorry

end correct_tangent_line_b_value_l408_408937


namespace tangent_line_at_one_monotonicity_conditions_l408_408863

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ :=
  a / x + 2 / (x + 1)^2

def tangent_line_eq (a : ℝ) (y : ℝ) : Prop :=
  y = (4 * a + 2) / 4 * x - (2 * a) / 4

theorem tangent_line_at_one (a : ℝ) :
  tangent_line_eq a (f_prime a 1) :=
sorry

theorem monotonicity_conditions (a : ℝ) :
  (a > 0 → (∀ x, 0 < x → f_prime a x > 0)) ∧
  (a < -1/2 → (∀ x, 0 < x → f_prime a x < 0)) ∧
  (a = -1/2 → (∀ x, 0 < x → f_prime a x < 0)) ∧
  (-1/2 < a ∧ a < 0 →
    (∀ x, 0 < x ∧ x < -(a+1)-sqrt(2*a+1) / a → f_prime a x > 0) ∧
    (∀ x, x > -(a+1)-sqrt(2*a+1) / a → f_prime a x < 0)) :=
sorry

end tangent_line_at_one_monotonicity_conditions_l408_408863


namespace parabola_and_line_eq_l408_408099

-- Definitions for conditions
def parabola_vertex_origin (x y : ℝ) : Prop := y^2 = 4 * x
def parabola_symmetric_x_axis (x y : ℝ) : Prop := y^2 = 4 * x
def parabola_passes_through_P (x y px py : ℝ) : Prop := y^2 = 4 * x ∧ x = px ∧ y = py
def directrix_equation : ℝ → Prop := λ x, x = -1
def line_not_pass_through_P (m px py : ℝ) : Prop := ∀ x y, y = m * x + (py - m * px) → x ≠ px ∨ y ≠ py
def line_slope_one (m : ℝ) : Prop := m = 1
def line_intersects_parabola (m b : ℝ) : Prop := ∃ x y, y = m * x + b ∧ y^2 = 4 * x
def circle_diameter_AB_through_P (A B P : ℝ × ℝ) : Prop := 
    let (x1, y1) := A; let (x2, y2) := B; let (xp, yp) := P in
        (x1 - xp)^2 + (y1 - yp)^2 = (x2 - xp)^2 + (y2 - yp)^2

-- Problem statement in Lean 4
theorem parabola_and_line_eq (C P A B l : ℝ × ℝ) :
  parabola_vertex_origin (C.1) (C.2) →
  parabola_symmetric_x_axis (C.1) (C.2) →
  parabola_passes_through_P (C.1) (C.2) (1) (2) →
  directrix_equation (-1) →
  line_slope_one (1) →
  line_not_pass_through_P (1) (1) (2) →
  line_intersects_parabola (1) (l.2) →
  circle_diameter_AB_through_P A B (1, 2) →
  (C.2)^2 = 4 * (C.1) ∧ (-1) = -1 ∧ l.1 - l.2 - 7 = 0 := sorry

end parabola_and_line_eq_l408_408099


namespace stratified_sampling_size_l408_408661

theorem stratified_sampling_size (a_ratio b_ratio c_ratio : ℕ) (total_items_A : ℕ) (h_ratio : a_ratio + b_ratio + c_ratio = 10)
  (h_A_ratio : a_ratio = 2) (h_B_ratio : b_ratio = 3) (h_C_ratio : c_ratio = 5) (items_A : total_items_A = 20) : 
  ∃ n : ℕ, n = total_items_A * 5 := 
by {
  -- The proof should go here. Since we only need the statement:
  sorry
}

end stratified_sampling_size_l408_408661


namespace next_four_digit_number_after_1818_l408_408683

-- Formalization of the given conditions as Lean definitions
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def next_valid_number (n : ℕ) : ℕ :=
  let d1 := n / 1000 in
  let d2 := (n / 100) % 10 in
  let d3 := (n / 10) % 10 in
  let d4 := n % 10 in
  let next_n := (d1 * 10 + d2) * 100 + (d3 * 10 + d4) + 14*10 in -- moving to the next valid 4 digits in base 10 series
  next_n

def condition (n : ℕ) : Prop :=
  let d3d4 := n % 100 in
  is_perfect_square (18 * d3d4)

theorem next_four_digit_number_after_1818 :
  ∀ n : ℕ, 
    condition 1818 → 
    next_valid_number 1818 = 1832 :=
begin
  intro n,
  intro cond,
  sorry
end

end next_four_digit_number_after_1818_l408_408683


namespace meet_time_l408_408412

noncomputable def elina_speed : ℝ := 12
noncomputable def gustavo_speed : ℝ := 5
noncomputable def initial_travel_time : ℝ := 12 / 60  -- in hours
noncomputable def distance_elina : ℝ := elina_speed * initial_travel_time
noncomputable def distance_gustavo : ℝ := gustavo_speed * initial_travel_time

noncomputable def distance_between : ℝ := Real.sqrt (distance_elina ^ 2 + distance_gustavo ^ 2)
noncomputable def relative_speed : ℝ := elina_speed + gustavo_speed
noncomputable def time_to_meet : ℝ := distance_between / relative_speed

noncomputable def total_time_in_minutes : ℝ := (time_to_meet * 60) + 12

theorem meet_time : total_time_in_minutes ≈ 21.18 :=
by
  have h1 : distance_elina = 2.4,
  have h2 : distance_gustavo = 1,
  have h3 : distance_between = 2.6,
  have h4 : relative_speed = 17,
  have h5 : time_to_meet ≈ 0.153,
  have h6 : total_time_in_minutes ≈ 21.18,
  sorry

end meet_time_l408_408412


namespace solve_system_equations_l408_408260

theorem solve_system_equations (x y z b : ℝ) :
  (3 * x * y * z - x^3 - y^3 - z^3 = b^3) ∧ 
  (x + y + z = 2 * b) ∧ 
  (x^2 + y^2 + z^2 = b^2) → 
  b = 0 ∧ (∃ t, (x = 0 ∧ y = t ∧ z = -t) ∨ 
                (x = t ∧ y = 0 ∧ z = -t) ∨ 
                (x = -t ∧ y = t ∧ z = 0)) :=
by
  sorry -- Proof to be provided

end solve_system_equations_l408_408260


namespace part1_omega_part1_monotonic_increasing_intervals_part2_max_value_l408_408092

noncomputable def ω := 1

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * ω * x - Real.pi / 6)

theorem part1_omega (h_per : (∃ p > 0, ∀ x, f(x + p) = f x) ∧ ∃ ω, ω > 0 ∧ f(x) = 2 * Real.sin (2 * ω * x - Real.pi / 6)) : ω = 1 :=
sorry

theorem part1_monotonic_increasing_intervals :
     ∀ k : ℤ, ∀ x, - (Real.pi / 6) + k * Real.pi ≤ x ∧ x ≤ Real.pi / 3 + k * Real.pi → f x > 0 :=
sorry

theorem part2_max_value (x : ℝ) (h_interval : 0 ≤ x ∧ x ≤ 5 * Real.pi / 12) : f x ≤ 2 :=
sorry

end part1_omega_part1_monotonic_increasing_intervals_part2_max_value_l408_408092


namespace Lacy_correct_percentage_l408_408232

def problems_exam (y : ℕ) := 10 * y
def problems_section1 (y : ℕ) := 6 * y
def problems_section2 (y : ℕ) := 4 * y
def missed_section1 (y : ℕ) := 2 * y
def missed_section2 (y : ℕ) := y
def solved_section1 (y : ℕ) := problems_section1 y - missed_section1 y
def solved_section2 (y : ℕ) := problems_section2 y - missed_section2 y
def total_solved (y : ℕ) := solved_section1 y + solved_section2 y
def percent_correct (y : ℕ) := (total_solved y : ℚ) / (problems_exam y) * 100

theorem Lacy_correct_percentage (y : ℕ) : percent_correct y = 70 := by
  -- Proof would go here
  sorry

end Lacy_correct_percentage_l408_408232


namespace units_digit_sum_of_squares_odd_integers_l408_408724

theorem units_digit_sum_of_squares_odd_integers :
  let units_digit (n : ℤ) : ℤ := n % 10
  let sum_units_digits (n : ℕ) : ℤ :=
    (List.range (2 * n + 1)).filter (λ x => x % 2 = 1).map (λ k => units_digit (k * k)).sum in
  units_digit (sum_units_digits 2023) = 5 :=
begin
  sorry
end

end units_digit_sum_of_squares_odd_integers_l408_408724


namespace mass_percentage_N_in_N2O5_is_25_93_l408_408450

-- Define the molar masses and the formula of Dinitrogen pentoxide
constant molar_mass_N : ℝ := 14.01
constant molar_mass_O : ℝ := 16.00
constant n_atoms_N2O5 : ℕ := 2
constant o_atoms_N2O5 : ℕ := 5

-- Define the total molar mass of Dinitrogen pentoxide
def molar_mass_N2O5 := (n_atoms_N2O5 * molar_mass_N) + (o_atoms_N2O5 * molar_mass_O)

-- Define the mass percentage of nitrogen in Dinitrogen pentoxide
def mass_percentage_N_in_N2O5 := (n_atoms_N2O5 * molar_mass_N) / molar_mass_N2O5 * 100

-- State the problem: Prove that the mass percentage of nitrogen in Dinitrogen pentoxide is approximately 25.93%
theorem mass_percentage_N_in_N2O5_is_25_93 : mass_percentage_N_in_N2O5 = 25.93 :=
by
  sorry

end mass_percentage_N_in_N2O5_is_25_93_l408_408450


namespace line_m_eq_line_n_eq_l408_408186
-- Definitions for conditions
def point_A : ℝ × ℝ := (-2, 1)
def line_l (x y : ℝ) := 2 * x - y - 3 = 0

-- Proof statement for part (1)
theorem line_m_eq :
  ∃ (m : ℝ → ℝ → Prop), (∀ x y, m x y ↔ (2 * x - y + 5 = 0)) ∧
    (∀ x y, line_l x y → m (-2) 1 → True) :=
sorry

-- Proof statement for part (2)
theorem line_n_eq :
  ∃ (n : ℝ → ℝ → Prop), (∀ x y, n x y ↔ (x + 2 * y = 0)) ∧
    (∀ x y, line_l x y → n (-2) 1 → True) :=
sorry

end line_m_eq_line_n_eq_l408_408186


namespace smallest_n_for_log_sum_l408_408391

theorem smallest_n_for_log_sum :
  ∃ n : ℕ, (∑ k in Finset.range (n + 1), Real.logBase 3 (1 + 1 / 3^(3^k))) ≥ 1 + Real.logBase 3 (10007 / 10008) ∧ (∀ m : ℕ, (m < n) → (∑ k in Finset.range (m + 1), Real.logBase 3 (1 + 1 / 3^(3^k))) < 1 + Real.logBase 3 (10007 / 10008)) ∧ n = 2 := sorry

end smallest_n_for_log_sum_l408_408391


namespace astroid_area_l408_408800

-- Definitions coming from the conditions
noncomputable def x (t : ℝ) := 4 * (Real.cos t)^3
noncomputable def y (t : ℝ) := 4 * (Real.sin t)^3

-- The theorem stating the area of the astroid
theorem astroid_area : (∫ t in (0 : ℝ)..(Real.pi / 2), y t * (deriv x t)) * 4 = 24 * Real.pi :=
by
  sorry

end astroid_area_l408_408800


namespace find_x_l408_408475

noncomputable def A (S : List ℝ) : List ℝ :=
  match S with
  | [] => []
  | [x] => []
  | x1::x2::xs => ((x1 + x2) / 2) :: A (x2::xs)

noncomputable def Am (S : List ℝ) (m : ℕ) : List ℝ :=
  match m with
  | 0 => S
  | (m+1) => A (Am S m)

theorem find_x (x : ℝ) (S : List ℝ) (hS : S = List.map (λ n : ℕ => x^n) (List.arange 51 ++ [0])) (h : 0 < x) :
  Am S 50 = [1 / 2^25] → x = Real.sqrt 2 - 1 :=
begin
  sorry
end

end find_x_l408_408475


namespace part1_part2_l408_408123

variable {f : ℝ → ℝ}
variable {x : ℝ}

-- The domain of f(x) is (0, +∞)
axiom domain (x : ℝ) : x > 0 → f x ≠ undefined

-- f(2) = 1
axiom f_two : f 2 = 1

-- f(xy) = f(x) + f(y)
axiom functional_equation (x y : ℝ) : f (x * y) = f x + f y

-- f is strictly increasing
axiom strictly_increasing (x1 x2 : ℝ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x1 < x2) : f x1 < f x2

theorem part1 : f 1 = 0 ∧ f 4 = 2 ∧ f 8 = 3 :=
by
  sorry

theorem part2 : 2 < x ∧ x ≤ 4 → f x + f (x - 2) ≤ 3 :=
by
  sorry

end part1_part2_l408_408123


namespace next_number_property_l408_408676

theorem next_number_property (a b c d : ℕ) (h : a = 1 ∧ b = 8):
  ((10 * a + b) * (10 * c + d)) = (n * n) → 
  (1818 < (1000 * a + 100 * b + 10 * c + d)) →
  (∃ abcd, (1000 * a + 100 * b + 10 * c + d) = 1832) :=
by
  sorry

end next_number_property_l408_408676


namespace waiter_earnings_l408_408410

variable (customers_served : ℕ)
variable (no_tip : ℕ)
variable (tip_3 : ℕ)
variable (tip_4 : ℕ)
variable (tip_5 : ℕ)
variable (couples_split : ℕ)
variable (tip_contributed : ℝ)
variable (meal_paid : ℝ)

-- Conditions given in the problem
def conditions : Prop :=
  customers_served = 25 ∧
  no_tip = 5 ∧
  tip_3 = 8 ∧
  tip_4 = 6 ∧
  tip_5 = 6 ∧
  couples_split = 2 ∧
  tip_contributed = 0.1 ∧
  meal_paid = 6 ∧
  meal_paid = 0.8 * 7.5

-- Correct answer from the solution
def total_earnings : ℝ := 64.20

-- Formal statement to be proved in Lean 4
theorem waiter_earnings : conditions →
  let tip_total := (tip_3 * 3) + (tip_4 * 4) + (tip_5 * 5) in
  let contribution := tip_contributed * tip_total in
  let net_tips := tip_total - contribution in
  let earnings := net_tips - meal_paid in
  earnings = total_earnings :=
by
  intros
  sorry

end waiter_earnings_l408_408410


namespace max_modulus_complex_l408_408583

theorem max_modulus_complex (w : ℂ) (h : complex.abs w = 2) :
  ∃ z : ℂ, complex.abs ((w - 2) ^ 2 * (w + 2)) = 24 :=
sorry

end max_modulus_complex_l408_408583


namespace triangle_side_relationship_l408_408712

variables {α : Type} [LinearOrderedField α]

-- Defining the sides of the triangle
variables (a b c : α)

-- Defining the condition that a line parallel to one side splits the triangle into equal area and perimeter parts
def equal_area_perimeter_split (a b c λ : α) : Prop :=
  λ = 1 / (Real.sqrt 2) ∧ 
  c = (Real.sqrt 2 - 1) * (a + b)

/- The statement we want to prove: -/
theorem triangle_side_relationship (a b c : α) (λ : α) (H : equal_area_perimeter_split a b c λ) : 
  c = (Real.sqrt 2 - 1) * (a + b) := 
by 
  sorry

end triangle_side_relationship_l408_408712


namespace sum_of_distances_to_sides_eq_height_l408_408623

-- Define an equilateral triangle
structure EquilateralTriangle := 
  (A B C M : Point)
  (a : ℝ)
  (AB_eq : dist A B = a)
  (AC_eq : dist A C = a)
  (BC_eq : dist B C = a)

-- Define the height function for an equilateral triangle
def height (T : EquilateralTriangle) : ℝ := 
  (sqrt 3 / 2) * T.a

-- Define distances from point M to the sides
def distance_to_sides (T : EquilateralTriangle) (M : Point) : ℝ × ℝ × ℝ :=
  -- Assumes existence of functions dist_to_line that calculates perpendicular distance from point to line
  (dist_to_line M T.B T.C, dist_to_line M T.C T.A, dist_to_line M T.A T.B)

theorem sum_of_distances_to_sides_eq_height (T : EquilateralTriangle) :
  let ⟨h1, h2, h3⟩ := distance_to_sides T T.M in 
  h1 + h2 + h3 = height T := 
sorry

end sum_of_distances_to_sides_eq_height_l408_408623


namespace find_n_l408_408344

def A_counts_forward (a : ℕ) : Prop := ∃ k : ℕ, a = 1 + 2 * k
def B_counts_backward (b n : ℕ) : Prop := ∃ k : ℕ, b = n - 2 * k
def same_speed (a b : ℕ) : Prop := A_counts_forward a ∧ B_counts_backward b _ ∧ ∃ c : ℕ, a = 1 + 2 * c ∧ b = c

theorem find_n (n : ℕ) (B_count_10 : B_counts_backward 89 n) : n = 107 :=
by 
  have A_19 : A_counts_forward 19 := sorry
  have B_89 : same_speed 19 89 := sorry
  exact sorry

end find_n_l408_408344


namespace combined_average_yield_l408_408816

variable (YieldA : ℝ) (PriceA : ℝ)
variable (YieldB : ℝ) (PriceB : ℝ)
variable (YieldC : ℝ) (PriceC : ℝ)

def AnnualIncome (Yield : ℝ) (Price : ℝ) : ℝ :=
  Yield * Price

def TotalAnnualIncome (IncomeA IncomeB IncomeC : ℝ) : ℝ :=
  IncomeA + IncomeB + IncomeC

def TotalInvestment (PriceA PriceB PriceC : ℝ) : ℝ :=
  PriceA + PriceB + PriceC

def CombinedAverageYield (TotalIncome TotalInvestment : ℝ) : ℝ :=
  TotalIncome / TotalInvestment

theorem combined_average_yield :
  YieldA = 0.20 → PriceA = 100 → YieldB = 0.12 → PriceB = 200 → YieldC = 0.25 → PriceC = 300 →
  CombinedAverageYield
    (TotalAnnualIncome
      (AnnualIncome YieldA PriceA)
      (AnnualIncome YieldB PriceB)
      (AnnualIncome YieldC PriceC))
    (TotalInvestment PriceA PriceB PriceC) = 0.1983 :=
by
  intro hYA hPA hYB hPB hYC hPC
  rw [hYA, hPA, hYB, hPB, hYC, hPC]
  sorry

end combined_average_yield_l408_408816


namespace asymptotic_line_hyperbola_l408_408266

theorem asymptotic_line_hyperbola 
  (hyperbola_eqn : ∀ x y : ℝ, 3 * x^2 - y^2 = 1) :
  ∀ x : ℝ, ∃ y : ℝ, y = ±(ℝ.sqrt 3) * x :=
by
  sorry

end asymptotic_line_hyperbola_l408_408266


namespace total_present_worth_correct_l408_408287

def present_value (FV : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  FV / (1 + r) ^ n

def total_present_value : ℝ :=
  let PV1 := present_value 242 0.10 2
  let PV2 := present_value 350 0.12 3
  let PV3 := present_value 500 0.08 4
  let PV4 := present_value 750 0.07 5
  PV1 + PV2 + PV3 + PV4

theorem total_present_worth_correct :
  total_present_value = 1351.59 :=
by
  sorry

end total_present_worth_correct_l408_408287


namespace alex_sam_sum_difference_is_10875_l408_408398

noncomputable def alex_sum : ℕ :=
(150 * 151) / 2

noncomputable def sam_sum : ℕ :=
30 * 15

def alex_sam_difference : ℕ :=
(abs (alex_sum - sam_sum))

theorem alex_sam_sum_difference_is_10875 : alex_sam_difference = 10875 := by
  sorry

end alex_sam_sum_difference_is_10875_l408_408398


namespace expression_evaluation_l408_408063

theorem expression_evaluation :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
sorry

end expression_evaluation_l408_408063


namespace bricks_in_wall_l408_408386

variables (w : ℕ) (a b : ℕ) (decreased_output : ℕ) (combine_time : ℕ)

-- Alice's and Bob's rates and the combined decreased productivity conditions
def work_alice := w / 8
def work_bob := w / 12
def combined_work := work_alice + work_bob - decreased_output
def finishing_time := 6

theorem bricks_in_wall :
  (a = 8) → (b = 12) → (decreased_output = 15) → (combine_time = finishing_time) → 
  6 * (work_alice + work_bob - decreased_output) = w → 
  w = 360 :=
begin
  intros ha hb hd ht equation,
  rw [ha, hb, hd, ht] at equation,
  exact sorry,
end

end bricks_in_wall_l408_408386


namespace remaining_coins_denomination_l408_408692

def denomination_of_remaining_coins (total_coins : ℕ) (total_value : ℕ) (paise_20_count : ℕ) (paise_20_value : ℕ) : ℕ :=
  let remaining_coins := total_coins - paise_20_count
  let remaining_value := total_value - paise_20_count * paise_20_value
  remaining_value / remaining_coins

theorem remaining_coins_denomination :
  denomination_of_remaining_coins 334 7100 250 20 = 25 :=
by
  sorry

end remaining_coins_denomination_l408_408692


namespace top_card_is_queen_probability_l408_408367

-- Define the conditions of the problem
def standard_deck_size := 52
def number_of_queens := 4

-- Problem statement: The probability that the top card is a Queen
theorem top_card_is_queen_probability : 
  (number_of_queens : ℚ) / standard_deck_size = 1 / 13 := 
sorry

end top_card_is_queen_probability_l408_408367


namespace sequence_sum_l408_408028

theorem sequence_sum :
  (∀ r: ℝ, 
    (∀ x y: ℝ,
      r = 1 / 4 → 
      x = 256 * r → 
      y = x * r → 
      x + y = 80)) :=
by sorry

end sequence_sum_l408_408028


namespace largest_number_in_ap_l408_408250

theorem largest_number_in_ap (a d : ℝ) :
  let s := [a, a + d, a + 2*d, a + 3*d, a + 4*d, a + 5*d, a + 6*d] in
  (∑ x in s, x^3 = 0) →
  (∑ x in s, x^2 = -224) →
  max a (a + 6*d) = 6 * Real.sqrt 2 := 
sorry

end largest_number_in_ap_l408_408250


namespace cone_volume_l408_408294

theorem cone_volume (V_cylinder : ℝ) (V_cone : ℝ) (h : V_cylinder = 81 * Real.pi) :
  V_cone = 27 * Real.pi :=
by
  sorry

end cone_volume_l408_408294


namespace expand_expression_l408_408834

theorem expand_expression (x y : ℝ) :
  (x + 3) * (4 * x - 5 * y) = 4 * x ^ 2 - 5 * x * y + 12 * x - 15 * y :=
by
  sorry

end expand_expression_l408_408834


namespace ratio_norm_lisa_l408_408341

-- Define the number of photos taken by each photographer.
variable (L M N : ℕ)

-- Given conditions
def norm_photos : Prop := N = 110
def photo_sum_condition : Prop := L + M = M + N - 60

-- Prove the ratio of Norm's photos to Lisa's photos.
theorem ratio_norm_lisa (h1 : norm_photos N) (h2 : photo_sum_condition L M N) : N / L = 11 / 5 := 
by
  sorry

end ratio_norm_lisa_l408_408341


namespace find_equidistant_points_l408_408298

noncomputable def plane (α : Type) := set (α × α × α)

structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def equidistant_from_planes (p: Point) (plane1 plane2 : plane ℝ) : Prop :=
  let d1 := shortest_distance p plane1
  let d2 := shortest_distance p plane2
  d1 = d2

def coordinate_planes : plane ℝ := 
  { (x, y, z) | x = 0 ∨ y = 0 }

theorem find_equidistant_points (plane1 plane2 : plane ℝ) (h1 : parallel plane1 plane2) :
  ∃ (B C E F : Point), 
    equidistant_from_planes B plane1 ∧ 
    equidistant_from_planes B plane2 ∧ 
    equidistant_from_planes B coordinate_planes ∧
    equidistant_from_planes C plane1 ∧ 
    equidistant_from_planes C plane2 ∧ 
    equidistant_from_planes C coordinate_planes ∧
    equidistant_from_planes E plane1 ∧ 
    equidistant_from_planes E plane2 ∧ 
    equidistant_from_planes E coordinate_planes ∧
    equidistant_from_planes F plane1 ∧ 
    equidistant_from_planes F plane2 ∧ 
    equidistant_from_planes F coordinate_planes :=
sorry

end find_equidistant_points_l408_408298


namespace sum_of_lengths_leq_0_5_l408_408616

theorem sum_of_lengths_leq_0_5 (s : Set ℝ) (segments : Finset (Set ℝ))
  (hs : ∀ x ∈ s, 0 ≤ x ∧ x ≤ 1)
  (H : (∀ I ∈ segments, ∀ x y ∈ I, abs (x - y) ≠ 0.1) ∧ (∀ I J ∈ segments, I ≠ J → ∀ x ∈ I, ∀ y ∈ J, abs (x - y) ≠ 0.1)) :
  Finset.sum segments (λ I, measure_theory.measure_Icc measurable_set_Icc (inf I ∩ s) (sup I ∩ s)) ≤ 0.5 := sorry

end sum_of_lengths_leq_0_5_l408_408616


namespace triangle_perimeter_l408_408190

noncomputable theory

variables {A B C : ℝ}
variables {a b c : ℝ}

-- Define the perimeter of the triangle
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Define the conditions
axiom cos_eq_half : cos A = 1/2
axiom given_equation : c * cos B + b * cos C = 2 * a * cos A
axiom side_a : a = 2
axiom area_eq : (1 / 2) * b * c * sin A = sqrt 3

-- The theorem to prove
theorem triangle_perimeter : perimeter a b c = 6 :=
sorry

end triangle_perimeter_l408_408190


namespace train_crossing_time_l408_408151

-- Definitions based on given problem conditions
def length_of_train : ℝ := 165
def speed_of_train_kmph : ℝ := 54
def length_of_bridge : ℝ := 625

-- Helper definition for conversion factor
def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)

-- Defining the speed of the train in meters per second
def speed_of_train_mps : ℝ := kmph_to_mps speed_of_train_kmph

-- Total distance to be covered by the train
def total_distance : ℝ := length_of_train + length_of_bridge

-- Time to cross the bridge
def time_to_cross_bridge : ℝ := total_distance / speed_of_train_mps

-- Statement to prove the time to cross the bridge
theorem train_crossing_time : time_to_cross_bridge = 52.67 :=
by
  -- Using sorry to skip the actual proof steps
  sorry

end train_crossing_time_l408_408151


namespace height_of_removed_player_l408_408636

theorem height_of_removed_player (S : ℕ) (x : ℕ) (total_height_11 : S + x = 182 * 11)
  (average_height_10 : S = 181 * 10): x = 192 :=
by
  sorry

end height_of_removed_player_l408_408636


namespace set_intersection_complement_l408_408222

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set ℕ := {1, 2, 3}

-- Define set B
def B : Set ℕ := {3, 4, 5}

-- State the theorem
theorem set_intersection_complement :
  (U \ A) ∩ B = {4, 5} := by
  sorry

end set_intersection_complement_l408_408222


namespace next_perfect_square_after_1818_l408_408688

def is_next_perfect_square_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (∃ m : ℕ, n = 1800 + m ∧ m < 100 ∧ (∃ k : ℕ, 18 * m = k^2))

theorem next_perfect_square_after_1818 : ∃ n : ℕ, is_next_perfect_square_number n ∧ n > 1818 ∧ n = 1832 :=
by
  existsi 1832
  unfold is_next_perfect_square_number
  split
  . apply and.intro
    { linarith }
    { split
      . existsi 32
        split
        . linarith
        . existsi 18
          linarith
      . sorry }

end next_perfect_square_after_1818_l408_408688


namespace average_velocity_first_30_seconds_l408_408792

noncomputable def velocity (t : ℝ) : ℝ := t^2 - 3 * t + 8

def average_velocity (v : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (∫ t in a..b, v t) / (b - a)

theorem average_velocity_first_30_seconds : average_velocity velocity 0 30 = 263 :=
by
  sorry

end average_velocity_first_30_seconds_l408_408792


namespace claire_gift_card_balance_l408_408812

/--
Claire has a $100 gift card to her favorite coffee shop.
A latte costs $3.75.
A croissant costs $3.50.
Claire buys one latte and one croissant every day for a week.
Claire buys 5 cookies, each costing $1.25.

Prove that the amount of money left on Claire's gift card after a week is $43.00.
-/
theorem claire_gift_card_balance :
  let initial_balance : ℝ := 100
  let latte_cost : ℝ := 3.75
  let croissant_cost : ℝ := 3.50
  let daily_expense : ℝ := latte_cost + croissant_cost
  let weekly_expense : ℝ := daily_expense * 7
  let cookie_cost : ℝ := 1.25
  let total_cookie_expense : ℝ := cookie_cost * 5
  let total_expense : ℝ := weekly_expense + total_cookie_expense
  let remaining_balance : ℝ := initial_balance - total_expense
  remaining_balance = 43 :=
by
  sorry

end claire_gift_card_balance_l408_408812


namespace geometric_sequence_product_exceeds_1000_l408_408966

theorem geometric_sequence_product_exceeds_1000 :
  ∃ n : ℕ, n > 0 ∧
  (∀ k : ℕ, k > 0 ∧ k < n → (∏ i in finset.range k, (3 : ℕ)^(i.succ / 6)) ≤ (1000 : ℕ)) ∧
  (∏ i in finset.range n, (3 : ℕ)^(i.succ / 6)) > (1000 : ℕ) :=
sorry

end geometric_sequence_product_exceeds_1000_l408_408966


namespace volume_of_revolution_l408_408183

noncomputable def volume_of_solid_of_revolution : ℝ :=
  2 * π * ∫ y in 0..1, real.exp (y / 2)

theorem volume_of_revolution :
  volume_of_solid_of_revolution = 4 * π * (real.sqrt real.e - 1) :=
by
  sorry

end volume_of_revolution_l408_408183


namespace james_initial_winnings_l408_408561

noncomputable def initial_winnings (final_amount : ℝ) : ℝ :=
  final_amount / 0.6598125

theorem james_initial_winnings:
  initial_winnings 320 = 485 := 
by
  unfold initial_winnings
  have : 320 / 0.6598125 = 485 := sorry
  exact this

end james_initial_winnings_l408_408561


namespace least_positive_integer_l408_408316

/-- Define the conditions for the least positive integer divisible by each of the first ten positive integers and also ends with the digit '0' -/
def conditions (n : ℕ) : Prop :=
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ (n % 10 = 0)

/-- Statement of the proof problem: the least positive integer satisfying the conditions is 2520 -/
theorem least_positive_integer :
  ∃ n : ℕ, conditions n ∧ (∀ m : ℕ, conditions m → n ≤ m) :=
⟨2520, 
  by {
    -- The proof goes here
    sorry
  }
⟩

end least_positive_integer_l408_408316


namespace jill_total_watch_time_l408_408564

theorem jill_total_watch_time :
  ∀ (length_first_show length_second_show total_watch_time : ℕ),
    length_first_show = 30 →
    length_second_show = 4 * length_first_show →
    total_watch_time = length_first_show + length_second_show →
    total_watch_time = 150 :=
by
  sorry

end jill_total_watch_time_l408_408564


namespace employee_payment_l408_408742

theorem employee_payment 
    (total_pay : ℕ)
    (pay_A : ℕ)
    (pay_B : ℕ)
    (h1 : total_pay = 560)
    (h2 : pay_A = 3 * pay_B / 2)
    (h3 : pay_A + pay_B = total_pay) :
    pay_B = 224 :=
sorry

end employee_payment_l408_408742


namespace sum_inequalities_l408_408214

theorem sum_inequalities 
  (n : ℕ) 
  (h1 : 1 ≤ n)
  (x : Fin n → ℝ) 
  (h2 : ∀ i, 0 < x i) 
  (h3 : ∑ i in Finset.univ, x i = 1) :
  1 ≤ ∑ i in Finset.range n, x i / (Real.sqrt (1 + ∑ j in Finset.range i, x j) * Real.sqrt (∑ j in Finset.Ico i n, x j)) 
  ∧ 
  ∑ i in Finset.range n, x i / (Real.sqrt (1 + ∑ j in Finset.range i, x j) * Real.sqrt (∑ j in Finset.Ico i n, x j)) < Real.pi / 2 :=
sorry

end sum_inequalities_l408_408214


namespace Xingyou_age_is_3_l408_408557

theorem Xingyou_age_is_3 (x : ℕ) (h1 : x = x) (h2 : x + 3 = 2 * x) : x = 3 :=
by
  sorry

end Xingyou_age_is_3_l408_408557


namespace sum_series_eq_five_over_eight_l408_408833

theorem sum_series_eq_five_over_eight : 
  (∑ n : ℕ in Finset.range (∞), (2 * n.succ) / (5 ^ n.succ)) = 5 / 8 :=
sorry

end sum_series_eq_five_over_eight_l408_408833


namespace next_valid_number_after_1818_l408_408680

open Nat

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem next_valid_number_after_1818 : 
  ∀ (a b : ℕ), 10 * a + b > 18 → 18 * (10 * a + b) = 1832 → isPerfectSquare (18 * (10 * a + b)) → 
  10 * 10 * 8 + 10 * a + b = 1832 :=
by {
  intros,
  sorry
}

end next_valid_number_after_1818_l408_408680


namespace Martin_cost_l408_408796

/-- Define the costs as real numbers. -/
variables (P N E : ℝ)

/-- Define the conditions given in the problem. -/
def Gloria_purchase : Prop := N + E = 85
def Zachary_purchase : Prop := P + E = 45
def Total_cost      : Prop := P + N + E = 105

/-- Prove that Martin paid 80 cents for a pencil and a notebook. -/
theorem Martin_cost 
  (h1 : Gloria_purchase N E)
  (h2 : Zachary_purchase P E)
  (h3 : Total_cost P N E) : P + N = 80 := 
by
  sorry

end Martin_cost_l408_408796


namespace length_PS_l408_408551

def quadrilateral (P Q R S: Type) : Prop := 
  ∃ (PQ QR RS: ℕ) (α β: ℚ),
    PQ = 7 ∧ QR = 10 ∧ RS = 25 ∧ α = 90 ∧ β = 90 

theorem length_PS (P Q R S: Type) (PQ QR RS: ℕ) (α β: ℚ) (h: quadrilateral P Q R S):
  PQ = 7 → QR = 10 → RS = 25 → α = 90 → β = 90 → 
  ∃ (PS: ℚ), PS = 2 * Real.sqrt 106 :=
by 
  intros hPQ hQR hRS hα hβ
  use 2 * Real.sqrt 106
  sorry

end length_PS_l408_408551


namespace smallest_x_for_multiple_l408_408320

theorem smallest_x_for_multiple (x : ℕ) (h : x > 0) :
  (450 * x) % 500 = 0 ↔ x = 10 := by
  sorry

end smallest_x_for_multiple_l408_408320


namespace normal_distribution_probability_between_l408_408955

open Probability

noncomputable def normal_dist (μ σ : ℝ) : EventSpace :=
  NormalDistribution μ (σ ^ 2)

variable {X : ℝ}

theorem normal_distribution_probability_between
  (μ σ : ℝ)
  (hX : X ∈ normal_dist 1 (σ ^ 2))
  (h0 : P(X ≤ 0) = 0.3) : P(0 < X < 2) = 0.4 :=
by
  sorry

end normal_distribution_probability_between_l408_408955


namespace measure_of_angle_B_range_of_cos_A_plus_cos_C_l408_408180

-- Define the conditions
def is_acute_triangle (A B C : ℝ) : Prop :=
  A + B + C = π ∧ 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

def condition (A C : ℝ) : Prop :=
  sqrt 3 * tan A * tan C = tan A + tan C + sqrt 3

-- Part (1): Prove the measure of angle B
theorem measure_of_angle_B {A C : ℝ} (h1 : is_acute_triangle A B C)
  (h2 : condition A C) : B = π / 3 :=
sorry

-- Part (2): Prove the range of values for cos A + cos C
theorem range_of_cos_A_plus_cos_C {A C : ℝ} (h1 : is_acute_triangle A B C)
  (h2 : condition A C) (h3 : B = π / 3) : 
  ∃ x, ∃ y, x = cos A + cos C ∧ (x = sqrt 3 / 2 ∨ x = 1) ∧ sqrt 3 / 2 < x ∧ x <= 1 :=
sorry

end measure_of_angle_B_range_of_cos_A_plus_cos_C_l408_408180


namespace OH_parallel_AC_l408_408618

/--
In a triangle \( \Delta ABC \), let \(O\) be the circumcenter and \(H\) be the orthocenter.
Given the properties of these points and the geometry of the triangle,
prove that \(OH \parallel AC\).
-/
theorem OH_parallel_AC
  (A B C O H : Type)
  [IsCircumcenter O A B C] -- Condition 1
  [IsOrthocenter H A B C] -- Condition 2
  : Parallel O H A C := 
sorry

end OH_parallel_AC_l408_408618


namespace magnitude_product_complex_l408_408414

theorem magnitude_product_complex :
  let z1 := Complex.mk 7 (-4)
  let z2 := Complex.mk 3 11
  Complex.abs (z1 * z2) = Real.sqrt 8450 :=
by
  sorry

end magnitude_product_complex_l408_408414


namespace monotonic_decreasing_interval_l408_408283

def function_y (x : ℝ) : ℝ := (1/2) * x^2 - real.log x

theorem monotonic_decreasing_interval :
  ∃ I : set ℝ, I = set.Ioo 0 1 ∧ 
  ∀ x ∈ I, 
    function_y' x < 0 ∧ 
    x > 0 := 
begin
  sorry
end

end monotonic_decreasing_interval_l408_408283


namespace students_remaining_after_fourth_stop_l408_408162

variable (n : ℕ)
variable (frac : ℚ)

def initial_students := (64 : ℚ)
def fraction_remaining := (2/3 : ℚ)

theorem students_remaining_after_fourth_stop : 
  let after_first_stop := initial_students * fraction_remaining
  let after_second_stop := after_first_stop * fraction_remaining
  let after_third_stop := after_second_stop * fraction_remaining
  let after_fourth_stop := after_third_stop * fraction_remaining
  after_fourth_stop = (1024 / 81) := 
by 
  sorry

end students_remaining_after_fourth_stop_l408_408162


namespace downstream_speed_l408_408761

theorem downstream_speed (upstream_speed man_still_water_speed : ℕ) (h1 : upstream_speed = 25) (h2 : man_still_water_speed = 33) : 
  let C := man_still_water_speed - upstream_speed in
  let downstream_speed := man_still_water_speed + C in 
  downstream_speed = 41 := 
by 
  sorry

end downstream_speed_l408_408761


namespace bounded_region_area_l408_408314

-- Define the lines
def line1 (x : ℝ) : ℝ := 3 * x - 6
def line2 (x : ℝ) : ℝ := -2 * x + 14

-- Intersection point
def intersection : ℝ × ℝ := (4, 6)

-- y-intercepts
def y_intercept_line1 : ℝ × ℝ := (0, -6)
def y_intercept_line2 : ℝ × ℝ := (0, 14)

-- Prove the area of the region bounded by the lines and the y-axis is 40 square units
theorem bounded_region_area : 
  let base := (y_intercept_line2.2 - y_intercept_line1.2).abs,
      height := (intersection.1).abs in
  (1/2) * base * height = 40 :=
by
  sorry

end bounded_region_area_l408_408314


namespace fedora_cleaning_time_l408_408336

theorem fedora_cleaning_time
  (cleaned_sections_time : ℕ)
  (cleaned_sections : ℕ)
  (total_sections : ℕ)
  : cleaned_sections_time = 34 → cleaned_sections = 3 → total_sections = 15 →  
  let time_per_section := cleaned_sections_time / cleaned_sections in
  let remaining_sections := total_sections - cleaned_sections in
  let remaining_time := remaining_sections * time_per_section in
  remaining_time = 136 :=
by
  intros h1 h2 h3
  let time_per_section := 34 / 3 in -- derived from h1 and h2
  let remaining_sections := 15 - 3 in -- derived from h3 and h2
  let remaining_time := remaining_sections * time_per_section in
  have time_per_section_approx : time_per_section ≈ 11.33 := sorry
  have remaining_time_approx : remaining_time ≈ 136 := sorry
  exact sorry

end fedora_cleaning_time_l408_408336


namespace gcd_of_repeated_six_digit_integers_l408_408366

-- Given condition
def is_repeated_six_digit_integer (n : ℕ) : Prop :=
  100 ≤ n / 1000 ∧ n / 1000 < 1000 ∧ n = 1001 * (n / 1000)

-- Theorem to prove
theorem gcd_of_repeated_six_digit_integers :
  ∀ n : ℕ, is_repeated_six_digit_integer n → gcd n 1001 = 1001 :=
by sorry

end gcd_of_repeated_six_digit_integers_l408_408366


namespace claire_balance_after_week_l408_408811

theorem claire_balance_after_week :
  ∀ (gift_card : ℝ) (latte_cost croissant_cost : ℝ) (days : ℕ) (cookie_cost : ℝ) (cookies : ℕ),
  gift_card = 100 ∧
  latte_cost = 3.75 ∧
  croissant_cost = 3.50 ∧
  days = 7 ∧
  cookie_cost = 1.25 ∧
  cookies = 5 →
  (gift_card - (days * (latte_cost + croissant_cost) + cookie_cost * cookies) = 43) :=
by
  -- Skipping proof details with sorry
  sorry

end claire_balance_after_week_l408_408811


namespace six_digit_number_divisible_by_7_l408_408993

-- Definition of the problem in Lean 4
theorem six_digit_number_divisible_by_7
  (A B C D E F : ℕ) 
  (HABCDEF : nat.digits 10 (A * 100000 + B * 10000 + C * 1000 + D * 100 + E * 10 + F) = [A, B, C, D, E, F]) 
  (HDEF : D = 1 ∧ E = 2 ∧ F = 4 ∨ D = 1 ∧ E = 4 ∧ F = 2 ∨ D = 2 ∧ E = 1 ∧ F = 4 ∨ D = 2 ∧ E = 4 ∧ F = 1 ∨ D = 4 ∧ E = 1 ∧ F = 2 ∨ D = 4 ∧ E = 2 ∧ F = 1) :
  ∃ G H I J K L : ℕ, (nat.digits 10 (G * 100000 + H * 10000 + I * 1000 + J * 100 + K * 10 + L) = [G, H, I, J, K, L] ∧ (J = 1 ∧ K = 2 ∧ L = 4 ∨ J = 1 ∧ K = 4 ∧ L = 2 ∨ J = 2 ∧ K = 1 ∧ L = 4 ∨ J = 2 ∧ K = 4 ∧ L = 1 ∨ J = 4 ∧ K = 1 ∧ L = 2 ∨ J = 4 ∧ K = 2 ∧ L = 1) ∧ (G * 100000 + H * 10000 + I * 1000 + J * 100 + K * 10 + L) % 7 = 0) 
  ∨ ((A * 100 + B * 10 + C) % 7 = 0) :=
sorry

end six_digit_number_divisible_by_7_l408_408993


namespace line_MN_eq_l408_408105

open Real

variable (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
variable (ellipse_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)

def c : ℝ := sqrt (a^2 - b^2)

def right_focus : ℝ × ℝ := (c, 0)

variable (P Q : ℝ × ℝ)
variable (hP : ellipse_eq P.fst P.snd)
variable (hQ : ellipse_eq Q.fst Q.snd)
variable (chord_PQ_through_focus : ∃ k : ℝ, P = (c + k * Q.fst, k * Q.snd))

theorem line_MN_eq {M N : ℝ × ℝ}
  (hM : M = intersection_point (line_through A P) (line_through Q B))
  (hN : N = intersection_point (line_through B P) (line_through A Q)) :
  M.fst = N.fst → M.fst = a^2 / c := 
sorry

end line_MN_eq_l408_408105


namespace radius_of_circle_l408_408483

theorem radius_of_circle
  (length width : ℝ)
  (length_eq : length = 10)
  (width_eq : width = 6)
  (r A B C D O E F : ℝ)
  (circle_through_AD : (A, D): r)
  (circle_tangent_BC : tangent (BC) (circle_through_AD)): 
  r = 3 :=
by
  sorry

end radius_of_circle_l408_408483


namespace area_PQR_is_2_sqrt_3_l408_408809

-- Definitions and assumptions
variables {M N O P Q R : Point}
variables {ω1 ω2 ω3 : Circle}
variables {A B C : Point}

-- Assuming the conditions
axiom h1 : center(ω1) = M
axiom h2 : center(ω2) = N
axiom h3 : center(ω3) = O
axiom h4 : tangent_points(ω2, ω3) = A
axiom h5 : tangent_points(ω3, ω1) = B
axiom h6 : tangent_points(ω1, ω2) = C
axiom h7 : equilateral_triangle A B C (1 : ℝ)
axiom h8 : line_intersects_twice MO ω3 P Q
axiom h9 : line_intersects_twice MO ω1 P Q
axiom h10 : line_intersects_twice AP ω2 R

-- Goal
theorem area_PQR_is_2_sqrt_3 : area (triangle P Q R) = 2 * sqrt 3 :=
by
  sorry

end area_PQR_is_2_sqrt_3_l408_408809


namespace minimum_heat_correct_l408_408193

noncomputable def minimum_heat_to_shoot_out (M S l1 l2 n p0 g : ℝ) : ℝ :=
  let V1 := l1 * S
  let V2 := l2 * S
  let Q1 := (3/2) * V1 * p0
  let p1 := p0 + (M * g / S)
  let Q2 := (5/2) * p1 * V2
  let Q3 := (3/2) * (l1 + l2) * S * ((p1 + M * g / S) - p1)
  Q1 + Q2 + Q3

theorem minimum_heat_correct :
  minimum_heat_to_shoot_out 10 0.001 0.1 0.15 1 10^5 10 = 127.5 :=
by
  -- Here the proof should go, which will apply the calculations already demonstrated
  sorry

end minimum_heat_correct_l408_408193


namespace four_digit_numbers_l408_408464

theorem four_digit_numbers : 
  ∃ (count : ℕ), count = 18 ∧ 
  ∀ (digits : ℕ → ℕ), 
    (∀ n, n < 4 → digits n ∈ {1, 2, 3}) ∧
    (∃ d, (list.count d [digits 0, digits 1, digits 2, digits 3] = 2) ∧
      (∀ n, n < 3 → digits n ≠ digits (n + 1))) :=
begin
  sorry
end

end four_digit_numbers_l408_408464


namespace fraction_of_menu_can_eat_l408_408831

theorem fraction_of_menu_can_eat (T : ℕ) (hT : T ≠ 0) :
  let sugar_free_dishes := T / 10 in
  let shellfish_free_dishes := 3 * sugar_free_dishes / 4 in
  shellfish_free_dishes / T = 3 / 40 :=
by
  sorry

end fraction_of_menu_can_eat_l408_408831


namespace concrete_volume_is_six_l408_408368

def to_yards (feet : ℕ) (inches : ℕ) : ℚ :=
  feet * (1 / 3) + inches * (1 / 36)

def sidewalk_volume (width_feet : ℕ) (length_feet : ℕ) (thickness_inches : ℕ) : ℚ :=
  to_yards width_feet 0 * to_yards length_feet 0 * to_yards 0 thickness_inches

def border_volume (border_width_feet : ℕ) (border_thickness_inches : ℕ) (sidewalk_length_feet : ℕ) : ℚ :=
  to_yards (2 * border_width_feet) 0 * to_yards sidewalk_length_feet 0 * to_yards 0 border_thickness_inches

def total_concrete_volume (sidewalk_width_feet : ℕ) (sidewalk_length_feet : ℕ) (sidewalk_thickness_inches : ℕ)
  (border_width_feet : ℕ) (border_thickness_inches : ℕ) : ℚ :=
  sidewalk_volume sidewalk_width_feet sidewalk_length_feet sidewalk_thickness_inches +
  border_volume border_width_feet border_thickness_inches sidewalk_length_feet

def volume_in_cubic_yards (w1_feet : ℕ) (l1_feet : ℕ) (t1_inches : ℕ) (w2_feet : ℕ) (t2_inches : ℕ) : ℚ :=
  total_concrete_volume w1_feet l1_feet t1_inches w2_feet t2_inches

theorem concrete_volume_is_six :
  -- conditions
  volume_in_cubic_yards 4 80 4 1 2 = 6 :=
by
  -- Proof omitted
  sorry

end concrete_volume_is_six_l408_408368


namespace average_perm_sum_eq_33_l408_408853

open Finset
open Function

-- Define the sum for a given permutation
def perm_sum (σ : Equiv.Perm (Fin 12)) : ℝ :=
  abs (σ 0 - σ 1) + abs (σ 2 - σ 3) + abs (σ 4 - σ 5) +
  abs (σ 6 - σ 7) + abs (σ 8 - σ 9) + abs (σ 10 - σ 11)

-- Main theorem statement
theorem average_perm_sum_eq_33 : 
  (1 / 12! : ℝ) * ∑ σ in univ, perm_sum σ = 33 :=
by sorry

end average_perm_sum_eq_33_l408_408853


namespace sequence_x_y_sum_l408_408044

theorem sequence_x_y_sum (r : ℝ) (x y : ℝ)
  (h₁ : r = 1 / 4)
  (h₂ : x = 256 * r)
  (h₃ : y = x * r) :
  x + y = 80 :=
by
  sorry

end sequence_x_y_sum_l408_408044


namespace ella_age_l408_408267

theorem ella_age (s t e : ℕ) (h1 : s + t + e = 36) (h2 : e - 5 = s) (h3 : t + 4 = (3 * (s + 4)) / 4) : e = 15 := by
  sorry

end ella_age_l408_408267


namespace find_m_l408_408126

-- Parameters and definitions based on the problem statement
variables {m : ℝ} {P Q : ℝ × ℝ} (hP : P = (-2, m)) (hQ : Q = (m, 4))
def slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

-- The statement of the proof
theorem find_m (h1 : slope P Q = 1) : m = 1 :=
by
  sorry

end find_m_l408_408126


namespace exponent_multiplication_l408_408005

theorem exponent_multiplication (a : ℝ) : (a^3) * (a^2) = a^5 := 
by
  -- Using the property of exponents: a^m * a^n = a^(m + n)
  sorry

end exponent_multiplication_l408_408005


namespace real_roots_of_polynomial_l408_408849

theorem real_roots_of_polynomial :
  (∃ x : ℝ, x^4 - 4*x^3 + 5*x^2 + 2*x - 8 = 0) ↔ (x = 1 + real.sqrt 3 ∨ x = 1 - real.sqrt 3) := by
sorry

end real_roots_of_polynomial_l408_408849


namespace hall_length_l408_408281

variable (breadth length : ℝ)

def condition1 : Prop := length = breadth + 5
def condition2 : Prop := length * breadth = 750

theorem hall_length : condition1 breadth length ∧ condition2 breadth length → length = 30 :=
by
  intros
  sorry

end hall_length_l408_408281


namespace red_or_blue_probability_is_half_l408_408319

-- Define the number of each type of marble
def num_red_marbles : ℕ := 3
def num_blue_marbles : ℕ := 2
def num_yellow_marbles : ℕ := 5

-- Define the total number of marbles
def total_marbles : ℕ := num_red_marbles + num_blue_marbles + num_yellow_marbles

-- Define the number of marbles that are either red or blue
def num_red_or_blue_marbles : ℕ := num_red_marbles + num_blue_marbles

-- Define the probability of drawing a red or blue marble
def probability_red_or_blue : ℚ := num_red_or_blue_marbles / total_marbles

-- Theorem stating the probability is 0.5
theorem red_or_blue_probability_is_half : probability_red_or_blue = 0.5 := by
  sorry

end red_or_blue_probability_is_half_l408_408319


namespace find_g_2_l408_408272

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_eq (x : ℝ) (hx : x ≠ 0) : 2 * g x - 3 * g (1 / x) = x ^ 2

theorem find_g_2 : g 2 = 8.25 :=
by {
  sorry
}

end find_g_2_l408_408272


namespace number_of_books_is_8_l408_408696

def books_and_albums (x y p_a p_b : ℕ) : Prop :=
  (x * p_b = 1056) ∧ (p_b = p_a + 100) ∧ (x = y + 6)

theorem number_of_books_is_8 (y p_a p_b : ℕ) (h : books_and_albums 8 y p_a p_b) : 8 = 8 :=
by
  sorry

end number_of_books_is_8_l408_408696


namespace triangle_BC_l408_408174

theorem triangle_BC (A B C: Point)
  (h_triangle : ∠ A B C = 90°)
  (h_cosB : cos (∠ B C A) = 4/5)
  (h_AB : dist A B = 40) :
  dist B C = 24 := 
sorry

end triangle_BC_l408_408174


namespace complex_modulus_problem_l408_408493

noncomputable def imaginary_unit : ℂ := Complex.I

theorem complex_modulus_problem (z : ℂ) (h : (1 + Real.sqrt 3 * imaginary_unit)^2 * z = 1 - imaginary_unit^3) :
  Complex.abs z = Real.sqrt 2 / 4 :=
by
  sorry

end complex_modulus_problem_l408_408493


namespace exists_at_least_3_red_in_2x2_subgrid_l408_408541
open Finset Function Real

def GridCell := (Fin 9) × (Fin 9)

def is_red (cells : Finset GridCell) (i j : Fin 9) : Prop := (i, j) ∈ cells

theorem exists_at_least_3_red_in_2x2_subgrid (cells : Finset GridCell) (h : cells.card = 46) :
  ∃ (i j : Fin 8), 
    (is_red cells i j) + (is_red cells i (j + 1)) + (is_red cells (i + 1) j) + (is_red cells (i + 1) (j + 1)) ≥ 3 := 
by 
  sorry

end exists_at_least_3_red_in_2x2_subgrid_l408_408541


namespace solve_x_l408_408258

theorem solve_x : ∃ (x : ℚ), (3*x - 17) / 4 = (x + 9) / 6 ∧ x = 69 / 7 :=
by
  sorry

end solve_x_l408_408258


namespace problem_solution_l408_408143

def a : ℝ × ℝ := (5/13, 12/13)
def b : ℝ × ℝ := (4/5, 3/5)

noncomputable def vectors_perpendicular (a b : ℝ × ℝ) : Prop :=
  let a_plus_b := (a.1 + b.1, a.2 + b.2)
  let a_minus_b := (a.1 - b.1, a.2 - b.2)
  (a_plus_b.1 * a_minus_b.1 + a_plus_b.2 * a_minus_b.2) = 0

theorem problem_solution : vectors_perpendicular a b := 
by sorry

end problem_solution_l408_408143


namespace probability_one_male_l408_408465

-- Define the conditions
def males : ℕ := 3
def females : ℕ := 2
def students : ℕ := 5
def select : ℕ := 2

-- Lean theorem statement
theorem probability_one_male :
  (males : ℝ ∈ ℕ) → (females ∈ ℝ ∈ ℕ) → ∀ students : ℕ, students = males + females →
  ∀ select : ℕ, select = 2 →
  (∃ (p : ℚ), p = 3 / 5) := by
  sorry

end probability_one_male_l408_408465


namespace r_can_complete_work_in_R_days_l408_408740

theorem r_can_complete_work_in_R_days (W : ℝ) : 
  (∀ p q r P Q R : ℝ, 
    (P = W / 24) ∧
    (Q = W / 9) ∧
    (10.000000000000002 * (W / 24) + 3 * (W / 9 + W / R) = W) 
  -> R = 12) :=
by
  intros
  sorry

end r_can_complete_work_in_R_days_l408_408740


namespace set_A_is_listed_correctly_l408_408065

def A : Set ℤ := { x | -3 < x ∧ x < 1 }

theorem set_A_is_listed_correctly : A = {-2, -1, 0} := 
by
  sorry

end set_A_is_listed_correctly_l408_408065


namespace angle_between_a_b_l408_408867

noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  real.acos ((inner a b) / (∥a∥ * ∥b∥))

theorem angle_between_a_b (a b : EuclideanSpace ℝ (Fin 3))
  (ha : ∥a∥ = 1) (hb : ∥b∥ = Real.sqrt 2)
  (hab : ∥a - 2 • b∥ = Real.sqrt 5) : 
  angle_between_vectors a b = π / 4 :=
sorry

end angle_between_a_b_l408_408867


namespace passenger_catches_thief_l408_408369

def passenger_catches_thief_time (v : ℕ) (passenger_speed : ℕ) (bus_speed : ℕ) (travel_time : ℕ) : ℕ :=
  let distance_covered_by_thief := v * travel_time
  let relative_speed := passenger_speed - v
  let catch_up_time := distance_covered_by_thief / relative_speed
  travel_time + catch_up_time

theorem passenger_catches_thief (v : ℕ) (h_passenger_speed : passenger_speed = 2 * v) 
                               (h_bus_speed : bus_speed = 10 * v) 
                               (h_travel_time : travel_time = 40) : 
  passenger_catches_thief_time v (2 * v) (10 * v) 40 = 80 := 
by
  rw [passenger_catches_thief_time, h_passenger_speed, h_bus_speed, h_travel_time]
  -- calculations for the result, use appropriate Lean tactics
  sorry

end passenger_catches_thief_l408_408369


namespace number_of_common_elements_l408_408169

open Set

variables (x y : Set ℤ) 

def symmetric_difference := (x ∪ y) \ (x ∩ y)

theorem number_of_common_elements
  (h₁ : Fintype.card x = 16)
  (h₂ : Fintype.card y = 18)
  (h₃ : Fintype.card (symmetric_difference x y) = 22) :
  Fintype.card (x ∩ y) = 12 :=
sorry

end number_of_common_elements_l408_408169


namespace mac_preference_count_l408_408970

theorem mac_preference_count
    (total_students : ℕ)
    (no_preference : ℕ)
    (windows_to_mac_preference : ℕ)
    (third_mac_to_both : ℕ → ℕ)
    (prefer_mac_to_windows : ℕ) : prefer_mac_to_windows = 60 :=
by
  have h1 : total_students = 210 := sorry
  have h2 : no_preference = 90 := sorry
  have h3 : windows_to_mac_preference = 40 := sorry
  have h4 : third_mac_to_both prefer_mac_to_windows = prefer_mac_to_windows / 3 := sorry
  have preferred_students := total_students - no_preference
  have h5 : preferred_students = 120 := by simp [h1, h2, h3]
  have preference_eqn : prefer_mac_to_windows + (prefer_mac_to_windows / 3) + windows_to_mac_preference = preferred_students := sorry
  -- derive equation using the assumptions and solve it
  have step1 : prefer_mac_to_windows + windows_to_mac_preference + (prefer_mac_to_windows / 3) = 120 := sorry
  have step2 : prefer_mac_to_windows + 40 + (prefer_mac_to_windows / 3) = 120 := by simp [h3]
  have step3 : (3 * prefer_mac_to_windows + 120 + prefer_mac_to_windows / 3) = 360 := sorry
  have step4 : (4 * prefer_mac_to_windows + 120) = 360 := sorry
  have step5 : (4 * prefer_mac_to_windows + 120) - 120 = 240 := sorry
  have step6 : 4 * prefer_mac_to_windows = 240 := sorry
  have step7 : prefer_mac_to_windows = 60 := by simp [step6]
  exact step7

end mac_preference_count_l408_408970


namespace probability_white_given_popped_and_fizzed_l408_408773

theorem probability_white_given_popped_and_fizzed :
  (P_white : ℚ) (P_popped_given_white : ℚ) (P_popped_given_yellow : ℚ) (P_yellow : ℚ)
  (P_fizzed_given_popped : ℚ) :
  (P_white = 3/4) →
  (P_popped_given_white = 2/3) →
  (P_popped_given_yellow = 3/4) →
  (P_yellow = 1/4) →
  (P_fizzed_given_popped = 1/4) →
  let P_white_popped_fizzed := P_white * P_popped_given_white * P_fizzed_given_popped,
      P_yellow_popped_fizzed := P_yellow * P_popped_given_yellow * P_fizzed_given_popped,
      P_popped_fizzed := P_white_popped_fizzed + P_yellow_popped_fizzed,
      P_white_given_popped_fizzed := P_white_popped_fizzed / P_popped_fizzed
  in
  P_white_given_popped_fizzed = 8/11 :=
by
  sorry

end probability_white_given_popped_and_fizzed_l408_408773


namespace angle_CFD_equals_60_l408_408997

theorem angle_CFD_equals_60 
  (O A B C D E F : Type*) 
  [circle O]
  (diam_AB : diameter O A B)
  (point_on_circle_E : on_circle O E)
  (angle_BAE_30 : angle B A E = 30)
  (tangent_BC : tangent B C)
  (tangent_ED : tangent E D)
  (intersection_C_T : intersects C tangent B tangent E)
  (intersection_D_T : intersects D tangent E AE)
  (extension_AE_F : extends_to_circle AE F) :
  angle C F D = 60 :=
sorry

end angle_CFD_equals_60_l408_408997


namespace ivan_bus_problem_l408_408432

theorem ivan_bus_problem :
  ∀ (arrive_randomly : ℝ → Prop)
    (schedule : ℕ → ℕ → ℝ)
    (takes_first_bus : ∀ (time : ℝ), ℕ)
    (A_prob : ℝ)
    (B_prob : ℝ),
    (∀ t : ℝ, take_first_bus (arrive_randomly t) = B → t < 40) →
    (∀ t : ℝ, take_first_bus (arrive_randomly t) = A → 40 ≤ t < 60) →
    (∑ t in (0..60), schedule A t) = A_prob →
    (∑ t in (0..60), schedule B t) = B_prob →
    A_prob + B_prob = 1 →
    B_prob = 2 / 3 ∧ A_prob = 1 / 3 ∧ 
    (∑ t : ℕ, A (arrive_randomly t) = arr_time t = some_frequency ∨ 
     ∑ t : ℕ, B (arrive_randomly t) = arr_time t = some_frequency) :=
begin
  sorry
end

end ivan_bus_problem_l408_408432


namespace magnitude_of_two_a_l408_408141

-- Define the vector a
def a : ℝ × ℝ × ℝ := (-1, 2, 2)

-- Define the magnitude function for a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- The penultimate goal: the magnitude of 2a which we need to show equals 6
def two_times_a : ℝ × ℝ × ℝ := (2 * a.1, 2 * a.2, 2 * a.3)

-- The formal theorem statement
theorem magnitude_of_two_a : magnitude two_times_a = 6 := by
  sorry

end magnitude_of_two_a_l408_408141


namespace solve_for_x_l408_408257

theorem solve_for_x (x : ℝ) (h : (3 + 2 / x)^(1 / 3) = 2) : x = 2 / 5 :=
by
  sorry

end solve_for_x_l408_408257


namespace lollipops_Lou_received_l408_408227

def initial_lollipops : ℕ := 42
def given_to_Emily : ℕ := 2 * initial_lollipops / 3
def kept_by_Marlon : ℕ := 4
def lollipops_left_after_Emily : ℕ := initial_lollipops - given_to_Emily
def lollipops_given_to_Lou : ℕ := lollipops_left_after_Emily - kept_by_Marlon

theorem lollipops_Lou_received : lollipops_given_to_Lou = 10 := by
  sorry

end lollipops_Lou_received_l408_408227


namespace moles_of_Mg_required_undetermined_l408_408930

-- Define the involved entities and conditions
def reaction_equation := "Mg + H2SO4 -> MgSO4 + H2"
def stoichiometry (Mg H2SO4 : ℝ) := Mg / H2SO4 = 1
def given_H2SO4 : ℝ := 1.5
def incomplete_reaction := true

-- Prove that the number of moles of Mg required is undetermined given an incomplete reaction
theorem moles_of_Mg_required_undetermined 
  (Mg : ℝ) (H2SO4 : ℝ) 
  (h_eq : reaction_equation)
  (h_stoich : stoichiometry Mg H2SO4)
  (h_given_H2SO4 : H2SO4 = 1.5)
  (h_incomplete : incomplete_reaction) : 
  ∃ x : ℝ, x = Mg :=
sorry

end moles_of_Mg_required_undetermined_l408_408930


namespace time_difference_correct_l408_408224

-- Definitions based on conditions
def malcolm_speed : ℝ := 5 -- Malcolm's speed in minutes per mile
def joshua_speed : ℝ := 7 -- Joshua's speed in minutes per mile
def race_length : ℝ := 12 -- Length of the race in miles

-- Calculate times based on speeds and race length
def malcolm_time : ℝ := malcolm_speed * race_length
def joshua_time : ℝ := joshua_speed * race_length

-- The statement that the difference in finish times is 24 minutes
theorem time_difference_correct : joshua_time - malcolm_time = 24 :=
by
  -- Proof goes here
  sorry

end time_difference_correct_l408_408224


namespace ivan_bus_problem_l408_408433

theorem ivan_bus_problem :
  ∀ (arrive_randomly : ℝ → Prop)
    (schedule : ℕ → ℕ → ℝ)
    (takes_first_bus : ∀ (time : ℝ), ℕ)
    (A_prob : ℝ)
    (B_prob : ℝ),
    (∀ t : ℝ, take_first_bus (arrive_randomly t) = B → t < 40) →
    (∀ t : ℝ, take_first_bus (arrive_randomly t) = A → 40 ≤ t < 60) →
    (∑ t in (0..60), schedule A t) = A_prob →
    (∑ t in (0..60), schedule B t) = B_prob →
    A_prob + B_prob = 1 →
    B_prob = 2 / 3 ∧ A_prob = 1 / 3 ∧ 
    (∑ t : ℕ, A (arrive_randomly t) = arr_time t = some_frequency ∨ 
     ∑ t : ℕ, B (arrive_randomly t) = arr_time t = some_frequency) :=
begin
  sorry
end

end ivan_bus_problem_l408_408433


namespace circumcircles_equality_l408_408239

noncomputable def triangle (A B C H : Type) := Prop

theorem circumcircles_equality 
  (A B C : Type) (H : Type)
  [triangle A B C H]
  (circumcircleABC : Set Type)
  (circumcircleABH : Set Type)
  (circumcircleBCH : Set Type)
  (circumcircleCAH : Set Type) :
  circumcircleABH = circumcircleABC ∧ circumcircleBCH = circumcircleABC ∧ circumcircleCAH = circumcircleABC :=
sorry

end circumcircles_equality_l408_408239


namespace find_equation_of_line_l_l408_408498

noncomputable def midpoint := sorry
noncomputable def distance := sorry

variables {P A B : Point}

-- Define point P
def P := Point.mk 2 2

-- Define circle C
def circleC (x y : ℝ) := (x - 1)^2 + y^2 = 6

-- Define the perpendicular slope condition line
def line_l_midpoint (A B : Point) : Prop :=
  let l := slope (P, A, B)
  in P = midpoint A B → line_eq (x + 2y - 6 = 0)

-- Define the condition for line l where distance length is given
def line_l_distance (A B : Point) : Prop :=
  let d := distance (P, A, B)
  in (distance P A B = 2 * sqrt 5) → 
    (line_eq (x = 2) ∨ line_eq (3 x - 4 y + 2 = 0))

-- Main theorem statement to be proved
theorem find_equation_of_line_l :
  (forall A B : Point, P = midpoint A B → line_eq (x + 2y - 6 = 0)) ∧
  (forall A B : Point, distance P A B = 2 * sqrt 5 → (line_eq (x = 2) ∨ line_eq (3 x - 4 y + 2 = 0))) :=
  by
    sorry -- proof can be completed here

end find_equation_of_line_l_l408_408498


namespace probability_f_le_zero_is_1_over_7_l408_408504

noncomputable def f (x : ℝ) : ℝ := log x / log 2

def probability_f_le_zero : ℝ :=
  let a := 1 / 2
  let b := 4
  let interval_length := b - a
  let favourable_interval := 1 - 1 / 2
  favourable_interval / interval_length

theorem probability_f_le_zero_is_1_over_7 :
  probability_f_le_zero = 1 / 7 :=
  by
    sorry

end probability_f_le_zero_is_1_over_7_l408_408504


namespace solution_set_logarithmic_equation_l408_408021

theorem solution_set_logarithmic_equation :
  {a : ℝ | log 2 (a^2 - 12 * a) = 6} = {16, -4} :=
by
  sorry

end solution_set_logarithmic_equation_l408_408021


namespace hcf_of_210_and_605_l408_408279

theorem hcf_of_210_and_605 :
  ∃ hcf : ℕ, ∃ lcm : ℕ, (lcm = Nat.lcm 210 605) ∧ (hcf = Nat.gcd 210 605) ∧ lcm = 2310 ∧ hcf = 55 := 
by
  use Nat.gcd 210 605
  use Nat.lcm 210 605
  split
  { exact rfl }
  split
  { exact rfl }
  split
  { sorry }  -- Here you would prove Nat.lcm 210 605 = 2310
  { sorry }  -- Here you would prove Nat.gcd 210 605 = 55

end hcf_of_210_and_605_l408_408279


namespace equivalent_proof_problem_l408_408747

variable (a b d e c f g h : ℚ)

def condition1 : Prop := 8 = (6 / 100) * a
def condition2 : Prop := 6 = (8 / 100) * b
def condition3 : Prop := 9 = (5 / 100) * d
def condition4 : Prop := 7 = (3 / 100) * e
def condition5 : Prop := c = b / a
def condition6 : Prop := f = d / a
def condition7 : Prop := g = e / b

theorem equivalent_proof_problem (hac1 : condition1 a)
                                 (hac2 : condition2 b)
                                 (hac3 : condition3 d)
                                 (hac4 : condition4 e)
                                 (hac5 : condition5 a b c)
                                 (hac6 : condition6 a d f)
                                 (hac7 : condition7 b e g) :
    h = f + g ↔ h = (803 / 20) * c := 
by sorry

end equivalent_proof_problem_l408_408747


namespace expression_identity_l408_408270

-- Mathematical problem statement
theorem expression_identity (a : ℝ) (h : a ≠ 0) : 
  a^3 - a^(-3) = (a - a^(-1)) * (a^2 + 1 + a^(-2)) :=
by sorry

end expression_identity_l408_408270


namespace intersection_point_l408_408700

def curve1 (x : ℝ) : ℝ := 4 * x^2 + 3 * x - 2

def curve2 (x : ℝ) : ℝ := 2 * x^3 + x^2 + 7

theorem intersection_point : curve1 (-1) = -1 ∧ curve2 (-1) = 6 :=
by
  have h1 : curve1 (-1) = 4 * (-1)^2 + 3 * (-1) - 2 := rfl
  rw [h1]
  simp
  have h2 : curve2 (-1) = 2 * (-1)^3 + (-1)^2 + 7 := rfl
  rw [h2]
  simp
  split
  case left => sorry
  case right => sorry

end intersection_point_l408_408700


namespace students_remaining_after_fourth_stop_l408_408161

variable (n : ℕ)
variable (frac : ℚ)

def initial_students := (64 : ℚ)
def fraction_remaining := (2/3 : ℚ)

theorem students_remaining_after_fourth_stop : 
  let after_first_stop := initial_students * fraction_remaining
  let after_second_stop := after_first_stop * fraction_remaining
  let after_third_stop := after_second_stop * fraction_remaining
  let after_fourth_stop := after_third_stop * fraction_remaining
  after_fourth_stop = (1024 / 81) := 
by 
  sorry

end students_remaining_after_fourth_stop_l408_408161


namespace binary_to_octal_conversion_l408_408823

-- Define the binary number 11010 in binary
def bin_value : ℕ := 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0

-- Define the octal value 32 in octal as decimal
def oct_value : ℕ := 3 * 8^1 + 2 * 8^0

-- The theorem to prove the binary equivalent of 11010 is the octal 32
theorem binary_to_octal_conversion : bin_value = oct_value :=
by
  -- Skip actual proof
  sorry

end binary_to_octal_conversion_l408_408823


namespace count_values_l408_408213

noncomputable def g (x : ℝ) : ℝ := -3 * Real.cos (Real.pi / 2 * x)

def valid_values_count : ℝ := 
  let domain := Set.Icc (-3) (3)
  (Set.filter (λ x, g (g (g x)) = g x) domain).card

theorem count_values (N : ℝ) :
  valid_values_count = N :=
sorry

end count_values_l408_408213


namespace sec_330_eq_2sqrt3_div_3_l408_408446

theorem sec_330_eq_2sqrt3_div_3 : sec 330 = 2 * sqrt 3 / 3 := by
  sorry

end sec_330_eq_2sqrt3_div_3_l408_408446


namespace find_position_of_8_over_9_l408_408136

-- Define the sequence as described
def seq : ℕ → ℚ
| 0 := 1 -- the first term is 1
| n+1 := sorry  -- Define the recursion as per the given sequence rules

-- Define a function that returns the position of a given rational number in the sequence
def position (x : ℚ) : ℕ := sorry

theorem find_position_of_8_over_9 : position (8 / 9) = 128 := 
by sorry

end find_position_of_8_over_9_l408_408136


namespace ivan_obs_correct_l408_408435

def bus_schedule (time_intervals : set ℝ) : Prop := sorry

def ivan_arrival (random_time : ℝ) (time_intervals : set ℝ) : Prop := sorry

def bus_first_comes 
  (time_intervals_A time_intervals_B : set ℝ) 
  (random_time : ℝ) 
  (arrival_freq_A arrival_freq_B : ℝ) : Prop := 
arrival_freq_A * time_intervals_A = arrival_freq_B * time_intervals_B

theorem ivan_obs_correct 
  (arrival_freq_A arrival_freq_B : ℝ) 
  (time_intervals_A time_intervals_B : set ℝ)
  (random_time : ℝ) : 
  ivan_arrival random_time (time_intervals_A ∪ time_intervals_B) →
  bus_first_comes time_intervals_A time_intervals_B random_time arrival_freq_A arrival_freq_B →
  (arrival_freq_B = 2 * arrival_freq_A) ∧ 
  ((size time_intervals_B) / (size (time_intervals_A ∪ time_intervals_B))) = 2 / 3 :=
sorry

end ivan_obs_correct_l408_408435


namespace sequence_sum_l408_408032

theorem sequence_sum :
  ∃ (r : ℝ) (x y : ℝ),
    (r = 1 / 4) ∧
    (x = 256 * r) ∧
    (y = x * r) ∧
    (x + y = 80) :=
by
  use 1 / 4, 64, 16
  split
  . exact rfl
  split
  . exact rfl
  split
  . exact rfl
  . norm_num


end sequence_sum_l408_408032


namespace f_irreducible_l408_408593

noncomputable def f (n : ℕ) (x : ℤ) : ℤ := x^n + 5 * x^(n-1) + 3

theorem f_irreducible (n : ℕ) (hn : n > 1) : Irreducible (f n) :=
sorry

end f_irreducible_l408_408593


namespace W_on_similarity_circle_Postolny_points_depend_on_figures_l408_408337

-- Given conditions
variable (F1 F2 F3 : Type) -- Types for similar figures
variable (l1 l2 l3 : Type) -- Types for lines
variable (W : Type) -- Type for the intersection point

-- Assumption that l1, l2, l3 intersect at W
axiom intersect_at_W : ∀ (l1 l2 l3 W : Type), (l1 = l2 ∧ l2 = l3) → W

-- Additional assumption that we have a similarity circle
axiom similarity_circle : (F1 F2 F3 : Type) → Type

-- Define the Postolny points
variable (J1 J2 J3 : Type) -- Types for intersection points J1, J2, J3

-- Part (a): Prove W lies on the similarity circle
theorem W_on_similarity_circle (F1 F2 F3 : Type) (l1 l2 l3 : Type) (W : Type)
  (intersect : ∀ (l1 l2 l3 W : Type), (l1 = l2 ∧ l2 = l3) → W)
  (circle : (F1 F2 F3 : Type) → Type) : W ∈ circle F1 F2 F3 := by
  sorry

-- Part (b): Prove that J1, J2, J3 depend only on the figures F1, F2, F3
theorem Postolny_points_depend_on_figures (F1 F2 F3 : Type) (l1 l2 l3 : Type) (W : Type)
  (intersect : ∀ (l1 l2 l3 W : Type), (l1 = l2 ∧ l2 = l3) → W)
  (circle : (F1 F2 F3 : Type) → Type) (J1 J2 J3 : Type)
  (points_on_circle : ∀ (l1 l2 l3 : Type) (J1 J2 J3 W : Type), ((J1 ≠ W) ∧ (J2 ≠ W) ∧ (J3 ≠ W)) → (J1 ∈ circle F1 F2 F3) ∧ (J2 ∈ circle F1 F2 F3) ∧ (J3 ∈ circle F1 F2 F3)) : 
  (J1 ∈ circle F1 F2 F3) ∧ (J2 ∈ circle F1 F2 F3) ∧ (J3 ∈ circle F1 F2 F3) := by
  sorry

end W_on_similarity_circle_Postolny_points_depend_on_figures_l408_408337


namespace unique_solution_of_system_of_equations_l408_408624
open Set

variable {α : Type*} (A B X : Set α)

theorem unique_solution_of_system_of_equations :
  (X ∩ (A ∪ B) = X) ∧
  (A ∩ (B ∪ X) = A) ∧
  (B ∩ (A ∪ X) = B) ∧
  (X ∩ A ∩ B = ∅) →
  (X = (A \ B) ∪ (B \ A)) :=
by
  sorry

end unique_solution_of_system_of_equations_l408_408624


namespace distance_between_trees_l408_408784

def yard_length : ℕ := 414
def number_of_trees : ℕ := 24

theorem distance_between_trees : yard_length / (number_of_trees - 1) = 18 := 
by sorry

end distance_between_trees_l408_408784


namespace max_two_integers_abs_leq_50_l408_408461

theorem max_two_integers_abs_leq_50
  (a b c : ℤ) (h_a : a > 100) :
  ∀ {x1 x2 x3 : ℤ}, (abs (a * x1^2 + b * x1 + c) ≤ 50) →
                    (abs (a * x2^2 + b * x2 + c) ≤ 50) →
                    (abs (a * x3^2 + b * x3 + c) ≤ 50) →
                    false :=
sorry

end max_two_integers_abs_leq_50_l408_408461


namespace paul_reading_time_l408_408234

theorem paul_reading_time :
  (∀ (n_books_per_week : Nat) (pages_per_book : Nat) (reading_rate : Nat) (num_weeks : Nat),
    n_books_per_week = 10 →
    pages_per_book = 300 →
    reading_rate = 50 →
    num_weeks = 9 →
    (n_books_per_week * pages_per_book) * num_weeks / reading_rate = 540) :=
begin
  intros n_books_per_week pages_per_book reading_rate num_weeks h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  norm_num,
end

end paul_reading_time_l408_408234


namespace factor_theorem_solutions_l408_408070

-- Defining the polynomial
def polynomial (x : ℝ) : ℝ := 8 * x^2 + 18 * x - 10

-- Lean statement to prove the factor condition
theorem factor_theorem (t : ℝ) : (∃ f : ℝ → ℝ, polynomial = λ x, (x - t) * f x) ↔ (polynomial t = 0) :=
by sorry

-- Expected result
theorem solutions : ∃ t1 t2 : ℝ, (t1 = 1/2 ∧ t2 = -5/2 ∧ polynomial t1 = 0 ∧ polynomial t2 = 0) :=
by sorry

end factor_theorem_solutions_l408_408070


namespace same_terminal_side_l408_408376

theorem same_terminal_side (k : ℤ) : 
  ∃ (α : ℤ), α = k * 360 + 330 ∧ (α = 510 ∨ α = 150 ∨ α = -150 ∨ α = -390) :=
by
  sorry

end same_terminal_side_l408_408376


namespace pentagon_product_condition_l408_408106

theorem pentagon_product_condition :
  ∃ (a b c d e : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ a + b + c + d + e = 1 ∧
  ∃ (a' b' c' d' e' : ℝ), 
    (a', b', c', d', e') ∈ {perm | perm = (a, b, c, d, e) ∨ perm = (b, c, d, e, a) ∨ perm = (c, d, e, a, b) ∨ perm = (d, e, a, b, c) ∨ perm = (e, a, b, c, d)} ∧
    (a'*b' ≤ 1/9 ∧ b'*c' ≤ 1/9 ∧ c'*d' ≤ 1/9 ∧ d'*e' ≤ 1/9 ∧ e'*a' ≤ 1/9) := sorry

end pentagon_product_condition_l408_408106


namespace find_x_average_is_3_l408_408101

theorem find_x_average_is_3 (x : ℝ) (h : (2 + 4 + 1 + 3 + x) / 5 = 3) : x = 5 :=
sorry

end find_x_average_is_3_l408_408101


namespace f_value_third_quadrant_l408_408091

-- Definition of f(α)
def f (α : ℝ) : ℝ :=
  (sin (π - α) * cos (2 * π - α) * tan (-α + π)) / (-tan (-α - π) * cos (π / 2 - α))

-- Conditions: α is in the third quadrant and cos(α - 3π/2) = 1/5
variables (α : ℝ) (h_quadrant : π < α ∧ α < 3 * π) (h_cos_cond : cos(α - 3 * π / 2) = 1 / 5)

-- Lemma for the simplified form of f(α)
lemma simpl_f_eq_neg_cos : f(α) = -cos(α) :=
sorry

-- Theorem for the calculated value of f(α) under the given condition
theorem f_value_third_quadrant : f(α) = 2 * sqrt 6 / 5 :=
begin
  have h_sin : sin α = -1 / 5,
  {
    sorry -- Proof of sin(α) = -1/5
  },
  have h_cos : cos α = -2 * sqrt 6 / 5,
  {
    sorry -- Proof of cos(α) = -2 * sqrt 6 / 5
  },
  rw simpl_f_eq_neg_cos,
  rw h_cos,
  norm_num,
end

end f_value_third_quadrant_l408_408091


namespace diagonal_passes_through_cubes_l408_408748

/-- A 200×400×500 rectangular solid is made by gluing together 1×1×1 cubes.
An internal diagonal of this solid passes through the interiors of exactly 900 of these 1×1×1 cubes. -/
theorem diagonal_passes_through_cubes :
  let l := 200;
      m := 400;
      n := 500 in
  let gcd := Nat.gcd in
  l + m + n - gcd l m - gcd m n - gcd n l + gcd l (gcd m n) = 900 :=
by
  sorry

end diagonal_passes_through_cubes_l408_408748


namespace cosine_identity_l408_408113

theorem cosine_identity (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = Real.sqrt 3 / 2) :
  Real.cos (Real.pi / 3 - α) = Real.sqrt 3 / 2 := 
by
  sorry

end cosine_identity_l408_408113


namespace range_alpha_plus_beta_div_2_l408_408861

theorem range_alpha_plus_beta_div_2 (alpha beta : ℝ) 
  (h1 : -real.pi / 2 < alpha) 
  (h2 : alpha < beta) 
  (h3 : beta < real.pi / 2) : 
  -real.pi / 2 < (alpha + beta) / 2 ∧ (alpha + beta) / 2 < 0 := 
by 
  sorry

end range_alpha_plus_beta_div_2_l408_408861


namespace willowbrook_team_combinations_l408_408664

theorem willowbrook_team_combinations :
  let girls := 5
  let boys := 5
  let choose_three (n : ℕ) := n.choose 3
  let team_count := choose_three girls * choose_three boys
  team_count = 100 :=
by
  let girls := 5
  let boys := 5
  let choose_three (n : ℕ) := n.choose 3
  let team_count := choose_three girls * choose_three boys
  have h1 : choose_three girls = 10 := by sorry
  have h2 : choose_three boys = 10 := by sorry
  have h3 : team_count = 10 * 10 := by sorry
  exact h3

end willowbrook_team_combinations_l408_408664


namespace seq_a6_l408_408138

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 3 * a n - 2

theorem seq_a6 (a : ℕ → ℕ) (h : seq a) : a 6 = 1 :=
by
  sorry

end seq_a6_l408_408138


namespace calc_3_pow_6_mul_4_pow_6_l408_408387

theorem calc_3_pow_6_mul_4_pow_6 : (3^6) * (4^6) = 2985984 :=
by 
  sorry

end calc_3_pow_6_mul_4_pow_6_l408_408387


namespace series_equals_value_l408_408079

noncomputable def series_value : ℝ :=
  3^(∑' n : ℕ in finset.range (1000), (n + 1) / 3^(n + 1))

theorem series_equals_value : series_value = 3^(3 / 4) :=
by
  -- the proof goes here
  sorry

end series_equals_value_l408_408079


namespace compute_expression_l408_408393

theorem compute_expression : 9 * (1 / 13) * 26 = 18 :=
by
  sorry

end compute_expression_l408_408393


namespace f_f_neg_pi_div_3_eq_one_l408_408503

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log (1 / 2) else Real.cos x

theorem f_f_neg_pi_div_3_eq_one : f (f (-Real.pi / 3)) = 1 := by
  -- This is the placeholder for the proof
  sorry

end f_f_neg_pi_div_3_eq_one_l408_408503


namespace students_failed_to_get_degree_l408_408619

/-- 
Out of 1,500 senior high school students, 70% passed their English exams,
80% passed their Mathematics exams, and 65% passed their Science exams.
To get their degree, a student must pass in all three subjects.
Assume independence of passing rates. This Lean proof shows that
the number of students who failed to get their degree is 954.
-/
theorem students_failed_to_get_degree :
  let total_students := 1500
  let p_english := 0.70
  let p_math := 0.80
  let p_science := 0.65
  let p_all_pass := p_english * p_math * p_science
  let students_all_pass := p_all_pass * total_students
  total_students - students_all_pass = 954 :=
by
  sorry

end students_failed_to_get_degree_l408_408619


namespace find_c_deg3_l408_408820

-- Define the polynomials f and g.
def f (x : ℚ) : ℚ := 2 - 10 * x + 4 * x^2 - 5 * x^3 + 7 * x^4
def g (x : ℚ) : ℚ := 5 - 3 * x - 8 * x^3 + 11 * x^4

-- The statement that needs proof.
theorem find_c_deg3 (c : ℚ) : (∀ x : ℚ, f x + c * g x ≠ 0 → f x + c * g x = 2 - 10 * x + 4 * x^2 - 5 * x^3 - c * 8 * x^3) ↔ c = -7 / 11 :=
sorry

end find_c_deg3_l408_408820


namespace problem_statements_l408_408022

theorem problem_statements :
  (∀ x, ∃ c, y = c * x^2) ∧
  (∃ x, (x ∈ Ioo (1/e) 1) ∧ (x + log x = 0)) ∧
  (∀ (A B C D : Point) (λ : ℝ),
    (vec AD = λ * (vec AB / ∥vec AB ∥ + vec AC / ∥vec AC∥) →
    D is_on bisector of ∠ BAC)) ∧
  (¬ (∀ (r : ℝ), (|r| < 1 → weaker linear correlation between two variables))) := 
begin
  sorry
end

end problem_statements_l408_408022


namespace right_triangle_BC_length_l408_408171

theorem right_triangle_BC_length
  (A B C : Type)
  (angle_A_eq_90 : angle A = 90)
  (cos_B_eq_4_div_5 : cos B = 4 / 5)
  (AB_eq_40 : AB = 40) :
  BC = 24 :=
by
  sorry

end right_triangle_BC_length_l408_408171


namespace find_d_l408_408158

-- Define the base-10 values of mnp, nmp, mmp, and nnp using digits m, n, p
def mnp (m n p : ℕ) := 100 * m + 10 * n + p
def nmp (m n p : ℕ) := 100 * n + 10 * m + p
def mmp (m n p : ℕ) := 100 * m + 10 * m + p
def nnp (m n p : ℕ) := 100 * n + 10 * n + p

-- Given conditions:
variable (m n p d : ℕ)
variable h1 : mnp m n p - nmp m n p = 180
variable h2 : mmp m n p - nnp m n p = d

-- The proof statement
theorem find_d (h : h1) : d = 220 :=
sorry

end find_d_l408_408158


namespace sum_alpha_beta_eq_pi_over_4_l408_408292

theorem sum_alpha_beta_eq_pi_over_4 (a : ℝ) (h : a > 2) (α β : ℝ)
  (h1 : (α ∈ Ioo (-π/2) (π/2)))
  (h2 : (β ∈ Ioo (-π/2) (π/2)))
  (h_roots : (∀ x : ℝ, (x^2 + 3 * a * x + 3 * a + 1 = 0) ↔ (x = tan α ∨ x = tan β))) :
  α + β = π / 4 :=
sorry

end sum_alpha_beta_eq_pi_over_4_l408_408292


namespace average_annual_growth_rate_l408_408629

theorem average_annual_growth_rate (income_2014 income_2016 : ℝ) (h2014 : income_2014 = 2600) (h2016 : income_2016 = 5096) :
  ∃ x : ℝ, income_2016 = income_2014 * (1 + x)^2 ∧ 1 + x = 1.4 :=
by {
  use 0.4,
  split,
  {
    rw [h2014, h2016],
    norm_num,
  },
  {
    norm_num,
  }
}

end average_annual_growth_rate_l408_408629


namespace exp_gt_one_iff_a_gt_one_l408_408515

theorem exp_gt_one_iff_a_gt_one (a : ℝ) : 
  (∀ x : ℝ, 0 < x → a^x > 1) ↔ a > 1 :=
by
  sorry

end exp_gt_one_iff_a_gt_one_l408_408515


namespace angle_B_is_pi_third_max_dot_product_l408_408960

/-- Given conditions in triangle ABC, prove the measure of angle B is π/3 --/
theorem angle_B_is_pi_third (a b c : ℝ) (A B C : ℝ)
  (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π)
  (h7 : a = sin A) (h8 : b = sin B) (h9 : c = sin C)
  (h10 : (2 * a - c) * cos B = b * cos C) :
  B = π / 3 := sorry

/-- Given vectors m and n, find the maximum value of dot product --
theorem max_dot_product (A : ℝ) (hA : A > 0 ∧ A ≤ π / 2) :
  let m := (sin A, cos (2 * A))
      n := (6 : ℝ, 1 : ℝ) in max (m.1 * n.1 + m.2 * n.2) 5 := sorry

end angle_B_is_pi_third_max_dot_product_l408_408960


namespace units_digit_of_M_is_1_l408_408999

def Q (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  if units = 0 then 0 else tens / units

def T (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem units_digit_of_M_is_1 (a b : ℕ) (h₁ : 1 ≤ a) (h₂ : b ≤ 9) (h₃ : 10*a + b = Q (10*a + b) + T (10*a + b)) :
  b = 1 :=
by
  sorry

end units_digit_of_M_is_1_l408_408999


namespace s_point_condition_l408_408589

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x)

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 2 * a * x
noncomputable def g_prime (a : ℝ) (x : ℝ) : ℝ := 1 / x

theorem s_point_condition (a : ℝ) (x₀ : ℝ) (h_f_g : f a x₀ = g a x₀) (h_f'g' : f_prime a x₀ = g_prime a x₀) :
  a = 2 / Real.exp 1 :=
by
  sorry

end s_point_condition_l408_408589


namespace inequality_relationship_cannot_be_established_l408_408468

variable (a b : ℝ)
variable (h : a < b ∧ b < 0)

theorem inequality_relationship_cannot_be_established :
  ¬ (1 / (a - b) > 1 / a) :=
by sorry

end inequality_relationship_cannot_be_established_l408_408468


namespace prob_all_pass_prob_at_least_one_pass_most_likely_event_l408_408544

noncomputable def probability_A := 2 / 5
noncomputable def probability_B := 3 / 4
noncomputable def probability_C := 1 / 3
noncomputable def prob_none_pass := (1 - probability_A) * (1 - probability_B) * (1 - probability_C)
noncomputable def prob_one_pass := 
  (probability_A * (1 - probability_B) * (1 - probability_C)) +
  ((1 - probability_A) * probability_B * (1 - probability_C)) +
  ((1 - probability_A) * (1 - probability_B) * probability_C)
noncomputable def prob_two_pass := 
  (probability_A * probability_B * (1 - probability_C)) +
  (probability_A * (1 - probability_B) * probability_C) +
  ((1 - probability_A) * probability_B * probability_C)

-- Prove that the probability that all three candidates pass is 1/10
theorem prob_all_pass : probability_A * probability_B * probability_C = 1 / 10 := by
  sorry

-- Prove that the probability that at least one candidate passes is 9/10
theorem prob_at_least_one_pass : 1 - prob_none_pass = 9 / 10 := by
  sorry

-- Prove that the most likely event of passing is exactly one candidate passing with probability 5/12
theorem most_likely_event : prob_one_pass > prob_two_pass ∧ prob_one_pass > probability_A * probability_B * probability_C ∧ prob_one_pass > prob_none_pass ∧ prob_one_pass = 5 / 12 := by
  sorry

end prob_all_pass_prob_at_least_one_pass_most_likely_event_l408_408544


namespace ellipse_a_plus_k_l408_408791

theorem ellipse_a_plus_k (f1 f2 p : Real × Real) (a b h k : Real) :
  f1 = (2, 0) →
  f2 = (-2, 0) →
  p = (5, 3) →
  (∀ x y, ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1) →
  a > 0 →
  b > 0 →
  h = 0 →
  k = 0 →
  a = (3 * Real.sqrt 2 + Real.sqrt 58) / 2 →
  a + k = (3 * Real.sqrt 2 + Real.sqrt 58) / 2 :=
by
  intros
  sorry

end ellipse_a_plus_k_l408_408791


namespace complex_power_l408_408116

theorem complex_power :
  ∀ i : ℂ, i^2 = -1 → (1 + i) / (1 - i)) ^ 2018 = -1 :=
begin
  intro i,
  intro hi,
  sorry
end

end complex_power_l408_408116


namespace probabilityOfModulusSqrt2_l408_408920

-- Define the sets A and B according to the problem
def isElementOfA (z : ℂ) : Prop :=
  ∃ n : ℕ, z = (finset.range (n + 1)).sum (λ k, i^k)

def isElementOfB (ω : ℂ) : Prop :=
  ∃ z1 z2 : ℂ, isElementOfA z1 ∧ isElementOfA z2 ∧ ω = z1 * z2

-- Define the property that an element has a modulus of √2
def hasModulusSqrt2 (z : ℂ) : Prop :=
  complex.abs z = real.sqrt 2

-- Define the set of elements in B with modulus √2
def elementsOfBWithModulusSqrt2 : set ℂ :=
  {ω | isElementOfB ω ∧ hasModulusSqrt2 ω}

-- Calculate the probability
def probabilityModulusSqrt2 : ℚ :=
  (fintype.card elementsOfBWithModulusSqrt2) / (fintype.card {ω | isElementOfB ω})

-- The proof statement
theorem probabilityOfModulusSqrt2 :
  probabilityModulusSqrt2 = 2 / 7 :=
sorry

end probabilityOfModulusSqrt2_l408_408920


namespace find_x_l408_408086

variable (f g : ℝ → ℝ)

noncomputable def α (x : ℝ) : ℝ := 4 * x + 9
noncomputable def β (x : ℝ) : ℝ := 7 * x + 6

theorem find_x : ∃ x : ℝ, α (β x) = 4 ∧ x = -29 / 28 :=
by
  exists -29 / 28
  split
  -- Proof omitted
  sorry

end find_x_l408_408086


namespace problem1_problem2_l408_408632

theorem problem1 (x : ℝ) (h : (2*x) * (2*x - 1) * (2*x - 2) * (2*x - 3) = 60 * x * (x - 1) * (x - 2)) : x = 3 :=
sorry

theorem problem2 (n : ℕ) (h : nat.choose (n + 3) 2 = (nat.choose (n + 1) 2 + nat.choose (n + 1) 1 + nat.choose n 2)) : n = 4 :=
sorry

end problem1_problem2_l408_408632


namespace factorize_expression_l408_408068

theorem factorize_expression (a : ℝ) : 
  (2 * a + 1) * a - 4 * a - 2 = (2 * a + 1) * (a - 2) :=
by 
  -- proof is skipped with sorry
  sorry

end factorize_expression_l408_408068


namespace units_digit_sum_squares_first_2023_odd_integers_l408_408721

theorem units_digit_sum_squares_first_2023_odd_integers :
  let pattern := [1, 9, 5, 9, 1] in
  let full_cycles := 2023 / 5 in
  let remaining := 2023 % 5 in
  (full_cycles * (pattern.foldl (λ acc x => acc + x) 0) + (remaining * pattern.foldl (λ acc x => acc + if x = (remaining - 1).natAbs then x else 0) 0)) % 10 = 5 :=
by
  sorry

end units_digit_sum_squares_first_2023_odd_integers_l408_408721


namespace alpha_beta_inequality_l408_408083

theorem alpha_beta_inequality (α β : ℝ) :
  (∃ (k : ℝ), ∀ (x y : ℝ), 0 < x → 0 < y → x^α * y^β < k * (x + y)) ↔ (0 ≤ α ∧ 0 ≤ β ∧ α + β = 1) :=
by
  sorry

end alpha_beta_inequality_l408_408083


namespace power_series_interval_of_convergence_l408_408843

theorem power_series_interval_of_convergence (x : ℝ) :
  (∑ n in Nat, (-1)^n * (x^n / ((n + 1) * 2^n))).converges ↔ -2 < x ∧ x ≤ 2 :=
sorry

end power_series_interval_of_convergence_l408_408843


namespace sum_of_coeffs_is_32_sum_of_all_coeffs_is_neg1_no_constant_term_coeff_x3_is_80_l408_408240

def poly (x : ℝ) : ℝ := (1 / x - 2 * x) ^ 5

theorem sum_of_coeffs_is_32 : (∑ k in range 6, (coeff poly k)) = 32 :=
sorry

theorem sum_of_all_coeffs_is_neg1 : poly 1 = -1 :=
sorry

theorem no_constant_term : ¬ ∃ a, poly a = a :=
sorry

theorem coeff_x3_is_80 : coeff poly 3 = 80 :=
sorry

end sum_of_coeffs_is_32_sum_of_all_coeffs_is_neg1_no_constant_term_coeff_x3_is_80_l408_408240


namespace specially_monotonous_count_l408_408401

def specially_monotonous (n : ℕ) : Prop :=
(n < 10) ∨ 
(∃ (d : list ℕ) (h1 : d.length = n) (h2 : n ≤ 5), 
  (∀ (i j : ℕ) (h3 : i < j) (h4 : j < d.length), d.nth_le i h3 < d.nth_le j h4 ∨ d.nth_le i h3 > d.nth_le j h4) ∨ 
  (∃ (zero_index : ℕ) (h5 : zero_index < n), d.zero_index = 0 ∧ 
   (∀ (i : ℕ) (h6 : i < zero_index), d.nth_le i h6 ≠ 0) ∧ 
   (∀ (j k : ℕ) (h7 : j < k) (h8 : k < zero_index), d.nth_le j h7 < d.nth_le k h8 ∨ d.nth_le j h7 > d.nth_le k h8)))

theorem specially_monotonous_count : 
  card {n : ℕ | specially_monotonous n } = 1519 := 
sorry

end specially_monotonous_count_l408_408401


namespace complex_modulus_l408_408424

theorem complex_modulus :
  abs ((7 - 4*complex.I) * (3 + 11*complex.I)) = Real.sqrt 8450 :=
by
  sorry

end complex_modulus_l408_408424


namespace decode_encrypted_message_l408_408640

def group_digits (digits : list ℕ) (groups : list (list ℕ)) : Prop :=
  -- ensures digits are divided into non-overlapping groups.
  list.all_disjoint groups ∧ list.perm digits (list.join groups)

noncomputable def encode_message (n : ℕ) : string :=
  let alphabet := "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ".data in
  alphabet.get ⟨n % alphabet.length, sorry⟩.toString

def decode_message (numbers : list ℕ) (table : list ℕ × list string) : string :=
  let mapping := (table.fst.zip table.snd).to_map in
  numbers.map (λ n, (mapping.find n).get_or_else "").as_string
  
theorem decode_encrypted_message :
  ∀ digits groups message table,
    group_digits digits groups → encode_message (decimal_of_list digits) = message →
    decode_message [8, 7, 3, 1, 4, 6, 5, 0, 7, 3, 8, 1] table = "НАУКА" :=
by
  intros digits groups message table Hgroup Hencode
  sorry

end decode_encrypted_message_l408_408640


namespace largest_common_term_l408_408076

theorem largest_common_term (a b : ℕ) :
  ∃ (n : ℕ), a = 976 ∧ n < 1000 ∧ (a = 1 + 3 * (n - 1) ∧ b = 5 + 9 * (n - 1)) :=
begin
  sorry
end

end largest_common_term_l408_408076


namespace sequence_sum_l408_408033

theorem sequence_sum :
  ∃ (r : ℝ) (x y : ℝ),
    (r = 1 / 4) ∧
    (x = 256 * r) ∧
    (y = x * r) ∧
    (x + y = 80) :=
by
  use 1 / 4, 64, 16
  split
  . exact rfl
  split
  . exact rfl
  split
  . exact rfl
  . norm_num


end sequence_sum_l408_408033


namespace sec_330_eq_2_sqrt_3_over_3_l408_408440

theorem sec_330_eq_2_sqrt_3_over_3 :
  (sec (330 * real.pi / 180) = 2 * real.sqrt 3 / 3) :=
by
  have h1: cos (330 * real.pi / 180) = cos (30 * real.pi / 180),
  {
    have h2: cos (330 * real.pi / 180) = cos ((360 - 30) * real.pi / 180), by norm_num,
    have h3: cos ((360 - 30) * real.pi / 180) = cos (-30 * real.pi / 180), by norm_num,
    have h4: cos (-30 * real.pi / 180) = cos (30 * real.pi / 180), by norm_num,
    exact eq.trans (eq.trans h2 h3) h4,
  },
  rw [sec, h1],
  have h_cos_30: cos (30 * real.pi / 180) = real.sqrt 3 / 2, by norm_num,
  rw h_cos_30,
  norm_num

end sec_330_eq_2_sqrt_3_over_3_l408_408440


namespace minimum_value_is_4_l408_408585

noncomputable def minimum_value (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z) : ℝ :=
  real.sqrt ((x^2 + y^2 + z^2) * (4 * x^2 + y^2 + z^2)) / (x * y * z)

theorem minimum_value_is_4 (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z) : (minimum_value x y z h) = 4 :=
sorry

end minimum_value_is_4_l408_408585


namespace find_b_k_plus_3_l408_408656

theorem find_b_k_plus_3 (b_k b_k1 b_k2 b_k3 b_k4 : ℕ) 
  (h1 : 874 = b_k * 6! + b_k1 * 5! + b_k2 * 4! + b_k3 * 3! + b_k4 * 2!) 
  (h2 : b_k1 = b_k + 1)
  (h3 : b_k2 = b_k + 2)
  (h4 : b_k3 = b_k + 3)
  (h5 : b_k4 = b_k + 4) : 
  b_k3 = 4 := 
sorry

end find_b_k_plus_3_l408_408656


namespace ratio_of_ages_after_two_years_l408_408356

theorem ratio_of_ages_after_two_years (S M : ℕ) (h1 : S = 23) (h2 : M = S + 25) :
  (M + 2) / (S + 2) = 2 :=
begin
  sorry
end

end ratio_of_ages_after_two_years_l408_408356


namespace arithmetic_sequence_ratio_q_l408_408103

theorem arithmetic_sequence_ratio_q :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ), 
    (0 < q) →
    (S 2 = 3 * a 2 + 2) →
    (S 4 = 3 * a 4 + 2) →
    (q = 3 / 2) :=
by
  sorry

end arithmetic_sequence_ratio_q_l408_408103


namespace blocks_per_color_l408_408783

theorem blocks_per_color (total_blocks : ℕ) (total_colors : ℕ) (h1 : total_blocks = 49) (h2 : total_colors = 7) : total_blocks / total_colors = 7 :=
by 
  rw [h1, h2]
  norm_num
  -- Alternatively, you can use sorry instead of the proof
  -- sorry

end blocks_per_color_l408_408783


namespace correlation_statements_correct_l408_408307

variable (x y : ℝ) (r : ℝ)

theorem correlation_statements_correct :
  (r > 0 → (∀ x₁ x₂, x₁ < x₂ → y x₁ < y x₂)) ∧
  (r < 0 → (∀ x₁ x₂, x₁ < x₂ → y x₁ > y x₂)) ∧
  ((r = 1 ∨ r = -1) → ∃ f : ℝ → ℝ, ∀ x, y x = f x) →
  ∀ (s₁ s₃ : Bool), (s₁ ↔ s₃) :=
by
  sorry

end correlation_statements_correct_l408_408307


namespace hattie_jumps_l408_408524

theorem hattie_jumps (H : ℝ) (h1 : Lorelei_jumps1 = (3/4) * H)
  (h2 : Hattie_jumps2 = (2/3) * H)
  (h3 : Lorelei_jumps2 = (2/3) * H + 50)
  (h4 : H + Lorelei_jumps1 + Hattie_jumps2 + Lorelei_jumps2 = 605) : H = 180 :=
by
  sorry

noncomputable def Lorelei_jumps1 (H : ℝ) := (3/4) * H
noncomputable def Hattie_jumps2 (H : ℝ) := (2/3) * H
noncomputable def Lorelei_jumps2 (H : ℝ) := (2/3) * H + 50

end hattie_jumps_l408_408524


namespace perimeter_AKM_l408_408192

theorem perimeter_AKM 
  (A B C K M O : Type*) 
  [Triangle ABC]
  (hAC : AC = 1)
  (hAB : AB = 2)
  (hO : is_angle_bisector_intersection O ABC)
  (h_parallel : parallel_segment_through O BC AC = K ∧ parallel_segment_through O BC AB = M)
  : perimeter AKM = 3 := sorry

end perimeter_AKM_l408_408192


namespace sequence_x_y_sum_l408_408045

theorem sequence_x_y_sum (r : ℝ) (x y : ℝ)
  (h₁ : r = 1 / 4)
  (h₂ : x = 256 * r)
  (h₃ : y = x * r) :
  x + y = 80 :=
by
  sorry

end sequence_x_y_sum_l408_408045


namespace parabola_equation_find_k_l408_408895

/-- The equation of the parabola is y^2 = 8x -/
theorem parabola_equation 
  (vertex : (ℝ × ℝ)) (focus_on_x_axis : Bool)
  (point_distance : ℝ) (A : (ℝ × ℝ)) 
  (focusDistance : A.1 = 4 ∧ A.2 = m ∧ m_d = 6) : 
  ∃ p, y^2 = 8 * x := 
sorry

/-- The value of k is 2 given the conditions -/
theorem find_k 
  (parabola_eq : y^2 = 8x) 
  (intersection_line : y = k * x - 2) 
  (midpoint : midpoint_x = 2) 
  (intersection_pts : ∃ A B : (ℝ × ℝ), A ≠ B) : 
  k = 2 := 
sorry

end parabola_equation_find_k_l408_408895


namespace angle_between_vectors_l408_408492

variables {V : Type*} [inner_product_space ℝ V]

theorem angle_between_vectors (a b : V) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : ⟪a - (2 : ℝ) • b, a⟫ = 0) (h2 : ⟪b - (2 : ℝ) • a, b⟫ = 0) :
  real.angle a b = real.pi / 3 :=
by sorry

end angle_between_vectors_l408_408492


namespace find_three_digit_numbers_l408_408838
open Nat

theorem find_three_digit_numbers (n : ℕ) (h1 : 100 ≤ n) (h2 : n < 1000) (h3 : ∀ (k : ℕ), n^k % 1000 = n % 1000) : n = 625 ∨ n = 376 :=
sorry

end find_three_digit_numbers_l408_408838


namespace sequence_sum_l408_408051

noncomputable def r : ℝ := 1 / 4
def t₀ : ℝ := 4096
def t₁ : ℝ := t₀ * r
def t₂ : ℝ := t₁ * r
def x : ℝ := t₂ * r
def y : ℝ := x * r

theorem sequence_sum : x + y = 80 := by
  -- The proof will go here
  sorry

end sequence_sum_l408_408051


namespace ratio_of_lost_diaries_to_total_diaries_l408_408614

theorem ratio_of_lost_diaries_to_total_diaries 
  (original_diaries : ℕ)
  (bought_diaries : ℕ)
  (current_diaries : ℕ)
  (h1 : original_diaries = 8)
  (h2 : bought_diaries = 2 * original_diaries)
  (h3 : current_diaries = 18) :
  (original_diaries + bought_diaries - current_diaries) / gcd (original_diaries + bought_diaries - current_diaries) (original_diaries + bought_diaries) 
  = 1 / 4 :=
by
  sorry

end ratio_of_lost_diaries_to_total_diaries_l408_408614


namespace four_digit_numbers_count_l408_408529

theorem four_digit_numbers_count :
  let digit_set := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let first_digit_choices := {5, 6, 7, 8, 9}
  let middle_digit_pairs := { (a, b) | a ∈ digit_set ∧ b ∈ digit_set ∧ a * b > 10 }
  let last_digit_choices := digit_set
  first_digit_choices.card * middle_digit_pairs.card * last_digit_choices.card = 2700 :=
by
  sorry

end four_digit_numbers_count_l408_408529


namespace min_det_is_neg_six_l408_408269

-- Define the set of possible values for a, b, c, d
def values : List ℤ := [-1, 1, 2]

-- Define the determinant function for a 2x2 matrix
def det (a b c d : ℤ) : ℤ := a * d - b * c

-- State the theorem that the minimum value of the determinant is -6
theorem min_det_is_neg_six :
  ∃ (a b c d : ℤ), a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ 
  (∀ (a' b' c' d' : ℤ), a' ∈ values → b' ∈ values → c' ∈ values → d' ∈ values → det a b c d ≤ det a' b' c' d') ∧ det a b c d = -6 :=
by
  sorry

end min_det_is_neg_six_l408_408269


namespace area_of_triangle_formed_by_lines_l408_408708

theorem area_of_triangle_formed_by_lines :
  let L1 := λ x : ℝ, 3*x - 6
  let L2 := λ x : ℝ, -4*x + 24
  let intersection_xy : ℝ × ℝ := (30/7, 48/7)
  let intercept_y1 : ℝ × ℝ := (0, -6)
  let intercept_y2 : ℝ × ℝ := (0, 24)
  let area_triangle : ℝ := (1 / 2) * 30 * (30 / 7)
  L1 (30/7) = 48/7 ∧ L2 (30/7) = 48/7 →
  area_triangle = 450 / 7 := 
  by
    intros
    sorry

end area_of_triangle_formed_by_lines_l408_408708


namespace minimum_k_unique_weights_l408_408875

noncomputable def min_weights (n : ℕ) : ℕ :=
  Nat.find (λ m, (n : ℕ) ≤ (3^m - 1) / 2 ∧ (3^(m - 1) - 1) / 2 < n)

theorem minimum_k (n : ℕ) (k :ℕ) (h: k = min_weights n) : 
  (3^(k - 1) - 1) / 2 < n ∧ n ≤ (3^k - 1) / 2 := 
sorry

noncomputable def f_weights (n : ℕ) : list ℕ := 
let m := min_weights n in [seq 0 m].map (λ i, 3^i)

theorem unique_weights (n : ℕ) :
   ∃! (weights : list ℕ), weights = f_weights n ↔ n = (3^min_weights n - 1) / 2 :=
sorry

end minimum_k_unique_weights_l408_408875


namespace right_triangle_acute_angle_l408_408969

theorem right_triangle_acute_angle (A B C : ℝ) (h_triangle: ∠A + ∠B + ∠C = 180°) (h_right: ∠A = 90°) (h_acute: ∠B = 30°) : ∠C = 60° :=
sorry

end right_triangle_acute_angle_l408_408969


namespace magnitude_product_complex_l408_408416

theorem magnitude_product_complex :
  let z1 := Complex.mk 7 (-4)
  let z2 := Complex.mk 3 11
  Complex.abs (z1 * z2) = Real.sqrt 8450 :=
by
  sorry

end magnitude_product_complex_l408_408416


namespace polynomial_value_at_2_l408_408016

noncomputable def P (x : ℚ) := x^4 - 8 * x^2 + 4

theorem polynomial_value_at_2 :
  (P(2) = -12) :=
by
  sorry

end polynomial_value_at_2_l408_408016


namespace rectangle_area_correct_l408_408968

noncomputable def RectangleArea (x1 x2 y1 y2 : ℝ) : ℝ :=
  (abs (x2 - x1)) * (abs (y2 - y1))

theorem rectangle_area_correct :
  ∃ y : ℝ, RectangleArea (-8) 1 y (-7) = 72 ∧ y = 1 :=
by
  use 1
  have h_length : abs (1 - (-8)) = 9, by norm_num
  have h_width : abs (1 - (-7)) = 8, by norm_num
  rw [RectangleArea, h_length, h_width]
  norm_num
  split
  { rfl }
  { rfl }

-- Proof is skipped, so we add 'sorry' to the end of theorem to indicate it.

end rectangle_area_correct_l408_408968


namespace vector_collinear_l408_408519

theorem vector_collinear 
  (a : ℝ × ℝ)
  (b : ℝ × ℝ)
  (c : ℝ × ℝ)
  (k : ℝ)
  (h_a : a = (Real.sqrt 3, 1))
  (h_b : b = (0, -1))
  (h_c : c = (k, Real.sqrt 3))
  (h_collinear : ∃ (λ : ℝ), a + 2 • b = λ • c) :
  k = -3 := sorry

end vector_collinear_l408_408519


namespace find_ellipse_params_l408_408790

def foci1 : ℝ × ℝ := (1, 3)
def foci2 : ℝ × ℝ := (1, -1)
def point_on_ellipse : ℝ × ℝ := (10, 5)

theorem find_ellipse_params (a b h k : ℝ) :
  let d1 := Real.sqrt ((10 - 1)^2 + (5 - 3)^2),
      d2 := Real.sqrt ((10 - 1)^2 + (5 + 1)^2),
      total_dist := d1 + d2,
      foci_dist := Real.sqrt ((1 - 1)^2 + ((3) - (-1))^2),
      h := 1,
      k := 1,
      a := (total_dist / 2),
      b := Real.sqrt (a^2 - (foci_dist / 2)^2)
  in
    (a, b, h, k) = (total_dist / 2, Real.sqrt ((total_dist / 2)^2 - 16), 1, 1) := 
by {
  -- We skip the proof since the task only requires the statement.
  sorry
}

end find_ellipse_params_l408_408790


namespace max_value_of_f_on_interval_l408_408845

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2

theorem max_value_of_f_on_interval : 
  ∃ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), f x = 2 ∧ ∀ y ∈ set.Icc (-1 : ℝ) (1 : ℝ), f y ≤ f x :=
by
  sorry

end max_value_of_f_on_interval_l408_408845


namespace find_FC_l408_408085

noncomputable def DC := 9
noncomputable def CB := 8
noncomputable def AD : ℝ := 68 / 3
noncomputable def AB : ℝ := (1 / 4) * AD
noncomputable def ED : ℝ := (3 / 4) * AD
noncomputable def CA : ℝ := CB + AB

theorem find_FC (h1 : DC = 9) (h2 : CB = 8) (h3 : AB = (1 / 4) * AD) (h4 : ED = (3 / 4) * AD) :
  let FC := (ED * CA) / AD in
  FC = 10.25 :=
by
  sorry

end find_FC_l408_408085


namespace isosceles_tangent_circle_l408_408202

-- Let $\triangle ABC$ be isosceles with vertex $A$
variables (A B C D M N : Point)
variables (h_isosceles : IsIsoscelesTriangle A B C)
variables (h_Dmid : IsMidpoint D B C)
variables (Gamma : Circle)
variables (h_center : IsCenter D Gamma)
variables (h_tangent_AB : IsTangent Gamma (LineSegment A B))
variables (h_tangent_AC : IsTangent Gamma (LineSegment A C))
variables (h_tangent_MN : IsTangent Gamma (LineSegment M N))

-- Show that $BD^2 = BM \times CN$
theorem isosceles_tangent_circle (h1: h_isosceles) (h2: h_Dmid) (h3: h_center)
  (h4: h_tangent_AB) (h5: h_tangent_AC) (h6: h_tangent_MN) : 
  let BD := Distance B D,
      BM := Distance B M,
      CN := Distance C N in
  BD^2 = BM * CN := by sorry

end isosceles_tangent_circle_l408_408202


namespace volume_correct_l408_408326

-- Define the diameter and depth of the well
def diameter : ℝ := 2
def depth : ℝ := 10
def radius : ℝ := diameter / 2

-- Define the volume function for a cylinder
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

-- Calculate the volume of the earth dug out
def volume_dug_out : ℝ := volume_cylinder radius depth

-- The main theorem to prove
theorem volume_correct :
  volume_dug_out = 31.4159 :=
by
  -- Proof not required
  sorry

end volume_correct_l408_408326


namespace trajectory_of_center_P_value_of_QA_QB_l408_408874

-- Definitions based on problem conditions
def Point := (ℝ × ℝ)

def Circle (center : Point) (radius : ℝ) :=
  { p : Point | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 }

def A : Point := (-2, 0)
def B : Circle := Circle (2,0) 6

-- Problem statement 1: Prove the equation of the trajectory E of the center P
theorem trajectory_of_center_P :
  { P : Point | (P.1^2) / 9 + (P.2^2) / 5 = 1 } = 
  { P : Point | let P := (P.1, P.2) in (abs (dist_pair P A + dist_pair P B) = 6 ∧ dist_pair P A + dist_pair P B = 6) } := sorry

-- Problem statement 2: For point Q on trajectory E with angle AQB = 60 degrees, find |QA| * |QB|
theorem value_of_QA_QB (Q : Point) (hQ : Q ∈ { P : Point | (P.1^2) / 9 + (P.2^2) / 5 = 1 }) :
  ∠(Q, A, (2, 0)) = 60 → abs (dist_pair Q A * dist_pair Q (2, 0)) = 20 / 3 := sorry


end trajectory_of_center_P_value_of_QA_QB_l408_408874


namespace sum_product_even_subsets_is_24_255_l408_408658

noncomputable def product_number (S : Set ℚ) : ℚ :=
  S.prod id

noncomputable def sum_product_numbers_even_subsets (M : Set ℚ) : ℚ :=
  (Finset.powerset M.to_finset).filter (λ s, s.card % 2 = 0).sum (λ s, product_number s)

theorem sum_product_even_subsets_is_24_255 :
  let M := (List.range' 1 100).to_finset.map ⟨λ n, 1/(n+1), sorry⟩ in
  sum_product_numbers_even_subsets M = 24.255 :=
by
  sorry

end sum_product_even_subsets_is_24_255_l408_408658


namespace principal_amount_l408_408776

noncomputable def exponential (r t : ℝ) :=
  Real.exp (r * t)

theorem principal_amount (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) :
  A = 5673981 ∧ r = 0.1125 ∧ t = 7.5 ∧ P = 2438978.57 →
  P = A / exponential r t := 
by
  intros h
  sorry

end principal_amount_l408_408776


namespace find_n_l408_408962

def P_X_eq_2 (n : ℕ) : Prop :=
  (3 * n) / ((n + 3) * (n + 2)) = (7 : ℚ) / 30

theorem find_n (n : ℕ) (h : P_X_eq_2 n) : n = 7 :=
by sorry

end find_n_l408_408962


namespace tax_percentage_first_40000_l408_408176

theorem tax_percentage_first_40000 {P : ℝ} :
  let income := 51999.99
  let first_40000 := 40000.0
  let excess_income := income - first_40000
  let tax_on_excess := 0.20 * excess_income
  let total_tax := 8000.0
  let tax_on_first_40000 := total_tax - tax_on_excess
  P = (tax_on_first_40000 / first_40000) * 100 →
  P = 14 :=
begin
  sorry
end

end tax_percentage_first_40000_l408_408176


namespace equilateral_PQR_l408_408572

-- Define the equilateral triangle setup
def equilateral_triangle (A B C : Type*) [metric_space A] [metric_space B] [metric_space C] (a b c : A × B × C) :=
  dist A B = dist A C ∧ dist B C = dist A B ∧ dist A C = dist B C

-- Define conditions on points A', B', C'
def points_on_sides (A' B' C' A B C : Type*) [metric_space A'] [metric_space B'] [metric_space C'] [metric_space A] [metric_space B] [metric_space C]
  (x y z : A' × B' × C' × (A × B × C)) (k : ℝ) :=
  dist B A' = k ∧ dist C B' = k ∧ dist A C' = k ∧ k < 1

-- Define the intersections of AA', BB', CC'
def intersections (P Q R A' B' C' A B C : Type*) [metric_space P] [metric_space Q] [metric_space R] [metric_space A'] [metric_space B'] [metric_space C']
  [metric_space A] [metric_space B] [metric_space C] (p q r : P × Q × R) :=
  ∃ (AA' BB' CC' : set (A × A')), 
  AA'.nonempty ∧ BB'.nonempty ∧ CC'.nonempty ∧ ∃! p : P, p ∈ AA' ∧ ∃! q : Q, q ∈ BB' ∧ ∃! r : R, r ∈ CC'

theorem equilateral_PQR (A B C A' B' C' P Q R : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space A'] 
  [metric_space B'] [metric_space C'] [metric_space P] [metric_space Q] [metric_space R] (a b c : A × B × C) 
  (a' b' c' : A' × B' × C') (p q r : P × Q × R) (k : ℝ) :
  equilateral_triangle A B C a b c →
  points_on_sides A' B' C' A B C a' b' c' k →
  intersections P Q R A' B' C' A B C p q r →
  dist P Q = dist Q R ∧ dist Q R = dist R P ∧ dist R P = 4 * (1 - k) / real.sqrt (k^2 - 2*k + 4) :=
sorry

end equilateral_PQR_l408_408572


namespace find_x_value_l408_408453

theorem find_x_value :
  ∀ x : ℝ, sqrt (4 - 5 * x) = 10 → x = -19.2 :=
by 
  intro x 
  intro h
  -- proof skipped
  sorry

end find_x_value_l408_408453


namespace concurrency_of_AD_DZ_XY_l408_408994

open EuclideanGeometry

variables {A B C I D D' X Y Z : Point}
variables [Incenter I triangle ABC]
variables (hD : ∃ (D : Point), foot_of_altitude_from I D BC)
variables (hD' : reflection_point D I D' )
variables (hID' : ∃ (D' : Point), (∥A D'∥ = ∥I D'∥))
variables (Γ : Circle) 
variables (hΓ : circle_centered_at D' passes_through A I)
variables (hX : ∃ (X : Point), (X ≠ A ∧ intersect_with Γ AB))
variables (hY : ∃ (Y : Point), (Y ≠ A ∧ intersect_with Γ AC))
variables (hZ : ∃ (Z : Point), point_on_Γ_with_AZ_perpendicular_to_BC)

theorem concurrency_of_AD_DZ_XY
  (h_incenter : incenter I A B C)
  (h_foot : foot_of_altitude_from I D BC)
  (h_reflection : reflection_point D I D')
  (h_equal : ∥A D'∥ = ∥I D'∥)
  (h_circle : circle_centered_at D' passes_through A I)
  (h_intersections_X : ∃ (X : Point), (X ≠ A) ∧ intersect_with Γ AB)
  (h_intersections_Y : ∃ (Y : Point), (Y ≠ A) ∧ intersect_with Γ AC)
  (h_point_Z : ∃ (Z : Point), point_on_Γ_with_AZ_perpendicular_to_BC) :
  concurrent_lines AD D'Z XY :=
sorry

end concurrency_of_AD_DZ_XY_l408_408994


namespace one_cow_eats_one_bag_in_40_days_l408_408328

theorem one_cow_eats_one_bag_in_40_days :
  (∀ (n m k : ℕ), n * k = m → m * 1 = n → (n = 40 ∧ m = 40 ∧ k = 40) → k = 40) :=
begin
  intros n m k hnm hmk hcond,
  cases hcond with hn hrest,
  cases hrest with hm hk,
  rw [hn, hm, hk] at *,
  apply hk,
end

end one_cow_eats_one_bag_in_40_days_l408_408328


namespace steven_card_count_l408_408633

theorem steven_card_count (num_groups : ℕ) (cards_per_group : ℕ) (h_groups : num_groups = 5) (h_cards : cards_per_group = 6) : num_groups * cards_per_group = 30 := by
  sorry

end steven_card_count_l408_408633


namespace problem_statement_l408_408602

noncomputable def a : ℝ := Real.floor (6 - Real.sqrt 10)
noncomputable def b : ℝ := 6 - Real.sqrt 10 - a

theorem problem_statement : (2 * a + Real.sqrt 10) * b = 6 := 
by 
  -- Proof will be filled in here
  sorry

end problem_statement_l408_408602


namespace min_m_plus_n_l408_408261

theorem min_m_plus_n (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 32 * m = n^5) : m + n = 3 :=
  sorry

end min_m_plus_n_l408_408261


namespace length_of_field_is_16_l408_408741

-- Definitions for the problem conditions
def width (field : Type) [has_width field (8 : ℝ)] : ℝ := 8
def length (field : Type) [has_length field (16 : ℝ)] : ℝ := 16
def area_p  := (length field) * (width field)
def area_f : ℝ := 2 * (width field) ^ 2

-- Given condition: The area of the pond is 1/2 the area of the field
axiom half_area (field : Type) [has_pond_area field] : area_p = 1/2 * area_f
axiom pond_area : pond_area = 64

-- Main proof statement
theorem length_of_field_is_16 (field : Type) [has_width field (w : ℝ)] [has_length field (2 * w : ℝ)] (h : pond_area = 64) : length field = 16 :=
by 
    sorry

end length_of_field_is_16_l408_408741


namespace complex_problem_l408_408474

def complex_number (z : ℂ) := z = (1 + complex.i) / (1 - complex.i)

theorem complex_problem : ∀ z : ℂ, complex_number z → |z| - conj z = 1 + complex.i :=
by
  intros z h
  rw complex_number at h
  -- We skip the detailed proof steps here
  sorry

end complex_problem_l408_408474


namespace units_digit_sum_squares_first_2023_odd_integers_l408_408720

theorem units_digit_sum_squares_first_2023_odd_integers :
  let pattern := [1, 9, 5, 9, 1] in
  let full_cycles := 2023 / 5 in
  let remaining := 2023 % 5 in
  (full_cycles * (pattern.foldl (λ acc x => acc + x) 0) + (remaining * pattern.foldl (λ acc x => acc + if x = (remaining - 1).natAbs then x else 0) 0)) % 10 = 5 :=
by
  sorry

end units_digit_sum_squares_first_2023_odd_integers_l408_408720


namespace sequence_value_l408_408037

theorem sequence_value (r x y : ℝ) 
  (h1 : r = 1/4) 
  (h2 : x = 256 * r)
  (h3 : y = x * r) : 
  x + y = 80 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end sequence_value_l408_408037


namespace max_value_of_symmetric_f_l408_408949

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_f :
  ∀ (a b : ℝ),
    (f 1 a b = 0) →
    (f (-1) a b = 0) →
    (f (-5) a b = 0) →
    (f (-3) a b = 0) →
    (∃ x : ℝ, f x 8 15 = 16) :=
by
  sorry

end max_value_of_symmetric_f_l408_408949


namespace smaller_sphere_surface_area_proof_l408_408986

-- Definitions based on conditions
noncomputable def cube_edge_length : ℝ := 1
noncomputable def larger_sphere_radius : ℝ := cube_edge_length / 2
noncomputable def diagonal_distance_from_center_to_vertex : ℝ := (↑(Real.sqrt 3)) * cube_edge_length / 2

-- Applying the result from the provided solution
noncomputable def smaller_sphere_radius : ℝ := (2 - Real.sqrt 3) / 2
noncomputable def surface_area_smaller_sphere : ℝ := 4 * Real.pi * smaller_sphere_radius^2

-- Statement to prove
theorem smaller_sphere_surface_area_proof :
  surface_area_smaller_sphere = π * (7 - 4 * Real.sqrt 3) := sorry

end smaller_sphere_surface_area_proof_l408_408986


namespace max_value_expression_l408_408019

theorem max_value_expression : ∃ s_max : ℝ, 
  (∀ s : ℝ, -3 * s^2 + 24 * s - 7 ≤ -3 * s_max^2 + 24 * s_max - 7) ∧
  (-3 * s_max^2 + 24 * s_max - 7 = 41) :=
sorry

end max_value_expression_l408_408019


namespace six_digit_numbers_with_conditions_l408_408641

theorem six_digit_numbers_with_conditions :
  let digits := {1, 2, 3, 4, 5, 6}
  in (∃ nums : Finset ℕ, nums.card = 24 ∧
              all_digits_in_numbers nums digits ∧
              all_digits_distinct_in_numbers nums ∧
              two_before_one_and_one_before_three nums) :=
sorry

-- Auxiliary definitions used in the theorem statement.
-- These should define what it means for the conditions to hold true.

def all_digits_in_numbers (nums : Finset ℕ) (digits : Finset ℕ) : Prop :=
  ∀ n ∈ nums, ∀ d ∈ digits, d ∈ n.digits

def all_digits_distinct_in_numbers (nums : Finset ℕ) : Prop :=
  ∀ n ∈ nums, n.digits.nodup

def two_before_one_and_one_before_three (nums : Finset ℕ) : Prop :=
  ∀ n ∈ nums, (∃ i j k, i < j ∧ j < k ∧ n.nth_digit(i) = 2 ∧ n.nth_digit(j) = 1 ∧ n.nth_digit(k) = 3)


end six_digit_numbers_with_conditions_l408_408641


namespace calculate_P_1_lt_X_lt_3_l408_408876

variable (X : ℝ → ℝ)
variable (hX : X ~ Normal(3, σ^2))
variable (h1 : P(X < 5) = 0.8)

theorem calculate_P_1_lt_X_lt_3 : P(1 < X < 3) = 0.3 :=
sorry

end calculate_P_1_lt_X_lt_3_l408_408876


namespace complement_of_M_in_U_l408_408921

def universal_set : Set ℝ := {x | x > 0}
def set_M : Set ℝ := {x | x > 1}
def complement (U M : Set ℝ) : Set ℝ := {x | x ∈ U ∧ x ∉ M}

theorem complement_of_M_in_U :
  complement universal_set set_M = {x | 0 < x ∧ x ≤ 1} :=
by
  sorry

end complement_of_M_in_U_l408_408921


namespace geometric_series_sum_l408_408006

theorem geometric_series_sum :
  let a := 1
  let r := (1 / 4 : ℚ)
  (a / (1 - r)) = 4 / 3 :=
by
  sorry

end geometric_series_sum_l408_408006


namespace range_of_f_l408_408864

def f (x : ℝ) : ℝ := 2 * x^2 + 1

theorem range_of_f : set.image f {-1, 0, 1} = {1, 3} :=
by
  sorry

end range_of_f_l408_408864


namespace height_of_congruent_triangle_l408_408156

theorem height_of_congruent_triangle
  (ABC DEF : Triangle)
  (congr : congruent ABC DEF)
  (AB : ℝ := 6)
  (area_DEF : area DEF = 12) :
  height_on_side ABC AB = 4 := 
begin
  sorry
end

end height_of_congruent_triangle_l408_408156


namespace murtha_total_items_at_day_10_l408_408613

-- Define terms and conditions
def num_pebbles (n : ℕ) : ℕ := n
def num_seashells (n : ℕ) : ℕ := 1 + 2 * (n - 1)

def total_pebbles (n : ℕ) : ℕ :=
  (n * (1 + n)) / 2

def total_seashells (n : ℕ) : ℕ :=
  (n * (1 + num_seashells n)) / 2

-- Define main proposition
theorem murtha_total_items_at_day_10 : total_pebbles 10 + total_seashells 10 = 155 := by
  -- Placeholder for proof
  sorry

end murtha_total_items_at_day_10_l408_408613


namespace shaded_area_l408_408112

-- Definition of square side lengths
def side_lengths : List ℕ := [2, 4, 6, 8, 10]

-- Definition for the area of the largest square
def largest_square_area : ℕ := 10 * 10

-- Definition for the area of the smallest non-shaded square
def smallest_square_area : ℕ := 2 * 2

-- Total area of triangular regions
def triangular_area : ℕ := 2 * (2 * 4 + 2 * 6 + 2 * 8 + 2 * 10)

-- Question to prove
theorem shaded_area : largest_square_area - smallest_square_area - triangular_area = 40 := by
  sorry

end shaded_area_l408_408112


namespace ab_value_l408_408102

variables (a b c : ℝ) (angle_C : ℝ)

-- Conditions: Triangle ABC with sides a, b, and c, angle C = 60 degrees, and given equation a^2 + b^2 - c^2 = 4
def is_triangle (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

axiom angle_C_is_60 : angle_C = 60

axiom given_equation : a^2 + b^2 - c^2 = 4

-- Target: Prove ab = 4
theorem ab_value : a * b = 4 :=
by {
  have cos_60 : real.cos (angle_C * real.pi / 180) = 1 / 2, from calc
    real.cos (angle_C * real.pi / 180) = real.cos (60 * real.pi / 180) : by rw angle_C_is_60
    ... = 1 / 2 : by norm_num,
  sorry
}

end ab_value_l408_408102


namespace distance_between_hyperbola_vertices_l408_408075

theorem distance_between_hyperbola_vertices : 
  (∀ x y : ℝ, (x^2 / 36 - y^2 / 25 = 1) → (abs (2 * sqrt 36) = 12)) :=
by 
  intros x y h
  have h1 : 2 * sqrt 36 = 2 * 6 := by sorry
  have h2 : abs (2 * 6) = 12 := by sorry
  exact h2

end distance_between_hyperbola_vertices_l408_408075


namespace intersecting_graphs_l408_408651

theorem intersecting_graphs (a b c d : ℝ) 
  (h1 : -2 * |1 - a| + b = 4) 
  (h2 : 2 * |1 - c| + d = 4)
  (h3 : -2 * |7 - a| + b = 0) 
  (h4 : 2 * |7 - c| + d = 0) : a + c = 10 := 
sorry

end intersecting_graphs_l408_408651


namespace smallest_palindrome_in_bases_3_and_5_l408_408814

def is_palindrome {α : Type*} [DecidableEq α] (l : List α) : Prop :=
l = l.reverse

def to_base (n b : ℕ) : List ℕ :=
if b ≤ 1 then []
else let rec to_base_aux (n b : ℕ) : List ℕ :=
  if n = 0 then []
  else (n % b) :: to_base_aux (n / b) b
in (to_base_aux n b).reverse

def is_palindrome_in_base (n b : ℕ) : Prop :=
is_palindrome (to_base n b)

theorem smallest_palindrome_in_bases_3_and_5 :
  ∃ n : ℕ, 6 < n ∧ is_palindrome_in_base n 3 ∧ is_palindrome_in_base n 5 ∧
  ∀ m : ℕ, 6 < m ∧ is_palindrome_in_base m 3 ∧ is_palindrome_in_base m 5 → n ≤ m := 
by
  use 26
  split
  . exact dec_trivial
  split
  . sorry -- Prove that 26 is a palindrome in base 3
  split
  . sorry -- Prove that 26 is a palindrome in base 5
  . intros m hm_pal3_pal5
    have : 26 ≤ m := sorry
    exact this

end smallest_palindrome_in_bases_3_and_5_l408_408814


namespace next_four_digit_number_l408_408667

def isPerfectSquare (n : ℕ) : Prop :=
∃ m : ℕ, m * m = n

def satisfiesProperty (n : ℕ) : Prop :=
  ∃ a b : ℕ, 10 ≤ a ∧ a < 100 ∧ 0 ≤ b ∧ b < 10 ∧ n = 1800 + a * 10 + b ∧ isPerfectSquare (18 * a)

theorem next_four_digit_number (n : ℕ) (h₀ : n = 1818) (h₁ : satisfiesProperty n) :
  ∃ m : ℕ, m > n ∧ m < 2000 ∧ satisfiesProperty m ∧ (∀ k : ℕ, n < k ∧ k < m → ¬ satisfiesProperty k) :=
begin
  use 1832,
  split,
  { exact 1832 > 1818, },
  split,
  { exact 1832 < 2000, },
  split,
  { unfold satisfiesProperty,
    use 32,
    use 0,
    split, { exact nat.le_of_lt 32 10 100, },
    split, { exact nat.lt_of_le_of_lt 0 32 40, },
    split, { exact nat.le_of_lt 0 0 10, },
    split, { exact nat.lt_of_le_of_lt 0 0 10, },
    split, { refl, },
    { unfold isPerfectSquare,
      use 24,
      exact nat.mul_self_eq 18 32 32, }, },
  { intros k hk,
    unfold satisfiesProperty at hk,
    cases hk with a ha,
    cases ha with b hb,
    cases hb with hb1 hb2,
    cases hb2 with hb3 hb4,
    cases hb4 with hb5 hb6,
    cases hb6 with hb7 hb8,
    cases hb8 with hb9 hb10,
    cases hb10 with hb11 hb12,
    exact nat.ne_of_lt_decides hk hb2 hb8 24, },
end

end next_four_digit_number_l408_408667


namespace next_four_digit_number_after_1818_l408_408686

-- Formalization of the given conditions as Lean definitions
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def next_valid_number (n : ℕ) : ℕ :=
  let d1 := n / 1000 in
  let d2 := (n / 100) % 10 in
  let d3 := (n / 10) % 10 in
  let d4 := n % 10 in
  let next_n := (d1 * 10 + d2) * 100 + (d3 * 10 + d4) + 14*10 in -- moving to the next valid 4 digits in base 10 series
  next_n

def condition (n : ℕ) : Prop :=
  let d3d4 := n % 100 in
  is_perfect_square (18 * d3d4)

theorem next_four_digit_number_after_1818 :
  ∀ n : ℕ, 
    condition 1818 → 
    next_valid_number 1818 = 1832 :=
begin
  intro n,
  intro cond,
  sorry
end

end next_four_digit_number_after_1818_l408_408686


namespace perimeter_triangle_AF1F2_l408_408899

noncomputable theory

-- Given the ellipse C: x^2/36 + y^2/9 = 1
def ellipse (x y : ℝ) : Prop := (x^2 / 36) + (y^2 / 9) = 1

-- Let F₁ (-c, 0) and F₂ (c, 0) be the foci
def left_focus (c : ℝ) : ℝ × ℝ := (-c, 0)
def right_focus (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define the parameter 'c' from the ellipse equation
def c : ℝ := Real.sqrt (36 - 9)

-- Define any point A (x, y) on the ellipse
variable (x y : ℝ)
def A_on_ellipse : Prop := ellipse x y

-- Prove the perimeter of ΔAF₁F₂ is 12 + 6 * sqrt 3
theorem perimeter_triangle_AF1F2 : A_on_ellipse x y → 
  let a := 6 in
  let c_f := c in
  (Real.sqrt ((x + c_f) ^ 2 + y ^ 2) + Real.sqrt ((x - c_f) ^ 2 + y ^ 2) + 2 * c_f) = 12 + 6 * Real.sqrt 3 :=
by
  intros hA
  dsimp only [c, A_on_ellipse, ellipse] at *
  sorry

end perimeter_triangle_AF1F2_l408_408899


namespace f_increasing_on_interval_l408_408271

def f (x : ℝ) : ℝ := 1 + x - Real.sin x

theorem f_increasing_on_interval : ∀ x ∈ Set.Ioo 0 (2 * Real.pi), 0 < 1 - Real.cos x :=
by
  sorry

end f_increasing_on_interval_l408_408271


namespace cylinder_combined_value_equals_l408_408456

noncomputable def pi := Real.pi
def diameter : ℝ := 7  -- in cm
def radius : ℝ := diameter / 2
def height : ℝ := 40  -- in cm

def volume (r h : ℝ) := pi * r^2 * h
def curvedSurfaceArea (r h : ℝ) := 2 * pi * r * h
def totalSurfaceArea (r h : ℝ) := 2 * pi * r * h + 2 * pi * r^2

def combinedValue (V CSA TSA : ℝ) := V + CSA + TSA

theorem cylinder_combined_value_equals :
  combinedValue (volume radius height) (curvedSurfaceArea radius height) (totalSurfaceArea radius height) = 1074.5 * pi := by
  sorry

end cylinder_combined_value_equals_l408_408456


namespace hollis_student_loan_l408_408150

theorem hollis_student_loan
  (interest_loan1 : ℝ)
  (interest_loan2 : ℝ)
  (total_loan1 : ℝ)
  (total_loan2 : ℝ)
  (additional_amount : ℝ)
  (total_interest_paid : ℝ) :
  interest_loan1 = 0.07 →
  total_loan1 = total_loan2 + additional_amount →
  additional_amount = 1500 →
  total_interest_paid = 617 →
  total_loan2 = 4700 →
  total_loan1 * interest_loan1 + total_loan2 * interest_loan2 = total_interest_paid →
  total_loan2 = 4700 :=
by
  sorry

end hollis_student_loan_l408_408150


namespace inequality_proof_l408_408253

theorem inequality_proof (a b : ℝ) : 
  a^2 + b^2 + 2 * (a - 1) * (b - 1) ≥ 1 := 
by 
  sorry

end inequality_proof_l408_408253


namespace gretchen_flavors_l408_408149

/-- 
Gretchen's local ice cream shop offers 100 different flavors. She tried a quarter of the flavors 2 years ago and double that amount last year. Prove how many more flavors she needs to try this year to have tried all 100 flavors.
-/
theorem gretchen_flavors (F T2 T1 T R : ℕ) (h1 : F = 100)
  (h2 : T2 = F / 4)
  (h3 : T1 = 2 * T2)
  (h4 : T = T2 + T1)
  (h5 : R = F - T) : R = 25 :=
sorry

end gretchen_flavors_l408_408149


namespace derivative_at_1_eq_ln_2_l408_408496

noncomputable def f : ℝ → ℝ
| x := (f 1) * x^3 - 2^x

theorem derivative_at_1_eq_ln_2 : deriv f 1 = Real.log 2 :=
by
  sorry

end derivative_at_1_eq_ln_2_l408_408496


namespace total_earnings_correct_l408_408340

section
  -- Define the conditions
  def wage : ℕ := 8
  def hours_Monday : ℕ := 8
  def hours_Tuesday : ℕ := 2

  -- Define the calculation for the total earnings
  def earnings (hours : ℕ) (rate : ℕ) : ℕ := hours * rate

  -- State the total earnings
  def total_earnings : ℕ := earnings hours_Monday wage + earnings hours_Tuesday wage

  -- Theorem: Prove that Will's total earnings in those two days is $80
  theorem total_earnings_correct : total_earnings = 80 := by
    sorry
end

end total_earnings_correct_l408_408340


namespace domain_of_f_l408_408710

noncomputable def f (x : ℝ) : ℝ :=
  Math.logBase 3 (Math.logBase 4 (Math.logBase 6 (Math.logBase 7 x)))

theorem domain_of_f :
  ∀ x, (7 ^ 1296 < x) ↔ (x ∈ Set.Ioo (7 ^ 1296) ∞) :=
by
  intro x
  sorry

end domain_of_f_l408_408710


namespace min_value_expression_l408_408597

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ 64 :=
by
  sorry

end min_value_expression_l408_408597


namespace part1_part2_l408_408908

noncomputable def f (a x : ℝ) : ℝ := log (a * x) - 1 + 1 / x

theorem part1 (h : ∃ x, f a x = 0) : a = 1 := 
sorry

theorem part2 (n : ℕ) (pos_n : 0 < n) (x : ℝ) (pos_x : 0 < x):
  (x - 1) * exp (x - 1 / n) ≥ log (x / n) := 
sorry

end part1_part2_l408_408908


namespace focal_length_of_curve_l408_408645

theorem focal_length_of_curve : 
  (∀ θ : ℝ, ∃ x y : ℝ, x = 2 * Real.cos θ ∧ y = Real.sin θ) →
  ∃ f : ℝ, f = 2 * Real.sqrt 3 :=
by sorry

end focal_length_of_curve_l408_408645


namespace neighborhood_b_boxes_l408_408989

theorem neighborhood_b_boxes (hA : (10 : ℕ) * 2 * 2 = 40)
                             (hB : 50 > 40)
                             (cost_per_box : 2)
                             (total_homes_b : 5)
                             (total_revenue : 50) :
  total_revenue / cost_per_box / total_homes_b = 5 :=
by sorry

end neighborhood_b_boxes_l408_408989


namespace coupon_value_l408_408762

theorem coupon_value (C : ℝ) (original_price : ℝ := 120) (final_price : ℝ := 99) 
(membership_discount : ℝ := 0.1) (reduced_price : ℝ := original_price - C) :
0.9 * reduced_price = final_price → C = 10 :=
by sorry

end coupon_value_l408_408762


namespace dolly_should_buy_more_tickets_l408_408406

-- Define the conditions
def ferris_wheel_rides := 2
def roller_coaster_rides := 3
def log_ride_rides := 7

def ferris_wheel_cost_per_ride := 2
def roller_coaster_cost_per_ride := 5
def log_ride_cost_per_ride := 1

def initial_tickets := 20

-- Declare the theorem to prove
theorem dolly_should_buy_more_tickets :
  let total_needed_tickets := 
      ferris_wheel_rides * ferris_wheel_cost_per_ride + 
      roller_coaster_rides * roller_coaster_cost_per_ride + 
      log_ride_rides * log_ride_cost_per_ride in
  let additional_tickets := total_needed_tickets - initial_tickets in
  additional_tickets = 6 :=
by
  sorry

end dolly_should_buy_more_tickets_l408_408406


namespace find_m_range_minimum_value_fm_solve_quadratic_inequality_l408_408915

def quadratic_inequality_solution_set (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * m * x + m + 2 ≥ 0

theorem find_m_range : {m : ℝ | quadratic_inequality_solution_set m} = Icc (-1) 2 :=
sorry

def f (m : ℝ) : ℝ := m + 3 / (m + 2)

theorem minimum_value_fm :
  ∀ m : ℝ, -1 ≤ m ∧ m ≤ 2 → f m ≥ 2 * Real.sqrt 3 - 2 :=
sorry

theorem solve_quadratic_inequality (m x : ℝ) :
  -1 ≤ m ∧ m ≤ 2 →
  x^2 + (m - 3) * x - 3 * m > 0 ↔ x ∈ Ioo (-∞) (-m) ∪ Ioo 3 ∞ :=
sorry

end find_m_range_minimum_value_fm_solve_quadratic_inequality_l408_408915


namespace total_pizza_equivalents_l408_408769

-- Definitions of the conditions
def pizzas_lunch : ℕ := 9
def pizzas_dinner : ℕ := 6
def calzones_lunch : ℕ := 4
def calzone_to_pizza_equiv : ℝ := 0.5

-- Total pizza equivalents served today
theorem total_pizza_equivalents : 
    pizzas_lunch + pizzas_dinner + (calzones_lunch * calzone_to_pizza_equiv).toNat = 17 :=
by
  sorry

end total_pizza_equivalents_l408_408769


namespace relation_ab_c_l408_408872

noncomputable def f : ℝ → ℝ := sorry
def f' (x : ℝ) := deriv f x

-- Given condition: when x > 0, xf'(x) - f(x) < 0
axiom condition (x : ℝ) (hx : 0 < x) : x * f' x - f x < 0

def a := 2 * f 1
def b := f 2
def c := 4 * f (1 / 2)

theorem relation_ab_c : b < a ∧ a < c :=
by sorry

end relation_ab_c_l408_408872


namespace candle_cost_correct_l408_408992

-- Variables and conditions
def candles_per_cake : Nat := 8
def num_cakes : Nat := 3
def candles_needed : Nat := candles_per_cake * num_cakes

def candles_per_box : Nat := 12
def boxes_needed : Nat := candles_needed / candles_per_box

def cost_per_box : ℝ := 2.5
def total_cost : ℝ := boxes_needed * cost_per_box

-- Proof statement
theorem candle_cost_correct :
  total_cost = 5 := by
  sorry

end candle_cost_correct_l408_408992


namespace no_arithmetic_mean_l408_408395
open_locale classical

noncomputable def f1 : ℚ := 5 / 8
noncomputable def f2 : ℚ := 9 / 12
noncomputable def f3 : ℚ := 7 / 10

noncomputable def c1 := 75 / 120
noncomputable def c2 := 90 / 120
noncomputable def c3 := 84 / 120

theorem no_arithmetic_mean : (∀ x y z : ℚ, (x = f1 ∨ x = f2 ∨ x = f3) ∧ (y = f1 ∨ y = f2 ∨ y = f3) ∧ (z = f1 ∨ z = f2 ∨ z = f3) ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z → 
(arith_mean x y ≠ z) ∨ (arith_mean x z ≠ y) ∨ (arith_mean y z ≠ x)) := by
  sorry

noncomputable def arith_mean (x y : ℚ) : ℚ := (x + y) / 2

end no_arithmetic_mean_l408_408395


namespace sum_first_20_terms_l408_408894

noncomputable def sum_first_n_terms (a d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

-- Define given data
def S₁₀ : ℝ := 30
def S₃₀ : ℝ := 210

theorem sum_first_20_terms (a d : ℝ) :
  sum_first_n_terms a d 10 = S₁₀ →
  sum_first_n_terms a d 30 = S₃₀ →
  sum_first_n_terms a d 20 = 100 :=
begin
  intros h₁ h₂,
  sorry
end

end sum_first_20_terms_l408_408894


namespace correct_value_l408_408931

theorem correct_value (x : ℝ) (h : x + 2.95 = 9.28) : x - 2.95 = 3.38 :=
by
  sorry

end correct_value_l408_408931


namespace number_of_pies_l408_408560

theorem number_of_pies (total_apples unripe_apples apples_per_pie : ℕ) (h1 : total_apples = 128) (h2 : unripe_apples = 23) (h3 : apples_per_pie = 7) :
  (total_apples - unripe_apples) / apples_per_pie = 15 :=
by
  rw [h1, h2, h3]
  -- below simplify to calculate number of pies
  calc
    (128 - 23) / 7 = 105 / 7 := by norm_num
                   ... = 15 := by norm_num

end number_of_pies_l408_408560


namespace shelf_arrangement_count_l408_408852

theorem shelf_arrangement_count :
  let products := [A, B, C, D, E] in
  adjacent A B products →
  ¬adjacent C D products →
  count_arrangements products = 24 :=
by sorry

end shelf_arrangement_count_l408_408852


namespace min_a2_b2_l408_408020

theorem min_a2_b2 (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) :
  a^2 + b^2 ≥ 4 / 5 :=
sorry

end min_a2_b2_l408_408020


namespace sec_330_eq_2sqrt3_div_3_l408_408445

theorem sec_330_eq_2sqrt3_div_3 : sec 330 = 2 * sqrt 3 / 3 := by
  sorry

end sec_330_eq_2sqrt3_div_3_l408_408445


namespace farmer_needs_to_plough_120_hectares_daily_l408_408758

variable (dailyArea targetArea actualDailyArea remainingArea extraDays : ℕ)
variables (plannedDays : ℕ)

-- Given conditions
def targetAreaCondition : Prop := targetArea = 720
def actualDailyAreaCondition : Prop := actualDailyArea = 85
def remainingAreaCondition : Prop := remainingArea = 40
def extraDaysCondition : Prop := extraDays = 2

-- Define the equation for planned days
def plannedDaysEquation (d : ℕ) : Prop := 
  actualDailyArea * (d + extraDays) + remainingArea = targetArea

-- Define the solution to the number of hectares needed daily
def requiredDailyArea : ℕ := targetArea / plannedDays

-- Theorem stating the farmer needed to plough 120 hectares daily to finish on time
theorem farmer_needs_to_plough_120_hectares_daily
  (h1 : targetAreaCondition)
  (h2 : actualDailyAreaCondition)
  (h3 : remainingAreaCondition)
  (h4 : extraDaysCondition)
  (h5 : ∃ d, plannedDaysEquation d):
  dailyArea = 120 :=
sorry

end farmer_needs_to_plough_120_hectares_daily_l408_408758


namespace time_to_pass_train_l408_408371

noncomputable def time_to_pass (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  let relative_speed := (train_speed + man_speed) * (5/18) -- convert km/hr to m/s
  train_length / relative_speed

theorem time_to_pass_train :
  time_to_pass 605 60 6 ≈ 33 := sorry

end time_to_pass_train_l408_408371


namespace histogram_groups_l408_408543

theorem histogram_groups 
  (max_height : ℕ)
  (min_height : ℕ)
  (class_interval : ℕ)
  (h_max : max_height = 176)
  (h_min : min_height = 136)
  (h_interval : class_interval = 6) :
  Nat.ceil ((max_height - min_height) / class_interval) = 7 :=
by
  sorry

end histogram_groups_l408_408543


namespace mike_initial_cards_l408_408229

-- Define the conditions
def initial_cards (x : ℕ) := x + 13 = 100

-- Define the proof statement
theorem mike_initial_cards : initial_cards 87 :=
by
  sorry

end mike_initial_cards_l408_408229


namespace pq_sum_l408_408023

theorem pq_sum (p q : ℝ) 
  (h1 : p / 3 = 9) 
  (h2 : q / 3 = 15) : 
  p + q = 72 :=
sorry

end pq_sum_l408_408023


namespace units_digit_first_2023_odd_squares_l408_408729

noncomputable def units_digit_of_squares_sum (n : ℕ) : ℕ :=
  let odd_squares_units := λ k, (2 * k + 1) ^ 2 % 10 in
  (List.sum (List.map odd_squares_units (List.range n))) % 10

theorem units_digit_first_2023_odd_squares :
  units_digit_of_squares_sum 2023 = 5 :=
sorry

end units_digit_first_2023_odd_squares_l408_408729


namespace find_line_equation_l408_408476

noncomputable def line_through_point (P Q : (ℝ × ℝ)) : ℝ × ℝ → bool :=
λ x : ℝ × ℝ,
  let (a, b) := P in
  let (c, d) := Q in
  if (a - c) = 0 then x.1 = a  -- Line is vertical
  else x.2 = ((d - b) / (c - a)) * x.1 + (b - ((d - b) / (c - a)) * a)

def is_midpoint (M A B : (ℝ × ℝ)) : bool :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

def distance (P Q : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

theorem find_line_equation
  (P : (ℝ × ℝ)) (hP : P = (1, 2))
  (circle : (ℝ × ℝ) → bool) (h_circle : ∀ x, circle x = (x.1 ^ 2 + x.2 ^ 2 = 6)) :

  (∀ A B, is_midpoint P A B → circle A ∧ circle B → line_through_point P A = λ x, x.1 + 2 * x.2 - 5 = 0) ∧
  (∀ A B, distance A B = 2 * real.sqrt 5 → circle A ∧ circle B → 
    (line_through_point P A = λ x, x.1 = 1 ∨ line_through_point P A = λ x, 3 * x.1 - 4 * x.2 + 5 = 0)) :=
by sorry

end find_line_equation_l408_408476


namespace last_triangle_perimeter_l408_408205

noncomputable def T1_side1 := 1003
noncomputable def T1_side2 := 1004
noncomputable def T1_side3 := 1005

def perimeter_last_triangle : ℚ :=
  have triangle_existence (n : ℕ) := n < 11,
  if triangle_existence 10 then 3 * (251 / 128) else 0

theorem last_triangle_perimeter :
  perimeter_last_triangle = 753 / 128 :=
by
  sorry

end last_triangle_perimeter_l408_408205


namespace books_left_after_giveaways_l408_408438

def initial_books : ℝ := 48.0
def first_giveaway : ℝ := 34.0
def second_giveaway : ℝ := 3.0

theorem books_left_after_giveaways : 
  initial_books - first_giveaway - second_giveaway = 11.0 :=
by
  sorry

end books_left_after_giveaways_l408_408438


namespace range_of_a_in_quadratic_l408_408509

theorem range_of_a_in_quadratic :
  (∃ x1 x2 : ℝ, x1 < -1 ∧ x2 > 1 ∧ x1 ≠ x2 ∧ x1^2 + a * x1 - 2 = 0 ∧ x2^2 + a * x2 - 2 = 0) → -1 < a ∧ a < 1 :=
by
  sorry

end range_of_a_in_quadratic_l408_408509


namespace gcd_of_repeated_six_digit_integers_l408_408365

-- Given condition
def is_repeated_six_digit_integer (n : ℕ) : Prop :=
  100 ≤ n / 1000 ∧ n / 1000 < 1000 ∧ n = 1001 * (n / 1000)

-- Theorem to prove
theorem gcd_of_repeated_six_digit_integers :
  ∀ n : ℕ, is_repeated_six_digit_integer n → gcd n 1001 = 1001 :=
by sorry

end gcd_of_repeated_six_digit_integers_l408_408365


namespace factorize_problem_1_factorize_problem_2_l408_408069

variables (x y : ℝ)

-- Problem 1: Prove that x^2 * y - 4 * x * y + 4 * y = y * (x - 2) ^ 2
theorem factorize_problem_1 : 
  x^2 * y - 4 * x * y + 4 * y = y * (x - 2) ^ 2 :=
sorry

-- Problem 2: Prove that x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y)
theorem factorize_problem_2 : 
  x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y) :=
sorry

end factorize_problem_1_factorize_problem_2_l408_408069


namespace P_locus_case_a_P_locus_case_b_l408_408622

def point := (ℝ × ℝ)

def A : point := (-1, 0)
def B : point := (1, 0)
def C : point := (0, Real.sqrt 3)

def dist_squared (P Q : point) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem P_locus_case_a (P : point) :
  dist_squared P A + dist_squared P B = dist_squared P C ↔
  P.1^2 + (P.2 + Real.sqrt 3)^2 = 4 := sorry

theorem P_locus_case_b (P : point) :
  dist_squared P A + dist_squared P B = 2 * dist_squared P C ↔
  P.2 = Real.sqrt 3 / 3 := sorry

end P_locus_case_a_P_locus_case_b_l408_408622


namespace complex_modulus_l408_408422

theorem complex_modulus :
  abs ((7 - 4*complex.I) * (3 + 11*complex.I)) = Real.sqrt 8450 :=
by
  sorry

end complex_modulus_l408_408422


namespace perimeter_of_triangle_l408_408652

def hyperbola (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1

def distance (P1 P2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

theorem perimeter_of_triangle
  (x y : ℝ)
  (hx : hyperbola x y)
  (PF2 : ℝ)
  (hPF2 : PF2 = 7) :
  ∃ PF1 : ℝ, PF1 = 13 ∧ ∃ PF1_PF2 : ℝ, PF1_PF2 = 10 ∧ PF2 + PF1 + PF1_PF2 = 30 :=
by
  sorry

end perimeter_of_triangle_l408_408652


namespace competition_score_difference_l408_408967

theorem competition_score_difference :
  let total_students := 40 in
  let score_60 := 12 * total_students / 100 in
  let score_75 := 18 * total_students / 100 in
  let score_82 := 25 * total_students / 100 in
  let score_88 := 20 * total_students / 100 in
  let score_94 := 25 * total_students / 100 in
  let scores := replicate score_60 60 ++ replicate score_75 75 ++ 
                replicate score_82 82 ++ replicate score_88 88 ++ 
                replicate score_94 94 in
  let median := (scores.nth (total_students / 2 - 1) + scores.nth (total_students / 2)) / 2 in
  let mean := (60 * score_60 + 75 * score_75 + 82 * score_82 + 88 * score_88 + 94 * score_94) / total_students in
  (mean.to_real - median) = 0.225 := 
sorry

end competition_score_difference_l408_408967


namespace max_children_arrangement_l408_408693

theorem max_children_arrangement (n : ℕ) (h1 : n = 49) 
  (h2 : ∀ i j, i ≠ j → 1 ≤ i ∧ i ≤ 49 → 1 ≤ j ∧ j ≤ 49 → (i * j < 100)) : 
  ∃ k, k = 18 :=
by
  sorry

end max_children_arrangement_l408_408693


namespace union_of_A_and_B_l408_408919

def A : Set ℝ := { x | x^2 - 2 ≥ 0 }
def B : Set ℝ := { x | x^2 - 4x + 3 ≤ 0 }

theorem union_of_A_and_B : A ∪ B = { x | x ≤ -Real.sqrt 2 ∨ x ≥ 1 } :=
by
  sorry

end union_of_A_and_B_l408_408919


namespace bus_students_remain_l408_408164

theorem bus_students_remain (init_students : ℕ) 
  (third_got_off : ℕ → ℕ) 
  (first_stop_second_third_fourth : ℕ ≠ 0 ∧ init_students = 64 ∧ 
   ∀ s, third_got_off s = (s * 2) / 3 ∧ 
   third_got_off (init_students * 2 / 3) = ((init_students * 2 / 3) * 2) / 3 ∧ 
   third_got_off ((init_students * 2 / 3) * 2 / 3) = (((init_students * 2 / 3) * 2 / 3) * 2) / 3 ∧ 
   third_got_off ((((init_students * 2 / 3) * 2 / 3) * 2 / 3) * 2) / 3) : 
  (((((init_students * 2) / 3) * 2) / 3) * 2 / 3 * 2) / 3 * 2 = 1024 / 81 :=
by sorry

end bus_students_remain_l408_408164


namespace range_of_log_in_interval_l408_408130

def f (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem range_of_log_in_interval : set.range (λ x : {x // 1/2 ≤ x ∧ x ≤ 4}, f x.1) = set.Icc (-1 : ℝ) 2 :=
sorry

end range_of_log_in_interval_l408_408130


namespace next_perfect_square_after_1818_l408_408687

def is_next_perfect_square_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (∃ m : ℕ, n = 1800 + m ∧ m < 100 ∧ (∃ k : ℕ, 18 * m = k^2))

theorem next_perfect_square_after_1818 : ∃ n : ℕ, is_next_perfect_square_number n ∧ n > 1818 ∧ n = 1832 :=
by
  existsi 1832
  unfold is_next_perfect_square_number
  split
  . apply and.intro
    { linarith }
    { split
      . existsi 32
        split
        . linarith
        . existsi 18
          linarith
      . sorry }

end next_perfect_square_after_1818_l408_408687


namespace tangent_line_at_origin_inequality_iff_a_nonneg_l408_408901

-- Define the function f
def f (a x : ℝ) : ℝ := (a * x + 1) * Real.exp x - (a + 1) * x - 1

-- Define the first statement: Tangent line at (0, f(0)) is y = 0
theorem tangent_line_at_origin (a : ℝ) : 
  let f0 := f a 0 in
  let f'_at_0 := (a * 0 + 1 + a) * Real.exp 0 - (a + 1) in
  f0 = 0 ∧ f'_at_0 = 0 → f'_at_0 = 0 := 
by 
  sorry
  
-- Define the second statement: Inequality f(x) > 0 for x > 0 holds if and only if a >= 0
theorem inequality_iff_a_nonneg (a : ℝ) : 
  (∀ x > 0, f a x > 0) ↔ (0 ≤ a) :=
by 
  sorry

end tangent_line_at_origin_inequality_iff_a_nonneg_l408_408901


namespace find_x_l408_408189

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2))

theorem find_x (x : ℝ) : 
  distance (-3, 4, 0) (x, -1, 6) = 10 ↔ x = 2 ∨ x = -8 :=
by { sorry }

end find_x_l408_408189


namespace fraction_of_good_firecrackers_set_off_l408_408562

variable (total_firecrackers confiscated_firecrackers defective_fraction good_firecrackers set_off : ℕ)
variable (fraction_set_off : ℚ)

-- Initial conditions
def initial_conditions : Prop :=
  total_firecrackers = 48 ∧
  confiscated_firecrackers = 12 ∧
  defective_fraction = 1/6

-- Computations based on initial conditions
def remaining_firecrackers : ℕ := total_firecrackers - confiscated_firecrackers
def defective_firecrackers : ℕ := (defective_fraction * remaining_firecrackers).toInt
def good_firecrackers' : ℕ := remaining_firecrackers - defective_firecrackers

-- Condition that Jerry sets off some good firecrackers
def jerry_sets_off : Prop := set_off = 15 ∧ good_firecrackers' = 30

-- The question statement as a proof goal
theorem fraction_of_good_firecrackers_set_off (h1 : initial_conditions) (h2 : jerry_sets_off) :
  fraction_set_off = 15 / 30 :=
by
  sorry

end fraction_of_good_firecrackers_set_off_l408_408562


namespace next_perfect_square_after_1818_l408_408690

def is_next_perfect_square_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (∃ m : ℕ, n = 1800 + m ∧ m < 100 ∧ (∃ k : ℕ, 18 * m = k^2))

theorem next_perfect_square_after_1818 : ∃ n : ℕ, is_next_perfect_square_number n ∧ n > 1818 ∧ n = 1832 :=
by
  existsi 1832
  unfold is_next_perfect_square_number
  split
  . apply and.intro
    { linarith }
    { split
      . existsi 32
        split
        . linarith
        . existsi 18
          linarith
      . sorry }

end next_perfect_square_after_1818_l408_408690


namespace complex_modulus_l408_408423

theorem complex_modulus :
  abs ((7 - 4*complex.I) * (3 + 11*complex.I)) = Real.sqrt 8450 :=
by
  sorry

end complex_modulus_l408_408423


namespace next_four_digit_number_after_1818_l408_408684

-- Formalization of the given conditions as Lean definitions
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def next_valid_number (n : ℕ) : ℕ :=
  let d1 := n / 1000 in
  let d2 := (n / 100) % 10 in
  let d3 := (n / 10) % 10 in
  let d4 := n % 10 in
  let next_n := (d1 * 10 + d2) * 100 + (d3 * 10 + d4) + 14*10 in -- moving to the next valid 4 digits in base 10 series
  next_n

def condition (n : ℕ) : Prop :=
  let d3d4 := n % 100 in
  is_perfect_square (18 * d3d4)

theorem next_four_digit_number_after_1818 :
  ∀ n : ℕ, 
    condition 1818 → 
    next_valid_number 1818 = 1832 :=
begin
  intro n,
  intro cond,
  sorry
end

end next_four_digit_number_after_1818_l408_408684


namespace richard_more_pins_than_patrick_l408_408979

theorem richard_more_pins_than_patrick :
  ∀ (R P R2 P2 : ℕ), 
    P = 70 → 
    R > P →
    P2 = 2 * R →
    R2 = P2 - 3 → 
    (R + R2) = (P + P2) + 12 → 
    R = 70 + 15 := 
by 
  intros R P R2 P2 hP hRp hP2 hR2 hTotal
  sorry

end richard_more_pins_than_patrick_l408_408979


namespace lilian_height_conversion_l408_408223

noncomputable def lilian_height_cm : ℝ :=
  let feet_to_inches := 12
  let inches_to_cm := 2.54
  let height_inches := (5 * feet_to_inches + 4) in
  let height_cm_raw := height_inches * inches_to_cm in
  Real.round (height_cm_raw * 10) / 10

theorem lilian_height_conversion :
  lilian_height_cm = 162.6 :=
by
  sorry

end lilian_height_conversion_l408_408223


namespace length_of_second_train_correct_l408_408306

noncomputable def length_of_second_train (speed1_kmph : ℝ) (speed2_kmph : ℝ) (length1_m : ℝ) (clearing_time_s : ℝ) : ℝ :=
  let relative_speed_mps := (speed1_kmph + speed2_kmph) * (1000 / 3600)
  let total_distance_m := relative_speed_mps * clearing_time_s
  total_distance_m - length1_m

theorem length_of_second_train_correct :
  length_of_second_train 42 30 100 10.999120070394369 ≈ 119.98 :=
by
  sorry

end length_of_second_train_correct_l408_408306


namespace find_y_l408_408850

theorem find_y (y : ℝ) 
  (h : (real.sqrt 1.1) / (real.sqrt y) + (real.sqrt 1.44) / (real.sqrt 0.49) = 2.879628878919216) : 
  y = 0.8095238095238095 :=
sorry

end find_y_l408_408850


namespace max_value_of_symmetric_function_l408_408952

def f (x a b : ℝ) := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function 
  (a b : ℝ)
  (symmetric : ∀ t : ℝ, f (-2 + t) a b = f (-2 - t) a b) :
  ∃ M : ℝ, M = 16 ∧ ∀ x : ℝ, f x a b ≤ M :=
by
  use 16
  sorry

end max_value_of_symmetric_function_l408_408952


namespace series_value_is_correct_l408_408200

noncomputable def check_series_value : ℚ :=
  let p : ℚ := 1859 / 84
  let q : ℚ := -1024 / 63
  let r : ℚ := 512 / 63
  let m : ℕ := 3907
  let n : ℕ := 84
  100 * m + n

theorem series_value_is_correct : check_series_value = 390784 := 
by 
  sorry

end series_value_is_correct_l408_408200


namespace constant_expression_l408_408255

noncomputable def sum_fifth_powers (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ i, i^5)
  
noncomputable def sum_third_powers (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ i, i^3)

noncomputable def sum_naturals (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ i, i)

theorem constant_expression (n : ℕ) (h : n > 1) :
  (sum_fifth_powers n - sum_third_powers n) /
  (sum_fifth_powers n - (sum_naturals n)^3) = 4 :=
by
  sorry

end constant_expression_l408_408255


namespace percentage_increase_l408_408763

variable (presentIncome : ℝ) (newIncome : ℝ)

theorem percentage_increase (h1 : presentIncome = 12000) (h2 : newIncome = 12240) :
  ((newIncome - presentIncome) / presentIncome) * 100 = 2 := by
  sorry

end percentage_increase_l408_408763


namespace range_of_a_l408_408482

variables (a x : ℝ) -- Define real number variables a and x

-- Define proposition p
def p : Prop := (a - 2) * x * x + 2 * (a - 2) * x - 4 < 0 -- Inequality condition for any real x

-- Define proposition q
def q : Prop := 0 < a ∧ a < 1 -- Condition for logarithmic function to be strictly decreasing

-- Lean 4 statement for the proof problem
theorem range_of_a (Hpq : (p a x ∨ q a) ∧ ¬ (p a x ∧ q a)) :
  (1 ≤ a ∧ a ≤ 2) ∨ (-2 < a ∧ a ≤ 0) :=
sorry

end range_of_a_l408_408482


namespace domain_f_l408_408644

open Real

def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2) - 3

theorem domain_f :
  {x : ℝ | g x > 0} = {x : ℝ | x < 0 ∨ x > 3} :=
by 
  sorry

end domain_f_l408_408644


namespace alpha_beta_value_l408_408888

noncomputable def alpha_beta_sum : ℝ := 75

theorem alpha_beta_value (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h : |Real.sin α - (1 / 2)| + Real.sqrt (Real.tan β - 1) = 0) :
  α + β = α_beta_sum := 
  sorry

end alpha_beta_value_l408_408888


namespace number_of_vans_needed_l408_408826

theorem number_of_vans_needed (capacity_per_van : ℕ) (students : ℕ) (adults : ℕ)
  (h_capacity : capacity_per_van = 9)
  (h_students : students = 40)
  (h_adults : adults = 14) :
  (students + adults + capacity_per_van - 1) / capacity_per_van = 6 := by
  sorry

end number_of_vans_needed_l408_408826


namespace lollipops_Lou_received_l408_408228

def initial_lollipops : ℕ := 42
def given_to_Emily : ℕ := 2 * initial_lollipops / 3
def kept_by_Marlon : ℕ := 4
def lollipops_left_after_Emily : ℕ := initial_lollipops - given_to_Emily
def lollipops_given_to_Lou : ℕ := lollipops_left_after_Emily - kept_by_Marlon

theorem lollipops_Lou_received : lollipops_given_to_Lou = 10 := by
  sorry

end lollipops_Lou_received_l408_408228


namespace functions_with_inverses_l408_408018

variable (p : ℝ → ℝ) (q : ℝ → ℝ) (r : ℝ → ℝ) (s : ℝ → ℝ) (t : ℝ → ℝ) (u : ℝ → ℝ) (v : ℝ → ℝ) (w : ℝ → ℝ)

def p_def := ∀ x, x ≤ 4 → p x = Real.sqrt (4 - x)
def q_def := ∀ x, q x = x^3 + x
def r_def := ∀ x, 0 < x → r x = x - 3 / x
def s_def := ∀ x, 1 ≤ x → s x = 3 * x^2 + 6 * x + 8
def t_def := ∀ x, t x = |x - 1| + |x + 2|
def u_def := ∀ x, u x = 2^x + 5^x
def v_def := ∀ x, 0 < x → v x = x + 2 / x
def w_def := ∀ x, -3 ≤ x ∧ x < 9 → w x = x / 3

theorem functions_with_inverses :
  (bijective p) ∧ 
  (bijective q) ∧ 
  (bijective r) ∧ 
  (bijective s) ∧ 
  (¬ bijective t) ∧ 
  (bijective u) ∧ 
  (bijective v) ∧ 
  (bijective w) := 
by sorry

end functions_with_inverses_l408_408018


namespace geometric_progression_sum_of_cubes_l408_408873

theorem geometric_progression_sum_of_cubes :
  ∃ (a r : ℕ) (seq : Fin 6 → ℕ), (seq 0 = a) ∧ (seq 1 = a * r) ∧ (seq 2 = a * r^2) ∧ (seq 3 = a * r^3) ∧ (seq 4 = a * r^4) ∧ (seq 5 = a * r^5) ∧
  (∀ i, 0 ≤ seq i ∧ seq i < 100) ∧
  (seq 0 + seq 1 + seq 2 + seq 3 + seq 4 + seq 5 = 326) ∧
  (∃ T : ℕ, (∀ i, ∃ k, seq i = k^3 → k * k * k = seq i) ∧ T = 64) :=
sorry

end geometric_progression_sum_of_cubes_l408_408873


namespace problem_statement_l408_408505

def f (x : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) : ℝ := a*x^2 + b*x + c

theorem problem_statement 
(b a c : ℝ) 
(hb_gt_a : b > a) 
(hf_nonneg : ∀ x : ℝ, f x a b c ≥ 0) 
(T := (f (-2) a b c) / (f 2 a b c - f 0 a b c)) 
(T_min_condition : T = (4*a - 2*b + c)/(4*a + 2*b + c) ≥ (4*a - 2*b + (b^2/(4*a)))/(4*a + 2*b)):
f (x + 2) 1 0 4 = (x + 2)^2 ∧ 
∀ x1 x2 ∈ [(-3: ℝ) * a, (-a: ℝ)], abs ((abs ((f x1 a b c) - a)) - (abs ((f x2 a b c) - a))) ≤ 2 * a → 
(0 < a ∧ a ≤ (2 + real.sqrt 3)/2) := 
begin
  sorry
end

end problem_statement_l408_408505


namespace sequence_value_l408_408038

theorem sequence_value (r x y : ℝ) 
  (h1 : r = 1/4) 
  (h2 : x = 256 * r)
  (h3 : y = x * r) : 
  x + y = 80 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end sequence_value_l408_408038


namespace sequence_sum_l408_408050

noncomputable def r : ℝ := 1 / 4
def t₀ : ℝ := 4096
def t₁ : ℝ := t₀ * r
def t₂ : ℝ := t₁ * r
def x : ℝ := t₂ * r
def y : ℝ := x * r

theorem sequence_sum : x + y = 80 := by
  -- The proof will go here
  sorry

end sequence_sum_l408_408050


namespace volume_of_rectangular_prism_l408_408360

theorem volume_of_rectangular_prism {l w h : ℝ} 
  (h1 : l * w = 12) 
  (h2 : w * h = 18) 
  (h3 : l * h = 24) : 
  l * w * h = 72 :=
by
  sorry

end volume_of_rectangular_prism_l408_408360


namespace problem_statement_l408_408900

def F (x : ℝ) : ℝ :=
  if x > 0 then 1
  else if x = 0 then 0
  else -1

variable {g : ℝ → ℝ}
variable (decreasing_g : ∀ x y : ℝ, x < y → g x > g y) -- g is decreasing
variable {a : ℝ} (a_pos : 0 < a) (a_lt_one : a < 1)

noncomputable def f (x : ℝ) : ℝ :=
  g x - g (a * x)

theorem problem_statement (x : ℝ) (decreasing_g : ∀ x y : ℝ, x < y → g x > g y) (a_pos : 0 < a) (a_lt_one : a < 1) :
  F (f x) = -F x :=
by
  sorry

end problem_statement_l408_408900


namespace modulus_product_l408_408425

open Complex

theorem modulus_product :
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  (Complex.abs (z1 * z2)) = Real.sqrt 8450 :=
by
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  sorry

end modulus_product_l408_408425


namespace medians_concurrent_angle_bisectors_concurrent_altitudes_concurrent_coords_G_coords_I_coords_H_l408_408308

noncomputable def barycentric_coords_G (A B C : Triangle) : BarycentricCoordinates :=
(1, 1, 1)

noncomputable def barycentric_coords_I (A B C : Triangle) : BarycentricCoordinates :=
(Real.sin A.angle, Real.sin B.angle, Real.sin C.angle)

noncomputable def barycentric_coords_H (A B C : Triangle) : BarycentricCoordinates :=
(Real.tan A.angle, Real.tan B.angle, Real.tan C.angle)

theorem medians_concurrent (A B C : Triangle) :
  concurrent (median A B C) :=
sorry

theorem angle_bisectors_concurrent (A B C : Triangle) :
  concurrent (angle_bisector A B C) :=
sorry

theorem altitudes_concurrent (A B C : Triangle) :
  concurrent (altitude A B C) :=
sorry

theorem coords_G (A B C : Triangle) : barycentric_coords_G A B C = (1, 1, 1) :=
sorry

theorem coords_I (A B C : Triangle) :
  barycentric_coords_I A B C = (Real.sin A.angle, Real.sin B.angle, Real.sin C.angle) :=
sorry

theorem coords_H (A B C : Triangle) :
  barycentric_coords_H A B C = (Real.tan A.angle, Real.tan B.angle, Real.tan C.angle) :=
sorry

end medians_concurrent_angle_bisectors_concurrent_altitudes_concurrent_coords_G_coords_I_coords_H_l408_408308


namespace find_a_l408_408860

def f (x a : ℝ) : ℝ := sin (2 * x - (Real.pi / 6)) - cos (2 * x + (Real.pi / 3)) + a

def g (x a : ℝ) : ℝ := 2 * sin (x + (Real.pi / 6)) + a

theorem find_a :
  ∀ (a : ℝ),
    let max_val := 2 + a in
    let min_val := 1 + a in
    max_val + min_val = 5 → a = 1 := by
  intros a max_val min_val h
  have h1 : max_val = 2 + a := rfl
  have h2 : min_val = 1 + a := rfl
  sorry

end find_a_l408_408860


namespace bits_required_for_ABC12_l408_408388

-- Definition of the hexadecimal number as a decimal.
def hex_ABC12_dec : ℕ := 10 * 16^4 + 11 * 16^3 + 12 * 16^2 + 1 * 16 + 2

-- The statement: Prove that the number of bits required to represent the decimal equivalent of ABC12_16 is 20.
theorem bits_required_for_ABC12 : (nat.log2 hex_ABC12_dec).succ = 20 := 
by
  sorry

end bits_required_for_ABC12_l408_408388


namespace triangle_inequality_l408_408170

theorem triangle_inequality
  (A B C : ℝ)
  (h : A + B + C = Real.pi) :
  (sqrt (sin A * sin B) / sin (C / 2) + sqrt (sin B * sin C) / sin (A / 2) + sqrt (sin C * sin A) / sin (B / 2)) ≥ 3 * sqrt 3 :=
by 
  sorry

end triangle_inequality_l408_408170


namespace part1_prices_part2_schemes_l408_408347

-- Condition definitions
def price_soccer_ball := 60
def price_basketball := 80

-- Part 1: Prices of soccer and basketballs
theorem part1_prices (x y : ℕ) (h₀ : 8 * x + 14 * y = 1600) (h₁ : y = x + 20) : 
  x = price_soccer_ball ∧ y = price_basketball := 
by {
  have h₂ : 8 * x + 14 * (x + 20) = 1600, by { rw [h₁], exact h₀ },
  simp at h₂,
  linarith,
  cases h₂,
  exact ⟨h₂_left, h₂_right⟩,
  sorry
}

-- Part 2: Possible purchasing schemes
theorem part2_schemes (balls : ℕ) (cost_min cost_max : ℕ) 
  (h₂ : balls = 50) (h₃ : cost_min = 3200) (h₄ : cost_max = 3240) : 
  ∃ (y : ℕ), y ∈ {38, 39, 40} ∧ (price_soccer_ball * y + price_basketball * (balls - y) ∈ Icc cost_min cost_max) :=
by {
  rw mem_set_of_eq,
  split,
  -- Case 1: y = 38
  { use 38,
    split,
    {
      left, right, left, refl,
    },
    {
      simp,
      exact sorry
    }
  },
  -- Case 2: y = 39
  { use 39,
    split,
    {
      left, right, right, left, refl,
    },
    {
      simp,
      exact sorry
    }
  },
  -- Case 3: y = 40
  { use 40,
    split,
    {
      right, right, refl,
    },
    {
      simp,
      exact sorry
    }
  }
}

end part1_prices_part2_schemes_l408_408347


namespace evaluate_expression_l408_408429

theorem evaluate_expression : 3 * (3 * (3 * (3 + 2) + 2) + 2) + 2 = 161 := sorry

end evaluate_expression_l408_408429
