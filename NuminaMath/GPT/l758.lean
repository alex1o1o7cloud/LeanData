import Mathlib

namespace quentavious_gum_pieces_l758_758731

-- Conditions expressed in Lean
variable (num_nickels num_dimes num_quarters : ℕ)
variable (exchange_rate_nickel exchange_rate_dime exchange_rate_quarter discount_rate : ℕ)
variable (remaining_nickels remaining_dimes : ℕ)

def original_nickels := 5
def original_dimes := 6
def original_quarters := 4
def exchange_rate_nickel := 2
def exchange_rate_dime := 3
def exchange_rate_quarter := 5
def discount_rate := 15
def remaining_nickels := 2
def remaining_dimes := 1

-- Hypothesis defining the problem scenario
theorem quentavious_gum_pieces :
  num_nickels = original_nickels →
  num_dimes = original_dimes →
  num_quarters = original_quarters →
  exchange_rate_nickel = 2 →
  exchange_rate_dime = 3 →
  exchange_rate_quarter = 5 →
  discount_rate = 15 →
  remaining_nickels = 2 →
  remaining_dimes = 1 →
  num_nickels - remaining_nickels > 0 →
  num_dimes - remaining_dimes > 0 →
  num_quarters > 0 →
  (num_nickels - remaining_nickels) * exchange_rate_nickel + 
  (num_dimes - remaining_dimes) * exchange_rate_dime + 
  num_quarters * exchange_rate_quarter = 41 →
  discount_rate = 15 →
  (num_nickels - remaining_nickels ≠ 0 ∧ 
  num_dimes - remaining_dimes ≠ 0 ∧ 
  num_quarters ≠ 0) →
  15 = 15 :=
by
  intros
  sorry

end quentavious_gum_pieces_l758_758731


namespace intersection_A_B_l758_758515

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_A_B : A ∩ B = {-2, 2} := by sorry

end intersection_A_B_l758_758515


namespace hyperbola_range_l758_758587

variable (m : ℝ)

noncomputable def hyperbola_equation : Prop :=
  (|m| - 1) * (m - 2) > 0

theorem hyperbola_range :
  hyperbola_equation m → (m ∈ set.Ioo (-1) 1 ∪ set.Ioi 2) :=
by
  sorry

end hyperbola_range_l758_758587


namespace geometric_progression_vertex_l758_758997

theorem geometric_progression_vertex (a b c d : ℝ) (q : ℝ)
  (h1 : b = 1)
  (h2 : c = 2)
  (h3 : q = c / b)
  (h4 : a = b / q)
  (h5 : d = c * q) :
  a + d = 9 / 2 :=
sorry

end geometric_progression_vertex_l758_758997


namespace trent_total_blocks_travelled_l758_758333

theorem trent_total_blocks_travelled :
  ∀ (walk_blocks bus_blocks : ℕ), 
  walk_blocks = 4 → 
  bus_blocks = 7 → 
  (walk_blocks + bus_blocks) * 2 = 22 := by
  intros walk_blocks bus_blocks hw hb
  rw [hw, hb]
  norm_num
  done

end trent_total_blocks_travelled_l758_758333


namespace second_smallest_five_digit_in_pascals_triangle_l758_758832

theorem second_smallest_five_digit_in_pascals_triangle : ∃ n k, (10000 < binomial n k) ∧ (binomial n k < 100000) ∧ 
    (∀ m l, (m < n ∨ (m = n ∧ l < k)) → (10000 ≤ binomial m l → binomial m l < binomial n k)) ∧ 
    binomial n k = 10001 :=
begin
  sorry
end

end second_smallest_five_digit_in_pascals_triangle_l758_758832


namespace quadratic_roots_conjugate_l758_758241

theorem quadratic_roots_conjugate (c d : ℝ) (h : ∃ r : ℂ, r = 3 - 4 * complex.I ∧ (r = (3 + 4 * complex.I) ∨ r = (3 - 4 * complex.I))) :
    (c, d) = (-6, 25) :=
sorry

end quadratic_roots_conjugate_l758_758241


namespace sum_infinite_series_l758_758457

theorem sum_infinite_series : 
  (∑ k in (Finset.range ∞), (3^k) / (9^k - 1)) = 1 / 2 :=
  sorry

end sum_infinite_series_l758_758457


namespace problem_l758_758238

noncomputable def a : ℝ := Real.log 25
noncomputable def b : ℝ := Real.log 36

theorem problem : 6^(a / b) + 5^(b / a) = 11 := by
  sorry

end problem_l758_758238


namespace martha_total_clothes_l758_758718

-- Define the conditions
def jackets_bought : ℕ := 4
def t_shirts_bought : ℕ := 9
def free_jacket_condition : ℕ := 2
def free_t_shirt_condition : ℕ := 3

-- Define calculations based on conditions
def free_jackets : ℕ := jackets_bought / free_jacket_condition
def free_t_shirts : ℕ := t_shirts_bought / free_t_shirt_condition
def total_jackets := jackets_bought + free_jackets
def total_t_shirts := t_shirts_bought + free_t_shirts
def total_clothes := total_jackets + total_t_shirts

-- Prove the total number of clothes
theorem martha_total_clothes : total_clothes = 18 :=
by
    sorry

end martha_total_clothes_l758_758718


namespace impossible_to_divide_into_three_similar_piles_l758_758651

def similar (a b : ℝ) : Prop :=
  a / b ≤ real.sqrt 2 ∧ b / a ≤ real.sqrt 2

theorem impossible_to_divide_into_three_similar_piles (pile : ℝ) (h : 0 < pile) :
  ¬ ∃ (x y z : ℝ), 
    x + y + z = pile ∧
    similar x y ∧ similar y z ∧ similar z x :=
by
  sorry

end impossible_to_divide_into_three_similar_piles_l758_758651


namespace sum_cn_formula_l758_758143

theorem sum_cn_formula (n : ℕ) :
  let a : ℕ → ℕ := λ i, i
  let b : ℕ → ℕ := λ i, 2^(i-1)
  let c : ℕ → ℕ := λ i, a i + b i
  (∑ i in finset.range n, c (i + 1)) = (n^2 + n) / 2 + 2^n - 1 :=
by sorry

end sum_cn_formula_l758_758143


namespace revenue_reaches_3000_profit_maximized_at_16_5_l758_758195

namespace Bayberries

noncomputable def daily_sales_volume (initial_volume : ℕ) (price_decrease : ℝ) : ℕ := 
  initial_volume + 20 * (price_decrease : ℕ)

noncomputable def daily_revenue (initial_price : ℝ) (price_decrease : ℝ) (initial_volume : ℕ) : ℝ :=
  (initial_price - price_decrease) * (daily_sales_volume initial_volume price_decrease)

theorem revenue_reaches_3000 (initial_price : ℝ) (initial_volume : ℕ) (price_decrease1 price_decrease2 : ℝ) :
  initial_price = 20 → initial_volume = 100 →
  daily_revenue initial_price price_decrease1 initial_volume = 3000 →
  daily_revenue initial_price price_decrease2 initial_volume = 3000 →
  price_decrease1 = 5 ∨ price_decrease1 = 10 :=
by
  sorry

noncomputable def daily_profit (cost_price selling_price : ℝ) (initial_volume : ℕ) (price_decrease : ℕ) : ℝ :=
  (selling_price - cost_price) * (daily_sales_volume initial_volume price_decrease)

theorem profit_maximized_at_16_5 (initial_price cost_price : ℝ) (initial_volume : ℕ) : 
  initial_price = 20 → cost_price = 8 → initial_volume = 100 →
  ∃ (selling_price : ℝ), selling_price = 16.5 ∧ 
  daily_profit cost_price selling_price initial_volume (selling_price - 8) = 1445 :=
by
  sorry

end Bayberries

end revenue_reaches_3000_profit_maximized_at_16_5_l758_758195


namespace find_m_abc_inequality_l758_758548

noncomputable def f (m x : ℝ) := m - |x - 2|

theorem find_m : 
  (∃ m, ∀ x, x ∈ Icc (-1 : ℝ) (1 : ℝ) → f m (x + 2) ≥ 0) → m = 1 :=
  sorry

theorem abc_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : 1/a + 1/(2*b) + 1/(3*c) = 1) :
  a + 2*b + 3*c ≥ 9 :=
  sorry

end find_m_abc_inequality_l758_758548


namespace compare_2_pow_n_n_sq_l758_758357

theorem compare_2_pow_n_n_sq (n : ℕ) (h : n > 0) :
  (n = 1 → 2^n > n^2) ∧
  (n = 2 → 2^n = n^2) ∧
  (n = 3 → 2^n < n^2) ∧
  (n = 4 → 2^n = n^2) ∧
  (n ≥ 5 → 2^n > n^2) :=
by sorry

end compare_2_pow_n_n_sq_l758_758357


namespace xy_product_approx_25_l758_758190

noncomputable def approx_eq (a b : ℝ) (ε : ℝ := 1e-6) : Prop :=
  |a - b| < ε

theorem xy_product_approx_25 (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) 
  (hxy : x / y = 36) (hy : y = 0.8333333333333334) : approx_eq (x * y) 25 :=
by
  sorry

end xy_product_approx_25_l758_758190


namespace ratio_is_three_to_one_l758_758788

variable (car_distance lawn_chair_distance birdhouse_distance : ℕ)

axiom car_condition : car_distance = 200
axiom lawn_chair_condition : lawn_chair_distance = 2 * car_distance
axiom birdhouse_condition : birdhouse_distance = 1200

theorem ratio_is_three_to_one
  (car_distance lawn_chair_distance birdhouse_distance : ℕ)
  (h_car : car_distance = 200)
  (h_lawn_chair : lawn_chair_distance = 2 * car_distance)
  (h_birdhouse : birdhouse_distance = 1200) :
  birdhouse_distance / lawn_chair_distance = 3 :=
by
  rw [h_lawn_chair, h_car, h_birdhouse]
  norm_num
  sorry

end ratio_is_three_to_one_l758_758788


namespace calculate_expression_l758_758421

theorem calculate_expression : 7 + 15 / 3 - 5 * 2 = 2 :=
by sorry

end calculate_expression_l758_758421


namespace reciprocal_of_neg_five_halves_l758_758778

theorem reciprocal_of_neg_five_halves : 
  let x : ℚ := -5/2 in 
  x⁻¹ = -2/5 :=
by
  sorry

end reciprocal_of_neg_five_halves_l758_758778


namespace focus_of_parabola_l758_758001

theorem focus_of_parabola (f d : ℝ) :
  (∀ x : ℝ, x^2 + (4*x^2 - f)^2 = (4*x^2 - d)^2) → 8*f + 8*d = 1 → f^2 = d^2 → f = 1/16 :=
by
  intro hEq hCoeff hSq
  sorry

end focus_of_parabola_l758_758001


namespace geom_seq_11th_term_l758_758756

/-!
The fifth and eighth terms of a geometric sequence are -2 and -54, respectively. 
What is the 11th term of this progression?
-/
theorem geom_seq_11th_term {a : ℕ → ℤ} (r : ℤ) 
  (h1 : a 5 = -2) (h2 : a 8 = -54) 
  (h3 : ∀ n : ℕ, a (n + 3) = a n * r ^ 3) : 
  a 11 = -1458 :=
sorry

end geom_seq_11th_term_l758_758756


namespace hyperbola_eccentricity_l758_758129

theorem hyperbola_eccentricity (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
    (C : P → Prop) 
    (angle_F1PF2 : ∀ {x y z : P}, ∀ (h : x ∈ C) (h₁ : y ∈ C) (h₂ : z ∈ C), is_angle x y z = 60)
    (dist_PF1_PF2 : ∀ (h : P ∈ C), dist P F1 = 3 * dist P F2) : 
    F1 ∈ C → F2 ∈ C → eccentricity C = (Real.sqrt 7) / 2 :=
by sorry

end hyperbola_eccentricity_l758_758129


namespace sum_first_five_special_l758_758399

def is_special (n : ℕ) : Prop :=
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p^2 * q^2

theorem sum_first_five_special :
  let special_numbers := [36, 100, 196, 484, 676]
  (∀ n ∈ special_numbers, is_special n) →
  special_numbers.sum = 1492 := by {
  sorry
}

end sum_first_five_special_l758_758399


namespace Rachel_picked_apples_l758_758732

theorem Rachel_picked_apples :
  let apples_from_first_tree := 8
  let apples_from_second_tree := 10
  let apples_from_third_tree := 12
  let apples_from_fifth_tree := 6
  apples_from_first_tree + apples_from_second_tree + apples_from_third_tree + apples_from_fifth_tree = 36 :=
by
  sorry

end Rachel_picked_apples_l758_758732


namespace value_of_b_l758_758309

noncomputable def polynomial_b : ℚ :=
  let d := 10 in              -- y-intercept gives d = 10
  let product_zeros := -d / 3 in
  let mean_zeros := -d / 3  in
  let a := 40 in               -- from the mean of zeros condition
  let c := -10 in              -- given as part of consistent properties
  (mean_zeros - 3 - a - c - d)

-- Statement to prove b = -139/3
theorem value_of_b : polynomial_b = -139/3 := 
  by
    sorry

end value_of_b_l758_758309


namespace ratio_seventh_terms_l758_758561

-- Definitions of sequences and their properties
variable (a_n b_n : ℕ → ℕ)
variable (A_n B_n : ℕ → ℕ)

-- Conditions:
-- 1. Sequences are arithmetic
-- 2. Sum of first n terms of {a_n} is A_n and for {b_n} is B_n
-- 3. Given ratio of sums of first n terms
axiom arithmetic_sequences (a_n b_n : ℕ → ℕ) : Prop
axiom sum_first_n_terms (A_n B_n : ℕ → ℕ) : ∀ n : ℕ, A_n = ∑ k in finset.range (n + 1), a_n k ∧ B_n = ∑ k in finset.range (n + 1), b_n k
axiom given_ratio (A_n B_n : ℕ → ℕ) : ∀ n : ℕ, A_n n / B_n n = (7 * n + 45) / (n + 3)

-- Theorem to prove
theorem ratio_seventh_terms : ∀ {a_n b_n : ℕ → ℕ} {A_n B_n : ℕ → ℕ},
  arithmetic_sequences a_n b_n →
  sum_first_n_terms A_n B_n →
  given_ratio A_n B_n →
  a_n 7 / b_n 7 = 17 / 2 := sorry

end ratio_seventh_terms_l758_758561


namespace count_triangles_with_area_2_5_l758_758327

-- Definition of the problem setup
def vertices_grid : list (ℕ × ℕ) :=
  [(r, c) | r ← [0, 1, 2, 3, 4], c ← [0, 1, 2]]

-- Definition for checking if 3 vertices form a triangle with area 2.5 cm²
def forms_triangle_with_area (v1 v2 v3 : ℕ × ℕ) : Prop :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  let (x3, y3) := v3
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) = 5

-- Main theorem: count the number of valid triangles
theorem count_triangles_with_area_2_5 : 
  let vertices := vertices_grid in
  124 = (cardinal.mk (set_of (λ (v1 v2 v3 : (ℕ × ℕ)), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ v1 ∈ vertices ∧ v2 ∈ vertices ∧ v3 ∈ vertices ∧ forms_triangle_with_area v1 v2 v3))).to_nat :=
sorry

end count_triangles_with_area_2_5_l758_758327


namespace total_cost_including_tax_l758_758225

-- Given conditions
def n := 150 -- Cost of Nikes
def w := 120 -- Cost of work boots
def j := 60  -- Original price of jacket
def d := 0.30 -- Discount on jacket
def t := 0.10 -- Tax rate

-- Prove that the total amount John paid including tax is $343.20
theorem total_cost_including_tax :
  let discounted_price_j := j * (1 - d),
      total_cost_before_tax := n + w + discounted_price_j,
      tax := total_cost_before_tax * t,
      total_cost := total_cost_before_tax + tax
  in total_cost = 343.20 := by
  sorry

end total_cost_including_tax_l758_758225


namespace geometric_sequence_problem_l758_758499

variable {a : ℕ → ℝ}

-- Given conditions
def geometric_sequence (a : ℕ → ℝ) :=
  ∃ q r, (∀ n, a (n + 1) = q * a n ∧ a 0 = r)

-- Define the conditions from the problem
def condition1 (a : ℕ → ℝ) :=
  a 3 + a 6 = 6

def condition2 (a : ℕ → ℝ) :=
  a 5 + a 8 = 9

-- Theorem to be proved
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (hgeom : geometric_sequence a)
  (h1 : condition1 a)
  (h2 : condition2 a) :
  a 7 + a 10 = 27 / 2 :=
sorry

end geometric_sequence_problem_l758_758499


namespace constant_sum_of_squares_l758_758493

-- Define the points O, A, B, C, and D
structure Point :=
  (x : ℝ)
  (y : ℝ)

def O : Point := ⟨0, 0⟩
def A (R : ℝ) : Point := ⟨R, 0⟩
def B (R : ℝ) : Point := ⟨0, R⟩
def C (x : ℝ) : Point := ⟨x, 0⟩
def D (R x : ℝ) : Point := ⟨x, Real.sqrt (R^2 - x^2)⟩

-- Define the distance squared between two points
def dist2 (P Q : Point) : ℝ := (P.x - Q.x)^2 + (P.y - Q.y)^2

-- The theorem to prove the constant sum of distances squared
theorem constant_sum_of_squares (R : ℝ) (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ R) :
  dist2 (B R) (C x) + dist2 (C x) (D R x) = 2 * R^2 :=
by
  sorry

end constant_sum_of_squares_l758_758493


namespace triangle_area_l758_758945

-- Define the points
def point1 : ℝ × ℝ := (0, 0)
def point2 : ℝ × ℝ := (0, 7)
def point3 : ℝ × ℝ := (-7, 0)

-- Define the base and height of the right triangle
def base : ℝ := 7
def height : ℝ := 7

-- Define the expected area
def expected_area : ℝ := 24.5

-- The theorem statement
theorem triangle_area : 1 / 2 * base * height = expected_area := by
  sorry

end triangle_area_l758_758945


namespace existence_of_large_difference_l758_758599

theorem existence_of_large_difference
  (n : ℕ) (hn : 2 ≤ n) 
  (f : (ℕ × ℕ) → ℕ) 
  (hf : bij_on f (λ p, p.snd = p.snd → p.fst = p.fst → (p.fst + p.snd < n²)) { i | i.fst < n ∧ i.snd < n }) :
  ∃ (i j : ℕ × ℕ), (i.snd = j.snd ∨ i.fst = j.fst) ∧ abs (f i - f j) ≥ n :=
sorry

end existence_of_large_difference_l758_758599


namespace distance_walked_l758_758343

variable (D : ℝ) 

def walk_time (distance speed : ℝ) : ℝ := distance / speed

theorem distance_walked :
  ∀ (D : ℝ),
    walk_time D 3 - walk_time D 4 = 1 / 2 →
    D = 6 :=
by
  intros D h
  sorry

end distance_walked_l758_758343


namespace incenter_divides_ratio_l758_758278

noncomputable def incenter_divides_angle_bisector_ratio (A B C O : Type) [IsTriangle A B C]
  (a b c : ℝ) (ha : a = side_length B C) (hb : b = side_length A C) (hc : c = side_length A B)
  (O_is_incenter : is_incenter A B C O) : Prop :=
  divides_in_ratio (angle_bisectors A B C O) (a + b) c

theorem incenter_divides_ratio (A B C O : Type) [IsTriangle A B C]
  (a b c : ℝ) (ha : a = side_length B C) (hb : b = side_length A C) (hc : c = side_length A B)
  (O_is_incenter : is_incenter A B C O) :
  incenter_divides_angle_bisector_ratio A B C O a b c ha hb hc O_is_incenter := sorry

end incenter_divides_ratio_l758_758278


namespace steve_ate_bags_l758_758617

-- Given conditions
def total_macaroons : Nat := 12
def weight_per_macaroon : Nat := 5
def num_bags : Nat := 4
def total_weight_remaining : Nat := 45

-- Derived conditions
def total_weight_macaroons : Nat := total_macaroons * weight_per_macaroon
def macaroons_per_bag : Nat := total_macaroons / num_bags
def weight_per_bag : Nat := macaroons_per_bag * weight_per_macaroon
def bags_remaining : Nat := total_weight_remaining / weight_per_bag

-- Proof statement
theorem steve_ate_bags : num_bags - bags_remaining = 1 := by
  sorry

end steve_ate_bags_l758_758617


namespace find_number_under_fourth_root_l758_758955

theorem find_number_under_fourth_root :
  ∃ (x : ℝ), (sqrt 1.21) / (sqrt 0.64) + (sqrt 1.44) / (sqrt x) = 3.0892857142857144 ∧ x = 0.49 :=
by
  sorry

end find_number_under_fourth_root_l758_758955


namespace focus_of_parabola_y_eq_4x_sq_l758_758010

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := (0 : ℝ, 1 / 16 : ℝ)
  in f

theorem focus_of_parabola_y_eq_4x_sq :
  (0, 1 / 16) = parabola_focus := by
  unfold parabola_focus
  sorry

end focus_of_parabola_y_eq_4x_sq_l758_758010


namespace surface_area_of_solid_of_revolution_l758_758403

theorem surface_area_of_solid_of_revolution (S α : ℝ) : 
  SurfaceAreaOfSolidOfRevolution S α = 4 * sqrt 2 * π * S * sin (α / 2 + π / 4) :=
sorry

end surface_area_of_solid_of_revolution_l758_758403


namespace number_of_ways_to_choose_3_cards_l758_758323

theorem number_of_ways_to_choose_3_cards (cards : Finset ℕ) (h : cards = Finset.range 130 + 502) :
  (∃ n = 119282, ∑ (t : Finset ℕ) in cards.powerset.filter (λ s, s.card = 3), (if ∑ x in t, x % 3 = 0 then 1 else 0) = n) :=
by
  have h_arith : ∀ x ∈ cards, ∃ k, x = 502 + 2 * k :=
    sorry
  have h_len : cards.card = 130 :=
    sorry
  have h_sum_div_3 : 
    ∑ (t : Finset ℕ) in cards.powerset.filter (λ s, s.card = 3), (if ∑ x in t, x % 3 = 0 then 1 else 0) = 119282 :=
    sorry
  exact ⟨119282, h_sum_div_3⟩

end number_of_ways_to_choose_3_cards_l758_758323


namespace sum_possible_values_q_l758_758710

/-- If natural numbers k, l, p, and q satisfy the given conditions,
the sum of all possible values of q is 4 --/
theorem sum_possible_values_q (k l p q : ℕ) 
    (h1 : ∀ a b : ℝ, a ≠ b → a * b = l → a + b = k → (∃ (c d : ℝ), c + d = (k * (l + 1)) / l ∧ c * d = (l + 2 + 1 / l))) 
    (h2 : a + 1 / b ≠ b + 1 / a)
    : q = 4 :=
sorry

end sum_possible_values_q_l758_758710


namespace weight_combinations_1985_l758_758993

theorem weight_combinations_1985 (s : set (set ℕ)) :
  (∀ w ∈ s, w.card = 4 ∧ ∀ x ∈ w, ∃ n : ℕ, 1 ≤ n ∧ n ≤ 1985) →
  s.card = 397 := 
sorry

end weight_combinations_1985_l758_758993


namespace num_divisible_digits_l758_758054

def divisible_by_n (num : ℕ) (n : ℕ) : Prop :=
  n ≠ 0 ∧ num % n = 0

def count_divisible_digits : ℕ :=
  (List.filter (λ n => divisible_by_n (150 + n) n)
    [1, 2, 3, 4, 5, 6, 7, 8, 9]).length

theorem num_divisible_digits : count_divisible_digits = 7 := by
  sorry

end num_divisible_digits_l758_758054


namespace course_length_l758_758342

-- Definitions based on conditions
def cyclist1_speed := 14   -- speed in miles per hour
def cyclist2_speed := 16   -- speed in miles per hour
def meeting_time := 1.5    -- time in hours

-- The problem statement requiring proof
theorem course_length :
  cyclist1_speed * meeting_time + cyclist2_speed * meeting_time = 45 := 
sorry

end course_length_l758_758342


namespace series_sum_eq_half_l758_758462

theorem series_sum_eq_half : (∑' n : ℕ, 1 ≤ n → ℚ, (3^n) / (9^n - 1)) = 1 / 2 := by
  sorry

end series_sum_eq_half_l758_758462


namespace correct_conclusions_l758_758490

noncomputable def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

def a_constraint (a : ℝ) : Prop := a < 0
def b_constraint (b : ℝ) : Prop := b > 0
def c_constraint (a b : ℝ) (b_constr : b_constraint b) : Prop := -1 / 2 < b / (2 * a)
def parabola_passes_through (x1 y1 x2 y2 : ℝ) (a b c : ℝ) : Prop :=
  parabola a b c x1 = y1 ∧ parabola a b c x2 = y2
def axis_of_symmetry (a b : ℝ) : ℝ := -b / (2 * a)

theorem correct_conclusions (a b c m : ℝ)
  (h1 : 1 < m) (h2 : m < 2)
  (h_parabola : parabola_passes_through (-1) 0 m 0 a b c)
  (h_a : a_constraint a)
  (h_b : b_constraint b) :
  b_constraint b ∧ (m = 3 / 2 → 3 * a + 2 * c = 0 → False) ∧
  (∀ (x1 x2 y1 y2 : ℝ), x1 < x2 → x1 + x2 > 1 → parabola a b c x1 = y1 → parabola a b c x2 = y2 → y1 > y2) ∧
  (a ≤ -1 → ∃ (x1 x2 : ℝ), a * x1^2 + b * x1 + c = 1 ∧ a * x2^2 + b * x2 + c = 1 ∧ x1 ≠ x2) := sorry

end correct_conclusions_l758_758490


namespace cost_of_bananas_l758_758708

theorem cost_of_bananas
  (apple_cost : ℕ)
  (orange_cost : ℕ)
  (banana_cost : ℕ)
  (num_apples : ℕ)
  (num_oranges : ℕ)
  (num_bananas : ℕ)
  (total_paid : ℕ) 
  (discount_threshold : ℕ)
  (discount_amount : ℕ)
  (total_fruits : ℕ)
  (total_without_discount : ℕ) :
  apple_cost = 1 → 
  orange_cost = 2 → 
  num_apples = 5 → 
  num_oranges = 3 → 
  num_bananas = 2 → 
  total_paid = 15 → 
  discount_threshold = 5 → 
  discount_amount = 1 → 
  total_fruits = num_apples + num_oranges + num_bananas →
  total_without_discount = (num_apples * apple_cost) + (num_oranges * orange_cost) + (num_bananas * banana_cost) →
  (total_without_discount - (discount_amount * (total_fruits / discount_threshold))) = total_paid →
  banana_cost = 3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end cost_of_bananas_l758_758708


namespace incorrect_statement_C_l758_758964

theorem incorrect_statement_C (x : ℝ) : 
  (x > -2) → 
  (y = (x + 2)^2 - 1) → 
  ∀ y1 y2, (x1 x2 : ℝ), (x1 > -2) ∧ (x2 > x1) → ((x1 + 2)^2 - 1 < (x2 + 2)^2 - 1) → y1 < y2 :=
sorry

end incorrect_statement_C_l758_758964


namespace impossibility_of_dividing_into_three_similar_piles_l758_758655

theorem impossibility_of_dividing_into_three_similar_piles:
  ∀ (x y z : ℝ), ¬ (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x + y + z = 1  ∧ (x ≤ sqrt 2 * y ∧ y ≤ sqrt 2 * x) ∧ (y ≤ sqrt 2 * z ∧ z ≤ sqrt 2 * y) ∧ (z ≤ sqrt 2 * x ∧ x ≤ sqrt 2 * z)) :=
by
  sorry

end impossibility_of_dividing_into_three_similar_piles_l758_758655


namespace inequality_one_inequality_two_l758_758412

variable (a b c : ℝ)

-- Conditions given in the problem
axiom positive_a : 0 < a
axiom positive_b : 0 < b
axiom positive_c : 0 < c
axiom sum_eq_one : a + b + c = 1

-- Statements to prove
theorem inequality_one : ab + bc + ac ≤ 1 / 3 :=
sorry

theorem inequality_two : a^2 / b + b^2 / c + c^2 / a ≥ 1 :=
sorry

end inequality_one_inequality_two_l758_758412


namespace contractor_absent_days_l758_758844

-- Definition of problem conditions
def total_days : ℕ := 30
def daily_wage : ℝ := 25
def daily_fine : ℝ := 7.5
def total_amount_received : ℝ := 620

-- Function to define the constraint equations
def equation1 (x y : ℕ) : Prop := x + y = total_days
def equation2 (x y : ℕ) : Prop := (daily_wage * x - daily_fine * y) = total_amount_received

-- The proof problem translation as Lean 4 statement
theorem contractor_absent_days (x y : ℕ) (h1 : equation1 x y) (h2 : equation2 x y) : y = 8 :=
by
  sorry

end contractor_absent_days_l758_758844


namespace binomial_coefficient_fourth_term_l758_758607

theorem binomial_coefficient_fourth_term :
  ∃ (T : ℤ), 
    T = ((nat.choose 6 3) * (2^3)) ∧
    T = 160 :=
by
  sorry

end binomial_coefficient_fourth_term_l758_758607


namespace impossible_divide_into_three_similar_l758_758675

noncomputable def sqrt2 : ℝ := Real.sqrt 2

def similar (x y : ℝ) : Prop :=
  x ≤ sqrt2 * y

theorem impossible_divide_into_three_similar (N : ℝ) :
  ¬ ∃ (x y z : ℝ), x + y + z = N ∧ similar x y ∧ similar y z ∧ similar x z := 
by
  sorry

end impossible_divide_into_three_similar_l758_758675


namespace series_equals_one_half_l758_758463

noncomputable def series_sum : ℕ → ℚ
| k := 3^k / (9^k - 1)

theorem series_equals_one_half :
  ∑' k, series_sum k = 1 / 2 :=
sorry

end series_equals_one_half_l758_758463


namespace fraction_eval_l758_758445

theorem fraction_eval :
  (3 / 7 + 5 / 8) / (5 / 12 + 1 / 4) = 177 / 112 :=
by
  sorry

end fraction_eval_l758_758445


namespace Dirichlet_properties_l758_758755

noncomputable def Dirichlet_function (x : ℝ) : ℝ :=
  if x ∈ ℚ then 1 else 0

theorem Dirichlet_properties :
  (∀ x, Dirichlet_function (Dirichlet_function x) = 1) ∧
  (∀ x, Dirichlet_function (-x) = Dirichlet_function x) ∧
  (∀ (x : ℝ) (T : ℚ), T ≠ 0 → Dirichlet_function (x + T) = Dirichlet_function x) ∧
  ∃ (x1 x2 x3 : ℝ), x1 = - (real.sqrt 3) / 3 ∧ x2 = 0 ∧ x3 = (real.sqrt 3) / 3 ∧
    Dirichlet_function x1 = 0 ∧ Dirichlet_function x2 = 1 ∧ Dirichlet_function x3 = 0 ∧
    (2 * (x1 - x2)^2 + (0 - 1)^2 = (2 * (x2 - x3)^2 + (1 - 0)^2) ∧ 
    2 * (x1 - x3)^2 + (0 - 0)^2 = (2 * (x2 - x3)^2 + (1 - 0)^2)) :=
by sorry

end Dirichlet_properties_l758_758755


namespace select_numbers_with_sum_713_l758_758322

noncomputable def is_suitable_sum (numbers : List ℤ) : Prop :=
  ∃ subset : List ℤ, subset ⊆ numbers ∧ (subset.sum % 10000 = 713)

theorem select_numbers_with_sum_713 :
  ∀ numbers : List ℤ, 
  numbers.length = 1000 → 
  (∀ n ∈ numbers, n % 2 = 1 ∧ n % 5 ≠ 0) →
  is_suitable_sum numbers :=
sorry

end select_numbers_with_sum_713_l758_758322


namespace area_of_trapezoid_l758_758204

theorem area_of_trapezoid (P Q R S T : Type) [trapezoid P Q R S] (h1 : PQ ∥ RS) 
  (h2 : intersect PR QS = T)
  (h3 : area_of_triangle PQT = 40)
  (h4 : area_of_triangle PRT = 25) :
  area_of_trapezoid PQRS = 105.625 :=
begin
  sorry
end

end area_of_trapezoid_l758_758204


namespace diagram_square_count_l758_758299

theorem diagram_square_count (n_operations : ℕ) (initial_squares : ℕ := 5) :
  let final_count := initial_squares + 3 * n_operations in
  n_operations = 6 → final_count = 23 :=
by
  intros n_operations_eq_six
  have n_operations_eq_6: n_operations = 6 := by assumption
  have final_count_eq_23: final_count = 23 := by
    rw [n_operations_eq_6]
    sorry
  exact final_count_eq_23

end diagram_square_count_l758_758299


namespace arc_segment_difference_l758_758205

noncomputable def arc_length_AB (r θ : ℝ) := r * θ

noncomputable def segment_length_AD : ℝ := 2 * Real.tan (Real.pi / 12) -- 15 degrees in radians

theorem arc_segment_difference :
  (|arc_length_AB 1 (Real.pi / 6) - segment_length_AD| = 0.0122) :=
by
  sorry

end arc_segment_difference_l758_758205


namespace find_n_l758_758937

theorem find_n (n : ℕ) (h1 : n < 128^97) (h2 : n.num_divisors = 2019) :
  n = 2^672 * 3^2 ∨ n = 2^672 * 5^2 ∨ n = 2^672 * 7^2 ∨ n = 2^672 * 11^2 := 
sorry

end find_n_l758_758937


namespace CeDe_Squared_Sum_l758_758237

theorem CeDe_Squared_Sum {O A B C D E : Point} (hO : Circle O 10) 
  (hAB : diameter O A B) (hCD : Chord O C D) (hE : E ∈ line AB ∩ line CD) 
  (hBE : distance B E = 3) (hAngle : ∠A E C = 45) :
  (distance C E)^2 + (distance D E)^2 = 200 :=
sorry

end CeDe_Squared_Sum_l758_758237


namespace hyperbola_eccentricity_l758_758117

def hyperbola_foci (F1 F2 P : ℝ) (θ : ℝ) (PF1 PF2 : ℝ) : Prop :=
  θ = 60 ∧ PF1 = 3 * PF2

def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity (F1 F2 P : ℝ) (θ PF1 PF2 : ℝ)
  (h : hyperbola_foci F1 F2 P θ PF1 PF2) :
  eccentricity 1 (sqrt 7 / 2) = sqrt 7 / 2 :=
by {
  sorry
}

end hyperbola_eccentricity_l758_758117


namespace find_a_10_l758_758142

-- We define the arithmetic sequence and sum properties
def arithmetic_seq (a_1 d : ℚ) (a_n : ℕ → ℚ) :=
  ∀ n, a_n n = a_1 + d * n

def sum_arithmetic_seq (a : ℕ → ℚ) (S_n : ℕ → ℚ) :=
  ∀ n, S_n n = n * (a 1 + a n) / 2

-- Conditions given in the problem
def given_conditions (a_1 : ℚ) (a_n : ℕ → ℚ) (S_n : ℕ → ℚ) :=
  arithmetic_seq a_1 1 a_n ∧ sum_arithmetic_seq a_n S_n ∧ S_n 6 = 4 * S_n 3

-- The theorem to prove
theorem find_a_10 (a_1 : ℚ) (a_n : ℕ → ℚ) (S_n : ℕ → ℚ) 
  (h : given_conditions a_1 a_n S_n) : a_n 10 = 19 / 2 :=
by sorry

end find_a_10_l758_758142


namespace point_B_position_l758_758265

theorem point_B_position (A B : ℝ) (d : ℝ) (hA : A = -1) (hD : |A - B| = d) : 
  d = 3 → (B = 2 ∨ B = -4) :=
by
  intro h3
  rw [hA] at hD
  have h : |(-1) - B| = 3 := by rw hD
  sorry

end point_B_position_l758_758265


namespace lcm_reciprocal_sum_le_one_l758_758230

-- Define the LCM function using the property of greatest common divisor
def lcm (x y : ℕ) : ℕ := (x * y) / Nat.gcd x y

-- Define the main statement
theorem lcm_reciprocal_sum_le_one (a b c d e : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
    (1 : ℝ) / lcm a b + (1 : ℝ) / lcm b c + (1 : ℝ) / lcm c d + 2 / (lcm d e : ℝ) ≤ 1 :=
begin
  sorry
end

end lcm_reciprocal_sum_le_one_l758_758230


namespace Tn_lt_2_l758_758496

variable {n : ℕ}
variable (a : ℕ → ℕ) (S T : ℕ → ℚ)

-- Conditions
axiom H1 : ∀ n : ℕ, a n = n
axiom H2 : S n = (n * (n + 1)) / 2
axiom H3 : T n = ∑ i in finset.range n, 1 / S i.succ

-- Proof obligation
theorem Tn_lt_2 : T n < 2 :=
by sorry

end Tn_lt_2_l758_758496


namespace problem_statement_l758_758539

variables (f : ℝ → ℝ)

def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def condition (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (2 - x)

theorem problem_statement (h_odd : is_odd f) (h_cond : condition f) : f 2010 = 0 := 
sorry

end problem_statement_l758_758539


namespace food_consumption_decrease_l758_758200

-- Define the given conditions
def initial_students := 100
def decrease_percentage := 0.10
def increase_percentage := 0.20
def initial_cost_per_student := 1.0

-- Define the problem in Lean
theorem food_consumption_decrease :
  let new_students := initial_students * (1 - decrease_percentage)
  let new_cost_per_student := initial_cost_per_student * (1 + increase_percentage)
  let target_total_cost := initial_students * initial_cost_per_student
  let new_consumption_per_student := target_total_cost / (new_students * new_cost_per_student)
  let decrease_in_consumption := 1 - new_consumption_per_student
  abs (decrease_in_consumption - 0.0741) < 0.0001 :=
by
  sorry

end food_consumption_decrease_l758_758200


namespace five_x_ge_seven_y_iff_exists_abcd_l758_758729

theorem five_x_ge_seven_y_iff_exists_abcd (x y : ℕ) :
  (5 * x ≥ 7 * y) ↔ ∃ (a b c d : ℕ), x = a + 2 * b + 3 * c + 7 * d ∧ y = b + 2 * c + 5 * d :=
by sorry

end five_x_ge_seven_y_iff_exists_abcd_l758_758729


namespace hyperbola_equation_l758_758538

theorem hyperbola_equation (a b : ℝ) : 
  let foci := [(0, -2), (0, 2)], 
      P := (-3, 2) in 
  (∃ a b : ℝ, a^2 + b^2 = 4 ∧ (9 / a^2) - (4 / b^2) = 1) → 
  a = 1 ∧ b = √3 → 
  y^2 - (x^2 / 3) = 1 := 
by 
  sorry

end hyperbola_equation_l758_758538


namespace max_points_of_intersection_one_circle_three_lines_l758_758809

-- Define the maximum number of points of intersection for 1 circle and 3 different straight lines
theorem max_points_of_intersection_one_circle_three_lines :
  ∀ (C : Type) (L : Type), 
  (circle : C) (lines : fin 3 → L), 
  (intersects : C → L → ℕ) (lines_intersect : L → L → ℕ),
  (∀ l, intersects circle (lines l) ≤ 2) →
  (∀ l1 l2, lines_intersect (lines l1) (lines l2) ≤ 1) →
  ∑ l, intersects circle (lines l) + ∑ (i j : fin 3) (h : i < j), lines_intersect (lines i) (lines j) = 9 :=
sorry

end max_points_of_intersection_one_circle_three_lines_l758_758809


namespace find_sixth_term_of_geometric_sequence_l758_758456

noncomputable def common_ratio (a b : ℚ) : ℚ := b / a

noncomputable def geometric_sequence_term (a r : ℚ) (k : ℕ) : ℚ := a * (r ^ (k - 1))

theorem find_sixth_term_of_geometric_sequence :
  geometric_sequence_term 5 (common_ratio 5 1.25) 6 = 5 / 1024 :=
by
  sorry

end find_sixth_term_of_geometric_sequence_l758_758456


namespace least_integer_with_exactly_eight_factors_l758_758825

theorem least_integer_with_exactly_eight_factors : ∃ n : ℕ, (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 8 ∨ d = 6 ∨ d = 12 ∨ d = 24) ∧
  (∀ m : ℕ, (∀ d : ℕ, d ∣ m → d = 1 ∨ d = 2 ∨ d = 3 ∨ d
= 4 ∨ d = 8 ∨ d = 6 ∨ d = 12 ∨ d = 24) → m = n) :=
begin
  sorry
end

end least_integer_with_exactly_eight_factors_l758_758825


namespace a_equal_1_sufficient_not_necessary_l758_758543

variable (a : ℝ)

def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0

def z : ℂ := (a^2 - 1) + (a - 2) * Complex.I

theorem a_equal_1_sufficient_not_necessary :
  (is_purely_imaginary(z a) ↔ a = 1) = false ∧
  (a = 1 → is_purely_imaginary(z a)) = true := by
  sorry

end a_equal_1_sufficient_not_necessary_l758_758543


namespace count_valid_n_divisibility_l758_758077

theorem count_valid_n_divisibility : 
  (finset.univ.filter (λ n : ℕ, n ∈ finset.range 10 ∧ (15 * n) % n = 0)).card = 5 :=
by sorry

end count_valid_n_divisibility_l758_758077


namespace find_focus_of_parabola_4x2_l758_758017

-- Defining what it means to be the focus of a parabola given an equation y = ax^2.
def is_focus (a : ℝ) (f : ℝ) : Prop :=
  ∀ (x : ℝ), (x ^ 2 + (a * x ^ 2 - f) ^ 2) = ((a * x ^ 2 - (f * 8)) ^ 2)

-- Specific instance of the parabola y = 4x^2.
def parabola_4x2 := (4 : ℝ)

theorem find_focus_of_parabola_4x2 : ∃ f : ℝ, is_focus parabola_4x2 f :=
begin
  use (1/16 : ℝ),
  sorry -- The proof will be filled in by the theorem prover.
end

end find_focus_of_parabola_4x2_l758_758017


namespace lower_limit_of_x_in_third_inequality_l758_758189

-- Definitions of conditions
variables (x : ℤ)
axiom cond1 : 0 < x ∧ x < 7
axiom cond2 : 0 < x ∧ x < 15
axiom cond3 : ∃ l, l < x ∧ x < 5
axiom cond4 : x < 3 ∧ 0 < x
axiom cond5 : x + 2 < 4
axiom cond6 : x = 1

theorem lower_limit_of_x_in_third_inequality (x : ℤ) :
  (∃ l, l < x ∧ x < 5) :=
by
  exact cond3
  sorry

end lower_limit_of_x_in_third_inequality_l758_758189


namespace archie_marbles_left_l758_758890

def initial_marbles : ℕ := 100
def street_loss_rate : ℚ := 0.6
def sewer_loss_rate : ℚ := 0.5

theorem archie_marbles_left :
  let remaining_after_street := initial_marbles * (1 - street_loss_rate) in
  let remaining_after_sewer := remaining_after_street * (1 - sewer_loss_rate) in
  remaining_after_sewer = 20 := 
by
  sorry

end archie_marbles_left_l758_758890


namespace max_distinct_colorings_5x5_l758_758326

theorem max_distinct_colorings_5x5 (n : ℕ) :
  ∃ N, N ≤ (n^25 + 4 * n^15 + n^13 + 2 * n^7) / 8 :=
sorry

end max_distinct_colorings_5x5_l758_758326


namespace simplify_and_evaluate_l758_758739

theorem simplify_and_evaluate (a : ℕ) (h : a = 2022) :
  (a - 1) / a / (a - 1 / a) = 1 / 2023 :=
by
  sorry

end simplify_and_evaluate_l758_758739


namespace cells_count_at_day_8_l758_758386

theorem cells_count_at_day_8 :
  let initial_cells := 3
  let common_ratio := 2
  let days := 8
  let interval := 2
  ∃ days_intervals, days_intervals = days / interval ∧ initial_cells * common_ratio ^ days_intervals = 48 :=
by
  sorry

end cells_count_at_day_8_l758_758386


namespace constant_term_in_expansion_l758_758968

-- Define the expression components
def sqrt_x (x : ℝ) : ℝ := real.sqrt x
def inv_cbrt_x (x : ℝ) : ℝ := 1 / (2 * real.cbrt x)

-- Given condition on binomial coefficients equality
lemma binomial_coefficients_equal {n : ℕ} (h : n = 5) :
  nat.choose n 2 = nat.choose n 3 :=
by
  rw [h, nat.choose, nat.factorial] -- This would expand to show equality of the two binomial coefficients
  sorry

-- Main statement: the constant term in the expansion is -5/4
theorem constant_term_in_expansion (x : ℝ) (h : true) :
  (sqrt_x x - inv_cbrt_x x)^5 = -5/4 :=
by
  -- Expansion and simplification steps leading to the constant term
  sorry

end constant_term_in_expansion_l758_758968


namespace distance_from_origin_to_point_is_15_l758_758595

-- Define the Euclidean distance function
def euclidean_distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define the given points
def origin : (ℝ × ℝ) := (0, 0)
def point : (ℝ × ℝ) := (9, -12)

-- Prove the distance from the origin to the point is 15
theorem distance_from_origin_to_point_is_15 :
  euclidean_distance origin.1 origin.2 point.1 point.2 = 15 := by
  sorry

end distance_from_origin_to_point_is_15_l758_758595


namespace least_integer_with_exactly_eight_factors_l758_758823

theorem least_integer_with_exactly_eight_factors : ∃ n : ℕ, (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 8 ∨ d = 6 ∨ d = 12 ∨ d = 24) ∧
  (∀ m : ℕ, (∀ d : ℕ, d ∣ m → d = 1 ∨ d = 2 ∨ d = 3 ∨ d
= 4 ∨ d = 8 ∨ d = 6 ∨ d = 12 ∨ d = 24) → m = n) :=
begin
  sorry
end

end least_integer_with_exactly_eight_factors_l758_758823


namespace impossible_to_divide_three_similar_parts_l758_758693

theorem impossible_to_divide_three_similar_parts 
  (n : ℝ) 
  (p : ℝ → Prop) 
  (similar : ℝ → ℝ → Prop) 
  (h_similar : ∀ a b : ℝ, similar a b ↔ a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2) : 
  ¬ ∃ p1 p2 p3 : ℝ, p p1 ∧ p p2 ∧ p p3 ∧ (p1 + p2 + p3 = n) ∧ similar p1 p2 ∧ similar p2 p3 ∧ similar p1 p3 :=
sorry

end impossible_to_divide_three_similar_parts_l758_758693


namespace investment_time_period_l758_758032

theorem investment_time_period :
  ∀ (A P : ℝ) (R : ℝ) (T : ℝ),
  A = 896 → P = 799.9999999999999 → R = 5 →
  (A - P) = (P * R * T / 100) → T = 2.4 :=
by
  intros A P R T hA hP hR hSI
  sorry

end investment_time_period_l758_758032


namespace quadrilateral_angle_l758_758201

theorem quadrilateral_angle (x y : ℝ) (h1 : 3 * x ^ 2 - x + 4 = 5) (h2 : x ^ 2 + y ^ 2 = 9) :
  x = (1 + Real.sqrt 13) / 6 :=
by
  sorry

end quadrilateral_angle_l758_758201


namespace union_of_A_and_B_eq_C_l758_758991

open Set

def A := {x : ℝ | -3 < x ∧ x < 3}
def B := {x : ℝ | x^2 - x - 6 ≤ 0}
def C := {x : ℝ | -3 < x ∧ x ≤ 3}

theorem union_of_A_and_B_eq_C : A ∪ B = C := 
by 
  sorry

end union_of_A_and_B_eq_C_l758_758991


namespace average_age_before_new_students_l758_758747

theorem average_age_before_new_students
  (A : ℝ) (N : ℕ)
  (h1 : N = 15)
  (h2 : 15 * 32 + N * A = (N + 15) * (A - 4)) :
  A = 40 :=
by {
  sorry
}

end average_age_before_new_students_l758_758747


namespace sin_complementary_angle_l758_758141

theorem sin_complementary_angle (θ : ℝ) (h1 : Real.tan θ = 2) (h2 : Real.cos θ < 0) : 
  Real.sin (Real.pi / 2 - θ) = -Real.sqrt 5 / 5 :=
sorry

end sin_complementary_angle_l758_758141


namespace units_digit_sum_base8_l758_758033

theorem units_digit_sum_base8 : 
  let units_digit (x : ℕ) : ℕ := x % 8 in
  units_digit (units_digit (65 + 74) + 3) = 4 := 
by 
  sorry

end units_digit_sum_base8_l758_758033


namespace eccentricity_of_hyperbola_l758_758122

open Real

-- Definitions of our conditions
variables {F1 F2 P : Point}
variables (a : ℝ) (m : ℝ)
variable (hyperbola_C : Hyperbola F1 F2)

-- Given conditions
axiom on_hyperbola : P ∈ hyperbola_C
axiom angle_F1P_F2 : angle F1 P F2 = π / 3
axiom distances : dist P F1 = 3 * dist P F2

-- Goal: Prove that the eccentricity of the hyperbola is sqrt(7)/2
theorem eccentricity_of_hyperbola : hyperbola.C.eccentricity = sqrt 7 / 2 := by
  sorry

end eccentricity_of_hyperbola_l758_758122


namespace count_valid_n_divisibility_l758_758076

theorem count_valid_n_divisibility : 
  (finset.univ.filter (λ n : ℕ, n ∈ finset.range 10 ∧ (15 * n) % n = 0)).card = 5 :=
by sorry

end count_valid_n_divisibility_l758_758076


namespace Q1_Q2_Q3_Q4_correct_choice_l758_758563

variable {m n : Line}
variable {α β : Plane}

-- Conditions:
axiom m_parallel_n : m ∥ n
axiom m_perp_alpha : m ⟂ α
axiom alpha_parallel_beta : α ∥ β
axiom m_not_in_alpha : ¬ (m ⊆ α)
axiom n_not_in_beta : ¬ (n ⊆ β)
axiom m_parallel_alpha : m ∥ α

-- Q1: If m ∥ n and m ⟂ α, then n ⟂ α
theorem Q1 : m ∥ n ∧ m ⟂ α → n ⟂ α := sorry

-- Q2: If α ∥ β, m ∉ α, and n ∉ β, then n ⟂ α
theorem Q2 : α ∥ β ∧ ¬ (m ⊆ α) ∧ ¬ (n ⊆ β) → n ⟂ α := sorry

-- Q3: If m ∥ n and m ∥ α, then n ∥ α
theorem Q3 : m ∥ n ∧ m ∥ α → n ∥ α := sorry

-- Q4: If α ∥ β, m ∥ n, and m ⟂ α, then n ⟂ β
theorem Q4 : α ∥ β ∧ m ∥ n ∧ m ⟂ α → n ⟂ β := sorry

-- The correct choice is A: Q1 Q4
theorem correct_choice : Q1 ∧ Q4 ∧ ¬ Q2 ∧ ¬ Q3 := sorry

end Q1_Q2_Q3_Q4_correct_choice_l758_758563


namespace greatest_two_digit_number_with_digit_product_16_l758_758810

def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digit_product (n m : ℕ) : Prop :=
  n * m = 16

def from_digits (n m : ℕ) : ℕ :=
  10 * n + m

theorem greatest_two_digit_number_with_digit_product_16 :
  ∀ n m, is_two_digit_number (from_digits n m) → digit_product n m → (82 ≥ from_digits n m) :=
by
  intros n m h1 h2
  sorry

end greatest_two_digit_number_with_digit_product_16_l758_758810


namespace intersection_A_B_is_C_l758_758519

def A := { x : ℤ | abs x < 3 }
def B := { x : ℤ | abs x > 1 }
def C := { -2, 2 : ℤ }

theorem intersection_A_B_is_C : (A ∩ B) = C := 
  sorry

end intersection_A_B_is_C_l758_758519


namespace complement_of_A_eq_l758_758152

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x > 1}

theorem complement_of_A_eq {U : Set ℝ} (U_eq : U = Set.univ) {A : Set ℝ} (A_eq : A = {x | x > 1}) :
    U \ A = {x | x ≤ 1} :=
by
  sorry

end complement_of_A_eq_l758_758152


namespace find_a11_times_a55_l758_758891

noncomputable def a_ij (i j : ℕ) : ℝ := 
  if i = 4 ∧ j = 1 then -2 else
  if i = 4 ∧ j = 3 then 10 else
  if i = 2 ∧ j = 4 then 4 else sorry

theorem find_a11_times_a55 
  (arithmetic_first_row : ∀ j, a_ij 1 (j + 1) = a_ij 1 1 + (j * 6))
  (geometric_columns : ∀ i j, a_ij (i + 1) j = a_ij 1 j * (2 ^ i) ∨ a_ij (i + 1) j = a_ij 1 j * ((-2) ^ i))
  (a24_eq_4 : a_ij 2 4 = 4)
  (a41_eq_neg2 : a_ij 4 1 = -2)
  (a43_eq_10 : a_ij 4 3 = 10) :
  a_ij 1 1 * a_ij 5 5 = -11 :=
by sorry

end find_a11_times_a55_l758_758891


namespace sprinting_vs_jogging_l758_758430

variable (distance_sprinted distance_jogged difference_in_distance : ℝ)

theorem sprinting_vs_jogging :
  distance_sprinted = 0.875 →
  distance_jogged = 0.75 →
  difference_in_distance = distance_sprinted - distance_jogged →
  difference_in_distance = 0.125 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sprinting_vs_jogging_l758_758430


namespace solution_exists_l758_758907

def valid_grid (grid : List (List Nat)) : Prop :=
  grid = [[2, 3, 6], [6, 3, 2]] ∨
  grid = [[2, 4, 8], [8, 4, 2]]

theorem solution_exists :
  ∃ (grid : List (List Nat)), valid_grid grid := by
  sorry

end solution_exists_l758_758907


namespace race_winner_l758_758792

theorem race_winner
  (faster : String → String → Prop)
  (Minyoung Yoongi Jimin Yuna : String)
  (cond1 : faster Minyoung Yoongi)
  (cond2 : faster Yoongi Jimin)
  (cond3 : faster Yuna Jimin)
  (cond4 : faster Yuna Minyoung) :
  ∀ s, s ≠ Yuna → faster Yuna s :=
by
  sorry

end race_winner_l758_758792


namespace b_is_integer_l758_758311

theorem b_is_integer (n : ℕ) (a : Fin n → ℝ) (b c : ℝ) (h0 : ∀ i, 0 ≤ a i)
  (h1 : ∑ i, a i = 1)
  (h2 : ∑ i, (↑i + 1) * a i = b)
  (h3 : ∑ i, (↑i + 1)^2 * a i = c)
  (h4 : c = b^2) : ∃ k : ℕ, b = k := 
sorry

end b_is_integer_l758_758311


namespace impossible_divide_into_three_similar_parts_l758_758688

noncomputable def similar (x y : ℝ) : Prop := x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

theorem impossible_divide_into_three_similar_parts (s : ℝ → ℝ → Prop) :
  (∀ s, similar s)) → ¬ (∃ a b c : ℝ, s a b → s b c → s c a → a + b + c = 1) :=
by
  intros h_similar h
  sorry

end impossible_divide_into_three_similar_parts_l758_758688


namespace trent_total_distance_l758_758338

-- Conditions
def walked_to_bus_stop : ℕ := 4
def bus_ride_to_library : ℕ := 7
def total_distance_to_library : ℕ := walked_to_bus_stop + bus_ride_to_library
def distance_back_home : ℕ := total_distance_to_library

-- Theorem stating that the total distance Trent traveled is 22 blocks
theorem trent_total_distance : 
  let total_distance := total_distance_to_library + distance_back_home in
  total_distance = 22 := 
by
  sorry

end trent_total_distance_l758_758338


namespace tan_equation_solution_set_l758_758573

theorem tan_equation_solution_set (x : ℝ) (h : 0 ≤ x ∧ x < real.pi) :
  (tan (4 * x - real.pi / 4) = 1) ↔ (x = real.pi / 8 ∨ x = 3 * real.pi / 8 ∨ x = 5 * real.pi / 8 ∨ x = 7 * real.pi / 8) :=
by
  sorry

end tan_equation_solution_set_l758_758573


namespace perimeter_of_regular_polygon_l758_758404

-- Define the problem conditions in Lean
def exterior_angle (n : ℕ) : ℝ := 360 / n
def side_length : ℝ := 7
def n (angle : ℝ) : ℕ := (360 / angle).to_nat

-- Define the number of sides based on the given exterior angle
def number_of_sides : ℕ := n 90

-- Define the perimeter of the polygon
def perimeter (sides : ℕ) (length : ℝ) : ℝ := sides * length

-- Theorem to be proven: the perimeter of the regular polygon is 28 units
theorem perimeter_of_regular_polygon : perimeter number_of_sides side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l758_758404


namespace focus_of_parabola_y_eq_4x_sq_l758_758014

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := (0 : ℝ, 1 / 16 : ℝ)
  in f

theorem focus_of_parabola_y_eq_4x_sq :
  (0, 1 / 16) = parabola_focus := by
  unfold parabola_focus
  sorry

end focus_of_parabola_y_eq_4x_sq_l758_758014


namespace profit_function_and_max_profit_l758_758862

theorem profit_function_and_max_profit :
  (∀ x : ℝ, 0 < x ∧ x < 50 → 
    (500 * x - (10 * x^2 + 100 * x + 800) - 250 = -10 * x^2 + 400 * x - 1050)) ∧ 
  (∀ x : ℝ, x ≥ 50 → 
    (500 * x - (504 * x + 10000 / (x - 2) - 6450) - 250 = -4 * x - 10000 / (x - 2) + 6200)) ∧
  (let maximum_profit := (52, -4 * 52 - 10000 / (52 - 2) + 6200) in
   maximum_profit = (52, 5792)) :=
begin
  sorry
end

end profit_function_and_max_profit_l758_758862


namespace divide_pile_l758_758681

theorem divide_pile (pile : ℝ) (similar : ℝ → ℝ → Prop) :
  (∀ x y, similar x y ↔ x ≤ y * Real.sqrt 2 ∧ y ≤ x * Real.sqrt 2) →
  ¬∃ a b c, a + b + c = pile ∧ similar a b ∧ similar b c ∧ similar a c :=
by sorry

end divide_pile_l758_758681


namespace find_real_numbers_l758_758455

theorem find_real_numbers (a b c : ℝ)    :
  (a + b + c = 3) → (a^2 + b^2 + c^2 = 35) → (a^3 + b^3 + c^3 = 99) → 
  (a = 1 ∧ b = -3 ∧ c = 5) ∨ (a = 1 ∧ b = 5 ∧ c = -3) ∨ 
  (a = -3 ∧ b = 1 ∧ c = 5) ∨ (a = -3 ∧ b = 5 ∧ c = 1) ∨
  (a = 5 ∧ b = 1 ∧ c = -3) ∨ (a = 5 ∧ b = -3 ∧ c = 1) :=
by intros h1 h2 h3; sorry

end find_real_numbers_l758_758455


namespace ratio_future_age_l758_758775

variable (S M : ℕ)

theorem ratio_future_age (h1 : (S : ℝ) / M = 7 / 2) (h2 : S - 6 = 78) : 
  ((S + 16) : ℝ) / (M + 16) = 5 / 2 := 
by
  sorry

end ratio_future_age_l758_758775


namespace impossible_to_divide_into_three_similar_parts_l758_758643

-- Define similarity condition
def similar_sizes (x y : ℝ) : Prop := x ≤ √2 * y

-- Main proof problem
theorem impossible_to_divide_into_three_similar_parts (x : ℝ) :
  ¬ ∃ (a b c : ℝ), a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes c a := 
sorry

end impossible_to_divide_into_three_similar_parts_l758_758643


namespace impossible_divide_into_three_similar_parts_l758_758685

noncomputable def similar (x y : ℝ) : Prop := x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

theorem impossible_divide_into_three_similar_parts (s : ℝ → ℝ → Prop) :
  (∀ s, similar s)) → ¬ (∃ a b c : ℝ, s a b → s b c → s c a → a + b + c = 1) :=
by
  intros h_similar h
  sorry

end impossible_divide_into_three_similar_parts_l758_758685


namespace smallest_determinant_and_min_ab_l758_758980

def determinant (a b : ℤ) : ℤ :=
  36 * b - 81 * a

theorem smallest_determinant_and_min_ab :
  (∃ (a b : ℤ), 0 < determinant a b ∧ determinant a b = 9 ∧ ∀ a' b', determinant a' b' = 9 → a' + b' ≥ a + b) ∧
  (∃ (a b : ℤ), a = 3 ∧ b = 7) :=
sorry

end smallest_determinant_and_min_ab_l758_758980


namespace Gina_makes_30_per_hour_l758_758481

variable (rose_cups_per_hour lily_cups_per_hour : ℕ)
variable (rose_cup_order lily_cup_order total_payment : ℕ)
variable (total_hours : ℕ)

def Gina_hourly_rate (rose_cups_per_hour: ℕ) (lily_cups_per_hour: ℕ) (rose_cup_order: ℕ) (lily_cup_order: ℕ) (total_payment: ℕ) : Prop :=
    let rose_time := rose_cup_order / rose_cups_per_hour
    let lily_time := lily_cup_order / lily_cups_per_hour
    let total_time := rose_time + lily_time
    total_payment / total_time = total_hours

theorem Gina_makes_30_per_hour :
    let rose_cups_per_hour := 6
    let lily_cups_per_hour := 7
    let rose_cup_order := 6
    let lily_cup_order := 14
    let total_payment := 90
    Gina_hourly_rate rose_cups_per_hour lily_cups_per_hour rose_cup_order lily_cup_order total_payment 30 :=
by
    sorry

end Gina_makes_30_per_hour_l758_758481


namespace triangle_ineq_l758_758852

theorem triangle_ineq (r : ℝ) (DE DF EF : ℝ) (F : ℝ) (t : ℝ) 
  (h1 : t = DF + EF)
  (h2 : DE = 4 * r)
  (h3 : 2 * r ≤ r * sqrt(2)) :
  t^2 ≤ 32 * r^2 :=
by
  sorry

end triangle_ineq_l758_758852


namespace second_smallest_five_digit_in_pascals_triangle_l758_758835

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem second_smallest_five_digit_in_pascals_triangle :
  (∃ n k : ℕ, n > 0 ∧ k > 0 ∧ (10000 ≤ binomial n k) ∧ (binomial n k < 100000) ∧
    (∀ m l : ℕ, m > 0 ∧ l > 0 ∧ (10000 ≤ binomial m l) ∧ (binomial m l < 100000) →
    (binomial n k < binomial m l → binomial n k ≥ 31465)) ∧  binomial n k = 31465) :=
sorry

end second_smallest_five_digit_in_pascals_triangle_l758_758835


namespace impossible_to_divide_three_similar_parts_l758_758700

theorem impossible_to_divide_three_similar_parts 
  (n : ℝ) 
  (p : ℝ → Prop) 
  (similar : ℝ → ℝ → Prop) 
  (h_similar : ∀ a b : ℝ, similar a b ↔ a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2) : 
  ¬ ∃ p1 p2 p3 : ℝ, p p1 ∧ p p2 ∧ p p3 ∧ (p1 + p2 + p3 = n) ∧ similar p1 p2 ∧ similar p2 p3 ∧ similar p1 p3 :=
sorry

end impossible_to_divide_three_similar_parts_l758_758700


namespace focus_of_parabola_l758_758000

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := 1 / 16 in
    (0, f)

theorem focus_of_parabola (x : ℝ) : 
  let focus := parabola_focus in
  focus = (0, 1 / 16) :=
by
  sorry

end focus_of_parabola_l758_758000


namespace at_least_half_girls_probability_eq_21_div_32_l758_758223

-- Define the total number of children
def n : ℕ := 6

-- Define probability of having a girl
def p : ℝ := 0.5

-- Define the probability of having at least 3 girls out of 6 children
noncomputable def probability_at_least_half_girls : ℝ :=
  (∑ i in (Icc 3 n), (.binom n i : ℝ) * p ^ i * (1 - p) ^ (n - i))

-- The goal is to prove that this probability is 21 / 32
theorem at_least_half_girls_probability_eq_21_div_32 :
  probability_at_least_half_girls = 21 / 32 :=
by
  sorry

end at_least_half_girls_probability_eq_21_div_32_l758_758223


namespace k_time_travel_l758_758376

theorem k_time_travel (x : ℝ) (h1 : x > 0) (h2 : x - 1/2 > 0) : 
  (∃ (time_M : ℝ), (time_M = 60 / (x - 1 / 2)) ∧ (60 / x = time_M - 2 / 3)) → 
  60 / x = 60 / x :=
by 
  intro h,
  exact rfl -- Placeholder not needing the proof steps

end k_time_travel_l758_758376


namespace complementary_angles_positive_difference_l758_758766

theorem complementary_angles_positive_difference :
  ∀ (θ₁ θ₂ : ℝ), (θ₁ + θ₂ = 90) → (θ₁ = 3 * θ₂) → (|θ₁ - θ₂| = 45) :=
by
  intros θ₁ θ₂ h₁ h₂
  sorry

end complementary_angles_positive_difference_l758_758766


namespace positive_integer_solution_count_l758_758492

noncomputable def number_of_solutions (n : ℕ) : ℕ :=
  Nat.divisorCount (n ^ 2) - 1

theorem positive_integer_solution_count (n : ℕ) (hn : 0 < n) :
  ∃ f : ℕ → ℕ → Bool, ( ∀ x y, f x y = (if (1 / n : ℚ) = (1 / x + 1 / y : ℚ) ∧ x ≠ y then true else false)) ∧
  (number_of_solutions n = ∑ x in (Finset.range (2 * n)), ∑ y in (Finset.range (2 * n)), if f x y then 1 else 0) :=
by
  sorry

end positive_integer_solution_count_l758_758492


namespace hyperbola_eccentricity_l758_758113

def hyperbola_foci (F1 F2 P : ℝ) (θ : ℝ) (PF1 PF2 : ℝ) : Prop :=
  θ = 60 ∧ PF1 = 3 * PF2

def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity (F1 F2 P : ℝ) (θ PF1 PF2 : ℝ)
  (h : hyperbola_foci F1 F2 P θ PF1 PF2) :
  eccentricity 1 (sqrt 7 / 2) = sqrt 7 / 2 :=
by {
  sorry
}

end hyperbola_eccentricity_l758_758113


namespace symmetric_points_on_line_l758_758551

noncomputable def f (x : ℝ) : ℝ := (Real.exp (x - 1)) / (Real.exp x)

theorem symmetric_points_on_line (k : ℝ) (h : k ≠ 0) :
  (∃ x₁ y₁, y₁ = k * (x₁ + 1) ∧ y₁ = f x₁ ∧
   ∃ x₂ y₂, y₂ = k * (x₂ + 1) ∧ y₂ = f x₂ ∧ x₂ = -x₁) →
  k ∈ (-∞, -1) ∪ (-1, 0) :=
sorry

end symmetric_points_on_line_l758_758551


namespace initial_distance_planes_l758_758723

theorem initial_distance_planes (speed_A speed_B : ℝ) (time_seconds : ℝ) (time_hours : ℝ) (distance_A distance_B : ℝ) (total_distance : ℝ) :
  speed_A = 240 ∧ speed_B = 360 ∧ time_seconds = 72000 ∧ time_hours = 20 ∧ 
  time_hours = time_seconds / 3600 ∧
  distance_A = speed_A * time_hours ∧ 
  distance_B = speed_B * time_hours ∧ 
  total_distance = distance_A + distance_B →
  total_distance = 12000 :=
by
  intros
  sorry

end initial_distance_planes_l758_758723


namespace cost_equation_l758_758750

def cost (W : ℕ) : ℕ :=
  if W ≤ 10 then 5 * W + 10 else 7 * W - 10

theorem cost_equation (W : ℕ) : cost W = 
  if W ≤ 10 then 5 * W + 10 else 7 * W - 10 :=
by
  -- Proof goes here
  sorry

end cost_equation_l758_758750


namespace cost_of_painting_walls_l758_758366

theorem cost_of_painting_walls :
  let room_length := 10
  let room_width := 7
  let room_height := 5
  let door_length := 1
  let door_height := 3
  let num_doors := 2
  let window_large_length := 2
  let window_large_height := 1.5
  let window_small_length := 1
  let window_small_height := 1.5
  let num_small_windows := 2
  let cost_per_sqm := 3

  let wall1_area := 2 * (room_length * room_height)
  let wall2_area := 2 * (room_width * room_height)
  let total_wall_area := wall1_area + wall2_area

  let door_area := door_length * door_height
  let total_door_area := num_doors * door_area

  let large_window_area := window_large_length * window_large_height
  let small_window_area := window_small_length * window_small_height
  let total_window_area := large_window_area + (num_small_windows * small_window_area)

  let area_to_paint := total_wall_area - (total_door_area + total_window_area)
  let total_cost := area_to_paint * cost_per_sqm

  total_cost = 474 :=
by
  let room_length := 10
  let room_width := 7
  let room_height := 5
  let door_length := 1
  let door_height := 3
  let num_doors := 2
  let window_large_length := 2
  let window_large_height := 1.5
  let window_small_length := 1
  let window_small_height := 1.5
  let num_small_windows := 2
  let cost_per_sqm := 3

  let wall1_area := 2 * (room_length * room_height)
  let wall2_area := 2 * (room_width * room_height)
  let total_wall_area := wall1_area + wall2_area

  let door_area := door_length * door_height
  let total_door_area := num_doors * door_area

  let large_window_area := window_large_length * window_large_height
  let small_window_area := window_small_length * window_small_height
  let total_window_area := large_window_area + (num_small_windows * small_window_area)

  let area_to_paint := total_wall_area - (total_door_area + total_window_area)
  let total_cost := area_to_paint * cost_per_sqm

  have : total_cost = 474 := sorry
  exact this

end cost_of_painting_walls_l758_758366


namespace focus_of_parabola_l758_758028

theorem focus_of_parabola (x f d : ℝ) (h : ∀ x, y = 4 * x^2 → (x = 0 ∧ y = f) → PF^2 = PQ^2 ∧ 
(PF^2 = x^2 + (4 * x^2 - f) ^ 2) := (PQ^2 = (4 * x^2 - d) ^ 2)) : 
  f = 1 / 16 := 
sorry

end focus_of_parabola_l758_758028


namespace verify_amounts_l758_758737

-- Define the compound interest formula
def compound_interest (P : Float) (r : Float) (n : Float) (t : Float) : Float :=
  P * (1 + r / n) ^ (n * t)

-- Define the initial conditions, interest rates, compounding periods, and durations
def P_sam : Float := 6000
def r_sam : Float := 0.08
def n_sam : Float := 4
def t_sam : Float := 3

def P_priya : Float := 8000
def r_priya : Float := 0.10
def n_priya : Float := 12
def t_priya : Float := 2

def P_rahul : Float := 10000
def r_rahul : Float := 0.12
def n_rahul : Float := 1
def t_rahul : Float := 4

-- Define the expected results
def A_sam_expected : Float := 7614.94
def A_priya_expected : Float := 9755.13
def A_rahul_expected : Float := 15748

-- The theorem to prove the expected final amounts
theorem verify_amounts :
  compound_interest P_sam r_sam n_sam t_sam ≈ A_sam_expected ∧
  compound_interest P_priya r_priya n_priya t_priya ≈ A_priya_expected ∧
  compound_interest P_rahul r_rahul n_rahul t_rahul ≈ A_rahul_expected :=
by
  sorry

end verify_amounts_l758_758737


namespace focus_of_parabola_l758_758005

theorem focus_of_parabola (f d : ℝ) :
  (∀ x : ℝ, x^2 + (4*x^2 - f)^2 = (4*x^2 - d)^2) → 8*f + 8*d = 1 → f^2 = d^2 → f = 1/16 :=
by
  intro hEq hCoeff hSq
  sorry

end focus_of_parabola_l758_758005


namespace part1_part2_part3_l758_758591

variables {A B C D E F H : Type*} [has_coe_to_fun A B C D E F H] 
variables (a b c R : ℝ)

-- Given conditions in the problem
variable (ΔABC : Type*) -- Triangle ABC
variable [is_altitude : (ΔABC → A B C → Prop)]
variable {orthocenter : (ΔABC → Prop)} -- H is the orthocenter

-- Part 1
theorem part1 (h : orthocenter ΔABC) : 
  (AH H * HD H) = (a^2 + b^2 + c^2)/2 - 4*R^2 := sorry

-- Part 2
theorem part2 :
  S_AEF = triangle * cos^2 A := sorry

-- Part 3
theorem part3 :
  DE + EF + FD = (a * b * c) / (2 * R^2) := sorry

end part1_part2_part3_l758_758591


namespace polynomial_value_sum_l758_758086

theorem polynomial_value_sum
  (a b c d : ℝ)
  (f : ℝ → ℝ)
  (Hf : ∀ x, f x = x^4 + a * x^3 + b * x^2 + c * x + d)
  (H1 : f 1 = 1) (H2 : f 2 = 2) (H3 : f 3 = 3) :
  f 0 + f 4 = 28 :=
sorry

end polynomial_value_sum_l758_758086


namespace exists_point_D_geometric_mean_l758_758229

theorem exists_point_D_geometric_mean (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  (∃ D: ℝ, 0 < D ∧ D < A + B ∧ ((D * D = (A * (B - D)) ∨ (A * (B - D) = D * D)))
    ↔ (sin A * sin B ≤ (sin (C / 2))^2)) := sorry

end exists_point_D_geometric_mean_l758_758229


namespace repeated_digit_sum_in_sequence_l758_758090

-- Definitions for the problem
def polynomial (R : Type*) [CommRing R] := list R

noncomputable def decimal_sum (n : ℕ) : ℕ :=
sorries -- Placeholder for the actual implementation

-- Statement of the theorem
theorem repeated_digit_sum_in_sequence
  (P : polynomial ℤ)
  (a_n : ℕ → ℕ)
  (hP : ∀ n, a_n n = decimal_sum (eval n P))
  : ∃ N, ∃ᶠ n in at_top, a_n n = N := sorry

end repeated_digit_sum_in_sequence_l758_758090


namespace matrix_addition_l758_758501

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 0], ![-1, 2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![2, 4], ![1, -3]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 4], ![0, -1]]

theorem matrix_addition : A + B = C := by
    sorry

end matrix_addition_l758_758501


namespace relationship_among_a_b_c_l758_758998

theorem relationship_among_a_b_c (a b c : ℝ)
  (h1 : a = 2 ^ 0.3)
  (h2 : b = Real.log 3 / Real.log 0.2)
  (h3 : c = Real.log 2 / Real.log 3) :
  b < c ∧ c < a := 
sorry

end relationship_among_a_b_c_l758_758998


namespace max_value_expr_l758_758453

-- Define the expression
def expr (a b c d : ℝ) : ℝ :=
  a + b + c + d - a * b - b * c - c * d - d * a

-- The main theorem
theorem max_value_expr :
  (∀ (a b c d : ℝ), 0 ≤ a ∧ a ≤ 1 → 0 ≤ b ∧ b ≤ 1 → 0 ≤ c ∧ c ≤ 1 → 0 ≤ d ∧ d ≤ 1 → expr a b c d ≤ 2) ∧
  (∃ (a b c d : ℝ), 0 ≤ a ∧ a = 1 ∧ 0 ≤ b ∧ b = 0 ∧ 0 ≤ c ∧ c = 1 ∧ 0 ≤ d ∧ d = 0 ∧ expr a b c d = 2) :=
  by
  sorry

end max_value_expr_l758_758453


namespace find_y_such_that_log_36_6y_eq_3_l758_758447

theorem find_y_such_that_log_36_6y_eq_3 : 
  ∃ (y : ℝ), log 36 (6 * y) = 3 ∧ y = 7776 :=
by
  use 7776
  split
  · apply log_eq_iff_eq_rpow
    -- Here we expect the logarithm properties
    -- For simplicity here, assume all necessary properties about log are available
    sorry
  · simp
    sorry

end find_y_such_that_log_36_6y_eq_3_l758_758447


namespace hyperbola_eccentricity_l758_758118

def hyperbola_foci (F1 F2 P : ℝ) (θ : ℝ) (PF1 PF2 : ℝ) : Prop :=
  θ = 60 ∧ PF1 = 3 * PF2

def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity (F1 F2 P : ℝ) (θ PF1 PF2 : ℝ)
  (h : hyperbola_foci F1 F2 P θ PF1 PF2) :
  eccentricity 1 (sqrt 7 / 2) = sqrt 7 / 2 :=
by {
  sorry
}

end hyperbola_eccentricity_l758_758118


namespace seats_required_l758_758378

def children := 58
def per_seat := 2
def seats_needed (children : ℕ) (per_seat : ℕ) := children / per_seat

theorem seats_required : seats_needed children per_seat = 29 := 
by
  sorry

end seats_required_l758_758378


namespace number_of_minimally_intersecting_triples_mod_1000_l758_758431

-- Definition of a minimally intersecting triple (A, B, C)
def isMinimallyIntersecting (A B C : Set ℕ) : Prop :=
  (|A ∩ B| = 1) ∧ (|B ∩ C| = 1) ∧ (|C ∩ A| = 1) ∧ (A ∩ B ∩ C = ∅)

-- The universal set of elements from which A, B, and C are subsets
def universalSet : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- The main theorem statement
theorem number_of_minimally_intersecting_triples_mod_1000 : 
  (card {abc | ∃ (A B C : Set ℕ), A ⊆ universalSet ∧ B ⊆ universalSet ∧ C ⊆ universalSet ∧ isMinimallyIntersecting A B C}) % 1000 = 64 :=
  sorry

end number_of_minimally_intersecting_triples_mod_1000_l758_758431


namespace dice_sum_prob_l758_758188

theorem dice_sum_prob :
  (3 / 6) * (3 / 6) * (2 / 5) * (1 / 6) * 2 = 13 / 216 :=
by sorry

end dice_sum_prob_l758_758188


namespace arccos_range_l758_758979

theorem arccos_range (a : ℝ) (x : ℝ) (h1 : x = Real.sin a) (h2 : a ∈ Set.Icc (-Real.pi / 4) (3 * Real.pi / 4)) :
  Set.Icc 0 (3 * Real.pi / 4) = Set.image Real.arccos (Set.Icc (-Real.sqrt 2 / 2) 1) :=
by
  sorry

end arccos_range_l758_758979


namespace prob_of_sum_3_7_8_is_correct_l758_758801

def prob_face (die_faces : List ℕ) (n : ℕ) : ℚ :=
  (die_faces.count n : ℚ) / (die_faces.length : ℚ)

def prob_sum (die1_faces die2_faces : List ℕ) (sum_val : ℕ) : ℚ :=
  die1_faces.to_finset.to_list.bind (λ x, 
    die2_faces.to_finset.to_list.map (λ y, 
      if x + y = sum_val then prob_face die1_faces x * prob_face die2_faces y else 0)).sum

def prob_sum_3_7_8 (die1_faces die2_faces : List ℕ) : ℚ :=
  prob_sum die1_faces die2_faces 3 + prob_sum die1_faces die2_faces 7 + prob_sum die1_faces die2_faces 8

noncomputable def first_die := [1, 1, 2, 2, 3, 5]
noncomputable def second_die := [2, 3, 4, 5, 6, 7]

theorem prob_of_sum_3_7_8_is_correct :
  prob_sum_3_7_8 first_die second_die = 13 / 36 := by sorry

end prob_of_sum_3_7_8_is_correct_l758_758801


namespace determine_b_l758_758734

def p (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x + b

def q (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x + (b + 360)

theorem determine_b (r s : ℝ) (a b : ℝ) (h1 : p r a b = 0) (h2 : p s a b = 0) 
(h3 : q (r + 3) a b = 0) (h4 : q (s - 2) a b = 0) : 
b = -1330 / 27 ∨ b = -6340 / 27 :=
begin
  sorry
end

end determine_b_l758_758734


namespace main_theorem_l758_758972

noncomputable def f (x a b : ℝ) := log x - a * x - b

theorem main_theorem (a b : ℝ) (a_pos : a > 0) :
  (∃ x : ℝ, 0 < x ∧ f x a b ≥ 0) → a * b ≤ 1 / Real.exp 2 := 
by
  sorry

end main_theorem_l758_758972


namespace hyperbola_eccentricity_thm_l758_758106

noncomputable def hyperbola_eccentricity 
  (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
  (angle_F1PF2 : ℝ) (dist_PF1 dist_PF2 : ℝ)
  (H1 : angle_F1PF2 = 60)
  (H2 : dist_PF1 = 3 * dist_PF2) : ℝ :=
let a := dist_PF2 in
let c := (a * sqrt 7) / 2 in 
c / a

theorem hyperbola_eccentricity_thm
  (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
  (angle_F1PF2 : ℝ) (dist_PF1 dist_PF2 : ℝ)
  (H1 : angle_F1PF2 = 60)
  (H2 : dist_PF1 = 3 * dist_PF2) :
  @hyperbola_eccentricity F1 F2 P _ _ _ angle_F1PF2 dist_PF1 dist_PF2 H1 H2 = (sqrt 7) / 2 :=
sorry

end hyperbola_eccentricity_thm_l758_758106


namespace intersection_eq_l758_758521

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_eq : A ∩ B = ({-2, 2} : Set ℤ) :=
by
  sorry

end intersection_eq_l758_758521


namespace seven_digit_number_subtraction_l758_758803

theorem seven_digit_number_subtraction 
  (n : ℕ)
  (d1 d2 d3 d4 d5 d6 d7 : ℕ)
  (h1 : n = d1 * 10^6 + d2 * 10^5 + d3 * 10^4 + d4 * 10^3 + d5 * 10^2 + d6 * 10 + d7)
  (h2 : d1 < 10 ∧ d2 < 10 ∧ d3 < 10 ∧ d4 < 10 ∧ d5 < 10 ∧ d6 < 10 ∧ d7 < 10)
  (h3 : n - (d1 + d3 + d4 + d5 + d6 + d7) = 9875352) :
  n - (d1 + d3 + d4 + d5 + d6 + d7 - d2) = 9875357 :=
sorry

end seven_digit_number_subtraction_l758_758803


namespace boys_camp_percentage_l758_758198

theorem boys_camp_percentage (x : ℕ) (total_boys : ℕ) (percent_science : ℕ) (not_science_boys : ℕ) 
    (percent_not_science : ℕ) (h1 : not_science_boys = percent_not_science * (x / 100) * total_boys) 
    (h2 : percent_not_science = 100 - percent_science) (h3 : percent_science = 30) 
    (h4 : not_science_boys = 21) (h5 : total_boys = 150) : x = 20 :=
by 
  sorry

end boys_camp_percentage_l758_758198


namespace option_c_option_d_l758_758784

-- Definitions and conditions
def Sn (n : ℕ) : ℤ := -(n^2 : ℤ) + 7 * n

-- Proving the options
theorem option_c (n : ℕ) (h : n > 4) : 
  let a_n := Sn n - Sn (n - 1)
  in a_n < 0 :=
by
  sorry

theorem option_d : 
  ∀ n : ℕ, (Sn 3 ≥ Sn n ∧ Sn 4 ≥ Sn n) :=
by
  sorry

end option_c_option_d_l758_758784


namespace sum_infinite_series_l758_758458

theorem sum_infinite_series : 
  (∑ k in (Finset.range ∞), (3^k) / (9^k - 1)) = 1 / 2 :=
  sorry

end sum_infinite_series_l758_758458


namespace cos_value_in_second_quadrant_l758_758534

variable (a : ℝ)
variables (h1 : π/2 < a ∧ a < π) (h2 : Real.sin a = 5/13)

theorem cos_value_in_second_quadrant : Real.cos a = -12/13 :=
  sorry

end cos_value_in_second_quadrant_l758_758534


namespace triangle_sides_sum_squares_equal_l758_758722

open Finset

theorem triangle_sides_sum_squares_equal :
  ∃ (A B C D E F G H I : ℕ), {A, B, C, D, E, F, G, H, I} = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  (A^2 + B^2 + C^2 = 95) ∧ 
  (D^2 + E^2 + F^2 = 95) ∧ 
  (G^2 + H^2 + I^2 = 95) :=
begin
  refine ⟨2, 3, 4, 1, 6, 8, 5, 7, 9, by simp, by norm_num, by norm_num, by norm_num⟩
end

end triangle_sides_sum_squares_equal_l758_758722


namespace total_number_of_monthly_allocations_l758_758401

-- Define the conditions given in the problem
def budget : ℕ := 12600
def total_spent_in_6_months : ℕ := 6580
def overage : ℕ := 280

-- Define the correct answer in terms of a proof problem
theorem total_number_of_monthly_allocations : 
  budget / ((total_spent_in_6_months - overage) / 6) = 12 :=
by simp; sorry

end total_number_of_monthly_allocations_l758_758401


namespace infinite_chains_of_tangent_circles_exist_l758_758728

theorem infinite_chains_of_tangent_circles_exist
  (R₁ R₂ : Circle) 
  (h_disjoint : ¬ (R₁ ⋂ R₂).nonempty)
  (T₁ : Circle)
  (h_tangent_R₁ : T₁.TangentTo R₁)
  (h_tangent_R₂ : T₁.TangentTo R₂)
  : ∃ (T : ℕ → Circle), ∀ n, 
    (T n).TangentTo R₁ ∧ 
    (T n).TangentTo R₂ ∧ 
    (∀ m, T m.TangentTo (T (m + 1)) ∧ T m.TangentTo (T (m - 1))) :=
sorry

end infinite_chains_of_tangent_circles_exist_l758_758728


namespace tan_ratio_l758_758084

theorem tan_ratio (α : ℝ) (h : 5 * sin (2 * α) = sin (2 * (real.pi / 180))) :
  tan (α + real.pi / 180) / tan (α - real.pi / 180) = -3 / 2 :=
by
  sorry

end tan_ratio_l758_758084


namespace ten_faucets_fill_50_gallon_in_60_seconds_l758_758039

-- Define the conditions
def five_faucets_fill_tub (faucet_rate : ℝ) : Prop :=
  5 * faucet_rate * 8 = 200

def all_faucets_same_rate (tub_capacity time : ℝ) (num_faucets : ℕ) (faucet_rate : ℝ) : Prop :=
  num_faucets * faucet_rate * time = tub_capacity

-- Define the main theorem to be proven
theorem ten_faucets_fill_50_gallon_in_60_seconds (faucet_rate : ℝ) :
  (∃ faucet_rate, five_faucets_fill_tub faucet_rate) →
  all_faucets_same_rate 50 1 10 faucet_rate →
  10 * faucet_rate * (1 / 60) = 50 :=
by
  sorry

end ten_faucets_fill_50_gallon_in_60_seconds_l758_758039


namespace impossibility_of_dividing_into_three_similar_piles_l758_758654

theorem impossibility_of_dividing_into_three_similar_piles:
  ∀ (x y z : ℝ), ¬ (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x + y + z = 1  ∧ (x ≤ sqrt 2 * y ∧ y ≤ sqrt 2 * x) ∧ (y ≤ sqrt 2 * z ∧ z ≤ sqrt 2 * y) ∧ (z ≤ sqrt 2 * x ∧ x ≤ sqrt 2 * z)) :=
by
  sorry

end impossibility_of_dividing_into_three_similar_piles_l758_758654


namespace impossible_to_divide_into_three_similar_parts_l758_758642

-- Define similarity condition
def similar_sizes (x y : ℝ) : Prop := x ≤ √2 * y

-- Main proof problem
theorem impossible_to_divide_into_three_similar_parts (x : ℝ) :
  ¬ ∃ (a b c : ℝ), a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes c a := 
sorry

end impossible_to_divide_into_three_similar_parts_l758_758642


namespace angle_inequality_in_triangle_l758_758610

variable {A B C M : Type} [MetricSpace A]
variable (triangle_ABC : Triangle A B C) (M_lies_BC : LiesOn M B C)

theorem angle_inequality_in_triangle :
  (dist A M - dist A C) * dist B C ≤ (dist A B - dist A C) * dist M C :=
  sorry

end angle_inequality_in_triangle_l758_758610


namespace equilateral_triangle_min_perimeter_l758_758274

theorem equilateral_triangle_min_perimeter (a b c : ℝ) (S : ℝ) :
  let p := (a + b + c) / 2 in
  let area := sqrt (p * (p - a) * (p - b) * (p - c)) in
  area = S →
  ∀ a b c, p = (a + b + c) / 2 →
  sqrt (p * (p - a) * (p - b) * (p - c)) = S →
  a = b = c :=
by sorry

end equilateral_triangle_min_perimeter_l758_758274


namespace option_C_incorrect_l758_758988

variables {Plane Line : Type} (alpha beta : Plane) (m n : Line)
variables (perpendicular parallel intersect : Line → Plane → Prop)

-- The conditions
def m_parallel_alpha := parallel m alpha
def alpha_intersect_beta_eq_n := intersect α β = n

-- The theorem stating that option C is incorrect
theorem option_C_incorrect (h1 : m_parallel_alpha) (h2 : alpha_intersect_beta_eq_n) : ¬ parallel m n :=
by sorry

end option_C_incorrect_l758_758988


namespace annual_salary_is_20_l758_758177

-- Define the conditions
variable (months_worked : ℝ) (total_received : ℝ) (turban_price : ℝ)
variable (S : ℝ)

-- Actual values from the problem
axiom h1 : months_worked = 9 / 12
axiom h2 : total_received = 55
axiom h3 : turban_price = 50

-- Define the statement to prove
theorem annual_salary_is_20 : S = 20 := by
  -- Conditions derived from the problem
  have cash_received := total_received - turban_price
  have fraction_of_salary := months_worked * S
  -- Given the servant worked 9 months and received Rs. 55 including Rs. 50 turban
  have : cash_received = fraction_of_salary := by sorry
  -- Solving the equation 3/4 S = 5 for S
  have : S = 20 := by sorry
  sorry -- Final proof step

end annual_salary_is_20_l758_758177


namespace part1_part2_l758_758990

variables (a b : ℝ) (f g : ℝ → ℝ)

-- Step 1: Given a > 0, b > 0 and f(x) = |x - a| - |x + b|, prove that if max(f) = 3, then a + b = 3.
theorem part1 (ha : a > 0) (hb : b > 0) (hf : ∀ x, f x = abs (x - a) - abs (x + b)) (hmax : ∀ x, f x ≤ 3) :
  a + b = 3 :=
sorry

-- Step 2: For g(x) = -x^2 - ax - b, if g(x) < f(x) for all x ≥ a, prove that 1/2 < a < 3.
theorem part2 (ha : a > 0) (hb : b > 0) (hf : ∀ x, f x = abs (x - a) - abs (x + b)) (hmax : ∀ x, f x ≤ 3)
    (hg : ∀ x, g x = -x^2 - a * x - b) (hcond : ∀ x, x ≥ a → g x < f x) :
    1 / 2 < a ∧ a < 3 :=
sorry

end part1_part2_l758_758990


namespace fruit_seller_apples_l758_758845

theorem fruit_seller_apples (original_apples : ℝ) (sold_percent : ℝ) (remaining_apples : ℝ)
  (h1 : sold_percent = 0.40)
  (h2 : remaining_apples = 420)
  (h3 : original_apples * (1 - sold_percent) = remaining_apples) :
  original_apples = 700 :=
by
  sorry

end fruit_seller_apples_l758_758845


namespace intersection_eq_l758_758524

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_eq : A ∩ B = ({-2, 2} : Set ℤ) :=
by
  sorry

end intersection_eq_l758_758524


namespace sqrt_of_expression_l758_758358

theorem sqrt_of_expression (x : ℝ) (h : x = 2) : Real.sqrt (2 * x - 3) = 1 :=
by
  rw [h]
  simp
  sorry

end sqrt_of_expression_l758_758358


namespace log_expression_simplifies_to_zero_l758_758837

theorem log_expression_simplifies_to_zero :
  (log 3 270 / log 54 3 - log 3 540 / log 27 3) = 0 :=
by
  sorry

end log_expression_simplifies_to_zero_l758_758837


namespace φ_is_homomorphism_and_correct_kernel_l758_758631

variables (a b x y : ℤ) (h : a * x + b * y = 1)

def φ (u v : ℤ) : ℤ / (a * b) := ((u * b * y) + (v * a * x)) % (a * b)

theorem φ_is_homomorphism_and_correct_kernel :
  ∀ (u v : ℤ), is_group_hom (λ p : ℤ × ℤ, φ a b x y p.1 p.2) ∧
    ∀ (u v : ℤ), φ a b x y u v = 0 ↔ ∃ (k l : ℤ), u = k * a ∧ v = l * b :=
begin
  sorry
end

end φ_is_homomorphism_and_correct_kernel_l758_758631


namespace variation_of_variables_l758_758578

variables (k j : ℝ) (x y z : ℝ)

theorem variation_of_variables (h1 : x = k * y^2) (h2 : y = j * z^3) : ∃ m : ℝ, x = m * z^6 :=
by
  -- Placeholder for the proof
  sorry

end variation_of_variables_l758_758578


namespace intersection_of_complements_l758_758560

open Set

universe u

constant U : Set Int := {-1, 0, 1, 2, 3}
constant A : Set Int := {-1, 0, 2}
constant B : Set Int := {0, 1}

theorem intersection_of_complements :
  (U \ A) ∩ (U \ B) = {3} := 
by
  sorry

end intersection_of_complements_l758_758560


namespace semicircle_segment_length_l758_758287

theorem semicircle_segment_length (AB AC BD : ℝ) (h_AB : AB = 13) (h_AC : AC = 5) (h_BD : BD = 5) : 
  ∃ (a b : ℕ), a.gcd b = 1 ∧ (CD : ℚ) = rat.mk a b ∧ a + b = 132 :=
by
  sorry

end semicircle_segment_length_l758_758287


namespace impossible_to_divide_into_three_similar_piles_l758_758648

def similar (a b : ℝ) : Prop :=
  a / b ≤ real.sqrt 2 ∧ b / a ≤ real.sqrt 2

theorem impossible_to_divide_into_three_similar_piles (pile : ℝ) (h : 0 < pile) :
  ¬ ∃ (x y z : ℝ), 
    x + y + z = pile ∧
    similar x y ∧ similar y z ∧ similar z x :=
by
  sorry

end impossible_to_divide_into_three_similar_piles_l758_758648


namespace extreme_values_range_a_l758_758162

theorem extreme_values (a : ℝ) (h : (a * Real.exp(1) + 2) * Real.log (Real.exp(1)) - ((Real.exp(1))^2 + a * Real.exp(1) - a - 1) = (Real.exp(1)-2*Real.exp(1)+2/Real.exp(1))) :
  ∃ x_max, ∀ x, f(x) ≤ f(x_max) ∧ f(x_max) = 0 :=
by
  sorry

theorem range_a (a : ℝ) (h_range : ∀ x > 1, (a * x + 2) * Real.log x - (x^2 + a * x - a - 1) < 0) :
  a ∈ Set.Iic 4 :=
by
  sorry

noncomputable def f : ℝ → ℝ := λ x, (λ a x, (a * x + 2) * log x - (x^2 + a * x - a - 1)) a x

end extreme_values_range_a_l758_758162


namespace even_diff_colored_segments_l758_758440

theorem even_diff_colored_segments 
  (n : ℕ) 
  (A : ℕ → Prop) 
  (colors : ℕ → bool) 
  (h0 : colors 0 = false) 
  (hn : colors n = false) : 
  (∑ i in finset.range n, if colors i ≠ colors (i + 1) then 1 else 0) % 2 = 0 := 
sorry

end even_diff_colored_segments_l758_758440


namespace a_n_negative_S_n_maximum_l758_758783

def S (n : ℕ) : ℕ → ℤ := λ n, -n^2 + 7 * n

-- 1. Prove ∀ n > 4, a_n < 0 given S_n = -n^2 + 7n
theorem a_n_negative (n : ℕ) (h : n > 4) : ∀ n > 4, (-2 * (↑n : ℤ) + 8) < 0 :=
by {
  assume (n : ℕ) (h : n > 4),
  calc (-2 * (↑n : ℤ) + 8)
      = 8 - 2 * (↑n : ℤ) : by ring
  ... < 0 : by linarith,
}

-- 2. Prove S_n reaches its maximum value when n = 3 or 4 given S_n = -n^2 + 7n
theorem S_n_maximum : ∃ (n : ℕ) (max1 max2 : ℕ → ℤ), 
  (S 3 ≥ S n) ∧ (S 4 ≥ S n) :=
by {
  use 3,
  use 4,
  assume (n : ℕ),
  have axis_symmetry : 7 / 2 = 3.5 := by norm_num,
  by_cases h1: n ≤ 3,
  { linarith },
  { have h2 : 3 < n,
    from lt_of_not_ge h1, 
    linarith,
  },
}

#check a_n_negative
#check S_n_maximum

end a_n_negative_S_n_maximum_l758_758783


namespace angle_measure_l758_758258

theorem angle_measure (m n: Line) (angle1 angle2 angle5 angle6: ℝ) 
  (h1 : m ∥ n) 
  (h2 : angle1 = (1/6) * angle2) 
  (h3 : angle5 = angle1) 
  (h4 : angle2 + angle5 = 180) 
  (supplementary: angle6 + angle5 = 180) : 
  angle6 = 1080 / 7 :=
by
  sorry

end angle_measure_l758_758258


namespace find_d_minus_e_l758_758040

noncomputable def a_n (n : ℕ) (h : n > 1) : ℝ := 1 / (Real.log 3003 / Real.log n)

def d : ℝ := a_n 3 (by norm_num) + a_n 4 (by norm_num) + a_n 5 (by norm_num) + a_n 6 (by norm_num)
def e : ℝ := a_n 15 (by norm_num) + a_n 16 (by norm_num) + a_n 17 (by norm_num) + a_n 18 (by norm_num) + a_n 19 (by norm_num)

theorem find_d_minus_e : 
  d - e = -2 * Real.log 2 / Real.log 3003 - 2 * Real.log 3 / Real.log 3003 - Real.log 7 / Real.log 3003 - Real.log 23 / Real.log 3003 :=
sorry

end find_d_minus_e_l758_758040


namespace simplify_fraction_l758_758289

theorem simplify_fraction (x : ℤ) : 
    (2 * x + 3) / 4 + (5 - 4 * x) / 3 = (-10 * x + 29) / 12 := 
by
  sorry

end simplify_fraction_l758_758289


namespace product_of_roots_l758_758946

theorem product_of_roots :
  let a := 24
  let c := -216
  ∀ x : ℝ, (24 * x^2 + 36 * x - 216 = 0) → (c / a = -9) :=
by
  intros
  sorry

end product_of_roots_l758_758946


namespace focus_of_parabola_l758_758003

theorem focus_of_parabola (f d : ℝ) :
  (∀ x : ℝ, x^2 + (4*x^2 - f)^2 = (4*x^2 - d)^2) → 8*f + 8*d = 1 → f^2 = d^2 → f = 1/16 :=
by
  intro hEq hCoeff hSq
  sorry

end focus_of_parabola_l758_758003


namespace find_parallel_line_through_intersection_l758_758941

/-- Define the intersection point of two lines -/
def intersection (a1 b1 c1 a2 b2 c2 : ℝ) : ℝ × ℝ :=
  let x := (b1 * c2 - c1 * b2) / (a1 * b2 - b1 * a2)
  let y := (c1 * a2 - a1 * c2) / (a1 * b2 - b1 * a2)
  (x, y)

/-- Statement of the main problem -/
theorem find_parallel_line_through_intersection :
  ∃ (a b c : ℝ),
    a * 1 + b * 2 + c = 0 ∧
    a * 3 + b * 6 - c = 0 ∧
    a = 3 ∧ b = 6 ∧ c = -2 :=
by
  sorry

end find_parallel_line_through_intersection_l758_758941


namespace angle_AKP_eq_angle_PKC_l758_758730

-- Definitions based on the conditions
variables {α : Type} [euclidean_geometry α]
variables (A B C D O M N P K : α)
variable [incircle A B C D O]
variables [conincident (line_through A D) (line_through B C) M]
variables [conincident (line_through A B) (line_through C D) N]
variables [conincident (line_through A C) (line_through B D) P]
variables [conincident (line_through O P) (line_through M N) K]

-- Lean 4 statement to prove the corresponding problem
theorem angle_AKP_eq_angle_PKC :
  ∠(A, K, P) = ∠(P, K, C) := sorry

end angle_AKP_eq_angle_PKC_l758_758730


namespace positive_difference_complementary_angles_l758_758769

theorem positive_difference_complementary_angles (a b : ℝ) 
  (h1 : a + b = 90) 
  (h2 : 3 * b = a) :
  |a - b| = 45 :=
by
  sorry

end positive_difference_complementary_angles_l758_758769


namespace intersection_of_A_and_B_l758_758510

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} :=
by
  sorry

end intersection_of_A_and_B_l758_758510


namespace even_positive_integer_exists_roots_of_unity_sum_one_l758_758938

theorem even_positive_integer_exists_roots_of_unity_sum_one {n : ℕ} (h : n > 0) :
  (∃ i j k : ℕ, i < n ∧ j < n ∧ k < n ∧ (complex.exp (2 * real.pi * complex.I * i / n) + 
  complex.exp (2 * real.pi * complex.I * j / n) + complex.exp (2 * real.pi * complex.I * k / n) = 1)) ↔ n % 2 = 0 :=
by sorry

end even_positive_integer_exists_roots_of_unity_sum_one_l758_758938


namespace negation_proposition_l758_758773

theorem negation_proposition (a b : ℝ) (h : a ≤ b) : 2^a ≤ 2^b - 1 :=
sorry

end negation_proposition_l758_758773


namespace range_of_triangle_area_l758_758761

-- Definitions and assumptions based on the provided conditions
def line (x y : ℝ) : Prop := x + y - 6 = 0
def ellipse (x y : ℝ) : Prop := x^2 + 2 * y^2 = 6

-- Points A and B are where the line intersects the x-axis and y-axis
def A := (6 : ℝ, 0 : ℝ)
def B := (0 : ℝ, 6 : ℝ)

-- The area of the triangle formed by points A, B, and P
noncomputable def area_of_triangle (A B P : ℝ × ℝ) : ℝ :=
  (1 / 2) * |(A.1 * (B.2 - P.2) + B.1 * (P.2 - A.2) + P.1 * (A.2 - B.2))|

-- The main theorem stating the range of the area of triangle ABP
theorem range_of_triangle_area : 
  ∀ P : ℝ × ℝ, ellipse P.1 P.2 → 9 ≤ area_of_triangle A B P ∧ area_of_triangle A B P ≤ 27 :=
begin
  sorry
end

end range_of_triangle_area_l758_758761


namespace probability_convex_quadrilateral_l758_758927

def num_points := 8
def num_chords : ℕ := nat.choose num_points 2
def num_select_four_chords : ℕ := nat.choose num_chords 4
def num_convex_quadrilateral : ℕ := nat.choose num_points 4

theorem probability_convex_quadrilateral (h : num_points = 8) :
  (num_convex_quadrilateral : ℚ) / num_select_four_chords = 2 / 585 :=
by {
  sorry
}

end probability_convex_quadrilateral_l758_758927


namespace product_of_100_consecutive_not_100th_power_l758_758279

theorem product_of_100_consecutive_not_100th_power (n k : ℕ) :
  ∏ i in finset.range (100), (n + i) ≠ k ^ 100 := 
sorry

end product_of_100_consecutive_not_100th_power_l758_758279


namespace solve_sqrt_equation_l758_758292

theorem solve_sqrt_equation (x : ℝ) (h: sqrt (3 + 4 * x) = 7) : x = 11.5 :=
by
  sorry

end solve_sqrt_equation_l758_758292


namespace trent_total_distance_l758_758336

theorem trent_total_distance
  (house_to_bus : ℕ)
  (bus_to_library : ℕ)
  (house_to_bus = 4)
  (bus_to_library = 7)
  : (house_to_bus + bus_to_library) * 2 = 22 :=
by
  sorry

end trent_total_distance_l758_758336


namespace ten_faucets_fill_50_gallons_in_60_seconds_l758_758036

-- Define the rate of water dispensed by each faucet and time calculation
def fill_tub_time (num_faucets tub_volume faucet_rate : ℝ) : ℝ :=
  tub_volume / (num_faucets * faucet_rate)

-- Given conditions
def five_faucets_fill_200_gallons_in_8_minutes : Prop :=
  ∀ faucet_rate : ℝ, 5 * faucet_rate * 8 = 200

-- Main theorem: Ten faucets fill a 50-gallon tub in 60 seconds
theorem ten_faucets_fill_50_gallons_in_60_seconds 
  (faucet_rate : ℝ) 
  (h : five_faucets_fill_200_gallons_in_8_minutes) : 
  fill_tub_time 10 50 faucet_rate = 1 := by
  sorry

end ten_faucets_fill_50_gallons_in_60_seconds_l758_758036


namespace min_value_fraction_l758_758533

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) : 
  (∃ (c : ℝ), c ≤ (2 / a + 1 / b)) ∧ (∀ (c : ℝ), c > 8 -> (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 2 * b = 1 ∧ 2 / a + 1 / b = c)) := 
begin
  sorry
end

end min_value_fraction_l758_758533


namespace cos_sub_eq_five_over_eight_l758_758137

theorem cos_sub_eq_five_over_eight (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5 / 8 := 
by sorry

end cos_sub_eq_five_over_eight_l758_758137


namespace count_valid_digits_l758_758047

theorem count_valid_digits :
  {n : ℕ | 1 ≤ n ∧ n ≤ 9 ∧ 15 * n % n = 0}.card = 7 :=
by sorry

end count_valid_digits_l758_758047


namespace area_of_region_below_and_left_l758_758433

theorem area_of_region_below_and_left (x y : ℝ) :
  (∃ (x y : ℝ), (x - 4)^2 + y^2 = 4^2) ∧ y ≤ 0 ∧ y ≤ x - 4 →
  π * 4^2 / 4 = 4 * π :=
by sorry

end area_of_region_below_and_left_l758_758433


namespace impossibility_of_dividing_into_three_similar_piles_l758_758657

theorem impossibility_of_dividing_into_three_similar_piles:
  ∀ (x y z : ℝ), ¬ (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x + y + z = 1  ∧ (x ≤ sqrt 2 * y ∧ y ≤ sqrt 2 * x) ∧ (y ≤ sqrt 2 * z ∧ z ≤ sqrt 2 * y) ∧ (z ≤ sqrt 2 * x ∧ x ≤ sqrt 2 * z)) :=
by
  sorry

end impossibility_of_dividing_into_three_similar_piles_l758_758657


namespace binomial_variance_proof_l758_758035

open ProbabilityTheory

noncomputable def binomial_variance (n : ℕ) (p : ℝ) : ℝ :=
  let q : ℝ := 1 - p
  n * p * q

theorem binomial_variance_proof (n : ℕ) (p : ℝ) (X : ℕ → ℝ) (hX : ∀ k, X k = if k < n then if k = 1 then p else 0 else 0) :
  variance (binomial n p) = binomial_variance n p :=
sorry

end binomial_variance_proof_l758_758035


namespace k_value_l758_758236

-- Conditions as definitions
variable (y : ℝ)
variable (k : ℝ)
variable (hy : log 8 5 = y)
variable (hk : log 2 125 = k * y)

-- Statement of the proof problem
theorem k_value (y : ℝ) (k : ℝ) (hy : log 8 5 = y) (hk : log 2 125 = k * y) : k = 9 :=
sorry

end k_value_l758_758236


namespace intersection_A_B_is_C_l758_758520

def A := { x : ℤ | abs x < 3 }
def B := { x : ℤ | abs x > 1 }
def C := { -2, 2 : ℤ }

theorem intersection_A_B_is_C : (A ∩ B) = C := 
  sorry

end intersection_A_B_is_C_l758_758520


namespace min_value_3x_4y_l758_758185

theorem min_value_3x_4y {x y : ℝ} (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) :
    3 * x + 4 * y ≥ 5 :=
sorry

end min_value_3x_4y_l758_758185


namespace hyperbola_eccentricity_thm_l758_758101

noncomputable def hyperbola_eccentricity 
  (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
  (angle_F1PF2 : ℝ) (dist_PF1 dist_PF2 : ℝ)
  (H1 : angle_F1PF2 = 60)
  (H2 : dist_PF1 = 3 * dist_PF2) : ℝ :=
let a := dist_PF2 in
let c := (a * sqrt 7) / 2 in 
c / a

theorem hyperbola_eccentricity_thm
  (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
  (angle_F1PF2 : ℝ) (dist_PF1 dist_PF2 : ℝ)
  (H1 : angle_F1PF2 = 60)
  (H2 : dist_PF1 = 3 * dist_PF2) :
  @hyperbola_eccentricity F1 F2 P _ _ _ angle_F1PF2 dist_PF1 dist_PF2 H1 H2 = (sqrt 7) / 2 :=
sorry

end hyperbola_eccentricity_thm_l758_758101


namespace max_value_correct_min_value_correct_max_value_set_correct_min_value_set_correct_l758_758942

noncomputable def function_def (x : ℝ) : ℝ := 3 * Real.cos(2 * x + Real.pi / 3)

def max_value : ℝ := 3

def min_value : ℝ := -3

def max_value_set : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi - Real.pi / 6}

def min_value_set : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.pi / 3}

theorem max_value_correct : ∀ x, function_def x ≤ max_value :=
sorry

theorem min_value_correct : ∀ x, function_def x ≥ min_value :=
sorry

theorem max_value_set_correct : ∀ x, function_def x = max_value ↔ x ∈ max_value_set :=
sorry

theorem min_value_set_correct : ∀ x, function_def x = min_value ↔ x ∈ min_value_set :=
sorry

end max_value_correct_min_value_correct_max_value_set_correct_min_value_set_correct_l758_758942


namespace volume_of_parallelepiped_l758_758847

theorem volume_of_parallelepiped (a b c : ℕ) (V : ℝ) 
  (parallelepiped : {P : Type} [affine_space P ℝ] [has_lattice_points P])
  (P_contains_a : ∃ points : finset P, points.card = a ∧ points.subset P.interior)
  (P_contains_b : ∃ points : finset P, points.card = b ∧ points ⊆ P.boundary ∧ points ⊆ P.faces)
  (P_contains_c : ∃ points : finset P, points.card = c ∧ points ⊆ P.edges) :
  V = 1 + a + (b / 2) + (c / 4) := 
sorry

end volume_of_parallelepiped_l758_758847


namespace least_positive_integer_with_eight_factors_l758_758812

theorem least_positive_integer_with_eight_factors :
  ∃ n : ℕ, n = 24 ∧ (8 = (nat.factors n).length) :=
sorry

end least_positive_integer_with_eight_factors_l758_758812


namespace Sm_of_5_eq_2525_l758_758413

variable (x : ℝ)

def Sm (m : ℕ) : ℝ := x^m + (1/x)^m

theorem Sm_of_5_eq_2525
  (h : x + (1 / x) = 5) :
  Sm x 5 = 2525 := 
sorry

end Sm_of_5_eq_2525_l758_758413


namespace expr_f_max_f_angle_C_prod_ab_area_ABC_l758_758173

variables (a b c : ℝ) (A B C : ℝ)
def a_vec (x : ℝ) := (2 * real.cos x, real.sin x - real.cos x)
def b_vec (x : ℝ) := (real.sqrt 3 * real.sin x, real.sin x + real.cos x)
def f (x : ℝ) := a_vec x.1 * b_vec x.1 + a_vec x.2 * b_vec x.2

-- Conditions provided in the problem
axiom h1 : a + b = 2 * real.sqrt 3
axiom h2 : c = real.sqrt 6
axiom h3 : f C = 2

-- Target statements to prove
theorem expr_f (x : ℝ) : f x = 2 * real.sin (2 * x - real.pi / 6) :=
sorry

theorem max_f : { x : ℝ | ∃ k : ℤ, x = k * real.pi + real.pi / 3 } =
{ x : ℝ | f x = 2 } :=
sorry

theorem angle_C : C = real.pi / 3 :=
sorry

theorem prod_ab : a * b = 2 :=
sorry

theorem area_ABC : real.sqrt 3 / 2 :=
sorry

end expr_f_max_f_angle_C_prod_ab_area_ABC_l758_758173


namespace percent_of_amount_rounded_l758_758806

-- Define the given percentage and amount
def percentage : ℝ := 0.02 
def amount : ℝ := 12356

-- Define the calculation function
def calc_percent (p : ℝ) (a : ℝ) : ℝ := (p / 100) * a

-- The rounding to nearest cent
def round_to_nearest_cent (x : ℝ) : ℝ := (Real.floor (x * 100 + 0.5)) / 100

-- The theorem that needs to be proven
theorem percent_of_amount_rounded :
  round_to_nearest_cent (calc_percent percentage amount) = 2.47 := 
sorry

end percent_of_amount_rounded_l758_758806


namespace restaurant_vegetarian_dishes_l758_758405

theorem restaurant_vegetarian_dishes (n : ℕ) : 
    5 ≥ 2 → 200 < Nat.choose 5 2 * Nat.choose n 2 → n ≥ 7 :=
by
  intros h_combinations h_least
  sorry

end restaurant_vegetarian_dishes_l758_758405


namespace positive_difference_x_coordinates_at_y_10_l758_758500

noncomputable def line_l_x_at_y_10 : ℝ :=
  let slope_l := (0 - 4) / (4 - 0) in
  let b_l := 4 in
  -(10 - b_l) / slope_l

noncomputable def line_m_x_at_y_10 : ℝ :=
  let slope_m := (0 - 3) / (6 - 0) in
  let b_m := 3 in
  -(10 - b_m) / slope_m

theorem positive_difference_x_coordinates_at_y_10 :
  |line_l_x_at_y_10 - line_m_x_at_y_10| = 8 :=
by
  -- Convert slope calculations to use given points
  let slope_l := (0 - 4) / (4 - 0)
  let slope_m := (0 - 3) / (6 - 0)
  let bl := 4
  let bm := 3
  -- x-coordinates when y = 10
  have xl_10 := -(10 - bl) / slope_l
  have xm_10 := -(10 - bm) / slope_m
  -- Calculate the positive difference
  have h1 : xl_10 = -6 := by sorry
  have h2 : xm_10 = -14 := by sorry
  -- hence, positive difference = | -6 + 14 | = 8
  calc
    | xl_10 - xm_10 |
      = | -6 - (-14) | : by sorry
    ... = | 14 - 6 |  : by sorry
    ... = 8          : by norm_num

end positive_difference_x_coordinates_at_y_10_l758_758500


namespace intersection_A_B_is_C_l758_758517

def A := { x : ℤ | abs x < 3 }
def B := { x : ℤ | abs x > 1 }
def C := { -2, 2 : ℤ }

theorem intersection_A_B_is_C : (A ∩ B) = C := 
  sorry

end intersection_A_B_is_C_l758_758517


namespace term_containing_x_cubed_term_with_largest_coefficient_l758_758541

theorem term_containing_x_cubed (n : ℕ) (h : (1 : ℝ)/(2 : ℝ) = (real.of_nat n)/(2 * (n.choose 3))) : 
  (x - real.sqrt 2)^4 = x ^ 4 - 4 * real.sqrt 2 * x ^ 3 + 12 * x ^ 2 - 8 * real.sqrt 2 * x + 4 → 
  C(4, 1) * x^3 * (-real.sqrt 2) = -4 * real.sqrt 2 * x^3 :=
sorry
  
theorem term_with_largest_coefficient :
  (x - real.sqrt 2)^4 = x ^ 4 - 4 * real.sqrt 2 * x ^ 3 + 12 * x ^ 2 - 8 * real.sqrt 2 * x + 4 → 
  largest_coeff = 12 :=
sorry

end term_containing_x_cubed_term_with_largest_coefficient_l758_758541


namespace triangle_side_length_l758_758611

variables {BC AC : ℝ} {α β γ : ℝ}

theorem triangle_side_length :
  α = 45 ∧ β = 75 ∧ AC = 6 ∧ α + β + γ = 180 →
  BC = 6 * (Real.sqrt 3 - 1) :=
by
  intros h
  sorry

end triangle_side_length_l758_758611


namespace impossible_divide_into_three_similar_l758_758669

noncomputable def sqrt2 : ℝ := Real.sqrt 2

def similar (x y : ℝ) : Prop :=
  x ≤ sqrt2 * y

theorem impossible_divide_into_three_similar (N : ℝ) :
  ¬ ∃ (x y z : ℝ), x + y + z = N ∧ similar x y ∧ similar y z ∧ similar x z := 
by
  sorry

end impossible_divide_into_three_similar_l758_758669


namespace count_valid_digits_l758_758046

theorem count_valid_digits :
  {n : ℕ | 1 ≤ n ∧ n ≤ 9 ∧ 15 * n % n = 0}.card = 7 :=
by sorry

end count_valid_digits_l758_758046


namespace find_sum_zero_l758_758537

open Complex

noncomputable def complex_numbers_satisfy (a1 a2 a3 : ℂ) : Prop :=
  a1^2 + a2^2 + a3^2 = 0 ∧
  a1^3 + a2^3 + a3^3 = 0 ∧
  a1^4 + a2^4 + a3^4 = 0

theorem find_sum_zero (a1 a2 a3 : ℂ) (h : complex_numbers_satisfy a1 a2 a3) :
  a1 + a2 + a3 = 0 :=
by {
  sorry
}

end find_sum_zero_l758_758537


namespace constant_from_sin_cos_even_constant_l758_758262

def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem constant_from_sin_cos_even_constant (c : ℝ)
  (h : ∃ (g : ℝ → ℝ), (∀ x, g x = c) ∧ 
       (g = f ∨ g = deriv f ∨ g = deriv (deriv f) ∨ 
        ∃ u v : ℝ → ℝ, (u = f ∨ u = deriv f ∨ u = deriv (deriv f)) ∧ 
                        (v = f ∨ v = deriv f ∨ v = deriv (deriv f)) ∧ 
                        (g = u + v ∨ g = u * v))) : ∃ n : ℤ, c = 2 * n := 
sorry

end constant_from_sin_cos_even_constant_l758_758262


namespace series_convergence_l758_758219

noncomputable def series (x : ℝ) := ∑' n, (real.sin (n * x)) / (real.exp (n * x))

theorem series_convergence (x : ℝ) : (∀ ε > 0, ∃ N, ∀ n ≥ N, |∑' k in finset.range n, (real.sin (k * x)) / (real.exp (k * x)) - series x| < ε) ↔ x ≥ 0 := 
sorry

end series_convergence_l758_758219


namespace quadratic_roots_l758_758034

theorem quadratic_roots (a b: ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0)
  (root_condition1 : a * (-1/2)^2 + b * (-1/2) + 2 = 0)
  (root_condition2 : a * (1/3)^2 + b * (1/3) + 2 = 0) 
  : a - b = -10 := 
by {
  sorry
}

end quadratic_roots_l758_758034


namespace part1_part2_l758_758975

open Set

def f (x : ℝ) : ℝ := abs (x + 2) - abs (2 * x - 1)

def M : Set ℝ := { x | f x > 0 }

theorem part1 :
  M = { x | - (1 / 3 : ℝ) < x ∧ x < 3 } :=
sorry

theorem part2 :
  ∀ (x y : ℝ), x ∈ M → y ∈ M → abs (x + y + x * y) < 15 :=
sorry

end part1_part2_l758_758975


namespace tower_height_l758_758480

theorem tower_height (h d : ℝ) 
  (tan_30_eq : Real.tan (Real.pi / 6) = h / d)
  (tan_45_eq : Real.tan (Real.pi / 4) = h / (d - 20)) :
  h = 20 * Real.sqrt 3 :=
by
  sorry

end tower_height_l758_758480


namespace _l758_758133

noncomputable def hyperbola_eccentricity_theorem {F1 F2 P : Point} 
  (hyp : is_hyperbola F1 F2 P)
  (angle_F1PF2 : ∠F1 P F2 = 60)
  (dist_PF1_3PF2 : dist P F1 = 3 * dist P F2) : 
  eccentricity F1 F2 = sqrt 7 / 2 :=
by 
  sorry

end _l758_758133


namespace rationalize_fraction_has_proper_gcd_l758_758285

def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

def is_rel_prime (a b c : ℕ) : Prop := gcd a (gcd b c) = 1

theorem rationalize_fraction_has_proper_gcd :
  ∃ P Q R S : ℤ, S > 0 ∧ (¬ ∃ p : ℕ, p > 1 ∧ p * p ∣ Q) ∧ is_rel_prime (Int.natAbs P) (Int.natAbs R) (Int.natAbs S) ∧
    ((8 : ℚ) / (3 + real.sqrt 7) = (P * real.sqrt Q + R) / S) ∧ (P + Q + R + S = 12) :=
by
  sorry

end rationalize_fraction_has_proper_gcd_l758_758285


namespace arithmetic_sequence_sum_l758_758600

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
  (h : ∀ n, a n = a 1 + (n - 1) * d) (h_6 : a 6 = 1) :
  a 2 + a 10 = 2 := 
sorry

end arithmetic_sequence_sum_l758_758600


namespace arc_length_60_10_max_area_at_α_2_l758_758092

open Real

namespace Sector

-- Assumptions
def α_deg : ℝ := 60
def R_cm : ℝ := 10
def perimeter_sector_cm : ℝ := 20

-- Problem 1: Arc length l when α = 60° and R = 10 cm
def arc_length (α : ℝ) (R : ℝ) : ℝ := (α * π / 180) * R

theorem arc_length_60_10 :
  arc_length 60 10 = (10 * π / 3) :=
sorry

-- Problem 2: Maximizing area when perimeter is 20 cm
noncomputable def sector_area (R : ℝ) : ℝ := 10 * R - R^2

theorem max_area_at_α_2 :
  ∀ R, (R > 0) → (1 * (perimeter_sector_cm - 2 * R) * R) / 2 = sector_area R ∧
  (10 - R)^2 + Max := (∃! (α : ℝ), sector_area 5 = 25 ∧ α = 2) :=
sorry

end Sector

end arc_length_60_10_max_area_at_α_2_l758_758092


namespace volume_function_max_volume_angle_l758_758389

noncomputable def R : ℝ := 3 * real.sqrt 3
noncomputable def h : ℝ
noncomputable def V (h : ℝ) : ℝ := -(1/3) * real.pi * h^3 + 9 * real.pi * h

theorem volume_function (h : ℝ) : V h = -(1/3) * real.pi * h^3 + 9 * real.pi * h := 
by
  -- Given the conditions and simplifications
  sorry

theorem max_volume_angle : ∀ α : ℝ, (∃ h : ℝ, α = (6 - 2 * real.sqrt 6) / 3 * real.pi) := 
by
  -- Given the provided conditions and maximum value derivation
  sorry

end volume_function_max_volume_angle_l758_758389


namespace least_integer_with_exactly_eight_factors_l758_758824

theorem least_integer_with_exactly_eight_factors : ∃ n : ℕ, (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 8 ∨ d = 6 ∨ d = 12 ∨ d = 24) ∧
  (∀ m : ℕ, (∀ d : ℕ, d ∣ m → d = 1 ∨ d = 2 ∨ d = 3 ∨ d
= 4 ∨ d = 8 ∨ d = 6 ∨ d = 12 ∨ d = 24) → m = n) :=
begin
  sorry
end

end least_integer_with_exactly_eight_factors_l758_758824


namespace constant_term_expansion_l758_758749

theorem constant_term_expansion (x : ℝ) (hx : x ≠ 0) : 
  let term (r : ℕ) : ℝ := (1 / 2) ^ (9 - r) * (-1) ^ r * Nat.choose 9 r * x ^ (3 / 2 * r - 9)
  term 6 = 21 / 2 :=
by
  sorry

end constant_term_expansion_l758_758749


namespace sequence_terms_before_neg_fifty_l758_758571

theorem sequence_terms_before_neg_fifty 
  (a₁ : ℤ) (d : ℤ) (n : ℕ)
  (h₁ : a₁ = 20)
  (h₂ : d = -5)
  (h₃ : 25 - 5 * n = -50) :
  n - 1 = 14 :=
by
  -- The proof is omitted here
  sorry

end sequence_terms_before_neg_fifty_l758_758571


namespace find_focus_of_parabola_4x2_l758_758015

-- Defining what it means to be the focus of a parabola given an equation y = ax^2.
def is_focus (a : ℝ) (f : ℝ) : Prop :=
  ∀ (x : ℝ), (x ^ 2 + (a * x ^ 2 - f) ^ 2) = ((a * x ^ 2 - (f * 8)) ^ 2)

-- Specific instance of the parabola y = 4x^2.
def parabola_4x2 := (4 : ℝ)

theorem find_focus_of_parabola_4x2 : ∃ f : ℝ, is_focus parabola_4x2 f :=
begin
  use (1/16 : ℝ),
  sorry -- The proof will be filled in by the theorem prover.
end

end find_focus_of_parabola_4x2_l758_758015


namespace quadratic_expression_positive_l758_758939

theorem quadratic_expression_positive (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 5) * x + k + 2 > 0) ↔ (7 - 4 * Real.sqrt 2 < k ∧ k < 7 + 4 * Real.sqrt 2) :=
by
  sorry

end quadratic_expression_positive_l758_758939


namespace total_earnings_l758_758908

def oil_change_cost : ℕ := 20
def repair_cost : ℕ := 30
def car_wash_cost : ℕ := 5

def num_oil_changes : ℕ := 5
def num_repairs : ℕ := 10
def num_car_washes : ℕ := 15

theorem total_earnings :
  (num_oil_changes * oil_change_cost) +
  (num_repairs * repair_cost) +
  (num_car_washes * car_wash_cost) = 475 :=
by
  sorry

end total_earnings_l758_758908


namespace circle_circumference_l758_758572

def ticket_length := 10.4 -- length of each ticket in cm
def overlap_length := 3.5 -- length of overlap in cm
def tickets_count := 16 -- number of tickets

def effective_length (ticket_length overlap_length : ℝ) : ℝ := ticket_length - overlap_length

def circumference (effective_length : ℝ) (tickets_count : ℕ) : ℝ :=
  tickets_count * effective_length

theorem circle_circumference :
  circumference (effective_length ticket_length overlap_length) tickets_count = 110.4 :=
by
  sorry

end circle_circumference_l758_758572


namespace find_k_l758_758483

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 3)

def vec_add_2b (k : ℝ) : ℝ × ℝ := (2 + 2 * k, 7)
def vec_sub_b (k : ℝ) : ℝ × ℝ := (4 - k, -1)

def vectors_not_parallel (k : ℝ) : Prop :=
  (vec_add_2b k).fst * (vec_sub_b k).snd ≠ (vec_add_2b k).snd * (vec_sub_b k).fst

theorem find_k (k : ℝ) (h : vectors_not_parallel k) : k ≠ 6 :=
by
  sorry

end find_k_l758_758483


namespace no_division_into_three_similar_piles_l758_758662

theorem no_division_into_three_similar_piles :
    ∀ (x : ℝ),
    ∀ (y z : ℝ),
    (x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = x) →
    (x <= sqrt 2 * y ∧ y <= sqrt 2 * z ∧ z <= sqrt 2 * x) →
    false :=
by
  intro x y z
  sorry

end no_division_into_three_similar_piles_l758_758662


namespace ratio_of_tangents_l758_758865

noncomputable def tangency_condition (p q : ℝ) : Prop :=
  p + q = 9 ∧ p < q

theorem ratio_of_tangents :
  ∃ p q : ℝ, tangency_condition p q ∧ (p / q) = 1 / 2 :=
begin
  sorry
end

end ratio_of_tangents_l758_758865


namespace semicircle_area_l758_758381

-- Define basic parameters and conditions
def length := 3
def width := 2
def rect_diagonal : ℝ := Real.sqrt (3^2 + 2^2)

-- Confirm the rectangle is inscribed in the semicircle
def diameter_of_circle := rect_diagonal
def radius_of_circle := diameter_of_circle / 2

-- Calculate the expected area of the semicircle
def area_of_semicircle := (π * (radius_of_circle ^ 2)) / 2

-- The main theorem statement
theorem semicircle_area : 
  length = 3 →
  width = 2 →
  area_of_semicircle = 3.125 * π :=
by
  -- Lean statement logic would be here
  sorry

end semicircle_area_l758_758381


namespace trent_total_blocks_travelled_l758_758334

theorem trent_total_blocks_travelled :
  ∀ (walk_blocks bus_blocks : ℕ), 
  walk_blocks = 4 → 
  bus_blocks = 7 → 
  (walk_blocks + bus_blocks) * 2 = 22 := by
  intros walk_blocks bus_blocks hw hb
  rw [hw, hb]
  norm_num
  done

end trent_total_blocks_travelled_l758_758334


namespace sum_powers_of_i_l758_758912

variable (n : ℕ) (i : ℂ) (h_multiple_of_6 : n % 6 = 0) (h_i : i^2 = -1)

theorem sum_powers_of_i (h_n6 : n = 6) :
    1 + 2*i + 3*i^2 + 4*i^3 + 5*i^4 + 6*i^5 + 7*i^6 = 6*i - 7 := by
  sorry

end sum_powers_of_i_l758_758912


namespace total_area_of_figure_l758_758425

theorem total_area_of_figure :
  let h := 7
  let w1 := 6
  let h1 := 2
  let h2 := 3
  let h3 := 1
  let w2 := 5
  let a1 := h * w1
  let a2 := (h - h1) * (11 - 7)
  let a3 := (h - h1 - h2) * (11 - 7)
  let a4 := (15 - 11) * h3
  a1 + a2 + a3 + a4 = 74 :=
by
  sorry

end total_area_of_figure_l758_758425


namespace eesha_usual_time_l758_758365

noncomputable def usual_time_to_office (T : ℝ) : Prop :=
  let delayed_time := T + 50 in
  let late_start := 30 in
  let slower_speed_ratio := 0.75 in
  T / delayed_time = slower_speed_ratio

theorem eesha_usual_time : ∃ T : ℝ, usual_time_to_office T ∧ T = 150 :=
by
  let T := 150
  have h1 : usual_time_to_office T,
  { unfold usual_time_to_office,
    simp [T, (T + 50 : ℝ) = 200],
    norm_num,
    }
  exact ⟨T, h1, rfl⟩

end eesha_usual_time_l758_758365


namespace max_value_of_expression_l758_758245

variable {x a : ℝ}

theorem max_value_of_expression (hx : 0 < x) (ha : 0 < a) :
  ∃ M, M = \frac{4}{2*(ceil^(2)+sqrt(a))} ∧ (∀ x \ y : a : ℝ, (0 < x) x > 0  (0 < y) (0 < a) | 
  (hx : 0 < x) (ha : 0 < a)) x^2 + 2 - sqrt(x^4 + a)) x) ≤ M :=
sorry

end max_value_of_expression_l758_758245


namespace collinear_points_l758_758305

theorem collinear_points 
  (A B C D K L P Q E : Type) 
  (h1 : ∃! E, ∃ (AD BC : Line), AD ∥ BC ∧ meets AD E ∧ meets BC E)
  (h2 : K ∈ internal_angle_bisector B ∧ L ∈ external_angle_bisector B)
  (h3 : P ∈ internal_angle_bisector C ∧ Q ∈ external_angle_bisector D)
  (h4 : ∀ (triangle : Triangle), 
         (K ∈ angle_bisector_line E) ∧ (L ∈ angle_bisector_line E) ∧ 
         (P ∈ angle_bisector_line E) ∧ (Q ∈ angle_bisector_line E)) :
  collinear K L P Q :=
sorry

end collinear_points_l758_758305


namespace focus_of_parabola_l758_758022

theorem focus_of_parabola (x f d : ℝ) (h : ∀ x, y = 4 * x^2 → (x = 0 ∧ y = f) → PF^2 = PQ^2 ∧ 
(PF^2 = x^2 + (4 * x^2 - f) ^ 2) := (PQ^2 = (4 * x^2 - d) ^ 2)) : 
  f = 1 / 16 := 
sorry

end focus_of_parabola_l758_758022


namespace red_marbles_in_bag_l758_758382

theorem red_marbles_in_bag (T R : ℕ) (hT : T = 84)
    (probability_not_red : ((T - R : ℚ) / T)^2 = 36 / 49) : 
    R = 12 := 
sorry

end red_marbles_in_bag_l758_758382


namespace exists_k_l758_758277

theorem exists_k (m n : ℕ) : ∃ k : ℕ, (Real.sqrt m + Real.sqrt (m - 1)) ^ n = Real.sqrt k + Real.sqrt (k - 1) := by
  sorry

end exists_k_l758_758277


namespace valid_root_l758_758291

theorem valid_root:
  ∃ x : ℚ, 
    (3 * x^2 + 5) / (x - 2) - (3 * x + 10) / 4 + (5 - 9 * x) / (x - 2) + 2 = 0 ∧ x = 2 / 3 := 
by
  sorry

end valid_root_l758_758291


namespace largest_value_l758_758437

theorem largest_value (A B C D E : ℕ)
  (hA : A = (3 + 5 + 2 + 8))
  (hB : B = (3 * 5 + 2 + 8))
  (hC : C = (3 + 5 * 2 + 8))
  (hD : D = (3 + 5 + 2 * 8))
  (hE : E = (3 * 5 * 2 * 8)) :
  max (max (max (max A B) C) D) E = E := 
sorry

end largest_value_l758_758437


namespace count_valid_n_divisibility_l758_758078

theorem count_valid_n_divisibility : 
  (finset.univ.filter (λ n : ℕ, n ∈ finset.range 10 ∧ (15 * n) % n = 0)).card = 5 :=
by sorry

end count_valid_n_divisibility_l758_758078


namespace rooms_in_second_wing_each_hall_l758_758257

theorem rooms_in_second_wing_each_hall
  (floors_first_wing : ℕ)
  (halls_per_floor_first_wing : ℕ)
  (rooms_per_hall_first_wing : ℕ)
  (floors_second_wing : ℕ)
  (halls_per_floor_second_wing : ℕ)
  (total_rooms : ℕ)
  (h1 : floors_first_wing = 9)
  (h2 : halls_per_floor_first_wing = 6)
  (h3 : rooms_per_hall_first_wing = 32)
  (h4 : floors_second_wing = 7)
  (h5 : halls_per_floor_second_wing = 9)
  (h6 : total_rooms = 4248) :
  (total_rooms - floors_first_wing * halls_per_floor_first_wing * rooms_per_hall_first_wing) / 
  (floors_second_wing * halls_per_floor_second_wing) = 40 :=
  by {
  sorry
}

end rooms_in_second_wing_each_hall_l758_758257


namespace sequence_2017th_l758_758878

noncomputable def sequence : ℕ → ℝ
| 0       := -real.sqrt 2 / 2
| 1       := -real.sqrt 3 / 4
| 2       :=  real.sqrt 4 / 8
| (n + 3) := 
  let sign := if (n + 1) % 3 = 0 ∨ (n + 1) % 3 = 2 then 1 else -1 in
  (sign * real.sqrt (n + 2)) / (2 ^ (n + 2))

theorem sequence_2017th :
  sequence 2016 = -real.sqrt 2018 / 2 ^ 2017 := sorry

end sequence_2017th_l758_758878


namespace impossible_divide_into_three_similar_parts_l758_758691

noncomputable def similar (x y : ℝ) : Prop := x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

theorem impossible_divide_into_three_similar_parts (s : ℝ → ℝ → Prop) :
  (∀ s, similar s)) → ¬ (∃ a b c : ℝ, s a b → s b c → s c a → a + b + c = 1) :=
by
  intros h_similar h
  sorry

end impossible_divide_into_three_similar_parts_l758_758691


namespace math_proof_problem_l758_758618

noncomputable def f : ℕ → ℕ 
| 0 := 0 -- Using 0 to shift the indexing
| 1 := 2
| (n+1) := f n ^ 2 - f n + 1

theorem math_proof_problem (n : ℕ) (h : n > 1) :
  1 - 1 / 2 ^ (2 ^ (n - 1)) < (∑ i in Finset.range n, 1 / f (i + 1)) ∧ 
  (∑ i in Finset.range n, 1 / f (i + 1)) < 1 - 1 / 2 ^ (2 ^ n) :=
sorry

end math_proof_problem_l758_758618


namespace monotonicity_f_g_inequality_l758_758554

-- Defining the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2 - (a + 2) * x

-- Defining the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1/2) * x^2 - x

-- Defining the conditions:
axiom a_pos : ∀ (a : ℝ), 0 < a

axiom x1_cond : ∀ (a x1 x2 : ℝ), 0 < x1 ∧ x1 < (a / 2)

axiom x2_cond : ∀ (a x1 x2 : ℝ), sqrt a < x2 ∧ x2 < 1

-- Problem 1: Monotonicity of f(x) based on the cases of a
theorem monotonicity_f (a : ℝ) (h : 0 < a) :
  (a = 2 → ∀ x, 0 < x → f a x > 0) ∧
  (a > 2 → (∀ x, (0 < x ∧ x < 1) → f a x > 0) ∧ (∀ x, (x > 1 ∧ x < a / 2) → f a x < 0) ∧ (∀ x, x > a / 2 → f a x > 0)) ∧
  (a < 2 → (∀ x, (0 < x ∧ x < a / 2) → f a x > 0) ∧ (∀ x, (x > a / 2 ∧ x < 1) → f a x < 0) ∧ (∀ x, x > 1 → f a x > 0)) := 
sorry

-- Problem 2: Prove that g(x1) - g(x2) < 0.5
theorem g_inequality (a x1 x2 : ℝ) (h1 : 0 < a) (h2 : 0 < x1 ∧ x1 < a / 2) (h3 : sqrt a < x2 ∧ x2 < 1) :
  g a x1 - g a x2 < (1 / 2) := 
sorry

end monotonicity_f_g_inequality_l758_758554


namespace inequality_flip_l758_758259

theorem inequality_flip (a b : ℤ) (c : ℤ) (h1 : a < b) (h2 : c < 0) : 
  c * a > c * b :=
sorry

end inequality_flip_l758_758259


namespace tetrahedron_is_regular_l758_758095

-- Definitions of the tetrahedron and conditions
variables {A1 A2 A3 A4 Q : Point}
variables {S1 S2 S3 S4 : Sphere}
variable {r R : ℝ}

-- Mutually tangent spheres centered at each vertex of the tetrahedron
axiom mutually_tangent (S1 S2 S3 S4 : Sphere) : 
  (tangent S1 S2) ∧ (tangent S1 S3) ∧ (tangent S1 S4) ∧ 
  (tangent S2 S3) ∧ (tangent S2 S4) ∧ (tangent S3 S4)

-- Sphere of radius r centered at Q tangent to S1, S2, S3 and S4
axiom sphere_tangent_to_vertices (Q : Point) (r : ℝ) (S1 S2 S3 S4 : Sphere) : 
  ∀ (i : ℕ), i ∈ {1, 2, 3, 4} → tangent (Sphere.mk Q r) (Si i)

-- Sphere of radius R centered at Q tangent to all the edges of the tetrahedron
axiom sphere_tangent_to_edges (Q : Point) (R : ℝ) (A1 A2 A3 A4 : Point) : 
  ∀ (i j : ℕ), {i, j} ⊆ {1, 2, 3, 4} ∧ i ≠ j → tangent (Sphere.mk Q R) (Edge.mk (A i) (A j))

-- Conclusion that should be proved
theorem tetrahedron_is_regular (A1 A2 A3 A4 Q : Point) (S1 S2 S3 S4 : Sphere) (r R : ℝ) :
  mutually_tangent S1 S2 S3 S4 →
  sphere_tangent_to_vertices Q r S1 S2 S3 S4 →
  sphere_tangent_to_edges Q R A1 A2 A3 A4 →
  regular_tetrahedron A1 A2 A3 A4 :=
by sorry

end tetrahedron_is_regular_l758_758095


namespace log_greater_than_sum_l758_758155

noncomputable def log_sum_inequality (n : ℕ) (h : 0 < n) : Prop := 
  log (n + 1) > ∑ k in Finset.range n, (1 / (2 * k.succ + 1))

theorem log_greater_than_sum (n : ℕ) (h : 0 < n) : log_sum_inequality n h :=
  sorry

end log_greater_than_sum_l758_758155


namespace proof_problem_l758_758096

noncomputable def ellipse_equation (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h : a > b) (ecc : a^2 - b^2 = (1/2)^2 * a^2) : Prop := 
  (a^2 = 4) ∧ (b^2 = 3) ∧ 
  (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) = (x, y) ∈ set_of (λ z : ℝ × ℝ, (z.1^2 / 4 + z.2^2 / 3 = 1)))

noncomputable def intersects_x_axis (P A B E Q : ℝ × ℝ) : Prop :=
  P = (4, 0) ∧ A = (A.1, -A.2) ∧ B = (B.1, B.2) ∧ Q = (1, 0) ∧ 
  B ∈ set_of (λ z : ℝ × ℝ, (z.1^2 / 4 + z.2^2 / 3 = 1)) ∧ E ∈ set_of (λ z : ℝ × ℝ, (z.1^2 / 4 + z.2^2 / 3 = 1)) ∧ 
  (∀ x : ℝ, (∃ y : ℕ, x = 2 * A.1 * E.1 - 4 * (A.1 + E.1)) / (A.1 + E.1 - 8) = x) ∧ 
  Q ∈ set_of (λ z : ℝ × ℝ, z.2 = 0)

noncomputable def range_of_values (M N : ℝ × ℝ) (Q : ℝ × ℝ) : set ℝ :=
  let O := (0, 0): ℝ × ℝ in
  let dot (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 in
  { v : ℝ | ∃ (m : ℝ), m ≥ 0 ∧ (dot O M + dot O N) = v ∧
    ((dot O M + dot O N) = -5 / 4 - 33 / (4 * (4 * m^2 + 3))) } ∪ { -5 / 4 }

theorem proof_problem (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h : a > b) (ecc : a^2 - b^2 = (1/2)^2 * a^2) 
  (P A B E Q M N : ℝ × ℝ) : 
  ellipse_equation a b a_pos b_pos h ecc ∧ intersects_x_axis P A B E Q ∧ (range_of_values M N Q = set.Icc (-4 : ℝ) (-5/4)) :=
sorry

end proof_problem_l758_758096


namespace increasing_intervals_range_of_m_l758_758160

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 2) * x + a * Real.log x

theorem increasing_intervals (a : ℝ) (h : a > 0) :
  (if a < 2 then (0, a/2) ∪ (1, +∞) else if a = 2 then (0, +∞) else (0, 1) ∪ (a/2, +∞)).increasing :=
sorry

theorem range_of_m (a : ℝ) (h : a = 4) :
  ∃ (m : ℝ), (4 * Real.log 2 - 8 < m ∧ m < -5) ∧
  ∃ x1 x2 x3 : ℝ, f 4 x1 = m ∧ f 4 x2 = m ∧ f 4 x3 = m ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 :=
sorry

end increasing_intervals_range_of_m_l758_758160


namespace apples_in_first_and_third_bags_l758_758328

def num_apples (A B C : ℕ) : Prop :=
  A + B = 11 ∧
  B + C = 18 ∧
  A + B + C = 24 →

  A + C = 19

theorem apples_in_first_and_third_bags (A B C : ℕ) (h : num_apples A B C) : A + C = 19 := by
  cases h with
  | intro h1 h2 h3 =>
    have a1 : A + C = 24 - B := by sorry
    have a2 : B = 24 - A - C := by sorry
    have a3 : B = 24 - h1 - h2 := by sorry
    show 19 = A + C from by sorry

end apples_in_first_and_third_bags_l758_758328


namespace distinct_convex_polygons_count_l758_758797

-- Twelve points on a circle
def twelve_points_on_circle := 12

-- Calculate the total number of subsets of twelve points
def total_subsets : ℕ := 2 ^ twelve_points_on_circle

-- Calculate the number of subsets with fewer than three members
def subsets_fewer_than_three : ℕ :=
  (Finset.card (Finset.powersetLen 0 (Finset.range twelve_points_on_circle)) +
   Finset.card (Finset.powersetLen 1 (Finset.range twelve_points_on_circle)) +
   Finset.card (Finset.powersetLen 2 (Finset.range twelve_points_on_circle)))

-- The number of convex polygons that can be formed using three or more points
def distinct_convex_polygons : ℕ := total_subsets - subsets_fewer_than_three

-- Lean theorem statement
theorem distinct_convex_polygons_count :
  distinct_convex_polygons = 4017 := by sorry

end distinct_convex_polygons_count_l758_758797


namespace intersection_two_sets_l758_758559

theorem intersection_two_sets (M N : Set ℤ) (h1 : M = {1, 2, 3, 4}) (h2 : N = {-2, 2}) :
  M ∩ N = {2} := 
by
  sorry

end intersection_two_sets_l758_758559


namespace initial_money_l758_758875

theorem initial_money (M : ℝ) (h_clothes : M' = (5 / 7) * M)
  (h_food : M'' = (10 / 13) * M') 
  (h_travel : M''' = (4 / 5) * M'') 
  (h_entertainment : M'''' = (8 / 11) * M''')
  (h_final : M'''' = 5400) :
  M = 16890 :=
begin
  sorry
end

end initial_money_l758_758875


namespace complex_multiplication_conjugate_l758_758999

noncomputable theory

open Complex

-- Given condition
def condition (z : ℂ) : Prop := 4 / (1 + z) = 1 - Complex.i

-- Statement to prove
theorem complex_multiplication_conjugate (z : ℂ) (h : condition z) : z * conj z = 5 := sorry

end complex_multiplication_conjugate_l758_758999


namespace exists_k_lt_ak_by_2001_fac_l758_758250

theorem exists_k_lt_ak_by_2001_fac (a : ℕ → ℝ) (H0 : a 0 = 1)
(Hn : ∀ n : ℕ, n > 0 → a n = a (⌊(7 * n / 9)⌋₊) + a (⌊(n / 9)⌋₊)) :
  ∃ k : ℕ, k > 0 ∧ a k < k / ↑(Nat.factorial 2001) := by
  sorry

end exists_k_lt_ak_by_2001_fac_l758_758250


namespace spherical_coordinates_shape_is_plane_l758_758468

def spherical_coordinates_shape (ρ θ ϕ : ℝ) : Prop := θ = c

-- Let's assume there is a constant c
variables (c : ℝ)

theorem spherical_coordinates_shape_is_plane : ∀ (ρ ϕ : ℝ), spherical_coordinates_shape ρ c ϕ → (shape_is_plane θ c) :=
by
  sorry

end spherical_coordinates_shape_is_plane_l758_758468


namespace hyperbola_eccentricity_thm_l758_758102

noncomputable def hyperbola_eccentricity 
  (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
  (angle_F1PF2 : ℝ) (dist_PF1 dist_PF2 : ℝ)
  (H1 : angle_F1PF2 = 60)
  (H2 : dist_PF1 = 3 * dist_PF2) : ℝ :=
let a := dist_PF2 in
let c := (a * sqrt 7) / 2 in 
c / a

theorem hyperbola_eccentricity_thm
  (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
  (angle_F1PF2 : ℝ) (dist_PF1 dist_PF2 : ℝ)
  (H1 : angle_F1PF2 = 60)
  (H2 : dist_PF1 = 3 * dist_PF2) :
  @hyperbola_eccentricity F1 F2 P _ _ _ angle_F1PF2 dist_PF1 dist_PF2 H1 H2 = (sqrt 7) / 2 :=
sorry

end hyperbola_eccentricity_thm_l758_758102


namespace rearrangements_count_l758_758181

-- Definitions of the letters and invalid pairs
def letters : list ℕ := [0, 1, 2, 3]   -- Corresponding to e, f, g, h respectively
def invalid_pairs : list (ℕ × ℕ) := [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]

-- Function to check if two letters are in invalid pairs
def is_invalid_pair (a b : ℕ) : Prop :=
  (a, b) ∈ invalid_pairs ∨ (b, a) ∈ invalid_pairs

-- Function to check if a list of letters is a valid arrangement
def is_valid_arrangement (arr : list ℕ) : Prop :=
  ∀ i, i < arr.length - 1 → ¬ is_invalid_pair (arr.nth_le i sorry) (arr.nth_le (i + 1) sorry)

-- Counting valid arrangements
def count_valid_arrangements : ℕ :=
  (list.permutations letters).countp is_valid_arrangement

-- Problem statement
theorem rearrangements_count : count_valid_arrangements = 4 := 
by sorry

end rearrangements_count_l758_758181


namespace total_earnings_l758_758909

def oil_change_cost : ℕ := 20
def repair_cost : ℕ := 30
def car_wash_cost : ℕ := 5

def num_oil_changes : ℕ := 5
def num_repairs : ℕ := 10
def num_car_washes : ℕ := 15

theorem total_earnings :
  (num_oil_changes * oil_change_cost) +
  (num_repairs * repair_cost) +
  (num_car_washes * car_wash_cost) = 475 :=
by
  sorry

end total_earnings_l758_758909


namespace no_three_digit_number_l758_758923

theorem no_three_digit_number :
  ¬ ∃ (a b c : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧ (100 * a + 10 * b + c = 3 * (100 * b + 10 * c + a)) :=
by
  sorry

end no_three_digit_number_l758_758923


namespace monotone_f_l758_758254

def s (n : ℕ) : ℕ :=
  -- sum of the digits of n (dummy definition, needs real implementation)
  sorry

def f : ℕ → ℕ
  | 0         => 0
  | n + 1     => f(n + 1 - s(n + 1)) + 1

theorem monotone_f (n m : ℕ) (hnm : n ≤ m) : f(n) ≤ f(m) := by
  induction' m with m ih
  -- Proofs skipped
  sorry

end monotone_f_l758_758254


namespace angle_between_vectors_is_90_degrees_l758_758579

/-- If the magnitudes of the sums and differences of two vectors are equal, then
    the angle between them is 90 degrees. -/
theorem angle_between_vectors_is_90_degrees
  {V : Type*} [inner_product_space ℝ V] (a b : V)
  (h : ∥a + b∥ = ∥a - b∥) :
  (∠ (0 : V) a b) = real.pi / 2 :=
by
  sorry

end angle_between_vectors_is_90_degrees_l758_758579


namespace count_valid_digits_l758_758043

theorem count_valid_digits :
  {n : ℕ | 1 ≤ n ∧ n ≤ 9 ∧ 15 * n % n = 0}.card = 7 :=
by sorry

end count_valid_digits_l758_758043


namespace complementary_angles_positive_difference_l758_758763

/-- Two angles are complementary if their sum is 90 degrees.
    If the measures of these angles are in the ratio 3:1,
    then their positive difference is 45 degrees. -/
theorem complementary_angles_positive_difference (x : ℝ) (h1 : (3 * x) + x = 90) :
  abs ((3 * x) - x) = 45 :=
by
  sorry

end complementary_angles_positive_difference_l758_758763


namespace product_of_100_consecutive_not_100th_power_l758_758280

theorem product_of_100_consecutive_not_100th_power (n k : ℕ) :
  ∏ i in finset.range (100), (n + i) ≠ k ^ 100 := 
sorry

end product_of_100_consecutive_not_100th_power_l758_758280


namespace count_elements_in_arithmetic_sequence_l758_758569

theorem count_elements_in_arithmetic_sequence :
  let seq := list.range' 22 250 (+7).reverse
  seq.length = 34 :=
by
  let seq := list.range' 22 250 (+7).reverse
  exact (seq.length = 34 : Prop)

end count_elements_in_arithmetic_sequence_l758_758569


namespace barbie_earrings_l758_758899

theorem barbie_earrings (total_earrings_alissa : ℕ) (alissa_triple_given : ℕ → ℕ) 
  (given_earrings_double_bought : ℕ → ℕ) (pairs_of_earrings : ℕ) : 
  total_earrings_alissa = 36 → 
  alissa_triple_given (total_earrings_alissa / 3) = total_earrings_alissa → 
  given_earrings_double_bought (total_earrings_alissa / 3) = total_earrings_alissa →
  pairs_of_earrings = 12 :=
by
  intros h1 h2 h3
  sorry

end barbie_earrings_l758_758899


namespace unique_B_cubed_l758_758626

def B : Type := Matrix (Fin 2) (Fin 2) ℝ

theorem unique_B_cubed (B : B) (h : B^4 = 0) : B^3 = 0 := by
  sorry

end unique_B_cubed_l758_758626


namespace range_of_m_l758_758628

open Set

variable {α : Type*}

theorem range_of_m (A : Set ℝ) (n m : ℝ) (x : ℝ) : 
  (B ⊆ A)  → (B = { x | -m < x ∧ x < 2 }) → 
  (f x = n * (x + 1)) → 
  m ≤ (1 / 2) := sorry

end range_of_m_l758_758628


namespace player_A_wins_on_1994x1994_chessboard_l758_758261

theorem player_A_wins_on_1994x1994_chessboard :
  ∃ strategy_for_A : (ℕ × ℕ) → List (ℕ × ℕ) → (ℕ × ℕ),
    (∀ moves : List (ℕ × ℕ), 
        let current_position := ← some moves.first
        let next_position := strategy_for_A current_position moves
        next_position.1 = current_position.1 + 1 ∨ next_position.1 = current_position.1 − 1) ∧
    (∀ moves : List (ℕ × ℕ), 
        let current_position := ← some moves.first
        ∀ next_position : (ℕ × ℕ),
        next_position = (current_position.1, current_position.2 + 1) ∨ next_position = (current_position.1, current_position.2 - 1) → 
        next_position ∉ moves) ∧
    ∀ moves : List (ℕ × ℕ), 
    ¬ (strategy_for_A (some moves.first) moves ∈ moves) ∧ 
    (∀ next_position : (ℕ × ℕ), 
    B cannot_move next_position moves → strategy_for_A (some moves.first) moves = current_position := sorry

-- placeholder for B's strategy and cannot_move definitions. Adjust accordingly.

end player_A_wins_on_1994x1994_chessboard_l758_758261


namespace ellipse_property_l758_758247

theorem ellipse_property (P : ℝ × ℝ) 
    (hP : (P.1^2 / 16) + (P.2^2 / 9) = 1) 
    (not_endpoints : P ≠ (4, 0) ∧ P ≠ (-4, 0)) :
    let F1 := (-(Real.sqrt 7), 0)
    let F2 := (Real.sqrt 7, 0)
    let O := (0, 0) in
    (Real.sqrt ((P.1 + Real.sqrt 7)^2 + P.2^2) * 
     Real.sqrt ((P.1 - Real.sqrt 7)^2 + P.2^2) + 
     (P.1^2 + P.2^2)) = 25 :=
by
    sorry

end ellipse_property_l758_758247


namespace max_abs_z_l758_758582

open Complex

theorem max_abs_z (z : ℂ) (h : abs (z + I) + abs (z - I) = 2) : abs z ≤ 1 :=
sorry

end max_abs_z_l758_758582


namespace eccentricity_of_hyperbola_l758_758120

open Real

-- Definitions of our conditions
variables {F1 F2 P : Point}
variables (a : ℝ) (m : ℝ)
variable (hyperbola_C : Hyperbola F1 F2)

-- Given conditions
axiom on_hyperbola : P ∈ hyperbola_C
axiom angle_F1P_F2 : angle F1 P F2 = π / 3
axiom distances : dist P F1 = 3 * dist P F2

-- Goal: Prove that the eccentricity of the hyperbola is sqrt(7)/2
theorem eccentricity_of_hyperbola : hyperbola.C.eccentricity = sqrt 7 / 2 := by
  sorry

end eccentricity_of_hyperbola_l758_758120


namespace log_sum_eq_neg_four_l758_758590

variable {x α β : ℝ}

def log_base_change (a b : ℝ) := Real.log b / Real.log a

theorem log_sum_eq_neg_four (h_eq : (Real.log x)^2 - 2 * (Real.log x) - 2 = 0) 
  (h_alpha_beta : α ≠ 1 ∧ β ≠ 1 ∧ α ≠ β) 
  (h_roots : (Real.log α) + (Real.log β) = 2 ∧ (Real.log α) * (Real.log β) = -2) : 
  log_base_change α β + log_base_change β α = -4 := 
by 
  sorry

end log_sum_eq_neg_four_l758_758590


namespace intersection_sets_l758_758527

open Set

def A := {x : ℤ | abs x < 3}
def B := {x : ℤ | abs x > 1}

theorem intersection_sets :
  A ∩ B = {-2, 2} := by
  sorry

end intersection_sets_l758_758527


namespace find_tangent_circle_l758_758562

-- Define circles and line
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0
def line_l (x y : ℝ) : Prop := x + 2*y = 0

-- Define the problem statement as a theorem
theorem find_tangent_circle :
  ∃ (x0 y0 : ℝ), (x - x0)^2 + (y - y0)^2 = 5/4 ∧ (x0, y0) = (1/2, 1) ∧
                   ∀ (x y : ℝ), (circle1 x y → circle2 x y → line_l (x0 + x) (y0 + y) ) :=
sorry

end find_tangent_circle_l758_758562


namespace Kara_and_Lee_walking_times_l758_758226

theorem Kara_and_Lee_walking_times (Kara_time_for_3_km Lee_time_for_4_km: ℕ) 
(Kara_ratio Lee_distance_4_km: ℕ) (Kara_distance_3_km Kara_distance_7_km Lee_distance_10_km: ℕ)
(Kara_: ℝ) (Lee_: ℝ) :
(Kara_distance_3_km = 3) →
(Kara_ratio = 1 / 3) →
(Lee_distance_4_km = 4) →
(Lee_time_for_4_km = 36) →
(Kara_ = (Lee_time_for_4_km: ℝ) * Kara_ratio) →
Kara_ * (Kara_distance_7_km / Kara_distance_3_km) = 28 ∧ (Lee_: ℝ) * Lee_distance_10_km / Lee_distance_4_km = 90 :=
by
  intros h_Kara_dist3 h_Kara_ratio h_Lee_dist_4 h_Lee_time4 h_Kara_time_for_3_km
  rw [h_Kara_dist3, h_Kara_ratio, h_Lee_dist_4, h_Lee_time4] at *
  norm_num at *
  split
  . exact calc
    Kara_ * (7 / 3) = (12) * (7 / 3) : by rw [h_Kara_time_for_3_km]
    ... = 28 : by norm_num
  . exact calc
    (36 / 4) * 10 = 9 * 10 : by norm_num
    ... = 90 : by norm_num

end Kara_and_Lee_walking_times_l758_758226


namespace divide_pile_l758_758682

theorem divide_pile (pile : ℝ) (similar : ℝ → ℝ → Prop) :
  (∀ x y, similar x y ↔ x ≤ y * Real.sqrt 2 ∧ y ≤ x * Real.sqrt 2) →
  ¬∃ a b c, a + b + c = pile ∧ similar a b ∧ similar b c ∧ similar a c :=
by sorry

end divide_pile_l758_758682


namespace smallest_n_P_lt_recip_2023_l758_758790

def P (n : ℕ) : ℚ :=
  (∏ k in finset.range (n - 1), (2 * (k + 1) : ℚ) / (2 * (k + 1) + 1)) * (1 / (2 * n + 1))

theorem smallest_n_P_lt_recip_2023 :
  ∃ n : ℕ, n = 32 ∧ P n < 1 / 2023 :=
begin
  sorry
end

end smallest_n_P_lt_recip_2023_l758_758790


namespace impossible_to_divide_into_three_similar_parts_l758_758637

-- Define similarity condition
def similar_sizes (x y : ℝ) : Prop := x ≤ √2 * y

-- Main proof problem
theorem impossible_to_divide_into_three_similar_parts (x : ℝ) :
  ¬ ∃ (a b c : ℝ), a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes c a := 
sorry

end impossible_to_divide_into_three_similar_parts_l758_758637


namespace distance_from_Martins_house_to_Lawrences_house_l758_758707

def speed : ℝ := 2 -- Martin's speed is 2 miles per hour
def time : ℝ := 6  -- Time taken is 6 hours
def distance : ℝ := speed * time -- Distance formula

theorem distance_from_Martins_house_to_Lawrences_house : distance = 12 := by
  sorry

end distance_from_Martins_house_to_Lawrences_house_l758_758707


namespace area_of_inscribed_triangle_l758_758883

theorem area_of_inscribed_triangle 
  (x : ℝ) 
  (h1 : (2:ℝ) * x ≤ (3:ℝ) * x ∧ (3:ℝ) * x ≤ (4:ℝ) * x) 
  (h2 : (4:ℝ) * x = 2 * 4) :
  ∃ (area : ℝ), area = 12.00 :=
by
  sorry

end area_of_inscribed_triangle_l758_758883


namespace arun_profit_percentage_l758_758892

-- Define the quantities and prices of wheat
def kg1 : ℝ := 30
def kg2 : ℝ := 20
def price1 : ℝ := 11.50
def price2 : ℝ := 14.25
def selling_price_per_kg : ℝ := 13.86

-- Calculate the total cost of the purchased wheat
def cost1 := kg1 * price1
def cost2 := kg2 * price2
def total_cost := cost1 + cost2

-- Calculate the total weight of the wheat mixture
def total_weight := kg1 + kg2

-- Calculate the total selling price of the mixture
def total_selling_price := total_weight * selling_price_per_kg

-- Calculate the profit
def profit := total_selling_price - total_cost

-- Calculate the profit percentage
def profit_percentage := (profit / total_cost) * 100

-- Statement to be proved
theorem arun_profit_percentage : profit_percentage = 10 := by
  sorry

end arun_profit_percentage_l758_758892


namespace trig_identity_proof_l758_758903

theorem trig_identity_proof :
  (sin (140 * real.pi / 180) * cos (50 * real.pi / 180) + sin (130 * real.pi / 180) * cos (40 * real.pi / 180)) = 1 :=
by sorry

end trig_identity_proof_l758_758903


namespace positive_difference_of_squares_l758_758786

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 8) : a^2 - b^2 = 320 :=
by
  sorry

end positive_difference_of_squares_l758_758786


namespace general_term_expression_l758_758093

noncomputable def a : ℕ → ℤ 
| 0       := 2
| (n + 1) := a n + 3

theorem general_term_expression (n : ℕ) : a n = 3 * n - 1 := sorry

end general_term_expression_l758_758093


namespace intersection_of_A_and_B_l758_758508

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} :=
by
  sorry

end intersection_of_A_and_B_l758_758508


namespace angle_amd_eq_63_point_43_degrees_l758_758735

theorem angle_amd_eq_63_point_43_degrees (AB BC : ℝ) (M : ℝ) (h₁ : AB = 8) (h₂ : BC = 4) (h₃ : M = 2) (h₄ : ∀ A C D, angle (mk A B) M D = angle M (mk C B) D) 
: ∠ (mk A B) M (mk D C) = 63.43 :=
by
  sorry

end angle_amd_eq_63_point_43_degrees_l758_758735


namespace union_of_A_and_B_intersection_of_A_and_complementB_range_of_m_l758_758992

open Set

def setA : Set ℝ := {x | -4 < x ∧ x < 2}
def setB : Set ℝ := {x | x < -5 ∨ x > 1}
def setComplementB : Set ℝ := {x | -5 ≤ x ∧ x ≤ 1}

theorem union_of_A_and_B : setA ∪ setB = {x | x < -5 ∨ x > -4} := by
  sorry

theorem intersection_of_A_and_complementB : setA ∩ setComplementB = {x | -4 < x ∧ x ≤ 1} := by
  sorry

noncomputable def setC (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m + 1}

theorem range_of_m (m : ℝ) (h : setB ∩ (setC m) = ∅) : -4 ≤ m ∧ m ≤ 0 := by
  sorry

end union_of_A_and_B_intersection_of_A_and_complementB_range_of_m_l758_758992


namespace A_inter_B_l758_758231

open Set

def A : Set ℤ := { x | -3 ≤ (2 * x - 1) ∧ (2 * x - 1) < 3 }

def B : Set ℤ := { x | ∃ k : ℤ, x = 2 * k + 1 }

theorem A_inter_B : A ∩ B = {-1, 1} :=
by
  sorry

end A_inter_B_l758_758231


namespace bottle_caps_per_group_is_175_groups_count_is_28_l758_758294

-- Definitions based on conditions
def total_bottle_caps : ℕ := 5000
def percentage_per_group : ℝ := 3.5 / 100

-- Questions translated as proof statements or requisite conditions/assertions
noncomputable def bottle_caps_per_group : ℝ := percentage_per_group * total_bottle_caps

theorem bottle_caps_per_group_is_175 :
  bottle_caps_per_group = 175 := by
  sorry

theorem groups_count_is_28 :
  (total_bottle_caps / (percentage_per_group * total_bottle_caps)).floor = 28 := by
  sorry

end bottle_caps_per_group_is_175_groups_count_is_28_l758_758294


namespace solve_inequality_l758_758634

open Set

variable {f : ℝ → ℝ}
open Function

theorem solve_inequality (h_inc : ∀ x y, 0 < x → 0 < y → x < y → f x < f y)
  (h_func_eq : ∀ x y, 0 < x → 0 < y → f (x / y) = f x - f y)
  (h_f3 : f 3 = 1)
  (x : ℝ) (hx_pos : 0 < x)
  (hx_ge : x > 5)
  (h_ineq : f x - f (1 / (x - 5)) ≥ 2) :
  x ≥ (5 + Real.sqrt 61) / 2 := sorry

end solve_inequality_l758_758634


namespace sine_omega_roots_l758_758242

theorem sine_omega_roots (ω : ℝ) (hω : ω > 0) :
  (∃! x ∈ set.Icc 0 (2 * real.pi), abs (real.sin (ω * x - real.pi / 4)) = 1) ↔ (7/8 ≤ ω ∧ ω < 11/8) :=
sorry

end sine_omega_roots_l758_758242


namespace problem_statement_l758_758252

def proper_divisors_product (n : ℕ) : ℕ :=
  (Finset.filter (λ d, d ≠ n) (Finset.divisors n)).prod id

def count_n_not_dividing_g (upper_limit : ℕ) : ℕ :=
  let count_violations := (Finset.range upper_limit).filter (λ n, 
    let g_n := proper_divisors_product n
    in n ≥ 2 ∧ n ∣ g_n -> False)
  in count_violations.card

theorem problem_statement : count_n_not_dividing_g 101 = 29 :=
by sorry

end problem_statement_l758_758252


namespace selection_of_projects_l758_758868

-- Mathematical definitions
def numberOfWaysToSelect2ProjectsFrom4KeyAnd6General (key: Finset ℕ) (general: Finset ℕ) : ℕ :=
  (key.card.choose 2) * (general.card.choose 2)

def numberOfWaysToSelectAtLeastOneProjectAorB (key: Finset ℕ) (general: Finset ℕ) (A B: ℕ) : ℕ :=
  let total_ways := (key.card.choose 2) * (general.card.choose 2)
  let ways_without_A := ((key.erase A).card.choose 2) * (general.card.choose 2)
  let ways_without_B := (key.card.choose 2) * ((general.erase B).card.choose 2)
  let ways_without_A_and_B := ((key.erase A).card.choose 2) * ((general.erase B).card.choose 2)
  total_ways - ways_without_A_and_B

-- Theorem we need to prove
theorem selection_of_projects (key general: Finset ℕ) (A B: ℕ) (hA: A ∈ key) (hB: B ∈ general) (h_key_card: key.card = 4) (h_general_card: general.card = 6) :
  numberOfWaysToSelectAtLeastOneProjectAorB key general A B = 60 := 
sorry

end selection_of_projects_l758_758868


namespace part_I_part_II_l758_758161

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2 * x - 4) * Real.exp x + a * (x + 2) ^ 2

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := (2 * x - 2) * Real.exp x + 2 * a * (x + 2)

-- Theorem for Part (I)
theorem part_I (a : ℝ) : (∀ x > 0, f' x a ≥ 0) → a ≥ 1 / 2 :=
by
  -- Add appropriate conditions and proof steps
  sorry

-- Theorem for Part (II)
theorem part_II {a : ℝ} (h₀ : 0 < a) (h₁ : a < 1 / 2) : 
  ∃ x, (∀ y, f y a ≥ f x a) ∧ can_min_value_range : (∃ x > 0, x < 1 ∧ f x a ∈ (-2 * Real.exp 1, -2)) :=
by
  -- Add appropriate conditions and proof steps
  sorry

end part_I_part_II_l758_758161


namespace possible_remainders_of_a2_l758_758622

theorem possible_remainders_of_a2 (p : ℕ) (k : ℕ) (hp : Nat.Prime p) (hk : 0 < k) 
  (hresidue : ∀ i : ℕ, i < p → ∃ j : ℕ, j < p ∧ ((j^k+j) % p = i)) :
  ∃ s : Finset ℕ, s = Finset.range p ∧ (2^k + 2) % p ∈ s := 
sorry

end possible_remainders_of_a2_l758_758622


namespace incenter_circumcenter_midpoints_concyclic_l758_758217

open Real EuclideanGeometry

variable {A B C : Point}

theorem incenter_circumcenter_midpoints_concyclic
  (h1 : 2 * dist A B = dist B C + dist C A) : 
  ∃ (O I D E : Point),
    is_circumcenter O A B C ∧ 
    is_incenter I A B C ∧ 
    midpoint B C D ∧ 
    midpoint A C E ∧ 
    concyclic O I D E :=
sorry

end incenter_circumcenter_midpoints_concyclic_l758_758217


namespace num_divisible_digits_l758_758049

def divisible_by_n (num : ℕ) (n : ℕ) : Prop :=
  n ≠ 0 ∧ num % n = 0

def count_divisible_digits : ℕ :=
  (List.filter (λ n => divisible_by_n (150 + n) n)
    [1, 2, 3, 4, 5, 6, 7, 8, 9]).length

theorem num_divisible_digits : count_divisible_digits = 7 := by
  sorry

end num_divisible_digits_l758_758049


namespace cos_sub_eq_five_over_eight_l758_758138

theorem cos_sub_eq_five_over_eight (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5 / 8 := 
by sorry

end cos_sub_eq_five_over_eight_l758_758138


namespace no_division_into_three_similar_piles_l758_758665

theorem no_division_into_three_similar_piles :
    ∀ (x : ℝ),
    ∀ (y z : ℝ),
    (x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = x) →
    (x <= sqrt 2 * y ∧ y <= sqrt 2 * z ∧ z <= sqrt 2 * x) →
    false :=
by
  intro x y z
  sorry

end no_division_into_three_similar_piles_l758_758665


namespace polygon_edges_l758_758315

theorem polygon_edges :
  ∃ a b : ℕ, a + b = 2014 ∧
              (a * (a - 3) / 2 + b * (b - 3) / 2 = 1014053) ∧
              a ≤ b ∧
              a = 952 :=
by
  sorry

end polygon_edges_l758_758315


namespace eccentricity_of_hyperbola_l758_758121

open Real

-- Definitions of our conditions
variables {F1 F2 P : Point}
variables (a : ℝ) (m : ℝ)
variable (hyperbola_C : Hyperbola F1 F2)

-- Given conditions
axiom on_hyperbola : P ∈ hyperbola_C
axiom angle_F1P_F2 : angle F1 P F2 = π / 3
axiom distances : dist P F1 = 3 * dist P F2

-- Goal: Prove that the eccentricity of the hyperbola is sqrt(7)/2
theorem eccentricity_of_hyperbola : hyperbola.C.eccentricity = sqrt 7 / 2 := by
  sorry

end eccentricity_of_hyperbola_l758_758121


namespace impossible_divide_into_three_similar_l758_758674

noncomputable def sqrt2 : ℝ := Real.sqrt 2

def similar (x y : ℝ) : Prop :=
  x ≤ sqrt2 * y

theorem impossible_divide_into_three_similar (N : ℝ) :
  ¬ ∃ (x y z : ℝ), x + y + z = N ∧ similar x y ∧ similar y z ∧ similar x z := 
by
  sorry

end impossible_divide_into_three_similar_l758_758674


namespace remainder_x3_minus_4x2_plus_3x_plus_2_div_x_minus_1_l758_758947

def p (x : ℝ) : ℝ := x^3 - 4 * x^2 + 3 * x + 2

theorem remainder_x3_minus_4x2_plus_3x_plus_2_div_x_minus_1 :
  p 1 = 2 := by
  -- solution needed, for now we put a placeholder
  sorry

end remainder_x3_minus_4x2_plus_3x_plus_2_div_x_minus_1_l758_758947


namespace least_positive_integer_with_eight_factors_l758_758820

noncomputable def numDivisors (n : ℕ) : ℕ :=
  (List.range (n+1)).count (λ d => d > 0 ∧ n % d = 0)

theorem least_positive_integer_with_eight_factors : ∃ n : ℕ, n > 0 ∧ numDivisors n = 8 ∧ (∀ m : ℕ, m > 0 → numDivisors m = 8 → n ≤ m) := 
  sorry

end least_positive_integer_with_eight_factors_l758_758820


namespace num_digits_divisible_l758_758061

theorem num_digits_divisible (h : ∀ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}, divides n (150 + n)) :
  {n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} | divides n (150 + n)}.card = 5 := 
sorry

end num_digits_divisible_l758_758061


namespace impossibility_of_dividing_into_three_similar_piles_l758_758659

theorem impossibility_of_dividing_into_three_similar_piles:
  ∀ (x y z : ℝ), ¬ (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x + y + z = 1  ∧ (x ≤ sqrt 2 * y ∧ y ≤ sqrt 2 * x) ∧ (y ≤ sqrt 2 * z ∧ z ≤ sqrt 2 * y) ∧ (z ≤ sqrt 2 * x ∧ x ≤ sqrt 2 * z)) :=
by
  sorry

end impossibility_of_dividing_into_three_similar_piles_l758_758659


namespace entered_summer_l758_758602

-- Define the conditions for each location
def median_location_A : ℝ := 24
def mode_location_A : ℝ := 22

def median_location_B : ℝ := 25
def mean_location_B : ℝ := 24

def mean_location_C : ℝ := 22
def mode_location_C : ℝ := 22

def temp_28_location_D : ℝ := 28
def mean_location_D : ℝ := 24
def variance_location_D : ℝ := 4.8

-- Prove that location A and location D have entered the summer region
theorem entered_summer (A B C D : fin 5 → ℝ) :
  median A = median_location_A ∧ mode A = mode_location_A →
  median B = median_location_B ∧ mean B = mean_location_B →
  mean C = mean_location_C ∧ mode C = mode_location_C →
  A 3 = temp_28_location_D ∧ mean D = mean_location_D ∧ variance D = variance_location_D →
  (all (λ t, t ≥ 22) (univ.map A)) ∧ (all (λ t, t ≥ 22) (univ.map D)) := sorry

end entered_summer_l758_758602


namespace range_of_a_to_have_no_zeros_l758_758546

noncomputable def f (x : ℝ) : ℝ := Real.sin (π * x - π / 3)

def has_no_zeros (f : ℝ → ℝ) := ∀ x : ℝ, f x ≠ 0

theorem range_of_a_to_have_no_zeros (a : ℝ) :
  has_no_zeros (fun x => f (a * Real.sin x + 1)) ↔ a ∈ set.Ioo (-1/3 : ℝ) (1/3 : ℝ) :=
by
  sorry

end range_of_a_to_have_no_zeros_l758_758546


namespace return_time_possibilities_l758_758876

variables (d v w : ℝ) (t_return : ℝ)

-- Condition 1: Flight against wind takes 84 minutes
axiom flight_against_wind : d / (v - w) = 84

-- Condition 2: Return trip with wind takes 9 minutes less than without wind
axiom return_wind_condition : d / (v + w) = d / v - 9

-- Problem Statement: Find the possible return times
theorem return_time_possibilities :
  t_return = d / (v + w) → t_return = 63 ∨ t_return = 12 :=
sorry

end return_time_possibilities_l758_758876


namespace ten_faucets_fill_50_gallons_in_60_seconds_l758_758037

-- Define the rate of water dispensed by each faucet and time calculation
def fill_tub_time (num_faucets tub_volume faucet_rate : ℝ) : ℝ :=
  tub_volume / (num_faucets * faucet_rate)

-- Given conditions
def five_faucets_fill_200_gallons_in_8_minutes : Prop :=
  ∀ faucet_rate : ℝ, 5 * faucet_rate * 8 = 200

-- Main theorem: Ten faucets fill a 50-gallon tub in 60 seconds
theorem ten_faucets_fill_50_gallons_in_60_seconds 
  (faucet_rate : ℝ) 
  (h : five_faucets_fill_200_gallons_in_8_minutes) : 
  fill_tub_time 10 50 faucet_rate = 1 := by
  sorry

end ten_faucets_fill_50_gallons_in_60_seconds_l758_758037


namespace perpendicular_vectors_acute_angle_l758_758503

open Real

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, -2)

-- Part 1: Proving the value of k for perpendicular vectors
theorem perpendicular_vectors (k : ℝ) : 
  let v1 := (k * a.1 + b.1, k * a.2 + b.2)
  let v2 := (a.1 - 3 * b.1, a.2 - 3 * b.2)
  (v1.1 * v2.1 + v1.2 * v2.2 = 0) ↔ k = 23 / 13 := sorry

-- Part 2: Finding the range for λ
theorem acute_angle (λ : ℝ) :
  let v := (a.1 + λ * b.1, a.2 + λ * b.2)
  ((a.1 * v.1 + a.2 * v.2 > 0) ↔ (λ < 5 / 7 ∧ λ ≠ 0)) := sorry

end perpendicular_vectors_acute_angle_l758_758503


namespace num_ways_at_least_2_past_l758_758898

noncomputable def num_ways_with_at_least_2_past_officers : ℕ :=
  let total_ways := Nat.choose 20 4
  let no_past_officers := Nat.choose 10 0 * Nat.choose 10 4
  let one_past_officer := Nat.choose 10 1 * Nat.choose 10 3
  let fewer_than_2_past_officers := no_past_officers + one_past_officer
  total_ways - fewer_than_2_past_officers

theorem num_ways_at_least_2_past (total_candidates past_officers : ℕ) (chosen_officers : ℕ) (H1: total_candidates = 20) (H2: past_officers = 10) (H3: chosen_officers = 4) :
  num_ways_with_at_least_2_past_officers = 3435 :=
by
  rw [← H1, ← H2, ← H3]
  unfold num_ways_with_at_least_2_past_officers
  sorry

end num_ways_at_least_2_past_l758_758898


namespace least_n_multiple_of_35_l758_758633

noncomputable def sequence_b : ℕ → ℕ
| 7     := 7
| (n+1) := if n < 6 then 7 else 50 * sequence_b n + (n + 1)

theorem least_n_multiple_of_35 : ∃ n, n > 7 ∧ sequence_b n % 35 = 0 ∧ ∀ m, m > 7 ∧ m < n → sequence_b m % 35 ≠ 0 :=
by
  have b7 := sequence_b 7
  have b8 := sequence_b 8
  -- more intermediate steps are avoided as we are skipping the proof
  -- based on the solution provided
  use 21
  split
  · exact by norm_num
  split
  · exact by sorry
  · intros m hm1 hm2
    exact by sorry

end least_n_multiple_of_35_l758_758633


namespace count_zero_product_factors_l758_758474

noncomputable def prod_expression (n : ℕ) : ℂ :=
∏ k in finset.range n, ((1 + complex.exp (2 * real.pi * complex.I * (k : ℂ) / n)) ^ (2 * n) + 1)

theorem count_zero_product_factors : 
  let count := ∑ n in finset.range 3001, if prod_expression n = 0 then 1 else 0
  in count = 500 :=
begin
  sorry
end

end count_zero_product_factors_l758_758474


namespace smallest_number_with_exactly_eight_factors_l758_758826

theorem smallest_number_with_exactly_eight_factors
    (n : ℕ)
    (h1 : ∃ a b : ℕ, (a + 1) * (b + 1) = 8)
    (h2 : ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ n = p^a * q^b) : 
    n = 24 := by
  sorry

end smallest_number_with_exactly_eight_factors_l758_758826


namespace logarithm_decreasing_l758_758149

noncomputable def logarithm_base (a x : ℝ) := real.log (6 - a * x) / real.log a

theorem logarithm_decreasing (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Ioo (-3 : ℝ) (2 : ℝ), f x = real.log (6 - a * x) / real.log a) →
  ∀ x y ∈ Ioo (-3 : ℝ) (2 : ℝ), x < y → f y < f x ↔ 1 < a ∧ a < 3 :=
by
  sorry

end logarithm_decreasing_l758_758149


namespace equal_k_p_l758_758926

variable (k p : ℕ)

def x_i (i : fin k) : ℤ := 2 * (p : ℤ) + 3
def y_j (j : fin p) : ℤ := 5 - 2 * (k : ℤ)

theorem equal_k_p (h: ((k : ℤ) * (2 * (p : ℤ) + 3) + (p : ℤ) * (5 - 2 * (k : ℤ))) / (k + p) = 4) : k = p :=
by
  sorry

end equal_k_p_l758_758926


namespace incorrect_statement_about_parabola_l758_758962

def optionC_incorrect : Prop :=
  ∀ x : ℝ, x > -2 → (x + 2)^2 - 1 > 0

theorem incorrect_statement_about_parabola :
  ∃ x : ℝ, x > -2 ∧ ¬(optionC_incorrect) :=
begin
  use 0,
  split,
  { linarith, }, -- this proves 0 > -2
  { intros h_optionC,
    apply h_optionC,
    linarith, -- this proves the contradiction
  }
end

end incorrect_statement_about_parabola_l758_758962


namespace geometric_mean_2_6_l758_758301

theorem geometric_mean_2_6 : ∃ x : ℝ, x^2 = 2 * 6 ∧ (x = 2 * Real.sqrt 3 ∨ x = - (2 * Real.sqrt 3)) :=
by
  sorry

end geometric_mean_2_6_l758_758301


namespace problem_equivalent_l758_758246

noncomputable def h (y : ℝ) : ℝ := y^5 - y^3 + 2
noncomputable def k (y : ℝ) : ℝ := y^2 - 3

theorem problem_equivalent (y₁ y₂ y₃ y₄ y₅ : ℝ) (h_roots : ∀ y, h y = 0 ↔ y = y₁ ∨ y = y₂ ∨ y = y₃ ∨ y = y₄ ∨ y = y₅) :
  (k y₁) * (k y₂) * (k y₃) * (k y₄) * (k y₅) = 104 :=
sorry

end problem_equivalent_l758_758246


namespace evaluate_expression_l758_758904

theorem evaluate_expression :
  2 - (-3) - 4 + (-5) - 6 + 7 = -3 :=
by
  sorry

end evaluate_expression_l758_758904


namespace symmetric_line_eq_x_axis_l758_758754

theorem symmetric_line_eq_x_axis (x y : ℝ) :
  (3 * x - 4 * y + 5 = 0) → (3 * x + 4 * (-y) + 5 = 0) :=
by
  sorry

end symmetric_line_eq_x_axis_l758_758754


namespace one_head_two_tails_probability_l758_758839

noncomputable def probability_of_one_head_two_tails :=
  let total_outcomes := 8
  let favorable_outcomes := 3
  favorable_outcomes / total_outcomes

theorem one_head_two_tails_probability :
  probability_of_one_head_two_tails = 3 / 8 :=
by
  -- Proof would go here
  sorry

end one_head_two_tails_probability_l758_758839


namespace intersection_A_B_is_C_l758_758516

def A := { x : ℤ | abs x < 3 }
def B := { x : ℤ | abs x > 1 }
def C := { -2, 2 : ℤ }

theorem intersection_A_B_is_C : (A ∩ B) = C := 
  sorry

end intersection_A_B_is_C_l758_758516


namespace percentage_of_hours_on_other_services_l758_758869

def total_hours : ℝ := 68.33333333333333
def software_hours : ℝ := 24
def help_user_hours : ℝ := 17
def other_services_hours : ℝ := total_hours - (software_hours + help_user_hours)
def percentage_other_services : ℝ := (other_services_hours / total_hours) * 100

theorem percentage_of_hours_on_other_services :
  percentage_other_services = 40 :=
begin
  -- Proof would go here
  sorry
end

end percentage_of_hours_on_other_services_l758_758869


namespace largest_possible_k_l758_758318

-- Definitions pertaining to prime factors condition
def has_at_most_two_prime_factors (n : ℕ) : Prop :=
  (nat.factors n).num_unique ≤ 2

-- The main problem statement
theorem largest_possible_k : ∃ (k : ℕ), k = 44 ∧
  (∀ l : ℕ, l > k → ∃ (A B : set ℕ), 
    (A ∪ B = finset.range (2 * l + 1).to_set) ∧
    (A ∩ B = ∅) ∧ 
    (A.card = B.card) ∧ 
    (∀ a b ∈ A, has_at_most_two_prime_factors a ∧ has_at_most_two_prime_factors b) ∧ 
    (∀ a b ∈ B, has_at_most_two_prime_factors a ∧ has_at_most_two_prime_factors b)) ∧
  (∀ (A B : set ℕ), 
    (A ∪ B = finset.range (2 * 44 + 1).to_set) ∧ 
    (A ∩ B = ∅) ∧ 
    (A.card = B.card) ∧ 
    (∀ a b ∈ A, has_at_most_two_prime_factors a ∧ has_at_most_two_prime_factors b) ∧ 
    (∀ a b ∈ B, has_at_most_two_prime_factors a ∧ has_at_most_two_prime_factors b) →
    false) :=
by sorry

end largest_possible_k_l758_758318


namespace hyperbola_eccentricity_thm_l758_758103

noncomputable def hyperbola_eccentricity 
  (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
  (angle_F1PF2 : ℝ) (dist_PF1 dist_PF2 : ℝ)
  (H1 : angle_F1PF2 = 60)
  (H2 : dist_PF1 = 3 * dist_PF2) : ℝ :=
let a := dist_PF2 in
let c := (a * sqrt 7) / 2 in 
c / a

theorem hyperbola_eccentricity_thm
  (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
  (angle_F1PF2 : ℝ) (dist_PF1 dist_PF2 : ℝ)
  (H1 : angle_F1PF2 = 60)
  (H2 : dist_PF1 = 3 * dist_PF2) :
  @hyperbola_eccentricity F1 F2 P _ _ _ angle_F1PF2 dist_PF1 dist_PF2 H1 H2 = (sqrt 7) / 2 :=
sorry

end hyperbola_eccentricity_thm_l758_758103


namespace problem_solution_l758_758097

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem problem_solution (f : ℝ → ℝ)
  (H1 : even_function f)
  (H2 : ∀ x, f (x + 4) = -f x)
  (H3 : ∀ ⦃x y⦄, 0 ≤ x → x < y → y ≤ 4 → f y < f x) :
  f 13 < f 10 ∧ f 10 < f 15 :=
  by
    sorry

end problem_solution_l758_758097


namespace number_of_toddlers_l758_758439

-- Definitions based on the conditions provided in the problem
def total_children := 40
def newborns := 4
def toddlers (T : ℕ) := T
def teenagers (T : ℕ) := 5 * T

-- The theorem to prove
theorem number_of_toddlers : ∃ T : ℕ, newborns + toddlers T + teenagers T = total_children ∧ T = 6 :=
by
  sorry

end number_of_toddlers_l758_758439


namespace problem_solution_l758_758777

theorem problem_solution (x : ℝ) (hx : x + 1/x = Real.sqrt 5) : x^11 - 7 * x^7 + x^3 = 0 := 
sorry

end problem_solution_l758_758777


namespace perfect_square_trinomial_t_l758_758153

theorem perfect_square_trinomial_t (a b t : ℝ) :
  (∃ (x y : ℝ), x = a ∧ y = 2 * b ∧ a^2 + (2 * t - 1) * a * b + 4 * b^2 = (x + y)^2) →
  (t = 5 / 2 ∨ t = -3 / 2) :=
by
  sorry

end perfect_square_trinomial_t_l758_758153


namespace impossible_to_divide_into_three_similar_piles_l758_758650

def similar (a b : ℝ) : Prop :=
  a / b ≤ real.sqrt 2 ∧ b / a ≤ real.sqrt 2

theorem impossible_to_divide_into_three_similar_piles (pile : ℝ) (h : 0 < pile) :
  ¬ ∃ (x y z : ℝ), 
    x + y + z = pile ∧
    similar x y ∧ similar y z ∧ similar z x :=
by
  sorry

end impossible_to_divide_into_three_similar_piles_l758_758650


namespace fraction_more_than_one_child_l758_758385

-- Define constants for fractions given in the problem.
def fraction_more_than_3_children : ℝ := 2 / 5
def fraction_2_or_3_children : ℝ := 0.2

-- Theorem statement
theorem fraction_more_than_one_child :
  fraction_more_than_3_children + fraction_2_or_3_children = 3 / 5 :=
by
  -- Proof omitted, stated with sorry
  sorry

end fraction_more_than_one_child_l758_758385


namespace loop_stops_after_20_iterations_l758_758757

theorem loop_stops_after_20_iterations :
  ∀ (S i : ℕ), i = 20 →
  (∀ x : ℕ, S = S + x → i = i - 1) →
  (∑ x in range 20, S + x) / 20 =
  20 →
  i = 0 := by sorry

end loop_stops_after_20_iterations_l758_758757


namespace quadratic_two_distinct_real_roots_find_k_l758_758165

-- Part 1: Prove the quadratic equation always has two distinct real roots
theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  ∀ k : ℝ, let a := 1, b := 2 * k + 1, c := k^2 + k in
  let Δ := b^2 - 4 * a * c in Δ > 0 :=
by
  intros
  let Δ := (2 * k + 1)^2 - 4 * 1 * (k^2 + k)
  show Δ = 1
  sorry

-- Part 2: Find the value of k if the roots satisfy the given condition
theorem find_k (k x1 x2 : ℝ) : 
  ∀ k : ℝ, (x1 + x2 = x1 * x2 - 1) ∧ x1 + x2 = -(2 * k + 1) ∧ x1 * x2 = k^2 + k -> k = 0 ∨ k = -3 :=
by
  intros k h
  have h_sum := h.2.1
  have h_product := h.2.2
  show x1 + x2 = -(2 * k + 1) and x1 * x2 = k^2 + k = k = 0 ∨ k = -3
  sorry

end quadratic_two_distinct_real_roots_find_k_l758_758165


namespace max_path_length_correct_l758_758393

noncomputable def maxFlyPathLength : ℝ :=
  2 * Real.sqrt 2 + Real.sqrt 6 + 6

theorem max_path_length_correct :
  ∀ (fly_path_length : ℝ), (fly_path_length = maxFlyPathLength) :=
by
  intro fly_path_length
  sorry

end max_path_length_correct_l758_758393


namespace find_range_m_l758_758505

variables (m : ℝ)

def p (m : ℝ) : Prop :=
  (∀ x y : ℝ, (x^2 / (2 * m)) - (y^2 / (m - 1)) = 1) → false

def q (m : ℝ) : Prop :=
  (∀ e : ℝ, (1 < e ∧ e < 2) → (∀ x y : ℝ, (y^2 / 5) - (x^2 / m) = 1)) → false

noncomputable def range_m (m : ℝ) : Prop :=
  p m = false ∧ q m = false ∧ (p m ∨ q m) = true → (1/3 ≤ m ∧ m < 15)

theorem find_range_m : ∀ m : ℝ, range_m m :=
by
  intro m
  simp [range_m, p, q]
  sorry

end find_range_m_l758_758505


namespace z_in_first_quadrant_l758_758586

-- The imaginary unit
def i : ℂ := complex.I

-- Given condition: z = (-2 + 3 * i) / i
def z : ℂ := (-2 + 3 * i) / i

-- The theorem to prove that z is in the first quadrant
theorem z_in_first_quadrant : ∃ (x y : ℝ), z = complex.of_real x + complex.I * y ∧ 0 < x ∧ 0 < y :=
by
  sorry

end z_in_first_quadrant_l758_758586


namespace linear_function_increasing_implies_positive_slope_l758_758583

-- Define the linear function
def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ (x : ℝ), f(x) = m * x + b

-- Define the increasing function property
def is_increasing_on (f : ℝ → ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f(x₁) < f(x₂)

-- The proof problem
theorem linear_function_increasing_implies_positive_slope (f : ℝ → ℝ) :
  (is_linear f ∧ is_increasing_on f) → (∃ (m : ℝ), ∀ (x : ℝ), f(x) = m * x + b ∧ m > 0) :=
sorry

end linear_function_increasing_implies_positive_slope_l758_758583


namespace basketball_team_committee_count_l758_758383

theorem basketball_team_committee_count :
  ∃ (n k : ℕ), n = 12 ∧ k = 3 ∧ Nat.choose n k = 220 :=
by
  use 12, 3
  split
  case left => rfl
  case right => 
    split
    case left => rfl
    case right => sorry

end basketball_team_committee_count_l758_758383


namespace determine_q_l758_758615

theorem determine_q :
  (∀ x, 0 ≤ x → 0 ≤ arctan x ∧ arctan x < π / 2) →
  (∫ x in 1..2, x⁻¹ * arctan (1 + x) = q * π * log 2) →
  q = 3 / 8 :=
by
  intros h_arctan h_integral
  sorry

end determine_q_l758_758615


namespace main_problem_l758_758157

noncomputable def sin_func (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem main_problem 
  (ω : ℝ) 
  (φ : ℝ) 
  (hω : ω > 0) 
  (hφ : φ ∈ Set.Ioo (-Real.pi) Real.pi) 
  (zero1 : sin_func ω φ (π / 3) = 0) 
  (zero2 : sin_func ω φ (5 * π / 6) = 0) : 
  (∃ k : ℤ, k ∈ (7 * π / 12 + (k : ℝ) * (π / 2))) ∧ 
  (∃ k : ℤ, φ = k * π - (2 * π / 3)) :=
sorry

end main_problem_l758_758157


namespace smallest_constant_M_l758_758950

theorem smallest_constant_M :
  ∃ M, M = 2 ∧ ∀ (x y z w : ℝ), 0 < x → 0 < y → 0 < z → 0 < w →
    sqrt (x / (y + z + w)) +
    sqrt (y / (x + z + w)) +
    sqrt (z / (x + y + w)) +
    sqrt (w / (x + y + z)) < M := 
begin
  use 2,
  split,
  { refl },
  { intros x y z w hx hy hz hw,
    sorry
  }
end

end smallest_constant_M_l758_758950


namespace integral_rational_eq_log_minus_frac_l758_758850

noncomputable def integral_rational_function (x : ℝ) : ℝ := ∫ (t : ℝ) in 0..x, (t^3 + 6 * t^2 + 14 * t + 4) / ((t - 2) * (t + 2)^3)

theorem integral_rational_eq_log_minus_frac (x : ℝ) (C : ℝ) :
  integral_rational_function x = ln |x - 2| - 1 / (x + 2)^2 + C :=
sorry

end integral_rational_eq_log_minus_frac_l758_758850


namespace find_number_l758_758856

theorem find_number (x : ℝ) : (0.75 * x = 0.45 * 1500 + 495) -> x = 1560 :=
by
  sorry

end find_number_l758_758856


namespace acute_angle_sum_l758_758147

theorem acute_angle_sum (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h1 : Real.sin α = (2 * Real.sqrt 5) / 5) (h2 : Real.sin β = (3 * Real.sqrt 10) / 10) :
    α + β = 3 * Real.pi / 4 :=
sorry

end acute_angle_sum_l758_758147


namespace cos_gamma_value_l758_758232

noncomputable def cos_gamma (cos_alpha cos_beta : ℝ) : ℝ :=
  let cos_alpha_sq := cos_alpha ^ 2
  let cos_beta_sq := cos_beta ^ 2
  let cos_gamma_sq := 1 - cos_alpha_sq - cos_beta_sq
  real.sqrt cos_gamma_sq

theorem cos_gamma_value :
  cos_gamma (2 / 5) (1 / 4) = real.sqrt 311 / 20 :=
by
  sorry

end cos_gamma_value_l758_758232


namespace tenth_term_is_55_l758_758744

-- Define the sequence formula
def sequence (n : ℕ) : ℕ := n * (n + 1) / 2

-- The theorem to prove that the 10th term is 55
theorem tenth_term_is_55 : sequence 10 = 55 :=
by 
sorry

end tenth_term_is_55_l758_758744


namespace line_perpendicular_in_plane_l758_758471

/-- For any line a and plane α, there exists a line b within plane α such that b is perpendicular to a. -/
theorem line_perpendicular_in_plane (a : Line) (α : Plane) : ∃ b : Line, b ∈ α ∧ is_perpendicular a b :=
by sorry

end line_perpendicular_in_plane_l758_758471


namespace complementary_angles_positive_difference_l758_758764

/-- Two angles are complementary if their sum is 90 degrees.
    If the measures of these angles are in the ratio 3:1,
    then their positive difference is 45 degrees. -/
theorem complementary_angles_positive_difference (x : ℝ) (h1 : (3 * x) + x = 90) :
  abs ((3 * x) - x) = 45 :=
by
  sorry

end complementary_angles_positive_difference_l758_758764


namespace num_digits_divisible_l758_758065

theorem num_digits_divisible (h : ∀ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}, divides n (150 + n)) :
  {n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} | divides n (150 + n)}.card = 5 := 
sorry

end num_digits_divisible_l758_758065


namespace masha_guessed_number_l758_758286

theorem masha_guessed_number (a b : ℕ) (h1 : a + b = 2002 ∨ a * b = 2002)
  (h2 : ∀ x y, x + y = 2002 → x ≠ 1001 → y ≠ 1001)
  (h3 : ∀ x y, x * y = 2002 → x ≠ 1001 → y ≠ 1001) :
  b = 1001 :=
by {
  sorry
}

end masha_guessed_number_l758_758286


namespace f_second_derivative_at_zero_l758_758081

noncomputable def f (x : ℝ) (n : ℕ) := 
  (finset.range (n + 1)).sum (λ k, (1 + x)^k)

theorem f_second_derivative_at_zero (n : ℕ) : 
  (deriv^[2] (λ x, f x n)) 0 = n * (n - 1) / 2 :=
by
  sorry

end f_second_derivative_at_zero_l758_758081


namespace least_positive_integer_with_eight_factors_l758_758815

theorem least_positive_integer_with_eight_factors :
  ∃ n : ℕ, n = 24 ∧ (8 = (nat.factors n).length) :=
sorry

end least_positive_integer_with_eight_factors_l758_758815


namespace pow_neg_one_sum_l758_758317

theorem pow_neg_one_sum : (-1)^3 + (-1)^2 + (-1) = -1 := by
  -- Definitions of each term, derived from conditions
  have h1 : (-1)^3 = -1 := sorry
  have h2 : (-1)^2 = 1 := sorry
  have h3 : (-1) = -1 := sorry
  -- Using the derived values to prove the final sum
  calc
    (-1)^3 + (-1)^2 + (-1)
      = -1 + 1 + (-1) : by rw [h1, h2, h3]
      ... = 0 + (-1) : by ring
      ... = -1 : by ring

end pow_neg_one_sum_l758_758317


namespace find_coefficients_of_series_l758_758567

-- Define the given condition as a theorem in Lean 4
theorem find_coefficients_of_series (a_n : ℕ → ℝ) : 
  (∀ t : ℝ, ∑ n in Nat.range 1000, a_n * (t^n) / (Nat.factorial n) = 
    ((∑ k in Nat.range 500, (t^(2*k)) / (Nat.factorial (2*k)))^2) * 
    ((∑ j in Nat.range 1000, (t^j) / (Nat.factorial j))^3)) →
  (∀ n : ℕ, a_n = (5^n + 2 * 3^n + 1) / 4) :=
by
  intro h
  sorry

end find_coefficients_of_series_l758_758567


namespace max_expression_values_l758_758476

theorem max_expression_values (x : ℝ) : 
  (∃ k1 : ℤ, x = π / 6 + 2 * k1 * π) ∨ (∃ k2 : ℤ, x = 5 * π / 6 + 2 * k2 * π) ↔ 
  ∀ y, (sqrt (2 * sin x) - sin x) ≤ 1 / 2 := 
sorry

end max_expression_values_l758_758476


namespace path_of_Q_l758_758724

theorem path_of_Q (m n : ℝ) (x y : ℝ) (h1 : m^2 + n^2 = 1) (h2 : x = m - n) (h3 : y = 2 * m * n) :
  |x| ≤ real.sqrt 2 ∧ x^2 + y = 1 :=
sorry

end path_of_Q_l758_758724


namespace sodium_chloride_moles_l758_758943

theorem sodium_chloride_moles (NaCl moles_HNO3 moles_HCl : ℕ) 
    (reaction : NaCl + moles_HNO3 ↔ NaNO3 + moles_HCl)
    (moles_HNO3_cond : moles_HNO3 = 3)
    (moles_HCl_cond : moles_HCl = 3) :
    NaCl = 3 :=
begin
  sorry
end

end sodium_chloride_moles_l758_758943


namespace impossible_to_divide_three_similar_parts_l758_758696

theorem impossible_to_divide_three_similar_parts 
  (n : ℝ) 
  (p : ℝ → Prop) 
  (similar : ℝ → ℝ → Prop) 
  (h_similar : ∀ a b : ℝ, similar a b ↔ a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2) : 
  ¬ ∃ p1 p2 p3 : ℝ, p p1 ∧ p p2 ∧ p p3 ∧ (p1 + p2 + p3 = n) ∧ similar p1 p2 ∧ similar p2 p3 ∧ similar p1 p3 :=
sorry

end impossible_to_divide_three_similar_parts_l758_758696


namespace impossible_divide_into_three_similar_l758_758673

noncomputable def sqrt2 : ℝ := Real.sqrt 2

def similar (x y : ℝ) : Prop :=
  x ≤ sqrt2 * y

theorem impossible_divide_into_three_similar (N : ℝ) :
  ¬ ∃ (x y z : ℝ), x + y + z = N ∧ similar x y ∧ similar y z ∧ similar x z := 
by
  sorry

end impossible_divide_into_three_similar_l758_758673


namespace jia_peak_count_l758_758375

-- Define the constants and parameters according to the problem
def AC_ratio : ℝ := 1 / 3
def speed_ratio (v : ℝ) : ℝ := 6 / 5
def downhill_factor : ℝ := 1.5

-- Define the uphill and downhill speeds for Jia and Yi
def uphill_speed_jia (v : ℝ) := 6 * v
def uphill_speed_yi (v : ℝ) := 5 * v

def downhill_speed_jia (v : ℝ) := downhill_factor * uphill_speed_jia v
def downhill_speed_yi (v : ℝ) := downhill_factor * uphill_speed_yi v

-- Define the total number of times Jia has reached the peak when she sees Yi climbing AC for the second time
def number_of_peaks (v : ℝ) : ℕ := 9

theorem jia_peak_count (v : ℝ) :
  (uphill_speed_jia v / uphill_speed_yi v) = (6 / 5) →
  (downhill_speed_jia v = downhill_factor * uphill_speed_jia v) →
  (downhill_speed_yi v = downhill_factor * uphill_speed_yi v) →
  number_of_peaks v = 9 :=
by
  sorry

end jia_peak_count_l758_758375


namespace triangle_prime_sides_l758_758467

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem triangle_prime_sides :
  ∃ (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ is_prime a ∧ is_prime b ∧ is_prime c ∧ 
  a + b + c = 25 ∧
  (a = b ∨ b = c ∨ a = c) ∧
  (∀ (x y z : ℕ), x ≤ y ∧ y ≤ z ∧ is_prime x ∧ is_prime y ∧ is_prime z ∧ x + y + z = 25 → (x, y, z) = (3, 11, 11) ∨ (x, y, z) = (7, 7, 11)) :=
by
  sorry

end triangle_prime_sides_l758_758467


namespace find_length_of_first_tract_l758_758178

theorem find_length_of_first_tract (L : ℝ) : 
  let area_first_tract := L * 500
  let area_second_tract := 250 * 630
  let combined_area := area_first_tract + area_second_tract
  (combined_area = 307500) → (L = 300) :=
by 
  let area_first_tract := L * 500
  let area_second_tract := 250 * 630
  let combined_area := area_first_tract + area_second_tract
  intros h_combined_area_eq
  have h : L * 500 = 307500 - 157500 := by sorry
  have h_div : L = 150000 / 500 := by sorry
  exact h_div

end find_length_of_first_tract_l758_758178


namespace fraction_of_6_l758_758913

theorem fraction_of_6 (x y : ℕ) (h : (x / y : ℚ) * 6 + 6 = 10) : (x / y : ℚ) = 2 / 3 :=
by
  sorry

end fraction_of_6_l758_758913


namespace hyperbola_eccentricity_l758_758109

open Real

-- Conditions from the problem
variables (F₁ F₂ P : Point) (C : ℝ) (a : ℝ)
variables (angle_F1PF2 : angle F₁ P F₂ = 60)
variables (distance_PF1_PF2 : dist P F₁ = 3 * dist P F₂)
variables (focus_condition : 2 * C = dist F₁ F₂)

-- Statement of the problem
theorem hyperbola_eccentricity:
  let e := C / a in
  e = sqrt 7 / 2 :=
sorry

end hyperbola_eccentricity_l758_758109


namespace eval_at_2_l758_758973

theorem eval_at_2 (a b : ℝ) (h : (λ x, a * x^3 + b * x - 4) (-2) = 2) : (λ x, a * x^3 + b * x - 4) 2 = -10 :=
by
  have h1 : a * (-2:ℝ)^3 + b * (-2:ℝ) - 4 = 2 := h
  have h2 : -8 * a - 2 * b - 4 = 2 := by rw[←h1]
  have h3 : 8 * a + 2 * b = -6 := by linarith
  have h4 : (λ x, a * x^3 + b * x - 4) 2 = a * 8 + b * 2 - 4 := by simp
  rw[h3, h4]
  linarith
sorry

end eval_at_2_l758_758973


namespace lettuce_types_l758_758743

/-- Let L be the number of types of lettuce. 
    Given that Terry has 3 types of tomatoes, 4 types of olives, 
    and 2 types of soup. The total number of options for his lunch combo is 48. 
    Prove that L = 2. --/

theorem lettuce_types (L : ℕ) (H : 3 * 4 * 2 * L = 48) : L = 2 :=
by {
  -- beginning of the proof
  sorry
}

end lettuce_types_l758_758743


namespace shaded_region_area_proof_l758_758914

/-- Define the geometric properties of the problem -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

structure Circle :=
  (radius : ℝ)
  (center : ℝ × ℝ)

noncomputable def shaded_region_area (rect : Rectangle) (circle1 circle2 : Circle) : ℝ :=
  let rect_area := rect.width * rect.height
  let circle_area := (Real.pi * circle1.radius ^ 2) + (Real.pi * circle2.radius ^ 2)
  rect_area - circle_area

theorem shaded_region_area_proof : shaded_region_area 
  {width := 10, height := 12} 
  {radius := 3, center := (0, 0)} 
  {radius := 3, center := (12, 10)} = 120 - 18 * Real.pi :=
by
  sorry

end shaded_region_area_proof_l758_758914


namespace sequence_avg_ge_neg_half_c_l758_758094

noncomputable def sequence_condition (n : ℕ) (c : ℝ) (x : ℕ → ℝ) := 
  x 1 = 0 ∧ (∀ i, 2 ≤ i ∧ i ≤ n → |x i| = |x (i - 1) + c|)

theorem sequence_avg_ge_neg_half_c (n : ℕ) (c : ℝ) (x : ℕ → ℝ) 
  (hc : 0 < c) (hx : sequence_condition n c x) :
  (1 / n) * ∑ i in Finset.range n, x (i + 1) ≥ -c / 2 :=
sorry

end sequence_avg_ge_neg_half_c_l758_758094


namespace election_majority_l758_758206

theorem election_majority (V : ℕ) (hV : V = 460) (perc_win : ℝ) (hperc_win : perc_win = 0.70) : 
  let votes_win := perc_win * V
      votes_lose := (1 - perc_win) * V
      majority := votes_win - votes_lose
  in majority = 184 :=
by
  let votes_win := perc_win * V
  let votes_lose := (1 - perc_win) * V
  let majority := votes_win - votes_lose
  have : votes_win = 0.70 * 460, by sorry
  have : votes_lose = 0.30 * 460, by sorry
  have : majority = 0.40 * 460, by sorry
  have : 0.40 * 460 = 184, by sorry
  sorry

end election_majority_l758_758206


namespace find_g_inv_f_at_neg9_l758_758893

noncomputable def inv_of_f_g (f g : ℝ → ℝ) : ℝ :=
  λ x, 7 * x - 4

theorem find_g_inv_f_at_neg9 (f g : ℝ → ℝ) 
  (h : ∀ x, f⁻¹ (g x) = inv_of_f_g f g x) :
  g⁻¹ (f (-9)) = -5/7 := by
  sorry

end find_g_inv_f_at_neg9_l758_758893


namespace smallest_constant_M_l758_758949

theorem smallest_constant_M :
  ∃ M, M = 2 ∧ ∀ (x y z w : ℝ), 0 < x → 0 < y → 0 < z → 0 < w →
    sqrt (x / (y + z + w)) +
    sqrt (y / (x + z + w)) +
    sqrt (z / (x + y + w)) +
    sqrt (w / (x + y + z)) < M := 
begin
  use 2,
  split,
  { refl },
  { intros x y z w hx hy hz hw,
    sorry
  }
end

end smallest_constant_M_l758_758949


namespace martha_total_clothes_l758_758713

def jackets_purchased : ℕ := 4
def tshirts_purchased : ℕ := 9
def jackets_free : ℕ := jackets_purchased / 2
def tshirts_free : ℕ := tshirts_purchased / 3
def total_jackets : ℕ := jackets_purchased + jackets_free
def total_tshirts : ℕ := tshirts_purchased + tshirts_free

theorem martha_total_clothes : total_jackets + total_tshirts = 18 := by
  sorry

end martha_total_clothes_l758_758713


namespace valid_digits_count_l758_758059

theorem valid_digits_count : { n ∈ Finset.range 10 | n > 0 ∧ 15 * n % n = 0 }.card = 5 :=
by
  sorry

end valid_digits_count_l758_758059


namespace ratio_of_areas_l758_758624

variable (s : ℝ)
def equilateral_triangle (a b c : ℝ) := a = b ∧ b = c

theorem ratio_of_areas 
  (hABC : equilateral_triangle s s s)
  (hBB' : ∃ b', 2 * s = b')
  (hCC' : ∃ c', 2 * s = c')
  (hAA' : ∃ a', 2 * s = a') :
  let area_ABC := (real.sqrt 3 / 4) * s^2 in
  let area_A'B'C' := (real.sqrt 3 / 4) * (3 * s)^2 in
  area_A'B'C' / area_ABC = 9 := 
sorry

end ratio_of_areas_l758_758624


namespace proof_l758_758996

noncomputable def problem_statement (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (∀ x : ℝ, |x + a| + |x - b| + c ≥ 4)

theorem proof (a b c : ℝ) (h : problem_statement a b c) :
  a + b + c = 4 ∧ (∀ x : ℝ, 1 / a + 4 / b + 9 / c ≥ 9) :=
by
  sorry

end proof_l758_758996


namespace true_proposition_l758_758629

noncomputable def proposition_1 (l m : Type) [line l] [line m] (α β : Type) [plane α] [plane β] 
  (hp : α ∥ β) (hl : l ⊥ α) : l ⊥ β := sorry

noncomputable def proposition_2 (l m : Type) [line l] [line m] (α β : Type) [plane α] [plane β] 
  (hlm : l ∥ m) (hla : l ⊆ α) (hmb : m ⊆ β) : α ∥ β := sorry

noncomputable def proposition_3 (l m : Type) [line l] [line m] (α : Type) [plane α] 
  (hma : m ⊥ α) (hlm : l ⊥ m) : l ∥ α := sorry

noncomputable def proposition_4 (l m : Type) [line l] [line m] (α β : Type) [plane α] [plane β] 
  (hpab : α ⊥ β) (hla : l ⊆ α) (hmb : m ⊆ β) : l ⊥ m := sorry

theorem true_proposition : 
  proposition_1 ∧ ¬proposition_2 ∧ ¬proposition_3 ∧ ¬proposition_4 :=
begin
  split,
  { sorry }, -- proof of proposition_1
  split,
  { intro h, sorry }, -- proof of ¬proposition_2
  split,
  { intro h, sorry }, -- proof of ¬proposition_3
  { intro h, sorry }, -- proof of ¬proposition_4
end

end true_proposition_l758_758629


namespace remainder_theorem_l758_758352

noncomputable def polynomial := λ x : ℕ, 8 * x^4 - 18 * x^3 + 27 * x^2 - 31 * x + 14
def divisor := λ x : ℕ, 4 * x - 8
def remainder := 30

theorem remainder_theorem (x : ℕ) (h : divisor 2 = 0) : polynomial 2 = remainder := by
  sorry

end remainder_theorem_l758_758352


namespace least_integer_with_exactly_eight_factors_l758_758822

theorem least_integer_with_exactly_eight_factors : ∃ n : ℕ, (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 8 ∨ d = 6 ∨ d = 12 ∨ d = 24) ∧
  (∀ m : ℕ, (∀ d : ℕ, d ∣ m → d = 1 ∨ d = 2 ∨ d = 3 ∨ d
= 4 ∨ d = 8 ∨ d = 6 ∨ d = 12 ∨ d = 24) → m = n) :=
begin
  sorry
end

end least_integer_with_exactly_eight_factors_l758_758822


namespace cos_diff_proof_l758_758139

theorem cos_diff_proof (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2)
  (h2 : Real.cos A + Real.cos B = 1) :
  Real.cos (A - B) = 5 / 8 := 
by
  sorry

end cos_diff_proof_l758_758139


namespace distinct_triples_sum_l758_758630

theorem distinct_triples_sum (m : ℕ) (hodd : m % 2 = 1) (hge : m ≥ 5) : 
  let T : ℕ := ∑ (a₁ ∈ Finset.range (m+1)) (a₂ ∈ Finset.range (m+1)) (a₃ ∈ Finset.range (m+1)),
    if a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ = m - (a₁ + a₂) then 1 else 0 
  in T = m * (m - 3) / 6 := 
sorry

end distinct_triples_sum_l758_758630


namespace count_divisible_digits_l758_758067

def is_divisible (a b : ℕ) : Prop := a % b = 0

theorem count_divisible_digits :
  let count := (finset.range 10).filter (λ n, n > 0 ∧ is_divisible (15 * n) n) in
  count.card = 6 :=
by
  let valid_digits := [1, 2, 3, 5, 6, 9].to_finset
  have valid_digits_card : valid_digits.card = 6 := by simp
  have matching_digits : ∀ n : ℕ, n > 0 ∧ n < 10 ∧ 15 * n % n = 0 ↔ n ∈ valid_digits := by
    intros n
    simp
    split
      intros ⟨n_pos, n_lt, hn⟩
      fin_cases n
        simp [is_divisible] at *
        contradiction,
        [1, 2, 3, 5, 6, 9].cases_on (λ n, n ∈ valid_digits) (by simp; exact valid_digits_card)
  
  sorry

end count_divisible_digits_l758_758067


namespace percentage_of_16_l758_758387

theorem percentage_of_16 (p : ℝ) (h : (p / 100) * 16 = 0.04) : p = 0.25 :=
by
  sorry

end percentage_of_16_l758_758387


namespace hyperbola_eccentricity_l758_758148

theorem hyperbola_eccentricity (a b : ℝ) (h : b > 0) (h_asymptote : b / a = sqrt 3) :
  let e := sqrt (1 + (b^2 / a^2)) in e = 2 :=
by sorry

end hyperbola_eccentricity_l758_758148


namespace _l758_758131

noncomputable def hyperbola_eccentricity_theorem {F1 F2 P : Point} 
  (hyp : is_hyperbola F1 F2 P)
  (angle_F1PF2 : ∠F1 P F2 = 60)
  (dist_PF1_3PF2 : dist P F1 = 3 * dist P F2) : 
  eccentricity F1 F2 = sqrt 7 / 2 :=
by 
  sorry

end _l758_758131


namespace ratio_vegan_gluten_free_cupcakes_l758_758398

theorem ratio_vegan_gluten_free_cupcakes :
  let total_cupcakes := 80
  let gluten_free_cupcakes := total_cupcakes / 2
  let vegan_cupcakes := 24
  let non_vegan_gluten_cupcakes := 28
  let vegan_gluten_free_cupcakes := gluten_free_cupcakes - non_vegan_gluten_cupcakes
  (vegan_gluten_free_cupcakes / vegan_cupcakes) = 1 / 2 :=
by {
  let total_cupcakes := 80
  let gluten_free_cupcakes := total_cupcakes / 2
  let vegan_cupcakes := 24
  let non_vegan_gluten_cupcakes := 28
  let vegan_gluten_free_cupcakes := gluten_free_cupcakes - non_vegan_gluten_cupcakes
  have h : vegan_gluten_free_cupcakes = 12 := by norm_num
  have r : 12 / 24 = 1 / 2 := by norm_num
  exact r
}

end ratio_vegan_gluten_free_cupcakes_l758_758398


namespace range_of_m_l758_758989

variable (m : ℝ)

def proposition_p : Prop := (m - 2) / (m - 3) ≤ 2 / 3

def proposition_q : Prop := ∀ x : ℝ, ¬ (x^2 - 4 * x + m^2 ≤ 0)

theorem range_of_m (h₀ : proposition_p ∨ proposition_q)
    (h₁ : ¬ (proposition_p ∧ proposition_q)) :
  m ∈ Set.Iic (-2) ∪ Set.Icc 0 2 ∪ Set.Ici 3 := sorry

end range_of_m_l758_758989


namespace vector_trigonometric_identity_l758_758172

theorem vector_trigonometric_identity 
  (θ : ℝ)
  (h1 : θ < π / 2)  -- θ is an acute angle
  (a : ℝ × ℝ)
  (b : ℝ × ℝ)
  (ha : a = (Real.tan θ, -1))
  (hb : b = (1, -2))
  (h2 : (a.1 + b.1, a.2 + b.2) ∙ (a.1 - b.1, a.2 - b.2) = 0) :
  (1 / (2 * Real.sin θ * Real.cos θ + Real.cos θ ^ 2)) = 1 := sorry

end vector_trigonometric_identity_l758_758172


namespace arrange_natural_numbers_l758_758959

noncomputable def is_possible_arrangement (n : ℕ) : Prop :=
  ∃ (f : ℕ → ℕ), (∀ i : ℕ, f i ∈ (range n).map (λ x, x + 1)) ∧ 
  (∀ (i j k : ℕ), i < j ∧ j < k → (2 * f j ≠ f i + f k))

theorem arrange_natural_numbers (n : ℕ) : is_possible_arrangement n :=
sorry

end arrange_natural_numbers_l758_758959


namespace quadratic_real_roots_iff_l758_758187

theorem quadratic_real_roots_iff (k : ℝ) : 
  (∃ x : ℝ, (k-1) * x^2 + 3 * x - 1 = 0) ↔ k ≥ -5 / 4 ∧ k ≠ 1 := sorry

end quadratic_real_roots_iff_l758_758187


namespace impossible_divide_into_three_similar_l758_758672

noncomputable def sqrt2 : ℝ := Real.sqrt 2

def similar (x y : ℝ) : Prop :=
  x ≤ sqrt2 * y

theorem impossible_divide_into_three_similar (N : ℝ) :
  ¬ ∃ (x y z : ℝ), x + y + z = N ∧ similar x y ∧ similar y z ∧ similar x z := 
by
  sorry

end impossible_divide_into_three_similar_l758_758672


namespace intersection_sets_l758_758528

open Set

def A := {x : ℤ | abs x < 3}
def B := {x : ℤ | abs x > 1}

theorem intersection_sets :
  A ∩ B = {-2, 2} := by
  sorry

end intersection_sets_l758_758528


namespace area_of_circumcircle_of_ABC_l758_758218

-- We need to define the concepts and conditions given in Lean notation.
noncomputable def radius_of_circumcircle (a : ℝ) (A : ℝ) : ℝ :=
  a / (2 * Real.sin A)

-- Define the problem statement in Lean
theorem area_of_circumcircle_of_ABC (A : ℝ) (a : ℝ) (hA : A = Real.pi / 3) (ha : a = Real.sqrt 3) :
  let R := radius_of_circumcircle a A
  in π * R^2 = π :=
by
  sorry

end area_of_circumcircle_of_ABC_l758_758218


namespace compute_expression_l758_758424

theorem compute_expression : 7^3 - 5 * (6^2) + 2^4 = 179 :=
by
  sorry

end compute_expression_l758_758424


namespace original_bill_amount_l758_758863

theorem original_bill_amount (n : ℕ) (share : ℝ) (tip_rate : ℝ) (total_paid : ℝ) (original_bill : ℝ) :
  n = 9 →
  share = 16.99 →
  tip_rate = 0.10 →
  total_paid = n * share →
  original_bill ≈ total_paid / (1 + tip_rate) →
  original_bill ≈ 139.01 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end original_bill_amount_l758_758863


namespace hyperbola_eccentricity_l758_758130

theorem hyperbola_eccentricity (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
    (C : P → Prop) 
    (angle_F1PF2 : ∀ {x y z : P}, ∀ (h : x ∈ C) (h₁ : y ∈ C) (h₂ : z ∈ C), is_angle x y z = 60)
    (dist_PF1_PF2 : ∀ (h : P ∈ C), dist P F1 = 3 * dist P F2) : 
    F1 ∈ C → F2 ∈ C → eccentricity C = (Real.sqrt 7) / 2 :=
by sorry

end hyperbola_eccentricity_l758_758130


namespace a4_plus_a5_eq_27_l758_758212

-- Define the geometric sequence conditions
variables (a : ℕ → ℝ) (q : ℝ)
axiom a_pos : ∀ n, a n > 0
axiom a_2 : a 2 = 1 - a 1
axiom a_4 : a 4 = 9 - a 3

-- Define the geometric sequence property
axiom geom_seq : ∀ n, a (n + 1) = a n * q

theorem a4_plus_a5_eq_27 : a 4 + a 5 = 27 := sorry

end a4_plus_a5_eq_27_l758_758212


namespace find_volume_tetrahedron_l758_758208

noncomputable def volume_of_tetrahedron (EF E F G H : Point) (EF = 4 : ℝ) 
  (area_EFG = 20 : ℝ) (area_EFH = 16 : ℝ) (angle_EFG_EFH = 45 : ℝ) : ℝ :=
(sorry)

theorem find_volume_tetrahedron (EF E F G H : Point) 
  (h_EF : dist(E, F) = 4) 
  (h_area_EFG : area(E, F, G) = 20) 
  (h_area_EFH : area(E, F, H) = 16) 
  (h_angle_EFG_EFH : angle(E, F, G, H) = 45) :
  volume_of_tetrahedron(EF E F G H 4 20 16 45) = (80 * real.sqrt(2)) / 3 :=
sorry

end find_volume_tetrahedron_l758_758208


namespace find_BP_QT_l758_758207

-- Given conditions
variables (A B C D P Q R S T : Type)
variables [rectABCD : Rectangle A B C D] [APD : ∠ A P D = 90°]
variables [BP_PT : BP = PT] [TS_perpendicular : Line TS ⊥ BC]
variables [PD_intersect_TS : intersection_line PD TS = Q]
variables [RA_pass_Q : Point R on CD, Line RA passes through Q]
variables [PA_AQ_QP : PA = 24] [AQ_length : AQ = 10] [QP_length : QP = 26]

theorem find_BP_QT : BP = 8 * sqrt 5 ∧ QT = 2 * sqrt 89 := sorry

end find_BP_QT_l758_758207


namespace intersection_A_B_l758_758513

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_A_B : A ∩ B = {-2, 2} := by sorry

end intersection_A_B_l758_758513


namespace intersection_one_point_l758_758087

open Set

def A (x y : ℝ) : Prop := x^2 - 3*x*y + 4*y^2 = 7 / 2
def B (k x y : ℝ) : Prop := k > 0 ∧ k*x + y = 2

theorem intersection_one_point (k : ℝ) (h : k > 0) :
  (∃ x y : ℝ, A x y ∧ B k x y) → (∀ x₁ y₁ x₂ y₂ : ℝ, (A x₁ y₁ ∧ B k x₁ y₁) ∧ (A x₂ y₂ ∧ B k x₂ y₂) → x₁ = x₂ ∧ y₁ = y₂) ↔ k = 1 / 4 :=
sorry

end intersection_one_point_l758_758087


namespace greatest_fraction_l758_758360

theorem greatest_fraction :
  (∃ frac, frac ∈ {
    (44444 : ℚ)/55555,
    (5555 : ℚ)/6666,
    (666 : ℚ)/777,
    (77 : ℚ)/88,
    (8 : ℚ)/9
  } ∧ frac = (8 : ℚ)/9 ∧ ∀ f ∈ {
    (44444 : ℚ)/55555,
    (5555 : ℚ)/6666,
    (666 : ℚ)/777,
    (77 : ℚ)/88,
    (8 : ℚ)/9
  }, frac ≥ f) :=
sorry

end greatest_fraction_l758_758360


namespace range_of_x_l758_758488

theorem range_of_x (k : ℝ) (x : ℝ) :
  (∃ k : ℝ, x^4 - 2 * k * x^2 + k^2 + 2 * k - 3 = 0) → -real.sqrt 2 ≤ x ∧ x ≤ real.sqrt 2 :=
by 
  sorry

end range_of_x_l758_758488


namespace intersection_A_B_is_C_l758_758518

def A := { x : ℤ | abs x < 3 }
def B := { x : ℤ | abs x > 1 }
def C := { -2, 2 : ℤ }

theorem intersection_A_B_is_C : (A ∩ B) = C := 
  sorry

end intersection_A_B_is_C_l758_758518


namespace base_five_product_l758_758901

theorem base_five_product (n1 n2 : ℕ) (h1 : n1 = 1 * 5^2 + 3 * 5^1 + 1 * 5^0) 
                          (h2 : n2 = 1 * 5^1 + 2 * 5^0) :
  let product_dec := (n1 * n2 : ℕ)
  let product_base5 := 2 * 125 + 1 * 25 + 2 * 5 + 2 * 1
  product_dec = 287 ∧ product_base5 = 2122 := by
                                -- calculations to verify statement omitted
                                sorry

end base_five_product_l758_758901


namespace focus_of_parabola_l758_758024

theorem focus_of_parabola (x f d : ℝ) (h : ∀ x, y = 4 * x^2 → (x = 0 ∧ y = f) → PF^2 = PQ^2 ∧ 
(PF^2 = x^2 + (4 * x^2 - f) ^ 2) := (PQ^2 = (4 * x^2 - d) ^ 2)) : 
  f = 1 / 16 := 
sorry

end focus_of_parabola_l758_758024


namespace limit_of_sequence_l758_758451

theorem limit_of_sequence : 
  (Real.Lim (fun n : ℕ => (2 * (n : ℝ) - 5) / ((n : ℝ) + 1)) = 2) := 
  by { sorry }

end limit_of_sequence_l758_758451


namespace count_divisible_digits_l758_758069

def is_divisible (a b : ℕ) : Prop := a % b = 0

theorem count_divisible_digits :
  let count := (finset.range 10).filter (λ n, n > 0 ∧ is_divisible (15 * n) n) in
  count.card = 6 :=
by
  let valid_digits := [1, 2, 3, 5, 6, 9].to_finset
  have valid_digits_card : valid_digits.card = 6 := by simp
  have matching_digits : ∀ n : ℕ, n > 0 ∧ n < 10 ∧ 15 * n % n = 0 ↔ n ∈ valid_digits := by
    intros n
    simp
    split
      intros ⟨n_pos, n_lt, hn⟩
      fin_cases n
        simp [is_divisible] at *
        contradiction,
        [1, 2, 3, 5, 6, 9].cases_on (λ n, n ∈ valid_digits) (by simp; exact valid_digits_card)
  
  sorry

end count_divisible_digits_l758_758069


namespace focus_of_parabola_l758_758025

theorem focus_of_parabola (x f d : ℝ) (h : ∀ x, y = 4 * x^2 → (x = 0 ∧ y = f) → PF^2 = PQ^2 ∧ 
(PF^2 = x^2 + (4 * x^2 - f) ^ 2) := (PQ^2 = (4 * x^2 - d) ^ 2)) : 
  f = 1 / 16 := 
sorry

end focus_of_parabola_l758_758025


namespace monotonic_intervals_and_k_range_summation_inequality_l758_758553

def f (x k : ℝ) : ℝ := Real.log (x - 1) - k * (x - 1) + 1

theorem monotonic_intervals_and_k_range (k : ℝ) :
  (∀ x > 1, f x k ≤ 0) → (k ≥ 1) :=
sorry

theorem summation_inequality (n : ℕ) (h : 1 < n) :
  ( ∑ i in Finset.range n, Real.log (i+2) / (i + 3) ) < n * (n - 1) / 4 :=
sorry

end monotonic_intervals_and_k_range_summation_inequality_l758_758553


namespace impossible_to_divide_into_three_similar_piles_l758_758652

def similar (a b : ℝ) : Prop :=
  a / b ≤ real.sqrt 2 ∧ b / a ≤ real.sqrt 2

theorem impossible_to_divide_into_three_similar_piles (pile : ℝ) (h : 0 < pile) :
  ¬ ∃ (x y z : ℝ), 
    x + y + z = pile ∧
    similar x y ∧ similar y z ∧ similar z x :=
by
  sorry

end impossible_to_divide_into_three_similar_piles_l758_758652


namespace proof_problem_l758_758918

noncomputable def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def condition (f : ℝ → ℝ) : Prop :=
∀ x1 x2, (0 ≤ x1) → (0 ≤ x2) → (x1 ≠ x2) → (x1 - x2) * (f x1 - f x2) > 0

theorem proof_problem (f : ℝ → ℝ) (hf_even : even_function f) (hf_condition : condition f) :
  f 1 < f (-2) ∧ f (-2) < f 3 := sorry

end proof_problem_l758_758918


namespace find_smallest_k_l758_758377

open Nat

-- Definition of a valid grid that satisfies the given conditions
def valid_grid (rows : ℕ) (columns : ℕ) (n : ℕ) : Prop :=
  ∃ m : matrix (Fin rows) (Fin columns) ℕ, 
    (∀ j : Fin columns, ∑ i, m i j = n) ∧
    (∀ i : Fin rows, ∀ a : Fin (n + 1), ∃ j : Fin columns, m i j = a)

noncomputable def smallest_k (n : ℕ) : ℕ := Nat.ceil (3 * (n + 1) / 2 : ℚ)

theorem find_smallest_k {n : ℕ} (hn : n > 0) : 
  ∃ k : ℕ, (∀ m : ℕ, valid_grid 3 m n → m ≥ k) ∧ 
           (valid_grid 3 k n) := 
begin
  let k := smallest_k n,
  use k,
  split,
  {
    intros m h_valid_grid,
    unfold smallest_k at *,
    simp only [Nat.ceil_le] at *,
    unfold valid_grid at h_valid_grid,
    let total_sum := 3 * (n * (n + 1) / 2),
    have H : m * n = total_sum,
    {
      cases h_valid_grid with grid h_grid,
      cases h_grid with h_col_sum h_row_appear,
      rw [← Nat.cast_add, ← Finset.sum_fin_eq_cast_sum_range (range m) n],
      sorry,
    },
    exact Nat.le_of_sub_eq_zero (by linarith),
  },
  sorry,
end

end find_smallest_k_l758_758377


namespace function_domain_l758_758435

open Set

def f (x : ℝ) : ℝ := (sqrt (-(x^2) - 3 * x + 4)) / (log (x + 1))

theorem function_domain :
  {x : ℝ | (-x^2 - 3 * x + 4 ≥ 0) ∧ (x + 1 > 0) ∧ (x + 1 ≠ 1)} =
  Ioo -1 0 ∪ Icc 0 1 :=
by
  sorry

end function_domain_l758_758435


namespace arithmetic_sequence_sum_ratio_l758_758475

theorem arithmetic_sequence_sum_ratio :
  (∀ n : ℕ, ∃ S_n T_n : ℝ, S_n = (n + 1) * a + n * d ∧ T_n = (n + 1) * b + n * e ∧ S_n / T_n = (7 * n + 2) / (n + 3))
  → (∑ i in (finset.range 21), (a + i * d)) / (∑ i in (finset.range 21), (b + i * e)) = 149 / 24 :=
sorry

end arithmetic_sequence_sum_ratio_l758_758475


namespace intersection_eq_l758_758525

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_eq : A ∩ B = ({-2, 2} : Set ℤ) :=
by
  sorry

end intersection_eq_l758_758525


namespace meaningful_expression_range_l758_758774

theorem meaningful_expression_range (x : ℝ) :
  (x ≥ 1 ∧ x ≠ 3) ↔ (∃ y, y = (sqrt (x - 1) / (3 - x))) :=
by
  sorry

end meaningful_expression_range_l758_758774


namespace hyperbola_eccentricity_l758_758125

theorem hyperbola_eccentricity (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
    (C : P → Prop) 
    (angle_F1PF2 : ∀ {x y z : P}, ∀ (h : x ∈ C) (h₁ : y ∈ C) (h₂ : z ∈ C), is_angle x y z = 60)
    (dist_PF1_PF2 : ∀ (h : P ∈ C), dist P F1 = 3 * dist P F2) : 
    F1 ∈ C → F2 ∈ C → eccentricity C = (Real.sqrt 7) / 2 :=
by sorry

end hyperbola_eccentricity_l758_758125


namespace intersection_A_B_l758_758512

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_A_B : A ∩ B = {-2, 2} := by sorry

end intersection_A_B_l758_758512


namespace valid_digits_count_l758_758057

theorem valid_digits_count : { n ∈ Finset.range 10 | n > 0 ∧ 15 * n % n = 0 }.card = 5 :=
by
  sorry

end valid_digits_count_l758_758057


namespace exists_increasing_cost_sequence_20_l758_758612

noncomputable def digit_price (n : ℕ) : ℕ :=
  if n = 0 then 20
  else if n = 1 then 21
  else if n = 2 then 22
  else if n = 3 then 23
  else if n = 4 then 24
  else if n = 5 then 25
  else if n = 6 then 26
  else if n = 7 then 27
  else if n = 8 then 28
  else if n = 9 then 29
  else 0

def number_price (n : ℕ) : ℕ :=
  (n.digitToString.map (fun c => digit_price c.toNat - '0'.toNat)).sum

theorem exists_increasing_cost_sequence_20 :
  ∃ (s : ℕ → ℕ), (∀ n, s n = number_price n) ∧ (∃ k, ∀ i < 20, s (k + i) < s (k + i + 1)) :=
by
  sorry

end exists_increasing_cost_sequence_20_l758_758612


namespace least_positive_integer_with_eight_factors_l758_758818

noncomputable def numDivisors (n : ℕ) : ℕ :=
  (List.range (n+1)).count (λ d => d > 0 ∧ n % d = 0)

theorem least_positive_integer_with_eight_factors : ∃ n : ℕ, n > 0 ∧ numDivisors n = 8 ∧ (∀ m : ℕ, m > 0 → numDivisors m = 8 → n ≤ m) := 
  sorry

end least_positive_integer_with_eight_factors_l758_758818


namespace explicit_formula_for_f_range_of_c_for_inequality_maximum_value_of_y_l758_758547

-- Problem 1: Proving the explicit formula for f(x)
theorem explicit_formula_for_f (a b : ℝ) :
  (a = -3 ∧ b = 5) ↔ (∀ x, f x = -3 * x^2 - 3 * x + 18) :=
sorry

-- Problem 2: Proving the range of c for the inequality to hold over ℝ
theorem range_of_c_for_inequality (c : ℝ) :
  (∀ x, -3 * x^2 + 5 * x + c ≤ 0) ↔ (c ≤ -25/12) :=
sorry

-- Problem 3: Proving the maximum value of y
theorem maximum_value_of_y (f : ℝ → ℝ) (x : ℝ) :
  (∀ x > -1, y(x) = (f(x) - 21) / (x + 1)) → (∃ y_max, y_max = -3) :=
sorry

end explicit_formula_for_f_range_of_c_for_inequality_maximum_value_of_y_l758_758547


namespace impossible_divide_into_three_similar_parts_l758_758690

noncomputable def similar (x y : ℝ) : Prop := x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

theorem impossible_divide_into_three_similar_parts (s : ℝ → ℝ → Prop) :
  (∀ s, similar s)) → ¬ (∃ a b c : ℝ, s a b → s b c → s c a → a + b + c = 1) :=
by
  intros h_similar h
  sorry

end impossible_divide_into_three_similar_parts_l758_758690


namespace three_digit_number_is_275_l758_758881

noncomputable def digits (n : ℕ) : ℕ × ℕ × ℕ :=
  (n / 100 % 10, n / 10 % 10, n % 10)

theorem three_digit_number_is_275 :
  ∃ (n : ℕ), n / 100 % 10 + n % 10 = n / 10 % 10 ∧
              7 * (n / 100 % 10) = n % 10 + n / 10 % 10 + 2 ∧
              n / 100 % 10 + n / 10 % 10 + n % 10 = 14 ∧
              n = 275 :=
by
  sorry

end three_digit_number_is_275_l758_758881


namespace ratio_MN_BD_is_half_l758_758263

variable (A B C D P Q M N O : Type)

def is_parallelogram (A B C D : Type) : Prop := sorry

def is_midpoint (O : Type) (X Y : Type) : Prop := sorry

def divides_into_trisegments (P Q : Type) (X Y : Type) : Prop := sorry

def intersects_at (C P M X Y : Type) : Prop := sorry

def is_medium_ratio (MN BD : Type) (ratio : ℚ) : Prop := sorry

theorem ratio_MN_BD_is_half
  (A B C D P Q M N O : Type)
  (h_parallelogram : is_parallelogram A B C D)
  (h_midpoint_O : is_midpoint O B D)
  (h_trisegments_BP_PQ_QD : divides_into_trisegments P Q B D)
  (h_intersects_CP_AB_at_M : intersects_at C P M A B)
  (h_intersects_CQ_AD_at_N : intersects_at C Q N A D) :
  is_medium_ratio M N B D (1/2) :=
sorry

end ratio_MN_BD_is_half_l758_758263


namespace shortest_tangent_length_l758_758625

/-- Define the circles C1 and C2 -/
def C1 (x y : ℝ) : Prop := (x - 3) ^ 2 + (y - 4) ^ 2 = 25
def C2 (x y : ℝ) : Prop := (x + 6) ^ 2 + (y + 5) ^ 2 = 16

/-- The statement to be proved: the shortest line segment tangent to both circles C1 and C2 -/
theorem shortest_tangent_length :
  ∃ P Q : ℝ × ℝ, 
  (∃ x y, C1 P.1 P.2 ∧ P = (x, y)) ∧
  (∃ u v, C2 Q.1 Q.2 ∧ Q = (u, v)) ∧
  PQ_tangent : P ≠ Q ∧
  tangent_to_circle P Q C1 C2 → 
  dist P Q = 9 * real.sqrt 2 - 9 :=
sorry

end shortest_tangent_length_l758_758625


namespace focus_of_parabola_l758_758007

theorem focus_of_parabola (f d : ℝ) :
  (∀ x : ℝ, x^2 + (4*x^2 - f)^2 = (4*x^2 - d)^2) → 8*f + 8*d = 1 → f^2 = d^2 → f = 1/16 :=
by
  intro hEq hCoeff hSq
  sorry

end focus_of_parabola_l758_758007


namespace martin_walk_distance_l758_758704

-- Define the conditions
def time : ℝ := 6 -- Martin's walking time in hours
def speed : ℝ := 2 -- Martin's walking speed in miles per hour

-- Define the target distance
noncomputable def distance : ℝ := 12 -- Distance from Martin's house to Lawrence's house

-- The theorem to prove the target distance given the conditions
theorem martin_walk_distance : (speed * time = distance) :=
by
  sorry

end martin_walk_distance_l758_758704


namespace identity_proof_l758_758283

noncomputable def log_b := λ (b a : ℝ), real.log a / real.log b

theorem identity_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hbn1 : b ≠ 1) :
  a ^ (log_b b (log_b b a) / log_b b a) = log_b b a :=
by
  sorry

end identity_proof_l758_758283


namespace parabola_focus_directrix_l758_758873

noncomputable def parabola_equation (x y : ℝ) : Prop := 
  25 * x^2 + 40 * x * y + 16 * y^2 - 4 * x - 128 * y + 256 = 0

theorem parabola_focus_directrix {a b c d e f : ℝ}
  (focus : (ℝ × ℝ)) (directrix : ℝ → ℝ → Prop)
  (h_focus : focus = (2, 4))
  (h_directrix : ∀ x y : ℝ, directrix x y ↔ 4 * x + 5 * y = 20) :
  (∃ (x y : ℝ), parabola_equation x y) :=
begin
  use [1, 1], -- This is a placeholder to ensure the statement compiles; the actual proof follows from the given solution.
  sorry
end

end parabola_focus_directrix_l758_758873


namespace product_of_consecutive_numbers_not_100th_power_l758_758281

theorem product_of_consecutive_numbers_not_100th_power (n : ℕ):
  ¬ (∃ k : ℕ, (list.range 100).prod (λ i, n + i) = k ^ 100) :=
by
  sorry

end product_of_consecutive_numbers_not_100th_power_l758_758281


namespace range_of_k_for_intersecting_circles_l758_758544

/-- Given circle \( C \) with equation \( x^2 + y^2 - 8x + 15 = 0 \) and a line \( y = kx - 2 \),
    prove that if there exists at least one point on the line such that a circle with this point
    as the center and a radius of 1 intersects with circle \( C \), then \( 0 \leq k \leq \frac{4}{3} \). -/
theorem range_of_k_for_intersecting_circles (k : ℝ) :
  (∃ (x y : ℝ), y = k * x - 2 ∧ (x - 4) ^ 2 + y ^ 2 - 1 ≤ 1) → 0 ≤ k ∧ k ≤ 4 / 3 :=
by {
  sorry
}

end range_of_k_for_intersecting_circles_l758_758544


namespace determine_coefficients_l758_758477

theorem determine_coefficients (A B C : ℝ) 
  (h1 : 3 * A - 1 = 0)
  (h2 : 3 * A^2 + 3 * B = 0)
  (h3 : A^3 + 6 * A * B + 3 * C = 0) :
  A = 1 / 3 ∧ B = -1 / 9 ∧ C = 5 / 81 :=
by 
  sorry

end determine_coefficients_l758_758477


namespace parallel_lines_k_l758_758564

theorem parallel_lines_k (k : ℝ) 
  (h₁ : k ≠ 0)
  (h₂ : ∀ x y : ℝ, (x - k * y - k = 0) = (y = (1 / k) * x - 1))
  (h₃ : ∀ x : ℝ, (y = k * (x - 1))) :
  k = -1 :=
by
  sorry

end parallel_lines_k_l758_758564


namespace martha_clothes_total_l758_758716

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

end martha_clothes_total_l758_758716


namespace question_1_question_2_l758_758967

def sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 2
  else if n = 1 then 6
  else sequence_a (n - 1) * 5 - sequence_a (n - 2) * 6

def sequence_b (n : ℕ) : ℕ := 
  Int.floor (Real.log (sequence_a n + 1) / Real.log 5)

def sequence_s (n : ℕ) : Real :=
  ∑ i in Finset.range n, 1000 / (sequence_b i * sequence_b (i + 1))

theorem question_1 :
  ∃ r : ℕ, ∀ n : ℕ, n > 0 → sequence_a (n + 1) - sequence_a n = 4 * (5 ^ (n - 1)) := 
sorry

theorem question_2 :
  [sequence_s 2023] = 999 :=
sorry

end question_1_question_2_l758_758967


namespace angle_of_non_collinear_vectors_l758_758502

noncomputable def angle_between_vectors (a b : ℝ^3) : ℝ :=
  sorry

theorem angle_of_non_collinear_vectors
  (a b : ℝ^3)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : ∥a∥ = ∥b∥)
  (h4 : ⟪a, a - 2 • b⟫ = 0) :
  angle_between_vectors a b = π / 3 :=
sorry

end angle_of_non_collinear_vectors_l758_758502


namespace equal_AS_SP_l758_758228

-- Defining the points and conditions
variables {A B C P Q S : Type} [IsPoint A] [IsPoint B] [IsPoint C] [IsPoint P] [IsPoint Q] [IsPoint S]

-- Assume: A triangle ABC
-- P is the midpoint of BC
-- Q is on CA such that |CQ| = 2|QA|
-- S is the intersection of BQ and AP
axiom triangle_ABC : IsTriangle A B C
axiom midpoint_P : IsMidpoint P B C
axiom point_Q_on_CA : PointOnSegment Q C A
axiom ratio_CQ_QA : Distance CQ = 2 * Distance QA
axiom intersection_S : IntersectPoint S (LineThrough B Q) (LineThrough A P)

-- Proving |AS| = |SP|
theorem equal_AS_SP : Distance AS = Distance SP :=
by
  sorry

end equal_AS_SP_l758_758228


namespace impossible_to_divide_into_three_similar_piles_l758_758645

def similar (a b : ℝ) : Prop :=
  a / b ≤ real.sqrt 2 ∧ b / a ≤ real.sqrt 2

theorem impossible_to_divide_into_three_similar_piles (pile : ℝ) (h : 0 < pile) :
  ¬ ∃ (x y z : ℝ), 
    x + y + z = pile ∧
    similar x y ∧ similar y z ∧ similar z x :=
by
  sorry

end impossible_to_divide_into_three_similar_piles_l758_758645


namespace impossible_to_divide_three_similar_parts_l758_758695

theorem impossible_to_divide_three_similar_parts 
  (n : ℝ) 
  (p : ℝ → Prop) 
  (similar : ℝ → ℝ → Prop) 
  (h_similar : ∀ a b : ℝ, similar a b ↔ a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2) : 
  ¬ ∃ p1 p2 p3 : ℝ, p p1 ∧ p p2 ∧ p p3 ∧ (p1 + p2 + p3 = n) ∧ similar p1 p2 ∧ similar p2 p3 ∧ similar p1 p3 :=
sorry

end impossible_to_divide_three_similar_parts_l758_758695


namespace find_t_l758_758970

def vector (α : Type) : Type := (α × α)

def dot_product (v1 v2 : vector ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def orthogonal (v1 v2 : vector ℝ) : Prop :=
  dot_product v1 v2 = 0

theorem find_t (t : ℝ) :
  let a : vector ℝ := (1, -1)
  let b : vector ℝ := (2, t)
  orthogonal a b → t = 2 := by
  sorry

end find_t_l758_758970


namespace sixteen_power_sum_leq_eight_l758_758272

theorem sixteen_power_sum_leq_eight {a b c : ℝ} (ha : a ≥ 1 / 4) (hb : b ≥ 1 / 4) (hc : c ≥ 1 / 4) (h_sum : a + b + c = 1) : 16^a + 16^b + 16^c ≤ 8 := 
by 
  sorry

end sixteen_power_sum_leq_eight_l758_758272


namespace martha_total_clothes_l758_758719

-- Define the conditions
def jackets_bought : ℕ := 4
def t_shirts_bought : ℕ := 9
def free_jacket_condition : ℕ := 2
def free_t_shirt_condition : ℕ := 3

-- Define calculations based on conditions
def free_jackets : ℕ := jackets_bought / free_jacket_condition
def free_t_shirts : ℕ := t_shirts_bought / free_t_shirt_condition
def total_jackets := jackets_bought + free_jackets
def total_t_shirts := t_shirts_bought + free_t_shirts
def total_clothes := total_jackets + total_t_shirts

-- Prove the total number of clothes
theorem martha_total_clothes : total_clothes = 18 :=
by
    sorry

end martha_total_clothes_l758_758719


namespace _l758_758134

noncomputable def hyperbola_eccentricity_theorem {F1 F2 P : Point} 
  (hyp : is_hyperbola F1 F2 P)
  (angle_F1PF2 : ∠F1 P F2 = 60)
  (dist_PF1_3PF2 : dist P F1 = 3 * dist P F2) : 
  eccentricity F1 F2 = sqrt 7 / 2 :=
by 
  sorry

end _l758_758134


namespace proof_expression_one_proof_expression_two_l758_758905

noncomputable def calculate_expression_one : ℚ :=
  (1 : ℚ) * (27/8) ^ (-2/3 : ℚ) + (2/1000) ^ (-1/2 : ℚ) -
  (10 : ℚ) * (real.sqrt 5 - 2)⁻¹ + (real.sqrt 2 - real.sqrt 3) ^ (0 : ℚ)

noncomputable def calculate_expression_two : ℚ :=
  real.log_base 2.5 6.25 + real.log10 (1/100) + real.log (real.sqrt real.e) + 2^(1 + real.log_base 2 3)

theorem proof_expression_one : calculate_expression_one = -167/9 :=
by
  sorry

theorem proof_expression_two : calculate_expression_two = 13/2 :=
by
  sorry

end proof_expression_one_proof_expression_two_l758_758905


namespace find_g_eq_lambda_val_l758_758085

noncomputable def f (x c : ℝ) := x^2 + c
noncomputable def g (x c : ℝ) := f (f x c) c

-- The first problem statement
theorem find_g_eq (c : ℝ) (H : f (f x c) c = f (x^2 + 1) c) : 
  g x c = x^4 + 2*x^2 + 2 :=
  sorry

noncomputable def φ (x λ c : ℝ) := (x^4 + 2*x^2 + 2) - λ * (x^2 + c)

-- The second problem statement
theorem lambda_val (λ : ℝ) (H : ∀ x : ℝ, x < -1 → φ x λ 1 < φ x λ 0) : 
  λ = 4 :=
  sorry

end find_g_eq_lambda_val_l758_758085


namespace geometric_log_sum_l758_758594

theorem geometric_log_sum (a : ℕ → ℝ) (h₁ : ∀ n, 0 < a n) 
(h₂ : a 5 * a 6 = 9) (h₃ : ∀ n, a (n + 1) / a n = a 2 / a 1) :
  log 3 (a 1) + log 3 (a 2) + log 3 (a 3) + log 3 (a 4) + log 3 (a 5)
  + log 3 (a 6) + log 3 (a 7) + log 3 (a 8) + log 3 (a 9) + log 3 (a 10) = 10 := 
sorry

end geometric_log_sum_l758_758594


namespace even_odd_subsets_equal_l758_758089

theorem even_odd_subsets_equal (S : Finset α) (h : S.nonempty) :
  (Finset.filter (λ s, s.card % 2 = 0) (S.powerset)).card = 
  (Finset.filter (λ s, s.card % 2 = 1) (S.powerset)).card :=
sorry

end even_odd_subsets_equal_l758_758089


namespace max_trig_expression_eq_l758_758240

theorem max_trig_expression_eq (a b c : ℝ) : 
  ∃ θ ∈ Icc (0 : ℝ) (2 * Real.pi), (a * Real.sin θ + b * Real.cos (2 * θ) + c * Real.sin (2 * θ)) = Real.sqrt (a^2 + b^2 + c^2) :=
begin
  sorry
end

end max_trig_expression_eq_l758_758240


namespace greatest_x_inequality_l758_758029

theorem greatest_x_inequality (x : ℝ):
  x^2 - 12 * x + 32 ≤ 0 → x ≤ 8 :=
begin
  sorry
end

end greatest_x_inequality_l758_758029


namespace smallest_number_with_exactly_eight_factors_l758_758827

theorem smallest_number_with_exactly_eight_factors
    (n : ℕ)
    (h1 : ∃ a b : ℕ, (a + 1) * (b + 1) = 8)
    (h2 : ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ n = p^a * q^b) : 
    n = 24 := by
  sorry

end smallest_number_with_exactly_eight_factors_l758_758827


namespace percentage_spent_on_food_l758_758886

theorem percentage_spent_on_food 
  (initial_money : ℝ)
  (remaining_money : ℝ)
  (food_percentage : ℝ) 
  (phone_bill_percentage : ℝ) 
  (entertainment_expense : ℝ) : 
  initial_money = 200 → 
  remaining_money = 40 → 
  phone_bill_percentage = 0.25 → 
  entertainment_expense = 20 → 
  (initial_money - (food_percentage / 100) * initial_money) - 
  phone_bill_percentage * (initial_money - (food_percentage / 100) * initial_money) - 
  entertainment_expense = remaining_money → 
  food_percentage = 51.43 :=
begin
  intros,
  sorry
end

end percentage_spent_on_food_l758_758886


namespace focus_of_parabola_l758_758026

theorem focus_of_parabola (x f d : ℝ) (h : ∀ x, y = 4 * x^2 → (x = 0 ∧ y = f) → PF^2 = PQ^2 ∧ 
(PF^2 = x^2 + (4 * x^2 - f) ^ 2) := (PQ^2 = (4 * x^2 - d) ^ 2)) : 
  f = 1 / 16 := 
sorry

end focus_of_parabola_l758_758026


namespace necessary_not_sufficient_condition_l758_758486

variable {x : ℝ}

theorem necessary_not_sufficient_condition (h : x > 2) : x > 1 :=
by
  sorry

end necessary_not_sufficient_condition_l758_758486


namespace num_divisible_digits_l758_758052

def divisible_by_n (num : ℕ) (n : ℕ) : Prop :=
  n ≠ 0 ∧ num % n = 0

def count_divisible_digits : ℕ :=
  (List.filter (λ n => divisible_by_n (150 + n) n)
    [1, 2, 3, 4, 5, 6, 7, 8, 9]).length

theorem num_divisible_digits : count_divisible_digits = 7 := by
  sorry

end num_divisible_digits_l758_758052


namespace dampening_factor_l758_758864

theorem dampening_factor (s r : ℝ) 
  (h1 : s / (1 - r) = 16) 
  (h2 : s * r / (1 - r^2) = -6) :
  r = -3 / 11 := 
sorry

end dampening_factor_l758_758864


namespace geom_seq_arith_seq_l758_758213

theorem geom_seq_arith_seq 
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_arith : 2 * (1/2 * a 5) = 3 * a 1 + 2 * a 3) :
  (a 9 + a 10) / (a 7 + a 8) = 3 :=
by
  have h1 : a 5 = a 1 * q ^ 4 := h_geom 4
  have h2 : a 3 = a 1 * q ^ 2 := h_geom 2
  have h3 : 2 * (1/2 * a 1 * q ^ 4) = 3 * a 1 + 2 * a 1 * q ^ 2 := by rw [h1, h2]
  have h4 : q ^4 = 3 + 2 * q ^ 2 := by linarith
  have h5 : q ^ 2 = 3 := sorry -- solving the quadratic equation
  have h6 : a 9 = a 1 * q ^ 8 := h_geom 8
  have h7 : a 10 = a 1 * q ^ 9 := h_geom 9
  have h8 : a 7 = a 1 * q ^ 6 := h_geom 6
  have h9 : a 8 = a 1 * q ^ 7 := h_geom 7
  have h10 : (a 1 * q ^ 8 + a 1 * q ^ 9) / (a 1 * q ^ 6 + a 1 * q ^ 7) = q ^ 2 := sorry -- simplifying
  rw h5 at h10
  exact h10

end geom_seq_arith_seq_l758_758213


namespace cos_diff_proof_l758_758140

theorem cos_diff_proof (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2)
  (h2 : Real.cos A + Real.cos B = 1) :
  Real.cos (A - B) = 5 / 8 := 
by
  sorry

end cos_diff_proof_l758_758140


namespace winning_candidate_percentage_is_57_l758_758854

def candidate_votes : List ℕ := [1136, 7636, 11628]

def total_votes : ℕ := candidate_votes.sum

def winning_votes : ℕ := candidate_votes.maximum?.getD 0

def winning_percentage (votes : ℕ) (total : ℕ) : ℚ :=
  (votes * 100) / total

theorem winning_candidate_percentage_is_57 :
  winning_percentage winning_votes total_votes = 57 := by
  sorry

end winning_candidate_percentage_is_57_l758_758854


namespace necessary_but_not_sufficient_not_sufficient_x2_gt_y2_iff_x_lt_y_lt_0_l758_758082

variable (x y : ℝ)

theorem necessary_but_not_sufficient (hx : x < y ∧ y < 0) : x^2 > y^2 :=
sorry

theorem not_sufficient (hx : x^2 > y^2) : ¬ (x < y ∧ y < 0) :=
sorry

-- Optional: Combining the two to create a combined theorem statement
theorem x2_gt_y2_iff_x_lt_y_lt_0 : (∀ x y : ℝ, x < y ∧ y < 0 → x^2 > y^2) ∧ (∃ x y : ℝ, x^2 > y^2 ∧ ¬ (x < y ∧ y < 0)) :=
sorry

end necessary_but_not_sufficient_not_sufficient_x2_gt_y2_iff_x_lt_y_lt_0_l758_758082


namespace impossible_divide_into_three_similar_l758_758671

noncomputable def sqrt2 : ℝ := Real.sqrt 2

def similar (x y : ℝ) : Prop :=
  x ≤ sqrt2 * y

theorem impossible_divide_into_three_similar (N : ℝ) :
  ¬ ∃ (x y z : ℝ), x + y + z = N ∧ similar x y ∧ similar y z ∧ similar x z := 
by
  sorry

end impossible_divide_into_three_similar_l758_758671


namespace cooperation_to_lose_l758_758410

structure Board :=
  (size : ℕ)
  (paintedSquares : fin size × fin size → Prop)

inductive Player
| Andy | Bess | Charley | Dick

inductive Rectangle
| two_by_one (x y : fin 1000)
| one_by_two (x y : fin 1000)
| one_by_three (x y : fin 1000)
| three_by_one (x y : fin 1000)

def can_paint (b : Board) (r : Rectangle) : Prop :=
  match r with
  | Rectangle.two_by_one (x, y)   => ¬ b.paintedSquares (x,y) ∧ ¬ b.paintedSquares (x+1,y)
  | Rectangle.one_by_two (x, y)   => ¬ b.paintedSquares (x,y) ∧ ¬ b.paintedSquares (x,y+1)
  | Rectangle.one_by_three (x, y) => ¬ b.paintedSquares (x,y) ∧ ¬ b.paintedSquares (x,y+1) ∧ ¬ b.paintedSquares (x,y+2)
  | Rectangle.three_by_one (x, y) => ¬ b.paintedSquares (x,y) ∧ ¬ b.paintedSquares (x+1,y) ∧ ¬ b.paintedSquares (x+2,y)

def move (p : Player) (r : Rectangle) (b : Board) : Board :=
  { b with paintedSquares := λ coord, b.paintedSquares coord ∨ match r with
    | Rectangle.two_by_one (x,y)   => coord = (x,y) ∨ coord = (x+1,y)
    | Rectangle.one_by_two (x,y)   => coord = (x,y) ∨ coord = (x,y+1)
    | Rectangle.one_by_three (x,y) => coord = (x,y) ∨ coord = (x,y+1) ∨ coord = (x,y+2)
    | Rectangle.three_by_one (x,y) => coord = (x,y) ∨ coord = (x+1,y) ∨ coord = (x+2,y)
  }

theorem cooperation_to_lose : ∃ (strategy : list (Player × Rectangle → Board → Board)),
  ∀ (b : Board) (rC : Rectangle),
    can_paint b rC →
    (move Player.Charley rC b ∉ strategy) →
    ∃ (pABDs : Player → Board → Prop),
      pABDs Player.Andy b ∧
      pABDs Player.Bess b ∧
      pABDs Player.Dick b ∧
      ¬ pABDs Player.Charley b :=
sorry

end cooperation_to_lose_l758_758410


namespace spending_Mar_Apr_May_l758_758300

-- Define the expenditures at given points
def e_Feb : ℝ := 0.7
def e_Mar : ℝ := 1.2
def e_May : ℝ := 4.4

-- Define the amount spent from March to May
def amount_spent_Mar_Apr_May := e_May - e_Feb

-- The main theorem to prove
theorem spending_Mar_Apr_May : amount_spent_Mar_Apr_May = 3.7 := by
  sorry

end spending_Mar_Apr_May_l758_758300


namespace solve_for_n_l758_758151

theorem solve_for_n (n : ℕ) :
  (∀ r k : ℕ, r = 1 ∧ k = 2 
    → (abs ((2 * (2 * n + 1)) / (4 * (2 * n + 1) * n)) = 1/8))
  → n = 4 :=
by sorry

end solve_for_n_l758_758151


namespace distance_center_to_plane_l758_758597

noncomputable def sphere_distance_to_plane : ℝ :=
let V := 4 * Real.sqrt 3 * Real.pi in
let R := Real.sqrt 3 in
let AB := 1 in
let BC := Real.sqrt 2 in
let theta := (Real.sqrt 3 / 3) * Real.pi / R in
let AC := R * theta in
Real.sqrt (R^2 - (Real.sqrt 3 / 2)^2)

theorem distance_center_to_plane (V : ℝ) (R : ℝ) (AB : ℝ) (BC : ℝ) (theta : ℝ) (d : ℝ) :
  V = 4 * Real.sqrt 3 * Real.pi →
  R = Real.sqrt 3 →
  AB = 1 →
  BC = Real.sqrt 2 →
  theta = (Real.sqrt 3 / 3) * Real.pi / R →
  d = Real.sqrt (R^2 - (Real.sqrt 3 / 2)^2) →
  d = 3/2 :=
by
  intros
  sorry

end distance_center_to_plane_l758_758597


namespace log_evaluation_l758_758932

theorem log_evaluation
  (y : ℝ)
  (log_def : 8 ^ y = 50)
  : y = (1 + 2 * log (50) / log(2)) / 3 :=
by
  sorry

end log_evaluation_l758_758932


namespace jacket_cost_l758_758738

noncomputable def cost_of_shorts : ℝ := 13.99
noncomputable def cost_of_shirt : ℝ := 12.14
noncomputable def total_spent : ℝ := 33.56
noncomputable def cost_of_jacket : ℝ := total_spent - (cost_of_shorts + cost_of_shirt)

theorem jacket_cost : cost_of_jacket = 7.43 := by
  sorry

end jacket_cost_l758_758738


namespace smallest_integer_prime_l758_758353

theorem smallest_integer_prime (x : ℤ) :
  (|10 * x^2 - 61 * x + 21|.prime) →
  (x = 3) :=
by
  sorry

end smallest_integer_prime_l758_758353


namespace sam_reads_72_pages_in_an_hour_l758_758441

variable (pages_per_hour_dustin : ℕ)
variable (pages_more_than_sam : ℕ)
variable (pages_per_hour_sam : ℕ)

-- Conditions
def condition_1 : Prop := pages_per_hour_dustin = 75
def condition_2 (pages_per_40m_sam : ℕ) : Prop := 
  (2/3 * pages_per_hour_dustin - pages_more_than_sam = pages_per_40m_sam)

-- Question: Prove that Sam can read 72 pages in an hour
theorem sam_reads_72_pages_in_an_hour (h1 : condition_1) (h2 : ∃ pages_per_40m_sam, condition_2 pages_per_40m_sam) :
  pages_per_hour_sam = 72 :=
begin
  sorry
end

end sam_reads_72_pages_in_an_hour_l758_758441


namespace inequality_solution_range_l758_758960

theorem inequality_solution_range (k : ℝ) :
  (∀ (x : ℤ), x > 0 → (1 / 2)^(k * x - 1) < (1 / 2)^(5 * x - 2) → false) ↔ k ≤ 4 :=
by
  sorry

end inequality_solution_range_l758_758960


namespace minLengthFG_l758_758210

noncomputable def EF : ℕ := 6
noncomputable def EG : ℕ := 15
noncomputable def HG : ℕ := 10
noncomputable def HF : ℕ := 25

theorem minLengthFG : ∃ FG : ℕ, (FG > (EG - EF)) ∧ (FG > (HF - HG)) ∧ FG = 15 := 
by
  use 15
  split
  · show 15 > 15 - 6
    sorry
  split
  · show 15 > 25 - 10
    sorry
  · show 15 = 15
    rfl

end minLengthFG_l758_758210


namespace num_divisible_digits_l758_758050

def divisible_by_n (num : ℕ) (n : ℕ) : Prop :=
  n ≠ 0 ∧ num % n = 0

def count_divisible_digits : ℕ :=
  (List.filter (λ n => divisible_by_n (150 + n) n)
    [1, 2, 3, 4, 5, 6, 7, 8, 9]).length

theorem num_divisible_digits : count_divisible_digits = 7 := by
  sorry

end num_divisible_digits_l758_758050


namespace smallest_number_with_exactly_eight_factors_l758_758829

theorem smallest_number_with_exactly_eight_factors
    (n : ℕ)
    (h1 : ∃ a b : ℕ, (a + 1) * (b + 1) = 8)
    (h2 : ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ n = p^a * q^b) : 
    n = 24 := by
  sorry

end smallest_number_with_exactly_eight_factors_l758_758829


namespace winnie_lollipops_remainder_l758_758361

theorem winnie_lollipops_remainder :
  ∃ (k : ℕ), k = 505 % 14 ∧ k = 1 :=
by
  sorry

end winnie_lollipops_remainder_l758_758361


namespace martha_total_clothes_l758_758712

def jackets_purchased : ℕ := 4
def tshirts_purchased : ℕ := 9
def jackets_free : ℕ := jackets_purchased / 2
def tshirts_free : ℕ := tshirts_purchased / 3
def total_jackets : ℕ := jackets_purchased + jackets_free
def total_tshirts : ℕ := tshirts_purchased + tshirts_free

theorem martha_total_clothes : total_jackets + total_tshirts = 18 := by
  sorry

end martha_total_clothes_l758_758712


namespace lyle_percentage_l758_758268

theorem lyle_percentage (chips : ℕ) (ian_ratio lyle_ratio : ℕ) (h_ratio_sum : ian_ratio + lyle_ratio = 10) (h_chips : chips = 100) :
  (lyle_ratio / (ian_ratio + lyle_ratio) : ℚ) * 100 = 60 := 
by
  sorry

end lyle_percentage_l758_758268


namespace procedure_cost_l758_758794

variable (C : ℝ)
variable (months : ℕ := 24)
variable (monthly_cost : ℝ := 20)
variable (insurance_coverage : ℝ := 0.80)
variable (saved_amount : ℝ := 3520)

theorem procedure_cost :
  insurance_coverage * C = saved_amount →
  C = 4400 :=
by
  intro h
  calc
    C = (saved_amount / insurance_coverage) : sorry
    ... = 4400 : sorry

end procedure_cost_l758_758794


namespace sin_theta_value_l758_758184

theorem sin_theta_value (θ : ℝ) (h₁ : θ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) 
  (h₂ : Real.sin (2 * θ) = (3 * Real.sqrt 7) / 8) : Real.sin θ = 3 / 4 :=
sorry

end sin_theta_value_l758_758184


namespace find_value_of_pow_function_l758_758091

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem find_value_of_pow_function :
  (∃ α : ℝ, power_function α 4 = 1/2) →
  ∃ α : ℝ, power_function α (1/4) = 2 :=
by
  sorry

end find_value_of_pow_function_l758_758091


namespace chocolate_cost_l758_758860

theorem chocolate_cost (cost_per_box : ℕ) (candies_per_box : ℕ) (total_candies : ℕ) :
  cost_per_box = 9 → candies_per_box = 30 → total_candies = 450 → 
  (total_candies / candies_per_box) * cost_per_box = 135 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end chocolate_cost_l758_758860


namespace domino_chain_possibility_l758_758613

theorem domino_chain_possibility : 
  ¬ ∃ (chain : List (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ chain → p.1 ≤ 6 ∧ p.2 ≤ 6) ∧ 
    (chain.length = 28) ∧ 
    (count_occurrences chain 6 = 7) ∧ 
    (count_occurrences chain 5 = 7) ∧ 
    (valid_chain chain) ∧ 
    (chain.head.1 = 6 ∨ chain.head.2 = 6) ∧ 
    (chain.last.1 = 5 ∨ chain.last.2 = 5)
    :=
by 
  sorry

def count_occurrences (chain : List (ℕ × ℕ)) (n : ℕ) : ℕ := 
  chain.foldr (λ (p : ℕ × ℕ) (acc : ℕ), acc + if p.1 = n ∨ p.2 = n then 1 else 0) 0

def valid_chain (chain : List (ℕ × ℕ)) : Prop := 
  ∀ (i : ℕ), i < chain.length - 1 → (chain.get! i).2 = (chain.get! (i + 1)).1

end domino_chain_possibility_l758_758613


namespace intersection_sets_l758_758530

open Set

def A := {x : ℤ | abs x < 3}
def B := {x : ℤ | abs x > 1}

theorem intersection_sets :
  A ∩ B = {-2, 2} := by
  sorry

end intersection_sets_l758_758530


namespace factorization_correct_l758_758446

noncomputable def factorize_diff_of_squares (a b : ℝ) : ℝ :=
  36 * a * a - 4 * b * b

theorem factorization_correct (a b : ℝ) : factorize_diff_of_squares a b = 4 * (3 * a + b) * (3 * a - b) :=
by
  sorry

end factorization_correct_l758_758446


namespace least_positive_integer_with_eight_factors_l758_758811

theorem least_positive_integer_with_eight_factors :
  ∃ n : ℕ, n = 24 ∧ (8 = (nat.factors n).length) :=
sorry

end least_positive_integer_with_eight_factors_l758_758811


namespace determine_g_l758_758741

-- Definitions based on conditions from the problem
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ

-- The theorem statement representing the proof problem
theorem determine_g (h : ∀ x : ℝ, f (g x) = 9 * x^2 - 6 * x + 1) :
  (∀ x : ℝ, g x = 3 * x - 1) ∨ (∀ x : ℝ, g x = -3 * x + 1) :=
sorry

end determine_g_l758_758741


namespace angle_c_b_l758_758585

section

variables (a b c : EuclideanVector)
variables (angle_ab : Real)
variables (angle_ca : Real)

-- Condition: The angle between vector a and vector b is 60 degrees
axiom h1 : angle_ab = 60

-- Condition: The vector c is perpendicular to vector a, hence the angle between them is 90 degrees
axiom h2 : angle_ca = 90 /- degrees -/

-- Define the angle between vector c and vector b
noncomputable def angle_cb (angle_ab angle_ca : Real) : Real :=
  angle_ca - angle_ab

-- Theorem: The angle between vector c and vector b is 30 degrees
theorem angle_c_b (h1 : angle_ab = 60) (h2 : angle_ca = 90) : angle_cb angle_ab angle_ca = 30 :=
by
  unfold angle_cb
  rw [h1, h2]
  norm_num
  sorry

end

end angle_c_b_l758_758585


namespace annual_interest_rate_Paul_annual_interest_rate_Emma_annual_interest_rate_Harry_l758_758224

-- Define principal amounts for Paul, Emma and Harry
def principalPaul : ℚ := 5000
def principalEmma : ℚ := 3000
def principalHarry : ℚ := 7000

-- Define time periods for Paul, Emma and Harry
def timePaul : ℚ := 2
def timeEmma : ℚ := 4
def timeHarry : ℚ := 3

-- Define interests received from Paul, Emma and Harry
def interestPaul : ℚ := 2200
def interestEmma : ℚ := 3400
def interestHarry : ℚ := 3900

-- Define the simple interest formula 
def simpleInterest (P : ℚ) (R : ℚ) (T : ℚ) : ℚ := P * R * T

-- Prove the annual interest rates for each loan 
theorem annual_interest_rate_Paul : 
  ∃ (R : ℚ), simpleInterest principalPaul R timePaul = interestPaul ∧ R = 0.22 := 
by
  sorry

theorem annual_interest_rate_Emma : 
  ∃ (R : ℚ), simpleInterest principalEmma R timeEmma = interestEmma ∧ R = 0.2833 := 
by
  sorry

theorem annual_interest_rate_Harry : 
  ∃ (R : ℚ), simpleInterest principalHarry R timeHarry = interestHarry ∧ R = 0.1857 := 
by
  sorry

end annual_interest_rate_Paul_annual_interest_rate_Emma_annual_interest_rate_Harry_l758_758224


namespace pentagon_area_min_pentagon_area_l758_758372

variables (a b : ℝ)
variables (AB BC a b : ℝ) (h1 : 0 < a) (h2 : a < b)

-- Problem (1)
theorem pentagon_area (AB BC : ℝ) (h : AB = a) (h' : BC = b) (h1 : a < b) :
  (∃ (O E F G : ℝ), 
    let EF := some (line_through O ∧ E ∈ [BC] ∧ F ∈ [DA]) in
    let EF_fold := fold ECDF along EF in
    ∃ G, coincide G A (EF_fold),
    area (ABEFG) = a * (3 * b ^ 2 - a ^ 2) / (4 * b)) :=
sorry

-- Problem (2)
theorem min_pentagon_area (b : ℕ) (hb : b > 1) :
  let S := 1 * (3 * b ^ 2 - 1) / (4 * b) in
  min_area (λ b, S) ≥ (11 / 8) :=
sorry

end pentagon_area_min_pentagon_area_l758_758372


namespace probability_C_l758_758885

variable (P : Type) [Field P]

-- Define the probabilities of A, B, and D as given conditions
def P_A : P := 2 / 5
def P_B : P := 1 / 4
def P_D : P := 1 / 5

-- The theorem stating the probability of C
theorem probability_C : 1 - P_A - P_B - P_D = 3 / 20 := by
  sorry

end probability_C_l758_758885


namespace additional_track_length_l758_758402

/-
A Lean 4 statement to prove that the additional length of track needed
to decrease the slope from a 4% grade to a 1% grade, given a vertical climb of 800 feet, 
is 60000 feet.
-/
theorem additional_track_length (climb : ℝ) (grade1 : ℝ) (grade2 : ℝ) (h_climb : climb = 800) 
  (h_grade1 : grade1 = 0.04) (h_grade2 : grade2 = 0.01) : 
  let length1 := climb / grade1 in 
  let length2 := climb / grade2 in 
  length2 - length1 = 60000 := 
by 
  have h_length1 : length1 = 20000 := by sorry
  have h_length2 : length2 = 80000 := by sorry
  calc
    length2 - length1 = 80000 - 20000 := by rw [h_length1, h_length2]
                ... = 60000 := by norm_num

end additional_track_length_l758_758402


namespace rabbit_probability_l758_758197

theorem rabbit_probability (rabbits : Finset ℕ) (measured_rabbits : Finset ℕ) (h_rabbits_card : rabbits.card = 5) (h_measured_card : measured_rabbits.card = 3) :
  ∃ (selected : Finset (Finset ℕ)), ∃ (probability : ℚ),
  (∀ (sel ∈ selected), sel.card = 3) ∧
  (∃ (favorable : Finset (Finset ℕ)), (∀ (fav ∈ favorable, ∃ (measured_count : ℕ), ∃ (unmeasured_count : ℕ), 
    fav.card = 3 ∧ (measured_count + unmeasured_count = 3) ∧ measured_count = 2 ∧ unmeasured_count = 1))) ∧
  probability = (favorably.card : ℚ) / (selected.card : ℚ) ∧
  probability = 3 / 5 :=
sorry

end rabbit_probability_l758_758197


namespace points_D_D_E_E_concyclic_and_center_A_l758_758983

variables {A B C H P D D' E E' : Point}
variables {circleABH circleACH : Circle}

-- Given conditions
axiom acute_angle_triangle_ABC : ∀ (A B C : Point), AcuteTriangle A B C
axiom altitude_AH : Perpendicular A H C
axiom point_P_on_altitude : OnLine A H P
axiom perp_P_to_AB_meet_circumcircle_ABH : Perpendicular P (line AB) (CircleMeet (circumcircle ABH) P)
axiom perp_P_to_AC_meet_circumcircle_ACH : Perpendicular P (line AC) (CircleMeet (circumcircle ACH) P)

-- To Prove
theorem points_D_D_E_E_concyclic_and_center_A :
  Concyclic D D' E E' ∧ Center (Circumcircle D D' E E') = A :=
sorry

end points_D_D_E_E_concyclic_and_center_A_l758_758983


namespace interval_f_has_two_roots_l758_758545

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2 + a * x

theorem interval_f_has_two_roots (a : ℝ) : (∀ x : ℝ, f a x = 0 → ∃ u v : ℝ, u ≠ v ∧ f a u = 0 ∧ f a v = 0) ↔ 0 < a ∧ a < 1 / 8 := 
sorry

end interval_f_has_two_roots_l758_758545


namespace problem_conditions_l758_758163

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 * Real.log x

theorem problem_conditions 
  (h_tangent : tangent_line (f) (1, f 1) = λ x, 2 * x - 2)
  (h_interval : ∀ x ∈ Icc (1/Real.exp 1) Real.exp 1, True) :
  (f (Real.exp 1) = 2 * (Real.exp 1)^2) ∧ 
  (f (1/Real.exp 1) = -2 / (Real.exp 1)^2) ∧ 
  (f (Real.exp (-1/2)) = -1 / Real.exp 1) ∧ 
  (∀ x ∈ (set.Ioc (1 / Real.exp 1) (Real.exp (-1 / 2))), deriv f x < 0) ∧
  (∀ x ∈ (set.Ioc (Real.exp (-1 / 2)) Real.exp 1), deriv f x > 0) :=
begin
  sorry
end

end problem_conditions_l758_758163


namespace find_x_l758_758400

theorem find_x (x : ℕ) (h : x * 5^4 = 75625) : x = 121 :=
by
  sorry

end find_x_l758_758400


namespace max_subsets_cardinality_l758_758428

theorem max_subsets_cardinality (A : set ℕ)
  (hA : A = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (B : ℕ → set ℕ)
  (h_nonempty : ∀ i, (B i) ≠ ∅)
  (h_intersect : ∀ i j, i ≠ j → (B i ∩ B j).card ≤ 2) :
  ∃ k, k = 175 :=
begin
  use 175,
  sorry
end

end max_subsets_cardinality_l758_758428


namespace distinct_values_of_z_l758_758542

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def reverse_digits (x : ℕ) : ℕ :=
  let d1 := x % 10
  let d2 := (x / 10) % 10
  let d3 := (x / 100) % 10
  let d4 := x / 1000
  1000 * d1 + 100 * d2 + 10 * d3 + d4

theorem distinct_values_of_z :
  ∃ (distinct_z_count : ℕ), 
  (∀ x y z : ℕ,
    is_four_digit x → 
    is_four_digit y → 
    y = reverse_digits x → 
    z = abs (x - y) → 
    true) →
  true :=
by
  sorry

end distinct_values_of_z_l758_758542


namespace algebra_expr_eval_l758_758487

theorem algebra_expr_eval {x y : ℝ} (h : x - 2 * y = 3) : 5 - 2 * x + 4 * y = -1 :=
by sorry

end algebra_expr_eval_l758_758487


namespace proof_angles_constant_l758_758176

noncomputable theory
open_locale classical

def const_sum_angles (O1 O2 : Circle) (A A' : Point) (B C D E : Point) : Prop :=
(∃ (line : Line), is_on_line B line ∧ is_on_line C line ∧ is_on_circle B O1 ∧ is_on_circle C O1 ∧
 is_on_line D line ∧ is_on_line E line ∧ is_on_circle D O2 ∧ is_on_circle E O2 ∧
 (is_collinear C D B E ∨ is_collinear B E C D) → (angle B A D + angle C A E = const))

def const_abs_diff_angles (O1 O2 : Circle) (A A' : Point) (B C D E : Point) : Prop :=
(∃ (line : Line), is_on_line B line ∧ is_on_line C line ∧ is_on_circle B O1 ∧ is_on_circle C O1 ∧
 is_on_line D line ∧ is_on_line E line ∧ is_on_circle D O2 ∧ is_on_circle E O2 ∧
 (¬is_collinear C D B E ∧ ¬is_collinear B E C D) → (|angle B A D - angle C A E| = const))

theorem proof_angles_constant (O1 O2 : Circle) (A A' : Point) : 
  ∀ B C D E, const_sum_angles O1 O2 A A' B C D E ∨ const_abs_diff_angles O1 O2 A A' B C D E :=
by sorry

end proof_angles_constant_l758_758176


namespace equilateral_triangle_min_perimeter_l758_758275

theorem equilateral_triangle_min_perimeter (a b c : ℝ) (S : ℝ) :
  let p := (a + b + c) / 2 in
  let area := sqrt (p * (p - a) * (p - b) * (p - c)) in
  area = S →
  ∀ a b c, p = (a + b + c) / 2 →
  sqrt (p * (p - a) * (p - b) * (p - c)) = S →
  a = b = c :=
by sorry

end equilateral_triangle_min_perimeter_l758_758275


namespace trent_total_distance_l758_758337

theorem trent_total_distance
  (house_to_bus : ℕ)
  (bus_to_library : ℕ)
  (house_to_bus = 4)
  (bus_to_library = 7)
  : (house_to_bus + bus_to_library) * 2 = 22 :=
by
  sorry

end trent_total_distance_l758_758337


namespace ratio_of_quadrilateral_area_l758_758603

noncomputable def ratio_of_areas (AB CD KLMN ABCD : ℝ) := KLMN / ABCD

theorem ratio_of_quadrilateral_area {AB AD K L M N : ℝ}
  (h1 : ∠BAD = 60)
  (h2 : AB = 2)
  (h3 : AD = 5)
  (h4 : is_bisector_intersection_angle BAD ABC K)
  (h5 : is_bisector_intersection_angle BAD CDA L)
  (h6 : is_bisector_intersection_angle BCD CDA M)
  (h7 : is_bisector_intersection_angle BCD ABC N) :
  ratio_of_areas (area_of_quadrilateral K L M N) (area_of_parallelogram ABCD) = 9 / 20 :=
sorry

end ratio_of_quadrilateral_area_l758_758603


namespace total_charge_for_3_hours_l758_758363

namespace TherapyCharges

-- Conditions
variables (A F : ℝ)
variable (h1 : F = A + 20)
variable (h2 : F + 4 * A = 300)

-- Prove that the total charge for 3 hours of therapy is 188
theorem total_charge_for_3_hours : F + 2 * A = 188 :=
by
  sorry

end TherapyCharges

end total_charge_for_3_hours_l758_758363


namespace updated_mean_is_correct_l758_758367

-- define the initial mean and the number of observations
def initial_mean : ℝ := 200
def num_observations : ℕ := 50

-- define the decrement per observation
def decrement : ℝ := 34

-- calculate the initial total sum
def initial_total_sum : ℝ := initial_mean * num_observations

-- calculate the total decrement
def total_decrement : ℝ := decrement * num_observations

-- calculate the updated total sum
def updated_total_sum : ℝ := initial_total_sum - total_decrement

-- calculate the updated mean
def updated_mean : ℝ := updated_total_sum / num_observations

-- the statement to prove
theorem updated_mean_is_correct : updated_mean = 166 := by
  -- proof goes here
  sorry

end updated_mean_is_correct_l758_758367


namespace _l758_758132

noncomputable def hyperbola_eccentricity_theorem {F1 F2 P : Point} 
  (hyp : is_hyperbola F1 F2 P)
  (angle_F1PF2 : ∠F1 P F2 = 60)
  (dist_PF1_3PF2 : dist P F1 = 3 * dist P F2) : 
  eccentricity F1 F2 = sqrt 7 / 2 :=
by 
  sorry

end _l758_758132


namespace probability_vowel_initials_l758_758196

theorem probability_vowel_initials :
  let total_students := 30
  let vowels := ['A', 'E', 'I', 'O', 'U']
  let consonants := ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'] 
  let students_with_vowel_initials := 5 * 2
  let probability := students_with_vowel_initials / total_students 
  probability = 1 / 3 :=
by {
  let total_students := 30
  let students_with_vowel_initials := 5 * 2
  let probability := students_with_vowel_initials / total_students
  show probability = 1 / 3
}

end probability_vowel_initials_l758_758196


namespace focus_of_parabola_y_eq_4x_sq_l758_758013

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := (0 : ℝ, 1 / 16 : ℝ)
  in f

theorem focus_of_parabola_y_eq_4x_sq :
  (0, 1 / 16) = parabola_focus := by
  unfold parabola_focus
  sorry

end focus_of_parabola_y_eq_4x_sq_l758_758013


namespace solve_for_s_l758_758921

theorem solve_for_s (s : ℝ) (h : 8 = 2^(3 * s + 2)) : s = 1 / 3 := by
  sorry

end solve_for_s_l758_758921


namespace problem1_problem2_l758_758532

open Real

/-- Given conditions and hypothesis. -/
variables (S_n a_n : ℕ → ℝ)
variables (vec_a vec_b : ℕ → ℝ × ℝ)

-- Definitions based on given conditions
def vec_a (n : ℕ) : ℝ × ℝ := (S_n n, 1)
def vec_b (n : ℕ) : ℝ × ℝ := (-1, 2 * a_n n + 2^(n + 1))
def orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Given conditions
axiom h1 : ∀ n, orthogonal (vec_a n) (vec_b n)

-- Problem 1: Prove {a_n / 2^n} is an arithmetic sequence
theorem problem1 : ∃ d : ℝ, ∀ n, a_n (n + 1) / 2^(n + 1) = a_n n / 2^n + d :=
sorry

-- Problem 2: Existence of n_0 such that for any k(k∈ℕ*),  bk ≤ b_{n_0}
def b_n (n : ℕ) : ℝ := (n - 2011) / (n + 1) * a_n n
theorem problem2 : ∃ n_0, ∀ k : ℕ, 0 < k → b_n k ≤ b_n n_0 :=
sorry

end problem1_problem2_l758_758532


namespace area_of_abs_x_plus_abs_3y_eq_12_l758_758807

theorem area_of_abs_x_plus_abs_3y_eq_12 :
  (set_integral (λ x y, if |x| + |3 * y| <= 12 then 1 else 0) * 4) = 96 := 
  sorry

end area_of_abs_x_plus_abs_3y_eq_12_l758_758807


namespace odd_divisor_probability_l758_758308

/-! # Problem
Given the factorial of 25, which has numerous positive integer divisors, prove that the probability of randomly selecting an odd divisor is 1/23.
-/

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem odd_divisor_probability :
  let n := 25!
  let total_divisors := (22 + 1) * (3 + 1) * (2 + 1) * (3 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  let odd_divisors := (3 + 1) * (2 + 1) * (3 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  total_divisors ≠ 0 → 
  (odd_divisors / total_divisors) = 1 / 23 :=
by
  intro n total_divisors odd_divisors h
  sorry

end odd_divisor_probability_l758_758308


namespace hyperbola_eccentricity_l758_758108

open Real

-- Conditions from the problem
variables (F₁ F₂ P : Point) (C : ℝ) (a : ℝ)
variables (angle_F1PF2 : angle F₁ P F₂ = 60)
variables (distance_PF1_PF2 : dist P F₁ = 3 * dist P F₂)
variables (focus_condition : 2 * C = dist F₁ F₂)

-- Statement of the problem
theorem hyperbola_eccentricity:
  let e := C / a in
  e = sqrt 7 / 2 :=
sorry

end hyperbola_eccentricity_l758_758108


namespace max_difference_y_coords_intersection_l758_758427

def f (x : ℝ) : ℝ := 4 - x^2 + x^3
def g (x : ℝ) : ℝ := x^2 + x^4

theorem max_difference_y_coords_intersection : ∀ x : ℝ, 
  (f x = g x) → 
  (∀ x₁ x₂ : ℝ, f x₁ = g x₁ ∧ f x₂ = g x₂ → |f x₁ - f x₂| = 0) := 
by
  sorry

end max_difference_y_coords_intersection_l758_758427


namespace smallest_constant_M_l758_758952

theorem smallest_constant_M :
  ∃ M : ℝ, (∀ x y z w : ℝ, 0 < x → 0 < y → 0 < z → 0 < w →
  (real.sqrt (x / (y + z + w)) + real.sqrt (y / (x + z + w)) + real.sqrt (z / (x + y + w)) + real.sqrt (w / (x + y + z)) < M)) ∧
  M = 4 / real.sqrt 3 :=
sorry

end smallest_constant_M_l758_758952


namespace ratio_proof_l758_758848

theorem ratio_proof (a b c d e : ℕ) (h1 : a * 4 = 3 * b) (h2 : b * 9 = 7 * c)
  (h3 : c * 7 = 5 * d) (h4 : d * 13 = 11 * e) : a * 468 = 165 * e :=
by
  sorry

end ratio_proof_l758_758848


namespace valid_digits_count_l758_758058

theorem valid_digits_count : { n ∈ Finset.range 10 | n > 0 ∧ 15 * n % n = 0 }.card = 5 :=
by
  sorry

end valid_digits_count_l758_758058


namespace minimum_midpoint_plotter_uses_l758_758872

theorem minimum_midpoint_plotter_uses :
  ∃ (n : ℕ), n = 17 ∧ (∃ (a : ℕ), (65 = a) ∧ (a:ℤ) ∈ set.Ioc (⌈(2^17 / 2017 : ℝ)⌉₊) (⌊(2^17 / 2016 : ℝ)⌋₊)) :=
by
  sorry

end minimum_midpoint_plotter_uses_l758_758872


namespace hexagon_area_l758_758350

-- Define the area of a triangle
def triangle_area (base height: ℝ) : ℝ := 0.5 * base * height

-- Given dimensions for each triangle
def base_unit := 1
def original_height := 3
def new_height := 4

-- Calculate areas of each triangle in the new configuration
def single_triangle_area := triangle_area base_unit new_height
def total_triangle_area := 4 * single_triangle_area

-- The area of the rectangular region formed by the hexagon and triangles
def rectangular_region_area := (base_unit + original_height + original_height) * new_height

-- Prove the area of the hexagon
theorem hexagon_area : rectangular_region_area - total_triangle_area = 32 :=
by
  -- We will provide the proof here
  sorry

end hexagon_area_l758_758350


namespace min_partitions_to_connect_rabbits_l758_758601

theorem min_partitions_to_connect_rabbits :
  let grid := fin 10 × fin 10
  let partitions := {p : grid × grid // p.1.1 = p.2.1 ∧ (p.1.2 ≠ p.2.2 ∨ p.1.1 ≠ p.2.1)}
  ∀ (r1 r2 : grid), r1 ≠ r2 → 
  (∃ path : list (grid × grid), 
    path.head = some (r1, r1) ∧ path.last = some (r2, r2) ∧ 
    length path ≤ 17 ∧ 
    ∀ (e ∈ path), e ∈ partitions) →
    length partitions = 100 :=
sorry

end min_partitions_to_connect_rabbits_l758_758601


namespace ten_faucets_fill_50_gallon_in_60_seconds_l758_758038

-- Define the conditions
def five_faucets_fill_tub (faucet_rate : ℝ) : Prop :=
  5 * faucet_rate * 8 = 200

def all_faucets_same_rate (tub_capacity time : ℝ) (num_faucets : ℕ) (faucet_rate : ℝ) : Prop :=
  num_faucets * faucet_rate * time = tub_capacity

-- Define the main theorem to be proven
theorem ten_faucets_fill_50_gallon_in_60_seconds (faucet_rate : ℝ) :
  (∃ faucet_rate, five_faucets_fill_tub faucet_rate) →
  all_faucets_same_rate 50 1 10 faucet_rate →
  10 * faucet_rate * (1 / 60) = 50 :=
by
  sorry

end ten_faucets_fill_50_gallon_in_60_seconds_l758_758038


namespace robyn_should_do_5_tasks_l758_758736

theorem robyn_should_do_5_tasks (robyn_tasks : ℕ) (sasha_tasks : ℕ) (total_tasks : ℕ) :
  robyn_tasks = 4 → sasha_tasks = 14 → total_tasks = robyn_tasks + sasha_tasks → 
  ∃ tasks_to_do_from_sasha : ℕ, tasks_to_do_from_sasha = 5 ∧ (robyn_tasks + tasks_to_do_from_sasha = sasha_tasks - tasks_to_do_from_sasha) :=
begin
  -- The setup and conditions are included, proof is not required.
  sorry
end

end robyn_should_do_5_tasks_l758_758736


namespace part1_part2_l758_758373

-- Define the terms
def sqrt18 : ℝ := Real.sqrt 18
def sqrt9 : ℝ := Real.sqrt 9
def sqrt14 : ℝ := Real.sqrt (1 / 4)
def twosqrt2 : ℝ := 2 * Real.sqrt 2
def sqrt32 : ℝ := Real.sqrt 32

-- Define the expression
def expr := sqrt18 / sqrt9 - sqrt14 * twosqrt2 + sqrt32

-- Define the equation
def equation := ∀ x : ℝ, (x^2 - 2 * x = 3)

-- State the theorems
theorem part1 : expr = 4 * Real.sqrt 2 :=
  sorry

theorem part2 : equation → (∀ x : ℝ, x = 3 ∨ x = -1) :=
  sorry

end part1_part2_l758_758373


namespace ordering_of_powers_l758_758346

theorem ordering_of_powers : 6^10 < 3^20 ∧ 3^20 < 2^30 :=
by {
  have h1 : 6^10 = 3^10 * 2^10 := sorry,
  have h2 : 3^20 = 3^10 * 3^10 := sorry,
  have h3 : 2^30 = (2^10)^3 := sorry,
  have h4 : 3^10 > 2^10 := sorry,
  have h5 : 3^20 > 6^10 := sorry,
  have h6 : 3^{20} < 2^{30} := sorry,
  exact ⟨h5, h6⟩
}

end ordering_of_powers_l758_758346


namespace impossible_to_divide_three_similar_parts_l758_758698

theorem impossible_to_divide_three_similar_parts 
  (n : ℝ) 
  (p : ℝ → Prop) 
  (similar : ℝ → ℝ → Prop) 
  (h_similar : ∀ a b : ℝ, similar a b ↔ a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2) : 
  ¬ ∃ p1 p2 p3 : ℝ, p p1 ∧ p p2 ∧ p p3 ∧ (p1 + p2 + p3 = n) ∧ similar p1 p2 ∧ similar p2 p3 ∧ similar p1 p3 :=
sorry

end impossible_to_divide_three_similar_parts_l758_758698


namespace no_division_into_three_similar_piles_l758_758667

theorem no_division_into_three_similar_piles :
    ∀ (x : ℝ),
    ∀ (y z : ℝ),
    (x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = x) →
    (x <= sqrt 2 * y ∧ y <= sqrt 2 * z ∧ z <= sqrt 2 * x) →
    false :=
by
  intro x y z
  sorry

end no_division_into_three_similar_piles_l758_758667


namespace intersection_sets_l758_758529

open Set

def A := {x : ℤ | abs x < 3}
def B := {x : ℤ | abs x > 1}

theorem intersection_sets :
  A ∩ B = {-2, 2} := by
  sorry

end intersection_sets_l758_758529


namespace find_smallest_positive_z_l758_758740

def smallest_positive_z (x z : ℝ) (k m : ℤ) : Prop :=
  sin x = 0 ∧ cos (x + z) = sqrt 3 / 2 ∧ (x = k * π) ∧ (m = 0) ∧ (z = π / 6)

theorem find_smallest_positive_z : 
  ∃ z > 0, ∀ k m : ℤ, smallest_positive_z 0 z k m :=
by
  sorry

end find_smallest_positive_z_l758_758740


namespace brian_breath_proof_l758_758900

def breath_holding_time (initial_time: ℕ) (week1_factor: ℝ) (week2_factor: ℝ) 
  (missed_days: ℕ) (missed_decrease: ℝ) (week3_factor: ℝ): ℝ := by
  let week1_time := initial_time * week1_factor
  let hypothetical_week2_time := week1_time * (1 + week2_factor)
  let missed_decrease_total := week1_time * missed_decrease * missed_days
  let effective_week2_time := hypothetical_week2_time - missed_decrease_total
  let final_time := effective_week2_time * (1 + week3_factor)
  exact final_time

theorem brian_breath_proof :
  breath_holding_time 10 2 0.75 2 0.1 0.5 = 46.5 := 
by
  sorry

end brian_breath_proof_l758_758900


namespace z_is_1_point_5_decades_younger_than_x_l758_758369

-- Defining the ages of x, y, z, and w as X, Y, Z, and W respectively.
variables (X Y Z W : ℝ) (k : ℝ)

-- Conditions
def condition1 : Prop := X + Y = Y + Z + 15
def condition2 : Prop := X = 3 * Z

-- Question: How many decades younger is z compared to x?
def decades_younger : ℝ := (X - Z) / 10

-- Proof statement that Z is exactly 1.5 decades younger than X given the conditions
theorem z_is_1_point_5_decades_younger_than_x 
  (h1 : condition1 X Y Z)
  (h2 : condition2 X Z) : decades_younger X Z = 1.5 :=
by {
  sorry
}

end z_is_1_point_5_decades_younger_than_x_l758_758369


namespace trajectory_equation_minimum_area_l758_758604

noncomputable section

-- Condition: Coordinates of points E' and F'
def E' : (ℝ × ℝ) := (0, Real.sqrt 3)
def F' : (ℝ × ℝ) := (0, -Real.sqrt 3)

-- Definition of moving point G
def G (x y : ℝ) : Prop :=
  (y - Real.sqrt 3) * (y + Real.sqrt 3) = -3 / 4 * (x * x)

-- Equivalent proof problem for Part 1
theorem trajectory_equation (x y : ℝ) (h : G x y) : 
  (x^2 / 4) + (y^2 / 3) = 1 :=
sorry

-- Equivalent proof problem for Part 2
theorem minimum_area (A B : (ℝ × ℝ))
  (hA : A ∈ {G | ∃ x y, G x y})
  (hB : B ∈ {G | ∃ x y, G x y})
  (h_perpendicular : (A.1 = 0 → B.1 = 0) ∧ (A.2 = 0 → B.2 = 0)) :
  let OA := Real.sqrt (A.1 ^ 2 + A.2 ^ 2);
      OB := Real.sqrt (B.1 ^ 2 + B.2 ^ 2);
      area := 1/2 * OA * OB
  in area = 12 / 7 :=
sorry

end trajectory_equation_minimum_area_l758_758604


namespace num_digits_divisible_l758_758062

theorem num_digits_divisible (h : ∀ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}, divides n (150 + n)) :
  {n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} | divides n (150 + n)}.card = 5 := 
sorry

end num_digits_divisible_l758_758062


namespace harry_weekly_earnings_l758_758568

def dogs_walked_MWF := 7
def dogs_walked_Tue := 12
def dogs_walked_Thu := 9
def pay_per_dog := 5

theorem harry_weekly_earnings : 
  dogs_walked_MWF * pay_per_dog * 3 + dogs_walked_Tue * pay_per_dog + dogs_walked_Thu * pay_per_dog = 210 :=
by
  sorry

end harry_weekly_earnings_l758_758568


namespace number_of_zeros_compare_ln_a_neag2b_l758_758158

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + b * x - log x

theorem number_of_zeros (a b : ℝ) (h₁ : a = 8) (h₂ : b = -6) : ∃ x1 x2 : ℝ, f a b x1 = 0 ∧ f a b x2 = 0 ∧ x1 ≠ x2 :=
  by sorry

theorem compare_ln_a_neag2b (a b : ℝ) (h₁ : 0 < a) (h₂ : f a b 1 = a + b - log 1) (h₃ : ∂/(∂ x) (f a b x) = 0 at 1) : log a < -2 * b :=
  by sorry

end number_of_zeros_compare_ln_a_neag2b_l758_758158


namespace ribbon_total_length_l758_758362

theorem ribbon_total_length (R : ℝ)
  (h_first : R - (1/2)*R = (1/2)*R)
  (h_second : (1/2)*R - (1/3)*((1/2)*R) = (1/3)*R)
  (h_third : (1/3)*R - (1/2)*((1/3)*R) = (1/6)*R)
  (h_remaining : (1/6)*R = 250) :
  R = 1500 :=
sorry

end ribbon_total_length_l758_758362


namespace circle_equation_l758_758146

-- Define the problem conditions
def is_circle_tangent_to_line (center : ℝ × ℝ) (N : ℝ × ℝ) (slope : ℝ) : Prop :=
  ∃ a : ℝ, a = center.1 ∧
  N = (3, -Real.sqrt 3) ∧
  (center = (a, 0) ∧ slope = -Real.sqrt 3 / 3) ∧
  let r := Real.sqrt ((a - 3)^2 + 3) in
  let d := Real.abs ((Real.sqrt 3 / 3) * a) / Real.sqrt ((Real.sqrt 3 / 3)^2 + 1) in
  r = d

-- Prove the equation of the circle
theorem circle_equation
  (a : ℝ) (center : ℝ × ℝ := (a, 0)) (N : ℝ × ℝ := (3, -Real.sqrt 3))
  (slope : ℝ := -Real.sqrt 3 / 3)
  (h : is_circle_tangent_to_line center N slope) :
  (a = 4) → ((fun x y : ℝ => (x - 4)^2 + y^2) = (fun x y : ℝ => 4)) :=
by
  intro ha
  rw ha
  funext
  simp
  sorry

end circle_equation_l758_758146


namespace skew_lines_in_prism_l758_758902

theorem skew_lines_in_prism (prism : Type) [regular_triangular_prism prism]
  (num_lines : ℕ)
  (top_edges : ℕ) (bottom_edges : ℕ) (lateral_edges : ℕ)
  (pairs_skew_top : ℕ) (pairs_skew_bottom : ℕ) (pairs_skew_lateral : ℕ) :
  num_lines = 15 →
  top_edges = 3 →
  bottom_edges = 3 →
  lateral_edges = 6 →
  pairs_skew_top = 3 * 5 →
  pairs_skew_bottom = 3 * 5 →
  pairs_skew_lateral = (3 * 4) / 2 →
  (pairs_skew_top + pairs_skew_bottom + pairs_skew_lateral) = 36 :=
by
  intros
  sorry

end skew_lines_in_prism_l758_758902


namespace product_of_consecutive_numbers_not_100th_power_l758_758282

theorem product_of_consecutive_numbers_not_100th_power (n : ℕ):
  ¬ (∃ k : ℕ, (list.range 100).prod (λ i, n + i) = k ^ 100) :=
by
  sorry

end product_of_consecutive_numbers_not_100th_power_l758_758282


namespace fraction_grades_C_l758_758592

def fraction_grades_A (students : ℕ) : ℕ := (1 / 5) * students
def fraction_grades_B (students : ℕ) : ℕ := (1 / 4) * students
def num_grades_D : ℕ := 5
def total_students : ℕ := 100

theorem fraction_grades_C :
  (total_students - (fraction_grades_A total_students + fraction_grades_B total_students + num_grades_D)) / total_students = 1 / 2 :=
by
  sorry

end fraction_grades_C_l758_758592


namespace smallest_integer_k_l758_758836

theorem smallest_integer_k (k : ℤ) : k > 2 ∧ k % 19 = 2 ∧ k % 7 = 2 ∧ k % 4 = 2 ↔ k = 534 :=
by
  sorry

end smallest_integer_k_l758_758836


namespace bill_amount_each_person_shared_l758_758368

noncomputable def total_bill := 139.00
noncomputable def tip_percentage := 0.10
noncomputable def number_of_people := 9

noncomputable def tip := tip_percentage * total_bill
noncomputable def total_with_tip := total_bill + tip
noncomputable def amount_each_person_pays := total_with_tip / number_of_people

theorem bill_amount_each_person_shared :
  amount_each_person_pays = 16.99 :=
by
  sorry

end bill_amount_each_person_shared_l758_758368


namespace avg_divisible_by_4_l758_758450

theorem avg_divisible_by_4 (N : ℕ) (h₁ : ∀ k, k ∈ finset.Icc 7 N → 4 ∣ k) (h₂ : ( ( finset.Icc 7 N ).filter (λ x, 4 ∣ x)).card > 0 ) (h_avg : ( 2 * 22 = 8 + N) ) : N = 36 :=
by sorry

end avg_divisible_by_4_l758_758450


namespace n_consecutive_even_sum_l758_758253

theorem n_consecutive_even_sum (n k : ℕ) (hn : n > 2) (hk : k > 2) : 
  ∃ (a : ℕ), (n * (n - 1)^(k - 1)) = (2 * a + (2 * a + 2 * (n - 1))) / 2 * n :=
by
  sorry

end n_consecutive_even_sum_l758_758253


namespace evaluate_complex_expression_l758_758934

variable (ω : ℂ)

theorem evaluate_complex_expression
  (h : ω = 7 + 4 * complex.I) :
  complex.abs (ω^2 + 10 * ω + 88) = real.sqrt 313 * 13 :=
by
  sorry

end evaluate_complex_expression_l758_758934


namespace f_neg_five_pi_over_three_l758_758394

noncomputable def f : ℝ → ℝ :=
  sorry  -- The definition of the function will be provided in the proof

theorem f_neg_five_pi_over_three :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ x : ℝ, -π/2 ≤ x ∧ x ≤ 0 → f x = sin x) →
  f (-5 * π / 3) = √3 / 2 :=
sorry

end f_neg_five_pi_over_three_l758_758394


namespace target_heart_rate_l758_758392

/-- Given a cyclist's age, the target heart rate is 85% of the maximum heart rate. 
    The maximum heart rate is found by subtracting the cyclist's age from 230. 
    Prove that the target heart rate of a 26-year-old cyclist is 173 beats per minute. -/
theorem target_heart_rate (age : ℕ) (hr_max : ℕ) (target_hr : ℕ) 
  (h1 : hr_max = 230 - age) 
  (h2 : target_hr = (0.85 * hr_max).round) 
  (h_age : age = 26) :
  target_hr = 173 :=
by
  sorry

end target_heart_rate_l758_758392


namespace dhoni_more_small_monkey_dolls_l758_758438

theorem dhoni_more_small_monkey_dolls
  (spend : ℕ)
  (cost_large : ℕ)
  (diff : ℕ)
  (more_dolls : ℕ) : 
  spend = 300 → 
  cost_large = 6 → 
  diff = 2 →
  more_dolls = (spend / (cost_large - diff)) - (spend / cost_large) →
  more_dolls = 25 :=
by
  intros spend_eq cost_large_eq diff_eq more_dolls_eq
  rw [spend_eq, cost_large_eq, diff_eq, more_dolls_eq]
  norm_num
  -- easier to see that:
  -- 25 = 300 / (6 - 2) - 300 / 6
  -- 25 = 300 / 4 - 50
  -- 25 = 75 - 50
  -- 25 = 25
  sorry

end dhoni_more_small_monkey_dolls_l758_758438


namespace hyperbola_eccentricity_l758_758127

theorem hyperbola_eccentricity (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
    (C : P → Prop) 
    (angle_F1PF2 : ∀ {x y z : P}, ∀ (h : x ∈ C) (h₁ : y ∈ C) (h₂ : z ∈ C), is_angle x y z = 60)
    (dist_PF1_PF2 : ∀ (h : P ∈ C), dist P F1 = 3 * dist P F2) : 
    F1 ∈ C → F2 ∈ C → eccentricity C = (Real.sqrt 7) / 2 :=
by sorry

end hyperbola_eccentricity_l758_758127


namespace minimum_quotient_value_l758_758831

-- Helper definition to represent the quotient 
def quotient (a b c d : ℕ) : ℚ := (1000 * a + 100 * b + 10 * c + d) / (a + b + c + d)

-- Conditions: digits are distinct and non-zero 
def distinct_and_nonzero (a b c d : ℕ) : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem minimum_quotient_value :
  ∀ (a b c d : ℕ), distinct_and_nonzero a b c d → quotient a b c d = 71.9 :=
by sorry

end minimum_quotient_value_l758_758831


namespace log8_50_equals_l758_758929

theorem log8_50_equals : log 8 50 = (1 + 2 * log 2 5) / 3 :=
by
  sorry

end log8_50_equals_l758_758929


namespace complex_roots_in_annulus_l758_758284

noncomputable def is_root (p : ℂ → ℂ) (z : ℂ) : Prop := p z = 0

def annulus (z : ℂ) (r1 r2 : ℝ) : Prop := r1 < abs z ∧ abs z < r2

theorem complex_roots_in_annulus :
  ∀ (z : ℂ), is_root (λ z, z^3 + z + 1) z → (annulus z (sqrt 13 / 3) (5 / 4)) ∨ (z = real_root)
:= 
sorry

end complex_roots_in_annulus_l758_758284


namespace possible_value_of_b_l758_758588

-- Definition of the linear function
def linear_function (x : ℝ) (b : ℝ) : ℝ := -2 * x + b

-- Condition for the linear function to pass through the second, third, and fourth quadrants
def passes_second_third_fourth_quadrants (b : ℝ) : Prop :=
  b < 0

-- Lean 4 statement expressing the problem
theorem possible_value_of_b (b : ℝ) (h : passes_second_third_fourth_quadrants b) : b = -1 :=
  sorry

end possible_value_of_b_l758_758588


namespace number_times_one_fourth_squared_eq_four_cubed_l758_758355

theorem number_times_one_fourth_squared_eq_four_cubed :
  ∃ x : ℕ, x * (1 / 4: ℝ)^2 = (4: ℝ)^3 :=
by
  use 1024
  sorry

end number_times_one_fourth_squared_eq_four_cubed_l758_758355


namespace min_log_value_l758_758166

-- Definition of the conditions
def geometric_sequence (b : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b n * q

def log_identity (b : ℕ → ℝ) (n : ℕ) : Prop :=
  log 2 (b (2) * b (3) * ... * b (n)) = log 2 (b (1) ^ 2)

def is_divisor_of_6 (n : ℕ) : Prop :=
  ∃ k : ℤ, 6 = k * (n - 3)

-- Main theorem statement
theorem min_log_value (b : ℕ → ℝ) (q : ℝ) (n : ℕ)
  (h1 : geometric_sequence b q)
  (h2 : log_identity b n)
  (h3 : is_divisor_of_6 n) 
: ∃ m, log q (b 1 ^ 2) = -12 :=
sorry

end min_log_value_l758_758166


namespace sum_of_valid_ns_l758_758954

-- Definition for proving that 73 cents is the greatest postage that cannot be formed.
def cannot_form (denominations : List ℕ) (value : ℕ) : Prop :=
  ∀ (a b c : ℕ), 4 * a + denominations.head * b + (denominations.head + 2) * c ≠ value

theorem sum_of_valid_ns :
  ∑ n in {n : ℕ | cannot_form [4, n, n+2] 73 ∧ n = 19 ∨ n = 23}, n = 42 :=
by
  sorry

end sum_of_valid_ns_l758_758954


namespace inequality_proof_l758_758248

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  1 / a^2 + 1 / b^2 + 1 / c^2 ≥ 2 * (a^3 + b^3 + c^3) / (a * b * c) + 3 :=
by
  sorry

end inequality_proof_l758_758248


namespace intersection_A_B_l758_758511

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_A_B : A ∩ B = {-2, 2} := by sorry

end intersection_A_B_l758_758511


namespace ones_divisible_by_power_of_three_l758_758273

theorem ones_divisible_by_power_of_three (n : ℕ) (hn : n ≥ 1) :
  (∑ i in finset.range (3^n), 10^i) % 3^n = 0 :=
by
  sorry

end ones_divisible_by_power_of_three_l758_758273


namespace centroid_moves_along_line_l758_758540

-- Definitions for points, line segments, centroids, and movements are assumed to come from the necessary mathematical libraries.
open Point
open Segment
open Geometry

variables {A B : Point} -- A and B are fixed points
variable {line_C : Line} -- line_C is the line along which C moves

-- C is a point that moves along the line_C.
noncomputable def moving_C (t : ℝ) : Point := line_C.point_at t

-- Given the base AB is fixed and C moves along a line, prove the centroid moves along a straight line.
theorem centroid_moves_along_line (t : ℝ):
  let C := moving_C t in
  let G := centroid (triangle_of_points A B C) in
  exists line_G : Line, ∀ t : ℝ, G ∈ line_G :=
sorry

end centroid_moves_along_line_l758_758540


namespace parabola_translation_eq_l758_758795

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := -x^2 + 2

-- Define the translated parabola function
def translated_parabola (x : ℝ) : ℝ := - (x - 2)^2 - 1

-- State the theorem to prove the translated function
theorem parabola_translation_eq :
  ∀ x : ℝ, translated_parabola x = - (x - 2)^2 - 1 :=
by
  sorry

end parabola_translation_eq_l758_758795


namespace number_of_people_in_team_l758_758180

variable (x : ℕ) -- Number of people in the team

-- Conditions as definitions
def average_age_all (x : ℕ) : ℝ := 25
def leader_age : ℝ := 45
def average_age_without_leader (x : ℕ) : ℝ := 23

-- Proof problem statement
theorem number_of_people_in_team (h1 : (x : ℝ) * average_age_all x = x * (average_age_without_leader x - 1) + leader_age) : x = 11 := by
  sorry

end number_of_people_in_team_l758_758180


namespace sum_inequality_l758_758251

theorem sum_inequality (n : ℕ) (a b x : Fin n → ℝ)
  (hx : ∀ i j, i ≤ j → x i ≤ x j)
  (hcond1 : ∀ k : Fin (n-1), (Finset.univ.pred k).sum a ≥ (Finset.univ.pred k).sum b)
  (hcond2 : (Finset.univ).sum a = (Finset.univ).sum b) 
  : (Finset.univ).sum (λ i, a i * x i) ≤ (Finset.univ).sum (λ i, b i * x i) :=
sorry

end sum_inequality_l758_758251


namespace boat_distance_l758_758858

theorem boat_distance (v_b : ℝ) (v_s : ℝ) (t_downstream : ℝ) (t_upstream : ℝ) (d : ℝ) :
  v_b = 7 ∧ t_downstream = 2 ∧ t_upstream = 5 ∧ d = (v_b + v_s) * t_downstream ∧ d = (v_b - v_s) * t_upstream → d = 20 :=
by {
  sorry
}

end boat_distance_l758_758858


namespace vector_operation_result_l758_758566

-- Definitions of vectors a and b
def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (2, -3)

-- The operation 2a - b
def operation (a b : ℝ × ℝ) : ℝ × ℝ :=
(2 * a.1 - b.1, 2 * a.2 - b.2)

-- The theorem stating the result of the operation
theorem vector_operation_result : operation a b = (-4, 5) :=
by
  sorry

end vector_operation_result_l758_758566


namespace proof_of_c_value_l758_758379

theorem proof_of_c_value (a b : ℝ) (ha : 8 = (6 / 100) * a) (hb : 6 = (8 / 100) * b) : 
  let c := b / a in c ≈ 0.5625 :=
by
  sorry

end proof_of_c_value_l758_758379


namespace triangle_side_value_l758_758192

theorem triangle_side_value (a b c : ℝ) (A B C : ℝ) 
  (h1 : a^2 - c^2 = 2 * b)
  (h2 : sin A * cos C = 3 * cos A * sin C) : 
  b = 4 := 
sorry

end triangle_side_value_l758_758192


namespace circumcenter_on_AP_l758_758984

noncomputable def circumcenter_lies_on_line (A B C B' C' P : Point) : Prop :=
  acute_angle_triangle A B C ∧
  symmetric_point B B' AC ∧
  symmetric_point C C' AB ∧
  intersection P (circumcircle A B B') (circumcircle A C C') A ∧
  line_contains (circumcenter A B C) (line_through P A)

theorem circumcenter_on_AP
  (A B C : Point)
  (h1 : acute_angle_triangle A B C)
  (B' C' : Point)
  (h2 : symmetric_point B B' (line_through A C))
  (h3 : symmetric_point C C' (line_through A B))
  (P : Point)
  (h4 : P ≠ A)
  (h5 : P ∈ intersection (circumcircle A B B') (circumcircle A C C')) : 
  line_contains (circumcenter A B C) (line_through P A) := 
  sorry

end circumcenter_on_AP_l758_758984


namespace time_to_pick_up_dog_l758_758222

def commute_time : ℕ := 30
def grocery_time : ℕ := 30
def dry_cleaning_time : ℕ := 10
def cooking_time : ℕ := 90
def dinner_time_in_minutes : ℕ := 180  -- 7:00 pm - 4:00 pm in minutes

def total_known_time : ℕ := commute_time + grocery_time + dry_cleaning_time + cooking_time

theorem time_to_pick_up_dog : (dinner_time_in_minutes - total_known_time) = 20 :=
by
  -- Proof goes here.
  sorry

end time_to_pick_up_dog_l758_758222


namespace martha_clothes_total_l758_758714

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

end martha_clothes_total_l758_758714


namespace divide_pile_l758_758684

theorem divide_pile (pile : ℝ) (similar : ℝ → ℝ → Prop) :
  (∀ x y, similar x y ↔ x ≤ y * Real.sqrt 2 ∧ y ≤ x * Real.sqrt 2) →
  ¬∃ a b c, a + b + c = pile ∧ similar a b ∧ similar b c ∧ similar a c :=
by sorry

end divide_pile_l758_758684


namespace dimension_of_cycle_and_cut_space_l758_758255

-- Define the connected graph G with n vertices and m edges
variables (G : Type) [graph G] [connected G]
variables (n m : ℕ) [fintype G] (n : ℕ := fintype.card Vertex) (m : ℕ := finset.card Edge)

-- Define the spanning tree T of the graph G
variables (T : finset (subgraph G)) [spanning_tree T]

-- Define that the fundamental graph and fundamental cut form the basis of the cycle space and cut space
variables (C G : Type) [cycle_space C G] [cut_space C G]

-- Prove that dim cycle space and cut space given the conditions
theorem dimension_of_cycle_and_cut_space : 
  dim(𝒞(G)) = m - n + 1 ∧ dim(𝒞*(G)) = n - 1 := by 
  sorry

end dimension_of_cycle_and_cut_space_l758_758255


namespace f_strictly_increasing_intervals_l758_758549

noncomputable def f (x : Real) : Real :=
  x * Real.sin x + Real.cos x

noncomputable def f' (x : Real) : Real :=
  x * Real.cos x

theorem f_strictly_increasing_intervals :
  ∀ (x : Real), (-π < x ∧ x < -π / 2 ∨ 0 < x ∧ x < π / 2) → f' x > 0 :=
by
  intros x h
  sorry

end f_strictly_increasing_intervals_l758_758549


namespace solve_for_s_and_t_l758_758436

theorem solve_for_s_and_t : 
  ∃ s t : ℝ, 7 * s + 3 * t = 102 ∧ s = (t - 3) ^ 2 ∧ t ≈ 6.44 ∧ s ≈ 11.83 :=
begin
  sorry
end

end solve_for_s_and_t_l758_758436


namespace larger_number_1655_l758_758297

theorem larger_number_1655 (L S : ℕ) (h1 : L - S = 1325) (h2 : L = 5 * S + 5) : L = 1655 :=
by sorry

end larger_number_1655_l758_758297


namespace circle_quadrilateral_equivalence_l758_758720

-- Define the points A, B, C, and D as they lie on a circle ω₁
variables (A B C D : Point) (ω₁ : Circle)

-- Define the cyclic property: points A, C, B, D lie on circle ω₁ in order
axiom cyclic_order_on_circle : cyclic_order ω₁ [A, C, B, D]

-- Define midpoint arc condition
def midpoint_arc (P Q : Point) (M : Point) :=
  ∃ O : Circle, P ≠ Q ∧ P, Q, M ∈ O ∧ (arc_len O P M + arc_len O M Q = arc_len O P Q / 2)

-- Define the length of segments
variables {AC BC AD BD CD : ℝ}

-- Condition stating lengths without concerns for constructions inside the proof
axiom ac_bc_ad_bd : length A C = AC ∧ length B C = BC ∧ length A D = AD ∧ length B D = BD ∧ length C D = CD

theorem circle_quadrilateral_equivalence :
  CD^2 = AC * BC + AD * BD ↔ midpoint_arc A B C ∨ midpoint_arc A B D := 
sorry

end circle_quadrilateral_equivalence_l758_758720


namespace ceil_floor_arith_l758_758417

theorem ceil_floor_arith :
  (Int.ceil (((15: ℚ) / 8)^2 * (-34 / 4)) - Int.floor ((15 / 8) * Int.floor (-34 / 4))) = -12 :=
by sorry

end ceil_floor_arith_l758_758417


namespace find_geometric_ratio_l758_758606

-- Definitions for the conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + (a 1 - a 0)

def geometric_sequence (a1 a3 a4 : ℝ) (q : ℝ) : Prop :=
  a3 * a3 = a1 * a4 ∧ a3 = a1 * q ∧ a4 = a3 * q

-- Definition for the proof statement
theorem find_geometric_ratio (a : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hnz : ∀ n, a n ≠ 0)
  (hq : ∃ (q : ℝ), geometric_sequence (a 0) (a 2) (a 3) q) :
  ∃ q, q = 1 ∨ q = 1 / 2 := sorry

end find_geometric_ratio_l758_758606


namespace focus_of_parabola_y_eq_4x_sq_l758_758011

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := (0 : ℝ, 1 / 16 : ℝ)
  in f

theorem focus_of_parabola_y_eq_4x_sq :
  (0, 1 / 16) = parabola_focus := by
  unfold parabola_focus
  sorry

end focus_of_parabola_y_eq_4x_sq_l758_758011


namespace focus_of_parabola_l758_758002

theorem focus_of_parabola (f d : ℝ) :
  (∀ x : ℝ, x^2 + (4*x^2 - f)^2 = (4*x^2 - d)^2) → 8*f + 8*d = 1 → f^2 = d^2 → f = 1/16 :=
by
  intro hEq hCoeff hSq
  sorry

end focus_of_parabola_l758_758002


namespace log_evaluation_l758_758931

theorem log_evaluation
  (y : ℝ)
  (log_def : 8 ^ y = 50)
  : y = (1 + 2 * log (50) / log(2)) / 3 :=
by
  sorry

end log_evaluation_l758_758931


namespace hyperbola_eccentricity_l758_758116

def hyperbola_foci (F1 F2 P : ℝ) (θ : ℝ) (PF1 PF2 : ℝ) : Prop :=
  θ = 60 ∧ PF1 = 3 * PF2

def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity (F1 F2 P : ℝ) (θ PF1 PF2 : ℝ)
  (h : hyperbola_foci F1 F2 P θ PF1 PF2) :
  eccentricity 1 (sqrt 7 / 2) = sqrt 7 / 2 :=
by {
  sorry
}

end hyperbola_eccentricity_l758_758116


namespace construct_rectangle_parallel_diagonals_l758_758288

-- Definitions used in the problem
variables {A B C D P Q R S : Type} [square A B C D] (point_P : P ∈ segment A B)

-- Lean 4 statement encapsulating the problem
theorem construct_rectangle_parallel_diagonals (h : P ∈ segment A B) :
  ∃ (Q R S : Type), is_rectangle P Q R S ∧ 
                     ((parallel (line P Q) (diagonal A C) ∧ parallel (line R S) (diagonal A C)) ∨
                      (parallel (line P Q) (diagonal B D) ∧ parallel (line R S) (diagonal B D))) := sorry

end construct_rectangle_parallel_diagonals_l758_758288


namespace afternoon_sales_l758_758846

theorem afternoon_sales (x : ℕ) (h : 3 * x = 510) : 2 * x = 340 :=
by sorry

end afternoon_sales_l758_758846


namespace curvilinear_triangle_area_l758_758391

theorem curvilinear_triangle_area (R : ℝ) (hR : 0 < R) : 
  let equilateral_triangle_area := (sqrt 3) * R^2
  let circular_sector_area := (π * R^2) / 2
  let curvilinear_triangle_area := equilateral_triangle_area - circular_sector_area
  curvilinear_triangle_area = (R^2 * (2 * sqrt 3 - π)) / 2 :=
by
  -- Placeholder for the proof
  sorry

end curvilinear_triangle_area_l758_758391


namespace remainder_T10_mod_5_l758_758042

noncomputable def T : ℕ → ℕ
| 0     => 1
| 1     => 2
| (n+2) => T (n+1) + T n + T n

theorem remainder_T10_mod_5 :
  (T 10) % 5 = 4 :=
sorry

end remainder_T10_mod_5_l758_758042


namespace salon_visitors_l758_758897

noncomputable def total_customers (x : ℕ) : ℕ :=
  let revenue_customers_with_one_visit := 10 * x
  let revenue_customers_with_two_visits := 30 * 18
  let revenue_customers_with_three_visits := 10 * 26
  let total_revenue := revenue_customers_with_one_visit + revenue_customers_with_two_visits + revenue_customers_with_three_visits
  if total_revenue = 1240 then
    x + 30 + 10
  else
    0

theorem salon_visitors : 
  ∃ x, total_customers x = 84 :=
by
  use 44
  sorry

end salon_visitors_l758_758897


namespace sin_angle_PBO_l758_758227

-- Definitions for the regular tetrahedron, centroid, and points
variables {A B C D O P : Type} [regular_tetrahedron ABCD] [centroid O (triangle BCD)]
variable (min_point_on_AO : minimizes_on_line_segment P (line_segment AO) 
  (dist PA + 2 * (dist PB + dist PC + dist PD)))

-- The main theorem to prove
theorem sin_angle_PBO :
  ∃ (P : Type), minimizes_on_line_segment P (line_segment AO) 
    (dist PA + 2 * (dist PB + dist PC + dist PD))
    → sin (angle_at_point P B O) = 1/6 := sorry

end sin_angle_PBO_l758_758227


namespace rectangular_to_polar_l758_758915

theorem rectangular_to_polar (x y : ℝ) (h : x = 3 ∧ y = 3 * real.sqrt 3) :
    ∃ (r θ : ℝ), r > 0 ∧ (0 ≤ θ ∧ θ < 2 * real.pi) ∧ (r, θ) = (6, real.pi / 3) :=
begin
  use [6, real.pi / 3],
  split,
  { simpa using show 6 > 0, from dec_trivial },
  split,
  { split, 
    { simp [real.pi_pos] }, 
    { linarith [real.pi_pos] } },
  { exact ⟨6, real.pi / 3⟩ }
end

#eval rectangular_to_polar 3 (3 * real.sqrt 3) ⟨rfl, rfl⟩

end rectangular_to_polar_l758_758915


namespace direction_vector_of_line_l758_758557

theorem direction_vector_of_line {x y : ℝ} (h : x + y + 1 = 0) : ∃ v : ℝ × ℝ, v = (1, -1) :=
by 
  use (1, -1)
  sorry

end direction_vector_of_line_l758_758557


namespace volume_of_cylinder_l758_758957

-- Define the length and width of the rectangle
def length_rect : ℝ := 10
def width_rect : ℝ := 8

-- Define the radius and height of the resulting cylinder
def radius_cylinder : ℝ := width_rect / 2
def height_cylinder : ℝ := length_rect

-- Define the expected volume of the cylinder
def expected_volume : ℝ := 160 * Real.pi

-- Define a theorem to prove the volume of the cylinder
theorem volume_of_cylinder :
  let V := Real.pi * (radius_cylinder ^ 2) * height_cylinder in
  V = expected_volume :=
by
  sorry

end volume_of_cylinder_l758_758957


namespace find_p_l758_758211

def Point := (ℝ × ℝ)

def A : Point := (4, 12)
def B : Point := (12, 0)
def C (p : ℝ) : Point := (0, p)
def Q : Point := (0, 12)

def TriangleArea (A B C : Point) : ℝ :=
  |(A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))| / 2

theorem find_p (p : ℝ) (h : TriangleArea A B (C p) = 20) : p = 6.5 :=
by
  sorry

end find_p_l758_758211


namespace remainder_86592_8_remainder_8741_13_l758_758345

theorem remainder_86592_8 :
  86592 % 8 = 0 :=
by
  sorry

theorem remainder_8741_13 :
  8741 % 13 = 5 :=
by
  sorry

end remainder_86592_8_remainder_8741_13_l758_758345


namespace infinite_indices_exist_l758_758249

theorem infinite_indices_exist (a : ℕ → ℕ) (h_seq : ∀ n, a n < a (n + 1)) :
  ∃ᶠ m in ⊤, ∃ x y h k : ℕ, 0 < h ∧ h < k ∧ k < m ∧ a m = x * a h + y * a k :=
by sorry

end infinite_indices_exist_l758_758249


namespace impossible_to_divide_three_similar_parts_l758_758699

theorem impossible_to_divide_three_similar_parts 
  (n : ℝ) 
  (p : ℝ → Prop) 
  (similar : ℝ → ℝ → Prop) 
  (h_similar : ∀ a b : ℝ, similar a b ↔ a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2) : 
  ¬ ∃ p1 p2 p3 : ℝ, p p1 ∧ p p2 ∧ p p3 ∧ (p1 + p2 + p3 = n) ∧ similar p1 p2 ∧ similar p2 p3 ∧ similar p1 p3 :=
sorry

end impossible_to_divide_three_similar_parts_l758_758699


namespace gas_cost_for_james_trip_l758_758221

noncomputable def total_cost_of_trip
  (init_odo : ℕ)
  (grocery_odo : ℕ)
  (friend_odo : ℕ)
  (efficiency : ℝ)
  (gas_price : ℝ) : ℝ :=
  let distance_to_grocery := grocery_odo - init_odo
  let distance_to_friend := friend_odo - grocery_odo
  let total_distance := distance_to_grocery + distance_to_friend
  let gallons_used := total_distance / efficiency
  gallons_used * gas_price

theorem gas_cost_for_james_trip
  (h1 : 55300)
  (h2 : 55328)
  (h3 : 55345)
  (h4 : 25 : ℝ)
  (h5 : 3.80 : ℝ) :
  total_cost_of_trip 55300 55328 55345 25 3.80 = 6.84 :=
  by sorry

end gas_cost_for_james_trip_l758_758221


namespace possible_values_of_polynomial_l758_758310

theorem possible_values_of_polynomial (x : ℝ) (h : x^2 - 7 * x + 12 < 0) : 
48 < x^2 + 7 * x + 12 ∧ x^2 + 7 * x + 12 < 64 :=
sorry

end possible_values_of_polynomial_l758_758310


namespace fraction_of_credit_extended_l758_758414

noncomputable def C_total : ℝ := 342.857
noncomputable def P_auto : ℝ := 0.35
noncomputable def C_company : ℝ := 40

theorem fraction_of_credit_extended :
  (C_company / (C_total * P_auto)) = (1 / 3) :=
  by
    sorry

end fraction_of_credit_extended_l758_758414


namespace smallest_translation_theta_l758_758303

theorem smallest_translation_theta:
  ∀ θ : ℝ, θ > 0 →
  (∀ x : ℝ, cos(x - θ + 4 * π / 3) = cos(-x - θ + 4 * π / 3)) →
  θ = π / 3 := 
sorry

end smallest_translation_theta_l758_758303


namespace max_coefficient_l758_758748

theorem max_coefficient (a : ℝ) (h : binomialCoeff 5 1 * (-a) = -5) :
  binomialCoeff 5 2 = 10 := by
  sorry

end max_coefficient_l758_758748


namespace eccentricity_of_hyperbola_l758_758123

open Real

-- Definitions of our conditions
variables {F1 F2 P : Point}
variables (a : ℝ) (m : ℝ)
variable (hyperbola_C : Hyperbola F1 F2)

-- Given conditions
axiom on_hyperbola : P ∈ hyperbola_C
axiom angle_F1P_F2 : angle F1 P F2 = π / 3
axiom distances : dist P F1 = 3 * dist P F2

-- Goal: Prove that the eccentricity of the hyperbola is sqrt(7)/2
theorem eccentricity_of_hyperbola : hyperbola.C.eccentricity = sqrt 7 / 2 := by
  sorry

end eccentricity_of_hyperbola_l758_758123


namespace impossible_divide_into_three_similar_parts_l758_758686

noncomputable def similar (x y : ℝ) : Prop := x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

theorem impossible_divide_into_three_similar_parts (s : ℝ → ℝ → Prop) :
  (∀ s, similar s)) → ¬ (∃ a b c : ℝ, s a b → s b c → s c a → a + b + c = 1) :=
by
  intros h_similar h
  sorry

end impossible_divide_into_three_similar_parts_l758_758686


namespace point_on_line_l758_758098

theorem point_on_line (θ : ℝ) (hθ : θ ∈ Ioo 0 (Real.pi / 2)) 
  (hP : (sin θ) + (3 * sin θ + 1) - 3 = 0) : θ = Real.pi / 6 :=
sorry

end point_on_line_l758_758098


namespace volume_of_tetrahedron_l758_758838

theorem volume_of_tetrahedron 
  (PQ PR PS QR QS RS : ℝ)
  (hPQ : PQ = 3)
  (hPR : PR = 4)
  (hPS : PS = 5)
  (hQR : QR = Real.sqrt 17)
  (hQS : QS = Real.sqrt 26)
  (hRS : RS = Real.sqrt 29) :
  (volume PQ PR PS QR QS RS = 9) :=
sorry

end volume_of_tetrahedron_l758_758838


namespace find_logb_csc_x_l758_758577

variable (b x a : ℝ)
variable (h1 : b > 1)
variable (h2 : sin x > 0)
variable (h3 : cos x > 0)
variable (h4 : log b (tan x) = a)

theorem find_logb_csc_x (b x a : ℝ) (h1 : b > 1) (h2 : sin x > 0) (h3 : cos x > 0) (h4 : log b (tan x) = a) :
  log b (csc x) = - (1/2) * log b (1 - 1 / (b^(2 * a) + 1)) :=
sorry

end find_logb_csc_x_l758_758577


namespace irrigation_system_flow_rates_l758_758965

-- Define the conditions
variable (q0 : ℝ) -- Flow rate in channel BC

-- Channels' flow rates
variable (qAB qAH q_total : ℝ)

-- Define the conditions as hypotheses
axiom H1 : qAB = 1/2 * q0
axiom H2 : qAH = 3/4 * q0
axiom H3 : q_total = qAB + qAH

-- Prove the results
theorem irrigation_system_flow_rates
  (q0 : ℝ)
  (qAB qAH q_total : ℝ)
  (H1 : qAB = 1/2 * q0)
  (H2 : qAH = 3/4 * q0)
  (H3 : q_total = qAB + qAH) :
  qAB = 1/2 * q0 ∧ qAH = 3/4 * q0 ∧ q_total = 7/4 * q0 :=
by {
  split,
  exact H1,
  split,
  exact H2,
  rw [H3, H1, H2],
  linarith
}

end irrigation_system_flow_rates_l758_758965


namespace fraction_girls_on_trip_correct_l758_758616

-- Given conditions and definitions
def total_students : ℕ := 200
def boys : ℕ := 100
def girls : ℕ := 100

def fraction_girls_on_trip : ℚ := 5 / 8
def fraction_boys_on_trip : ℚ := 3 / 5

def number_girls_on_trip : ℕ := (fraction_girls_on_trip * girls).to_nat
def number_boys_on_trip : ℕ := (fraction_boys_on_trip * boys).to_nat

def total_students_on_trip : ℕ := number_girls_on_trip + number_boys_on_trip

-- The theorem to prove
theorem fraction_girls_on_trip_correct :
  (number_girls_on_trip % total_students_on_trip) = 63 / 123 := sorry

end fraction_girls_on_trip_correct_l758_758616


namespace full_price_revenue_l758_758390

def total_tickets := 250
def total_revenue := 3500

variable (f h q p : ℕ)
variable (f_price half_price quarter_price : ℕ)

-- Define the conditions
def full_price_ticket := f_price = p
def half_price_ticket := half_price = p / 2
def quarter_price_ticket := quarter_price = p / 4

def total_ticket_count := f + h + q = total_tickets
def total_revenue_count := f * p + h * (p / 2) + q * (p / 4) = total_revenue

-- The theorem to prove
theorem full_price_revenue :
  full_price_ticket ∧
  half_price_ticket ∧
  quarter_price_ticket ∧
  total_ticket_count ∧
  total_revenue_count →
  f * p = 3000 :=
sorry

end full_price_revenue_l758_758390


namespace count_valid_digits_l758_758045

theorem count_valid_digits :
  {n : ℕ | 1 ≤ n ∧ n ≤ 9 ∧ 15 * n % n = 0}.card = 7 :=
by sorry

end count_valid_digits_l758_758045


namespace split_numbers_even_sum_l758_758888

-- Define the problem conditions
def numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def valid_split (A B : Set ℕ) (h : A ∪ B = numbers) : Prop :=
  A.Nonempty ∧ B.Nonempty ∧
  (((A.sum id) % 2 = 0) ∨ ((B.sum id) % 2 = 0))

-- Define the main theorem
theorem split_numbers_even_sum :
  ∃ (A B : Set ℕ), valid_split A B ∧
  (card {p | ∃ (A B : Set ℕ), p = (A, B) ∧ valid_split A B} = 1022) :=
sorry

end split_numbers_even_sum_l758_758888


namespace least_positive_integer_with_eight_factors_l758_758814

theorem least_positive_integer_with_eight_factors :
  ∃ n : ℕ, n = 24 ∧ (8 = (nat.factors n).length) :=
sorry

end least_positive_integer_with_eight_factors_l758_758814


namespace degree_measure_of_subtracted_angle_l758_758347

def angle := 30

theorem degree_measure_of_subtracted_angle :
  let supplement := 180 - angle
  let complement_of_supplement := 90 - supplement
  let twice_complement := 2 * (90 - angle)
  twice_complement - complement_of_supplement = 180 :=
by
  sorry

end degree_measure_of_subtracted_angle_l758_758347


namespace num_divisible_digits_l758_758051

def divisible_by_n (num : ℕ) (n : ℕ) : Prop :=
  n ≠ 0 ∧ num % n = 0

def count_divisible_digits : ℕ :=
  (List.filter (λ n => divisible_by_n (150 + n) n)
    [1, 2, 3, 4, 5, 6, 7, 8, 9]).length

theorem num_divisible_digits : count_divisible_digits = 7 := by
  sorry

end num_divisible_digits_l758_758051


namespace other_root_of_quadratic_l758_758584

theorem other_root_of_quadratic (m : ℝ) : 
  (Polynomial.X ^ 2 + Polynomial.C m * Polynomial.X + Polynomial.C (-12)).eval 3 = 0 → 
  (Polynomial.X ^ 2 + Polynomial.C m * Polynomial.X + Polynomial.C (-12)).roots.find (λ x, x ≠ 3) = some (-4) :=
  sorry

end other_root_of_quadratic_l758_758584


namespace nearly_perfect_is_odd_l758_758877

open Nat

def is_nearly_perfect (n : ℕ) : Prop :=
  ∑ d in divisors n, d = 2 * n + 1

theorem nearly_perfect_is_odd {n : ℕ} (h : is_nearly_perfect n) : Odd n :=
  sorry

end nearly_perfect_is_odd_l758_758877


namespace candidate1_fails_by_l758_758861

-- Define the total marks (T), passing marks (P), percentage marks (perc1 and perc2), and the extra marks.
def T : ℝ := 600
def P : ℝ := 160
def perc1 : ℝ := 0.20
def perc2 : ℝ := 0.30
def extra_marks : ℝ := 20

-- Define the marks obtained by the candidates.
def marks_candidate1 : ℝ := perc1 * T
def marks_candidate2 : ℝ := perc2 * T

-- The theorem stating the number of marks by which the first candidate fails.
theorem candidate1_fails_by (h_pass: perc2 * T = P + extra_marks) : P - marks_candidate1 = 40 :=
by
  -- The proof would go here.
  sorry

end candidate1_fails_by_l758_758861


namespace distinct_convex_polygons_l758_758800

def twelve_points : Finset (Fin 12) := (Finset.univ : Finset (Fin 12))

noncomputable def polygon_count_with_vertices (n : ℕ) : ℕ :=
  2^n - 1 - n - (n * (n - 1)) / 2

theorem distinct_convex_polygons :
  polygon_count_with_vertices 12 = 4017 := 
by
  sorry

end distinct_convex_polygons_l758_758800


namespace positive_difference_complementary_angles_l758_758770

theorem positive_difference_complementary_angles (a b : ℝ) 
  (h1 : a + b = 90) 
  (h2 : 3 * b = a) :
  |a - b| = 45 :=
by
  sorry

end positive_difference_complementary_angles_l758_758770


namespace divide_pile_l758_758679

theorem divide_pile (pile : ℝ) (similar : ℝ → ℝ → Prop) :
  (∀ x y, similar x y ↔ x ≤ y * Real.sqrt 2 ∧ y ≤ x * Real.sqrt 2) →
  ¬∃ a b c, a + b + c = pile ∧ similar a b ∧ similar b c ∧ similar a c :=
by sorry

end divide_pile_l758_758679


namespace smaller_number_eq_l758_758776

variable (m n t s : ℝ)
variable (h_ratio : m / n = t)
variable (h_sum : m + n = s)
variable (h_t_gt_one : t > 1)

theorem smaller_number_eq : n = s / (1 + t) :=
by sorry

end smaller_number_eq_l758_758776


namespace find_smallest_positive_integer_l758_758953

noncomputable def smallestPositiveInteger : ℕ :=
  20 * x^2 + 80 * x * y + 95 * y^2

theorem find_smallest_positive_integer :
  ∃ (x y : ℤ), smallestPositiveInteger x y = 67 ∧ (∀ (x y : ℤ), smallestPositiveInteger x y ≥ 67) :=
begin
  sorry
end

end find_smallest_positive_integer_l758_758953


namespace min_soldiers_in_square_formations_l758_758789

theorem min_soldiers_in_square_formations : ∃ (a : ℕ), 
  ∃ (k : ℕ), 
    (a = k^2 ∧ 
    11 * a + 1 = (m : ℕ) ^ 2) ∧ 
    (∀ (b : ℕ), 
      (∃ (j : ℕ), b = j^2 ∧ 11 * b + 1 = (n : ℕ) ^ 2) → a ≤ b) ∧ 
    a = 9 := 
sorry

end min_soldiers_in_square_formations_l758_758789


namespace true_propositions_serial_numbers_l758_758312

def inverse_congruent_triangles_have_equal_area (P : Prop) : Prop :=
¬(P → Q ↔ (¬Q → ¬P))

def negation_of_ab_zero_then_a_zero (P Q : Prop) : Prop :=
(P ∧ Q) ↔ (P → Q)

def contrapositive_equilateral_triangle_angles (P Q : Prop) : Prop :=
(P → Q) ↔ (¬Q → ¬P)

theorem true_propositions_serial_numbers :
  (inverse_congruent_triangles_have_equal_area →
  (negation_of_ab_zero_then_a_zero →
  contrapositive_equilateral_triangle_angles))
  ↔ (true_propositions_serial_numbers = {2, 3}) :=
by
  sorry

end true_propositions_serial_numbers_l758_758312


namespace find_x1_l758_758531

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) :
  x1 = 4 / 5 := 
sorry

end find_x1_l758_758531


namespace count_triangles_l758_758977

noncomputable def number_of_integer_valued_triangles : ℕ :=
  let x_vals := {x : ℕ | 7 < x ∧ x < 13}
  x_vals.to_finset.card

theorem count_triangles : number_of_integer_valued_triangles = 5 := by
  sorry

end count_triangles_l758_758977


namespace curve_equation_midpoint_trajectory_l758_758088

-- (Ⅰ) Given conditions and required proof for finding the equation of curve E
theorem curve_equation (P F1 F2 : ℝ × ℝ) (a b x y : ℝ) :
  F1 = (- √3, 0) ∧ F2 = (√3, 0) ∧ (dist P F1 + dist P F2) = 4 →
  ∃ x y, P = (x, y) ∧ x^2 / 4 + y^2 / 3 = 1 :=
by
  -- Proof steps will go here
  sorry

-- (Ⅱ) Given conditions and required proof for finding the equation of the trajectory of M
theorem midpoint_trajectory (A : ℝ × ℝ) (P : ℝ × ℝ) (x y : ℝ) :
  A = (1, 1) ∧ (P.1^2 / 4 + P.2^2 / 3 = 1) →
  ∃ x y, let M := ((1 + P.1) / 2, (1 + P.2) / 2) in
  (x = M.1) ∧ (y = M.2) ∧ ((x - 1/2)^2 + (4 / 3) * (y - 1/2)^2 = 1) :=
by
  -- Proof steps will go here
  sorry

end curve_equation_midpoint_trajectory_l758_758088


namespace hyperbola_eccentricity_l758_758126

theorem hyperbola_eccentricity (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
    (C : P → Prop) 
    (angle_F1PF2 : ∀ {x y z : P}, ∀ (h : x ∈ C) (h₁ : y ∈ C) (h₂ : z ∈ C), is_angle x y z = 60)
    (dist_PF1_PF2 : ∀ (h : P ∈ C), dist P F1 = 3 * dist P F2) : 
    F1 ∈ C → F2 ∈ C → eccentricity C = (Real.sqrt 7) / 2 :=
by sorry

end hyperbola_eccentricity_l758_758126


namespace quadratic_polynomial_with_special_roots_l758_758944

theorem quadratic_polynomial_with_special_roots :
  ∀ (a b c : ℤ), (a ≠ 0) ∧ 
  (a^2 + b * a + c = 0) ∧ (b^2 + b * a + c = 0) ∧ (c^2 + b * a + c = 0) →
  ((a + b) * (a - b) = a - c - b) ∨
  ((a + c) * (a - b) = a - b - c) ∨
  ((b + c) * (a - c) = a - b - c) →
  false :=
begin
  sorry
end

end quadratic_polynomial_with_special_roots_l758_758944


namespace average_age_of_adults_l758_758296

theorem average_age_of_adults
    (total_members : ℕ)
    (avg_age_total : ℝ)
    (girls : ℕ)
    (boys : ℕ)
    (adults : ℕ)
    (avg_age_girls : ℝ)
    (avg_age_boys : ℝ)
    (avg_age_adults : ℝ) :
    total_members = 50 →
    avg_age_total = 20 →
    girls = 25 →
    boys = 18 →
    adults = 7 →
    avg_age_girls = 18 →
    avg_age_boys = 19 →
    avg_age_adults = 29.71 :=
by
  assume h1 : total_members = 50,
  assume h2 : avg_age_total = 20,
  assume h3 : girls = 25,
  assume h4 : boys = 18,
  assume h5 : adults = 7,
  assume h6 : avg_age_girls = 18,
  assume h7 : avg_age_boys = 19,
  sorry

end average_age_of_adults_l758_758296


namespace g_2_minus_g_8_l758_758758

noncomputable def g : ℝ → ℝ := sorry

def g_linear : Prop := ∀ x y : ℝ, ∀ a b : ℝ, g(a * x + b * y) = a * g(x) + b * g(y)
def g_condition : Prop := ∀ x : ℝ, g(x + 2) - g(x) = 5

theorem g_2_minus_g_8 : g_linear ∧ g_condition → g(2) - g(8) = -15 :=
by sorry

end g_2_minus_g_8_l758_758758


namespace martin_walk_distance_l758_758705

-- Define the conditions
def time : ℝ := 6 -- Martin's walking time in hours
def speed : ℝ := 2 -- Martin's walking speed in miles per hour

-- Define the target distance
noncomputable def distance : ℝ := 12 -- Distance from Martin's house to Lawrence's house

-- The theorem to prove the target distance given the conditions
theorem martin_walk_distance : (speed * time = distance) :=
by
  sorry

end martin_walk_distance_l758_758705


namespace valid_digits_count_l758_758056

theorem valid_digits_count : { n ∈ Finset.range 10 | n > 0 ∧ 15 * n % n = 0 }.card = 5 :=
by
  sorry

end valid_digits_count_l758_758056


namespace B_work_days_l758_758384

theorem B_work_days (A B C : ℕ) (hA : A = 15) (hC : C = 30) (H : (5 / 15) + ((10 * (1 / C + 1 / B)) / (1 / C + 1 / B)) = 1) : B = 30 := by
  sorry

end B_work_days_l758_758384


namespace distance_from_Martins_house_to_Lawrences_house_l758_758706

def speed : ℝ := 2 -- Martin's speed is 2 miles per hour
def time : ℝ := 6  -- Time taken is 6 hours
def distance : ℝ := speed * time -- Distance formula

theorem distance_from_Martins_house_to_Lawrences_house : distance = 12 := by
  sorry

end distance_from_Martins_house_to_Lawrences_house_l758_758706


namespace tan_angle_BAD_l758_758216

-- Given definitions
def triangle (A B C : Type) := Type

def midpoint {A : Type} [has_add A] [has_div A] (x y : A) : A := (x + y) / 2

variables (A B C D : Type)

-- Conditions
def is_angle_45 (C : Type) : Prop := sorry -- Represents ∠ C = 45°
def length_eq (BC : Type) : Prop := sorry -- Represents BC = 6
def is_midpoint (D: Type) : Prop := midpoint B C = D -- D is the midpoint of BC
def length_eq_AC (AC : Type) : Prop := sorry -- Represents AC = 6

-- Conclusion to prove
theorem tan_angle_BAD {A B C D : Type} [is_angle_45 C] [length_eq BC] [is_midpoint D] [length_eq_AC AC] :
  sorry : Type := sorry -- tan(∠BAD) = 1/3

-- The statement of the theorem with the correct value
example (A B C : Type) (D : midpoint B C = D)
  (h1 : is_angle_45 C)
  (h2 : length_eq BC)
  (h3 : length_eq_AC AC) :
  tan_angle_BAD = 1 / 3 := 
sorry -- Proof omitted

end tan_angle_BAD_l758_758216


namespace log_xy_l758_758995

-- Definitions from conditions
def log (z : ℝ) : ℝ := sorry -- Assume a definition of log function
variables (x y : ℝ)
axiom h1 : log (x^2 * y^2) = 1
axiom h2 : log (x^3 * y) = 2

-- The proof goal
theorem log_xy (x y : ℝ) (h1 : log (x^2 * y^2) = 1) (h2 : log (x^3 * y) = 2) : log (x * y) = 1/2 :=
sorry

end log_xy_l758_758995


namespace mean_and_median_of_sequence_l758_758419

-- Definitions based on conditions
def seq_start := 5
def common_diff := 1
def num_terms := 60
def nth_term (n : ℕ) : ℕ := seq_start + (n - 1) * common_diff

-- Definitions to represent known facts
def sum_of_sequence (n : ℕ) : ℕ := (n * (seq_start + nth_term n)) / 2
def arithmetic_mean (n : ℕ) : ℕ := sum_of_sequence n / n
def median (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    (nth_term (n / 2) + nth_term (n / 2 + 1)) / 2
  else nth_term (n / 2 + 1)

-- Theorem to prove
theorem mean_and_median_of_sequence :
  arithmetic_mean num_terms = 34.5 ∧ median num_terms = 34.5 := by
  sorry

end mean_and_median_of_sequence_l758_758419


namespace no_division_into_three_similar_piles_l758_758668

theorem no_division_into_three_similar_piles :
    ∀ (x : ℝ),
    ∀ (y z : ℝ),
    (x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = x) →
    (x <= sqrt 2 * y ∧ y <= sqrt 2 * z ∧ z <= sqrt 2 * x) →
    false :=
by
  intro x y z
  sorry

end no_division_into_three_similar_piles_l758_758668


namespace tangency_point_l758_758030

def parabola1 (x : ℝ) : ℝ := x^2 + 10 * x + 18
def parabola2 (y : ℝ) : ℝ := y^2 + 60 * y + 910

theorem tangency_point (x y : ℝ) (h1 : y = parabola1 x) (h2 : x = parabola2 y) :
  x = -9 / 2 ∧ y = -59 / 2 :=
by
  sorry

end tangency_point_l758_758030


namespace trapezium_parallel_sides_difference_60_l758_758882

theorem trapezium_parallel_sides_difference_60 (AB CD : ℝ) (O M : EuclideanGeometry.Point) (angleAMB angleAMD : ℝ) 
  (h : EuclideanGeometry.trapezium_inscribed_in_circle AB CD O)
  (hABCD : AB.parallel_to(CD))
  (hDiagonalsIntersect : EuclideanGeometry.diagonals_intersect_in_point AC BD M)
  (hOM : OM = 2)
  (hAngleAMB : angleAMB = 60)
  (hAngleAMD : angleAMD = 60)
  : abs (CD - AB) = 2 * Float.sqrt 3 := 
sorry

end trapezium_parallel_sides_difference_60_l758_758882


namespace diagonal_sum_correct_l758_758370

variable (n : ℕ) (A : ℕ → ℕ → ℝ)
hypothesis (n_ge_4 : n ≥ 4)
hypothesis (row_arithmetic : ∀ i, ∃ d, ∀ j, A i (j + 1) = A i j + d)
hypothesis (col_geometric : ∃ r, ∀ i j, A (i + 1) j = A i j * r)
hypothesis (a24_eq_1 : A 2 4 = 1)
hypothesis (a42_eq_1_8 : A 4 2 = 1 / 8)
hypothesis (a43_eq_3_16 : A 4 3 = 3 / 16)

noncomputable def find_diagonal_sum : ℝ := 
  if h : n ≥ 1 then
    2 - (n + 2) / 2^n 
  else 0

theorem diagonal_sum_correct
  (S : ℝ)
  (hS_eq : S = ∑ k in finset.range n, A k k) :
  S = find_diagonal_sum n :=
sorry

end diagonal_sum_correct_l758_758370


namespace principal_amount_l758_758351

theorem principal_amount (P : ℝ) (SI : ℝ) (R : ℝ) (T : ℝ)
  (h1 : SI = (P * R * T) / 100)
  (h2 : SI = 640)
  (h3 : R = 8)
  (h4 : T = 2) :
  P = 4000 :=
sorry

end principal_amount_l758_758351


namespace rectangles_with_one_gray_cell_l758_758179

-- Define the number of gray cells
def gray_cells : ℕ := 40

-- Define the total rectangles containing exactly one gray cell
def total_rectangles : ℕ := 176

-- The theorem we want to prove
theorem rectangles_with_one_gray_cell (h : gray_cells = 40) : total_rectangles = 176 := 
by 
  sorry

end rectangles_with_one_gray_cell_l758_758179


namespace total_current_ages_l758_758316

theorem total_current_ages (T : ℕ) : (T - 12 = 54) → T = 66 :=
by
  sorry

end total_current_ages_l758_758316


namespace common_area_of_circumscribed_square_and_triangle_l758_758407

theorem common_area_of_circumscribed_square_and_triangle (R : ℝ) :
  let circle_radius := R,
      square_area := (2 * circle_radius) ^ 2,
      triangle_height := 3 * circle_radius,
      BN1 := circle_radius,
      FN1 := circle_radius * (Real.sqrt 3 / 3),
      DF := circle_radius - FN1,
      DM := DF * (Real.sqrt 3),
      triangle_area := 1/2 * DF * DM,
      overlap_area := square_area - 2 * triangle_area
  in overlap_area = (circle_radius ^ 2 * (4 - ((2 * Real.sqrt 3 - 3) / 3)))
:=
  sorry

end common_area_of_circumscribed_square_and_triangle_l758_758407


namespace necessary_but_not_sufficient_l758_758536

variable (a : ℝ)

theorem necessary_but_not_sufficient : (a > 2) → (a > 1) ∧ ¬((a > 1) → (a > 2)) :=
by
  sorry

end necessary_but_not_sufficient_l758_758536


namespace focus_of_parabola_y_eq_4x_sq_l758_758012

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := (0 : ℝ, 1 / 16 : ℝ)
  in f

theorem focus_of_parabola_y_eq_4x_sq :
  (0, 1 / 16) = parabola_focus := by
  unfold parabola_focus
  sorry

end focus_of_parabola_y_eq_4x_sq_l758_758012


namespace probability_earning_2500_is_6_over_125_l758_758264

noncomputable def probability_earning_2500 : ℚ :=
let total_outcomes := 5 ^ 3 in
let desired_outcomes := 6 in
desired_outcomes / total_outcomes

theorem probability_earning_2500_is_6_over_125 :
  probability_earning_2500 = 6 / 125 :=
by sorry

end probability_earning_2500_is_6_over_125_l758_758264


namespace eccentricity_of_ellipse_l758_758100

variables (F1 F2 P : EuclideanGeometry.Point)
variable (C : EuclideanGeometry.Ellipse F1 F2)

-- Given conditions
variable (h1 : P ∈ C)
variable (h2 : ∠ F1 P F2 = π / 3)
variable (h3 : ∠ P F1 F2 = π / 2)

-- Expected result
theorem eccentricity_of_ellipse (e : ℝ) : e = Real.sqrt 3 - 1 :=
begin
  sorry
end

end eccentricity_of_ellipse_l758_758100


namespace count_valid_n_divisibility_l758_758075

theorem count_valid_n_divisibility : 
  (finset.univ.filter (λ n : ℕ, n ∈ finset.range 10 ∧ (15 * n) % n = 0)).card = 5 :=
by sorry

end count_valid_n_divisibility_l758_758075


namespace ann_favorite_store_l758_758411

theorem ann_favorite_store :
  ∃ (top_cost : ℤ),
    (let short_cost := 5 * 7 in
     let shoe_cost := 2 * 10 in
     let total_cost := 75 in
     let other_cost := total_cost - (short_cost + shoe_cost) in
     other_cost = 4 * top_cost) :=
by
  sorry

end ann_favorite_store_l758_758411


namespace calculate_expression_l758_758978

theorem calculate_expression (x y : ℕ) (hx : x = 3) (hy : y = 4) : 
  (1 / (y + 1)) / (1 / (x + 2)) = 1 := by
  sorry

end calculate_expression_l758_758978


namespace sum_of_series_l758_758466

theorem sum_of_series :
  ∑ n in Finset.range (2021), (1 / ((n + 1) * (n + 2))) = 2020 / 2021 := sorry

end sum_of_series_l758_758466


namespace num_digits_divisible_l758_758064

theorem num_digits_divisible (h : ∀ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}, divides n (150 + n)) :
  {n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} | divides n (150 + n)}.card = 5 := 
sorry

end num_digits_divisible_l758_758064


namespace smallest_value_3a_2_l758_758574

theorem smallest_value_3a_2 (a : ℝ) (h : 8 * a^2 + 6 * a + 5 = 2) : 3 * a + 2 = - (5 / 2) := sorry

end smallest_value_3a_2_l758_758574


namespace complementary_angles_positive_difference_l758_758765

theorem complementary_angles_positive_difference :
  ∀ (θ₁ θ₂ : ℝ), (θ₁ + θ₂ = 90) → (θ₁ = 3 * θ₂) → (|θ₁ - θ₂| = 45) :=
by
  intros θ₁ θ₂ h₁ h₂
  sorry

end complementary_angles_positive_difference_l758_758765


namespace lcm_of_9_12_18_l758_758348

-- Let's declare the numbers involved
def num1 : ℕ := 9
def num2 : ℕ := 12
def num3 : ℕ := 18

-- Define what it means for a number to be the LCM of num1, num2, and num3
def is_lcm (a b c l : ℕ) : Prop :=
  l % a = 0 ∧ l % b = 0 ∧ l % c = 0 ∧
  ∀ m, (m % a = 0 ∧ m % b = 0 ∧ m % c = 0) → l ≤ m

-- Now state the theorem
theorem lcm_of_9_12_18 : is_lcm num1 num2 num3 36 :=
by
  sorry

end lcm_of_9_12_18_l758_758348


namespace series_sum_eq_half_l758_758460

theorem series_sum_eq_half : (∑' n : ℕ, 1 ≤ n → ℚ, (3^n) / (9^n - 1)) = 1 / 2 := by
  sorry

end series_sum_eq_half_l758_758460


namespace impossible_to_divide_into_three_similar_parts_l758_758639

-- Define similarity condition
def similar_sizes (x y : ℝ) : Prop := x ≤ √2 * y

-- Main proof problem
theorem impossible_to_divide_into_three_similar_parts (x : ℝ) :
  ¬ ∃ (a b c : ℝ), a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes c a := 
sorry

end impossible_to_divide_into_three_similar_parts_l758_758639


namespace impossible_divide_into_three_similar_l758_758676

noncomputable def sqrt2 : ℝ := Real.sqrt 2

def similar (x y : ℝ) : Prop :=
  x ≤ sqrt2 * y

theorem impossible_divide_into_three_similar (N : ℝ) :
  ¬ ∃ (x y z : ℝ), x + y + z = N ∧ similar x y ∧ similar y z ∧ similar x z := 
by
  sorry

end impossible_divide_into_three_similar_l758_758676


namespace smallest_x_l758_758920

theorem smallest_x (x : ℕ) (M : ℕ) (h : 1800 * x = M^3) :
  x = 30 :=
by
  sorry

end smallest_x_l758_758920


namespace integral_x_squared_l758_758956

theorem integral_x_squared :
  (∀ (a : ℝ), (∃ (h : a^5 = 1),
  (let coeff := 6 * a^5 * (sqrt 3 / 6) in coeff = sqrt 3) → (∫ x in 0..a, x^2) = 1 / 3)) :=
begin
  sorry
end

end integral_x_squared_l758_758956


namespace minimum_reciprocal_sum_l758_758175

theorem minimum_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 
  (∃ z : ℝ, (∀ x y : ℝ, 0 < x → 0 < y → x + 2 * y = 1 → z ≤ (1 / x + 2 / y)) ∧ z = 35 / 6) :=
  sorry

end minimum_reciprocal_sum_l758_758175


namespace hyperbola_eccentricity_l758_758112

open Real

-- Conditions from the problem
variables (F₁ F₂ P : Point) (C : ℝ) (a : ℝ)
variables (angle_F1PF2 : angle F₁ P F₂ = 60)
variables (distance_PF1_PF2 : dist P F₁ = 3 * dist P F₂)
variables (focus_condition : 2 * C = dist F₁ F₂)

-- Statement of the problem
theorem hyperbola_eccentricity:
  let e := C / a in
  e = sqrt 7 / 2 :=
sorry

end hyperbola_eccentricity_l758_758112


namespace section_from_cow_ovary_l758_758802

-- Define the chromosome conditions
-- c_1: The number of chromosomes in some cells is half that of somatic cells.
-- c_2: The number of chromosomes in other cells is the same as that of somatic cells.

def chromosome_condition (c_1 c_2 : Type) : Prop :=
  ∃ (x : c_1) (y : c_2), (number_of_chromosomes x = 0.5 * number_of_somatic_chromosomes) ∧
                         (number_of_chromosomes y = number_of_somatic_chromosomes)

-- The proof statement: Given above conditions, the section comes from Cow’s ovary (B)
theorem section_from_cow_ovary (c_1 c_2 : Type) (number_of_chromosomes number_of_somatic_chromosomes : c_1 → ℝ) : 
  (number_of_chromosomes : c_2 → ℝ) →
  chromosome_condition c_1 c_2 → section_is "B" :=
sorry

end section_from_cow_ovary_l758_758802


namespace sampling_incorrect_description_A_l758_758889

theorem sampling_incorrect_description_A :
  ¬ (∀ (individual : Type) (n : ℕ), simple_sampling individual → probability_of_being_sampled individual n > probability_of_being_sampled individual (n + 1)) :=
by sorry

-- Definitions as used in the conditions above
def simple_sampling (individual : Type) : Prop := 
∀ (individual : Type), probability_of_being_sampled individual = constant_probability

def systematic_sampling (individual : Type) : Prop := 
∀ (individual : Type), equal_interval_sampling individual

def stratified_sampling (individual : Type) : Prop := 
∀ (individual : Type), sample_within_stratum individual = equal_probability

def sampling_principle (individual : Type) : Prop :=
stir_evenly individual ∧ sample_each_individual_with_equal_probability individual

end sampling_incorrect_description_A_l758_758889


namespace count_valid_digits_l758_758044

theorem count_valid_digits :
  {n : ℕ | 1 ≤ n ∧ n ≤ 9 ∧ 15 * n % n = 0}.card = 7 :=
by sorry

end count_valid_digits_l758_758044


namespace total_boys_school_l758_758596

variable (B : ℕ)
variables (percMuslim percHindu percSikh boysOther : ℕ)

-- Defining the conditions
def condition1 : percMuslim = 44 := by sorry
def condition2 : percHindu = 28 := by sorry
def condition3 : percSikh = 10 := by sorry
def condition4 : boysOther = 54 := by sorry

-- Main theorem statement
theorem total_boys_school (h1 : percMuslim = 44) (h2 : percHindu = 28) (h3 : percSikh = 10) (h4 : boysOther = 54) : 
  B = 300 := by sorry

end total_boys_school_l758_758596


namespace can_form_triangle_triangle_inequality_problem_l758_758409

theorem can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_inequality_problem :
  ¬ can_form_triangle 1 2 3 ∧
  ¬ can_form_triangle 2 2 4 ∧
  can_form_triangle 3 4 5 ∧
  ¬ can_form_triangle 3 5 9 :=
by
  have hA := not_and_of_not_right (1 + 2 > 3) (by simp)
  have hB := not_and_of_not_right (2 + 2 > 4) (by simp)
  have hC := and.intro (by simp) (and.intro (by simp) (by simp))
  have hD := not_and_of_not_right (3 + 5 > 9) (by simp)
  exact ⟨hA, hB, hC, hD⟩

end can_form_triangle_triangle_inequality_problem_l758_758409


namespace divide_pile_l758_758677

theorem divide_pile (pile : ℝ) (similar : ℝ → ℝ → Prop) :
  (∀ x y, similar x y ↔ x ≤ y * Real.sqrt 2 ∧ y ≤ x * Real.sqrt 2) →
  ¬∃ a b c, a + b + c = pile ∧ similar a b ∧ similar b c ∧ similar a c :=
by sorry

end divide_pile_l758_758677


namespace arithmetic_sequence_probability_l758_758420

def favorable_sequences : List (List ℕ) :=
  [[1, 2, 3], [1, 3, 5], [2, 3, 4], [2, 4, 6], [3, 4, 5], [4, 5, 6], 
   [3, 2, 1], [5, 3, 1], [4, 3, 2], [6, 4, 2], [5, 4, 3], [6, 5, 4], 
   [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]]

def total_outcomes : ℕ := 216
def favorable_outcomes : ℕ := favorable_sequences.length

theorem arithmetic_sequence_probability : (favorable_outcomes : ℚ) / total_outcomes = 1 / 12 := by
  sorry

end arithmetic_sequence_probability_l758_758420


namespace min_cardinality_intersection_l758_758894

-- Given sets A, B, and C
variables (A B C : Set)

-- Given conditions
axiom cond1 : cardinality (A) + cardinality (B) + cardinality (C) = cardinality (A ∪ B ∪ C)
axiom cond2 : cardinality (A) = 50
axiom cond3 : cardinality (B) = 60
axiom cond4 : cardinality (A ∩ B) = 25

-- Proof statement
theorem min_cardinality_intersection : min_cardinality_intersection (A ∩ B ∩ C) = 25 :=
sorry

end min_cardinality_intersection_l758_758894


namespace shortest_distance_l758_758780

-- Definition of the parabola y = x^2
def on_parabola (p : ℝ × ℝ) : Prop := p.2 = p.1^2

-- Definition of the line 2x - y = 4
def on_line (p : ℝ × ℝ) : Prop := 2 * p.1 - p.2 = 4

-- Distance formula from a point (x0, y0) to a line Ax + By + C = 0
noncomputable def distance_to_line (p : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  abs (A * p.1 + B * p.2 + C) / real.sqrt (A^2 + B^2)

-- The shortest distance from a point on y = x^2 to the line 2x - y = 4
theorem shortest_distance : 
  ∃ p : ℝ × ℝ, on_parabola p ∧ (distance_to_line p 2 (-1) (-4) = 3 * real.sqrt 5 / 5) :=
sorry

end shortest_distance_l758_758780


namespace fraction_of_males_on_time_l758_758896

open Real

variable (A : ℝ) (M : ℝ)

theorem fraction_of_males_on_time
  (h1 : (3/5) * A + (2/5) * A = A)
  (h2 : (9/10) * (2/5) * A + M * (3/5) * A = 0.885 * A) :
  M = 0.875 :=
by 
  have h3 : (3 / 5 + 2 / 5) * A = A := by sorry -- This simplifies from h1
  have h4 : (3 / 5 * M + 0.36 * 2 / 5 = 0.885) := by sorry -- This converts the equation to a simpler form
  have h5 : (3 * M + 1.8 = 4.425) := by sorry -- Another form simplification
  have h6 : (3 * M = 2.625) := by sorry -- Isolate the term related to M
  have h7 : (M = 0.875) := by sorry -- Solve for M
  exact h7 -- Complete the proof

end fraction_of_males_on_time_l758_758896


namespace find_angle_BXY_l758_758608

noncomputable def angle_AXE (angle_CYX : ℝ) : ℝ := 3 * angle_CYX - 108

theorem find_angle_BXY
  (AB_parallel_CD : Prop)
  (h_parallel : ∀ (AXE CYX : ℝ), angle_AXE CYX = AXE)
  (x : ℝ) :
  (angle_AXE x = x) → x = 54 :=
by
  intro h₁
  unfold angle_AXE at h₁
  sorry

end find_angle_BXY_l758_758608


namespace exists_digit_sum_div_11_in_39_succ_nums_l758_758276

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_digit_sum_div_11_in_39_succ_nums (n : ℕ) :
  ∃ k, k ∈ list.range' n 39 ∧ digit_sum k % 11 = 0 :=
by
  -- The proof would go here
  sorry

end exists_digit_sum_div_11_in_39_succ_nums_l758_758276


namespace caterpillars_left_on_tree_l758_758324

-- Definitions based on conditions
def initialCaterpillars : ℕ := 14
def hatchedCaterpillars : ℕ := 4
def caterpillarsLeftToCocoon : ℕ := 8

-- The proof problem statement in Lean
theorem caterpillars_left_on_tree : initialCaterpillars + hatchedCaterpillars - caterpillarsLeftToCocoon = 10 :=
by
  -- solution steps will go here eventually
  sorry

end caterpillars_left_on_tree_l758_758324


namespace abs_inequality_solution_l758_758031

theorem abs_inequality_solution (x : ℝ) : 
  3 < |x + 2| ∧ |x + 2| ≤ 6 ↔ (1 < x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x < -5) := 
by
  sorry

end abs_inequality_solution_l758_758031


namespace find_k_l758_758186

variable (k : ℝ) (t : ℝ) (a : ℝ)

theorem find_k (h1 : t = (5 / 9) * (k - 32) + a * k) (h2 : t = 20) (h3 : a = 3) : k = 10.625 := by
  sorry

end find_k_l758_758186


namespace smallest_number_with_exactly_eight_factors_l758_758830

theorem smallest_number_with_exactly_eight_factors
    (n : ℕ)
    (h1 : ∃ a b : ℕ, (a + 1) * (b + 1) = 8)
    (h2 : ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ n = p^a * q^b) : 
    n = 24 := by
  sorry

end smallest_number_with_exactly_eight_factors_l758_758830


namespace acute_triangle_sin_sum_gt_2_l758_758727

open Real

theorem acute_triangle_sin_sum_gt_2 (α β γ : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) (h_sum : α + β + γ = π) :
  sin α + sin β + sin γ > 2 :=
sorry

end acute_triangle_sin_sum_gt_2_l758_758727


namespace num_digits_divisible_l758_758063

theorem num_digits_divisible (h : ∀ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}, divides n (150 + n)) :
  {n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} | divides n (150 + n)}.card = 5 := 
sorry

end num_digits_divisible_l758_758063


namespace impossibility_of_dividing_into_three_similar_piles_l758_758656

theorem impossibility_of_dividing_into_three_similar_piles:
  ∀ (x y z : ℝ), ¬ (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x + y + z = 1  ∧ (x ≤ sqrt 2 * y ∧ y ≤ sqrt 2 * x) ∧ (y ≤ sqrt 2 * z ∧ z ≤ sqrt 2 * y) ∧ (z ≤ sqrt 2 * x ∧ x ≤ sqrt 2 * z)) :=
by
  sorry

end impossibility_of_dividing_into_three_similar_piles_l758_758656


namespace log8_50_equals_l758_758930

theorem log8_50_equals : log 8 50 = (1 + 2 * log 2 5) / 3 :=
by
  sorry

end log8_50_equals_l758_758930


namespace martha_clothes_total_l758_758715

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

end martha_clothes_total_l758_758715


namespace sin_double_angle_15_eq_half_l758_758374

theorem sin_double_angle_15_eq_half : 2 * Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) = 1 / 2 := 
sorry

end sin_double_angle_15_eq_half_l758_758374


namespace five_lines_properties_l758_758925

-- Define the basic assumptions and setup for the problem.
def five_lines := {l : set (set ℝ × set ℝ) | ∃ l1 l2 l3 l4 l5 : set (set ℝ × set ℝ),
  l1 ≠ l2 ∧ l1 ≠ l3 ∧ l1 ≠ l4 ∧ l1 ≠ l5 ∧ l2 ≠ l3 ∧ l2 ≠ l4 ∧ l2 ≠ l5 ∧ l3 ≠ l4 ∧ l3 ≠ l5 ∧ l4 ≠ l5 ∧
  (∀ (i j : set (set ℝ × set ℝ)), i ∈ {l1, l2, l3, l4, l5} ∧ j ∈ {l1, l2, l3, l4, l5} → i ≠ j →
     ∃! p : set ℝ × set ℝ, p ∈ i ∧ p ∈ j) ∧ l = {l1, l2, l3, l4, l5} }

-- State the theorem for the given problem.
theorem five_lines_properties (l : set (set (set ℝ × set ℝ))) (h : l = five_lines) :
  ∃ n_points n_points_per_line n_segments max_isosceles_tris,
    n_points = 10 ∧
    n_points_per_line = 4 ∧
    n_segments = 30 ∧
    max_isosceles_tris = 10 :=
sorry

end five_lines_properties_l758_758925


namespace intersection_eq_l758_758523

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_eq : A ∩ B = ({-2, 2} : Set ℤ) :=
by
  sorry

end intersection_eq_l758_758523


namespace multiple_of_larger_number_l758_758313

variables (S L M : ℝ)

-- Conditions
def small_num := S = 10.0
def sum_eq := S + L = 24
def multiplication_relation := 7 * S = M * L

-- Theorem statement
theorem multiple_of_larger_number (S L M : ℝ) 
  (h1 : small_num S) 
  (h2 : sum_eq S L) 
  (h3 : multiplication_relation S L M) : 
  M = 5 := by
  sorry

end multiple_of_larger_number_l758_758313


namespace arc_length_of_curve_l758_758418

open Real

noncomputable def arc_length : ℝ :=
  let ρ (φ : ℝ) := 2 * exp (4 * φ / 3)
  let dρ_dφ (φ : ℝ) := (8 / 3) * exp (4 * φ / 3)
  ∫ φ in -π / 2..π / 2, sqrt ((ρ φ) ^ 2 + (dρ_dφ φ) ^ 2)

theorem arc_length_of_curve :
  arc_length = 5 * sinh (2 * π / 3) := by
  sorry

end arc_length_of_curve_l758_758418


namespace minor_arc_LB_correct_l758_758593

-- Define the measure of angle LBU
def angle_LBU : ℝ := 58 

-- Define a function calculating the measure of minor arc LB
def minor_arc_LB (angle_LBU : ℝ) : ℝ :=
  let major_arc_BU := 2 * angle_LBU in
  180 - major_arc_BU

theorem minor_arc_LB_correct : minor_arc_LB angle_LBU = 64 := by
  -- Proof skipped with sorry
  sorry

end minor_arc_LB_correct_l758_758593


namespace fourth_vertex_of_parallelogram_l758_758168

structure Point where
  x : ℤ
  y : ℤ

def midPoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

def isMidpoint (M P Q : Point) : Prop :=
  M = midPoint P Q

theorem fourth_vertex_of_parallelogram (A B C D : Point)
  (hA : A = {x := -2, y := 1})
  (hB : B = {x := -1, y := 3})
  (hC : C = {x := 3, y := 4})
  (h1 : isMidpoint (midPoint A C) B D ∨
        isMidpoint (midPoint A B) C D ∨
        isMidpoint (midPoint B C) A D) :
  D = {x := 2, y := 2} ∨ D = {x := -6, y := 0} ∨ D = {x := 4, y := 6} := by
  sorry

end fourth_vertex_of_parallelogram_l758_758168


namespace asymptote_problem_l758_758760

-- Definitions for the problem
def r (x : ℝ) : ℝ := -3 * (x + 2) * (x - 1)
def s (x : ℝ) : ℝ := (x + 2) * (x - 4)

-- Assertion to prove
theorem asymptote_problem : r (-1) / s (-1) = 6 / 5 :=
by {
  -- This is where the proof would be carried out
  sorry
}

end asymptote_problem_l758_758760


namespace exists_balanced_set_balanced_and_centerless_set_l758_758469

def balanced_set (S : Finset ℝ) : Prop :=
  ∀ (A B ∈ S), A ≠ B → ∃ C ∈ S, (abs (A - C) = abs (B - C))

def centerless_set (S : Finset ℝ) : Prop :=
  ∀ (A B C ∈ S), A ≠ B ∧ B ≠ C ∧ A ≠ C → ¬∃ P ∈ S, (abs (A - P) = abs (B - P) ∧ abs (B - P) = abs (C - P))

-- Problem (1)
theorem exists_balanced_set (n : ℕ) (h : n ≥ 3) : ∃ S : Finset ℝ, S.card = n ∧ balanced_set S := 
sorry

-- Problem (2)
theorem balanced_and_centerless_set (n : ℕ) (h : n ≥ 3) : 
  (∃ S : Finset ℝ, S.card = n ∧ balanced_set S ∧ centerless_set S) ↔ (odd n) := 
sorry

end exists_balanced_set_balanced_and_centerless_set_l758_758469


namespace focus_of_parabola_l758_758023

theorem focus_of_parabola (x f d : ℝ) (h : ∀ x, y = 4 * x^2 → (x = 0 ∧ y = f) → PF^2 = PQ^2 ∧ 
(PF^2 = x^2 + (4 * x^2 - f) ^ 2) := (PQ^2 = (4 * x^2 - d) ^ 2)) : 
  f = 1 / 16 := 
sorry

end focus_of_parabola_l758_758023


namespace problem_l758_758239

theorem problem (a b : ℝ) : a^6 + b^6 ≥ a^4 * b^2 + a^2 * b^4 := 
by sorry

end problem_l758_758239


namespace solve_for_k_l758_758857

noncomputable def find_k (k : ℕ) : Prop :=
  let total_balls := 8 + k
  let probability_green := 8 / total_balls
  let probability_purple := k / total_balls
  let expected_value := (probability_green * 5) + (probability_purple * (-3))
  expected_value = 1

theorem solve_for_k : ∃ k : ℕ, k ≠ 0 ∧ find_k k := by
  use 8
  split
  -- Prove that 8 is a positive integer
  exact Nat.succ_ne_zero 7
  -- Prove that expected value satisfies the condition
  sorry

end solve_for_k_l758_758857


namespace part1_geometric_sequence_part2_sum_sequence_b_l758_758779

noncomputable def sequence_a (λ : ℝ) : ℕ → ℝ
| 0       := 2
| (n + 1) := λ * sequence_a n + 2^n

-- Part 1: Proving λ = 1 makes {a_n} a geometric sequence and a_n = 2^n
def is_geometric (s : ℕ → ℝ) : Prop :=
∀ n m, n ≠ m → (s n ≠ 0 ∧ s m ≠ 0) → s (n+1) / s n = s (m+1) / s m

theorem part1_geometric_sequence (λ : ℝ) : 
  (∃ (u : ℕ → ℝ), is_geometric u ∧ u 0 = 2 ∧ ∀ n, u (n + 1) = λ * u n + 2^n) ↔ λ = 1 ∧ (∀ n, sequence_a 1 n = 2 ^ n) :=
sorry

-- Part 2: Sum of first n terms of the sequence {b_n} when λ = 2.
noncomputable def sequence_b (n : ℕ) : ℝ :=
sequence_a 2 n / 2^n

theorem part2_sum_sequence_b (n : ℕ) : 
  ∑ i in Finset.range n, sequence_b i = n * (n + 3) / 4 :=
sorry

end part1_geometric_sequence_part2_sum_sequence_b_l758_758779


namespace length_AE_circle_O_l758_758423

theorem length_AE_circle_O 
  (O : Type*) [metric_space O]
  (A B C D E : O)
  (r : ℝ)
  (h_circle_O : ∀ (P : O), dist O P = r ↔ P = A ∨ P = B ∨ P = D ∨ P = E)
  (h_radius_O : r = 2)
  (h_diameter_AB : dist A B = 2 * r)
  (h_triangle_right : ∠ACB = π / 2 ∧ ∠ABC = π / 3)
  (h_intersection_D : D = intersection (circle O r) (line AC))
  (h_intersection_E : E = intersection (circle O r) (line BC))
  :
  dist A E = 2 :=
sorry

end length_AE_circle_O_l758_758423


namespace cyclic_O_K_P_Q_l758_758753

-- Definitions of the given problem
variables {Point : Type} [Nonempty Point] [AffineSpace Point] [MetricSpace Point]

variables (A B C D O K L M P Q : Point)
variables (S1 S2 : set Point)
variable (circumcircle : ∀ (A B C O : Point), set Point)

-- Conditions given in the proof problem
def is_inscribed_quadrilateral (A B C D O : Point) : Prop :=
  let cyclic_quadrilateral := true in  -- A placeholder for the condition that ABCD is inscribed
  let diagonals_intersect_at := O in  -- The diagonals of ABCD intersect at O
  cyclic_quadrilateral ∧ diagonals_intersect_at = O

def are_circumcircles (S1 S2 : set Point) (ABO CDO : Point) : Prop :=
  circumcircle A B O = S1 ∧ circumcircle C D O = S2

def second_intersection (S1 S2 : set Point) (K : Point) : Prop :=
  K ∈ S1 ∧ K ∈ S2 ∧ (K ≠ A ∧ K ≠ B ∧ K ≠ C ∧ K ≠ D)

def lines_parallel_to_sides (O AB CD : Point) (L M : Point) (S1 S2 : set Point) : Prop :=
  L ∈ S1 ∧ M ∈ S2 ∧ (∀ line, (Parallel line O AB ↔ line intersects S1 at L) ∧ (Parallel line O CD ↔ line intersects S2 at M))

def points_on_segments (O L M P Q : Point) : Prop :=
  (P ∈ segment O L) ∧ (Q ∈ segment O M) ∧ (distance O P / distance P L = distance M Q / distance Q O)

-- Goal statement
theorem cyclic_O_K_P_Q : 
  is_inscribed_quadrilateral A B C D O →
  are_circumcircles S1 S2 ABO CDO →
  second_intersection S1 S2 K →
  lines_parallel_to_sides O (line_through_points A B) (line_through_points C D) L M S1 S2 →
  points_on_segments O L M P Q →
  cyclic_points O K P Q :=
sorry

end cyclic_O_K_P_Q_l758_758753


namespace Lyle_percentage_of_chips_l758_758266

theorem Lyle_percentage_of_chips (total_chips : ℕ) (ratio_Ian_Lyle : ℕ × ℕ) (h_total_chips : total_chips = 100) (h_ratio : ratio_Ian_Lyle = (4, 6)) :
  let total_parts := ratio_Ian_Lyle.1 + ratio_Ian_Lyle.2 in
  let chips_per_part := total_chips / total_parts in
  let Lyle_chips := ratio_Ian_Lyle.2 * chips_per_part in
  let percentage_Lyle := (Lyle_chips * 100) / total_chips in
  percentage_Lyle = 60 :=
by
  intros
  sorry

end Lyle_percentage_of_chips_l758_758266


namespace series_equals_one_half_l758_758465

noncomputable def series_sum : ℕ → ℚ
| k := 3^k / (9^k - 1)

theorem series_equals_one_half :
  ∑' k, series_sum k = 1 / 2 :=
sorry

end series_equals_one_half_l758_758465


namespace recurring_decimal_to_fraction_l758_758444

theorem recurring_decimal_to_fraction :
  (0.3 + (λ x : ℝ, x = (coeff x)^3) = (109 / 330))
:= sorry

end recurring_decimal_to_fraction_l758_758444


namespace option_c_option_d_l758_758785

-- Definitions and conditions
def Sn (n : ℕ) : ℤ := -(n^2 : ℤ) + 7 * n

-- Proving the options
theorem option_c (n : ℕ) (h : n > 4) : 
  let a_n := Sn n - Sn (n - 1)
  in a_n < 0 :=
by
  sorry

theorem option_d : 
  ∀ n : ℕ, (Sn 3 ≥ Sn n ∧ Sn 4 ≥ Sn n) :=
by
  sorry

end option_c_option_d_l758_758785


namespace trent_total_distance_l758_758339

-- Conditions
def walked_to_bus_stop : ℕ := 4
def bus_ride_to_library : ℕ := 7
def total_distance_to_library : ℕ := walked_to_bus_stop + bus_ride_to_library
def distance_back_home : ℕ := total_distance_to_library

-- Theorem stating that the total distance Trent traveled is 22 blocks
theorem trent_total_distance : 
  let total_distance := total_distance_to_library + distance_back_home in
  total_distance = 22 := 
by
  sorry

end trent_total_distance_l758_758339


namespace division_remainder_l758_758948

def P (x : ℝ) : ℝ := x^6 - x^5 - x^4 + x^3 + x^2 - x
def Q (x : ℝ) : ℝ := (x^2 - 1) * (x - 2)
def R (x : ℝ) : ℝ := (17/2) * x^2 + (1/2) * x - 9

theorem division_remainder : ∃ (q: ℝ → ℝ), 
  ∀ x : ℝ, P x = Q x * q x + R x := by
  sorry

end division_remainder_l758_758948


namespace least_positive_integer_with_eight_factors_l758_758819

noncomputable def numDivisors (n : ℕ) : ℕ :=
  (List.range (n+1)).count (λ d => d > 0 ∧ n % d = 0)

theorem least_positive_integer_with_eight_factors : ∃ n : ℕ, n > 0 ∧ numDivisors n = 8 ∧ (∀ m : ℕ, m > 0 → numDivisors m = 8 → n ≤ m) := 
  sorry

end least_positive_integer_with_eight_factors_l758_758819


namespace parabola_translation_l758_758331

-- Definitions for the conditions (original parabola and transformations)
def original_parabola (x : ℝ) : ℝ := x^2

def translate_right (y : ℝ) (x : ℝ) : ℝ := y(x - 1)
def translate_up (y : ℝ) (units : ℝ) : ℝ := y + units

-- Proof statement that combines the conditions and the question to get the answer
theorem parabola_translation :
  ∀ (x : ℝ), translate_up (translate_right original_parabola x) 2 = (x - 1)^2 + 2 :=
by
  -- Here we'd provide the proof steps
  sorry

end parabola_translation_l758_758331


namespace martha_total_clothes_l758_758717

-- Define the conditions
def jackets_bought : ℕ := 4
def t_shirts_bought : ℕ := 9
def free_jacket_condition : ℕ := 2
def free_t_shirt_condition : ℕ := 3

-- Define calculations based on conditions
def free_jackets : ℕ := jackets_bought / free_jacket_condition
def free_t_shirts : ℕ := t_shirts_bought / free_t_shirt_condition
def total_jackets := jackets_bought + free_jackets
def total_t_shirts := t_shirts_bought + free_t_shirts
def total_clothes := total_jackets + total_t_shirts

-- Prove the total number of clothes
theorem martha_total_clothes : total_clothes = 18 :=
by
    sorry

end martha_total_clothes_l758_758717


namespace Ron_needs_to_drink_80_percent_l758_758851

theorem Ron_needs_to_drink_80_percent 
  (volume_each : ℕ)
  (volume_intelligence : ℕ)
  (volume_beauty : ℕ)
  (volume_strength : ℕ)
  (volume_second_pitcher : ℕ)
  (effective_volume : ℕ)
  (volume_intelligence_left : ℕ)
  (volume_beauty_left : ℕ)
  (volume_strength_left : ℕ)
  (total_volume : ℕ)
  (Ron_needs : ℕ)
  (intelligence_condition : effective_volume = 30)
  (initial_volumes : volume_each = 300)
  (first_drink : volume_intelligence = volume_each / 2)
  (mix_before_second_drink : volume_second_pitcher = volume_intelligence + volume_beauty)
  (Hermione_drink : volume_second_pitcher / 2 = volume_intelligence_left + volume_beauty_left)
  (Harry_drink : volume_strength_left = volume_each / 2)
  (second_mix : volume_second_pitcher = volume_intelligence_left + volume_beauty_left + volume_strength_left)
  (final_mix : volume_second_pitcher / 2 = volume_intelligence_left + volume_beauty_left + volume_strength_left)
  (Ron_needs_condition : Ron_needs = effective_volume / volume_intelligence_left * 100)
  : Ron_needs = 80 := sorry

end Ron_needs_to_drink_80_percent_l758_758851


namespace diagonal_length_of_square_with_area_50_l758_758808

-- Define the side length of a square with area 50 square meters
def side_length_of_square : ℝ := real.sqrt 50

-- Define the diagonal of a square with side length s
def diagonal_of_square (s : ℝ) : ℝ := real.sqrt (2 * s^2)

-- Proof that the diagonal length of a square with area 50 is 10 meters
theorem diagonal_length_of_square_with_area_50 :
  diagonal_of_square side_length_of_square = 10 :=
sorry

end diagonal_length_of_square_with_area_50_l758_758808


namespace divide_pile_l758_758678

theorem divide_pile (pile : ℝ) (similar : ℝ → ℝ → Prop) :
  (∀ x y, similar x y ↔ x ≤ y * Real.sqrt 2 ∧ y ≤ x * Real.sqrt 2) →
  ¬∃ a b c, a + b + c = pile ∧ similar a b ∧ similar b c ∧ similar a c :=
by sorry

end divide_pile_l758_758678


namespace find_focus_of_parabola_4x2_l758_758018

-- Defining what it means to be the focus of a parabola given an equation y = ax^2.
def is_focus (a : ℝ) (f : ℝ) : Prop :=
  ∀ (x : ℝ), (x ^ 2 + (a * x ^ 2 - f) ^ 2) = ((a * x ^ 2 - (f * 8)) ^ 2)

-- Specific instance of the parabola y = 4x^2.
def parabola_4x2 := (4 : ℝ)

theorem find_focus_of_parabola_4x2 : ∃ f : ℝ, is_focus parabola_4x2 f :=
begin
  use (1/16 : ℝ),
  sorry -- The proof will be filled in by the theorem prover.
end

end find_focus_of_parabola_4x2_l758_758018


namespace parallel_planes_l758_758215

noncomputable def vec_axb : (A B C : ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ)
| (a1, a2, a3), (b1, b2, b3), (c1, c2, c3) := 
  ((b1 - a1) * (c2 - a2) - (b2 - a2) * (c1 - a1), (b2 - a2) * (c3 - a3) - (b3 - a3) * (c2 - a2), (b3 - a3) * (c1 - a1) - (b1 - a1) * (c3 - a3))

theorem parallel_planes (A B C : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) (hA : A = (0, 1, 0)) (hB : B = (1, 1, 1)) (hC : C = (0, 2, 1)) (hn : n = (2, 2, -2)) :
  ∃ k : ℝ, vec_axb A B C = (x, y, z) ∧ n = (k * x, k * y, k * z) :=
sorry

end parallel_planes_l758_758215


namespace probability_of_even_on_8_sided_die_l758_758359

theorem probability_of_even_on_8_sided_die (fair_die : Prop) (sides : ℕ) 
  (even_count : ℕ) (total_outcomes : ℕ) (h1 : sides = 8) 
  (h2 : even_count = 4) (h3 : total_outcomes = sides) : 
  even_count / total_outcomes = 1 / 2 :=
by {
  rw [h1, h2, h3],
  norm_num,
}

end probability_of_even_on_8_sided_die_l758_758359


namespace circle_area_with_diameter_CD_l758_758271

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem circle_area_with_diameter_CD (C D E : ℝ × ℝ)
  (hC : C = (-1, 2)) (hD : D = (5, -6)) (hE : E = (2, -2))
  (hE_midpoint : E = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  ∃ (A : ℝ), A = 25 * Real.pi :=
by
  -- Define the coordinates of points C and D
  let Cx := -1
  let Cy := 2
  let Dx := 5
  let Dy := -6

  -- Calculate the distance (diameter) between C and D
  let diameter := distance Cx Cy Dx Dy

  -- Calculate the radius of the circle
  let radius := diameter / 2

  -- Calculate the area of the circle
  let area := Real.pi * radius^2

  -- Prove the area is 25π
  use area
  sorry

end circle_area_with_diameter_CD_l758_758271


namespace least_positive_integer_with_eight_factors_l758_758816

noncomputable def numDivisors (n : ℕ) : ℕ :=
  (List.range (n+1)).count (λ d => d > 0 ∧ n % d = 0)

theorem least_positive_integer_with_eight_factors : ∃ n : ℕ, n > 0 ∧ numDivisors n = 8 ∧ (∀ m : ℕ, m > 0 → numDivisors m = 8 → n ≤ m) := 
  sorry

end least_positive_integer_with_eight_factors_l758_758816


namespace no_consecutive_product_l758_758906

theorem no_consecutive_product (n : ℕ) : ¬ ∃ k : ℕ, n^2 + 7n + 8 = k * (k + 1) := 
sorry

end no_consecutive_product_l758_758906


namespace unique_tiling_min_k_l758_758535

theorem unique_tiling_min_k {n : ℕ} (h_pos : n > 0) :
  ∃ k, k = 2 * n ∧ ∀ (markings : list (ℕ × ℕ)), 
    (∀ mark ∈ markings, mark.1 < 2 * n ∧ mark.2 < 2 * n) →
    unique_tiling (2 * n) (2 * n) (markings.length = k) markings ∧
    ¬ any_domino_covers_two_marks markings ∧
    (∀ k' < k, ¬ unique_tiling (2 * n) (2 * n) (markings.length = k') markings) :=
begin
  sorry
end

end unique_tiling_min_k_l758_758535


namespace no_division_into_three_similar_piles_l758_758666

theorem no_division_into_three_similar_piles :
    ∀ (x : ℝ),
    ∀ (y z : ℝ),
    (x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = x) →
    (x <= sqrt 2 * y ∧ y <= sqrt 2 * z ∧ z <= sqrt 2 * x) →
    false :=
by
  intro x y z
  sorry

end no_division_into_three_similar_piles_l758_758666


namespace age_of_other_replaced_man_l758_758295

theorem age_of_other_replaced_man (A B C D : ℕ) (h1 : A = 23) (h2 : ((52 + C + D) / 4 > (A + B + C + D) / 4)) :
  B < 29 := 
by
  sorry

end age_of_other_replaced_man_l758_758295


namespace sum_interior_angles_l758_758751

theorem sum_interior_angles (n : ℕ) (h : 180 * (n - 2) = 3240) : 180 * ((n + 3) - 2) = 3780 := by
  sorry

end sum_interior_angles_l758_758751


namespace max_value_expr_l758_758620

theorem max_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 3) :
  (∀ a b c, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 3 → 
    ∑ cyclic [a, b, c], a / (a^3 + b^2 + c) ≤ 1) ∧ 
  (1 <= ∑ cyclic [a, b, c], a / (a^3 + b^2 + c) → 
    (exists a b c, a = 1 ∧ b = 1 ∧ c = 1 ∧ a + b + c = 3))
:= sorry

end max_value_expr_l758_758620


namespace sum_sine_sequence_l758_758364

theorem sum_sine_sequence (n : ℕ) (α : ℝ) : 
  (Finset.sum (Finset.range n.succ) (λ k, Real.sin ((k+1) * α))) = 
  (Real.sin ((n * α) / 2) * Real.sin (((n + 1) * α) / 2)) / Real.sin (α / 2) :=
sorry

end sum_sine_sequence_l758_758364


namespace martha_total_clothes_l758_758711

def jackets_purchased : ℕ := 4
def tshirts_purchased : ℕ := 9
def jackets_free : ℕ := jackets_purchased / 2
def tshirts_free : ℕ := tshirts_purchased / 3
def total_jackets : ℕ := jackets_purchased + jackets_free
def total_tshirts : ℕ := tshirts_purchased + tshirts_free

theorem martha_total_clothes : total_jackets + total_tshirts = 18 := by
  sorry

end martha_total_clothes_l758_758711


namespace incorrect_statement_C_l758_758963

theorem incorrect_statement_C (x : ℝ) : 
  (x > -2) → 
  (y = (x + 2)^2 - 1) → 
  ∀ y1 y2, (x1 x2 : ℝ), (x1 > -2) ∧ (x2 > x1) → ((x1 + 2)^2 - 1 < (x2 + 2)^2 - 1) → y1 < y2 :=
sorry

end incorrect_statement_C_l758_758963


namespace min_value_f_l758_758771

noncomputable def f (x : ℝ) : ℝ := (8^x + 5) / (2^x + 1)

theorem min_value_f : ∃ x : ℝ, f x = 3 :=
sorry

end min_value_f_l758_758771


namespace select_numbers_with_sum_713_l758_758321

noncomputable def is_suitable_sum (numbers : List ℤ) : Prop :=
  ∃ subset : List ℤ, subset ⊆ numbers ∧ (subset.sum % 10000 = 713)

theorem select_numbers_with_sum_713 :
  ∀ numbers : List ℤ, 
  numbers.length = 1000 → 
  (∀ n ∈ numbers, n % 2 = 1 ∧ n % 5 ≠ 0) →
  is_suitable_sum numbers :=
sorry

end select_numbers_with_sum_713_l758_758321


namespace max_value_of_expression_l758_758733

theorem max_value_of_expression :
  ∃ x : ℝ, ∀ y : ℝ, -x^2 + 4*x + 10 ≤ -y^2 + 4*y + 10 ∧ -x^2 + 4*x + 10 = 14 :=
sorry

end max_value_of_expression_l758_758733


namespace shape_after_rotation_is_C_l758_758380

-- Define the original T-like shape and the marked center point
def T_shape : Type := sorry -- Placeholder for the actual shape type
def center_point : T_shape := sorry -- Placeholder for the actual center point

-- Define a function that performs a 180-degree rotation on the shape around the center point
noncomputable def rotate_180 (shape : T_shape) (point : T_shape) : T_shape := sorry

-- Define the options as types
def option_A : T_shape := sorry
def option_B : T_shape := sorry
def option_C : T_shape := sorry
def option_D : T_shape := sorry
def option_E : T_shape := sorry

-- State the theorem to be proved
theorem shape_after_rotation_is_C :
  rotate_180 T_shape center_point = option_C :=
sorry -- Proof goes here

end shape_after_rotation_is_C_l758_758380


namespace impossible_to_divide_into_three_similar_parts_l758_758641

-- Define similarity condition
def similar_sizes (x y : ℝ) : Prop := x ≤ √2 * y

-- Main proof problem
theorem impossible_to_divide_into_three_similar_parts (x : ℝ) :
  ¬ ∃ (a b c : ℝ), a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes c a := 
sorry

end impossible_to_divide_into_three_similar_parts_l758_758641


namespace unique_solution_of_functional_eqn_l758_758448

theorem unique_solution_of_functional_eqn (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y - 1) + f x * f y = 2 * x * y - 1) → (∀ x : ℝ, f x = x) :=
by
  intros h
  sorry

end unique_solution_of_functional_eqn_l758_758448


namespace hyperbola_eccentricity_thm_l758_758105

noncomputable def hyperbola_eccentricity 
  (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
  (angle_F1PF2 : ℝ) (dist_PF1 dist_PF2 : ℝ)
  (H1 : angle_F1PF2 = 60)
  (H2 : dist_PF1 = 3 * dist_PF2) : ℝ :=
let a := dist_PF2 in
let c := (a * sqrt 7) / 2 in 
c / a

theorem hyperbola_eccentricity_thm
  (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
  (angle_F1PF2 : ℝ) (dist_PF1 dist_PF2 : ℝ)
  (H1 : angle_F1PF2 = 60)
  (H2 : dist_PF1 = 3 * dist_PF2) :
  @hyperbola_eccentricity F1 F2 P _ _ _ angle_F1PF2 dist_PF1 dist_PF2 H1 H2 = (sqrt 7) / 2 :=
sorry

end hyperbola_eccentricity_thm_l758_758105


namespace intersection_of_A_and_B_l758_758507

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} :=
by
  sorry

end intersection_of_A_and_B_l758_758507


namespace count_valid_digits_l758_758048

theorem count_valid_digits :
  {n : ℕ | 1 ≤ n ∧ n ≤ 9 ∧ 15 * n % n = 0}.card = 7 :=
by sorry

end count_valid_digits_l758_758048


namespace speed_of_current_l758_758397

variable (c : ℚ) -- Speed of the current in miles per hour
variable (d : ℚ) -- Distance to the certain point in miles

def boat_speed := 16 -- Boat's speed relative to water in mph
def upstream_time := (20:ℚ) / 60 -- Time upstream in hours 
def downstream_time := (15:ℚ) / 60 -- Time downstream in hours

theorem speed_of_current (h1 : d = (boat_speed - c) * upstream_time)
                         (h2 : d = (boat_speed + c) * downstream_time) :
    c = 16 / 7 :=
  by
  sorry

end speed_of_current_l758_758397


namespace impossible_divide_into_three_similar_parts_l758_758692

noncomputable def similar (x y : ℝ) : Prop := x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

theorem impossible_divide_into_three_similar_parts (s : ℝ → ℝ → Prop) :
  (∀ s, similar s)) → ¬ (∃ a b c : ℝ, s a b → s b c → s c a → a + b + c = 1) :=
by
  intros h_similar h
  sorry

end impossible_divide_into_three_similar_parts_l758_758692


namespace optimal_vs_suboptimal_l758_758083

theorem optimal_vs_suboptimal (a : Fin 12 → ℕ) 
  (distinct : ∀ i j : Fin 12, i ≠ j → a i ≠ a j) 
  (h_sum : ∑ i in (FinSet.univ : Finset (Fin 12)), a i < 2007) : 
  ∃ k : ℕ, k = ∏ i in (FinSet.range 11), a i := 
sorry

end optimal_vs_suboptimal_l758_758083


namespace checkerboard_divisibility_l758_758623

theorem checkerboard_divisibility (p : ℕ) (hp : p ≥ 5) (prime_p : nat.prime p) :
  let r := (finset.univ.powerset.filter (λ s, s.card = p ∧ ∃ i j, i ≠ j ∧ i ∈ s ∧ j ∈ s ∧ (∀ k ∈ s, k.1 = i.1 ∨ k.1 = j.1))).card in
  p^5 ∣ r :=
begin
  sorry
end

end checkerboard_divisibility_l758_758623


namespace impossibility_of_dividing_into_three_similar_piles_l758_758658

theorem impossibility_of_dividing_into_three_similar_piles:
  ∀ (x y z : ℝ), ¬ (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x + y + z = 1  ∧ (x ≤ sqrt 2 * y ∧ y ≤ sqrt 2 * x) ∧ (y ≤ sqrt 2 * z ∧ z ≤ sqrt 2 * y) ∧ (z ≤ sqrt 2 * x ∧ x ≤ sqrt 2 * z)) :=
by
  sorry

end impossibility_of_dividing_into_three_similar_piles_l758_758658


namespace probability_even_sum_l758_758341

theorem probability_even_sum :
  let balls := Finset.range 1 13
  let all_possibilities := balls.product (balls \ {i}) -- excluding one already drawn
  let favorable_even := (balls.filter (λ i, i % 2 = 0)).product (balls.filter (λ i, i % 2 = 0 \ {i})) ∪
                        (balls.filter (λ i, i % 2 = 1)).product (balls.filter (λ i, i % 2 = 1 \ {i}))
  (favorable_even.card : ℚ) / (all_possibilities.card : ℚ) = 5 / 11 := 
by
  sorry

end probability_even_sum_l758_758341


namespace remainder_S_mod_1000_l758_758234

open Nat

noncomputable def S : ℕ :=
  ∑ n in Finset.range 669, (-1) ^ n * Nat.choose 2004 (3 * n)

theorem remainder_S_mod_1000 : S % 1000 = 6 := 
  by
  sorry

end remainder_S_mod_1000_l758_758234


namespace can_wrap_unit_cube_l758_758220

theorem can_wrap_unit_cube (a b : ℝ) (ha : a = 1) (hb : b = 1) : 
  ∃ (l : ℝ), l = 3 ∧ l * l = 9 ∧ ∀ (c : ℝ), c = (3 * real.sqrt 2) → l >= c :=
by
  sorry

end can_wrap_unit_cube_l758_758220


namespace max_value_of_a_sqrt_1_add_b_squared_l758_758144

theorem max_value_of_a_sqrt_1_add_b_squared (a b : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a^2 + b^2 / 2 = 1) : 
  a * sqrt (1 + b^2) ≤ 3 * sqrt 2 / 4 :=
sorry

end max_value_of_a_sqrt_1_add_b_squared_l758_758144


namespace irrigation_system_flow_rates_l758_758966

-- Define the conditions
variable (q0 : ℝ) -- Flow rate in channel BC

-- Channels' flow rates
variable (qAB qAH q_total : ℝ)

-- Define the conditions as hypotheses
axiom H1 : qAB = 1/2 * q0
axiom H2 : qAH = 3/4 * q0
axiom H3 : q_total = qAB + qAH

-- Prove the results
theorem irrigation_system_flow_rates
  (q0 : ℝ)
  (qAB qAH q_total : ℝ)
  (H1 : qAB = 1/2 * q0)
  (H2 : qAH = 3/4 * q0)
  (H3 : q_total = qAB + qAH) :
  qAB = 1/2 * q0 ∧ qAH = 3/4 * q0 ∧ q_total = 7/4 * q0 :=
by {
  split,
  exact H1,
  split,
  exact H2,
  rw [H3, H1, H2],
  linarith
}

end irrigation_system_flow_rates_l758_758966


namespace intersection_A_B_l758_758514

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_A_B : A ∩ B = {-2, 2} := by sorry

end intersection_A_B_l758_758514


namespace impossible_to_divide_into_three_similar_parts_l758_758640

-- Define similarity condition
def similar_sizes (x y : ℝ) : Prop := x ≤ √2 * y

-- Main proof problem
theorem impossible_to_divide_into_three_similar_parts (x : ℝ) :
  ¬ ∃ (a b c : ℝ), a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes c a := 
sorry

end impossible_to_divide_into_three_similar_parts_l758_758640


namespace hyperbola_eccentricity_l758_758128

theorem hyperbola_eccentricity (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
    (C : P → Prop) 
    (angle_F1PF2 : ∀ {x y z : P}, ∀ (h : x ∈ C) (h₁ : y ∈ C) (h₂ : z ∈ C), is_angle x y z = 60)
    (dist_PF1_PF2 : ∀ (h : P ∈ C), dist P F1 = 3 * dist P F2) : 
    F1 ∈ C → F2 ∈ C → eccentricity C = (Real.sqrt 7) / 2 :=
by sorry

end hyperbola_eccentricity_l758_758128


namespace point_trajectory_l758_758170

def distance (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((b.1 - a.1) ^ 2 + (b.2 - a.2) ^ 2)

theorem point_trajectory (P : ℝ × ℝ) :
  let F1 := (-1, 0)
  let F2 := (1, 0)
  (distance P F1 + distance P F2 = distance F1 F2) →
  ∃ a, -1 ≤ a ∧ a ≤ 1 ∧ P = (a, 0) := by
  sorry

end point_trajectory_l758_758170


namespace option_A_option_C_l758_758243

def z (m : ℝ) : ℂ := (m + 2) + (m - 3) * Complex.I

theorem option_A (m : ℝ) (h : Im (z m) = 0) : m = 3 :=
by sorry

theorem option_C (h : z 1 = 3 - 2 * Complex.I) : Complex.abs (z 1) = Real.sqrt 13 :=
by sorry

end option_A_option_C_l758_758243


namespace geo_seq_ratio_l758_758702

theorem geo_seq_ratio (S : ℕ → ℝ) (r : ℝ) (hS : ∀ n, S n = (1 - r^(n+1)) / (1 - r))
  (hS_ratio : S 10 / S 5 = 1 / 2) : S 15 / S 5 = 3 / 4 := 
by
  sorry

end geo_seq_ratio_l758_758702


namespace fraction_given_to_jerry_l758_758330

-- Define the problem conditions
def initial_apples := 2
def slices_per_apple := 8
def total_slices := initial_apples * slices_per_apple -- 2 * 8 = 16

def remaining_slices_after_eating := 5
def slices_before_eating := remaining_slices_after_eating * 2 -- 5 * 2 = 10
def slices_given_to_jerry := total_slices - slices_before_eating -- 16 - 10 = 6

-- Define the proof statement to verify that the fraction of slices given to Jerry is 3/8
theorem fraction_given_to_jerry : (slices_given_to_jerry : ℚ) / total_slices = 3 / 8 :=
by
  -- skip the actual proof, just outline the goal
  sorry

end fraction_given_to_jerry_l758_758330


namespace impossible_divide_into_three_similar_parts_l758_758687

noncomputable def similar (x y : ℝ) : Prop := x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

theorem impossible_divide_into_three_similar_parts (s : ℝ → ℝ → Prop) :
  (∀ s, similar s)) → ¬ (∃ a b c : ℝ, s a b → s b c → s c a → a + b + c = 1) :=
by
  intros h_similar h
  sorry

end impossible_divide_into_three_similar_parts_l758_758687


namespace circle_properties_l758_758981

open Real

noncomputable def circle_center_radius (x y : ℝ) : Prop := 
    (x - 1)^2 + (y - 2)^2 = 5

noncomputable def chord_length (x y : ℝ) : Prop := 
    (3 * x - y - 6) = 0 → 
    (2 * sqrt (5 - (abs (3 * 1 + -1 * 2 - 6) / sqrt ((3)^2 + (-1)^2))^2)) = sqrt(10)

theorem circle_properties :
    (∃ x y : ℝ, circle_center_radius x y) ∧
    chord_length 1 2 := 
by
    constructor
    · use 1, 2
      unfold circle_center_radius
      norm_num
    · intro h
      unfold chord_length
      norm_num
      sorry

end circle_properties_l758_758981


namespace fraction_simplification_l758_758354

theorem fraction_simplification : 
  (2222 - 2123) ^ 2 / 121 = 81 :=
by
  sorry

end fraction_simplification_l758_758354


namespace trent_total_distance_l758_758340

-- Conditions
def walked_to_bus_stop : ℕ := 4
def bus_ride_to_library : ℕ := 7
def total_distance_to_library : ℕ := walked_to_bus_stop + bus_ride_to_library
def distance_back_home : ℕ := total_distance_to_library

-- Theorem stating that the total distance Trent traveled is 22 blocks
theorem trent_total_distance : 
  let total_distance := total_distance_to_library + distance_back_home in
  total_distance = 22 := 
by
  sorry

end trent_total_distance_l758_758340


namespace div_BnBk_l758_758855

noncomputable def A (n : ℕ) : ℕ := sorry 

def B (n : ℕ) : ℕ := (List.range n).map (λ i => A (i + 1)).prod

theorem div_BnBk (n k : ℕ) : B (n + k) ∣ B n * B k :=
sorry

end div_BnBk_l758_758855


namespace pages_for_35_dollars_l758_758614

def cost_per_page : ℚ := 7 / 3

theorem pages_for_35_dollars : 
  let cents_35_dollars := 3500 in 
  let pages := cents_35_dollars / cost_per_page in 
  pages = 1500 := 
by
  sorry

end pages_for_35_dollars_l758_758614


namespace unique_solution_l758_758449

theorem unique_solution : ∃! (x : ℝ), x^2 + 4 * x + 4 * x * sqrt(x + 3) = 13 ∧ x = 1 :=
by
  sorry -- proof required

end unique_solution_l758_758449


namespace num_customers_left_more_than_remaining_l758_758408

theorem num_customers_left_more_than_remaining (initial remaining : ℕ) (h : initial = 11 ∧ remaining = 3) : (initial - remaining) = (remaining + 5) :=
by sorry

end num_customers_left_more_than_remaining_l758_758408


namespace least_integer_with_exactly_eight_factors_l758_758821

theorem least_integer_with_exactly_eight_factors : ∃ n : ℕ, (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 8 ∨ d = 6 ∨ d = 12 ∨ d = 24) ∧
  (∀ m : ℕ, (∀ d : ℕ, d ∣ m → d = 1 ∨ d = 2 ∨ d = 3 ∨ d
= 4 ∨ d = 8 ∨ d = 6 ∨ d = 12 ∨ d = 24) → m = n) :=
begin
  sorry
end

end least_integer_with_exactly_eight_factors_l758_758821


namespace hyperbola_eccentricity_thm_l758_758104

noncomputable def hyperbola_eccentricity 
  (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
  (angle_F1PF2 : ℝ) (dist_PF1 dist_PF2 : ℝ)
  (H1 : angle_F1PF2 = 60)
  (H2 : dist_PF1 = 3 * dist_PF2) : ℝ :=
let a := dist_PF2 in
let c := (a * sqrt 7) / 2 in 
c / a

theorem hyperbola_eccentricity_thm
  (F1 F2 P : Type) [MetricSpace F1] [MetricSpace F2] [MetricSpace P]
  (angle_F1PF2 : ℝ) (dist_PF1 dist_PF2 : ℝ)
  (H1 : angle_F1PF2 = 60)
  (H2 : dist_PF1 = 3 * dist_PF2) :
  @hyperbola_eccentricity F1 F2 P _ _ _ angle_F1PF2 dist_PF1 dist_PF2 H1 H2 = (sqrt 7) / 2 :=
sorry

end hyperbola_eccentricity_thm_l758_758104


namespace eccentricity_of_hyperbola_l758_758119

open Real

-- Definitions of our conditions
variables {F1 F2 P : Point}
variables (a : ℝ) (m : ℝ)
variable (hyperbola_C : Hyperbola F1 F2)

-- Given conditions
axiom on_hyperbola : P ∈ hyperbola_C
axiom angle_F1P_F2 : angle F1 P F2 = π / 3
axiom distances : dist P F1 = 3 * dist P F2

-- Goal: Prove that the eccentricity of the hyperbola is sqrt(7)/2
theorem eccentricity_of_hyperbola : hyperbola.C.eccentricity = sqrt 7 / 2 := by
  sorry

end eccentricity_of_hyperbola_l758_758119


namespace quadrilateral_is_trapezoid_l758_758725

variables {A B C D K L M : Type}
variables [AffineSpace ℝ V] [NormedAddTorsor V V] (A B C D K L M : Affine V)
variables (hK : midpoint (A, B) K) 
variables (hL : midpoint (B, C) L)
variables (hM : ∃ (a : ℝ), ratio_eq CM DM 2 1)
variables (hDK_BM : parallel DK BM)
variables (hAL_CD : parallel AL CD)

theorem quadrilateral_is_trapezoid (AB CD : Affine V) :
  isTrapezoid A B C D := sorry

end quadrilateral_is_trapezoid_l758_758725


namespace simplify_and_evaluate_l758_758290

/-- 
Given the expression (1 + 1 / (x - 2)) ÷ ((x ^ 2 - 2 * x + 1) / (x - 2)), 
prove that it evaluates to -1 when x = 0.
-/
theorem simplify_and_evaluate (x : ℝ) (h : x = 0) :
  (1 + 1 / (x - 2)) / ((x^2 - 2 * x + 1) / (x - 2)) = -1 :=
by
  sorry

end simplify_and_evaluate_l758_758290


namespace range_of_a_false_proposition_for_all_l758_758558

variable {a : ℝ}

theorem range_of_a (h : ∀ x ∈ Ioo (2 : ℝ) 3, x^2 + 5 > a * x) : a < 2 * Real.sqrt 5 :=
sorry

theorem false_proposition_for_all (h : ¬∀ x ∈ Ioo (2 : ℝ) 3, x^2 + 5 > a * x) : a ≥ 2 * Real.sqrt 5 :=
sorry

end range_of_a_false_proposition_for_all_l758_758558


namespace tangent_segment_length_equality_l758_758721

-- Definitions of the problem
variables {O A B C D E : Type*}
variables (circle : set (euclidean_space ℝ 2))
variables (tangent_to_angle : O → Prop)
variables (diametrically_opposite : A B → Prop)
variables (tangent_at_B : B → set (euclidean_space ℝ 2))
variables (intersects_sides_at_C_and_D : tangent_at_B → Prop)
variables (intersects_OA_at_E : B → tangent_at_B → A → E → Prop)

-- Given the conditions, prove that BC = DE
theorem tangent_segment_length_equality
  (h_tangent_to_angle : tangent_to_angle O)
  (h_diametrically_opposite : diametrically_opposite A B)
  (h_intersects_sides : intersects_sides_at_C_and_D (tangent_at_B B))
  (h_intersects_OA : intersects_OA_at_E B (tangent_at_B B) A E) :
  dist (B, C) = dist (D, E) :=
sorry

end tangent_segment_length_equality_l758_758721


namespace point_inside_circle_l758_758589

theorem point_inside_circle (a b : ℝ) (h : ∀ x y : ℝ, (x^2 + y^2 = 4) → (ax + by ≠ 4)) :
  a^2 + b^2 < 4 := 
sorry

end point_inside_circle_l758_758589


namespace interior_angle_D_l758_758796

-- Main Statement
theorem interior_angle_D {x : ℝ} (h_total : (x + 60) + (2 * x + 40) + (3 * x - 12) = 360):
   65.33 = ((2 * x + 40) / 2) :=
begin
  sorry
end

end interior_angle_D_l758_758796


namespace intersection_sets_l758_758526

open Set

def A := {x : ℤ | abs x < 3}
def B := {x : ℤ | abs x > 1}

theorem intersection_sets :
  A ∩ B = {-2, 2} := by
  sorry

end intersection_sets_l758_758526


namespace trent_total_distance_l758_758335

theorem trent_total_distance
  (house_to_bus : ℕ)
  (bus_to_library : ℕ)
  (house_to_bus = 4)
  (bus_to_library = 7)
  : (house_to_bus + bus_to_library) * 2 = 22 :=
by
  sorry

end trent_total_distance_l758_758335


namespace sum_infinite_series_l758_758459

theorem sum_infinite_series : 
  (∑ k in (Finset.range ∞), (3^k) / (9^k - 1)) = 1 / 2 :=
  sorry

end sum_infinite_series_l758_758459


namespace trent_total_blocks_travelled_l758_758332

theorem trent_total_blocks_travelled :
  ∀ (walk_blocks bus_blocks : ℕ), 
  walk_blocks = 4 → 
  bus_blocks = 7 → 
  (walk_blocks + bus_blocks) * 2 = 22 := by
  intros walk_blocks bus_blocks hw hb
  rw [hw, hb]
  norm_num
  done

end trent_total_blocks_travelled_l758_758332


namespace incorrect_statement_about_parabola_l758_758961

def optionC_incorrect : Prop :=
  ∀ x : ℝ, x > -2 → (x + 2)^2 - 1 > 0

theorem incorrect_statement_about_parabola :
  ∃ x : ℝ, x > -2 ∧ ¬(optionC_incorrect) :=
begin
  use 0,
  split,
  { linarith, }, -- this proves 0 > -2
  { intros h_optionC,
    apply h_optionC,
    linarith, -- this proves the contradiction
  }
end

end incorrect_statement_about_parabola_l758_758961


namespace circle_equation_and_line_equation_l758_758504

def point (x y : ℝ) := (x, y)

def center_on_line (a b : ℝ) : Prop := a - b - 4 = 0
def tangent_to_y_axis (a b : ℝ) (r : ℝ) : Prop := (0 - a)^2 + (-2 - b)^2 = r^2
def chord_length (x1 y1 r : ℝ) (slope : ℝ) (line : ℝ × ℝ → ℝ) : Prop := 
  ∀ p1 p2, chord_length = 2 * sqrt(2) → line (x1, y1) = 0 → (x1 - 2)^2 + (y1 + 2)^2 = r^2

theorem circle_equation_and_line_equation
  (P : ℝ × ℝ)
  (center : ℝ × ℝ)
  (r : ℝ)
  (a b : ℝ)
  (k : ℝ)
  (l : ℝ × ℝ → ℝ) :
  P = (4,0) →
  center = (a, b) →
  center_on_line a b →
  tangent_to_y_axis a b r →
  chord_length (4,0) r k l → 
  (∀ x y, (x - 2)^2 + (y + 2)^2 = r^2) ∧ 
  (l (x, y) = (2 + sqrt(3)) * (x - 4)) ∨ (l (x, y) = (2 - sqrt(3)) * (x - 4)) :=
  sorry

end circle_equation_and_line_equation_l758_758504


namespace probability_of_3rd_term_is_3_p_plus_q_l758_758235

noncomputable def permutations_without_2_first : Finset (Fin 6 → Fin 6) :=
  {σ | ∃ σ₁ σ₂ σ₃ σ₄ σ₅ σ₆, σ = ![σ₁, σ₂, σ₃, σ₄, σ₅, σ₆] ∧ σ₁ ≠ 1}

noncomputable def favorable_permutations (T : Finset (Fin 6 → Fin 6)) : Finset (Fin 6 → Fin 6) :=
  {σ ∈ T | (σ 2 = 2)}

noncomputable def probability (T : Finset (Fin 6 → Fin 6)) (favorable : Finset (Fin 6 → Fin 6)) : ℚ :=
  favorable.card / T.card

theorem probability_of_3rd_term_is_3 (T : Finset (Fin 6 → Fin 6)) (hT : T = permutations_without_2_first) :
  probability T (favorable_permutations T) = 1 / 5 :=
by sorry

theorem p_plus_q : (1 + 5 = 6) :=
by linarith

end probability_of_3rd_term_is_3_p_plus_q_l758_758235


namespace find_a_l758_758550

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if (0 : ℝ) < x ∧ x < 3 then log x - a * x else sorry

theorem find_a (f : ℝ → ℝ) (a : ℝ) (h1 : ∀ x, f (x + 3) = 3 * f x)
  (h2 : ∀ x, (0 : ℝ) < x ∧ x < 3 → f x = log x - a * x)
  (h3 : a > 1 / 3)
  (h4 : ∀ x, (-6 : ℝ) < x ∧ x < -3 → f x ≤ -1/9) :
  a = 1 :=
sorry

end find_a_l758_758550


namespace sin_double_angle_l758_758971

theorem sin_double_angle (α : ℝ) (h : sin α + cos α = 2 / 3) : sin (2 * α) = -5 / 9 :=
sorry

end sin_double_angle_l758_758971


namespace benjamin_skating_time_l758_758580

-- Defining the conditions
def distance : ℕ := 80 -- Distance in kilometers
def speed : ℕ := 10   -- Speed in kilometers per hour

-- The main theorem statement
theorem benjamin_skating_time : ∀ (T : ℕ), T = distance / speed → T = 8 := by
  sorry

end benjamin_skating_time_l758_758580


namespace complementary_angles_positive_difference_l758_758762

/-- Two angles are complementary if their sum is 90 degrees.
    If the measures of these angles are in the ratio 3:1,
    then their positive difference is 45 degrees. -/
theorem complementary_angles_positive_difference (x : ℝ) (h1 : (3 * x) + x = 90) :
  abs ((3 * x) - x) = 45 :=
by
  sorry

end complementary_angles_positive_difference_l758_758762


namespace annual_population_change_l758_758887

theorem annual_population_change (initial_population : Int) (moved_in : Int) (moved_out : Int) (final_population : Int) (years : Int) : 
  initial_population = 780 → 
  moved_in = 100 →
  moved_out = 400 →
  final_population = 60 →
  years = 4 →
  (initial_population + moved_in - moved_out - final_population) / years = 105 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end annual_population_change_l758_758887


namespace largest_cylinder_volume_l758_758870

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ :=
  π * r^2 * h

theorem largest_cylinder_volume (r h : ℝ) (crate_height crate_width crate_length : ℝ)
  (h_radius : r = 7)
  (h_dimensions : crate_height = 7 ∧ crate_width = 8 ∧ crate_length = 12)
  (h_pillar_height : h ≤ crate_height) :
  volume_of_cylinder r h = 343 * π :=
by
  have h_r : r = 7 := h_radius
  have h_h : h = 7 := le_antisymm h_pillar_height (le_of_eq h_radius)
  have volume_eq : volume_of_cylinder r h = π * r^2 * h := rfl
  rw [h_r, h_h] at volume_eq
  rw volume_eq
  norm_num
  rw [mul_assoc, ←mul_assoc 7, sq, ←mul_assoc]
  norm_num
  sorry

end largest_cylinder_volume_l758_758870


namespace prob_of_point_below_x_axis_l758_758491

noncomputable def probability_point_below_x_axis (a : ℝ) (ha : a ∈ set.Icc (-1 : ℝ) 2) : ℚ :=
  (set.Icc (-1 : ℝ) 0).measure / (set.Icc (-1 : ℝ) 2).measure

theorem prob_of_point_below_x_axis : probability_point_below_x_axis = (1 / 3 : ℚ) := by
  sorry

end prob_of_point_below_x_axis_l758_758491


namespace min_customers_type_A_proof_selling_price_type_A_on_Womens_Day_proof_l758_758388

noncomputable def min_customers_type_A (x y : ℕ) : Prop :=
  (x + y = 480) ∧ (y ≤ 6 * x / 10) ∧ (forall z : ℕ, z = 300)

theorem min_customers_type_A_proof : min_customers_type_A 300 180 :=
by
  have h1 : 300 + 180 = 480 := rfl
  have h2 : 180 ≤ 6 * 300 / 10 := rfl
  exact ⟨h1, h2, rfl⟩

noncomputable def selling_price_type_A_on_Womens_Day (m x y : ℕ) : Prop :=
  let new_price_A := 90 + m
  let new_customers_A := 300 - 5 * m / 3
  let new_customers_B := 180 - m
  (new_customers_A * new_price_A + new_customers_B * 50 = 27000 + 9000) ∧ (forall p:ℕ, p = 150)

theorem selling_price_type_A_on_Womens_Day_proof : selling_price_type_A_on_Womens_Day 60 300 180 :=
by
  let new_price_A := 90 + 60
  let new_customers_A := 300 - 5 * 60 / 3
  let new_customers_B := 180 - 60
  have h1 : new_customers_A * new_price_A + new_customers_B * 50 = 36000 := rfl
  exact ⟨h1, rfl⟩

end min_customers_type_A_proof_selling_price_type_A_on_Womens_Day_proof_l758_758388


namespace positive_difference_complementary_angles_l758_758768

theorem positive_difference_complementary_angles (a b : ℝ) 
  (h1 : a + b = 90) 
  (h2 : 3 * b = a) :
  |a - b| = 45 :=
by
  sorry

end positive_difference_complementary_angles_l758_758768


namespace problem_solution_l758_758371

-- Definitions of the conditions in the problem
def unit_square (A B C D : Point) : Prop := sorry
def midpoint (F A B : Point) : Prop := sorry
def circle_centered_at (F C G : Point) : Prop := sorry
def circle_diameter_intersect (G A H : Point) : Prop := sorry
def quarter_point (E A C : Point) : Prop := sorry
def lengths (BE e BH h BF f : ℝ) : Prop := sorry

-- Main theorem to prove the solution
theorem problem_solution (A B C D F G H E : Point) (e h f : ℝ) :
  unit_square A B C D →
  midpoint F A B →
  circle_centered_at F C G →
  circle_diameter_intersect G A H →
  quarter_point E A C →
  lengths (distance B E) e (distance B H) h (distance B F) f →
  ((e - f) / (h - f) ≈ 6.90) →
  (required_pi_digits 6)
  sorry

end problem_solution_l758_758371


namespace correct_operation_l758_758842

theorem correct_operation (a b : ℝ) : a * b^2 - b^2 * a = 0 := by
  sorry

end correct_operation_l758_758842


namespace trapezium_area_l758_758940

theorem trapezium_area (a b h : ℝ) (h₁ : a = 24) (h₂ : b = 18) (h₃ : h = 15) : 
  (1 / 2 * (a + b) * h) = 315 :=
by
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end trapezium_area_l758_758940


namespace students_study_all_three_l758_758895

open Finset

variables (U : Finset ℕ) (M L C : Finset ℕ)
variables (students_total : U.card = 425)
variables (M_card : M.card = 351) (L_card : L.card = 71) (C_card : C.card = 203)
variables (more_than_one_subject : (M ∩ L).card + (M ∩ C).card + (L ∩ C).card - 2 * (M ∩ L ∩ C).card = 199)
variables (no_subject : U.card - (M ∪ L ∪ C).card = 8)

theorem students_study_all_three
    (U M L C : Finset ℕ)
    (students_total : U.card = 425)
    (M_card : M.card = 351)
    (L_card : L.card = 71)
    (C_card : C.card = 203)
    (more_than_one_subject : (M ∩ L).card + (M ∩ C).card + (L ∩ C).card - 2 * (M ∩ L ∩ C).card = 199)
    (no_subject : U.card - (M ∪ L ∪ C).card = 8)
    : (M ∩ L ∩ C).card = 9 :=
begin
  sorry
end

end students_study_all_three_l758_758895


namespace horses_eat_oats_twice_a_day_l758_758270

-- Define the main constants and assumptions
def number_of_horses : ℕ := 4
def oats_per_meal : ℕ := 4
def grain_per_day : ℕ := 3
def total_food : ℕ := 132
def duration_in_days : ℕ := 3

-- Main theorem statement
theorem horses_eat_oats_twice_a_day (x : ℕ) (h : duration_in_days * number_of_horses * (oats_per_meal * x + grain_per_day) = total_food) : x = 2 := 
sorry

end horses_eat_oats_twice_a_day_l758_758270


namespace min_red_chips_l758_758859

variable (w b r : ℕ)

theorem min_red_chips :
  (b ≥ w / 3) → (b ≤ r / 4) → (w + b ≥ 70) → r ≥ 72 :=
by
  sorry

end min_red_chips_l758_758859


namespace max_cards_saved_l758_758791

def is_valid_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 6 ∨ d = 8 ∨ d = 9

def is_palindromic_number (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  is_valid_digit hundreds ∧ is_valid_digit tens ∧ is_valid_digit units ∧ hundreds = units

def is_valid_upside_down_number (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  is_valid_digit hundreds ∧ is_valid_digit tens ∧ is_valid_digit units

def valid_three_digit_numbers := (100:ℕ, 999:ℕ).filter(is_valid_upside_down_number)

def palindromic_numbers := valid_three_digit_numbers.filter(is_palindromic_number)

theorem max_cards_saved : 
  ∃ (cards_saved : ℕ), 
  cards_saved = (valid_three_digit_numbers.length - palindromic_numbers.length) / 2 :=
sorry

end max_cards_saved_l758_758791


namespace domain_of_f_l758_758298

def f (x : ℝ) : ℝ := 1 / (log (x + 1)) + sqrt (4 - x^2)

theorem domain_of_f : {x : ℝ | ∃x, ((x + 1 > 0) ∧ (x + 1 ≠ 1) ∧ (4 - x^2 ≥ 0))} = set.union {x : ℝ | -1 < x ∧ x < 0} {x : ℝ | 0 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_f_l758_758298


namespace three_zeros_of_f_l758_758976

noncomputable def f (a x b : ℝ) : ℝ := (1/2) * a * x^2 - (a^2 + a + 2) * x + (2 * a + 2) * (Real.log x) + b

theorem three_zeros_of_f (a b : ℝ) (h1 : a > 3) (h2 : a^2 + a + 1 < b) (h3 : b < 2 * a^2 - 2 * a + 2) : 
  ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f a x1 b = 0 ∧ f a x2 b = 0 ∧ f a x3 b = 0 :=
by
  sorry

end three_zeros_of_f_l758_758976


namespace arithmetic_sequence_value_a4_l758_758209

noncomputable def arithmetic_seq (a_1 d : ℝ) : ℕ → ℝ
| 0     := a_1
| (n+1) := a_1 + n * d

def sum_arithmetic_seq (a_1 d : ℝ) (n : ℕ) :=
  (n : ℝ) * (a_1 + (a_1 + (n - 1) * d)) / 2

theorem arithmetic_sequence_value_a4 (a_1 d : ℝ) 
  (h_sum : sum_arithmetic_seq a_1 d 10 = 60)
  (h_a7 : arithmetic_seq a_1 d 7 = 7) :
  arithmetic_seq a_1 d 4 = 5 :=
by
  sorry

end arithmetic_sequence_value_a4_l758_758209


namespace calculate_expression_l758_758422

theorem calculate_expression :
  -15 - 21 + 8 = -28 :=
by
  sorry

end calculate_expression_l758_758422


namespace hyperbola_eccentricity_l758_758111

open Real

-- Conditions from the problem
variables (F₁ F₂ P : Point) (C : ℝ) (a : ℝ)
variables (angle_F1PF2 : angle F₁ P F₂ = 60)
variables (distance_PF1_PF2 : dist P F₁ = 3 * dist P F₂)
variables (focus_condition : 2 * C = dist F₁ F₂)

-- Statement of the problem
theorem hyperbola_eccentricity:
  let e := C / a in
  e = sqrt 7 / 2 :=
sorry

end hyperbola_eccentricity_l758_758111


namespace dominic_domino_problem_l758_758924

theorem dominic_domino_problem 
  (num_dominoes : ℕ)
  (pips_pairs : ℕ → ℕ)
  (hexagonal_ring : ℕ → ℕ → Prop) : 
  ∀ (adj : ℕ → ℕ → Prop), 
  num_dominoes = 6 → 
  (∀ i j, hexagonal_ring i j → pips_pairs i = pips_pairs j) →
  ∃ k, k = 2 :=
by {
  sorry
}

end dominic_domino_problem_l758_758924


namespace system_has_solution_l758_758260

theorem system_has_solution {a b : ℝ} 
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (sin x1 + a = b * x1) ∧ (sin x2 + a = b * x2)) :
  ∃ x : ℝ, (sin x + a = b * x) ∧ (cos x = b) :=
by 
  sorry

end system_has_solution_l758_758260


namespace optimal_thickness_l758_758329

noncomputable def C (x : ℝ) (k : ℝ) := k / (3 * x + 5)

def C₀ : ℝ := 8

noncomputable def f (x : ℝ) (k : ℝ) := (20 * C x k) + 6 * x

theorem optimal_thickness :
  (∃ k, C 0 k = C₀) →
  ∃ x_min, (0 ≤ x_min ∧ x_min ≤ 10) ∧ f x_min 40 = 70 :=
by {
  intro,
  use 40,
  exists 5,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { norm_num }
}


end optimal_thickness_l758_758329


namespace convex_hull_partition_count_l758_758485

def number_of_partitions (n : ℕ) (h : n > 2) : ℕ :=
  n * (n - 1) / 2

theorem convex_hull_partition_count (n : ℕ) (h : n > 2) 
  (no_three_collinear : ∀ (points : fin n → ℝ × ℝ), ¬ ∃ (a b c : fin n), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    collinear (points a) (points b) (points c)) :
  number_of_partitions n h = n * (n - 1) / 2 := 
sorry

end convex_hull_partition_count_l758_758485


namespace impossible_to_divide_three_similar_parts_l758_758694

theorem impossible_to_divide_three_similar_parts 
  (n : ℝ) 
  (p : ℝ → Prop) 
  (similar : ℝ → ℝ → Prop) 
  (h_similar : ∀ a b : ℝ, similar a b ↔ a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2) : 
  ¬ ∃ p1 p2 p3 : ℝ, p p1 ∧ p p2 ∧ p p3 ∧ (p1 + p2 + p3 = n) ∧ similar p1 p2 ∧ similar p2 p3 ∧ similar p1 p3 :=
sorry

end impossible_to_divide_three_similar_parts_l758_758694


namespace Lyle_percentage_of_chips_l758_758267

theorem Lyle_percentage_of_chips (total_chips : ℕ) (ratio_Ian_Lyle : ℕ × ℕ) (h_total_chips : total_chips = 100) (h_ratio : ratio_Ian_Lyle = (4, 6)) :
  let total_parts := ratio_Ian_Lyle.1 + ratio_Ian_Lyle.2 in
  let chips_per_part := total_chips / total_parts in
  let Lyle_chips := ratio_Ian_Lyle.2 * chips_per_part in
  let percentage_Lyle := (Lyle_chips * 100) / total_chips in
  percentage_Lyle = 60 :=
by
  intros
  sorry

end Lyle_percentage_of_chips_l758_758267


namespace existence_of_graph_with_conditions_l758_758805

variable (G : Type) [graph G]

def chromatic_number (G : Type) : ℕ := sorry
def flowing_chromatic_number (G : Type) : ℕ := sorry
def has_no_short_cycle (G : Type) (k : ℕ) : Prop := sorry

theorem existence_of_graph_with_conditions :
  (∃ G : Type, chromatic_number G ≤ 3 ∧ flowing_chromatic_number G ≥ 8 ∧ has_no_short_cycle G 2017) ∧
  (∀ m > 3, ¬ ∃ G : Type, chromatic_number G ≤ m ∧ flowing_chromatic_number G ≥ 2^m ∧ has_no_short_cycle G 2017) :=
sorry

end existence_of_graph_with_conditions_l758_758805


namespace cos_C_l758_758193

-- Definitions
variables {α : Type*} [linear_ordered_field α] 
variables (A B C : Type*) [is_right_triangle A B C]
variable (k : α)

/- Conditions -/
def angle_A_is_90 (A B C : Type*) [is_right_triangle A B C] := 
  angle A B C = 90
  
def tan_C_is_5 (A B C : Type*) [is_right_triangle A B C] (k : α) := 
  tan (C angle B) = 5

/- Proof Statement -/
theorem cos_C (A B C : Type*) [is_right_triangle A B C] (k : α)
  (h1 : angle_A_is_90 A B C)
  (h2 : tan_C_is_5 A B C k) :
  cos (angle C B) = 1 / sqrt 26 :=
sorry

end cos_C_l758_758193


namespace sum_of_divisors_form_840_eq_5_l758_758314

-- Define the given conditions as a universally quantified statement
theorem sum_of_divisors_form_840_eq_5 (i j : ℕ) (h_form : ∃ k : ℕ, k = 2^i * 3^j) 
  (h_sum_divisors : ∑ d in divisors (2^i * 3^j), d = 840) : i + j = 5 := sorry

end sum_of_divisors_form_840_eq_5_l758_758314


namespace compare_a_b_c_l758_758484

noncomputable def a : ℝ := 3 ^ 0.2
noncomputable def b : ℝ := Real.log 7 / Real.log 6
noncomputable def c : ℝ := Real.log 6 / Real.log 5

theorem compare_a_b_c : a > c ∧ c > b := by
  sorry

end compare_a_b_c_l758_758484


namespace midpoint_line_independence_l758_758169

theorem midpoint_line_independence
  (O1 O2 A B T1 T2 M1 M2 : Point)
  (l1 l2 : Line)
  (c1 c2 : Circle)
  (H1 : c1 = Circle.mk O1 (dist O1 A))
  (H2 : c2 = Circle.mk O2 (dist O2 A))
  (H3 : A ∈ c1)
  (H4 : A ∈ c2)
  (H5 : tangent l1 c1 A)
  (H6 : tangent l2 c2 A)
  (H7 : T1 ∈ c1)
  (H8 : T2 ∈ c2)
  (H9 : ∠ T1 O1 A = ∠ A O2 T2)
  (H10 : tangent (Line.through T1 M1) c1 T1)
  (H11 : tangent (Line.through T2 M2) c2 T2)
  (H12 : Line.through (Line.through T1 M1) l2)
  (H13 : Line.through (Line.through T2 M2) l1) :
  ∃ l : Line, ∀ T1 T2, midpoint M1 M2 ∈ l := sorry

end midpoint_line_independence_l758_758169


namespace smallest_number_with_exactly_eight_factors_l758_758828

theorem smallest_number_with_exactly_eight_factors
    (n : ℕ)
    (h1 : ∃ a b : ℕ, (a + 1) * (b + 1) = 8)
    (h2 : ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ n = p^a * q^b) : 
    n = 24 := by
  sorry

end smallest_number_with_exactly_eight_factors_l758_758828


namespace time_to_finish_task_l758_758726

-- Define the conditions
def printerA_rate (total_pages : ℕ) (time_A_alone : ℕ) : ℚ := total_pages / time_A_alone
def printerB_rate (rate_A : ℚ) : ℚ := rate_A + 10

-- Define the combined rate of printers working together
def combined_rate (rate_A : ℚ) (rate_B : ℚ) : ℚ := rate_A + rate_B

-- Define the time taken to finish the task together
def time_to_finish (total_pages : ℕ) (combined_rate : ℚ) : ℚ := total_pages / combined_rate

-- Given conditions
def total_pages : ℕ := 35
def time_A_alone : ℕ := 60

-- Definitions derived from given conditions
def rate_A : ℚ := printerA_rate total_pages time_A_alone
def rate_B : ℚ := printerB_rate rate_A

-- Combined rate when both printers work together
def combined_rate_AB : ℚ := combined_rate rate_A rate_B

-- Lean theorem statement to prove time taken by both printers
theorem time_to_finish_task : time_to_finish total_pages combined_rate_AB = 210 / 67 := 
by
  sorry

end time_to_finish_task_l758_758726


namespace probability_of_one_girl_conditional_probability_of_one_girl_given_at_least_one_l758_758199

/- Define number of boys and girls -/
def num_boys : ℕ := 5
def num_girls : ℕ := 3

/- Define number of students selected -/
def num_selected : ℕ := 2

/- Define the total number of ways to select -/
def total_ways : ℕ := Nat.choose (num_boys + num_girls) num_selected

/- Define the number of ways to select exactly one girl -/
def ways_one_girl : ℕ := Nat.choose num_girls 1 * Nat.choose num_boys 1

/- Define the number of ways to select at least one girl -/
def ways_at_least_one_girl : ℕ := total_ways - Nat.choose num_boys num_selected

/- Define the first probability: exactly one girl participates -/
def prob_one_girl : ℚ := ways_one_girl / total_ways

/- Define the second probability: exactly one girl given at least one girl -/
def prob_one_girl_given_at_least_one : ℚ := ways_one_girl / ways_at_least_one_girl

theorem probability_of_one_girl : prob_one_girl = 15 / 28 := by
  sorry

theorem conditional_probability_of_one_girl_given_at_least_one : prob_one_girl_given_at_least_one = 5 / 6 := by
  sorry

end probability_of_one_girl_conditional_probability_of_one_girl_given_at_least_one_l758_758199


namespace find_focus_of_parabola_4x2_l758_758016

-- Defining what it means to be the focus of a parabola given an equation y = ax^2.
def is_focus (a : ℝ) (f : ℝ) : Prop :=
  ∀ (x : ℝ), (x ^ 2 + (a * x ^ 2 - f) ^ 2) = ((a * x ^ 2 - (f * 8)) ^ 2)

-- Specific instance of the parabola y = 4x^2.
def parabola_4x2 := (4 : ℝ)

theorem find_focus_of_parabola_4x2 : ∃ f : ℝ, is_focus parabola_4x2 f :=
begin
  use (1/16 : ℝ),
  sorry -- The proof will be filled in by the theorem prover.
end

end find_focus_of_parabola_4x2_l758_758016


namespace find_m_l758_758985

-- Define the arithmetic sequence and its properties
variable {α : Type*} [OrderedRing α]
variable (a : Nat → α) (S : Nat → α) (m : ℕ)

-- The conditions from the problem
variable (is_arithmetic_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
variable (sum_of_terms : ∀ n, S n = (n * (a 0 + a (n - 1))) / 2)
variable (m_gt_one : m > 1)
variable (condition1 : a (m - 1) + a (m + 1) - a m ^ 2 - 1 = 0)
variable (condition2 : S (2 * m - 1) = 39)

-- Prove that m = 20
theorem find_m : m = 20 :=
sorry

end find_m_l758_758985


namespace second_circle_diameter_l758_758867

theorem second_circle_diameter (R : ℝ) (r : ℝ) (m n : ℕ) : 
  R = 10 →
  points_equally_spaced (A B C D E F) R →
  second_circle_tangent_to_lines_and_internally (A B F) r (R - r) →
  second_circle_diameter_eq (√m + n) (2 * r) →
  m + n = 1240 :=
by
  intros,
  sorry

end second_circle_diameter_l758_758867


namespace standard_colony_generated_l758_758304

theorem standard_colony_generated (initial_bacteria : Nat) (culture_medium : String) :
    (initial_bacteria = 1 ∧ culture_medium = "solid medium") ↔ 
    (∀ colony, 
        (colony.origins_from_single_bacterium ∨ colony.origins_from_few_bacteria) ∧ 
        colony.is_visible ∧ 
        ¬ (colony.is_formed_by_many_different_bacteria) → 
        colony.is_standard) :=
by
  -- sorry for the omitted proof
  sorry

end standard_colony_generated_l758_758304


namespace quadratic_root_sum_l758_758581

theorem quadratic_root_sum (a b : ℤ) (h₁ : (∃ x : ℝ, (x * x = 7 - 4 * real.sqrt 3) ∧ x = real.sqrt (7 - 4 * real.sqrt 3)))
                           (h₂ : ∃ y : ℝ, y * y = 7 - 4 * real.sqrt 3 ∧ y = 2 - real.sqrt 3)
                           (h₃ : ∃ z : ℝ, z * z = 7 - 4 * real.sqrt 3 ∧ z = 2 + real.sqrt 3)
                           (h₄ : (2 - real.sqrt 3) ≠ (2 + real.sqrt 3)) :
  a + b = -3 := sorry

end quadratic_root_sum_l758_758581


namespace r_at_zero_l758_758244

noncomputable def r : ℝ → ℝ := sorry

axiom r_degree : ∀ x : ℝ, polynomial.degree (polynomial.C (r x)) = 6

axiom r_conditions : ∀ n : ℕ, n ≤ 6 → r (3^n) = (1 / 3^n)

theorem r_at_zero : r 0 = -1 :=
by
  sorry

end r_at_zero_l758_758244


namespace weekly_earnings_l758_758958

theorem weekly_earnings (total_earnings : ℕ) (weeks : ℕ) (h1 : total_earnings = 133) (h2 : weeks = 19) : 
  round (total_earnings / weeks : ℝ) = 7 := 
by 
  sorry

end weekly_earnings_l758_758958


namespace partition_first_1000_squares_l758_758922

/-- Define the sum of the first n square numbers --/
def sum_of_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

/-- Define the partition identity --/
lemma partition_identity (x : ℕ) :
  x^2 + (x + 3)^2 + (x + 5)^2 + (x + 6)^2 = (x + 1)^2 + (x + 2)^2 + (x + 4)^2 + (x + 7)^2 :=
by sorry

/-- Prove that we can partition the sum of the first 1000 square numbers into two equal parts according to the identity --/
theorem partition_first_1000_squares :
  ∃ (a b : set ℕ), a ∪ b = { n^2 | n ∈ finset.range 1000 } ∧ a ∩ b = ∅ ∧ ∑ a = ∑ b :=
by sorry

end partition_first_1000_squares_l758_758922


namespace abs_sum_inequality_l758_758472

theorem abs_sum_inequality (k : ℝ) : (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 := 
sorry

end abs_sum_inequality_l758_758472


namespace second_smallest_five_digit_in_pascals_triangle_l758_758833

theorem second_smallest_five_digit_in_pascals_triangle : ∃ n k, (10000 < binomial n k) ∧ (binomial n k < 100000) ∧ 
    (∀ m l, (m < n ∨ (m = n ∧ l < k)) → (10000 ≤ binomial m l → binomial m l < binomial n k)) ∧ 
    binomial n k = 10001 :=
begin
  sorry
end

end second_smallest_five_digit_in_pascals_triangle_l758_758833


namespace seq_arithmetic_sum_seq_l758_758494

variable (a : ℕ → ℚ) (T : ℕ → ℚ)

-- Condition for the sequence {T_n}
axiom T_def : ∀ n, T n = 1 - a n

-- Question 1: Prove {1 / T_n} is an arithmetic sequence.
theorem seq_arithmetic (h : ∀ n, T n = 1 - a n) : ∃ c : ℚ, ∀ n, (1 / T n) = c + n := 
by sorry

-- Question 2: Find the sum of the first n terms of {a_n / T_n}.
theorem sum_seq (h : ∀ n, T n = 1 - a n) : (∀ n, (∑ i in Finset.range n, a i / T i) = n * (n + 1) / 2) :=
by sorry

end seq_arithmetic_sum_seq_l758_758494


namespace a1_geq_2_pow_k_l758_758621

-- Definitions of the problem conditions in Lean 4
def conditions (a : ℕ → ℕ) (n k : ℕ) : Prop :=
  (∀ i, 1 ≤ i ∧ i ≤ n → a i < 2 * n) ∧
  (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ i ≠ j → ¬(a i ∣ a j)) ∧
  (3^k < 2 * n ∧ 2 * n < 3^(k+1))

-- The main theorem to be proven
theorem a1_geq_2_pow_k (a : ℕ → ℕ) (n k : ℕ) (h : conditions a n k) : 
  a 1 ≥ 2^k :=
sorry

end a1_geq_2_pow_k_l758_758621


namespace find_n_of_expression_l758_758969

theorem find_n_of_expression : 
  (∃ n : ℕ, 2 * (nat.choose n 1) + 2^2 * (nat.choose n 2) + ... + 2^(n-1) * (nat.choose n (n-1)) + 2^n = 80) → 
  n = 4 :=
by {
  -- The proof part specifying that n = 4 should be written here.
  sorry
}

end find_n_of_expression_l758_758969


namespace half_angle_quadrant_l758_758994

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2) :
  (¬even k → π/2 < α/2 ∧ α/2 < π) ∨ (even k → π < α/2 ∧ α/2 < 3*π/2) :=
by
  sorry

end half_angle_quadrant_l758_758994


namespace part_one_solution_set_part_two_range_of_a_l758_758555

def f (x : ℝ) : ℝ := abs (2 * x - 4) + abs (x + 1)

theorem part_one_solution_set :
  { x : ℝ | f x ≤ 9 } = { x : ℝ | -2 ≤ x ∧ x ≤ 4 } :=
sorry

theorem part_two_range_of_a (a : ℝ) (B := { x : ℝ | x^2 - 3 * x < 0 })
  (A := { x : ℝ | f x < 2 * x + a }) :
  B ⊆ A → 5 ≤ a :=
sorry

end part_one_solution_set_part_two_range_of_a_l758_758555


namespace impossible_to_divide_into_three_similar_parts_l758_758638

-- Define similarity condition
def similar_sizes (x y : ℝ) : Prop := x ≤ √2 * y

-- Main proof problem
theorem impossible_to_divide_into_three_similar_parts (x : ℝ) :
  ¬ ∃ (a b c : ℝ), a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes c a := 
sorry

end impossible_to_divide_into_three_similar_parts_l758_758638


namespace focus_of_parabola_l758_758027

theorem focus_of_parabola (x f d : ℝ) (h : ∀ x, y = 4 * x^2 → (x = 0 ∧ y = f) → PF^2 = PQ^2 ∧ 
(PF^2 = x^2 + (4 * x^2 - f) ^ 2) := (PQ^2 = (4 * x^2 - d) ^ 2)) : 
  f = 1 / 16 := 
sorry

end focus_of_parabola_l758_758027


namespace number_of_large_boxes_l758_758325

theorem number_of_large_boxes (total_boxes : ℕ) (small_weight large_weight remaining_small remaining_large : ℕ) :
  total_boxes = 62 →
  small_weight = 5 →
  large_weight = 3 →
  remaining_small = 15 →
  remaining_large = 15 →
  ∀ (small_boxes large_boxes : ℕ),
    total_boxes = small_boxes + large_boxes →
    ((large_boxes * large_weight) + (remaining_small * small_weight) = (small_boxes * small_weight) + (remaining_large * large_weight)) →
    large_boxes = 27 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end number_of_large_boxes_l758_758325


namespace triangle_side_ratio_l758_758598

theorem triangle_side_ratio (A B C O : Point) (r R : ℝ) (h1 : circumscribe_triangle A B C = O) 
(h2 : incircle_center A B C = O) 
(h3 : R = circumradius A B C) 
(h4 : r = inradius A B C) 
(h5 : R > r) : 
∀ (a b c : ℝ), 
(a = side_length A B) → 
(b = side_length B C) → 
(c = side_length C A) → 
(max a (max b c)) / (min a (min b c)) < 2 :=
sorry

end triangle_side_ratio_l758_758598


namespace algebra_books_needed_l758_758871

theorem algebra_books_needed (A' H' S' M' E' : ℕ) (x y : ℝ) (z : ℝ)
  (h1 : y > x)
  (h2 : A' ≠ H' ∧ A' ≠ S' ∧ A' ≠ M' ∧ A' ≠ E' ∧ H' ≠ S' ∧ H' ≠ M' ∧ H' ≠ E' ∧ S' ≠ M' ∧ S' ≠ E' ∧ M' ≠ E')
  (h3 : A' * x + H' * y = z)
  (h4 : S' * x + M' * y = z)
  (h5 : E' * x = 2 * z) :
  E' = (2 * A' * M' - 2 * S' * H') / (M' - H') :=
by
  sorry

end algebra_books_needed_l758_758871


namespace simplify_expression_l758_758416

variable (m : ℕ) (h1 : m ≠ 2) (h2 : m ≠ 3)

theorem simplify_expression : 
  (m - 3) / (2 * m - 4) / (m + 2 - 5 / (m - 2)) = 1 / (2 * m + 6) :=
by sorry

end simplify_expression_l758_758416


namespace product_approximation_l758_758356

theorem product_approximation (a b : ℝ) (h₀ : a = 0.000625) (h₁ : b = 3_142_857) :
  a * b ≈ 1800 :=
by {
  rw [h₀, h₁],
  -- this would be the proof part which calculates the product and approximate it
  sorry
}

end product_approximation_l758_758356


namespace eccentricity_of_hyperbola_l758_758124

open Real

-- Definitions of our conditions
variables {F1 F2 P : Point}
variables (a : ℝ) (m : ℝ)
variable (hyperbola_C : Hyperbola F1 F2)

-- Given conditions
axiom on_hyperbola : P ∈ hyperbola_C
axiom angle_F1P_F2 : angle F1 P F2 = π / 3
axiom distances : dist P F1 = 3 * dist P F2

-- Goal: Prove that the eccentricity of the hyperbola is sqrt(7)/2
theorem eccentricity_of_hyperbola : hyperbola.C.eccentricity = sqrt 7 / 2 := by
  sorry

end eccentricity_of_hyperbola_l758_758124


namespace series_sum_eq_half_l758_758461

theorem series_sum_eq_half : (∑' n : ℕ, 1 ≤ n → ℚ, (3^n) / (9^n - 1)) = 1 / 2 := by
  sorry

end series_sum_eq_half_l758_758461


namespace value_of_fraction_l758_758575

theorem value_of_fraction (a b c d e f : ℚ) (h1 : a / b = 1 / 3) (h2 : c / d = 1 / 3) (h3 : e / f = 1 / 3) :
  (3 * a - 2 * c + e) / (3 * b - 2 * d + f) = 1 / 3 :=
by
  sorry

end value_of_fraction_l758_758575


namespace cost_of_each_green_hat_l758_758804

theorem cost_of_each_green_hat
  (total_hats : ℕ) (cost_blue_hat : ℕ) (total_price : ℕ) (green_hats : ℕ) (blue_hats : ℕ) (cost_green_hat : ℕ)
  (h1 : total_hats = 85) 
  (h2 : cost_blue_hat = 6) 
  (h3 : total_price = 550) 
  (h4 : green_hats = 40) 
  (h5 : blue_hats = 45) 
  (h6 : green_hats + blue_hats = total_hats) 
  (h7 : total_price = green_hats * cost_green_hat + blue_hats * cost_blue_hat) :
  cost_green_hat = 7 := 
sorry

end cost_of_each_green_hat_l758_758804


namespace chord_length_proof_line_eq_proof_l758_758609

noncomputable def curve1 := {ρ : ℝ | ρ = 2}
noncomputable def curve2 := {ρ : ℝ × θ : ℝ | ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2}

noncomputable def chord_length (ρ₁ ρ₂ : ℝ) : ℝ :=
  2 * Real.sqrt (ρ₁^2 - (Real.abs (ρ₁ * Real.sin (ρ₂ - Real.pi / 4) - Real.sqrt 2)))

theorem chord_length_proof :
  ∀ (A B : ℝ × ℝ), A ∈ curve1 → B ∈ curve2 → (chord_length 2 (2 - Real.sqrt 2) = 2 * Real.sqrt 2) :=
  by
    sorry

noncomputable def line_passing_through (C : ℝ × ℝ) :=
  {l : ℝ | l = C.1 - C.2}

theorem line_eq_proof :
  ∀ (C : ℝ × ℝ), C = (1, 0) → 
  (⟨sqrt 2, by simp [Real.sin (C.2 - π / 4)]⟩ = 1) :=
  by
    sorry

#check chord_length_proof
#check line_eq_proof

end chord_length_proof_line_eq_proof_l758_758609


namespace maximum_n_product_gt_one_l758_758498

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ r, a (n + 1) = a n * r

noncomputable def product_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∏ i in finset.range (n + 1), a i

theorem maximum_n_product_gt_one
  (a : ℕ → ℝ) (T : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_T : ∀ n, T n = product_first_n_terms a n)
  (h1 : a 1 > 1)
  (h2008_2009 : a 2008 * a 2009 > 1)
  (h_ineq : (a 2008 - 1) * (a 2009 - 1) < 0) :
  ∃ n, n = 4016 ∧ T n > 1 ∧ ∀ k > 4016, T k ≤ 1 :=
sorry

end maximum_n_product_gt_one_l758_758498


namespace divide_pile_l758_758680

theorem divide_pile (pile : ℝ) (similar : ℝ → ℝ → Prop) :
  (∀ x y, similar x y ↔ x ≤ y * Real.sqrt 2 ∧ y ≤ x * Real.sqrt 2) →
  ¬∃ a b c, a + b + c = pile ∧ similar a b ∧ similar b c ∧ similar a c :=
by sorry

end divide_pile_l758_758680


namespace select_numbers_to_sum_713_l758_758319

open Set

-- Definitions based on the problem statement
def is_odd (n : ℕ) : Prop := n % 2 = 1
def not_divisible_by_5 (n : ℕ) : Prop := n % 5 ≠ 0
def ends_in_713 (n : ℕ) : Prop := n % 10000 = 713

-- Main theorem statement
theorem select_numbers_to_sum_713 (S : Set ℕ) (h1 : S.card = 1000)
  (h2 : ∀ s ∈ S, is_odd s) (h3 : ∀ s ∈ S, not_divisible_by_5 s) :
  ∃ T ⊆ S, ends_in_713 (T.sum id) := sorry

end select_numbers_to_sum_713_l758_758319


namespace acute_triangles_in_cube_l758_758793

noncomputable def numberOfAcuteTrianglesInCube : Nat := 8

theorem acute_triangles_in_cube (vertices : Fin 8 → ℝ^3) (h_cube : is_cube vertices) :
  (countAcuteTriangles vertices) = numberOfAcuteTrianglesInCube := sorry

end acute_triangles_in_cube_l758_758793


namespace min_value_expr_l758_758099

theorem min_value_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 9) :
  ∃ C : ℝ, ∀ x y : ℝ, (x + 2 * y = 9) → (0 < x) → (0 < y) →
  (frac(2, y) + frac(1, x)) ≥ C ∧ (frac(2, y) + frac(1, x)) = C := sorry

end min_value_expr_l758_758099


namespace find_distance_l758_758495

theorem find_distance (α : ℝ) (a b : ℝ) 
  (h1 : cos (2 * α) = 2 / 3) 
  (h2 : tan α = (b - a) / (2 - 1)) :
  |a - b| = sqrt 5 / 5 :=
by sorry

end find_distance_l758_758495


namespace number_of_lions_on_saturday_l758_758928

-- Definitions for the conditions
def animals_on_saturday (lions : ℕ) : ℕ := lions + 2  -- assuming 2 elephants
def animals_on_sunday : ℕ := 2 + 5  -- 2 buffaloes and 5 leopards
def animals_on_monday : ℕ := 5 + 3   -- 5 rhinos and 3 warthogs
def total_animals_seen (lions : ℕ) : ℕ := animals_on_saturday lions + animals_on_sunday + animals_on_monday

-- Main theorem to prove the number of lions
theorem number_of_lions_on_saturday : ∃ l : ℕ, total_animals_seen l = 20 ∧ l = 3 :=
by
  use 3
  dsimp [total_animals_seen, animals_on_saturday, animals_on_sunday, animals_on_monday]
  split
  · norm_num
  · rfl

end number_of_lions_on_saturday_l758_758928


namespace impossible_to_divide_into_three_similar_parts_l758_758644

-- Define similarity condition
def similar_sizes (x y : ℝ) : Prop := x ≤ √2 * y

-- Main proof problem
theorem impossible_to_divide_into_three_similar_parts (x : ℝ) :
  ¬ ∃ (a b c : ℝ), a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes c a := 
sorry

end impossible_to_divide_into_three_similar_parts_l758_758644


namespace no_division_into_three_similar_piles_l758_758664

theorem no_division_into_three_similar_piles :
    ∀ (x : ℝ),
    ∀ (y z : ℝ),
    (x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = x) →
    (x <= sqrt 2 * y ∧ y <= sqrt 2 * z ∧ z <= sqrt 2 * x) →
    false :=
by
  intro x y z
  sorry

end no_division_into_three_similar_piles_l758_758664


namespace smallest_positive_period_and_intervals_of_monotonic_decrease_range_of_a_l758_758482

def AC (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2) + Real.sin (x / 2), -Real.sin (x / 2))

def BC (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2) - Real.sin (x / 2), 2 * Real.cos (x / 2))

def f (x : ℝ) : ℝ := AC x • BC x -- Dot product

-- 1. Prove the smallest positive period and the intervals of monotonic decrease
theorem smallest_positive_period_and_intervals_of_monotonic_decrease :
  (∀ x : ℝ, f (x + 2 * Real.pi) = f x) ∧
  (∀ k : ℤ, is_monotonic_decrease (f x) (x ∈ set.Icc (-Real.pi / 4 + 2 * k * Real.pi) (3 * Real.pi / 4 + 2 * k * Real.pi))) :=
sorry

-- 2. Prove the range of values for a
theorem range_of_a (a : ℝ) :
  a ∈ set.Ico 1 (Real.sqrt 2) ↔ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ -Real.pi / 2 ≤ x1 ∧ x1 ≤ Real.pi / 2 ∧ -Real.pi / 2 ≤ x2 ∧ x2 ≤ Real.pi / 2 ∧ f x1 = a ∧ f x2 = a ) :=
sorry

end smallest_positive_period_and_intervals_of_monotonic_decrease_range_of_a_l758_758482


namespace _l758_758136

noncomputable def hyperbola_eccentricity_theorem {F1 F2 P : Point} 
  (hyp : is_hyperbola F1 F2 P)
  (angle_F1PF2 : ∠F1 P F2 = 60)
  (dist_PF1_3PF2 : dist P F1 = 3 * dist P F2) : 
  eccentricity F1 F2 = sqrt 7 / 2 :=
by 
  sorry

end _l758_758136


namespace impossible_divide_into_three_similar_parts_l758_758689

noncomputable def similar (x y : ℝ) : Prop := x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

theorem impossible_divide_into_three_similar_parts (s : ℝ → ℝ → Prop) :
  (∀ s, similar s)) → ¬ (∃ a b c : ℝ, s a b → s b c → s c a → a + b + c = 1) :=
by
  intros h_similar h
  sorry

end impossible_divide_into_three_similar_parts_l758_758689


namespace probability_even_sum_l758_758478

open Finset

theorem probability_even_sum :
  let cards := {1, 2, 3, 4, 5} in
  let card_combinations := cards.to_finset.subsets 2 in
  let even_sum_combinations := card_combinations.filter (λ s, (s.sum id) % 2 = 0) in
  (even_sum_combinations.card.to_nat / card_combinations.card.to_nat) = (2 / 5) := 
sorry

end probability_even_sum_l758_758478


namespace hyperbola_eccentricity_l758_758110

open Real

-- Conditions from the problem
variables (F₁ F₂ P : Point) (C : ℝ) (a : ℝ)
variables (angle_F1PF2 : angle F₁ P F₂ = 60)
variables (distance_PF1_PF2 : dist P F₁ = 3 * dist P F₂)
variables (focus_condition : 2 * C = dist F₁ F₂)

-- Statement of the problem
theorem hyperbola_eccentricity:
  let e := C / a in
  e = sqrt 7 / 2 :=
sorry

end hyperbola_eccentricity_l758_758110


namespace focus_of_parabola_y_eq_4x_sq_l758_758008

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := (0 : ℝ, 1 / 16 : ℝ)
  in f

theorem focus_of_parabola_y_eq_4x_sq :
  (0, 1 / 16) = parabola_focus := by
  unfold parabola_focus
  sorry

end focus_of_parabola_y_eq_4x_sq_l758_758008


namespace divide_pile_l758_758683

theorem divide_pile (pile : ℝ) (similar : ℝ → ℝ → Prop) :
  (∀ x y, similar x y ↔ x ≤ y * Real.sqrt 2 ∧ y ≤ x * Real.sqrt 2) →
  ¬∃ a b c, a + b + c = pile ∧ similar a b ∧ similar b c ∧ similar a c :=
by sorry

end divide_pile_l758_758683


namespace select_numbers_to_sum_713_l758_758320

open Set

-- Definitions based on the problem statement
def is_odd (n : ℕ) : Prop := n % 2 = 1
def not_divisible_by_5 (n : ℕ) : Prop := n % 5 ≠ 0
def ends_in_713 (n : ℕ) : Prop := n % 10000 = 713

-- Main theorem statement
theorem select_numbers_to_sum_713 (S : Set ℕ) (h1 : S.card = 1000)
  (h2 : ∀ s ∈ S, is_odd s) (h3 : ∀ s ∈ S, not_divisible_by_5 s) :
  ∃ T ⊆ S, ends_in_713 (T.sum id) := sorry

end select_numbers_to_sum_713_l758_758320


namespace min_medians_l758_758256

open Finset

universe u
variables {α : Type u} [decidable_eq α] [linear_order α]

/-- Definition of collinear points -/
def collinear (P Q R : α × α) : Prop :=
  (P.snd - Q.snd) * (P.fst - R.fst) = (P.snd - R.snd) * (P.fst - Q.fst)

/-- Definition of a median -/
def is_median {n : ℕ} (points : finset (α × α)) (l : α × α → Prop) : Prop :=
  ∃ P1 P2 ∈ points, ¬collinear P1 P2 P2 ∧ (l P1 ∨ l P2) ∧
  2 * (points.filter (λ p, l p)).card = points.card - 2

/-- Problem statement: For a set of 2n points with no three points collinear,
    the minimum number of medians is n. -/
theorem min_medians {n : ℕ} (points : finset (α × α)) (h_points : points.card = 2 * n) 
  (h_collinear : ∀ P1 P2 P3 ∈ points, ¬collinear P1 P2 P3) : 
  ∃ (median_count : ℕ), median_count = n ∧ ∀ l, 
    is_median points l → (points.filter l).card = median_count := 
sorry

end min_medians_l758_758256


namespace find_focus_of_parabola_4x2_l758_758019

-- Defining what it means to be the focus of a parabola given an equation y = ax^2.
def is_focus (a : ℝ) (f : ℝ) : Prop :=
  ∀ (x : ℝ), (x ^ 2 + (a * x ^ 2 - f) ^ 2) = ((a * x ^ 2 - (f * 8)) ^ 2)

-- Specific instance of the parabola y = 4x^2.
def parabola_4x2 := (4 : ℝ)

theorem find_focus_of_parabola_4x2 : ∃ f : ℝ, is_focus parabola_4x2 f :=
begin
  use (1/16 : ℝ),
  sorry -- The proof will be filled in by the theorem prover.
end

end find_focus_of_parabola_4x2_l758_758019


namespace solve_arithmetic_sequence_problem_l758_758497

noncomputable def arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) (m : ℕ) : Prop :=
  (∀ n, a n = a 0 + n * (a 1 - a 0)) ∧  -- Condition: sequence is arithmetic
  (∀ n, S n = (n * (a 0 + a (n - 1))) / 2) ∧  -- Condition: sum of first n terms
  (m > 1) ∧  -- Condition: m > 1
  (a (m - 1) + a (m + 1) - a m ^ 2 = 0) ∧  -- Given condition
  (S (2 * m - 1) = 38)  -- Given that sum of first 2m-1 terms equals 38

-- The statement we need to prove
theorem solve_arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) (m : ℕ) :
  arithmetic_sequence_problem a S m → m = 10 :=
by
  sorry  -- Proof to be completed

end solve_arithmetic_sequence_problem_l758_758497


namespace area_of_square_same_perimeter_as_triangle_l758_758879

noncomputable def square_perimeter_from_triangle_sides (a b c : ℝ) : ℝ :=
  a + b + c

noncomputable def area_of_square_with_perimeter (p : ℝ) : ℝ :=
  (p / 4) ^ 2

theorem area_of_square_same_perimeter_as_triangle (a b c : ℝ) (h₁ : a = 7.5) 
  (h₂ : b = 9.3) (h₃ : c = 12.2) :
  let p := square_perimeter_from_triangle_sides a b c in
  let area := area_of_square_with_perimeter p in
  area = 52.5625 :=
by
  have hp : p = 29 := by
    calc
      p = a + b + c := rfl
      ... = 7.5 + 9.3 + 12.2 := by rw [h₁, h₂, h₃]
      ... = 29 := by norm_num
  have ha : area = (29 / 4) ^ 2 := by
    unfold area_of_square_with_perimeter
    rw [hp]
  calc 
    area = (29 / 4)^2 := by rw [ha]
    ... = 52.5625 := by norm_num

end area_of_square_same_perimeter_as_triangle_l758_758879


namespace max_min_distance_max_min_k_l758_758987

-- Define the circle C as the set of points satisfying the equation
def on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 14*y + 45 = 0

-- Define point Q
def Q : ℝ × ℝ := (-2, 3)

-- The distances |MQ| and the corresponding points on the circle.
def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define k related to m, n
def k (m n : ℝ) : ℝ :=
  (n - 3) / (m + 2)

-- Prove maximum and minimum for |MQ|
theorem max_min_distance (M : ℝ × ℝ) (hM : on_circle M.1 M.2) :
  let d := distance M.1 M.2 Q.1 Q.2 in
  d ≤ 6 * Real.sqrt 2 ∧ d ≥ 2 * Real.sqrt 2 :=
sorry

-- Prove maximum and minimum for k = (n-3)/(m+2)
theorem max_min_k (m n : ℝ) (h : on_circle m n) :
  k m n ≤ 2 + Real.sqrt 3 ∧ k m n ≥ 2 - Real.sqrt 3 :=
sorry

end max_min_distance_max_min_k_l758_758987


namespace iesha_total_books_l758_758183

theorem iesha_total_books (schoolBooks sportsBooks : ℕ) (h1 : schoolBooks = 19) (h2 : sportsBooks = 39) : schoolBooks + sportsBooks = 58 :=
by
  sorry

end iesha_total_books_l758_758183


namespace hyperbola_eccentricity_l758_758107

open Real

-- Conditions from the problem
variables (F₁ F₂ P : Point) (C : ℝ) (a : ℝ)
variables (angle_F1PF2 : angle F₁ P F₂ = 60)
variables (distance_PF1_PF2 : dist P F₁ = 3 * dist P F₂)
variables (focus_condition : 2 * C = dist F₁ F₂)

-- Statement of the problem
theorem hyperbola_eccentricity:
  let e := C / a in
  e = sqrt 7 / 2 :=
sorry

end hyperbola_eccentricity_l758_758107


namespace angle_between_vectors_is_30_degrees_l758_758174

open Real

noncomputable def vector_a : ℝ × ℝ := (cos (35 * pi / 180), sin (35 * pi / 180))
noncomputable def vector_b : ℝ × ℝ := (cos (65 * pi / 180), sin (65 * pi / 180))

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem angle_between_vectors_is_30_degrees :
  let θ := acos ((dot_product vector_a vector_b) / (magnitude vector_a * magnitude vector_b))
  θ = 30 * pi / 180 :=
by
  sorry

end angle_between_vectors_is_30_degrees_l758_758174


namespace decagon_hexagon_area_ratio_l758_758202

theorem decagon_hexagon_area_ratio :
  ∃ (K L M N O P Q R S : Point), 
    is_regular_decagon ABCDEFGHIJ ∧
    divides_into_ratio (3:1) A B K ∧
    divides_into_ratio (3:1) B C L ∧
    divides_into_ratio (3:1) C D M ∧
    divides_into_ratio (3:1) D E N ∧
    divides_into_ratio (3:1) E F O ∧
    divides_into_ratio (3:1) F G P ∧
    divides_into_ratio (3:1) G H Q ∧
    divides_into_ratio (3:1) H I R ∧
    divides_into_ratio (3:1) I J S ∧
    area_ratio_of_hexagon_to_decagon K M O Q S U ABCDEFGHIJ = (3 * sqrt 3) / 40 :=
by sorry

end decagon_hexagon_area_ratio_l758_758202


namespace find_n_312_l758_758470

noncomputable def floor_log_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ x => Int.floor (Real.log2 (x + 1)))

theorem find_n_312 (S_n : ℕ) (h : S_n = 1994) :
  ∃ n : ℕ, floor_log_sum n = 1994 ∧ n = 312 :=
by
  use 312
  have h1 : floor_log_sum 312 = 1994 := sorry
  exact ⟨h1, rfl⟩

end find_n_312_l758_758470


namespace focus_of_parabola_y_eq_4x_sq_l758_758009

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := (0 : ℝ, 1 / 16 : ℝ)
  in f

theorem focus_of_parabola_y_eq_4x_sq :
  (0, 1 / 16) = parabola_focus := by
  unfold parabola_focus
  sorry

end focus_of_parabola_y_eq_4x_sq_l758_758009


namespace permutations_not_adjacent_l758_758479

open Nat

theorem permutations_not_adjacent (n : ℕ) :
  (∑ k in Finset.range (n+1), (-1 : ℤ)^k * (Nat.choose n k) * 2^k * (3*n - k)!)
  = ∑ k in Finset.range (n+1), (-1 : ℤ)^k * (Nat.choose n k) * (3*n - k)! :=
by sorry

end permutations_not_adjacent_l758_758479


namespace min_value_expression_l758_758635

theorem min_value_expression (x y z : ℝ) (h1 : -1/2 < x ∧ x < 1/2) (h2 : -1/2 < y ∧ y < 1/2) (h3 : -1/2 < z ∧ z < 1/2) :
  (1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) + 1 / 2) ≥ 2.5 :=
by {
  sorry
}

end min_value_expression_l758_758635


namespace least_positive_integer_with_eight_factors_l758_758817

noncomputable def numDivisors (n : ℕ) : ℕ :=
  (List.range (n+1)).count (λ d => d > 0 ∧ n % d = 0)

theorem least_positive_integer_with_eight_factors : ∃ n : ℕ, n > 0 ∧ numDivisors n = 8 ∧ (∀ m : ℕ, m > 0 → numDivisors m = 8 → n ≤ m) := 
  sorry

end least_positive_integer_with_eight_factors_l758_758817


namespace milk_cost_correct_l758_758041

-- Definitions of the given conditions
def bagelCost : ℝ := 0.95
def orangeJuiceCost : ℝ := 0.85
def sandwichCost : ℝ := 4.65
def lunchExtraCost : ℝ := 4.0

-- Total cost of breakfast
def breakfastCost : ℝ := bagelCost + orangeJuiceCost

-- Total cost of lunch
def lunchCost : ℝ := breakfastCost + lunchExtraCost

-- Cost of milk
def milkCost : ℝ := lunchCost - sandwichCost

-- Theorem to prove the cost of milk
theorem milk_cost_correct : milkCost = 1.15 :=
by
  sorry

end milk_cost_correct_l758_758041


namespace monotonic_increasing_interval_l758_758306

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem monotonic_increasing_interval :
  {x : ℝ | x > 0} = {x : ℝ | f' x > 0} :=
by
  sorry

end monotonic_increasing_interval_l758_758306


namespace students_present_l758_758849

/-- Number of students present in the class, given total students and percentage absent. -/
theorem students_present (total_students : ℕ) (percentage_absent : ℕ) (percentage_present : ℕ) :
    total_students = 50 → percentage_absent = 10 → percentage_present = 90 →
    (percentage_present * total_students / 100) = 45 := 
by
  intros h_total h_absent h_present
  rw [h_total, h_present]
  norm_num
  exact eq.refl 45

end students_present_l758_758849


namespace right_triangle_9_12_15_l758_758843

-- Axiomatic statement that describes the problem
theorem right_triangle_9_12_15 :
  (9^2 + 12^2 = 15^2) := by
  calc
    9^2 + 12^2 = 81 + 144 := by rw[←pow_two 9, ←pow_two 12]
    ...         = 225     := by norm_num
    15^2       = 225     := by norm_num

end right_triangle_9_12_15_l758_758843


namespace mean_seventh_median_eighth_level_a_percentage_eighth_seventh_better_campaign_l758_758443

noncomputable def data_seventh := [0.8, 0.9, 0.8, 0.9, 1.1, 1.7, 2.3, 1.1, 1.9, 1.6]
noncomputable def data_eighth := [1.0, 0.9, 1.3, 1.0, 1.9, 1.0, 0.9, 1.7, 2.3, 1.0]

noncomputable def mean (data: List Float) : Float :=
  (data.foldr (· + ·) 0) / data.length

noncomputable def median (data: List Float) : Float :=
  let sorted := data.qsort (· < ·)
  if sorted.length % 2 = 0 then
    (sorted.get ⟨sorted.length / 2 - 1, sorry⟩ + sorted.get ⟨sorted.length / 2, sorry⟩) / 2
  else
    sorted.get ⟨sorted.length / 2, sorry⟩

noncomputable def level_a_percentage (data: List Float) : Float :=
  (data.filter (< 1)).length * 100 / data.length

theorem mean_seventh : mean data_seventh = 1.31 := by
  sorry

theorem median_eighth : median data_eighth = 1.0 := by
  sorry

theorem level_a_percentage_eighth : level_a_percentage data_eighth = 20 := by
  sorry

theorem seventh_better_campaign : 
  let mode_seventh := 0.8  -- Given
  let mode_eighth := 1.0   -- Given
  let level_a_percentage_seventh := 40 -- Given
  level_a_percentage_seventh > level_a_percentage data_eighth ∧ mode_seventh < mode_eighth := by
  sorry

end mean_seventh_median_eighth_level_a_percentage_eighth_seventh_better_campaign_l758_758443


namespace impossible_to_divide_into_three_similar_piles_l758_758647

def similar (a b : ℝ) : Prop :=
  a / b ≤ real.sqrt 2 ∧ b / a ≤ real.sqrt 2

theorem impossible_to_divide_into_three_similar_piles (pile : ℝ) (h : 0 < pile) :
  ¬ ∃ (x y z : ℝ), 
    x + y + z = pile ∧
    similar x y ∧ similar y z ∧ similar z x :=
by
  sorry

end impossible_to_divide_into_three_similar_piles_l758_758647


namespace impossible_to_divide_into_three_similar_piles_l758_758646

def similar (a b : ℝ) : Prop :=
  a / b ≤ real.sqrt 2 ∧ b / a ≤ real.sqrt 2

theorem impossible_to_divide_into_three_similar_piles (pile : ℝ) (h : 0 < pile) :
  ¬ ∃ (x y z : ℝ), 
    x + y + z = pile ∧
    similar x y ∧ similar y z ∧ similar z x :=
by
  sorry

end impossible_to_divide_into_three_similar_piles_l758_758646


namespace circle_area_l758_758302

/--
Given the polar equation of a circle r = -4 * cos θ + 8 * sin θ,
prove that the area of the circle is 20π.
-/
theorem circle_area (θ : ℝ) (r : ℝ) (cos : ℝ → ℝ) (sin : ℝ → ℝ) 
  (h_eq : ∀ θ : ℝ, r = -4 * cos θ + 8 * sin θ) : 
  ∃ A : ℝ, A = 20 * Real.pi :=
by
  sorry

end circle_area_l758_758302


namespace inequality_sum_squares_l758_758632

theorem inequality_sum_squares (n : ℕ) (a : Fin n → ℝ) 
  (h_pos : ∀ i, 0 < a i)
  (h_sum : (Finset.univ.sum (λ i, a i)) = 1) :
  (Finset.univ.sum (λ i, (a i + (1 / a i))^2)) ≥ (n^2 + 1)^2 / n := 
  sorry

end inequality_sum_squares_l758_758632


namespace rhombus_diagonal_length_l758_758752

theorem rhombus_diagonal_length (d1 d2 : ℝ) (area : ℝ) (h1 : d2 = 16) (h2 : area = 88) :
  area = (d1 * d2) / 2 → d1 = 11 :=
by
  assume h3 : area = (d1 * d2) / 2
  sorry

end rhombus_diagonal_length_l758_758752


namespace negation_proposition_l758_758307

theorem negation_proposition :
  (¬ (∀ x : ℝ, abs x + x^2 ≥ 0)) ↔ (∃ x₀ : ℝ, abs x₀ + x₀^2 < 0) :=
by
  sorry

end negation_proposition_l758_758307


namespace second_smallest_five_digit_in_pascals_triangle_l758_758834

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem second_smallest_five_digit_in_pascals_triangle :
  (∃ n k : ℕ, n > 0 ∧ k > 0 ∧ (10000 ≤ binomial n k) ∧ (binomial n k < 100000) ∧
    (∀ m l : ℕ, m > 0 ∧ l > 0 ∧ (10000 ≤ binomial m l) ∧ (binomial m l < 100000) →
    (binomial n k < binomial m l → binomial n k ≥ 31465)) ∧  binomial n k = 31465) :=
sorry

end second_smallest_five_digit_in_pascals_triangle_l758_758834


namespace range_of_a_l758_758164

noncomputable def f (x : ℝ) := x * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x ≥ -x^2 + a*x - 6) → a ≤ 5 + Real.log 2 :=
by
  sorry

end range_of_a_l758_758164


namespace debra_probability_theorem_l758_758916

-- Define event for Debra's coin flipping game starting with "HTT"
def debra_coin_game_event : Prop := 
  let heads_probability : ℝ := 0.5
  let tails_probability : ℝ := 0.5
  let initial_prob : ℝ := heads_probability * tails_probability * tails_probability
  let Q : ℝ := 1 / 3  -- the computed probability of getting HH after HTT
  let final_probability : ℝ := initial_prob * Q
  final_probability = 1 / 24

-- The theorem statement
theorem debra_probability_theorem :
  debra_coin_game_event := 
by
  sorry

end debra_probability_theorem_l758_758916


namespace count_divisible_digits_l758_758071

def is_divisible (a b : ℕ) : Prop := a % b = 0

theorem count_divisible_digits :
  let count := (finset.range 10).filter (λ n, n > 0 ∧ is_divisible (15 * n) n) in
  count.card = 6 :=
by
  let valid_digits := [1, 2, 3, 5, 6, 9].to_finset
  have valid_digits_card : valid_digits.card = 6 := by simp
  have matching_digits : ∀ n : ℕ, n > 0 ∧ n < 10 ∧ 15 * n % n = 0 ↔ n ∈ valid_digits := by
    intros n
    simp
    split
      intros ⟨n_pos, n_lt, hn⟩
      fin_cases n
        simp [is_divisible] at *
        contradiction,
        [1, 2, 3, 5, 6, 9].cases_on (λ n, n ∈ valid_digits) (by simp; exact valid_digits_card)
  
  sorry

end count_divisible_digits_l758_758071


namespace num_divisible_digits_l758_758053

def divisible_by_n (num : ℕ) (n : ℕ) : Prop :=
  n ≠ 0 ∧ num % n = 0

def count_divisible_digits : ℕ :=
  (List.filter (λ n => divisible_by_n (150 + n) n)
    [1, 2, 3, 4, 5, 6, 7, 8, 9]).length

theorem num_divisible_digits : count_divisible_digits = 7 := by
  sorry

end num_divisible_digits_l758_758053


namespace symmetric_axis_of_quadratic_fn_l758_758787

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 + 8 * x + 9 

-- State the theorem that the axis of symmetry for the quadratic function y = x^2 + 8x + 9 is x = -4
theorem symmetric_axis_of_quadratic_fn : ∃ h : ℝ, h = -4 ∧ ∀ x, quadratic_function x = quadratic_function (2 * h - x) :=
by sorry

end symmetric_axis_of_quadratic_fn_l758_758787


namespace find_m_for_collinear_vectors_l758_758565

theorem find_m_for_collinear_vectors :
  ∃ m : ℝ, let a : ℝ × ℝ := (2, 3),
                b : ℝ × ℝ := (-1, 2),
                v1 : ℝ × ℝ := (2 * m + 4, 3 * m + 8),
                v2 : ℝ × ℝ := (4, -1)
            in collinear (λ p : ℝ × ℝ, ∃ k : ℝ, p = (k * (fst v2), k * (snd v2))) v1 ∧ m = -2 := 
sorry

end find_m_for_collinear_vectors_l758_758565


namespace circle_properties_radius_properties_l758_758154

theorem circle_properties (m x y : ℝ) :
  (x^2 + y^2 - 2*(m + 3)*x + 2*(1 - 4*m^2)*y + 16*m^4 + 9 = 0) ↔
    (-((1 : ℝ) / (7 : ℝ)) < m ∧ m < (1 : ℝ)) :=
sorry

theorem radius_properties (m : ℝ) (h : -((1 : ℝ) / (7 : ℝ)) < m ∧ m < (1 : ℝ)) :
  ∃ r : ℝ, (0 < r ∧ r ≤ (4 / Real.sqrt 7)) :=
sorry

end circle_properties_radius_properties_l758_758154


namespace part1_not_monotonic_range_k_part2_max_k_for_inequality_l758_758556

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x
noncomputable def g (x k : ℝ) : ℝ := k * x^3 - x - 2

-- Part 1: Prove that the range of k such that g(x) is not monotonic in (1, 2) is (1/12, 1/3).
theorem part1_not_monotonic_range_k :
  ∀ k : ℝ, (g' (x : ℝ) (k : ℝ) := 3 * k * x^2 - 1) → ¬ (MonoDec (1 < x) ∧ MonoInc (x < 2)) ↔ (1 / 12 < k ∧ k < 1 / 3) :=
sorry

-- Part 2: Prove that the maximum value of k such that f(x) ≥ g(x) for all x in [0, +∞) is 1/6.
theorem part2_max_k_for_inequality :
  ∃ k : ℝ, f (x : ℝ) ≥ g x k ∧ (∀ x : ℝ, 0 ≤ x → (x < +∞ → f x ≥ g x k)) ↔ (0 < k ∧ k ≤ 1 / 6) :=
sorry

end part1_not_monotonic_range_k_part2_max_k_for_inequality_l758_758556


namespace total_earnings_correct_l758_758911

-- Given conditions
def charge_oil_change : ℕ := 20
def charge_repair : ℕ := 30
def charge_car_wash : ℕ := 5

def number_oil_changes : ℕ := 5
def number_repairs : ℕ := 10
def number_car_washes : ℕ := 15

-- Calculation of earnings based on the conditions
def earnings_from_oil_changes : ℕ := charge_oil_change * number_oil_changes
def earnings_from_repairs : ℕ := charge_repair * number_repairs
def earnings_from_car_washes : ℕ := charge_car_wash * number_car_washes

-- The total earnings
def total_earnings : ℕ := earnings_from_oil_changes + earnings_from_repairs + earnings_from_car_washes

-- Proof statement: Prove that the total earnings are $475
theorem total_earnings_correct : total_earnings = 475 := by -- our proof will go here
  sorry

end total_earnings_correct_l758_758911


namespace arithmetic_square_root_of_sqrt_81_l758_758746

-- Define the concept of arithmetic square root as the non-negative square root of a number.
def arithmetic_sqrt (x : ℝ) : ℝ := if x < 0 then 0 else Real.sqrt x

-- The main theorem statement:
theorem arithmetic_square_root_of_sqrt_81 : arithmetic_sqrt (Real.sqrt 81) = 9 := by
  -- We provide the proof later.
  sorry

end arithmetic_square_root_of_sqrt_81_l758_758746


namespace product_of_roots_positive_imag_part_l758_758432

/-- Given the polynomial \(z^7 - z^5 + z^4 + z^3 + z^2 - z + 1 = 0\), prove that
the angle \(\theta\) for the product of the roots with a positive imaginary part
expressed as \(r(\cos{\theta}+i\sin{\theta})\), where \(r > 0\) and \(0 \leq \theta < 360\), is \(308.58^\circ\). -/
theorem product_of_roots_positive_imag_part (θ : ℝ) :
  0 ≤ θ ∧ θ < 360 ∧
  (∃ r > 0, (z^7 - z^5 + z^4 + z^3 + z^2 - z + 1 = 0) ∧
  (∃ p, p = r * (cos θ + complex.I * sin θ)) ∧
  (θ = 308.58)) :=
sorry

end product_of_roots_positive_imag_part_l758_758432


namespace intersection_of_A_and_B_l758_758509

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} :=
by
  sorry

end intersection_of_A_and_B_l758_758509


namespace octagon_area_from_square_l758_758759

theorem octagon_area_from_square (a : ℝ) :
  let x := a / 2 * (2 - Real.sqrt 2) in
  (a > 0) →
  (2 * a^2 * (Real.sqrt 2 - 1) = a^2 - 2 * x^2) :=
by
  intro h
  let x := a / 2 * (2 - Real.sqrt 2)
  have x_squared_expr : x^2 = (a / 2 * (2 - Real.sqrt 2))^2 := by sorry
  have small_triangles_area : 4 * (1 / 2 * x^2) = 2 * x^2 := by sorry
  have total_cut_area : a^2 - 2 * x^2 = 2 * a^2 * (Real.sqrt 2 - 1) := by sorry
  exact total_cut_area

end octagon_area_from_square_l758_758759


namespace complementary_angles_positive_difference_l758_758767

theorem complementary_angles_positive_difference :
  ∀ (θ₁ θ₂ : ℝ), (θ₁ + θ₂ = 90) → (θ₁ = 3 * θ₂) → (|θ₁ - θ₂| = 45) :=
by
  intros θ₁ θ₂ h₁ h₂
  sorry

end complementary_angles_positive_difference_l758_758767


namespace part1_part2_l758_758159

noncomputable theory

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + m * x + 3

-- Proof problem for part (1)
theorem part1 (m : ℝ) : (∀ x : ℝ, f m x > 0) ↔ (0 ≤ m ∧ m < 12) :=
sorry

-- Proof problem for part (2)
theorem part2 (m x : ℝ) : f m x > (3 * m - 1) * x + 5 ↔ 
  (m < -1/2 → -1/m < x ∧ x < 2) ∧
  (m = -1/2 → False) ∧
  (-1/2 < m ∧ m < 0 → 2 < x ∧ x < -1/m) ∧
  (m = 0 → 2 < x) ∧
  (m > 0 → x < -1/m ∨ x > 2) :=
sorry

end part1_part2_l758_758159


namespace count_divisible_digits_l758_758070

def is_divisible (a b : ℕ) : Prop := a % b = 0

theorem count_divisible_digits :
  let count := (finset.range 10).filter (λ n, n > 0 ∧ is_divisible (15 * n) n) in
  count.card = 6 :=
by
  let valid_digits := [1, 2, 3, 5, 6, 9].to_finset
  have valid_digits_card : valid_digits.card = 6 := by simp
  have matching_digits : ∀ n : ℕ, n > 0 ∧ n < 10 ∧ 15 * n % n = 0 ↔ n ∈ valid_digits := by
    intros n
    simp
    split
      intros ⟨n_pos, n_lt, hn⟩
      fin_cases n
        simp [is_divisible] at *
        contradiction,
        [1, 2, 3, 5, 6, 9].cases_on (λ n, n ∈ valid_digits) (by simp; exact valid_digits_card)
  
  sorry

end count_divisible_digits_l758_758070


namespace focus_of_parabola_l758_758006

theorem focus_of_parabola (f d : ℝ) :
  (∀ x : ℝ, x^2 + (4*x^2 - f)^2 = (4*x^2 - d)^2) → 8*f + 8*d = 1 → f^2 = d^2 → f = 1/16 :=
by
  intro hEq hCoeff hSq
  sorry

end focus_of_parabola_l758_758006


namespace sphere_diameter_l758_758434

open Real

def sphere_volume (r : ℝ) : ℝ := (4/3) * π * r^3

theorem sphere_diameter (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) 
  (b_cubefree : ∀ k : ℕ, k^3 ∣ b → k = 1) :
  let r₁ := 6
  let V₁ := sphere_volume r₁
  let V₂ := 3 * V₁
  let r₂ := (V₂ / (4 / 3 * π))^(1 / 3)
  let d₂ := 2 * r₂ in
  d₂ = a * (b:ℝ)^(1/3) → a = 12 ∧ b = 3 :=
by
  sorry

end sphere_diameter_l758_758434


namespace general_formula_a_sum_formula_T_l758_758982

section 
  -- Define the sequence and its summation function
  def a (n : ℕ) : ℕ :=
    if n = 1 then 1 else 4 * 3^(n - 2)

  def S (n : ℕ) : ℕ := Nat.sum (Finset.range n) (λi, a (i + 1))

  -- given conditions
  variable {n : ℕ}

  -- Condition a_1 = 1
  axiom a1 : a 1 = 1

  -- Condition S_{n+1} = 3S_n + 2
  axiom Sn : S (n + 1) = 3 * S n + 2

  -- Question 1: Prove general formula for a_n
  theorem general_formula_a (n : ℕ) : a n = 
    if n = 1 then 1 else 4 * 3^(n - 2) :=
  sorry

  -- Define b_n
  def b (n : ℕ) : ℝ :=
    if n = 1 
    then 8 / (a 2 - a 1) 
    else 8 * n / (a (n + 1) - a n)

  -- Define T_n based on b_n
  def T (n : ℕ) : ℝ := Nat.sum (Finset.range n) (λ i, b (i + 1))

  -- Question 2: Prove sum formula for T_n
  theorem sum_formula_T (n : ℕ) : T n = 
    (77 / 12) - ((n / 2) + (3 / 4)) * (1 / 3) ^ (n - 2) :=
  sorry
end

end general_formula_a_sum_formula_T_l758_758982


namespace sqrt_sum_eq_2_sqrt_2_l758_758933

theorem sqrt_sum_eq_2_sqrt_2 : sqrt (5 - 2 * sqrt 6) + sqrt (5 + 2 * sqrt 6) = 2 * sqrt 2 :=
  sorry

end sqrt_sum_eq_2_sqrt_2_l758_758933


namespace quadratic_complex_roots_condition_l758_758772

theorem quadratic_complex_roots_condition (l : ℝ) :
  (1 - complex.i) * (1 - complex.i) ≠ 0 ∧
  complex.re ( (1 - complex.i) * (1 - complex.i) ) = complex.re ( - (1 + complex.i * l) * (4*(1 - complex.i)) ) 
  → l ≠ 2 := sorry

end quadratic_complex_roots_condition_l758_758772


namespace cross_product_computation_l758_758576

variable {V : Type} [AddCommGroup V] [Module ℝ V] [NormedSpace ℝ V]
variable (v w : V)
variables (v_cross_w : V)

-- Condition
axiom hvw : v × w = ⟨5, -2, 4⟩

-- Theorem to prove
theorem cross_product_computation :
  (2 • v - w) × (3 • w - v) = ⟨25, -10, 20⟩ := 
by
  sorry

end cross_product_computation_l758_758576


namespace number_of_divisors_eighteen_l758_758570

-- Define what it means for a number to be a divisor of another number.
def is_divisor (a b : ℕ) : Prop := b % a = 0

-- Define the set of divisors of a number n.
def divisors (n : ℕ) : set ℕ := { d | is_divisor d n }

-- Define the number 18
def eighteen : ℕ := 18

-- Prove that the set of divisors of 18 has 6 elements
theorem number_of_divisors_eighteen : (divisors eighteen).to_finset.card = 6 :=
sorry

end number_of_divisors_eighteen_l758_758570


namespace no_division_into_three_similar_piles_l758_758663

theorem no_division_into_three_similar_piles :
    ∀ (x : ℝ),
    ∀ (y z : ℝ),
    (x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = x) →
    (x <= sqrt 2 * y ∧ y <= sqrt 2 * z ∧ z <= sqrt 2 * x) →
    false :=
by
  intro x y z
  sorry

end no_division_into_three_similar_piles_l758_758663


namespace impossible_to_divide_three_similar_parts_l758_758697

theorem impossible_to_divide_three_similar_parts 
  (n : ℝ) 
  (p : ℝ → Prop) 
  (similar : ℝ → ℝ → Prop) 
  (h_similar : ∀ a b : ℝ, similar a b ↔ a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2) : 
  ¬ ∃ p1 p2 p3 : ℝ, p p1 ∧ p p2 ∧ p p3 ∧ (p1 + p2 + p3 = n) ∧ similar p1 p2 ∧ similar p2 p3 ∧ similar p1 p3 :=
sorry

end impossible_to_divide_three_similar_parts_l758_758697


namespace selling_price_at_6_kg_selling_price_at_6_kg_statement_l758_758709

-- Define the relationship between weight (in kg) and price (in yuan)
def price (x : ℝ) : ℝ := 1.4 * x

-- The theorem stating our proof problem
theorem selling_price_at_6_kg : price 6 = 8.4 :=
by { simp [price], norm_num }

#eval selling_price_at_6_kg -- To output the result as a check

-- Note: We could have also simply written the statement without the proof:
theorem selling_price_at_6_kg_statement : price 6 = 8.4 :=
sorry -- skipping the proof

end selling_price_at_6_kg_selling_price_at_6_kg_statement_l758_758709


namespace right_triangle_bisector_perpendicular_l758_758203

theorem right_triangle_bisector_perpendicular (A B C L K N : Point) (h_right : ∠C = 90) (h_non_isosceles : A ≠ B)
  (h_angle_bisector : is_angle_bisector CL)
  (h_K_on_hypotenuse : K ∈ segment AB) (h_AL_eq_BK : dist A L = dist B K)
  (h_perpendicular : perpendicular_to_through_point K AB CL N) :
  dist K N = dist A B :=
sorry

end right_triangle_bisector_perpendicular_l758_758203


namespace a_n_negative_S_n_maximum_l758_758782

def S (n : ℕ) : ℕ → ℤ := λ n, -n^2 + 7 * n

-- 1. Prove ∀ n > 4, a_n < 0 given S_n = -n^2 + 7n
theorem a_n_negative (n : ℕ) (h : n > 4) : ∀ n > 4, (-2 * (↑n : ℤ) + 8) < 0 :=
by {
  assume (n : ℕ) (h : n > 4),
  calc (-2 * (↑n : ℤ) + 8)
      = 8 - 2 * (↑n : ℤ) : by ring
  ... < 0 : by linarith,
}

-- 2. Prove S_n reaches its maximum value when n = 3 or 4 given S_n = -n^2 + 7n
theorem S_n_maximum : ∃ (n : ℕ) (max1 max2 : ℕ → ℤ), 
  (S 3 ≥ S n) ∧ (S 4 ≥ S n) :=
by {
  use 3,
  use 4,
  assume (n : ℕ),
  have axis_symmetry : 7 / 2 = 3.5 := by norm_num,
  by_cases h1: n ≤ 3,
  { linarith },
  { have h2 : 3 < n,
    from lt_of_not_ge h1, 
    linarith,
  },
}

#check a_n_negative
#check S_n_maximum

end a_n_negative_S_n_maximum_l758_758782


namespace circumcircle_tangent_l758_758619

def cyclic (K : Type*) (p₁ p₂ p₃ p₄ : K) : Prop := sorry

def tangent_circles (c₁ c₂ : Type*) (p : c₁) : Prop := sorry 

variables {K : Type*} [field K]

theorem circumcircle_tangent
  (A B C O X Y : K)
  (h1 : cyclic K A X Y X)
  (h2 : cyclic K B O C O)
  (h3 : B ∈ line_through X Y)
  (h4 : C ∈ line_through X Y)
  (h5 : reflection_over_line B C X Y = tangent (circumcircle A X Y)) :
  tangent_circles (circumcircle A X Y) (circumcircle B O C) :=
sorry

end circumcircle_tangent_l758_758619


namespace hyperbola_eccentricity_l758_758114

def hyperbola_foci (F1 F2 P : ℝ) (θ : ℝ) (PF1 PF2 : ℝ) : Prop :=
  θ = 60 ∧ PF1 = 3 * PF2

def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity (F1 F2 P : ℝ) (θ PF1 PF2 : ℝ)
  (h : hyperbola_foci F1 F2 P θ PF1 PF2) :
  eccentricity 1 (sqrt 7 / 2) = sqrt 7 / 2 :=
by {
  sorry
}

end hyperbola_eccentricity_l758_758114


namespace intersection_eq_l758_758522

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_eq : A ∩ B = ({-2, 2} : Set ℤ) :=
by
  sorry

end intersection_eq_l758_758522


namespace count_valid_n_divisibility_l758_758073

theorem count_valid_n_divisibility : 
  (finset.univ.filter (λ n : ℕ, n ∈ finset.range 10 ∧ (15 * n) % n = 0)).card = 5 :=
by sorry

end count_valid_n_divisibility_l758_758073


namespace find_focus_of_parabola_4x2_l758_758021

-- Defining what it means to be the focus of a parabola given an equation y = ax^2.
def is_focus (a : ℝ) (f : ℝ) : Prop :=
  ∀ (x : ℝ), (x ^ 2 + (a * x ^ 2 - f) ^ 2) = ((a * x ^ 2 - (f * 8)) ^ 2)

-- Specific instance of the parabola y = 4x^2.
def parabola_4x2 := (4 : ℝ)

theorem find_focus_of_parabola_4x2 : ∃ f : ℝ, is_focus parabola_4x2 f :=
begin
  use (1/16 : ℝ),
  sorry -- The proof will be filled in by the theorem prover.
end

end find_focus_of_parabola_4x2_l758_758021


namespace atomic_number_l758_758150

theorem atomic_number (mass_number : ℕ) (neutrons : ℕ) (protons : ℕ) :
  mass_number = 288 →
  neutrons = 169 →
  (protons = mass_number - neutrons) →
  protons = 119 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end atomic_number_l758_758150


namespace smallest_constant_M_l758_758951

theorem smallest_constant_M :
  ∃ M : ℝ, (∀ x y z w : ℝ, 0 < x → 0 < y → 0 < z → 0 < w →
  (real.sqrt (x / (y + z + w)) + real.sqrt (y / (x + z + w)) + real.sqrt (z / (x + y + w)) + real.sqrt (w / (x + y + z)) < M)) ∧
  M = 4 / real.sqrt 3 :=
sorry

end smallest_constant_M_l758_758951


namespace angle_between_vectors_l758_758171

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem angle_between_vectors :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-1, 3)
  let θ := real.arccos (dot_product a b / (magnitude a * magnitude b))
  θ = real.pi / 4 :=
by
  sorry

end angle_between_vectors_l758_758171


namespace impossible_to_divide_into_three_similar_piles_l758_758649

def similar (a b : ℝ) : Prop :=
  a / b ≤ real.sqrt 2 ∧ b / a ≤ real.sqrt 2

theorem impossible_to_divide_into_three_similar_piles (pile : ℝ) (h : 0 < pile) :
  ¬ ∃ (x y z : ℝ), 
    x + y + z = pile ∧
    similar x y ∧ similar y z ∧ similar z x :=
by
  sorry

end impossible_to_divide_into_three_similar_piles_l758_758649


namespace rectangle_area_l758_758406

theorem rectangle_area (c h x : ℝ) (h_pos : 0 < h) (c_pos : 0 < c) : 
  (A : ℝ) = (x * (c * x / h)) :=
by
  sorry

end rectangle_area_l758_758406


namespace range_of_t_given_sets_l758_758167

variable (x t : ℝ)

def setA : set ℝ := {x | (x + 8) / (x - 5) ≤ 0}
def setB : set ℝ := {x | t + 1 ≤ x ∧ x ≤ 2 * t - 1}

theorem range_of_t_given_sets :
  setB t ≠ ∅ → (setA ∩ setB t) = ∅ → t ≥ 4 := by
  sorry

end range_of_t_given_sets_l758_758167


namespace num_terminating_decimals_l758_758473

theorem num_terminating_decimals : 
  ∃ k, k = 24 ∧ ∀ n, 1 ≤ n ∧ n ≤ 474 → (decimal_representation_terminates (n / 475) ↔ n % 19 = 0) :=
by
  sorry

end num_terminating_decimals_l758_758473


namespace region_divided_by_line_l758_758429

noncomputable theory
open_locale classical

-- Declare the necessary parameters and definitions
def circle (c : ℝ × ℝ) (r : ℝ) : set (ℝ × ℝ) := {p | (p.1 - c.1)^2 + (p.2 - c.2)^2 < r^2}
def line (a b c : ℝ) : set (ℝ × ℝ) := {p | a * p.1 = b * p.2 + c}

-- Problem statement
theorem region_divided_by_line :
  let centers := [(1, 1), (3, 1), (5, 1), (1, 3), (3, 3), (5, 3), (1, 5), (3, 5), (5, 5), (1, 7), (3, 7), (5, 7)]
  let circles := centers.map (λ c, circle c 1)
  let region_𝒬 := ⋃ c in circles, c
  let line_m := line 2 (-1) 0
  (∃ m : ℝ → ℝ, (∀ p ∈ region_𝒬, p.2 = 2 * p.1 <-> p ∈ region_𝒬) ∧ m 1 = 2 ∧ m 3 = 4) ∧ 2^2 + (-1)^2 + 0^2 = 5 :=
sorry

end region_divided_by_line_l758_758429


namespace _l758_758135

noncomputable def hyperbola_eccentricity_theorem {F1 F2 P : Point} 
  (hyp : is_hyperbola F1 F2 P)
  (angle_F1PF2 : ∠F1 P F2 = 60)
  (dist_PF1_3PF2 : dist P F1 = 3 * dist P F2) : 
  eccentricity F1 F2 = sqrt 7 / 2 :=
by 
  sorry

end _l758_758135


namespace tangent_line_y_intercept_eq_5_l758_758866

open Real

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem tangent_line_y_intercept_eq_5
  (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ)
  (h1 : c1 = (3, 1)) (h2 : r1 = 3)
  (h3 : c2 = (7, 0)) (h4 : r2 = 2)
  (tangent : ∀ p : ℝ × ℝ, dist p c1 = r1 → dist p c2 = r2 → (p.1 > 0 ∧ p.2 > 0)) :
  ∃ y : ℝ, y = 5 :=
by
  exists 5
  sorry

end tangent_line_y_intercept_eq_5_l758_758866


namespace least_positive_integer_with_eight_factors_l758_758813

theorem least_positive_integer_with_eight_factors :
  ∃ n : ℕ, n = 24 ∧ (8 = (nat.factors n).length) :=
sorry

end least_positive_integer_with_eight_factors_l758_758813


namespace distinct_convex_polygons_l758_758799

def twelve_points : Finset (Fin 12) := (Finset.univ : Finset (Fin 12))

noncomputable def polygon_count_with_vertices (n : ℕ) : ℕ :=
  2^n - 1 - n - (n * (n - 1)) / 2

theorem distinct_convex_polygons :
  polygon_count_with_vertices 12 = 4017 := 
by
  sorry

end distinct_convex_polygons_l758_758799


namespace count_divisible_digits_l758_758068

def is_divisible (a b : ℕ) : Prop := a % b = 0

theorem count_divisible_digits :
  let count := (finset.range 10).filter (λ n, n > 0 ∧ is_divisible (15 * n) n) in
  count.card = 6 :=
by
  let valid_digits := [1, 2, 3, 5, 6, 9].to_finset
  have valid_digits_card : valid_digits.card = 6 := by simp
  have matching_digits : ∀ n : ℕ, n > 0 ∧ n < 10 ∧ 15 * n % n = 0 ↔ n ∈ valid_digits := by
    intros n
    simp
    split
      intros ⟨n_pos, n_lt, hn⟩
      fin_cases n
        simp [is_divisible] at *
        contradiction,
        [1, 2, 3, 5, 6, 9].cases_on (λ n, n ∈ valid_digits) (by simp; exact valid_digits_card)
  
  sorry

end count_divisible_digits_l758_758068


namespace total_earnings_correct_l758_758910

-- Given conditions
def charge_oil_change : ℕ := 20
def charge_repair : ℕ := 30
def charge_car_wash : ℕ := 5

def number_oil_changes : ℕ := 5
def number_repairs : ℕ := 10
def number_car_washes : ℕ := 15

-- Calculation of earnings based on the conditions
def earnings_from_oil_changes : ℕ := charge_oil_change * number_oil_changes
def earnings_from_repairs : ℕ := charge_repair * number_repairs
def earnings_from_car_washes : ℕ := charge_car_wash * number_car_washes

-- The total earnings
def total_earnings : ℕ := earnings_from_oil_changes + earnings_from_repairs + earnings_from_car_washes

-- Proof statement: Prove that the total earnings are $475
theorem total_earnings_correct : total_earnings = 475 := by -- our proof will go here
  sorry

end total_earnings_correct_l758_758910


namespace f_eq_l758_758156

def f (x : ℝ) : ℝ :=
if x > 0 then log x / log 2 else (3 : ℝ) ^ x

theorem f_eq :
  f 2 + f (-2) = 10 / 9 := by
  sorry

end f_eq_l758_758156


namespace hyperbola_eccentricity_l758_758115

def hyperbola_foci (F1 F2 P : ℝ) (θ : ℝ) (PF1 PF2 : ℝ) : Prop :=
  θ = 60 ∧ PF1 = 3 * PF2

def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity (F1 F2 P : ℝ) (θ PF1 PF2 : ℝ)
  (h : hyperbola_foci F1 F2 P θ PF1 PF2) :
  eccentricity 1 (sqrt 7 / 2) = sqrt 7 / 2 :=
by {
  sorry
}

end hyperbola_eccentricity_l758_758115


namespace n_minus_m_eq_singleton_6_l758_758917

def set_difference (A B : Set α) : Set α :=
  {x | x ∈ A ∧ x ∉ B}

def M : Set ℕ := {1, 2, 3, 5}
def N : Set ℕ := {2, 3, 6}

theorem n_minus_m_eq_singleton_6 : set_difference N M = {6} :=
by
  sorry

end n_minus_m_eq_singleton_6_l758_758917


namespace impossibility_of_dividing_into_three_similar_piles_l758_758660

theorem impossibility_of_dividing_into_three_similar_piles:
  ∀ (x y z : ℝ), ¬ (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x + y + z = 1  ∧ (x ≤ sqrt 2 * y ∧ y ≤ sqrt 2 * x) ∧ (y ≤ sqrt 2 * z ∧ z ≤ sqrt 2 * y) ∧ (z ≤ sqrt 2 * x ∧ x ≤ sqrt 2 * z)) :=
by
  sorry

end impossibility_of_dividing_into_three_similar_piles_l758_758660


namespace impossibility_of_dividing_into_three_similar_piles_l758_758653

theorem impossibility_of_dividing_into_three_similar_piles:
  ∀ (x y z : ℝ), ¬ (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x + y + z = 1  ∧ (x ≤ sqrt 2 * y ∧ y ≤ sqrt 2 * x) ∧ (y ≤ sqrt 2 * z ∧ z ≤ sqrt 2 * y) ∧ (z ≤ sqrt 2 * x ∧ x ≤ sqrt 2 * z)) :=
by
  sorry

end impossibility_of_dividing_into_three_similar_piles_l758_758653


namespace total_fills_l758_758415

open Real

-- Definitions based on conditions
def needs_flour : ℝ := 10 / 3
def needs_milk : ℝ := 3 / 2
def cup_capacity : ℝ := 1 / 3

-- Theorem stating the total number of fills required
theorem total_fills (needs_flour needs_milk cup_capacity : ℝ) : 
  let flour_fills := needs_flour / cup_capacity
  let milk_fills := (needs_milk / cup_capacity).ceil
  flour_fills + milk_fills = 15 :=
by
  sorry

end total_fills_l758_758415


namespace find_age_of_B_l758_758194

-- Define A and B as natural numbers (assuming ages are non-negative integers)
variables (A B : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := A + 10 = 2 * (B - 10)
def condition2 : Prop := A = B + 6

-- The goal is to prove that B = 36 given the conditions
theorem find_age_of_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 36 :=
sorry

end find_age_of_B_l758_758194


namespace max_regular_hours_l758_758396

/-- A man's regular pay is $3 per hour up to a certain number of hours, and his overtime pay rate
    is twice the regular pay rate. The man was paid $180 and worked 10 hours overtime.
    Prove that the maximum number of hours he can work at his regular pay rate is 40 hours.
-/
theorem max_regular_hours (P R OT : ℕ) (hP : P = 180) (hOT : OT = 10) (reg_rate overtime_rate : ℕ)
  (hreg_rate : reg_rate = 3) (hovertime_rate : overtime_rate = 2 * reg_rate) :
  P = reg_rate * R + overtime_rate * OT → R = 40 :=
by
  sorry

end max_regular_hours_l758_758396


namespace find_focus_of_parabola_4x2_l758_758020

-- Defining what it means to be the focus of a parabola given an equation y = ax^2.
def is_focus (a : ℝ) (f : ℝ) : Prop :=
  ∀ (x : ℝ), (x ^ 2 + (a * x ^ 2 - f) ^ 2) = ((a * x ^ 2 - (f * 8)) ^ 2)

-- Specific instance of the parabola y = 4x^2.
def parabola_4x2 := (4 : ℝ)

theorem find_focus_of_parabola_4x2 : ∃ f : ℝ, is_focus parabola_4x2 f :=
begin
  use (1/16 : ℝ),
  sorry -- The proof will be filled in by the theorem prover.
end

end find_focus_of_parabola_4x2_l758_758020


namespace valid_digits_count_l758_758060

theorem valid_digits_count : { n ∈ Finset.range 10 | n > 0 ∧ 15 * n % n = 0 }.card = 5 :=
by
  sorry

end valid_digits_count_l758_758060


namespace min_length_tangent_circle_l758_758426

theorem min_length_tangent_circle :
  ∀ (x₀ y₀ : ℝ), (x₀^2 + y₀^2 = 1) →
    let A := (2 * x₀^2, 0)
    let B := (0, 2 * y₀^2)
    ∃ (min_dist : ℝ), min_dist = 2 ∧ dist A B ≥ min_dist :=
begin
  intros x₀ y₀ h,
  let A := (2 * x₀^2, 0),
  let B := (0, 2 * y₀^2),
  use 2,
  split,
  {
    refl,
  },
  {
    have h1 : x₀^4 + y₀^4 ≤ 1,
    {
      calc x₀^4 + y₀^4
          = (x₀^2)^2 + (y₀^2)^2 : by rw [pow_two, pow_two, pow_two, pow_two]
      ... ≤ (x₀^2 + y₀^2)^2      : by apply sum_squares_le_square_sum,
      rw h,
      norm_num,
    },
    calc dist A B
        = real.sqrt ((2 * x₀^2 - 0)^2 + (0 - 2 * y₀^2)^2) : by simp [dist]
    ... = real.sqrt (4 * x₀^4 + 4 * y₀^4) : by ring
    ... = 2 * real.sqrt (x₀^4 + y₀^4) : by rw real.sqrt_mul (by norm_num) (x₀^4 + y₀^4)
    ... ≥ 2 * real.sqrt 1 : by { apply mul_le_mul_left, norm_num, exact real.sqrt_le_sqrt h1, },
    rw real.sqrt_one
  }
end

end min_length_tangent_circle_l758_758426


namespace optimal_washing_effect_l758_758853

noncomputable def total_capacity : ℝ := 20 -- kilograms
noncomputable def weight_clothes : ℝ := 5 -- kilograms
noncomputable def weight_detergent_existing : ℝ := 2 * 0.02 -- kilograms
noncomputable def optimal_concentration : ℝ := 0.004 -- kilograms per kilogram of water

theorem optimal_washing_effect :
  ∃ (additional_detergent additional_water : ℝ),
    additional_detergent = 0.02 ∧ additional_water = 14.94 ∧
    weight_clothes + additional_water + weight_detergent_existing + additional_detergent = total_capacity ∧
    weight_detergent_existing + additional_detergent = optimal_concentration * additional_water :=
by
  sorry

end optimal_washing_effect_l758_758853


namespace delta_product_range_exists_t_for_eta_AB_length_comparison_l758_758605

-- Problem (1)
theorem delta_product_range 
    (x y : ℝ) 
    (h : x^2 / 4 + y^2 = 1) : 
    ∃ (d1 d2 : ℝ), 
    (d1 = (x - 2 * y) / (Real.sqrt 5)) ∧ 
    (d2 = (x + 2 * y) / (Real.sqrt 5)) ∧ 
    (-4/5 ≤ (d1 * d2) ∧ (d1 * d2) ≤ 4/5) :=
sorry

-- Problem (2)
theorem exists_t_for_eta 
    (α t : ℝ) : 
    let e1 := (-t * Real.cos α - 2) / (Real.sqrt (Real.cos α^2 + 4 * Real.sin α^2)) in
    let e2 := (t * Real.cos α - 2) / (Real.sqrt (Real.cos α^2 + 4 * Real.sin α^2)) in
    (∀ α, (e1 * e2 = 1) ↔ t = Real.sqrt 3 ∨ t = -Real.sqrt 3) :=
sorry

-- Problem (3)
theorem AB_length_comparison 
    {m n a b : ℝ}
    (h₀ : a > b ∧ b > 0)
    (F1x := -Real.sqrt (a^2 - b^2))
    (F2x := Real.sqrt (a^2 - b^2))
    (λ1 := (n - m * F1x) / (Real.sqrt (1 + m^2)))
    (λ2 := (n + m * F2x) / (Real.sqrt (1 + m^2)))
    (h₁ : λ1 * λ2 > b^2) :
    let A := (-n / m, 0) in 
    let B := (0, n) in 
    Float.sqrt((-n / m)^2 + n^2) > a + b := 
sorry

end delta_product_range_exists_t_for_eta_AB_length_comparison_l758_758605


namespace time_to_cover_distance_l758_758880

theorem time_to_cover_distance
  (length_main : ℝ := 5280) -- feet
  (width : ℝ := 30) -- feet
  (extension : ℝ := 528) -- feet
  (speed : ℝ := 8) -- miles per hour
  (mile_in_feet : ℝ := 5280)
  :
  (let radius := width / 2 -- radius of semicircles
       n_semi := length_main / width -- number of semicircles
       semi_distance := n_semi * (2 * radius * Real.pi / 2) -- distance covered in semicircles
       total_distance := semi_distance + extension -- total distance in feet
       total_miles := total_distance / mile_in_feet -- total distance in miles
    in total_miles / speed = (Real.pi + 0.2) / 16) 
  :=
sorry

end time_to_cover_distance_l758_758880


namespace red_marbles_initial_count_l758_758182

theorem red_marbles_initial_count (r g : ℕ) 
  (h1 : 3 * r = 5 * g)
  (h2 : 4 * (r - 18) = g + 27) :
  r = 29 :=
sorry

end red_marbles_initial_count_l758_758182


namespace part_I_extreme_values_part_II_three_distinct_real_roots_part_III_compare_sizes_l758_758552

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := (1 / 3) * x^3 + (1 / 2) * (p - 1) * x^2 + q * x

theorem part_I_extreme_values : 
  (∀ x, f x (-3) 3 = (1 / 3) * x^3 - 2 * x^2 + 3 * x) → 
  (f 1 (-3) 3 = f 3 (-3) 3) := 
sorry

theorem part_II_three_distinct_real_roots : 
  (∀ x, f x (-3) 3 = (1 / 3) * x^3 - 2 * x^2 + 3 * x) → 
  (∀ g : ℝ → ℝ, g x = f x (-3) 3 - 1 → 
  (∀ x, g x ≠ 0) → 
  ∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ g a = 0 ∧ g b = 0 ∧ g c = 0) :=
sorry

theorem part_III_compare_sizes (x1 x2 p a l q: ℝ) :
  f (x : ℝ) (-3) 3 = (1 / 3) * x^3 - 2 * x^2 + 3 * x → 
  x1 < x2 → 
  x2 - x1 > l → 
  x1 > a → 
  (a^2 + p * a + q) > x1 := 
sorry

end part_I_extreme_values_part_II_three_distinct_real_roots_part_III_compare_sizes_l758_758552


namespace impossible_divide_into_three_similar_l758_758670

noncomputable def sqrt2 : ℝ := Real.sqrt 2

def similar (x y : ℝ) : Prop :=
  x ≤ sqrt2 * y

theorem impossible_divide_into_three_similar (N : ℝ) :
  ¬ ∃ (x y z : ℝ), x + y + z = N ∧ similar x y ∧ similar y z ∧ similar x z := 
by
  sorry

end impossible_divide_into_three_similar_l758_758670


namespace probability_red_ball_first_occurrence_l758_758079

theorem probability_red_ball_first_occurrence 
  (P : ℕ → ℝ) : 
  ∃ (P1 P2 P3 P4 : ℝ),
    P 1 = 0.4 ∧ P 2 = 0.3 ∧ P 3 = 0.2 ∧ P 4 = 0.1 :=
  sorry

end probability_red_ball_first_occurrence_l758_758079


namespace find_mass_of_substance_l758_758452

noncomputable def mass_of_substance
  (R k : ℝ)
  (sphere1 sphere2 : ℝ × ℝ × ℝ → Prop)
  (density : ℝ × ℝ × ℝ → ℝ)
  : ℝ :=
  ∫ (x y z : ℝ) in set_of (fun point => sphere1 point ∧ sphere2 point), density (x, y, z) * x * y * z

axiom sphere1_eq : ∀ (x y z : ℝ), sphere1 (x, y, z) = (x^2 + y^2 + z^2 = R^2)
axiom sphere2_eq : ∀ (x y z : ℝ), sphere2 (x, y, z) = (x^2 + y^2 + z^2 = 2 * R * z)
axiom density_eq : ∀ (x y z : ℝ), density (x, y, z) = k * z

theorem find_mass_of_substance 
  (R k : ℝ)
  (sphere1 sphere2 : ℝ × ℝ × ℝ → Prop)
  (density : ℝ × ℝ × ℝ → ℝ)
  : mass_of_substance R k sphere1 sphere2 density = - (5 * R^4 * k * real.pi / 24) :=
sorry

end find_mass_of_substance_l758_758452


namespace eval_modulus_expression_l758_758935

variable (ω : ℂ)

theorem eval_modulus_expression (h : ω = 5 + 4 * Complex.I) : |ω^2 + 4 * ω + 41| = 2 * Real.sqrt 2009 :=
by
  sorry

end eval_modulus_expression_l758_758935


namespace solve_large_bill_denomination_l758_758344

noncomputable def larger_bill_denomination (total_money : ℕ) (fraction : ℚ) 
  (small_bill_denom : ℕ) (total_bills : ℕ) : ℕ :=
  let small_bills_amount := (fraction * total_money)
  let num_small_bills := small_bills_amount / small_bill_denom
  let larger_bills := total_bills - num_small_bills
  let larger_bills_amount := total_money - small_bills_amount
  larger_bills_amount / larger_bills

theorem solve_large_bill_denomination :
  larger_bill_denomination 1000 (3 / 10) 50 13 = 100 :=
by
  /-
  The proof should go here.
  -/
  sorry

end solve_large_bill_denomination_l758_758344


namespace valid_digits_count_l758_758055

theorem valid_digits_count : { n ∈ Finset.range 10 | n > 0 ∧ 15 * n % n = 0 }.card = 5 :=
by
  sorry

end valid_digits_count_l758_758055


namespace different_graphs_l758_758841

def EquationI (x : ℝ) : ℝ := 2 * x + 3
def EquationII (x : ℝ) : ℝ := if x ≠ -3/2 then (4 * x^2 + 12 * x + 9) / (2 * x + 3) else 0
def EquationIII (x : ℝ) : Prop := (2 * x + 3) * (EquationI x) = 4 * x^2 + 12 * x + 9

theorem different_graphs : 
  ¬(∀ x : ℝ, EquationI x = EquationII x) ∧ 
  ¬(∀ x : ℝ, EquationI x = EquationIII x) ∧ 
  ¬(∀ x : ℝ, EquationII x = EquationIII x) :=
sorry

end different_graphs_l758_758841


namespace count_valid_n_divisibility_l758_758074

theorem count_valid_n_divisibility : 
  (finset.univ.filter (λ n : ℕ, n ∈ finset.range 10 ∧ (15 * n) % n = 0)).card = 5 :=
by sorry

end count_valid_n_divisibility_l758_758074


namespace number_of_elements_beginning_with_one_l758_758233

noncomputable def leading_digit_is_one_count : ℕ :=
  let S := {n : ℕ | ∃ k : ℕ, (0 ≤ k ∧ k ≤ 2004 ∧ n = 5^k)} in
  let has_leading_digit_one (x : ℕ) : Prop := x / (10 ^ ((Nat.log x / Nat.log 10 : ℝ).floor)) = 1 in
  {n ∈ S | has_leading_digit_one n}.card

theorem number_of_elements_beginning_with_one : leading_digit_is_one_count = 604 :=
sorry

end number_of_elements_beginning_with_one_l758_758233


namespace series_equals_one_half_l758_758464

noncomputable def series_sum : ℕ → ℚ
| k := 3^k / (9^k - 1)

theorem series_equals_one_half :
  ∑' k, series_sum k = 1 / 2 :=
sorry

end series_equals_one_half_l758_758464


namespace count_divisible_digits_l758_758072

def is_divisible (a b : ℕ) : Prop := a % b = 0

theorem count_divisible_digits :
  let count := (finset.range 10).filter (λ n, n > 0 ∧ is_divisible (15 * n) n) in
  count.card = 6 :=
by
  let valid_digits := [1, 2, 3, 5, 6, 9].to_finset
  have valid_digits_card : valid_digits.card = 6 := by simp
  have matching_digits : ∀ n : ℕ, n > 0 ∧ n < 10 ∧ 15 * n % n = 0 ↔ n ∈ valid_digits := by
    intros n
    simp
    split
      intros ⟨n_pos, n_lt, hn⟩
      fin_cases n
        simp [is_divisible] at *
        contradiction,
        [1, 2, 3, 5, 6, 9].cases_on (λ n, n ∈ valid_digits) (by simp; exact valid_digits_card)
  
  sorry

end count_divisible_digits_l758_758072


namespace price_of_mixture_l758_758293

theorem price_of_mixture (x : ℝ) :
  let cost1 := 126
  let cost2 := 135
  let cost3 := 177.5
  let quantity1 := x
  let quantity2 := x
  let quantity3 := 2 * x
  let total_cost := cost1 * quantity1 + cost2 * quantity2 + cost3 * quantity3
  let total_quantity := quantity1 + quantity2 + quantity3
  (total_cost / total_quantity) = 154 :=
by
  intros
  let cost1 := 126 : ℝ
  let cost2 := 135 : ℝ
  let cost3 := 177.5 : ℝ
  let quantity1 := x
  let quantity2 := x
  let quantity3 := 2 * x
  let total_cost := cost1 * quantity1 + cost2 * quantity2 + cost3 * quantity3
  let total_quantity := quantity1 + quantity2 + quantity3
  have step1 : total_cost = 616 * x, by sorry
  have step2 : total_quantity = 4 * x, by sorry
  have step3 : total_cost / total_quantity = 154, by sorry
  exact step3

end price_of_mixture_l758_758293


namespace area_of_inscribed_triangle_l758_758884

theorem area_of_inscribed_triangle 
  (x : ℝ) 
  (h1 : (2:ℝ) * x ≤ (3:ℝ) * x ∧ (3:ℝ) * x ≤ (4:ℝ) * x) 
  (h2 : (4:ℝ) * x = 2 * 4) :
  ∃ (area : ℝ), area = 12.00 :=
by
  sorry

end area_of_inscribed_triangle_l758_758884


namespace minimum_op_coordinates_cos_angle_apb_l758_758214

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

def cos_angle (v1 v2 : ℝ × ℝ) : ℝ :=
  (dot_product v1 v2) / (magnitude v1 * magnitude v2)

def vector_op (t : ℝ) : ℝ × ℝ := (t, 2 * t)
def vector_pa (t : ℝ) : ℝ × ℝ := (1 - t, 5 - 2 * t)
def vector_pb (t : ℝ) : ℝ × ℝ := (7 - t, 1 - 2 * t)

theorem minimum_op_coordinates : vector_op 2 = (2, 4) :=
by
  sorry

theorem cos_angle_apb : cos_angle (vector_pa 2) (vector_pb 2) = -4 * real.sqrt 17 / 17 :=
by
  sorry

end minimum_op_coordinates_cos_angle_apb_l758_758214


namespace five_points_in_square_l758_758986

theorem five_points_in_square (p1 p2 p3 p4 p5 : ℝ × ℝ) :
  (∀ i ∈ {p1, p2, p3, p4, p5}, 0 < i.1 ∧ i.1 < 1 ∧ 0 < i.2 ∧ i.2 < 1) →
  ∃ (x y ∈ {p1, p2, p3, p4, p5}), x ≠ y ∧ dist x y < 1 / Real.sqrt 2 :=
begin
  sorry
end

end five_points_in_square_l758_758986


namespace triangle_angle_sums_l758_758745

theorem triangle_angle_sums (A B C H P Q : Type) [euclidean_geometry A B C H P Q]
  (isosceles : ¬ab = ac)
  (AH_perp_BC : perp AH BC)
  (CP_eq_BC : CP = BC)
  (PQ_intersects_AH : Q = CP ∩ AH)
  (area_ratio : area(BHQ) = (1/4) * area(APQ)) :
  ∠BAC = 30 ∧ ∠ABC = 75 ∧ ∠ACB = 75 :=
by { sorry }

end triangle_angle_sums_l758_758745


namespace line_slope_and_point_l758_758919

theorem line_slope_and_point (x y : ℝ) (h : x / 4 + y / 3 = 1): 
  ∃ m b, y = m * x + b ∧ m = -3 / 4 ∧ (∀ x y, x = 6 → y ≠ -1 → x / 4 + y / 3 ≠ 1) :=
by
  existsi -3 / 4
  existsi 3
  split
  . intro h
    sorry
  split
  . sorry
  . intros x y hx hy
    sorry

end line_slope_and_point_l758_758919


namespace focus_of_parabola_l758_758004

theorem focus_of_parabola (f d : ℝ) :
  (∀ x : ℝ, x^2 + (4*x^2 - f)^2 = (4*x^2 - d)^2) → 8*f + 8*d = 1 → f^2 = d^2 → f = 1/16 :=
by
  intro hEq hCoeff hSq
  sorry

end focus_of_parabola_l758_758004


namespace lyle_percentage_l758_758269

theorem lyle_percentage (chips : ℕ) (ian_ratio lyle_ratio : ℕ) (h_ratio_sum : ian_ratio + lyle_ratio = 10) (h_chips : chips = 100) :
  (lyle_ratio / (ian_ratio + lyle_ratio) : ℚ) * 100 = 60 := 
by
  sorry

end lyle_percentage_l758_758269


namespace find_angle_A_find_area_l758_758191

variables {A B C a b c R : ℝ}
variables {m n : ℝ × ℝ}

-- Condition definitions
def angle_A_condition (A B C : ℝ) (a b c : ℝ) (R : ℝ) : Prop :=
  let m := (Real.sin C, Real.sin B * Real.cos A) in
  let n := (b, 2 * c) in
  let sin_laws := (b = 2 * R * Real.sin B) ∧ (c = 2 * R * Real.sin C) in
  m.1 * n.1 + m.2 * n.2 = 0

def area_condition (A B C a b c : ℝ) : Prop :=
  let cos_law := (Real.cos A = (b ^ 2 + c ^ 2 - a ^ 2) / (2 * b * c)) in
  a = 2 * Real.sqrt 3 ∧ c = 2

-- Proof goal for angle A
theorem find_angle_A (h : angle_A_condition A B C a b c R) : A = 120 :=
sorry

-- Proof goal for area
theorem find_area {A A_proof : ℝ} (hA : A = 120) (h_area : area_condition A B C a b c) : 
  let S := 1 / 2 * b * c * Real.sqrt 3 / 2 in
  S = Real.sqrt 3 :=
sorry

end find_angle_A_find_area_l758_758191


namespace min_weights_for_balance_l758_758349

theorem min_weights_for_balance : (∃ (weights : List ℕ), 
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 100 → 
    ∃ (left right : ℕ), (left + n + right = list.sum weights ∧ 
    list.subset [left, right].bind (λ x, weights) [1, 3, 9, 27, 81]))) 
  ∧ ∀ (weights' : List ℕ), (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 100 → 
    ∃ (left' right' : ℕ), (left' + n + right' = list.sum weights' → list.length weights' ≥ 5)) := sorry

end min_weights_for_balance_l758_758349


namespace problem_statement_l758_758627

noncomputable def a : ℝ := Real.sqrt (25 / 44)
noncomputable def b : ℝ := Real.sqrt ((3 + Real.sqrt 5)^2 / 11)
noncomputable def (a_b_4 : ℝ) := (a + b)^4

theorem problem_statement : 
  a^2 = 25 / 44 ∧ b^2 = (3 + Real.sqrt 5)^2 / 11 ∧ a > 0 ∧ b > 0 ∧
  ∃ (x y z : ℕ), a_b_4 = (x * Real.sqrt y) / z ∧ x + y + z = 14349 :=
by
  sorry

end problem_statement_l758_758627


namespace extreme_value_at_3_increasing_on_interval_l758_758701

def f (a : ℝ) (x : ℝ) : ℝ := 2*x^3 - 3*(a+1)*x^2 + 6*a*x + 8

theorem extreme_value_at_3 (a : ℝ) : (∃ x, x = 3 ∧ 6*x^2 - 6*(a+1)*x + 6*a = 0) → a = 3 :=
by
  sorry

theorem increasing_on_interval (a : ℝ) : (∀ x, x < 0 → 6*(x-a)*(x-1) > 0) → 0 ≤ a :=
by
  sorry

end extreme_value_at_3_increasing_on_interval_l758_758701


namespace quadratic_eq_solutions_l758_758781

-- Define the problem specific variables and conditions
def quadratic_equation (x : ℝ) : Prop := (x - 1) ^ 2 = 4

-- Statement of the theorem based on the problem and the solution answer
theorem quadratic_eq_solutions :
  (quadratic_equation 3) ∧ (quadratic_equation (-1)) :=
by
  split
  exact sorry
  exact sorry

end quadratic_eq_solutions_l758_758781


namespace contradiction_with_angles_l758_758840

-- Definitions of conditions
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180 ∧ α > 0 ∧ β > 0 ∧ γ > 0

-- The proposition we want to prove by contradiction
def at_least_one_angle_not_greater_than_60 (α β γ : ℝ) : Prop := α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60

-- The assumption for contradiction
def all_angles_greater_than_60 (α β γ : ℝ) : Prop := α > 60 ∧ β > 60 ∧ γ > 60

-- The proof problem
theorem contradiction_with_angles (α β γ : ℝ) (h : triangle α β γ) :
  ¬ all_angles_greater_than_60 α β γ → at_least_one_angle_not_greater_than_60 α β γ :=
sorry

end contradiction_with_angles_l758_758840


namespace intersection_of_A_and_B_l758_758506

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} :=
by
  sorry

end intersection_of_A_and_B_l758_758506


namespace find_hyperbola_and_triangle_area_l758_758489

noncomputable def hyperbola_equation (F1 F2 : ℝ × ℝ) (e : ℝ) (p : ℝ × ℝ) : Prop :=
  F1 = (2 * sqrt 2, 0) ∧ F2 = (-2 * sqrt 2, 0) ∧ e = sqrt 2 ∧ p = (4, -2 * sqrt 2) ∧
  ∃ λ : ℝ, (λ = 8 ∧ ∀ x y, (x, y) ∈ λ → x^2 - y^2 = λ)

theorem find_hyperbola_and_triangle_area :
  hyperbola_equation (2 * sqrt 2, 0) (-2 * sqrt 2, 0) (sqrt 2) (4, -2 * sqrt 2) →
  (∀ x y, (x, y) ∈ 8 → x^2 - y^2 = 8) ∧
  (∀ M : ℝ × ℝ, M.2 > 0 → (let MF1 := (M.1 - (2 * sqrt 2), M.2)
      MF2 := (M.1 + (2 * sqrt 2), M.2) in
      (MF1.1 * MF2.1 + MF1.2 * MF2.2 = 0 →
      ∃ a : ℝ, a = 8 ∧ (1 / 2) * |MF1.1 * MF2.1| = 8))) := 
sorry

end find_hyperbola_and_triangle_area_l758_758489


namespace max_integer_a_l758_758974

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x * Real.log x + (1 - a) * x + a

theorem max_integer_a :
  (∀ x : ℝ, 1 < x → f x a > 0) → ∃! a : ℤ, a ≤ 3 :=
by
  intro h
  have hf : ∀ x : ℝ, 1 < x → f x a > 0 := h
  let d := a - Real.exp (a - 2)
  have h_d : d ≤ 0 := by sorry
  exact ExistsUnique.intro 3 h (by sorry) (by sorry)

end max_integer_a_l758_758974


namespace num_digits_divisible_l758_758066

theorem num_digits_divisible (h : ∀ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}, divides n (150 + n)) :
  {n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} | divides n (150 + n)}.card = 5 := 
sorry

end num_digits_divisible_l758_758066


namespace unique_perfect_square_solution_l758_758936

theorem unique_perfect_square_solution :
  ∃ n : ℤ, n = 10 ∧ (∃ k : ℤ, n ^ 4 + 6 * n ^ 3 + 11 * n ^ 2 + 3 * n + 31 = k ^ 2) :=
by
  use 10
  split
  { refl }
  { use 131
    sorry }

end unique_perfect_square_solution_l758_758936


namespace positive_difference_of_b_l758_758636

def g (n : Int) : Int :=
  if n < 0 then n^2 + 3 else 2 * n - 25

theorem positive_difference_of_b :
  let s := g (-3) + g 3
  let t b := g b = -s
  ∃ a b, t a ∧ t b ∧ a ≠ b ∧ |a - b| = 18 :=
by
  sorry

end positive_difference_of_b_l758_758636


namespace no_division_into_three_similar_piles_l758_758661

theorem no_division_into_three_similar_piles :
    ∀ (x : ℝ),
    ∀ (y z : ℝ),
    (x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = x) →
    (x <= sqrt 2 * y ∧ y <= sqrt 2 * z ∧ z <= sqrt 2 * x) →
    false :=
by
  intro x y z
  sorry

end no_division_into_three_similar_piles_l758_758661


namespace integral_x_plus_cos_x_l758_758442

theorem integral_x_plus_cos_x :
  ∫ x in 0..π, (x + cos x) = (π^2) / 2 :=
by
  -- Proof goes here
  sorry

end integral_x_plus_cos_x_l758_758442


namespace tangent_case_intersect_case_outside_case_l758_758395

-- Definitions of the conditions in Lean
variable (r x a b c : ℝ) (A B C M : ℝ) (O P Q R : Point) -- Point is assumed to be a predefined type in Lean
variable (AP BQ CR : ℝ) -- Representing the segments

-- Define conditions
axiom horizontal_line_center_circle (O M : Point) : horizontal_line m O M -- horizontal line m passes through center O of circle
axiom perp_intersect_line (m l : Line) : perp_line l m M -- line l is perpendicular to m and intersects at M
axiom points_on_line_l (A B C : Point) : on_line_l l [A, B, C] -- points A, B, and C are on line l
axiom points_outside_circle (O : Point) (A B C : Point) : outside_circle O [A, B, C] -- points A, B, C are outside circle
axiom points_above_line (m : Line) (A B C : Point) : above_line m [A, B, C] -- points are above line m
axiom order_of_points (A B C M : Point) : furthest_from_M A B C M -- A furthest, B middle, C closest to M
axiom tangents_to_circle (A B C P Q R : Point) : tangent_to_circle O [A, B, C] [P, Q, R] -- Tangents from A, B, C to circle touch at P, Q, R

-- Define expressions
noncomputable def expr1 : ℝ := AB * CR + BC * AP
noncomputable def expr2 : ℝ := AC * BQ

-- Define propositions
theorem tangent_case (h : l.tangent): expr1 = expr2 := sorry
theorem intersect_case (h : l.intersect): expr1 < expr2 := sorry
theorem outside_case (h : l.disjoint): expr1 > expr2 := sorry

end tangent_case_intersect_case_outside_case_l758_758395


namespace distinct_convex_polygons_count_l758_758798

-- Twelve points on a circle
def twelve_points_on_circle := 12

-- Calculate the total number of subsets of twelve points
def total_subsets : ℕ := 2 ^ twelve_points_on_circle

-- Calculate the number of subsets with fewer than three members
def subsets_fewer_than_three : ℕ :=
  (Finset.card (Finset.powersetLen 0 (Finset.range twelve_points_on_circle)) +
   Finset.card (Finset.powersetLen 1 (Finset.range twelve_points_on_circle)) +
   Finset.card (Finset.powersetLen 2 (Finset.range twelve_points_on_circle)))

-- The number of convex polygons that can be formed using three or more points
def distinct_convex_polygons : ℕ := total_subsets - subsets_fewer_than_three

-- Lean theorem statement
theorem distinct_convex_polygons_count :
  distinct_convex_polygons = 4017 := by sorry

end distinct_convex_polygons_count_l758_758798


namespace suraj_new_average_l758_758742

theorem suraj_new_average 
  (A : ℚ) -- Original average
  (h1 : (105 + 110 + 92 + 115 + 100 : ℚ) = 522) -- Total of next 5 innings
  (h2 : (30 * A + 522) / 35 = A + 10) -- Equation for new average
  : 44.4 ∈ Set.Icc (real.ratCast A) (real.ratCast (10 + A)) :=
begin
  sorry
end

end suraj_new_average_l758_758742


namespace column_product_matches_row_product_l758_758080

open Finset

noncomputable theory

variables {α : Type*} [CommRing α]

def column_product_is_constant (a b : Fin n → α) (c : α) : Prop :=
  ∀ j : Fin n, ∏ i in range n, a i + b j = c

def row_product_is_constant (a b : Fin n → α) (c : α) : Prop :=
  ∀ i : Fin n, ∏ j in range n, a i + b j = c

theorem column_product_matches_row_product (a b : Fin n → ℝ) (c : ℝ) 
  (distinct_a : ∀ i j, i ≠ j → a i ≠ a j) 
  (distinct_b : ∀ i j, i ≠ j → b i ≠ b j) 
  (row_prop : row_product_is_constant a b c) : 
  ∃ d : ℝ, column_product_is_constant a b d ∧ d = c :=
begin
  sorry
end

end column_product_matches_row_product_l758_758080


namespace person_speed_approx_l758_758874

noncomputable def convertDistance (meters : ℝ) : ℝ := meters * 0.000621371
noncomputable def convertTime (minutes : ℝ) (seconds : ℝ) : ℝ := (minutes + (seconds / 60)) / 60
noncomputable def calculateSpeed (distance_miles : ℝ) (time_hours : ℝ) : ℝ := distance_miles / time_hours

theorem person_speed_approx (street_length_meters : ℝ) (time_min : ℝ) (time_sec : ℝ) :
  street_length_meters = 900 →
  time_min = 3 →
  time_sec = 20 →
  abs ((calculateSpeed (convertDistance street_length_meters) (convertTime time_min time_sec)) - 10.07) < 0.01 :=
by
  sorry

end person_speed_approx_l758_758874


namespace find_natural_number_l758_758454

theorem find_natural_number (n : ℕ) (h : ∃ k : ℕ, n^2 - 19 * n + 95 = k^2) : n = 5 ∨ n = 14 := by
  sorry

end find_natural_number_l758_758454


namespace max_bottles_drunk_l758_758145

theorem max_bottles_drunk (e b : ℕ) (h1 : e = 16) (h2 : b = 4) : 
  ∃ n : ℕ, n = 5 :=
by
  sorry

end max_bottles_drunk_l758_758145


namespace martian_angle_one_third_circle_l758_758703

theorem martian_angle_one_third_circle :
  let clerts_in_full_circle := 500
  in (clerts_in_full_circle / 3 : ℤ) = 167 :=
by
  let clerts_in_full_circle := 500
  show (clerts_in_full_circle / 3 : ℤ) = 167
  sorry

end martian_angle_one_third_circle_l758_758703
