import Mathlib

namespace number_of_truthful_dwarfs_l625_625601

/-- 
Each of the 10 dwarfs either always tells the truth or always lies. 
Each dwarf likes exactly one type of ice cream: vanilla, chocolate, or fruit.
When asked, every dwarf raised their hand for liking vanilla ice cream.
When asked, 5 dwarfs raised their hand for liking chocolate ice cream.
When asked, only 1 dwarf raised their hand for liking fruit ice cream.
Prove that the number of truthful dwarfs is 4.
-/
theorem number_of_truthful_dwarfs (T L : ℕ) 
  (h1 : T + L = 10) 
  (h2 : T + 2 * L = 16) : 
  T = 4 := 
by
  -- Proof omitted
  sorry

end number_of_truthful_dwarfs_l625_625601


namespace incorrect_conclusion_d_l625_625656

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - π / 6) - 1
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x) - 1

theorem incorrect_conclusion_d : ¬∀ x, f x = g (x - π / 6) :=
by sorry

end incorrect_conclusion_d_l625_625656


namespace remainder_of_product_l625_625060

theorem remainder_of_product (a b n : ℕ) (ha : a % n = 7) (hb : b % n = 1) :
  ((a * b) % n) = 7 :=
by
  -- Definitions as per the conditions
  let a := 63
  let b := 65
  let n := 8
  /- Now prove the statement -/
  sorry

end remainder_of_product_l625_625060


namespace ab_is_18_l625_625144

open Complex

noncomputable def complex_numbers_cond (z a b : ℂ) : Prop :=
  ∀ (z : ℂ),
    (∃ t : ℝ, z = (1 - t) * (2 - I) + t * (-1 + 4 * I)) →
    a * z + b * conj z + 5 = 0

theorem ab_is_18 (a b : ℂ) :
  (∀ (z : ℂ), (∃ (t : ℝ), z = (1 - t) * (2 - I) + t * (-1 + 4 * I)) → 
    a * z + b * conj z + 5 = 0) →
    (a = 3 + 3 * I) →
    (b = 3 - 3 * I) →
    a * b = 18 :=
  by
  intros h1 h2 h3
  rw [h2, h3]
  norm_num

end ab_is_18_l625_625144


namespace triangle_obtuse_l625_625706

theorem triangle_obtuse (A B C : ℝ) (h1 : 0 < tan A * tan B) (h2 : tan A * tan B < 1) : 
  -- Conclusion that triangle ABC is obtuse
  ∃ (p q r : ℝ) (a b c : ℝ), 
  (A + B + C = π) ∧ (A > π / 2 ∨ B > π / 2 ∨ C > π / 2) := 
sorry

end triangle_obtuse_l625_625706


namespace toy_train_probability_l625_625529

noncomputable def tune_probability_same_type : ℚ :=
  let favorable_outcomes := 2 in
  let total_outcomes := 2^3 in
  favorable_outcomes / total_outcomes

theorem toy_train_probability :
  tune_probability_same_type = 1 / 4 :=
by
  -- Proof is omitted
  sorry

end toy_train_probability_l625_625529


namespace employees_count_l625_625527

theorem employees_count (n : ℕ) (h1 : n = 927) (h2 : ∃ e : ℕ, e > 1 ∧ (h1 + 1) % e = 0) :
  n = 29 :=
sorry

end employees_count_l625_625527


namespace pie_distribution_l625_625027

theorem pie_distribution (x y : ℕ) (h1 : x + y + 2 * x = 13) (h2 : x < y) (h3 : y < 2 * x) :
  x = 3 ∧ y = 4 ∧ 2 * x = 6 := by
  sorry

end pie_distribution_l625_625027


namespace prob_both_males_prob_A_not_X_l625_625107

-- Definitions and conditions
def male_students : List Char := ['A', 'B', 'C']
def female_students : List Char := ['X', 'Y', 'Z']
def all_students : List Char := male_students ++ female_students

def subsets_of_size_two {α : Type*} (l : List α) : List (Set α) :=
  l.powerset.filter (λ s, s.card = 2)

def subsets_of_one_male_one_female : List (Char × Char) :=
  male_students.product female_students

-- Statements
theorem prob_both_males :
  (subsets_of_size_two all_students).count (λ s, s ⊆ set.of_list male_students) / (subsets_of_size_two all_students).length = (1 : ℚ) / 5 := sorry

theorem prob_A_not_X :
  subsets_of_one_male_one_female.count (λ p, p.1 = 'A' ∧ p.2 ≠ 'X') / subsets_of_one_male_one_female.length = (2 : ℚ) / 9 := sorry

end prob_both_males_prob_A_not_X_l625_625107


namespace solution_set_of_inequality_l625_625380

theorem solution_set_of_inequality :
  {x : ℝ | (x - 1) / (x - 3) ≤ 0} = {x : ℝ | 1 ≤ x ∧ x < 3} := 
by
  sorry

end solution_set_of_inequality_l625_625380


namespace mark_score_is_46_l625_625259

theorem mark_score_is_46 (highest_score : ℕ) (range: ℕ) (mark_score : ℕ) :
  highest_score = 98 →
  range = 75 →
  (mark_score = 2 * (highest_score - range)) →
  mark_score = 46 := by
  intros
  sorry

end mark_score_is_46_l625_625259


namespace number_division_l625_625050

theorem number_division (n : ℕ) (h1 : n / 25 = 5) (h2 : n % 25 = 2) : n = 127 :=
by
  sorry

end number_division_l625_625050


namespace only_n_eq_2_ensures_zero_table_l625_625514

noncomputable def zero_table_possible (n : ℕ) : Prop :=
  ∀ (table : List (List ℕ)), (∀ row, row ∈ table → ∀ element, 0 < element) →
    (∃ (steps : List (List (List ℕ) → List (List ℕ))),
    let table' := List.foldl (λ (t : List (List ℕ)) (s : List (List ℕ) → List (List ℕ)), s t) table steps
    in ∀ row, row ∈ table' → ∀ element, element = 0)

theorem only_n_eq_2_ensures_zero_table :
  ∀ n : ℕ, (zero_table_possible n ↔ n = 2) :=
begin
  intro n,
  split,
  {
    intro h,
    sorry, -- Proof that zero_table_possible n implies n = 2
  },
  {
    intro h,
    rw ←h,
    sorry, -- Proof that zero_table_possible 2
  }
end

end only_n_eq_2_ensures_zero_table_l625_625514


namespace min_bdf_proof_exists_l625_625062

noncomputable def minBDF (a b c d e f : ℕ) (A : ℕ) :=
  (A = 3 * a ∧ A = 4 * c ∧ A = 5 * e) →
  (a / b * c / d * e / f = A) →
  b * d * f = 60

theorem min_bdf_proof_exists :
  ∃ (a b c d e f A : ℕ), minBDF a b c d e f A :=
by
  sorry

end min_bdf_proof_exists_l625_625062


namespace complex_product_l625_625556

theorem complex_product : (1 + complex.i) * complex.i = -1 + complex.i :=
by
  -- Proof steps were skipped here.
  sorry

end complex_product_l625_625556


namespace two_digit_numbers_with_property_count_l625_625020

/--
Let the two-digit number be $10a + b$ where $a$ and $b$ are the tens and units digits respectively,
with $1 \leq a \leq 9$ and $0 \leq b \leq 9$. Prove that the number of two-digit numbers such that
when the sum of its digits is subtracted from the number, the units digit of the result is 3, 
is 10.
-/
theorem two_digit_numbers_with_property_count {a b : ℕ} (h_a : 1 ≤ a ∧ a ≤ 9) (h_b : 0 ≤ b ∧ b ≤ 9) :
    ∃ count : ℕ, count = 10 ∧ ∀ n, (n = 10 * a + b) → 
                let digit_sum := a + b 
                in (n - digit_sum) % 10 = 3 :=
by
  sorry

end two_digit_numbers_with_property_count_l625_625020


namespace problem_statement_l625_625233

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : (x - 3)^4 + 81 / (x - 3)^4 = 63 :=
by
  sorry

end problem_statement_l625_625233


namespace truthful_dwarfs_count_l625_625589

def dwarf (n : ℕ) := n < 10
def vanilla_ice_cream (n : ℕ) := dwarf n ∧ (∀ m, dwarf m)
def chocolate_ice_cream (n : ℕ) := dwarf n ∧ m % 2 = 0
def fruit_ice_cream (n : ℕ) := dwarf n ∧ m % 9 = 0

theorem truthful_dwarfs_count :
  ∃ T L : ℕ, T + L = 10 ∧ T + 2 * L = 16 ∧ T = 4 :=
by
  sorry

end truthful_dwarfs_count_l625_625589


namespace LCM_20_45_75_is_900_l625_625875

def prime_factorization_20 := (2^2, 5)
def prime_factorization_45 := (3^2, 5)
def prime_factorization_75 := (3, 5^2)

theorem LCM_20_45_75_is_900 
  (pf_20 : prime_factorization_20 = (2^2, 5))
  (pf_45 : prime_factorization_45 = (3^2, 5))
  (pf_75 : prime_factorization_75 = (3, 5^2)) : 
  Nat.lcm (Nat.lcm 20 45) 75 = 900 := 
  by sorry

end LCM_20_45_75_is_900_l625_625875


namespace solution_to_problem_l625_625153

theorem solution_to_problem (x y : ℕ) : 
  (x.gcd y + x.lcm y = x + y) ↔ 
  ∃ (d k : ℕ), (x = d ∧ y = d * k) ∨ (x = d * k ∧ y = d) :=
by sorry

end solution_to_problem_l625_625153


namespace range_of_a_l625_625652

theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x ∈ Icc (-2 : ℝ) 2, a ^ x < 2) ↔ (a ∈ Set.Ioo (1 / Real.sqrt 2) 1 ∪ Set.Ioo 1 (Real.sqrt 2)) :=
by
  sorry

end range_of_a_l625_625652


namespace relatively_prime_sums_l625_625289

theorem relatively_prime_sums (x y : ℤ) (h : Int.gcd x y = 1) 
  : Int.gcd (x^2 + x * y + y^2) (x^2 + 3 * x * y + y^2) = 1 :=
by
  sorry

end relatively_prime_sums_l625_625289


namespace exists_square_all_invisible_l625_625931

open Nat

theorem exists_square_all_invisible (n : ℕ) : 
  ∃ a b : ℤ, ∀ i j : ℕ, i < n → j < n → gcd (a + i) (b + j) > 1 := 
sorry

end exists_square_all_invisible_l625_625931


namespace rectangle_ratio_l625_625010

theorem rectangle_ratio {l w : ℕ} (h_w : w = 5) (h_A : 50 = l * w) : l / w = 2 := by 
  sorry

end rectangle_ratio_l625_625010


namespace roots_of_quadratic_solve_inequality_l625_625221

theorem roots_of_quadratic (a b : ℝ) (h1 : ∀ x : ℝ, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :
  a = 1 ∧ b = 2 :=
by
  sorry

theorem solve_inequality (c : ℝ) :
  let a := 1
  let b := 2
  ∀ x : ℝ, a * x^2 - (a * c + b) * x + b * x < 0 ↔
    (c > 0 → (0 < x ∧ x < c)) ∧
    (c = 0 → false) ∧
    (c < 0 → (c < x ∧ x < 0)) :=
by
  sorry

end roots_of_quadratic_solve_inequality_l625_625221


namespace acute_triangle_side_c_range_l625_625716

theorem acute_triangle_side_c_range {a b c : ℝ} (h1 : a = 2) (h2 : b = 3) :
  sqrt 5 < c ∧ c < sqrt 13 ↔ (3^2 + 2^2 > c^2) ∧ (2^2 + c^2 > 3^2) :=
by sorry

end acute_triangle_side_c_range_l625_625716


namespace quotient_of_squares_mod_13_range_1_to_12_l625_625798

theorem quotient_of_squares_mod_13_range_1_to_12 :
  let m := ∑ n in (Finset.range 12).filter (λ n, n + 1) (λ n, (n + 1) * (n + 1) % 13) in m / 13 = 3 :=
by {
  let m := 1 + 4 + 9 + 3 + 12 + 10,
  calc
    m / 13 = 39 / 13 := by rw [Finset.sum_eq_add, ...] -- This would involve actual Lean code for the finite sum
         ... = 3       := by norm_num,
}

end quotient_of_squares_mod_13_range_1_to_12_l625_625798


namespace relationship_among_m_n_r_l625_625629

theorem relationship_among_m_n_r (a b c m n r : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : 1 < c) 
  (hm : m = log a c) (hn : n = log b c) (hr : r = a^c) : n < m ∧ m < r :=
by
  sorry

end relationship_among_m_n_r_l625_625629


namespace range_of_a_l625_625644

variable (f : ℝ → ℝ)
variable (a : ℝ)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_monotonically_increasing (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem range_of_a (h1 : is_even f)
                   (h2 : is_monotonically_increasing f (Set.Iic 0))
                   (h3 : f (2 ^ (Real.log 3 a)) > f (-real.sqrt 2)) :
                   0 < a ∧ a < Real.sqrt 3 := 
  sorry

end range_of_a_l625_625644


namespace volume_ratio_is_correct_l625_625853

noncomputable def cone_volume_ratio (d h l : ℝ) : ℝ :=
  let m := h
  let m1 := h - l
  let r := d / 2
  let V := (1 / 3) * π * r ^ 2 * m
  let a := (m / (m + m1)) * sqrt (r ^ 2 + m1 ^ 2)
  let b := r * sqrt ((m - m1) / (m + m1))
  let MM_prime := (m - m1) * r / sqrt (r ^ 2 + m1 ^ 2)
  let V1 := (1 / 3) * π * a * b * MM_prime
  V1 / (V - V1)

-- Given conditions as constants
def base_diameter := 26.0  -- in cm
def cone_height := 39.0    -- in cm
def plane_distance := 30.0 -- in cm

-- Final ratio
def volume_ratio := cone_volume_ratio base_diameter cone_height plane_distance

theorem volume_ratio_is_correct : volume_ratio = 0.4941 / 0.5059 :=
by sorry

end volume_ratio_is_correct_l625_625853


namespace handshake_count_l625_625549

noncomputable def unique_handshakes : ℕ := 
  let twin_count := 12 * 2
  let triplet_count := 8 * 3

  let handshakes_among_twins := (twin_count * (twin_count - 2)) / 2
  let handshakes_among_triplets := (triplet_count * (triplet_count - 3)) / 2

  let cross_handshakes := twin_count * (2 * triplet_count) / 3

  handshakes_among_twins + handshakes_among_triplets + cross_handshakes

theorem handshake_count : unique_handshakes = 900 := 
by
  rw [unique_handshakes, nat.cast_mul, nat.cast_sub, nat.cast_bit0, nat.cast_bit1]
  norm_num
  sorry

end handshake_count_l625_625549


namespace ones_digit_sum_is_2_l625_625967

def ones_digit (n : ℕ) : ℕ := n % 10

def eval_sum : ℕ := (∑ n in Finset.range 11, ones_digit (n ^ 2009 - n!))

theorem ones_digit_sum_is_2 :
  ones_digit eval_sum = 2 :=
sorry

end ones_digit_sum_is_2_l625_625967


namespace ratio_of_areas_l625_625085

def parabola : Type := { p : ℝ × ℝ // p.2^2 = 4 * p.1 }

def focus : parabola := ⟨(1, 0), by norm_num⟩

def line (t n : ℝ) : parabola → Prop := λ p, p.1 = t * p.2 + n

def point_M : parabola := ⟨(4, 0), by norm_num⟩

noncomputable def intersecting_points (t n : ℝ) (h : n = 1) : parabola :=
{ p | p.1 = t * p.2 + n ∧ p.2^2 = 4 * (t * p.2 + n) }

noncomputable def extended_intersections (t : ℝ) : parabola :=
{ p | p.2 = -16 / p.2 ∧ p.2^2 = 4 * ((-16 / p.2) * p.2) }

noncomputable def area_CDMM (t : ℝ) : ℝ := 96 * real.sqrt (t^2 + 1)
noncomputable def area_ABMM (t : ℝ) : ℝ := 6 * real.sqrt (t^2 + 1)

theorem ratio_of_areas (t : ℝ) (h : n = 1)
  (A B : parabola)
  (C D : extended_intersections t)
  (M : parabola := ⟨(4, 0), by norm_num⟩) :
  let S_CDMM := area_CDMM t,
      S_ABMM := area_ABMM t in
  (S_CDMM / S_ABMM) = 16 :=
by
  sorry

end ratio_of_areas_l625_625085


namespace distance_against_stream_l625_625262

variable (vs : ℝ) -- speed of the stream

-- condition: in one hour, the boat goes 9 km along the stream
def cond1 (vs : ℝ) := 7 + vs = 9

-- condition: the speed of the boat in still water (7 km/hr)
def speed_still_water := 7

-- theorem to prove: the distance the boat goes against the stream in one hour
theorem distance_against_stream (vs : ℝ) (h : cond1 vs) : 
  (speed_still_water - vs) * 1 = 5 :=
by
  rw [speed_still_water, mul_one]
  sorry

end distance_against_stream_l625_625262


namespace algebraic_sum_of_coefficients_l625_625570

open Nat

theorem algebraic_sum_of_coefficients
  (u : ℕ → ℤ)
  (h1 : u 1 = 5)
  (hrec : ∀ n : ℕ, n > 0 → u (n + 1) - u n = 3 + 4 * (n - 1)) :
  (∃ P : ℕ → ℤ, (∀ n, u n = P n) ∧ (P 1 + P 0 = 5)) :=
sorry

end algebraic_sum_of_coefficients_l625_625570


namespace range_of_m_l625_625667

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x - m)/2 ≥ 2 ∧ x - 4 ≤ 3 * (x - 2)) →
  ∃ x : ℝ, x = 2 ∧ -3 < m ∧ m ≤ -2 :=
by
  sorry

end range_of_m_l625_625667


namespace find_number_l625_625093

def sum_of_digits (n : Nat) : Nat :=
  n.digits.Sum

theorem find_number (n : Nat) (h : n * sum_of_digits n = 2008) : n = 251 := 
by
  sorry

end find_number_l625_625093


namespace trick_deck_cost_l625_625408

theorem trick_deck_cost (x : ℝ) (h1 : 6 * x + 2 * x = 64) : x = 8 :=
  sorry

end trick_deck_cost_l625_625408


namespace total_revenue_generated_l625_625014

theorem total_revenue_generated : 
  let color_stamps_count : ℤ := 578833
  let bw_stamps_count : ℤ := 523776
  let color_stamp_price : ℝ := 0.15
  let bw_stamp_price : ℝ := 0.10
  let color_revenue : ℝ := color_stamps_count * color_stamp_price
  let bw_revenue : ℝ := bw_stamps_count * bw_stamp_price
in color_revenue + bw_revenue = 139202.55 := by
  sorry

end total_revenue_generated_l625_625014


namespace log_abs_is_even_l625_625727

open Real

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

noncomputable def f (x : ℝ) : ℝ := log (abs x)

theorem log_abs_is_even : is_even_function f :=
by
  sorry

end log_abs_is_even_l625_625727


namespace jake_not_drop_coffee_l625_625282

theorem jake_not_drop_coffee :
  let p_trip := 0.40
  let p_drop_trip := 0.25
  let p_step := 0.30
  let p_drop_step := 0.20
  let p_no_drop_trip := 1 - (p_trip * p_drop_trip)
  let p_no_drop_step := 1 - (p_step * p_drop_step)
  (p_no_drop_trip * p_no_drop_step) = 0.846 :=
by
  sorry

end jake_not_drop_coffee_l625_625282


namespace find_a_l625_625247

theorem find_a (a : ℝ) :
  (∀ (x y : ℝ), (a * x + 2 * y + 3 * a = 0) → (3 * x + (a - 1) * y = a - 7)) → 
  a = 3 :=
by
  sorry

end find_a_l625_625247


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625423

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, (n > 0 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∧ ∃ k : ℕ, n = k^2 ∧ n = 225 := 
begin
  sorry
end

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625423


namespace profit_percentage_l625_625056

variable {C S : ℝ}

theorem profit_percentage (h : 19 * C = 16 * S) :
  ((S - C) / C) * 100 = 18.75 := by
  sorry

end profit_percentage_l625_625056


namespace quadrilateral_with_opposite_angles_equal_is_parallelogram_l625_625720

def quadrilateral (V : Type) [AddCommGroup V] [Module ℝ V] :=
{x : fin 4 → V // Function.Injective x}

def opposite_angles_equal (V : Type) [AddCommGroup V] [Module ℝ V] 
    (q : quadrilateral V) : Prop :=
(angle (q 0) (q 1) (q 2) = angle (q 2) (q 3) (q 0)) ∧
(angle (q 1) (q 2) (q 3) = angle (q 3) (q 0) (q 1))

def is_parallelogram (V : Type) [AddCommGroup V] [Module ℝ V] (q : quadrilateral V) : Prop :=
(Vector.parallel (q 1 - q 0) (q 3 - q 2)) ∧
(Vector.parallel (q 2 - q 1) (q 0 - q 3))

theorem quadrilateral_with_opposite_angles_equal_is_parallelogram
    (V : Type) [AddCommGroup V] [Module ℝ V] (q : quadrilateral V) :
    opposite_angles_equal V q → is_parallelogram V q :=
by
  sorry

end quadrilateral_with_opposite_angles_equal_is_parallelogram_l625_625720


namespace main_theorem_l625_625209

def f (x : ℝ) : ℝ := Real.cos x * Real.sin (2 * x)

theorem main_theorem :
  ((∀ x, f(2 * π - x) = -f(x)) ∧
   (∀ x, f(π - x) = f(x)) ∧
   (∀ x, f(-x) = -f(x)) ∧
   (∀ x, f(x + 2 * π) = f(x))) :=
by
  sorry

end main_theorem_l625_625209


namespace range_of_a_l625_625632

-- Definitions of propositions p and q
def f (x a : ℝ) := x^2 - 2*a*x + 1 - 2*a
def g (x a : ℝ) := abs (x - a) - a*x

def p (a : ℝ) : Prop := 
  -- f(x) has two distinct intersections with x-axis in [0, 1]
  let delta := (2*a)^2 - 4 * 1 * (1 - 2*a) in
  delta > 0 ∧ (0 < a ∧ a < 1) ∧ f 0 a ≥ 0 ∧ f 1 a ≥ 0

def q (a : ℝ) : Prop := 
  -- g(x) has a minimum value
  let g1 := λ x, (1 - a) * x - a -- when x >= a
  let g2 := λ x, -(1 + a) * x + a -- when x < a
  (∀ x, x < a → g a x = g2 x) ∧ (∀ x, x ≥ a → g a x = g1 x)

def valid_range_a (a : ℝ) : Prop := 
  a ∈ Set.Ioc 0 (Real.sqrt 2 - 1) ∪ Set.Ioc (1/2) 1

theorem range_of_a (a : ℝ) (h_condition : a > 0) :
  ((¬ p a) ∧ q a) → valid_range_a a := by
  sorry

end range_of_a_l625_625632


namespace print_output_l625_625469

-- Conditions
def a : Nat := 10

/-- The print statement with the given conditions should output "a=10" -/
theorem print_output : "a=" ++ toString a = "a=10" :=
sorry

end print_output_l625_625469


namespace distribution_methods_count_l625_625385

-- Define the problem conditions
def students := {A, B, C, D}
def classA : Finset students := {}
def classB : Finset students := {A}

axiom studentA_classB : A ∈ classB
axiom classA_capacity : ∀ s ∈ classA, s ≠ A → classA.card ≤ 3
axiom classB_capacity : ∀ s ∈ classB, classB.card ≤ 3

-- Define the theorem to prove
theorem distribution_methods_count : 
  (∀ (classA classB : Finset students), A ∈ classB ∧ classA ∩ classB = ∅ 
  ∧ classA.card ≤ 3 ∧ classB.card ≤ 3 
  ∧ (students \ classB ∪ Finset.singleton A) = students 
  → (classA ∪ classB = students))
  → 7 := 
sorry

end distribution_methods_count_l625_625385


namespace scientific_notation_8790000_l625_625965

theorem scientific_notation_8790000 :
  ∀ (n : ℕ), n = 8790000 → 8.79 * 10^6 = 8790000 :=
by
  intro n
  intro hn
  rw hn
  sorry

end scientific_notation_8790000_l625_625965


namespace remainder_of_4000th_term_l625_625957

def sequence_term_position (n : ℕ) : ℕ :=
  n^2

def sum_of_squares_up_to (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

theorem remainder_of_4000th_term : 
  ∃ n : ℕ, sum_of_squares_up_to n ≥ 4000 ∧ (n-1) * n * (2 * (n-1) + 1) / 6 < 4000 ∧ (n % 7) = 1 :=
by 
  sorry

end remainder_of_4000th_term_l625_625957


namespace find_n_l625_625611

def sum_prod (n : ℕ) : ℕ :=
  (finset.range (n - 1)).sum (λ k, k * (k + 1))

def sum_squares (n : ℕ) : ℕ :=
  (finset.range (n - 1)).sum (λ k, k * k)

theorem find_n (n : ℕ) (hn : n ≥ 2) :
  4 * (sum_prod n) + 2 * (sum_squares n) = 55 * n^2 + 61 * n - 116 ↔ n = 29 :=
begin
  sorry
end

end find_n_l625_625611


namespace original_price_of_shoes_l625_625686

noncomputable def original_price (final_price : ℝ) (sales_tax : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  final_price / sales_tax / (discount1 * discount2)

theorem original_price_of_shoes :
  original_price 51 1.07 0.40 0.85 = 140.18 := by
    have h_pre_tax_price : 47.66 = 51 / 1.07 := sorry
    have h_price_relation : 47.66 = 0.85 * 0.40 * 140.18 := sorry
    sorry

end original_price_of_shoes_l625_625686


namespace sum_visible_faces_l625_625032

-- Define the conditions
def face_sum_opposite (die : ℕ) : Prop :=
  ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 6 → (7 - x) ≠ x

def assembled_to_larger_cube (dices : ℕ) : Prop :=
  dices = 27

def dimensions_of_larger_cube (n : ℕ) : Prop :=
  n = 3

-- Define the proof problem statement
theorem sum_visible_faces (dices : ℕ) (n : ℕ) :
  assembled_to_larger_cube(dices) →
  dimensions_of_larger_cube(n) →
  (∀ die, face_sum_opposite(die)) →
  ∃ (s : ℕ), 90 ≤ s ∧ s ≤ 288 :=
by
  -- Begin the proof
  intro h1 h2 h3
  -- Here goes the proof steps, which we skip with sorry
  sorry

end sum_visible_faces_l625_625032


namespace num_arrangements_teachers_students_l625_625546

theorem num_arrangements_teachers_students : 
  (∃ (teachers students : Type) (teacherGroup1 teacherGroup2 : teachers) (studentGroupA studentGroupB studentGroupC studentGroupD : students), 
    (∀ (locationA_teachers locationB_teachers : set teachers)
       (locationA_students locationB_students : set students),
      (locationA_teachers ∪ locationB_teachers = {teacherGroup1, teacherGroup2}) ∧
      (locationA_teachers ∩ locationB_teachers = ∅) ∧
      (locationA_students ∪ locationB_students = {studentGroupA, studentGroupB, studentGroupC, studentGroupD}) ∧
      (locationA_students ∩ locationB_students = ∅) ∧
      locationA_teachers.card = 1 ∧
      locationB_teachers.card = 1 ∧
      locationA_students.card = 2 ∧
      locationB_students.card = 2)
  ) → choose 2 1 * choose 4 2 = 12 := 
by sorry

end num_arrangements_teachers_students_l625_625546


namespace sufficient_but_not_necessary_l625_625901

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 1 → x^2 + 2 * x > 0) ∧ ¬(x^2 + 2 * x > 0 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_l625_625901


namespace prove_range_of_a_l625_625168

noncomputable def condition_p (a : ℝ) : Prop :=
  0 < a ∧ a < 1

noncomputable def condition_q (a : ℝ) : Prop :=
  (2 * a - 3) ^ 2 - 4 > 0

noncomputable def range_of_a : Set ℝ :=
  {a | (1 / 2) ≤ a ∧ a < 1} ∪ {a | (5 / 2) < a}

theorem prove_range_of_a (a : ℝ) :
  (¬ (condition_p a ∧ condition_q a) ∧ (condition_p a ∨ condition_q a)) →
  a ∈ range_of_a :=
begin
  sorry
end

end prove_range_of_a_l625_625168


namespace exists_group_of_four_l625_625710

-- Define the given conditions
variables (students : Finset ℕ) (h_size : students.card = 21)
variables (done_homework : Finset ℕ → Prop)
variables (hw_unique : ∀ (s : Finset ℕ), s.card = 3 → done_homework s)

-- Define the theorem with the assertion to be proved
theorem exists_group_of_four (students : Finset ℕ) (h_size : students.card = 21)
  (done_homework : Finset ℕ → Prop)
  (hw_unique : ∀ s, s.card = 3 → done_homework s) :
  ∃ (grp : Finset ℕ), grp.card = 4 ∧ 
    (∀ (s : Finset ℕ), s ⊆ grp ∧ s.card = 3 → done_homework s) :=
sorry

end exists_group_of_four_l625_625710


namespace abs_neg_implies_nonpositive_l625_625052

theorem abs_neg_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 :=
by
  sorry

end abs_neg_implies_nonpositive_l625_625052


namespace not_and_implies_at_most_one_true_l625_625250

def at_most_one_true (p q : Prop) : Prop := (p → ¬ q) ∧ (q → ¬ p)

theorem not_and_implies_at_most_one_true (p q : Prop) (h : ¬ (p ∧ q)) : at_most_one_true p q :=
begin
  sorry
end

end not_and_implies_at_most_one_true_l625_625250


namespace ordering_of_a_b_c_l625_625231

variable (a b c : ℝ)

-- Conditions 
def condition_a : Prop := a = 4^(0.5)
def condition_b : Prop := b = Real.logBase (Real.pi) 3
def condition_c : Prop := c = Real.logBase (Real.pi) 4

-- The main proof statement
theorem ordering_of_a_b_c (h1 : condition_a a) (h2 : condition_b b) (h3 : condition_c c) : a > c ∧ c > b := 
  sorry  -- Proof omitted

end ordering_of_a_b_c_l625_625231


namespace enhanced_inequality_l625_625752

theorem enhanced_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 * a^2 / (b + c) + 2 * b^2 / (c + a) + 2 * c^2 / (a + b) ≥ a + b + c + (2 * a - b - c)^2 / (a + b + c)) :=
sorry

end enhanced_inequality_l625_625752


namespace system_of_equations_solution_l625_625349

theorem system_of_equations_solution (x y : ℝ) :
  (2 * x + y + 2 * x * y = 11 ∧ 2 * x^2 * y + x * y^2 = 15) ↔
  ((x = 1/2 ∧ y = 5) ∨ (x = 1 ∧ y = 3) ∨ (x = 3/2 ∧ y = 2) ∨ (x = 5/2 ∧ y = 1)) :=
by 
  sorry

end system_of_equations_solution_l625_625349


namespace joan_video_games_total_cost_l625_625284

theorem joan_video_games_total_cost :
  let basketball_game_cost := 5.20
  let racing_game_cost := 4.23
  let puzzle_game_cost := 3.64
  let adventure_game_cost := 7.15
  let puzzle_discount := 0.10
  let adventure_discount := 0.08
  let sales_tax := 0.05
  let discounted_puzzle_cost := puzzle_game_cost * (1 - puzzle_discount)
  let discounted_adventure_cost := adventure_game_cost * (1 - adventure_discount)
  let total_before_tax := basketball_game_cost + racing_game_cost + discounted_puzzle_cost + discounted_adventure_cost
  let total_sales_tax := total_before_tax * sales_tax
  let total_cost := total_before_tax + total_sales_tax
  in total_cost = 20.25 :=
by
  sorry

end joan_video_games_total_cost_l625_625284


namespace tank_destroys_shots_l625_625411

theorem tank_destroys_shots (grid_size : ℕ) (initial_positions : Finset (Fin grid_size × Fin grid_size)) 
    (neighboring_positions : Fin grid_size × Fin grid_size → Finset (Fin grid_size × Fin grid_size))
    (tank_hit_twice : (Fin grid_size × Fin grid_size) → Prop)
    (tank_moves_to_neighbor : ∀ p, p ∈ initial_positions → ∀ q, q ∈ neighboring_positions p → tank_hit_twice q)
    (checkerboard_cover : (Fin grid_size × Fin grid_size) → Prop)
    (total_shots_needed : ℕ)
    (proof_checkerboard_85_shots : grid_size = 13 → total_shots_needed = 169 + 85)
    (proof_checkerboard_pattern_suffice : grid_size = 13 → tank_hit_twice)
    :
    total_shots_needed = 254 := 
by 
  -- proof goes here
  sorry

end tank_destroys_shots_l625_625411


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625450

/-- Problem statement: 
    The smallest positive perfect square that is divisible by 2, 3, and 5 is 900.
-/
theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ n * n % 2 = 0 ∧ n * n % 3 = 0 ∧ n * n % 5 = 0 ∧ n * n = 900 := 
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625450


namespace number_of_chocolates_l625_625838

-- Define the dimensions of the box
def W_box := 30
def L_box := 20
def H_box := 5

-- Define the dimensions of one chocolate
def W_chocolate := 6
def L_chocolate := 4
def H_chocolate := 1

-- Calculate the volume of the box
def V_box := W_box * L_box * H_box

-- Calculate the volume of one chocolate
def V_chocolate := W_chocolate * L_chocolate * H_chocolate

-- Lean theorem statement for the proof problem
theorem number_of_chocolates : V_box / V_chocolate = 125 := 
by
  sorry

end number_of_chocolates_l625_625838


namespace expand_polynomial_l625_625608

open Polynomial

noncomputable def expand_p1 : Polynomial ℚ := Polynomial.monomial 3 4 - Polynomial.monomial 2 3 + Polynomial.monomial 1 2 + Polynomial.C 7

noncomputable def expand_p2 : Polynomial ℚ := Polynomial.monomial 4 5 + Polynomial.monomial 3 1 - Polynomial.monomial 1 3 + Polynomial.C 9

theorem expand_polynomial :
  (expand_p1 * expand_p2) = Polynomial.monomial 7 20 - Polynomial.monomial 5 27 + Polynomial.monomial 4 8 + Polynomial.monomial 3 45 - Polynomial.monomial 2 4 + Polynomial.monomial 1 51 + Polynomial.C 196 :=
by {
  sorry
}

end expand_polynomial_l625_625608


namespace ellipse_equation_standard_max_triangle_area_l625_625194

-- Conditions of the problem
def center_at_origin (C : Ellipse) : Prop := C.center = (0, 0)
def focus_on_x_axis (C : Ellipse) : Prop := C.foci.1.y = 0 ∧ C.foci.2.y = 0
def eccentricity (C : Ellipse) : Prop := C.eccentricity = Real.sqrt 3 / 2
def point_on_ellipse (C : Ellipse) (Q : Point) : Prop := Q = (Real.sqrt 2, Real.sqrt 2 / 2) ∧ C.contains Q

-- Statement for the first proof
theorem ellipse_equation_standard (C : Ellipse) 
  (h_center : center_at_origin C) 
  (h_focus : focus_on_x_axis C) 
  (h_ecc : eccentricity C) 
  (h_point : point_on_ellipse C (Real.sqrt 2, Real.sqrt 2 / 2)) : 
  C = Ellipse.mk (0, 0) 2 1 :=
sorry  -- Proof omitted

-- Additional conditions for the second proof
def line_slope_nonzero (k : ℝ) : Prop := k ≠ 0
def intersects_at_A_B (n : Line) (C : Ellipse) (A B : Point) : Prop := n.slope = k ∧ A ∈ C ∧ B ∈ C
def arithmetic_sequence (k_OA k k_OB : ℝ) : Prop := 2 * k = k_OA + k_OB
def point_M := (1, 1)

-- Statement for the second proof
theorem max_triangle_area (C : Ellipse) (n : Line) (k : ℝ) (A B : Point) 
  (h_center : center_at_origin C)
  (h_focus : focus_on_x_axis C) 
  (h_ecc : eccentricity C) 
  (h_point : point_on_ellipse C (Real.sqrt 2, Real.sqrt 2 / 2)) 
  (h_slope : line_slope_nonzero k) 
  (h_intersect : intersects_at_A_B n C A B) 
  (h_arith : arithmetic_sequence (slope (origin A)) k (slope (origin B))) : 
  ∃ S : ℝ, S = Real.sqrt 5 :=
sorry  -- Proof omitted

end ellipse_equation_standard_max_triangle_area_l625_625194


namespace geom_sequence_S9_l625_625173

variable {S : ℕ → ℝ}
variable {r : ℝ}

def is_geom_sum (S : ℕ → ℝ) (r : ℝ) := ∀ n, S n = a * ((r^n - 1) / (r - 1))

theorem geom_sequence_S9 (S : ℕ → ℝ) (r : ℝ) (h_geom : is_geom_sum S r) :
  S 3 = 7 →
  S 6 = 63 →
  S 9 = 511 :=
by
  intros hS3 hS6
  have h1 : S 3 = 7 := hS3
  have h2 : S 6 = 63 := hS6
  sorry

end geom_sequence_S9_l625_625173


namespace terminating_decimal_count_l625_625622

theorem terminating_decimal_count : 
  let divisor := 2520;
  let range := (1, 500);
  let prime_factorization := (8, 9, 5, 7);
  ∃ (number_of_values : ℕ), number_of_values = 23 :=
by
  let divisor := 2520
  let prime_factorization := (8, 9, 5, 7) -- 2^3, 3^2, 5, 7
  let count_multiples x y := (List.range' 1 y).countp (λ n, n % x = 0);
  let number_of_values := count_multiples 21 500;
  exact ⟨number_of_values, number_of_values⟩
  sorry

end terminating_decimal_count_l625_625622


namespace even_and_period_pi_over_2_l625_625201

noncomputable def f (x : ℝ) : ℝ := (1 + real.cos (2 * x)) * (real.sin x)^2

theorem even_and_period_pi_over_2 : 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, f (x + π / 2) = f x) :=
sorry

end even_and_period_pi_over_2_l625_625201


namespace find_t_l625_625305

noncomputable def distance_squared (x1 x2 y1 y2 : ℝ) := (x2 - x1) ^ 2 + (y2 - y1) ^ 2

noncomputable def midpoint (x1 x2 y1 y2 : ℝ) := ((x1 + x2) / 2, (y1 + y2) / 2)

theorem find_t (t : ℝ) 
  (A_x : ℝ := t - 3) 
  (A_y : ℝ := 0) 
  (B_x : ℝ := -1)
  (B_y : ℝ := t + 2)
  (mid_x mid_y : ℝ := (A_x + B_x) / 2, (A_y + B_y) / 2) 
  (dist_sq_mid_A : ℝ := distance_squared mid_x A_x mid_y A_y)
  (h1 : dist_sq_mid_A = t^2 + 1) :
  t = Real.sqrt 2 ∨ t = -Real.sqrt 2 :=
by {
  sorry
}

end find_t_l625_625305


namespace weight_of_first_cube_is_8_l625_625918

noncomputable def weight_of_first_cube (s : ℝ) (weight_second_cube : ℝ) (h : weight_second_cube = 64) : ℝ :=
  (weight_second_cube / (2 * 2 * 2))

theorem weight_of_first_cube_is_8 :
  ∀ (s : ℝ) (weight_second_cube : ℝ),
  weight_second_cube = 64 →
  let first_cube_weight := weight_of_first_cube s weight_second_cube 64 in
  first_cube_weight = 8 :=
by 
  intros s weight_second_cube h,
  simp [weight_of_first_cube, h],
  sorry

end weight_of_first_cube_is_8_l625_625918


namespace combined_mpg_l625_625788

-- Definitions based on conditions
def ray_mpg : ℕ := 50
def tom_mpg : ℕ := 25
def ray_distance : ℕ := 100
def tom_distance : ℕ := 200

-- Combined rate of miles per gallon proof statement
theorem combined_mpg (ray_mpg tom_mpg : ℕ) (ray_distance tom_distance : ℕ) :
  (ray_mpg = 50) → (tom_mpg = 25) → (ray_distance = 100) → (tom_distance = 200) →
  (ray_distance + tom_distance) / ((ray_distance / ray_mpg) + (tom_distance / tom_mpg)) = 30 := 
by {
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  norm_num,
  sorry
}

end combined_mpg_l625_625788


namespace solution_pair_l625_625143

-- Define the conditions from the problem
def equation1 (x y : ℝ) : Prop := x + 2 * y = (7 - x) + (3 - 2 * y)
def equation2 (x y : ℝ) : Prop := x - 3 * y = (x + 2) - (y - 2)

-- Define the theorem to prove that the solution is the correct pair
theorem solution_pair : ∃ (x y : ℝ), equation1 x y ∧ equation2 x y ∧ x = 9 ∧ y = -2 :=
by
  existsi (9 : ℝ)
  existsi (-2 : ℝ)
  split; [{ apply sorry }]; split; [{ apply sorry }]; split; [{refl}]; {refl}

end solution_pair_l625_625143


namespace series_sum_l625_625567

theorem series_sum :
  (∑' n in (Finset.range' 3 ∞), (n^4 + 2 * n^3 + 10 * n + 15) / (2^n * (n^4 + 9) : ℝ))
  = 0.3567 := 
sorry

end series_sum_l625_625567


namespace geometric_power_inequality_l625_625306

theorem geometric_power_inequality {a : ℝ} {n k : ℕ} (h₀ : 1 < a) (h₁ : 0 < n) (h₂ : n < k) :
  (a^n - 1) / n < (a^k - 1) / k :=
sorry

end geometric_power_inequality_l625_625306


namespace standard_equation_of_circle_l625_625240

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def directrix_equation : ℝ → ℝ := λ x, -1

noncomputable def circle_radius : ℝ := sqrt (3^2 + 2^2)

noncomputable def circle_equation : ℝ × ℝ → ℝ
| (x, y) := (x - 1)^2 + y^2

theorem standard_equation_of_circle :
  circle_equation (1 + circle_radius, 0) = 13 :=
by
  sorry -- Proof to be filled in

end standard_equation_of_circle_l625_625240


namespace regression_model_suitability_and_equation_l625_625507

theorem regression_model_suitability_and_equation :
  let x := [2, 4, 6, 8, 10, 12]
  let y := [0.9, 2.0, 4.2, 3.9, 5.2, 5.1]
  let λ := [0.7, 1.4, 1.8, 2.1, 2.3, 2.5]
  let μ := [1.4, 2.0, 2.4, 2.8, 3.2, 3.5]
  let x̄ := 7
  let ȳ := 3.55
  let λ̄ := 1.80
  let μ̄ := 2.55
  let sum_λ_var := 2.20
  let sum_μ_var := 2.89
  let sum_y_λ_cov := 5.55
  let sum_y_μ_cov := 6.32
  let sqrt_λ_denom := 5.76
  let sqrt_μ_denom := 6.61,
  0 < sum_y_λ_cov / sqrt_λ_denom ∧ sum_y_λ_cov / sqrt_λ_denom > sum_y_μ_cov / sqrt_μ_denom →
  ∃ a b, ∀ x, (y = a * λ - b) := 
by
  sorry

end regression_model_suitability_and_equation_l625_625507


namespace car_with_highest_avg_speed_l625_625559

-- Conditions
def distance_A : ℕ := 715
def time_A : ℕ := 11
def distance_B : ℕ := 820
def time_B : ℕ := 12
def distance_C : ℕ := 950
def time_C : ℕ := 14

-- Average Speeds
def avg_speed_A : ℚ := distance_A / time_A
def avg_speed_B : ℚ := distance_B / time_B
def avg_speed_C : ℚ := distance_C / time_C

-- Theorem
theorem car_with_highest_avg_speed : avg_speed_B > avg_speed_A ∧ avg_speed_B > avg_speed_C :=
by
  sorry

end car_with_highest_avg_speed_l625_625559


namespace probability_abs_diff_gt_0_4_l625_625331

open Set ProbabilityTheory MeasureTheory

def coin_flip_distribution : Measure ℝ := sorry -- Assuming appropriate measure is defined

noncomputable def random_variable_from_coin_flip (flip_1 flip_2 : Bool) : ℝ :=
if flip_1 then (if flip_2 then uniform [0, 0.5] else uniform [0.5, 1]) else (uniform [0, 0.3])

def random_pair (flip_1_x flip_2_x flip_1_y flip_2_y : Bool) : ℝ × ℝ :=
(random_variable_from_coin_flip flip_1_x flip_2_x, random_variable_from_coin_flip flip_1_y flip_2_y)

theorem probability_abs_diff_gt_0_4 :
  ∀ (flip_1_x flip_2_x flip_1_y flip_2_y : Bool),
    P (|random_pair flip_1_x flip_2_x flip_1_y flip_2_y.1 - random_pair flip_1_x flip_2_x flip_1_y flip_2_y.2| > 0.4)
    = 1 / 10 :=
by
  sorry

end probability_abs_diff_gt_0_4_l625_625331


namespace five_fourths_of_x_over_3_l625_625154

theorem five_fourths_of_x_over_3 (x : ℚ) : (5/4) * (x/3) = 5 * x / 12 :=
by
  sorry

end five_fourths_of_x_over_3_l625_625154


namespace sum_of_squares_l625_625022

theorem sum_of_squares (a b c : ℝ)
  (h1 : a + b + c = 19)
  (h2 : a * b + b * c + c * a = 131) :
  a^2 + b^2 + c^2 = 99 :=
by
  sorry

end sum_of_squares_l625_625022


namespace binom_np_n_mod_p2_l625_625783

   theorem binom_np_n_mod_p2 (p n : ℕ) (hp : Nat.Prime p) : (Nat.choose (n * p) n) % (p ^ 2) = n % (p ^ 2) :=
   by
     sorry
   
end binom_np_n_mod_p2_l625_625783


namespace monthly_average_growth_rate_proof_total_profit_proof_l625_625318

-- Define the given conditions as hypotheses
variable (initialSales : ℕ) (finalSales : ℕ) (costPricePerVehicle : ℝ) (sellingPricePerVehicle : ℝ)

-- Condition values
def initialSalesValue := 100
def finalSalesValue := 121
def costPrice := 15.3
def sellingPrice := 16.3

-- Part 1: Prove the monthly average growth rate
theorem monthly_average_growth_rate_proof :
  ∃ (x : ℝ), initialSales = initialSalesValue →
               finalSales = finalSalesValue →
               (100 : ℝ) * (1 + x)^2 = 121 :=
by
  sorry

-- Part 2: Prove the total profit from January to March
theorem total_profit_proof :
  ∃ (profit : ℝ), initialSales = initialSalesValue →
                   finalSales = finalSalesValue →
                   costPricePerVehicle = costPrice →
                   sellingPricePerVehicle = sellingPrice →
                   profit = 331 :=
by
  sorry

end monthly_average_growth_rate_proof_total_profit_proof_l625_625318


namespace range_of_m_l625_625005

-- Define the variables and assumptions
variable {f : ℝ → ℝ} {m : ℝ}

def is_increasing (f : ℝ → ℝ) := ∀ x y, x < y → f(x) < f(y)

-- State the problem
theorem range_of_m (hf : is_increasing f) (h : f (2 * m) > f (-m + 9)) : m > 3 :=
sorry

end range_of_m_l625_625005


namespace minimum_number_of_Geometers_l625_625515

theorem minimum_number_of_Geometers 
       (total_students : ℕ) (total_groups : ℕ) (students_per_group : ℕ) 
       (G A : ℕ) (majority : ℕ) :
  total_students = 121 →
  total_groups = 11 →
  students_per_group = 11 →
  G + A = total_students →
  majority = 6 →
  ∃ (G_min : ℕ), G_min = 36 ∧
  (G_min * majority = 36 * majority) :=
by 
  intro h1 h2 h3 h4 h5
  use 36
  simp [h2, h5]
  sorry

end minimum_number_of_Geometers_l625_625515


namespace problem_statement_l625_625758

noncomputable def x : ℕ → ℝ
| 0       := sqrt 2
| (n + 1) := x n + (1 / x n)

theorem problem_statement :
  (∑ n in finRange 2019, 
    (x n ^ 2) / (2 * x n * x (n + 1) - 1)) 
    > (2019 ^ 2) / (x 2019 ^ 2 + 1 / (x 2019 ^ 2)) := 
begin
  sorry
end

end problem_statement_l625_625758


namespace smallest_perfect_square_divisible_by_2_3_5_l625_625451

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧
         (∀ k : ℕ, 
           (k > 0 ∧ (∃ l : ℕ, k = l * l) ∧ (2 ∣ k) ∧ (3 ∣ k) ∧ (5 ∣ k)) → n ≤ k) :=
sorry

end smallest_perfect_square_divisible_by_2_3_5_l625_625451


namespace sum_sq_div_sub_eq_zero_l625_625761

theorem sum_sq_div_sub_eq_zero (y : Fin 50 → ℝ) 
  (h1 : ∑ i, y i = 2)
  (h2 : ∑ i, (y i) / (2 - y i) = 2) :
  ∑ i, (y i)^2 / (2 - y i) = 0 := 
by
  sorry

end sum_sq_div_sub_eq_zero_l625_625761


namespace xiaojun_dice_game_l625_625475

theorem xiaojun_dice_game :
  let end_square := 8
  ∧ ∀ rolls : List ℕ,
    rolls.length = 3
    ∧ (∀ r, r ∈ rolls → r ∈ {1, 2, 3, 4, 5})
    → (∃ seqs : List (List ℕ),
        seqs.length = 3 
        ∧ (∀ s, s ∈ seqs → (sum s = end_square
                            ∧ ∀ x ∈ s, x ∈ {1, 2, 3, 4, 5})))
    ➔ seqs.sum = 19 := sorry

end xiaojun_dice_game_l625_625475


namespace probability_of_conditions_zero_l625_625304

noncomputable def probability_condition (x : ℝ) (h1 : 100 ≤ x ∧ x < 200) (h2 : ⌊real.sqrt x⌋ = 14) : ℝ :=
  if ⌊real.sqrt (50 * x)⌋ = 140 then 1 else 0

theorem probability_of_conditions_zero : 
  ∀ (x : ℝ), (100 ≤ x ∧ x < 200) → ⌊real.sqrt x⌋ = 14 → probability_condition x ⟨100 ≤ x, x < 200⟩ (⌊real.sqrt x⌋ = 14) = 0 :=
by
  intros
  sorry

end probability_of_conditions_zero_l625_625304


namespace transform_polynomial_l625_625237

open Real

variable {x y : ℝ}

theorem transform_polynomial 
  (h1 : y = x + 1 / x) 
  (h2 : x^4 + x^3 - 4 * x^2 + x + 1 = 0) : 
  x^2 * (y^2 + y - 6) = 0 := 
sorry

end transform_polynomial_l625_625237


namespace number_of_people_quit_l625_625314

-- Define the conditions as constants.
def initial_team_size : ℕ := 25
def new_members : ℕ := 13
def final_team_size : ℕ := 30

-- Define the question as a function.
def people_quit (Q : ℕ) : Prop :=
  initial_team_size - Q + new_members = final_team_size

-- Prove the main statement assuming the conditions.
theorem number_of_people_quit (Q : ℕ) (h : people_quit Q) : Q = 8 :=
by
  sorry -- Proof is not required, so we use sorry to skip it.

end number_of_people_quit_l625_625314


namespace hole_empties_tank_in_sixty_hours_l625_625486

-- Condition: A pipe can fill a tank in 15 hours.
def fill_rate_pipe : ℝ := 1 / 15

-- Condition: With a hole, the tank fills in 20 hours.
def effective_fill_rate_with_hole : ℝ := 1 / 20

-- Definition of the hole's emptying rate in terms of the pipe's fill rate and effective fill rate.
def hole_empties_tank : ℝ := fill_rate_pipe - effective_fill_rate_with_hole

-- The hole's emptying rate is the reciprocal of the time taken to empty the tank.
def time_to_empty_tank : ℝ := 1 / hole_empties_tank

-- Proof statement to verify the time taken for the hole to empty the tank.
theorem hole_empties_tank_in_sixty_hours : time_to_empty_tank = 60 := 
by
  -- This is where the proof would go.
  sorry

end hole_empties_tank_in_sixty_hours_l625_625486


namespace number_of_truthful_dwarfs_is_4_l625_625597

def dwarf := {x : ℕ // 1 ≤ x ≤ 10}
def likes_vanilla (d : dwarf) : Prop := sorry
def likes_chocolate (d : dwarf) : Prop := sorry
def likes_fruit (d : dwarf) : Prop := sorry
def tells_truth (d : dwarf) : Prop := sorry
def tells_lie (d : dwarf) : Prop := sorry

noncomputable def number_of_truthful_dwarfs : ℕ :=
  let total_dwarfs := 10 in
  let vanilla_raises := 10 in
  let chocolate_raises := 5 in
  let fruit_raises := 1 in
  -- T + L = total_dwarfs
  -- T + 2L = vanilla_raises + chocolate_raises + fruit_raises
  let T := total_dwarfs - 2 * (vanilla_raises + chocolate_raises + fruit_raises - total_dwarfs) in
  T

theorem number_of_truthful_dwarfs_is_4 : number_of_truthful_dwarfs = 4 := 
  by
    sorry

end number_of_truthful_dwarfs_is_4_l625_625597


namespace harrys_mothers_age_l625_625223

theorem harrys_mothers_age 
  (h : ℕ)  -- Harry's age
  (f : ℕ)  -- Father's age
  (m : ℕ)  -- Mother's age
  (h_age : h = 50)
  (f_age : f = h + 24)
  (m_age : m = f - h / 25) 
  : (m - h = 22) := 
by
  sorry

end harrys_mothers_age_l625_625223


namespace tom_age_ratio_l625_625030

theorem tom_age_ratio (T N : ℕ) (h1 : sum_ages = T) (h2 : T - N = 3 * (sum_ages_N_years_ago))
  (h3 : sum_ages = T) (h4 : sum_ages_N_years_ago = T - 4 * N) :
  T / N = 11 / 2 := 
by
  sorry

end tom_age_ratio_l625_625030


namespace lcm_of_20_45_75_l625_625864

-- Definitions for the given numbers and their prime factorizations
def num1 : ℕ := 20
def num2 : ℕ := 45
def num3 : ℕ := 75

def factor1 : ℕ → Prop := λ n, n = 2 ^ 2 * 5
def factor2 : ℕ → Prop := λ n, n = 3 ^ 2 * 5
def factor3 : ℕ → Prop := λ n, n = 3 * 5 ^ 2

-- The condition of using the least common multiple function from mathlib
def lcm_def (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- The statement to prove
theorem lcm_of_20_45_75 : lcm_def num1 num2 num3 = 900 := by
  -- Factors condition (Note: These help ensure the numbers' factors are as stated)
  have h1 : factor1 num1 := by { unfold num1 factor1, exact rfl }
  have h2 : factor2 num2 := by { unfold num2 factor2, exact rfl }
  have h3 : factor3 num3 := by { unfold num3 factor3, exact rfl }
  sorry -- This is the place where the proof would go.

end lcm_of_20_45_75_l625_625864


namespace quarters_needed_to_buy_items_l625_625519

-- Define the costs of each item in cents
def cost_candy_bar : ℕ := 25
def cost_chocolate : ℕ := 75
def cost_juice : ℕ := 50

-- Define the quantities of each item
def num_candy_bars : ℕ := 3
def num_chocolates : ℕ := 2
def num_juice_packs : ℕ := 1

-- Define the value of a quarter in cents
def value_of_quarter : ℕ := 25

-- Define the total cost of the items
def total_cost : ℕ := (num_candy_bars * cost_candy_bar) + (num_chocolates * cost_chocolate) + (num_juice_packs * cost_juice)

-- Calculate the number of quarters needed
def num_quarters_needed : ℕ := total_cost / value_of_quarter

-- The theorem to prove that the number of quarters needed is 11
theorem quarters_needed_to_buy_items : num_quarters_needed = 11 := by
  -- Proof omitted
  sorry

end quarters_needed_to_buy_items_l625_625519


namespace smallest_n_watches_l625_625080

variable {n d : ℕ}

theorem smallest_n_watches (h1 : d > 0)
  (h2 : 10 * n - 30 = 100) : n = 13 :=
by
  sorry

end smallest_n_watches_l625_625080


namespace sum_of_theta_values_between_0_and_360_is_540_l625_625888

noncomputable def theta_values_sum :=
  let θ_set := {θ | θ ∈ {60, 120, 150, 210} }
  θ_set.sum id

theorem sum_of_theta_values_between_0_and_360_is_540 
  (h: ∀ θ, θ ∈ ({60, 120, 150, 210} : Set ℝ) → θ ∈ Set.Icc 0 360)
  (isosceles_condition : 
    ∀ θ, θ ∈ Set.Icc 0 360 →
    let A := (Real.cos (π / 6), Real.sin (π / 6))
        B := (Real.cos (π / 2), Real.sin (π / 2))
        C := (Real.cos (θ * π / 180), Real.sin (θ * π / 180))
    IsoscelesTriangle A B (Cos θ, Sin θ)) :
  theta_values_sum = 540 := 
by
  sorry

end sum_of_theta_values_between_0_and_360_is_540_l625_625888


namespace average_of_three_numbers_is_165_l625_625384

variable (x y z : ℕ)
variable (hy : y = 90)
variable (h1 : z = 4 * y)
variable (h2 : y = 2 * x)

theorem average_of_three_numbers_is_165 : (x + y + z) / 3 = 165 := by
  sorry

end average_of_three_numbers_is_165_l625_625384


namespace exists_group_of_four_l625_625709

-- Define the given conditions
variables (students : Finset ℕ) (h_size : students.card = 21)
variables (done_homework : Finset ℕ → Prop)
variables (hw_unique : ∀ (s : Finset ℕ), s.card = 3 → done_homework s)

-- Define the theorem with the assertion to be proved
theorem exists_group_of_four (students : Finset ℕ) (h_size : students.card = 21)
  (done_homework : Finset ℕ → Prop)
  (hw_unique : ∀ s, s.card = 3 → done_homework s) :
  ∃ (grp : Finset ℕ), grp.card = 4 ∧ 
    (∀ (s : Finset ℕ), s ⊆ grp ∧ s.card = 3 → done_homework s) :=
sorry

end exists_group_of_four_l625_625709


namespace solution_set_l625_625830

theorem solution_set (x : ℝ) : (2 : ℝ) ^ (|x-2| + |x-4|) > 2^6 ↔ x < 0 ∨ x > 6 :=
by
  sorry

end solution_set_l625_625830


namespace find_lambda_l625_625674

variables {E : Type*} [AddCommGroup E] [Module ℝ E]

open Module.End

def non_collinear (a b : E) : Prop :=
  ¬(∃ k : ℝ, a = k • b)

theorem find_lambda (a b : E) (h : non_collinear a b) :
  ∃ λ : ℝ, λ = -1/2 ∧ ((a + λ • b) = (-1/2) • (-b + 2 • a)) :=
by
  use -1/2
  split
  . norm_num
  . sorry

end find_lambda_l625_625674


namespace coffee_equals_milk_l625_625025

theorem coffee_equals_milk (S : ℝ) (h : 0 < S ∧ S < 1/2) :
  let initial_milk := 1 / 2
  let initial_coffee := 1 / 2
  let glass1_initial := initial_milk
  let glass2_initial := initial_coffee
  let glass2_after_first_transfer := glass2_initial + S
  let coffee_transferred_back := (S * initial_coffee) / (initial_coffee + S)
  let milk_transferred_back := (S^2) / (initial_coffee + S)
  let glass1_after_second_transfer := glass1_initial - S + milk_transferred_back
  let glass2_after_second_transfer := glass2_initial + S - coffee_transferred_back
  (glass1_initial - S + milk_transferred_back) = (glass2_initial + S - coffee_transferred_back) :=
sorry

end coffee_equals_milk_l625_625025


namespace clara_weight_l625_625832

variables (a c : ℚ)

theorem clara_weight :
  a + c = 240 ∧ c - a = a / 3 → c = 960 / 7 :=
by
  intro h
  cases h with h1 h2
  sorry

end clara_weight_l625_625832


namespace estimate_qualified_helmets_l625_625491

-- Given Conditions
def stable_frequency : ℝ := 0.96

-- Problem Statement
theorem estimate_qualified_helmets (total_helmets : ℕ) (stable_frequency : ℝ) : 
  total_helmets = 10000 → stable_frequency = 0.96 → 
  ∃ (qualified_helmets : ℕ), qualified_helmets = 9600 :=
by 
  intros ht hf 
  use 9600
  rw ht
  rw hf
  linarith
  sorry

end estimate_qualified_helmets_l625_625491


namespace sum_of_interior_angles_l625_625714

theorem sum_of_interior_angles (H : ∀ (n : ℕ), n > 2 → ∃ p : Polygon, 
                p.num_sides = n ∧ ∀ e, p.exterior_angle e = 40) : 
  ∃ (n : ℕ) (p : Polygon), p.num_sides = 9 ∧ (∑ i in p.interior_angles, i = 1260) :=
by 
  -- The proof steps would go here 
  sorry

end sum_of_interior_angles_l625_625714


namespace polynomial_div_degree_l625_625513

noncomputable def degree_of_divisor {R : Type*} [CommRing R] 
  (f d q r : R[X]) : ℕ :=
if h : degree r < degree d then degree f - degree q else 0

theorem polynomial_div_degree {R : Type*} [CommRing R] (f d q r : R[X])
  (hf : degree f = 16)
  (hq : degree q = 10)
  (hr : r = 5 * X ^ 4 + 2 * X ^ 2 - 3 * X + 7)
  (hd : f = d * q + r)
  (hr_deg_lt_hd : degree r < degree d) :
  degree d = 6 :=
by {
  have h1 : degree f = degree d + degree q := degree_mul_leading_coeff zero_ne_one,
  rw [hf, hq] at h1,
  exact h1,
}

end polynomial_div_degree_l625_625513


namespace balance_three_diamonds_l625_625624

-- Define the problem conditions
variables (a b c : ℕ)

-- Four Δ's and two ♦'s will balance twelve ●'s
def condition1 : Prop :=
  4 * a + 2 * b = 12 * c

-- One Δ will balance a ♦ and two ●'s
def condition2 : Prop :=
  a = b + 2 * c

-- Theorem to prove how many ●'s will balance three ♦'s
theorem balance_three_diamonds (h1 : condition1 a b c) (h2 : condition2 a b c) : 3 * b = 2 * c :=
by sorry

end balance_three_diamonds_l625_625624


namespace smallest_of_fours_l625_625537

theorem smallest_of_fours : 
  let n1 := (-5 : ℝ)
  let n2 := (-Real.sqrt 3)
  let n3 := 0
  let n4 := -Real.pi
  n1 < n2 ∧ n1 < n3 ∧ n1 < n4 :=
by {
  let n1 := (-5 : ℝ)
  let n2 := (-Real.sqrt 3)
  let n3 := 0
  let n4 := -Real.pi
  have h1 : n1 < n2 := sorry,
  have h2 : n1 < n3 := sorry,
  have h3 : n1 < n4 := sorry,
  exact ⟨h1, h2, h3⟩
}

end smallest_of_fours_l625_625537


namespace carB_highest_avg_speed_l625_625562

-- Define the distances and times for each car
def distanceA : ℕ := 715
def timeA : ℕ := 11
def distanceB : ℕ := 820
def timeB : ℕ := 12
def distanceC : ℕ := 950
def timeC : ℕ := 14

-- Define the average speeds
def avgSpeedA : ℚ := distanceA / timeA
def avgSpeedB : ℚ := distanceB / timeB
def avgSpeedC : ℚ := distanceC / timeC

theorem carB_highest_avg_speed : avgSpeedB > avgSpeedA ∧ avgSpeedB > avgSpeedC :=
by
  -- Proof will be filled in here
  sorry

end carB_highest_avg_speed_l625_625562


namespace rectangle_area_l625_625369

variable (l w : ℕ)

def length_is_three_times_width := l = 3 * w

def perimeter_is_160 := 2 * l + 2 * w = 160

theorem rectangle_area : 
  length_is_three_times_width l w → 
  perimeter_is_160 l w → 
  l * w = 1200 :=
by
  intros h₁ h₂
  sorry

end rectangle_area_l625_625369


namespace susan_remaining_money_l625_625801

theorem susan_remaining_money :
  let initial_amount := 90
  let food_spent := 20
  let game_spent := 3 * food_spent
  let total_spent := food_spent + game_spent
  initial_amount - total_spent = 10 :=
by 
  sorry

end susan_remaining_money_l625_625801


namespace max_value_char_l625_625679

theorem max_value_char (m x a b : ℕ) (h_sum : 28 * m + x + a + 2 * b = 368)
  (h1 : x ≤ 23) (h2 : x > a) (h3 : a > b) (h4 : b ≥ 0) :
  m + x ≤ 35 := 
sorry

end max_value_char_l625_625679


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625426

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, (n > 0 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∧ ∃ k : ℕ, n = k^2 ∧ n = 225 := 
begin
  sorry
end

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625426


namespace base6_addition_correct_l625_625293

theorem base6_addition_correct (S H E : ℕ) (h1 : S < 6) (h2 : H < 6) (h3 : E < 6) 
  (distinct : S ≠ H ∧ H ≠ E ∧ S ≠ E) 
  (h4: S + H * 6 + E * 6^2 +  H * 6 = H + E * 6 + H * 6^2 + E * 6^1) :
  S + H + E = 12 :=
by sorry

end base6_addition_correct_l625_625293


namespace lcm_of_20_45_75_l625_625865

-- Definitions for the given numbers and their prime factorizations
def num1 : ℕ := 20
def num2 : ℕ := 45
def num3 : ℕ := 75

def factor1 : ℕ → Prop := λ n, n = 2 ^ 2 * 5
def factor2 : ℕ → Prop := λ n, n = 3 ^ 2 * 5
def factor3 : ℕ → Prop := λ n, n = 3 * 5 ^ 2

-- The condition of using the least common multiple function from mathlib
def lcm_def (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- The statement to prove
theorem lcm_of_20_45_75 : lcm_def num1 num2 num3 = 900 := by
  -- Factors condition (Note: These help ensure the numbers' factors are as stated)
  have h1 : factor1 num1 := by { unfold num1 factor1, exact rfl }
  have h2 : factor2 num2 := by { unfold num2 factor2, exact rfl }
  have h3 : factor3 num3 := by { unfold num3 factor3, exact rfl }
  sorry -- This is the place where the proof would go.

end lcm_of_20_45_75_l625_625865


namespace part_I_part_II_l625_625206

def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + x + a) / (2 * Real.exp x)

theorem part_I (a : ℝ) (h : a ≥ 0) (H : ∀ x, f a x ≤ 5 / (2 * Real.exp 1)) : a = 2 := sorry

theorem part_II (a b : ℝ) (h : a ≤ 0) (H : ∀ x ∈ Set.Ici 0, f a x ≤ b * Real.log (x + 1) / 2) : b ≥ 1 := sorry

end part_I_part_II_l625_625206


namespace smallest_positive_perfect_square_div_by_2_3_5_l625_625433

-- Definition capturing the conditions of the problem
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def divisible_by (n m : ℕ) : Prop :=
  m % n = 0

-- The theorem statement combining all conditions and requiring the proof of the correct answer
theorem smallest_positive_perfect_square_div_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ is_perfect_square n ∧ divisible_by 2 n ∧ divisible_by 3 n ∧ divisible_by 5 n ∧ n = 900 :=
sorry

end smallest_positive_perfect_square_div_by_2_3_5_l625_625433


namespace second_round_score_l625_625316

/-- 
  Given the scores in three rounds of darts, where the second round score is twice the
  first round score, and the third round score is 1.5 times the second round score,
  prove that the score in the second round is 48, given that the maximum score in the 
  third round is 72.
-/
theorem second_round_score (x y z : ℝ) (h1 : y = 2 * x) (h2 : z = 1.5 * y) (h3 : z = 72) : y = 48 :=
sorry

end second_round_score_l625_625316


namespace selected_student_in_eighteenth_group_l625_625851

def systematic_sampling (first_number common_difference nth_term : ℕ) : ℕ :=
  first_number + (nth_term - 1) * common_difference

theorem selected_student_in_eighteenth_group :
  systematic_sampling 22 50 18 = 872 :=
by
  sorry

end selected_student_in_eighteenth_group_l625_625851


namespace find_min_abs_diff_l625_625229

theorem find_min_abs_diff (a b : ℕ) (h : a > 0 ∧ b > 0) (h_eq : ab - 4a + 3b = 504) :
  ∃ a b : ℕ, ∃ h₁ : a > 0, ∃ h₂ : b > 0, ∃ h_eq : (a * b - 4 * a + 3 * b = 504), 
  ∀ a' b' : ℕ, (a' * b' - 4 * a' + 3 * b' = 504) → ((b' > 0) ∧ (a' > 0)) →
  |a - b| ≤ |a' - b'| :=
sorry

end find_min_abs_diff_l625_625229


namespace book_transaction_difference_l625_625317

def number_of_books : ℕ := 15
def cost_per_book : ℕ := 11
def selling_price_per_book : ℕ := 25

theorem book_transaction_difference :
  number_of_books * selling_price_per_book - number_of_books * cost_per_book = 210 :=
by
  sorry

end book_transaction_difference_l625_625317


namespace cats_vs_vasyas_l625_625889

variable {α : Type} [Fintype α] (C V : Finset α)

theorem cats_vs_vasyas (C V : Finset α) : 
  (C ∩ V).card = (V ∩ C).card := 
by
  exact Fintype.card_inter_comm C V

end cats_vs_vasyas_l625_625889


namespace meat_market_sold_on_thursday_l625_625773

theorem meat_market_sold_on_thursday : 
  ∃ T : ℕ, let planned := 500, extra := 325, saturday := 130, sunday := 65, total := planned + extra in
    total = 3 * T + saturday + sunday ∧ T = 210 :=
by
  sorry

end meat_market_sold_on_thursday_l625_625773


namespace relationship_between_q_and_r_l625_625701
noncomputable theory

-- Define propositions p, q, and r
variables {m n : Prop}

-- Proposition p: "If m, then n"
def p : Prop := m → n 

-- Inverse proposition q: "If n, then m"
def q : Prop := n → m

-- Contrapositive proposition r: "If not n, then not m"
def r : Prop := ¬n → ¬m

-- Theorem: The relationship between q and r is that they are negations of each other
theorem relationship_between_q_and_r : q ↔ ¬r :=
sorry

end relationship_between_q_and_r_l625_625701


namespace roots_quad_sum_abs_gt_four_sqrt_three_l625_625671

theorem roots_quad_sum_abs_gt_four_sqrt_three
  (p r1 r2 : ℝ)
  (h1 : r1 + r2 = -p)
  (h2 : r1 * r2 = 12)
  (h3 : p^2 > 48) : 
  |r1 + r2| > 4 * Real.sqrt 3 := 
by 
  sorry

end roots_quad_sum_abs_gt_four_sqrt_three_l625_625671


namespace coefficient_x10_expansion_l625_625040

theorem coefficient_x10_expansion : 
  ∀ (x : ℝ), (∑ k in finset.range (12), (nat.choose 11 k) * x^k * (-1)^(11 - k) : ℝ) = (∑ k in finset.range (12), (nat.choose 11 k) * (-1)^(11 - k) * x^k : ℝ) → 
  (∑ k in finset.range (12), if k = 10 then (nat.choose 11 k) * (-1)^(11 - k) else 0) = -11 :=
by
  intros x h
  sorry

end coefficient_x10_expansion_l625_625040


namespace darryl_break_even_l625_625573

noncomputable def break_even_machines (fixed_costs : ℕ) (variable_cost_per_machine : ℕ) (selling_price_per_machine : ℕ) : ℕ :=
let total_fixed_costs := fixed_costs in
let contribution_margin_per_machine := selling_price_per_machine - variable_cost_per_machine in
nat_ceil (total_fixed_costs / contribution_margin_per_machine)

theorem darryl_break_even :
  let parts_cost := 3600 in
  let patent_cost := 4500 in
  let marketing_cost := 2000 in
  let variable_cost := 25 in
  let selling_price := 180 in
  break_even_machines (parts_cost + patent_cost + marketing_cost) variable_cost selling_price = 66 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end darryl_break_even_l625_625573


namespace price_of_brand_X_pen_l625_625149

variable (P : ℝ)

theorem price_of_brand_X_pen :
  (∀ (n : ℕ), n = 12 → 6 * P + 6 * 2.20 = 42 - 13.20) →
  P = 4.80 :=
by
  intro h₁
  have h₂ := h₁ 12 rfl
  sorry

end price_of_brand_X_pen_l625_625149


namespace equilibrium_force_correct_l625_625898

noncomputable def equilibrium_force_range (m : ℝ) (mu : ℝ) (g : ℝ) (alpha : ℝ) : Prop :=
  let F_max :=  (2 * m * g) / (1 - 0.1 * Real.sqrt 3) in
  let F_min :=  (2 * m * g) / (1 + 0.1 * Real.sqrt 3) in
  17.05 ≤ (2 * m * g) / (1 + 0.1 * Real.sqrt 3) ∧ (2 * m * g) / (1 - 0.1 * Real.sqrt 3) ≤ 24.18

theorem equilibrium_force_correct :
  equilibrium_force_range 1 0.1 9.8 (Real.pi / 3) :=
by
  let F_max := (2 * 1 * 9.8) / (1 - 0.1 * Real.sqrt 3)
  let F_min := (2 * 1 * 9.8) / (1 + 0.1 * Real.sqrt 3)
  have h1 : 17.05 ≤ F_min := by sorry
  have h2 : F_max ≤ 24.18 := by sorry
  exact ⟨h1, h2⟩

end equilibrium_force_correct_l625_625898


namespace opposite_of_six_is_neg_six_l625_625690

-- Define the condition that \( a \) is the opposite of \( 6 \)
def is_opposite_of_six (a : Int) : Prop := a = -6

-- Prove that \( a = -6 \) given that \( a \) is the opposite of \( 6 \)
theorem opposite_of_six_is_neg_six (a : Int) (h : is_opposite_of_six a) : a = -6 :=
by
  sorry

end opposite_of_six_is_neg_six_l625_625690


namespace range_of_x_satisfying_inequality_l625_625658

def f (x : ℝ) : ℝ := (x - 1) ^ 4 + 2 * |x - 1|

theorem range_of_x_satisfying_inequality :
  {x : ℝ | f x > f (2 * x)} = {x : ℝ | 0 < x ∧ x < (2 : ℝ) / 3} :=
by
  sorry

end range_of_x_satisfying_inequality_l625_625658


namespace axis_of_symmetry_shifted_cos_symmetry_of_shifted_graph_l625_625337

theorem axis_of_symmetry_shifted_cos:
  ∃ k : ℤ, g (x + π / 6) = cos (x) + ∃ (k : ℤ), x = k * π - π / 6 := sorry

noncomputable def f : ℝ → ℝ := λ x, sin (π / 2 - x)

noncomputable def g : ℝ → ℝ := λ x, cos (x + π / 6)

theorem symmetry_of_shifted_graph :
  ∃ k : ℤ, ∃ x : ℝ, x = ↑k * π - π / 6 ∧ ∀ y : ℝ, g (2 * x - y) = g y :=
sorry

end axis_of_symmetry_shifted_cos_symmetry_of_shifted_graph_l625_625337


namespace infinite_product_equal_root_l625_625131

theorem infinite_product_equal_root : 
  (∀ a b : ℝ, a^b = Real.rpow a b) →
  (3: ℝ)^(1/4) * (9: ℝ)^(1/16) * (27: ℝ)^(1/64) * (81: ℝ)^(1/256) * ... = Real.sqrt[3](9) := 
by
  sorry

end infinite_product_equal_root_l625_625131


namespace number_of_truthful_dwarfs_l625_625586

-- Given conditions
variables (D : Type) [Fintype D] [DecidableEq D] [Card D = 10]
variables (IceCream : Type) [DecidableEq IceCream] (vanilla chocolate fruit : IceCream)
-- Assuming each dwarf likes exactly one type of ice cream
variable (Likes : D → IceCream)
-- Functions indicating if a dwarf raised their hand for each type of ice cream
variables (raisedHandForVanilla raisedHandForChocolate raisedHandForFruit : D → Prop)

-- Given conditions translated to Lean
axiom all_dwarfs_raised_for_vanilla : ∀ d, raisedHandForVanilla d
axiom half_dwarfs_raised_for_chocolate : Fintype.card {d // raisedHandForChocolate d} = 5
axiom one_dwarf_raised_for_fruit : Fintype.card {d // raisedHandForFruit d} = 1

-- Define that a dwarf either always tells the truth or always lies
inductive TruthStatus
| truthful : TruthStatus
| liar : TruthStatus

variable (Status : D → TruthStatus)

-- Definitions related to hand-raising based on dwarf's status and ice cream they like
def raisedHandCorrectly (d : D) : Prop :=
  match Status d with
  | TruthStatus.truthful => 
      raisedHandForVanilla d ↔ Likes d = vanilla ∧
      raisedHandForChocolate d ↔ Likes d = chocolate ∧
      raisedHandForFruit d ↔ Likes d = fruit
  | TruthStatus.liar =>
      raisedHandForVanilla d ↔ Likes d ≠ vanilla ∧
      raisedHandForChocolate d ↔ Likes d ≠ chocolate ∧
      raisedHandForFruit d ↔ Likes d ≠ fruit

-- Goal to prove
theorem number_of_truthful_dwarfs : Fintype.card {d // Status d = TruthStatus.truthful} = 4 :=
by sorry

end number_of_truthful_dwarfs_l625_625586


namespace max_surface_area_correct_l625_625407

-- Define conditions as constants
constant edge_length : ℕ := 1 -- Edge length of each small cube is 1
constant top_view : ℕ := 8    -- Top view consists of 8 squares in 4x2 arrangement
constant num_cols : ℕ := 4    -- 4 columns from the front view
constant num_rows : ℕ := 2    -- Each column is 2 cubes high from front view

-- Definition of the problem
def calculate_max_surface_area (edge_length : ℕ) (top_view : ℕ) (num_cols : ℕ) (num_rows : ℕ) : ℕ :=
  let face_area := top_view * edge_length in
  let top_and_bottom := 2 * face_area in
  let front_and_back := 2 * (num_cols * num_rows) in
  let left_and_right := 2 * (num_rows * num_cols) in
  top_and_bottom + front_and_back + left_and_right

-- Theorem statement to prove surface area is 48
theorem max_surface_area_correct :
  calculate_max_surface_area edge_length top_view num_cols num_rows = 48 :=
by sorry

end max_surface_area_correct_l625_625407


namespace rational_disjoint_union_l625_625984

def s (x y : ℝ) : Set ℕ := {s | ∃ n : ℕ, s = Int.floor (n * x + y)}

theorem rational_disjoint_union (r : ℚ) (hr : r > 1) :
  ∃ u v : ℝ, (u = r.num.to_real / (r.num.to_real - r.denom.to_real)) ∧ 
             (-1 / (r.num.to_real - r.denom.to_real) ≤ v ∧ v < 0) ∧ 
             s r.to_real 0 ∩ s u v = ∅ ∧ 
             s r.to_real 0 ∪ s u v = Set.univ :=
by
  sorry

end rational_disjoint_union_l625_625984


namespace smallest_alpha_l625_625161

theorem smallest_alpha :
  ∃ α β : ℝ, α > 0 ∧ β > 0 ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → sqrt (1 + x) + sqrt (1 - x) ≤ 2 - x^α / β) ∧ 
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → sqrt (1 + x) + sqrt (1 - x) ≤ 2 - x^a / b) → α ≤ a) := 
  ⟨2, 4, 
   by
     intro x hx,
     sorry, -- Note: Proof steps would go here
   intro a b ha hb,
     -- Hence show α = 2 is the smallest"
     sorry -- Note: Proof steps would go here
  ⟩

end smallest_alpha_l625_625161


namespace negation_of_diagonals_equal_l625_625218

-- We define our initial proposition p
def p : Prop := ∀ (rect : Rectangle), rect.diagonals_equal

-- We define the negation of p
def neg_p : Prop := ∀ (rect : Rectangle), ¬ rect.diagonals_equal

-- Now, we state the theorem
theorem negation_of_diagonals_equal : neg_p :=
by sorry

end negation_of_diagonals_equal_l625_625218


namespace inscribed_circle_radius_l625_625100

theorem inscribed_circle_radius (ABC BCD ACD : Type) (r r1 r2 : ℝ) 
  (hABC : is_right_triangle ABC)
  (hBCD_ACD_split_ABC : splits_by_altitude ABC BCD ACD)
  (hBCD_radius : inscribed_radius BCD = 4)
  (hACD_radius : inscribed_radius ACD = 3) : inscribed_radius ABC = 5 := 
sorry

end inscribed_circle_radius_l625_625100


namespace feet_perpendiculars_concyclic_l625_625998

variables {S A B C D O M N P Q : Type} 

-- Given conditions
variables (is_convex_quadrilateral : convex_quadrilateral A B C D)
variables (diagonals_perpendicular : ∀ (AC BD : Line), perpendicular AC BD)
variables (foot_perpendicular : ∀ (O : Point), intersection_point O = foot (perpendicular_from S (base_quadrilateral A B C D)))

-- Define the proof statement
theorem feet_perpendiculars_concyclic
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : diagonals_perpendicular AC BD)
  (h3 : foot_perpendicular O) :
  concyclic (feet_perpendicular_pts O (face S A B)) (feet_perpendicular_pts O (face S B C)) 
            (feet_perpendicular_pts O (face S C D)) (feet_perpendicular_pts O (face S D A)) := sorry

end feet_perpendiculars_concyclic_l625_625998


namespace sum_sq_div_sub_eq_zero_l625_625762

theorem sum_sq_div_sub_eq_zero (y : Fin 50 → ℝ) 
  (h1 : ∑ i, y i = 2)
  (h2 : ∑ i, (y i) / (2 - y i) = 2) :
  ∑ i, (y i)^2 / (2 - y i) = 0 := 
by
  sorry

end sum_sq_div_sub_eq_zero_l625_625762


namespace mu_convergence_to_zero_l625_625739

noncomputable def f (x : ℝ) : ℝ := sorry

def mu (n : ℕ) : ℝ :=
  ∫ x in 0..∞, x^n * f x

lemma f_inequality : ∀ x ≥ 0, |f x| ≤ exp (-sqrt x) :=
sorry

lemma f_derivative : ∀ x > 0, deriv f x = -3 * f x + 6 * f (2 * x) :=
sorry

lemma mu_expression (mu0 : ℝ) : ∀ n, mu n = mu0 * ∏ k in finset.range (n + 1), (1 / (1 - 1 / 2^k)) :=
sorry

theorem mu_convergence_to_zero (mu0 : ℝ) : 
  (∀ n, mu n = mu0 * ∏ k in finset.range (n + 1), (1 / (1 - 1 / 2^k))) →
  tendsto (λ n, (3^n * mu n) / n!) at_top (𝓝 0) ↔ mu0 = 0 :=
sorry

end mu_convergence_to_zero_l625_625739


namespace incorrect_statement_d_l625_625949

-- Definitions based on conditions
def linear_correlation (x y : ℝ) : Prop :=
  ∃ (intercept slope : ℝ), (∀ i, y i = intercept + slope * x i)

def regression_equation (x : ℝ) : ℝ :=
  0.85 * x - 85.71

def regression_line_passes_through_mean (x y : ℝ) (x̄ ȳ : ℝ) : Prop :=
  ȳ = regression_equation x̄

-- Incorrect statement D assertion
def incorrect_statement (x : ℝ) (y : ℝ) : Prop :=
  (x = 170) → (y = 58.79)

theorem incorrect_statement_d {x y : ℝ} (h : linear_correlation x y)
    (h_eq : regression_equation x = y)
    (h_pass : regression_line_passes_through_mean x y (170) (58.79)) :
  ¬ incorrect_statement x y := by
  sorry

end incorrect_statement_d_l625_625949


namespace product_of_base_9_digits_of_9876_l625_625887

def base9_digits (n : ℕ) : List ℕ := 
  let rec digits_aux (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc else digits_aux (n / 9) ((n % 9) :: acc)
  digits_aux n []

def product (lst : List ℕ) : ℕ := lst.foldl (· * ·) 1

theorem product_of_base_9_digits_of_9876 :
  product (base9_digits 9876) = 192 :=
by 
  sorry

end product_of_base_9_digits_of_9876_l625_625887


namespace option_C_is_correct_l625_625892

def fA (x : ℝ) : ℝ := -x^2 + 2
def fB (x : ℝ) : ℝ := -Real.log x
def fC (x : ℝ) : ℝ := 1 / x
def fD (x : ℝ) : ℝ := Real.sin x

theorem option_C_is_correct :
  (∀ x : ℝ, x > 0 → fC (-x) = -fC x) ∧ (∀ (x1 x2 : ℝ), 0 < x1 ∧ x1 < x2 → fC x1 > fC x2) :=
sorry

end option_C_is_correct_l625_625892


namespace strawberry_pie_proof_l625_625566

variable (christine_picked : ℝ)
variable (rachel_picked : ℝ)
variable (strawberries_per_pie : ℝ)
variable (total_strawberries : ℝ)
variable (total_pies : ℝ)

theorem strawberry_pie_proof :
  christine_picked = 10 ∧ rachel_picked = 2 * christine_picked ∧ strawberries_per_pie = 3 → 
  total_strawberries = christine_picked + rachel_picked → 
  total_pies = total_strawberries / strawberries_per_pie → 
  total_pies = 10 :=
by
  intros h1 h2 h3
  cases h1 with hcpick rachpick
  rw rachpick at *
  rw hcpick at *
  rw h2 at *
  rw h3 at *
  sorry

end strawberry_pie_proof_l625_625566


namespace combinations_of_letters_l625_625774

-- Definitions based on the conditions in the problem statement.
def word : List Char := ['B', 'I', 'O', 'L', 'O', 'G', 'Y']
def vowels : List Char := ['I', 'O', 'O']
def consonants : List Char := ['B', 'L', 'G', 'G']

-- The main theorem to prove.
theorem combinations_of_letters : 
  ∃ n : ℕ, n = 12 ∧ (∃ (vowel_combinations consonant_combinations : List (Finset Char)),
  vowel_combinations.length = 3 ∧ consonant_combinations.length = 4 
  ∧
  (vowel_combinations.product consonant_combinations).length = n) :=
sorry

end combinations_of_letters_l625_625774


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625448

/-- Problem statement: 
    The smallest positive perfect square that is divisible by 2, 3, and 5 is 900.
-/
theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ n * n % 2 = 0 ∧ n * n % 3 = 0 ∧ n * n % 5 = 0 ∧ n * n = 900 := 
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625448


namespace four_kids_wash_three_whiteboards_in_20_minutes_l625_625625

-- Condition: It takes one kid 160 minutes to wash six whiteboards
def time_per_whiteboard_for_one_kid : ℚ := 160 / 6

-- Calculation involving four kids
def time_per_whiteboard_for_four_kids : ℚ := time_per_whiteboard_for_one_kid / 4

-- The total time it takes for four kids to wash three whiteboards together
def total_time_for_four_kids_washing_three_whiteboards : ℚ := time_per_whiteboard_for_four_kids * 3

-- Statement to prove
theorem four_kids_wash_three_whiteboards_in_20_minutes : 
  total_time_for_four_kids_washing_three_whiteboards = 20 :=
by
  sorry

end four_kids_wash_three_whiteboards_in_20_minutes_l625_625625


namespace modulus_of_complex_number_l625_625764

theorem modulus_of_complex_number 
  (Z : ℂ) 
  (h : (1 + complex.i) * Z = complex.i) : 
  complex.abs Z = real.sqrt 2 / 2 :=
sorry

end modulus_of_complex_number_l625_625764


namespace fraction_of_area_covered_by_squares_in_triangle_l625_625525

theorem fraction_of_area_covered_by_squares_in_triangle 
  (equilateral_triangle : Type)
  (inscribe_square : equilateral_triangle → Type)
  (total_area : ℝ)
  (infinite_series_area : ℝ) :
  (infinite_series_area / total_area) = (3 - real.sqrt 3)/2 :=
sorry

end fraction_of_area_covered_by_squares_in_triangle_l625_625525


namespace lcm_20_45_75_l625_625884

def lcm (a b : ℕ) : ℕ := nat.lcm a b

theorem lcm_20_45_75 : lcm (lcm 20 45) 75 = 900 :=
by
  sorry

end lcm_20_45_75_l625_625884


namespace expected_value_poisson_l625_625975

open MeasureTheory
open ProbabilityTheory

variables {Ω : Type*} {P : ProbabilityMeasure Ω} {λ : ℝ} (X : Ω → ℕ)

-- Define Poisson distribution pmf
noncomputable def poisson_pmf (k : ℕ) : ℝ := (λ^k * Real.exp(-λ)) / (k.factorial)

-- Define that X follows a Poisson distribution with parameter λ
axiom poisson_distributed : ∀ k : ℕ, P {ω | X ω = k} = poisson_pmf λ k

-- Prove the expected value of X is λ
theorem expected_value_poisson (h : ∀ k : ℕ, P {ω | X ω = k} = poisson_pmf λ k) : 
  (∫ ω, X ω ∂P) = λ :=
sorry

end expected_value_poisson_l625_625975


namespace routes_are_equal_length_l625_625524

-- Define points and segments based on given problem
variables (A B C D E F K L M : Point)
variables (side_ABCD : ℝ) (side_DEF : ℝ) (side_AKLM : ℝ)

-- Conditions
axiom square_ABCD : square A B C D
axiom side_length_ABCD : side_ABCD = 2

axiom square_DEF : square D E F K
axiom side_length_DEF : side_DEF = 1

axiom square_AKLM : square A K L M
axiom side_length_AKLM : side_AKLM = 3

axiom positioned_next_to_each_other :
  positioned_next_to_each_other_on_upper_side A B C D D E F K A K L M

-- Routes to compare
def route_1 : Path := ⟨A, E, F, B⟩
def route_2 : Path := ⟨C, K, D, L⟩

-- Goal: The lengths of the routes AEFB and CKDL are equal
theorem routes_are_equal_length :
  length route_1 = length route_2 :=
sorry

end routes_are_equal_length_l625_625524


namespace brenda_travel_distance_l625_625553

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def start_point : ℝ × ℝ := (-3, 6)
def stop_point : ℝ × ℝ := (1, 1)
def end_point : ℝ × ℝ := (6, -3)

theorem brenda_travel_distance :
  distance start_point stop_point + distance stop_point end_point = 2 * real.sqrt 41 :=
by
  sorry

end brenda_travel_distance_l625_625553


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625422

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, (n > 0 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∧ ∃ k : ℕ, n = k^2 ∧ n = 225 := 
begin
  sorry
end

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625422


namespace angle_between_hands_at_seven_l625_625477

-- Define the conditions
def clock_parts := 12 -- The clock is divided into 12 parts
def degrees_per_part := 30 -- Each part is 30 degrees

-- Define the position of the hour and minute hands at 7:00 AM
def hour_position_at_seven := 7 -- Hour hand points to 7
def minute_position_at_seven := 0 -- Minute hand points to 12

-- Calculate the number of parts between the two positions
def parts_between_hands := if minute_position_at_seven = 0 then hour_position_at_seven else 12 - hour_position_at_seven

-- Calculate the angle between the hour hand and the minute hand at 7:00 AM
def angle_at_seven := degrees_per_part * parts_between_hands

-- State the theorem
theorem angle_between_hands_at_seven : angle_at_seven = 150 :=
by
  sorry

end angle_between_hands_at_seven_l625_625477


namespace water_added_l625_625391

theorem water_added (initial_volume : ℕ) (initial_sugar_percentage : ℝ) (final_sugar_percentage : ℝ) (V : ℝ) : 
  initial_volume = 3 →
  initial_sugar_percentage = 0.4 →
  final_sugar_percentage = 0.3 →
  V = 1 :=
by
  sorry

end water_added_l625_625391


namespace locus_is_circle_l625_625390
noncomputable def CircleLocus (k : ℝ) (A O1 O2 : EuclideanGeometry.Point) (M N P : EuclideanGeometry.Point) 
  [SegmentDivider M N P k] : Prop :=
  ∃ (C : EuclideanGeometry.Circle), ∀ P, P ∈ C
 
theorem locus_is_circle (k : ℝ) (A O1 O2 : EuclideanGeometry.Point) (M N P : EuclideanGeometry.Point) 
  (h1 : M ∈ IntersectingPoints O1 O2 A)
  (h2 : N ∈ IntersectingPoints O1 O2 A)
  (h3 : SegmentDivider M N P k)
  : CircleLocus k A O1 O2 M N P :=
  sorry

end locus_is_circle_l625_625390


namespace number_of_men_l625_625387

theorem number_of_men (M W : ℕ) (h1 : W = 2) (h2 : ∃k, k = 4) : M = 4 :=
by
  sorry

end number_of_men_l625_625387


namespace youngest_child_age_l625_625906

theorem youngest_child_age
  (ten_years_ago_avg_age : Nat) (family_initial_size : Nat) (present_avg_age : Nat)
  (age_difference : Nat) (age_ten_years_ago_total : Nat)
  (age_increase : Nat) (current_age_total : Nat)
  (current_family_size : Nat) (total_age_increment : Nat) :
  ten_years_ago_avg_age = 24 →
  family_initial_size = 4 →
  present_avg_age = 24 →
  age_difference = 2 →
  age_ten_years_ago_total = family_initial_size * ten_years_ago_avg_age →
  age_increase = family_initial_size * 10 →
  current_age_total = age_ten_years_ago_total + age_increase →
  current_family_size = family_initial_size + 2 →
  total_age_increment = current_family_size * present_avg_age →
  total_age_increment - current_age_total = 8 →
  ∃ (Y : Nat), Y + Y + age_difference = 8 ∧ Y = 3 :=
by
  intros
  sorry

end youngest_child_age_l625_625906


namespace inscribed_circle_radius_l625_625279

variables {A B C : Type*} [metric_space A] [metric_space B] [metric_space C] 
variables (M1 M2 : A) (AB BC AC : set (A)) 

def distances (M1 AB BC AC : A) := 
  let d1 := dist M1 AB = 1,
      d2 := dist M1 BC = 3,
      d3 := dist M1 AC = 15
  in (d1, d2, d3)

def other_distances (M2 AB BC AC : A) := 
  let d1 := dist M2 AB = 4,
      d2 := dist M2 BC = 5,
      d3 := dist M2 AC = 11
  in (d1, d2, d3)

theorem inscribed_circle_radius (M1 M2 : A) (AB BC AC : set (A)) 
  (d1 : dist M1 AB = 1) (d2: dist M1 BC = 3) (d3 : dist M1 AC = 15)
  (d4 : dist M2 AB = 4) (d5: dist M2 BC = 5) (d6 : dist M2 AC = 11) :
  ∃ r : ℝ, r = 7 := 
sorry

end inscribed_circle_radius_l625_625279


namespace largest_divisor_of_consecutive_even_product_l625_625857

theorem largest_divisor_of_consecutive_even_product (n : ℕ) (h : n % 2 = 1) :
  ∃ d, (∀ n, n % 2 = 1 → d ∣ (n+2) * (n+4) * (n+6) * (n+8) * (n+10)) ∧ d = 8 :=
begin
  existsi 8,
  split,
  { intros n hn,
    repeat { sorry },
  },
  { refl }
end

end largest_divisor_of_consecutive_even_product_l625_625857


namespace least_common_multiple_of_20_45_75_l625_625868

theorem least_common_multiple_of_20_45_75 :
  Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
sorry

end least_common_multiple_of_20_45_75_l625_625868


namespace sum_of_middle_coefficients_l625_625623

theorem sum_of_middle_coefficients (b : ℝ) (hb : b ≠ 0) : 
  let coeffs := (list.range 8).map (λ k, (nat.binom 7 k * (-2)^k : ℤ)) in
  (coeffs.nth 3).get_or_else 0 + (coeffs.nth 4).get_or_else 0 + (coeffs.nth 5).get_or_else 0 = -56 :=
by 
  sorry

end sum_of_middle_coefficients_l625_625623


namespace area_of_quadrilateral_l625_625489

theorem area_of_quadrilateral
  (A B C D A1 B1 C1 D1 : Type)
  [geometry.quadrilateral ABCD]
  [geometry.point_on_side AB A1]
  [geometry.point_on_side AB B1]
  [geometry.point_on_side CD C1]
  [geometry.point_on_side CD D1]
  (p : Real) (h1 : p < 1/2)
  (hAA1 : dist A A1 = p * dist A B)
  (hBB1 : dist B B1 = p * dist A B)
  (hCC1 : dist C C1 = p * dist C D)
  (hDD1 : dist D D1 = p * dist C D) :
  geometry.area A1 B1 C1 D1 = (1 - 2 * p) * geometry.area ABCD :=
sorry

end area_of_quadrilateral_l625_625489


namespace twenty_fifth_digit_sum_l625_625038

/--
The 25th digit after the decimal point of the sum of the decimal equivalents 
for the fractions 1/9 and 1/11 is 2.
-/
theorem twenty_fifth_digit_sum (n : ℕ) :
  (n : ℝ) = 25 →
  let d1 := 1 / 9 in
  let d2 := 1 / 11 in
  let s := d1 + d2 in
  let dig := nat.pred (n % 6) in
  (((20.2020202020 : ℝ) : ℕ) % 10) = 2 :=
by 
  sorry

end twenty_fifth_digit_sum_l625_625038


namespace not_and_implies_at_most_one_true_l625_625249

def at_most_one_true (p q : Prop) : Prop := (p → ¬ q) ∧ (q → ¬ p)

theorem not_and_implies_at_most_one_true (p q : Prop) (h : ¬ (p ∧ q)) : at_most_one_true p q :=
begin
  sorry
end

end not_and_implies_at_most_one_true_l625_625249


namespace max_binomial_term_l625_625609

theorem max_binomial_term :
  ∃ k, 0 ≤ k ∧ k ≤ 209 ∧ 
       (∀ n, 0 ≤ n ∧ n ≤ 209 → 
              (binom 209 k * (sqrt 5)^k) ≥ (binom 209 n * (sqrt 5)^n)) ∧ k = 145 :=
begin
  sorry
end

end max_binomial_term_l625_625609


namespace evaluate_expression_l625_625151

theorem evaluate_expression : 2 + 1 / (2 + 1 / (2 + 1 / 3)) = 41 / 17 := by
  sorry

end evaluate_expression_l625_625151


namespace tan_pi_seven_root_of_unity_l625_625612
open Complex

theorem tan_pi_seven_root_of_unity : 
  (tan (π / 7) + Complex.i) / (tan (π / 7) - Complex.i) = 
  cos (12 * π / 14) + Complex.i * sin (12 * π / 14) := 
by
  sorry

end tan_pi_seven_root_of_unity_l625_625612


namespace integral_inequalities_l625_625238

noncomputable def S1 : ℝ := ∫ x in 1..2, x^2
noncomputable def S2 : ℝ := ∫ x in 1..2, 1/x
noncomputable def S3 : ℝ := ∫ x in 1..2, exp x

theorem integral_inequalities : S2 < S1 ∧ S1 < S3 := 
by {
  have h1 : S1 = (2^3 / 3) - (1^3 / 3) := sorry,
  have h2 : S2 = log 2 := sorry,
  have h3 : S3 = exp 2 - exp 1 := sorry,
  -- Use the given inqualities obtained from calculations
  have h4 : log 2 < 2^3 / 3 - 1^3 / 3 := sorry,
  have h5 : 2^3 / 3 - 1^3 / 3 < exp 2 - exp 1 := sorry,
  exact ⟨h4, h5⟩
}

end integral_inequalities_l625_625238


namespace range_of_a_in_fx_l625_625253

theorem range_of_a_in_fx (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, (f x = y)) ↔ (1 < a ∧ a ≤ 4) :=
begin
  let f : ℝ → ℝ :=
    λ x, if x > 1 then a^x else (4 - a/2)x + 2,
  split,
  {
    intro h,
    split,
    { sorry, },
    { sorry, },
  },
  {
    intro ha,
    intros x,
    use f x,
    sorry,
  }
end

end range_of_a_in_fx_l625_625253


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625436

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ (n : ℕ), (∃ k : ℕ, n = k^2) ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, (∃ k : ℕ, m = k^2) ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m :=
sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625436


namespace rabbits_ate_27_watermelons_l625_625733

theorem rabbits_ate_27_watermelons
  (original_watermelons : ℕ)
  (watermelons_left : ℕ)
  (watermelons_eaten : ℕ)
  (h1 : original_watermelons = 35)
  (h2 : watermelons_left = 8)
  (h3 : original_watermelons - watermelons_left = watermelons_eaten) :
  watermelons_eaten = 27 :=
by {
  -- Proof skipped
  sorry
}

end rabbits_ate_27_watermelons_l625_625733


namespace non_empty_subsets_count_l625_625224

open Finset

theorem non_empty_subsets_count :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  (∃ k, 1 ≤ k ∧
    ∀ (x : ℕ), x ∈ S → ¬(x + 1 ∈ S)) ∧
    ∀ (S_sub : Finset ℕ), S_sub.card = k →
    ∀ x ∈ S_sub, k ≤ x →
  S.count = 143 := by
  sorry

end non_empty_subsets_count_l625_625224


namespace tony_exercise_hours_per_week_l625_625396

variable (dist_walk dist_run speed_walk speed_run days_per_week : ℕ)

#eval dist_walk : ℕ := 3
#eval dist_run : ℕ := 10
#eval speed_walk : ℕ := 3
#eval speed_run : ℕ := 5
#eval days_per_week : ℕ := 7

theorem tony_exercise_hours_per_week :
  (dist_walk / speed_walk + dist_run / speed_run) * days_per_week = 21 := by
  sorry

end tony_exercise_hours_per_week_l625_625396


namespace largest_divisor_of_five_even_numbers_l625_625855

theorem largest_divisor_of_five_even_numbers (n : ℕ) (h₁ : n % 2 = 1) : 
  ∃ d, (∀ n, n % 2 = 1 → d ∣ (n+2)*(n+4)*(n+6)*(n+8)*(n+10)) ∧ 
       (∀ d', (∀ n, n % 2 = 1 → d' ∣ (n+2)*(n+4)*(n+6)*(n+8)*(n+10)) → d' ≤ d) ∧ 
       d = 480 := sorry

end largest_divisor_of_five_even_numbers_l625_625855


namespace quadratic_expression_value_l625_625361

theorem quadratic_expression_value (x1 x2 : ℝ)
    (h1: x1^2 + 5 * x1 + 1 = 0)
    (h2: x2^2 + 5 * x2 + 1 = 0) :
    ( (x1 * Real.sqrt 6 / (1 + x2))^2 + (x2 * Real.sqrt 6 / (1 + x1))^2 ) = 220 := 
sorry

end quadratic_expression_value_l625_625361


namespace proportion_MN_AD_l625_625274

variable {ABC : Triangle}

variables {D : Point} {BC BD DC : Length}
variables {E K : Point} {AC AE EK : Length}
variables {M N : Point} {BE ME BM BK BN NK AD MN : Length}

-- Conditions:
variable (h1 : BD = 1 * DC)
variable (h2 : (BM / ME) = 7 / 5)
variable (h3 : (BN / NK) = 2 / 3)

-- Statement of the problem:
theorem proportion_MN_AD : (MN / AD) = 11 / 45 :=
by
  sorry

end proportion_MN_AD_l625_625274


namespace travel_distance_of_wheel_l625_625106

theorem travel_distance_of_wheel (r : ℝ) (revolutions : ℕ) (h_r : r = 2) (h_revolutions : revolutions = 2) : 
    ∃ d : ℝ, d = 8 * Real.pi :=
by
  sorry

end travel_distance_of_wheel_l625_625106


namespace num_multiples_231_is_valid_multiples_count_l625_625575

theorem num_multiples_231 (i j : ℕ) (hik : 0 ≤ i) (hjk : i < j ∧ j ≤ 200) :
  ∃ n, 231 * n = 10^j - 10^i ∧ n > 0 :=
sorry

theorem is_valid_multiples_count :
  (∃ f : ℕ → (ℕ × ℕ), 
    (∀ k, 0 ≤ f k.2 ∧ f k.1 < f k.2 ∧ f k.2 ≤ 200) ∧ 
    (∃ n, 231 * n = 10^(f k).2 - 10^(f k).1) ∧ 
    n > 0 ) ↔ 
  ∃ count, count = 3267 :=
sorry

end num_multiples_231_is_valid_multiples_count_l625_625575


namespace distance_wandered_l625_625117

constant speed : ℝ := 2.0
constant time : ℝ := 1.5

theorem distance_wandered : speed * time = 3.0 := by
  sorry

end distance_wandered_l625_625117


namespace factorial_divisor_perfect_square_sum_l625_625568

-- Definitions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def is_divisor (d n : ℕ) : Prop :=
  n % d = 0

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

def relatively_prime (m n : ℕ) : Prop :=
  gcd m n = 1

-- Statement: Given a factorial and specific divisors' conditions, prove the sum of m+n.
theorem factorial_divisor_perfect_square_sum :
  let n := 15
  let fact := factorial n
  let potential_divisors := { d : ℕ // is_divisor d fact ∧ is_divisor 13 d }
  let perfect_square_divisors := { d : potential_divisors // is_perfect_square d.val }
  let p := perfect_square_divisors.to_finset.card.to_nat / potential_divisors.to_finset.card.to_nat
  let m := p.num
  let n := p.den
in relatively_prime m n → m + n = 22 := 
by
  sorry

end factorial_divisor_perfect_square_sum_l625_625568


namespace sum_of_integers_l625_625978

theorem sum_of_integers (n : ℕ) (h1 : 1.2 * n - 4.4 < 5.2) : n < 8 ∧ n > 0 ∧ ∑ i in finset.range 8, i = 28 := 
by
  sorry

end sum_of_integers_l625_625978


namespace simplify_and_evaluate_l625_625792

variable (a : ℝ)
axiom a_cond : a = -1 / 3

theorem simplify_and_evaluate : (3 * a - 1) ^ 2 + 3 * a * (3 * a + 2) = 3 :=
by
  have ha : a = -1 / 3 := a_cond
  sorry

end simplify_and_evaluate_l625_625792


namespace probability_same_number_selected_l625_625334

-- Definitions of multiples and the probability
open Real

def is_multiple_of_15 (n : ℕ) : Prop := n % 15 = 0
def is_multiple_of_20 (n : ℕ) : Prop := n % 20 = 0
def less_than_250 (n : ℕ) : Prop := n < 250

theorem probability_same_number_selected :
  (∀ n_s n_m, is_multiple_of_15 n_s → is_multiple_of_20 n_m → less_than_250 n_s → less_than_250 n_m →
   let possibilities := (finset.range 250).filter is_multiple_of_15.length * (finset.range 250).filter is_multiple_of_20.length,
       same_numbers := (finset.range 250).filter (λ n, is_multiple_of_15 n ∧ is_multiple_of_20 n).length
   in (same_numbers / possibilities : ℚ) = 1 / 48) := sorry

end probability_same_number_selected_l625_625334


namespace eval_f_neg_2_l625_625235

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem eval_f_neg_2 : f (-2) = 19 :=
by
  sorry

end eval_f_neg_2_l625_625235


namespace growth_rate_is_ten_percent_sales_will_not_exceed_forty_thousand_l625_625816

def monthly_growth_rate (sales_june sales_aug : ℕ) : ℕ :=
  let x := (sales_aug / sales_june : ℚ).sqrt - 1
  x

theorem growth_rate_is_ten_percent :
  monthly_growth_rate 30000 36300 = 10 := by
  sorry

def will_sales_exceed (current_sales : ℕ) (growth_rate : ℚ) : Prop :=
  current_sales * (1 + growth_rate) > 40000

theorem sales_will_not_exceed_forty_thousand :
  ¬ will_sales_exceed 36300 0.1 := by
  sorry

end growth_rate_is_ten_percent_sales_will_not_exceed_forty_thousand_l625_625816


namespace largest_quotient_l625_625416

theorem largest_quotient (S : set ℤ) (h : S = {-30, -5, -1, 0, 2, 10, 15}) : 
  ∃ a b ∈ S, b ≠ 0 ∧ a / b = 30 :=
by
  have hS : S = {-30, -5, -1, 0, 2, 10, 15} by exact h
  sorry

end largest_quotient_l625_625416


namespace integral_value_l625_625699

theorem integral_value (a : ℝ) 
  (h : ∃ (a : ℝ) (H : 10.choose 3 - a * 10.choose 2 = 30), a = 2) : 
  ∫ x in 0..2, (3 * x^2 + 1) = 10 :=
by 
  use 2
  rw [integral, a]
  sorry

end integral_value_l625_625699


namespace find_p_X_expectation_l625_625036

variable (probA : ℚ := 2 / 3)  -- Probability of A guessing correctly
variable (probB : ℚ)           -- Probability of B guessing correctly
variable (probStarTeam : ℚ := 1 / 2)  -- Probability of Star Team guessing correctly

theorem find_p :
  probA * (1 - probB) + (1 - probA) * probB = probStarTeam :=
sorry

noncomputable def X_distribution :=
  [((0 : ℕ), 1 / 36), (1, 1 / 6), (2, 13 / 36), (3, 1 / 3), (4, 1 / 9)]

theorem X_expectation :
  X_distribution.sum (λ (p : ℕ × ℚ), p.1 * p.2) = 7 / 3 :=
sorry

end find_p_X_expectation_l625_625036


namespace min_value_expression_l625_625677

noncomputable theory

-- Define points O, A, and B as vectors
variable {V : Type*} [inner_product_space ℝ V]
variables (OA OB : V)
variables (t : ℝ)
variables (a b : ℝ)
variable h1 : OA ⬝ OB = 0     -- Orthogonal vectors
variable h2 : ∥OA∥ = 24        -- Magnitude of vector OA
variable h3 : ∥OB∥ = 24        -- Magnitude of vector OB
variable ht : 0 ≤ t ∧ t ≤ 1    -- t in [0, 1]

-- Define additional vectors
variable AB : V := OB - OA
variable AO : V := -OA
variable BO : V := -OB
variable BA : V := -AB

-- The expression to find the minimum value of
variable E1 := t • AB - AO
variable E2 := (5/12) • BO - (1 - t) • BA

-- Prove the minimum value of the expression 
theorem min_value_expression : ( ∥E1∥ + ∥E2∥ ) = 26 :=
by
  -- Define necessary geometric properties and minimum computation
  sorry

end min_value_expression_l625_625677


namespace rectangle_area_l625_625155

-- Definitions
variables {height length : ℝ} (h : height = length / 2)
variables {area perimeter : ℝ} (a : area = perimeter)

-- Problem statement
theorem rectangle_area : ∃ h : ℝ, ∃ l : ℝ, ∃ area : ℝ, 
  (l = 2 * h) ∧ (area = l * h) ∧ (area = 2 * (l + h)) ∧ (area = 18) :=
sorry

end rectangle_area_l625_625155


namespace fixed_point_passes_l625_625212

theorem fixed_point_passes (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : (∃ x : ℝ, y : ℝ, y = a^(x + 1) - 1 ∧ (x = -1 ∧ y = 0)) :=
by 
  use -1
  use 0
  split
  · rw [pow_add]
    simp
  · exact ⟨rfl, rfl⟩

end fixed_point_passes_l625_625212


namespace constant_term_in_expansion_l625_625809

theorem constant_term_in_expansion :
  let c := (2 * (x : ℝ) ^ 3 + 1 / sqrt x)^7,
       T := ∑ r in Finset.range 8, (binomial 7 r) * 2 ^ (7 - r) * (x ^ 3) ^ (7 - r) * (1 / (sqrt x)) ^ r 
       in (∃ r, 21 - (7 / 2) * r = 0) ∧ T = 14 :=
by
  sorry

end constant_term_in_expansion_l625_625809


namespace triangle_area_l625_625613

variable {R r a b c : ℝ}

-- Given conditions
def circumscribed_radius := c = 2 * R
def inscribed_radius := r = (a + b - c) / 2
def pythagorean_triple := a^2 + b^2 = c^2

theorem triangle_area (h1 : circumscribed_radius) (h2 : inscribed_radius) (h3 : pythagorean_triple) :
  ∃ S, S = r * (2 * R + r) := 
sorry

end triangle_area_l625_625613


namespace invariant_midpoint_treasure_l625_625061

theorem invariant_midpoint_treasure (p1 p2 p3 p4 p5 p6 : Point) 
  (h_on_circle : ∀ (p : Point), p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4 ∨ p = p5 ∨ p = p6) 
  (h_circle : ∀ p : Point, is_on_circumference p) :
  ∃ (midpoint : Point), ∀ (tri1 tri2 : Triangle), 
    tri1.vertices ⊆ {p1, p2, p3, p4, p5, p6} ∧ 
    tri2.vertices ⊆ {p1, p2, p3, p4, p5, p6} ∧ 
    (tri1.vertices ∩ tri2.vertices) = ∅ → 
    (midpoint = find_midpoint (orthocenter tri1) (orthocenter tri2)) := by
  sorry

end invariant_midpoint_treasure_l625_625061


namespace find_x_l625_625498

theorem find_x (x : ℝ) (h : 0.75 * x = (1 / 3) * x + 110) : x = 264 :=
sorry

end find_x_l625_625498


namespace jar_initial_water_fraction_l625_625731

theorem jar_initial_water_fraction (C W : ℝ) (hC : C > 0) (hW : W + C / 4 = 0.75 * C) : W / C = 0.5 :=
by
  -- necessary parameters and sorry for the proof 
  sorry

end jar_initial_water_fraction_l625_625731


namespace smallest_m_for_solutions_l625_625298

noncomputable def fractional_part (x : ℝ) : ℝ := x - x.floor

def g (x : ℝ) : ℝ := abs (3 * fractional_part x - 1.5)

-- Based on the given function definition and periodicity,
-- this statement should encapsulate the core problem's requirements.

theorem smallest_m_for_solutions : ∃ m : ℕ, (m > 0) ∧ (∀ x : ℝ, mg_gx_solutions m >= 1000) ∧ m = 19 :=
sorry

end smallest_m_for_solutions_l625_625298


namespace correct_quotient_division_l625_625228

variable (k : Nat) -- the unknown original number

def mistaken_division := k = 7 * 12 + 4

theorem correct_quotient_division (h : mistaken_division k) : 
  (k / 3) = 29 :=
by
  sorry

end correct_quotient_division_l625_625228


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625441

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ (n : ℕ), (∃ k : ℕ, n = k^2) ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, (∃ k : ℕ, m = k^2) ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m :=
sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625441


namespace post_height_l625_625104

theorem post_height 
  (circumference : ℕ) 
  (rise_per_circuit : ℕ) 
  (travel_distance : ℕ)
  (circuits : ℕ := travel_distance / circumference) 
  (total_rise : ℕ := circuits * rise_per_circuit) 
  (c : circumference = 3)
  (r : rise_per_circuit = 4)
  (t : travel_distance = 9) :
  total_rise = 12 := by
  sorry

end post_height_l625_625104


namespace final_number_is_correct_l625_625090

-- Define the problem conditions as Lean definitions/statements
def original_number : ℤ := 4
def doubled_number (x : ℤ) : ℤ := 2 * x
def resultant_number (x : ℤ) : ℤ := doubled_number x + 9
def final_number (x : ℤ) : ℤ := 3 * resultant_number x

-- Formulate the theorem using the conditions
theorem final_number_is_correct :
  final_number original_number = 51 :=
by
  sorry

end final_number_is_correct_l625_625090


namespace min_groups_l625_625835

theorem min_groups (students : ℕ) (max_per_group : ℕ) (h_students : students = 30) (h_max_per_group : max_per_group = 6) : 
  ∃ groups : ℕ, groups = students / max_per_group ∧ groups = 5 :=
by {
  use 5,
  split,
  { rw [h_students, h_max_per_group],
    norm_num,},
  { refl, }
  }

end min_groups_l625_625835


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625419

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, (n > 0 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∧ ∃ k : ℕ, n = k^2 ∧ n = 225 := 
begin
  sorry
end

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625419


namespace number_at_two_units_right_of_origin_l625_625322

theorem number_at_two_units_right_of_origin : 
  ∀ (n : ℝ), (n = 0) →
  ∀ (x : ℝ), (x = n + 2) →
  x = 2 := 
by
  sorry

end number_at_two_units_right_of_origin_l625_625322


namespace tiffany_bags_difference_l625_625029

theorem tiffany_bags_difference : 
  ∀ (monday_bags next_day_bags : ℕ), monday_bags = 7 → next_day_bags = 12 → next_day_bags - monday_bags = 5 := 
by
  intros monday_bags next_day_bags h1 h2
  sorry

end tiffany_bags_difference_l625_625029


namespace how_many_knights_l625_625321

-- Definitions for the conditions
def Knight : Type := Prop -- A proposition that represents a person being a knight
def Liar : Type := Prop -- A proposition that represents a person being a liar

-- Given conditions identified
variables (person1 person2 person3 person4 person5 person6 : Knight ∪ Liar)
variable  (birthdayClaim : person1 = Knight)
variable  (neighborConfirmation : person1 -> person2 = Knight)

-- The question to prove that the number of knights is 2
theorem how_many_knights
  (person1Knight : person1 = Knight)
  (person2Knight : person2 = Knight → person1 = Knight)
  (person3 : Knight ∪ Liar)
  (person4 : Knight ∪ Liar)
  (person5 : Knight ∪ Liar)
  (person6 : Knight ∪ Liar)
  (allSameAnswer : 
    (∀ p : Knight ∪ Liar, 
     (p = Knight → (neighbor1 p = Knight) ∧ (neighbor2 p = Knight)) ∨ 
     (p = Liar → ¬((neighbor1 p = Knight) ∨ (neighbor2 p = Knight))))
  (birthday : person1 = Knight ∧ neighbor1 person1 = Knight)
  (nextToBirthday : person2 = Knight): 
  (count_knights : nat := 
    [person1, person2, person3, person4, person5, person6].filter (λ x => x = Knight).length) = 2 :=
begin
  sorry
end

end how_many_knights_l625_625321


namespace five_m_plus_twelve_n_leq_581_l625_625287

theorem five_m_plus_twelve_n_leq_581
  (m n: ℕ)
  (a: fin m → ℕ)
  (b: fin n → ℕ)
  (h_even: ∀ i, a i % 2 = 0)
  (h_odd: ∀ j, b j % 2 = 1)
  (h_distinct_a: set.pairwise (set.range a) (≠))
  (h_distinct_b: set.pairwise (set.range b) (≠))
  (h_sum: (∑ i, a i) + (∑ j, b j) = 2019) :
  5 * m + 12 * n ≤ 581 :=
by {
  sorry
}

end five_m_plus_twelve_n_leq_581_l625_625287


namespace degree_to_radian_l625_625571

theorem degree_to_radian (h : 180 * real.pi.symm = 1) : 300 * real.pi.symm = 5 * π / 3 :=
sorry

end degree_to_radian_l625_625571


namespace coefficient_of_x4_in_expansion_of_2x_plus_sqrtx_l625_625576

noncomputable def coefficient_of_x4_expansion : ℕ :=
  let r := 2;
  let n := 5;
  let general_term_coefficient := Nat.choose n r * 2^(n-r);
  general_term_coefficient

theorem coefficient_of_x4_in_expansion_of_2x_plus_sqrtx :
  coefficient_of_x4_expansion = 80 :=
by
  -- We can bypass the actual proving steps by
  -- acknowledging that the necessary proof mechanism
  -- will properly verify the calculation:
  sorry

end coefficient_of_x4_in_expansion_of_2x_plus_sqrtx_l625_625576


namespace value_of_q_at_2_l625_625123

def q (x : ℝ) : ℝ := -- placeholder for the actual function, which we won't define here 

axiom q_point_on_graph : \(q(2) = 3\)

theorem value_of_q_at_2 : q(2.0) = 3 :=
by
  exact q_point_on_graph

end value_of_q_at_2_l625_625123


namespace distance_point_to_line_correct_l625_625972

noncomputable def distance_point_to_line (x0 y0 z0 m n p x1 y1 z1 : ℝ) : ℝ :=
  let l := (m, n, p)
  let M0M1 := (x1 - x0, y1 - y0, z1 - z0)
  let cross_prod := 
    ((y1 - y0) * p - (z1 - z0) * n,
     (z1 - z0) * m - (x1 - x0) * p,
     (x1 - x0) * n - (y1 - y0) * m)
  let mag_cross_prod := 
    real.sqrt (cross_prod.1^2 + cross_prod.2^2 + cross_prod.3^2)
  let mag_l :=
    real.sqrt (m^2 + n^2 + p^2)
  mag_cross_prod / mag_l

theorem distance_point_to_line_correct (x0 y0 z0 m n p x1 y1 z1 : ℝ) :
  distance_point_to_line x0 y0 z0 m n p x1 y1 z1 = 
    real.sqrt ((y1 - y0) * p - (z1 - z0) * n)^2 + 
              ((x1 - x0) * p - (z1 - z0) * m)^2 + 
              ((x1 - x0) * n - (y1 - y0) * m)^2 / 
    real.sqrt (m^2 + n^2 + p^2) :=
sorry

end distance_point_to_line_correct_l625_625972


namespace triangular_number_difference_l625_625948

-- Definition of the nth triangular number
def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Theorem stating the problem
theorem triangular_number_difference :
  triangular_number 2010 - triangular_number 2008 = 4019 :=
by
  sorry

end triangular_number_difference_l625_625948


namespace range_f_l625_625579

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x + 1) / (x - 3)

-- State the theorem about the range of the function f
theorem range_f : 
  ∀ y : ℝ, (∃ x : ℝ, x ≠ 3 ∧ y = f(x)) ↔ y ∈ ((-∞, 2) ∪ (2, +∞)) :=
by
  sorry

end range_f_l625_625579


namespace lcm_20_45_75_l625_625881

def lcm (a b : ℕ) : ℕ := nat.lcm a b

theorem lcm_20_45_75 : lcm (lcm 20 45) 75 = 900 :=
by
  sorry

end lcm_20_45_75_l625_625881


namespace radius_of_third_circle_l625_625847

theorem radius_of_third_circle (r₁ r₂ : ℝ) (r₁_val : r₁ = 23) (r₂_val : r₂ = 37) : 
  ∃ r : ℝ, r = 2 * Real.sqrt 210 :=
by
  sorry

end radius_of_third_circle_l625_625847


namespace complex_roots_circle_radius_l625_625111

noncomputable def radius_of_circle : ℝ := 2 / 3

theorem complex_roots_circle_radius (z : ℂ) (h : (z + 1) ^ 4 = 16 * z ^ 4) : 
  ∃ x y : ℝ, z = x + y * complex.I ∧ (x - 1/3)^2 + y^2 = (2/3)^2 :=
sorry

end complex_roots_circle_radius_l625_625111


namespace stability_indicator_is_std_dev_l625_625844

variable {α : Type*}
variable [RealField α]

-- Variables and conditions
variable (n : ℕ) (x : Fin n → α)

-- Statement to prove: the standard deviation is the best indicator of stability
theorem stability_indicator_is_std_dev (h : n > 0) :
  (stability_indicator x = stddev x) :=
sorry

end stability_indicator_is_std_dev_l625_625844


namespace no_negative_exponents_l625_625569

theorem no_negative_exponents (a b c d : ℤ) : 
  4^a + 4^b = 8^c + 27^d → a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 :=
by
  sorry

end no_negative_exponents_l625_625569


namespace joan_needs_more_flour_l625_625283

-- Definitions for the conditions
def total_flour : ℕ := 7
def flour_added : ℕ := 3

-- The theorem stating the proof problem
theorem joan_needs_more_flour : total_flour - flour_added = 4 :=
by
  sorry

end joan_needs_more_flour_l625_625283


namespace find_value_of_t_l625_625582

variable (a b v d t r : ℕ)

-- All variables are non-zero digits (1-9)
axiom non_zero_a : 0 < a ∧ a < 10
axiom non_zero_b : 0 < b ∧ b < 10
axiom non_zero_v : 0 < v ∧ v < 10
axiom non_zero_d : 0 < d ∧ d < 10
axiom non_zero_t : 0 < t ∧ t < 10
axiom non_zero_r : 0 < r ∧ r < 10

-- Given conditions
axiom condition1 : a + b = v
axiom condition2 : v + d = t
axiom condition3 : t + a = r
axiom condition4 : b + d + r = 18

theorem find_value_of_t : t = 9 :=
by sorry

end find_value_of_t_l625_625582


namespace mixture_ratio_3_5_l625_625479

-- Assumptions
def initial_alcohol : ℚ := 4
def initial_water : ℚ := 4

-- Hypothesis: Adding a specific amount of water to achieve a 3:5 ratio of alcohol to water.
def water_to_add : ℚ := 3

theorem mixture_ratio_3_5 :
  initial_alcohol / (initial_water + water_to_add) = 3 / 5 :=
by 
  unfold initial_alcohol initial_water water_to_add
  calc
    4 / (4 + 3) = 4 / 7 : rfl
    ... = 3 / 5 : sorry  -- Proof needed here, but not required for the statement

end mixture_ratio_3_5_l625_625479


namespace min_value_k_triangle_l625_625165

theorem min_value_k_triangle (s : Finset ℕ) (h₁ : ∀ x ∈ s, x ∈ Finset.range 1001) (h₂ : s.card = 16) :
  ∃ a b c ∈ s, a + b > c ∧ a + c > b ∧ b + c > a :=
sorry

end min_value_k_triangle_l625_625165


namespace truthful_dwarfs_count_l625_625588

def dwarf (n : ℕ) := n < 10
def vanilla_ice_cream (n : ℕ) := dwarf n ∧ (∀ m, dwarf m)
def chocolate_ice_cream (n : ℕ) := dwarf n ∧ m % 2 = 0
def fruit_ice_cream (n : ℕ) := dwarf n ∧ m % 9 = 0

theorem truthful_dwarfs_count :
  ∃ T L : ℕ, T + L = 10 ∧ T + 2 * L = 16 ∧ T = 4 :=
by
  sorry

end truthful_dwarfs_count_l625_625588


namespace sum_of_first_10_even_numbers_l625_625842

theorem sum_of_first_10_even_numbers : (∑ i in Finset.range 10, 2 * (i + 1)) = 110 := 
  sorry

end sum_of_first_10_even_numbers_l625_625842


namespace prime_quadratic_root_range_l625_625755

theorem prime_quadratic_root_range (p : ℕ) (hprime : Prime p) 
  (hroots : ∃ x1 x2 : ℤ, x1 * x2 = -580 * p ∧ x1 + x2 = p) : 20 < p ∧ p < 30 :=
by
  sorry

end prime_quadratic_root_range_l625_625755


namespace induction_limits_l625_625558

theorem induction_limits :
  ¬ (∀ (P : ℕ → Prop), (∀ n, P n) → (P 0 ∧ (∀ n, P n → P (n + 1)))) ∧
  ¬ (∀ (P : ℕ → Prop), ¬ (∀ n, ¬ P n) → (¬ (P 0) ∧ (∀ n, ¬ P n → ¬ P (n + 1)))) :=
begin
  sorry
end

end induction_limits_l625_625558


namespace find_number_l625_625066

theorem find_number (a b some_number : ℕ) (h1 : a = 69842) (h2 : b = 30158) (h3 : (a^2 - b^2) / some_number = 100000) : some_number = 39684 :=
by {
  -- Proof skipped
  sorry
}

end find_number_l625_625066


namespace modified_cube_edges_l625_625523

/--
A solid cube with a side length of 4 has different-sized solid cubes removed from three of its corners:
- one corner loses a cube of side length 1,
- another corner loses a cube of side length 2,
- and a third corner loses a cube of side length 1.

The total number of edges of the modified solid is 22.
-/
theorem modified_cube_edges :
  let original_edges := 12
  let edges_removed_1x1 := 6
  let edges_added_2x2 := 16
  original_edges - 2 * edges_removed_1x1 + edges_added_2x2 = 22 := by
  sorry

end modified_cube_edges_l625_625523


namespace part1_part2_part3_l625_625996

noncomputable def seq (a : ℝ) (n : ℕ) : ℝ := if n = 0 then 0 else (1 - a) / n

theorem part1 (a : ℝ) (h_pos : ∀ n : ℕ, n > 0 → seq a n > 0)
  (a1_eq : seq a 1 = 1 / 2) (a2_eq : seq a 2 = 1 / 4) : true :=
by trivial

theorem part2 (a : ℝ) (h_pos : ∀ n : ℕ, n > 0 → seq a n > 0)
  (n : ℕ) (hn : n > 0) : 0 < seq a n ∧ seq a n < 1 :=
sorry

theorem part3 (a : ℝ) (h_pos : ∀ n : ℕ, n > 0 → seq a n > 0)
  (n : ℕ) (hn : n > 0) : seq a n > seq a (n + 1) :=
sorry

end part1_part2_part3_l625_625996


namespace greatest_integer_value_of_x_l625_625976

theorem greatest_integer_value_of_x :
  ∃ x : ℤ, (3 * |2 * x + 1| + 10 > 28) ∧ (∀ y : ℤ, 3 * |2 * y + 1| + 10 > 28 → y ≤ x) :=
sorry

end greatest_integer_value_of_x_l625_625976


namespace function_composition_l625_625138

def f (x : ℝ) : ℝ := x + 3
def f_inv (x : ℝ) : ℝ := x - 3
def g (x : ℝ) : ℝ := x / 4
def g_inv (x : ℝ) : ℝ := 4 * x

theorem function_composition : 
  f (g_inv (f_inv (f_inv (g (f 23))))) = 5 := 
by
  sorry

end function_composition_l625_625138


namespace find_a_if_f_is_even_l625_625245

-- Defining f as given in the problem conditions
noncomputable def f (x a : ℝ) : ℝ := (x + a) * 3 ^ (x - 2 + a ^ 2) - (x - a) * 3 ^ (8 - x - 3 * a)

-- Statement of the proof problem with the conditions
theorem find_a_if_f_is_even (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → (a = -5 ∨ a = 2) :=
by
  sorry

end find_a_if_f_is_even_l625_625245


namespace number_multiplied_by_sum_of_digits_eq_2008_l625_625092

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem number_multiplied_by_sum_of_digits_eq_2008 : ∃ n : ℕ, n * sum_of_digits n = 2008 ∧ n = 251 := by
  use 251
  have h : sum_of_digits 251 = 8 := by sorry
  have h' : 251 * sum_of_digits 251 = 2008 := by
    rw [h]
    norm_num
  exact ⟨h', rfl⟩

end number_multiplied_by_sum_of_digits_eq_2008_l625_625092


namespace cosine_of_angle_between_planes_l625_625749

theorem cosine_of_angle_between_planes :
  let n1 := ⟨3, -2, -1⟩
  let n2 := ⟨9, -6, -4⟩
  cos_angle n1 n2 = 43 / Real.sqrt 1862 :=
by
  -- Definitions for the normal vectors
  let n1 := ⟨3, -2, -1⟩ : ℝ × ℝ × ℝ
  let n2 := ⟨9, -6, -4⟩ : ℝ × ℝ × ℝ
  -- Define a custom function to compute the cosine of the angle between two vectors
  def cos_angle (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
    let dot_prod := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3
    let mag_v1 := Real.sqrt (v1.1 * v1.1 + v1.2 * v1.2 + v1.3 * v1.3)
    let mag_v2 := Real.sqrt (v2.1 * v2.1 + v2.2 * v2.2 + v2.3 * v2.3)
    dot_prod / (mag_v1 * mag_v2)
  -- Use the defined function
  have h := cos_angle n1 n2
  have result : h = 43 / Real.sqrt 1862 := sorry
  exact result

end cosine_of_angle_between_planes_l625_625749


namespace set_star_result_l625_625139

-- Define the sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

-- Define the operation ∗ between sets A and B
def set_star (A B : Set ℕ) : Set ℕ := {x | ∃ x1 ∈ A, ∃ x2 ∈ B, x = x1 + x2}

-- Rewrite the main theorem to be proven
theorem set_star_result : set_star A B = {2, 3, 4, 5} :=
  sorry

end set_star_result_l625_625139


namespace coefficient_x10_expansion_l625_625041

theorem coefficient_x10_expansion : 
  ∀ (x : ℝ), (∑ k in finset.range (12), (nat.choose 11 k) * x^k * (-1)^(11 - k) : ℝ) = (∑ k in finset.range (12), (nat.choose 11 k) * (-1)^(11 - k) * x^k : ℝ) → 
  (∑ k in finset.range (12), if k = 10 then (nat.choose 11 k) * (-1)^(11 - k) else 0) = -11 :=
by
  intros x h
  sorry

end coefficient_x10_expansion_l625_625041


namespace unique_solution_b_l625_625961

theorem unique_solution_b (b : ℝ) : (|x^2 + 2 * b * x + 2 * b| ≤ 3).countRoots = 1 ↔ b = 3 ∨ b = -1 := 
sorry

end unique_solution_b_l625_625961


namespace find_solutions_l625_625970

-- Definitions
def is_solution (x y z n : ℕ) : Prop :=
  x^3 + y^3 + z^3 = n * (x^2) * (y^2) * (z^2)

-- Theorem statement
theorem find_solutions :
  {sol : ℕ × ℕ × ℕ × ℕ | is_solution sol.1 sol.2.1 sol.2.2.1 sol.2.2.2} =
  {(1, 1, 1, 3), (1, 2, 3, 1), (2, 1, 3, 1)} :=
by sorry

end find_solutions_l625_625970


namespace sum_of_areas_l625_625135

-- Given definitions
def radius := 5 -- cm
def diameter := 2 * radius -- cm
def max_rectangle_area := 50 -- cm² (Area of square inside the circle)
def max_parallelogram_area := 50 -- cm² (Parallelogram tangent to circle)

-- Proof problem statement
theorem sum_of_areas (r : ℝ) (d : ℝ) (rect_area : ℝ) (para_area : ℝ)
  (h1: r = radius)
  (h2: d = diameter)
  (h3: rect_area = max_rectangle_area)
  (h4: para_area = max_parallelogram_area) :
  rect_area + para_area = 100 :=
by
  -- Use provided conditions to establish the result
  rw [h3, h4]
  exact rfl

end sum_of_areas_l625_625135


namespace sum_squares_fraction_l625_625759

theorem sum_squares_fraction (y : Fin 50 → ℝ) (h1 : ∑ i, y i = 2) (h2 : ∑ i, y i / (2 - y i) = 2) :
  ∑ i, y i ^ 2 / (2 - y i) = 2 :=
by
  sorry

end sum_squares_fraction_l625_625759


namespace range_of_a_l625_625662

-- Given function definition under the condition
def f (a x : ℝ) : ℝ := x * (Real.ln x - 2 * a * x)

-- Conditions: x > 0 and seeking range of a
noncomputable def has_two_extreme_points (a : ℝ) : Prop :=
  let g (x : ℝ) := Real.ln x + 1 - 4 * a * x in
  -- Two extreme points condition -> a must satisfy:
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (f a x1).deriv = 0 ∧ (f a x2).deriv = 0

-- Prove the range of a
theorem range_of_a : ∀ a : ℝ, has_two_extreme_points a ↔ 0 < a ∧ a < 1 / 4 := by
  -- Proof goes here
  sorry

end range_of_a_l625_625662


namespace find_lambda_l625_625678

def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos (3/2 * x), Real.sin (3/2 * x))
def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos (1/2 * x), -Real.sin (1/2 * x))

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2
def magnitude (a b : ℝ × ℝ) : ℝ := Real.sqrt (a.1^2 + 2 * (dot_product a b) + b.1^2 + b.2^2)
def f (x λ : ℝ) : ℝ := let a := vector_a x in let b := vector_b x in (dot_product a b - 2 * λ * magnitude a b)

theorem find_lambda (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi/2) (λ : ℝ) :
  (f x λ) = -3/2 ↔ λ = 1/2 :=
by
  sorry

end find_lambda_l625_625678


namespace series_sum_l625_625483

-- Statement of the problem in Lean:
theorem series_sum : 
  (∑ k in finset.range 50, (2 * (100 - k) + 1) * 1) = 5050 := 
by {
    sorry
}

end series_sum_l625_625483


namespace smallest_number_among_l625_625538

theorem smallest_number_among
  (π : ℝ) (Hπ_pos : π > 0) :
  ∀ (a b c d : ℝ), 
    (a = 0) → 
    (b = -1) → 
    (c = -1.5) → 
    (d = π) → 
    (∀ (x y : ℝ), (x > 0) → (y > 0) → (x > y) ↔ x - y > 0) → 
    (∀ (x : ℝ), x < 0 → x < 0) → 
    (∀ (x y : ℝ), (x > 0) → (y < 0) → x > y) → 
    (∀ (x y : ℝ), (x < 0) → (y < 0) → (|x| > |y|) → x < y) → 
  c = -1.5 := 
by
  intros a b c d Ha Hb Hc Hd Hpos Hneg HposNeg Habs
  sorry

end smallest_number_among_l625_625538


namespace percent_area_shaded_l625_625034

-- Conditions: Square $ABCD$ has a side length of 10, and square $PQRS$ has a side length of 15.
-- The overlap of these squares forms a rectangle $AQRD$ with dimensions $20 \times 25$.

theorem percent_area_shaded 
  (side_ABCD : ℕ := 10) 
  (side_PQRS : ℕ := 15) 
  (dim_AQRD_length : ℕ := 25) 
  (dim_AQRD_width : ℕ := 20) 
  (area_AQRD : ℕ := dim_AQRD_length * dim_AQRD_width)
  (overlap_side : ℕ := 10) 
  (area_shaded : ℕ := overlap_side * overlap_side)
  : (area_shaded * 100) / area_AQRD = 20 := 
by 
  sorry

end percent_area_shaded_l625_625034


namespace chairs_to_remove_l625_625078

/-- Given conditions:
1. Each row holds 13 chairs.
2. There are 169 chairs initially.
3. There are 95 expected attendees.

Task: 
Prove that the number of chairs to be removed to ensure complete rows and minimize empty seats is 65. -/
theorem chairs_to_remove (chairs_per_row total_chairs expected_attendees : ℕ)
  (h1 : chairs_per_row = 13)
  (h2 : total_chairs = 169)
  (h3 : expected_attendees = 95) :
  ∃ chairs_to_remove : ℕ, chairs_to_remove = 65 :=
by
  sorry -- proof omitted

end chairs_to_remove_l625_625078


namespace algebraic_expression_value_l625_625990

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 3*x - 1 = 0) :
  (x - 3)^2 - (2*x + 1)*(2*x - 1) - 3*x = 7 :=
sorry

end algebraic_expression_value_l625_625990


namespace fraction_of_seniors_study_japanese_l625_625950

variable (J S : ℕ)
variable (x : ℚ)
variable (num_juniors num_seniors : ℕ)
variable (fraction_students_study_japanese fraction_juniors_study_japanese fraction_seniors_study_japanese : ℚ)

-- Conditions
noncomputable def senior_twice_junior : Prop := S = 2 * J
noncomputable def fraction_juniors_study_japanese : Prop := fraction_juniors_study_japanese = 1 / 4
noncomputable def total_fraction_study_japanese : Prop := fraction_students_study_japanese = 1 / 3
noncomputable def specific_fraction_senior_study_japanese : Prop := (1 / 4) * J + x * S = (1 / 3) * (J + S)
noncomputable def num_study_japanese : Prop := x = 3 / 8

-- Theorem statement
theorem fraction_of_seniors_study_japanese 
(J S : ℕ)
(fraction_students_study_japanese : ℚ)
(fraction_juniors_study_japanese : ℚ)
(senior_twice_junior : S = 2 * J)
(specific_fraction_senior_study_japanese : (1 / 4) * J + (3 / 8) * S = (1 / 3) * (J + S))
: 3 / 8 = 3 / 8 :=
begin
  sorry
end

end fraction_of_seniors_study_japanese_l625_625950


namespace average_age_of_women_l625_625358

-- Defining the conditions
def average_age_of_men : ℝ := 40
def number_of_men : ℕ := 15
def increase_in_average : ℝ := 2.9
def ages_of_replaced_men : List ℝ := [26, 32, 41, 39]
def number_of_women : ℕ := 4

-- Stating the proof problem
theorem average_age_of_women :
  let total_age_of_men := average_age_of_men * number_of_men
  let total_age_of_replaced_men := ages_of_replaced_men.sum
  let new_average_age := average_age_of_men + increase_in_average
  let new_total_age_of_group := new_average_age * number_of_men
  let total_age_of_women := new_total_age_of_group - (total_age_of_men - total_age_of_replaced_men)
  let average_age_of_women := total_age_of_women / number_of_women
  average_age_of_women = 45.375 :=
sorry

end average_age_of_women_l625_625358


namespace youngest_child_age_is_3_l625_625904

noncomputable def family_age_problem : Prop :=
  ∃ (age_diff_2 : ℕ) (age_10_years_ago : ℕ) (new_family_members : ℕ) (same_present_avg_age : ℕ) (youngest_child_age : ℕ),
    age_diff_2 = 2 ∧
    age_10_years_ago = 4 * 24 ∧
    new_family_members = 2 ∧
    same_present_avg_age = 24 ∧
    youngest_child_age = 3 ∧
    (96 + 4 * 10 + (youngest_child_age + (youngest_child_age + age_diff_2)) = 6 * same_present_avg_age)

theorem youngest_child_age_is_3 : family_age_problem := sorry

end youngest_child_age_is_3_l625_625904


namespace z_sq_sub_abs_sq_l625_625989

noncomputable def i := complex.I
noncomputable def z := (1 : ℂ) + i

theorem z_sq_sub_abs_sq : z^2 - complex.abs z ^ 2 = 2 * i - 2 := by
  sorry

end z_sq_sub_abs_sq_l625_625989


namespace distinct_flags_count_l625_625919

def colors : Finset ℕ := {1, 2, 3, 4, 5} -- Assume each color is represented by a unique integer
def num_strips : ℕ := 3
def adjacent_condition (a b : ℕ) : Prop := a ≠ b

theorem distinct_flags_count : 
  (∃ (middle left right : ℕ), 
    middle ∈ colors ∧ 
    left ∈ colors ∧ 
    right ∈ colors ∧ 
    adjacent_condition middle left ∧ 
    adjacent_condition middle right ∧ 
    ∀ flag ∈ {middle, left, right}, flag ∈ colors) →
  (num_strips = 3 ∧ Finset.card colors = 5 ∧ 
    ∃ n, n = 80) :=
sorry

end distinct_flags_count_l625_625919


namespace zero_of_f_in_interval_l625_625819

theorem zero_of_f_in_interval :
  ∃! x ∈ set.Ioo 0 1, (λ x : ℝ, 2^x + 3 * x - 2) x = 0 :=
sorry

end zero_of_f_in_interval_l625_625819


namespace S_n_correct_T_n_correct_l625_625364

def seq_a (n : ℕ) : ℝ := n^2 * (Real.cos (n * Real.pi / 3)^2 - Real.sin (n * Real.pi / 3)^2)

def S_n (n : ℕ) : ℝ :=
  if n % 3 = 1 then -n / 3 - 1 / 6
  else if n % 3 = 2 then (n + 1) * (1 - 3 * n) / 6
  else n * (3 * n + 4) / 6

def seq_b (n : ℕ) : ℝ := S_n (3 * n) / (n * 4^n)

def T_n (n : ℕ) : ℝ := (8 / 3 - 1 / (3 * 2^(2*n - 3)) - 3 * n / 2^(2*n + 1))

theorem S_n_correct (n : ℕ) : S_n n = 
  if n % 3 = 1 then -n / 3 - 1 / 6
  else if n % 3 = 2 then (n + 1) * (1 - 3 * n) / 6
  else n * (3 * n + 4) / 6 := by
  sorry

theorem T_n_correct (n : ℕ) : 
  T_n n = (8 / 3 - 1 / (3 * 2^(2*n - 3)) - 3 * n / 2^(2*n + 1)) := by
  sorry

end S_n_correct_T_n_correct_l625_625364


namespace ship_capacity_and_tax_l625_625518

-- Definitions of conditions from part (a)
def initial_cargo_steel : ℝ := 3428
def initial_cargo_timber : ℝ := 1244
def initial_cargo_electronics : ℝ := 1301
def initial_cargo_total : ℝ := 5973

def additional_cargo_steel : ℝ := 3057
def additional_cargo_textiles : ℝ := 2364
def additional_cargo_timber : ℝ := 1517
def additional_cargo_electronics : ℝ := 1785
def additional_cargo_total : ℝ := 8723

def maximum_ship_capacity : ℝ := 20000

def tax_per_ton_steel : ℝ := 50
def tax_per_ton_timber : ℝ := 75
def tax_per_ton_electronics : ℝ := 100
def tax_per_ton_textiles : ℝ := 40

-- Proof statement
theorem ship_capacity_and_tax :
    let total_cargo := initial_cargo_total + additional_cargo_total in
    total_cargo <= maximum_ship_capacity ∧
    ( (initial_cargo_steel + additional_cargo_steel) * tax_per_ton_steel +
      (initial_cargo_timber + additional_cargo_timber) * tax_per_ton_timber +
      (initial_cargo_electronics + additional_cargo_electronics) * tax_per_ton_electronics +
      additional_cargo_textiles * tax_per_ton_textiles ) = 934485 := 
by 
  sorry

end ship_capacity_and_tax_l625_625518


namespace number_of_truthful_dwarfs_l625_625598

/-- 
Each of the 10 dwarfs either always tells the truth or always lies. 
Each dwarf likes exactly one type of ice cream: vanilla, chocolate, or fruit.
When asked, every dwarf raised their hand for liking vanilla ice cream.
When asked, 5 dwarfs raised their hand for liking chocolate ice cream.
When asked, only 1 dwarf raised their hand for liking fruit ice cream.
Prove that the number of truthful dwarfs is 4.
-/
theorem number_of_truthful_dwarfs (T L : ℕ) 
  (h1 : T + L = 10) 
  (h2 : T + 2 * L = 16) : 
  T = 4 := 
by
  -- Proof omitted
  sorry

end number_of_truthful_dwarfs_l625_625598


namespace line_and_curve_properties_l625_625216

noncomputable def line_param_equation (t : ℝ) : ℝ × ℝ :=
  (-1 - (3 / 5) * t, 2 + (4 / 5) * t)

noncomputable def curve_polar_equation (theta : ℝ) : ℝ :=
  2 * real.sqrt 2 * real.cos (theta - real.pi / 4)

theorem line_and_curve_properties :
  (∀ t, line_param_equation t = ((-1 - (3 / 5) * t), (2 + (4 / 5) * t))) →
  (∀ theta, curve_polar_equation theta = 2 * real.sqrt 2 * real.cos (theta - real.pi / 4)) →
  (line_cartesian_eq : ∀ x y, 4*x + 3*y - 2 = 0) →
  (curve_cartesian_eq : ∀ x y, x^2 + y^2 - 2*x - 2*y = 0) →
  (|AB_length| : ∀ t1 t2, |t1 - t2| = 2) :=
sorry

end line_and_curve_properties_l625_625216


namespace not_exists_sequence_exists_sequence_l625_625481
open Nat

-- Part (a)
theorem not_exists_sequence (a : ℕ → ℕ) : ¬ (∀ i j, i < j → gcd (a i + j) (a j + i) = 1) :=
sorry

-- Part (b)
theorem exists_sequence (p : ℕ) [Prime p] (h : p ≠ 2) : 
  ∃ a : ℕ → ℕ, ∀ i j, i < j → ¬ p ∣ gcd (a i + j) (a j + i) :=
sorry

end not_exists_sequence_exists_sequence_l625_625481


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625444

/-- Problem statement: 
    The smallest positive perfect square that is divisible by 2, 3, and 5 is 900.
-/
theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ n * n % 2 = 0 ∧ n * n % 3 = 0 ∧ n * n % 5 = 0 ∧ n * n = 900 := 
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625444


namespace mike_spent_on_speakers_l625_625772

-- Definitions of the conditions:
def total_car_parts_cost : ℝ := 224.87
def new_tires_cost : ℝ := 106.33

-- Statement of the proof problem:
theorem mike_spent_on_speakers : total_car_parts_cost - new_tires_cost = 118.54 :=
by
  sorry

end mike_spent_on_speakers_l625_625772


namespace number_of_truthful_dwarfs_l625_625600

/-- 
Each of the 10 dwarfs either always tells the truth or always lies. 
Each dwarf likes exactly one type of ice cream: vanilla, chocolate, or fruit.
When asked, every dwarf raised their hand for liking vanilla ice cream.
When asked, 5 dwarfs raised their hand for liking chocolate ice cream.
When asked, only 1 dwarf raised their hand for liking fruit ice cream.
Prove that the number of truthful dwarfs is 4.
-/
theorem number_of_truthful_dwarfs (T L : ℕ) 
  (h1 : T + L = 10) 
  (h2 : T + 2 * L = 16) : 
  T = 4 := 
by
  -- Proof omitted
  sorry

end number_of_truthful_dwarfs_l625_625600


namespace exists_unique_x_n_limit_x_n_l625_625067

-- Part (a)
theorem exists_unique_x_n (n : ℕ) : ∃! x : ℝ, 0 < x ∧ x^n + x^(n+1) = 1 :=
sorry

-- Part (b)
theorem limit_x_n (x_n : ℕ → ℝ) (h : ∀ n : ℕ, ∃! x : ℝ, 0 < x ∧ x^n + x^(n+1) = 1) : 
  filter.tendsto (x_n) filter.at_top (filter.principal {1}) :=
sorry

end exists_unique_x_n_limit_x_n_l625_625067


namespace volunteers_from_third_grade_l625_625845

theorem volunteers_from_third_grade 
    (total_students : ℕ) (first_grade_students : ℕ) 
    (second_grade_students : ℕ) (third_grade_students : ℕ)
    (total_volunteers : ℕ) :
    total_students = 2040 →
    first_grade_students = 680 →
    second_grade_students = 850 →
    third_grade_students = 510 →
    total_volunteers = 12 →
    third_grade_students * total_volunteers / total_students = 3 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h4, h5]
  norm_num
  sorry

end volunteers_from_third_grade_l625_625845


namespace largest_c_interval_not_contain_integer_l625_625577

theorem largest_c_interval_not_contain_integer :
  ∃ (c : ℝ), c = 6 - 4 * Real.sqrt 2 ∧
  (∀ (n : ℕ), ∀ (k : ℤ),
    ¬((↑n * Real.sqrt 2 - c / ↑n < ↑k) ∧ (↑k < ↑n * Real.sqrt 2 + c / ↑n))) :=
by
  use 6 - 4 * Real.sqrt 2
  split
  · exact rfl 
  · intros n k h 
    sorry

end largest_c_interval_not_contain_integer_l625_625577


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625446

/-- Problem statement: 
    The smallest positive perfect square that is divisible by 2, 3, and 5 is 900.
-/
theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ n * n % 2 = 0 ∧ n * n % 3 = 0 ∧ n * n % 5 = 0 ∧ n * n = 900 := 
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625446


namespace determine_f_function_l625_625635

variable (f : ℝ → ℝ)

theorem determine_f_function (x : ℝ) (h : f (1 - x) = 1 + x) : f x = 2 - x := 
sorry

end determine_f_function_l625_625635


namespace handshakes_at_gathering_l625_625547

theorem handshakes_at_gathering (n : ℕ) (people : Fin n → Type) (couples : Set (Type × Type)) :
  n = 16 ∧ (∀ p ∈ people, ∃ s ∈ people, (s, p) ∈ couples) ∧
  (∀ p ∈ people, ∃ q ∈ people, q ≠ p ∧ q ≠ (λ s, ∃ c ∈ couples, (s, p) = c ∨ (p, s) = c)) ↔
  handshakes = 104 :=
by
  sorry

end handshakes_at_gathering_l625_625547


namespace tony_exercise_hours_per_week_l625_625397

variable (dist_walk dist_run speed_walk speed_run days_per_week : ℕ)

#eval dist_walk : ℕ := 3
#eval dist_run : ℕ := 10
#eval speed_walk : ℕ := 3
#eval speed_run : ℕ := 5
#eval days_per_week : ℕ := 7

theorem tony_exercise_hours_per_week :
  (dist_walk / speed_walk + dist_run / speed_run) * days_per_week = 21 := by
  sorry

end tony_exercise_hours_per_week_l625_625397


namespace sum_a_le_n2_l625_625309

open Nat

theorem sum_a_le_n2 (n : ℕ) (a : ℕ → ℕ) (h_positive: 0 < n)
  (h_periodic : ∀ i, a (n + i) = a i)
  (h_ordered : ∀ i, i ≤ n → a i ≤ a (i + 1))
  (h_bound : ∀ i, i ≤ n → a i ≤ a 1 + n)
  (h_condition : ∀ i, 1 ≤ i ∧ i ≤ n → a (a i) ≤ n + i - 1) :
  (∑ i in finRange n, a i.succ) ≤ n^2 := by
  sorry

end sum_a_le_n2_l625_625309


namespace range_of_m_l625_625649

noncomputable def condition_p (x : ℝ) : Prop := -2 < x ∧ x < 10
noncomputable def condition_q (x m : ℝ) : Prop := (x - 1)^2 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) :
  (∀ x, condition_p x → condition_q x m) ∧ (∃ x, ¬ condition_p x ∧ condition_q x m) ↔ 9 ≤ m := sorry

end range_of_m_l625_625649


namespace three_layer_overlap_area_l625_625018

/-!
## Problem:
The school principal, the janitor, and the parent committee each bought a rug for the school assembly hall sized 10 × 10 meters.
- First rug: 6 × 8 meters, placed in one corner.
- Second rug: 6 × 6 meters, placed in the opposite corner.
- Third rug: 5 × 7 meters, placed in one of the remaining corners.

Prove that the area of the part of the hall covered by rugs in three layers is 6 square meters.
-/

def hall_length : ℕ := 10
def hall_width : ℕ := 10

def rug1_length : ℕ := 6
def rug1_width : ℕ := 8

def rug2_length : ℕ := 6
def rug2_width : ℕ := 6

def rug3_length : ℕ := 5
def rug3_width : ℕ := 7

theorem three_layer_overlap_area : 
  (part_of_hall_covered_by_rugs_in_three_layers hall_length hall_width rug1_length rug1_width rug2_length rug2_width rug3_length rug3_width) = 6 := 
sorry

-- Definitions for understanding the problem context, the area calculations, and placement logic
noncomputable def part_of_hall_covered_by_rugs_in_three_layers 
  (hl hw r1l r1w r2l r2w r3l r3w : ℕ) : ℕ :=
-- This part would involve the actual calculations and understanding of the overlap logic
sorry

end three_layer_overlap_area_l625_625018


namespace range_of_f_gt_f_2x_l625_625660

def f (x : ℝ) : ℝ := (x - 1) ^ 4 + 2 * abs (x - 1)

theorem range_of_f_gt_f_2x :
  {x : ℝ | f x > f (2 * x)} = set.Ioo 0 (2 / 3) :=
by
  sorry

end range_of_f_gt_f_2x_l625_625660


namespace five_digit_sine_rule_count_l625_625081

theorem five_digit_sine_rule_count :
    ∃ (count : ℕ), 
        (∀ (a b c d e : ℕ), 
          (a <  b) ∧
          (b >  c) ∧
          (c >  d) ∧
          (d <  e) ∧
          (a >  d) ∧
          (b >  e) ∧
          (∃ (num : ℕ), num = 10000 * a + 1000 * b + 100 * c + 10 * d + e))
        →
        count = 2892 :=
sorry

end five_digit_sine_rule_count_l625_625081


namespace simplify_expr_l625_625340

/-- Theorem: Simplify the expression -/
theorem simplify_expr
  (x y z w : ℝ)
  (hx : x = sqrt 3 - 1)
  (hy : y = sqrt 3 + 1)
  (hz : z = 1 - sqrt 2)
  (hw : w = 1 + sqrt 2) :
  (x ^ z / y ^ w) = 2 ^ (1 - sqrt 2) * (4 - 2 * sqrt 3) :=
by
  sorry

end simplify_expr_l625_625340


namespace weekly_exercise_time_l625_625393

def milesWalked := 3
def walkingSpeed := 3 -- in miles per hour
def milesRan := 10
def runningSpeed := 5 -- in miles per hour
def daysInWeek := 7

theorem weekly_exercise_time : (milesWalked / walkingSpeed + milesRan / runningSpeed) * daysInWeek = 21 := 
by
  -- The actual proof part is intentionally omitted as per the instruction
  sorry

end weekly_exercise_time_l625_625393


namespace nth_group_sum_correct_l625_625822

-- Define the function that computes the sum of the numbers in the nth group
def nth_group_sum (n : ℕ) : ℕ :=
  n * (n^2 + 1) / 2

-- The theorem statement
theorem nth_group_sum_correct (n : ℕ) : 
  nth_group_sum n = n * (n^2 + 1) / 2 := by
  sorry

end nth_group_sum_correct_l625_625822


namespace either_d_or_2d_is_perfect_square_l625_625821

theorem either_d_or_2d_is_perfect_square
  (a c d : ℕ) (hrel_prime : Nat.gcd a c = 1) (hd : ∃ D : ℝ, D = d ∧ (D:ℝ) > 0)
  (hdiam : d^2 = 2 * a^2 + c^2) :
  ∃ m : ℕ, m^2 = d ∨ m^2 = 2 * d :=
by
  sorry

end either_d_or_2d_is_perfect_square_l625_625821


namespace Tabitha_final_money_l625_625802

def initial_money : ℕ := 45
def money_given_to_mom : ℕ := 10
def investment_percentage : Rat := 0.60
def number_of_items : ℕ := 12
def cost_per_item : Rat := 0.75

theorem Tabitha_final_money :
  (initial_money - money_given_to_mom - (investment_percentage * (initial_money - money_given_to_mom)).toNat - (number_of_items * cost_per_item).toNat) = 5 :=
  by
  sorry

end Tabitha_final_money_l625_625802


namespace solve_xyz_l625_625413

theorem solve_xyz (x y z : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) :
  (x / 21) * (y / 189) + z = 1 ↔ x = 21 ∧ y = 567 ∧ z = 0 :=
sorry

end solve_xyz_l625_625413


namespace g_12_eq_36_l625_625494

noncomputable def f (x : ℝ) : ℝ := x * (x + 1)
noncomputable def g (x : ℝ) : ℝ := (-1 + Real.sqrt(36 * x^2 + 12 * x + 1)) / 2

theorem g_12_eq_36 : g 12 = 36 := by
  sorry

end g_12_eq_36_l625_625494


namespace exists_multiple_of_n_with_ones_l625_625301

theorem exists_multiple_of_n_with_ones (n : ℤ) (hn1 : n ≥ 1) (hn2 : Int.gcd n 10 = 1) :
  ∃ k : ℕ, n ∣ (10^k - 1) / 9 :=
by sorry

end exists_multiple_of_n_with_ones_l625_625301


namespace minimum_expression_value_l625_625167

theorem minimum_expression_value (m n : ℝ) (h1 : m > 1) (h2 : n > 0) (h3 : m^2 - 3 * m + n = 0) :
  ∃ x, (∀ m, 1 < m ∧ m < 3 → let n := 3 * m - m^2 in x ≤ (4 / (m - 1) + m / n)) ∧ x = 9 / 2 :=
begin
  sorry
end

end minimum_expression_value_l625_625167


namespace complex_translation_example_l625_625530

def translate (z : ℂ) (w : ℂ) : ℂ := z + w

theorem complex_translation_example :
    (∀ z : ℂ, (1 + 3 * I) → (5 + 7 * I) → z) →
    translate (2 - 2 * I) (4 + 4 * I) = 6 + 2 * I :=
by
  intro h
  sorry

end complex_translation_example_l625_625530


namespace mark_score_is_46_l625_625260

theorem mark_score_is_46 (highest_score : ℕ) (range: ℕ) (mark_score : ℕ) :
  highest_score = 98 →
  range = 75 →
  (mark_score = 2 * (highest_score - range)) →
  mark_score = 46 := by
  intros
  sorry

end mark_score_is_46_l625_625260


namespace lcm_20_45_75_l625_625886

def lcm (a b : ℕ) : ℕ := nat.lcm a b

theorem lcm_20_45_75 : lcm (lcm 20 45) 75 = 900 :=
by
  sorry

end lcm_20_45_75_l625_625886


namespace BillyFish_l625_625552

def BenFish : Nat := 4
def JudyFish : Nat := 1
def JimFish : Nat := 2
def SusieFish : Nat := 5
def ThrownBackFish : Nat := 3
def FishFilets : Nat := 24
def FiletsPerFish : Nat := 2

theorem BillyFish :
  let TotalFishKept := FishFilets / FiletsPerFish in
  let TotalFishCaught := TotalFishKept + ThrownBackFish in
  ∃ BillyFish : Nat, BillyFish + BenFish + JudyFish + JimFish + SusieFish = TotalFishCaught :=
  by
    sorry

end BillyFish_l625_625552


namespace min_value_shift_symmetry_l625_625818

noncomputable def f : ℝ → ℝ :=
  λ x, sin (x + π / 3) + sin x

theorem min_value_shift_symmetry :
  ∃ a : ℝ, (∀ x : ℝ, f (x - a) = f (x + a)) ∧ a = π / 3 := by
  sorry

end min_value_shift_symmetry_l625_625818


namespace exchange_silver_cards_l625_625071

theorem exchange_silver_cards : 
  (∃ red gold silver : ℕ,
    (∀ (r g s : ℕ), ((2 * g = 5 * r) ∧ (g = r + s) ∧ (r = 3) ∧ (g = 3) → s = 7))) :=
by
  sorry

end exchange_silver_cards_l625_625071


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625440

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ (n : ℕ), (∃ k : ℕ, n = k^2) ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, (∃ k : ℕ, m = k^2) ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m :=
sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625440


namespace smallest_marble_count_l625_625563

theorem smallest_marble_count (N : ℕ) (a b c : ℕ) (h1 : N > 1)
  (h2 : N ≡ 2 [MOD 5])
  (h3 : N ≡ 2 [MOD 7])
  (h4 : N ≡ 2 [MOD 9]) : N = 317 :=
sorry

end smallest_marble_count_l625_625563


namespace equilateral_triangle_if_cycle_complete_l625_625177

noncomputable def triangle_is_equilateral 
  (P : ℕ → ℝ × ℝ)
  (A : ℕ → ℝ × ℝ)
  (rotation : (ℝ × ℝ) → ℝ → (ℝ × ℝ) → (ℝ × ℝ)) : Prop :=
  ∀ k, P (k + 1) = rotation (A (k + 1)) (2 * Math.pi / 3) (P k)

theorem equilateral_triangle_if_cycle_complete 
  (P : ℕ → ℝ × ℝ)
  (A : Nat → ℝ × ℝ)
  (rotation : (ℝ × ℝ) → ℝ → (ℝ × ℝ) → (ℝ × ℝ))
  (hA : ∀ s ≥ 4, A s = A (s - 3))
  (hP_seq : triangle_is_equilateral P A rotation)
  (hCycle : P 1986 = P 0) :
  ∀ (a1 a2 a3 : ℝ × ℝ), Set.Pairwise ({a1, a2, a3} : Set (ℝ × ℝ)) (≠) → 
    ∀ (A1 A2 A3 : ℕ → ℝ × ℝ), A1 1 = a1 → A2 2 = a2 → A3 3 = a3 →
    Set.Equilateral (A1 1) (A2 2) (A3 3) :=
sorry

end equilateral_triangle_if_cycle_complete_l625_625177


namespace day_197_of_2005_is_tuesday_l625_625243

-- Definitions based on conditions and question
def is_day_of_week (n : ℕ) (day : ℕ) : Prop := 
  (day % 7) = n

theorem day_197_of_2005_is_tuesday (h : is_day_of_week 2 15) : is_day_of_week 2 197 := 
by
  -- Here, we will state the equivalence under Lean definition (saying remainder modulo 7)
  sorry

end day_197_of_2005_is_tuesday_l625_625243


namespace complex_number_problem_l625_625753

def z : ℂ := 1 + complex.I

theorem complex_number_problem : (2 / z + conj(z) = 2 - 2 * complex.I) :=
by
  sorry

end complex_number_problem_l625_625753


namespace decreasing_log_range_l625_625210

theorem decreasing_log_range (a : ℝ) :
  (∀ x : ℝ, x ≤ 1 → (1 - a * x > 0) ∧ (a > 0 ∧ a < 1) → f(x) = log (3:ℝ) (1 - a * x) ∧ ∀ y : ℝ, y ≤ 1 → f(x) < f(y)) ↔ (0 < a ∧ a < 1) :=
by
  sorry

end decreasing_log_range_l625_625210


namespace parallel_vectors_x_value_l625_625186

/-
Given that \(\overrightarrow{a} = (1,2)\) and \(\overrightarrow{b} = (2x, -3)\) are parallel vectors, prove that \(x = -\frac{3}{4}\).
-/
theorem parallel_vectors_x_value (x : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (2 * x, -3)) 
  (h_parallel : (a.1 * b.2) - (a.2 * b.1) = 0) : 
  x = -3 / 4 := by
  sorry

end parallel_vectors_x_value_l625_625186


namespace position_relationship_l625_625834

-- Define the parabola C
def parabolaC (p : ℝ) (x y : ℝ) := y^2 = x

-- Define the circle centered at M(2,0) and tangent to line x=1
def circleM (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Given conditions
def conditions (P Q : ℝ × ℝ) :=
  P.1 = 1 ∧ Q.1 = 1 ∧ P.2 = sqrt 2 ∧ Q.2 = -sqrt 2 ∧
  (O : ℝ × ℝ) := (0,0) ∧
  (P.1 - O.1) * (Q.1 - O.1) + (P.2 - O.2) * (Q.2 - O.2) = 0

-- Positions on the parabola C, with lines tangent to the circle
def tangent_lines (A1 A2 A3 : ℝ × ℝ) :=
  parabolaC 1 A1.1 A1.2 ∧ parabolaC 1 A2.1 A2.2 ∧ parabolaC 1 A3.1 A3.2 ∧
  (lineA1A2 := λ x : ℝ, (A1.1 + A2.1) * x - (A1.1 * A2.1)) ∧
  (lineA1A3 := λ x : ℝ, (A1.1 + A3.1) * x - (A1.1 * A3.1)) ∧
  circleM (2, 0) 1 A2.1 A2.2 ∧ circleM (2, 0) 1 A3.1 A3.2

-- Main theorem
theorem position_relationship (A1 A2 A3 : ℝ × ℝ) :
  tangent_lines A1 A2 A3 →
  ∀ (O : ℝ × ℝ), (A2.1 - O.1) * (A3.1 - O.1) + (A2.2 - O.2) * (A3.2 - O.2) = 0 → circleM (2,0) 1 (A2.1) (A2.2) :=
sorry

end position_relationship_l625_625834


namespace log_expression_value_l625_625557

theorem log_expression_value : 
  (Real.logb 10 (Real.sqrt 2) + Real.logb 10 (Real.sqrt 5) + 2 ^ 0 + (5 ^ (1 / 3)) ^ 2 * Real.sqrt 5 = 13 / 2) := 
by 
  -- The proof is omitted as per the instructions
  sorry

end log_expression_value_l625_625557


namespace sum_first_15_terms_l625_625213

def a_n (n : ℕ) : ℚ :=
  if n % 2 = 1 then 1 / (n * (n + 2)) else n - 7

def S_15 : ℚ := 
  (Finset.range 15).sum (λ n, a_n (n + 1))

theorem sum_first_15_terms : S_15 = 127 / 17 := by
  sorry

end sum_first_15_terms_l625_625213


namespace john_total_distance_l625_625485

theorem john_total_distance (speed1 time1 speed2 time2 : ℕ) (distance1 distance2 : ℕ) :
  speed1 = 35 →
  time1 = 2 →
  speed2 = 55 →
  time2 = 3 →
  distance1 = speed1 * time1 →
  distance2 = speed2 * time2 →
  distance1 + distance2 = 235 := by
  intros
  sorry

end john_total_distance_l625_625485


namespace number_of_truthful_dwarfs_l625_625585

-- Given conditions
variables (D : Type) [Fintype D] [DecidableEq D] [Card D = 10]
variables (IceCream : Type) [DecidableEq IceCream] (vanilla chocolate fruit : IceCream)
-- Assuming each dwarf likes exactly one type of ice cream
variable (Likes : D → IceCream)
-- Functions indicating if a dwarf raised their hand for each type of ice cream
variables (raisedHandForVanilla raisedHandForChocolate raisedHandForFruit : D → Prop)

-- Given conditions translated to Lean
axiom all_dwarfs_raised_for_vanilla : ∀ d, raisedHandForVanilla d
axiom half_dwarfs_raised_for_chocolate : Fintype.card {d // raisedHandForChocolate d} = 5
axiom one_dwarf_raised_for_fruit : Fintype.card {d // raisedHandForFruit d} = 1

-- Define that a dwarf either always tells the truth or always lies
inductive TruthStatus
| truthful : TruthStatus
| liar : TruthStatus

variable (Status : D → TruthStatus)

-- Definitions related to hand-raising based on dwarf's status and ice cream they like
def raisedHandCorrectly (d : D) : Prop :=
  match Status d with
  | TruthStatus.truthful => 
      raisedHandForVanilla d ↔ Likes d = vanilla ∧
      raisedHandForChocolate d ↔ Likes d = chocolate ∧
      raisedHandForFruit d ↔ Likes d = fruit
  | TruthStatus.liar =>
      raisedHandForVanilla d ↔ Likes d ≠ vanilla ∧
      raisedHandForChocolate d ↔ Likes d ≠ chocolate ∧
      raisedHandForFruit d ↔ Likes d ≠ fruit

-- Goal to prove
theorem number_of_truthful_dwarfs : Fintype.card {d // Status d = TruthStatus.truthful} = 4 :=
by sorry

end number_of_truthful_dwarfs_l625_625585


namespace angle_RQS_l625_625265

theorem angle_RQS (P Q R S : Type) [MetricSpace P]
  (hpq : PQ = PR)
  (hqs : PQ = QS)
  (h_angle : ∠QPR = 20) :
  ∠RQS = 60 :=
sorry

end angle_RQS_l625_625265


namespace bus_speed_is_30_l625_625930

def pedestrian_speed : ℝ := 5
def bus_encounter_time : ℝ := 5 / 60  -- in hours
def observation_time : ℝ := 2
def bus_difference : ℕ := 4
def total_encounters : ℕ := 24

theorem bus_speed_is_30 : ∃ v_b : ℝ, v_b = 30 ∧
  let N_oncoming := total_encounters / 2 + bus_difference / 2 in
  let N_overtaking := total_encounters / 2 - bus_difference / 2 in
  N_oncoming * (v_b + pedestrian_speed) = N_overtaking * (v_b - pedestrian_speed) :=
begin
  sorry
end

end bus_speed_is_30_l625_625930


namespace studentC_spending_l625_625389

-- Definitions based on the problem conditions

-- Prices of Type A and Type B notebooks, respectively
variables (x y : ℝ)

-- Number of each type of notebook bought by Student A
def studentA : Prop := x + y = 3

-- Number of Type A notebooks bought by Student B
variables (a : ℕ)

-- Total cost and number of notebooks bought by Student B
def studentB : Prop := (x * a + y * (8 - a) = 11)

-- Constraints on the number of Type A and B notebooks bought by Student C
def studentC_notebooks : Prop := ∃ b : ℕ, b = 8 - a ∧ b = a

-- The total amount spent by Student C
def studentC_cost : ℝ := (8 - a) * x + a * y

-- The statement asserting the cost is 13 yuan
theorem studentC_spending (x y : ℝ) (a : ℕ) (hA : studentA x y) (hB : studentB x y a) (hC : studentC_notebooks a) : studentC_cost x y a = 13 := sorry

end studentC_spending_l625_625389


namespace fair_coin_head_prob_l625_625057

theorem fair_coin_head_prob : 
  (∀ (Ω : Type) (flip : Ω → bool), (flip = fun _ => if (1 / 2 : ℝ) = 1 / 2 then tt else ff) →
  (flip ()) = tt ↔ (1 / 2 : ℝ) = 1 / 2) :=
by 
  sorry

end fair_coin_head_prob_l625_625057


namespace odd_function_f_l625_625248

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x - 1 else -(x - 1)

theorem odd_function_f (x : ℝ) (hx : x < 0) : f x = x + 1 :=
by
  have h_neg_x_pos : -x > 0 := by linarith
  have h_f_neg_x : f (-x) = -x - 1 := by simp [f, h_neg_x_pos]
  have h_odd_f : f (-x) = -f x := by rw [h_f_neg_x, f, hx]; linarith
  rw [← h_odd_f]; linarith

#print odd_function_f

end odd_function_f_l625_625248


namespace average_infect_influence_l625_625116

theorem average_infect_influence
  (x : ℝ)
  (h : (1 + x)^2 = 100) :
  x = 9 :=
sorry

end average_infect_influence_l625_625116


namespace stratified_sampling_l625_625075

theorem stratified_sampling (total_elderly: ℕ) (total_middle_aged: ℕ) (total_young: ℕ) (sample_size: ℕ)
  (h_elderly: total_elderly = 55) (h_middle_aged: total_middle_aged = 108) (h_young: total_young = 162) (h_sample_size: sample_size = 36) :
  let total_population := total_elderly + total_middle_aged + total_young
  let sampling_ratio := (sample_size: ℝ) / total_population
  let n_elderly := (sampling_ratio * total_elderly : ℝ).floor
  let n_middle_aged := (sampling_ratio * total_middle_aged : ℝ).floor
  let n_young := (sampling_ratio * total_young : ℝ).floor
  n_elderly = 6 ∧ n_middle_aged = 12 ∧ n_young = 18 :=
by
  sorry

end stratified_sampling_l625_625075


namespace integral_ge_one_l625_625738

open IntervalIntegral

theorem integral_ge_one (f : ℝ → ℝ) (h_int : IntervalIntegrable f volume 0 1)
  (h_pos : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 < f x)
  (h_sym : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x * f (1 - x) = 1) :
  ∫ x in 0..1, f x ≥ 1 :=
by
  sorry

end integral_ge_one_l625_625738


namespace bridge_length_l625_625087

-- Definitions based on the provided conditions.
def speed_km_per_hr : ℝ := 5
def time_min : ℝ := 15

-- Conversion constants
def km_to_m : ℝ := 1000
def hr_to_min : ℝ := 60

-- Calculate the speed in m/min.
def speed_m_per_min : ℝ := (speed_km_per_hr * km_to_m) / hr_to_min

-- Calculate the distance covered in 15 minutes.
def distance : ℝ := speed_m_per_min * time_min

-- The goal is to prove that the distance is approximately 1250 meters.
theorem bridge_length : distance ≈ 1250 := 
by
  sorry

end bridge_length_l625_625087


namespace emiliano_fruit_consumption_l625_625256

-- Define the problem conditions
def num_apples := 15
def ratio_apples_oranges := 4
def ratio_bananas_oranges := 3

-- Define Fruit Type
structure FruitBasket where
  apples : ℕ
  oranges : ℕ
  bananas : ℕ

-- Given the basket information
def basket : FruitBasket :=
  { apples := num_apples,
    oranges := num_apples / ratio_apples_oranges,
    bananas := (num_apples / ratio_apples_oranges) * ratio_bananas_oranges }

-- Function to calculate the fruits Emiliano consumes
def emilianoConsume (basket : FruitBasket) : ℕ :=
  let apples_eaten := (3 / 5 : ℚ) * basket.apples
  let oranges_eaten := (2 / 3 : ℚ) * basket.oranges
  let bananas_eaten := (4 / 7 : ℚ) * basket.bananas
  apples_eaten.to_nat + oranges_eaten.to_nat + bananas_eaten.to_nat

-- Prove that Emiliano consumes a total of 16 fruits.
theorem emiliano_fruit_consumption : emilianoConsume basket = 16 := by
  sorry

end emiliano_fruit_consumption_l625_625256


namespace area_parabola_tangent_l625_625806

theorem area_parabola_tangent :
  let parabola := λ (x : ℝ), x^2,
      tangent (x : ℝ) := 2*x - 1 in
  (∫ x in (0 : ℝ)..(1 : ℝ), parabola x) - (1 / 2 * 1 / 2 * 1) = 1 / 12 :=
by
  sorry

end area_parabola_tangent_l625_625806


namespace bungee_spring_constant_l625_625509

-- Defining the conditions
def mass (m : ℝ) : Prop := m > 0
def max_distance_fallen (H : ℝ) : Prop := H > 0
def initial_length (L0 : ℝ) : Prop := L0 > 0
def final_length_stretched (L0 h : ℝ) : Prop := L0 + h > L0
def max_tension (m g : ℝ) : Prop := 4 * m * g

-- The proof problem
theorem bungee_spring_constant (m H L0 h k g : ℝ)
  (hm : mass m)
  (hH : max_distance_fallen H)
  (hL0 : initial_length L0)
  (hL : final_length_stretched L0 h)
  (hT : max_tension m g) :
  k = 8 * m * g / H :=
sorry

end bungee_spring_constant_l625_625509


namespace sum_of_odd_coefficients_in_binomial_expansion_l625_625264

theorem sum_of_odd_coefficients_in_binomial_expansion :
  let a_0 := 1
  let a_1 := 10
  let a_2 := 45
  let a_3 := 120
  let a_4 := 210
  let a_5 := 252
  let a_6 := 210
  let a_7 := 120
  let a_8 := 45
  let a_9 := 10
  let a_10 := 1
  (a_1 + a_3 + a_5 + a_7 + a_9) = 512 := by
  sorry

end sum_of_odd_coefficients_in_binomial_expansion_l625_625264


namespace gcd_840_1764_l625_625008

theorem gcd_840_1764 : gcd 840 1764 = 84 := 
by 
  sorry

end gcd_840_1764_l625_625008


namespace problem_statement_l625_625362

noncomputable def roots (a b c : ℝ) : ℝ × ℝ := 
let Δ := b^2 - 4*a*c in
if Δ < 0 then (0, 0) else ((-b + Real.sqrt Δ) / (2*a), (-b - Real.sqrt Δ) / (2*a))

theorem problem_statement : 
  let (x1, x2) := roots 1 5 1 in
  (x1^2 + 5*x1 + 1 = 0 ∧ x2^2 + 5*x2 + 1 = 0) →
  (let expr := (x1 * Real.sqrt 6 / (1 + x2))^2 + (x2 * Real.sqrt 6 / (1 + x1))^2 
  in expr = 220) := 
by {
  sorry
}

end problem_statement_l625_625362


namespace region_area_l625_625059

/-- 
  Trapezoid has side lengths 10, 10, 10, and 22. 
  Each side of the trapezoid is the diameter of a semicircle 
  with the two semicircles on the two parallel sides of the trapezoid facing outside 
  and the other two semicircles facing inside the trapezoid.
  The region bounded by these four semicircles has area m + nπ, where m and n are positive integers.
  Prove that m + n = 188.5.
-/
theorem region_area (m n : ℝ) (h1: m = 128) (h2: n = 60.5) : m + n = 188.5 :=
by
  rw [h1, h2]
  norm_num -- simplifies the expression and checks it is equal to 188.5

end region_area_l625_625059


namespace log_a_b_eq_pi_l625_625915

theorem log_a_b_eq_pi (a b : ℝ) (C r : ℝ) 
    (h1 : r = log 10 (3 * a^3))
    (h2 : C = log 10 (4 * b^6))
    (h3 : C = 2 * π * r) : 
    log a b = π := by 
  sorry

end log_a_b_eq_pi_l625_625915


namespace even_function_a_value_l625_625244

def f (x a : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_a_value (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 1 := by
  sorry

end even_function_a_value_l625_625244


namespace number_less_than_neg_two_l625_625113

theorem number_less_than_neg_two :
  ∃ n ∈ ({1, 0, -1, -3} : set ℤ), n < -2 ∧ ∀ m ∈ ({1, 0, -1, -3} : set ℤ), m < -2 → m = n :=
begin
  sorry
end

end number_less_than_neg_two_l625_625113


namespace probability_non_first_class_product_l625_625787

theorem probability_non_first_class_product (P_A P_B P_C : ℝ) (hA : P_A = 0.65) (hB : P_B = 0.2) (hC : P_C = 0.1) : 1 - P_A = 0.35 :=
by
  sorry

end probability_non_first_class_product_l625_625787


namespace find_k_value_l625_625382

theorem find_k_value : 
  let a := 3 ^ 1001
  let b := 4 ^ 1002
  (a + b) ^ 2 - (a - b) ^ 2 = 16 * 12 ^ 1001 :=
by
  let a := 3 ^ 1001
  let b := 4 ^ 1002
  sorry

end find_k_value_l625_625382


namespace sum_factorial_eq_l625_625327

-- Define the sum S_n for natural n: 1*1! + 2*2! + ... + n*n!
def S (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), (i + 1) * Nat.factorial (i + 1)

-- The proposition to prove
theorem sum_factorial_eq (n : ℕ) : S n = Nat.factorial (n + 1) - 1 := 
sorry

end sum_factorial_eq_l625_625327


namespace range_of_a_l625_625636

theorem range_of_a (p q : Prop) (hp : ∀ x ∈ set.Ioi (0 : ℝ), x + x⁻¹ ≥ a^2 - a) 
  (hq : ∃ x : ℝ, x + abs (x - 1) = 2 * a) : 
  (1 / 2 : ℝ) ≤ a ∧ a ≤ 2 :=
by sorry

end range_of_a_l625_625636


namespace solve_system_l625_625831

-- The system of equations as conditions in Lean
def system1 (x y : ℤ) : Prop := 5 * x + 2 * y = 25
def system2 (x y : ℤ) : Prop := 3 * x + 4 * y = 15

-- The statement that asserts the solution is (x = 5, y = 0)
theorem solve_system : ∃ x y : ℤ, system1 x y ∧ system2 x y ∧ x = 5 ∧ y = 0 :=
by
  sorry

end solve_system_l625_625831


namespace triangle_ADC_area_l625_625273

noncomputable def area_of_triangle_ADC (y : ℝ) (h₁ : 100^2 + y^2 = (3 * y - 10)^2) : ℝ :=
  let AC := 3 * y - 10 in
  let BD := (100 / AC) * (100^2 - AC^2 / y - 1) in
  let DC := y - BD in
  (1 / 2) * 100 * DC

theorem triangle_ADC_area (y : ℝ) (h₁ : 100^2 + y^2 = (3 * y - 10)^2) :
  round (area_of_triangle_ADC y h₁) = 134 :=
sorry

end triangle_ADC_area_l625_625273


namespace find_initial_apples_l625_625386

def initial_apples (a b c : ℕ) : Prop :=
  b + c = a

theorem find_initial_apples (a b initial_apples : ℕ) (h : b + initial_apples = a) : initial_apples = 8 :=
by
  sorry

end find_initial_apples_l625_625386


namespace smallest_positive_perfect_square_div_by_2_3_5_l625_625432

-- Definition capturing the conditions of the problem
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def divisible_by (n m : ℕ) : Prop :=
  m % n = 0

-- The theorem statement combining all conditions and requiring the proof of the correct answer
theorem smallest_positive_perfect_square_div_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ is_perfect_square n ∧ divisible_by 2 n ∧ divisible_by 3 n ∧ divisible_by 5 n ∧ n = 900 :=
sorry

end smallest_positive_perfect_square_div_by_2_3_5_l625_625432


namespace pretzels_eaten_properly_l625_625803

noncomputable def initial_pretzels : ℕ := 30

theorem pretzels_eaten_properly :
  ∃ x : ℕ, x = 30 ∧
  (let after_hatter := x / 2 - 1 in
   let after_hare := after_hatter / 2 - 1 in
   let after_dormouse := after_hare / 2 - 1 in
   let after_cat := after_dormouse / 2 - 1 in
   after_cat = 0) :=
begin
  use 30,
  split,
  { exact rfl },
  { sorry }
end

end pretzels_eaten_properly_l625_625803


namespace quarters_needed_l625_625521

-- Define the cost of items in cents and declare the number of items to purchase.
def quarter_value : ℕ := 25
def candy_bar_cost : ℕ := 25
def chocolate_cost : ℕ := 75
def juice_cost : ℕ := 50

def num_candy_bars : ℕ := 3
def num_chocolates : ℕ := 2
def num_juice_packs : ℕ := 1

-- Theorem stating the number of quarters needed to buy the given items.
theorem quarters_needed : 
  (num_candy_bars * candy_bar_cost + num_chocolates * chocolate_cost + num_juice_packs * juice_cost) / quarter_value = 11 := 
sorry

end quarters_needed_l625_625521


namespace number_of_integers_divisible_by_neither_6_nor_8_l625_625374

def less_than_1200 (n : ℕ) := n < 1200
def divisible_by_6 (n : ℕ) := n % 6 = 0
def divisible_by_8 (n : ℕ) := n % 8 = 0
def divisible_by_24 (n : ℕ) := n % 24 = 0

theorem number_of_integers_divisible_by_neither_6_nor_8 : 
  (∑ n in finset.Ico 1 1200, if ¬(divisible_by_6 n ∨ divisible_by_8 n) then 1 else 0) = 900 :=
by
  sorry

end number_of_integers_divisible_by_neither_6_nor_8_l625_625374


namespace coeff_x10_in_binomial_expansion_l625_625043

theorem coeff_x10_in_binomial_expansion : 
  (polynomial.mk (λ n, if n = 10 then (algebra.algebra_map ℚ) (-11) else (algebra.algebra_map ℚ) 0)) = 
  (polynomial.expand (algebra.algebra_map ℚ) (x - 1) ^ 11) :=
by
  sorry

end coeff_x10_in_binomial_expansion_l625_625043


namespace vasya_incorrect_answers_l625_625118

def anya_answers : Array Bool := #[false, true, false, true, false, true]
def borya_answers : Array Bool := #[false, false, true, true, false, true]
def vasya_answers : Array Bool := #[true, false, false, false, true, false]
def correct_answers : Array Bool := #[true, true, true, true, true, true]

theorem vasya_incorrect_answers :
  (count (fun i => anya_answers[i] ≠ correct_answers[i]) 6 = 2) →
  (count (fun i => borya_answers[i] = correct_answers[i]) 6 = 2) →
  (count (fun i => vasya_answers[i] ≠ correct_answers[i]) 6 = 4) :=
by
  sorry

end vasya_incorrect_answers_l625_625118


namespace trapezoid_area_is_correct_l625_625986

def square_side_lengths : List ℕ := [1, 3, 5, 7]
def total_base_length : ℕ := square_side_lengths.sum
def tallest_square_height : ℕ := 7

noncomputable def trapezoid_area_between_segment_and_base : ℚ :=
  let height_at_x (x : ℚ) : ℚ := x * (7/16)
  let base_1 := 4
  let base_2 := 9
  let height_1 := height_at_x base_1
  let height_2 := height_at_x base_2
  ((height_1 + height_2) * (base_2 - base_1) / 2)

theorem trapezoid_area_is_correct :
  trapezoid_area_between_segment_and_base = 14.21875 :=
sorry

end trapezoid_area_is_correct_l625_625986


namespace complete_square_l625_625796

theorem complete_square 
  (x : ℝ) : 
  (2 * x^2 - 3 * x - 1 = 0) → 
  ((x - (3/4))^2 = (17/16)) :=
sorry

end complete_square_l625_625796


namespace max_real_part_sum_l625_625757

theorem max_real_part_sum :
  let z : ℂ → Prop := λ z, ∃ k : ℕ, k < 10 ∧ z = 8 * exp (2 * π * (k : ℂ) / 10 * I)
  let w : ℕ → ℂ := λ j, if j % 4 == 0 then 8 * exp (2 * π * (j : ℂ) / 10 * I) else
                            if j % 4 == 2 then 8 * I * exp (2 * π * (j : ℂ) / 10 * I) else
                            if j % 4 == 4 then 8 * exp (2 * π * (j : ℂ) / 10 * I) else
                            8 * I * exp (2 * π * (j : ℂ) / 10 * I)
  in Re (∑ j in finset.range 10, w j) = 9.2656 :=
sorry

end max_real_part_sum_l625_625757


namespace sum_of_second_and_third_smallest_l625_625024

theorem sum_of_second_and_third_smallest (a b c : ℕ) (h₀ : List.sorted (≤) [a, b, c]) (h₁ : a = 10) (h₂ : b = 11) (h₃ : c = 12) :
  a + c = 23 :=
by
  sorry

end sum_of_second_and_third_smallest_l625_625024


namespace least_common_multiple_of_20_45_75_l625_625871

theorem least_common_multiple_of_20_45_75 :
  Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
sorry

end least_common_multiple_of_20_45_75_l625_625871


namespace eccentricity_hyperbola_proof_l625_625198

noncomputable def eccentricity_of_hyperbola (a b: ℝ) (ha: 0 < a) (hb: 0 < b) 
(f1 f2 m n: (ℝ × ℝ)) 
(hf1: f1 = (-c, 0)) 
(hf2: f2 = (c, 0))
(hMN: ∃ k: ℝ, (k * 3) = 1 ∧ k * (n - m) = (f2 - f1)) 
(horthogonal : ((m - f1) * (n - f2)).fst = 0) : 
    c : ℝ :=
    let c := sqrt (a^2 + b^2)
    in 
    sqrt (c^2 / a^2)

theorem eccentricity_hyperbola_proof (a b c: ℝ) (ha: 0 < a) (hb: 0 < b) 
(f1 f2 m n: (ℝ × ℝ)) 
(hf1: f1 = (-c, 0)) 
(hf2: f2 = (c, 0))
(horth: ((m.1 + c)^2 + m.2^2 = b^2 + c^2 ∧ (n.1 - c)^2 + n.2^2 = b^2 + c^2)
    : ℝ := by 
    have hc : c = sqrt (a^2 + b^2) := sorry
    have e : c / a := sorry
    exact sqrt (5) + sqrt (2)

end eccentricity_hyperbola_proof_l625_625198


namespace first_player_wins_always_l625_625914

structure Chessboard :=
  (rows : ℕ)
  (columns : ℕ)
  (king_position : ℕ × ℕ)

def initial_king_position : Chessboard := 
  { rows := 8, columns := 8, king_position := (1, 1) }

def top_right_corner_position := (8, 8)

def optimal_move (current_pos : ℕ × ℕ) : ℕ × ℕ :=
  (current_pos.1 + 1, current_pos.2 + 1) -- diagonal move

noncomputable def player_first_wins (current_pos : ℕ × ℕ) : Prop := 
  if current_pos = top_right_corner_position 
  then true 
  else sorry -- This represents the strategy to always lead to win for the first player

theorem first_player_wins_always : player_first_wins initial_king_position :=
sorry

end first_player_wins_always_l625_625914


namespace geometric_sequence_first_term_l625_625004

theorem geometric_sequence_first_term (a r : ℝ) (h1 : a * r^4 = (9! : ℝ)) (h2 : a * r^7 = (11! : ℝ)) :
    a = (9! : ℝ) / (r^4) :=
by
  sorry

end geometric_sequence_first_term_l625_625004


namespace exists_group_of_four_l625_625712

-- Assuming 21 students, and any three have done homework together exactly once in either mathematics or Russian.
-- We aim to prove there exists a group of four students such that any three of them have done homework together in the same subject.
noncomputable def students : Type := Fin 21

-- Define a predicate to show that three students have done homework together.
-- We use "math" and "russian" to denote the subjects.
inductive Subject
| math
| russian

-- Define a relation expressing that any three students have done exactly one subject homework together.
axiom homework_done (s1 s2 s3 : students) : Subject 

theorem exists_group_of_four :
  ∃ (a b c d : students), 
    (homework_done a b c = homework_done a b d) ∧
    (homework_done a b c = homework_done a c d) ∧
    (homework_done a b c = homework_done b c d) ∧
    (homework_done a b d = homework_done a c d) ∧
    (homework_done a b d = homework_done b c d) ∧
    (homework_done a c d = homework_done b c d) :=
sorry

end exists_group_of_four_l625_625712


namespace math_problem_l625_625689

theorem math_problem (a b : ℝ) (h : Real.sqrt (a + 2) + |b - 1| = 0) : (a + b) ^ 2023 = -1 := 
by
  sorry

end math_problem_l625_625689


namespace majority_votes_calculation_l625_625717

def candidate_share := 0.60
def total_votes := 6500
def majority_votes := 1300

theorem majority_votes_calculation :
  (candidate_share * total_votes - (total_votes - candidate_share * total_votes)) = majority_votes :=
by
  sorry

end majority_votes_calculation_l625_625717


namespace solve_ineq_system_l625_625350

theorem solve_ineq_system (x : ℝ) :
  (x - 1) / (x + 2) ≤ 0 ∧ x^2 - 2 * x - 3 < 0 ↔ -1 < x ∧ x ≤ 1 :=
by sorry

end solve_ineq_system_l625_625350


namespace ben_current_age_l625_625808

theorem ben_current_age (a b c : ℕ) 
  (h1 : a + b + c = 36) 
  (h2 : c = 2 * a - 4) 
  (h3 : b + 5 = 3 * (a + 5) / 4) : 
  b = 5 := 
by
  sorry

end ben_current_age_l625_625808


namespace equilateral_triangle_centers_and_sides_l625_625493

variable {A B C A' B' C' P Q R : Type}

-- Noncomputable to define geometric properties if necessary
noncomputable def is_equilateral (T : Triangle) : Prop :=
  -- Assuming T is a type representing a triangle
  ∃ a b c : Point, T = triangle a b c ∧ dist a b = dist b c ∧ dist b c = dist c a

-- Define the relevant points and their relationships
def outward_equilateral (A B C A' B' C' : Point)
                        (P Q R : Point) : Prop :=
  is_equilateral (triangle A B A') ∧
  is_equilateral (triangle A C C') ∧
  is_equilateral (triangle B C B') ∧
  centroid (triangle A B A') = P ∧
  centroid (triangle A C C') = Q ∧
  centroid (triangle B C B') = R

-- The proof statements
theorem equilateral_triangle_centers_and_sides (A B C A' B' C' : Point) (P Q R : Point)
  (h : outward_equilateral A B C A' B' C' P Q R) :
  distance A A' = distance B B' ∧ distance B B' = distance C C' ∧
  distance P Q = distance Q R ∧ distance Q R = distance R P :=
sorry

end equilateral_triangle_centers_and_sides_l625_625493


namespace no_valid_digit_replacement_l625_625900

theorem no_valid_digit_replacement :
  ¬ ∃ (A B C D E M X : ℕ),
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ M ∧ A ≠ X ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ M ∧ B ≠ X ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ M ∧ C ≠ X ∧
     D ≠ E ∧ D ≠ M ∧ D ≠ X ∧
     E ≠ M ∧ E ≠ X ∧
     M ≠ X ∧
     0 ≤ A ∧ A < 10 ∧
     0 ≤ B ∧ B < 10 ∧
     0 ≤ C ∧ C < 10 ∧
     0 ≤ D ∧ D < 10 ∧
     0 ≤ E ∧ E < 10 ∧
     0 ≤ M ∧ M < 10 ∧
     0 ≤ X ∧ X < 10 ∧
     A * B * C * D + 1 = C * E * M * X) :=
sorry

end no_valid_digit_replacement_l625_625900


namespace fraction_of_surface_area_is_red_l625_625504

structure Cube :=
  (edge_length : ℕ)
  (small_cubes : ℕ)
  (num_red_cubes : ℕ)
  (num_blue_cubes : ℕ)
  (blue_cube_edge_length : ℕ)
  (red_outer_layer : ℕ)

def surface_area (c : Cube) : ℕ := 6 * (c.edge_length * c.edge_length)

theorem fraction_of_surface_area_is_red (c : Cube) 
  (h_edge_length : c.edge_length = 4)
  (h_small_cubes : c.small_cubes = 64)
  (h_num_red_cubes : c.num_red_cubes = 40)
  (h_num_blue_cubes : c.num_blue_cubes = 24)
  (h_blue_cube_edge_length : c.blue_cube_edge_length = 2)
  (h_red_outer_layer : c.red_outer_layer = 1)
  : (surface_area c) / (surface_area c) = 1 := 
by
  sorry

end fraction_of_surface_area_is_red_l625_625504


namespace sequence_x_is_32_l625_625269

theorem sequence_x_is_32 :
  ∃ x : ℕ, (∀ i : ℕ, i < 5 → diffs !i = [3, 6, 9, 12, 15] !i) →
  (nth_term 0 = 2 ∧ nth_term 1 = 5 ∧ nth_term 2 = 11 ∧ nth_term 3 = 20 ∧ nth_term 5 = 47) →
  nth_term 4 = 32 :=
by
  sorry

-- Definitions and conditions as per the problem
def diffs : list ℕ := [3, 6, 9, 12, 15]

def nth_term (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 5
  | 2 => 11
  | 3 => 20
  | 4 => 20 + diffs.head
  | 5 => 47
  | _ => 0  -- Unsure outside provided terms, matching needed for other terms

end sequence_x_is_32_l625_625269


namespace smallest_positive_perfect_square_div_by_2_3_5_l625_625428

-- Definition capturing the conditions of the problem
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def divisible_by (n m : ℕ) : Prop :=
  m % n = 0

-- The theorem statement combining all conditions and requiring the proof of the correct answer
theorem smallest_positive_perfect_square_div_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ is_perfect_square n ∧ divisible_by 2 n ∧ divisible_by 3 n ∧ divisible_by 5 n ∧ n = 900 :=
sorry

end smallest_positive_perfect_square_div_by_2_3_5_l625_625428


namespace unique_solution_arith_prog_system_l625_625840

theorem unique_solution_arith_prog_system (x y : ℝ) : 
  (6 * x + 9 * y = 12) ∧ (15 * x + 18 * y = 21) ↔ (x = -1) ∧ (y = 2) :=
by sorry

end unique_solution_arith_prog_system_l625_625840


namespace unique_prime_solution_l625_625971

-- Define the variables and properties
def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the proof goal
theorem unique_prime_solution (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (hp_pos : 0 < p) (hq_pos : 0 < q) :
  p^2 - q^3 = 1 → (p = 3 ∧ q = 2) :=
by sorry

end unique_prime_solution_l625_625971


namespace bridge_length_l625_625086

-- Definitions based on the provided conditions.
def speed_km_per_hr : ℝ := 5
def time_min : ℝ := 15

-- Conversion constants
def km_to_m : ℝ := 1000
def hr_to_min : ℝ := 60

-- Calculate the speed in m/min.
def speed_m_per_min : ℝ := (speed_km_per_hr * km_to_m) / hr_to_min

-- Calculate the distance covered in 15 minutes.
def distance : ℝ := speed_m_per_min * time_min

-- The goal is to prove that the distance is approximately 1250 meters.
theorem bridge_length : distance ≈ 1250 := 
by
  sorry

end bridge_length_l625_625086


namespace total_surface_area_hemisphere_cylinder_l625_625083

noncomputable def total_surface_area (r h : ℝ) : ℝ :=
  let area_cylinder_top := π * r^2
  let area_hemisphere := 2 * π * r^2 -- 1/2 of the surface area of a sphere is 2πr²
  let area_cylinder_side := 2 * π * r * h
  area_cylinder_top + area_hemisphere + area_cylinder_side

theorem total_surface_area_hemisphere_cylinder :
  total_surface_area 10 1 = 320 * π :=
by
  sorry

end total_surface_area_hemisphere_cylinder_l625_625083


namespace ratio_proof_l625_625277

variable {A B C D E K M N : Point}
variable {BD DC BE ME BN NK : Real}

-- Conditions
axiom h1 : B ∈ Segment(C, D) 
axiom h2 : BD = 1 * DC / 3 
axiom h3 : E ∈ Segment(A, C)
axiom h4 : K ∈ Segment(A, C)
axiom h5 : E ∈ Segment(A, K)
axiom h6 : M ∈ Segment(B, E)
axiom h7 : N ∈ Segment(B, K)
axiom h8 : Segment(A, D) ∩ Segment(B, E) = {M}
axiom h9 : Segment(A, D) ∩ Segment(B, K) = {N}
axiom h10 : BM = 7/12 * BE
axiom h11 : BN = 2/5 * BK

theorem ratio_proof : MN / AD = 11 / 45 := sorry

end ratio_proof_l625_625277


namespace lcm_20_45_75_l625_625882

def lcm (a b : ℕ) : ℕ := nat.lcm a b

theorem lcm_20_45_75 : lcm (lcm 20 45) 75 = 900 :=
by
  sorry

end lcm_20_45_75_l625_625882


namespace original_dining_bill_l625_625082

theorem original_dining_bill :
  ∃ B : ℝ, (B > 0) ∧ (B + 0.15 * B) / 10 ≈ 24.265 :=
begin
  use 211,
  split,
  { linarith },
  { simp, norm_num }
end

end original_dining_bill_l625_625082


namespace length_of_grassy_plot_correct_l625_625097

noncomputable def getLengthOfGrassyPlot (width path_width: ℝ) (cost total_cost: ℝ) : ℝ :=
  let adjusted_length (L: ℝ) : ℝ := (total_cost / (cost * (width - 2 * path_width))) + 2 * path_width
  adjusted_length

theorem length_of_grassy_plot_correct :
  getLengthOfGrassyPlot 70 2.5 0.90 742.5 = 17.7 := 
by
  sorry

end length_of_grassy_plot_correct_l625_625097


namespace decreasing_interval_l625_625202

theorem decreasing_interval
  (a b : ℝ)
  (h_extremum : deriv (λ x : ℝ, x^2 * (a * x + b)) 2 = 0)
  (h_tangent_parallel : deriv (λ x : ℝ, x^2 * (a * x + b)) 1 = -3) :
  {x : ℝ | 0 < x ∧ x < 2} = {x : ℝ | x ∈ set.Ioo 0 2} := 
by
  sorry

end decreasing_interval_l625_625202


namespace prove_primes_l625_625170

noncomputable def is_prime (p : ℕ) : Prop := p.prime

theorem prove_primes (n : ℕ) :
  (n ≥ 2) →
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ Nat.sqrt (n / 3) → is_prime (k^2 + k + n)) →
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → is_prime (k^2 + k + n)) :=
by
  intros h1 h2
  sorry

end prove_primes_l625_625170


namespace four_numbers_sum_divisible_by_2016_l625_625490

theorem four_numbers_sum_divisible_by_2016 {x : Fin 65 → ℕ} (h_distinct: Function.Injective x) (h_range: ∀ i, x i ≤ 2016) :
  ∃ a b c d : Fin 65, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (x a + x b - x c - x d) % 2016 = 0 :=
by
  -- Proof omitted
  sorry

end four_numbers_sum_divisible_by_2016_l625_625490


namespace well_digging_rate_l625_625615

noncomputable def rate_per_cubic_meter (depth : ℝ) (radius : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (Real.pi * radius^2 * depth)

theorem well_digging_rate : 
  rate_per_cubic_meter 14 1.5 1781.28 ≈ 17.99 :=
by
  sorry

end well_digging_rate_l625_625615


namespace not_collinear_bc_l625_625296

open Locale.VectorSpace

variables {α : Type*} [Field α] [VectorSpace α] 
variables (a b c : α → α) -- Assuming vectors as functions from α to α for simplicity
variables (k : α) (h_a_non_zero : a ≠ 0) (h_b_non_zero : b ≠ 0) (h_c_non_zero : c ≠ 0)

-- stating the non-collinearity of a and b
def not_collinear (u v : α → α) : Prop :=
  ∀ k : α, u ≠ k • v

-- stating the collinearity of a and c
def collinear (u v : α → α) : Prop :=
  ∃ k : α, v = k • u

-- The theorem to be proved
theorem not_collinear_bc (h_not_collinear_ab : not_collinear a b) (h_collinear_ac : collinear a c) :
  not_collinear b c :=
sorry

end not_collinear_bc_l625_625296


namespace polar_to_cartesian_line_param_to_general_max_distance_from_M_l625_625268

-- Define the line parametric equations
def line_param_eqn (t : ℝ) : ℝ × ℝ := (8 + 4 * t, 1 - t)

-- Define the polar equation of the curve C
def polar_eqn (ρ θ : ℝ) : Prop := ρ^2 * (5 - 4 * Real.cos (2 * θ)) = 9

-- Define the Cartesian equation of the curve C
def cartesian_eqn (x y : ℝ) : Prop := x^2 + y^2 + 8 * y^2 = 9

-- Define the general equation of the line derived from its parametric equations
def general_eqn_of_line (x y : ℝ) : Prop := x + 4 * y - 12 = 0

-- Define the calculation of distance from point M on curve C to the line
def distance_from_M_to_l (x y : ℝ) : ℝ :=
  abs (x + 4 * y - 12) / Real.sqrt 17

theorem polar_to_cartesian (ρ θ : ℝ) (h1 : polar_eqn ρ θ) :
  ∃ (x y : ℝ), cartesian_eqn x y :=
sorry

theorem line_param_to_general :
  ∃ t : ℝ, general_eqn_of_line (8 + 4 * t) (1 - t) :=
sorry

theorem max_distance_from_M (θ : ℝ) :
  ∃ (x y : ℝ), distance_from_M_to_l x y = Real.sqrt 17 :=
sorry

end polar_to_cartesian_line_param_to_general_max_distance_from_M_l625_625268


namespace daughter_current_age_l625_625941

-- Define the conditions
def mother_current_age := 42
def years_later := 9
def mother_age_in_9_years := mother_current_age + years_later
def daughter_age_in_9_years (D : ℕ) := D + years_later

-- Define the statement we need to prove
theorem daughter_current_age : ∃ D : ℕ, mother_age_in_9_years = 3 * daughter_age_in_9_years D ∧ D = 8 :=
by {
  sorry
}

end daughter_current_age_l625_625941


namespace cone_volume_correct_l625_625376

-- Define the given conditions
def radius_sector : ℝ := 3
def angle_sector_rad : ℝ := 2 * Real.pi / 3

-- Base circumference equality
def base_circumference (circumference : ℝ) : Prop :=
  circumference = 2 * Real.pi

-- Radius of the base equality
def radius_base (radius : ℝ) : Prop :=
  radius = 1
  
-- Height of the cone equality
def height_cone (height : ℝ) : Prop :=
  height = Real.sqrt (radius_sector ^ 2 - (radius_base 1).choose ^ 2)

-- Volume of the cone
def volume_cone (volume : ℝ) : Prop :=
  volume = (1 / 3) * Real.pi * (radius_base 1).choose ^ 2 * (height_cone (Real.sqrt (radius_sector ^ 2 - (radius_base 1).choose ^ 2)).choose)

-- Main theorem statement
theorem cone_volume_correct :
  volume_cone (2 * Real.sqrt 2 / 3 * Real.pi) :=
sorry

end cone_volume_correct_l625_625376


namespace angle_at_two_oclock_l625_625120

def minute_hand_angle (time : Nat) : ℝ :=
  0

def hour_hand_angle (time : Nat) : ℝ :=
  (time / 12) * 360

def angle_between_hands (time : Nat) : ℝ :=
  abs (hour_hand_angle time - minute_hand_angle time)

theorem angle_at_two_oclock :
  angle_between_hands 2 = 60 :=
sorry

end angle_at_two_oclock_l625_625120


namespace bobby_pancakes_left_l625_625126

theorem bobby_pancakes_left (initial_pancakes : ℕ) (bobby_ate : ℕ) (dog_ate : ℕ) :
  initial_pancakes = 21 → bobby_ate = 5 → dog_ate = 7 → initial_pancakes - (bobby_ate + dog_ate) = 9 :=
by
  intros h1 h2 h3
  sorry

end bobby_pancakes_left_l625_625126


namespace negation_of_universal_statement_l625_625620

theorem negation_of_universal_statement :
  ¬ (∀ x : ℝ, x^3 - 3 * x > 0) ↔ ∃ x : ℝ, x^3 - 3 * x ≤ 0 :=
by
  sorry

end negation_of_universal_statement_l625_625620


namespace triangle_area_inscribed_circle_l625_625532

theorem triangle_area_inscribed_circle
  (x : ℝ)
  (r : ℝ)
  (h_radius : r = 4)
  (h_sides_ratio : 2^2 + 3^2 = 13) 
  (h_hypotenuse : x * real.sqrt 13 = 2 * r) :
  (1/2) * (2 * (8 / real.sqrt 13)) * (3 * (8 / real.sqrt 13)) = 384 / 13 :=
by
  sorry

end triangle_area_inscribed_circle_l625_625532


namespace sqrt6_special_op_l625_625704

-- Define the binary operation (¤) as given in the problem.
def special_op (x y : ℝ) : ℝ := (x + y) ^ 2 - (x - y) ^ 2

-- States that √6 ¤ √6 is equal to 24.
theorem sqrt6_special_op : special_op (Real.sqrt 6) (Real.sqrt 6) = 24 :=
by
  sorry

end sqrt6_special_op_l625_625704


namespace neg_and_implication_l625_625251

variable (p q : Prop)

theorem neg_and_implication : ¬ (p ∧ q) → ¬ p ∨ ¬ q := by
  sorry

end neg_and_implication_l625_625251


namespace min_students_in_tournament_l625_625270

theorem min_students_in_tournament :
  ∀ (problems students : ℕ) (solve_problem : ℕ → students → Prop),
  problems = 6 →
  (∀ p : ℕ, p < problems → ∃ S : Finset students, S.card = 1000 ∧ ∀ x ∈ S, solve_problem p x) →
  (∀ x y : students, x ≠ y → (∀ p : ℕ, p < problems → (solve_problem p x ↔ solve_problem p y) → false)) →
  students ≥ 2000 :=
by
  intros problems students solve_problem h_problems h_solve h_no_two_solve_all
  -- Here is the proof placeholder, indicating the formal proof should be written.
  sorry

end min_students_in_tournament_l625_625270


namespace sequence_general_formula_l625_625179

theorem sequence_general_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) 
  (hS : ∀ n, S n = 3 / 2 * a n - 3) : 
  (∀ n, a n = 2 * 3 ^ n) :=
by 
  sorry

end sequence_general_formula_l625_625179


namespace notable_features_points_l625_625325

namespace Points3D

def is_first_octant (x y z : ℝ) : Prop := (x > 0) ∧ (y > 0) ∧ (z > 0)
def is_second_octant (x y z : ℝ) : Prop := (x < 0) ∧ (y > 0) ∧ (z > 0)
def is_eighth_octant (x y z : ℝ) : Prop := (x > 0) ∧ (y < 0) ∧ (z < 0)
def lies_in_YOZ_plane (x y z : ℝ) : Prop := (x = 0) ∧ (y ≠ 0) ∧ (z ≠ 0)
def lies_on_OY_axis (x y z : ℝ) : Prop := (x = 0) ∧ (y ≠ 0) ∧ (z = 0)
def is_origin (x y z : ℝ) : Prop := (x = 0) ∧ (y = 0) ∧ (z = 0)

theorem notable_features_points :
  is_first_octant 3 2 6 ∧
  is_second_octant (-2) 3 1 ∧
  is_eighth_octant 1 (-4) (-2) ∧
  is_eighth_octant 1 (-2) (-1) ∧
  lies_in_YOZ_plane 0 4 1 ∧
  lies_on_OY_axis 0 2 0 ∧
  is_origin 0 0 0 :=
by
  sorry

end Points3D

end notable_features_points_l625_625325


namespace track_length_l625_625355

theorem track_length
  (weekly_meters : ℕ)
  (days_per_week : ℕ)
  (loops_per_day : ℕ)
  (weekly_meters = 3500)
  (days_per_week = 7)
  (loops_per_day = 10) :
  (3500 / 7 / 10 = 50) :=
by
  sorry

end track_length_l625_625355


namespace statues_added_in_third_year_l625_625222

/-
Definition of the turtle statues problem:

1. Initially, there are 4 statues in the first year.
2. In the second year, the number of statues quadruples.
3. In the third year, x statues are added, and then 3 statues are broken.
4. In the fourth year, 2 * 3 new statues are added.
5. In total, at the end of the fourth year, there are 31 statues.
-/

def year1_statues : ℕ := 4
def year2_statues : ℕ := 4 * year1_statues
def before_hailstorm_year3_statues (x : ℕ) : ℕ := year2_statues + x
def after_hailstorm_year3_statues (x : ℕ) : ℕ := before_hailstorm_year3_statues x - 3
def total_year4_statues (x : ℕ) : ℕ := after_hailstorm_year3_statues x + 2 * 3

theorem statues_added_in_third_year (x : ℕ) (h : total_year4_statues x = 31) : x = 12 :=
by
  sorry

end statues_added_in_third_year_l625_625222


namespace smallest_perfect_square_divisible_by_2_3_5_l625_625465

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, (n > 0) ∧ (nat.sqrt n * nat.sqrt n = n) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ n = 900 := sorry

end smallest_perfect_square_divisible_by_2_3_5_l625_625465


namespace lcm_of_20_45_75_l625_625862

-- Definitions for the given numbers and their prime factorizations
def num1 : ℕ := 20
def num2 : ℕ := 45
def num3 : ℕ := 75

def factor1 : ℕ → Prop := λ n, n = 2 ^ 2 * 5
def factor2 : ℕ → Prop := λ n, n = 3 ^ 2 * 5
def factor3 : ℕ → Prop := λ n, n = 3 * 5 ^ 2

-- The condition of using the least common multiple function from mathlib
def lcm_def (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- The statement to prove
theorem lcm_of_20_45_75 : lcm_def num1 num2 num3 = 900 := by
  -- Factors condition (Note: These help ensure the numbers' factors are as stated)
  have h1 : factor1 num1 := by { unfold num1 factor1, exact rfl }
  have h2 : factor2 num2 := by { unfold num2 factor2, exact rfl }
  have h3 : factor3 num3 := by { unfold num3 factor3, exact rfl }
  sorry -- This is the place where the proof would go.

end lcm_of_20_45_75_l625_625862


namespace stock_price_problem_l625_625375

theorem stock_price_problem :
  let P := 1000 in
  let P1 := P + 0.30 * P in
  let P2 := P1 - 0.10 * P1 in
  let P3 := P2 + 0.20 * P2 in
  let y := 29 in -- Based on the given correct answer
  P3 - y / 100 * P3 = P :=
  by sorry

end stock_price_problem_l625_625375


namespace breadth_of_hall_l625_625923

/-- Given a hall of length 20 meters and a uniform verandah width of 2.5 meters,
    with a cost of Rs. 700 for flooring the verandah at Rs. 3.50 per square meter,
    prove that the breadth of the hall is 15 meters. -/
theorem breadth_of_hall (h_length : ℝ) (v_width : ℝ) (cost : ℝ) (rate : ℝ) (b : ℝ) :
  h_length = 20 ∧ v_width = 2.5 ∧ cost = 700 ∧ rate = 3.50 →
  25 * (b + 5) - 20 * b = 200 →
  b = 15 :=
by
  intros hc ha
  sorry

end breadth_of_hall_l625_625923


namespace smallest_positive_perfect_square_div_by_2_3_5_l625_625431

-- Definition capturing the conditions of the problem
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def divisible_by (n m : ℕ) : Prop :=
  m % n = 0

-- The theorem statement combining all conditions and requiring the proof of the correct answer
theorem smallest_positive_perfect_square_div_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ is_perfect_square n ∧ divisible_by 2 n ∧ divisible_by 3 n ∧ divisible_by 5 n ∧ n = 900 :=
sorry

end smallest_positive_perfect_square_div_by_2_3_5_l625_625431


namespace compare_abc_l625_625633

noncomputable def a : ℝ :=
  (1/2) * Real.cos 16 - (Real.sqrt 3 / 2) * Real.sin 16

noncomputable def b : ℝ :=
  2 * Real.tan 14 / (1 + (Real.tan 14) ^ 2)

noncomputable def c : ℝ :=
  Real.sqrt ((1 - Real.cos 50) / 2)

theorem compare_abc : b > c ∧ c > a :=
  by sorry

end compare_abc_l625_625633


namespace reciprocal_sub_fraction_l625_625048

theorem reciprocal_sub_fraction :
  (1 / 3 - 1 / 4)⁻¹ = 12 := 
sorry

end reciprocal_sub_fraction_l625_625048


namespace lambda_range_l625_625178

theorem lambda_range (a : ℕ → ℝ) (λ : ℝ) (h1 : a 1 = 1) (h2 : a 2 = 2)
  (h3 : ∀ n : ℕ, n > 0 → n * a (n + 2) - (n + 2) * a n = λ * (n^2 + 2 * n))
  (h4 : ∀ n : ℕ, n > 0 → a n < a (n + 1)) :
  0 ≤ λ :=
begin
  sorry
end

end lambda_range_l625_625178


namespace worker_c_painting_time_l625_625482

theorem worker_c_painting_time (ha : ℝ) (hb : ℝ) (hc : ℝ) :
  ha = 20 → hb = 15 → 
  (let wa := 1 / ha in
   let wb := 1 / hb in
   let work_completed := 6 * (wa + wb) in
   let remaining_work := 1 - work_completed in
   let wc := 1 / hc in
   let work_ac := wa + wc in
   5 * work_ac = remaining_work) →
  hc = 100 :=
by 
  intros h1 h2 h3
  sorry

end worker_c_painting_time_l625_625482


namespace solution_inequality_l625_625255

-- Conditions
variables {a b x : ℝ}
theorem solution_inequality (h1 : a < 0) (h2 : b = a) :
  {x : ℝ | (ax + b) ≤ 0} = {x : ℝ | x ≥ -1} →
  {x : ℝ | (ax + b) / (x - 2) > 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end solution_inequality_l625_625255


namespace valid_tuples_count_l625_625292

def count_valid_tuples : ℕ := by
  exact 300 -- Based on our correct answer derived from mathematical analysis

theorem valid_tuples_count :
  (∃ (b_3 b_2 b_1 b_0 : ℕ), 
    2023 = b_3 * 10^3 + b_2 * 10^2 + b_1 * 10 + b_0 ∧ 
    0 ≤ b_0 ∧ b_0 ≤ 99 ∧ 
    0 ≤ b_1 ∧ b_1 ≤ 99 ∧
    0 ≤ b_2 ∧ b_2 ≤ 99 ∧
    0 ≤ b_3 ∧ b_3 ≤ 99) → 
  count_valid_tuples = 300 := 
begin 
  sorry
end

end valid_tuples_count_l625_625292


namespace cube_cross_section_area_l625_625096

def cube_edge_length (a : ℝ) := a > 0

def plane_perpendicular_body_diagonal := 
  ∃ (p : ℝ × ℝ × ℝ), ∀ (x y z : ℝ), 
  p = (x / 2, y / 2, z / 2) ∧ 
  (x + y + z) = (1 : ℝ)

theorem cube_cross_section_area
  (a : ℝ) 
  (h : cube_edge_length a) 
  (plane : plane_perpendicular_body_diagonal) : 
  ∃ (A : ℝ), 
  A = (3 * a^2 * Real.sqrt 3 / 4) := sorry

end cube_cross_section_area_l625_625096


namespace average_temps_l625_625124

-- Define the temperature lists
def temps_C : List ℚ := [
  37.3, 37.2, 36.9, -- Sunday
  36.6, 36.9, 37.1, -- Monday
  37.1, 37.3, 37.2, -- Tuesday
  36.8, 37.3, 37.5, -- Wednesday
  37.1, 37.7, 37.3, -- Thursday
  37.5, 37.4, 36.9, -- Friday
  36.9, 37.0, 37.1  -- Saturday
]

def temps_K : List ℚ := [
  310.4, 310.3, 310.0, -- Sunday
  309.8, 310.0, 310.2, -- Monday
  310.2, 310.4, 310.3, -- Tuesday
  309.9, 310.4, 310.6, -- Wednesday
  310.2, 310.8, 310.4, -- Thursday
  310.6, 310.5, 310.0, -- Friday
  310.0, 310.1, 310.2  -- Saturday
]

def temps_R : List ℚ := [
  558.7, 558.6, 558.1, -- Sunday
  557.7, 558.1, 558.3, -- Monday
  558.3, 558.7, 558.6, -- Tuesday
  558.0, 558.7, 559.1, -- Wednesday
  558.3, 559.4, 558.7, -- Thursday
  559.1, 558.9, 558.1, -- Friday
  558.1, 558.2, 558.3  -- Saturday
]

-- Calculate the average of a list of temperatures
def average (temps : List ℚ) : ℚ :=
  temps.sum / temps.length

-- Define the average temperatures
def avg_C := average temps_C
def avg_K := average temps_K
def avg_R := average temps_R

-- State that the computed averages are equal to the provided values
theorem average_temps :
  avg_C = 37.1143 ∧
  avg_K = 310.1619 ∧
  avg_R = 558.2524 :=
by
  -- Proof can be completed here
  sorry

end average_temps_l625_625124


namespace total_animals_in_farm_l625_625121

theorem total_animals_in_farm (C B : ℕ) (h1 : C = 5) (h2 : 2 * C + 4 * B = 26) : C + B = 9 :=
by
  sorry

end total_animals_in_farm_l625_625121


namespace james_points_per_correct_answer_l625_625351

noncomputable def points_per_correct_answer (Q M T B R : ℕ) (points : ℕ) : Prop :=
  Q = 25 ∧ M = 1 ∧ T = 66 ∧ B = 4 ∧ R = 5 ∧
  let total_correct := Q - M in
  let rounds_with_bonus := R - 1 in
  let bonus_points := R * B - B in
  let total_points_without_bonus := T - bonus_points in
  total_points_without_bonus / total_correct = points

theorem james_points_per_correct_answer
  : points_per_correct_answer 25 1 66 4 5 2 :=
by
  unfold points_per_correct_answer
  split
  sorry

end james_points_per_correct_answer_l625_625351


namespace male_students_in_grade_l625_625528

-- Define the total number of students and the number of students in the sample
def total_students : ℕ := 1200
def sample_students : ℕ := 30

-- Define the number of female students in the sample
def female_students_sample : ℕ := 14

-- Calculate the number of male students in the sample
def male_students_sample := sample_students - female_students_sample

-- State the main theorem
theorem male_students_in_grade :
  (male_students_sample : ℕ) * total_students / sample_students = 640 :=
by
  -- placeholder for calculations based on provided conditions
  sorry

end male_students_in_grade_l625_625528


namespace number_of_truthful_dwarfs_is_4_l625_625593

def dwarf := {x : ℕ // 1 ≤ x ≤ 10}
def likes_vanilla (d : dwarf) : Prop := sorry
def likes_chocolate (d : dwarf) : Prop := sorry
def likes_fruit (d : dwarf) : Prop := sorry
def tells_truth (d : dwarf) : Prop := sorry
def tells_lie (d : dwarf) : Prop := sorry

noncomputable def number_of_truthful_dwarfs : ℕ :=
  let total_dwarfs := 10 in
  let vanilla_raises := 10 in
  let chocolate_raises := 5 in
  let fruit_raises := 1 in
  -- T + L = total_dwarfs
  -- T + 2L = vanilla_raises + chocolate_raises + fruit_raises
  let T := total_dwarfs - 2 * (vanilla_raises + chocolate_raises + fruit_raises - total_dwarfs) in
  T

theorem number_of_truthful_dwarfs_is_4 : number_of_truthful_dwarfs = 4 := 
  by
    sorry

end number_of_truthful_dwarfs_is_4_l625_625593


namespace weekly_exercise_time_l625_625392

def milesWalked := 3
def walkingSpeed := 3 -- in miles per hour
def milesRan := 10
def runningSpeed := 5 -- in miles per hour
def daysInWeek := 7

theorem weekly_exercise_time : (milesWalked / walkingSpeed + milesRan / runningSpeed) * daysInWeek = 21 := 
by
  -- The actual proof part is intentionally omitted as per the instruction
  sorry

end weekly_exercise_time_l625_625392


namespace henry_little_brother_stickers_l625_625681

noncomputable def count_partitions (n k : ℕ) : ℕ := 
  (Multiset.range (n + k)).powerset.filter (λ s, s.sum = n ∧ s.card = k).card

theorem henry_little_brother_stickers : count_partitions 10 5 = 30 :=
sorry

end henry_little_brother_stickers_l625_625681


namespace max_value_theta_argz_l625_625648

open Complex

theorem max_value_theta_argz (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) :
  (let z := (⟨3 * Real.cos θ, 2 * Real.sin θ⟩ : ℂ),
       y := θ - Complex.arg z in
   θ = Real.arctan (Real.sqrt 3 / 2)) :=
by 
  let z := (⟨3 * Real.cos θ, 2 * Real.sin θ⟩ : ℂ),
      y := θ - Complex.arg z
  sorry

end max_value_theta_argz_l625_625648


namespace proof_problem_l625_625199

-- Define the sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℕ := (n^2 + n) / 2

-- Define the arithmetic sequence a_n based on S_n
def a (n : ℕ) : ℕ := if n = 1 then 1 else S n - S (n - 1)

-- Define the geometric sequence b_n with initial conditions
def b (n : ℕ) : ℕ :=
  if n = 1 then a 1 + 1
  else if n = 2 then a 2 + 2
  else 2^n

-- Define the sum of the first n terms of the geometric sequence b_n
def T (n : ℕ) : ℕ := 2 * (2^n - 1)

-- Main theorem to prove
theorem proof_problem :
  (∀ n, a n = n) ∧
  (∀ n, n ≥ 1 → b n = 2^n) ∧
  (∃ n, T n + a n > 300 ∧ ∀ m < n, T m + a m ≤ 300) :=
by {
  sorry
}

end proof_problem_l625_625199


namespace exist_divisible_n_and_n1_l625_625329

theorem exist_divisible_n_and_n1 (d : ℕ) (hd : 0 < d) :
  ∃ (n n1 : ℕ), n % d = 0 ∧ n1 % d = 0 ∧ n ≠ n1 ∧
  (∃ (k a b c : ℕ), b ≠ 0 ∧ n = 10^k * (10 * a + b) + c ∧ n1 = 10^k * a + c) :=
by
  sorry

end exist_divisible_n_and_n1_l625_625329


namespace smallest_perfect_square_divisible_by_2_3_5_l625_625466

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, (n > 0) ∧ (nat.sqrt n * nat.sqrt n = n) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ n = 900 := sorry

end smallest_perfect_square_divisible_by_2_3_5_l625_625466


namespace burn_time_for_structure_l625_625554

-- Define the parameters for the problem
def toothpicks_burned_time (rows cols : ℕ) (initial_burn_time : ℕ) : ℕ :=
  rows * cols * initial_burn_time

-- Define the problem statement
theorem burn_time_for_structure :
  ∀ (initial_burn_time : ℕ), 
  ∀ (rows cols : ℕ) (toothpicks : ℕ), 
  toothpicks = 38 → rows = 3 → cols = 5 → initial_burn_time = 10 →
  ∃ burn_time : ℕ, burn_time = 65 :=
by 
  intros initial_burn_time rows cols toothpicks h1 h2 h3 h4,
  use 65,
  sorry

end burn_time_for_structure_l625_625554


namespace oranges_apples_ratio_l625_625102

variable (A O P : ℕ)
variable (n : ℚ)
variable (h1 : O = n * A)
variable (h2 : P = 4 * O)
variable (h3 : A = (0.08333333333333333 : ℚ) * P)

theorem oranges_apples_ratio (A O P : ℕ) (n : ℚ) 
  (h1 : O = n * A) (h2 : P = 4 * O) (h3 : A = (0.08333333333333333 : ℚ) * P) : n = 3 := 
by
  sorry

end oranges_apples_ratio_l625_625102


namespace meeting_occurs_with_prob_1_div_4_l625_625510

noncomputable def meeting_probability : ℝ := 
  let a := 2^3      -- Total possible volume in hours^3
  let b := 2        -- Volume for the favorable conditions
  b / a

theorem meeting_occurs_with_prob_1_div_4 :
  meeting_probability = 1 / 4 :=
by
  -- Probability calculation as shown in the problem solution
  sorry

end meeting_occurs_with_prob_1_div_4_l625_625510


namespace find_third_vertex_l625_625850

-- Definitions for the problem conditions
def vertex1 : ℝ × ℝ := (6, 4)
def vertex2 : ℝ × ℝ := (0, 0)
def area (x : ℝ) : ℝ := 1 / 2 * abs (vertex1.1 * (vertex2.2 - 0) + vertex2.1 * (0 - vertex1.2) + x * (vertex1.2 - vertex2.2))

-- Statement of the proof problem
theorem find_third_vertex (x : ℝ) (hx : x > 6) (harea : area x = 48) : x = 24 :=
sorry

end find_third_vertex_l625_625850


namespace bakery_combinations_l625_625499

theorem bakery_combinations (h : ∀ (a b c : ℕ), a + b + c = 8 ∧ a > 0 ∧ b > 0 ∧ c > 0) : 
  ∃ count : ℕ, count = 25 := 
sorry

end bakery_combinations_l625_625499


namespace max_diff_n_a_n_l625_625983

def a_n (n : ℕ) : ℕ :=
  Nat.find (λ k, ∃ x : ℕ, (1 ≤ x ∧ x ≤ n ∧ ∀ m, m ∈ list.range' n k → Nat.gcd x m = 1))

theorem max_diff_n_a_n (n : ℕ) (h : n < 100) : 
  ∃ k : ℕ, k = 16 ∧ ∀ m : ℕ, m < n → a_n m = a_n (n - k) :=
sorry

end max_diff_n_a_n_l625_625983


namespace carB_highest_avg_speed_l625_625561

-- Define the distances and times for each car
def distanceA : ℕ := 715
def timeA : ℕ := 11
def distanceB : ℕ := 820
def timeB : ℕ := 12
def distanceC : ℕ := 950
def timeC : ℕ := 14

-- Define the average speeds
def avgSpeedA : ℚ := distanceA / timeA
def avgSpeedB : ℚ := distanceB / timeB
def avgSpeedC : ℚ := distanceC / timeC

theorem carB_highest_avg_speed : avgSpeedB > avgSpeedA ∧ avgSpeedB > avgSpeedC :=
by
  -- Proof will be filled in here
  sorry

end carB_highest_avg_speed_l625_625561


namespace min_vertical_segment_length_l625_625367

def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) : ℝ := -x^2 - 4 * x - 3
def L (x : ℝ) : ℝ := f x - g x

theorem min_vertical_segment_length : ∃ (x : ℝ), L x = 10 :=
by
  sorry

end min_vertical_segment_length_l625_625367


namespace calculate_expression_l625_625953

-- Condition Definitions
def a : ℂ := 3 + 2 * complex.I
def b : ℂ := 1 - 2 * complex.I

-- Statement of the problem
theorem calculate_expression : 3 * a - 4 * b = 5 + 14 * complex.I :=
by
  -- This is where the proof would go
  sorry

end calculate_expression_l625_625953


namespace circle_radius_five_c_value_l625_625963

theorem circle_radius_five_c_value {c : ℝ} :
  (∀ x y : ℝ, x^2 + 8 * x + y^2 + 2 * y + c = 0) → 
  (∃ x y : ℝ, (x + 4)^2 + (y + 1)^2 = 25) → 
  c = 42 :=
by
  sorry

end circle_radius_five_c_value_l625_625963


namespace tony_exercises_hours_per_week_l625_625398

theorem tony_exercises_hours_per_week
  (distance_walked : ℝ)
  (speed_walking : ℝ)
  (distance_ran : ℝ)
  (speed_running : ℝ)
  (days_per_week : ℕ)
  (distance_walked = 3)
  (speed_walking = 3)
  (distance_ran = 10)
  (speed_running = 5)
  (days_per_week = 7) :
  let time_walking := distance_walked / speed_walking,
      time_running := distance_ran / speed_running,
      total_time_per_day := time_walking + time_running
  in total_time_per_day * days_per_week = 21 :=
by
  -- Proof goes here
  sorry

end tony_exercises_hours_per_week_l625_625398


namespace simplify_problem_l625_625342

noncomputable def simplify_expression : ℝ :=
  let numer := (Real.sqrt 3 - 1) ^ (1 - Real.sqrt 2)
  let denom := (Real.sqrt 3 + 1) ^ (1 + Real.sqrt 2)
  numer / denom

theorem simplify_problem :
  simplify_expression = 2 ^ (1 - Real.sqrt 2) * (4 - 2 * Real.sqrt 3) :=
by
  sorry

end simplify_problem_l625_625342


namespace total_transaction_loss_l625_625924

theorem total_transaction_loss
  (h_sp : ℝ) (h_loss_pct : ℝ) (h_sp_given : h_sp = 15000) (h_loss_pct_given : h_loss_pct = 0.25)
  (s_sp : ℝ) (s_gain_pct : ℝ) (s_sp_given : s_sp = 14000) (s_gain_pct_given : s_gain_pct = 0.1667)
  (v_sp : ℝ) (v_gain_pct : ℝ) (v_sp_given : v_sp = 18000) (v_gain_pct_given : v_gain_pct = 0.125) :
  let h_cp := h_sp / (1 - h_loss_pct),
      s_cp := s_sp / (1 + s_gain_pct),
      v_cp := v_sp / (1 + v_gain_pct),
      total_cost := h_cp + s_cp + v_cp,
      total_revenue := h_sp + s_sp + v_sp,
      loss := total_cost - total_revenue
  in loss = 1000 := sorry

end total_transaction_loss_l625_625924


namespace exists_circle_no_marked_point_l625_625775

-- Define the structure representing the integer-coordinated points on the plane
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Condition: No four points marked on the plane lie on the same circle
def no_four_points_on_same_circle (pts : set Point) : Prop :=
  ∀ (p1 p2 p3 p4 : Point), p1 ∈ pts → p2 ∈ pts → p3 ∈ pts → p4 ∈ pts → 
  ¬∃ (c : Point) (r : ℝ), r > 0 ∧ 
  (c.x - p1.x)^2 + (c.y - p1.y)^2 = r^2 ∧
  (c.x - p2.x)^2 + (c.y - p2.y)^2 = r^2 ∧
  (c.x - p3.x)^2 + (c.y - p3.y)^2 = r^2 ∧
  (c.x - p4.x)^2 + (c.y - p4.y)^2 = r^2

-- Main theorem statement
theorem exists_circle_no_marked_point (pts : set Point) (h_no_four : no_four_points_on_same_circle pts) : 
  ∃ (c : Point) (r : ℝ), r = 1995 ∧ ∀ (p : Point), p ∈ pts → (c.x - p.x)^2 + (c.y - p.y)^2 ≠ r^2 :=
sorry

end exists_circle_no_marked_point_l625_625775


namespace MarksScore_l625_625257

theorem MarksScore (h_highest : ℕ) (h_range : ℕ) (h_relation : h_highest - h_least = h_range) (h_mark_twice : Mark = 2 * h_least) : Mark = 46 :=
by
    let h_highest := 98
    let h_range := 75
    let h_least := h_highest - h_range
    let Mark := 2 * h_least
    have := h_relation
    have := h_mark_twice
    sorry

end MarksScore_l625_625257


namespace intervals_of_monotonicity_m_in_terms_of_x0_at_least_two_tangents_l625_625211

noncomputable def h (a x : ℝ) : ℝ := a * x^3 - 1
noncomputable def g (x : ℝ) : ℝ := Real.log x

noncomputable def f (a x : ℝ) : ℝ := h a x + 3 * x * g x
noncomputable def F (a x : ℝ) : ℝ := (a - (1/3)) * x^3 + (1/2) * x^2 * g a - h a x - 1

theorem intervals_of_monotonicity (a : ℝ) (ha : f a 1 = -1) :
  ((a = 0) → (∀ x : ℝ, (0 < x ∧ x < Real.exp (-1) → f 0 x < f 0 x + 3 * x * g x)) ∧
    (Real.exp (-1) < x ∧ 0 < x → f 0 x + 3 * x * g x > f 0 x)) := sorry

theorem m_in_terms_of_x0 (a x0 m : ℝ) (ha : a > Real.exp (10 / 3))
  (tangent_line : ∀ y, y - ( -(1 / 3) * x0^3 + (1 / 2) * x0^2 * g a) = 
    (-(x0^2) + x0 * g a) * (x - x0)) :
  m = (2 / 3) * x0^3 - (1 + (1 / 2) * g a) * x0^2 + x0 * g a := sorry

theorem at_least_two_tangents (a m : ℝ) (ha : a > Real.exp (10 / 3))
  (at_least_two : ∃ x0 y, x0 ≠ y ∧ F a x0 = m ∧ F a y = m) :
  m = 4 / 3 := sorry

end intervals_of_monotonicity_m_in_terms_of_x0_at_least_two_tangents_l625_625211


namespace number_of_truthful_dwarfs_l625_625602

/-- 
Each of the 10 dwarfs either always tells the truth or always lies. 
Each dwarf likes exactly one type of ice cream: vanilla, chocolate, or fruit.
When asked, every dwarf raised their hand for liking vanilla ice cream.
When asked, 5 dwarfs raised their hand for liking chocolate ice cream.
When asked, only 1 dwarf raised their hand for liking fruit ice cream.
Prove that the number of truthful dwarfs is 4.
-/
theorem number_of_truthful_dwarfs (T L : ℕ) 
  (h1 : T + L = 10) 
  (h2 : T + 2 * L = 16) : 
  T = 4 := 
by
  -- Proof omitted
  sorry

end number_of_truthful_dwarfs_l625_625602


namespace total_invested_in_bonds_l625_625073

-- definitions for conditions
def principal_5_75 := 20000
def rate_5_75 := 0.0575
def rate_6_25 := 0.0625
def total_interest := 1900

-- proof statement
theorem total_invested_in_bonds : 
  let interest_from_5_75 := principal_5_75 * rate_5_75 in
  ∃ (principal_6_25 : ℝ), 
    interest_from_5_75 + principal_6_25 * rate_6_25 = total_interest ∧ 
    let total_invested := principal_5_75 + principal_6_25 in 
    total_invested = 32000 :=
begin
  sorry
end

end total_invested_in_bonds_l625_625073


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625421

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, (n > 0 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∧ ∃ k : ℕ, n = k^2 ∧ n = 225 := 
begin
  sorry
end

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625421


namespace zero_is_smallest_natural_number_l625_625372

theorem zero_is_smallest_natural_number : ∀ n : ℕ, 0 ≤ n :=
by
  intro n
  exact Nat.zero_le n

#check zero_is_smallest_natural_number  -- confirming the theorem check

end zero_is_smallest_natural_number_l625_625372


namespace original_ratio_l625_625037

theorem original_ratio (x y : ℕ) (h1 : y = 40) (h2 : (x + 10) / (y + 10) = 4 / 5) : x / y = 3 / 4 :=
by
  have h3 : y + 10 = 50 := by rw [h1]; exact rfl
  have h4 : (x + 10) / 50 = 4 / 5 := by rw [h3] at h2; exact h2
  sorry

end original_ratio_l625_625037


namespace number_of_truthful_dwarfs_l625_625605

def dwarf_condition := 
  ∀ (dwarfs : ℕ) (truthful_dwarfs : ℕ) (lying_dwarfs : ℕ),
    dwarfs = 10 ∧ 
    (∀ n, n ∈ {truthful_dwarfs, lying_dwarfs} -> n ≥ 0) ∧ 
    truthful_dwarfs + lying_dwarfs = dwarfs ∧
    truthful_dwarfs + 2 * lying_dwarfs = 16

theorem number_of_truthful_dwarfs : ∃ (truthful_dwarfs : ℕ), (dwarf_condition ∧ truthful_dwarfs = 4) :=
by {
  let dwarfs := 10,
  let lying_dwarfs := 6,
  let truthful_dwarfs := dwarfs - lying_dwarfs,
  have h: truthful_dwarfs = 4,
  { calc
    truthful_dwarfs = dwarfs - lying_dwarfs : by rfl
    ... = 10 - 6 : by rfl
    ... = 4 : by rfl },
  existsi (4 : ℕ),
  refine ⟨_, ⟨dwarfs, truthful_dwarfs, lying_dwarfs, rfl, _, _, _⟩⟩,
  -- Now we can provide the additional details for lean to understand the conditions hold
  {
    intros n hn,
    simp,
    exact hn
  },
  {
    exact add_comm 6 4
  },
  {
    dsimp,
    ring,
  },
  {
    exact h,
  }
  -- Skip the actual proof with sorry
  sorry
}

end number_of_truthful_dwarfs_l625_625605


namespace horner_method_evaluation_v3_at_1_l625_625406

def f (x : ℤ) : ℤ := 7 * x ^ 7 + 5 * x ^ 5 + 4 * x ^ 4 + 2 * x ^ 2 + x + 2

theorem horner_method_evaluation_v3_at_1 :
  let x := 1,
      v0 := 7,
      v1 := v0 * x,
      v2 := v1 * x + 5,
      v3 := v2 * x + 4
  in v3 = 16 :=
by
  let x := 1
  let v0 := 7
  let v1 := v0 * x
  let v2 := v1 * x + 5
  let v3 := v2 * x + 4
  exact eq.refl 16

end horner_method_evaluation_v3_at_1_l625_625406


namespace product_of_fractions_is_27_l625_625129

theorem product_of_fractions_is_27 :
  (1/3) * (9/1) * (1/27) * (81/1) * (1/243) * (729/1) = 27 :=
by
  sorry

end product_of_fractions_is_27_l625_625129


namespace LCM_20_45_75_is_900_l625_625877

def prime_factorization_20 := (2^2, 5)
def prime_factorization_45 := (3^2, 5)
def prime_factorization_75 := (3, 5^2)

theorem LCM_20_45_75_is_900 
  (pf_20 : prime_factorization_20 = (2^2, 5))
  (pf_45 : prime_factorization_45 = (3^2, 5))
  (pf_75 : prime_factorization_75 = (3, 5^2)) : 
  Nat.lcm (Nat.lcm 20 45) 75 = 900 := 
  by sorry

end LCM_20_45_75_is_900_l625_625877


namespace imaginary_condition_l625_625192

noncomputable def is_imaginary (z : ℂ) : Prop := z.im ≠ 0

theorem imaginary_condition (z1 z2 : ℂ) :
  ( ∃ (z1 : ℂ), is_imaginary z1 ∨ is_imaginary z2 ∨ (is_imaginary (z1 - z2))) ↔
  ∃ (z1 z2 : ℂ), is_imaginary z1 ∨ is_imaginary z2 ∧ ¬ (is_imaginary (z1 - z2)) :=
sorry

end imaginary_condition_l625_625192


namespace christopher_age_l625_625484

theorem christopher_age :
  ∃ C : ℕ, 
    let George := C + 8,
        Ford := C - 2 in
    George + Ford + C = 60 ∧ C = 18 :=
sorry

end christopher_age_l625_625484


namespace jennifer_dogs_l625_625728

theorem jennifer_dogs (D : ℕ) (groom_time_per_dog : ℕ) (groom_days : ℕ) (total_groom_time : ℕ) :
  groom_time_per_dog = 20 →
  groom_days = 30 →
  total_groom_time = 1200 →
  groom_days * (groom_time_per_dog * D) = total_groom_time →
  D = 2 :=
by
  intro h1 h2 h3 h4
  sorry

end jennifer_dogs_l625_625728


namespace parallelogram_base_l625_625156

theorem parallelogram_base (height area : ℕ) (h_height : height = 18) (h_area : area = 612) : ∃ base, base = 34 :=
by
  -- The proof would go here
  sorry

end parallelogram_base_l625_625156


namespace minimal_beacons_required_l625_625725

noncomputable def maze_beacons (num_rooms : ℕ) (segments : ℕ → ℕ → ℕ) (beacon_distances : ℕ → ℕ → ℕ) : Prop :=
  ∃ (beacon_positions : list ℕ), beacon_positions.length = 3 ∧
  ∀ room₁ room₂, room₁ ≠ room₂ →
  (∃ beacon₁ beacon₂ beacon₃ ∈ beacon_positions,
    beacon_distances room₁ beacon₁ ≠ beacon_distances room₂ beacon₁ ∨
    beacon_distances room₁ beacon₂ ≠ beacon_distances room₂ beacon₂ ∨
    beacon_distances room₁ beacon₃ ≠ beacon_distances room₂ beacon₃)

theorem minimal_beacons_required (num_rooms : ℕ) (segments : ℕ → ℕ → ℕ):
  ∃ (beacon_distances : ℕ → ℕ → ℕ), maze_beacons num_rooms segments beacon_distances :=
sorry

end minimal_beacons_required_l625_625725


namespace travel_time_third_to_first_l625_625910

variable (boat_speed current_speed : ℝ) -- speeds of the boat and current
variable (d1 d2 d3 : ℝ) -- distances between the docks

-- Conditions
variable (h1 : 30 / 60 = d1 / (boat_speed - current_speed)) -- 30 minutes from one dock to another against current
variable (h2 : 18 / 60 = d2 / (boat_speed + current_speed)) -- 18 minutes from another dock to the third with current
variable (h3 : d1 + d2 = d3) -- Total distance is sum of d1 and d2

theorem travel_time_third_to_first : (d3 / (boat_speed - current_speed)) * 60 = 72 := 
by 
  -- here goes the proof which is omitted
  sorry

end travel_time_third_to_first_l625_625910


namespace max_neg_4_l625_625695

def is_maximum_neg_ints (a b c d e f g h i : ℤ) : Prop := 
  |a| ≤ 10 ∧ |b| ≤ 10 ∧ |c| ≤ 30 ∧ (a * b + c * d * e * f + g * h * i) < 0 → 
  max [(a < 0).toInt, (b < 0).toInt, (c < 0).toInt, (d < 0).toInt, (e < 0).toInt, (f < 0).toInt, (g < 0).toInt, (h < 0).toInt, (i < 0).toInt] = 4

theorem max_neg_4 (a b c d e f g h i : ℤ) : is_maximum_neg_ints a b c d e f g h i :=
by {
  unfold is_maximum_neg_ints, 
  sorry 
}

end max_neg_4_l625_625695


namespace calculate_f_of_555_l625_625920

def is_linear_in_each {n : ℕ} (f : (fin n → ℝ) → ℝ) : Prop :=
  ∀ (i : fin n) (x y : fin n → ℝ) (a b : ℝ), f (λ j, if j = i then a * x j + b * y j else x j) =
  a * f x + b * f y

noncomputable def f (x : fin n → ℝ) : ℝ :=
  if ∀ i, x i ∈ ({3, 4} : set ℝ) then 1 / ∏ i, x i else
  sorry -- Definition for cases where x_i are not in {3, 4} is unspecified

theorem calculate_f_of_555 (n : ℕ) (x : fin n → ℝ)
  (hx : ∀ i, x i = 5) (hlin : is_linear_in_each f) :
  f x = 1 / 6^n :=
begin
  sorry -- Proof goes here
end

end calculate_f_of_555_l625_625920


namespace positive_y_percent_y_eq_16_l625_625932

theorem positive_y_percent_y_eq_16 (y : ℝ) (hy : 0 < y) (h : 0.01 * y * y = 16) : y = 40 :=
by
  sorry

end positive_y_percent_y_eq_16_l625_625932


namespace base_conversion_403_base_6_eq_223_base_8_l625_625122

theorem base_conversion_403_base_6_eq_223_base_8 :
  (6^2 * 4 + 6^1 * 0 + 6^0 * 3 : ℕ) = (8^2 * 2 + 8^1 * 2 + 8^0 * 3 : ℕ) :=
by
  sorry

end base_conversion_403_base_6_eq_223_base_8_l625_625122


namespace smallest_perfect_square_divisible_by_2_3_5_l625_625462

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, (n > 0) ∧ (nat.sqrt n * nat.sqrt n = n) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ n = 900 := sorry

end smallest_perfect_square_divisible_by_2_3_5_l625_625462


namespace f_even_f_monotonic_increasing_find_x_l625_625205

-- Define the function f
def f (x : ℝ) : ℝ := 2^x + 2^(-x)

-- Prove that f is an even function
theorem f_even : ∀ x : ℝ, f (-x) = f x :=
by sorry

-- Prove that f is monotonically increasing on (0, +∞)
theorem f_monotonic_increasing : ∀ x1 x2 : ℝ, (0 < x1) → (x1 < x2) → f x1 < f x2 :=
by sorry

-- Find the value of x such that f(x) = 5 * 2^(-x) + 3
theorem find_x : ∀ x : ℝ, f x = 5 * 2^(-x) + 3 → x = 2 :=
by sorry

end f_even_f_monotonic_increasing_find_x_l625_625205


namespace lcm_20_45_75_l625_625880

def lcm (a b : ℕ) : ℕ := nat.lcm a b

theorem lcm_20_45_75 : lcm (lcm 20 45) 75 = 900 :=
by
  sorry

end lcm_20_45_75_l625_625880


namespace vertex_quadratic_m_n_l625_625810

theorem vertex_quadratic_m_n (n : ℝ) (m : ℝ) (h : (m, 1) = (1, n - 1)) : m - n = -1 :=
by
  have h1 : m = 1 := by
    rw [←h.left]
    rfl
  have h2 : n - 1 = 1 := by
    rw [←h.right]
  have h3 : n = 2 := by
    linarith
  have h4 : m - n = 1 - 2 := by
    rw [h1, h3]
    rfl
  show m - n = -1 from
    sorry

end vertex_quadratic_m_n_l625_625810


namespace probability_blue_face_facing_up_l625_625412

-- Define the context
def octahedron_faces : ℕ := 8
def blue_faces : ℕ := 5
def red_faces : ℕ := 3
def total_faces : ℕ := blue_faces + red_faces

-- The probability calculation theorem
theorem probability_blue_face_facing_up (h : total_faces = octahedron_faces) :
  (blue_faces : ℝ) / (octahedron_faces : ℝ) = 5 / 8 :=
by
  -- Placeholder for proof
  sorry

end probability_blue_face_facing_up_l625_625412


namespace total_weight_AlF3_10_moles_l625_625417

noncomputable def molecular_weight_AlF3 (atomic_weight_Al: ℝ) (atomic_weight_F: ℝ) : ℝ :=
  atomic_weight_Al + 3 * atomic_weight_F

theorem total_weight_AlF3_10_moles :
  let atomic_weight_Al := 26.98
  let atomic_weight_F := 19.00
  let num_moles := 10
  molecular_weight_AlF3 atomic_weight_Al atomic_weight_F * num_moles = 839.8 :=
by
  sorry

end total_weight_AlF3_10_moles_l625_625417


namespace simplify_and_evaluate_l625_625791

variable (a : ℝ)
axiom a_cond : a = -1 / 3

theorem simplify_and_evaluate : (3 * a - 1) ^ 2 + 3 * a * (3 * a + 2) = 3 :=
by
  have ha : a = -1 / 3 := a_cond
  sorry

end simplify_and_evaluate_l625_625791


namespace three_c_minus_d_value_l625_625001

-- Define the equation and the conditions
def equation (x : ℝ) := x^2 - 6 * x + 15 = 27

-- Solutions c and d where c >= d
noncomputable def c := 3 + real.sqrt 21
noncomputable def d := 3 - real.sqrt 21

-- Theorem stating the relationship
theorem three_c_minus_d_value : equation c ∧ equation d → (3 * c - d = 6 + 4 * real.sqrt 21) := by
  sorry

end three_c_minus_d_value_l625_625001


namespace price_decrease_percentage_l625_625917

variable (current_price : ℕ) (price_decrease : ℕ)

def original_price := current_price + price_decrease
def percentage_decrease : ℚ := price_decrease / original_price

theorem price_decrease_percentage (hc : current_price = 3800) (hd : price_decrease = 200) : 
  percentage_decrease current_price price_decrease = 5 / 100 :=
by
  sorry

end price_decrease_percentage_l625_625917


namespace wooden_block_length_is_correct_l625_625370

noncomputable def length_of_block : ℝ :=
  let initial_length := 31
  let reduction := 30 / 100
  initial_length - reduction

theorem wooden_block_length_is_correct :
  length_of_block = 30.7 :=
by
  sorry

end wooden_block_length_is_correct_l625_625370


namespace distinct_flavor_ratios_l625_625926

theorem distinct_flavor_ratios : 
  let total_ratios := (5 + 1) * (4 + 1) - 1 in
  let no_purple_dupes := 3 in
  let no_orange_dupes := 4 in
  let same_number_dupes := 3 in
  let specific_dupes := 1 in
  total_ratios - (no_purple_dupes + no_orange_dupes + same_number_dupes + specific_dupes) = 18 :=
by
  -- Definitions of total_ratios, no_purple_dupes, no_orange_dupes, same_number_dupes, specific_dupes
  let total_ratios := (5 + 1) * (4 + 1) - 1
  let no_purple_dupes := 3
  let no_orange_dupes := 4
  let same_number_dupes := 3
  let specific_dupes := 1
  calc total_ratios - (no_purple_dupes + no_orange_dupes + same_number_dupes + specific_dupes)
      = 30 - 11 : by rw [total_ratios]
      _ = 19 : by norm_num
      _ - 1 = 18 : by norm_num


end distinct_flavor_ratios_l625_625926


namespace baseball_games_in_season_l625_625026

theorem baseball_games_in_season 
  (games_per_month : ℕ) 
  (months_in_season : ℕ)
  (h1 : games_per_month = 7) 
  (h2 : months_in_season = 2) :
  games_per_month * months_in_season = 14 := by
  sorry


end baseball_games_in_season_l625_625026


namespace length_of_MN_l625_625639

theorem length_of_MN (A B C D K L M N : Type) 
  (h1 : A → B → C → D → Prop) -- Condition for rectangle ABCD
  (h2 : K → L → Prop) -- Condition for circle intersecting AB at K and L
  (h3 : M → N → Prop) -- Condition for circle intersecting CD at M and N
  (AK KL DN : ℝ)
  (h4 : AK = 10)
  (h5 : KL = 17)
  (h6 : DN = 7) :
  ∃ MN : ℝ, MN = 23 := 
sorry

end length_of_MN_l625_625639


namespace moe_share_of_pie_l625_625550

-- Definitions based on conditions
def leftover_pie : ℚ := 8 / 9
def num_people : ℚ := 3

-- Theorem to prove the amount of pie Moe took home
theorem moe_share_of_pie : (leftover_pie / num_people) = 8 / 27 := by
  sorry

end moe_share_of_pie_l625_625550


namespace find_non_specific_analysis_idiom_l625_625829

def idiom_A : Prop := "Prescribe the right medicine for the illness; Make clothes to fit the person" reflects "specific analysis of specific issues"
def idiom_B : Prop := "Let go to catch; Attack the east while feigning the west" does not reflect "specific analysis of specific issues"
def idiom_C : Prop := "Act according to the situation; Adapt to local conditions" reflects "specific analysis of specific issues"
def idiom_D : Prop := "Teach according to aptitude; Differentiate instruction based on individual differences" reflects "specific analysis of specific issues"

theorem find_non_specific_analysis_idiom :
  idiom_B :=
sorry

end find_non_specific_analysis_idiom_l625_625829


namespace g_at_32_l625_625814

variable {α : Type} [Add α] [Mul α] [HasPow α]

def g (x : α) : α

axiom functional_eqn : ∀ x : α, g (x + g x) = 6 * g x
axiom g_at_2 : g 2 = 5

theorem g_at_32 : g 32 = 180 := by
  sorry

end g_at_32_l625_625814


namespace tony_exercises_hours_per_week_l625_625400

theorem tony_exercises_hours_per_week
  (distance_walked : ℝ)
  (speed_walking : ℝ)
  (distance_ran : ℝ)
  (speed_running : ℝ)
  (days_per_week : ℕ)
  (distance_walked = 3)
  (speed_walking = 3)
  (distance_ran = 10)
  (speed_running = 5)
  (days_per_week = 7) :
  let time_walking := distance_walked / speed_walking,
      time_running := distance_ran / speed_running,
      total_time_per_day := time_walking + time_running
  in total_time_per_day * days_per_week = 21 :=
by
  -- Proof goes here
  sorry

end tony_exercises_hours_per_week_l625_625400


namespace average_percentage_decrease_l625_625074

theorem average_percentage_decrease (p1 p2 : ℝ) (n : ℕ) (h₀ : p1 = 2000) (h₁ : p2 = 1280) (h₂ : n = 2) :
  ((p1 - p2) / p1 * 100) / n = 18 := 
by
  sorry

end average_percentage_decrease_l625_625074


namespace vertical_complementary_perpendicular_l625_625696

theorem vertical_complementary_perpendicular (α β : ℝ) (l1 l2 : ℝ) :
  (α = β ∧ α + β = 90) ∧ l1 = l2 -> l1 + l2 = 90 := by
  sorry

end vertical_complementary_perpendicular_l625_625696


namespace incorrect_variance_l625_625981

-- Given dataset
def dataset : List ℕ := [2, 3, 6, 9, 3, 7]

-- Definitions from the question (conditions)
def mode (lst : List ℕ) : ℕ := lst.foldr (λ x acc, if lst.count x > lst.count acc then x else acc) lst.head
def mean (lst : List ℕ) : ℚ := (lst.sum : ℚ) / lst.length
def variance (lst : List ℕ) : ℚ := 
  let m := mean lst
  (lst.map (λ x => (x - m)^2)).sum / (lst.length)

def median (lst : List ℕ) : ℚ :=
  let sorted := lst.qsort (≤)
  if sorted.length % 2 = 0 then ((sorted.nth (sorted.length / 2 - 1)).getD 0 + (sorted.nth (sorted.length / 2)).getD 0 : ℚ) / 2
  else (sorted.nth (sorted.length / 2)).getD 0

-- Proof problem statement
theorem incorrect_variance :
  mode dataset = 3 ∧
  variance dataset ≠ 4 ∧
  mean dataset = 5 ∧
  median dataset = 4.5 :=
by
  sorry

end incorrect_variance_l625_625981


namespace cakes_served_during_lunch_l625_625099

theorem cakes_served_during_lunch (T D L : ℕ) (h1 : T = 15) (h2 : D = 9) : L = T - D → L = 6 :=
by
  intros h
  rw [h1, h2] at h
  exact h

end cakes_served_during_lunch_l625_625099


namespace find_negative_x_l625_625817

noncomputable def median (l : List ℤ) : ℤ := 
  let sorted := l.qsort (≤)
  sorted.get! (sorted.length / 2)

noncomputable def mean (l : List ℤ) : ℤ :=
  l.sum / l.length

theorem find_negative_x :
  ∃ x : ℤ, (x < 0) ∧ (median [20, 50, 55, x, 22] = mean [20, 50, 55, x, 22] - 7) ∧ (x = -2) :=
by {
  let x := -2,
  have hx : x < 0 := by norm_num,
  have hmedian_mean : median [20, 50, 55, x, 22] = mean [20, 50, 55, x, 22] - 7 := by sorry,
  exact ⟨x, hx, hmedian_mean, rfl⟩
}

end find_negative_x_l625_625817


namespace cos_double_angle_l625_625187

theorem cos_double_angle (a : ℝ) (h : Real.sin a = 1 / 3) : Real.cos (2 * a) = 7 / 9 :=
by
  sorry

end cos_double_angle_l625_625187


namespace point_same_side_of_line_l625_625580

def same_side (p₁ p₂ : ℝ × ℝ) (a b c : ℝ) : Prop :=
  (a * p₁.1 + b * p₁.2 + c > 0) ↔ (a * p₂.1 + b * p₂.2 + c > 0)

theorem point_same_side_of_line :
  same_side (1, 2) (1, 0) 2 (-1) 1 :=
by
  unfold same_side
  sorry

end point_same_side_of_line_l625_625580


namespace number_of_truthful_dwarfs_l625_625607

def dwarf_condition := 
  ∀ (dwarfs : ℕ) (truthful_dwarfs : ℕ) (lying_dwarfs : ℕ),
    dwarfs = 10 ∧ 
    (∀ n, n ∈ {truthful_dwarfs, lying_dwarfs} -> n ≥ 0) ∧ 
    truthful_dwarfs + lying_dwarfs = dwarfs ∧
    truthful_dwarfs + 2 * lying_dwarfs = 16

theorem number_of_truthful_dwarfs : ∃ (truthful_dwarfs : ℕ), (dwarf_condition ∧ truthful_dwarfs = 4) :=
by {
  let dwarfs := 10,
  let lying_dwarfs := 6,
  let truthful_dwarfs := dwarfs - lying_dwarfs,
  have h: truthful_dwarfs = 4,
  { calc
    truthful_dwarfs = dwarfs - lying_dwarfs : by rfl
    ... = 10 - 6 : by rfl
    ... = 4 : by rfl },
  existsi (4 : ℕ),
  refine ⟨_, ⟨dwarfs, truthful_dwarfs, lying_dwarfs, rfl, _, _, _⟩⟩,
  -- Now we can provide the additional details for lean to understand the conditions hold
  {
    intros n hn,
    simp,
    exact hn
  },
  {
    exact add_comm 6 4
  },
  {
    dsimp,
    ring,
  },
  {
    exact h,
  }
  -- Skip the actual proof with sorry
  sorry
}

end number_of_truthful_dwarfs_l625_625607


namespace factorial_expression_l625_625128

theorem factorial_expression : (fact (fact 4)) / (fact 4) = fact 23 := by
  sorry

end factorial_expression_l625_625128


namespace number_of_truthful_dwarfs_is_4_l625_625595

def dwarf := {x : ℕ // 1 ≤ x ≤ 10}
def likes_vanilla (d : dwarf) : Prop := sorry
def likes_chocolate (d : dwarf) : Prop := sorry
def likes_fruit (d : dwarf) : Prop := sorry
def tells_truth (d : dwarf) : Prop := sorry
def tells_lie (d : dwarf) : Prop := sorry

noncomputable def number_of_truthful_dwarfs : ℕ :=
  let total_dwarfs := 10 in
  let vanilla_raises := 10 in
  let chocolate_raises := 5 in
  let fruit_raises := 1 in
  -- T + L = total_dwarfs
  -- T + 2L = vanilla_raises + chocolate_raises + fruit_raises
  let T := total_dwarfs - 2 * (vanilla_raises + chocolate_raises + fruit_raises - total_dwarfs) in
  T

theorem number_of_truthful_dwarfs_is_4 : number_of_truthful_dwarfs = 4 := 
  by
    sorry

end number_of_truthful_dwarfs_is_4_l625_625595


namespace water_level_at_half_height_l625_625409

noncomputable def rate_of_water_level_rising (v H R h : ℝ) : ℝ :=
let pi := Real.pi in
let cone_volume := (1 / 3) * pi * (h^3 / (H^2)) in
let time := (cone_volume * 17.28) / v in
let dh_dt := (v * 17.28 / pi) / (3 * h^2) in
dh_dt

theorem water_level_at_half_height
  (v : ℝ := 1.2 * 10^(-3))
  (H : ℝ := 2.4)
  (R : ℝ := 1)
  (h : ℝ := 1.2) :
  rate_of_water_level_rising v H R h ≈ 1.528 * 10^(-3) :=
by sorry

end water_level_at_half_height_l625_625409


namespace inscribed_square_ab_l625_625526

theorem inscribed_square_ab (a b : ℝ) :
    (∃ (a b : ℝ), 
        (∀ (A : ℝ), A = 16 → side_length_of_square A = 4) ∧ 
        (∀ (B : ℝ), B = 18 → side_length_of_square B = 3 * Real.sqrt 2) ∧ 
        rotated(30) ∧ 
        a + b = 3 * Real.sqrt 2 ∧
        Real.sqrt (a^2 + b^2) = 4 * Real.sqrt 3)
    → a * b = -15 :=
by
  sorry

end inscribed_square_ab_l625_625526


namespace decimal_to_vulgar_fraction_l625_625414

-- Define the decimal as a fraction
def decimal_as_fraction : ℚ := 0.38

-- State the fraction form of the decimal is 38/100
def fraction_form : ℚ := 38 / 100

-- Define greatest common divisor (GCD) used for simplifying
def gcd_num_den : ℕ := Nat.gcd 38 100

-- Simplify the fraction using the GCD
def simplified_fraction : ℚ := (38 / gcd_num_den) / (100 / gcd_num_den)

-- Main theorem to prove
theorem decimal_to_vulgar_fraction : decimal_as_fraction = simplified_fraction := by
  have : gcd_num_den = 2 := by nat.gcd_simp
  simp [decimal_as_fraction, fraction_form, simplified_fraction, this]
  sorry

end decimal_to_vulgar_fraction_l625_625414


namespace find_lambda_l625_625676

variables (a b : ℝ^3) (m : ℝ) (λ : ℝ)
noncomputable def magnitude (v : ℝ^3) : ℝ := real.sqrt (v.dot v)

-- Given Conditions
def conditions (a b : ℝ^3) (m λ : ℝ) :=
  magnitude a = m ∧
  magnitude b = 2 * m ∧
  (a.dot b) = m * (2 * m) * real.cos (2 * real.pi / 3) ∧
  (a.dot (a - λ • b)) = 0

theorem find_lambda (a b : ℝ^3) (m : ℝ) (λ : ℝ) (h : conditions a b m λ) : λ = -1 :=
sorry

end find_lambda_l625_625676


namespace escape_path_in_convex_forest_l625_625784

theorem escape_path_in_convex_forest (S : ℝ) (hS : 0 < S) :
  ∃ L : ℝ, L ≤ sqrt (2 * Real.pi * S) ∧ ∃ path : ℝ → ℝ × ℝ, continuous path ∧ (∀ t, 0 ≤ t ∧ t ≤ L → is_exit path t) :=
sorry

end escape_path_in_convex_forest_l625_625784


namespace fewest_seats_to_be_occupied_l625_625383

theorem fewest_seats_to_be_occupied (n : ℕ) (h : n = 120) : ∃ m, m = 40 ∧
  ∀ a b, a + b = n → a ≥ m → ∀ x, (x > 0 ∧ x ≤ n) → (x > 1 → a = m → a + (b / 2) ≥ n / 3) :=
sorry

end fewest_seats_to_be_occupied_l625_625383


namespace induction_step_expression_l625_625781

theorem induction_step_expression (n : ℕ) (h : 0 < n) :
  ∀ k : ℕ, (k+1) * (k+2) * ... * (k + k) = 2^k * 1 * 2 * ... * (2k - 1) → 
  (k+2) * (k+3) * ... * (2k + 1) * (2k + 2) / ((k+1) * (k+2) * ... * 2k) = 2 * (2k + 1) :=
by sorry

end induction_step_expression_l625_625781


namespace prove_sin_cos_theta_symmetric_l625_625697

noncomputable def sin_cos_theta (a θ : ℝ) : ℝ :=
  if (C1_center : -a / 2 = 1 / 2 ∧ 2 * (-a / 2) = 1) ∧
     (C2_center : -a = a ∧ -1 + (tan θ) / 2 - 1 = 0) then
    (sin θ * cos θ = -2 / 5)
  else
    0 -- This handles the impossible branch if the conditions are not met. More robust handling may be added.
    
theorem prove_sin_cos_theta_symmetric (a θ : ℝ) 
  (h1 : C1_equation := x^2 + y^2 + a * x = 0)
  (h2 : C2_equation := x^2 + y^2 + 2 * a * x + y * tan θ = 0)
  (h_symm : symmetric_about 2 * x - y - 1 = 0) 
  : sin θ * cos θ = -2 / 5 :=
sorry

end prove_sin_cos_theta_symmetric_l625_625697


namespace energy_drinks_consumption_l625_625935

theorem energy_drinks_consumption (k : ℕ)
  (h_relaxing_day : ℕ) (g_relaxing_day : ℕ)
  (h_intensive_day : ℕ) 
  (g_intensive_day : ℕ)
  (inverse_proportional : ∀ h g : ℕ, g * h = k) 
  (relaxing_day_cond : h_relaxing_day = 4 ∧ g_relaxing_day = 5) 
  (intensive_day_cond : h_intensive_day = 2) :
  g_intensive_day = 10 :=
by {
  -- Adding required conditions
  have k_value : k = 20,
  { rw [← inverse_proportional 4 5, relaxing_day_cond.left, relaxing_day_cond.right],
    exact rfl, },
  have g_intensive_day_eq : g_intensive_day * 2 = 20,
  { rw [← inverse_proportional 2 g_intensive_day, intensive_day_cond],
    exact k_value, },
  -- conclude the result
  exact (nat.mul_right_inj' zero_lt_two).1 g_intensive_day_eq,
}

end energy_drinks_consumption_l625_625935


namespace lock_code_digits_l625_625019

noncomputable def valid_lock_code_count (n : ℕ) : ℕ :=
  if n < 2 then 0
  else 4 * 4 * Nat.descFactorial 5 (n - 2)

theorem lock_code_digits : ∃ n : ℕ, valid_lock_code_count n = 240 :=
begin
  use 4,
  unfold valid_lock_code_count,
  norm_num,
  rw Nat.descFactorial_eq_factorial_div_factorial,
  norm_num,
end

end lock_code_digits_l625_625019


namespace cosine_of_largest_angle_l625_625705

theorem cosine_of_largest_angle (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  ∃ C, cosine_law a b c = C ∧ C = -1/4 :=
by {
  sorry
}

end cosine_of_largest_angle_l625_625705


namespace weekly_exercise_time_l625_625394

def milesWalked := 3
def walkingSpeed := 3 -- in miles per hour
def milesRan := 10
def runningSpeed := 5 -- in miles per hour
def daysInWeek := 7

theorem weekly_exercise_time : (milesWalked / walkingSpeed + milesRan / runningSpeed) * daysInWeek = 21 := 
by
  -- The actual proof part is intentionally omitted as per the instruction
  sorry

end weekly_exercise_time_l625_625394


namespace cos_alpha_correct_l625_625621

theorem cos_alpha_correct (α : ℝ) (x y : ℝ) (h : (x, y) = (-1, 2)) : 
  let r := real.sqrt (x^2 + y^2),
  cos α = x / r := 
begin
  have hx : x = -1, from by { rw h, refl },
  have hy : y = 2, from by { rw h, refl },
  have hr : r = real.sqrt ((-1)^2 + 2^2), from by { rw [hx, hy], refl },
  have hr2 : r = real.sqrt 5, from by { simp [hr] },
  rw [hx, hr2],
  simp,
  field_simp,
  ring,
end

end cos_alpha_correct_l625_625621


namespace part1_solve_inequality_part2_range_of_a_l625_625765

def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + a)

theorem part1_solve_inequality (x : ℝ) (h : -2 < x ∧ x < -2/3) :
    f x 1 > 1 :=
by
  sorry

theorem part2_range_of_a (h : ∀ x, 2 ≤ x ∧ x ≤ 3 → f x (a : ℝ) > 0) :
    -5/2 < a ∧ a < -2 :=
by
  sorry

end part1_solve_inequality_part2_range_of_a_l625_625765


namespace actual_cost_of_article_l625_625055

noncomputable def actual_cost : ℝ := 421.05

theorem actual_cost_of_article (x : ℝ) (h : 0.76 * x = 320) : x ≈ actual_cost :=
by
  sorry

end actual_cost_of_article_l625_625055


namespace lucas_mp3_player_l625_625770

noncomputable def song_order_probability (n : ℕ) (first_song_duration : ℕ) (increment : ℕ) (favorite_duration : ℕ) (total_time : ℕ) : ℚ :=
  let total_arrangements := nat.factorial n
  let valid_arrangements := nat.factorial (n - 1)
  1 - valid_arrangements / total_arrangements

theorem lucas_mp3_player (n first_song_duration increment favorite_duration total_time : ℕ) :
  n = 12 →
  first_song_duration = 45 →
  increment = 45 →
  favorite_duration = 4 * 60 + 15 →
  total_time = 5 * 60 →
  song_order_probability n first_song_duration increment favorite_duration total_time = 11 / 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end lucas_mp3_player_l625_625770


namespace no_coprime_x_and_y_l625_625782

-- Define square-free integer
def square_free (n : ℕ) : Prop :=
  ∀ (p : ℕ), p.prime → p^2 ∣ n → false

-- Define coprime integers
def coprime (x y : ℕ) : Prop :=
  Nat.gcd x y = 1

-- State the main theorem
theorem no_coprime_x_and_y (n x y : ℕ) (hsqfree : square_free n) (hcoprime : coprime x y) :
  ¬ (x + y)^3 ∣ x^n + y^n :=
  sorry

end no_coprime_x_and_y_l625_625782


namespace quarters_needed_to_buy_items_l625_625520

-- Define the costs of each item in cents
def cost_candy_bar : ℕ := 25
def cost_chocolate : ℕ := 75
def cost_juice : ℕ := 50

-- Define the quantities of each item
def num_candy_bars : ℕ := 3
def num_chocolates : ℕ := 2
def num_juice_packs : ℕ := 1

-- Define the value of a quarter in cents
def value_of_quarter : ℕ := 25

-- Define the total cost of the items
def total_cost : ℕ := (num_candy_bars * cost_candy_bar) + (num_chocolates * cost_chocolate) + (num_juice_packs * cost_juice)

-- Calculate the number of quarters needed
def num_quarters_needed : ℕ := total_cost / value_of_quarter

-- The theorem to prove that the number of quarters needed is 11
theorem quarters_needed_to_buy_items : num_quarters_needed = 11 := by
  -- Proof omitted
  sorry

end quarters_needed_to_buy_items_l625_625520


namespace trapezoid_QR_length_l625_625805

variable (PQ RS Area Alt QR : ℝ)
variable (h1 : Area = 216)
variable (h2 : Alt = 9)
variable (h3 : PQ = 12)
variable (h4 : RS = 20)
variable (h5 : QR = 11)

theorem trapezoid_QR_length : 
  (∃ (PQ RS Area Alt QR : ℝ), 
    Area = 216 ∧
    Alt = 9 ∧
    PQ = 12 ∧
    RS = 20) → QR = 11 :=
by
  sorry

end trapezoid_QR_length_l625_625805


namespace find_number_l625_625094

def sum_of_digits (n : Nat) : Nat :=
  n.digits.Sum

theorem find_number (n : Nat) (h : n * sum_of_digits n = 2008) : n = 251 := 
by
  sorry

end find_number_l625_625094


namespace range_of_a_l625_625703

theorem range_of_a (a : ℝ) :
  (¬ ∃ t : ℝ, t^2 - 2 * t - a < 0) ↔ a ≤ -1 :=
by sorry

end range_of_a_l625_625703


namespace triangle_area_l625_625272

theorem triangle_area (A B C : Type) [EuclideanGeometry A B C] 
  (angle_BAC : angle A B C = 60)
  (AB : dist A B = 5)
  (AC : dist A C = 6) : 
  area A B C = (15 * Real.sqrt 3) / 2 := 
  sorry

end triangle_area_l625_625272


namespace rectangle_areas_sum_m_n_equals_l625_625960

-- Definitions corresponding to the conditions
def square_area : ℝ := 98
def ratio1_width : ℝ := 2
def ratio1_length : ℝ := 3
def ratio2_width : ℝ := 3
def ratio2_length : ℝ := 8
variable (x y : ℝ)

-- Theorem statement
theorem rectangle_areas_sum_m_n_equals (h1 : 10 * x + 22 * y = 28 * Real.sqrt 2) 
    (h2 : 6 * x ^ 2 = 24 * y ^ 2) : 
    let m := 64 in
    let n := 3 in
    m + n = 67 := by
  sorry -- Proof goes here

end rectangle_areas_sum_m_n_equals_l625_625960


namespace cost_of_hat_l625_625015

variable (H : ℝ)

-- Condition definitions
def toy_price : ℝ := 20
def num_toys : ℕ := 2
def num_hats : ℕ := 3
def initial_money : ℝ := 100
def change_received : ℝ := 30

-- Arithmetical transformations based on given conditions
def total_cost_of_toys := num_toys * toy_price
def total_money_spent := initial_money - change_received
def equation := total_cost_of_toys + num_hats * H = total_money_spent

-- The main statement to prove
theorem cost_of_hat : equation → H = 10 := by
  sorry

end cost_of_hat_l625_625015


namespace correct_simplification_l625_625891

theorem correct_simplification (x y : ℝ) (hy : y ≠ 0):
  3 * x^4 * y / (x^2 * y) = 3 * x^2 :=
by
  sorry

end correct_simplification_l625_625891


namespace arithmetic_sequence_difference_l625_625353

noncomputable def is_arithmetic_sequence (b : ℕ → ℚ) :=
  ∃ d : ℚ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_difference (b : ℕ → ℚ)
  (h1 : is_arithmetic_sequence b)
  (h2 : ∑ i in (finset.range 150).image (λ i, i + 1), b i = 150)
  (h3 : ∑ i in (finset.range 150).image (λ i, i + 151), b i = 450) :
  b 2 - b 1 = 1 / 75 :=
sorry

end arithmetic_sequence_difference_l625_625353


namespace exists_m_even_none_m_odd_forall_m_not_even_forall_m_not_odd_l625_625946

-- Define the function f
def f (x m : ℝ) : ℝ := x^2 + m * x

-- Define what it means for a function to be even
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define what it means for a function to be odd
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem exists_m_even : ∃ m : ℝ, is_even_function (λ x, f x m) :=
by {
  use 0,
  unfold is_even_function f,
  intro x,
  simp,
}

theorem none_m_odd : ¬∃ m : ℝ, is_odd_function (λ x, f x m) :=
sorry

theorem forall_m_not_even : ¬∀ m : ℝ, is_even_function (λ x, f x m) :=
sorry

theorem forall_m_not_odd : ¬∀ m : ℝ, is_odd_function (λ x, f x m) :=
sorry

end exists_m_even_none_m_odd_forall_m_not_even_forall_m_not_odd_l625_625946


namespace point_quadrant_l625_625263

theorem point_quadrant (x y : ℤ) (h1: x = 2) (h2: y = -5) : 
  (x > 0) ∧ (y < 0) :=
by {
  rw [h1, h2],
  exact and.intro (by linarith) (by linarith),
}

end point_quadrant_l625_625263


namespace ratio_proof_l625_625276

variable {A B C D E K M N : Point}
variable {BD DC BE ME BN NK : Real}

-- Conditions
axiom h1 : B ∈ Segment(C, D) 
axiom h2 : BD = 1 * DC / 3 
axiom h3 : E ∈ Segment(A, C)
axiom h4 : K ∈ Segment(A, C)
axiom h5 : E ∈ Segment(A, K)
axiom h6 : M ∈ Segment(B, E)
axiom h7 : N ∈ Segment(B, K)
axiom h8 : Segment(A, D) ∩ Segment(B, E) = {M}
axiom h9 : Segment(A, D) ∩ Segment(B, K) = {N}
axiom h10 : BM = 7/12 * BE
axiom h11 : BN = 2/5 * BK

theorem ratio_proof : MN / AD = 11 / 45 := sorry

end ratio_proof_l625_625276


namespace proof_problem_l625_625995

-- Definitions given the conditions
def S (n : ℕ) : ℚ := 3 * n^2 - 2 * n

def a (n : ℕ) : ℚ := 6 * n - 5

def b (n : ℕ) : ℚ := 3 / (a n * a (n + 1))

def T (n : ℕ) : ℚ := (Finset.range n).sum (λ i, b (i + 1))

-- The statement we need to prove
theorem proof_problem (n : ℕ) :
  (1 ≤ n) →
  | T n - (1/2) | < 1/100 ↔ n = 9 :=
sorry

end proof_problem_l625_625995


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625424

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, (n > 0 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∧ ∃ k : ℕ, n = k^2 ∧ n = 225 := 
begin
  sorry
end

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625424


namespace right_triangle_count_l625_625225

theorem right_triangle_count : 
  (set.count { (a, b) | ∃ a b : ℕ, a^2 + b^2 = (b + 1)^2 ∧ b < 100 }) = 6 :=
sorry

end right_triangle_count_l625_625225


namespace sum_even_pos_int_b_quadratic_rational_roots_l625_625979

theorem sum_even_pos_int_b_quadratic_rational_roots :
  ∑ b in {b | ∃ k : ℤ, 3 * ((49 - k^2) / 12) = b ∧ (49 - k^2) % 12 = 0 ∧ b > 0 ∧ b % 2 = 0}.to_finset = 6 :=
by
  sorry

end sum_even_pos_int_b_quadratic_rational_roots_l625_625979


namespace find_tan_alpha_plus_pi_four_l625_625643

variable (α β : ℝ)

axiom tan_add (a b : ℝ) : Mathlib.tan (a + b) = (Mathlib.tan a + Mathlib.tan b) / (1 - Mathlib.tan a * Mathlib.tan b)
axiom tan_sub (a b : ℝ) : Mathlib.tan (a - b) = (Mathlib.tan a - Mathlib.tan b) / (1 + Mathlib.tan a * Mathlib.tan b)


theorem find_tan_alpha_plus_pi_four 
    (h1 : Mathlib.tan (α + β) = 2/5)
    (h2: Mathlib.tan (β - (Real.pi / 4)) = 1/4) : 
    Mathlib.tan (α + (Real.pi / 4)) = 3/22 :=
sorry

end find_tan_alpha_plus_pi_four_l625_625643


namespace parallel_lines_perpendicular_lines_l625_625215

def line1 (m : ℝ) : ℝ × ℝ → ℝ := λ p, 2 * p.1 + (m + 1) * p.2 + 4
def line2 (m : ℝ) : ℝ × ℝ → ℝ := λ p, m * p.1 + 3 * p.2 - 6

theorem parallel_lines (m : ℝ) : 
  (∀ x y : ℝ, line1 m (x, y) = 0 → line2 m (x, y) = 0) → m = 2 :=
by
  sorry

theorem perpendicular_lines (m : ℝ) : 
  (∀ x y : ℝ, line1 m (x, y) = 0 → line2 m (x, y) = 0) → m = -3 / 5 :=
by
  sorry

end parallel_lines_perpendicular_lines_l625_625215


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625435

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ (n : ℕ), (∃ k : ℕ, n = k^2) ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, (∃ k : ℕ, m = k^2) ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m :=
sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625435


namespace find_y_l625_625470

theorem find_y (y : ℝ) (h : 2 * y / 3 = 30) : y = 45 :=
by
  sorry

end find_y_l625_625470


namespace solve_sqrt_equation_l625_625347

theorem solve_sqrt_equation (x : ℝ) (hx : real.sqrt x ≠ 1) :
  real.sqrt x + 1 = 1 / (real.sqrt x - 1) → x = 2 := by
  sorry

end solve_sqrt_equation_l625_625347


namespace prime_appears_in_seven_or_more_l625_625736

def is_rational_root (p q r : ℕ) (x : ℚ) : Prop :=
  p * x^2 + q * x + r = 0

def S : set (ℕ × ℕ × ℕ) :=
  { (p, q, r) | ∃ x : ℚ, is_rational_root p q r x }

def count_prime_occurrences_in_S (prime : ℕ) : ℕ :=
  {t ∈ S | t.1 = prime ∨ t.2 = prime ∨ t.3 = prime}.card

theorem prime_appears_in_seven_or_more (prime := 2) :
  count_prime_occurrences_in_S prime ≥ 7 :=
sorry

end prime_appears_in_seven_or_more_l625_625736


namespace scale_model_height_l625_625929

/-- 
Given a scale model ratio and the actual height of the skyscraper in feet,
we can deduce the height of the model in inches.
-/
theorem scale_model_height
  (scale_ratio : ℕ := 25)
  (actual_height_feet : ℕ := 1250) :
  (actual_height_feet / scale_ratio) * 12 = 600 :=
by 
  sorry

end scale_model_height_l625_625929


namespace sequence_a_1000_l625_625707

noncomputable def a : ℕ → ℤ
| 0 := 2010
| 1 := 2011
| (n + 2) := (λ n, (n + 1) / 2) (2 * n + 3 - a n - a (n + 1))

theorem sequence_a_1000 : a 1000 = 2343 := sorry

end sequence_a_1000_l625_625707


namespace marked_cells_at_least_one_tile_l625_625319

noncomputable def cells_in_tile : ℕ := sorry -- Define the number of cells in one tile (n cells)

def total_tiles := 2009

def is_odd_covered (cover_count : ℕ) : Prop := cover_count % 2 = 1

def marked_cells_count (grid : ℕ → ℕ → ℕ) : ℕ :=
  (grid.cells).countp (λ cell_cover_count, is_odd_covered cell_cover_count)

theorem marked_cells_at_least_one_tile 
  (grid : ℕ → ℕ → ℕ) -- Assume a grid function that gives the count of coverings for any cell
  (total_covered_count : ∀ i j : ℕ, grid i j → total_tiles) :
  marked_cells_count grid ≥ cells_in_tile :=
begin
  sorry
end

end marked_cells_at_least_one_tile_l625_625319


namespace Elise_savings_l625_625150

theorem Elise_savings :
  let initial_dollars := 8
  let saved_euros := 11
  let euro_to_dollar := 1.18
  let comic_cost := 2
  let puzzle_pounds := 13
  let pound_to_dollar := 1.38
  let euros_to_dollars := saved_euros * euro_to_dollar
  let total_after_saving := initial_dollars + euros_to_dollars
  let after_comic := total_after_saving - comic_cost
  let pounds_to_dollars := puzzle_pounds * pound_to_dollar
  let final_amount := after_comic - pounds_to_dollars
  final_amount = 1.04 :=
by
  sorry

end Elise_savings_l625_625150


namespace number_of_truthful_dwarfs_l625_625604

def dwarf_condition := 
  ∀ (dwarfs : ℕ) (truthful_dwarfs : ℕ) (lying_dwarfs : ℕ),
    dwarfs = 10 ∧ 
    (∀ n, n ∈ {truthful_dwarfs, lying_dwarfs} -> n ≥ 0) ∧ 
    truthful_dwarfs + lying_dwarfs = dwarfs ∧
    truthful_dwarfs + 2 * lying_dwarfs = 16

theorem number_of_truthful_dwarfs : ∃ (truthful_dwarfs : ℕ), (dwarf_condition ∧ truthful_dwarfs = 4) :=
by {
  let dwarfs := 10,
  let lying_dwarfs := 6,
  let truthful_dwarfs := dwarfs - lying_dwarfs,
  have h: truthful_dwarfs = 4,
  { calc
    truthful_dwarfs = dwarfs - lying_dwarfs : by rfl
    ... = 10 - 6 : by rfl
    ... = 4 : by rfl },
  existsi (4 : ℕ),
  refine ⟨_, ⟨dwarfs, truthful_dwarfs, lying_dwarfs, rfl, _, _, _⟩⟩,
  -- Now we can provide the additional details for lean to understand the conditions hold
  {
    intros n hn,
    simp,
    exact hn
  },
  {
    exact add_comm 6 4
  },
  {
    dsimp,
    ring,
  },
  {
    exact h,
  }
  -- Skip the actual proof with sorry
  sorry
}

end number_of_truthful_dwarfs_l625_625604


namespace order_a_b_c_l625_625988

def a : ℝ := 2^(2 * real.log 3)
def b : ℝ := 3^(3 * real.log 2)
def c : ℝ := 5^(real.log 5)

theorem order_a_b_c : a < b ∧ b < c :=
by {
  sorry -- Proof to be completed
}

end order_a_b_c_l625_625988


namespace number_of_valid_integers_l625_625956

theorem number_of_valid_integers (n : ℕ) (h1 : 1000 ≤ n) (h2 : n < 10000) :
  (let M_4 := n -- interpreting as base 4 could be implicit
       M_7 := n -- interpreting as base 7 could be implicit
       T := M_4 + M_7 in
   T % 100 = (3 * n) % 100) ↔ n = 32 := 
sorry

end number_of_valid_integers_l625_625956


namespace rhonda_investment_interest_rate_l625_625332

theorem rhonda_investment_interest_rate :
  ∃ r : ℝ, r ≈ 0.03781 ∧ 
    let total_investment := 4725 in
    let invested_at_11_percent := 1925 in
    let other_investment := total_investment - invested_at_11_percent in
    let interest_at_11_percent := invested_at_11_percent * 0.11 in
    let interest_at_other := other_investment * r in
    interest_at_11_percent = 2 * interest_at_other :=
by 
  have r := 105.875 / 2800
  exists r
  have interest_at_11_percent := 1925 * 0.11
  have interest_at_other := 2800 * r
  have : interest_at_11_percent = 2 * interest_at_other,
  sorry

end rhonda_investment_interest_rate_l625_625332


namespace calculate_abc_sum_l625_625137

-- Defining the conditions for Danica's trip
variable (a b c : ℕ) -- Representing the digits as natural numbers

-- Initial conditions
axiom condition1 : a ≥ 1
axiom condition2 : a + b + c ≤ 9

-- Calculating the odometer change condition
axiom distance_condition : 
  ∃ n : ℕ, n * 45 = 100 * c + 10 * a + b - (100 * a + 10 * b + c)

-- Theorem statement that abc + bca + cab must equal 999
theorem calculate_abc_sum : 
  ∃ (abc bca cab : ℕ), abc = 100 * a + 10 * b + c ∧ bca = 100 * b + 10 * c + a ∧ cab = 100 * c + 10 * a + b ∧ abc + bca + cab = 999 := 
begin 
  sorry -- Proof is not required as per the instruction
end

end calculate_abc_sum_l625_625137


namespace cars_needed_to_double_earnings_l625_625478

-- Define the conditions
def baseSalary : Int := 1000
def commissionPerCar : Int := 200
def januaryEarnings : Int := 1800

-- The proof goal
theorem cars_needed_to_double_earnings : 
  ∃ (carsSoldInFeb : Int), 
    1000 + commissionPerCar * carsSoldInFeb = 2 * januaryEarnings :=
by
  sorry

end cars_needed_to_double_earnings_l625_625478


namespace find_MP_l625_625095

variables (A B C D K N P M : Type)
variables (AD AB CD BC KD MP MN : ℝ)

-- let trapezoid ABCD, where AD, AB, CD, and BC are the sides
-- and KD = AD - CD following from the given condition KD = 8
def trapezoid (AD AB CD BC KD : ℝ) : Prop :=
AD = 21 ∧ AB = 10 ∧ CD = 10 ∧ BC = 21 ∧ KD = 8

-- let N be the midpoint of AB and geometry intersection definitions.
-- We specifically need to find that MP = 35 under the given conditions.

theorem find_MP (h : trapezoid AD AB CD BC KD) (hN : midpoint N A B)
  (hK : projection K C AD) (hMN : is_midpoint MN AD BC) :
  MP = 35 :=
sorry

end find_MP_l625_625095


namespace part_a_l625_625897

variable (A B : Matrix (Fin 3) (Fin 3) ℝ)

theorem part_a (hA_gt_B : Matrix.rank A > Matrix.rank B) :
  Matrix.rank (A ⬝ A) ≥ Matrix.rank (B ⬝ B) :=
sorry

end part_a_l625_625897


namespace number_of_truthful_dwarfs_l625_625606

def dwarf_condition := 
  ∀ (dwarfs : ℕ) (truthful_dwarfs : ℕ) (lying_dwarfs : ℕ),
    dwarfs = 10 ∧ 
    (∀ n, n ∈ {truthful_dwarfs, lying_dwarfs} -> n ≥ 0) ∧ 
    truthful_dwarfs + lying_dwarfs = dwarfs ∧
    truthful_dwarfs + 2 * lying_dwarfs = 16

theorem number_of_truthful_dwarfs : ∃ (truthful_dwarfs : ℕ), (dwarf_condition ∧ truthful_dwarfs = 4) :=
by {
  let dwarfs := 10,
  let lying_dwarfs := 6,
  let truthful_dwarfs := dwarfs - lying_dwarfs,
  have h: truthful_dwarfs = 4,
  { calc
    truthful_dwarfs = dwarfs - lying_dwarfs : by rfl
    ... = 10 - 6 : by rfl
    ... = 4 : by rfl },
  existsi (4 : ℕ),
  refine ⟨_, ⟨dwarfs, truthful_dwarfs, lying_dwarfs, rfl, _, _, _⟩⟩,
  -- Now we can provide the additional details for lean to understand the conditions hold
  {
    intros n hn,
    simp,
    exact hn
  },
  {
    exact add_comm 6 4
  },
  {
    dsimp,
    ring,
  },
  {
    exact h,
  }
  -- Skip the actual proof with sorry
  sorry
}

end number_of_truthful_dwarfs_l625_625606


namespace quadratic_eq_mn_sum_l625_625163

theorem quadratic_eq_mn_sum (m n : ℤ) 
  (h1 : m - 1 = 2) 
  (h2 : 16 + 4 * n = 0) 
  : m + n = -1 :=
by
  sorry

end quadratic_eq_mn_sum_l625_625163


namespace line_passes_second_and_third_quadrants_l625_625634

theorem line_passes_second_and_third_quadrants 
  (a b c p : ℝ)
  (h1 : a * b * c ≠ 0)
  (h2 : (a + b) / c = p)
  (h3 : (b + c) / a = p)
  (h4 : (c + a) / b = p) :
  ∀ (x y : ℝ), y = p * x + p → 
  ((x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
sorry

end line_passes_second_and_third_quadrants_l625_625634


namespace flight_time_correct_bounce_height_correct_l625_625911
open System

-- Definitions based on the conditions:
def L : ℝ := 5
def alpha : ℝ := Real.pi / 4  -- 45 degrees converted to radians
def g : ℝ := 10
def tau : ℝ := Real.sqrt 2 / 2

-- Definition of the flight time T
def flight_time(T : ℝ) : Prop := T = 2 * Real.sqrt (L * Real.tan(alpha) / g)

-- Definition of the height H where the ball bounced
def bounce_height(H : ℝ) : Prop := H = (1 / 2) * g * tau ^ 2

-- The theorem to prove that the flight time is approximately 1.4 seconds
theorem flight_time_correct : ∃ T : ℝ, flight_time(T) ∧ Real.abs(T - 1.4) < 0.01 := sorry

-- The theorem to prove the height where the ball bounced is 2.5 meters
theorem bounce_height_correct : ∃ H : ℝ, bounce_height(H) ∧ Real.abs(H - 2.5) < 0.01 := sorry

end flight_time_correct_bounce_height_correct_l625_625911


namespace increasing_only_ln_x_plus_2_l625_625471

theorem increasing_only_ln_x_plus_2 :
  ∀ (x : ℝ), (0 < x) → (∀ (y : ℝ → ℝ), 
    (y = λ x, Real.log (x + 2)) ∨ 
    (y = λ x, -Real.sqrt (x + 1)) ∨ 
    (y = λ x, (1/2) ^ x) ∨ 
    (y = λ x, x + 1/x) → 
    (y = λ x, Real.log (x + 2)) →
    (0 < (Real.log (x + 2)).deriv x)) :=
begin
  sorry -- proof not required
end

end increasing_only_ln_x_plus_2_l625_625471


namespace multiplication_result_l625_625049

theorem multiplication_result :
  (500 ^ 50) * (2 ^ 100) = 10 ^ 75 :=
by
  sorry

end multiplication_result_l625_625049


namespace measurable_masses_l625_625511

theorem measurable_masses (k : ℤ) (h : -121 ≤ k ∧ k ≤ 121) : 
  ∃ (a b c d e : ℤ), k = a * 1 + b * 3 + c * 9 + d * 27 + e * 81 ∧ 
  (a = -1 ∨ a = 0 ∨ a = 1) ∧
  (b = -1 ∨ b = 0 ∨ b = 1) ∧
  (c = -1 ∨ c = 0 ∨ c = 1) ∧
  (d = -1 ∨ d = 0 ∨ d = 1) ∧
  (e = -1 ∨ e = 0 ∨ e = 1) :=
sorry

end measurable_masses_l625_625511


namespace determine_a_l625_625813

noncomputable theory
open Real

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * x + 1

def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2

theorem determine_a (a : ℝ) (ha : a ≠ 0) (h_tangent : f' a 1 = -1) :
  a = -1 :=
by
  sorry

end determine_a_l625_625813


namespace range_of_slope_through_focus_l625_625214

noncomputable def slope_range_of_line_through_right_focus {x y : ℝ} : set ℝ :=
  { m : ℝ |
    ∃ x y, (x^2 / 12 - y^2 / 4 = 1) ∧ ∃ F : ℝ × ℝ, F.1 = sqrt 12 ∧ m = (F.2 - y) / (F.1 - x) }

theorem range_of_slope_through_focus (m : ℝ) :
  m ∈ slope_range_of_line_through_right_focus ↔ - (sqrt 3 / 3) ≤ m ∧ m ≤ sqrt 3 / 3 :=
sorry

end range_of_slope_through_focus_l625_625214


namespace paul_spent_252_dollars_l625_625324

noncomputable def total_cost_before_discounts : ℝ :=
  let dress_shirts := 4 * 15
  let pants := 2 * 40
  let suit := 150
  let sweaters := 2 * 30
  dress_shirts + pants + suit + sweaters

noncomputable def store_discount : ℝ := 0.20

noncomputable def coupon_discount : ℝ := 0.10

noncomputable def total_cost_after_store_discount : ℝ :=
  let initial_total := total_cost_before_discounts
  initial_total - store_discount * initial_total

noncomputable def final_total : ℝ :=
  let intermediate_total := total_cost_after_store_discount
  intermediate_total - coupon_discount * intermediate_total

theorem paul_spent_252_dollars :
  final_total = 252 := by
  sorry

end paul_spent_252_dollars_l625_625324


namespace rounding_4995000_l625_625908

theorem rounding_4995000 :
  round_nearest_million 4995000 = 5000000 :=
by
  -- Definition of the rounding function would go here
  sorry

end rounding_4995000_l625_625908


namespace line_and_circle_intersection_l625_625721

-- Statement of the problem in Lean 4
theorem line_and_circle_intersection
  (parametric_eqns_line : ∀ t : ℝ, (ℝ × ℝ) := λ t, (3 - (√2 / 2) * t, √5 + (√2 / 2) * t))
  (polar_eqn_circle : ℝ → ℝ := λ θ, 2 * √5 * sin θ) :
  let l_cartesian (x y : ℝ) := x + y - 3 - √5 = 0,
      c_cartesian (x y : ℝ) := x^2 + (y - √5)^2 = 5,
      P := (3, √5),
      equation_subst t := (3 - (√2 / 2) * t)^2 + ((√5 + (√2 / 2) * t) - √5)^2 in
  (∀ x y : ℝ, l_cartesian x y ↔ ∃ t : ℝ, (x, y) = parametric_eqns_line t) ∧
  (∀ x y : ℝ, c_cartesian x y ↔ ∃ θ : ℝ, (x^2 + y^2 = (polar_eqn_circle θ)^2)) ∧
  ∃ t1 t2 : ℝ, equation_subst t1 = 5 ∧ equation_subst t2 = 5 ∧ (|t1| + |t2| = 3 * √2) :=
sorry

end line_and_circle_intersection_l625_625721


namespace least_value_l625_625045

theorem least_value : ∀ x y : ℝ, (xy + 1)^2 + (x - y)^2 ≥ 1 :=
by
  sorry

end least_value_l625_625045


namespace probability_top_card_is_king_correct_l625_625936

noncomputable def probability_top_card_is_king (total_cards kings : ℕ) : ℚ :=
kings / total_cards

theorem probability_top_card_is_king_correct :
  probability_top_card_is_king 52 4 = 1 / 13 :=
by
  sorry

end probability_top_card_is_king_correct_l625_625936


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625438

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ (n : ℕ), (∃ k : ℕ, n = k^2) ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, (∃ k : ℕ, m = k^2) ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m :=
sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625438


namespace intersection_of_sets_l625_625379

def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x | 2 < x ∧ x < 4 }

theorem intersection_of_sets : A ∩ B = { x | 2 < x ∧ x ≤ 3 } := 
by 
  sorry

end intersection_of_sets_l625_625379


namespace inscribed_triangle_perimeter_geq_half_l625_625330

theorem inscribed_triangle_perimeter_geq_half (a : ℝ) (s' : ℝ) (h_a_pos : a > 0) 
  (h_equilateral : ∀ (A B C : Type) (a b c : A), a = b ∧ b = c ∧ c = a) :
  2 * s' >= (3 * a) / 2 :=
by
  sorry

end inscribed_triangle_perimeter_geq_half_l625_625330


namespace sum_of_heights_of_30_students_l625_625070

noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_of_heights_of_30_students:
  ∃ (a₁ d : ℝ),
    (sum_arithmetic_sequence a₁ d 10 = 12.5) ∧
    (sum_arithmetic_sequence a₁ d 20 = 26.5) ∧
    (sum_arithmetic_sequence a₁ d 30 = 42) :=
begin
  sorry
end

end sum_of_heights_of_30_students_l625_625070


namespace min_distance_from_curve_to_line_l625_625242

open Real

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  abs ((x₁ - x₂) + (y₁ - y₂)**2).sqrt / ((1**2 + (-1)**2)).sqrt  

theorem min_distance_from_curve_to_line : 
  ∀ P : ℝ × ℝ, P.2 = P.1^2 - log P.1 → ∃ Q : ℝ, Q = sqrt 2 :=
by
  sorry

end min_distance_from_curve_to_line_l625_625242


namespace range_of_a_is_0_to_2_l625_625195

noncomputable def is_decreasing (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → f x ≥ f y

theorem range_of_a_is_0_to_2 (f : ℝ → ℝ) (a : ℝ) (h_decreasing : is_decreasing f) : a ∈ set.Icc 0 2 :=
sorry

end range_of_a_is_0_to_2_l625_625195


namespace height_of_the_barbed_wire_is_3_l625_625357

-- Define constants used in the problem
def area_of_square_field : ℝ := 3136
def cost_per_meter : ℝ := 1.40
def width_of_gate : ℝ := 1
def number_of_gates : ℕ := 2
def total_cost : ℝ := 932.40

-- Calculate the side length and reduced perimeter
def side_length : ℝ := Real.sqrt area_of_square_field
def perimeter_minus_gates : ℝ := 4 * side_length - number_of_gates * width_of_gate

-- Define the equation representing the total cost
def height_of_barbed_wire (h : ℝ) : Prop :=
  cost_per_meter * perimeter_minus_gates * h = total_cost

-- The statement of the proof problem
theorem height_of_the_barbed_wire_is_3 : height_of_barbed_wire 3 :=
  sorry

end height_of_the_barbed_wire_is_3_l625_625357


namespace least_common_multiple_of_20_45_75_l625_625866

theorem least_common_multiple_of_20_45_75 :
  Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
sorry

end least_common_multiple_of_20_45_75_l625_625866


namespace hyperbola_properties_l625_625158

noncomputable def foci (a b c : ℝ) := (-(Real.sqrt c), 0), (Real.sqrt c, 0)
noncomputable def lengths_of_axes (a b : ℝ) := (2 * a, 2 * b)
noncomputable def eccentricity (a c : ℝ) := c / a
noncomputable def equations_of_asymptotes (a b : ℝ) := (fun x : ℝ => (b / a) * x), (fun x : ℝ => -(b / a) * x)

theorem hyperbola_properties :
  ∃ a b c : ℝ,
    (16 * a ^ 2 - 25 * b ^ 2 = 400) ∧
    (a = 5) ∧
    (b = 4) ∧
    (c = Real.sqrt (a ^ 2 + b ^ 2)) ∧
    (foci a b c = (-(√41), 0), (√41, 0)) ∧
    (lengths_of_axes a b = (10, 8)) ∧
    (eccentricity a c = √41 / 5) ∧
    (equations_of_asymptotes a b = (fun x => (4 / 5) * x, fun x => -(4 / 5) * x)) :=
begin
 sorry
end

end hyperbola_properties_l625_625158


namespace trailing_zeros_mod_100_l625_625747

def trailing_zeros_product_factorials (n : ℕ) : ℕ :=
  let count_factors (p k : ℕ) : ℕ :=  if k < p then 0 else k / p + count_factors p (k / p)
  count_factors 5 n

theorem trailing_zeros_mod_100 : 
  let M := trailing_zeros_product_factorials 50
  M % 100 = 12 :=
by
  let M := trailing_zeros_product_factorials 50
  have hM : M = 12 := by sorry
  exact congrArg (λ x, x % 100) hM

end trailing_zeros_mod_100_l625_625747


namespace greatest_integer_radius_of_circle_l625_625698

theorem greatest_integer_radius_of_circle (r : ℕ) (A : ℝ) (hA : A < 80 * Real.pi) :
  r <= 8 ∧ r * r < 80 :=
sorry

end greatest_integer_radius_of_circle_l625_625698


namespace min_vertical_distance_l625_625365

theorem min_vertical_distance :
  ∃ (d : ℝ), ∀ (x : ℝ),
    (y1 x = |x - 1| ∧ y2 x = -x^2 - 4*x - 3) ∧ 
    (d = infi (λ x, abs (y1 x - y2 x))) ∧
    (d = 7 / 4) :=
by
  let y1 := λ x : ℝ, abs (x - 1)
  let y2 := λ x : ℝ, -x^2 - 4 * x - 3
  exists 7 / 4
  simp
  sorry

end min_vertical_distance_l625_625365


namespace problem_1_problem_2_l625_625619

-- Problem 1: Prove sqrt((-16 : ℝ)^2) - real.cbrt (-8) = 18.
theorem problem_1 : real.sqrt ((-16 : ℝ)^2) - real.cbrt (-8) = 18 := 
sorry

-- Problem 2: Prove (real.sqrt 12 - real.sqrt 27) * real.sqrt 3 = -3.
theorem problem_2 : (real.sqrt 12 - real.sqrt 27) * real.sqrt 3 = -3 := 
sorry

end problem_1_problem_2_l625_625619


namespace barbara_wins_gameA_l625_625902

noncomputable def gameA_winning_strategy : Prop :=
∃ (has_winning_strategy : (ℤ → ℝ) → Prop),
  has_winning_strategy (fun n => n : ℤ → ℝ)

theorem barbara_wins_gameA :
  gameA_winning_strategy := sorry

end barbara_wins_gameA_l625_625902


namespace range_of_x_satisfying_inequality_l625_625659

def f (x : ℝ) : ℝ := (x - 1) ^ 4 + 2 * |x - 1|

theorem range_of_x_satisfying_inequality :
  {x : ℝ | f x > f (2 * x)} = {x : ℝ | 0 < x ∧ x < (2 : ℝ) / 3} :=
by
  sorry

end range_of_x_satisfying_inequality_l625_625659


namespace sequence_product_simplification_l625_625795

theorem sequence_product_simplification :
  (∏ n in Finset.range ((3036 / 6) - 1), (6 * (n + 1) + 6) / (6 * (n + 1))) = 506 :=
by
  sorry

end sequence_product_simplification_l625_625795


namespace midpoints_form_parallelogram_l625_625638

-- Assume a plane geometry setting where points and lines exist
variables {Point : Type}
variables {ABCD : Point → Point → Point → Point → Prop}
variables (midpoint : Point → Point → Point)
variables {M N P Q A B C D : Point}

-- Conditions
axiom quad_exists : ∃A B C D, ABCD A B C D
axiom M_midpoint_AB : M = midpoint A B
axiom N_midpoint_BC : N = midpoint B C
axiom P_midpoint_CD : P = midpoint C D
axiom Q_midpoint_DA : Q = midpoint D A

-- Proof that MNPQ is a parallelogram based on midpoints
theorem midpoints_form_parallelogram (h : ABCD A B C D) : 
  is_parallelogram (midpoint A B) (midpoint B C) (midpoint C D) (midpoint D A) :=
sorry

end midpoints_form_parallelogram_l625_625638


namespace alice_average_speed_l625_625944

/-- Alice's journey conditions:
  - The total distance travelled is 540 miles
  - The total time taken is 9.75 hours
  Prove that Alice's average speed for her entire journey is 55.38 miles per hour -/
theorem alice_average_speed (total_distance : ℝ) (total_time : ℝ) (h1 : total_distance = 540) (h2 : total_time = 9.75) :
  total_distance / total_time = 55.38 :=
by
  rw [h1, h2]
  have h : 540 / 9.75 = 55.38461538461539 := sorry
  linarith

end alice_average_speed_l625_625944


namespace fraction_operation_correct_l625_625945

theorem fraction_operation_correct 
  (a b : ℝ) : 
  (0.2 * (3 * a + 10 * b) = 6 * a + 20 * b) → 
  (0.1 * (2 * a + 5 * b) = 2 * a + 5 * b) →
  (∀ c : ℝ, c ≠ 0 → (a / b = (a * c) / (b * c))) ∨
  (∀ x y : ℝ, ((x - y) / (x + y) ≠ (y - x) / (x - y))) ∨
  (∀ x : ℝ, (x + x * x * x + x * y ≠ 1 / x * x)) →
  ((0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b)) :=
sorry

end fraction_operation_correct_l625_625945


namespace problem_solution_l625_625826

theorem problem_solution (a b : ℝ) (h₀ : a ≠ 1) (h₁ : a ≠ 0)
  (h : {a, b / a, 1} = {a^2, a + b, 0}) : a ^ 2009 + b ^ 2009 = -1 :=
by
  sorry

end problem_solution_l625_625826


namespace angle_same_terminal_side_l625_625539

theorem angle_same_terminal_side (k : ℤ) : 
  ∃ θ : ℝ, θ = 375 ∧ (dfrac (π) (12) + 2 * k * π) % (2 * π) = θ % 360 :=
by
  sorry

end angle_same_terminal_side_l625_625539


namespace monotonicity_of_g_range_of_a_l625_625166

noncomputable def f (x : ℝ) : ℝ := real.exp (x / 2) - x / 4

noncomputable def f' (x : ℝ) : ℝ := (1 / 2) * real.exp (x / 2) - 1 / 4

noncomputable def g (x : ℝ) : ℝ := (x + 1) * f' x

noncomputable def F (x : ℝ) (a : ℝ) : ℝ := real.log (x + 1) - a * f x + 4

theorem monotonicity_of_g : ∀ x > -1, 0 < g' x :=
    by sorry

theorem range_of_a (a : ℝ) : ∀ x, 0 < x →
  ∀ a > 4, ¬ ∃ x, F x a = 0 :=
    by sorry

end monotonicity_of_g_range_of_a_l625_625166


namespace maize_donation_amount_l625_625943

-- Definitions and Conditions
def monthly_storage : ℕ := 1
def months_in_year : ℕ := 12
def years : ℕ := 2
def stolen_tonnes : ℕ := 5
def total_tonnes_at_end : ℕ := 27

-- Theorem statement
theorem maize_donation_amount :
  let total_stored := monthly_storage * (months_in_year * years)
  let remaining_after_theft := total_stored - stolen_tonnes
  total_tonnes_at_end - remaining_after_theft = 8 :=
by
  -- This part is just the statement, hence we use sorry to omit the proof.
  sorry

end maize_donation_amount_l625_625943


namespace ellipse_equation_l625_625000

theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (h1 : a > b)
    (h2 : e = (Real.sqrt 3) / 2)
    (h3 : b = a / 2)
    (h4 : ∀ (x y : ℝ), x = 0 ∨ x = 4 * a / 5 → y = x - a / 2 →
           ((x = 0 → y = -a / 2) ∧ (x = (4 * a) / 5 → y = (3 * a) / 10)))
    (h5 : ∀ O A B : (ℝ × ℝ), (O = (0, 0)) →
          (A = (0, -a / 2) ∨ A = ((4 * a) / 5, (3 * a) / 10)) ∧
          (B = ((4 * a) / 5, (3 * a) / 10) ∨ B = (0, -a / 2)) →
          (O.1 * A.1 + O.2 * A.2) * (O.1 * B.1 + O.2 * B.2) = (32 / 5) * Real.cot (Real.angle O A B)) :
  ∀ (x y : ℝ), (x^2 / 16) + (y^2 / 4) = 1 :=
by 
  sorry

end ellipse_equation_l625_625000


namespace initial_price_reduction_l625_625162

theorem initial_price_reduction (x : ℝ) (hx : 0 < x) :
  let new_price_1 := x * 0.8,
      new_price_2 := new_price_1 * 0.85,
      new_price_3 := new_price_2 * 0.9
  in x - new_price_3 = 0.388 * x :=
by
  sorry

end initial_price_reduction_l625_625162


namespace symmetric_line_with_respect_to_x_axis_l625_625003

theorem symmetric_line_with_respect_to_x_axis (x y : ℝ) :
  let original_line := 3 * x - 4 * y + 5 = 0,
      symmetric_line := 3 * x + 4 * y + 5 = 0 in
  ∀ (x y : ℝ), original_line → symmetric_line :=
by sorry

end symmetric_line_with_respect_to_x_axis_l625_625003


namespace sum_squares_fraction_l625_625760

theorem sum_squares_fraction (y : Fin 50 → ℝ) (h1 : ∑ i, y i = 2) (h2 : ∑ i, y i / (2 - y i) = 2) :
  ∑ i, y i ^ 2 / (2 - y i) = 2 :=
by
  sorry

end sum_squares_fraction_l625_625760


namespace find_focus_with_larger_x_l625_625958

def hyperbola_foci_coordinates : Prop :=
  let center := (5, 10)
  let a := 7
  let b := 3
  let c := Real.sqrt (a^2 + b^2)
  let focus1 := (5 + c, 10)
  let focus2 := (5 - c, 10)
  focus1 = (5 + Real.sqrt 58, 10)
  
theorem find_focus_with_larger_x : hyperbola_foci_coordinates := 
  by
    sorry

end find_focus_with_larger_x_l625_625958


namespace minimal_removals_l625_625683

theorem minimal_removals (N : ℕ) (hN : N = 2023) : 
  ∃ S ⊆ finset.range (N + 1), 
    finset.card S = 43 ∧ 
    ∀ a b ∈ (finset.range (N + 1) \ S), a ≠ b →
      a * b ∉ (finset.range (N + 1) \ S) := 
sorry

end minimal_removals_l625_625683


namespace even_four_digit_strict_inc_count_l625_625684

theorem even_four_digit_strict_inc_count :
  let numbers := { (a, b, c, d) | 1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ (d = 4 ∨ d = 6 ∨ d = 8) ∧ d % 2 = 0 } in
  ∑ d in {4, 6, 8}, (if d = 4 then 1 else if d = 6 then (Nat.choose 5 3) else (Nat.choose 7 3)) = 46 :=
by
  sorry

end even_four_digit_strict_inc_count_l625_625684


namespace min_value_z_l625_625185

variable (x y : ℝ)

theorem min_value_z : ∃ (x y : ℝ), 2 * x + 3 * y = 9 :=
sorry

end min_value_z_l625_625185


namespace point_in_quadrant_l625_625184

theorem point_in_quadrant (a b : ℝ) (h1 : a - b > 0) (h2 : a * b < 0) : 
  (a > 0 ∧ b < 0) ∧ ¬(a > 0 ∧ b > 0) ∧ ¬(a < 0 ∧ b > 0) ∧ ¬(a < 0 ∧ b < 0) := 
by 
  sorry

end point_in_quadrant_l625_625184


namespace Frank_read_books_l625_625627

noncomputable def books_read (total_days : ℕ) (days_per_book : ℕ) : ℕ :=
total_days / days_per_book

theorem Frank_read_books : books_read 492 12 = 41 := by
  sorry

end Frank_read_books_l625_625627


namespace complex_inequality_l625_625754

noncomputable theory
open Complex

theorem complex_inequality (a b c : ℂ) (m n : ℝ) 
  (h1 : |a + b| = m) 
  (h2 : |a - b| = n) 
  (h3 : m * n ≠ 0) : 
  max (|a * c + b|) (|a + b * c|) ≥ (m * n) / sqrt (m * m + n * n) :=
sorry

end complex_inequality_l625_625754


namespace expectedValue17Seconds_l625_625843

noncomputable def TimSequence : Nat → Nat
| 0 => 4  -- Initial figure is a square (4 sides)
| n + 1 =>
  if TimSequence n = 4 then if Nat.random % 2 = 0 then 6 else 8  -- Change from square to hexagon or octagon
  else if TimSequence n = 6 then if Nat.random % 2 = 0 then 4 else 8  -- Change from hexagon to square or octagon
  else if TimSequence n = 8 then if Nat.random % 2 = 0 then 4 else 6  -- Change from octagon to square or hexagon
  else 0  -- unreachable

def timeToCreate (fig : Nat) : Nat :=
  if fig = 4 then 4 else if fig = 6 then 6 else if fig = 8 then 8 else 0 -- time to make each figure

noncomputable def totalTime : Nat → Nat
| 0 => timeToCreate (TimSequence 0)
| n + 1 => totalTime n + timeToCreate (TimSequence (n + 1))

theorem expectedValue17Seconds : 
  ∃ n : Nat, totalTime n = 17 ∧ (TimSequence n = 4 ∨ TimSequence n = 6 ∨ TimSequence n = 8) ∧ 
  (TimSequence n = 7 := 7) :=
sorry

end expectedValue17Seconds_l625_625843


namespace housewife_spent_approximately_l625_625508

theorem housewife_spent_approximately :
  ∀ (S : ℝ) (percentage_saved : ℝ), 
    S = 2.75 → 
    percentage_saved = 12.087912087912088 → 
    let P := (S * 100) / percentage_saved in
    P ≈ 22.75 :=
by
  intros S percentage_saved hS hPS
  let P := (S * 100) / percentage_saved
  have : S = 2.75 := hS
  have : percentage_saved = 12.087912087912088 := hPS
  sorry

end housewife_spent_approximately_l625_625508


namespace rectangle_area_l625_625933

def triangle_sides := (7 : ℝ, 9 : ℝ, 10 : ℝ)
def rectangle_perimeter (L W : ℝ) := 2 * (L + W)

theorem rectangle_area :
  let L := (2 : ℝ) in
  let W := (26 / 6 : ℝ) in
  rectangle_perimeter (2 * W) W = 26 ∧ (L * W) = (338 / 9) :=
by
  sorry

end rectangle_area_l625_625933


namespace min_value_of_f_l625_625666

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin (Real.pi * x) - Real.cos (Real.pi * x) + 2) / Real.sqrt x

theorem min_value_of_f : 
  ∀ x : ℝ, (1 / 4) ≤ x ∧ x ≤ (5 / 4) → 
  ∃ y : ℝ, y = (4 * Real.sqrt 5 / 5) ∧ 
           ∀ z : ℝ, (1 / 4) ≤ z ∧ z ≤ (5 / 4) → f(y) ≤ f(z) := 
by
  sorry

end min_value_of_f_l625_625666


namespace countless_lines_through_one_point_l625_625028

theorem countless_lines_through_one_point :
  ∀ (P : Point), ∃ (S : Set Line), (S = { l | P ∈ l }) ∧ (∞ = Cardinal.mk S) :=
sorry

end countless_lines_through_one_point_l625_625028


namespace youngest_child_age_is_3_l625_625903

noncomputable def family_age_problem : Prop :=
  ∃ (age_diff_2 : ℕ) (age_10_years_ago : ℕ) (new_family_members : ℕ) (same_present_avg_age : ℕ) (youngest_child_age : ℕ),
    age_diff_2 = 2 ∧
    age_10_years_ago = 4 * 24 ∧
    new_family_members = 2 ∧
    same_present_avg_age = 24 ∧
    youngest_child_age = 3 ∧
    (96 + 4 * 10 + (youngest_child_age + (youngest_child_age + age_diff_2)) = 6 * same_present_avg_age)

theorem youngest_child_age_is_3 : family_age_problem := sorry

end youngest_child_age_is_3_l625_625903


namespace three_divides_two_pow_n_plus_one_l625_625985

theorem three_divides_two_pow_n_plus_one (n : ℕ) (hn : n > 0) : 
  (3 ∣ 2^n + 1) ↔ Odd n := 
sorry

end three_divides_two_pow_n_plus_one_l625_625985


namespace problem_l625_625641

/-
 Given an ellipse \(\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1 \) and a circle with radius equal to the semi-minor axis of the ellipse,
 and a tangent line \(x - y + \sqrt{6} = 0\), if the eccentricity of the ellipse is \(\frac{1}{2}\):
 (1) Prove the ellipse equation is \(\frac{x^2}{4} + \frac{y^2}{3} = 1\).
 (2) Prove that if a line \( y = kx + m \) intersects the ellipse at points \( A \) and \( B \), and \( k_{OA} \cdot k_{OB} = -\frac{3}{4} \),
 the area of \(\triangle AOB\) is \(\sqrt{3}\).
-/

-- Definitions of the conditions and the problem statement
def ellipse (x y : ℝ) (a b : ℝ) := x^2 / a^2 + y^2 / b^2 = 1
def circle (x y radius : ℝ) := x^2 + y^2 = radius^2
def line (x y k m : ℝ) := y = k * x + m
def condition_1 := ellipse x y 2 (real.sqrt 3)
def condition_2 := circle x y (real.sqrt 3)

theorem problem :
  let a := 2 in let b := real.sqrt 3 in
  (∀ x y : ℝ, ellipse x y 2 b → ellipse x y 2 b)
  ∧
  (∀ k m : ℝ, let l := line in
  ∀ x1 y1 x2 y2 : ℝ,
    l x1 y1 k m ∧ l x2 y2 k m ∧ ellipse x1 y1 2 b ∧ ellipse x2 y2 2 b →
    k * OA x1 y1 * OB x2 y2 = -3/4 →
    area Δ O A B = real.sqrt 3) :=
sorry

end problem_l625_625641


namespace union_A_B_l625_625335

def set_A : Set ℝ := { x | 1 / x ≤ 0 }
def set_B : Set ℝ := { x | x^2 - 1 < 0 }

theorem union_A_B : set_A ∪ set_B = { x | x < 1 } :=
by
  sorry

end union_A_B_l625_625335


namespace lcm_of_20_45_75_l625_625863

-- Definitions for the given numbers and their prime factorizations
def num1 : ℕ := 20
def num2 : ℕ := 45
def num3 : ℕ := 75

def factor1 : ℕ → Prop := λ n, n = 2 ^ 2 * 5
def factor2 : ℕ → Prop := λ n, n = 3 ^ 2 * 5
def factor3 : ℕ → Prop := λ n, n = 3 * 5 ^ 2

-- The condition of using the least common multiple function from mathlib
def lcm_def (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- The statement to prove
theorem lcm_of_20_45_75 : lcm_def num1 num2 num3 = 900 := by
  -- Factors condition (Note: These help ensure the numbers' factors are as stated)
  have h1 : factor1 num1 := by { unfold num1 factor1, exact rfl }
  have h2 : factor2 num2 := by { unfold num2 factor2, exact rfl }
  have h3 : factor3 num3 := by { unfold num3 factor3, exact rfl }
  sorry -- This is the place where the proof would go.

end lcm_of_20_45_75_l625_625863


namespace length_PQ_eq_one_eighth_RS_l625_625779

theorem length_PQ_eq_one_eighth_RS 
  (P Q R S : Point)
  (hPQ_on_RS : Segment P Q R S)
  (length_RP_eq_3_PS : length (R - P) = 3 * length (P - S))
  (length_RQ_eq_7_QS : length (R - Q) = 7 * length (Q - S)) :
  length (P - Q) = (1 / 8) * length (R - S) :=
by sorry

end length_PQ_eq_one_eighth_RS_l625_625779


namespace inequality_proof_l625_625899

theorem inequality_proof {n : ℕ} (a b : Fin n → ℝ) (h1 : ∀ i j : Fin n, a i < a j → b i ≤ b j) 
  (h2 : ∀ i j : Fin n, a i < (1 / n * ∑ k, a k) ∧ (1 / n * ∑ k, a k) < a j → b i ≤ b j) : 
  n * ∑ k, (a k) * (b k) ≥ (∑ k, a k) * (∑ k, b k) :=
sorry

end inequality_proof_l625_625899


namespace volume_of_rotated_solid_l625_625839

theorem volume_of_rotated_solid :
  let square_side := 4
  let rect_width := 3
  let rect_height := 5

  let volume_square := π * (square_side / 2)^2 * square_side
  let volume_rectangle := π * (rect_width / 2)^2 * rect_height

  volume_square + volume_rectangle = 109 * π / 4 :=
by
  let square_side := 4
  let rect_width := 3
  let rect_height := 5
  let volume_square := π * (square_side / 2) ^ 2 * square_side
  let volume_rectangle := π * ((rect_width / 2) ^ 2) * rect_height

  show volume_square + volume_rectangle = 109 * π / 4 from sorry

end volume_of_rotated_solid_l625_625839


namespace surface_area_of_pyramid_l625_625750

theorem surface_area_of_pyramid {A B C D : Type*}
  (is_triangle : ∀ (x y z: Type*), Triangle x y z)
  (point_outside_plane : ¬ Collinear D A B C)
  (all_edges_lengths_20_or_50 : ∀ e ∈ edges D A B C, e.length = 20 ∨ e.length = 50)
  (no_face_is_equilateral : ∀ face ∈ faces D A B C, ¬ Equilateral face) :
  surface_area D A B C = 800 * Real.sqrt 6 := 
begin
  sorry
end

end surface_area_of_pyramid_l625_625750


namespace shoe_store_sale_l625_625581

theorem shoe_store_sale (total_sneakers : ℕ) (total_sandals : ℕ) (total_shoes : ℕ) (total_boots : ℕ) 
  (h1 : total_sneakers = 2) 
  (h2 : total_sandals = 4) 
  (h3 : total_shoes = 17) 
  (h4 : total_boots = total_shoes - (total_sneakers + total_sandals)) : 
  total_boots = 11 :=
by
  rw [h1, h2, h3] at h4
  exact h4
-- sorry

end shoe_store_sale_l625_625581


namespace richard_older_than_david_by_l625_625912

-- Definitions based on given conditions

def richard : ℕ := sorry
def david : ℕ := 14 -- David is 14 years old.
def scott : ℕ := david - 8 -- Scott is 8 years younger than David.

-- In 8 years, Richard will be twice as old as Scott
axiom richard_in_8_years : richard + 8 = 2 * (scott + 8)

-- To prove: How many years older is Richard than David?
theorem richard_older_than_david_by : richard - david = 6 := sorry

end richard_older_than_david_by_l625_625912


namespace smallest_perfect_square_divisible_by_2_3_5_l625_625453

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧
         (∀ k : ℕ, 
           (k > 0 ∧ (∃ l : ℕ, k = l * l) ∧ (2 ∣ k) ∧ (3 ∣ k) ∧ (5 ∣ k)) → n ≤ k) :=
sorry

end smallest_perfect_square_divisible_by_2_3_5_l625_625453


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625437

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ (n : ℕ), (∃ k : ℕ, n = k^2) ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, (∃ k : ℕ, m = k^2) ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m :=
sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625437


namespace compare_y_values_l625_625241

theorem compare_y_values :
  let y₁ := 2 / (-2)
  let y₂ := 2 / (-1)
  y₁ > y₂ := by sorry

end compare_y_values_l625_625241


namespace percentage_of_triple_compared_to_front_squat_is_90_l625_625285

-- Definitions for the problem conditions
def original_back_squat : ℝ := 200
def increase : ℝ := 50
def new_back_squat : ℝ := original_back_squat + increase
def front_squat : ℝ := 0.80 * new_back_squat
def total_weight_moved_in_three_triples : ℝ := 540
def weight_of_one_triple : ℝ := total_weight_moved_in_three_triples / 3

-- The proof statement in Lean 4
theorem percentage_of_triple_compared_to_front_squat_is_90 :
    (weight_of_one_triple / front_squat) * 100 = 90 := by
  sorry

end percentage_of_triple_compared_to_front_squat_is_90_l625_625285


namespace problem_statement_l625_625363

noncomputable def roots (a b c : ℝ) : ℝ × ℝ := 
let Δ := b^2 - 4*a*c in
if Δ < 0 then (0, 0) else ((-b + Real.sqrt Δ) / (2*a), (-b - Real.sqrt Δ) / (2*a))

theorem problem_statement : 
  let (x1, x2) := roots 1 5 1 in
  (x1^2 + 5*x1 + 1 = 0 ∧ x2^2 + 5*x2 + 1 = 0) →
  (let expr := (x1 * Real.sqrt 6 / (1 + x2))^2 + (x2 * Real.sqrt 6 / (1 + x1))^2 
  in expr = 220) := 
by {
  sorry
}

end problem_statement_l625_625363


namespace total_path_length_l625_625928

open Real

theorem total_path_length (a b : ℝ) (h : a < b) :
  let α := 60 * (π / 180) in
  ∃ x : ℝ, x = a + 2 * sqrt ((a^2 + a * b + b^2) / 3) :=
sorry

end total_path_length_l625_625928


namespace line_PQ_parallel_to_axes_l625_625642

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / 4) + y^2 = 1

noncomputable def line_l (x y : ℝ) (m : ℝ) : Prop :=
  y = (1 / 2) * x + m

theorem line_PQ_parallel_to_axes
  (xA yA xB yB m : ℝ)
  (hxA : ellipse_eq xA yA)
  (hxB : ellipse_eq xB yB)
  (hintersect_A : line_l xA yA m)
  (hintersect_B : line_l xB yB m)
  (hdistinct : (xA ≠ sqrt(2) ∨ yA ≠ sqrt(2)/2) ∧ (xB ≠ sqrt(2) ∨ yB ≠ sqrt(2)/2))
  (P : (ℝ × ℝ)) (hP : P = (sqrt(2), sqrt(2)/2)) :
  (∃ Q, ellipse_eq Q.1 Q.2 ∧ (Q ≠ P) ∧
    (angle_bisector P (xA, yA) (xB, yB)).intersect (other_point_on_ellipse)) →
  (Q.1 = sqrt(2) ∨ Q.2 = sqrt(2)/2) :=
sorry

end line_PQ_parallel_to_axes_l625_625642


namespace least_common_multiple_of_20_45_75_l625_625872

theorem least_common_multiple_of_20_45_75 :
  Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
sorry

end least_common_multiple_of_20_45_75_l625_625872


namespace sum_a_b_c_l625_625303

noncomputable def x := Real.sqrt ((Real.sqrt 53) / 2 + 3 / 2)
axiom pos_integers (a b c : ℕ) : x ^ 100 = 2 * x ^ 98 + 14 * x ^ 96 + 11 * x ^ 94 - x ^ 50 + a * x ^ 46 + b * x ^ 44 + c * x ^ 40

theorem sum_a_b_c (a b c : ℕ) (h : pos_integers a b c) : a + b + c = 157 :=
sorry

end sum_a_b_c_l625_625303


namespace area_of_black_region_l625_625103

-- Definitions for the side lengths of the smaller and larger squares
def s₁ : ℕ := 4
def s₂ : ℕ := 8

-- The mathematical problem statement in Lean 4
theorem area_of_black_region : (s₂ * s₂) - (s₁ * s₁) = 48 := by
  sorry

end area_of_black_region_l625_625103


namespace area_difference_correct_l625_625077

noncomputable def area_difference (r : ℝ) (s : ℝ) : ℝ :=
  let circle_area := π * r^2
  let triangle_area := (sqrt 3 / 4) * s^2
  circle_area - triangle_area

theorem area_difference_correct :
  area_difference 3 6 = 9 * (π - sqrt 3) :=
by
  let r := 3
  let s := 6
  have circle_area := π * r^2
  have triangle_area := (sqrt 3 / 4) * s^2
  have difference := circle_area - triangle_area
  calc
    area_difference r s
      = difference := by rfl
    ... = (π * 3^2) - (sqrt 3 / 4 * 6^2) := by simp [circle_area, triangle_area]
    ... = 9 * (π - sqrt 3) := by ring

end area_difference_correct_l625_625077


namespace min_deliveries_l625_625735

theorem min_deliveries (cost_per_delivery_income: ℕ) (cost_per_delivery_gas: ℕ) (van_cost: ℕ) (d: ℕ) : 
  (d * (cost_per_delivery_income - cost_per_delivery_gas) ≥ van_cost) ↔ (d ≥ van_cost / (cost_per_delivery_income - cost_per_delivery_gas)) :=
by
  sorry

def john_deliveries : ℕ := 7500 / (15 - 5)

example : john_deliveries = 750 :=
by
  sorry

end min_deliveries_l625_625735


namespace pen_tip_movement_l625_625778

-- Definition of movements
def move_left (x : Int) : Int := -x
def move_right (x : Int) : Int := x

theorem pen_tip_movement :
  move_left 6 + move_right 3 = -3 :=
by
  sorry

end pen_tip_movement_l625_625778


namespace find_h_l625_625751

noncomputable def nested_radical_seq (b : ℝ) : ℝ :=
  let seq := λ x, real.sqrt (b^2 + x)
  classical.some (real.exists_fixed_point seq)

theorem find_h (h : ℝ) : (5 + nested_radical_seq h = 10) → h = 2 * real.sqrt 5 :=
by
  sorry

end find_h_l625_625751


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625439

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ (n : ℕ), (∃ k : ℕ, n = k^2) ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, (∃ k : ℕ, m = k^2) ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m :=
sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625439


namespace largest_divisor_of_five_even_numbers_l625_625856

theorem largest_divisor_of_five_even_numbers (n : ℕ) (h₁ : n % 2 = 1) : 
  ∃ d, (∀ n, n % 2 = 1 → d ∣ (n+2)*(n+4)*(n+6)*(n+8)*(n+10)) ∧ 
       (∀ d', (∀ n, n % 2 = 1 → d' ∣ (n+2)*(n+4)*(n+6)*(n+8)*(n+10)) → d' ≤ d) ∧ 
       d = 480 := sorry

end largest_divisor_of_five_even_numbers_l625_625856


namespace radius_of_circle_l625_625002

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

-- Prove that given the circle's equation, the radius is 1
theorem radius_of_circle (x y : ℝ) :
  circle_equation x y → ∃ (r : ℝ), r = 1 :=
by
  sorry

end radius_of_circle_l625_625002


namespace range_of_f_gt_f_2x_l625_625661

def f (x : ℝ) : ℝ := (x - 1) ^ 4 + 2 * abs (x - 1)

theorem range_of_f_gt_f_2x :
  {x : ℝ | f x > f (2 * x)} = set.Ioo 0 (2 / 3) :=
by
  sorry

end range_of_f_gt_f_2x_l625_625661


namespace solution_to_problem_l625_625133

noncomputable def x (r : ℝ) : ℝ := r^(1/6)

theorem solution_to_problem : ∃ (x : ℝ), 0 < x ∧ x^(2 * x^6) = 3 :=
begin
  use x 3,
  have h : x 3 = 3^(1/6),
  { 
    unfold x,
  },
  split,
  { 
    -- Prove x is positive
    rw h,
    exact real.rpow_pos_of_pos (by norm_num) (1/6),
  },
  {
    -- Prove x^(2 * x^6) = 3
    rw [h, ←real.rpow_mul],
    norm_num,
    exact real.rpow_one 3,
  }
end

end solution_to_problem_l625_625133


namespace find_f_7_over_2_l625_625189

section
variable {f : ℝ → ℝ}

-- Conditions
axiom odd_fn : ∀ x : ℝ, f (-x) = -f (x)
axiom even_shift_fn : ∀ x : ℝ, f (x + 1) = f (1 - x)
axiom range_x : Π x : ℝ, -1 ≤ x ∧ x ≤ 0 → f (x) = 2 * x^2

-- Prove that f(7/2) = 1/2
theorem find_f_7_over_2 : f (7 / 2) = 1 / 2 :=
sorry
end

end find_f_7_over_2_l625_625189


namespace katie_cupcakes_count_l625_625551

def original_cupcakes : ℕ := 26
def sold_cupcakes : ℕ := 20
def new_batch_percentage : ℝ := 0.75

def new_cupcakes : ℕ := real.floor (new_batch_percentage * original_cupcakes)

def remaining_cupcakes : ℕ := original_cupcakes - sold_cupcakes
def total_cupcakes : ℕ := remaining_cupcakes + new_cupcakes

theorem katie_cupcakes_count 
  (original_cupcakes = 26)
  (sold_cupcakes = 20)
  (new_batch_percentage = 0.75) :
  total_cupcakes = 25 :=
by
  sorry

end katie_cupcakes_count_l625_625551


namespace interior_diagonals_of_dodecahedron_l625_625685

-- Definition of the conditions is assumed as how they are required in the problem.
def dodecahedron : Type :=
{vertex : Type, face : set (set vertex) // 
  ∃ (vertices : fin 20), 
  (∀ v : vertices, (∃ (f : fin 12), set.finite (set.filter (λ s, s v) (set.range f)) = 3)) ∧ 
  (∀ f: fin 12, set.finite f = 5)}

-- Statement of the theorem
theorem interior_diagonals_of_dodecahedron (G : dodecahedron) : 
  (number_of_interiors_diagonals G = 100) :=
sorry

end interior_diagonals_of_dodecahedron_l625_625685


namespace anna_final_stamp_count_l625_625540

theorem anna_final_stamp_count (anna_initial : ℕ) (alison_initial : ℕ) (jeff_initial : ℕ)
  (anna_receive_from_alison : ℕ) (anna_give_jeff : ℕ) (anna_receive_jeff : ℕ) :
  anna_initial = 37 →
  alison_initial = 28 →
  jeff_initial = 31 →
  anna_receive_from_alison = alison_initial / 2 →
  anna_give_jeff = 2 →
  anna_receive_jeff = 1 →
  ∃ result : ℕ, result = 50 :=
by
  intros
  sorry

end anna_final_stamp_count_l625_625540


namespace intersection_is_correct_l625_625220

def A : Set ℤ := {0, 3, 4}
def B : Set ℤ := {-1, 0, 2, 3}

theorem intersection_is_correct : A ∩ B = {0, 3} := by
  sorry

end intersection_is_correct_l625_625220


namespace sum_of_exterior_angles_of_pentagon_l625_625021

theorem sum_of_exterior_angles_of_pentagon (P : Type) [Polygon P] (h1 : sides P = 5) : 
  sum_exterior_angles P = 360 :=
by
  -- Assume that the sum of exterior angles of any polygon is 360 degrees
  have H : ∀ P, sum_exterior_angles P = 360 := sorry,
  -- Apply this to the pentagon
  exact H P

end sum_of_exterior_angles_of_pentagon_l625_625021


namespace odd_sum_exceeds_even_sum_l625_625488

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

theorem odd_sum_exceeds_even_sum :
  let even_numbers := (1 to 1000).filter (λ n, n % 2 = 0),
      odd_numbers := (1 to 1000).filter (λ n, n % 2 = 1),
      even_sum := even_numbers.map sum_of_digits |>.sum,
      odd_sum := odd_numbers.map sum_of_digits |>.sum in
  odd_sum - even_sum = 499 := by
sorry

end odd_sum_exceeds_even_sum_l625_625488


namespace distinct_complex_roots_count_l625_625352

noncomputable def P (z : ℂ) : ℂ := (z + 1) * (z^2 + b * z + c)
noncomputable def Q (z : ℂ) : ℂ := (z + 1) * (a * z + d)
noncomputable def R (z : ℂ) : ℂ := (z + 1)^2 * (z^2 + e * z + f)

theorem distinct_complex_roots_count
  (deg_P : ∀ z : ℂ, polynomial.degree P = 3)
  (deg_Q : ∀ z : ℂ, polynomial.degree Q = 2)
  (deg_R : ∀ z : ℂ, polynomial.degree R = 4)
  (const_P : P 0 = 2)
  (const_Q : Q 0 = 3)
  (const_R : R 0 = 6)
  (root_P : P (-1) = 0)
  (root_Q : Q (-1) = 0) :
  ∃ M : ℕ, M = 1 := 
sorry

end distinct_complex_roots_count_l625_625352


namespace inequality_holds_equality_condition_l625_625743

noncomputable def max_k (ABCD : Type) [ConvexQuadrilateral ABCD] (P : Point)
  [Intersection P (Diagonal AC) (Diagonal BD)] (E F G H : Point)
  [Midpoint E AB] [Midpoint F BC] [Midpoint G CD] [Midpoint H DA] : ℝ :=
1 + real.sqrt 3

theorem inequality_holds (ABCD : Type) [ConvexQuadrilateral ABCD] (P : Point)
  [Intersection P (Diagonal AC) (Diagonal BD)] (E F G H : Point)
  [Midpoint E AB] [Midpoint F BC] [Midpoint G CD] [Midpoint H DA]
  (d : ℝ) (s : ℝ) (d_def : d = DiagonalLengthSum ABCD) (s_def : s = Semiperimeter ABCD) :
  ∃ k : ℝ, k = max_k ABCD P E F G H ∧ ∀ (EG HF : ℝ), EG + 3 * HF ≥ k * d + (1 - k) * s := sorry

theorem equality_condition (ABCD : Type) [ConvexQuadrilateral ABCD]
  [Rectangle ABCD] (P : Point) [Intersection P (Diagonal AC) (Diagonal BD)] 
  (E F G H : Point) [Midpoint E AB] [Midpoint F BC] [Midpoint G CD] [Midpoint H DA]
  (d : ℝ) (s : ℝ) (d_def : d = DiagonalLengthSum ABCD) (s_def : s = Semiperimeter ABCD)
  (EG HF : ℝ) : 
  EG + 3 * HF = max_k ABCD P E F G H * d + (1 - max_k ABCD P E F G H) * s ↔ ABCD_is_rectangle ABCD := sorry

end inequality_holds_equality_condition_l625_625743


namespace find_coordinates_of_B_find_k_if_perpendicular_l625_625119

-- Definitions based on the problem conditions.
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

def A : ℝ × ℝ := (-2, 0)
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

-- Slope of line AB
def line_AB (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x + 2)

-- Intersection condition of line AB with the ellipse
def intersects (k : ℝ) (x y : ℝ) : Prop :=
  ellipse x y ∧ line_AB k x y

-- Coordinates of point B in terms of k
def B (k : ℝ) : ℝ × ℝ :=
  ( (6 - 8 * k^2) / (3 + 4 * k^2), (12 * k) / (3 + 4 * k^2) )

-- Proof statement
theorem find_coordinates_of_B (k : ℝ) :
  ∃ (x y : ℝ), intersects k x y ∧ (x, y) = B k := sorry

theorem find_k_if_perpendicular (k : ℝ) :
  (∃ (x y : ℝ), intersects k x y ∧ (x, y) = B k) →
  (F1.1 < x → (∀ (x y : ℝ), intersect k x y → y ≠ 0 → F1.y /(B(k).2-x) = -(1/k)) →
  k = sqrt(6)/12 ∨ k = -sqrt(6)/12 := sorry

end find_coordinates_of_B_find_k_if_perpendicular_l625_625119


namespace lathe_defective_probability_l625_625715

variables (A1 A2 B : Type) [ProbabilitySpace A1] [ProbabilitySpace A2] [ProbabilitySpace B]

def P_A1 : ℝ := 0.4
def P_A2 : ℝ := 0.6
def P_B_given_A1 : ℝ := 0.06
def P_B_given_A2 : ℝ := 0.05

theorem lathe_defective_probability :
  P(A2 ∩ B) = 0.03 :=
by
  sorry

end lathe_defective_probability_l625_625715


namespace average_of_consecutive_numbers_l625_625261

theorem average_of_consecutive_numbers (n : ℕ) (s : ℕ) 
  (h1 : n = 99) 
  (h2 : ∀i ∈ (list.range n).map ((+) s), i ∈ ℕ) 
  (h3 : s + 98 = 25.5 * s) : 
  (list.sum (list.range n).map ((+) s) / n = 53) :=
sorry

end average_of_consecutive_numbers_l625_625261


namespace number_of_truthful_dwarfs_l625_625603

def dwarf_condition := 
  ∀ (dwarfs : ℕ) (truthful_dwarfs : ℕ) (lying_dwarfs : ℕ),
    dwarfs = 10 ∧ 
    (∀ n, n ∈ {truthful_dwarfs, lying_dwarfs} -> n ≥ 0) ∧ 
    truthful_dwarfs + lying_dwarfs = dwarfs ∧
    truthful_dwarfs + 2 * lying_dwarfs = 16

theorem number_of_truthful_dwarfs : ∃ (truthful_dwarfs : ℕ), (dwarf_condition ∧ truthful_dwarfs = 4) :=
by {
  let dwarfs := 10,
  let lying_dwarfs := 6,
  let truthful_dwarfs := dwarfs - lying_dwarfs,
  have h: truthful_dwarfs = 4,
  { calc
    truthful_dwarfs = dwarfs - lying_dwarfs : by rfl
    ... = 10 - 6 : by rfl
    ... = 4 : by rfl },
  existsi (4 : ℕ),
  refine ⟨_, ⟨dwarfs, truthful_dwarfs, lying_dwarfs, rfl, _, _, _⟩⟩,
  -- Now we can provide the additional details for lean to understand the conditions hold
  {
    intros n hn,
    simp,
    exact hn
  },
  {
    exact add_comm 6 4
  },
  {
    dsimp,
    ring,
  },
  {
    exact h,
  }
  -- Skip the actual proof with sorry
  sorry
}

end number_of_truthful_dwarfs_l625_625603


namespace points_distance_is_correct_l625_625371

noncomputable def points_intersection_distance : ℚ :=
  let a : ℝ := 3
  let b : ℝ := 2
  let c : ℝ := -7
  let discriminant := b^2 - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let x₁ := (-b + sqrt_discriminant) / (2 * a)
  let x₂ := (-b - sqrt_discriminant) / (2 * a)
  let distance := Real.abs (x₁ - x₂)
  distance

theorem points_distance_is_correct :
  points_intersection_distance = (2 * Real.sqrt 22) / 3 :=
sorry

end points_distance_is_correct_l625_625371


namespace value_set_fraction_l625_625799

noncomputable def value_set (m n t : ℝ) : set ℝ :=
  {x : ℝ | ∃ (θ : ℝ), θ ∈ [-real.sqrt 3 / 3, real.sqrt 3 / 3] ∧ x = (n / t) / ((m / t) - 2)}

theorem value_set_fraction (m n t : ℝ) (ht : t ≠ 0) (h : m^2 + n^2 = t^2) :
  value_set m n t = [-real.sqrt 3 / 3, real.sqrt 3 / 3] :=
sorry

end value_set_fraction_l625_625799


namespace complex_roots_lie_on_circle_l625_625109

noncomputable def radius_of_circle_with_complex_roots : ℝ :=
  (2 / 3)

theorem complex_roots_lie_on_circle :
  ∀ z : ℂ, (z + 1)^4 = 16 * z^4 → |z + 1| = 2 * |z| → (∃ r : ℝ, r = radius_of_circle_with_complex_roots ∧ ∀ z : ℂ, (z + 1)^4 = 16 * z^4 → abs (z + 1) = 2 * abs z → abs z = r) :=
sorry

end complex_roots_lie_on_circle_l625_625109


namespace quadratic_inequality_solution_l625_625348

theorem quadratic_inequality_solution (x m : ℝ) :
  (x^2 + (2*m + 1)*x + m^2 + m > 0) ↔ (x > -m ∨ x < -m - 1) :=
by
  sorry

end quadratic_inequality_solution_l625_625348


namespace point_divides_edge_l625_625991

-- Define the cube with points A, B, C, D, A1, B1, C1, D1
-- We use Vectors to represent coordinates in the 3-dimensional space

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def A : Point3D := ⟨0, 0, 0⟩
def B : Point3D := ⟨1, 0, 0⟩
def C : Point3D := ⟨1, 1, 0⟩
def D : Point3D := ⟨0, 1, 0⟩
def A1 : Point3D := ⟨0, 0, 1⟩
def B1 : Point3D := ⟨1, 0, 1⟩
def C1 : Point3D := ⟨1, 1, 1⟩
def D1 : Point3D := ⟨0, 1, 1⟩

-- Define centers K and H of the respective faces
def K : Point3D := ⟨0.5, 0.5, 1⟩  -- Center of face A1 B1 C1 D1
def H : Point3D := ⟨1, 0.5, 0.5⟩  -- Center of face B1 C1 CB

-- Assume we have point E which lies on edge B1 C1
noncomputable def E : Point3D := sorry  -- Determined by plane intersection

-- Define the condition E lies on the plane containing A, K, H
def liesOnPlane (P Q R S : Point3D) : Prop :=
  let u := ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩
  let v := ⟨R.x - P.x, R.y - P.y, R.z - P.z⟩
  let w := ⟨S.x - P.x, S.y - P.y, S.z - P.z⟩
  (u.y * v.z - u.z * v.y) * w.x + (u.z * v.x - u.x * v.z) * w.y + (u.x * v.y - u.y * v.x) * w.z = 0

-- Define the condition E lies on edge B1 C1
def liesOnEdge (P Q R : Point3D) : Prop :=
  let u := ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩
  let v := ⟨R.x - Q.x, R.y - Q.y, R.z - Q.z⟩
  u.x * v.y = u.y * v.z ∧ u.z * v.x = u.x * v.z

-- The theorem to prove
theorem point_divides_edge :
  liesOnPlane A K H E ∧ liesOnEdge B1 C1 E → (B1.x - E.x) / (E.x - C1.x) = 2 :=
  by
    sorry

end point_divides_edge_l625_625991


namespace prove_propositions_l625_625114

-- Definitions and conditions for the propositions
variables (α β γ : Type) [plane α] [plane β] [plane γ]
variables (a b l : Type) [line a] [line b] [line l]
variables (h1 : α ∩ β = a)
variables (h2 : b ⊆ α)
variables (h3 : a ⊥ b)
variables (h4 : a ⊆ α)
variables (h5 : b ⊆ β)
variables (h6 : a ∥ β)
variables (h7 : b ∥ α)
variables (h8 : α ∥ γ)
variables (h9 : γ ∥ β)
variables (h10 : α ∩ β = l)
variables (h11 : α ⊥ γ)
variables (h12 : β ⊥ γ)
variables (h13 : α ⊥ β)
variables (h14 : a ⊥ β)

-- Proposition statement
theorem prove_propositions :
  (¬ (α ∩ β = a ∧ b ⊆ α ∧ a ⊥ b → α ⊥ β)) ∧
  (a ⊆ α ∧ b ⊆ β ∧ a ∥ β ∧ b ∥ α → α ∥ β) ∧
  (α ⊥ γ ∧ β ⊥ γ ∧ α ∩ β = l → l ⊥ γ) ∧
  (α ⊥ β ∧ a ⊥ β → (a ⊆ α ∨ a ∥ α)) :=
by sorry

end prove_propositions_l625_625114


namespace correct_assignment_l625_625890

theorem correct_assignment (M : Type) (A B x y : Type) : 
  (M = -M) ∧ (¬(3 = A)) ∧ (¬(B = A = 2)) ∧ (¬(x + y = 0)) :=
by
  sorry

end correct_assignment_l625_625890


namespace f_500_eq_39_l625_625506

axiom f : ℕ → ℕ
axiom functional_eqn : ∀ x y : ℕ, f(x * y) = f(x) + f(y)
axiom f_10 : f(10) = 14
axiom f_40 : f(40) = 20

theorem f_500_eq_39 : f(500) = 39 :=
by
  sorry

end f_500_eq_39_l625_625506


namespace find_b_l625_625833

theorem find_b (b : ℝ) :
  (∃ x1 x2 : ℝ, (x1 + x2 = -2) ∧
    ((x1 + 1)^3 + x1 / (x1 + 1) = -x1 + b) ∧
    ((x2 + 1)^3 + x2 / (x2 + 1) = -x2 + b)) →
  b = 0 :=
by
  sorry

end find_b_l625_625833


namespace solution_set_of_quadratic_inequality_l625_625647

variable {a b x : ℝ}

-- Given condition: the solution set of the inequality ax > b is (-∞, 1/5)
-- That is, (∀ x, ax > b ↔ x ∈ set.Iio (1/5))
axiom ax_gt_b_solution_set (a b : ℝ) (x : ℝ) :
  (∀ x, a * x > b ↔ x ∈ set.Iio (1 / 5))

-- Theorem to be proved: the solution set of the inequality ax^2 + bx - (4/5)a > 0 is (-1, 4/5)
theorem solution_set_of_quadratic_inequality (a b : ℝ) (x : ℝ) :
  (a < 0) -> (b / a = 1 / 5) -> 
  (x^2 + (b / a) * x - 4 / 5 < 0) ↔ x ∈ set.Ioo (-1) (4 / 5) :=
by
  -- "by" indicates no proof is required, and "sorry" is used as a placeholder for the proof.
  sorry

end solution_set_of_quadratic_inequality_l625_625647


namespace total_students_in_class_l625_625771

-- Define conditions as constants
constant madhav_rank_top : ℕ
constant madhav_rank_bottom : ℕ

-- Given conditions
axiom h1 : madhav_rank_top = 17
axiom h2 : madhav_rank_bottom = 15

-- Lean 4 statement to prove the total number of students
theorem total_students_in_class : (madhav_rank_top + madhav_rank_bottom - 1) = 31 :=
by
  rw [h1, h2]
  exact rfl

end total_students_in_class_l625_625771


namespace roots_of_x2_eq_x_l625_625017

theorem roots_of_x2_eq_x : ∀ x : ℝ, x^2 = x ↔ (x = 0 ∨ x = 1) := 
by
  sorry

end roots_of_x2_eq_x_l625_625017


namespace bus_dispatch_interval_l625_625476

/-- Xiao Wang walks at a constant speed along the street. A No. 18 bus passes him from behind every 6 minutes. 
A No. 18 bus comes towards him every 3 minutes. Each No. 18 bus travels at the same speed, and the 
No. 18 bus terminal dispatches a bus at fixed intervals. Prove that the interval between each bus dispatch is 4 minutes. -/
theorem bus_dispatch_interval (a b t : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : at = 6 * (a - b)) (h4 : at = 3 * (a + b)) : t = 4 := 
by sorry

end bus_dispatch_interval_l625_625476


namespace number_multiplied_by_sum_of_digits_eq_2008_l625_625091

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem number_multiplied_by_sum_of_digits_eq_2008 : ∃ n : ℕ, n * sum_of_digits n = 2008 ∧ n = 251 := by
  use 251
  have h : sum_of_digits 251 = 8 := by sorry
  have h' : 251 * sum_of_digits 251 = 2008 := by
    rw [h]
    norm_num
  exact ⟨h', rfl⟩

end number_multiplied_by_sum_of_digits_eq_2008_l625_625091


namespace cubic_polynomial_b_value_l625_625323

theorem cubic_polynomial_b_value (a b c d : ℝ)
  (h₁ : f = λ x, a * x^3 + b * x^2 + c * x + d)
  (h₂ : f (-2) = 0)
  (h₃ : f 2 = 0)
  (h₄ : f 0 = 3) : b = 3 / 4 := 
sorry

end cubic_polynomial_b_value_l625_625323


namespace area_of_triangle_proof_l625_625175

noncomputable def area_of_triangle {P F1 F2 : ℝ × ℝ} (h1 : P ∈ hyperbola 9 16)
                                  (h2 : dist P F1 * dist P F2 = 32)
                                  (h_foci_1 : F1 = (-5, 0))
                                  (h_foci_2 : F2 = (5, 0)) : ℝ :=
  if ∃ a b, hyperbola_eq a b 9 16 P
  then 16
  else 0

theorem area_of_triangle_proof {P F1 F2 : ℝ × ℝ} (h1 : P ∈ hyperbola 9 16)
                                  (h2 : dist P F1 * dist P F2 = 32)
                                  (h_foci_1 : F1 = (-5, 0))
                                  (h_foci_2 : F2 = (5, 0)) :
  area_of_triangle h1 h2 h_foci_1 h_foci_2 = 16 := by
  sorry

end area_of_triangle_proof_l625_625175


namespace glycerin_solution_l625_625068

theorem glycerin_solution (x : ℝ) :
    let total_volume := 100
    let final_glycerin_percentage := 0.75
    let volume_first_solution := 75
    let volume_second_solution := 75
    let second_solution_percentage := 0.90
    let final_glycerin_volume := final_glycerin_percentage * total_volume
    let glycerin_second_solution := second_solution_percentage * volume_second_solution
    let glycerin_first_solution := x * volume_first_solution / 100
    glycerin_first_solution + glycerin_second_solution = final_glycerin_volume →
    x = 10 :=
by
    sorry

end glycerin_solution_l625_625068


namespace number_of_truthful_dwarfs_l625_625599

/-- 
Each of the 10 dwarfs either always tells the truth or always lies. 
Each dwarf likes exactly one type of ice cream: vanilla, chocolate, or fruit.
When asked, every dwarf raised their hand for liking vanilla ice cream.
When asked, 5 dwarfs raised their hand for liking chocolate ice cream.
When asked, only 1 dwarf raised their hand for liking fruit ice cream.
Prove that the number of truthful dwarfs is 4.
-/
theorem number_of_truthful_dwarfs (T L : ℕ) 
  (h1 : T + L = 10) 
  (h2 : T + 2 * L = 16) : 
  T = 4 := 
by
  -- Proof omitted
  sorry

end number_of_truthful_dwarfs_l625_625599


namespace symmetry_line_between_C1_and_C2_l625_625669

-- Define the equations for the parabolas
def C1 (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 1
def C2 (x : ℝ) : ℝ := C1 (x - 3)

-- Define the axes of symmetry for C1 and C2
def axis_symmetry_C1 : ℝ := 1
def axis_symmetry_C2 : ℝ := 4

-- Define the expected line of symmetry
def expected_symmetry_line : ℝ := (axis_symmetry_C1 + axis_symmetry_C2) / 2

-- Theorem statement: Proving the line of symmetry between the parabolas C1 and C2 is x = 5/2
theorem symmetry_line_between_C1_and_C2 : expected_symmetry_line = 5 / 2 :=
by
  sorry

end symmetry_line_between_C1_and_C2_l625_625669


namespace anna_stamp_count_correct_l625_625542

-- Defining the initial counts of stamps
def anna_initial := 37
def alison_initial := 28
def jeff_initial := 31

-- Defining the operations
def alison_gives_half_to_anna := alison_initial / 2
def anna_after_receiving_from_alison := anna_initial + alison_gives_half_to_anna
def anna_final := anna_after_receiving_from_alison - 2 + 1

-- Formalizing the proof problem
theorem anna_stamp_count_correct : anna_final = 50 := by
  -- proof omitted
  sorry

end anna_stamp_count_correct_l625_625542


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625425

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, (n > 0 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∧ ∃ k : ℕ, n = k^2 ∧ n = 225 := 
begin
  sorry
end

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625425


namespace jessica_has_100_dollars_l625_625333

-- Define the variables for Rodney, Ian, and Jessica
variables (R I J : ℝ)

-- Given conditions
axiom rodney_more_than_ian : R = I + 35
axiom ian_half_of_jessica : I = J / 2
axiom jessica_more_than_rodney : J = R + 15

-- The statement to prove
theorem jessica_has_100_dollars : J = 100 :=
by
  -- Proof will be completed here
  sorry

end jessica_has_100_dollars_l625_625333


namespace compound_interest_rate_l625_625047

theorem compound_interest_rate (P A : ℝ) (t n : ℝ)
  (hP : P = 5000) 
  (hA : A = 7850)
  (ht : t = 8)
  (hn : n = 1) : 
  ∃ r : ℝ, 0.057373 ≤ (r * 100) ∧ (r * 100) ≤ 5.7373 :=
by
  sorry

end compound_interest_rate_l625_625047


namespace monotonicity_two_zeros_and_sum_l625_625207

noncomputable def f (x a : ℝ) : ℝ := Real.log x + a * x ^ 2 + 2

theorem monotonicity (a : ℝ) : 
  (∀ x > 0, a ≥ 0 → deriv (λ x, f x a) x > 0) ∧
  (∀ x > 0, a < 0 → (∀ y, 0 < y ∧ y < Real.sqrt (-1 / (2 * a)) → deriv (λ x, f x a) y > 0) ∧
                      (∀ y, y > Real.sqrt (-1 / (2 * a)) → deriv (λ x, f x a) y < 0)) :=
sorry

theorem two_zeros_and_sum (x1 x2 : ℝ) (h1 : x1 > 0) (h2 : x2 > 0) (h_zeros : f x1 (-1) = 0 ∧ f x2 (-1) = 0) :
  x1 + x2 > Real.sqrt 2 :=
sorry

end monotonicity_two_zeros_and_sum_l625_625207


namespace sequence_sum_l625_625966

theorem sequence_sum (r x y : ℝ) (h1 : r = 1/4) 
  (h2 : x = 256 * r)
  (h3 : y = x * r) : x + y = 80 :=
by
  sorry

end sequence_sum_l625_625966


namespace complex_roots_circle_radius_l625_625112

noncomputable def radius_of_circle : ℝ := 2 / 3

theorem complex_roots_circle_radius (z : ℂ) (h : (z + 1) ^ 4 = 16 * z ^ 4) : 
  ∃ x y : ℝ, z = x + y * complex.I ∧ (x - 1/3)^2 + y^2 = (2/3)^2 :=
sorry

end complex_roots_circle_radius_l625_625112


namespace dan_age_l625_625572

theorem dan_age (D : ℕ) (h : D + 20 = 7 * (D - 4)) : D = 8 :=
by
  sorry

end dan_age_l625_625572


namespace sum_of_first_9_terms_l625_625722

-- Define an arithmetic sequence {a_n}
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Sum of the first n terms of an arithmetic sequence with first term a and common difference d
def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Define the specific problem condition
def condition := arithmetic_sequence a d 3 + arithmetic_sequence a d 7 = 10

theorem sum_of_first_9_terms (a d : ℕ) (h : condition) : sum_arithmetic_sequence a d 9 = 45 :=
by
  sorry

end sum_of_first_9_terms_l625_625722


namespace smallest_perfect_square_divisible_by_2_3_5_l625_625455

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧
         (∀ k : ℕ, 
           (k > 0 ∧ (∃ l : ℕ, k = l * l) ∧ (2 ∣ k) ∧ (3 ∣ k) ∧ (5 ∣ k)) → n ≤ k) :=
sorry

end smallest_perfect_square_divisible_by_2_3_5_l625_625455


namespace evaluate_mn_l625_625688

theorem evaluate_mn (m n : ℤ) (h1 : m + 3 = 2) (h2 : n - 1 = 2) : (m^n = -1) :=
by
  sorry

end evaluate_mn_l625_625688


namespace tangent_length_l625_625176

-- Define the point P and the circle equation
def PointP := (2, 3 : ℝ)
def CircleEquation (x y : ℝ) := (x - 1)^2 + (y - 1)^2 = 1

-- Define the center of the circle
def CenterC := (1, 1 : ℝ)

-- Define the distance function
def Distance (P1 P2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

-- Specify the radius of the circle
def radius : ℝ := 1

-- The theorem that we need to prove
theorem tangent_length : 
  Distance PointP CenterC = real.sqrt 5 → 
  sqrt (5 - radius^2) = 2 := 
by
  intros h
  sorry

end tangent_length_l625_625176


namespace transformed_complex_number_l625_625405

noncomputable def initial_complex : ℂ := -4 - 6 * complex.I
noncomputable def rotation_angle : ℂ := complex.cis (real.pi / 6)  -- cis 30 degrees
noncomputable def scale_factor : ℂ := 2

theorem transformed_complex_number : 
  (initial_complex * (rotation_angle * scale_factor)) = (6 - 4 * real.sqrt 3) + (-6 * real.sqrt 3 - 4) * complex.I :=
by
  sorry

end transformed_complex_number_l625_625405


namespace square_area_ratio_l625_625820

theorem square_area_ratio (a b : ℕ) (h : 4 * a = 4 * (4 * b)) : (a^2) = 16 * (b^2) := 
by sorry

end square_area_ratio_l625_625820


namespace dice_probability_l625_625938

theorem dice_probability : 
  let total_outcomes := 36
  let favorable_outcomes := 3
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 12 :=
by
  let total_outcomes := 36
  let favorable_outcomes := 3
  have h : (favorable_outcomes : ℚ) / total_outcomes = 1 / 12 := by norm_num
  exact h
  sorry

end dice_probability_l625_625938


namespace triangle_side_and_altitude_sum_l625_625531

theorem triangle_side_and_altitude_sum 
(x y : ℕ) (h1 : x < 75) (h2 : y < 28)
(h3 : x * 60 = 75 * 28) (h4 : 100 * y = 75 * 28) : 
x + y = 56 := 
sorry

end triangle_side_and_altitude_sum_l625_625531


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625447

/-- Problem statement: 
    The smallest positive perfect square that is divisible by 2, 3, and 5 is 900.
-/
theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ n * n % 2 = 0 ∧ n * n % 3 = 0 ∧ n * n % 5 = 0 ∧ n * n = 900 := 
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625447


namespace apples_left_l625_625729

-- Define the initial number of apples and the conditions
def initial_apples := 150
def percent_sold_to_jill := 20 / 100
def percent_sold_to_june := 30 / 100
def apples_given_to_teacher := 2

-- Formulate the problem statement in Lean
theorem apples_left (initial_apples percent_sold_to_jill percent_sold_to_june apples_given_to_teacher : ℕ) :
  let sold_to_jill := percent_sold_to_jill * initial_apples
  let remaining_after_jill := initial_apples - sold_to_jill
  let sold_to_june := percent_sold_to_june * remaining_after_jill
  let remaining_after_june := remaining_after_jill - sold_to_june
  let final_apples := remaining_after_june - apples_given_to_teacher
  final_apples = 82 := 
by 
  sorry

end apples_left_l625_625729


namespace length_reduction_by_50_percent_l625_625009

variable (L B L' : ℝ)

def rectangle_dimension_change (L B : ℝ) (perc_area_change : ℝ) (new_breadth_factor : ℝ) : Prop :=
  let original_area := L * B
  let new_breadth := new_breadth_factor * B
  let new_area := L' * new_breadth
  let expected_new_area := (1 + perc_area_change) * original_area
  new_area = expected_new_area

theorem length_reduction_by_50_percent (L B : ℝ) (h1: rectangle_dimension_change L B L' 0.5 3) : 
  L' = 0.5 * L :=
by
  unfold rectangle_dimension_change at h1
  simp at h1
  sorry

end length_reduction_by_50_percent_l625_625009


namespace smallest_positive_perfect_square_div_by_2_3_5_l625_625434

-- Definition capturing the conditions of the problem
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def divisible_by (n m : ℕ) : Prop :=
  m % n = 0

-- The theorem statement combining all conditions and requiring the proof of the correct answer
theorem smallest_positive_perfect_square_div_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ is_perfect_square n ∧ divisible_by 2 n ∧ divisible_by 3 n ∧ divisible_by 5 n ∧ n = 900 :=
sorry

end smallest_positive_perfect_square_div_by_2_3_5_l625_625434


namespace final_value_exceeds_N_iff_k_gt_100r_over_100_minus_r_l625_625232

variable (k r s N : ℝ)
variable (h_pos_k : 0 < k)
variable (h_pos_r : 0 < r)
variable (h_pos_s : 0 < s)
variable (h_pos_N : 0 < N)
variable (h_r_lt_80 : r < 80)

theorem final_value_exceeds_N_iff_k_gt_100r_over_100_minus_r :
  N * (1 + k / 100) * (1 - r / 100) + 10 * s > N ↔ k > 100 * r / (100 - r) :=
sorry

end final_value_exceeds_N_iff_k_gt_100r_over_100_minus_r_l625_625232


namespace probability_of_valid_path_ending_at_H_l625_625502

def vertex : Type := {A B C D E F G H}
def cube_edges : vertex → vertex → Prop := sorry -- Define the edges of the cube

def bug_moves (start : vertex) (path : list vertex) : Prop :=
  path.head = start ∧
  (∀ n, n < 9 → cube_edges (path.nth n) (path.nth (n + 1))) ∧
  (path.nodup) ∧
  (path.length = 9)

noncomputable def equal_probability (path : list vertex) : ℝ :=
  1 / (3 ^ 9)

noncomputable def valid_paths (start end : vertex) :=
  {path : list vertex // bug_moves start path ∧ path.last = end}

theorem probability_of_valid_path_ending_at_H (start : vertex) :
  (start = A) →
  (equal_probability = 1 / (3 ^ 9)) →
  (card (valid_paths start H) = 4) →
  (@probability_of_valid_path_ending_at_H start H = 4 / (3 ^ 9)) :=
by sorry

end probability_of_valid_path_ending_at_H_l625_625502


namespace car_with_highest_avg_speed_l625_625560

-- Conditions
def distance_A : ℕ := 715
def time_A : ℕ := 11
def distance_B : ℕ := 820
def time_B : ℕ := 12
def distance_C : ℕ := 950
def time_C : ℕ := 14

-- Average Speeds
def avg_speed_A : ℚ := distance_A / time_A
def avg_speed_B : ℚ := distance_B / time_B
def avg_speed_C : ℚ := distance_C / time_C

-- Theorem
theorem car_with_highest_avg_speed : avg_speed_B > avg_speed_A ∧ avg_speed_B > avg_speed_C :=
by
  sorry

end car_with_highest_avg_speed_l625_625560


namespace find_percentage_decrease_in_fourth_month_l625_625708

theorem find_percentage_decrease_in_fourth_month
  (P0 : ℝ) (P1 : ℝ) (P2 : ℝ) (P3 : ℝ) (x : ℝ) :
  (P0 = 100) →
  (P1 = P0 + 0.30 * P0) →
  (P2 = P1 - 0.15 * P1) →
  (P3 = P2 + 0.10 * P2) →
  (P0 = P3 - x / 100 * P3) →
  x = 18 :=
by
  sorry

end find_percentage_decrease_in_fourth_month_l625_625708


namespace perp_bisector_chord_l625_625196

theorem perp_bisector_chord (x y : ℝ) :
  (2 * x + 3 * y + 1 = 0) ∧ (x^2 + y^2 - 2 * x + 4 * y = 0) → 
  ∃ k l m : ℝ, (3 * x - 2 * y - 7 = 0) :=
by
  sorry

end perp_bisector_chord_l625_625196


namespace cans_collected_on_first_day_l625_625474

-- Declare the main theorem
theorem cans_collected_on_first_day 
  (x : ℕ) -- Number of cans collected on the first day
  (total_cans : x + (x + 5) + (x + 10) + (x + 15) + (x + 20) = 150) :
  x = 20 :=
sorry

end cans_collected_on_first_day_l625_625474


namespace remainder_zero_by_68_l625_625320

theorem remainder_zero_by_68 (N R1 Q2 : ℕ) (h1 : N = 68 * 269 + R1) (h2 : N % 67 = 1) : R1 = 0 := by
  sorry

end remainder_zero_by_68_l625_625320


namespace area_of_triangle_l625_625849

theorem area_of_triangle : 
  let A := (0 : ℝ, 0 : ℝ),
      B := (2 : ℝ, 0 : ℝ),
      C := (2 : ℝ, 1 : ℝ) in
  (1/2) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)| = 1 :=
by
  let A := (0 : ℝ, 0 : ℝ)
  let B := (2 : ℝ, 0 : ℝ)
  let C := (2 : ℝ, 1 : ℝ)
  sorry

end area_of_triangle_l625_625849


namespace smallest_perfect_square_divisible_by_2_3_5_l625_625457

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧
         (∀ k : ℕ, 
           (k > 0 ∧ (∃ l : ℕ, k = l * l) ∧ (2 ∣ k) ∧ (3 ∣ k) ∧ (5 ∣ k)) → n ≤ k) :=
sorry

end smallest_perfect_square_divisible_by_2_3_5_l625_625457


namespace cartesian_curve_eq_of_polar_eq_intersection_value_l625_625148

variable {t α ρ θ x y : ℝ}
variable (l : ℝ → ℝ × ℝ)
def C_polar_eq (ρ θ : ℝ) : Prop := ρ * (sin θ) ^ 2 = 8 * cos θ
def C_cartesian_eq (x y : ℝ) : Prop := y^2 = 8 * x

def line_eq (t : ℝ) (α : ℝ) : ℝ × ℝ := (2 + t * cos α, t * sin α)

theorem cartesian_curve_eq_of_polar_eq (ρ θ : ℝ) : (C_polar_eq ρ θ) → (C_cartesian_eq (ρ * cos θ) (ρ * sin θ)) :=
by sorry

theorem intersection_value (α : ℝ) (hα : sin α ≠ 0) :
  let l_param := (line_eq t α) in
  let A : ℝ × ℝ := l_param t1 in
  let B : ℝ × ℝ := l_param t2 in
  let F := (2, 0) in
  A, B ∈ C_cartesian_eq ∧
  (1 / real.dist A F + 1 / real.dist B F) = 1 / 2 :=
by sorry

end cartesian_curve_eq_of_polar_eq_intersection_value_l625_625148


namespace white_lambs_count_l625_625147

theorem white_lambs_count (total_lambs black_lambs white_lambs : ℕ) (h1 : total_lambs = 6048) (h2 : black_lambs = 5855) : white_lambs = 193 :=
by 
  have h3 : white_lambs = total_lambs - black_lambs,
  sorry

end white_lambs_count_l625_625147


namespace fraction_to_decimal_l625_625968

theorem fraction_to_decimal : (47 : ℝ) / 160 = 0.29375 :=
by
  sorry

end fraction_to_decimal_l625_625968


namespace quadratic_function_properties_l625_625663

noncomputable def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  (m + 2) * x^(m^2 + m - 4)

theorem quadratic_function_properties :
  (∀ m, (m^2 + m - 4 = 2) → (m = -3 ∨ m = 2))
  ∧ (m = -3 → quadratic_function m 0 = 0) 
  ∧ (m = -3 → ∀ x, x > 0 → quadratic_function m x ≤ quadratic_function m 0 ∧ quadratic_function m x < 0)
  ∧ (m = -3 → ∀ x, x < 0 → quadratic_function m x ≤ quadratic_function m 0 ∧ quadratic_function m x < 0) :=
by
  -- Proof will be supplied here.
  sorry

end quadratic_function_properties_l625_625663


namespace complex_roots_lie_on_circle_l625_625110

noncomputable def radius_of_circle_with_complex_roots : ℝ :=
  (2 / 3)

theorem complex_roots_lie_on_circle :
  ∀ z : ℂ, (z + 1)^4 = 16 * z^4 → |z + 1| = 2 * |z| → (∃ r : ℝ, r = radius_of_circle_with_complex_roots ∧ ∀ z : ℂ, (z + 1)^4 = 16 * z^4 → abs (z + 1) = 2 * abs z → abs z = r) :=
sorry

end complex_roots_lie_on_circle_l625_625110


namespace tangent_square_sum_eq_eight_l625_625545

-- Define the setup with square ABCD inscribed in a circle.
variable (r : ℝ) -- radius of the inscribed circle
variable {θ : ℝ} -- parameter for point P on the circle

-- Define point P on the circle with coordinates using parameter θ
def P_coords (θ : ℝ) : ℝ × ℝ := (r * Real.cos θ, r * Real.sin θ)

-- Define the angles
parameter {α β : ℝ} -- α and β are the angles mentioned respectively

-- Lean statement for the proof problem
theorem tangent_square_sum_eq_eight (r : ℝ) (θ α β : ℝ)
  (hα : α = Real.arctan ((1 - Real.sin θ) / (1 - Real.cos θ)) - Real.arctan ((1 + Real.sin θ) / (1 + Real.cos θ)))
  (hβ : β = Real.arctan (-(1 - Real.sin θ) / (1 + Real.cos θ)) - Real.arctan (-(1 + Real.sin θ) / (1 - Real.cos θ))) :
  (Real.tan α)^2 + (Real.tan β)^2 = 8 :=
sorry

end tangent_square_sum_eq_eight_l625_625545


namespace r_and_s_earns_per_day_l625_625058

variable (P Q R S : Real)

-- Conditions as given in the problem
axiom cond1 : P + Q + R + S = 2380 / 9
axiom cond2 : P + R = 600 / 5
axiom cond3 : Q + S = 800 / 6
axiom cond4 : Q + R = 910 / 7
axiom cond5 : P = 150 / 3

theorem r_and_s_earns_per_day : R + S = 143.33 := by
  sorry

end r_and_s_earns_per_day_l625_625058


namespace Bridget_weight_is_correct_l625_625127

-- Definitions based on conditions
def Martha_weight : ℕ := 2
def weight_difference : ℕ := 37

-- Bridget's weight based on the conditions
def Bridget_weight : ℕ := Martha_weight + weight_difference

-- Proof problem: Prove that Bridget's weight is 39
theorem Bridget_weight_is_correct : Bridget_weight = 39 := by
  -- Proof goes here
  sorry

end Bridget_weight_is_correct_l625_625127


namespace hat_guessing_strategy_exists_l625_625907

theorem hat_guessing_strategy_exists :
  ∀ (n : ℕ) (hat_numbers : Fin n → ℕ), 
  1 < n → (∀ i, 1 ≤ hat_numbers i ∧ hat_numbers i ≤ n) →
  ∃ i, hat_numbers i = 
        (∑ j in Finset.univ.filter (λ x, x ≠ i), hat_numbers j - 
        (∑ j in Finset.univ, hat_numbers j)) % n :=
by
  sorry

end hat_guessing_strategy_exists_l625_625907


namespace orthographic_projection_area_l625_625827

def side_length : ℝ := 2
def area_equilateral_triangle (a : ℝ) : ℝ := (sqrt 3 / 4) * a^2
def orthographic_projection (S : ℝ) : ℝ := (sqrt 2 / 4) * S

theorem orthographic_projection_area :
  orthographic_projection (area_equilateral_triangle side_length) = sqrt 6 / 4 :=
by
  sorry

end orthographic_projection_area_l625_625827


namespace number_of_truthful_dwarfs_l625_625583

-- Given conditions
variables (D : Type) [Fintype D] [DecidableEq D] [Card D = 10]
variables (IceCream : Type) [DecidableEq IceCream] (vanilla chocolate fruit : IceCream)
-- Assuming each dwarf likes exactly one type of ice cream
variable (Likes : D → IceCream)
-- Functions indicating if a dwarf raised their hand for each type of ice cream
variables (raisedHandForVanilla raisedHandForChocolate raisedHandForFruit : D → Prop)

-- Given conditions translated to Lean
axiom all_dwarfs_raised_for_vanilla : ∀ d, raisedHandForVanilla d
axiom half_dwarfs_raised_for_chocolate : Fintype.card {d // raisedHandForChocolate d} = 5
axiom one_dwarf_raised_for_fruit : Fintype.card {d // raisedHandForFruit d} = 1

-- Define that a dwarf either always tells the truth or always lies
inductive TruthStatus
| truthful : TruthStatus
| liar : TruthStatus

variable (Status : D → TruthStatus)

-- Definitions related to hand-raising based on dwarf's status and ice cream they like
def raisedHandCorrectly (d : D) : Prop :=
  match Status d with
  | TruthStatus.truthful => 
      raisedHandForVanilla d ↔ Likes d = vanilla ∧
      raisedHandForChocolate d ↔ Likes d = chocolate ∧
      raisedHandForFruit d ↔ Likes d = fruit
  | TruthStatus.liar =>
      raisedHandForVanilla d ↔ Likes d ≠ vanilla ∧
      raisedHandForChocolate d ↔ Likes d ≠ chocolate ∧
      raisedHandForFruit d ↔ Likes d ≠ fruit

-- Goal to prove
theorem number_of_truthful_dwarfs : Fintype.card {d // Status d = TruthStatus.truthful} = 4 :=
by sorry

end number_of_truthful_dwarfs_l625_625583


namespace hannah_strawberries_l625_625680

theorem hannah_strawberries (days give_away stolen remaining_strawberries x : ℕ) 
  (h1 : days = 30) 
  (h2 : give_away = 20) 
  (h3 : stolen = 30) 
  (h4 : remaining_strawberries = 100) 
  (hx : x = (remaining_strawberries + give_away + stolen) / days) : 
  x = 5 := 
by 
  -- The proof will go here
  sorry

end hannah_strawberries_l625_625680


namespace sequence_problem_l625_625997

theorem sequence_problem (S : ℕ → ℚ) (a : ℕ → ℚ) (h : ∀ n, S n + a n = 2 * n) :
  a 1 = 1 ∧ a 2 = 3 / 2 ∧ a 3 = 7 / 4 ∧ a 4 = 15 / 8 ∧ 
  (∀ n : ℕ, n > 0 → a n = (2^n - 1) / 2^(n-1)) :=
by
  sorry

end sequence_problem_l625_625997


namespace gratuity_calculation_correct_l625_625101

noncomputable def tax_rate (item: String): ℝ :=
  if item = "NY Striploin" then 0.10
  else if item = "Glass of wine" then 0.15
  else if item = "Dessert" then 0.05
  else if item = "Bottle of water" then 0.00
  else 0

noncomputable def base_price (item: String): ℝ :=
  if item = "NY Striploin" then 80
  else if item = "Glass of wine" then 10
  else if item = "Dessert" then 12
  else if item = "Bottle of water" then 3
  else 0

noncomputable def total_price_with_tax (item: String): ℝ :=
  base_price item + base_price item * tax_rate item

noncomputable def gratuity (item: String): ℝ :=
  total_price_with_tax item * 0.20

noncomputable def total_gratuity: ℝ :=
  gratuity "NY Striploin" + gratuity "Glass of wine" + gratuity "Dessert" + gratuity "Bottle of water"

theorem gratuity_calculation_correct :
  total_gratuity = 23.02 :=
by
  sorry

end gratuity_calculation_correct_l625_625101


namespace youngest_child_age_l625_625905

theorem youngest_child_age
  (ten_years_ago_avg_age : Nat) (family_initial_size : Nat) (present_avg_age : Nat)
  (age_difference : Nat) (age_ten_years_ago_total : Nat)
  (age_increase : Nat) (current_age_total : Nat)
  (current_family_size : Nat) (total_age_increment : Nat) :
  ten_years_ago_avg_age = 24 →
  family_initial_size = 4 →
  present_avg_age = 24 →
  age_difference = 2 →
  age_ten_years_ago_total = family_initial_size * ten_years_ago_avg_age →
  age_increase = family_initial_size * 10 →
  current_age_total = age_ten_years_ago_total + age_increase →
  current_family_size = family_initial_size + 2 →
  total_age_increment = current_family_size * present_avg_age →
  total_age_increment - current_age_total = 8 →
  ∃ (Y : Nat), Y + Y + age_difference = 8 ∧ Y = 3 :=
by
  intros
  sorry

end youngest_child_age_l625_625905


namespace number_of_distinct_pairs_l625_625962

theorem number_of_distinct_pairs :
  (∃ (x y : ℤ), 0 < x ∧ x < y ∧ real.sqrt 1690 = real.sqrt (x : ℝ) + real.sqrt (y : ℝ)) →
  (set.finite {p : ℤ × ℤ | 0 < p.1 ∧ p.1 < p.2 ∧ real.sqrt 1690 = real.sqrt (p.1 : ℝ) + real.sqrt (p.2 : ℝ)} ∧ 
  finset.card {p : ℤ × ℤ | 0 < p.1 ∧ p.1 < p.2 ∧ real.sqrt 1690 = real.sqrt (p.1 : ℝ) + real.sqrt (p.2 : ℝ)}.to_finset = 6) :=
by {
  sorry
}

end number_of_distinct_pairs_l625_625962


namespace lcm_20_45_75_l625_625883

def lcm (a b : ℕ) : ℕ := nat.lcm a b

theorem lcm_20_45_75 : lcm (lcm 20 45) 75 = 900 :=
by
  sorry

end lcm_20_45_75_l625_625883


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625442

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ (n : ℕ), (∃ k : ℕ, n = k^2) ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, (∃ k : ℕ, m = k^2) ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m :=
sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625442


namespace find_y_interval_l625_625610

noncomputable def satisfies_inequality (y : ℝ) : Prop :=
∀ x, (0 ≤ x ∧ x ≤ 2 * Real.pi) → cos (x - y) ≥ cos x - cos y

theorem find_y_interval :
  {y : ℝ | y ∈ set.Icc 0 (2 * Real.pi) ∧ satisfies_inequality y} = set.Icc 0 (2 * Real.pi) :=
by {
  sorry
}

end find_y_interval_l625_625610


namespace neg_and_implication_l625_625252

variable (p q : Prop)

theorem neg_and_implication : ¬ (p ∧ q) → ¬ p ∨ ¬ q := by
  sorry

end neg_and_implication_l625_625252


namespace no_such_number_exists_l625_625281

theorem no_such_number_exists : ¬∃ (N M : ℕ)(d : ℕ), (1 ≤ d ∧ d ≤ 9) ∧
  (∃ k ∈ {5, 6, 8}, N = d * 10 ^ k + M ∧ 10 * M + d = k * N) :=
sorry

end no_such_number_exists_l625_625281


namespace number_of_truthful_dwarfs_is_4_l625_625596

def dwarf := {x : ℕ // 1 ≤ x ≤ 10}
def likes_vanilla (d : dwarf) : Prop := sorry
def likes_chocolate (d : dwarf) : Prop := sorry
def likes_fruit (d : dwarf) : Prop := sorry
def tells_truth (d : dwarf) : Prop := sorry
def tells_lie (d : dwarf) : Prop := sorry

noncomputable def number_of_truthful_dwarfs : ℕ :=
  let total_dwarfs := 10 in
  let vanilla_raises := 10 in
  let chocolate_raises := 5 in
  let fruit_raises := 1 in
  -- T + L = total_dwarfs
  -- T + 2L = vanilla_raises + chocolate_raises + fruit_raises
  let T := total_dwarfs - 2 * (vanilla_raises + chocolate_raises + fruit_raises - total_dwarfs) in
  T

theorem number_of_truthful_dwarfs_is_4 : number_of_truthful_dwarfs = 4 := 
  by
    sorry

end number_of_truthful_dwarfs_is_4_l625_625596


namespace count_terms_simplified_expression_l625_625555

theorem count_terms_simplified_expression :
  let num_terms := ∑ a in finset.range 1005, 1005 - a
  num_terms = 508020 :=
by
  sorry

end count_terms_simplified_expression_l625_625555


namespace find_m_n_l625_625748

noncomputable def midpoint : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ)
| (x1, y1), (x2, y2) => ((x1 + x2) / 2, (y1 + y2) / 2)

def line1 : ℝ × ℝ → Prop := fun p => 6 * p.2 = 10 * p.1
def line2 : ℝ × ℝ → Prop := fun p => 12 * p.2 = 5 * p.1

theorem find_m_n (P Q : ℝ × ℝ) (hP : line1 P) (hQ : line2 Q)
  (h_midpoint : midpoint P Q = (10, 8)) :
  let PQ_dist := Euclidean_dist P Q in
  ∃ m n : ℕ, Nat.gcd m n = 1 ∧ PQ_dist = (1098 : ℚ) / 165 ∧ m + n = 1263 :=
begin 
  sorry
end

-- Additional helper function to calculate Euclidean distance
noncomputable def Euclidean_dist : (ℝ × ℝ) → (ℝ × ℝ) → ℝ
| (x1, y1), (x2, y2) => Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

end find_m_n_l625_625748


namespace MarksScore_l625_625258

theorem MarksScore (h_highest : ℕ) (h_range : ℕ) (h_relation : h_highest - h_least = h_range) (h_mark_twice : Mark = 2 * h_least) : Mark = 46 :=
by
    let h_highest := 98
    let h_range := 75
    let h_least := h_highest - h_range
    let Mark := 2 * h_least
    have := h_relation
    have := h_mark_twice
    sorry

end MarksScore_l625_625258


namespace bikes_can_be_assembled_l625_625921

def wheels (n : Nat) : Prop := n = 20
def wheels_per_bike (m : Nat) : Prop := m = 2

theorem bikes_can_be_assembled : ∀ n m : Nat, wheels n → wheels_per_bike m → (n / m) = 10 := by
  intros n m h_wheels h_wheels_per_bike
  rw [h_wheels, h_wheels_per_bike]
  exact Nat.div_eq_of_eq_mul_right (by decide) (by decide)

end bikes_can_be_assembled_l625_625921


namespace find_number_of_children_l625_625533

def admission_cost_adult : ℝ := 30
def admission_cost_child : ℝ := 15
def total_people : ℕ := 10
def soda_cost : ℝ := 5
def discount_rate : ℝ := 0.8
def total_paid : ℝ := 197

def total_cost_with_discount (adults children : ℕ) : ℝ :=
  discount_rate * (adults * admission_cost_adult + children * admission_cost_child)

theorem find_number_of_children (A C : ℕ) 
  (h1 : A + C = total_people)
  (h2 : total_cost_with_discount A C + soda_cost = total_paid) :
  C = 4 :=
sorry

end find_number_of_children_l625_625533


namespace area_of_trapezoid_DBCE_eq_36_l625_625536

theorem area_of_trapezoid_DBCE_eq_36 
  (ABC_is_isosceles : ∀ (A B C : Point), is_isosceles_triangle A B C)
  (area_smallest_triangle : ∀ (T : Triangle), area T = 1)
  (area_triangle_ABC : area ABC = 45) :
  area (trapezoid DBCE) = 36 := 
by sorry

end area_of_trapezoid_DBCE_eq_36_l625_625536


namespace seq_sum_eq_l625_625640

noncomputable def b : ℕ → ℚ
| 0       := 1 / 2
| (n + 1) := 1 / (2 - b n)

theorem seq_sum_eq : (∑ n in Finset.range 100, b (n + 1) / (n + 1 + 1) ^ 2) = 100 / 101 :=
by
  have h_bn : ∀ n, b (n + 1) = (n + 1 : ℚ) / (n + 2),
  { intro n, induction n with n hn,
    { rw [b, Nat.cast_zero, zero_add, div_self, b, Nat.add_succ, Nat.cast_add, Nat.cast_one,
          cast_two, sub_sub, sub_sub, div_div, div_self]; norm_cast; linarith },
    { have : b (n + 2) = 1 / (2 - b (n + 1)), by norm_cast,
      rw [this, hn, Nat.cast_succ],
      field_simp,
      ring_nf,
      norm_cast,
      linarith, }},
  apply Finset.sum_congr rfl,
  intros, 
  rw [h_bn, Nat.cast_add, Nat.cast_one, Nat.pow_two, Nat.cast_add, Nat.cast_one, Nat.cast_succ,
      Nat.cast_one, Nat.add_succ],
  field_simp,
  ring_nf,
  norm_cast,
  sorry

end seq_sum_eq_l625_625640


namespace solution_set_of_inequality_l625_625665

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2
  else if x > 0 then x - 2
  else 0

theorem solution_set_of_inequality :
  {x : ℝ | 2 * f x - 1 < 0} = {x | x < -3 / 2 ∨ (0 ≤ x ∧ x < 5 / 2)} :=
by
  sorry

end solution_set_of_inequality_l625_625665


namespace lambda_range_l625_625310

variable {S1 S2 S3 S4 : ℝ} (S : ℝ) (h1 : S1 ≤ S) (h2 : S2 ≤ S) (h3 : S3 ≤ S) (h4 : S4 ≤ S)
          (hmax : S = max (max S1 S2) (max S3 S4))

noncomputable def λ := (S1 + S2 + S3 + S4) / S

theorem lambda_range : 2 < λ S1 S2 S3 S4 S ∧ λ S1 S2 S3 S4 S ≤ 4 := by
  sorry

end lambda_range_l625_625310


namespace blake_initial_money_l625_625125

theorem blake_initial_money (amount_spent_oranges amount_spent_apples amount_spent_mangoes change_received initial_amount : ℕ)
  (h1 : amount_spent_oranges = 40)
  (h2 : amount_spent_apples = 50)
  (h3 : amount_spent_mangoes = 60)
  (h4 : change_received = 150)
  (h5 : initial_amount = (amount_spent_oranges + amount_spent_apples + amount_spent_mangoes) + change_received) :
  initial_amount = 300 :=
by
  sorry

end blake_initial_money_l625_625125


namespace expected_value_of_X_l625_625197

noncomputable def X : ℝ → MeasureTheory.Measure ℝ := MeasureTheory.ProbabilityTheory.Normal 6 (1 / 3)

theorem expected_value_of_X :
  @MeasureTheory.AeMeasureTheory.MeasureWithExpectation ℝ (X) MeasureTheory.ProbabilityTheory.μ 6 :=
by
  sorry

end expected_value_of_X_l625_625197


namespace age_of_new_person_l625_625807

theorem age_of_new_person (n : ℕ) (T A : ℕ) (h₁ : n = 10) (h₂ : T = 15 * n)
    (h₃ : (T + A) / (n + 1) = 17) : A = 37 := by
  sorry

end age_of_new_person_l625_625807


namespace initial_items_in_cart_l625_625534

theorem initial_items_in_cart (deleted_items : ℕ) (items_left : ℕ) (initial_items : ℕ) 
  (h1 : deleted_items = 10) (h2 : items_left = 8) : initial_items = 18 :=
by 
  -- Proof goes here
  sorry

end initial_items_in_cart_l625_625534


namespace compute_dot_product_l625_625295

noncomputable 
def vectors_eq (u v z : V) : Prop :=
  ∥v∥ = 2 ∧ ∥z∥ = 1 ∧ (u × v + z = u) ∧ (u × z = v)

theorem compute_dot_product (u v z : V) (h : vectors_eq u v z) : 
  z ⋅ (v × u) = -4 :=
sorry

end compute_dot_product_l625_625295


namespace abc_product_l625_625626

theorem abc_product (A B C D : ℕ) 
  (h1 : A + B + C + D = 64)
  (h2 : A + 3 = B - 3)
  (h3 : A + 3 = C * 3)
  (h4 : A + 3 = D / 3) :
  A * B * C * D = 19440 := 
by
  sorry

end abc_product_l625_625626


namespace math_proof_problem_l625_625171

variables {A B C O H P A1 B1 C1 A2 B2 C2 : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables [MetricSpace O] [MetricSpace H] [MetricSpace P]
variables [MetricSpace A1] [MetricSpace B1] [MetricSpace C1]
variables [MetricSpace A2] [MetricSpace B2] [MetricSpace C2]

-- Definitions of the circumcircle and orthocenter
def is_circumcircle (w : MetricSphere) (A B C : Type*) : Prop := sorry
def is_orthocenter (H : Type*) (A B C : Type*) : Prop := sorry
def is_interior_point (P : Type*) (A B C : Type*) : Prop := sorry
def reflect (X D: Type*) : Type* := sorry

-- Conditions of the problem given above
variables (w : MetricSphere)

def problem_conditions (A B C A1 B1 C1 A2 B2 C2 : Type*) :=
  is_circumcircle w A B C ∧
  is_interior_point P A B C ∧
  (∃ (ray_AP ray_BP ray_CP : Line),
    intersect ray_AP w = A1 ∧
    intersect ray_BP w = B1 ∧
    intersect ray_CP w = C1 ∧
    reflect A1 (midpoint B C) = A2 ∧
    reflect B1 (midpoint C A) = B2 ∧
    reflect C1 (midpoint A B) = C2)

-- Question to prove
def theorem_to_prove (A2 B2 C2 H : Type*) : Prop := 
  passes_through (circumcircle A2 B2 C2) H

-- Full problem statement
theorem math_proof_problem 
  (A B C A1 B1 C1 A2 B2 C2 H : Type*) (w : MetricSphere)
  (h_conditions : problem_conditions A B C A1 B1 C1 A2 B2 C2) :
  theorem_to_prove A2 B2 C2 H :=
sorry

end math_proof_problem_l625_625171


namespace LCM_20_45_75_is_900_l625_625874

def prime_factorization_20 := (2^2, 5)
def prime_factorization_45 := (3^2, 5)
def prime_factorization_75 := (3, 5^2)

theorem LCM_20_45_75_is_900 
  (pf_20 : prime_factorization_20 = (2^2, 5))
  (pf_45 : prime_factorization_45 = (3^2, 5))
  (pf_75 : prime_factorization_75 = (3, 5^2)) : 
  Nat.lcm (Nat.lcm 20 45) 75 = 900 := 
  by sorry

end LCM_20_45_75_is_900_l625_625874


namespace least_common_multiple_of_20_45_75_l625_625869

theorem least_common_multiple_of_20_45_75 :
  Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
sorry

end least_common_multiple_of_20_45_75_l625_625869


namespace count_middle_less_greater_l625_625894

def digit_sum (n : ℕ) : ℕ :=
let a := n / 100,
    b := (n / 10) % 10,
    c := n % 10 in
a + b + c

def middle_greater (n : ℕ) : Prop :=
let a := n / 100,
    b := (n / 10) % 10,
    c := n % 10 in
b > a ∧ b > c

def middle_less (n : ℕ) : Prop :=
let a := n / 100,
    b := (n / 10) % 10,
    c := n % 10 in
b < a ∧ b < c

theorem count_middle_less_greater :
  (100 ≤ count middle_less (100, 999)) >
  (100 ≤ count middle_greater (100, 999)) :=
sorry

end count_middle_less_greater_l625_625894


namespace solution_of_abs_square_eq_zero_l625_625234

-- Define the given conditions as hypotheses
variables {x y : ℝ}
theorem solution_of_abs_square_eq_zero (h : |x + 2| + (y - 1)^2 = 0) : x = -2 ∧ y = 1 :=
sorry

end solution_of_abs_square_eq_zero_l625_625234


namespace continuous_stripe_probability_l625_625146

noncomputable def probability_continuous_stripe : ℚ :=
  let total_configurations := 4^6
  let favorable_configurations := 48
  favorable_configurations / total_configurations

theorem continuous_stripe_probability : probability_continuous_stripe = 3 / 256 :=
  by
  sorry

end continuous_stripe_probability_l625_625146


namespace find_m_n_value_l625_625668

theorem find_m_n_value (x m n : ℝ) 
  (h1 : x - 3 * m < 0) 
  (h2 : n - 2 * x < 0) 
  (h3 : -1 < x)
  (h4 : x < 3) 
  : (m + n) ^ 2023 = -1 :=
sorry

end find_m_n_value_l625_625668


namespace rectangle_tiling_fibonacci_ninth_l625_625896

theorem rectangle_tiling_fibonacci_ninth :
  let f : ℕ → ℕ
  := λ n, if n = 0 then 0 else if n = 1 then 1 else f (n - 1) + f (n - 2)
  in f 9 = 34 :=
by {
  let f : ℕ → ℕ 
  := λ n, if n = 0 then 0 else if n = 1 then 1 else f (n - 1) + f (n - 2),
  have hf0 : f 0 = 0, from rfl,
  have hf1 : f 1 = 1, from rfl,
  have hf2 : f 2 = 1, from by 
    rw [←add_zero 1, hf0, hf1],
  have hf3 : f 3 = 2, from by 
    rw [←add_zero 2, by rw [hf2], by rw [hf1]],
  have hf4 : f 4 = 3, from by 
    rw [←add_zero 3, hf3, hf2],
  have hf5 : f 5 = 5, from by 
    rw [←add_zero 4, hf4, hf3],
  have hf6 : f 6 = 8, from by 
    rw [←add_zero 5, hf5, hf4],
  have hf7 : f 7 = 13, from by 
    rw [←add_zero 6, hf6, hf5],
  have hf8 : f 8 = 21, from by 
    rw [←add_zero 7, hf7, hf6],
  have hf9 : f 9 = 34, from by 
    rw [←add_zero 8, hf8, hf7],
  exact hf9
}

end rectangle_tiling_fibonacci_ninth_l625_625896


namespace bridge_length_is_correct_l625_625089

-- Definitions of the conditions
def speed_km_per_hr : ℝ := 5
def time_minutes : ℝ := 15
def speed_m_per_min : ℝ := (speed_km_per_hr * 1000) / 60 -- converting 5 km/hr to meters per minute
def correct_distance : ℝ := 1249.95

-- The actual distance calculation based on the given time
def distance_covered : ℝ := speed_m_per_min * time_minutes

-- Statement that the distance covered is equal to the correct distance
theorem bridge_length_is_correct : distance_covered = correct_distance :=
by
  sorry

end bridge_length_is_correct_l625_625089


namespace intersection_complement_l625_625672

def U := {1, 2, 3, 4, 5}
def A := {1, 2}
def B := {1, 4, 5}
def CU (u : Set Nat) := {x | x ∈ {1, 2, 3, 4, 5} ∧ x ∉ u}
def ACUB := {x | x ∈ A ∧ x ∈ CU B}

theorem intersection_complement :
  ACUB = {2} :=
by
  sorry

end intersection_complement_l625_625672


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625445

/-- Problem statement: 
    The smallest positive perfect square that is divisible by 2, 3, and 5 is 900.
-/
theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ n * n % 2 = 0 ∧ n * n % 3 = 0 ∧ n * n % 5 = 0 ∧ n * n = 900 := 
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625445


namespace max_value_7b_5c_l625_625657

theorem max_value_7b_5c (a b c : ℝ) (h_a : 1 ≤ a ∧ a ≤ 2)
  (h_f_x : ∀ x ∈ Icc (1 : ℝ) 2, a * x^2 + b * x + c ≤ 1) : 
  7 * b + 5 * c ≤ -6 :=
begin
  sorry -- proof to be completed
end

end max_value_7b_5c_l625_625657


namespace hoseok_multiplied_number_l625_625682

theorem hoseok_multiplied_number (n : ℕ) (h : 11 * n = 99) : n = 9 := 
sorry

end hoseok_multiplied_number_l625_625682


namespace mean_proportional_64_l625_625618

theorem mean_proportional_64 (M : ℝ) (B : ℝ) (h : M = 72.5) : 
  (M ^ 2) = 64 * B → B = 82.12890625 :=
by
  intro h₁
  have h2 : 72.5 ^ 2 = 5256.25 := by norm_num
  rw [h, h2] at h₁
  linarith

end mean_proportional_64_l625_625618


namespace sum_of_a_values_l625_625239

theorem sum_of_a_values : 
  (∀ (a x : ℝ), (a + x) / 2 ≥ x - 2 ∧ x / 3 - (x - 2) > 2 / 3 ∧ 
  (x - 1) / (4 - x) + (a + 5) / (x - 4) = -4 ∧ x < 2 ∧ (∃ n : ℤ, x = n ∧ 0 < n)) →
  ∃ I : ℤ, I = 12 :=
by
  sorry

end sum_of_a_values_l625_625239


namespace LCM_20_45_75_is_900_l625_625878

def prime_factorization_20 := (2^2, 5)
def prime_factorization_45 := (3^2, 5)
def prime_factorization_75 := (3, 5^2)

theorem LCM_20_45_75_is_900 
  (pf_20 : prime_factorization_20 = (2^2, 5))
  (pf_45 : prime_factorization_45 = (3^2, 5))
  (pf_75 : prime_factorization_75 = (3, 5^2)) : 
  Nat.lcm (Nat.lcm 20 45) 75 = 900 := 
  by sorry

end LCM_20_45_75_is_900_l625_625878


namespace An_is_integer_l625_625193

def A (a b : ℕ) (θ : ℝ) (n : ℕ) :=
  (a^2 + b^2)^n * Real.sin (n * θ)

axiom theta_def (a b : ℕ) (ha : a > b) : 
  ∃ θ, (0 < θ ∧ θ < Real.pi / 2) ∧ 
        Real.sin θ = 2 * a * b / (a^2 + b^2) ∧ 
        Real.cos θ = (a^2 - b^2) / (a^2 + b^2)

theorem An_is_integer (a b : ℕ) (ha : a > b) (n : ℕ) : 
  ∃ θ, (0 < θ ∧ θ < Real.pi / 2) ∧ 
        Real.sin θ = 2 * a * b / (a^2 + b^2) ∧ 
        Real.cos θ = (a^2 - b^2) / (a^2 + b^2) → 
        ∃ k : ℤ, A a b θ n = k :=
by
  sorry

end An_is_integer_l625_625193


namespace tony_exercises_hours_per_week_l625_625399

theorem tony_exercises_hours_per_week
  (distance_walked : ℝ)
  (speed_walking : ℝ)
  (distance_ran : ℝ)
  (speed_running : ℝ)
  (days_per_week : ℕ)
  (distance_walked = 3)
  (speed_walking = 3)
  (distance_ran = 10)
  (speed_running = 5)
  (days_per_week = 7) :
  let time_walking := distance_walked / speed_walking,
      time_running := distance_ran / speed_running,
      total_time_per_day := time_walking + time_running
  in total_time_per_day * days_per_week = 21 :=
by
  -- Proof goes here
  sorry

end tony_exercises_hours_per_week_l625_625399


namespace find_BP_l625_625278

open Real

variables (A B C M N O P : Point)
variables (AB BC AC BM BN BP AP PC : ℝ)
variables (triangle_ABC : Triangle A B C)
variables (point_M_on_AB : OnLineSegment M A B)
variables (point_N_on_BC : OnLineSegment N B C)
variables (BM_eq_BN : BM = BN)
variables (M_perpendicular_BC : Perpendicular (LinePassingThrough M) (LinePassingThrough B C))
variables (N_perpendicular_AB : Perpendicular (LinePassingThrough N) (LinePassingThrough A B))
variables (intersection_O : Intersect (LinePassingThrough M (LinePerpendicularTo M B C))
                                      (LinePassingThrough N (LinePerpendicularTo N A B)) O)
variables (extension_BO_intersects_AC_at_P : Intersect (LineExtension B O) (LinePassingThrough A C) P)
variables (AP_eq_5 : AP = 5)
variables (PC_eq_4 : PC = 4)
variables (BC_eq_6 : BC = 6)

theorem find_BP : BP = 5 := by
  sorry

end find_BP_l625_625278


namespace model_tower_height_l625_625769

theorem model_tower_height (real_height : ℝ) (real_volume : ℝ) (model_volume : ℝ) (h_real : real_height = 60) (v_real : real_volume = 200000) (v_model : model_volume = 0.2) :
  real_height / (real_volume / model_volume)^(1/3) = 0.6 :=
by
  rw [h_real, v_real, v_model]
  norm_num
  sorry

end model_tower_height_l625_625769


namespace smallest_positive_perfect_square_div_by_2_3_5_l625_625430

-- Definition capturing the conditions of the problem
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def divisible_by (n m : ℕ) : Prop :=
  m % n = 0

-- The theorem statement combining all conditions and requiring the proof of the correct answer
theorem smallest_positive_perfect_square_div_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ is_perfect_square n ∧ divisible_by 2 n ∧ divisible_by 3 n ∧ divisible_by 5 n ∧ n = 900 :=
sorry

end smallest_positive_perfect_square_div_by_2_3_5_l625_625430


namespace integer_solutions_to_equation_l625_625616

theorem integer_solutions_to_equation :
  ∃ (x y : ℤ), 2 * x^2 + 8 * y^2 = 17 * x * y - 423 ∧
               ((x = 11 ∧ y = 19) ∨ (x = -11 ∧ y = -19)) :=
by
  sorry

end integer_solutions_to_equation_l625_625616


namespace LCM_20_45_75_is_900_l625_625873

def prime_factorization_20 := (2^2, 5)
def prime_factorization_45 := (3^2, 5)
def prime_factorization_75 := (3, 5^2)

theorem LCM_20_45_75_is_900 
  (pf_20 : prime_factorization_20 = (2^2, 5))
  (pf_45 : prime_factorization_45 = (3^2, 5))
  (pf_75 : prime_factorization_75 = (3, 5^2)) : 
  Nat.lcm (Nat.lcm 20 45) 75 = 900 := 
  by sorry

end LCM_20_45_75_is_900_l625_625873


namespace isosceles_triangle_tangents_l625_625742

theorem isosceles_triangle_tangents (A B C D M N : Point) (Γ : Circle) :
  isosceles_triangle A B C ∧
  midpoint D B C ∧
  circle_centered_at D Γ ∧
  tangent_to_segment Γ A B ∧
  tangent_to_segment Γ A C ∧
  M_on_segment A B ∧
  N_on_segment A C ∧
  tangent_to_segment Γ M N →
  BD^2 = BM * CN :=
by
  sorry

end isosceles_triangle_tangents_l625_625742


namespace smallest_three_digit_solution_l625_625467

theorem smallest_three_digit_solution :
  ∃ n : ℕ, 70 * n ≡ 210 [MOD 350] ∧ 100 ≤ n ∧ n = 103 :=
by
  sorry

end smallest_three_digit_solution_l625_625467


namespace molecular_weight_of_NH4Cl_l625_625418

theorem molecular_weight_of_NH4Cl (weight_8_moles : ℕ) (weight_per_mole : ℕ) :
  weight_8_moles = 424 →
  weight_per_mole = 53 →
  weight_8_moles / 8 = weight_per_mole :=
by
  intro h1 h2
  sorry

end molecular_weight_of_NH4Cl_l625_625418


namespace lcm_of_20_45_75_l625_625860

-- Definitions for the given numbers and their prime factorizations
def num1 : ℕ := 20
def num2 : ℕ := 45
def num3 : ℕ := 75

def factor1 : ℕ → Prop := λ n, n = 2 ^ 2 * 5
def factor2 : ℕ → Prop := λ n, n = 3 ^ 2 * 5
def factor3 : ℕ → Prop := λ n, n = 3 * 5 ^ 2

-- The condition of using the least common multiple function from mathlib
def lcm_def (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- The statement to prove
theorem lcm_of_20_45_75 : lcm_def num1 num2 num3 = 900 := by
  -- Factors condition (Note: These help ensure the numbers' factors are as stated)
  have h1 : factor1 num1 := by { unfold num1 factor1, exact rfl }
  have h2 : factor2 num2 := by { unfold num2 factor2, exact rfl }
  have h3 : factor3 num3 := by { unfold num3 factor3, exact rfl }
  sorry -- This is the place where the proof would go.

end lcm_of_20_45_75_l625_625860


namespace equilateral_triangle_angle_gt_60_impossible_l625_625473

theorem equilateral_triangle_angle_gt_60_impossible :
  ∀ (T : Type*) [triangle T], (∃ (a b c : ℝ),
  triangle.is_equilateral T a b c ∧ angle a > 60) → false :=
by {
  sorry
}

end equilateral_triangle_angle_gt_60_impossible_l625_625473


namespace no_perfect_pairwise_meetings_l625_625922

theorem no_perfect_pairwise_meetings (N : ℕ) (M : ℕ) (D : ℕ → ℕ → Prop) :
  N = 100 → M = 3 →
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ N → ∃ d, D i j ∧ D j i → d ≤ D i j) →
  ∀ i, 1 ≤ i ∧ i ≤ N → 
  ¬ (∃ f : ℕ → finset (fin 100), (∀ n, (f n).card = M) ∧ 
       (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ N →
        ∃ (n : ℕ), n ∈ (finset.range 1650) ∧ (i ∈ f n) ∧ (j ∈ f n))) :=
begin
  sorry
end

end no_perfect_pairwise_meetings_l625_625922


namespace arithmetic_sequence_properties_b_sequence_sum_l625_625180

theorem arithmetic_sequence_properties 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_seq : ∀ n, a n = 2 * n + 1)
  (h_sum : ∀ n, S n = n^2 + 2 * n) :
  S 3 = 15 ∧ (a 3 + a 5) / 2 = 9 :=
by 
  sorry

theorem b_sequence_sum 
  (b : ℕ → ℝ) (T : ℕ → ℝ) 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_seq : ∀ n, a n = 2 * n + 1)
  (h_bn : ∀ n, b n = 4 / (a n ^ 2 - 1))
  (h_tn : ∀ k, T k = ∑ i in range (k + 1), (1 / (i + 1) - 1 / (i + 2))) :
  ∀ n, T n = n / (n + 1) :=
by 
  sorry

end arithmetic_sequence_properties_b_sequence_sum_l625_625180


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625449

/-- Problem statement: 
    The smallest positive perfect square that is divisible by 2, 3, and 5 is 900.
-/
theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ n * n % 2 = 0 ∧ n * n % 3 = 0 ∧ n * n % 5 = 0 ∧ n * n = 900 := 
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625449


namespace unique_solutions_of_system_l625_625136

def system_of_equations (x y : ℝ) : Prop :=
  (x - 1) * (x - 2) * (x - 3) = 0 ∧
  (|x - 1| + |y - 1|) * (|x - 2| + |y - 2|) * (|x - 3| + |y - 4|) = 0

theorem unique_solutions_of_system :
  ∀ (x y : ℝ), system_of_equations x y ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 3 ∧ y = 4) :=
by sorry

end unique_solutions_of_system_l625_625136


namespace lcm_of_20_45_75_l625_625861

-- Definitions for the given numbers and their prime factorizations
def num1 : ℕ := 20
def num2 : ℕ := 45
def num3 : ℕ := 75

def factor1 : ℕ → Prop := λ n, n = 2 ^ 2 * 5
def factor2 : ℕ → Prop := λ n, n = 3 ^ 2 * 5
def factor3 : ℕ → Prop := λ n, n = 3 * 5 ^ 2

-- The condition of using the least common multiple function from mathlib
def lcm_def (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- The statement to prove
theorem lcm_of_20_45_75 : lcm_def num1 num2 num3 = 900 := by
  -- Factors condition (Note: These help ensure the numbers' factors are as stated)
  have h1 : factor1 num1 := by { unfold num1 factor1, exact rfl }
  have h2 : factor2 num2 := by { unfold num2 factor2, exact rfl }
  have h3 : factor3 num3 := by { unfold num3 factor3, exact rfl }
  sorry -- This is the place where the proof would go.

end lcm_of_20_45_75_l625_625861


namespace share_of_B_l625_625790

-- Define the variables for the shares of A, B, and C
variables (A B C : ℝ)

-- Given conditions
def condition1 : Prop := A + B + C = 510
def condition2 : Prop := A = (2 / 3) * B
def condition3 : Prop := B = (1 / 4) * C

-- Theorem statement that we need to prove
theorem share_of_B : condition1 → condition2 → condition3 → B = 90 := by
  intros h1 h2 h3
  sorry

end share_of_B_l625_625790


namespace probability_of_ace_heart_queen_l625_625841
open Classical BigOperators

noncomputable def probability_first_ace_second_heart_third_queen : ℚ :=
  (3/52) * (12/51) * (4/50) + (3/52) * (1/51) * (3/50) +
  (1/52) * (11/51) * (4/50) + (1/52) * (1/51) * (3/50)

theorem probability_of_ace_heart_queen : 
  probability_first_ace_second_heart_third_queen = 1 / 663 :=
begin
  sorry
end

end probability_of_ace_heart_queen_l625_625841


namespace number_of_trailing_zeros_mod_100_l625_625745

theorem number_of_trailing_zeros_mod_100 :
  let P := ∏ (i : ℕ) in finset.range 51, nat.factorial i in
  let M := nat.trailing_zeroes P in
  M % 100 = 12 :=
by
  sorry

end number_of_trailing_zeros_mod_100_l625_625745


namespace all_students_competed_in_each_round_l625_625934

variable (N : ℕ)
variable (Teams : list (set ℕ))
variable (Rounds : list (set ℕ × set ℕ))

-- Conditions as definitions
def non_empty_teams (A B : set ℕ) : Prop := A ≠ ∅ ∧ B ≠ ∅
def disjoint_teams (A B : set ℕ) : Prop := A ∩ B = ∅
def complement_teams (A B : set ℕ) : Prop := A ∪ B = {i | i < N} ∧ ∀ i, i ∈ A ∨ i ∈ B
def unique_teams_in_round (Teams : list (set ℕ)) : Prop := 
  ∀ T, T ∈ Teams → (¬∃ T', T ≠ T' ∧ T' ∈ Teams ∧ T ∪ T' = {i | i < N})

-- Theorem statement
theorem all_students_competed_in_each_round 
  (hN : 1 ≤ N)
  (hTeams : ∀ {A B}, (A, B) ∈ Rounds → non_empty_teams A B)
  (hDisjoint : ∀ {A B}, (A, B) ∈ Rounds → disjoint_teams A B)
  (hComplement : ∀ {A B}, (A, B) ∈ Rounds → complement_teams A B)
  (hUnique : unique_teams_in_round Teams) :
  ∀ (A B : set ℕ), (A, B) ∈ Rounds → A ∪ B = {i | i < N} :=
begin
  sorry
end

end all_students_competed_in_each_round_l625_625934


namespace simplify_problem_l625_625341

noncomputable def simplify_expression : ℝ :=
  let numer := (Real.sqrt 3 - 1) ^ (1 - Real.sqrt 2)
  let denom := (Real.sqrt 3 + 1) ^ (1 + Real.sqrt 2)
  numer / denom

theorem simplify_problem :
  simplify_expression = 2 ^ (1 - Real.sqrt 2) * (4 - 2 * Real.sqrt 3) :=
by
  sorry

end simplify_problem_l625_625341


namespace add_solution_y_to_solution_x_l625_625344

theorem add_solution_y_to_solution_x
  (x_volume : ℝ) (x_percent : ℝ) (y_percent : ℝ) (desired_percent : ℝ) (final_volume : ℝ)
  (x_alcohol : ℝ := x_volume * x_percent / 100) (y : ℝ := final_volume - x_volume) :
  (x_percent = 10) → (y_percent = 30) → (desired_percent = 15) → (x_volume = 300) →
  (final_volume = 300 + y) →
  ((x_alcohol + y * y_percent / 100) / final_volume = desired_percent / 100) →
  y = 100 := by
    intros h1 h2 h3 h4 h5 h6
    sorry

end add_solution_y_to_solution_x_l625_625344


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625420

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, (n > 0 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∧ ∃ k : ℕ, n = k^2 ∧ n = 225 := 
begin
  sorry
end

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625420


namespace number_of_parallels_l625_625645

-- Definitions for the intersecting planes and the point.
variable (α β : Plane)
variable (A : Point)

-- Assumptions from the conditions.
axiom α_intersects_β : ∃ l : Line, l ∈ α ∧ l ∈ β 
axiom A_not_in_α : A ∉ α
axiom A_not_in_β : A ∉ β

-- Statement to be proven.
theorem number_of_parallels (α β : Plane) (A : Point) : 
  ∃! l : Line, passes_through l A ∧ (∀ π: Plane, π ∈ {α, β} → is_parallel_to l π) := 
  sorry

end number_of_parallels_l625_625645


namespace cylinder_height_l625_625246

theorem cylinder_height (r h : ℝ) 
  (H1 : 2 * real.pi * r * h = 12 * real.pi)
  (H2 : real.pi * r^2 * h = 12 * real.pi) : 
  h = 3 := 
sorry

end cylinder_height_l625_625246


namespace sum_2_pow_x_eq_1992_l625_625378

noncomputable def x : ℕ → ℝ
| 0     := 1992
| (n + 1) := -1992 / (n + 1) * ∑ k in Finset.range (n + 1), x k

theorem sum_2_pow_x_eq_1992 : (∑ n in Finset.range 1993, 2^n * x n) = 1992 :=
by sorry

end sum_2_pow_x_eq_1992_l625_625378


namespace boat_speed_in_still_water_l625_625909

theorem boat_speed_in_still_water (b : ℝ) (h : (36 / (b - 2)) - (36 / (b + 2)) = 1.5) : b = 10 :=
by
  sorry

end boat_speed_in_still_water_l625_625909


namespace train_pass_platform_in_21_6_seconds_l625_625105

/-- Length of the train in meters --/
def train_length : ℕ := 120

/-- Length of the platform in meters --/
def platform_length : ℕ := 240

/-- Speed of the train in kmph --/
def train_speed_kmph : ℕ := 60

/-- Convert speed from kmph to mps (meters per second) --/
def speed_mps : ℝ := (train_speed_kmph * 1000) / 3600

/-- Total distance to be covered by the train --/
def total_distance : ℕ := train_length + platform_length

/-- Time to pass the platform calculated as total distance / speed --/
def time_to_pass_platform : ℝ := total_distance / speed_mps

theorem train_pass_platform_in_21_6_seconds : time_to_pass_platform ≈ 21.6 := by
  sorry

end train_pass_platform_in_21_6_seconds_l625_625105


namespace muffs_bought_before_december_correct_l625_625951

/-- Total ear muffs bought by customers in December. -/
def muffs_bought_in_december := 6444

/-- Total ear muffs bought by customers in all. -/
def total_muffs_bought := 7790

/-- Ear muffs bought before December. -/
def muffs_bought_before_december : Nat :=
  total_muffs_bought - muffs_bought_in_december

/-- Theorem stating the number of ear muffs bought before December. -/
theorem muffs_bought_before_december_correct :
  muffs_bought_before_december = 1346 :=
by
  unfold muffs_bought_before_december
  unfold total_muffs_bought
  unfold muffs_bought_in_december
  sorry

end muffs_bought_before_december_correct_l625_625951


namespace exists_fixed_point_l625_625326

theorem exists_fixed_point (P Q : ℝ → ℝ²) (O : ℝ²) (v : ℝ) 
  (hP : ∃ t1, P t1 = O) 
  (hQ : ∃ t2, Q t2 = O)
  (h_const_speed : ∀ t, ∥P t - P (t - 1)∥ = ∥Q t - Q (t - 1)∥ := v)
  : ∃ A : ℝ², ∀ t : ℝ, ∥A - P t∥ = ∥A - Q t∥ :=
sorry

end exists_fixed_point_l625_625326


namespace parameterization_properties_l625_625011

theorem parameterization_properties (a b c d : ℚ)
  (h1 : a * (-1) + b = -3)
  (h2 : c * (-1) + d = 5)
  (h3 : a * 2 + b = 4)
  (h4 : c * 2 + d = 15) :
  a^2 + b^2 + c^2 + d^2 = 790 / 9 :=
sorry

end parameterization_properties_l625_625011


namespace calculate_q1_l625_625763

-- Define audacious polynomial
def audacious (q : ℝ → ℝ) : Prop :=
  ∀ x, q(q(x)) = 0 → x = r ∨ x = s ∨ x = t

-- Given conditions
variables {r s t : ℝ}
def q : ℝ → ℝ := λ x, (x - r) * (x - s) * (x - t)

-- Main theorem statement (without the explicit solution steps)
theorem calculate_q1 :
  audacious q →
  q(1) = correct_value :=
by
  sorry

end calculate_q1_l625_625763


namespace area_of_triangle_MNC_is_correct_l625_625999

noncomputable def area_triangle_MNC : ℝ :=
  let A := (0 : ℝ, 0 : ℝ)
  let B := (1 : ℝ, 0 : ℝ)
  let C := (1 : ℝ, 1 : ℝ)
  let D := (0 : ℝ, 1 : ℝ)
  let N := (1 : ℝ, 1 + Real.sqrt 2)
  let M := ((1 + 0) / 2, (0 + 1) / 2)
  1 / 2 * Real.dist M C * Real.dist C N

theorem area_of_triangle_MNC_is_correct :
  area_triangle_MNC = Real.sqrt 2 / 4 :=
sorry

end area_of_triangle_MNC_is_correct_l625_625999


namespace no_nonzero_integer_solution_l625_625786

theorem no_nonzero_integer_solution (m n p : ℤ) :
  (m + n * Real.sqrt 2 + p * Real.sqrt 3 = 0) → (m = 0 ∧ n = 0 ∧ p = 0) :=
by sorry

end no_nonzero_integer_solution_l625_625786


namespace solve_for_x_l625_625345

theorem solve_for_x (x : ℝ) (h : 12 - 2 * x = 6) : x = 3 :=
sorry

end solve_for_x_l625_625345


namespace num_divisors_of_450_pow8_l625_625574

def is_square (a b c : ℕ) : Prop :=
  even a ∧ even b ∧ even c

def is_cube (a b c : ℕ) : Prop :=
  (a % 3 = 0) ∧ (b % 3 = 0) ∧ (c % 3 = 0)

def is_sixth_power (a b c : ℕ) : Prop :=
  (a % 6 = 0) ∧ (b % 6 = 0) ∧ (c % 6 = 0)

theorem num_divisors_of_450_pow8
  (S : set (ℕ × ℕ × ℕ)) (C : set (ℕ × ℕ × ℕ)) (SC : set (ℕ × ℕ × ℕ)) :
  (∀ a b c, (a, b, c) ∈ S ↔ is_square a b c) →
  (∀ a b c, (a, b, c) ∈ C ↔ is_cube a b c) →
  (∀ a b c, (a, b, c) ∈ SC ↔ is_sixth_power a b c) →
  S.card = 405 →
  C.card = 108 →
  SC.card = 18 →
  S.card + C.card - SC.card = 495 :=
by sorry

end num_divisors_of_450_pow8_l625_625574


namespace parallelogram_base_l625_625157

-- Define the conditions given in the problem
def area : ℝ := 704
def height : ℝ := 22

-- State the theorem to prove the base of the parallelogram
theorem parallelogram_base (a h : ℝ) (h₁ : a = 704) (h₂ : h = 22) : a / h = 32 :=
by 
  -- Skipping the proof steps as mentioned
  sorry

end parallelogram_base_l625_625157


namespace shop_distance_l625_625401

noncomputable def G := (0, 400 : ℝ) -- Girls' camp coordinates
noncomputable def B := (800, 0 : ℝ) -- Boys' camp coordinates

def equidistant_point_from_G_and_B (S : ℝ × ℝ) : Prop :=
  dist S G = dist S B

theorem shop_distance :
  ∃ S : ℝ × ℝ, 
  equidistant_point_from_G_and_B S ∧
  dist S G = 500 ∧
  dist S B = 500 :=
by
  -- Placeholder for the proof
  sorry

end shop_distance_l625_625401


namespace problem1_problem2_problem3_problem4_l625_625895

variables (x : ℝ)

def sideLength := 20 * Real.sqrt 5 * x

def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

noncomputable def A := (0, 0)
noncomputable def B := (0, sideLength x)
noncomputable def C := (sideLength x, sideLength x)
noncomputable def D := (sideLength x, 0)

noncomputable def P := midpoint (D x) (C x)
noncomputable def Q := midpoint (B x) (C x)

theorem problem1 (x : ℝ) : 
  distance (A x) (P x) = 50 * x := sorry

theorem problem2 (x : ℝ) : 
  distance (P x) (Q x) = 10 * Real.sqrt 10 * x := sorry

theorem problem3 (x : ℝ) : 
  let diag := distance (A x) (C x)
  let distToPQ := distance (C x) (Q x) / Real.sqrt 2
  (diag - distToPQ) = 15 * Real.sqrt 10 * x := sorry

theorem problem4 (x : ℝ) (θ : ℝ) : 
  real.sqrt((50 * x) ^ 2 + (10 * Real.sqrt 10 * x) ^ 2) = 100 * sin θ := 
  let area1 := (1 / 2) * (50 * x) ^ 2 * sin θ
  let area2 := (1 / 2) * 10 * Real.sqrt 10 * x * 15 * Real.sqrt 10 * x
  area1 = area2 :=
  (3 / 5) := 
  d = 60 := sorry

end problem1_problem2_problem3_problem4_l625_625895


namespace sum_over_a_l625_625312

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → 2 * a n = a (n + 1)

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = ∑ i in range (n + 1), a i

theorem sum_over_a (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : is_geometric_sequence a) (h2 : sum_first_n_terms a S) :
  S 4 / a 2 = 15 / 2 :=
by
  sorry

end sum_over_a_l625_625312


namespace solution_set_l625_625637

noncomputable def f : ℝ → ℝ := sorry

axiom f_add : ∀ x₁ x₂ : ℝ, f(x₁ + x₂) = f(x₁) + f(x₂)
axiom f_pos : ∀ x : ℝ, x > 0 → f(x) > 0
axiom f_one : f(1) = 1

theorem solution_set (x : ℝ) : 
  (2^(1 + f(x)) + 2^(1 - f(x)) + 2 * f(x^2) ≤ 7) ↔ (x ∈ Icc (-1) 1) :=
sorry

end solution_set_l625_625637


namespace find_a_l625_625692

theorem find_a (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h1 : a^b = b^a) (h2 : b = 4 * a) : a = real.cbrt 4 :=
sorry

end find_a_l625_625692


namespace y_seq_limit_l625_625377

noncomputable def x_seq : ℕ → ℝ
| 0     := 1 / 2
| (n+1) := (Real.sqrt (x_seq n ^ 2 + 4 * x_seq n) + x_seq n) / 2

noncomputable def y_seq (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), 1 / (x_seq i) ^ 2

theorem y_seq_limit : ∃ L : ℝ, Filter.Tendsto y_seq Filter.atTop (Filter.const ℝ L) ∧ L = 6 := 
by { sorry }

end y_seq_limit_l625_625377


namespace percentage_bobby_pins_rounded_l625_625544

axiom annie_barettes : ℕ := 6
axiom scrunchies_eq_twice_barettes : ℕ := 2 * annie_barettes
axiom bobby_pins_eq_barettes_minus_three : ℕ := annie_barettes - 3
axiom total_decorations_eq_sum : ℕ := annie_barettes + scrunchies_eq_twice_barettes + bobby_pins_eq_barettes_minus_three

theorem percentage_bobby_pins_rounded : 
  (Float.ofNat bobby_pins_eq_barettes_minus_three / Float.ofNat total_decorations_eq_sum * 100).round = 14 :=
  sorry

end percentage_bobby_pins_rounded_l625_625544


namespace mn_eq_one_l625_625203

noncomputable def f (x : ℝ) : ℝ := |Real.log x / Real.log 2|

variables (m n : ℝ) (hmn : m < n) (hm_pos : 0 < m) (hn_pos : 0 < n) (hmn_equal : f m = f n)

theorem mn_eq_one : m * n = 1 := by
  sorry

end mn_eq_one_l625_625203


namespace find_integer_pairs_l625_625141

theorem find_integer_pairs :
  ∀ (a b : ℕ), 0 < a → 0 < b → a * b + 2 = a^3 + 2 * b →
  (a = 1 ∧ b = 1) ∨ (a = 3 ∧ b = 25) ∨ (a = 4 ∧ b = 31) ∨ (a = 5 ∧ b = 41) ∨ (a = 8 ∧ b = 85) :=
by
  intros a b ha hb hab_eq
  -- Proof goes here
  sorry

end find_integer_pairs_l625_625141


namespace mass_percentage_of_C_in_acetone_l625_625617

theorem mass_percentage_of_C_in_acetone :
  let mass_C := 12.01
  let mass_H := 1.008
  let mass_O := 16.00
  let acetone_formula := (3, 6, 1) -- represents C3H6O
  let molar_mass := 3 * mass_C + 6 * mass_H + mass_O
  let mass_percentage_C := (3 * mass_C / molar_mass) * 100
  in mass_percentage_C = 62.04 :=
by
  let mass_C := 12.01
  let mass_H := 1.008
  let mass_O := 16.00
  let acetone_formula := (3, 6, 1) -- represents C3H6O
  let molar_mass := 3 * mass_C + 6 * mass_H + mass_O
  let mass_percentage_C := (3 * mass_C / molar_mass) * 100
  sorry

end mass_percentage_of_C_in_acetone_l625_625617


namespace find_ellipse_properties_l625_625181

variables (a b : ℝ)
variable (M : ℝ × ℝ)
variable (P : ℝ × ℝ)
variable (Q : ℝ × ℝ)
variable (D : ℝ × ℝ)
variable (O : ℝ × ℝ := (0,0))

noncomputable theory

def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

def midpoint_condition (a b : ℝ) (M : ℝ × ℝ) : Prop :=
  M = (a / 2, 0)

def BM_AB_ratio (a b : ℝ) (M : ℝ × ℝ) : Prop :=
  (sqrt (b^2 + (a/2)^2)) / (sqrt (a^2 + b^2)) = sqrt 6 / 4

def first_quadrant_condition (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 > 0 ∧ (P.1^2 / a^2 + P.2^2 / b^2 = 1)

def angle_condition (a b : ℝ) (M P : ℝ × ℝ) : Prop :=
  real.angle_OF_coords M O P = π / 6

def PQ_intersects_ellipse (a b : ℝ) (P Q : ℝ × ℝ) : Prop :=
  P.1 * Q.1 + P.2 * Q.2 = 0 ∧ ellipse a b Q.1 Q.2

def AQ_PM_intersect (a b : ℝ) (A Q M D : ℝ × ℝ) : Prop :=
  -- Placeholder for intersection definition

-- Area of triangle PDQ is 5√3/12
def triangle_area (P D Q : ℝ × ℝ) : Prop :=
  -- Using the triangle area formula for coordinates
  abs ((P.1 * (D.2 - Q.2) + D.1 * (Q.2 - P.2) + Q.1 * (P.2 - D.2)) / 2) = 5 * sqrt 3 / 12

def eccentricity_and_standard_eq (a b : ℝ) : Prop :=
  let e := (2 * sqrt 5) / 5 in
  a^2 = 5 * b^2 ∧ a > b ∧ b > 0 ∧ 
  ellipse a b =
    √(5 + 1) * a / b

theorem find_ellipse_properties :
  ∃ a b M P Q D, 
    ellipse a b a b ∧
    midpoint_condition a b M ∧
    BM_AB_ratio a b M ∧
    first_quadrant_condition a b P ∧
    angle_condition a b M P ∧
    PQ_intersects_ellipse a b P Q ∧
    AQ_PM_intersect a b (a, 0) Q M D ∧
    triangle_area P D Q ∧
    eccentricity_and_standard_eq a b :=
sorry

end find_ellipse_properties_l625_625181


namespace problem_l625_625230

variable (a b : ℝ)

theorem problem (h₁ : a + b = 2) (h₂ : a - b = 3) : a^2 - b^2 = 6 := by
  sorry

end problem_l625_625230


namespace confectioner_pastry_l625_625503

theorem confectioner_pastry (P : ℕ) (h : P / 28 - 6 = P / 49) : P = 378 :=
sorry

end confectioner_pastry_l625_625503


namespace domain_h_l625_625973

noncomputable def h (x : ℝ) : ℝ := (x^3 + 11 * x - 2) / ((x^2 - 9) + |x + 1|)

theorem domain_h :
  ∀ x : ℝ, ((x^2 - 9) + |x + 1| ≠ 0) ↔ (x ∈ (-∞, -3) ∪ (-3, 2) ∪ (2, 3) ∪ (3, 5) ∪ (5, ∞)) :=
by
  -- proof would go here
  sorry

end domain_h_l625_625973


namespace parabola_properties_l625_625992

-- Definitions of the conditions
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def point_A (a b c : ℝ) : Prop := parabola a b c (-1) = 0
def point_B (a b c m : ℝ) : Prop := parabola a b c m = 0
def opens_downwards (a : ℝ) : Prop := a < 0
def valid_m (m : ℝ) : Prop := 1 < m ∧ m < 2

-- Conclusion ①
def conclusion_1 (a b : ℝ) : Prop := b > 0

-- Conclusion ②
def conclusion_2 (a c : ℝ) : Prop := 3 * a + 2 * c < 0

-- Conclusion ③
def conclusion_3 (a b c x1 x2 y1 y2 : ℝ) : Prop :=
  x1 < x2 ∧ x1 + x2 > 1 ∧ parabola a b c x1 = y1 ∧ parabola a b c x2 = y2 → y1 > y2

-- Conclusion ④
def conclusion_4 (a b c : ℝ) : Prop :=
  a ≤ -1 → ∃ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 1) ∧ (a * x2^2 + b * x2 + c = 1) ∧ (x1 ≠ x2)

-- The theorem to prove
theorem parabola_properties (a b c m : ℝ) :
  (opens_downwards a) →
  (point_A a b c) →
  (point_B a b c m) →
  (valid_m m) →
  (conclusion_1 a b) ∧ (conclusion_2 a c → false) ∧ (∀ x1 x2 y1 y2, conclusion_3 a b c x1 x2 y1 y2) ∧ (conclusion_4 a b c) :=
by
  sorry

end parabola_properties_l625_625992


namespace continuous_x_cubed_l625_625785

theorem continuous_x_cubed : ContinuousOn (λ x : ℝ, x^3) Set.univ :=
begin
    sorry
end

end continuous_x_cubed_l625_625785


namespace find_greatest_integer_not_exceeding_1000y_l625_625108

-- Define the conditions based on the problem statement
def cube_edge : ℝ := 2
def shadow_area_without_cube : ℝ := 147
def shadow_area_with_cube : ℝ := shadow_area_without_cube + cube_edge^2
def shadow_side_length : ℝ := Real.sqrt shadow_area_with_cube
def half_diagonal_of_top_face : ℝ := Real.sqrt 2
def half_diagonal_of_shadow : ℝ := shadow_side_length / 2

-- Define y based on similar triangles
def y : ℝ := (Real.sqrt shadow_area_with_cube - cube_edge) / 2

-- Define the greatest integer that does not exceed 1000y
def greatest_integer_not_exceeding_1000y : ℕ := floor (1000 * y)

-- The theorem to be proved
theorem find_greatest_integer_not_exceeding_1000y : 
  greatest_integer_not_exceeding_1000y = 5150 :=
by sorry -- The proof is omitted

end find_greatest_integer_not_exceeding_1000y_l625_625108


namespace area_proof_l625_625492

variable (x y z : ℤ)
variable (PQ PS RS : ℝ)
variable (T U : ℝ)
variable (PQ' QT : ℝ)

-- Conditions
variable (is_rectangle : PQ * PQ = PS * RS)
variable (T_on_PQ : T ∈ PQ)
variable (U_on_RS : U ∈ RS)
variable (QT_lt_RU : QT < RU)
variable (PQ'_eq_7 : PQ' = 7)
variable (QT_eq_29 : QT = 29)
variable (area_expr : x + y * sqrt z)

-- Required to prove: 
theorem area_proof : 
  PQ == 14 * sqrt 10 + 29 → 
  area_expr == 203 + 10 * sqrt 350 →
  x + y + z = 563 :=
by
  assume hPQ hAreaExpr
  -- skipping proof 
  sorry

end area_proof_l625_625492


namespace truthful_dwarfs_count_l625_625591

def dwarf (n : ℕ) := n < 10
def vanilla_ice_cream (n : ℕ) := dwarf n ∧ (∀ m, dwarf m)
def chocolate_ice_cream (n : ℕ) := dwarf n ∧ m % 2 = 0
def fruit_ice_cream (n : ℕ) := dwarf n ∧ m % 9 = 0

theorem truthful_dwarfs_count :
  ∃ T L : ℕ, T + L = 10 ∧ T + 2 * L = 16 ∧ T = 4 :=
by
  sorry

end truthful_dwarfs_count_l625_625591


namespace smallest_perfect_square_divisible_by_2_3_5_l625_625464

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, (n > 0) ∧ (nat.sqrt n * nat.sqrt n = n) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ n = 900 := sorry

end smallest_perfect_square_divisible_by_2_3_5_l625_625464


namespace monotonicity_f_f_greater_negone_l625_625655

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 - a * x + (a - 1) * Real.log x

theorem monotonicity_f (a : ℝ) (h_a : a > 1) :
  (∀ x ∈ Ioi (a - 1), x ∈ Ioc 0 1 → ∀ y ∈ Ioi x, f(y, a) > f(x, a)) ∧
  (∀ x ∈ Ioi 1, ∀ y ∈ Ioi x, f(y, a) > f(x, a)) ∧
  (∀ ε : ℝ, 0 < ε → ∀ δ : ℝ, δ > 0 → ∀ x ∈ Ioc 0 ε, ∀ y ∈ Ioi x, f(x, a) < f(y, a)) ∧
  (∀ x y : ℝ, x ∈ Icc 0 1 ∧ y ∈ Icc 0 1 ∧ x < y → f(x, a) > f(y, a)) :=
sorry

theorem f_greater_negone (a x1 x2 : ℝ) (h_a1 : a > 1) (h_a2 : a < 5) (h_x1 : x1 ∈ Ioi 0) (h_x2 : x2 ∈ Ioi 0) (h_diff : x1 ≠ x2) :
  (f(x1, a) - f(x2, a)) / (x1 - x2) > -1 :=
sorry

end monotonicity_f_f_greater_negone_l625_625655


namespace base_eight_to_base_ten_642_l625_625039

theorem base_eight_to_base_ten_642 :
  let d0 := 2
  let d1 := 4
  let d2 := 6
  let base := 8
  d0 * base^0 + d1 * base^1 + d2 * base^2 = 418 := 
by
  sorry

end base_eight_to_base_ten_642_l625_625039


namespace nine_point_circle_equation_l625_625974

theorem nine_point_circle_equation 
  (α β γ : ℝ) 
  (x y z : ℝ) :
  (x^2 * (Real.sin α) * (Real.cos α) + y^2 * (Real.sin β) * (Real.cos β) + z^2 * (Real.sin γ) * (Real.cos γ) = 
  y * z * (Real.sin α) + x * z * (Real.sin β) + x * y * (Real.sin γ))
:= sorry

end nine_point_circle_equation_l625_625974


namespace find_nm_l625_625159

theorem find_nm :
  ∃ n m : Int, (-120 : Int) ≤ n ∧ n ≤ 120 ∧ (-120 : Int) ≤ m ∧ m ≤ 120 ∧ 
  (Real.sin (n * Real.pi / 180) = Real.sin (580 * Real.pi / 180)) ∧ 
  (Real.cos (m * Real.pi / 180) = Real.cos (300 * Real.pi / 180)) ∧ 
  n = -40 ∧ m = -60 := by
  sorry

end find_nm_l625_625159


namespace number_of_truthful_dwarfs_is_4_l625_625594

def dwarf := {x : ℕ // 1 ≤ x ≤ 10}
def likes_vanilla (d : dwarf) : Prop := sorry
def likes_chocolate (d : dwarf) : Prop := sorry
def likes_fruit (d : dwarf) : Prop := sorry
def tells_truth (d : dwarf) : Prop := sorry
def tells_lie (d : dwarf) : Prop := sorry

noncomputable def number_of_truthful_dwarfs : ℕ :=
  let total_dwarfs := 10 in
  let vanilla_raises := 10 in
  let chocolate_raises := 5 in
  let fruit_raises := 1 in
  -- T + L = total_dwarfs
  -- T + 2L = vanilla_raises + chocolate_raises + fruit_raises
  let T := total_dwarfs - 2 * (vanilla_raises + chocolate_raises + fruit_raises - total_dwarfs) in
  T

theorem number_of_truthful_dwarfs_is_4 : number_of_truthful_dwarfs = 4 := 
  by
    sorry

end number_of_truthful_dwarfs_is_4_l625_625594


namespace concurrency_of_circles_l625_625740

noncomputable def point := ℝ × ℝ

structure Circle (center: point) (radius: ℝ) :=
(h_radius_nonneg : radius ≥ 0)

-- Definitions and conditions from the problem
variable (O A B E F : point)
variable (R R' : ℝ)
variable (k k' : Circle O R) (k'_extended : Circle O R')
variable (hR_lt_R' : R < R')
variable (h_line1 : ∃ L : set point, line_through O A B L)
variable (h_line2 : ∃ L : set point, line_through O E F L)
variable (hO_between_A_B : is_between O A B)
variable (hO_between_E_F : is_between O E F)

-- Statement of the theorem to be proven
theorem concurrency_of_circles :
  ∃ M : point, (M ∈ circumcircle (O, A, E)) ∧ 
               (M ∈ circumcircle (O, B, F)) ∧
               (M ∈ circle_with_diameter (E, F)) ∧
               (M ∈ circle_with_diameter (A, B)) :=
sorry

end concurrency_of_circles_l625_625740


namespace least_common_multiple_of_20_45_75_l625_625870

theorem least_common_multiple_of_20_45_75 :
  Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
sorry

end least_common_multiple_of_20_45_75_l625_625870


namespace all_numbers_divisible_by_41_l625_625183

theorem all_numbers_divisible_by_41 (a : Fin 1000 → Int) 
  (h : ∀ (k : Fin 1000), (∑ i in Finset.range 41, a ((k + i) % 1000).val ^ 2) % (41^2) = 0) :
  ∀ k, 41 ∣ a k :=
by
  sorry

end all_numbers_divisible_by_41_l625_625183


namespace ratio_QP_l625_625815

theorem ratio_QP {P Q : ℚ} 
  (h : ∀ x : ℝ, x ≠ 0 → x ≠ 4 → x ≠ -4 → 
    P / (x^2 - 5 * x) + Q / (x + 4) = (x^2 - 3 * x + 8) / (x^3 - 5 * x^2 + 4 * x)) : 
  Q / P = 7 / 2 := 
sorry

end ratio_QP_l625_625815


namespace smallest_positive_perfect_square_div_by_2_3_5_l625_625429

-- Definition capturing the conditions of the problem
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def divisible_by (n m : ℕ) : Prop :=
  m % n = 0

-- The theorem statement combining all conditions and requiring the proof of the correct answer
theorem smallest_positive_perfect_square_div_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ is_perfect_square n ∧ divisible_by 2 n ∧ divisible_by 3 n ∧ divisible_by 5 n ∧ n = 900 :=
sorry

end smallest_positive_perfect_square_div_by_2_3_5_l625_625429


namespace set_union_example_l625_625767

open Set

/-
  Conditions:
  M = {-1, 0, 2, 4}
  N = {0, 2, 3, 4}
  
  Prove:
  M ∪ N = {-1, 0, 2, 3, 4}
-/
theorem set_union_example :
  let M := { -1, 0, 2, 4 }
  let N := { 0, 2, 3, 4 }
  M ∪ N = { -1, 0, 2, 3, 4 } :=
by
  sorry

end set_union_example_l625_625767


namespace polygon_not_hexagon_if_quadrilateral_after_cut_off_l625_625694

-- Definition of polygonal shape and quadrilateral condition
def is_quadrilateral (sides : Nat) : Prop := sides = 4

-- Definition of polygonal shape with general condition of cutting off one angle
def after_cut_off (original_sides : Nat) (remaining_sides : Nat) : Prop :=
  original_sides > remaining_sides ∧ remaining_sides + 1 = original_sides

-- Problem statement: If a polygon's one angle cut-off results in a quadrilateral, then it is not a hexagon
theorem polygon_not_hexagon_if_quadrilateral_after_cut_off
  (original_sides : Nat) (remaining_sides : Nat) :
  after_cut_off original_sides remaining_sides → is_quadrilateral remaining_sides → original_sides ≠ 6 :=
by
  sorry

end polygon_not_hexagon_if_quadrilateral_after_cut_off_l625_625694


namespace height_of_platform_l625_625713

variables (l w h : ℕ)

theorem height_of_platform (hl1 : l + h - 2 * w = 36) (hl2 : w + h - l = 30) (hl3 : h = 2 * w) : h = 44 := 
sorry

end height_of_platform_l625_625713


namespace a_2010_value_l625_625994

noncomputable theory

def a : ℕ → ℤ
| 0     := 3 -- assuming definition with 0-based index for mathematical convenience
| 1     := 6
| (n+2) := a (n + 1) - a n

theorem a_2010_value : a 2009 = -3 := 
by sorry

end a_2010_value_l625_625994


namespace rectangle_area_l625_625959

def isSquare (A B C D : ℝ × ℝ) (s : ℝ) : Prop :=
  (A.1 = 0 ∧ A.2 = 0) ∧ (B.1 = s ∧ B.2 = 0) ∧ (C.1 = s ∧ C.2 = s) ∧ (D.1 = 0 ∧ D.2 = s)

def isParallel (l1 l2 : (ℝ × ℝ) → (ℝ × ℝ) → Prop) : Prop :=
  ∀ (P Q R S : ℝ × ℝ), l1 P Q → l2 R S → (P.2 - Q.2) * (R.2 - S.2) = (P.1 - Q.1) * (R.1 - S.1)

def isOnLine (p : (ℝ × ℝ) → (ℝ × ℝ) → Prop) (P : ℝ × ℝ) : Prop :=
  ∃ Q R : ℝ × ℝ, p Q R ∧ (P.2 - Q.2) * (R.1 - Q.1) = (R.2 - Q.2) * (P.1 - Q.1)

theorem rectangle_area : ∀ (A B C D E F : ℝ × ℝ) (s : ℝ)
  (hSquare : isSquare A B C D s)
  (hParallel : isParallel (λ P Q, Q = (P.1 + s, P.2 + s)) (λ P Q, Q = (P.1 - s, P.2 + s)))
  (hOnLineE : isOnLine (λ P D, D = (P.1 - s, P.2 + s)) E)
  (hOnLineF : isOnLine (λ P D, D = (P.1 - s, P.2 + s)) F),
  let area := s * s in area = 36 :=
sorry

end rectangle_area_l625_625959


namespace similar_triangle_GVN_GND_l625_625955

theorem similar_triangle_GVN_GND
  (circle : Type)
  [metric_space circle] 
  [inner_product_space ℝ circle] 
  (GH KL : circle) 
  (N K V G D : circle)
  (h1 : GH is_diameter)  -- GH is the diameter of the circle
  (h2 : GH perpendicular_bisector_at N KL) -- GH is the perpendicular bisector of KL at N
  (h3 : between K N V) -- V is between K and N
  (h4 : V ∈ segment G N)   -- V is a point on the segment GN
  (h5 : D ∈ line.extension G V ⧸⧺∂ circle) -- D is where GV extended meets the circle
  :
  similar_triangle (triangle G V N) (triangle G N D) :=
sorry

end similar_triangle_GVN_GND_l625_625955


namespace Jean_average_speed_correct_l625_625564

noncomputable def Jean_avg_speed_until_meet
    (total_distance : ℕ)
    (chantal_flat_distance : ℕ)
    (chantal_flat_speed : ℕ)
    (chantal_steep_distance : ℕ)
    (chantal_steep_ascend_speed : ℕ)
    (chantal_steep_descend_distance : ℕ)
    (chantal_steep_descend_speed : ℕ)
    (jean_meet_position_ratio : ℚ) : ℚ :=
  let chantal_flat_time := (chantal_flat_distance : ℚ) / chantal_flat_speed
  let chantal_steep_ascend_time := (chantal_steep_distance : ℚ) / chantal_steep_ascend_speed
  let chantal_steep_descend_time := (chantal_steep_descend_distance : ℚ) / chantal_steep_descend_speed
  let total_time_until_meet := chantal_flat_time + chantal_steep_ascend_time + chantal_steep_descend_time
  let jean_distance_until_meet := (jean_meet_position_ratio * chantal_steep_distance : ℚ) + chantal_flat_distance
  jean_distance_until_meet / total_time_until_meet

theorem Jean_average_speed_correct :
  Jean_avg_speed_until_meet 6 3 5 3 3 1 4 (1 / 3) = 80 / 37 :=
by
  sorry

end Jean_average_speed_correct_l625_625564


namespace number_of_trailing_zeros_mod_100_l625_625744

theorem number_of_trailing_zeros_mod_100 :
  let P := ∏ (i : ℕ) in finset.range 51, nat.factorial i in
  let M := nat.trailing_zeroes P in
  M % 100 = 12 :=
by
  sorry

end number_of_trailing_zeros_mod_100_l625_625744


namespace conic_section_is_ellipse_and_major_axis_is_8_l625_625939

noncomputable def isEllipse (x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 : ℝ) : Prop := sorry -- Definition of the conic section being an ellipse
noncomputable def majorAxisLength (x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 : ℝ) : ℝ := sorry -- Definition of the major axis length calculation

theorem conic_section_is_ellipse_and_major_axis_is_8 :
  (∀ (x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 : ℝ), x1 ≠ x2 → x1 ≠ x3 → x1 ≠ x4 → x1 ≠ x5 →
  y1 ≠ y2 → y1 ≠ y3 → y1 ≠ y4 → y1 ≠ y5 →
  y2 ≠ y3 → y2 ≠ y4 → y2 ≠ y5 →
  y3 ≠ y4 → y3 ≠ y5 →
  y4 ≠ y5 →
  isEllipse x1 y1 x2 y2 x3 y3 x4 y4 x5 y5) :=
by {
  intros,
  -- non-collinear conditions
  sorry
}

example : conic_section_is_ellipse_and_major_axis_is_8 (-2) 0 0 1 0 3 4 1 4 3 = 8 :=
by {
  sorry
}

end conic_section_is_ellipse_and_major_axis_is_8_l625_625939


namespace truthful_dwarfs_count_l625_625592

def dwarf (n : ℕ) := n < 10
def vanilla_ice_cream (n : ℕ) := dwarf n ∧ (∀ m, dwarf m)
def chocolate_ice_cream (n : ℕ) := dwarf n ∧ m % 2 = 0
def fruit_ice_cream (n : ℕ) := dwarf n ∧ m % 9 = 0

theorem truthful_dwarfs_count :
  ∃ T L : ℕ, T + L = 10 ∧ T + 2 * L = 16 ∧ T = 4 :=
by
  sorry

end truthful_dwarfs_count_l625_625592


namespace volume_calculation_correct_l625_625227

-- Defining constants
def height : ℝ := 15.3
def angle : ℝ := 37 + 42/60 -- Convert degree and minutes to a single number
def volume_full_capacity : ℝ := 1000
def tan_alpha : ℝ := 0.7820 -- Given tan(37°42')

-- Radius calculation based on full capacity volume and height
noncomputable def radius : ℝ := Real.sqrt (volume_full_capacity / (Real.pi * height))

-- Calculation of vertical drop due to tilt
noncomputable def x : ℝ := radius * tan_alpha

-- Adjusted height of water column
noncomputable def adjusted_height : ℝ := height - x

-- Volume of water in the tilted cylinder
noncomputable def volume_water : ℝ := (adjusted_height / height) * volume_full_capacity

theorem volume_calculation_correct :
  volume_water = 765.7 := by
  sorry

end volume_calculation_correct_l625_625227


namespace chess_team_arrangements_l625_625356

theorem chess_team_arrangements :
  let boys_positions := Nat.permutations 3
  let girls_positions := Nat.permutations 4
  boys_positions * girls_positions = 144 := by
  sorry

end chess_team_arrangements_l625_625356


namespace major_axis_length_l625_625512

def radius := 3
def minor_axis := 2 * radius
def major_axis := 1.40 * minor_axis

theorem major_axis_length : major_axis = 8.4 :=
by
  -- sorry to skip proof
  sorry

end major_axis_length_l625_625512


namespace find_tangent_line_through_point_l625_625614

theorem find_tangent_line_through_point :
  let P := (-3, -2)
  let C := λ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0
  ( ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b → (y = m * x + b) ∧ (P.1 = x ∧ P.2 = y) ∧ ((∀ x₀ y₀ : ℝ, C x₀ y₀ → (m * x₀ + b = y₀ ∨ m * x + b ≠ y₀))) ) ∨
  x = -3 ∨ (3 * x - 4 * y + 1 = 0) :=
sorry

end find_tangent_line_through_point_l625_625614


namespace limit_sum_reciprocal_seq_l625_625737

noncomputable def a : ℕ → ℝ
| 0       := 0 -- not used but needed for completeness
| 1       := 1
| 2       := 2
| (n + 1) := 1 + (List.prod (List.map a (List.range n))) + (List.prod (List.map a (List.range n)))^2

theorem limit_sum_reciprocal_seq : 
  tendsto (λ n, ∑ i in Finset.range n, 1 / a (i + 1)) at_top (𝓝 2) :=
begin
  sorry
end

end limit_sum_reciprocal_seq_l625_625737


namespace problem_statement_l625_625006

noncomputable theory

def f (x : ℝ) : ℝ := sorry
def f' (x : ℝ) : ℝ := sorry
def f'' (x : ℝ) : ℝ := sorry
def f''' (x : ℝ) : ℝ := sorry

axiom condition1 : ∀ x : ℝ, f(x^2) * f''(x) = f'(x) * f'(x^2)
axiom condition2 : f(1) = 1
axiom condition3 : f'''(1) = 8

-- Problem statement to prove f'(1) + f''(1) = 6
theorem problem_statement : f'(1) + f''(1) = 6 := sorry

end problem_statement_l625_625006


namespace range_of_a_l625_625208

theorem range_of_a (a : ℝ) :
  (∃ m n p : ℝ, m ≠ n ∧ n ≠ p ∧ p ≠ m ∧ f m = 2022 ∧ f n = 2022 ∧ f p = 2022 ∧ 
  (∀ x, f x = -x^3 + 3 * x + a)) → (2020 < a ∧ a < 2024) :=
sorry

end range_of_a_l625_625208


namespace no_solution_l625_625346

theorem no_solution (x : ℝ) : ¬ (6 + 3.5 * x = 2.5 * x - 30 + x) :=
by 
  intro h
  have : 0 = -36 := calc
    0 = 6 + 3.5 * x - (2.5 * x - 30 + x) : by rw h
     ... = 6 + 3.5 * x - 2.5 * x - 30 - x : by ring
     ... = 6 - 30 : by ring
     ... = -24 : by norm_num
  exact False.elim (by norm_num at this)

end no_solution_l625_625346


namespace arrange_in_ascending_order_l625_625631

-- Define the necessary conditions
def a : ℝ := Real.log 4 / Real.log 5
def b : ℝ := (Real.log 3 / Real.log 5)^2
def c : ℝ := Real.log 5 / Real.log 4

-- State the theorem to arrange a, b, c in ascending order
theorem arrange_in_ascending_order : b < a ∧ a < c := by
  sorry

end arrange_in_ascending_order_l625_625631


namespace inscribed_circle_diameter_in_triangle_l625_625415

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def area_heron (a b c : ℝ) : ℝ := 
  let s := semiperimeter a b c in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def inradius (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c in
  let K := area_heron a b c in
  K / s

noncomputable def inscribed_circle_diameter (a b c : ℝ) : ℝ :=
  2 * inradius a b c

theorem inscribed_circle_diameter_in_triangle (PQ PR QR : ℝ) (hPQ : PQ = 13) (hPR : PR = 14) (hQR : QR = 15) :
  inscribed_circle_diameter PQ PR QR = 8 := by
  rw [hPQ, hPR, hQR]
  sorry

end inscribed_circle_diameter_in_triangle_l625_625415


namespace circle_division_equal_areas_l625_625916

theorem circle_division_equal_areas (R : ℝ) (hR : R > 0) :
  ∃ (R1 R2 : ℝ), R1 = R * real.sqrt 3 / 3 ∧ R2 = R * real.sqrt 6 / 3 ∧
  ∀ (A1 A2 A3 : ℝ),
      A1 = π * R1^2 ∧
      A2 = π * R2^2 - π * R1^2 ∧
      A3 = π * R^2 - π * R2^2 →
      A1 = π * R^2 / 3 ∧
      A2 = π * R^2 / 3 ∧
      A3 = π * R^2 / 3 :=
by
  sorry

end circle_division_equal_areas_l625_625916


namespace no_positive_integer_exists_l625_625848

theorem no_positive_integer_exists
  (P1 P2 : ℤ → ℤ)
  (a : ℤ)
  (h_a_neg : a < 0)
  (h_common_root : P1 a = 0 ∧ P2 a = 0) :
  ¬ ∃ b : ℤ, b > 0 ∧ P1 b = 2007 ∧ P2 b = 2008 :=
sorry

end no_positive_integer_exists_l625_625848


namespace transformed_sin_eq_l625_625664

theorem transformed_sin_eq : 
  (∀ x, let f := (λ x, Real.sin x) in
  (f (2 * (x - Real.pi / 3)) = Real.sin (2 * x - 2 * Real.pi / 3))) :=
by
  intros x
  let f := λ x, Real.sin x
  sorry

end transformed_sin_eq_l625_625664


namespace smallest_perfect_square_divisible_by_2_3_5_l625_625454

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧
         (∀ k : ℕ, 
           (k > 0 ∧ (∃ l : ℕ, k = l * l) ∧ (2 ∣ k) ∧ (3 ∣ k) ∧ (5 ∣ k)) → n ≤ k) :=
sorry

end smallest_perfect_square_divisible_by_2_3_5_l625_625454


namespace inequality_of_factorials_and_polynomials_l625_625308

open Nat

theorem inequality_of_factorials_and_polynomials (m n : ℕ) (hm : m ≥ n) :
  2^n * n! ≤ (m+n)! / (m-n)! ∧ (m+n)! / (m-n)! ≤ (m^2 + m)^n :=
by
  sorry

end inequality_of_factorials_and_polynomials_l625_625308


namespace sum_of_coefficients_l625_625980

theorem sum_of_coefficients:
  let p := (3 * (3 * x^7 + 8 * x^4 - 7) + 7 * (x^5 - 7 * x^2 + 5)) in
  (p.eval 1) = 5 := by
  sorry

end sum_of_coefficients_l625_625980


namespace find_omega_theta_l625_625204

theorem find_omega_theta
  (ω θ : ℝ) (k : ℤ)
  (h_w_pos : ω > 0)
  (h_even : ∀ x : ℝ, 2 * sqrt 3 * sin (3 * ω * (x + θ) + π / 3) = 2 * sqrt 3 * sin (3 * ω * (x + θ) + π / 3))
  (h_period : ∀ x : ℝ, f(x) = f(x + 2 * π)) :
  ω = 1 / 3 ∧ ∃ k : ℤ, θ = k * π + π / 6 :=
sorry

end find_omega_theta_l625_625204


namespace domain_of_log3_neg_quadratic_l625_625811

theorem domain_of_log3_neg_quadratic (x : ℝ) : -x^2 - 2x > 0 ↔ -2 < x ∧ x < 0 :=
by 
  sorry

end domain_of_log3_neg_quadratic_l625_625811


namespace total_walkway_area_l625_625730

noncomputable def flower_bed_width : ℕ := 4
noncomputable def flower_bed_height : ℕ := 3
noncomputable def num_beds_per_row : ℕ := 3
noncomputable def num_rows : ℕ := 4
noncomputable def walkway_width : ℕ := 2

theorem total_walkway_area :
  let garden_width := (num_beds_per_row * flower_bed_width) + ((num_beds_per_row + 1) * walkway_width),
      garden_height := (num_rows * flower_bed_height) + ((num_rows + 1) * walkway_width),
      total_garden_area := garden_width * garden_height,
      bed_area := flower_bed_width * flower_bed_height,
      total_beds_area := num_beds_per_row * num_rows * bed_area
  in total_garden_area - total_beds_area = 296 :=
by
  sorry

end total_walkway_area_l625_625730


namespace perp_lines_possibilities_l625_625403

-- Define two lines and a third line that they are perpendicular to
variables {Line : Type} [has_perpendicular Line]

-- Assume l1 and l2 are perpendicular to the same line m
variables (l1 l2 m : Line)
variables (h1 : perpendicular l1 m) (h2 : perpendicular l2 m)

-- We need to show that lines l1 and l2 can be either parallel, intersect or skew
theorem perp_lines_possibilities :   
  (l1 = l2) ∨ (intersects l1 l2) ∨ (skew l1 l2) :=
sorry

end perp_lines_possibilities_l625_625403


namespace diamond_associative_l625_625313

def diamond (a b : ℕ) : ℕ := a ^ (b / a)

theorem diamond_associative (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  diamond a (diamond b c) = diamond (diamond a b) c :=
sorry

end diamond_associative_l625_625313


namespace AT_bisects_angle_A_l625_625063

-- Define the isosceles triangle ABC with AB = AC
variables {A B C : Point}
-- Define the function to check if the triangle is isosceles
def is_isosceles (A B C : Point) := distance A B = distance A C

-- Define the point X on segment BC
variables {X : Point}
axiom X_on_BC : is_on_segment B C X

-- Define points Z and Y and the angle condition
variables {Z Y : Point}
axiom Z_in_AC : is_on_segment A C Z
axiom Y_in_AB : is_on_segment A B Y
axiom angle_condition : ∠ B X Y = ∠ Z X C

-- Define the point T as the intersection of line through B parallel to YZ and segment XZ
variables {T : Point}
axiom line_parallel_YZ_through_B : parallel (line_through B T) (line_through Y Z)
axiom T_on_XZ : is_on_segment X Z T

-- State the theorem to be proved
theorem AT_bisects_angle_A : bisects (line_through A T) (∠ BAC) := by
  sorry

end AT_bisects_angle_A_l625_625063


namespace simplify_and_evaluate_l625_625793

-- Define the expression
def expr (a : ℚ) : ℚ := (3 * a - 1) ^ 2 + 3 * a * (3 * a + 2)

-- Given the condition
def a_value : ℚ := -1 / 3

-- State the theorem
theorem simplify_and_evaluate : expr a_value = 3 :=
by
  -- Proof will be added here
  sorry

end simplify_and_evaluate_l625_625793


namespace number_of_correct_statements_l625_625373

theorem number_of_correct_statements :
  let statements := ["There are three points A, B, C on a line. If AB = 2BC, then point C is the midpoint of segment AB.",
                     "The length of a segment between two points is called the distance between the two points.",
                     "Among all lines connecting two points, the segment is the shortest.",
                     "Ray AB and ray BA represent the same ray."]
  in (nat.count (λ s, s ∈ statements ∧ (s = statements[1] ∨ s = statements[2])) \= 2) sorry

end number_of_correct_statements_l625_625373


namespace sum_of_100_digit_mountainous_is_composite_l625_625768

-- Definition to check if a number is "mountainous"
def is_mountainous (n : ℕ) : Prop :=
  let digits := (n.digits 10).toList in
  1 < digits.length ∧
  ∃ (peak_idx : ℕ), 
    1 ≤ peak_idx ∧ peak_idx < digits.length - 1 ∧
    digits.nth peak_idx > some (digits.take peak_idx.unzip.2.max') ∧
    digits.nth peak_idx > some (digits.drop (peak_idx + 1).unzip.2.max')

-- All possible 100-digit natural numbers
def all_100_digit_numbers : List ℕ := 
  List.filter (λ n, n.digits 10).length = 100 (List.range (10^100))

-- Sum of all 100-digit mountainous numbers
def sum_of_100_digit_mountainous_numbers : ℕ :=
  all_100_digit_numbers.filter is_mountainous).sum

-- The theorem to be proved
theorem sum_of_100_digit_mountainous_is_composite :
  composite (sum_of_100_digit_mountainous_numbers) :=
sorry

end sum_of_100_digit_mountainous_is_composite_l625_625768


namespace gcd_gx_x_l625_625191

theorem gcd_gx_x (x : ℕ) (h : ∃ k, x = 360 * k) : 
  Nat.gcd ((5 * x + 3) * (11 * x + 2) * (7 * x + 4)^2 * (8 * x + 5), x) = 120 := 
by 
  sorry

end gcd_gx_x_l625_625191


namespace process_never_stops_l625_625776

theorem process_never_stops (N : ℕ := 10^900 - 1) : 
  ∀ k, let step := λ x : ℕ, let A := x / 100, B := x % 100 in 2 * A + 8 * B in
  (k > 1) → ∃ n, step^[n] N ≥ 100 :=
begin
  sorry
end

end process_never_stops_l625_625776


namespace find_number_l625_625497

theorem find_number :
  ∃ (x : ℝ), (0.65 * x - 25 = 90) ∧ (x ≈ 176.92) := sorry

end find_number_l625_625497


namespace students_participated_in_only_one_activity_l625_625072

variable (A B : Finset ℕ)
variable (n : ℕ) (a : ℕ) (b : ℕ)
variable h1 : n = 50
variable h2 : a = 30
variable h3 : b = 25
variable h4 : A.card = a
variable h5 : B.card = b
variable h6 : (A ∪ B).card = n

theorem students_participated_in_only_one_activity : (A ∪ B).card - (A ∩ B).card = 45 :=
by
  have h7 : a + b = (A ∩ B).card + n := by sorry
  have h8 : (A ∩ B).card = 5 := by sorry
  sorry

end students_participated_in_only_one_activity_l625_625072


namespace coplanarity_of_M_A_B_C_l625_625673

-- Definitions for vectors and coplanarity conditions
variables (A B C M O : Type)
variables [has_vector_space ℝ A] [has_vector_space ℝ B] [has_vector_space ℝ C] [has_vector_space ℝ O]

-- Point Coordinates
variables (OA OB OC OM : ℝ)

-- Coplanarity Condition
def coplanarity_condition (x y z : ℝ) : Prop := x + y + z = 1

-- Given Vector Equation Options
def option_A : Prop := coplanarity_condition 1 1 (-1)
def option_B : Prop := coplanarity_condition (1/3) (1/3) (1/3)
def option_C : Prop := coplanarity_condition 1 (1/2) (1/4)
def option_D : Prop := coplanarity_condition 3 (-1) (-1)

-- Theorem
theorem coplanarity_of_M_A_B_C (A B C M O : Type)( OA OB OC OM : ℝ)
  [has_vector_space ℝ A] [has_vector_space ℝ B] [has_vector_space ℝ C] [has_vector_space ℝ O] :
  option_A A B C M O OA OB OC OM ∧ option_B A B C M O OA OB OC OM ∧ option_D A B C M O OA OB OC OM :=
by {
  split,
  sorry,
  split,
  sorry,
  sorry,
}

end coplanarity_of_M_A_B_C_l625_625673


namespace area_of_given_cyclic_quad_l625_625854

noncomputable def area_cyclic_quad (a b c d : ℝ) :=
  let s := (a + b + c + d) / 2
  in real.sqrt ((s - a) * (s - b) * (s - c) * (s - d))

theorem area_of_given_cyclic_quad :
  let a := 4
  let b := 5
  let c := 7
  let d := 10
  area_cyclic_quad a b c d = 36 := 
by
  have semi_perimeter := (a + b + c + d) / 2
  rw [show a = 4, from rfl, show b = 5, from rfl, show c = 7, from rfl, show d = 10, from rfl,
      show semi_perimeter = 13, by norm_num],
  calc area_cyclic_quad 4 5 7 10
      = real.sqrt ((13 - 4) * (13 - 5) * (13 - 7) * (13 - 10)) : by rw [area_cyclic_quad, rfl]
  ... = real.sqrt (9 * 8 * 6 * 3) : by norm_num
  ... = real.sqrt (1296) : by norm_num
  ... = 36 : by norm_num

end area_of_given_cyclic_quad_l625_625854


namespace LCM_20_45_75_is_900_l625_625876

def prime_factorization_20 := (2^2, 5)
def prime_factorization_45 := (3^2, 5)
def prime_factorization_75 := (3, 5^2)

theorem LCM_20_45_75_is_900 
  (pf_20 : prime_factorization_20 = (2^2, 5))
  (pf_45 : prime_factorization_45 = (3^2, 5))
  (pf_75 : prime_factorization_75 = (3, 5^2)) : 
  Nat.lcm (Nat.lcm 20 45) 75 = 900 := 
  by sorry

end LCM_20_45_75_is_900_l625_625876


namespace true_propositions_l625_625299

-- Definitions for the problem conditions
variable (Line : Type) (Plane : Type)
variable (m n : Line)
variable (α β : Plane)
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Proposition statements as functions
def prop1 : Prop := (perpendicular α β ∧ parallel m α) → perpendicular m β
def prop2 : Prop := (perpendicular m α ∧ perpendicular n α) → parallel m n
def prop3 : Prop := (perpendicular m α ∧ perpendicular m n) → parallel n α
def prop4 : Prop := (perpendicular n α ∧ perpendicular n β) → parallel_planes β α

-- The theorem stating which propositions are true
theorem true_propositions : (¬ prop1) ∧ prop2 ∧ (¬ prop3) ∧ prop4 := by
  sorry

end true_propositions_l625_625299


namespace circle_polar_eq_l625_625267

-- Define the necessary geometric and algebraic structures
variables {a : ℝ} (h : a ≠ 0)

-- Define the properties of the circle in polar coordinates
def center_polar := (a / 2, π / 2)
def radius_polar := a / 2

-- Define the corresponding polar equation in Cartesian coordinates
def polar_equation (ρ θ : ℝ) : Prop := ρ = a * sin θ

-- The main statement to prove
theorem circle_polar_eq (ρ θ : ℝ) (h : ρ = a / 2 ∧ center_polar = (a / 2, π / 2)) :
  polar_equation ρ θ :=
by { sorry }

end circle_polar_eq_l625_625267


namespace count_valid_house_numbers_l625_625145

noncomputable def valid_house_numbers : ℕ :=
  let primes := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let even_digit_sum_primes := primes.filter (λ n, (n / 10) + (n % 10) % 2 = 0)
  let house_numbers := do
    wx ← primes
    yz ← even_digit_sum_primes
    guard (wx ≠ yz)
    pure (wx * 100 + yz)
  house_numbers.length

theorem count_valid_house_numbers : valid_house_numbers = 30 := by
  sorry

end count_valid_house_numbers_l625_625145


namespace inscribe_octahedron_in_cube_l625_625280

noncomputable def cube_edge_length := 4

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

structure Cube :=
  (vertices : List Point3D)
  (edges : List (Point3D × Point3D))
  (edge_length : ℝ)

structure Octahedron :=
  (vertices : List Point3D)
  (edges : List (Point3D × Point3D))

def inscribed (cube : Cube) (octahedron : Octahedron) (ratio : ℝ) : Prop :=
  ∀ (v : Point3D) (e : Point3D × Point3D), 
    e ∈ cube.edges →
    v ∈ octahedron.vertices →
    ∃ p₁ p₂, p₁ ≠ p₂ ∧ (v = p₁ ∨ v = p₂) ∧ 
               (distance p₁ p₂ = cube.edge_length * ratio)

theorem inscribe_octahedron_in_cube :
  ∃ (o : Octahedron),
    ∀ (c : Cube), 
      c.edge_length = 4 → 
      inscribed c o (1 / 4) :=
sorry

end inscribe_octahedron_in_cube_l625_625280


namespace initial_oranges_l625_625031

theorem initial_oranges (O : ℕ) (h1 : O + 6 - 3 = 6) : O = 3 :=
by
  sorry

end initial_oranges_l625_625031


namespace exists_positive_k_l625_625130

open Matrix Real

-- Define the vectors generating the parallelepiped
def v1 : Fin 3 → ℝ := ![3, 4, 5]
def v2 (k : ℝ) : Fin 3 → ℝ := ![1, k, 3]
def v3 (k : ℝ) : Fin 3 → ℝ := ![2, 3, k]

-- Define the matrix from the vectors
def matrix22 (k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  λ i j, 
    if i = 0 ∧ j = 0 then 3 else
    if i = 0 ∧ j = 1 then 1 else
    if i = 0 ∧ j = 2 then 2 else
    if i = 1 ∧ j = 0 then 4 else
    if i = 1 ∧ j = 1 then k else
    if i = 1 ∧ j = 2 then 3 else
    if i = 2 ∧ j = 0 then 5 else
    if i = 2 ∧ j = 1 then 3 else
    -- i = 2 ∧ j = 2
    k

-- Statement of the proof problem
theorem exists_positive_k (h : ∃ k : ℝ, k > 0 ∧ det (matrix22 k) = 18) :
  ∃! k : ℝ, k = (7 + sqrt 67) / 3 :=
by sorry

end exists_positive_k_l625_625130


namespace seating_arrangements_l625_625837

theorem seating_arrangements :
  let front_row := 11
  let back_row := 12
  let middle_excluded_seats := 3
  let front_avail_seats := front_row - middle_excluded_seats
  let back_avail_seats := back_row
  let not_next_to_each_other :=
    -- Function to calculate arrangements such that no two persons sit next to each other.
    -- This placeholder expresses the configuration condition.
    λ (total_seats : ℕ), 
      if total_seats < 2 then 0 else total_seats * (total_seats - 1)
  in
  let front_row_arrangements := not_next_to_each_other front_avail_seats
  let back_row_arrangements := not_next_to_each_other back_avail_seats
  let separate_row_arrangements := front_avail_seats * back_avail_seats
  front_row_arrangements + back_row_arrangements + separate_row_arrangements = 346 := 
sorry

end seating_arrangements_l625_625837


namespace smallest_perfect_square_divisible_by_2_3_5_l625_625452

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧
         (∀ k : ℕ, 
           (k > 0 ∧ (∃ l : ℕ, k = l * l) ∧ (2 ∣ k) ∧ (3 ∣ k) ∧ (5 ∣ k)) → n ≤ k) :=
sorry

end smallest_perfect_square_divisible_by_2_3_5_l625_625452


namespace terminal_side_on_y_axis_l625_625646

theorem terminal_side_on_y_axis (α : ℝ) (h : abs (sin α) = 1) : α = π / 2 ∨ α = 3 * π / 2 :=
sorry

end terminal_side_on_y_axis_l625_625646


namespace smallest_m_l625_625982

def D (n : ℕ) : Set ℕ := { d | d > 0 ∧ n % d = 0 }

def F (i : ℕ) (n : ℕ) : Set ℕ := { a ∈ D n | a % 4 = i }

def f (i : ℕ) (n : ℕ) : ℕ := (F i n).toFinset.card

theorem smallest_m : ∃ m : ℕ, (2 * f 1 m) - (f 2 m) = 2017 ∧ 
  (∀ n : ℕ, (2 * f 1 n) - (f 2 n) = 2017 → n ≥ m) ∧ m = 2 * 5 ^ 2016 :=
sorry

end smallest_m_l625_625982


namespace complex_square_area_l625_625724

noncomputable def square_area (z : ℂ) : ℝ :=
  let side_length : ℝ := complex.abs (z^4 - z)
  in side_length^2

theorem complex_square_area (z : ℂ) (h1 : complex.abs z ≠ 0) (h2 : complex.abs z = 1)
  (h3 : z^4 ≠ z) (h4 : z^6 ≠ z) (h5 : z ≠ z^4) (h6 : z ≠ z^6) :
  square_area z = 3 := by
  sorry

end complex_square_area_l625_625724


namespace board_transform_diagonal_l625_625012

theorem board_transform_diagonal :
  let original_diagonal := (λ k : ℕ, 32 * k + 1)
  ∀ n ∈ (Finset.range 32.succ), ( λ k, board_transform 32 !! (k + 32 * k) ) = original_diagonal n := by
  sorry

end board_transform_diagonal_l625_625012


namespace proof_problem_l625_625300

/- Definitions -/
variables {Line Plane : Type}
variables (m n : Line) (alpha beta : Plane)

/- Conditions -/
def different_lines : Prop := m ≠ n
def different_planes : Prop := alpha ≠ beta

def parallel_line_plane (l : Line) (p : Plane) : Prop := -- insert the definition of line parallel to plane
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := -- insert the definition of line perpendicular to plane
def parallel_lines (l1 l2 : Line) : Prop := -- insert the definition of line parallel to line
def parallel_planes (p1 p2 : Plane) : Prop := -- insert the definition of plane parallel to plane

/- Statements -/
def statement_A : Prop := parallel_line_plane m alpha ∧ parallel_line_plane n alpha → parallel_lines m n
def statement_B : Prop := parallel_line_plane m alpha ∧ parallel_line_plane m beta → parallel_planes alpha beta
def statement_C : Prop := parallel_lines m n ∧ perpendicular_line_plane m alpha → perpendicular_line_plane n alpha
def statement_D : Prop := parallel_line_plane m alpha ∧ perpendicular_line_plane alpha beta → perpendicular_line_plane m beta

/- Proof problem -/
theorem proof_problem (h₁ : different_lines) (h₂ : different_planes) :
  ¬ statement_A ∧ ¬ statement_B ∧ statement_C ∧ ¬ statement_D := by {
    sorry
  }

end proof_problem_l625_625300


namespace inequality_proof_l625_625328

variable {a b c : ℝ}
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem inequality_proof : 
  sqrt (a^2 - a * b + b^2) + sqrt (b^2 - b * c + c^2) ≥ sqrt (a^2 + a * c + c^2) :=
by sorry

end inequality_proof_l625_625328


namespace value_of_n_l625_625702

theorem value_of_n (n : ℤ) :
  (∀ x : ℤ, (x + n) * (x + 2) = x^2 + 2 * x + n * x + 2 * n → 2 + n = 0) → n = -2 := 
by
  intro h
  have h1 := h 0
  sorry

end value_of_n_l625_625702


namespace probability_none_needs_attention_probability_at_least_one_needs_attention_l625_625942

open Probability

-- Define the probabilities of needing attention
def P_A : ℝ := 0.9
def P_B : ℝ := 0.8
def P_C : ℝ := 0.85

-- Given these probabilities, we aim to prove the following:
theorem probability_none_needs_attention :
  let P_not_A := 1 - P_A in
  let P_not_B := 1 - P_B in
  let P_not_C := 1 - P_C in
  P_not_A * P_not_B * P_not_C = 0.003 :=
  by
    sorry

theorem probability_at_least_one_needs_attention :
  let P_not_A := 1 - P_A in
  let P_not_B := 1 - P_B in
  let P_not_C := 1 - P_C in
  1 - (P_not_A * P_not_B * P_not_C) = 0.997 :=
  by
    sorry

end probability_none_needs_attention_probability_at_least_one_needs_attention_l625_625942


namespace edward_total_earnings_l625_625064

theorem edward_total_earnings :
  ∀ (earnings_per_lawn : ℕ) (total_lawns : ℕ) (lawns_forgotten : ℕ),
    earnings_per_lawn = 4 →
    total_lawns = 17 →
    lawns_forgotten = 9 →
    (total_lawns - lawns_forgotten) * earnings_per_lawn = 32 := by
  intros earnings_per_lawn total_lawns lawns_forgotten hp ht hf
  rw [hp, ht, hf]
  exact Nat.mul_comm 8 4 ▸ rfl


end edward_total_earnings_l625_625064


namespace least_common_multiple_of_20_45_75_l625_625867

theorem least_common_multiple_of_20_45_75 :
  Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
sorry

end least_common_multiple_of_20_45_75_l625_625867


namespace find_MO_l625_625719

theorem find_MO
  (K L M N O : Point)
  (MN KM LN : Line)
  (h1 : is_perpendicular MN KM)
  (h2 : is_perpendicular KL LN)
  (h3 : length MN = 65)
  (h4 : length KL = 28)
  (h5 : intersects_at_perpendicular KN L MN = O)
  (h6 : length (segment K O) = 8) :
  length (segment M O) = 90 := by
  sorry

end find_MO_l625_625719


namespace John_next_birthday_l625_625734

noncomputable def John's_next_birthday_age : ℝ :=
  let j, c, b : ℝ := sorry, sorry, sorry
  let h1 : c = 1.3 * b := sorry
  let h2 : j = 0.75 * c := sorry
  let h3 : j + c + b = 30 := sorry
  j + 1

theorem John_next_birthday :
  (∀ (j c b : ℝ), c = 1.3 * b -> j = 0.75 * c -> j + c + b = 30 -> j + 1 = 9) :=
by {
  intros j c b h1 h2 h3,
  sorry
}

end John_next_birthday_l625_625734


namespace eighth_root_of_large_number_l625_625140

theorem eighth_root_of_large_number : (∃ n : ℕ, n = 5487587353601 ∧ ∃ k : ℕ, k = 8 ∧ (kfst : k = 8 ∧ (100+1)^k = n) ∧ ∃m: ℕ, m^k = n ∧ m = 101) :=
begin
  use 5487587353601,
  split,
  -- Show that our number is correct
  refl,
  use 101,
  split,
  -- Show that k = 8
  refl,
  split,
  -- Establish that (100+1)^8 = 5487587353601 
  exact calc (100 + 1)^8 = 5487587353601 : sorry,
  -- Finally, prove the actual equality
  refl,
end

end eighth_root_of_large_number_l625_625140


namespace desired_depth_l625_625496

-- Define the given conditions
def men_hours_30m (d : ℕ) : ℕ := 18 * 8 * d
def men_hours_Dm (d1 : ℕ) (D : ℕ) : ℕ := 40 * 6 * d1

-- Define the proportion
def proportion (d d1 : ℕ) (D : ℕ) : Prop :=
  (men_hours_30m d) / 30 = (men_hours_Dm d1 D) / D

-- The main theorem to prove the desired depth
theorem desired_depth (d d1 : ℕ) (H : proportion d d1 50) : 50 = 50 :=
by sorry

end desired_depth_l625_625496


namespace problem_l625_625825

def sequence : nat → ℤ
| 0       := 1
| 1       := 2
| (n + 2) := sequence n + (sequence (n + 1))^2

theorem problem : (sequence 2006) % 7 = 6 := by
  sorry

end problem_l625_625825


namespace inverse_function_value_l625_625700

noncomputable def f : ℝ → ℝ := λ x, (Real.sqrt 2) x

theorem inverse_function_value :
  (∃ f : ℝ → ℝ, (∀ x : ℝ, f (2 ^ x) = x) ∧ f (2) = 1) :=
begin
  use (λ x, log x / log 2),
  split,
  {
    intros x,
    rw [Real.log_pow, Real.log_div, Real.log_two, div_mul_eq_mul_div _ _ _ ],
    sorry, -- Proof steps showing f(x) is indeed the inverse function of 2^x would go here
  },
  {
    exact div_self (ne_of_gt (log_pos (zero_lt_two))),
  },
end

end inverse_function_value_l625_625700


namespace tank_capacity_l625_625480

theorem tank_capacity
  (C : ℕ)
  (leak_rate : C / 6)
  (inlet_rate : 3.5 * 60)
  (net_emptying_rate : C / 8) :
  210 - (C / 6) = (C / 8) → C = 720 := sorry

end tank_capacity_l625_625480


namespace quarters_needed_l625_625522

-- Define the cost of items in cents and declare the number of items to purchase.
def quarter_value : ℕ := 25
def candy_bar_cost : ℕ := 25
def chocolate_cost : ℕ := 75
def juice_cost : ℕ := 50

def num_candy_bars : ℕ := 3
def num_chocolates : ℕ := 2
def num_juice_packs : ℕ := 1

-- Theorem stating the number of quarters needed to buy the given items.
theorem quarters_needed : 
  (num_candy_bars * candy_bar_cost + num_chocolates * chocolate_cost + num_juice_packs * juice_cost) / quarter_value = 11 := 
sorry

end quarters_needed_l625_625522


namespace uc_berkeley_grid_impossible_l625_625804

theorem uc_berkeley_grid_impossible :
  ¬ ∃ f : (Fin 2001 × Fin 2001) → Finset (Fin 2001 × Fin 2001),
      (∀ r : Fin 2001 × Fin 2001,
        f r = {p | p.1 = r.1 ∧ (p.2 = r.2 + 1 ∨ p.2 = r.2 - 1) ∨
                      p.2 = r.2 ∧ (p.1 = r.1 + 1 ∨ p.1 = r.1 - 1)}.toFinset ∧
        f r ≠ ∅ ∧
        Finset.card (f r) = 2) :=
by
  sorry

end uc_berkeley_grid_impossible_l625_625804


namespace coeff_x10_in_binomial_expansion_l625_625042

theorem coeff_x10_in_binomial_expansion : 
  (polynomial.mk (λ n, if n = 10 then (algebra.algebra_map ℚ) (-11) else (algebra.algebra_map ℚ) 0)) = 
  (polynomial.expand (algebra.algebra_map ℚ) (x - 1) ^ 11) :=
by
  sorry

end coeff_x10_in_binomial_expansion_l625_625042


namespace grandfather_age_l625_625505

variable (F S G : ℕ)

theorem grandfather_age (h1 : F = 58) (h2 : F - S = S) (h3 : S - 5 = (1 / 2) * G) : G = 48 := by
  sorry

end grandfather_age_l625_625505


namespace max_lattice_points_in_unit_circle_l625_625046

-- Define a point with integer coordinates
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

-- Define the condition for a lattice point to be strictly inside a given circle
def strictly_inside_circle (p : LatticePoint) (center : Prod ℤ ℤ) (r : ℝ) : Prop :=
  let dx := (p.x - center.fst : ℝ)
  let dy := (p.y - center.snd : ℝ)
  dx^2 + dy^2 < r^2

-- Define the problem statement
theorem max_lattice_points_in_unit_circle : ∀ (center : Prod ℤ ℤ) (r : ℝ),
  r = 1 → 
  ∃ (ps : Finset LatticePoint), 
    (∀ p ∈ ps, strictly_inside_circle p center r) ∧ 
    ps.card = 4 :=
by
  sorry

end max_lattice_points_in_unit_circle_l625_625046


namespace set_equality_example_l625_625969

theorem set_equality_example : {x : ℕ | 2 * x + 3 ≥ 3 * x} = {0, 1, 2, 3} := by
  sorry

end set_equality_example_l625_625969


namespace domain_of_f_l625_625044

def f (x : ℝ) : ℝ := real.sqrt (x + 1) + real.root 4 (x - 3)

theorem domain_of_f : ∀ x, (x ∈ Icc 3 real.top) ↔ (0 ≤ x + 1 ∧ 0 ≤ x - 3) :=
by sorry

end domain_of_f_l625_625044


namespace largest_divisor_of_consecutive_even_product_l625_625858

theorem largest_divisor_of_consecutive_even_product (n : ℕ) (h : n % 2 = 1) :
  ∃ d, (∀ n, n % 2 = 1 → d ∣ (n+2) * (n+4) * (n+6) * (n+8) * (n+10)) ∧ d = 8 :=
begin
  existsi 8,
  split,
  { intros n hn,
    repeat { sorry },
  },
  { refl }
end

end largest_divisor_of_consecutive_even_product_l625_625858


namespace dirichlet_convolution_multiplicative_l625_625307

open_locale big_operators

variable {α : Type*}
variables (f g : ℕ → α)

def multiplicative_function (f : ℕ → α) : Prop :=
∀ (m n : ℕ), gcd(m, n) = 1 → f(m * n) = f(m) * f(n)

def dirichlet_convolution (f g : ℕ → α) (n : ℕ) : α :=
∑ d in divisors n, f d * g (n / d)

theorem dirichlet_convolution_multiplicative
  (hf : multiplicative_function f)
  (hg : multiplicative_function g) : multiplicative_function (dirichlet_convolution f g) :=
sorry

end dirichlet_convolution_multiplicative_l625_625307


namespace weights_balance_impossible_l625_625023

open Nat

theorem weights_balance_impossible 
    (weights : Finset ℕ)
    (h_weights : weights = (Finset.range 40).map (λ n, n + 1))
    (left_scale : Finset ℕ)
    (right_scale : Finset ℕ)
    (h_left_scale : left_scale.card = 10)
    (h_right_scale : right_scale.card = 10)
    (h_left_even : ∀ x ∈ left_scale, Even x)
    (h_right_odd : ∀ x ∈ right_scale, Odd x)
    (h_scales_union : left_scale ∪ right_scale = weights)
    (h_balance : left_scale.sum id = right_scale.sum id) :
  False :=
sorry

end weights_balance_impossible_l625_625023


namespace number_of_special_pairs_even_l625_625174

-- Define polygonal_chain and vertices_non_collinear conditions
variable (polygonal_chain : List (ℝ × ℝ)) -- a List of vertices (x, y) representing a polygonal chain
variable (vertices_non_collinear : ∀ (v1 v2 v3 : ℝ × ℝ), (v1 ∈ polygonal_chain) → (v2 ∈ polygonal_chain) → (v3 ∈ polygonal_chain) → (v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3) → ¬collinear v1 v2 v3)
variable (non_self_intersecting : ∀ (s1 s2 : ℝ ≫ ℝ), (s1 ∈ segments_of polygonal_chain) → (s2 ∈ segments_of polygonal_chain) → (s1 ≠ s2) → ¬self_intersecting s1 s2)

-- Define what it means for pairs of segments to be special
def special_pair (s1 s2 : ℝ ≫ ℝ) : Prop :=
  (s1 ∈ segments_of polygonal_chain) → (s2 ∈ segments_of polygonal_chain) → nonadjacent s1 s2 → intersect_extension s1 s2

-- Statement to prove
theorem number_of_special_pairs_even
  (polygonal_chain : List (ℝ × ℝ)) 
  (vertices_non_collinear : ∀ (v1 v2 v3 : ℝ × ℝ), (v1 ∈ polygonal_chain) → (v2 ∈ polygonal_chain) → (v3 ∈ polygonal_chain) → (v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3) → ¬collinear v1 v2 v3)
  (non_self_intersecting : ∀ (s1 s2 : ℝ ≫ ℝ), (s1 ∈ segments_of polygonal_chain) → (s2 ∈ segments_of polygonal_chain) → (s1 ≠ s2) → ¬self_intersecting s1 s2) :
  even (card (set_of (λ ⟨s1, s2⟩, special_pair polygonal_chain s1 s2))) :=
sorry

end number_of_special_pairs_even_l625_625174


namespace common_ratio_of_geometric_sequence_is_2_or_neg2_l625_625266

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def product_sequence_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∏ i in finset.range n, a i

theorem common_ratio_of_geometric_sequence_is_2_or_neg2 :
  ∃ q : ℝ, ∀ (a : ℕ → ℝ),
  a 1 = 1 ∧ is_geometric_sequence a q ∧ product_sequence_first_n_terms a 5 = 1024 →
  (q = 2 ∨ q = -2) :=
begin
  sorry
end

end common_ratio_of_geometric_sequence_is_2_or_neg2_l625_625266


namespace smallest_perfect_square_divisible_by_2_3_5_l625_625456

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧
         (∀ k : ℕ, 
           (k > 0 ∧ (∃ l : ℕ, k = l * l) ∧ (2 ∣ k) ∧ (3 ∣ k) ∧ (5 ∣ k)) → n ≤ k) :=
sorry

end smallest_perfect_square_divisible_by_2_3_5_l625_625456


namespace eccentricity_of_hyperbola_l625_625359

theorem eccentricity_of_hyperbola :
  let a := Real.sqrt 5
  let b := 2
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  (∃ (x y : ℝ), (x^2 / 5) - (y^2 / 4) = 1 ∧ e = (3 * Real.sqrt 5) / 5) := sorry

end eccentricity_of_hyperbola_l625_625359


namespace bobby_initial_candy_count_l625_625952

-- Definitions for the conditions
def bobbyInitialCandy (initial : Nat) (extra : Nat) (total : Nat) : Prop :=
  initial + extra = total

-- The theorem that needs to be proved
theorem bobby_initial_candy_count (initial : Nat) (extra : Nat) (total : Nat)
  (h : bobbyInitialCandy initial extra total) : initial = 26 :=
by
  -- Using the given conditions and the proof for the correct number of initial candies
  have h1 : 26 + extra = total := sorry
  exact sorry

-- Instantiating specific values from the problem statement
example : bobby_initial_candy_count 26 17 43 := by
  have h : bobbyInitialCandy 26 17 43 := by
    unfold bobbyInitialCandy
    exact rfl
  exact bobby_initial_candy_count 26 17 43 h

end bobby_initial_candy_count_l625_625952


namespace independence_condition_l625_625051

variables (A B : Prop) (P : Prop → ℝ)

-- Define conditional probability
def P_cond (X Y : Prop) := P (X ∧ Y) / P Y

-- Define independence
def independent (A B : Prop) := P (A ∧ B) = P A * P B

theorem independence_condition :
  (P_cond A B = P_cond A (not B) ∨ P A = P_cond A B) ↔ independent A B :=
sorry

end independence_condition_l625_625051


namespace ratio_of_areas_l625_625789

theorem ratio_of_areas (p1 p2 p3 p4 : ℕ) (h1 : p1 = 16) (h2 : p2 = 32) (h3 : p3 = 48) (h4 : p4 = 2 * p1) :
  (let area1 := (p1 / 4) ^ 2;
       area4 := (p4 / 4) ^ 2 in
  area1 / area4 = 1 / 4) :=
by
  sorry

end ratio_of_areas_l625_625789


namespace john_swimming_improvement_l625_625286

theorem john_swimming_improvement :
  let initial_lap_time := 35 / 15 -- initial lap time in minutes per lap
  let current_lap_time := 33 / 18 -- current lap time in minutes per lap
  initial_lap_time - current_lap_time = 1 / 9 := 
by
  -- Definition of initial and current lap times are implied in Lean.
  sorry

end john_swimming_improvement_l625_625286


namespace difference_in_circumferences_l625_625846

theorem difference_in_circumferences (r_inner r_outer : ℝ) (h1 : r_inner = 15) (h2 : r_outer = r_inner + 8) : 
  2 * Real.pi * r_outer - 2 * Real.pi * r_inner = 16 * Real.pi :=
by
  rw [h1, h2]
  sorry

end difference_in_circumferences_l625_625846


namespace archer_expected_hits_l625_625115

noncomputable def binomial_expected_value (n : ℕ) (p : ℝ) : ℝ :=
  n * p

theorem archer_expected_hits :
  binomial_expected_value 10 0.9 = 9 :=
by
  sorry

end archer_expected_hits_l625_625115


namespace system_of_equations_is_B_l625_625723

-- Define the given conditions and correct answer
def condition1 (x y : ℝ) : Prop := 5 * x + y = 3
def condition2 (x y : ℝ) : Prop := x + 5 * y = 2
def correctAnswer (x y : ℝ) : Prop := 5 * x + y = 3 ∧ x + 5 * y = 2

theorem system_of_equations_is_B (x y : ℝ) : condition1 x y ∧ condition2 x y ↔ correctAnswer x y := by
  -- Proof goes here
  sorry

end system_of_equations_is_B_l625_625723


namespace sum_of_cos4_eq_273_over_8_l625_625132

open Real

theorem sum_of_cos4_eq_273_over_8 :
    (∑ k in Finset.range 91, (cos (k * (π / 180)))^4) = 273 / 8 :=
  sorry

end sum_of_cos4_eq_273_over_8_l625_625132


namespace a_range_l625_625653

noncomputable theory
open Classical

def f (a : ℝ) (x : ℕ) : ℝ :=
  if x ≤ 12 then (1 - 2 * a) * x + 5 else a ^ (x - 13)

def a_sequence (a : ℝ) (n : ℕ) : ℝ :=
  f a n

def decreasing_sequence (a_sequence : ℕ → ℝ) : Prop :=
  ∀ m n : ℕ, m ≠ n → (m - n) * (a_sequence m - a_sequence n) < 0

theorem a_range (a : ℝ) (h_decreasing : decreasing_sequence (a_sequence a)) : 
  a > 1 / 2 ∧ a < 2 / 3 :=
sorry

end a_range_l625_625653


namespace parallelogram_of_dividing_segments_l625_625404

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

structure Parallelogram (a b c d : V) : Prop :=
  (ab_parallel_cd : (b - a) = (d - c))
  (ad_parallel_bc : (d - a) = (c - b))

theorem parallelogram_of_dividing_segments
  (A A' B B' C C' D D' A'' B'' C'' D'' : V)
  (h1 : Parallelogram A B C D)
  (h2 : Parallelogram A' B' C' D')
  (λ : ℝ)
  (hλ : 0 < λ ∧ λ < 1)
  (hA'' : A'' = λ • A + (1 - λ) • A')
  (hB'' : B'' = λ • B + (1 - λ) • B')
  (hC'' : C'' = λ • C + (1 - λ) • C')
  (hD'' : D'' = λ • D + (1 - λ) • D') :
  Parallelogram A'' B'' C'' D'' :=
sorry

end parallelogram_of_dividing_segments_l625_625404


namespace existence_of_B_l625_625741

-- We define n as a positive integer
variable (n : ℕ) (h_pos : 0 < n)

-- We define A as a square of side length n, divided into n^2 unit squares.
def A := fin n × fin n

-- We define a coloring function that assigns each square one of n distinct colors,
-- and each color appears exactly n times.
variable (coloring : A → fin n)
variable (color_count : ∀ c : fin n, (coloring ⁻¹' {c}).card = n)

-- The theorem statement we aim to prove.
theorem existence_of_B (N : ℕ) :
  ∃ N : ℕ, ∀ n : ℕ, n > N →
  ∃ (B : set A), (∃ (a b : fin n), B = { (i, j) | i ≥ a ∧ i < a + ⌊sqrt n⌋ ∧ j ≥ b ∧ j < b + ⌊sqrt n⌋ }) ∧ 
  (∃ colors : fin 4 → fin n, ∀ i j ∈ B, ∃ c ∈ range 4, coloring (i, j) = colors c) :=
sorry

end existence_of_B_l625_625741


namespace no_two_inscribed_triangles_l625_625940

-- Definitions of triangles and the inscribed relation
structure Triangle :=
  (A B C : Point)
  (distinct_vertices : A ≠ B ∧ B ≠ C ∧ C ≠ A)

def is_inscribed (H1 H2 : Triangle) : Prop :=
  (H1.A ∈ line_segment H2.B H2.C ∧
   H1.B ∈ line_segment H2.A H2.C ∧
   H1.C ∈ line_segment H2.A H2.B)

-- The main theorem to prove
theorem no_two_inscribed_triangles : ¬ (∃ H1 H2 : Triangle, is_inscribed H1 H2 ∧ is_inscribed H2 H1) :=
begin
  sorry
end

end no_two_inscribed_triangles_l625_625940


namespace patrick_purchased_pencils_l625_625777

theorem patrick_purchased_pencils 
  (S : ℝ) -- selling price of one pencil
  (C : ℝ) -- cost price of one pencil
  (P : ℕ) -- number of pencils purchased
  (h1 : C = 1.3333333333333333 * S) -- condition 1: cost of pencils is 1.3333333 times the selling price
  (h2 : (P : ℝ) * C - (P : ℝ) * S = 20 * S) -- condition 2: loss equals selling price of 20 pencils
  : P = 60 := 
sorry

end patrick_purchased_pencils_l625_625777


namespace smallest_n_for_S_n_integer_l625_625294

noncomputable def S_n (n : ℕ) : ℚ := 
  let K := (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) in
  (n * 5^(n - 1)) * K

theorem smallest_n_for_S_n_integer :
  ∃ (n : ℕ), (n > 0) ∧ (S_n n).denom = 1 ∧ ∀ m : ℕ, (m > 0) ∧ (m < n) → (S_n m).denom ≠ 1 := 
by
  sorry

end smallest_n_for_S_n_integer_l625_625294


namespace quadratic_expression_value_l625_625360

theorem quadratic_expression_value (x1 x2 : ℝ)
    (h1: x1^2 + 5 * x1 + 1 = 0)
    (h2: x2^2 + 5 * x2 + 1 = 0) :
    ( (x1 * Real.sqrt 6 / (1 + x2))^2 + (x2 * Real.sqrt 6 / (1 + x1))^2 ) = 220 := 
sorry

end quadratic_expression_value_l625_625360


namespace prove_height_l625_625079

-- Define the structure of the cuboid's dimensions based on the ratio 2:3:4.
structure CuboidDimensions where
  L : ℝ
  W : ℝ
  H : ℝ
  ratio : L = 2 * W / 3 ∧ W = 3 * L / 2 ∧ H = 4 * L / 2

-- Define the volume and base area conditions.
axiom volume (c : CuboidDimensions) : c.L * c.W * c.H = 144
axiom base_area (c : CuboidDimensions) : c.L * c.W = 18

-- The theorem to prove the height of the cuboid is approximately 7.268 meters.
theorem prove_height (c : CuboidDimensions) (h₁ : volume c) (h₂ : base_area c) : c.H = 7.268 := 
sorry

end prove_height_l625_625079


namespace correct_relationship_5_25_l625_625893

theorem correct_relationship_5_25 : 5^2 = 25 :=
by
  sorry

end correct_relationship_5_25_l625_625893


namespace phone_price_l625_625535

theorem phone_price:
  ∀ (aliyah_phones vivienne_phones total_money per_phone : ℕ)
    (h1: aliyah_phones = vivienne_phones + 10)
    (h2: vivienne_phones = 40)
    (h3: total_money = 36000)
    (h4: aliyah_phones + vivienne_phones ≠ 0), -- ensure total number of phones is non-zero to avoid division by zero
    per_phone = total_money / (aliyah_phones + vivienne_phones) :=
by
  assume aliyah_phones vivienne_phones total_money per_phone
  assume h1 h2 h3 h4
  sorry

end phone_price_l625_625535


namespace molecular_weight_CaSO4_2H2O_l625_625578

def Ca := 40.08
def S := 32.07
def O := 16.00
def H := 1.008

def Ca_weight := 1 * Ca
def S_weight := 1 * S
def O_in_sulfate_weight := 4 * O
def O_in_water_weight := 4 * O
def H_in_water_weight := 4 * H

def total_weight := Ca_weight + S_weight + O_in_sulfate_weight + O_in_water_weight + H_in_water_weight

theorem molecular_weight_CaSO4_2H2O : total_weight = 204.182 := 
by {
  sorry
}

end molecular_weight_CaSO4_2H2O_l625_625578


namespace compute_product_of_factors_l625_625134

-- Define the problem as a product of factors from 2 to 11
def product_factors : ℝ :=
  ∏ n in (Finset.range 10 \ {0, 1}).map (Function.Embedding.subtype Embedding.refl), 1 + (1 / (n: ℝ))

-- The theorem to show that the product equals 6
theorem compute_product_of_factors :
  product_factors = 6 := sorry

end compute_product_of_factors_l625_625134


namespace lcm_of_20_45_75_l625_625859

-- Definitions for the given numbers and their prime factorizations
def num1 : ℕ := 20
def num2 : ℕ := 45
def num3 : ℕ := 75

def factor1 : ℕ → Prop := λ n, n = 2 ^ 2 * 5
def factor2 : ℕ → Prop := λ n, n = 3 ^ 2 * 5
def factor3 : ℕ → Prop := λ n, n = 3 * 5 ^ 2

-- The condition of using the least common multiple function from mathlib
def lcm_def (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- The statement to prove
theorem lcm_of_20_45_75 : lcm_def num1 num2 num3 = 900 := by
  -- Factors condition (Note: These help ensure the numbers' factors are as stated)
  have h1 : factor1 num1 := by { unfold num1 factor1, exact rfl }
  have h2 : factor2 num2 := by { unfold num2 factor2, exact rfl }
  have h3 : factor3 num3 := by { unfold num3 factor3, exact rfl }
  sorry -- This is the place where the proof would go.

end lcm_of_20_45_75_l625_625859


namespace map_coloring_5_areas_l625_625098

/-- Theorem: The number of ways to color a map with 5 administrative areas such that no two adjacent areas use the same color, given 4 available colors, is 72. -/
theorem map_coloring_5_areas : 
  let regions := 5 in
  let colors := 4 in
  number_of_valid_colorings regions colors = 72 := sorry

end map_coloring_5_areas_l625_625098


namespace combined_teaching_years_l625_625852

-- Define the variables
variables 
  (A : ℕ)    -- Number of years Adrienne has taught
  (V : ℕ)    -- Number of years Virginia has taught
  (D : ℕ)    -- Number of years Dennis has taught

-- Define the conditions
def condition1 := V = A + 9
def condition2 := V = D - 9
def condition3 := D = 43

theorem combined_teaching_years :
  condition1 →
  condition2 →
  condition3 →
  A + V + D = 102 :=
by
  intros
  sorry

end combined_teaching_years_l625_625852


namespace find_solns_to_eqn_l625_625152

theorem find_solns_to_eqn (x y z w : ℕ) :
  2^x * 3^y - 5^z * 7^w = 1 ↔ (x, y, z, w) = (1, 0, 0, 0) ∨ 
                                        (x, y, z, w) = (3, 0, 0, 1) ∨ 
                                        (x, y, z, w) = (1, 1, 1, 0) ∨ 
                                        (x, y, z, w) = (2, 2, 1, 1) := 
sorry -- Placeholder for the actual proof

end find_solns_to_eqn_l625_625152


namespace one_over_x2_minus_x_eq_neg_one_l625_625693

def complex_x (i : ℂ) : ℂ := (1 + i * Real.sqrt 3) / 2

theorem one_over_x2_minus_x_eq_neg_one (i : ℂ) (hi : i ^ 2 = -1) :
  let x := complex_x i
  1 / (x ^ 2 - x) = -1 :=
by
  sorry

end one_over_x2_minus_x_eq_neg_one_l625_625693


namespace positive_integer_product_divisibility_l625_625013

theorem positive_integer_product_divisibility (x : ℕ → ℕ) (n p k : ℕ)
    (P : ℕ) (hx : ∀ i, 1 ≤ i → i ≤ n → x i < 2 * x 1)
    (hpos : ∀ i, 1 ≤ i → i ≤ n → 0 < x i)
    (hstrict : ∀ i j, 1 ≤ i → i < j → j ≤ n → x i < x j)
    (hn : 3 ≤ n)
    (hp : Nat.Prime p)
    (hk : 0 < k)
    (hP : P = ∏ i in Finset.range n, x (i + 1))
    (hdiv : p ^ k ∣ P) : 
  (P / p^k) ≥ Nat.factorial n := by
  sorry

end positive_integer_product_divisibility_l625_625013


namespace trailing_zeros_mod_100_l625_625746

def trailing_zeros_product_factorials (n : ℕ) : ℕ :=
  let count_factors (p k : ℕ) : ℕ :=  if k < p then 0 else k / p + count_factors p (k / p)
  count_factors 5 n

theorem trailing_zeros_mod_100 : 
  let M := trailing_zeros_product_factorials 50
  M % 100 = 12 :=
by
  let M := trailing_zeros_product_factorials 50
  have hM : M = 12 := by sorry
  exact congrArg (λ x, x % 100) hM

end trailing_zeros_mod_100_l625_625746


namespace sum_of_coefficients_nonzero_y_power_l625_625650

theorem sum_of_coefficients_nonzero_y_power :
  let expr := (5*x + 3*y - 4) * (2*x - 3*y + 6) in
    coeffsSumNonzeroY expr = 12 :=
  by sorry

end sum_of_coefficients_nonzero_y_power_l625_625650


namespace problem_l625_625756

theorem problem (x : ℝ) (h : x^2 - real.sqrt 6 * x + 1 = 0) :
  abs (x^4 - (1 / x)^4) = 8 * real.sqrt 3 :=
sorry

end problem_l625_625756


namespace arithmetic_and_geometric_sequences_problem_l625_625182

theorem arithmetic_and_geometric_sequences_problem :
  (∃ (a1 a2 b1 b2 b3 : ℝ) (d q : ℝ), -9 + 3 * d = -1 ∧ (-9) * q^4 = -1 ∧ q = real.sqrt (3^2) / 3 ∧ 
    b2 = -9 * q^2 ∧ b2 * (a2 - a1) = -8) → b2 * (a2 - a1) = -8 :=
begin
  sorry
end

end arithmetic_and_geometric_sequences_problem_l625_625182


namespace largest_sum_of_distinct_factors_of_1764_l625_625271

theorem largest_sum_of_distinct_factors_of_1764 :
  ∃ (A B C : ℕ), A * B * C = 1764 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A + B + C = 33 :=
by
  sorry

end largest_sum_of_distinct_factors_of_1764_l625_625271


namespace last_three_digits_of_sum_palindromic_l625_625160

theorem last_three_digits_of_sum_palindromic :
  let digit_sum := 
    let term1 := 1 % 1000,
        term2 := 121 % 1000,
        common_term := 321 % 1000 in
    term1 + term2 + common_term * 2008
  in digit_sum % 1000 = 690 :=
by
  sorry

end last_three_digits_of_sum_palindromic_l625_625160


namespace lemonade_price_fraction_l625_625084

theorem lemonade_price_fraction :
  (2 / 5) * (L / S) = 0.35714285714285715 → L / S = 0.8928571428571429 :=
by
  intro h
  sorry

end lemonade_price_fraction_l625_625084


namespace inappropriate_survey_method_l625_625053

def survey_method_appropriate (method : String) : Bool :=
  method = "sampling" -- only sampling is considered appropriate in this toy model

def survey_approps : Bool :=
  let A := survey_method_appropriate "sampling"
  let B := survey_method_appropriate "sampling"
  let C := ¬ survey_method_appropriate "census"
  let D := survey_method_appropriate "census"
  C

theorem inappropriate_survey_method :
  survey_approps = true :=
by
  sorry

end inappropriate_survey_method_l625_625053


namespace switches_in_position_A_after_all_steps_l625_625388

-- Definitions for the conditions
def initial_positions : List Nat := List.replicate 10000 0  -- 0 represents position A
def switch_labels : List Int :=
  List.map (λ x => 2^(x.1) * 3^(x.2) * 5^(x.3) * 7^(x.4)) ((List.finRange 10).pow 4)

-- Predicate to compute if a switch is in position A after steps
def is_in_position_A (adv_count : Nat) := adv_count % 5 = 0

-- Count the number of advances for a given switch
def advance_count (i : Nat) : Nat :=
  let di := switch_labels.get i;
  (List.foldl (λ s j => if di ∣ switch_labels.get j then s + 1 else s) 0 (List.finRange 10000)) + 1

-- Final proof problem
theorem switches_in_position_A_after_all_steps :
  List.countp (λ i => is_in_position_A (advance_count i)) (List.finRange 10000) = 8704 :=
sorry

end switches_in_position_A_after_all_steps_l625_625388


namespace second_quarter_profit_l625_625937

theorem second_quarter_profit (q1 q3 q4 annual : ℕ) (h1 : q1 = 1500) (h2 : q3 = 3000) (h3 : q4 = 2000) (h4 : annual = 8000) :
  annual - (q1 + q3 + q4) = 1500 :=
by
  sorry

end second_quarter_profit_l625_625937


namespace exists_group_of_four_l625_625711

-- Assuming 21 students, and any three have done homework together exactly once in either mathematics or Russian.
-- We aim to prove there exists a group of four students such that any three of them have done homework together in the same subject.
noncomputable def students : Type := Fin 21

-- Define a predicate to show that three students have done homework together.
-- We use "math" and "russian" to denote the subjects.
inductive Subject
| math
| russian

-- Define a relation expressing that any three students have done exactly one subject homework together.
axiom homework_done (s1 s2 s3 : students) : Subject 

theorem exists_group_of_four :
  ∃ (a b c d : students), 
    (homework_done a b c = homework_done a b d) ∧
    (homework_done a b c = homework_done a c d) ∧
    (homework_done a b c = homework_done b c d) ∧
    (homework_done a b d = homework_done a c d) ∧
    (homework_done a b d = homework_done b c d) ∧
    (homework_done a c d = homework_done b c d) :=
sorry

end exists_group_of_four_l625_625711


namespace regression_line_eq_l625_625670

theorem regression_line_eq (b a : ℝ) (x y : ℝ) (center : ℝ × ℝ) 
  (h1 : a = 3) (h2 : center = (1, 2)) 
  (h3 : y = b * x + a) :
  y = -x + 3 :=
by
  -- Center point relationship
  let (c_x, c_y) := center
  have h_center : c_y = b * c_x + a := by rw [h2, h3, h1]
  -- Solve for b using the center point
  have h_b : 2 = b * 1 + 3 := h_center
  have h_b_solve : b = -1 := by linarith
  -- Substitute b into the regression line equation and simplify
  rw [h_b_solve]
  -- Simplified regression line equation is y = -x + 3
  rw [h3, h1, h_b_solve]
  sorry

end regression_line_eq_l625_625670


namespace prove_distinct_range_of_a_l625_625691

noncomputable def distinct_range_of_a (a b c : ℝ) : Prop :=
  (b^2 + c^2 = 2 * a^2 + 16 * a + 14) ∧
  (bc = a^2 - 4 * a - 5) ∧
  b ≠ c ∧ a ≠ b ∧ a ≠ c →
  (a > -1) ∧
  (a ≠ -5 / 6) ∧ 
  (a ≠ (1 + real.sqrt 21) / 4) ∧ 
  (a ≠ (1 - real.sqrt 21) / 4) ∧ 
  (a ≠ -7 / 8)

theorem prove_distinct_range_of_a (a b c : ℝ) :
  distinct_range_of_a a b c :=
sorry

end prove_distinct_range_of_a_l625_625691


namespace salary_of_E_l625_625824

theorem salary_of_E (A B C D E : ℕ) (avg_salary : ℕ) 
  (hA : A = 8000) 
  (hB : B = 5000) 
  (hC : C = 11000) 
  (hD : D = 7000) 
  (h_avg : avg_salary = 8000) 
  (h_total_avg : avg_salary * 5 = A + B + C + D + E) : 
  E = 9000 :=
by {
  sorry
}

end salary_of_E_l625_625824


namespace pears_value_equivalence_l625_625800

-- Condition: $\frac{3}{4}$ of $16$ apples are worth $12$ pears
def apples_to_pears (a p : ℕ) : Prop :=
  (3 * 16 / 4 * a = 12 * p)

-- Question: How many pears (p) are equivalent in value to $\frac{2}{3}$ of $9$ apples?
def pears_equivalent_to_apples (p : ℕ) : Prop :=
  (2 * 9 / 3 * p = 6)

theorem pears_value_equivalence (p : ℕ) (a : ℕ) (h1 : apples_to_pears a p) (h2 : pears_equivalent_to_apples p) : 
  p = 6 :=
sorry

end pears_value_equivalence_l625_625800


namespace jennys_wedding_guests_l625_625732

noncomputable def total_guests (C S : ℕ) : ℕ := C + S

theorem jennys_wedding_guests :
  ∃ (C S : ℕ), (S = 3 * C) ∧
               (18 * C + 25 * S = 1860) ∧
               (total_guests C S = 80) :=
sorry

end jennys_wedding_guests_l625_625732


namespace sum_inequality_l625_625169

variable {n : ℕ}
variable {a b : Fin n → ℝ}
variable (h_pos1 : ∀ i, 0 < a i)
variable (h_pos2 : ∀ i, 0 < b i)
variable (h_sum : (∑ i, a i) = (∑ i, b i))

theorem sum_inequality :
  (∑ i, a i^2 / (a i + b i)) ≥ (1 / 2) * ∑ i, a i := by
  sorry

end sum_inequality_l625_625169


namespace vector_parallel_l625_625217

theorem vector_parallel (x : ℝ) (a b : ℝ × ℝ) (h : a = (3,4) ∧ b = (x, 1/2) ∧ a.1 * b.2 = a.2 * b.1) : x = 3 / 8 :=
by
  have ha : a = (3,4) := h.1
  have hb : b = (x,1/2) := h.2.1
  have h_parallel : 3 * (1/2) = 4 * x := h.2.2
  sorry

end vector_parallel_l625_625217


namespace LCM_20_45_75_is_900_l625_625879

def prime_factorization_20 := (2^2, 5)
def prime_factorization_45 := (3^2, 5)
def prime_factorization_75 := (3, 5^2)

theorem LCM_20_45_75_is_900 
  (pf_20 : prime_factorization_20 = (2^2, 5))
  (pf_45 : prime_factorization_45 = (3^2, 5))
  (pf_75 : prime_factorization_75 = (3, 5^2)) : 
  Nat.lcm (Nat.lcm 20 45) 75 = 900 := 
  by sorry

end LCM_20_45_75_is_900_l625_625879


namespace triangle_area_l625_625035

theorem triangle_area 
  (h1 : ∀ x y, (y = (1 : ℝ) / 3 * x + 2 / 3) ∨ (y = 3 * x - 2))
  (h2 : x + y = 8)
  (p1 : (1 : ℝ), (1 : ℝ))
  (p2 : (5.5 : ℝ), (2.5 : ℝ))
  (p3 : (2.5 : ℝ), (5.5 : ℝ)) :
  let area := (1 / 2 : ℝ) * abs (1 * (5.5 - 2.5) + 2.5 * (2.5 - 1) + 5.5 * (1 - 5.5))
  in area = 9 := 
  by 
  sorry

end triangle_area_l625_625035


namespace no_integer_roots_in_transformation_l625_625381

theorem no_integer_roots_in_transformation :
  ∀ (b c : ℤ), 
  (10 ≤ b ∧ b ≤ 20) ∧ (10 ≤ c ∧ c ≤ 20) ∧ 
  (∀ k : ℤ, k ∈ {b - initial_b, initial_c - c} → k = 1 ∨ k = -1) → 
  ¬(∃ b c : ℤ, b^2 - 4 * c ≥ 0 ∧ square (b^2 - 4 * c)) :=
sorry

end no_integer_roots_in_transformation_l625_625381


namespace domain_change_l625_625065

theorem domain_change (f : ℝ → ℝ) :
  (∀ x : ℝ, -2 ≤ x + 1 ∧ x + 1 ≤ 3) →
  (∀ x : ℝ, -2 ≤ 1 - 2 * x ∧ 1 - 2 * x ≤ 3) →
  ∀ x : ℝ, -3 / 2 ≤ x ∧ x ≤ 1 :=
by {
  sorry
}

end domain_change_l625_625065


namespace charity_years_l625_625927

theorem charity_years :
  ∃! pairs : List (ℕ × ℕ), 
    (∀ (w m : ℕ), (w, m) ∈ pairs → 18 * w + 30 * m = 55 * 12) ∧
    pairs.length = 6 :=
by
  sorry

end charity_years_l625_625927


namespace proportion_MN_AD_l625_625275

variable {ABC : Triangle}

variables {D : Point} {BC BD DC : Length}
variables {E K : Point} {AC AE EK : Length}
variables {M N : Point} {BE ME BM BK BN NK AD MN : Length}

-- Conditions:
variable (h1 : BD = 1 * DC)
variable (h2 : (BM / ME) = 7 / 5)
variable (h3 : (BN / NK) = 2 / 3)

-- Statement of the problem:
theorem proportion_MN_AD : (MN / AD) = 11 / 45 :=
by
  sorry

end proportion_MN_AD_l625_625275


namespace max_and_min_product_of_nonzero_naturals_l625_625188

theorem max_and_min_product_of_nonzero_naturals 
  (a b : ℕ) (h_nonzero : a ≠ 0 ∧ b ≠ 0) (h_sum : a + b = 100) : 
  (max (a * b) (min (a * b)) = 2500 ∧ min (a * b) = 99) := 
by
  sorry

end max_and_min_product_of_nonzero_naturals_l625_625188


namespace distance_y_axis_l625_625487

-- Define the coordinates of point P
def P : ℝ × ℝ := (x, -4)

-- Define the distance of P from the x-axis and y-axis
def d_x (P : ℝ × ℝ) : ℝ := abs P.snd
def d_y (P : ℝ × ℝ) : ℝ := abs P.fst

-- State the problem
theorem distance_y_axis (x : ℝ) (h : d_x P = 1/2 * d_y P) : d_y P = 8 :=
by sorry

end distance_y_axis_l625_625487


namespace lcm_20_45_75_l625_625885

def lcm (a b : ℕ) : ℕ := nat.lcm a b

theorem lcm_20_45_75 : lcm (lcm 20 45) 75 = 900 :=
by
  sorry

end lcm_20_45_75_l625_625885


namespace michael_points_l625_625315

noncomputable def points_calculation (freshman_year_points : ℕ) (improvement_sophomore : ℕ) 
                                     (junior_more_percent : ℚ) (senior_more_percent : ℚ) 
                                     (senior_extra_points : ℕ) : ℚ :=
  let sophomore_points := freshman_year_points + (improvement_sophomore / 100) * freshman_year_points
  let junior_points := sophomore_points + (junior_more_percent / 100) * sophomore_points
  let senior_points := junior_points + (senior_more_percent / 100) * junior_points + senior_extra_points
  freshman_year_points + sophomore_points + junior_points + senior_points

theorem michael_points (P : ℚ) : 
  points_calculation 260 15 P 12 17 = 1450 → P ≈ 38 := 
by
  let freshman_year_points := 260
  let improvement_sophomore := 15
  let senior_more_percent := 12
  let senior_extra_points := 17
  let sophomore_points := freshman_year_points + (improvement_sophomore / 100) * freshman_year_points
  let junior_points := sophomore_points + (P / 100) * sophomore_points
  let senior_points := junior_points + (senior_more_percent / 100) * junior_points + senior_extra_points
  have total_points := freshman_year_points + sophomore_points + junior_points + senior_points
  have total_points_eq := total_points = 1450
  sorry

end michael_points_l625_625315


namespace min_m_plus_n_l625_625987

-- Lean statement
variable (λ μ m n : ℝ)

def vector_OA := (1, 0 : ℝ × ℝ)
def vector_OB := (1, 1 : ℝ × ℝ)

def x_y := λ • vector_OA + μ • vector_OB

theorem min_m_plus_n :
  (0 ≤ λ ∧ λ ≤ 1 ∧ 1 ≤ μ ∧ μ ≤ 2) →
  (m > 0 ∧ n > 0) →
  (∃ max_z : ℝ, max_z = 2) →
  (m + n) ≥ (5/2 + real.sqrt(6)) :=
by
  sorry

end min_m_plus_n_l625_625987


namespace tickets_distribution_correct_l625_625947

def tickets_distribution (tickets programs : nat) (A_tickets_min : nat) : nat :=
sorry

theorem tickets_distribution_correct :
  tickets_distribution 6 4 3 = 17 :=
by
  sorry

end tickets_distribution_correct_l625_625947


namespace shortest_distance_parabola_line_l625_625290

noncomputable def shortest_distance : ℝ :=
  let parabola := λ x : ℝ, x^2 - 4 * x + 8
  let line := λ x : ℝ, 2 * x - 3
  let distance (a : ℝ) : ℝ := |2 * a - (parabola a) + 3| / real.sqrt (2^2 + 1^2)
  real.sqrt (2^2 + 1^1) * infi (λ a : ℝ, |-(a - 1) * (a - 5)| / real.sqrt (2^2 + 1^1))

theorem shortest_distance_parabola_line :
  shortest_distance = 4 * real.sqrt 5 / 5 :=
  sorry

end shortest_distance_parabola_line_l625_625290


namespace simplify_expression_l625_625343

theorem simplify_expression : 
  (sqrt 450 / sqrt 400 + sqrt 98 / sqrt 56) = (3 + 2 * sqrt 7) / 4 := 
by
  sorry

end simplify_expression_l625_625343


namespace complex_division_l625_625190

noncomputable def imagine_unit : ℂ := Complex.I

theorem complex_division :
  (Complex.mk (-3) 1) / (Complex.mk 1 (-1)) = (Complex.mk (-2) 1) :=
by
sorry

end complex_division_l625_625190


namespace factorization_problem_l625_625812

theorem factorization_problem :
  ∃ (a b : ℤ), (25 * x^2 - 130 * x - 120 = (5 * x + a) * (5 * x + b)) ∧ (a + 3 * b = -86) := by
  sorry

end factorization_problem_l625_625812


namespace part_1_part_2_l625_625200

theorem part_1 (m : ℝ) : ∀ (x : ℝ), ∃ p q : ℝ, (p ≠ q ∧ (x + m) ^ 2 - 4 = 0) :=
by
  sorry

theorem part_2 (p q m : ℝ) (h : (p + m) ^ 2 - 4 = 0) (hpq : pq = p + q) : (m = -1 + sqrt 5) ∨ (m = -1 - sqrt 5) :=
by
  sorry

end part_1_part_2_l625_625200


namespace minimum_tablets_each_kind_l625_625501

variables (tabletA tabletB : Type) [fintype tabletA] [fintype tabletB]
variables (nA nB : ℕ)

-- Conditions
def box_contains (n : ℕ) (tablets : Type) [fintype tablets] := fintype.card tablets = n

-- Predicate for the least number of tablets to ensure a certain number of each kind
def least_number_to_ensure_each_kind (least : ℕ) (tablet1 tablet2 : Type) [fintype tablet1] [fintype tablet2] :=
  (box_contains 10 tablet1) ∧ (box_contains 10 tablet2) ∧ least = 12

-- Theorem statement
theorem minimum_tablets_each_kind (least : ℕ) (tablet1 tablet2 : Type) [fintype tablet1] [fintype tablet2] :
  least_number_to_ensure_each_kind least tablet1 tablet2 → nA = 2 ∧ nB = 1 :=
sorry

end minimum_tablets_each_kind_l625_625501


namespace polygon_inequality_l625_625288

noncomputable def sum_reciprocal_parts (n : ℕ) (a : ℕ → ℝ) (p : ℝ) : ℝ :=
  ∑ i in Finset.range n, (a i) / (p - (a i))

theorem polygon_inequality (n : ℕ) (a : ℕ → ℝ) (p : ℝ)
  (h₁ : n ≥ 3)
  (h₂ : p = ∑ i in Finset.range n, a i)
  (h₃ : ∀ i, i ∈ Finset.range n → a i > 0) :
  sum_reciprocal_parts n a p < 2 := 
sorry

end polygon_inequality_l625_625288


namespace anna_stamp_count_correct_l625_625543

-- Defining the initial counts of stamps
def anna_initial := 37
def alison_initial := 28
def jeff_initial := 31

-- Defining the operations
def alison_gives_half_to_anna := alison_initial / 2
def anna_after_receiving_from_alison := anna_initial + alison_gives_half_to_anna
def anna_final := anna_after_receiving_from_alison - 2 + 1

-- Formalizing the proof problem
theorem anna_stamp_count_correct : anna_final = 50 := by
  -- proof omitted
  sorry

end anna_stamp_count_correct_l625_625543


namespace system_solution_5_3_l625_625472

variables (x y : ℤ)

theorem system_solution_5_3 :
  (x = 5) ∧ (y = 3) → (2 * x - 3 * y = 1) :=
by intros; sorry

end system_solution_5_3_l625_625472


namespace smallest_perfect_square_divisible_by_2_3_5_l625_625460

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, (n > 0) ∧ (nat.sqrt n * nat.sqrt n = n) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ n = 900 := sorry

end smallest_perfect_square_divisible_by_2_3_5_l625_625460


namespace number_of_truthful_dwarfs_l625_625587

-- Given conditions
variables (D : Type) [Fintype D] [DecidableEq D] [Card D = 10]
variables (IceCream : Type) [DecidableEq IceCream] (vanilla chocolate fruit : IceCream)
-- Assuming each dwarf likes exactly one type of ice cream
variable (Likes : D → IceCream)
-- Functions indicating if a dwarf raised their hand for each type of ice cream
variables (raisedHandForVanilla raisedHandForChocolate raisedHandForFruit : D → Prop)

-- Given conditions translated to Lean
axiom all_dwarfs_raised_for_vanilla : ∀ d, raisedHandForVanilla d
axiom half_dwarfs_raised_for_chocolate : Fintype.card {d // raisedHandForChocolate d} = 5
axiom one_dwarf_raised_for_fruit : Fintype.card {d // raisedHandForFruit d} = 1

-- Define that a dwarf either always tells the truth or always lies
inductive TruthStatus
| truthful : TruthStatus
| liar : TruthStatus

variable (Status : D → TruthStatus)

-- Definitions related to hand-raising based on dwarf's status and ice cream they like
def raisedHandCorrectly (d : D) : Prop :=
  match Status d with
  | TruthStatus.truthful => 
      raisedHandForVanilla d ↔ Likes d = vanilla ∧
      raisedHandForChocolate d ↔ Likes d = chocolate ∧
      raisedHandForFruit d ↔ Likes d = fruit
  | TruthStatus.liar =>
      raisedHandForVanilla d ↔ Likes d ≠ vanilla ∧
      raisedHandForChocolate d ↔ Likes d ≠ chocolate ∧
      raisedHandForFruit d ↔ Likes d ≠ fruit

-- Goal to prove
theorem number_of_truthful_dwarfs : Fintype.card {d // Status d = TruthStatus.truthful} = 4 :=
by sorry

end number_of_truthful_dwarfs_l625_625587


namespace parallelogram_area_l625_625226

theorem parallelogram_area (A B C D : Point) 
(hA : A = (1, 2)) 
(hB : B = (7, 2)) 
(hC : C = (5, 10)) 
(hD : D = (11, 10)) 
(h : Parallelogram A B C D) :
  area A B C D = 48 :=
sorry

end parallelogram_area_l625_625226


namespace invisible_points_square_l625_625925

theorem invisible_points_square (L : ℕ) (hL : L > 0) :
  ∃ (x y : ℤ), ∀ (i j : ℤ), 0 ≤ i ∧ i ≤ L → 0 ≤ j ∧ j ≤ L →
  ∃ p : ℕ, Prime p ∧ x + i ≡ 0 [MOD p] ∧ y + j ≡ 0 [MOD p] := by
-- proof steps would go here, but are omitted as per instructions
sorry

end invisible_points_square_l625_625925


namespace simplify_and_evaluate_l625_625794

-- Define the expression
def expr (a : ℚ) : ℚ := (3 * a - 1) ^ 2 + 3 * a * (3 * a + 2)

-- Given the condition
def a_value : ℚ := -1 / 3

-- State the theorem
theorem simplify_and_evaluate : expr a_value = 3 :=
by
  -- Proof will be added here
  sorry

end simplify_and_evaluate_l625_625794


namespace find_f_g_one_l625_625297

def f (x : ℝ) := x^2 - 2*x + 1
def g (x : ℝ) := x^2 + 1

theorem find_f_g_one : f (g 1) = 1 := 
by
  let h₁ := by rfl : g 1 = 2
  let h₂ := by rfl : f 2 = 1
  have h : f (g 1) = f 2 := by rw h₁
  rw [h, h₂]

end find_f_g_one_l625_625297


namespace surface_area_of_rectangular_prism_l625_625468

def SurfaceArea (length : ℕ) (width : ℕ) (height : ℕ) : ℕ :=
  2 * ((length * width) + (width * height) + (height * length))

theorem surface_area_of_rectangular_prism 
  (l w h : ℕ) 
  (hl : l = 1) 
  (hw : w = 2) 
  (hh : h = 2) : 
  SurfaceArea l w h = 16 := by
  sorry

end surface_area_of_rectangular_prism_l625_625468


namespace smallest_abundant_18_l625_625687

-- Define proper divisors for each number.
def properDivisors (n : ℕ) : List ℕ :=
  (List.range n).filter (fun d => d > 0 ∧ n % d = 0)

-- Define abundant number
def isAbundant (n : ℕ) : Prop :=
  (properDivisors n).sum > n

-- Define function that checks abundants in the list.
def smallestAbundant (ns : List ℕ) : Option ℕ :=
  ns.filter isAbundant |>.minimum?

-- Main statement.
theorem smallest_abundant_18 : smallestAbundant [18, 20, 22, 24] = some 18 :=
by
  sorry

end smallest_abundant_18_l625_625687


namespace find_angle_between_vectors_l625_625675

noncomputable def angle_between_vectors (a b : Vector ℝ 3) : ℝ := sorry

theorem find_angle_between_vectors
  (a b : Vector ℝ 3)
  (h_nonzero_a : a ≠ 0)
  (h_nonzero_b : b ≠ 0)
  (h_perpendicular : dot a (a + b) = 0)
  (h_length : norm b = real.sqrt 2 * norm a) :
  angle_between_vectors a b = 3 * real.pi / 4 :=
sorry

end find_angle_between_vectors_l625_625675


namespace calculate_sum_l625_625828

theorem calculate_sum (P r : ℝ) (h1 : 2 * P * r = 10200) (h2 : P * ((1 + r) ^ 2 - 1) = 11730) : P = 17000 :=
sorry

end calculate_sum_l625_625828


namespace simplify_expr_l625_625339

/-- Theorem: Simplify the expression -/
theorem simplify_expr
  (x y z w : ℝ)
  (hx : x = sqrt 3 - 1)
  (hy : y = sqrt 3 + 1)
  (hz : z = 1 - sqrt 2)
  (hw : w = 1 + sqrt 2) :
  (x ^ z / y ^ w) = 2 ^ (1 - sqrt 2) * (4 - 2 * sqrt 3) :=
by
  sorry

end simplify_expr_l625_625339


namespace sequence_bounded_l625_625516

open scoped Classical

noncomputable def isMax (f : ℕ → ℝ) (n : ℕ) (m : ℝ) : Prop :=
∀ i j, i + j = n → f i + f j ≤ m

def seq_condition (a : ℕ → ℝ) (D : ℕ) : Prop :=
∀ n, n > D → a n = - (some (λ m, isMax a n m))

theorem sequence_bounded (a : ℕ → ℝ) (D : ℕ) (h : seq_condition a D) : 
  ∃ M, ∀ n, |a n| ≤ M := 
sorry

end sequence_bounded_l625_625516


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l625_625443

/-- Problem statement: 
    The smallest positive perfect square that is divisible by 2, 3, and 5 is 900.
-/
theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ n * n % 2 = 0 ∧ n * n % 3 = 0 ∧ n * n % 5 = 0 ∧ n * n = 900 := 
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l625_625443


namespace minimum_chocolates_l625_625797

theorem minimum_chocolates (x : ℤ) (h1 : x ≥ 150) (h2 : x % 15 = 7) : x = 157 :=
sorry

end minimum_chocolates_l625_625797


namespace sequence_general_term_l625_625219

theorem sequence_general_term (n : ℕ) (hn : 0 < n) :
  let a_n := 2 * n + 1 + (1 / 2)^(n + 1) in 
  a_n ∈ { 3 + (1 / 2)^2, 5 + (1 / 2)^3, 7 + (1 / 2)^4, 9 + (1 / 2)^5 } :=
sorry

end sequence_general_term_l625_625219


namespace min_vertical_distance_l625_625366

theorem min_vertical_distance :
  ∃ (d : ℝ), ∀ (x : ℝ),
    (y1 x = |x - 1| ∧ y2 x = -x^2 - 4*x - 3) ∧ 
    (d = infi (λ x, abs (y1 x - y2 x))) ∧
    (d = 7 / 4) :=
by
  let y1 := λ x : ℝ, abs (x - 1)
  let y2 := λ x : ℝ, -x^2 - 4 * x - 3
  exists 7 / 4
  simp
  sorry

end min_vertical_distance_l625_625366


namespace smallest_perfect_square_divisible_by_2_3_5_l625_625459

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, (n > 0) ∧ (nat.sqrt n * nat.sqrt n = n) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ n = 900 := sorry

end smallest_perfect_square_divisible_by_2_3_5_l625_625459


namespace least_subtraction_to_divisible_l625_625977

theorem least_subtraction_to_divisible :
  ∃ m : ℕ, m = 1852745 % 251 := 
begin
  use 130,
  have h : 1852745 % 251 = 130 := rfl,  -- This step is to align with the given solution
  exact h
end

end least_subtraction_to_divisible_l625_625977


namespace expr1_eq_expr2_eq_l625_625495

-- Problem 1
theorem expr1_eq : ( (4 / 9 : ℚ) ^ (-3 / 2 : ℚ) + ((-2 : ℚ) ^ 6) ^ (1 / 2 : ℚ) -
  Real.log10 0.4 - 2 * Real.log10 0.5 - 14 * Real.logb 2 (Real.sqrt 2) = 35 / 8 ) := 
by
  sorry

-- Problem 2
theorem expr2_eq (α : ℝ) (h : Real.cos α = (1 / 2) * Real.sin α) :
  (Real.cos (π - α) + Real.sin (π + α)) / (Real.cos (π / 2 - α) + Real.sin (π / 2 + α)) + 
    2 * Real.sin α * Real.cos α = -1/5 :=
by
  sorry

end expr1_eq_expr2_eq_l625_625495


namespace charlyn_total_viewable_area_l625_625565

theorem charlyn_total_viewable_area :
  let side_length := 5
  let visibility_radius := 0.5
  let total_area := (side_length^2) - ((side_length - 2 * visibility_radius)^2) +
                    4 * (side_length * visibility_radius) + (4 * (Real.pi * (visibility_radius^2) / 4))
  let rounded_total_area := Real.floor (total_area + 0.5)
  rounded_total_area = 22 :=
by
  let side_length := 5
  let visibility_radius := 0.5
  let total_area := (side_length^2) - ((side_length - 2 * visibility_radius)^2) +
                    4 * (side_length * visibility_radius) + (4 * (Real.pi * (visibility_radius^2) / 4))
  let rounded_total_area := Real.floor (total_area + 0.5)
  exact eq.refl rounded_total_area
  -- sorry

end charlyn_total_viewable_area_l625_625565


namespace smallest_perfect_square_divisible_by_2_3_5_l625_625461

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, (n > 0) ∧ (nat.sqrt n * nat.sqrt n = n) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ n = 900 := sorry

end smallest_perfect_square_divisible_by_2_3_5_l625_625461


namespace problem_statement_l625_625766

theorem problem_statement (x : ℝ) (h : 2^x + log x / log 2 = 0) : x < 1 ∧ 1 < 2^x :=
by sorry

end problem_statement_l625_625766


namespace legs_on_ground_l625_625913

theorem legs_on_ground (n_men n_horses : ℕ) (h_eq: n_men = n_horses) (h_men: n_men = 18) (h_riding: n_men / 2) : 
∑ x in (finset.range (n_men / 2)), 2 + ∑ x in (finset.range (n_men / 2)), 4 * x = 54 :=
by
  -- We need to add some actual logic here, the current logic is a placeholder
  sorry

end legs_on_ground_l625_625913


namespace unique_prime_number_satisfying_conditions_l625_625311

lemma prime_condition_unique_prime (p : ℕ) (x y : ℕ)
  (hp_prime : p.prime)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h₁ : p - 1 = 2 * x^2)
  (h₂ : p^2 - 1 = 2 * y^2) : p = 3 :=
by {
  sorry
}

theorem unique_prime_number_satisfying_conditions : 
  ∃! p : ℕ, (∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ p.prime ∧ p - 1 = 2 * x^2 ∧ p^2 - 1 = 2 * y^2) :=
by {
  use 3,
  split,
  {
    use 1,
    use 2,
    split,
    { exact one_pos },
    split,
    { linarith },
    split,
    { norm_num },
    split,
    { norm_num },
    { norm_num }
  },
  {
    intros p h,
    rcases h with ⟨x, y, hx_pos, hy_pos, hp_prime, h₁, h₂⟩,
    exact prime_condition_unique_prime p x y hp_prime hx_pos hy_pos h₁ h₂
  }
}

end unique_prime_number_satisfying_conditions_l625_625311


namespace simplify_fraction_l625_625338

theorem simplify_fraction :
  (1 / (1 / (Real.sqrt 3 + 1) + 3 / (Real.sqrt 5 - 2))) = 2 / (Real.sqrt 3 + 6 * Real.sqrt 5 + 11) :=
by
  sorry

end simplify_fraction_l625_625338


namespace tony_exercise_hours_per_week_l625_625395

variable (dist_walk dist_run speed_walk speed_run days_per_week : ℕ)

#eval dist_walk : ℕ := 3
#eval dist_run : ℕ := 10
#eval speed_walk : ℕ := 3
#eval speed_run : ℕ := 5
#eval days_per_week : ℕ := 7

theorem tony_exercise_hours_per_week :
  (dist_walk / speed_walk + dist_run / speed_run) * days_per_week = 21 := by
  sorry

end tony_exercise_hours_per_week_l625_625395


namespace range_of_m_l625_625993

theorem range_of_m (m : ℝ) : 
  (∀ (n : ℕ), n > 0 → ∑ i in finset.range (n + 1), 1 / (i^2 + 3 * i + 2) < m)
  → (m ≥ 3 / 4) := 
sorry

end range_of_m_l625_625993


namespace smallest_perfect_square_divisible_by_2_3_5_l625_625458

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧
         (∀ k : ℕ, 
           (k > 0 ∧ (∃ l : ℕ, k = l * l) ∧ (2 ∣ k) ∧ (3 ∣ k) ∧ (5 ∣ k)) → n ≤ k) :=
sorry

end smallest_perfect_square_divisible_by_2_3_5_l625_625458


namespace g_difference_l625_625236

def g (n : ℕ) : ℚ := (1 / 4) * n * (n + 1) * (n + 2) * (n + 3)

theorem g_difference (s : ℕ) : g s - g (s - 1) = s * (s + 1) * (s + 2) := 
by sorry

end g_difference_l625_625236


namespace find_two_digit_number_l625_625054

theorem find_two_digit_number (a b : ℕ) (h : 17.factorial = 355687 * 10^10 + a * 10^9 + b * 10^8 + 8096000) : (10 * a + b) = 75 :=
by {
  sorry
}

end find_two_digit_number_l625_625054


namespace percent_games_lost_l625_625823

def games_ratio (won lost : ℕ) : Prop :=
  won * 3 = lost * 7

def total_games (won lost : ℕ) : Prop :=
  won + lost = 50

def percentage_lost (lost total : ℕ) : ℕ :=
  lost * 100 / total

theorem percent_games_lost (won lost : ℕ) (h1 : games_ratio won lost) (h2 : total_games won lost) : 
  percentage_lost lost 50 = 30 := 
by
  sorry

end percent_games_lost_l625_625823


namespace circles_centers_distance_l625_625033

open Real

theorem circles_centers_distance (R1 R2 : ℝ) (angle_tangents : ℝ) 
  (hR1 : R1 = 15) (hR2 : R2 = 95) (h_angle : angle_tangents = 60) : 
  distance_circles_centers R1 R2 angle_tangents = 160 := 
sorry

def distance_circles_centers (R1 R2 angle_tangents : ℝ) := 
  2 * R2 * cos (angle_tangents * pi / 180 / 2) - 2 * R1 * cos (angle_tangents * pi / 180 / 2)

end circles_centers_distance_l625_625033


namespace smallest_positive_perfect_square_div_by_2_3_5_l625_625427

-- Definition capturing the conditions of the problem
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def divisible_by (n m : ℕ) : Prop :=
  m % n = 0

-- The theorem statement combining all conditions and requiring the proof of the correct answer
theorem smallest_positive_perfect_square_div_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ is_perfect_square n ∧ divisible_by 2 n ∧ divisible_by 3 n ∧ divisible_by 5 n ∧ n = 900 :=
sorry

end smallest_positive_perfect_square_div_by_2_3_5_l625_625427


namespace graph_not_through_third_quadrant_l625_625007

theorem graph_not_through_third_quadrant (k : ℝ) (h_nonzero : k ≠ 0) (h_decreasing : k < 0) : 
  ¬(∃ x y : ℝ, y = k * x - k ∧ x < 0 ∧ y < 0) :=
sorry

end graph_not_through_third_quadrant_l625_625007


namespace number_of_truthful_dwarfs_l625_625584

-- Given conditions
variables (D : Type) [Fintype D] [DecidableEq D] [Card D = 10]
variables (IceCream : Type) [DecidableEq IceCream] (vanilla chocolate fruit : IceCream)
-- Assuming each dwarf likes exactly one type of ice cream
variable (Likes : D → IceCream)
-- Functions indicating if a dwarf raised their hand for each type of ice cream
variables (raisedHandForVanilla raisedHandForChocolate raisedHandForFruit : D → Prop)

-- Given conditions translated to Lean
axiom all_dwarfs_raised_for_vanilla : ∀ d, raisedHandForVanilla d
axiom half_dwarfs_raised_for_chocolate : Fintype.card {d // raisedHandForChocolate d} = 5
axiom one_dwarf_raised_for_fruit : Fintype.card {d // raisedHandForFruit d} = 1

-- Define that a dwarf either always tells the truth or always lies
inductive TruthStatus
| truthful : TruthStatus
| liar : TruthStatus

variable (Status : D → TruthStatus)

-- Definitions related to hand-raising based on dwarf's status and ice cream they like
def raisedHandCorrectly (d : D) : Prop :=
  match Status d with
  | TruthStatus.truthful => 
      raisedHandForVanilla d ↔ Likes d = vanilla ∧
      raisedHandForChocolate d ↔ Likes d = chocolate ∧
      raisedHandForFruit d ↔ Likes d = fruit
  | TruthStatus.liar =>
      raisedHandForVanilla d ↔ Likes d ≠ vanilla ∧
      raisedHandForChocolate d ↔ Likes d ≠ chocolate ∧
      raisedHandForFruit d ↔ Likes d ≠ fruit

-- Goal to prove
theorem number_of_truthful_dwarfs : Fintype.card {d // Status d = TruthStatus.truthful} = 4 :=
by sorry

end number_of_truthful_dwarfs_l625_625584


namespace correct_number_of_conclusions_l625_625651

-- Given conditions
def cond1 (a b : ℕ) (n : ℕ) : Prop := (a * b)^n = a^n * b^n → (a + b)^n = a^n + b^n
def cond2 (α β : ℝ) : Prop := (log (a * b) = log a + log b) → (sin (α + β) = sin α * sin β)
def cond3 {R : Type*} [ordered_semiring R] (a b : R) : Prop := (a + b)^2 = a^2 + 2 * a * b + b^2

-- Correctness of conclusions
def is_correct_cond1 := ¬ cond1 2 2 2
def is_correct_cond2 := ¬ cond2 1 1
def is_correct_cond3 {R : Type*} [ordered_semiring R] := cond3 (1:R) 1

-- Number of correct conclusions
def num_correct_conclusions : ℕ := if is_correct_cond1 then 0 else 1 +
                                   if is_correct_cond2 then 0 else 1 +
                                   if is_correct_cond3 then 1 else 0

-- Theorem to prove the number of correct conclusions
theorem correct_number_of_conclusions : num_correct_conclusions = 1 :=
by
  -- Proof to be provided
  sorry

end correct_number_of_conclusions_l625_625651


namespace range_of_m_l625_625354

variable (f : Real → Real)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom decreasing_function : ∀ x y, x < y → -1 < x ∧ y < 1 → f x > f y
axiom domain : ∀ x, -1 < x ∧ x < 1 → true

-- The statement to be proved
theorem range_of_m (m : Real) : 
  f (1 - m) + f (1 - m^2) < 0 → 0 < m → m < 1 :=
by
  sorry

end range_of_m_l625_625354


namespace bird_watcher_total_l625_625500

theorem bird_watcher_total
  (M : ℕ) (T : ℕ) (W : ℕ)
  (h1 : M = 70)
  (h2 : T = M / 2)
  (h3 : W = T + 8) :
  M + T + W = 148 :=
by
  -- proof omitted
  sorry

end bird_watcher_total_l625_625500


namespace quadratic_inequality_solution_set_l625_625254

theorem quadratic_inequality_solution_set
  (a b : ℝ)
  (h1 : 2 + 3 = -a)
  (h2 : 2 * 3 = b) :
  ∀ x : ℝ, 6 * x^2 - 5 * x + 1 > 0 ↔ x < (1 / 3) ∨ x > (1 / 2) := by
  sorry

end quadratic_inequality_solution_set_l625_625254


namespace isosceles_triangle_area_equality_l625_625291

def heron (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem isosceles_triangle_area_equality :
  let A := heron 13 13 10
  let B := heron 13 13 24
  A = B :=
by
  let A := heron 13 13 10
  let B := heron 13 13 24
  sorry

end isosceles_triangle_area_equality_l625_625291


namespace balls_in_third_pile_l625_625836

theorem balls_in_third_pile (a b c x : ℕ) (h1 : a + b + c = 2012) (h2 : b - x = 17) (h3 : a - x = 2 * (c - x)) : c = 665 := by
  sorry

end balls_in_third_pile_l625_625836


namespace integer_values_of_b_l625_625142

theorem integer_values_of_b (b : ℤ) :
  (∃ b, ∀ x : ℤ, x^2 + b * x - 2 ≤ 0 → (x = -3 ∨ x = -2 ∨ x = 2 ∨ x = 3)) → 
  ∃! (b: ℤ), (card {x : ℤ | x^2 + b * x - 2 ≤ 0} = 3) := sorry

end integer_values_of_b_l625_625142


namespace find_digits_l625_625172

def five_digit_subtraction (a b c d e : ℕ) : Prop :=
    let n1 := 10000 * a + 1000 * b + 100 * c + 10 * d + e
    let n2 := 10000 * e + 1000 * d + 100 * c + 10 * b + a
    (n1 - n2) % 10 = 2 ∧ (((n1 - n2) / 10) % 10) = 7 ∧ a > e ∧ a - e = 2 ∧ b - a = 7

theorem find_digits 
    (a b c d e : ℕ) 
    (h : five_digit_subtraction a b c d e) :
    a = 9 ∧ e = 7 :=
by 
    sorry

end find_digits_l625_625172


namespace perpendicular_k_value_parallel_k_value_l625_625628

-- Problem conditions
def a : (ℝ × ℝ) := (1, 2)
def b : (ℝ × ℝ) := (-3, 2)

-- Question 1: Prove that k = 19 if k*a + b is perpendicular to a-3*b
theorem perpendicular_k_value : 
  ∀ k : ℝ, (let ka_plus_b := ((k-3), 2*k+2) in ka_plus_b.1 * 10 + ka_plus_b.2 * (-4) = 0) → k = 19 := 
sorry

-- Question 2: Prove that k = -1/3 if k*a + b is parallel to a-3*b
theorem parallel_k_value :
  ∀ k : ℝ, (let ka_plus_b := ((k-3), 2*k+2)  in ka_plus_b.1 * (-4) - ka_plus_b.2 * (10) = 0) → k = (-1) / 3 :=
sorry

end perpendicular_k_value_parallel_k_value_l625_625628


namespace carrie_shopping_l625_625954

noncomputable def total_spent (t_shirts: ℕ) (t_shirt_price: ℝ) (jeans: ℕ) (jean_price: ℝ) (socks: ℕ) (sock_price: ℝ)
    (t_shirt_discount: ℝ) (jean_discount: ℝ) (sales_tax: ℝ): ℝ :=
let t_shirt_total := t_shirts * t_shirt_price in
let jean_total := jeans * jean_price in
let sock_total := socks * sock_price in
let t_shirt_discounted := t_shirt_total * t_shirt_discount in
let jean_discounted := jean_total * jean_discount in
let before_tax := (t_shirt_total - t_shirt_discounted) + (jean_total - jean_discounted) + sock_total in
let tax := before_tax * sales_tax in
before_tax + tax

theorem carrie_shopping: 
    total_spent 12 9.65 3 29.95 5 4.5 0.15 0.10 0.08 = 217.93 :=
by sorry

end carrie_shopping_l625_625954


namespace irene_total_income_l625_625726

noncomputable def irene_income (weekly_hours : ℕ) (base_pay : ℕ) (overtime_pay : ℕ) (hours_worked : ℕ) : ℕ :=
  base_pay + (if hours_worked > weekly_hours then (hours_worked - weekly_hours) * overtime_pay else 0)

theorem irene_total_income :
  irene_income 40 500 20 50 = 700 :=
by
  sorry

end irene_total_income_l625_625726


namespace anna_final_stamp_count_l625_625541

theorem anna_final_stamp_count (anna_initial : ℕ) (alison_initial : ℕ) (jeff_initial : ℕ)
  (anna_receive_from_alison : ℕ) (anna_give_jeff : ℕ) (anna_receive_jeff : ℕ) :
  anna_initial = 37 →
  alison_initial = 28 →
  jeff_initial = 31 →
  anna_receive_from_alison = alison_initial / 2 →
  anna_give_jeff = 2 →
  anna_receive_jeff = 1 →
  ∃ result : ℕ, result = 50 :=
by
  intros
  sorry

end anna_final_stamp_count_l625_625541


namespace necessary_but_not_sufficient_l625_625302

def simple_prop (p q : Prop) :=
  (¬ (p ∧ q)) → (¬ (p ∨ q))

theorem necessary_but_not_sufficient (p q : Prop) (h : simple_prop p q) :
  ((¬ (p ∧ q)) → (¬ (p ∨ q))) ∧ ¬ ((¬ (p ∨ q)) → (¬ (p ∧ q))) := by
sorry

end necessary_but_not_sufficient_l625_625302


namespace determine_a_l625_625654

noncomputable def f : ℝ → ℝ :=
λ x, if x > 2 then f (x - 5) else a * Real.exp x

theorem determine_a (a : ℝ) (H : f 2017 = Real.exp 2) : a = 1 :=
sorry

end determine_a_l625_625654


namespace range_combined_set_l625_625336

-- Define set X and set Y
def setX := {p : ℕ | prime p ∧ 10 ≤ p ∧ p < 1000}
def setY := {n : ℕ | 1 ≤ n ∧ 9 * n < 150}

-- Define the range function
def range (s : Set ℕ) : ℕ :=
  let min := s.toFinset.min' (by sorry) -- existence proof omitted
  let max := s.toFinset.max' (by sorry) -- existence proof omitted
  max - min

-- Prove that the range of the union of sets X and Y is 986
theorem range_combined_set : range (setX ∪ setY) = 986 :=
by
  -- proof omitted
  sorry

end range_combined_set_l625_625336


namespace nonnegative_sum_of_any_two_converse_nonnegative_sum_l625_625410

variable {n : ℕ}
variables  (a : fin n → ℝ) (x : fin n → ℝ)

theorem nonnegative_sum_of_any_two (hsum_nonneg : ∀ i j, a i + a j ≥ 0)
  (hnn : ∀ i, 0 ≤ x i) (hsum_x : ∑ i, x i = 1) :
  ∑ i, a i * x i ≥ ∑ i, a i * (x i)^2 :=
sorry

theorem converse_nonnegative_sum (hineq : ∑ i, a i * x i ≥ ∑ i, a i * (x i)^2) :
  a 0 + a 1 ≥ 0 :=
sorry

end nonnegative_sum_of_any_two_converse_nonnegative_sum_l625_625410


namespace prime_divides_sum_l625_625780

theorem prime_divides_sum 
  (a b c : ℕ) 
  (h1 : a^3 + 4 * b + c = a * b * c)
  (h2 : a ≥ c)
  (h3 : Prime (a^2 + 2 * a + 2)) : 
  (a^2 + 2 * a + 2) ∣ (a + 2 * b + 2) := 
sorry

end prime_divides_sum_l625_625780


namespace fraction_of_married_women_l625_625548

theorem fraction_of_married_women
  (total_employees : ℕ)
  (women_percentage : ℝ)
  (married_percentage : ℝ)
  (men_single_percentage : ℝ)
  (H_total_employees : total_employees = 100)
  (H_women_percentage : women_percentage = 0.58)
  (H_married_percentage : married_percentage = 0.60)
  (H_men_single_percentage : men_single_percentage = 2 / 3) :
  let total_women := (women_percentage * total_employees).to_nat,
      total_men := total_employees - total_women,
      married_men := (1 - men_single_percentage) * total_men,
      total_married_employees := (married_percentage * total_employees).to_nat,
      married_women := total_married_employees - married_men.to_nat,
      fraction_married_women := (married_women : ℝ) / total_women in
  fraction_married_women = 23 / 29 :=
  sorry

end fraction_of_married_women_l625_625548


namespace min_vertical_segment_length_l625_625368

def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) : ℝ := -x^2 - 4 * x - 3
def L (x : ℝ) : ℝ := f x - g x

theorem min_vertical_segment_length : ∃ (x : ℝ), L x = 10 :=
by
  sorry

end min_vertical_segment_length_l625_625368


namespace unbounded_n_satisfies_modified_triangle_property_l625_625517

theorem unbounded_n_satisfies_modified_triangle_property
  (n : ℕ) (T : finset ℕ) (hT : ∀ (y ∈ T), y ≥ 5 ∧ y ≤ n ∧ y ≠ 5 ∧ y ≠ 6 ∧ y ≠ 7)
  (ht : T.card = 10) :
  ∃ (a b c : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a^2 + b^2 = c^2 :=
begin
  sorry
end

end unbounded_n_satisfies_modified_triangle_property_l625_625517


namespace bridge_length_is_correct_l625_625088

-- Definitions of the conditions
def speed_km_per_hr : ℝ := 5
def time_minutes : ℝ := 15
def speed_m_per_min : ℝ := (speed_km_per_hr * 1000) / 60 -- converting 5 km/hr to meters per minute
def correct_distance : ℝ := 1249.95

-- The actual distance calculation based on the given time
def distance_covered : ℝ := speed_m_per_min * time_minutes

-- Statement that the distance covered is equal to the correct distance
theorem bridge_length_is_correct : distance_covered = correct_distance :=
by
  sorry

end bridge_length_is_correct_l625_625088


namespace check_interval_of_quadratic_l625_625164

theorem check_interval_of_quadratic (z : ℝ) : (z^2 - 40 * z + 344 ≤ 0) ↔ (20 - 2 * Real.sqrt 14 ≤ z ∧ z ≤ 20 + 2 * Real.sqrt 14) :=
sorry

end check_interval_of_quadratic_l625_625164


namespace student_attempt_total_l625_625718

theorem student_attempt_total
  (correct_answers : ℕ)
  (total_marks : ℕ)
  (marks_per_correct : ℕ)
  (marks_lost_per_wrong : ℕ)
  (correct_answered : ℕ)
  (marks_secured : ℕ)
  (total_attempted : ℕ) :
  marks_per_correct = 4 → 
  marks_lost_per_wrong = 1 → 
  marks_secured = 110 → 
  correct_answered = 34 → 
  total_marks = marks_secured → 
  total_marks = marks_per_correct * correct_answered - (total_attempted - correct_answered) → 
  total_attempted = correct_answered + (total_attempted - correct_answered) →
  total_attempted = 60 :=
by
  intros,
  sorry

end student_attempt_total_l625_625718


namespace sector_properties_l625_625016

noncomputable def radius : ℝ := 2
noncomputable def central_angle : ℝ := 2

theorem sector_properties :
  let arc_length := central_angle * radius,
      sector_area := 1 / 2 * central_angle * radius^2
  in arc_length = 4 ∧ sector_area = 4 := by
  sorry

end sector_properties_l625_625016


namespace points_form_convex_polygon_l625_625630

/-- 
  Given N points in the plane, if no three points are collinear, 
  and for any three points A, B, C, no point from the set lies 
  inside triangle ABC, then the points can be arranged to form 
  a convex polygon.
-/
theorem points_form_convex_polygon
  (N : ℕ)
  (points : Fin N → Point)
  (h_no_three_collinear : ∀ (i j k : Fin N), ¬Collinear (points i) (points j) (points k))
  (h_no_point_inside_triangle : 
    ∀ (i j k l : Fin N),  
    (points l) ∉ Triangle (points i) (points j) (points k)
  ) : 
  ∃ (labels : Fin N → Fin N), is_convex_polygon (labels ∘ points) :=
sorry -- proof is not required

end points_form_convex_polygon_l625_625630


namespace smallest_perfect_square_divisible_by_2_3_5_l625_625463

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, (n > 0) ∧ (nat.sqrt n * nat.sqrt n = n) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ n = 900 := sorry

end smallest_perfect_square_divisible_by_2_3_5_l625_625463


namespace angle_AEC_is_right_l625_625076

noncomputable def circle_radius : ℝ := 15
def center_triangle (C : Type) : Prop := ∀ A B : Type, triangle (equilateral A B C) ∧ Circumcircle E A B C
def circle_passes_through_vertices (circle_radius : ℝ) (C A B : Type) : Prop := 
  radius C A circle_radius ∧ radius C B circle_radius
def intersection_point (E B C : Type) : Prop :=
  lies_on_line E B C ∧ lies_on_circle E B circle_radius

theorem angle_AEC_is_right {C A B E : Type} 
  (h₁ : circle_radius = 15)
  (h₂ : center_triangle C)
  (h₃ : circle_passes_through_vertices 15 C A B)
  (h₄ : intersection_point E B C) :
  angle A E C = 90 :=
sorry

end angle_AEC_is_right_l625_625076


namespace arrangement_count_l625_625964

-- Define the parameters for the problem
def num_people : ℕ := 5
def num_exits : ℕ := 4
def exit1_people : ℕ := 2
def other_exits_count : ℕ := 3

-- Define the combinatorial functions that we need
def C (n k : ℕ) : ℕ := nat.choose n k
def A (n k : ℕ) : ℕ := nat.factorial n / nat.factorial (n - k)

-- Define the main theorem statement
theorem arrangement_count : (C num_people exit1_people) * (A other_exits_count other_exits_count) = 60 :=
by
  sorry

end arrangement_count_l625_625964


namespace truthful_dwarfs_count_l625_625590

def dwarf (n : ℕ) := n < 10
def vanilla_ice_cream (n : ℕ) := dwarf n ∧ (∀ m, dwarf m)
def chocolate_ice_cream (n : ℕ) := dwarf n ∧ m % 2 = 0
def fruit_ice_cream (n : ℕ) := dwarf n ∧ m % 9 = 0

theorem truthful_dwarfs_count :
  ∃ T L : ℕ, T + L = 10 ∧ T + 2 * L = 16 ∧ T = 4 :=
by
  sorry

end truthful_dwarfs_count_l625_625590


namespace stone_slab_length_l625_625069

theorem stone_slab_length (n : ℕ) (A : ℝ) (h_n : n = 30) (h_A : A = 50.7) :
  let slab_area := A / n in
  let slab_length := real.sqrt slab_area in
  slab_length * 100 = 130 :=
by
  sorry

end stone_slab_length_l625_625069


namespace perimeters_are_equal_l625_625402

-- Definitions/Conditions from the problem
def length_wire : Type := ℝ
def length_rectangle_wire (l_r : length_wire) : length_wire := l_r
def length_square_wire (l_s : length_wire) : length_wire := l_s

-- Given Conditions
variables (l_r l_s : length_wire)
hypothesis (h : l_r = l_s)

theorem perimeters_are_equal : length_rectangle_wire l_r = length_square_wire l_s :=
by {
  rw h,
  sorry
}

end perimeters_are_equal_l625_625402
