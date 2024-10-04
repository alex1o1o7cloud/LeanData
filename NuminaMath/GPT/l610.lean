import Mathlib

namespace females_wearing_glasses_l610_610482

theorem females_wearing_glasses (total_population : ℕ) 
                                (number_of_males : ℕ) 
                                (percent_females_with_glasses : ℚ) 
                                (male_population_condition : number_of_males = 2000)
                                (total_population_condition : total_population = 5000)
                                (percent_condition : percent_females_with_glasses = 0.30) :
                                total_population - number_of_males = 3000 ∧ 
                                ((total_population - number_of_males) * percent_females_with_glasses).natAbs = 900 := 
by 
  rw [total_population_condition, male_population_condition] 
  simp
  sorry

end females_wearing_glasses_l610_610482


namespace clock_cost_price_l610_610045

theorem clock_cost_price :
  (∃ C : ℝ, 
    let sp_gain_40 := 40 * C * 1.1,
        sp_gain_50 := 50 * C * 1.2,
        sp_uniform_90 := 90 * C * 1.15 in
    sp_gain_40 + sp_gain_50 - sp_uniform_90 = 40) →
  C = 80 :=
  by sorry

end clock_cost_price_l610_610045


namespace range_of_r_a_l610_610450

theorem range_of_r_a (a : ℝ) : set.range (λ x : ℝ, 1 / (a - x)^2) = set.Ioi 0 := 
sorry

end range_of_r_a_l610_610450


namespace diameter_of_circle_C_l610_610134

noncomputable def area_circle (r : ℝ) : ℝ := π * r ^ 2

theorem diameter_of_circle_C (d : ℝ) (r_C r_D : ℝ) (ratio : ℝ) 
  (h1 : r_D = 16) 
  (h2 : r_C = d / 2) 
  (h3 : ratio = 7) 
  (h4 : area_circle r_D = 256 * π)
  (h5 : area_circle r_D = area_circle r_C + ratio * area_circle r_C) :
  d = 8 * Real.sqrt 2 :=
  sorry

end diameter_of_circle_C_l610_610134


namespace number_of_people_l610_610287

-- Definitions based on the conditions
def condition1 (x : ℕ) : Prop := 5 * x + 45
def condition2 (x : ℕ) : Prop := 7 * x + 3

-- The theorem to prove the number of people in the group is 21
theorem number_of_people (x : ℕ) (h₁ : condition1 x = condition2 x) : x = 21 := by
  sorry

end number_of_people_l610_610287


namespace ratio_eq_neg_1009_l610_610914

theorem ratio_eq_neg_1009 (p q : ℝ) (h : (1 / p + 1 / q) / (1 / p - 1 / q) = 1009) : (p + q) / (p - q) = -1009 := 
by 
  sorry

end ratio_eq_neg_1009_l610_610914


namespace interval_of_increase_of_even_function_l610_610682

def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 6
noncomputable def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

theorem interval_of_increase_of_even_function :
  (∀ x, f x a = x^2 + 6) → (is_even (λ x, f x a)) → (∀ x : ℝ, x ∈ set.Ici 0 -> deriv (λ x, f x a) x ≥ 0)
:= begin
  sorry
end

end interval_of_increase_of_even_function_l610_610682


namespace number_of_handshakes_l610_610539

-- Definitions based on the conditions:
def number_of_teams : ℕ := 4
def number_of_women_per_team : ℕ := 2
def total_women : ℕ := number_of_teams * number_of_women_per_team

-- Each woman shakes hands with all others except her partner
def handshakes_per_woman : ℕ := total_women - 1 - (number_of_women_per_team - 1)

-- Calculate total handshakes, considering each handshake is counted twice
def total_handshakes : ℕ := (total_women * handshakes_per_woman) / 2

-- Statement to prove
theorem number_of_handshakes :
  total_handshakes = 24 := 
sorry

end number_of_handshakes_l610_610539


namespace smallest_b_value_is_6_l610_610316

noncomputable def smallest_b_value (a b c : ℝ) (h_arith : a + c = 2 * b) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 216) : ℝ :=
b

theorem smallest_b_value_is_6 (a b c : ℝ) (h_arith : a + c = 2 * b) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 216) : 
  smallest_b_value a b c h_arith h_pos h_prod = 6 :=
sorry

end smallest_b_value_is_6_l610_610316


namespace more_likely_same_sector_l610_610109

theorem more_likely_same_sector 
  (p : ℕ → ℝ) 
  (n : ℕ) 
  (hprob_sum_one : ∑ i in Finset.range n, p i = 1) 
  (hprob_nonneg : ∀ i, 0 ≤ p i) : 
  ∑ i in Finset.range n, (p i) ^ 2 
  > ∑ i in Finset.range n, (p i) * (p ((i + 1) % n)) :=
by
  sorry

end more_likely_same_sector_l610_610109


namespace max_positive_numbers_l610_610863

theorem max_positive_numbers (nums : Fin 30 → ℝ) (h_sum_zero : (∑ i, nums i) = 0) : ∃ (m : ℕ), m ≤ 29 ∧ m = Finset.card { i : Fin 30 | nums i > 0 } :=
by
  sorry

end max_positive_numbers_l610_610863


namespace m_divides_product_iff_composite_ne_4_l610_610185

theorem m_divides_product_iff_composite_ne_4 (m : ℕ) : 
  (m ∣ Nat.factorial (m - 1)) ↔ 
  (∃ a b : ℕ, a ≠ b ∧ 1 < a ∧ 1 < b ∧ m = a * b ∧ m ≠ 4) := 
sorry

end m_divides_product_iff_composite_ne_4_l610_610185


namespace find_a_l610_610990

-- Given function f and conditions
def f (a x : ℝ) : ℝ := 1 + real.log x / real.log a

-- Conditions
variables {a : ℝ} {x : ℝ}

-- Hypotheses
def conditions := 
  a > 0 ∧ a ≠ 1 ∧ (∃ (f_inv : ℝ → ℝ), f_inv 3 = 4 ∧ (∀ y, f_inv y = x ↔ f y = y))

-- Statement to prove
theorem find_a (h : conditions) : a = 2 :=
sorry

end find_a_l610_610990


namespace chord_slope_through_point_P_l610_610986

noncomputable theory

def point : Type := ℝ × ℝ

def ellipse (x y : ℝ) : Prop := (x^2 / 36) + (y^2 / 9) = 1

def midpoint (A B P : point) : Prop := 
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

def slope (A B : point) : ℝ := (B.2 - A.2) / (B.1 - A.1)

theorem chord_slope_through_point_P
  (A B : point)
  (H_A : ellipse A.1 A.2)
  (H_B : ellipse B.1 B.2)
  (P : point)
  (H_P : P = (4, 2))
  (H_midpoint : midpoint A B P) :
  slope A B = -1 / 2 := 
sorry

end chord_slope_through_point_P_l610_610986


namespace desired_percentage_total_annual_income_l610_610910

variable (investment1 : ℝ)
variable (investment2 : ℝ)
variable (rate1 : ℝ)
variable (rate2 : ℝ)

theorem desired_percentage_total_annual_income (h1 : investment1 = 2000)
  (h2 : rate1 = 0.05)
  (h3 : investment2 = 1000-1e-13)
  (h4 : rate2 = 0.08):
  ((investment1 * rate1 + investment2 * rate2) / (investment1 + investment2) * 100) = 6 := by
  sorry

end desired_percentage_total_annual_income_l610_610910


namespace sin_sum_identity_l610_610273

noncomputable def triangle_side_lengths (a b c : ℝ) : Prop :=
  ∃ A B C : ℝ, (sinRule : (a / Real.sin A) = (b / Real.sin B)) ∧ ((a / Real.sin A) = (c / Real.sin C))

theorem sin_sum_identity (a b c : ℝ) (h : triangle_side_lengths a b c) :
  a^2 * Real.sin (2 * B) + b^2 * Real.sin (2 * A) = 2 * a * b * Real.sin C :=
sorry

end sin_sum_identity_l610_610273


namespace friends_raise_funds_l610_610077

theorem friends_raise_funds (total_amount friends_count min_amount amount_per_person: ℕ)
  (h1 : total_amount = 3000)
  (h2 : friends_count = 10)
  (h3 : min_amount = 300)
  (h4 : amount_per_person = total_amount / friends_count) :
  amount_per_person = min_amount :=
by
  sorry

end friends_raise_funds_l610_610077


namespace city_blocks_proof_distance_A_C_proof_distance_B_D_proof_min_coins_proof_max_coins_proof_l610_610356

def city_block_count : ℕ := (9 - 1) * (15 - 1)

def manhattan_distance_A_C : ℕ := 100 * (|5 - 2| + |12 - 3| - 2)

def manhattan_distance_B_D : ℕ := 100 * (|5 - 2| + |12 - 3| + 2)

def min_coins : ℕ := manhattan_distance_A_C / 100

def max_coins : ℕ := manhattan_distance_B_D / 100

theorem city_blocks_proof : city_block_count = 112 := by
  sorry

theorem distance_A_C_proof : manhattan_distance_A_C = 1000 := by
  sorry

theorem distance_B_D_proof : manhattan_distance_B_D = 1400 := by
  sorry

theorem min_coins_proof : min_coins = 10 := by
  sorry

theorem max_coins_proof : max_coins = 14 := by
  sorry

end city_blocks_proof_distance_A_C_proof_distance_B_D_proof_min_coins_proof_max_coins_proof_l610_610356


namespace A_B_finish_l610_610464

theorem A_B_finish (A B C : ℕ → ℝ) (h1 : A + B + C = 1 / 6) (h2 : C = 1 / 10) :
  1 / (A + B) = 15 :=
by
  sorry

end A_B_finish_l610_610464


namespace hyperbola_foci_y_axis_l610_610229

theorem hyperbola_foci_y_axis (a b : ℝ) (h : ∀ x y : ℝ, a * x^2 + b * y^2 = 1 → (1/a < 0 ∧ 1/b > 0)) : a < 0 ∧ b > 0 :=
by
  sorry

end hyperbola_foci_y_axis_l610_610229


namespace maximum_expression_value_l610_610623

-- Definitions for conditions
variables (n : ℕ) (x : Fin n → ℝ) (h_n : 1 < n) (h_sum : ∑ i, (x i) ^ 2 = 1)

-- Lean theorem statement
theorem maximum_expression_value :
  (∑ k in Finset.range n, (k + 1) * (x ⟨k, universe h_n⟩) ^ 2 +
   ∑ i in Finset.range n, ∑ j in Finset.Ico (i + 1) n, (i + 1 + (j + 1)) * (x ⟨i, universe h_n⟩) * (x ⟨j, universe h_n⟩))
  ≤ ( (n * (n + 1) / 4 : ℝ) + (n / 2) * real.sqrt ((n + 1) * (2 * n + 1) / 6) ) := 
sorry

end maximum_expression_value_l610_610623


namespace mark_total_flowers_l610_610352

theorem mark_total_flowers (yellow purple green total : ℕ) 
  (hyellow : yellow = 10)
  (hpurple : purple = yellow + (yellow * 80 / 100))
  (hgreen : green = (yellow + purple) * 25 / 100)
  (htotal : total = yellow + purple + green) : 
  total = 35 :=
by
  sorry

end mark_total_flowers_l610_610352


namespace bunyakovsky_hit_same_sector_l610_610117

variable {n : ℕ} (p : Fin n → ℝ)

theorem bunyakovsky_hit_same_sector (h_sum : ∑ i in Finset.univ, p i = 1) :
  (∑ i in Finset.univ, (p i)^2) >
  (∑ i in Finset.univ, (p i) * (p (Fin.rotate 1 i))) := 
sorry

end bunyakovsky_hit_same_sector_l610_610117


namespace slant_asymptote_sum_l610_610171
   
   theorem slant_asymptote_sum (x : ℝ) :
     let y := 3 * x ^ 2 + 5 * x - 11,
         a := x - 4,
         quotient := y / a,
         m := 3,
         b := 17
     in (m + b) = 20 :=
   by {
     sorry
   }
   
end slant_asymptote_sum_l610_610171


namespace tan_theta_eq_l610_610612

theorem tan_theta_eq :
  (∃ θ : ℝ, sin θ = 1 / 3 ∧ θ ∈ Ioo (π / 2) π) →
  (tan θ = - (Real.sqrt 2) / 4) :=
by
  sorry

end tan_theta_eq_l610_610612


namespace seeds_per_ear_l610_610492

theorem seeds_per_ear (rows_ears : ℕ) (bag_seeds : ℕ) (pay_per_row : ℕ) (dinner_cost : ℕ) (bags_used : ℕ) (ears_per_row : ℕ) (seeds_per_bag : ℕ) (money_spent_on_dinner : ℕ) :
  (rows_ears = 70) → (bag_seeds = 48) → (pay_per_row = 15 / 10) → (dinner_cost = 36) → (bags_used = 140) →
  money_spent_on_dinner = 36 →
  let total_ears := (72 / (15 / 10)) * rows_ears in -- Each kid planted 48 rows, rows_ears = 70
  let total_seeds := bags_used * bag_seeds in -- Each kid used 140 bags, bag_seeds = 48
  total_seeds / total_ears = 2 :=
begin
  intros h1 h2 h3 h4 h5 h6,
  have total_ears : ℕ := (72 / (15 / 10)) * rows_ears,
  have total_seeds : ℕ := bags_used * bag_seeds,
  have seeds_per_ear : ℕ := total_seeds / total_ears,
  exact seeds_per_ear = 2
end

end seeds_per_ear_l610_610492


namespace second_smallest_prime_perimeter_l610_610141

open Nat

-- Define the conditions
def is_prime := prime
def distinct (a b c : Nat) := a ≠ b ∧ b ≠ c ∧ a ≠ c
def scalene_triangle (a b c : Nat) := a + b > c ∧ a + c > b ∧ b + c > a
def prime_perimeter (a b c : Nat) := is_prime (a + b + c)

-- Define the main proof statement
theorem second_smallest_prime_perimeter :
  ∃ (a b c : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧ distinct a b c ∧ scalene_triangle a b c ∧ prime_perimeter a b c ∧ (a + b + c = 29) :=
sorry

end second_smallest_prime_perimeter_l610_610141


namespace school_dance_boys_count_l610_610425

theorem school_dance_boys_count
  (total_attendees : ℕ)
  (percent_faculty_staff : ℝ)
  (fraction_girls : ℝ)
  (h1 : total_attendees = 100)
  (h2 : percent_faculty_staff = 0.1)
  (h3 : fraction_girls = 2/3) :
  let faculty_staff := total_attendees * percent_faculty_staff in
  let students := total_attendees - faculty_staff in
  let girls := students * fraction_girls in
  let boys := students - girls in
  boys = 30 :=
by
  -- Skipping the proof
  sorry

end school_dance_boys_count_l610_610425


namespace twenty_odd_rows_fifteen_odd_cols_impossible_sixteen_by_sixteen_with_126_crosses_possible_l610_610520

-- Conditions and helper definitions
def is_odd (n: ℕ) := n % 2 = 1
def count_odd_rows (table: ℕ × ℕ → bool) (n: ℕ) := 
  (List.range n).countp (λ i, is_odd ((List.range n).countp (λ j, table (i, j))))
  
def count_odd_columns (table: ℕ × ℕ → bool) (n: ℕ) := 
  (List.range n).countp (λ j, is_odd ((List.range n).countp (λ i, table (i, j))))

-- a) Proof problem statement
theorem twenty_odd_rows_fifteen_odd_cols_impossible (n: ℕ): 
  ∀ (table: ℕ × ℕ → bool), count_odd_rows table n = 20 → count_odd_columns table n = 15 → False := 
begin
  intros table h_rows h_cols,
  sorry
end

-- b) Proof problem statement
theorem sixteen_by_sixteen_with_126_crosses_possible :
  ∃ (table: ℕ × ℕ → bool), count_odd_rows table 16 = 16 ∧ count_odd_columns table 16 = 16 ∧ 
  (List.range 16).sum (λ i, (List.range 16).countp (λ j, table (i, j))) = 126 :=
begin
  sorry
end

end twenty_odd_rows_fifteen_odd_cols_impossible_sixteen_by_sixteen_with_126_crosses_possible_l610_610520


namespace find_m_range_l610_610225

variable {f : ℝ → ℝ}

-- Conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a ≤ b → ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

-- Theorem statement
theorem find_m_range (h_odd : is_odd_function f) (h_monotone : is_monotonically_decreasing_on f (-2) 2) :
  ∀ m : ℝ, (f (1 - m) + f (1 - m^2) < 0) → m ∈ Ioo (-1 : ℝ) (1 : ℝ) :=
by sorry

end find_m_range_l610_610225


namespace part_a_part_b_l610_610509

-- Definitions based on the problem conditions
def is_square_table (n : ℕ) (table : List (List Bool)) : Prop := 
  table.length = n ∧ ∀ row, row ∈ table → row.length = n

def is_odd_row (row : List Bool) : Prop := 
  row.count (λ x => x = true) % 2 = 1

def is_odd_column (n : ℕ) (table : List (List Bool)) (c : ℕ) : Prop :=
  n > 0 ∧ c < n ∧ (List.count (λ row => row.get! c = true) table) % 2 = 1

-- Part (a) statement: Prove it's impossible to have exactly 20 odd rows and 15 odd columns
theorem part_a (n : ℕ) (table : List (List Bool)) :
  n = 16 → is_square_table n table → 
  (List.count is_odd_row table) = 20 → 
  ((List.range n).count (is_odd_column n table)) = 15 → 
  False := 
sorry

-- Part (b) statement: Prove that it's possible to arrange 126 crosses in a 16x16 table with all odd rows and columns
theorem part_b (table : List (List Bool)) :
  is_square_table 16 table →
  ((table.map (λ row => row.count (λ x => x = true))).sum = 126) →
  (∀ row, row ∈ table → is_odd_row row) →
  (∀ c, c < 16 → is_odd_column 16 table c) →
  True :=
sorry

end part_a_part_b_l610_610509


namespace females_wearing_glasses_in_town_l610_610479

theorem females_wearing_glasses_in_town : 
  (total_population females males : ℕ) 
  (wear_glasses_percentage : ℚ) 
  (h1 : total_population = 5000) 
  (h2 : males = 2000) 
  (h3 : wear_glasses_percentage = 0.30) 
  : 
  ∃ (females_wear_glasses : ℕ), females_wear_glasses = 900 :=
by
  have total_females : ℕ := total_population - males
  have females_wear_glasses : ℚ := total_females * wear_glasses_percentage
  use females_wear_glasses.toNat
  sorry

end females_wearing_glasses_in_town_l610_610479


namespace parallel_line_plane_D_l610_610681

def b_vec_D : Vector3 ℝ := ⟨1, -1, 3⟩
def n_vec_D : Vector3 ℝ := ⟨0, 3, 1⟩

theorem parallel_line_plane_D (b : Vector3 ℝ) (n : Vector3 ℝ) :
  b = b_vec_D → n = n_vec_D → (b.dot n = 0) :=
by
  intro hb hn
  rw [hb, hn]
  sorry

end parallel_line_plane_D_l610_610681


namespace area_IJKL_l610_610796

noncomputable def square_area (side_length : ℝ) : ℝ := side_length * side_length

variables (W X Y Z I J K L : Type) (d_WI : ℝ)

axiom square_WXYZ (hWXYZ : ∀ {A B C D : Type}, W X Y Z ≠ A ∧ A = B ∧ B = C ∧ C = D ∧ side_length = 10)
axiom square_IJKL (hIJKL : ∀ {A B C D : Type}, I J K L ≠ A ∧ A = B ∧ B = C ∧ C = D)
axiom side_length_WXYZ : d_WI = 2

theorem area_IJKL : ∃ (x : ℝ), square_area x = 98 := by
  sorry

end area_IJKL_l610_610796


namespace amount_C_invested_l610_610902

theorem amount_C_invested (A B C_profit Total_profit : ℕ) (hA : A = 27000) (hB : B = 72000)
    (hC_profit : C_profit = 36000) (hTotal_profit : Total_profit = 80000) :
    ∃ C : ℕ, C = 81000 :=
by
  -- Definitions based on conditions
  let C : ℕ := (C_profit * (A + B)) / (Total_profit - C_profit)
  -- Proof that C equals 81000
  have hC : C = 81000 := 
  sorry
  use C
  exact hC

end amount_C_invested_l610_610902


namespace dart_more_likely_l610_610103

noncomputable def dart_probabilities (p : ℕ → ℝ) (n : ℕ) : Prop :=
  (∑ i in Finset.range n, p i = 1) →
  (∑ i in Finset.range n, p i ^ 2 >
   ∑ i in Finset.range n, p i * p ((i + 1) % n))

-- Example theorem statement
theorem dart_more_likely (p : ℕ → ℝ) (n : ℕ) (h : ∑ i in Finset.range n, p i = 1) :
  (∑ i in Finset.range n, p i ^ 2 >
   ∑ i in Finset.range n, p i * p ((i + 1) % n)) :=
begin
  sorry -- Proof not required as per instructions
end

end dart_more_likely_l610_610103


namespace find_m_symmetry_l610_610222

theorem find_m_symmetry (A B : ℝ × ℝ) (m : ℝ)
  (hA : A = (-3, m)) (hB : B = (3, 4)) (hy : A.2 = B.2) : m = 4 :=
sorry

end find_m_symmetry_l610_610222


namespace ratio_of_segments_of_hypotenuse_l610_610279

theorem ratio_of_segments_of_hypotenuse (x : ℝ) : 
  let AC := 2 * x,
      BC := 3 * x,
      AB := x * Real.sqrt 13,
      CD := Real.sqrt ((2 * x) * (3 * x)) / (2 * x).sqrt := -- Using similar triangles and geometric mean
  ratio_of_segments := (CD / (2 * x)) in
  ratio_of_segments = Real.sqrt 6 / 3 :=
sorry

end ratio_of_segments_of_hypotenuse_l610_610279


namespace julieta_total_cost_l610_610307

variable (initial_backpack_price : ℕ)
variable (initial_binder_price : ℕ)
variable (backpack_price_increase : ℕ)
variable (binder_price_reduction : ℕ)
variable (discount_rate : ℕ)
variable (num_binders : ℕ)

def calculate_total_cost (initial_backpack_price initial_binder_price backpack_price_increase binder_price_reduction discount_rate num_binders : ℕ) : ℝ :=
  let new_backpack_price := initial_backpack_price + backpack_price_increase
  let new_binder_price := initial_binder_price - binder_price_reduction
  let total_bindable_cost := min num_binders ((num_binders + 1) / 2 * new_binder_price)
  let total_pre_discount := new_backpack_price + total_bindable_cost
  let discount_amount := total_pre_discount * discount_rate / 100
  let total_price := total_pre_discount - discount_amount
  total_price

theorem julieta_total_cost
  (initial_backpack_price : ℕ)
  (initial_binder_price : ℕ)
  (backpack_price_increase : ℕ)
  (binder_price_reduction : ℕ)
  (discount_rate : ℕ)
  (num_binders : ℕ)
  (h_initial_backpack : initial_backpack_price = 50)
  (h_initial_binder : initial_binder_price = 20)
  (h_backpack_inc : backpack_price_increase = 5)
  (h_binder_red : binder_price_reduction = 2)
  (h_discount : discount_rate = 10)
  (h_num_binders : num_binders = 3) :
  calculate_total_cost initial_backpack_price initial_binder_price backpack_price_increase binder_price_reduction discount_rate num_binders = 81.90 :=
by
  sorry

end julieta_total_cost_l610_610307


namespace minimum_a2_plus_b2_l610_610998

theorem minimum_a2_plus_b2 (a b : ℝ) 
  (h : ∀ x : ℝ, x^2 - 2*a*x + a^2 - a*b + 4 ≤ 0 → (∃ ! x0, x^2 - 2*a*x + a^2 - a*b + 4 = 0)) :
  a^2 + b^2 = 8 :=
by
  sorry

end minimum_a2_plus_b2_l610_610998


namespace pizza_sales_calculation_l610_610496

def pizzas_sold_in_spring (total_sales : ℝ) (summer_sales : ℝ) (fall_percentage : ℝ) (winter_percentage : ℝ) : ℝ :=
  total_sales - summer_sales - (fall_percentage * total_sales) - (winter_percentage * total_sales)

theorem pizza_sales_calculation :
  let summer_sales := 5;
  let fall_percentage := 0.1;
  let winter_percentage := 0.2;
  ∃ (total_sales : ℝ), 0.4 * total_sales = summer_sales ∧
    pizzas_sold_in_spring total_sales summer_sales fall_percentage winter_percentage = 3.75 :=
by
  sorry

end pizza_sales_calculation_l610_610496


namespace number_of_true_statements_l610_610530

def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def has_inverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, ∀ x : ℝ, g (f x) = x ∧ f (g x) = x

theorem number_of_true_statements :
  let f : ℝ → ℝ,
      g : ℝ → ℝ in
      (∀ C : ℝ, (f = (λ x, C)) → (is_even f ∧ (C = 0 → is_odd f))) ∧
      (is_odd f → ¬ has_inverse f) ∧
      (is_odd f → is_odd (λ x, real.sin (f x))) ∧
      (is_odd f ∧ is_even g → is_even (λ x, g (f x))) →
  (1 + 0 + 1 + 1 = 3) :=
by
  sorry

end number_of_true_statements_l610_610530


namespace minimal_sum_at_P4_l610_610212

-- Definition of the problem statements in Lean 4

def P1 : ℝ := sorry
def P2 : ℝ := sorry
def P3 : ℝ := sorry
def P4 : ℝ := sorry
def P5 : ℝ := sorry
def P6 : ℝ := sorry
def P7 : ℝ := sorry

-- Assume the points are ordered such that P1 < P2 < P3 < P4 < P5 < P6 < P7
axiom points_in_order : P1 < P2 ∧ P2 < P3 ∧ P3 < P4 ∧ P4 < P5 ∧ P5 < P6 ∧ P6 < P7

-- Define distance sum function s
def s (P : ℝ) : ℝ := 
  abs (P - P1) + abs (P - P2) + abs (P - P3) + abs (P - P4) + abs (P - P5) + abs (P - P6) + abs (P - P7)

-- Main theorem to be proven
theorem minimal_sum_at_P4 : ∀ P : ℝ, ∃ P, P = P4 ∧ ∀ P_point, s P ≥ s P4 := sorry

end minimal_sum_at_P4_l610_610212


namespace average_greater_than_median_by_22_l610_610666

/-- Define the weights of the siblings -/
def hammie_weight : ℕ := 120
def triplet1_weight : ℕ := 4
def triplet2_weight : ℕ := 4
def triplet3_weight : ℕ := 7
def brother_weight : ℕ := 10

/-- Define the list of weights -/
def weights : List ℕ := [hammie_weight, triplet1_weight, triplet2_weight, triplet3_weight, brother_weight]

/-- Define the median and average weight -/
def median_weight : ℕ := 7
def average_weight : ℕ := 29

theorem average_greater_than_median_by_22 : average_weight - median_weight = 22 := by
  sorry

end average_greater_than_median_by_22_l610_610666


namespace unique_solution_l610_610577

theorem unique_solution (x y z n : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hn : 2 ≤ n) (h_y_bound : y ≤ 5 * 2^(2*n)) :
  x^(2*n+1) - y^(2*n+1) = x * y * z + 2^(2*n+1) → (x, y, z, n) = (3, 1, 70, 2) :=
by
  sorry

end unique_solution_l610_610577


namespace number_of_people_in_group_l610_610290

-- Defining the conditions as constants and assumptions
variables (x : ℕ) -- x represents the number of people in the group
constant total_cost_sheep : ℕ -- total cost of the sheep

-- Assuming the given conditions
axiom h1 : 5 * x + 45 = total_cost_sheep
axiom h2 : 7 * x + 3 = total_cost_sheep

-- The theorem to prove that x equals 21.
theorem number_of_people_in_group : x = 21 :=
by
  -- Below, we would normally proceed to prove the statement.
  sorry

end number_of_people_in_group_l610_610290


namespace quadratic_function_statement_correct_l610_610457

theorem quadratic_function_statement_correct {x : ℝ} (h : x < 0) : 
  ∀ x, (2 * x ^ 2) ≥ 0 :=
begin
  sorry
end

end quadratic_function_statement_correct_l610_610457


namespace min_points_each_player_scored_l610_610484

theorem min_points_each_player_scored {n : ℕ} (total_players : ℕ) (total_points : ℕ) 
(max_points : ℕ) (player_points : Fin total_players → ℕ) 
(h_total_players : total_players = 12) (h_total_points : total_points = 100) 
(h_max_points : max_points = 23) (h_sum_points : (∑ x : Fin total_players, player_points x) = total_points) 
(h_player_points_lb : ∀ x, player_points x ≥ n) (h_player_points_ub : ∃ x, player_points x = max_points) :
  n = 7 :=
by
  sorry

end min_points_each_player_scored_l610_610484


namespace triangle_perimeter_minimum_l610_610693

theorem triangle_perimeter_minimum :
  ∃ (BC : ℝ), ∀ (A B C : ℝ)
  (angle_ACB : A * B * C = 60)
  (BC_gt_one : BC > 1)
  (AC_eq_AB_plus_half : A = B + 1/2),
  (BC = 1 + real.sqrt(2) / 2) :=
by
  sorry

end triangle_perimeter_minimum_l610_610693


namespace train_crossing_time_approx_l610_610047

def length_of_train : ℝ := 110 -- meters
def speed_of_train_kmph : ℝ := 60 -- kmph
def length_of_bridge : ℝ := 340 -- meters

noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * (1000 / 3600)

noncomputable def total_distance : ℝ := length_of_train + length_of_bridge

theorem train_crossing_time_approx :
  total_distance / speed_of_train_mps ≈ 27 :=
by
  sorry -- proof omitted

end train_crossing_time_approx_l610_610047


namespace determine_conjugate_l610_610203

noncomputable def complex_number (z : ℂ) : Prop :=
  (z / (z - complex.i)) = complex.i 

theorem determine_conjugate {z : ℂ} (h : complex_number z) :
  conj z = (1 / 2) - (1 / 2) * complex.i :=
by
  sorry

end determine_conjugate_l610_610203


namespace locus_of_D_l610_610967

noncomputable def reflection (A B : Point) : Point :=
  2 * A - B

theorem locus_of_D (A B : Point) (e : ℝ) : 
  ∀ (D : Point), 
  (∃ C : Point, (∥A - C∥ = e ∧ ∥B - D∥ = ∥A - B∥ ∧ C + D = A + B)) →
  (∥D - reflection A B∥ = e) :=
by
  sorry

end locus_of_D_l610_610967


namespace number_of_sections_is_five_l610_610904

-- Define the variables as given in the problem statements
def total_area : ℝ := 300
def section_area : ℝ := 60

-- Define the number of sections calculated as the total area divided by the area of each section
def num_sections : ℝ := total_area / section_area

-- The theorem statement proving the number of sections is 5
theorem number_of_sections_is_five : num_sections = 5 :=
by
  -- The proof is omitted
  sorry

end number_of_sections_is_five_l610_610904


namespace correct_sum_is_132_l610_610291

-- Let's define the conditions:
-- The ones digit B is mistakenly taken as 1 (when it should be 7)
-- The tens digit C is mistakenly taken as 6 (when it should be 4)
-- The incorrect sum is 146

def correct_ones_digit (mistaken_ones_digit : Nat) : Nat :=
  -- B was mistaken for 1, so B should be 7
  if mistaken_ones_digit = 1 then 7 else mistaken_ones_digit

def correct_tens_digit (mistaken_tens_digit : Nat) : Nat :=
  -- C was mistaken for 6, so C should be 4
  if mistaken_tens_digit = 6 then 4 else mistaken_tens_digit

def correct_sum (incorrect_sum : Nat) : Nat :=
  -- Correcting the sum based on the mistakes
  incorrect_sum + 6 - 20 -- 6 to correct ones mistake, minus 20 to correct tens mistake

theorem correct_sum_is_132 : correct_sum 146 = 132 :=
  by
    -- The theorem is here to check that the corrected sum equals 132
    sorry

end correct_sum_is_132_l610_610291


namespace probability_product_even_gt_one_fourth_l610_610431

def n := 100
def is_even (x : ℕ) : Prop := x % 2 = 0
def is_odd (x : ℕ) : Prop := ¬ is_even x

theorem probability_product_even_gt_one_fourth :
  (∃ (p : ℝ), p > 0 ∧ p = 1 - (50 * 49 * 48 : ℝ) / (100 * 99 * 98) ∧ p > 1 / 4) :=
sorry

end probability_product_even_gt_one_fourth_l610_610431


namespace smallest_n_to_reach_distance_l610_610736

-- Definitions
def A (n : ℕ) : ℝ × ℝ := (n, 0)  -- Points on the x-axis for simplicity
def B (n : ℕ) : ℝ × ℝ := (n, n^2) -- Points on the graph y = x^2

def equilateral_triangle (A₁ A₂ B : ℝ × ℝ) : Prop :=
  let dist := λ P Q : ℝ × ℝ, real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  dist A₁ B = dist A₂ B ∧ dist A₁ B = dist A₁ A₂

def a (n : ℕ) : ℝ := 2 / 3 * n -- Distance increment

def total_distance (n : ℕ) : ℝ :=
  finset.sum (finset.range (n + 1)) a

-- Theorem statement
theorem smallest_n_to_reach_distance : ∃ (n : ℕ), total_distance n ≥ 50 ∧ ∀ m < n, total_distance m < 50 :=
by
  sorry


end smallest_n_to_reach_distance_l610_610736


namespace radius_of_circle_through_points_l610_610816

theorem radius_of_circle_through_points : 
  ∃ (x : ℝ), 
  (dist (x, 0) (2, 5) = dist (x, 0) (3, 4)) →
  (∃ (r : ℝ), r = dist (x, 0) (2, 5) ∧ r = 5) :=
by
  sorry

end radius_of_circle_through_points_l610_610816


namespace prod_inequality_l610_610600

open BigOperators

theorem prod_inequality (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, 0 < x i) :
    ∏ i in Finset.range n, (1 + ∑ j in Finset.range i.succ, x ⟨j, Nat.lt_succ_iff.mpr (Finset.mem_range_succ_iff.mpr j.zero_lt_succ)⟩) 
    ≥ Real.sqrt ((n + 1)^(n + 1)) * Real.sqrt (∏ i in Finset.range n, x i) := sorry

end prod_inequality_l610_610600


namespace proof_total_distance_l610_610719

-- Define the total distance
def total_distance (D : ℕ) :=
  let by_foot := (1 : ℚ) / 6
  let by_bicycle := (1 : ℚ) / 4
  let by_bus := (1 : ℚ) / 3
  let by_car := 10
  let by_train := (1 : ℚ) / 12
  D - (by_foot + by_bicycle + by_bus + by_train) * D = by_car

-- Given proof problem
theorem proof_total_distance : ∃ D : ℕ, total_distance D ∧ D = 60 :=
sorry

end proof_total_distance_l610_610719


namespace original_length_before_final_cut_l610_610439

-- Defining the initial length of the board
def initial_length : ℕ := 143

-- Defining the length after the first cut
def length_after_first_cut : ℕ := initial_length - 25

-- Defining the length after the final cut
def length_after_final_cut : ℕ := length_after_first_cut - 7

-- Stating the theorem to prove that the original length of the board before cutting the final 7 cm is 125 cm
theorem original_length_before_final_cut : initial_length - 25 + 7 = 125 :=
sorry

end original_length_before_final_cut_l610_610439


namespace evaluate_expression_l610_610258

theorem evaluate_expression : ∀ (x y : ℕ), x = 3 → y = 2 → 3 * x - 4 * y + 2 = 3 :=
by
  intros x y hx hy
  rw [hx, hy]
  calc
    3 * 3 - 4 * 2 + 2 = 9 - 8 + 2 : by simp [mul_sub]
    ... = 1 + 2 : by simp [nat.sub_add]
    ... = 3 : by simp

end evaluate_expression_l610_610258


namespace gcd_105_490_l610_610163

theorem gcd_105_490 : Nat.gcd 105 490 = 35 := by
sorry

end gcd_105_490_l610_610163


namespace sourav_srinath_same_fruit_days_l610_610357

def sourav_same_fruit_days (m n : ℕ) (hmn_gcd : Nat.gcd m n = 1) : ℕ :=
  (m * n + 1) / 2

theorem sourav_srinath_same_fruit_days (m n : ℕ) (hmn_gcd : Nat.gcd m n = 1) :
  let total_days := m * n
  let same_fruit_days := sourav_same_fruit_days m n hmn_gcd
  same_fruit_days = (total_days + 1) / 2 :=
by
  sorry

end sourav_srinath_same_fruit_days_l610_610357


namespace light_nanosecond_distance_l610_610081

theorem light_nanosecond_distance :
  let c := 3 * 10^8 -- speed of light in meters per second
  let t := 1 / 10^9 -- one nanosecond in seconds
  let d := c * t -- distance traveled in one nanosecond in meters
  in d * 100 = 30 :=  -- converting meters to centimeters
by
  sorry

end light_nanosecond_distance_l610_610081


namespace range_of_a_l610_610952

-- Define the conditions in Lean
variable {a : ℝ} (hx : a > 0) (hx_ne_1 : a ≠ 1)
def f (x : ℝ) : ℝ := log a ((3 - a) * x - a) 

theorem range_of_a (a_pos : 0 < a) (a_ne_1 : a ≠ 1) :
  (∃ (x : ℝ), (3 - a) * x - a > 0) ∧ a < 3 -> 1 < a ∧ a < 3 :=
by
  -- Prove here
  sorry

end range_of_a_l610_610952


namespace probability_is_7_over_18_l610_610441

notation "ℝ" => Real

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

def possible_outcomes : ℕ := 36

def valid_combinations : List (ℕ × ℕ) :=
  [(1, 5), (5, 1), (2, 5), (5, 2), (3, 5), (5, 3), (4, 5), (5, 4), 
   (5, 5), (6, 5), (5, 6), (3, 3), (4, 4), (6, 6)]

def number_of_valid_combinations : ℕ := valid_combinations.length

def probability_of_isosceles : ℝ :=
  number_of_valid_combinations / possible_outcomes

theorem probability_is_7_over_18 :
  probability_of_isosceles = 7 / 18 := by
  sorry

end probability_is_7_over_18_l610_610441


namespace range_of_m_exist_real_root_l610_610973

-- Define the quadratic equation with roots x1 and x2
def quadratic_eq (m : ℝ) : Polynomial ℝ :=
  Polynomial.X^2 - 2*(m + 1)*Polynomial.X + (m^2 + 3)

-- Define the statement that x1 and x2 are roots of the quadratic equation
def are_roots (m x1 x2 : ℝ) : Prop :=
  quadratic_eq m.eval x1 = 0 ∧ quadratic_eq m.eval x2 = 0

-- Statement for the range of m
theorem range_of_m (m : ℝ) : are_roots m x1 x2 → (m ≥ 1) :=
sorry

-- Statement that (x1 - 1)(x2 - 1) = m + 6 implies m = 4
theorem exist_real_root (m x1 x2 : ℝ) : are_roots m x1 x2 → (x1 - 1) * (x2 - 1) = m + 6 → m = 4 :=
sorry

end range_of_m_exist_real_root_l610_610973


namespace problem_proof_l610_610040

theorem problem_proof :
  (3 ∣ 18) ∧
  (17 ∣ 187 ∧ ¬ (17 ∣ 52)) ∧
  ¬ ((24 ∣ 72) ∧ (24 ∣ 67)) ∧
  ¬ (13 ∣ 26 ∧ ¬ (13 ∣ 52)) ∧
  (8 ∣ 160) :=
by 
  sorry

end problem_proof_l610_610040


namespace greatest_four_digit_multiple_of_17_l610_610011

theorem greatest_four_digit_multiple_of_17 :
  ∃ n, (n % 17 = 0) ∧ (1000 ≤ n ∧ n ≤ 9999) ∧ ∀ m, (m % 17 = 0) ∧ (1000 ≤ m ∧ m ≤ 9999) → n ≥ m :=
begin
  use 9996,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm2a hm2b,
    exact nat.mul_le_mul_right _ (nat.div_le_of_le_mul (nat.le_sub_one_of_lt (lt_of_le_of_lt (nat.mul_le_mul_right _ (nat.le_of_dvd hm1)) (by norm_num)))),
  },
  sorry
end

end greatest_four_digit_multiple_of_17_l610_610011


namespace function_property_l610_610197

-- Define the function f with the given properties and the goal to prove f(2012) = 2016
theorem function_property (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f(x + 3) ≤ f(x) + 3) 
                         (h2 : ∀ x : ℝ, f(x + 2) ≥ f(x) + 2) 
                         (h3 : f 998 = 1002) : f 2012 = 2016 :=
begin
  sorry
end

end function_property_l610_610197


namespace range_of_log_condition_l610_610951

theorem range_of_log_condition {a b c : ℝ} (h : a > 0) :
  (∀ y : ℝ, ∃ x : ℝ, y = log (a * x^2 - b * x - c)) ↔ ∃ x : ℝ, a * x^2 ≤ b * x + c :=
sorry

end range_of_log_condition_l610_610951


namespace indoor_tables_count_l610_610378

theorem indoor_tables_count
  (I : ℕ)  -- the number of indoor tables
  (O : ℕ)  -- the number of outdoor tables
  (H1 : O = 12)  -- Condition 1: O = 12
  (H2 : 3 * I + 3 * O = 60)  -- Condition 2: Total number of chairs
  : I = 8 :=
by
  -- Insert the actual proof here
  sorry

end indoor_tables_count_l610_610378


namespace distance_problem_l610_610083

-- Given conditions
def rowing_speed : ℝ := 4
def wind_speed : ℝ := -1
def current_speed : ℝ := 2
def round_trip_time : ℝ := 1.5

-- Prove distance to the place, D, is approximately 4.09 km
theorem distance_problem (D : ℝ) 
  (H : rowing_speed + current_speed = 6 ∧ rowing_speed + current_speed + wind_speed = 5) :
  (D / 6 + D / 5 = round_trip_time) → D ≈ 4.09 :=
sorry

end distance_problem_l610_610083


namespace coloring_ways_l610_610919

-- Let's define the problem conditions
def color := {r : Type} [fintype : fin 4]

-- The question is to prove that the number of ways to color the vertices is exactly 5184
theorem coloring_ways : ∃! (f : fin 12 → color),
  (∀ (i j : fin 12), adjacent i j → f i ≠ f j) →
  fintype.card (set_of (λ (f : fin 12 → color), (∀ i j, adjacent i j → f i ≠ f j))) = 5184 :=
begin
  sorry
end

end coloring_ways_l610_610919


namespace function_domain_l610_610381

theorem function_domain :
  { x : ℝ | (x - 3 ≥ 0) ∧ (4 - x > 0) ∧ (log 2 (4 - x) ≠ 0) } = { x : ℝ | 3 < x ∧ x < 4 } :=
sorry

end function_domain_l610_610381


namespace div_by_17_l610_610326

theorem div_by_17 (n : ℕ) (h : ¬ 17 ∣ n) : 17 ∣ (n^8 + 1) ∨ 17 ∣ (n^8 - 1) := 
by sorry

end div_by_17_l610_610326


namespace periodic_sequence_characteristic_polynomial_l610_610158

-- Definitions
def homogeneous_linear_recursive_sequence (a : ℕ → ℂ) (k : ℕ) (c : Fin k → ℂ) : Prop := 
∀ n, a (n + k) = ∑ i in Fin k, c i * a (n + i)

noncomputable def is_periodic (a : ℕ → ℂ) (T : ℕ) : Prop :=
∀ n, a n = a (n + T)

-- Problem Statement
theorem periodic_sequence_characteristic_polynomial (a : ℕ → ℂ) (T k : ℕ) (c : Fin k → ℂ) 
  (h_rec : homogeneous_linear_recursive_sequence a k c) 
  (h_period : is_periodic a T) :
  ∃ P : Polynomial ℂ, P.is_cyclotomic ∧ (P.eval x = 0 ↔ ∀ n, x ^ T = 1 ∧ P.is_root x) ∧ (P.roots N).nodup :=
sorry

end periodic_sequence_characteristic_polynomial_l610_610158


namespace factorization_l610_610574

theorem factorization (c : ℝ) : 196 * c^3 + 28 * c^2 = 28 * c^2 * (7 * c + 1) :=
by
  sorry

end factorization_l610_610574


namespace find_N_l610_610284

-- Define the committee with specified conditions
def alien := ℕ
def robot := ℕ
def human := ℕ

-- Let A denote the set of aliens, R set of robots, and H set of humans 
constant A : fin 6 → alien
constant R : fin 6 → robot
constant H : fin 6 → human

-- Define positions and seating rules
constant position : fin 18 → (alien ⊕ robot ⊕ human)
axiom seating_rule_1 : position 0 = some (inl A)  -- Alien at chair 1
axiom seating_rule_2 : position 17 = some (inr (inr H))  -- Human at chair 18
axiom no_human_left_of_alien : ∀ i, position (i + 1) = some (inl A) → position i ≠ some (inr (inr H))
axiom no_alien_left_of_robot : ∀ i, position (i + 1) = some (inr (inl R)) → position i ≠ some (inl A)
axiom no_robot_left_of_human : ∀ i, position (i + 1) = some (inr (inr H)) → position i ≠ some (inr (inl R))

-- Define the number of possible arrangements
noncomputable def num_arrangements : ℕ := 614000 * (6! * 6! * 6!)

-- Prove the number of possible seating arrangements equals N * (6!)^3
theorem find_N : num_arrangements = 614000 * (6! * 6! * 6!) :=
by
  sorry

end find_N_l610_610284


namespace prove_KM_perp_DL_l610_610730

open EuclideanGeometry

noncomputable def trapezium_problem (ABCD : Trapezium) (k : Circle)
    (E : Point) (K L : Point) (M : Point) : Prop :=
  (ABCD.inscribed_in k) ∧
  (E = diagonal_inter AB CD) ∧
  (Circle.center k = B) ∧ (Circle.radius k = BE) ∧
  (Circle.meet_at k K L) ∧
  (Line.perpendicular_to BD E M.intersect_at CD)

// The theorem to be proven
theorem prove_KM_perp_DL (ABCD : Trapezium) (k : Circle) (E : Point)
    (K L : Point) (M : Point):
  trapezium_problem ABCD k E K L M → Perpendicular KM DL := sorry

end prove_KM_perp_DL_l610_610730


namespace solve_for_x_l610_610587

theorem solve_for_x : 
  ∃ x > 0, log 5 (x + 3) + log (5^2) (x^2 + 5) + log (5^(-1)) (x + 3) = 3 ∧ x = Real.sqrt 15620 :=
by
  sorry

end solve_for_x_l610_610587


namespace rectangles_containment_exists_l610_610532

open Set

-- Defining the condition for rectangle containment
def is_contained_in (R1 R2 : ℕ × ℕ) : Prop :=
  R1.1 ≤ R2.1 ∧ R1.2 ≤ R2.2

-- Infinite set of rectangles
variable (rectangles : Set (ℕ × ℕ))
variable (h_infinite : Infinite rectangles)

theorem rectangles_containment_exists :
  ∃ R1 R2 ∈ rectangles, is_contained_in R1 R2 ∨ is_contained_in R2 R1 :=
sorry

end rectangles_containment_exists_l610_610532


namespace smallest_prime_sum_of_five_distinct_primes_l610_610849

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def distinct (a b c d e : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

theorem smallest_prime_sum_of_five_distinct_primes :
  ∃ a b c d e : ℕ, is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ distinct a b c d e ∧ (a + b + c + d + e = 43) ∧ is_prime 43 :=
sorry

end smallest_prime_sum_of_five_distinct_primes_l610_610849


namespace equal_perimeters_of_parallel_cross_sections_l610_610680

variable (A1 A2 A3 A4 : Point)
variable (len : ℝ)

-- Conditions
axiom equal_edges : dist A1 A2 = dist A3 A4

-- Theorem to prove
theorem equal_perimeters_of_parallel_cross_sections :
  ∀ (P1 P2 P3 P4 : Plane), 
  (parallel P1 P2 A1 A2) → (parallel P3 P4 A3 A4) → 
  (parallelogram (section P1 P2)) ∧ (parallelogram (section P3 P4)) ∧ 
  (perimeter (section P1 P2) = 2 * len) ∧ (perimeter (section P3 P4) = 2 * len) := 
sorry

end equal_perimeters_of_parallel_cross_sections_l610_610680


namespace max_points_plane_no_obtuse_max_points_space_no_obtuse_l610_610026

-- Definition of obtuse angle
def is_obtuse_angle (a b c : ℝ) : Prop :=
  a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2

-- Definition of no obtuse angles
def no_obtuse_triangles (points : Finset (ℝ × ℝ)) : Prop :=
  ∀ (p1 p2 p3 : ℝ × ℝ) (h1 : p1 ∈ points) (h2 : p2 ∈ points) (h3 : p3 ∈ points),
    ¬ is_obtuse_angle (dist p1 p2) (dist p2 p3) (dist p3 p1)

-- Maximum number of points that can be placed on a plane with no obtuse triangles
theorem max_points_plane_no_obtuse : 
  ∀ (points : Finset (ℝ × ℝ)), (no_obtuse_triangles points ∧ ∀ (p1 p2 p3 : ℝ × ℝ), (p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points) → p1 ≠ p2 ∨ p2 ≠ p3 ∨ p1 ≠ p3) → points.card ≤ 4 := 
sorry

-- Maximum number of points that can be placed in space with no obtuse triangles
theorem max_points_space_no_obtuse : 
  ∀ (points : Finset (ℝ × ℝ × ℝ)), (no_obtuse_triangles points ∧ ∀ (p1 p2 p3 : ℝ × ℝ × ℝ), (p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points) → p1 ≠ p2 ∨ p2 ≠ p3 ∨ p1 ≠ p3) → points.card ≤ 8 := 
sorry

end max_points_plane_no_obtuse_max_points_space_no_obtuse_l610_610026


namespace part_a_l610_610046

theorem part_a (m : ℕ) (A B : ℕ) (hA : A = (10^(2 * m) - 1) / 9) (hB : B = 4 * ((10^m - 1) / 9)) :
  ∃ k : ℕ, A + B + 1 = k^2 :=
sorry

end part_a_l610_610046


namespace relationship_between_abcd_l610_610851

theorem relationship_between_abcd (a b c d : ℝ) (h : d ≠ 0) :
  (∀ x : ℝ, (a * x + c) / (b * x + d) = (a + c * x) / (b + d * x)) ↔ a / b = c / d :=
by
  sorry

end relationship_between_abcd_l610_610851


namespace no_extremum_points_l610_610653

def f (a x : ℝ) : ℝ := x - (1 / x) - a * Real.log x

def f_prime (a x : ℝ) [hx : x > 0] : ℝ := (x^2 - a * x + 1) / (x^2)

theorem no_extremum_points (a : ℝ) :
  (∀ x > 0, f_prime a x ≥ 0) ↔ (-2 < a ∧ a ≤ 2) :=
sorry

end no_extremum_points_l610_610653


namespace greatest_four_digit_multiple_of_17_l610_610004

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_multiple_of (n d : ℕ) : Prop :=
  ∃ k : ℕ, n = k * d

theorem greatest_four_digit_multiple_of_17 : ∃ n, is_four_digit n ∧ is_multiple_of n 17 ∧
  ∀ m, is_four_digit m → is_multiple_of m 17 → m ≤ n :=
  by
  existsi 9996
  sorry

end greatest_four_digit_multiple_of_17_l610_610004


namespace seating_problem_l610_610877

-- Define the problem parameters and proof goal
def smallest_N (chairs : ℕ) : ℕ :=
  let N := 25 in
  if chairs = 100 then N else 0

theorem seating_problem :
  smallest_N 100 = 25 :=
by
  sorry

end seating_problem_l610_610877


namespace part1_part2_l610_610477

theorem part1 (x y z : ℝ) : 
  x^2 + 2 * y^2 + 3 * z^2 ≥ sqrt 3 * (x * y + y * z + z * x) := 
by sorry

theorem part2 (x y z : ℝ) : 
  ∃ (k : ℝ), k > sqrt 3 ∧ (x^2 + 2 * y^2 + 3 * z^2 ≥ k * (x * y + y * z + z * x)) := 
begin
  use 2,
  split,
  { linarith },
  { sorry }
end

end part1_part2_l610_610477


namespace arithmetic_sequences_count_l610_610912

theorem arithmetic_sequences_count :
  ∃ (x y z : ℕ), (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧ (x + y + z < 100) ∧ number_of_arithmetic_sequences x y z = 1056 :=
sorry

end arithmetic_sequences_count_l610_610912


namespace new_concentration_l610_610099

def vessel1 := (3 : ℝ)  -- 3 litres
def conc1 := (0.25 : ℝ) -- 25% alcohol

def vessel2 := (5 : ℝ)  -- 5 litres
def conc2 := (0.40 : ℝ) -- 40% alcohol

def vessel3 := (7 : ℝ)  -- 7 litres
def conc3 := (0.60 : ℝ) -- 60% alcohol

def vessel4 := (4 : ℝ)  -- 4 litres
def conc4 := (0.15 : ℝ) -- 15% alcohol

def total_volume := (25 : ℝ) -- Total vessel capacity

noncomputable def alcohol_total : ℝ :=
  (vessel1 * conc1) + (vessel2 * conc2) + (vessel3 * conc3) + (vessel4 * conc4)

theorem new_concentration : (alcohol_total / total_volume = 0.302) :=
  sorry

end new_concentration_l610_610099


namespace derivative_f_l610_610380

noncomputable def f (x : ℝ) : ℝ := 1 + Real.cos x

theorem derivative_f (x : ℝ) : deriv f x = -Real.sin x := 
by 
  sorry

end derivative_f_l610_610380


namespace correct_mapping_l610_610784

-- Define the complex numbers
def z1 : ℂ := 5
def z2 : ℂ := -3 * complex.I
def z3 : ℂ := 3 + 2 * complex.I
def z4 : ℂ := 5 - 2 * complex.I
def z5 : ℂ := -3 + 2 * complex.I
def z6 : ℂ := -1 - 5 * complex.I

-- Define the expected points on the complex plane
def point1 : ℂ := (5 : ℂ)
def point2 : ℂ := (0 : ℂ) - 3 * complex.I
def point3 : ℂ := (3 : ℂ) + 2 * complex.I
def point4 : ℂ := (5 : ℂ) - 2 * complex.I
def point5 : ℂ := (-3 : ℂ) + 2 * complex.I
def point6 : ℂ := (-1 : ℂ) - 5 * complex.I

-- Prove that the complex numbers map correctly to the expected points
theorem correct_mapping :
  (z1 = point1) ∧
  (z2 = point2) ∧
  (z3 = point3) ∧
  (z4 = point4) ∧
  (z5 = point5) ∧
  (z6 = point6) :=
by
  repeat { split }, 
  all_goals { refl },
  sorry

end correct_mapping_l610_610784


namespace speed_of_sound_in_hydrogen_l610_610030

-- Definitions of the given conditions and the derived formula
def length : ℝ := 2 / 3
def frequency : ℝ := 990

-- Question statement: What is the speed of sound in hydrogen?
theorem speed_of_sound_in_hydrogen : ∃ (c : ℝ), (frequency = c / (2 * length)) ∧ (c = 1320) :=
sorry

end speed_of_sound_in_hydrogen_l610_610030


namespace find_standard_equation_find_points_P_l610_610964

-- Definitions for conditions
def ellipse (a b : ℝ) : Prop := ∀ x y : ℝ, (x^2)/(a^2) + (y^2)/(b^2) = 1
def e (c a : ℝ) : Prop := c / a = 1 / 3
def tangent_distance (b : ℝ) : Prop := b = 2 * Real.sqrt 2
def ellipse_equation_spec (x y a b c : ℝ) (h1 : a > b ∧ b > 0) (h2 : e c a) (h3 : tangent_distance b) : Prop :=
  a = 3 ∧ c = 1 ∧ ellipse 3 (2 * Real.sqrt 2) x y

-- Theorem for part 1
theorem find_standard_equation : 
  ∃ a b c : ℝ, a > b ∧ b > 0 ∧ 
               e c a ∧ 
               tangent_distance b ∧ 
               ellipse_equation_spec x y a b c :=
sorry

-- Definitions for part 2
def point_P (t : ℝ) : Prop := t = 3 ∨ t = -3
def slope_product_const (t : ℝ) (y1 y2 x1 x2 : ℝ) : Prop :=
  ∀ m : ℝ, (y1 * y2) / (x1 * x2 - t * (x1 + x2) + t^2) = - 4 / 9 ∨ 
           (y1 * y2) / (x1 * x2 - t * (x1 + x2) + t^2) = -16 / 9

-- Theorem for part 2
theorem find_points_P (x y : ℝ) (t : ℝ):
  (∃ P : ℝ × ℝ, P = (-3, 0) ∨ P = (3, 0))  ∧ 
  (∀ M N : ℝ × ℝ, slope_product_const t N.2 M.2 N.1 M.1) :=
sorry

end find_standard_equation_find_points_P_l610_610964


namespace part_a_l610_610631

theorem part_a (
  a b : ℕ
) (coprime_ab : Nat.gcd a b = 1)
  (b_odd : b % 2 = 1)
  (a_gt_2 : a > 2)
  (a_even : a % 2 = 0) :
  ¬ ∃ m n p : ℕ, (m > 0) ∧ (n > 0) ∧ (p > 0) ∧ (x : ℕ → ℕ),
      (x 0 = 2) ∧
      (x 1 = a) ∧
      (∀ k ≥ 1, x (k + 2) = a * x (k + 1) + b * x k) ∧ 
      (x m) / ((x n) * (x p)) ∈ ℕ := 
sorry

end part_a_l610_610631


namespace same_foci_ellipse_hyperbola_l610_610987

theorem same_foci_ellipse_hyperbola (m : ℝ) 
  (h1 : ∀ x y, x^2 / 4 + y^2 / m^2 = 1) 
  (h2 : ∀ x y, x^2 / m^2 - y^2 / 2 = 1)
  (h_foci : sqrt (4 - m^2) = sqrt (m^2 + 2)) : 
  m = 1 ∨ m = -1 :=
sorry

end same_foci_ellipse_hyperbola_l610_610987


namespace probability_of_intersection_l610_610889

def intersects_circle_probability (k : ℝ) : Prop :=
  let line : ℝ → ℝ := λ x => k * (x + 2)
  let circle : ℝ × ℝ → Prop := λ ⟨x, y⟩ => x^2 + y^2 = 1
  let distance : ℝ := |2 * k| / real.sqrt (k^2 + 1)
  distance < 1

theorem probability_of_intersection :
  ∀ (k : ℝ), k ∈ set.Icc (-1:ℝ) (1:ℝ) →
  @probability_of_event (set.Icc (-1:ℝ) (1:ℝ)) {k | intersects_circle_probability k} = real.sqrt 3 / 3 :=
sorry

end probability_of_intersection_l610_610889


namespace school_dance_boys_count_l610_610426

theorem school_dance_boys_count
  (total_attendees : ℕ)
  (percent_faculty_staff : ℝ)
  (fraction_girls : ℝ)
  (h1 : total_attendees = 100)
  (h2 : percent_faculty_staff = 0.1)
  (h3 : fraction_girls = 2/3) :
  let faculty_staff := total_attendees * percent_faculty_staff in
  let students := total_attendees - faculty_staff in
  let girls := students * fraction_girls in
  let boys := students - girls in
  boys = 30 :=
by
  -- Skipping the proof
  sorry

end school_dance_boys_count_l610_610426


namespace smallest_N_l610_610421

noncomputable def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n
  
noncomputable def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

noncomputable def is_fifth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 5 = n

theorem smallest_N :
  ∃ N : ℕ, is_square (N / 2) ∧ is_cube (N / 3) ∧ is_fifth_power (N / 5) ∧
  N = 2^15 * 3^10 * 5^6 :=
by
  exists 2^15 * 3^10 * 5^6
  sorry

end smallest_N_l610_610421


namespace mathematician_daily_questions_l610_610088

theorem mathematician_daily_questions :
  (518 + 476) / 7 = 142 := by
  sorry

end mathematician_daily_questions_l610_610088


namespace fill_time_is_40_minutes_l610_610345

-- Definitions based on the conditions
def pool_volume : ℝ := 60 -- gallons
def filling_rate : ℝ := 1.6 -- gallons per minute
def leaking_rate : ℝ := 0.1 -- gallons per minute

-- Net filling rate
def net_filling_rate : ℝ := filling_rate - leaking_rate

-- Required time to fill the pool
def time_to_fill_pool : ℝ := pool_volume / net_filling_rate

-- Theorem to prove the time is 40 minutes
theorem fill_time_is_40_minutes : time_to_fill_pool = 40 := 
by
  -- This is where the proof would go
  sorry

end fill_time_is_40_minutes_l610_610345


namespace lizzie_garbage_l610_610342

/-- Let G be the amount of garbage Lizzie's group collected. 
We are given that the second group collected G - 39 pounds of garbage,
and the total amount collected by both groups is 735 pounds.
We need to prove that G is 387 pounds. -/
theorem lizzie_garbage (G : ℕ) (h1 : G + (G - 39) = 735) : G = 387 :=
sorry

end lizzie_garbage_l610_610342


namespace real_part_of_complex_l610_610410

theorem real_part_of_complex :
  let z := (i - 1) * (i - 1) + 2 / (i + 1)
  in complex.re z = 0 :=
by
  let z := (i - 1) * (i - 1) + 2 / (i + 1)
  have h : z = 0 := sorry
  exact h

end real_part_of_complex_l610_610410


namespace greatest_four_digit_multiple_of_17_l610_610022

theorem greatest_four_digit_multiple_of_17 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n ∧ n = 9996 :=
by
  use 9996
  split
  { exact by norm_num }
  split
  { exact by norm_num [lt.sub (ofNat 10000) (ofNat 4)] }
  split
  { exact by norm_num }
  { exact by norm_num [dvd_of_mod_eq_zero (by norm_num : 10000 % 17 = 4)] }

end greatest_four_digit_multiple_of_17_l610_610022


namespace find_EC_length_l610_610643

-- Definitions based on given conditions
def angle_A : ℝ := 45
def length_BC : ℕ := 24
def BD_perp_AC : Prop := True
def CE_perp_AB : Prop := True
def angle_DBC_eq_4_mult_angle_ECB (angle_DBC angle_ECB : ℝ) := angle_DBC = 4 * angle_ECB

-- Main theorem statement
theorem find_EC_length (BD_perp_AC : BD_perp_AC)
  (CE_perp_AB : CE_perp_AB) 
  (angle_A : ℝ) (length_BC : ℕ)
  (h1 : angle_A = 45)
  (h2 : length_BC = 24)
  (h3 : ∃ (angle_DBC angle_ECB : ℝ), angle_DBC_eq_4_mult_angle_ECB angle_DBC angle_ECB) :
  ∃ (a b c : ℕ), a = 2 ∧ b = 3 ∧ c = 5 ∧ 
  length_EC angle_A length_BC BD_perp_AC CE_perp_AB = a * (Real.sqrt b + Real.sqrt c) := sorry

end find_EC_length_l610_610643


namespace bug_total_distance_l610_610486

def total_distance (p1 p2 p3 p4 : ℤ) : ℤ :=
  abs (p2 - p1) + abs (p3 - p2) + abs (p4 - p3)

theorem bug_total_distance : total_distance (-3) (-8) 0 6 = 19 := 
by sorry

end bug_total_distance_l610_610486


namespace correct_reasoning_statements_l610_610715

inductive ReasoningStatements
| InductivePartToWhole : ReasoningStatements
| InductiveGeneralToGeneral : ReasoningStatements
| DeductiveGeneralToSpecific : ReasoningStatements
| AnalogicalSpecificToGeneral : ReasoningStatements
| AnalogicalSpecificToSpecific : ReasoningStatements

open ReasoningStatements

def correct_statements : set ReasoningStatements := 
  {InductivePartToWhole, DeductiveGeneralToSpecific, AnalogicalSpecificToSpecific}

theorem correct_reasoning_statements :
  ∀ (s : ReasoningStatements), s ∈ {InductivePartToWhole, InductiveGeneralToGeneral, DeductiveGeneralToSpecific, AnalogicalSpecificToGeneral, AnalogicalSpecificToSpecific} → 
                             (s = InductivePartToWhole ∨ s = DeductiveGeneralToSpecific ∨ s = AnalogicalSpecificToSpecific) ↔ s ∈ correct_statements := 
by
  sorry

end correct_reasoning_statements_l610_610715


namespace find_a5_l610_610661

def S (n : ℕ) : ℕ := n^2 + 1

theorem find_a5 : a 5 = 9 :=
by
  sorry

end find_a5_l610_610661


namespace convert_rect_to_polar_l610_610142

theorem convert_rect_to_polar (y x : ℝ) (h : y = x) : ∃ θ : ℝ, θ = π / 4 :=
by
  sorry

end convert_rect_to_polar_l610_610142


namespace compute_exp_l610_610753

theorem compute_exp (a b : ℚ) (ha : a = 4 / 7) (hb : b = 5 / 6) : a^3 * b^(-2) = 2304 / 8575 := by
  sorry

end compute_exp_l610_610753


namespace max_n_distinct_set_sum_squares_l610_610449

theorem max_n_distinct_set_sum_squares (k : ℕ → ℕ) :
  (∃ n, (∀ i j, i ≠ j → k i ≠ k j) ∧ (k 0)^2 + (k 1)^2 + ... + (k (n-1))^2 = 2010 ∧
  ∀ m, (∃ k', (∀ i j, i ≠ j → k' i ≠ k' j) ∧ (k' 0)^2 + (k' 1)^2 + ... + (k' (m-1))^2 = 2010 → m ≤ 17)) :=
by
  sorry

end max_n_distinct_set_sum_squares_l610_610449


namespace problem_l610_610674

theorem problem (x y z : ℕ) (hx : x < 9) (hy : y < 9) (hz : z < 9) 
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 9])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 9])
  (h3 : 2 * x + y + 3 * z ≡ 5 [MOD 9]) :
  (x * y * z % 9 = 0) :=
sorry

end problem_l610_610674


namespace erdos_limsup_l610_610332

noncomputable def f (n : ℕ) : ℝ := 
  ∑ p in (finset.filter (λ p, nat.prime p ∧ ∃ α, p^α ≤ n ∧ n < p^(α + 1)) (finset.range (n+1))),
  p ^ (nat.floor (real.log n / real.log p))

theorem erdos_limsup : 
  tendsto (λ n, f n * (real.log (real.log n) / (n * real.log n))) at_top (𝓝 1) :=
sorry

end erdos_limsup_l610_610332


namespace arrangement_count_is_12_l610_610908

-- The problem: Arranging the letters a, a, b, b, c, c in a 3x2 grid with no repeats in any row or column
def letters : List Char := ['a', 'a', 'b', 'b', 'c', 'c']

-- Definition of a valid 3x2 arrangement
def valid_arrangement (grid : List (List Char)) : Prop :=
  grid.length = 3 ∧
  ∀ row, row ∈ grid → row.length = 2 ∧ row.nodup ∧ -- Rows must have 2 different elements
  ∀ i, i < 2 → (list_erase_nth grid i).nodup      -- Columns must have different elements

-- Define the total number of valid arrangements
def count_valid_arrangements : Nat :=
  (List.permutations letters).count (λ p, valid_arrangement (List.chunk 2 p))

theorem arrangement_count_is_12 : count_valid_arrangements = 12 :=
by
  sorry

end arrangement_count_is_12_l610_610908


namespace odd_digits_sum_div_by_11_l610_610319

def reverseDigits (n : ℕ) : ℕ :=
  -- Assuming a function exists that reverses the digits of n.
  sorry

theorem odd_digits_sum_div_by_11 (s : ℕ) (hs : s > 0) (odd_digits : (natDigits 10 s).length % 2 = 1) :
  (s % 11 = 0 ↔ (s + reverseDigits s) % 11 = 0) :=
sorry

end odd_digits_sum_div_by_11_l610_610319


namespace part_a_part_b_l610_610734
-- Import the entire math library to provide necessary background for measure theory

-- Use the noncomputable theory for measure theory concepts
noncomputable theory

open Set

-- Define the conditions
variables {Ω : Type*} {F : Set (Set Ω)} {P : Measure Ω}
variables (A : Set Ω) {a : Type*} (As : a → Set Ω)

-- Define the distance measure
def d (A B : Set Ω) : ℝ := P (A ∆ B)

-- Define a sigma algebra generated by an algebra
def sigma_algebra (A : Set (Set Ω)) : Set (Set Ω) := generate_measurable_space A

-- Part (a)
theorem part_a (A : Set (Set Ω)) (B : Set (Set Ω)) :
  ( ∀ ε > 0, ∀ B ∈ sigma_algebra A, ∃ A' ∈ A, P (A' ∆ B) ≤ ε ) :=
begin
  sorry
end

-- Part (b)
theorem part_b (A : Set (Set Ω)) (B : Set Ω) :
  ( ∀ B ∈ sigma_algebra A, ∃ As : ℕ → Set Ω, (∀ n, As n ∈ A) ∧ B ⊆ ⋃ n, As n ∧ P (B ∆ (⋃ n, As n)) ≤ ε ) :=
begin
  sorry
end

end part_a_part_b_l610_610734


namespace prod_inequality_l610_610601

open BigOperators

theorem prod_inequality (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, 0 < x i) :
    ∏ i in Finset.range n, (1 + ∑ j in Finset.range i.succ, x ⟨j, Nat.lt_succ_iff.mpr (Finset.mem_range_succ_iff.mpr j.zero_lt_succ)⟩) 
    ≥ Real.sqrt ((n + 1)^(n + 1)) * Real.sqrt (∏ i in Finset.range n, x i) := sorry

end prod_inequality_l610_610601


namespace triangle_sides_proportional_l610_610401

theorem triangle_sides_proportional (a b c r d : ℝ)
  (h1 : 2 * r < a) 
  (h2 : a < b) 
  (h3 : b < c) 
  (h4 : a = 2 * r + d)
  (h5 : b = 2 * r + 2 * d)
  (h6 : c = 2 * r + 3 * d)
  (hr_pos : r > 0)
  (hd_pos : d > 0) :
  ∃ k : ℝ, k > 0 ∧ a = 3 * k ∧ b = 4 * k ∧ c = 5 * k :=
sorry

end triangle_sides_proportional_l610_610401


namespace shape_described_by_phi_eq_c_is_cone_l610_610936

-- Definitions based on the conditions
def spherical_coords := (ρ θ φ : ℝ)

-- Statement of the problem
theorem shape_described_by_phi_eq_c_is_cone (c : ℝ) :
  ∀ (ρ θ : ℝ), ∃ P : ℝ × ℝ × ℝ, 
    spherical_coords P.1 P.2 c := P ∧ 
    -- Add the necessary conditions that describe the shape being a cone
    sorry

end shape_described_by_phi_eq_c_is_cone_l610_610936


namespace correct_rates_l610_610759

def number_of_pears_picked := 50
def hours_picking_pears := 4
def hours_cooking_bananas := 2
def sandrine_washed_dishes := 5

def b := 3 * number_of_pears_picked               -- Number of bananas Charles cooked (3 times the number of pears he picked)
def d := b + 10                                   -- Number of dishes Sandrine washed (10 more than bananas cooked)
def r1 := number_of_pears_picked / hours_picking_pears -- Rate at which Charles picks pears (pears per hour)
def r2 := b / hours_cooking_bananas               -- Rate at which Charles cooks bananas (bananas per hour)
def r3 := d / sandrine_washed_dishes              -- Rate at which Sandrine washes dishes (dishes per hour)

theorem correct_rates : 
  r1 = 12.5 ∧ r2 = 75 ∧ r3 = 32 := 
by
  sorry

end correct_rates_l610_610759


namespace part_I_part_II_min_value_of_T_l610_610413

-- Definitions based on conditions
def a_n (n : ℕ) : ℝ := sorry  -- will be defined as sqrt(n) - sqrt(n-1) for the proof but we use a placeholder

def S_n (n : ℕ) : ℝ := ∑ i in finset.range n, a_n (i + 1)

-- Condition given in the problem
axiom condition_1 (n : ℕ) : 2 * a_n n * S_n n - (a_n n) ^ 2 = 1

-- Questions to be proved
theorem part_I (n : ℕ) : (S_n n) ^ 2 = n := sorry

theorem part_II_min_value_of_T (n : ℕ) : 
  let T_n := ∑ i in finset.range n, (2 / (4 * (S_n(i + 1)) ^ 4 - 1)) 
  in T_n ≥ 2 / 3 := sorry

end part_I_part_II_min_value_of_T_l610_610413


namespace shaded_area_is_110_l610_610293

-- Definitions based on conditions
def equilateral_triangle_area : ℕ := 10
def num_triangles_small : ℕ := 1
def num_triangles_medium : ℕ := 3
def num_triangles_large : ℕ := 7

-- Total area calculation
def total_area : ℕ := (num_triangles_small + num_triangles_medium + num_triangles_large) * equilateral_triangle_area

-- The theorem statement
theorem shaded_area_is_110 : total_area = 110 := 
by 
  sorry

end shaded_area_is_110_l610_610293


namespace max_cardinality_of_S_existence_of_S_with_n_n_4_div_6_elements_l610_610786

theorem max_cardinality_of_S (n : ℕ) (S : Finset (Finset (Fin n))) 
  (h1 : ∀ s ∈ S, s.card = 3)
  (h2 : ∀ s1 s2 ∈ S, s1 ≠ s2 → (s1 ∩ s2).card ≤ 1) :
  S.card ≤ n * (n - 1) / 6 :=
sorry

theorem existence_of_S_with_n_n_4_div_6_elements (n : ℕ)
  (h3 : n ≥ 4) :
  ∃ S : Finset (Finset (Fin n)),
    (∀ s ∈ S, s.card = 3) ∧
    (∀ s1 s2 ∈ S, s1 ≠ s2 → (s1 ∩ s2).card ≤ 1) ∧
    S.card = (n * (n - 4)) / 6 :=
sorry

end max_cardinality_of_S_existence_of_S_with_n_n_4_div_6_elements_l610_610786


namespace value_range_piecewise_function_l610_610827

noncomputable def piecewise_function (x : ℝ) : ℝ :=
if x ≥ 0 then x - 1 else 1 - x

theorem value_range_piecewise_function :
  set.range piecewise_function = set.Ici (-1) :=
by {
  sorry
}

end value_range_piecewise_function_l610_610827


namespace probability_two_out_of_four_in_favor_l610_610699

/-- In a recent survey, it was found that 60% of a town's residents are in favor of a new public park.
A research group conducts four separate polls, each time randomly selecting a resident to ask about their opinion on the new park.
This theorem states the probability that exactly two out of these four residents are in favor of the new park. -/
theorem probability_two_out_of_four_in_favor (p : ℝ) (n : ℕ) (k : ℕ) (prob_a : p = 0.6) (polls : n = 4) (selected_favor : k = 2) :
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) = 0.3456 :=
by
  sorry

end probability_two_out_of_four_in_favor_l610_610699


namespace problem_proof_l610_610614

theorem problem_proof (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = 3) : 3 * a^2 * b + 3 * a * b^2 = 18 := 
by
  sorry

end problem_proof_l610_610614


namespace partition_and_multiple_of_three_l610_610560

open Set

theorem partition_and_multiple_of_three (n : ℕ) (h1 : 6 ≤ n) :
  (∃ (A1 A2 A3 : Set ℕ), pairwise_disjoint A1 A2 A3 n ∧ same_cardinality A1 A2 A3 ∧ same_sum A1 A2 A3) ↔ n % 3 = 0 :=
sorr

-- Define pairwise_disjoint relation
def pairwise_disjoint (A1 A2 A3 : Set ℕ) (n : ℕ) : Prop :=
  (A1 ∩ A2 = ∅) ∧ (A2 ∩ A3 = ∅) ∧ (A3 ∩ A1 = ∅) ∧ (A1 ∪ A2 ∪ A3 = {1, 2, ..., n})

-- Define same_cardinality relation
def same_cardinality (A1 A2 A3 : Set ℕ) : Prop :=
  card A1 = card A2 ∧ card A1 = card A3

-- Define same_sum relation
def same_sum (A1 A2 A3 : Set ℕ) : Prop :=
  (∑ x in A1, x) = (∑ x in A2, x) ∧ (∑ x in A1, x) = (∑ x in A3, x)

end partition_and_multiple_of_three_l610_610560


namespace acute_triangle_sums_to_pi_over_4_l610_610272

theorem acute_triangle_sums_to_pi_over_4 
    (A B : ℝ) 
    (hA : 0 < A ∧ A < π / 2) 
    (hB : 0 < B ∧ B < π / 2) 
    (h_sinA : Real.sin A = (Real.sqrt 5)/5) 
    (h_sinB : Real.sin B = (Real.sqrt 10)/10) : 
    A + B = π / 4 := 
sorry

end acute_triangle_sums_to_pi_over_4_l610_610272


namespace greatest_four_digit_multiple_of_17_l610_610013

theorem greatest_four_digit_multiple_of_17 :
  ∃ n, (n % 17 = 0) ∧ (1000 ≤ n ∧ n ≤ 9999) ∧ ∀ m, (m % 17 = 0) ∧ (1000 ≤ m ∧ m ≤ 9999) → n ≥ m :=
begin
  use 9996,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm2a hm2b,
    exact nat.mul_le_mul_right _ (nat.div_le_of_le_mul (nat.le_sub_one_of_lt (lt_of_le_of_lt (nat.mul_le_mul_right _ (nat.le_of_dvd hm1)) (by norm_num)))),
  },
  sorry
end

end greatest_four_digit_multiple_of_17_l610_610013


namespace paint_for_cube_l610_610836

theorem paint_for_cube (paint_per_unit_area : ℕ → ℕ → ℕ)
  (h2 : paint_per_unit_area 2 1 = 1) :
  paint_per_unit_area 6 1 = 9 :=
by
  sorry

end paint_for_cube_l610_610836


namespace average_age_calculated_years_ago_l610_610370

theorem average_age_calculated_years_ago
  (n m : ℕ) (a b : ℕ) 
  (total_age_original : ℝ)
  (average_age_original : ℝ)
  (average_age_new : ℝ) :
  n = 6 → 
  a = 19 → 
  m = 7 → 
  b = 1 → 
  total_age_original = n * a → 
  average_age_original = a → 
  average_age_new = a →
  (total_age_original + b) / m = a → 
  1 = 1 := 
by
  intros _ _ _ _ _ _ _ _
  sorry

end average_age_calculated_years_ago_l610_610370


namespace jimmy_calorie_consumption_l610_610916

variables (crackers_calories cookies_calories total_calories : ℕ) (cookies_eaten : ℕ)

def cracker_count_needed (cals_from_cookies remaining_cals : ℕ) : ℕ :=
  remaining_cals / crackers_calories

theorem jimmy_calorie_consumption : 
  crackers_calories = 15 → 
  cookies_calories = 50 → 
  cookies_eaten = 7 →
  total_calories = 500 →
  cracker_count_needed cookies_calories (total_calories - cookies_calories * cookies_eaten) = 10 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end jimmy_calorie_consumption_l610_610916


namespace annika_hike_distance_l610_610861

theorem annika_hike_distance (rate : ℝ) (initial_distance : ℝ) (total_time : ℕ) (remaining_time : ℝ) (total_distance : ℝ) :
  rate = 12 ∧
  initial_distance = 2.75 ∧
  total_time = 40 ∧
  remaining_time = (total_time - (initial_distance * rate)) / rate ∧
  total_distance = initial_distance + remaining_time →
  total_distance ≈ 3.333 :=  
by
  intros h
  sorry

end annika_hike_distance_l610_610861


namespace find_a1_l610_610205

-- Define the conditions
variables (a1 : ℝ) (q : ℝ) (n : ℕ)
variable (seq : ℕ → ℝ := λ k => a1 * q^k)

-- Conditions as per problem statement
axiom sum_odd_terms : a1 * (1 + q^2 + q^4 + ... + q^(2*n)) = 255
axiom sum_even_terms : a1 * (q + q^3 + q^5 + ... + q^(2*n-1)) = -126
axiom last_term : a1 * q^(2*n) = 192

-- Goal: Prove the first term a1 equals 3
theorem find_a1 : a1 = 3 :=
sorry

end find_a1_l610_610205


namespace two_equal_circles_cannot_have_one_common_tangent_l610_610839

noncomputable def commonTangentsInPlane (r : ℝ) : ℕ := sorry

theorem two_equal_circles_cannot_have_one_common_tangent
  (r : ℝ) (h1 : r > 0) :
  ∀ (n : ℕ), n = 1 → commonTangentsInPlane r ≠ n :=
begin
  sorry
end

end two_equal_circles_cannot_have_one_common_tangent_l610_610839


namespace compute_a3_binv2_l610_610745

-- Define variables and their values
def a : ℚ := 4 / 7
def b : ℚ := 5 / 6

-- State the main theorem that directly translates the problem to Lean
theorem compute_a3_binv2 : (a^3 * b^(-2)) = (2304 / 8575) :=
by
  -- proof left as an exercise for the user
  sorry

end compute_a3_binv2_l610_610745


namespace part1_part2_l610_610941

/-- Given vectors a and b, and function g(x). -/
def vec_a (x: ℝ) (k: ℝ) : ℝ × ℝ := (sqrt 3 * sin x, k * cos x)
def vec_b (x: ℝ) (k: ℝ) : ℝ × ℝ := (2 * k * cos x, 2 * cos x)

def g (x: ℝ) (k: ℝ) : ℝ := 
  let a := vec_a x k 
  let b := vec_b x k
  a.1 * b.1 + a.2 * b.2 - k + 1

def perfect_triangle_function (f : ℝ → ℝ) (s : set ℝ) : Prop :=
  ∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → a ∈ s → b ∈ s → c ∈ s → 
  f a + f b > f c ∧ f b + f c > f a ∧ f c + f a > f b

/-- Prove the range of k for given conditions -/
theorem part1 (k : ℝ) : 
  perfect_triangle_function (λ x => g x k) (Icc 0 (Real.pi / 2)) → 
  k ∈ Ioo (-(1/5)) (1/4) := sorry

/-- Define h(x) -/
def h (x : ℝ) : ℝ := 
  sin (2 * x) - (13 * sqrt 2 / 5) * sin (x + Real.pi / 4) + 369 / 100

/-- Proves the range of k under additional conditions -/
theorem part2 (k : ℝ) : 
  k > 0 → 
  (∀ (x1 : ℝ) (hx1 : x1 ∈ Icc 0 (Real.pi / 2)), ∃ (x2 : ℝ), x2 ∈ Icc 0 (Real.pi / 2) ∧ g x2 k ≥ h x1) → 
  k ∈ Ici (9 / 200) ∩ Iio (1 / 4) := sorry

end part1_part2_l610_610941


namespace find_m_l610_610915

noncomputable def s : ℕ → ℚ
| 1        := 2
| (n + 1) := if (n + 1) % 3 = 0 then 1 + s ((n + 1) / 3) else 1 / s n

theorem find_m (m : ℕ) (h : s m = 34 / 81) : m = 82 :=
by
  sorry

end find_m_l610_610915


namespace circle_area_l610_610161

theorem circle_area (C : ℝ) (hC : C = 31.4) : 
  ∃ (A : ℝ), A = 246.49 / π := 
by
  sorry -- proof not required

end circle_area_l610_610161


namespace Cody_age_is_14_l610_610553

variable (CodyGrandmotherAge CodyAge : ℕ)

theorem Cody_age_is_14 (h1 : CodyGrandmotherAge = 6 * CodyAge) (h2 : CodyGrandmotherAge = 84) : CodyAge = 14 := by
  sorry

end Cody_age_is_14_l610_610553


namespace polynomial_value_l610_610033

theorem polynomial_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end polynomial_value_l610_610033


namespace total_processing_time_correct_l610_610528

-- Define the number of pictures for each type
def trees_pictures : ℕ := 320
def flowers_pictures : ℕ := 400
def grass_pictures : ℕ := 240

-- Define the processing time per picture for each type in minutes
def tree_processing_time : ℝ := 1.5
def flower_processing_time : ℝ := 2.5
def grass_processing_time : ℝ := 1.0

-- Define the total processing times
def total_tree_time := trees_pictures * tree_processing_time
def total_flower_time := flowers_pictures * flower_processing_time
def total_grass_time := grass_pictures * grass_processing_time

-- Define the total time in minutes
def total_time_minutes := total_tree_time + total_flower_time + total_grass_time

-- Convert total time from minutes to hours
def total_time_hours := total_time_minutes / 60

-- Prove that the total processing time in hours is 28.67
theorem total_processing_time_correct 
  (ht : total_tree_time = 320 * 1.5) 
  (hf : total_flower_time = 400 * 2.5) 
  (hg : total_grass_time = 240 * 1.0) 
  (hm : total_time_minutes = 1720) 
  (hh : total_time_hours = 28.6667) : 
  total_time_hours = 28.67 :=
by
  sorry

end total_processing_time_correct_l610_610528


namespace limonia_largest_unachievable_l610_610708

noncomputable def largest_unachievable_amount (n : ℕ) : ℕ :=
  12 * n^2 + 14 * n - 1

theorem limonia_largest_unachievable (n : ℕ) :
  ∀ k, ¬ ∃ a b c d : ℕ, 
    k = a * (6 * n + 1) + b * (6 * n + 4) + c * (6 * n + 7) + d * (6 * n + 10) 
    → k = largest_unachievable_amount n :=
sorry

end limonia_largest_unachievable_l610_610708


namespace circle_center_radius_l610_610807

/-- The endpoints of a diameter of the circle C are (2, -3) and (8, 5). -/
def endPoint1 : ℝ × ℝ := (2, -3)
def endPoint2 : ℝ × ℝ := (8, 5)

/-- Function to calculate the midpoint of two points in the Cartesian plane. -/
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

/-- Function to calculate the Euclidean distance between two points in the Cartesian plane. -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

/-- Prove that the center and radius of the circle C are correctly calculated. -/
theorem circle_center_radius :
  midpoint endPoint1 endPoint2 = (5, 1) ∧ distance (midpoint endPoint1 endPoint2) endPoint1 = 5 := 
by
  sorry

end circle_center_radius_l610_610807


namespace greatest_four_digit_multiple_of_17_l610_610020

theorem greatest_four_digit_multiple_of_17 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n ∧ n = 9996 :=
by
  use 9996
  split
  { exact by norm_num }
  split
  { exact by norm_num [lt.sub (ofNat 10000) (ofNat 4)] }
  split
  { exact by norm_num }
  { exact by norm_num [dvd_of_mod_eq_zero (by norm_num : 10000 % 17 = 4)] }

end greatest_four_digit_multiple_of_17_l610_610020


namespace transform_sine_cosine_l610_610835

theorem transform_sine_cosine :
  ∀ x, √2 * sin (2 * (x - π / 8)) = sin (2 * x) - cos (2 * x) :=
by
  sorry

end transform_sine_cosine_l610_610835


namespace range_of_y_over_x_l610_610688

theorem range_of_y_over_x : ∀ (x y : ℝ), x^2 + (y - 3)^2 = 1 → 
  ∃ k : ℝ, k = y / x ∧ k ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) :=
by
  intros x y h
  sorry

end range_of_y_over_x_l610_610688


namespace novels_next_to_each_other_l610_610852

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem novels_next_to_each_other (n_essays n_novels : Nat) (condition_novels : n_novels = 2) (condition_essays : n_essays = 3) :
  let total_units := (n_novels - 1) + n_essays
  factorial total_units * factorial n_novels = 48 :=
by
  sorry

end novels_next_to_each_other_l610_610852


namespace sequence_sum_l610_610718

theorem sequence_sum (A B C D E F G H I J : ℤ)
  (h1 : D = 7)
  (h2 : A + B + C = 24)
  (h3 : B + C + D = 24)
  (h4 : C + D + E = 24)
  (h5 : D + E + F = 24)
  (h6 : E + F + G = 24)
  (h7 : F + G + H = 24)
  (h8 : G + H + I = 24)
  (h9 : H + I + J = 24) : 
  A + J = 105 :=
sorry

end sequence_sum_l610_610718


namespace greatest_points_scored_l610_610056

theorem greatest_points_scored (n : ℕ) (points : ℕ) (min_points : ℕ) (p : ℕ → ℕ) :
  n = 12 ∧ points = 100 ∧ (∀ i, i < n → p i ≥ 7) ∧ (∑ i in Finset.range n, p i) = points →
  ∃ i, p i = 23 :=
by
  sorry

end greatest_points_scored_l610_610056


namespace solve_sin_solve_cos_l610_610955

variable (α : ℝ)
variable (h1 : α ∈ set.Ioo (Float.pi / 2) Float.pi)
variable (h2 : tan α = -2)

theorem solve_sin : sin (Float.pi / 4 + α) = sqrt 10 / 10 := by
  sorry

theorem solve_cos : cos (2 * Float.pi / 3 - 2 * α) = (3 - 4 * sqrt 3) / 10 := by
  sorry

end solve_sin_solve_cos_l610_610955


namespace circle_equation_from_diameter_l610_610383

theorem circle_equation_from_diameter (A B : ℝ × ℝ) (hA : A = (-3, -1)) (hB : B = (5, 5)) :
  ∃ C r, (C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) ∧ 
         (r = real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)) ∧ 
         ((x - C.1)^2 + (y - C.2)^2 = r^2) :=
by
  sorry

end circle_equation_from_diameter_l610_610383


namespace circles_intersect_common_point_l610_610722

variables {A B C M M_A M_B M_C : Point}
variable [triangle ABC]

-- Definitions of medians intersecting at point M
axiom median_A : A⟶M_A
axiom median_B : B⟶M_B
axiom median_C : C⟶M_C
axiom medians_intersect : median_A = median_B ∩ median_C = M

-- Definitions of circles passing through midpoints and tangent to segments
axiom circle_Omega_A : Circle Ω_A
axiom circle_Omega_B : Circle Ω_B
axiom circle_Omega_C : Circle Ω_C

axiom Omega_A_prop : midpoint A M ∈ Ω_A ∧ tangent Ω_A BC M_A
axiom Omega_B_prop : midpoint B M ∈ Ω_B ∧ tangent Ω_B AC M_B
axiom Omega_C_prop : midpoint C M ∈ Ω_C ∧ tangent Ω_C AB M_C

-- Theorem to prove that the circles intersect at a common point
theorem circles_intersect_common_point :
  ∃ X, X ∈ Ω_A ∧ X ∈ Ω_B ∧ X ∈ Ω_C := 
sorry

end circles_intersect_common_point_l610_610722


namespace median_and_mean_of_set_l610_610267

noncomputable def mode_of_set : set ℕ → ℕ
| {2, 3, x, 5, 7} := 3 -- given condition mode is 3

-- The main theorem statement
theorem median_and_mean_of_set (x : ℕ)
  (h_mode : mode_of_set {2, 3, x, 5, 7} = 3) :
  let sorted_set := [2, 3, 3, 5, 7] in
  (sorted_set.nth 2).get_or_else 0 = 3 ∧ 
  ((2 + 3 + 3 + 5 + 7) / 5 = 4) := 
by
  sorry

end median_and_mean_of_set_l610_610267


namespace divisibility_polynomial_coeffs_l610_610308

/--
Let ϕ(n) denote the number of positive integers less than or equal to n 
which are relatively prime to n. Determine the number of positive integers 
2 ≤ n ≤ 50 such that all coefficients of the polynomial 
(x^ϕ(n) - 1) - ∏_{1 ≤ k ≤ n with gcd(k, n) = 1}(x - k) 
are divisible by n.
-/
theorem divisibility_polynomial_coeffs :
  let ϕ : ℕ → ℕ := Euler.totient
  in (count (λ n, 2 ≤ n ∧ n ≤ 50 ∧ 
      ∀ k : ℕ, k ≤ ϕ(n) → coeff ((X ^ ϕ(n) - 1) - 
      ∏ i in (finset.range n).filter (λ i, nat.coprime i n), 
        (X - C i)) k % n = 0) (finset.range 51)) = 19 := 
sorry

end divisibility_polynomial_coeffs_l610_610308


namespace negative_x_is_positive_l610_610177

theorem negative_x_is_positive (x : ℝ) (hx : x < 0) : -x > 0 :=
sorry

end negative_x_is_positive_l610_610177


namespace Jack_total_money_in_dollars_l610_610724

variable (Jack_dollars : ℕ) (Jack_euros : ℕ) (euro_to_dollar : ℕ)

noncomputable def total_dollars (Jack_dollars : ℕ) (Jack_euros : ℕ) (euro_to_dollar : ℕ) : ℕ :=
  Jack_dollars + Jack_euros * euro_to_dollar

theorem Jack_total_money_in_dollars : 
  Jack_dollars = 45 → 
  Jack_euros = 36 → 
  euro_to_dollar = 2 → 
  total_dollars 45 36 2 = 117 :=
by
  intro h1 h2 h3
  unfold total_dollars
  rw [h1, h2, h3]
  -- skipping the actual proof
  sorry

end Jack_total_money_in_dollars_l610_610724


namespace horner_value_at_neg4_l610_610444

noncomputable def f (x : ℝ) : ℝ := 10 + 25 * x - 8 * x^2 + x^4 + 6 * x^5 + 2 * x^6

def horner_rewrite (x : ℝ) : ℝ := (((((2 * x + 6) * x + 1) * x + 0) * x - 8) * x + 25) * x + 10

theorem horner_value_at_neg4 : horner_rewrite (-4) = -36 :=
by sorry

end horner_value_at_neg4_l610_610444


namespace max_ab_l610_610948

theorem max_ab (a b : ℝ) (h : ∀ x : ℝ, exp (x + 1) ≥ a * x + b) : ab ≤ (1/2) * exp 3 :=
sorry

end max_ab_l610_610948


namespace difference_of_digits_l610_610805

theorem difference_of_digits (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
  (h_diff : (10 * x + y) - (10 * y + x) = 54) : x - y = 6 :=
sorry

end difference_of_digits_l610_610805


namespace polynomial_inequality_l610_610733

variable (n : ℕ) (p : Polynomial ℝ) (a : ℕ → ℝ) (k : ℕ)

theorem polynomial_inequality
  (h1 : n ≥ 2)
  (h2 : p = Polynomial.monomial n 1 + ∑ i in Finset.range n, Polynomial.monomial i (a i))
  (h3 : (Polynomial.X - 1)^(k + 1) ∣ p) :
  (∑ j in Finset.range n, abs (a j)) > 1 + (2 * k^2) / n := sorry

end polynomial_inequality_l610_610733


namespace value_of_a_strict_increasing_interval_l610_610230

-- Definitions based on conditions
def curve (a : ℝ) (x : ℝ) := a * x^2 - Real.log x
def tangent_parallel_x (a : ℝ) := (deriv (curve a)) 1 = 0

-- The proof problem part 1: finding a
theorem value_of_a :
  tangent_parallel_x (1/2) :=
by
  -- Proof goes here
  sorry

-- The proof problem part 2: determining the interval of strict increase
theorem strict_increasing_interval :
  (∀ x > 1, deriv (curve (1/2)) x > 0) :=
by
  -- Proof goes here
  sorry

end value_of_a_strict_increasing_interval_l610_610230


namespace problem_A_problem_B_problem_D_final_answer_l610_610038

-- Define the problem statements as Lean theorems or lemmas.
theorem problem_A : sin (1 / 2) < sin (5 * π / 6) :=
sorry

theorem problem_B : cos (3 * π / 4) > cos (5 * π / 6) :=
sorry

theorem problem_D : sin (π / 5) < cos (π / 5) :=
sorry

-- Compile the results to form the conclusion
theorem final_answer : sin (1 / 2) < sin (5 * π / 6) ∧ cos (3 * π / 4) > cos (5 * π / 6) ∧ sin (π / 5) < cos (π / 5) :=
by
  constructor
  . exact problem_A
  . constructor
    . exact problem_B
    . exact problem_D

end problem_A_problem_B_problem_D_final_answer_l610_610038


namespace probability_of_double_domino_l610_610555

theorem probability_of_double_domino :
  let integers := Finset.range 13
  let all_pairs := (integers.product integers).filter (λ ⟨i, j⟩, i ≤ j)
  let doubles := all_pairs.filter (λ ⟨i, j⟩, i = j)
  (doubles.card : ℚ) / all_pairs.card = (1 : ℚ) / 13 :=
by
  -- Definitions
  let integers := Finset.range 13
  let all_pairs := (integers.product integers).filter (λ ⟨i, j⟩, i ≤ j)
  let doubles := all_pairs.filter (λ ⟨i, j⟩, i = j)
  
  -- Total number of doubles
  have h1 : doubles.card = 13 := by sorry
  
  -- Total number of pairs
  have h2 : all_pairs.card = 169 := by sorry
  
  -- Probability calculation
  have h3 : (doubles.card : ℚ) / all_pairs.card = (1 : ℚ) / 13 := by sorry
  
  exact h3


end probability_of_double_domino_l610_610555


namespace sum_of_perfect_square_l610_610972

theorem sum_of_perfect_square (x k : ℤ) (h : x = k^2) : x + (k + 1)^2 = 2x + 2 * k + 1 := by
  finish sorry

end sum_of_perfect_square_l610_610972


namespace triangle_properties_l610_610282
-- Import the necessary Lean library

-- Define the conditions as hypotheses for the Lean theorem
theorem triangle_properties (a b c A B C : ℝ)
  (hApos : 0 < A) (hAacute : A < π)
  (hAangle : A + B + C = π) 
  (hAacute2 : A = π / 3)
  (htriangle : a / sin A = b / sin B ∧ b / sin B = c / sin C)
  (hsin_eq : sin B^2 + sin C^2 = sin A^2 + sin B * sin C):
  A = π / 3 ∧
  a = 3 → (b + c) ≤ 6 := 
by
  sorry

end triangle_properties_l610_610282


namespace power_of_ten_zeros_l610_610454

theorem power_of_ten_zeros (n : ℕ) (h : n = 10000) : (n ^ 50).digits.count 0 = 200 :=
by sorry

end power_of_ten_zeros_l610_610454


namespace compute_a3_binv2_l610_610744

-- Define variables and their values
def a : ℚ := 4 / 7
def b : ℚ := 5 / 6

-- State the main theorem that directly translates the problem to Lean
theorem compute_a3_binv2 : (a^3 * b^(-2)) = (2304 / 8575) :=
by
  -- proof left as an exercise for the user
  sorry

end compute_a3_binv2_l610_610744


namespace compare_abc_l610_610257

noncomputable def a := Real.log (1 / 2)
noncomputable def b := (1 / 3) ^ 0.8
noncomputable def c := 2 ^ (1 / 3)

theorem compare_abc : a < b ∧ b < c :=
by {
  -- no need to consider the solution steps
  sorry
}

end compare_abc_l610_610257


namespace sandwich_percentage_not_vegetables_l610_610499

noncomputable def percentage_not_vegetables (total_weight : ℝ) (vegetable_weight : ℝ) : ℝ :=
  (total_weight - vegetable_weight) / total_weight * 100

theorem sandwich_percentage_not_vegetables :
  percentage_not_vegetables 180 50 = 72.22 :=
by
  sorry

end sandwich_percentage_not_vegetables_l610_610499


namespace midpoint_incenter_of_triangle_l610_610220

variables {A B C P Q M O O₁ : Type*}

def is_tangent (O O₁ : Type*) (A B : Type*) : Prop := sorry

def is_incenter (M A B C : Type*) : Prop := sorry

theorem midpoint_incenter_of_triangle
  (h_tangent_circles : is_tangent O O₁ (O₁ : Type*))
  (h_tangent_AB : is_tangent AB P (O₁ : Type*))
  (h_tangent_AC : is_tangent AC Q (O₁ : Type*))
  (h_midpoint_PQ : sorry) :
  is_incenter M A B C :=
sorry

end midpoint_incenter_of_triangle_l610_610220


namespace regular_polygon_angle_ratio_3_2_l610_610819

theorem regular_polygon_angle_ratio_3_2 (pairs : List (ℕ × ℕ)) : 
  -- Condition: the ratio of interior angles is 3:2 and r, k > 2
  (∀ r k, (r, k) ∈ pairs → 
          (180 - 360 / r) / (180 - 360 / k) = 3/2 ∧ r > 2 ∧ k > 2) → 
  -- Question: Prove that there are exactly 3 pairs
  pairs.length = 3 :=
begin
  sorry -- proof goes here
end

end regular_polygon_angle_ratio_3_2_l610_610819


namespace arithmetic_sequence_sum_l610_610285

noncomputable def isArithmeticSeq (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_first_n (a : ℕ → ℤ) (n : ℕ) := (n + 1) * (a 0 + a n) / 2

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (h_legal_seq : isArithmeticSeq a) (h_sum : sum_first_n a 9 = 120) : 
  a 1 + a 8 = 24 := by
  sorry

end arithmetic_sequence_sum_l610_610285


namespace cycle_selling_price_l610_610880

theorem cycle_selling_price
  (cost_price : ℝ)
  (selling_price : ℝ)
  (percentage_gain : ℝ)
  (h_cost_price : cost_price = 1000)
  (h_percentage_gain : percentage_gain = 8) :
  selling_price = cost_price + (percentage_gain / 100) * cost_price :=
by
  sorry

end cycle_selling_price_l610_610880


namespace positional_relationship_parallel_or_skew_l610_610685

-- given definitions and conditions
variable (a b a' b' α : Type)
variable [Plane α]
variables [IsProjection (a, α) a'] [IsProjection (b, α) b']
variables [Parallel a' b']

-- theorem statement
theorem positional_relationship_parallel_or_skew :
  (Parallel a b ∨ Skew a b) :=
sorry

end positional_relationship_parallel_or_skew_l610_610685


namespace option_d_correct_l610_610039

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1 / 3) ^ x else Real.logBase 3 x

-- State the theorem
theorem option_d_correct : f (f (1 / 9)) = 9 :=
by
  sorry

end option_d_correct_l610_610039


namespace park_area_l610_610415

variable {x : ℝ}
variable (ratio_length_width : 3 * x / (2 * x) = 3 / 2)
variable (fencing_cost : 10 * x * 0.60 = 150)

theorem park_area (h1 : ratio_length_width) (h2 : fencing_cost) : 6 * x^2 = 3750 := by
  -- Proof here
  sorry

end park_area_l610_610415


namespace angle_between_lines_1_angle_between_lines_2_angle_between_lines_3_angle_between_lines_4_angle_between_lines_5_angle_between_lines_6_angle_between_lines_7_l610_610584

noncomputable def slope (l : ℝ × ℝ × ℝ) : ℝ :=
  -l.1 / l.2

noncomputable def angle_between_lines (k₁ k₂ : ℝ) : ℝ :=
  abs ((k₂ - k₁) / (1 + k₁ * k₂))

theorem angle_between_lines_1 :
  angle_between_lines (-2) (3) = real.arctan (1) := sorry

theorem angle_between_lines_2 :
  angle_between_lines (-2) (-2) = 0 := sorry

theorem angle_between_lines_3 :
  angle_between_lines (real.sqrt 3) (-real.sqrt 3) = real.arctan (real.sqrt 3) := sorry

theorem angle_between_lines_4 :
  angle_between_lines (-2 / 3) (3 / 2) = real.pi / 2 := sorry

theorem angle_between_lines_5 :
  angle_between_lines (-3 / 4) (5 / 2) ≈ real.arctan (-26 / 7) := sorry

theorem angle_between_lines_6 :
  angle_between_lines 2 0 = real.arctan (2) := sorry

theorem angle_between_lines_7 :
  angle_between_lines (1 / 2) (∞) ≈ real.pi / 2 := sorry

end angle_between_lines_1_angle_between_lines_2_angle_between_lines_3_angle_between_lines_4_angle_between_lines_5_angle_between_lines_6_angle_between_lines_7_l610_610584


namespace exists_real_number_a_l610_610231

noncomputable def f (a x : ℝ) : ℝ := log a (1 - (2 / (x + 1)))

theorem exists_real_number_a (m n : ℝ) (h1 : m < n) (h2 : 0 < a) (h3 : a < (3 - 2 * Real.sqrt 2)) : 
  ∃ a ∈ Ioo 0 (3 - 2 * Real.sqrt 2), 
  ∀ x ∈ Icc m n, 
  f a x = 1 + log a x :=
sorry

end exists_real_number_a_l610_610231


namespace farthest_point_l610_610456

def euclidean_distance (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

theorem farthest_point :
  let p0 := (0, 8)
  let p1 := (2, 3)
  let p2 := (4, -5)
  let p3 := (7, 0)
  let p4 := (-3, -3)
  ∀ (p : (ℝ × ℝ)),
    p = p0 ∨ p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4 →
    euclidean_distance p.1 p.2 ≤ euclidean_distance 4 (-5) :=
by
  sorry

end farthest_point_l610_610456


namespace bug_at_vertex_A_after_8_meters_l610_610330

theorem bug_at_vertex_A_after_8_meters (P : ℕ → ℚ) (h₀ : P 0 = 1)
(h : ∀ n, P (n + 1) = 1/3 * (1 - P n)) : 
P 8 = 1823 / 6561 := 
sorry

end bug_at_vertex_A_after_8_meters_l610_610330


namespace nylon_cord_length_approx_nylon_cord_is_approx_9_55_l610_610466

theorem nylon_cord_length_approx (arc_length : Real) (pi_approx : Real) (h : arc_length ≈ 30) (pi_approx ≈ 3.14) :
  Real :=
begin
  let r := 30 / pi_approx,
  exact r,
end

theorem nylon_cord_is_approx_9_55 
  (arc_length : Real) 
  (pi_approx : Real) 
  (h_arc : arc_length ≈ 30) 
  (h_pi : pi_approx ≈ 3.14) : 
  arc_length = π * (30 / pi_approx) :=
by sorry

end nylon_cord_length_approx_nylon_cord_is_approx_9_55_l610_610466


namespace hyperbola_asymptote_ratio_l610_610236

theorem hyperbola_asymptote_ratio (a b : ℝ) (h : a > b) 
  (hyp : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (|x / y| = |a / b + 3 + 2 * real.sqrt 2)) :
  a / b = 3 + 2 * real.sqrt 2 :=
sorry

end hyperbola_asymptote_ratio_l610_610236


namespace school_xx_percentage_increase_l610_610815

theorem school_xx_percentage_increase
  (X Y : ℕ) -- denote the number of students at school XX and YY last year
  (H_Y : Y = 2400) -- condition: school YY had 2400 students last year
  (H_total : X + Y = 4000) -- condition: total number of students last year was 4000
  (H_increase_YY : YY_increase = (3 * Y) / 100) -- condition: 3 percent increase at school YY
  (H_difference : XX_increase = YY_increase + 40) -- condition: school XX grew by 40 more students than YY
  : (XX_increase * 100) / X = 7 :=
by
  sorry

end school_xx_percentage_increase_l610_610815


namespace shortest_edge_of_tetrahedron_l610_610208

theorem shortest_edge_of_tetrahedron (α β : ℝ) (hαβ : α + β = π / 2)
  (AB : ℝ) (hAB : AB = 1) :
  ∃ (CD : ℝ), CD = (sqrt 5 - 1) / 2 ^ (3 / 2) :=
sorry

end shortest_edge_of_tetrahedron_l610_610208


namespace simplify_expression_l610_610792

theorem simplify_expression : (225 / 10125) * 45 = 1 := by
  sorry

end simplify_expression_l610_610792


namespace solve_equation_l610_610581

theorem solve_equation (x : ℝ) : x^4 + (3 - x)^4 = 82 ↔ x = 2.5 ∨ x = 0.5 := by
  sorry

end solve_equation_l610_610581


namespace fraction_of_students_getting_A_l610_610275

theorem fraction_of_students_getting_A
    (frac_B : ℚ := 1/2)
    (frac_C : ℚ := 1/8)
    (frac_D : ℚ := 1/12)
    (frac_F : ℚ := 1/24)
    (passing_grade_frac: ℚ := 0.875) :
    (1 - (frac_B + frac_C + frac_D + frac_F) = 1/8) :=
by
  sorry

end fraction_of_students_getting_A_l610_610275


namespace solve_cryptarithm_l610_610651

namespace cryptarithm

def unique_digits (digits : List ℕ) : Prop := 
  ∀ (i j : ℕ), i ≠ j → digits.nth i ≠ digits.nth j

def cryptarithm_holds (j a l o s e n b : ℕ) : Prop :=
  1000 * j + 100 * a + 10 * l + o +
  1000 * l + 100 * o + 10 * j + a =
  10000 * o + 1000 * s + 100 * e + 10 * n + b

theorem solve_cryptarithm :
  ∃ j a l o s e n b, 
  unique_digits [j, a, l, o, s, e, n, b] ∧
  cryptarithm_holds j a l o s e n b ∧ 
  a = 8 :=
sorry

end cryptarithm

end solve_cryptarithm_l610_610651


namespace fourth_largest_is_three_fifths_l610_610420

theorem fourth_largest_is_three_fifths :
  let lst := [1.7, 1 / 5, 1 / 5, 1, 3 / 5, 3 / 8, 1.4]
  in (lst.sorted (λ x y => y < x)).nth 3 = some (3 / 5) :=
by
  sorry

end fourth_largest_is_three_fifths_l610_610420


namespace find_a_l610_610954

noncomputable def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def problemCondition : ℝ → ℝ := λ x, (Real.exp (2 * x) - Real.exp (a * x)) * Real.cos x

theorem find_a (a : ℝ) (h : isOddFunction (λ x, (Real.exp (2 * x) - Real.exp (a * x)) * Real.cos x)) :
  a = -2 :=
sorry

end find_a_l610_610954


namespace yoongi_caught_frogs_l610_610461

theorem yoongi_caught_frogs (initial_frogs caught_later : ℕ) (h1 : initial_frogs = 5) (h2 : caught_later = 2) : (initial_frogs + caught_later = 7) :=
by
  sorry

end yoongi_caught_frogs_l610_610461


namespace base7_addition_l610_610925

theorem base7_addition (X Y : ℕ) (h1 : X + 5 = 9) (h2 : Y + 2 = 4) : X + Y = 6 :=
by
  sorry

end base7_addition_l610_610925


namespace cole_drive_time_to_work_in_minutes_l610_610862

noncomputable theory

-- Assuming these conditions
variables (D : ℝ) (T_work T_home : ℝ)
variables (avg_speed_work avg_speed_home total_time_work_total_home : ℝ)
variables (avg_speed_work_val avg_speed_home_val total_time_val : Prop)

-- Definitions provided based on conditions
def avg_speed_work := 75
def avg_speed_home := 105
def total_time_work_total_home := 6

axiom distance_def : D = 262.5
axiom time_work_def : T_work = D / avg_speed_work
axiom time_home_def : T_home = D / avg_speed_home
axiom total_time : T_work + T_home = total_time_work_total_home

theorem cole_drive_time_to_work_in_minutes : T_work * 60 = 210 := by
  sorry

end cole_drive_time_to_work_in_minutes_l610_610862


namespace car_discount_l610_610898

variables (P P_b P_s : ℝ) (D : ℝ)
def purchase_price := P * (1 - D / 100)
def selling_price := 2 * purchase_price

theorem car_discount (H1 : P_b = purchase_price) 
    (H2 : P_s = 2 * P_b) 
    (H3 : 2 * (P * (1 - D / 100)) - P = 0.6000000000000001 * P) : 
  D = 20 :=
by
sorry

end car_discount_l610_610898


namespace day_of_week_after_10_pow_90_days_l610_610218

theorem day_of_week_after_10_pow_90_days :
  let initial_day := "Friday"
  ∃ day_after_10_pow_90 : String,
  day_after_10_pow_90 = "Saturday" :=
by
  sorry

end day_of_week_after_10_pow_90_days_l610_610218


namespace school_dance_boys_count_l610_610427

theorem school_dance_boys_count
  (total_attendees : ℕ)
  (percent_faculty_staff : ℝ)
  (fraction_girls : ℝ)
  (h1 : total_attendees = 100)
  (h2 : percent_faculty_staff = 0.1)
  (h3 : fraction_girls = 2/3) :
  let faculty_staff := total_attendees * percent_faculty_staff in
  let students := total_attendees - faculty_staff in
  let girls := students * fraction_girls in
  let boys := students - girls in
  boys = 30 :=
by
  -- Skipping the proof
  sorry

end school_dance_boys_count_l610_610427


namespace white_pairs_coincide_l610_610569

def triangles_in_each_half (red blue white: Nat) : Prop :=
  red = 5 ∧ blue = 6 ∧ white = 9

def folding_over_centerline (r_pairs b_pairs rw_pairs bw_pairs: Nat) : Prop :=
  r_pairs = 3 ∧ b_pairs = 2 ∧ rw_pairs = 3 ∧ bw_pairs = 1

theorem white_pairs_coincide
    (red_triangles blue_triangles white_triangles : Nat)
    (r_pairs b_pairs rw_pairs bw_pairs : Nat) :
    triangles_in_each_half red_triangles blue_triangles white_triangles →
    folding_over_centerline r_pairs b_pairs rw_pairs bw_pairs →
    ∃ coinciding_white_pairs, coinciding_white_pairs = 5 :=
by
  intros half_cond fold_cond
  sorry

end white_pairs_coincide_l610_610569


namespace arithmetic_sequence_21st_term_and_sum_l610_610386

theorem arithmetic_sequence_21st_term_and_sum 
    (a1 : Int)
    (d : Int)
    : a1 = 2 → d = 5 →
      (arithSeqTerm a1 d 21 = 102 ∧ arithSeqSum a1 d 21 = 1092) := by {
    intros,
    sorry
}

end arithmetic_sequence_21st_term_and_sum_l610_610386


namespace sum_of_digits_of_smallest_palindromic_prime_greater_than_300_l610_610467

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def smallest_palindromic_prime_greater_than (m : ℕ) : ℕ :=
  Nat.find_greatest (λ p, p > m ∧ Prime p ∧ is_palindrome p)

theorem sum_of_digits_of_smallest_palindromic_prime_greater_than_300 : 
  ∑ d in (313.digits 10), d = 7 :=
by
  sorry

end sum_of_digits_of_smallest_palindromic_prime_greater_than_300_l610_610467


namespace points_satisfy_diamond_eq_l610_610143

noncomputable def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem points_satisfy_diamond_eq (x y : ℝ) :
  (diamond x y = diamond y x) ↔ ((x = 0) ∨ (y = 0) ∨ (x = y) ∨ (x = -y)) := 
by
  sorry

end points_satisfy_diamond_eq_l610_610143


namespace win_sector_area_l610_610063

theorem win_sector_area (r : ℝ) (h1 : r = 8) (h2 : (1 / 4) = 1 / 4) : 
  ∃ (area : ℝ), area = 16 * Real.pi := 
by
  existsi (16 * Real.pi); exact sorry

end win_sector_area_l610_610063


namespace odd_rows_cols_impossible_arrange_crosses_16x16_l610_610513

-- Define the conditions for part (a)
def square (α : Type*) := α × α
def is_odd_row (table : square nat → bool) (n : nat) :=
  ∃ (i : fin n), ∑ j in finset.range n, table (i, j) = 1
def is_odd_col (table : square nat → bool) (n : nat) :=
  ∃ (j : fin n), ∑ i in finset.range n, table (i, j) = 1

-- Part (a) statement
theorem odd_rows_cols_impossible (table : square nat → bool) (n : nat) :
  n = 16 ∧ (∃ (r : ℕ), r = 20) ∧ (∃ (c : ℕ), c = 15) → ¬(is_odd_row table n ∧ is_odd_col table n) :=
begin 
  -- Sorry placeholder for the proof
  sorry,
end

-- Define the conditions for part (b)
def odd_placement_possible (table : square nat → bool) :=
  ∃ (n : nat), n = 16 ∧ (∑ i in finset.range 16, ∑ j in finset.range 16, table (i, j) = 126) ∧ 
  (∀ i, is_odd_row table 16) ∧ (∀ j, is_odd_col table 16)

-- Part (b) statement
theorem arrange_crosses_16x16 (table : square nat → bool) :
  odd_placement_possible table :=
begin 
  -- Sorry placeholder for the proof
  sorry,
end

end odd_rows_cols_impossible_arrange_crosses_16x16_l610_610513


namespace cube_distance_l610_610879

-- The Lean 4 statement
theorem cube_distance (side_length : ℝ) (h1 h2 h3 : ℝ) (r s t : ℕ) 
  (h1_eq : h1 = 18) (h2_eq : h2 = 20) (h3_eq : h3 = 22) (side_length_eq : side_length = 15) :
  r = 57 ∧ s = 597 ∧ t = 3 ∧ r + s + t = 657 :=
by
  sorry

end cube_distance_l610_610879


namespace inequality_positive_numbers_l610_610603

/-- For positive numbers x_1, x_2, ..., x_n, prove the inequality
(1 + x_1) * (1 + x_1 + x_2) * ... * (1 + x_1 + x_2 + ... + x_n) ≥ (n + 1)^(n + 1) * (x_1 * x_2 * ... * x_n) -/
theorem inequality_positive_numbers (n : ℕ) (x : ℕ → ℝ) (hx : ∀ i, 1 ≤ i → i ≤ n → 0 < x i) :
    (∏ i in range n.succ, ∑ j in range i, x (j + 1) + 1) ^ 2 >=
      (n + 1)^(n + 1) * ∏ i in range n, x (i + 1) := by
  sorry

end inequality_positive_numbers_l610_610603


namespace number_of_reality_shows_l610_610355

-- Definitions based on conditions:
def reality_show_duration := 28
def cartoon_duration := 10
def total_tv_time := 150

-- Problem statement
theorem number_of_reality_shows (R : ℕ) : reality_show_duration * R + cartoon_duration = total_tv_time → R = 5 :=
by 
  -- state the conditions
  intro h,
  -- we proceed with the proof as shown in the solution steps
  sorry

end number_of_reality_shows_l610_610355


namespace surface_area_combination_l610_610825

noncomputable def smallest_surface_area : ℕ :=
  let s1 := 3
  let s2 := 5
  let s3 := 8
  let surface_area := 6 * (s1 * s1 + s2 * s2 + s3 * s3)
  let overlap_area := (s1 * s1) * 4 + (s2 * s2) * 2 
  surface_area - overlap_area

theorem surface_area_combination :
  smallest_surface_area = 502 :=
by
  -- Proof goes here
  sorry

end surface_area_combination_l610_610825


namespace maximum_marked_segments_l610_610905

theorem maximum_marked_segments (n : ℕ) :
  ∃ (m : ℕ), m = n * (n + 1) ∧
  ∀ segments : set (fin n × fin n), 
    (∀ (t : fin n × fin n × fin n), 
      (t.1.1 ≠ t.2.1 ∧ t.1.1 ≠ t.2.2 ∧ t.2.1 ≠ t.2.2) →
      set.card ({t.1, t.2, t.3} ∩ segments) < 3) :=
begin
  sorry
end

end maximum_marked_segments_l610_610905


namespace max_value_of_k_l610_610679

-- Define the conditions
variables {x y k : ℝ}
variables (hx : x > 0) (hy : y > 0) (hk : k > 0)

-- State the problem in terms of Lean definitions.
def given_condition : Prop :=
  4 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)

-- State the theorem to be proven
theorem max_value_of_k (hx : x > 0) (hy : y > 0) (hk : k > 0) :
  given_condition → k ≤ 4 := by
    sorry

end max_value_of_k_l610_610679


namespace area_under_arccos_cos_l610_610160

noncomputable def arccos_cos (x : ℝ) : ℝ := Real.arccos (Real.cos x)

theorem area_under_arccos_cos :
  ∫ (x : ℝ) in 0..(2 * Real.pi), arccos_cos x = Real.pi ^ 2 :=
by
  sorry

end area_under_arccos_cos_l610_610160


namespace hyperbola_eccentricity_range_l610_610959

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  (1 < (Real.sqrt (a^2 + b^2)) / a) ∧ ((Real.sqrt (a^2 + b^2)) / a < (2 * Real.sqrt 3) / 3) :=
sorry

end hyperbola_eccentricity_range_l610_610959


namespace range_of_a_l610_610944

theorem range_of_a (a : ℝ) : (∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + 1 < 0) ↔ a < 1 :=
by
  sorry

end range_of_a_l610_610944


namespace compute_a3_b_neg2_l610_610750

noncomputable def a : ℚ := 4 / 7
noncomputable def b : ℚ := 5 / 6

theorem compute_a3_b_neg2 :
  a^3 * b^(-2) = 2304 / 8575 := 
by
  sorry

end compute_a3_b_neg2_l610_610750


namespace problem_a_impossible_problem_b_possible_l610_610519

-- Definitions based on the given conditions
def is_odd_row (table : ℕ → ℕ → bool) (n : ℕ) (r : ℕ) : Prop :=
  ∑ c in finset.range n, if table r c then 1 else 0 % 2 = 1

def is_odd_column (table : ℕ → ℕ → bool) (n : ℕ) (c : ℕ) : Prop :=
  ∑ r in finset.range n, if table r c then 1 else 0 % 2 = 1

-- Problem(a): No existence of 20 odd rows and 15 odd columns in any square table
theorem problem_a_impossible (table : ℕ → ℕ → bool) (n : ℕ) :
  (∃r_set c_set, r_set.card = 20 ∧ c_set.card = 15 ∧
  ∀ r ∈ r_set, is_odd_row table n r ∧ ∀ c ∈ c_set, is_odd_column table n c) → false :=
sorry

-- Problem(b): Existence of a 16 x 16 table with 126 crosses where all rows and columns are odd
theorem problem_b_possible : 
  ∃ (table : ℕ → ℕ → bool), 
  (∑ r in finset.range 16, ∑ c in finset.range 16, if table r c then 1 else 0) = 126 ∧
  (∀ r, is_odd_row table 16 r) ∧
  (∀ c, is_odd_column table 16 c) :=
sorry

end problem_a_impossible_problem_b_possible_l610_610519


namespace construct_triangle_from_altitudes_l610_610384

noncomputable theory
open_locale classical

-- Define points A1, B1, C1 representing the feet of the altitudes
variables (A1 B1 C1 : Point)

-- Define a structure for a triangle
structure Triangle :=
(A B C : Point)

-- The theorem statement to construct a triangle given the feet of its altitudes
theorem construct_triangle_from_altitudes (A1 B1 C1 : Point) : 
  ∃ (ABC : Triangle), 
  let α1 := 180 - 2 * α in
  let cos_α1 := cos α1 in
  let α := 90 - α1 / 2 in
  let s1 := (a1 + b1 + c1) / 2 in
  let cos_α := sqrt ((s1 - b1) * (s1 - c1) / (b1 * c1)) in
  let a := a1 * sqrt (b1 * c1 / ((s1 - b1) * (s1 - c1))) in
  let b := b1 * sqrt (a1 * c1 / ((s1 - a1) * (s1 - c1))) in
  let c := c1 * sqrt (a1 * b1 / ((s1 - a1) * (s1 - b1))) in
  Triangle A B C :=
begin
  sorry
end

end construct_triangle_from_altitudes_l610_610384


namespace find_general_term_of_sequence_l610_610822

def sum_of_first_n_terms (S : ℕ → ℤ) (f : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = f n

def general_term (a S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = S n - S (n - 1)

theorem find_general_term_of_sequence (S a : ℕ → ℤ) 
    (hS : sum_of_first_n_terms S (λ n, n^2 - 4 * n))
    (h_initial: a 1 = 2 * 1 - 5) : 
    general_term a S ∧ (∀ n : ℕ, a n = 2 * n - 5) :=
by
  sorry

end find_general_term_of_sequence_l610_610822


namespace smallest_n_is_125_l610_610178

noncomputable def smallest_n : ℕ :=
  Inf {n | ∃ (x : Fin n → ℝ), (∑ i, x i = 1000) ∧ (∑ i, (x i)^4 = 512000)}

theorem smallest_n_is_125 : smallest_n = 125 := 
  sorry

end smallest_n_is_125_l610_610178


namespace not_eq_positive_integers_l610_610756

theorem not_eq_positive_integers (a b : ℤ) (ha : a > 0) (hb : b > 0) : 
  a^3 + (a + b)^2 + b ≠ b^3 + a + 2 :=
by {
  sorry
}

end not_eq_positive_integers_l610_610756


namespace number_of_boys_l610_610430

-- Define the conditions
def total_attendees : Nat := 100
def faculty_percentage : Rat := 0.1
def faculty_count : Nat := total_attendees * faculty_percentage
def student_count : Nat := total_attendees - faculty_count
def girls_fraction : Rat := 2 / 3
def girls_count : Nat := student_count * girls_fraction

-- Define the question in terms of a Lean theorem
theorem number_of_boys :
  total_attendees = 100 →
  faculty_percentage = 0.1 →
  faculty_count = 10 →
  student_count = 90 →
  girls_fraction = 2 / 3 →
  girls_count = 60 →
  student_count - girls_count = 30 :=
by
  intros
  sorry -- Skip the proof

end number_of_boys_l610_610430


namespace power_function_odd_l610_610226

variable {a b : ℝ}

theorem power_function_odd (h1 : a^2 - 6 * a + 10 = 1 / 3) (h2 : a^b = 1 / 3) :
  ∃ f : ℝ → ℝ, f = (λ x => x^(-1)) ∧ (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end power_function_odd_l610_610226


namespace part1_part2_l610_610654

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

-- (1) Prove the range of a given f(x) > |1 - 3 * a| always holds for x
theorem part1 (a : ℝ) (h : ∀ x : ℝ, f(x) > |1 - 3 * a|) : -1 < a ∧ a < 5/3 :=
by
  sorry

-- (2) Prove the range of m given the quadratic equation t^2 - 4√2 t + f(m) = 0 has real roots
theorem part2 (m : ℝ) (h : ∃ t t' : ℝ, t^2 - 4 * Real.sqrt 2 * t + f(m) = 0 ∧ t = t') : -3/2 ≤ m ∧ m ≤ 5/2 :=
by
  sorry

end part1_part2_l610_610654


namespace fuel_savings_l610_610920

theorem fuel_savings (old_efficiency new_efficiency : ℝ) (gas_cost diesel_cost : ℝ)
  (h1 : new_efficiency = 1.8 * old_efficiency)
  (h2 : diesel_cost = 1.3 * gas_cost) :
  let old_cost := gas_cost * old_efficiency in
  let new_cost := diesel_cost * (old_efficiency / new_efficiency) in
  (old_cost - new_cost) / old_cost = 0.2778 :=
by
  -- Placeholder for the complex proof
  sorry

end fuel_savings_l610_610920


namespace maximize_area_l610_610091

-- Let x be the length of each side perpendicular to the barn
variable (x : ℝ)

-- Conditions based on the given problem
def total_cost : ℝ := 2400
def cost_per_foot : ℝ := 10
def total_length_fence : ℝ := total_cost / cost_per_foot

-- Length of the side parallel to the barn
def parallel_side_length : ℝ := total_length_fence - 2 * x

-- Area of the pasture
def area (x : ℝ) : ℝ := x * parallel_side_length x

-- Objective: Prove that the length of the side parallel to the barn that maximizes the area equals 120 feet
theorem maximize_area (x_opt : ℝ) (hx_opt : x_opt = 60) : 
  parallel_side_length x_opt = 120 := by
  -- Proof to be provided
  sorry

end maximize_area_l610_610091


namespace sum_of_money_l610_610061

theorem sum_of_money (x : ℝ)
  (hC : 0.50 * x = 64)
  (hB : ∀ x, B_shares = 0.75 * x)
  (hD : ∀ x, D_shares = 0.25 * x) :
  let total_sum := x + 0.75 * x + 0.50 * x + 0.25 * x
  total_sum = 320 :=
by
  sorry

end sum_of_money_l610_610061


namespace total_cost_of_living_room_set_l610_610726

theorem total_cost_of_living_room_set :
  let couch_price := 2500
  let sectional_price := 3500
  let entertainment_center_price := 1500
  let rug_price := 800
  let coffee_table_price := 700
  let accessories_price := 500

  let couch_discount := couch_price * 0.10
  let sectional_discount := sectional_price * 0.10
  let entertainment_center_discount := entertainment_center_price * 0.05
  let rug_discount := rug_price * 0.05
  let coffee_table_discount := coffee_table_price * 0.12
  let accessories_discount := accessories_price * 0.15

  let couch_final := couch_price - couch_discount
  let sectional_final := sectional_price - sectional_discount
  let entertainment_center_final := entertainment_center_price - entertainment_center_discount
  let rug_final := rug_price - rug_discount
  let coffee_table_final := coffee_table_price - coffee_table_discount
  let accessories_final := accessories_price - accessories_discount

  let subtotal := couch_final + sectional_final + entertainment_center_final + rug_final + coffee_table_final + accessories_final

  let sales_tax := subtotal * 0.0825
  let total_before_service_fee := subtotal + sales_tax
  let white_glove_service_fee := 250
  let total_cost := total_before_service_fee + white_glove_service_fee
  in total_cost = 9587.65 :=
by
  sorry

end total_cost_of_living_room_set_l610_610726


namespace quadratic_equal_roots_l610_610268

theorem quadratic_equal_roots (k : ℝ) : (∃ r : ℝ, (r^2 - 2 * r + k = 0)) → k = 1 := 
by
  sorry

end quadratic_equal_roots_l610_610268


namespace collinear_points_a_b_coplanar_points_a_l610_610301

-- Define the points
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define vectors for collinearity condition
def collinear_vectors (a b : ℝ) : Prop :=
  let A : Point3D := ⟨2, a, -1⟩
  let B : Point3D := ⟨-2, 3, b⟩
  let C : Point3D := ⟨1, 2, -2⟩
  ∃ (λ : ℝ), (C.x - A.x) = λ * (C.x - B.x) ∧ (C.y - A.y) = λ * (C.y - B.y) ∧ (C.z - A.z) = λ * (C.z - B.z)

-- Define the main problem for collinearity
theorem collinear_points_a_b (a b : ℝ) : collinear_vectors a b → a = 5 / 3 ∧ b = -5 :=
by
  sorry

-- Define vectors for coplanarity condition
def coplanar_vectors (a b : ℝ) : Prop :=
  let A : Point3D := ⟨2, a, -1⟩
  let B : Point3D := ⟨-2, 3, b⟩
  let C : Point3D := ⟨1, 2, -2⟩
  let D : Point3D := ⟨-1, 3, -3⟩
  let AB : ℝ × ℝ × ℝ := (B.x - A.x, B.y - A.y, B.z - A.z)
  let AC : ℝ × ℝ × ℝ := (C.x - A.x, C.y - A.y, C.z - A.z)
  let AD : ℝ × ℝ × ℝ := (D.x - A.x, D.y - A.y, D.z - A.z)
  let normal : ℝ × ℝ × ℝ := (AC.2 * AD.3 - AC.3 * AD.2, AC.3 * AD.1 - AC.1 * AD.3, AC.1 * AD.2 - AC.2 * AD.1)
  let col_check := normal.1 * AB.1 + normal.2 * AB.2 + normal.3 * AB.3
  col_check = 0 
  
-- Define the main problem for coplanarity
theorem coplanar_points_a (a : ℝ) : coplanar_vectors a (-3) → a = 1 :=
by
  sorry

end collinear_points_a_b_coplanar_points_a_l610_610301


namespace multiple_of_q_capital_l610_610260

variable (P Q R k : ℝ)
variable (totalProfit shareR : ℝ)
variable (H1 : 4 * P = k * Q)
variable (H2 : 4 * P = 10 * R)
variable (H3 : totalProfit = 4030)
variable (H4 : shareR = 780)
variable (H5 : shareR / totalProfit = R / (P + Q + R))

theorem multiple_of_q_capital (H1 : 4 * P = k * Q)
                              (H2 : 4 * P = 10 * R)
                              (H3 : totalProfit = 4030)
                              (H4 : shareR = 780)
                              (H5 : shareR / totalProfit = R / (P + Q + R)) :
  k ≈ 6 :=
by {
  sorry
}

end multiple_of_q_capital_l610_610260


namespace odd_function_m_n_l610_610979

open Function

theorem odd_function_m_n (m n : ℝ) (h1 : ∀ x : ℝ, f (-x) = -f x)
  (h2 : ∀ x : ℝ, f x = x^3 + x + m)
  (h3 : -2 - n + 2 * n = 0) :
  m + n = 2 :=
by
  repeat {sorry}

end odd_function_m_n_l610_610979


namespace van_speed_60_kph_l610_610468

theorem van_speed_60_kph (t₁ : ℕ) (d : ℕ) (h₁ : t₁ = 5) (h₂ : d = 450) :
  let t₂ := (3 / 2 : ℚ) * t₁ in
  let v₂ := d / t₂ in
  v₂ = 60 := by
{
  sorry
}

end van_speed_60_kph_l610_610468


namespace ellipse_properties_l610_610209

noncomputable theory

-- Defining the conditions
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

def P : ℝ × ℝ := (1, 3 / 2)

-- Defining the main problem
theorem ellipse_properties
  (a b : ℝ) (h_ab : a > b > 0)
  (h_ecc : (1 / 2) = ((a^2 - b^2) / a^2).sqrt)
  (h_P_on_C : ellipse_C P.1 P.2)
  (A B : ℝ × ℝ)
  (h_A_on_C : ellipse_C A.1 A.2)
  (h_B_on_C : ellipse_C B.1 B.2)
  (M G H : ℝ × ℝ)
  (h_M_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (h_O : (0, 0) = (0 : ℝ, 0 : ℝ)) :
  ((∀ G H,
    |(G.1 - M.1)| + |(G.2 - M.2)| + |(H.1 - M.1)| + |(H.2 - M.2)| = 2 * sqrt 2)
  ↔ (area_triangle A (0, 0) B = sqrt 3)) := 
sorry

end ellipse_properties_l610_610209


namespace age_ratio_in_4_years_l610_610869

-- Definitions based on conditions
def Age6YearsAgoVimal := 12
def Age6YearsAgoSaroj := 10
def CurrentAgeSaroj := 16
def CurrentAgeVimal := Age6YearsAgoVimal + 6

-- Lean statement to prove the problem
theorem age_ratio_in_4_years (x : ℕ) 
  (h_ratio : (CurrentAgeVimal + x) / (CurrentAgeSaroj + x) = 11 / 10) :
  x = 4 := 
sorry

end age_ratio_in_4_years_l610_610869


namespace evaluate_polynomial_103_l610_610035

theorem evaluate_polynomial_103 :
  103 ^ 4 - 4 * 103 ^ 3 + 6 * 103 ^ 2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end evaluate_polynomial_103_l610_610035


namespace problem_l610_610243

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 3}
def complement_U (S : Set ℕ) : Set ℕ := U \ S

theorem problem : ((complement_U M) ∩ (complement_U N)) = {5} :=
by
  sorry

end problem_l610_610243


namespace quadratic_equation_from_absolute_value_l610_610558

theorem quadratic_equation_from_absolute_value :
  ∃ b c : ℝ, (∀ x : ℝ, |x - 8| = 3 ↔ x^2 + b * x + c = 0) ∧ (b, c) = (-16, 55) :=
sorry

end quadratic_equation_from_absolute_value_l610_610558


namespace near_A_4_order_golden_section_point_l610_610903

def is_n_order_golden_section_point (A B C : ℝ) (n : ℕ) (k_n : ℝ) : Prop :=
  (B - C) / (real.sqrt n * (C - A)) = k_n ∧
  (real.sqrt n * (C - A)) / (B - A) = k_n

theorem near_A_4_order_golden_section_point (A B C : ℝ) (h : A < C ∧ C < B) :
  is_n_order_golden_section_point A B C 4 (real.sqrt 17 - 1) / 4 :=
sorry

end near_A_4_order_golden_section_point_l610_610903


namespace minimum_cards_four_of_a_kind_l610_610853

def rank := {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14} -- 11: "J", 12: "Q", 13: "K", 14: "A"
def card (n : ℕ) := if n ∈ rank then (1 : Type) else (0 : Type) -- Each rank has 4 cards

def standard_deck := fin 52
def jokers := {53, 54}

theorem minimum_cards_four_of_a_kind :
  ∀ (draw_count : ℕ), draw_count = 42 →
  (∀ drawn_cards : fin draw_count → fin 54,  
    ∃ r ∈ rank, ∃ c, set.count (drawn_cards (set.univ.filter (λ drawn_card, drawn_card = card r))) = 4) :=
sorry

end minimum_cards_four_of_a_kind_l610_610853


namespace range_of_m_l610_610615

-- Definition of p: x / (x - 2) < 0 implies 0 < x < 2
def p (x : ℝ) : Prop := x / (x - 2) < 0

-- Definition of q: 0 < x < m
def q (x m : ℝ) : Prop := 0 < x ∧ x < m

-- Main theorem: If p is a necessary but not sufficient condition for q to hold, then the range of m is (2, +∞)
theorem range_of_m {m : ℝ} (h : ∀ x, p x → q x m) (hs : ∃ x, ¬(q x m) ∧ p x) : 
  2 < m :=
sorry

end range_of_m_l610_610615


namespace least_number_divisible_by_forth_number_l610_610399

theorem least_number_divisible_by_forth_number
  (n : ℕ) (h1 : n = 856)
  (h2 : (n + 8) % 24 = 0)
  (h3 : (n + 8) % 32 = 0)
  (h4 : (n + 8) % 36 = 0) :
  ∃ x, x = 3 ∧ (n + 8) % x = 0 :=
by
  sorry

end least_number_divisible_by_forth_number_l610_610399


namespace original_board_length_before_final_cut_l610_610438

-- Given conditions
def initial_length : ℕ := 143
def first_cut_length : ℕ := 25
def final_cut_length : ℕ := 7

def board_length_after_first_cut : ℕ := initial_length - first_cut_length
def board_length_after_final_cut : ℕ := board_length_after_first_cut - final_cut_length

-- The theorem to be proved
theorem original_board_length_before_final_cut : board_length_after_first_cut + final_cut_length = 125 :=
by
  sorry

end original_board_length_before_final_cut_l610_610438


namespace combination_sum_eq_l610_610453

theorem combination_sum_eq : 
  let C (n k : ℕ) := nat.choose n k in
  C 6 3 + C 6 2 = C 7 3 :=
by
  sorry

end combination_sum_eq_l610_610453


namespace twenty_odd_rows_fifteen_odd_cols_impossible_sixteen_by_sixteen_with_126_crosses_possible_l610_610523

-- Conditions and helper definitions
def is_odd (n: ℕ) := n % 2 = 1
def count_odd_rows (table: ℕ × ℕ → bool) (n: ℕ) := 
  (List.range n).countp (λ i, is_odd ((List.range n).countp (λ j, table (i, j))))
  
def count_odd_columns (table: ℕ × ℕ → bool) (n: ℕ) := 
  (List.range n).countp (λ j, is_odd ((List.range n).countp (λ i, table (i, j))))

-- a) Proof problem statement
theorem twenty_odd_rows_fifteen_odd_cols_impossible (n: ℕ): 
  ∀ (table: ℕ × ℕ → bool), count_odd_rows table n = 20 → count_odd_columns table n = 15 → False := 
begin
  intros table h_rows h_cols,
  sorry
end

-- b) Proof problem statement
theorem sixteen_by_sixteen_with_126_crosses_possible :
  ∃ (table: ℕ × ℕ → bool), count_odd_rows table 16 = 16 ∧ count_odd_columns table 16 = 16 ∧ 
  (List.range 16).sum (λ i, (List.range 16).countp (λ j, table (i, j))) = 126 :=
begin
  sorry
end

end twenty_odd_rows_fifteen_odd_cols_impossible_sixteen_by_sixteen_with_126_crosses_possible_l610_610523


namespace quadrilaterals_similar_l610_610188

section QuadrilateralSimilarity

variables {A B C D A₁ B₁ C₁ D₁ : Type*} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
           [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D]
           (BD AC : Subtype (convex_hull ℝ (set.range (coe : A → ℝ^2))))

-- Conditions
variables (h1 : ∀a, a ∈ set.range A → ∃(a1 : A₁), is_foot_of_perpendicular a AC a1)
variables (h2 : ∀b, b ∈ set.range B → ∃(b1 : B₁), is_foot_of_perpendicular b BD b1)
variables (h3 : ∀c, c ∈ set.range C → ∃(c1 : C₁), is_foot_of_perpendicular c AC c1)
variables (h4 : ∀d, d ∈ set.range D → ∃(d1 : D₁), is_foot_of_perpendicular d BD d1)

-- Proof to show that quadrilateral formed by the feet of the perpendiculars is similar to the original quadrilateral
theorem quadrilaterals_similar (α : ℝ) :
  similar (A₁₀ B₁₁ C₁₀ D₁₁) (A B C D) :=
sorry

end QuadrilateralSimilarity

end quadrilaterals_similar_l610_610188


namespace triangle_area_given_conditions_l610_610304

theorem triangle_area_given_conditions (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6) (h2 : C = Real.pi / 3) : 
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end triangle_area_given_conditions_l610_610304


namespace divide_equilateral_into_acute_scalene_triangles_l610_610777

-- Define an equilateral triangle with vertices A, B, C
structure EquilateralTriangle (α : Type _) :=
(A B C : α)
(is_equilateral : dist A B = dist B C ∧ dist B C = dist C A)

-- Define the midpoints of the sides of the equilateral triangle
def midpoint {α : Type _} [MetricSpace α] (x y : α) : α :=
sorry -- Standard definition of midpoint; implementation is omitted.

-- Define the problem: Prove that the triangle can be divided into 4 scalene and acute triangles
theorem divide_equilateral_into_acute_scalene_triangles 
    {α : Type _} [MetricSpace α] 
    (T : EquilateralTriangle α) : 
    ∃ (M N P : α), 
    midpoint T.A T.B = M ∧ 
    midpoint T.B T.C = N ∧ 
    midpoint T.A T.C = P ∧ 
    ∀ (T1 T2 T3 T4 : Triangle α),
    divides_by_midpoints T.A T.B T.C M N P T1 T2 T3 T4 → 
    (is_acute T1 ∧ is_acute T2 ∧ is_acute T3 ∧ is_acute T4) ∧ 
    (non_equilateral T1 ∧ non_equilateral T2 ∧ non_equilateral T3 ∧ non_equilateral T4) :=
sorry -- Proof is not provided here.

end divide_equilateral_into_acute_scalene_triangles_l610_610777


namespace union_A_B_at_a_3_inter_B_compl_A_at_a_3_B_subset_A_imp_a_range_l610_610945

open Set

variables (U : Set ℝ) (A B : Set ℝ) (a : ℝ)

def A_def : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }
def B_def (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ a + 2 }
def comp_U_A : Set ℝ := { x | x < 1 ∨ x > 4 }

theorem union_A_B_at_a_3 (h : a = 3) :
  A_def ∪ B_def 3 = { x | 1 ≤ x ∧ x ≤ 5 } :=
sorry

theorem inter_B_compl_A_at_a_3 (h : a = 3) :
  B_def 3 ∩ comp_U_A = { x | 4 < x ∧ x ≤ 5 } :=
sorry

theorem B_subset_A_imp_a_range (h : B_def a ⊆ A_def) :
  1 ≤ a ∧ a ≤ 2 :=
sorry

end union_A_B_at_a_3_inter_B_compl_A_at_a_3_B_subset_A_imp_a_range_l610_610945


namespace sum_of_solutions_l610_610559

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
if x < 3 then
  5 * x + 20
else
  3 * x - 15

-- Define the condition for x we need to check
def x_valid (x : ℝ) : Prop :=
f(x) = 10

-- Define the sum of two specific values of x
def sum_of_x : ℝ := -2 + 25 / 3

-- The theorem we need to prove
theorem sum_of_solutions : ∑ x in {x | x_valid x}.to_finset, x = sum_of_x :=
  sorry

end sum_of_solutions_l610_610559


namespace nancy_chips_left_l610_610766

-- Define the initial conditions and parameters
def initial_chips : ℝ := 50
def chips_to_brother : ℝ := 12.5
def fraction_to_sister : ℝ := 1 / 3
def percentage_to_cousin : ℝ := 25 / 100

-- Define the theorem stating the final number of chips left for Nancy
theorem nancy_chips_left : 
  let remaining_after_brother := initial_chips - chips_to_brother in
  let chips_to_sister := fraction_to_sister * remaining_after_brother in
  let remaining_after_sister := remaining_after_brother - chips_to_sister in
  let chips_to_cousin := percentage_to_cousin * remaining_after_sister in
  let remaining_after_cousin := remaining_after_sister - chips_to_cousin in
  remaining_after_cousin = 18.75 :=
by {
  sorry
}

end nancy_chips_left_l610_610766


namespace tangent_line_eq_l610_610204

def f (x : ℝ) : ℝ := 2 * f (2 - x) + Real.exp (x - 1) + x^2

theorem tangent_line_eq : 
  let f (x : ℝ) := -(1 / 3) * (2 * Real.exp (1 - x) + Real.exp (x - 1) + 3 * x^2 - 8 * x + 8) in
  let f_prime (x : ℝ) := -(1 / 3) * (2 * Real.exp (1 - x) + Real.exp (x - 1) + 6 * x - 8) in
  (f 1 = -2) → (f_prime 1 = 1) → ∃ m b, (m = f_prime 1) ∧ (b = f 1) ∧ (∀ x, m * (x - 1) + b = x - 3) :=
by
  sorry

end tangent_line_eq_l610_610204


namespace integral_result_l610_610151

theorem integral_result :
  ∫ x in 0..1, (Real.sqrt (1 - (x - 1)^2) - 1) = (Real.pi / 4 - 1) := 
sorry

end integral_result_l610_610151


namespace locus_of_point_Q_eqn_l610_610432

theorem locus_of_point_Q_eqn
  (p : ℝ) (h₀ : p > 0)
  (λ₁ λ₂ : ℝ) (h₁ : λ₁ > 0) (h₂ : λ₂ > 0)
  (h_cond : 2 / λ₁ + 3 / λ₂ = 15)
  : ∀ (x y : ℝ), (y^2 = 2 * p * x) →
                 (λx = -4) →
                 (B : ℝ × ℝ) (hB : B = (1, 2)) →
                 (D : ℝ × ℝ) (hD : D.2 = 0) →
                 (P : ℝ × ℝ) (hP : P ≠ B) →
                 (E : ℝ × ℝ) (hE : E = (P.1 + λ₁ * (-4 - P.1), P.2 + λ₁ * (0 - P.2))) →
                 (F : ℝ × ℝ) (hF : F = (P.1 + λ₂ * (1 - P.1), P.2 + λ₂ * (2 - P.2))) →
                 (Q : ℝ × ℝ) (hQ : ∃ k : ℝ, k * (D.1 - P.1) = E.1 - F.1 ∧ k * (D.2 - P.2) = E.2 - F.2) →
                 y^2 = (8 / 3) * (x + (1 / 3)) 
                 :=
sorry

end locus_of_point_Q_eqn_l610_610432


namespace greatest_four_digit_multiple_of_17_l610_610019

theorem greatest_four_digit_multiple_of_17 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n ∧ n = 9996 :=
by
  use 9996
  split
  { exact by norm_num }
  split
  { exact by norm_num [lt.sub (ofNat 10000) (ofNat 4)] }
  split
  { exact by norm_num }
  { exact by norm_num [dvd_of_mod_eq_zero (by norm_num : 10000 % 17 = 4)] }

end greatest_four_digit_multiple_of_17_l610_610019


namespace calculate_expression_I_calculate_expression_II_l610_610131

theorem calculate_expression_I :
  (1 + 9 / 16)^(1 / 2) + (0.01)^(-1) + (2 + 10 / 27)^(-2 / 3) - 2 * Real.pi^0 + 3 / 16 = 100 :=
by
  sorry

theorem calculate_expression_II :
  Real.log 2 (4 * 8^2) + Real.log 3 18 - Real.log 3 2 + (Real.log 4 3 * Real.log 3 16) = 12 :=
by
  sorry

end calculate_expression_I_calculate_expression_II_l610_610131


namespace city_raised_money_for_charity_l610_610379

-- Definitions based on conditions from part a)
def price_regular_duck : ℝ := 3.0
def price_large_duck : ℝ := 5.0
def number_regular_ducks_sold : ℕ := 221
def number_large_ducks_sold : ℕ := 185

-- Definition to represent the main theorem: Total money raised
noncomputable def total_money_raised : ℝ :=
  price_regular_duck * number_regular_ducks_sold + price_large_duck * number_large_ducks_sold

-- Theorem to prove that the total money raised is $1588.00
theorem city_raised_money_for_charity : total_money_raised = 1588.0 := by
  sorry

end city_raised_money_for_charity_l610_610379


namespace cos_inequality_for_acute_angles_l610_610594

theorem cos_inequality_for_acute_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  cos (α + β) < cos α + cos β := 
sorry

end cos_inequality_for_acute_angles_l610_610594


namespace least_possible_value_of_f_1998_l610_610913

-- Definitions to work with functions from positive integers to positive integers
def f : ℕ+ → ℕ+ → ℕ+ := sorry

-- Condition given in the problem
axiom condition (f : ℕ+ → ℕ+ → ℕ+) : ∀ s t : ℕ+, f t^2 (f s) = (s * (f t)^2)

-- Prove the least possible value of f(1998) is 120
theorem least_possible_value_of_f_1998 (f : ℕ+ → ℕ+ → ℕ+) :
  (∀ s t : ℕ+, f t^2 (f s) = (s * (f t)^2)) → f 1998 = 120 :=
by
  intro h
  have h1 := condition f
  sorry

end least_possible_value_of_f_1998_l610_610913


namespace angle_AKO_eq_angle_DAC_l610_610629

theorem angle_AKO_eq_angle_DAC
  (ABC_acute : IsAcuteTriangle A B C)
  (O_circum : IsCircumcircle O A B C)
  (H_orthocenter : IsOrthocenter H A B C)
  (D_on_circum : OnCircumcircle D O)
  (K_on_AB : ∃ K, IsPerpendicularBisector K (D, H) ∧ K ∈ Line(A, B)) :
  ∠AKO = ∠DAC := 
sorry

end angle_AKO_eq_angle_DAC_l610_610629


namespace not_periodic_l610_610199

noncomputable def a_n (x : ℝ) (n : ℕ) : ℝ :=
  (floor (x^(n+1))) - x * (floor (x^n))

theorem not_periodic (x : ℝ) (h1 : x > 1) (h2 : ¬ (∃ n : ℤ, x = n)) : 
  ¬ (∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, a_n x (n + p) = a_n x n) :=
by
  sorry

end not_periodic_l610_610199


namespace area_ratio_independent_l610_610762

-- Definitions related to the problem
variables (AB BC CD : ℝ) (e f g : ℝ)

-- Let the lengths be defined as follows
def AB_def : Prop := AB = 2 * e
def BC_def : Prop := BC = 2 * f
def CD_def : Prop := CD = 2 * g

-- Let the areas be defined as follows
def area_quadrilateral (e f g : ℝ) : ℝ :=
  2 * (e + f) * (f + g)

def area_enclosed (e f g : ℝ) : ℝ :=
  (e + f + g) ^ 2 + f ^ 2 - e ^ 2 - g ^ 2

-- Prove the ratio is 2 / π
theorem area_ratio_independent (e f g : ℝ) (h1 : AB_def AB e)
  (h2 : BC_def BC f) (h3 : CD_def CD g) :
  (area_quadrilateral e f g) / ((area_enclosed e f g) * (π / 2)) = 2 / π :=
by
  sorry

end area_ratio_independent_l610_610762


namespace max_wind_power_speed_l610_610396

noncomputable def force (C S ρ v0 v : ℝ): ℝ :=
  (C * S * ρ * (v0 - v)^2) / 2

noncomputable def power (C S ρ v0 v : ℝ): ℝ :=
  force C S ρ v0 v * v

theorem max_wind_power_speed: ∀ (C ρ: ℝ), 
  power C 5 ρ 6 2 = N
where 
  N := power C 5 ρ 6 2 :=
by
  sorry

end max_wind_power_speed_l610_610396


namespace total_cost_of_cloth_l610_610727

/-- Define the length of the cloth in meters --/
def length_of_cloth : ℝ := 9.25

/-- Define the cost per meter in dollars --/
def cost_per_meter : ℝ := 46

/-- Theorem stating that the total cost is $425.50 given the length and cost per meter --/
theorem total_cost_of_cloth : length_of_cloth * cost_per_meter = 425.50 := by
  sorry

end total_cost_of_cloth_l610_610727


namespace roots_triple_relationship_l610_610598

variable {α : Type}

-- Lean Definitions equivalent to Lean Condition
def quadratic_equation (a b c x : α) [Field α] : Prop :=
  a * x^2 + b * x + c = 0

-- Lean proof problem
theorem roots_triple_relationship (a b c α β : α) [Field α] (h_quad : quadratic_equation a b c α)
  (h_beta : β = 3 * α)
  (h_vieta1 : α + β = -b / a)
  (h_vieta2 : α * β = c / a) :
  3 * b^2 = 16 * a * c :=
sorry

end roots_triple_relationship_l610_610598


namespace compute_exp_l610_610752

theorem compute_exp (a b : ℚ) (ha : a = 4 / 7) (hb : b = 5 / 6) : a^3 * b^(-2) = 2304 / 8575 := by
  sorry

end compute_exp_l610_610752


namespace sum_of_geometric_progression_l610_610980

theorem sum_of_geometric_progression (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (a1 a3 : ℝ) (h1 : a1 + a3 = 5) (h2 : a1 * a3 = 4)
  (h3 : a 1 = a1) (h4 : a 3 = a3)
  (h5 : ∀ k, a (k + 1) > a k)  -- Sequence is increasing
  (h6 : S n = a 1 * ((1 - (2:ℝ) ^ n) / (1 - 2)))
  (h7 : n = 6) :
  S 6 = 63 :=
sorry

end sum_of_geometric_progression_l610_610980


namespace distribute_candy_bars_l610_610799

theorem distribute_candy_bars (candies bags : ℕ) (h1 : candies = 15) (h2 : bags = 5) :
  candies / bags = 3 :=
by
  sorry

end distribute_candy_bars_l610_610799


namespace problem1_problem2_l610_610868

theorem problem1
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 :=
sorry

theorem problem2
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  real.sqrt (x^2 + y^2) > real.cbrt (x^3 + y^3) :=
sorry

end problem1_problem2_l610_610868


namespace george_slices_l610_610096

def num_small_pizzas := 3
def num_large_pizzas := 2
def slices_per_small_pizza := 4
def slices_per_large_pizza := 8
def slices_leftover := 10
def slices_per_person := 3
def total_pizza_slices := (num_small_pizzas * slices_per_small_pizza) + (num_large_pizzas * slices_per_large_pizza)
def slices_eaten := total_pizza_slices - slices_leftover
def G := 6 -- Slices George would like to eat

theorem george_slices :
  G + (G + 1) + ((G + 1) / 2) + (3 * slices_per_person) = slices_eaten :=
by
  sorry

end george_slices_l610_610096


namespace line_tangent_to_circle_l610_610886

noncomputable def line_rotated (x y : ℝ) : Prop := sqrt 3 * x - y = 0

noncomputable def circle (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 3

noncomputable def center_circle : ℝ × ℝ := (2, 0)

noncomputable def distance_to_center (line : ℝ → ℝ → Prop) (center : ℝ × ℝ) : ℝ :=
  -- Given that A * x₀ + B * y₀ + C = 0 and A = sqrt(3), B = -1, C = 0 for the line sqrt(3)x - y = 0
  float.abs (sqrt 3 * center.1 - center.2) / real.sqrt (3 + 1)

theorem line_tangent_to_circle : 
  float.abs (sqrt 3 * (2:ℝ)) / real.sqrt (3 + 1) = real.sqrt 3 → Prop :=
begin
  intro h,
  rw h,
  unfold center_circle,
  unfold circle,
  unfold line_rotated,
  sorry
end

end line_tangent_to_circle_l610_610886


namespace ak_squared_eq_kl_km_l610_610772

-- Define the problem in Lean
variables {A B C D K L M : Type} [AddCommGroup A] [AffineSpace A]

-- Define A, B, C, D as points forming a parallelogram
def is_parallelogram (A B C D : A) : Prop :=
  vector.to_param (A, B, C, D) ≠ 0 ∧
  dist A B = dist C D ∧ 
  dist A D = dist B C ∧
  dist A B * dist A D = dist A C * dist A D

-- Define a point K on diagonal BD
def on_diagonal (B D K : A) : Prop :=
  K ∈ line_through B D

-- Define point L on CD and M on BC
def intersects (A K C D B L M : A) : Prop :=
  K ∈ line_through A D ∧ L ∈ line_join C D ∧ M ∈ line_join B C

theorem ak_squared_eq_kl_km 
  (A B C D K L M : A) 
  [AddCommGroup A] [AffineSpace A]
  (hParallelogram : is_parallelogram A B C D)
  (hOnDiagonal : on_diagonal B D K)
  (hIntersects : intersects A K C D B L M) :
  dist A K ^ 2 = dist K L * dist K M :=
sorry

end ak_squared_eq_kl_km_l610_610772


namespace probability_obtuse_given_conditions_l610_610774

noncomputable def point (x y : ℝ) : Type := (x, y)

def A : point := point 0 3
def B : point := point 5 0
def C : point := point (2 * Real.pi + 2) 0
def D : point := point (2 * Real.pi + 2) 5
def E : point := point 0 5

def in_pentagon (P : point) : Prop :=
  -- Define that P is inside the pentagon using its vertices
  sorry

def north_of_line_AB (P : point) : Prop :=
  -- Define the condition that P is north of line AB
  sorry

def angle_APB_is_obtuse (P : point) : Prop :=
  -- Define the condition that ∠APB is obtuse
  sorry

def probability_obtuse_angle_given_conditions : ℝ :=
  -- Define the probability as the area ratio
  sorry

theorem probability_obtuse_given_conditions :
  probability_obtuse_angle_given_conditions = 5 / (20 + 35 / Real.pi) :=
sorry

end probability_obtuse_given_conditions_l610_610774


namespace max_wind_power_speed_l610_610395

noncomputable def force (C S ρ v0 v : ℝ): ℝ :=
  (C * S * ρ * (v0 - v)^2) / 2

noncomputable def power (C S ρ v0 v : ℝ): ℝ :=
  force C S ρ v0 v * v

theorem max_wind_power_speed: ∀ (C ρ: ℝ), 
  power C 5 ρ 6 2 = N
where 
  N := power C 5 ρ 6 2 :=
by
  sorry

end max_wind_power_speed_l610_610395


namespace perfect_square_subset_exists_l610_610361

noncomputable def prime_factors (p : ℕ) (a : ℕ) : ℕ :=
  if a = 0 then 0 else (nat.factors a).count p

theorem perfect_square_subset_exists (n : ℕ) (primes : fin n → ℕ) (a : fin (n+1) → ℕ) 
  (h : ∀ i, ∀ p ∈ primes, prime_factors p (a i) > 0) : 
  ∃ (s : finset (fin (n+1))), ∏ i in s, a i = k^2 for some k : ℕ :=
sorry

end perfect_square_subset_exists_l610_610361


namespace find_m_from_inequality_l610_610997

theorem find_m_from_inequality :
  (∀ x, x^2 - (m+2)*x > 0 ↔ (x < 0 ∨ x > 2)) → m = 0 :=
by
  sorry

end find_m_from_inequality_l610_610997


namespace bin_subtraction_binary_conversion_binary_subtraction_correct_l610_610546

/- Definitions for binary numbers -/
def bin11011 : ℕ := 27
def bin101 : ℕ := 5
def binResult : ℕ := 22

/- Binary subtraction -/
theorem bin_subtraction : bin11011 - bin101 = binResult := by
  -- Leave proof as an exercise
  sorry

/- Binary conversion -/
theorem binary_conversion : nat.binary_repr binResult = "10110" := by
  -- Leave proof as an exercise
  sorry

/- Combining the two theorems -/
theorem binary_subtraction_correct : nat.binary_repr (bin11011 - bin101) = "10110" := by
  have h₁ : bin11011 - bin101 = binResult := bin_subtraction
  have h₂ : nat.binary_repr binResult = "10110" := binary_conversion
  rw [h₁, h₂]
  sorry

end bin_subtraction_binary_conversion_binary_subtraction_correct_l610_610546


namespace smallest_n_l610_610918

noncomputable def P (n : ℕ) : ℚ := 1 / (n * (n^2 + 1))

theorem smallest_n (h: ∃ n : ℕ, P(n) < 1/3000): ∃ n : ℕ, n = 15 := by
  use 15
  have : P(15) < 1 / 3000 := sorry
  assumption

end smallest_n_l610_610918


namespace number_of_paths_from_A_to_B_l610_610556

-- Define the points A, B, C, D, E, F, G as constants
constant A B C D E F G : Type

-- Define the connections between points as given in the problem
def segments : List (Type × Type) :=
  [(A, C), (A, D), (A, F), (A, G), (B, C), (B, F), (B, G), 
   (C, D), (C, E), (C, F), (D, E), (D, G), (E, F), (E, G), (F, G)]
   
-- Define the set of points
def points : List Type := [A, B, C, D, E, F, G]

-- Define the problem to prove the number of paths from A to B
theorem number_of_paths_from_A_to_B : 
  (number_of_continuous_paths_no_revisit A B points segments) = 16 :=
sorry

end number_of_paths_from_A_to_B_l610_610556


namespace minimum_value_l610_610947

/-- 
Given \(a > 0\), \(b > 0\), and \(a + 2b = 1\),
prove that the minimum value of \(\frac{2}{a} + \frac{1}{b}\) is 8.
-/
theorem minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2 * b = 1) : 
  (∀ a b : ℝ, (a > 0) → (b > 0) → (a + 2 * b = 1) → (∃ c : ℝ, c = 8 ∧ ∀ x y : ℝ, (x = a) → (y = b) → (c ≤ (2 / x) + (1 / y)))) :=
sorry

end minimum_value_l610_610947


namespace pyramid_radius_to_height_ratio_l610_610118

-- Define the problem conditions
variable (a : ℝ) (r h : ℝ)

-- Define the problem statement
theorem pyramid_radius_to_height_ratio (h_eq : h = a * sqrt 6 / 3) (r_eq : r = h / 4) : r / h = 1 / 4 :=
by 
  -- This would be normally where the proof steps are written
  sorry

end pyramid_radius_to_height_ratio_l610_610118


namespace right_triangle_median_hypotenuse_l610_610227

theorem right_triangle_median_hypotenuse
  (a b : ℝ)
  (h : (√(a - 5) + |b - 12| = 0))
  (hypotenuse : ℝ := real.sqrt (a^2 + b^2)) :
  hypotenuse / 2 = 13 / 2 :=
by
  have ha : a = 5, from sorry
  have hb : b = 12, from sorry
  rw [ha, hb]
  have hc : hypotenuse = 13, from sorry
  exact sorry

end right_triangle_median_hypotenuse_l610_610227


namespace original_length_before_final_cut_l610_610440

-- Defining the initial length of the board
def initial_length : ℕ := 143

-- Defining the length after the first cut
def length_after_first_cut : ℕ := initial_length - 25

-- Defining the length after the final cut
def length_after_final_cut : ℕ := length_after_first_cut - 7

-- Stating the theorem to prove that the original length of the board before cutting the final 7 cm is 125 cm
theorem original_length_before_final_cut : initial_length - 25 + 7 = 125 :=
sorry

end original_length_before_final_cut_l610_610440


namespace direction_vector_correct_l610_610814

noncomputable def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ := 
  ![
    ![3/5, 4/5],
    ![4/5, -3/5]
  ]

noncomputable def direction_vector : ℚ × ℚ := (2, 1)

theorem direction_vector_correct : reflection_matrix.mul_vec (2, 1) = (2, 1) ∧ (Nat.gcd 2 1 = 1) := by
  sorry

end direction_vector_correct_l610_610814


namespace not_minimum_on_l610_610996

noncomputable def f (x m : ℝ) : ℝ :=
  x * Real.exp x - (m / 2) * x ^ 2 - m * x

theorem not_minimum_on (m : ℝ) : 
  ¬ (∃ x ∈ Set.Icc 1 2, f x m = Real.exp 2 - 2 * m ∧ 
  ∀ y ∈ Set.Icc 1 2, f y m ≥ f x m) :=
sorry

end not_minimum_on_l610_610996


namespace cannot_return_to_initial_positions_l610_610490

-- definition of the points A, B, C and X
variables {A B C X : Type}
variable [EuclideanGeometry]

-- conditions extracted from the problem statement
variables {triangle_ABC : A × B × C}
variable (X_in_ABC : PointInTriangle X triangle_ABC)
variable (policemen_positions : T : Set (A, B, C))
variable (equidistant_replacement : A → B → C → A)
variable (policemen_central_replacement : ∀ {A B C : T}, equidistant_replacement A B C)

-- theorem to state the proof problem
theorem cannot_return_to_initial_positions
  (H : ∀ t : Time, (repeated_operations t).orig_positions ≠ (initial_positions ABC)) : 
  -- state that ABC cannot revert back to being the policemen positions after repeated operations
  ∀ t : Time, ¬ (policemen_positions t = initial_positions ABC) :=
begin
  sorry -- skip the proof
end

end cannot_return_to_initial_positions_l610_610490


namespace b_2_pow_100_value_l610_610318

def seq (b : ℕ → ℕ) : Prop :=
  b 1 = 3 ∧ ∀ n > 0, b (2 * n) = 2 * n * b n

theorem b_2_pow_100_value
  (b : ℕ → ℕ)
  (h_seq : seq b) :
  b (2^100) = 2^5050 * 3 :=
by
  sorry

end b_2_pow_100_value_l610_610318


namespace size_of_angle_A_length_of_side_a_l610_610694

-- Definitions
variables {A B C : ℝ} {a b c : ℝ} 

-- Conditions
axiom triangle_sides_opposite : ∀ {A B C : ℝ} {a b c : ℝ}, (a : ℝ) = (some (λ x, abs x = abs (sin B))) ∧ 
  (b : ℝ) = (some (λ x, abs x = abs (sin C))) ∧ (c : ℝ) = (some (λ x, abs x = abs (sin A)))

axiom condition_1 : b * cos A + a * sin B = 0
axiom condition_2 : b + c = 2 + sqrt 2
axiom condition_3 : 1 / 2 * b * c * sin A = 1

-- Proof statement for size of angle A
theorem size_of_angle_A : A = 3 * real.pi / 4 :=
  sorry

-- Proof statement for length of side a
theorem length_of_side_a : a = √10 :=
  sorry

end size_of_angle_A_length_of_side_a_l610_610694


namespace total_payroll_l610_610531

theorem total_payroll 
  (heavy_operator_pay : ℕ) 
  (laborer_pay : ℕ) 
  (total_people : ℕ) 
  (laborers : ℕ)
  (heavy_operators : ℕ)
  (total_payroll : ℕ)
  (h1: heavy_operator_pay = 140)
  (h2: laborer_pay = 90)
  (h3: total_people = 35)
  (h4: laborers = 19)
  (h5: heavy_operators = total_people - laborers)
  (h6: total_payroll = (heavy_operators * heavy_operator_pay) + (laborers * laborer_pay)) :
  total_payroll = 3950 :=
by sorry

end total_payroll_l610_610531


namespace distance_they_both_run_l610_610487

theorem distance_they_both_run
  (time_A time_B : ℕ)
  (distance_advantage: ℝ)
  (speed_A speed_B : ℝ)
  (D : ℝ) :
  time_A = 198 →
  time_B = 220 →
  distance_advantage = 300 →
  speed_A = D / time_A →
  speed_B = D / time_B →
  speed_A * time_B = D + distance_advantage →
  D = 2700 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end distance_they_both_run_l610_610487


namespace technicians_in_workshop_l610_610377

theorem technicians_in_workshop (T R : ℕ) 
    (h1 : 700 * 15 = 800 * T + 650 * R)
    (h2 : T + R = 15) : T = 5 := 
by
  sorry

end technicians_in_workshop_l610_610377


namespace sphere_surface_area_l610_610691

noncomputable def volume (R : ℝ) : ℝ := (4 * π * R^3) / 3
noncomputable def surface_area (R : ℝ) : ℝ := 4 * π * R^2

theorem sphere_surface_area (h : volume (sqrt 3) = 4 * sqrt 3 * π) : surface_area (sqrt 3) = 12 * π :=
by
  sorry

end sphere_surface_area_l610_610691


namespace eighth_term_is_66_l610_610385

noncomputable def sequence : ℕ → ℕ
| 0     := 1
| 1     := 4
| 2     := 2
| 3     := 3
| n + 4 := sequence n + sequence (n + 1) + sequence (n + 2) + sequence (n + 3)

theorem eighth_term_is_66 : sequence 7 = 66 :=
by sorry

end eighth_term_is_66_l610_610385


namespace lambda_div_mu_eq_neg_half_l610_610214

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (O A B C : V)
variables (λ μ : ℝ)
variable (hO : (O = 1/3 • (A + B + C)))
variable (h : (C - O = λ • (B - A) + μ • (C - A)))

theorem lambda_div_mu_eq_neg_half (hO : O = (1 / 3) • (A + B + C))
  (h : C - O = λ • (B - A) + μ • (C - A)) : 
  λ / μ = -1 / 2 :=
sorry

end lambda_div_mu_eq_neg_half_l610_610214


namespace intersection_eq_l610_610608

def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}

theorem intersection_eq : A ∩ B = {x | -1 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_eq_l610_610608


namespace calculate_expression_l610_610451

theorem calculate_expression : 287 * 287 + 269 * 269 - (2 * 287 * 269) = 324 :=
by
  sorry

end calculate_expression_l610_610451


namespace max_three_digit_sum_l610_610669

theorem max_three_digit_sum : ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ (0 ≤ A ∧ A < 10) ∧ (0 ≤ B ∧ B < 10) ∧ (0 ≤ C ∧ C < 10) ∧ (111 * A + 10 * C + 2 * B = 976) := sorry

end max_three_digit_sum_l610_610669


namespace minimum_perimeter_of_8_sided_polygon_formed_by_zeros_l610_610324

noncomputable def Q (z : ℂ) : ℂ := z^8 + (6 * Real.sqrt 2 + 8) * z^4 - (6 * Real.sqrt 2 + 9)

theorem minimum_perimeter_of_8_sided_polygon_formed_by_zeros (P : List ℂ) (h₁: (∀ p, p ∈ P → Q p = 0)) (h₂: P.length = 8) : 
  minimum_perimeter P = 8 * Real.sqrt 2 := 
sorry

end minimum_perimeter_of_8_sided_polygon_formed_by_zeros_l610_610324


namespace choose_three_consecutive_circles_l610_610074

theorem choose_three_consecutive_circles (n : ℕ) (hn : n = 33) : 
  ∃ (ways : ℕ), ways = 57 :=
by
  sorry

end choose_three_consecutive_circles_l610_610074


namespace tennis_tournament_handshakes_l610_610542

theorem tennis_tournament_handshakes :
  let teams := 4
  let players_per_team := 2
  let total_players := teams * players_per_team
  let handshakes_per_player := total_players - players_per_team in
  (total_players * handshakes_per_player) / 2 = 24 :=
by
  let teams := 4
  let players_per_team := 2
  let total_players := teams * players_per_team
  let handshakes_per_player := total_players - players_per_team
  have h : (total_players * handshakes_per_player) / 2 = 24 := sorry
  exact h

end tennis_tournament_handshakes_l610_610542


namespace inversely_proportional_percentage_change_l610_610476

variable {x y k : ℝ}
variable (a b : ℝ)

/-- Given that x and y are positive numbers and inversely proportional,
if x increases by a% and y decreases by b%, then b = 100a / (100 + a) -/
theorem inversely_proportional_percentage_change
  (hx : 0 < x) (hy : 0 < y) (hinv : y = k / x)
  (ha : 0 < a) (hb : 0 < b)
  (hchange : ((1 + a / 100) * x) * ((1 - b / 100) * y) = k) :
  b = 100 * a / (100 + a) :=
sorry

end inversely_proportional_percentage_change_l610_610476


namespace complex_expr_1_complex_expr_2_l610_610554

/--
Prove the equality of the complex expression
(1)(1 - i)(-1/2 + (sqrt 3 / 2)*i)(1 + i) = -1 + sqrt 3 * i.
-/
theorem complex_expr_1 : 
    (1 : ℂ) * (1 - complex.i) * (-1/2 + (real.sqrt 3 / 2) * complex.i) * (1 + complex.i) 
    = -1 + (real.sqrt 3) * complex.i := 
by sorry

/--
Prove the equality of the complex expression
(2 + 2*i) / (1 - i)^2 + (sqrt 2 / (1 + i))^2010 = -1.
-/
theorem complex_expr_2 : 
    ((2 : ℂ) + (2 * complex.i)) / (1 - complex.i)^2 + (real.sqrt 2 / (1 + complex.i))^2010 
    = -1 := 
by sorry

end complex_expr_1_complex_expr_2_l610_610554


namespace equation_of_circle_length_of_AB_l610_610975

-- Define center and radius of the circle
def center : ℝ × ℝ := (1, -3)
def radius : ℝ := 2

-- Define the equation of the line
def line (x y : ℝ) : Prop := x - y = 2

-- Define the equation of the circle
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 3)^2 = 4

-- Prove the equation of the circle
theorem equation_of_circle :
  ∀ (x y : ℝ), circle x y ↔ (x - 1)^2 + (y + 3)^2 = 4 := 
by 
  intro x y
  apply iff.rfl

-- Prove the length of segment AB given the intersection points
theorem length_of_AB :
  ∀ (A B : ℝ × ℝ), A = (1,-1) → B = (-1,-3) → 
  (sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 2 * sqrt 2) :=
by
  intro A B hA hB
  cases A with Ax Ay
  cases B with Bx By
  simp only [sqrt_eq_iff_sq_eq, sub_eq_add_neg, add_sq, mul_eq_mul_right_iff, eq_self_iff_true, true_and, neg_sq]
  sorry


end equation_of_circle_length_of_AB_l610_610975


namespace solve_for_x_l610_610671

theorem solve_for_x (x : ℚ) (h : (1 / 3 - 1 / 4 = 4 / x)) : x = 48 := by
  sorry

end solve_for_x_l610_610671


namespace zach_weekly_allowance_l610_610858

theorem zach_weekly_allowance
  (bike_cost : ℕ)
  (saved_amount : ℕ)
  (needed_more : ℕ)
  (allowance : ℕ)
  (mowing_income : ℕ)
  (babysit_income_per_hour : ℕ)
  (babysit_hours : ℕ)
  (total_bike_fund : ℕ)
  (needed_additional : ℕ)
  (total_income_from_mowing_babysitting : ℕ)
  (remaining_amount : ℕ)
  (weekly_allowance : ℕ) :
  bike_cost = 100 →
  saved_amount = 65 →
  needed_more = 6 →
  mowing_income = 10 →
  babysit_income_per_hour = 7 →
  babysit_hours = 2 →
  total_bike_fund = (bike_cost - needed_more) →
  needed_additional = (total_bike_fund - saved_amount) →
  total_income_from_mowing_babysitting = (mowing_income + babysit_income_per_hour * babysit_hours) →
  remaining_amount = (needed_additional - total_income_from_mowing_babysitting) →
  weekly_allowance = 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  rw [h1, h2, h3, h4, h5, h6] at h7 h8 h9 h10
  sorry

end zach_weekly_allowance_l610_610858


namespace no_feasible_distribution_l610_610276

-- Define the initial conditions
def initial_runs_player_A : ℕ := 320
def initial_runs_player_B : ℕ := 450
def initial_runs_player_C : ℕ := 550

def initial_innings : ℕ := 10

def required_increase_A : ℕ := 4
def required_increase_B : ℕ := 5
def required_increase_C : ℕ := 6

def total_run_limit : ℕ := 250

-- Define the total runs required after 11 innings
def total_required_runs_after_11_innings (initial_runs avg_increase : ℕ) : ℕ :=
  (initial_runs / initial_innings + avg_increase) * 11

-- Calculate the additional runs needed in the next innings
def additional_runs_needed (initial_runs avg_increase : ℕ) : ℕ :=
  total_required_runs_after_11_innings initial_runs avg_increase - initial_runs

-- Calculate the total additional runs needed for all players
def total_additional_runs_needed : ℕ :=
  additional_runs_needed initial_runs_player_A required_increase_A +
  additional_runs_needed initial_runs_player_B required_increase_B +
  additional_runs_needed initial_runs_player_C required_increase_C

-- The statement to verify if the total additional required runs exceed the limit
theorem no_feasible_distribution :
  total_additional_runs_needed > total_run_limit :=
by 
  -- Skipping proofs and just stating the condition is what we aim to show.
  sorry

end no_feasible_distribution_l610_610276


namespace not_possible_20_odd_rows_15_odd_columns_possible_126_crosses_in_16x16_l610_610500

-- Define the general properties and initial conditions for the problems.
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Question a: Can there be exactly 20 odd rows and 15 odd columns in the table?
theorem not_possible_20_odd_rows_15_odd_columns (n : ℕ) (n_odd_rows : ℕ) (n_odd_columns : ℕ) (crosses : ℕ) (h_crosses_odd : is_odd crosses) 
  (h_odd_rows : n_odd_rows = 20) (h_odd_columns : n_odd_columns = 15) : 
  false := 
sorry

-- Question b: Can 126 crosses be arranged in a \(16 \times 16\) table so that all rows and columns are odd?
theorem possible_126_crosses_in_16x16 (crosses : ℕ) (n : ℕ) (h_crosses : crosses = 126) (h_table_size : n = 16) : 
  ∃ (table : matrix ℕ ℕ bool), 
    (∀ i : fin n, is_odd (count (λ j, table i j) (list.range n))) ∧
    (∀ j : fin n, is_odd (count (λ i, table i j) (list.range n))) :=
sorry

end not_possible_20_odd_rows_15_odd_columns_possible_126_crosses_in_16x16_l610_610500


namespace greatest_distance_between_vertices_l610_610097

theorem greatest_distance_between_vertices 
    (inner_perimeter outer_perimeter : ℝ) 
    (inner_square_perimeter_eq : inner_perimeter = 16)
    (outer_square_perimeter_eq : outer_perimeter = 40)
    : ∃ max_distance, max_distance = 2 * Real.sqrt 34 :=
by
  sorry

end greatest_distance_between_vertices_l610_610097


namespace collinear_points_coplanar_points_l610_610303

-- Part (1): Proof of collinearity
theorem collinear_points (a b : ℝ) 
  (A : EuclideanSpace ℝ (fin 3)) (B : EuclideanSpace ℝ (fin 3)) (C : EuclideanSpace ℝ (fin 3))
  (hA : A = ![2, a, -1]) (hB : B = ![-2, 3, b]) (hC : C = ![1, 2, -2])
  (h_collinear : ∃ (λ : ℝ), A - C = λ • (B - C)):
  a = 5/3 ∧ b = -5 :=
sorry

-- Part (2): Proof of coplanarity
theorem coplanar_points (a : ℝ) 
  (A : EuclideanSpace ℝ (fin 3)) (B : EuclideanSpace ℝ (fin 3)) (C : EuclideanSpace ℝ (fin 3)) (D : EuclideanSpace ℝ (fin 3))
  (hA : A = ![2, a, -1]) (hB : B = ![-2, 3, -3]) (hC : C = ![1, 2, -2]) (hD : D = ![-1, 3, -3])
  (h_b_value : -3 = -3)
  (h_coplanar : ∃ (x y : ℝ), D - C = x • (A - C) + y • (B - C)):
  a = 1 :=
sorry

end collinear_points_coplanar_points_l610_610303


namespace differential_savings_l610_610269

theorem differential_savings (income : ℕ) (tax_rate_before : ℝ) (tax_rate_after : ℝ) : 
  income = 36000 → tax_rate_before = 0.46 → tax_rate_after = 0.32 →
  ((income * tax_rate_before) - (income * tax_rate_after)) = 5040 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end differential_savings_l610_610269


namespace vector_parallel_eq_l610_610339

theorem vector_parallel_eq (k : ℝ) (a b : ℝ × ℝ) 
  (h_a : a = (k, 2)) (h_b : b = (1, 1)) (h_parallel : (∃ c : ℝ, a = (c * 1, c * 1))) : k = 2 := by
  sorry

end vector_parallel_eq_l610_610339


namespace proveRandomEvent_l610_610854

def eventA (a : ℝ) (h : ¬ rational a) := sqrt a > 0
def eventB := ∀ (T : Triangle), existsCircumcircle T
def eventC := randomEvent (chooseTVChannel, findMoviePlaying)
def eventD := ¬ possibleToRunAtSpeed 50

-- Prove that eventC is a random event
theorem proveRandomEvent : randomEvent (chooseTVChannel, findMoviePlaying) :=
sorry

end proveRandomEvent_l610_610854


namespace max_island_visits_l610_610999
noncomputable def gamma (n : ℕ) : ℕ :=
  let m := Nat.primeFactors n;
  let i := m.count (λ x => ¬(Nat.powOfPrime n x > 1));
  let α := if h : 2 ∣ n then (multiplicity 2 n).get (Nat.multiplicity_finite_of_gt_one (by norm_num)) else 0;
  if α = 0 then 3 * m.length - i
  else if α = 1 then 3 * m.length - i + 1
  else 3 * m.length - i + 3

theorem max_island_visits (n a : ℕ) (h1 : 2 ≤ n) (h2 : Nat.coprime a n) :
  ∃ max_islands, max_islands = 1 + gamma n :=
sorry

end max_island_visits_l610_610999


namespace money_made_arkansas_game_is_8722_l610_610802

def price_per_tshirt : ℕ := 98
def tshirts_sold_arkansas_game : ℕ := 89
def total_money_made_arkansas_game (price_per_tshirt tshirts_sold_arkansas_game : ℕ) : ℕ :=
  price_per_tshirt * tshirts_sold_arkansas_game

theorem money_made_arkansas_game_is_8722 :
  total_money_made_arkansas_game price_per_tshirt tshirts_sold_arkansas_game = 8722 :=
by
  sorry

end money_made_arkansas_game_is_8722_l610_610802


namespace max_value_f_max_value_b_ac_l610_610235

noncomputable def f (x : Real) : Real := |x - 1| - 2 * |x + 1|

theorem max_value_f : ∃ (k : Real), ∀ (x : Real), f(x) ≤ k ∧ f(−1) = 2 := sorry

variables (a b c : Real)
theorem max_value_b_ac : (a^2 + c^2) / 2 + b^2 = 2 → b*(a + c) ≤ 2 := sorry

end max_value_f_max_value_b_ac_l610_610235


namespace max_value_of_sqrt_expression_l610_610320

noncomputable def max_sqrt_expr (x y z : ℝ) : ℝ :=
  sqrt (2 * x + 3) + sqrt (2 * y + 3) + sqrt (2 * z + 3)

theorem max_value_of_sqrt_expression (x y z : ℝ)
  (hxyz_sum : x + y + z = 7)
  (hx : x ≥ 2) (hy : y ≥ 2) (hz : z ≥ 2) :
  max_sqrt_expr x y z ≤ sqrt 69 :=
sorry

end max_value_of_sqrt_expression_l610_610320


namespace greatest_four_digit_multiple_of_17_l610_610007

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_multiple_of (n d : ℕ) : Prop :=
  ∃ k : ℕ, n = k * d

theorem greatest_four_digit_multiple_of_17 : ∃ n, is_four_digit n ∧ is_multiple_of n 17 ∧
  ∀ m, is_four_digit m → is_multiple_of m 17 → m ≤ n :=
  by
  existsi 9996
  sorry

end greatest_four_digit_multiple_of_17_l610_610007


namespace find_constants_C_and_A_l610_610159

theorem find_constants_C_and_A :
  ∃ (C A : ℚ), (C * x + 7 - 17)/(x^2 - 9 * x + 20) = A / (x - 4) + 2 / (x - 5) ∧ B = 7 ∧ C = 12/5 ∧ A = 2/5 := sorry

end find_constants_C_and_A_l610_610159


namespace sum_of_reciprocals_at_least_eight_terms_l610_610366

open Nat

def is_arithmetic_progression (n : ℕ) : Prop := ∃ k : ℕ, n = 2 + 3 * k

theorem sum_of_reciprocals_at_least_eight_terms :
  (∃ s : Finset ℕ, (∀ a ∈ s, is_arithmetic_progression a) ∧ (∑ a in s, (a:Real)⁻¹ = 1) ∧ s.card < 8) → False := by
  sorry

end sum_of_reciprocals_at_least_eight_terms_l610_610366


namespace double_summation_value_l610_610547

theorem double_summation_value :
  (∑ i in Finset.range 50, ∑ j in Finset.range 50, 2 * (i + j + 2)) = 255000 := 
sorry

end double_summation_value_l610_610547


namespace martha_cards_gave_3_l610_610764

theorem martha_cards_gave_3 (start_cards end_cards gave_cards : ℝ) (h1 : start_cards = 76.0) (h2 : end_cards = 73) : gave_cards = 3 := 
by 
  have h3 : gave_cards = start_cards - end_cards := sorry
  simp at *
  exact h3.symm
  sorry

end martha_cards_gave_3_l610_610764


namespace number_of_zeros_of_g_is_zero_l610_610144

noncomputable def f : ℝ → ℝ := sorry
def g (x : ℝ) : ℝ := f x + x⁻¹

theorem number_of_zeros_of_g_is_zero
  (hf_diff : ∀ x : ℝ, differentiable_at ℝ f x)
  (hf_cont : continuous f)
  (h_condition : ∀ x : ℝ, x ≠ 0 → deriv f x + x⁻¹ * f x > 0) :
  ∀ x : ℝ, g x ≠ 0 := by
  sorry

end number_of_zeros_of_g_is_zero_l610_610144


namespace vertical_asymptote_at_x_4_l610_610262

def P (x : ℝ) : ℝ := x^2 + 2 * x + 8
def Q (x : ℝ) : ℝ := x^2 - 8 * x + 16

theorem vertical_asymptote_at_x_4 : ∃ x : ℝ, Q x = 0 ∧ P x ≠ 0 ∧ x = 4 :=
by
  use 4
  -- Proof skipped
  sorry

end vertical_asymptote_at_x_4_l610_610262


namespace alternating_series_divisibility_l610_610644

noncomputable def alternating_series_sum : ℚ :=
  (Finset.range 1320).sum (λ k, if even k then -((k + 1)⁻¹ : ℚ) else ((k + 1)⁻¹ : ℚ))

theorem alternating_series_divisibility :
  let S := alternating_series_sum in
  ∃ p q : ℕ, S = p / q ∧ Nat.coprime p q ∧ 1979 ∣ p := 
by 
  sorry 

end alternating_series_divisibility_l610_610644


namespace simplify_exponential_expression_l610_610139

theorem simplify_exponential_expression : 
  (1 / 2) ^ (-1 : ℤ) + (1 / 4) ^ (0 : ℤ) - (9 : ℝ) ^ (1 / 2 : ℝ) = 0 :=
begin
  sorry
end

end simplify_exponential_expression_l610_610139


namespace parametric_curve_length_l610_610164

theorem parametric_curve_length :
  (∫ t in 0..(2 * π), sqrt ((deriv (λ t, 3 * sin t)) ^ 2 + (deriv (λ t, 3 * cos t)) ^ 2)) = 6 * π :=
by
  sorry

end parametric_curve_length_l610_610164


namespace max_deflection_angle_l610_610495

variable (M m : ℝ)
variable (h : M > m)

theorem max_deflection_angle :
  ∃ α : ℝ, α = Real.arcsin (m / M) := by
  sorry

end max_deflection_angle_l610_610495


namespace mutually_exclusive_events_l610_610696

def box : set (fin 12) := {i | i < 12}
def genuine : set (fin 12) := {i | 0 ≤ i ∧ i < 10}
def defective : set (fin 12) := {i | 10 ≤ i ∧ i < 12}

-- Event A: Exactly 1 defective product
def exactly_one_defective (s : set (fin 12)) : Prop :=
  ∃ d ∈ defective, ∃ g ∈ genuine, s = {d, g}

-- Event A: Exactly 2 defective products
def exactly_two_defective (s : set (fin 12)) : Prop :=
  s ⊆ defective ∧ s.card = 2

-- Event B: At least 1 defective product
def at_least_one_defective (s : set (fin 12)) : Prop :=
  ∃ x ∈ defective, x ∈ s

-- Event B: Both are defective products (same as exactly_two_defective)
-- Event C: At least 1 genuine product
def at_least_one_genuine (s : set (fin 12)) : Prop :=
  ∃ x ∈ genuine, x ∈ s

-- Event D: Both are genuine products
def both_genuine (s : set (fin 12)) : Prop :=
  s ⊆ genuine ∧ s.card = 2

theorem mutually_exclusive_events :
  ∀ s : set (fin 12), 
  (exactly_one_defective s → exactly_two_defective s → false) ∧
  (at_least_one_defective s → both_genuine s → false) :=
by {
  intros s,
  split;
  intros H1 H2;
  cases H1; sorry
}

end mutually_exclusive_events_l610_610696


namespace angle_between_axes_of_symmetry_l610_610881

theorem angle_between_axes_of_symmetry :
  ∀ (φ : ℝ), (∃ (f : ℝ → ℝ), ∀ x, f (f x) = x ∧ (f ∘ f) x = x ∧ f has exactly 2 axes of symmetry) →
  φ = 90 :=
by
  sorry

end angle_between_axes_of_symmetry_l610_610881


namespace data_set_average_l610_610095

theorem data_set_average (a : ℝ) (h : (2 + 3 + 3 + 4 + a) / 5 = 3) : a = 3 := 
sorry

end data_set_average_l610_610095


namespace square_tiles_count_l610_610698

theorem square_tiles_count 
  (h s : ℕ)
  (total_tiles : h + s = 30)
  (total_edges : 6 * h + 4 * s = 128) : 
  s = 26 :=
by
  sorry

end square_tiles_count_l610_610698


namespace cone_base_circumference_l610_610094

theorem cone_base_circumference
  (V : ℝ)
  (r h : ℝ)
  (h_eq : h = 3 * r)
  (V_eq : V = 18 * Real.pi)
  (vol_eq : V = (1 / 3) * Real.pi * r^2 * h) :
  2 * Real.pi * Real.cbrt 18 = 2 * Real.pi * r :=
by
  sorry

end cone_base_circumference_l610_610094


namespace weaving_increase_is_sixteen_over_twentynine_l610_610801

-- Conditions for the problem as definitions
def first_day_weaving := 5
def total_days := 30
def total_weaving := 390

-- The arithmetic series sum formula for 30 days
def sum_arithmetic_series (a d : ℚ) (n : ℕ) := n * a + (n * (n-1) / 2) * d

-- The question is to prove the increase in chi per day is 16/29
theorem weaving_increase_is_sixteen_over_twentynine
  (d : ℚ)
  (h : sum_arithmetic_series first_day_weaving d total_days = total_weaving) :
  d = 16 / 29 :=
sorry

end weaving_increase_is_sixteen_over_twentynine_l610_610801


namespace trigonometric_identity_proof_l610_610611

theorem trigonometric_identity_proof (α : ℝ)
  (h : cos (π / 6 - α) = sqrt 3 / 3) :
  cos (5 * π / 6 + α) + sin (α - π / 6)^2 = (2 - sqrt 3) / 3 := 
by sorry

end trigonometric_identity_proof_l610_610611


namespace compare_abc_l610_610317

noncomputable section
open Real

def a : ℝ := log (1/2) 5 
def b : ℝ := (1/3) ^ 0.2
def c : ℝ := 2 ^ (1/3)

theorem compare_abc : a < b ∧ b < c :=
by
  have ha : a = log (1/2) 5 := rfl
  have hb : b = (1/3) ^ 0.2 := rfl
  have hc : c = 2 ^ (1/3) := rfl
  sorry

end compare_abc_l610_610317


namespace min_value_of_K_l610_610027

variable (a b : ℝ)  -- a and b are real numbers
variable (ha : a > 0) (hb : b > 0)  -- a and b are positive constants

theorem min_value_of_K (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x : ℝ, ∀ y : ℝ, 
    sqrt ((y - a) ^ 2 + b ^ 2) + sqrt ((y - b) ^ 2 + a ^ 2) ≥
    sqrt (2 * (a ^ 2 + b ^ 2))) :=
sorry

end min_value_of_K_l610_610027


namespace sub_of_neg_l610_610130

theorem sub_of_neg : -3 - 2 = -5 :=
by 
  sorry

end sub_of_neg_l610_610130


namespace compute_a3_binv2_l610_610747

-- Define variables and their values
def a : ℚ := 4 / 7
def b : ℚ := 5 / 6

-- State the main theorem that directly translates the problem to Lean
theorem compute_a3_binv2 : (a^3 * b^(-2)) = (2304 / 8575) :=
by
  -- proof left as an exercise for the user
  sorry

end compute_a3_binv2_l610_610747


namespace intersection_A_B_is_1_and_2_l610_610641

-- Define sets A and B as per the given conditions
def setA : Set ℤ := {x : ℤ | -1 ≤ x ∧ x ≤ 3}
def setB : Set ℚ := {x : ℚ | 0 < x ∧ x < 3}

-- Assert the intersection of A and B is {1,2}
theorem intersection_A_B_is_1_and_2 : 
  (setA.inter {x : ℤ | ∃ (q : ℚ), x = q ∧ 0 < q ∧ q < 3}) = {1, 2} :=
sorry

end intersection_A_B_is_1_and_2_l610_610641


namespace compute_a3_b_neg2_l610_610748

noncomputable def a : ℚ := 4 / 7
noncomputable def b : ℚ := 5 / 6

theorem compute_a3_b_neg2 :
  a^3 * b^(-2) = 2304 / 8575 := 
by
  sorry

end compute_a3_b_neg2_l610_610748


namespace average_distance_scientific_notation_l610_610376

theorem average_distance_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ a * 10 ^ n = 384000000 ∧ a = 3.84 ∧ n = 8 :=
sorry

end average_distance_scientific_notation_l610_610376


namespace parallel_line_exists_l610_610445

noncomputable def line_through_point_parallel_to_given_line
  (l : Set (ℝ × ℝ)) (hl : ∃ A B, A ≠ B ∧ {A, B} ⊆ l)
  (B : ℝ × ℝ) (hB : B ∉ l) : Set (ℝ × ℝ) :=
sorry

theorem parallel_line_exists 
  (l : Set (ℝ × ℝ)) (hl : ∃ A B, A ≠ B ∧ {A, B} ⊆ l) 
  (B : ℝ × ℝ) (hB : B ∉ l) : 
  ∃ (BC : Set (ℝ × ℝ)), is_line_through B BC ∧ is_parallel BC l :=
sorry

end parallel_line_exists_l610_610445


namespace solve_for_x_l610_610712

theorem solve_for_x (x : ℝ) (h : (1 / 4) + (5 / x) = (12 / x) + (1 / 15)) : x = 420 / 11 := 
by
  sorry

end solve_for_x_l610_610712


namespace f_sqrt_45_l610_610334

noncomputable def f : ℝ → ℝ :=
λ x, if (∃ n : ℤ, x = n) then 7 * x + 3 else ⌊x⌋ + 6

theorem f_sqrt_45 : f (Real.sqrt 45) = 12 := by
  sorry

end f_sqrt_45_l610_610334


namespace initial_distance_is_432_l610_610874

variables (S D : ℝ)

-- Given Conditions
def initial_distance (speed time: ℝ) : ℝ := speed * time
def new_time (initial_time : ℝ) : ℝ := initial_time * (3 / 2)
def distance_by_new_speed (new_speed new_time : ℝ) : ℝ := new_speed * new_time

theorem initial_distance_is_432 :
  initial_distance S 6 = 432 :=
begin
  -- Definitions
  have def_initial_distance := initial_distance S 6,
  have def_new_time := new_time 6,
  have def_new_speed_distance := distance_by_new_speed 48 (new_time 6),

  -- Given Conditions
  have h1:  def_new_time = 9, from rfl,
  have h2:  def_new_speed_distance = 48 * def_new_time, from rfl,

  -- Proof
  calc
    def_initial_distance
        = S * 6 : rfl
    ... = 48 * 9 : by rw [h1, def_new_speed_distance, h2]
    ... = 432 : by norm_num
end

end initial_distance_is_432_l610_610874


namespace trapezoid_area_l610_610294

theorem trapezoid_area (outer_triangle_area inner_triangle_area : ℝ) (congruent_trapezoids : ℕ) 
  (h1 : outer_triangle_area = 36) (h2 : inner_triangle_area = 4) (h3 : congruent_trapezoids = 3) :
  (outer_triangle_area - inner_triangle_area) / congruent_trapezoids = 32 / 3 :=
by sorry

end trapezoid_area_l610_610294


namespace cube_edge_length_surface_area_equals_volume_l610_610930

theorem cube_edge_length_surface_area_equals_volume (a : ℝ) (h : 6 * a ^ 2 = a ^ 3) : a = 6 := 
by {
  sorry
}

end cube_edge_length_surface_area_equals_volume_l610_610930


namespace max_value_of_f_l610_610200

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt(2 * x ^ 3 + 7 * x ^ 2 + 6 * x)) / (x ^ 2 + 4 * x + 3)

theorem max_value_of_f : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → f x ≤ 1 / 2 :=
begin
  intros x hx,
  sorry
end

end max_value_of_f_l610_610200


namespace product_of_AB_AC_l610_610281

theorem product_of_AB_AC 
  (A B C P Q X Y : ℝ)
  (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (hP_perp : P = C ∧ P < A ∧ P > B) -- P is the foot of the perpendicular from C to AB
  (hQ_perp : Q = B ∧ Q < A ∧ Q > C) -- Q is the foot of the perpendicular from B to AC
  (hPQ_inter : ∃ t u : ℝ, PQ = t * (t * X + u * Y) ∧ t ≠ 0 ∧ u ≠ 0) -- PQ intersects the extended line of the circumcircle of ABC at X and Y
  (hXP : XP = 12)
  (hPQ : PQ = 30)
  (hQY : QY = 18)
  : (AB * AC = 1152 * √2) :=
sorry

end product_of_AB_AC_l610_610281


namespace operation_doubling_l610_610590

def operation_on_percent (x : ℝ) : ℝ := x / 4

theorem operation_doubling (y : ℝ) (h : y = 0.04) : 2 * operation_on_percent y = 0.02 :=
by
  rw [h, operation_on_percent]
  norm_num

#check operation_doubling

end operation_doubling_l610_610590


namespace school_dance_boys_count_l610_610422

theorem school_dance_boys_count :
  let total_attendees := 100
  let faculty_and_staff := total_attendees * 10 / 100
  let students := total_attendees - faculty_and_staff
  let girls := 2 * students / 3
  let boys := students - girls
  boys = 30 := by
  sorry

end school_dance_boys_count_l610_610422


namespace surface_area_of_segmented_part_l610_610557

theorem surface_area_of_segmented_part (h_prism : ∀ (base_height prism_height : ℝ), base_height = 9 ∧ prism_height = 20)
  (isosceles_triangle : ∀ (a b c : ℝ), a = 18 ∧ b = 15 ∧ c = 15 ∧ b = c)
  (midpoints : ∀ (X Y Z : ℝ), X = 9 ∧ Y = 10 ∧ Z = 9) 
  : let triangle_CZX_area := 45
    let triangle_CZY_area := 45
    let triangle_CXY_area := 9
    let triangle_XYZ_area := 9
    (triangle_CZX_area + triangle_CZY_area + triangle_CXY_area + triangle_XYZ_area = 108) :=
sorry

end surface_area_of_segmented_part_l610_610557


namespace range_of_m_l610_610848

theorem range_of_m {m : ℝ} : (-1 ≤ 1 - m ∧ 1 - m ≤ 1) ↔ (0 ≤ m ∧ m ≤ 2) := 
sorry

end range_of_m_l610_610848


namespace range_for_a_l610_610433

noncomputable def line_not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (3 * a - 1) * x + (2 - a) * y - 1 = 0 → (x ≥ 0 ∨ y ≥ 0)

theorem range_for_a (a : ℝ) :
  (line_not_in_second_quadrant a) ↔ a ≥ 2 := by
  sorry

end range_for_a_l610_610433


namespace Callum_seq_final_value_l610_610132

theorem Callum_seq_final_value :
  let initial_value := (9 : ℤ)^6
  let seq_15_steps := List.foldl (λ acc step, 
                                   if step % 2 == 0 then acc * 3 else acc / 2)
                                 initial_value (List.range 15)
  ∃ a b : ℤ, a = 3 ∧ b = 13 ∧ seq_15_steps = (a ^ b / 2^7) :=
by
  sorry -- proof omitted

end Callum_seq_final_value_l610_610132


namespace quadratic_polynomial_value_l610_610635

noncomputable def f (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

theorem quadratic_polynomial_value (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (h1 : f 1 (-(a + b + c)) (a*b + b*c + a*c) a = b * c)
  (h2 : f 1 (-(a + b + c)) (a*b + b*c + a*c) b = c * a)
  (h3 : f 1 (-(a + b + c)) (a*b + b*c + a*c) c = a * b) :
  f 1 (-(a + b + c)) (a*b + b*c + a*c) (a + b + c) = a * b + b * c + a * c := sorry

end quadratic_polynomial_value_l610_610635


namespace M_lt_N_l610_610409

variables (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

def N : ℝ := |a + b + c| + |2 * a - b|
def M : ℝ := |a - b + c| + |2 * a + b|

axiom h1 : f 1 < 0  -- a + b + c < 0
axiom h2 : f (-1) > 0  -- a - b + c > 0
axiom h3 : a > 0
axiom h4 : -b / (2 * a) > 1

theorem M_lt_N : M a b c < N a b c :=
by
  sorry

end M_lt_N_l610_610409


namespace linear_regression_equation_l610_610813

def x₁ := 3
def y₁ := 10
def x₂ := 7
def y₂ := 20
def x₃ := 11
def y₃ := 24

def x̄ := (x₁ + x₂ + x₃) / 3 
def ȳ := (y₁ + y₂ + y₃) / 3 

def b := ((x₁ - x̄) * (y₁ - ȳ) + (x₂ - x̄) * (y₂ - ȳ) + (x₃ - x̄) * (y₃ - ȳ)) / ((x₁ - x̄)^2 + (x₂ - x̄)^2 + (x₃ - x̄)^2) 
def a := ȳ - b * x̄ 

theorem linear_regression_equation : a = 5.75 ∧ b = 1.75 := by
  sorry

end linear_regression_equation_l610_610813


namespace total_area_of_rectangles_l610_610122

theorem total_area_of_rectangles (side_length : ℕ) (sum_perimeters : ℕ)
  (h1 : side_length = 6) (h2 : sum_perimeters = 92) : 
  let l1 := (11 - side_length) in
  let l2 := (11 - side_length) in
  2 * side_length * (l1 + l2) = 132 := by
  sorry

end total_area_of_rectangles_l610_610122


namespace next_hexagon_dots_l610_610187

theorem next_hexagon_dots (base_dots : ℕ) (increment : ℕ) : base_dots = 2 → increment = 2 → 
  (2 + 6*2) + 6*(2*2) + 6*(3*2) + 6*(4*2) = 122 := 
by
  intros hbd hi
  sorry

end next_hexagon_dots_l610_610187


namespace calories_burned_each_player_l610_610387

theorem calories_burned_each_player :
  ∀ (num_round_trips stairs_per_trip calories_per_stair : ℕ),
  num_round_trips = 40 →
  stairs_per_trip = 32 →
  calories_per_stair = 2 →
  (num_round_trips * (2 * stairs_per_trip) * calories_per_stair) = 5120 :=
by
  intros num_round_trips stairs_per_trip calories_per_stair h_num_round_trips h_stairs_per_trip h_calories_per_stair
  rw [h_num_round_trips, h_stairs_per_trip, h_calories_per_stair]
  simp
  rfl

#eval calories_burned_each_player 40 32 2 rfl rfl rfl

end calories_burned_each_player_l610_610387


namespace train_average_speed_with_stoppages_l610_610043

theorem train_average_speed_with_stoppages :
  (∀ d t_without_stops t_with_stops : ℝ, t_without_stops = d / 400 → 
  t_with_stops = d / (t_without_stops * (10/9)) → 
  t_with_stops = d / 360) :=
sorry

end train_average_speed_with_stoppages_l610_610043


namespace sum_of_squares_leq_equality_conditions_l610_610795

namespace AirlineProblem

variables {n m : ℕ}
variables {d : Fin n → ℕ}
variables (h1 : ∀ i : Fin n, 1 ≤ d i ∧ d i ≤ 2010)
variables (h2 : ∑ i, d i = 2 * m)

theorem sum_of_squares_leq (h1 : ∀ i : Fin n, 1 ≤ d i ∧ d i ≤ 2010) (h2 : ∑ i, d i = 2 * m) : 
  ∑ i, (d i) ^ 2 ≤ 4022 * m - 2010 * n := 
sorry

theorem equality_conditions (h1 : ∀ i : Fin n, 1 ≤ d i ∧ d i ≤ 2010) (h2 : ∑ i, d i = 2 * m) : 
  (∑ i, (d i) ^ 2 = 4022 * m - 2010 * n → 
  ((∀ i : Fin n, d i = 1 ∨ d i = 2010) ∧ (n % 2 = 0 ∨ (n % 2 = 1 ∧ n ≥ 2011)))) := 
sorry

end AirlineProblem

end sum_of_squares_leq_equality_conditions_l610_610795


namespace compute_large_expression_l610_610138

theorem compute_large_expression :
  ( (11^4 + 484) * (23^4 + 484) * (35^4 + 484) * (47^4 + 484) * (59^4 + 484) ) / 
  ( (5^4 + 484) * (17^4 + 484) * (29^4 + 484) * (41^4 + 484) * (53^4 + 484) ) = 552.42857 := 
sorry

end compute_large_expression_l610_610138


namespace compute_a3_b_neg2_l610_610751

noncomputable def a : ℚ := 4 / 7
noncomputable def b : ℚ := 5 / 6

theorem compute_a3_b_neg2 :
  a^3 * b^(-2) = 2304 / 8575 := 
by
  sorry

end compute_a3_b_neg2_l610_610751


namespace integer_condition_l610_610678

theorem integer_condition (p : ℕ) (h : p > 0) : 
  (∃ n : ℤ, (3 * (p: ℤ) + 25) = n * (2 * (p: ℤ) - 5)) ↔ (3 ≤ p ∧ p ≤ 35) :=
sorry

end integer_condition_l610_610678


namespace find_a9_l610_610970

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a9 (h_arith : is_arithmetic_sequence a)
  (h_a1 : a 1 = 3)
  (h_a4a6 : a 4 + a 6 = 8) : 
  a 9 = 5 :=
sorry

end find_a9_l610_610970


namespace range_of_c_l610_610687

noncomputable theory

open Real

theorem range_of_c (c : ℝ) (h : c ≤ -1/2) : 
  ¬ (∀ x : ℝ, x^2 + 4 * c * x + 1 > 0) :=
sorry

end range_of_c_l610_610687


namespace evaluate_expression_l610_610152

theorem evaluate_expression (b : ℕ) (hb : b = 2) : (b^3 * b^4) - b^2 = 124 :=
by
  -- leave the proof empty with a placeholder
  sorry

end evaluate_expression_l610_610152


namespace number_of_handshakes_l610_610541

-- Definitions based on the conditions:
def number_of_teams : ℕ := 4
def number_of_women_per_team : ℕ := 2
def total_women : ℕ := number_of_teams * number_of_women_per_team

-- Each woman shakes hands with all others except her partner
def handshakes_per_woman : ℕ := total_women - 1 - (number_of_women_per_team - 1)

-- Calculate total handshakes, considering each handshake is counted twice
def total_handshakes : ℕ := (total_women * handshakes_per_woman) / 2

-- Statement to prove
theorem number_of_handshakes :
  total_handshakes = 24 := 
sorry

end number_of_handshakes_l610_610541


namespace arrangement_count_l610_610176

theorem arrangement_count {α : Type*} [Fintype α] [DecidableEq α]
  (P Q R S T : α) :
  let L := [P, Q, R, S, T] in
  ∃ (n : ℕ), n = 36 ∧
  (∀ (L' : List α), L'.perm L → 
    ∀ i, (i < 4 → L'.nth i ≠ some P ∨ L'.nth (i+1) ≠ some Q) ∧ 
         (i < 4 → L'.nth i ≠ some P ∨ L'.nth (i+1) ≠ some R)) :=
by
  sorry

end arrangement_count_l610_610176


namespace c_n_zero_implies_n_zero_l610_610593

noncomputable def a_n := sorry -- Define a_n according to the problem
noncomputable def b_n := sorry -- Define b_n according to the problem
noncomputable def c_n := sorry -- Define c_n according to the problem

theorem c_n_zero_implies_n_zero (n : ℤ) (h : n ≥ 0) 
  (hn : (1 + 4 * (2 ^ (1/3)) - 4 * (4 ^ (1/3)))^n = a_n + b_n * (2 ^ (1/3)) + c_n * (4 ^ (1/3))) :
  c_n = 0 → n = 0 :=
sorry

end c_n_zero_implies_n_zero_l610_610593


namespace divide_shape_into_equal_parts_l610_610562

-- Definitions and conditions
structure Shape where
  has_vertical_symmetry : Bool
  -- Other properties of the shape can be added as necessary

def vertical_line_divides_equally (s : Shape) : Prop :=
  s.has_vertical_symmetry

-- Theorem statement
theorem divide_shape_into_equal_parts (s : Shape) (h : s.has_vertical_symmetry = true) :
  vertical_line_divides_equally s :=
by
  -- Begin proof
  sorry

end divide_shape_into_equal_parts_l610_610562


namespace odd_function_neg_expression_l610_610808

theorem odd_function_neg_expression (f : ℝ → ℝ) (h₀ : ∀ x > 0, f x = x^3 + x + 1)
    (h₁ : ∀ x, f (-x) = -f x) : ∀ x < 0, f x = x^3 + x - 1 :=
by
  sorry

end odd_function_neg_expression_l610_610808


namespace area_bounded_region_is_2_l610_610865

noncomputable def area_bounded_region : ℝ :=
  ∫ x in -Real.pi / 2 .. Real.pi / 2, 1 / (1 + Real.cos x)

theorem area_bounded_region_is_2 : area_bounded_region = 2 :=
by
  sorry

end area_bounded_region_is_2_l610_610865


namespace range_of_a_l610_610663

def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }
def B (a : ℝ) : Set ℝ := { x | x ≥ a }

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a ≤ -2 :=
by
  sorry

end range_of_a_l610_610663


namespace unique_fixed_point_l610_610963

structure Point :=
(x : ℝ) (y : ℝ)

structure Triangle :=
(A B C : Point)

def midpoint (P₁ P₂ : Point) : Point :=
{ x := (P₁.x + P₂.x) / 2,
  y := (P₁.y + P₂.y) / 2 }

def f (T : Triangle) (P : Point) : Point :=
let Q := midpoint T.A P in
let R := midpoint T.B Q in
midpoint T.C R

theorem unique_fixed_point (T : Triangle) : 
  ∃! P : Point, f T P = P :=
sorry

end unique_fixed_point_l610_610963


namespace flag_distance_false_l610_610592

theorem flag_distance_false (track_length : ℕ) (num_flags : ℕ) (flag1_flagN : 2 ≤ num_flags)
  (h1 : track_length = 90) (h2 : num_flags = 10) :
  ¬ (track_length / (num_flags - 1) = 9) :=
by
  sorry

end flag_distance_false_l610_610592


namespace midpoint_AC_on_DE_l610_610446

open EuclideanGeometry

-- Definitions for the conditions
variables (A B C E D M : Point)
variables (h_triangle : Triangle A B C)
variables (h_eq_angles_1 : ∠BAC = ∠BCA)
variables (h_E_on_bisector : E ∈ interior (bisector ∠ABC))
variables (h_eq_angles_2 : ∠EAB = ∠BCA)
variables (h_D_on_BC : D ∈ Line_Segment B C)
variables (h_BD_AB : Distance B D = Distance A B)
variables (h_M_midpoint_AC : IsMidpoint M A C)

-- Statement of the theorem
theorem midpoint_AC_on_DE :
  Collinear D E M :=
sorry

end midpoint_AC_on_DE_l610_610446


namespace symmetric_scanning_codes_count_l610_610893

theorem symmetric_scanning_codes_count :
  let num_symmetric_code := 2^5 - 2
  in num_symmetric_code = 30 := by
sorry

end symmetric_scanning_codes_count_l610_610893


namespace greatest_four_digit_multiple_of_17_l610_610015

theorem greatest_four_digit_multiple_of_17 : ∃ n, (1000 ≤ n) ∧ (n ≤ 9999) ∧ (17 ∣ n) ∧ ∀ m, (1000 ≤ m) ∧ (m ≤ 9999) ∧ (17 ∣ m) → m ≤ n :=
begin
  sorry
end

end greatest_four_digit_multiple_of_17_l610_610015


namespace sum_divisible_by_ten_l610_610604

    -- Given conditions
    def is_natural_number (n : ℕ) : Prop := true

    -- Sum S as defined in the conditions
    def S (n : ℕ) : ℕ := n ^ 2 + (n + 1) ^ 2 + (n + 2) ^ 2 + (n + 3) ^ 2

    -- The equivalent math proof problem in Lean 4 statement
    theorem sum_divisible_by_ten (n : ℕ) : S n % 10 = 0 ↔ n % 5 = 1 := by
      sorry
    
end sum_divisible_by_ten_l610_610604


namespace question1_question2_l610_610245

noncomputable def vector_m (x : ℝ) : Vector ℝ 2 :=
  vector.of_fn [sqrt 3 * sin (x / 4), 1]

noncomputable def vector_n (x : ℝ) : Vector ℝ 2 :=
  vector.of_fn [cos (x / 4), cos (x / 4) ^ 2]

def perpendicular (m n : Vector ℝ 2) : Prop :=
  (m.head * n.head + m.tail.head * n.tail.head) = 0

def cos_expr(x: ℝ) : ℝ :=
  cos ((2 * real.pi) / 3 - x)

def f (x : ℝ) : ℝ := 
  let m := vector_m x
  let n := vector_n x
  (m.head * n.head + m.tail.head * n.tail.head)

theorem question1 (x : ℝ) (h : perpendicular (vector_m x) (vector_n x)) : 
  cos_expr x = - (1 / 2) := sorry

theorem question2 (A a b c B C : ℝ) (h1 : (2 * a - c) * cos B = b * cos C) (h2 : B = real.pi / 3) : 1 < f A ∧ f A < (3 / 2) := sorry

end question1_question2_l610_610245


namespace intersection_A_B_is_1_and_2_l610_610640

-- Define sets A and B as per the given conditions
def setA : Set ℤ := {x : ℤ | -1 ≤ x ∧ x ≤ 3}
def setB : Set ℚ := {x : ℚ | 0 < x ∧ x < 3}

-- Assert the intersection of A and B is {1,2}
theorem intersection_A_B_is_1_and_2 : 
  (setA.inter {x : ℤ | ∃ (q : ℚ), x = q ∧ 0 < q ∧ q < 3}) = {1, 2} :=
sorry

end intersection_A_B_is_1_and_2_l610_610640


namespace geometric_sequence_common_ratio_l610_610690

theorem geometric_sequence_common_ratio :
  (∃ q : ℝ, 1 + q + q^2 = 13 ∧ (q = 3 ∨ q = -4)) :=
by
  sorry

end geometric_sequence_common_ratio_l610_610690


namespace max_cylinder_volume_l610_610545

/-- Given a rectangle with perimeter 18 cm, when rotating it around one side to form a cylinder, 
    the maximum volume of the cylinder and the corresponding side length of the rectangle. -/
theorem max_cylinder_volume (x y : ℝ) (h_perimeter : 2 * (x + y) = 18) (hx : x > 0) (hy : y > 0)
  (h_cylinder_volume : ∃ (V : ℝ), V = π * x * (y / 2)^2) :
  (x = 3 ∧ y = 6 ∧ ∀ V, V = 108 * π) := sorry

end max_cylinder_volume_l610_610545


namespace number_of_boys_l610_610428

-- Define the conditions
def total_attendees : Nat := 100
def faculty_percentage : Rat := 0.1
def faculty_count : Nat := total_attendees * faculty_percentage
def student_count : Nat := total_attendees - faculty_count
def girls_fraction : Rat := 2 / 3
def girls_count : Nat := student_count * girls_fraction

-- Define the question in terms of a Lean theorem
theorem number_of_boys :
  total_attendees = 100 →
  faculty_percentage = 0.1 →
  faculty_count = 10 →
  student_count = 90 →
  girls_fraction = 2 / 3 →
  girls_count = 60 →
  student_count - girls_count = 30 :=
by
  intros
  sorry -- Skip the proof

end number_of_boys_l610_610428


namespace f_even_f_period_l610_610198

def f (x: ℝ) : ℝ := sin (2 * x - π / 2)

theorem f_even (x : ℝ) : f (-x) = f x := sorry

theorem f_period (T : ℝ) : (∀ x, f (x + T) = f x) ↔ T = π := sorry

end f_even_f_period_l610_610198


namespace ratio_condition_l610_610775

theorem ratio_condition (ABC : Triangle)
  (P : Point)
  (h1 : P ∈ interior ABC)
  (h2 : angle B P C = 120)
  (h3 : dist A P * sqrt 2 = dist B P + dist C P) :
  let a := 1, b := 15, c := 5 in
  gcd a c = 1 ∧ 100 * a + 10 * b + c = 255 :=
by
  sorry

end ratio_condition_l610_610775


namespace number_of_flags_l610_610943

theorem number_of_flags (colors : Finset String)
  (stripes : Fin) : colors.card = 3 ∧ stripes = 3 → 
  ∃ n : ℕ, n = 27 :=
begin
  assume h,
  have color_number : colors.card = 3, from h.1,
  have stripe_number : stripes = 3, from h.2,
  use (colors.card ^ stripes),
  rw [color_number, stripe_number],
  norm_num,
end

end number_of_flags_l610_610943


namespace expected_value_of_sixs_on_two_eight_sided_dice_l610_610442

theorem expected_value_of_sixs_on_two_eight_sided_dice : 
  let p := (1 / 8) in
  (2 * p * (1 - p) + 2 * p^2) = (1 / 4) :=
by
  sorry

end expected_value_of_sixs_on_two_eight_sided_dice_l610_610442


namespace find_angle_degree_l610_610929

theorem find_angle_degree (x : ℝ) (h1 : 90 - x = (2 / 5) * (180 - x)) : x = 30 :=
sorry

end find_angle_degree_l610_610929


namespace min_x2_y2_z2_l610_610617

theorem min_x2_y2_z2 (x y z : ℝ) (h : 2 * x + 3 * y + 3 * z = 1) : 
  x^2 + y^2 + z^2 ≥ 1 / 22 :=
by
  sorry

end min_x2_y2_z2_l610_610617


namespace range_of_a_l610_610435

open Real

noncomputable def doesNotPassThroughSecondQuadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (3 * a - 1) * x + (2 - a) * y - 1 ≠ 0

theorem range_of_a : {a : ℝ | doesNotPassThroughSecondQuadrant a} = {a : ℝ | 2 ≤ a } :=
by
  ext
  sorry

end range_of_a_l610_610435


namespace locus_equation_polar_l610_610965

-- Definitions and conditions
def circle_polar (rho θ : ℝ) : Prop := rho = 2
def line_polar (rho θ : ℝ) : Prop := rho * (cos θ + sin θ) = 2
def locus_Q (rho θ : ℝ) : Prop := rho = 2 * (cos θ + sin θ)

-- Theorem statement
theorem locus_equation_polar (rho_1 rho_2 rho : ℝ) (θ : ℝ) :
  (circle_polar rho_2 θ) → (line_polar rho_1 θ) → (rho * rho_1 = rho_2^2) → locus_Q rho θ :=
by
  intros h_circle h_line h_cond
  sorry

end locus_equation_polar_l610_610965


namespace calculate_WZ_squared_l610_610704

variable (A B C D W Z : Point)

-- Define coordinates for the points based on the problem conditions
def D := (0, 0)
def C := (26, 0)
def A := (-13, 13 * Real.sqrt 3)
def B := (13, 15)

-- Define the midpoints W and Z
def W := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
def Z := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)

-- Define AB = BC = 15, CD = DA = 26, and ∠D = 60°
axiom AB_eq_BC_eq_15 : dist A B = 15 ∧ dist B C = 15
axiom CD_eq_DA_eq_26 : dist C D = 26 ∧ dist D A = 26
axiom angle_D_eq_60_deg : ∠ (C - D) (A - D) = π / 3 -- 60 degrees in radians

-- The theorem to prove WZ^2 = 352.5 - 97.5√3
theorem calculate_WZ_squared : dist_sq W Z = 352.5 - 97.5 * Real.sqrt 3 := by
  sorry

end calculate_WZ_squared_l610_610704


namespace polygon_sides_eq_nine_l610_610823

theorem polygon_sides_eq_nine (n : ℕ) 
  (interior_sum : ℕ := (n - 2) * 180)
  (exterior_sum : ℕ := 360)
  (condition : interior_sum = 4 * exterior_sum - 180) : 
  n = 9 :=
by {
  sorry
}

end polygon_sides_eq_nine_l610_610823


namespace compute_exp_l610_610755

theorem compute_exp (a b : ℚ) (ha : a = 4 / 7) (hb : b = 5 / 6) : a^3 * b^(-2) = 2304 / 8575 := by
  sorry

end compute_exp_l610_610755


namespace ratio_IK_KJ_l610_610803

-- Define the conditions of the problem
variables {A B C I J O_b O_c K : Point}
variables {ω_b ω_c : Circle}

-- Introduce the geometry problem and the conditions
def geometry_problem (triangle_ABC : Triangle A B C) (bisectors_I : IsBisector A B I ∧ IsBisector A C I)
  (external_bisectors_J : IsExternalBisector B J ∧ IsExternalBisector C J)
  (circle_ωb : CenteredAt ω_b O_b ∧ PassesThrough ω_b B ∧ TangentAt ω_b I C)
  (circle_ωc : CenteredAt ω_c O_c ∧ PassesThrough ω_c C ∧ TangentAt ω_c I B)
  (intersection_K : IntersectsAt (Line O_b O_c) (Line I J) K) : Prop :=
  Ratio I K K J = 1 / 3

-- Statement of the proof problem
theorem ratio_IK_KJ :
  ∀ (A B C I J O_b O_c K : Point) (ω_b ω_c : Circle),
    (triangle_ABC : Triangle A B C) →
    (bisectors_I : IsBisector A B I ∧ IsBisector A C I) →
    (external_bisectors_J : IsExternalBisector B J ∧ IsExternalBisector C J) →
    (circle_ωb : CenteredAt ω_b O_b ∧ PassesThrough ω_b B ∧ TangentAt ω_b I C) →
    (circle_ωc : CenteredAt ω_c O_c ∧ PassesThrough ω_c C ∧ TangentAt ω_c I B) →
    (intersection_K : IntersectsAt (Line O_b O_c) (Line I J) K) →
    Ratio I K K J = 1 / 3 :=
by
  intros
  sorry

end ratio_IK_KJ_l610_610803


namespace union_sets_l610_610969

-- Given sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5}

theorem union_sets : A ∪ B = {1, 2, 3, 4, 5} := by
  sorry

end union_sets_l610_610969


namespace smaller_angle_is_85_l610_610697

-- Conditions
def isParallelogram (α β : ℝ) : Prop :=
  α + β = 180

def angleExceedsBy10 (α β : ℝ) : Prop :=
  β = α + 10

-- Proof Problem
theorem smaller_angle_is_85 (α β : ℝ)
  (h1 : isParallelogram α β)
  (h2 : angleExceedsBy10 α β) :
  α = 85 :=
by
  sorry

end smaller_angle_is_85_l610_610697


namespace hyperbola_real_axis_length_l610_610812

theorem hyperbola_real_axis_length :
  ∀ x y : ℝ, (x^2 / 3) - (y^2 / 6) = 1 → real_axis_length = 2 * sqrt 3 := 
by
  sorry

end hyperbola_real_axis_length_l610_610812


namespace greatest_four_digit_multiple_of_17_l610_610021

theorem greatest_four_digit_multiple_of_17 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n ∧ n = 9996 :=
by
  use 9996
  split
  { exact by norm_num }
  split
  { exact by norm_num [lt.sub (ofNat 10000) (ofNat 4)] }
  split
  { exact by norm_num }
  { exact by norm_num [dvd_of_mod_eq_zero (by norm_num : 10000 % 17 = 4)] }

end greatest_four_digit_multiple_of_17_l610_610021


namespace conference_center_distance_l610_610363

variables (d t: ℝ)

theorem conference_center_distance
  (h1: ∃ t: ℝ, d = 45 * (t + 1.5))
  (h2: ∃ t: ℝ, d - 45 = 55 * (t - 1.25)):
  d = 478.125 :=
by
  sorry

end conference_center_distance_l610_610363


namespace arithmetic_sequence_geometric_condition_l610_610140

theorem arithmetic_sequence_geometric_condition 
  (a : ℕ → ℝ) (d : ℝ) (h_nonzero : d ≠ 0) 
  (h_a3 : a 3 = 7)
  (h_geo_seq : (a 2 - 1)^2 = (a 1 - 1) * (a 4 - 1)) : 
  a 10 = 21 :=
sorry

end arithmetic_sequence_geometric_condition_l610_610140


namespace part_a_part_b_l610_610506

-- Definitions based on the problem conditions
def is_square_table (n : ℕ) (table : List (List Bool)) : Prop := 
  table.length = n ∧ ∀ row, row ∈ table → row.length = n

def is_odd_row (row : List Bool) : Prop := 
  row.count (λ x => x = true) % 2 = 1

def is_odd_column (n : ℕ) (table : List (List Bool)) (c : ℕ) : Prop :=
  n > 0 ∧ c < n ∧ (List.count (λ row => row.get! c = true) table) % 2 = 1

-- Part (a) statement: Prove it's impossible to have exactly 20 odd rows and 15 odd columns
theorem part_a (n : ℕ) (table : List (List Bool)) :
  n = 16 → is_square_table n table → 
  (List.count is_odd_row table) = 20 → 
  ((List.range n).count (is_odd_column n table)) = 15 → 
  False := 
sorry

-- Part (b) statement: Prove that it's possible to arrange 126 crosses in a 16x16 table with all odd rows and columns
theorem part_b (table : List (List Bool)) :
  is_square_table 16 table →
  ((table.map (λ row => row.count (λ x => x = true))).sum = 126) →
  (∀ row, row ∈ table → is_odd_row row) →
  (∀ c, c < 16 → is_odd_column 16 table c) →
  True :=
sorry

end part_a_part_b_l610_610506


namespace shells_found_l610_610343

/-- 
Lucy originally had 68 shells and now she has 89 shells. 
We want to find out how many more shells she found.
-/
theorem shells_found (original new found : ℕ) (h₁ : original = 68) (h₂ : new = 89) (h₃ : found = new - original) : found = 21 :=
by
  simp [h₁, h₂, h₃]
  sorry

end shells_found_l610_610343


namespace percentage_of_employees_at_picnic_l610_610049

variable (total_employees : ℝ) (men_percent women_percent men_picnic_percent women_picnic_percent : ℝ)
variable (men employees_at_picnic : ℝ)

def men_total : ℝ := total_employees * men_percent
def women_total : ℝ := total_employees * women_percent

def men_at_picnic : ℝ := men_total * men_picnic_percent
def women_at_picnic : ℝ := women_total * women_picnic_percent

theorem percentage_of_employees_at_picnic
  (h1 : men_percent = 0.45) 
  (h2 : women_percent = 0.55)
  (h3 : men_picnic_percent = 0.20)
  (h4 : women_picnic_percent = 0.40)
  (h5 : total_employees = 100) :
  ((men_at_picnic + women_at_picnic) / total_employees) * 100 = 31 := 
by 
  sorry

end percentage_of_employees_at_picnic_l610_610049


namespace kenneth_money_left_l610_610729

noncomputable def baguettes : ℝ := 2 * 2
noncomputable def water : ℝ := 2 * 1

noncomputable def chocolate_bars_cost_before_discount : ℝ := 2 * 1.5
noncomputable def chocolate_bars_cost_after_discount : ℝ := chocolate_bars_cost_before_discount * (1 - 0.20)
noncomputable def chocolate_bars_final_cost : ℝ := chocolate_bars_cost_after_discount * 1.08

noncomputable def milk_cost_after_discount : ℝ := 3.5 * (1 - 0.10)

noncomputable def chips_cost_before_tax : ℝ := 2.5 + (2.5 * 0.50)
noncomputable def chips_final_cost : ℝ := chips_cost_before_tax * 1.08

noncomputable def total_cost : ℝ :=
  baguettes + water + chocolate_bars_final_cost + milk_cost_after_discount + chips_final_cost

noncomputable def initial_amount : ℝ := 50
noncomputable def amount_left : ℝ := initial_amount - total_cost

theorem kenneth_money_left : amount_left = 50 - 15.792 := by
  sorry

end kenneth_money_left_l610_610729


namespace general_term_max_sum_value_l610_610228

-- Define the arithmetic sequence with given first term and common difference
def a_n (n : ℕ) : ℤ := 20 + (n - 1) * (-2)

-- Define the sum S_n of the first n terms of the sequence
def S_n (n : ℕ) : ℤ :=
  n * (20 + a_n n) / 2

-- Theorem to prove the general term of the arithmetic sequence
theorem general_term (n : ℕ) : a_n n = -2 * n + 22 :=
by
  sorry  -- Proof to be completed

-- Theorem to prove the maximum value of the sum of the first n terms
theorem max_sum_value : ∃ n, n ∈ {10, 11} ∧ S_n n = 110 :=
by
  sorry  -- Proof to be completed

end general_term_max_sum_value_l610_610228


namespace square_area_eq_36_l610_610895

theorem square_area_eq_36 (A_triangle : ℝ) (P_triangle : ℝ) 
  (h1 : A_triangle = 16 * Real.sqrt 3)
  (h2 : P_triangle = 3 * (Real.sqrt (16 * 4 * Real.sqrt 3)))
  (h3 : ∀ a, 4 * a = P_triangle) : 
  a^2 = 36 :=
by sorry

end square_area_eq_36_l610_610895


namespace max_ships_on_10x10_board_l610_610057

theorem max_ships_on_10x10_board : 
  ∀ (board : fin 10 × fin 10 → bool) (ship : fin 4 → fin 2) (no_overlap : (∀ i j, ship i.1 ≠ ship j.1 → ∀ x y, x ≠ y → board (x, i.2) = tt → board (y, j.2) ≠ tt)), 
  ∃ (max_ships : ℕ), max_ships = 24 :=
begin
  sorry,
end

end max_ships_on_10x10_board_l610_610057


namespace problem_1_1_problem_2_1_problem_2_2_1_problem_2_2_2_minimum_value_3_expr_problem_3_l610_610785

variables {x : ℝ}

-- Define point coordinates and their relationship
def distance (a b : ℝ) := abs (a - b)

-- Proof that distance between A at x and B at 2 equals |x-2|
theorem problem_1_1 (x : ℝ) : distance x 2 = abs (x - 2) :=
  sorry

-- Proof that minimum value of |x-2| + |x-4| is 2
theorem problem_2_1 : ∀ (x : ℝ), 2 ≤ x ∧ x ≤ 4 → abs (x - 2) + abs (x - 4) = 2 :=
  sorry

-- Evaluate expression for specific values
def value_3_expr := λ x, abs (x - 2) + abs (x - 4) + abs (x - 10)

theorem problem_2_2_1 : value_3_expr 3 = 9 :=
  sorry

theorem problem_2_2_2 : value_3_expr 4 = 8 :=
  sorry

-- Proof that |x-2| + |x-4| + |x-10| has minimum value 8 when x = 4
theorem minimum_value_3_expr : ∀ x : ℝ, x = 4 → value_3_expr x = 8 :=
  sorry

-- Proof that the optimal location for charging station P is parking lot B
theorem problem_3 (A B C D : ℝ) (hB_close_scenic : true) : true :=
  sorry

end problem_1_1_problem_2_1_problem_2_2_1_problem_2_2_2_minimum_value_3_expr_problem_3_l610_610785


namespace abs_diff_imaginary_l610_610985

theorem abs_diff_imaginary (a : ℝ) (h : a + (10 * complex.I) / (3 - complex.I) = 0 + (a - (1 - 3 * complex.I))) :
  complex.abs (complex.mk a 0 - complex.I * 2) = real.sqrt 5 :=
sorry

end abs_diff_imaginary_l610_610985


namespace hexagon_area_increase_l610_610028

theorem hexagon_area_increase (s : ℝ) : 
  let A := (3 * Real.sqrt 3 / 2) * s^2,
      s' := 1.25 * s,
      A' := (3 * Real.sqrt 3 / 2) * s'^2 in
  ((A' - A) / A) * 100 = 56.25 := 
by
  -- Proof goes here
  sorry

end hexagon_area_increase_l610_610028


namespace distance_to_focus_from_A_l610_610981

namespace parabola_problem

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := (1 / 4) * x^2

-- Define point A with an ordinate of 4
def point_A : (ℝ × ℝ) := let y := 4 in (2 * real.sqrt y, y) -- Note: A point (x, y) on the parabola should satisfy y = (1/4)x^2, therefore x = 2√y

-- Define a function to compute the distance between two points in ℝ²
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the focus of the parabola (for a standard parabola y = (1 / 4) x^2)
def focus : ℝ × ℝ := (0, -1)

-- Define the axis of the parabola
def axis : ℝ := -1

-- Compute the distance from point A to the axis (abs is used since y1 and axis could be negative)
def distance_to_axis (point : ℝ × ℝ) (axis : ℝ) : ℝ := abs (point.2 - axis)

-- The theorem that needs to be proved
theorem distance_to_focus_from_A :
  distance_to_axis point_A axis = 5 :=
sorry

end parabola_problem

end distance_to_focus_from_A_l610_610981


namespace find_f_a_plus_b_plus_c_l610_610634

open Polynomial

variables {a b c p q r : ℝ}
variables (f : ℝ → ℝ)

-- Polynomial conditions
def f_poly (x : ℝ) := p * x^2 + q * x + r

-- Given distinct real numbers a, b, c
variables (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
-- Given polynomial conditions
variables (h1 : f a = b * c)
variables (h2 : f b = c * a)
variables (h3 : f c = a * b)

-- The goal statement
theorem find_f_a_plus_b_plus_c :
  f (a + b + c) = a * b + b * c + c * a := sorry

end find_f_a_plus_b_plus_c_l610_610634


namespace range_of_a_l610_610686

theorem range_of_a (a : ℝ) :
  (¬ ∃ t : ℝ, t^2 - 2*t - a < 0) → (a ≤ -1) :=
by 
  sorry

end range_of_a_l610_610686


namespace product_11_29_product_leq_20_squared_product_leq_half_m_squared_l610_610768

-- Definition of natural numbers
variable (a b m : ℕ)

-- Statement 1: Prove that 11 × 29 = 20^2 - 9^2
theorem product_11_29 : 11 * 29 = 20^2 - 9^2 := sorry

-- Statement 2: Prove ∀ a, b ∈ ℕ, if a + b = 40, then ab ≤ 20^2.
theorem product_leq_20_squared (a b : ℕ) (h : a + b = 40) : a * b ≤ 20^2 := sorry

-- Statement 3: Prove ∀ a, b ∈ ℕ, if a + b = m, then ab ≤ (m/2)^2.
theorem product_leq_half_m_squared (a b : ℕ) (m : ℕ) (h : a + b = m) : a * b ≤ (m / 2)^2 := sorry

end product_11_29_product_leq_20_squared_product_leq_half_m_squared_l610_610768


namespace odd_rows_cols_impossible_arrange_crosses_16x16_l610_610512

-- Define the conditions for part (a)
def square (α : Type*) := α × α
def is_odd_row (table : square nat → bool) (n : nat) :=
  ∃ (i : fin n), ∑ j in finset.range n, table (i, j) = 1
def is_odd_col (table : square nat → bool) (n : nat) :=
  ∃ (j : fin n), ∑ i in finset.range n, table (i, j) = 1

-- Part (a) statement
theorem odd_rows_cols_impossible (table : square nat → bool) (n : nat) :
  n = 16 ∧ (∃ (r : ℕ), r = 20) ∧ (∃ (c : ℕ), c = 15) → ¬(is_odd_row table n ∧ is_odd_col table n) :=
begin 
  -- Sorry placeholder for the proof
  sorry,
end

-- Define the conditions for part (b)
def odd_placement_possible (table : square nat → bool) :=
  ∃ (n : nat), n = 16 ∧ (∑ i in finset.range 16, ∑ j in finset.range 16, table (i, j) = 126) ∧ 
  (∀ i, is_odd_row table 16) ∧ (∀ j, is_odd_col table 16)

-- Part (b) statement
theorem arrange_crosses_16x16 (table : square nat → bool) :
  odd_placement_possible table :=
begin 
  -- Sorry placeholder for the proof
  sorry,
end

end odd_rows_cols_impossible_arrange_crosses_16x16_l610_610512


namespace cylinder_volume_ratio_l610_610758

theorem cylinder_volume_ratio
  (S1 S2 : ℝ) (v1 v2 : ℝ)
  (lateral_area_equal : 2 * Real.pi * S1.sqrt = 2 * Real.pi * S2.sqrt)
  (base_area_ratio : S1 / S2 = 16 / 9) :
  v1 / v2 = 4 / 3 :=
by
  sorry

end cylinder_volume_ratio_l610_610758


namespace percentage_of_C_students_l610_610535

theorem percentage_of_C_students :
  let scores := [93, 71, 55, 98, 81, 89, 77, 72, 78, 62, 87, 80, 68, 82, 91, 67, 76, 84, 70, 95] in
  let grade_C := filter (λ x, 76 ≤ x ∧ x ≤ 85) scores in
  (grade_C.length / scores.length.to_float) * 100 = 35 :=
by
  sorry

end percentage_of_C_students_l610_610535


namespace min_distance_when_alpha_2pi_3_range_of_alpha_for_intersection_l610_610632

-- Parametric definitions for circle C and line L
def circle (θ : ℝ) : ℝ × ℝ := (1 + Real.cos θ, Real.sin θ)
def line (t α : ℝ) : ℝ × ℝ := (2 + t * Real.cos α, Real.sqrt 3 + t * Real.sin α)

-- Conditions
def circle_eqn (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def line_eqn (x y α : ℝ) : Prop := Real.sqrt 3 * x + y - 3 * Real.sqrt 3 = 0

-- Questions
theorem min_distance_when_alpha_2pi_3 :
  (∀ θ : ℝ, circle_eqn (circle θ).fst (circle θ).snd) →
  (∀ t : ℝ, line_eqn (line t (2 * Real.pi / 3)).fst (line t (2 * Real.pi / 3)).snd (2 * Real.pi / 3)) →
  (sqrt 3 - 1) := 
by
  sorry

theorem range_of_alpha_for_intersection :
  (∀ θ : ℝ, circle_eqn (circle θ).fst (circle θ).snd) →
  ∀ α : ℝ, ∃ t : ℝ, (circle_eqn (line t α).fst (line t α).snd) →
  (π / 6 ≤ α ∧ α < π / 2) := 
by
  sorry

end min_distance_when_alpha_2pi_3_range_of_alpha_for_intersection_l610_610632


namespace remainder_of_12345678910_div_101_l610_610029

theorem remainder_of_12345678910_div_101 :
  12345678910 % 101 = 31 :=
sorry

end remainder_of_12345678910_div_101_l610_610029


namespace correlation_coefficient_is_one_l610_610280

noncomputable def correlation_coefficient (x_vals y_vals : List ℝ) : ℝ := sorry

theorem correlation_coefficient_is_one 
  (n : ℕ) 
  (x y : Fin n → ℝ) 
  (h1 : n ≥ 2) 
  (h2 : ∃ i j, i ≠ j ∧ x i ≠ x j) 
  (h3 : ∀ i, y i = 3 * x i + 1) : 
  correlation_coefficient (List.ofFn x) (List.ofFn y) = 1 := 
sorry

end correlation_coefficient_is_one_l610_610280


namespace symmetric_points_addition_l610_610705

theorem symmetric_points_addition (m n : ℤ) (h₁ : m = 2) (h₂ : n = -3) : m + n = -1 := by
  rw [h₁, h₂]
  norm_num

end symmetric_points_addition_l610_610705


namespace min_surface_area_of_spherical_blank_l610_610897

theorem min_surface_area_of_spherical_blank
  (angle_A : ℝ)
  (BC : ℝ)
  (PA : ℝ)
  (h_angle_A : angle_A = 150)
  (h_BC : BC = 3)
  (h_PA : PA = 2)
  (h_perpendicular : ⊥ PA (EuclideanSpace, A, B, C)) :
  ∃ (radius r : ℝ), r = sqrt (9 + 1) ∧ 4 * π * r^2 = 40 * π :=
begin
  sorry
end

end min_surface_area_of_spherical_blank_l610_610897


namespace natasha_quarters_l610_610767

-- Define the conditions for the number of quarters Natasha has
def quarters_condition (q : ℕ) : Prop := 
  40 < q ∧ q < 800 ∧ 
  q % 4 = 2 ∧ q % 5 = 2 ∧ q % 6 = 2

-- Define the set of valid quarters
def valid_quarters_set : set ℕ :=
  {q | ∃ k, 1 ≤ k ∧ k ≤ 13 ∧ q = 60 * k + 2}

-- Prove that given the conditions, the quarters are in the valid set
theorem natasha_quarters (q : ℕ) :
  quarters_condition q → q ∈ valid_quarters_set :=
by
  sorry

end natasha_quarters_l610_610767


namespace reinforcement_size_l610_610883

theorem reinforcement_size (initial_men : ℕ) (initial_days : ℕ) (days_before_reinforcement : ℕ) (days_remaining : ℕ) (reinforcement : ℕ) : 
  initial_men = 150 → initial_days = 31 → days_before_reinforcement = 16 → days_remaining = 5 → (150 * 15) = (150 + reinforcement) * 5 → reinforcement = 300 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end reinforcement_size_l610_610883


namespace largest_n_for_inequality_l610_610024

theorem largest_n_for_inequality :
  ∃ n : ℕ, 3 * n^2007 < 3^4015 ∧ ∀ m : ℕ, 3 * m^2007 < 3^4015 → m ≤ 8 ∧ n = 8 :=
by
  sorry

end largest_n_for_inequality_l610_610024


namespace cats_count_l610_610887

-- Definitions based on conditions
def heads_eqn (H C : ℕ) : Prop := H + C = 15
def legs_eqn (H C : ℕ) : Prop := 2 * H + 4 * C = 44

-- The main proof problem
theorem cats_count (H C : ℕ) (h1 : heads_eqn H C) (h2 : legs_eqn H C) : C = 7 :=
by
  sorry

end cats_count_l610_610887


namespace win_sector_area_l610_610064

theorem win_sector_area (r : ℝ) (h1 : r = 8) (h2 : (1 / 4) = 1 / 4) : 
  ∃ (area : ℝ), area = 16 * Real.pi := 
by
  existsi (16 * Real.pi); exact sorry

end win_sector_area_l610_610064


namespace distinct_sets_gcd_lcm_same_l610_610247

theorem distinct_sets_gcd_lcm_same :
  ∃ (A B : Finset ℕ), A ≠ B ∧ A.card = 10 ∧ B.card = 10 ∧
  (∀ x y ∈ A, ∃ z w ∈ B, gcd x y = gcd z w ∧ lcm x y = lcm z w) :=
sorry

end distinct_sets_gcd_lcm_same_l610_610247


namespace bouquet_cost_l610_610354

-- Define the conditions as hypotheses
def michael_has : ℕ := 50
def needs_more : ℕ := 11
def cake_cost : ℕ := 20
def balloon_cost : ℕ := 5

-- Define the final proof statement
theorem bouquet_cost : 
  let total_cost := michael_has + needs_more in
  let non_bouquet_cost := cake_cost + balloon_cost in
  total_cost - non_bouquet_cost = 36 :=
by
  -- Proof omitted
  sorry

end bouquet_cost_l610_610354


namespace binom_19_11_value_l610_610126

theorem binom_19_11_value :
  (binom 19 11) = 82654 :=
by
  have h1 : (binom 17 9) = 24310 := by sorry
  have h2 : (binom 17 7) = 19448 := by sorry
  -- Here we'll use the conditions in h1 and h2 to build the proof.
  sorry

end binom_19_11_value_l610_610126


namespace constant_term_eq_neg252_l610_610154

theorem constant_term_eq_neg252 :
  let f := (fun x : ℝ ↦ x + 1/x - 2) in
  (∑ i in (finset.range 6), 
    (nat.choose 5 i) * f (x)^(5 - i) * (-2)^i) 
  = 252 then constant_term_eq_neg252 sorry

end constant_term_eq_neg252_l610_610154


namespace bunyakovsky_hit_same_sector_l610_610116

variable {n : ℕ} (p : Fin n → ℝ)

theorem bunyakovsky_hit_same_sector (h_sum : ∑ i in Finset.univ, p i = 1) :
  (∑ i in Finset.univ, (p i)^2) >
  (∑ i in Finset.univ, (p i) * (p (Fin.rotate 1 i))) := 
sorry

end bunyakovsky_hit_same_sector_l610_610116


namespace cost_of_gas_correct_l610_610763

def odometer_start := 82435
def odometer_end := 82475
def fuel_consumption_rate := 25 -- miles per gallon
def price_per_gallon := 3.75 -- dollars per gallon

def distance_traveled := odometer_end - odometer_start
def gallons_used := distance_traveled / fuel_consumption_rate
def cost_of_gas := gallons_used * price_per_gallon

theorem cost_of_gas_correct :
  cost_of_gas = 6.00 := by
  sorry

end cost_of_gas_correct_l610_610763


namespace problem_a_impossible_problem_b_possible_l610_610515

-- Definitions based on the given conditions
def is_odd_row (table : ℕ → ℕ → bool) (n : ℕ) (r : ℕ) : Prop :=
  ∑ c in finset.range n, if table r c then 1 else 0 % 2 = 1

def is_odd_column (table : ℕ → ℕ → bool) (n : ℕ) (c : ℕ) : Prop :=
  ∑ r in finset.range n, if table r c then 1 else 0 % 2 = 1

-- Problem(a): No existence of 20 odd rows and 15 odd columns in any square table
theorem problem_a_impossible (table : ℕ → ℕ → bool) (n : ℕ) :
  (∃r_set c_set, r_set.card = 20 ∧ c_set.card = 15 ∧
  ∀ r ∈ r_set, is_odd_row table n r ∧ ∀ c ∈ c_set, is_odd_column table n c) → false :=
sorry

-- Problem(b): Existence of a 16 x 16 table with 126 crosses where all rows and columns are odd
theorem problem_b_possible : 
  ∃ (table : ℕ → ℕ → bool), 
  (∑ r in finset.range 16, ∑ c in finset.range 16, if table r c then 1 else 0) = 126 ∧
  (∀ r, is_odd_row table 16 r) ∧
  (∀ c, is_odd_column table 16 c) :=
sorry

end problem_a_impossible_problem_b_possible_l610_610515


namespace math_proof_problem_l610_610657

noncomputable def f (x a b : ℝ) : ℝ := log x - a * x + b / x

def f_prop_1 (a b : ℝ) (h : ∀ x, f x a b + f (1/x) a b = 0) : Prop :=
  ∀ x, f x a b = log x + 2 * x - 2 / x -> f 1 a b = 0 -> a = 2

def f_prop_2 (a : ℝ) (h : 0 < a ∧ a < 1) (h1 : a = 2) : Prop :=
  f (a^2 / 2) a 2 > 0

def f_prop_3 (a : ℝ) (h : ∀ x, f x a 2 + f (1/x) a 2 = 0) : Prop :=
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ f x1 a 2 = 0 ∧ f x2 a 2 = 0 ∧ f x3 a 2 = 0) -> 
  0 < a ∧ a < 1/2

theorem math_proof_problem (a b : ℝ) (h_cond : ∀ x, f x a b + f (1/x) a b = 0) (h_tangent : ∀ a b, f 1 a b = 0 ∧ f' 1 a b passes through (2, 5)):
  f_prop_1 a b h_cond ∧
  (0 < a ∧ a < 1) -> f_prop_2 a (0 < a ∧ a < 1) ∧
  f_prop_3 a h_cond :=
by sorry

end math_proof_problem_l610_610657


namespace females_wearing_glasses_l610_610481

theorem females_wearing_glasses (total_population : ℕ) 
                                (number_of_males : ℕ) 
                                (percent_females_with_glasses : ℚ) 
                                (male_population_condition : number_of_males = 2000)
                                (total_population_condition : total_population = 5000)
                                (percent_condition : percent_females_with_glasses = 0.30) :
                                total_population - number_of_males = 3000 ∧ 
                                ((total_population - number_of_males) * percent_females_with_glasses).natAbs = 900 := 
by 
  rw [total_population_condition, male_population_condition] 
  simp
  sorry

end females_wearing_glasses_l610_610481


namespace zeros_of_f_trig_identity_1_trig_identity_2_l610_610991

open Real

theorem zeros_of_f :
  (∃ x : ℝ, (6*x^2 + x - 1 = 0) ∧ (x = 1/3)) ∧ (∃ x : ℝ, (6*x^2 + x - 1 = 0) ∧ (x = -1/2)) := by
  sorry

theorem trig_identity_1 (α : ℝ) (h : 0 < α ∧ α < π/2) (h1 : sin α = 1/3) :
  (tan (π + α) * cos (-α) / (cos (π/2 - α) * sin (π - α)) = 3) := by
  sorry

theorem trig_identity_2 (α : ℝ) (h : 0 < α ∧ α < π/2) (h1 : sin α = 1/3) :
  sin (α + π/6) = (sqrt(3) + 2 * sqrt(2)) / 6 := by
  sorry

end zeros_of_f_trig_identity_1_trig_identity_2_l610_610991


namespace cardinality_PQ_l610_610737

def P : Set ℕ := {3, 4, 5}
def Q : Set ℕ := {4, 5, 6, 7}

def cartesian_product (A B : Set ℕ) : Set (ℕ × ℕ) :=
  {p | ∃ a b, p = (a, b) ∧ a ∈ A ∧ b ∈ B}

def PQ : Set (ℕ × ℕ) := cartesian_product P Q

theorem cardinality_PQ : (PQ.to_finset.card = 12) := by
  sorry

end cardinality_PQ_l610_610737


namespace greatest_four_digit_multiple_of_17_l610_610023

theorem greatest_four_digit_multiple_of_17 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n ∧ n = 9996 :=
by
  use 9996
  split
  { exact by norm_num }
  split
  { exact by norm_num [lt.sub (ofNat 10000) (ofNat 4)] }
  split
  { exact by norm_num }
  { exact by norm_num [dvd_of_mod_eq_zero (by norm_num : 10000 % 17 = 4)] }

end greatest_four_digit_multiple_of_17_l610_610023


namespace sum_k_log_term_l610_610136

-- Define the main theorem statement
theorem sum_k_log_term : 
  ∑ k in Finset.range 1500, k * (⌈Real.log k / Real.log (Real.sqrt 3)⌉ - ⌊Real.log k / Real.log (Real.sqrt 3)⌋) = 1124657 := 
sorry

end sum_k_log_term_l610_610136


namespace infinite_solutions_l610_610586

theorem infinite_solutions (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ (k : ℕ), x = k^3 + 1 ∧ y = (k^3 + 1) * k := 
sorry

end infinite_solutions_l610_610586


namespace product_of_reals_condition_l610_610147

theorem product_of_reals_condition (x : ℝ) (h : x + 1/x = 3 * x) : 
  ∃ x1 x2 : ℝ, x1 + 1/x1 = 3 * x1 ∧ x2 + 1/x2 = 3 * x2 ∧ x1 * x2 = -1/2 := 
sorry

end product_of_reals_condition_l610_610147


namespace consecutive_rolls_probability_l610_610362

theorem consecutive_rolls_probability : 
  let total_outcomes := 36
  let consecutive_events := 10
  (consecutive_events / total_outcomes : ℚ) = 5 / 18 :=
by
  sorry

end consecutive_rolls_probability_l610_610362


namespace minimum_value_of_f_l610_610658

-- Define the function y = f(x)
def f (x : ℝ) : ℝ := x^2 + 8 * x + 25

-- We need to prove that the minimum value of f(x) is 9
theorem minimum_value_of_f : ∃ x : ℝ, f x = 9 ∧ ∀ y : ℝ, f y ≥ 9 :=
by
  sorry

end minimum_value_of_f_l610_610658


namespace equation_has_no_solution_l610_610186

theorem equation_has_no_solution (k : ℝ) : ¬ (∃ x : ℝ , (x ≠ 3 ∧ x ≠ 4) ∧ (x - 1) / (x - 3) = (x - k) / (x - 4)) ↔ k = 2 :=
by
  sorry

end equation_has_no_solution_l610_610186


namespace part_a_part_b_l610_610505

-- Definitions based on the problem conditions
def is_square_table (n : ℕ) (table : List (List Bool)) : Prop := 
  table.length = n ∧ ∀ row, row ∈ table → row.length = n

def is_odd_row (row : List Bool) : Prop := 
  row.count (λ x => x = true) % 2 = 1

def is_odd_column (n : ℕ) (table : List (List Bool)) (c : ℕ) : Prop :=
  n > 0 ∧ c < n ∧ (List.count (λ row => row.get! c = true) table) % 2 = 1

-- Part (a) statement: Prove it's impossible to have exactly 20 odd rows and 15 odd columns
theorem part_a (n : ℕ) (table : List (List Bool)) :
  n = 16 → is_square_table n table → 
  (List.count is_odd_row table) = 20 → 
  ((List.range n).count (is_odd_column n table)) = 15 → 
  False := 
sorry

-- Part (b) statement: Prove that it's possible to arrange 126 crosses in a 16x16 table with all odd rows and columns
theorem part_b (table : List (List Bool)) :
  is_square_table 16 table →
  ((table.map (λ row => row.count (λ x => x = true))).sum = 126) →
  (∀ row, row ∈ table → is_odd_row row) →
  (∀ c, c < 16 → is_odd_column 16 table c) →
  True :=
sorry

end part_a_part_b_l610_610505


namespace revenue_increase_37_655_percent_l610_610408

variables {P S : ℝ} (hP : P > 0) (hS : S > 0)

-- Define the original and new conditions
def original_revenue := P * S
def increased_price := 1.9 * P
def decreased_sales := 0.7 * S
def new_revenue := increased_price * decreased_sales
def discounted_price := 1.71 * P
def loyalty_revenue := discounted_price * decreased_sales
def final_price_with_tax := 1.9665 * P
def final_revenue_with_tax := final_price_with_tax * decreased_sales
def revenue_effect := final_revenue_with_tax - original_revenue

-- The theorem statement in Lean 4
theorem revenue_increase_37_655_percent :
  revenue_effect / original_revenue = 0.37655 :=
by
  /- Proof goes here -/
  sorry

end revenue_increase_37_655_percent_l610_610408


namespace triangle_angle_l610_610221

-- Definitions of the conditions and theorem
variables {a b c : ℝ}
variables {A B C : ℝ}

theorem triangle_angle (h : b^2 + c^2 - a^2 = bc)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hA : 0 < A) (hA_max : A < π) :
  A = π / 3 :=
by
  sorry

end triangle_angle_l610_610221


namespace cube_root_fraction_l610_610923

theorem cube_root_fraction : 
  (∛(5 / 6 * 20.25) = (3 * 5^(2/3)) / 2) :=
by
  have h1 : 20.25 = 81 / 4 := by norm_num
  have h2 : (5 / 6) * (81 / 4) = 135 / 8 := by norm_num
  rw [←h1, ←h2]
  sorry

end cube_root_fraction_l610_610923


namespace not_possible_20_odd_rows_15_odd_columns_possible_126_crosses_in_16x16_l610_610504

-- Define the general properties and initial conditions for the problems.
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Question a: Can there be exactly 20 odd rows and 15 odd columns in the table?
theorem not_possible_20_odd_rows_15_odd_columns (n : ℕ) (n_odd_rows : ℕ) (n_odd_columns : ℕ) (crosses : ℕ) (h_crosses_odd : is_odd crosses) 
  (h_odd_rows : n_odd_rows = 20) (h_odd_columns : n_odd_columns = 15) : 
  false := 
sorry

-- Question b: Can 126 crosses be arranged in a \(16 \times 16\) table so that all rows and columns are odd?
theorem possible_126_crosses_in_16x16 (crosses : ℕ) (n : ℕ) (h_crosses : crosses = 126) (h_table_size : n = 16) : 
  ∃ (table : matrix ℕ ℕ bool), 
    (∀ i : fin n, is_odd (count (λ j, table i j) (list.range n))) ∧
    (∀ j : fin n, is_odd (count (λ i, table i j) (list.range n))) :=
sorry

end not_possible_20_odd_rows_15_odd_columns_possible_126_crosses_in_16x16_l610_610504


namespace second_tallest_creature_l610_610076

-- Define the heights of the creatures as real numbers
def height_giraffe : ℝ := 2.3
def height_baby : ℝ := 9 / 10
def height_gorilla : ℝ := 1.8
def height_dinosaur : ℝ := 2.5

-- Define the problem statement to identify the second tallest creature
theorem second_tallest_creature :
  (height_giraffe ∈ [height_giraffe, height_baby, height_gorilla, height_dinosaur].sort (≤).reverse.tail.head) := 
by sorry

end second_tallest_creature_l610_610076


namespace bird_nest_difference_l610_610831

theorem bird_nest_difference : 
  (number_of_birds number_of_nests : ℕ) (number_of_birds = 6) (number_of_nests = 3) :
  number_of_birds - number_of_nests = 3 := 
by
  sorry

end bird_nest_difference_l610_610831


namespace max_value_of_f_min_value_of_f_l610_610173

open Real

-- First statement: Maximum value of f(x)
theorem max_value_of_f (x : ℝ) (k : ℤ) : 
  let f : ℝ → ℝ := λ x, sin (x + π / 6)
  x = 2 * (k : ℝ) * π + π / 3 →
  f x = 1 := by
  intros
  sorry

-- Second statement: Minimum value of f(x)
theorem min_value_of_f (x : ℝ) (k : ℤ) : 
  let f : ℝ → ℝ := λ x, sin (x + π / 6)
  x = 2 * (k : ℝ) * π - 2 * π / 3 →
  f x = -1 := by
  intros
  sorry

end max_value_of_f_min_value_of_f_l610_610173


namespace product_has_no_linear_term_l610_610270

theorem product_has_no_linear_term (m : ℝ) (h : ((x : ℝ) → (x - m) * (x - 3) = x^2 + 3 * m)) : m = -3 := 
by
  sorry

end product_has_no_linear_term_l610_610270


namespace banana_cost_l610_610371

theorem banana_cost (cost_total : ℝ) (cost_milk : ℝ) (num_cereal : ℝ) (cost_cereal_per_box : ℝ) (num_apples : ℝ) (cost_apple_per_unit : ℝ) (num_bananas : ℝ) (num_cookies : ℝ) (milk_multiplier : ℝ) (cost_banana_per_unit : ℝ):
  cost_total = 25 →
  cost_milk = 3 →
  num_cereal = 2 →
  cost_cereal_per_box = 3.5 →
  num_apples = 4 →
  cost_apple_per_unit = 0.5 →
  num_bananas = 4 →
  num_cookies = 2 →
  milk_multiplier = 2 →
  let cost_cookies_per_box := milk_multiplier * cost_milk in
  let total_known_cost := cost_milk + (num_cereal * cost_cereal_per_box) + (num_apples * cost_apple_per_unit) + (num_cookies * cost_cookies_per_box) in
  let remaining_cost := cost_total - total_known_cost in
  cost_banana_per_unit = remaining_cost / num_bananas :=
begin
  intros,
  sorry
end

end banana_cost_l610_610371


namespace Q_coordinates_l610_610299

-- Define the coordinates of point P in the space.
def P : ℝ × ℝ × ℝ := (1, 2, 3)

-- Define the projection of point P onto the xOy plane.
def projection_onto_xOy (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (P.1, P.2, 0)

-- Define point Q as the projection of P onto the xOy plane.
def Q : ℝ × ℝ × ℝ := projection_onto_xOy P

-- Theorem statement that Q has the correct coordinates.
theorem Q_coordinates : Q = (1, 2, 0) :=
  sorry

end Q_coordinates_l610_610299


namespace probability_increasing_l610_610933

noncomputable def p (x : ℝ) : ℝ := sorry -- This is a placeholder for the actual function definition.

theorem probability_increasing (E : set ℝ) (hE1 : [0,1] ⊆ E) (hE2 : E ⊆ [0, +∞]) (hE3 : is_compact E) : 
  ∀ x y : ℝ, -1 ≤ x ∧ x < 0 ∧ -1 ≤ y ∧ y < 0 ∧ x < y → p x ≤ p y := 
sorry

end probability_increasing_l610_610933


namespace median_locus_of_triangle_l610_610628

-- Definitions for geometric elements
variables (O A B C : Point)
-- Conditions of the problem
axiom vertex_condition : is_vertex O
axiom point_on_edge_A : on_edge A O
axiom point_on_edge_B : on_edge B O
axiom point_on_edge_C : on_edge C O

-- The locus to prove, similar to a formal theorem
theorem median_locus_of_triangle :
  ∀ (O A B C : Point) (is_vertex O) (on_edge A O) (on_edge B O) (on_edge C O),
  locus_of_medians_triangle O A B C = plane_parallel_to_face_OBC_and_one_third_away_from_A :=
sorry

end median_locus_of_triangle_l610_610628


namespace compound_interest_paid_is_approximately_l610_610843

noncomputable def principal_amount : ℝ := 
  let r := 0.10
  let n := 1
  let t := 2
  let total_interest := 147.0000000000001
  let A := P + total_interest
  let P := A / (1 + r/n)^(n*t)
  P

theorem compound_interest_paid_is_approximately :
  abs (principal_amount - 700) < 1 :=
sorry

end compound_interest_paid_is_approximately_l610_610843


namespace not_possible_20_odd_rows_15_odd_columns_possible_126_crosses_in_16x16_l610_610502

-- Define the general properties and initial conditions for the problems.
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Question a: Can there be exactly 20 odd rows and 15 odd columns in the table?
theorem not_possible_20_odd_rows_15_odd_columns (n : ℕ) (n_odd_rows : ℕ) (n_odd_columns : ℕ) (crosses : ℕ) (h_crosses_odd : is_odd crosses) 
  (h_odd_rows : n_odd_rows = 20) (h_odd_columns : n_odd_columns = 15) : 
  false := 
sorry

-- Question b: Can 126 crosses be arranged in a \(16 \times 16\) table so that all rows and columns are odd?
theorem possible_126_crosses_in_16x16 (crosses : ℕ) (n : ℕ) (h_crosses : crosses = 126) (h_table_size : n = 16) : 
  ∃ (table : matrix ℕ ℕ bool), 
    (∀ i : fin n, is_odd (count (λ j, table i j) (list.range n))) ∧
    (∀ j : fin n, is_odd (count (λ i, table i j) (list.range n))) :=
sorry

end not_possible_20_odd_rows_15_odd_columns_possible_126_crosses_in_16x16_l610_610502


namespace problem_l610_610675

theorem problem (x y z : ℕ) (hx : x < 9) (hy : y < 9) (hz : z < 9) 
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 9])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 9])
  (h3 : 2 * x + y + 3 * z ≡ 5 [MOD 9]) :
  (x * y * z % 9 = 0) :=
sorry

end problem_l610_610675


namespace solve_for_x_l610_610368

theorem solve_for_x : ∀ x : ℝ, (x - 7)^5 = (1 / 32) ^ (-5 / 5) → x = 9 := by
  intro x
  sorry

end solve_for_x_l610_610368


namespace minimum_dot_product_l610_610664

variables {α β : Type} [Add α] [Mul α] [Norm α] [InnerProduct α] 
variables (a b : α)

-- Define the two conditions
theorem minimum_dot_product (α β : α) (conditions : NormedAdd β α) :
  ∥α + 2 • β∥ = 3 ∧ ∥2 • α + 3 • β∥ = 4 →
  (α ⬝ β) = -170 :=
sorry

end minimum_dot_product_l610_610664


namespace rahid_tower_heights_l610_610782

-- Define the lengths of the blocks
def block_lengths : List ℕ := [4, 6, 10]

-- Define the function to calculate the distinct tower heights
def distinct_tower_heights : List ℕ :=
  List.eraseDups $ List.map (fun l => List.sum l) $ List.nthComb 3 block_lengths

-- Number of distinct tower heights is 9
theorem rahid_tower_heights : List.length distinct_tower_heights = 9 := 
  by
    sorry

end rahid_tower_heights_l610_610782


namespace simplify_expression_is_correct_l610_610791

-- Given: a is an element in an appropriate domain where exponentiation is defined
variable (a : Type) [Monoid a] (x : a)

-- Define powers and multiplication in this context
noncomputable def simplify_expression (x : a) : a := (x^5 * x^3) * (x^2)^4

-- Statement to prove
theorem simplify_expression_is_correct : simplify_expression x = x^16 :=
by
  sorry

end simplify_expression_is_correct_l610_610791


namespace cos_2x_min_value_l610_610232

noncomputable def f (x : ℝ) : ℝ := 9 / (8 * Real.cos (2 * x) + 16) - (Real.sin x)^2

theorem cos_2x_min_value (x : ℝ) (h : ∀ y : ℝ, f y ≥ f x) : Real.cos (2 * x) = -1/2 :=
begin
  -- Focusing on the equivalence of the problem statement, the proof is omitted.
  sorry
end

end cos_2x_min_value_l610_610232


namespace net_profit_Elizabeth_l610_610572

theorem net_profit_Elizabeth
  (initial_price : ℝ := 6.00) (discounted_price : ℝ := 4.00) (ingredient_cost : ℝ := 3.00)
  (total_bags : ℕ := 20) (sold_bags_initial : ℕ := 15) :
  let total_revenue := sold_bags_initial * initial_price + (total_bags - sold_bags_initial) * discounted_price;
      total_cost := total_bags * ingredient_cost 
  in total_revenue - total_cost = 50.00 :=
by 
  sorry

end net_profit_Elizabeth_l610_610572


namespace polynomial_solution_l610_610576

theorem polynomial_solution {P : ℝ[X]} :
  (∀ x : ℝ, 16 * P.eval (x^2) = P.eval (2*x)^2) ↔ 
  (∃ n : ℕ, ∀ x : ℝ, P.eval x = 16 * (x / 4) ^ n) :=
sorry

end polynomial_solution_l610_610576


namespace angle_BAP_is_13_degrees_l610_610463

theorem angle_BAP_is_13_degrees
  (A B C P : Type)
  [AddGroup A] [HasSmul ℝ A] [vector_space ℝ A]
  (h_iso : dist A B = dist A C)
  (h_BCP : angle B C P = 30)
  (h_APB : angle A P B = 150)
  (h_CAP : angle C A P = 39) :
  angle B A P = 13 := sorry

end angle_BAP_is_13_degrees_l610_610463


namespace leading_coefficient_polynomial_l610_610931

theorem leading_coefficient_polynomial :
  let p := 5 * (x^5 - 2 * x^4 + 3 * x^3) - 6 * (x^5 + x^3 + x) + 3 * (3 * x^5 - x^4 + 4 * x^2 + 2)
  in leading_coeff p = 8 :=
by
  -- proof goes here
  sorry

end leading_coefficient_polynomial_l610_610931


namespace maximum_instantaneous_power_l610_610393

noncomputable def sailboat_speed : ℝ → ℝ → ℝ → ℝ 
  | C, S, v_0, v =>
  (C * S * ((v_0 - v) ^ 2)) / 2

theorem maximum_instantaneous_power (C ρ : ℝ)
  (S : ℝ := 5)
  (v_0 : ℝ := 6) :
  let v := (2 : ℝ) 
  (sailboat_speed C S v_0(v) * v)
  = (C * 5 * ρ / 2 -> v = 2) :=
by
  sorry

end maximum_instantaneous_power_l610_610393


namespace divisibility_check_l610_610606

variable (d : ℕ) (h1 : d % 2 = 1) (h2 : d % 5 ≠ 0)
variable (δ : ℕ) (h3 : ∃ m : ℕ, 10 * δ + 1 = m * d)
variable (N : ℕ)

def last_digit (N : ℕ) : ℕ := N % 10
def remove_last_digit (N : ℕ) : ℕ := N / 10

theorem divisibility_check (h4 : ∃ N' u : ℕ, N = 10 * N' + u ∧ N = N' * 10 + u ∧ N' = remove_last_digit N ∧ u = last_digit N)
  (N' : ℕ) (u : ℕ) (N1 : ℕ) (h5 : N1 = N' - δ * u) :
  d ∣ N1 → d ∣ N := by
  sorry

end divisibility_check_l610_610606


namespace volume_SPQR_l610_610360

noncomputable def volume_of_pyramid_SPQR : ℝ := 450

axiom P : Type
axiom Q : Type
axiom R : Type
axiom S : Type

axiom SP : ℝ := 15
axiom SQ : ℝ := 15
axiom SR : ℝ := 12

axiom SP_perpendicular_SQ : true
axiom SP_perpendicular_SR : true
axiom SQ_perpendicular_SR : true

theorem volume_SPQR (P Q R S : Type) (SP SQ SR : ℝ)
  (h1 : SP = 15) (h2 : SQ = 15) (h3 : SR = 12)
  (h4 : SP_perpendicular_SQ) (h5 : SP_perpendicular_SR) (h6 : SQ_perpendicular_SR) :
  volume_of_pyramid_SPQR = 450 :=
begin
  sorry
end

end volume_SPQR_l610_610360


namespace tennis_tournament_handshakes_l610_610543

theorem tennis_tournament_handshakes :
  let teams := 4
  let players_per_team := 2
  let total_players := teams * players_per_team
  let handshakes_per_player := total_players - players_per_team in
  (total_players * handshakes_per_player) / 2 = 24 :=
by
  let teams := 4
  let players_per_team := 2
  let total_players := teams * players_per_team
  let handshakes_per_player := total_players - players_per_team
  have h : (total_players * handshakes_per_player) / 2 = 24 := sorry
  exact h

end tennis_tournament_handshakes_l610_610543


namespace find_valid_triplets_l610_610157

def is_prime (n : ℕ) : Prop := nat.prime n

def satisfies_conditions (p q r : ℕ) : Prop := 
  p ≥ q ∧ q ≥ r ∧ 
  (∃ a b, (a = p ∨ a = q ∨ a = r) ∧ (b = p ∨ b = q ∨ b = r) ∧ a ≠ b ∧ is_prime a ∧ is_prime b) ∧
  ∃ k : ℕ, k * p * q * r = (p + q + r) ^ 2

theorem find_valid_triplets : 
  ∃ (p q r : ℕ), 
    satisfies_conditions p q r ∧ 
    ((p = 3 ∧ q = 3 ∧ r = 3) ∨ 
     (p = 2 ∧ q = 2 ∧ r = 4) ∨ 
     (p = 3 ∧ q = 3 ∧ r = 12) ∨ 
     (p = 3 ∧ q = 2 ∧ r = 1) ∨ 
     (p = 3 ∧ q = 2 ∧ r = 25)) :=
sorry

end find_valid_triplets_l610_610157


namespace wrapping_paper_l610_610788

theorem wrapping_paper (total_used : ℚ) (decoration_used : ℚ) (presents : ℕ) (other_presents : ℕ) (individual_used : ℚ) 
  (h1 : total_used = 5 / 8) 
  (h2 : decoration_used = 1 / 24) 
  (h3 : presents = 4) 
  (h4 : other_presents = 3) 
  (h5 : individual_used = (5 / 8 - 1 / 24) / 3) : 
  individual_used = 7 / 36 := 
by
  -- The theorem will be proven here.
  sorry

end wrapping_paper_l610_610788


namespace one_eighth_percent_of_160_plus_half_l610_610447

theorem one_eighth_percent_of_160_plus_half :
  ((1 / 8) / 100 * 160) + 0.5 = 0.7 :=
  sorry

end one_eighth_percent_of_160_plus_half_l610_610447


namespace range_of_y_l610_610259

theorem range_of_y (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 132) : y ∈ Ioo (-12) (-11) :=
sorry

end range_of_y_l610_610259


namespace radius_of_second_sphere_l610_610828

theorem radius_of_second_sphere (r1 r2 w1 w2 : ℝ) (h1 : r1 = 0.15) (h2 : w1 = 8) (h3 : w2 = 32)
    (h4 : ∀ a b, w1 / (4 * Real.pi * r1^2) = w2 / (4 * Real.pi * b^2) → a = b) : r2 = 0.3 :=
by
  have h_area1 : 4 * Real.pi * r1^2 = 4 * Real.pi * (0.15)^2, from congrArg (4 * Real.pi * ·) h1
  have h_area2 : 4 * Real.pi * r2^2 = 4 * Real.pi * r2^2, from rfl
  have h_proportionality : w1 / (4 * Real.pi * (0.15)^2) = w2 / (4 * Real.pi * r2^2), sorry
  apply h4 (4 * Real.pi * (0.15)^2) (4 * Real.pi * r2^2) h_proportionality
  sorry

end radius_of_second_sphere_l610_610828


namespace least_possible_g_l610_610800

def tetrahedron : Type := sorry
def point : Type := sorry

variables (E F G H : point)
variables (EH FG EG FH EF GH : ℝ)
variables (edge_cond : EH = 30 ∧ FG = 30 ∧ EG = 40 ∧ FH = 40 ∧ EF = 48 ∧ GH = 48)

def g (Y : point) : ℝ := sorry

theorem least_possible_g (Y : point) (h : tetrahedron) :
  g(E, F, G, H, Y) = 4 * Real.sqrt 578 := sorry

end least_possible_g_l610_610800


namespace min_dist_C1_to_l_l610_610707

-- Define the parametric line equation
def line_l (t : ℝ) : ℝ × ℝ := (-sqrt 5 + (sqrt 2)/2 * t, sqrt 5 + (sqrt 2)/2 * t)

-- Parametric equation of line in Cartesian form
def line_eq (x y : ℝ) : Prop :=
  x + y = 0

-- Polar equation of the curve C
def polar_eq (θ : ℝ) : ℝ :=
  4 * Real.cos θ

-- Cartesian equation of the curve C
def cartesian_eq (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + y ^ 2 = 4

-- We define curve C_1 after transformations
def C1_eq (x y : ℝ) : Prop :=
  4 * x ^ 2 + y ^ 2 = 4

-- Minimum distance between points on C_1 and line l
def min_distance (x y : ℝ) : ℝ :=
  (abs (x + y) / Real.sqrt 2)

/-- Proof that the minimum distance from points on curve C_1 to line l is 0 -/
theorem min_dist_C1_to_l : ∀ x y : ℝ, C1_eq x y → line_eq x y → min_distance x y = 0 :=
sorry

end min_dist_C1_to_l_l610_610707


namespace math_proof_problem_l610_610647

open Real

theorem math_proof_problem :
  ∃ (a b c : ℤ), (sqrt (2 * (a : ℝ) + 3) = 3 ∨ sqrt (2 * (a : ℝ) + 3) = -3)
  ∧ (cbrt (3 * (b : ℝ) - 2 * (c : ℝ)) = 2)
  ∧ (c = ⌊sqrt 6⌋)
  ∧ (a = 3 ∧ b = 4 ∧ c = 2)
  ∧ (sqrt ((a + 6 * b - c) : ℝ) = 5) :=
  sorry

end math_proof_problem_l610_610647


namespace inequality_correct_l610_610613

theorem inequality_correct (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - c > b - c := by
  sorry

end inequality_correct_l610_610613


namespace quadrilateral_area_l610_610237

theorem quadrilateral_area 
  (p : ℝ) (hp : p > 0)
  (P : ℝ × ℝ) (hP : P = (1, 1 / 4))
  (focus : ℝ × ℝ) (hfocus : focus = (0, 1))
  (directrix : ℝ → Prop) (hdirectrix : ∀ y, directrix y ↔ y = 1)
  (F : ℝ × ℝ) (hF : F = (0, 1))
  (M : ℝ × ℝ) (hM : M = (0, 1))
  (Q : ℝ × ℝ) 
  (PQ : ℝ)
  (area : ℝ) 
  (harea : area = 13 / 8) :
  ∃ (PQMF : ℝ), PQMF = 13 / 8 :=
sorry

end quadrilateral_area_l610_610237


namespace T_n_bound_l610_610627

def sequence_an : ℕ+ → ℕ
| 1     := 2
| (n+1) := have Sn := list.sum (list.map sequence_an (list.range (n+1))) in Sn - n + 3

def sequence_bn (n : ℕ+) : ℚ :=
  let Sn := (list.sum (list.map sequence_an (list.range n))) in
  ↑n / (Sn - n + 2)

def Sn (n : ℕ) : ℝ :=
  if n = 0 then 0 else (list.sum (list.map sequence_an (list.range n)))

noncomputable def T_n (n : ℕ) : ℝ :=
  if n = 0 then 0 else (list.range n).sum (λ i, sequence_bn ⟨i+1, nat.succ_pos' i⟩)

theorem T_n_bound (n : ℕ) : T_n (n + 1) < 4/3 :=
sorry

end T_n_bound_l610_610627


namespace number_of_handshakes_l610_610540

-- Definitions based on the conditions:
def number_of_teams : ℕ := 4
def number_of_women_per_team : ℕ := 2
def total_women : ℕ := number_of_teams * number_of_women_per_team

-- Each woman shakes hands with all others except her partner
def handshakes_per_woman : ℕ := total_women - 1 - (number_of_women_per_team - 1)

-- Calculate total handshakes, considering each handshake is counted twice
def total_handshakes : ℕ := (total_women * handshakes_per_woman) / 2

-- Statement to prove
theorem number_of_handshakes :
  total_handshakes = 24 := 
sorry

end number_of_handshakes_l610_610540


namespace modem_speed_ratio_l610_610306

/-- Given that it takes 25.5 minutes to download a file using modem A and 150 minutes using modem B,
    prove that the speed of modem B compared to modem A is approximately 16.99% -/
theorem modem_speed_ratio 
  (tA tB : ℝ)
  (h₁ : tA = 25.5)
  (h₂ : tB = 150) :
  (1 / (tB / tA)) * 100 ≈ 16.99 := 
sorry

end modem_speed_ratio_l610_610306


namespace length_of_other_parallel_side_l610_610926

theorem length_of_other_parallel_side (a b h : ℝ) (area : ℝ) (h_area : 323 = 1/2 * (20 + b) * 17) :
  b = 18 :=
sorry

end length_of_other_parallel_side_l610_610926


namespace room_height_l610_610806

theorem room_height (h : ℝ) :
  let length := 25
  let width := 15
  let door_area := 6 * 3
  let window_area := 4 * 3
  let total_area := 7248
  let cost_per_sqft := 8
  7248 = (80 * h - (door_area + 3 * window_area)) * cost_per_sqft → h = 12 :=
by
  intro h
  let length := 25
  let width := 15
  let door_area := 6 * 3
  let window_area := 4 * 3
  let total_area := 7248
  let cost_per_sqft := 8
  let perimeter := 2 * (length + width)
  let wall_area := perimeter * h
  have : 7248 = (wall_area - (door_area + 3 * window_area)) * cost_per_sqft,
    from sorry
  rw [perimeter, length, width, door_area, window_area, 7248, cost_per_sqft] at this,
  have : wall_area - (door_area + 3 * window_area) = 80 * h - 54,
    from sorry
  have : 7248 = (80 * h - 54) * 8,
    from sorry
  have : (80 * h - 54) = 7248 / 8,
    from sorry
  have : 80 * h = (7248 + 432) / 8,
    from sorry
  have : 80 * h = 7680,
    by linarith
  have : h = 7680 / 80,
    by linarith
  have : h = 12,
    by norm_num
  exact this

end room_height_l610_610806


namespace hexadecagon_area_hexadecagon_inscribed_area_l610_610884

theorem hexadecagon_area (P : ℝ) (A : ℝ) : P = 160 → A = 1400 :=
by
  have s : ℝ := P / 4
  have segment_length : ℝ := s / 4
  have total_square_area : ℝ := s * s
  have triangle_area : ℝ := 4 * (1 / 2 * segment_length * segment_length)
  have hexadecagon_area : ℝ := total_square_area - triangle_area
  have hexadecagon_area_proof : hexadecagon_area = A
  { -- This is where the proof would go
    sorry }

theorem hexadecagon_inscribed_area : 
∀ (P : ℝ), P = 160 → ∃ (A : ℝ), A = 1400 :=
by 
  intro P hP
  use 1400
  exact ⟨punit.star, by proxy ⟩
  sorry

end hexadecagon_area_hexadecagon_inscribed_area_l610_610884


namespace geometric_sequence_log_abs_sum_eq_58_l610_610216

theorem geometric_sequence_log_abs_sum_eq_58
  (a : ℕ → ℝ)
  (n : ℕ)
  (S : ℕ → ℝ)
  (h1 : a 1 = 32)
  (h2 : ∀ n, S n = (a 1 * (1 - real.geom_row_sum (a 2 / a 1) n)) / (1 - a 2 / a 1))
  (h3 : S 6 / S 3 = 65 / 64) :
  (finset.sum (finset.range 10) (λ n, abs (real.log 2 (a n)))) = 58 := 
sorry

end geometric_sequence_log_abs_sum_eq_58_l610_610216


namespace candy_ounces_correct_l610_610119

-- Defining the ounces in a bag of candy
def ounces_in_bag_of_candy (C : ℕ) : Prop :=
  let candy_cost := 1
  let chips_cost := 1.40
  let chips_ounces := 17
  let dollars := 7
  let max_ounces := 85
  let bags_of_candy := dollars / candy_cost
  let bags_of_chips := dollars / chips_cost
  let total_ounces_of_chips := chips_ounces * bags_of_chips
  let total_ounces_of_candy := C * bags_of_candy
  total_ounces_of_candy = max_ounces ∧ total_ounces_of_chips = max_ounces

theorem candy_ounces_correct : ounces_in_bag_of_candy 12 :=
  sorry

end candy_ounces_correct_l610_610119


namespace max_distance_sum_l610_610968

-- Define the ellipse and points
def is_point_on_ellipse (x y : ℝ) : Prop :=
  (x^2 / 25) + (y^2 / 9) = 1

def point_A : ℝ × ℝ := (4, 0)
def point_B : ℝ × ℝ := (2, 2)

-- Define the function that computes the distance between two points
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The maximum value of |MA| + |MB| for M on the ellipse
theorem max_distance_sum (M : ℝ × ℝ)
  (hM_on_ellipse : is_point_on_ellipse M.1 M.2) :
  dist M point_A + dist M point_B ≤ 10 + 2 * real.sqrt 10 :=
sorry

end max_distance_sum_l610_610968


namespace ellipse_major_axis_foci_on_y_axis_l610_610210

theorem ellipse_major_axis_foci_on_y_axis (m : ℝ) : 
  (∃ b a : ℝ, 2 * b = 8 ∧ b^2 = m ∧ a^2 = 10) → m = 16 := by
  intro h
  rcases h with ⟨b, a, major_axis, semi_major_axis, semi_minor_axis⟩
  have h1 : b = 4 := by 
    linarith
  rw [h1] at semi_major_axis
  norm_num at semi_major_axis
  exact semi_major_axis

end ellipse_major_axis_foci_on_y_axis_l610_610210


namespace integer_polynomial_roots_abs_sum_eq_104_l610_610940

theorem integer_polynomial_roots_abs_sum_eq_104 (p q r m : ℤ) (h1 : polynomial.eval (p : ℚ) (X^3 - 2022 * X + polynomial.C (m : ℚ)) = 0)
  (h2 : polynomial.eval (q : ℚ) (X^3 - 2022 * X + polynomial.C (m : ℚ)) = 0)
  (h3 : polynomial.eval (r : ℚ) (X^3 - 2022 * X + polynomial.C (m : ℚ)) = 0) :
  abs p + abs q + abs r = 104 :=
sorry

end integer_polynomial_roots_abs_sum_eq_104_l610_610940


namespace minimum_value_quadratic_function_l610_610403

noncomputable def quadratic_function (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

theorem minimum_value_quadratic_function : ∀ x, x ≥ 0 → quadratic_function x ≥ 1 :=
by
  sorry

end minimum_value_quadratic_function_l610_610403


namespace count_valid_pairs_l610_610942

theorem count_valid_pairs : 
  ∃ (n : ℕ), n = 131 ∧ 
  ∀ (x y : ℕ), (0 < x ∧ 0 < y ∧ (x^2 + y^2) % 11 = 0 ∧ (x^2 + y^2) / 11 ≤ 1991) ↔
  ∃ (x y : ℕ), x^2 + y^2 = 11 * (n * 165 + k) ∧ k < 11 :=
begin
  sorry
end

end count_valid_pairs_l610_610942


namespace find_magnitude_l610_610976

-- Given conditions
variable {α a b : ℝ}
variable {i : ℂ} (h : i = complex.I)

-- Define the condition that the conjugate of (α + i) / i is b + i
def conjugate_condition : Prop :=
  complex.conj ((α + i) / i) = b + complex.I

-- Statement: Prove that |a + bi| = √2
theorem find_magnitude (h_conj: conjugate_condition) : abs (a + b * i) = real.sqrt 2 :=
sorry

end find_magnitude_l610_610976


namespace harriet_siblings_product_l610_610246

theorem harriet_siblings_product (Harry_sisters Harry_brothers : ℕ) 
(h_harry_sisters : Harry_sisters = 4) 
(h_harry_brothers : Harry_brothers = 3) :
  let S := Harry_sisters - 1 -- number of sisters Harriet has
  let B := Harry_brothers -- number of brothers Harriet has
  S * B = 9 :=
by
  have S_def : S = 3 := by
    rw [←h_harry_sisters]
    exact Nat.sub_self 1
  have B_def : B = 3 := by rw [←h_harry_brothers]
  rw [S_def, B_def]
  norm_num

end harriet_siblings_product_l610_610246


namespace carlos_fraction_l610_610551

theorem carlos_fraction (f : ℝ) :
  (1 - f) ^ 4 * 64 = 4 → f = 1 / 2 :=
by
  intro h
  sorry

end carlos_fraction_l610_610551


namespace regular_polygon_sides_l610_610498

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 / n = 160) : n = 18 :=
by
  sorry

end regular_polygon_sides_l610_610498


namespace find_f_a_plus_b_plus_c_l610_610633

open Polynomial

variables {a b c p q r : ℝ}
variables (f : ℝ → ℝ)

-- Polynomial conditions
def f_poly (x : ℝ) := p * x^2 + q * x + r

-- Given distinct real numbers a, b, c
variables (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
-- Given polynomial conditions
variables (h1 : f a = b * c)
variables (h2 : f b = c * a)
variables (h3 : f c = a * b)

-- The goal statement
theorem find_f_a_plus_b_plus_c :
  f (a + b + c) = a * b + b * c + c * a := sorry

end find_f_a_plus_b_plus_c_l610_610633


namespace trig_expression_constant_l610_610780

theorem trig_expression_constant (α : ℝ) :
  sin (250 * (π / 180) + α) * cos (200 * (π / 180) - α) -
  cos (240 * (π / 180)) * cos (220 * (π / 180) - 2 * α) = 1 / 2 := 
sorry

end trig_expression_constant_l610_610780


namespace sum_of_x_when_fx_zero_l610_610333

def f (x : ℝ) : ℝ :=
if x < 3 then 6 * x + 21 else 3 * x - 9

theorem sum_of_x_when_fx_zero : (∑' (x : ℝ) in ({x : ℝ | f x = 0}), x) = -0.5 :=
by
  sorry

end sum_of_x_when_fx_zero_l610_610333


namespace intervals_of_monotonicity_range_of_m_extremal_points_sum_bounds_l610_610655

noncomputable def f (x : ℝ) (m n k : ℝ) : ℝ := (Real.exp x) / (m * x^2 + n * x + k)

theorem intervals_of_monotonicity (f : ℝ → ℝ) (h : ∀ x, f x = (Real.exp x) / (x^2 + x + 1)) : 
  (∀ x, ((0 < x) ∧ (x < 1) → f' x < 0) ∧ ((x < 0) ∨ (x > 1) → f' x > 0)) → 
  (f is decreasing on (0, 1)) ∧ (f is increasing on (-∞, 0) ∪ (1,+ ∞)) := sorry

theorem range_of_m (m : ℝ) (h : ∀ x, x ≥ 0 → (Real.exp x) / (m * x^2 + x + 1) ≥ 1) :
  0 ≤ m ∧ m ≤ 1/2 := sorry

theorem extremal_points_sum_bounds (m : ℝ) (h : ∀ x, f x = (Real.exp x) / (m * x^2 + 1)) :
  (m > 0) ∧ (f has two extremal points) → 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (Real.exp x1 / (2 * m * x1)) + (Real.exp x2 / (2 * m * x2)) > (e * sqrt m) / m 
  ∧ (Real.exp x1 / (2 * m * x1)) + (Real.exp x2 / (2 * m * x2)) < (e^2 + 1) / 2) := sorry

end intervals_of_monotonicity_range_of_m_extremal_points_sum_bounds_l610_610655


namespace angle_BPC_measure_l610_610702

-- Let θ be a real number representing the base angles in triangle ABC
variables (θ : ℝ)

-- Let A, B, C be points representing the vertices of the triangle ABC
-- and P be a point on side BC such that BP = 2PC
variables (A B C P : Point)

-- Define the structures and the conditions given in the problem
def isosceles_triangle_ABC (A B C : Point) : Prop :=
  dist A B = dist A C ∧ ∠BAC = θ ∧ ∠BCA = θ

def point_P_on_BC (B C P : Point) : Prop :=
  ∃ k : ℝ, 0 < k ∧ k * dist P C = dist B P ∧ k = 2

-- The main theorem we want to prove
theorem angle_BPC_measure (A B C P : Point) (θ : ℝ) (h1 : isosceles_triangle_ABC A B C θ)
  (h2 : point_P_on_BC B C P) : ∠BPC = 2 * (180 - θ) / 3 :=
  sorry

end angle_BPC_measure_l610_610702


namespace gilbert_herb_plants_l610_610192

theorem gilbert_herb_plants
  (initial_basil : ℕ)
  (initial_parsley : ℕ)
  (initial_mint : ℕ)
  (extra_basil_mid_spring : ℕ)
  (mint_eaten_by_rabbit : ℕ) :
  initial_basil = 3 →
  initial_parsley = 1 →
  initial_mint = 2 →
  extra_basil_mid_spring = 1 →
  mint_eaten_by_rabbit = 2 →
  initial_basil + initial_parsley + initial_mint + extra_basil_mid_spring - mint_eaten_by_rabbit = 5 :=
by
  intros h_basil h_parsley h_mint h_extra h_eaten
  simp [h_basil, h_parsley, h_mint, h_extra, h_eaten]
  done
  sorry

end gilbert_herb_plants_l610_610192


namespace remainder_when_xyz_divided_by_9_is_0_l610_610673

theorem remainder_when_xyz_divided_by_9_is_0
  (x y z : ℕ)
  (hx : x < 9)
  (hy : y < 9)
  (hz : z < 9)
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 9])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 9])
  (h3 : 2 * x + y + 3 * z ≡ 5 [MOD 9]) :
  (x * y * z) % 9 = 0 := by
  sorry

end remainder_when_xyz_divided_by_9_is_0_l610_610673


namespace mixture_concentration_l610_610901

def concentration_of_mixture (v1 : ℝ) (c1 : ℝ) (v2 : ℝ) (c2 : ℝ) (total_vol : ℝ) : ℝ :=
(v1 * c1 + v2 * c2) / total_vol

theorem mixture_concentration :
  concentration_of_mixture 3 0.25 5 0.40 10 * 100 = 27.5 := by
sory

end mixture_concentration_l610_610901


namespace intersection_complement_l610_610761

open Set

variable (U : Set ℕ) (P Q : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7})
variable (hP : P = {1, 2, 3, 4, 5})
variable (hQ : Q = {3, 4, 5, 6, 7})

theorem intersection_complement :
  P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end intersection_complement_l610_610761


namespace not_possible_20_odd_rows_15_odd_columns_possible_126_crosses_in_16x16_l610_610501

-- Define the general properties and initial conditions for the problems.
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Question a: Can there be exactly 20 odd rows and 15 odd columns in the table?
theorem not_possible_20_odd_rows_15_odd_columns (n : ℕ) (n_odd_rows : ℕ) (n_odd_columns : ℕ) (crosses : ℕ) (h_crosses_odd : is_odd crosses) 
  (h_odd_rows : n_odd_rows = 20) (h_odd_columns : n_odd_columns = 15) : 
  false := 
sorry

-- Question b: Can 126 crosses be arranged in a \(16 \times 16\) table so that all rows and columns are odd?
theorem possible_126_crosses_in_16x16 (crosses : ℕ) (n : ℕ) (h_crosses : crosses = 126) (h_table_size : n = 16) : 
  ∃ (table : matrix ℕ ℕ bool), 
    (∀ i : fin n, is_odd (count (λ j, table i j) (list.range n))) ∧
    (∀ j : fin n, is_odd (count (λ i, table i j) (list.range n))) :=
sorry

end not_possible_20_odd_rows_15_odd_columns_possible_126_crosses_in_16x16_l610_610501


namespace trig_inequality_l610_610778

theorem trig_inequality (x : ℝ) : 
  (sin x + 2 * cos (2 * x)) * (2 * sin (2 * x) - cos x) < 4.5 :=
by 
  sorry

end trig_inequality_l610_610778


namespace statement_C_incorrect_l610_610412

theorem statement_C_incorrect : 
  ∀ (consumption charge : ℝ), 
  (∀ (n : ℕ), charge = 0.55 * consumption) → (charge = 2.75 → consumption ≠ 6) :=
by
  intros consumption charge h_linear h_charge
  have h_contradiction : 6 * 0.55 ≠ 2.75 := by
    norm_num
  exact h_contradiction

end statement_C_incorrect_l610_610412


namespace horizontal_distance_between_points_l610_610090

def parabola (x : ℝ) : ℝ := x^2 - 2 * x - 8

theorem horizontal_distance_between_points :
  ∃ P Q : ℝ, parabola P = 8 ∧ parabola Q = -4 ∧ |P - Q| = |sqrt 17 - sqrt 5| :=
by
  sorry

end horizontal_distance_between_points_l610_610090


namespace inequality_abc_l610_610367

variable {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem inequality_abc :
  (a / (sqrt (7 * a^2 + b^2 + c^2)) + 
   b / (sqrt (a^2 + 7 * b^2 + c^2)) + 
   c / (sqrt (a^2 + b^2 + 7 * c^2))) ≤ 1 := 
sorry

end inequality_abc_l610_610367


namespace at_least_2001_pairs_l610_610829

theorem at_least_2001_pairs (points : Set ℝ) (h : ∀ p ∈ points, abs p ≤ 1) (hcard : points.card = 212) : 
  ∃ (pairs : Set (ℝ × ℝ)), pairs.card ≥ 2001 ∧ ∀ ⟨x, y⟩ ∈ pairs, dist x y ≤ 1 :=
sorry

end at_least_2001_pairs_l610_610829


namespace system_solutions_l610_610962

theorem system_solutions (a b c : ℝ) (x : ℕ → ℝ) (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → a ≠ (0 : ℝ) ∧ 
  a * (x (i % n + 1)) ^ 2 + b * x (i % n + 1) + c = x ((i + 1) % n + 1)) :
  ((b - 1)^2 - 4 * a * c < 0 → ¬∃ x, ∀ i, 1 ≤ i ∧ i ≤ n → a * (x (i % n + 1)) ^ 2 + b * x (i % n + 1) + c = x ((i + 1) % n + 1)) ∧ 
  ((b - 1)^2 - 4 * a * c = 0 → ∃! x, ∀ i, 1 ≤ i ∧ i ≤ n → a * (x (i % n + 1)) ^ 2 + b * x (i % n + 1) + c = x ((i + 1) % n + 1)) ∧ 
  ((b - 1)^2 - 4 * a * c > 0 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ ∀ i, 1 ≤ i ∧ i ≤ n → 
    (a * (x₁ (i % n + 1)) ^ 2 + b * x₁ (i % n + 1) + c = x₁ ((i + 1) % n + 1)) ∧ 
    (a * (x₂ (i % n + 1)) ^ 2 + b * x₂ (i % n + 1) + c = x₂ ((i + 1) % n + 1))) := by
  sorry

end system_solutions_l610_610962


namespace find_incorrect_statements_l610_610042

-- Define vector space and orthogonality
variables {V : Type*} [InnerProductSpace ℝ V]

-- Define the parallel relationship
def parallel (a b : V) : Prop := ∃ k : ℝ, a = k • b

-- Statement A
def statement_a (a b c : V) : Prop :=
  parallel a b ∧ parallel b c → parallel a c

-- Statement B
def statement_b (a b : V) : Prop :=
  ∥a∥ = ∥b∥ ∧ parallel a b → a = b

-- Statement C
def statement_c (a b : V) [Nonempty V] : Prop :=
  ∥a + b∥ = ∥a - b∥ → ⟪a, b⟫ = 0

-- Statement D
def statement_d (a b : V) : Prop :=
  parallel a b → ∃! k : ℝ, a = k • b

-- The final theorem to prove which of the statements are incorrect
theorem find_incorrect_statements (a b c : V) :
  ¬ statement_a a b c ∧ ¬ statement_b a b ∧ ¬ statement_d a b :=
begin
  -- Proof omitted
  sorry
end

end find_incorrect_statements_l610_610042


namespace greatest_four_digit_multiple_of_17_l610_610003

theorem greatest_four_digit_multiple_of_17 :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 17 = 0 ∧ ∀ m, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 17 = 0) → m ≤ n :=
  ⟨9996, by {
        split,
        { linarith },
        { split,
            { linarith },
            { split,
                { exact ModEq.rfl },
                { intros m hm hle,
                  have h : m ≤ 9999 := hm.2.2,
                  have : m = 17 * (m / 17) := (Nat.div_mul_cancel hm.2.1).symm,
                  have : 17 * (m / 17) ≤ 17 * 588 := Nat.mul_le_mul_left 17 (Nat.div_le_of_le_mul (by linarith)),
                  linarith,
                },
            },
        },
    },
  ⟩ sorry

end greatest_four_digit_multiple_of_17_l610_610003


namespace sum_of_valid_x_values_l610_610170

theorem sum_of_valid_x_values : 
  (∑ x in {120, 150, 180, 210, 240}, x) = 900 :=
begin
  sorry
end

end sum_of_valid_x_values_l610_610170


namespace trigonometric_identity_l610_610589

theorem trigonometric_identity :
  (2 * real.cos (real.pi / 18) - real.sin (real.pi / 9)) / real.sin (7 * real.pi / 18) = real.sqrt 3 :=
by
  -- Proof goes here
  sorry

end trigonometric_identity_l610_610589


namespace prove_fraction_l610_610646

variables {a : ℕ → ℝ} {b : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

def forms_arithmetic_sequence (x y z : ℝ) : Prop :=
2 * y = x + z

theorem prove_fraction
  (ha : is_arithmetic_sequence a)
  (hb : is_geometric_sequence b)
  (h_ar : forms_arithmetic_sequence (a 1 + 2 * b 1) (a 3 + 4 * b 3) (a 5 + 8 * b 5)) :
  (b 3 * b 7) / (b 4 ^ 2) = 1 / 4 :=
sorry

end prove_fraction_l610_610646


namespace general_formulas_max_b_seq_l610_610217

noncomputable def a_seq (n : ℕ) : ℕ := 4 * n - 2
noncomputable def b_seq (n : ℕ) : ℕ := 4 * n - 2 - 2^(n - 1)

-- The general formulas to be proved
theorem general_formulas :
  (∀ n : ℕ, a_seq n = 4 * n - 2) ∧ 
  (∀ n : ℕ, b_seq n = 4 * n - 2 - 2^(n - 1)) :=
by
  sorry

-- The maximum value condition to be proved
theorem max_b_seq :
  ((∀ n : ℕ, b_seq n ≤ b_seq 3) ∨ (∀ n : ℕ, b_seq n ≤ b_seq 4)) :=
by
  sorry

end general_formulas_max_b_seq_l610_610217


namespace type1_pieces_count_l610_610896

theorem type1_pieces_count (n : ℕ) (pieces : ℕ → ℕ)  (nonNegative : ∀ i, pieces i ≥ 0) :
  pieces 1 ≥ 4 * n - 1 :=
sorry

end type1_pieces_count_l610_610896


namespace songs_in_first_two_albums_l610_610125

/-
Beyonce releases 5 different singles on iTunes.
She releases 2 albums that each has some songs.
She releases 1 album that has 20 songs.
Beyonce has released 55 songs in total.
Prove that the total number of songs in the first two albums is 30.
-/

theorem songs_in_first_two_albums {A B : ℕ} 
  (h1 : 5 + A + B + 20 = 55) : 
  A + B = 30 :=
by
  sorry

end songs_in_first_two_albums_l610_610125


namespace gilbert_herb_plants_count_l610_610190

variable (initial_basil : Nat) (initial_parsley : Nat) (initial_mint : Nat)
variable (dropped_basil_seeds : Nat) (rabbit_ate_all_mint : Bool)

def total_initial_plants (initial_basil initial_parsley initial_mint : Nat) : Nat :=
  initial_basil + initial_parsley + initial_mint

def total_plants_after_dropping_seeds (initial_basil initial_parsley initial_mint dropped_basil_seeds : Nat) : Nat :=
  total_initial_plants initial_basil initial_parsley initial_mint + dropped_basil_seeds

def total_plants_after_rabbit (initial_basil initial_parsley initial_mint dropped_basil_seeds : Nat) (rabbit_ate_all_mint : Bool) : Nat :=
  if rabbit_ate_all_mint then 
    total_plants_after_dropping_seeds initial_basil initial_parsley initial_mint dropped_basil_seeds - initial_mint 
  else 
    total_plants_after_dropping_seeds initial_basil initial_parsley initial_mint dropped_basil_seeds

theorem gilbert_herb_plants_count
  (h1 : initial_basil = 3)
  (h2 : initial_parsley = 1)
  (h3 : initial_mint = 2)
  (h4 : dropped_basil_seeds = 1)
  (h5 : rabbit_ate_all_mint = true) :
  total_plants_after_rabbit initial_basil initial_parsley initial_mint dropped_basil_seeds rabbit_ate_all_mint = 5 := by
  sorry

end gilbert_herb_plants_count_l610_610190


namespace unit_vector_of_a_l610_610297

noncomputable def vector_projections
  (a : ℝ × ℝ)
  (hx : a.1 = -3)
  (hy : a.2 = 4) : ℝ × ℝ :=
let mag := real.sqrt ((-3)^2 + 4^2) in
  (a.1 / mag, a.2 / mag)

theorem unit_vector_of_a 
  (a : ℝ × ℝ)
  (hx : a.1 = -3)
  (hy : a.2 = 4) :
  vector_projections a hx hy = ( -3 / 5, 4 / 5 ) ∨ vector_projections a hx hy = ( 3 / 5, -4 / 5 ) :=
sorry

end unit_vector_of_a_l610_610297


namespace range_of_a_l610_610417

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x - 4 < 0 ) ↔ (-16 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l610_610417


namespace greatest_four_digit_multiple_of_17_l610_610016

theorem greatest_four_digit_multiple_of_17 : ∃ n, (1000 ≤ n) ∧ (n ≤ 9999) ∧ (17 ∣ n) ∧ ∀ m, (1000 ≤ m) ∧ (m ≤ 9999) ∧ (17 ∣ m) → m ≤ n :=
begin
  sorry
end

end greatest_four_digit_multiple_of_17_l610_610016


namespace trapezium_area_l610_610470

theorem trapezium_area (a b h : ℝ) (ha : a = 22) (hb : b = 18) (hh : h = 15) :
  (1 / 2) * (a + b) * h = 300 :=
by 
  rw [ha, hb, hh]
  norm_num
  sorry

end trapezium_area_l610_610470


namespace percentage_both_pets_owners_l610_610701

theorem percentage_both_pets_owners (total_students both_pets_owners : ℕ) 
  (h1 : total_students = 500) 
  (h2 : both_pets_owners = 50) : 
  (both_pets_owners * 100) / total_students = 10 := 
by
  rw [h1, h2]
  norm_num
  sorry

end percentage_both_pets_owners_l610_610701


namespace area_triangle_MNR_l610_610292

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

/-- Given the quadrilateral PQRS with the midpoints M and N of PQ and QR 
and specified lengths, prove the calculated area of triangle MNR. -/
theorem area_triangle_MNR : 
  let P : (ℝ × ℝ) := (0, 5)
  let Q : (ℝ × ℝ) := (10, 5)
  let R : (ℝ × ℝ) := (14, 0)
  let S : (ℝ × ℝ) := (7, 0)
  let M : (ℝ × ℝ) := (5, 5)  -- Midpoint of PQ
  let N : (ℝ × ℝ) := (12, 2.5) -- Midpoint of QR
  distance M.fst M.snd N.fst N.snd = 7.435 →
  ((5 - 0 : ℝ) / 2 = 2.5) →
  (1 / 2 * 7.435 * 2.5) = 9.294375 :=
by
  sorry

end area_triangle_MNR_l610_610292


namespace lcm_of_23_46_827_l610_610471

theorem lcm_of_23_46_827 : Nat.lcm (Nat.lcm 23 46) 827 = 38042 :=
by
  sorry

end lcm_of_23_46_827_l610_610471


namespace sum_of_betas_l610_610148

theorem sum_of_betas :
    let Q : ℂ[X] := (∑ i in Finset.range 13, X^i)^3 - X^12
    ∃ (β : Fin 36 → ℝ) (s : Fin 36 → ℝ), 
    (∀ k, 0 < β k ∧ β k < 1 ∧ 0 < s k) ∧
    Multiset.map (λ α, complex.abs α) (Q.roots * (map (\(z_k, k) => s_k • exp(2 * real.pi * I * β_k)) (Q.roots)).1).injective ∧
    (β 0 + β 1 + β 2 + β 3 + β 4 = 37 / 52)
by
  sorry

end sum_of_betas_l610_610148


namespace mark_total_trees_l610_610349

theorem mark_total_trees (initial_trees : ℕ) (trees_per_tree : ℕ) (additional_trees : ℕ) (total_trees : ℕ) :
  initial_trees = 93 → 
  trees_per_tree = 8 → 
  additional_trees = initial_trees * trees_per_tree →
  total_trees = initial_trees + additional_trees → 
  total_trees = 837 :=
by
  intros h_initial h_per_tree h_additional h_total
  rw [h_initial, h_per_tree] at h_additional
  change initial_trees * trees_per_tree = 744 at h_additional
  rw [h_initial] at h_total
  change initial_trees + additional_trees = 837 at h_total
  rw [h_additional] at h_total
  exact h_total

end mark_total_trees_l610_610349


namespace greatest_four_digit_multiple_of_17_l610_610012

theorem greatest_four_digit_multiple_of_17 :
  ∃ n, (n % 17 = 0) ∧ (1000 ≤ n ∧ n ≤ 9999) ∧ ∀ m, (m % 17 = 0) ∧ (1000 ≤ m ∧ m ≤ 9999) → n ≥ m :=
begin
  use 9996,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm2a hm2b,
    exact nat.mul_le_mul_right _ (nat.div_le_of_le_mul (nat.le_sub_one_of_lt (lt_of_le_of_lt (nat.mul_le_mul_right _ (nat.le_of_dvd hm1)) (by norm_num)))),
  },
  sorry
end

end greatest_four_digit_multiple_of_17_l610_610012


namespace find_f_of_expression_l610_610966

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3
noncomputable def g (x : ℝ) : ℝ := (x - 3) / 2

theorem find_f_of_expression (h : ∀ x, f (g (x)) = x ∧ g (f (x)) = x) : 
  f 7 = 17 :=
by 
  have h1 := h 7, 
  sorry

end find_f_of_expression_l610_610966


namespace complement_A_l610_610242

section Complements

variable (R : Set ℝ) (A : Set ℝ) (U : Set ℝ)

def A := {x : ℝ | -1 ≤ x ∧ x < 2}
def U := Set.univ

theorem complement_A :
  (U \ A) = {x : ℝ | x < -1 ∨ x ≥ 2} :=
by {
  sorry
}

end Complements

end complement_A_l610_610242


namespace compute_a3_binv2_l610_610746

-- Define variables and their values
def a : ℚ := 4 / 7
def b : ℚ := 5 / 6

-- State the main theorem that directly translates the problem to Lean
theorem compute_a3_binv2 : (a^3 * b^(-2)) = (2304 / 8575) :=
by
  -- proof left as an exercise for the user
  sorry

end compute_a3_binv2_l610_610746


namespace dots_not_visible_l610_610591

theorem dots_not_visible (total_dice : ℕ) (face_values : ℕ → ℕ) (visible_faces : List ℕ) 
  (num_faces : ℕ) (die_face_sum : ℕ) (total_visible_faces : ℕ) :
  total_dice = 5 →
  (∀ i, face_values i ∈ {1, 2, 3, 4, 5, 6}) →
  visible_faces = [1, 2, 2, 3, 3, 3, 4, 4, 5, 6] →
  num_faces = 6 →
  die_face_sum = 21 →
  total_visible_faces = 10 →
  (total_dice * die_face_sum) - visible_faces.sum = 72 := by
  intros h1 h2 h3 h4 h5 h6
  have h_sum : visible_faces.sum = 33, from List.sum_eq 33 h3.rfl
  rw [h5, h1]
  calc
    5 * 21
      = 105 : by ring
    ... - 33
      = 72 : by rw h_sum; ring

end dots_not_visible_l610_610591


namespace tamika_greater_than_carlos_l610_610798

theorem tamika_greater_than_carlos :
  let A := {10, 11, 12}
  let B := {4, 5, 7}
  let sum_pairs := {(a + b) | a in A, b in A, a ≠ b}
  let prod_pairs := {(c * d) | c in B, d in B, c ≠ d}
  let successful_pairs := {(s, p) | s in sum_pairs, p in prod_pairs, s > p}
  (successful_pairs.card : ℚ) / (sum_pairs.card * prod_pairs.card) = 1 / 3 := by
  sorry

end tamika_greater_than_carlos_l610_610798


namespace win_sector_area_l610_610067

-- Define the radius of the circle (spinner)
def radius : ℝ := 8

-- Define the probability of winning on one spin
def probability_winning : ℝ := 1 / 4

-- Define the area of the circle, calculated from the radius
def total_area : ℝ := Real.pi * radius^2

-- The area of the WIN sector to be proven
theorem win_sector_area : (probability_winning * total_area) = 16 * Real.pi := by
  sorry

end win_sector_area_l610_610067


namespace johnny_needs_total_planks_l610_610728

theorem johnny_needs_total_planks :
  let small_table_surface := 3
  let small_table_legs := 4
  let medium_table_surface := 5
  let medium_table_legs := 4
  let large_table_surface := 7
  let large_table_legs := 4
  let num_small_tables := 3
  let num_medium_tables := 2
  let num_large_tables := 1
  let total_planks :=
    (small_table_surface * num_small_tables) +
    (small_table_legs * num_small_tables) +
    (medium_table_surface * num_medium_tables) +
    (medium_table_legs * num_medium_tables) +
    (large_table_surface * num_large_tables) +
    (large_table_legs * num_large_tables)
  in
  total_planks = 50 := by
  sorry

end johnny_needs_total_planks_l610_610728


namespace dart_more_likely_l610_610105

noncomputable def dart_probabilities (p : ℕ → ℝ) (n : ℕ) : Prop :=
  (∑ i in Finset.range n, p i = 1) →
  (∑ i in Finset.range n, p i ^ 2 >
   ∑ i in Finset.range n, p i * p ((i + 1) % n))

-- Example theorem statement
theorem dart_more_likely (p : ℕ → ℝ) (n : ℕ) (h : ∑ i in Finset.range n, p i = 1) :
  (∑ i in Finset.range n, p i ^ 2 >
   ∑ i in Finset.range n, p i * p ((i + 1) % n)) :=
begin
  sorry -- Proof not required as per instructions
end

end dart_more_likely_l610_610105


namespace cosine_angle_BHD_l610_610296

theorem cosine_angle_BHD {HF BH DH : ℝ}
  (h_angle_DHG : ∠ DHG = 45)
  (h_angle_FHB : ∠ FHB = 60)
  (h_HF : HF = 1)
  (h_cos_FHB : HF / cos 60 = BH)
  (h_pythag_BHF : BH^2 = BF^2 + HF^2)
  (h_BH_value : BH = 2)
  (h_BF_value : BF = sqrt 3)
  (h_DH_value : DH = sqrt 6)
  (h_DB_value : DB = 2)
  :
  cos (∠ BHD) = sqrt 6 / 4 :=
sorry

end cosine_angle_BHD_l610_610296


namespace current_number_of_people_l610_610834

theorem current_number_of_people (a b : ℕ) : 0 ≤ a → 0 ≤ b → 48 - a + b ≥ 0 := by
  sorry

end current_number_of_people_l610_610834


namespace dart_more_likely_l610_610104

noncomputable def dart_probabilities (p : ℕ → ℝ) (n : ℕ) : Prop :=
  (∑ i in Finset.range n, p i = 1) →
  (∑ i in Finset.range n, p i ^ 2 >
   ∑ i in Finset.range n, p i * p ((i + 1) % n))

-- Example theorem statement
theorem dart_more_likely (p : ℕ → ℝ) (n : ℕ) (h : ∑ i in Finset.range n, p i = 1) :
  (∑ i in Finset.range n, p i ^ 2 >
   ∑ i in Finset.range n, p i * p ((i + 1) % n)) :=
begin
  sorry -- Proof not required as per instructions
end

end dart_more_likely_l610_610104


namespace infinite_a_exists_l610_610595

theorem infinite_a_exists (n : ℕ) : ∃ (a : ℕ+), ∃ (m : ℕ+), n^6 + 3 * (a : ℕ) = m^3 :=
  sorry

end infinite_a_exists_l610_610595


namespace count_positive_integers_l610_610567

theorem count_positive_integers (n : ℕ) : ∃ k : ℕ, k = 9 ∧  ∀ n, 1 ≤ n → n < 10 → 3 * n + 20 < 50 :=
by
  sorry

end count_positive_integers_l610_610567


namespace triangular_pyramid_volume_l610_610174

theorem triangular_pyramid_volume
  (b : ℝ) (h : ℝ) (H : ℝ)
  (b_pos : b = 4.5) (h_pos : h = 6) (H_pos : H = 8) :
  let base_area := (b * h) / 2
  let volume := (base_area * H) / 3
  volume = 36 := by
  sorry

end triangular_pyramid_volume_l610_610174


namespace distinct_numbers_diff_3_4_or_7_l610_610618

open Set

theorem distinct_numbers_diff_3_4_or_7:
  ∀ (S : Set ℕ), S.card = 700 → (∀ n ∈ S, n ≤ 2017) →
  ∃ (a b ∈ S), a ≠ b ∧ (a - b = 3 ∨ a - b = 4 ∨ a - b = 7 ∨ b - a = 3 ∨ b - a = 4 ∨ b - a = 7) :=
by
  intros S h_card h_le
  sorry

end distinct_numbers_diff_3_4_or_7_l610_610618


namespace no_valid_prime_l610_610373

open Nat

def base_p_polynomial (p : ℕ) (coeffs : List ℕ) : ℕ → ℕ :=
  fun (n : ℕ) => coeffs.foldl (λ sum coef => sum * p + coef) 0

def num_1013 (p : ℕ) := base_p_polynomial p [1, 0, 1, 3]
def num_207 (p : ℕ) := base_p_polynomial p [2, 0, 7]
def num_214 (p : ℕ) := base_p_polynomial p [2, 1, 4]
def num_100 (p : ℕ) := base_p_polynomial p [1, 0, 0]
def num_10 (p : ℕ) := base_p_polynomial p [1, 0]

def num_321 (p : ℕ) := base_p_polynomial p [3, 2, 1]
def num_403 (p : ℕ) := base_p_polynomial p [4, 0, 3]
def num_210 (p : ℕ) := base_p_polynomial p [2, 1, 0]

theorem no_valid_prime (p : ℕ) [Fact (Nat.Prime p)] :
  num_1013 p + num_207 p + num_214 p + num_100 p + num_10 p ≠
  num_321 p + num_403 p + num_210 p := by
  sorry

end no_valid_prime_l610_610373


namespace negation_equiv_l610_610405

variables (Student : Type) (student : Student → Prop) (shares_truth : Student → Prop)

theorem negation_equiv :
  ¬ (∀ x, student x → shares_truth x) ↔ ∃ x, student x ∧ ¬ shares_truth x :=
begin
  sorry
end

end negation_equiv_l610_610405


namespace school_dance_boys_count_l610_610423

theorem school_dance_boys_count :
  let total_attendees := 100
  let faculty_and_staff := total_attendees * 10 / 100
  let students := total_attendees - faculty_and_staff
  let girls := 2 * students / 3
  let boys := students - girls
  boys = 30 := by
  sorry

end school_dance_boys_count_l610_610423


namespace max_value_condition_l610_610620

noncomputable def f (a x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ a then
  Real.log x
else
  if x > a then
    a / x
  else
    0 -- This case should not happen given the domain conditions

theorem max_value_condition (a : ℝ) : 
  (∃ M, ∀ x > 0, x ≤ a → f a x ≤ M) ∧ (∀ x > a, f a x ≤ M) ↔ a ≥ Real.exp 1 :=
sorry

end max_value_condition_l610_610620


namespace committee_lock_key_count_l610_610533

-- Define a committee member as a type
inductive Member : Type
| A | B | C | D | E

def num_locks (members : List Member) : Nat :=
  members.combination 2 |>.length

def keys_per_member (members : List Member) (total_locks : Nat) : Nat :=
  total_locks - members.length + 1

theorem committee_lock_key_count : 
  let members := [Member.A, Member.B, Member.C, Member.D, Member.E]
  let total_locks := num_locks members
  let keys_per := keys_per_member members total_locks
  total_locks = 10 ∧ keys_per = 6 :=
by 
  let members := [Member.A, Member.B, Member.C, Member.D, Member.E]
  let total_locks := num_locks members
  let keys_per := keys_per_member members total_locks
  have h1 : total_locks = 10, by
    sorry
  have h2 : keys_per = 6, by
    sorry
  exact ⟨h1, h2⟩

end committee_lock_key_count_l610_610533


namespace loss_percentage_l610_610100

theorem loss_percentage (CP SP SP_new : ℝ) (L : ℝ) 
  (h1 : CP = 1428.57)
  (h2 : SP = CP - (L / 100 * CP))
  (h3 : SP_new = CP + 0.04 * CP)
  (h4 : SP_new = SP + 200) :
  L = 10 := by
    sorry

end loss_percentage_l610_610100


namespace count_unique_products_l610_610248

open Set Finset

def A : Finset ℕ := {1, 3, 4, 7, 13}

def valid_combinations (s : Finset ℕ) : Finset (Finset ℕ) :=
  (powerset s).filter (λ t => 2 ≤ card t ∧ ¬(3 ∈ t ∧ 13 ∈ t))

def products (s : Finset ℕ) : Finset ℕ :=
  (valid_combinations s).image (λ t => t.fold (*) 1)

theorem count_unique_products : (products A).card = 7 := by
  sorry

end count_unique_products_l610_610248


namespace largest_number_by_removing_6_digits_l610_610460

def sequence := 2357111317192329

def remove_digits (original : ℕ) (positions : list ℕ) : ℕ :=
  sorry -- This would be the implementation for removing digits based on positions.

theorem largest_number_by_removing_6_digits :
  remove_digits sequence [1, 2, 3, 8, 9, 10] = 7317192329 :=
sorry

end largest_number_by_removing_6_digits_l610_610460


namespace part_a_part_b_l610_610508

-- Definitions based on the problem conditions
def is_square_table (n : ℕ) (table : List (List Bool)) : Prop := 
  table.length = n ∧ ∀ row, row ∈ table → row.length = n

def is_odd_row (row : List Bool) : Prop := 
  row.count (λ x => x = true) % 2 = 1

def is_odd_column (n : ℕ) (table : List (List Bool)) (c : ℕ) : Prop :=
  n > 0 ∧ c < n ∧ (List.count (λ row => row.get! c = true) table) % 2 = 1

-- Part (a) statement: Prove it's impossible to have exactly 20 odd rows and 15 odd columns
theorem part_a (n : ℕ) (table : List (List Bool)) :
  n = 16 → is_square_table n table → 
  (List.count is_odd_row table) = 20 → 
  ((List.range n).count (is_odd_column n table)) = 15 → 
  False := 
sorry

-- Part (b) statement: Prove that it's possible to arrange 126 crosses in a 16x16 table with all odd rows and columns
theorem part_b (table : List (List Bool)) :
  is_square_table 16 table →
  ((table.map (λ row => row.count (λ x => x = true))).sum = 126) →
  (∀ row, row ∈ table → is_odd_row row) →
  (∀ c, c < 16 → is_odd_column 16 table c) →
  True :=
sorry

end part_a_part_b_l610_610508


namespace find_x_squared_plus_one_div_x_squared_l610_610692

variable (x : ℝ)
hypothesis : x + (1 / x) = 2

theorem find_x_squared_plus_one_div_x_squared : x^2 + (1 / x^2) = 2 := by
  sorry

end find_x_squared_plus_one_div_x_squared_l610_610692


namespace question_geom_sequence_question_sum_expression_l610_610207

-- Sequence definition and conditions
def sequence (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
∀ n : ℕ, 0 < n → 3 * a n = 2 * S n + n

-- Asserts that {a_n + 1/2} is a geometric progression
def geometric (f : ℕ → ℚ) (r : ℚ) : Prop :=
∀ n : ℕ, 0 < n → f (n+1) = r * f n

-- Questions to prove:
theorem question_geom_sequence (a S : ℕ → ℚ) :
  sequence a S →
  geometric (λ n, a n + 1/2) (3/2) :=
sorry

theorem question_sum_expression (a S : ℕ → ℚ) (T : ℕ → ℚ) :
  sequence a S →
  (∀ n : ℕ, 0 < n →
    T n = ∑ i in finset.range n.succ, S i) →
  (∀ n : ℕ, 0 < n →
    T n = (3^(n + 2) - 9) / 8 - (n^2 + 4 * n) / 4) :=
sorry

end question_geom_sequence_question_sum_expression_l610_610207


namespace max_sailboat_speed_correct_l610_610391

noncomputable def max_sailboat_speed (C S ρ : ℝ) (v₀ : ℝ) : ℝ :=
  (v₀ / 3)

theorem max_sailboat_speed_correct :
  ∀ (C ρ : ℝ) (v₀ : ℝ), v₀ = 6 → S = 5 →
  max_sailboat_speed C S ρ v₀ = 2 :=
by
  intros C ρ v₀ hv₀ hS
  unfold max_sailboat_speed
  rw [hv₀, hS]
  norm_num

end max_sailboat_speed_correct_l610_610391


namespace twenty_odd_rows_fifteen_odd_cols_impossible_sixteen_by_sixteen_with_126_crosses_possible_l610_610522

-- Conditions and helper definitions
def is_odd (n: ℕ) := n % 2 = 1
def count_odd_rows (table: ℕ × ℕ → bool) (n: ℕ) := 
  (List.range n).countp (λ i, is_odd ((List.range n).countp (λ j, table (i, j))))
  
def count_odd_columns (table: ℕ × ℕ → bool) (n: ℕ) := 
  (List.range n).countp (λ j, is_odd ((List.range n).countp (λ i, table (i, j))))

-- a) Proof problem statement
theorem twenty_odd_rows_fifteen_odd_cols_impossible (n: ℕ): 
  ∀ (table: ℕ × ℕ → bool), count_odd_rows table n = 20 → count_odd_columns table n = 15 → False := 
begin
  intros table h_rows h_cols,
  sorry
end

-- b) Proof problem statement
theorem sixteen_by_sixteen_with_126_crosses_possible :
  ∃ (table: ℕ × ℕ → bool), count_odd_rows table 16 = 16 ∧ count_odd_columns table 16 = 16 ∧ 
  (List.range 16).sum (λ i, (List.range 16).countp (λ j, table (i, j))) = 126 :=
begin
  sorry
end

end twenty_odd_rows_fifteen_odd_cols_impossible_sixteen_by_sixteen_with_126_crosses_possible_l610_610522


namespace original_board_length_before_final_cut_l610_610437

-- Given conditions
def initial_length : ℕ := 143
def first_cut_length : ℕ := 25
def final_cut_length : ℕ := 7

def board_length_after_first_cut : ℕ := initial_length - first_cut_length
def board_length_after_final_cut : ℕ := board_length_after_first_cut - final_cut_length

-- The theorem to be proved
theorem original_board_length_before_final_cut : board_length_after_first_cut + final_cut_length = 125 :=
by
  sorry

end original_board_length_before_final_cut_l610_610437


namespace determine_b_l610_610407

-- Definitions based on the problem conditions
variables {p : ℝ} {a b c : ℝ}
def parabola_eqn := ∀ x : ℝ, a * x^2 + b * x + c

-- The vertex of the parabola is at (p, -p)
def vertex_condition := ∃ a b c : ℝ, ∀ x : ℝ, (parabola_eqn x = a * (x - p)^2 - p)

-- The y-intercept of the parabola is at (0, p)
def y_intercept_condition := parabola_eqn 0 = p

theorem determine_b (h1 : p ≠ 0) (h2 : vertex_condition) (h3 : y_intercept_condition) :
  b = -4 :=
sorry

end determine_b_l610_610407


namespace sequence_1978_reappears_sequence_1526_never_appears_l610_610298

def sequence_next_digit (seq : List ℕ) : ℕ :=
  (seq.takeRight 4).sum % 10

def digit_sequence (initial : List ℕ) (n : ℕ) : List ℕ :=
  if n < initial.length then initial.take n else
  let m := n - initial.length
  let next_digits := List.range m |>.map (λ i, sequence_next_digit (initial ++ next_digits.take i))
  initial ++ next_digits

theorem sequence_1978_reappears :
  ∀ n, digit_sequence [1, 9, 7, 8] n ∈ (digit_sequence [1, 9, 7, 8] (n + 30)) :=
sorry

theorem sequence_1526_never_appears :
  ∀ n, ¬([1, 5, 2, 6] ∈ (digit_sequence [1, 9, 7, 8] n)) :=
sorry

end sequence_1978_reappears_sequence_1526_never_appears_l610_610298


namespace more_likely_same_sector_l610_610107

theorem more_likely_same_sector 
  (p : ℕ → ℝ) 
  (n : ℕ) 
  (hprob_sum_one : ∑ i in Finset.range n, p i = 1) 
  (hprob_nonneg : ∀ i, 0 ≤ p i) : 
  ∑ i in Finset.range n, (p i) ^ 2 
  > ∑ i in Finset.range n, (p i) * (p ((i + 1) % n)) :=
by
  sorry

end more_likely_same_sector_l610_610107


namespace compute_exp_l610_610754

theorem compute_exp (a b : ℚ) (ha : a = 4 / 7) (hb : b = 5 / 6) : a^3 * b^(-2) = 2304 / 8575 := by
  sorry

end compute_exp_l610_610754


namespace sum_of_z_and_conj_z_l610_610649

-- Define the given complex number.
def z : ℂ := 1 / (1 - complex.I)

-- Define the conjugate of z.
def conj_z : ℂ := complex.conj z

-- The theorem to prove that z + conj_z = 1.
theorem sum_of_z_and_conj_z : z + conj_z = 1 :=
by
  -- The proof is omitted.
  sorry

end sum_of_z_and_conj_z_l610_610649


namespace monotonic_decrease_interval_l610_610993

def f (x : ℝ) := x^2 - 2 * x

theorem monotonic_decrease_interval :
  ∀ x ∈ set.Icc (0 : ℝ) 4, deriv f x < 0 := 
by
  sorry

end monotonic_decrease_interval_l610_610993


namespace car_rental_daily_rate_l610_610488

theorem car_rental_daily_rate (x : ℝ) : 
  (x + 0.18 * 48 = 18.95 + 0.16 * 48) -> 
  x = 17.99 :=
by 
  sorry

end car_rental_daily_rate_l610_610488


namespace solve_real_equation_l610_610580

theorem solve_real_equation (x : ℝ) (h : x^4 + (3 - x)^4 = 82) : x = 2.5 ∨ x = 0.5 :=
sorry

end solve_real_equation_l610_610580


namespace bunyakovsky_hit_same_sector_l610_610114

variable {n : ℕ} (p : Fin n → ℝ)

theorem bunyakovsky_hit_same_sector (h_sum : ∑ i in Finset.univ, p i = 1) :
  (∑ i in Finset.univ, (p i)^2) >
  (∑ i in Finset.univ, (p i) * (p (Fin.rotate 1 i))) := 
sorry

end bunyakovsky_hit_same_sector_l610_610114


namespace player1_wins_a_l610_610469

def game_winner_a (coins2 coins1 : ℕ) : string :=
  if coins2 = 7 ∧ coins1 = 7 ∧ coins2 * 2 + coins1 = 21 then
    "Player 1"
  else
    "Invalid game state"

theorem player1_wins_a : game_winner_a 7 7 = "Player 1" :=
by
  -- We would provide the full proof here.
  sorry

end player1_wins_a_l610_610469


namespace right_angled_triangle_l610_610416
  
theorem right_angled_triangle (x : ℝ) (hx : 0 < x) :
  let a := 5 * x
  let b := 12 * x
  let c := 13 * x
  a^2 + b^2 = c^2 :=
by
  let a := 5 * x
  let b := 12 * x
  let c := 13 * x
  sorry

end right_angled_triangle_l610_610416


namespace probability_area_condition_l610_610717

open Set Real

noncomputable def prob_area_greater_16_over_9 : Prop :=
  ∀ (a : ℝ), (1 < a ∧ a < 2) → (∃ (pr : ℝ), pr = 2/3 ∧ (measure {a | a^2 > 16/9}/2 - 1) = pr)

theorem probability_area_condition : prob_area_greater_16_over_9 := sorry

end probability_area_condition_l610_610717


namespace upstream_swim_distance_l610_610101

-- Definition of the speeds and distances
def downstream_speed (v : ℝ) := 5 + v
def upstream_speed (v : ℝ) := 5 - v
def distance := 54
def time := 6
def woman_speed_in_still_water := 5

-- Given condition: downstream_speed * time = distance
def downstream_condition (v : ℝ) := downstream_speed v * time = distance

-- Given condition: upstream distance is 'd' km
def upstream_distance (v : ℝ) := upstream_speed v * time

-- Prove that given the above conditions and solving the necessary equations, 
-- the distance swam upstream is 6 km.
theorem upstream_swim_distance {d : ℝ} (v : ℝ) (h1 : downstream_condition v) : upstream_distance v = 6 :=
by
  sorry

end upstream_swim_distance_l610_610101


namespace length_of_second_train_is_correct_l610_610044

-- Conditions
def length_of_first_train : ℝ := 240
def speed_of_first_train : ℝ := 120 * 1000 / 3600 -- Converting kmph to m/s
def speed_of_second_train : ℝ := 80 * 1000 / 3600 -- Converting kmph to m/s
def time_to_cross : ℝ := 9

-- The question: What is the length of the second train?
def length_of_second_train : ℝ :=
  let relative_speed := speed_of_first_train + speed_of_second_train in
  let total_distance := relative_speed * time_to_cross in
  total_distance - length_of_first_train

-- The correct answer
def correct_length_of_second_train : ℝ := 259.95

-- Proof statement (without proof implementation)
theorem length_of_second_train_is_correct :
  length_of_second_train = correct_length_of_second_train := sorry

end length_of_second_train_is_correct_l610_610044


namespace geometric_sequence_T9_l610_610958

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = a n * r

noncomputable def T_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  List.prod (List.map a (List.range n))

theorem geometric_sequence_T9 (a : ℕ → ℝ)
  (h1 : is_geometric_sequence a)
  (h2 : log 2 (a 3) + log 2 (a 7) = 2) :
  T_n a 9 = 512 :=
sorry

end geometric_sequence_T9_l610_610958


namespace area_of_WIN_sector_correct_l610_610070

-- Define variables and constants
def radius : ℝ := 8
def probability_of_winning : ℝ := 1 / 4

-- Define the area of the circle
def area_of_circle (r : ℝ) := real.pi * r ^ 2

-- Define the area of the WIN sector given the area of the circle and the probability of winning
def area_of_WIN_sector (area_circle : ℝ) (prob_win : ℝ) := prob_win * area_circle

-- Theorem that the area of the WIN sector is 16π square centimeters
theorem area_of_WIN_sector_correct :
  area_of_WIN_sector (area_of_circle radius) probability_of_winning = 16 * real.pi :=
by 
-- Proof omitted
sorry

end area_of_WIN_sector_correct_l610_610070


namespace christopher_sword_length_l610_610552

variable (C J U : ℤ)

def jameson_sword (C : ℤ) : ℤ := 2 * C + 3
def june_sword (J : ℤ) : ℤ := J + 5
def june_sword_christopher (C : ℤ) : ℤ := C + 23

theorem christopher_sword_length (h1 : J = jameson_sword C)
                                (h2 : U = june_sword J)
                                (h3 : U = june_sword_christopher C) :
                                C = 15 :=
by
  sorry

end christopher_sword_length_l610_610552


namespace correct_propositions_l610_610610

-- Define the necessary structures and axioms
structure Plane :=
(id : ℕ)

structure Line :=
(id : ℕ)

def perpendicular (m : Line) (α : Plane) : Prop := sorry
def parallel (m : Line) (α : Plane) : Prop := sorry
def parallel_lines (m : Line) (n : Line) : Prop := sorry
def intersection (α β : Plane) : Line := sorry
def angle_between (m : Line) (α : Plane) : ℝ := sorry

-- Define the propositions
def proposition1 (m n : Line) (π : Plane) : Prop :=
  (parallel m π ∧ parallel n π) → parallel_lines m n

def proposition2 (m n : Line) (α : Plane) : Prop := 
  (perpendicular m α ∧ parallel n α) → perpendicular m n

def proposition3 (α β : Plane) : Prop := 
  (¬ parallel α β) → ¬ ∃ l : Line, parallel l β ∧ parallel l α

def proposition4 (m n : Line) (α β : Plane) : Prop := 
  (parallel_lines n (intersection α β) ∧ parallel_lines m n) → 
  (parallel m α ∧ parallel m β)

def proposition5 (m n : Line) (α β : Plane) : Prop := 
  (parallel_lines m n ∧ parallel α β) → 
  (angle_between m α = angle_between n β)

-- Define the truth values of the propositions
def truth_values : list Prop :=
  [¬ proposition1, proposition2, ¬ proposition3, ¬ proposition4, proposition5]

-- The final theorem to prove the given propositions ②, ⑤ are correct
theorem correct_propositions (α β : Plane) (m n : Line) : 
  (truth_values = [¬ proposition1 m n α, proposition2 m n α, ¬ proposition3 α β, ¬ proposition4 m n α β, proposition5 m n α β]) :=
sorry

end correct_propositions_l610_610610


namespace cylinder_volume_l610_610890

theorem cylinder_volume (α β l : ℝ) : 
  ∀ (R H : ℝ), 
  H = l * Real.sin β ∧ R = (l * Real.cos β) / (2 * Real.cos α) →
  (π * R^2 * H = (π * l^3 * Real.cos β^2 * Real.sin β) / (4 * Real.cos α^2)) :=
begin
  intros R H h,
  cases h with Hl HR,
  sorry
end

end cylinder_volume_l610_610890


namespace polynomial_value_l610_610032

theorem polynomial_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end polynomial_value_l610_610032


namespace train_passes_jogger_in_approximately_25_8_seconds_l610_610080

noncomputable def jogger_speed_kmh := 7
noncomputable def train_speed_kmh := 60
noncomputable def jogger_head_start_m := 180
noncomputable def train_length_m := 200

noncomputable def kmh_to_ms (speed_kmh : ℕ) : ℕ := speed_kmh * 1000 / 3600

noncomputable def jogger_speed_ms := kmh_to_ms jogger_speed_kmh
noncomputable def train_speed_ms := kmh_to_ms train_speed_kmh

noncomputable def relative_speed_ms := train_speed_ms - jogger_speed_ms
noncomputable def total_distance_to_cover_m := jogger_head_start_m + train_length_m
noncomputable def time_to_pass_sec := total_distance_to_cover_m / (relative_speed_ms : ℝ) 

theorem train_passes_jogger_in_approximately_25_8_seconds :
  abs (time_to_pass_sec - 25.8) < 0.1 := sorry

end train_passes_jogger_in_approximately_25_8_seconds_l610_610080


namespace min_cost_to_fence_land_l610_610344

theorem min_cost_to_fence_land (w l : ℝ) (h1 : l = 2 * w) (h2 : 2 * w ^ 2 ≥ 500) : 
  5 * (2 * (l + w)) = 150 * Real.sqrt 10 := 
by
  sorry

end min_cost_to_fence_land_l610_610344


namespace abs_z_l610_610984

-- Define the complex number z
def z : ℂ := complex.I * (1 - complex.I)

-- State the theorem with the given condition and prove that |z| = √2
theorem abs_z : complex.abs z = real.sqrt 2 := 
by sorry

end abs_z_l610_610984


namespace divisibility_of_solutions_l610_610311

theorem divisibility_of_solutions (p : ℕ) (k : ℕ) (x₀ y₀ z₀ t₀ : ℕ) 
  (hp_prime : Nat.Prime p)
  (hp_form : p = 4 * k + 3)
  (h_eq : x₀^(2*p) + y₀^(2*p) + z₀^(2*p) = t₀^(2*p)) : 
  p ∣ x₀ ∨ p ∣ y₀ ∨ p ∣ z₀ ∨ p ∣ t₀ :=
sorry

end divisibility_of_solutions_l610_610311


namespace calories_burned_each_player_l610_610388

theorem calories_burned_each_player :
  ∀ (num_round_trips stairs_per_trip calories_per_stair : ℕ),
  num_round_trips = 40 →
  stairs_per_trip = 32 →
  calories_per_stair = 2 →
  (num_round_trips * (2 * stairs_per_trip) * calories_per_stair) = 5120 :=
by
  intros num_round_trips stairs_per_trip calories_per_stair h_num_round_trips h_stairs_per_trip h_calories_per_stair
  rw [h_num_round_trips, h_stairs_per_trip, h_calories_per_stair]
  simp
  rfl

#eval calories_burned_each_player 40 32 2 rfl rfl rfl

end calories_burned_each_player_l610_610388


namespace sum_of_a_and_b_l610_610891

-- Define the distance function between two points
def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

-- Define the vertices of the quadrilateral
def vertex1 := (0, 1)
def vertex2 := (3, 4)
def vertex3 := (4, 3)
def vertex4 := (3, 0)

-- Calculate distances between consecutive vertices
def d1 := distance 0 1 3 4
def d2 := distance 3 4 4 3
def d3 := distance 4 3 3 0
def d4 := distance 3 0 0 1

-- Define the perimeter
def perimeter := d1 + d2 + d3 + d4

-- Define a and b such that perimeter = a * Real.sqrt 2 + b * Real.sqrt 10
def a : ℤ := 4
def b : ℤ := 2

-- Define the sum of a and b
def a_plus_b : ℤ := a + b

-- State the theorem
theorem sum_of_a_and_b : a_plus_b = 6 := 
by 
  -- Proof is not required, so we use sorry
  sorry

end sum_of_a_and_b_l610_610891


namespace vendor_sells_50_percent_on_first_day_l610_610526

variables (A : ℝ) (S : ℝ)

theorem vendor_sells_50_percent_on_first_day 
  (h : 0.2 * A * (1 - S) + 0.5 * A * (1 - S) * 0.8 = 0.3 * A) : S = 0.5 :=
  sorry

end vendor_sells_50_percent_on_first_day_l610_610526


namespace angle_ACD_l610_610714

theorem angle_ACD (E : ℝ) (arc_eq : ∀ (AB BC CD : ℝ), AB = BC ∧ BC = CD) (angle_eq : E = 40) : ∃ (ACD : ℝ), ACD = 15 :=
by
  sorry

end angle_ACD_l610_610714


namespace dart_more_likely_l610_610102

noncomputable def dart_probabilities (p : ℕ → ℝ) (n : ℕ) : Prop :=
  (∑ i in Finset.range n, p i = 1) →
  (∑ i in Finset.range n, p i ^ 2 >
   ∑ i in Finset.range n, p i * p ((i + 1) % n))

-- Example theorem statement
theorem dart_more_likely (p : ℕ → ℝ) (n : ℕ) (h : ∑ i in Finset.range n, p i = 1) :
  (∑ i in Finset.range n, p i ^ 2 >
   ∑ i in Finset.range n, p i * p ((i + 1) % n)) :=
begin
  sorry -- Proof not required as per instructions
end

end dart_more_likely_l610_610102


namespace dike_position_and_purpose_l610_610402

-- Definitions of the conditions
def is_above_ground_river (r : String) : Prop := 
  r = "Yellow River"

def located_in_temperate_monsoon_climate_zone (r : String) : Prop :=
  r = "Yellow River"

def concentrated_heavy_rainfall (r : String) : Prop :=
  r = "Yellow River"

def flood_season_threat_significant (r : String) : Prop := 
  r = "Yellow River"

def consequences_of_breach_disastrous (r : String) : Prop := 
  r = "Yellow River"

-- Main theorem stating the problem
theorem dike_position_and_purpose (r : String) 
  (h1 : is_above_ground_river r)
  (h2 : located_in_temperate_monsoon_climate_zone r)
  (h3 : concentrated_heavy_rainfall r)
  (h4 : flood_season_threat_significant r)
  (h5 : consequences_of_breach_disastrous r) :
  (dike_position r = "concave bank") ∧ (dike_purpose r = "alleviating the impact of water flow") := 
sorry

end dike_position_and_purpose_l610_610402


namespace concurrency_of_lines_l610_610652

abbreviation Point := ℝ × ℝ
noncomputable def ellipse (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

def A : Point := (-2, 0)
def B : Point := (0, -1)
def l₁ (x : ℝ) := x = -2
def l₂ (y : ℝ) := y = -1
def P (x₀ y₀ : ℝ) : Prop := ellipse x₀ y₀ ∧ x₀ > 0 ∧ y₀ > 0

-- Theorem statement
theorem concurrency_of_lines : ∀ {x₀ y₀ : ℝ},
  P x₀ y₀ →
  (let C : Point := (-2, -1) in
   let D : Point := (4 * (y₀ + 1) / x₀, -1) in
   let E : Point := (-2, (x₀ + 2) / (2 * y₀)) in
   ∃ M : Point, M ∈ line through A D ∧ M ∈ line through B E ∧ M ∈ line through C ⟨x₀, y₀⟩)
:=
begin
  -- Placeholder for the proof
  sorry
end

end concurrency_of_lines_l610_610652


namespace additional_people_needed_l610_610253

-- Define the conditions
def num_people_initial := 9
def work_done_initial := 3 / 5
def days_initial := 14
def days_remaining := 4

-- Calculated values based on conditions
def work_rate_per_person : ℚ :=
  work_done_initial / (num_people_initial * days_initial)

def work_remaining : ℚ := 1 - work_done_initial

def total_people_needed : ℚ :=
  work_remaining / (work_rate_per_person * days_remaining)

-- Formulate the statement to prove
theorem additional_people_needed :
  total_people_needed - num_people_initial = 12 :=
by
  sorry

end additional_people_needed_l610_610253


namespace circuit_malfunction_probability_l610_610838

noncomputable def failure_rate_A : ℝ := 0.2
noncomputable def failure_rate_B : ℝ := 0.5

def prob_component_not_failed (rate: ℝ) : ℝ := 1 - rate

theorem circuit_malfunction_probability : 
  prob_component_not_failed failure_rate_A * prob_component_not_failed failure_rate_B = 0.4 → 
  1 - (prob_component_not_failed failure_rate_A * prob_component_not_failed failure_rate_B) = 0.6 :=
by sorry

#eval circuit_malfunction_probability

end circuit_malfunction_probability_l610_610838


namespace proof_problem_l610_610578

noncomputable def problem_statement : set ℝ :=
  {x : ℝ | (⌊x * ⌊x⌋⌋ = 17)}

noncomputable def solution_set : set ℝ :=
  {x : ℝ | 4.25 ≤ x ∧ x < 4.5}

theorem proof_problem : problem_statement = solution_set := by
  sorry

end proof_problem_l610_610578


namespace sum_of_squares_formula_l610_610781

theorem sum_of_squares_formula (n : ℕ) (h : n > 0) : 
  ∑ i in Finset.range (n + 1), i^2 = n * (n + 1) * (2 * n + 1) / 6 :=
by
  sorry

end sum_of_squares_formula_l610_610781


namespace true_product_of_two_digit_number_l610_610098

theorem true_product_of_two_digit_number (a b : ℕ) (h1 : b = 2 * a) (h2 : 136 * (10 * b + a) = 136 * (10 * a + b) + 1224) : 136 * (10 * a + b) = 1632 := 
by sorry

end true_product_of_two_digit_number_l610_610098


namespace S_100_is_2500_l610_610961

noncomputable def a_n : ℕ → ℝ := λ n, (1 / 4) + (n - 1) * (1 / 2)
noncomputable def S_n : ℕ → ℝ := λ n, (n * (2 * (1 / 4) + (n - 1) * (1 / 2))) / 2

theorem S_100_is_2500 
  (a_seq_arith : ∀ (n : ℕ), a_n n = (1 / 4) + (n - 1) * (1 / 2))
  (sqrt_sn_arith : ∀ (n : ℕ), sqrt (S_n n) = (sqrt (S_n 1)) + (n - 1) * (1 / 2))
  : S_n 100 = 2500 := 
by
  sorry

end S_100_is_2500_l610_610961


namespace shadow_length_change_l610_610443

theorem shadow_length_change (NorthernHemisphere : Prop) 
(bamboo_upright : Prop) 
(observation_duration : Prop) : 
  (NorthernHemisphere ∧ bamboo_upright ∧ observation_duration) → 
  "shadow length changes from long to short, and then becomes long again" :=
by sorry

end shadow_length_change_l610_610443


namespace range_of_a_l610_610436

open Real

noncomputable def doesNotPassThroughSecondQuadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (3 * a - 1) * x + (2 - a) * y - 1 ≠ 0

theorem range_of_a : {a : ℝ | doesNotPassThroughSecondQuadrant a} = {a : ℝ | 2 ≤ a } :=
by
  ext
  sorry

end range_of_a_l610_610436


namespace investment_time_Q_l610_610473

theorem investment_time_Q 
    (investment_ratio_pq : ℕ) 
    (profit_ratio_pq : ℕ) 
    (investment_time_p : ℕ)
    (common_multiple : ℕ) 
    (invest_p : ℕ := 7 * common_multiple) 
    (invest_q : ℕ := 5 * common_multiple)
    (profit_p : ℕ := 7) 
    (profit_q : ℕ := 10) 
    (inv_time_p : ℕ := 7)
    : investment_time_q = 14 :=
by 
  unfold investment_time_q
  sorry

end investment_time_Q_l610_610473


namespace part_I_part_II_part_III_l610_610181

def E (n : ℕ) : set ℕ := { k | k ≥ 1 ∧ k ≤ n }

def P (n : ℕ) : set ℝ := { x | ∃ (a b : ℕ) (ha : a ∈ E n) (hb : b ∈ E n), x = a / Real.sqrt b }

def has_property_Omega (A : set ℝ) : Prop := ∀ (x₁ x₂ : ℝ), x₁ ∈ A → x₂ ∈ A → x₁ ≠ x₂ → ∀ (k : ℕ), k > 0 → x₁ + x₂ ≠ (k : ℝ) ^ 2

theorem part_I : 
  |P 3| = 9 ∧
  |P 5| = 23 ∧
  ¬ has_property_Omega (P 3) := 
  sorry

theorem part_II :
  ¬ ∃ (A B : set ℝ), has_property_Omega A ∧ has_property_Omega B ∧ A ∩ B = ∅ ∧ A ∪ B = E 15 :=
  sorry

theorem part_III :
  ∃ (n : ℕ), ∀ (m : ℕ), m > n → ¬ ∃ (A B : set ℝ), has_property_Omega A ∧ has_property_Omega B ∧ A ∩ B = ∅ ∧ A ∪ B = P m ∧ n = 14 :=
  sorry

end part_I_part_II_part_III_l610_610181


namespace quadrilateral_CDFE_area_correct_l610_610783

/-
Rectangle ABCD has a length of 2 units and width of 1 unit.
Points E and F are taken respectively on sides AB and AD such that AE = 1/3 and AF = 2/3.
Prove that the maximum area of quadrilateral CDFE is 4/9.
-/

noncomputable def area_of_quadrilateral_CDFE (AB AD AE AF : ℝ) : ℝ :=
  let E := (AE, 0)
      F := (0, AF)
      G := (0, 1)
      H := (2, 1)
      area_triangle_EGF := (1 / 2) * (2 / 3) * (1 / 3)
      area_triangle_EGC := (1 / 2) * (2 / 3) * 1
  in area_triangle_EGF + area_triangle_EGC

theorem quadrilateral_CDFE_area_correct :
  ∀ (AB AD : ℝ), 
    ∀ (AE AF : ℝ),
      AE = 1 / 3 →
      AF = 2 / 3 →
      AB = 2 →
      AD = 1 →
      area_of_quadrilateral_CDFE AB AD AE AF = 4 / 9 :=
by
  intros AB AD AE AF hAE hAF hAB hAD
  rw [hAE, hAF, hAB, hAD]
  sorry

end quadrilateral_CDFE_area_correct_l610_610783


namespace area_difference_l610_610837

theorem area_difference
  (w₁ h₁ : ℕ) (w₂ h₂ : ℕ)
  (w₁_val : w₁ = 4) (h₁_val : h₁ = 5)
  (w₂_val : w₂ = 3) (h₂_val : h₂ = 6) :
  w₁ * h₁ - w₂ * h₂ = 2 :=
by
  rw [w₁_val, h₁_val, w₂_val, h₂_val]
  -- You would normally prove that 4 * 5 - 3 * 6 = 2 here
  sorry

end area_difference_l610_610837


namespace possible_values_of_a_l610_610261

theorem possible_values_of_a (a b x : ℝ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = x) :
  a = x * (1 + Real.sqrt(115 / 3)) / 2 ∨ a = x * (1 - Real.sqrt(115 / 3)) / 2 :=
sorry

end possible_values_of_a_l610_610261


namespace sarah_tom_probability_not_next_to_each_other_l610_610700

theorem sarah_tom_probability_not_next_to_each_other : 
  let total_ways : ℕ := 45 in
  let ways_next_to_each_other : ℕ := 9 in
  let probability_not_next_to_each_other : ℚ := (total_ways - ways_next_to_each_other) / total_ways in
  probability_not_next_to_each_other = 4 / 5 :=
by
  sorry

end sarah_tom_probability_not_next_to_each_other_l610_610700


namespace categorization_l610_610155

def numbers : List ℚ := [ -1/3, 22/7, -1, -7/10, 11, -25, 0, 85/100 ]

def positive_number_set : Set ℚ := {22/7, 11, 85/100}
def integer_set : Set ℤ := {-1, 11, -25, 0}
def non_negative_number_set : Set ℚ := {22/7, 11, 0, 85/100}
def negative_fraction_set : Set ℚ := {-1/3, -7/10}

theorem categorization (n : ℚ) :
  n ∈ positive_number_set ↔ n > 0 ∧ n ∈ numbers ∨
  (n ∈ negative_fraction_set ↔ n < 0 ∧ ¬ is_integer n ∧ n ∈ numbers) ∨
  (n ∈ integer_set ↔ is_integer n ∧ n ∈ numbers) ∨
  (n ∈ non_negative_number_set ↔ (n ≥ 0) ∧ n ∈ numbers) := sorry

def is_integer (n : ℚ) : Prop :=
∃ z : ℤ, n = z

end categorization_l610_610155


namespace volume_of_region_l610_610175

noncomputable def region : Set (ℝ × ℝ × ℝ) :=
  {p | let (x, y, z) := p in |x - y + z| + |x - y - z| + |x + y - z| + |-x + y - z| ≤ 6}

theorem volume_of_region : volume(region) = 36 := 
sorry

end volume_of_region_l610_610175


namespace collinear_points_a_b_coplanar_points_a_l610_610300

-- Define the points
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define vectors for collinearity condition
def collinear_vectors (a b : ℝ) : Prop :=
  let A : Point3D := ⟨2, a, -1⟩
  let B : Point3D := ⟨-2, 3, b⟩
  let C : Point3D := ⟨1, 2, -2⟩
  ∃ (λ : ℝ), (C.x - A.x) = λ * (C.x - B.x) ∧ (C.y - A.y) = λ * (C.y - B.y) ∧ (C.z - A.z) = λ * (C.z - B.z)

-- Define the main problem for collinearity
theorem collinear_points_a_b (a b : ℝ) : collinear_vectors a b → a = 5 / 3 ∧ b = -5 :=
by
  sorry

-- Define vectors for coplanarity condition
def coplanar_vectors (a b : ℝ) : Prop :=
  let A : Point3D := ⟨2, a, -1⟩
  let B : Point3D := ⟨-2, 3, b⟩
  let C : Point3D := ⟨1, 2, -2⟩
  let D : Point3D := ⟨-1, 3, -3⟩
  let AB : ℝ × ℝ × ℝ := (B.x - A.x, B.y - A.y, B.z - A.z)
  let AC : ℝ × ℝ × ℝ := (C.x - A.x, C.y - A.y, C.z - A.z)
  let AD : ℝ × ℝ × ℝ := (D.x - A.x, D.y - A.y, D.z - A.z)
  let normal : ℝ × ℝ × ℝ := (AC.2 * AD.3 - AC.3 * AD.2, AC.3 * AD.1 - AC.1 * AD.3, AC.1 * AD.2 - AC.2 * AD.1)
  let col_check := normal.1 * AB.1 + normal.2 * AB.2 + normal.3 * AB.3
  col_check = 0 
  
-- Define the main problem for coplanarity
theorem coplanar_points_a (a : ℝ) : coplanar_vectors a (-3) → a = 1 :=
by
  sorry

end collinear_points_a_b_coplanar_points_a_l610_610300


namespace limit_na_n_l610_610564

noncomputable def L (x : ℝ) : ℝ := x - (x^2) / 2

noncomputable def a_n (n : ℕ) : ℝ :=
  let rec iterate_L (k : ℕ) (x : ℝ) : ℝ :=
    if k = 0 then x else iterate_L (k - 1) (L x)
  iterate_L n (25 / n)

theorem limit_na_n : tendsto (fun (n: ℕ) => n * a_n n) at_top (nhds (50 / 27)) := sorry

end limit_na_n_l610_610564


namespace largest_lcm_l610_610025

def lcm_list : List ℕ := [
  Nat.lcm 15 3,
  Nat.lcm 15 5,
  Nat.lcm 15 9,
  Nat.lcm 15 10,
  Nat.lcm 15 12,
  Nat.lcm 15 15
]

theorem largest_lcm : List.maximum lcm_list = 60 := by
  sorry

end largest_lcm_l610_610025


namespace prove_a_eq_b_l610_610938

noncomputable def least_prime_divisor (m : ℕ) := Nat.min_fac m

theorem prove_a_eq_b (a b : ℕ) (h1 : a > 1) (h2 : b > 1)
    (h3 : a^2 + b = least_prime_divisor a + (least_prime_divisor b)^2) : a = b := by
  sorry

end prove_a_eq_b_l610_610938


namespace player2_wins_game_l610_610483

def PlayerWins (m n : ℕ) : Prop := ∃ (strategy : (Σ' P : ℕ, P < m * n → bool)), 
  ∀ (turns : list (ℕ × ℕ)), 
  strategy.fst = 2 ∧ PlayerLoses (strategy.snd turns.last)

-- Conditions derived from the problem
def game_conditions := ∀ (m n : ℕ), m = 100 ∧ n = 100 ∧
 (∀ turn, ∃ (i j : ℕ), glue_adjacents (i, j) ∧ (shape_connected_after_move(m,n,turns) → PlayerLoses))

-- Converting problem into Lean statement

theorem player2_wins_game : game_conditions 100 100 →
  PlayerWins 100 100 :=
sorry

end player2_wins_game_l610_610483


namespace problem1_problem2_l610_610596

-- Problem 1 Definition: Operation ※
def operation (m n : ℚ) : ℚ := 3 * m - n

-- Lean 4 statement: Prove 2※10 = -4
theorem problem1 : operation 2 10 = -4 := by
  sorry

-- Lean 4 statement: Prove that ※ does not satisfy the distributive law
theorem problem2 (a b c : ℚ) : 
  operation a (b + c) ≠ operation a b + operation a c := by
  sorry

end problem1_problem2_l610_610596


namespace num_solid_figures_is_four_l610_610529

/-- List of figures to consider: -/
def figures := ["circle", "square", "cone", "cuboid", "line segment", "sphere", "triangular prism", "right-angled triangle"]

/-- Predicate to identify whether a figure is solid -/
def isSolid (fig : String) : Prop :=
  fig = "cone" ∨ fig = "cuboid" ∨ fig = "sphere" ∨ fig = "triangular prism"

/-- Number of solid figures in the given list is 4 -/
theorem num_solid_figures_is_four : (figures.filter isSolid).length = 4 :=
by
  sorry

end num_solid_figures_is_four_l610_610529


namespace percent_less_than_l610_610277

variables (m d : ℝ)
variables (dist : ℝ → ℝ) -- Representation of the distribution function or density function

-- Assume the distribution is symmetric about mean m
def symmetric_about_mean (m : ℝ) (dist : ℝ → ℝ) : Prop :=
  ∀ x, dist (m + x) = dist (m - x)

-- Assume 68% of the distribution lies within one standard deviation d of the mean m
def within_one_std_dev (m d : ℝ) (dist : ℝ → ℝ) : Prop :=
  ∫ x in (m - d)..(m + d), dist x = 0.68

-- Prove that the percent of the distribution less than m + d is 84%
theorem percent_less_than (h_sym: symmetric_about_mean m dist) (h_within: within_one_std_dev m d dist) :
  ∫ x in -∞..(m + d), dist x = 0.84 :=
sorry

end percent_less_than_l610_610277


namespace magnitude_of_z_l610_610650

def z : ℂ := 3 + 4 * complex.I

theorem magnitude_of_z : complex.abs z = 5 := by
  sorry

end magnitude_of_z_l610_610650


namespace description_of_set_T_l610_610313

theorem description_of_set_T
  (a b c : ℝ)
  (T : set (ℝ × ℝ × ℝ))
  (h1 : b + 1 = 5 ∨ c - 3 = 5 ∨ b + 1 = c - 3)
  (h2 : {5, b + 1, c - 3}.min ≤ b + 1 ∧ {5, b + 1, c - 3}.min ≤ c - 3)
  : T = {p : ℝ × ℝ × ℝ | (p.2.1 = 4 ∧ p.2.2 ≤ 8) ∨ (p.2.2 = 8 ∧ p.2.1 ≤ 4) ∨ (p.2.2 = p.2.1 + 4 ∧ p.2.1 ≥ 4) ∧ (p.1 = a)} :=
by
  sorry

end description_of_set_T_l610_610313


namespace count_eight_letter_good_words_l610_610565

def is_good_word (w : String) : Prop :=
  (w.length = 8) ∧
  (∀ i, i < w.length - 1 → 
    (w.get i = 'A' → w.get (i + 1) ≠ 'B') ∧
    (w.get i = 'B' → w.get (i + 1) ≠ 'C') ∧
    (w.get i = 'C' → w.get (i + 1) ≠ 'A')) ∧
  (w.head ≠ 'A' ∨ w.get (w.length - 1) ≠ 'C')

theorem count_eight_letter_good_words : ∃ n : Nat, n = 160 ∧ (Finset.filter is_good_word (Finset.range (3 ^ 8))).card = n := 
by
  sorry

end count_eight_letter_good_words_l610_610565


namespace sum_of_x_coords_eq_3_l610_610642

-- Definitions of the conditions
variable (m : ℝ) (h1 : 1 < m) (h2 : m < 4)
def C := { P : ℝ × ℝ | P.1 ^ 2 / 4 + P.2 ^ 2 / (4 - m) = 1 }
def E := { P : ℝ × ℝ | P.1 ^ 2 - P.2 ^ 2 / (m - 1) = 1 }

-- Foci of the ellipse
def F1 := (- real.sqrt(4 - m), 0)
def F2 := (real.sqrt(4 - m), 0)

-- P and l
def P := {P : ℝ × ℝ | (P ∈ C) ∧ (P ∈ E) ∧ (0 < P.1) ∧ (0 < P.2)}
def l := {l : ℝ × ℝ -> Prop | ∃ x0 y0, l = λ P : ℝ × ℝ, x0 * P.1 / 4 + y0 * P.2 / (4 - m) = 1}

-- Incenter M and intersection point N
def M := {M : ℝ × ℝ | ∃ r y0, 2 * real.sqrt(m) * y0 = (4 + 2 * real.sqrt(m)) * r ∧ M = (1, r)}
def N := {N : ℝ × ℝ | ∃ xM l F1 F2 P M, N.1 = 2 ∧ P ∈ C ∧ P ∈ E ∧ M = (1, r) ∧ l ∩ F1M = N}

-- Prove the sum of x-coordinates of M and N is 3
theorem sum_of_x_coords_eq_3 : ∀ (m : ℝ) (h1 : 1 < m) (h2 : m < 4), 1 + 2 = 3 :=
by
  sorry

end sum_of_x_coords_eq_3_l610_610642


namespace max_value_of_trig_expr_l610_610165

theorem max_value_of_trig_expr :
  ∀ (θ1 θ2 θ3 θ4 θ5 : ℝ), 
  (cos θ1 * sin θ2 + cos θ2 * sin θ3 + cos θ3 * sin θ4 + cos θ4 * sin θ5 + cos θ5 * sin θ1) ≤ 5 / 2 :=
by
  intros θ1 θ2 θ3 θ4 θ5
  -- Proof to be filled in.
  sorry

end max_value_of_trig_expr_l610_610165


namespace div_condition_for_lcm_l610_610052

theorem div_condition_for_lcm (x y : ℕ) (hx : x > 1) (hy : y > 1)
  (h : Nat.lcm (x + 2) (y + 2) - Nat.lcm (x + 1) (y + 1) = Nat.lcm (x + 1) (y + 1) - Nat.lcm x y) :
  x ∣ y ∨ y ∣ x :=
sorry

end div_condition_for_lcm_l610_610052


namespace initial_days_to_complete_work_l610_610078

-- We define the initial problem's conditions and the proof goal.
theorem initial_days_to_complete_work (D : ℕ) :
  let original_men := 20 in
  let absent_men := 10 in
  let working_days_after_absence := 40 in
  (original_men - absent_men) * working_days_after_absence = original_men * D → D = 20 :=
by
  intros h
  sorry

end initial_days_to_complete_work_l610_610078


namespace find_a2016_l610_610625

theorem find_a2016 (S : ℕ → ℕ)
  (a : ℕ → ℤ)
  (h₁ : S 1 = 6)
  (h₂ : S 2 = 4)
  (h₃ : ∀ n, S n > 0)
  (h₄ : ∀ n, (S (2 * n - 1))^2 = S (2 * n) * S (2 * n + 2))
  (h₅ : ∀ n, 2 * S (2 * n + 2) = S (2 * n - 1) + S (2 * n + 1))
  : a 2016 = -1009 := 
  sorry

end find_a2016_l610_610625


namespace fixed_point_chord_through_plane_l610_610957

theorem fixed_point_chord_through_plane
  {S : Type*} [normed_group S]
  (circle_S : circle S)
  (tangent_point_N : S)
  (tangent_line_l : line S)
  (diameter_NM : line S)
  (fixed_point_A : S)
  (circle_through_A : circle S)
  (intersecting_points_C_D : S)
  (P Q : S) :
  (tangent_point_N ∈ circle_S) ∧ (tangent_line_l ∈ tangent_point_N) ∧ 
  (diameter_NM ∈ circle_S) ∧ (fixed_point_A ∈ diameter_NM) ∧ 
  (circle_through_A ∈ fixed_point_A) ∧ (circle_through_A ∈ tangent_line_l) ∧ 
  (intersecting_points_C_D ∈ intersect circle_through_A tangent_line_l) ∧ 
  (P ∈ circle_through_A) ∧ (Q ∈ circle_through_A) →
  ∃ K : S, K ∈ diameter_NM ∧ chord_through (P, Q) K := sorry

end fixed_point_chord_through_plane_l610_610957


namespace inverse_of_A_squared_l610_610254

variable {k : ℝ}

theorem inverse_of_A_squared
  (A_inv : Matrix (Fin 2) (Fin 2) ℝ)
  (hA_inv : A_inv = ![-3, k; 0, 3]) :
  (A_inv * A_inv) = ![9, 0; 0, 9] :=
by
  sorry

end inverse_of_A_squared_l610_610254


namespace sum_of_rational_roots_is_zero_l610_610169

def f (x : ℚ) : ℚ := x^3 - 6*x^2 + 5*x + 2

theorem sum_of_rational_roots_is_zero :
  (∀ r : ℚ, f r = 0 → r = 1 ∨ r = -1 ∨ r = 2 ∨ r = -2) → 
  ∑ r in {r : ℚ | f r = 0}.toFinset, r = 0 :=
by 
  intro h
  -- the proof can be filled in here
  sorry

end sum_of_rational_roots_is_zero_l610_610169


namespace inequality_pos_real_l610_610757

theorem inequality_pos_real (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c)) ≥ (2 / 3) := 
sorry

end inequality_pos_real_l610_610757


namespace probability_of_selecting_kids_from_both_workshops_l610_610909

-- Given definitions
def total_kids : ℕ := 30
def coding_kids : ℕ := 22
def robotics_kids : ℕ := 19

-- Statement of the problem
theorem probability_of_selecting_kids_from_both_workshops :
  let total_pairs := (total_kids * (total_kids - 1)) / 2,
      coding_only := coding_kids - (coding_kids + robotics_kids - total_kids),
      robotics_only := robotics_kids - (coding_kids + robotics_kids - total_kids),
      coding_only_pairs := (coding_only * (coding_only - 1)) / 2,
      robotics_only_pairs := (robotics_only * (robotics_only - 1)) / 2,
      non_mixed_pairs := coding_only_pairs + robotics_only_pairs,
      probability_non_mixed := non_mixed_pairs.to_rat / total_pairs.to_rat,
      probability_mixed := 1 - probability_non_mixed
  in probability_mixed = 32 / 39 :=
sorry

end probability_of_selecting_kids_from_both_workshops_l610_610909


namespace sequence_equivalence_l610_610309

noncomputable theory
open_locale classical

theorem sequence_equivalence (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = a - b) :
  (∀ n : ℕ, ∃ k : ℕ, x_k = n ↔ y_k = x_k repeats at least three times) :=
begin
  sorry
end

end sequence_equivalence_l610_610309


namespace calculate_expression_l610_610548

theorem calculate_expression : 2 * Real.sin (60 * Real.pi / 180) + (-1/2)⁻¹ + abs (2 - Real.sqrt 3) = 0 :=
by
  sorry

end calculate_expression_l610_610548


namespace determine_F_value_l610_610870

theorem determine_F_value (D E F : ℕ) (h1 : (9 + 6 + D + 1 + E + 8 + 2) % 3 = 0) (h2 : (5 + 4 + E + D + 2 + 1 + F) % 3 = 0) : 
  F = 2 := 
by
  sorry

end determine_F_value_l610_610870


namespace man_speed_down_l610_610084

/-- Suppose a man travels up at a speed of 15 km/hr and down at a speed v km/hr. 
Given that his average speed for the entire trip is 19.53488372093023 km/hr, 
we prove that his speed while going down, v, is approximately 27.91 km/hr. -/
theorem man_speed_down
  (d : ℝ) (v : ℝ)
  (h1 : 15 ≠ 0)
  (h2 : v ≠ 0)
  (h3 : 2 * d / (d / 15 + d / v) = 19.53488372093023) :
  v ≈ 27.91 := 
sorry

end man_speed_down_l610_610084


namespace number_of_people_l610_610288

-- Definitions based on the conditions
def condition1 (x : ℕ) : Prop := 5 * x + 45
def condition2 (x : ℕ) : Prop := 7 * x + 3

-- The theorem to prove the number of people in the group is 21
theorem number_of_people (x : ℕ) (h₁ : condition1 x = condition2 x) : x = 21 := by
  sorry

end number_of_people_l610_610288


namespace alternating_sum_value_l610_610331

variable {a : ℕ → ℝ}

-- Conditions
axiom sequence_condition (k : ℕ) (hk : k > 0) :
  ∑ n in Finset.range k, (Nat.choose n k) * a n + Finset.sum (Finset.filter (λ i, i ≥ k) (Finset.range k)) (λ n => (Nat.choose n k) * a n) = 1 / (5 ^ k)

-- Equivalent proof problem
theorem alternating_sum_value :
  (100 * 5 + 42) = 542 :=
by
  sorry

end alternating_sum_value_l610_610331


namespace triangle_identity_l610_610723

variable (a b c C : ℝ)
variable (h : ∠ABC) -- Denote that this is a triangle angle in some manner.

-- Utilize geometric conditions and identities
axiom law_of_cosines : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C
axiom pythagorean_identity : Real.cos (C / 2)^2 + Real.sin (C / 2)^2 = 1

theorem triangle_identity :
  (a - b)^2 * Real.cos (C / 2)^2 + (a + b)^2 * Real.sin (C / 2)^2 = c^2 :=
by
  sorry

end triangle_identity_l610_610723


namespace minimum_lambda_exists_l610_610325

noncomputable def f (a b c : ℝ) : ℝ :=
  1 / Real.sqrt (1 + 2 * a) + 1 / Real.sqrt (1 + 2 * b) + 1 / Real.sqrt (1 + 2 * c)

theorem minimum_lambda_exists (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 1) :
  (∃ (λ : ℝ), ∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 → f a b c < λ) ∧
  (∀ (x : ℝ), (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧ f a b c < x) → x ≥ 2) :=
sorry

end minimum_lambda_exists_l610_610325


namespace desired_selling_price_per_pound_of_mixture_l610_610072

theorem desired_selling_price_per_pound_of_mixture :
  ∀ (candies : Type) (cost : candies → ℕ) (weight : candies → ℕ) (is_four_dollar_candy : candies → Prop),
  (weight c₃ = 16) → (cost c₄ = 2) →
  (Σ' (c : candies), weight c = 80 ∧ is_four_dollar_candy c) → 
  Σ (mix : candies),
  ((weight mix = 80) ∧ (∑ c ∈ mix, weight c * cost c) / 80 = 2.20) := 
by
  sorry

end desired_selling_price_per_pound_of_mixture_l610_610072


namespace solution_set_l610_610637

variables (a c d b x : ℝ) (y1 y2 : ℝ → ℝ)
variable (m : ℝ)

-- Definitions based on conditions
def y1 := λ x : ℝ, a * x + b
def y2 := λ x : ℝ, c * x + d

-- Hypotheses based on the conditions
axiom h1 : a > c
axiom h2 : c > 0
axiom h3 : (a * 2 + b) = (c * 2 + d)

-- The statement we need to prove
theorem solution_set : (a - c) * x ≤ d - b → x ≤ 2  :=
sorry

end solution_set_l610_610637


namespace greatest_four_digit_multiple_of_17_l610_610009

theorem greatest_four_digit_multiple_of_17 :
  ∃ n, (n % 17 = 0) ∧ (1000 ≤ n ∧ n ≤ 9999) ∧ ∀ m, (m % 17 = 0) ∧ (1000 ≤ m ∧ m ≤ 9999) → n ≥ m :=
begin
  use 9996,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm2a hm2b,
    exact nat.mul_le_mul_right _ (nat.div_le_of_le_mul (nat.le_sub_one_of_lt (lt_of_le_of_lt (nat.mul_le_mul_right _ (nat.le_of_dvd hm1)) (by norm_num)))),
  },
  sorry
end

end greatest_four_digit_multiple_of_17_l610_610009


namespace eventually_periodic_sequence_l610_610310

theorem eventually_periodic_sequence
  (a : ℕ → ℕ) (h_pos : ∀ n, 0 < a n)
  (h_div : ∀ n m, (a (n + 2 * m)) ∣ (a n + a (n + m))) :
  ∃ N d, 0 < N ∧ 0 < d ∧ ∀ n, N < n → a n = a (n + d) :=
by
  sorry

end eventually_periodic_sequence_l610_610310


namespace integral_result_l610_610648

theorem integral_result:
  ( ∃ a : ℝ, (∀ x : ℝ, (x + 1 / (2 * a * x)) ^ 9 = ((binom 9 3) * (1 / (2 * a)) ^ 3 * x ^ 3) +
               ∑ r in finset.range 9, if r ≠ 3 then (((binom 9 r) * (1 / (2 * a)) ^ r * x ^ (9 - 2 * r)) : ℝ) else 0) /\
   (∃ a : ℝ, a = -1) ) ->
  ( ∫ x in 1..e, (x + a / x) = (e^2 - 3) / 2)
:=
by
  sorry

end integral_result_l610_610648


namespace odd_rows_cols_impossible_arrange_crosses_16x16_l610_610511

-- Define the conditions for part (a)
def square (α : Type*) := α × α
def is_odd_row (table : square nat → bool) (n : nat) :=
  ∃ (i : fin n), ∑ j in finset.range n, table (i, j) = 1
def is_odd_col (table : square nat → bool) (n : nat) :=
  ∃ (j : fin n), ∑ i in finset.range n, table (i, j) = 1

-- Part (a) statement
theorem odd_rows_cols_impossible (table : square nat → bool) (n : nat) :
  n = 16 ∧ (∃ (r : ℕ), r = 20) ∧ (∃ (c : ℕ), c = 15) → ¬(is_odd_row table n ∧ is_odd_col table n) :=
begin 
  -- Sorry placeholder for the proof
  sorry,
end

-- Define the conditions for part (b)
def odd_placement_possible (table : square nat → bool) :=
  ∃ (n : nat), n = 16 ∧ (∑ i in finset.range 16, ∑ j in finset.range 16, table (i, j) = 126) ∧ 
  (∀ i, is_odd_row table 16) ∧ (∀ j, is_odd_col table 16)

-- Part (b) statement
theorem arrange_crosses_16x16 (table : square nat → bool) :
  odd_placement_possible table :=
begin 
  -- Sorry placeholder for the proof
  sorry,
end

end odd_rows_cols_impossible_arrange_crosses_16x16_l610_610511


namespace compute_a3_b_neg2_l610_610749

noncomputable def a : ℚ := 4 / 7
noncomputable def b : ℚ := 5 / 6

theorem compute_a3_b_neg2 :
  a^3 * b^(-2) = 2304 / 8575 := 
by
  sorry

end compute_a3_b_neg2_l610_610749


namespace find_general_term_l610_610219

theorem find_general_term (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n, a (n + 1) = 2 * a n + n^2) :
  ∀ n, a n = 7 * 2^(n - 1) - n^2 - 2 * n - 3 :=
by
  sorry

end find_general_term_l610_610219


namespace cost_of_chewing_gum_l610_610907

theorem cost_of_chewing_gum (initial : ℝ) (count_gum : ℕ) (count_chocolate : ℕ) (cost_chocolate : ℝ) (count_candy_cane : ℕ) (cost_candy_cane : ℝ) (leftover : ℝ) : 
  initial = 10 → 
  count_gum = 3 → 
  count_chocolate = 5 → 
  cost_chocolate = 1 → 
  count_candy_cane = 2 → 
  cost_candy_cane = 0.5 → 
  leftover = 1 → 
  let total_spent : ℝ := initial - leftover in
  let total_chocolate : ℝ := count_chocolate * cost_chocolate in
  let total_candy_cane : ℝ := count_candy_cane * cost_candy_cane in
  let total_gum : ℝ := total_spent - (total_chocolate + total_candy_cane) in
  total_gum / count_gum = 1 :=
sorry

end cost_of_chewing_gum_l610_610907


namespace problem_solution_l610_610605

theorem problem_solution (a : ℝ) : 
  (∃ x1 x2 : ℝ, 
    (log 2 (2 * x1 ^ 2 - x1 - 2 * a - 4 * a ^ 2) + 3 * log (1 / 8) (x1 ^ 2 - a * x1 - 2 * a ^ 2) = 0 
    ∧ log 2 (2 * x2 ^ 2 - x2 - 2 * a - 4 * a ^ 2) + 3 * log (1 / 8) (x2 ^ 2 - a * x2 - 2 * a ^ 2) = 0)
    ∧ x1 ≠ x2 
    ∧ 4 < x1 ^ 2 + x2 ^ 2 
    ∧ x1 ^ 2 + x2 ^ 2 < 8
  ) ↔ 
  (a ∈ set.Ioo (3 / 5 : ℝ) 1) :=
sorry

end problem_solution_l610_610605


namespace school_dance_boys_count_l610_610424

theorem school_dance_boys_count :
  let total_attendees := 100
  let faculty_and_staff := total_attendees * 10 / 100
  let students := total_attendees - faculty_and_staff
  let girls := 2 * students / 3
  let boys := students - girls
  boys = 30 := by
  sorry

end school_dance_boys_count_l610_610424


namespace total_cost_correct_l610_610031

noncomputable def total_cost (sandwiches: ℕ) (price_per_sandwich: ℝ) (sodas: ℕ) (price_per_soda: ℝ) (discount: ℝ) (tax: ℝ) : ℝ :=
  let total_sandwich_cost := sandwiches * price_per_sandwich
  let total_soda_cost := sodas * price_per_soda
  let discounted_sandwich_cost := total_sandwich_cost * (1 - discount)
  let total_before_tax := discounted_sandwich_cost + total_soda_cost
  let total_with_tax := total_before_tax * (1 + tax)
  total_with_tax

theorem total_cost_correct : 
  total_cost 2 3.49 4 0.87 0.10 0.05 = 10.25 :=
by
  sorry

end total_cost_correct_l610_610031


namespace rectangle_pairs_l610_610180

theorem rectangle_pairs (w l : ℕ) (h_w_pos : 0 < w) (h_l_pos : 0 < l) (h_area : w * l = 18) :
  (w, l) ∈ {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} :=
sorry

end rectangle_pairs_l610_610180


namespace BD_tangent_to_circumcircle_ADZ_l610_610329

variables {A B C D E X Y Z O : Type*}

-- Angle at A is 90 degrees
def angle_A_90 (A B C : Type*) : Prop :=
  ∠ A B C = 90

-- Angle B is less than angle C
def angle_B_lt_angle_C (B C : Type*) : Prop :=
  ∠ B < ∠ C

-- D is the point where the tangent at A meets BC
def tangent_at_A_meets_BC (A ω B C D : Type*) : Prop :=
  is_tangent ω A ∧ meets (tangent A ω) (line B C) D

-- E is the reflection of A across BC
def reflection_of_A_across_BC (A B C E : Type*) : Prop :=
  reflection A (line B C) E

-- X is the foot of the perpendicular from A to BE
def foot_of_perpendicular_A_to_BE (A B E X : Type*) : Prop :=
  perpendicular A (line B E) X

-- Y is the midpoint of AX
def midpoint_of_AX (A X Y : Type*) : Prop :=
  midpoint (segment A X) Y

-- Z is where BY intersects ω again
def BY_intersects_ω_again (B Y ω Z : Type*) : Prop :=
  intersects (line B Y) ω Z

-- The theorem to be proved:
theorem BD_tangent_to_circumcircle_ADZ
  (A B C D E X Y Z : Type*)
  (h1 : angle_A_90 A B C)
  (h2 : angle_B_lt_angle_C B C)
  (h3 : tangent_at_A_meets_BC A ω B C D)
  (h4 : reflection_of_A_across_BC A B C E)
  (h5 : foot_of_perpendicular_A_to_BE A B E X)
  (h6 : midpoint_of_AX A X Y)
  (h7 : BY_intersects_ω_again B Y ω Z) :
  tangent (line B D) (circumcircle (triangle A D Z)) :=
sorry

end BD_tangent_to_circumcircle_ADZ_l610_610329


namespace exists_integers_a_b_l610_610789

theorem exists_integers_a_b (n : ℕ) (hn : n ≥ 1) : 
  ∃ a b : ℤ, n ∣ (4 * a^2 + 9 * b^2 - 1) := 
by 
  sorry

end exists_integers_a_b_l610_610789


namespace more_likely_same_sector_l610_610106

theorem more_likely_same_sector 
  (p : ℕ → ℝ) 
  (n : ℕ) 
  (hprob_sum_one : ∑ i in Finset.range n, p i = 1) 
  (hprob_nonneg : ∀ i, 0 ≤ p i) : 
  ∑ i in Finset.range n, (p i) ^ 2 
  > ∑ i in Finset.range n, (p i) * (p ((i + 1) % n)) :=
by
  sorry

end more_likely_same_sector_l610_610106


namespace parallelogram_cosine_l610_610414

theorem parallelogram_cosine (a b : ℝ) (h : a ≠ b) :
  ∃ α : ℝ, cos α = 2 * a * b / (a ^ 2 + b ^ 2) :=
sorry

end parallelogram_cosine_l610_610414


namespace distance_travelled_l610_610493

-- Define the existing conditions
def b : ℝ := 10 -- speed of the boat in calm water is 10 mph
def t_with_current : ℝ := 2 -- time to travel with the current is 2 hours
def t_against_current : ℝ := 3 -- time to travel against the current is 3 hours

-- Define the speed of the current
variable c : ℝ

-- Define the distances with and against the current
def d_with_current : ℝ := (b + c) * t_with_current
def d_against_current : ℝ := (b - c) * t_against_current

-- State the theorem to prove the distance d is 24 miles
theorem distance_travelled : d_with_current = 24 :=
by
  -- Given d_with_current = d_against_current
  have h : d_with_current = d_against_current, from sorry
  -- Calculate the current speed c
  have c := ((b * t_against_current) - (b * t_with_current)) / (t_with_current + t_against_current)
  -- Define the distance d with the current
  have d_with_current := (b + c) * t_with_current
  -- Define the distance d against the current
  have d_against_current := (b - c) * t_against_current
  -- Show that both distances are equal to 24
  exact sorry

end distance_travelled_l610_610493


namespace numberSatisfyingConditions_numberWhereEqual_l610_610866

-- Define the range for the numbers to be between 1000 and 9999
def inRange (n : Nat) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define the function to get the digits of a number
def digits (n : Nat) : List Nat :=
  let d₀ := n % 10
  let d₁ := (n / 10) % 10
  let d₂ := (n / 100) % 10
  let d₃ := (n / 1000) % 10
  [d₃, d₂, d₁, d₀]

-- Define the condition to check the sum of the digits is greater than or equal to their product
def sumGTEProd (digits : List Nat) : Prop :=
  let sum := digits.foldl Nat.add 0
  let prod := digits.foldl Nat.mul 1
  sum ≥ prod

-- Condition to check if a number contains zero among its digits
def containsZero (digits : List Nat) : Prop :=
  digits.any (λ d => d = 0)

-- Get the total number of numbers meeting the specified condition
def totalMeetingCondition :=
  let numbers := List.range' 1000 9000
  numbers.filter (λ n => sumGTEProd (digits n)).length

-- Get the number of numbers where the sum equals the product
def sumEqualsProdCount :=
  let numbers := List.range' 1000 9000
  numbers.filter (λ n => let d := digits n; d.foldl Nat.add 0 = d.foldl Nat.mul 1).length

-- Main theorem to be proved
theorem numberSatisfyingConditions : totalMeetingCondition = 2502 :=
  sorry

theorem numberWhereEqual :
  sumEqualsProdCount = 12 :=
  sorry

end numberSatisfyingConditions_numberWhereEqual_l610_610866


namespace license_plate_increase_l610_610695

theorem license_plate_increase :
  let old_license_plates := 26^2 * 10^3
  let new_license_plates := 26^2 * 10^4
  new_license_plates / old_license_plates = 10 :=
by
  sorry

end license_plate_increase_l610_610695


namespace factor_of_polynomial_l610_610156

theorem factor_of_polynomial (t : ℝ) : (x : ℝ) - t ∣ 4 * x^2 + 17 * x - 15 ↔ t = 3/4 ∨ t = -5 :=
by
  simp only [polynomial.div]
  sorry

end factor_of_polynomial_l610_610156


namespace number_of_surjections_l610_610621

def is_surjection {A B : Type} (f : A → B) [Fintype A] [Fintype B] : Prop :=
  ∀ b : B, ∃ a : A, f a = b

theorem number_of_surjections (A B : Type) [Fintype A] [Fintype B]
  (hA : Fintype.card A = 4) (hB : Fintype.card B = 3) :
  (Fintype.card {f : A → B // is_surjection f} = 36) :=
sorry

end number_of_surjections_l610_610621


namespace nineteen_times_eight_pow_n_plus_seventeen_is_composite_l610_610779

theorem nineteen_times_eight_pow_n_plus_seventeen_is_composite 
  (n : ℕ) (h : n > 0) : ¬ Nat.Prime (19 * 8^n + 17) := 
sorry

end nineteen_times_eight_pow_n_plus_seventeen_is_composite_l610_610779


namespace distance_traveled_l610_610048

-- Let T be the time in hours taken to travel the actual distance D at 10 km/hr.
-- Let D be the actual distance traveled by the person.
-- Given: D = 10 * T and D + 40 = 20 * T prove that D = 40.

theorem distance_traveled (T : ℝ) (D : ℝ) 
  (h1 : D = 10 * T)
  (h2 : D + 40 = 20 * T) : 
  D = 40 := by
  sorry

end distance_traveled_l610_610048


namespace mark_garden_total_flowers_l610_610350

theorem mark_garden_total_flowers :
  let yellow := 10
  let purple := yellow + (80 / 100) * yellow
  let total_yellow_purple := yellow + purple
  let green := (25 / 100) * total_yellow_purple
  total_yellow_purple + green = 35 :=
by
  let yellow := 10
  let purple := yellow + (80 / 100) * yellow
  let total_yellow_purple := yellow + purple
  let green := (25 / 100) * total_yellow_purple
  simp [yellow, purple, total_yellow_purple, green]
  sorry

end mark_garden_total_flowers_l610_610350


namespace hitting_same_sector_more_likely_l610_610112

theorem hitting_same_sector_more_likely
  {n : ℕ} (p : Fin n → ℝ) 
  (h_pos : ∀ i, 0 ≤ p i) 
  (h_sum : ∑ i, p i = 1) :
  (∑ i, (p i) ^ 2) > (∑ i, (p i) * (p ((i + 1) % n))) :=
by
  sorry

end hitting_same_sector_more_likely_l610_610112


namespace solve_for_x_l610_610794

theorem solve_for_x (x : ℝ) (h : (x - 6)^4 = (1 / 16)⁻¹) : x = 8 := 
by 
  sorry

end solve_for_x_l610_610794


namespace max_consecutive_sum_l610_610846

theorem max_consecutive_sum (n : ℕ) : 
  (∀ (n : ℕ), (n*(n + 1))/2 ≤ 400 → n ≤ 27) ∧ ((27*(27 + 1))/2 ≤ 400) :=
by
  sorry

end max_consecutive_sum_l610_610846


namespace problem_a_impossible_problem_b_possible_l610_610516

-- Definitions based on the given conditions
def is_odd_row (table : ℕ → ℕ → bool) (n : ℕ) (r : ℕ) : Prop :=
  ∑ c in finset.range n, if table r c then 1 else 0 % 2 = 1

def is_odd_column (table : ℕ → ℕ → bool) (n : ℕ) (c : ℕ) : Prop :=
  ∑ r in finset.range n, if table r c then 1 else 0 % 2 = 1

-- Problem(a): No existence of 20 odd rows and 15 odd columns in any square table
theorem problem_a_impossible (table : ℕ → ℕ → bool) (n : ℕ) :
  (∃r_set c_set, r_set.card = 20 ∧ c_set.card = 15 ∧
  ∀ r ∈ r_set, is_odd_row table n r ∧ ∀ c ∈ c_set, is_odd_column table n c) → false :=
sorry

-- Problem(b): Existence of a 16 x 16 table with 126 crosses where all rows and columns are odd
theorem problem_b_possible : 
  ∃ (table : ℕ → ℕ → bool), 
  (∑ r in finset.range 16, ∑ c in finset.range 16, if table r c then 1 else 0) = 126 ∧
  (∀ r, is_odd_row table 16 r) ∧
  (∀ c, is_odd_column table 16 c) :=
sorry

end problem_a_impossible_problem_b_possible_l610_610516


namespace minimum_value_of_f_l610_610404

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem minimum_value_of_f : ∃ x : ℝ, ∀ y : ℝ, f(x) ≤ f(y) ∧ f(x) = -2 :=
by
  sorry

end minimum_value_of_f_l610_610404


namespace pi_cubed_integers_l610_610983

theorem pi_cubed_integers :
  let π := 3.1415926
  in floor (π^3) = 31 ∧ ceil (π^3) = 32 :=
by
  let π := 3.1415926
  have : π^3 = 31.006276680299816, by norm_num1
  exact ⟨by norm_num1, by norm_num1⟩

end pi_cubed_integers_l610_610983


namespace exists_positive_integers_l610_610419

noncomputable def exists_pos_ints (a b : ℝ) (h_a : 0 < a) (h_b : a < b) : Prop :=
  ∃ (p q r s : ℕ),
    0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s ∧
    a < (p : ℝ) / (q : ℝ) ∧ (p : ℝ) / (q : ℝ) < (r : ℝ) / (s : ℝ) ∧ (r : ℝ) / (s : ℝ) < b ∧
    p^2 + q^2 = r^2 + s^2

theorem exists_positive_integers (a b : ℝ) (h_a : 0 < a) (h_b : a < b) :
  exists_pos_ints a b h_a h_b :=
begin
  sorry
end

end exists_positive_integers_l610_610419


namespace max_wind_power_speed_l610_610397

noncomputable def force (C S ρ v0 v : ℝ): ℝ :=
  (C * S * ρ * (v0 - v)^2) / 2

noncomputable def power (C S ρ v0 v : ℝ): ℝ :=
  force C S ρ v0 v * v

theorem max_wind_power_speed: ∀ (C ρ: ℝ), 
  power C 5 ρ 6 2 = N
where 
  N := power C 5 ρ 6 2 :=
by
  sorry

end max_wind_power_speed_l610_610397


namespace part_a_part_b_l610_610507

-- Definitions based on the problem conditions
def is_square_table (n : ℕ) (table : List (List Bool)) : Prop := 
  table.length = n ∧ ∀ row, row ∈ table → row.length = n

def is_odd_row (row : List Bool) : Prop := 
  row.count (λ x => x = true) % 2 = 1

def is_odd_column (n : ℕ) (table : List (List Bool)) (c : ℕ) : Prop :=
  n > 0 ∧ c < n ∧ (List.count (λ row => row.get! c = true) table) % 2 = 1

-- Part (a) statement: Prove it's impossible to have exactly 20 odd rows and 15 odd columns
theorem part_a (n : ℕ) (table : List (List Bool)) :
  n = 16 → is_square_table n table → 
  (List.count is_odd_row table) = 20 → 
  ((List.range n).count (is_odd_column n table)) = 15 → 
  False := 
sorry

-- Part (b) statement: Prove that it's possible to arrange 126 crosses in a 16x16 table with all odd rows and columns
theorem part_b (table : List (List Bool)) :
  is_square_table 16 table →
  ((table.map (λ row => row.count (λ x => x = true))).sum = 126) →
  (∀ row, row ∈ table → is_odd_row row) →
  (∀ c, c < 16 → is_odd_column 16 table c) →
  True :=
sorry

end part_a_part_b_l610_610507


namespace hyperbola_equation_l610_610706

-- Definitions based on the conditions
def parabola_focus : (ℝ × ℝ) := (2, 0)
def point_on_hyperbola : (ℝ × ℝ) := (1, 0)
def hyperbola_center : (ℝ × ℝ) := (0, 0)
def right_focus_of_hyperbola : (ℝ × ℝ) := parabola_focus

-- Given the above definitions, we should prove that the standard equation of hyperbola C is correct
theorem hyperbola_equation :
  ∃ (a b : ℝ), (a = 1) ∧ (2^2 = a^2 + b^2) ∧
  (hyperbola_center = (0, 0)) ∧ (point_on_hyperbola = (1, 0)) →
  (x^2 - (y^2 / 3) = 1) :=
by
  sorry

end hyperbola_equation_l610_610706


namespace cotangent_expression_l610_610670

variable {x : ℝ}

-- These definitions directly stem from the problem's condition.
def is_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Given the condition:
-- \(\sin x, \cos x, \tan x\) form a geometric sequence.
axiom geom_seq : is_geometric_sequence (sin x) (cos x) (tan x)

-- Prove that \(\cot^6 x - \cot^2 x = 1\)
theorem cotangent_expression : cot x ^ 6 - cot x ^ 2 = 1 :=
by
  sorry

end cotangent_expression_l610_610670


namespace smallest_factor_to_end_with_four_zeros_l610_610478

theorem smallest_factor_to_end_with_four_zeros :
  ∃ x : ℕ, (975 * 935 * 972 * x) % 10000 = 0 ∧
           (∀ y : ℕ, (975 * 935 * 972 * y) % 10000 = 0 → x ≤ y) ∧
           x = 20 := by
  -- The proof would go here.
  sorry

end smallest_factor_to_end_with_four_zeros_l610_610478


namespace prob_three_odds_in_five_dice_is_5_over_16_l610_610341

def is_odd (n : ℕ) : Prop := n % 2 = 1

def prob_exactly_three_odds_in_five_dice :
  ℝ :=
  let p_one_odd := (1 : ℝ) / 2  -- Probability of one die showing an odd number
  let p_three_odds_and_two_evens := (p_one_odd)^3 * (1 - p_one_odd)^2  -- Probability for three odds and two evens
  let num_ways_to_choose_three_odds := Nat.choose 5 3  -- Number of ways to choose three out of five
  num_ways_to_choose_three_odds * p_three_odds_and_two_evens

theorem prob_three_odds_in_five_dice_is_5_over_16 :
  prob_exactly_three_odds_in_five_dice = 5 / 16 :=
by
  sorry

end prob_three_odds_in_five_dice_is_5_over_16_l610_610341


namespace player_b_can_win_condition_one_player_b_cannot_win_condition_two_l610_610840

-- Definitions and conditions
def is_5x5_square (grid : list (list ℕ)) (x y : ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < 5 ∧ 0 ≤ j ∧ j < 5 → grid (x + i) (y + j) = 1

def has_at_least_one_white_corner (grid : list (list ℕ)) (x y : ℕ) : Prop :=
  (grid x y = 1) ∨ (grid (x + 4) y = 1) ∨ (grid x (y + 4) = 1) ∨ (grid (x + 4) (y + 4) = 1)

def has_at_least_two_white_corners (grid : list (list ℕ)) (x y : ℕ) : Prop :=
  ((grid x y = 1 ∧ grid (x + 4) y = 1) ∨
   (grid x y = 1 ∧ grid x (y + 4) = 1) ∨
   (grid x y = 1 ∧ grid (x + 4) (y + 4) = 1) ∨
   (grid (x + 4) y = 1 ∧ grid x (y + 4) = 1) ∨
   (grid (x + 4) y = 1 ∧ grid (x + 4) (y + 4) = 1) ∨
   (grid x (y + 4) = 1 ∧ grid (x + 4) (y + 4) = 1))

noncomputable def player_b_winning_strategy_one (grid : list (list ℕ)) : Prop :=
  ∀ x y, is_5x5_square grid x y → has_at_least_one_white_corner grid x y

noncomputable def player_b_winning_strategy_two (grid : list (list ℕ)) : Prop :=
  ∀ x y, is_5x5_square grid x y → has_at_least_two_white_corners grid x y

-- Theorem statements (without proofs)
theorem player_b_can_win_condition_one (grid : list (list ℕ)) :
  player_b_winning_strategy_one grid :=
sorry

theorem player_b_cannot_win_condition_two (grid : list (list ℕ)) :
  ¬ player_b_winning_strategy_two grid :=
sorry

end player_b_can_win_condition_one_player_b_cannot_win_condition_two_l610_610840


namespace coefficient_x2_sum_of_binomials_l610_610713

theorem coefficient_x2_sum_of_binomials :
  (∑ n in (multiset.range 10).map ((+) 3), (n.choose 2)) = 285 :=
by
  sorry

end coefficient_x2_sum_of_binomials_l610_610713


namespace smallest_n_not_integer_l610_610166

theorem smallest_n_not_integer : ∃ n : ℕ, 0 < n ∧ ∀ m : ℕ, (0 < m ∧ m < n) → Nat.NumFactors n 2 ≠ Nat.NumFactors (2 * n) 2 - Nat.NumFactors n 2 - n := sorry

end smallest_n_not_integer_l610_610166


namespace intersection_M_N_l610_610760

open Set

-- Define the sets M and N based on the provided conditions
def M := {-1, 0, 1}
def N := {x : ℤ | x^2 = x}

-- State the theorem to prove the intersection M ∩ N = {0, 1}
theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l610_610760


namespace relationship_among_abc_l610_610949

noncomputable def a : ℝ := 2 ^ 0.1
noncomputable def b : ℝ := (1 / 2) ^ (-0.4)
noncomputable def c : ℝ := 2 * Real.log 2 / Real.log 7

theorem relationship_among_abc : c < a ∧ a < b := by
  sorry

end relationship_among_abc_l610_610949


namespace find_a_b_l610_610233

noncomputable def f (a b x: ℝ) : ℝ := x / (a * x + b)

theorem find_a_b (a b : ℝ) (h₁ : a ≠ 0) (h₂ : f a b (-4) = 4) (h₃ : ∀ x, f a b x = f b a x) :
  a + b = 3 / 2 :=
sorry

end find_a_b_l610_610233


namespace correct_answers_count_l610_610771

theorem correct_answers_count (C I : ℕ) (h1 : C + I + 3 = 82) (h2 : C - 0.25 * I = 67) : C = 69 :=
by
  sorry

end correct_answers_count_l610_610771


namespace problem1_problem2_l610_610549

theorem problem1 : |Real.sqrt 2 - Real.sqrt 3| + 2 * Real.sqrt 2 = Real.sqrt 3 + Real.sqrt 2 := 
by {
  sorry
}

theorem problem2 : Real.sqrt 5 * (Real.sqrt 5 - 1 / Real.sqrt 5) = 4 := 
by {
  sorry
}

end problem1_problem2_l610_610549


namespace mathematician_daily_questions_l610_610087

theorem mathematician_daily_questions :
  (518 + 476) / 7 = 142 := by
  sorry

end mathematician_daily_questions_l610_610087


namespace distance_between_poles_l610_610092

theorem distance_between_poles (L W : ℝ) (n_poles : ℕ) (hL : L = 90) (hW : W = 60) (h_n : n_poles = 60) :
  let P := 2 * (L + W) in
  let D := P / (n_poles - 1) in
  D = 300 / 59 :=
by 
  sorry

end distance_between_poles_l610_610092


namespace interval_monotonically_decreasing_l610_610365

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) - cos (2 * x)

def g (x : ℝ) (varphi : ℝ) : ℝ := f (x + varphi)

theorem interval_monotonically_decreasing (varphi : ℝ) (k : ℤ) :
  (0 < varphi) ∧ (varphi < π / 2) →
  (∀ x : ℝ, g x varphi ≤ abs (g (π / 6) varphi)) →
  𝒮 = set.Icc (k * π + π / 12) (k * π + 7 * π / 12) ∧ 
  (∀ x ∈ 𝒮, ∀ y ∈ 𝒮, x < y → g y varphi < g x varphi) :=
sorry

end interval_monotonically_decreasing_l610_610365


namespace intersection_A_B_l610_610638

def set_A : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}

def set_B : Set ℝ := {x | x^2 - 3 * x < 0}

theorem intersection_A_B :
  {x : ℤ | x ∈ set_A ∧ x ∈ set_B} = {1, 2} :=
sorry

end intersection_A_B_l610_610638


namespace B_finishes_job_in_37_5_days_l610_610491

variable (eff_A eff_B eff_C : ℝ)
variable (effA_eq_half_effB : eff_A = (1 / 2) * eff_B)
variable (effB_eq_two_thirds_effC : eff_B = (2 / 3) * eff_C)
variable (job_in_15_days : 15 * (eff_A + eff_B + eff_C) = 1)

theorem B_finishes_job_in_37_5_days :
  (1 / eff_B) = 37.5 :=
by
  sorry

end B_finishes_job_in_37_5_days_l610_610491


namespace mathematician_daily_questions_l610_610085

/-- Given 518 questions for the first project and 476 for the second project,
if all questions are to be completed in 7 days, prove that the number
of questions completed each day is 142. -/
theorem mathematician_daily_questions (q1 q2 days questions_per_day : ℕ) 
  (h1 : q1 = 518) (h2 : q2 = 476) (h3 : days = 7) 
  (h4 : q1 + q2 = 994) (h5 : questions_per_day = 994 / 7) :
  questions_per_day = 142 :=
sorry

end mathematician_daily_questions_l610_610085


namespace shoveling_time_proof_l610_610773

-- Definitions based on the conditions
def initial_shoveling_rate : ℕ := 25
def decrease_rate_per_hour : ℕ := 1
def driveway_width : ℕ := 5
def driveway_length : ℕ := 12
def depth_at_top : ℕ := 2
def depth_at_bottom : ℕ := 4
def snowfall_per_hour : ℕ := 2

-- Calculation of total snow volume
def total_snow_volume : ℕ := driveway_width * driveway_length * ((depth_at_top + depth_at_bottom) / 2)

-- Define effective shoveling rate as a function
def effective_shoveling_rate (h : ℕ) : ℕ := initial_shoveling_rate - h - snowfall_per_hour

-- Statement of the theorem to prove
theorem shoveling_time_proof : ∃ n : ℕ, (∑ h in finset.range n, effective_shoveling_rate h) ≥ total_snow_volume ∧ n = 10 := 
by sorry

end shoveling_time_proof_l610_610773


namespace sandy_change_l610_610133

noncomputable def cappuccino_cost := 2
noncomputable def iced_tea_cost := 3
noncomputable def cafe_latte_cost := 1.5
noncomputable def espresso_cost := 1
noncomputable def mocha_cost := 2.5
noncomputable def hot_chocolate_cost := 2

noncomputable def cappuccinos := 4 * cappuccino_cost
noncomputable def iced_teas := 3 * iced_tea_cost
noncomputable def cafe_lattes := 5 * cafe_latte_cost
noncomputable def espressos := 3 * espresso_cost
noncomputable def mochas := 2 * mocha_cost
noncomputable def hot_chocolates := 2 * hot_chocolate_cost

noncomputable def total_cost_of_drinks := cappuccinos + iced_teas + cafe_lattes + espressos + mochas + hot_chocolates
noncomputable def custom_tip := 5
noncomputable def total_cost := total_cost_of_drinks + custom_tip
noncomputable def amount_paid := 60
noncomputable def change := amount_paid - total_cost

theorem sandy_change : change = 18.5 := by
  /- Calculation skipped -/
  sorry

end sandy_change_l610_610133


namespace find_functions_satisfying_eq_l610_610575

def satisfies_functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x + y) * f(x - y) = (f(x) + f(y))^2 - 4 * x^2 * y^2

theorem find_functions_satisfying_eq :
  (∀ f : ℝ → ℝ, satisfies_functional_eq f → (∀ x : ℝ, f(x) = x^2 ∨ f(x) = -x^2)) :=
by
  sorry

end find_functions_satisfying_eq_l610_610575


namespace inequality_positive_numbers_l610_610602

/-- For positive numbers x_1, x_2, ..., x_n, prove the inequality
(1 + x_1) * (1 + x_1 + x_2) * ... * (1 + x_1 + x_2 + ... + x_n) ≥ (n + 1)^(n + 1) * (x_1 * x_2 * ... * x_n) -/
theorem inequality_positive_numbers (n : ℕ) (x : ℕ → ℝ) (hx : ∀ i, 1 ≤ i → i ≤ n → 0 < x i) :
    (∏ i in range n.succ, ∑ j in range i, x (j + 1) + 1) ^ 2 >=
      (n + 1)^(n + 1) * ∏ i in range n, x (i + 1) := by
  sorry

end inequality_positive_numbers_l610_610602


namespace find_second_term_of_ratio_l610_610811

theorem find_second_term_of_ratio
  (a b c d : ℕ)
  (h1 : a = 6)
  (h2 : b = 7)
  (h3 : c = 3)
  (h4 : (a - c) * 4 < a * d) :
  d = 5 :=
by
  sorry

end find_second_term_of_ratio_l610_610811


namespace time_jogging_l610_610676

def distance := 25     -- Distance jogged (in kilometers)
def speed := 5        -- Speed (in kilometers per hour)

theorem time_jogging :
  (distance / speed) = 5 := 
by
  sorry

end time_jogging_l610_610676


namespace cone_volume_l610_610489

theorem cone_volume (slant_height : ℝ) (central_angle_deg : ℝ) (volume : ℝ) :
  slant_height = 1 ∧ central_angle_deg = 120 ∧ volume = (2 * Real.sqrt 2 / 81) * Real.pi →
  ∃ r h, h = Real.sqrt (slant_height^2 - r^2) ∧
    r = (1/3) ∧
    h = (2 * Real.sqrt 2 / 3) ∧
    volume = (1/3) * Real.pi * r^2 * h := 
by
  sorry

end cone_volume_l610_610489


namespace infinite_set_divisor_l610_610312

open Set

noncomputable def exists_divisor (A : Set ℕ) : Prop :=
  ∃ (d : ℕ), d > 1 ∧ ∀ (a : ℕ), a ∈ A → d ∣ a

theorem infinite_set_divisor (A : Set ℕ) (hA1 : ∀ (b : Finset ℕ), (↑b ⊆ A) → ∃ (d : ℕ), d > 1 ∧ ∀ (a : ℕ), a ∈ b → d ∣ a) :
  exists_divisor A :=
sorry

end infinite_set_divisor_l610_610312


namespace smallest_pieces_to_remove_l610_610121

theorem smallest_pieces_to_remove 
  (total_fruit : ℕ)
  (friends : ℕ)
  (h_fruit : total_fruit = 30)
  (h_friends : friends = 4) 
  : ∃ k : ℕ, k = 2 ∧ ((total_fruit - k) % friends = 0) :=
sorry

end smallest_pieces_to_remove_l610_610121


namespace second_catches_first_at_17_l610_610888

-- Define the distance function for the first particle.
def distance_first (t : ℝ) : ℝ := 5 * (6.8 + t)

-- Define the distance function for the second particle based on the arithmetic sequence provided.
def distance_second (t : ℝ) : ℝ := (t / 2) * (5.5 + 0.5 * t)

-- The theorem to prove that the second particle catches up the first particle at t = 17 minutes.
theorem second_catches_first_at_17 : 
  ∃ t : ℝ, t = 17 ∧ distance_second t = distance_first t := 
by 
  use 17
  split
  {
    refl
  }
  {
    -- Skipping the actual proof calculation
    sorry
  }

end second_catches_first_at_17_l610_610888


namespace intersection_of_sets_l610_610609

def E : Set ℝ := { θ | cos θ < sin θ ∧ 0 ≤ θ ∧ θ ≤ 2 * Real.pi }
def F : Set ℝ := { θ | tan θ < sin θ }

theorem intersection_of_sets (θ : ℝ) :
  θ ∈ E ∩ F ↔ θ ∈ Set.Ioo (Real.pi / 2) Real.pi :=
by
  sorry -- Proof not required

end intersection_of_sets_l610_610609


namespace vertical_horizontal_segment_sum_eq_l610_610071

open Set

variables {P : Set (ℝ × ℝ)} (hP : Convex P) (h1 : ∀ v ∈ P, v ∈ (SetOf (λ (x y : ℝ), x ∈ ℤ ∧ y = (v.2 : ℤ))) ∧ (y ∈ (SetOf (λ (x y : ℝ), y ∈ ℤ ∧ x = (v.1 : ℤ))))
  (gridPoints : ∀ e ∈ ∂P, e ∉ {x : ℝ | ∃ k : ℤ, x = k} ∪ {y : ℝ | ∃ k : ℤ, y = k})

theorem vertical_horizontal_segment_sum_eq 
: ∑ (v_segment ∈ {e ∈ (P) | e aligned vertically}) length(v_segment) = ∑ (h_segment ∈ {e ∈ (P) | e aligned horizontally}) length(h_segment) :=
sorry

end vertical_horizontal_segment_sum_eq_l610_610071


namespace sum_first_10_terms_l610_610937

def arithmetic_sequence (a d : Int) (n : Int) : Int :=
  a + (n - 1) * d

def arithmetic_sum (a d : Int) (n : Int) : Int :=
  (n : Int) * a + (n * (n - 1) / 2) * d

theorem sum_first_10_terms  
  (a d : Int)
  (h1 : (a + 3 * d)^2 = (a + 2 * d) * (a + 6 * d))
  (h2 : arithmetic_sum a d 8 = 32)
  : arithmetic_sum a d 10 = 60 :=
sorry

end sum_first_10_terms_l610_610937


namespace problem_a_impossible_problem_b_possible_l610_610517

-- Definitions based on the given conditions
def is_odd_row (table : ℕ → ℕ → bool) (n : ℕ) (r : ℕ) : Prop :=
  ∑ c in finset.range n, if table r c then 1 else 0 % 2 = 1

def is_odd_column (table : ℕ → ℕ → bool) (n : ℕ) (c : ℕ) : Prop :=
  ∑ r in finset.range n, if table r c then 1 else 0 % 2 = 1

-- Problem(a): No existence of 20 odd rows and 15 odd columns in any square table
theorem problem_a_impossible (table : ℕ → ℕ → bool) (n : ℕ) :
  (∃r_set c_set, r_set.card = 20 ∧ c_set.card = 15 ∧
  ∀ r ∈ r_set, is_odd_row table n r ∧ ∀ c ∈ c_set, is_odd_column table n c) → false :=
sorry

-- Problem(b): Existence of a 16 x 16 table with 126 crosses where all rows and columns are odd
theorem problem_b_possible : 
  ∃ (table : ℕ → ℕ → bool), 
  (∑ r in finset.range 16, ∑ c in finset.range 16, if table r c then 1 else 0) = 126 ∧
  (∀ r, is_odd_row table 16 r) ∧
  (∀ c, is_odd_column table 16 c) :=
sorry

end problem_a_impossible_problem_b_possible_l610_610517


namespace fraction_order_l610_610855

theorem fraction_order :
  (21:ℚ) / 17 < (23:ℚ) / 18 ∧ (23:ℚ) / 18 < (25:ℚ) / 19 :=
by
  sorry

end fraction_order_l610_610855


namespace number_of_boys_l610_610429

-- Define the conditions
def total_attendees : Nat := 100
def faculty_percentage : Rat := 0.1
def faculty_count : Nat := total_attendees * faculty_percentage
def student_count : Nat := total_attendees - faculty_count
def girls_fraction : Rat := 2 / 3
def girls_count : Nat := student_count * girls_fraction

-- Define the question in terms of a Lean theorem
theorem number_of_boys :
  total_attendees = 100 →
  faculty_percentage = 0.1 →
  faculty_count = 10 →
  student_count = 90 →
  girls_fraction = 2 / 3 →
  girls_count = 60 →
  student_count - girls_count = 30 :=
by
  intros
  sorry -- Skip the proof

end number_of_boys_l610_610429


namespace triangle_area_l610_610864

-- Conditions
variable (perimeter : ℝ) (inradius : ℝ)
variable (semi_perimeter : ℝ := perimeter / 2)

-- Definitions
def area_of_triangle (perimeter : ℝ) (inradius : ℝ) : ℝ :=
  inradius * (perimeter / 2)

-- Statement to prove
theorem triangle_area :
  area_of_triangle 39 1.5 = 29.25 :=
by
  sorry

end triangle_area_l610_610864


namespace sequence_sum_l610_610689

open Nat

theorem sequence_sum :
  (∀ n, 
    (n = 1 → a n = 2 ∧ b n = 0) ∧
    (n > 1 → 2 * a (n + 1) = 3 * a n + b n + 2 ∧ 2 * b (n + 1) = a n + 3 * b n - 2)
  ) →
  a 2024 + b 2023 = 3 * 2 ^ 2022 + 1 :=
by
  sorry

end sequence_sum_l610_610689


namespace relationship_among_abc_l610_610950

-- Defining the variables according to the problem statement
def a : ℝ := 3^(-1/2)
def b : ℝ := real.log 2 / real.log 3  -- b is log base 3 of 1/2 which is log(1/2)/log(3)
def c : ℝ := real.log 3 / real.log 2  -- c is log base 1/2 of 1/3 which is log(3) / log(1/2)

-- Statement of the theorem
theorem relationship_among_abc : c > a ∧ a > b :=
by
  sorry

end relationship_among_abc_l610_610950


namespace vector_sum_zero_l610_610194

theorem vector_sum_zero 
  (m n : ℝ)
  (a : ℝ × ℝ × ℝ)
  (b : ℝ × ℝ × ℝ)
  (h_a : a = (-1, m, 2))
  (h_b : b = (n, 1, -2))
  (h_parallel : ∃ t : ℝ, (a.fst + 3 * b.fst, a.snd + 3 * b.snd, a.snd_2 + 3 * b.snd_2) 
    = (t * (3 * a.fst - b.fst), t * (3 * a.snd - b.snd), t * (3 * a.snd_2 - b.snd_2))) :
  m + n = 0 := 
sorry

end vector_sum_zero_l610_610194


namespace henry_age_is_20_l610_610824

open Nat

def sum_ages (H J : ℕ) : Prop := H + J = 33
def age_relation (H J : ℕ) : Prop := H - 6 = 2 * (J - 6)

theorem henry_age_is_20 (H J : ℕ) (h1 : sum_ages H J) (h2 : age_relation H J) : H = 20 :=
by
  -- Proof goes here
  sorry

end henry_age_is_20_l610_610824


namespace problem_statement_l610_610314

-- Define the universe of natural numbers as U
def U := Nat

-- Define the set S
def S : Set U := {x | x * x - x = 0}

-- Define the set T
def T : Set U := {x | x > 2 ∧ (6 / (x - 2)) ∈ Int}

-- Statement of the problem
theorem problem_statement : S ∩ T = S :=
sorry

end problem_statement_l610_610314


namespace cuboid_third_edge_length_l610_610382

theorem cuboid_third_edge_length
  (l w : ℝ)
  (A : ℝ)
  (h : ℝ)
  (hl : l = 4)
  (hw : w = 5)
  (hA : A = 148)
  (surface_area_formula : A = 2 * (l * w + l * h + w * h)) :
  h = 6 :=
by
  sorry

end cuboid_third_edge_length_l610_610382


namespace incorrect_external_diagonals_l610_610037

-- Variables for the side lengths of the prism
variables {a b c : ℝ}

-- Definition of the squared diagonal lengths
def squared_diagonals (a b c : ℝ) : set ℝ := { a^2 + b^2, b^2 + c^2, a^2 + c^2 }

-- Sets provided in the problem
def setA : set ℝ := { 5^2, 6^2, 9^2 }
def setB : set ℝ := { 5^2, 8^2, 9^2 }
def setC : set ℝ := { 6^2, 8^2, 10^2 }
def setD : set ℝ := { 7^2, 8^2, 11^2 }
def setE : set ℝ := { 6^2, 7^2, 10^2 }

-- The theorem to prove that setD could not be the lengths of the external diagonals
theorem incorrect_external_diagonals (a b c : ℝ) :
  setD ≠ squared_diagonals a b c := by sorry

end incorrect_external_diagonals_l610_610037


namespace angle_equality_l610_610735

-- Definitions based on given conditions
variable (C1 C2 : Circle)
variable (O1 O2 A P1 P2 Q1 Q2 M1 M2 : Point)
variable (h1 : C1.center = O1)
variable (h2 : C2.center = O2)
variable (h3 : A ∈ C1 ∧ A ∈ C2)
variable (h4 : P1 ∈ C1 ∧ Q1 ∈ C1)
variable (h5 : P2 ∈ C2 ∧ Q2 ∈ C2)
variable (h6 : tangent P1 P2 ↔ tangent Q1 Q2)
variable (h7 : M1 = midpoint P1 Q1)
variable (h8 : M2 = midpoint P2 Q2)

-- The statement to prove
theorem angle_equality : ∠(O1, A, O2) = ∠(M1, A, M2) :=
sorry

end angle_equality_l610_610735


namespace find_exponent_l610_610683

theorem find_exponent (m x y a : ℝ) (h : y = m * x ^ a) (hx : x = 1 / 4) (hy : y = 1 / 2) : a = 1 / 2 :=
by
  sorry

end find_exponent_l610_610683


namespace minimum_value_of_expression_l610_610315

noncomputable def min_value_expr (a b : ℝ) : ℝ :=
  (a + 1/b) * (a + 1/b - 2023) + (b + 1/a) * (b + 1/a - 2023)

theorem minimum_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  min_value_expr a b = -2031948.5 :=
  sorry

end minimum_value_of_expression_l610_610315


namespace correct_answer_is_C_l610_610857

-- Definitions of terms and polynomials
def coefficient (term : ℚ × (ℕ × ℕ)) : ℚ := term.fst
def degree (term : ℚ × (ℕ × ℕ)) : ℕ := term.snd.fst + term.snd.snd

-- Polynomial data
def term1 := (-3 : ℚ, (2, 1)) -- Representing -3x^2y
def monomial := (1 : ℚ, (1, 0)) -- Representing x
def term2 := (-1 : ℚ, (3, 0)) -- Representing -x^3
def term3 := (-2 : ℚ, (2, 2)) -- Representing -2x^2y^2
def term4 := (3 : ℚ, (0, 2)) -- Representing 3y^2

-- Condition1: The polynomial terms
def polynomial := [term2, term3, term4]

-- Conditions
def condition1 : coefficient term1 = -3 ∧ degree term1 = 2 + 1 := by sorry
def condition2 : coefficient monomial = 1 ∧ degree monomial = 1 := by sorry
def condition3 : list.length polynomial = 3 ∧ 
                 polynomial.map degree = [3, 4, 2] := by sorry
def condition4 : true := by sorry -- Assuming 0 is a polynomial with a special degree

-- Main theorem
theorem correct_answer_is_C : condition1 ∧ condition2 ∧ condition3 ∧ condition4 → true := 
by sorry

end correct_answer_is_C_l610_610857


namespace samantha_routes_l610_610787

noncomputable def num_routes : ℕ := 8

theorem samantha_routes :
  let 
    house_to_library_ways := 2
    library_to_school_ways := 4
  in 
    house_to_library_ways * 1 * library_to_school_ways = num_routes := by
  sorry

end samantha_routes_l610_610787


namespace vacation_cost_l610_610050

theorem vacation_cost (C P : ℕ) 
    (h1 : C = 5 * P)
    (h2 : C = 7 * (P - 40))
    (h3 : C = 8 * (P - 60)) : C = 700 := 
by 
    sorry

end vacation_cost_l610_610050


namespace find_8b_l610_610252

variable (a b : ℚ)

theorem find_8b (h1 : 4 * a + 3 * b = 5) (h2 : a = b - 3) : 8 * b = 136 / 7 := by
  sorry

end find_8b_l610_610252


namespace trajectory_of_M_l610_610223

variable (P : ℝ × ℝ) (A : ℝ × ℝ := (4, 0))
variable (M : ℝ × ℝ)

theorem trajectory_of_M (hP : P.1^2 + 4 * P.2^2 = 4) (hM : M = ((P.1 + 4) / 2, P.2 / 2)) :
  (M.1 - 2)^2 + 4 * M.2^2 = 1 :=
by
  sorry

end trajectory_of_M_l610_610223


namespace pentagon_property_l610_610568

-- Define the regular pentagon inscribed in a circle
def is_regular_pentagon {α : Type} [metric_space α] (A B C D E : α) : Prop :=
∀ (O : α), dist O A = dist O B ∧ dist O B = dist O C ∧ dist O C = dist O D ∧ dist O D = dist O E ∧
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D E ∧ dist D E = dist E A

-- Statement of the proposition
theorem pentagon_property
  {α : Type} [metric_space α] 
  {A B C D E P : α} {O : α} 
  (h_pentagon : is_regular_pentagon A B C D E) 
  (h_circum : is_circum_circle O A B C D E)
  (h_arc : on_arc A E P O) :
  dist P B + dist P D = dist P A + dist P C + dist P E := 
sorry

end pentagon_property_l610_610568


namespace rectangle_width_l610_610400

theorem rectangle_width (w l A : ℕ) 
  (h1 : l = 3 * w)
  (h2 : A = l * w)
  (h3 : A = 108) : 
  w = 6 := 
sorry

end rectangle_width_l610_610400


namespace area_of_M_geq_l610_610323

open Complex MeasureTheory

noncomputable def M (n : ℕ) : Set ℂ := {z | ∑ k in Finset.range n, (1 : ℝ) / (|z - (k + 1)|) ≥ 1}

/--
  Let \( M=\left\{z \in \mathbb{C} \,\middle\lvert\, \sum_{k=1}^{n} \frac{1}{|z-k|} \geq 1\right\} \).
  In the complex plane, the area of the corresponding region \(\Omega\) is \( S \).
  Prove that:
  \(S \geq \frac{\left(11 n^{2}+1\right) \pi}{12} \).
-/
theorem area_of_M_geq (n : ℕ) :
  volume (M n) ≥ (11 * n ^ 2 + 1) * Real.pi / 12 := sorry

end area_of_M_geq_l610_610323


namespace sum_of_circular_three_digit_numbers_invariant_l610_610406

theorem sum_of_circular_three_digit_numbers_invariant (digits : List ℕ)
  (h1 : Multiset.range 1 (Finset.last' (Finset.range 10) (by simp)) = Multiset.ofList digits)
  (h2 : ∀ (i : ℕ), i ∈ digits → 1 ≤ i ∧ i ≤ 9) :
  ∑ i in Finset.range 9, (100 * digits[i % 9] + 10 * digits[(i+1) % 9] + digits[(i+2) % 9]) = 4995 :=
by {
  sorry
}

end sum_of_circular_three_digit_numbers_invariant_l610_610406


namespace limonia_largest_none_providable_amount_l610_610711

def is_achievable (n : ℕ) (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), x = a * (6 * n + 1) + b * (6 * n + 4) + c * (6 * n + 7) + d * (6 * n + 10)

theorem limonia_largest_none_providable_amount (n : ℕ) : 
  ∃ s, ¬ is_achievable n s ∧ (∀ t, t > s → is_achievable n t) ∧ s = 12 * n^2 + 14 * n - 1 :=
by
  sorry

end limonia_largest_none_providable_amount_l610_610711


namespace contacts_in_first_box_l610_610527

noncomputable def cost_per_contact_second_box : ℝ := 33 / 99
noncomputable def cost_per_contact_chosen_box : ℝ := 1 / 3
noncomputable def cost_of_first_box : ℝ := 25

theorem contacts_in_first_box (X : ℝ) (h : cost_of_first_box / X = cost_per_contact_chosen_box) : X = 75 :=
by
  have cost_per_contact_second_eq : cost_per_contact_second_box = 0.3333 := by simp [cost_per_contact_second_box]
  have cost_per_contact_chosen_eq : cost_per_contact_chosen_box = 0.3333 := by simp [cost_per_contact_chosen_box]
  have h_correct : 25 / 75 = cost_per_contact_chosen_box := by simp
  sorry

end contacts_in_first_box_l610_610527


namespace find_original_radius_l610_610571

theorem find_original_radius :
  (∃ r : ℝ, 3 * Real.pi * (r + 5)^2 = 6 * Real.pi * r^2 ∧ r = -15 + 5 * Real.sqrt 6) :=
begin
  use -15 + 5 * Real.sqrt 6,
  split,
  {
    calc
      3 * Real.pi * ((-15 + 5 * Real.sqrt 6) + 5)^2
        = 3 * Real.pi * (5 * Real.sqrt 6 - 10)^2 : by simp
    ... = 3 * Real.pi * (25 * 6 - 100 + 100 - 25 * 6) : by ring
    ... = 6 * Real.pi * (-15 + 5 * Real.sqrt 6)^2 : by ring,
  },
  {
    refl,
  }
end

end find_original_radius_l610_610571


namespace max_sailboat_speed_correct_l610_610389

noncomputable def max_sailboat_speed (C S ρ : ℝ) (v₀ : ℝ) : ℝ :=
  (v₀ / 3)

theorem max_sailboat_speed_correct :
  ∀ (C ρ : ℝ) (v₀ : ℝ), v₀ = 6 → S = 5 →
  max_sailboat_speed C S ρ v₀ = 2 :=
by
  intros C ρ v₀ hv₀ hS
  unfold max_sailboat_speed
  rw [hv₀, hS]
  norm_num

end max_sailboat_speed_correct_l610_610389


namespace CatsFavoriteNumber_l610_610911

theorem CatsFavoriteNumber :
  ∃ n : ℕ, 
    (10 ≤ n ∧ n < 100) ∧ 
    (∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ n = p1 * p2 * p3) ∧ 
    (∀ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      n ≠ a ∧ n ≠ b ∧ n ≠ c ∧ n ≠ d ∧
      a + b - c = d ∨ b + c - d = a ∨ c + d - a = b ∨ d + a - b = c →
      (a = 30 ∧ b = 42 ∧ c = 66 ∧ d = 78)) ∧
    (n = 70) := by
  sorry

end CatsFavoriteNumber_l610_610911


namespace hitting_same_sector_more_likely_l610_610111

theorem hitting_same_sector_more_likely
  {n : ℕ} (p : Fin n → ℝ) 
  (h_pos : ∀ i, 0 ≤ p i) 
  (h_sum : ∑ i, p i = 1) :
  (∑ i, (p i) ^ 2) > (∑ i, (p i) * (p ((i + 1) % n))) :=
by
  sorry

end hitting_same_sector_more_likely_l610_610111


namespace min_point_translated_graph_l610_610809

theorem min_point_translated_graph :
  let f x := 2 * abs (x + 1) + 5
  let translated_f x := f (x - 4)
  (∃ x, ∀ x', translated_f x ≤ translated_f x' ∧ (x, translated_f x) = (3, 5)) :=
by
  let f := λ x : ℝ, 2 * |x + 1| + 5
  let translated_f := λ x : ℝ, f (x - 4)
  have m:
    ∃ x, (∀ x', translated_f x ≤ translated_f x') := sorry
  use 3
  existsi (∃ x, ∀ x', translated_f x ≤ translated_f x' ∧ (x, translated_f x) = (3,5))
  sorry

end min_point_translated_graph_l610_610809


namespace balls_drawn_ensure_single_color_ge_20_l610_610055

theorem balls_drawn_ensure_single_color_ge_20 (r g y b w bl : ℕ) (h_r : r = 34) (h_g : g = 28) (h_y : y = 23) (h_b : b = 18) (h_w : w = 12) (h_bl : bl = 11) : 
  ∃ (n : ℕ), n ≥ 20 →
    (r + g + y + b + w + bl - n) + 1 > 20 :=
by
  sorry

end balls_drawn_ensure_single_color_ge_20_l610_610055


namespace laborer_monthly_income_l610_610472

theorem laborer_monthly_income
  (I : ℕ)
  (D : ℕ)
  (h1 : 6 * I + D = 510)
  (h2 : 4 * I - D = 270) : I = 78 := by
  sorry

end laborer_monthly_income_l610_610472


namespace part_I_part_II_l610_610626

open Nat

def sequence (a: ℕ → ℕ) : Prop :=
(a 1 = 1) ∧ (∀ n, n ≥ 2 → a n = 3^(n-1) + a (n-1))

theorem part_I (a: ℕ → ℕ) (h: sequence a) :
  a 2 = 4 ∧ a 3 = 13 :=
by
  sorry

theorem part_II (a: ℕ → ℕ) (h: sequence a) :
  ∀ n, a n = (3^n - 1) / 2 :=
by
  sorry

end part_I_part_II_l610_610626


namespace average_age_add_person_l610_610375

theorem average_age_add_person (n : ℕ) (h1 : (∀ T, T = n * 14 → (T + 34) / (n + 1) = 16)) : n = 9 :=
by
  sorry

end average_age_add_person_l610_610375


namespace mark_total_flowers_l610_610353

theorem mark_total_flowers (yellow purple green total : ℕ) 
  (hyellow : yellow = 10)
  (hpurple : purple = yellow + (yellow * 80 / 100))
  (hgreen : green = (yellow + purple) * 25 / 100)
  (htotal : total = yellow + purple + green) : 
  total = 35 :=
by
  sorry

end mark_total_flowers_l610_610353


namespace estimate_less_than_2ε_l610_610278

variable (x y ε : ℝ)
variable (h1 : x > y)
variable (h2 : y > 0)
variable (h3 : x - y < ε)
variable (h4 : ε > 0)

theorem estimate_less_than_2ε :
  (x + 2 * ε) - (y - ε) < 2 * ε :=
by
  calc
    (x + 2 * ε) - (y - ε) = x - y + 2 * ε + ε : by ring
    ... < ε + 2 * ε + ε : by linarith
    ... = 4 * ε : by ring
    ... < 2 * ε : by sorry

end estimate_less_than_2ε_l610_610278


namespace tennis_tournament_handshakes_l610_610544

theorem tennis_tournament_handshakes :
  let teams := 4
  let players_per_team := 2
  let total_players := teams * players_per_team
  let handshakes_per_player := total_players - players_per_team in
  (total_players * handshakes_per_player) / 2 = 24 :=
by
  let teams := 4
  let players_per_team := 2
  let total_players := teams * players_per_team
  let handshakes_per_player := total_players - players_per_team
  have h : (total_players * handshakes_per_player) / 2 = 24 := sorry
  exact h

end tennis_tournament_handshakes_l610_610544


namespace quadratic_polynomial_value_l610_610636

noncomputable def f (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

theorem quadratic_polynomial_value (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (h1 : f 1 (-(a + b + c)) (a*b + b*c + a*c) a = b * c)
  (h2 : f 1 (-(a + b + c)) (a*b + b*c + a*c) b = c * a)
  (h3 : f 1 (-(a + b + c)) (a*b + b*c + a*c) c = a * b) :
  f 1 (-(a + b + c)) (a*b + b*c + a*c) (a + b + c) = a * b + b * c + a * c := sorry

end quadratic_polynomial_value_l610_610636


namespace possible_values_of_x_l610_610494

theorem possible_values_of_x : 
  let x_values := { x : ℕ // x ≠ 0 ∧ 36 % x = 0 ∧ 54 % x = 0 } in
  set.size x_values = 6 :=
by
  sorry

end possible_values_of_x_l610_610494


namespace pq_conditions_l610_610244

theorem pq_conditions (p q : ℝ) (hp : p > 1) (hq : q > 1) (hq_inverse : 1 / p + 1 / q = 1) (hpq : p * q = 9) :
  (p = (9 + 3 * Real.sqrt 5) / 2 ∧ q = (9 - 3 * Real.sqrt 5) / 2) ∨ (p = (9 - 3 * Real.sqrt 5) / 2 ∧ q = (9 + 3 * Real.sqrt 5) / 2) :=
  sorry

end pq_conditions_l610_610244


namespace time_to_fill_pool_l610_610347

variable (pool_volume : ℝ) (fill_rate : ℝ) (leak_rate : ℝ)

theorem time_to_fill_pool (h_pool_volume : pool_volume = 60)
    (h_fill_rate : fill_rate = 1.6)
    (h_leak_rate : leak_rate = 0.1) :
    (pool_volume / (fill_rate - leak_rate)) = 40 :=
by
  -- We skip the proof step, only the theorem statement is required
  sorry

end time_to_fill_pool_l610_610347


namespace students_in_section_A_l610_610830

theorem students_in_section_A :
  ∃ (x : ℕ), 
  let total_weight_A := 60 * x,
      total_weight_B := 70 * 80,
      total_weight_class := total_weight_A + total_weight_B,
      total_students_class := x + 70,
      avg_weight_class := total_weight_class / total_students_class in
  x ≈ 60 ∧ avg_weight_class ≈ 70.77 :=
sorry

end students_in_section_A_l610_610830


namespace prism_non_existence_l610_610093

-- Definitions for clarity
variables {A B C A₁ B₁ C₁ D K : Type}
variables [inhabited A] [inhabited B] [inhabited C] [inhabited A₁] [inhabited B₁] [inhabited C₁] [inhabited D] [inhabited K]

def regular_triangular_prism (prism : Type) :=
  ∃ (A B C A₁ B₁ C₁ : prism), -- vertices of the prism
  geometric_conditions -- placeholder for geometric conditions ensuring it is a regular triangular prism

-- Conditions of the problem
axiom prism_inscribed_in_sphere : regular_triangular_prism A → ∃ sphere, sphere.locked_inscribed
axiom diameter_C1D : C₁D == ∅
axiom midpoint_K : K == midpoint C C₁
axiom given_DK : DK == 2 * sqrt(6)
axiom given_DA : DA == 6

-- Theorem to prove
theorem prism_non_existence : ¬ ∃ (prism : Type) (A B C A₁ B₁ C₁ D K : prism), 
  regular_triangular_prism prism ∧
  prism_inscribed_in_sphere prism ∧
  diameter_C1D ∧
  midpoint_K ∧
  given_DK ∧
  given_DA
:= sorry

end prism_non_existence_l610_610093


namespace geometric_series_common_ratio_l610_610928

theorem geometric_series_common_ratio (a r : ℝ) (n : ℕ) 
(h1 : a = 7 / 3) 
(h2 : r = 49 / 21)
(h3 : r = 343 / 147):
  r = 7 / 3 :=
by
  sorry

end geometric_series_common_ratio_l610_610928


namespace find_x_l610_610172

theorem find_x : ∃ x : ℝ, sqrt (x + 4) = 12 ∧ x = 140 := 
by
  existsi 140
  split
  · norm_num
    sorry
  · norm_num
    sorry

end find_x_l610_610172


namespace nonOpaqueGlasses_l610_610128

-- Each glass can be rotated to 0°, 120°, or 240°
def rotations := {0, 120, 240}

-- Function to calculate the number of ways to arrange five glasses such that the stack is not opaque.
def numberOfNonOpaqueArrangements (n : ℕ) (k : ℕ) : ℕ :=
  let totalArrangements := factorial n * k^n
  let opaqueArrangements := 6000 -- calculated directly from the problem
  totalArrangements - opaqueArrangements

-- There are 5! possible permutations of the five glasses
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- k = 3, 3 possible rotations for the top 4 glasses
def num_rotations : ℕ := 3

-- n = 4, 4 free rotations (other than bottom fixed one)
def num_free_glasses : ℕ := 4

-- Prove the number of non-opaque arrangements is 3720
theorem nonOpaqueGlasses : numberOfNonOpaqueArrangements 5 num_rotations = 3720 := by
  sorry

end nonOpaqueGlasses_l610_610128


namespace monotonically_increasing_range_exists_a_minimum_value_one_l610_610989

def f (a x : ℝ) : ℝ := Real.log (a * x + 1 / 2) + 2 / (2 * x + 1)

-- Problem for (Ⅰ)
theorem monotonically_increasing_range (a : ℝ) (h_pos_a : 0 < a) :
    (∀ x y : ℝ, 0 < x → 0 < y → x < y → f a x ≤ f a y) → 2 ≤ a := sorry

-- Problem for (Ⅱ)
theorem exists_a_minimum_value_one :
    ∃ a : ℝ, (∀ x : ℝ, 0 < x → 1 ≤ f a x) ∧ (∃ x : ℝ, 0 < x ∧ f a x = 1) ↔ a = 1 := sorry

end monotonically_increasing_range_exists_a_minimum_value_one_l610_610989


namespace stratified_sampling_number_of_grade12_students_in_sample_l610_610894

theorem stratified_sampling_number_of_grade12_students_in_sample 
  (total_students : ℕ)
  (students_grade10 : ℕ)
  (students_grade11_minus_grade12 : ℕ)
  (sampled_students_grade10 : ℕ)
  (total_students_eq : total_students = 1290)
  (students_grade10_eq : students_grade10 = 480)
  (students_grade11_minus_grade12_eq : students_grade11_minus_grade12 = 30)
  (sampled_students_grade10_eq : sampled_students_grade10 = 96) :
  ∃ n : ℕ, n = 78 :=
by
  -- Proof would go here, but we are skipping with "sorry"
  sorry

end stratified_sampling_number_of_grade12_students_in_sample_l610_610894


namespace fraction_computation_l610_610137

theorem fraction_computation :
  (2 + 4 - 8 + 16 + 32 - 64) / (4 + 8 - 16 + 32 + 64 - 128) = 1 / 2 :=
by
  sorry

end fraction_computation_l610_610137


namespace range_of_m_l610_610956

def f (x : ℝ) : ℝ := x * |x|

theorem range_of_m (m : ℝ) : (∀ x ≥ 1, f(x + m) + m * f x < 0) ↔ m ≤ -1 := by
  sorry

end range_of_m_l610_610956


namespace saturday_earnings_l610_610872

variable (S : ℝ)
variable (totalEarnings : ℝ := 5182.50)
variable (difference : ℝ := 142.50)

theorem saturday_earnings : 
  S + (S - difference) = totalEarnings → S = 2662.50 := 
by 
  intro h 
  sorry

end saturday_earnings_l610_610872


namespace range_for_a_l610_610434

noncomputable def line_not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (3 * a - 1) * x + (2 - a) * y - 1 = 0 → (x ≥ 0 ∨ y ≥ 0)

theorem range_for_a (a : ℝ) :
  (line_not_in_second_quadrant a) ↔ a ≥ 2 := by
  sorry

end range_for_a_l610_610434


namespace triangle_table_distinct_lines_l610_610826

theorem triangle_table_distinct_lines (a : ℕ) (h : a > 1) : 
  ∀ (n : ℕ) (line : ℕ → ℕ), 
  (line 0 = a) → 
  (∀ k, line (2*k + 1) = line k ^ 2 ∧ line (2*k + 2) = line k + 1) → 
  ∀ i j, i < 2^n → j < 2^n → (i ≠ j → line i ≠ line j) := 
by {
  sorry
}

end triangle_table_distinct_lines_l610_610826


namespace odd_rows_cols_impossible_arrange_crosses_16x16_l610_610510

-- Define the conditions for part (a)
def square (α : Type*) := α × α
def is_odd_row (table : square nat → bool) (n : nat) :=
  ∃ (i : fin n), ∑ j in finset.range n, table (i, j) = 1
def is_odd_col (table : square nat → bool) (n : nat) :=
  ∃ (j : fin n), ∑ i in finset.range n, table (i, j) = 1

-- Part (a) statement
theorem odd_rows_cols_impossible (table : square nat → bool) (n : nat) :
  n = 16 ∧ (∃ (r : ℕ), r = 20) ∧ (∃ (c : ℕ), c = 15) → ¬(is_odd_row table n ∧ is_odd_col table n) :=
begin 
  -- Sorry placeholder for the proof
  sorry,
end

-- Define the conditions for part (b)
def odd_placement_possible (table : square nat → bool) :=
  ∃ (n : nat), n = 16 ∧ (∑ i in finset.range 16, ∑ j in finset.range 16, table (i, j) = 126) ∧ 
  (∀ i, is_odd_row table 16) ∧ (∀ j, is_odd_col table 16)

-- Part (b) statement
theorem arrange_crosses_16x16 (table : square nat → bool) :
  odd_placement_possible table :=
begin 
  -- Sorry placeholder for the proof
  sorry,
end

end odd_rows_cols_impossible_arrange_crosses_16x16_l610_610510


namespace greatest_four_digit_multiple_of_17_l610_610018

theorem greatest_four_digit_multiple_of_17 : ∃ n, (1000 ≤ n) ∧ (n ≤ 9999) ∧ (17 ∣ n) ∧ ∀ m, (1000 ≤ m) ∧ (m ≤ 9999) ∧ (17 ∣ m) → m ≤ n :=
begin
  sorry
end

end greatest_four_digit_multiple_of_17_l610_610018


namespace lemming_average_distance_is_correct_l610_610885

def lemming_average_distance : Prop :=
  let side_length : ℝ := 12
  let diagonal_length : ℝ := Real.sqrt (2 * (side_length ^ 2))
  let fraction_moved : ℝ := 7.8 / diagonal_length
  let initial_x : ℝ := fraction_moved * side_length
  let initial_y : ℝ := fraction_moved * side_length
  let angle : ℝ := Real.pi / 3
  let move_distance : ℝ := 3
  let delta_x : ℝ := move_distance * Real.cos(angle / 2)
  let delta_y : ℝ := move_distance * Real.sin(angle / 2)
  let final_x : ℝ := initial_x + delta_x
  let final_y : ℝ := initial_y + delta_y
  let distance_to_side : ℝ → ℝ := λ pos, if pos < side_length / 2 then pos else side_length - pos
  let dist_left := distance_to_side final_x
  let dist_bottom := distance_to_side final_y
  let dist_right := distance_to_side (side_length - final_x)
  let dist_top := distance_to_side (side_length - final_y)
  let average_dist := (dist_left + dist_bottom + dist_right + dist_top) / 4
  average_dist = 6

theorem lemming_average_distance_is_correct : 
  lemming_average_distance := 
by
  -- Proof here
  sorry

end lemming_average_distance_is_correct_l610_610885


namespace probability_greater_equal_zero_l610_610624

noncomputable def random_variable : Type := ℝ -- representing the real-valued random variable

variables (μ σ : ℝ) (X : random_variable) [NormalDist X μ σ]

theorem probability_greater_equal_zero (hX_mean : μ = 1) (hX_prob : P(X > 2) = 0.3) : P(X ≥ 0) = 0.7 :=
by sorry

end probability_greater_equal_zero_l610_610624


namespace problem1_problem2_l610_610721

-- The first proof problem
theorem problem1 (A : ℝ) (h1 : sin (A + π/6) = 2 * cos A) : A = π/3 :=
sorry

-- The second proof problem
theorem problem2 (A B C : ℝ) (a b c : ℝ)
  (h1 : cos A = 1/3) 
  (h2 : b = 3 * c) 
  (h3 : ∀ x y z : ℝ, x^2 = y^2 + z^2 - 2 * y * z * cos A)
  (h4 : ∀ x y z : ℝ, x/z = y/sin y) 
  : sin C = 1/3 :=
sorry

end problem1_problem2_l610_610721


namespace remainder_is_neg_x_plus_60_l610_610738

theorem remainder_is_neg_x_plus_60 (R : Polynomial ℝ) :
  (R.eval 10 = 50) ∧ (R.eval 50 = 10) → 
  ∃ Q : Polynomial ℝ, R = (Polynomial.X - 10) * (Polynomial.X - 50) * Q + (- Polynomial.X + 60) :=
by
  sorry

end remainder_is_neg_x_plus_60_l610_610738


namespace greatest_four_digit_multiple_of_17_l610_610001

theorem greatest_four_digit_multiple_of_17 :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 17 = 0 ∧ ∀ m, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 17 = 0) → m ≤ n :=
  ⟨9996, by {
        split,
        { linarith },
        { split,
            { linarith },
            { split,
                { exact ModEq.rfl },
                { intros m hm hle,
                  have h : m ≤ 9999 := hm.2.2,
                  have : m = 17 * (m / 17) := (Nat.div_mul_cancel hm.2.1).symm,
                  have : 17 * (m / 17) ≤ 17 * 588 := Nat.mul_le_mul_left 17 (Nat.div_le_of_le_mul (by linarith)),
                  linarith,
                },
            },
        },
    },
  ⟩ sorry

end greatest_four_digit_multiple_of_17_l610_610001


namespace expand_product_l610_610921

theorem expand_product (x : ℝ) : 4 * (x + 3) * (x + 6) = 4 * x^2 + 36 * x + 72 :=
by
  sorry

end expand_product_l610_610921


namespace minimum_value_of_quadratic_l610_610810

theorem minimum_value_of_quadratic :
  ∀ (a : ℝ), (1 : ℝ) * (1 : ℝ) = -(a / (2 * (1 : ℝ))) → ∃ y_min, y_min = 1 ∧ ∀ x : ℝ, (x^2 - a*x + 2) ≥ y_min := by
  intros a axis_of_symmetry
  have h : a = 2 := by
    -- solving for 'a' from the symmetry axis condition
    rw [← mul_eq_mul_right_iff] at axis_of_symmetry
    cases axis_of_symmetry
    · linarith
    · linarith
  use 1
  sorry

end minimum_value_of_quadratic_l610_610810


namespace gilbert_herb_plants_count_l610_610189

variable (initial_basil : Nat) (initial_parsley : Nat) (initial_mint : Nat)
variable (dropped_basil_seeds : Nat) (rabbit_ate_all_mint : Bool)

def total_initial_plants (initial_basil initial_parsley initial_mint : Nat) : Nat :=
  initial_basil + initial_parsley + initial_mint

def total_plants_after_dropping_seeds (initial_basil initial_parsley initial_mint dropped_basil_seeds : Nat) : Nat :=
  total_initial_plants initial_basil initial_parsley initial_mint + dropped_basil_seeds

def total_plants_after_rabbit (initial_basil initial_parsley initial_mint dropped_basil_seeds : Nat) (rabbit_ate_all_mint : Bool) : Nat :=
  if rabbit_ate_all_mint then 
    total_plants_after_dropping_seeds initial_basil initial_parsley initial_mint dropped_basil_seeds - initial_mint 
  else 
    total_plants_after_dropping_seeds initial_basil initial_parsley initial_mint dropped_basil_seeds

theorem gilbert_herb_plants_count
  (h1 : initial_basil = 3)
  (h2 : initial_parsley = 1)
  (h3 : initial_mint = 2)
  (h4 : dropped_basil_seeds = 1)
  (h5 : rabbit_ate_all_mint = true) :
  total_plants_after_rabbit initial_basil initial_parsley initial_mint dropped_basil_seeds rabbit_ate_all_mint = 5 := by
  sorry

end gilbert_herb_plants_count_l610_610189


namespace square_exists_l610_610211

-- The points A, B, C, and D are assumed to be collinear
variables (A B C D : ℝ)

-- A function to check if a square with the described properties can be constructed
def construct_square : Prop :=
  ∃ (E F G H : ℝ × ℝ), 
  -- these points form a square
  (dist E F = dist F G ∧ dist G H = dist H E ∧ dist E F = dist G H ∧ dist F G = dist H E) ∧
  -- two opposite sides pass through points A and B
  (line_through A B = line_through E F ∨ line_through A B = line_through G H) ∧ 
  -- the other two sides pass through points C and D
  (line_through C D = line_through E G ∨ line_through C D = line_through F H)

theorem square_exists : construct_square A B C D :=
sorry

end square_exists_l610_610211


namespace min_cd_of_square_l610_610677

theorem min_cd_of_square (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : a + b + c + d = n^2) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : c ≠ d) (h8 : a ≠ c) (h9 : a ≠ d) (h10 : b ≠ d) :
  a < b < c < d ∧ c + d = 11 :=
by
  sorry

end min_cd_of_square_l610_610677


namespace DivisorProductCondition_l610_610364

theorem DivisorProductCondition (n : ℕ) :
  let P := (finset.divisors n).prod id
  let Q := (finset.divisors n).prod (λ d, d + 1)
  P ∣ Q → n = 1 ∨ n = 2 :=
sorry

end DivisorProductCondition_l610_610364


namespace division_result_l610_610684

theorem division_result (x : ℝ) (h : (x - 2) / 13 = 4) : (x - 5) / 7 = 7 := by
  sorry

end division_result_l610_610684


namespace average_coins_per_day_l610_610906

theorem average_coins_per_day :
  let a := 10
  let d := 10
  let n := 7
  let extra := 20
  let total_coins := a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d) + (a + 6 * d + extra)
  total_coins = 300 →
  total_coins / n = 300 / 7 :=
by
  sorry

end average_coins_per_day_l610_610906


namespace base16_to_base2_digits_l610_610455

theorem base16_to_base2_digits (n : ℕ) (h : n = 6*16^4 + 6*16^3 + 6*16^2 + 6*16 + 6) : 
  (nat.log2 n) + 1 = 19 :=
sorry

end base16_to_base2_digits_l610_610455


namespace quadratic_real_roots_a_leq_2_l610_610271

theorem quadratic_real_roots_a_leq_2
    (a : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 4*x1 + 2*a = 0) ∧ (x2^2 - 4*x2 + 2*a = 0)) →
    a ≤ 2 :=
by sorry

end quadratic_real_roots_a_leq_2_l610_610271


namespace henry_trip_duration_l610_610667

theorem henry_trip_duration :
  ∃ t1 t2 : Time,
    overlap t1 ∧ overlap t2 ∧
    9 ≤ t1.hour ∧ t1.hour < 10 ∧
    15 ≤ t2.hour ∧ t2.hour < 16 ∧
    t2 - t1 = 6 * 3600 + 33 * 60 :=
sorry

end henry_trip_duration_l610_610667


namespace pure_imaginary_root_l610_610265

theorem pure_imaginary_root (m : ℂ) (h_im : ∃ b : ℝ, m = b * complex.I) :
  (∃ x : ℝ, x^2 + (1 + 2 * complex.I) * x - 2 * (m + 1) = 0) ↔ (m = complex.I ∨ m = -2 * complex.I) :=
by
  sorry

end pure_imaginary_root_l610_610265


namespace twenty_four_x_eq_a_cubed_t_l610_610193

-- Define conditions
variables {x : ℝ} {a t : ℝ}
axiom h1 : 2^x = a
axiom h2 : 3^x = t

-- State the theorem
theorem twenty_four_x_eq_a_cubed_t : 24^x = a^3 * t := 
by sorry

end twenty_four_x_eq_a_cubed_t_l610_610193


namespace minimal_crossings_l610_610793

constant Person : Type
constant MrWebster MrsWebster Father Son Mother DaughterInLaw : Person
constant boat_holds_two : ∀ (p1 p2: Person), Prop
constant quarreling : Person → Person → Prop
constant forbidden_groups : set (set Person)

axiom boat_cap_limit : boat_holds_two MrWebster MrsWebster
axiom quarreling_pairs : quarreling MrWebster Father ∧ quarreling MrWebster Son ∧ quarreling MrsWebster Mother ∧ quarreling MrsWebster DaughterInLaw
axiom forbidden_pairs : forbidden_groups = {{MrWebster, Father, Son}, {MrsWebster, Mother, DaughterInLaw}}

-- Define the starting condition where everyone is on one bank and the goal is to get everyone to the other bank in minimum trips
def start_conditions (persons : set Person) : Prop := 
  persons = {MrWebster, MrsWebster, Father, Son, Mother, DaughterInLaw}

def safe_cross (p1 p2 : Person) (left right : set Person) : Prop :=
  -- Check the constraints regarding quarrels and forbidden groups
  ¬(quarreling p1 p2) ∧ ¬({p1, p2} ∈ forbidden_groups)

-- Define the final goal condition to be achieved in 9 trips
def end_conditions (persons : set Person) : Prop :=
  persons = {MrWebster, MrsWebster, Father, Son, Mother, DaughterInLaw}

theorem minimal_crossings : 
  ∃ t: ℕ, (start_conditions {MrWebster, MrsWebster, Father, Son, Mother, DaughterInLaw} → 
  end_conditions {MrWebster, MrsWebster, Father, Son, Mother, DaughterInLaw} ∧ t = 9) :=
sorry

end minimal_crossings_l610_610793


namespace find_total_votes_cast_l610_610873

-- Definitions based on conditions
def total_votes := ℝ
def candidate_votes (total_votes: ℝ) := 0.35 * total_votes
def rival_votes (total_votes: ℝ) := 0.35 * total_votes + 2340 

-- The proof problem
theorem find_total_votes_cast (V: total_votes) 
  (h1 : candidate_votes V = 0.35 * V)
  (h2 : rival_votes V = 0.35 * V + 2340)
  (h3 : candidate_votes V + rival_votes V = V) : 
  V = 7800 :=
by
  sorry

end find_total_votes_cast_l610_610873


namespace sequence_property_l610_610206

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Given conditions:
-- (1) S_n denotes the sum of the first n terms of the sequence a_n
def Sn (n : ℕ) : ℝ := ∑ i in Finset.range n, a i

-- Statement of the condition
axiom condition_S (n : ℕ) : S (2 * n - 1) ^ 2 + S (2 * n) ^ 2 = 4 * (a (2 * n) - 2)

-- Proof Problem:
-- Prove that 2 * a 1 + a 100 = 0

theorem sequence_property : 2 * a 1 + a 100 = 0 := 
by
  sorry

end sequence_property_l610_610206


namespace final_bicycle_price_l610_610485

def original_price := 200.0
def first_discount := 0.60
def second_discount := 0.20
def third_discount := 0.10

theorem final_bicycle_price : 
  let price_after_first_discount := original_price * (1 - first_discount),
      price_after_second_discount := price_after_first_discount * (1 - second_discount),
      price_after_third_discount := price_after_second_discount * (1 - third_discount)
  in price_after_third_discount = 57.60 :=
by
  sorry

end final_bicycle_price_l610_610485


namespace fill_time_is_40_minutes_l610_610346

-- Definitions based on the conditions
def pool_volume : ℝ := 60 -- gallons
def filling_rate : ℝ := 1.6 -- gallons per minute
def leaking_rate : ℝ := 0.1 -- gallons per minute

-- Net filling rate
def net_filling_rate : ℝ := filling_rate - leaking_rate

-- Required time to fill the pool
def time_to_fill_pool : ℝ := pool_volume / net_filling_rate

-- Theorem to prove the time is 40 minutes
theorem fill_time_is_40_minutes : time_to_fill_pool = 40 := 
by
  -- This is where the proof would go
  sorry

end fill_time_is_40_minutes_l610_610346


namespace limonia_largest_unachievable_l610_610709

noncomputable def largest_unachievable_amount (n : ℕ) : ℕ :=
  12 * n^2 + 14 * n - 1

theorem limonia_largest_unachievable (n : ℕ) :
  ∀ k, ¬ ∃ a b c d : ℕ, 
    k = a * (6 * n + 1) + b * (6 * n + 4) + c * (6 * n + 7) + d * (6 * n + 10) 
    → k = largest_unachievable_amount n :=
sorry

end limonia_largest_unachievable_l610_610709


namespace boarders_initial_count_l610_610818

noncomputable def initial_boarders (x : ℕ) : ℕ := 7 * x

theorem boarders_initial_count (x : ℕ) (h1 : 80 + initial_boarders x = (2 : ℝ) * 16) :
  initial_boarders x = 560 :=
by
  sorry

end boarders_initial_count_l610_610818


namespace triangle_perimeter_range_l610_610630

open Real

theorem triangle_perimeter_range (A B C : ℝ) (a b c R : ℝ)
  (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) (hC : 0 < C ∧ C < π / 2)
  (h_triangle : A + B + C = π) (h_sides : sin A = 2 * a * sin B)
  (hR : R = sqrt 3 / 6)
  (h_sines : a / sin A = b / sin B ∧ b / sin B = c / sin C ∧ c / sin C = 2 * R) :
  let l := a + b + c in
  l ∈ Set.Ioc ((1 + sqrt 3) / 2) (3 / 2) :=
by
  sorry

end triangle_perimeter_range_l610_610630


namespace real_number_pure_imaginary_number_fourth_quadrant_l610_610588

namespace ComplexNumberProblem

def isRealNumber (a : ℝ) : Prop :=
  a^2 + a - 12 = 0

def isPureImaginaryNumber (a : ℝ) : Prop := 
  (a^2 - 2a - 3 = 0) ∧ (a^2 + a - 12 ≠ 0)

def isInFourthQuadrant (a : ℝ) : Prop := 
  (a^2 - 2a - 3 > 0) ∧ (a^2 + a - 12 < 0)

theorem real_number (a : ℝ) : isRealNumber a ↔ (a = -4 ∨ a = 3) := by 
  sorry

theorem pure_imaginary_number (a : ℝ) : isPureImaginaryNumber a ↔ (a = -1) := by 
  sorry

theorem fourth_quadrant (a : ℝ) : isInFourthQuadrant a ↔ (-4 < a ∧ a < -1) := by 
  sorry

end ComplexNumberProblem

end real_number_pure_imaginary_number_fourth_quadrant_l610_610588


namespace greatest_four_digit_multiple_of_17_l610_610014

theorem greatest_four_digit_multiple_of_17 : ∃ n, (1000 ≤ n) ∧ (n ≤ 9999) ∧ (17 ∣ n) ∧ ∀ m, (1000 ≤ m) ∧ (m ≤ 9999) ∧ (17 ∣ m) → m ≤ n :=
begin
  sorry
end

end greatest_four_digit_multiple_of_17_l610_610014


namespace exists_sequence_with_2009_distinct_elements_l610_610599

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 0
  else n.digits.map (λ d, if d = 0 then 1 else d).product

def sequence (a k : ℕ) : ℕ → ℕ
| 0       := a
| (n + 1) := sequence n + k * digit_product (sequence n)

theorem exists_sequence_with_2009_distinct_elements :
  ∃ a k : ℕ, ∀ m1 m2 : ℕ, m1 < 2009 → m2 < 2009 → m1 ≠ m2 → sequence a k m1 ≠ sequence a k m2 :=
  sorry

end exists_sequence_with_2009_distinct_elements_l610_610599


namespace mathematician_daily_questions_l610_610086

/-- Given 518 questions for the first project and 476 for the second project,
if all questions are to be completed in 7 days, prove that the number
of questions completed each day is 142. -/
theorem mathematician_daily_questions (q1 q2 days questions_per_day : ℕ) 
  (h1 : q1 = 518) (h2 : q2 = 476) (h3 : days = 7) 
  (h4 : q1 + q2 = 994) (h5 : questions_per_day = 994 / 7) :
  questions_per_day = 142 :=
sorry

end mathematician_daily_questions_l610_610086


namespace distribution_plans_l610_610878

/--
A company is recruiting 8 employees and plans to evenly distribute them
to two sub-departments, A and B. Two of the employees are English translators
who cannot be placed in the same department. In addition, three of the 
employees are computer programmers who cannot all be in the same department.

Prove that the number of different distribution plans is 48.
-/
theorem distribution_plans (translators programmers remaining : Finset ℕ) (total : Finset ℕ) :
  translators.card = 2 →
  programmers.card = 3 →
  remaining.card = 3 →
  total.card = 8 →
  translators ∩ programmers = ∅ →
  translators ∩ remaining = ∅ →
  programmers ∩ remaining = ∅ →
  translators ∪ programmers ∪ remaining = total →
  (∃ t1 t2, t1 ≠ t2 ∧ t1 ∈ translators ∧ t2 ∈ translators ∧
   (∃ p1 p2 p3,
      p1 ∈ programmers ∧ p2 ∈ programmers ∧ p3 ∈ programmers ∧
      ∀ a b c, {a, b, c} = {p1, p2, p3} → ¬(a ∈ {p1, p2, p3} ∧ b ∈ {p1, p2, p3} ∧ c ∈ {p1, p2, p3}) →
   (∃ e1 e2 e3,
      e1 ∈ remaining ∧ e2 ∈ remaining ∧ e3 ∈ remaining ∧
      ∀ d1 d2, {d1, d2} = {t1, t2} →
      ∀ setB A, (A ∪ setB = total ∧ A.card = 4 ∧ setB.card = 4) →
      (∃ A_trans B_trans, A_trans.card = 1 ∧ B_trans.card = 1 ∧
      A_trans.inter B_trans = ∅ ∧
      A_trans ∪ B_trans = translators ∧
      (∃ scenarios, (scenarios.card = 6 ∧ 
      (∃ A_remans B_remans, A_remans.card = 3 ∧ B_remans.card = 1 ∧
      scenarios ∪ (A_remans × B_remans) = programmers ∧
      (∃ freeDist, freeDist.card = 8 ∧
      ¬(translators ∪ programmers ∪ remaining ∪ freeDist).nonempty
      → ¬(translators ∪ programmers ∪ remaining ∪ freeDist ∪ { ↔ }).nonempty)))
]))))) = 48 := by sorry

end distribution_plans_l610_610878


namespace edward_baseball_cards_total_l610_610149

theorem edward_baseball_cards_total :
  (7 * 109 = 763) := 
begin
  -- Using Lean's built-in arithmetic capabilities
  sorry
end

end edward_baseball_cards_total_l610_610149


namespace num_different_telephone_numbers_l610_610563

-- Definitions based on conditions
def available_digits := {2, 3, 4, 6, 7, 8, 9}
def num_required_digits := 6

-- The theorem we want to prove
theorem num_different_telephone_numbers : 
  ∃ (n : ℕ), n = fintype.card {t : multiset ℕ // t ⊆ available_digits ∧ t.card = num_required_digits} :=
by
  sorry

end num_different_telephone_numbers_l610_610563


namespace parabola_hyperbola_focus_l610_610266

/-- 
Proof problem: If the focus of the parabola y^2 = 2px coincides with the right focus of the hyperbola x^2/3 - y^2/1 = 1, then p = 2.
-/
theorem parabola_hyperbola_focus (p : ℝ) :
    ∀ (focus_parabola : ℝ × ℝ) (focus_hyperbola : ℝ × ℝ),
      (focus_parabola = (p, 0)) →
      (focus_hyperbola = (2, 0)) →
      (focus_parabola = focus_hyperbola) →
        p = 2 :=
by
  intros focus_parabola focus_hyperbola h1 h2 h3
  sorry

end parabola_hyperbola_focus_l610_610266


namespace perpendicular_lines_have_a_zero_l610_610645

theorem perpendicular_lines_have_a_zero {a : ℝ} :
  ∀ x y : ℝ, (ax + y - 1 = 0) ∧ (x + a*y - 1 = 0) → a = 0 :=
by
  sorry

end perpendicular_lines_have_a_zero_l610_610645


namespace minimize_expression_l610_610974

theorem minimize_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
(h4 : x^2 + y^2 + z^2 = 1) : 
  z = Real.sqrt 2 - 1 :=
sorry

end minimize_expression_l610_610974


namespace solve_for_x_l610_610251

theorem solve_for_x :
  ∀ (x y : ℚ), (3 * x - 4 * y = 8) → (2 * x + 3 * y = 1) → x = 28 / 17 :=
by
  intros x y h1 h2
  sorry

end solve_for_x_l610_610251


namespace base8_9257_digits_product_sum_l610_610847

theorem base8_9257_digits_product_sum :
  let base10 := 9257
  let base8_digits := [2, 2, 0, 5, 1] -- base 8 representation of 9257
  let product_of_digits := 2 * 2 * 0 * 5 * 1
  let sum_of_digits := 2 + 2 + 0 + 5 + 1
  product_of_digits = 0 ∧ sum_of_digits = 10 := 
by
  sorry

end base8_9257_digits_product_sum_l610_610847


namespace number_of_people_l610_610079

theorem number_of_people (x : ℕ) (h1 : 175 = 175) (h2: 2 = 2) (h3 : ∀ (p : ℕ), p * x = 175 + p * 10) : x = 7 :=
sorry

end number_of_people_l610_610079


namespace focaccia_cost_l610_610153

theorem focaccia_cost :
  let almond_croissant := 4.50
  let salami_cheese_croissant := 4.50
  let plain_croissant := 3.00
  let latte := 2.50
  let total_spent := 21.00
  let known_costs := almond_croissant + salami_cheese_croissant + plain_croissant + 2 * latte
  let focaccia_cost := total_spent - known_costs
  focaccia_cost = 4.00 := 
by
  sorry

end focaccia_cost_l610_610153


namespace solve_equation_l610_610582

theorem solve_equation (x : ℝ) : x^4 + (3 - x)^4 = 82 ↔ x = 2.5 ∨ x = 0.5 := by
  sorry

end solve_equation_l610_610582


namespace nth_row_CMG_0789_l610_610459

theorem nth_row_CMG_0789 (n : ℕ) :
  ∑ i in finset.range (2 * n + 1), (2 * n^2 + n + i)^2 = ∑ i in finset.range (n + 1), (2 * n^2 + 2 * n + 1 + i)^2 := 
  sorry

end nth_row_CMG_0789_l610_610459


namespace calc_expression_l610_610129
-- Lean 4 Statement

open Real

theorem calc_expression : 
  8^(-1/3) + (log 6 / log 2) - log2 3 = 3/2 := 
by 
  sorry

end calc_expression_l610_610129


namespace win_sector_area_l610_610066

-- Define the radius of the circle (spinner)
def radius : ℝ := 8

-- Define the probability of winning on one spin
def probability_winning : ℝ := 1 / 4

-- Define the area of the circle, calculated from the radius
def total_area : ℝ := Real.pi * radius^2

-- The area of the WIN sector to be proven
theorem win_sector_area : (probability_winning * total_area) = 16 * Real.pi := by
  sorry

end win_sector_area_l610_610066


namespace find_polynomial_l610_610597

theorem find_polynomial (n : ℕ) (h : n ≥ 2) : 
  ∃ (P_n : ℚ[X]), P_n.eval (real.sqrt_n 2 n) = 1 / (1 + real.sqrt_n 2 n) := 
sorry

end find_polynomial_l610_610597


namespace infinite_set_irrational_is_integer_l610_610732

noncomputable def is_n_free (n : ℕ) (k : ℕ) : Prop :=
  ∀ p : ℕ, nat.prime p → ¬ (p ^ n ∣ k)

theorem infinite_set_irrational_is_integer (n : ℕ) (M : Set ℚ) 
  (h_n_ge_2 : n ≥ 2) 
  (h_n_free : ∀ (k : ℕ), is_n_free n k ↔ ¬ ∃ p : ℕ, nat.prime p ∧ p ^ n ∣ k) 
  (h_M_infinite : M.infinite) 
  (h_M_n_free : ∀ (s : Finset (M : Set ℚ)), s.card = n → is_n_free n (s.prod id)) : ∀ (x : ℚ), x ∈ M → x ∈ ℤ := 
by
  sorry

end infinite_set_irrational_is_integer_l610_610732


namespace gcd_sequence_l610_610743

def sequence (P : ℤ → ℤ) (x : ℕ → ℤ) : Prop :=
  x 0 = 0 ∧ (∀ i : ℕ, x (i + 1) = P (x i))

theorem gcd_sequence (P : ℤ → ℤ) 
  (hP : ∃ a b c d : ℤ, P = λ x, a * x^3 + b * x^2 + c * x + d)
  (hP0 : P 0 = 1) :
  ∃ᶠ n : ℕ in at_top, Nat.gcd (Nat.cast (x n)) (n + 2019) = 1 :=
begin
  sorry
end

end gcd_sequence_l610_610743


namespace knight_liar_paradox_l610_610871

structure Person :=
  (is_knight : Prop)
  (is_liar : Prop)
  (statement : Prop)

def A : Person := { is_knight := False, is_liar := True, statement := B.is_knight }
def B : Person := { is_knight := False, is_liar := True, statement := A.is_liar }

theorem knight_liar_paradox (A B : Person)
  (hA : A.statement = B.is_knight)
  (hB : B.statement = A.is_liar) :
  (B.statement ∧ ¬B.is_knight) ∨ (¬A.statement ∧ ¬A.is_liar) :=
by
  sorry

end knight_liar_paradox_l610_610871


namespace range_of_f_sin_A_in_triangle_l610_610656

def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin (x / 3) * cos (x / 3) - 2 * (sin (x / 3))^2

theorem range_of_f (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π) : 0 ≤ f(x) ∧ f(x) ≤ 1 :=
sorry

theorem sin_A_in_triangle (A B C a b c: ℝ) 
  (hC : f(C) = 1) 
  (h_b_squared : b^2 = a * c) 
  (h_tri: C = π / 2)
  (h_c_squared : c^2 = a^2 + b^2) : 
  sin A = (sqrt 5 - 1) / 2 :=
sorry

end range_of_f_sin_A_in_triangle_l610_610656


namespace maximum_instantaneous_power_l610_610394

noncomputable def sailboat_speed : ℝ → ℝ → ℝ → ℝ 
  | C, S, v_0, v =>
  (C * S * ((v_0 - v) ^ 2)) / 2

theorem maximum_instantaneous_power (C ρ : ℝ)
  (S : ℝ := 5)
  (v_0 : ℝ := 6) :
  let v := (2 : ℝ) 
  (sailboat_speed C S v_0(v) * v)
  = (C * 5 * ρ / 2 -> v = 2) :=
by
  sorry

end maximum_instantaneous_power_l610_610394


namespace cricket_innings_l610_610804

theorem cricket_innings (n : ℕ) (h_avg_initial : (15 : ℚ)) 
  (h_next_runs : (59 : ℚ)) 
  (h_new_avg : (h_avg_initial + 4 = 19)) 
  (h_eq : (15 * n + h_next_runs = 19 * (n + 1))) : 
  n = 10 := 
by 
  sorry

end cricket_innings_l610_610804


namespace bianca_birthday_money_l610_610935

/-- Define the number of friends Bianca has -/
def number_of_friends : ℕ := 5

/-- Define the amount of dollars each friend gave -/
def dollars_per_friend : ℕ := 6

/-- The total amount of dollars Bianca received -/
def total_dollars_received : ℕ := number_of_friends * dollars_per_friend

/-- Prove that the total amount of dollars Bianca received is 30 -/
theorem bianca_birthday_money : total_dollars_received = 30 :=
by
  sorry

end bianca_birthday_money_l610_610935


namespace exists_subset_with_perfect_power_means_l610_610790

open Set

def is_perfect_power (n : ℕ) : Prop :=
∃ k m : ℕ, m > 1 ∧ n = k^m

theorem exists_subset_with_perfect_power_means :
  ∃ A : Set ℕ, (A ⊆ {n | n > 0}) ∧ (A.card = 2022) ∧
    (∀ B ⊆ A, ∃ k m : ℕ, m > 1 ∧ (∑ x in B, x / B.card) = k^m) :=
sorry

end exists_subset_with_perfect_power_means_l610_610790


namespace find_a2_plus_a8_l610_610286

variable {a_n : ℕ → ℤ}  -- Assume the sequence is indexed by natural numbers and maps to integers

-- Define the condition in the problem
def seq_property (a_n : ℕ → ℤ) := a_n 3 + a_n 4 + a_n 5 + a_n 6 + a_n 7 = 25

-- Statement to prove
theorem find_a2_plus_a8 (h : seq_property a_n) : a_n 2 + a_n 8 = 10 :=
sorry

end find_a2_plus_a8_l610_610286


namespace black_car_overtakes_red_car_in_1_hour_l610_610474

theorem black_car_overtakes_red_car_in_1_hour :
  ∀ (dist speed_red speed_black : ℝ), 
    speed_red = 30 → 
    speed_black = 50 → 
    dist = 20 → 
    (dist / (speed_black - speed_red)) = 1 := 
by
  intros dist speed_red speed_black h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end black_car_overtakes_red_car_in_1_hour_l610_610474


namespace math_problem_solution_l610_610856

-- Definition (1)
noncomputable def isFunction1 : Prop :=
  ∀ t : ℝ, ∃ s : ℝ, s = t^2

-- Definition (2)
noncomputable def areFunctionsEqual2 : Prop :=
  ∀ x : ℝ, (|x - 1| = (if x > 1 then x - 1 else 1 - x))

-- Definition (3)
noncomputable def isIncreasing3 : Prop :=
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, x < y → (x < 0) → f x < f y) →
               (∀ x y : ℝ, x < y → (0 ≤ x) → f x < f y) →
               ∀ x y : ℝ, x < y → f x < f y

-- Definition (4)
noncomputable def isValidMSet4 : Prop :=
  ∀ m : ℝ, (({x : ℝ | x^2 + x - 6 = 0} ∪ {x : ℝ | m * x + 1 = 0}) = {x : ℝ | x^2 + x - 6 = 0}) →
           m ∈ {-(1 / 2), 1 / 3}

-- Definition (5)
noncomputable def isNotMonotonicallyDecreasing5 : Prop :=
  ∀ f : ℝ → ℝ, f 2 > f 1 → ∃ x y : ℝ, x < y ∧ f x < f y

-- Definition (6)
noncomputable def isNotGraphTranslate6 : Prop :=
  ∀ f : ℝ → ℝ, (∃ x : ℝ, f (2 * x - 1) = f (2 * x)) ∧ (∃ y : ℝ, f (2 * y - 1) ≠ f (2 * y - 1 + 1))

-- The correct answers
noncomputable def correctAnswers :=
  isFunction1 ∧ ¬ areFunctionsEqual2 ∧ ¬ isIncreasing3 ∧ ¬ isValidMSet4 ∧ isNotMonotonicallyDecreasing5 ∧ ¬ isNotGraphTranslate6

theorem math_problem_solution : correctAnswers := by sorry

end math_problem_solution_l610_610856


namespace train_length_approximation_l610_610900

theorem train_length_approximation :
  let speed_kmph := 54
  let time_seconds := 67.66125376636536
  let bridge_length := 850
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * time_seconds
  let train_length := total_distance - bridge_length
  abs (train_length - 164.92) < 1e-6 := sorry

end train_length_approximation_l610_610900


namespace smallest_n_exists_l610_610731

noncomputable def smallest_n (k : ℕ) : ℕ := if k % 2 = 0 then k / 2 else k / 2 + 1

theorem smallest_n_exists {k : ℕ} (h : 0 < k) : 
  ∃ (v : Fin k → EuclideanSpace ℝ (smallest_n k)), 
    (∀ i j, |i - j| > 1 → inner (v i) (v j) = 0) ∧ 
    (∀ i, ‖v i‖ ≠ 0) :=
sorry

end smallest_n_exists_l610_610731


namespace angles_of_triangle_l610_610082

theorem angles_of_triangle (A B C M : Point)
    (h1 : Geometry.BETWEEN A B C)
    (AM B AB : Line)
    (h2 : BM = AB)
    (h3 : Geometry.angle_bam = 35)
    (h4 : Geometry.angle_cam = 15)
    : Geometry.angle_bac = 50 ∧ Geometry.angle_abc = 110 ∧ Geometry.angle_acb = 20 :=
sorry

end angles_of_triangle_l610_610082


namespace unequal_numbers_l610_610240

theorem unequal_numbers {k : ℚ} (h : 3 * (1 : ℚ) + 7 * (1 : ℚ) + 2 * k = 0) (d : (7^2 : ℚ) - 4 * 3 * 2 * k = 0) : 
    (3 : ℚ) ≠ (7 : ℚ) ∧ (3 : ℚ) ≠ k ∧ (7 : ℚ) ≠ k :=
by
  -- adding sorry for skipping proof
  sorry

end unequal_numbers_l610_610240


namespace largest_divisor_of_odd_product_l610_610845

theorem largest_divisor_of_odd_product (n : ℕ) (h : Even n ∧ n > 0) :
  ∃ m, m > 0 ∧ (∀ k, (n+1)*(n+3)*(n+7)*(n+9)*(n+11) % k = 0 ↔ k ≤ 15) := by
  -- Proof goes here
  sorry

end largest_divisor_of_odd_product_l610_610845


namespace evaluate_polynomial_103_l610_610034

theorem evaluate_polynomial_103 :
  103 ^ 4 - 4 * 103 ^ 3 + 6 * 103 ^ 2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end evaluate_polynomial_103_l610_610034


namespace limit_of_function_l610_610127

theorem limit_of_function :
  (Real.limit (fun x => (x^3 - Real.pi^3) * Real.sin (5 * x) / (Real.exp (Real.sin x ^ 2) - 1)) Real.pi) = -15 * Real.pi ^ 2 :=
by
  sorry

end limit_of_function_l610_610127


namespace mean_not_always_greater_than_each_piece_of_data_l610_610458

variables (α : Type*) (s : set α) (population : set α) (data : set ℝ) (mean : ℝ) (mode : ℝ) (median : ℝ) (variance : ℝ)

def is_population := population = s

def central_tendency := (mean, mode, median) -- Mean, mode, and median describe central tendency.

def fluctuation (variance : ℝ) : Prop := 
  (∀ ε > 0, ∃ n : ℕ, ∀ x ∈ data, |x - mean| < ε → var <= variance )

def incorrect_mean_statement := 
  ∃ s : set ℝ, ∃ mean, s ⊆ data ∧ mean < Sup s

theorem mean_not_always_greater_than_each_piece_of_data
  (h1 : is_population population)
  (h2 : central_tendency (mean, mode, median))
  (h3 : fluctuation variance) :
  incorrect_mean_statement :=
sorry

end mean_not_always_greater_than_each_piece_of_data_l610_610458


namespace find_angle_between_generators_l610_610867

-- Definitions based on the conditions in a)
def angle_between_generators (α : ℝ) (h : α ≤ (2 * π / 3)) : ℝ :=
  2 * Real.arcsin (1 / 2 * Real.tan (α / 2))

-- Lean 4 statement equivalent to the mathematical proof problem
theorem find_angle_between_generators (α : ℝ) (h : α ≤ (2 * π / 3)) :
  angle_between_generators α h = 2 * Real.arcsin (1 / 2 * Real.tan (α / 2)) :=
  sorry

end find_angle_between_generators_l610_610867


namespace same_probability_of_selection_l610_610892

variable (students : Type) [Fintype students]

variable (team1_sample team2_sample : Finset students)

variable (n : ℕ) (h1 : Fintype.card team1_sample = n) (h2 : Fintype.card team2_sample = n)

variable (reasonable_sampling : (x : students) → (x ∈ team1_sample) = (x ∈ team2_sample))

theorem same_probability_of_selection (students : Type) [Fintype students] (team1_sample team2_sample : Finset students) 
  (n : ℕ) (h1 : Fintype.card team1_sample = n) (h2 : Fintype.card team2_sample = n)
  (reasonable_sampling : (x : students) → (x ∈ team1_sample) = (x ∈ team2_sample)) :
  ∀ (x : students), (x ∈ team1_sample) = (x ∈ team2_sample) :=
begin
  sorry
end

end same_probability_of_selection_l610_610892


namespace fraction_left_after_3_days_l610_610465

-- Defining work rates of A and B
def A_rate := 1 / 15
def B_rate := 1 / 20

-- Total work rate of A and B when working together
def combined_rate := A_rate + B_rate

-- Work completed by A and B in 3 days
def work_done := 3 * combined_rate

-- Fraction of work left
def fraction_work_left := 1 - work_done

-- Statement to prove:
theorem fraction_left_after_3_days : fraction_work_left = 13 / 20 :=
by
  have A_rate_def: A_rate = 1 / 15 := rfl
  have B_rate_def: B_rate = 1 / 20 := rfl
  have combined_rate_def: combined_rate = A_rate + B_rate := rfl
  have work_done_def: work_done = 3 * combined_rate := rfl
  have fraction_work_left_def: fraction_work_left = 1 - work_done := rfl
  sorry

end fraction_left_after_3_days_l610_610465


namespace divides_N_l610_610742

theorem divides_N (n : ℕ) (hn : 1065 < n ∧ n < 1982) : 
  n ∣ (915.factorial + ∑ k in finset.range 1066, (915 + k + 1).factorial / k.factorial) :=
sorry

end divides_N_l610_610742


namespace count_positive_integers_satisfying_volume_l610_610462

noncomputable def volume (x : ℕ) : ℤ :=
(x + 3) * (x - 3) * (x^3 - 5 * x + 25)

def minVolume (x : ℕ) : Prop :=
volume x < 1500

theorem count_positive_integers_satisfying_volume : 
  ∃! (n : ℕ), n = 4 ∧ ∀ (x : ℕ), 0 < x ∧ minVolume x ↔ x ∈ {1, 2, 3, 4} := sorry

end count_positive_integers_satisfying_volume_l610_610462


namespace total_worksheets_l610_610525

/- Define the conditions -/
def problems_per_worksheet := 4
def graded_worksheets := 8
def remaining_problems := 32

/- Using the conditions to define total worksheets -/
theorem total_worksheets : 
  (graded_worksheets * problems_per_worksheet) + (remaining_problems / problems_per_worksheet) = 16 :=
by 
  -- Define the graded problems
  have graded_problems : graded_worksheets * problems_per_worksheet = 32 := rfl,
  -- Define the remaining worksheets
  have remaining_worksheets : remaining_problems / problems_per_worksheet = 8 := by norm_num,
  -- Summing the graded worksheets and remaining worksheets
  calc
    8 + 8 = 16 := by norm_num

end total_worksheets_l610_610525


namespace ML_eq_MN_l610_610359

-- Definitions of points and associated geometric conditions.
variables {A B C P L N : Type}
variables [IsTriangle A B C]
variables [IsPointInTriangle P A B C]
variables [AngleEq PAC PBC : IsAngleEq (angle P A C) (angle P B C)]
variables [Midpoint M A B : IsMidpoint M A B]
variables [PerpendicularFoot L P BC : IsPerpendicularFoot L P BC]
variables [PerpendicularFoot N P CA : IsPerpendicularFoot N P CA]

-- The proposition to be proved.
theorem ML_eq_MN : (dist M L) = (dist M N) :=
by sorry

end ML_eq_MN_l610_610359


namespace collin_petals_per_flower_l610_610135

theorem collin_petals_per_flower (c_f : ℕ) (i_f : ℕ) (total_f : ℕ) (total_p : ℕ) (p_per_f : ℕ) :
  c_f = 25 → i_f = 33 → total_f = c_f + i_f / 3 → total_p = 144 → p_per_f = total_p / total_f → p_per_f = 4 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h3
  norm_num at h3
  rw [h4] at h5
  norm_num at h5
  exact h5

end collin_petals_per_flower_l610_610135


namespace village_population_l610_610054

theorem village_population (x : ℝ) (h : 0.96 * x = 23040) : x = 24000 := sorry

end village_population_l610_610054


namespace spinner_four_digit_probability_divisible_by_4_l610_610570

/--
Given a spinner which equally likely outputs the digits 1, 4, or 8, and is spun four times to
construct a four-digit number (with the first spin determining the thousands place, the second spin
determining the hundreds place, the third spin determining the tens place, and the fourth spin 
determining the units place), the probability that the formed four-digit number is divisible by 4 
is 4/9.
-/
theorem spinner_four_digit_probability_divisible_by_4 :
  let digit_options := {1, 4, 8}
  (num_combinations := 81)
  (valid_endings := { (4, 4), (4, 8), (8, 4), (8, 8) })
  (valid_combinations := 36)
  (prob := valid_combinations / num_combinations)
  (fraction_final := 4 / 9)
  prob = fraction_final :=
by
  sorry

end spinner_four_digit_probability_divisible_by_4_l610_610570


namespace collinear_points_coplanar_points_l610_610302

-- Part (1): Proof of collinearity
theorem collinear_points (a b : ℝ) 
  (A : EuclideanSpace ℝ (fin 3)) (B : EuclideanSpace ℝ (fin 3)) (C : EuclideanSpace ℝ (fin 3))
  (hA : A = ![2, a, -1]) (hB : B = ![-2, 3, b]) (hC : C = ![1, 2, -2])
  (h_collinear : ∃ (λ : ℝ), A - C = λ • (B - C)):
  a = 5/3 ∧ b = -5 :=
sorry

-- Part (2): Proof of coplanarity
theorem coplanar_points (a : ℝ) 
  (A : EuclideanSpace ℝ (fin 3)) (B : EuclideanSpace ℝ (fin 3)) (C : EuclideanSpace ℝ (fin 3)) (D : EuclideanSpace ℝ (fin 3))
  (hA : A = ![2, a, -1]) (hB : B = ![-2, 3, -3]) (hC : C = ![1, 2, -2]) (hD : D = ![-1, 3, -3])
  (h_b_value : -3 = -3)
  (h_coplanar : ∃ (x y : ℝ), D - C = x • (A - C) + y • (B - C)):
  a = 1 :=
sorry

end collinear_points_coplanar_points_l610_610302


namespace g_15_33_eq_165_l610_610740

noncomputable def g : ℕ → ℕ → ℕ := sorry

axiom g_self (x : ℕ) : g x x = x
axiom g_comm (x y : ℕ) : g x y = g y x
axiom g_equation (x y : ℕ) : (x + y) * g x y = y * g x (x + y)

theorem g_15_33_eq_165 : g 15 33 = 165 := by sorry

end g_15_33_eq_165_l610_610740


namespace quadrilateral_front_view_iff_cylinder_or_prism_l610_610075

inductive Solid
| cone : Solid
| cylinder : Solid
| triangular_pyramid : Solid
| quadrangular_prism : Solid

def has_quadrilateral_front_view (s : Solid) : Prop :=
  s = Solid.cylinder ∨ s = Solid.quadrangular_prism

theorem quadrilateral_front_view_iff_cylinder_or_prism (s : Solid) :
  has_quadrilateral_front_view s ↔ s = Solid.cylinder ∨ s = Solid.quadrangular_prism :=
by
  sorry

end quadrilateral_front_view_iff_cylinder_or_prism_l610_610075


namespace rectangle_dissection_l610_610327

theorem rectangle_dissection (R : Rectangle) (n : ℕ) (points : Fin n → Point) 
  (h1 : ∀ i j : Fin n, i ≠ j → ¬(points i).x = (points j).x ∧ ¬(points i).y = (points j).y)
  (dissect : List Rectangle)
  (h2 : ∀ r ∈ dissect, sides_parallel_to(R, r))
  (h3 : ∀ i : Fin n, ∀ r ∈ dissect, ¬point_in_interior(points i, r)) :
  dissect.length ≥ n + 1 :=
by
  sorry

end rectangle_dissection_l610_610327


namespace hitting_same_sector_more_likely_l610_610113

theorem hitting_same_sector_more_likely
  {n : ℕ} (p : Fin n → ℝ) 
  (h_pos : ∀ i, 0 ≤ p i) 
  (h_sum : ∑ i, p i = 1) :
  (∑ i, (p i) ^ 2) > (∑ i, (p i) * (p ((i + 1) % n))) :=
by
  sorry

end hitting_same_sector_more_likely_l610_610113


namespace minimal_horses_and_ponies_l610_610860

-- Definitions and conditions
def ponies_with_horseshoes (P : ℕ) := (5 / 6) * P
def icelandic_ponies_with_horseshoes (P : ℕ) := (2 / 3) * (ponies_with_horseshoes P)
def horses (P : ℕ) := P + 4

-- The goal is to prove the minimal value
theorem minimal_horses_and_ponies : ∃ (P H : ℕ), ponies_with_horseshoes P ∈ ℕ ∧ icelandic_ponies_with_horseshoes P ∈ ℕ ∧ H = horses P ∧ P + H = 40 :=
by {
  sorry
}

end minimal_horses_and_ponies_l610_610860


namespace time_to_fill_pool_l610_610348

variable (pool_volume : ℝ) (fill_rate : ℝ) (leak_rate : ℝ)

theorem time_to_fill_pool (h_pool_volume : pool_volume = 60)
    (h_fill_rate : fill_rate = 1.6)
    (h_leak_rate : leak_rate = 0.1) :
    (pool_volume / (fill_rate - leak_rate)) = 40 :=
by
  -- We skip the proof step, only the theorem statement is required
  sorry

end time_to_fill_pool_l610_610348


namespace second_grade_students_in_sample_is_correct_l610_610059

-- Define the number of students in each grade
def first_grade_students : ℕ := 1200
def second_grade_students : ℕ := 900
def third_grade_students : ℕ := 1500

-- Define the total number of students
def total_students : ℕ := first_grade_students + second_grade_students + third_grade_students

-- Define the sample size
def sample_size : ℕ := 720

-- Calculate the proportion of second-grade students
def proportion_second_grade : ℚ := second_grade_students / total_students

-- Calculate the number of second-grade students in the sample
def second_grade_sample : ℕ := proportion_second_grade * sample_size

-- Theorem stating the number of second-grade students in the sample
theorem second_grade_students_in_sample_is_correct : second_grade_sample = 480 := by
  sorry

end second_grade_students_in_sample_is_correct_l610_610059


namespace sphere_segment_volume_l610_610817

theorem sphere_segment_volume (r : ℝ) (ratio_surface_to_base : ℝ) : r = 10 → ratio_surface_to_base = 10 / 7 → ∃ V : ℝ, V = 288 * π :=
by
  intros
  sorry

end sphere_segment_volume_l610_610817


namespace greatest_four_digit_multiple_of_17_l610_610002

theorem greatest_four_digit_multiple_of_17 :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 17 = 0 ∧ ∀ m, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 17 = 0) → m ≤ n :=
  ⟨9996, by {
        split,
        { linarith },
        { split,
            { linarith },
            { split,
                { exact ModEq.rfl },
                { intros m hm hle,
                  have h : m ≤ 9999 := hm.2.2,
                  have : m = 17 * (m / 17) := (Nat.div_mul_cancel hm.2.1).symm,
                  have : 17 * (m / 17) ≤ 17 * 588 := Nat.mul_le_mul_left 17 (Nat.div_le_of_le_mul (by linarith)),
                  linarith,
                },
            },
        },
    },
  ⟩ sorry

end greatest_four_digit_multiple_of_17_l610_610002


namespace flag_arrangement_remainder_l610_610833

theorem flag_arrangement_remainder : 
  let red_flags := 11
  let white_flags := 6
  let total_flags := 17
  let flagpoles := 2
  --  Calculate arrangements ensuring each flagpole has at least one flag and no two white flags are adjacent
  let valid_arrangements := (let slots := red_flags + white_flags 
                             let ways_to_place_white := Nat.C (red_flags + 1) white_flags 
                             let valid_with_divider := ways_to_place_white * (red_flags + 1)
                             let overcount := 2 * Nat.C red_flags white_flags 
                             valid_with_divider - overcount)
  in valid_arrangements % 1000 = 164 :=
  sorry

end flag_arrangement_remainder_l610_610833


namespace find_incorrect_statements_l610_610041

-- Define vector space and orthogonality
variables {V : Type*} [InnerProductSpace ℝ V]

-- Define the parallel relationship
def parallel (a b : V) : Prop := ∃ k : ℝ, a = k • b

-- Statement A
def statement_a (a b c : V) : Prop :=
  parallel a b ∧ parallel b c → parallel a c

-- Statement B
def statement_b (a b : V) : Prop :=
  ∥a∥ = ∥b∥ ∧ parallel a b → a = b

-- Statement C
def statement_c (a b : V) [Nonempty V] : Prop :=
  ∥a + b∥ = ∥a - b∥ → ⟪a, b⟫ = 0

-- Statement D
def statement_d (a b : V) : Prop :=
  parallel a b → ∃! k : ℝ, a = k • b

-- The final theorem to prove which of the statements are incorrect
theorem find_incorrect_statements (a b c : V) :
  ¬ statement_a a b c ∧ ¬ statement_b a b ∧ ¬ statement_d a b :=
begin
  -- Proof omitted
  sorry
end

end find_incorrect_statements_l610_610041


namespace portfolio_market_values_l610_610497

-- Definitions based on the conditions
def total_portfolio_value := 10000
def stock_A_percent := 0.40
def stock_B_percent := 0.30
def stock_C_percent := 0.30

-- Definition of individual market values
def stock_A_value := total_portfolio_value * stock_A_percent
def stock_B_value := total_portfolio_value * stock_B_percent
def stock_C_value := total_portfolio_value * stock_C_percent

-- Theorem to prove the market values
theorem portfolio_market_values :
  stock_A_value = 4000 ∧ stock_B_value = 3000 ∧ stock_C_value = 3000 :=
by
  sorry

end portfolio_market_values_l610_610497


namespace exists_x_in_0_pi_div_2_f_x_neg_l610_610953

def f (x : ℝ) := Real.sin x - x

theorem exists_x_in_0_pi_div_2_f_x_neg : ∃ x ∈ Set.Ioo 0 (Real.pi / 2), f x < 0 :=
by
  sorry

end exists_x_in_0_pi_div_2_f_x_neg_l610_610953


namespace proof_equivalent_l610_610934

variable {n : ℕ}
variables {a b c : ℝ}
noncomputable def s : ℕ → ℝ
| n := a^n + b^n + c^n

axiom s1_value : s 1 = 2
axiom s2_value : s 2 = 6
axiom s3_value : s 3 = 14

theorem proof_equivalent (h : n > 1) : |(s n)^2 - (s (n-1)) * (s (n+1))| = 8 := 
sorry

end proof_equivalent_l610_610934


namespace math_problem_l610_610328

def fractional_part (x : ℚ) : ℚ := x - (floor x)

noncomputable def problem_statement (p : ℕ) [fact (p.prime)] (hp4 : p % 4 = 1) : ℚ :=
  ∑ k in finset.range (p-1), fractional_part ((k^2 : ℚ) / p)

theorem math_problem (p : ℕ) [fact (p.prime)] (hp_odd : p % 2 = 1) (hp4 : p % 4 = 1) :
  problem_statement p hp4 = (p-1) / 2 := 
sorry

end math_problem_l610_610328


namespace vertex_y_coordinate_l610_610917

theorem vertex_y_coordinate (x : ℝ) : 
    let a := -6
    let b := 24
    let c := -7
    ∃ k : ℝ, k = 17 ∧ ∀ x : ℝ, (a * x^2 + b * x + c) = (a * (x - 2)^2 + k) := 
by 
  sorry

end vertex_y_coordinate_l610_610917


namespace f_even_f_periodic_l610_610336

variable {R : Type} [Real R]

-- Let f : ℝ → ℝ satisfy the given functional equation and condition.
def f (x : R) : R

def functional_eq (a b : R) : Prop :=
  f (a + b) + f (a - b) = 2 * f a * f b

def f_nonzero : Prop := f 0 ≠ 0

-- (1) Prove that f(x) is an even function.
theorem f_even (H : ∀ a b : R, functional_eq a b) (H₀ : f_nonzero) : ∀ x : R, f x = f (-x) :=
by
  sorry

-- (2) If there exists a positive number m such that f(m) = 0, find a value of T ≠ 0 such that f(x+T) = f(x).
theorem f_periodic (m_positive : ∃ m : R, m > 0 ∧ f m = 0) : ∃ T : R, T ≠ 0 ∧ ∀ x : R, f (x + T) = f x :=
by
  sorry

end f_even_f_periodic_l610_610336


namespace points_lie_on_circle_l610_610184

theorem points_lie_on_circle (s : ℝ) :
  ( (2 - s^2) / (2 + s^2) )^2 + ( 3 * s / (2 + s^2) )^2 = 1 :=
by sorry

end points_lie_on_circle_l610_610184


namespace win_sector_area_l610_610062

theorem win_sector_area (r : ℝ) (h1 : r = 8) (h2 : (1 / 4) = 1 / 4) : 
  ∃ (area : ℝ), area = 16 * Real.pi := 
by
  existsi (16 * Real.pi); exact sorry

end win_sector_area_l610_610062


namespace not_possible_20_odd_rows_15_odd_columns_possible_126_crosses_in_16x16_l610_610503

-- Define the general properties and initial conditions for the problems.
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Question a: Can there be exactly 20 odd rows and 15 odd columns in the table?
theorem not_possible_20_odd_rows_15_odd_columns (n : ℕ) (n_odd_rows : ℕ) (n_odd_columns : ℕ) (crosses : ℕ) (h_crosses_odd : is_odd crosses) 
  (h_odd_rows : n_odd_rows = 20) (h_odd_columns : n_odd_columns = 15) : 
  false := 
sorry

-- Question b: Can 126 crosses be arranged in a \(16 \times 16\) table so that all rows and columns are odd?
theorem possible_126_crosses_in_16x16 (crosses : ℕ) (n : ℕ) (h_crosses : crosses = 126) (h_table_size : n = 16) : 
  ∃ (table : matrix ℕ ℕ bool), 
    (∀ i : fin n, is_odd (count (λ j, table i j) (list.range n))) ∧
    (∀ j : fin n, is_odd (count (λ i, table i j) (list.range n))) :=
sorry

end not_possible_20_odd_rows_15_odd_columns_possible_126_crosses_in_16x16_l610_610503


namespace minkowski_inequality_l610_610475

variable {k : ℕ} {r : ℝ} (a b : Fin k → ℝ)

theorem minkowski_inequality (h : 1 < r) : 
  (∑ i, ((a i + b i)^r))^(1/r) ≤ (∑ i, (a i)^r)^(1/r) + (∑ i, (b i)^r)^(1/r) :=
sorry

end minkowski_inequality_l610_610475


namespace max_sailboat_speed_correct_l610_610390

noncomputable def max_sailboat_speed (C S ρ : ℝ) (v₀ : ℝ) : ℝ :=
  (v₀ / 3)

theorem max_sailboat_speed_correct :
  ∀ (C ρ : ℝ) (v₀ : ℝ), v₀ = 6 → S = 5 →
  max_sailboat_speed C S ρ v₀ = 2 :=
by
  intros C ρ v₀ hv₀ hS
  unfold max_sailboat_speed
  rw [hv₀, hS]
  norm_num

end max_sailboat_speed_correct_l610_610390


namespace trigonometric_identity_and_sine_of_Q_l610_610283

theorem trigonometric_identity_and_sine_of_Q
  (PQ PR : ℝ)
  (hPQ : PQ = 15)
  (hPR : PR = 9)
  (angle_R : ∠ PQR = π/2) :
  let QR := Real.sqrt (PQ^2 - PR^2)
  let sin_Q := QR / PQ
  let cos_Q := PR / PQ
  sin_Q = 4/5 ∧ sin_Q^2 + cos_Q^2 = 1 := 
by 
  have hQR : QR = Real.sqrt (PQ^2 - PR^2), by sorry
  have hsinQ : sin_Q = QR / PQ, by sorry
  have hcosQ : cos_Q = PR / PQ, by sorry
  sorry

end trigonometric_identity_and_sine_of_Q_l610_610283


namespace turbo_multiple_20_turbo_multiple_9_turbo_multiple_45_smallest_number_turbo_multiple_1110_l610_610418

-- Define the turbo multiple condition
def isTurboMultiple (n m : ℕ) : Prop :=
  m % n = 0 ∧ ∀ d ∈ toDigits 10 m, d = 0 ∨ d = 1

-- Problem a)
theorem turbo_multiple_20 :
  ∃ m, isTurboMultiple 20 m ∧ m = 100 :=
by sorry

-- Problem b)
theorem turbo_multiple_9 :
  ∃ m, isTurboMultiple 9 m ∧ m = 111111111 :=
by sorry

-- Problem c)
theorem turbo_multiple_45 :
  ∃ m, isTurboMultiple 45 m ∧ m = 1111111110 :=
by sorry

-- Problem d)
theorem smallest_number_turbo_multiple_1110 :
  ∃ n, isTurboMultiple n 1110 ∧ n = 6 :=
by sorry

end turbo_multiple_20_turbo_multiple_9_turbo_multiple_45_smallest_number_turbo_multiple_1110_l610_610418


namespace minimal_lambda_correct_l610_610622

noncomputable def minimal_lambda (n : ℕ) (h : n ≥ 2) : ℝ :=
  1 / 2 * (1 / (n - 1)) ^ (n - 1)

theorem minimal_lambda_correct (n : ℕ) (h : n ≥ 2) :
  ∃ λ, (λ = minimal_lambda n h) ∧ 
  (∀ (a : Fin n → ℝ) (b : Fin n → ℝ) 
      (hpos : ∀ i, 0 < a i) 
      (hbi : ∀ i, 0 ≤ b i ∧ b i ≤ 1 / 2) 
      (hsum_a : (∑ i, a i) = 1) 
      (hsum_b : (∑ i, b i) = 1),
      ((∏ i, a i) ≤ λ * (∑ i, a i * b i))) :=
sorry

end minimal_lambda_correct_l610_610622


namespace limonia_largest_none_providable_amount_l610_610710

def is_achievable (n : ℕ) (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), x = a * (6 * n + 1) + b * (6 * n + 4) + c * (6 * n + 7) + d * (6 * n + 10)

theorem limonia_largest_none_providable_amount (n : ℕ) : 
  ∃ s, ¬ is_achievable n s ∧ (∀ t, t > s → is_achievable n t) ∧ s = 12 * n^2 + 14 * n - 1 :=
by
  sorry

end limonia_largest_none_providable_amount_l610_610710


namespace problem_statement_l610_610739

noncomputable def f : ℝ → ℝ := λ x, (3 * x + 2) / (x + 1)
def S : set ℝ := {y | ∃ x : ℝ, 0 ≤ x ∧ y = f x}

theorem problem_statement : 3 ∉ S ∧ 2 ∈ S := 
by
  sorry

end problem_statement_l610_610739


namespace range_of_a_l610_610234

def f (x : ℝ) : ℝ :=
  if x < 2 then x^2 - 2 * x + 4
  else (3 / 2) * x + 1 / x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ |x + a|) ↔ -15 / 4 ≤ a ∧ a ≤ 3 / 2 :=
by
  sorry

end range_of_a_l610_610234


namespace gilbert_herb_plants_l610_610191

theorem gilbert_herb_plants
  (initial_basil : ℕ)
  (initial_parsley : ℕ)
  (initial_mint : ℕ)
  (extra_basil_mid_spring : ℕ)
  (mint_eaten_by_rabbit : ℕ) :
  initial_basil = 3 →
  initial_parsley = 1 →
  initial_mint = 2 →
  extra_basil_mid_spring = 1 →
  mint_eaten_by_rabbit = 2 →
  initial_basil + initial_parsley + initial_mint + extra_basil_mid_spring - mint_eaten_by_rabbit = 5 :=
by
  intros h_basil h_parsley h_mint h_extra h_eaten
  simp [h_basil, h_parsley, h_mint, h_extra, h_eaten]
  done
  sorry

end gilbert_herb_plants_l610_610191


namespace percentage_increase_is_50_l610_610060

-- Definition of the given values
def original_time : ℕ := 30
def new_time : ℕ := 45

-- Assertion stating that the percentage increase is 50%
theorem percentage_increase_is_50 :
  (new_time - original_time) * 100 / original_time = 50 := 
sorry

end percentage_increase_is_50_l610_610060


namespace incorrect_statements_of_problem_l610_610120

theorem incorrect_statements_of_problem :
  (¬ (∀ (P : Prop), (P ↔ P⁻¹) → ¬P)) →
  (∀ (A B C : ℝ), (A + B + C = 180) → (B = 60 → A + C = 120)) →
  (¬ (∀ (x y : ℝ), (x > 1 ∧ y > 2 ↔ x + y > 3 ∧ x * y > 2))) →
  (¬ (∀ (a b m : ℝ), (a < b ↔ am^2 < bm^2))) →
  (¬ (((¬ (P : Prop), (P ↔ P⁻¹) → ¬P)) ∧ (∀ (A B C : ℝ), (A + B + C = 180) → (B = 60 → A + C = 120)))) :=
begin
  sorry

end incorrect_statements_of_problem_l610_610120


namespace vector_combination_l610_610340

variables {Vect : Type*} [AddCommGroup Vect] [Module ℝ Vect]
variables (C D Q : Vect)
variables (t u : ℝ)

-- Define the ratio condition
def ratio_condition (CQ QD : ℝ): Prop := CQ / QD = 7 / 2

-- State the main theorem
theorem vector_combination
  (h_ratio : ratio_condition (CQ D Q) (D Q)) :
  Q = t • C + u • D ↔ t = 7 / 9 ∧ u = 2 / 9 :=
begin
  sorry
end

end vector_combination_l610_610340


namespace S_n_correct_l610_610338

noncomputable def S : ℕ → ℚ
| 0     := 0 -- This will serve as a base case to avoid issues with ℕ^*
| (n+1) := (n : ℚ + 1) / ((n : ℚ + 1) + 1)

variable {a : ℕ → ℚ}

def condition : ℕ → Prop
| 0 := true
| (n+1) := (S (n+1) - 1) ^ 2 = a (n+1) * S (n+1)

theorem S_n_correct : ∀ n : ℕ, n > 0 → S n = n / (n + 1) :=
by
  sorry

end S_n_correct_l610_610338


namespace max_chord_length_l610_610988

theorem max_chord_length (θ : ℝ) : 
  let C := λ θ : ℝ, 2 * (2 * real.sin θ - real.cos θ + 3) * x^2 - (8 * real.sin θ + real.cos θ + 1) * y = 0 in
  let L := y = 2 * x in
  ∃ θ : ℝ, ∃ x : ℝ, ∃ y : ℝ, C θ x y ∧ L x y ∧ (max_chord_length C L = 8 * real.sqrt 5) :=
sorry

end max_chord_length_l610_610988


namespace number_of_ways_to_split_cities_l610_610274

-- Definitions of the grid and capitals
def city := ℤ × ℤ
def is_gondor_capital (c : city) : Prop := c = (-1, 1)
def is_mordor_capital (c : city) : Prop := c = (1, -1)
def grid := finset.city([(i, j) | i in [-1, 0, 1], j in [-1, 0, 1]])

-- Conditions
def central_city := (0, 0)
def reachable_from (start : city) (target : city) (countries : city → Prop) : Prop :=
  ∃ path : list city, path.head = start ∧ path.last = target ∧
  (∀ c ∈ path, countries c) ∧ (∀ (c₁ c₂ : city), (c₁, c₂) ∈ list.zip path (list.tail path) → ((abs (c₁.fst - c₂.fst) = 1) ∨ (abs (c₁.snd - c₂.snd) = 1)))

-- Total number of ways to split the cities
theorem number_of_ways_to_split_cities : 
  (∃ countries : city → Prop, 
    (countries central_city ∧ 
    reachable_from (-1, 1) central_city countries ∧
    reachable_from (1, -1) central_city (λ c, ¬countries c))) ∨
  (∃ countries : city → Prop, 
    (¬countries central_city ∧ 
    reachable_from (1, -1) central_city countries ∧ 
    reachable_from (-1, 1) central_city (λ c, ¬countries c))) :=
by sorry

end number_of_ways_to_split_cities_l610_610274


namespace maximum_instantaneous_power_l610_610392

noncomputable def sailboat_speed : ℝ → ℝ → ℝ → ℝ 
  | C, S, v_0, v =>
  (C * S * ((v_0 - v) ^ 2)) / 2

theorem maximum_instantaneous_power (C ρ : ℝ)
  (S : ℝ := 5)
  (v_0 : ℝ := 6) :
  let v := (2 : ℝ) 
  (sailboat_speed C S v_0(v) * v)
  = (C * 5 * ρ / 2 -> v = 2) :=
by
  sorry

end maximum_instantaneous_power_l610_610392


namespace smallest_solutions_sum_l610_610168

noncomputable def find_smallest_solutions_sum : ℝ := 11 + 29 / 60

theorem smallest_solutions_sum :
  (∀ x : ℝ, (x > 0) → ((x - x.floor = 1 / (x.floor + 1)) ) -> 
    x = 1.5 ∨ x = 7/3 ∨ x = 13/4 ∨ x = 21/5) →
  (1.5 + 7/3 + 13/4 + 21/5 = 11 + 29/60) := by
  intros h
  sorry

end smallest_solutions_sum_l610_610168


namespace twenty_odd_rows_fifteen_odd_cols_impossible_sixteen_by_sixteen_with_126_crosses_possible_l610_610524

-- Conditions and helper definitions
def is_odd (n: ℕ) := n % 2 = 1
def count_odd_rows (table: ℕ × ℕ → bool) (n: ℕ) := 
  (List.range n).countp (λ i, is_odd ((List.range n).countp (λ j, table (i, j))))
  
def count_odd_columns (table: ℕ × ℕ → bool) (n: ℕ) := 
  (List.range n).countp (λ j, is_odd ((List.range n).countp (λ i, table (i, j))))

-- a) Proof problem statement
theorem twenty_odd_rows_fifteen_odd_cols_impossible (n: ℕ): 
  ∀ (table: ℕ × ℕ → bool), count_odd_rows table n = 20 → count_odd_columns table n = 15 → False := 
begin
  intros table h_rows h_cols,
  sorry
end

-- b) Proof problem statement
theorem sixteen_by_sixteen_with_126_crosses_possible :
  ∃ (table: ℕ × ℕ → bool), count_odd_rows table 16 = 16 ∧ count_odd_columns table 16 = 16 ∧ 
  (List.range 16).sum (λ i, (List.range 16).countp (λ j, table (i, j))) = 126 :=
begin
  sorry
end

end twenty_odd_rows_fifteen_odd_cols_impossible_sixteen_by_sixteen_with_126_crosses_possible_l610_610524


namespace greatest_four_digit_multiple_of_17_l610_610000

theorem greatest_four_digit_multiple_of_17 :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 17 = 0 ∧ ∀ m, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 17 = 0) → m ≤ n :=
  ⟨9996, by {
        split,
        { linarith },
        { split,
            { linarith },
            { split,
                { exact ModEq.rfl },
                { intros m hm hle,
                  have h : m ≤ 9999 := hm.2.2,
                  have : m = 17 * (m / 17) := (Nat.div_mul_cancel hm.2.1).symm,
                  have : 17 * (m / 17) ≤ 17 * 588 := Nat.mul_le_mul_left 17 (Nat.div_le_of_le_mul (by linarith)),
                  linarith,
                },
            },
        },
    },
  ⟩ sorry

end greatest_four_digit_multiple_of_17_l610_610000


namespace area_of_WIN_sector_correct_l610_610069

-- Define variables and constants
def radius : ℝ := 8
def probability_of_winning : ℝ := 1 / 4

-- Define the area of the circle
def area_of_circle (r : ℝ) := real.pi * r ^ 2

-- Define the area of the WIN sector given the area of the circle and the probability of winning
def area_of_WIN_sector (area_circle : ℝ) (prob_win : ℝ) := prob_win * area_circle

-- Theorem that the area of the WIN sector is 16π square centimeters
theorem area_of_WIN_sector_correct :
  area_of_WIN_sector (area_of_circle radius) probability_of_winning = 16 * real.pi :=
by 
-- Proof omitted
sorry

end area_of_WIN_sector_correct_l610_610069


namespace num_handshakes_l610_610537

-- Definition of the conditions
def num_teams : Nat := 4
def women_per_team : Nat := 2
def total_women : Nat := num_teams * women_per_team
def handshakes_per_woman : Nat := total_women -1 - women_per_team

-- Statement of the problem to prove
theorem num_handshakes (h: total_women = 8) : (total_women * handshakes_per_woman) / 2 = 24 := 
by
  -- Proof goes here
  sorry

end num_handshakes_l610_610537


namespace sum_S5_is_201_l610_610561

/-- Define the original sequence -/
def original_seq (n : ℕ) : ℝ := 3 * n + Real.log2 n

/-- Define the new sequence taking every 2^n-th term from the original sequence -/
def new_seq (n : ℕ) : ℝ := 3 * (2 ^ n) + n

/-- Define the sum of the first five terms of the new sequence -/
def sum_first_five_terms : ℝ := (new_seq 1) + (new_seq 2) + (new_seq 3) + (new_seq 4) + (new_seq 5)

/-- Prove that the sum of the first five terms of the new sequence is 201 -/
theorem sum_S5_is_201 : sum_first_five_terms = 201 := by
  sorry

end sum_S5_is_201_l610_610561


namespace log_base_five_of_625_l610_610573

theorem log_base_five_of_625 : ∃ x : ℤ, 5 ^ x = 625 ∧ x = 4 := by
  use 4
  split
  · norm_num
  · rfl

end log_base_five_of_625_l610_610573


namespace num_handshakes_l610_610536

-- Definition of the conditions
def num_teams : Nat := 4
def women_per_team : Nat := 2
def total_women : Nat := num_teams * women_per_team
def handshakes_per_woman : Nat := total_women -1 - women_per_team

-- Statement of the problem to prove
theorem num_handshakes (h: total_women = 8) : (total_women * handshakes_per_woman) / 2 = 24 := 
by
  -- Proof goes here
  sorry

end num_handshakes_l610_610536


namespace solution_in_interval_l610_610398

def f (x : ℝ) : ℝ := 2^x + x - 2

theorem solution_in_interval : 
  (f 0 < 0) → (f 1 > 0) → ∃ x, (0 < x ∧ x < 1 ∧ f x = 0) :=
by
  sorry

end solution_in_interval_l610_610398


namespace sum_abs_b_i_l610_610939

noncomputable def R (x : ℝ) := 1 - (1/2)*x + (1/4)*x^2

noncomputable def S (x : ℝ) := R(x) * R(x^2) * R(x^4) * R(x^6) * R(x^8)

theorem sum_abs_b_i : (∑ i in Finset.range 41, |coeff R i|) = 3087 / 1024 :=
by
  sorry

end sum_abs_b_i_l610_610939


namespace multiples_of_seven_l610_610859

theorem multiples_of_seven (a b : ℤ) (q : set ℤ) (h1 : a % 14 = 0) (h2 : b % 14 = 0) (h3 : q = set.Icc a b) (h4 : (q.filter (λ x, x % 14 = 0)).card = 12) : 
  (q.filter (λ x, x % 7 = 0)).card = 24 :=
sorry

end multiples_of_seven_l610_610859


namespace product_of_reals_condition_l610_610146

theorem product_of_reals_condition (x : ℝ) (h : x + 1/x = 3 * x) : 
  ∃ x1 x2 : ℝ, x1 + 1/x1 = 3 * x1 ∧ x2 + 1/x2 = 3 * x2 ∧ x1 * x2 = -1/2 := 
sorry

end product_of_reals_condition_l610_610146


namespace fair_die_probability_l610_610073

noncomputable def probability_greater_or_equal (sides : ℕ) : ℚ :=
  let total_outcomes := sides * sides
  let favorable_outcomes := ∑ i in finset.range (sides + 1), i
  (favorable_outcomes : ℚ) / total_outcomes

theorem fair_die_probability : probability_greater_or_equal 6 = 7 / 12 :=
  sorry

end fair_die_probability_l610_610073


namespace ellen_calories_l610_610150

theorem ellen_calories (b l a d : Nat) (hb : b = 353) (hl : l = 885) (ha : a = 130) (hd : d = 832) : 
  b + l + a + d = 2200 := by
  rw [hb, hl, ha, hd]
  norm_num
  sorry

end ellen_calories_l610_610150


namespace hitting_same_sector_more_likely_l610_610110

theorem hitting_same_sector_more_likely
  {n : ℕ} (p : Fin n → ℝ) 
  (h_pos : ∀ i, 0 ≤ p i) 
  (h_sum : ∑ i, p i = 1) :
  (∑ i, (p i) ^ 2) > (∑ i, (p i) * (p ((i + 1) % n))) :=
by
  sorry

end hitting_same_sector_more_likely_l610_610110


namespace sum_of_distances_on_ellipse_l610_610821

/-- The sum of distances from a point P on the given parametric curve to points A(-2,0) and B(2,0) --/
theorem sum_of_distances_on_ellipse (θ : ℝ) :
  let x := 4 * cos θ,
      y := 2 * sqrt 3 * sin θ,
      A := (-2, 0),
      B := (2, 0)
  in dist (x, y) A + dist (x, y) B = 8 :=
by
  sorry

end sum_of_distances_on_ellipse_l610_610821


namespace a_n_correct_b_n_correct_c_k_n_correct_sum_c_n_correct_l610_610982

-- Definition of {a_n} as an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, a (n + m) = a n + a m - a 0

-- Definitions based on problem conditions
def a (n : ℕ) : ℕ := n

def S (n : ℕ) : ℕ := 3^n - 1

def b (n : ℕ) : ℕ := if n = 0 then 0 else 2 * 3^(n-1)  -- b_0 is a dummy value for consistency

def c_k_n (k n : ℕ) : ℕ := ((2*k + 1 + n) * 2 * 3^(n-1))/(n+1)

def sum_c_n (n : ℕ) : ℕ :=
  (list.sum (Finset.univ.product Finset.univ).val.map (λ ⟨k, i⟩ : ℕ × ℕ, if i ≤ n then c_k_n k i else 0))

-- Proof statements
theorem a_n_correct : ∀ n, a n = n := by sorry

theorem b_n_correct : ∀ n, b n = if n = 0 then 0 else 2 * 3^(n-1) := by sorry

theorem c_k_n_correct : ∀ n k, c_k_n k n = ((2*k + 1 + n) * 2 * 3^(n-1))/(n+1) := by sorry

theorem sum_c_n_correct : ∀ n, sum_c_n n = 1 + (2 * n - 1) * 3^n := by sorry

end a_n_correct_b_n_correct_c_k_n_correct_sum_c_n_correct_l610_610982


namespace sum_of_roots_l610_610321

theorem sum_of_roots {x1 x2 x3 k m : ℝ} (h1 : x1 ≠ x2) (h2 : x2 ≠ x3) (h3 : x1 ≠ x3)
  (h4 : 2 * x1^3 - k * x1 = m) (h5 : 2 * x2^3 - k * x2 = m) (h6 : 2 * x3^3 - k * x3 = m) :
  x1 + x2 + x3 = 0 :=
sorry

end sum_of_roots_l610_610321


namespace angle_equality_l610_610358

variable (A B C D M N : Point)
variable [Square ABCD]
variable (hM_on_CD : M ∈ Segment C D)
variable (hN_on_AD : N ∈ Segment A D)
variable (h_dist_eq : distance C M + distance A N = distance B N)

theorem angle_equality : angle C B M = angle M B N := by
  sorry

end angle_equality_l610_610358


namespace height_of_smaller_cone_removed_l610_610882

noncomputable def frustum_area_lower_base : ℝ := 196 * Real.pi
noncomputable def frustum_area_upper_base : ℝ := 16 * Real.pi
def frustum_height : ℝ := 30

theorem height_of_smaller_cone_removed (r1 r2 H : ℝ)
  (h1 : r1 = Real.sqrt (frustum_area_lower_base / Real.pi))
  (h2 : r2 = Real.sqrt (frustum_area_upper_base / Real.pi))
  (h3 : r2 / r1 = 2 / 7)
  (h4 : frustum_height = (5 / 7) * H) :
  H - frustum_height = 12 :=
by 
  sorry

end height_of_smaller_cone_removed_l610_610882


namespace problem_a_impossible_problem_b_possible_l610_610518

-- Definitions based on the given conditions
def is_odd_row (table : ℕ → ℕ → bool) (n : ℕ) (r : ℕ) : Prop :=
  ∑ c in finset.range n, if table r c then 1 else 0 % 2 = 1

def is_odd_column (table : ℕ → ℕ → bool) (n : ℕ) (c : ℕ) : Prop :=
  ∑ r in finset.range n, if table r c then 1 else 0 % 2 = 1

-- Problem(a): No existence of 20 odd rows and 15 odd columns in any square table
theorem problem_a_impossible (table : ℕ → ℕ → bool) (n : ℕ) :
  (∃r_set c_set, r_set.card = 20 ∧ c_set.card = 15 ∧
  ∀ r ∈ r_set, is_odd_row table n r ∧ ∀ c ∈ c_set, is_odd_column table n c) → false :=
sorry

-- Problem(b): Existence of a 16 x 16 table with 126 crosses where all rows and columns are odd
theorem problem_b_possible : 
  ∃ (table : ℕ → ℕ → bool), 
  (∑ r in finset.range 16, ∑ c in finset.range 16, if table r c then 1 else 0) = 126 ∧
  (∀ r, is_odd_row table 16 r) ∧
  (∀ c, is_odd_column table 16 c) :=
sorry

end problem_a_impossible_problem_b_possible_l610_610518


namespace factorization_l610_610924

theorem factorization (x : ℝ) : 2 * x^3 - 4 * x^2 + 2 * x = 2 * x * (x - 1)^2 :=
sorry

end factorization_l610_610924


namespace unique_set_of_consecutive_integers_l610_610249

theorem unique_set_of_consecutive_integers (a b c : ℕ) : 
  (a + b + c = 36) ∧ (b = a + 1) ∧ (c = a + 2) → 
  ∃! a : ℕ, (a = 11 ∧ b = 12 ∧ c = 13) := 
sorry

end unique_set_of_consecutive_integers_l610_610249


namespace Jack_total_money_in_dollars_l610_610725

variable (Jack_dollars : ℕ) (Jack_euros : ℕ) (euro_to_dollar : ℕ)

noncomputable def total_dollars (Jack_dollars : ℕ) (Jack_euros : ℕ) (euro_to_dollar : ℕ) : ℕ :=
  Jack_dollars + Jack_euros * euro_to_dollar

theorem Jack_total_money_in_dollars : 
  Jack_dollars = 45 → 
  Jack_euros = 36 → 
  euro_to_dollar = 2 → 
  total_dollars 45 36 2 = 117 :=
by
  intro h1 h2 h3
  unfold total_dollars
  rw [h1, h2, h3]
  -- skipping the actual proof
  sorry

end Jack_total_money_in_dollars_l610_610725


namespace num_x_f_f_x_eq_7_l610_610263

def f (x : ℝ) : ℝ :=
  if x ≥ 2 then x^2 - 2 else x + 4

theorem num_x_f_f_x_eq_7 : set_of (λ x : ℝ, f (f x) = 7).card = 2 := by
  sorry

end num_x_f_f_x_eq_7_l610_610263


namespace ellipse_equation_from_hyperbola_l610_610660

theorem ellipse_equation_from_hyperbola :
  let H := ∀ x y : ℝ, (y^2 / 12) - (x^2 / 4) = 1 in
  let f1 := (0, 4) in let f2 := (0, -4) in
  let v1 := (0, 2 * Real.sqrt 3) in let v2 := (0, -2 * Real.sqrt 3) in
  let E := ∀ x y : ℝ, (y^2 / 16) + (x^2 / 4) = 1 in
  E ↔ (H ∧ f1 ∧ f2 ∧ v1 ∧ v2) :=
by
  sorry

end ellipse_equation_from_hyperbola_l610_610660


namespace trig_expression_value_l610_610946

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 3) :
  2 * Real.sin α ^ 2 + 4 * Real.sin α * Real.cos α - 9 * Real.cos α ^ 2 = 21 / 10 :=
by
  sorry

end trig_expression_value_l610_610946


namespace equilateral_triangle_roots_l610_610322

theorem equilateral_triangle_roots (p q : ℂ) (z1 z2 : ℂ) (h1 : z2 = Complex.exp (2 * Real.pi * Complex.I / 3) * z1)
  (h2 : 0 + p * z1 + q = 0) (h3 : p = -z1 - z2) (h4 : q = z1 * z2) : (p^2 / q) = 1 :=
by
  sorry

end equilateral_triangle_roots_l610_610322


namespace sin_alpha_neg_point_two_l610_610255

theorem sin_alpha_neg_point_two (a : ℝ) (h : Real.sin (Real.pi + a) = 0.2) : Real.sin a = -0.2 := 
by
  sorry

end sin_alpha_neg_point_two_l610_610255


namespace intersection_A_B_l610_610639

def set_A : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}

def set_B : Set ℝ := {x | x^2 - 3 * x < 0}

theorem intersection_A_B :
  {x : ℤ | x ∈ set_A ∧ x ∈ set_B} = {1, 2} :=
sorry

end intersection_A_B_l610_610639


namespace greatest_four_digit_multiple_of_17_l610_610008

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_multiple_of (n d : ℕ) : Prop :=
  ∃ k : ℕ, n = k * d

theorem greatest_four_digit_multiple_of_17 : ∃ n, is_four_digit n ∧ is_multiple_of n 17 ∧
  ∀ m, is_four_digit m → is_multiple_of m 17 → m ≤ n :=
  by
  existsi 9996
  sorry

end greatest_four_digit_multiple_of_17_l610_610008


namespace greatest_four_digit_multiple_of_17_l610_610006

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_multiple_of (n d : ℕ) : Prop :=
  ∃ k : ℕ, n = k * d

theorem greatest_four_digit_multiple_of_17 : ∃ n, is_four_digit n ∧ is_multiple_of n 17 ∧
  ∀ m, is_four_digit m → is_multiple_of m 17 → m ≤ n :=
  by
  existsi 9996
  sorry

end greatest_four_digit_multiple_of_17_l610_610006


namespace solve_for_y_l610_610369

theorem solve_for_y (y : ℝ) : 16^(y + 3) = 64^(2*y - 5) → y = 21 / 4 :=
by
  intro h
  sorry  -- Proof goes here

end solve_for_y_l610_610369


namespace least_k_divisible_by_2160_l610_610264

theorem least_k_divisible_by_2160 (k : ℤ) : k^3 ∣ 2160 → k ≥ 60 := by
  sorry

end least_k_divisible_by_2160_l610_610264


namespace find_a_l610_610659

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(2 * x - 1) + 3 else 1 - Real.log 2 x

theorem find_a (a : ℝ) (h : f a = 4) : a = 1/8 :=
by
  sorry

end find_a_l610_610659


namespace find_a_monotonicity_find_m_l610_610977

-- Definition of f(x)
def f (x : ℝ) (a : ℝ) : ℝ := -1 / 2 + a / (3^x + 1)

-- Assume f(x) is an odd function
axiom h_odd : ∀ x : ℝ, f x 1 = -f (-x) 1

-- Prove the value of a
theorem find_a : (f 0 a = 0) → a = 1 := by
  intro h
  sorry

-- Prove that f(x) is decreasing
theorem monotonicity : (∀ x1 x2 : ℝ, x1 < x2 → f x1 1 > f x2 1) := by
  intro x1 x2 h
  sorry

-- Prove the range of m
theorem find_m (t : ℝ) (ht : t ∈ Ioo (1 : ℝ) 2) : (∀ t ∈ Ioo (1 : ℝ) 2, f (-2 * t^2 + t + 1) 1 + f (t^2 - 2 * m * t) 1 ≤ 0) → m ≤ 1 / 2 := by
  intro h
  sorry

end find_a_monotonicity_find_m_l610_610977


namespace solve_real_equation_l610_610579

theorem solve_real_equation (x : ℝ) (h : x^4 + (3 - x)^4 = 82) : x = 2.5 ∨ x = 0.5 :=
sorry

end solve_real_equation_l610_610579


namespace modulus_of_z_is_sqrt5_l610_610335

-- Define the complex number z
def z : ℂ := (-1 + 2 * complex.I) / complex.I

-- Prove the modulus (absolute value) of z is sqrt(5)
theorem modulus_of_z_is_sqrt5 : complex.abs z = real.sqrt 5 := by sorry

end modulus_of_z_is_sqrt5_l610_610335


namespace solution_l610_610716

variable {a : ℕ → ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def condition1 (a : ℕ → ℝ) [is_geometric_sequence a] : Prop :=
  a 2 * a 6 = 16

def condition2 (a : ℕ → ℝ) [is_geometric_sequence a] : Prop :=
  a 4 + a 8 = 8

-- Statement to prove
theorem solution (a : ℕ → ℝ) [is_geometric_sequence a] :
  condition1 a → condition2 a → (a 20 / a 10 = 1) :=
by sorry

end solution_l610_610716


namespace females_wearing_glasses_in_town_l610_610480

theorem females_wearing_glasses_in_town : 
  (total_population females males : ℕ) 
  (wear_glasses_percentage : ℚ) 
  (h1 : total_population = 5000) 
  (h2 : males = 2000) 
  (h3 : wear_glasses_percentage = 0.30) 
  : 
  ∃ (females_wear_glasses : ℕ), females_wear_glasses = 900 :=
by
  have total_females : ℕ := total_population - males
  have females_wear_glasses : ℚ := total_females * wear_glasses_percentage
  use females_wear_glasses.toNat
  sorry

end females_wearing_glasses_in_town_l610_610480


namespace monotonicity_of_f_range_of_a_l610_610992

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x) - a * x

theorem monotonicity_of_f (a : ℝ) (ha : a ≠ 0) :
  (∀ x < 0, f a x ≥ f a (x + 1)) ∧ (∀ x > 0, f a x ≤ f a (x + 1)) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, f a x ≥ Real.sin x - Real.cos x + 2 - a * x) ↔ a ∈ Set.Ici 1 :=
sorry

end monotonicity_of_f_range_of_a_l610_610992


namespace mark_garden_total_flowers_l610_610351

theorem mark_garden_total_flowers :
  let yellow := 10
  let purple := yellow + (80 / 100) * yellow
  let total_yellow_purple := yellow + purple
  let green := (25 / 100) * total_yellow_purple
  total_yellow_purple + green = 35 :=
by
  let yellow := 10
  let purple := yellow + (80 / 100) * yellow
  let total_yellow_purple := yellow + purple
  let green := (25 / 100) * total_yellow_purple
  simp [yellow, purple, total_yellow_purple, green]
  sorry

end mark_garden_total_flowers_l610_610351


namespace triangles_congruent_and_parallel_l610_610241

variables {P : Type*} [AffineSpace P ℝ]

-- Define vertices as points in the affine space
variables (A1 A2 A3 B1 B2 B3 C1 C2 C3 : P)

-- Define midpoints
def midpoint (p1 p2 : P) : P := AffineMap.lineMap p1 p2 (1/2)

noncomputable def A13 := midpoint A1 A3
noncomputable def A23 := midpoint A2 A3
noncomputable def B13 := midpoint B1 B3
noncomputable def B23 := midpoint B2 B3
noncomputable def C13 := midpoint C1 C3
noncomputable def C23 := midpoint C2 C3

-- Specify the given conditions as hypotheses
variables (h1 : congruent A1 B1 C1 A2 B2 C2)
variables (h2 : parallel (A1, A2) (B1, B2) ∧ parallel (B1, B2) (C1, C2))

-- Statement to prove
theorem triangles_congruent_and_parallel :
  triangle_congruent A13 B13 C13 A23 B23 C23 ∧ 
  parallel (A13, A23) (B13, B23) ∧ 
  parallel (B13, B23) (C13, C23) :=
sorry

end triangles_congruent_and_parallel_l610_610241


namespace bunyakovsky_hit_same_sector_l610_610115

variable {n : ℕ} (p : Fin n → ℝ)

theorem bunyakovsky_hit_same_sector (h_sum : ∑ i in Finset.univ, p i = 1) :
  (∑ i in Finset.univ, (p i)^2) >
  (∑ i in Finset.univ, (p i) * (p (Fin.rotate 1 i))) := 
sorry

end bunyakovsky_hit_same_sector_l610_610115


namespace coeff_x60_is_11_l610_610927

-- Define the polynomial
noncomputable def poly : Polynomial ℝ :=
  (X - 1) * (X^2 - 2) * (X^3 - 3) * (X^4 - 4) * (X^5 - 5) *
  (X^6 - 6) * (X^7 - 7) * (X^8 - 8) * (X^9 - 9) * (X^10 - 10) * (X^11 - 11)

-- Define the theorem we want to prove
theorem coeff_x60_is_11 : poly.coeff 60 = 11 :=
sorry

end coeff_x60_is_11_l610_610927


namespace coefficient_x4_l610_610162

-- Define the polynomial expression
def poly := 5 * (x^4 - 2 * x^3 + x^2) - 3 * (x^2 - x + 1) + 4 * (x^6 - 3 * x^4 + x^3)

-- The theorem we want to prove
theorem coefficient_x4 : (coeff (expand poly) 4) = -7 := 
sorry

end coefficient_x4_l610_610162


namespace more_likely_same_sector_l610_610108

theorem more_likely_same_sector 
  (p : ℕ → ℝ) 
  (n : ℕ) 
  (hprob_sum_one : ∑ i in Finset.range n, p i = 1) 
  (hprob_nonneg : ∀ i, 0 ≤ p i) : 
  ∑ i in Finset.range n, (p i) ^ 2 
  > ∑ i in Finset.range n, (p i) * (p ((i + 1) % n)) :=
by
  sorry

end more_likely_same_sector_l610_610108


namespace max_value_and_x_values_angle_C_in_triangle_l610_610994

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x + 2 * cos x ^ 2 - 1

theorem max_value_and_x_values :
  (∀ x : ℝ, f x ≤ 2) ∧ (∃ (k : ℤ), ∀ x : ℝ, f x = 2 ↔ x = ↑k * π + π / 6) :=
sorry

theorem angle_C_in_triangle 
  (a b : ℝ) (A : ℝ) (fA : ℝ) (h_a : a = 1) (h_b : b = sqrt 2) (h_fA : f A = 2) :
  C = 7 * π / 12 ∨ C = π / 12 :=
sorry

end max_value_and_x_values_angle_C_in_triangle_l610_610994


namespace length_of_CP_l610_610374

theorem length_of_CP 
  (right_triangle : Type)
  [metric_space right_triangle] 
  {A B C : right_triangle} 
  (h_angle_right : ∃ (B : right_triangle), abs (angle B) = pi / 2)
  (h_AC : dist A C = sqrt 61)
  (h_AB : dist A B = 5)
  (circle_center : metric_space.point right_triangle)
  (h_tangent : is_tangent circle_center AC ∧ is_tangent circle_center BC) 
  : dist C P = 6 := 
sorry

end length_of_CP_l610_610374


namespace trucks_after_redistribution_l610_610832

/-- Problem Statement:
   Prove that the total number of trucks after redistribution is 10.
-/

theorem trucks_after_redistribution
    (num_trucks1 : ℕ)
    (boxes_per_truck1 : ℕ)
    (num_trucks2 : ℕ)
    (boxes_per_truck2 : ℕ)
    (containers_per_box : ℕ)
    (containers_per_truck_after : ℕ)
    (h1 : num_trucks1 = 7)
    (h2 : boxes_per_truck1 = 20)
    (h3 : num_trucks2 = 5)
    (h4 : boxes_per_truck2 = 12)
    (h5 : containers_per_box = 8)
    (h6 : containers_per_truck_after = 160) :
  (num_trucks1 * boxes_per_truck1 + num_trucks2 * boxes_per_truck2) * containers_per_box / containers_per_truck_after = 10 := by
  sorry

end trucks_after_redistribution_l610_610832


namespace proof_problem_l610_610960

variable {ℕ : Type} [linear_ordered_field ℕ]

-- Define the arithmetic sequence a_n
variable {a : ℕ → ℕ}

-- Define the geometric sequence b_n
variable {b : ℕ → ℕ}

-- Define the conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, a(m + 1) - a(m) = a(n + 1) - a(n)

def is_geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, b(m + 1) / b(m) = b(n + 1) / b(n)

theorem proof_problem :
  is_arithmetic_sequence a →
  is_geometric_sequence b →
  2 * a(2) + 2 * a(12) = a(7)^2 →
  b(7) = a(7) →
  b(5) * b(9) = 16 :=
by 
  sorry

end proof_problem_l610_610960


namespace remainder_when_xyz_divided_by_9_is_0_l610_610672

theorem remainder_when_xyz_divided_by_9_is_0
  (x y z : ℕ)
  (hx : x < 9)
  (hy : y < 9)
  (hz : z < 9)
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 9])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 9])
  (h3 : 2 * x + y + 3 * z ≡ 5 [MOD 9]) :
  (x * y * z) % 9 = 0 := by
  sorry

end remainder_when_xyz_divided_by_9_is_0_l610_610672


namespace odd_rows_cols_impossible_arrange_crosses_16x16_l610_610514

-- Define the conditions for part (a)
def square (α : Type*) := α × α
def is_odd_row (table : square nat → bool) (n : nat) :=
  ∃ (i : fin n), ∑ j in finset.range n, table (i, j) = 1
def is_odd_col (table : square nat → bool) (n : nat) :=
  ∃ (j : fin n), ∑ i in finset.range n, table (i, j) = 1

-- Part (a) statement
theorem odd_rows_cols_impossible (table : square nat → bool) (n : nat) :
  n = 16 ∧ (∃ (r : ℕ), r = 20) ∧ (∃ (c : ℕ), c = 15) → ¬(is_odd_row table n ∧ is_odd_col table n) :=
begin 
  -- Sorry placeholder for the proof
  sorry,
end

-- Define the conditions for part (b)
def odd_placement_possible (table : square nat → bool) :=
  ∃ (n : nat), n = 16 ∧ (∑ i in finset.range 16, ∑ j in finset.range 16, table (i, j) = 126) ∧ 
  (∀ i, is_odd_row table 16) ∧ (∀ j, is_odd_col table 16)

-- Part (b) statement
theorem arrange_crosses_16x16 (table : square nat → bool) :
  odd_placement_possible table :=
begin 
  -- Sorry placeholder for the proof
  sorry,
end

end odd_rows_cols_impossible_arrange_crosses_16x16_l610_610514


namespace range_of_a_l610_610239

theorem range_of_a (a : ℝ) (h_decreasing : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → (x:ℝ)^(-1/2) > (y:ℝ)^(-1/2)) 
  (h1 : 0 ≤ a - 1) (h2 : 0 ≤ 8 - 2 * a) : 3 < a ∧ a < 4 :=
by
  have h_decreasing' := h_decreasing (a - 1) (8 - 2 * a)
  sorry

end range_of_a_l610_610239


namespace number_of_people_in_group_l610_610289

-- Defining the conditions as constants and assumptions
variables (x : ℕ) -- x represents the number of people in the group
constant total_cost_sheep : ℕ -- total cost of the sheep

-- Assuming the given conditions
axiom h1 : 5 * x + 45 = total_cost_sheep
axiom h2 : 7 * x + 3 = total_cost_sheep

-- The theorem to prove that x equals 21.
theorem number_of_people_in_group : x = 21 :=
by
  -- Below, we would normally proceed to prove the statement.
  sorry

end number_of_people_in_group_l610_610289


namespace cos_alpha_value_l610_610616

theorem cos_alpha_value (α : ℝ) 
  (h1 : sin (α + π / 4) = 12 / 13) 
  (h2 : π / 4 < α ∧ α < 3 * π / 4) : 
  cos α = 7 * Real.sqrt 2 / 26 := 
by 
  sorry

end cos_alpha_value_l610_610616


namespace hexagon_ratio_l610_610411

noncomputable def regular_hexagon (ABCDEF : Type) := 
  ∃ AC CE : ABCDEF,
    ∃ M N : ABCDEF,
      ∃ r : ℝ,
        -- Regular hexagon condition and diagonal definitions
        ABCDEF.is_regular_hexagon ∧
        is_diagonal AC ABCDEF ∧
        is_diagonal CE ABCDEF ∧
        -- Ratio condition definition
        AM / AC = r ∧
        CN / CE = r ∧
        -- Collinearity condition
        are_collinear B M N

theorem hexagon_ratio (ABCDEF : Type) (B M N : ABCDEF) (AC CE : ABCDEF) (r : ℝ) :
  regular_hexagon ABCDEF →
  -- condition definitions
  (AM / AC = r) →
  (CN / CE = r) →
  are_collinear B M N →
  r = 1 / real.sqrt 3 := sorry

end hexagon_ratio_l610_610411


namespace gary_egg_collection_l610_610607

-- Conditions
def initial_chickens : ℕ := 4
def multiplier : ℕ := 8
def eggs_per_chicken_per_day : ℕ := 6
def days_in_week : ℕ := 7

-- Definitions derived from conditions
def current_chickens : ℕ := initial_chickens * multiplier
def eggs_per_day : ℕ := current_chickens * eggs_per_chicken_per_day
def eggs_per_week : ℕ := eggs_per_day * days_in_week

-- Proof statement
theorem gary_egg_collection : eggs_per_week = 1344 := by
  unfold eggs_per_week
  unfold eggs_per_day
  unfold current_chickens
  sorry

end gary_egg_collection_l610_610607


namespace sum_of_axes_l610_610305

theorem sum_of_axes (r_cyl r_sph d : ℝ) (h_r_cyl : r_cyl = 6) (h_r_sph : r_sph = 6) (h_d : d = 13) :
  let a := d / 2 in
  let b := r_cyl in
  2 * a + 2 * b = 25 :=
by
  sorry

end sum_of_axes_l610_610305


namespace sum_y_coordinates_of_other_vertices_l610_610250

theorem sum_y_coordinates_of_other_vertices (x1 y1 x2 y2 : ℤ) 
  (h1 : (x1, y1) = (2, 10)) (h2 : (x2, y2) = (-6, -6)) :
  (∃ y3 y4 : ℤ, (4 : ℤ) = y3 + y4) :=
by
  sorry

end sum_y_coordinates_of_other_vertices_l610_610250


namespace group_value_21_le_a_lt_41_l610_610036

theorem group_value_21_le_a_lt_41 : 
  (∀ a: ℤ, 21 ≤ a ∧ a < 41 → (21 + 41) / 2 = 31) :=
by 
  sorry

end group_value_21_le_a_lt_41_l610_610036


namespace num_of_valid_As_l610_610183

theorem num_of_valid_As : ∃! A : ℕ, (A ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ (41 % A = 0) ∧ (∃ n : ℕ, 1 * 100 + A * 10 + 8 = 8 * n) :=
sorry

end num_of_valid_As_l610_610183


namespace identify_element_l610_610875

theorem identify_element (molecular_weight : ℝ) (atomic_weight_Ca : ℝ) (atomic_weight_H : ℝ) :
  molecular_weight = 42 →
  atomic_weight_Ca = 40.08 →
  atomic_weight_H = 1.008 →
  ∃ element : ℝ, molecular_weight - atomic_weight_Ca ≈ atomic_weight_H :=
by
  intros h₁ h₂ h₃
  use atomic_weight_H
  calc
    42 - 40.08 = 1.92 : by norm_num
    1.92 ≈ 1.008 : sorry

end identify_element_l610_610875


namespace value_of_complex_fraction_l610_610741

theorem value_of_complex_fraction (i : ℂ) (h : i ^ 2 = -1) : ((1 - i) / (1 + i)) ^ 2 = -1 :=
by
  sorry

end value_of_complex_fraction_l610_610741


namespace Odell_Kershaw_meetings_l610_610769

noncomputable def circumference (radius : ℝ) : ℝ :=
  2 * Real.pi * radius

noncomputable def angular_speed (speed : ℝ) (radius : ℝ) : ℝ :=
  (speed / circumference radius) * 2 * Real.pi

noncomputable def time_to_meet_once (ω_O ω_K : ℝ) : ℝ :=
  2 * Real.pi / (ω_O + ω_K)

noncomputable def total_meetings (total_time delay : ℝ) (meet_time : ℝ) : ℕ :=
  ⌊(total_time - delay) / meet_time⌋

theorem Odell_Kershaw_meetings :
  let radius_O := 55
      radius_K := 65
      speed_O  := 200
      speed_K  := 260
      total_time := 40
      delay := 5
      ω_O    := angular_speed speed_O radius_O
      ω_K    := angular_speed speed_K radius_K
      meet_time := time_to_meet_once ω_O ω_K
  in total_meetings total_time delay meet_time = 21 :=
by
  -- Definitions and intermediate steps are based on the problem conditions
  let radius_O := 55
  let radius_K := 65
  let speed_O  := 200
  let speed_K  := 260
  let total_time := 40
  let delay := 5
  
  -- Calculating angular speeds
  let ω_O := angular_speed speed_O radius_O
  let ω_K := angular_speed speed_K radius_K

  -- Calculating time to meet once
  let meet_time := time_to_meet_once ω_O ω_K

  -- Calculating total meetings
  have result : total_meetings total_time delay meet_time = 21 := 
    by sorry
  
  exact result

end Odell_Kershaw_meetings_l610_610769


namespace area_of_WIN_sector_correct_l610_610068

-- Define variables and constants
def radius : ℝ := 8
def probability_of_winning : ℝ := 1 / 4

-- Define the area of the circle
def area_of_circle (r : ℝ) := real.pi * r ^ 2

-- Define the area of the WIN sector given the area of the circle and the probability of winning
def area_of_WIN_sector (area_circle : ℝ) (prob_win : ℝ) := prob_win * area_circle

-- Theorem that the area of the WIN sector is 16π square centimeters
theorem area_of_WIN_sector_correct :
  area_of_WIN_sector (area_of_circle radius) probability_of_winning = 16 * real.pi :=
by 
-- Proof omitted
sorry

end area_of_WIN_sector_correct_l610_610068


namespace shape_of_constant_phi_is_cone_l610_610179

def sphericalShape (c : ℝ) : Type :=
  {x y z : ℝ // y = c}

theorem shape_of_constant_phi_is_cone (c : ℝ) : sphericalShape c := by
  sorry

end shape_of_constant_phi_is_cone_l610_610179


namespace f_increasing_exists_a_odd_function_l610_610196

section Problem

variable (a : ℝ) (f : ℝ → ℝ) (x : ℝ) (x1 x2 : ℝ)

-- Define the function f(x)
def f (x : ℝ) : ℝ := a - 2 / (3^x + 1)

-- Theorem 1: Prove that f(x) is an increasing function on ℝ.
theorem f_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 := by sorry

-- Theorem 2: Show there exists a real number a such that f(x) is an odd function and find that a.
theorem exists_a_odd_function : ∃ a : ℝ, ∀ x : ℝ, f a (-x) = -f a x := by
  use 1
  sorry

end Problem

end f_increasing_exists_a_odd_function_l610_610196


namespace find_constant_a_l610_610224

theorem find_constant_a (a : ℝ) : 
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 2 ∧ ax^2 + 2 * a * x + 1 = 9) → (a = -8 ∨ a = 1) :=
by
  sorry

end find_constant_a_l610_610224


namespace find_a_find_m_l610_610978

-- Define the function f(x)
def f (a x : ℝ) : ℝ := log (x + a) / log 2 - log (x - 1) / log 2

-- Property of odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

-- The first problem converted to Lean statement
theorem find_a (a : ℝ) (h_pos : a > 0) (h_odd : is_odd (f a)) : a = 1 := 
sorry

-- Definition for the second problem
def g (m x : ℝ) : ℝ := log m / log 2 - log (x - 1) / log 2

-- The second problem converted to Lean statement
theorem find_m (m : ℝ) (h_pos : 0 < m) (h_range : ∀ x, 1 < x ∧ x ≤ 4 → f 1 x > g m x) : m ≤ 2 := 
sorry

end find_a_find_m_l610_610978


namespace intersection_of_planes_intersects_skew_lines_l610_610971

variables (m n l : set (ℝ × ℝ × ℝ))
variables (α β : set (ℝ × ℝ × ℝ))
variables [skew_lines : skew m n]

-- Definitions for the conditions
def is_skew (m n : set (ℝ × ℝ × ℝ)) : Prop := 
  ¬ ∃ p, p ∈ m ∧ p ∈ n

def in_plane (l : set (ℝ × ℝ × ℝ)) (α : set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ p ∈ l, p ∈ α

def plane_intersection (α β : set (ℝ × ℝ × ℝ)) (l : set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ p, p ∈ l ↔ p ∈ α ∧ p ∈ β

-- Definitions for our specific lines and planes
def m_in_α : Prop := in_plane m α
def n_in_β : Prop := in_plane n β
def α_β_intersect_l : Prop := plane_intersection α β l

-- Main theorem statement
theorem intersection_of_planes_intersects_skew_lines :
  is_skew m n → m_in_α → n_in_β → α_β_intersect_l → ∃ p, p ∈ l ∧ (p ∈ m ∨ p ∈ n) :=
by
  sorry

end intersection_of_planes_intersects_skew_lines_l610_610971


namespace part1_part2_l610_610665

noncomputable def A : set ℝ := {x | x^2 + 4 * x = 0}

noncomputable def B (a : ℝ) : set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem part1 (a : ℝ) : A ∪ (B a) = B a ↔ a = 1 := 
by 
  sorry

theorem part2 (a : ℝ) : A ∩ (B a) = B a ↔ a ≤ -1 ∨ a = 1 := 
by 
  sorry

end part1_part2_l610_610665


namespace find_unit_prices_and_evaluate_discount_schemes_l610_610550

theorem find_unit_prices_and_evaluate_discount_schemes :
  ∃ (x y : ℝ),
    40 * x + 100 * y = 280 ∧
    30 * x + 200 * y = 260 ∧
    x = 6 ∧
    y = 0.4 ∧
    (∀ m : ℝ, m > 200 → 
      (50 * 6 + 0.4 * (m - 50) < 50 * 6 + 0.4 * 200 + 0.4 * 0.8 * (m - 200) ↔ m < 450) ∧
      (50 * 6 + 0.4 * (m - 50) = 50 * 6 + 0.4 * 200 + 0.4 * 0.8 * (m - 200) ↔ m = 450) ∧
      (50 * 6 + 0.4 * (m - 50) > 50 * 6 + 0.4 * 200 + 0.4 * 0.8 * (m - 200) ↔ m > 450)) :=
sorry

end find_unit_prices_and_evaluate_discount_schemes_l610_610550


namespace total_number_of_seeds_l610_610842

section WatermelonSeeds

def seeds_in_first_watermelon := 40 * (20 + 15 + 10)
def seeds_in_second_watermelon := 30 * (25 + 20 + 15)
def seeds_in_third_watermelon := 50 * (15 + 10 + 5 + 5)

def total_seeds := seeds_in_first_watermelon + seeds_in_second_watermelon + seeds_in_third_watermelon

theorem total_number_of_seeds : total_seeds = 5350 := 
by
  unfold seeds_in_first_watermelon seeds_in_second_watermelon seeds_in_third_watermelon total_seeds
  sorry

end WatermelonSeeds

end total_number_of_seeds_l610_610842


namespace problem_equivalence_l610_610202

-- Define the events and conditions
def bag := {red: ℕ, white: ℕ, black: ℕ}
def initial_bag : bag := {red := 3, white := 2, black := 1}

def event_at_least_one_white (draw: list bag) : Prop :=
  list.any draw (λ b, b.white > 0)

def event_one_red_one_black (draw: list bag) : Prop :=
  list.count ((λ b, b.red > 0) ∧ (b.black > 0)) draw = 1

def mutually_exclusive (A B : Prop) : Prop :=
  ¬(A ∧ B)

def not_complementary (A B : Prop) : Prop :=
  ∃ s, ¬(s = A ∨ s = B)

-- State the formalized equivalent problem
theorem problem_equivalence :
  mutually_exclusive
    (event_at_least_one_white [initial_bag])
    (event_one_red_one_black [initial_bag])
  ∧ not_complementary
    (event_at_least_one_white [initial_bag])
    (event_one_red_one_black [initial_bag]) :=
sorry  -- Proof is not required.

end problem_equivalence_l610_610202


namespace percentage_relation_l610_610668

theorem percentage_relation (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 12) : y = 60 := by
  sorry

end percentage_relation_l610_610668


namespace triangle_areas_inequality_l610_610201

theorem triangle_areas_inequality 
  (A B C D E F : Type) 
  (x y z : ℝ) 
  (S_ABC S_BDF S_CEF : ℝ)
  (H1 : D ∈ line_segment AB)
  (H2 : E ∈ line_segment AC)
  (H3 : F ∈ line_segment DE)
  (H4 : AD / AB = x)
  (H5 : AE / AC = y)
  (H6 : DF / DE = z)
  (H7 : S_BDF = (1 - x) * y * z * S_ABC)
  (H8 : S_CEF = x * (1 - y) * (1 - z) * S_ABC)
  :
  S_BDF = (1 - x) * y * z * S_ABC ∧ 
  S_CEF = x * (1 - y) * (1 - z) * S_ABC ∧ 
  (real.cbrt S_BDF + real.cbrt S_CEF <= real.cbrt S_ABC) := 
sorry

end triangle_areas_inequality_l610_610201


namespace geometry_problem_l610_610295

variables {Point : Type*} [euclidean_space Point]
variables (A B C D E F M N : Point)

def square (a b c d : Point) : Prop :=
  euclidean_geometry.parallel a c ∧
  euclidean_geometry.parallel b d ∧
  euclidean_geometry.orthogonal a b ∧
  euclidean_geometry.cong (a - b) (b - c)

def midpoint (p1 p2 : Point) (mid : Point) : Prop :=
  mid = (p1 + p2) / 2

theorem geometry_problem
  (h1 : square A B C D)
  (h2 : square A D E F)
  (hM : midpoint B D M)
  (hN : midpoint A E N) :
  euclidean_geometry.orthogonal (M - N) (A - D) ∧
  euclidean_geometry.parallel (M - N) (plane A B F) :=
by
  sorry

end geometry_problem_l610_610295


namespace greatest_prime_factor_175_l610_610448

theorem greatest_prime_factor_175 : ∃ (p : ℕ), prime p ∧ ∃ (k : ℕ), 175 = k * p ∧ ∀ (q : ℕ), prime q → ∃ (j : ℕ), 175 = j * q → q ≤ p :=
by
  sorry

end greatest_prime_factor_175_l610_610448


namespace cos_x_plus_2y_equals_one_l610_610619

theorem cos_x_plus_2y_equals_one (x y a : ℝ) 
  (hx : x ∈ set.Icc (-(Real.pi / 4)) (Real.pi / 4)) 
  (hy : y ∈ set.Icc (-(Real.pi / 4)) (Real.pi / 4))
  (h1 : x^3 + Real.sin x - 2 * a = 0)
  (h2 : 4 * y^3 + (1/2) * Real.sin (2 * y) + a = 0) : 
  Real.cos (x + 2 * y) = 1 := 
begin
  sorry
end

end cos_x_plus_2y_equals_one_l610_610619


namespace win_sector_area_l610_610065

-- Define the radius of the circle (spinner)
def radius : ℝ := 8

-- Define the probability of winning on one spin
def probability_winning : ℝ := 1 / 4

-- Define the area of the circle, calculated from the radius
def total_area : ℝ := Real.pi * radius^2

-- The area of the WIN sector to be proven
theorem win_sector_area : (probability_winning * total_area) = 16 * Real.pi := by
  sorry

end win_sector_area_l610_610065


namespace optimal_redistribution_minimum_transfer_l610_610372

variables {n : ℕ} {N : ℕ } {a : fin n.succ → ℤ}

noncomputable def b (i : ℕ) : ℤ := ∑ j in finset.range i, (a ⟨j, nat.lt_succ_of_lt j.2⟩ - N)

theorem optimal_redistribution_minimum_transfer 
  (h1 : n ≥ 3) 
  (h2 : (∑ i in finset.range (n+1), a ⟨i, nat.lt_succ_self n⟩) = n * N) : 
  ∃ (x1: ℤ), 
  ∀ x1', 
  (x1 = b (n / 2)) → 
  (∑ i in finset.range (n+1), abs (x1 - b i)) ≤ ∑ i in finset.range (n+1), abs (x1' - b i) :=
sorry

end optimal_redistribution_minimum_transfer_l610_610372


namespace quadratic_has_non_real_roots_l610_610145

theorem quadratic_has_non_real_roots (c : ℝ) (h : c > 16) :
    ∃ (a b : ℂ), (x^2 - 8 * x + c = 0) = (a * a = -1) ∧ (b * b = -1) :=
sorry

end quadratic_has_non_real_roots_l610_610145


namespace product_eq_zero_implies_either_zero_l610_610776

theorem product_eq_zero_implies_either_zero (a b : ℝ) (h : a * b = 0) : a = 0 ∨ b = 0 :=
by {
  assume hc : ¬(a = 0 ∨ b = 0),
  have hna : a ≠ 0 := (not_or_distrib.mp hc).left,
  have hnb : b ≠ 0 := (not_or_distrib.mp hc).right,
  sorry
}

end product_eq_zero_implies_either_zero_l610_610776


namespace train_speed_conversion_l610_610899

theorem train_speed_conversion (s_mps : ℝ) (h : s_mps = 30.002399999999998) : 
  s_mps * 3.6 = 108.01 :=
by
  sorry

end train_speed_conversion_l610_610899


namespace time_for_A_to_finish_job_alone_l610_610058

-- Given conditions
def work_rate_A (x : ℝ) : ℝ := 1 / x
def work_rate_B : ℝ := 1 / 20
def combined_work_rate (x : ℝ) : ℝ := work_rate_A x + work_rate_B
def work_done_together (x : ℝ) : ℝ := 3 * combined_work_rate x

-- Fraction of work left
def work_left_fraction : ℝ := 0.65
def work_done_fraction : ℝ := 1 - work_left_fraction

-- Theorem statement
theorem time_for_A_to_finish_job_alone (x : ℝ) (h : work_done_together x = work_done_fraction) : x = 15 := by
  sorry

end time_for_A_to_finish_job_alone_l610_610058


namespace snow_difference_l610_610124

theorem snow_difference (a b : ℕ) (h1 : a = 29) (h2 : b = 17) : a - b = 12 :=
by
  rw [h1, h2]
  norm_num

end snow_difference_l610_610124


namespace smallest_x_value_l610_610167

-- Definition of the given equation
def given_equation (x : ℝ) : Prop :=
  (15 * x^2 - 40 * x + 20) / (4 * x - 3) + 7 * x = 8 * x - 3

-- Statement that the smallest value of x is (25 - sqrt 141) / 22
theorem smallest_x_value :
  ∃ x : ℝ, given_equation x ∧ x = (25 - real.sqrt 141) / 22 :=
by sorry

end smallest_x_value_l610_610167


namespace Lisa_income_percentage_J_M_combined_l610_610765

variables (T M J L : ℝ)

-- Conditions as definitions
def Mary_income_eq_1p6_T (M T : ℝ) : Prop := M = 1.60 * T
def Tim_income_eq_0p5_J (T J : ℝ) : Prop := T = 0.50 * J
def Lisa_income_eq_1p3_M (L M : ℝ) : Prop := L = 1.30 * M
def Lisa_income_eq_0p75_J (L J : ℝ) : Prop := L = 0.75 * J

-- Theorem statement
theorem Lisa_income_percentage_J_M_combined (M T J L : ℝ)
  (h1 : Mary_income_eq_1p6_T M T)
  (h2 : Tim_income_eq_0p5_J T J)
  (h3 : Lisa_income_eq_1p3_M L M)
  (h4 : Lisa_income_eq_0p75_J L J) :
  (L / (M + J)) * 100 = 41.67 := 
sorry

end Lisa_income_percentage_J_M_combined_l610_610765


namespace clock_angle_at_930_l610_610123

theorem clock_angle_at_930 :
  let hour_hand_rotation_rate := 0.5 -- Rotation rate of the hour hand in degrees per minute
  let minute_hand_rotation_rate := 6 -- Rotation rate of the minute hand in degrees per minute
  let initial_angle := 270 -- Initial angle between hour and minute hands at 9:00 in degrees
  let hour_rotation_at_930 := 30 * hour_hand_rotation_rate -- Rotation of the hour hand from 9:00 to 9:30
  let minute_rotation_at_930 := 30 * minute_hand_rotation_rate -- Rotation of the minute hand from 9:00 to 9:30
  -- Calculate the angle at 9:30 by adding the hour hand rotation, subtracting the minute hand rotation, and adding the initial angle
  let angle_at_930 := initial_angle + hour_rotation_at_930 - minute_rotation_at_930
  angle_at_930 = 105 :=
begin
  sorry
end

end clock_angle_at_930_l610_610123


namespace num_handshakes_l610_610538

-- Definition of the conditions
def num_teams : Nat := 4
def women_per_team : Nat := 2
def total_women : Nat := num_teams * women_per_team
def handshakes_per_woman : Nat := total_women -1 - women_per_team

-- Statement of the problem to prove
theorem num_handshakes (h: total_women = 8) : (total_women * handshakes_per_woman) / 2 = 24 := 
by
  -- Proof goes here
  sorry

end num_handshakes_l610_610538


namespace count_integers_satisfying_log_condition_l610_610182

theorem count_integers_satisfying_log_condition :
  {x : ℕ | 35 < x ∧ x < 65 ∧ (log 4 (x - 35) * log 4 (65 - x) < 3)}.card = 20 :=
sorry

end count_integers_satisfying_log_condition_l610_610182


namespace slope_of_line_through_points_l610_610452

-- Define the points
def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (4, -6)

-- Definition of the slope function given two points
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.snd - p1.snd) / (p2.fst - p1.fst)

-- The theorem: the slope of the line passing through point1 and point2 is -3
theorem slope_of_line_through_points :
  slope point1 point2 = -3 :=
sorry

end slope_of_line_through_points_l610_610452


namespace find_ratio_AF_FB_l610_610720

-- Define the vector space over reals
variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Definitions of points A, B, C, D, F, P
variables (a b c d f p : V)

-- Given conditions as hypotheses
variables (h1 : (p = 2 / 5 • a + 3 / 5 • d))
variables (h2 : (p = 5 / 7 • f + 2 / 7 • c))
variables (hd : (d = 1 / 3 • b + 2 / 3 • c))
variables (hf : (f = 1 / 4 • a + 3 / 4 • b))

-- Theorem statement
theorem find_ratio_AF_FB : (41 : ℝ) / 15 = (41 : ℝ) / 15 := 
by sorry

end find_ratio_AF_FB_l610_610720


namespace find_a_b_tangent_line_at_zero_l610_610995

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3
noncomputable def f' (a b x : ℝ) : ℝ := 2 * a * x + b

theorem find_a_b :
  ∃ a b : ℝ, (a ≠ 0) ∧ (∀ x, f' a b x = 2 * x - 8) := 
sorry

noncomputable def g (x : ℝ) : ℝ := Real.exp x * Real.sin x + x^2 - 8 * x + 3
noncomputable def g' (x : ℝ) : ℝ := Real.exp x * Real.sin x + Real.exp x * Real.cos x + 2 * x - 8

theorem tangent_line_at_zero :
  g' 0 = -7 ∧ g 0 = 3 ∧ (∀ y, y = 3 + (-7) * x) := 
sorry

end find_a_b_tangent_line_at_zero_l610_610995


namespace value_of_expression_l610_610850

theorem value_of_expression (x : ℤ) (h : x = 4) : (3 * x + 7) ^ 2 = 361 := by
  rw [h] -- Replace x with 4
  norm_num -- Simplify the expression
  done

end value_of_expression_l610_610850


namespace smallest_k_for_64_pow_k_gt_4_pow_19_l610_610051

theorem smallest_k_for_64_pow_k_gt_4_pow_19 : ∃ k : ℤ, (64 ^ k > 4 ^ 19) ∧ (∀ m : ℤ, 64 ^ m > 4 ^ 19 → m ≥ k) :=
begin
  use 7,
  split,
  {
    -- Proof that 64 ^ 7 > 4 ^ 19 goes here
    sorry,
  },
  {
    -- Proof that 7 is the smallest integer goes here
    sorry,
  }
end

end smallest_k_for_64_pow_k_gt_4_pow_19_l610_610051


namespace multinomial_coefficient_l610_610585

theorem multinomial_coefficient (n : ℕ) (m : ℕ) (k : Fin m → ℕ) (H : ∑ i, k i = n) :
  (a : Fin m → ℕ) → (∃! (a : Fin m → ℕ), (∏ i, a i ^ (k i)) = (a i ^ n)) → 
  (∏ i, (nat.choose n (k i))) = nat.factorial n / (∏ i, nat.factorial (k i)) :=
by
  sorry

end multinomial_coefficient_l610_610585


namespace difference_divisible_by_three_l610_610844

theorem difference_divisible_by_three :
  let digits := [7, 3, 1, 4, 9],
      largest := 9 * 10000 + 7 * 1000 + 4 * 100 + 3 * 10 + 1,
      least := 1 * 10000 + 3 * 1000 + 4 * 100 + 7 * 10 + 9,
      difference := largest - least
  in difference % 3 = 0 :=
by {
  let digits := [7, 3, 1, 4, 9],
  let largest := 9 * 10000 + 7 * 1000 + 4 * 100 + 3 * 10 + 1,
  let least := 1 * 10000 + 3 * 1000 + 4 * 100 + 7 * 10 + 9,
  let difference := largest - least,
  calc difference % 3 = 83952 % 3 : by rfl
  ... = 0 : by norm_num,
}

end difference_divisible_by_three_l610_610844


namespace compound_interest_rate_l610_610820

theorem compound_interest_rate
(SI : ℝ) (CI : ℝ) (P1 : ℝ) (r : ℝ) (t1 t2 : ℕ) (P2 R : ℝ)
(h1 : SI = (P1 * r * t1) / 100)
(h2 : SI = CI / 2)
(h3 : CI = P2 * (1 + R / 100) ^ t2 - P2)
(h4 : P1 = 3500)
(h5 : r = 6)
(h6 : t1 = 2)
(h7 : P2 = 4000)
(h8 : t2 = 2) : R = 10 := by
  sorry

end compound_interest_rate_l610_610820


namespace cos_theta_value_l610_610215

open Real

theorem cos_theta_value (theta: ℝ) 
  (h1: θ ∈ Ioo π 2π) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (cos θ, sin θ)) 
  (h2 : ∃ k : ℝ, k ≠ 0 ∧ a = k • b) : 
  cos θ = -sqrt 5 / 5 :=
sorry

end cos_theta_value_l610_610215


namespace greatest_four_digit_multiple_of_17_l610_610005

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_multiple_of (n d : ℕ) : Prop :=
  ∃ k : ℕ, n = k * d

theorem greatest_four_digit_multiple_of_17 : ∃ n, is_four_digit n ∧ is_multiple_of n 17 ∧
  ∀ m, is_four_digit m → is_multiple_of m 17 → m ≤ n :=
  by
  existsi 9996
  sorry

end greatest_four_digit_multiple_of_17_l610_610005


namespace proof_problem_l610_610213

def necessary_but_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

theorem proof_problem (x : ℝ) :
  necessary_but_not_sufficient ((x+3)*(x-1) = 0) (x-1 = 0) :=
by
  sorry

end proof_problem_l610_610213


namespace greatest_four_digit_multiple_of_17_l610_610017

theorem greatest_four_digit_multiple_of_17 : ∃ n, (1000 ≤ n) ∧ (n ≤ 9999) ∧ (17 ∣ n) ∧ ∀ m, (1000 ≤ m) ∧ (m ≤ 9999) ∧ (17 ∣ m) → m ≤ n :=
begin
  sorry
end

end greatest_four_digit_multiple_of_17_l610_610017


namespace volume_region_l610_610932

noncomputable def f (x y z w : ℝ) : ℝ :=
  |x + y + z + w| + |x + y + z - w| + |x + y - z + w| + |x - y + z + w| + |-x + y + z + w|

theorem volume_region (S : Set (ℝ × ℝ × ℝ × ℝ)) :
  S = {p | let (x, y, z, w) := p in f x y z w ≤ 6} →
  let vol := MeasureTheory.volume.measure_univ.to_real in
  vol = (2 : ℝ) / 3 :=
by
  sorry

end volume_region_l610_610932


namespace greatest_four_digit_multiple_of_17_l610_610010

theorem greatest_four_digit_multiple_of_17 :
  ∃ n, (n % 17 = 0) ∧ (1000 ≤ n ∧ n ≤ 9999) ∧ ∀ m, (m % 17 = 0) ∧ (1000 ≤ m ∧ m ≤ 9999) → n ≥ m :=
begin
  use 9996,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm2a hm2b,
    exact nat.mul_le_mul_right _ (nat.div_le_of_le_mul (nat.le_sub_one_of_lt (lt_of_le_of_lt (nat.mul_le_mul_right _ (nat.le_of_dvd hm1)) (by norm_num)))),
  },
  sorry
end

end greatest_four_digit_multiple_of_17_l610_610010


namespace probability_area_equals_sum_is_zero_l610_610089

noncomputable def dice_roll_sum_determines_diameter :=
  ∀ (d1 d2 : ℕ), (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) →
  let S := d1 + d2 in
  ∀ (π : ℝ), (π > 3 ∧ π < 4) →           -- we approximate π for simplicity
  let A := π * (S/2)^2 in
  A ≠ S

theorem probability_area_equals_sum_is_zero :
  dice_roll_sum_determines_diameter :=
begin
  sorry
end

end probability_area_equals_sum_is_zero_l610_610089


namespace total_right_handed_players_is_60_l610_610770

def total_players : ℕ := 70
def throwers : ℕ := 40
def non_throwers : ℕ := total_players - throwers
def left_handed_non_throwers : ℕ := non_throwers / 3
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers
def right_handed_throwers : ℕ := throwers
def total_right_handed_players : ℕ := right_handed_throwers + right_handed_non_throwers

theorem total_right_handed_players_is_60 : total_right_handed_players = 60 := by
  sorry

end total_right_handed_players_is_60_l610_610770


namespace exists_consecutive_with_degree_condition_l610_610566

def degree (n : ℕ) : ℕ :=
  (factorization n).values.sum

theorem exists_consecutive_with_degree_condition :
  ∃ (xs : List ℕ), xs.length = 2018 ∧ (∃ t : ℕ, t < 11 ∧ (xs.filter (λ x => degree x < 11)).length = 1000) :=
sorry

end exists_consecutive_with_degree_condition_l610_610566


namespace find_x_for_g_equal_20_l610_610797

theorem find_x_for_g_equal_20 (g f : ℝ → ℝ) (h₁ : ∀ x, g x = 4 * (f⁻¹ x))
    (h₂ : ∀ x, f x = 30 / (x + 5)) :
    ∃ x, g x = 20 ∧ x = 3 := by
  sorry

end find_x_for_g_equal_20_l610_610797


namespace a_eq_b_l610_610256

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

-- Conditions
def a := log_base 16 625
def b := log_base 4 25

-- Theorem to prove that a equals b
theorem a_eq_b : a = b := by sorry

end a_eq_b_l610_610256


namespace compare_abc_l610_610195

noncomputable def a : ℝ := 1 / (1 + Real.exp 2)
noncomputable def b : ℝ := 1 / Real.exp 1
noncomputable def c : ℝ := Real.log ((1 + Real.exp 2) / (Real.exp 2))

theorem compare_abc : b > c ∧ c > a := by
  sorry

end compare_abc_l610_610195


namespace hypotenuse_of_right_triangle_l610_610841

theorem hypotenuse_of_right_triangle (a b : ℝ) (h₁ : (1 / 3) * π * a * b^2 = 675 * π) (h₂ : (1 / 3) * π * b * a^2 = 2430 * π) :
  ∃ c : ℝ, c = Real.sqrt (a^2 + b^2) ∧ c = 30.79 :=
by {
  have eq1 : a * b^2 = 2025, by {
    sorry
  },
  have eq2 : b * a^2 = 7290, by {
    sorry
  },
  have quotient_eq : a / b = 3.6, by {
    sorry
  },
  have sub_a : a = 3.6 * b, by {
    sorry
  },
  have sub_b : b = 8.24, by {
    sorry
  },
  let c : ℝ := Real.sqrt (a^2 + b^2),
  use c,
  split,
  { exact rfl },
  { sorry }
}

end hypotenuse_of_right_triangle_l610_610841


namespace area_of_region_B_l610_610053

theorem area_of_region_B : 
  let z := ℂ in
  let condition1 := ∀ z, -1 ≤ (z / 30).re ∧ (z / 30).re ≤ 1 ∧ -1 ≤ (z / 30).im ∧ (z / 30).im ≤ 1 in
  let condition2 := ∀ z, -1 ≤ (30 / conj z).re ∧ (30 / conj z).re ≤ 1 ∧ -1 ≤ (30 / conj z).im ∧ (30 / conj z).im ≤ 1 in
  (condition1 z ∧ condition2 z) → (area_of B = 3600 - 337.5 * Real.pi) :=
by
  sorry

end area_of_region_B_l610_610053


namespace average_payment_is_correct_l610_610876

/-
Given conditions:
1. The debt is paid in 52 installments from January 1 to December 31.
2. Each of the first 25 payments is $500.
3. For the remaining payments, the increase amount for each payment follows the pattern:
   - The 26th payment is $100 more than the 1st payment,
   - The 27th payment is $200 more than the 2nd payment,
   - The 28th payment is $300 more than the 3rd payment, and so on.
-/
def first25_payments : Array Nat := Array.mkArray 25 500

def remaining27_payments (n : Nat) : Nat :=
  if n < 26 then 0 else 500 + (n - 25) * 100

def total_amount_paid : Nat :=
  (Array.foldr (· + ·) 0 first25_payments) +
  (Array.foldr (· + ·) 0 (Array.map remaining27_payments (Array.mkArray 27 (λ i => i + 26))))

def average_payment : Float :=
  total_amount_paid.toFloat / 52

theorem average_payment_is_correct : average_payment = 1226.92 :=
by
  sorry

end average_payment_is_correct_l610_610876


namespace fraction_to_terminating_decimal_l610_610922

theorem fraction_to_terminating_decimal :
  (45 : ℚ) / 64 = (703125 : ℚ) / 1000000 := by
  sorry

end fraction_to_terminating_decimal_l610_610922


namespace sequence_is_k_plus_n_l610_610583

theorem sequence_is_k_plus_n (a : ℕ → ℕ) (k : ℕ) (h : ∀ n : ℕ, a (n + 2) * (a (n + 1) - 1) = a n * (a (n + 1) + 1))
  (pos: ∀ n: ℕ, a n > 0) : ∀ n: ℕ, a n = k + n := 
sorry

end sequence_is_k_plus_n_l610_610583


namespace Annika_hiking_rate_is_correct_l610_610534

def AnnikaHikingRate
  (distance_partial_east distance_total_east : ℕ)
  (time_back_to_start : ℕ)
  (equality_rate : Nat) : Prop :=
  distance_partial_east = 2750 / 1000 ∧
  distance_total_east = 3500 / 1000 ∧
  time_back_to_start = 51 ∧
  equality_rate = 34

theorem Annika_hiking_rate_is_correct :
  ∃ R : ℕ, ∀ d1 d2 t,
  AnnikaHikingRate d1 d2 t R → R = 34 :=
by
  sorry

end Annika_hiking_rate_is_correct_l610_610534


namespace find_m_l610_610238

-- Define vectors as tuples
def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, -1)
def c (m : ℝ) : ℝ × ℝ := (4, m)

-- Define vector subtraction
def sub_vect (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

-- Define dot product
def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove the condition that (a - b) ⊥ c implies m = 4
theorem find_m (m : ℝ) (h : dot_prod (sub_vect a (b m)) (c m) = 0) : m = 4 :=
by
  sorry

end find_m_l610_610238


namespace f_inequality_l610_610337

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_dom : ∀ x : ℝ, x > 0 → f x > 0
axiom f_ineq : ∀ (x y : ℝ), (0 < x) → (0 < y) → f (x * y) ≤ f x * f y

theorem f_inequality (x : ℝ) (hx : x > 0) (n : ℕ) :
  f(x^n) ≤ (∏ i in finset.range(n-1), f(x^(i+1))) ^ (1 / (i + 1)) :=
sorry

end f_inequality_l610_610337


namespace number_of_subsets_l610_610662

def set_A := {a, b, c}

theorem number_of_subsets : ∃ n : ℕ, n = 8 ∧ (∀ S, S ⊆ set_A) := 
sorry

end number_of_subsets_l610_610662


namespace probability_of_drawing_green_ball_l610_610703

variable (total_balls green_balls : ℕ)
variable (total_balls_eq : total_balls = 10)
variable (green_balls_eq : green_balls = 4)

theorem probability_of_drawing_green_ball (h_total : total_balls = 10) (h_green : green_balls = 4) :
  (green_balls : ℚ) / total_balls = 2 / 5 := by
  sorry

end probability_of_drawing_green_ball_l610_610703


namespace twenty_odd_rows_fifteen_odd_cols_impossible_sixteen_by_sixteen_with_126_crosses_possible_l610_610521

-- Conditions and helper definitions
def is_odd (n: ℕ) := n % 2 = 1
def count_odd_rows (table: ℕ × ℕ → bool) (n: ℕ) := 
  (List.range n).countp (λ i, is_odd ((List.range n).countp (λ j, table (i, j))))
  
def count_odd_columns (table: ℕ × ℕ → bool) (n: ℕ) := 
  (List.range n).countp (λ j, is_odd ((List.range n).countp (λ i, table (i, j))))

-- a) Proof problem statement
theorem twenty_odd_rows_fifteen_odd_cols_impossible (n: ℕ): 
  ∀ (table: ℕ × ℕ → bool), count_odd_rows table n = 20 → count_odd_columns table n = 15 → False := 
begin
  intros table h_rows h_cols,
  sorry
end

-- b) Proof problem statement
theorem sixteen_by_sixteen_with_126_crosses_possible :
  ∃ (table: ℕ × ℕ → bool), count_odd_rows table 16 = 16 ∧ count_odd_columns table 16 = 16 ∧ 
  (List.range 16).sum (λ i, (List.range 16).countp (λ j, table (i, j))) = 126 :=
begin
  sorry
end

end twenty_odd_rows_fifteen_odd_cols_impossible_sixteen_by_sixteen_with_126_crosses_possible_l610_610521
